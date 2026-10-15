"""R40 capture core — default-OFF diagnostic-capture contract + bounded shard core.

Mission ``ZM-ZEGRID-R40-B2-A-SHARD-CORE`` (bounded task A of R40-B2). This module
implements the *frozen* R40 storage/identity/privacy/boundedness/stale-generation
mechanics from ``docs/refactor/R40_UNEXPLAINED_STRUCTURE_OBSERVABILITY_PLAN.md``
(sections §3 sample identity, §6 binding/splits, §8 capture-safety contract) using
**synthetic / in-memory fixtures only**.

Hard invariants (enforced and tested):

* **import / default-OFF has zero filesystem effect** — no module-level I/O, no
  path resolution that creates anything, no config/GUI/default change;
* the generation writer refuses unless the capture is **explicitly enabled** with a
  strict ``bool``;
* a **versioned, little-endian, C-order, fixed-width** sparse record dtype with
  **no object dtype and no pickle** (M3 pre ``f32`` / post ``f64``, M6 ``int64``,
  coordinates ``int32``, bool validity, ``f64`` support/weights/ramps);
* the deterministic ``§3`` sample key and domain-tagged pair-set / candidate-set
  digests;
* an **atomic, bounded** generation writer (fresh generation dir, one shard per
  relation, strict-JSON index, per-shard count/bytes/SHA-256, every shard fsynced,
  **COMMIT written last**, directory fsync) that never overwrites a committed
  generation and quarantines/cleans a partial one;
* **fail-closed path safety** (traversal, symlinked components/targets, protected
  source/dataset roots, non-fresh destination) and a **non-mutating, prose-aware**
  privacy boundary (privacy failure happens **before** COMMIT);
* **strict finite JSON** (NaN/Inf refused, never coerced to zero);
* a fail-closed identity checker binding the frozen §6.2/§8.7 fields incl. the
  terminal accepted B1 v2 protocol/criteria/result/claim;
* a stale-invalidation / exact-resume primitive (checkpoint only between completed
  relation shards; no final COMMIT while partial);
* a manifest link builder for exactly one additive ``r40_diagnostic_capture`` block
  proving all pre-existing manifest fields canonical-identical ON/OFF (never a
  whole-manifest byte-identity claim);
* pure, validated sparse worker-payload structures + deterministic parent-join
  helpers that never retain full patch E7–E9 planes.

**Not wired into production.** ``_assemble_per_cell_feathered`` and any worker /
process-pool / backend / GUI / config seam are deliberately untouched by this module.
Nothing here authorizes R40-C/R40-D, a real H4 run, a correction, or B2 acceptance:
the shard core alone does **not** accept B2.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Frozen schema / contract identity
# ---------------------------------------------------------------------------

SCHEMA_NAME = "zemosaic.zegrid.r40.capture"
SCHEMA_VERSION = "r40-1"
CAPTURE_CONTRACT_VERSION = "r40-capture-1"
CAPTURE_STAGE = "pre_offset_pre_feather"
CORRECTION_APPLIED = False
PRODUCTION_WIRED = False
# The default-OFF capture core is STRICT here: it refuses a ``production_wired`` value
# other than this one. The later, separately gated wiring seam (which will forward the
# sparse payload into ``_assemble_per_cell_feathered``) must pass
# ``expected_production_wired=True`` explicitly to the identity checker/digest/writer;
# nothing in this task wires it, and default-OFF semantics stay strict False.
EXPECTED_PRODUCTION_WIRED = False
QUALIFICATION_EXACT = "EXACT"
MISSION = "ZM-ZEGRID-R40-B2-A-SHARD-CORE"

ORIENTATIONS = ("h", "v")
N_CHANNELS = 3
RECORD_ORDERING = ("orientation", "u", "v", "channel", "gy", "gx")

SAMPLE_KEY_TAG = "zemosaic.zegrid.r40.pair.v1"
PAIRSET_TAG = "zemosaic.zegrid.r40.pairset.v1"
CANDIDATE_SET_TAG = "zemosaic.zegrid.r40.candidateset.v1"
IDENTITY_TAG = "zemosaic.zegrid.r40.identity.v1"
MANIFEST_LINK_TAG = "zemosaic.zegrid.r40.manifest_link.v1"
MANIFEST_LINK_KEY = "r40_diagnostic_capture"

SOURCE_SET_DIGEST_ALGO = "sha256_sorted_unique_relative_logical_paths_v1"

GENERATION_DIR_PREFIX = "r40_capture_"
GENERATION_INDEX_BASENAME = "index.json"
GENERATION_COMMIT_BASENAME = "COMMIT"
GENERATION_SHARD_PREFIX = "shard-"
GENERATION_SHARD_SUFFIX = ".bin"
INCOMPLETE_MARKER_BASENAME = "INCOMPLETE"

DEFAULT_DISK_BUDGET_BYTES = 256 * 1024 * 1024
COMMIT_BYTES_ALLOWANCE = 4096

# Terminal accepted B1 v2 bindings — `ZM-ZEGRID-R40-B1V-ONE-SHOT` = ACCEPT.
# result sha256 / claim id quoted from the accepted one-shot validation result.
B1_PROTOCOL_SHA256 = "d689b2d78cebebad0222f503f804a8d6c9a2bf60316c88067b1851e1a447ffa7"
B1_CRITERIA_SHA256 = "67be63e4423f782646f71bc350be5ff5a33a2007b67f0999a76a841154da92b9"
B1_RESULT_SHA256 = "d597cd1d67b0cd91593e1da777ed2dde4ff0105864cecd63fbfe259d327ab474"
B1_CLAIM_ID = "c965187cd826ade68c556a59e1f9373d5a4dc500"
B1_ACCEPTED_DECISIONS = "18/18"

PAYLOAD_SCHEMA = "zemosaic.zegrid.r40.worker_payload"
PAYLOAD_VERSION = "r40-payload-1"
PAYLOAD_ROLES = ("u", "v")


# ---------------------------------------------------------------------------
# Errors (fail closed)
# ---------------------------------------------------------------------------


class R40CaptureError(RuntimeError):
    """Base class for every R40 capture-core refusal."""


class CaptureDisabledError(R40CaptureError):
    """The capture is default OFF and was not explicitly enabled."""


class SchemaError(R40CaptureError):
    """A record / header / payload violates the frozen schema."""


class NonFiniteError(R40CaptureError):
    """A non-finite (NaN/Inf) value was found and refused (never coerced)."""


class IdentityError(R40CaptureError):
    """A frozen identity binding is missing or mismatched."""


class BudgetError(R40CaptureError):
    """The hard disk budget would be exceeded; refusing to publish."""


class PathSafetyError(R40CaptureError):
    """A destination path is unsafe (traversal/symlink/protected/non-fresh)."""


class PrivacyError(R40CaptureError):
    """A durable value leaks an absolute private path; refusing (no redaction)."""


class StaleGenerationError(R40CaptureError):
    """A checkpoint no longer matches the frozen identity; stale-invalidated."""


class DuplicateKeyError(R40CaptureError):
    """Two records share the same sample key; deterministic order is undefined."""


class GenerationExistsError(R40CaptureError):
    """A committed generation already exists; never overwritten."""


# ---------------------------------------------------------------------------
# Strict JSON + hashing helpers
# ---------------------------------------------------------------------------


def dumps_strict(payload: Any) -> str:
    """Deterministic strict JSON (sorted keys, ``allow_nan=False``)."""
    return json.dumps(payload, sort_keys=True, allow_nan=False, indent=2, ensure_ascii=True)


def canonical_json(payload: Any) -> str:
    """Canonical compact JSON used for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    """Raw-byte SHA-256 (no decoding)."""
    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_payload(payload: Any, tag: str) -> str:
    h = hashlib.sha256()
    h.update((str(tag) + "\n").encode("utf-8"))
    h.update(canonical_json(payload).encode("utf-8"))
    return h.hexdigest()


def assert_finite_json(payload: Any, prefix: str = "$") -> None:
    """Recursively refuse NaN/Inf anywhere in a JSON-like structure."""
    if isinstance(payload, Mapping):
        for key in sorted(payload):
            assert_finite_json(payload[key], f"{prefix}.{key}")
    elif isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            assert_finite_json(item, f"{prefix}[{i}]")
    elif isinstance(payload, float) and not math.isfinite(payload):
        raise NonFiniteError(f"non-finite float at {prefix}; refusing (never coerced to 0)")


def assert_finite_array(name: str, array: Any) -> None:
    """Refuse a non-finite float array and any object-dtype array.

    Handles structured dtypes (checks every field), so it can be applied to the
    encoded R40 record array (its float fields are ``<f4``/``<f8``).
    """
    arr = np.asarray(array)
    if arr.dtype.hasobject:
        raise SchemaError(f"object dtype is forbidden in array {name!r}")
    if arr.dtype.names:
        for field_name in arr.dtype.names:
            assert_finite_array(f"{name}.{field_name}", arr[field_name])
        return
    if arr.dtype.kind == "f" and not bool(np.isfinite(arr).all()):
        raise NonFiniteError(f"non-finite value in array {name!r}; refusing (never coerced to 0)")


# ---------------------------------------------------------------------------
# Privacy: fail-closed, prose-aware, NON-mutating
# ---------------------------------------------------------------------------
#
# A genuine absolute/private path (whole value or embedded) refuses the artifact.
# Nothing is redacted in place: scientific prose that merely *looks* path-like is
# never classified as a leak (the embedded form requires >=2 token-anchored
# segments), and a real leak is never silently scrubbed.

_WIN_DRIVE_RE = re.compile(r"(?:^|[^A-Za-z0-9])[A-Za-z]:[\\/]")
_UNC_RE = re.compile(r"(?:^|[^\\])\\\\[A-Za-z0-9._$-]+[\\]")
_FILE_URI_RE = re.compile(r"file://", re.IGNORECASE)
_EMBEDDED_ABS_RE = re.compile(r"(?<![\w:/])/(?:[A-Za-z0-9._~+\-]+/)+[A-Za-z0-9._~+\-]+")


def path_leak_category(value: str) -> str | None:
    """Classify a genuine absolute/private path carried by ``value``."""
    if not isinstance(value, str) or not value:
        return None
    if _FILE_URI_RE.search(value):
        return "file_uri"
    if _WIN_DRIVE_RE.search(value):
        return "windows_drive_absolute"
    if value.startswith("\\\\") or _UNC_RE.search(value):
        return "unc"
    if value.startswith("/"):
        return "posix_absolute"
    if _EMBEDDED_ABS_RE.search(value):
        return "posix_absolute_embedded"
    return None


def scan_path_leaks(payload: Any, prefix: str = "$") -> list[dict]:
    found: list[dict] = []
    if isinstance(payload, Mapping):
        for key in sorted(payload):
            found.extend(scan_path_leaks(payload[key], f"{prefix}.{key}"))
    elif isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            found.extend(scan_path_leaks(item, f"{prefix}[{i}]"))
    elif isinstance(payload, str):
        cat = path_leak_category(payload)
        if cat is not None:
            found.append({"json_path": prefix, "category": cat})
    return found


def assert_no_path_leaks(payload: Any) -> None:
    """Fail closed: a genuine leak refuses the artifact; nothing is redacted."""
    leaks = scan_path_leaks(payload)
    if leaks:
        raise PrivacyError(
            "genuine absolute/private path leak; refusing to write (no redaction): "
            + ", ".join(sorted({f"{l['json_path']}[{l['category']}]" for l in leaks}))
        )


# ---------------------------------------------------------------------------
# Path safety (fail closed) + fresh destination
# ---------------------------------------------------------------------------

_WIN_DRIVE_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def is_drive_absolute(raw: str) -> bool:
    return bool(_WIN_DRIVE_ABS_RE.match(str(raw).strip()))


def has_traversal_component(raw: str) -> bool:
    return any(p == ".." for p in re.split(r"[\\/]+", str(raw)))


def _assert_no_symlink_components(path: Path) -> None:
    """Refuse when ``path`` or any existing ancestor is a symlink."""
    parts = Path(path).absolute().parts
    acc = Path(parts[0])
    for part in parts[1:]:
        acc = acc / part
        try:
            if acc.is_symlink():
                raise PathSafetyError(f"symlinked path component refused: {acc.name!r}")
        except OSError as exc:  # pragma: no cover - defensive
            raise PathSafetyError(f"cannot stat path component {acc.name!r}") from exc


def _assert_not_inside(path: Path, roots: Sequence[Path]) -> None:
    resolved = Path(path).expanduser().resolve()
    for root in roots:
        rr = Path(root).expanduser().resolve()
        if resolved == rr or rr in resolved.parents:
            raise PathSafetyError("destination is inside a protected read-only tree; refusing")


def resolve_fresh_output_root(raw: Any, *, protected_roots: Sequence[Path] = ()) -> Path:
    """Validate an explicitly provided output root; fail closed on anything unsafe."""
    if raw is None:
        raise PathSafetyError("no output root supplied; refusing (only an explicit fresh root is writable)")
    if isinstance(raw, Path):
        s = str(raw)
    elif isinstance(raw, str):
        s = raw
    else:
        raise PathSafetyError("output root must be a path-like value")
    if not s.strip():
        raise PathSafetyError("empty output root; refusing")
    if s.lower().startswith("file://"):
        raise PathSafetyError("file:// output roots are refused")
    if is_drive_absolute(s) or s.startswith("\\\\"):
        raise PathSafetyError("absolute drive/UNC output roots are refused")
    if has_traversal_component(s):
        raise PathSafetyError("parent-traversal output roots are refused")
    p = Path(s).expanduser()
    if p.is_symlink():
        raise PathSafetyError("symlinked output root is refused")
    _assert_no_symlink_components(p)
    _assert_not_inside(p, protected_roots)
    return p


def assert_generation_is_fresh(gen_dir: Path) -> None:
    """A generation directory must not pre-exist (never overwrite evidence)."""
    gen_dir = Path(gen_dir)
    if gen_dir.is_symlink():
        raise PathSafetyError("symlinked generation directory is refused")
    if gen_dir.exists():
        raise GenerationExistsError(
            f"generation directory already exists; refusing to overwrite: {gen_dir.name}"
        )


# ---------------------------------------------------------------------------
# Sample identity: keys + domain-tagged digests (§3 / §6.2)
# ---------------------------------------------------------------------------


def sample_key(relation_id: str, channel: int, gx: int, gy: int) -> str:
    """Frozen §3 sample key.

    ``sha256("zemosaic.zegrid.r40.pair.v1\\n" + relation_id + "\\n" + channel +
    "\\n" + gx + "\\n" + gy)``.
    """
    h = hashlib.sha256()
    h.update((SAMPLE_KEY_TAG + "\n").encode("utf-8"))
    h.update(f"{relation_id}\n{int(channel)}\n{int(gx)}\n{int(gy)}".encode("utf-8"))
    return h.hexdigest()


def pairset_digest(sample_keys: Iterable[str]) -> str:
    """Canonical sample-set digest: ``sha256("…pairset.v1\\n" + sorted(keys))``."""
    keys = sorted(str(k) for k in sample_keys)
    h = hashlib.sha256()
    h.update((PAIRSET_TAG + "\n").encode("utf-8"))
    h.update("\n".join(keys).encode("utf-8"))
    return h.hexdigest()


def candidate_set_digest(coords: Iterable[tuple[int, int]]) -> str:
    """Deterministic candidate-set digest over unique ``(gx, gy)`` pairs."""
    pairs = sorted({(int(gx), int(gy)) for gx, gy in coords})
    h = hashlib.sha256()
    h.update((CANDIDATE_SET_TAG + "\n").encode("utf-8"))
    h.update("\n".join(f"{gx},{gy}" for gx, gy in pairs).encode("utf-8"))
    return h.hexdigest()


def relation_id_for(u: str, v: str) -> str:
    return f"{u}|{v}"


# ---------------------------------------------------------------------------
# Frozen §1.3 integer/block inverse (normalized -> EXACT global integer pixel)
# ---------------------------------------------------------------------------

INV_TOL = 1e-9


def forward_coord(index, dim: int) -> float:
    """Frozen §1.3 forward coordinate ``X = 2(c+0.5)/W - 1`` (``dim`` = W or H)."""
    return 2.0 * (float(index) + 0.5) / float(dim) - 1.0


def inverse_index(coord: float, dim: int) -> int:
    """Frozen §1.3 normalized->integer inverse (round-half-up, ``INV_TOL``).

    NEVER a raw floor of a normalized float: the recovered index must sit within
    ``INV_TOL`` of an integer and inside ``[0, dim)``, else the recovery REFUSES.
    """
    p_float = ((float(coord) + 1.0) * float(dim) / 2.0) - 0.5
    if not math.isfinite(p_float):
        raise IdentityError("inverse_index: non-finite recovered coordinate")
    nearest = math.floor(p_float + 0.5)
    if abs(p_float - nearest) > INV_TOL:
        raise IdentityError(
            f"inverse_index: recovered index {p_float!r} is not within INV_TOL of an integer"
        )
    p = int(nearest)
    if not (0 <= p < int(dim)):
        raise IdentityError(f"inverse_index: recovered index {p} out of range [0,{dim})")
    return p


def assert_block_parity(gx: int, gy: int, block_px: int) -> None:
    """Frozen §1.3 block-parity held-out predicate (BOTH block ids ODD)."""
    if (int(gx) // int(block_px)) % 2 != 1 or (int(gy) // int(block_px)) % 2 != 1:
        raise IdentityError(
            f"held-out key ({int(gx)},{int(gy)}) does not satisfy the odd/odd "
            f"block-parity predicate at block_px={int(block_px)}"
        )


def heldout_keys_from_xy(xy, *, canvas_width: int, canvas_height: int,
                         block_px: int) -> tuple[tuple[int, int], ...]:
    """Recover the EXACT held-out global integer keys from normalized ``holdout_xy``.

    Applies the frozen §1.3 inverse (round-half-up, ``INV_TOL``) on ``X`` (dim = W)
    and ``Y`` (dim = H), verifies each key reproduces its stored normalized
    coordinates within ``INV_TOL`` and satisfies the odd/odd block-parity
    predicate, then returns the deterministically ordered unique ``(gx, gy)``
    tuple. Any ambiguity REFUSES (``IdentityError``): the selection is never
    approximated and never a raw floor of normalized floats.
    """
    rows = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    out: set[tuple[int, int]] = set()
    for x, y in rows:
        gx = inverse_index(float(x), int(canvas_width))
        gy = inverse_index(float(y), int(canvas_height))
        if (abs(forward_coord(gx, int(canvas_width)) - float(x)) > INV_TOL
                or abs(forward_coord(gy, int(canvas_height)) - float(y)) > INV_TOL):
            raise IdentityError(
                "held-out sample does not reproduce its stored X/Y within INV_TOL"
            )
        assert_block_parity(gx, gy, int(block_px))
        out.add((gx, gy))
    return tuple(sorted(out))


# ---------------------------------------------------------------------------
# Frozen record dtype (little-endian, fixed width, C order, no object)
# ---------------------------------------------------------------------------

R40_RECORD_FIELDS: tuple[tuple[str, str], ...] = (
    ("gy", "<i4"),
    ("gx", "<i4"),
    ("channel", "u1"),
    ("sci_u_pre", "<f4"),
    ("sci_v_pre", "<f4"),
    ("sci_u_post", "<f8"),
    ("sci_v_post", "<f8"),
    ("valid_u", "?"),
    ("valid_v", "?"),
    ("estimator_weight_sum_u", "<f8"),
    ("estimator_weight_sum_v", "<f8"),
    ("surviving_sample_count_u", "<i8"),
    ("surviving_sample_count_v", "<i8"),
    ("support_w1_u", "<f8"),
    ("support_w1_v", "<f8"),
    ("support_w2_u", "<f8"),
    ("support_w2_v", "<f8"),
    ("n_eff_support_u", "<f8"),
    ("n_eff_support_v", "<f8"),
    ("ramp_u", "<f8"),
    ("ramp_v", "<f8"),
    ("sample_key", "S64"),
)
R40_RECORD_NAMES: tuple[str, ...] = tuple(name for name, _ in R40_RECORD_FIELDS)
R40_RECORD_DTYPE = np.dtype(list(R40_RECORD_FIELDS))
R40_RECORD_ITEMSIZE = int(R40_RECORD_DTYPE.itemsize)

_FLOAT_FIELDS = frozenset(
    name for name, spec in R40_RECORD_FIELDS if spec == "<f8" or spec == "<f4"
)
_INT_FIELDS = frozenset({"gy", "gx", "surviving_sample_count_u", "surviving_sample_count_v"})
_BOOL_FIELDS = frozenset({"valid_u", "valid_v"})
# Frozen integer ranges (half-open): int32 coordinates, canonical int64 depths.
_INT_RANGES: Mapping[str, tuple[int, int]] = MappingProxyType({
    "gy": (-(2**31), 2**31),
    "gx": (-(2**31), 2**31),
    "surviving_sample_count_u": (-(2**63), 2**63),
    "surviving_sample_count_v": (-(2**63), 2**63),
})

# Per-cell plane fields carried by one worker payload (sparse candidates only).
PAYLOAD_FIELDS_BY_ROLE: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "u": (
        "sci_u_pre",
        "sci_u_post",
        "valid_u",
        "estimator_weight_sum_u",
        "surviving_sample_count_u",
        "support_w1_u",
        "support_w2_u",
        "n_eff_support_u",
        "ramp_u",
    ),
    "v": (
        "sci_v_pre",
        "sci_v_post",
        "valid_v",
        "estimator_weight_sum_v",
        "surviving_sample_count_v",
        "support_w1_v",
        "support_w2_v",
        "n_eff_support_v",
        "ramp_v",
    ),
})
# Union of both side field lists = the pair-record projection (excluding ids/key).
SAMPLE_PLANE_FIELDS: tuple[str, ...] = tuple(
    name for name in R40_RECORD_NAMES if name not in ("gy", "gx", "channel", "sample_key")
)


def record_dtype_descriptor() -> list[dict]:
    """Deterministic, JSON-safe description of the frozen record dtype."""
    return [
        {
            "name": name,
            "dtype": R40_RECORD_DTYPE[name].str,
            "bytes": int(R40_RECORD_DTYPE[name].itemsize),
        }
        for name in R40_RECORD_NAMES
    ]


def assert_record_dtype_frozen() -> dict:
    """Structural proof of the frozen dtype (little-endian, no object/flexible)."""
    descr = R40_RECORD_DTYPE.descr
    for name, spec in descr:
        if "|O" in spec or "O" in spec:
            raise SchemaError("object dtype is forbidden")
    if R40_RECORD_DTYPE.hasobject:
        raise SchemaError("record dtype must not contain object fields")
    if R40_RECORD_DTYPE.kind != "V":
        raise SchemaError("record dtype must be a fixed-width structured dtype")
    return {
        "itemsize": R40_RECORD_ITEMSIZE,
        "n_fields": len(R40_RECORD_NAMES),
        "byteorder": R40_RECORD_DTYPE.byteorder,
        "isnative": bool(R40_RECORD_DTYPE.isnative),
        "descr": record_dtype_descriptor(),
    }


# ---------------------------------------------------------------------------
# Relation headers (M1 / M9 / M10) + sparse record encoding
# ---------------------------------------------------------------------------

HEADER_REQUIRED_FIELDS = (
    "gauge_mode",
    "stack_tokens",
    "reference_frame_id",
    "frame_order_digest",
    "excluded",
    "measurement_px",
    "feather_px",
    "halo_px",
    "block_px",
)
_STACK_TOKEN_KEYS = (
    "normalization",
    "weighting",
    "rejection",
    "combine",
    "taper",
    "taper_px",
    "taper_floor",
    "equalize_rgb",
    "sigma_low",
    "sigma_high",
    "winsor",
)
_PX_HEADER_FIELDS = ("measurement_px", "feather_px", "halo_px", "block_px")


def _require(condition: bool, message: str, exc: type[R40CaptureError] = SchemaError) -> None:
    if not condition:
        raise exc(message)


def validate_relation_header(header: Mapping, *, u: str, v: str, orientation: str) -> dict:
    """Validate + canonicalise one relation header (M1/M9/M10); fail closed."""
    _require(isinstance(header, Mapping), "relation header must be a mapping")
    _require(orientation in ORIENTATIONS, f"orientation must be one of {ORIENTATIONS}")
    for name in (u, v):
        _require(
            isinstance(name, str) and bool(re.match(r"^[A-Za-z0-9_-]+$", name)),
            f"cell id {name!r} is not a safe identifier",
        )
    for key in HEADER_REQUIRED_FIELDS:
        _require(key in header, f"relation header missing required field {key!r}")
    _require(isinstance(header["gauge_mode"], str) and header["gauge_mode"], "gauge_mode must be a non-empty string")
    tokens = header["stack_tokens"]
    _require(isinstance(tokens, Mapping), "stack_tokens must be a mapping")
    for key in _STACK_TOKEN_KEYS:
        _require(key in tokens, f"stack_tokens missing {key!r}")
    for key in ("sigma_low", "sigma_high", "winsor", "taper_floor"):
        value = tokens.get(key)
        if value is not None and not isinstance(value, (bool, str)):
            fv = float(value)
            _require(math.isfinite(fv), f"stack token {key!r} must be finite", NonFiniteError)
    for field_name in ("reference_frame_id", "frame_order_digest", "excluded"):
        value = header[field_name]
        _require(isinstance(value, Mapping), f"{field_name} must be a mapping keyed by cell")
        _require(set(value) == {u, v}, f"{field_name} must carry exactly the two cell ids")
    for field_name in ("reference_frame_id", "frame_order_digest"):
        for cell in (u, v):
            _require(
                isinstance(header[field_name][cell], str) and header[field_name][cell],
                f"{field_name}[{cell}] must be a non-empty string",
            )
    for cell in (u, v):
        value = header["excluded"][cell]
        _require(isinstance(value, (list, tuple)), f"excluded[{cell}] must be a sequence")
        _require(all(isinstance(x, str) for x in value), f"excluded[{cell}] entries must be strings")
    for field_name in _PX_HEADER_FIELDS:
        value = header[field_name]
        _require(
            isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) > 0,
            f"{field_name} must be a positive int",
        )
    canonical = {
        "relation_id": relation_id_for(u, v),
        "u": u,
        "v": v,
        "orientation": orientation,
        "gauge_mode": header["gauge_mode"],
        "stack_tokens": {k: _canonical_scalar(header["stack_tokens"][k]) for k in _STACK_TOKEN_KEYS},
        "reference_frame_id": {u: header["reference_frame_id"][u], v: header["reference_frame_id"][v]},
        "frame_order_digest": {u: header["frame_order_digest"][u], v: header["frame_order_digest"][v]},
        "excluded": {u: list(header["excluded"][u]), v: list(header["excluded"][v])},
        "measurement_px": int(header["measurement_px"]),
        "feather_px": int(header["feather_px"]),
        "halo_px": int(header["halo_px"]),
        "block_px": int(header["block_px"]),
    }
    assert_finite_json(canonical, "$.header")
    return canonical


def _canonical_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        _require(math.isfinite(value), "non-finite float in header", NonFiniteError)
        return value
    if isinstance(value, bool) or isinstance(value, int) or isinstance(value, str):
        return value
    if value is None:
        return None
    raise SchemaError(f"unsupported scalar type in relation header: {type(value).__name__}")


def encode_records(records: Sequence[Mapping], *, relation_id: str) -> np.ndarray:
    """Validate + encode sparse records into the frozen structured array.

    Ordering is deterministic ``(channel, gy, gx)`` within a shard; duplicate
    sample keys are refused; non-finite values are refused (never coerced).
    """
    rows: list[dict] = []
    for record in records:
        _require(isinstance(record, Mapping), "each record must be a mapping")
        row: dict[str, Any] = {}
        for name in R40_RECORD_NAMES:
            if name == "sample_key":
                continue
            _require(name in record, f"record missing required field {name!r}")
            row[name] = record[name]
        channel = row["channel"]
        _require(
            isinstance(channel, (int, np.integer)) and not isinstance(channel, bool),
            "channel must be an int",
        )
        channel = int(channel)
        _require(0 <= channel < N_CHANNELS, f"channel {channel} out of range 0..{N_CHANNELS - 1}")
        for name in _INT_FIELDS:
            value = row[name]
            _require(
                isinstance(value, (int, np.integer)) and not isinstance(value, bool),
                f"{name} must be an int",
            )
            ivalue = int(value)
            low, high = _INT_RANGES[name]
            if not (low <= ivalue < high):
                raise SchemaError(
                    f"{name}={ivalue} is outside the frozen dtype range; refusing (no silent wrap/narrow)"
                )
            row[name] = ivalue
        for name in _BOOL_FIELDS:
            _require(type(row[name]) is bool, f"{name} must be a strict bool")
        for name in _FLOAT_FIELDS:
            value = row[name]
            _require(
                isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool),
                f"{name} must be numeric",
            )
            try:
                fv = float(value)
            except OverflowError as exc:
                raise NonFiniteError(
                    f"{name} cannot be converted to a finite float; refusing (never coerced to 0)"
                ) from exc
            _require(math.isfinite(fv), f"{name} must be finite", NonFiniteError)
            spec = R40_RECORD_DTYPE[name]
            with np.errstate(over="ignore", invalid="ignore"):
                cast = np.array([fv], dtype=spec)[0]
            if not bool(np.isfinite(cast)):
                raise NonFiniteError(
                    f"{name}={fv!r} cannot be represented finitely in the frozen {spec.str} "
                    "dtype; refusing (never coerced to 0)"
                )
            row[name] = float(cast)
        key = sample_key(relation_id, channel, row["gx"], row["gy"])
        supplied = record.get("sample_key")
        if supplied is not None:
            given = supplied.decode("ascii") if isinstance(supplied, (bytes, bytearray)) else str(supplied)
            _require(given == key, "supplied sample_key does not match the frozen §3 key")
        row["sample_key"] = key.encode("ascii")
        rows.append(row)

    rows.sort(key=lambda r: (r["channel"], r["gy"], r["gx"]))
    keys = [row["sample_key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise DuplicateKeyError("duplicate sample key inside one relation shard; ordering undefined")

    array = np.empty(len(rows), dtype=R40_RECORD_DTYPE)
    for i, row in enumerate(rows):
        for name in R40_RECORD_NAMES:
            array[name][i] = row[name]
    # Fail closed on the ENCODED result: a finite input that cannot be represented in
    # the frozen dtype must never be silently stored as inf/NaN.
    assert_finite_array("r40.records", array)
    return array


@dataclass(frozen=True)
class RelationShard:
    """One relation's sparse shard: frozen header (M9/M10) + validated records.

    ``candidate_coords`` is the deterministic block-parity candidate geometry
    (``_r40_candidate_coords``) the worker→parent transport was built from, when
    the writer supplies it. The production seam ALWAYS supplies it, so the frozen
    candidate-set digest is bound into the generation index together with the
    final pair-set digest. It is optional so a purely synthetic shard can be built
    without a transported geometry.
    """

    u: str
    v: str
    orientation: str
    header: Mapping
    records: tuple
    candidate_coords: tuple | None = None

    @property
    def relation_id(self) -> str:
        return relation_id_for(self.u, self.v)

    def n_candidates(self) -> int:
        """Distinct candidate coordinates (0 when no geometry was supplied)."""
        if not self.candidate_coords:
            return 0
        return len({(int(gx), int(gy)) for gx, gy in self.candidate_coords})

    def candidate_digest(self) -> str | None:
        """Frozen candidate-set digest (``None`` when no geometry was supplied)."""
        if not self.candidate_coords:
            return None
        return candidate_set_digest(self.candidate_coords)

    def order_key(self) -> tuple:
        return (self.orientation, self.u, self.v)

    def array(self) -> np.ndarray:
        return encode_records(self.records, relation_id=self.relation_id)

    def data(self) -> bytes:
        return self.array().tobytes(order="C")

    def shard_name(self) -> str:
        return f"{GENERATION_SHARD_PREFIX}{self.orientation}-{self.u}-{self.v}{GENERATION_SHARD_SUFFIX}"

    def header_payload(self) -> dict:
        return validate_relation_header(self.header, u=self.u, v=self.v, orientation=self.orientation)

    def sample_keys(self) -> list[str]:
        array = self.array()
        return [k.decode("ascii") for k in array["sample_key"].tolist()]


# ---------------------------------------------------------------------------
# Default-OFF config (strict bool)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class R40CaptureConfig:
    """Frozen capture request. Constructing it performs **no** filesystem I/O."""

    enabled: bool = False
    output_root: Any = None
    disk_budget_bytes: int = DEFAULT_DISK_BUDGET_BYTES
    protected_roots: tuple = field(default_factory=tuple)
    on_partial: str = "quarantine"

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise SchemaError("enabled must be a strict bool (no truthy coercion)")
        if not isinstance(self.disk_budget_bytes, (int, np.integer)) or isinstance(self.disk_budget_bytes, bool):
            raise SchemaError("disk_budget_bytes must be an int")
        if int(self.disk_budget_bytes) <= 0:
            raise BudgetError("disk_budget_bytes must be positive")
        if self.on_partial not in ("quarantine", "cleanup"):
            raise SchemaError("on_partial must be 'quarantine' or 'cleanup'")


def default_capture_config() -> R40CaptureConfig:
    """The default capture configuration: OFF, no output root, no I/O."""
    return R40CaptureConfig()


def assert_capture_enabled(config: R40CaptureConfig | None) -> None:
    """Refuse unless the capture is explicitly enabled with a strict ``bool``."""
    if config is None:
        raise CaptureDisabledError("R40 capture is default OFF and no config was supplied")
    if type(config.enabled) is not bool:
        raise SchemaError("enabled must be a strict bool (no truthy coercion)")
    if config.enabled is not True:
        raise CaptureDisabledError("R40 capture is default OFF; refusing to write")


# ---------------------------------------------------------------------------
# Fail-closed identity checker
# ---------------------------------------------------------------------------

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CAPTURE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

IDENTITY_REQUIRED_FIELDS = (
    "capture_contract_version",
    "schema",
    "schema_version",
    "capture_stage",
    "correction_applied",
    "production_wired",
    "capture_id",
    "source_set_digest_algo",
    "source_set_digest",
    "source_set_count",
    "n_frames",
    "layout",
    "canvas",
    "gauge_mode",
    "measurement_px",
    "feather_px",
    "halo_px",
    "block_px",
    "holdout_capped",
    "qualification",
    "n_dropped",
    "repo_head",
    "b1_protocol_sha256",
    "b1_criteria_sha256",
    "b1_result_sha256",
    "b1_claim_id",
)


def _positive_int(value: Any, name: str) -> int:
    _require(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) > 0,
        f"identity field {name!r} must be a positive int",
        IdentityError,
    )
    return int(value)


def check_capture_identity(
    identity: Mapping,
    *,
    expected: Mapping | None = None,
    expected_production_wired: bool = EXPECTED_PRODUCTION_WIRED,
) -> dict:
    """Validate the frozen §6.2/§8.7 identity; any missing/mismatched field refuses.

    ``expected_production_wired`` is strict by default (``False`` = the default-OFF
    core). The later, separately gated wiring seam must pass ``True`` explicitly; this
    task does **not** wire it, so the default-OFF contract is unweakened.
    """
    _require(isinstance(identity, Mapping), "identity must be a mapping", IdentityError)
    for name in IDENTITY_REQUIRED_FIELDS:
        _require(name in identity, f"identity missing required field {name!r}", IdentityError)

    if identity["capture_contract_version"] != CAPTURE_CONTRACT_VERSION:
        raise IdentityError("capture_contract_version mismatch")
    if identity["schema"] != SCHEMA_NAME:
        raise IdentityError("schema mismatch")
    if identity["schema_version"] != SCHEMA_VERSION:
        raise IdentityError("schema_version mismatch")
    if identity["capture_stage"] != CAPTURE_STAGE:
        raise IdentityError("capture_stage mismatch")
    if identity["correction_applied"] is not False:
        raise IdentityError("correction_applied must be exactly False")
    if type(identity["production_wired"]) is not bool:
        raise IdentityError("production_wired must be a strict bool")
    if bool(identity["production_wired"]) is not bool(expected_production_wired):
        raise IdentityError(
            f"production_wired={identity['production_wired']!r} does not match the expected "
            f"{bool(expected_production_wired)!r}; the default-OFF core is strict False and the "
            "later wired seam must pass expected_production_wired=True explicitly"
        )
    if not (isinstance(identity["capture_id"], str) and _CAPTURE_ID_RE.match(identity["capture_id"])):
        raise IdentityError("capture_id must be a non-empty safe identifier")
    if identity["source_set_digest_algo"] != SOURCE_SET_DIGEST_ALGO:
        raise IdentityError("source_set_digest_algo mismatch")
    if not (isinstance(identity["source_set_digest"], str) and _HEX64.match(identity["source_set_digest"])):
        raise IdentityError("source_set_digest must be 64-hex")
    _positive_int(identity["source_set_count"], "source_set_count")
    _positive_int(identity["n_frames"], "n_frames")

    layout = identity["layout"]
    _require(isinstance(layout, Mapping) and {"nx", "ny"} <= set(layout), "layout must carry nx, ny", IdentityError)
    _positive_int(layout["nx"], "layout.nx")
    _positive_int(layout["ny"], "layout.ny")

    canvas = identity["canvas"]
    _require(
        isinstance(canvas, Mapping) and {"width", "height", "id"} <= set(canvas),
        "canvas must carry width, height, id",
        IdentityError,
    )
    _positive_int(canvas["width"], "canvas.width")
    _positive_int(canvas["height"], "canvas.height")
    _require(isinstance(canvas["id"], str) and canvas["id"], "canvas.id must be a non-empty string", IdentityError)

    _require(isinstance(identity["gauge_mode"], str) and identity["gauge_mode"], "gauge_mode must be a string", IdentityError)
    for name in ("measurement_px", "feather_px", "halo_px", "block_px"):
        _positive_int(identity[name], name)
    if identity["holdout_capped"] is not False:
        raise IdentityError("holdout_capped must be exactly False")
    if identity["qualification"] != QUALIFICATION_EXACT:
        raise IdentityError(f"qualification must be {QUALIFICATION_EXACT!r}")
    if not (isinstance(identity["n_dropped"], (int, np.integer)) and not isinstance(identity["n_dropped"], bool)):
        raise IdentityError("n_dropped must be an int")
    if int(identity["n_dropped"]) != 0:
        raise IdentityError("n_dropped must be exactly 0")
    if not (isinstance(identity["repo_head"], str) and _HEX40.match(identity["repo_head"])):
        raise IdentityError("repo_head must be a 40-hex git sha")

    _require(isinstance(identity["b1_claim_id"], str) and identity["b1_claim_id"], "b1_claim_id must be a string", IdentityError)

    exp = dict(expected or {})
    for key, value in (
        ("b1_protocol_sha256", B1_PROTOCOL_SHA256),
        ("b1_criteria_sha256", B1_CRITERIA_SHA256),
        ("b1_result_sha256", B1_RESULT_SHA256),
        ("b1_claim_id", B1_CLAIM_ID),
    ):
        if exp.get(key) != value:
            exp[key] = value
    for key in ("b1_protocol_sha256", "b1_criteria_sha256", "b1_result_sha256"):
        _require(isinstance(identity[key], str) and _HEX64.match(identity[key]), f"{key} must be 64-hex", IdentityError)
    for key, want in exp.items():
        if key in ("b1_claim_id", "b1_protocol_sha256", "b1_criteria_sha256", "b1_result_sha256"):
            if identity[key] != want:
                raise IdentityError(f"identity field {key!r} does not match the accepted B1 v2 binding")
        elif key in identity and identity[key] != want:
            raise IdentityError(f"identity field {key!r} does not match the expected value")

    normalized = {name: identity[name] for name in IDENTITY_REQUIRED_FIELDS}
    normalized["layout"] = {"nx": int(layout["nx"]), "ny": int(layout["ny"])}
    normalized["canvas"] = {
        "width": int(canvas["width"]),
        "height": int(canvas["height"]),
        "id": canvas["id"],
    }
    normalized["n_frames"] = int(identity["n_frames"])
    normalized["n_dropped"] = int(identity["n_dropped"])
    normalized["source_set_count"] = int(identity["source_set_count"])
    for name in ("measurement_px", "feather_px", "halo_px", "block_px"):
        normalized[name] = int(identity[name])
    normalized["holdout_capped"] = False
    normalized["correction_applied"] = False
    normalized["production_wired"] = bool(identity["production_wired"])
    assert_finite_json(normalized, "$.identity")
    return normalized


def identity_digest(identity: Mapping, *, expected_production_wired: bool = EXPECTED_PRODUCTION_WIRED) -> str:
    """Domain-tagged digest of a validated identity mapping."""
    return sha256_payload(
        check_capture_identity(identity, expected_production_wired=expected_production_wired),
        IDENTITY_TAG,
    )


def make_capture_identity(
    *,
    capture_id: str,
    source_set_digest: str,
    n_frames: int,
    layout: Mapping,
    canvas: Mapping,
    gauge_mode: str,
    measurement_px: int = 16,
    feather_px: int = 8,
    halo_px: int = 8,
    block_px: int = 4,
    repo_head: str,
    source_set_count: int | None = None,
    n_dropped: int = 0,
    holdout_capped: bool = False,
    qualification: str = QUALIFICATION_EXACT,
    b1_protocol_sha256: str = B1_PROTOCOL_SHA256,
    b1_criteria_sha256: str = B1_CRITERIA_SHA256,
    b1_result_sha256: str = B1_RESULT_SHA256,
    b1_claim_id: str = B1_CLAIM_ID,
    correction_applied: bool = CORRECTION_APPLIED,
    production_wired: bool = PRODUCTION_WIRED,
) -> dict:
    """Build a canonical identity mapping (values are validated by the checker)."""
    return {
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "capture_stage": CAPTURE_STAGE,
        "correction_applied": correction_applied,
        "production_wired": production_wired,
        "capture_id": capture_id,
        "source_set_digest_algo": SOURCE_SET_DIGEST_ALGO,
        "source_set_digest": source_set_digest,
        "source_set_count": int(source_set_count if source_set_count is not None else n_frames),
        "n_frames": int(n_frames),
        "layout": dict(layout),
        "canvas": dict(canvas),
        "gauge_mode": gauge_mode,
        "measurement_px": int(measurement_px),
        "feather_px": int(feather_px),
        "halo_px": int(halo_px),
        "block_px": int(block_px),
        "holdout_capped": holdout_capped,
        "qualification": qualification,
        "n_dropped": int(n_dropped),
        "repo_head": repo_head,
        "b1_protocol_sha256": b1_protocol_sha256,
        "b1_criteria_sha256": b1_criteria_sha256,
        "b1_result_sha256": b1_result_sha256,
        "b1_claim_id": b1_claim_id,
    }


# ---------------------------------------------------------------------------
# Disk budget (conservative, checked before and during writes)
# ---------------------------------------------------------------------------


def estimate_generation_bytes(shards: Sequence[RelationShard]) -> dict:
    """Deterministic byte plan for a generation (exact for the frozen writer)."""
    per_shard = []
    total_records = 0
    shard_bytes = 0
    for shard in sorted(shards, key=lambda s: s.order_key()):
        count = len(shard.records)
        nbytes = count * R40_RECORD_ITEMSIZE
        per_shard.append({
            "relation_id": shard.relation_id,
            "orientation": shard.orientation,
            "count": int(count),
            "bytes": int(nbytes),
        })
        shard_bytes += nbytes
        total_records += count
    index_bytes = len(_index_bytes_for_estimate(shards))
    commit_bytes = COMMIT_BYTES_ALLOWANCE
    return {
        "per_shard": per_shard,
        "shard_bytes": int(shard_bytes),
        "index_bytes": int(index_bytes),
        "commit_bytes": int(commit_bytes),
        "total_bytes": int(shard_bytes + index_bytes + commit_bytes),
        "n_records": int(total_records),
        "record_itemsize": R40_RECORD_ITEMSIZE,
    }


def check_disk_budget(*, planned_bytes: int, budget_bytes: int) -> None:
    """Refuse conservatively when the plan exceeds the hard budget."""
    if int(planned_bytes) > int(budget_bytes):
        raise BudgetError(
            f"planned generation bytes {int(planned_bytes)} exceed the hard disk budget "
            f"{int(budget_bytes)}; refusing before publish"
        )


def _index_bytes_for_estimate(shards: Sequence[RelationShard]) -> bytes:
    skeleton = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "record_itemsize": R40_RECORD_ITEMSIZE,
        "shards": [
            {
                "relation_id": s.relation_id,
                "u": s.u,
                "v": s.v,
                "orientation": s.orientation,
                "filename": s.shard_name(),
                "count": len(s.records),
                "bytes": len(s.records) * R40_RECORD_ITEMSIZE,
                "sha256": "0" * 64,
                "pairset_digest": "0" * 64,
                "candidate_set_digest": (
                    None if s.candidate_digest() is None else "0" * 64
                ),
                "n_candidates": int(s.n_candidates()),
                "header": s.header_payload(),
            }
            for s in sorted(shards, key=lambda s: s.order_key())
        ],
    }
    return dumps_strict(skeleton).encode("utf-8")


# ---------------------------------------------------------------------------
# Atomic generation writer (fresh, bounded, COMMIT last)
# ---------------------------------------------------------------------------


def _fsync_dir(path: Path) -> None:
    try:
        dfd = os.open(str(path), os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _quarantine_name(gen_dir: Path) -> Path:
    base = gen_dir.with_name(gen_dir.name + ".incomplete")
    if not base.exists():
        return base
    for i in range(1, 1000):
        candidate = gen_dir.with_name(f"{gen_dir.name}.incomplete-{i}")
        if not candidate.exists():
            return candidate
    raise R40CaptureError("cannot allocate a quarantine name for a partial generation")


def _handle_partial(gen_dir: Path, mode: str) -> None:
    """Clean or quarantine an uncommitted generation; never touch a committed one."""
    gen_dir = Path(gen_dir)
    if not gen_dir.exists():
        return
    if (gen_dir / GENERATION_COMMIT_BASENAME).exists():
        return  # committed: never silently discard evidence
    if mode == "cleanup":
        shutil.rmtree(gen_dir, ignore_errors=True)
        _fsync_dir(gen_dir.parent)
        return
    try:
        (gen_dir / INCOMPLETE_MARKER_BASENAME).write_text(
            "partial R40 generation; no COMMIT was published\n", encoding="utf-8"
        )
    except OSError:  # pragma: no cover - defensive
        pass
    target = _quarantine_name(gen_dir)
    os.replace(gen_dir, target)
    _fsync_dir(gen_dir.parent)


def _write_file_atomic(path: Path, data: bytes, *, _fail_at: str | None, stage: str) -> str:
    parent = path.parent
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if _fail_at == f"{stage}_fsync":
            raise R40CaptureError(f"injected failure at {stage} fsync")
        if tmp.is_symlink() or path.is_symlink():
            raise PathSafetyError("refusing to write through a symlink")
        if _fail_at == f"{stage}_replace":
            raise R40CaptureError(f"injected failure at {stage} replace")
        os.replace(tmp, path)
        _fsync_dir(parent)
    except BaseException:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:  # pragma: no cover - defensive
            pass
        raise
    return sha256_bytes(data)


def _build_generation_payloads(
    shards: Sequence[RelationShard],
    *,
    identity: Mapping,
    expected_production_wired: bool = EXPECTED_PRODUCTION_WIRED,
) -> tuple[list[dict], dict, dict]:
    """Serialise shard bytes, index payload and COMMIT payload (pure, no I/O)."""
    ident = check_capture_identity(identity, expected_production_wired=expected_production_wired)
    ordered = sorted(shards, key=lambda s: s.order_key())
    seen: set[str] = set()
    shard_blobs: list[dict] = []
    total_records = 0
    total_shard_bytes = 0
    for shard in ordered:
        _require(shard.orientation in ORIENTATIONS, "shard orientation must be h or v")
        if shard.relation_id in seen:
            raise SchemaError(f"duplicate relation shard: {shard.relation_id}")
        seen.add(shard.relation_id)
        data = shard.data()
        digest = sha256_bytes(data)
        header = shard.header_payload()
        keys = shard.sample_keys()
        # ZM-ZEGRID-R40-B2-I (Nono B2-G LOW a): the frozen ``candidate_set_digest``
        # must be RE-DERIVED from the declared geometry on the WRITE side, not merely
        # stored. A declared candidate geometry is refused when it is empty, carries
        # duplicate coordinates, fails to re-derive its own frozen digest, or does not
        # cover every coordinate actually recorded in the shard (the geometry the
        # sparse transport was built from must be a superset of the captured pairs).
        candidate_digest = shard.candidate_digest()
        n_candidates = int(shard.n_candidates())
        if shard.candidate_coords:
            coords = tuple((int(gx), int(gy)) for gx, gy in shard.candidate_coords)
            _require(
                len(coords) > 0,
                f"declared candidate geometry is empty for {shard.relation_id}",
            )
            _require(
                len(coords) == len(set(coords)),
                f"declared candidate geometry has duplicate coordinates for {shard.relation_id}",
            )
            _require(
                candidate_set_digest(coords) == candidate_digest,
                "candidate_set_digest does not re-derive from the declared geometry "
                f"for {shard.relation_id}",
            )
            _require(
                n_candidates == len(set(coords)),
                f"declared n_candidates disagrees with the candidate geometry for {shard.relation_id}",
            )
            geometry = set(coords)
            recorded = {(int(r["gx"]), int(r["gy"])) for r in shard.records}
            _require(
                recorded <= geometry,
                "recorded pair coordinate lies outside the declared candidate geometry "
                f"for {shard.relation_id}",
            )
        shard_blobs.append({
            "relation_id": shard.relation_id,
            "u": shard.u,
            "v": shard.v,
            "orientation": shard.orientation,
            "filename": shard.shard_name(),
            "count": int(len(keys)),
            "bytes": int(len(data)),
            "sha256": digest,
            "pairset_digest": pairset_digest(keys),
            "candidate_set_digest": candidate_digest,
            "n_candidates": n_candidates,
            "header": header,
            "_data": data,
        })
        total_records += len(keys)
        total_shard_bytes += len(data)

    ident_digest = sha256_payload(ident, IDENTITY_TAG)
    index_payload = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "mission": MISSION,
        "capture_id": ident["capture_id"],
        "capture_stage": CAPTURE_STAGE,
        "correction_applied": False,
        "production_wired": bool(ident["production_wired"]),
        "identity_digest": ident_digest,
        "identity": dict(ident),
        "record_itemsize": R40_RECORD_ITEMSIZE,
        "record_fields": list(R40_RECORD_NAMES),
        "record_dtype": record_dtype_descriptor(),
        "byte_order": "little",
        "order": "C",
        "ordering": list(RECORD_ORDERING),
        "n_shards": len(shard_blobs),
        "n_records": int(total_records),
        "shard_bytes": int(total_shard_bytes),
        "shards": [{k: v for k, v in blob.items() if k != "_data"} for blob in shard_blobs],
    }
    commit_payload = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "mission": MISSION,
        "capture_id": ident["capture_id"],
        "capture_stage": CAPTURE_STAGE,
        "identity_digest": ident_digest,
        "n_shards": len(shard_blobs),
        "n_records": int(total_records),
        "shard_bytes": int(total_shard_bytes),
    }
    return shard_blobs, index_payload, commit_payload


def write_capture_generation(
    output_root: Any,
    *,
    shards: Sequence[RelationShard],
    identity: Mapping,
    config: R40CaptureConfig | None = None,
    protected_roots: Sequence[Path] = (),
    on_partial: str | None = None,
    expected_production_wired: bool = EXPECTED_PRODUCTION_WIRED,
    _fail_at: str | None = None,
) -> dict:
    """Write one atomic, bounded R40 capture generation.

    The capture must be **explicitly enabled** (default OFF). The generation is
    written into a **fresh** generation directory under ``output_root``; the
    strict-JSON index and the ``COMMIT`` marker are written last, every shard is
    fsynced, and the directory is fsynced. Any failure before ``COMMIT`` leaves
    **no committed generation** (partial is cleaned or quarantined).

    Budget semantics (exact, fail-closed): the declared ``disk_budget_bytes`` must
    cover the shard plan **and** the exact final published total (shards + index +
    the COMMIT whose ``index_sha256`` is already bound). Both are refused **before**
    anything is published, so a published generation can never exceed its budget.
    """
    assert_capture_enabled(config)
    cfg = config
    assert cfg is not None
    on_partial = on_partial or cfg.on_partial
    roots = tuple(protected_roots) + tuple(cfg.protected_roots or ())
    root = resolve_fresh_output_root(output_root if output_root is not None else cfg.output_root, protected_roots=roots)
    ident = check_capture_identity(identity, expected_production_wired=expected_production_wired)
    _require(len(shards) > 0, "at least one relation shard is required")

    shard_blobs, index_payload, commit_payload = _build_generation_payloads(
        shards, identity=ident, expected_production_wired=expected_production_wired
    )
    # The COMMIT is bound to the exact index bytes it accompanies. Compute the FINAL
    # bytes (already including ``index_sha256``) BEFORE any publish/budget decision so
    # the published total can never exceed the declared budget.
    index_bytes = dumps_strict(index_payload).encode("utf-8")
    index_sha = sha256_bytes(index_bytes)
    commit_payload = dict(commit_payload)
    commit_payload["index_sha256"] = index_sha
    commit_bytes = dumps_strict(commit_payload).encode("utf-8")

    assert_finite_json(index_payload, "$.index")
    assert_no_path_leaks(index_payload)
    assert_finite_json(commit_payload, "$.commit")
    assert_no_path_leaks(commit_payload)

    planned_shard_bytes = sum(blob["bytes"] for blob in shard_blobs)
    exact_final_total = planned_shard_bytes + len(index_bytes) + len(commit_bytes)
    check_disk_budget(planned_bytes=planned_shard_bytes, budget_bytes=cfg.disk_budget_bytes)
    check_disk_budget(planned_bytes=exact_final_total, budget_bytes=cfg.disk_budget_bytes)

    gen_dir = root / f"{GENERATION_DIR_PREFIX}{ident['capture_id']}"
    assert_generation_is_fresh(gen_dir)

    created_dir = False
    try:
        root.mkdir(parents=True, exist_ok=True)
        gen_dir.mkdir(parents=False, exist_ok=False)
        created_dir = True
        if _fail_at == "shard_write":
            raise R40CaptureError("injected failure at shard write")

        written = 0
        for blob in shard_blobs:
            _write_file_atomic(
                gen_dir / blob["filename"], blob["_data"], _fail_at=_fail_at, stage="shard"
            )
            written += blob["bytes"]
            check_disk_budget(planned_bytes=written, budget_bytes=cfg.disk_budget_bytes)

        if _fail_at == "index_serialize":
            raise R40CaptureError("injected failure at index serialization")

        check_disk_budget(
            planned_bytes=written + len(index_bytes) + len(commit_bytes),
            budget_bytes=cfg.disk_budget_bytes,
        )
        _write_file_atomic(
            gen_dir / GENERATION_INDEX_BASENAME, index_bytes, _fail_at=_fail_at, stage="index"
        )

        if _fail_at == "commit_write":
            raise R40CaptureError("injected failure at COMMIT write")
        # COMMIT (already bound to the exact index bytes above) is written LAST.
        commit_sha = _write_file_atomic(
            gen_dir / GENERATION_COMMIT_BASENAME, commit_bytes, _fail_at=_fail_at, stage="commit"
        )
        _fsync_dir(gen_dir)
    except BaseException:
        if created_dir:
            _handle_partial(gen_dir, on_partial)
        raise

    published_total = int(planned_shard_bytes + len(index_bytes) + len(commit_bytes))
    assert published_total <= int(cfg.disk_budget_bytes), "internal budget invariant violated"
    return {
        "status": "COMMITTED",
        "generation_dir": gen_dir.name,
        "capture_id": ident["capture_id"],
        "identity_digest": index_payload["identity_digest"],
        "index_filename": GENERATION_INDEX_BASENAME,
        "index_sha256": index_sha,
        "commit_filename": GENERATION_COMMIT_BASENAME,
        "commit_sha256": commit_sha,
        "n_shards": len(shard_blobs),
        "n_records": index_payload["n_records"],
        "shard_bytes": index_payload["shard_bytes"],
        "total_bytes": published_total,
        "disk_budget_bytes": int(cfg.disk_budget_bytes),
    }


COMMIT_FIELDS = (
    "schema",
    "schema_version",
    "mission",
    "capture_id",
    "capture_stage",
    "identity_digest",
    "n_shards",
    "n_records",
    "shard_bytes",
    "index_sha256",
)


def verify_generation(gen_dir: Any, *, expected_production_wired: bool = EXPECTED_PRODUCTION_WIRED) -> dict:
    """Independently verify a published generation from on-disk bytes.

    Verifies the COMMIT contents (not just its presence) by rebuilding the expected
    COMMIT from the index, rejects unknown/extra COMMIT fields, re-checks the index's
    own identity digest and shard set, and flags any file present in the generation
    directory that the index does not list.
    """
    gen_dir = Path(gen_dir)
    commit_path = gen_dir / GENERATION_COMMIT_BASENAME
    index_path = gen_dir / GENERATION_INDEX_BASENAME
    if not commit_path.exists():
        raise R40CaptureError("no COMMIT published; generation is not complete")
    try:
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise R40CaptureError("COMMIT is not valid JSON") from exc
    if not index_path.exists():
        raise R40CaptureError("index missing; generation is not complete")
    index_bytes = index_path.read_bytes()
    index_sha = sha256_bytes(index_bytes)
    index = json.loads(index_bytes.decode("utf-8"))

    problems: list[str] = []

    # --- COMMIT field set + exact content binding -----------------------------
    if not isinstance(commit, Mapping):
        raise R40CaptureError("COMMIT must be a JSON object")
    if set(commit) != set(COMMIT_FIELDS):
        unknown = sorted(set(commit) - set(COMMIT_FIELDS))
        missing = sorted(set(COMMIT_FIELDS) - set(commit))
        problems.append(f"COMMIT field set mismatch (unknown={unknown}, missing={missing})")
    expected_commit = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "mission": MISSION,
        "capture_id": index.get("capture_id"),
        "capture_stage": CAPTURE_STAGE,
        "identity_digest": index.get("identity_digest"),
        "n_shards": index.get("n_shards"),
        "n_records": index.get("n_records"),
        "shard_bytes": index.get("shard_bytes"),
        "index_sha256": index_sha,
    }
    for field_name, want in expected_commit.items():
        if field_name in commit and commit[field_name] != want:
            problems.append(f"COMMIT {field_name} mismatch")

    # --- index schema + identity consistency ---------------------------------
    if index.get("schema") != SCHEMA_NAME or index.get("schema_version") != SCHEMA_VERSION:
        problems.append("index schema/schema_version mismatch")
    ident = index.get("identity")
    if not isinstance(ident, Mapping):
        problems.append("index identity missing")
    else:
        try:
            checked = check_capture_identity(ident, expected_production_wired=expected_production_wired)
        except R40CaptureError as exc:
            problems.append(f"index identity invalid: {type(exc).__name__}")
        else:
            if sha256_payload(checked, IDENTITY_TAG) != index.get("identity_digest"):
                problems.append("index identity_digest mismatch")
            if checked["capture_id"] != index.get("capture_id"):
                problems.append("index capture_id/identity mismatch")
            if checked["production_wired"] != index.get("production_wired"):
                problems.append("index production_wired/identity mismatch")

    # --- shard set consistency ------------------------------------------------
    entries = index.get("shards")
    if not isinstance(entries, list):
        problems.append("index shards missing/not a list")
        entries = []
    listed: dict[str, Mapping] = {}
    for entry in entries:
        filename = entry.get("filename") if isinstance(entry, Mapping) else None
        if not isinstance(filename, str):
            problems.append("index shard entry without filename")
            continue
        if filename in listed:
            problems.append(f"duplicate shard entry {filename}")
            continue
        listed[filename] = entry
    if index.get("n_shards") != len(entries):
        problems.append("index n_shards mismatch")

    checked_records = 0
    total_bytes = 0
    for filename, entry in listed.items():
        shard_path = gen_dir / filename
        if not shard_path.exists():
            problems.append(f"missing shard {filename}")
            continue
        data = shard_path.read_bytes()
        if sha256_bytes(data) != entry.get("sha256"):
            problems.append(f"shard hash mismatch {filename}")
        if len(data) != entry.get("bytes"):
            problems.append(f"shard byte mismatch {filename}")
        if entry.get("count") is not None and entry["count"] * R40_RECORD_ITEMSIZE != entry.get("bytes"):
            problems.append(f"shard count/itemsize mismatch {filename}")
        # --- frozen-rule binding (present on every PRODUCTION-seam generation) ----
        # When the writer declared the transported candidate geometry, the pair set
        # must be recoverable from the bytes AND every captured key must satisfy
        # the frozen §1.3 odd/odd block-parity predicate, with the pair-set digest
        # re-derived from those very bytes. A generation built without a declared
        # geometry (purely synthetic shards) is verified exactly as before.
        if entry.get("candidate_set_digest") is not None:
            array = np.frombuffer(data, dtype=R40_RECORD_DTYPE)
            header = entry.get("header")
            block_px = None
            if isinstance(header, Mapping) and "block_px" in header:
                try:
                    block_px = int(header["block_px"])
                except (TypeError, ValueError):
                    block_px = None
            # ZM-ZEGRID-R40-B2-I (Nono B2-G LOW b): a shard that DECLARES its frozen
            # candidate geometry must expose a usable ``block_px`` so the odd/odd
            # block-parity predicate can be re-checked from the bytes. Silently
            # skipping the re-check because ``block_px`` is absent would restore the
            # weaker verification level the declared geometry exists to prevent, so it
            # is REFUSED explicitly instead.
            if block_px is None:
                problems.append(
                    "declared candidate geometry without a usable 'block_px' header "
                    f"field; refusing to skip the frozen block-parity re-check {filename}"
                )
            elif block_px <= 0:
                problems.append(
                    f"declared candidate geometry with non-positive 'block_px' {filename}"
                )
                block_px = None
            keys: list[str] = []
            coord_seen: set[tuple[int, int]] = set()
            for row in array:
                gx, gy = int(row["gx"]), int(row["gy"])
                coord_seen.add((gx, gy))
                if block_px is not None:
                    try:
                        assert_block_parity(gx, gy, block_px)
                    except IdentityError:
                        problems.append(
                            f"captured key ({gx},{gy}) violates the frozen odd/odd "
                            f"block-parity predicate {filename}"
                        )
                keys.append(row["sample_key"].decode("ascii"))
            if len(keys) != len(set(keys)):
                problems.append(f"duplicate sample key in shard {filename}")
            if pairset_digest(keys) != entry.get("pairset_digest"):
                problems.append(f"pairset digest mismatch {filename}")
            if len(coord_seen) > int(entry.get("n_candidates") or 0):
                problems.append(f"pairset larger than the candidate geometry {filename}")
        checked_records += int(entry.get("count", 0))
        total_bytes += int(entry.get("bytes", 0))
    if checked_records != index.get("n_records"):
        problems.append("index n_records mismatch")
    if total_bytes != index.get("shard_bytes"):
        problems.append("index shard_bytes mismatch")

    # --- no unlisted/extra files in the committed generation dir ----------------
    allowed = {GENERATION_COMMIT_BASENAME, GENERATION_INDEX_BASENAME} | set(listed)
    for child in sorted(gen_dir.iterdir(), key=lambda p: p.name):
        if child.name not in allowed:
            problems.append(f"unlisted file in generation dir: {child.name}")

    return {
        "generation_dir": gen_dir.name,
        "index_sha256": index_sha,
        "n_shards": len(entries),
        "n_records": int(index.get("n_records") or 0),
        "checked_records": int(checked_records),
        "checked_bytes": int(total_bytes),
        "problems": problems,
        "ok": not problems,
    }


# ---------------------------------------------------------------------------
# Stale invalidation / exact resume primitive
# ---------------------------------------------------------------------------

STALE_INVALIDATED = "STALE_INVALIDATED"
RESUMABLE = "RESUMABLE"
COMPLETE = "COMPLETE"


def build_checkpoint(
    *,
    identity: Mapping,
    completed_shards: Sequence[Mapping],
    expected_production_wired: bool = EXPECTED_PRODUCTION_WIRED,
) -> dict:
    """Build a checkpoint covering only *completed* relation shards."""
    ident = check_capture_identity(identity, expected_production_wired=expected_production_wired)
    entries = []
    for shard in sorted(completed_shards, key=lambda s: str(s.get("relation_id"))):
        entries.append({
            "relation_id": str(shard["relation_id"]),
            "count": int(shard["count"]),
            "bytes": int(shard["bytes"]),
            "sha256": str(shard["sha256"]),
        })
    payload = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "identity_digest": sha256_payload(ident, IDENTITY_TAG),
        "completed_shards": entries,
        "n_completed": len(entries),
    }
    assert_finite_json(payload, "$.checkpoint")
    return payload


def check_resume(
    checkpoint: Mapping,
    *,
    identity: Mapping,
    completed_shards: Sequence[Mapping],
    expected_relation_ids: Sequence[str] | None = None,
    expected_production_wired: bool = EXPECTED_PRODUCTION_WIRED,
) -> dict:
    """Exact-resume check; any mismatch ⇒ ``STALE_INVALIDATED`` (no partial reuse)."""
    ident = check_capture_identity(identity, expected_production_wired=expected_production_wired)
    want_digest = sha256_payload(ident, IDENTITY_TAG)
    if checkpoint.get("identity_digest") != want_digest:
        raise StaleGenerationError(f"{STALE_INVALIDATED}: checkpoint identity digest mismatch")
    stored = {str(e["relation_id"]): e for e in checkpoint.get("completed_shards", [])}
    for shard in completed_shards:
        rid = str(shard["relation_id"])
        if rid not in stored:
            raise StaleGenerationError(f"{STALE_INVALIDATED}: shard {rid} absent from checkpoint")
        entry = stored[rid]
        for field_name in ("count", "bytes", "sha256"):
            if int_or_str(entry[field_name]) != int_or_str(shard[field_name]):
                raise StaleGenerationError(
                    f"{STALE_INVALIDATED}: shard {rid} {field_name} mismatch; refusing reuse"
                )
    if expected_relation_ids is not None:
        expected = {str(r) for r in expected_relation_ids}
        if set(stored) - expected:
            raise StaleGenerationError(f"{STALE_INVALIDATED}: checkpoint has unknown shards")
        missing = expected - set(stored)
        if missing:
            return {
                "state": RESUMABLE,
                "resume_from": sorted(set(stored)),
                "remaining": sorted(missing),
                "n_completed": len(stored),
            }
    return {"state": COMPLETE, "resume_from": sorted(set(stored)), "remaining": [], "n_completed": len(stored)}


def int_or_str(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def assert_checkpoint_complete(checkpoint: Mapping, *, expected_relation_ids: Sequence[str]) -> None:
    """No final COMMIT while partial: every expected relation shard must be complete."""
    stored = {str(e["relation_id"]) for e in checkpoint.get("completed_shards", [])}
    missing = {str(r) for r in expected_relation_ids} - stored
    if missing:
        raise StaleGenerationError(
            f"INCOMPLETE_PANEL: cannot finalise while shards are missing: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Manifest link (exactly one additive block; pre-existing fields canonical-identical)
# ---------------------------------------------------------------------------


def build_manifest_link(
    off_manifest: Mapping,
    *,
    capture_id: str,
    filename: str,
    sha256: str,
    nbytes: int,
) -> dict:
    """Return the ON manifest: ``off_manifest`` plus exactly one additive block.

    The pre-existing fields are copied verbatim (canonical-identical by
    construction). This is **not** a whole-manifest byte-identity claim: only the
    pre-existing fields are canonical-identical ON/OFF.
    """
    _require(isinstance(off_manifest, Mapping), "off_manifest must be a mapping")
    _require(MANIFEST_LINK_KEY not in off_manifest, f"{MANIFEST_LINK_KEY!r} already present in the OFF manifest")
    _require(isinstance(capture_id, str) and bool(capture_id), "capture_id must be a non-empty string")
    _require(isinstance(filename, str) and filename and Path(filename).name == filename, "filename must be a bare basename")
    _require("/" not in filename and "\\" not in filename, "filename must not contain a path separator")
    _require(isinstance(sha256, str) and bool(_HEX64.match(sha256)), "sha256 must be 64-hex")
    _require(
        isinstance(nbytes, (int, np.integer)) and not isinstance(nbytes, bool) and int(nbytes) >= 0,
        "bytes must be a non-negative int",
    )
    on_manifest = copy.deepcopy(dict(off_manifest))
    on_manifest[MANIFEST_LINK_KEY] = {
        "capture_id": capture_id,
        "filename": filename,
        "sha256": sha256,
        "bytes": int(nbytes),
    }
    return on_manifest


def canonical_manifest_without_link(manifest: Mapping) -> str:
    """Canonical JSON of a manifest with the R40 additive block removed."""
    reduced = {k: v for k, v in manifest.items() if k != MANIFEST_LINK_KEY}
    return canonical_json(reduced)


def assert_manifest_link_additive(off_manifest: Mapping, on_manifest: Mapping) -> dict:
    """Prove the ONLY difference is the single additive ``r40_diagnostic_capture`` block."""
    added = sorted(set(on_manifest) - set(off_manifest))
    removed = sorted(set(off_manifest) - set(on_manifest))
    if added != [MANIFEST_LINK_KEY] or removed:
        raise SchemaError(
            "manifest link must add exactly one block and remove nothing "
            f"(added={added}, removed={removed})"
        )
    if canonical_manifest_without_link(off_manifest) != canonical_manifest_without_link(on_manifest):
        raise SchemaError("pre-existing manifest fields are not canonical-identical ON/OFF")
    block = on_manifest[MANIFEST_LINK_KEY]
    if set(block) != {"capture_id", "filename", "sha256", "bytes"}:
        raise SchemaError("r40_diagnostic_capture block has unexpected fields")
    return {
        "additive": True,
        "added_keys": added,
        "pre_existing_canonical_identical": True,
        "block_keys": sorted(block),
        "whole_manifest_byte_identical": False,
    }


# ---------------------------------------------------------------------------
# Sparse worker payload + deterministic parent join (no full E7–E9 planes)
# ---------------------------------------------------------------------------


def _validate_payload_sample(name: str, value: Any) -> Any:
    if name in _BOOL_FIELDS:
        _require(type(value) is bool, f"payload sample {name} must be a strict bool")
        return value
    if name in _INT_FIELDS:
        _require(
            isinstance(value, (int, np.integer)) and not isinstance(value, bool),
            f"payload sample {name} must be an int",
        )
        return int(value)
    _require(
        isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool),
        f"payload sample {name} must be numeric",
    )
    fv = float(value)
    _require(math.isfinite(fv), f"payload sample {name} must be finite", NonFiniteError)
    return fv


def validate_sparse_payload(payload: Mapping) -> dict:
    """Validate a versioned sparse worker payload (fail closed).

    A payload carries only the sparse candidate coordinates of ONE cell role; it
    never carries a full patch plane.
    """
    _require(isinstance(payload, Mapping), "payload must be a mapping")
    for key in ("schema", "schema_version", "cell_id", "role", "relation_id", "channel", "samples"):
        _require(key in payload, f"payload missing field {key!r}")
    _require(payload["schema"] == PAYLOAD_SCHEMA, "payload schema mismatch")
    _require(payload["schema_version"] == PAYLOAD_VERSION, "payload schema_version mismatch")
    _require(isinstance(payload["cell_id"], str) and bool(payload["cell_id"]), "payload cell_id must be a string")
    _require(payload["role"] in PAYLOAD_ROLES, f"payload role must be one of {PAYLOAD_ROLES}")
    role = payload["role"]
    channel = payload["channel"]
    _require(isinstance(channel, (int, np.integer)) and not isinstance(channel, bool), "payload channel must be an int")
    channel = int(channel)
    _require(0 <= channel < N_CHANNELS, "payload channel out of range")

    side_fields = PAYLOAD_FIELDS_BY_ROLE[role]
    samples = list(payload["samples"])
    seen: set[tuple[int, int]] = set()
    rows: list[dict] = []
    for sample in samples:
        _require(isinstance(sample, Mapping), "each payload sample must be a mapping")
        _require("gx" in sample and "gy" in sample, "payload sample must carry gx, gy")
        gx = int(sample["gx"])
        gy = int(sample["gy"])
        key = (gx, gy)
        if key in seen:
            raise DuplicateKeyError(f"duplicate candidate key {key} in payload")
        seen.add(key)
        row = {"gx": gx, "gy": gy}
        for name in side_fields:
            _require(name in sample, f"payload sample missing {name!r}")
            row[name] = _validate_payload_sample(name, sample[name])
        rows.append(row)
    rows.sort(key=lambda r: (int(r["gy"]), int(r["gx"])))
    digest = candidate_set_digest([(r["gx"], r["gy"]) for r in rows])
    return {
        "schema": PAYLOAD_SCHEMA,
        "schema_version": PAYLOAD_VERSION,
        "cell_id": payload["cell_id"],
        "role": role,
        "relation_id": str(payload["relation_id"]),
        "channel": channel,
        "count": len(rows),
        "candidate_set_digest": digest,
        "samples": rows,
    }


def build_sparse_payload(
    *,
    cell_id: str,
    role: str,
    relation_id: str,
    channel: int,
    samples: Sequence[Mapping],
) -> dict:
    """Build a validated sparse payload (candidate coordinates only; no planes)."""
    return validate_sparse_payload({
        "schema": PAYLOAD_SCHEMA,
        "schema_version": PAYLOAD_VERSION,
        "cell_id": cell_id,
        "role": role,
        "relation_id": relation_id,
        "channel": channel,
        "samples": [dict(s) for s in samples],
    })


def parent_join_records(
    payload_u: Mapping,
    payload_v: Mapping,
    *,
    keys: Iterable[tuple[int, int]] | None = None,
) -> list[dict]:
    """Deterministically join two sparse payloads at the exact held-out keys.

    No full patch plane is retained: only the sparse candidate samples. The join
    is fail-closed on schema, relation/channel identity and duplicate keys.
    """
    pu = validate_sparse_payload(payload_u)
    pv = validate_sparse_payload(payload_v)
    _require(pu["role"] == "u" and pv["role"] == "v", "join requires roles 'u' and 'v'")
    _require(pu["relation_id"] == pv["relation_id"], "join requires the same relation_id")
    _require(pu["channel"] == pv["channel"], "join requires the same channel")

    by_u = {(int(s["gx"]), int(s["gy"])): s for s in pu["samples"]}
    by_v = {(int(s["gx"]), int(s["gy"])): s for s in pv["samples"]}
    if keys is None:
        selected = sorted(set(by_u) & set(by_v))
    else:
        selected = sorted({(int(gx), int(gy)) for gx, gy in keys})
        for key in selected:
            if key not in by_u or key not in by_v:
                raise SchemaError(f"held-out key {key} absent from a payload; refusing the join")
    rows: list[dict] = []
    for key in selected:
        row: dict[str, Any] = {"gx": key[0], "gy": key[1], "channel": pu["channel"]}
        for field_name in PAYLOAD_FIELDS_BY_ROLE["u"]:
            row[field_name] = by_u[key][field_name]
        for field_name in PAYLOAD_FIELDS_BY_ROLE["v"]:
            row[field_name] = by_v[key][field_name]
        rows.append(row)
    return rows


def assemble_relation_records(
    payload_pairs: Sequence[Sequence[Mapping]],
    *,
    keys: Iterable[tuple[int, int]] | None = None,
) -> tuple[dict, ...]:
    """Assemble a full relation's records across channels in frozen order.

    Ordering is deterministic ``(channel, gy, gx)``; a channel may appear in only
    one pair.
    """
    rows: list[dict] = []
    seen_channels: set[int] = set()
    for pair in payload_pairs:
        _require(len(pair) == 2, "each payload pair must be (payload_u, payload_v)")
        channel = validate_sparse_payload(pair[0])["channel"]
        if channel in seen_channels:
            raise DuplicateKeyError(f"channel {channel} provided more than once for one relation")
        seen_channels.add(channel)
        rows.extend(parent_join_records(pair[0], pair[1], keys=keys))
    rows.sort(key=lambda r: (int(r["channel"]), int(r["gy"]), int(r["gx"])))
    return tuple(rows)


def payload_retained_scalars(payload: Mapping) -> int:
    """Number of retained scalars — proves the payload is sparse, not a full plane."""
    validated = validate_sparse_payload(payload)
    return int(validated["count"] * (2 + len(PAYLOAD_FIELDS_BY_ROLE[validated["role"]])))


__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "CAPTURE_CONTRACT_VERSION",
    "CAPTURE_STAGE",
    "QUALIFICATION_EXACT",
    "MISSION",
    "ORIENTATIONS",
    "N_CHANNELS",
    "RECORD_ORDERING",
    "SAMPLE_KEY_TAG",
    "PAIRSET_TAG",
    "CANDIDATE_SET_TAG",
    "IDENTITY_TAG",
    "MANIFEST_LINK_KEY",
    "SOURCE_SET_DIGEST_ALGO",
    "GENERATION_COMMIT_BASENAME",
    "GENERATION_INDEX_BASENAME",
    "DEFAULT_DISK_BUDGET_BYTES",
    "B1_PROTOCOL_SHA256",
    "B1_CRITERIA_SHA256",
    "B1_RESULT_SHA256",
    "B1_CLAIM_ID",
    "EXPECTED_PRODUCTION_WIRED",
    "COMMIT_FIELDS",
    "R40_RECORD_DTYPE",
    "R40_RECORD_NAMES",
    "R40_RECORD_ITEMSIZE",
    "SAMPLE_PLANE_FIELDS",
    "R40CaptureConfig",
    "RelationShard",
    "R40CaptureError",
    "CaptureDisabledError",
    "SchemaError",
    "NonFiniteError",
    "IdentityError",
    "BudgetError",
    "PathSafetyError",
    "PrivacyError",
    "StaleGenerationError",
    "DuplicateKeyError",
    "GenerationExistsError",
    "dumps_strict",
    "canonical_json",
    "sha256_bytes",
    "sha256_text",
    "sha256_payload",
    "assert_finite_json",
    "assert_finite_array",
    "path_leak_category",
    "scan_path_leaks",
    "assert_no_path_leaks",
    "resolve_fresh_output_root",
    "assert_generation_is_fresh",
    "sample_key",
    "pairset_digest",
    "candidate_set_digest",
    "relation_id_for",
    "INV_TOL",
    "forward_coord",
    "inverse_index",
    "assert_block_parity",
    "heldout_keys_from_xy",
    "record_dtype_descriptor",
    "assert_record_dtype_frozen",
    "validate_relation_header",
    "encode_records",
    "default_capture_config",
    "assert_capture_enabled",
    "check_capture_identity",
    "identity_digest",
    "make_capture_identity",
    "estimate_generation_bytes",
    "check_disk_budget",
    "write_capture_generation",
    "verify_generation",
    "build_checkpoint",
    "check_resume",
    "assert_checkpoint_complete",
    "build_manifest_link",
    "canonical_manifest_without_link",
    "assert_manifest_link_additive",
    "validate_sparse_payload",
    "build_sparse_payload",
    "parent_join_records",
    "assemble_relation_records",
    "payload_retained_scalars",
]
