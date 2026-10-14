#!/usr/bin/env python3
"""ZM-ZEGRID-R38-A — bounded, read-only, offline causal diagnostic harness.

MISSION
-------
Decide (falsify or confirm) whether *per-Cell gauge choices* (frame set,
reference, coefficients, weights/exclusions) are a dominant cause of the
inter-Cell disagreement observed on the exact Caldwell Windows-816 pre-feather
capture. This is a DIAGNOSTIC. It applies NO correction, wires NOTHING into the
production path, and writes NOTHING under the read-only Caldwell tree.

WHAT IT DOES (three explicit, opt-in modes)
-------------------------------------------
``inventory`` (default, writes only the strict-JSON report)
    1. strictly validates the exact capture + manifest identity/linkage
       (816 frames / 6x4 / EXACT / expected capture id + artifact sha256 +
       source-set digest), independently recomputing the digest and the file
       hashes;
    2. resolves every manifest frame id to EXACTLY one read-only source file;
    3. rebuilds the accepted canvas / layout / Cell membership by REUSING the
       production geometry + layout primitives (no duplicated pixel math);
    4. freezes a deterministic, BIASED DEV relation panel (six geometric
       relations, both orientations, the R37-declared worst edge, median/low
       controls, at least one canvas corner) BEFORE any factor result, and
       labels it NOT QUALIFICATION;
    5. estimates the bounded resource cost of the probe (frames / disk / time).

``gauge`` (needs ``--execute-real``)
    Additionally reproduces the LOCAL per-Cell gauge facts on each selected
    Cell's FULL accepted domain: it streams the Cell's aligned patch from the
    read-only FITS (production decoder / crop / reproject primitives, one frame
    resident) into a scratch aligned cache, then runs the FROZEN estimator
    (``photometric.compute_cell_gauge``) on that provider and records the
    ordered frame ids, reference id/support, coefficients, weights, active
    flags and exclusions. Cross-side comparisons are then computed on the
    common active frames.

``factors`` (needs ``--execute-real`` and an existing ``gauge`` run)
    Additionally runs the one-variable-at-a-time factor probe A/B/C/D on the
    bounded shared-pixel ROI of each probed relation, on a FIXED CPU backend,
    reusing the canonical streaming executor with a fixed gauge and a read-only
    ROI provider (no dense N-frame x canvas cube, no duplicated science).

DISCIPLINE
----------
* Module-scope imports are stdlib + numpy only; every ``zemosaic`` import is
  lazy, so importing this tool never activates the engine (import audit below).
* Read-only w.r.t. the source tree: the tool refuses any output/scratch path
  inside the capture dir, the source dir, or the Caldwell tree.
* Strict JSON: ``allow_nan=False``; a non-finite value is emitted as ``null`` /
  an explicit missing marker, never as a fabricated ``0``.
* No new dependency, no network, no third-party code.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Schema + frozen expectations for the CHECKED run
# ---------------------------------------------------------------------------

SCHEMA_NAME = "zemosaic.zegrid.r38a.causal_probe"
SCHEMA_VERSION = "r38a-1"
MISSION = "ZM-ZEGRID-R38-A-CAUSAL"
CAPTURE_ARTIFACT_BASENAME = "zegrid_r37_pre_feather.json"
MANIFEST_BASENAME = "zegrid_manifest.json"
CAPTURE_STAGE = "pre_offset_pre_feather"
SOURCE_SET_DIGEST_ALGO = "sha256_sorted_unique_relative_logical_paths_v1"

# The exact accepted Windows-816 capture this checked run is recorded against.
CHECKED_IDENTITY = {
    "n_source_frames": 816,
    "canvas_width": 4117,
    "canvas_height": 4377,
    "canvas_id": "e8f1877f16f8cca4",
    "nx": 6,
    "ny": 4,
    "capture_id": "6c311af729425e69c030f87969fe6369dd9e9fc4ab9c133379662d70b65a8e00",
    "artifact_sha256": "a260b3309c39fbfc5264cacc60ba3048c678f83cf8bf6bf9f39c94a134a84032",
    "artifact_bytes": 23252804,
    "source_set_digest": "e3f4c16af30fca1a4c660ba5f76ea6f3d62868dd825bc09a9d83802bfccee3ae",
    "qualification": "EXACT",
    "effective_cell_gauge_mode": "per_cell",
    "halo_px": 8,
    "measurement_px": 16,
    "feather_px": 8,
}

# R37 declared the per-edge p90 worst relation of the Windows-816 witness (from
# the accepted scientific result report). Used ONLY as a named panel input.
DECLARED_WORST_EDGE = ("r0003c0002", "r0003c0003")

# Panel composition rule (frozen BEFORE any factor result).
PANEL_SLOTS = (
    "declared_worst_edge",
    "highest_burden_vertical",
    "highest_burden_overall",
    "median_burden",
    "minimum_burden",
    "corner_relation",
)

# Pre-registered conclusion thresholds (frozen; DEV diagnostic only).
GAUGE_STEP_DOMINANCE_FRAC = 0.80     # |step| >= 80% of observed -> dominant candidate
GAUGE_STEP_PARTIAL_FRAC = 0.50       # |step| >= 50% of observed -> partial contributor
GAUGE_STEP_NEGLIGIBLE_FRAC = 0.05    # |step| <= 5% of observed -> falsifies gauge step
GAUGE_STEP_CONSISTENCY_MAX = 0.25    # mad(|step|) / |step| <= 25% -> stable step
MIN_PROBED_RELATIONS = 2             # fewer probed relations -> INCONCLUSIVE

REPORT_DIR_DEFAULT = Path(
    os.environ.get("R38A_REPORT_DIR")
    or (Path.home() / ".openclaw" / "workspace" / ".a2a-reports" / "ZM-ZEGRID-R38-A-CAUSAL")
)
SCRATCH_DIR_DEFAULT = Path(os.environ.get("R38A_SCRATCH_DIR")
                           or (Path(tempfile.gettempdir()) / "openclaw-r38a"))

# Read-only trees: the RUN's own capture/source roots are always protected, a
# checked run additionally derives a mandatory dataset root, and extra roots can be
# supplied via the ``R38A_FORBIDDEN_ROOTS`` environment variable (os.pathsep
# -separated, additive). No private absolute path is hard-coded here.


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ProbeError(RuntimeError):
    """Deterministic, user-facing harness failure (never a traceback dump)."""


class IdentityError(ProbeError):
    pass


class ResolutionError(ProbeError):
    pass


class BudgetError(ProbeError):
    pass


# ---------------------------------------------------------------------------
# Import audit (module scope must never pull ``zemosaic``)
# ---------------------------------------------------------------------------

def module_scope_imports() -> list[str]:
    """Return the fully-qualified module names imported at MODULE scope.

    Parses this file's own AST: the returned list is the static import
    surface of the tool, used by the tests to prove the tool never imports a
    ``zemosaic`` module at import time (all engine access is lazy).
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def import_audit() -> dict:
    """Bounded, strict-JSON import audit (part of every report)."""
    names = module_scope_imports()
    zem = sorted(n for n in names if n == "zemosaic" or n.startswith("zemosaic."))
    return {
        "module_scope_imports": sorted(names),
        "zemosaic_module_scope_imports": zem,
        "engine_access_is_lazy": not zem,
    }


# ---------------------------------------------------------------------------
# Strict JSON helpers
# ---------------------------------------------------------------------------

def _reject_constant(name: str):
    raise ValueError(f"non-finite JSON literal {name!r} is not allowed")


def load_json_strict(path: Path) -> Any:
    """Load JSON refusing NaN/Infinity literals (strict, non-tautological)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh, parse_constant=_reject_constant)


def _fin(x) -> float | None:
    """Finite float or ``None`` (never a fabricated zero)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def dumps_strict(payload: Any) -> str:
    """Canonical strict JSON (sorted keys, allow_nan=False)."""
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_set_digest(identifiers) -> str:
    """Re-implementation of the accepted ``background_diagnostics`` digest.

    The canonical algorithm tag is hashed first, then each SORTED, unique,
    newline-terminated relative logical id. Kept byte-compatible on purpose so
    an independent recomputation can be compared to the artifact/manifest.
    """
    uniq = sorted({str(x) for x in identifiers if str(x)})
    h = hashlib.sha256()
    h.update((SOURCE_SET_DIGEST_ALGO + "\n").encode("utf-8"))
    for item in uniq:
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Path privacy: logical labels + recursive fail-closed scan/scrub
# ---------------------------------------------------------------------------

_WIN_DRIVE_RE = re.compile(r"(?:^|[^A-Za-z0-9])[A-Za-z]:[\\/]")
_UNC_RE = re.compile(r"(?:^|[^\\])\\\\[A-Za-z0-9._$-]+[\\]")
_FILE_URI_RE = re.compile(r"file://", re.IGNORECASE)
# Generalized EMBEDDED POSIX absolute path: a single leading '/' (not a URL scheme
# separator, not a fraction/ratio, not a Windows drive) followed by at least one
# path-like component. Deliberately NOT limited to a fixed root allowlist, so an
# unknown mount ('see /custom/mount/x') is still caught.
_POSIX_ABS_EMBEDDED_RE = re.compile(
    r"(?<![\w:/])(?:/(?:[A-Za-z0-9._~+\-]+/)+[A-Za-z0-9._~+\-]*|/[A-Za-z0-9._~+\-]+)"
)
REDACTED_PATH = "<redacted:absolute-path>"


def path_leak_category(value: str) -> str | None:
    """Return a leak category when ``value`` carries an absolute/private path.

    Recognises POSIX-absolute (whole value OR embedded under ANY root), Windows
    drive-absolute, UNC and ``file://`` URI forms. Ratios (``1/2``), relative
    ids (``a/b.fit``) and URL schemes (``https://x/y``) are not flagged.
    """
    if not isinstance(value, str) or not value:
        return None
    if _FILE_URI_RE.search(value):
        return "file_uri"
    if _WIN_DRIVE_RE.search(value):
        return "windows_drive_absolute"
    if _UNC_RE.search(value) or value.startswith("\\\\"):
        return "unc"
    if value.startswith("/"):
        return "posix_absolute"
    if _POSIX_ABS_EMBEDDED_RE.search(value):
        return "posix_absolute_embedded"
    return None


def scan_path_leaks(payload: Any, prefix: str = "$") -> list[dict]:
    """Recursively scan a JSON-like payload for absolute/private paths (fail closed)."""
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
            found.append({"json_path": prefix, "category": cat, "value": payload})
    return found


def scrub_paths(payload: Any, roots: Sequence[Path]) -> Any:
    """Recursively replace every known private root with a stable logical label.

    Defence in depth on top of never emitting absolute paths: any occurrence of a
    known root (whole value or embedded in text) becomes ``<root:LABEL>``.
    """
    pairs = []
    for i, root in enumerate(roots):
        if root is None:
            continue
        label = f"<root:{Path(root).name or 'root'}>"
        try:
            pairs.append((str(Path(root).expanduser().resolve()), label))
        except Exception:
            pairs.append((str(root), label))
        pairs.append((str(root), label))
    pairs.sort(key=lambda kv: -len(kv[0]))

    def _walk(obj):
        if isinstance(obj, Mapping):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_walk(v) for v in obj)
        if isinstance(obj, str):
            out = obj
            for raw, label in pairs:
                if raw and raw in out:
                    out = out.replace(raw, label)
            return out
        return obj

    return _walk(payload)


def redact_path_leaks(payload: Any) -> Any:
    """Recursively REPLACE every value carrying a detected absolute path.

    Unlike :func:`scrub_paths` (which only rewrites KNOWN roots), this is
    unconditional: any offending value becomes ``<redacted:absolute-path>`` so the
    final bytes can never persist the leak. Applied before every write, including
    the FAILED path.
    """
    if isinstance(payload, Mapping):
        return {k: redact_path_leaks(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [redact_path_leaks(v) for v in payload]
    if isinstance(payload, tuple):
        return tuple(redact_path_leaks(v) for v in payload)
    if isinstance(payload, str) and path_leak_category(payload) is not None:
        return REDACTED_PATH
    return payload


def minimal_failure_record(*, stage: str, leak_locations: Sequence[Mapping]) -> dict:
    """Smallest SAFE failure record (never contains the offending values)."""
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "mission": MISSION,
        "status": "FAILED",
        "status_reason": "durable report failed the privacy boundary",
        "qualification": "DEV_NOT_QUALIFICATION",
        "scientific_acceptance": False,
        "r38b_authorized": False,
        "failed": {
            "stage": str(stage),
            "leak_count": int(len(leak_locations)),
            "categories": sorted({str(l.get("category")) for l in leak_locations}),
            "json_paths": sorted({str(l.get("json_path")) for l in leak_locations}),
        },
        "privacy": {"policy": "no absolute/private path may appear in the durable artifact",
                    "leaks": int(len(leak_locations))},
    }


def enforce_privacy_boundary(payload: Any, roots: Sequence[Path]) -> tuple[Any, list[dict]]:
    """Scrub known roots, redact ANY remaining leak, then rescan (fail closed).

    Returns ``(safe_payload, remaining_leaks)``. The caller MUST refuse to write
    when ``remaining_leaks`` is non-empty.
    """
    scrubbed = scrub_paths(payload, roots)
    redacted = redact_path_leaks(scrubbed)
    return redacted, scan_path_leaks(redacted)


def one_way_path_identity(tag: str, path) -> str:
    """Domain-separated one-way identity of a NORMALIZED resolved path.

    The raw path is never persisted; two different locations never collide here.
    """
    try:
        norm = str(Path(path).expanduser().resolve())
    except Exception:
        norm = str(path)
    h = hashlib.sha256()
    h.update((str(tag) + "\n").encode("utf-8"))
    h.update(norm.encode("utf-8"))
    return h.hexdigest()


def derive_dataset_root(capture_dir: Path, source_dir: Path) -> Path:
    """Nearest common ancestor of the resolved capture and source dirs (fail closed).

    Refuses a too-broad derivation (filesystem/drive root, the user's home or any
    ancestor of it, or a single-component path) instead of silently weakening the
    no-writes-under-the-dataset invariant.
    """
    try:
        common = Path(os.path.commonpath([str(Path(capture_dir).resolve()),
                                          str(Path(source_dir).resolve())]))
    except ValueError as exc:
        raise ResolutionError(f"cannot derive a dataset root: {type(exc).__name__}") from exc
    try:
        common = common.resolve()
    except OSError as exc:
        raise ResolutionError(f"cannot resolve the derived dataset root: {type(exc).__name__}") from exc
    if common == Path(common.anchor):
        raise ResolutionError("derived dataset root is a filesystem/drive root; refusing (too broad)")
    if len(common.parts) < 2:
        raise ResolutionError("derived dataset root is too shallow; refusing (too broad)")
    home = Path.home().resolve()
    if common == home or common in home.parents:
        raise ResolutionError("derived dataset root is home or an ancestor of home; refusing (too broad)")
    return common


def build_write_policy(*, capture_dir: Path, source_dir: Path, expect: str,
                       env_roots: Sequence[Path] = ()) -> dict:
    """Immutable write policy: the protected read-only roots for this run.

    ``expect == "checked"`` derives a MANDATORY dataset root (nearest common
    ancestor of capture and source) and refuses an unsafe derivation. Generic
    ``expect == "any"`` protects only the capture/source roots plus ``env_roots``
    and says so explicitly (``mode = generic_capture_source_only``); the path is
    never persisted — only the mode and a one-way identity hash.
    """
    capture_dir = Path(capture_dir).resolve()
    source_dir = Path(source_dir).resolve()
    protected = [capture_dir, source_dir] + [Path(p).expanduser().resolve() for p in env_roots]
    policy = {
        "mode": "generic_capture_source_only",
        "protected_roots": tuple(protected),
        "dataset_root_identity_sha256": None,
        "env_roots_count": len(list(env_roots)),
    }
    if expect == "checked":
        dataset_root = derive_dataset_root(capture_dir, source_dir)
        policy["protected_roots"] = tuple(protected + [dataset_root])
        policy["mode"] = "derived_common_ancestor"
        policy["dataset_root_identity_sha256"] = one_way_path_identity(
            "zemosaic.zegrid.r38a.dataset_root.v1", dataset_root)
    elif expect != "any":
        raise ProbeError(f"unknown expect {expect!r}")
    return policy


def _assert_write_allowed(path: Path, *, label: str, policy: Mapping | None = None,
                          extra_forbidden: Sequence[Path] = ()) -> Path:
    """Refuse an output/scratch/report path inside any protected read-only root."""
    p = Path(path).expanduser()
    rp = p.resolve()
    roots = list(extra_forbidden)
    if policy is not None:
        roots.extend(policy.get("protected_roots") or ())
    for root in roots:
        rr = Path(root).expanduser().resolve()
        if rp == rr or rr in rp.parents:
            raise BudgetError(f"{label} is inside a protected read-only tree; refusing")
    return p


def _is_drive_absolute(raw: str) -> bool:
    """True for Windows drive-absolute ids (drive letter + ``/`` or backslash)."""
    return bool(re.match(r"^[A-Za-z]:[\\/]", str(raw).strip()))


def _symlink_component(path: Path, root: Path) -> bool:
    """True when any component of ``path`` relative to ``root`` is a symlink."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    cur = root
    for part in rel.parts:
        cur = cur / part
        try:
            if cur.is_symlink():
                return True
        except OSError:
            return True
    return False


# ---------------------------------------------------------------------------
# Lazy engine access
# ---------------------------------------------------------------------------

def _ensure_src_on_path() -> None:
    repo = Path(__file__).resolve().parents[1]
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _lazy_zemosaic():
    """Import the engine surface lazily; never at module scope."""
    _ensure_src_on_path()
    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid import auto_layout as zal
    from zemosaic.core.zegrid import background_model as zbm
    from zemosaic.core.zegrid import file_provider as zfp
    from zemosaic.core.zegrid import geometry as zg
    from zemosaic.core.zegrid import photometric as zphot
    from zemosaic.core.zegrid import streaming as zstream
    from zemosaic.core.zegrid import sweep as zsw
    from zemosaic.core.zegrid.science_adapter import MiniTileScienceConfig

    return {
        "zz": zz,
        "zal": zal,
        "zbm": zbm,
        "zfp": zfp,
        "zg": zg,
        "zphot": zphot,
        "zstream": zstream,
        "zsw": zsw,
        "cfg_cls": MiniTileScienceConfig,
    }


def _quiet() -> None:
    """Silence the product's verbose loader logging (stdout noise only)."""
    logging.disable(logging.INFO)


def _not_run_exceptions() -> tuple:
    """Exception types that may honestly map to a NOT_RUN decision.

    ONLY documented domain/budget failures. A programming error (TypeError,
    AttributeError, IndexError, broadcast ValueError, ...) is deliberately NOT in
    this tuple so it fails the run instead of being disguised as scientific
    NOT_RUN.
    """
    z = _lazy_zemosaic()
    from zemosaic.core.canonical_stacking import (
        CanonicalStackFailure,
        CanonicalStackValidationError,
    )

    return (
        BudgetError,
        ProbeError,
        ResolutionError,
        IdentityError,
        CanonicalStackFailure,
        CanonicalStackValidationError,
        FileNotFoundError,
        MemoryError,
    )


# ---------------------------------------------------------------------------
# 1. Identity validation
# ---------------------------------------------------------------------------

def _require(cond: bool, message: str, exc=IdentityError) -> None:
    if not cond:
        raise exc(message)


def validate_identity(
    capture_dir: Path,
    *,
    expect: str = "checked",
) -> dict:
    """Strictly validate the exact capture + manifest identity/linkage.

    ``expect="checked"`` additionally requires the recorded CHECKED identity
    (816 / 6x4 / EXACT / expected capture id + artifact sha256 + source digest).
    ``expect="any"`` accepts another valid artifact but records the computed
    identity. Both paths recompute the source-set digest, the on-disk artifact
    sha256/bytes and cross-link manifest <-> artifact.
    """
    cap_json = capture_dir / CAPTURE_ARTIFACT_BASENAME
    man_json = capture_dir / MANIFEST_BASENAME
    _require(cap_json.is_file(), f"capture artifact missing: {cap_json.name}")
    _require(man_json.is_file(), f"manifest missing: {man_json.name}")

    artifact = load_json_strict(cap_json)
    manifest = load_json_strict(man_json)
    _require(isinstance(artifact, dict), "capture artifact is not a JSON object")
    _require(isinstance(manifest, dict), "manifest is not a JSON object")

    art_sha = sha256_file(cap_json)
    art_bytes = int(cap_json.stat().st_size)

    # --- artifact provenance + shape facts --------------------------------
    _require(artifact.get("schema") == "zemosaic.zegrid.background_diagnostics",
             "artifact schema mismatch")
    _require(artifact.get("schema_version") == "r37c-1", "artifact schema_version mismatch")
    prov = artifact.get("provenance")
    _require(isinstance(prov, Mapping), "artifact provenance missing")
    cap = prov.get("capture")
    _require(isinstance(cap, Mapping), "artifact provenance.capture missing")
    _require(cap.get("capture_stage") == CAPTURE_STAGE, "artifact capture_stage mismatch")
    _require(bool(prov.get("correction_applied")) is False, "artifact applied a correction")
    _require(bool(prov.get("production_wired")) is True, "artifact not production-wired")

    meta = artifact.get("meta")
    _require(isinstance(meta, Mapping), "artifact meta missing")

    art_facts = {
        "capture_id": str(cap.get("capture_id", "")),
        "n_source_frames": int(cap.get("n_source_frames", -1)),
        "source_set_digest": str(cap.get("source_set_digest", "")),
        "source_set_digest_algo": str(cap.get("source_set_digest_algo", "")),
        "effective_cell_gauge_mode": str(cap.get("effective_cell_gauge_mode", "")),
        "halo_px": int(cap.get("halo_px", -1)),
        "measurement_px": int(cap.get("measurement_px", -1)),
        "feather_px": int(cap.get("feather_px", -1)),
        "n_relations": int(meta.get("n_relations", -1)),
        "n_dropped": int(meta.get("n_dropped", -1)),
        "nx": int(meta.get("nx", -1)),
        "ny": int(meta.get("ny", -1)),
        "canvas_hw": [int(v) for v in meta.get("canvas_hw", [])],
    }

    # --- manifest facts ----------------------------------------------------
    _require(manifest.get("engine") == "zegrid", "manifest engine != zegrid")
    fids = manifest.get("frame_ids")
    _require(isinstance(fids, list) and fids, "manifest frame_ids missing")
    _require(all(isinstance(x, str) and x for x in fids), "manifest frame_ids malformed")
    uniq = sorted(set(fids))
    _require(len(uniq) == len(fids), "manifest frame_ids contain duplicates")
    n_frames = int(manifest.get("n_frames", -1))
    _require(n_frames == len(fids), "manifest n_frames != len(frame_ids)")

    gauge = manifest.get("photometric_gauge")
    _require(isinstance(gauge, Mapping), "manifest photometric_gauge missing")
    effective_mode = str(gauge.get("mode", ""))

    feather = manifest.get("cell_feather")
    _require(isinstance(feather, Mapping), "manifest cell_feather missing")
    _require(bool(feather.get("enabled")) is True, "manifest cell_feather disabled")
    capblk = feather.get("r37_diagnostic_capture")
    _require(isinstance(capblk, Mapping), "manifest r37 capture block missing")

    layout = manifest.get("layout")
    canvas = manifest.get("canvas")
    _require(isinstance(layout, Mapping) and isinstance(canvas, Mapping),
             "manifest canvas/layout missing")

    computed_digest = source_set_digest(uniq)
    _require(
        computed_digest == art_facts["source_set_digest"],
        f"recomputed source-set digest {computed_digest} != artifact "
        f"{art_facts['source_set_digest']}",
    )
    _require(art_facts["source_set_digest_algo"] == SOURCE_SET_DIGEST_ALGO,
             "artifact source_set_digest_algo mismatch")
    _require(str(capblk.get("source_set_digest", "")) == computed_digest,
             "manifest capture block source-set digest mismatch")
    _require(str(capblk.get("source_set_digest_algo", "")) == SOURCE_SET_DIGEST_ALGO,
             "manifest capture block digest algo mismatch")
    _require(int(capblk.get("n_source_frames", -1)) == len(uniq),
             "manifest capture block source count mismatch")

    # --- manifest <-> artifact linkage (non-tautological: disk-derived) ----
    _require(str(capblk.get("capture_id", "")) == art_facts["capture_id"],
             "manifest capture_id != artifact capture_id")
    _require(str(capblk.get("filename", "")) == CAPTURE_ARTIFACT_BASENAME,
             "manifest capture filename mismatch")
    _require(str(capblk.get("sha256", "")) == art_sha,
             "manifest capture sha256 != on-disk artifact sha256")
    _require(int(capblk.get("bytes", -1)) == art_bytes,
             "manifest capture bytes != on-disk artifact bytes")

    # --- EXACT qualification ----------------------------------------------
    qualification = str(capblk.get("qualification", ""))
    holdout_exact = bool(capblk.get("holdout_exact"))
    holdout_capped = bool(capblk.get("holdout_capped"))
    _require(holdout_capped is False, "capture holdout is capped (not EXACT)")
    _require(holdout_exact is True, "capture holdout_exact is not true")
    _require(art_facts["n_dropped"] == 0, "artifact dropped relations")

    computed = {
        "artifact_sha256": art_sha,
        "artifact_bytes": art_bytes,
        "source_set_digest": computed_digest,
        "n_source_frames": len(uniq),
        "canvas_width": int(canvas.get("width", -1)),
        "canvas_height": int(canvas.get("height", -1)),
        "canvas_id": str(canvas.get("id", "")),
        "nx": int(layout.get("nx", -1)),
        "ny": int(layout.get("ny", -1)),
        "qualification": qualification,
        "effective_cell_gauge_mode": effective_mode,
        "capture_id": art_facts["capture_id"],
        "halo_px": art_facts["halo_px"],
        "measurement_px": art_facts["measurement_px"],
        "feather_px": art_facts["feather_px"],
        "n_relations": art_facts["n_relations"],
        "canvas_hw_artifact": art_facts["canvas_hw"],
    }

    # artifact canvas_hw must equal the manifest canvas (row, col order)
    _require(
        art_facts["canvas_hw"] == [computed["canvas_height"], computed["canvas_width"]],
        "artifact canvas_hw != manifest canvas",
    )
    _require(int(meta.get("nx", -1)) == computed["nx"] and int(meta.get("ny", -1)) == computed["ny"],
             "artifact nx/ny != manifest layout")
    _require(effective_mode == "per_cell", "effective cell gauge mode != per_cell")

    if expect == "checked":
        for key, want in CHECKED_IDENTITY.items():
            got = computed.get(key)
            _require(got == want, f"checked identity mismatch for {key!r}: got {got!r} want {want!r}")
    else:
        _require(expect == "any", f"unknown expect {expect!r}")

    checks = {
        "expect": expect,
        "checked_identity_matched": bool(expect == "checked"),
        "linkage": {
            "capture_id": True,
            "artifact_sha256": True,
            "artifact_bytes": True,
            "source_set_digest_recomputed": True,
            "manifest_qualification": qualification,
            "holdout_exact": holdout_exact,
            "holdout_capped": holdout_capped,
            "n_dropped": art_facts["n_dropped"],
        },
    }
    return {
        "capture_dir_label": capture_dir.name,
        "artifact": {
            "basename": CAPTURE_ARTIFACT_BASENAME,
            "sha256": art_sha,
            "bytes": art_bytes,
            "capture_id": art_facts["capture_id"],
            "schema": artifact.get("schema"),
            "schema_version": artifact.get("schema_version"),
            "relations": artifact.get("relations"),
            "meta": dict(meta),
        },
        "manifest": {
            "basename": MANIFEST_BASENAME,
            "n_frames": n_frames,
            "unique_frame_ids": len(uniq),
            "canvas": {"width": computed["canvas_width"], "height": computed["canvas_height"],
                       "id": computed["canvas_id"]},
            "layout": {"nx": computed["nx"], "ny": computed["ny"]},
            "effective_cell_gauge_mode": effective_mode,
            "qualification": qualification,
        },
        "computed": computed,
        "checks": checks,
        "frame_ids": uniq,
    }


# ---------------------------------------------------------------------------
# 2. Source resolution (read-only)
# ---------------------------------------------------------------------------

def _scan_fits(root: Path) -> list[Path]:
    """Read-only FITS inventory under ``root`` (fail-closed on symlinks)."""
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in (".fit", ".fits", ".fts"):
            continue
        try:
            if not p.is_file() or p.is_symlink():
                continue
        except OSError:
            continue
        if _symlink_component(p, root):
            continue
        out.append(p)
    return out


def resolve_sources(source_dir: Path, frame_ids: Sequence[str]) -> dict:
    """Resolve each manifest id to exactly one file under ``source_dir``.

    Deterministic; relative-path match first, then unique basename. Fail-closed:
    rejects absolute ids (POSIX/UNC/Windows drive-absolute, incl. slash, backslash
    and case variants), ``file://`` URIs, traversal, missing/ambiguous basenames,
    duplicate resolution, symlinked candidates (and any symlinked path component),
    and any resolved path escaping the source root. A symlinked source root is
    refused. Never writes anything.
    """
    _require(source_dir.is_dir(), f"source dir is not a directory: {source_dir.name}", ResolutionError)
    if source_dir.is_symlink():
        raise ResolutionError("source dir is a symlink; refusing (fail-closed)")
    root = source_dir
    root_res = root.resolve()
    files = _scan_fits(root)
    by_rel: dict[str, Path] = {}
    by_base: dict[str, list[Path]] = {}
    for f in files:
        by_rel.setdefault(f.relative_to(root).as_posix(), f)
        by_base.setdefault(f.name, []).append(f)

    resolved: list[Path] = []
    seen: set[Path] = set()
    problems: list[str] = []
    for idx, fid in enumerate(frame_ids):
        raw = str(fid)
        if _is_drive_absolute(raw):
            problems.append(f"frame_ids[{idx}]: Windows drive-absolute id refused")
            continue
        if raw.strip().lower().startswith("file://"):
            problems.append(f"frame_ids[{idx}]: file:// URI id refused")
            continue
        if "\\" in raw:
            problems.append(f"frame_ids[{idx}]: noncanonical backslash separator refused")
            continue
        norm = raw
        segs = norm.split("/")
        if any(s == "" for s in segs):
            problems.append(f"frame_ids[{idx}]: noncanonical empty path segment refused")
            continue
        if any(s == "." for s in segs):
            problems.append(f"frame_ids[{idx}]: noncanonical '.' path segment refused")
            continue
        pp = PurePosixPath(norm)
        if norm == "" or norm.startswith("/") or pp.is_absolute():
            problems.append(f"frame_ids[{idx}]: absolute/empty id refused")
            continue
        if any(part == ".." for part in pp.parts):
            problems.append(f"frame_ids[{idx}]: traversal refused")
            continue
        chosen = by_rel.get(norm)
        if chosen is None:
            matches = by_base.get(pp.name, [])
            if len(matches) == 1:
                chosen = matches[0]
            elif len(matches) > 1:
                problems.append(f"frame_ids[{idx}]: ambiguous basename {pp.name!r}")
                continue
            else:
                problems.append(f"frame_ids[{idx}]: not found under the source root")
                continue
        try:
            chosen.resolve().relative_to(root_res)
        except ValueError:
            problems.append(f"frame_ids[{idx}]: resolves outside the source root")
            continue
        if chosen.is_symlink():
            problems.append(f"frame_ids[{idx}]: symlinked candidate refused")
            continue
        if chosen in seen:
            problems.append(f"frame_ids[{idx}]: duplicate resolution")
            continue
        seen.add(chosen)
        resolved.append(chosen)
    if problems:
        raise ResolutionError(
            f"could not resolve {len(problems)} of {len(frame_ids)} ids: " + "; ".join(problems[:10])
        )
    _require(len(resolved) == len(frame_ids), "resolution count mismatch", ResolutionError)
    return {
        "source_root_label": root.name,
        "n_files_on_disk": len(files),
        "n_resolved": len(resolved),
        "paths": tuple(resolved),
    }


# ---------------------------------------------------------------------------
# 3. Geometry reconstruction (reuses production primitives)
# ---------------------------------------------------------------------------

def reconstruct_geometry(source_paths: Sequence[Path], source_dir: Path) -> dict:
    """Build descriptors + canvas + layout + per-Cell membership (production APIs)."""
    z = _lazy_zemosaic()
    zg, zal, zsw, zz = z["zg"], z["zal"], z["zsw"], z["zz"]
    from zemosaic.zemosaic_stack_plan import FrameInfo

    frames_info = [FrameInfo(path=Path(p), order=i) for i, p in enumerate(source_paths)]
    descs, rejected = zz._build_frame_descriptors(frames_info, str(source_dir), None, sip_mode="keep")
    _require(len(descs) == len(source_paths), f"descriptors built {len(descs)} != {len(source_paths)}")
    _require(not rejected, f"{len(rejected)} frames rejected at WCS qualification")
    canvas = zg.build_canvas(descs)
    footprints = zal.project_footprints(descs, canvas)
    contexts = zsw.build_cell_contexts_from_footprints(footprints, canvas, 6, 4, halo_px=8)

    cells: dict[str, dict] = {}
    cell_ctx: dict[str, tuple] = {}
    for (row, col, cell, patch, mem) in contexts:
        cells[cell.cell_id] = {
            "row": int(row),
            "col": int(col),
            "core": [int(patch.patch.x0 - (cell.core.x0 - patch.patch.x0)), 0, 0, 0],  # replaced below
            "patch": [int(patch.patch.x0), int(patch.patch.y0), int(patch.patch.x1), int(patch.patch.y1)],
            "patch_hw": [int(v) for v in patch.patch_shape_hw],
            "core_ids": list(mem.core_ids),
            "patch_ids": list(mem.patch_ids),
            "n_core": len(mem.core_ids),
            "n_patch": len(mem.patch_ids),
        }
        # canonical half-open core bounds (x0,y0,x1,y1)
        cells[cell.cell_id]["core"] = [
            int(cell.core.x0), int(cell.core.y0), int(cell.core.x1), int(cell.core.y1),
        ]
        cell_ctx[cell.cell_id] = (cell, patch, mem)

    total_memberships = sum(c["n_patch"] for c in cells.values())
    return {
        "engine": {
            "sip_mode": "keep",
            "reuse": [
                "zemosaic.zemosaic_zegrid_mode._build_frame_descriptors",
                "zemosaic.core.zegrid.geometry.build_canvas",
                "zemosaic.core.zegrid.auto_layout.project_footprints",
                "zemosaic.core.zegrid.sweep.build_cell_contexts_from_footprints",
            ],
        },
        "canvas": {
            "width": int(canvas.width),
            "height": int(canvas.height),
            "id": str(canvas.canvas_id),
            "resolution_deg": _fin(canvas.resolution_deg),
        },
        "layout": {"nx": 6, "ny": 4, "halo_px": 8, "cells": 24},
        "median_projected_footprint_wh": [_fin(footprints.median_w), _fin(footprints.median_h)],
        "total_patch_memberships": int(total_memberships),
        "cells": {k: cells[k] for k in sorted(cells)},
        "frames": [d.frame_id.logical_path for d in descs],
        "_cell_ctx": cell_ctx,
        "_descs": descs,
        "_canvas": canvas,
    }


# ---------------------------------------------------------------------------
# 4. Panel freeze (BIASED DEV; NOT QUALIFICATION)
# ---------------------------------------------------------------------------

def _relation_orientation(u: str, v: str) -> str:
    ur, uc = int(u[1:5]), int(u[6:10])
    vr, vc = int(v[1:5]), int(v[6:10])
    return "h" if ur == vr else "v"


def _canvas_corner_cells(nx: int, ny: int) -> set[str]:
    return {
        f"r{0:04d}c{0:04d}", f"r{0:04d}c{nx - 1:04d}",
        f"r{ny - 1:04d}c{0:04d}", f"r{ny - 1:04d}c{nx - 1:04d}",
    }


def relation_burdens(artifact: Mapping) -> dict:
    """Per-relation seam burden = max over channels |median pair difference|.

    A transparent, deliberately BIASED level-step proxy computed ONLY from the
    accepted capture artifact, used to freeze the DEV panel. It is NOT a
    qualification statistic.
    """
    per: dict[tuple, dict] = {}
    for rel in artifact.get("relations", []):
        u, v, ch = str(rel["u"]), str(rel["v"]), int(rel["channel"])
        key = (u, v)
        rec = per.setdefault(key, {"orientation": str(rel.get("orientation", _relation_orientation(u, v))),
                                   "channels": {}})
        rec["channels"][ch] = {
            "local_offset_train": _fin(rel.get("local_offset_train")),
            "robust_sigma": _fin(rel.get("robust_sigma")),
            "n_train": int(rel.get("n_train", 0)),
            "rejection_fraction": _fin(rel.get("rejection_fraction")),
            "level_mean": _fin(rel.get("level_mean")),
        }
    for key, rec in per.items():
        vals = [abs(c["local_offset_train"]) for c in rec["channels"].values()
                if c["local_offset_train"] is not None]
        rec["burden_adu"] = max(vals) if vals else None
    return per


def freeze_panel(artifact: Mapping, cells: Mapping[str, Mapping], nx: int, ny: int) -> dict:
    """Freeze the deterministic six-relation DEV panel (BEFORE factor results)."""
    per = relation_burdens(artifact)
    keys = sorted(per)
    for k in keys:
        per[k]["orientation"] = _relation_orientation(k[0], k[1])
    burden = {k: (per[k]["burden_adu"] if per[k]["burden_adu"] is not None else -1.0) for k in keys}
    ranked = sorted(keys, key=lambda k: (-burden[k], k))
    vertical = [k for k in ranked if per[k]["orientation"] == "v"]
    corner_cells = _canvas_corner_cells(nx, ny)
    corner_rels = [k for k in ranked if k[0] in corner_cells or k[1] in corner_cells]
    med_burden = float(np.median([burden[k] for k in keys]))

    slot_pick: dict[str, tuple | None] = {
        "declared_worst_edge": DECLARED_WORST_EDGE if DECLARED_WORST_EDGE in per else None,
        "highest_burden_vertical": vertical[0] if vertical else None,
        "highest_burden_overall": ranked[0] if ranked else None,
        "median_burden": None,
        "minimum_burden": min(keys, key=lambda k: (burden[k], k)) if keys else None,
        "corner_relation": None,
    }
    # median-burden relation: closest to the median burden (deterministic tie-break)
    if keys:
        slot_pick["median_burden"] = min(keys, key=lambda k: (abs(burden[k] - med_burden), k))
    # corner relation: cheapest (fewest contributors) corner relation NOT already
    # used, so a canvas corner gets an explicit panel slot (deterministic)
    if corner_rels:
        def _cost(k):
            return (cells[k[0]]["n_patch"] + cells[k[1]]["n_patch"], k)
        free_corner = [k for k in corner_rels if
                       k not in (slot_pick["declared_worst_edge"],
                                 slot_pick["highest_burden_vertical"],
                                 slot_pick["highest_burden_overall"],
                                 slot_pick["median_burden"],
                                 slot_pick["minimum_burden"])]
        slot_pick["corner_relation"] = min(free_corner or corner_rels, key=_cost)

    chosen: list[dict] = []
    used: set[tuple] = set()
    for slot in PANEL_SLOTS:
        k = slot_pick.get(slot)
        if k is None or k in used:
            continue
        used.add(k)
        chosen.append({
            "slot": slot,
            "u": k[0], "v": k[1],
            "orientation": per[k]["orientation"],
            "burden_adu": _fin(burden[k]),
            "n_patch_cells": [int(cells[k[0]]["n_patch"]), int(cells[k[1]]["n_patch"])],
            "membership_cost": int(cells[k[0]]["n_patch"] + cells[k[1]]["n_patch"]),
            "channels": {str(c): per[k]["channels"][c] for c in sorted(per[k]["channels"])},
        })
    # fill any remaining slots from the ranked list (determinism preserved)
    for k in ranked:
        if len(chosen) >= len(PANEL_SLOTS):
            break
        if k in used:
            continue
        used.add(k)
        chosen.append({
            "slot": "fill", "u": k[0], "v": k[1],
            "orientation": per[k]["orientation"], "burden_adu": _fin(burden[k]),
            "n_patch_cells": [int(cells[k[0]]["n_patch"]), int(cells[k[1]]["n_patch"])],
            "membership_cost": int(cells[k[0]]["n_patch"] + cells[k[1]]["n_patch"]),
            "channels": {str(c): per[k]["channels"][c] for c in sorted(per[k]["channels"])},
        })

    orientations = sorted({p["orientation"] for p in chosen})
    corners_present = sorted({
        c for p in chosen for c in (p["u"], p["v"]) if c in corner_cells
    })
    return {
        "label": "NOT_QUALIFICATION",
        "label_reason": (
            "biased DEV panel: the selection deliberately favours the declared worst edge "
            "and the highest-burden relations; it cannot qualify any model and no metric "
            "computed on it is an independent validation"
        ),
        "selection_rule": {
            "burden_definition": "max over channels |local_offset_train| (ADU) from the accepted capture",
            "slots": list(PANEL_SLOTS),
            "declared_worst_edge": list(DECLARED_WORST_EDGE),
            "median_burden_adu": _fin(med_burden),
            "rule": (
                "declared_worst_edge; highest-burden vertical; highest-burden overall; "
                "median-burden; minimum-burden; cheapest corner relation; dedup in slot order, "
                "then fill from the ranked list"
            ),
        },
        "inputs": {
            "n_relations_in_capture": len(keys),
            "canvas_corner_cells": sorted(corner_cells),
        },
        "relations": chosen,
        "orientations_covered": orientations,
        "corner_cells_covered": corners_present,
        "frozen_before_factor_results": True,
    }


# ---------------------------------------------------------------------------
# 5. Bounded aligned-cache build + full-domain gauge facts
# ---------------------------------------------------------------------------

def _subset_fixed_by_ids(fixed, ids_all: Sequence[str], keep: Sequence[str]):
    """Subset a FixedNormalization to ``keep`` (order-preserving), FAITHFULLY.

    Returns ``(subset, remap)``. ``remap`` truthfully records what happened to the
    source gauge's recorded exclusions: every exclusion whose frame is inside the
    subset is REMAPPED to its new index and carried into the subset; exclusions of
    frames outside the subset are recorded as ``dropped_out_of_subset`` (they are
    metadata about frames the subset no longer contains). Nothing is silently
    discarded.
    """
    z = _lazy_zemosaic()
    from zemosaic.core.canonical_stacking import FrameExclusion
    from zemosaic.core.canonical_streaming import FixedNormalization

    gidx = {fid: i for i, fid in enumerate(ids_all)}
    idx = [gidx[f] for f in keep]
    pos = {orig: new for new, orig in enumerate(idx)}
    ref = int(fixed.reference_index)
    keep_ref = pos.get(ref, -1)
    if keep_ref < 0:
        # No silent re-anchor: refuse (the caller must supply a gauge that covers
        # the reference, e.g. the shared gauge anchored on side u).
        raise ProbeError("gauge subset does not contain its reference frame")
    kept_exc = []
    dropped = []
    for e in fixed.exclusions:
        ei = int(e.index)
        if ei in pos:
            kept_exc.append(FrameExclusion(index=pos[ei], stage=e.stage, reason=e.reason,
                                           detail=e.detail))
        else:
            dropped.append({"index": ei, "stage": str(e.stage), "reason": str(e.reason)})
    subset = FixedNormalization(
        reference_index=keep_ref,
        coefficients=np.asarray(fixed.coefficients)[idx],
        norm_active=np.asarray(fixed.norm_active)[idx],
        weights=np.asarray(fixed.weights)[idx],
        weight_active=np.asarray(fixed.weight_active)[idx],
        exclusions=tuple(kept_exc),
        frame_ids=tuple(keep),
    )
    remap = {
        "n_frames_in": int(len(ids_all)),
        "n_frames_out": int(len(keep)),
        "reference_kept": True,
        "exclusions_remapped": len(kept_exc),
        "exclusions_dropped_out_of_subset": dropped,
    }
    return subset, remap


def _binding_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "r38a_binding.json"


def source_root_identity(source_root: Path) -> str:
    """One-way, domain-separated identity of the NORMALIZED resolved source root.

    The raw path is never persisted. Two different roots collide here only if they
    resolve to the same absolute path; two roots with the same
    basename but different locations produce DIFFERENT identities.
    """
    try:
        norm = str(Path(source_root).expanduser().resolve())
    except Exception:
        norm = str(source_root)
    h = hashlib.sha256()
    h.update(b"zemosaic.zegrid.r38a.source_root.v1\n")
    h.update(norm.encode("utf-8"))
    return h.hexdigest()


def _stat_identity(path: str) -> tuple:
    """Deterministic stat fingerprint (size, mtime_ns, dev, ino where supported)."""
    try:
        st = os.stat(path)
        return (int(st.st_size), int(getattr(st, "st_mtime_ns", 0)),
                int(getattr(st, "st_dev", 0)), int(getattr(st, "st_ino", 0)))
    except OSError:
        return (-1, -1, -1, -1)


def compute_cache_binding(cell_id: str, ctx: Mapping, identity: Mapping,
                          source_root: Path, source_files_on_disk: int,
                          cfg_tokens: Mapping) -> dict:
    """Bounded, private-root-free binding (SCHEMA v2) describing WHICH run a cache
    belongs to. Rejects a stale cache from a different capture / source set /
    source TREE / WCS / layout / patch / config.

    PRIVACY: only ``source_root_identity_sha256`` (one-way) and the root BASENAME
    label are persisted — never the path.

    GUARANTEE (honest): the binding rejects a *different tree* (different resolved
    root) and ordinary IN-PLACE replacement/touch of a source frame (stat
    fingerprint + descriptor facts). It is NOT a cryptographic content-identity
    proof: the source bytes are not hashed (unbounded I/O would be needed), so a
    byte-identical-size/different-content substitution that also preserves
    size+mtime_ns+dev+ino could in principle collide (an adversarial case that
    requires reproducing inode identity, not the ordinary stale-tree class).
    """
    cell, patch, mem = ctx
    ids = list(mem.patch_ids)
    rows = []
    for fid in ids:
        d = _all_descs_by_id[fid]
        size, mtime_ns, dev, ino = _stat_identity(d.source_path)
        rows.append((fid, size, mtime_ns, dev, ino, d.header_sha256 or ""))
    frames_blob = "\n".join("\t".join(str(v) for v in row) for row in rows)
    return {
        "schema_version": 2,
        "cell_id": cell_id,
        "capture_id": identity["capture_id"],
        "artifact_sha256": identity["artifact_sha256"],
        "source_set_digest": identity["source_set_digest"],
        "n_source_frames": int(identity["n_source_frames"]),
        "source_root_label": Path(source_root).name,
        "source_root_identity_sha256": source_root_identity(source_root),
        "n_source_files_on_disk": int(source_files_on_disk),
        "canvas_id": identity["canvas_id"],
        "layout": {"nx": int(identity["nx"]), "ny": int(identity["ny"]),
                   "halo_px": int(identity["halo_px"])},
        "patch": [int(patch.patch.x0), int(patch.patch.y0), int(patch.patch.x1), int(patch.patch.y1)],
        "patch_hw": [int(v) for v in patch.patch_shape_hw],
        "config_tokens": {str(k): str(v) for k, v in sorted(cfg_tokens.items())},
        "n_frames": len(ids),
        "frame_ids_digest": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "descriptors_digest": hashlib.sha256(frames_blob.encode()).hexdigest(),
        "source_fingerprint": hashlib.sha256(frames_blob.encode()).hexdigest(),
        "stat_identity_included": True,
        "content_bytes_hashed": False,
        "guarantee": (
            "rejects a different resolved source root and in-place source "
            "replacement/touch (stat fingerprint + descriptor facts); NOT a "
            "cryptographic pixel-content proof. CAVEAT: mtime_ns granularity is "
            "filesystem-dependent, so a touch that does not change the recorded "
            "stat values cannot be detected"
        ),
    }


def read_cache_binding(cache_dir: Path) -> dict | None:
    p = _binding_path(cache_dir)
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh, parse_constant=_reject_constant)
    except Exception:
        return None


def build_cell_cache(cell_id: str, ctx: Mapping, source_paths_index: Mapping[str, Path],
                     scratch_dir: Path, *, max_frames: int, binding: Mapping,
                     deadline: float | None = None) -> dict:
    """Stream one Cell's FULL accepted patch domain to a scratch aligned cache.

    Reuses the production primitives verbatim (product decoder, source-ROI plan,
    ``slice_wcs``, ``reproject_cropped``, ``AlignedCacheBuilder``) so the aligned
    values are bit-identical to the production per-Cell cache. One frame resident.

    RESUMABLE and BOUND: a cache is reused ONLY when the accepted R6
    ``cache_is_complete`` predicate holds AND the persisted ``r38a_binding.json``
    equals the current binding (capture id, source-set digest, canvas/layout/patch,
    config tokens, per-frame source fingerprints). Otherwise it is rebuilt within
    the caller's caps. ``deadline`` is a SOFT wall-clock budget checked between
    frames (a single decode/reproject calls cannot be interrupted).
    """
    z = _lazy_zemosaic()
    zg, zfp, zz, zsw = z["zg"], z["zfp"], z["zz"], z["zsw"]
    cell, patch, mem = ctx
    ids = list(mem.patch_ids)
    _require(len(ids) <= max_frames,
             f"cell {cell_id}: {len(ids)} contributors exceed cap {max_frames}", BudgetError)
    cache_dir = scratch_dir / "cells" / cell_id
    if (zfp.cache_is_complete(cache_dir, ids)
            and read_cache_binding(cache_dir) == dict(binding)):
        return {
            "cell_id": cell_id,
            "cache_key": f"cells/{cell_id}",
            "n_contributors": len(ids),
            "reused": True,
            "binding_validated": True,
            "bytes": 0,
            "seconds": 0.0,
        }
    builder = zfp.AlignedCacheBuilder(cache_dir)
    t0 = time.perf_counter()
    bytes_written = 0
    for n, fid in enumerate(ids, 1):
        if deadline is not None and time.perf_counter() > deadline:
            raise BudgetError(
                f"soft wall-clock budget exhausted while building {cell_id} "
                f"(frame {n}/{len(ids)}); cache left incomplete and will be rebuilt"
            )
        desc = _all_descs_by_id[fid]
        plan = zg.plan_source_roi(desc, _canvas_ref[0], patch)
        _require(plan is not None, f"cell {cell_id}: no source ROI for a contributor", ResolutionError)
        sb = plan.source_bounds
        hwc = zz._decode_frame_hwc(desc, None)
        crop = hwc[sb.y0:sb.y1, sb.x0:sb.x1]
        crop_chw = np.ascontiguousarray(np.moveaxis(crop, -1, 0))
        cropped_wcs = zz.zxe.slice_wcs(desc.wcs(), sb)
        rgb, geom = zz.zxe.reproject_cropped(
            crop_chw, cropped_wcs, patch.patch_wcs(), patch.patch_shape_hw
        )
        builder.add(fid, rgb, geom)
        del hwc, crop, crop_chw, rgb, geom
    manifest = builder.finish()
    bytes_written = int(manifest["total_bytes"])
    _binding_path(cache_dir).write_text(dumps_strict(dict(binding)), encoding="utf-8")
    return {
        "cell_id": cell_id,
        "cache_key": f"cells/{cell_id}",
        "n_contributors": len(ids),
        "reused": False,
        "binding_validated": True,
        "bytes": bytes_written,
        "seconds": round(time.perf_counter() - t0, 3),
    }


_all_descs_by_id: dict[str, object] = {}
_canvas_ref: list = []


def probe_cell_gauge(cell_id: str, cache_dir: Path, cfg) -> dict:
    """Reproduce ONE Cell's local gauge facts on its FULL accepted domain."""
    z = _lazy_zemosaic()
    zfp, zphot, zstream = z["zfp"], z["zphot"], z["zstream"]
    provider = zfp.MemmapCanonicalProvider(cache_dir)
    try:
        request = zstream.build_streaming_request(cfg, provider.n_frames)
        cres = zphot.compute_cell_gauge(provider, request)
        ids = list(provider.frame_ids)
    finally:
        provider.close()
    if cres.status != zphot.CELL_GAUGE_STATUS_OK:
        return {"cell_id": cell_id, "status": cres.status, "detail": cres.detail,
                "n_contributors": len(ids)}
    g = cres.gauge
    coef = np.asarray(g.coefficients)
    weights = np.asarray(g.weights)
    active = np.asarray(g.weight_active, dtype=bool)
    norm_active = np.asarray(g.norm_active, dtype=bool)
    a_vals = sorted({round(float(v), 12) for v in coef[active, :, 0].ravel()})
    exclusions = [
        {"index": int(e.index), "frame_id": ids[int(e.index)], "stage": str(e.stage),
         "reason": str(e.reason), "detail": (None if e.detail is None else str(e.detail))}
        for e in g.exclusions
    ]
    return {
        "cell_id": cell_id,
        "status": "ok",
        "n_contributors": len(ids),
        "reference_index": int(g.reference_index),
        "reference_frame_id": ids[int(g.reference_index)],
        "active_frames": int(np.count_nonzero(active)),
        "norm_active_frames": int(np.count_nonzero(norm_active)),
        "excluded": exclusions,
        "n_excluded": len(exclusions),
        "weights_summary": _vec_summary(weights[active]),
        "coefficients_a_values": a_vals,
        "coefficients": coef.tolist(),
        "weights": weights.tolist(),
        "norm_active": norm_active.astype(bool).tolist(),
        "weight_active": active.astype(bool).tolist(),
        "frame_ids": ids,
    }


def _vec_summary(v) -> dict:
    a = np.asarray(v, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "min": _fin(a.min()), "p10": _fin(np.percentile(a, 10)),
        "median": _fin(np.median(a)), "p90": _fin(np.percentile(a, 90)),
        "max": _fin(a.max()),
    }


def compare_gauges(gauge_u: Mapping, gauge_v: Mapping, observed: Mapping) -> dict:
    """Cross-side comparison of two Cells' local gauges on common active frames.

    For ``sky_mean`` the applied transform is ``a=1``, ``b`` additive, so for the
    SAME frame at the SAME pixel the two sides differ by exactly ``b_u - b_v``.
    The median of that difference over common active frames is the gauge-induced
    per-Cell level step; its MAD measures how frame/mask dependent it is.
    """
    if gauge_u.get("status") != "ok" or gauge_v.get("status") != "ok":
        return {"comparable": False, "reason": "gauge_not_ok"}
    ids_u, ids_v = list(gauge_u["frame_ids"]), list(gauge_v["frame_ids"])
    common = [f for f in ids_u if f in set(ids_v) and
              gauge_u["weight_active"][ids_u.index(f)] and gauge_v["weight_active"][ids_v.index(f)]]
    au = sorted(gauge_u["coefficients_a_values"])
    av = sorted(gauge_v["coefficients_a_values"])
    comparable = (au == [1.0] and av == [1.0])
    out = {
        "comparable": bool(comparable),
        "reason": None if comparable else "non_additive_coefficients",
        "u": gauge_u["cell_id"], "v": gauge_v["cell_id"],
        "n_common_active_frames": len(common),
        "reference_u": gauge_u["reference_frame_id"],
        "reference_v": gauge_v["reference_frame_id"],
        "references_agree": gauge_u["reference_frame_id"] == gauge_v["reference_frame_id"],
        "frame_set_asymmetry": {
            "active_only_u": sorted(set(gauge_u["frame_ids"]) - set(gauge_v["frame_ids"])),
            "active_only_v": sorted(set(gauge_v["frame_ids"]) - set(gauge_u["frame_ids"])),
            "excluded_u": sorted(e["frame_id"] for e in gauge_u["excluded"]),
            "excluded_v": sorted(e["frame_id"] for e in gauge_v["excluded"]),
        },
        "per_channel": {},
    }
    if not comparable or not common:
        return out
    cu = np.asarray(gauge_u["coefficients"], dtype=np.float64)
    cv = np.asarray(gauge_v["coefficients"], dtype=np.float64)
    iu = {f: i for i, f in enumerate(ids_u)}
    iv = {f: i for i, f in enumerate(ids_v)}
    for ch in range(cu.shape[1]):
        d = []
        obs_vals = []
        for f in common:
            d.append(float(cu[iu[f], ch, 1] - cv[iv[f], ch, 1]))
            ob = observed.get(str(ch), {}).get("local_offset_train")
            if ob is not None:
                obs_vals.append(float(ob))
        d = np.asarray(d, dtype=np.float64)
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med)))
        obs = float(np.median(obs_vals)) if obs_vals else None
        ratio = None
        if obs is not None and abs(obs) > 1e-12:
            ratio = med / obs
        out["per_channel"][str(ch)] = {
            "n": int(d.size),
            "gauge_level_step_median_adu": _fin(med),
            "gauge_level_step_mad_adu": _fin(mad),
            "gauge_level_step_p10_adu": _fin(np.percentile(d, 10)),
            "gauge_level_step_p90_adu": _fin(np.percentile(d, 90)),
            "consistency_mad_over_abs_median": _fin(mad / abs(med)) if abs(med) > 1e-12 else None,
            "observed_pair_diff_adu": _fin(obs),
            "gauge_step_over_observed": _fin(ratio),
        }
    return out


# ---------------------------------------------------------------------------
# 6. Bounded factor probe (A/B/C/D) on a shared-pixel ROI
# ---------------------------------------------------------------------------

class _RoiProvider:
    """Read-only view of one Cell's aligned cache restricted to a patch-local ROI.

    Serves the canonical streaming executor's ``get_raw_frame`` / ``get_tile``
    contract for a sub-window, so the executor can restack a BOUNDED ROI with a
    fixed gauge without ever materialising N x patch.
    """

    def __init__(self, cache_dir: Path, frame_ids: Sequence[str], roi: tuple[int, int, int, int]):
        z = _lazy_zemosaic()
        zfp = z["zfp"]
        cache_dir = Path(cache_dir)
        man = zfp.load_cache_manifest(cache_dir)
        all_ids = list(man["frame_ids"])
        idx = {fid: i for i, fid in enumerate(all_ids)}
        self._cache = cache_dir
        self._man = man
        self._sel = [idx[f] for f in frame_ids]
        self.frame_ids = tuple(frame_ids)
        y0, y1, x0, x1 = (int(v) for v in roi)
        self._roi = (y0, y1, x0, x1)
        self.n_frames = len(self._sel)
        self.height = int(y1 - y0)
        self.width = int(x1 - x0)
        self.channels = int(man["channels"])
        self.original_mono = bool(man["original_mono"])
        self.original_ndim = int(man["original_ndim"])
        self.original_shape = tuple(man["original_shape"])
        self._open: dict[int, tuple] = {}

    def _frame(self, k: int):
        if k not in self._open:
            ent = self._man["frames"][k]
            rgb = np.load(self._cache / ent["rgb"], mmap_mode="r")
            sup = np.load(self._cache / ent["support"], mmap_mode="r")
            self._open[k] = (rgb, sup)
        return self._open[k]

    def close(self):
        for rgb, sup in self._open.values():
            for arr in (rgb, sup):
                try:
                    arr._mmap.close()
                except Exception:
                    pass
        self._open = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def shape(self):
        return (self.height, self.width)

    def get_raw_frame(self, i: int):
        k = self._sel[i]
        rgb, sup = self._frame(k)
        y0, y1, x0, x1 = self._roi
        return np.asarray(rgb[y0:y1, x0:x1]), np.asarray(sup[y0:y1, x0:x1])

    def get_tile(self, i: int, y0: int, y1: int, x0: int, x1: int):
        k = self._sel[i]
        rgb, sup = self._frame(k)
        ry0, _ry1, rx0, _rx1 = self._roi
        tile = np.asarray(rgb[ry0 + y0:ry0 + y1, rx0 + x0:rx0 + x1])
        sup_tile = np.asarray(sup[ry0 + y0:ry0 + y1, rx0 + x0:rx0 + x1])
        from zemosaic.core.canonical_stacking import prepare_canonical_inputs
        batch = prepare_canonical_inputs([tile], [sup_tile])
        return batch.images[0], batch.valid_mask[0]


def _restack_roi(cache_dir: Path, frame_ids: Sequence[str], roi, fixed, cfg, rejection: str):
    z = _lazy_zemosaic()
    from zemosaic.core.canonical_streaming import run_canonical_stack_streaming
    zstream = z["zstream"]
    prov = _RoiProvider(cache_dir, frame_ids, roi)
    try:
        req = zstream.build_streaming_request(cfg, prov.n_frames)
        req = _replace_request(req, rejection=rejection)
        res = run_canonical_stack_streaming(prov, req, tile_size=None, fixed=fixed)
        # contributors BEFORE rejection: frames with any geometric support in the ROI
        pre = 0
        sups = []
        for i in range(prov.n_frames):
            _raw, sup = prov.get_raw_frame(i)
            sups.append(np.asarray(sup, dtype=bool))
            if bool(np.any(sup)):
                pre += 1
        valid = np.asarray(res.valid_mask, dtype=bool)
        if valid.ndim == 2:
            valid = valid[:, :, None]
        # contributors AFTER rejection: frames with support that retain at least one
        # non-rejected supported sample (the acceptance criterion: a frame that is
        # present AND not fully rejected for this pixel block)
        post = 0
        rm = getattr(res, "rejection_mask", None)
        if rm is not None:
            rm = np.asarray(rm, dtype=bool)
            if rm.ndim == 4:
                for i in range(prov.n_frames):
                    if not sups[i].any():
                        continue
                    if bool(np.any(sups[i][:, :, None] & ~rm[i])):
                        post += 1
        return {
            "science": np.asarray(res.science, dtype=np.float64),
            "valid": np.asarray(res.valid_mask, dtype=bool),
            "rejected_fraction": _fin(res.rejected_fraction),
            "rejected_sample_count": int(res.rejected_sample_count),
            "n_frames": int(prov.n_frames),
            "contributors_pre_rejection": int(pre),
            "contributors_post_rejection": int(post),
            "contributors_definition": (
                "pre = frames with any geometric support in the ROI; "
                "post = frames with support retaining >=1 non-rejected supported sample"
            ),
        }
    finally:
        prov.close()


def _replace_request(req, *, rejection: str):
    from zemosaic.core.canonical_engine import CanonicalStackRequest
    return CanonicalStackRequest(
        images=req.images, geometric_support=req.geometric_support,
        normalization=req.normalization, weighting=req.weighting,
        rejection=rejection, combine=req.combine, reference_index=req.reference_index,
        taper=req.taper, taper_px=req.taper_px, taper_floor=req.taper_floor,
        sigma_low=req.sigma_low, sigma_high=req.sigma_high, max_iters=req.max_iters,
        winsor_limit_low=req.winsor_limit_low, winsor_limit_high=req.winsor_limit_high,
        backend=req.backend, equalize_rgb=req.equalize_rgb,
    )


def _roi_metrics(du: Mapping, dv: Mapping) -> dict:
    """Robust, bounded metrics of a same-global-pixel pair difference."""
    su, sv = np.asarray(du["science"], dtype=np.float64), np.asarray(dv["science"], dtype=np.float64)
    if su.ndim == 2:
        su = su[:, :, None]
        sv = sv[:, :, None]
    vu, vv = np.asarray(du["valid"]), np.asarray(dv["valid"])
    # the canonical result's valid_mask is per-channel (H, W, C) for RGB: reduce it
    # to the shared 2-D support before intersecting (never silently ignore channels).
    if vu.ndim == 3:
        vu = vu.all(axis=-1)
    if vv.ndim == 3:
        vv = vv.all(axis=-1)
    vmask = vu & vv
    mask = vmask & np.isfinite(su).all(axis=-1) & np.isfinite(sv).all(axis=-1)
    h, w = mask.shape
    if not mask.any():
        return {"n": 0, "reason": "no_shared_valid_pixels"}
    out: dict = {"n": int(mask.sum()), "roi_hw": [int(h), int(w)]}
    ys, xs = np.nonzero(mask)
    gx = xs.astype(np.float64) + 0.5
    gy = ys.astype(np.float64) + 0.5
    per_ch = {}
    for ch in range(su.shape[2]):
        d = (su[:, :, ch] - sv[:, :, ch])[mask]
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med)))
        tangent = gx if w >= h else gy
        normal = gy if w >= h else gx
        per_ch[str(ch)] = {
            "median_adu": _fin(med),
            "rms_adu": _fin(math.sqrt(float(np.mean(d * d)))),
            "p90_abs_adu": _fin(np.percentile(np.abs(d), 90)),
            "mad_adu": _fin(mad),
            "tangent_slope_adu_per_px": _fin(_slope(tangent, d)),
            "normal_slope_adu_per_px": _fin(_slope(normal, d)),
            "level_corr": _fin(_corr(0.5 * (su[:, :, ch] + sv[:, :, ch])[mask], d)),
            "block_median_spread_adu": _fin(_block_spread(gx, gy, d)),
        }
    out["per_channel"] = per_ch
    out["rejected_fraction_u"] = du.get("rejected_fraction")
    out["rejected_fraction_v"] = dv.get("rejected_fraction")
    out["contributors_pre_rejection"] = [du.get("contributors_pre_rejection"),
                                         dv.get("contributors_pre_rejection")]
    out["contributors_post_rejection"] = [du.get("contributors_post_rejection"),
                                          dv.get("contributors_post_rejection")]
    out["contributors_definition"] = du.get("contributors_definition")
    return out


def _slope(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return None
    xc = x - x.mean()
    den = float(np.sum(xc * xc))
    if den <= 0:
        return None
    return float(np.sum(xc * (y - y.mean())) / den)


def _corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return None
    xc = x - x.mean()
    yc = y - y.mean()
    den = math.sqrt(float(np.sum(xc * xc)) * float(np.sum(yc * yc)))
    if den <= 0:
        return None
    return float(np.sum(xc * yc) / den)


def _block_spread(gx, gy, d, block=32):
    bx = (gx // block).astype(np.int64)
    by = (gy // block).astype(np.int64)
    key = bx * 100000 + by
    meds = []
    for k in np.unique(key):
        vals = d[key == k]
        if vals.size >= 8:
            meds.append(float(np.median(vals)))
    if len(meds) < 2:
        return None
    return float(np.percentile(meds, 90) - np.percentile(meds, 10))


def _cache_dir_of(entry):
    """Accept a real scratch Path OR a ``{"cache_dir": ...}`` mapping (tests)."""
    if isinstance(entry, Mapping):
        return entry.get("cache_dir")
    return entry


def _uniform_weights_only(shared):
    """Return a FixedNormalization identical to ``shared`` EXCEPT the numeric
    weights of the active frames, which become uniform 1.0.

    Everything else is preserved byte-for-byte: coefficients, reference index,
    frame order, ``norm_active``, ``weight_active`` membership and ``exclusions``.
    This is what makes the ``W`` factor step genuinely weights-only.
    """
    from zemosaic.core.canonical_streaming import FixedNormalization

    w = np.asarray(shared.weights, dtype=np.float64).copy()
    active = np.asarray(shared.weight_active, dtype=bool)
    w[active] = 1.0
    return FixedNormalization(
        reference_index=int(shared.reference_index),
        coefficients=np.asarray(shared.coefficients, dtype=np.float64).copy(),
        norm_active=np.asarray(shared.norm_active, dtype=bool).copy(),
        weights=w,
        weight_active=active.copy(),
        exclusions=tuple(shared.exclusions),
        frame_ids=(None if shared.frame_ids is None else tuple(shared.frame_ids)),
    )


def _factor_contract() -> dict:
    """The frozen factor-ladder contract (single source of truth for docs/tests)."""
    return {
        "baseline": "BASE: own frames + own local gauge + own weights + declared rejection",
        "A": "BASE -> FRAME SET only (common intersection; each side keeps its OWN gauge, faithfully subset)",
        "G": "A -> GAUGE only (both sides use side u's gauge over the common frames)",
        "W": "G -> WEIGHTS only (identical coefficients/reference/order/active masks/exclusions; only active weights -> 1.0)",
        "D": "G -> REJECTION only (none; DIAGNOSTIC ONLY, never a correction)",
        "B0": "CONTROL (multi-variable): identity gauge a=1,b=0 AND uniform weights; NOT part of the one-variable ladder",
        "chain": "BASE -(frame set)-> A -(gauge)-> G -(weights)-> W ; G -(rejection)-> D",
        "single_variable": {
            "A": "frame_set only", "G": "gauge only", "W": "weights only", "D": "rejection only",
        },
        "backend": "cpu (fixed)",
        "non_additivity": "the frame set, gauge and weights interact: per-setting metrics are reported, never summed into percentages",
        "tautology": {
            "G": (
                "both sides recombine the SAME common frames with the SAME gauge over the SAME global "
                "ROI, so a near-zero difference is an EXPECTED IDENTITY sanity check, NOT independent "
                "causal evidence"
            ),
        },
    }


def run_factor_probe(relation, gauges, caches, roi_cfg, cfg) -> dict:
    """One-variable-at-a-time factor evidence for ONE probed relation.

    Chain (each step changes exactly ONE variable from its predecessor):
      BASE  own frames + own local gauge + own weights + declared rejection
      A     FRAME SET only  -> common intersection, each side keeps its OWN gauge
      G     GAUGE only      -> both sides use side u's gauge over the common frames
      W     WEIGHTS only    -> coefficients kept, weights set uniform
      D     REJECTION only  -> none (DIAGNOSTIC ONLY)
      B0    control         -> identity gauge (a=1,b=0) + uniform weights
    """
    u, v = relation["u"], relation["v"]
    gu, gv = gauges.get(u), gauges.get(v)
    if not gu or not gv or gu.get("status") != "ok" or gv.get("status") != "ok":
        return {"status": "NOT_RUN", "reason": "gauge_facts_unavailable"}
    cu, cv = caches.get(u), caches.get(v)
    if not cu or not cv:
        return {"status": "NOT_RUN", "reason": "cell_cache_unavailable"}
    du_dir, dv_dir = _cache_dir_of(cu), _cache_dir_of(cv)
    if du_dir is None or dv_dir is None:
        return {"status": "NOT_RUN", "reason": "cell_cache_unavailable"}
    cap = int(roi_cfg["max_cell_contributors"])
    if gu["n_contributors"] > cap or gv["n_contributors"] > cap:
        return {"status": "NOT_RUN",
                "reason": f"contributor_cap_exceeded ({gu['n_contributors']},{gv['n_contributors']} > {cap})"}

    # Patch-local ROI per side relative to the shared GLOBAL band.
    cell_u, patch_u, _memu = _ctx_ref[u]
    cell_v, patch_v, _memv = _ctx_ref[v]
    y0 = max(patch_u.patch.y0, patch_v.patch.y0)
    y1 = min(patch_u.patch.y1, patch_v.patch.y1)
    x0 = max(patch_u.patch.x0, patch_v.patch.x0)
    x1 = min(patch_u.patch.x1, patch_v.patch.x1)
    if y1 <= y0 or x1 <= x0:
        return {"status": "NOT_RUN", "reason": "no_patch_overlap"}
    span = int(roi_cfg["max_tangent_px"])
    if relation["orientation"] == "h":
        boundary = min(cell_u.core.x1, cell_v.core.x1)  # shared vertical edge x
        bx0 = max(x0, boundary - int(roi_cfg["band_px"]))
        bx1 = min(x1, boundary + int(roi_cfg["band_px"]))
        cy = (y0 + y1) // 2
        by0 = max(y0, cy - span // 2)
        by1 = min(y1, by0 + span)
        roi = (by0, by1, bx0, bx1)
    else:
        boundary = min(cell_u.core.y1, cell_v.core.y1)
        by0 = max(y0, boundary - int(roi_cfg["band_px"]))
        by1 = min(y1, boundary + int(roi_cfg["band_px"]))
        cx = (x0 + x1) // 2
        bx0 = max(x0, cx - span // 2)
        bx1 = min(x1, bx0 + span)
        roi = (by0, by1, bx0, bx1)
    roi_u = (roi[0] - patch_u.patch.y0, roi[1] - patch_u.patch.y0,
             roi[2] - patch_u.patch.x0, roi[3] - patch_u.patch.x0)
    roi_v = (roi[0] - patch_v.patch.y0, roi[1] - patch_v.patch.y0,
             roi[2] - patch_v.patch.x0, roi[3] - patch_v.patch.x0)

    ids_u, ids_v = list(gu["frame_ids"]), list(gv["frame_ids"])
    common = [f for f in ids_u if f in set(ids_v) and gu["weight_active"][ids_u.index(f)]
              and gv["weight_active"][ids_v.index(f)]]
    if not common or gu["reference_frame_id"] not in common:
        return {"status": "NOT_RUN", "reason": "no_common_active_frames_covering_reference_u"}

    from zemosaic.core.canonical_stacking import FrameExclusion
    from zemosaic.core.canonical_streaming import FixedNormalization

    def _full(gauge, ids):
        """Faithful FixedNormalization from the recorded gauge facts (incl. exclusions)."""
        return FixedNormalization(
            reference_index=ids.index(gauge["reference_frame_id"]),
            coefficients=np.asarray(gauge["coefficients"], dtype=np.float64),
            norm_active=np.asarray(gauge["norm_active"], dtype=bool),
            weights=np.asarray(gauge["weights"], dtype=np.float64),
            weight_active=np.asarray(gauge["weight_active"], dtype=bool),
            exclusions=tuple(
                FrameExclusion(
                    index=int(e["index"]), stage=str(e["stage"]), reason=str(e["reason"]),
                    detail=(None if e.get("detail") is None else str(e["detail"])),
                )
                for e in gauge.get("excluded", [])
            ),
            frame_ids=tuple(ids),
        )

    full_u = _full(gu, ids_u)
    full_v = _full(gv, ids_v)
    shared, remap_u = _subset_fixed_by_ids(full_u, ids_u, common)
    own_u = shared
    own_v = None
    remap_v = None
    try:
        own_v, remap_v = _subset_fixed_by_ids(full_v, ids_v, common)
    except ProbeError:
        own_v = None

    # W is GENUINELY weights-only relative to G: identical coefficients,
    # reference, frame order, norm_active, weight_active AND exclusions; ONLY the
    # numeric weights of the active frames change (uniform 1.0, inactive kept).
    _w = np.asarray(shared.weights, dtype=np.float64).copy()
    _wa = np.asarray(shared.weight_active, dtype=bool)
    _w[_wa] = 1.0
    uniform = _uniform_weights_only(shared)
    identity = FixedNormalization(
        reference_index=0,
        coefficients=np.zeros((len(common), 3, 2), dtype=np.float64) + np.array([1.0, 0.0]),
        norm_active=np.ones(len(common), dtype=bool),
        weights=np.ones(len(common), dtype=np.float64),
        weight_active=np.ones(len(common), dtype=bool),
        exclusions=(), frame_ids=tuple(common),
    )

    results: dict = {}

    def _run(setting):
        if setting == "BASE":  # own frames + own gauge + own weights + declared rejection
            du = _restack_roi(du_dir, ids_u, roi_u, full_u, cfg, cfg.rejection)
            dv = _restack_roi(dv_dir, ids_v, roi_v, full_v, cfg, cfg.rejection)
        elif setting == "A":  # variable: FRAME SET -> common intersection, own gauges kept
            if own_v is None:
                raise ProbeError("side-v reference not in the common frame set")
            du = _restack_roi(du_dir, common, roi_u, own_u, cfg, cfg.rejection)
            dv = _restack_roi(dv_dir, common, roi_v, own_v, cfg, cfg.rejection)
        elif setting == "G":  # variable: GAUGE -> side v aligned to side u's gauge
            du = _restack_roi(du_dir, common, roi_u, shared, cfg, cfg.rejection)
            dv = _restack_roi(dv_dir, common, roi_v, shared, cfg, cfg.rejection)
        elif setting == "W":  # variable: WEIGHTS -> uniform active weights (everything else kept)
            du = _restack_roi(du_dir, common, roi_u, uniform, cfg, cfg.rejection)
            dv = _restack_roi(dv_dir, common, roi_v, uniform, cfg, cfg.rejection)
        elif setting == "D":  # variable: REJECTION -> none (DIAGNOSTIC ONLY)
            du = _restack_roi(du_dir, common, roi_u, shared, cfg, "none")
            dv = _restack_roi(dv_dir, common, roi_v, shared, cfg, "none")
        elif setting == "B0":  # CONTROL: identity gauge AND uniform weights (multi-variable)
            du = _restack_roi(du_dir, common, roi_u, identity, cfg, cfg.rejection)
            dv = _restack_roi(dv_dir, common, roi_v, identity, cfg, cfg.rejection)
        else:
            raise ProbeError(setting)
        return du, dv

    variables = {
        "BASE": ["baseline"],
        "A": ["frame_set"],
        "G": ["gauge"],
        "W": ["weights"],
        "D": ["rejection"],
        "B0": ["gauge_choice", "weights"],
    }
    for setting in ("BASE", "A", "G", "W", "D", "B0"):
        try:
            du, dv = _run(setting)
            results[setting] = {"status": "RUN", "variables_changed": variables[setting],
                                "metrics": _roi_metrics(du, dv),
                                "frames_u": du["n_frames"], "frames_v": dv["n_frames"]}
        except _not_run_exceptions() as exc:  # expected domain/budget refusal
            results[setting] = {"status": "NOT_RUN", "variables_changed": variables[setting],
                                "reason": f"{type(exc).__name__}: {exc}"}

    return {
        "status": "RUN",
        "contract": _factor_contract(),
        "gauge_reconstruction": {
            "u": remap_u,
            "v": remap_v,
            "reference_kept_u": True,
            "reference_kept_v": (own_v is not None),
        },
        "limitations": [
            "the ROI is restacked with the accepted streaming executor on a bounded sub-window; "
            "the footprint taper is therefore computed within the ROI (edge effect) while the "
            "gauge coefficients/weights come from the FULL accepted domain",
            "interactions are reported per setting and never summed into an additive attribution",
        ],
        "roi_global": [int(v) for v in roi],
        "common_frames": len(common),
        "settings": results,
    }


_ctx_ref: dict[str, tuple] = {}


# ---------------------------------------------------------------------------
# 7. Conclusion (pre-registered decision rule)
# ---------------------------------------------------------------------------

MIN_QUALIFYING_RELATIONS = 2   # PARTIAL_CAUSE needs >= 2 distinct QUALIFYING relations
REQUIRED_ORIENTATIONS = ("h", "v")  # ... and BOTH orientations


def conclude(gauge_results: Sequence[Mapping], factor_results: Sequence[Mapping], *,
             executed_relations: Sequence[Mapping] = (), panel_total: int | None = None) -> dict:
    """Apply the frozen RELATION-LEVEL decision rule; never overclaim.

    Channels are aggregated INSIDE each relation first: a relation QUALIFIES when
    at least one of its channels has a stable gauge step explaining >= ``partial_frac``
    of the observed pair difference (and is not negligible). ``PARTIAL_CAUSE``
    requires >= ``MIN_QUALIFYING_RELATIONS`` distinct qualifying relations AND both
    orientations among them; one relation's correlated channels can never satisfy
    it. ``DOMINANT_CAUSE_CONFIRMED`` is NOT reachable in this bounded A harness (no
    restack-dominance criterion exists), by construction.
    """
    per_relation: dict[str, dict] = {}
    for g in gauge_results:
        cmp = g.get("comparison") or g
        if not cmp.get("comparable"):
            continue
        u, v = str(cmp.get("u")), str(cmp.get("v"))
        orientation = None
        rel = cmp.get("relation")
        if isinstance(rel, Mapping):
            orientation = rel.get("orientation")
        if orientation not in REQUIRED_ORIENTATIONS and u.startswith("r") and v.startswith("r"):
            orientation = _relation_orientation(u, v)
        chans: dict[str, dict] = {}
        for ch, vals in sorted((cmp.get("per_channel") or {}).items()):
            ratio = vals.get("gauge_step_over_observed")
            cons = vals.get("consistency_mad_over_abs_median")
            obs = vals.get("observed_pair_diff_adu")
            if ratio is None or obs is None:
                chans[str(ch)] = {"ratio_abs": None, "consistency": cons,
                                  "qualifying": False, "negligible": False}
                continue
            r = abs(float(ratio))
            stable = (cons is None or float(cons) <= GAUGE_STEP_CONSISTENCY_MAX)
            chans[str(ch)] = {
                "ratio_abs": r, "consistency": cons,
                "qualifying": bool(stable and r >= GAUGE_STEP_PARTIAL_FRAC),
                "negligible": bool(r <= GAUGE_STEP_NEGLIGIBLE_FRAC),
            }
        key = f"{u}|{v}"
        per_relation[key] = {
            "u": u, "v": v, "orientation": orientation, "channels": chans,
            "qualifying": bool(any(c["qualifying"] for c in chans.values())),
            "all_negligible": bool(chans) and all(c["negligible"] for c in chans.values()),
        }

    n_comparable = len(per_relation)
    qualifying = sorted(k for k, d in per_relation.items() if d["qualifying"])
    qualifying_orientations = sorted({per_relation[k]["orientation"] for k in qualifying
                                      if per_relation[k]["orientation"] in REQUIRED_ORIENTATIONS})
    negligible_relations = sorted(k for k, d in per_relation.items() if d["all_negligible"])
    executed_orientations = sorted({d["orientation"] for d in per_relation.values()
                                   if d["orientation"] in REQUIRED_ORIENTATIONS})
    factors_executed = [f for f in factor_results if f.get("status") == "RUN"]

    if n_comparable < MIN_PROBED_RELATIONS:
        state = "INCONCLUSIVE"
        why = (f"only {n_comparable} comparable probed relation(s) < {MIN_PROBED_RELATIONS}; "
               "the state cannot be decided from a single relation")
    elif not per_relation:
        state, why = "INCONCLUSIVE", "no comparable gauge-step observation"
    elif len(negligible_relations) == n_comparable:
        state = "CAUSE_FALSIFIED"
        why = "the gauge-induced level step is negligible on every probed relation/channel"
    elif len(qualifying) >= MIN_QUALIFYING_RELATIONS and set(qualifying_orientations) == set(REQUIRED_ORIENTATIONS):
        state = "PARTIAL_CAUSE"
        why = (f"{len(qualifying)} distinct relations qualify (>= {GAUGE_STEP_PARTIAL_FRAC:.0%} of the "
               f"observed pair difference with a stable gauge step) across BOTH orientations "
               f"({', '.join(qualifying)}); no restack-dominance proof exists on this host")
        if factors_executed:
            why += "; factor probe executed (see factor_evidence), dominance not established"
    else:
        state = "INCONCLUSIVE"
        why = (f"{len(qualifying)} qualifying relation(s) / orientations "
               f"{qualifying_orientations or '[]'}; PARTIAL_CAUSE needs >= {MIN_QUALIFYING_RELATIONS} "
               "distinct relations covering both orientations")

    executed = [{"u": r.get("u"), "v": r.get("v"), "orientation": r.get("orientation")}
                for r in executed_relations]
    coverage = {
        "executed_relations": len(per_relation),
        "executed_relations_detail": executed,
        "panel_relations_total": int(panel_total) if panel_total is not None else None,
        "executed_fraction": _fin(len(per_relation) / panel_total) if panel_total else None,
        "executed_orientations": executed_orientations,
        "qualifying_relations": qualifying,
        "qualifying_orientations": qualifying_orientations,
        "negligible_relations": negligible_relations,
        "channel_summaries": {k: v["channels"] for k, v in sorted(per_relation.items())},
    }

    return {
        "state": state,
        "reason": why,
        "thresholds": {
            "dominance_frac": GAUGE_STEP_DOMINANCE_FRAC,
            "partial_frac": GAUGE_STEP_PARTIAL_FRAC,
            "negligible_frac": GAUGE_STEP_NEGLIGIBLE_FRAC,
            "consistency_max": GAUGE_STEP_CONSISTENCY_MAX,
            "min_probed_relations": MIN_PROBED_RELATIONS,
            "min_qualifying_relations": MIN_QUALIFYING_RELATIONS,
            "required_orientations": list(REQUIRED_ORIENTATIONS),
        },
        "aggregation": "channels aggregated INSIDE each relation, then relations aggregated",
        "n_probed_relations": n_comparable,
        "coverage": coverage,
        "dominant_cause_confirmed": False,
        "dominant_cause_available": False,
        "dominant_cause_note": (
            "DOMINANT_CAUSE_CONFIRMED is not reachable in this bounded R38-A harness: it would "
            "require a pre-registered restack-dominance criterion that does not exist here"
        ),
        "claim_limits": [
            "no correction is applied anywhere",
            "a small-band or single-worst-edge result alone is never treated as causal",
            "channels of the same relation are correlated and are never counted as separate relations",
            "factor interactions are reported, never summed into an additive attribution",
        ],
    }


# ---------------------------------------------------------------------------
# 8. Budgets + CLI
# ---------------------------------------------------------------------------

def estimate_costs(panel, cells, *, max_cell_contributors, max_frames_total) -> dict:
    """Bounded execution plan: orientation coverage first, then ascending cost."""
    order = sorted(panel["relations"], key=lambda r: (r["membership_cost"], r["u"], r["v"]))
    cells_needed: list[str] = []
    frames = 0
    selected: list[dict] = []
    selected_keys: set[tuple] = set()

    def _fits(rel):
        cu, cv = rel["u"], rel["v"]
        if cells[cu]["n_patch"] > max_cell_contributors or cells[cv]["n_patch"] > max_cell_contributors:
            return None
        add = [c for c in (cu, cv) if c not in cells_needed]
        add_frames = sum(cells[c]["n_patch"] for c in add)
        if frames + add_frames > max_frames_total:
            return None
        return add, add_frames

    def _take(rel):
        got = _fits(rel)
        if got is None:
            return False
        add, add_frames = got
        nonlocal frames
        cells_needed.extend(add)
        frames += add_frames
        selected.append({"u": rel["u"], "v": rel["v"], "orientation": rel["orientation"],
                         "membership_cost": rel["membership_cost"]})
        selected_keys.add((rel["u"], rel["v"]))
        return True

    # pass 1: guarantee one executed relation per orientation when feasible
    for want in ("h", "v"):
        if any(s["orientation"] == want for s in selected):
            continue
        for rel in order:
            if rel["orientation"] == want and _take(rel):
                break
    # pass 2: ascending cost within the remaining budget
    for rel in order:
        if (rel["u"], rel["v"]) in selected_keys:
            continue
        _take(rel)

    return {
        "execution_order_rule": (
            "pass 1 ensures one executed relation per orientation (cheapest first); "
            "pass 2 adds the remaining panel relations by ascending membership cost"
        ),
        "selected_relations": selected,
        "selected_cells": cells_needed,
        "estimated_frames": int(frames),
        "estimated_disk_mib": _fin(frames * 10.1),
        "estimated_seconds_at_2_9s_per_frame": _fin(frames * 2.9),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r38_causal_probe.py",
        description="ZM-ZEGRID-R38-A bounded read-only causal diagnostic harness",
    )
    p.add_argument("--capture-dir", required=True,
                   help="directory holding the exact R37 pre-feather capture + manifest (READ-ONLY)")
    p.add_argument("--source-dir", required=True, help="read-only source FITS directory")
    p.add_argument("--mode", choices=("inventory", "gauge", "factors"), default="inventory")
    p.add_argument("--out", default=None, help="strict-JSON report path")
    p.add_argument("--report-dir", default=str(REPORT_DIR_DEFAULT))
    p.add_argument("--scratch-dir", default=str(SCRATCH_DIR_DEFAULT))
    p.add_argument("--expect-identity", choices=("checked", "any"), default="checked")
    p.add_argument("--execute-real", action="store_true",
                   help="required for the real-data gauge/factor probe (never for inventory)")
    p.add_argument("--gauge-cache", default=None,
                   help="reuse an existing gauge-mode scratch dir instead of rebuilding")
    p.add_argument("--max-cell-contributors", type=int, default=220)
    p.add_argument("--max-frames-total", type=int, default=460)
    p.add_argument("--max-seconds", type=float, default=1200.0)
    p.add_argument("--factor-max-contributors", type=int, default=200)
    p.add_argument("--factor-band-px", type=int, default=8)
    p.add_argument("--factor-max-tangent-px", type=int, default=256)
    p.add_argument("--determinism-check", action="store_true")
    return p


def _write_report(out_path: Path, report: Mapping, policy: Mapping) -> int:
    """Atomic strict-JSON write under an IMMUTABLE validated write policy.

    Hardened against a check-then-write hole: the output path is re-validated
    (a) before creating the parent, (b) AFTER the parent exists, and (c) immediately
    before ``os.replace``. The temporary file is created with ``tempfile.mkstemp``
    next to the destination (``O_EXCL``, no following of an attacker-controlled
    ``.tmp`` symlink, 0600 perms) and is removed on any failure.
    """
    out_path = _assert_write_allowed(out_path, label="report file", policy=policy)
    parent = out_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise BudgetError("report parent is a symlink; refusing")
    out_path = _assert_write_allowed(out_path, label="report file", policy=policy)
    text = dumps_strict(report)
    fd, tmp_name = tempfile.mkstemp(prefix=out_path.name + ".", suffix=".tmp",
                                    dir=str(parent), text=True)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        # final re-validation immediately before the replace
        _assert_write_allowed(out_path, label="report file", policy=policy)
        if tmp.is_symlink() or out_path.is_symlink():
            raise BudgetError("report destination or temp is a symlink; refusing")
        os.replace(tmp, out_path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return len(text)


def _prepare_run(args):
    """Validate the RAW arguments (symlinks included) then resolve and prepare.

    L3r: the RAW expanded argument is checked for a symlink BEFORE ``.resolve()``,
    so a symlinked source/capture root is refused by the real CLI path (not only by
    the ``resolve_sources`` unit entry point). Output/scratch roots are still
    resolved and compared against the read-only trees by ``_assert_write_allowed``.
    """
    raw_capture = Path(args.capture_dir).expanduser()
    raw_source = Path(args.source_dir).expanduser()
    for label, raw in (("capture dir", raw_capture), ("source dir", raw_source)):
        if raw.is_symlink():
            raise ResolutionError(f"{label} is a symlink; refusing (fail-closed)")
    capture_dir = raw_capture.resolve()
    source_dir = raw_source.resolve()
    env_roots = [Path(p) for p in os.environ.get("R38A_FORBIDDEN_ROOTS", "").split(os.pathsep) if p.strip()]
    policy = build_write_policy(capture_dir=capture_dir, source_dir=source_dir,
                                expect=args.expect_identity, env_roots=env_roots)
    scratch = _assert_write_allowed(args.scratch_dir, label="scratch dir", policy=policy)
    report_dir = _assert_write_allowed(args.report_dir, label="report dir", policy=policy)
    out_path = Path(args.out).expanduser() if args.out else (report_dir / f"r38a_{args.mode}.json")
    _assert_write_allowed(out_path, label="report file", policy=policy)
    identity = validate_identity(capture_dir, expect=args.expect_identity)
    resolution = resolve_sources(source_dir, identity["frame_ids"])
    geometry = reconstruct_geometry(resolution["paths"], source_dir)
    return (capture_dir, source_dir, scratch, report_dir, out_path,
            identity, resolution, geometry, policy)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _quiet()
    t_start = time.perf_counter()
    deadline = t_start + float(args.max_seconds)

    try:
        (capture_dir, source_dir, scratch, report_dir, out_path,
         identity, resolution, geometry, policy) = _prepare_run(args)
    except _not_run_exceptions() as exc:
        # M1n (fail closed): preparation failed BEFORE output safety was
        # established. Write NOTHING anywhere (the configured output may live in a
        # refused tree), emit only a compact, path-free status, and return nonzero.
        print(json.dumps({"status": "FAILED", "mode": args.mode, "stage": "prepare",
                          "reason": type(exc).__name__,
                          "wrote_report": False}, sort_keys=True))
        return 1

    report: dict = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "mission": MISSION,
        "mode": args.mode,
        "status": "PARTIAL",
        "status_reason": "set at the end of the run",
        "qualification": "DEV_NOT_QUALIFICATION",
        "scientific_acceptance": False,
        "r38b_authorized": False,
        "write_policy": {
            "dataset_root_protection": policy["mode"],
            "dataset_root_identity_sha256": policy["dataset_root_identity_sha256"],
            "protected_roots_count": len(policy["protected_roots"]),
            "env_roots_count": policy["env_roots_count"],
            "note": (
                "checked runs derive the nearest common ancestor of capture and source as a "
                "MANDATORY protected root (never persisted); generic 'any' runs protect only "
                "capture/source plus R38A_FORBIDDEN_ROOTS and are labelled accordingly"
            ),
        },
        "evidence_scope": {
            "kind": "diagnostic_dev_bounded",
            "panel": "biased DEV panel frozen BEFORE results; NOT a qualification corpus",
            "conclusion_aggregation": "relation-level (channels aggregated inside a relation)",
            "r37_heldout_consumed": True,
            "correction_applied": False,
            "dominant_cause_available": False,
        },
        "identity": {k: v for k, v in identity.items() if k != "artifact"},
        "resolution": {"source_root_label": resolution["source_root_label"],
                       "n_resolved": resolution["n_resolved"],
                       "n_files_on_disk": resolution["n_files_on_disk"]},
        "geometry": {k: v for k, v in geometry.items() if not k.startswith("_")},
        "import_audit": import_audit(),
        "budgets": {
            "max_cell_contributors": int(args.max_cell_contributors),
            "max_frames_total": int(args.max_frames_total),
            "max_seconds": _fin(args.max_seconds),
            "max_seconds_semantics": (
                "SOFT wall-clock budget: checked between frames and between settings; a single "
                "canonical decode/reproject/restack call cannot be interrupted, so a run may "
                "overshoot by at most one such call"
            ),
            "factor_max_contributors": int(args.factor_max_contributors),
        },
        "not_run": [],
    }

    # global refs used by the (re)factor helpers
    _all_descs_by_id.clear()
    _canvas_ref.clear()
    _canvas_ref.append(geometry["_canvas"])
    for d in geometry["_descs"]:
        _all_descs_by_id[d.frame_id.logical_path] = d
    for cid, ctx in geometry["_cell_ctx"].items():
        _ctx_ref[cid] = ctx

    panel = freeze_panel(identity["artifact"], geometry["cells"], 6, 4)
    panel["inputs"]["artifact_sha256"] = identity["artifact"]["sha256"]
    report["panel"] = panel

    costs = estimate_costs(panel, geometry["cells"],
                           max_cell_contributors=args.max_cell_contributors,
                           max_frames_total=args.max_frames_total)
    report["probe_plan"] = costs
    report["not_run"].append({
        "item": "panel_relations_outside_execution_budget",
        "reason": "bounded host: only relations whose cells fit the per-cell/global caps are executed",
        "relations": [
            {"u": r["u"], "v": r["v"], "orientation": r["orientation"],
             "membership_cost": r["membership_cost"]}
            for r in panel["relations"]
            if not any(s["u"] == r["u"] and s["v"] == r["v"] for s in costs["selected_relations"])
        ],
    })

    comparisons = []
    executed = []
    factor_results: list = []
    unexpected: dict | None = None
    if args.mode in ("gauge", "factors"):
        if not args.execute_real:
            report["not_run"].append({"item": "real_gauge_probe",
                                      "reason": "--execute-real not given"})
        else:
            try:
                comparisons, executed, factor_results = _run_real_probe(
                    args, report, geometry, panel, costs, scratch, t_start, deadline
                )
            except _not_run_exceptions() as exc:  # expected domain/budget refusal
                report["not_run"].append({"item": "real_gauge_probe",
                                          "reason": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:  # programming error: FAIL the run, never hide it
                unexpected = {"stage": "real_probe", "type": type(exc).__name__,
                              "message": str(exc)}

    if unexpected is None and args.mode in ("gauge", "factors") and args.execute_real:
        report["conclusion"] = conclude(comparisons, factor_results,
                                        executed_relations=executed,
                                        panel_total=len(panel["relations"]))

    planned = len(costs["selected_relations"])
    done = len(executed)
    if unexpected is not None:
        report["status"] = "FAILED"
        report["status_reason"] = f"unexpected {unexpected['type']} during the real probe"
        report["failed"] = unexpected
    elif args.mode in ("gauge", "factors") and args.execute_real:
        if done < planned or done < len(panel["relations"]):
            report["status"] = "PARTIAL"
            report["status_reason"] = (
                f"executed {done}/{planned} planned relations and {done}/{len(panel['relations'])} "
                "panel relations; the remainder is NOT_RUN (bounded host)"
            )
        else:
            report["status"] = "PASS"
            report["status_reason"] = "every planned relation executed"
    else:
        report["status"] = "PASS"
        report["status_reason"] = "inventory-only run (no pixel probe requested)"
    report["evidence_scope"].update({
        "panel_relations_total": len(panel["relations"]),
        "relations_executed": done,
        "orientations_executed": sorted({r["orientation"] for r in executed}),
    })

    det: dict = {}
    if args.determinism_check:
        # Real repeat: rebuild the identity checks and the panel from the raw
        # artifact/capture independently and compare the canonical JSON.
        identity2 = validate_identity(capture_dir, expect=args.expect_identity)
        panel2 = freeze_panel(identity2["artifact"], geometry["cells"], 6, 4)
        panel2["inputs"]["artifact_sha256"] = identity2["artifact"]["sha256"]
        a = dumps_strict({"identity": {k: v for k, v in identity.items() if k != "artifact"},
                          "panel": panel})
        b = dumps_strict({"identity": {k: v for k, v in identity2.items() if k != "artifact"},
                          "panel": panel2})
        det["identity_panel_recompute"] = {
            "canonical_sha256": hashlib.sha256(a.encode()).hexdigest(),
            "stable": a == b,
            "kind": "same_process_recompute_of_identity_and_panel",
        }
    if comparisons or factor_results:
        # Canonical hash of the CONCLUSION-DRIVING evidence, plus a strict-JSON
        # round-trip recomputation of the decision from the serialized copy.
        ev = {"gauge_comparison": comparisons, "factor_evidence": factor_results}
        blob = dumps_strict(ev)
        rt = json.loads(blob)
        concl_rt = conclude(rt["gauge_comparison"], rt["factor_evidence"],
                            executed_relations=executed,
                            panel_total=len(panel["relations"]))
        det["conclusion_driving"] = {
            "canonical_sha256": hashlib.sha256(blob.encode()).hexdigest(),
            "kind": "same_process_json_roundtrip_recompute",
            "independent_real_data_repeat": False,
            "roundtrip_state": concl_rt.get("state"),
            "roundtrip_equivalent": bool(
                concl_rt.get("state") == (report.get("conclusion") or {}).get("state")
            ),
            "note": ("this is a repeatability check of the deterministic payload, NOT an "
                     "independent real-data rerun"),
        }
    report["determinism"] = det

    report["runtime"] = {
        "seconds": round(time.perf_counter() - t_start, 3),
        "peak_rss_kib": _peak_rss_kib(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "repo_head": _git_head(),
    }

    # --- durable-report privacy boundary (truly fail closed) ----------------
    scrubbed, leaks = enforce_privacy_boundary(report, (capture_dir, source_dir, scratch,
                                                        report_dir))
    if leaks:
        # redaction still left a leak: write ONLY a minimal safe failure record
        safe = minimal_failure_record(stage="privacy_scan", leak_locations=leaks)
        safe_leaks = scan_path_leaks(safe)
        if safe_leaks:  # paranoia: never write an unverified record
            safe = {"schema": SCHEMA_NAME, "schema_version": SCHEMA_VERSION, "mission": MISSION,
                    "status": "FAILED", "failed": {"stage": "privacy_scan"},
                    "privacy": {"policy": "no absolute/private path may appear in the durable artifact"}}
            if scan_path_leaks(safe):
                print(json.dumps({"status": "FAILED", "mode": args.mode,
                                  "reason": "privacy_scan_unwritable"}, sort_keys=True))
                return 3
        size = _write_report(out_path, safe, policy)
        print(f"R38A {args.mode}: FAILED privacy scan; wrote minimal safe record ({size} bytes)")
        print(json.dumps({"status": "FAILED", "mode": args.mode,
                          "reason": "privacy_scan", "out": str(out_path)}, sort_keys=True))
        return 3

    scrubbed["privacy"] = {
        "scan": "recursive over every persisted value (after known-root scrub + redaction)",
        "categories": ["posix_absolute", "windows_drive_absolute", "unc", "file_uri",
                       "posix_absolute_embedded"],
        "leaks": 0,
        "policy": "no absolute/private path may appear in the durable artifact",
    }
    final_leaks = scan_path_leaks(scrubbed)
    if final_leaks:  # the annotated record itself must always pass the scanner
        print(json.dumps({"status": "FAILED", "mode": args.mode,
                          "reason": "privacy_scan_postcheck"}, sort_keys=True))
        return 3

    size = _write_report(out_path, scrubbed, policy)
    print(f"R38A {args.mode}: wrote {out_path} ({size} bytes)")
    print(json.dumps({"status": scrubbed["status"], "mode": args.mode,
                      "conclusion": (scrubbed.get("conclusion") or {}).get("state"),
                      "out": str(out_path)}, sort_keys=True))
    return 2 if scrubbed["status"] == "FAILED" else 0


def _peak_rss_kib() -> int:
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return 0


def _git_head() -> str | None:
    try:
        import subprocess
        repo = Path(__file__).resolve().parents[1]
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _run_real_probe(args, report, geometry, panel, costs, scratch, t_start, deadline):
    """Execute the bounded real-data probe; returns (comparisons, executed, factors)."""
    z = _lazy_zemosaic()
    cfg = z["cfg_cls"](
        normalization="sky_mean", weighting="noise_variance",
        rejection="winsorized_sigma_clip", combine="mean",
        taper="footprint", taper_px=8.0, backend="cpu",
    )
    cfg_tokens = {
        "normalization": cfg.normalization, "weighting": cfg.weighting,
        "rejection": cfg.rejection, "combine": cfg.combine,
        "taper": cfg.taper, "taper_px": cfg.taper_px, "backend": cfg.backend,
    }
    ident = report["identity"]["computed"]
    n_files = report["resolution"]["n_files_on_disk"]
    source_root = Path(args.source_dir).expanduser().resolve()
    selected_cells = list(costs["selected_cells"])
    report["probe_plan"]["executed_cells"] = selected_cells
    gauge_results: list[dict] = []
    cache_dirs: dict[str, Path] = {}
    cache_info: dict[str, dict] = {}
    gauge_reuse = Path(args.gauge_cache).expanduser() if args.gauge_cache else None
    for cid in selected_cells:
        if time.perf_counter() > deadline:
            report["not_run"].append({"item": f"cell_gauge:{cid}",
                                      "reason": "soft wall-clock budget exhausted before this cell"})
            continue
        try:
            ctx = geometry["_cell_ctx"][cid]
            binding = compute_cache_binding(cid, ctx, ident, source_root, n_files, cfg_tokens)
            if gauge_reuse is not None:
                cdir = gauge_reuse / "cells" / cid
                if not (cdir / "manifest.json").is_file():
                    raise ProbeError(f"reused gauge cache missing for cell {cid}")
                if read_cache_binding(cdir) != binding:
                    raise ProbeError(f"reused gauge cache binding mismatch for cell {cid}")
                cache_info[cid] = {"cell_id": cid, "cache_key": f"cells/{cid}", "reused": True,
                                   "binding_validated": True}
            else:
                info = build_cell_cache(cid, ctx, None, scratch,
                                        max_frames=args.max_cell_contributors,
                                        binding=binding, deadline=deadline)
                cache_info[cid] = info
                cdir = scratch / "cells" / cid
            cache_dirs[cid] = cdir
            facts = probe_cell_gauge(cid, cdir, cfg)
            facts["cache"] = cache_info.get(cid)
            gauge_results.append(facts)
        except _not_run_exceptions() as exc:  # expected domain/budget refusal
            report["not_run"].append({"item": f"cell_gauge:{cid}",
                                      "reason": f"{type(exc).__name__}: {exc}"})
    by_id = {g["cell_id"]: g for g in gauge_results}

    report["cache_binding"] = {
        "cells": {cid: cache_info[cid] for cid in sorted(cache_info)},
        "validated": [cid for cid in sorted(cache_info) if cache_info[cid].get("binding_validated")],
        "note": ("a resumable scratch cache is reused ONLY when the accepted R6 completeness "
                 "predicate holds AND the persisted r38a_binding.json equals the current "
                 "capture/source/layout/patch/config/fingerprint binding"),
    }

    report["gauge_facts"] = [
        {k: v for k, v in g.items() if k not in ("coefficients", "weights", "norm_active", "weight_active")}
        | {"coefficients_sha256": hashlib.sha256(
            np.asarray(g["coefficients"]).tobytes()).hexdigest()} if g.get("status") == "ok" else g
        for g in gauge_results
    ]

    comparisons = []
    executed = []
    for rel in costs["selected_relations"]:
        u, v = rel["u"], rel["v"]
        if u in by_id and v in by_id:
            obs = _observed_channels(panel, u, v)
            cmp = compare_gauges(by_id[u], by_id[v], obs)
            cmp["relation"] = {"u": u, "v": v, "orientation": rel["orientation"]}
            comparisons.append(cmp)
            executed.append(rel)
    report["gauge_comparison"] = comparisons
    report["executed_relations"] = executed

    factor_results: list = []
    if args.mode == "factors":
        roi_cfg = {"max_cell_contributors": args.factor_max_contributors,
                   "band_px": args.factor_band_px, "max_tangent_px": args.factor_max_tangent_px}
        for rel in executed:
            if time.perf_counter() > deadline:
                factor_results.append({"relation": rel, "status": "NOT_RUN",
                                       "reason": "soft wall-clock budget exhausted before this relation"})
                continue
            factor_results.append({
                "relation": rel,
                **run_factor_probe(rel, by_id, cache_dirs, roi_cfg, cfg),
            })
        report["factor_evidence"] = factor_results

    return comparisons, executed, factor_results


def _observed_channels(panel, u, v):
    for rel in panel.get("relations", []):
        if rel["u"] == u and rel["v"] == v:
            return rel.get("channels", {})
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
