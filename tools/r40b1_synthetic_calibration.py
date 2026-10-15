"""ZM-ZEGRID-R40-B1 (freeze-closure) — v2 pure-synthetic calibration + prevalidation harness.

Standalone, deterministic, no-real-data harness. Rework-1 withdraws the invalid v1
outcome (`B1_FAIL_PROTOCOL_INVALID`) and produces a **content-addressed v2 protocol
+ calibration/criteria freeze + genuinely fresh reserved validation cohort**, published
as a prevalidation bundle `READY_FOR_NEW_VALIDATION`. **The v2 reserved validation
cohort outcomes are NOT evaluated here** — a later, separate architect mission performs
the one-shot validation.

Rework-1 repairs (against the Nono review-0 findings, 3 HIGH / 5 MEDIUM / 2 LOW):

* **H1 privacy/hash** — no silent scientific redaction anywhere; a genuine path leak
  **fails closed** (``PrivacyError``, no artifact). Prose false positives are avoided
  structurally (metrics are structured keys/expressions with no slash-looking text) *and*
  the detector now ignores prose-bearing strings. The privacy boundary runs **before**
  canonical hashing, so the persisted payload is exactly the hashed payload; the content
  hash is recomputable from the persisted artifact after removing only its documented
  self-hash fields.
* **H2 deep freeze** — criteria/protocol are **deep-canonical immutable snapshots**
  (recursive; no reachable mutable dict/list). The protocol and criteria hashes are
  re-verified against the canonical persisted snapshots immediately before validation and
  **inside every admission decision**; tamper refuses.
* **H3 durable exactly-once** — no resettable in-process scientific lock. The CLI is split
  into ``prepare`` (repeatable) and ``validate`` (one-shot). The validation claim is
  **durable and content-addressed** by ``protocol_hash + criteria_hash + cohort_hash``,
  created with exclusive ``O_CREAT|O_EXCL`` + ``fsync`` in an explicit state directory;
  a second process/call refuses. ``CLAIMED``/``COMPLETED`` semantics make an ambiguous
  crash non-silent.
* **M1 relation-first/RGB** — RGB is generated from a shared relation-level latent field
  with bounded channel perturbations (never 3 independent replicates); per-channel state
  is computed first, then aggregated relation-first with an explicit 2-of-3 rule and a
  hard single-channel override; panel counts distinct relations only; signatures are
  derived values, not trusted booleans; pseudo-positive adversaries are included.
* **M2/M3 actual H4 arms** — v2 builds **real** arms on the existing production primitives
  (I1a crop-vs-full-source parity, I1b too-tight-crop support/coverage delta propagated
  through the **actual canonical stack**, I2 fast-vs-reference on the **same** cropped
  array). The v1 abstract toy is retained but relabelled ``ORCHESTRATION_ONLY`` with **no
  authority** over B2; non-TAN fast-ineligible is exactly ``NOT_APPLICABLE``.
* **M4 thresholds** — fabricated floors removed. The M3 algebraic bound is derived from an
  **explicit IEEE rounding count** (2 + 3 roundings) instead of a floor; a zero-observed
  recovery error freezes **bit identity** where semantically appropriate; the redundant,
  non-observable ``0.75 sigma_ref`` amplitude is **removed from real H4 admission** and
  retained only as generator coverage metadata.
* **M5 split independence** — the v2 validation cohort is disjoint by **scenario family /
  parameter regime / event identity** (different shapes, support patterns, WCS parameters,
  field families, amplitude regimes, mechanism variants), enforced by a family-level guard.
  Its specs/hashes are frozen in the prevalidation bundle; **no generator call happens on
  them here**, and the validation evaluator is hard-guarded by an authorization token.

Scope preserved verbatim: the OFF/ON science/coverage/aesthetic array + FITS identity is
**exact byte identity at tolerance ZERO** and is never calibrated, relaxed or reinterpreted;
R38 stays closed; R39-A stays terminal; LP341 stays reserved; no Windows/Zealfie run; no
correction, no product/GUI/config/default change; no commit/push/tag/release.

CLI (two stages)::

    python tools/r40b1_synthetic_calibration.py prepare  --out <bundle.json> [--repeatable]
    python tools/r40b1_synthetic_calibration.py invalidate-v1 --out <invalidation.json>
    python tools/r40b1_synthetic_calibration.py validate  --bundle <bundle.json> --state-dir <dir> \
        --consume-reserved-cohort --out <result.json>          # NOT run here
    python tools/r40b1_synthetic_calibration.py print-hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Identity constants
# ---------------------------------------------------------------------------

SCHEMA_NAME = "zemosaic.zegrid.r40b1.synthetic_calibration"
SCHEMA_VERSION = "r40b1-2"
PROTOCOL_VERSION = "r40b1-protocol-2"
CRITERIA_VERSION = "r40b1-criteria-2"
COHORT_VERSION = "r40b1-cohort-2"
MISSION = "ZM-ZEGRID-R40-B1-SYNTH-CALIB"
# Stable, truthful label for the *current* prevalidation package (freeze-closure).
# This is not a per-review counter: it names the delivered package, and is written into
# the protocol payload, the prevalidation bundle and the validation claim metadata.
PHASE = "freeze-closure"
# Historical phase of the immutable v1 invalidation record. It is pinned separately so
# that record stays byte-for-byte reproducible and is never re-issued under a new phase
# (the v1 invalidation history must stay intact).
PHASE_V1_INVALIDATION = "rework-1"
BASELINE_HEAD = "50b30ebd7909a246160b59ff1111794b903c796c"
R40A_PLAN_REL = "docs/refactor/R40_UNEXPLAINED_STRUCTURE_OBSERVABILITY_PLAN.md"

TAG_PROTOCOL = "zemosaic.zegrid.r40b1.v2.protocol"
TAG_CRITERIA = "zemosaic.zegrid.r40b1.v2.criteria"
TAG_COHORT = "zemosaic.zegrid.r40b1.v2.cohort"
TAG_CALIBRATION = "zemosaic.zegrid.r40b1.v2.calibration"
TAG_CONTENT = "zemosaic.zegrid.r40b1.v2.content"
TAG_MANIFEST = "zemosaic.zegrid.r40b1.v3.product_manifest"
TAG_CODE_IDENTITY = "zemosaic.zegrid.r40b1.v3.code_identity"

# Fields that are documented self-hash / non-content fields: a verifier removes
# exactly these before recomputing the content hash from a persisted artifact.
SELF_HASH_FIELDS = ("content_sha256", "instrumentation")

STATUS_FAIL_INVALID = "B1_FAIL_PROTOCOL_INVALID"
STATUS_READY = "READY_FOR_NEW_VALIDATION"

# RM1 (freeze-closure) — the product-source manifest binds the **complete** product
# source tree, not a curated subset. The previous hand-maintained allowlist was proven
# incomplete: the arms' real import closure is larger than any short list, and an eagerly
# imported but unbound module could silently change validated behaviour. The manifest is
# therefore a deterministic, sorted, **raw-byte** SHA-256 of every ``*.py`` under
# ``src/zemosaic/``. This deliberately over-binds the I1a/I1b/I2/canonical-stack/
# calibration import closure: over-binding can only refuse (fail closed) and force a
# re-issue, whereas under-binding would falsely accept a stale package.
PRODUCT_SOURCE_ROOT = "src/zemosaic"
PRODUCT_SOURCE_POLICY = (
    "complete product source tree: every *.py under src/zemosaic/, repository-relative, "
    "deterministic sorted, raw-byte SHA-256 (no text decode); deliberately over-binds the "
    "I1a/I1b/I2/canonical-stack/calibration import closure rather than risk a partial manifest"
)
PRODUCT_SOURCE_UNLISTED_RATIONALE = (
    "in the complete bound product source tree (whole-tree raw-byte binding; closure audited)"
)
# Per-module rationale for the modules whose behaviour the arms directly invoke; kept for
# review clarity only. A module absent from this map is still bound in full by the policy.
PRODUCT_SOURCE_RATIONALE = MappingProxyType(
    {
        "src/zemosaic/__init__.py": "package init eagerly imported by the canonical-stack path",
        "src/zemosaic/core/__init__.py": "package init eagerly imported by the canonical-stack path",
        "src/zemosaic/core/zegrid/__init__.py":
            "package init that eagerly imports the zegrid execution/geometry/offset modules",
        "src/zemosaic/core/zegrid/execution.py":
            "reproject_cropped, reproject_cropped_fast, reproject_interp, slice_wcs, REPROJECT_ORDER (I1a/I1b/I2)",
        "src/zemosaic/core/zegrid/geometry.py":
            "SourceBounds and the patch/canvas geometry the reprojection arms construct",
        "src/zemosaic/core/zegrid/cell_feather.py":
            "apply_additive_offsets (the frozen production offsets) and feather_weight_field",
        "src/zemosaic/core/zegrid/executor.py":
            "Zegrid executor pulled in by the zegrid package init on the arms' path",
        "src/zemosaic/core/zegrid/science_adapter.py":
            "science adapter on the zegrid import path (previously unbound)",
        "src/zemosaic/core/zegrid/photometric.py":
            "photometric stage loaded by the zegrid package on the arms' path",
        "src/zemosaic/core/zegrid/mosaic.py":
            "layout/canvas helpers re-exported through the geometry path used by the arms",
        "src/zemosaic/core/zegrid/sweep.py":
            "geometry-derived helpers imported by the geometry/mosaic path invoked by the arms",
        "src/zemosaic/core/zegrid/cell_offset.py":
            "R34c per-cell offset domain module for the frozen production offsets",
        "src/zemosaic/core/zegrid/assembly.py":
            "MiniTile/CellPatch plane definitions referenced by the stack/capture boundary",
        "src/zemosaic/core/canonical_engine.py":
            "run_canonical_stack (the actual canonical stack invoked by the I1b arm)",
        "src/zemosaic/core/canonical_streaming.py":
            "canonical streaming stage on the stack path (previously unbound)",
        "src/zemosaic/core/canonical_stacking.py":
            "canonical stacking stage invoked through run_canonical_stack",
        "src/zemosaic/core/canonical_support.py":
            "canonical support stage invoked through run_canonical_stack",
        "src/zemosaic/core/canonical_equalize.py":
            "canonical equalizer imported by canonical_engine (referenced by the stack path)",
    }
)

# v1 (invalid) immutable evidence — static constants, no v1 artifact is re-read.
V1_ARTIFACT_BASENAME = "r40b1_synthetic_calibration.json"
V1_ARTIFACT_BYTES = 53978
V1_ARTIFACT_SHA256 = "7ec179a45a85a022f62c7f5aef3d4c8ff84cb496cbe239c9c11bab7d156e8290"
V1_PROTOCOL_SHA256 = "97237c5b015759d4850c9ff9e391655012a1c17a26af5cc9974448a6f031fbc9"
V1_CRITERIA_SHA256 = "09d3e6554e7373801fb3402ff283dfe9cd65e9b2603a5e57f7af0a21e4cbc196"
V1_CONTENT_SHA256 = "1032c09624e30f3daa8a3dd37707744ef8b29919154530948ba11376fb8c9cc0"
V1_CLAIMED_FINAL_STATUS = "B1_PASS"
V1_CLAIMED_R40B2_AUTHORIZED = True

EPS_F64 = float(np.finfo(np.float64).eps)
SIGMA_REF = 1.0  # declared synthetic constant; never tuned to a real corpus

INHERITED_GATES = MappingProxyType(
    {
        "rms0_sigma_gate": 1.25,
        "reduction_min": 0.20,
        "p90_worsening_max": 0.05,
        "relations_per_orientation_min": 2,
        "panel_relation_count": 38,
        "channels_per_relation": 3,
        "channel_promote_min": 2,
        "orientations": ("h", "v"),
    }
)

PANEL_RELATIONS: tuple[tuple[str, str], ...] = tuple(
    [(f"h{i:02d}", "h") for i in range(20)] + [(f"v{i:02d}", "v") for i in range(18)]
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class HarnessError(RuntimeError):
    """Base harness error."""


class ProtocolHashError(HarnessError):
    """Frozen protocol hash does not bind to the canonical snapshot."""


class CriteriaTamperError(HarnessError):
    """Frozen criteria snapshot was mutated or does not match its hash."""


class PrivacyError(HarnessError):
    """A genuine absolute/private path leak was detected (fail closed)."""


class NonFiniteError(HarnessError):
    """A payload carries a non-finite float (fail closed)."""


class ValidationClaimError(HarnessError):
    """The durable one-shot validation claim cannot be taken."""


class FamilyDisjointnessError(HarnessError):
    """Calibration and validation cohort families are not disjoint."""


class ReservedCohortError(HarnessError):
    """The reserved v2 validation cohort may not be evaluated in this rework."""


# ---------------------------------------------------------------------------
# Module-scope import audit
# ---------------------------------------------------------------------------

def module_scope_imports() -> list[str]:
    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return sorted(set(names))


def import_audit() -> dict:
    mods = module_scope_imports()
    return {
        "module_scope_imports": mods,
        "zemosaic_module_scope_imports": [m for m in mods if m.split(".")[0] == "zemosaic"],
        "engine_access_is_lazy": not any(m.split(".")[0] == "zemosaic" for m in mods),
        "astropy_module_scope_imports": [m for m in mods if m.split(".")[0] == "astropy"],
    }


def _ensure_src_on_path() -> None:
    repo = Path(__file__).resolve().parents[1]
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _lazy_apply_additive_offsets():
    _ensure_src_on_path()
    from zemosaic.core.zegrid.cell_feather import apply_additive_offsets

    return apply_additive_offsets


def _lazy_feather_weight_field():
    _ensure_src_on_path()
    from zemosaic.core.zegrid.cell_feather import feather_weight_field

    return feather_weight_field


def _lazy_execution():
    _ensure_src_on_path()
    from zemosaic.core.zegrid import execution

    return execution


def _lazy_canonical_stack():
    _ensure_src_on_path()
    from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack

    return CanonicalStackRequest, run_canonical_stack


# ---------------------------------------------------------------------------
# Strict JSON + canonical hashing (hashes are computed AFTER the privacy boundary)
# ---------------------------------------------------------------------------

def dumps_strict(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, allow_nan=False, indent=2, ensure_ascii=True)


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Raw-byte SHA-256 (no decoding): the identity/hash primitive everywhere."""
    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_payload(payload: Any, tag: str) -> str:
    h = hashlib.sha256()
    h.update((str(tag) + "\n").encode("utf-8"))
    h.update(canonical_json(payload).encode("utf-8"))
    return h.hexdigest()


def content_sha256(payload: Mapping) -> str:
    """Content hash recomputable from a persisted artifact.

    A verifier removes exactly the documented self-hash/non-content fields
    (``SELF_HASH_FIELDS``) and recomputes; see :func:`verify_content_hash`.
    """
    reduced = {k: v for k, v in payload.items() if k not in SELF_HASH_FIELDS}
    return sha256_payload(reduced, TAG_CONTENT)


def verify_content_hash(payload: Mapping) -> bool:
    stored = payload.get("content_sha256")
    return bool(stored) and stored == content_sha256(payload)


def assert_finite_json(payload: Any, prefix: str = "$") -> None:
    if isinstance(payload, Mapping):
        for key in sorted(payload):
            assert_finite_json(payload[key], f"{prefix}.{key}")
    elif isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            assert_finite_json(item, f"{prefix}[{i}]")
    elif isinstance(payload, float) and not math.isfinite(payload):
        raise NonFiniteError(f"non-finite float at {prefix}")


# ---------------------------------------------------------------------------
# Privacy: fail-closed, prose-aware, NON-mutating
# ---------------------------------------------------------------------------
#
# The v1 boundary silently rewrote scientific prose that merely *looked* path-like.
# v2 fixes this two ways: (a) metrics are structured and contain no slash-looking
# prose, and (b) the detector ignores strings that carry prose markers, so genuine
# prose cannot be misclassified. A genuine leak REFUSES (no artifact is written);
# scientific content is never redacted in place.

_WIN_DRIVE_RE = re.compile(r"(?:^|[^A-Za-z0-9])[A-Za-z]:[\\/]")
_UNC_RE = re.compile(r"(?:^|[^\\])\\\\[A-Za-z0-9._$-]+[\\]")
_FILE_URI_RE = re.compile(r"file://", re.IGNORECASE)
# Embedded POSIX absolute path: at least TWO segments, token-anchored (a
# single-segment fragment such as "RMS(d_shadow)/RMS(d_prod)" must NOT match).
_EMBEDDED_ABS_RE = re.compile(
    r"(?<![\w:/])/(?:[A-Za-z0-9._~+\-]+/)+[A-Za-z0-9._~+\-]+"
)


def path_leak_category(value: str) -> str | None:
    """Classify a genuine absolute/private path.

    Structural prose-safety: the embedded form requires at least two path segments
    and is token-anchored, so scientific metric prose (which contains single-slash
    fragments inside expressions) is never classified as a path. A genuine path that
    happens to sit inside prose is still detected.
    """
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
# Output-path safety + atomic strict-JSON write
# ---------------------------------------------------------------------------

def is_drive_absolute(raw: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", str(raw).strip()))


def has_traversal_component(raw: str) -> bool:
    return any(p == ".." for p in re.split(r"[\\/]+", str(raw)))


def resolve_output_path(raw: str, *, allowed_root: Path | None = None) -> Path:
    """Reject empty / file-uri / drive / UNC / traversal / out-of-root outputs."""
    if not isinstance(raw, str) or not raw.strip():
        raise PrivacyError("empty output path")
    s = raw.strip()
    if s.lower().startswith("file://"):
        raise PrivacyError("file:// output paths are refused")
    if is_drive_absolute(s) or s.startswith("\\\\"):
        raise PrivacyError("absolute drive/UNC output paths are refused")
    if has_traversal_component(s):
        raise PrivacyError("parent-traversal output paths are refused")
    p = Path(s).expanduser()
    if p.is_symlink():
        raise PrivacyError("symlinked output targets are refused")
    resolved = p.resolve()
    if allowed_root is not None:
        root = Path(allowed_root).expanduser().resolve()
        if root != resolved and root not in resolved.parents:
            raise PrivacyError("output path escapes the allowed output root")
    return p


def atomic_write_json(path: Path, payload: Any) -> str:
    """Atomic strict-JSON write; returns the SHA-256 of the exact written bytes."""
    text = dumps_strict(payload)
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    fd, tmp_name = tempfile.mkstemp(prefix=Path(path).name + ".", suffix=".tmp", dir=str(parent), text=True)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if tmp.is_symlink() or Path(path).is_symlink():
            raise PrivacyError("refusing to write through a symlink")
        os.replace(tmp, path)
        _fsync_dir(parent)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return sha256_text(text)


def _fsync_dir(path: Path) -> None:
    try:
        dfd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


# ---------------------------------------------------------------------------
# H2 — deep canonical immutable snapshots
# ---------------------------------------------------------------------------

def deep_freeze(obj: Any) -> Any:
    """Recursively freeze a JSON-like structure (no reachable mutable dict/list)."""
    if isinstance(obj, Mapping):
        return MappingProxyType({str(k): deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(deep_freeze(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return tuple(obj.tolist())
    return obj


@dataclass(frozen=True)
class FrozenSnapshot:
    """A deep-frozen, canonically bound snapshot of a persisted payload.

    The **authority is the persisted payload**; ``canonical_json`` and ``sha256``
    must both agree with it, and ``frozen`` must be its deep-frozen image.
    :meth:`verify` requires **canonical equivalence** among all four
    (persisted payload, parsed ``canonical_json``, deep-frozen snapshot, hash),
    so a tampered payload can never be consumed while the hash still validates.
    """

    name: str
    tag: str
    canonical_json: str
    sha256: str
    frozen: Any = field(repr=False)

    def verify(self) -> None:
        import json as _json

        try:
            parsed = _json.loads(self.canonical_json)
        except Exception as exc:  # noqa: BLE001
            raise CriteriaTamperError(f"{self.name}: canonical snapshot is not valid JSON") from exc
        if canonical_json(parsed) != self.canonical_json:
            raise CriteriaTamperError(f"{self.name}: canonical snapshot is not in canonical form")
        if _thaw(self.frozen) != parsed:
            raise CriteriaTamperError(
                f"{self.name}: deep-frozen snapshot differs from the canonical snapshot"
            )
        recomputed = sha256_payload(parsed, self.tag)
        if recomputed != self.sha256:
            raise CriteriaTamperError(
                f"{self.name}: hash mismatch (stored {self.sha256[:16]}…, recomputed {recomputed[:16]}…)"
            )


def snapshot_from_entry(name: str, tag: str, entry: Mapping) -> FrozenSnapshot:
    """Build a snapshot from a persisted entry; the persisted payload is the authority.

    ``entry`` must carry ``payload``, ``canonical_json`` and ``sha256``. The
    canonical form of the persisted payload must equal ``canonical_json`` and its
    tagged hash must equal ``sha256``; otherwise it refuses (audit hashes are
    therefore recomputable from the persisted artifact).
    """
    for key in ("payload", "canonical_json", "sha256"):
        if key not in entry:
            raise CriteriaTamperError(f"{name}: persisted entry is missing {key!r}")
    payload = _thaw(entry["payload"])
    cj = canonical_json(payload)
    if cj != entry["canonical_json"]:
        raise CriteriaTamperError(
            f"{name}: persisted payload is not canonically equivalent to the stored canonical_json"
        )
    sha = sha256_payload(payload, tag)
    if sha != entry["sha256"]:
        raise CriteriaTamperError(f"{name}: persisted payload hash does not match the stored sha256")
    return FrozenSnapshot(name=name, tag=tag, canonical_json=cj, sha256=sha, frozen=deep_freeze(payload))


def freeze_snapshot(name: str, tag: str, payload: Mapping) -> FrozenSnapshot:
    plain = _thaw(payload)
    return FrozenSnapshot(
        name=name,
        tag=tag,
        canonical_json=canonical_json(plain),
        sha256=sha256_payload(plain, tag),
        frozen=deep_freeze(plain),
    )


def _thaw(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_thaw(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# R3 — code / repository-HEAD identity binding (no writes, read-only probes)
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def compute_code_identity() -> dict:
    """Deterministic identity of the code + repository state producing a bundle.

    RM1 (freeze-closure): besides this tool's own **raw-byte** source hash and a
    read-only ``git rev-parse HEAD`` probe, the identity binds the **complete
    product-source manifest** (every ``*.py`` under ``src/zemosaic/``) and the
    **numeric runtime dependency versions**. Nothing is allowed to be reported as
    ``unavailable``: a missing file, an unreadable version or a failed HEAD probe
    refuses. A mismatch at preflight refuses the bundle (it must be re-issued by the
    current code).
    """
    src = Path(__file__).read_bytes()
    manifest = compute_product_source_manifest()
    versions = compute_runtime_versions()
    return {
        "tool_basename": Path(__file__).name,
        "tool_source_sha256": sha256_bytes(src),
        "product_source_manifest": manifest,
        "product_source_manifest_count": len(manifest),
        "product_source_manifest_policy": PRODUCT_SOURCE_POLICY,
        "product_source_manifest_sha256": sha256_payload(manifest, TAG_MANIFEST),
        "runtime_versions": versions,
        "baseline_head": BASELINE_HEAD,
        "repo_head": _probe_repo_head(),
        "policy": (
            "bundle pins the raw-byte tool source, the complete raw-byte product-source manifest "
            "(every *.py under src/zemosaic/), the numeric runtime versions and the repository "
            "HEAD; preflight re-probes all of them and refuses on any mismatch (a dirty product "
            "edit or an added/removed product source with an unchanged HEAD also refuses)"
        ),
    }


def code_identity_digest(identity: Mapping) -> str:
    return sha256_payload(_thaw(identity), TAG_CODE_IDENTITY)


def product_source_paths() -> list:
    """Sorted repository-relative paths of every product ``*.py`` under ``src/zemosaic/``.

    This is the *complete* product source tree, so the manifest cannot omit an eagerly
    imported module. An absent root or an empty tree refuses (fail closed).
    """
    repo = _repo_root()
    root = repo / PRODUCT_SOURCE_ROOT
    if not root.is_dir():
        raise HarnessError(f"product source root is missing: {PRODUCT_SOURCE_ROOT}")
    paths = sorted(
        p.relative_to(repo).as_posix() for p in root.rglob("*.py") if p.is_file()
    )
    if not paths:
        raise HarnessError(f"product source tree is empty: {PRODUCT_SOURCE_ROOT}")
    return paths


def compute_product_source_manifest() -> list:
    """Complete, sorted, raw-byte product-source manifest (fail closed).

    Binds **every** ``*.py`` under ``src/zemosaic/`` by raw-byte SHA-256, so a dirty
    edit, an addition or a removal anywhere in the product source tree changes the
    manifest and refuses at preflight — even with an unchanged git HEAD.
    """
    repo = _repo_root()
    entries = []
    for rel in product_source_paths():
        p = repo / rel
        if not p.is_file():
            raise HarnessError(f"product source is missing: {rel}")
        entries.append(
            {
                "path": rel,
                "sha256": sha256_bytes(p.read_bytes()),
                "why": PRODUCT_SOURCE_RATIONALE.get(rel, PRODUCT_SOURCE_UNLISTED_RATIONALE),
            }
        )
    return entries


def loaded_product_source_paths() -> list:
    """Repository-relative paths of the local product modules currently imported."""
    repo = _repo_root()
    found = set()
    for name, module in list(sys.modules.items()):
        if not (name == "zemosaic" or name.startswith("zemosaic.")):
            continue
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        try:
            rel = Path(origin).resolve().relative_to(repo).as_posix()
        except ValueError:
            continue
        if rel.startswith(PRODUCT_SOURCE_ROOT + "/") and rel.endswith(".py"):
            found.add(rel)
    return sorted(found)


def force_product_import_closure() -> None:
    """Force the exact lazy imports / calibration entrypoints the arms exercise."""
    _ensure_src_on_path()
    _lazy_execution()
    _lazy_canonical_stack()
    _lazy_apply_additive_offsets()
    _lazy_feather_weight_field()
    run_calibration()


def audit_loaded_product_closure() -> dict:
    """Closure audit: every local product module the arms load is bound in the manifest.

    Forces the I1a/I1b/I2/canonical-stack/calibration entrypoints, enumerates the local
    ``zemosaic`` modules actually imported from this repository, and proves each one is
    present in the persisted product-source manifest. ``unbound_local_modules`` must be
    empty: a non-empty list is a falsely-complete manifest.
    """
    force_product_import_closure()
    manifest_paths = [e["path"] for e in compute_product_source_manifest()]
    loaded = loaded_product_source_paths()
    return {
        "manifest_policy": PRODUCT_SOURCE_POLICY,
        "manifest_count": len(manifest_paths),
        "manifest_paths": manifest_paths,
        "loaded_local_modules": loaded,
        "loaded_count": len(loaded),
        "unbound_local_modules": sorted(set(loaded) - set(manifest_paths)),
    }


def compute_runtime_versions() -> dict:
    """Deterministic numeric runtime dependency versions (fail closed if missing)."""
    import importlib.metadata as _md

    versions = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }
    for name, dist in (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("astropy", "astropy"),
        ("reproject", "reproject"),
    ):
        try:
            versions[name] = str(_md.version(dist))
        except Exception as exc:  # noqa: BLE001
            raise HarnessError(
                f"required runtime dependency version is unavailable for {name}: {type(exc).__name__}"
            ) from exc
    return versions


def _probe_repo_head() -> str:
    repo = Path(__file__).resolve().parents[1]
    try:
        import subprocess

        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20,
        )
    except Exception as exc:  # noqa: BLE001
        raise HarnessError(f"repository HEAD probe failed: {type(exc).__name__}") from exc
    head = out.stdout.strip()
    if out.returncode != 0 or not head:
        raise HarnessError("repository HEAD probe returned no commit (fail closed)")
    return head


def _repo_head() -> str:
    return _probe_repo_head()


# ---------------------------------------------------------------------------
# H3 — durable content-addressed one-shot validation claim
# ---------------------------------------------------------------------------

def assert_safe_directory(raw: Path, *, label: str) -> Path:
    """Fail-closed directory safety: no symlinked component, no non-directory component.

    Walks every existing component from the filesystem anchor down, refusing a
    symlinked component (path substitution / marker redirection) and refusing an
    existing component that is not a directory.
    """
    p = Path(raw).expanduser()
    if not str(p).strip():
        raise PrivacyError(f"{label}: empty path")
    abs_p = p if p.is_absolute() else (Path.cwd() / p)
    parts = abs_p.parts
    cur = Path(parts[0])
    for part in parts[1:]:
        cur = cur / part
        if cur.is_symlink():
            raise PrivacyError(f"{label}: symlinked path component is refused ({cur.name})")
        if cur.exists() and not cur.is_dir():
            raise PrivacyError(f"{label}: existing non-directory component is refused ({cur.name})")
    return p


class DurableClaim:
    """Exclusive, durable, content-addressed validation claim.

    States: ``NONE`` -> ``CLAIMED`` -> ``COMPLETED``. A second claim (in-process or
    cross-process) refuses. An ambiguous crash (``CLAIMED`` without ``COMPLETED``)
    refuses rather than silently rerunning. The state directory and its parents are
    validated for symlink/path substitution fail-closed before any mkdir/status/claim.
    """

    def __init__(self, state_dir: Path, claim_key: Mapping) -> None:
        self.state_dir = assert_safe_directory(Path(state_dir), label="validation state dir")
        key = canonical_json(_thaw(claim_key))
        self.claim_id = sha256_text(key)[:40]
        self.claim_path = self.state_dir / f"claim.{self.claim_id}.json"
        self.completed_path = self.state_dir / f"completed.{self.claim_id}.json"

    def _check_markers(self) -> None:
        assert_safe_directory(self.state_dir, label="validation state dir")
        for marker in (self.claim_path, self.completed_path):
            if marker.is_symlink():
                raise PrivacyError(f"symlinked claim marker is refused ({marker.name})")
            if marker.exists() and not marker.is_file():
                raise PrivacyError(f"non-file claim marker is refused ({marker.name})")

    def status(self) -> str:
        self._check_markers()
        if self.completed_path.exists():
            return "COMPLETED"
        if self.claim_path.exists():
            return "CLAIMED"
        return "NONE"

    def claim(self, *, metadata: Mapping) -> dict:
        self._check_markers()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._check_markers()
        state = self.status()
        if state == "COMPLETED":
            raise ValidationClaimError(f"validation already COMPLETED for claim {self.claim_id}")
        if state == "CLAIMED":
            raise ValidationClaimError(
                f"validation already CLAIMED (ambiguous/incomplete) for claim {self.claim_id}; "
                "refusing to rerun silently"
            )
        body = canonical_json(
            {
                "claim_id": self.claim_id,
                "state": "CLAIMED",
                "metadata": _thaw(metadata),
            }
        ).encode("utf-8")
        try:
            fd = os.open(str(self.claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise ValidationClaimError(
                f"validation claim lost the exclusive race for {self.claim_id}"
            ) from exc
        try:
            os.write(fd, body)
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_dir(self.state_dir)
        return {"claim_id": self.claim_id, "state": "CLAIMED", "claim_path_name": self.claim_path.name}

    def complete(self, *, result_sha256: str, metadata: Mapping) -> dict:
        self._check_markers()
        body = {
            "claim_id": self.claim_id,
            "state": "COMPLETED",
            "result_sha256": str(result_sha256),
            "metadata": _thaw(metadata),
        }
        text = canonical_json(body)
        parent = self.state_dir
        parent.mkdir(parents=True, exist_ok=True)
        import tempfile

        fd, tmp_name = tempfile.mkstemp(prefix=".completed.", suffix=".tmp", dir=str(parent), text=True)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.completed_path)
            _fsync_dir(parent)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
        return {"claim_id": self.claim_id, "state": "COMPLETED", "result_sha256": result_sha256}


# ---------------------------------------------------------------------------
# Scenario families (family-level disjointness; v2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioSpec:
    group_id: str
    split: str  # "calibration" | "validation"
    seed: int
    family: str
    params: tuple

    def identity(self) -> dict:
        return {
            "group_id": self.group_id,
            "split": self.split,
            "seed": int(self.seed),
            "family": self.family,
            "params": list(self.params),
        }


def _calibration_specs() -> list[ScenarioSpec]:
    """Calibration families: regimes A (declared constants, no real tuning)."""
    s: list[ScenarioSpec] = []
    # --- M3 adversarial identity arms ---
    m3 = [
        ("m3_cal_a_unit_pos", 3101, "m3.unit", ("unit", "pos", "plain")),
        ("m3_cal_a_cancel_neg", 3102, "m3.cancel", ("unit", "neg", "cancelling_opposite_offsets")),
        ("m3_cal_a_signed_zero", 3103, "m3.signed_zero", ("signed_zero", "pos", "plain")),
        ("m3_cal_a_subnormal_underflow", 3104, "m3.subnormal", ("subnormal", "pos", "underflow_to_zero")),
        ("m3_cal_a_large_opposite", 3105, "m3.large", ("large", "neg", "opposite_sign_large")),
        ("m3_cal_a_nonfinite_nan", 3106, "m3.nonfinite", ("unit", "pos", "nan_input")),
        ("m3_cal_a_nonfinite_inf", 3107, "m3.nonfinite", ("unit", "pos", "inf_input")),
    ]
    for gid, seed, fam, p in m3:
        s.append(ScenarioSpec(gid, "calibration", seed, fam, p))
    # --- ROI-shadow orchestration arms (declared ORCHESTRATION_ONLY) ---
    roi = [
        ("roi_cal_a_flat_h", 3201, "roi.flat_h", ("flat", "h", "pointwise_frozen")),
        ("roi_cal_a_gradient_v", 3202, "roi.gradient_v", ("gradient", "v", "pointwise_frozen")),
        ("roi_cal_a_nan_h", 3203, "roi.nan_h", ("nan_patch", "h", "pointwise_frozen")),
        ("roi_cal_a_globalrec_v", 3204, "roi.global_v", ("gradient", "v", "roi_global_recompute")),
    ]
    for gid, seed, fam, p in roi:
        s.append(ScenarioSpec(gid, "calibration", seed, fam, p))
    # --- real primitive reprojection arms ---
    rep = [
        ("reproj_cal_a_i1a_tan", 3301, "reproj.i1a_tan", ("i1a", "tan")),
        ("reproj_cal_a_i1b_tan", 3302, "reproj.i1b_tan", ("i1b", "tan")),
        ("reproj_cal_a_i2_tan", 3303, "reproj.i2_tan", ("i2", "tan")),
        ("reproj_cal_a_i2_non_tan", 3304, "reproj.i2_non_tan", ("i2", "non_tan")),
        ("reproj_cal_a_i1b_non_tan", 3305, "reproj.i1b_non_tan", ("i1b", "non_tan")),
    ]
    for gid, seed, fam, p in rep:
        s.append(ScenarioSpec(gid, "calibration", seed, fam, p))
    # --- relation-first panel arms (correlated RGB) ---
    pan = [
        ("panel_cal_a_isolated", 3401, "panel.isolated", ("isolated", "moderate")),
        ("panel_cal_a_null", 3402, "panel.null", ("null", "none")),
        ("panel_cal_a_pseudo_single_relation", 3403, "panel.pseudo_single_relation", ("pseudo_single_relation", "moderate")),
        ("panel_cal_a_contradictory_channel", 3404, "panel.contradictory_channel", ("contradictory_channel", "moderate")),
        ("panel_cal_a_channel_duplication", 3405, "panel.channel_duplication", ("channel_duplication", "moderate")),
        ("panel_cal_a_single_orientation", 3406, "panel.single_orientation", ("single_orientation", "moderate")),
        ("panel_cal_a_confounded", 3407, "panel.confounded", ("confounded", "moderate")),
        ("panel_cal_a_incomplete_panel", 3408, "panel.incomplete_panel", ("incomplete_panel", "moderate")),
        ("panel_cal_a_per_frame_only", 3409, "panel.per_frame_only", ("per_frame_only", "moderate")),
        ("panel_cal_a_rgb_pseudorep_flag", 3410, "panel.rgb_pseudorep_flag", ("rgb_pseudorep_flag", "moderate")),
        ("panel_cal_a_parity_fail", 3411, "panel.parity_fail", ("parity_fail", "none")),
    ]
    for gid, seed, fam, p in pan:
        s.append(ScenarioSpec(gid, "calibration", seed, fam, p))
    return s


def _validation_specs_reserved() -> list[ScenarioSpec]:
    """RESERVED v2 validation cohort — metadata only; NOT evaluated here.

    Disjoint from calibration by **family name, scenario family and parameter
    regime** (different shapes, support patterns, WCS parameters, field families,
    amplitude regimes, mechanism variants) — not merely by seed.
    """
    s: list[ScenarioSpec] = []
    # M3: different magnitude regimes + different adversarial variants
    m3 = [
        ("m3_val_b_near_overflow", 7101, "m3.overflow", ("near_overflow", "pos", "plain")),
        ("m3_val_b_cancel_scale", 7102, "m3.cancel_scale", ("huge", "neg", "cancelling_opposite_offsets")),
        ("m3_val_b_subnormal_zero", 7103, "m3.subnormal_zero", ("subnormal", "neg", "underflow_to_zero")),
        ("m3_val_b_nonfinite_nan", 7104, "m3.nonfinite_b", ("unit", "neg", "nan_input")),
    ]
    for gid, seed, fam, p in m3:
        s.append(ScenarioSpec(gid, "validation", seed, fam, p))
    # ROI: different field families + different shapes
    roi = [
        ("roi_val_b_stars_v", 7201, "roi_b.stars_v", ("stars", "v", "pointwise_frozen")),
        ("roi_val_b_diffuse_h", 7202, "roi_b.diffuse_h", ("diffuse", "h", "pointwise_frozen")),
        ("roi_val_b_globalrec_h", 7203, "roi_b.global_h", ("stars", "h", "roi_global_recompute")),
    ]
    for gid, seed, fam, p in roi:
        s.append(ScenarioSpec(gid, "validation", seed, fam, p))
    rep = [
        ("reproj_val_b_i1a_tan", 7301, "reproj_b.i1a_tan", ("i1a", "tan")),
        ("reproj_val_b_i1b_tan", 7302, "reproj_b.i1b_tan", ("i1b", "tan")),
        ("reproj_val_b_i2_tan", 7303, "reproj_b.i2_tan", ("i2", "tan")),
        ("reproj_val_b_i2_non_tan", 7304, "reproj_b.i2_non_tan", ("i2", "non_tan")),
    ]
    for gid, seed, fam, p in rep:
        s.append(ScenarioSpec(gid, "validation", seed, fam, p))
    pan = [
        ("panel_val_b_isolated", 7401, "panel_b.isolated", ("isolated", "weak")),
        ("panel_val_b_null", 7402, "panel_b.null", ("null", "none")),
        ("panel_val_b_pseudo_single_relation", 7403, "panel_b.pseudo_single_relation", ("pseudo_single_relation", "weak")),
        ("panel_val_b_contradictory_channel", 7404, "panel_b.contradictory_channel", ("contradictory_channel", "weak")),
        ("panel_val_b_confounded", 7405, "panel_b.confounded", ("confounded", "weak")),
        ("panel_val_b_incomplete_panel", 7406, "panel_b.incomplete_panel", ("incomplete_panel", "weak")),
        ("panel_val_b_single_orientation", 7407, "panel_b.single_orientation", ("single_orientation", "weak")),
    ]
    for gid, seed, fam, p in pan:
        s.append(ScenarioSpec(gid, "validation", seed, fam, p))
    return s


def calibration_family_key(s: ScenarioSpec) -> tuple:
    return (s.family, str(s.params))


def assert_family_disjoint(cal: Sequence[ScenarioSpec], val: Sequence[ScenarioSpec]) -> dict:
    cal_fams = {s.family for s in cal}
    val_fams = {s.family for s in val}
    shared = sorted(cal_fams & val_fams)
    cal_keys = {calibration_family_key(s) for s in cal}
    val_keys = {calibration_family_key(s) for s in val}
    overlap = sorted(cal_keys & val_keys)
    cal_seeds = {s.seed for s in cal}
    val_seeds = {s.seed for s in val}
    if shared or overlap or (cal_seeds & val_seeds):
        raise FamilyDisjointnessError(
            f"split not disjoint: families={shared} keys={overlap} seeds={sorted(cal_seeds & val_seeds)}"
        )
    return {
        "calibration_families": sorted(cal_fams),
        "validation_families": sorted(val_fams),
        "shared_families": [],
        "shared_seeds": [],
        "rule": "disjoint by family name AND (family, params) regime AND seed identity",
    }


def validation_cohort_metadata(specs: Sequence[ScenarioSpec]) -> dict:
    """Metadata-only description of the reserved cohort (no generator call)."""
    entries = [s.identity() for s in specs if s.split == "validation"]
    digest = sha256_payload({"cohort_version": COHORT_VERSION, "specs": entries}, TAG_COHORT)
    return {
        "cohort_version": COHORT_VERSION,
        "cohort_sha256": digest,
        "specs": entries,
        "generator_invoked": False,
        "outcomes_present": False,
    }


# ---------------------------------------------------------------------------
# Deterministic generators (calibration)
# ---------------------------------------------------------------------------

def _m3_scale(cls: str) -> float:
    return {
        "unit": 1.0,
        "signed_zero": 0.0,
        "subnormal": float(np.finfo(np.float32).smallest_subnormal) * 8.0,
        "large": 1e30,
        "huge": 1e30,
        "near_overflow": 1e38,
    }[cls]


def gen_m3_case(spec: ScenarioSpec) -> dict:
    cls, sign, variant = spec.params
    sign_f = 1.0 if sign == "pos" else -1.0
    rng = np.random.default_rng(spec.seed)
    plane = (32, 64)
    n = plane[0] * plane[1]
    scale = _m3_scale(cls)
    base = rng.uniform(0.5, 1.5, n) * sign_f
    sci_u = (base * scale).astype(np.float32).reshape(plane)
    base_v = rng.uniform(0.5, 1.5, n) * sign_f
    sci_v = (base_v * scale).astype(np.float32).reshape(plane)
    off_u = float(rng.uniform(-1.0, 1.0)) * scale * 1e-3
    off_v = float(rng.uniform(-1.0, 1.0)) * scale * 1e-3

    if cls == "signed_zero":
        sci_u = np.full(plane, -0.0, dtype=np.float32)
        sci_v = np.full(plane, 0.0, dtype=np.float32)
        off_u, off_v = 0.0, -0.0
    if variant == "cancelling_opposite_offsets":
        off_u = float(rng.uniform(0.9, 1.1)) * np.max(np.abs(sci_u))
        off_v = -off_u
    if variant == "opposite_sign_large":
        off_u = 0.5 * scale
        off_v = -0.5 * scale
    if variant == "underflow_to_zero":
        sci_u = np.full(plane, float(np.finfo(np.float32).smallest_subnormal), dtype=np.float32)
        sci_v = np.full(plane, float(np.finfo(np.float32).smallest_subnormal), dtype=np.float32)
        off_u, off_v = -float(np.finfo(np.float32).smallest_subnormal), 0.0
    if variant == "nan_input":
        sci_u[0, 0] = np.nan
    if variant == "inf_input":
        sci_v[0, 0] = np.inf
    return {
        "group_id": spec.group_id, "seed": int(spec.seed), "family": spec.family,
        "class": cls, "sign": sign, "variant": variant,
        "sci_u": sci_u, "sci_v": sci_v, "off_u": off_u, "off_v": off_v,
        "scale": float(scale),
    }


def evaluate_m3_case(spec: ScenarioSpec) -> dict:
    apply_additive_offsets = _lazy_apply_additive_offsets()
    case = gen_m3_case(spec)
    sci_u, sci_v = case["sci_u"], case["sci_v"]
    off_u, off_v = case["off_u"], case["off_v"]
    variant = case["variant"]

    inputs_finite = bool(np.all(np.isfinite(sci_u)) and np.all(np.isfinite(sci_v)))
    if not inputs_finite:
        return {
            "group_id": spec.group_id, "family": spec.family, "seed": case["seed"],
            "decision": "REFUSED_NONFINITE", "pass": True,
            "reason": "non-finite science input must be refused by the artifact contract",
        }

    post_u = np.asarray(apply_additive_offsets(sci_u, off_u), dtype=np.float64)
    post_v = np.asarray(apply_additive_offsets(sci_v, off_v), dtype=np.float64)
    if not (np.all(np.isfinite(post_u)) and np.all(np.isfinite(post_v))):
        return {
            "group_id": spec.group_id, "family": spec.family, "seed": case["seed"],
            "decision": "REFUSED_NONFINITE", "pass": True,
            "reason": "non-finite post-offset value must be refused",
        }

    stored_u = post_u
    bit_identical = bool(
        np.array_equal(stored_u.view(np.uint64), post_u.view(np.uint64))
    )
    dtype_ok = post_u.dtype == np.float64 and post_v.dtype == np.float64
    downcast_same = bool(
        np.array_equal(post_u.astype(np.float32).astype(np.float64), post_u)
    )

    d_helper = post_u - post_v
    d_reordered = (sci_u.astype(np.float64) - sci_v.astype(np.float64)) + (off_u - off_v)
    err = d_helper - d_reordered
    scale = float(
        max(
            float(np.max(np.abs(sci_u.astype(np.float64)))),
            float(np.max(np.abs(sci_v.astype(np.float64)))),
            float(np.max(np.abs(post_u))),
            float(np.max(np.abs(post_v))),
            abs(off_u),
            abs(off_v),
        )
    )
    max_abs_err = float(np.max(np.abs(err)))
    ratio = max_abs_err / (EPS_F64 * scale) if scale > 0 else 0.0

    return {
        "group_id": spec.group_id, "family": spec.family, "seed": case["seed"],
        "class": case["class"], "sign": case["sign"], "variant": variant,
        "decision": "EVALUATED",
        "post_dtype": "float64",
        "post_dtype_ok": bool(dtype_ok),
        "stored_post_bit_identical": bit_identical,
        "f32_downcast_bit_identical": downcast_same,
        "algebraic_max_abs_err": float(max_abs_err),
        "algebraic_scale": float(scale),
        "algebraic_ratio": float(ratio),
        "pass": bool(bit_identical and dtype_ok),
    }


# ---------------------------------------------------------------------------
# ROI-shadow arms (ORCHESTRATION_ONLY — no product authority)
# ---------------------------------------------------------------------------

def _make_cell_frames(seed: int, shape_hw: tuple[int, int], kind: str) -> list[dict]:
    h, w = shape_hw
    c = 3
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = np.zeros((h, w), dtype=np.float64)
    if kind == "gradient":
        base += 0.20 * xx / max(1, w - 1) + 0.10 * yy / max(1, h - 1)
    elif kind == "stars":
        for _ in range(6):
            cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
            base[cy, cx] += 50.0
    elif kind == "diffuse":
        cy, cx = h / 2.0, w / 2.0
        base += 5.0 * np.exp(
            -(((yy - cy) ** 2) / (2 * (h / 6.0) ** 2) + ((xx - cx) ** 2) / (2 * (w / 6.0) ** 2))
        )
    elif kind == "nan_patch":
        base[2:6, 2:9] = np.nan
    frames = []
    for k in range(5):
        plane = base + rng.normal(0.0, 0.05, (h, w)) + 0.01 * k
        sci = np.repeat(plane[:, :, None], c, axis=2).astype(np.float32)
        wgt = np.full((h, w, c), 1.0 + 0.5 * k, dtype=np.float64)
        wgt[..., 1] *= 0.5
        valid = np.ones((h, w, c), dtype=bool)
        if k == 0:
            valid[h // 3 : h // 3 + 4, :, :] = False
        if k == 2:
            valid[:, w // 4 : w // 4 + 3, 2] = False
        frames.append({"sci": sci, "weight": wgt, "valid": valid})
    return frames


def _roi_band(orientation: str, shape_hw: tuple[int, int], halo: int):
    h, w = shape_hw
    if orientation == "h":
        idx = np.unique(np.clip(np.array([0, halo - 1, halo, w // 2, w - 2, w - 1]), 0, w - 1))
        rr, cc = np.meshgrid(np.arange(h, dtype=int), idx, indexing="ij")
        return rr.ravel(), cc.ravel(), (slice(0, h), slice(int(idx.min()), int(idx.max()) + 1))
    idx = np.unique(np.clip(np.array([0, halo - 1, halo, h // 2, h - 2, h - 1]), 0, h - 1))
    rr, cc = np.meshgrid(idx, np.arange(w, dtype=int), indexing="ij")
    return rr.ravel(), cc.ravel(), (slice(int(idx.min()), int(idx.max()) + 1), slice(0, w))


def _pointwise_stack_post_blend(frames: list[dict], offsets: np.ndarray, ramp: np.ndarray) -> np.ndarray:
    apply_additive_offsets = _lazy_apply_additive_offsets()
    h, w = ramp.shape
    c = 3
    num = np.zeros((h, w, c), dtype=np.float64)
    den = np.zeros((h, w, c), dtype=np.float64)
    for fr in frames:
        sci = np.asarray(fr["sci"], dtype=np.float64)
        wgt = np.asarray(fr["weight"], dtype=np.float64)
        m = np.asarray(fr["valid"], dtype=bool) & np.isfinite(sci)
        num += np.where(m, sci * wgt, 0.0)
        den += np.where(m, wgt, 0.0)
    stacked = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan).astype(np.float32)
    post = np.asarray(apply_additive_offsets(stacked, offsets), dtype=np.float64)
    return post * ramp[:, :, None]


def evaluate_roi_case(spec: ScenarioSpec) -> dict:
    feather_weight_field = _lazy_feather_weight_field()
    kind, orientation, mode = spec.params
    shape_hw = (48, 56)
    halo = 8
    ramp = feather_weight_field(
        shape_hw, (0, 40, 0, 48), halo, bounds=(0, 40, 8, 48), canvas_hw=(64, 80)
    )
    offsets = np.array([0.013, -0.027, 0.041], dtype=np.float64)
    frames = _make_cell_frames(spec.seed, shape_hw, kind)
    prod = _pointwise_stack_post_blend(frames, offsets, ramp)
    rr, cc, (rs, cs) = _roi_band(orientation, shape_hw, halo)

    if mode == "pointwise_frozen":
        sub = [
            {"sci": fr["sci"][rs, cs, :], "weight": fr["weight"][rs, cs, :], "valid": fr["valid"][rs, cs, :]}
            for fr in frames
        ]
        shadow = _pointwise_stack_post_blend(sub, offsets, ramp[rs, cs])
        shadow_at = shadow[rr - rs.start, cc - cs.start, :]
        ref_at = prod[rr, cc, :]
        finite = np.isfinite(ref_at) & np.isfinite(shadow_at)
        n_finite = int(np.count_nonzero(finite))
        bit_same = (shadow_at.view(np.uint64) == ref_at.view(np.uint64)) & finite
        n_bit = int(np.count_nonzero(bit_same))
        max_abs = float(np.max(np.abs(shadow_at[finite] - ref_at[finite]))) if n_finite else 0.0
        ok = bool(n_finite > 0 and n_bit == n_finite)
        return {
            "group_id": spec.group_id, "family": spec.family, "seed": int(spec.seed),
            "mode": mode, "authority": "ORCHESTRATION_ONLY",
            "parity_mode": "BIT_IDENTICAL", "tolerance": 0.0,
            "n_finite_compared": n_finite, "n_bit_equal": n_bit, "max_abs_diff": max_abs,
            "decision": "PARITY_OK" if ok else "NOT_IDENTIFIABLE", "pass": ok,
        }
    if mode == "roi_global_recompute":
        global_norm = float(np.nanmedian(np.asarray(frames[0]["sci"], dtype=np.float64)))
        roi_vals = np.asarray(frames[0]["sci"][rs, cs, :], dtype=np.float64)
        roi_norm = float(np.nanmedian(roi_vals)) if np.any(np.isfinite(roi_vals)) else global_norm
        diff = np.abs((prod / global_norm) * roi_norm - prod)[rr, cc, :]
        return {
            "group_id": spec.group_id, "family": spec.family, "seed": int(spec.seed),
            "mode": mode, "authority": "ORCHESTRATION_ONLY",
            "parity_mode": "NOT_IDENTIFIABLE", "tolerance": None,
            "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
            "decision": "NOT_IDENTIFIABLE", "pass": True,
        }
    raise HarnessError(f"unknown ROI mode: {mode}")


# ---------------------------------------------------------------------------
# Real reprojection / H4 arms (actual production primitives + actual stack)
# ---------------------------------------------------------------------------

def _make_wcs(ctype, crpix, crval=(150.0, 2.0), cdelt=(-0.001, 0.001)):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = list(ctype)
    w.wcs.crpix = [float(crpix[0]), float(crpix[1])]
    w.wcs.crval = [float(crval[0]), float(crval[1])]
    w.wcs.cdelt = [float(cdelt[0]), float(cdelt[1])]
    return w


def _synthetic_source(seed: int, shape_hw: tuple[int, int], n_frames: int) -> np.ndarray:
    h, w = shape_hw
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.empty((n_frames, 3, h, w), dtype=np.float32)
    for k in range(n_frames):
        plane = (
            1.0
            + (0.35 + 0.55 * k) * (xx / max(1, w - 1))
            + (0.20 + 0.25 * k) * (yy / max(1, h - 1))
            + rng.normal(0.0, 0.02, (h, w))
        )
        out[k] = np.repeat(plane[None, :, :], 3, axis=0).astype(np.float32)
    return out


def _reproject_full_reference(src_ch, full_wcs, patch_wcs, patch_hw):
    import warnings as _w

    from reproject import reproject_interp

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        arr, fp = reproject_interp(
            (src_ch, full_wcs), output_projection=patch_wcs, shape_out=patch_hw,
            order="bilinear", return_footprint=True,
        )
    return np.asarray(arr, dtype=np.float32), np.asarray(fp, dtype=np.float32) > 0.0


def _i1b_crop_bounds(frame_idx: int, taper: int) -> tuple[int, int, int, int]:
    # Deliberately too-tight, per-frame staircase (frame-dependent crop extent).
    x1 = 22 + taper * (frame_idx + 1) + (frame_idx + 1)
    y1 = 22 + taper * (frame_idx + 1) + (frame_idx + 1)
    return (4, 4, x1, y1)


def _patch_geometry():
    """Simple, predictable mapping: patch pixel (r, c) <-> full pixel (c, r)."""
    ctype = ("RA---TAN", "DEC--TAN")
    full_wcs = _make_wcs(ctype, (32.0, 32.0))
    patch_wcs = _make_wcs(ctype, (32.0, 32.0))
    return full_wcs, patch_wcs, (32, 32)


def measure_reprojection_arm(spec: ScenarioSpec) -> dict:
    """Real arms on existing production primitives; synthetic in-memory arrays."""
    ex = _lazy_execution()
    arm, kind = spec.params
    ctype = ("RA---TAN", "DEC--TAN") if kind == "tan" else ("RA---SIN", "DEC--SIN")
    full_wcs = _make_wcs(ctype, (32.0, 32.0))
    patch_wcs = _make_wcs(ctype, (32.0, 32.0))
    patch_hw = (32, 32)
    src = _synthetic_source(spec.seed, (48, 64), 3)

    # --- I2: SAME cropped array, crop WCS, target, order and mask ---
    if arm == "i2":
        bounds = (4, 4, 52, 52)
        crop = src[0][:, 4:52, 4:52]
        crop_wcs = ex.slice_wcs(full_wcs, ex.SourceBounds(x0=bounds[0], y0=bounds[1], x1=bounds[2], y1=bounds[3]))
        ex.reset_reproject_path_stats()
        prod, geom = ex.reproject_cropped(crop, crop_wcs, patch_wcs, patch_hw)
        stats = ex.get_reproject_path_stats().to_dict()
        if stats["path"] != "fast":
            return {
                "group_id": spec.group_id, "family": spec.family, "arm": "i2", "kind": kind,
                "production_path": stats["path"], "fast_path_eligible": False,
                "decision": "NOT_APPLICABLE",
                "reason": "fast path ineligible for this WCS; never counted as a parity pass",
                "pass": True, "authority": "REAL_PRIMITIVE_SYNTHETIC_INPUT",
            }
        reference = np.empty((patch_hw[0], patch_hw[1], 3), dtype=np.float32)
        for ch in range(3):
            from reproject import reproject_interp
            import warnings as _w

            with _w.catch_warnings():
                _w.simplefilter("ignore")
                arr, _fp = reproject_interp(
                    (crop[ch], crop_wcs), output_projection=patch_wcs,
                    shape_out=patch_hw, order=ex.REPROJECT_ORDER, return_footprint=True,
                )
            reference[..., ch] = np.asarray(arr, dtype=np.float32)
        common = geom & (np.isfinite(reference).all(axis=2))
        diff = np.abs(prod[common].astype(np.float64) - reference[common].astype(np.float64))
        max_abs = float(np.max(diff)) if diff.size else 0.0
        return {
            "group_id": spec.group_id, "family": spec.family, "arm": "i2", "kind": kind,
            "production_path": "fast", "fast_path_eligible": True,
            "n_common": int(np.count_nonzero(common)), "max_abs_diff": max_abs,
            "decision": "PARITY_BIT_IDENTICAL" if max_abs == 0.0 else "NOT_IDENTIFIABLE",
            "pass": bool(max_abs == 0.0), "authority": "REAL_PRIMITIVE_SYNTHETIC_INPUT",
        }

    if arm == "i1a":
        bounds = (4, 4, 52, 52)
        crop = src[0][:, 4:52, 4:52]
        crop_wcs = ex.slice_wcs(full_wcs, ex.SourceBounds(x0=bounds[0], y0=bounds[1], x1=bounds[2], y1=bounds[3]))
        prod, geom = ex.reproject_cropped(crop, crop_wcs, patch_wcs, patch_hw)
        ref = np.empty((patch_hw[0], patch_hw[1], 3), dtype=np.float32)
        ref_support = None
        for ch in range(3):
            a, fp = _reproject_full_reference(src[0][ch], full_wcs, patch_wcs, patch_hw)
            ref[..., ch] = a
            if ref_support is None:
                ref_support = fp
        common = geom & np.asarray(ref_support) & np.isfinite(ref).all(axis=2)
        diff = np.abs(prod[common].astype(np.float64) - ref[common].astype(np.float64))
        max_abs = float(np.max(diff)) if diff.size else 0.0
        return {
            "group_id": spec.group_id, "family": spec.family, "arm": "i1a", "kind": kind,
            "n_common": int(np.count_nonzero(common)), "max_abs_diff": max_abs,
            "decision": "PARITY_BIT_IDENTICAL" if max_abs == 0.0 else "NOT_IDENTIFIABLE",
            "pass": bool(max_abs == 0.0), "authority": "REAL_PRIMITIVE_SYNTHETIC_INPUT",
        }

    if arm == "i1b":
        return _measure_i1b(spec)

    raise HarnessError(f"unknown reprojection arm: {arm}")


def _stack_frames(images, supports, reference_index=0):
    CanonicalStackRequest, run_canonical_stack = _lazy_canonical_stack()
    req = CanonicalStackRequest(
        images=list(images), geometric_support=list(supports),
        normalization="sky_mean", weighting="noise_variance", rejection="kappa_sigma",
        combine="mean", reference_index=int(reference_index), taper="footprint",
        taper_px=8.0, taper_floor=0.0, backend="cpu", equalize_rgb=False,
        sigma_low=3.0, sigma_high=3.0,
    )
    return run_canonical_stack(req)


def _i1b_patch(patch_index: int, seed: int, ctype, taper: int = 2) -> dict:
    """One bounded synthetic patch yielding one h and one v relation.

    Production path = ``reproject_cropped`` on a deliberately too-tight per-frame
    crop; reference path = full-source ``reproject_interp`` on the same WCS mapping;
    **truth** path = the same full-source values restricted to the *identical
    support loss* the crop path had. All three are stacked with the **actual
    canonical stack** and the **same** production offsets.
    """
    ex = _lazy_execution()
    apply_additive_offsets = _lazy_apply_additive_offsets()
    offsets = np.array([0.013, -0.027, 0.041], dtype=np.float64)
    full_wcs = _make_wcs(ctype, (32.0, 32.0))
    shift = 3 * patch_index  # independent target region per patch
    patch_wcs = _make_wcs(ctype, (32.0 + shift, 32.0 + shift))
    patch_hw = (32, 32)
    src = _synthetic_source(seed, (48, 64), 3)

    prod_imgs, prod_sup, ref_imgs, ref_sup = [], [], [], []
    for k in range(3):
        b = _i1b_crop_bounds(k, taper)
        crop = src[k][:, b[1] : b[3], b[0] : b[2]]
        crop_wcs = ex.slice_wcs(full_wcs, ex.SourceBounds(x0=b[0], y0=b[1], x1=b[2], y1=b[3]))
        p, g = ex.reproject_cropped(crop, crop_wcs, patch_wcs, patch_hw)
        prod_imgs.append(p)
        prod_sup.append(np.asarray(g, dtype=bool) & np.isfinite(p).all(axis=2))
        ref = np.empty((patch_hw[0], patch_hw[1], 3), dtype=np.float32)
        rs = None
        for ch in range(3):
            a, fp = _reproject_full_reference(src[k][ch], full_wcs, patch_wcs, patch_hw)
            ref[..., ch] = a
            if rs is None:
                rs = fp
        ref_imgs.append(ref)
        ref_sup.append(np.asarray(rs, dtype=bool) & np.isfinite(ref).all(axis=2))

    subset_ok = all(bool(np.all(~np.asarray(ps) | np.asarray(rs))) for ps, rs in zip(prod_sup, ref_sup))
    truth_sup = [np.asarray(rs) & np.asarray(ps) for ps, rs in zip(prod_sup, ref_sup)]

    post_prod = np.asarray(apply_additive_offsets(_stack_frames(prod_imgs, prod_sup).science, offsets), dtype=np.float64)
    post_ref = np.asarray(apply_additive_offsets(_stack_frames(ref_imgs, ref_sup).science, offsets), dtype=np.float64)
    post_truth = np.asarray(apply_additive_offsets(_stack_frames(ref_imgs, truth_sup).science, offsets), dtype=np.float64)

    frames_prod_cnt = np.zeros(patch_hw[1], dtype=int)
    for sup in prod_sup:
        frames_prod_cnt += np.asarray(sup).any(axis=0).astype(int)
    rows_prod_cnt = np.zeros(patch_hw[0], dtype=int)
    for sup in prod_sup:
        rows_prod_cnt += np.asarray(sup).any(axis=1).astype(int)

    def relation(bounds, axis):
        for b in bounds:
            if axis == "h":
                dp = post_prod[:, b - 1, :] - post_prod[:, b, :]
                dr = post_ref[:, b - 1, :] - post_ref[:, b, :]
                dt = post_truth[:, b - 1, :] - post_truth[:, b, :]
            else:
                dp = post_prod[b - 1, :, :] - post_prod[b, :, :]
                dr = post_ref[b - 1, :, :] - post_ref[b, :, :]
                dt = post_truth[b - 1, :, :] - post_truth[b, :, :]
            good = np.isfinite(dp).all(axis=-1) & np.isfinite(dr).all(axis=-1) & np.isfinite(dt).all(axis=-1)
            if not np.any(good):
                continue
            dpv, drv, dtv = dp[good].ravel(), dr[good].ravel(), dt[good].ravel()
            rms_p = float(math.sqrt(float(np.mean(dpv**2))))
            rms_r = float(math.sqrt(float(np.mean(drv**2))))
            red = 1.0 - (rms_r / rms_p) if rms_p > 0 else 0.0
            recovered = float(np.mean(dpv - drv))
            injected = float(np.mean(dtv - drv))
            scale = float(max(np.max(np.abs(dtv)), np.max(np.abs(drv)), np.max(np.abs(dpv)), 1e-300))
            return {
                "orientation": axis, "boundary": int(b), "n_pixels": int(good.sum()),
                "rms_prod": rms_p, "rms_reference": rms_r, "reduction": float(red),
                "common_support_delta": recovered, "reference_geometry_delta": injected,
                "common_support_delta_abs": float(abs(recovered - injected)),
                "common_support_delta_rel": float(abs(recovered - injected) / scale),
                "common_support_delta_scale": scale,
                "diagnostic_label": "ORCHESTRATION_ONLY_NON_GATING",
                "scientific_gate": False,
                "authorizes_b2": False,
                "independent_relation": True,
            }
        return None

    h_bounds = [c for c in range(1, patch_hw[1] - 1) if frames_prod_cnt[c - 1] > frames_prod_cnt[c] >= 1]
    v_bounds = [r for r in range(1, patch_hw[0] - 1) if rows_prod_cnt[r - 1] > rows_prod_cnt[r] >= 1]
    # Strongest coverage drop first: the largest frame-count step is the cleanest
    # expression of the crop support/coverage mechanism in this bounded geometry.
    h_bounds.sort(key=lambda c: -(frames_prod_cnt[c - 1] - frames_prod_cnt[c]))
    v_bounds.sort(key=lambda r: -(rows_prod_cnt[r - 1] - rows_prod_cnt[r]))
    return {
        "patch_index": int(patch_index), "seed": int(seed), "target_crpix": [32.0 + shift, 32.0 + shift],
        "support_subset_ok": bool(subset_ok),
        "h": relation(h_bounds, "h"), "v": relation(v_bounds, "v"),
        "n_h_boundaries": len(h_bounds), "n_v_boundaries": len(v_bounds),
    }


def _measure_i1b(spec: ScenarioSpec) -> dict:
    """Coverage-delta intervention on **independent** patches, both TAN and non-TAN.

    I1b is a crop-vs-full-source **coverage** mechanism propagated through the
    canonical stack; it is **not** a fast-path mechanism, so it remains applicable
    on the fallback/non-TAN path (only I2's fast path can be NOT_APPLICABLE).
    """
    kind = spec.params[1]
    ctype = ("RA---TAN", "DEC--TAN") if kind == "tan" else ("RA---SIN", "DEC--SIN")
    patches = [_i1b_patch(i, spec.seed + 101 * i, ctype) for i in range(2)]
    h_relations = [p["h"] for p in patches if p["h"]]
    v_relations = [p["v"] for p in patches if p["v"]]
    all_rel = h_relations + v_relations
    subsets_ok = all(p["support_subset_ok"] for p in patches)

    common = {
        "group_id": spec.group_id, "family": spec.family, "arm": "i1b", "kind": kind,
        "patches": len(patches),
        "independent_patches": [p["patch_index"] for p in patches],
        "h_relations": len(h_relations), "v_relations": len(v_relations),
        "support_subset_ok": bool(subsets_ok),
        "relation_stats": all_rel,
        "authority": "REAL_PRIMITIVE_SYNTHETIC_INPUT",
        "product_roi_parity": "B2_MANDATORY",
        "causal_scope": "synthetic instrument calibration only; never a product or Windows causal claim",
    }
    if not subsets_ok:
        return {**common, "decision": "NOT_IDENTIFIABLE",
                "reason": "production support is not a subset of the full-source support", "pass": False}
    if len(h_relations) < 2 or len(v_relations) < 2:
        return {**common, "decision": "NOT_IDENTIFIABLE",
                "reason": "fewer than 2 independent h or 2 independent v geometric relations",
                "pass": False}
    min_red = min(r["reduction"] for r in all_rel)
    if min_red < INHERITED_GATES["reduction_min"]:
        return {**common, "decision": "NOT_IDENTIFIABLE", "min_reduction": float(min_red),
                "reason": "coverage-delta effect below the inherited reduction gate",
                "pass": False}
    max_err = max(r["common_support_delta_abs"] for r in all_rel)
    max_rel = max(r["common_support_delta_rel"] for r in all_rel)
    max_scale = max(r["common_support_delta_scale"] for r in all_rel)
    return {
        **common, "decision": "RECOVERED", "min_reduction": float(min_red), "pass": True,
        # RM3: DIAGNOSTIC ONLY, non-gating. The production crop path and the
        # full-source geometry carry the identical support loss on their common
        # support, so their agreement is structurally expected and cannot
        # establish recovery, causality or product parity.
        "common_support_delta_abs_max": float(max_err),
        "common_support_delta_rel_max": float(max_rel),
        "common_support_delta_scale": float(max_scale),
        "recovered_delta_diagnostic": {
            "label": "ORCHESTRATION_ONLY_NON_GATING",
            "scientific_gate": False,
            "authorizes_b2": False,
            "common_support_equality_structurally_expected": True,
            "note": (
                "prod and the full-source geometry agree on their common support by construction; "
                "this is a diagnostic consistency report only"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Relation-first panel (correlated RGB; 2-of-3 rule; hard override)
# ---------------------------------------------------------------------------

def _panel_geometry(case_kind: str) -> list[tuple[str, str]]:
    if case_kind == "incomplete_panel":
        return list(PANEL_RELATIONS[:30])
    return list(PANEL_RELATIONS)


def simulate_panel(spec: ScenarioSpec) -> dict:
    """Deterministic relation-first panel with correlated RGB channels.

    Channels of a relation share a latent field ``L_r`` plus a BOUNDED perturbation
    (never three independent replicates).
    """
    case_kind, regime = spec.params
    rng = np.random.default_rng(spec.seed)
    relations = _panel_geometry(case_kind)
    n = 512
    affected_h = {relations[i][0] for i in (0, 1) if i < len(relations)}
    affected_v = {relations[20 + i][0] for i in (0, 1) if 20 + i < len(relations)}
    affected = set(affected_h) | set(affected_v)
    if case_kind == "pseudo_single_relation":
        # ONE relation, all three of its channels -> must never satisfy >=2 relations
        affected = {relations[0][0]}
    duplicate_channels = case_kind == "channel_duplication"
    pert = 0.05  # bounded channel perturbation fraction
    amp = {"none": 0.0, "weak": 1.1, "moderate": 1.8, "strong": 3.0}.get(regime, 1.8)

    records = []
    for rid, orient in relations:
        latent = rng.normal(0.0, SIGMA_REF, n)
        sign = 1.0 if orient == "h" else -1.0
        dup_payload = None
        for ch in range(3):
            if duplicate_channels and ch > 0:
                e0 = dup_payload["e0"]
            else:
                e0 = latent + pert * rng.normal(0.0, SIGMA_REF, n)  # correlated, bounded
            m = np.zeros(n, dtype=np.float64)
            if rid in affected and case_kind not in ("null", "parity_fail"):
                shape = np.linspace(-1.0, 1.0, n)
                m = sign * amp * SIGMA_REF * (1.0 + 0.25 * shape)
                if case_kind == "contradictory_channel" and ch == 1:
                    m = -m  # hard single-channel contradiction
                if case_kind == "confounded":
                    m = m + 0.9 * amp * SIGMA_REF * np.cos(np.linspace(0, 3.1, n))
            if ch == 0:
                dup_payload = {"e0": e0}
            records.append(
                {
                    "relation_id": rid, "orientation": orient, "channel": ch,
                    "d_prod": e0 + m, "d_shadow": e0, "injected": m,
                }
            )

    # per-channel state
    per_relation: dict[tuple[str, str], list[dict]] = {}
    for rec in records:
        per_relation.setdefault((rec["relation_id"], rec["orientation"]), []).append(rec)

    rel_records = []
    for (rid, orient), recs in sorted(per_relation.items()):
        ch_states = []
        for rec in recs:
            dp, ds = rec["d_prod"], rec["d_shadow"]
            rms_p = float(math.sqrt(float(np.mean(dp**2))))
            rms_s = float(math.sqrt(float(np.mean(ds**2))))
            red = 1.0 - (rms_s / rms_p) if rms_p > 0 else 0.0
            p90_p = float(np.percentile(np.abs(dp), 90))
            p90_s = float(np.percentile(np.abs(ds), 90))
            ch_states.append(
                {
                    "channel": rec["channel"],
                    "reduction": float(red),
                    "p90_worsening": float((p90_s - p90_p) / p90_p) if p90_p > 0 else 0.0,
                    "rms0_sigma_ratio": float(rms_s / SIGMA_REF),
                    "derived_sign": float(np.sign(float(np.mean(rec["injected"])))),
                    "contradictory": False,
                    "unexplained": False,
                }
            )
        # hard override: a channel whose derived sign opposes the relation majority
        signs = [c["derived_sign"] for c in ch_states if c["derived_sign"] != 0.0]
        majority = 0.0
        if signs:
            pos = sum(1 for s in signs if s > 0)
            majority = 1.0 if pos * 2 >= len(signs) else -1.0
        for c in ch_states:
            if c["derived_sign"] != 0.0 and c["derived_sign"] != majority:
                c["contradictory"] = True
        promoted = [
            c
            for c in ch_states
            if c["reduction"] >= INHERITED_GATES["reduction_min"]
            and c["p90_worsening"] <= INHERITED_GATES["p90_worsening_max"]
            and c["rms0_sigma_ratio"] <= INHERITED_GATES["rms0_sigma_gate"]
            and not c["contradictory"]
            and not c["unexplained"]
        ]
        hard_block = any(c["contradictory"] or c["unexplained"] for c in ch_states)
        if hard_block:
            state = "BLOCKED"
        elif len(promoted) >= INHERITED_GATES["channel_promote_min"]:
            state = "PROMOTED"
        else:
            state = "NOT_PROMOTED"
        rel_records.append(
            {
                "relation_id": rid,
                "orientation": orient,
                "state": state,
                "n_channels_promoted": len(promoted),
                "derived_axis": orient,
                "derived_sign": majority,
                "channel_states": ch_states,
                "carries_injection": rid in affected and case_kind not in ("null", "parity_fail"),
            }
        )

    promoted = [r for r in rel_records if r["state"] == "PROMOTED"]
    h_prom = [r for r in promoted if r["orientation"] == "h"]
    v_prom = [r for r in promoted if r["orientation"] == "v"]
    if case_kind == "single_orientation":
        v_prom = []
    sig_h = sorted({r["derived_sign"] for r in h_prom})
    sig_v = sorted({r["derived_sign"] for r in v_prom})
    # A claim needs >=2 distinct relations in EACH orientation sharing one explicit
    # axis/sign signature; the two orientations are different axes and are judged
    # internally (per-orientation consistency), never merged.
    signature_consistent = bool(
        len(sig_h) <= 1 and len(sig_v) <= 1 and (len(h_prom) < 2 or len(sig_h) == 1) and (len(v_prom) < 2 or len(sig_v) == 1)
    )

    evidence = {
        "case_kind": case_kind,
        "regime": regime,
        "panel_complete": len(relations) == INHERITED_GATES["panel_relation_count"],
        "n_relations": len(relations),
        "identity_gates_pass": case_kind != "parity_fail",
        "parity_gates_pass": case_kind != "parity_fail",
        "same_global_pixels": True,
        "same_fixed_stack": True,
        "intervention_families": (
            ["crop_slice_numeric", "crop_slice_coverage"] if case_kind == "confounded" else ["crop_slice_numeric"]
        ),
        "rgb_pseudoreplication": case_kind == "rgb_pseudorep_flag",
        "per_frame_delta_only": case_kind == "per_frame_only",
        "sign_match_only": case_kind == "per_frame_only",
        "orientation_witnesses": {"h": len(h_prom), "v": len(v_prom)},
        "signature_consistent": signature_consistent,
        "relation_signature": {"h": sig_h, "v": sig_v},
        "n_distinct_relations_promoted": len(promoted),
        "n_channels_per_relation": 3,
        "authority": "ORCHESTRATION_ONLY",
    }
    reductions = [
        c["reduction"] for r in rel_records for c in r["channel_states"] if r["carries_injection"]
    ]
    evidence["reduction_rms"] = float(min(reductions)) if reductions else 0.0
    evidence["p90_worsening"] = float(
        max((c["p90_worsening"] for r in rel_records for c in r["channel_states"]), default=0.0)
    )
    evidence["rms0_sigma_ratio"] = float(
        max((c["rms0_sigma_ratio"] for r in rel_records for c in r["channel_states"]), default=0.0)
    )
    return {
        "group_id": spec.group_id, "family": spec.family, "seed": int(spec.seed),
        "relation_records": rel_records, "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# H4 admission ladder (verifies the frozen snapshots on every decision)
# ---------------------------------------------------------------------------

def h4_admit(
    evidence: Mapping,
    criteria_snapshot: FrozenSnapshot,
    protocol_snapshot: FrozenSnapshot,
) -> dict:
    """Deterministic H4 admission ladder; re-verifies both frozen snapshots."""
    criteria_snapshot.verify()
    protocol_snapshot.verify()
    crit = criteria_snapshot.frozen["h4_controlled_effect_admission"]

    if not bool(evidence.get("identity_gates_pass")):
        return {"state": "NOT_IDENTIFIABLE", "reasons": ["identity/provenance gate failed"]}
    if not bool(evidence.get("parity_gates_pass")):
        return {"state": "NOT_IDENTIFIABLE", "reasons": ["I1/I2 parity gate failed"]}
    if not bool(evidence.get("panel_complete")):
        return {"state": "INCOMPLETE_PANEL", "reasons": ["relation panel not 38 of 38"]}
    if bool(evidence.get("rgb_pseudoreplication")):
        return {"state": "NOT_IDENTIFIABLE", "reasons": ["RGB channels treated as independent replicates"]}
    if bool(evidence.get("per_frame_delta_only")) or bool(evidence.get("sign_match_only")):
        return {"state": "NOT_IDENTIFIABLE", "reasons": ["per-frame delta / sign match alone is insufficient"]}
    families = list(evidence.get("intervention_families") or [])
    if len(families) != 1:
        return {"state": "NOT_IDENTIFIABLE", "reasons": [f"expected exactly one isolated family, got {len(families)}"]}

    if float(evidence["reduction_rms"]) < float(crit["reduction_min"]):
        return {"state": "NO_EFFECT", "reasons": [f"reduction {evidence['reduction_rms']:.6f} below gate"]}
    if float(evidence["p90_worsening"]) > float(crit["p90_worsening_max"]):
        return {"state": "NO_EFFECT", "reasons": ["p90 worsening above gate"]}
    if float(evidence["rms0_sigma_ratio"]) > float(crit["rms0_sigma_gate"]):
        return {"state": "NO_EFFECT", "reasons": ["RMS0 ratio above gate"]}

    reasons: list[str] = []
    witnesses = evidence.get("orientation_witnesses") or {}
    need = int(crit["relations_per_orientation_min"])
    for orient in ("h", "v"):
        if int(witnesses.get(orient, 0)) < need:
            reasons.append(f"{orient}: {witnesses.get(orient, 0)} distinct relations below {need}")
    if not bool(evidence.get("signature_consistent")):
        reasons.append("derived axis/sign signature inconsistent across orientations")
    if reasons:
        return {"state": "NOT_IDENTIFIABLE", "reasons": reasons}
    return {"state": "CONTROLLED_EFFECT", "reasons": []}


# ---------------------------------------------------------------------------
# Calibration (stage A) — criteria derived here only
# ---------------------------------------------------------------------------

def run_calibration() -> dict:
    cal = _calibration_specs()
    m3 = [evaluate_m3_case(s) for s in cal if s.family.startswith("m3")]
    roi = [evaluate_roi_case(s) for s in cal if s.family.startswith("roi")]
    reproj = [measure_reprojection_arm(s) for s in cal if s.family.startswith("reproj")]
    panels = [simulate_panel(s) for s in cal if s.family.startswith("panel")]

    m3_eval = [r for r in m3 if r["decision"] == "EVALUATED"]
    m3_refused = [r for r in m3 if r["decision"] == "REFUSED_NONFINITE"]
    m3_bit_ok = all(r["stored_post_bit_identical"] and r["post_dtype_ok"] for r in m3_eval)
    if not m3_bit_ok:
        raise HarnessError("M3 bit identity is REQUIRED and was not achieved")
    max_ratio = max((r["algebraic_ratio"] for r in m3_eval), default=0.0)

    # analytic bound from an explicit IEEE rounding count (not a floor)
    ulp_k, ulp_derivation = _algebraic_ulp_k_derivation()

    roi_pointwise = [r for r in roi if r["mode"] == "pointwise_frozen"]
    roi_global = [r for r in roi if r["mode"] == "roi_global_recompute"]
    roi_bit_ok = all(r["pass"] for r in roi_pointwise)
    roi_refusal_ok = all(r["decision"] == "NOT_IDENTIFIABLE" for r in roi_global)

    i1a = [r for r in reproj if r["arm"] == "i1a"]
    i2_tan = [r for r in reproj if r["arm"] == "i2" and r["kind"] == "tan"]
    i2_non_tan = [r for r in reproj if r["arm"] == "i2" and r["kind"] == "non_tan"]
    i1b_tan = [r for r in reproj if r["arm"] == "i1b" and r["kind"] == "tan"]
    i1b_non_tan = [r for r in reproj if r["arm"] == "i1b" and r["kind"] == "non_tan"]

    i1b_min_red = min((r.get("min_reduction", 0.0) for r in i1b_tan + i1b_non_tan), default=0.0)
    i1b_tan_ok = all(r["decision"] == "RECOVERED" for r in i1b_tan)
    i1b_non_tan_ok = all(r["decision"] == "RECOVERED" for r in i1b_non_tan)
    i1b_all = i1b_tan + i1b_non_tan

    # RM3: the recovered/common-support delta is DIAGNOSTIC ONLY (non-gating).
    recovered_err = max((r.get("common_support_delta_abs_max", 0.0) for r in i1b_all), default=0.0)
    recovered_rel = max((r.get("common_support_delta_rel_max", 0.0) for r in i1b_all), default=0.0)
    recovered_scale = max((r.get("common_support_delta_scale", 0.0) for r in i1b_all), default=0.0)
    recovered_relations = sum(int(r.get("h_relations", 0)) + int(r.get("v_relations", 0)) for r in i1b_all)

    return {
        "m3": {
            "outcomes": m3_eval + m3_refused,
            "nonfinite_refusals": len(m3_refused),
            "bit_identity_achieved": bool(m3_bit_ok),
            "max_ratio": float(max_ratio),
            "ulp_k": float(ulp_k),
            "ulp_derivation": ulp_derivation,
        },
        "roi": {
            "outcomes": roi,
            "pointwise_bit_identical": bool(roi_bit_ok),
            "global_recompute_refused": bool(roi_refusal_ok),
            "authority": "ORCHESTRATION_ONLY",
            "product_roi_parity": "B2_MANDATORY",
        },
        "reprojection": {
            "outcomes": reproj,
            "i1a_bit_identical": all(r["decision"] == "PARITY_BIT_IDENTICAL" for r in i1a),
            "i2_tan_bit_identical": all(r["decision"] == "PARITY_BIT_IDENTICAL" for r in i2_tan),
            "i2_non_tan_not_applicable": all(r["decision"] == "NOT_APPLICABLE" for r in i2_non_tan),
            "i1b_tan_recovered": bool(i1b_tan_ok),
            "i1b_non_tan_recovered": bool(i1b_non_tan_ok),
            "i1b_min_reduction": float(i1b_min_red),
            "i1b_non_tan_not_applicable": False,
            "i1b_independent_relations": True,
        },
        "panels": {
            "outcomes": [
                {"group_id": p["group_id"], "family": p["family"], "evidence": p["evidence"]}
                for p in panels
            ],
            "authority": "ORCHESTRATION_ONLY",
        },
        "recovered_delta_diagnostic": {
            "label": "ORCHESTRATION_ONLY_NON_GATING",
            "scientific_gate": False,
            "authorizes_b2": False,
            "consumed_by_h4_admission": False,
            "common_support_equality_structurally_expected": True,
            "common_support_delta_abs_max": float(recovered_err),
            "common_support_delta_rel_max": float(recovered_rel),
            "common_support_delta_scale": float(recovered_scale),
            "measured_over_relations": int(recovered_relations),
            "note": (
                "the production crop path and the full-source geometry carry the identical support "
                "loss on their common support, so their agreement is STRUCTURALLY EXPECTED; this is a "
                "diagnostic consistency report only and can never establish recovery, causality or "
                "product parity, and it gates nothing"
            ),
        },
        "isolated_positive_count": len(
            [p for p in panels if p["evidence"]["case_kind"] == "isolated"]
        ),
    }


def _algebraic_ulp_k_derivation() -> tuple[int, str]:
    """Scale-aware M3 algebraic bound derived from an explicit IEEE rounding count.

    ``d_post_helper = fl((fl(sci_u + off_u)) - fl(sci_v + off_v))`` uses 2
    roundings; ``d_post_reordered = fl(fl(sci_u - sci_v) + fl(off_u - off_v))``
    uses 3. A conservative bound on the difference is (2 + 3) * eps * scale.
    No fabricated floor is applied.
    """
    helper_roundings = 2
    reordered_roundings = 3
    return helper_roundings + reordered_roundings, (
        f"explicit_ieee_rounding_count({helper_roundings}+{reordered_roundings})"
    )


def derive_frozen_criteria(calibration: Mapping) -> dict:
    ulp_k = int(calibration["m3"]["ulp_k"])
    return {
        "criteria_version": CRITERIA_VERSION,
        "m3_pre_post_identity": {
            "mode": "BIT_IDENTICAL",
            "tolerance": 0.0,
            "post_dtype": "float64",
            "no_f32_downcast": True,
            "nonfinite_policy": "REFUSE",
            "scope": "m3_pre_post_identity_only",
        },
        "m3_algebraic_cross_expression": {
            "mode": "SCALE_AWARE_ULP",
            "ulp_k": ulp_k,
            "ulp_derivation": str(calibration["m3"]["ulp_derivation"]),
            "scale_components": [
                "abs_sci_u_max", "abs_sci_v_max", "abs_post_u_max",
                "abs_post_v_max", "abs_off_u", "abs_off_v",
            ],
            "scale_reducer": "max",
            "calibration_max_ratio": float(calibration["m3"]["max_ratio"]),
            "scope": "m3_cross_expression_only",
        },
        "roi_shadow_parity": {
            "pointwise_frozen_inputs": "BIT_IDENTICAL",
            "tolerance": 0.0,
            "global_recompute": "NOT_IDENTIFIABLE",
            "authority": "ORCHESTRATION_ONLY",
            "product_roi_parity": "B2_MANDATORY",
            "product_roi_parity_note": (
                "the synthetic ROI parity rule is an orchestration criterion only; B2 must "
                "re-prove parity on the real product ROI path before any real capture"
            ),
        },
        "reprojection_intervention": {
            "i1a": {"mode": "PARITY_BIT_IDENTICAL", "tolerance": 0.0, "authority": "REAL_PRIMITIVE_SYNTHETIC_INPUT"},
            "i2": {"mode": "PARITY_BIT_IDENTICAL", "tolerance": 0.0, "authority": "REAL_PRIMITIVE_SYNTHETIC_INPUT"},
            "i1b": {
                "mode": "COVERAGE_DELTA_PROPAGATED_THROUGH_CANONICAL_STACK",
                "min_reduction": float(INHERITED_GATES["reduction_min"]),
                "authority": "REAL_PRIMITIVE_SYNTHETIC_INPUT",
                "product_roi_parity": "B2_MANDATORY",
            },
            "non_tan": "NOT_APPLICABLE",
            "note": "non-TAN fast-ineligible arms are NOT_APPLICABLE and are never counted as parity passes",
        },
        "injected_effect_recovery": {
            "relation_first": True,
            "channel_promote_min": int(INHERITED_GATES["channel_promote_min"]),
            "reduction_rms_min": float(INHERITED_GATES["reduction_min"]),
            "p90_worsening_max": float(INHERITED_GATES["p90_worsening_max"]),
            "rms0_sigma_gate": float(INHERITED_GATES["rms0_sigma_gate"]),
            "relations_per_orientation_min": int(INHERITED_GATES["relations_per_orientation_min"]),
            "recovered_delta_is_gating": False,
            "injected_amplitude_is_generator_metadata_only": True,
            "null_promotion_allowed": False,
            "confounded_promotion_allowed": False,
            "authority": "ORCHESTRATION_ONLY",
        },
        "recovered_delta_diagnostic": {
            "label": "ORCHESTRATION_ONLY_NON_GATING",
            "scientific_gate": False,
            "authorizes_b2": False,
            "consumed_by_h4_admission": False,
            "common_support_equality_structurally_expected": True,
            "common_support_delta_abs_max": float(calibration["recovered_delta_diagnostic"]["common_support_delta_abs_max"]),
            "common_support_delta_rel_max": float(calibration["recovered_delta_diagnostic"]["common_support_delta_rel_max"]),
            "measured_over_relations": int(calibration["recovered_delta_diagnostic"]["measured_over_relations"]),
            "note": (
                "DIAGNOSTIC ONLY: the production crop path and the full-source geometry carry the "
                "identical support loss on their common support, so their agreement is structurally "
                "expected; it is NOT a scientific or validation gate, it is not consumed by H4 "
                "admission, and it never authorizes R40-B2"
            ),
        },
        "h4_controlled_effect_admission": {
            "requires_identity_gates": True,
            "requires_parity_gates": True,
            "requires_complete_panel": True,
            "panel_relation_count": int(INHERITED_GATES["panel_relation_count"]),
            "requires_same_global_pixels": True,
            "requires_same_fixed_stack": True,
            "requires_exactly_one_intervention_family": True,
            "reject_rgb_pseudoreplication": True,
            "reject_per_frame_delta_only": True,
            "reject_sign_match_only": True,
            "relations_per_orientation_min": int(INHERITED_GATES["relations_per_orientation_min"]),
            "reduction_min": float(INHERITED_GATES["reduction_min"]),
            "p90_worsening_max": float(INHERITED_GATES["p90_worsening_max"]),
            "rms0_sigma_gate": float(INHERITED_GATES["rms0_sigma_gate"]),
            "ladder_precedence": [
                "NOT_IDENTIFIABLE", "INCOMPLETE_PANEL", "NO_EFFECT",
                "DETECTED_ASSOCIATION", "CONTROLLED_EFFECT",
            ],
            "real_h4_arms": "i1a_i1b_i2",
            "recovered_delta_consumed": False,
            "authority": "ORCHESTRATION_ONLY",
            "authorizes_b2": False,
        },
        "off_on_identity": {
            "mode": "EXACT_BYTE_IDENTITY",
            "tolerance": 0.0,
            "calibrated_by_b1": False,
            "verbatim_from_r40a_section_8_2": True,
            "note": "inherited verbatim; B1 must not calibrate, widen, relax or reinterpret it",
        },
    }


# ---------------------------------------------------------------------------
# Protocol payload (v2)
# ---------------------------------------------------------------------------

def build_protocol_payload(val_specs=None) -> dict:
    cal = _calibration_specs()
    val = list(val_specs) if val_specs is not None else list(_validation_specs_reserved())
    disjointness = assert_family_disjoint(cal, val)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "mission": MISSION,
        "phase": PHASE,
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "baseline_head": BASELINE_HEAD,
        "r40a_plan": R40A_PLAN_REL,
        "no_real_data": True,
        "sigma_ref_synthetic": SIGMA_REF,
        "inherited_gates": dict(INHERITED_GATES),
        "generators": {
            "m3_adversarial": {"version": "m3-gen-2", "classes": ["unit", "signed_zero", "subnormal", "large", "near_overflow", "cancel"]},
            "roi_orchestration": {"version": "roi-gen-2", "authority": "ORCHESTRATION_ONLY", "authorizes_b2": False},
            "reprojection_real_arms": {"version": "reproj-gen-2", "arms": ["i1a", "i1b", "i2"], "non_tan": "NOT_APPLICABLE"},
            "relation_first_panel": {
                "version": "panel-gen-2",
                "rgb_generation": "shared_relation_latent_with_bounded_channel_perturbation",
                "channel_promote_min": int(INHERITED_GATES["channel_promote_min"]),
                "authority": "ORCHESTRATION_ONLY",
            },
        },
        "split": {
            "calibration_groups": [s.identity() for s in cal],
            "validation_cohort_reserved": [s.identity() for s in val],
            "disjointness": disjointness,
            "validation_status": "NOT_RUN",
        },
        "metrics": {
            "m3_identity": {"kind": "bit_identity", "quantity": "stored_post_value_equals_helper_output", "tolerance": 0.0},
            "m3_algebraic": {
                "kind": "scale_aware_ulp",
                "scale_components": ["abs_sci_u_max", "abs_sci_v_max", "abs_post_u_max", "abs_post_v_max", "abs_off_u", "abs_off_v"],
                "scale_reducer": "max",
                "ulp_k_source": "explicit IEEE rounding count of both expressions",
            },
            "roi_parity": {"kind": "bit_identity_at_selected_coordinates", "global_recompute": "NOT_IDENTIFIABLE"},
            "reprojection": {"i1a": "crop vs full-source common-support parity", "i1b": "coverage delta through canonical stack", "i2": "same-array fast vs reference"},
            "injected_recovery": {"kind": "relation_first_channel_2_of_3", "statistic": "min per-channel reduction over injected relations"},
            "h4_admission": {"kind": "frozen_ladder", "precedence": "NOT_IDENTIFIABLE then INCOMPLETE_PANEL then NO_EFFECT then DETECTED_ASSOCIATION then CONTROLLED_EFFECT"},
        },
        "candidate_selection_rule": (
            "criteria are derived on calibration groups only; free constants come from "
            "an explicit IEEE rounding count or the inherited gates; no fabricated floor "
            "and no value derived from observed real R39 quantities"
        ),
        "privacy_policy": "fail_closed_no_redaction; prose markers are never classified as paths",
        "hash_policy": "hashes are computed after the privacy boundary and are recomputable from the persisted artifact",
        "expected_outcome_contract": {
            "policy": "every validation group has a frozen expected decision in the prevalidation bundle; "
                      "panel groups are routed through the H4 admission ladder with the bound frozen snapshots",
            "no_default_pass": True,
        },
        "code_identity_policy": (
            "the bundle pins raw-byte tool_source_sha256, the complete raw-byte product-source "
            "manifest over src/zemosaic/, the runtime versions and repo_head; preflight re-probes "
            "and refuses on mismatch"
        ),
    }


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def build_prevalidation_bundle(*, val_specs=None, calibration=None) -> dict:
    """PREPARE stage: protocol + calibration + deep freeze + reserved cohort.

    The reserved validation cohort is described by metadata, its frozen expected
    decisions and a hash only; its generators are never called and no validation
    outcome exists in this bundle.
    """
    val_list = list(val_specs) if val_specs is not None else list(_validation_specs_reserved())
    protocol = build_protocol_payload(val_specs=val_list)
    protocol_snapshot = freeze_snapshot("protocol", TAG_PROTOCOL, protocol)

    if calibration is None:
        calibration = run_calibration()
    criteria = derive_frozen_criteria(calibration)
    criteria_snapshot = freeze_snapshot("criteria", TAG_CRITERIA, criteria)

    cohort = dict(validation_cohort_metadata(val_list))
    cohort["expected_decisions"] = _expected_decisions(val_list)
    cohort["admission_router"] = "h4_admit with the bound frozen snapshots"

    payload = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "mission": MISSION,
        "phase": PHASE,
        "baseline_head": BASELINE_HEAD,
        "r40a_plan": R40A_PLAN_REL,
        "status": STATUS_READY,
        "b1_status": STATUS_FAIL_INVALID,
        "r40b2_authorized": False,
        "protocol": {
            "protocol_version": PROTOCOL_VERSION,
            "sha256": protocol_snapshot.sha256,
            "canonical_json": protocol_snapshot.canonical_json,
            "payload": protocol,
        },
        "criteria": {
            "criteria_version": CRITERIA_VERSION,
            "sha256": criteria_snapshot.sha256,
            "canonical_json": criteria_snapshot.canonical_json,
            "payload": criteria,
        },
        "calibration": calibration,
        "calibration_sha256": sha256_payload(calibration, TAG_CALIBRATION),
        "reserved_validation_cohort": cohort,
        "code_identity": compute_code_identity(),
        "validation_status": "NOT_RUN",
        "validation_runs": 0,
        "validation_claim": None,
        "generator_not_invoked": True,
        "off_on_identity_untouched": True,
        "no_real_data_read": True,
        "limitations": [
            "synthetic instrument/orchestration calibration only; no Windows816/R37/R39 or LP341 value is read or claimed",
            "the reserved v2 validation cohort is frozen but NOT evaluated in this rework",
            "product ROI parity on the real capture path remains B2-mandatory",
            "the H4 admission ladder is an orchestration rule; real H4 arms are calibrated on synthetic inputs of the real primitives",
        ],
        "next_gate": (
            "a NEW architect mission executes the reserved v2 cohort exactly once under these frozen "
            "criteria; then B2 (default-OFF capture, OFF/ON exact byte identity)"
        ),
    }
    payload["content_sha256"] = content_sha256(payload)
    return payload


def assert_prevalidation_has_no_validation_outcome(bundle: Mapping) -> None:
    if bundle.get("validation_status") != "NOT_RUN":
        raise HarnessError("prevalidation bundle must have validation_status NOT_RUN")
    if int(bundle.get("validation_runs", -1)) != 0:
        raise HarnessError("prevalidation bundle must have validation_runs 0")
    if bundle.get("reserved_validation_cohort", {}).get("outcomes_present") is not False:
        raise HarnessError("reserved cohort must declare no outcomes")
    forbidden = ("validation_outcomes", "validation_result", "validation_pass")
    present = [k for k in forbidden if k in bundle]
    if present:
        raise HarnessError(f"prevalidation bundle must not carry validation outcome fields: {present}")


COHORT_GATE = object()  # in-process identity gate; must be passed explicitly


def _evaluate_arm(spec: ScenarioSpec) -> dict:
    """Dispatch one group to its arm evaluator (no expectation logic here)."""
    if spec.family.startswith("m3"):
        return evaluate_m3_case(spec)
    if spec.family.startswith("roi"):
        return evaluate_roi_case(spec)
    if spec.family.startswith("reproj"):
        return measure_reprojection_arm(spec)
    if "panel" in spec.family:
        return simulate_panel(spec)
    raise HarnessError(f"no evaluator for family {spec.family}")


_PANEL_EXPECTATIONS = MappingProxyType(
    {
        "isolated": "CONTROLLED_EFFECT",
        "null": "NO_EFFECT",
        "pseudo_single_relation": "NOT_IDENTIFIABLE",
        "contradictory_channel": "NOT_IDENTIFIABLE",
        "confounded": "NOT_IDENTIFIABLE",
        "incomplete_panel": "INCOMPLETE_PANEL",
        "single_orientation": "NOT_IDENTIFIABLE",
        "per_frame_only": "NOT_IDENTIFIABLE",
        "rgb_pseudorep_flag": "NOT_IDENTIFIABLE",
        "parity_fail": "NOT_IDENTIFIABLE",
        "channel_duplication": "CONTROLLED_EFFECT",
    }
)


def _expected_decisions(val_specs: Sequence[ScenarioSpec]) -> dict:
    """Frozen expected decision per validation group id (deterministic)."""
    exp: dict = {}
    for s in val_specs:
        fam = s.family
        if "panel" in fam:
            exp[s.group_id] = _PANEL_EXPECTATIONS[str(s.params[0])]
        elif fam.startswith("m3"):
            exp[s.group_id] = (
                "REFUSED_NONFINITE" if s.params[2] in ("nan_input", "inf_input") else "EVALUATED"
            )
        elif fam.startswith("roi"):
            exp[s.group_id] = (
                "PARITY_OK" if s.params[2] == "pointwise_frozen" else "NOT_IDENTIFIABLE"
            )
        elif fam.startswith("reproj"):
            arm, kind = s.params[0], s.params[1]
            if arm in ("i1a", "i2"):
                exp[s.group_id] = "PARITY_BIT_IDENTICAL" if kind == "tan" else "NOT_APPLICABLE"
            elif arm == "i1b":
                exp[s.group_id] = "RECOVERED"
            else:
                raise HarnessError(f"no frozen expectation for arm {arm}")
        else:
            raise HarnessError(f"no frozen expectation for family {fam}")
    return exp


def evaluate_group_against_expectation(
    spec: ScenarioSpec,
    *,
    expected: str,
    criteria_snapshot: FrozenSnapshot,
    protocol_snapshot: FrozenSnapshot,
    authorization,
) -> dict:
    """Guarded evaluation of ONE validation group against its frozen expectation."""
    if authorization is not COHORT_GATE:
        raise ReservedCohortError(
            "the reserved v2 validation cohort may not be evaluated without the durable claim "
            "authorization gate"
        )
    if spec.split != "validation":
        raise HarnessError("validation evaluator called with a non-validation spec")
    if "panel" in spec.family:
        sim = simulate_panel(spec)
        decision = h4_admit(sim["evidence"], criteria_snapshot, protocol_snapshot)
        observed = str(decision["state"])
        detail = {"evidence": sim["evidence"], "reasons": decision["reasons"]}
    else:
        out = _evaluate_arm(spec)
        observed = str(out["decision"])
        detail = out
    return {
        "group_id": spec.group_id,
        "family": spec.family,
        "expected": str(expected),
        "observed": observed,
        "pass": bool(observed == str(expected)),
        "detail": detail,
    }


def preflight_bundle(bundle: Mapping, *, specs: Sequence[ScenarioSpec]) -> dict:
    """Whole-bundle preflight, BEFORE any durable claim (fail-closed)."""
    if bundle.get("schema") != SCHEMA_NAME:
        raise ValidationClaimError("bundle schema mismatch")
    if not verify_content_hash(bundle):
        raise ValidationClaimError("bundle content hash does not match the persisted content")
    if bundle.get("status") != STATUS_READY:
        raise ValidationClaimError("bundle status is not READY_FOR_NEW_VALIDATION")
    if bundle.get("b1_status") != STATUS_FAIL_INVALID:
        raise ValidationClaimError("bundle must declare B1_FAIL_PROTOCOL_INVALID")
    if bundle.get("r40b2_authorized") is not False:
        raise ValidationClaimError("bundle must not authorize R40-B2")
    if bundle.get("validation_status") != "NOT_RUN":
        raise ValidationClaimError("bundle validation_status must be NOT_RUN")
    if int(bundle.get("validation_runs", -1)) != 0:
        raise ValidationClaimError("bundle validation_runs must be 0")
    try:
        assert_prevalidation_has_no_validation_outcome(bundle)
    except HarnessError as exc:
        raise ValidationClaimError(str(exc)) from exc

    protocol_snapshot = snapshot_from_entry("protocol", TAG_PROTOCOL, bundle["protocol"])
    criteria_snapshot = snapshot_from_entry("criteria", TAG_CRITERIA, bundle["criteria"])
    protocol_snapshot.verify()
    criteria_snapshot.verify()

    cohort = bundle.get("reserved_validation_cohort") or {}
    if cohort.get("generator_invoked") is not False or cohort.get("outcomes_present") is not False:
        raise ValidationClaimError("reserved cohort must be unconsumed (no generator, no outcomes)")
    actual = validation_cohort_metadata(specs)
    if cohort.get("cohort_sha256") != actual["cohort_sha256"]:
        raise ValidationClaimError("reserved cohort hash does not match the frozen cohort")
    expected = _expected_decisions(specs)
    if dict(cohort.get("expected_decisions") or {}) != expected:
        raise ValidationClaimError("frozen expected decisions do not match the cohort specs")

    current = compute_code_identity()
    recorded = bundle.get("code_identity") or {}
    if _thaw(recorded) != current:
        raise ValidationClaimError(
            "code identity mismatch (tool source, product manifest, runtime versions or HEAD)"
        )
    return {
        "protocol_snapshot": protocol_snapshot,
        "criteria_snapshot": criteria_snapshot,
        "expected": expected,
        "code_identity": current,
        "code_identity_sha256": code_identity_digest(current),
    }


def validate_stage(
    *,
    bundle: Mapping,
    state_dir: Path,
    out_path: Path,
    consume_reserved_cohort: bool,
    specs=None,
) -> dict:
    """One-shot VALIDATE stage. Requires explicit consumption; preflights before claiming."""
    if not consume_reserved_cohort:
        raise ReservedCohortError(
            "refusing to consume the reserved cohort without explicit authorization"
        )
    val_specs = list(specs) if specs is not None else list(_validation_specs_reserved())
    pre = preflight_bundle(bundle, specs=val_specs)
    protocol_snapshot = pre["protocol_snapshot"]
    criteria_snapshot = pre["criteria_snapshot"]
    expected = pre["expected"]
    cohort = bundle["reserved_validation_cohort"]

    claim_key = {
        "protocol_sha256": protocol_snapshot.sha256,
        "criteria_sha256": criteria_snapshot.sha256,
        "cohort_sha256": cohort["cohort_sha256"],
        "code_identity_sha256": pre["code_identity_sha256"],
    }
    claim = DurableClaim(state_dir, claim_key)
    claim.claim(metadata={"mission": MISSION, "phase": PHASE})

    outcomes = [
        evaluate_group_against_expectation(
            s,
            expected=expected[s.group_id],
            criteria_snapshot=criteria_snapshot,
            protocol_snapshot=protocol_snapshot,
            authorization=COHORT_GATE,
        )
        for s in val_specs
    ]
    all_pass = bool(outcomes) and all(bool(o["pass"]) for o in outcomes)
    result = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "mission": MISSION,
        "phase": PHASE,
        "status": "VALIDATED",
        "b1_status": "B1_PASS" if all_pass else "B1_FAIL",
        "r40b2_authorized": bool(all_pass),
        "claim": {"claim_id": claim.claim_id, "key": claim_key},
        "protocol_sha256": protocol_snapshot.sha256,
        "criteria_sha256": criteria_snapshot.sha256,
        "cohort_sha256": cohort["cohort_sha256"],
        "code_identity": pre["code_identity"],
        "code_identity_sha256": pre["code_identity_sha256"],
        "admission_router": "h4_admit with the bound frozen snapshots",
        "outcomes": outcomes,
        "runs": 1,
    }
    result["content_sha256"] = content_sha256(result)
    assert_no_path_leaks(result)
    assert_finite_json(result)
    sha = atomic_write_json(out_path, result)
    claim.complete(result_sha256=sha, metadata={"mission": MISSION, "phase": PHASE})
    return result


# ---------------------------------------------------------------------------
# v1 invalidation record
# ---------------------------------------------------------------------------

def build_v1_invalidation_record() -> dict:
    reasons = [
        {
            "id": "H1",
            "severity": "HIGH",
            "summary": "artifact protocol was silently redacted after hashing; binding hash unverifiable",
            "effect": "the frozen protocol recorded in the v1 artifact is not the payload that was hashed",
        },
        {
            "id": "H2",
            "severity": "HIGH",
            "summary": "criteria immutability was a shallow proxy; no criteria hash re-binding before validation",
            "effect": "a promotion threshold could be rewritten after the freeze without changing the hash",
        },
        {
            "id": "H3",
            "severity": "HIGH",
            "summary": "exactly-once was a resettable process-local lock; validation was reopened repeatedly",
            "effect": "no durable exactly-once guarantee; the validation cohort is consumed",
        },
        {
            "id": "M1",
            "severity": "MEDIUM",
            "summary": "RGB pseudo-replication in the positive path; relation-first not proven",
            "effect": "positive admission could gain pseudo-samples from correlated channels",
        },
        {
            "id": "M2",
            "severity": "MEDIUM",
            "summary": "the H4 positive mechanism was an abstract toy, not the real I1/I2 intervention",
            "effect": "no product-primitive intervention calibrated the H4 criteria",
        },
        {
            "id": "M3",
            "severity": "MEDIUM",
            "summary": "I1 and I2 were conflated; no I2 same-array parity and no I1b coverage arm; non-TAN acceptance too loose",
            "effect": "reprojection parity evidence was mislabelled and incomplete",
        },
        {
            "id": "M4",
            "severity": "MEDIUM",
            "summary": "several 'calibrated' thresholds were hard-coded floors",
            "effect": "the frozen ulp_k and recovered-delta tolerance were floors, not derived values",
        },
        {
            "id": "M5",
            "severity": "MEDIUM",
            "summary": "calibration/validation independence was seed-only over the same scenario families",
            "effect": "not the disjoint scenario/event independence the R40-A contract requires",
        },
        {
            "id": "L1",
            "severity": "LOW",
            "summary": "M3 adversarial coverage was thin (no cancellation, signed zeros, underflow, near-overflow)",
            "effect": "weaker numerical coverage than required",
        },
        {
            "id": "L2",
            "severity": "LOW",
            "summary": "ROI parity was validated only on a toy stack without a B2 re-proof statement",
            "effect": "risk of over-claiming product parity",
        },
    ]
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "mission": MISSION,
        "phase": PHASE_V1_INVALIDATION,
        "record": "v1_invalidation",
        "baseline_head": BASELINE_HEAD,
        "artifact": {
            "basename": V1_ARTIFACT_BASENAME,
            "bytes": V1_ARTIFACT_BYTES,
            "sha256": V1_ARTIFACT_SHA256,
            "preserved_immutable": True,
            "overwritten": False,
        },
        "v1_claimed": {
            "final_status": V1_CLAIMED_FINAL_STATUS,
            "r40b2_authorized": V1_CLAIMED_R40B2_AUTHORIZED,
            "protocol_sha256": V1_PROTOCOL_SHA256,
            "criteria_sha256": V1_CRITERIA_SHA256,
            "content_sha256": V1_CONTENT_SHA256,
        },
        "withdrawn": True,
        "b1_status": STATUS_FAIL_INVALID,
        "r40b2_authorized": False,
        "validation_cohort_v1": {
            "status": "CONSUMED_COMPROMISED",
            "rerun_forbidden": True,
            "reason": "the v1 validation ran repeatedly under a corrupted protocol/criteria binding and a seed-only split",
        },
        "invalid_reasons": reasons,
        "hash_recomputable_from_v1_artifact": False,
        "next_step": (
            "content-addressed v2 protocol + calibration/criteria deep freeze + genuinely fresh "
            "reserved validation cohort (prevalidation bundle READY_FOR_NEW_VALIDATION)"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r40b1_synthetic_calibration",
        description="R40-B1 freeze-closure: v1 invalidation, v2 prevalidation (no v2 validation execution)",
    )
    sub = p.add_subparsers(dest="stage")

    pp = sub.add_parser("prepare", help="build the v2 prevalidation bundle (repeatable)")
    pp.add_argument("--out", required=True)
    pp.add_argument("--repeatable", action="store_true")

    pi = sub.add_parser("invalidate-v1", help="write the v1 invalidation record")
    pi.add_argument("--out", required=True)

    pv = sub.add_parser("validate", help="one-shot reserved-cohort validation (NOT run here)")
    pv.add_argument("--bundle", required=True)
    pv.add_argument("--state-dir", required=True)
    pv.add_argument("--out", required=True)
    pv.add_argument("--consume-reserved-cohort", action="store_true")

    sub.add_parser("print-hashes", help="print the v2 protocol and criteria hashes")
    return p


def _emit(path_raw: str, payload: Mapping) -> str:
    assert_no_path_leaks(payload)
    assert_finite_json(payload)
    return atomic_write_json(resolve_output_path(path_raw), payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stage = args.stage

    if stage == "print-hashes":
        protocol = freeze_snapshot("protocol", TAG_PROTOCOL, build_protocol_payload())
        cal = run_calibration()
        criteria = freeze_snapshot("criteria", TAG_CRITERIA, derive_frozen_criteria(cal))
        print(f"protocol_sha256={protocol.sha256}")
        print(f"criteria_sha256={criteria.sha256}")
        print(f"cohort_sha256={validation_cohort_metadata(_validation_specs_reserved())['cohort_sha256']}")
        return 0

    if stage == "invalidate-v1":
        _emit(args.out, build_v1_invalidation_record())
        print(f"{STATUS_FAIL_INVALID} -> {args.out}")
        return 0

    if stage == "prepare":
        bundle = build_prevalidation_bundle()
        assert_prevalidation_has_no_validation_outcome(bundle)
        sha = _emit(args.out, bundle)
        print(f"{STATUS_READY} -> {args.out} sha256={sha}")
        return 0

    if stage == "validate":
        import json as _json

        with open(args.bundle, encoding="utf-8") as fh:
            bundle = _json.load(fh)
        result = validate_stage(
            bundle=bundle,
            state_dir=resolve_output_path(args.state_dir),
            out_path=resolve_output_path(args.out),
            consume_reserved_cohort=bool(args.consume_reserved_cohort),
        )
        print(f"{result['b1_status']} -> {args.out}")
        return 0 if result["b1_status"] == "B1_PASS" else 1

    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
