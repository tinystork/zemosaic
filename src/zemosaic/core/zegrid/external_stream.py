"""ZM-ZEGRID-AUTO-EXTERNAL-STREAMING M2 — exact external working set.

One Cell, one spatial tile: spool the per-frame PREPARED tile values to disk,
then reduce them with a column micro-batch that keeps the peak working set
``O(column_batch x N) + O(tile_area)`` instead of ``O(N x tile_area)``.

Why this is exact (not an approximation)
----------------------------------------
The canonical phase-2 math has two independent axes:

* per-frame preparation (normalization coefficients, validity, footprint
  taper) — already computed per frame by the RAM path;
* per-pixel-channel reduction across ALL N frames (sigma clip, winsor clip,
  mean/median combine, support) — every primitive is column-wise independent.

The spool stores, for every frame, the SAME float32 prepared tile the RAM path
builds (``images_t``: NaN outside the valid mask) and the float32 per-pixel
taper factor ``valid ? taper : 0``. The reducer reads ALL N frames for a
contiguous row range of the tile interior (never a subset of N) and calls the
same rejection/combine primitives on that slice. Because those primitives are
independent per column, slicing the column axis is bit-exact; slicing the N axis
would not be (quantiles/medians need the full N axis) and is never done.

The estimator prices every term explicitly and names the unknowns (filesystem
block rounding, metadata/journal amplification, page-cache residency) instead of
returning zero for them.

Scope (M2 only)
---------------
New module; nothing in the product path calls it. No scheduler/admission/GUI
change, no default change, no science/halo/floor/GPU change. The two small
``canonical_streaming`` helpers (``_finalize_phase2_outputs`` and
``finalize_streaming_result``) are pure extractions of existing code so the
external executor builds byte-identical results without duplicating science.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zemosaic.core.canonical_engine import (
    CanonicalStackRequest,
    CanonicalStackResult,
)
from zemosaic.core.canonical_stacking import (
    CanonicalStackValidationError,
    _as_host_numpy,
    _astype_float32,
    _axis_sum_f64,
    _get_xp,
    _kappa_sigma_step,
    _nan_axis_median,
    _resolve_backend,
    _safe_divide,
    _validate_max_iters,
    _validate_reject_sigma,
    _validate_winsor_limit,
    _winsorized_sigma_clip_step,
)
from zemosaic.core.canonical_support import make_footprint_taper
from zemosaic.core.canonical_streaming import (
    _COMBINE_METHODS,
    _Phase1,
    _fixed_phase1,
    _finalize_phase2_outputs,
    _iter_tiles,
    _resolve_method_token,
    _resolve_reject_token,
    _resolve_taper,
    _tile_dims,
    _validate_rejection_sink,
    finalize_streaming_result,
)
from zemosaic.core.zegrid.gpu import is_gpu_runtime_error

__all__ = [
    "SPOOL_SCHEMA",
    "SPOOL_MANIFEST_NAME",
    "SPOOL_OWNER_NAME",
    "COLUMN_BATCH_MIN",
    "REDUCTION_BYTES_PER_POSITION_FRAME",
    "EXTERNAL_BATCH_BYTES_PER_POSITION_FRAME",
    "EXTERNAL_BATCH_SAMPLE_BYTES_PER_POSITION_FRAME",
    "EXTERNAL_BATCH_FIXED_POSITION_BYTES",
    "external_batch_bytes",
    "external_equalize_peak_bytes_per_pixel",
    "EXTERNAL_BATCH_COMMON_BYTES_PER_POSITION_FRAME",
    "EXTERNAL_BATCH_REJECTION_BYTES_PER_POSITION_FRAME",
    "EXTERNAL_BATCH_COMBINE_BYTES_PER_POSITION_FRAME",
    "EXTERNAL_BATCH_WORST_BYTES_PER_POSITION_FRAME",
    "FINALIZE_WORKING_BYTES_PER_PIXEL",
    "EQUALIZE_RGB_BYTES_PER_PIXEL",
    "EXTERNAL_HOST_MODEL_SCHEMA",
    "RESULT_PLANE_BYTES_PER_POSITION",
    "RESULT_SUPPORT_BYTES_PER_PIXEL",
    "RESULT_N_EFF_BYTES_PER_PIXEL",
    "REJECTION_MASK_BYTES_PER_POSITION_FRAME",
    "WRITE_TILE_BYTES_PER_PIXEL_CHANNEL",
    "WRITE_TILE_AUX_BYTES_PER_PIXEL",
    "external_cell_host_components",
    "ExternalSpoolError",
    "ExternalSpoolPathError",
    "ExternalSpoolStaleError",
    "ExternalSpoolCorruptError",
    "ExternalSpoolWriteError",
    "ExternalDiskShortfall",
    "ExternalSpoolLiveOwnerError",
    "ExternalSpoolCommittedError",
    "SPOOL_STATE_WRITING",
    "SPOOL_STATE_READY",
    "SPOOL_STATE_COMMITTED",
    "classify_attempt",
    "discard_stale_for_cell",
    "cleanup_after_commit",
    "scan_spool_root",
    "ExternalBudgetError",
    "ExternalSpoolSpec",
    "TileSpool",
    "ExternalSpool",
    "validate_tile",
    "estimate_external_working_set",
    "largest_admissible_column_batch",
    "shrink_frozen_column_batch",
    "reduce_spooled_tile",
    "run_cell_streaming_external",
]

SPOOL_SCHEMA = "zemosaic-zegrid-external-spool-v1"
SPOOL_MANIFEST_NAME = "manifest.json"
SPOOL_OWNER_NAME = "ATTEMPT"
IMAGES_NAME = "images.npy"
WEIGHTS_NAME = "weights.npy"

#: Transactional attempt states (M4). The spool is never reusable as a
#: completed Cell until the durable store commit exists; the caller marks it.
SPOOL_STATE_WRITING = "writing"
SPOOL_STATE_READY = "ready"
SPOOL_STATE_COMMITTED = "committed"
SPOOL_STATES = (SPOOL_STATE_WRITING, SPOOL_STATE_READY, SPOOL_STATE_COMMITTED)

#: RESULT planes per (pixel x channel) of one Cell: science f32 (4) +
#: estimator_weight_sum f64 (8) + valid_mask bool (1) + surviving int64 (8).
#: These are allocated over the FULL cell patch before the tile loop and held
#: until the durable commit releases them.
RESULT_PLANE_BYTES_PER_POSITION = 4 + 8 + 1 + 8  # 21
#: support_w1 f64 + support_w2 f64, per patch pixel.
RESULT_SUPPORT_BYTES_PER_PIXEL = 8 + 8  # 16
#: n_eff_support f64, per patch pixel, allocated at finalization.
RESULT_N_EFF_BYTES_PER_PIXEL = 8
#: Dense rejection mask bool per (frame x pixel x channel), allocated only when
#: NO bounded rejection sink is supplied. The product external path ALWAYS
#: supplies ``BoundedCountingRejectionSink``, so this term is 0 there.
REJECTION_MASK_BYTES_PER_POSITION_FRAME = 1

#: Spool-WRITE tile preparation, per tile pixel x channel: raw provider tile
#: f32 (4) + src f64 (8) + transformed f64 (8) + transformed f32 (4) +
#: output frame f32 (4) = 28c.
WRITE_TILE_BYTES_PER_PIXEL_CHANNEL = 4 + 8 + 8 + 4 + 4  # 28
#: Spool-WRITE taper/support auxiliaries per tile pixel (NOT per channel):
#: finite/pre/post bools (3) + taper f32 (4) + distance-transform f64 (8) +
#: frac f32 (4) + np.where f64 (8) + taper output f32 (4) = 31.
WRITE_TILE_AUX_BYTES_PER_PIXEL = 3 + 4 + 8 + 4 + 8 + 4  # 31

#: Common reduction-batch arrays per (position x frame), simultaneously live:
#:   img f32 4 + weights f64 8 + orig f64 8 + wmap f64 8
#:   finite/initial/survivor/rejection/eligible/degenerate bools 6
EXTERNAL_BATCH_COMMON_BYTES_PER_POSITION_FRAME = 4 + 8 + 8 + 8 + 6  # 34
#: Additional live bytes for each rejection/combine variant (rejection and
#: combine are SEQUENTIAL; the worst variant, not their sum, is added).
#:   kappa: masked 8 + median sort copy 8 + interval/equality/keep/degenerate/new2
#:          masks 5 = 21? -> conservative 24
#:   winsor: masked 8 + winsor 8 + quantile sort copy 8 + interval/equality/keep/
#:          degenerate/new2 masks 5 = 29 -> conservative 32
#:   mean: masked 8 + w_contrib 8 + product 8 = 24
#:   median: masked 8 + median sort copy 8 + output 8 + count/masks 2 = 26
EXTERNAL_BATCH_REJECTION_BYTES_PER_POSITION_FRAME = {
    "none": 0,
    "kappa_sigma": 24,
    "winsorized_sigma_clip": 32,
}
EXTERNAL_BATCH_COMBINE_BYTES_PER_POSITION_FRAME = {
    "mean": 24,
    "median": 26,
}
EXTERNAL_BATCH_WORST_BYTES_PER_POSITION_FRAME = (
    EXTERNAL_BATCH_COMMON_BYTES_PER_POSITION_FRAME
    + max(EXTERNAL_BATCH_REJECTION_BYTES_PER_POSITION_FRAME.values())
    + max(EXTERNAL_BATCH_COMBINE_BYTES_PER_POSITION_FRAME.values())
)  # 34 + 32 + 26 = 92
#: N-SCALED batch bytes per (position x frame). This is the worst-case sum of
#: the arrays shaped ``(n_frames, positions)`` plus headroom. It is an UPPER
#: bound for every supported rejection/combine pair (direct-shape tests).
EXTERNAL_BATCH_SAMPLE_BYTES_PER_POSITION_FRAME = 96
#: FIXED batch bytes per position, independent of n_frames. The quantile,
#: rejection and combine paths allocate many ``(positions,)`` arrays
#: (counts/effective counts/indices/low/high/fractions/selected values/
#: differences/results/conditions, interval bounds, active/degenerate masks).
#: 512 B/position is a conservative UPPER bound for their simultaneous live set,
#: including the quantile return/condition temporaries at low N.
EXTERNAL_BATCH_FIXED_POSITION_BYTES = 512
#: Backward-compatible name: the N-scaled component only. Admission MUST use
#: ``external_batch_bytes`` / the two-part formula, never this alone.
EXTERNAL_BATCH_BYTES_PER_POSITION_FRAME = (
    EXTERNAL_BATCH_SAMPLE_BYTES_PER_POSITION_FRAME
)

#: Equalization peak per patch pixel after the safe release of the mask/lum
#: arrays before the final product (see canonical_equalize.py). Mask stage:
#: out-of-place copy 12 (3 ch x 4) + finite/positive/and bools 9 + valid_rgb 1
#: + luminance 8 + mask/where bools 4 = 34. Product stage: copy 12 + product 12
#: + astype output 12 = 36. 48 is the conservative upper bound.
EQUALIZE_RGB_BYTES_PER_PIXEL = 48


def external_equalize_peak_bytes_per_pixel(channels: int = 3) -> int:
    """Conservative equalization peak B/px for the released-lifetime sequence."""
    c = int(channels)
    if c < 1:
        raise ExternalBudgetError(f"channels must be >= 1, got {channels!r}")
    copy = 4 * c
    mask_stage = copy + c + c + c + 1 + 8 + 4
    product_stage = copy + 4 * c + 4 * c
    return int(max(mask_stage, product_stage))


def external_batch_bytes(n_frames: int, positions: int) -> int:
    """Complete batch live-set bound: N-scaled + fixed per-position terms."""
    n = int(n_frames)
    pos = int(positions)
    if n < 1 or pos < 1:
        raise ExternalBudgetError(
            f"external batch bytes require n_frames>=1 and positions>=1, got "
            f"{n_frames!r}, {positions!r}"
        )
    return int(
        pos
        * (
            n * EXTERNAL_BATCH_SAMPLE_BYTES_PER_POSITION_FRAME
            + EXTERNAL_BATCH_FIXED_POSITION_BYTES
        )
    )

#: Finalization working set per patch pixel. The bounded
#: ``_compute_n_eff_bounded`` uses at most: n_eff/w1_sq f64 8 + sqrt f64 8 +
#: safe f64 8 + overflow/valid/bad/finite/negative bools 5 = 29; 40 leaves
#: headroom and covers the restore-shape copy for mono.
FINALIZE_WORKING_BYTES_PER_PIXEL = 40

#: Backwards-compatible public name: the truthful column-batch workspace.
REDUCTION_BYTES_PER_POSITION_FRAME = EXTERNAL_BATCH_BYTES_PER_POSITION_FRAME

#: Explicit external HOST model schema (not a disk model).
EXTERNAL_HOST_MODEL_SCHEMA = "zemosaic-external-host-components-v1"

#: Documented minimum batch: ONE interior tile row (its pixel-channel positions).
COLUMN_BATCH_MIN = 1

#: Cleanup margin on the spool peak (5 %, at least 1 MiB).
_CLEANUP_MARGIN_NUM = 5
_CLEANUP_MARGIN_DEN = 100
_CLEANUP_MARGIN_MIN_BYTES = 1 << 20

_UNKNOWN_DISK_TERMS = (
    "filesystem_block_rounding_bytes",
    "filesystem_metadata_journal_bytes",
    "page_cache_residency_bytes",
)

_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------

class ExternalSpoolError(RuntimeError):
    """Base class for every external-working-set failure."""


class ExternalSpoolPathError(ExternalSpoolError, ValueError):
    """A caller-supplied path/component would escape the spool root."""


class ExternalSpoolStaleError(ExternalSpoolError):
    """An existing spool is not owned by (or not complete for) this attempt."""


class ExternalSpoolCorruptError(ExternalSpoolError):
    """A spool file/manifest is truncated, corrupted or inconsistent."""


class ExternalSpoolWriteError(ExternalSpoolError):
    """Writing the spool failed (disk full, permission, I/O error)."""


class ExternalDiskShortfall(ExternalSpoolError):
    """A cell boundary refused to start: free space < the admitted envelope.

    Resumable: the run may be relaunched after freeing space; already-committed
    Cells remain valid and must not be recomputed.
    """

    resumable = True
    committed_cells_intact = True


class ExternalSpoolLiveOwnerError(ExternalSpoolError):
    """An existing spool is owned by a PROVABLY LIVE other owner (never steal)."""


class ExternalSpoolCommittedError(ExternalSpoolError):
    """The attempt belongs to a Cell already committed; clean it, do not reuse."""

    committed = True


class ExternalBudgetError(ExternalSpoolError, ValueError):
    """No admissible column batch fits the requested memory budget."""


# ---------------------------------------------------------------------------
# Small validators
# ---------------------------------------------------------------------------

def _require_int(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}={value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _require_component(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExternalSpoolPathError(f"{name} must be a non-empty string")
    if "\x00" in value or "/" in value or "\\" in value:
        raise ExternalSpoolPathError(
            f"{name} must be a single path component (no separators): {value!r}"
        )
    if value in (".", "..") or os.path.isabs(value):
        raise ExternalSpoolPathError(f"{name} must not be '.'/'..'/absolute: {value!r}")
    if not _COMPONENT_RE.match(value):
        raise ExternalSpoolPathError(
            f"{name} contains unsupported characters: {value!r}"
        )
    return value


def _require_root(root) -> Path:
    if root is None:
        raise ExternalSpoolPathError("spool_root is required (never a hidden default)")
    path = Path(root)
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ExternalSpoolPathError(f"spool_root does not exist: {path}") from exc
    if not resolved.is_dir():
        raise ExternalSpoolPathError(f"spool_root is not a directory: {resolved}")
    return resolved


def _confined_child(root: Path, *parts: str) -> Path:
    child = root.joinpath(*parts)
    resolved = child.resolve(strict=False)
    if root != resolved and root not in resolved.parents:
        raise ExternalSpoolPathError(
            f"resolved path {resolved} escapes spool root {root}"
        )
    return child


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _canonical_json(payload) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _manifest_hash(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    data = _canonical_json(payload)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Spec / handles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExternalSpoolSpec:
    """Everything one Cell spool needs; no hidden defaults for geometry."""

    root: object
    n_frames: int
    height: int
    width: int
    channels: int = 3
    halo_px: int = 0
    tile_size: object | None = None
    cell_id: str = "cell"
    attempt_id: str = "attempt"
    frame_ids: tuple | None = None
    created_unix_ns: int | None = None
    # M4 transactional ownership. ``owner_token`` is an opaque run-level
    # identity; ``owner_record`` is the run-store owner document used by the
    # liveness predicate (never PID-only).
    owner_token: str | None = None
    owner_record: dict | None = None
    owner_liveness: object | None = None
    stale_policy: str = "discard"  # discard | refuse
    retain_on_failure: bool = False


@dataclass(frozen=True)
class TileSpool:
    """One written tile of the cell spool (paths + exact shapes)."""

    y0: int
    y1: int
    x0: int
    x1: int
    ey0: int
    ey1: int
    ex0: int
    ex1: int
    th: int
    tw: int
    images_path: str
    weights_path: str
    images_shape: tuple
    weights_shape: tuple
    images_bytes: int
    weights_bytes: int

    def to_dict(self) -> dict:
        return {
            "y0": int(self.y0), "y1": int(self.y1),
            "x0": int(self.x0), "x1": int(self.x1),
            "ey0": int(self.ey0), "ey1": int(self.ey1),
            "ex0": int(self.ex0), "ex1": int(self.ex1),
            "th": int(self.th), "tw": int(self.tw),
            "images_file": Path(self.images_path).name,
            "weights_file": Path(self.weights_path).name,
            "images_shape": [int(v) for v in self.images_shape],
            "weights_shape": [int(v) for v in self.weights_shape],
            "images_bytes": int(self.images_bytes),
            "weights_bytes": int(self.weights_bytes),
        }


# ---------------------------------------------------------------------------
# M4 — transactional attempt classification / stale-or-orphan handling
# ---------------------------------------------------------------------------

def _dir_bytes(path: Path) -> int:
    total = 0
    try:
        for entry in Path(path).rglob("*"):
            if entry.is_file():
                try:
                    total += int(entry.stat().st_size)
                except OSError:
                    pass
    except OSError:
        pass
    return int(total)


def _read_attempt_marker(attempt_dir: Path) -> dict | None:
    path = Path(attempt_dir) / SPOOL_OWNER_NAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a missing/corrupt marker is stale
        return None
    return raw if isinstance(raw, dict) else None


def _owner_alive(record, liveness) -> bool:
    if record is None:
        return False
    if liveness is not None:
        try:
            return bool(liveness(record))
        except Exception:  # noqa: BLE001 - unprovable liveness is NOT alive
            return False
    try:
        from . import run_store as _run_store

        return bool(_run_store.owner_is_live(record))
    except Exception:  # noqa: BLE001
        return False


def _handle_existing_attempt(
    attempt_dir: Path,
    *,
    owner_token,
    owner_liveness,
    stale_policy: str,
) -> str:
    """Classify an existing attempt dir; return 'live' or 'discarded'.

    A committed attempt is never reused (raise); a same-owner incomplete
    attempt is discarded (a retry gets a fresh attempt); another owner's spool
    is only refused, never stolen.
    """
    marker = _read_attempt_marker(attempt_dir)
    if marker is None:
        if stale_policy == "refuse":
            raise ExternalSpoolStaleError(
                f"spool attempt has no valid owner marker: {attempt_dir}"
            )
        shutil.rmtree(attempt_dir, ignore_errors=True)
        return "discarded"
    state = str(marker.get("state") or "")
    if state == SPOOL_STATE_COMMITTED:
        raise ExternalSpoolCommittedError(
            f"spool attempt belongs to a committed Cell: {attempt_dir}"
        )
    existing_token = marker.get("owner_token")
    if owner_token is not None and existing_token == owner_token:
        if stale_policy == "refuse":
            raise ExternalSpoolStaleError(
                f"same-owner incomplete attempt is refused: {attempt_dir}"
            )
        shutil.rmtree(attempt_dir, ignore_errors=True)
        return "discarded"
    if _owner_alive(marker.get("owner_record"), owner_liveness):
        return "live"
    # Liveness that cannot be PROVEN from a recorded owner document is never
    # treated as death: stealing an unprovable owner's spool is forbidden.
    if marker.get("owner_record") is None:
        raise ExternalSpoolStaleError(
            "cannot prove the existing attempt owner is dead (no owner "
            f"record); refusing to steal it: {attempt_dir}"
        )
    if stale_policy == "refuse":
        raise ExternalSpoolStaleError(
            f"stale attempt from a dead owner is refused: {attempt_dir}"
        )
    shutil.rmtree(attempt_dir, ignore_errors=True)
    return "discarded"


def classify_attempt(attempt_dir, *, owner_liveness=None) -> dict:
    """Strict JSON-safe classification of ONE attempt directory."""
    path = Path(attempt_dir)
    marker = _read_attempt_marker(path)
    if marker is None:
        return {
            "path": str(path), "state": None, "owner_token": None,
            "cell_id": None, "attempt_id": None, "committed": False,
            "owner_alive": False, "has_manifest": (path / SPOOL_MANIFEST_NAME).exists(),
            "bytes": _dir_bytes(path),
        }
    state = str(marker.get("state") or "")
    return {
        "path": str(path),
        "state": state or None,
        "owner_token": marker.get("owner_token"),
        "cell_id": marker.get("cell_id"),
        "attempt_id": marker.get("attempt_id"),
        "committed": state == SPOOL_STATE_COMMITTED,
        "owner_alive": _owner_alive(marker.get("owner_record"), owner_liveness),
        "has_manifest": (path / SPOOL_MANIFEST_NAME).exists(),
        "bytes": _dir_bytes(path),
    }


def discard_stale_for_cell(
    root,
    cell_id: str,
    *,
    owner_token=None,
    owner_liveness=None,
    committed_cell_ids=(),
) -> dict:
    """Clean stale/orphan attempts of ONE Cell; never touch a live owner.

    A Cell already present in ``committed_cell_ids`` is authoritative: its
    leftover spool is cleaned WITHOUT touching the commit and MUST NOT be
    recomputed. A committed attempt is always cleaned. A same-owner incomplete
    attempt is discarded (fresh retry). Another live owner's attempt is skipped.
    """
    root_path = _require_root(root)
    cell = _require_component(cell_id, "cell_id")
    cell_dir = _confined_child(root_path, cell)
    report = {
        "schema": "zemosaic-zegrid-external-spool-scan-v1",
        "cell_id": cell,
        "committed_cell": cell in set(committed_cell_ids or ()),
        "attempts_seen": 0,
        "discarded_stale": 0,
        "discarded_same_owner": 0,
        "cleaned_committed": 0,
        "live_skipped": 0,
        "bytes_freed": 0,
        "failures": [],
    }
    if not cell_dir.is_dir():
        return report
    for attempt in sorted(p for p in cell_dir.iterdir() if p.is_dir()):
        report["attempts_seen"] += 1
        info = classify_attempt(attempt, owner_liveness=owner_liveness)
        state = info["state"]
        is_committed = bool(info["committed"] or report["committed_cell"])
        if is_committed:
            action = "cleaned_committed"
        elif owner_token is not None and info["owner_token"] == owner_token:
            action = "discarded_same_owner"
        elif info["owner_alive"]:
            action = "live_skipped"
        else:
            action = "discarded_stale"
        if action == "live_skipped":
            report["live_skipped"] += 1
            continue
        try:
            shutil.rmtree(attempt, ignore_errors=False)
            report[action] += 1
            report["bytes_freed"] += int(info["bytes"])
        except Exception as exc:  # noqa: BLE001 - accounted, never fatal
            report["failures"].append(
                {"path": str(attempt), "error": f"{type(exc).__name__}: {exc}"}
            )
    return report


def cleanup_after_commit(root, cell_id: str, *, owner_token=None) -> dict:
    """Post-commit spool cleanup for ONE Cell (commit is authoritative)."""
    report = discard_stale_for_cell(
        root,
        cell_id,
        owner_token=owner_token,
        committed_cell_ids=(cell_id,),
    )
    # M5: leave NO residue once every attempt of this Cell is gone. Only EMPTY
    # directories are pruned (rmdir never removes content), so a live/late
    # attempt or any foreign file blocks the prune instead of being destroyed.
    try:
        root_path = _require_root(root)
        cell_dir = root_path / str(cell_id)
        for candidate in (cell_dir, root_path):
            try:
                candidate.rmdir()
            except OSError:
                break
    except Exception:
        pass
    return report


def scan_spool_root(
    root,
    *,
    owner_token=None,
    owner_liveness=None,
    committed_cell_ids=(),
) -> dict:
    """Scan every Cell spool under ``root`` and clean stale/orphan attempts."""
    root_path = _require_root(root)
    totals = {
        "schema": "zemosaic-zegrid-external-spool-scan-root-v1",
        "cells_scanned": 0,
        "attempts_seen": 0,
        "discarded_stale": 0,
        "discarded_same_owner": 0,
        "cleaned_committed": 0,
        "live_skipped": 0,
        "bytes_freed": 0,
        "failures": [],
    }
    try:
        cell_dirs = sorted(p for p in root_path.iterdir() if p.is_dir())
    except OSError as exc:
        totals["failures"].append({"path": str(root_path), "error": str(exc)})
        return totals
    for cell_dir in cell_dirs:
        totals["cells_scanned"] += 1
        report = discard_stale_for_cell(
            root_path,
            cell_dir.name,
            owner_token=owner_token,
            owner_liveness=owner_liveness,
            committed_cell_ids=committed_cell_ids,
        )
        for key in (
            "attempts_seen", "discarded_stale", "discarded_same_owner",
            "cleaned_committed", "live_skipped", "bytes_freed",
        ):
            totals[key] += int(report.get(key, 0))
        totals["failures"].extend(report.get("failures", []))
    return totals


# ---------------------------------------------------------------------------
# Spool writer / lifecycle
# ---------------------------------------------------------------------------

class ExternalSpool:
    """Owns ONE ``<root>/<cell_id>/<attempt_id>`` spool directory.

    The attempt directory is created fresh and stamped with an owner marker.
    Any pre-existing non-empty attempt directory is REFUSED as stale: a later
    attempt never silently reuses another attempt's (possibly partial) data.
    Tiles are written, validated, reduced and discarded one at a time, so the
    on-disk spool peak is exactly ONE worst tile, not the whole Cell.
    """

    def __init__(self, spec: ExternalSpoolSpec) -> None:
        if not isinstance(spec, ExternalSpoolSpec):
            raise ValueError(f"expected ExternalSpoolSpec, got {type(spec).__name__}")
        self.root = _require_root(spec.root)
        self.cell_id = _require_component(spec.cell_id, "cell_id")
        self.attempt_id = _require_component(spec.attempt_id, "attempt_id")
        self.n_frames = _require_int(spec.n_frames, "n_frames", minimum=1)
        self.height = _require_int(spec.height, "height", minimum=1)
        self.width = _require_int(spec.width, "width", minimum=1)
        self.channels = _require_int(spec.channels, "channels", minimum=1)
        self.halo_px = _require_int(spec.halo_px, "halo_px")
        self.tile_size = spec.tile_size
        if spec.frame_ids is not None:
            ids = tuple(str(v) for v in spec.frame_ids)
            if len(ids) != self.n_frames:
                raise ValueError("frame_ids length must equal n_frames")
            self.frame_ids = ids
        else:
            self.frame_ids = None
        self.created_unix_ns = int(spec.created_unix_ns or time.time_ns())
        self.owner_token = spec.owner_token
        self.owner_record = (None if spec.owner_record is None else dict(spec.owner_record))
        self._owner_liveness = spec.owner_liveness
        self.stale_policy = str(spec.stale_policy)
        if self.stale_policy not in ("discard", "refuse"):
            raise ValueError("stale_policy must be 'discard' or 'refuse'")
        self.retain_on_failure = bool(spec.retain_on_failure)
        self._tiles: list[TileSpool] = []
        self._finalized = False
        self._cleaned = False
        self.state = SPOOL_STATE_WRITING

        cell_dir = _confined_child(self.root, self.cell_id)
        self.cell_dir = cell_dir
        attempt_dir = _confined_child(cell_dir, self.attempt_id)
        if attempt_dir.exists():
            if any(attempt_dir.iterdir()):
                resolved = _handle_existing_attempt(
                    attempt_dir,
                    owner_token=self.owner_token,
                    owner_liveness=self._owner_liveness,
                    stale_policy=self.stale_policy,
                )
                if resolved == "live":
                    raise ExternalSpoolLiveOwnerError(
                        f"spool attempt {attempt_dir} belongs to a provably "
                        "live owner; refusing to steal it"
                    )
            else:
                attempt_dir.rmdir()
        attempt_dir.mkdir(parents=True, exist_ok=False)
        self.attempt_dir = attempt_dir
        marker = {
            "schema": SPOOL_SCHEMA,
            "cell_id": self.cell_id,
            "attempt_id": self.attempt_id,
            "created_unix_ns": int(self.created_unix_ns),
            "n_frames": int(self.n_frames),
            "height": int(self.height),
            "width": int(self.width),
            "channels": int(self.channels),
            "halo_px": int(self.halo_px),
            "state": SPOOL_STATE_WRITING,
            "owner_token": self.owner_token,
            "owner_record": self.owner_record,
        }
        _atomic_json(attempt_dir / SPOOL_OWNER_NAME, marker)

    # -- transactional state -------------------------------------------------
    def mark_state(self, state: str) -> dict:
        if state not in SPOOL_STATES:
            raise ValueError(f"unknown spool state {state!r}")
        if self._cleaned:
            raise ExternalSpoolStaleError("spool already cleaned")
        marker_path = self.attempt_dir / SPOOL_OWNER_NAME
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ExternalSpoolCorruptError(
                f"cannot read attempt marker {marker_path}"
            ) from exc
        marker["state"] = str(state)
        marker["updated_unix_ns"] = int(time.time_ns())
        _atomic_json(marker_path, marker)
        self.state = str(state)
        return marker

    def mark_ready(self) -> dict:
        return self.mark_state(SPOOL_STATE_READY)

    def mark_committed(self) -> dict:
        return self.mark_state(SPOOL_STATE_COMMITTED)

    # -- lifecycle ----------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.attempt_dir / SPOOL_MANIFEST_NAME

    def finalize_manifest(
        self,
        *,
        taper_kind: str,
        taper_px: float,
        taper_floor: float,
    ) -> dict:
        if self._cleaned:
            raise ExternalSpoolStaleError("spool already cleaned")
        tiles = [t.to_dict() for t in self._tiles]
        manifest = {
            "schema": SPOOL_SCHEMA,
            "cell_id": self.cell_id,
            "attempt_id": self.attempt_id,
            "created_unix_ns": int(self.created_unix_ns),
            "n_frames": int(self.n_frames),
            "height": int(self.height),
            "width": int(self.width),
            "channels": int(self.channels),
            "halo_px": int(self.halo_px),
            "frame_ids": list(self.frame_ids) if self.frame_ids else None,
            "taper": {
                "kind": str(taper_kind),
                "px": float(taper_px),
                "floor": float(taper_floor),
            },
            "tile_size": (
                None if self.tile_size is None else list(_tile_dims(self.tile_size))
            ),
            "dtypes": {"images": "float32", "weights": "float32"},
            "tiles": tiles,
            "tile_count": len(tiles),
            "images_bytes": int(sum(t["images_bytes"] for t in tiles)),
            "weights_bytes": int(sum(t["weights_bytes"] for t in tiles)),
            "total_payload_bytes": int(
                sum(t["images_bytes"] + t["weights_bytes"] for t in tiles)
            ),
        }
        manifest["manifest_sha256"] = _manifest_hash(manifest)
        _atomic_json(self.manifest_path, manifest)
        self._finalized = True
        return manifest

    def load_manifest(self) -> dict:
        path = self.manifest_path
        try:
            raw = path.read_bytes()
        except FileNotFoundError as exc:
            raise ExternalSpoolCorruptError(
                f"spool manifest is missing: {path}"
            ) from exc
        try:
            manifest = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - typed corruption
            raise ExternalSpoolCorruptError(
                f"spool manifest is not valid JSON: {path}"
            ) from exc
        if not isinstance(manifest, dict):
            raise ExternalSpoolCorruptError("spool manifest must be a JSON object")
        if manifest.get("schema") != SPOOL_SCHEMA:
            raise ExternalSpoolCorruptError(
                f"spool manifest schema {manifest.get('schema')!r} != {SPOOL_SCHEMA!r}"
            )
        expected = manifest.get("manifest_sha256")
        actual = _manifest_hash(manifest)
        if not isinstance(expected, str) or expected != actual:
            raise ExternalSpoolCorruptError(
                "spool manifest SHA-256 mismatch (truncated/corrupted manifest)"
            )
        return manifest

    def cleanup(self) -> None:
        """Delete the attempt directory (only inside the resolved root)."""
        if self._cleaned:
            return
        target = self.attempt_dir.resolve(strict=False)
        if self.root not in target.parents:
            raise ExternalSpoolPathError(
                f"refusing to delete {target}: outside spool root {self.root}"
            )
        shutil.rmtree(target, ignore_errors=False)
        self._cleaned = True

    discard = cleanup

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.cleanup()
        except Exception:
            if exc is None:
                raise
        return False

    # -- tile writing -------------------------------------------------------
    def _tile_dir(self, y0: int, x0: int) -> Path:
        name = f"tile_y{int(y0):06d}_x{int(x0):06d}"
        return _confined_child(self.attempt_dir, name)

    def write_tile(
        self,
        provider,
        p1: _Phase1,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        *,
        taper_kind: str,
        taper_px: float,
        taper_floor: float,
        halo_px: int,
    ) -> TileSpool:
        """Prepare and spool ONE extended tile, frame by frame, exactly as RAM."""
        n = int(p1.n_frames)
        c = int(p1.channels)
        ey0 = max(0, int(y0) - int(halo_px))
        ey1 = min(int(self.height), int(y1) + int(halo_px))
        ex0 = max(0, int(x0) - int(halo_px))
        ex1 = min(int(self.width), int(x1) + int(halo_px))
        th = ey1 - ey0
        tw = ex1 - ex0
        tile_dir = self._tile_dir(y0, x0)
        if tile_dir.exists():
            raise ExternalSpoolStaleError(
                f"tile spool directory already exists: {tile_dir}"
            )
        tile_dir.mkdir(parents=True, exist_ok=False)
        images_path = tile_dir / IMAGES_NAME
        weights_path = tile_dir / WEIGHTS_NAME
        try:
            images = np.lib.format.open_memmap(
                images_path, mode="w+", dtype=np.float32, shape=(n, th, tw, c)
            )
            weights = np.lib.format.open_memmap(
                weights_path, mode="w+", dtype=np.float32, shape=(n, th, tw)
            )
            try:
                for i in range(n):
                    if not p1.norm_active[i]:
                        images[i] = np.nan
                        weights[i] = 0.0
                        continue
                    raw_tile, pre_valid = provider.get_tile(
                        i, ey0, ey1, ex0, ex1
                    )
                    a = p1.coefficients[i]
                    src64 = raw_tile.astype(np.float64)
                    transformed = np.empty((th, tw, c), dtype=np.float64)
                    for ch in range(c):
                        transformed[:, :, ch] = (
                            a[ch, 0] * src64[:, :, ch] + a[ch, 1]
                        )
                    with np.errstate(over="ignore", invalid="ignore"):
                        transformed32 = transformed.astype(np.float32)
                    finite_all = np.all(np.isfinite(transformed32), axis=-1)
                    post_valid = pre_valid & finite_all
                    frame = np.empty((th, tw, c), dtype=np.float32)
                    frame[:] = np.nan
                    frame[post_valid] = transformed32[post_valid]
                    images[i] = frame
                    if taper_kind == "footprint":
                        taper = make_footprint_taper(
                            post_valid, feather_px=taper_px, floor=taper_floor
                        ).astype(np.float32)
                    else:
                        taper = np.ones((th, tw), dtype=np.float32)
                    weights[i] = np.where(
                        post_valid, taper, np.float32(0.0)
                    ).astype(np.float32)
            finally:
                images.flush()
                weights.flush()
                del images
                del weights
        except OSError as exc:
            shutil.rmtree(tile_dir, ignore_errors=True)
            raise ExternalSpoolWriteError(
                f"failed to write tile spool {tile_dir}: {exc}"
            ) from exc
        except Exception:
            # A partial tile must never survive for a later attempt.
            shutil.rmtree(tile_dir, ignore_errors=True)
            raise
        handle = TileSpool(
            y0=int(y0), y1=int(y1), x0=int(x0), x1=int(x1),
            ey0=int(ey0), ey1=int(ey1), ex0=int(ex0), ex1=int(ex1),
            th=int(th), tw=int(tw),
            images_path=str(images_path), weights_path=str(weights_path),
            images_shape=(n, th, tw, c), weights_shape=(n, th, tw),
            images_bytes=int(images_path.stat().st_size),
            weights_bytes=int(weights_path.stat().st_size),
        )
        self._tiles.append(handle)
        return handle

    def open_tile(self, handle: TileSpool):
        """Open + validate the tile payload; return read-only memmaps."""
        validate_tile(handle)
        try:
            images = np.load(handle.images_path, mmap_mode="r")
            weights = np.load(handle.weights_path, mmap_mode="r")
        except Exception as exc:  # noqa: BLE001 - typed corruption
            raise ExternalSpoolCorruptError(
                f"cannot open tile spool files for {handle.y0},{handle.x0}: {exc}"
            ) from exc
        return images, weights

    def discard_tile(self, handle: TileSpool) -> None:
        tile_dir = Path(handle.images_path).parent
        if self.attempt_dir.resolve(strict=False) not in tile_dir.resolve(
            strict=False
        ).parents:
            raise ExternalSpoolPathError(
                f"refusing to delete tile outside the attempt dir: {tile_dir}"
            )
        shutil.rmtree(tile_dir, ignore_errors=True)


def validate_tile(handle: TileSpool) -> None:
    """Validate existence/shape/dtype/size of one tile payload (typed errors)."""
    for attr, expected_shape in (
        ("images_path", tuple(handle.images_shape)),
        ("weights_path", tuple(handle.weights_shape)),
    ):
        path = Path(getattr(handle, attr))
        if not path.exists():
            raise ExternalSpoolCorruptError(f"missing spool file: {path}")
        try:
            arr = np.load(path, mmap_mode="r")
            got_shape = tuple(int(v) for v in arr.shape)
            got_dtype = np.dtype(arr.dtype)
            file_size = int(path.stat().st_size)
            payload_end = int(arr.offset) + int(arr.nbytes)
        except ExternalSpoolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExternalSpoolCorruptError(
                f"unreadable spool file {path}: {exc}"
            ) from exc
        if got_shape != expected_shape:
            raise ExternalSpoolCorruptError(
                f"spool shape {got_shape} != expected {expected_shape} ({path})"
            )
        if got_dtype != np.dtype(np.float32):
            raise ExternalSpoolCorruptError(
                f"spool dtype {got_dtype} != float32 ({path})"
            )
        if file_size < payload_end:
            raise ExternalSpoolCorruptError(
                f"truncated spool file {path}: {file_size} B < {payload_end} B"
            )
        del arr


# ---------------------------------------------------------------------------
# Honest resource pricing
# ---------------------------------------------------------------------------

def _tile_geometry(
    height: int, width: int, tile_size, halo_px: int
) -> tuple[list[tuple[int, int, int, int, int, int]], int, int]:
    """All interior tiles with their extended bounds; max ext dims."""
    tiles = []
    max_th = max_tw = 0
    for (y0, y1, x0, x1) in _iter_tiles(height, width, tile_size):
        ey0 = max(0, y0 - halo_px)
        ey1 = min(height, y1 + halo_px)
        ex0 = max(0, x0 - halo_px)
        ex1 = min(width, x1 + halo_px)
        tiles.append((y0, y1, x0, x1, ey1 - ey0, ex1 - ex0))
        max_th = max(max_th, ey1 - ey0)
        max_tw = max(max_tw, ex1 - ex0)
    return tiles, max_th, max_tw


def _max_interior_width(tiles) -> int:
    best = 0
    for (y0, y1, x0, x1, _th, _tw) in tiles:
        best = max(best, int(x1) - int(x0))
    return best


def _positions_per_row(interior_width: int, channels: int) -> int:
    return int(interior_width) * int(channels)


def batch_positions_for(
    column_batch: int, interior_width: int, channels: int
) -> tuple[int, int]:
    """Return ``(rows_per_batch, cols_per_batch)`` for the 2D micro-block policy.

    A batch is ``rows_per_batch`` interior rows by ``cols_per_batch`` interior
    columns (``positions = rows * cols * channels``). When ``column_batch``
    covers at least one full row, whole rows are batched. When it is smaller
    than one full row, a single row is split into column chunks (blocks narrower
    than a row), each at least one pixel wide, so the reduction workspace is
    bounded below one full row.
    """
    c = _require_int(channels, "channels", minimum=1)
    iw = _require_int(interior_width, "interior_width", minimum=1)
    req = _require_int(column_batch, "column_batch", minimum=1)
    ppr = iw * c  # positions per full row
    if req >= ppr:
        return req // ppr, iw
    cols = max(1, req // c)  # whole pixels that fit the request
    return 1, cols


def resolve_column_batch(
    column_batch: int, *, interior_hw: tuple[int, int], channels: int,
) -> dict:
    """Resolve a requested column batch against the REAL interior surface.

    Single source of truth for the batch clamp: the host model, the pricing
    function and the published cell record all read this. ``< 1`` is rejected
    (``ExternalBudgetError``); a value merely LARGER than the tile interior is
    normalized, never refused (a resumed run may carry one that never had an
    effect). ``clamped`` is True only when the interior bound actually reduced
    the batch; ordinary row/pixel quantization is not a clamp.
    """
    req = _require_int(column_batch, "column_batch", minimum=1)
    ih = _require_int(interior_hw[0], "interior_height", minimum=1)
    iw = _require_int(interior_hw[1], "interior_width", minimum=1)
    c = _require_int(channels, "channels", minimum=1)
    rows, cols = batch_positions_for(req, iw, c)
    rows_eff = min(int(rows), ih)
    cols_eff = min(int(cols), iw)
    positions = int(rows_eff * cols_eff * c)
    return {
        "requested_column_batch": int(req),
        # ``effective_column_batch`` is expressed in POSITIONS (the same unit a
        # caller passes as ``column_batch``), i.e. the real interior surface.
        "effective_column_batch": positions,
        "rows_per_batch": int(rows_eff),
        "cols_per_batch": int(cols_eff),
        "batch_positions": positions,
        "unclamped_batch_positions": int(rows * cols * c),
        "column_batch_clamped": bool(rows_eff != rows or cols_eff != cols),
    }


def external_batch_live_bytes_per_position_frame(
    rejection: str = "winsorized_sigma_clip", combine: str = "mean"
) -> int:
    """Conservative worst simultaneous batch bytes per (position x frame).

    The value is the common live set plus the MAX of the rejection and combine
    variant terms (the two stages are sequential). It is a UPPER bound for every
    supported pair; the admission coefficient is the global worst case across
    all pairs (``EXTERNAL_BATCH_BYTES_PER_POSITION_FRAME``).
    """
    rej = str(rejection).strip().lower()
    comb = str(combine).strip().lower()
    if rej not in EXTERNAL_BATCH_REJECTION_BYTES_PER_POSITION_FRAME:
        raise ExternalBudgetError(
            f"unsupported rejection {rejection!r} for the external batch model"
        )
    if comb not in EXTERNAL_BATCH_COMBINE_BYTES_PER_POSITION_FRAME:
        raise ExternalBudgetError(
            f"unsupported combine {combine!r} for the external batch model"
        )
    return int(
        EXTERNAL_BATCH_COMMON_BYTES_PER_POSITION_FRAME
        + EXTERNAL_BATCH_REJECTION_BYTES_PER_POSITION_FRAME[rej]
        + EXTERNAL_BATCH_COMBINE_BYTES_PER_POSITION_FRAME[comb]
    )


def external_cell_host_components(
    *,
    n_frames: int,
    patch_hw: tuple[int, int],
    tile_hw: tuple[int, int],
    interior_hw: tuple[int, int],
    channels: int,
    column_batch: int,
    rejection_sink: bool = True,
    equalize_rgb: bool = False,
    rejection: str = "winsorized_sigma_clip",
    combine: str = "mean",
) -> dict:
    """Explicit, integer HOST-RAM components of ONE external Cell.

    Lifetime model (all values are the REAL arrays inspected in
    ``run_cell_streaming_external`` / ``reduce_spooled_tile`` / ``write_tile``):

    * ``persistent`` — result planes allocated BEFORE the tile loop and held
      until the durable commit releases them: science f32, weight_sum f64,
      valid_mask bool, surviving int64 over the full patch, plus support_w1/w2
      f64 over the patch. The dense rejection mask is added only when no
      bounded sink is used (the product path always supplies one).
    * ``write`` — while the persistent planes are live, one tile is prepared
      from the provider: the provider window (one full prepared RGB f32 frame
      + bool support over the patch), the per-frame transformed planes, and the
      taper/distance transform. The PREVIOUS tile's output planes are still
      referenced until the next reduction rebinds them, so they are included.
    * ``reduce`` — while the persistent planes are live, the tile-local output
      planes (four f32/f64/bool/int64 planes + interior support_w1/w2) and the
      column-batch workspace coexist.
    * ``finalize`` — n_eff_support f64 is allocated after the tile loop while
      the persistent planes are still live.

    The returned ``peak_bytes`` is ``max(write, reduce, finalize)``: the
    sub-phases are SEQUENTIAL within one Cell. Spool bytes are deliberately
    absent — they belong to the DISK contract, never to host RAM. Final mosaic
    assembly is a LATER, separately admitted phase and is absent too.
    """
    n = _require_int(n_frames, "n_frames", minimum=1)
    c = _require_int(channels, "channels", minimum=1)
    ph, pw = int(patch_hw[0]), int(patch_hw[1])
    th, tw = int(tile_hw[0]), int(tile_hw[1])
    ih, iw = int(interior_hw[0]), int(interior_hw[1])
    for name, value in (("patch_h", ph), ("patch_w", pw), ("tile_h", th),
                        ("tile_w", tw), ("interior_h", ih), ("interior_w", iw)):
        if value < 1:
            raise ExternalBudgetError(
                f"external host model requires a positive {name}, got {value}"
            )
    req = _require_int(column_batch, "column_batch", minimum=1)
    sink = bool(rejection_sink)

    patch_px = ph * pw
    tile_px = th * tw
    interior_px = ih * iw

    persistent = patch_px * (
        c * RESULT_PLANE_BYTES_PER_POSITION + RESULT_SUPPORT_BYTES_PER_PIXEL
    )
    dense_rejection = (
        0 if sink else n * patch_px * c * REJECTION_MASK_BYTES_PER_POSITION_FRAME
    )
    persistent += dense_rejection
    # The finalization is a bounded exact sequence (see
    # ``_compute_n_eff_bounded``). It still holds the persistent planes, the
    # n_eff/sqrt/safe working buffers and boolean masks; request-aware
    # equalization adds its own full-patch overlap.
    finalize_working = patch_px * FINALIZE_WORKING_BYTES_PER_PIXEL
    finalize_equalize = (
        patch_px * EQUALIZE_RGB_BYTES_PER_PIXEL if bool(equalize_rgb) else 0
    )
    finalize_restore = 4 * patch_px if int(c) == 1 else 0
    finalize_total = (
        int(finalize_working) + int(finalize_equalize) + int(finalize_restore)
    )

    provider_window = patch_px * (4 * c + 1)
    write_tile = tile_px * (
        c * WRITE_TILE_BYTES_PER_PIXEL_CHANNEL + WRITE_TILE_AUX_BYTES_PER_PIXEL
    )
    reduce_tile = (
        tile_px * c * RESULT_PLANE_BYTES_PER_POSITION
        + interior_px * RESULT_SUPPORT_BYTES_PER_PIXEL
    )

    # ZM-M33 lot 1: a batch can never hold more positions than the interior
    # tile actually has (``batch_positions_for`` sizes rows from the request
    # alone). The clamp is RECORDED, not silent: the caller publishes
    # requested/effective/clamped in the cell record.
    resolved = resolve_column_batch(req, interior_hw=(ih, iw), channels=c)
    rows = resolved["rows_per_batch"]
    cols = resolved["cols_per_batch"]
    positions = resolved["batch_positions"]
    batch = external_batch_bytes(n, positions)

    first_write = persistent + provider_window + write_tile
    # The previous tile's outputs stay referenced through the next write_tile.
    write_phase = persistent + reduce_tile + provider_window + write_tile
    reduce_phase = persistent + reduce_tile + batch
    finalize = persistent + finalize_total
    peak = max(first_write, write_phase, reduce_phase, finalize)

    return {
        "schema": EXTERNAL_HOST_MODEL_SCHEMA,
        "observation_domain": "net_available",
        "n_frames": int(n),
        "patch_hw": [int(ph), int(pw)],
        "tile_hw": [int(th), int(tw)],
        "interior_hw": [int(ih), int(iw)],
        "channels": int(c),
        "column_batch": int(req),
        "requested_column_batch": int(resolved["requested_column_batch"]),
        "effective_column_batch": int(resolved["batch_positions"]),
        "column_batch_clamped": bool(resolved["column_batch_clamped"]),
        "rows_per_batch": int(rows),
        "cols_per_batch": int(cols),
        "batch_positions": int(positions),
        "batch_sample_bytes_per_position_frame": int(
            EXTERNAL_BATCH_SAMPLE_BYTES_PER_POSITION_FRAME
        ),
        "batch_fixed_position_bytes": int(
            EXTERNAL_BATCH_FIXED_POSITION_BYTES
        ),
        "batch_total_bytes": int(batch),
        "persistent_result_bytes": int(persistent),
        "dense_rejection_mask_bytes": int(dense_rejection),
        "finalize_n_eff_output_bytes": int(
            patch_px * RESULT_N_EFF_BYTES_PER_PIXEL
        ),
        "finalize_working_bytes": int(finalize_working),
        "finalize_equalize_bytes": int(finalize_equalize),
        "finalize_restore_copy_bytes": int(finalize_restore),
        "finalize_total_bytes": int(finalize_total),
        "equalize_rgb": bool(equalize_rgb),
        "rejection": str(rejection),
        "combine": str(combine),
        "batch_live_bound_bytes_per_position_frame": int(
            external_batch_live_bytes_per_position_frame(rejection, combine)
        ),
        "batch_bytes_per_position_frame": int(
            EXTERNAL_BATCH_BYTES_PER_POSITION_FRAME
        ),
        "provider_window_bytes": int(provider_window),
        "write_tile_bytes": int(write_tile),
        "reduce_tile_outputs_bytes": int(reduce_tile),
        "reduction_batch_bytes": int(batch),
        "phases": {
            "first_write": int(first_write),
            "write": int(write_phase),
            "reduce": int(reduce_phase),
            "finalize": int(finalize),
        },
        "reduce_fixed_bytes": int(persistent + reduce_tile),
        "fixed_peak_bytes": int(max(write_phase, finalize, first_write)),
        "peak_bytes": int(peak),
        # Spool/disk bytes are intentionally NOT a host component.
        "disk_bytes_in_host_peak": 0,
    }


def estimate_external_working_set(
    *,
    n_frames: int,
    height: int,
    width: int,
    channels: int,
    tile_size,
    halo_px: int,
    column_batch: int,
    equalize_rgb: bool = False,
    rejection: str = "winsorized_sigma_clip",
    combine: str = "mean",
) -> dict:
    """Pure, geometry-derived pricing of the external working set.

    Every term is explicit. Unknown terms are NAMED (never returned as zero).
    ``equalize_rgb`` / ``rejection`` / ``combine`` select the request-aware HOST
    components (finalization equalization overlap and batch live-set bound).
    """
    n = _require_int(n_frames, "n_frames", minimum=1)
    h = _require_int(height, "height", minimum=1)
    w = _require_int(width, "width", minimum=1)
    c = _require_int(channels, "channels", minimum=1)
    halo = _require_int(halo_px, "halo_px")
    req = _require_int(column_batch, "column_batch", minimum=1)
    tiles, max_th, max_tw = _tile_geometry(h, w, tile_size, halo)
    per_px_frame = 4 * c + 4  # images f32 + weights f32

    worst_tile_bytes = n * max_th * max_tw * per_px_frame
    cumulative_writes = int(
        sum(n * th * tw * per_px_frame for (_y0, _y1, _x0, _x1, th, tw) in tiles)
    )
    amplification = 1.0  # contiguous row slices: each payload byte read once

    max_iw = _max_interior_width(tiles)
    max_ih = max(int(y1 - y0) for (y0, y1, _x0, _x1, _th, _tw) in tiles)
    # ZM-M33 lot 1: a batch never exceeds the real interior surface. Pricing
    # and host components read the SAME resolved positions so the published
    # telemetry cannot disagree.
    resolved = resolve_column_batch(req, interior_hw=(max_ih, max_iw), channels=c)
    rows = resolved["rows_per_batch"]
    cols = resolved["cols_per_batch"]
    positions = resolved["batch_positions"]
    reduction_bytes = external_batch_bytes(n, positions)

    # Inherent O(tile_area) planes: per-tile outputs + support, plus the
    # interior rejection block for one tile (only when no bounded sink is used).
    tile_area = max_th * max_tw
    output_bytes = tile_area * c * (4 + 8 + 1 + 8) + max_th * max_tw * 16
    rejection_block_bytes = n * tile_area * c
    host = external_cell_host_components(
        n_frames=n,
        patch_hw=(h, w),
        tile_hw=(max_th, max_tw),
        interior_hw=(max_ih, max_iw),
        channels=c,
        column_batch=req,
        rejection_sink=True,
        equalize_rgb=bool(equalize_rgb),
        rejection=str(rejection),
        combine=str(combine),
    )

    cleanup_margin = max(
        _CLEANUP_MARGIN_MIN_BYTES,
        (worst_tile_bytes * _CLEANUP_MARGIN_NUM) // _CLEANUP_MARGIN_DEN,
    )
    return {
        "schema": "zemosaic-zegrid-external-pricing-v1",
        "geometry": {
            "n_frames": n, "height": h, "width": w, "channels": c,
            "tile_size": None if tile_size is None else int(tile_size),
            "halo_px": halo, "tile_count": len(tiles),
            "max_extent_hw": [int(max_th), int(max_tw)],
            "max_interior_width": int(max_iw),
        },
        "ram": {
            "reduction_workspace_bytes": int(reduction_bytes),
            "column_batch": int(req),
            "requested_column_batch": int(resolved["requested_column_batch"]),
            "effective_column_batch": int(resolved["batch_positions"]),
            "column_batch_clamped": bool(resolved["column_batch_clamped"]),
            "rows_per_batch": int(rows),
            "cols_per_batch": int(cols),
            "batch_positions": int(positions),
            "positions_per_row": int(_positions_per_row(max_iw, c)),
            "bytes_per_position_frame": int(
                EXTERNAL_BATCH_SAMPLE_BYTES_PER_POSITION_FRAME
            ),
            "fixed_position_bytes": int(
                EXTERNAL_BATCH_FIXED_POSITION_BYTES
            ),
            "batch_total_bytes": int(reduction_bytes),
            "tile_output_temporaries_bytes": int(
                host["reduce_tile_outputs_bytes"]
            ),
            "rejection_block_bytes_if_dense": int(rejection_block_bytes),
            "persistent_result_bytes": int(host["persistent_result_bytes"]),
            "provider_window_bytes": int(host["provider_window_bytes"]),
            "write_tile_bytes": int(host["write_tile_bytes"]),
            "host_peak_bytes": int(host["peak_bytes"]),
            "host_fixed_peak_bytes": int(host["fixed_peak_bytes"]),
            "host_reduce_fixed_bytes": int(host["reduce_fixed_bytes"]),
            "independent_of_patch_area": True,
            "host_peak_depends_on_patch_area": True,
            "depends_on": ["n_frames", "column_batch", "tile_area"],
        },
        "host": host,
        "disk": {
            "spool_peak_bytes_worst_tile": int(worst_tile_bytes),
            "cumulative_writes_bytes": int(cumulative_writes),
            "cumulative_reads_bytes": int(round(cumulative_writes * amplification)),
            "read_amplification": float(amplification),
            "cleanup_margin_bytes": int(cleanup_margin),
            "total_peak_bytes": int(worst_tile_bytes + cleanup_margin),
            "layout": "frame-major contiguous row slices",
            "unknown_terms": list(_UNKNOWN_DISK_TERMS),
            "unknown_bytes": None,
        },
        "io": {
            "write_bytes_per_cell": int(cumulative_writes),
            "read_bytes_per_cell": int(round(cumulative_writes * amplification)),
        },
    }


def shrink_frozen_column_batch(
    *,
    frozen_batch: int,
    phase_budget_bytes: int,
    n_frames: int,
    height: int,
    width: int,
    channels: int,
    tile_size,
    halo_px: int,
    equalize_rgb: bool = False,
    rejection: str = "winsorized_sigma_clip",
    combine: str = "mean",
) -> dict | None:
    """Largest STRICTLY SMALLER batch whose EXACT host peak fits the budget.

    ZM-ZEGRID-FROZEN-BATCH-SHRINK: a resumed ``external_spool`` run freezes its
    ``column_batch`` at creation. The batch is a STAGING parameter, not a
    science parameter (bit-identical payloads proven across 8/64/512 on real
    M106 data), so a resume may re-derive a SMALLER one when, and only when,
    the frozen value no longer fits the phase-local budget.

    The ladder is finite, deterministic and derived from the SAME accounting
    the ``frozen_batch_ram`` refusal already uses:

    1. the exact admissible-batch inversion
       (:func:`largest_admissible_column_batch`) over the residual budget
       ``phase_budget_bytes - reduce_fixed_bytes``;
    2. then the halving cascade of that candidate down to one pixel, always
       STRICTLY below ``frozen_batch``.

    Every candidate is priced with :func:`estimate_external_working_set` (the
    SAME host model as the refusal) and the FIRST (largest) candidate whose
    ``host.peak_bytes`` fits is accepted. Returns ``None`` when the frozen
    batch already fits (nothing to do, always the case for a fresh run) or
    when NO smaller batch fits (the caller keeps today's refusal byte-exact).
    It never returns a value >= ``frozen_batch``.
    """
    frozen = _require_int(frozen_batch, "frozen_batch", minimum=1)
    budget = _require_int(phase_budget_bytes, "phase_budget_bytes", minimum=1)
    n = _require_int(n_frames, "n_frames", minimum=1)
    h = _require_int(height, "height", minimum=1)
    w = _require_int(width, "width", minimum=1)
    c = _require_int(channels, "channels", minimum=1)
    halo = _require_int(halo_px, "halo_px")

    def _price(batch: int) -> dict:
        return estimate_external_working_set(
            n_frames=n,
            height=h,
            width=w,
            channels=c,
            tile_size=tile_size,
            halo_px=halo,
            column_batch=int(batch),
            equalize_rgb=bool(equalize_rgb),
            rejection=str(rejection),
            combine=str(combine),
        )

    frozen_pricing = _price(frozen)
    frozen_host = frozen_pricing["host"]
    frozen_peak = int(frozen_host["peak_bytes"])
    if frozen_peak <= budget:
        return None
    reduce_fixed = int(frozen_host["reduce_fixed_bytes"])
    residual = int(budget) - int(reduce_fixed)
    tiles, _max_th, _max_tw = _tile_geometry(h, w, tile_size, halo)
    max_ih = max(int(y1 - y0) for (y0, y1, _x0, _x1, _th, _tw) in tiles)
    max_iw = _max_interior_width(tiles)

    ladder: list[int] = []
    if residual >= 1:
        try:
            inversion = int(
                largest_admissible_column_batch(
                    budget_bytes=int(residual),
                    n_frames=int(n),
                    interior_width=int(max_iw),
                    channels=int(c),
                    fixed_position_bytes=int(
                        EXTERNAL_BATCH_FIXED_POSITION_BYTES
                    ),
                )
            )
        except ExternalBudgetError:
            inversion = 0
        if 1 <= inversion < frozen:
            ladder.append(int(inversion))
            candidate = int(inversion)
            while candidate > 1:
                candidate = max(1, candidate // 2)
                if candidate < frozen and candidate not in ladder:
                    ladder.append(int(candidate))
    if 1 < frozen and 1 not in ladder:
        # The declared floor: ONE pixel (``channels`` positions). It is only
        # reached when every larger rung failed the exact host pricing.
        ladder.append(1)
    trials: list[dict] = []
    for candidate in ladder:
        if not 1 <= int(candidate) < frozen:
            continue
        pricing = _price(candidate)
        host = pricing["host"]
        peak = int(host["peak_bytes"])
        fits = peak <= budget
        trials.append(
            {
                "column_batch": int(candidate),
                "host_peak_bytes": int(peak),
                "fits": bool(fits),
            }
        )
        if fits:
            return {
                "schema": "zemosaic-zegrid-frozen-batch-shrink-v1",
                "from_batch": int(frozen),
                "to_batch": int(candidate),
                "frozen_batch": int(frozen),
                "required_peak_bytes": int(frozen_peak),
                "admitted_peak_bytes": int(peak),
                "phase_budget_bytes": int(budget),
                "reduce_fixed_bytes": int(reduce_fixed),
                "residual_batch_budget_bytes": int(residual),
                "reason": (
                    "frozen column_batch "
                    f"{frozen} yields a host peak of {frozen_peak} B but only "
                    f"{budget} B remain in the phase-local budget; the "
                    "largest smaller admissible batch "
                    f"{candidate} fits at {peak} B (staging-only change, "
                    "identity unchanged)"
                ),
                "ladder": trials,
                "requested_column_batch": int(
                    host["requested_column_batch"]
                ),
                "effective_column_batch": int(
                    host["effective_column_batch"]
                ),
                "column_batch_clamped": bool(
                    host["column_batch_clamped"]
                ),
            }
    return None


def largest_admissible_column_batch(
    *,
    budget_bytes: int,
    n_frames: int,
    interior_width: int,
    channels: int,
    floor_positions: int = COLUMN_BATCH_MIN,
    fixed_position_bytes: int = 0,
) -> int:
    """Largest PIXEL-granular batch that fits ``budget_bytes``.

    The unit is one pixel (``channels`` positions), NOT one interior row. A
    batch is therefore any whole number of pixels and may be narrower than a
    row. :func:`batch_positions_for` already turns such a request into 2D
    micro-blocks of at least one pixel, and :func:`_gather_block` reduces them
    bit-exactly with the full-row path (documented 2D tiling), so the product
    decider may select a sub-line batch instead of refusing outright.

    ``interior_width`` no longer floors the batch: one full row is merely the
    natural upper shape the caller passes to
    :func:`estimate_external_working_set`. Above one full row the result stays
    a whole-row multiple (historical contiguous-slice behavior); below it, it
    stays a pixel multiple. Refuses when even ONE pixel (or
    ``floor_positions``, whichever is larger) does not fit.

    ``fixed_position_bytes`` is the allocation-independent ``(positions,)``
    term (quantile/rejection/combine temporaries). The per-position cost is
    ``n_frames * sample_bytes + fixed_position_bytes``; a fixed per-sample
    coefficient alone is NOT a valid inversion at low N.
    """
    budget = _require_int(budget_bytes, "budget_bytes", minimum=1)
    n = _require_int(n_frames, "n_frames", minimum=1)
    iw = _require_int(interior_width, "interior_width", minimum=1)
    c = _require_int(channels, "channels", minimum=1)
    floor = _require_int(floor_positions, "floor_positions", minimum=1)
    fixed = _require_int(fixed_position_bytes, "fixed_position_bytes")
    unit = max(c, floor)  # one pixel == ``channels`` positions
    # COMPLETE two-part cost: N-scaled sample arrays + fixed per-position
    # quantile/rejection/combine arrays. Never a fixed per-sample coefficient.
    per_position = (
        n * EXTERNAL_BATCH_SAMPLE_BYTES_PER_POSITION_FRAME + int(fixed)
    )
    if budget < unit * per_position:
        raise ExternalBudgetError(
            f"budget {budget} B cannot hold the minimum batch of {unit} "
            f"positions x {n} frames ({unit * per_position} B required: "
            f"{n} x {EXTERNAL_BATCH_SAMPLE_BYTES_PER_POSITION_FRAME} B/sample "
            f"+ {fixed} B/position fixed)"
        )
    positions = (budget // per_position // c) * c
    ppr = iw * c
    if positions >= ppr:
        # Above one full row keep the historical whole-row multiple, so the
        # reduction still reads contiguous row slices. Only the FLOOR moved
        # down to one pixel; the above-row granularity is unchanged.
        positions = (positions // ppr) * ppr
    return int(max(unit, positions))


# ---------------------------------------------------------------------------
# Exact column-batch reduction
# ---------------------------------------------------------------------------

def _gather_block(
    images,
    weights,
    n: int,
    tw: int,
    c: int,
    iy0: int,
    ix0: int,
    w_int: int,
    a: int,
    b: int,
    x0: int,
    x1: int,
):
    """All-N contiguous slices for one interior 2D block (+ raw weights).

    The block spans interior rows ``[a, b)`` and interior columns ``[x0, x1)``
    (``x0``/``x1`` are relative to the interior left edge ``ix0``). Positions are
    row-major over ``(row, column, channel)`` within the block — identical to the
    full-row gather restricted to the column window ``[x0, x1)``, so a 2D tiling
    is bit-exact with the full-row path.
    """
    bw = x1 - x0
    positions = (b - a) * bw * c
    img = np.empty((n, positions), dtype=np.float32)
    wt = np.empty((n, positions), dtype=np.float64)
    for i in range(n):
        parts = []
        for iy in range(a, b):
            start = ((iy0 + iy) * tw + ix0 + x0) * c
            parts.append(images[i].reshape(-1)[start:start + bw * c])
        img[i] = parts[0] if len(parts) == 1 else np.concatenate(parts)
        w_rows = np.asarray(
            weights[i][iy0 + a:iy0 + b, ix0 + x0:ix0 + x1], dtype=np.float64
        )
        wt[i] = np.repeat(w_rows, c, axis=1).reshape(-1)
    return img, wt


def reduce_spooled_tile(
    tile: TileSpool,
    p1: _Phase1,
    request: CanonicalStackRequest,
    *,
    column_batch: int,
    rejection_sink=None,
    rejection_mask=None,
    backend: str = "cpu",
) -> dict:
    """Reduce ONE spooled tile with exact contiguous row micro-batches.

    Returns the tile-local planes and this tile's scalar diagnostics. Only the
    INTERIOR is reduced (the RAM path writes/counts only the interior); the
    extended border exists solely for the taper and is never consumed.
    """
    validate_tile(tile)
    images = np.load(tile.images_path, mmap_mode="r")
    weights = np.load(tile.weights_path, mmap_mode="r")
    n = int(p1.n_frames)
    c = int(p1.channels)
    th, tw = int(tile.th), int(tile.tw)
    iy0 = int(tile.y0 - tile.ey0)
    ix0 = int(tile.x0 - tile.ex0)
    h_int = int(tile.y1 - tile.y0)
    w_int = int(tile.x1 - tile.x0)
    if h_int <= 0 or w_int <= 0:
        raise ExternalSpoolCorruptError("tile interior is empty")

    reject_token = _resolve_reject_token(request.rejection)
    combine_token = _resolve_method_token(
        request.combine, _COMBINE_METHODS, "combine"
    )
    s_low = _validate_reject_sigma(request.sigma_low, "sigma_low")
    s_high = _validate_reject_sigma(request.sigma_high, "sigma_high")
    iters = _validate_max_iters(request.max_iters)
    w_low = _validate_winsor_limit(request.winsor_limit_low, "winsor_limit_low")
    w_high = _validate_winsor_limit(request.winsor_limit_high, "winsor_limit_high")
    if w_low + w_high >= 1.0:
        raise CanonicalStackValidationError(
            f"winsor_limit_low + winsor_limit_high must be < 1, got {w_low} + {w_high}"
        )

    # A3 — backend-neutral reduction: one shared xp-generic algorithm on NumPy
    # (CPU, default) or CuPy (GPU, opt-in). ``_resolve_backend`` raises for an
    # unavailable GPU (never a silent CPU fallback); the rejection step
    # primitives and the mean/median reductions are bit-exact across backends
    # (see canonical_stacking), so CPU and GPU produce identical host outputs.
    backend_token = _resolve_backend(backend)
    xp = _get_xp(backend_token)

    active = np.asarray(p1.weight_active, dtype=bool)
    q = np.asarray(p1.weights, dtype=np.float64)

    science_t = np.empty((th, tw, c), dtype=np.float32)
    weight_sum_t = np.empty((th, tw, c), dtype=np.float64)
    valid_mask_t = np.empty((th, tw, c), dtype=bool)
    surviving_t = np.empty((th, tw, c), dtype=np.int64)

    # --- support: exact sequential per-frame accumulation over the interior ---
    # (same order and same values as the RAM `w1_tile += wmap[i]` loop).
    support_w1 = np.zeros((h_int, w_int), dtype=np.float64)
    support_w2 = np.zeros((h_int, w_int), dtype=np.float64)
    for i in range(n):
        if active[i]:
            wi = q[i] * np.asarray(
                weights[i][iy0:iy0 + h_int, ix0:ix0 + w_int], dtype=np.float64
            )
            support_w1 += wi
            support_w2 += wi * wi

    rows_per_batch, cols_per_batch = batch_positions_for(
        column_batch, w_int, c
    )
    initial_count = 0
    rejected_count = 0
    low_n_count = 0
    degenerate_count = 0
    tile_iters = 0
    # ZM-ZEGRID-BATCH-TRUTH-AND-USAGE: the EXECUTED block geometry is recorded
    # by the reducer itself (not re-derived by a caller). ``max_block_positions``
    # is the largest live column-batch block this tile actually formed.
    n_blocks = 0
    max_block_positions = 0

    for a in range(0, h_int, rows_per_batch):
        b = min(h_int, a + rows_per_batch)
        for x0 in range(0, w_int, cols_per_batch):
            x1 = min(w_int, x0 + cols_per_batch)
            npos = (b - a) * (x1 - x0) * c
            n_blocks += 1
            if npos > max_block_positions:
                max_block_positions = int(npos)
            img_b, w_b = _gather_block(
                images, weights, n, tw, c, iy0, ix0, w_int, a, b, x0, x1
            )
            orig_b = img_b.astype(np.float64)
            finite_b = np.isfinite(img_b)
            # wmap = q * (valid ? taper : 0), zero outside valid (RAM expression).
            wmap_b = q[:, None] * w_b
            wmap_b = np.where(finite_b, wmap_b, 0.0)
            initial_b = active[:, None] & finite_b

            survivor_b = initial_b.copy()
            degenerate_b = np.zeros(npos, dtype=bool)
            if reject_token == "none":
                batch_iters = 0
            else:
                batch_iters = 0
                # xp-generic rejection: the step primitives use the bit-exact
                # host-side mean/std reductions, so the survivor mask is identical
                # CPU vs GPU. Upload once, iterate on the device, download once.
                orig2 = xp.asarray(orig_b)
                survivor2 = xp.asarray(survivor_b)
                degenerate_seen = xp.zeros(npos, dtype=bool)
                for _ in range(iters):
                    count = survivor2.sum(axis=0)
                    if reject_token == "kappa_sigma":
                        new2, degenerate = _kappa_sigma_step(
                            orig2, survivor2, count, s_low, s_high, xp
                        )
                    else:
                        new2, degenerate = _winsorized_sigma_clip_step(
                            orig2, survivor2, count, s_low, s_high,
                            w_low, w_high, xp,
                        )
                    batch_iters += 1
                    degenerate_seen |= degenerate
                    if bool(xp.array_equal(new2, survivor2)):
                        survivor2 = new2
                        break
                    survivor2 = new2
                survivor_b = _as_host_numpy(survivor2, xp)
                degenerate_b = _as_host_numpy(degenerate_seen, xp)
            tile_iters = max(tile_iters, batch_iters)

            rejection_b = initial_b & ~survivor_b
            eligible_b = survivor_b & (wmap_b > 0.0) & finite_b

            if combine_token == "mean":
                images64_xp = xp.asarray(orig_b)
                wmap_xp = xp.asarray(wmap_b)
                eligible_xp = xp.asarray(eligible_b)
                masked = xp.where(eligible_xp, images64_xp, 0.0)
                w_contrib = xp.where(eligible_xp, wmap_xp, 0.0)
                # _axis_sum_f64 reduces on the host for the GPU backend (matching
                # np.sum), so the combined science is bit-identical CPU vs GPU.
                numerator = _axis_sum_f64(masked * w_contrib, xp)
                denominator = _axis_sum_f64(w_contrib, xp)
                den_valid = denominator > 0.0
                estimate64 = _safe_divide(numerator, denominator, np)
                science32 = _astype_float32(estimate64, np)
                final_finite = np.isfinite(estimate64) & np.isfinite(science32)
                valid_out = den_valid & final_finite
                science_b = np.where(valid_out, science32, np.nan)
                weight_sum_b = np.where(valid_out, denominator, 0.0)
            else:  # median
                images64_xp = xp.asarray(orig_b)
                eligible_xp = xp.asarray(eligible_b)
                masked = xp.where(eligible_xp, images64_xp, xp.nan)
                median64 = _as_host_numpy(_nan_axis_median(masked, xp), xp)
                science32 = _astype_float32(median64, np)
                count_cells = eligible_b.sum(axis=0).astype(np.float64)
                den_valid = count_cells > 0.0
                final_finite = np.isfinite(median64) & np.isfinite(science32)
                valid_out = den_valid & final_finite
                science_b = np.where(valid_out, science32, np.nan)
                weight_sum_b = np.where(valid_out, count_cells, 0.0)

            surviving_b = eligible_b.sum(axis=0).astype(np.int64)
            valid_mask_b = weight_sum_b > 0.0

            # The interior is NOT contiguous in the extended tile's flat order:
            # write it into the exact interior window (row range × column range).
            tile_rows = slice(iy0 + a, iy0 + b)
            tile_cols = slice(ix0 + x0, ix0 + x1)
            science_t[tile_rows, tile_cols, :] = science_b.reshape(b - a, x1 - x0, c)
            weight_sum_t[tile_rows, tile_cols, :] = weight_sum_b.reshape(b - a, x1 - x0, c)
            valid_mask_t[tile_rows, tile_cols, :] = valid_mask_b.reshape(b - a, x1 - x0, c)
            surviving_t[tile_rows, tile_cols, :] = surviving_b.reshape(b - a, x1 - x0, c)

            block = rejection_b.reshape(n, b - a, x1 - x0, c)
            if rejection_sink is not None:
                rejection_sink[:, tile.y0 + a:tile.y0 + b, tile.x0 + x0:tile.x0 + x1, :] = block
            else:
                rejection_mask[:, tile.y0 + a:tile.y0 + b, tile.x0 + x0:tile.x0 + x1, :] = block

            init_sum = initial_b.sum(axis=0)
            initial_count += int(init_sum.sum())
            rejected_count += int(rejection_b.sum())
            low_n_count += int((init_sum < 3).sum())
            degenerate_count += int(degenerate_b.sum())

    return {
        "science": science_t,
        "weight_sum": weight_sum_t,
        "valid_mask": valid_mask_t,
        "surviving": surviving_t,
        "support_w1": support_w1,
        "support_w2": support_w2,
        "tile_iters": int(tile_iters),
        "initial_count": int(initial_count),
        "rejected_count": int(rejected_count),
        "low_n_count": int(low_n_count),
        "degenerate_count": int(degenerate_count),
        # Executed (measured) block geometry: what this tile REALLY reduced.
        "batch_geometry": {
            "column_batch": int(column_batch),
            "interior_hw": [int(h_int), int(w_int)],
            "rows_per_batch": int(rows_per_batch),
            "cols_per_batch": int(cols_per_batch),
            "n_blocks": int(n_blocks),
            "max_block_positions": int(max_block_positions),
        },
    }


# ---------------------------------------------------------------------------
# Public executor
# ---------------------------------------------------------------------------

def run_cell_streaming_external(
    provider,
    request: CanonicalStackRequest,
    *,
    tile_size,
    column_batch: int,
    spool_root,
    fixed=None,
    rejection_sink=None,
    cell_id: str = "cell",
    attempt_id: str = "attempt",
    keep_spool: bool = False,
    cleanup_on_success: bool = True,
    spool_out: dict | None = None,
    owner_token: str | None = None,
    owner_record: dict | None = None,
    owner_liveness=None,
    stale_policy: str = "discard",
    retain_on_failure: bool = False,
    gpu_fallback_out: dict | None = None,
) -> CanonicalStackResult:
    """External-working-set execution of ONE canonical request.

    Bit-equivalent to ``run_canonical_stack_streaming(..., tile_size=tile_size)``
    for the same provider/request/fixed/sink, but the per-tile N-axis values are
    spooled to disk and reduced in column rows.
    """
    if not isinstance(request, CanonicalStackRequest):
        raise CanonicalStackValidationError(
            f"request must be a CanonicalStackRequest, got {type(request).__name__}"
        )
    # A3 — the external working set now runs CPU or GPU (opt-in). ``_resolve_backend``
    # validates the token and raises for an unavailable GPU (never a silent CPU
    # fallback INSIDE the executor — the ZeGrid orchestrator owns the loud degrade).
    token = _resolve_backend(request.backend)
    # F8: the REQUESTED backend is kept for provenance; the EFFECTIVE backend
    # starts identical and becomes "cpu" after the first classified GPU runtime
    # failure. From then on the whole Cell reduces on CPU (sticky fallback, one
    # GPU->CPU transition) and the published provenance names the real backend.
    effective_token = str(token)
    gpu_active = str(token) == "gpu"
    for attr in ("n_frames", "height", "width", "channels", "original_mono"):
        if not hasattr(provider, attr):
            raise CanonicalStackValidationError(f"provider is missing attribute {attr!r}")
    if provider.n_frames != len(request.images):
        raise CanonicalStackValidationError(
            f"provider.n_frames {provider.n_frames} != request.images length "
            f"{len(request.images)}"
        )
    c = int(provider.channels)
    n = int(provider.n_frames)
    h = int(provider.height)
    w = int(provider.width)
    if int(column_batch) < 1 or isinstance(column_batch, bool):
        raise CanonicalStackValidationError(
            f"column_batch must be an int >= 1, got {column_batch!r}"
        )

    taper_kind, taper_px, taper_floor, halo = _resolve_taper(request)
    if rejection_sink is not None:
        _validate_rejection_sink(rejection_sink, n, h, w, c)

    p1 = _fixed_phase1(provider, request, fixed)

    science_c = np.empty((h, w, c), dtype=np.float32)
    weight_sum_c = np.zeros((h, w, c), dtype=np.float64)
    valid_mask_c = np.zeros((h, w, c), dtype=bool)
    surviving_c = np.zeros((h, w, c), dtype=np.int64)
    support_w1 = np.zeros((h, w), dtype=np.float64)
    support_w2 = np.zeros((h, w), dtype=np.float64)

    if rejection_sink is None:
        rejection_mask = np.zeros((n, h, w, c), dtype=bool)
        rejection_mask_out = rejection_mask
    else:
        rejection_mask = None
        rejection_mask_out = None

    iterations_used = 0
    initial_count = 0
    rejected_count = 0
    low_n_count = 0
    degenerate_count = 0
    # ZM-ZEGRID-BATCH-TRUTH-AND-USAGE: the reducer reports the block geometry it
    # ACTUALLY formed for every tile; this is the executed batch truth, never a
    # re-derivation from the request.
    executed_tiles: list[dict] = []

    spec = ExternalSpoolSpec(
        root=spool_root,
        n_frames=n,
        height=h,
        width=w,
        channels=c,
        halo_px=halo,
        tile_size=tile_size,
        cell_id=cell_id,
        attempt_id=attempt_id,
        frame_ids=tuple(provider.frame_ids) if hasattr(provider, "frame_ids") else None,
        owner_token=owner_token,
        owner_record=owner_record,
        owner_liveness=owner_liveness,
        stale_policy=stale_policy,
        retain_on_failure=retain_on_failure,
    )
    spool = ExternalSpool(spec)
    gpu_fallback_tiles = []
    try:
        for (y0, y1, x0, x1) in _iter_tiles(h, w, tile_size):
            handle = spool.write_tile(
                provider, p1, y0, y1, x0, x1,
                taper_kind=taper_kind,
                taper_px=taper_px,
                taper_floor=taper_floor,
                halo_px=halo,
            )
            sink_cp = (
                rejection_sink.checkpoint() if rejection_sink is not None else None
            )
            try:
                if gpu_active:
                    try:
                        tile = reduce_spooled_tile(
                            handle, p1, request,
                            column_batch=int(column_batch),
                            rejection_sink=rejection_sink,
                            rejection_mask=rejection_mask,
                            backend="gpu",
                        )
                    except Exception as exc:
                        # A3 — transactional fallback: a classified GPU runtime
                        # failure (OOM / CUDA error) re-reduces the SAME spooled
                        # tile on CPU (bit-exact across backends). The counting
                        # sink is rolled back to its pre-tile checkpoint so the
                        # replayed tile is counted exactly once. A science or
                        # programming error is NEVER reclassified: it propagates.
                        if not is_gpu_runtime_error(exc):
                            raise
                        if rejection_sink is not None:
                            rejection_sink.rollback(sink_cp)
                        # F8: sticky fallback — ONE GPU->CPU transition per Cell.
                        # Every following tile goes straight to CPU (no repeated
                        # GPU attempts), and the effective backend becomes CPU.
                        gpu_active = False
                        effective_token = "cpu"
                        tile = reduce_spooled_tile(
                            handle, p1, request,
                            column_batch=int(column_batch),
                            rejection_sink=rejection_sink,
                            rejection_mask=rejection_mask,
                            backend="cpu",
                        )
                        gpu_fallback_tiles.append((int(y0), int(x0)))
                else:
                    tile = reduce_spooled_tile(
                        handle, p1, request,
                        column_batch=int(column_batch),
                        rejection_sink=rejection_sink,
                        rejection_mask=rejection_mask,
                        backend="cpu",
                    )
            finally:
                spool.discard_tile(handle)
            if rejection_sink is not None:
                rejection_sink.flush_batch()

            iy0 = int(handle.y0 - handle.ey0)
            ix0 = int(handle.x0 - handle.ex0)
            h_int = int(handle.y1 - handle.y0)
            w_int = int(handle.x1 - handle.x0)
            science_c[y0:y1, x0:x1] = tile["science"][
                iy0:iy0 + h_int, ix0:ix0 + w_int
            ]
            weight_sum_c[y0:y1, x0:x1] = tile["weight_sum"][
                iy0:iy0 + h_int, ix0:ix0 + w_int
            ]
            valid_mask_c[y0:y1, x0:x1] = tile["valid_mask"][
                iy0:iy0 + h_int, ix0:ix0 + w_int
            ]
            surviving_c[y0:y1, x0:x1] = tile["surviving"][
                iy0:iy0 + h_int, ix0:ix0 + w_int
            ]
            support_w1[y0:y1, x0:x1] = tile["support_w1"]
            support_w2[y0:y1, x0:x1] = tile["support_w2"]

            iterations_used = max(iterations_used, int(tile["tile_iters"]))
            initial_count += int(tile["initial_count"])
            rejected_count += int(tile["rejected_count"])
            low_n_count += int(tile["low_n_count"])
            degenerate_count += int(tile["degenerate_count"])
            if "batch_geometry" in tile:
                executed_tiles.append(dict(tile["batch_geometry"]))

        spool.finalize_manifest(
            taper_kind=taper_kind, taper_px=taper_px, taper_floor=taper_floor
        )
        if gpu_fallback_out is not None:
            gpu_fallback_out.update({
                "fallback_count": len(gpu_fallback_tiles),
                "fallback_tiles": list(gpu_fallback_tiles),
                "requested_backend": str(token),
            })
        # M4: the attempt is READY (result computed) but NOT committed. The
        # caller commits the Cell durably first, then cleans it.
        spool.mark_ready()
        if spool_out is not None:
            manifest = spool.load_manifest()
            payload = int(manifest.get("total_payload_bytes", 0))
            _max_pos = max(
                (int(g["max_block_positions"]) for g in executed_tiles),
                default=0,
            )
            _max_tile = next(
                (g for g in executed_tiles
                 if int(g["max_block_positions"]) == _max_pos),
                None,
            )
            spool_out.update({
                "attempt_id": str(spool.attempt_id),
                "spool_root": str(spool.root),
                "attempt_dir": str(spool.attempt_dir),
                "state": SPOOL_STATE_READY,
                "bytes_written": int(payload),
                "bytes_read": int(payload),
                "tile_count": int(manifest.get("tile_count", 0)),
                "executed_batch": {
                    "schema": "zemosaic-zegrid-executed-batch-v1",
                    "requested_column_batch": int(column_batch),
                    "tiles": len(executed_tiles),
                    "max_block_positions": int(_max_pos),
                    "max_block_interior_hw": (
                        None if _max_tile is None
                        else [int(_max_tile["interior_hw"][0]),
                              int(_max_tile["interior_hw"][1])]
                    ),
                    "rows_per_batch_at_max": (
                        None if _max_tile is None
                        else int(_max_tile["rows_per_batch"])
                    ),
                    "cols_per_batch_at_max": (
                        None if _max_tile is None
                        else int(_max_tile["cols_per_batch"])
                    ),
                    "max_blocks_in_one_tile": max(
                        (int(g["n_blocks"]) for g in executed_tiles),
                        default=0,
                    ),
                    "distinct_rows_per_batch": sorted({
                        int(g["rows_per_batch"]) for g in executed_tiles
                    }),
                    "distinct_cols_per_batch": sorted({
                        int(g["cols_per_batch"]) for g in executed_tiles
                    }),
                },
            })
    except BaseException:
        if not retain_on_failure:
            try:
                spool.cleanup()
            except Exception:
                pass
        raise

    if cleanup_on_success and not keep_spool:
        spool.cleanup()

    (science, weight_sum, valid_mask, surviving, support_w1, support_w2, n_eff, agg) = (
        _finalize_phase2_outputs(
            p1,
            science_c=science_c,
            weight_sum_c=weight_sum_c,
            valid_mask_c=valid_mask_c,
            surviving_c=surviving_c,
            support_w1=support_w1,
            support_w2=support_w2,
            iterations_used=iterations_used,
            initial_count=initial_count,
            rejected_count=rejected_count,
            low_n_count=low_n_count,
            degenerate_count=degenerate_count,
            rejection_mask_out=rejection_mask_out,
            prefetch_report=None,
        )
    )
    return finalize_streaming_result(
        provider, request, p1,
        science=science, weight_sum=weight_sum, valid_mask=valid_mask,
        surviving=surviving, support_w1=support_w1, support_w2=support_w2,
        n_eff=n_eff, agg=agg, backend_token=effective_token,
        backend_requested_token=token,
        tile_prefetch_plan={"requested": "off"},
    )
