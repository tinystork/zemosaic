"""ZM-ZEGRID-R7 — production entry: run the NEW ZeGrid engine end-to-end.

Thin orchestrator over ``src/zemosaic/core/zegrid/*``. It does **NOT**
re-implement the canonical pipeline or the R1-R6 science; it wires the frozen
foundation modules into the production ``stack_plan.csv`` path:

* **FRAME-MAJOR decode-once** — each frame is decoded ONCE via the existing
  product decoder (:func:`zemosaic.zemosaic_utils.load_image_with_optional_alpha`,
  i.e. ``load_and_validate_fits`` + ``debayer_image``), then for every Cell whose
  patch that frame touches the required source ROI is reprojected and
  APPENDED to that Cell's aligned disk cache (R6 ``file_provider`` format,
  never ``/tmp``). Peak memory is O(1 frame).
* **LAYOUT** — R4 RAM-aware Auto layout (:func:`auto_layout.choose_layout`),
  budgeted from the PORTABLE probe ``psutil.virtual_memory().available``.
* **PER-CELL mode policy** — in-memory canonical (:func:`science_adapter.
  run_minitile_stack`) when the R4 bound fits the budget, else the R6 streaming
  executor (:func:`canonical_streaming.run_canonical_stack_streaming` via the
  file-backed :class:`file_provider.MemmapCanonicalProvider`). Both are bit-equal.
* **ASSEMBLY** — R3 :func:`mosaic.assemble_canvas` -> legacy-compatible
  ``mosaic_grid.fits`` + ``mosaic_grid_coverage.fits`` + a JSON manifest.

Normalization is ``sky_mean`` by DEFAULT (Tristan's decision; the frozen
``linear_fit`` is known-defective on real data). An explicit ``linear_fit`` is
honoured but logs a clear WARNING (see ``core/zegrid/science_adapter.py``).

There is **NO silent engine fallback**: any ZeGrid failure raises (mirroring the
removed legacy Grid abort). The legacy Grid engine was REMOVED (ZM-ZEGRID-R8)
and is ARCHIVED at ``origin/archive/zegrid-legacy-grid-5.0.0``; ``stack_plan.csv``
routes DIRECTLY to this ZeGrid path (no ``grid_engine`` setting).

## Reproducibility (IMPORTANT, durable)

The output science ``mosaic_grid.fits`` depends on the chosen **layout**: each
Cell's ``sky_mean`` normalization is computed over that Cell's halo-extended
patch, so a different (nx, ny) partition shifts the per-frame additive offset
slightly and thus the science values. The **coverage** map
(``mosaic_grid_coverage.fits``, per-pixel stack depth) is pure geometry and is
**layout-invariant**. The layout is chosen RAM-adaptively from
``psutil.virtual_memory().available`` by default, so the SAME ``stack_plan.csv``
can produce slightly different science on different hosts / at different times.

To make a run **reproducible**, pin the layout with the ``zegrid_layout`` config
key (e.g. ``"6x5"``). When set, the RAM-adaptive choice is bypassed and exactly
that layout is used (scientific floors are still enforced; an infeasible pinned
layout raises). The chosen layout and its source (``"auto"`` | ``"pinned"``) are
recorded in ``zegrid_manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import psutil
from astropy.io import fits
from astropy.wcs import WCS

from . import eta_service as zeta
from . import zemosaic_stack_plan as _stack_plan
from .zegrid_halo import (
    ZEGRID_HALO_CONFIG_KEY,
    ZEGRID_HALO_DEFAULT_PX,
    ZEGRID_HALO_SOURCE_EXPLICIT,
    ZEGRID_HALO_SOURCE_INTERNAL_ONLY,
    ZEGRID_HALO_SOURCE_PRODUCT_DEFAULT,
    ZeGridHaloError,
    build_halo_provenance,
    resolve_run_halo_provenance,
    resolve_zegrid_halo_px,
)
from .zemosaic_utils import (
    NONFINITE_FILL_VALUE,
    load_image_with_optional_alpha,
    record_nonfinite_fill,
    sanitize_nonfinite_float32,
)
from .core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
    subset_fixed_normalization,
)
from .core.zegrid import assembly as za
from .core.zegrid import auto_layout as zal
from .core.zegrid import cell_feather as zcf
from .core.zegrid import cell_offset as zco
from .core.zegrid import execution as zxe
from .core.zegrid import file_provider as zfp
from .core.zegrid import final_mosaic_finishing as zfin
from .core.zegrid import geometry as zg
from .core.zegrid import gpu as zgpu
from .core.zegrid import gpu_readiness as zgr
from .core.zegrid import instrumentation as zin
from .core.zegrid import mosaic as zmosaic
from .core.zegrid import observability as zobs
from .core.zegrid import parallel as zpar
from .core.zegrid import photometric as zphot
from .core.zegrid import ram_contract as rcontract
from .core.zegrid import resource_envelope as renv
from .core.zegrid import scheduler as zsched
from .core.zegrid import science_adapter as zs
from .core.zegrid import streaming as zstream
from .core.zegrid import sweep as zsw
from .core.zegrid.executor import ExecutorConfig

logger = logging.getLogger("ZeMosaicWorker").getChild("zegrid_mode")

ProgressCallback = Optional[Callable[[str, object, str], None]]

# Disk cache directory name (under the per-mount output folder). NEVER /tmp.
CACHE_DIR_NAME = "__zegrid_cache__"
# Run-log filename written into the run's output folder (the user reported it absent).
RUN_LOG_NAME = "zegrid_run.log"
# Streaming tile size for the R6 executor (safe, memory-bounded default).
STREAM_TILE_SIZE = 128

# ZM-ZEGRID-R37-D: fixed diagnostic-capture contract. The capture artifact is a
# diagnostic-only, default-OFF pre-offset/pre-feather snapshot; its basename is
# FIXED (no caller-controlled path) and its stage is explicit. The default
# held-out cap mirrors ``background_diagnostics.HOLDOUT_CAP`` (a test pins the
# equality) WITHOUT importing the diagnostic module on the disabled path.
R37_CAPTURE_STAGE = "pre_offset_pre_feather"
R37_CAPTURE_FILENAME = "zegrid_r37_pre_feather.json"
R37_CAPTURE_HOLDOUT_CAP_DEFAULT = 64

# ZM-ZEGRID-R36: deterministic CPU streaming-tile ladder, ordered
# performance -> economy. The planner picks the LARGEST tile whose worst-cell
# streaming bound fits the RAM budget. ``128`` is the historical/preferred tile;
# ``32`` is the documented PERFORMANCE FLOOR (never a science floor — geometry,
# Cell/ProcessingPatch/MiniTile, halo, gauge, weights, rejection and combine are
# UNCHANGED). R36 never goes below 32.
CPU_STREAMING_TILE_LADDER = (128, 112, 96, 80, 64, 48, 32)
CPU_STREAMING_TILE_PREFERRED = 128
CPU_STREAMING_TILE_MINIMUM = 32

# Fraction of available memory the R4 in-memory bound must fit within to select
# the in-memory path (headroom for decode/cache transients + allocator variance).
INMEM_AVAILABLE_FRAC = 0.80
# Normalization token for the ZeGrid engine when none is explicitly requested.
DEFAULT_NORMALIZATION = "sky_mean"

# ZM-ZEGRID-R41-B: the bounded admission replan budget. At most this many
# replans (each caused by a RAM observation change) before an explicit,
# quantitative refusal — never an unbounded retry loop.
R41_ADMISSION_MAX_REPLANS = 2
R41_RAM_CONTRACT_SCHEMA = "zemosaic-zegrid-r41-ram-contract-v1"


# ---------------------------------------------------------------------------
# Emit + portable memory probe
# ---------------------------------------------------------------------------

def _emit(msg: str, *, lvl: str = "INFO", callback: ProgressCallback = None, **kwargs) -> None:
    tag = f"[ZEGRID] {msg}"
    level = getattr(logging, str(lvl).upper(), logging.INFO)
    try:
        logger.log(level, tag)
    except Exception:
        pass
    if callback:
        try:
            callback(tag, None, str(lvl).upper(), **kwargs)
        except Exception:
            try:
                logger.debug("Progress callback failed for %s", tag, exc_info=True)
            except Exception:
                pass


def available_memory_bytes() -> int:
    """Portable available-memory probe (psutil). No POSIX-only ``resource``//proc."""
    return int(psutil.virtual_memory().available)


# ZM-ZEGRID-R15: Windows releases file handles asynchronously, so a memmap/``.npy``
# may still be open when the cache is deleted -> ``WinError 32``. Deletion is an
# OPTIMISATION and must NEVER fail a run: retry a few times with a short backoff
# and, on final failure, emit a WARN and continue.
_RMTREE_ATTEMPTS = 3
_RMTREE_BACKOFF_S = 0.2


def _safe_rmtree(path, progress_callback=None) -> bool:
    """Best-effort recursive delete (never raises); emits a WARN on final failure."""
    path = Path(path)
    if not path.exists():
        return True
    last_exc = None
    for attempt in range(_RMTREE_ATTEMPTS):
        try:
            shutil.rmtree(str(path))
            return True
        except OSError as exc:  # noqa: BLE001 - deletion is best-effort by design
            last_exc = exc
            if attempt + 1 < _RMTREE_ATTEMPTS:
                time.sleep(_RMTREE_BACKOFF_S * (attempt + 1))
    _emit(
        f"cache cleanup FAILED (non-fatal): {last_exc}; leftover cache at {path} "
        f"will be reused or ignored",
        lvl="WARN",
        callback=progress_callback,
    )
    return False


# ---------------------------------------------------------------------------
# Normalization policy (sky_mean default; linear_fit honoured + warned)
# ---------------------------------------------------------------------------

def _resolve_normalization(stack_norm_method) -> str:
    """Resolve the ZeGrid normalization token (sky_mean default).

    ``linear_fit`` is honoured (explicit user choice) but a clear WARNING is
    emitted pointing to the known MAD/bright-core defect documented in
    ``core/zegrid/science_adapter.py``. ``none``/``sky_mean`` are passed through;
    anything else (empty / unknown) falls back to ``sky_mean``.
    """
    token = str(stack_norm_method or "").strip().lower()
    if token == "linear_fit":
        _emit(
            "WARNING: linear_fit normalization explicitly selected; this token is "
            "known-defective on real data (MAD/bright-core rejection collapses the "
            "slope — see core/zegrid/science_adapter.py SKY_MEAN_VARIANT_REASON). "
            "sky_mean is the recommended default.",
            lvl="WARN",
        )
        return "linear_fit"
    if token in ("sky_mean", "none"):
        return token
    return DEFAULT_NORMALIZATION


def resolve_normalization(stack_norm_method) -> str:
    """Public wrapper (testable) — see :func:`_resolve_normalization`."""
    return _resolve_normalization(stack_norm_method)


# ---------------------------------------------------------------------------
# GPU preference resolution (ZM-ZEGRID-R22): ONE canonical bool with explicit
# precedence + strict coercion, resolved from the generic argument and the
# product's GPU flags. This is the single source of truth for whether the
# user asked for GPU in the ZeGrid engine.
# ---------------------------------------------------------------------------

# Resolution order (first non-None wins) for the zconfig GPU flags. ``use_gpu_grid``
# is the grid-specific flag (most direct for the ZeGrid engine); ``stack_use_gpu`` /
# ``use_gpu_stack`` are the stacking flags; ``use_gpu_phase5`` is the GUI canonical
# phase-5 checkbox that ``_normalize_gpu_flags`` synchronises the others onto.
_GPU_PREFERENCE_KEYS = ("use_gpu_grid", "stack_use_gpu", "use_gpu_stack", "use_gpu_phase5")


def _coerce_bool_pref(value):
    """Strict bool coercion for a GPU preference flag (None-aware).

    Returns ``True``/``False`` for an explicit value, or ``None`` when the value
    is absent/empty/unparseable (so the caller falls through to the next source).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        t = value.strip().lower()
        if t in {"1", "true", "yes", "on", "enable", "enabled"}:
            return True
        if t in {"0", "false", "no", "off", "disable", "disabled", "none", ""}:
            return False
        return None
    try:
        return bool(value)
    except Exception:
        return None


def resolve_gpu_preference(use_gpu=None, zconfig=None):
    """Resolve ONE canonical GPU preference with explicit precedence.

    Precedence (first explicit, non-None value wins):
      1. the generic ``use_gpu`` argument (explicit caller intent);
      2. ``use_gpu_grid`` (grid-specific GUI flag);
      3. ``stack_use_gpu`` (stacking GPU flag);
      4. ``use_gpu_stack`` (legacy alias);
      5. ``use_gpu_phase5`` (GUI canonical phase-5 flag).

    Returns ``(requested: bool, source: str)``. Unparseable/absent values are
    treated as unset (fall through); the default is ``(False, "default")``.
    """
    if use_gpu is not None:
        v = _coerce_bool_pref(use_gpu)
        if v is not None:
            return bool(v), "argument"
    if zconfig is not None:
        for key in _GPU_PREFERENCE_KEYS:
            v = _coerce_bool_pref(getattr(zconfig, key, None))
            if v is not None:
                return bool(v), key
    return False, "default"


# ---------------------------------------------------------------------------
# Rejection mapping (ZM-ZEGRID-R16): honour the user's stack_reject_algo + sigma
# / winsor choice where the canonical engine supports it; surface anything it
# cannot honour (never silently drop a science choice).
# ---------------------------------------------------------------------------

# Canonical rejection tokens supported by the frozen SCI-05 engine. Aliases are
# NOT accepted; anything else (incl. the removed ``linear_fit_clip``) cannot be
# honoured and must be surfaced, never silently dropped.
_SUPPORTED_REJECT_ALGOS = ("none", "kappa_sigma", "winsorized_sigma_clip")


def _valid_sigma(value) -> float | None:
    """Validate a kappa/sigma value against the canonical engine (finite, > 0)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f <= 0.0:
        return None
    return f


def _valid_winsor_limits(value) -> tuple[float, float] | None:
    """Validate winsor limits: each finite in [0, 0.5) and low+high < 1."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    if not (0.0 <= low < 0.5 and 0.0 <= high < 0.5):
        return None
    if low + high >= 1.0:
        return None
    return (low, high)


def resolve_rejection_science(
    stack_reject_algo,
    stack_kappa_low,
    stack_kappa_high,
    winsor_limits,
) -> tuple[dict, dict]:
    """Map the user's rejection choice into canonical science-config overrides.

    Returns ``(overrides, unhonoured)``:
      * ``overrides`` — dict of :class:`MiniTileScienceConfig` fields the frozen
        engine CAN honour (``rejection``, ``sigma_low/high``, ``winsor_limit_*``).
      * ``unhonoured`` — dict of ``{arg_name: value}`` it CANNOT honour (kept in
        the ignored-run-args list + a WARN, never silent).

    ``kappa_sigma`` -> ``rejection="kappa_sigma"``; ``winsorized_sigma_clip`` ->
    ``rejection="winsorized_sigma_clip"`` (so it is ACTUALLY applied, not the
    frozen ``kappa_sigma``); ``none`` -> ``rejection="none"``. Anything else and
    any out-of-range sigma/winsor value is surfaced, never silently dropped.
    """
    overrides: dict = {}
    unhonoured: dict = {}

    algo = str(stack_reject_algo or "").strip().lower()
    if algo in _SUPPORTED_REJECT_ALGOS:
        overrides["rejection"] = algo
    else:
        unhonoured["stack_reject_algo"] = stack_reject_algo

    sigma_low = _valid_sigma(stack_kappa_low)
    sigma_high = _valid_sigma(stack_kappa_high)
    if sigma_low is not None and sigma_high is not None:
        overrides["sigma_low"] = sigma_low
        overrides["sigma_high"] = sigma_high
    else:
        unhonoured["stack_kappa_low"] = stack_kappa_low
        unhonoured["stack_kappa_high"] = stack_kappa_high

    wl = _valid_winsor_limits(winsor_limits)
    if wl is not None:
        overrides["winsor_limit_low"] = wl[0]
        overrides["winsor_limit_high"] = wl[1]
    else:
        unhonoured["winsor_limits"] = winsor_limits

    return overrides, unhonoured


def _parse_pinned_layout(value) -> tuple[int, int] | None:
    """Parse a pinned layout spec (``"NXxNY"``) into ``(nx, ny)``, or ``None``.

    Accepts ``"NXxNY"`` (also ``"*"`` or ``","`` separators) and a 2-tuple/list.
    Raises ``ValueError`` on an unparseable value (never silently ignored).
    """
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid pinned layout {value!r}") from exc
    text = str(value).strip().lower().replace(" ", "")
    if not text:
        return None
    for sep in ("x", "*", ","):
        if sep in text:
            parts = text.split(sep)
            if len(parts) == 2:
                try:
                    return int(parts[0]), int(parts[1])
                except ValueError as exc:
                    raise ValueError(
                        f"invalid pinned layout {value!r} (expected 'NXxNY')"
                    ) from exc
    raise ValueError(f"invalid pinned layout {value!r} (expected 'NXxNY')")


# ---------------------------------------------------------------------------
# FrameDescriptor construction from legacy FrameInfo
# ---------------------------------------------------------------------------

def _build_frame_descriptors(frames_info, input_folder, progress_callback, *, sip_mode="keep"):
    """Convert legacy ``FrameInfo`` (WCS populated) to ZeGrid ``FrameDescriptor``.

    Each frame's WCS is QUALIFIED (2-D celestial TAN — plain or SIP). Unqualified
    frames are recorded (never silently dropped) and skipped; if NONE qualify a
    ``RuntimeError`` is raised (no silent degradation).

    ``sip_mode`` controls SIP handling:
      * ``"keep"``  (default) — accept and correctly apply SIP distortion.
      * ``"strip"`` — strip the SIP coefficients (legacy-consistent plain TAN).
    Returns ``(descs, rejected)``.
    """
    base = Path(input_folder).expanduser().resolve()
    descs = []
    rejected = []
    for fi in frames_info:
        if fi.wcs is None or fi.shape_hw is None:
            ok = _stack_plan.load_frame_wcs(fi, progress_callback=progress_callback)
            if not ok or fi.wcs is None or fi.shape_hw is None:
                rejected.append({"path": str(fi.path), "reason": "no usable celestial WCS/shape"})
                continue
        wcs = fi.wcs
        if sip_mode == "strip" and zg._has_sip(wcs):
            wcs = zg.strip_sip(wcs)
        reason = zg.qualify_wcs(wcs)
        if reason is not None:
            rejected.append({"path": str(fi.path), "reason": reason})
            continue
        try:
            rel = Path(fi.path).resolve().relative_to(base).as_posix()
        except Exception:
            rel = Path(fi.path).name
        descs.append(
            zg.FrameDescriptor(
                frame_id=zg.FrameId(rel),
                source_path=str(fi.path),
                shape_hw=(int(fi.shape_hw[0]), int(fi.shape_hw[1])),
                wcs_header=zg._serialize_wcs(wcs),
                header_sha256="",
                instrument="",
            )
        )
    if rejected:
        from collections import Counter

        breakdown = Counter(r["reason"] for r in rejected)
        _emit(
            f"ZeGrid: rejected {len(rejected)} frame(s) with unqualified WCS — "
            f"breakdown: " + ", ".join(f"{reason} x{count}" for reason, count in sorted(breakdown.items()))
            + f" | paths: " + ", ".join(r["path"] for r in rejected),
            lvl="WARN",
            callback=progress_callback,
        )
    if not descs:
        raise RuntimeError(
            "ZeGrid: no frames with a qualified (2-D celestial TAN) WCS; cannot build a mosaic"
        )
    descs.sort(key=lambda d: d.frame_id)
    return descs, rejected


def _decode_frame_hwc(frame_desc, progress_callback):
    """Decode ONE raw frame via the existing product decoder -> HWC float32 RGB.

    Uses :func:`zemosaic.zemosaic_utils.load_image_with_optional_alpha` verbatim
    (``load_and_validate_fits`` + ``debayer_image``). Returns ``(H, W, 3)``
    float32. Mono (non-Bayer) inputs are replicated to 3 channels to match the
    RGB-oriented ZeGrid pipeline. Peak memory O(1 frame).
    """
    arr, _weights = load_image_with_optional_alpha(
        Path(frame_desc.source_path), progress_callback=progress_callback
    )
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] != 3:
        raise ValueError(f"unexpected decoded channel count {arr.shape[-1]} (expected 3)")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _cell_cache_frame_worker(task):
    """Module-level (picklable) worker: decode + crop + reproject ONE (frame, cell).

    Task = ``(frame_desc, source_bounds, patch_wcs_header, patch_shape_hw,
    rgb_path, sup_path, index, frame_id)``. Reproduces the serial frame-major
    crop/reproject bit-for-bit (decode -> crop -> moveaxis -> ``slice_wcs`` ->
    ``reproject_cropped``) and writes the ``.npy`` files directly (no array is
    returned over the process boundary, so large-patch parallelism stays cheap).
    """
    (
        frame_desc,
        bounds,
        patch_wcs_header,
        patch_hw,
        rgb_path,
        sup_path,
        index,
        frame_id,
    ) = task
    hwc = _decode_frame_hwc(frame_desc, None)
    crop_hwc = hwc[bounds.y0:bounds.y1, bounds.x0:bounds.x1]
    crop_chw = np.ascontiguousarray(np.moveaxis(crop_hwc, -1, 0))
    cropped_wcs = zxe.slice_wcs(frame_desc.wcs(), bounds)
    rgb, geom = zxe.reproject_cropped(
        crop_chw, cropped_wcs, WCS(patch_wcs_header), tuple(patch_hw)
    )
    rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.float32))
    geom = np.ascontiguousarray(np.asarray(geom, dtype=bool))
    np.save(rgb_path, rgb)
    np.save(sup_path, geom)
    return {
        "index": index,
        "frame_id": frame_id,
        "rgb": os.path.basename(str(rgb_path)),
        "rgb_sha256": zfp._sha256(str(rgb_path)),
        "rgb_bytes": int(os.path.getsize(str(rgb_path))),
        "support": os.path.basename(str(sup_path)),
        "support_sha256": zfp._sha256(str(sup_path)),
        "support_bytes": int(os.path.getsize(str(sup_path))),
    }


def _build_one_cell_cache(
    descs, canvas, cell, patch, mem, cache_dir, *,
    workers: int = 1, reuse_cache: bool = True, progress_callback=None,
):
    """Build ONE cell's aligned disk cache (parallel reproject), resumable.

    Mirrors the serial frame-major build's per-cell output exactly: the same
    sorted-FrameId frame order, the same source-ROI plans, the same crop/slice/
    reproject primitives, and the same ``frame_XXXX_rgb.npy`` / ``_support.npy``
    naming + manifest. The reprojection (the dominant cost) is spread across
    ``workers`` processes; the written bytes are bit-identical to the serial
    build (verified by hash in the R12 tests).
    """
    by_id = {f.frame_id.logical_path: f for f in descs}
    patch_frames = [by_id[k] for k in mem.patch_ids]  # mem.patch_ids is sorted
    planned = []
    for f in patch_frames:
        plan = zg.plan_source_roi(f, canvas, patch)
        if plan is not None:
            planned.append((f, plan))
    frame_ids = [f.frame_id.logical_path for f, _p in planned]

    cache_dir = Path(cache_dir)
    patch_identity = zfp.patch_geometry_identity(patch)
    if reuse_cache and zfp.cache_is_complete(
        str(cache_dir), frame_ids,
        expected_hw=patch.patch_shape_hw,
        expected_patch_identity=patch_identity,
    ):
        return zfp.load_cache_manifest(str(cache_dir))
    # R15: rebuild wipe is best-effort (Windows WinError 32 on open handles).
    _safe_rmtree(cache_dir, progress_callback)
    cache_dir.mkdir(parents=True, exist_ok=True)

    patch_wcs_header = patch.patch_wcs_header
    patch_hw = patch.patch_shape_hw
    tasks = []
    for idx, (f, plan) in enumerate(planned):
        key = f.frame_id.logical_path
        rgb_path = cache_dir / f"frame_{idx:04d}_rgb.npy"
        sup_path = cache_dir / f"frame_{idx:04d}_support.npy"
        tasks.append(
            (
                f, plan.source_bounds, patch_wcs_header, patch_hw,
                str(rgb_path), str(sup_path), idx, key,
            )
        )
    results = zpar.pmap(_cell_cache_frame_worker, tasks, workers)
    results.sort(key=lambda r: r["index"])
    meta = zfp._meta_from_hwc(patch_hw[0], patch_hw[1], 3)
    return zfp._write_manifest(
        cache_dir, meta, results, frame_ids,
        sum(r["rgb_bytes"] + r["support_bytes"] for r in results),
        patch_identity=patch_identity,
    )


# ---------------------------------------------------------------------------
# Frame-major decode-once -> per-Cell aligned disk cache (R6 format)
# ---------------------------------------------------------------------------

def _build_aligned_cache_frame_major(descs, canvas, cell_ctxs, cache_root, progress_callback):
    """Frame-major decode-once -> per-Cell aligned disk cache (R6 format).

    For each frame (sorted FrameId): decode ONCE (O(1 frame)), then for every
    Cell whose patch that frame touches, plan the source ROI, crop, reproject and
    APPEND to that Cell's cache. Memory O(1 frame); nothing held across frames.

    Cache REUSE: a Cell whose cache is already COMPLETE (R6 ``cache_is_complete``)
    is reused (no re-decode, no wipe); only missing/incomplete Cell caches are
    rebuilt. The wipe is scoped to the single Cell dir (never outside
    ``<cache_root>/<cell>``). Returns ``(cache_dirs, manifests, cache_report)``.
    """
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    builders = {}      # cell_id -> AlignedCacheBuilder (rebuild cells only)
    cache_dirs = {}    # cell_id -> Path
    patch_id_sets = {} # cell_id -> set of frame ids
    active = []        # (cell, patch, mem) for rebuild cells
    reused = []        # cell ids whose cache was reused
    rebuilt = []       # cell ids whose cache was (re)built
    for (_row, _col, cell, patch, mem) in cell_ctxs:
        if not mem.patch_ids:
            continue
        cid = cell.cell_id
        cache_dir = cache_root / cid
        cache_dirs[cid] = cache_dir
        patch_id_sets[cid] = set(mem.patch_ids)
        patch_identity = zfp.patch_geometry_identity(patch)
        if zfp.cache_is_complete(
            cache_dir, list(mem.patch_ids),
            expected_hw=patch.patch_shape_hw,
            expected_patch_identity=patch_identity,
        ):
            reused.append(cid)
            continue  # reuse: no wipe, no decode
        builders[cid] = zfp.AlignedCacheBuilder(
            cache_dir, patch_identity=patch_identity
        )  # wipes THIS cell dir only
        rebuilt.append(cid)
        active.append((cell, patch, mem))

    if reused:
        _emit(
            f"ZeGrid: cache reused for {len(reused)} cell(s): {', '.join(reused)}",
            lvl="INFO",
            callback=progress_callback,
        )
    if rebuilt:
        _emit(
            f"ZeGrid: cache (re)built for {len(rebuilt)} cell(s): {', '.join(rebuilt)}",
            lvl="INFO",
            callback=progress_callback,
        )

    # Precompute per-frame (cell, crop_plan) so the inner loop is cheap.
    frame_plans = {}
    for f in descs:
        key = f.frame_id.logical_path
        lst = []
        for (cell, patch, mem) in active:
            if key not in patch_id_sets[cell.cell_id]:
                continue
            try:
                plan = zg.plan_source_roi(f, canvas, patch)
            except Exception as exc:
                _emit(
                    f"source-ROI planning failed for {key} @ {cell.cell_id}: {exc}",
                    lvl="WARN",
                    callback=progress_callback,
                )
                plan = None
            if plan is not None:
                lst.append((cell, patch, plan))
        frame_plans[key] = lst

    total = len(descs)
    if active:
        for i, f in enumerate(sorted(descs, key=lambda d: d.frame_id), 1):
            key = f.frame_id.logical_path
            _emit(f"decode+cache: frame {i}/{total} ({key})", lvl="DEBUG", callback=progress_callback)
            hwc = _decode_frame_hwc(f, progress_callback)
            for (cell, patch, plan) in frame_plans[key]:
                sb = plan.source_bounds
                crop_hwc = hwc[sb.y0:sb.y1, sb.x0:sb.x1]
                crop_chw = np.ascontiguousarray(np.moveaxis(crop_hwc, -1, 0))
                cropped_wcs = zxe.slice_wcs(f.wcs(), sb)
                rgb, geom = zxe.reproject_cropped(
                    crop_chw, cropped_wcs, patch.patch_wcs(), patch.patch_shape_hw
                )
                builders[cell.cell_id].add(key, rgb, geom)
                del crop_hwc, crop_chw, rgb, geom
            del hwc

    manifests = {}
    for cid, b in builders.items():
        manifests[cid] = b.finish()
    for cid in reused:
        manifests[cid] = zfp.load_cache_manifest(cache_dirs[cid])
    return cache_dirs, manifests, {"reused": reused, "rebuilt": rebuilt}


# ---------------------------------------------------------------------------
# Per-cell execution (in-memory vs streaming, both bit-equal)
# ---------------------------------------------------------------------------

def _pick_mode(n_contributors, patch_area_px, patch_hw, available_bytes, tile_size=STREAM_TILE_SIZE):
    """Return ``("inmem" | "stream", selected_mode_bound_bytes)`` for one cell.

    The returned bound is the SELECTED mode's OWN bound (R4 in-memory bound for
    ``"inmem"``; R6 streaming bound for ``"stream"``), so the manifest reports the
    bound of the mode actually used (L2).
    """
    inmem_bound = zal.FITTED_MEMORY_MODEL.predict_bound_bytes(n_contributors, patch_area_px)
    if inmem_bound <= available_bytes * INMEM_AVAILABLE_FRAC:
        return "inmem", int(inmem_bound)
    stream_bound = _estimate_streaming_bytes(n_contributors, patch_hw, tile_size)
    return "stream", int(stream_bound)


# ---------------------------------------------------------------------------
# Mode-aware layout selection (R4 floors + in-memory OR streaming bound)
# ---------------------------------------------------------------------------

def _estimate_streaming_bytes(n_contributors, patch_hw, tile_size=STREAM_TILE_SIZE, channels=3):
    """Streaming per-cell peak bound in BYTES (R6 estimate, KiB -> bytes)."""
    return int(zstream.estimate_streaming_peak_kib(n_contributors, patch_hw, tile_size, channels) * 1024)


def _iter_cell_bounds_mode_aware(canvas, footprints, nx, ny, halo_px, tile_size):
    """Yield per-Cell dicts with BOTH in-memory and streaming peak bounds.

    Reuses the R4 membership primitive (:func:`auto_layout._cell_contributor_count`)
    and the R4/R6 published models verbatim — no science is duplicated.
    """
    layout = zg.build_layout(canvas, nx, ny)
    for row, col, bounds in layout.iter_cells(canvas):
        cid = zg.cell_id(row, col)
        cell = zg.ZeGridCell(cid, canvas.canvas_id, layout.layout_id, row, col, bounds)
        patch = zg.build_patch(canvas, cell, halo_px)
        n = zal._cell_contributor_count(footprints, canvas, patch.patch)
        area = patch.patch.width * patch.patch.height
        inmem_b = zal.FITTED_MEMORY_MODEL.predict_bound_bytes(n, area)
        stream_b = _estimate_streaming_bytes(n, patch.patch_shape_hw, tile_size)
        yield {
            "cell_id": cid, "row": row, "col": col, "n": n,
            "patch_area": area, "patch_hw": list(patch.patch_shape_hw),
            "inmem_bound_bytes": int(inmem_b),
            "stream_bound_bytes": int(stream_b),
            "cheaper_mode": "inmem" if inmem_b <= stream_b else "stream",
        }


def _scan_layout_mode_aware(canvas, footprints, nx, ny, halo_px, tile_size):
    """Worst-cell cheaper-mode bound for a candidate layout.

    For every Cell, takes the CHEAPER of the in-memory and streaming bounds and
    maximises over Cells (mirrors ``auto_layout._scan_layout`` but mode-aware).

    ZM-ZEGRID-R36 augments the returned worst-cell dict (in ONE membership pass)
    with the RAM-adaptive CPU-tile inputs consumed by
    :func:`_select_cpu_streaming_tile`:

    * ``inmem_worst_bytes`` — worst in-memory bound over the Cells;
    * ``stream_worst_by_tile`` — worst streaming bound for EACH ladder tile.

    The primary ``cheaper_bound_bytes`` (at the passed ``tile_size``) is
    UNCHANGED, so existing consumers (R28 parity, ``predicted_bound_bytes``)
    keep their exact semantics.
    """
    worst = None
    inmem_worst = 0
    stream_worst_by_tile = {int(t): 0 for t in CPU_STREAMING_TILE_LADDER}
    for c in _iter_cell_bounds_mode_aware(canvas, footprints, nx, ny, halo_px, tile_size):
        cheaper = min(c["inmem_bound_bytes"], c["stream_bound_bytes"])
        if worst is None or cheaper > worst["cheaper_bound_bytes"]:
            worst = dict(c)
            worst["cheaper_bound_bytes"] = cheaper
        if c["inmem_bound_bytes"] > inmem_worst:
            inmem_worst = int(c["inmem_bound_bytes"])
        hw = tuple(c["patch_hw"])
        n = c["n"]
        for t in CPU_STREAMING_TILE_LADDER:
            b = int(_estimate_streaming_bytes(n, hw, int(t)))
            if b > stream_worst_by_tile[int(t)]:
                stream_worst_by_tile[int(t)] = b
    worst = worst or {}
    worst["inmem_worst_bytes"] = int(inmem_worst)
    worst["stream_worst_by_tile"] = stream_worst_by_tile
    return worst


def _select_cpu_streaming_tile(stream_worst_by_tile, ram_budget):
    """Largest ladder tile whose worst-cell streaming bound fits ``ram_budget``.

    Returns ``(selected_tile, trials, reason)``:

    * ``selected_tile`` — the largest tile in :data:`CPU_STREAMING_TILE_LADDER`
      whose ``stream_worst_by_tile`` bound is ``<= ram_budget`` (or the largest
      tile at all when ``ram_budget is None``); ``None`` when NO ladder tile fits
      (then the layout is admissible only via the in-memory bound).
    * ``trials`` — ordered ``[{"tile": t, "worst_bound_bytes": b, "fits": f}]``
      (descending tile order) so the log/manifest can show why 128 dropped.
    * ``reason`` — a compact human string.

    ``stream_worst_by_tile`` is the dict produced by :func:`_scan_layout_mode_aware`.
    """
    trials = []
    selected = None
    for tile in CPU_STREAMING_TILE_LADDER:
        bound = int(stream_worst_by_tile.get(int(tile), 0) or 0)
        fits = (ram_budget is None) or (bound <= int(ram_budget))
        trials.append({"tile": int(tile), "worst_bound_bytes": bound, "fits": bool(fits)})
        if fits and selected is None:
            selected = int(tile)
    if selected is None:
        reason = (
            "no CPU streaming tile fits the budget "
            f"(minimum tile {CPU_STREAMING_TILE_MINIMUM} bound "
            f"{int(stream_worst_by_tile.get(CPU_STREAMING_TILE_MINIMUM, 0)) / 2**20:.1f} MiB"
            + (f" > budget {int(ram_budget) / 2**20:.1f} MiB" if ram_budget is not None else "")
            + "); streaming not viable — in-memory only"
        )
    elif selected == CPU_STREAMING_TILE_PREFERRED:
        reason = "preferred CPU streaming tile 128 fits the budget"
    else:
        # Explain why 128 dropped: list the coarser tiles that did not fit.
        dropped = [t["tile"] for t in trials if not t["fits"]]
        reason = (
            f"CPU streaming tile reduced from {CPU_STREAMING_TILE_PREFERRED} to "
            f"{selected} to fit the budget (tile(s) {dropped} exceed "
            + (f"{int(ram_budget) / 2**20:.1f} MiB" if ram_budget is not None else "budget")
            + ")"
        )
    return selected, trials, reason


def _cpu_streaming_block(selected_tile, trials, reason, stream_worst_by_tile, inmem_worst, ram_budget):
    """Build the public ``cpu_streaming`` provenance block (ZM-ZEGRID-R36).

    Carries the adaptive CPU streaming-tile decision WITHOUT a bare ``tile_size``
    name (the selected/preferred/minimum tiles are named explicitly). JSON-safe.
    """
    est_tile = int(selected_tile) if selected_tile is not None else int(CPU_STREAMING_TILE_MINIMUM)
    stream_bound = int(stream_worst_by_tile.get(int(est_tile), 0) or 0)
    return {
        "adaptive": True,
        "preferred_tile_size": int(CPU_STREAMING_TILE_PREFERRED),
        "selected_tile_size": (int(selected_tile) if selected_tile is not None else None),
        "minimum_tile_size": int(CPU_STREAMING_TILE_MINIMUM),
        "selection_reason": str(reason),
        "predicted_peak_bytes": int(stream_bound),
        "ram_budget_bytes": (int(ram_budget) if ram_budget is not None else None),
        "inmem_bound_bytes": int(inmem_worst),
        "stream_bound_bytes": int(stream_bound),
        # The legacy preferred-tile (128) streaming bound, kept SEPARATELY for
        # diagnostics/compat — it is NOT the plan's predicted bound (ZM-ZEGRID-R36 F3).
        "preferred_stream_bound_bytes": int(
            stream_worst_by_tile.get(CPU_STREAMING_TILE_PREFERRED, 0) or 0
        ),
        "estimate_tile_size": int(est_tile),
        "ladder": [int(t) for t in CPU_STREAMING_TILE_LADDER],
        "trials": list(trials),
    }


def _selected_admissible_bound(inmem_ok, inmem_worst, selected_tile, stream_worst_by_tile, legacy_fallback):
    """Bound the layout was ACTUALLY admitted with (ZM-ZEGRID-R36, F3).

    ``min`` over the ADMITTED modes' worst-cell bounds: the in-memory bound when
    it fits, and/or the selected streaming tile's bound when a tile fits. For an
    ACCEPTED layout at least one of them fits, so the result is ``<= ram_budget``.
    Falls back to ``legacy_fallback`` (the pre-R36 cheaper bound) only when
    neither is available (no-budget / degenerate).
    """
    bounds = []
    if inmem_ok:
        bounds.append(int(inmem_worst))
    if selected_tile is not None:
        bounds.append(int(stream_worst_by_tile.get(int(selected_tile), 0) or 0))
    return min(bounds) if bounds else int(legacy_fallback)


def _choose_layout_mode_aware(
    canvas,
    frames,
    ram_budget,
    *,
    available_bytes=None,
    tile_size=STREAM_TILE_SIZE,
    floors=None,
    halo_px=zsw.HALO_PX,
    pinned_layout=None,
    emit=None,
    log_line=None,
    observer=None,
    footprints_out=None,
    footprints_in=None,
    min_refinement_rank=None,
    max_refinement_rank=None,
):
    """MODE-AWARE RAM-aware layout: coarsest candidate whose cheaper mode fits.

    Reuses the R4 candidate enumeration (``REFINEMENT_FACTORS``, scientific
    floors, footprint membership) verbatim, but a candidate is FEASIBLE iff the
    CHEAPER-fitting per-cell mode's worst bound fits ``ram_budget``. Deterministic
    (identical enumeration order); returns a dict (layout + per-cell predictions).

    ``pinned_layout`` (``(nx, ny)`` or ``None``) BYPASSES the RAM-adaptive search
    and uses exactly that layout (scientific floors still enforced; an infeasible
    pinned layout raises). ``layout_source`` records ``"pinned"`` vs ``"auto"``.

    ``halo_px`` (ZM-ZEGRID-R53) is the patch extension used for candidate
    geometry, floors and per-cell contributor/area predictions. The product path
    passes the validated effective value explicitly; the default (``sweep.
    HALO_PX``) is only the historical low-level helper default for direct/unit
    calls, never a product default.

    ``emit`` (optional) is ``callable(message, lvl="INFO")``: when supplied, the
    scan is EXPLAINED live (ZM-ZEGRID-R14) — each candidate ``(nx, ny)``, why it
    was rejected (which floor / which bound), and the final chosen layout.

    ``log_line`` (optional, ZM-ZEGRID-R27) is ``callable(line)`` that appends one
    line to the incremental run log; candidate ACCEPTED/REJECTED and CHOSEN
    messages are routed here too (durably logged, not only the logger/GUI).

    ``observer`` (optional, ZM-ZEGRID-R27) is a :class:`SubstepReporter` that
    brackets the footprint-preparation and candidate-scan sub-steps with
    throttled progress counters + per-sub-step elapsed timings. ``None`` (default)
    keeps the exact previous behaviour (no observer, no progress callbacks).

    ``footprints_out`` (optional, ZM-ZEGRID-R28) is a mutable list that, when
    supplied, receives the run-local
    :class:`auto_layout.ProjectedFootprints` artifact (the exact
    once-projected polygons + median). It is an INTERNAL seam so the caller can
    build the final chosen-cell contexts from the SAME polygons without
    re-projecting WCS. The artifact is never placed in the returned public
    layout dict. ``None`` (default) keeps the exact previous behaviour.

    ZM-ZEGRID-R41-B (bounded replanning seams; ALL default-``None`` so every
    existing caller keeps the exact previous behaviour):

    * ``footprints_in`` — a previously projected footprint artifact to REUSE
      instead of re-projecting the WCS (projection is layout-independent, so a
      bounded replan reuses the once-projected polygons);
    * ``min_refinement_rank`` — skip candidates COARSER than this cell-count rank
      (a replan may never coarsen);
    * ``max_refinement_rank`` — skip candidates FINER than this cell-count rank
      (used to pin the exact geometry of an already-started run).

    A bounded refinement can therefore only move the chosen layout toward FINER
    candidates, in the existing deterministic ``REFINEMENT_FACTORS`` order with
    the scientific floors unchanged.
    """

    def _log(msg, lvl="INFO"):
        if emit is not None:
            try:
                emit(msg, lvl)
            except Exception:
                pass
        if log_line is not None:
            try:
                log_line(msg)
            except Exception:
                pass

    floors = floors or zal.ScientificFloors()
    n_frames = len(frames)
    if footprints_in is None and observer is not None:
        observer.start(
            zobs.LAYOUT_SUBSTEP_FOOTPRINTS,
            "WCS footprints / layout inputs",
            total=n_frames,  # ZM-ZEGRID-R28: ONE exact projection pass (was 2*N)
        )

    def _fp_progress():
        def cb(done, total, item_id):
            if observer is not None:
                observer.progress(int(done), item_id=item_id, total=n_frames)
        return cb

    # ZM-ZEGRID-R28: project each frame footprint EXACTLY ONCE and derive the
    # median width/height from those SAME polygons (no second projection pass).
    # ZM-ZEGRID-R41-B: a bounded replan REUSES the already-projected artifact
    # (``footprints_in``) — projection is layout-independent, so no second
    # expensive WCS pass is needed.
    if footprints_in is not None:
        footprints = footprints_in
    else:
        footprints = zal.project_footprints(frames, canvas, progress=_fp_progress())
    mw, mh = footprints.median_w, footprints.median_h
    if not (mw > 0 and mh > 0):
        if footprints_in is None and observer is not None:
            observer.end()
        raise zal.LayoutInfeasible("median projected footprint is degenerate")
    if footprints_in is None and observer is not None:
        observer.end()
    if footprints_out is not None:
        # Private artifact channel: the caller reuses these exact polygons for
        # the final chosen-cell memberships (never placed in the returned dict).
        footprints_out.append(footprints)

    if pinned_layout is not None:
        nx, ny = int(pinned_layout[0]), int(pinned_layout[1])
        if not (1 <= nx <= canvas.width and 1 <= ny <= canvas.height):
            raise ValueError(
                f"pinned layout {nx}x{ny} out of canvas bounds {canvas.width}x{canvas.height}"
            )
        geom = zal._nominal_geometry(canvas, nx, ny, halo_px)
        patch_ok = geom["min_patch_area"] >= floors.min_patch_area_px
        halo_ok = geom["halo_overhead"] <= floors.max_halo_overhead
        if not (patch_ok and halo_ok):
            msg = (
                f"pinned layout {nx}x{ny} violates scientific floors: "
                f"min_patch_area={geom['min_patch_area']} (limit {floors.min_patch_area_px}), "
                f"halo_overhead={geom['halo_overhead']:.4f} (limit {floors.max_halo_overhead}). "
                "Suggest increasing available RAM or pinning a FINER layout via the "
                "'zegrid_layout' config key."
            )
            _log(msg, "ERROR")
            raise zal.LayoutInfeasible(msg)
        cells = list(_iter_cell_bounds_mode_aware(canvas, footprints, nx, ny, halo_px, tile_size))
        max_n = max((c["n"] for c in cells), default=0)
        worst_bound = max((min(c["inmem_bound_bytes"], c["stream_bound_bytes"]) for c in cells), default=0)
        inmem_worst = max((c["inmem_bound_bytes"] for c in cells), default=0)
        stream_worst_by_tile = {int(t): 0 for t in CPU_STREAMING_TILE_LADDER}
        for c in cells:
            hw = tuple(c["patch_hw"])
            for t in CPU_STREAMING_TILE_LADDER:
                b = int(_estimate_streaming_bytes(c["n"], hw, int(t)))
                if b > stream_worst_by_tile[int(t)]:
                    stream_worst_by_tile[int(t)] = b
        # ZM-ZEGRID-R36: geometry stays EXACTLY the pinned (nx, ny); only the CPU
        # streaming tile adapts. Admissible iff the in-memory bound fits OR some
        # ladder tile's worst-cell streaming bound fits; fail loud otherwise.
        selected_tile, trials, sel_reason = _select_cpu_streaming_tile(stream_worst_by_tile, ram_budget)
        inmem_ok = (ram_budget is None) or (inmem_worst <= ram_budget)
        if not inmem_ok and selected_tile is None:
            min_bound = int(stream_worst_by_tile.get(CPU_STREAMING_TILE_MINIMUM, 0))
            msg = (
                f"pinned layout {nx}x{ny} cannot fit any mode within budget "
                f"{ram_budget / 2**20:.1f} MiB: in-memory worst-cell bound "
                f"{inmem_worst / 2**20:.1f} MiB and the finest streaming tile "
                f"(minimum tile {CPU_STREAMING_TILE_MINIMUM}) worst-cell bound "
                f"{min_bound / 2**20:.1f} MiB both exceed it. The next step would be "
                "a deeper disk-backed streaming executor (not available in R36). Suggest "
                "increasing available RAM or pinning a FINER layout via the "
                "'zegrid_layout' config key."
            )
            _log(msg, "ERROR")
            raise zal.LayoutInfeasible(msg)
        contrib_ok = max_n >= floors.min_contributors
        warnings = []
        if not contrib_ok:
            warnings.append(
                f"deepest cell has {max_n} contributors < min_contributors "
                f"({floors.min_contributors}); the layout may be scientifically degraded"
            )
        _log(
            f"layout scan: CHOSEN {nx}x{ny} ({nx * ny} cells, source=pinned) "
            f"CPU streaming tile {selected_tile} "
            f"(stream@{selected_tile}={int(stream_worst_by_tile.get(int(selected_tile), 0) if selected_tile is not None else CPU_STREAMING_TILE_MINIMUM) / 2**20:.1f} MiB); "
            f"in-memory worst {inmem_worst / 2**20:.1f} MiB; {sel_reason}"
        )
        _predicted_bound = _selected_admissible_bound(
            inmem_ok, inmem_worst, selected_tile, stream_worst_by_tile, worst_bound
        )
        return {
            "nx": nx, "ny": ny,
            "cell_count": nx * ny,
            "ram_budget_bytes": ram_budget,
            "available_bytes": available_bytes if available_bytes is not None else ram_budget,
            "max_contributors": max_n,
            "refinement_factor": None,
            "predicted_bound_bytes": int(_predicted_bound),
            "max_patch_area": max((c["patch_area"] for c in cells), default=0),
            "budget_bound_choice": False,
            "warnings": tuple(warnings),
            "cells": cells,
            "layout_source": "pinned",
            "cpu_streaming": _cpu_streaming_block(
                selected_tile, trials, sel_reason, stream_worst_by_tile, inmem_worst, ram_budget
            ),
            "floors": {
                "min_patch_area_px": {"value": geom["min_patch_area"],
                                       "limit": floors.min_patch_area_px, "ok": patch_ok},
                "max_halo_overhead": {"value": round(geom["halo_overhead"], 6),
                                       "limit": floors.max_halo_overhead, "ok": halo_ok},
                "min_contributors": {"value": max_n, "limit": floors.min_contributors, "ok": contrib_ok},
            },
        }

    if observer is not None:
        observer.start(
            zobs.LAYOUT_SUBSTEP_CANDIDATES,
            "candidate grid scan",
            total=None,  # candidate count is not known up-front; never fabricate
        )
    candidates = []
    seen = set()
    for factor in zal.REFINEMENT_FACTORS:
        nx = max(1, int(math.ceil(canvas.width / (mw / factor))))
        ny = max(1, int(math.ceil(canvas.height / (mh / factor))))
        nx = min(nx, canvas.width)
        ny = min(ny, canvas.height)
        if (nx, ny) in seen:
            continue
        seen.add((nx, ny))
        # ZM-ZEGRID-R41-B: bounded refinement rank filter. Candidates are
        # enumerated coarse -> fine, so candidates coarser than the floor are
        # SKIPPED and candidates finer than the ceiling BREAK the scan (all later
        # candidates are finer still). ``None`` keeps the exact previous scan.
        _rank = int(nx) * int(ny)
        if min_refinement_rank is not None and _rank < int(min_refinement_rank):
            continue
        if max_refinement_rank is not None and _rank > int(max_refinement_rank):
            break
        geom = zal._nominal_geometry(canvas, nx, ny, halo_px)
        if not (geom["min_patch_area"] >= floors.min_patch_area_px and
                geom["halo_overhead"] <= floors.max_halo_overhead):
            # Explain WHICH floor blocked this candidate (ZM-ZEGRID-R14).
            if geom["min_patch_area"] < floors.min_patch_area_px:
                _log(
                    f"layout scan: candidate {nx}x{ny} REJECTED — "
                    f"min_patch_area floor: {geom['min_patch_area']} px < "
                    f"{floors.min_patch_area_px} px"
                )
            if geom["halo_overhead"] > floors.max_halo_overhead:
                _log(
                    f"layout scan: candidate {nx}x{ny} REJECTED — "
                    f"max_halo_overhead floor: {geom['halo_overhead']:.4f} > "
                    f"{floors.max_halo_overhead}"
                )
            _log("layout scan: floors are monotonic in refinement — stopping")
            break  # floors are monotonic in refinement
        worst = _scan_layout_mode_aware(canvas, footprints, nx, ny, halo_px, tile_size)
        selected_tile, trials, sel_reason = _select_cpu_streaming_tile(
            worst["stream_worst_by_tile"], ram_budget
        )
        inmem_ok = (ram_budget is None) or (worst["inmem_worst_bytes"] <= ram_budget)
        # ZM-ZEGRID-R36 admissibility: the in-memory bound fits OR some (>= minimum)
        # CPU streaming tile fits. This strictly widens the pre-R36 rule
        # (min(inmem, stream@preferred) <= budget) whenever a smaller tile helps.
        memory_ok = inmem_ok or (selected_tile is not None)
        bound_mib = worst["cheaper_bound_bytes"] / 2**20
        inmem_mib = worst["inmem_worst_bytes"] / 2**20
        min_tile_mib = worst["stream_worst_by_tile"].get(CPU_STREAMING_TILE_MINIMUM, 0) / 2**20
        budget_mib = (ram_budget / 2**20) if ram_budget is not None else None
        if memory_ok:
            _log(
                f"layout scan: candidate {nx}x{ny} ACCEPTED "
                f"(in-memory worst {inmem_mib:.1f} MiB"
                f"{'; CPU streaming tile ' + str(selected_tile) + ' fits' if selected_tile is not None else ''}"
                + (f"; budget {budget_mib:.1f} MiB" if budget_mib is not None else "") + ")"
            )
        else:
            _log(
                f"layout scan: candidate {nx}x{ny} REJECTED — "
                f"cheaper-mode bound {bound_mib:.1f} MiB > budget {budget_mib:.1f} MiB "
                f"(in-memory worst {inmem_mib:.1f} MiB; no CPU streaming tile fits down "
                f"to minimum {CPU_STREAMING_TILE_MINIMUM}, bound {min_tile_mib:.1f} MiB)"
            )
        candidates.append({
            "factor": factor, "nx": nx, "ny": ny,
            "bound": worst["cheaper_bound_bytes"], "worst_cell": worst,
            "memory_ok": memory_ok, "geom": geom,
            "inmem_ok": inmem_ok, "selected_tile": selected_tile,
            "trials": trials, "selection_reason": sel_reason,
        })
        if memory_ok:
            # ZM-ZEGRID-R28: the returned choice is the FIRST floor-feasible +
            # memory-admissible candidate (see the chosen loop below). Stop the
            # production scan here rather than evaluating finer candidates that
            # are never used. Prior rejected candidates are already in ``candidates``
            # so ``budget_bound_choice`` and the no-fit error remain exact.
            break

    chosen = None
    chosen_index = None
    for i, cand in enumerate(candidates):
        if cand["memory_ok"]:
            chosen = cand
            chosen_index = i
            break

    if observer is not None:
        observer.end()

    if chosen is None and not candidates and (
        min_refinement_rank is not None or max_refinement_rank is not None
    ):
        # ZM-ZEGRID-R41-B: the bounded refinement-rank window admitted NO candidate
        # (all candidates were excluded by the coarse/fine rank bounds). Refuse
        # explicitly and point at the FINER direction only. This is the pinned /
        # exact-geometry refusal: no silent geometry change.
        msg = (
            "no layout candidate lies inside the requested refinement-rank window "
            f"(min={min_refinement_rank}, max={max_refinement_rank}, cell-count rank): "
            "the exact/pinned geometry cannot be re-planned under the current RAM "
            "budget. Refusing rather than changing the layout. A FINER layout "
            "(more cells) or more available RAM would be required."
        )
        _log(msg, "ERROR")
        raise zal.LayoutInfeasible(msg)

    if chosen is None:
        finest = candidates[-1] if candidates else None
        reasons = []
        if ram_budget is not None and finest is not None:
            wc = finest["worst_cell"]
            min_tile_bound = wc.get("stream_worst_by_tile", {}).get(CPU_STREAMING_TILE_MINIMUM, 0)
            reasons.append(
                f"even the finest floor-feasible layout ({finest['nx']}x{finest['ny']}) is "
                f"infeasible: in-memory worst-cell bound "
                f"{wc.get('inmem_worst_bytes', 0) / 2**20:.1f} MiB and the finest CPU "
                f"streaming tile (minimum tile {CPU_STREAMING_TILE_MINIMUM}) worst-cell "
                f"bound {min_tile_bound / 2**20:.1f} MiB both exceed budget "
                f"{ram_budget / 2**20:.1f} MiB (cheaper-mode bound "
                f"{finest['bound'] / 2**20:.1f} MiB)"
            )
        reasons.append(
            f"min_patch_area floor ({floors.min_patch_area_px} px) / "
            f"max_halo_overhead floor ({floors.max_halo_overhead}) cannot be satisfied "
            f"together with the budget (mode-aware, adaptive CPU streaming tile down to "
            f"{CPU_STREAMING_TILE_MINIMUM})"
        )
        msg = (
            "RAM budget cannot satisfy scientific floors (mode-aware): "
            + "; ".join(reasons)
            + ". A FINER layout (more cells) is the direction that lowers the "
            "per-cell footprint; a deeper disk-backed streaming executor is not "
            "available in R36. Suggest either increasing available RAM, choosing a "
            "finer layout, or pinning an explicit layout via the 'zegrid_layout' "
            "config key when reproducible geometry is required."
        )
        _log(msg, "ERROR")
        raise zal.LayoutInfeasible(msg)

    cells = list(_iter_cell_bounds_mode_aware(
        canvas, footprints, chosen["nx"], chosen["ny"], halo_px, tile_size
    ))
    max_n = max((c["n"] for c in cells), default=0)
    contrib_ok = max_n >= floors.min_contributors
    warnings = []
    if not contrib_ok:
        warnings.append(
            f"deepest cell has {max_n} contributors < min_contributors "
            f"({floors.min_contributors}); the layout may be scientifically degraded"
        )

    constrained = False
    if ram_budget is not None and chosen_index is not None:
        for cand in candidates[:chosen_index]:
            if not cand["memory_ok"]:
                constrained = True
                break

    _predicted_bound = _selected_admissible_bound(
        chosen["inmem_ok"], chosen["worst_cell"]["inmem_worst_bytes"],
        chosen["selected_tile"], chosen["worst_cell"]["stream_worst_by_tile"],
        chosen["bound"],
    )
    _sel_tile = chosen["selected_tile"]
    _sel_stream = int(
        chosen["worst_cell"]["stream_worst_by_tile"].get(
            int(_sel_tile) if _sel_tile is not None else CPU_STREAMING_TILE_MINIMUM, 0
        )
    )
    _log(
        f"layout scan: CHOSEN {chosen['nx']}x{chosen['ny']} "
        f"({chosen['nx'] * chosen['ny']} cells, mode-aware, source=auto) "
        f"predicted bound {_predicted_bound / 2**20:.1f} MiB; "
        f"in-memory worst {chosen['worst_cell'].get('inmem_worst_bytes', 0) / 2**20:.1f} MiB; "
        f"CPU streaming tile {_sel_tile} (stream@{_sel_tile}={_sel_stream / 2**20:.1f} MiB) "
        f"({chosen['selection_reason']})"
    )

    return {
        "nx": chosen["nx"], "ny": chosen["ny"],
        "cell_count": chosen["nx"] * chosen["ny"],
        "ram_budget_bytes": ram_budget,
        "available_bytes": available_bytes if available_bytes is not None else ram_budget,
        "max_contributors": max_n,
        "refinement_factor": chosen["factor"],
        "predicted_bound_bytes": int(_predicted_bound),
        "max_patch_area": chosen["worst_cell"]["patch_area"],
        "budget_bound_choice": constrained,
        "warnings": tuple(warnings),
        "cells": cells,
        "layout_source": "auto",
        "cpu_streaming": _cpu_streaming_block(
            chosen["selected_tile"], chosen["trials"], chosen["selection_reason"],
            chosen["worst_cell"]["stream_worst_by_tile"],
            chosen["worst_cell"]["inmem_worst_bytes"], ram_budget,
        ),
        "floors": {
            "min_patch_area_px": {"value": chosen["geom"]["min_patch_area"],
                                   "limit": floors.min_patch_area_px,
                                   "ok": chosen["geom"]["min_patch_area"] >= floors.min_patch_area_px},
            "max_halo_overhead": {"value": round(chosen["geom"]["halo_overhead"], 6),
                                   "limit": floors.max_halo_overhead,
                                   "ok": chosen["geom"]["halo_overhead"] <= floors.max_halo_overhead},
            "min_contributors": {"value": max_n, "limit": floors.min_contributors, "ok": contrib_ok},
        },
    }


def _build_cell_sres(result, frame_ids):
    """Build a MiniTileScienceResult from a canonical result + frame order."""
    order = list(frame_ids)
    ref_idx = int(result.provenance["reference"]["index"])
    reference_frame_id = order[ref_idx] if 0 <= ref_idx < len(order) else None
    excluded = tuple(
        (order[idx] if 0 <= idx < len(order) else f"<index {idx}>", stage, reason)
        for idx, stage, reason in result.provenance["excluded_frames"]
    )
    return zs.MiniTileScienceResult(
        result=result,
        reference_frame_id=reference_frame_id,
        excluded=excluded,
        frame_order=tuple(order),
    )


def _run_cell_inmem(cache_dir, patch, config, progress_callback, fixed=None, tile_size=None):
    """In-memory cell run: aligned frames resident, single-tile streaming executor.

    ZM-ZEGRID-R11: the in-memory path now goes through
    :func:`run_canonical_stack_streaming` with an :class:`InMemoryCanonicalProvider`
    (bit-equal to the engine by the R5/R6 contract) so it can consume the SAME
    fixed photometric gauge as the streaming path. ``fixed`` is the Cell-local
    :class:`FixedNormalization`; when None the per-Cell phase-1 is computed as
    before (legacy behaviour).

    ``tile_size`` (ZM-ZEGRID-R22): ``None`` keeps the legacy single-tile full-patch
    CPU behaviour; a bounded tile is passed when the cell runs on the GPU so the
    device never materialises ``N x full-cell`` (VRAM-bounded).
    """
    provider = zfp.MemmapCanonicalProvider(cache_dir)
    try:
        frame_ids = list(provider.frame_ids)
        images = []
        supports = []
        for i in range(provider.n_frames):
            rgb, sup = provider.get_raw_frame(i)
            images.append(np.array(rgb, dtype=np.float32, copy=True))
            supports.append(np.array(sup, dtype=bool, copy=True))
    finally:
        # R15 (rework-1 L1): release the memmap handles ALWAYS, even if the
        # read/materialise loop raises, so the caller can delete the cache
        # (Windows: an open .npy cannot be deleted -> WinError 32).
        provider.close()
    inmem_provider = InMemoryCanonicalProvider(images, supports)
    request = zstream.build_streaming_request(config, inmem_provider.n_frames)
    result = run_canonical_stack_streaming(
        inmem_provider, request, tile_size=tile_size, fixed=fixed
    )
    sres = _build_cell_sres(result, frame_ids)
    mt = za.extract_minitile(patch, sres)
    return mt, sres


def _run_cell_stream(cache_dir, patch, config, progress_callback, tile_size=STREAM_TILE_SIZE, fixed=None):
    provider = zfp.MemmapCanonicalProvider(cache_dir)
    try:
        request = zstream.build_streaming_request(config, provider.n_frames)
        result = run_canonical_stack_streaming(provider, request, tile_size=tile_size, fixed=fixed)
        sres = _build_cell_sres(result, list(provider.frame_ids))
        mt = za.extract_minitile(patch, sres)
        return mt, sres
    finally:
        # R15: release the memmap handles before the caller deletes the cache
        # (Windows: an open .npy cannot be deleted -> WinError 32).
        provider.close()


# ---------------------------------------------------------------------------
# ZM-ZEGRID-R20: concurrent per-cell stacking worker (process-pool safe)
# ---------------------------------------------------------------------------

# Module-level globals holding the per-cell worker's read-only inputs, handed to
# the pooled workers via an initializer (the R19 start-method-agnostic pattern:
# under fork the children inherit, under spawn they re-import and the initializer
# re-populates them, under threads they are shared). Read-only, never pickled
# per-task (descs/gauge can be large).
_STACK_DESCS = None
_STACK_CANVAS = None
_STACK_CONFIG = None
_STACK_GAUGE = None
_STACK_GLOBAL_FRAME_IDS = None
_STACK_GLOBAL_REF = None
_STACK_TILE_SIZE = STREAM_TILE_SIZE
_STACK_CACHE_WORKERS = 1
# ZM-ZEGRID-R40-B2-C: the worker-local R40 sparse transport is armed ONLY when the
# parent hands this mapping ({cell_id: candidate coords}) to the worker
# initializer. ``None`` (the default and the only value the capture-OFF path ever
# sets) means "transport absent": no extra IPC payload, no retention, no key on a
# worker result, no measurable behaviour change.
_STACK_R40_WORKER_COORDS = None


def _init_stack_worker(descs, canvas, science_config, global_gauge, global_frame_ids, global_reference_frame_id, tile_size=STREAM_TILE_SIZE, cache_workers=1, r40_worker_coords=None):
    """Child/parent initializer: publish the read-only per-cell stack inputs.

    ``r40_worker_coords`` (ZM-ZEGRID-R40-B2-C) is the OPTIONAL parent-computed
    per-cell candidate geometry for the R40 worker→parent sparse transport. It is
    appended to ``initargs`` by the caller ONLY when the capture is ON; with the
    capture OFF the initargs tuple is byte-identical to the pre-R40 transport and
    this parameter stays ``None`` (transport fully absent).
    """
    global _STACK_DESCS, _STACK_CANVAS, _STACK_CONFIG, _STACK_GAUGE
    global _STACK_GLOBAL_FRAME_IDS, _STACK_GLOBAL_REF, _STACK_TILE_SIZE
    global _STACK_CACHE_WORKERS, _STACK_R40_WORKER_COORDS
    _STACK_DESCS = descs
    _STACK_CANVAS = canvas
    _STACK_CONFIG = science_config
    _STACK_GAUGE = global_gauge
    _STACK_GLOBAL_FRAME_IDS = global_frame_ids
    _STACK_GLOBAL_REF = global_reference_frame_id
    _STACK_TILE_SIZE = tile_size
    _STACK_CACHE_WORKERS = cache_workers
    _STACK_R40_WORKER_COORDS = r40_worker_coords


def _resolve_cell_normalization(mode, global_gauge, global_frame_ids, provider, science_config):
    """Return ``(fixed, status, detail)`` for ONE Cell (ZM-CELL-PHOTOMETRY-R34B).

    ``fixed`` is the :class:`FixedNormalization` to apply (``None`` when the Cell
    has no usable gauge). ``status``/``detail`` describe the outcome for the
    record/manifest (``"ok"`` or ``"empty"``).

    * ``"global"`` (default): subset the R11 global gauge to the Cell's frames in
      the Cell's order — NO re-anchor. This is the unchanged production path; a
      Cell with no globally-active frame still raises ``cell gauge has no active
      frame`` from ``subset_fixed_normalization`` (pre-R34B behaviour).
    * ``"per_cell"`` (opt-in): compute the Cell's OWN self-anchored gauge from its
      own aligned frames via the FROZEN estimator (``photometric.compute_cell_gauge``).
      A zero-support Cell yields ``(None, "empty", detail)`` explicitly (never a
      fabricated gauge). This is the path that removes the Caldwell
      "no active frame" failure for geometrically-populated Cells.
    """
    resolved = zphot.resolve_cell_gauge_mode(mode)
    if resolved == zphot.CELL_GAUGE_MODE_PER_CELL:
        request = zstream.build_streaming_request(science_config, provider.n_frames)
        cres = zphot.compute_cell_gauge(provider, request)
        if cres.status != zphot.CELL_GAUGE_STATUS_OK:
            return None, cres.status, cres.detail
        return cres.gauge, cres.status, None
    fixed = subset_fixed_normalization(
        global_gauge, global_frame_ids, list(provider.frame_ids)
    )
    return fixed, zphot.CELL_GAUGE_STATUS_OK, None


# ---------------------------------------------------------------------------
# ZM-ZEGRID-R40-B2-B — default-OFF diagnostic seam capture (reuses r40_capture)
# ---------------------------------------------------------------------------
#
# The R40 capture is wired at the SAME pinned pre-offset/pre-feather seam as the
# accepted R37-D capture: inside ``_assemble_per_cell_feathered``, after the
# complete ``CellPatch`` objects coexist and BEFORE the inter-cell offset graph /
# the feather assembly are built. It REUSES the accepted, already-reviewed core
# ``core/zegrid/r40_capture`` (frozen dtype, fail-closed identity, hard disk
# budget, privacy, path safety, atomic COMMIT-last generation, additive manifest
# block) and duplicates NO pixel/difference maths.
#
# Hard invariants:
#   * strictly default OFF (strict ``bool``); the core is imported LAZILY and
#     ONLY when the capture is explicitly enabled — importing this module and
#     running with the capture OFF must not import the core nor change behaviour;
#   * diagnostic-only: nothing on this path mutates a CellPatch, the offset
#     graph, the additive offsets, the feather weights or any science / coverage
#     / aesthetic array;
#   * fail-closed: any identity / mandatory-observable / budget / privacy / path
#     failure refuses WITHOUT publishing a partial artifact and WITHOUT altering
#     the assembly result;
#   * sparse only: only the per-relation candidate pixels are read from the
#     in-memory planes — never a full f64 patch plane, never a raster/cutout,
#     never a final-product FITS (``mosaic_grid_science``/aesthetic/coverage).
#
# Explicit NON-CLAIMS (see the mission report): this seam wiring alone does NOT
# accept R40-B2. The OFF/ON FITS byte-identity gate, the exact §1.3 R37
# held-out-key inverse/parity intersection, the 38-relation panel qualification,
# the injected-effect recovery re-proof and the real product ROI parity remain
# NOT_RUN / B2_MANDATORY. R40-C / R40-D / HUMAN_GATE stay NOT_AUTHORIZED.
R40_CAPTURE_STAGE = "pre_offset_pre_feather"
R40_SEAM_CAPTURE_DISK_BUDGET_BYTES = 256 * 1024 * 1024
# Result-dict key carrying a worker's sparse covariate samples. The M5
# (``estimator_weight_sum``) and M7 (``support_w1`` / ``support_w2`` /
# ``n_eff_support``) planes live on the in-memory MiniTile INSIDE the worker and
# are never parent-visible (plan §2.2); the parent seam therefore takes them from
# this transport and REFUSES (fail-closed) rather than inventing them.
R40_SPARSE_RESULT_KEY = "r40_sparse"


_R40_CONTEXT_REQUIRED_KEYS = (
    "output_root",
    "disk_budget_bytes",
    "capture_id",
    "source_set_digest",
    "source_set_count",
    "n_frames",
    "repo_head",
    "gauge_mode",
    "stack_tokens",
    "canvas_id",
    "canvas_width",
    "canvas_height",
    "channels",
    "measurement_px",
    "feather_px",
    "halo_px",
    "block_px",
    "reference_frame_id",
    "frame_order_digest",
    "excluded",
)


def _r40_lazy_core():
    """Import the accepted R40 capture core LAZILY (never on the OFF path)."""
    from .core.zegrid import r40_capture as _rc

    return _rc


def _validate_r40_seam_capture(enabled, output_root, disk_budget_bytes):
    """Strict validation of the R40 seam-capture opt-in (no silent coercion).

    Returns the normalised ``(enabled, output_root, disk_budget_bytes)``.
    ``enabled`` must be a real ``bool`` (not 0/1/"true"); when it is True an
    explicit, non-empty output root is REQUIRED (a capture that cannot publish is
    refused up front). The budget must be a real positive ``int``.
    """
    if not isinstance(enabled, bool):
        raise ValueError(
            f"r40_seam_capture must be a bool, got {type(enabled).__name__}"
        )
    root = None
    if enabled:
        if not isinstance(output_root, (str, os.PathLike)):
            raise ValueError(
                "r40_seam_capture requires an explicit output root "
                f"(str/PathLike), got {type(output_root).__name__}"
            )
        root = str(output_root)
        if not root.strip():
            raise ValueError("r40_seam_capture output root must be non-empty")
    budget = (
        R40_SEAM_CAPTURE_DISK_BUDGET_BYTES
        if disk_budget_bytes is None
        else disk_budget_bytes
    )
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise ValueError(
            "r40_seam_capture_disk_budget_bytes must be an int, got "
            f"{type(budget).__name__}"
        )
    if int(budget) <= 0:
        raise ValueError(
            f"r40_seam_capture_disk_budget_bytes must be positive, got {int(budget)}"
        )
    return enabled, root, int(budget)


def _r40_cell_index(cid):
    """Parse a ZeGrid cell id ``r%04dc%04d`` into ``(row, col)`` (else ``None``)."""
    s = str(cid)
    if not s.startswith("r") or "c" not in s[1:]:
        return None
    rpart, _, cpart = s[1:].partition("c")
    if not rpart.isdigit() or not cpart.isdigit():
        return None
    return int(rpart), int(cpart)


def _r40_orientation(u, v):
    """Relation orientation (``h``/``v``) from two adjacent cell ids (else None)."""
    iu, iv = _r40_cell_index(u), _r40_cell_index(v)
    if iu is None or iv is None:
        return None
    return "h" if iu[0] == iv[0] else "v"


def _r40_overlap_band(patch_u, patch_v):
    """Global ``(y0, y1, x0, x1)`` intersection of two Cell patches."""
    uy0, uy1, ux0, ux1 = patch_u.bounds
    vy0, vy1, vx0, vx1 = patch_v.bounds
    return (max(uy0, vy0), min(uy1, vy1), max(ux0, vx0), min(ux1, vx1))


def _r40_candidate_coords(patches, u, v, *, block_px):
    """Deterministic block-parity candidate GLOBAL coords in the u/v overlap band.

    Reuses the frozen §1.3 block-parity rule (``bx % 2 == 1 AND by % 2 == 1``,
    ``bx = gx // block_px``, ``by = gy // block_px``) over the shared halo band,
    ordered deterministically by ``(gy, gx)``. This is the geometry SUPERSET the
    worker→parent transport is built from. It is NOT the capture's sample set:
    the parent intersects it with the EXACT §1.3 held-out selection recovered by
    :func:`_r40_exact_heldout_keys` (``observe_edge`` + the frozen integer/block
    inverse), fail-closed — see R40-B2-G, plan §15.13.
    """
    y0, y1, x0, x1 = _r40_overlap_band(patches[u], patches[v])
    if y1 <= y0 or x1 <= x0:
        return ()
    gy = np.arange(int(y0), int(y1), dtype=np.int64)
    gx = np.arange(int(x0), int(x1), dtype=np.int64)
    keep_y = ((gy // int(block_px)) % 2) == 1
    keep_x = ((gx // int(block_px)) % 2) == 1
    return tuple(
        (int(x), int(y)) for y in gy[keep_y] for x in gx[keep_x]
    )


def _r40_exact_heldout_keys(patches, u, v, channel, *, canvas_width, canvas_height,
                            nx, ny, block_px, measurement_px):
    """EXACT frozen-rule held-out keys for ONE relation/channel (fail-closed).

    The capture's sample set must BE the accepted R37 held-out sample set, not
    merely its candidate geometry superset. The exact set is therefore the
    accepted evidence itself: the FROZEN production observation
    (``background_model.observe_edge``) is evaluated on the two patches with the
    frozen measurement band and the frozen relation normal boundary (reused from
    ``background_model.relation_normal_boundary`` — never re-derived), and its
    normalized ``holdout_xy`` is recovered to EXACT global INTEGER keys through
    the frozen §1.3 integer/block inverse (round-half-up, ``INV_TOL``, parity
    verified) — never a raw floor of normalized floats.

    Returns ``None`` when the frozen observation itself refuses (that
    relation/channel carries no R37 held-out evidence, exactly as the accepted
    artifact drops it); any malformed/ambiguous recovery REFUSES.
    """
    from .core.zegrid import background_diagnostics as _bgd
    from .core.zegrid import background_model as zbm

    rc = _r40_lazy_core()
    canvas_hw = (int(canvas_height), int(canvas_width))
    boundary = zbm.relation_normal_boundary(u, v, int(nx), int(ny), canvas_hw)
    obs, _reason = zbm.observe_edge(
        _bgd._as_plan_cell(patches[u]), _bgd._as_plan_cell(patches[v]), canvas_hw,
        int(channel), block_px=int(block_px), measurement_px=int(measurement_px),
        normal_boundary=float(boundary),
    )
    if obs is None:
        return None
    return rc.heldout_keys_from_xy(
        obs.holdout_xy, canvas_width=int(canvas_width),
        canvas_height=int(canvas_height), block_px=int(block_px),
    )


def _r40_plane_at(plane, ly, lx, channel):
    """Read ONE pixel from a 2-D or 3-D plane (never a full plane)."""
    arr = np.asarray(plane)
    if arr.ndim == 3:
        return arr[ly, lx, channel]
    return arr[ly, lx]


def _r40_post_offset_value(science, ly, lx, channel, offsets):
    """M3 post-offset value via the FROZEN production helper (§3.4, f64 order)."""
    off = np.asarray(offsets, dtype=np.float64).ravel()
    pix = np.asarray(science)[ly:ly + 1, lx:lx + 1, channel:channel + 1]
    post = zcf.apply_additive_offsets(pix, off[channel:channel + 1])
    return float(np.asarray(post, dtype=np.float64)[0, 0, 0])


def _r40_worker_extract_covariates(task):
    """Module-level PICKLABLE worker-side sparse covariate extraction.

    ``task = (cell_id, coords, planes)`` where ``planes`` is a plain mapping of
    plane name -> array plus the scalar ``bounds``/``channel`` entries. Returns a
    plain, picklable list of sparse covariate samples (M5/M7 + M6 + M4) carrying
    ONLY the candidate coordinates — never a full f64 patch plane, never a raster,
    never a cutout, never a FITS read. This is the worker→parent transport used by
    the seam (via ``result_by_cid[cid]['r40_sparse']``) and by the spawn/pickling
    parity proof.
    """
    cell_id, coords, planes = task
    y0 = int(planes["bounds"][0])
    x0 = int(planes["bounds"][2])
    channel = int(planes["channel"])
    sci = np.asarray(planes["science"])
    val = np.asarray(planes["valid_mask"])
    surv = np.asarray(planes["surviving_sample_count"])
    ews = np.asarray(planes["estimator_weight_sum"])
    w1 = np.asarray(planes["support_w1"])
    w2 = np.asarray(planes["support_w2"])
    neff = np.asarray(planes["n_eff_support"])
    out = []
    for gx, gy in coords:
        ly, lx = int(gy) - y0, int(gx) - x0
        out.append({
            "cell_id": str(cell_id),
            "gx": int(gx),
            "gy": int(gy),
            "channel": channel,
            "sci_pre": float(_r40_plane_at(sci, ly, lx, channel)),
            "valid": bool(_r40_plane_at(val, ly, lx, channel)),
            "surviving_sample_count": int(_r40_plane_at(surv, ly, lx, channel)),
            "estimator_weight_sum": float(_r40_plane_at(ews, ly, lx, channel)),
            "support_w1": float(_r40_plane_at(w1, ly, lx, channel)),
            "support_w2": float(_r40_plane_at(w2, ly, lx, channel)),
            "n_eff_support": float(_r40_plane_at(neff, ly, lx, channel)),
        })
    return out


class _R40BoundsProxy:
    """Minimal ``.bounds`` carrier for the parent-side candidate geometry.

    The sealed per-relation candidate geometry (``_r40_candidate_coords``) reads
    only ``patch.bounds``; this proxy lets the PARENT reuse that exact geometry on
    the worker task bounds without duplicating any block/parity maths and without
    materialising a ``CellPatch`` (or any pixel plane).
    """

    __slots__ = ("bounds",)

    def __init__(self, bounds):
        self.bounds = tuple(int(v) for v in bounds)


def _r40_worker_candidate_coords(bounds_by_cell, nx, ny, *, block_px):
    """Parent-side per-cell candidate coords for the worker→parent transport.

    For every complete adjacent Cell relation of the FROZEN ``grid_adjacency``
    grid this returns, for EACH endpoint, the deterministic block-parity candidate
    coordinates of the relation's shared halo band — computed by the SAME sealed
    ``_r40_candidate_coords`` helper the seam uses (no duplicated geometry or
    pixel/difference maths). The per-cell union is deduplicated and deterministically
    ordered by ``(gy, gx)`` so the pooled worker and the in-process reference agree
    exactly.

    Size is ``O(candidate records)`` — never ``O(full patch area)`` and never
    ``N_frames x canvas``.
    """
    proxy = {
        str(cid): _R40BoundsProxy(bounds)
        for cid, bounds in (bounds_by_cell or {}).items()
    }
    per_cell: dict = {}
    for (u, v) in zcf.grid_adjacency(int(nx), int(ny)):
        if u not in proxy or v not in proxy:
            continue
        for (gx, gy) in _r40_candidate_coords(proxy, u, v, block_px=int(block_px)):
            per_cell.setdefault(str(u), {})[(int(gx), int(gy))] = None
            per_cell.setdefault(str(v), {})[(int(gx), int(gy))] = None
    return {
        str(cid): tuple(sorted(coords, key=lambda k: (k[1], k[0])))
        for cid, coords in per_cell.items()
    }


def _r40_worker_transport_sparse(cell_id, coords, planes):
    """Worker-side R40 sparse transport entry point (module-level, PICKLABLE).

    Called by the pooling worker :func:`_stack_cell` with the WORKER-LOCAL MiniTile
    planes; also the exact entry point exercised through a real ``spawn`` pool by
    the B2-C proof (e) test. Loops the channels and delegates every single-pixel
    read to the B2-B extractor :func:`_r40_worker_extract_covariates` (no duplicated
    maths). Returns a plain, picklable list of samples — ``O(len(coords) x
    channels)``, never a full f64 patch plane, never a raster/cutout, never a FITS
    read.
    """
    sci = np.asarray(planes["science"])
    n_channels = int(sci.shape[2]) if sci.ndim == 3 else 1
    out: list = []
    for channel in range(n_channels):
        planes_ch = dict(planes)
        planes_ch["channel"] = int(channel)
        out.extend(_r40_worker_extract_covariates((str(cell_id), tuple(coords or ()), planes_ch)))
    return out


def _r40_maybe_worker_sparse(cell_id, minitile, patch):
    """Worker-side R40 transport, armed ONLY when the parent supplied coords.

    Returns ``None`` when the capture is OFF: the caller then adds NO key and keeps
    NO payload, so the worker result stays byte-identical to the pre-R40 transport.
    When armed, builds the plane mapping from the in-memory MiniTile planes (never
    a FITS read, never a raster, never a full f64 patch plane) and returns the
    sparse per-candidate samples of THIS cell only.
    """
    coords_by_cell = _STACK_R40_WORKER_COORDS
    if coords_by_cell is None:
        return None
    coords = coords_by_cell.get(str(cell_id))
    if not coords:
        return ()
    planes = {
        "bounds": (patch.patch.y0, patch.patch.y1, patch.patch.x0, patch.patch.x1),
        "science": minitile.science,
        "valid_mask": minitile.valid_mask,
        "surviving_sample_count": minitile.surviving_sample_count,
        "estimator_weight_sum": minitile.estimator_weight_sum,
        "support_w1": minitile.support_w1,
        "support_w2": minitile.support_w2,
        "n_eff_support": minitile.n_eff_support,
    }
    return _r40_worker_transport_sparse(str(cell_id), coords, planes)


def _r40_index_worker_sparse(result_by_cid):
    """Index the worker→parent sparse covariate samples by (cid, relation, ch)."""
    frames: dict = {}
    for cid, result in (result_by_cid or {}).items():
        if not isinstance(result, dict):
            continue
        samples = result.get(R40_SPARSE_RESULT_KEY)
        if not samples:
            continue
        for sample in samples:
            key = (str(cid), int(sample["channel"]), int(sample["gx"]), int(sample["gy"]))
            frames[key] = sample
    return frames


def _r40_relation_header(*, u, v, orientation, gauge_mode, stack_tokens,
                         reference_frame_id, frame_order_digest, excluded,
                         measurement_px, feather_px, halo_px, block_px):
    """Build the frozen M9/M10 relation header (validated by the accepted core)."""
    return {
        "gauge_mode": str(gauge_mode),
        "stack_tokens": dict(stack_tokens),
        "reference_frame_id": {u: str(reference_frame_id[u]), v: str(reference_frame_id[v])},
        "frame_order_digest": {u: str(frame_order_digest[u]), v: str(frame_order_digest[v])},
        "excluded": {u: list(excluded[u]), v: list(excluded[v])},
        "measurement_px": int(measurement_px),
        "feather_px": int(feather_px),
        "halo_px": int(halo_px),
        "block_px": int(block_px),
    }


def _r40_seam_capture(patches, nx, ny, result_by_cid, *, context):
    """Build ONE bounded R40 diagnostic generation at the pinned seam (default OFF).

    ``context`` carries the run-level identity/tokens the seam cannot invent.
    Any missing mandatory observable, identity, budget, privacy or path failure
    REFUSES (raising) before anything is published; the assembly inputs are never
    mutated.
    """
    rc = _r40_lazy_core()
    ctx = dict(context or {})
    missing = [k for k in _R40_CONTEXT_REQUIRED_KEYS if k not in ctx]
    if missing:
        raise rc.SchemaError(
            "R40 seam capture context is missing required keys: " + ", ".join(sorted(missing))
        )
    channels = int(ctx["channels"])
    if channels != int(rc.N_CHANNELS):
        raise rc.SchemaError(
            f"R40 seam capture expects {rc.N_CHANNELS} channels, got {channels}"
        )
    block_px = int(ctx["block_px"])
    covariance = _r40_index_worker_sparse(result_by_cid)

    present = [cid for cid in patches]
    edges = [
        (u, v) for (u, v) in zcf.grid_adjacency(int(nx), int(ny))
        if u in patches and v in patches
    ]
    if not edges:
        raise rc.SchemaError("R40 seam capture found no complete adjacent Cell relation")

    # The frozen R34c additive offsets. They are supplied by the caller (the
    # assembly's own ``build_cell_offset_graph`` solve is left byte-for-byte
    # unchanged, so the capture cannot influence the assembly); no pixel /
    # difference maths is duplicated here.
    offsets = ctx.get("offsets")
    if offsets is None:
        raise rc.SchemaError(
            "R40 seam capture requires the frozen R34c offsets (context['offsets'])"
        )
    offsets_by_cell = {str(k): np.asarray(v, dtype=np.float64) for k, v in offsets.items()}

    shards = []
    for (u, v) in edges:
        orientation = _r40_orientation(u, v)
        if orientation is None:
            raise rc.SchemaError(f"R40 seam capture: cannot resolve orientation of {u!r}/{v!r}")
        relation_id = rc.relation_id_for(u, v)
        header = _r40_relation_header(
            u=u, v=v, orientation=orientation,
            gauge_mode=ctx["gauge_mode"], stack_tokens=ctx["stack_tokens"],
            reference_frame_id=ctx["reference_frame_id"],
            frame_order_digest=ctx["frame_order_digest"],
            excluded=ctx["excluded"],
            measurement_px=ctx["measurement_px"], feather_px=ctx["feather_px"],
            halo_px=ctx["halo_px"], block_px=block_px,
        )
        pu, pv = patches[u], patches[v]
        uy0, _, ux0, _ = pu.bounds
        vy0, _, vx0, _ = pv.bounds
        candidates = _r40_candidate_coords(patches, u, v, block_px=block_px)
        candidates_set = set(candidates)
        records = []
        for channel in range(channels):
            # EXACT held-out selection (frozen §1.3): the accepted R37 held-out
            # evidence itself, recovered through the frozen integer/block inverse —
            # never the raw candidate geometry superset and never a box/validity
            # approximation thereof.
            keys = _r40_exact_heldout_keys(
                patches, u, v, channel,
                canvas_width=int(ctx["canvas_width"]),
                canvas_height=int(ctx["canvas_height"]),
                nx=int(nx), ny=int(ny), block_px=block_px,
                measurement_px=int(ctx["measurement_px"]),
            )
            if keys is None:
                # The frozen observation refuses: no R37 held-out evidence for
                # this relation/channel (the accepted artifact drops it too).
                continue
            # Fail-closed: every EXACT key must lie in the candidate geometry the
            # workers transported; a missing candidate means an incomplete
            # transport and REFUSES the capture (never a silent drop).
            missing = [k for k in keys if k not in candidates_set]
            if missing:
                raise rc.SchemaError(
                    "R40 seam capture: exact held-out key(s) absent from the "
                    f"candidate geometry for {u!r}/{v!r} channel {channel}: "
                    f"{missing[:4]}"
                )
            if not keys:
                continue
            samples_u, samples_v = [], []
            for (gx, gy) in keys:
                lyu, lxu = int(gy) - uy0, int(gx) - ux0
                lyv, lxv = int(gy) - vy0, int(gx) - vx0
                cov_u = covariance.get((str(u), channel, int(gx), int(gy)))
                cov_v = covariance.get((str(v), channel, int(gx), int(gy)))
                if cov_u is None or cov_v is None:
                    raise rc.SchemaError(
                        "R40 seam capture: mandatory observable M5/M7 "
                        "(estimator_weight_sum / support_w1 / support_w2 / "
                        f"n_eff_support) unavailable for cell {u!r}/{v!r} "
                        f"channel {channel} at ({int(gx)},{int(gy)}); refusing "
                        "rather than fabricating it"
                    )
                ramp_u = float(zcf.feather_weight_field(
                    np.asarray(pu.science).shape[:2], pu.core_slice, int(ctx["halo_px"]),
                    bounds=pu.bounds, canvas_hw=(int(ctx["canvas_height"]),
                                                 int(ctx["canvas_width"])),
                )[lyu, lxu])
                ramp_v = float(zcf.feather_weight_field(
                    np.asarray(pv.science).shape[:2], pv.core_slice, int(ctx["halo_px"]),
                    bounds=pv.bounds, canvas_hw=(int(ctx["canvas_height"]),
                                                 int(ctx["canvas_width"])),
                )[lyv, lxv])
                samples_u.append({
                    "gx": int(gx), "gy": int(gy),
                    "sci_u_pre": float(cov_u["sci_pre"]),
                    "sci_u_post": _r40_post_offset_value(
                        pu.science, lyu, lxu, channel, offsets_by_cell[str(u)]),
                    "valid_u": bool(cov_u["valid"]),
                    "estimator_weight_sum_u": float(cov_u["estimator_weight_sum"]),
                    "surviving_sample_count_u": int(cov_u["surviving_sample_count"]),
                    "support_w1_u": float(cov_u["support_w1"]),
                    "support_w2_u": float(cov_u["support_w2"]),
                    "n_eff_support_u": float(cov_u["n_eff_support"]),
                    "ramp_u": ramp_u,
                })
                samples_v.append({
                    "gx": int(gx), "gy": int(gy),
                    "sci_v_pre": float(cov_v["sci_pre"]),
                    "sci_v_post": _r40_post_offset_value(
                        pv.science, lyv, lxv, channel, offsets_by_cell[str(v)]),
                    "valid_v": bool(cov_v["valid"]),
                    "estimator_weight_sum_v": float(cov_v["estimator_weight_sum"]),
                    "surviving_sample_count_v": int(cov_v["surviving_sample_count"]),
                    "support_w1_v": float(cov_v["support_w1"]),
                    "support_w2_v": float(cov_v["support_w2"]),
                    "n_eff_support_v": float(cov_v["n_eff_support"]),
                    "ramp_v": ramp_v,
                })
            pay_u = rc.build_sparse_payload(
                cell_id=str(u), role="u", relation_id=relation_id,
                channel=channel, samples=samples_u,
            )
            pay_v = rc.build_sparse_payload(
                cell_id=str(v), role="v", relation_id=relation_id,
                channel=channel, samples=samples_v,
            )
            records.extend(rc.parent_join_records(pay_u, pay_v, keys=keys))
        if not records:
            continue
        shards.append(rc.RelationShard(
            u=str(u), v=str(v), orientation=orientation, header=header,
            records=tuple(records),
            candidate_coords=tuple((int(gx), int(gy)) for (gx, gy) in candidates),
        ))

    if not shards:
        raise rc.SchemaError("R40 seam capture produced no relation shard (no exact held-out samples)")

    # ZM-ZEGRID-R40-B2-I (Nono B2-G LOW a + c). A production-wired seam generation
    # MUST declare the frozen candidate geometry for EVERY shard (a future seam
    # regression must not silently fall back to the weaker, no-declared-geometry
    # verification level), and the declared geometry MUST re-derive the frozen
    # candidate-set digest from the SEALED ``_r40_candidate_coords`` set — so the
    # digest is re-derived on the write side, never merely stored.
    for shard in shards:
        if shard.candidate_coords is None:
            raise rc.SchemaError(
                "R40 seam capture: a production-wired generation must declare the "
                f"candidate geometry for every shard ({shard.relation_id})"
            )
        sealed = tuple(
            (int(gx), int(gy))
            for (gx, gy) in _r40_candidate_coords(
                patches, str(shard.u), str(shard.v), block_px=block_px
            )
        )
        declared = tuple((int(gx), int(gy)) for gx, gy in shard.candidate_coords)
        if rc.candidate_set_digest(declared) != rc.candidate_set_digest(sealed):
            raise rc.SchemaError(
                "R40 seam capture: declared candidate geometry does not re-derive the "
                f"sealed frozen geometry for {shard.relation_id}"
            )

    identity = rc.make_capture_identity(
        capture_id=str(ctx["capture_id"]),
        source_set_digest=str(ctx["source_set_digest"]),
        source_set_count=int(ctx["source_set_count"]),
        n_frames=int(ctx["n_frames"]),
        layout={"nx": int(nx), "ny": int(ny)},
        canvas={
            "width": int(ctx["canvas_width"]),
            "height": int(ctx["canvas_height"]),
            "id": str(ctx["canvas_id"]),
        },
        gauge_mode=str(ctx["gauge_mode"]),
        measurement_px=int(ctx["measurement_px"]),
        feather_px=int(ctx["feather_px"]),
        halo_px=int(ctx["halo_px"]),
        block_px=block_px,
        repo_head=str(ctx["repo_head"]),
        production_wired=True,
    )
    config = rc.R40CaptureConfig(
        enabled=True,
        output_root=str(ctx["output_root"]),
        disk_budget_bytes=int(ctx["disk_budget_bytes"]),
        protected_roots=tuple(ctx.get("protected_roots") or ()),
    )
    written = rc.write_capture_generation(
        ctx["output_root"], shards=shards, identity=identity, config=config,
        protected_roots=tuple(ctx.get("protected_roots") or ()),
        expected_production_wired=True,
    )
    # §8.3 cross-link block: exactly the frozen identity keys. ``filename`` is the
    # generation directory basename, ``sha256``/``bytes`` are the strict-JSON index
    # that describes (and binds) every shard.
    index_path = None
    gen_dir = Path(str(ctx["output_root"])) / str(written["generation_dir"])
    index_path = gen_dir / str(written["index_filename"])
    return {
        "capture_id": str(identity["capture_id"]),
        "filename": str(written["generation_dir"]),
        "sha256": str(written["index_sha256"]),
        "bytes": int(index_path.stat().st_size) if index_path.exists() else 0,
        "index_filename": str(written["index_filename"]),
        "commit_sha256": str(written["commit_sha256"]),
        "n_shards": int(written["n_shards"]),
        "n_records": int(written["n_records"]),
        "shard_bytes": int(written["shard_bytes"]),
        "total_bytes": int(written["total_bytes"]),
        "disk_budget_bytes": int(written["disk_budget_bytes"]),
        "capture_stage": R40_CAPTURE_STAGE,
    }


def _r40_repo_head(start=None):
    """Best-effort 40-hex git HEAD of this checkout (else ``None``; never invented)."""
    try:
        root = Path(start or __file__).resolve()
    except OSError:  # pragma: no cover - defensive
        return None
    for base in [root, *root.parents]:
        head = base / ".git" / "HEAD"
        if not head.is_file():
            continue
        try:
            text = head.read_text(encoding="utf-8").strip()
        except OSError:  # pragma: no cover - defensive
            return None
        sha = text
        if text.startswith("ref:"):
            ref_name = text.split(":", 1)[1].strip()
            ref = base / ".git" / ref_name
            if ref.is_file():
                try:
                    sha = ref.read_text(encoding="utf-8").strip()
                except OSError:  # pragma: no cover - defensive
                    return None
            else:
                packed = base / ".git" / "packed-refs"
                sha = None
                if packed.is_file():
                    try:
                        for line in packed.read_text(encoding="utf-8").splitlines():
                            if not line or line.startswith(("#", "^")):
                                continue
                            parts = line.split()
                            if len(parts) == 2 and parts[1] == ref_name:
                                sha = parts[0]
                                break
                    except OSError:  # pragma: no cover - defensive
                        sha = None
        if isinstance(sha, str) and len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
            return sha
    return None


def _r40_capture_id(*, start_ts, pass_label, n_frames, source_set_digest_value, nonce):
    """Stable opaque R40 capture identity (deterministic hash; nonce = provenance only)."""
    h = hashlib.sha256()
    for part in (
        "zemosaic.zegrid.r40.seam.capture.v1",
        str(start_ts), str(pass_label), str(int(n_frames)),
        str(source_set_digest_value), str(nonce),
    ):
        h.update(part.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _r40_build_seam_context(*, output_root, disk_budget_bytes, capture_id, descs,
                            science_config, canvas, nx, ny, result_by_cid, halo_px,
                            measurement_px, feather_px, block_px, channels,
                            protected_roots=(), repo_head=None, worker_sparse=None):
    """Build the run-level R40 seam-capture context (caller owned; nothing invented).

    Raises (fail-closed) when a mandatory identity/provenance input is missing.
    Only called on the explicitly-enabled path.
    """
    rc = _r40_lazy_core()
    head = repo_head or _r40_repo_head()
    if not head:
        raise rc.IdentityError(
            "R40 seam capture: cannot resolve a 40-hex repo HEAD; refusing to publish"
        )
    logical = [str(d.frame_id.logical_path) for d in (descs or [])]
    # Source-set digest: the SAME frozen algorithm as the accepted R37-D capture
    # (reused, never re-defined), so the two captures bind the identical input set.
    from .core.zegrid import background_diagnostics as _bgd

    src_digest = _bgd.source_set_digest(logical)
    frame_order_digest_value = hashlib.sha256(
        ("zemosaic.zegrid.r40.frame_order.v1\n" + "\n".join(logical)).encode("utf-8")
    ).hexdigest()
    reference_frame_id, excluded = {}, {}
    for cid, result in (result_by_cid or {}).items():
        record = (result or {}).get("record") or {}
        ref = record.get("reference_frame_id")
        if not ref:
            raise rc.IdentityError(
                f"R40 seam capture: cell {cid!r} carries no reference_frame_id (M9); refusing"
            )
        reference_frame_id[str(cid)] = str(ref)
        excluded[str(cid)] = [
            ":".join(str(x) for x in (item if isinstance(item, (list, tuple)) else [item]))
            for item in (record.get("excluded") or [])
        ]
    rejection = str(science_config.rejection)
    tokens = {
        "normalization": str(science_config.normalization),
        "weighting": str(science_config.weighting),
        "rejection": rejection,
        "combine": str(science_config.combine),
        "taper": str(science_config.taper),
        "taper_px": float(science_config.taper_px),
        "taper_floor": float(science_config.taper_floor),
        "equalize_rgb": bool(science_config.equalize_rgb),
        "sigma_low": float(science_config.sigma_low),
        "sigma_high": float(science_config.sigma_high),
        "winsor": (
            float(max(science_config.winsor_limit_low, science_config.winsor_limit_high))
            if "winsor" in rejection.lower() else None
        ),
    }
    return {
        "output_root": str(output_root),
        "disk_budget_bytes": int(disk_budget_bytes),
        "protected_roots": tuple(protected_roots or ()),
        "capture_id": str(capture_id),
        "source_set_digest": src_digest,
        "source_set_count": int(len(logical)),
        "n_frames": int(len(logical)),
        "repo_head": head,
        "gauge_mode": str(getattr(science_config, "cell_gauge_mode", None)),
        "stack_tokens": tokens,
        "canvas_id": str(canvas.canvas_id),
        "canvas_width": int(canvas.width),
        "canvas_height": int(canvas.height),
        "channels": int(channels),
        "measurement_px": int(measurement_px),
        "feather_px": int(feather_px),
        "halo_px": int(halo_px),
        "block_px": int(block_px),
        "reference_frame_id": reference_frame_id,
        "frame_order_digest": {str(c): frame_order_digest_value for c in reference_frame_id},
        "excluded": excluded,
        "worker_sparse_cells": tuple(sorted((worker_sparse or {}).keys())),
    }


def _validate_r37_diagnostic_capture(capture, holdout_cap):
    """Strict validation of the R37-D capture opt-in (no silent coercion).

    Returns the normalised ``(capture, holdout_cap)``. ``capture`` must be a real
    ``bool`` (not 0/1/"true"); ``holdout_cap`` must be a real ``int`` (not a bool)
    and ``>= 4`` WHEN capture is enabled (the accepted artifact refuses a smaller
    cap). A disabled request never activates or touches the diagnostic module.
    """
    if not isinstance(capture, bool):
        raise ValueError(
            f"r37_diagnostic_capture must be a bool, got {type(capture).__name__}"
        )
    if isinstance(holdout_cap, bool) or not isinstance(holdout_cap, (int, np.integer)):
        raise ValueError(
            f"r37_diagnostic_holdout_cap must be an int, got {type(holdout_cap).__name__}"
        )
    cap = int(holdout_cap)
    if capture and cap < 4:
        raise ValueError(
            f"r37_diagnostic_holdout_cap must be >= 4 when capture is enabled, got {cap}"
        )
    return capture, cap


def _assemble_per_cell_feathered(canvas, nx, ny, result_by_cid, *, halo_px=None, emit=None,
                                 capture=False, holdout_cap=None, output_dir=None,
                                 capture_identity=None,
                                 r40_capture=False, r40_context=None):
    """R34D: apply inter-cell additive offsets + FEATHERED assembly (opt-in).

    Builds :class:`cell_feather.CellPatch` objects from the per-cell worker results
    (patch planes), solves the inter-cell additive offset graph (R34c, central
    anchor, support-count weights, pinned per-cell references), applies the
    per-channel additive offsets (additive-only), and assembles the canvas with a
    weighted feather blend (weights sum to 1).

    Disconnected policy (Nono I3): ``on_disconnected="zero"`` — Cells unreachable
    from the anchor keep offset 0 (their own self-anchored level, recorded); NO
    local re-anchor, NO fabricated bridge. A WARN lists them.

    ``halo_px`` (ZM-ZEGRID-R53) is the EFFECTIVE feather width (and drives the
    diagnostic measurement band ``2H``). The product path always passes it; the
    ``None`` fallback is only the historical low-level helper default for direct
    calls.
    """
    halo_px = int(zsw.HALO_PX if halo_px is None else halo_px)
    patches: dict = {}
    for cid, r in result_by_cid.items():
        if r.get("status") != "complete":
            continue
        p = r.get("patch")
        if not p:
            continue
        patches[cid] = zcf.CellPatch(
            cell_id=cid,
            bounds=tuple(int(v) for v in p["bounds"]),
            core_slice=tuple(int(v) for v in p["core_slice"]),
            science=p["science"],
            valid_mask=p["valid_mask"],
            surviving=p.get("surviving"),
        )
    if not patches:
        raise RuntimeError(
            "per_cell feather assembly: no complete cell patches were produced"
        )

    # ------------------------------------------------------------------
    # ZM-ZEGRID-R37-D: explicit, default-OFF diagnostic capture at the EXACT
    # pre-offset/pre-feather point. The complete CellPatch objects coexist here but
    # the inter-cell offset graph, the additive offsets and the feather weights do
    # not exist yet. The payload is the ACCEPTED R37-C strict-JSON artifact built
    # by the UNCHANGED ``background_diagnostics.build_payload`` (no duplicate
    # pixel/difference math) and published atomically into THIS run's actual
    # output directory. Diagnostic-only: the patches, graph, offsets, feather
    # weights and every science/aesthetic/coverage array are left untouched.
    # ------------------------------------------------------------------
    capture_info = None
    if capture:
        from .core.zegrid import background_diagnostics as _bgd

        if output_dir is None:
            raise RuntimeError(
                "R37 diagnostic capture requested but no output directory was "
                "provided; refusing to run without publishing the artifact"
            )
        cap = R37_CAPTURE_HOLDOUT_CAP_DEFAULT if holdout_cap is None else int(holdout_cap)
        # Measurement + feather widths explicit and consistent with the real halo
        # overlap (normally halo=8 -> measurement band 2*halo=16, feather=halo=8;
        # a halo-less patch falls back to the accepted default band). Recorded in
        # the provenance + manifest (L1).
        halo_i = int(halo_px)
        measurement_px = int(2 * halo_i) if halo_i > 0 else int(_bgd.DEFAULT_MEASUREMENT_PX)
        feather_px = halo_i if halo_i > 0 else int(_bgd.DEFAULT_FEATHER_PX)
        payload = _bgd.build_payload(
            patches, nx, ny, (int(canvas.height), int(canvas.width)),
            measurement_px=measurement_px, feather_px=feather_px, holdout_cap=cap,
        )
        # M1: REPLACE the pure/offline default provenance with truthful,
        # single-valued production-capture provenance (never "production_wired"
        # both false and true). Identity is generated once in ``_run_single``; a
        # direct/pure call without identity falls back to a deterministic
        # content-derived id (no hidden RNG).
        ident = dict(capture_identity or {})
        if not ident.get("capture_id"):
            ident = {
                "capture_id": _bgd.capture_run_id(
                    start_ts="direct", pass_label="direct", n_frames=int(nx * ny),
                    source_set_digest_value=_bgd.source_set_digest(sorted(patches)),
                ),
                "run_label": "direct",
                "n_source_frames": int(nx * ny),
                "source_set_digest": _bgd.source_set_digest(sorted(patches)),
                "source_set_digest_algo": _bgd.SOURCE_SET_DIGEST_ALGO,
            }
        payload["provenance"] = _bgd.capture_provenance(
            capture_id=ident.get("capture_id"),
            run_label=ident.get("run_label"),
            n_source_frames=ident.get("n_source_frames", 0),
            source_set_digest_value=ident.get("source_set_digest"),
            source_set_digest_algo=ident.get("source_set_digest_algo", _bgd.SOURCE_SET_DIGEST_ALGO),
            halo_px=halo_i, measurement_px=measurement_px, feather_px=feather_px,
            effective_cell_gauge_mode=zphot.CELL_GAUGE_MODE_PER_CELL,
        )
        capture_info = _bgd.publish_artifact(output_dir, payload)
        capture_info.update({
            "capture_id": str(ident.get("capture_id")),
            "run_label": str(ident.get("run_label")),
            "n_source_frames": int(ident.get("n_source_frames", 0)),
            "source_set_digest": str(ident.get("source_set_digest")),
            "source_set_digest_algo": str(
                ident.get("source_set_digest_algo", _bgd.SOURCE_SET_DIGEST_ALGO)
            ),
            "effective_cell_gauge_mode": zphot.CELL_GAUGE_MODE_PER_CELL,
            "halo_px": halo_i,
            "measurement_px": int(measurement_px),
            "feather_px": int(feather_px),
        })
        if emit is not None:
            try:
                emit(
                    "ZeGrid R37-D diagnostic capture: "
                    f"capture_id={capture_info['capture_id']} "
                    f"file={capture_info['filename']} stage={capture_info['capture_stage']} "
                    f"schema={capture_info['schema']}/{capture_info['schema_version']} "
                    f"relations={capture_info['n_relations']} dropped={capture_info['n_dropped']} "
                    f"capped={capture_info['holdout_capped']} "
                    f"qualification={capture_info['qualification']} "
                    f"bytes={capture_info['bytes']} sha256={capture_info['sha256']}",
                    "INFO",
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # ZM-ZEGRID-R40-B2-B: explicit, default-OFF R40 diagnostic capture at the
    # SAME pinned pre-offset/pre-feather point. The complete CellPatch objects
    # coexist here; the inter-cell offset graph, the additive offsets applied to
    # the patches and the feather blend do not exist yet. The capture reuses the
    # accepted default-OFF ``core/zegrid/r40_capture`` shard core (NO duplicated
    # pixel/difference maths) and is diagnostic-only: the patches, graph,
    # offsets, feather weights and every science/aesthetic/coverage array are
    # left untouched. Fail-closed: a refusal publishes NOTHING and leaves the
    # assembly result byte-identical.
    # ------------------------------------------------------------------
    r40_info = None
    if r40_capture:
        if r40_context is None:
            raise RuntimeError(
                "R40 diagnostic seam capture requested but no capture context was "
                "provided; refusing to run without a publishable identity"
            )
        _r40_ctx = dict(r40_context)
        _r40_ctx.setdefault("canvas_width", int(canvas.width))
        _r40_ctx.setdefault("canvas_height", int(canvas.height))
        _r40_ctx.setdefault("channels", 3)
        if _r40_ctx.get("offsets") is None:
            # The frozen R34c offsets are obtained from the SAME production solver
            # the assembly uses below (pure, read-only, deterministic). The
            # assembly's own solve is left byte-for-byte unchanged, so the capture
            # cannot influence it.
            _r40_graph = zcf.build_cell_offset_graph(
                canvas, nx, ny, patches,
                on_disconnected=zcf.FEATHER_DISCONNECTED_POLICY,
            )
            _r40_ctx["offsets"] = {
                cid: np.asarray(_r40_graph.offsets[i], dtype=np.float64)
                for i, cid in enumerate(_r40_graph.cell_ids)
            }
        r40_info = _r40_seam_capture(patches, nx, ny, result_by_cid, context=_r40_ctx)
        if emit is not None:
            try:
                emit(
                    "ZeGrid R40-B2-B diagnostic seam capture: "
                    f"capture_id={r40_info['capture_id']} "
                    f"generation={r40_info['filename']} "
                    f"shards={r40_info['n_shards']} records={r40_info['n_records']} "
                    f"shard_bytes={r40_info['shard_bytes']} "
                    f"total_bytes={r40_info['total_bytes']} "
                    f"budget={r40_info['disk_budget_bytes']} "
                    f"index_sha256={r40_info['sha256']}",
                    "INFO",
                )
            except Exception:
                pass

    graph = zcf.build_cell_offset_graph(
        canvas, nx, ny, patches, on_disconnected=zcf.FEATHER_DISCONNECTED_POLICY
    )
    if graph.disconnected and emit is not None:
        try:
            emit(
                "ZeGrid: per_cell offset graph — "
                f"{len(graph.disconnected)} disconnected cell(s) keep offset 0 "
                f"(policy={graph.policy}): {list(graph.disconnected[:8])}"
                f"{'...' if len(graph.disconnected) > 8 else ''}",
                "WARN",
            )
        except Exception:
            pass
    offsets = {cid: graph.offsets[i] for i, cid in enumerate(graph.cell_ids)}
    if emit is not None:
        try:
            emit(
                f"ZeGrid: per_cell offsets solved — anchor={graph.anchor} "
                f"edges={graph.edges_used}/{graph.edges_built} "
                f"cells={len(graph.cell_ids)} channels={graph.channels} "
                f"resid_rms={[round(float(x), 6) for x in graph.weighted_residual_rms]}",
                "INFO",
            )
        except Exception:
            pass
    assembled = zcf.assemble_canvas_feathered(
        canvas, nx, ny, patches, offsets, halo_px=halo_px
    )
    info = {
        "anchor": graph.anchor,
        "cell_ids": list(graph.cell_ids),
        "channels": int(graph.channels),
        "edges_used": int(graph.edges_used),
        "edges_built": int(graph.edges_built),
        "weighted_residual_rms": [float(x) for x in graph.weighted_residual_rms],
        "disconnected": list(graph.disconnected),
        "policy": graph.policy,
        "offsets": {cid: [float(v) for v in offsets[cid]] for cid in offsets},
        "halo_px": halo_px,
    }
    if capture_info is not None:
        info["r37_diagnostic_capture"] = capture_info
    if r40_info is not None:
        info["r40_diagnostic_capture"] = r40_info
    return assembled, info


def _assemble_cells(canvas, nx, ny, mode, cores, result_by_cid, *, halo_px=None, emit=None,
                    capture=False, holdout_cap=None, output_dir=None, capture_identity=None,
                    r40_capture=False, r40_context=None):
    """Dispatch assembly by cell gauge mode (ZM-CELL-PHOTOMETRY-R34D).

    Returns ``(assembled, feather_info)``. ``"global"`` (default) -> the
    UNCHANGED one-owner :func:`mosaic.assemble_canvas` (``feather_info`` is None);
    ``"per_cell"`` (opt-in) -> inter-cell additive offsets + feathered assembly.
    ZM-ZEGRID-R37-D: the diagnostic-capture opt-in is forwarded ONLY to the
    per_cell path (the only point where pre-feather CellPatches exist); the
    global path never captures.
    """
    resolved = zphot.resolve_cell_gauge_mode(mode)
    if resolved == zphot.CELL_GAUGE_MODE_PER_CELL:
        return _assemble_per_cell_feathered(
            canvas, nx, ny, result_by_cid, halo_px=halo_px, emit=emit,
            capture=capture, holdout_cap=holdout_cap, output_dir=output_dir,
            capture_identity=capture_identity,
            r40_capture=r40_capture, r40_context=r40_context,
        )
    # ZM-ZEGRID-R53: make the mode semantics explicit. The halo still drives the
    # REAL patch geometry/resources, but the anti-checkerboard feather is only
    # applied by the per_cell assembly (one-owner global assembly is unchanged).
    _halo_for_log = int(zsw.HALO_PX if halo_px is None else halo_px)
    if emit is not None:
        try:
            emit(
                "ZeGrid: cell_gauge_mode=global — anti-checkerboard feather NOT "
                "applied (single-owner assembly); requested "
                f"zegrid_halo_px={_halo_for_log} still defines the patch/gauge "
                "geometry",
                "INFO",
            )
        except Exception:
            pass
    return zmosaic.assemble_canvas(canvas, nx, ny, cores), None


def _stack_cell(task):
    """Module-level (picklable) worker: build cache + stack ONE cell (R20).

    Task = ``(cell, patch, mem, cache_dir, mode, bound)`` (the cell context is
    pre-built in the parent so membership is not recomputed); the read-only
    frame/canvas/gauge inputs come from the module globals set by
    :func:`_init_stack_worker`. Builds the aligned cache SERIALISED (workers=1
    -> no nested pool; the cache build is still concurrent ACROSS cells via the
    cell pool), subsets the global gauge, runs the in-memory or streaming stack
    (bit-equal to the serial path), crops the core, deletes the cache, and
    returns a picklable result. Identical in the parent (serial fallback) and in
    a pool worker.
    """
    (cell, patch, mem, cache_dir, mode, bound) = task
    cid = cell.cell_id
    _patch_bounds = [int(patch.patch.x0), int(patch.patch.y0),
                     int(patch.patch.x1), int(patch.patch.y1)]
    _patch_shape_hw = [int(patch.patch.height), int(patch.patch.width)]
    if not mem.patch_ids:
        return {"cell_id": cid, "status": "empty",
                "record": {
                    "cell_id": cid, "status": "empty", "mode": None,
                    "n_contributors": 0,
                    "n_patch_contributors": 0,
                    "patch_bounds": _patch_bounds,
                    "patch_shape_hw": _patch_shape_hw,
                },
                "reproject": None}
    cache_dir = Path(cache_dir)

    # Build the cell cache (parallel reproject; the worker count is bounded by
    # the joint planner — full budget when only one cell is active, cpu//K for K
    # concurrent cells, so the total never explodes to K x cache_build_workers).
    zxe.reset_reproject_path_stats()
    t0 = time.perf_counter()
    manifest = _build_one_cell_cache(
        _STACK_DESCS, _STACK_CANVAS, cell, patch, mem, cache_dir,
        workers=int(_STACK_CACHE_WORKERS),
    )
    cache_build_s = time.perf_counter() - t0

    # Per-Cell photometric normalization (ZM-CELL-PHOTOMETRY-R34B).
    #   "global"  (default): subset the R11 global gauge to this Cell's frames
    #             (no re-anchor) — bit-for-bit unchanged production behaviour.
    #   "per_cell" (opt-in): compute this Cell's OWN self-anchored gauge from its
    #             aligned frames via the FROZEN estimator, so a Cell with no
    #             globally-active frame calibrates instead of raising
    #             "cell gauge has no active frame".
    cell_gauge_mode = zphot.resolve_cell_gauge_mode(
        getattr(_STACK_CONFIG, "cell_gauge_mode", None)
    )
    cell_provider = zfp.MemmapCanonicalProvider(cache_dir)
    try:
        cell_frame_ids = list(cell_provider.frame_ids)
        cell_fixed, gauge_status, gauge_detail = _resolve_cell_normalization(
            cell_gauge_mode, _STACK_GAUGE, _STACK_GLOBAL_FRAME_IDS,
            cell_provider, _STACK_CONFIG,
        )
    finally:
        cell_provider.close()

    if cell_fixed is None:
        # Explicit, DETERMINISTIC zero-support Cell: no gauge, no fabricated data.
        # (Distinct from a geometrically-empty Cell — this one HAS contributors but
        # none carry finite support.)
        _safe_rmtree(cache_dir, None)
        return {
            "cell_id": cid, "status": "empty", "reproject": None,
            "gauge_status": gauge_status, "gauge_detail": gauge_detail,
            "n_frames": 0, "total_bytes": 0, "cache_build_s": cache_build_s,
            "stack_s": 0.0, "peak_rss_kib": zsw.peak_rss_kib(),
            "cleanup_failed": False, "cache_dir": str(cache_dir),
            "record": {
                "cell_id": cid, "row": cell.row, "col": cell.col,
                "status": "empty", "mode": mode,
                "n_contributors": len(mem.patch_ids),
                "n_patch_contributors": len(mem.patch_ids),
                "patch_bounds": _patch_bounds,
                "patch_shape_hw": _patch_shape_hw,
                "cell_gauge_mode": cell_gauge_mode,
                "gauge_status": gauge_status, "gauge_detail": gauge_detail,
            },
        }

    t0 = time.perf_counter()
    if mode == "inmem":
        # ZM-ZEGRID-R22: when the cell runs on the GPU, use the VRAM-bounded tile
        # size (never a single N x full-cell tile); CPU keeps the legacy
        # single-tile full-patch behaviour.
        inmem_tile = _STACK_TILE_SIZE if getattr(_STACK_CONFIG, "backend", "cpu") == "gpu" else None
        mt, sres = _run_cell_inmem(cache_dir, patch, _STACK_CONFIG, None,
                                   fixed=cell_fixed, tile_size=inmem_tile)
    else:
        mt, sres = _run_cell_stream(cache_dir, patch, _STACK_CONFIG, None,
                                    tile_size=_STACK_TILE_SIZE, fixed=cell_fixed)
    stack_s = time.perf_counter() - t0

    core = za.crop_all_planes_to_core(mt)
    peak_rss = zsw.peak_rss_kib()

    prov = _reference_provenance(
        _STACK_GLOBAL_REF, cell_frame_ids, sres.reference_frame_id,
        per_cell=(cell_gauge_mode == zphot.CELL_GAUGE_MODE_PER_CELL),
    )
    record = {
        "cell_id": cid, "row": cell.row, "col": cell.col, "status": "complete",
        "mode": mode, "n_contributors": len(mem.patch_ids),
        "n_patch_contributors": len(mem.patch_ids),
        "patch_bounds": _patch_bounds,
        "patch_shape_hw": _patch_shape_hw,
        "cell_gauge_mode": cell_gauge_mode,
        "backend_used": getattr(_STACK_CONFIG, "backend", "cpu"),
        "reference_frame_id": sres.reference_frame_id,
        "reference_frame_role": prov["reference_frame_role"],
        "bookkeeping_reference_frame_id": prov["bookkeeping_reference_frame_id"],
        "excluded": [list(e) for e in sres.excluded],
        "bound_bytes": int(bound),
    }

    # Per-cell temp reuse: delete the cache after stacking (bounded disk). R15
    # best-effort; a failure is reported so the parent can WARN + record.
    cleanup_failed = not _safe_rmtree(cache_dir, None)

    result = {
        "cell_id": cid, "status": "complete", "core": core, "record": record,
        "backend_used": getattr(_STACK_CONFIG, "backend", "cpu"),
        "n_frames": int(manifest.get("n_frames", 0)),
        "total_bytes": int(manifest.get("total_bytes", 0)),
        "cache_build_s": cache_build_s, "stack_s": stack_s,
        "peak_rss_kib": peak_rss, "cleanup_failed": cleanup_failed,
        "reproject": zxe.get_reproject_path_stats().to_dict(),
        "cache_dir": str(cache_dir),
        # R34D: in per_cell mode the parent needs the PATCH planes (core + halo)
        # to measure the inter-cell overlaps and to feather-blend. Global mode
        # returns only the core (byte-unchanged payload).
        "patch": (
            {
                "bounds": (patch.patch.y0, patch.patch.y1, patch.patch.x0, patch.patch.x1),
                "core_slice": (
                    patch.core_slice.y0, patch.core_slice.y1,
                    patch.core_slice.x0, patch.core_slice.x1,
                ),
                "science": mt.science,
                "valid_mask": mt.valid_mask,
                "surviving": mt.surviving_sample_count,
            }
            if cell_gauge_mode == zphot.CELL_GAUGE_MODE_PER_CELL
            else None
        ),
    }

    # ZM-ZEGRID-R40-B2-C: attach the sparse covariate transport ONLY when the
    # parent armed it (capture ON). When OFF this adds NO key, keeps NO payload and
    # leaves the result byte-identical to the pre-R40 transport.
    _r40_sparse = _r40_maybe_worker_sparse(cid, mt, patch)
    if _r40_sparse is not None:
        result[R40_SPARSE_RESULT_KEY] = _r40_sparse
    return result


# ---------------------------------------------------------------------------
# FITS output (legacy-compatible paths)
# ---------------------------------------------------------------------------

def _canvas_header(canvas, ndim, channels=None):
    w = canvas.wcs()
    header = w.to_header(relax=True)
    header["NAXIS"] = int(ndim)
    header["NAXIS1"] = int(canvas.width)
    header["NAXIS2"] = int(canvas.height)
    if ndim == 3:
        header["NAXIS3"] = int(channels)
    return header


def _array_sha256(data: np.ndarray) -> str:
    """SHA-256 of a float32 array's raw bytes (contiguous, deterministic)."""
    arr = np.ascontiguousarray(data)
    return hashlib.sha256(arr.tobytes()).hexdigest()


HALO_PROVENANCE_NOTE = (
    "H is an output-canvas geometric/feather width. It softens transitions "
    "between Cells but does NOT remove gradients present in the input images. "
    "In global gauge mode the one-owner assembly is used and the "
    "anti-checkerboard feather is not applied; H still drives patch geometry, "
    "contributor membership and stacking."
)


def _halo_provenance_for_pass(
    base_provenance,
    *,
    effective_px,
    cell_ctxs,
    requested_gauge_mode,
    effective_gauge_mode,
    feather_info,
):
    """Add the PASS-local truthful facts to the run-level halo provenance.

    ``base_provenance`` is the compact object resolved ONCE by
    :func:`run_zegrid_mode` (requested/effective/source/units). This function never
    re-reads configuration. It fails loudly if the effective width or the
    feather/state facts contradict each other, so the manifest, FITS cards and
    run log can never publish incompatible claims.
    """
    eff = int(effective_px)
    if base_provenance is None:
        base = build_halo_provenance(None, eff, ZEGRID_HALO_SOURCE_INTERNAL_ONLY)
    else:
        base = dict(base_provenance)
        if int(base["effective_px"]) != eff:
            raise RuntimeError(
                "ZeGrid halo provenance contradiction: run-level effective "
                f"{base['effective_px']} != pass effective {eff}"
            )
    if int(base.get("effective_px", eff)) != eff:
        raise RuntimeError(
            "ZeGrid halo provenance contradiction: run-level effective "
            f"{base.get('effective_px')} != pass effective {eff}"
        )

    contexts = list(cell_ctxs or [])
    geometry_ok = bool(contexts) and all(
        int(patch.halo_px) == eff for (_r, _c, _cell, patch, _mem) in contexts
    )
    mode = str(effective_gauge_mode)
    if mode not in ("global", "per_cell"):
        raise RuntimeError(f"ZeGrid halo provenance: unresolved gauge mode {mode!r}")
    feather_applied = bool(mode == "per_cell" and feather_info is not None)
    if mode == "per_cell" and feather_info is None:
        raise RuntimeError(
            "ZeGrid halo provenance contradiction: per_cell mode ran without "
            "feather assembly"
        )
    if mode != "per_cell" and feather_info is not None:
        raise RuntimeError(
            "ZeGrid halo provenance contradiction: feather assembly ran in "
            f"{mode} gauge mode"
        )
    if feather_applied:
        got = int((feather_info or {}).get("halo_px", -1))
        if got != eff:
            raise RuntimeError(
                "ZeGrid halo provenance contradiction: feather halo "
                f"{got} != effective {eff}"
            )

    return {
        **base,
        "controls_patch_geometry": bool(geometry_ok),
        "controls_contributor_membership": bool(geometry_ok),
        "controls_gauge_patch_domain": bool(mode == "per_cell"),
        "requested_gauge_mode": str(requested_gauge_mode),
        "effective_gauge_mode": mode,
        "feather_applied": bool(feather_applied),
        "feather_px": int(eff) if feather_applied else None,
        "measurement_px": int(2 * eff) if mode == "per_cell" else None,
        "note": HALO_PROVENANCE_NOTE,
    }


def _validate_halo_provenance_consistency(
    halo_provenance, science_config, cell_feather_info
):
    """Cross-check the pass provenance against the EXISTING gauge/feather facts.

    Pure, side-effect-free helper: it never reads config and never writes. It is
    called in ``_run_single`` BEFORE the raw science FITS (or any current-run
    output) is published, and defensively again in ``_write_outputs``. One
    implementation, one source of checks.
    """
    if halo_provenance is None:
        return
    _hp_mode = str(halo_provenance.get("effective_gauge_mode"))
    _cfg_mode = str(getattr(science_config, "cell_gauge_mode", "global"))
    if _hp_mode != _cfg_mode:
        raise RuntimeError(
            "ZeGrid halo provenance contradiction: provenance gauge "
            f"{_hp_mode!r} != science_config {_cfg_mode!r}"
        )
    _feather_ran = cell_feather_info is not None
    if bool(halo_provenance.get("feather_applied")) != bool(_feather_ran):
        raise RuntimeError(
            "ZeGrid halo provenance contradiction: feather_applied="
            f"{halo_provenance.get('feather_applied')} but cell_feather "
            f"present={_feather_ran}"
        )
    if _feather_ran:
        if int(cell_feather_info.get("halo_px")) != int(
            halo_provenance["effective_px"]
        ):
            raise RuntimeError(
                "ZeGrid halo provenance contradiction: cell_feather.halo_px="
                f"{cell_feather_info.get('halo_px')} != effective "
                f"{halo_provenance['effective_px']}"
            )
        if _hp_mode != "per_cell":
            raise RuntimeError(
                "ZeGrid halo provenance contradiction: feather ran in "
                f"non-per_cell mode {_hp_mode!r}"
            )


def _reproducibility_note(effective_gauge_mode):
    """Mode-conditional reproducibility wording (no contradictory claim)."""
    if str(effective_gauge_mode) == "per_cell":
        return (
            "output science uses the OPT-IN per-Cell gauge: every Cell "
            "self-anchors on its own greatest-support frame; inter-Cell "
            "additive offsets and the feathered assembly are applied by the "
            "per_cell path (see photometric_gauge and zegrid_halo). The "
            "global-gauge layout-independence statement does NOT apply in "
            "this mode."
        )
    return (
        "output science is GLOBAL-gauge normalized (one reference frame + "
        "per-frame coefficients computed over the full canvas), so the science "
        "is layout-INDEPENDENT under the corrected global gauge "
        "(ZM-ZEGRID-R11 rework-1: Cells lacking the global reference keep its "
        "level, not a re-anchored local level). Pin the layout via the "
        "'zegrid_layout' config key (e.g. '6x5') for reproducible output; "
        "coverage (stack depth) is pure geometry."
    )


def _record_halo_cards(header, provenance):
    """Add the documented compact ZeGrid halo cards to a FITS primary header.

    Applied to EVERY emitted ZeGrid FITS (raw science, aesthetic float, coverage,
    optional uint16). Card names stay short/standard (<= 8 chars) and carry only
    scalar provenance; array content/dtype/axis order are untouched.
    """
    if provenance is None:
        return header
    header["ZGHALO"] = (
        int(provenance["effective_px"]),
        "ZeGrid halo width, output canvas px",
    )
    header["ZGFEATH"] = (
        bool(provenance["feather_applied"]),
        "anti-checkerboard feather applied",
    )
    header["ZGGMODE"] = (
        str(provenance["effective_gauge_mode"]),
        "effective ZeGrid cell gauge mode",
    )
    return header


def _format_halo_provenance(provenance) -> str:
    """ONE stable run-log line for the halo provenance (requested->effective)."""
    p = provenance or {}
    req = "absent" if p.get("requested_px") is None else f"{p['requested_px']} px"
    feather = "yes" if p.get("feather_applied") else "no"
    feather_px = "n/a" if p.get("feather_px") is None else f"{p['feather_px']} px"
    meas_px = "n/a" if p.get("measurement_px") is None else f"{p['measurement_px']} px"
    return (
        "ZeGrid halo provenance: "
        f"requested={req} (source={p.get('source')}) -> "
        f"effective={p.get('effective_px')} px ({p.get('units')}); "
        f"gauge requested={p.get('requested_gauge_mode')} -> "
        f"effective={p.get('effective_gauge_mode')}; "
        f"patch_geometry={'true' if p.get('controls_patch_geometry') else 'false'} "
        f"contributors={'true' if p.get('controls_contributor_membership') else 'false'} "
        f"gauge_patch_domain={'true' if p.get('controls_gauge_patch_domain') else 'false'}; "
        f"feather_applied={feather} feather={feather_px} measurement={meas_px}; "
        f"{HALO_PROVENANCE_NOTE}"
    )


def _science_header(
    canvas,
    *,
    role: str,
    dtype: str,
    dbe_state: str,
    sha256: str,
    related_file: str | None = None,
) -> fits.Header:
    """Canvas WCS header + science-output role/relationship metadata.

    ``role`` is ``SCI`` (immutable pre-finishing scientific reference) or
    ``AESTH`` (delivered legacy light-DBE aesthetic float32). ``dbe_state`` is
    one of ``off`` / ``on`` / ``noop`` / ``failed`` / ``n/a``. ``related_file``
    records the pre/post-finishing counterpart filename.
    """
    header = _canvas_header(canvas, ndim=3, channels=3)
    header["SCIROLE"] = (role, "science output role")
    header["SCIDTYPE"] = (dtype, "science array dtype")
    header["DBESTAT"] = (dbe_state, "DBE finishing state (off/on/failed/n/a)")
    header["SCIHASH"] = (sha256, "SHA-256 of science array bytes")
    if related_file:
        header["SCIREF"] = (related_file, "related pre/post-finishing output")
    return header


# ---------------------------------------------------------------------------
# Single-pass pipeline (one mount group)
# ---------------------------------------------------------------------------

def _gauge_decode(frame_desc):
    """Module-level (picklable) 1-arg decode wrapper for the parallel gauge."""
    return _decode_frame_hwc(frame_desc, None)


def _reference_provenance(global_reference_frame_id, cell_frame_ids, cell_reference_frame_id, per_cell=False):
    """ZM-ZEGRID-R11 L1 (provenance honesty): classify a Cell's reference.

    A Cell's ``reference_frame_id`` is the TRUE photometric anchor only when the
    Cell contains the global reference frame; otherwise it is a BOOKKEEPING
    placeholder (the Cell's highest-weight active frame) and the true anchor is
    ``photometric_gauge.global_reference_frame_id``.

    ``per_cell`` (ZM-CELL-PHOTOMETRY-R34B): in the opt-in per-Cell gauge mode the
    Cell's reference is its OWN self-anchoring frame (a TRUE local photometric
    anchor), so it is labelled accordingly and there is no global bookkeeping
    placeholder.

    Returns ``{reference_frame_role, bookkeeping_reference_frame_id}``.
    """
    if per_cell:
        return {
            "reference_frame_role": "cell_local_photometric_anchor",
            "bookkeeping_reference_frame_id": None,
        }
    has_global = global_reference_frame_id in cell_frame_ids
    return {
        "reference_frame_role": (
            "global_photometric_anchor" if has_global else "bookkeeping_placeholder"
        ),
        "bookkeeping_reference_frame_id": (
            None if has_global else cell_reference_frame_id
        ),
    }


# Canonical ZeGrid phase order (used for the R14 global ETA weights).
_PHASE_ORDER = ("setup", "layout", "gauge", "cache_build", "per_cell_stack", "assembly")


def _fmt_throughput(items, elapsed_s, unit):
    """Human-readable throughput (``items / elapsed_s``), or ``None`` when unknown."""
    try:
        if elapsed_s is None or elapsed_s <= 0.0:
            return None
        return f"{float(items) / float(elapsed_s):.1f} {unit}"
    except Exception:
        return None


def _open_run_log(output_dir, start_ts):
    """Create the run log with a header so it is readable DURING the run.

    rework-1 (I2): a re-run into the SAME output folder would silently replace the
    previous run log. We keep the single ``zegrid_run.log`` name (the manifest/
    tests reference it) but, when a previous log already exists, record that it is
    being replaced in the new header so the overwrite is explicit, not silent.
    """
    try:
        path = Path(output_dir) / RUN_LOG_NAME
        replaced = ""
        if path.exists():
            replaced = "replaces_previous_log: true (previous run log overwritten)\n"
        path.write_text(
            "ZeGrid run log\n===============\n"
            f"started: {start_ts}\n"
            f"{replaced}\n"
            "[Live phase log]\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def _append_run_log_line(output_dir, line):
    """Append + flush one line to the run log (readable during the run)."""
    try:
        with (Path(output_dir) / RUN_LOG_NAME).open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass


def _manifest_gpu_block(gpu_used, gpu_context=None) -> dict:
    """ZM-ZEGRID-R22 (rework-1): truthful manifest ``gpu`` block from EXECUTION evidence.

    ``used`` / ``effective`` / ``backend`` are derived from the ACTUAL executed
    cells (per-cell ``backend_used`` aggregated in ``_run_single``), never from the
    start-time selection alone. Distinguishes requested / available / attempted
    (selected) / final_effective / actually_used / final_backend + start-time and
    runtime fallback reasons.
    """
    ctx = gpu_context or {}
    actually_used = bool(ctx.get("actually_used"))
    final_backend = ctx.get("final_backend") or ("gpu" if actually_used else "cpu")
    backend = "gpu" if actually_used else "cpu"
    return {
        "used": actually_used,
        "requested": bool(ctx.get("requested")),
        "requested_source": ctx.get("requested_source"),
        "available": bool(ctx.get("available")),
        "attempted": bool(ctx.get("attempted")),
        "effective": bool(ctx.get("final_effective", ctx.get("attempted"))),
        "actually_used": actually_used,
        "final_backend": final_backend,
        "gpu_executed_cells": ctx.get("gpu_executed_cells"),
        "device": ctx.get("device"),
        "cupy_version": ctx.get("cupy_version"),
        "vram_total_bytes": ctx.get("vram_total_bytes"),
        "vram_free_bytes": ctx.get("vram_free_bytes"),
        "vram_budget_bytes": ctx.get("vram_budget_bytes"),
        "fallback_reason": ctx.get("fallback_reason"),
        "runtime_fallback": ctx.get("runtime_fallback"),
        # ZM-ZEGRID-R36 (F2): a GPU selected at start but RAM-infeasible is caught
        # BEFORE execution and degraded to exact CPU (recorded, never hidden).
        "pre_execution_fallback": ctx.get("pre_execution_fallback"),
        # ZM-ZEGRID-R36: the CPU streaming tile (RAM-adaptive; used verbatim by any
        # GPU->CPU fallback) and the GPU streaming tile (the GPU success path).
        "cpu_streaming_tile": ctx.get("cpu_streaming_tile"),
        "gpu_streaming_tile": ctx.get("gpu_streaming_tile"),
        "gpu_streaming_host_peak_bytes": ctx.get("gpu_streaming_host_peak_bytes"),
        "cpu_fallback_plan": ctx.get("cpu_fallback_plan"),
        "backend": {
            "gauge_rejection": "cpu",
            "gauge_combine": "cpu",
            "per_cell_rejection": backend,
            "per_cell_combine": backend,
        },
        "note": zin.GPU_USAGE_NOTE,
    }


def _write_run_log(
    output_dir,
    timings,
    *,
    start_ts,
    frames_loaded,
    n_included,
    n_rejected,
    canvas,
    layout,
    gpu_used,
    ignored_settings,
    peak_rss_kib,
    cache_info,
    global_reference_frame_id,
    ignored_run_args=None,
    gauge_diagnostics=None,
    per_cell_diagnostics=None,
    finishing_info=None,
    gpu_context=None,
    aggregate_peak_rss_kib=None,
    halo_provenance=None,
):
    """Append the run-log SUMMARY into the output folder.

    The user's real complaint included "le log est absent": ZeGrid produced no
    run log. The live header + per-phase lines are flushed DURING the run (see
    :func:`_open_run_log` / :func:`_append_run_log_line`); this appends the final
    summary (per-phase wall-clock timings, an explicit GPU-usage statement, and
    the list of ignored product settings) so the complete log is readable both
    during AND after the run.
    """
    lines = [
        "",
        f"frames_loaded: {frames_loaded}  included: {n_included}  rejected_wcs: {n_rejected}",
        f"canvas: {canvas.width}x{canvas.height} (resolution {canvas.resolution_deg:.3g} deg/px)",
        f"layout: {layout['nx']}x{layout['ny']} (source={layout.get('layout_source', 'auto')})",
        _format_halo_provenance(halo_provenance),
        "",
        "Timings (wall-clock):",
    ]
    for name, sec in timings.to_dict().items():
        lines.append(f"  {name}: {sec:.3f}s")
    lines.append(f"  total: {timings.total():.3f}s")
    lines.append("")
    lines.append("GPU usage:")
    lines.append(f"  used: {bool(gpu_used)}")
    _gctx = gpu_context or {}
    lines.append(
        f"  requested: {bool(_gctx.get('requested'))} (source={_gctx.get('requested_source')})  "
        f"available: {bool(_gctx.get('available'))}  attempted: {bool(_gctx.get('attempted'))}"
    )
    lines.append(
        f"  actually_used: {bool(_gctx.get('actually_used'))}  "
        f"final_backend: {_gctx.get('final_backend')}  "
        f"gpu_executed_cells: {_gctx.get('gpu_executed_cells')}"
    )
    if _gctx.get("device"):
        lines.append(
            f"  device: {_gctx.get('device')}  cupy={_gctx.get('cupy_version')}  "
            f"vram_total={_gctx.get('vram_total_bytes')}  vram_free={_gctx.get('vram_free_bytes')}  "
            f"budget={_gctx.get('vram_budget_bytes')}"
        )
    if _gctx.get("fallback_reason"):
        lines.append(f"  fallback_reason: {_gctx.get('fallback_reason')}")
    if _gctx.get("runtime_fallback"):
        _rf = _gctx["runtime_fallback"]
        lines.append(f"  runtime_fallback: type={_rf.get('type')} reason={_rf.get('reason')}")
    lines.append(f"  {zin.describe_gpu_usage()}")
    lines.append("")
    lines.append("Ignored product settings (ZeGrid honours use_gpu_* / stack_use_gpu; "
                 "no post-stack processing):")
    lines.extend(zin.ignored_settings_warning_lines(ignored_settings))
    lines.append("")
    lines.append("Accepted-but-ignored run_zegrid_mode arguments:")
    lines.extend(zin.describe_ignored_run_args(ignored_run_args or {}))
    lines.append("")
    lines.append("Gauge parallelism (ZM-ZEGRID-R19 diagnostic):")
    if gauge_diagnostics:
        gd = gauge_diagnostics
        lines.append(
            f"  executor: {gd.get('executor')}  parent_daemon: {gd.get('parent_daemon')}  "
            f"workers: {gd.get('workers')}  fallback: {gd.get('fallback')}"
        )
        for phase_name in ("counts", "pairs"):
            ph = (gd.get("phases") or {}).get(phase_name) or {}
            lines.append(
                f"  phase={phase_name}: executor={ph.get('executor')} units={ph.get('units')} "
                f"seconds={ph.get('seconds')} seconds_per_unit={ph.get('seconds_per_unit')}"
                + (f" fallback_reason={ph.get('fallback_reason')}" if ph.get('fallback_reason') else "")
            )
        # ZM-ZEGRID-R32a: bounded per-subsystem gauge instrumentation (additive;
        # never changes science). Rendered only when present.
        _ginstr = gd.get("instrumentation") or {}
        if _ginstr:
            _lv = _ginstr.get("levers") or {}
            _ct = _ginstr.get("counts") or {}
            _pr = _ginstr.get("pairs") or {}
            _rf = _ginstr.get("reference") or {}
            _tot = _ginstr.get("totals") or {}
            _mem = _ginstr.get("memory") or {}
            lines.append(
                "  instrumentation(R32a): "
                f"prefilter_pairs={_lv.get('prefilter_pairs')} "
                f"dedup_reference={_lv.get('dedup_reference')} "
                f"bounded_pairs={_lv.get('bounded_pairs')}"
            )
            lines.append(
                f"    counts: reproject_s={_ct.get('reproject_s')} "
                f"bbox_area_px={_ct.get('bbox_area_px')} "
                f"bytes_reprojected={_ct.get('bytes_reprojected')}"
            )
            lines.append(
                f"    pairs: reproject_s={_pr.get('reproject_s')} "
                f"normalize_s={_pr.get('normalize_s')} weight_s={_pr.get('weight_s')} "
                f"crop_area_px={_pr.get('crop_area_px')} "
                f"prefiltered={_pr.get('pairs_prefiltered')} "
                f"reproject_s_per_crop_px={_pr.get('reproject_s_per_crop_px')}"
            )
            lines.append(
                f"    reference: reproject_s={_rf.get('reproject_s')} "
                f"normalize_s={_rf.get('normalize_s')} weight_s={_rf.get('weight_s')} "
                f"area_px={_rf.get('area_px')}"
            )
            lines.append(
                f"    totals: wall_s={_tot.get('wall_seconds')} "
                f"reproject_seconds_sum={_tot.get('reproject_seconds_sum')} "
                f"bytes_lifted={_tot.get('bytes_lifted')}"
            )
            lines.append(
                f"    memory: peak_rss_kib_start={_mem.get('peak_rss_kib_start')} "
                f"peak_rss_kib_end={_mem.get('peak_rss_kib_end')} "
                f"worker_rss_kib_max={_mem.get('worker_rss_kib_max')}"
            )
    else:
        lines.append("  (no gauge diagnostics recorded)")
    lines.append("Per-cell stacking concurrency (ZM-ZEGRID-R20 diagnostic):")
    if per_cell_diagnostics:
        pcd = per_cell_diagnostics
        lines.append(
            f"  executor: {pcd.get('executor')}  parent_daemon: {pcd.get('parent_daemon')}  "
            f"cells_in_flight: {pcd.get('cells_in_flight')}  workers: {pcd.get('workers')}  "
            f"fallback: {pcd.get('fallback')}"
        )
        lines.append(
            f"  cpu: {pcd.get('cpu')}  avail: {pcd.get('available_bytes', 0) / 2**30:.2f}GiB  "
            f"budget: {pcd.get('ram_budget_bytes', 0) / 2**30:.2f}GiB  "
            f"footprint: {pcd.get('per_cell_footprint_bytes', 0) / 2**20:.0f}MiB"
        )
        lines.append(
            f"  cells: {pcd.get('cells')}  seconds: {pcd.get('seconds')}  "
            f"seconds_per_unit: {pcd.get('seconds_per_unit')}  "
            f"cache_build_cpu_sum_s: {pcd.get('cache_build_cpu_sum_s')}  "
            f"stack_cpu_sum_s: {pcd.get('stack_cpu_sum_s')}"
            + (f" fallback_reason={pcd.get('fallback_reason')}" if pcd.get('fallback_reason') else "")
        )
    else:
        lines.append("  (no per-cell concurrency diagnostics recorded)")
    lines.append("Reprojection path (ZM-ZEGRID-R21 diagnostic):")
    _reproj = zxe.merge_reproject_stats(
        (gauge_diagnostics or {}).get("reprojection"),
        (per_cell_diagnostics or {}).get("reprojection"),
    )
    lines.append(
        f"  path: {_reproj.get('path')}  fast_calls={_reproj.get('fast_calls')} "
        f"fallback_calls={_reproj.get('fallback_calls')} "
        f"fast_seconds={_reproj.get('fast_seconds')} "
        f"fallback_seconds={_reproj.get('fallback_seconds')}"
    )
    if _reproj.get("fallback_reason"):
        lines.append(f"  fallback_reason: {_reproj.get('fallback_reason')}")
    lines.append("Final-mosaic finishing (ZM-ZEGRID-R23 rework-3):")
    fin = finishing_info or {}
    if not fin or not fin.get("enabled"):
        lines.append("  disabled (no finishing settings applied)")
    else:
        dbe = fin.get("dbe", {})
        rgb = fin.get("rgb_equalize", {})
        hf = fin.get("hole_fill", {})
        u16 = fin.get("uint16", {})
        lines.append(f"  enabled: {bool(fin.get('enabled'))}")
        lines.append(
            f"  failed: {bool(fin.get('failed'))} {fin.get('failure_reason') or ''}".rstrip()
        )
        lines.append(
            f"  dbe: enabled={dbe.get('enabled')} applied={dbe.get('applied')} "
            f"attempted={dbe.get('attempted')} reason={dbe.get('reason') or ''} "
            f"algorithm={dbe.get('algorithm')} "
            f"strength={dbe.get('strength')} "
            f"params_source={dbe.get('params_source')} "
            f"params={dbe.get('params')}"
        )
        for ch in (dbe.get("channels") or []):
            if isinstance(ch, dict):
                before = ch.get("before") or {}
                after = ch.get("after") or {}
                lines.append(
                    f"    ch{ch.get('channel')}: applied={ch.get('applied')} "
                    f"median={ch.get('median')} robust_sigma={ch.get('robust_sigma')} "
                    f"obj_frac={ch.get('obj_frac')} "
                    f"before_full_std={ch.get('before_full_std')} "
                    f"after_full_std={ch.get('after_full_std')} "
                    f"full_ratio={ch.get('full_ratio')} "
                    f"neg_frac {before.get('neg_frac')} -> {after.get('neg_frac')}"
                )
        lines.append(
            f"  rgb_equalize: enabled={rgb.get('enabled')} applied={rgb.get('applied')} "
            f"skies_before={rgb.get('skies_before')} skies_after={rgb.get('skies_after')}"
        )
        lines.append(
            f"  hole_fill: enabled={hf.get('enabled')} applied={hf.get('applied')} "
            f"reason={hf.get('reason') or ''} filled_px={hf.get('filled_px')} "
            f"hole_px={hf.get('hole_px')} max_radius_px={hf.get('max_radius_px')} "
            f"blend={hf.get('blend')} only_near_seams={hf.get('only_near_seams')} "
            f"protect_stars_details={hf.get('protect_stars_details')}"
        )
        lines.append(
            f"  uint16: enabled={u16.get('enabled')} written={u16.get('written')} "
            f"vmin={u16.get('vmin')} vmax={u16.get('vmax')}"
        )
    lines.append(
        "  NOTE: aesthetic output is not photometrically neutral; use the "
        "science reference FITS for measurement."
    )
    lines.append("")
    lines.append(f"peak_rss_kib: {peak_rss_kib}")
    lines.append(
        f"aggregate_peak_rss_kib: {aggregate_peak_rss_kib if aggregate_peak_rss_kib is not None else 'n/a'} "
        "(parent + cell workers, UPPER BOUND incl. per-worker import baseline)"
    )
    lines.append(f"cache: {json.dumps(cache_info, sort_keys=True)}")
    lines.append(f"photometric_gauge.global_reference_frame_id: {global_reference_frame_id}")
    text = "\n".join(lines) + "\n"
    try:
        with (Path(output_dir) / RUN_LOG_NAME).open("a", encoding="utf-8") as f:
            f.write(text)
            f.flush()
    except Exception:
        pass
    return text


def _build_chosen_cell_contexts(footprints, canvas, nx, ny, layout_substep=None, *, halo_px=None):
    """Build the final chosen-cell contexts from the cached footprints (R28).

    Brackets the work with the ``layout:chosen_cells`` sub-step (START + throttled
    progress + END) and returns ``[(row, col, cell, patch, mem), ...]`` in
    row-major order. ``layout_substep`` is a :class:`SubstepReporter` (or ``None``
    to skip observability). This is the production seam for the chosen-cell
    membership/context preparation (no per-cell WCS re-projection — it reuses the
    once-projected footprint set).

    ``halo_px`` (ZM-ZEGRID-R53) is the EFFECTIVE product halo width. The product
    path always passes it; ``None`` keeps the historical low-level helper default
    (``sweep.HALO_PX``) for direct/unit calls only.
    """
    _halo = int(zsw.HALO_PX if halo_px is None else halo_px)
    cell_total = nx * ny
    if layout_substep is not None:
        layout_substep.start(
            zobs.LAYOUT_SUBSTEP_CHOSEN_CELLS,
            "chosen-cell membership/context preparation",
            total=cell_total,
        )
    cell_ctxs = []
    done = 0
    for row, col, cell, patch, mem in zsw.build_cell_contexts_from_footprints(
        footprints, canvas, nx, ny, halo_px=_halo
    ):
        cell_ctxs.append((row, col, cell, patch, mem))
        done += 1
        if layout_substep is not None:
            layout_substep.progress(done, item_id=cell.cell_id, total=cell_total)
    if layout_substep is not None:
        layout_substep.end()
    return cell_ctxs


# ---------------------------------------------------------------------------
# ZM-ZEGRID-R41-B — ONE executable bundle built under the PLAN budget (80 %)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionBundle:
    """The COMPLETE executable plan built under ONE raw-observation budget.

    This is the SINGLE source of truth for the per-cell execution: layout,
    memberships, CPU streaming tile, GPU effective tile/host bound, per-cell
    bounds/specs, the joint mode/concurrency/cache-worker decision, the tasks
    the executor really runs, the CPU fallback bundle, and the auditable
    :class:`ram_contract.PlanEnvelope`. No second, independent plan exists.
    """

    layout: dict
    cell_ctxs: list
    footprints: object
    cpu_streaming: dict
    cpu_streaming_tile: int
    gpu_tile_size: int | None
    gpu_streaming_tile: int | None
    gpu_streaming_host_peak_bytes: int | None
    gpu_used: bool
    gpu_pre_execution_fallback: dict | None
    cell_specs: list
    cell_tasks: list
    plan: dict
    chosen_mode: str
    cells_in_flight: int
    cache_workers_per_cell: int
    per_cell_footprint: int
    cpu_fallback_info: dict
    cpu_fallback_plan: dict | None
    cpu_fallback_tasks: list | None
    envelope: object
    shared_bytes: int
    cache_worker_footprint_bytes: int
    worst_patch_px: int
    gauge_workers: int
    cache_build_workers: int
    worker_budget_bytes: int
    attempts: list


def _build_execution_bundle(
    canvas,
    descs,
    footprints,
    *,
    raw_available_bytes,
    budget_bytes,
    pinned_layout=None,
    gpu_used=False,
    vram_budget_bytes=None,
    cpu_count=None,
    cache_build_workers=None,
    gauge_workers=None,
    cache_root=None,
    emit=None,
    log_line=None,
    layout_substep=None,
    min_refinement_rank=None,
    max_refinement_rank=None,
    max_cpu_tile=None,
    max_gpu_tile=None,
    max_cells_in_flight=None,
    halo_px=None,
):
    """Build the COMPLETE executable bundle under the PLAN budget (ZM-ZEGRID-R41-B).

    Reuses ONLY accepted seams: :func:`_choose_layout_mode_aware` (scientific
    floors + candidate enumeration), :func:`_build_chosen_cell_contexts` (R28
    memberships), :func:`zgpu.choose_gpu_tile_size` (VRAM tile) and the opt-in
    joint planner ``zpar.plan_cell_concurrency`` with phase accounting. All the
    budget arithmetic lives in :mod:`ram_contract` — this function NEVER recomputes
    a budget from a float factor.

    ``raw_available_bytes`` is the RAW observation ``budget_bytes`` was derived
    from (``budget_bytes == rcontract.plan_budget_bytes(raw_available_bytes)``);
    it is carried verbatim into the envelope so the admission controller can
    re-validate the candidate against the exact miss raw.

    Bounded refinement (never coarser, never a growth):

    * ``min_refinement_rank`` — never pick a layout coarser than this cell count;
    * ``max_refinement_rank`` — fix the exact geometry (used once scientific work
      has started, so a GPU→CPU fallback cannot relayout the mosaic);
    * ``max_cpu_tile`` / ``max_cells_in_flight`` — the previous envelope's tile /
      concurrency are upper bounds, so a bounded replan can only shrink them
      (the accepted ``assert_no_growth`` invariant).

    Raises :class:`auto_layout.LayoutInfeasible` when the pinned / exact geometry
    cannot be planned, and :class:`parallel.ModeInfeasible` when even one cell
    cannot fit under the PLAN budget (never a clamp-to-1 rescue).

    ``halo_px`` (ZM-ZEGRID-R53) is the SAME effective width for the layout, the
    memberships, the resource arithmetic and every bounded replan / fallback /
    worker payload of ONE run. ``None`` keeps the historical low-level helper
    default (``sweep.HALO_PX``) for direct/unit calls only; the product path
    always passes the validated value.
    """
    _halo = int(zsw.HALO_PX if halo_px is None else halo_px)
    budget_bytes = int(budget_bytes)
    raw_available_bytes = int(raw_available_bytes)
    cpu_count = int(cpu_count or os.cpu_count() or 1)
    shared_bytes = int(zpar._PER_WORKER_BASELINE_BYTES)
    attempts: list = []
    rank_floor = None if min_refinement_rank is None else int(min_refinement_rank)
    rank_ceiling = None if max_refinement_rank is None else int(max_refinement_rank)
    # A pinned / exact-geometry request can NEVER relocate the layout, so the
    # refinement loop below must refuse instead of going finer.
    exact_geometry = pinned_layout is not None or rank_ceiling is not None

    while True:
        _fp_out = []
        layout = _choose_layout_mode_aware(
            canvas, descs, ram_budget=budget_bytes,
            available_bytes=raw_available_bytes, tile_size=STREAM_TILE_SIZE,
            pinned_layout=pinned_layout, emit=emit, log_line=log_line,
            observer=layout_substep, footprints_out=_fp_out,
            footprints_in=footprints,
            min_refinement_rank=rank_floor, max_refinement_rank=rank_ceiling,
            halo_px=_halo,
        )
        if footprints is None:
            footprints = _fp_out[0] if _fp_out else None
        nx, ny = int(layout["nx"]), int(layout["ny"])
        cell_ctxs = _build_chosen_cell_contexts(
            footprints, canvas, nx, ny, layout_substep, halo_px=_halo
        )
        worst_patch_px = 0
        patch_px_for_workers = 0
        for _row, _col, _cell, _patch, _mem in cell_ctxs:
            try:
                _ph, _pw = _patch.patch_shape_hw
                _px = int(_ph) * int(_pw)
            except Exception:
                continue
            patch_px_for_workers = max(patch_px_for_workers, _px)
            if _mem.patch_ids:
                worst_patch_px = max(worst_patch_px, _px)
        cache_worker_footprint_bytes = int(zpar.cache_footprint_bytes(worst_patch_px))
        # Workers are budgeted from the SAME PLAN budget (no second float factor).
        worker_budget = int(budget_bytes)
        canvas_px = int(canvas.width) * int(canvas.height)
        _gauge_fp = zpar.gauge_footprint_bytes(canvas_px)
        _cache_fp = zpar.cache_footprint_bytes(patch_px_for_workers or canvas_px)
        _gauge_workers = (
            zpar.choose_workers(None, worker_budget, _gauge_fp)
            if gauge_workers is None else int(gauge_workers)
        )
        _cache_workers = (
            zpar.choose_workers(None, worker_budget, _cache_fp)
            if cache_build_workers is None else int(cache_build_workers)
        )

        # ZM-ZEGRID-R22/R36: VRAM-bounded GPU tile, determined BEFORE the joint
        # plan so the GPU RAM gate is evaluated at the GPU's OWN execution tile.
        effective_gpu = bool(gpu_used)
        gpu_tile_size = None
        gpu_pre_execution_fallback = None
        _cache_root = None if cache_root is None else Path(cache_root)
        if effective_gpu:
            max_n = max(
                (len(m.patch_ids) for (_r, _c, _ce, _p, m) in cell_ctxs if m.patch_ids),
                default=1,
            )
            worst_hw = (1, 1)
            worst_area = -1
            for (_r, _c, _ce, _p, m) in cell_ctxs:
                if m.patch_ids:
                    hw = _p.patch_shape_hw
                    area = hw[0] * hw[1]
                    if area > worst_area:
                        worst_area = area
                        worst_hw = hw
            _gpu_ts = zgpu.choose_gpu_tile_size(
                max_n, 3, worst_hw, vram_budget_bytes
            )
            if _gpu_ts is None:
                gpu_pre_execution_fallback = {
                    "reason": "vram_tile_infeasible",
                    "detail": "choose_gpu_tile_size returned None (VRAM budget too small)",
                }
                effective_gpu = False
            else:
                gpu_tile_size = int(_gpu_ts)
                if max_gpu_tile is not None:
                    # Never grow the GPU tile either (accepted no-growth invariant).
                    gpu_tile_size = min(int(gpu_tile_size), int(max_gpu_tile))

        # ---- CPU streaming tile: select across the ACTUAL permitted ladder ----
        # ZM-ZEGRID-R41-B (rework-1): the per-cell streaming estimate is NOT
        # monotone in tile size, so a numeric clamp is wrong. EVERY ladder tile
        # permitted by ``max_cpu_tile`` is evaluated against its REAL bound and the
        # budget — a smaller tile may be the only feasible one (and 64 can beat 32).
        permitted_tiles = [
            int(t) for t in CPU_STREAMING_TILE_LADDER
            if max_cpu_tile is None or int(t) <= int(max_cpu_tile)
        ]
        if not permitted_tiles:
            permitted_tiles = [int(CPU_STREAMING_TILE_MINIMUM)]

        active = [
            (idx, len(mem.patch_ids), tuple(patch.patch_shape_hw),
             int(patch.patch.width) * int(patch.patch.height))
            for idx, (row, col, cell, patch, mem) in enumerate(cell_ctxs)
            if mem.patch_ids
        ]
        inmem_by_idx = {
            idx: int(zal.FITTED_MEMORY_MODEL.predict_bound_bytes(n, area))
            for idx, n, hw, area in active
        }

        def _stream_by_idx(tile):
            return {
                idx: int(_estimate_streaming_bytes(n, hw, int(tile)))
                for idx, n, hw, area in active
            }

        gpu_stream_by_idx = (
            _stream_by_idx(gpu_tile_size) if effective_gpu else None
        )

        def _plan_for(pairs, backend_gpu, cells_in_flight_cap):
            return zpar.plan_cell_concurrency(
                pairs, cpu_count, budget_bytes,
                gpu_backend=bool(backend_gpu),
                cache_build_workers=int(_cache_workers),
                shared_bytes=shared_bytes,
                cache_worker_px=int(worst_patch_px),
                max_cells_in_flight=cells_in_flight_cap,
            )

        plan = None
        chosen_tile = int(CPU_STREAMING_TILE_MINIMUM)
        cpu_fallback_plan = None
        gpu_fallback_reason = None
        cpu_refusal = None
        tile_trials = []

        if effective_gpu:
            gpu_pairs = [
                (inmem_by_idx[i], gpu_stream_by_idx[i]) for i, _n, _hw, _a in active
            ]
            try:
                plan = _plan_for(gpu_pairs, True, max_cells_in_flight)
            except zpar.ModeInfeasible as exc:
                gpu_fallback_reason = {
                    "reason": "gpu_stream_host_ram_infeasible",
                    "gpu_streaming_tile": int(gpu_tile_size),
                    "gpu_streaming_host_peak_bytes": int(
                        max((b for _a, b in gpu_pairs), default=0)
                    ),
                    "ram_budget_bytes": int(budget_bytes),
                    "detail": str(exc),
                }
                effective_gpu = False
                gpu_tile_size = None
                gpu_stream_by_idx = None

        if plan is None:
            # CPU: walk the PERMITTED ladder (largest first) and keep the first
            # jointly feasible tile. Never a numeric clamp, never a stale report.
            for t in permitted_tiles:
                stream_by_idx = _stream_by_idx(t)
                pairs = [(inmem_by_idx[i], stream_by_idx[i]) for i, _n, _hw, _a in active]
                worst_stream = int(max((b for _a, b in pairs), default=0))
                try:
                    candidate = _plan_for(pairs, False, max_cells_in_flight)
                except zpar.ModeInfeasible as exc:
                    cpu_refusal = exc
                    tile_trials.append({
                        "tile": int(t), "bound_bytes": worst_stream,
                        "worst_bound_bytes": worst_stream, "fits": False,
                    })
                    continue
                # R41-B (rework-2): the trial's ``fits`` describes whether THIS
                # tile's STREAM mode is the plan actually used — never claim a
                # stream tile fits when the plan passed via in-memory.
                fits_stream = (candidate["mode"] == "stream")
                tile_trials.append({
                    "tile": int(t), "bound_bytes": worst_stream,
                    "worst_bound_bytes": worst_stream, "fits": bool(fits_stream),
                })
                plan = candidate
                chosen_tile = int(t)
                break
        else:
            # GPU plan accepted: still plan the CPU fallback bundle across the
            # ladder, but NEVER abort the (feasible) GPU plan when no CPU mode
            # fits (R36 F-M2 D: a GPU success is allowed with no CPU fallback).
            for t in permitted_tiles:
                stream_by_idx = _stream_by_idx(t)
                pairs = [(inmem_by_idx[i], stream_by_idx[i]) for i, _n, _hw, _a in active]
                worst_stream = int(max((b for _a, b in pairs), default=0))
                try:
                    cpu_fallback_plan = _plan_for(pairs, False, max_cells_in_flight)
                except zpar.ModeInfeasible as exc:
                    cpu_refusal = exc
                    tile_trials.append({
                        "tile": int(t), "bound_bytes": worst_stream,
                        "worst_bound_bytes": worst_stream, "fits": False,
                    })
                    continue
                tile_trials.append({
                    "tile": int(t), "bound_bytes": worst_stream,
                    "worst_bound_bytes": worst_stream,
                    "fits": bool(cpu_fallback_plan.get("mode") == "stream"),
                })
                chosen_tile = int(t)
                break

        if plan is None:
            attempts.append({
                "layout": [nx, ny], "refinement_rank": nx * ny,
                "permitted_tiles": list(permitted_tiles),
                "tile_trials": list(tile_trials),
                "reason": str(cpu_refusal),
            })
            if exact_geometry:
                raise zpar.ModeInfeasible(
                    "exact/pinned geometry "
                    f"{nx}x{ny} is JOINTLY infeasible under the PLAN budget "
                    f"{budget_bytes / 2**20:.1f} MiB (shared "
                    f"{shared_bytes / 2**20:.1f} MiB counted once, worst patch "
                    f"{worst_patch_px} px) even after the FULL permitted CPU tile "
                    f"ladder {permitted_tiles}: {cpu_refusal}. Refusing to change the "
                    "geometry of an already-committed run."
                )
            # Auto: the full ladder is exhausted -> move to the NEXT FINER
            # candidate (never coarser).
            rank_floor = (nx * ny) + 1
            if rank_floor > canvas.width * canvas.height:
                raise zpar.ModeInfeasible(
                    "no layout (even the finest) is jointly feasible under the PLAN "
                    f"budget {budget_bytes / 2**20:.1f} MiB after the full CPU tile "
                    f"ladder {permitted_tiles}: {cpu_refusal}"
                )
            continue

        cpu_streaming_tile = int(chosen_tile)
        chosen_mode = plan["mode"]
        # R41-B (rework-2): the REPORTED stream tile is the one that will actually
        # execute a CPU stream plan — None when the CPU plan is in-memory only or
        # when no CPU fallback exists on a GPU plan (never a fake tile).
        if (not effective_gpu) and chosen_mode == "stream":
            selected_cpu_tile = int(chosen_tile)
        elif cpu_fallback_plan is not None and cpu_fallback_plan.get("mode") == "stream":
            selected_cpu_tile = int(chosen_tile)
        else:
            selected_cpu_tile = None
        # The reported block must describe EXACTLY what will execute. Integer keys
        # (``_cpu_streaming_block`` reads ``stream_worst_by_tile[int(tile)]``).
        _stream_worst_by_tile = {
            int(t): int(max(_stream_by_idx(t).values(), default=0))
            for t in permitted_tiles
        }
        _inmem_worst = int(max(inmem_by_idx.values(), default=0))
        _cs_reason = (
            f"joint tile selection over the permitted ladder {permitted_tiles}: "
            + (
                f"tile {selected_cpu_tile} is the largest jointly feasible stream tile"
                if selected_cpu_tile is not None
                else "no CPU stream tile is used (in-memory plan or no CPU fallback)"
            )
        )
        layout["cpu_streaming"] = _cpu_streaming_block(
            selected_cpu_tile,
            [dict(tr) for tr in tile_trials],
            _cs_reason,
            _stream_worst_by_tile,
            _inmem_worst,
            budget_bytes,
        )
        cpu_streaming = layout.get("cpu_streaming") or {}
        _cs_selected = cpu_streaming.get("selected_tile_size")
        cells_in_flight = int(plan["cells_in_flight"])
        cache_workers_per_cell = int(plan["cache_workers_per_cell"])
        gpu_streaming_tile = int(gpu_tile_size) if effective_gpu else None
        gpu_streaming_host_peak_bytes = (
            int(max(gpu_stream_by_idx.values(), default=0))
            if effective_gpu else None
        )

        cell_specs = []
        for (row, col, cell, patch, mem) in cell_ctxs:
            cid = cell.cell_id
            if not mem.patch_ids:
                cell_specs.append((
                    cell, patch, mem,
                    (None if _cache_root is None else str(_cache_root / cid)),
                    None, 0, None,
                ))
                continue
            n = len(mem.patch_ids)
            area = patch.patch.width * patch.patch.height
            inmem_b = int(zal.FITTED_MEMORY_MODEL.predict_bound_bytes(n, area))
            stream_b = _estimate_streaming_bytes(
                n, patch.patch_shape_hw, cpu_streaming_tile
            )
            gpu_stream_b = (
                _estimate_streaming_bytes(n, patch.patch_shape_hw, gpu_tile_size)
                if effective_gpu else None
            )
            cell_specs.append(
                (cell, patch, mem,
                 (str(_cache_root / cid) if _cache_root is not None else cid),
                 inmem_b, stream_b, gpu_stream_b)
            )

        cell_tasks = []
        for (cell, patch, mem, cache_dir, inmem_b, stream_b, gpu_stream_b) in cell_specs:
            if inmem_b is None:
                cell_tasks.append((cell, patch, mem, cache_dir, None, 0))
            elif chosen_mode == "inmem":
                cell_tasks.append((cell, patch, mem, cache_dir, "inmem", inmem_b))
            else:
                bound_b = (
                    gpu_stream_b
                    if (effective_gpu and gpu_stream_b is not None) else stream_b
                )
                cell_tasks.append((cell, patch, mem, cache_dir, "stream", bound_b))

        if effective_gpu and gpu_streaming_host_peak_bytes is not None:
            per_cell_footprint = int(gpu_streaming_host_peak_bytes)
        else:
            per_cell_footprint = max(
                (t[5] for t in cell_tasks if t[4] is not None), default=0
            )

        cpu_fallback_info = {
            "applicable": bool(effective_gpu), "feasible": False, "mode": None,
            "max_bound_bytes": None, "ram_budget_bytes": int(budget_bytes),
            "tile": int(chosen_tile),
            "cells_in_flight": None,
            "cache_workers_per_cell": None, "reason": None, "used": False,
            "raw_available_bytes": int(raw_available_bytes),
            "plan_budget_bytes": int(budget_bytes),
        }
        cpu_fallback_tasks = None
        if effective_gpu:
            if cpu_fallback_plan is None:
                # R36 F-M2 (D): the GPU is still allowed to run when no CPU mode
                # fits — recorded honestly as UNAVAILABLE (never a fake tile).
                cpu_fallback_info["feasible"] = False
                cpu_fallback_info["tile"] = None
                cpu_fallback_info["reason"] = (
                    None if cpu_refusal is None else str(cpu_refusal)
                )
            else:
                _fb_mode = cpu_fallback_plan["mode"]
                # An in-memory fallback does NOT use a streaming tile.
                cpu_fallback_info["tile"] = (
                    int(chosen_tile) if _fb_mode == "stream" else None
                )
                cpu_fallback_tasks = []
                for (cell, patch, mem, cache_dir, inmem_b, stream_b, _gpu_b) in cell_specs:
                    if inmem_b is None:
                        cpu_fallback_tasks.append((cell, patch, mem, cache_dir, None, 0))
                    elif _fb_mode == "inmem":
                        cpu_fallback_tasks.append((cell, patch, mem, cache_dir, "inmem", inmem_b))
                    else:
                        cpu_fallback_tasks.append((cell, patch, mem, cache_dir, "stream", stream_b))
                _fb_bound = max(
                    (c["max_bound_bytes"] for c in cpu_fallback_plan["candidates"]
                     if c["mode"] == _fb_mode), default=0,
                )
                cpu_fallback_info.update({
                    "feasible": True, "mode": _fb_mode,
                    "max_bound_bytes": int(_fb_bound),
                    "cells_in_flight": int(cpu_fallback_plan["cells_in_flight"]),
                    "cache_workers_per_cell": int(cpu_fallback_plan["cache_workers_per_cell"]),
                    "reason": cpu_fallback_plan.get("chosen_reason"),
                })

        envelope = rcontract.PlanEnvelope(
            phase=rcontract.PHASE_PLAN,
            raw_available_bytes=int(raw_available_bytes),
            factor_num=rcontract.PLAN_FACTOR_NUM,
            factor_den=rcontract.PLAN_FACTOR_DEN,
            budget_bytes=int(budget_bytes),
            layout=(nx, ny),
            refinement_rank=int(nx * ny),
            cpu_tile=int(cpu_streaming_tile),
            gpu_tile=gpu_streaming_tile,
            backend=(rcontract.BACKEND_GPU if effective_gpu else rcontract.BACKEND_CPU),
            cells_in_flight=int(cells_in_flight),
            cache_workers_per_cell=int(cache_workers_per_cell),
            shared_bytes=int(shared_bytes),
            stack_total_bytes=int(plan["stack_total_bytes"]),
            cache_total_bytes=int(plan["cache_total_bytes"]),
            aggregate_bytes=int(plan["aggregate_bytes"]),
            pinned=bool(pinned_layout is not None),
            reason=str(plan.get("chosen_reason") or ""),
        )
        rcontract.validate_envelope(envelope)
        if int(envelope.aggregate_bytes) > int(budget_bytes):
            raise zpar.ModeInfeasible(
                f"built bundle aggregate {envelope.aggregate_bytes} exceeds its PLAN "
                f"budget {budget_bytes}"
            )

        return ExecutionBundle(
            layout=layout,
            cell_ctxs=cell_ctxs,
            footprints=footprints,
            cpu_streaming=dict(cpu_streaming),
            cpu_streaming_tile=int(cpu_streaming_tile),
            gpu_tile_size=(int(gpu_tile_size) if effective_gpu else None),
            gpu_streaming_tile=gpu_streaming_tile,
            gpu_streaming_host_peak_bytes=gpu_streaming_host_peak_bytes,
            gpu_used=bool(effective_gpu),
            gpu_pre_execution_fallback=(
                gpu_pre_execution_fallback or gpu_fallback_reason
            ),
            cell_specs=cell_specs,
            cell_tasks=cell_tasks,
            plan=plan,
            chosen_mode=chosen_mode,
            cells_in_flight=cells_in_flight,
            cache_workers_per_cell=cache_workers_per_cell,
            per_cell_footprint=int(per_cell_footprint),
            cpu_fallback_info=cpu_fallback_info,
            cpu_fallback_plan=cpu_fallback_plan,
            cpu_fallback_tasks=cpu_fallback_tasks,
            envelope=envelope,
            shared_bytes=int(shared_bytes),
            cache_worker_footprint_bytes=int(cache_worker_footprint_bytes),
            worst_patch_px=int(worst_patch_px),
            gauge_workers=int(_gauge_workers),
            cache_build_workers=int(_cache_workers),
            worker_budget_bytes=int(worker_budget),
            attempts=attempts,
        )


def _build_execution_bundle_infeasible_message(exc) -> str:
    """Quantitative, FINER-direction refusal text (never coarser advice)."""
    return (
        "ZeGrid refused BEFORE any gauge/stack work: no executable bundle fits the "
        f"PLAN budget. {exc}"
    )


def _ram_contract_diagnostics(outcome, bundle, initial_bundle) -> dict:
    """JSON-safe R41-B manifest/log trail of the 80 %-plan / 90 %-admit contract.

    Records BOTH raw observations and their OWN factors/budgets (never a
    serialized float factor, never ``0.9 * B_plan``), the initial and final
    layouts, the admitted envelope, the replan count and the tolerance flag.
    """
    plan_env = bundle.envelope
    return {
        "schema": R41_RAM_CONTRACT_SCHEMA,
        "plan_factor": {
            "num": int(rcontract.PLAN_FACTOR_NUM),
            "den": int(rcontract.PLAN_FACTOR_DEN),
        },
        "admit_factor": {
            "num": int(rcontract.ADMIT_FACTOR_NUM),
            "den": int(rcontract.ADMIT_FACTOR_DEN),
        },
        "plan_observation": {
            "raw_available_bytes": int(plan_env.raw_available_bytes),
            "factor_num": int(plan_env.factor_num),
            "factor_den": int(plan_env.factor_den),
            "budget_bytes": int(plan_env.budget_bytes),
        },
        "admission_observations": [o.to_dict() for o in outcome.observations],
        "admission_history": [p.to_dict() for p in outcome.history],
        "launch_observation": outcome.final_observation.to_dict(),
        "replans": int(outcome.replans),
        "tolerance": bool(outcome.tolerance),
        "initial_layout": [
            int(initial_bundle.envelope.layout[0]),
            int(initial_bundle.envelope.layout[1]),
        ],
        "final_layout": [int(plan_env.layout[0]), int(plan_env.layout[1])],
        "final_envelope": plan_env.to_dict(),
        "shared_bytes": int(plan_env.shared_bytes),
        "stack_total_bytes": int(plan_env.stack_total_bytes),
        "cache_total_bytes": int(plan_env.cache_total_bytes),
        "aggregate_bytes": int(plan_env.aggregate_bytes),
        "cache_worker_footprint_bytes": int(bundle.cache_worker_footprint_bytes),
        "worst_patch_px": int(bundle.worst_patch_px),
        "worker_budget_bytes": int(bundle.worker_budget_bytes),
        "gauge_workers": int(bundle.gauge_workers),
        "cache_build_workers": int(bundle.cache_build_workers),
        "backend": str(plan_env.backend),
        "mode": str(bundle.chosen_mode),
        "cpu_streaming_tile": int(bundle.cpu_streaming_tile),
        "gpu_streaming_tile": (
            int(bundle.gpu_streaming_tile)
            if bundle.gpu_streaming_tile is not None else None
        ),
        "gpu_pre_execution_fallback": bundle.gpu_pre_execution_fallback,
        "refinement_attempts": list(bundle.attempts),
        "note": (
            "B_plan = floor(4/5 * raw plan observation); B_admit = "
            "floor(9/10 * a FRESH raw observation taken before every admission "
            "attempt and again immediately before launch. Each factor is applied "
            "ONCE to its own raw observation. The 90 % ceiling is admission-only: "
            "it never enlarges a layout, a tile or the concurrency. Host RAM only; "
            "the VRAM reserve policy is unchanged."
        ),
    }


def _r55_observe_error_block(exc, *, gpu_readiness=None) -> dict:
    """Honest fallback when observe construction fails (never aborts the run).

    Carries the closed ``R55_INTERNAL_ERROR`` code; no metric or plan is
    fabricated and execution stays on the legacy path. The additive GPU
    readiness block, when supplied, is recorded truthfully (including an
    unavailable/error state).
    """
    block = {
        "schema": zsched.MANIFEST_SCHEMA,
        "scheduler_version": zsched.SCHEDULER_VERSION,
        "mode": "observe_only",
        "requested_mode": zsched.MODE_OBSERVE,
        "status": "error",
        "error": repr(exc),
        "reason_code": renv.REASON_INTERNAL_ERROR,
        "note": (
            "observe-only plan construction failed; execution is unchanged. "
            "No observed metric is fabricated and no RUN decision was taken."
        ),
    }
    if gpu_readiness is not None:
        block["gpu_readiness"] = gpu_readiness
    return block


def _r55_gpu_readiness_observe_block(
    gpu_context, gpu_used, readiness=None
) -> dict:
    """Additive R55-C readiness block; observe-only, never controls execution.

    ``requested`` comes from the product GPU preference, ``effective`` from the
    actually executed backend, and ``available`` from ONE explicit readiness
    probe (which delegates to the canonical availability definition). The three
    are recorded separately and are never conflated.
    """
    ctx = gpu_context or {}
    snapshot = (
        zgr.probe_gpu_readiness_safe() if readiness is None else readiness
    )
    return zgr.readiness_manifest_dict(
        snapshot,
        requested_backend="gpu" if bool(ctx.get("requested")) else "cpu",
        effective_backend="gpu" if bool(gpu_used) else "cpu",
    )


def _r55_reconcile_gpu_readiness(block, gpu_ctx) -> dict:
    """Reconcile the additive R55-C provenance with the ACTUAL final backend.

    The observe block is built before stack execution, when only the selected
    backend is known. A classified runtime GPU failure can later perform an exact
    whole-batch CPU rerun. Immediately before any manifest/output write this
    helper copies only the nested ``gpu_readiness`` chain and updates
    ``product.effective_backend`` to the backend that actually produced the
    successful output (``gpu`` only when ``actually_used`` AND ``final_backend``
    prove GPU execution, otherwise ``cpu``).

    ``requested_backend`` is never changed. The helper never raises and never
    fabricates a missing block, so legacy mode and malformed shapes are preserved.
    """
    try:
        if not isinstance(block, dict):
            return block
        gpu_readiness = block.get("gpu_readiness")
        if not isinstance(gpu_readiness, dict):
            return block
        product = gpu_readiness.get("product")
        if not isinstance(product, dict):
            return block
        ctx = gpu_ctx if isinstance(gpu_ctx, dict) else {}
        executed_gpu = bool(
            ctx.get("actually_used") is True and ctx.get("final_backend") == "gpu"
        )
        effective = "gpu" if executed_gpu else "cpu"
        if product.get("effective_backend") == effective:
            return block
        new_block = dict(block)
        new_gpu = dict(gpu_readiness)
        new_product = dict(product)
        new_product["effective_backend"] = effective
        new_gpu["product"] = new_product
        new_block["gpu_readiness"] = new_gpu
        return new_block
    except Exception:
        # Provenance reconciliation must never abort a successful run.
        return block


def _build_r55_observe_block(
    *,
    bundle,
    chosen_mode,
    cell_ctxs,
    canvas,
    halo_px,
    output_dir,
    cache_root=None,
    frame_ids,
    scheduler_mode="observe",
    gpu_readiness=None,
    requested_backend=None,
    effective_backend=None,
):
    """Build the ADDITIVE R55 observe-only manifest block (ZM-ZEGRID-R55-B).

    Observe mode records what R55 WOULD choose; it never controls execution. The
    numbers below are the product's OWN declared bounds (per-cell streaming
    footprint, phase worker footprints, canvas-sized assembly/export buffers) —
    no empirical coefficient is invented and no observed metric is fabricated.

    Returns ``None`` in ``legacy`` mode. Raises :class:`scheduler.PlanRefused` when
    no candidate fits; the caller records the refusal and keeps executing legacy.
    """
    if scheduler_mode == zsched.MODE_LEGACY:
        return None
    if scheduler_mode != zsched.MODE_OBSERVE:
        # Validated at the product entry; never silently coerce here.
        raise zsched.InvalidSchedulerMode(
            f"unknown zegrid_scheduler_mode {scheduler_mode!r}"
        )

    active = [
        (cell, patch, mem)
        for (_row, _col, cell, patch, mem) in cell_ctxs
        if mem.patch_ids
    ]
    if not active:
        raise ValueError("R55 observe plan requires at least one non-empty cell")

    units_total = len(active)
    worst_patch_px = 0
    worst_patch_hw = (1, 1)
    core_area_px = 0
    patch_area_px = 0
    max_n_frames = 1
    for cell, patch, mem in active:
        patch_area = int(patch.patch.width) * int(patch.patch.height)
        core = cell.core
        core_area = max(0, int(core.x1) - int(core.x0)) * max(
            0, int(core.y1) - int(core.y0)
        )
        if patch_area > worst_patch_px:
            worst_patch_px = patch_area
            worst_patch_hw = (int(patch.patch.height), int(patch.patch.width))
        patch_area_px += patch_area
        core_area_px += core_area
        max_n_frames = max(max_n_frames, len(mem.patch_ids))

    phase = (
        renv.PHASE_CELL_STACK_INMEM
        if str(chosen_mode) == "inmem"
        else renv.PHASE_CELL_STACK_STREAM
    )
    channels = 3
    # Aligned-cache static bound: one float32 plane per channel per frame at the
    # WORST patch size. This is a dtype decomposition, not an empirical constant.
    if phase == renv.PHASE_CELL_STACK_STREAM:
        per_unit_cache = int(worst_patch_px) * channels * 4 * int(max_n_frames)
        per_unit_prefetch = int(per_unit_cache)
    else:
        per_unit_cache = 0
        per_unit_prefetch = 0

    per_unit_stack = int(bundle.per_cell_footprint)
    if per_unit_stack <= 0:
        per_unit_stack = max(
            (int(t[5]) for t in bundle.cell_tasks if t[4] is not None), default=1
        )

    # The aligned cache lives under the resolved temporary work root, which may
    # be a DIFFERENT mount from the output directory. D3 must observe the volume
    # that actually holds the cache (fall back to the output dir only when the
    # caller did not supply one).
    _disk_root = Path(cache_root) if cache_root is not None else Path(output_dir)
    try:
        disk_free = int(shutil.disk_usage(str(_disk_root)).free)
    except Exception:
        disk_free = 0

    canvas_px = int(canvas.width) * int(canvas.height)
    gauge_workers = max(1, int(bundle.gauge_workers))
    cache_workers = max(1, int(bundle.cache_build_workers))
    gauge_worker_bytes = int(zpar.gauge_footprint_bytes(canvas_px)) * gauge_workers
    cache_worker_bytes = int(zpar.cache_footprint_bytes(worst_patch_px)) * cache_workers

    # Assembly is retained cell cores + the halo band, never a blind per-phase add.
    bytes_per_px = 4 * 4  # science RGB float32 + stack_depth float32
    halo_band_bytes = max(0, int(patch_area_px) - int(core_area_px)) * bytes_per_px
    retained_core_bytes = int(canvas_px) * bytes_per_px

    requests = (
        zsched.PhaseRequest(
            phase=renv.PHASE_GAUGE,
            components={"gauge_worker_footprint_bytes": gauge_worker_bytes},
            units=gauge_workers,
            units_in_flight=gauge_workers,
            workers_per_unit=1,
        ),
        zsched.PhaseRequest(
            phase=renv.PHASE_CACHE_BUILD,
            components={"cache_build_worker_footprint_bytes": cache_worker_bytes},
            units=cache_workers,
            units_in_flight=cache_workers,
            workers_per_unit=1,
            disk_bytes=0,
        ),
        zsched.PhaseRequest(
            phase=renv.PHASE_ASSEMBLY,
            components={
                "halo_band_bytes": halo_band_bytes,
                "retained_core_bytes": retained_core_bytes,
            },
            units=1,
            units_in_flight=1,
            workers_per_unit=1,
            # D1: assembly's retained cores + halo coexist with the ACTIVE stack
            # phase (stream or inmem). Exactly one stack phase exists.
            overlaps_with=(phase,),
        ),
        zsched.PhaseRequest(
            phase=renv.PHASE_EXPORT,
            components={
                "coverage_plane_bytes": canvas_px * 4,
                "science_planes_bytes": canvas_px * channels * 4,
            },
            units=1,
            units_in_flight=1,
            workers_per_unit=1,
        ),
    )

    envelope = bundle.envelope
    plan = zsched.build_observe_plan(
        phase=phase,
        backend=renv.BACKEND_CPU,
        budget_bytes=int(envelope.budget_bytes),
        raw_available_bytes=int(envelope.raw_available_bytes),
        factor_num=int(envelope.factor_num),
        factor_den=int(envelope.factor_den),
        cpu_logical=int(os.cpu_count() or 1),
        units_total=int(units_total),
        shared_bytes=int(bundle.shared_bytes),
        per_unit_stack_bytes=int(per_unit_stack),
        per_unit_cache_bytes=int(per_unit_cache),
        per_unit_prefetch_bytes=int(per_unit_prefetch),
        disk_free_bytes=int(disk_free),
        layout=(int(bundle.layout["nx"]), int(bundle.layout["ny"])),
        refinement_rank=int(envelope.refinement_rank),
        patch_hw=worst_patch_hw,
        halo_px=int(halo_px),
        channels=channels,
        n_frames=int(max_n_frames),
        phase_requests=requests,
        cell_ids=tuple(cell.cell_id for cell, _p, _m in active),
        frame_ids=tuple(frame_ids or ()),
        mode=zsched.MODE_OBSERVE,
        run_id=str(canvas.canvas_id),
        observation_source="plan.available_bytes",
        disk_fraction_num=1,
        disk_fraction_den=2,
        margin_num=renv.DEFAULT_MARGIN_NUM,
        margin_den=renv.DEFAULT_MARGIN_DEN,
    )
    return plan.to_manifest_dict(
        include_gpu_readiness=gpu_readiness is not None,
        gpu_readiness=gpu_readiness,
        requested_backend=requested_backend,
        effective_backend=effective_backend,
    )


def _run_single(
    frames_info,
    input_folder,
    output_dir,
    *,
    progress_callback,
    science_config,
    zconfig,
    pinned_layout=None,
    sip_mode="keep",
    workers=None,
    ignored_run_args=None,
    finishing_config=None,
    gpu_context=None,
    pass_label="single",
    r37_diagnostic_capture=False,
    r37_diagnostic_holdout_cap=None,
    r40_seam_capture=False,
    r40_output_root=None,
    r40_seam_capture_disk_budget_bytes=None,
    halo_px=ZEGRID_HALO_DEFAULT_PX,
    halo_provenance=None,
    scheduler_mode=zsched.MODE_LEGACY,
):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    # ZM-ZEGRID-R53: ONE validated effective halo width for this run. The product
    # entry (``run_zegrid_mode``) resolves and validates it BEFORE any expensive
    # work and passes it explicitly; this defensive re-validation never changes a
    # valid value and keeps direct/internal calls on the same strict contract.
    try:
        halo_px = resolve_zegrid_halo_px(halo_px)
    except ZeGridHaloError as exc:
        _emit(f"ZeGrid: {exc}", lvl="ERROR", callback=progress_callback)
        raise
    if halo_provenance is not None and int(
        halo_provenance["effective_px"]
    ) != int(halo_px):
        raise RuntimeError(
            "ZeGrid halo provenance contradiction: run-level effective "
            f"{halo_provenance['effective_px']} != pass halo {halo_px}"
        )
    # ZM-ZEGRID-R37-D (F1): when a capture is requested, fail-closed invalidate the
    # fixed target in THIS run's actual output dir IMMEDIATELY at run start (before
    # descriptors/layout/gauge). Idempotent with the outer invalidation; protects
    # direct/internal calls and every pre-assembly failure phase (incl. auto->global).
    if r37_diagnostic_capture:
        from .core.zegrid import background_diagnostics as _bgd

        _bgd.invalidate_capture_target(output_dir)
    timings = zin.Timings()
    start_ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    # ZM-ETA-SERVICE-R30 F4: honest actual wall duration for the history record
    # (explicit perf_counter start; measured at the end, NOT a phase-sum guess).
    run_wall_start = time.perf_counter()

    # ZM-ZEGRID-R14: live observability — phase START/END lines, bounded
    # intra-phase progress, live ETA, crash-breadcrumb stage, and an
    # incrementally-flushed run log (readable DURING the run, not only at the end).
    _open_run_log(output_dir, start_ts)
    global_eta = zobs.GlobalEta(_PHASE_ORDER)

    def _emit_live(msg, lvl="INFO"):
        _emit(msg, lvl=lvl, callback=progress_callback)

    def _stage_live(stage_str, current, total):
        # Worker crash-breadcrumb `stage` (3-int progress_callback form).
        if progress_callback is None:
            return
        try:
            progress_callback(stage_str, int(current), int(total))
        except Exception:
            pass

    def _log_line(line):
        _append_run_log_line(output_dir, line)

    def _reporter():
        return zobs.PhaseReporter(
            _emit_live,
            stage=_stage_live,
            log_line=_log_line,
            global_eta=global_eta.estimate,
        )

    # ZM-ZEGRID-R22: explicit GPU-usage + ignored-settings surfacing (nothing
    # silently ignored). ``gpu_used`` is the EFFECTIVE GPU backend (never claimed
    # from CuPy initialisation alone); ``gpu_context`` carries the requested /
    # available / effective / device / VRAM truth + fallback reason.
    gpu_ctx = gpu_context or {}
    gpu_used = bool(gpu_ctx.get("effective"))
    ignored_settings = zin.ignored_settings_present(zconfig, gpu_honoured=gpu_used)
    if ignored_settings:
        for line in zin.ignored_settings_warning_lines(ignored_settings):
            _emit(line, lvl="WARN", callback=progress_callback)
    if gpu_used:
        _emit(
            f"ZeGrid: GPU backend ENABLED — device={gpu_ctx.get('device')} "
            f"vram_total={gpu_ctx.get('vram_total_bytes')} "
            f"vram_free={gpu_ctx.get('vram_free_bytes')} "
            f"budget={gpu_ctx.get('vram_budget_bytes')}",
            callback=progress_callback,
        )
    else:
        _emit(
            f"ZeGrid: CPU backend (gpu requested={gpu_ctx.get('requested')} "
            f"available={gpu_ctx.get('available')} "
            f"reason={gpu_ctx.get('fallback_reason')})",
            callback=progress_callback,
        )

    _emit(f"ZeGrid: setup — {len(frames_info)} frame(s) -> {output_dir}", callback=progress_callback)
    _setup_rep = _reporter()
    _setup_rep.start("setup", total=len(frames_info), unit="frames")
    with timings.timed("setup"):
        descs, rejected = _build_frame_descriptors(
            frames_info, input_folder, progress_callback, sip_mode=sip_mode
        )
        canvas = zg.build_canvas(descs)
    # ZM-ZEGRID-R35A: explicit inventory invariant (no silent drop).
    _cand = len(frames_info)
    _inc = len(descs)
    _rej = len(rejected)
    _extra = _cand - _inc - _rej
    _msg = (
        f"ZeGrid inventory (pass={pass_label}): candidates={_cand} "
        f"included={_inc} rejected_wcs={_rej}"
    )
    if _extra:
        _msg += f" (unaccounted={_extra})"
    _emit(_msg, callback=progress_callback)
    _setup_rep.end(
        throughput=_fmt_throughput(len(frames_info), timings.get("setup"), "frames/s")
    )
    global_eta.phase_completed("setup", timings.get("setup"))
    _emit(
        f"ZeGrid: canvas {canvas.width}x{canvas.height} "
        f"(resolution {canvas.resolution_deg:.3g} deg/px)",
        callback=progress_callback,
    )

    # ZM-ZEGRID-R37-D (M2): when the explicit capture is requested, generate ONE
    # stable opaque capture identity for THIS pass (single / EQ / ALTZ distinct)
    # BEFORE per-Cell stacking, and record it in the run log. The synthetic digest
    # hashes the canonical ordered RELATIVE logical frame identifiers (never
    # absolute private paths). Lazy import: the disabled path imports nothing.
    r37_capture_identity = None
    if r37_diagnostic_capture:
        from .core.zegrid import background_diagnostics as _bgd

        _ids = [d.frame_id.logical_path for d in descs]
        _src_digest = _bgd.source_set_digest(_ids)
        # F3: one high-resolution invocation nonce per pass (provenance only, never
        # scientific RNG) so same-second retries get DISTINCT capture ids.
        _nonce = time.time_ns()
        _capture_id = _bgd.capture_run_id(
            start_ts=start_ts, pass_label=pass_label,
            n_frames=len(descs), source_set_digest_value=_src_digest,
            nonce=_nonce,
        )
        r37_capture_identity = {
            "capture_id": _capture_id,
            "run_label": str(pass_label),
            "n_source_frames": int(len(descs)),
            "source_set_digest": _src_digest,
            "source_set_digest_algo": _bgd.SOURCE_SET_DIGEST_ALGO,
            "invocation_nonce_algo": "time.time_ns",
            "invocation_nonce": int(_nonce),
        }
        _log_line(
            "ZeGrid R37-D capture identity: "
            + json.dumps(r37_capture_identity, sort_keys=True)
        )
        _emit(
            f"ZeGrid R37-D capture requested — capture_id={_capture_id} "
            f"pass={pass_label} source_frames={len(descs)} "
            f"source_set_digest={_src_digest}",
            callback=progress_callback,
        )

    # ZM-ZEGRID-R40-B2-B: ONE stable opaque capture identity for THIS pass when the
    # R40 seam capture is explicitly requested (provenance-only nonce; never
    # scientific RNG). The disabled path imports/invents nothing.
    r40_capture_id = None
    r40_capture_source_digest = None
    if r40_seam_capture:
        from .core.zegrid import background_diagnostics as _bgd

        _r40_ids = [d.frame_id.logical_path for d in descs]
        r40_capture_source_digest = _bgd.source_set_digest(_r40_ids)
        r40_capture_id = _r40_capture_id(
            start_ts=start_ts, pass_label=pass_label, n_frames=len(descs),
            source_set_digest_value=r40_capture_source_digest, nonce=time.time_ns(),
        )
        _emit(
            f"ZeGrid R40-B2-B seam capture requested — capture_id={r40_capture_id} "
            f"pass={pass_label} source_frames={len(descs)} "
            f"source_set_digest={r40_capture_source_digest}",
            callback=progress_callback,
        )

    # ------------------------------------------------------------------
    # ZM-ZEGRID-R41-B — PLAN PHASE. Observe the RAW available RAM and derive the
    # EXACT plan budget ONCE with the shared contract (integer arithmetic only,
    # never a float factor). A nonpositive observation fails CLOSED here, before
    # any gauge/cache work.
    # ------------------------------------------------------------------
    plan_raw_bytes = int(available_memory_bytes())
    try:
        plan_budget_bytes = int(rcontract.plan_budget_bytes(plan_raw_bytes))
    except rcontract.InvalidObservation as _plan_exc:
        raise RuntimeError(
            "ZeGrid refusing to plan: the RAW available-RAM observation "
            f"({plan_raw_bytes} bytes) is nonpositive/unusable ({_plan_exc}); a real "
            "positive observation is required before any layout or stack work "
            "(fail closed, no unbounded fallback)."
        ) from _plan_exc
    available = int(plan_raw_bytes)
    _emit(
        f"ZeGrid: RAM plan observation — raw available {plan_raw_bytes / 2**20:.1f} MiB "
        f"-> B_plan {plan_budget_bytes / 2**20:.1f} MiB "
        f"(factor {rcontract.PLAN_FACTOR_NUM}/{rcontract.PLAN_FACTOR_DEN}); the admission "
        f"ceiling is re-measured at {rcontract.ADMIT_FACTOR_NUM}/"
        f"{rcontract.ADMIT_FACTOR_DEN} right before the cell workers start",
        callback=progress_callback,
    )

    # ZM-ZEGRID-R35A: large disk-backed intermediates follow the user's
    # Temporary Working Directory (empty -> the output folder, historical
    # behaviour). Scientific outputs STAY under output_dir. Resolved BEFORE the
    # plan so the bundle carries the REAL per-cell cache directories.
    from .zemosaic_utils import resolve_temporary_work_dir as _resolve_temp_root

    _temp_root = _resolve_temp_root(
        output_dir, getattr(zconfig, "temporary_work_dir", "") if zconfig is not None else ""
    )
    cache_root = Path(_temp_root) / CACHE_DIR_NAME
    cache_root.mkdir(parents=True, exist_ok=True)
    if str(Path(_temp_root)) != str(Path(output_dir)):
        _emit(
            f"ZeGrid: temporary work root = {_temp_root} (outputs stay in {output_dir})",
            callback=progress_callback,
        )

    # LAYOUT + COMPLETE EXECUTABLE BUNDLE, built under B_plan BEFORE the gauge.
    # The bundle is the SINGLE source of truth downstream (layout, memberships,
    # CPU/GPU tiles, per-cell bounds, joint mode/concurrency/cache workers, tasks,
    # CPU fallback bundle and validated PlanEnvelope). No second independent plan
    # is computed later.
    _layout_rep = _reporter()
    _layout_rep.start("layout")
    # ZM-ZEGRID-R27: layout sub-step observability — a bounded, throttled reporter
    # for the stable layout sub-steps (footprints, candidate scan, chosen cells).
    # It emits ONLY log/durable-detail events (never GUI stage/percent).
    layout_substep = zobs.SubstepReporter(_emit_live, log_line=_log_line)
    _bundle_common = dict(
        cpu_count=os.cpu_count(),
        gpu_used=bool(gpu_used),
        vram_budget_bytes=(gpu_ctx or {}).get("vram_budget_bytes"),
        cache_root=cache_root,
        halo_px=int(halo_px),
        emit=_emit_live,
        log_line=_log_line,
    )
    try:
        with timings.timed("layout"):
            _initial_bundle = _build_execution_bundle(
                canvas, descs, None,
                raw_available_bytes=plan_raw_bytes, budget_bytes=plan_budget_bytes,
                pinned_layout=pinned_layout,
                gauge_workers=workers,
                cache_build_workers=(workers if workers is not None else None),
                layout_substep=layout_substep,
                **_bundle_common,
            )
    except (zal.LayoutInfeasible, zpar.ModeInfeasible) as _plan_exc:
        _emit(
            "ZeGrid: NO executable plan fits the PLAN budget — refusing BEFORE the "
            f"gauge/stack: {_plan_exc}",
            lvl="ERROR", callback=progress_callback,
        )
        raise

    layout = _initial_bundle.layout
    footprints = _initial_bundle.footprints
    cell_ctxs = _initial_bundle.cell_ctxs
    nx, ny = int(layout["nx"]), int(layout["ny"])
    cpu_streaming = _initial_bundle.cpu_streaming
    _cs_selected = cpu_streaming.get("selected_tile_size")
    cpu_streaming_tile = int(_initial_bundle.cpu_streaming_tile)
    workers = int(_initial_bundle.gauge_workers)
    cache_build_workers = int(_initial_bundle.cache_build_workers)
    _emit(
        f"ZeGrid: CPU streaming tile — selected={_cs_selected} "
        f"(preferred={cpu_streaming.get('preferred_tile_size')}, "
        f"minimum={cpu_streaming.get('minimum_tile_size')}) "
        f"predicted={cpu_streaming.get('predicted_peak_bytes', 0) / 2**20:.1f} MiB "
        f"budget={(cpu_streaming.get('ram_budget_bytes') or 0) / 2**20:.1f} MiB — "
        f"{cpu_streaming.get('selection_reason')}",
        callback=progress_callback,
    )
    # Record the layout sub-step timings as SUBORDINATE to the layout phase (they
    # appear in the manifest/run-log but never become new top-level phases for the
    # global ETA).
    for sub_id, sec in layout_substep.timings().items():
        timings.add(sub_id, sec, subordinate=True)
    _layout_rep.end(
        throughput=_fmt_throughput(nx * ny, timings.get("layout"), "cells/s")
    )
    global_eta.phase_completed("layout", timings.get("layout"))
    _emit(
        f"ZeGrid: layout {nx}x{ny} ({layout['cell_count']} cells, "
        f"source={layout.get('layout_source', 'auto')}) from RAM plan budget "
        f"{plan_budget_bytes / 2**30:.2f} GiB; predicted bound "
        f"{layout['predicted_bound_bytes'] / 2**20:.1f} MiB (maxN={layout['max_contributors']}); "
        f"CPU streaming tile={_cs_selected}",
        callback=progress_callback,
    )
    for w in layout["warnings"]:
        _emit(f"ZeGrid: layout warning — {w}", lvl="WARN", callback=progress_callback)

    canvas_px = int(canvas.width) * int(canvas.height)
    _gauge_fp = zpar.gauge_footprint_bytes(canvas_px)
    _cache_fp = zpar.cache_footprint_bytes(
        _initial_bundle.worst_patch_px or canvas_px
    )
    _emit(
        f"ZeGrid: parallel workers — gauge={workers} "
        f"(CPU={os.cpu_count()}, raw={plan_raw_bytes / 2**30:.2f}GiB, "
        f"B_plan={plan_budget_bytes / 2**30:.2f}GiB, footprint={_gauge_fp / 2**20:.0f}MiB), "
        f"cache_build={cache_build_workers} (footprint={_cache_fp / 2**20:.0f}MiB)",
        callback=progress_callback,
    )

    # R15 (rework-1 L3): track cache cleanup failures (auditable in the manifest).
    cleanup_failures: list[str] = []

    # GLOBAL PHOTOMETRIC GAUGE (ZM-ZEGRID-R11): one reference + per-frame
    # normalization coefficients/weights computed ONCE over the full canvas,
    # so every Cell shares a single photometric anchor. Frame-major full-canvas
    # reprojection (parallelised); the disk cache is DELETED after use (bounded
    # disk — only the in-memory gauge coefficients are needed downstream).
    _emit(
        f"ZeGrid: computing global photometric gauge (full-canvas, frame-major, workers={workers})",
        callback=progress_callback,
    )
    gauge_cache_dir = cache_root / "__gauge__"
    _gauge_rep = _reporter()
    # rework-1 (M1): the gauge phase spans TWO sub-passes (counts N + pairs N-1),
    # so the honest phase total is 2N-1 and progress is CUMULATIVE across them.
    _gauge_rep.start("gauge", total=2 * len(descs) - 1, unit="frame-ops")

    def _gauge_progress(done, total, item_id):
        _gauge_rep.progress(done, item_id=item_id, total=total)

    gauge_diagnostics: dict = {}
    with timings.timed("gauge"):
        global_gauge, global_frame_ids = zphot.compute_global_gauge(
            descs, canvas, _gauge_decode, science_config, gauge_cache_dir, workers=workers,
            progress_callback=_gauge_progress,
            emit=_emit_live,
            diagnostics=gauge_diagnostics,
        )
    _gauge_rep.end(
        throughput=_fmt_throughput(len(descs), timings.get("gauge"), "frames/s")
    )
    global_eta.phase_completed("gauge", timings.get("gauge"))
    global_reference_frame_id = global_frame_ids[int(global_gauge.reference_index)]
    _emit(
        f"ZeGrid: global photometric gauge — reference frame {global_reference_frame_id!r} "
        f"(index {global_gauge.reference_index} of {len(global_frame_ids)} frames); "
        f"exclusions={len(global_gauge.exclusions)}",
        callback=progress_callback,
    )

    # ------------------------------------------------------------------
    # ZM-ZEGRID-R41-B — ADMISSION PHASE. A FRESH raw observation is taken for
    # EVERY admission attempt at the 90 % ceiling, immediately before the cell
    # workers. On a miss the bundle is REBUILT under the SAME observation's
    # exact 80 % B_plan (never 0.9 * B_plan, never a stale raw), bounded to
    # never coarsen the layout, grow the CPU/GPU tile or grow the concurrency.
    # ------------------------------------------------------------------
    requested_cell_gauge_mode = getattr(science_config, "cell_gauge_mode", None)

    def _admit(initial_bundle, *, fix_rank=None, backend_gpu=None,
               pinned_override=None):
        """Run ONE bounded admission cycle; return ``(outcome, final_bundle)``.

        ``fix_rank`` pins the EXACT geometry (cell-count rank) — used by the
        GPU->CPU runtime fallback so a started mosaic is never re-sliced.
        ``pinned_override`` pins the EXACT ``(nx, ny)`` tuple for every build in
        the cycle (a same rank is NOT proof of the same geometry).
        ``backend_gpu`` overrides the requested backend for the whole cycle.
        """
        _pin = pinned_layout if pinned_override is None else tuple(pinned_override)
        state = {"last_raw": None, "bundle": None, "attempts": []}

        def _measure():
            # The measurement-recorder closure: the admission controller calls
            # this before every attempt, so ``state["last_raw"]`` is EXACTLY the
            # raw that triggered the (possible) miss. The rebuild callback reuses
            # it verbatim instead of inverting B_plan back into a guessed raw.
            raw = int(available_memory_bytes())
            state["last_raw"] = raw
            return raw

        def _rebuild(b_plan, previous):
            raw = state["last_raw"]
            kw = dict(_bundle_common)
            if backend_gpu is not None:
                kw["gpu_used"] = bool(backend_gpu)
            bundle = _build_execution_bundle(
                canvas, descs, footprints,
                raw_available_bytes=int(raw), budget_bytes=int(b_plan),
                pinned_layout=_pin,
                min_refinement_rank=int(previous.refinement_rank),
                max_refinement_rank=(None if fix_rank is None else int(fix_rank)),
                max_cpu_tile=int(previous.cpu_tile),
                max_gpu_tile=(
                    int(previous.gpu_tile) if previous.gpu_tile is not None else None
                ),
                max_cells_in_flight=int(previous.cells_in_flight),
                gauge_workers=int(workers),
                layout_substep=None,
                **kw,
            )
            state["bundle"] = bundle
            state["attempts"].append(bundle)
            return bundle.envelope

        outcome = rcontract.AdmissionController(
            max_replans=R41_ADMISSION_MAX_REPLANS
        ).run(initial_bundle.envelope, _measure, _rebuild)
        final = initial_bundle if outcome.replans == 0 else state["bundle"]
        if final is None:  # defensive: a replan always records its bundle
            final = initial_bundle
        if final.envelope != outcome.plan:
            raise RuntimeError(
                "internal R41-B inconsistency: the accepted envelope is not the "
                "final bundle's envelope"
            )
        return outcome, final

    try:
        _adm, _final = _admit(_initial_bundle)
    except rcontract.AdmissionExhausted as _adm_exc:
        _emit(
            "ZeGrid refused: the executable plan could not be admitted under "
            f"{rcontract.ADMIT_FACTOR_NUM}/{rcontract.ADMIT_FACTOR_DEN} of the freshly "
            f"measured available RAM after {R41_ADMISSION_MAX_REPLANS} bounded "
            f"replan(s): {_adm_exc}",
            lvl="ERROR", callback=progress_callback,
        )
        raise zpar.ModeInfeasible(
            "ZeGrid admission refused (fresh RAM ceilings): " + str(_adm_exc)
        ) from _adm_exc
    if _adm.replans:
        _emit(
            f"ZeGrid: admitted after {_adm.replans} bounded replan(s) — layout "
            f"{_initial_bundle.envelope.layout[0]}x{_initial_bundle.envelope.layout[1]} "
            f"-> {_final.envelope.layout[0]}x{_final.envelope.layout[1]}, "
            f"aggregate {_final.envelope.aggregate_bytes / 2**20:.1f} MiB <= B_admit "
            f"{_adm.final_observation.budget_bytes / 2**20:.1f} MiB",
            lvl="WARN", callback=progress_callback,
        )
        _log_line(
            "ZeGrid RAM contract admission: replans="
            + json.dumps(_adm.to_dict(), sort_keys=True)
        )

    # The FINAL admitted bundle now owns every downstream decision.
    layout = _final.layout
    footprints = _final.footprints
    cell_ctxs = _final.cell_ctxs
    nx, ny = int(layout["nx"]), int(layout["ny"])
    cpu_streaming = _final.cpu_streaming
    _cs_selected = cpu_streaming.get("selected_tile_size")
    cpu_streaming_tile = int(_final.cpu_streaming_tile)
    cell_specs = _final.cell_specs
    cell_tasks = _final.cell_tasks
    chosen_mode = _final.chosen_mode
    cells_in_flight = int(_final.cells_in_flight)
    cache_workers_per_cell = int(_final.cache_workers_per_cell)
    per_cell_footprint = int(_final.per_cell_footprint)
    gpu_tile_size = _final.gpu_tile_size
    gpu_streaming_tile = _final.gpu_streaming_tile
    gpu_streaming_host_peak_bytes = _final.gpu_streaming_host_peak_bytes
    cpu_fallback_info = dict(_final.cpu_fallback_info)
    cpu_fallback_plan = _final.cpu_fallback_plan
    cpu_fallback_tasks = _final.cpu_fallback_tasks
    cache_build_workers = int(_final.cache_build_workers)
    total_cells = int(nx * ny)

    # ZM-CELL-PHOTOMETRY-R34F: "auto" resolves HERE — after the global gauge exists
    # (it already does, for progress/ETA) and the layout membership is known. It
    # picks "global" iff the global gauge serves EVERY non-empty Cell; otherwise it
    # switches the WHOLE run to "per_cell" (R34b gauges + R34c offsets + R34d
    # feather) — ONE coherent mode, never a mix of references. The DECIDED mode is
    # written back into ``science_config`` so every downstream consumer (cell
    # workers, assembly, manifest) sees the resolved effective mode, never "auto".
    auto_gauge_info = None
    if getattr(science_config, "cell_gauge_mode", None) == zphot.CELL_GAUGE_MODE_AUTO:
        _auto_membership = [
            (cell.cell_id, list(mem.patch_ids))
            for (_row, _col, cell, _patch, mem) in cell_ctxs
            if mem.patch_ids
        ]
        _auto_effective, _auto_trigger = zphot.select_auto_gauge_mode(
            global_gauge, global_frame_ids, _auto_membership
        )
        science_config = replace(science_config, cell_gauge_mode=_auto_effective)
        if _auto_trigger is None:
            auto_gauge_info = {
                "requested": zphot.CELL_GAUGE_MODE_AUTO,
                "resolved": _auto_effective,
                "fallback_fired": False,
                "cells_checked": len(_auto_membership),
                "trigger_count": 0,
                "first_cell": None,
                "reason": None,
                "cells": [],
            }
            _auto_msg = (
                "ZeGrid: cell_gauge_mode=auto -> global (all "
                f"{len(_auto_membership)} non-empty cell(s) have a globally-active frame)"
            )
        else:
            auto_gauge_info = {
                "requested": zphot.CELL_GAUGE_MODE_AUTO,
                "resolved": _auto_effective,
                "fallback_fired": True,
                "cells_checked": len(_auto_membership),
                "trigger_count": int(_auto_trigger["trigger_count"]),
                "first_cell": _auto_trigger["first_cell"],
                "reason": _auto_trigger["reason"],
                "cells": list(_auto_trigger["cells"]),
            }
            _auto_msg = (
                "ZeGrid: cell_gauge_mode=auto -> per_cell (global would fail for "
                f"{_auto_trigger['trigger_count']} cell(s); first="
                f"{_auto_trigger['first_cell']!r} reason={_auto_trigger['reason']!r}) "
                "— switching the WHOLE run to per_cell + offsets + feather"
            )
        _emit(
            _auto_msg,
            lvl=("WARN" if _auto_trigger is not None else "INFO"),
            callback=progress_callback,
        )
        _log_line(_auto_msg)

    # ZM-ZEGRID-R37-D: the capture seam only exists on the per_cell path (the only
    # point where complete pre-offset/pre-feather CellPatches coexist). Fail LOUDLY
    # here — after the effective mode is resolved (incl. auto->global) and BEFORE
    # any per-Cell stacking — rather than silently skipping the requested capture.
    if r37_diagnostic_capture and getattr(
        science_config, "cell_gauge_mode", None
    ) != zphot.CELL_GAUGE_MODE_PER_CELL:
        raise RuntimeError(
            "R37 diagnostic capture requested but the effective cell_gauge_mode is "
            f"{getattr(science_config, 'cell_gauge_mode', None)!r}, not 'per_cell'; "
            "there is no pre-offset/pre-feather per-Cell assembly point, so no "
            "artifact can be captured. Refusing to run rather than silently not "
            "capturing (use cell_gauge_mode='per_cell', or 'auto' on a corpus that "
            "resolves to per_cell)."
        )
    if gauge_cache_dir.exists():
        if not _safe_rmtree(gauge_cache_dir, progress_callback):
            cleanup_failures.append(str(gauge_cache_dir))

    # PER-CELL: build aligned cache (parallel reproject) -> stack -> DELETE cache
    # (per-cell temp reuse: peak disk is the LARGEST single cell, not the sum).
    cores = {}
    cell_records = []
    total_cells = nx * ny
    peak_rss_kib = zsw.peak_rss_kib()
    cache_total_bytes = 0
    cache_peak_bytes = 0
    cache_n_frames = 0

    # ZM-ZEGRID-R14: intra-phase progress for the two long per-cell phases.
    # cache_build reports FRAMES done/total (cumulative reprojected frames across
    # cells); per_cell_stack reports CELLS done/total.
    cache_total_frames = sum(len(mem.patch_ids) for (_r, _c, _ce, _p, mem) in cell_ctxs if mem.patch_ids)
    cache_frames_done = 0
    _cache_rep = _reporter()
    _stack_rep = _reporter()
    _cache_rep.start("cache_build", total=cache_total_frames, unit="frames")
    _stack_rep.start("per_cell_stack", total=total_cells, unit="cells")
    # ------------------------------------------------------------------
    # ZM-ZEGRID-R41-B — the FINAL admitted bundle is the ONLY source for the
    # tasks / executor / diagnostics / log / manifest (no second plan, no stale
    # per-cell budget). ``available_now`` / ``per_cell_budget`` below are the
    # FINAL PLAN observation and its exact 80 % budget (never a float factor).
    # ------------------------------------------------------------------
    _plan = _final.plan
    available_now = int(_final.envelope.raw_available_bytes)
    per_cell_budget = int(_final.envelope.budget_bytes)

    # ZM-ZEGRID-R36 (F-M1/M2): the GPU degradation decided by the bundle executor
    # is applied ONCE here (pre-execution), loudly, before any task is built.
    if bool(gpu_used) and not bool(_final.gpu_used):
        _pf = dict(_final.gpu_pre_execution_fallback or {})
        _emit(
            f"ZeGrid: GPU cannot execute this bundle ({_pf.get('reason')}) — "
            "degrading to exact CPU BEFORE execution (no GPU attempt, no silent "
            "fallback)",
            lvl="WARN", callback=progress_callback,
        )
    gpu_used = bool(_final.gpu_used)
    if bool(gpu_used) is False and _final.gpu_pre_execution_fallback is not None:
        science_config = replace(science_config, backend="cpu")
        gpu_ctx = dict(gpu_ctx)
        gpu_ctx["effective"] = False
        gpu_ctx["attempted"] = False
        gpu_ctx["final_effective"] = False
        gpu_ctx["actually_used"] = False
        gpu_ctx["final_backend"] = "cpu"
        gpu_ctx["pre_execution_fallback"] = dict(_final.gpu_pre_execution_fallback)
        gpu_ctx["fallback_reason"] = (
            gpu_ctx.get("fallback_reason")
            or str(_final.gpu_pre_execution_fallback.get("reason"))
        )
        ignored_settings = zin.ignored_settings_present(zconfig, gpu_honoured=False)

    _emit(
        f"ZeGrid: joint plan (ADMITTED, final) — mode={chosen_mode} "
        f"cells_in_flight={cells_in_flight} "
        f"cache_workers_per_cell={cache_workers_per_cell} "
        f"gpu_serialized={_plan['gpu_serialized']} — {_plan['chosen_reason']}",
        callback=progress_callback,
    )
    for c in _plan["candidates"]:
        _mk = c.get("makespan_rel")
        _feas = c.get("feasible", True)
        _emit(
            f"ZeGrid: plan candidate mode={c['mode']} cells_in_flight={c['cells_in_flight']} "
            f"max_bound={c['max_bound_bytes'] / 2**20:.1f}MiB "
            f"feasible={_feas}"
            + (f" makespan_rel={_mk:.3f}" if _mk is not None else f" ({c.get('infeasible_reason', 'infeasible')})"),
            callback=progress_callback,
        )

    # Per-cell bounds of the FINAL bundle (the values the workers really use).
    for (_row, _col, _cell, _patch, _mem), _spec in zip(cell_ctxs, _final.cell_specs):
        _inmem_b, _stream_b, _gpu_b = _spec[4], _spec[5], _spec[6]
        if _inmem_b is None:
            continue
        _idx = int(_row) * nx + int(_col)
        _n = len(_mem.patch_ids)
        _area = _patch.patch.width * _patch.patch.height
        _emit(
            f"ZeGrid: cell {_cell.cell_id} ({_idx + 1}/{total_cells}) N={_n} area={_area}px "
            f"inmem={_inmem_b / 2**20:.1f}MiB "
            f"stream(cpu{cpu_streaming_tile})={_stream_b / 2**20:.1f}MiB"
            + (
                f" stream(gpu{gpu_tile_size})={_gpu_b / 2**20:.1f}MiB"
                if _gpu_b is not None else ""
            ),
            callback=progress_callback,
        )

    _emit(
        f"ZeGrid: CPU fallback plan — feasible={cpu_fallback_info['feasible']} "
        f"mode={cpu_fallback_info['mode']} "
        f"max_bound={(cpu_fallback_info['max_bound_bytes'] or 0) / 2**20:.1f} MiB "
        f"budget={per_cell_budget / 2**20:.1f} MiB tile={cpu_streaming_tile} "
        f"cells_in_flight={cpu_fallback_info['cells_in_flight']} "
        f"reason={cpu_fallback_info['reason']}",
        callback=progress_callback,
    )
    _emit(
        f"ZeGrid: stack tiles — CPU streaming tile={cpu_streaming_tile} "
        f"(selected={_cs_selected}, preferred={CPU_STREAMING_TILE_PREFERRED}, "
        f"minimum={CPU_STREAMING_TILE_MINIMUM}) gpu_streaming_tile={gpu_streaming_tile} "
        f"gpu_streaming_host_peak_bytes={gpu_streaming_host_peak_bytes} "
        f"backend={science_config.backend}",
        callback=progress_callback,
    )
    _emit(
        f"ZeGrid: per-cell concurrency — cells_in_flight={cells_in_flight} "
        f"(cpu={os.cpu_count()}, plan_raw={available_now / 2**30:.2f}GiB, "
        f"B_plan={per_cell_budget / 2**30:.2f}GiB, "
        f"footprint={per_cell_footprint / 2**20:.0f}MiB, "
        f"cache_build_workers={cache_build_workers})",
        callback=progress_callback,
    )
    _emit(
        f"ZeGrid: RAM contract — plan raw {_final.envelope.raw_available_bytes / 2**20:.1f} MiB "
        f"-> B_plan {_final.envelope.budget_bytes / 2**20:.1f} MiB "
        f"({rcontract.PLAN_FACTOR_NUM}/{rcontract.PLAN_FACTOR_DEN}); "
        f"admit raw {_adm.final_observation.raw_available_bytes / 2**20:.1f} MiB "
        f"-> B_admit {_adm.final_observation.budget_bytes / 2**20:.1f} MiB "
        f"({rcontract.ADMIT_FACTOR_NUM}/{rcontract.ADMIT_FACTOR_DEN}); "
        f"aggregate {_final.envelope.aggregate_bytes / 2**20:.1f} MiB "
        f"(shared {_final.envelope.shared_bytes / 2**20:.1f} MiB + stack "
        f"{_final.envelope.stack_total_bytes / 2**20:.1f} / cache "
        f"{_final.envelope.cache_total_bytes / 2**20:.1f}); "
        f"replans={_adm.replans} tolerance={_adm.tolerance}; "
        f"layout {_initial_bundle.envelope.layout[0]}x{_initial_bundle.envelope.layout[1]} "
        f"-> {_final.envelope.layout[0]}x{_final.envelope.layout[1]}",
        callback=progress_callback,
    )

    task_cids = [t[0].cell_id for t in cell_tasks]
    per_cell_diagnostics = {
        "cells_in_flight": int(cells_in_flight),
        "cpu": int(os.cpu_count() or 1),
        "available_bytes": int(available_now),
        "ram_budget_bytes": int(per_cell_budget),
        "per_cell_footprint_bytes": int(per_cell_footprint),
        "cells": int(len(cell_tasks)),
        "cache_build_workers": int(cache_build_workers),
        "mode": chosen_mode,
        "cache_workers_per_cell": int(cache_workers_per_cell),
        "gpu_serialized": bool(_plan.get("gpu_serialized")),
        "plan_chosen_reason": _plan.get("chosen_reason"),
        "plan_candidates": _plan.get("candidates"),
        "cpu_streaming_tile": int(cpu_streaming_tile),
        "cpu_streaming_tile_selected": _cs_selected,
        "gpu_streaming_tile": (int(gpu_streaming_tile) if gpu_streaming_tile is not None else None),
        "gpu_streaming_host_peak_bytes": (
            int(gpu_streaming_host_peak_bytes) if gpu_streaming_host_peak_bytes is not None else None
        ),
        "cpu_streaming": dict(cpu_streaming),
        "cpu_fallback_used": False,
        "cpu_fallback_plan": dict(cpu_fallback_info),
        # ZM-ZEGRID-R41-B: the full 80 %-plan / 90 %-admit contract trail.
        "ram_contract": _ram_contract_diagnostics(_adm, _final, _initial_bundle),
        "requested_cell_gauge_mode": requested_cell_gauge_mode,
    }

    # ZM-ZEGRID-R55-B — OBSERVE-ONLY scheduler integration. The admitted legacy
    # bundle remains the ONLY execution authority: this block records what R55
    # WOULD choose, additively, in the manifest. It is built after admission and
    # before stacking, never mutates the bundle, and its failure never aborts the
    # run (a refused/erroring observe plan is recorded verbatim).
    _r55_scheduler_block = None
    if scheduler_mode == zsched.MODE_OBSERVE:
        # R55-C: ONE explicit, contained readiness probe, observe mode only.
        # It never changes execution authority and never aborts CPU work.
        _r55_gpu_snap = zgr.probe_gpu_readiness_safe()
        _r55_gpu_readiness_block = _r55_gpu_readiness_observe_block(
            gpu_ctx, getattr(_final, "gpu_used", False), _r55_gpu_snap
        )
        _r55_requested_backend = _r55_gpu_readiness_block["product"][
            "requested_backend"
        ]
        _r55_effective_backend = _r55_gpu_readiness_block["product"][
            "effective_backend"
        ]
        if getattr(_final, "gpu_used", False):
            # R55-B plans the CPU backend only. Record the refusal honestly
            # instead of planning from a GPU-shaped bundle (GPU behaviour is
            # explicitly out of scope for this mission).
            _r55_scheduler_block = {
                "schema": zsched.MANIFEST_SCHEMA,
                "scheduler_version": zsched.SCHEDULER_VERSION,
                "mode": "observe_only",
                "requested_mode": zsched.MODE_OBSERVE,
                "status": "not_supported_cpu_only",
                "reason_code": renv.REASON_UNKNOWN_BACKEND,
                "note": (
                    "R55-B is CPU-only; no GPU plan is predicted and the GPU "
                    "execution path is unchanged."
                ),
                "gpu_readiness": _r55_gpu_readiness_block,
            }
        else:
            _r55_frame_ids = tuple(
                getattr(getattr(f, "frame_id", None), "logical_path", str(f))
                for f in (descs or ())
            )
            try:
                _r55_scheduler_block = _build_r55_observe_block(
                    bundle=_final,
                    chosen_mode=chosen_mode,
                    cell_ctxs=cell_ctxs,
                    canvas=canvas,
                    halo_px=halo_px,
                    output_dir=output_dir,
                    cache_root=cache_root,
                    frame_ids=_r55_frame_ids,
                    scheduler_mode=scheduler_mode,
                    gpu_readiness=_r55_gpu_snap,
                    requested_backend=_r55_requested_backend,
                    effective_backend=_r55_effective_backend,
                )
                _emit(
                    "ZeGrid: R55 observe-only plan recorded — "
                    f"plan_hash={_r55_scheduler_block.get('plan_hash')} "
                    f"mode={_r55_scheduler_block.get('mode')} "
                    "(execution remains legacy)",
                    callback=progress_callback,
                )
            except zsched.PlanRefused as _r55_exc:
                _r55_scheduler_block = _r55_exc.refusal_manifest_dict(
                    mode=zsched.MODE_OBSERVE,
                    run_id=str(getattr(canvas, "canvas_id", "")),
                )
                _r55_scheduler_block["gpu_readiness"] = _r55_gpu_readiness_block
                _emit(
                    "ZeGrid: R55 observe-only plan REFUSED "
                    f"({_r55_exc.reason_code}); execution continues on the legacy plan",
                    lvl="WARN",
                    callback=progress_callback,
                )
            except Exception as _r55_exc:  # observe must never control execution
                _r55_scheduler_block = _r55_observe_error_block(
                    _r55_exc, gpu_readiness=_r55_gpu_readiness_block
                )
                _emit(
                    f"ZeGrid: R55 observe-only plan ERROR ({_r55_exc!r}); "
                    "execution continues on the legacy plan",
                    lvl="WARN",
                    callback=progress_callback,
                )
    per_cell_meta = {}

    def _cell_progress_for(tasks):
        def _cb(done, total):
            cid = task_cids[int(done) - 1] if 0 < int(done) <= len(task_cids) else None
            _stack_rep.progress(int(done), item_id=cid)
        return _cb

    t_block0 = time.perf_counter()

    # ZM-ZEGRID-R40-B2-C: when the seam capture is ON, the deterministic candidate
    # geometry is computed HERE in the parent (before any worker starts) from the
    # task Cell bounds and handed to every pooling worker so each worker returns
    # ONLY the sparse covariate samples of its OWN cell(s) from its worker-local
    # MiniTile planes. When the capture is OFF this stays ``None`` and the pool
    # initargs are byte-identical to the pre-R40 transport (no extra IPC payload).
    _r40_coords_by_cell = None
    if r40_seam_capture:
        from .core.zegrid import background_model as _r40bm

        _r40_bounds = {
            str(t[0].cell_id): (
                int(t[1].patch.y0), int(t[1].patch.y1),
                int(t[1].patch.x0), int(t[1].patch.x1),
            )
            for t in cell_tasks
        }
        _r40_coords_by_cell = _r40_worker_candidate_coords(
            _r40_bounds, nx, ny, block_px=int(_r40bm.BLOCK_PX)
        )

    def _run_cell_batch(science_cfg, tile_size, *, gpu_attempt,
                        tasks=None, workers=None, cache_workers=None):
        """Run the WHOLE per-cell batch once; returns (results, meta).

        ``gpu_attempt=True`` disables pmap's SAME-config serial retry (serial_fallback
        False) so a CuPy OOM/driver error propagates here for a one-shot CPU rerun
        instead of being retried with the same GPU config (which would repeat the
        error and crash).

        ``tasks``/``workers``/``cache_workers`` (ZM-ZEGRID-R36 F-M2) let the CPU
        fallback rerun HONOR an independently-computed CPU plan instead of the
        primary GPU plan (exact mode/tasks/bounds + safe concurrency/cache workers).
        """
        _tasks = cell_tasks if tasks is None else tasks
        _workers = cells_in_flight if workers is None else int(workers)
        _cache_workers = cache_workers_per_cell if cache_workers is None else int(cache_workers)
        _meta = {}
        # ZM-ZEGRID-R40-B2-C: the worker initargs are byte-identical to the
        # pre-R40 transport when the capture is OFF (the extra transport arg is
        # appended ONLY when the parent armed the capture and pre-computed the
        # candidate geometry).
        _initargs = (descs, canvas, science_cfg, global_gauge, global_frame_ids,
                     global_reference_frame_id, tile_size, _cache_workers)
        if _r40_coords_by_cell is not None:
            _initargs = _initargs + (_r40_coords_by_cell,)
        _res = zpar.pmap(
            _stack_cell, _tasks, workers=_workers,
            progress_callback=_cell_progress_for(_tasks), emit=_emit_live, meta=_meta,
            initializer=_init_stack_worker,
            initargs=_initargs,
            serial_fallback=not gpu_attempt,
        )
        return _res, _meta

    runtime_gpu_fallback = None
    if gpu_used:
        try:
            results, per_cell_meta = _run_cell_batch(
                science_config, gpu_tile_size, gpu_attempt=True
            )
        except Exception as exc:
            if zgpu.is_gpu_runtime_error(exc):
                # H1 / R36 F-M2 / R41-B: classified GPU runtime failure -> ONE
                # bounded, FULLY re-planned CPU admission cycle on the SAME final
                # geometry (a mosaic whose scientific cells may already have been
                # stacked is NEVER re-sliced), under a FRESH 80 % raw observation
                # and admitted against a FRESH 90 % observation. No stale CPU
                # budget is reused and no unchecked CPU rerun is possible.
                runtime_gpu_fallback = {"reason": repr(exc), "type": type(exc).__name__}
                _emit(
                    f"ZeGrid: GPU RUNTIME failure ({type(exc).__name__}); the failed "
                    "worker pool is no longer active (pmap closed it before re-raising) "
                    "and partial per-cell caches are removed BEFORE re-measuring RAM — "
                    f"no silent fallback: {exc}",
                    lvl="WARN", callback=progress_callback,
                )
                gpu_used = False
                science_config = replace(science_config, backend="cpu")
                gpu_ctx = dict(gpu_ctx)
                # R22 test-debt: ``attempted`` records the START-TIME selection
                # and the genuine execution attempt, NOT its outcome. A runtime
                # GPU failure means the GPU WAS attempted; only the pre-execution
                # admission fallback resets it to False. Keep it True so the
                # manifest never contradicts the runtime fallback evidence.
                gpu_ctx["effective"] = False
                gpu_ctx["final_effective"] = False
                gpu_ctx["actually_used"] = False
                gpu_ctx["final_backend"] = "cpu"
                gpu_ctx["runtime_fallback"] = runtime_gpu_fallback
                ignored_settings = zin.ignored_settings_present(zconfig, gpu_honoured=False)
                # Clean any partial per-cell cache left by the failed GPU workers
                # (bounded disk + Windows file-lock safety: the failed workers are
                # already closed when pmap re-raised; deletion is best-effort).
                for _t in cell_tasks:
                    if _t[4] is not None:
                        _safe_rmtree(_t[3], None)

                _fix_rank = int(_final.envelope.refinement_rank)
                # R41-B (rework-1): EXACT geometry. A same-cell-count rank is NOT
                # proof of the same (nx, ny); pin the final admitted tuple so every
                # fallback build keeps the same nx/ny, memberships and cell IDs.
                _fix_layout = tuple(int(v) for v in _final.envelope.layout)
                _fb_raw = int(available_memory_bytes())
                try:
                    _fb_budget = int(rcontract.plan_budget_bytes(_fb_raw))
                except rcontract.InvalidObservation as _fb_exc:
                    raise zpar.ModeInfeasible(
                        "GPU runtime failure and the fresh RAM observation is "
                        f"nonpositive/unusable ({_fb_exc}); refusing any CPU rerun"
                    ) from exc
                _emit(
                    f"ZeGrid: rebuilding an exact-CPU fallback bundle on the SAME "
                    f"{_final.envelope.layout[0]}x{_final.envelope.layout[1]} geometry "
                    f"under a FRESH plan budget {_fb_budget / 2**20:.1f} MiB "
                    f"(raw {_fb_raw / 2**20:.1f} MiB)",
                    lvl="WARN", callback=progress_callback,
                )
                try:
                    _fb_seed = _build_execution_bundle(
                        canvas, descs, footprints,
                        raw_available_bytes=_fb_raw, budget_bytes=_fb_budget,
                        pinned_layout=_fix_layout,
                        min_refinement_rank=_fix_rank,
                        max_refinement_rank=_fix_rank,
                        layout_substep=None,
                        **{**_bundle_common, "gpu_used": False},
                    )
                    _fb_adm, _fb_final = _admit(
                        _fb_seed, fix_rank=_fix_rank, backend_gpu=False,
                        pinned_override=_fix_layout,
                    )
                except (zal.LayoutInfeasible, zpar.ModeInfeasible) as _fb_exc:
                    _emit(
                        "ZeGrid: GPU runtime failure and NO exact-CPU bundle fits the "
                        f"fresh RAM budget on the SAME geometry — refusing an "
                        f"over-budget CPU rerun (loud failure): {_fb_exc}",
                        lvl="ERROR", callback=progress_callback,
                    )
                    raise zpar.ModeInfeasible(
                        "GPU runtime failure and no exact-CPU bundle fits the fresh RAM "
                        f"budget on the same geometry ({_fb_exc}); refusing an "
                        "over-budget CPU rerun"
                    ) from exc
                except rcontract.AdmissionExhausted as _fb_exc:
                    _emit(
                        "ZeGrid: GPU runtime failure and the exact-CPU bundle could not "
                        f"be admitted under a fresh 90 % ceiling — refusing an "
                        f"over-budget CPU rerun (loud failure): {_fb_exc}",
                        lvl="ERROR", callback=progress_callback,
                    )
                    raise zpar.ModeInfeasible(
                        "GPU runtime failure and the exact-CPU bundle could not be "
                        f"admitted under a fresh RAM ceiling ({_fb_exc})"
                    ) from exc
                # Adopt the freshly admitted CPU bundle as the ONLY source of truth.
                _gpu_attempt_contract = dict(_ram_contract_diagnostics(
                    _adm, _final, _initial_bundle
                ))
                _final = _fb_final
                _adm = _fb_adm
                cell_specs = _final.cell_specs
                cell_tasks = _final.cell_tasks
                cpu_streaming = _final.cpu_streaming
                cpu_streaming_tile = int(_final.cpu_streaming_tile)
                chosen_mode = _final.chosen_mode
                cells_in_flight = int(_final.cells_in_flight)
                cache_workers_per_cell = int(_final.cache_workers_per_cell)
                per_cell_footprint = int(_final.per_cell_footprint)
                # Provenance: KEEP the ATTEMPTED GPU tile/bound (what the GPU really
                # tried) — the CPU rerun has its own tile below.
                cpu_fallback_plan = _final.cpu_fallback_plan
                cpu_fallback_tasks = _final.cpu_fallback_tasks
                cache_build_workers = int(_final.cache_build_workers)
                cpu_fallback_info = dict(_final.cpu_fallback_info)
                cpu_fallback_info["used"] = True
                cpu_fallback_info["runtime_refresh"] = True
                per_cell_diagnostics["mode"] = chosen_mode
                per_cell_diagnostics["cells_in_flight"] = cells_in_flight
                per_cell_diagnostics["cache_workers_per_cell"] = cache_workers_per_cell
                per_cell_diagnostics["cpu_streaming_tile"] = int(cpu_streaming_tile)
                per_cell_diagnostics["cpu_fallback_used"] = True
                per_cell_diagnostics["cpu_fallback_plan"] = dict(cpu_fallback_info)
                per_cell_diagnostics["cpu_streaming"] = dict(cpu_streaming)
                per_cell_diagnostics["ram_contract"] = _ram_contract_diagnostics(
                    _adm, _final, _initial_bundle
                )
                # The GPU attempt's own contract history is retained SEPARATELY so a
                # runtime fallback never overwrites the only recorded evidence.
                per_cell_diagnostics["gpu_attempt_ram_contract"] = _gpu_attempt_contract
                per_cell_diagnostics["runtime_cpu_fallback_contract"] = (
                    _ram_contract_diagnostics(_adm, _final, _fb_seed)
                )
                _emit(
                    f"ZeGrid: exact-CPU fallback ADMITTED on the same geometry — "
                    f"mode={chosen_mode} cells_in_flight={cells_in_flight} "
                    f"tile={cpu_streaming_tile} replans={_adm.replans} "
                    f"tolerance={_adm.tolerance}",
                    lvl="WARN", callback=progress_callback,
                )
                results, per_cell_meta = _run_cell_batch(
                    science_config, cpu_streaming_tile, gpu_attempt=False,
                    tasks=cell_tasks, workers=cells_in_flight,
                    cache_workers=cache_workers_per_cell,
                )
                task_cids = [t[0].cell_id for t in cell_tasks]
                per_cell_diagnostics["cells"] = int(len(cell_tasks))
            else:
                raise  # arbitrary science/programming error propagates (never swallowed)
    else:
        results, per_cell_meta = _run_cell_batch(
            science_config, cpu_streaming_tile, gpu_attempt=False
        )

    block_wall = time.perf_counter() - t_block0

    # H2: actual backend EVIDENCE from the executed cells (never the selection).
    # `used=true` only when a successful cell actually ran the GPU canonical and
    # no whole-batch CPU rerun replaced it.
    gpu_executed_cells = sum(
        1 for r in results
        if r.get("status") == "complete" and r.get("backend_used") == "gpu"
    )
    actually_used_gpu = bool(gpu_executed_cells > 0 and runtime_gpu_fallback is None)
    gpu_ctx = dict(gpu_ctx)
    gpu_ctx["gpu_executed_cells"] = int(gpu_executed_cells)
    gpu_ctx["actually_used"] = bool(actually_used_gpu)
    gpu_ctx["final_backend"] = "gpu" if actually_used_gpu else "cpu"
    gpu_ctx["final_effective"] = bool(actually_used_gpu)
    # ZM-ZEGRID-R36: distinguish the GPU streaming tile from the CPU streaming
    # tile (the GPU->CPU fallback uses the CPU tile, never a re-coded 128).
    gpu_ctx["cpu_streaming_tile"] = int(cpu_streaming_tile)
    gpu_ctx["gpu_streaming_tile"] = (
        int(gpu_streaming_tile) if gpu_streaming_tile is not None else None
    )
    gpu_ctx["gpu_streaming_host_peak_bytes"] = (
        int(gpu_streaming_host_peak_bytes) if gpu_streaming_host_peak_bytes is not None else None
    )
    gpu_ctx["cpu_fallback_plan"] = dict(cpu_fallback_info)
    gpu_used = bool(actually_used_gpu)

    # R20 diagnostics (mirrors the R19 gauge diagnostics): effective executor,
    # parent daemon flag, workers/cells-in-flight, seconds/unit.
    per_cell_diagnostics["executor"] = per_cell_meta.get("executor", "serial")
    per_cell_diagnostics["parent_daemon"] = bool(per_cell_meta.get("parent_daemon"))
    per_cell_diagnostics["workers"] = int(per_cell_meta.get("workers", cells_in_flight))
    per_cell_diagnostics["tasks"] = int(per_cell_meta.get("tasks", len(cell_tasks)))
    per_cell_diagnostics["seconds"] = block_wall
    per_cell_diagnostics["seconds_per_unit"] = block_wall / max(1, len(cell_tasks))
    per_cell_diagnostics["fallback"] = bool(per_cell_meta.get("fallback"))
    if per_cell_meta.get("fallback_reason"):
        per_cell_diagnostics["fallback_reason"] = per_cell_meta["fallback_reason"]
    per_cell_diagnostics["cache_build_cpu_sum_s"] = round(
        sum(r.get("cache_build_s", 0.0) for r in results), 6
    )
    per_cell_diagnostics["stack_cpu_sum_s"] = round(
        sum(r.get("stack_s", 0.0) for r in results), 6
    )
    # ZM-ZEGRID-R21: which reprojection path the per-cell cache build took,
    # aggregated over every cell worker (each reports its process-local stats).
    per_cell_diagnostics["reprojection"] = zxe.merge_reproject_stats(
        *(r.get("reproject") for r in results)
    )

    # R20: cache build is FUSED into the concurrent per-cell phase (bounded disk
    # requires the cache to be built+deleted inside each cell worker), so the
    # phase's wall-clock is recorded under per_cell_stack; cache_build is 0 here.
    timings.add("per_cell_stack", block_wall)
    timings.add("cache_build", 0.0)

    result_by_cid = {r["cell_id"]: r for r in results}

    # Ordered (row-major) accounting + records, empty cells interleaved exactly
    # as the serial loop emits them.
    for (row, col, cell, patch, mem) in cell_ctxs:
        cid = cell.cell_id
        idx = row * nx + col
        r = result_by_cid[cid]
        if r["status"] == "empty":
            _emit(f"ZeGrid: cell {cid} ({idx + 1}/{total_cells}) empty — no patch contributors",
                  callback=progress_callback)
            cell_records.append(r["record"])
            continue
        cache_frames_done += r["n_frames"]
        _cache_rep.progress(cache_frames_done, item_id=cid)
        cache_total_bytes += r["total_bytes"]
        cache_n_frames += r["n_frames"]
        cache_peak_bytes = max(cache_peak_bytes, r["total_bytes"])
        cores[cid] = r["core"]
        peak_rss_kib = max(peak_rss_kib, r["peak_rss_kib"])
        if r["cleanup_failed"]:
            cleanup_failures.append(r["cache_dir"])
            _emit(f"ZeGrid: cell {cid} cache cleanup failed (non-fatal)", lvl="WARN",
                  callback=progress_callback)
        cell_records.append(r["record"])

    _cache_rep.end(
        throughput=_fmt_throughput(cache_total_frames, timings.get("cache_build"), "frames/s")
    )
    global_eta.phase_completed("cache_build", timings.get("cache_build"))
    _stack_rep.end(
        throughput=_fmt_throughput(total_cells, timings.get("per_cell_stack"), "cells/s")
    )
    global_eta.phase_completed("per_cell_stack", timings.get("per_cell_stack"))

    # ASSEMBLY.
    _gauge_mode = getattr(science_config, "cell_gauge_mode", "global")
    feather_info = None
    if _gauge_mode == zphot.CELL_GAUGE_MODE_PER_CELL:
        # R34D opt-in path: solve inter-cell additive offsets on the stacked
        # cells' halo overlaps, apply them (additive-only), then feather-blend.
        _emit(
            "ZeGrid: assembly (R34D per_cell feather: offsets + weighted blend)",
            callback=progress_callback,
        )
    else:
        _emit("ZeGrid: assembly (R3 assemble_canvas)", callback=progress_callback)
    _assembly_rep = _reporter()
    _assembly_rep.start("assembly")
    with timings.timed("assembly"):
        _r40_context = None
        if r40_seam_capture:
            from .core.zegrid import background_diagnostics as _bgd
            from .core.zegrid import background_model as _zbm

            _halo_i = int(halo_px)
            _meas_px = int(2 * _halo_i) if _halo_i > 0 else int(_bgd.DEFAULT_MEASUREMENT_PX)
            _feath_px = int(_halo_i) if _halo_i > 0 else int(_bgd.DEFAULT_FEATHER_PX)
            _r40_context = _r40_build_seam_context(
                output_root=(r40_output_root or output_dir),
                disk_budget_bytes=r40_seam_capture_disk_budget_bytes,
                capture_id=r40_capture_id,
                descs=descs, science_config=science_config, canvas=canvas,
                nx=nx, ny=ny, result_by_cid=result_by_cid,
                halo_px=_halo_i, measurement_px=_meas_px, feather_px=_feath_px,
                block_px=int(_zbm.BLOCK_PX), channels=3,
                protected_roots=(), repo_head=None,
            )
        assembled, feather_info = _assemble_cells(
            canvas, nx, ny, _gauge_mode, cores, result_by_cid,
            halo_px=int(halo_px), emit=_emit_live,
            capture=r37_diagnostic_capture,
            holdout_cap=r37_diagnostic_holdout_cap,
            output_dir=output_dir,
            capture_identity=r37_capture_identity,
            r40_capture=r40_seam_capture,
            r40_context=_r40_context,
        )
    peak_rss_kib = max(peak_rss_kib, zsw.peak_rss_kib())
    # ZM-ZEGRID-R22 honesty fix: peak_rss_kib is a SINGLE process's peak. Report
    # the batch aggregate (parent + all cell workers) as a clearly-labelled UPPER
    # bound alongside it, so the manifest no longer under-states parallel memory.
    aggregate_peak_rss_kib = zsw.aggregate_peak_rss_kib(
        zsw.peak_rss_kib(), [r.get("peak_rss_kib", 0) for r in results]
    )
    _assembly_rep.end(
        throughput=_fmt_throughput(len(assembled.complete_cells), timings.get("assembly"), "cells/s")
    )
    global_eta.phase_completed("assembly", timings.get("assembly"))

    # I2 — no covered pixels -> explicit abort (never an all-NaN mosaic).
    if assembled.coverage_pixels == 0:
        _emit(
            "ZeGrid: canvas has NO covered pixels; aborting (would produce an all-NaN mosaic)",
            lvl="ERROR",
            callback=progress_callback,
        )
        raise RuntimeError("ZeGrid: no covered pixels; cannot assemble a mosaic")

    _halo_provenance_pass = _halo_provenance_for_pass(
        halo_provenance,
        effective_px=int(halo_px),
        cell_ctxs=cell_ctxs,
        requested_gauge_mode=(
            (auto_gauge_info or {}).get("requested")
            or getattr(science_config, "cell_gauge_mode", "global")
        ),
        effective_gauge_mode=_gauge_mode,
        feather_info=feather_info,
    )
    # F-2: validate BEFORE the immutable raw science FITS, finishing, or any
    # current-run output is published. A contradiction therefore cannot leave a
    # current-run partial output behind.
    _validate_halo_provenance_consistency(
        _halo_provenance_pass, science_config, feather_info
    )

    cache_report = {
        "reused": [],
        "rebuilt": assembled.complete_cells,
        "total_bytes": cache_total_bytes,
        "peak_cell_bytes": cache_peak_bytes,
        "n_frame_entries": cache_n_frames,
        "retention": "per-cell temporary (deleted after stacking); gauge cache deleted after use",
        "cleanup_failures": cleanup_failures,
    }

    # ZM-ZEGRID-R23 rework-3 (H1): resolve the Classic-compatible output naming
    # from the finishing config (existing export_aesthetic_fits + cleaned suffixes).
    raw_science = np.asarray(assembled.science, dtype=np.float32)
    raw_science_path, aesthetic_path = _resolve_output_paths(output_dir, finishing_config)

    # ZM-PROGRESS-CONTRACT-R29: honest ``zegrid:finalize`` seam. It spans final
    # finishing + output publication (science/aesthetic/coverage/manifest) + the
    # completed run log, and is marked done ONLY after those writes succeed. It
    # participates in UI progress (its local 100% is still below the global 100%
    # until the explicit terminal success) but carries NO GlobalEta cost here.
    _finalize_rep = _reporter()
    _finalize_rep.start("finalize", total=1)

    # Write the immutable pre-finishing assembled science FIRST, so a finishing
    # exception can never destroy the scientific reference.
    _write_raw_science_fits(
        assembled, canvas, raw_science_path,
        related_file=(aesthetic_path.name if aesthetic_path is not None else None),
        halo_provenance=_halo_provenance_pass,
    )

    # ZM-ZEGRID-R18: FINAL-MOSAIC FINISHING (post-assembly, before outputs).
    # Opt-in + fail-safe: a finishing failure WARNS and lets the run complete with
    # the raw assembled science (never crashes a run; never silently dropped).
    finished_science = np.asarray(assembled.science, dtype=np.float32)
    finishing_info = {"enabled": False, "failed": False, "failure_reason": ""}
    fin_uint16 = None
    try:
        _fin = zfin.apply_final_mosaic_finishing(
            assembled.science, assembled.stack_depth, config=finishing_config
        )
        finished_science = _fin.science
        finishing_info = _fin.info
        fin_uint16 = _fin.uint16
        if finishing_info.get("enabled"):
            _emit(
                f"ZeGrid: final-mosaic finishing applied — dbe={finishing_info.get('dbe', {}).get('applied')} "
                f"rgb_equalize={finishing_info.get('rgb_equalize', {}).get('applied')} "
                f"uint16={finishing_info.get('uint16', {}).get('written')}",
                callback=progress_callback,
            )
    except Exception as exc:
        _emit(
            f"ZeGrid: final-mosaic finishing FAILED ({exc}); writing raw mosaic (WARN)",
            lvl="WARN", callback=progress_callback,
        )
        finished_science = np.asarray(assembled.science, dtype=np.float32)
        finishing_info = {
            "enabled": bool(finishing_config and any([
                finishing_config.get("dbe_enabled"),
                finishing_config.get("rgb_equalize"),
                finishing_config.get("save_uint16"),
            ])),
            "failed": True,
            "failure_reason": repr(exc),
        }
        fin_uint16 = None

    # R55-C rework-2: the observe block was built from the SELECTED backend.
    # Reconcile its nested readiness provenance with the backend that actually
    # produced the whole-batch output (runtime GPU->CPU fallback included) before
    # any manifest/output write. Never changes requested_backend or execution.
    _r55_scheduler_block = _r55_reconcile_gpu_readiness(
        _r55_scheduler_block, gpu_ctx
    )

    # Write Classic-compatible outputs (raw always; aesthetic only if checkbox).
    export_aesthetic = bool((finishing_config or {}).get("export_aesthetic_fits", False))
    sci_path, cov_path, manifest_path = _write_outputs(
        assembled, canvas, nx, ny, output_dir, descs, {}, cell_records,
        layout, science_config, peak_rss_kib, cache_report, progress_callback,
        rejected=rejected, sip_mode=sip_mode,
        frames_loaded=len(frames_info),
        global_reference_frame_id=global_reference_frame_id,
        timings=timings, gpu_used=gpu_used, ignored_settings=ignored_settings,
        ignored_run_args=ignored_run_args,
        gauge_diagnostics=gauge_diagnostics,
        per_cell_diagnostics=per_cell_diagnostics,
        auto_gauge_info=auto_gauge_info,
        cell_feather_info=feather_info,
        finished_science=finished_science, finishing_info=finishing_info,
        fin_uint16=fin_uint16,
        gpu_context=gpu_ctx,
        aggregate_peak_rss_kib=aggregate_peak_rss_kib,
        raw_science_path=raw_science_path,
        raw_science=raw_science,
        aesthetic_path=aesthetic_path if export_aesthetic else None,
        export_aesthetic_fits=export_aesthetic,
        scientific_fits_suffix=(finishing_config or {}).get(
            "scientific_fits_suffix", "_science"
        ),
        halo_provenance=_halo_provenance_pass,
        r55_scheduler=_r55_scheduler_block,
    )

    _write_run_log(
        output_dir,
        timings,
        start_ts=start_ts,
        frames_loaded=len(frames_info),
        n_included=len(descs),
        n_rejected=len(rejected or []),
        canvas=canvas,
        layout=layout,
        gpu_used=gpu_used,
        ignored_settings=ignored_settings,
        ignored_run_args=ignored_run_args,
        peak_rss_kib=peak_rss_kib,
        cache_info=cache_report,
        global_reference_frame_id=global_reference_frame_id,
        gauge_diagnostics=gauge_diagnostics,
        per_cell_diagnostics=per_cell_diagnostics,
        finishing_info=finishing_info,
        gpu_context=gpu_ctx,
        aggregate_peak_rss_kib=aggregate_peak_rss_kib,
        halo_provenance=_halo_provenance_pass,
    )

    # ZM-PROGRESS-CONTRACT-R29: the finalize seam completes (1/1) only now that
    # the immutable science + aesthetic + coverage + manifest + run log are all
    # written. Its local 100% still leaves GLOBAL progress below 100% until the
    # worker's explicit terminal success.
    finalize_wall = _finalize_rep.end()

    # ZM-ETA-SERVICE-R30: append ONE sanitized ZeGrid history record, best-effort,
    # before the terminal success. ``total_duration_s`` is the honest actual wall
    # duration of this ``_run_single`` segment from an explicit perf_counter start
    # (F4), NOT a phase-sum guess; the per-stage ``stage_seconds`` are unchanged
    # (``Timings`` covers setup..assembly; finalize is added here). It measures
    # the finalize wall duration honestly WITHOUT changing science/output order —
    # the manifest is already written, so the finalize duration simply does not
    # retroactively appear in it (documented). A history write failure is
    # nonfatal and cannot change output/science success, and no history is
    # written for a failed/cancelled segment (any earlier exception raised before
    # this point skips the write). For a split EQ/ALTZ run, one record is written
    # per successful ``_run_single`` segment (each is a valid comparable segment;
    # deliberate).
    try:
        zeta.append_zegrid_history(
            zeta.build_zegrid_history_record(
                total_duration_s=time.perf_counter() - run_wall_start,
                n_frames=len(frames_info),
                cell_count=total_cells,
                stage_seconds={
                    "setup": timings.get("setup"),
                    "layout": timings.get("layout"),
                    "gauge": timings.get("gauge"),
                    "per_cell_stack": timings.get("per_cell_stack"),
                    "assembly": timings.get("assembly"),
                    "finalize": finalize_wall,
                },
                stage_totals={
                    "setup": len(frames_info),
                    "gauge": 2 * len(descs) - 1,
                    "per_cell_stack": total_cells,
                    "finalize": 1,
                },
                workers=workers,
                backend="gpu" if gpu_used else "cpu",
            )
        )
    except Exception:
        pass

    _emit(
        f"ZeGrid: done — science={raw_science_path.name} "
        f"aesthetic={aesthetic_path.name if (export_aesthetic and aesthetic_path is not None) else 'n/a'} "
        f"({assembled.science.shape}) + coverage + run log, "
        f"complete={len(assembled.complete_cells)} incomplete={len(assembled.incomplete_cells)} "
        f"holes={assembled.hole_pixels}px peak_rss={peak_rss_kib}KiB "
        f"cache_total={cache_total_bytes / 2**20:.0f}MiB cache_peak={cache_peak_bytes / 2**20:.0f}MiB",
        lvl="SUCCESS",
        callback=progress_callback,
    )
    return sci_path


def _resolve_output_paths(output_dir, finishing_config):
    """Resolve the raw-science + aesthetic output paths (Classic naming).

    Base name is ``mosaic_grid``. Per the rework-3 Classic-compatible contract:

    * ``export_aesthetic_fits=false`` → raw science at ``mosaic_grid.fits``
      (pre-finishing, SCI role), NO aesthetic float companion;
    * ``export_aesthetic_fits=true`` → raw science at
      ``mosaic_grid<clean scientific suffix>.fits`` + aesthetic float32 at
      ``mosaic_grid<clean aesthetic suffix>.fits`` (defaults
      ``mosaic_grid_science.fits`` / ``mosaic_grid_aesthetic.fits``).

    Returns ``(raw_science_path, aesthetic_path_or_None)``.
    """
    cfg = finishing_config or {}
    export = bool(cfg.get("export_aesthetic_fits", False))
    sci_suffix = cfg.get("scientific_fits_suffix", "_science")
    aest_suffix = cfg.get("aesthetic_fits_suffix", "_aesthetic")
    output_dir = Path(output_dir)
    if export:
        raw_path = output_dir / f"mosaic_grid{sci_suffix}.fits"
        aest_path = output_dir / f"mosaic_grid{aest_suffix}.fits"
    else:
        raw_path = output_dir / "mosaic_grid.fits"
        aest_path = None
    return raw_path, aest_path


def _write_raw_science_fits(
    assembled, canvas, raw_science_path, role="SCI", related_file=None,
    halo_provenance=None,
):
    """Write the immutable pre-finishing assembled science FITS.

    Written BEFORE finishing runs so a finishing exception can never destroy the
    scientific reference. Same WCS/axis layout as the delivered output, float32,
    never clamped/offset/abs'd. Header records role (``SCI`` scientific reference),
    dtype + DBE state + hash, plus the related aesthetic file (when one is emitted).

    ZM-FITS-INTEROP-R26: non-finite samples (NaN/+Inf/-Inf) are replaced with
    float32 0.0 in the serialized buffer ONLY (never the in-memory science), so
    the on-disk FITS opens correctly in ASIFitsView Linux and Gwenview. The
    replaced count + fill value are recorded in the header.
    """
    raw_science = np.asarray(assembled.science, dtype=np.float32)
    raw_data = np.ascontiguousarray(np.moveaxis(raw_science, -1, 0))  # (3, H, W)
    raw_data, nf_replaced = sanitize_nonfinite_float32(raw_data, fill_value=0.0)
    raw_header = _science_header(
        canvas,
        role=role,
        dtype="float32",
        dbe_state="n/a",
        sha256=_array_sha256(raw_data),
        related_file=related_file,
    )
    _record_halo_cards(raw_header, halo_provenance)
    record_nonfinite_fill(raw_header, nf_replaced)
    _atomic_writeto(fits.PrimaryHDU(raw_data, header=raw_header), raw_science_path)
    return raw_science_path


def _remove_if_exists(path: Path) -> None:
    """Best-effort unlink of ``path``; never raises (auditable failure only)."""
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _atomic_writeto(hdu, path):
    """Write a FITS HDU atomically (temp file + rename) so no half-written file
    is silently presented as valid.

    Guarantees (ZM-ZEGRID-R24 A2):

    * on success, only the final target exists (no ``<target>.tmp`` remains);
    * if the temp write OR the final rename raises, the ``<target>.tmp`` is
      removed best-effort and a pre-existing valid target is left untouched
      (the target is only ever replaced by the atomic ``os.replace``, which
      runs strictly after a fully successful temp write).
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        hdu.writeto(tmp, overwrite=True)
    except Exception:
        _remove_if_exists(tmp)
        raise
    try:
        os.replace(tmp, path)
    except Exception:
        _remove_if_exists(tmp)
        raise


# Fixed output names that are NEVER a valid aesthetic companion (so a malicious/
# hand-edited prior manifest can never convince the stale cleaner to delete a
# science/coverage/uint16/log file under a fixed reserved name).
_RESERVED_NON_AESTHETIC = {
    "mosaic_grid.fits",
    "mosaic_grid_coverage.fits",
    "mosaic_grid_uint16.fits",
    RUN_LOG_NAME,
}


def _scientific_suffix_from_name(name):
    """Return the clean suffix of a ``mosaic_grid<suffix>.fits`` science name.

    ZM-ZEGRID-R25: a candidate name matching a SCIENTIFIC output pattern must
    never be removed. ``name`` is a recognised science-pattern name when it has
    the ``mosaic_grid*.fits`` shape with a NON-EMPTY middle suffix, in which
    case that middle suffix is returned (e.g. ``_science`` for
    ``mosaic_grid_science.fits``). Returns ``None`` otherwise — including the
    bare primary-science name ``mosaic_grid.fits`` (empty suffix, reserved
    separately via ``_RESERVED_NON_AESTHETIC``).
    """
    if not isinstance(name, str):
        return None
    if not (name.startswith("mosaic_grid") and name.endswith(".fits")):
        return None
    suffix = name[len("mosaic_grid"):-len(".fits")]
    return suffix or None


def _cleanup_stale_aesthetic(
    output_dir, current_aesthetic_path, protected_names=(), scientific_suffix=None
):
    """Best-effort removal of a prior manifest-declared stale aesthetic FITS.

    ZM-ZEGRID-R24 (A2): on a rerun into the same output directory, a previous
    run may have left an aesthetic companion (``mosaic_grid*.fits``) that is no
    longer a current output (export checkbox turned OFF, or the aesthetic suffix
    changed). Read ONLY the prior ``zegrid_manifest.json``; after the current
    outputs are successfully written and before the current manifest is
    published, remove that one prior-declared aesthetic FITS — and nothing else.

    Strict validation before ANY deletion:

    * the declared name must be a clean basename inside ``output_dir`` (no path
      separators, no ``..``, no absolute path);
    * it must be a recognised ``mosaic_grid*.fits`` name with a NON-EMPTY suffix
      (the bare ``mosaic_grid.fits`` primary-science name and the fixed coverage/
      uint16/log names are never deleted);
    * it must not collide with any current output name (raw science / coverage /
      uint16 / aesthetic / run log);
    * ZM-ZEGRID-R25 (L1): it must NOT match a SCIENTIFIC output pattern —
      ``mosaic_grid<scientific_suffix>.fits`` for the prior manifest's declared
      scientific suffix (when it records one) or the current run's resolved
      ``scientific_fits_suffix`` (default ``_science``). Both suffixes are
      honoured, so a corrupt/hand-edited prior manifest can never convince the
      cleaner to delete a raw science output, whatever name it declares.

    Never glob-deletes, never touches science/coverage/uint16/user files, and
    never a path outside ``output_dir``. Failures are recorded (never raised) so
    the manifest ``stale_cleanup`` block / run log can surface them.
    """
    record: dict = {
        "ran": True,
        "removed": [],
        "kept": [],
        "warnings": [],
    }
    output_dir = Path(output_dir)
    manifest_path = output_dir / "zegrid_manifest.json"
    if not manifest_path.exists():
        record["reason"] = "no_prior_manifest"
        return record
    try:
        prior = json.loads(manifest_path.read_text())
    except Exception as exc:
        record["reason"] = "prior_manifest_unreadable"
        record["warnings"].append(f"could not read prior manifest: {exc}")
        return record
    if not isinstance(prior, dict):
        record["reason"] = "prior_manifest_not_object"
        return record

    prior_name = None
    outputs = prior.get("outputs")
    if isinstance(outputs, dict):
        prior_name = outputs.get("aesthetic")
    if not prior_name:
        soc = prior.get("science_output_contract") or {}
        aest = soc.get("aesthetic") if isinstance(soc, dict) else None
        # ZM-ZEGRID-R25 (L1): only trust the contract's aesthetic record when it
        # actually identifies an aesthetic (role == "AESTH").
        if isinstance(aest, dict) and aest.get("role") == "AESTH":
            prior_name = aest.get("file")
    if not prior_name or not isinstance(prior_name, str):
        record["reason"] = "no_prior_aesthetic_declared"
        return record

    # ZM-ZEGRID-R25 (L1): resolve every SCIENTIFIC output suffix that must NEVER
    # be removed — the prior manifest's declared scientific suffix (when it
    # records one) and the current run's resolved scientific suffix (default
    # ``_science``). A name matching any of these patterns is a raw-science
    # output and is refused unconditionally.
    science_suffixes: set[str] = set()
    _soc = prior.get("science_output_contract")
    if isinstance(_soc, dict):
        _raw = _soc.get("raw")
        if isinstance(_raw, dict):
            _sfx = _scientific_suffix_from_name(_raw.get("file"))
            if _sfx:
                science_suffixes.add(_sfx)
    _outputs = prior.get("outputs")
    if isinstance(_outputs, dict):
        _sfx = _scientific_suffix_from_name(_outputs.get("science"))
        if _sfx:
            science_suffixes.add(_sfx)
    _sfx = _scientific_suffix_from_name(prior.get("science_reference"))
    if _sfx:
        science_suffixes.add(_sfx)
    science_suffixes.add(
        scientific_suffix
        if isinstance(scientific_suffix, str) and scientific_suffix
        else "_science"
    )

    prior_name = prior_name.strip()
    current_name = (
        Path(current_aesthetic_path).name if current_aesthetic_path is not None else None
    )
    if current_name == prior_name:
        record["reason"] = "prior_aesthetic_is_current"
        record["kept"].append(prior_name)
        return record

    problems: list[str] = []
    if not prior_name or prior_name in {"", ".", ".."}:
        problems.append("empty or reserved name")
    if Path(prior_name).name != prior_name or "/" in prior_name or "\\" in prior_name:
        problems.append("not a clean basename (contains a path separator)")
    if not (prior_name.startswith("mosaic_grid") and prior_name.endswith(".fits")):
        problems.append("not a recognised mosaic_grid*.fits name")
    if prior_name in _RESERVED_NON_AESTHETIC:
        problems.append("fixed reserved (non-aesthetic) output name")
    if _scientific_suffix_from_name(prior_name) in science_suffixes:
        problems.append("matches a scientific output name (never removed)")
    if prior_name in set(protected_names or ()):
        problems.append("collides with a current output name")

    if problems:
        record["reason"] = "invalid_prior_declaration"
        record["warnings"].append(
            f"refusing to remove prior-declared aesthetic {prior_name!r}: "
            + "; ".join(problems)
        )
        record["kept"].append(prior_name)
        return record

    target = output_dir / prior_name
    try:
        if target.exists():
            target.unlink()
            record["removed"].append(prior_name)
        else:
            record["reason"] = "prior_aesthetic_absent"
            record["kept"].append(prior_name)
    except Exception as exc:
        record["reason"] = "removal_failed"
        record["warnings"].append(f"could not remove {prior_name!r}: {exc}")
        record["kept"].append(prior_name)
    return record


def _write_outputs(
    assembled, canvas, nx, ny, output_dir, descs, manifests, cell_records,
    layout, science_config, peak_rss_kib, cache_report, progress_callback,
    rejected=None, sip_mode="keep", frames_loaded=None,
    global_reference_frame_id=None,
    timings=None, gpu_used=False, ignored_settings=None, ignored_run_args=None,
    gauge_diagnostics=None,
    per_cell_diagnostics=None,
    auto_gauge_info=None,
    cell_feather_info=None,
    finished_science=None, finishing_info=None, fin_uint16=None,
    gpu_context=None,
    aggregate_peak_rss_kib=None,
    raw_science_path=None,
    raw_science=None,
    aesthetic_path=None,
    export_aesthetic_fits=False,
    scientific_fits_suffix=None,
    halo_provenance=None,
    r55_scheduler=None,
):
    output_dir = Path(output_dir)
    # ZM-ZEGRID-R18: use the finished (aesthetic) science when provided (bit-equal
    # to the raw path when finishing is disabled -> ``finished_science`` is the same
    # array). The RAW science is written first (immutable reference), and the
    # aesthetic float is written ONLY when ``export_aesthetic_fits`` (Classic naming).
    science = np.asarray(
        assembled.science if finished_science is None else finished_science,
        dtype=np.float32,
    )  # (H, W, 3)
    # ZM-FITS-INTEROP-R26: serialize ZeGrid coverage as contiguous float32
    # (BITPIX=-32, no BSCALE/BZERO) so ASIFitsView Linux / Gwenview can read it;
    # the exact pre-fix integer stack-depth values (0..46) are preserved.
    stack_depth = np.ascontiguousarray(
        np.asarray(assembled.stack_depth, dtype=np.float32)
    )  # (H, W)

    sci_data = np.ascontiguousarray(np.moveaxis(science, -1, 0))  # (3, H, W)
    # ZM-FITS-INTEROP-R26: sanitize non-finite (NaN/+Inf/-Inf -> 0.0) in the
    # serialized science buffer ONLY (never the in-memory ``science``/raw
    # arrays), and keep the count for the auditable header/manifest.
    sci_data, _sci_nf_replaced = sanitize_nonfinite_float32(sci_data, fill_value=0.0)
    _raw_ref_chw = np.ascontiguousarray(np.moveaxis(
        np.asarray(assembled.science if raw_science is None else raw_science,
                   dtype=np.float32), -1, 0))
    _raw_ref_chw, _raw_nf_replaced = sanitize_nonfinite_float32(_raw_ref_chw, fill_value=0.0)

    # ZM-ZEGRID-R23 rework-3 (H1): Classic-compatible naming on base ``mosaic_grid``.
    # ``mosaic_grid<clean scientific suffix>.fits`` is the immutable pre-finishing
    # reference (SCI role, written by ``_run_single`` before finishing);
    # ``mosaic_grid<clean aesthetic suffix>.fits`` is the delivered aesthetic
    # float32 (AESTH role), written only when ``export_aesthetic_fits``.
    if raw_science_path is None:
        raw_science_path = output_dir / "mosaic_grid.fits"
        _write_raw_science_fits(
            assembled, canvas, raw_science_path,
            halo_provenance=halo_provenance,
        )
    raw_science_path = Path(raw_science_path)

    if aesthetic_path is not None:
        aesthetic_path = Path(aesthetic_path)

    dbe_applied = bool((finishing_info or {}).get("dbe", {}).get("applied"))
    dbe_attempted = bool((finishing_info or {}).get("dbe", {}).get("attempted"))
    dbe_enabled = bool((finishing_info or {}).get("dbe", {}).get("enabled"))
    finishing_failed = bool((finishing_info or {}).get("failed"))
    if finishing_failed:
        dbe_state = "failed"
    elif dbe_applied:
        dbe_state = "on"
    elif dbe_attempted:
        dbe_state = "noop"
    else:
        dbe_state = "off"

    # Aesthetic float is emitted only when the checkbox is set.
    sci_path = aesthetic_path if aesthetic_path is not None else raw_science_path
    if aesthetic_path is not None:
        # sci_data is already non-finite-sanitized (ZM-FITS-INTEROP-R26).
        sci_header = _science_header(
            canvas,
            role="AESTH",
            dtype="float32",
            dbe_state=dbe_state,
            sha256=_array_sha256(sci_data),
            related_file=raw_science_path.name,
        )
        _record_halo_cards(sci_header, halo_provenance)
        record_nonfinite_fill(sci_header, _sci_nf_replaced)
        _atomic_writeto(fits.PrimaryHDU(sci_data, header=sci_header), aesthetic_path)
    elif export_aesthetic_fits:
        # Checkbox set but no aesthetic path resolved -> do not fabricate a file.
        pass

    # ZM-ZEGRID-R18: optional uint16 render (save_final_as_uint16). Derived from
    # the post-aesthetic branch; clearly non-scientific.
    uint16_path = None
    if fin_uint16 is not None:
        u16 = np.asarray(fin_uint16, dtype=np.uint16)
        u16_header = _canvas_header(canvas, ndim=3, channels=3)
        u16_header["BUNIT"] = ("adu16", "uint16 render (see finishing.uint16)")
        u16_header["SCIROLE"] = ("uint16_render", "derived render, NOT a science reference")
        _record_halo_cards(u16_header, halo_provenance)
        _u16_info = (finishing_info or {}).get("uint16", {})
        if "vmin" in _u16_info:
            u16_header["U16VMIN"] = (float(_u16_info["vmin"]), "scaling vmin (float ADU)")
            u16_header["U16VMAX"] = (float(_u16_info["vmax"]), "scaling vmax (float ADU)")
        uint16_path = output_dir / "mosaic_grid_uint16.fits"
        _atomic_writeto(
            fits.PrimaryHDU(
                np.ascontiguousarray(np.moveaxis(u16, -1, 0)), header=u16_header
            ),
            uint16_path,
        )

    cov_header = _canvas_header(canvas, ndim=2)
    cov_header["BUNIT"] = ("count", "per-pixel stack depth (max over channels)")
    _record_halo_cards(cov_header, halo_provenance)
    cov_path = output_dir / "mosaic_grid_coverage.fits"
    _atomic_writeto(fits.PrimaryHDU(stack_depth, header=cov_header), cov_path)

    # ZM-ZEGRID-R24 (A2): conservative stale-aesthetic cleanup. Runs ONLY after
    # every current output above has been successfully written and BEFORE the
    # current manifest is published, so a failed current run never erases the
    # prior valid aesthetic (its manifest is still intact on disk). Read-only on
    # the prior manifest; never glob-deletes and never touches science/coverage/
    # uint16/user files or anything outside ``output_dir``.
    stale_cleanup = _cleanup_stale_aesthetic(
        output_dir,
        aesthetic_path,
        protected_names={
            raw_science_path.name,
            cov_path.name,
            RUN_LOG_NAME,
            *([uint16_path.name] if uint16_path is not None else []),
            *([aesthetic_path.name] if aesthetic_path is not None else []),
        },
        scientific_suffix=scientific_fits_suffix,
    )

    # ZM-ZEGRID-R24 (A1): the ``science_output_contract`` block is built here so
    # the ``aesthetic`` sub-record can be OMITTED ENTIRELY when no aesthetic FITS
    # is emitted (no ``file=None`` + in-memory SHA). ``outputs.aesthetic`` stays
    # an explicit ``null`` for a stable top-level schema; only the contract's
    # aesthetic truth is dropped when absent.
    science_output_contract = {
        "note": (
            "ZM-ZEGRID-R23 rework-3: 'science'/'science_reference' is the "
            "immutable pre-finishing assembled science (the scientific/"
            "photometric reference, always written); 'aesthetic' is the "
            "delivered legacy light-DBE aesthetic float32 (written only when "
            "export_aesthetic_fits is true; bit-identical to raw when finishing "
            "is disabled). 'uint16' is only an optional render derived from the "
            "aesthetic float32, never a scientific reference."
        ),
        "raw": {
            "file": raw_science_path.name,
            "role": "SCI",
            "dtype": "float32",
            "sha256": _array_sha256(_raw_ref_chw),
            "nonfinite_replaced": int(_raw_nf_replaced),
            "nonfinite_fill": NONFINITE_FILL_VALUE,
        },
    }
    if aesthetic_path is not None:
        science_output_contract["aesthetic"] = {
            "file": aesthetic_path.name,
            "role": "AESTH",
            "dtype": "float32",
            "dbe_state": dbe_state,
            "sha256": _array_sha256(sci_data),
        }

    # Cache accounting: prefer the explicit cache_report (per-cell temp reuse)
    # and fall back to summing the manifests (legacy persistent-cache path).
    cache_total_bytes = (cache_report or {}).get("total_bytes")
    cache_files = (cache_report or {}).get("n_frame_entries")
    if cache_total_bytes is None or cache_files is None:
        _tb = 0
        _cf = 0
        for _cid, m in (manifests or {}).items():
            if isinstance(m, dict):
                _tb += int(m.get("total_bytes", 0))
                _cf += int(m.get("n_frames", 0))
        cache_total_bytes = _tb if cache_total_bytes is None else cache_total_bytes
        cache_files = _cf if cache_files is None else cache_files
    cache_peak_bytes = (cache_report or {}).get("peak_cell_bytes", cache_total_bytes)
    cache_retention = (cache_report or {}).get("retention")

    # ZM-ZEGRID-R53b F-2: defensive cross-check at manifest publication. The
    # primary pre-publication check runs in ``_run_single`` BEFORE raw science.
    _validate_halo_provenance_consistency(
        halo_provenance, science_config, cell_feather_info
    )

    _effective_gauge_mode = (
        str(halo_provenance.get("effective_gauge_mode"))
        if halo_provenance is not None
        else str(getattr(science_config, "cell_gauge_mode", "global"))
    )

    manifest = {
        "schema": "zemosaic-zegrid-r7-manifest-v1",
        "engine": "zegrid",
        "normalization": science_config.normalization,
        "normalization_default": DEFAULT_NORMALIZATION,
        "canvas": {"width": canvas.width, "height": canvas.height,
                   "resolution_deg": canvas.resolution_deg, "id": canvas.canvas_id},
        "layout": {"nx": nx, "ny": ny,
                   "ram_budget_bytes": layout["ram_budget_bytes"],
                   "available_bytes": layout["available_bytes"],
                   "max_contributors": layout["max_contributors"],
                   "predicted_bound_bytes": layout["predicted_bound_bytes"]},
        "layout_source": layout.get("layout_source", "auto"),
        # ZM-ZEGRID-R53b: ONE stable top-level provenance block (same shape for
        # global / per_cell / auto outcomes). It is resolved once at the product
        # entry and threaded here explicitly; it never re-reads config.
        **(
            {"zegrid_halo": dict(halo_provenance)}
            if halo_provenance is not None else {}
        ),
        "reproducibility_note": _reproducibility_note(_effective_gauge_mode),
        "photometric_gauge": {
            "mode": getattr(science_config, "cell_gauge_mode", "global"),
            "requested_mode": (
                (auto_gauge_info or {}).get("requested")
                or getattr(science_config, "cell_gauge_mode", "global")
            ),
            # ZM-CELL-PHOTOMETRY-R34F: the run-level "auto" decision (resolved
            # effective mode + trigger), or {"enabled": False} when "auto" was
            # not requested. This is the audit trail of the self-healing default.
            "auto_selection": (
                auto_gauge_info if auto_gauge_info is not None else {"enabled": False}
            ),
            "global_reference_frame_id": global_reference_frame_id,
            "note": (
                "one reference frame + per-frame sky_mean coefficients computed once "
                "over the full canvas footprint; every Cell reuses the same gauge "
                "(Cells lacking the global reference keep its GLOBAL level)."
                if getattr(science_config, "cell_gauge_mode", "global") == "global"
                else (
                    "OPT-IN per-Cell gauge (ZM-CELL-PHOTOMETRY-R34B): every Cell "
                    "self-anchors on its OWN greatest-support frame via the FROZEN "
                    "estimator; global_reference_frame_id is the global-gauge "
                    "reference retained for provenance only. Inter-cell coherence "
                    "(R34c offset graph) and feathering (R34d) are NOT applied here."
                )
            ),
            "provenance_honesty": (
                "ZM-ZEGRID-R11 L1: a Cell's 'reference_frame_id' is the TRUE "
                "photometric anchor only when it equals global_reference_frame_id. "
                "When a Cell lacks the global reference, its reference_frame_id is "
                "a BOOKKEEPING placeholder (recorded in "
                "'cells[].bookkeeping_reference_frame_id' with "
                "reference_frame_role='bookkeeping_placeholder'); the true "
                "photometric anchor is always "
                "photometric_gauge.global_reference_frame_id."
                if getattr(science_config, "cell_gauge_mode", "global") == "global"
                else (
                    "ZM-CELL-PHOTOMETRY-R34B: in per_cell mode each Cell's "
                    "'reference_frame_id' IS its TRUE local photometric anchor "
                    "(reference_frame_role='cell_local_photometric_anchor'); Cells "
                    "are NOT forced onto a shared global level (global coherence is "
                    "R34c/R34d)."
                )
            ),
        },
        # ZM-CELL-PHOTOMETRY-R34D: inter-cell additive offset + feather diagnostics
        # (present only when the opt-in per_cell feather path ran).
        "cell_feather": (
            {
                "enabled": True,
                "anchor_cell": cell_feather_info.get("anchor"),
                "channels": cell_feather_info.get("channels"),
                "halo_px": cell_feather_info.get("halo_px"),
                "edges_used": cell_feather_info.get("edges_used"),
                "edges_built": cell_feather_info.get("edges_built"),
                "weighted_residual_rms": cell_feather_info.get("weighted_residual_rms"),
                "disconnected_cells": cell_feather_info.get("disconnected"),
                "disconnected_policy": cell_feather_info.get("policy"),
                "offsets": cell_feather_info.get("offsets"),
                "note": (
                    "ADDITIVE-ONLY per-channel offsets (one scalar per Cell), global "
                    "least-squares anchored on the central Cell; disconnected Cells "
                    "keep offset 0 (recorded, no re-anchor, no bridge). Science is "
                    "weighted-feather blended (weights sum to 1); no gain/plane."
                ),
                # ZM-ZEGRID-R37-D: compact diagnostic-capture metadata, present
                # ONLY when the explicit capture opt-in ran (disabled runs keep
                # the exact previous manifest shape).
                **(
                    {"r37_diagnostic_capture": cell_feather_info["r37_diagnostic_capture"]}
                    if cell_feather_info.get("r37_diagnostic_capture") is not None
                    else {}
                ),
            }
            if cell_feather_info is not None
            else {"enabled": False}
        ),
        "n_frames": len(descs),
        "frame_ids": [d.frame_id.logical_path for d in descs],
        "sip_mode": sip_mode,
        "rejected_frames": {
            "count": len(rejected or []),
            "by_reason": (
                {reason: sum(1 for r in rejected if r["reason"] == reason)
                 for reason in sorted({r["reason"] for r in rejected})}
                if rejected else {}
            ),
            "paths": [r["path"] for r in (rejected or [])],
        },
        "reconciliation": {
            "frames_loaded": frames_loaded if frames_loaded is not None else (
                len(descs) + len(rejected or [])
            ),
            "frames_included": len(descs),
            "frames_rejected_wcs": len(rejected or []),
            "note": (
                "frames_loaded == frames_included + frames_rejected_wcs "
                "(no silent drop outside the WCS gate; ZM-ZEGRID-R10 I1)"
            ),
        },
        "complete_cells": assembled.complete_cells,
        "incomplete_cells": assembled.incomplete_cells,
        "hole_pixels": assembled.hole_pixels,
        "coverage_pixels": assembled.coverage_pixels,
        "cells": cell_records,
        "cache": {"dir": CACHE_DIR_NAME, "total_bytes": int(cache_total_bytes),
                  "peak_cell_bytes": int(cache_peak_bytes),
                  "retention": cache_retention,
                  "n_frame_entries": int(cache_files),
                  "reuse_identity": {
                      "bound_to": ["patch_shape_hw", "patch_bounds", "patch_wcs"],
                      "note": (
                          "R53a cache reuse is GEOMETRY-BOUND: a cell cache is "
                          "reusable only when its patch shape (HW), global patch "
                          "bounds and patch WCS identity match the current patch. "
                          "A cache without this identity (or with a different one) "
                          "is never reused under the product expectation."
                      ),
                  },
                  "cleanup_failures": {
                      "count": len((cache_report or {}).get("cleanup_failures", [])),
                      "paths": list((cache_report or {}).get("cleanup_failures", [])),
                  },
                  "reused_cells": (cache_report or {}).get("reused", []),
                  "rebuilt_cells": (cache_report or {}).get("rebuilt", [])},
        "timings": (timings.to_dict() if timings is not None else {}),
        "gpu": _manifest_gpu_block(gpu_used, gpu_context),
        "ignored_settings": (ignored_settings or {}),
        "ignored_run_args": (ignored_run_args or {}),
        "reprojection": zxe.merge_reproject_stats(
            (gauge_diagnostics or {}).get("reprojection"),
            (per_cell_diagnostics or {}).get("reprojection"),
        ),
        "gauge_diagnostics": (gauge_diagnostics or {}),
        "per_cell_diagnostics": (per_cell_diagnostics or {}),
        # ZM-ZEGRID-R41-B: the 80 %-plan / 90 %-admit host-RAM contract trail is
        # ALSO published at TOP LEVEL (the per_cell_diagnostics copy is retained
        # for backward compatibility).
        "ram_contract": (
            (per_cell_diagnostics or {}).get("ram_contract")
            if isinstance(per_cell_diagnostics, dict) else None
        ),
        # ZM-ZEGRID-R36: the RAM-adaptive CPU streaming-tile decision (clear,
        # auditable block; on a comfortable machine selected_tile_size=128 with an
        # explicit reason). NEVER a bare ``tile_size``.
        "cpu_streaming": (
            layout.get("cpu_streaming")
            if isinstance(layout, dict) and layout.get("cpu_streaming")
            else {
                "adaptive": False,
                "preferred_tile_size": int(CPU_STREAMING_TILE_PREFERRED),
                "selected_tile_size": None,
                "minimum_tile_size": int(CPU_STREAMING_TILE_MINIMUM),
                "selection_reason": "no CPU streaming decision recorded for this run",
                "predicted_peak_bytes": None,
                "ram_budget_bytes": None,
            }
        ),
        "finishing": (finishing_info or {}),
        "peak_rss_kib": peak_rss_kib,
        "peak_rss_kib_note": (
            "peak RSS of a SINGLE process (RUSAGE_SELF on Linux / current RSS on "
            "Windows); see aggregate_peak_rss_kib for the parent+worker batch bound"
        ),
        "aggregate_peak_rss_kib": aggregate_peak_rss_kib,
        "aggregate_peak_rss_kib_note": (
            "parent + sum of cell-worker peak RSS (UPPER BOUND: each worker includes "
            "its own ~340 MiB import baseline; shared/copy-on-write pages are not "
            "deduplicated)"
        ),
        "outputs": {
            "science": raw_science_path.name,
            "aesthetic": (aesthetic_path.name if aesthetic_path is not None else None),
            "coverage": cov_path.name,
            "uint16": (uint16_path.name if uint16_path is not None else None),
            "run_log": RUN_LOG_NAME,
        },
        "science_reference": raw_science_path.name,
        "algorithm": ("legacy_grid_light_dbe" if dbe_applied else None),
        "aesthetic_warning": (
            f"aesthetic output ({aesthetic_path.name}) is not photometrically "
            f"neutral; use science_reference ({raw_science_path.name}) for "
            "measurement" if aesthetic_path is not None else None
        ),
        "science_output_contract": science_output_contract,
        "stale_cleanup": stale_cleanup,
        # ZM-ZEGRID-R55-B: ADDITIVE observe-only scheduler block. ``None`` in
        # legacy mode -> the exact pre-R55 manifest shape is preserved.
        **({"r55_scheduler": dict(r55_scheduler)} if r55_scheduler is not None else {}),
    }
    # ZM-ZEGRID-R40-B2-B: exactly ONE additive ``r40_diagnostic_capture`` manifest
    # block, produced by the accepted core (pre-existing fields canonical-identical
    # ON/OFF; NEVER a whole-manifest byte-identity claim). OFF -> no new field at
    # all, so a disabled run keeps the exact previous manifest shape.
    _r40_block = (cell_feather_info or {}).get("r40_diagnostic_capture")
    if _r40_block is not None:
        from .core.zegrid import r40_capture as _r40c

        manifest = _r40c.build_manifest_link(
            manifest,
            capture_id=str(_r40_block["capture_id"]),
            filename=str(_r40_block["filename"]),
            sha256=str(_r40_block["sha256"]),
            nbytes=int(_r40_block["bytes"]),
        )
    manifest_path = output_dir / "zegrid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return sci_path, cov_path, manifest_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_zegrid_mode(
    input_folder: str,
    output_folder: str,
    progress_callback: ProgressCallback = None,
    *,
    stack_norm_method: str = "sky_mean",
    stack_weight_method: str = "noise_variance",
    stack_reject_algo: str = "kappa_sigma",
    stack_kappa_low: float = 3.0,
    stack_kappa_high: float = 3.0,
    winsor_limits: tuple[float, float] = (0.05, 0.05),
    stack_final_combine: str = "mean",
    apply_radial_weight: bool = False,
    radial_feather_fraction: float = 0.8,
    radial_shape_power: float = 2.0,
    save_final_as_uint16: bool | None = None,
    legacy_rgb_cube: bool = False,
    grid_rgb_equalize: bool | None = None,
    use_gpu: bool | None = None,
    zconfig: object | None = None,
    workers: int | None = None,
    cell_gauge_mode: str | None = None,
    zegrid_halo_px: int | str | None = None,
    zegrid_scheduler_mode: str | None = None,
    frames_info: Sequence | None = None,
    selection_paths: Sequence[str] | None = None,
    global_wcs_output_path: str | None = None,
    r37_diagnostic_capture: bool = False,
    r37_diagnostic_holdout_cap: int = R37_CAPTURE_HOLDOUT_CAP_DEFAULT,
    r40_seam_capture: bool = False,
    r40_output_root: str | None = None,
    r40_seam_capture_disk_budget_bytes: int | None = None,
) -> None:
    """Run the NEW ZeGrid engine over the input folder's FITS inventory.

    ZM-ZEGRID-R35A: ZeGrid is an EXPLICIT mode and is fed by the eligible FITS
    inventory of ``input_folder`` (optionally narrowed by an explicit Filter
    ``selection_paths``); ``stack_plan.csv`` no longer activates or feeds it.
    ``frames_info`` may be supplied directly (worker-built) to avoid a rescan.

    Normalization defaults to ``sky_mean``;
    weighting/rejection/combine/taper use the frozen ZeGrid science config.

    ZM-ZEGRID-R37-D: ``r37_diagnostic_capture`` (default ``False``) is an explicit,
    diagnostic-only opt-in. When requested it publishes the accepted R37-C strict
    JSON pre-offset/pre-feather artifact into EACH actual output directory and
    records compact capture metadata in the manifest/run log. It never applies a
    correction and never changes the science/aesthetic/coverage outputs; the
    disabled path never imports, builds or writes the diagnostic. Capture requires
    the effective ``cell_gauge_mode`` to be ``per_cell`` (``auto`` is honoured when
    it resolves to per_cell; a global resolution fails loudly).
    """
    _emit("ZeGrid engine activated (explicit ZeGrid mode)", callback=progress_callback)

    # ZM-ZEGRID-R53: resolve the anti-checkerboard halo width FIRST, so an invalid
    # request fails before descriptors, layouts, gauge work or any output. The
    # precedence matches ``cell_gauge_mode``: explicit keyword > config key >
    # product default. No global is mutated; the validated int is threaded down.
    _halo_config_value = (
        getattr(zconfig, ZEGRID_HALO_CONFIG_KEY, None)
        if zconfig is not None else None
    )
    try:
        halo_provenance = resolve_run_halo_provenance(
            zegrid_halo_px, _halo_config_value
        )
    except ZeGridHaloError as exc:
        _emit(f"ZeGrid: {exc}", lvl="ERROR", callback=progress_callback)
        raise
    resolved_halo_px = int(halo_provenance["effective_px"])
    _requested_for_log = (
        "absent" if halo_provenance["requested_px"] is None
        else str(halo_provenance["requested_px"])
    )
    _emit(
        "ZeGrid: anti-checkerboard halo — requested="
        f"{_requested_for_log} -> effective={resolved_halo_px} px "
        f"(source={halo_provenance['source']}); this width drives patch geometry, "
        "contributor membership, gauge patch domain, measurement band=2H and "
        "feather=H in per_cell mode",
        callback=progress_callback,
    )

    # ZM-ZEGRID-R55-B: resolve the scheduler mode ONCE at the product entry.
    # Absent -> legacy (the historical behaviour). An explicit unknown value
    # fails visibly; it is never silently coerced to the default.
    _scheduler_config_value = (
        getattr(zconfig, "zegrid_scheduler_mode", None)
        if zconfig is not None else None
    )
    _scheduler_mode = zsched.normalize_scheduler_mode(
        zegrid_scheduler_mode
        if zegrid_scheduler_mode is not None else _scheduler_config_value
    )
    _emit(
        f"ZeGrid: scheduler mode = {_scheduler_mode} "
        + (
            "(R55 observe-only: records a predicted plan; execution remains the "
            "legacy path)"
            if _scheduler_mode == zsched.MODE_OBSERVE
            else "(legacy default: R55 scheduler disabled)"
        ),
        callback=progress_callback,
    )

    # ZM-ZEGRID-R37-D: validate the capture opt-in strictly and up front so a
    # mistyped request fails before any work.
    r37_diagnostic_capture, r37_diagnostic_holdout_cap = _validate_r37_diagnostic_capture(
        r37_diagnostic_capture, r37_diagnostic_holdout_cap
    )

    # ZM-ZEGRID-R40-B2-B: validate the R40 seam-capture opt-in strictly and up
    # front (no silent coercion; a capture that cannot publish is refused before
    # any work). The core is imported lazily, ONLY when the capture is enabled.
    (
        r40_seam_capture, r40_output_root, r40_seam_capture_disk_budget_bytes,
    ) = _validate_r40_seam_capture(
        r40_seam_capture, r40_output_root, r40_seam_capture_disk_budget_bytes
    )
    if r40_seam_capture and not r40_output_root:
        r40_output_root = output_folder

    # ZM-ZEGRID-R37-D (F1): when a capture is REQUESTED, fail-closed invalidate any
    # stale fixed artifact at run START — in the base output dir AND the known
    # mount-pass dirs (grid_EQ / grid_ALTZ) if they exist — BEFORE explicit-global
    # refusal, input discovery, descriptors, layout or gauge work. A pre-publication
    # abort therefore cannot leave a previous committed-looking artifact. The
    # DISABLED path touches nothing.
    if r37_diagnostic_capture:
        from .core.zegrid import background_diagnostics as _bgd

        _base_out = Path(output_folder).expanduser()
        _bgd.invalidate_capture_target(_base_out)
        for _pass_dir in ("grid_EQ", "grid_ALTZ"):
            _candidate = _base_out / _pass_dir
            if _candidate.is_dir():
                _bgd.invalidate_capture_target(_candidate)

    # ZM-ZEGRID-R22: resolve ONE canonical GPU preference and decide the backend
    # HONESTLY (requested -> available -> effective). ``use_gpu`` is now honoured
    # (never silently dropped); an unavailable/undersized GPU degrades loudly to
    # exact CPU with a recorded reason.
    gpu_requested, gpu_source = resolve_gpu_preference(use_gpu, zconfig)
    gpu_probe = zgpu.probe_gpu_backend()
    vram_budget = zgpu.vram_budget_bytes(gpu_probe)
    gpu_effective = False
    gpu_fallback_reason = None
    if gpu_requested:
        if not gpu_probe["available"]:
            gpu_fallback_reason = gpu_probe.get("reason") or "gpu_unavailable"
        elif vram_budget is None:
            gpu_fallback_reason = "vram_unknown_or_too_small"
        else:
            gpu_effective = True
    if gpu_requested and not gpu_effective:
        _emit(
            "ZeGrid: GPU requested (use_gpu=True) but NOT usable — "
            f"{gpu_fallback_reason}; degrading to exact CPU (bit-identical output, "
            "slower). No silent CPU fallback: this is recorded in the manifest/log.",
            lvl="WARN",
            callback=progress_callback,
        )
    elif gpu_requested and gpu_effective:
        _emit(
            f"ZeGrid: GPU ENABLED — device={gpu_probe['device']} "
            f"vram_total={gpu_probe['vram_total_bytes']} "
            f"vram_free={gpu_probe['vram_free_bytes']} budget={vram_budget} "
            f"(source={gpu_source})",
            callback=progress_callback,
        )
    else:
        _emit(
            f"ZeGrid: GPU not requested (source={gpu_source}); CPU backend",
            callback=progress_callback,
        )
    gpu_context = {
        "requested": bool(gpu_requested),
        "requested_source": gpu_source,
        "available": bool(gpu_probe["available"]),
        # ZM-ZEGRID-R22 rework-1: distinguish SELECTED (attempted) from EXECUTED
        # (actually_used). ``effective`` stays for back-compat (== attempted at
        # start); ``actually_used``/``final_backend`` are filled in `_run_single`
        # from per-cell execution evidence (H2).
        "attempted": bool(gpu_effective),
        "effective": bool(gpu_effective),
        "final_effective": bool(gpu_effective),
        "actually_used": False,
        "final_backend": "gpu" if gpu_effective else "cpu",
        "gpu_executed_cells": 0,
        "runtime_fallback": None,
        "device": gpu_probe["device"],
        "cupy_version": gpu_probe["cupy_version"],
        "vram_total_bytes": gpu_probe["vram_total_bytes"],
        "vram_free_bytes": gpu_probe["vram_free_bytes"],
        "vram_budget_bytes": vram_budget,
        "fallback_reason": gpu_fallback_reason,
    }

    # ZM-ZEGRID-R16: honour the user's rejection choice (stack_reject_algo +
    # kappa/winsor) where the canonical engine supports it; surface (WARN) anything
    # it cannot honour, never silently drop it. The R12 F2 list keeps the OTHER
    # accepted-but-ignored arguments (weighting/combine/taper/FITS outputs).
    rej_overrides, rej_unhonoured = resolve_rejection_science(
        stack_reject_algo, stack_kappa_low, stack_kappa_high, winsor_limits
    )
    ignored_run_args = {
        "stack_weight_method": stack_weight_method,
        "stack_final_combine": stack_final_combine,
        "apply_radial_weight": apply_radial_weight,
        "radial_feather_fraction": radial_feather_fraction,
        "radial_shape_power": radial_shape_power,
        "legacy_rgb_cube": legacy_rgb_cube,
    }
    ignored_run_args.update(rej_unhonoured)
    for line in zin.describe_ignored_run_args(ignored_run_args):
        _emit(line, lvl="WARN", callback=progress_callback)

    # ZM-ZEGRID-R18: the final-mosaic finishing settings are now HONOURED (they
    # were previously accepted-but-ignored). Resolve them once and surface the
    # resolved choice in the log so nothing is silently dropped.
    finishing_config = zfin.resolve_finishing_config(
        zconfig,
        grid_rgb_equalize=grid_rgb_equalize,
        save_final_as_uint16=save_final_as_uint16,
    )
    _emit(
        f"ZeGrid: final-mosaic finishing — DBE={finishing_config['dbe_enabled']} "
        f"(strength={finishing_config['dbe_strength']}, "
        f"params_source={finishing_config['dbe_params_source']}, "
        f"subtraction_factor={finishing_config['dbe_subtraction_factor']}, "
        f"params={finishing_config['dbe_params']}), rgb_equalize={finishing_config['rgb_equalize']}, "
        f"uint16={finishing_config['save_uint16']}",
        callback=progress_callback,
    )
    # ZM-ZEGRID-R24 (A4): surface any boolean-coercion fallback (unknown/typed
    # string that fell back to the field default) so it is never silent.
    for _fallback in finishing_config.get("bool_coercion_fallbacks", []) or []:
        _emit(
            f"ZeGrid: boolean coercion fallback — field={_fallback['field']!r} "
            f"value={_fallback['value']!r} -> fallback={_fallback['fallback']!r}",
            lvl="WARN",
            callback=progress_callback,
        )

    # ZM-ZEGRID-R35A: ZeGrid input is the eligible FITS inventory of the input
    # folder (optionally narrowed by an EXPLICIT Filter selection). stack_plan.csv
    # neither activates ZeGrid nor truncates its dataset.
    inventory_info: dict = {}
    if frames_info is None:
        from . import input_inventory as _zinv

        frames_info, inventory_info = _zinv.build_zegrid_frames(
            input_folder,
            output_folder,
            global_wcs_output_path=global_wcs_output_path,
            selection_paths=selection_paths,
        )
    _emit(
        f"ZeGrid input discovery: {inventory_info.get('discovered', len(frames_info))} FITS candidate(s)",
        callback=progress_callback,
    )
    if inventory_info.get("selection_applied"):
        _emit(
            f"ZeGrid input selection: explicit Filter selection -> "
            f"{inventory_info.get('selected')} of {inventory_info.get('discovered')} frame(s); "
            f"using {len(frames_info)}",
            callback=progress_callback,
        )
    if inventory_info.get("selection_unmatched"):
        _emit(
            "ZeGrid: explicit Filter selection matched no inventory FITS; "
            "using the FULL inventory (WARN)",
            lvl="WARN",
            callback=progress_callback,
        )
    if not frames_info:
        raise RuntimeError(
            "ZeGrid failed: no eligible FITS frame found in the input folder"
        )

    # Per-Cell photometric gauge mode (ZM-CELL-PHOTOMETRY-R34B): opt-in, default
    # "global" (unchanged production path). Resolution order: explicit keyword >
    # ``zegrid_cell_gauge_mode`` config key > default. An unknown token is a loud
    # error (never a silent photometric-strategy change).
    _mode_source = cell_gauge_mode
    if _mode_source is None and zconfig is not None:
        _mode_source = getattr(zconfig, "zegrid_cell_gauge_mode", None)
    try:
        resolved_cell_gauge_mode = zphot.resolve_cell_gauge_mode(_mode_source)
    except ValueError as exc:
        _emit(f"ZeGrid: {exc}", lvl="ERROR", callback=progress_callback)
        raise
    if resolved_cell_gauge_mode == zphot.CELL_GAUGE_MODE_PER_CELL:
        _emit(
            "ZeGrid: cell_gauge_mode=per_cell (OPT-IN) — every Cell self-anchors on "
            "its own greatest-support frame via the frozen estimator; inter-cell "
            "coherence/feathering are part of this mode (R34c/R34d).",
            lvl="WARN", callback=progress_callback,
        )
    elif resolved_cell_gauge_mode == zphot.CELL_GAUGE_MODE_AUTO:
        _emit(
            "ZeGrid: cell_gauge_mode=auto — the run uses the global gauge when it "
            "serves EVERY non-empty Cell, else falls back to per_cell+offsets+feather "
            "(resolved AFTER the global gauge is computed).",
            callback=progress_callback,
        )
    else:
        _emit(
            "ZeGrid: cell_gauge_mode=global (explicit) — R11 global gauge path.",
            callback=progress_callback,
        )

    # ZM-ZEGRID-R37-D: an EXPLICIT global mode has no pre-feather per-Cell assembly
    # point, so a capture request can never be satisfied — fail loudly and early
    # (before input discovery) rather than claim a capture that will not happen.
    # ``auto`` is deferred to the run-level resolution in ``_run_single``.
    if (
        r37_diagnostic_capture
        and resolved_cell_gauge_mode == zphot.CELL_GAUGE_MODE_GLOBAL
    ):
        raise RuntimeError(
            "R37 diagnostic capture requested but cell_gauge_mode is 'global' (no "
            "pre-offset/pre-feather per-Cell assembly point exists). Refusing to run "
            "rather than silently not capturing; use cell_gauge_mode='per_cell' or "
            "'auto'."
        )

    # ZM-ZEGRID-R40-B2-B: the R40 seam capture only exists on the per_cell path
    # (the only point where complete pre-offset/pre-feather CellPatches coexist).
    # An explicit global mode can never satisfy the request -> fail loudly and early.
    if (
        r40_seam_capture
        and resolved_cell_gauge_mode == zphot.CELL_GAUGE_MODE_GLOBAL
    ):
        raise RuntimeError(
            "R40 seam capture requested but cell_gauge_mode is 'global' (no "
            "pre-offset/pre-feather per-Cell assembly point exists). Refusing to run "
            "rather than silently not capturing; use cell_gauge_mode='per_cell' or "
            "'auto'."
        )

    science_config = replace(
        ExecutorConfig().science_config(),
        normalization=_resolve_normalization(stack_norm_method),
        backend=("gpu" if gpu_effective else "cpu"),
        cell_gauge_mode=resolved_cell_gauge_mode,
        **rej_overrides,
    )
    _emit(
        f"ZeGrid science config: normalization={science_config.normalization} "
        f"(default={DEFAULT_NORMALIZATION}), weighting={science_config.weighting}, "
        f"rejection={science_config.rejection} "
        f"(sigma={science_config.sigma_low:.2f}/{science_config.sigma_high:.2f}, "
        f"winsor={science_config.winsor_limit_low:.3f}/{science_config.winsor_limit_high:.3f}), "
        f"combine={science_config.combine}, taper={science_config.taper}, "
        f"backend={science_config.backend}, cell_gauge_mode={science_config.cell_gauge_mode}",
        callback=progress_callback,
    )

    # PINNABLE layout (reproducibility): read ``zegrid_layout`` from the same
    # config source (zconfig is SimpleNamespace(**worker_config_cache)).
    pinned_layout = None
    if zconfig is not None:
        try:
            pinned_layout = _parse_pinned_layout(getattr(zconfig, "zegrid_layout", None))
        except ValueError as exc:
            _emit(f"ZeGrid: {exc}", lvl="ERROR", callback=progress_callback)
            raise
        if pinned_layout is not None:
            _emit(
                f"ZeGrid: pinned layout {pinned_layout[0]}x{pinned_layout[1]} "
                f"(zegrid_layout); bypassing RAM-adaptive choice",
                callback=progress_callback,
            )

    # SIP mode (ZM-ZEGRID-R10): "keep" (default) applies SIP distortion
    # correctly; "strip" removes it for legacy-consistent plain-TAN behaviour.
    # Default "keep" is justified by the R10 cross-consistency measurement.
    sip_mode = "keep"
    if zconfig is not None:
        candidate = str(getattr(zconfig, "zegrid_sip_mode", "keep") or "keep").strip().lower()
        if candidate not in ("keep", "strip"):
            _emit(
                f"ZeGrid: invalid zegrid_sip_mode={candidate!r} (expected 'keep' or 'strip'); using 'keep'",
                lvl="WARN",
                callback=progress_callback,
            )
            candidate = "keep"
        sip_mode = candidate
    _emit(
        f"ZeGrid: SIP mode={sip_mode} ("
        + ("apply SIP distortion correctly" if sip_mode == "keep" else "strip SIP distortion (legacy-consistent)")
        + ")",
        callback=progress_callback,
    )

    # MOUNT SEGREGATION (same rule as the removed legacy Grid).
    known_mount_frames = [f for f in frames_info if f.mount]
    mount_values = {f.mount for f in known_mount_frames}
    base_out = Path(output_folder).expanduser()

    if len(known_mount_frames) == len(frames_info) and len(mount_values) >= 2:
        eq_frames = [f for f in frames_info if f.mount == "EQ"]
        altz_frames = [f for f in frames_info if f.mount == "ALTZ"]
        _emit(
            f"ZeGrid: mount segregation detected — EQ={len(eq_frames)}, ALTZ={len(altz_frames)}",
            callback=progress_callback,
        )
        if eq_frames:
            _run_single(eq_frames, input_folder, base_out / "grid_EQ",
                        progress_callback=progress_callback,
                        science_config=science_config, zconfig=zconfig,
                        pinned_layout=pinned_layout, sip_mode=sip_mode, workers=workers,
                        ignored_run_args=ignored_run_args,
                        finishing_config=finishing_config,
                        gpu_context=gpu_context, pass_label="EQ",
                        scheduler_mode=_scheduler_mode,
                        halo_px=resolved_halo_px,
                        halo_provenance=halo_provenance,
                        r37_diagnostic_capture=r37_diagnostic_capture,
                        r37_diagnostic_holdout_cap=r37_diagnostic_holdout_cap,
                        r40_seam_capture=r40_seam_capture,
                        r40_output_root=r40_output_root,
                        r40_seam_capture_disk_budget_bytes=r40_seam_capture_disk_budget_bytes)
        if altz_frames:
            _run_single(altz_frames, input_folder, base_out / "grid_ALTZ",
                        progress_callback=progress_callback,
                        science_config=science_config, zconfig=zconfig,
                        pinned_layout=pinned_layout, sip_mode=sip_mode, workers=workers,
                        ignored_run_args=ignored_run_args,
                        finishing_config=finishing_config,
                        gpu_context=gpu_context, pass_label="ALTZ",
                        scheduler_mode=_scheduler_mode,
                        halo_px=resolved_halo_px,
                        halo_provenance=halo_provenance,
                        r37_diagnostic_capture=r37_diagnostic_capture,
                        r37_diagnostic_holdout_cap=r37_diagnostic_holdout_cap,
                        r40_seam_capture=r40_seam_capture,
                        r40_output_root=r40_output_root,
                        r40_seam_capture_disk_budget_bytes=r40_seam_capture_disk_budget_bytes)
    else:
        _emit("ZeGrid: mount info missing or homogeneous — single pass",
              callback=progress_callback)
        _run_single(frames_info, input_folder, base_out,
                    progress_callback=progress_callback,
                    science_config=science_config, zconfig=zconfig,
                    pinned_layout=pinned_layout, sip_mode=sip_mode, workers=workers,
                    ignored_run_args=ignored_run_args,
                    finishing_config=finishing_config,
                    gpu_context=gpu_context, pass_label="single",
                    scheduler_mode=_scheduler_mode,
                    halo_px=resolved_halo_px,
                    halo_provenance=halo_provenance,
                    r37_diagnostic_capture=r37_diagnostic_capture,
                    r37_diagnostic_holdout_cap=r37_diagnostic_holdout_cap,
                    r40_seam_capture=r40_seam_capture,
                    r40_output_root=r40_output_root,
                    r40_seam_capture_disk_budget_bytes=r40_seam_capture_disk_budget_bytes)
