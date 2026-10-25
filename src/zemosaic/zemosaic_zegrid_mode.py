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
import threading
import time
import traceback
from datetime import datetime, timezone
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
from .core.zegrid import storage_profile as zsp
from .core.zegrid import rejection_sink as zrs
from .core.zegrid import gpu_memory as zgm
from .core.zegrid import instrumentation as zin
from .core.zegrid import mosaic as zmosaic
from .core.zegrid import observability as zobs
from .core.zegrid import parallel as zpar
from .core.zegrid import photometric as zphot
from .core.zegrid import ram_contract as rcontract
from .core.zegrid import resource_envelope as renv
from .core.zegrid import runtime_control as zrt
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

# Resolution order (first non-None wins) for the zconfig GPU flags.
# ``use_gpu_phase5`` is the canonical Qt checkbox and deliberately comes first so
# an obsolete ``use_gpu_grid`` value cannot contradict it. The remaining keys are
# legacy fallbacks used only when the canonical key is absent; their historical
# relative order is preserved.
_GPU_PREFERENCE_KEYS = ("use_gpu_phase5", "use_gpu_grid", "stack_use_gpu", "use_gpu_stack")


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
      2. ``use_gpu_phase5`` (GUI canonical checkbox / phase-5 flag);
      3. ``use_gpu_grid`` (legacy grid-specific alias);
      4. ``stack_use_gpu`` (legacy stacking alias);
      5. ``use_gpu_stack`` (legacy alias).

    The canonical ``use_gpu_phase5`` key deliberately outranks the legacy
    ``use_gpu_grid`` alias: an obsolete grid key must never contradict the Qt
    checkbox. When the canonical key is absent, the historical relative order of
    the legacy aliases is preserved.

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
    profile = zsp.normalize_storage_profile(_STACK_STORAGE_PROFILE)
    inmem_bound = zal.FITTED_MEMORY_MODEL.predict_bound_bytes(n_contributors, patch_area_px)
    if not zsp.is_bounded(profile) and inmem_bound <= available_bytes * INMEM_AVAILABLE_FRAC:
        return "inmem", int(inmem_bound)
    stream_bound = _profile_stream_bytes(
        n_contributors, patch_hw, tile_size, profile
    )
    return "stream", int(stream_bound)


# ---------------------------------------------------------------------------
# Mode-aware layout selection (R4 floors + in-memory OR streaming bound)
# ---------------------------------------------------------------------------

def _estimate_streaming_bytes(
    n_contributors, patch_hw, tile_size=STREAM_TILE_SIZE, channels=3,
    storage_profile=zsp.STORAGE_PROFILE_DENSE,
):
    """Streaming per-cell peak bound in BYTES (R6/Lot B estimate, KiB -> bytes)."""
    return int(
        zstream.estimate_streaming_peak_kib(
            n_contributors, patch_hw, tile_size, channels,
            storage_profile=storage_profile,
        ) * 1024
    )


def _profile_stream_bytes(n_contributors, patch_hw, tile_size, storage_profile):
    """Dispatch to the dense estimator WITHOUT changing its call signature.

    The dense path calls ``_estimate_streaming_bytes(n, hw, tile)`` exactly as
    before (so every existing test monkeypatch keeps working); only the bounded
    path passes the explicit keyword.
    """
    profile = zsp.normalize_storage_profile(storage_profile)
    if zsp.is_bounded(profile):
        return _estimate_streaming_bytes(
            n_contributors, patch_hw, tile_size, storage_profile=profile
        )
    return _estimate_streaming_bytes(n_contributors, patch_hw, tile_size)


# ---------------------------------------------------------------------------
# R55-E11 (host-credit correction) — ONE aligned-cache bytes contract.
#
# The aligned cache is a set of .npy files: one float32 RGB cube and one bool
# support plane per frame. ``manifest.total_bytes`` is the sum of the ACTUAL
# file sizes, i.e. data + the 128-byte .npy header of EACH file. The pipeline
# charges exactly that value to its host-staging credit (``_host_bytes``).
# Estimating only RGB DATA under-counted support + headers by ~8.3 % and made a
# prefetch=0 run fail its own first unit before any GPU work (R55F repro).
#
# This single helper is the contract used by the OBSERVE facts, the ACTIVE facts,
# admission, prefetch/disk residency and the runtime ledger. It models the
# ON-DISK cache file bytes (the staging domain); it is NOT a resident-RAM model
# and must not be added to ``cache_worker_footprint_bytes`` (worker RAM domain).
# ---------------------------------------------------------------------------

_R55_CACHE_CHANNELS = 3
_R55_CACHE_RGB_DTYPE_BYTES = 4  # float32
_R55_CACHE_SUPPORT_DTYPE_BYTES = 1  # bool
_R55_CACHE_NPY_HEADER_BYTES = 128  # exact for every supported shape (tested)
_R55_CACHE_BYTES_CONTRACT = (
    "aligned_cache_files__rgb_f32+support_bool+npy_header128__per_frame"
)


def _r55_cache_unit_bytes(
    worst_patch_px, max_n_frames, *, channels: int = _R55_CACHE_CHANNELS
) -> int:
    """Exact upper bound of ONE cell's aligned-cache bytes (manifest contract).

    ``worst_patch_px`` is the worst patch AREA and ``max_n_frames`` the maximum
    contributor count, so every cell's real payload is <= this value. Includes
    the support plane and BOTH .npy headers, matching ``manifest.total_bytes``
    bit-for-bit for a worst-shape cell (verified by test against real files).
    """
    px = max(0, int(worst_patch_px))
    frames = max(0, int(max_n_frames))
    per_frame = (
        int(px)
        * (
            int(channels) * _R55_CACHE_RGB_DTYPE_BYTES
            + _R55_CACHE_SUPPORT_DTYPE_BYTES
        )
        + 2 * _R55_CACHE_NPY_HEADER_BYTES
    )
    return int(per_frame) * int(frames)


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
        stream_b = _profile_stream_bytes(
            n, patch.patch_shape_hw, tile_size, _STACK_STORAGE_PROFILE
        )
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
            b = int(_profile_stream_bytes(n, hw, int(t), _STACK_STORAGE_PROFILE))
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

    ZM-ZEGRID-R55-E6 — bounded COARSER rescue for a large user-selected halo:
    the factor-1 candidate can be rejected SOLELY by ``max_halo_overhead``
    because finer Cells pay proportionally MORE halo. In that exact case (and
    ONLY that case: no earlier rank-admitted candidate, no pinned layout, no
    refinement-rank window), a deterministic, finite family of COARSER grids
    (:func:`auto_layout.coarser_rescue_grids`, nearest coarser first) is
    evaluated: scientific floors are re-checked, then the candidate's EXACT
    mode-aware RAM bound is computed. Coarsening is monotonically better for the
    halo floor every coarser grid has a non-increasing halo overhead) but NOT
    for memory, so a memory-rejected rescue stops the scan and is reported
    truthfully (no "coarser uses less RAM" claim). The chosen layout records
    ``layout_origin="auto_coarser_rescue"`` plus a ``layout_rescue`` provenance
    block; ordinary runs keep ``layout_origin="auto_refinement"`` byte-for-byte.
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
        raise zal.LayoutInfeasible(
            "median projected footprint is degenerate",
            refusal={"kind": "geometry", "source": "degenerate_footprint"},
        )
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
            # ZM-ZEGRID-R55-E6: the pinned geometry is NEVER changed, but the
            # advice must be truthful. Halo overhead grows with refinement, so
            # "pin FINER" is the wrong direction for a halo-floor conflict.
            if not halo_ok and patch_ok:
                _dir = (
                    "Lower the anti-checkerboard halo (e.g. "
                    f"{halo_px} -> {max(1, int(halo_px) // 2)} px) or increase the "
                    "core/cell size; a FINER layout raises halo overhead and cannot "
                    "fix this."
                )
            elif not patch_ok and halo_ok:
                _dir = (
                    "Increase the core/cell size (coarser layout) or the halo-free "
                    "core geometry; a FINER layout reduces patch area and cannot fix "
                    "this."
                )
            else:
                _dir = (
                    "Both floors are violated: increase the core/cell size and/or "
                    "lower the anti-checkerboard halo."
                )
            msg = (
                f"pinned layout {nx}x{ny} violates scientific floors: "
                f"min_patch_area={geom['min_patch_area']} (limit {floors.min_patch_area_px}), "
                f"halo_overhead={geom['halo_overhead']:.4f} (limit {floors.max_halo_overhead}). "
                + _dir
            )
            _log(msg, "ERROR")
            raise zal.LayoutInfeasible(
                msg,
                refusal={"kind": "geometry", "source": "pinned_floors"},
            )
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
            "layout_origin": "pinned",
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
    # ZM-ZEGRID-R55-E6: set ONLY when the FIRST rank-admitted auto candidate is
    # rejected solely by max_halo_overhead (patch floor OK). This is the sole
    # trigger for the bounded coarser rescue; every ordinary run leaves it None.
    halo_conflict = None
    halo_floor_stopped = None
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
                halo_floor_stopped = {
                    "nx": nx, "ny": ny, "factor": factor, "geom": geom,
                }
            if (
                not candidates
                and geom["min_patch_area"] >= floors.min_patch_area_px
                and geom["halo_overhead"] > floors.max_halo_overhead
            ):
                halo_conflict = {
                    "nx": nx, "ny": ny, "factor": factor, "geom": geom,
                }
            _log("layout scan: floors are monotonic in refinement — stopping")
            break  # floors are monotonic in refinement
        worst = _scan_layout_mode_aware(canvas, footprints, nx, ny, halo_px, tile_size)
        selected_tile, trials, sel_reason = _select_cpu_streaming_tile(
            worst["stream_worst_by_tile"], ram_budget
        )
        inmem_ok = (not zsp.is_bounded(_STACK_STORAGE_PROFILE)) and (
            (ram_budget is None) or (worst["inmem_worst_bytes"] <= ram_budget)
        )
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

    if chosen is None and not candidates and halo_conflict is None and (
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
        raise zal.LayoutInfeasible(
            msg,
            refusal={
                # Default diagnosis for a rank-window refusal. The bundle builder
                # upgrades it to ``host_ram/refinement_window`` ONLY when a REAL
                # memory-driven refinement preceded it (the historical
                # ``memory_exhausted`` marker), so a pinned/user window stays a
                # geometry refusal.
                "kind": "geometry",
                "source": "refinement_rank_window",
            },
        )

    # ------------------------------------------------------------------
    # ZM-ZEGRID-R55-E6 — bounded COARSER rescue (large user-selected halo)
    # ------------------------------------------------------------------
    # Trigger is deliberately narrow: the FIRST rank-admitted auto candidate was
    # rejected SOLELY by max_halo_overhead, so no memory candidate was ever
    # evaluated, and no pinned/refinement-rank contract is in force. Coarsening
    # monotonically REDUCES halo overhead (fixed-width halo band vs a larger
    # core), which is the only monotonic claim made here; memory is evaluated
    # exactly per candidate and is never assumed to fall.
    rescue_record = None
    rescue_memory_failure = None
    rescue_trials: list = []
    rescue_grids: tuple = ()
    if chosen is None and halo_conflict is not None:
        _rescue_allowed = (min_refinement_rank is None and max_refinement_rank is None)
        if _rescue_allowed:
            rescue_grids = zal.coarser_rescue_grids(
                canvas, mw, mh, halo_conflict["nx"], halo_conflict["ny"]
            )
            _log(
                f"layout scan: candidate {halo_conflict['nx']}x{halo_conflict['ny']} "
                f"REJECTED SOLELY by max_halo_overhead "
                f"({halo_conflict['geom']['halo_overhead']:.4f} > "
                f"{floors.max_halo_overhead}); attempting bounded COARSER rescue — "
                f"{len(rescue_grids)} deterministic candidate(s), nearest coarser "
                "first (coarsening monotonically lowers halo overhead)"
            )
            if not rescue_grids:
                _log(
                    "layout rescue: no coarser grid exists "
                    f"(rejected candidate is already {halo_conflict['nx']}x"
                    f"{halo_conflict['ny']})"
                )
            for rnx, rny in rescue_grids:
                rgeom = zal._nominal_geometry(canvas, rnx, rny, halo_px)
                rpatch_ok = rgeom["min_patch_area"] >= floors.min_patch_area_px
                rhalo_ok = rgeom["halo_overhead"] <= floors.max_halo_overhead
                if not (rpatch_ok and rhalo_ok):
                    _log(
                        f"layout rescue: candidate {rnx}x{rny} REJECTED — floors "
                        f"(min_patch_area {rgeom['min_patch_area']} px vs "
                        f"{floors.min_patch_area_px}; halo_overhead "
                        f"{rgeom['halo_overhead']:.4f} vs {floors.max_halo_overhead}); "
                        "coarsening further only relaxes these floors"
                    )
                    rescue_trials.append({
                        "nx": rnx, "ny": rny, "cell_count": rnx * rny,
                        "floors_ok": False,
                        "min_patch_area": int(rgeom["min_patch_area"]),
                        "halo_overhead": round(float(rgeom["halo_overhead"]), 6),
                        "memory_ok": None,
                    })
                    continue
                rworst = _scan_layout_mode_aware(
                    canvas, footprints, rnx, rny, halo_px, tile_size
                )
                rselected, rtrials, rreason = _select_cpu_streaming_tile(
                    rworst["stream_worst_by_tile"], ram_budget
                )
                rinmem_ok = (ram_budget is None) or (
                    rworst["inmem_worst_bytes"] <= ram_budget
                )
                rmemory_ok = rinmem_ok or (rselected is not None)
                _rbound = int(_selected_admissible_bound(
                    rinmem_ok, rworst["inmem_worst_bytes"], rselected,
                    rworst["stream_worst_by_tile"], rworst.get("cheaper_bound_bytes", 0),
                ))
                _rtrial = {
                    "nx": rnx, "ny": rny, "cell_count": rnx * rny,
                    "floors_ok": True,
                    "min_patch_area": int(rgeom["min_patch_area"]),
                    "halo_overhead": round(float(rgeom["halo_overhead"]), 6),
                    "inmem_worst_bytes": int(rworst["inmem_worst_bytes"]),
                    "stream_worst_by_tile": {
                        int(k): int(v) for k, v in rworst["stream_worst_by_tile"].items()
                    },
                    "selected_tile": (int(rselected) if rselected is not None else None),
                    "admissible_bound_bytes": _rbound,
                    "memory_ok": bool(rmemory_ok),
                }
                rescue_trials.append(_rtrial)
                if rmemory_ok:
                    # Nearest coarser candidate that satisfies BOTH the floors and
                    # the actual mode-aware RAM budget.
                    rf = round(float(mw) * float(rnx) / float(canvas.width), 6)
                    chosen = {
                        "factor": rf, "nx": rnx, "ny": rny,
                        "bound": _rbound, "worst_cell": rworst,
                        "memory_ok": True, "geom": rgeom,
                        "inmem_ok": bool(rinmem_ok),
                        "selected_tile": rselected,
                        "trials": rtrials, "selection_reason": rreason,
                    }
                    rescue_record = {
                        "applied": True,
                        "kind": "coarser_rescue",
                        "reason": "max_halo_overhead",
                        "trigger": {
                            "nx": halo_conflict["nx"], "ny": halo_conflict["ny"],
                            "refinement_factor": halo_conflict["factor"],
                            "halo_overhead": round(
                                float(halo_conflict["geom"]["halo_overhead"]), 6
                            ),
                            "max_halo_overhead": float(floors.max_halo_overhead),
                            "memory_evaluated": False,
                        },
                        "chosen": {
                            "nx": rnx, "ny": rny, "cell_count": rnx * rny,
                            "effective_coarsening_factor": rf,
                        },
                        "candidates_evaluated": [dict(t) for t in rescue_trials],
                        "monotonic_note": (
                            "coarsening monotonically lowers halo_overhead and raises "
                            "min_patch_area; memory is NOT monotone in coarsening and "
                            "was evaluated exactly for this candidate"
                        ),
                    }
                    _log(
                        f"layout rescue: candidate {rnx}x{rny} "
                        f"({rnx * rny} cells) ACCEPTED — floors ok "
                        f"(halo_overhead {rgeom['halo_overhead']:.4f} <= "
                        f"{floors.max_halo_overhead}); admissible bound "
                        f"{_rbound / 2**20:.1f} MiB"
                        + (f" <= budget {ram_budget / 2**20:.1f} MiB"
                           if ram_budget is not None else "")
                        + f"; CPU streaming tile {rselected} ({rreason})"
                    )
                    break
                # Nearest floor-feasible coarser candidate does not fit. Patch
                # area is non-decreasing under coarsening and the fitted model is
                # increasing in it, so the remaining (coarser) family is NOT
                # assumed to lower memory -> stop, never claim otherwise.
                rescue_memory_failure = dict(_rtrial)
                _log(
                    f"layout rescue: candidate {rnx}x{rny} REJECTED — admissible "
                    f"bound {_rbound / 2**20:.1f} MiB > budget "
                    f"{(ram_budget / 2**20) if ram_budget is not None else 0:.1f} MiB "
                    f"(in-memory worst {rworst['inmem_worst_bytes'] / 2**20:.1f} MiB; "
                    f"best CPU streaming tile {rselected}). Coarsening further "
                    "enlarges per-cell patch area, so it is NOT assumed to lower "
                    "memory — stopping the rescue scan (no further candidates)"
                )
                break
        else:
            _log(
                "layout rescue: NOT attempted — a bounded refinement-rank window "
                f"(min={min_refinement_rank}, max={max_refinement_rank}) forbids "
                "coarsening; refusing rather than changing the exact geometry"
            )

    if chosen is None and halo_conflict is not None:
        _half = max(1, int(halo_px) // 2)
        _conflict = (
            f"requested halo {halo_px}px conflicts with the max_halo_overhead floor "
            f"({halo_conflict['geom']['halo_overhead']:.4f} > "
            f"{floors.max_halo_overhead}) at the "
            f"{halo_conflict['nx']}x{halo_conflict['ny']} factor-"
            f"{halo_conflict['factor']:g} layout"
        )
        _direction = (
            f"Lower the anti-checkerboard halo (e.g. {halo_px} -> {_half} px) or "
            "increase the core/cell size. A FINER layout is NOT the fix: finer Cells "
            "raise halo overhead, which is the floor that rejected this candidate."
        )
        if rescue_memory_failure is not None:
            msg = (
                f"{_conflict}, and the nearest floor-feasible COARSER rescue "
                f"{rescue_memory_failure['nx']}x{rescue_memory_failure['ny']} needs "
                f"{rescue_memory_failure['admissible_bound_bytes'] / 2**20:.1f} MiB "
                f"> budget {ram_budget / 2**20:.1f} MiB "
                f"(in-memory worst "
                f"{rescue_memory_failure['inmem_worst_bytes'] / 2**20:.1f} MiB; best "
                f"CPU streaming tile {rescue_memory_failure['selected_tile']}). "
                "Coarsening further enlarges the per-cell patch area, so no coarser "
                "candidate is assumed to fit and the rescue is not scanned further. "
                f"{_direction}"
            )
            _refusal = {
                # A memory candidate WAS evaluated (the coarser rescue) and it
                # missed the budget: this is a REAL host-RAM refusal even though
                # the headline names the halo-floor conflict.
                "kind": "host_ram",
                "source": "coarser_rescue",
                "budget_bytes": int(ram_budget) if ram_budget is not None else 0,
                "required_bytes": int(
                    rescue_memory_failure["admissible_bound_bytes"]
                ),
                "inmem_worst_bytes": int(
                    rescue_memory_failure["inmem_worst_bytes"]
                ),
                "selected_tile": rescue_memory_failure["selected_tile"],
                "candidate": {
                    "nx": int(rescue_memory_failure["nx"]),
                    "ny": int(rescue_memory_failure["ny"]),
                },
            }
        elif not _rescue_allowed:
            msg = (
                f"{_conflict}. A COARSER layout would be needed to satisfy the halo "
                "floor, but the bounded replan refinement-rank window "
                f"(min={min_refinement_rank}, max={max_refinement_rank}) forbids "
                "coarsening; the exact geometry is preserved and the plan is refused. "
                f"{_direction}"
            )
            _refusal = {
                "kind": "geometry",
                "source": "halo_conflict_rank_window",
            }
        elif not rescue_grids:
            msg = (
                f"{_conflict}, and no COARSER layout exists "
                f"(the grid is already {halo_conflict['nx']}x{halo_conflict['ny']}). "
                f"This is a halo-floor conflict, not a RAM-budget conflict. {_direction}"
            )
            _refusal = {
                "kind": "geometry",
                "source": "halo_conflict_no_coarser_grid",
            }
        else:
            msg = (
                f"{_conflict}, and none of the {len(rescue_grids)} deterministic "
                "COARSER rescue candidates satisfies both scientific floors. This is "
                f"a halo-floor conflict, not a RAM-budget conflict. {_direction}"
            )
            _refusal = {
                "kind": "geometry",
                "source": "halo_conflict_no_feasible_rescue",
            }
        _log(msg, "ERROR")
        raise zal.LayoutInfeasible(msg, refusal=_refusal)

    if chosen is None and not candidates:
        # No candidate was ever memory-evaluated and the halo-only case above did
        # not apply: the forward scan stopped at the min_patch_area floor. Do not
        # emit a bogus "RAM budget" headline and do not point at FINER (finer
        # Cells have LESS patch area).
        msg = (
            "no layout candidate satisfies the min_patch_area floor "
            f"({floors.min_patch_area_px} px): the coarsest enumerated candidate is "
            "below it. This is a patch-area floor conflict, not a RAM-budget "
            "conflict (no memory candidate was evaluated). A COARSER layout (fewer, "
            "larger Cells) or a larger core geometry is the direction that raises "
            "patch area; a FINER layout reduces it and cannot help."
        )
        _log(msg, "ERROR")
        raise zal.LayoutInfeasible(
            msg,
            refusal={
                "kind": "geometry",
                "source": "min_patch_area_floor",
            },
        )

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
        if halo_floor_stopped is not None:
            # The finer direction is blocked by the halo floor, so never advise it.
            _finest_dir = (
                "Further refinement is BLOCKED by max_halo_overhead "
                f"({halo_floor_stopped['geom']['halo_overhead']:.4f} > "
                f"{floors.max_halo_overhead}) at "
                f"{halo_floor_stopped['nx']}x{halo_floor_stopped['ny']}; lower the "
                f"anti-checkerboard halo (e.g. {halo_px} -> {max(1, int(halo_px) // 2)} "
                "px) to allow finer layouts, or increase available RAM."
            )
        else:
            _finest_dir = (
                "A FINER layout (more cells) is the direction that lowers the "
                "per-cell footprint; a deeper disk-backed streaming executor is not "
                "available in R36. Suggest either increasing available RAM, choosing "
                "a finer layout, or pinning an explicit layout via the 'zegrid_layout' "
                "config key when reproducible geometry is required."
            )
        msg = (
            "RAM budget cannot satisfy scientific floors (mode-aware): "
            + "; ".join(reasons)
            + ". "
            + _finest_dir
        )
        _log(msg, "ERROR")
        if finest is not None and ram_budget is not None:
            _refusal = {
                "kind": "host_ram",
                "source": "finest_candidate",
                "budget_bytes": int(ram_budget),
                "required_bytes": int(finest["bound"]),
                "inmem_worst_bytes": int(
                    finest["worst_cell"].get("inmem_worst_bytes", 0)
                ),
                "candidate": {"nx": int(finest["nx"]), "ny": int(finest["ny"])},
            }
        else:
            _refusal = {"kind": "geometry", "source": "no_budget_candidate"}
        raise zal.LayoutInfeasible(msg, refusal=_refusal)

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
    _origin = "auto_coarser_rescue" if rescue_record is not None else "auto_refinement"
    _log(
        f"layout scan: CHOSEN {chosen['nx']}x{chosen['ny']} "
        f"({chosen['nx'] * chosen['ny']} cells, mode-aware, source=auto, "
        f"origin={_origin}) "
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
        "layout_origin": _origin,
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
        **({"layout_rescue": rescue_record} if rescue_record is not None else {}),
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


def _make_cell_provider(cache_dir, storage_profile):
    """Select the Cell provider for the admitted storage profile.

    Dense returns the historical ``MemmapCanonicalProvider`` BY IDENTITY;
    bounded returns the Lot A ``BoundedMemmapCanonicalProvider``. This is the
    single seam Gate E instruments.
    """
    profile = zsp.normalize_storage_profile(storage_profile)
    if zsp.is_bounded(profile):
        return zfp.BoundedMemmapCanonicalProvider(cache_dir)
    return zfp.MemmapCanonicalProvider(cache_dir)


def _run_cell_inmem(cache_dir, patch, config, progress_callback, fixed=None, tile_size=None,
                    storage_profile=None):
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
    _profile = zsp.normalize_storage_profile(
        storage_profile if storage_profile is not None else _STACK_STORAGE_PROFILE
    )
    if zsp.is_bounded(_profile):
        raise RuntimeError(
            "storage_profile='bounded' cannot run the in-memory cell path; "
            "the bounded profile requires the file-backed streaming executor"
        )
    provider = _make_cell_provider(cache_dir, _profile)
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


def _run_cell_stream(cache_dir, patch, config, progress_callback, tile_size=STREAM_TILE_SIZE,
                     fixed=None, storage_profile=None, storage_info=None):
    _profile = zsp.normalize_storage_profile(
        storage_profile if storage_profile is not None else _STACK_STORAGE_PROFILE
    )
    provider = _make_cell_provider(cache_dir, _profile)
    sink = None
    try:
        request = zstream.build_streaming_request(config, provider.n_frames)
        if zsp.is_bounded(_profile):
            # FRESH sink for THIS cell run: never reused across cells/retries.
            sink = zrs.BoundedCountingRejectionSink(
                (provider.n_frames, provider.height, provider.width, provider.channels)
            )
            result = run_canonical_stack_streaming(
                provider, request, tile_size=tile_size, fixed=fixed,
                rejection_sink=sink,
            )
        else:
            result = run_canonical_stack_streaming(
                provider, request, tile_size=tile_size, fixed=fixed
            )
        if storage_info is not None:
            storage_info.update({
                "profile": _profile,
                "model": zsp.storage_model_id(_profile),
                "provider": type(provider).__name__,
                "storage_domain": getattr(provider, "storage_domain", None),
                "bytes_read": getattr(provider, "bytes_read", None),
                "open_count": getattr(provider, "open_count", None),
                "open_failures": getattr(provider, "open_failures", None),
                "sink": None if sink is None else {
                    "type": type(sink).__name__,
                    "storage_domain": getattr(sink, "storage_domain", None),
                    "true_count": int(sink.true_count),
                    "written_cells": int(sink.written_cells),
                    "writes": int(sink.writes),
                    "flush_count": int(sink.flush_count),
                    "max_write_bytes": int(sink.max_write_bytes),
                    "max_block_bytes": int(sink.max_block_bytes),
                },
                "rejection_mask_available": result.rejection_mask is not None,
            })
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
# ZM-ZEGRID-R52-OOC Lot B: the ONE storage profile for this run. Selected before
# layout filtering and kept through PLAN/ADMIT/RUN/fallback. Default dense keeps
# every historical byte and path unchanged; bounded is an explicit internal seam.
_STACK_STORAGE_PROFILE = zsp.STORAGE_PROFILE_DENSE
# ZM-ZEGRID-ACTIVE-OOC-FALLBACK: run-scoped storage POLICY (not a GUI knob).
# ``requested`` is what the caller asked for ("auto" for every product call);
# ``effective`` is the profile actually executing. ``auto`` means: try dense, and
# switch to bounded ONCE on a QUALIFIED host-RAM refusal. An explicit
# dense/bounded request keeps its forced value (A/B witnesses never fall back).
_STORAGE_REQUESTED = zsp.STORAGE_PROFILE_AUTO
_STORAGE_EFFECTIVE = zsp.STORAGE_PROFILE_DENSE
_STORAGE_FALLBACK_COUNT = 0
_STORAGE_TRANSITIONS: list = []
_STORAGE_DENSE_REFUSAL = None
# True only for the ACTIVE product path: legacy/observe keep their exact pre-R52
# manifest shape (the storage provenance block is ACTIVE-only).
_STORAGE_ACTIVE_MODE = False


def _storage_policy_reset(requested):
    """Select the REQUESTED policy once per run and reset the transition trail."""
    global _STORAGE_REQUESTED, _STORAGE_EFFECTIVE
    global _STORAGE_FALLBACK_COUNT, _STORAGE_TRANSITIONS, _STORAGE_DENSE_REFUSAL
    _STORAGE_REQUESTED = zsp.normalize_storage_request(requested)
    _STORAGE_EFFECTIVE = (
        zsp.STORAGE_PROFILE_BOUNDED
        if _STORAGE_REQUESTED == zsp.STORAGE_PROFILE_BOUNDED
        else zsp.STORAGE_PROFILE_DENSE
    )
    _STORAGE_FALLBACK_COUNT = 0
    _STORAGE_TRANSITIONS = []
    _STORAGE_DENSE_REFUSAL = None
    # The run-scoped effective seam must agree with the policy from the very
    # first layout filter (no stale profile can survive a new run/reset).
    _set_stack_storage_profile(_STORAGE_EFFECTIVE)
    return _STORAGE_EFFECTIVE


def _storage_policy_snapshot() -> dict:
    """Readable provenance for the run log and the manifest."""
    return {
        "schema": "zemosaic-zegrid-storage-policy-v1",
        "requested": str(_STORAGE_REQUESTED),
        "effective": str(_STORAGE_EFFECTIVE),
        "model": zsp.storage_model_id(_STORAGE_EFFECTIVE),
        "fallback_count": int(_STORAGE_FALLBACK_COUNT),
        "auto": bool(_STORAGE_REQUESTED == zsp.STORAGE_PROFILE_AUTO),
        "forced": bool(_STORAGE_REQUESTED != zsp.STORAGE_PROFILE_AUTO),
        "dense_refusal": (
            None if _STORAGE_DENSE_REFUSAL is None
            else dict(_STORAGE_DENSE_REFUSAL)
        ),
        "transitions": [dict(item) for item in _STORAGE_TRANSITIONS],
    }


def _storage_fallback_available() -> bool:
    """True iff ONE dense->bounded fallback is still allowed this run."""
    return bool(
        _STORAGE_REQUESTED == zsp.STORAGE_PROFILE_AUTO
        and _STORAGE_EFFECTIVE == zsp.STORAGE_PROFILE_DENSE
        and int(_STORAGE_FALLBACK_COUNT) == 0
    )


def _activate_bounded_storage(*, phase, reason_code, detail, emit=None):
    """Switch the run to the bounded profile ONCE; returns True on success.

    The effective profile is published in the run-scoped seam BEFORE any rebuild,
    so PLAN facts, layout arithmetic, task tuples and every cell payload agree.
    """
    global _STORAGE_EFFECTIVE, _STORAGE_FALLBACK_COUNT, _STORAGE_DENSE_REFUSAL
    if not _storage_fallback_available():
        return False
    _STORAGE_DENSE_REFUSAL = {
        "phase": str(phase),
        "reason_code": str(reason_code),
        "detail": str(detail),
    }
    _STORAGE_EFFECTIVE = zsp.STORAGE_PROFILE_BOUNDED
    _STORAGE_FALLBACK_COUNT = 1
    _STORAGE_TRANSITIONS.append(
        {
            "from": zsp.STORAGE_PROFILE_DENSE,
            "to": zsp.STORAGE_PROFILE_BOUNDED,
            "phase": str(phase),
            "reason_code": str(reason_code),
            "detail": str(detail),
        }
    )
    _set_stack_storage_profile(zsp.STORAGE_PROFILE_BOUNDED)
    if emit is not None:
        try:
            emit(
                "ZeGrid: storage profile AUTO fell back to 'bounded' at "
                f"phase={phase} after a qualified host-RAM refusal "
                f"({reason_code}: {detail}); ONE bounded retry, same geometry/halo/"
                "science; no further profile switch",
                "WARN",
            )
        except Exception:
            pass
    return True


def _set_stack_storage_profile(profile):
    global _STACK_STORAGE_PROFILE
    _STACK_STORAGE_PROFILE = zsp.normalize_storage_profile(profile)
    return _STACK_STORAGE_PROFILE


def _plan_bundle_auto(
    *,
    canvas,
    descs,
    footprints,
    raw,
    budget,
    pinned_layout=None,
    gauge_workers=None,
    cache_build_workers=None,
    layout_substep=None,
    bundle_common=None,
    active=False,
    emit=None,
):
    """Build the initial bundle; in AUTO ACTIVE mode retry bounded ONCE.

    This is the PRODUCTION wiring used by ``_run_single``: the FIRST layout
    filter and the complete bundle are attempted with the currently effective
    profile (dense for an AUTO request). Only a QUALIFIED host-RAM refusal
    switches the run to bounded and rebuilds with the SAME request. A forced
    profile, a non-ACTIVE scheduler mode or a non-memory floor failure raises
    unchanged (never a silent retry).
    """
    kw = dict(bundle_common or {})

    def _build():
        return _build_execution_bundle(
            canvas, descs, footprints,
            raw_available_bytes=int(raw), budget_bytes=int(budget),
            pinned_layout=pinned_layout,
            gauge_workers=gauge_workers,
            cache_build_workers=cache_build_workers,
            layout_substep=layout_substep,
            **kw,
        )

    try:
        return _build()
    except (zal.LayoutInfeasible, zpar.ModeInfeasible) as exc:
        is_memory, code, detail = _storage_plan_refusal(exc)
        if not (
            active
            and is_memory
            and _activate_bounded_storage(
                phase="plan_layout", reason_code=code, detail=detail,
                emit=emit,
            )
        ):
            raise
        if emit is not None:
            try:
                emit(
                    "ZeGrid: dense layout/bundle refused for RAM — retrying "
                    "ONCE with the bounded storage profile (same layout "
                    f"request, same halo/science): {detail}",
                    "WARN",
                )
            except Exception:
                pass
        try:
            return _build()
        except (zal.LayoutInfeasible, zpar.ModeInfeasible) as exc2:
            if emit is not None:
                try:
                    emit(
                        "ZeGrid: NO executable plan fits the PLAN budget in "
                        f"EITHER storage profile (dense: {detail}; bounded: "
                        f"{exc2})",
                        "ERROR",
                    )
                except Exception:
                    pass
            raise zpar.ModeInfeasible(
                "no executable ZeGrid plan fits the PLAN budget in either "
                f"storage profile (dense: {detail}; bounded: {exc2})"
            ) from exc2


def _storage_admission_refusal(exc):
    """Typed eligibility for the pre-gauge ACTIVE admission fallback.

    Only a REAL capacity refusal may switch the storage profile. A config error,
    a growth/freeze violation, an invalid observation/envelope, a disk refusal or
    any programming error is NOT a memory shortage and is propagated unchanged.
    """
    code = getattr(exc, "reason_code", None)
    if code == renv.REASON_REFUSE_HOST_RAM:
        return True, str(code), str(exc)
    if isinstance(exc, rcontract.AdmissionExhausted):
        # The admission controller exhausted its bounded replans against a REAL
        # measured ceiling: this is a capacity refusal, not an observation error.
        return True, renv.REASON_REFUSE_HOST_RAM, str(exc)
    return False, None, str(exc)


def _storage_plan_refusal(exc):
    """Classify a PLAN/ADMIT refusal as a QUALIFIED host-RAM memory refusal.

    Only memory-qualified failures may trigger the bounded fallback. Scientific
    floors, degenerate geometry, WCS/FITS, permissions, cache corruption, disk
    full, programming errors and cancellation are NOT memory refusals and keep
    propagating unchanged.

    ZM-ZEGRID-PLAN-REFUSAL-CLASSIFICATION: eligibility is decided FIRST on the
    structured ``refusal`` payload attached by the planner (``kind ==
    "host_ram"`` -> eligible, any other explicit ``kind`` -> NOT eligible). The
    historical wording regex is kept ONLY as a fallback for exceptions that
    carry no structured payload (compatibility with older call sites), never as
    the primary decision when a payload exists.

    Returns ``(is_memory, reason_code, detail)``.
    """
    detail = str(exc)
    if isinstance(exc, zpar.ModeInfeasible):
        # ModeInfeasible means NO stacking mode/geometry fits the RAM budget.
        return True, renv.REASON_REFUSE_HOST_RAM, detail
    if isinstance(exc, zal.LayoutInfeasible):
        # Structured diagnosis first (facts, not prose).
        refusal = getattr(exc, "refusal", None)
        if isinstance(refusal, dict) and refusal.get("kind"):
            if str(refusal.get("kind")) == "host_ram":
                return True, renv.REASON_REFUSE_HOST_RAM, detail
            return False, None, detail
        # ---- legacy fallback: no structured payload on this exception ----
        # ``LayoutInfeasible`` also covers genuinely invalid scientific floors.
        # Only the explicit budget-miss diagnosis is a real memory refusal: the
        # wording is "...exceeds budget ..." or "...both exceed budget ...".
        import re as _re
        if _re.search(r"\bexceed(?:s)?\s+budget\b", detail):
            return True, renv.REASON_REFUSE_HOST_RAM, detail
        # After a memory-driven internal refinement, the bounded-rank window can
        # admit NO candidate and the final message is the rank-window refusal.
        # That is still a RAM/geometry-window miss ONLY when the builder recorded
        # real memory exhaustion (`memory_exhausted`); a pinned/user window is
        # not a capacity miss. A genuine floor invalidity refuses again under
        # bounded and surfaces as a loud double refusal.
        if (
            "refinement-rank window" in detail
            and getattr(exc, "memory_exhausted", False) is True
        ):
            return True, renv.REASON_REFUSE_HOST_RAM, detail
        return False, None, detail
    return False, None, detail
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


def _prepare_stack_cell(task):
    """ZM-ZEGRID-R55-E2: CPU-side staging of ONE cell (cache + cell gauge).

    This is the production half of the historical :func:`_stack_cell` that runs
    on the CPU producer side of the bounded pipeline. It is byte-for-byte the
    same sequence of operations (cache build -> cell gauge resolution) and
    returns either a prepared payload for :func:`_consume_stack_cell` or an
    already-final ``{"empty_result": ...}`` for an empty/zero-support cell.
    """
    if len(task) == 7:
        (cell, patch, mem, cache_dir, mode, bound, storage_profile) = task
    else:
        (cell, patch, mem, cache_dir, mode, bound) = task
        storage_profile = _STACK_STORAGE_PROFILE
    storage_profile = zsp.normalize_storage_profile(storage_profile)
    cid = cell.cell_id
    _patch_bounds = [int(patch.patch.x0), int(patch.patch.y0),
                     int(patch.patch.x1), int(patch.patch.y1)]
    _patch_shape_hw = [int(patch.patch.height), int(patch.patch.width)]
    if not mem.patch_ids:
        return {"empty_result": {
            "cell_id": cid, "status": "empty",
            "record": {
                "cell_id": cid, "status": "empty", "mode": None,
                "n_contributors": 0,
                "n_patch_contributors": 0,
                "patch_bounds": _patch_bounds,
                "patch_shape_hw": _patch_shape_hw,
            },
            "reproject": None,
        }}
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
    reproject = zxe.get_reproject_path_stats().to_dict()

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
    cell_provider = _make_cell_provider(cache_dir, storage_profile)
    _gauge_storage = None
    try:
        cell_frame_ids = list(cell_provider.frame_ids)
        cell_fixed, gauge_status, gauge_detail = _resolve_cell_normalization(
            cell_gauge_mode, _STACK_GAUGE, _STACK_GLOBAL_FRAME_IDS,
            cell_provider, _STACK_CONFIG,
        )
        _gauge_storage = {
            "provider": type(cell_provider).__name__,
            "storage_domain": getattr(
                cell_provider, "storage_domain", "aligned-npy-memmap-dense"
            ),
            "counters_available": hasattr(cell_provider, "bytes_read"),
            "bytes_read": getattr(cell_provider, "bytes_read", None),
            "open_count": getattr(cell_provider, "open_count", None),
            "open_failures": getattr(cell_provider, "open_failures", None),
        }
    finally:
        cell_provider.close()

    if cell_fixed is None:
        # Explicit, DETERMINISTIC zero-support Cell: no gauge, no fabricated data.
        # (Distinct from a geometrically-empty Cell — this one HAS contributors but
        # none carry finite support.)
        _safe_rmtree(cache_dir, None)
        return {"empty_result": {
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
        }}

    return {
        "cell": cell,
        "patch": patch,
        "mem": mem,
        "cache_dir": cache_dir,
        "mode": mode,
        "bound": bound,
        "patch_bounds": _patch_bounds,
        "patch_shape_hw": _patch_shape_hw,
        "manifest": manifest,
        "cache_build_s": cache_build_s,
        "reproject": reproject,
        "cell_gauge_mode": cell_gauge_mode,
        "cell_frame_ids": cell_frame_ids,
        "cell_fixed": cell_fixed,
        "gauge_status": gauge_status,
        "gauge_detail": gauge_detail,
        "gauge_storage": _gauge_storage,
        **(
            {"storage_profile": storage_profile}
            if zsp.is_bounded(storage_profile) else {}
        ),
    }


def _consume_stack_cell(prepared, *, tile_size=None):
    """GPU/CPU owner-side consumption of ONE prepared cell (stack + crop).

    ``tile_size`` (ZM-ZEGRID-R55-E11) is the EXACT streaming tile this unit must
    run with. The GPU pipeline carries the current RUN-adapted tile in the
    per-unit payload under ``"_stack_tile_size"`` so a mid-run shrink never
    depends on mutating the worker-global ``_STACK_TILE_SIZE`` (and never leaves
    a stale global behind). An explicit ``tile_size`` argument wins; otherwise the
    payload key is used; otherwise the historical worker-global behaviour is
    preserved for every existing caller.
    """
    if tile_size is None and isinstance(prepared, dict):
        _payload_tile = prepared.get("_stack_tile_size")
        if _payload_tile is not None:
            tile_size = int(_payload_tile)
    if isinstance(prepared, dict) and "empty_result" in prepared:
        return prepared["empty_result"]
    cell = prepared["cell"]
    patch = prepared["patch"]
    mem = prepared["mem"]
    cache_dir = prepared["cache_dir"]
    mode = prepared["mode"]
    bound = prepared["bound"]
    storage_profile = zsp.normalize_storage_profile(
        prepared.get("storage_profile", _STACK_STORAGE_PROFILE)
    )
    patch_bounds = prepared["patch_bounds"]
    patch_shape_hw = prepared["patch_shape_hw"]
    manifest = prepared["manifest"]
    cache_build_s = prepared["cache_build_s"]
    reproject = prepared["reproject"]
    cell_gauge_mode = prepared["cell_gauge_mode"]
    cell_frame_ids = prepared["cell_frame_ids"]
    cell_fixed = prepared["cell_fixed"]
    cid = cell.cell_id

    t0 = time.perf_counter()
    storage_info = {}
    if mode == "inmem" and zsp.is_bounded(storage_profile):
        raise RuntimeError(
            "storage_profile='bounded' received an in-memory cell task; the "
            "admitted storage model requires the file-backed streaming path"
        )
    if mode == "inmem":
        # ZM-ZEGRID-R22: when the cell runs on the GPU, use the VRAM-bounded tile
        # size (never a single N x full-cell tile); CPU keeps the legacy
        # single-tile full-patch behaviour.
        _effective_tile = int(
            _STACK_TILE_SIZE if tile_size is None else tile_size
        )
        inmem_tile = (
            _effective_tile
            if getattr(_STACK_CONFIG, "backend", "cpu") == "gpu"
            else None
        )
        mt, sres = _run_cell_inmem(cache_dir, patch, _STACK_CONFIG, None,
                                   fixed=cell_fixed, tile_size=inmem_tile)
    else:
        mt, sres = _run_cell_stream(
            cache_dir, patch, _STACK_CONFIG, None,
            tile_size=int(_STACK_TILE_SIZE if tile_size is None else tile_size),
            fixed=cell_fixed,
            storage_profile=storage_profile,
            storage_info=storage_info,
        )
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
        "patch_bounds": patch_bounds,
        "patch_shape_hw": patch_shape_hw,
        "cell_gauge_mode": cell_gauge_mode,
        "backend_used": getattr(_STACK_CONFIG, "backend", "cpu"),
        "reference_frame_id": sres.reference_frame_id,
        "reference_frame_role": prov["reference_frame_role"],
        "bookkeeping_reference_frame_id": prov["bookkeeping_reference_frame_id"],
        "excluded": [list(e) for e in sres.excluded],
        "bound_bytes": int(bound),
    }
    if zsp.is_bounded(storage_profile):
        record["storage_profile"] = storage_profile
        record["storage"] = dict(storage_info)

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
        "reproject": reproject,
        "cache_dir": str(cache_dir),
        **(
            {"storage_profile": storage_profile}
            if zsp.is_bounded(storage_profile) else {}
        ),
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


def _stack_cell(task):
    """Module-level (picklable) worker: build cache + stack ONE cell (R20).

    Preserved public behaviour: the exact same prepare -> consume sequence used
    by the bounded GPU pipeline. CPU/pool callers are byte-unchanged.
    """
    return _consume_stack_cell(_prepare_stack_cell(task))


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
        "vram_budget_source": ctx.get("vram_budget_source"),
        "legacy_probe_vram_budget_bytes": ctx.get(
            "legacy_probe_vram_budget_bytes"
        ),
        "gpu_initial_streaming_tile": ctx.get("gpu_initial_streaming_tile"),
        "gpu_tile_adaptation": ctx.get("gpu_tile_adaptation"),
        "engine_provenance": ctx.get("engine_provenance"),
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
# R55-E8 — GPU authority resolution (NO probe when the product says OFF)
# ---------------------------------------------------------------------------

def _r55_resolve_gpu_probe(use_gpu, zconfig):
    """Resolve GPU preference and probe ONLY when the product requests GPU.

    GPU OFF must make all four authority flags false and must NOT import CuPy,
    query the CUDA runtime or read VRAM. The returned probe dict is explicitly
    ``reason=gpu_not_requested`` so downstream manifests never imply a device
    observation that was forbidden.
    """
    requested, source = resolve_gpu_preference(use_gpu, zconfig)
    if requested:
        probe = zgpu.probe_gpu_backend()
        budget = zgpu.vram_budget_bytes(probe)
    else:
        probe = {
            "available": False,
            "device": None,
            "cupy_version": None,
            "vram_total_bytes": None,
            "vram_free_bytes": None,
            "reason": "gpu_not_requested",
        }
        budget = None
    return bool(requested), str(source), probe, budget


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
    # R55-E8: exact device workspace of the selected GPU tile and the bounded
    # per-tile host/device trial trail used by the shared GPU ladder. ``None``
    # for pure CPU bundles so every existing consumer keeps its semantics.
    gpu_device_workspace_bytes: int | None = None
    gpu_tile_trials: list | None = None
    storage_profile: str = zsp.STORAGE_PROFILE_DENSE


def _r55_margin_raw_ceiling(
    budget_bytes: int,
    *,
    margin_num: int = renv.DEFAULT_MARGIN_NUM,
    margin_den: int = renv.DEFAULT_MARGIN_DEN,
) -> int:
    """Largest raw phase peak whose margin-applied value still fits ``budget``.

    This is the inverse of :func:`resource_envelope.ceil_margin_div` in integer
    arithmetic: ``ceil(raw * (den+num)/den) <= budget`` iff
    ``raw <= (budget*den)//(den+num)``. It is the SAME margin the global envelope
    validation applies, so the gauge/cache worker choice can never be made under
    a looser budget than the one that will validate it.
    """
    num = int(margin_num)
    den = int(margin_den)
    if num < 0 or den < 1 or num > den:
        raise ValueError(
            f"invalid margin {num}/{den}: expected 0 <= num <= den, den >= 1"
        )
    return (int(budget_bytes) * den) // (den + num)


def _r55_coherent_worker_count(
    requested,
    worker_budget_bytes: int,
    per_worker_footprint_bytes: int,
) -> int:
    """Worker count selected under the SAME budget/margin that validates it.

    Reuses :func:`parallel.choose_workers` (the existing R16 rule) but adds the
    hard integer ceiling implied by the margin-aware phase budget, allowing a
    genuine 1-worker floor when even 2 workers cannot fit. Never grows past the
    existing CPU/RAM decision.
    """
    footprint = max(1, int(per_worker_footprint_bytes))
    budget = max(0, int(worker_budget_bytes))
    selected = int(
        zpar.choose_workers(
            requested, budget if budget > 0 else None, footprint
        )
    )
    hard_ceiling = max(1, budget // footprint) if budget > 0 else 1
    return max(1, min(selected, hard_ceiling))


def _r55_worker_ladder(start: int) -> tuple[int, ...]:
    """Bounded descending worker ladder down to 1 (deterministic)."""
    top = max(1, int(start))
    return tuple(range(top, 0, -1))


def _r55_gpu_tile_ladder(
    max_n: int,
    worst_hw: tuple[int, int],
    vram_budget_bytes: int,
    *,
    max_gpu_tile: int | None = None,
) -> tuple[tuple[tuple[int, int], ...], int | None]:
    """Finite explicit GPU tile ladder with EXACT device workspace per tile.

    The ladder always contains the VRAM-derived maximum and every permitted
    product tile <= it, in descending order. Each entry is
    ``(tile_px, device_workspace_bytes)``; a tile whose exact device workspace
    exceeds the explicit VRAM budget is dropped rather than guessed.
    ``max_gpu_tile`` is the accepted no-growth cap (a replan may only shrink the
    GPU tile).
    """
    max_tile = zgpu.choose_gpu_tile_size(
        int(max_n), 3, tuple(int(v) for v in worst_hw), int(vram_budget_bytes)
    )
    if max_tile is None:
        return (), None
    max_tile = int(max_tile)
    tiles = sorted(
        {max_tile}
        | {
            int(t)
            for t in CPU_STREAMING_TILE_LADDER
            if int(t) < max_tile
        },
        reverse=True,
    )
    if max_gpu_tile is not None:
        tiles = [int(t) for t in tiles if int(t) <= int(max_gpu_tile)]
    out: list[tuple[int, int]] = []
    taper = 2 * int(zgpu.GPU_TAPER_HALO_PX)
    for tile in tiles:
        device = int(
            zgpu.gpu_tile_workspace_bytes(
                int(max_n),
                int(tile) + taper,
                int(tile) + taper,
                3,
            )
        )
        if device <= int(vram_budget_bytes):
            out.append((int(tile), int(device)))
    return tuple(out), max_tile


def _cell_task(cell, patch, mem, cache_dir, mode, bound, storage_profile):
    """Build ONE cell task; dense stays the historical 6-tuple EXACTLY.

    The bounded profile appends a 7th element so the worker knows which provider
    and sink to use without a second scheduler or a mutable global in the child.
    """
    profile = zsp.normalize_storage_profile(storage_profile)
    task = (cell, patch, mem, cache_dir, mode, bound)
    if zsp.is_bounded(profile):
        return task + (profile,)
    return task


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
    # ZM-ZEGRID-ACTIVE-OOC-FALLBACK rework-1 (MEDIUM): track a REAL memory-driven
    # refinement. Only then may a final rank-window refusal be classified as a
    # capacity miss for the bounded fallback.
    _refinement_memory_pressure = False

    while True:
        _fp_out = []
        try:
            layout = _choose_layout_mode_aware(
                canvas, descs, ram_budget=budget_bytes,
                available_bytes=raw_available_bytes, tile_size=STREAM_TILE_SIZE,
                pinned_layout=pinned_layout, emit=emit, log_line=log_line,
                observer=layout_substep, footprints_out=_fp_out,
                footprints_in=footprints,
                min_refinement_rank=rank_floor,
                max_refinement_rank=rank_ceiling,
                halo_px=_halo,
            )
        except zal.LayoutInfeasible as _layout_exc:
            # The final rank-window message only qualifies when THIS bundle build
            # really exhausted a measured memory budget and moved finer. A
            # pinned / user-requested window without memory pressure is NOT a
            # capacity miss.
            if (
                _refinement_memory_pressure
                and "refinement-rank window" in str(_layout_exc)
            ):
                _layout_exc.memory_exhausted = True
                # Structured upgrade (ZM-ZEGRID-PLAN-REFUSAL-CLASSIFICATION): a
                # refinement-rank window refusal that follows REAL memory
                # pressure is a host-RAM refusal, not a geometry one. The
                # classifier decides on this payload, never on the wording.
                _layout_exc.refusal = {
                    "kind": "host_ram",
                    "source": "refinement_window",
                    "budget_bytes": int(budget_bytes),
                }
            raise
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
        # R55-E8: workers are budgeted from the SAME PLAN budget AND the SAME
        # 15 % margin the global envelope validation will apply. The previous
        # un-margined division filled the budget exactly and then froze values
        # the validator had to refuse (Windows halo128 gauge=14).
        worker_budget = int(
            _r55_margin_raw_ceiling(int(budget_bytes))
        )
        canvas_px = int(canvas.width) * int(canvas.height)
        _gauge_fp = zpar.gauge_footprint_bytes(canvas_px)
        _cache_fp = zpar.cache_footprint_bytes(patch_px_for_workers or canvas_px)
        _gauge_workers = _r55_coherent_worker_count(
            gauge_workers, worker_budget, _gauge_fp
        )
        _cache_workers = _r55_coherent_worker_count(
            cache_build_workers, worker_budget, _cache_fp
        )

        # ZM-ZEGRID-R22/R36: VRAM-bounded GPU tile, determined BEFORE the joint
        # plan so the GPU RAM gate is evaluated at the GPU's OWN execution tile.
        effective_gpu = bool(gpu_used)
        gpu_tile_size = None
        gpu_device_workspace_bytes = None
        gpu_tile_candidates: tuple[tuple[int, int], ...] = ()
        gpu_tile_trials: list = []
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
            # R55-E8: a finite GPU tile ladder (VRAM maximum + every permitted
            # product tile <= it), each with its EXACT device workspace. The
            # host streaming bound is evaluated per tile below. A tile is never
            # dropped because a LARGER tile failed host RAM.
            gpu_tile_candidates, _vram_max_tile = _r55_gpu_tile_ladder(
                max_n,
                (int(worst_hw[0]), int(worst_hw[1])),
                int(vram_budget_bytes or 0),
                max_gpu_tile=max_gpu_tile,
            )
            if not gpu_tile_candidates:
                gpu_pre_execution_fallback = {
                    "reason": "vram_tile_infeasible",
                    "detail": (
                        "no GPU streaming tile fits the explicit VRAM budget "
                        f"{vram_budget_bytes} B "
                        f"(VRAM-derived max tile={_vram_max_tile!r})"
                    ),
                    "vram_budget_bytes": (
                        None if vram_budget_bytes is None
                        else int(vram_budget_bytes)
                    ),
                    "vram_max_tile": _vram_max_tile,
                }
                effective_gpu = False
            else:
                gpu_tile_size = int(gpu_tile_candidates[0][0])
                gpu_device_workspace_bytes = int(gpu_tile_candidates[0][1])

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

        _profile = zsp.normalize_storage_profile(_STACK_STORAGE_PROFILE)
        _bounded = zsp.is_bounded(_profile)

        def _stream_by_idx(tile):
            return {
                idx: int(_profile_stream_bytes(n, hw, int(tile), _profile))
                for idx, n, hw, area in active
            }

        # The bounded profile never executes the in-memory path: its provider is
        # file-backed and its sink bounds the rejection diagnostics. Feed the
        # planner an impossible in-memory bound so the chosen mode is ALWAYS
        # stream (the real in-memory bound is still reported in cell_specs for
        # diagnostics only).
        planning_inmem_by_idx = (
            {idx: (1 << 62) for idx, _n, _hw, _a in active}
            if _bounded else inmem_by_idx
        )

        gpu_stream_by_idx = (
            _stream_by_idx(gpu_tile_size) if effective_gpu else None
        )
        if effective_gpu:
            gpu_stream_by_idx = None

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
            # R55-E8: evaluate each tile with its EXACT host streaming bound AND
            # its exact device workspace before exact CPU fallback. The first
            # (largest) jointly feasible tile wins; a smaller GPU tile is not
            # discarded because the VRAM-max tile missed host RAM
            # (Linux halo16: 213 failed, 128 passes).
            for _tile, _device in gpu_tile_candidates:
                _streams = _stream_by_idx(int(_tile))
                gpu_pairs = [
                    (planning_inmem_by_idx[i], _streams[i])
                    for i, _n, _hw, _a in active
                ]
                _host_peak = int(
                    max((b for _a, b in gpu_pairs), default=0)
                )
                try:
                    _gpu_plan = _plan_for(
                        gpu_pairs, True, max_cells_in_flight
                    )
                except zpar.ModeInfeasible as exc:
                    gpu_tile_trials.append(
                        {
                            "tile": int(_tile),
                            "host_stream_bound_bytes": int(_host_peak),
                            "device_workspace_bytes": int(_device),
                            "fits": False,
                            "detail": str(exc),
                        }
                    )
                    continue
                gpu_tile_trials.append(
                    {
                        "tile": int(_tile),
                        "host_stream_bound_bytes": int(_host_peak),
                        "device_workspace_bytes": int(_device),
                        "fits": True,
                        "aggregate_bytes": int(
                            _gpu_plan.get("aggregate_bytes", 0)
                        ),
                    }
                )
                plan = _gpu_plan
                gpu_tile_size = int(_tile)
                gpu_device_workspace_bytes = int(_device)
                gpu_stream_by_idx = _streams
                break
            if plan is None:
                gpu_fallback_reason = {
                    "reason": "gpu_stream_host_ram_infeasible",
                    "gpu_tile_trials": [dict(t) for t in gpu_tile_trials],
                    "ram_budget_bytes": int(budget_bytes),
                    "detail": (
                        "no GPU tile in the finite ladder is jointly feasible "
                        "under host RAM + explicit device budget"
                    ),
                }
                effective_gpu = False
                gpu_tile_size = None
                gpu_device_workspace_bytes = None
                gpu_stream_by_idx = None

        if plan is None:
            # CPU: walk the PERMITTED ladder (largest first) and keep the first
            # jointly feasible tile. Never a numeric clamp, never a stale report.
            for t in permitted_tiles:
                stream_by_idx = _stream_by_idx(t)
                pairs = [(planning_inmem_by_idx[i], stream_by_idx[i]) for i, _n, _hw, _a in active]
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
                pairs = [(planning_inmem_by_idx[i], stream_by_idx[i]) for i, _n, _hw, _a in active]
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
            # candidate (never coarser). This is a REAL memory-driven refinement.
            _refinement_memory_pressure = True
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
        if _bounded and chosen_mode != "stream":
            raise zpar.ModeInfeasible(
                "storage_profile='bounded' requires the streaming executor, but "
                f"the joint planner chose {chosen_mode!r}; refusing to admit a "
                "bounded plan whose residency the in-memory path would exceed"
            )
        if _bounded:
            # L-1: the bundle planner has no profile flag, so it is fed an
            # impossible in-memory bound internally. Sanitize the PUBLISHED
            # candidate trail back to the real bound; the infeasibility reason
            # is the profile, not a fabricated 4.6 EiB number.
            _inmem_real = int(max(inmem_by_idx.values(), default=0))
            for _cand in plan.get("candidates", ()):
                if _cand.get("mode") == "inmem":
                    _cand["max_bound_bytes"] = _inmem_real
                    _cand["infeasible_reason"] = (
                        "in-memory mode is not admitted under "
                        "storage_profile=bounded (the profile requires the "
                        "file-backed streaming executor)"
                    )
            layout["storage_profile"] = _profile
            layout["storage_model"] = zsp.storage_model_id(_profile)
            plan["storage"] = {
                "profile": _profile,
                "model": zsp.storage_model_id(_profile),
                "provider": "BoundedMemmapCanonicalProvider",
                "rejection": "BoundedCountingRejectionSink",
                "fresh_sink_per_cell": True,
                # L-1: honest real in-memory bound. The 1<<62 planner sentinel is
                # NEVER published; this is the measured model value.
                "inmem_worst_actual": int(
                    max(inmem_by_idx.values(), default=0)
                ),
            }
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
            stream_b = _profile_stream_bytes(
                n, patch.patch_shape_hw, cpu_streaming_tile, _profile
            )
            gpu_stream_b = (
                _profile_stream_bytes(n, patch.patch_shape_hw, gpu_tile_size, _profile)
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
                cell_tasks.append(_cell_task(
                    cell, patch, mem, cache_dir, None, 0, _profile))
            elif chosen_mode == "inmem":
                cell_tasks.append(_cell_task(
                    cell, patch, mem, cache_dir, "inmem", inmem_b, _profile))
            else:
                bound_b = (
                    gpu_stream_b
                    if (effective_gpu and gpu_stream_b is not None) else stream_b
                )
                cell_tasks.append(_cell_task(
                    cell, patch, mem, cache_dir, "stream", bound_b, _profile))

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
                        cpu_fallback_tasks.append(_cell_task(
                            cell, patch, mem, cache_dir, None, 0, _profile))
                    elif _fb_mode == "inmem":
                        cpu_fallback_tasks.append(_cell_task(
                            cell, patch, mem, cache_dir, "inmem", inmem_b, _profile))
                    else:
                        cpu_fallback_tasks.append(_cell_task(
                            cell, patch, mem, cache_dir, "stream", stream_b, _profile))
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
            gpu_device_workspace_bytes=(
                None
                if gpu_device_workspace_bytes is None
                else int(gpu_device_workspace_bytes)
            ),
            gpu_tile_trials=[dict(t) for t in gpu_tile_trials],
            storage_profile=_profile,
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
        per_unit_cache = _r55_cache_unit_bytes(
            worst_patch_px, max_n_frames, channels=channels
        )
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


# ---------------------------------------------------------------------------
# ZM-ZEGRID-R55-D2 — ACTIVE CPU integration. Legacy and observe paths are
# byte-unchanged; ``active`` is the ONLY mode that can grant execution
# authority, and only to a deterministic, fully mappable CPU winner.
# ---------------------------------------------------------------------------

_R55D2_ACTIVE_MISSION = "ZM-ZEGRID-R55-D2-CPU-ACTIVE-20261023"


@dataclass(frozen=True)
class _R55ActiveDecision:
    """The frozen active decision built BEFORE any gauge/stack work."""

    plan_set: object
    mapping: dict
    facts: dict
    inmem_context: object
    stream_context: object
    phase: str
    # R55-E8: the coherent gauge/cache worker values that actually execute and
    # the bounded trail of every larger pair the same PLAN/margin refused.
    gauge_workers: int = 0
    cache_workers: int = 0
    worker_trail: tuple = ()
    storage_profile: str = zsp.STORAGE_PROFILE_DENSE


def _r55_active_facts(
    *,
    bundle,
    cell_ctxs,
    canvas,
    halo_px,
    cache_root,
    output_dir,
    frame_ids,
    gauge_workers=None,
    cache_workers=None,
) -> dict:
    """Derive the integer facts the accepted scheduler needs (no probe).

    Everything is computed from the INITIAL bundle's frozen layout,
    memberships, frames and halo. No global and no solver state is mutated.
    """
    active = [
        (cell, patch, mem)
        for (_row, _col, cell, patch, mem) in cell_ctxs
        if mem.patch_ids
    ]
    if not active:
        raise zsched.SchedulerConfigError(
            "R55-D2 active requires at least one non-empty cell"
        )
    units_total = len(active)
    worst_patch_px = 0
    worst_patch_hw = (1, 1)
    core_area_px = 0
    patch_area_px = 0
    max_n_frames = 1
    inmem_worst = 0
    stream_worst_by_tile = {
        int(t): 0 for t in CPU_STREAMING_TILE_LADDER
    }
    for cell, patch, mem in active:
        patch_area = int(patch.patch.width) * int(patch.patch.height)
        core = cell.core
        core_area = max(0, int(core.x1) - int(core.x0)) * max(
            0, int(core.y1) - int(core.y0)
        )
        if patch_area > worst_patch_px:
            worst_patch_px = patch_area
            worst_patch_hw = (
                int(patch.patch.height), int(patch.patch.width)
            )
        patch_area_px += patch_area
        core_area_px += core_area
        n = len(mem.patch_ids)
        max_n_frames = max(max_n_frames, n)
        area = int(patch.patch.width) * int(patch.patch.height)
        inmem_worst = max(
            inmem_worst,
            int(zal.FITTED_MEMORY_MODEL.predict_bound_bytes(n, area)),
        )
        hw = tuple(int(v) for v in patch.patch_shape_hw)
        for tile in CPU_STREAMING_TILE_LADDER:
            stream_worst_by_tile[int(tile)] = max(
                stream_worst_by_tile[int(tile)],
                int(_profile_stream_bytes(n, hw, int(tile), _STACK_STORAGE_PROFILE)),
            )

    _profile = zsp.normalize_storage_profile(_STACK_STORAGE_PROFILE)
    # L-1: bounded forbids the in-memory stack in the SCHEDULER (with this real
    # bound), so no synthetic sentinel is needed anywhere in the facts.
    inmem_worst_actual = int(max(1, inmem_worst))

    channels = 3
    per_unit_cache = _r55_cache_unit_bytes(
        worst_patch_px, max_n_frames, channels=channels
    )
    _disk_root = Path(cache_root) if cache_root is not None else Path(output_dir)
    try:
        disk_free = int(shutil.disk_usage(str(_disk_root)).free)
    except Exception:
        disk_free = 0

    canvas_px = int(canvas.width) * int(canvas.height)
    gauge_footprint = int(zpar.gauge_footprint_bytes(canvas_px))
    cache_footprint = int(zpar.cache_footprint_bytes(worst_patch_px))
    _worker_raw_ceiling = _r55_margin_raw_ceiling(
        int(bundle.envelope.budget_bytes),
        margin_num=int(renv.DEFAULT_MARGIN_NUM),
        margin_den=int(renv.DEFAULT_MARGIN_DEN),
    )
    # NEVER grow past the bundle's frozen value: the override (or the bundle
    # value) is an upper bound, and the margin-aware ceiling may only shrink it.
    _gauge_start = int(bundle.gauge_workers) if gauge_workers is None else int(gauge_workers)
    _cache_start = (
        int(bundle.cache_build_workers)
        if cache_workers is None
        else int(cache_workers)
    )
    gauge_workers = _r55_coherent_worker_count(
        _gauge_start, _worker_raw_ceiling, gauge_footprint
    )
    cache_workers = _r55_coherent_worker_count(
        _cache_start, _worker_raw_ceiling, cache_footprint
    )
    bytes_per_px = 4 * 4
    halo_band_bytes = max(
        0, int(patch_area_px) - int(core_area_px)
    ) * bytes_per_px
    retained_core_bytes = int(canvas_px) * bytes_per_px

    facts = {
        "active": active,
        "units_total": int(units_total),
        "worst_patch_px": int(worst_patch_px),
        "worst_patch_hw": (int(worst_patch_hw[0]), int(worst_patch_hw[1])),
        "per_unit_cache": int(per_unit_cache),
        "inmem_worst": int(max(1, inmem_worst)),
        "inmem_worst_actual": int(inmem_worst_actual),
        "stream_worst_by_tile": dict(stream_worst_by_tile),
        "stream_worst": int(max(1, max(stream_worst_by_tile.values()))),
        # R55-E8: ``_estimate_streaming_bytes`` returns the TOTAL streaming peak,
        # which ALREADY includes the 340 MiB process baseline (the same value as
        # ``shared_bytes``). The scheduler counts ``shared_bytes`` once, so the
        # per-cell host bound it consumes must be the ADDED peak above that
        # baseline (the same contract as ``check_streaming_gate``). Passing the
        # total double-counted the baseline and made a valid tile128 plan fail
        # the 15 % margin on the Linux halo16 budgets. Static/in-memory callers
        # are unchanged.
        "stream_added_by_tile": {
            int(t): max(1, int(v) - int(rcontract.PROCESS_BASELINE_BYTES))
            for t, v in stream_worst_by_tile.items()
        },
        "stream_added_worst": int(
            max(
                1,
                max(stream_worst_by_tile.values())
                - int(rcontract.PROCESS_BASELINE_BYTES),
            )
        ),
        "disk_free": int(disk_free),
        "canvas_px": int(canvas_px),
        "channels": int(channels),
        "gauge_footprint_bytes": int(gauge_footprint),
        "cache_footprint_bytes": int(cache_footprint),
        "gauge_workers": int(gauge_workers),
        "cache_workers": int(cache_workers),
        "halo_band_bytes": int(halo_band_bytes),
        "retained_core_bytes": int(retained_core_bytes),
        # B_plan and the ONE margin the global envelope validation applies.
        "budget_bytes": int(bundle.envelope.budget_bytes),
        "margin_num": int(renv.DEFAULT_MARGIN_NUM),
        "margin_den": int(renv.DEFAULT_MARGIN_DEN),
        "requests": (),
        "cell_ids": tuple(cell.cell_id for cell, _p, _m in active),
        "frame_ids": tuple(str(f) for f in (frame_ids or ())),
        "max_n_frames": int(max_n_frames),
        "layout": (
            int(bundle.layout["nx"]), int(bundle.layout["ny"])
        ),
        "refinement_rank": int(bundle.envelope.refinement_rank),
        "halo_px": int(halo_px),
        "pinned": bool(bundle.envelope.pinned),
        "run_id": str(canvas.canvas_id),
        "storage_profile": _profile,
    }
    facts["requests"] = _r55_active_phase_requests(facts)
    return facts


def _r55_active_phase_requests(facts) -> tuple:
    """Build the non-stack phase requests for the CURRENT gauge/cache workers.

    Only the two non-overlapping concurrency phases (gauge, cache_build) vary;
    assembly/export are invariant in the frozen layout. Keeping this in ONE
    helper is what lets selection and global validation use the same numbers.
    """
    gauge_workers = max(1, int(facts["gauge_workers"]))
    cache_workers = max(1, int(facts["cache_workers"]))
    gauge_worker_bytes = int(facts["gauge_footprint_bytes"]) * gauge_workers
    cache_worker_bytes = int(facts["cache_footprint_bytes"]) * cache_workers
    canvas_px = int(facts["canvas_px"])
    return (
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
                "halo_band_bytes": int(facts["halo_band_bytes"]),
                "retained_core_bytes": int(facts["retained_core_bytes"]),
            },
            units=1,
            units_in_flight=1,
            workers_per_unit=1,
            # Rewritten to the CHOSEN stack phase by the joint builder.
            overlaps_with=(),
        ),
        zsched.PhaseRequest(
            phase=renv.PHASE_EXPORT,
            components={
                "coverage_plane_bytes": canvas_px * 4,
                "science_planes_bytes": canvas_px * int(facts["channels"]) * 4,
            },
            units=1,
            units_in_flight=1,
            workers_per_unit=1,
        ),
    )


def _r55_active_context(
    facts,
    *,
    phase,
    per_unit_stack,
    per_unit_cache,
    per_unit_prefetch,
    bundle,
    per_tile_stack_bytes=None,
    cache_workers_cap=None,
) -> object:
    env = bundle.envelope
    workers_ladder = tuple(int(w) for w in zsched.WORKERS_LADDER)
    if cache_workers_cap is not None:
        cap = max(1, int(cache_workers_cap))
        workers_ladder = tuple(w for w in workers_ladder if w <= cap) or (1,)
    return zsched.PlanContext(
        phase=phase,
        backend=renv.BACKEND_CPU,
        budget_bytes=int(env.budget_bytes),
        raw_available_bytes=int(env.raw_available_bytes),
        factor_num=int(env.factor_num),
        factor_den=int(env.factor_den),
        cpu_logical=int(os.cpu_count() or 1),
        units_total=int(facts["units_total"]),
        shared_bytes=int(bundle.shared_bytes),
        per_unit_stack_bytes=int(per_unit_stack),
        per_unit_cache_bytes=int(per_unit_cache),
        per_unit_prefetch_bytes=int(per_unit_prefetch),
        disk_free_bytes=int(facts["disk_free"]),
        layout=tuple(facts["layout"]),
        refinement_rank=int(facts["refinement_rank"]),
        patch_hw=tuple(facts["worst_patch_hw"]),
        halo_px=int(facts["halo_px"]),
        channels=int(facts["channels"]),
        n_frames=int(facts["max_n_frames"]),
        tile_ladder=tuple(int(t) for t in CPU_STREAMING_TILE_LADDER),
        # D2 executable prefetch is EXACTLY zero; R55-E owns prefetch.
        prefetch_ladder=(0,),
        workers_ladder=workers_ladder,
        # R55-E8: when a per-tile host map is supplied, every streaming
        # candidate uses its OWN exact bound (E7 contract), never the static
        # ladder maximum. The in-memory phase keeps its own bound.
        per_tile_stack_bytes=per_tile_stack_bytes,
        margin_num=int(renv.DEFAULT_MARGIN_NUM),
        margin_den=int(renv.DEFAULT_MARGIN_DEN),
        cell_ids=tuple(facts["cell_ids"]),
        frame_ids=tuple(facts["frame_ids"]),
        observation_source="plan.available_bytes",
        run_id=str(facts["run_id"]),
        pinned=bool(facts["pinned"]),
        storage_profile=str(facts.get("storage_profile", _STACK_STORAGE_PROFILE)),
    )


def _r55_active_cpu_contexts(facts, bundle, cache_workers_cap):
    """Build the CPU inmem+stream contexts for ONE cache-worker ceiling.

    R55-E8: the stream context carries the EXACT per-tile host map so every
    streaming candidate is bounded by its own tile (Linux halo16 t128
    1 909 342 208 B vs the old static t32 4 012 813 312 B). The cache-worker
    ceiling keeps the stack's per-cell cache workers within the cache phase
    value that was globally validated.
    """
    inmem_context = _r55_active_context(
        facts,
        phase=renv.PHASE_CELL_STACK_INMEM,
        per_unit_stack=int(facts["inmem_worst"]),
        per_unit_cache=0,
        per_unit_prefetch=0,
        bundle=bundle,
    )
    stream_context = _r55_active_context(
        facts,
        phase=renv.PHASE_CELL_STACK_STREAM,
        per_unit_stack=int(facts["stream_added_worst"]),
        per_unit_cache=int(facts["per_unit_cache"]),
        per_unit_prefetch=int(facts["per_unit_cache"]),
        bundle=bundle,
        per_tile_stack_bytes=dict(facts["stream_added_by_tile"]),
        cache_workers_cap=int(cache_workers_cap),
    )
    return inmem_context, stream_context


def _r55_worker_adaptation_detail(facts, trail):
    return {
        "schema": "zemosaic-r55e8-worker-adaptation-v1",
        "budget_bytes": int(facts["budget_bytes"]),
        "margin": [int(facts["margin_num"]), int(facts["margin_den"])],
        "gauge_footprint_bytes": int(facts["gauge_footprint_bytes"]),
        "cache_footprint_bytes": int(facts["cache_footprint_bytes"]),
        "gauge_ladder": list(_r55_worker_ladder(int(facts["gauge_workers"]))),
        "cache_ladder": list(_r55_worker_ladder(int(facts["cache_workers"]))),
        "selected_gauge_workers": int(facts["gauge_workers"]),
        "selected_cache_workers": int(facts["cache_workers"]),
        "trail": [dict(entry) for entry in trail],
    }


def _r55_active_decide(
    *, bundle, cell_ctxs, canvas, halo_px, cache_root, output_dir, frame_ids
) -> _R55ActiveDecision:
    """Build the deterministic CPU inmem+stream decision BEFORE gauge/stack.

    R55-E8: gauge and cache_build are non-overlapping concurrency phases. The
    pair is selected from bounded deterministic descending ladders down to 1
    under the SAME B_plan and 15 % margin that :func:`build_plan_set` will apply,
    so a larger pair can never be frozen and then refused by the validator.
    """
    facts = _r55_active_facts(
        bundle=bundle,
        cell_ctxs=cell_ctxs,
        canvas=canvas,
        halo_px=halo_px,
        cache_root=cache_root,
        output_dir=output_dir,
        frame_ids=frame_ids,
    )
    trail: list[dict] = []
    last_exc = None
    plan_set = None
    inmem_context = None
    stream_context = None
    for gauge in _r55_worker_ladder(int(facts["gauge_workers"])):
        for cache in _r55_worker_ladder(int(facts["cache_workers"])):
            trial = dict(facts)
            trial["gauge_workers"] = int(gauge)
            trial["cache_workers"] = int(cache)
            requests = _r55_active_phase_requests(trial)
            inmem_context, stream_context = _r55_active_cpu_contexts(
                trial, bundle, int(cache)
            )
            joint = zsched.JointPlanInput(
                contexts=(inmem_context, stream_context)
            )
            try:
                candidate_plan = zsched.plan_joint(
                    joint,
                    phase_requests=requests,
                    mode=zsched.MODE_ACTIVE,
                )
            except zsched.PlanRefused as exc:
                last_exc = exc
                trail.append(
                    {
                        "at": "worker_selection",
                        "gauge_workers": int(gauge),
                        "cache_workers": int(cache),
                        "accepted": False,
                        "reason_code": exc.reason_code,
                        "detail": exc.detail,
                        "gauge_phase_bytes": (
                            int(trial["gauge_footprint_bytes"]) * int(gauge)
                        ),
                        "cache_phase_bytes": (
                            int(trial["cache_footprint_bytes"]) * int(cache)
                        ),
                    }
                )
                continue
            plan_set = candidate_plan
            facts = trial
            trail.append(
                {
                    "at": "worker_selection",
                    "gauge_workers": int(gauge),
                    "cache_workers": int(cache),
                    "accepted": True,
                    "reason_code": "R55_OK",
                    "aggregate_host_peak_bytes": int(
                        plan_set.aggregate_host_peak_bytes
                    ),
                }
            )
            break
        if plan_set is not None:
            break
    if plan_set is None:
        reason = (
            zsched.REASON_REFUSE_HOST_RAM
            if last_exc is None
            else last_exc.reason_code
        )
        detail = (
            "no coherent gauge/cache worker configuration fits B_plan with the "
            f"{facts['margin_num']}/{facts['margin_den']} margin after the full "
            f"bounded ladders (gauge {list(_r55_worker_ladder(int(facts['gauge_workers'])))}"
            f" x cache {list(_r55_worker_ladder(int(facts['cache_workers'])))}): "
            + ("" if last_exc is None else str(last_exc.detail))
        )
        raise zsched.PlanRefused(
            reason,
            detail,
            candidates=() if last_exc is None else last_exc.candidates,
        )
    mapping = zsched.map_active_cpu_candidate(
        plan_set,
        tile_ladder=tuple(int(t) for t in CPU_STREAMING_TILE_LADDER),
        workers_ladder=tuple(int(w) for w in zsched.WORKERS_LADDER),
    )
    expected_identity = zsched.FrozenIdentity(
        layout=tuple(facts["layout"]),
        cell_ids=tuple(sorted(facts["cell_ids"])),
        frame_ids=tuple(sorted(facts["frame_ids"])),
        halo_px=int(facts["halo_px"]),
        refinement_rank=int(facts["refinement_rank"]),
    )
    zsched.assert_same_identity(expected_identity, plan_set.identity)
    # Structural exactness: the mapped values ARE the winner's values.
    candidate = plan_set.chosen
    if mapping["units_in_flight"] != int(candidate.units_in_flight):
        raise zsched.ActivePlanRejected(
            renv.REASON_REFUSE_HOST_RAM,
            "mapped units_in_flight does not equal the winner",
            candidate=candidate,
            mapping=mapping,
        )
    if mapping["workers_per_unit"] != int(candidate.workers_per_unit):
        raise zsched.ActivePlanRejected(
            renv.REASON_SHRINK_WORKERS,
            "mapped workers_per_unit does not equal the winner",
            candidate=candidate,
            mapping=mapping,
        )
    if mapping["cpu_streaming_tile"] != (
        None if candidate.tile_px is None else int(candidate.tile_px)
    ):
        raise zsched.ActivePlanRejected(
            renv.REASON_SHRINK_TILE,
            "mapped tile does not equal the winner tile",
            candidate=candidate,
            mapping=mapping,
        )
    return _R55ActiveDecision(
        plan_set=plan_set,
        mapping=mapping,
        facts=facts,
        inmem_context=inmem_context,
        stream_context=stream_context,
        phase=zsched.active_cpu_phase(candidate),
        gauge_workers=int(facts["gauge_workers"]),
        cache_workers=int(facts["cache_workers"]),
        worker_trail=tuple(dict(entry) for entry in trail),
        storage_profile=str(facts.get("storage_profile", _STACK_STORAGE_PROFILE)),
    )




# ---------------------------------------------------------------------------
# ZM-ZEGRID-R55-E2 — ACTIVE GPU decision, bounded pipeline and exact fallback.
# ---------------------------------------------------------------------------

_R55E2_GPU_MISSION = "ZM-ZEGRID-R55-E2-ACTIVE-GPU-PREFETCH-20261024"
_R55E2_PREFETCH_LADDER = (0, 1, 2, 3, 4)
_R55E2_PREFETCH_DEFAULT = 2

# Finite bound for joining the ASYNC PRODUCER once the pipeline has reached a
# terminal state (sentinel, error or cancellation). It is NOT a per-unit work
# budget: the GPU consumer runs on the calling thread, so a healthy cell that
# legitimately takes minutes is never cancelled by a fixed join window (the
# real RTX 3070 false-timeout witnessed at 15ada5c). A producer still alive
# after this bound is a genuine stuck thread and fails closed.
_R55E2_PIPELINE_JOIN_S = 60.0

# R55-E10: SELECTION-TIME VRAM headroom for the ACTIVE GPU streaming tile.
#
# R55-E9 derived the ladder top from the full fresh 70 % VRAM budget, so the
# SELECTED tile filled that budget almost exactly (recorded M106 witness: 1.2 MiB
# of margin). The device workspace is only an ESTIMATE (worst-case per-cell bytes
# over the halo-expanded tile area plus a fixed overhead), free VRAM fluctuates
# while the CuPy context/allocator warms up, and a miss is not recoverable in
# place: an OOM forces a whole-batch CPU retry. RUN-time revalidation re-reads
# fresh free VRAM once the CuPy context is live and saw a ~2 MiB drop on the
# recorded run, which was enough to reject the razor-edge tile. The selected
# tile must therefore fit only 85 % of that 70 % budget (~60 % of fresh free
# VRAM), leaving a real safety margin on every device.
#
# This constant bounds SELECTION only. The 70 % qualification, the frozen policy
# ladder and the RUN-time ``fresh_vram_revalidation`` are unchanged and remain
# the hard backstop.
_R55E10_VRAM_SELECTION_HEADROOM_PERCENT = 85


class _R55GpuRefused(RuntimeError):
    """A GPU candidate is not qualified; refused with a CLOSED reason."""

    def __init__(self, reason_code, detail, *, record=None):
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = str(reason_code)
        self.detail = str(detail)
        self.record = dict(record or {})


class _R55GpuFallback(RuntimeError):
    """A qualified GPU run failed at RUN time; one whole-batch CPU retry."""

    def __init__(self, reason_code, detail, *, telemetry=None):
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = str(reason_code)
        self.detail = str(detail)
        self.telemetry = dict(telemetry or {})


def _r55_active_gpu_tile_ladder(max_tile):
    """R55-E9: deterministic descending GPU streaming-tile ladder.

    ``max_tile`` is the EXACT interior side already derived from the fresh VRAM
    budget (and clamped to the patch geometry) by
    :func:`zemosaic.core.zegrid.gpu.choose_gpu_tile_size`. The ladder keeps that
    exact maximum as its top rung, then walks coarse 64-px-aligned rungs down to
    128, then the canonical CPU tail (<= 128, which the streaming executor
    already supports). It is device-agnostic (no model name, no fixed device
    ceiling) and finite: the rung count is bounded by ``max_tile // 64 +
    len(CPU_STREAMING_TILE_LADDER) + 1``.

    The result is STRICTLY descending and deduplicated so the existing scheduler
    contracts (declared-ladder membership, no growth, no duplicates) apply
    unchanged. Bounds are never interpolated here: the caller builds the exact
    per-tile host/device maps for EVERY returned rung.
    """
    top = zrt.strict_int(max_tile, "max_tile", minimum=1)
    if top < int(CPU_STREAMING_TILE_MINIMUM):
        return ()
    rungs = [int(top)]
    coarse = (int(top) // 64) * 64
    while coarse >= 128:
        if coarse < rungs[-1]:
            rungs.append(int(coarse))
        coarse -= 64
    for tail_tile in CPU_STREAMING_TILE_LADDER:
        if int(tail_tile) < rungs[-1]:
            rungs.append(int(tail_tile))
    return tuple(rungs)


@dataclass(frozen=True)
class _R55GpuActiveDecision:
    """The frozen ACTIVE GPU decision built before any gauge/stack work."""

    plan_set: object
    mapping: dict
    facts: dict
    joint: object
    gpu_facts: object
    gpu_context: object
    qualification: dict
    policy: object
    inmem_context: object
    stream_context: object
    device_peak_bytes_by_tile: dict
    # R55-E8: coherent gauge/cache workers + bounded adaptation trail.
    gauge_workers: int = 0
    cache_workers: int = 0
    worker_trail: tuple = ()
    # R55-E9: VRAM-derived max, declared GPU ladder and exact per-tile bounds.
    gpu_tile_max: int = 0
    gpu_tile_ladder: tuple = ()
    gpu_host_bound_by_tile: tuple = ()
    # R55-E10: the headroom-bounded budget the ladder was pruned against.
    gpu_selection_budget_bytes: int = 0
    # R55-E11: the exact facts the per-tile host bound is a function of. Persisted
    # so a RUN-time shrink can recompute the SAME bound for a smaller rung
    # instead of interpolating, guessing, or consulting a second planner.
    gpu_max_contributors: int = 1
    gpu_worst_patch_hw: tuple = (1, 1)


def _r55_gpu_host_bound_by_tile(*, max_n, worst_hw, tiles):
    """EXACT per-tile host bound for ACTIVE GPU admission (ADDED peak).

    ZM-ZEGRID-R56-GPU-HOST-BOUND: the bound MUST use the run's storage profile.
    The dense estimator prices N resident frames; under the bounded profile the
    real residency is ~2 frames, so pricing dense here refused every GPU
    candidate on a 341-frame corpus (minimum dense bound 14.03 GiB vs a 13.5 GiB
    PLAN budget) even though the bounded bound is 2.48 GiB at tile 128. The
    value stays the ADDED peak above the process baseline (the scheduler counts
    ``shared_bytes`` once).
    """
    return {
        int(t): max(
            1,
            int(_profile_stream_bytes(
                max_n, worst_hw, int(t), _STACK_STORAGE_PROFILE
            ))
            - int(rcontract.PROCESS_BASELINE_BYTES),
        )
        for t in tiles
    }


def _r55_active_gpu_decide(
    *,
    bundle,
    cell_ctxs,
    canvas,
    halo_px,
    cache_root,
    output_dir,
    frame_ids,
    gpu_probe,
    readiness,
):
    """Build the ACTIVE GPU decision from explicit integer facts only.

    No CuPy import happens here: ``gpu_probe`` and ``readiness`` come from the
    single explicit probe already performed by the product GPU selection. A
    non-qualified candidate is REFUSED with a closed reason (never coerced).
    """
    base = _r55_active_facts(
        bundle=bundle,
        cell_ctxs=cell_ctxs,
        canvas=canvas,
        halo_px=halo_px,
        cache_root=cache_root,
        output_dir=output_dir,
        frame_ids=frame_ids,
    )
    free = gpu_probe.get("vram_free_bytes")
    total = gpu_probe.get("vram_total_bytes")
    device_name = gpu_probe.get("device")
    if free is None or total is None:
        raise _R55GpuRefused(
            zgr.REASON_GPU_OBSERVATION_FAILED,
            "the GPU probe did not return integer VRAM total/free bytes",
        )
    free = zrt.strict_int(free, "vram_free_bytes", minimum=1)
    total = zrt.strict_int(total, "vram_total_bytes", minimum=free)
    policy = zgr.VramPolicy()
    initial = policy.candidate(free, policy.initial_percent)
    budget = int(initial.budget_bytes)
    # R55-E10: the tile ladder is bounded by a budget that already reserves the
    # selection headroom. The 70 % policy budget itself is untouched.
    selection_budget = (
        budget * _R55E10_VRAM_SELECTION_HEADROOM_PERCENT
    ) // 100

    max_n = 1
    worst_hw = (1, 1)
    worst_area = -1
    for (_cell, patch, mem) in base["active"]:
        n = len(mem.patch_ids)
        hw = tuple(int(v) for v in patch.patch_shape_hw)
        area = int(hw[0]) * int(hw[1])
        if area > worst_area:
            worst_area = area
            worst_hw = (int(hw[0]), int(hw[1]))
        max_n = max(max_n, int(n))

    max_tile = zgpu.choose_gpu_tile_size(max_n, 3, worst_hw, budget)
    if max_tile is None or int(max_tile) < int(CPU_STREAMING_TILE_MINIMUM):
        raise _R55GpuRefused(
            zgr.REASON_REFUSE_VRAM,
            "no GPU streaming tile fits the fresh VRAM budget "
            f"({budget} B); refusing rather than extrapolating",
        )
    max_tile = int(max_tile)
    # R55-E9: the EXECUTABLE ACTIVE GPU ladder is GPU-specific and derives its
    # top rung from the FRESH-VRAM maximum (already clamped to the patch
    # geometry). The historical max128 intersection with the CPU streaming
    # ladder discarded that capacity on every device; it is deliberately gone.
    # Coarse 64-px-aligned rungs down to 128 plus the canonical CPU tail keep
    # the ladder finite, strictly descending and device-agnostic. The existing
    # scheduler still decides feasibility (host PLAN/margin + device budget)
    # and may only shrink along these declared rungs.
    declared_tiles = _r55_active_gpu_tile_ladder(max_tile)
    if not declared_tiles:
        raise _R55GpuRefused(
            zgr.REASON_REFUSE_VRAM,
            "no GPU streaming tile is >= the minimum tile "
            f"({CPU_STREAMING_TILE_MINIMUM}) within the VRAM-derived maximum "
            f"({max_tile}); refusing rather than extrapolating",
        )
    declared_device_peaks = {
        int(t): int(
            zgpu.gpu_tile_workspace_bytes(
                max_n,
                int(t) + 2 * int(zgpu.GPU_TAPER_HALO_PX),
                int(t) + 2 * int(zgpu.GPU_TAPER_HALO_PX),
                3,
            )
        )
        for t in declared_tiles
    }
    # R55-E10: keep only rungs whose EXACT device workspace fits the headroom
    # budget. Rungs are never interpolated or resized: the declared ladder is a
    # pruned subset of the R55-E9 ladder, so the scheduler may still only shrink
    # along declared rungs and every rung keeps its own exact host/device bounds.
    gpu_tiles = tuple(
        int(t)
        for t in declared_tiles
        if int(declared_device_peaks[int(t)]) <= int(selection_budget)
    )
    if not gpu_tiles:
        raise _R55GpuRefused(
            zgr.REASON_REFUSE_VRAM,
            "no GPU streaming tile leaves the R55-E10 selection headroom "
            f"({_R55E10_VRAM_SELECTION_HEADROOM_PERCENT}% of the fresh "
            f"{policy.initial_percent}% VRAM budget = {selection_budget} B) "
            f"within the VRAM-derived maximum ({max_tile}); refusing rather "
            "than extrapolating",
        )
    device_peaks = {
        int(t): int(declared_device_peaks[int(t)]) for t in gpu_tiles
    }
    # R55-E8: the GPU host bound is the ADDED streaming peak (baseline already
    # counted once as ``shared_bytes``), never the total that double-counts it.
    gpu_host_bound_by_tile = _r55_gpu_host_bound_by_tile(
        max_n=max_n, worst_hw=worst_hw, tiles=gpu_tiles
    )

    qualification = zgr.qualify_gpu_execution(
        requested=True,
        readiness=readiness,
        parity_qualified=True,
        free_bytes=free,
        policy=policy,
        device_peak_bytes=max(device_peaks.values()),
        prefetch_units=_R55E2_PREFETCH_LADDER[-1],
        queue_max=_R55E2_PREFETCH_LADDER[-1] + 1,
        units_in_flight=1,
    )
    if not qualification.qualified:
        raise _R55GpuRefused(
            qualification.reason_code,
            qualification.detail,
            record={"qualification": qualification.to_dict()},
        )

    gpu_facts = zsched.GpuFacts(
        requested=True,
        available=bool(gpu_probe.get("available")),
        parity_qualified=True,
        vram_total_bytes=int(total),
        vram_free_bytes=int(free),
        vram_budget_bytes=int(budget),
        device_name=str(device_name or "unknown-gpu"),
        device_peak_bytes_by_tile=tuple(
            (int(t), int(device_peaks[int(t)])) for t in gpu_tiles
        ),
        prefetch_overlap_measured=False,
    )

    def _gpu_pair_contexts(gauge_workers, cache_workers, trial_facts):
        """Build CPU inmem/stream + GPU contexts for ONE worker pair."""
        in_c, st_c = _r55_active_cpu_contexts(
            trial_facts, bundle, int(cache_workers)
        )
        gpu_c = replace(
            st_c,
            backend=zsched.BACKEND_GPU,
            tile_ladder=tuple(int(t) for t in gpu_tiles),
            prefetch_ladder=tuple(int(p) for p in _R55E2_PREFETCH_LADDER),
            # R55-E7: ``per_unit_stack_bytes`` is a schema-required placeholder
            # only; the explicit per-tile map below is the SOLE host bound every
            # candidate consults. An incomplete/extra map is rejected by the
            # scheduler, so there is no static fallback for a GPU tile.
            # Schema-required placeholder only; the per-tile map is the sole
            # host bound. Use the largest EXECUTABLE ladder tile.
            per_unit_stack_bytes=int(gpu_host_bound_by_tile[int(gpu_tiles[0])]),
            per_tile_stack_bytes=tuple(
                (int(t), int(gpu_host_bound_by_tile[int(t)]))
                for t in gpu_tiles
            ),
            per_unit_prefetch_bytes=int(base["per_unit_cache"]),
            time_model=None,
            profile=None,
        )
        return in_c, st_c, gpu_c

    # Initial GPU feasibility is independent of the gauge/cache worker pair
    # (workers do not change the exact host/device peaks). Determine it once,
    # then adapt only the non-overlapping concurrency phases.
    _init_in, _init_st, _init_gpu = _gpu_pair_contexts(
        base["gauge_workers"], base["cache_workers"], base
    )
    _init_joint = zsched.JointPlanInput(
        contexts=(_init_in, _init_st, _init_gpu),
        gpu_facts=gpu_facts,
    )
    enumerated = zsched.enumerate_joint_candidates(_init_joint)
    feasible_gpu = [
        candidate for candidate in enumerated
        if candidate.backend == zsched.BACKEND_GPU and candidate.feasible
    ]
    if not feasible_gpu:
        trail = zsched.joint_refusal_trail(_init_joint)
        reason, detail, exhaustion = zsched.gpu_exhaustion_reason(
            _init_joint, enumerated
        )
        raise _R55GpuRefused(
            reason,
            detail,
            record={
                "refusal_trail": [e.to_dict() for e in trail],
                "gpu_exhaustion": dict(exhaustion),
                "gpu_host_bound_bytes_by_tile": {
                    int(t): int(gpu_host_bound_by_tile[int(t)])
                    for t in gpu_tiles
                },
                "device_peak_bytes_by_tile": {
                    int(t): int(device_peaks[int(t)]) for t in gpu_tiles
                },
                "b_plan_bytes": int(bundle.envelope.budget_bytes),
                "vram_budget_bytes": int(budget),
            },
        )

    # R55-E8: adapt gauge/cache under the SAME B_plan + margin that
    # build_joint_plan_set validates. Prefer the largest safe pair.
    worker_trail: list[dict] = []
    last_exc = None
    plan_set = None
    joint = None
    inmem_context = None
    stream_context = None
    gpu_context = None
    for gauge in _r55_worker_ladder(int(base["gauge_workers"])):
        for cache in _r55_worker_ladder(int(base["cache_workers"])):
            trial = dict(base)
            trial["gauge_workers"] = int(gauge)
            trial["cache_workers"] = int(cache)
            requests = _r55_active_phase_requests(trial)
            in_c, st_c, gpu_c = _gpu_pair_contexts(
                int(gauge), int(cache), trial
            )
            trial_joint = zsched.JointPlanInput(
                contexts=(in_c, st_c, gpu_c),
                gpu_facts=gpu_facts,
            )
            trial_enumerated = zsched.enumerate_joint_candidates(trial_joint)
            trial_feasible = [
                candidate
                for candidate in trial_enumerated
                if candidate.backend == zsched.BACKEND_GPU
                and candidate.feasible
            ]
            if not trial_feasible:
                worker_trail.append(
                    {
                        "at": "worker_selection",
                        "gauge_workers": int(gauge),
                        "cache_workers": int(cache),
                        "accepted": False,
                        "reason_code": zsched.REASON_REFUSE_HOST_RAM,
                        "detail": "no feasible GPU candidate under this "
                        "worker ceiling",
                    }
                )
                continue
            target = min(
                _R55E2_PREFETCH_DEFAULT,
                max(int(c.prefetch_units) for c in trial_feasible),
            )
            # R55-E8: the GPU ladder is walked largest-first, exactly like the
            # legacy product's finite tile ladder. Among jointly feasible tiles
            # the LARGEST wins (the corrected exact bounds are non-monotone, so
            # tile96 can have a marginally lower host peak than tile128); then
            # the bounded prefetch target, then the scheduler objective.
            # R55-E9: a candidate that fits the scheduler's OWN peak can still
            # miss the SAME B_plan once the non-overlapping gauge/cache phases
            # are added. Walk the ordered candidates and keep the FIRST whose
            # exact joint plan admits, instead of refusing the whole GPU plan
            # because the top candidate's preferred prefetch missed.
            ordered = sorted(
                trial_feasible,
                key=lambda c: (
                    -int(c.tile_px),
                    0 if int(c.prefetch_units) == int(target) else 1,
                    -int(c.prefetch_units),
                    zsched.objective_key(c),
                ),
            )
            candidate_plan = None
            chosen = None
            for option in ordered:
                try:
                    candidate_plan = zsched.build_joint_plan_set(
                        trial_joint,
                        candidate=option,
                        phase_requests=requests,
                        mode=zsched.MODE_ACTIVE,
                    )
                except zsched.PlanRefused as exc:
                    last_exc = exc
                    continue
                chosen = option
                break
            if candidate_plan is None:
                worker_trail.append(
                    {
                        "at": "worker_selection",
                        "gauge_workers": int(gauge),
                        "cache_workers": int(cache),
                        "accepted": False,
                        "reason_code": (
                            last_exc.reason_code
                            if last_exc is not None
                            else zsched.REASON_REFUSE_HOST_RAM
                        ),
                        "detail": (
                            last_exc.detail
                            if last_exc is not None
                            else "no jointly feasible GPU candidate"
                        ),
                        "gauge_phase_bytes": (
                            int(trial["gauge_footprint_bytes"]) * int(gauge)
                        ),
                        "cache_phase_bytes": (
                            int(trial["cache_footprint_bytes"]) * int(cache)
                        ),
                    }
                )
                continue
            plan_set = candidate_plan
            base = trial
            joint = trial_joint
            inmem_context, stream_context, gpu_context = in_c, st_c, gpu_c
            worker_trail.append(
                {
                    "at": "worker_selection",
                    "gauge_workers": int(gauge),
                    "cache_workers": int(cache),
                    "accepted": True,
                    "reason_code": "R55_OK",
                    "aggregate_host_peak_bytes": int(
                        plan_set.aggregate_host_peak_bytes
                    ),
                    "aggregate_device_peak_bytes": int(
                        plan_set.aggregate_device_peak_bytes
                    ),
                }
            )
            break
        if plan_set is not None:
            break
    if plan_set is None:
        detail = (
            "no coherent gauge/cache worker configuration fits the SAME B_plan "
            f"with the {base['margin_num']}/{base['margin_den']} margin after "
            "the full bounded ladders"
        )
        if last_exc is not None:
            detail = detail + ": " + str(last_exc.detail)
        raise _R55GpuRefused(
            zsched.REASON_REFUSE_HOST_RAM,
            detail,
            record={
                "worker_adaptation": _r55_worker_adaptation_detail(
                    base, worker_trail
                ),
                "b_plan_bytes": int(bundle.envelope.budget_bytes),
                "vram_budget_bytes": int(budget),
            },
        )
    mapping = zsched.map_active_gpu_candidate(
        plan_set,
        tile_ladder=tuple(int(t) for t in gpu_tiles),
        workers_ladder=tuple(int(w) for w in zsched.WORKERS_LADDER),
        prefetch_cap=int(_R55E2_PREFETCH_LADDER[-1]),
    )
    # Re-qualify with the ACTUAL mapped tile/prefetch so the recorded contract
    # is the one that executes (never a looser pre-selection bound).
    qualification = zgr.qualify_gpu_execution(
        requested=True,
        readiness=readiness,
        parity_qualified=True,
        free_bytes=free,
        policy=policy,
        device_peak_bytes=int(
            device_peaks[int(mapping["gpu_streaming_tile"])]
        ),
        prefetch_units=int(mapping["prefetch_units"]),
        queue_max=int(mapping["prefetch_units"]) + 1,
        units_in_flight=1,
    )
    if not qualification.qualified:
        raise _R55GpuRefused(
            qualification.reason_code,
            qualification.detail,
            record={"qualification": qualification.to_dict()},
        )
    return _R55GpuActiveDecision(
        plan_set=plan_set,
        mapping=mapping,
        facts=base,
        joint=joint,
        gpu_facts=gpu_facts,
        gpu_context=gpu_context,
        qualification=qualification.to_dict(),
        policy=policy,
        inmem_context=inmem_context,
        stream_context=stream_context,
        device_peak_bytes_by_tile=dict(device_peaks),
        gauge_workers=int(base["gauge_workers"]),
        cache_workers=int(base["cache_workers"]),
        worker_trail=tuple(dict(entry) for entry in worker_trail),
        gpu_tile_max=int(max_tile),
        gpu_tile_ladder=tuple(int(t) for t in gpu_tiles),
        gpu_selection_budget_bytes=int(selection_budget),
        gpu_host_bound_by_tile=tuple(
            (int(t), int(gpu_host_bound_by_tile[int(t)])) for t in gpu_tiles
        ),
        gpu_max_contributors=int(max_n),
        gpu_worst_patch_hw=(int(worst_hw[0]), int(worst_hw[1])),
    )


def _r55_gpu_candidate_lookup(decision, candidate):
    """Find the EXACT enumerated joint candidate for a tuple (or fail closed)."""
    key = candidate.ordering_key()
    for item in zsched.enumerate_joint_candidates(decision.joint):
        if item.ordering_key() == key:
            return item
    raise zsched.ActivePlanRejected(
        renv.REASON_INTERNAL_ERROR,
        f"active GPU tuple {key!r} is not in the joint enumeration",
        candidate=candidate,
    )


def _r55_next_gpu_shrink(decision, candidate, context):
    """Return the next FEASIBLE GPU shrink in the fixed global order."""
    current = candidate
    while True:
        name, nxt = zsched.next_shrink(current, context)
        if name is None or nxt is None:
            return None, None
        if nxt.backend != zsched.BACKEND_GPU:
            return None, None
        materialized = _r55_gpu_candidate_lookup(decision, nxt)
        zsched.assert_joint_shrink_only(current, materialized)
        if materialized.feasible:
            return name, materialized
        current = materialized


def _r55_next_gpu_vram_shrink(decision, candidate, *, free_bytes, percent):
    """R55-E9: next lower DECLARED GPU tile whose exact device peak fits.

    Walks the ladder frozen at planning time strictly DOWNWARD from the current
    tile (never up, never off-ladder, even if VRAM has recovered meanwhile) and
    returns the first rung that is jointly feasible AND a legal shrink. The
    fresh VRAM budget is evaluated with the SAME policy percent the run
    revalidation uses, against each rung's OWN exact
    ``gpu_tile_workspace_bytes`` — no interpolated or shared bound. ``None``
    means no lower rung fits, so the caller performs the qualified whole-batch
    CPU fallback.
    """
    ladder = tuple(int(t) for t in decision.gpu_context.tile_ladder)
    current_tile = int(candidate.tile_px)
    if current_tile not in ladder:
        return None
    budget = int(
        decision.policy.candidate(int(free_bytes), int(percent)).budget_bytes
    )
    for tile in ladder[ladder.index(current_tile) + 1:]:
        tile = int(tile)
        peak = int(decision.device_peak_bytes_by_tile[int(tile)])
        if peak > budget:
            continue
        probe = replace(candidate, tile_px=tile)
        materialized = _r55_gpu_candidate_lookup(decision, probe)
        try:
            zsched.assert_joint_shrink_only(candidate, materialized)
        except zsched.GrowthRejected:
            # A lower tile whose exact host bound is marginally higher (the
            # corrected bounds are not monotone) is not a legal shrink.
            continue
        if materialized.feasible:
            return materialized
    return None


def _r55_active_gpu_apply(
    bundle,
    *,
    mapping,
    decision,
    stage="preflight",
    plan_raw_available_bytes=None,
    plan_factor_num=None,
    plan_factor_den=None,
    plan_budget_bytes=None,
):
    """Apply an EXACT active GPU mapping to the frozen bundle."""
    if mapping.get("backend") != renv.BACKEND_GPU:
        raise zsched.ActivePlanRejected(
            renv.REASON_UNKNOWN_BACKEND,
            f"active GPU apply got backend={mapping.get('backend')!r}",
            mapping=mapping,
        )
    tile = int(mapping["gpu_streaming_tile"])
    workers = int(mapping["workers_per_unit"])
    units = int(mapping["units_in_flight"])
    prefetch = int(mapping["prefetch_units"])
    if units != 1:
        raise zsched.ActivePlanRejected(
            renv.REASON_REFUSE_HOST_RAM,
            f"active GPU execution is single-owner, got K={units}",
            mapping=mapping,
        )
    if prefetch < 0 or prefetch > _R55E2_PREFETCH_LADDER[-1]:
        raise zsched.ActivePlanRejected(
            renv.REASON_SHRINK_PREFETCH,
            f"active GPU prefetch {prefetch} outside the E2 bounded ladder",
            mapping=mapping,
        )
    if tile not in tuple(decision.gpu_context.tile_ladder):
        raise zsched.ActivePlanRejected(
            renv.REASON_SHRINK_TILE,
            f"active GPU tile {tile} is outside the declared GPU ladder",
            mapping=mapping,
        )

    cell_specs = []
    cell_tasks = []
    for (cell, patch, mem, cache_dir, _i, _s, _g) in bundle.cell_specs:
        if not mem.patch_ids:
            cell_specs.append((cell, patch, mem, cache_dir, None, 0, 0))
            cell_tasks.append(_cell_task(
                cell, patch, mem, cache_dir, None, 0, getattr(bundle, "storage_profile", _STACK_STORAGE_PROFILE)))
            continue
        n = len(mem.patch_ids)
        # TOTAL per-cell estimate: an execution/manifest input (includes the
        # process baseline). It is NOT the scheduler's host increment.
        gpu_b = int(_profile_stream_bytes(
            n, patch.patch_shape_hw, tile, getattr(bundle, "storage_profile", _STACK_STORAGE_PROFILE)))
        cell_specs.append((cell, patch, mem, cache_dir, None, 0, gpu_b))
        cell_tasks.append(_cell_task(
            cell, patch, mem, cache_dir, "stream", gpu_b, getattr(bundle, "storage_profile", _STACK_STORAGE_PROFILE)))

    # R55-E11 rework-3: planning uses the ADDED stack bound (estimate minus
    # PROCESS_BASELINE) because ``shared_bytes`` already carries that baseline
    # once. Using the TOTAL here added the baseline a second time (exactly
    # +356,515,840 B on the M106 witness). Keep the total for the execution
    # task above; use the authoritative added bound for every envelope / plan /
    # aggregate field that claims scheduler consistency.
    added_per_cell_stack = _r55_gpu_added_host_bound(decision, tile)
    per_cell_total = max(
        (int(t[5]) for t in cell_tasks if t[4] is not None), default=0
    )
    # Diagnostic/execution total; the envelope below uses the ADDED bound.
    per_cell_footprint = int(per_cell_total)
    base_env = bundle.envelope
    bound_raw = (
        int(base_env.raw_available_bytes)
        if plan_raw_available_bytes is None else int(plan_raw_available_bytes)
    )
    bound_fnum = (
        int(base_env.factor_num)
        if plan_factor_num is None else int(plan_factor_num)
    )
    bound_fden = (
        int(base_env.factor_den)
        if plan_factor_den is None else int(plan_factor_den)
    )
    bound_budget = (
        int(base_env.budget_bytes)
        if plan_budget_bytes is None else int(plan_budget_bytes)
    )
    if bound_fnum < 1 or bound_fden < 1:
        raise rcontract.InvalidEnvelope(
            f"invalid plan factor {bound_fnum}/{bound_fden}"
        )
    if bound_budget != (bound_fnum * bound_raw) // bound_fden:
        raise rcontract.InvalidEnvelope(
            "rebound plan budget does not equal (factor*raw)//den: "
            f"{bound_budget} != ({bound_fnum}*{bound_raw})//{bound_fden}"
        )
    stack_total = int(
        rcontract.stack_phase_total_bytes(
            int(bundle.shared_bytes), units, int(added_per_cell_stack)
        )
    )
    cache_per_cell = int(
        rcontract.cache_per_cell_bytes(
            rcontract.PROCESS_BASELINE_BYTES,
            workers,
            int(bundle.cache_worker_footprint_bytes),
        )
    )
    cache_total = int(
        rcontract.cache_phase_total_bytes(
            int(bundle.shared_bytes), units, cache_per_cell
        )
    )
    aggregate = int(
        rcontract.aggregate_phase_bytes(stack_total, cache_total)
    )
    reason = (
        "R55-E2 active GPU mapping: "
        f"phase={mapping['phase']} tile={tile} K=1 workers={workers} "
        f"prefetch_units={prefetch} backend=gpu"
    )
    envelope = rcontract.PlanEnvelope(
        phase=rcontract.PHASE_PLAN,
        raw_available_bytes=int(bound_raw),
        factor_num=int(bound_fnum),
        factor_den=int(bound_fden),
        budget_bytes=int(bound_budget),
        layout=(int(base_env.layout[0]), int(base_env.layout[1])),
        refinement_rank=int(base_env.refinement_rank),
        cpu_tile=0,
        gpu_tile=int(tile),
        backend=rcontract.BACKEND_GPU,
        cells_in_flight=1,
        cache_workers_per_cell=int(workers),
        shared_bytes=int(bundle.shared_bytes),
        stack_total_bytes=int(stack_total),
        cache_total_bytes=int(cache_total),
        aggregate_bytes=int(aggregate),
        pinned=bool(base_env.pinned),
        reason=reason,
    )
    rcontract.validate_envelope(envelope)
    plan = {
        "mode": "stream",
        "cells_in_flight": 1,
        "cache_workers_per_cell": int(workers),
        "gpu_serialized": True,
        "chosen_reason": reason,
        "candidates": [
            {
                "mode": "stream",
                "cells_in_flight": 1,
                "max_bound_bytes": int(added_per_cell_stack),
                "feasible": True,
                "makespan_rel": None,
                "infeasible_reason": None,
            }
        ],
        "stack_total_bytes": int(stack_total),
        "cache_total_bytes": int(cache_total),
        "aggregate_bytes": int(aggregate),
        "per_cell_stack_total_bytes": int(per_cell_total),
        "per_cell_stack_added_bytes": int(added_per_cell_stack),
        "baseline_counted": "once_via_shared_bytes",
    }
    attempts = list(bundle.attempts)
    attempts.append(
        {
            "active_gpu_mapping": True,
            "stage": str(stage),
            "phase": mapping["phase"],
            "gpu_streaming_tile": int(tile),
            "units_in_flight": 1,
            "workers_per_unit": int(workers),
            "prefetch_units": int(prefetch),
            "aggregate_bytes": int(aggregate),
            "raw_available_bytes": int(bound_raw),
            "factor_num": int(bound_fnum),
            "factor_den": int(bound_fden),
            "budget_bytes": int(bound_budget),
        }
    )
    return replace(
        bundle,
        cpu_streaming={},
        cpu_streaming_tile=0,
        gpu_tile_size=int(tile),
        gpu_streaming_tile=int(tile),
        gpu_streaming_host_peak_bytes=int(per_cell_total),
        gpu_used=True,
        gpu_pre_execution_fallback=None,
        cell_specs=cell_specs,
        cell_tasks=cell_tasks,
        plan=plan,
        chosen_mode="stream",
        cells_in_flight=1,
        cache_workers_per_cell=int(workers),
        per_cell_footprint=int(per_cell_footprint),
        cpu_fallback_info={
            "applicable": True,
            "feasible": bool(bundle.cpu_fallback_info.get("feasible", True)),
            "mode": str(bundle.cpu_fallback_info.get("mode", "stream")),
            "max_bound_bytes": bundle.cpu_fallback_info.get("max_bound_bytes"),
            "ram_budget_bytes": int(bound_budget),
            "tile": bundle.cpu_fallback_info.get("tile"),
            "cells_in_flight": bundle.cpu_fallback_info.get("cells_in_flight"),
            "cache_workers_per_cell": bundle.cpu_fallback_info.get(
                "cache_workers_per_cell"
            ),
            "reason": "R55-E2 active GPU plan (exact whole-batch CPU fallback "
            "remains available)",
            "used": False,
            "raw_available_bytes": int(bound_raw),
            "plan_budget_bytes": int(bound_budget),
        },
        envelope=envelope,
        attempts=attempts,
        # R55-E8: the coherent gauge/cache workers selected under the same
        # B_plan/margin are the values that execute.
        gauge_workers=int(decision.gauge_workers or bundle.gauge_workers),
        cache_build_workers=int(
            decision.cache_workers or bundle.cache_build_workers
        ),
    )


def _r55_active_gpu_admit(
    *,
    initial_bundle,
    decision,
    available_fn=available_memory_bytes,
    vram_free_fn=None,
):
    """Fresh 90 % + fresh VRAM admission of the ACTIVE GPU plan.

    Shrinks only through the fixed global order. Exhaustion is a qualified
    whole-batch CPU fallback (raised as :class:`_R55GpuFallback`), never a mixed
    partial result and never a silent CPU coercion inside the GPU plan.
    """
    candidate = _r55_gpu_candidate_lookup(decision, decision.plan_set.chosen)
    mapping = dict(decision.mapping)
    bundle = _r55_active_gpu_apply(
        initial_bundle, mapping=mapping, decision=decision, stage="preflight"
    )
    trail = []
    history = [bundle.envelope]
    replans = 0
    observations = []
    while True:
        raw = int(available_fn())
        if raw <= 0:
            raise rcontract.InvalidObservation(
                "R55-E2 active GPU admission requires a strictly positive raw "
                f"observation, got {raw}"
            )
        admit_obs = rcontract.Observation.admit(raw)
        observations.append(admit_obs)
        launch_obs = rcontract.Observation.launch(int(available_fn()))
        observations.append(launch_obs)
        free_vram = int(
            zgr.read_vram_free_bytes(0) if vram_free_fn is None else vram_free_fn()
        )
        gpu_decision = zgr.fresh_vram_revalidation(
            policy=decision.policy,
            current_percent=int(decision.qualification["percent"]),
            free_bytes=free_vram,
            device_peak_bytes=int(
                decision.device_peak_bytes_by_tile[
                    int(mapping["gpu_streaming_tile"])
                ]
            ),
        )
        if (
            bundle.envelope.aggregate_bytes <= admit_obs.budget_bytes
            and bundle.envelope.aggregate_bytes <= launch_obs.budget_bytes
            and gpu_decision.action != "fallback_cpu"
        ):
            outcome = rcontract.AdmissionOutcome(
                plan=bundle.envelope,
                observations=tuple(observations),
                history=tuple(history),
                replans=int(replans),
                tolerance=rcontract.requires_tolerance(
                    bundle.envelope.aggregate_bytes,
                    launch_obs.raw_available_bytes,
                ),
                final_observation=launch_obs,
            )
            return bundle, candidate, mapping, outcome, trail
        if replans >= R41_ADMISSION_MAX_REPLANS:
            raise _R55GpuFallback(
                gpu_decision.reason_code
                if gpu_decision.action == "fallback_cpu"
                else renv.REASON_REFUSE_HOST_RAM,
                "ACTIVE GPU admission cannot fit the fresh 90 % host ceiling "
                f"and/or the fresh VRAM budget after {replans} shrink(s)",
                telemetry={
                    "trail": [dict(item) for item in trail],
                    "observations": [
                        obs.to_dict() if hasattr(obs, "to_dict") else str(obs)
                        for obs in observations
                    ],
                },
            )
        plan_obs = rcontract.Observation.plan(int(launch_obs.raw_available_bytes))
        observations.append(plan_obs)
        context = replace(
            decision.gpu_context,
            raw_available_bytes=int(plan_obs.raw_available_bytes),
            factor_num=int(rcontract.PLAN_FACTOR_NUM),
            factor_den=int(rcontract.PLAN_FACTOR_DEN),
            budget_bytes=int(plan_obs.budget_bytes),
        )
        if gpu_decision.action == "fallback_cpu":
            # R55-E9: a fresh-VRAM miss shrinks the GPU STREAMING TILE along the
            # frozen declared ladder (largest jointly feasible lower rung)
            # instead of discarding the qualified GPU plan. Host pressure still
            # follows the fixed global shrink order below.
            nxt = _r55_next_gpu_vram_shrink(
                decision,
                candidate,
                free_bytes=int(free_vram),
                percent=int(decision.qualification["percent"]),
            )
            name = "tile_px" if nxt is not None else None
        else:
            name, nxt = _r55_next_gpu_shrink(decision, candidate, context)
        if name is None or nxt is None:
            raise _R55GpuFallback(
                gpu_decision.reason_code
                if gpu_decision.action == "fallback_cpu"
                else renv.REASON_REFUSE_HOST_RAM,
                "ACTIVE GPU admission miss and no legal GPU shrink remains; "
                "falling back to one exact whole-batch CPU run",
                telemetry={"trail": [dict(item) for item in trail]},
            )
        trail.append(
            {
                "step": name,
                "from": zsched.active_candidate_tuple(candidate, allow_gpu=True),
                "to": zsched.active_candidate_tuple(nxt, allow_gpu=True),
                "accepted": True,
                "reason_code": nxt.reason_code,
                "detail": nxt.infeasible_reason,
                "at": "admission_pressure",
                "stage": "preflight",
                "admit_raw_bytes": int(launch_obs.raw_available_bytes),
                "b_admit_bytes": int(launch_obs.budget_bytes),
                "b_plan_bytes": int(plan_obs.budget_bytes),
                "aggregate_bytes_before": int(bundle.envelope.aggregate_bytes),
                "vram_free_bytes": int(free_vram),
            }
        )
        candidate = nxt
        mapping = zsched.active_gpu_mapping_for_candidate(
            candidate,
            tile_ladder=tuple(int(t) for t in decision.gpu_context.tile_ladder),
            workers_ladder=tuple(int(w) for w in zsched.WORKERS_LADDER),
            prefetch_cap=int(_R55E2_PREFETCH_LADDER[-1]),
        )
        bundle = _r55_active_gpu_apply(
            initial_bundle,
            mapping=mapping,
            decision=decision,
            stage="preflight",
            plan_raw_available_bytes=int(plan_obs.raw_available_bytes),
            plan_factor_num=int(rcontract.PLAN_FACTOR_NUM),
            plan_factor_den=int(rcontract.PLAN_FACTOR_DEN),
            plan_budget_bytes=int(plan_obs.budget_bytes),
        )
        replans += 1
        history.append(bundle.envelope)

def _r55_active_apply(
    bundle,
    *,
    mapping,
    facts,
    plan_raw_available_bytes=None,
    plan_factor_num=None,
    plan_factor_den=None,
    plan_budget_bytes=None,
    stage="preflight",
):
    """Apply an EXACT mapping to the frozen bundle's existing execution knobs.

    Only the values that actually execute are changed: the cell task mode, the
    streaming tile, the cell-pool concurrency and the per-cell cache workers.
    Layout, memberships, frames, halo and science identity are untouched.

    ``plan_raw_available_bytes`` / ``plan_factor_*`` / ``plan_budget_bytes``
    rebind the returned PlanEnvelope to the observation that actually governs a
    pressure correction (the fresh 4/5 PLAN observation). When omitted the base
    envelope's own observation is preserved (initial plan or an unchanged
    preflight decision).
    """
    mode = mapping["cell_task_mode"]
    tile = mapping["cpu_streaming_tile"]
    units = int(mapping["units_in_flight"])
    workers = int(mapping["workers_per_unit"])
    if int(mapping["prefetch_units"]) != 0:
        raise zsched.ActivePlanRejected(
            renv.REASON_SHRINK_PREFETCH,
            "active execution refuses prefetch_units != 0",
            mapping=mapping,
        )
    if mode not in ("inmem", "stream"):
        raise zsched.ActivePlanRejected(
            renv.REASON_UNKNOWN_BACKEND,
            f"unknown cell task mode {mode!r}",
            mapping=mapping,
        )
    if mode == "inmem" and tile is not None:
        raise zsched.ActivePlanRejected(
            renv.REASON_SHRINK_TILE,
            f"inmem mapping must not carry a tile, got {tile!r}",
            mapping=mapping,
        )
    if mode == "stream" and tile not in tuple(CPU_STREAMING_TILE_LADDER):
        raise zsched.ActivePlanRejected(
            renv.REASON_SHRINK_TILE,
            f"stream tile {tile!r} is outside the canonical product ladder",
            mapping=mapping,
        )
    if units < 1 or workers < 1:
        raise zsched.ActivePlanRejected(
            renv.REASON_REFUSE_HOST_RAM,
            "active execution requires units_in_flight >= 1 and "
            "workers_per_unit >= 1",
            mapping=mapping,
        )

    cell_specs = []
    cell_tasks = []
    for (cell, patch, mem, cache_dir, _old_inmem, _old_stream, _old_gpu) in bundle.cell_specs:
        if not mem.patch_ids:
            cell_specs.append((cell, patch, mem, cache_dir, None, 0, None))
            cell_tasks.append(_cell_task(
                cell, patch, mem, cache_dir, None, 0, getattr(bundle, "storage_profile", _STACK_STORAGE_PROFILE)))
            continue
        n = len(mem.patch_ids)
        area = int(patch.patch.width) * int(patch.patch.height)
        inmem_b = int(zal.FITTED_MEMORY_MODEL.predict_bound_bytes(n, area))
        stream_b = (
            int(_profile_stream_bytes(
                n, patch.patch_shape_hw, int(tile), getattr(bundle, "storage_profile", _STACK_STORAGE_PROFILE)))
            if tile is not None else 0
        )
        cell_specs.append(
            (cell, patch, mem, cache_dir, inmem_b, stream_b, None)
        )
        if mode == "inmem":
            cell_tasks.append(_cell_task(
                cell, patch, mem, cache_dir, "inmem", inmem_b,
                getattr(bundle, "storage_profile", _STACK_STORAGE_PROFILE)))
        else:
            cell_tasks.append(_cell_task(
                cell, patch, mem, cache_dir, "stream", stream_b,
                getattr(bundle, "storage_profile", _STACK_STORAGE_PROFILE)))

    per_cell_footprint = max(
        (int(t[5]) for t in cell_tasks if t[4] is not None), default=0
    )
    base_env = bundle.envelope
    bound_raw = (
        int(base_env.raw_available_bytes)
        if plan_raw_available_bytes is None
        else int(plan_raw_available_bytes)
    )
    bound_fnum = (
        int(base_env.factor_num)
        if plan_factor_num is None else int(plan_factor_num)
    )
    bound_fden = (
        int(base_env.factor_den)
        if plan_factor_den is None else int(plan_factor_den)
    )
    bound_budget = (
        int(base_env.budget_bytes)
        if plan_budget_bytes is None else int(plan_budget_bytes)
    )
    if bound_fnum < 1 or bound_fden < 1:
        raise rcontract.InvalidEnvelope(
            f"invalid plan factor {bound_fnum}/{bound_fden}"
        )
    if bound_budget != (bound_fnum * bound_raw) // bound_fden:
        raise rcontract.InvalidEnvelope(
            "rebound plan budget does not equal (factor*raw)//den: "
            f"{bound_budget} != ({bound_fnum}*{bound_raw})//{bound_fden}"
        )
    cache_per_cell = int(rcontract.cache_per_cell_bytes(
        rcontract.PROCESS_BASELINE_BYTES,
        workers,
        int(bundle.cache_worker_footprint_bytes),
    ))
    stack_total = int(rcontract.stack_phase_total_bytes(
        int(bundle.shared_bytes), units, int(per_cell_footprint)
    ))
    cache_total = int(rcontract.cache_phase_total_bytes(
        int(bundle.shared_bytes), units, cache_per_cell
    ))
    aggregate = int(rcontract.aggregate_phase_bytes(stack_total, cache_total))
    reason = (
        "R55-D2 active CPU mapping: "
        f"phase={mapping['phase']} tile={tile!r} units_in_flight={units} "
        f"workers_per_unit={workers} prefetch_units=0"
    )
    envelope = rcontract.PlanEnvelope(
        phase=rcontract.PHASE_PLAN,
        raw_available_bytes=int(bound_raw),
        factor_num=int(bound_fnum),
        factor_den=int(bound_fden),
        budget_bytes=int(bound_budget),
        layout=(int(base_env.layout[0]), int(base_env.layout[1])),
        refinement_rank=int(base_env.refinement_rank),
        cpu_tile=(0 if tile is None else int(tile)),
        gpu_tile=None,
        backend=rcontract.BACKEND_CPU,
        cells_in_flight=int(units),
        cache_workers_per_cell=int(workers),
        shared_bytes=int(bundle.shared_bytes),
        stack_total_bytes=int(stack_total),
        cache_total_bytes=int(cache_total),
        aggregate_bytes=int(aggregate),
        pinned=bool(base_env.pinned),
        reason=reason,
    )
    rcontract.validate_envelope(envelope)

    stream_worst_by_tile = {
        int(k): int(v) for k, v in facts["stream_worst_by_tile"].items()
    }
    cpu_streaming = _cpu_streaming_block(
        tile,
        [],
        reason,
        stream_worst_by_tile,
        int(facts.get("inmem_worst_actual", facts["inmem_worst"])),
        int(bound_budget),
    )
    if tile is None:
        cpu_streaming["predicted_peak_bytes"] = int(
            facts.get("inmem_worst_actual", facts["inmem_worst"])
        )
        cpu_streaming["stream_bound_bytes"] = 0
    plan = {
        "mode": mode,
        "cells_in_flight": int(units),
        "cache_workers_per_cell": int(workers),
        "gpu_serialized": False,
        "chosen_reason": reason,
        "candidates": [
            {
                "mode": mode,
                "cells_in_flight": int(units),
                "max_bound_bytes": int(per_cell_footprint),
                "feasible": True,
                "makespan_rel": None,
                "infeasible_reason": None,
            }
        ],
        "stack_total_bytes": int(stack_total),
        "cache_total_bytes": int(cache_total),
        "aggregate_bytes": int(aggregate),
    }
    cpu_fallback_info = {
        "applicable": False,
        "feasible": True,
        "mode": mode,
        "max_bound_bytes": int(per_cell_footprint),
        "ram_budget_bytes": int(bound_budget),
        "tile": (None if tile is None else int(tile)),
        "cells_in_flight": int(units),
        "cache_workers_per_cell": int(workers),
        "reason": "R55-D2 active CPU plan",
        "used": False,
        "raw_available_bytes": int(bound_raw),
        "plan_budget_bytes": int(bound_budget),
    }
    attempts = list(bundle.attempts)
    attempts.append(
        {
            "active_mapping": True,
            "stage": str(stage),
            "phase": mapping["phase"],
            "cpu_streaming_tile": (None if tile is None else int(tile)),
            "units_in_flight": int(units),
            "workers_per_unit": int(workers),
            "prefetch_units": 0,
            "aggregate_bytes": int(aggregate),
            "raw_available_bytes": int(bound_raw),
            "factor_num": int(bound_fnum),
            "factor_den": int(bound_fden),
            "budget_bytes": int(bound_budget),
        }
    )
    return replace(
        bundle,
        cpu_streaming=dict(cpu_streaming),
        cpu_streaming_tile=(0 if tile is None else int(tile)),
        gpu_tile_size=None,
        gpu_streaming_tile=None,
        gpu_streaming_host_peak_bytes=None,
        gpu_used=False,
        # Preserve the legacy pre-execution fallback provenance. If the product
        # had requested GPU but the bundle already degraded to CPU, the caller's
        # post-gauge block still needs this reason to reconcile science_config
        # and gpu_context truthfully. A pure CPU bundle carries None here.
        gpu_pre_execution_fallback=bundle.gpu_pre_execution_fallback,
        cell_specs=cell_specs,
        cell_tasks=cell_tasks,
        plan=plan,
        chosen_mode=mode,
        cells_in_flight=int(units),
        cache_workers_per_cell=int(workers),
        per_cell_footprint=int(per_cell_footprint),
        cpu_fallback_info=cpu_fallback_info,
        cpu_fallback_plan=None,
        cpu_fallback_tasks=None,
        envelope=envelope,
        attempts=attempts,
        # R55-E8: the coherent gauge/cache workers selected under the same
        # B_plan/margin are the values that execute (never the pre-adaptation
        # bundle values).
        gauge_workers=int(facts["gauge_workers"]),
        cache_build_workers=int(facts["cache_workers"]),
    )


def _r55_active_shrink_context(decision, candidate):
    return (
        decision.inmem_context
        if candidate.tile_px is None
        else decision.stream_context
    )


def _r55_next_materialized_shrink(*, decision, candidate, plan_obs=None):
    """Advance one legal tuple shrink and MATERIALIZE it canonically.

    ``next_shrink`` only changes tuple fields and preserves stale projections.
    This helper re-enumerates the owning context (rebound to the fresh 4/5 PLAN
    observation when pressure created one), skips scheduler-infeasible tuples
    (still monotone), and returns the first FEASIBLE materialized candidate.

    Returns ``(step_name, candidate)`` or ``(None, None)`` when no legal
    feasible shrink remains.
    """
    context = _r55_active_shrink_context(decision, candidate)
    if plan_obs is not None:
        context = replace(
            context,
            raw_available_bytes=int(plan_obs.raw_available_bytes),
            factor_num=rcontract.PLAN_FACTOR_NUM,
            factor_den=rcontract.PLAN_FACTOR_DEN,
            budget_bytes=int(plan_obs.budget_bytes),
        )
    examined = []
    current = candidate
    while True:
        name, nxt = zsched.next_shrink(current, context)
        if name is None or nxt is None:
            return None, None, examined
        materialized = zsched.materialize_active_candidate(context, nxt)
        zsched.assert_shrink_only(current, materialized)
        examined.append(
            {
                "step": name,
                "from": zsched.active_candidate_tuple(current),
                "to": zsched.active_candidate_tuple(materialized),
                "accepted": bool(materialized.feasible),
                "reason_code": materialized.reason_code,
                "detail": materialized.infeasible_reason,
                "projections": {
                    "host_peak_bytes": int(materialized.host_peak_bytes),
                    "disk_bytes": int(materialized.disk_bytes),
                    "device_peak_bytes": int(materialized.device_peak_bytes),
                },
            }
        )
        current = materialized
        if materialized.feasible:
            return name, materialized, examined


def _r55_active_admit(
    *, initial_bundle, decision
):
    """Fit + admit the active plan BEFORE gauge/stack (fresh 90 % observations).

    Every correction is a scheduler-produced :func:`scheduler.next_shrink` step
    on the SAME frozen identity; growth is refused. Returns
    ``(bundle, mapping, admission_outcome, trail)``.
    """
    candidate = decision.plan_set.chosen
    mapping = decision.mapping
    # Defensive exactness: the decision object cannot be tampered into an
    # execution that differs from the accepted scheduler winner.
    if decision.plan_set.mode != zsched.MODE_ACTIVE:
        raise zsched.ActivePlanRejected(
            renv.REASON_UNKNOWN_BACKEND,
            "active admission requires an active PlanSet",
            candidate=candidate,
            mapping=mapping,
        )
    _expected_mapping = zsched.active_mapping_for_candidate(
        candidate,
        tile_ladder=tuple(int(t) for t in CPU_STREAMING_TILE_LADDER),
        workers_ladder=tuple(int(w) for w in zsched.WORKERS_LADDER),
    )
    if dict(mapping) != _expected_mapping:
        raise zsched.ActivePlanRejected(
            renv.REASON_INTERNAL_ERROR,
            "active mapping does not exactly match the scheduler winner "
            f"(mapping={mapping!r}, expected={_expected_mapping!r})",
            candidate=candidate,
            mapping=mapping,
        )
    # Materialize the accepted initial winner from canonical enumeration so the
    # final manifest never inherits stale projections, even with no shrink.
    candidate = zsched.materialize_active_candidate(
        _r55_active_shrink_context(decision, candidate), candidate
    )
    mapping = zsched.active_mapping_for_candidate(
        candidate,
        tile_ladder=tuple(int(t) for t in CPU_STREAMING_TILE_LADDER),
        workers_ladder=tuple(int(w) for w in zsched.WORKERS_LADDER),
    )
    bundle = _r55_active_apply(
        initial_bundle,
        mapping=mapping,
        facts=decision.facts,
        stage="preflight",
    )
    trail = []

    # 1) The mapped product envelope must still fit the INITIAL 80 % B_plan.
    #    (The scheduler's host model does not include the product's cache
    #    phase, so an honest exact mapping is re-checked here.)
    while bundle.envelope.aggregate_bytes > bundle.envelope.budget_bytes:
        name, nxt, examined = _r55_next_materialized_shrink(
            decision=decision, candidate=candidate, plan_obs=None
        )
        for entry in examined:
            trail.append(
                {
                    **entry,
                    "at": "plan_fit",
                    "stage": "preflight",
                    "aggregate_bytes_before": int(
                        bundle.envelope.aggregate_bytes
                    ),
                    "b_plan_bytes": int(bundle.envelope.budget_bytes),
                }
            )
        if name is None or nxt is None:
            raise rcontract.AdmissionExhausted(
                "R55-D2 active plan cannot fit its own B_plan and has no legal "
                "feasible shrink step left (aggregate "
                f"{bundle.envelope.aggregate_bytes} > "
                f"B_plan {bundle.envelope.budget_bytes})"
            )
        candidate = nxt
        mapping = zsched.active_mapping_for_candidate(
            candidate,
            tile_ladder=tuple(int(t) for t in CPU_STREAMING_TILE_LADDER),
            workers_ladder=tuple(int(w) for w in zsched.WORKERS_LADDER),
        )
        bundle = _r55_active_apply(
            initial_bundle,
            mapping=mapping,
            facts=decision.facts,
            stage="preflight",
        )

    def _measure():
        raw = int(available_memory_bytes())
        if raw <= 0:
            raise rcontract.InvalidObservation(
                "R55-D2 active admission requires a strictly positive raw "
                f"observation, got {raw}"
            )
        return raw

    observations = []
    history = [bundle.envelope]
    replans = 0
    while True:
        admit_obs = rcontract.Observation.admit(_measure())
        observations.append(admit_obs)
        if bundle.envelope.aggregate_bytes <= admit_obs.budget_bytes:
            launch_obs = rcontract.Observation.launch(_measure())
            observations.append(launch_obs)
            if bundle.envelope.aggregate_bytes <= launch_obs.budget_bytes:
                outcome = rcontract.AdmissionOutcome(
                    plan=bundle.envelope,
                    observations=tuple(observations),
                    history=tuple(history),
                    replans=int(replans),
                    tolerance=rcontract.requires_tolerance(
                        bundle.envelope.aggregate_bytes,
                        launch_obs.raw_available_bytes,
                    ),
                    final_observation=launch_obs,
                )
                return bundle, candidate, mapping, outcome, trail
            miss_obs = launch_obs
        else:
            miss_obs = admit_obs

        if replans >= R41_ADMISSION_MAX_REPLANS:
            raise rcontract.AdmissionExhausted(
                "R55-D2 active plan still exceeds the fresh admission ceiling "
                f"after {R41_ADMISSION_MAX_REPLANS} scheduler shrink(s): "
                f"aggregate {bundle.envelope.aggregate_bytes / 2**20:.1f} MiB > "
                f"B_admit {miss_obs.budget_bytes / 2**20:.1f} MiB",
                attempts=tuple(observations),
            )
        plan_obs = rcontract.Observation.plan(miss_obs.raw_available_bytes)
        observations.append(plan_obs)
        # Shrink under the SAME observation's B_plan, exactly like a rebuild,
        # but with scheduler-produced shrink-only choices. The resulting
        # PlanEnvelope is REBOUND to that fresh 4/5 observation so the final
        # manifest never reports a stale PLAN observation.
        while True:
            name, nxt, examined = _r55_next_materialized_shrink(
                decision=decision, candidate=candidate, plan_obs=plan_obs
            )
            for entry in examined:
                trail.append(
                    {
                        **entry,
                        "at": "admission_pressure",
                        "stage": "preflight",
                        "admit_raw_bytes": int(
                            miss_obs.raw_available_bytes
                        ),
                        "b_admit_bytes": int(miss_obs.budget_bytes),
                        "b_plan_bytes": int(plan_obs.budget_bytes),
                        "aggregate_bytes_before": int(
                            bundle.envelope.aggregate_bytes
                        ),
                    }
                )
            if name is None or nxt is None:
                raise rcontract.AdmissionExhausted(
                    "R55-D2 active admission miss and no legal FEASIBLE "
                    "shrink step remains (aggregate "
                    f"{bundle.envelope.aggregate_bytes} > "
                    f"B_admit {miss_obs.budget_bytes})",
                    attempts=tuple(observations),
                )
            candidate = nxt
            mapping = zsched.active_mapping_for_candidate(
                candidate,
                tile_ladder=tuple(int(t) for t in CPU_STREAMING_TILE_LADDER),
                workers_ladder=tuple(int(w) for w in zsched.WORKERS_LADDER),
            )
            bundle = _r55_active_apply(
                initial_bundle,
                mapping=mapping,
                facts=decision.facts,
                plan_raw_available_bytes=plan_obs.raw_available_bytes,
                plan_factor_num=rcontract.PLAN_FACTOR_NUM,
                plan_factor_den=rcontract.PLAN_FACTOR_DEN,
                plan_budget_bytes=plan_obs.budget_bytes,
                stage="preflight",
            )
            if bundle.envelope.aggregate_bytes <= plan_obs.budget_bytes:
                break
        replans += 1
        history.append(bundle.envelope)


def _r55_active_revalidate_launch(
    *, bundle, candidate, mapping, decision
):
    """Fresh 90 % launch/admission revalidation immediately before the stack.

    Starts from the ALREADY pre-gauge-admitted final bundle/candidate/mapping
    (never the original winner), so a post-gauge correction can only shrink.
    The gauge may already have run; exhaustion here is fatal with zero stack
    calls and no committed science/manifest output.

    Returns ``(bundle, candidate, mapping, admission_outcome, launch_trail)``.
    """
    trail = []

    def _measure():
        raw = int(available_memory_bytes())
        if raw <= 0:
            raise rcontract.InvalidObservation(
                "R55-D2 active launch revalidation requires a strictly positive "
                f"raw observation, got {raw}"
            )
        return raw

    observations = []
    history = [bundle.envelope]
    replans = 0
    while True:
        admit_obs = rcontract.Observation.admit(_measure())
        observations.append(admit_obs)
        if bundle.envelope.aggregate_bytes <= admit_obs.budget_bytes:
            launch_obs = rcontract.Observation.launch(_measure())
            observations.append(launch_obs)
            if bundle.envelope.aggregate_bytes <= launch_obs.budget_bytes:
                outcome = rcontract.AdmissionOutcome(
                    plan=bundle.envelope,
                    observations=tuple(observations),
                    history=tuple(history),
                    replans=int(replans),
                    tolerance=rcontract.requires_tolerance(
                        bundle.envelope.aggregate_bytes,
                        launch_obs.raw_available_bytes,
                    ),
                    final_observation=launch_obs,
                )
                return bundle, candidate, mapping, outcome, trail
            miss_obs = launch_obs
        else:
            miss_obs = admit_obs

        if replans >= R41_ADMISSION_MAX_REPLANS:
            raise rcontract.AdmissionExhausted(
                "R55-D2 active launch revalidation still exceeds the fresh "
                f"admission ceiling after {R41_ADMISSION_MAX_REPLANS} "
                "scheduler shrink(s): aggregate "
                f"{bundle.envelope.aggregate_bytes / 2**20:.1f} MiB > B_admit "
                f"{miss_obs.budget_bytes / 2**20:.1f} MiB",
                attempts=tuple(observations),
            )
        plan_obs = rcontract.Observation.plan(miss_obs.raw_available_bytes)
        observations.append(plan_obs)
        while True:
            name, nxt, examined = _r55_next_materialized_shrink(
                decision=decision, candidate=candidate, plan_obs=plan_obs
            )
            for entry in examined:
                trail.append(
                    {
                        **entry,
                        "at": "admission_pressure",
                        "stage": "launch",
                        "admit_raw_bytes": int(
                            miss_obs.raw_available_bytes
                        ),
                        "b_admit_bytes": int(miss_obs.budget_bytes),
                        "b_plan_bytes": int(plan_obs.budget_bytes),
                        "aggregate_bytes_before": int(
                            bundle.envelope.aggregate_bytes
                        ),
                    }
                )
            if name is None or nxt is None:
                raise rcontract.AdmissionExhausted(
                    "R55-D2 active launch revalidation miss and no legal "
                    "FEASIBLE shrink step remains (aggregate "
                    f"{bundle.envelope.aggregate_bytes} > B_admit "
                    f"{miss_obs.budget_bytes})",
                    attempts=tuple(observations),
                )
            candidate = nxt
            mapping = zsched.active_mapping_for_candidate(
                candidate,
                tile_ladder=tuple(int(t) for t in CPU_STREAMING_TILE_LADDER),
                workers_ladder=tuple(int(w) for w in zsched.WORKERS_LADDER),
            )
            bundle = _r55_active_apply(
                bundle,
                mapping=mapping,
                facts=decision.facts,
                plan_raw_available_bytes=plan_obs.raw_available_bytes,
                plan_factor_num=rcontract.PLAN_FACTOR_NUM,
                plan_factor_den=rcontract.PLAN_FACTOR_DEN,
                plan_budget_bytes=plan_obs.budget_bytes,
                stage="launch",
            )
            if bundle.envelope.aggregate_bytes <= plan_obs.budget_bytes:
                break
        replans += 1
        history.append(bundle.envelope)


def _r55_admission_dict(outcome) -> dict:
    """JSON-safe admission record for the active manifest."""
    return {
        "observations": [o.to_dict() for o in outcome.observations],
        "launch_observation": outcome.final_observation.to_dict(),
        "replans": int(outcome.replans),
        "tolerance": bool(outcome.tolerance),
        "history": [env.to_dict() for env in outcome.history],
    }


def _r55_active_plan_observation(bundle) -> dict:
    env = bundle.envelope
    return {
        "raw_available_bytes": int(env.raw_available_bytes),
        "factor_num": int(env.factor_num),
        "factor_den": int(env.factor_den),
        "budget_bytes": int(env.budget_bytes),
    }


# ---------------------------------------------------------------------------
# ZM-ZEGRID-R55-E11 — RUN-time, shrink-only GPU tile adaptation.
#
# R55-E9/E10 choose an admissible GPU tile ONCE and revalidate it unchanged
# before every cell. When fresh VRAM no longer fits that exact workspace the old
# path discarded the whole GPU batch. R55-E11 walks the SAME declared/pruned
# ladder strictly downward at the inter-cell checkpoint and hands the smaller
# tile to the runner. No growth, no new planner, no interpolated bound.
# ---------------------------------------------------------------------------

_R55E11_TILE_RUNTIME_SCHEMA = "zemosaic-r55e11-gpu-tile-runtime-v1"
_R55E11_ENGINE_PROVENANCE_SCHEMA = "zemosaic-r55e11-engine-provenance-v1"


def _r55_engine_provenance() -> dict:
    """Static engine provenance for the RUN record (no git subprocess).

    The installed-binary SHA is not known here; that gap is made VISIBLE rather
    than guessed. Package version + module path + the exact policy/schema
    constants are enough to tell which engine and which policy produced a run.
    """
    try:
        from importlib import metadata as _metadata

        package_version = str(_metadata.version("ZeMosaic"))
    except Exception:
        try:
            from . import __version__ as _pkg_version

            package_version = str(_pkg_version)
        except Exception:
            package_version = None
    return {
        "schema": _R55E11_ENGINE_PROVENANCE_SCHEMA,
        "package_version": package_version,
        "engine_module": str(Path(__file__).resolve()),
        "policy_schema": str(zgr.POLICY_SCHEMA),
        "readiness_schema": str(zgr.SCHEMA),
        "run_control_schema": str(zrt.RUN_CONTROL_SCHEMA),
        "tile_runtime_schema": _R55E11_TILE_RUNTIME_SCHEMA,
        "cache_bytes_contract": _R55_CACHE_BYTES_CONTRACT,
        "active_vram_percent": int(zgr.INITIAL_TARGET_PERCENT),
        "legacy_probe_percent": 50,
        "selection_headroom_percent": int(
            _R55E10_VRAM_SELECTION_HEADROOM_PERCENT
        ),
        "binary_sha_known": False,
        "binary_sha_note": (
            "The executed frozen binary SHA is not recorded by this build; "
            "package_version + module path are the available provenance."
        ),
    }


def _r55_gpu_added_host_bound(decision, tile: int) -> int:
    """The authoritative ADDED per-cell stack bound for one declared tile.

    ``decision.gpu_host_bound_by_tile`` is frozen at planning time as
    ``estimate_streaming_bytes(...) - PROCESS_BASELINE_BYTES``; the scheduler's
    host peak is ``shared_bytes + that added bound`` (baseline counted ONCE via
    ``shared_bytes``). This accessor is the single source of truth for the
    apply/plan envelope and the RUN-time host aggregate. Missing, duplicate or
    malformed entries fail closed; no incompatible total is ever recomputed.
    """
    raw = getattr(decision, "gpu_host_bound_by_tile", None)
    if isinstance(raw, dict):
        items = tuple(raw.items())
    elif isinstance(raw, (tuple, list)):
        items = tuple(raw)
    else:
        raise ValueError(
            "decision.gpu_host_bound_by_tile is missing or not a mapping/sequence"
        )
    bounds: dict[int, int] = {}
    for index, item in enumerate(items):
        if (
            isinstance(item, (str, bytes))
            or not isinstance(item, (tuple, list))
            or len(item) != 2
        ):
            raise ValueError(
                "gpu_host_bound_by_tile entries must be (tile_px, bytes) pairs "
                f"(bad entry at index {index}: {item!r})"
            )
        tile_px, bound = item
        if isinstance(tile_px, bool) or not isinstance(tile_px, int):
            raise ValueError(
                f"gpu_host_bound_by_tile tile must be a plain int, got {tile_px!r}"
            )
        if isinstance(bound, bool) or not isinstance(bound, int):
            raise ValueError(
                f"gpu_host_bound_by_tile bytes must be a plain int, got {bound!r}"
            )
        if int(tile_px) <= 0 or int(bound) <= 0:
            raise ValueError(
                "gpu_host_bound_by_tile tile/bytes must be strictly positive, "
                f"got ({tile_px}, {bound})"
            )
        if int(tile_px) in bounds:
            raise ValueError(
                f"gpu_host_bound_by_tile has duplicate tile {int(tile_px)}"
            )
        bounds[int(tile_px)] = int(bound)
    wanted = int(tile)
    if wanted not in bounds:
        raise ValueError(
            f"gpu_host_bound_by_tile has no authoritative bound for tile {wanted}"
        )
    return int(bounds[wanted])


def _r55_gpu_host_aggregate_for_tile(decision, bundle, tile: int) -> int:
    """Exact aggregate host bytes for ONE declared tile (same math as apply)."""
    # Authoritative ADDED stack bound: baseline counted ONCE via shared_bytes.
    # (The TOTAL estimate remains in the execution task/manifest only.)
    added_per_cell_stack = _r55_gpu_added_host_bound(decision, tile)
    workers = int(bundle.cache_workers_per_cell)
    stack_total = int(
        rcontract.stack_phase_total_bytes(
            int(bundle.shared_bytes), 1, int(added_per_cell_stack)
        )
    )
    cache_per_cell = int(
        rcontract.cache_per_cell_bytes(
            rcontract.PROCESS_BASELINE_BYTES,
            workers,
            int(bundle.cache_worker_footprint_bytes),
        )
    )
    cache_total = int(
        rcontract.cache_phase_total_bytes(
            int(bundle.shared_bytes), 1, cache_per_cell
        )
    )
    return int(rcontract.aggregate_phase_bytes(stack_total, cache_total))


def _r55_runtime_tile_check(
    decision,
    *,
    tile: int,
    free_bytes: int,
    host_budget_bytes: int,
    bundle,
) -> dict:
    """Evaluate ONE tile against the CURRENT facts (device AND exact host)."""
    tile = int(tile)
    peer = decision.device_peak_bytes_by_tile
    peak = int(peer[tile])
    vram = zgr.fresh_vram_revalidation(
        policy=decision.policy,
        current_percent=int(decision.qualification["percent"]),
        free_bytes=int(free_bytes),
        device_peak_bytes=peak,
    )
    aggregate = int(_r55_gpu_host_aggregate_for_tile(decision, bundle, tile))
    device_ok = vram.action != "fallback_cpu"
    host_ok = aggregate <= int(host_budget_bytes)
    return {
        "tile_px": int(tile),
        "device_peak_bytes": peak,
        "vram_budget_bytes": int(vram.budget_bytes or 0),
        "vram_free_bytes": int(free_bytes),
        "vram_action": str(vram.action),
        "vram_reason_code": str(vram.reason_code),
        "host_aggregate_bytes": aggregate,
        "host_budget_bytes": int(host_budget_bytes),
        "device_ok": bool(device_ok),
        "host_ok": bool(host_ok),
        "eligible": bool(device_ok and host_ok),
    }


def _r55_runtime_tile_walk(
    decision,
    *,
    current_tile: int,
    free_bytes: int,
    host_budget_bytes: int,
    bundle,
) -> tuple[list[dict], int | None]:
    """Strictly SMALLER rungs of the admitted ladder, descending.

    Finite: at most ``len(gpu_tile_ladder) - 1`` candidates. Never grows, never
    leaves the declared/pruned ladder, never consults a second planner. Every
    candidate is checked against its OWN exact device workspace and exact host
    aggregate on the CURRENT observations.
    """
    ladder = tuple(int(t) for t in decision.gpu_tile_ladder)
    current = int(current_tile)
    if current not in ladder:
        return [], None
    attempts: list[dict] = []
    for tile in ladder[ladder.index(current) + 1:]:
        attempt = _r55_runtime_tile_check(
            decision,
            tile=int(tile),
            free_bytes=int(free_bytes),
            host_budget_bytes=int(host_budget_bytes),
            bundle=bundle,
        )
        attempts.append(attempt)
        if attempt["eligible"]:
            return attempts, int(tile)
    return attempts, None


_R55E11_MISSION = "ZM-ZEGRID-R55-E11-RUNTIME-VRAM-ADAPTATION-20261024"

_R55E11_FAILURE_SCHEMA = "zemosaic-r55e11-gpu-failure-v1"
_R55E11_FAILURE_FILENAME = "zegrid_gpu_failure.json"
_R55E11_FAILURE_JSONL = "zegrid_gpu_failure.jsonl"
_R55E11_FAILURE_MAX_CHAIN = 8
_R55E11_FAILURE_MAX_TEXT = 16 * 1024
_R55E11_FAILURE_MAX_REVALIDATIONS = 32


def _r55_bounded_text(value, limit: int = _R55E11_FAILURE_MAX_TEXT) -> str:
    """Bound a diagnostic string WITHOUT changing the original exception."""
    text = str(value)
    if len(text) <= int(limit):
        return text
    return text[: int(limit)] + f"... [truncated {len(text) - int(limit)} chars]"


def _r55_exception_chain(exc) -> list[dict]:
    """Bounded type/module/message/repr + formatted traceback for exc and its
    ``__cause__``/``__context__`` chain (never raises)."""
    chain: list[dict] = []
    seen: set[int] = set()
    current = exc
    while (
        current is not None
        and id(current) not in seen
        and len(chain) < _R55E11_FAILURE_MAX_CHAIN
    ):
        seen.add(id(current))
        entry = {
            "type": type(current).__name__,
            "module": str(type(current).__module__),
            "message": _r55_bounded_text(current, 4096),
            "repr": _r55_bounded_text(repr(current), 4096),
            "traceback": "",
        }
        try:
            entry["traceback"] = _r55_bounded_text(
                "".join(
                    traceback.format_exception(
                        type(current), current, current.__traceback__
                    )
                )
            )
        except Exception as fmt_exc:  # defensive: never mask the original
            entry["traceback"] = f"<traceback_format_failed:{type(fmt_exc).__name__}>"
        chain.append(entry)
        nxt = (
            current.__cause__
            if current.__cause__ is not None
            else current.__context__
        )
        if nxt is exc:
            break
        current = nxt
    return chain


def _r55_gpu_failure_record(
    *,
    stage: str,
    category: str,
    reason_code,
    detail,
    exc=None,
    pipeline=None,
    tile_state=None,
    revalidations=None,
    unit=None,
    host_facts=None,
    device_snapshot=None,
    parent_memory=None,
    extra=None,
) -> dict:
    """One bounded, JSON-safe GPU failure record.

    Written IMMEDIATELY on the failure path (before the whole-batch CPU retry)
    so the original cause survives even when the run never reaches finalize and
    never writes a manifest.
    """
    chain = _r55_exception_chain(exc) if exc is not None else []
    revalidations = list(revalidations or [])
    tile_view = {}
    if isinstance(tile_state, dict):
        tile_view = {
            "tile_px": tile_state.get("tile"),
            "device_peak_bytes": tile_state.get("device_peak"),
            "final_cause": tile_state.get("final_cause"),
            "shrink_history": [
                dict(item)
                for item in (tile_state.get("shrink_history") or [])
            ],
            "attempts": [
                dict(item) for item in (tile_state.get("attempts") or [])
            ],
            "units": [
                {
                    "unit_index": record.get("unit_index"),
                    "tile_px": record.get("tile_px"),
                    "status": record.get("status"),
                    "action": record.get("action"),
                    "attempted": bool(record.get("attempted")),
                    "completed": bool(record.get("completed")),
                    "discarded": bool(record.get("discarded")),
                }
                for record in (tile_state.get("units") or [])
            ],
        }
    record = {
        "schema": _R55E11_FAILURE_SCHEMA,
        "mission": _R55E11_MISSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "stage": str(stage),
        "category": str(category),
        "reason_code": None if reason_code is None else str(reason_code),
        "detail": None if detail is None else _r55_bounded_text(detail, 4096),
        "exception": (
            None
            if not chain
            else {
                "type": chain[0]["type"],
                "module": chain[0]["module"],
                "message": chain[0]["message"],
                "repr": chain[0]["repr"],
                "traceback": chain[0]["traceback"],
            }
        ),
        "exception_chain": chain,
        "pipeline": None if pipeline is None else dict(pipeline),
        "unit": None if unit is None else dict(unit),
        "tile_state": tile_view,
        "host": {
            "last_checkpoint_facts": (
                None if host_facts is None else dict(host_facts)
            ),
            "parent_memory": (
                None if parent_memory is None else dict(parent_memory)
            ),
        },
        "device": (
            None if device_snapshot is None else dict(device_snapshot)
        ),
        "revalidations_total": int(len(revalidations)),
        "revalidations_tail": [
            dict(item)
            for item in revalidations[-_R55E11_FAILURE_MAX_REVALIDATIONS:]
        ],
        "extra": {} if extra is None else dict(extra),
        "note": (
            "Immediate GPU failure diagnostics. Written before the whole-batch "
            "CPU retry and independent of the final manifest."
        ),
    }
    return record


def _r55_write_gpu_failure_diagnostics(output_dir, record: dict) -> str | None:
    """Atomically persist one failure record; NEVER raises (best effort).

    Writes ``zegrid_gpu_failure.json`` (atomic replace) plus an append-only
    ``zegrid_gpu_failure.jsonl``. A write failure returns ``None`` and must not
    mask the original science/pipeline error.
    """
    try:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, ensure_ascii=False, indent=2, default=str)
        target = root / _R55E11_FAILURE_FILENAME
        tmp = root / (_R55E11_FAILURE_FILENAME + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, target)
        try:
            with (root / _R55E11_FAILURE_JSONL).open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )
        except Exception:
            pass
        return str(target)
    except Exception:
        return None




def _r55_tile_adaptation_from_state(tile_state, decision, initial_tile) -> dict:
    """Deterministic R55-E11 RUN tile-adaptation record (never volatile peaks)."""
    history = [dict(item) for item in tile_state.get("shrink_history", [])]
    units = []
    for record in tile_state.get("units", []):
        units.append(
            {
                "unit_index": int(record.get("unit_index", -1)),
                "tile_px": record.get("tile_px"),
                "status": record.get("status"),
                "action": record.get("action"),
                "attempted": bool(record.get("attempted")),
                "completed": bool(record.get("completed")),
                "discarded": bool(record.get("discarded")),
            }
        )
    return {
        "schema": _R55E11_TILE_RUNTIME_SCHEMA,
        "mission": _R55E11_MISSION,
        "initial_tile_px": int(initial_tile),
        "effective_tile_px": int(tile_state.get("tile", initial_tile)),
        "declared_ladder": [int(t) for t in decision.gpu_tile_ladder],
        "scale": "gpu_tile_px",
        "growth": "forbidden",
        "walk": "strictly_smaller_declared_rungs_descending",
        "final_cause": tile_state.get("final_cause"),
        "shrink_count": int(len(history)),
        "shrink_history": history,
        "attempts": [dict(item) for item in tile_state.get("attempts", [])],
        "units": units,
        "intra_cell_oom_supported": False,
        "intra_cell_policy": (
            "Pressure/OOM INSIDE a cell is not adapted here. Any in-cell failure "
            "cancels the pipeline, releases every staged unit, discards all GPU "
            "results and performs exactly ONE whole-batch exact CPU retry."
        ),
    }


def _r55_gpu_tile_decision_dict(
    decision, mapping, bundle, *, runtime=None, legacy_probe_budget_bytes=None
) -> dict:
    """R55-E9/E10/E11: deterministic diagnostics for the GPU streaming tile.

    Reports the VRAM-derived maximum, the frozen declared ladder, the selected
    tile/prefetch and BOTH peaks that were actually validated: the exact ADDED
    host bound of the selected tile and the exact device workspace. R55-E11 adds
    the INITIAL vs EFFECTIVE tile, the runtime shrink history (when supplied)
    and an explicit distinction between the ACTIVE 70 % budget and the legacy
    50 % probe budget, so neither can be read as the other.
    """
    tile = int(mapping["gpu_streaming_tile"])
    host_by_tile = {
        int(t): int(b) for t, b in decision.gpu_host_bound_by_tile
    }
    device_by_tile = {
        int(t): int(b) for t, b in decision.device_peak_bytes_by_tile.items()
    }
    return {
        "schema": "zemosaic-r55e9-gpu-tile-ladder-v1",
        "max_tile_px": int(decision.gpu_tile_max),
        "declared_ladder": [int(t) for t in decision.gpu_tile_ladder],
        "selected_tile_px": int(tile),
        "initial_tile_px": int(decision.mapping["gpu_streaming_tile"]),
        "effective_tile_px": int(tile),
        "selected_prefetch_units": int(mapping["prefetch_units"]),
        "selected_host_added_bound_bytes": int(host_by_tile[tile]),
        "selected_host_aggregate_bytes": int(bundle.envelope.aggregate_bytes),
        "selected_device_peak_bytes": int(device_by_tile[tile]),
        "host_added_bound_bytes_by_tile": {
            str(int(t)): int(host_by_tile[int(t)])
            for t in decision.gpu_tile_ladder
        },
        "device_peak_bytes_by_tile": {
            str(int(t)): int(device_by_tile[int(t)])
            for t in decision.gpu_tile_ladder
        },
        "vram_budget_bytes": int(decision.qualification["budget_bytes"]),
        "vram_percent": int(decision.qualification["percent"]),
        "selection_headroom_percent": int(
            _R55E10_VRAM_SELECTION_HEADROOM_PERCENT
        ),
        "selection_headroom_fraction": (
            _R55E10_VRAM_SELECTION_HEADROOM_PERCENT / 100.0
        ),
        "selection_budget_bytes": int(decision.gpu_selection_budget_bytes),
        "selected_device_selection_margin_bytes": int(
            decision.gpu_selection_budget_bytes - device_by_tile[tile]
        ),
        # R55-E11: budget provenance is explicit. ``vram_budget_bytes`` above is
        # the ACTIVE policy budget (70 %); the legacy probe budget is 50 % and is
        # reported separately so the two can never be confused.
        "vram_budget_source": "active_r55_policy_70",
        "legacy_probe_percent": 50,
        "legacy_probe_vram_budget_bytes": (
            None
            if legacy_probe_budget_bytes is None
            else int(legacy_probe_budget_bytes)
        ),
        "runtime_adaptation": (
            None if runtime is None else dict(runtime)
        ),
        "engine_provenance": _r55_engine_provenance(),
    }


def _r55_active_execution(bundle, mapping) -> dict:
    env = bundle.envelope
    if mapping.get("backend") == renv.BACKEND_GPU:
        return {
            "backend": "gpu",
            "cell_task_mode": "stream",
            "cpu_streaming_tile": None,
            "gpu_streaming_tile": int(mapping["gpu_streaming_tile"]),
            "units_in_flight": 1,
            "workers_per_unit": int(bundle.cache_workers_per_cell),
            "prefetch_units": int(mapping["prefetch_units"]),
            "queue_max": int(mapping["prefetch_units"]) + 1,
            "single_owner": True,
            "aggregate_bytes": int(env.aggregate_bytes),
            "stack_total_bytes": int(env.stack_total_bytes),
            "cache_total_bytes": int(env.cache_total_bytes),
            "shared_bytes": int(env.shared_bytes),
            "layout": [int(env.layout[0]), int(env.layout[1])],
            "refinement_rank": int(env.refinement_rank),
        }
    return {
        "backend": "cpu",
        "cell_task_mode": str(bundle.chosen_mode),
        "cpu_streaming_tile": mapping["cpu_streaming_tile"],
        "units_in_flight": int(bundle.cells_in_flight),
        "workers_per_unit": int(bundle.cache_workers_per_cell),
        "prefetch_units": 0,
        "aggregate_bytes": int(env.aggregate_bytes),
        "stack_total_bytes": int(env.stack_total_bytes),
        "cache_total_bytes": int(env.cache_total_bytes),
        "shared_bytes": int(env.shared_bytes),
        "layout": [int(env.layout[0]), int(env.layout[1])],
        "refinement_rank": int(env.refinement_rank),
    }


# ---------------------------------------------------------------------------
# ZM-ZEGRID-R55-E1 — CPU RUN checkpoints, deterministic waves and telemetry.
# Active CPU is the ONLY path partitioned into waves. Legacy/observe keep the
# whole-batch executor untouched.
# ---------------------------------------------------------------------------

_R55E1_RUN_MISSION = "ZM-ZEGRID-R55-E1-RUN-CHECKPOINTS-20261023"
_R55E1_RUN_STAGE = "run"
_R55E1_UNAVAILABLE_SHRINKS = (
    {
        "step": "prefetch_units",
        "available": False,
        "reason": "E1_PREFETCH_FROZEN_ZERO",
        "detail": (
            "E1 keeps prefetch_units=0 by contract; E2 owns producer/consumer "
            "prefetch execution"
        ),
    },
    {
        "step": "backend",
        "available": False,
        "reason": "E1_BACKEND_FROZEN_CPU",
        "detail": (
            "E1 keeps backend=cpu by contract; GPU selection/VRAM rungs are "
            "out of scope"
        ),
    },
)


class _R55RunRefused(RuntimeError):
    """A RUN checkpoint refused BEFORE the next wave (fail closed).

    ``storage_fallback_eligible`` is STRUCTURED evidence that the refusal comes
    from a VALID, freshly measured capacity shortage (the scheduler exhausted
    every adjacent shrink against a real host-RAM budget). A malformed / failed
    / missing observation, a disk refusal, an internal error or a config error is
    NEVER eligible: the bounded storage fallback must not turn a fail-closed
    observation failure into a published mosaic.
    """

    def __init__(self, reason_code, detail, *, record,
                 storage_fallback_eligible=False):
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = str(reason_code)
        self.detail = str(detail)
        self.record = dict(record)
        self.storage_fallback_eligible = bool(storage_fallback_eligible)
        self.record["storage_fallback_eligible"] = bool(
            storage_fallback_eligible
        )


def _r55_run_disk_root(cache_root, output_dir) -> Path:
    return Path(cache_root) if cache_root is not None else Path(output_dir)


def _r55_run_checkpoint_sample(disk_root):
    """One fresh RUN checkpoint fact set (strict 90 % host + disk + optional)."""
    return zrt.collect_run_facts(
        root=disk_root, available_fn=available_memory_bytes
    )


def _r55_sample_pressure(disk_root):
    """Lightweight in-wave pressure sample (never a decision by itself).

    Uses the SAME strict available-memory source as the checkpoint so tests
    that inject the host observation also control the in-wave latch. The
    sample is only a candidate for the NEXT boundary.
    """
    return zrt.collect_run_facts(
        root=disk_root,
        available_fn=available_memory_bytes,
        include_diagnostics=False,
    )


def _r55_active_run_context(
    decision, candidate, *, units_total, raw, budget, disk_free
):
    """Rebind the owning scheduler context to fresh RUN facts.

    ``units_total`` stays the FROZEN active-unit count (the scheduler context
    identity must remain exactly the accepted one). The remaining-unit
    projection is recorded and enforced by the wave bound, not by mutating the
    frozen identity.
    """
    base = _r55_active_shrink_context(decision, candidate)
    return replace(
        base,
        units_total=int(units_total),
        raw_available_bytes=int(raw),
        factor_num=int(rcontract.ADMIT_FACTOR_NUM),
        factor_den=int(rcontract.ADMIT_FACTOR_DEN),
        budget_bytes=int(budget),
        disk_free_bytes=int(disk_free),
    )


def _r55_run_checkpoint(
    *, decision, candidate, mapping, bundle, remaining_units, wave_index, facts,
    latched=None
):
    """Mandatory fresh checkpoint before one wave.

    The current candidate is an UPPER BOUND. Corrections are only the
    scheduler-produced adjacent shrink steps in the fixed global order
    ``prefetch -> workers -> K -> tile -> backend``. E1 records the frozen
    prefetch/backend endpoints as unavailable instead of inventing them.
    """
    try:
        facts = zrt.validate_checkpoint_facts(facts)
    except zrt.RuntimeControlError as exc:
        raise _R55RunRefused(
            renv.REASON_REFUSE_HOST_RAM,
            f"malformed RUN checkpoint facts: {exc}",
            record={
                "schema": zrt.RUN_CONTROL_SCHEMA,
                "status": "refused",
                "reason_code": renv.REASON_REFUSE_HOST_RAM,
                "detail": f"malformed RUN checkpoint facts: {exc}",
                "wave_index": int(wave_index),
                "remaining_units": int(max(0, remaining_units)),
                "shrink_steps": [],
                "unavailable_shrinks": [dict(x) for x in _R55E1_UNAVAILABLE_SHRINKS],
            },
        ) from exc
    fresh_raw = int(facts["host"]["raw_available_bytes"])
    fresh_budget = int(facts["host"]["budget_bytes"])
    latched_host = None
    if latched is not None:
        try:
            latched_host = zrt.validate_latched_fact(latched)
        except zrt.RuntimeControlError as exc:
            raise _R55RunRefused(
                renv.REASON_REFUSE_HOST_RAM,
                f"malformed latched RUN fact: {exc}",
                record={
                    "schema": zrt.RUN_CONTROL_SCHEMA,
                    "status": "refused",
                    "reason_code": renv.REASON_REFUSE_HOST_RAM,
                    "detail": f"malformed latched RUN fact: {exc}",
                    "wave_index": int(wave_index),
                    "remaining_units": int(max(0, remaining_units)),
                    "shrink_steps": [],
                    "unavailable_shrinks": [
                        dict(x) for x in _R55E1_UNAVAILABLE_SHRINKS
                    ],
                },
            ) from exc
    # Consume the pending mid-wave low-pressure fact exactly once and combine
    # it CONSERVATIVELY with the fresh sample. A transient drop that recovered
    # before this boundary still governs the next wave; it never mutates the
    # already-running wave.
    effective_raw = int(fresh_raw)
    host_source = "fresh"
    if latched_host is not None and int(
        latched_host["raw_available_bytes"]
    ) < int(fresh_raw):
        effective_raw = int(latched_host["raw_available_bytes"])
        host_source = "fresh+latched"
    raw = int(effective_raw)
    budget = int(zrt.run_budget_bytes(raw))
    disk_free = int(facts["disk"]["free_bytes"])
    remaining_units = zrt.strict_int(
        remaining_units, "remaining_units", minimum=0
    )
    # The frozen identity (layout/cells/frames/halo/rank) must not be mutated:
    # keep the accepted active-unit total so the context stays valid. The wave
    # itself is bounded by K and by the actual remaining task count.
    ctx_units = max(1, len(decision.facts["cell_ids"]))
    ctx = _r55_active_run_context(
        decision,
        candidate,
        units_total=ctx_units,
        raw=raw,
        budget=budget,
        disk_free=disk_free,
    )

    def _record(status, chosen_candidate, chosen_mapping, steps, changed):
        return {
            "schema": zrt.RUN_CONTROL_SCHEMA,
            "stage": _R55E1_RUN_STAGE,
            "wave_index": int(wave_index),
            "status": str(status),
            "remaining_units": int(remaining_units),
            "changed": bool(changed),
            "host": {
                "raw_available_bytes": int(raw),
                "budget_bytes": int(budget),
                "factor_num": int(rcontract.ADMIT_FACTOR_NUM),
                "factor_den": int(rcontract.ADMIT_FACTOR_DEN),
            },
            "fresh_host": {
                "raw_available_bytes": int(fresh_raw),
                "budget_bytes": int(fresh_budget),
                "factor_num": int(rcontract.ADMIT_FACTOR_NUM),
                "factor_den": int(rcontract.ADMIT_FACTOR_DEN),
            },
            "latched_host": (
                None if latched_host is None else dict(latched_host)
            ),
            "host_source": str(host_source),
            "disk": {"free_bytes": int(disk_free)},
            "candidate": zsched.active_candidate_tuple(chosen_candidate),
            "mapping": dict(chosen_mapping),
            "shrink_steps": [dict(step) for step in steps],
            "unavailable_shrinks": [
                dict(x) for x in _R55E1_UNAVAILABLE_SHRINKS
            ],
        }

    def _deterministic(record):
        # Include the latched fact ONLY when it actually governed the decision.
        # The presence/timing of an equal-pressure sample is observational and
        # must not make two otherwise identical runs differ.
        latched_for_decision = (
            record["latched_host"]
            if record.get("host_source") == "fresh+latched"
            else None
        )
        return {
            key: record[key]
            for key in (
                "schema",
                "stage",
                "wave_index",
                "status",
                "remaining_units",
                "changed",
                "host",
                "fresh_host",
                "host_source",
                "candidate",
                "mapping",
                "shrink_steps",
                "unavailable_shrinks",
            )
        } | {"latched_host": latched_for_decision}

    # Fast path: the accepted candidate still fits the fresh 90 % budget and the
    # already-admitted product envelope still fits too. No mapping change.
    try:
        materialized = zsched.materialize_active_candidate(ctx, candidate)
    except (zsched.SchedulerError, rcontract.RamContractError) as exc:
        raise _R55RunRefused(
            renv.REASON_INTERNAL_ERROR,
            f"cannot materialize the current RUN candidate: {exc}",
            record=_record("refused", candidate, mapping, [], False),
        ) from exc
    if (
        materialized.feasible
        and int(materialized.disk_bytes) <= int(disk_free)
        and int(bundle.envelope.aggregate_bytes) <= int(budget)
    ):
        record = _record("accepted", candidate, mapping, [], False)
        return {
            "record": record,
            "deterministic_record": _deterministic(record),
            "bundle": bundle,
            "candidate": candidate,
            "mapping": mapping,
        }

    examined = []
    # ZM-ZEGRID-ACTIVE-OOC-FALLBACK rework-2 (F5): classify the CURRENT candidate
    # BEFORE any shrink exhaustion. A disk shortage (or a non-memory materialization
    # failure) at the last rung must never be relabelled as host RAM.
    _current_feasible = bool(materialized.feasible)
    saw_disk = int(materialized.disk_bytes) > int(disk_free) or (
        not _current_feasible
        and materialized.reason_code == renv.REASON_REFUSE_DISK
    )
    saw_host_ram = (
        int(bundle.envelope.aggregate_bytes) > int(budget)
        or (
            not _current_feasible
            and materialized.reason_code == renv.REASON_REFUSE_HOST_RAM
        )
        or (
            not _current_feasible
            and int(materialized.host_peak_bytes) > int(budget)
        )
    )
    _NON_MEMORY_REFUSALS = (
        renv.REASON_INTERNAL_ERROR,
        renv.REASON_UNKNOWN_BACKEND,
        renv.REASON_REFUSE_VRAM,
        renv.REASON_PINNED_GEOMETRY,
        renv.REASON_PARITY_FAIL,
    )
    saw_non_memory = bool(
        not _current_feasible
        and materialized.reason_code in _NON_MEMORY_REFUSALS
    )
    # ZM-ZEGRID-ACTIVE-OOC-FALLBACK rework-3 (L-3): keep the SPECIFIC non-memory
    # refusal code for diagnostics (e.g. VRAM) instead of flattening everything
    # to INTERNAL at exhaustion. Eligibility is unaffected (non-RAM => False).
    non_memory_reason = (
        materialized.reason_code if saw_non_memory else None
    )
    current = candidate
    while True:
        step_name, nxt = zsched.next_shrink(current, ctx)
        if step_name is None or nxt is None:
            # F5: only a REAL host-budget overshoot, with no disk or non-memory
            # failure contributing, is a capacity refusal.
            if saw_disk:
                reason = renv.REASON_REFUSE_DISK
            elif saw_non_memory or not saw_host_ram:
                # Preserve a SPECIFIC non-memory refusal code (VRAM) when it is
                # the cause; other non-memory causes stay INTERNAL_ERROR.
                reason = (
                    renv.REASON_REFUSE_VRAM
                    if non_memory_reason == renv.REASON_REFUSE_VRAM
                    else renv.REASON_INTERNAL_ERROR
                )
            else:
                reason = renv.REASON_REFUSE_HOST_RAM
            detail = (
                "no feasible adjacent CPU shrink remains for RUN wave "
                f"{wave_index} (remaining_units={remaining_units}, raw={raw}, "
                f"budget={budget}, disk_free={disk_free}; "
                f"saw_disk={bool(saw_disk)}, "
                f"saw_host_ram={bool(saw_host_ram)}, "
                f"saw_non_memory={bool(saw_non_memory)})"
            )
            record = _record("refused", candidate, mapping, examined, False)
            record["reason_code"] = reason
            record["detail"] = detail
            record["last_candidate"] = zsched.active_candidate_tuple(current)
            record["classification"] = {
                "saw_disk": bool(saw_disk),
                "saw_host_ram": bool(saw_host_ram),
                "saw_non_memory": bool(saw_non_memory),
                "non_memory_reason": non_memory_reason,
            }
            and_eligible = bool(
                reason == renv.REASON_REFUSE_HOST_RAM
                and saw_host_ram
                and not saw_disk
                and not saw_non_memory
            )
            raise _R55RunRefused(
                reason, detail, record=record,
                storage_fallback_eligible=and_eligible,
            )
        try:
            zsched.assert_shrink_only(current, nxt)
            materialized = zsched.materialize_active_candidate(ctx, nxt)
        except (zsched.SchedulerError, rcontract.RamContractError) as exc:
            reason = renv.REASON_INTERNAL_ERROR
            detail = f"scheduler could not materialize an adjacent shrink: {exc}"
            record = _record("refused", candidate, mapping, examined, False)
            record["reason_code"] = reason
            record["detail"] = detail
            record["last_candidate"] = zsched.active_candidate_tuple(current)
            raise _R55RunRefused(reason, detail, record=record) from exc
        entry = {
            "step": step_name,
            "from": zsched.active_candidate_tuple(current),
            "to": zsched.active_candidate_tuple(materialized),
            "accepted": False,
            "reason_code": materialized.reason_code,
            "detail": materialized.infeasible_reason,
            "projections": {
                "host_peak_bytes": int(materialized.host_peak_bytes),
                "disk_bytes": int(materialized.disk_bytes),
                "device_peak_bytes": int(materialized.device_peak_bytes),
            },
        }
        if materialized.reason_code == renv.REASON_REFUSE_HOST_RAM:
            saw_host_ram = True
        elif materialized.reason_code in _NON_MEMORY_REFUSALS:
            saw_non_memory = True
            non_memory_reason = materialized.reason_code
        if materialized.reason_code == renv.REASON_REFUSE_DISK:
            saw_disk = True
        disk_ok = int(materialized.disk_bytes) <= int(disk_free)
        if not disk_ok:
            saw_disk = True
            entry["reason_code"] = renv.REASON_REFUSE_DISK
            entry["detail"] = (
                f"disk residency {materialized.disk_bytes} B > observed free "
                f"{disk_free} B"
            )
        if materialized.feasible and disk_ok:
            try:
                trial_mapping = zsched.active_mapping_for_candidate(
                    materialized,
                    tile_ladder=tuple(
                        int(t) for t in CPU_STREAMING_TILE_LADDER
                    ),
                    workers_ladder=tuple(
                        int(w) for w in zsched.WORKERS_LADDER
                    ),
                )
                trial_bundle = _r55_active_apply(
                    bundle,
                    mapping=trial_mapping,
                    facts=decision.facts,
                    plan_raw_available_bytes=raw,
                    plan_factor_num=rcontract.ADMIT_FACTOR_NUM,
                    plan_factor_den=rcontract.ADMIT_FACTOR_DEN,
                    plan_budget_bytes=budget,
                    stage=f"run_wave_{wave_index}",
                )
            except (zsched.SchedulerError, rcontract.RamContractError) as exc:
                entry["reason_code"] = renv.REASON_INTERNAL_ERROR
                entry["detail"] = (
                    "product envelope replay failed for the scheduler shrink: "
                    f"{exc}"
                )
            else:
                if trial_bundle.envelope.aggregate_bytes <= budget:
                    entry["accepted"] = True
                    entry["aggregate_bytes"] = int(
                        trial_bundle.envelope.aggregate_bytes
                    )
                    examined.append(entry)
                    record = _record(
                        "accepted", materialized, trial_mapping, examined, True
                    )
                    return {
                        "record": record,
                        "deterministic_record": _deterministic(record),
                        "bundle": trial_bundle,
                        "candidate": materialized,
                        "mapping": trial_mapping,
                    }
                entry["reason_code"] = renv.REASON_REFUSE_HOST_RAM
                entry["detail"] = (
                    "product aggregate "
                    f"{trial_bundle.envelope.aggregate_bytes} B > RUN budget "
                    f"{budget} B"
                )
                saw_host_ram = True
        examined.append(entry)
        current = materialized


def _r55_merge_wave_meta(metas, *, total_tasks, wave_count) -> dict:
    executors = []
    fallback_reasons = []
    for meta in metas:
        kind = str(meta.get("executor", "serial"))
        if kind not in executors:
            executors.append(kind)
        if meta.get("fallback_reason"):
            fallback_reasons.append(str(meta["fallback_reason"]))
    seconds = sum(float(meta.get("seconds", 0.0)) for meta in metas)
    return {
        "executor": (
            "wave:" + "+".join(executors) if executors else "serial"
        ),
        "wave_executors": list(executors),
        "parent_daemon": any(bool(meta.get("parent_daemon")) for meta in metas),
        "workers": max(
            (int(meta.get("workers", 1)) for meta in metas), default=1
        ),
        "tasks": int(total_tasks),
        "fallback": any(bool(meta.get("fallback")) for meta in metas),
        "fallback_reason": (
            "; ".join(fallback_reasons) if fallback_reasons else None
        ),
        "seconds": float(seconds),
        "seconds_per_unit": float(seconds) / max(1, int(total_tasks)),
        "waves": int(wave_count),
        "heartbeat_calls": int(
            sum(int(meta.get("heartbeat_calls", 0)) for meta in metas)
        ),
        "heartbeat_errors": int(
            sum(int(meta.get("heartbeat_errors", 0)) for meta in metas)
        ),
        "heartbeat_modes": sorted(
            {
                str(meta.get("heartbeat_mode", "none"))
                for meta in metas
            }
        ),
        "heartbeat_interval_s": float(zrt.RUN_MAX_SAMPLE_INTERVAL_S),
    }


def _r55_build_active_block(
    *,
    plan_set,
    final_candidate,
    mapping,
    bundle,
    initial_bundle,
    preflight_admission,
    launch_admission,
    preflight_trail,
    launch_trail,
    status,
    run=None,
    worker_adaptation=None,
    tile_ladder=None,
) -> dict:
    """Deterministic active manifest block (no fabricated observed metric).

    The PRIMARY ``plan_hash`` identifies the FINAL executable decision
    (``zsched.active_final_plan_hash``): final candidate + exact mapping +
    frozen identity + final validated PlanEnvelope. The D1 scheduler PlanSet
    hash is preserved separately as ``initial_plan_hash``.
    """
    hash_kwargs = {}
    if tile_ladder is not None:
        hash_kwargs["tile_ladder"] = tuple(int(t) for t in tile_ladder)
    final_hash = zsched.active_final_plan_hash(
        candidate=final_candidate,
        mapping=mapping,
        identity=plan_set.identity,
        envelope=bundle.envelope,
        **hash_kwargs,
    )
    combined_trail = []
    for item in preflight_trail:
        entry = dict(item)
        entry.setdefault("stage", "preflight")
        combined_trail.append(entry)
    for item in launch_trail:
        entry = dict(item)
        entry.setdefault("stage", "launch")
        combined_trail.append(entry)
    preflight_dict = _r55_admission_dict(preflight_admission)
    launch_dict = (
        None if launch_admission is None
        else _r55_admission_dict(launch_admission)
    )
    block = zsched.active_manifest_dict(
        plan_set,
        mapping=mapping,
        status=status,
        execution=_r55_active_execution(bundle, mapping),
        plan_observation=_r55_active_plan_observation(bundle),
        initial_plan_observation=_r55_active_plan_observation(initial_bundle),
        admission=launch_dict if launch_dict is not None else preflight_dict,
        preflight_admission=preflight_dict,
        launch_revalidation=launch_dict,
        shrink_trail=combined_trail,
        final_candidate=final_candidate,
        initial_candidate=plan_set.chosen,
        initial_plan_hash=plan_set.plan_hash,
        final_envelope=bundle.envelope,
        plan_hash=final_hash,
        plan_hash_schema=zsched.ACTIVE_FINAL_PLAN_SCHEMA,
        note=(
            (
                "R55-E2 active GPU: a single-owner (K=1) GPU plan was "
                "preflight-admitted BEFORE the gauge, then freshly revalidated "
                "against 90 % host RAM and fresh VRAM before every unit. The "
                "CPU producer stages at most prefetch_units+1 units into an "
                "explicit bounded queue; the single GPU consumer owns the "
                "device. Any pressure/OOM/error cancels, releases every staged "
                "buffer and performs at most ONE exact whole-batch CPU retry "
                "from a clean state (no mixed result). Deterministic decisions "
                "are separated from observational pipeline/VRAM telemetry; no "
                "observed metric is fabricated."
            )
            if mapping.get("backend") == renv.BACKEND_GPU
            else (
                "R55-D2/E1 active: a CPU plan was preflight-admitted BEFORE the "
                "gauge for zero-gauge refusals, then freshly revalidated at the "
                "90 % launch ceiling immediately before the stack. RUN waves are "
                "deterministic contiguous slices bounded by the accepted K; a "
                "mandatory fresh 90 % checkpoint precedes every wave. Pressure "
                "corrections are scheduler-produced shrink-only CPU steps on the "
                "frozen identity; PLAN/ADMIT/RUN observations and the rebound "
                "final envelope are recorded truthfully. The deterministic RUN "
                "decision structure is separated from observational telemetry "
                "(wall/RSS/cgroup/IO/disk free). No observed metric is fabricated "
                "and no second unreported plan exists."
            )
        ),
    )
    if worker_adaptation is not None:
        # R55-E8: deterministic bounded decision trail of the non-overlapping
        # gauge/cache concurrency phases actually selected under this B_plan.
        block["worker_adaptation"] = dict(worker_adaptation)
    if run is not None:
        run_payload = dict(run)
        block["run"] = run_payload
        block["run_hash"] = zrt.sha256_payload(run_payload)
        # The deterministic RUN structure is part of the auditable active
        # evidence. Re-stamp the block hash so it covers the RUN decisions too;
        # volatile telemetry lives in a separate ``r55_run_telemetry`` block and
        # is never part of this hash.
        payload = {k: v for k, v in block.items() if k != "evidence_hash"}
        block["evidence_hash"] = zrt.sha256_payload(payload)
    return block


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
    # ZM-ZEGRID-GPU-ADMISSION-FALLBACK: the EXACT initial-bundle request is kept
    # in one place so a later bounded retry rebuilds the initial bundle with the
    # SAME raw/budget/layout/halo/science demand (never a re-derived guess).
    _plan_request = dict(
        canvas=canvas,
        descs=descs,
        footprints=None,
        raw=plan_raw_bytes,
        budget=plan_budget_bytes,
        pinned_layout=pinned_layout,
        gauge_workers=workers,
        cache_build_workers=(workers if workers is not None else None),
        layout_substep=layout_substep,
        bundle_common=_bundle_common,
        active=(scheduler_mode == zsched.MODE_ACTIVE),
        emit=_emit_live,
    )
    try:
        with timings.timed("layout"):
            _initial_bundle = _plan_bundle_auto(**_plan_request)
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

    # ------------------------------------------------------------------
    # ZM-ZEGRID-R55-D2 — ACTIVE decision. The initial bundle has frozen the
    # layout, memberships, frame set, halo and science identity. Decide and
    # admit the CPU plan NOW, before any global gauge or stack work. Any
    # refusal/error is FATAL (no legacy fallback) and leaves no science output.
    # ------------------------------------------------------------------
    _r55_active_decision = None
    _r55_active_gpu_decision = None
    _r55_gpu_fallback_reason = None
    _r55_active_candidate = None
    _r55_active_mapping = None
    _r55_active_adm = None
    _r55_active_trail = []
    _r55_active_bundle = None
    _r55_active_preflight_adm = None
    _r55_active_preflight_trail = []
    _r55_active_launch_adm = None
    _r55_active_launch_trail = []
    if scheduler_mode == zsched.MODE_ACTIVE:
        _r55_frame_ids = tuple(
            getattr(getattr(f, "frame_id", None), "logical_path", str(f))
            for f in (descs or ())
        )
        # R55-E8: the PRODUCT GPU authority decides whether the richer ACTIVE
        # planner is entered. Requiring an already-host-feasible single-tile
        # GPU bundle would discard a feasible tile128 merely because the
        # VRAM-max tile213 missed host RAM (Linux halo16 legacy pre-fallback).
        # The bundle consistency check below still fails closed the inverse
        # case (GPU-shaped bundle while the product backend is CPU).
        _r55_gpu_active_requested = bool(gpu_used)
        if bool(_initial_bundle.gpu_used) and not _r55_gpu_active_requested:
            # Fail closed on an INCONSISTENT bundle (GPU-shaped but the product
            # GPU backend is not selected): never silently execute it on CPU.
            raise RuntimeError(
                "R55-E2 active refuses an inconsistent GPU bundle before the "
                "gauge/stack: gpu_used=True while the product GPU backend is "
                "not effective/requested (no silent CPU coercion)."
            )
        if _r55_gpu_active_requested:
            # ZM-ZEGRID-R55-E2: ACTIVE GPU is a qualified, single-owner plan.
            # A non-qualified candidate is REFUSED with a closed reason (never
            # coerced), while an ADMIT-time pressure miss is the ONE qualified
            # whole-batch CPU fallback path.
            #
            # ZM-ZEGRID-GPU-ADMISSION-FALLBACK: a QUALIFIED dense ACTIVE GPU
            # host-RAM refusal means the direct/in-memory model cannot hold the
            # resident frame set (the dense host bound is priced per tile over N
            # resident frames). Nothing has started yet (no gauge, no stack, no
            # cache work), so the run gets exactly ONE bounded retry with the
            # SAME initial request, mirroring the plan/admission fallbacks.
            _r55_readiness = zgr.probe_gpu_readiness_safe()

            def _r55_gpu_attempt(initial_bundle):
                """Decide + admit ONE ACTIVE GPU attempt on an initial bundle."""
                _attempt_decision = _r55_active_gpu_decide(
                    bundle=initial_bundle,
                    cell_ctxs=initial_bundle.cell_ctxs,
                    canvas=canvas,
                    halo_px=halo_px,
                    cache_root=cache_root,
                    output_dir=output_dir,
                    frame_ids=_r55_frame_ids,
                    gpu_probe=gpu_ctx,
                    readiness=_r55_readiness,
                )
                return (_attempt_decision,) + _r55_active_gpu_admit(
                    initial_bundle=initial_bundle,
                    decision=_attempt_decision,
                )

            def _r55_publish_gpu_admission(*, fallback_count):
                nonlocal _r55_active_preflight_adm, _r55_active_preflight_trail
                _r55_active_preflight_adm = _r55_active_adm
                _r55_active_preflight_trail = list(_r55_active_trail)
                _emit(
                    "ZeGrid: R55-E2/E9 active GPU plan ADMITTED before the gauge — "
                    f"tile={_r55_active_mapping['gpu_streaming_tile']} "
                    f"(auto max={_r55_active_gpu_decision.gpu_tile_max}, "
                    f"declared ladder={list(_r55_active_gpu_decision.gpu_tile_ladder)}) "
                    f"K={_r55_active_mapping['units_in_flight']} "
                    f"workers_per_unit={_r55_active_mapping['workers_per_unit']} "
                    f"prefetch_units={_r55_active_mapping['prefetch_units']} "
                    f"queue_max={_r55_active_mapping['prefetch_units'] + 1} "
                    f"host_aggregate="
                    f"{_r55_active_bundle.envelope.aggregate_bytes} "
                    f"device_peak="
                    f"{_r55_active_gpu_decision.device_peak_bytes_by_tile[int(_r55_active_mapping['gpu_streaming_tile'])]} "
                    f"vram_percent={_r55_active_gpu_decision.qualification['percent']} "
                    f"scheduler_plan_hash="
                    f"{_r55_active_gpu_decision.plan_set.plan_hash} "
                    f"storage_profile={_STORAGE_EFFECTIVE} "
                    f"storage_fallback_count={int(fallback_count)}",
                    callback=progress_callback,
                )

            def _r55_record_gpu_cpu_fallback(exc):
                nonlocal _r55_gpu_fallback_reason
                nonlocal _r55_active_gpu_decision, _r55_active_bundle
                # Qualified whole-batch CPU fallback from a clean state; the
                # GPU plan is abandoned entirely (no mixed result).
                _r55_gpu_fallback_reason = {
                    "stage": "admission",
                    "reason_code": exc.reason_code,
                    "detail": exc.detail,
                    "telemetry": dict(exc.telemetry),
                }
                _r55_active_gpu_decision = None
                _r55_active_bundle = None
                _emit(
                    "ZeGrid: R55-E2 active GPU admission fell back to ONE exact "
                    f"whole-batch CPU plan — {exc.reason_code}: {exc.detail}",
                    lvl="WARN", callback=progress_callback,
                )

            def _r55_double_refusal(dense_exc, bounded_exc):
                """TERMINAL refusal after BOTH storage profiles were tried ONCE."""
                dense_code = str(
                    getattr(dense_exc, "reason_code", type(dense_exc).__name__)
                )
                bounded_code = str(
                    getattr(bounded_exc, "reason_code", type(bounded_exc).__name__)
                )
                detail = (
                    "R55-E2 active GPU refused before the gauge/stack in BOTH "
                    "storage profiles (dense: "
                    f"{dense_code}: {dense_exc} | bounded: "
                    f"{bounded_code}: {bounded_exc}); fallback_exhausted=True "
                    "(no second retry, no silent GPU->CPU coercion)"
                )
                _emit(
                    "ZeGrid: " + detail, lvl="ERROR",
                    callback=progress_callback,
                )
                err = RuntimeError(detail)
                err.fallback_exhausted = True
                err.storage_refusals = {
                    "dense": {
                        "reason_code": dense_code,
                        "detail": str(dense_exc),
                    },
                    "bounded": {
                        "reason_code": bounded_code,
                        "detail": str(bounded_exc),
                    },
                }
                return err

            try:
                (
                    _r55_active_gpu_decision,
                    _r55_active_bundle,
                    _r55_active_candidate,
                    _r55_active_mapping,
                    _r55_active_adm,
                    _r55_active_trail,
                ) = _r55_gpu_attempt(_initial_bundle)
                _r55_publish_gpu_admission(
                    fallback_count=int(_STORAGE_FALLBACK_COUNT)
                )
            except _R55GpuRefused as _r55_gpu_exc:
                _r55_dense_gpu_refusal = _r55_gpu_exc
                if not (
                    str(_r55_dense_gpu_refusal.reason_code)
                    == renv.REASON_REFUSE_HOST_RAM
                    and _storage_fallback_available()
                ):
                    # Not a qualified dense host-RAM refusal (VRAM, disk,
                    # observation, qualification/parity, forced profile, or an
                    # already-consumed fallback): terminal, unchanged.
                    _emit(
                        "ZeGrid: R55-E2 active GPU REFUSED before the gauge/stack — "
                        f"{_r55_gpu_exc.reason_code}: {_r55_gpu_exc.detail}",
                        lvl="ERROR", callback=progress_callback,
                    )
                    raise RuntimeError(
                        "R55-E2 active GPU refused before the gauge/stack: "
                        f"{_r55_gpu_exc.reason_code}: {_r55_gpu_exc.detail}"
                    ) from _r55_gpu_exc
                # ONE bounded retry: same raw/budget/layout/halo/science.
                if not _activate_bounded_storage(
                    phase="plan_gpu_admission",
                    reason_code=str(_r55_dense_gpu_refusal.reason_code),
                    detail=str(_r55_dense_gpu_refusal.detail),
                    emit=_emit_live,
                ):
                    _emit(
                        "ZeGrid: R55-E2 active GPU REFUSED before the gauge/stack "
                        "and the bounded retry is no longer available — "
                        f"{_r55_gpu_exc.reason_code}: {_r55_gpu_exc.detail}",
                        lvl="ERROR", callback=progress_callback,
                    )
                    raise RuntimeError(
                        "R55-E2 active GPU refused before the gauge/stack: "
                        f"{_r55_gpu_exc.reason_code}: {_r55_gpu_exc.detail}"
                    ) from _r55_gpu_exc
                _emit(
                    "ZeGrid: dense ACTIVE GPU admission refused host RAM — "
                    "retrying ONCE with the bounded storage profile on the SAME "
                    "geometry/halo/science (the bounded retry is the ONLY profile "
                    "switch; a second refusal is terminal): "
                    f"{_r55_dense_gpu_refusal.detail}",
                    lvl="WARN", callback=progress_callback,
                )
                try:
                    # Rebuild the initial bundle with the EXACT same demand. The
                    # layout sub-step reporter already recorded the first attempt
                    # (observability only), so the retry carries None exactly like
                    # every other bounded rebuild.
                    _initial_bundle = _plan_bundle_auto(
                        **{**_plan_request, "layout_substep": None}
                    )
                    # EVERY derived consumer must reflect the bounded bundle:
                    # no stale dense layout/footprints/cell contexts/workers.
                    layout = _initial_bundle.layout
                    footprints = _initial_bundle.footprints
                    cell_ctxs = _initial_bundle.cell_ctxs
                    nx, ny = int(layout["nx"]), int(layout["ny"])
                    cpu_streaming = _initial_bundle.cpu_streaming
                    _cs_selected = cpu_streaming.get("selected_tile_size")
                    cpu_streaming_tile = int(_initial_bundle.cpu_streaming_tile)
                    workers = int(_initial_bundle.gauge_workers)
                    cache_build_workers = int(
                        _initial_bundle.cache_build_workers
                    )
                    _cache_fp = zpar.cache_footprint_bytes(
                        _initial_bundle.worst_patch_px or canvas_px
                    )
                    _emit(
                        "ZeGrid: bounded storage rebuilt the initial bundle — "
                        f"layout {nx}x{ny} ({layout['cell_count']} cells, "
                        f"source={layout.get('layout_source', 'auto')}), CPU "
                        f"streaming tile={_cs_selected}, gauge={workers}, "
                        f"cache_build={cache_build_workers}, predicted bound "
                        f"{layout['predicted_bound_bytes'] / 2**20:.1f} MiB",
                        callback=progress_callback,
                    )
                    (
                        _r55_active_gpu_decision,
                        _r55_active_bundle,
                        _r55_active_candidate,
                        _r55_active_mapping,
                        _r55_active_adm,
                        _r55_active_trail,
                    ) = _r55_gpu_attempt(_initial_bundle)
                except (
                    zal.LayoutInfeasible, zpar.ModeInfeasible,
                ) as _r55_bounded_exc:
                    raise _r55_double_refusal(
                        _r55_dense_gpu_refusal, _r55_bounded_exc
                    ) from _r55_bounded_exc
                except _R55GpuRefused as _r55_bounded_exc:
                    raise _r55_double_refusal(
                        _r55_dense_gpu_refusal, _r55_bounded_exc
                    ) from _r55_bounded_exc
                except _R55GpuFallback as _r55_bounded_exc:
                    # ADMIT-time pressure under the bounded profile keeps the
                    # EXISTING typed whole-batch CPU fallback (not a coercion).
                    _r55_record_gpu_cpu_fallback(_r55_bounded_exc)
                else:
                    _r55_publish_gpu_admission(fallback_count=1)
            except _R55GpuFallback as _r55_gpu_exc:
                _r55_record_gpu_cpu_fallback(_r55_gpu_exc)

        if _r55_active_gpu_decision is None:
            def _r55_cpu_preflight(bundle):
                decision = _r55_active_decide(
                    bundle=bundle,
                    cell_ctxs=bundle.cell_ctxs,
                    canvas=canvas,
                    halo_px=halo_px,
                    cache_root=cache_root,
                    output_dir=output_dir,
                    frame_ids=_r55_frame_ids,
                )
                return (decision, _r55_active_admit(
                    initial_bundle=bundle,
                    decision=decision,
                ))

            try:
                (
                    _r55_active_decision,
                    (
                        _r55_active_bundle,
                        _r55_active_candidate,
                        _r55_active_mapping,
                        _r55_active_adm,
                        _r55_active_trail,
                    ),
                ) = _r55_cpu_preflight(_initial_bundle)
            except (zsched.SchedulerError, rcontract.RamContractError) as _r55_exc:
                # ZM-ZEGRID-ACTIVE-OOC-FALLBACK rework-1 (MEDIUM): only a TYPED
                # capacity refusal is eligible. Nothing has started yet (no
                # gauge/stack), so a qualified host-RAM miss switches the run
                # ONCE to bounded on the SAME pinned geometry/halo/science.
                _adm_mem, _adm_code, _adm_detail = _storage_admission_refusal(
                    _r55_exc
                )
                if not (
                    _adm_mem
                    and _activate_bounded_storage(
                        phase="admission_preflight",
                        reason_code=_adm_code,
                        detail=_adm_detail, emit=_emit_live,
                    )
                ):
                    _emit(
                        "ZeGrid: R55-D2 active REFUSED before the gauge/stack — "
                        f"{_r55_exc}",
                        lvl="ERROR", callback=progress_callback,
                    )
                    raise RuntimeError(
                        "R55-D2 active refused before the gauge/stack: "
                        + str(_r55_exc)
                    ) from _r55_exc
                _emit(
                    "ZeGrid: dense ACTIVE admission refused for RAM — retrying "
                    "ONCE with the bounded storage profile on the SAME "
                    f"{int(nx)}x{int(ny)} geometry: {_r55_exc}",
                    lvl="WARN", callback=progress_callback,
                )
                try:
                    # ZM-ZEGRID-ACTIVE-OOC-FALLBACK rework-1 (HIGH): the exact
                    # admitted geometry/rank live on the VALIDATED envelope, not
                    # on the bundle object.
                    _pinned_layout = tuple(
                        int(v) for v in _initial_bundle.envelope.layout
                    )
                    _pinned_rank = int(
                        _initial_bundle.envelope.refinement_rank
                    )
                    _initial_bundle = _build_execution_bundle(
                        canvas, descs, footprints,
                        raw_available_bytes=plan_raw_bytes,
                        budget_bytes=plan_budget_bytes,
                        pinned_layout=_pinned_layout,
                        min_refinement_rank=_pinned_rank,
                        max_refinement_rank=_pinned_rank,
                        layout_substep=None,
                        **_bundle_common,
                    )
                    (
                        _r55_active_decision,
                        (
                            _r55_active_bundle,
                            _r55_active_candidate,
                            _r55_active_mapping,
                            _r55_active_adm,
                            _r55_active_trail,
                        ),
                    ) = _r55_cpu_preflight(_initial_bundle)
                except (
                    zal.LayoutInfeasible, zpar.ModeInfeasible,
                    zsched.SchedulerError, rcontract.RamContractError,
                ) as _r55_exc2:
                    _emit(
                        "ZeGrid: NO ACTIVE plan can be admitted in EITHER "
                        "storage profile before the gauge — refusing. dense: "
                        f"{_r55_exc} | bounded: {_r55_exc2}",
                        lvl="ERROR", callback=progress_callback,
                    )
                    raise RuntimeError(
                        "R55-D2 active refused in both storage profiles "
                        f"(dense: {_r55_exc}; bounded: {_r55_exc2})"
                    ) from _r55_exc2
                layout = _initial_bundle.layout
                cell_ctxs = _initial_bundle.cell_ctxs
                nx, ny = int(layout["nx"]), int(layout["ny"])
            _r55_active_preflight_adm = _r55_active_adm
            _r55_active_preflight_trail = list(_r55_active_trail)
            _emit(
                "ZeGrid: R55-D2 active plan ADMITTED before the gauge — "
                f"phase={_r55_active_mapping['phase']} "
                f"tile={_r55_active_mapping['cpu_streaming_tile']} "
                f"units_in_flight={_r55_active_mapping['units_in_flight']} "
                f"workers_per_unit={_r55_active_mapping['workers_per_unit']} "
                f"prefetch_units=0 "
                f"scheduler_plan_hash={_r55_active_decision.plan_set.plan_hash} "
                f"preflight_shrink_steps={len(_r55_active_preflight_trail)}",
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

    if scheduler_mode == zsched.MODE_ACTIVE:
        # R55-D2: the active plan was already fitted and admitted BEFORE the
        # gauge. Do NOT run the legacy admission controller (no second plan).
        _adm, _final = _r55_active_adm, _r55_active_bundle
        # R55-E8: the global gauge executes with the COHERENT worker value the
        # same B_plan/margin admitted, never the pre-adaptation bundle value.
        workers = int(_final.gauge_workers)
        cache_build_workers = int(_final.cache_build_workers)
        _r55_worker_selection = (
            _r55_active_gpu_decision
            if _r55_active_gpu_decision is not None
            else _r55_active_decision
        )
        if _r55_worker_selection is not None:
            _emit(
                "ZeGrid: R55-E8 coherent workers — gauge="
                f"{int(_r55_worker_selection.gauge_workers)} "
                f"(from {int(_initial_bundle.gauge_workers)}), cache_build="
                f"{int(_r55_worker_selection.cache_workers)} "
                f"(from {int(_initial_bundle.cache_build_workers)}), "
                f"B_plan={int(_final.envelope.budget_bytes)} B, margin="
                f"{int(renv.DEFAULT_MARGIN_NUM)}/{int(renv.DEFAULT_MARGIN_DEN)}, "
                f"trail_steps={len(_r55_worker_selection.worker_trail)}",
                callback=progress_callback,
            )
    else:
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

    # ------------------------------------------------------------------
    # ZM-ZEGRID-R55-D2 rework-1 — FRESH 90 % LAUNCH REVALIDATION.
    # The gauge and its cleanup may have changed available RAM. Immediately
    # before any stack worker starts, re-observe and re-validate the ALREADY
    # preflight-admitted final candidate/mapping/bundle. A correction may only
    # be a scheduler-produced shrink step on the frozen identity. Exhaustion here
    # is fatal: the gauge may have run, but zero stack calls and no committed
    # science/manifest output occur.
    # ------------------------------------------------------------------
    if scheduler_mode == zsched.MODE_ACTIVE and _r55_active_decision is not None:
        try:
            (
                _r55_active_bundle,
                _r55_active_candidate,
                _r55_active_mapping,
                _r55_active_launch_adm,
                _r55_active_launch_trail,
            ) = _r55_active_revalidate_launch(
                bundle=_r55_active_bundle,
                candidate=_r55_active_candidate,
                mapping=_r55_active_mapping,
                decision=_r55_active_decision,
            )
        except (zsched.SchedulerError, rcontract.RamContractError) as _r55_exc:
            _emit(
                "ZeGrid: R55-D2 active LAUNCH REVALIDATION REFUSED before the "
                f"stack — {_r55_exc}",
                lvl="ERROR", callback=progress_callback,
            )
            raise RuntimeError(
                "R55-D2 active launch revalidation refused before the stack: "
                + str(_r55_exc)
            ) from _r55_exc
        _r55_active_adm = _r55_active_launch_adm
        _r55_active_trail = (
            list(_r55_active_preflight_trail)
            + list(_r55_active_launch_trail)
        )
        _final = _r55_active_bundle
        # Re-adopt the freshly revalidated bundle as the ONLY execution source.
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
        _emit(
            "ZeGrid: R55-D2 active launch revalidation ADMITTED — "
            f"tile={_r55_active_mapping['cpu_streaming_tile']} "
            f"units_in_flight={_r55_active_mapping['units_in_flight']} "
            f"workers_per_unit={_r55_active_mapping['workers_per_unit']} "
            f"launch_replans={_r55_active_launch_adm.replans} "
            f"aggregate={_final.envelope.aggregate_bytes / 2**20:.1f} MiB",
            callback=progress_callback,
        )

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
                        tasks=None, workers=None, cache_workers=None,
                        progress_callback=None, heartbeat=None,
                        heartbeat_interval_s=zrt.RUN_MAX_SAMPLE_INTERVAL_S):
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
        _progress = (
            _cell_progress_for(_tasks)
            if progress_callback is None else progress_callback
        )
        # ZM-ZEGRID-R55-E11: pmap's SERIAL path calls the worker directly and
        # never runs the initializer. After a GPU attempt the module globals hold
        # the GPU science config, so a single-worker CPU fallback would silently
        # re-request the GPU (and fail). Publish the batch's OWN config in the
        # parent too; parallel children are still initialised by pmap exactly as
        # before, so this changes no parallel behaviour or science.
        _init_stack_worker(
            descs, canvas, science_cfg, global_gauge, global_frame_ids,
            global_reference_frame_id, tile_size, _cache_workers,
            _r40_coords_by_cell,
        )
        _res = zpar.pmap(
            _stack_cell, _tasks, workers=_workers,
            progress_callback=_progress, emit=_emit_live, meta=_meta,
            initializer=_init_stack_worker,
            initargs=_initargs,
            serial_fallback=not gpu_attempt,
            heartbeat=heartbeat,
            heartbeat_interval_s=heartbeat_interval_s,
        )
        return _res, _meta

    def _run_r55_active_cpu_waves(*, science_cfg, start_bundle,
                                  start_candidate, start_mapping):
        """E1: deterministic bounded waves + a fresh checkpoint before EACH.

        Only the active CPU path reaches this closure. Legacy/observe keep the
        whole-batch executor. Canonical task/result order is preserved because
        waves are contiguous slices of the canonical task list.
        """
        disk_root = _r55_run_disk_root(cache_root, output_dir)
        bundle = start_bundle
        candidate = start_candidate
        mapping = start_mapping
        # ZM-ZEGRID-ACTIVE-OOC-FALLBACK: the scheduler decision may be REPLACED
        # once by a bounded recovery; keep it local to the RUN closure so no
        # child/parent global is consulted.
        run_decision = _r55_active_decision
        all_tasks = list(start_bundle.cell_tasks)
        total_tasks = len(all_tasks)
        canonical_cids = [t[0].cell_id for t in all_tasks]
        offset = 0
        wave_index = 0
        results_local = []
        metas = []
        wave_records = []
        wave_telemetry = []
        checkpoints = []
        deterministic_checkpoints = []
        shrink_count = 0
        run_t0 = time.perf_counter()
        # ONE persistent latch for the whole RUN. A mid-wave sample is only a
        # candidate for the NEXT safe boundary; it never mutates this wave.
        latch = zrt.PressureLatch()

        def _bounded_run_recovery(*, wave_index, offset, remaining_units, record,
                                  governing_fact, fresh_facts,
                                  refused_candidate, refused_mapping):
            """ONE safe-boundary dense->bounded recovery; None when not allowed.

            Only a QUALIFIED host-RAM checkpoint refusal can trigger it. The
            EXACT geometry/halo/memberships/frames/science identity are pinned
            (the gauge already ran), completed waves are kept, and every not-yet
            started cell task is rebuilt with the bounded provider/sink profile.
            The GOVERNING observation of the refused boundary (min of the valid
            latched sample and the fresh sample) is carried through the whole
            retry: a recovered machine never buys a bigger wave budget.
            """
            nonlocal run_decision
            if int(remaining_units) <= 0:
                return None
            if str(record.get("reason_code")) != renv.REASON_REFUSE_HOST_RAM:
                return None
            # ZM-ZEGRID-ACTIVE-OOC-FALLBACK rework-1 (HIGH): recover ONLY from a
            # VALID measured capacity shortage. A malformed/failed/missing
            # observation, a disk miss or an internal/config error is fail-CLOSED
            # and must never be relabelled into a bounded retry.
            if record.get("storage_fallback_eligible") is not True:
                return None
            if "last_candidate" not in record:
                return None
            if not _storage_fallback_available():
                return None
            # ---- F4: the governing fact survives the SAME boundary retry ----
            _governing_raw = int(record["host"]["raw_available_bytes"])
            _fresh_raw = int(fresh_facts["host"]["raw_available_bytes"])
            _raw = min(_governing_raw, _fresh_raw)
            if governing_fact is not None:
                _latched_raw = int(governing_fact["raw_available_bytes"])
                _raw = min(_raw, _latched_raw)
            _budget = int(rcontract.plan_budget_bytes(_raw))
            if not _activate_bounded_storage(
                phase=f"run_wave_{int(wave_index)}",
                reason_code=renv.REASON_REFUSE_HOST_RAM,
                detail=str(record.get("detail", "")), emit=_emit_live,
            ):
                return None
            _fixed_layout = tuple(int(v) for v in bundle.envelope.layout)
            _fixed_rank = int(bundle.envelope.refinement_rank)
            try:
                _seed = _build_execution_bundle(
                    canvas, descs, bundle.footprints,
                    raw_available_bytes=_raw, budget_bytes=_budget,
                    pinned_layout=_fixed_layout,
                    min_refinement_rank=_fixed_rank,
                    max_refinement_rank=_fixed_rank,
                    layout_substep=None,
                    **{**_bundle_common, "gpu_used": False},
                )
                _dec = _r55_active_decide(
                    bundle=_seed, cell_ctxs=_seed.cell_ctxs, canvas=canvas,
                    halo_px=halo_px, cache_root=cache_root,
                    output_dir=output_dir, frame_ids=_r55_frame_ids,
                )
                (
                    _bnd, _cand, _map, _adm2, _trail2,
                ) = _r55_active_admit(
                    initial_bundle=_seed, decision=_dec,
                )
            except (
                zal.LayoutInfeasible, zpar.ModeInfeasible,
                zsched.SchedulerError, rcontract.RamContractError,
            ) as _rec_exc:
                _emit(
                    "ZeGrid: bounded RUN recovery is NOT feasible on the SAME "
                    f"geometry — keeping the honest refusal: {_rec_exc}",
                    lvl="ERROR", callback=progress_callback,
                )
                return None
            if tuple(int(v) for v in _bnd.envelope.layout) != _fixed_layout:
                _emit(
                    "ZeGrid: bounded RUN recovery refused an inexact geometry "
                    f"({_bnd.envelope.layout} != {_fixed_layout}); keeping the "
                    "honest refusal",
                    lvl="ERROR", callback=progress_callback,
                )
                return None
            # F4: the bounded rebuild must be a SHRINK of the refused candidate
            # (tile/K/workers/prefetch/backend), never a growth bought with a
            # recovered observation.
            try:
                zsched.assert_shrink_only(refused_candidate, _cand)
            except zsched.SchedulerError as _grow_exc:
                _emit(
                    "ZeGrid: bounded RUN recovery refused a NON-shrink rebuild "
                    f"({_grow_exc}); keeping the honest refusal",
                    lvl="ERROR", callback=progress_callback,
                )
                return None
            for _key in ("units_in_flight", "workers_per_unit", "prefetch_units"):
                if int(_map[_key]) > int(refused_mapping[_key]):
                    _emit(
                        "ZeGrid: bounded RUN recovery refused a grown mapping "
                        f"({_key}: {_map[_key]} > {refused_mapping[_key]}); "
                        "keeping the honest refusal",
                        lvl="ERROR", callback=progress_callback,
                    )
                    return None
            _old_tile = refused_mapping.get("cpu_streaming_tile")
            _new_tile = _map.get("cpu_streaming_tile")
            if (
                _old_tile is not None and _new_tile is not None
                and int(_new_tile) > int(_old_tile)
            ):
                _emit(
                    "ZeGrid: bounded RUN recovery refused a grown tile "
                    f"({_new_tile} > {_old_tile}); keeping the honest refusal",
                    lvl="ERROR", callback=progress_callback,
                )
                return None
            _new_tasks = list(_bnd.cell_tasks)
            if len(_new_tasks) != len(all_tasks):
                _emit(
                    "ZeGrid: bounded RUN recovery produced a different cell "
                    "set; keeping the honest refusal",
                    lvl="ERROR", callback=progress_callback,
                )
                return None
            for _a, _b in zip(all_tasks[:offset], _new_tasks[:offset]):
                if _a[0].cell_id != _b[0].cell_id:
                    _emit(
                        "ZeGrid: bounded RUN recovery changed the completed "
                        "cell prefix; keeping the honest refusal",
                        lvl="ERROR", callback=progress_callback,
                    )
                    return None
            run_decision = _dec
            _emit(
                "ZeGrid: storage profile AUTO switched to 'bounded' at the "
                f"wave {int(wave_index)} safe boundary (completed waves kept, "
                f"remaining units={int(remaining_units)}, geometry="
                f"{_fixed_layout[0]}x{_fixed_layout[1]} unchanged); continuing "
                "with the bounded provider/sink profile",
                lvl="WARN", callback=progress_callback,
            )
            return {
                "bundle": _bnd,
                "candidate": _cand,
                "mapping": _map,
                "tasks": _new_tasks,
                "admission": _adm2,
                "trail": list(_trail2),
                "governing_raw": int(_raw),
            }

        def _heartbeat():
            # Any collection failure is an admission-fact failure: latch it as
            # fatal for the next safe boundary AND re-raise so the pmap
            # heartbeat wrapper records an honest error counter. A malformed
            # returned fact is latched fatal by record() and also reported as
            # a failed callback (no double fatal record, no lost latch).
            try:
                sample = _r55_sample_pressure(disk_root)
            except Exception as exc:  # noqa: BLE001 - fatal at the boundary
                latch.record_error(exc)
                raise
            if not latch.record(sample):
                raise zrt.RuntimeControlError(
                    "malformed in-wave RUN pressure fact (latched fatal)"
                )

        def _wave_progress(base_offset):
            def _cb(done, _total):
                idx = int(base_offset) + int(done)
                cid = (
                    canonical_cids[idx - 1]
                    if 0 < idx <= len(canonical_cids) else None
                )
                _stack_rep.progress(idx, item_id=cid)
            return _cb

        def _terminalize_refusal(exc, *, wave_index, offset,
                                 fallback_exhausted=False):
            """Record/clean the terminal refusal of the CURRENT boundary.

            Two distinct paths:

            * ``fallback_exhausted=False`` — no recovery was permitted (or the
              recovery refused before a bounded retry).  The DENSE refusal is
              terminalized verbatim, keeping its structured reason/eligibility.
            * ``fallback_exhausted=True`` — the bounded retry of the SAME
              boundary refused.  The bounded refusal is terminalized, but the
              single allowed profile fallback is already spent, so its
              ``storage_fallback_eligible`` is forced to False and
              ``fallback_exhausted=True`` is published.  This prevents a log
              reader from assuming a second fallback remains available.
            """
            if fallback_exhausted:
                exc.storage_fallback_eligible = False
                exc.record["storage_fallback_eligible"] = False
                exc.record["fallback_exhausted"] = True
            cleanup = []
            for task in all_tasks[:offset]:
                if task[4] is None:
                    continue
                cache_dir = Path(task[3])
                existed = bool(cache_dir.exists())
                removed = bool(_safe_rmtree(task[3], None))
                cleanup.append(
                    {
                        "cell_id": str(task[0].cell_id),
                        "existed": bool(existed),
                        "removed": bool(removed),
                    }
                )
            exc.record["cleanup"] = cleanup
            exc.record["waves_completed"] = int(wave_index)
            exc.record["publication"] = {
                "fits": False,
                "manifest": False,
                "assembly": False,
            }
            exc.record["telemetry"] = zrt.build_run_telemetry(
                waves=wave_telemetry,
                checkpoints=checkpoints,
                shrink_count=shrink_count,
                refusal=exc.record,
                parent_end=zrt.read_parent_memory(),
            )
            _emit(
                "ZeGrid: R55-E1 RUN REFUSED before wave "
                f"{wave_index} — {exc.reason_code}: {exc.detail}. No FITS, "
                "manifest or science publication; completed per-cell temp "
                f"outputs cleaned best-effort ({len(cleanup)} task(s)).",
                lvl="ERROR", callback=progress_callback,
            )

        while offset < total_tasks:
            remaining_units = sum(
                1 for t in all_tasks[offset:] if t[4] is not None
            )
            facts = _r55_run_checkpoint_sample(disk_root)
            latched_fact = None
            try:
                try:
                    latched_fact = latch.consume_validated()
                except zrt.RuntimeControlError as exc:
                    raise _R55RunRefused(
                        renv.REASON_REFUSE_HOST_RAM,
                        f"malformed latched RUN fact: {exc}",
                        record={
                            "schema": zrt.RUN_CONTROL_SCHEMA,
                            "status": "refused",
                            "reason_code": renv.REASON_REFUSE_HOST_RAM,
                            "detail": (
                                f"malformed latched RUN fact: {exc}"
                            ),
                            "wave_index": int(wave_index),
                            "remaining_units": int(remaining_units),
                            "shrink_steps": [],
                            "unavailable_shrinks": [
                                dict(x)
                                for x in _R55E1_UNAVAILABLE_SHRINKS
                            ],
                        },
                    ) from exc
                checkpoint = _r55_run_checkpoint(
                    decision=run_decision,
                    candidate=candidate,
                    mapping=mapping,
                    bundle=bundle,
                    remaining_units=remaining_units,
                    wave_index=wave_index,
                    facts=facts,
                    latched=latched_fact,
                )
            except _R55RunRefused as exc:
                # ZM-ZEGRID-ACTIVE-OOC-FALLBACK rework-2 (F4): a bounded retry of
                # the SAME boundary carries the GOVERNING observation (min of the
                # valid latched sample and the fresh sample); it never re-samples
                # nor re-consumes the latch, and completed waves are kept.
                _recovery = _bounded_run_recovery(
                    wave_index=wave_index,
                    offset=offset,
                    remaining_units=remaining_units,
                    record=exc.record,
                    governing_fact=latched_fact,
                    fresh_facts=facts,
                    refused_candidate=candidate,
                    refused_mapping=mapping,
                )
                if _recovery is None:
                    _terminalize_refusal(
                        exc, wave_index=wave_index, offset=offset
                    )
                    raise
                bundle = _recovery["bundle"]
                candidate = _recovery["candidate"]
                mapping = _recovery["mapping"]
                all_tasks = _recovery["tasks"]
                total_tasks = len(all_tasks)
                canonical_cids = [t[0].cell_id for t in all_tasks]
                checkpoints.append({
                    "schema": zrt.RUN_CONTROL_SCHEMA,
                    "stage": _R55E1_RUN_STAGE,
                    "wave_index": int(wave_index),
                    "status": "storage_fallback",
                    "remaining_units": int(remaining_units),
                    "reason_code": renv.REASON_REFUSE_HOST_RAM,
                    "detail": (
                        "dense RUN checkpoint refused; ONE bounded retry under "
                        "the SAME governing observation (latched/min)"
                    ),
                    "storage_profile": zsp.STORAGE_PROFILE_BOUNDED,
                    "governing_raw_available_bytes": int(
                        _recovery["governing_raw"]
                    ),
                    "latched_governed": bool(latched_fact is not None),
                    "shrink_steps": [],
                })
                try:
                    checkpoint = _r55_run_checkpoint(
                        decision=run_decision,
                        candidate=candidate,
                        mapping=mapping,
                        bundle=bundle,
                        remaining_units=remaining_units,
                        wave_index=wave_index,
                        facts=facts,
                        latched=latched_fact,
                    )
                except _R55RunRefused as exc2:
                    _terminalize_refusal(
                        exc2, wave_index=wave_index, offset=offset,
                        fallback_exhausted=True,
                    )
                    raise

            bundle = checkpoint["bundle"]
            candidate = checkpoint["candidate"]
            mapping = checkpoint["mapping"]
            record = checkpoint["record"]
            checkpoints.append(record)
            deterministic_checkpoints.append(
                checkpoint["deterministic_record"]
            )
            if record["changed"]:
                shrink_count += sum(
                    1 for step in record["shrink_steps"] if step["accepted"]
                )

            units_bound = int(mapping["units_in_flight"])
            start, end = zrt.next_wave_bound(
                offset, total_tasks, units_bound
            )
            wave_tasks = bundle.cell_tasks[start:end]
            wave_units = int(len(wave_tasks))
            wave_t0 = time.perf_counter()
            wave_results, wave_meta = _run_cell_batch(
                science_cfg,
                int(bundle.cpu_streaming_tile),
                gpu_attempt=False,
                tasks=wave_tasks,
                workers=units_bound,
                cache_workers=int(bundle.cache_workers_per_cell),
                progress_callback=_wave_progress(start),
                heartbeat=_heartbeat,
                heartbeat_interval_s=zrt.RUN_MAX_SAMPLE_INTERVAL_S,
            )
            wave_wall = time.perf_counter() - wave_t0
            results_local.extend(wave_results)
            metas.append(wave_meta)
            remaining_after = sum(
                1 for t in all_tasks[end:] if t[4] is not None
            )
            heartbeat_dict = {
                key: wave_meta.get(key)
                for key in (
                    "heartbeat_mode",
                    "heartbeat_calls",
                    "heartbeat_errors",
                    "heartbeat_last_error",
                    "heartbeat_interval_s",
                )
                if key in wave_meta
            }
            wave_telemetry.append(
                zrt.build_wave_record(
                    index=wave_index,
                    offset=start,
                    wave_units=wave_units,
                    total_tasks=total_tasks,
                    remaining_units_after=remaining_after,
                    candidate=record["candidate"],
                    mapping=record["mapping"],
                    facts={
                        "host": record["host"],
                        "fresh_host": record["fresh_host"],
                        "disk": facts["disk"],
                        "remaining_units": remaining_units,
                    },
                    shrink_steps=record["shrink_steps"],
                    decision_latched=record.get("latched_host"),
                    pending_latched=latch.snapshot(),
                    results=wave_results,
                    wall_s=wave_wall,
                    parent=facts.get("parent"),
                    cgroup=facts.get("cgroup"),
                    heartbeat=heartbeat_dict,
                )
            )
            wave_records.append(
                {
                    "index": int(wave_index),
                    "offset": int(start),
                    "wave_units": int(wave_units),
                    "units_remaining_after": int(remaining_after),
                    "changed": bool(record["changed"]),
                    "host_source": str(record.get("host_source", "fresh")),
                    "candidate": dict(record["candidate"]),
                    "mapping": dict(record["mapping"]),
                }
            )
            offset = end
            wave_index += 1

        # TERMINAL safe boundary: the final wave may have latched a malformed
        # admission fact (or a collector failure) with no next wave. Never
        # silently drop it before assembly/publication. A valid low-pressure
        # terminal sample is recorded conservatively; we do NOT invent a
        # post-execution shrink or rewrite the final mapping.
        terminal_fact = None
        try:
            terminal_fact = latch.consume_validated()
        except zrt.RuntimeControlError as exc:
            cleanup = []
            for task in all_tasks:
                if task[4] is None:
                    continue
                cache_dir = Path(task[3])
                existed = bool(cache_dir.exists())
                removed = bool(_safe_rmtree(task[3], None))
                cleanup.append(
                    {
                        "cell_id": str(task[0].cell_id),
                        "existed": bool(existed),
                        "removed": bool(removed),
                    }
                )
            detail = (
                "malformed pending in-wave fact at terminal boundary: "
                f"{exc}"
            )
            record = {
                "schema": zrt.RUN_CONTROL_SCHEMA,
                "status": "refused",
                "reason_code": renv.REASON_REFUSE_HOST_RAM,
                "detail": detail,
                "wave_index": int(wave_index),
                "remaining_units": 0,
                "shrink_steps": [],
                "unavailable_shrinks": [
                    dict(x) for x in _R55E1_UNAVAILABLE_SHRINKS
                ],
                "cleanup": cleanup,
                "waves_completed": int(wave_index),
                "publication": {
                    "fits": False,
                    "manifest": False,
                    "assembly": False,
                },
            }
            record["telemetry"] = zrt.build_run_telemetry(
                waves=wave_telemetry,
                checkpoints=checkpoints,
                shrink_count=shrink_count,
                refusal=record,
                parent_end=zrt.read_parent_memory(),
            )
            _emit(
                "ZeGrid: R55-E1 RUN REFUSED at the terminal safe boundary — "
                f"{detail}. No FITS, manifest or science publication.",
                lvl="ERROR", callback=progress_callback,
            )
            raise _R55RunRefused(
                renv.REASON_REFUSE_HOST_RAM, detail, record=record
            ) from exc
        terminal_info = None
        if terminal_fact is not None:
            terminal_info = {
                "latched": dict(terminal_fact),
                "pressure_observed": True,
                "action": (
                    "recorded_only_no_post_execution_shrink_or_mapping_rewrite"
                ),
            }

        total_run_wall = time.perf_counter() - run_t0
        meta = _r55_merge_wave_meta(
            metas, total_tasks=total_tasks, wave_count=wave_index
        )
        meta["seconds"] = float(total_run_wall)
        meta["seconds_per_unit"] = float(total_run_wall) / max(1, total_tasks)
        run_block = {
            "schema": zrt.RUN_DECISION_SCHEMA,
            "mission": _R55E1_RUN_MISSION,
            "policy": {
                "wave_partition": "canonical_contiguous_slices",
                "wave_bound": "units_in_flight",
                "checkpoint": "mandatory_before_every_wave",
                "run_budget": {
                    "factor_num": int(rcontract.ADMIT_FACTOR_NUM),
                    "factor_den": int(rcontract.ADMIT_FACTOR_DEN),
                },
                "shrink_order": [
                    "prefetch_units",
                    "workers_per_unit",
                    "units_in_flight",
                    "tile_px",
                    "backend",
                ],
                "prefetch_units": 0,
                "backend": "cpu",
                "recovery_growth": "forbidden",
                "latched_pressure": (
                    "consumed_once_at_next_boundary_min_combined"
                ),
                "in_wave_monitor": (
                    "parent_side_heartbeat_no_thread_before_fork"
                ),
                "in_wave_sampler_max_interval_s": float(
                    zrt.RUN_MAX_SAMPLE_INTERVAL_S
                ),
            },
            "checkpoints": deterministic_checkpoints,
            "waves": wave_records,
            "final_execution": _r55_active_execution(bundle, mapping),
            "shrink_count": int(shrink_count),
            "examined_shrink_steps": int(
                sum(len(cp["shrink_steps"]) for cp in deterministic_checkpoints)
            ),
            "refusal_count": 0,
            "note": (
                "Deterministic RUN decision structure only. Exact volatile "
                "observations (wall/RSS/cgroup/IO/disk free) live in "
                "r55_run_telemetry and never enter a deterministic plan hash."
            ),
        }
        telemetry = zrt.build_run_telemetry(
            waves=wave_telemetry,
            checkpoints=checkpoints,
            shrink_count=shrink_count,
            refusal=None,
            parent_end=zrt.read_parent_memory(),
            deterministic_ref=zrt.sha256_payload(run_block),
            terminal=terminal_info,
        )
        return (
            results_local,
            meta,
            run_block,
            telemetry,
            bundle,
            candidate,
            mapping,
            shrink_count,
            wave_index,
        )

    def _run_r55_active_gpu_units(
        *, gpu_bundle, gpu_mapping, gpu_decision, gpu_candidate=None
    ):
        """R55-E2: bounded CPU->GPU producer/consumer pipeline (single owner).

        Producer: CPU cache build + cell gauge, at most ``prefetch_units + 1``
        staged units. Consumer: the ONE GPU owner. Fresh 90 % host + VRAM
        revalidation precedes every consumed unit. Any pressure/OOM/error
        cancels, releases every staged buffer and raises a qualified
        :class:`_R55GpuFallback` (the caller owns the ONE whole-batch CPU retry).
        """
        tile = int(gpu_mapping["gpu_streaming_tile"])
        workers = int(gpu_mapping["workers_per_unit"])
        prefetch = int(gpu_mapping["prefetch_units"])
        queue_max = int(prefetch + 1)
        device_peak = int(gpu_decision.device_peak_bytes_by_tile[tile])
        device_budget = int(gpu_decision.qualification["budget_bytes"])
        vram_percent = int(gpu_decision.qualification["percent"])
        per_unit_cache = int(gpu_decision.facts["per_unit_cache"])
        unit_host = max(1, int(per_unit_cache))
        host_budget = min(
            int(gpu_bundle.envelope.budget_bytes),
            int(unit_host) * int(queue_max),
        )
        if int(host_budget) < int(unit_host):
            raise _R55GpuFallback(
                renv.REASON_REFUSE_HOST_RAM,
                "the ACTIVE GPU pipeline cannot stage even one unit within the "
                f"fresh 90 % host budget ({host_budget} B < {unit_host} B)",
            )
        gpu_cfg = replace(science_config, backend="gpu")
        _init_stack_worker(
            descs, canvas, gpu_cfg, global_gauge, global_frame_ids,
            global_reference_frame_id, tile, workers,
        )
        disk_root = _r55_run_disk_root(cache_root, output_dir)
        revalidations = []
        run_checkpoints = []
        deterministic_checkpoints = []
        revalid_lock = threading.Lock()
        # R55-E11 addendum: capture the ORIGINAL exception at raise time (before
        # the pipeline's cleanup unwinds) so the durable record is not a
        # post-hoc generic wrapper.
        failure_lock = threading.Lock()
        failure_capture: dict = {}

        def _capture_failure(stage, exc, *, payload=None, task=None):
            try:
                with failure_lock:
                    if failure_capture:
                        return
                    unit = None
                    if isinstance(payload, dict):
                        unit = {
                            "unit_index": payload.get("_unit_index"),
                            "cell_id": str(
                                getattr(payload.get("cell"), "cell_id", "")
                            ),
                            "tile_px": payload.get("_stack_tile_size"),
                        }
                    elif task is not None:
                        try:
                            unit = {
                                "producer_task_index": task[0],
                                "cell_id": str(getattr(task[0], "cell_id", "")),
                            }
                        except Exception:
                            unit = None
                    failure_capture.update(
                        {
                            "stage": str(stage),
                            "exception": exc,
                            "unit": unit,
                            "captured_at": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }
                    )
            except Exception:
                # Diagnostics capture must NEVER interfere with the failure path.
                pass

        # R55-E11: ONE mutable tile state shared by the producer's credit callback
        # and the single-owner consumer. The consumer is the only writer; the
        # producer only reads the CURRENT peak when it stages a unit, so a shrink
        # can never race a credit acquisition. The pipeline's device budget stays
        # the admitted 70 % qualification budget (a hard, never-raised ceiling);
        # each unit's lease records the exact peak it was staged for, and each
        # unit's execution records the tile it ACTUALLY ran with.
        tile_state = {
            "tile": int(tile),
            "device_peak": int(device_peak),
            "mapping": dict(gpu_mapping),
            "bundle": gpu_bundle,
            "shrink_history": [],
            "units": [],
            "attempts": [],
            "last_host_facts": None,
        }
        tile_lock = threading.Lock()
        initial_tile = int(tile)
        initial_device_peak = int(device_peak)

        def _current_tile() -> int:
            with tile_lock:
                return int(tile_state["tile"])

        def _current_peak() -> int:
            with tile_lock:
                return int(tile_state["device_peak"])

        def _produce(task):
            try:
                prepared = _prepare_stack_cell(task)
            except BaseException as exc:
                _capture_failure("producer", exc, task=task)
                raise
            if "empty_result" in prepared:
                prepared["_empty_unit"] = True
            return prepared

        def _host_bytes(payload):
            if payload.get("_empty_unit"):
                return 0
            return int(payload["manifest"].get("total_bytes", 0))

        def _device_bytes(payload):
            if payload.get("_empty_unit"):
                return 0
            # The lease follows the tile current at STAGING time. A later shrink
            # only ever lowers the executing peak, so a lease is a conservative
            # upper bound for the unit it covers (never an oversubscription).
            return int(_current_peak())

        def _release(payload):
            if payload.get("_empty_unit"):
                return
            cache_dir = payload.get("cache_dir")
            if cache_dir is not None:
                _safe_rmtree(cache_dir, None)

        def _consume_guarded(payload):
            try:
                return _consume(payload)
            except BaseException as exc:
                _capture_failure("consumer", exc, payload=payload)
                raise

        def _consume(payload):
            unit_index = int(payload.get("_unit_index", 0))
            if payload.get("_empty_unit"):
                return _consume_stack_cell(payload)

            # --- R55-E11 inter-cell boundary -------------------------------
            # The previous unit's canonical stack has already purged its own
            # device temporaries at the cell boundary (bounded helper scope).
            # Synchronise + purge again here so consumer-side allocations from
            # the crop/pack step cannot masquerade as external pressure, THEN
            # read fresh VRAM. Snapshot, never a claimed high-water mark.
            boundary = zgm.gpu_cell_boundary_cleanup(
                reason="inter_cell_boundary"
            )
            facts = _r55_run_checkpoint_sample(disk_root)
            fresh = zrt.validate_checkpoint_facts(facts)
            raw = int(fresh["host"]["raw_available_bytes"])
            budget = int(fresh["host"]["budget_bytes"])
            host_facts = {
                "raw_available_bytes": int(raw),
                "budget_bytes": int(budget),
                "factor_num": int(rcontract.ADMIT_FACTOR_NUM),
                "factor_den": int(rcontract.ADMIT_FACTOR_DEN),
            }
            try:
                free_vram = int(zgr.read_vram_free_bytes(0))
            except zgr.InvalidVramObservation as exc:
                refused = {
                    "schema": zrt.RUN_CONTROL_SCHEMA,
                    "stage": "run",
                    "unit_index": int(unit_index),
                    "status": "refused",
                    "action": "fallback_cpu",
                    "attempted": True,
                    "completed": False,
                    "discarded": True,
                    "tile_px": int(_current_tile()),
                    "initial_tile_px": int(initial_tile),
                    "reason_code": str(zgr.REASON_GPU_OBSERVATION_FAILED),
                    "detail": f"fresh VRAM observation failed: {exc}",
                    "host": dict(host_facts),
                    "gpu_memory": {
                        "boundary": boundary,
                        "label": zgm.GPU_MEMORY_SNAPSHOT_LABEL,
                    },
                }
                tile_state["units"].append(refused)
                with revalid_lock:
                    run_checkpoints.append(refused)
                raise _R55GpuFallback(
                    zgr.REASON_GPU_OBSERVATION_FAILED,
                    f"fresh VRAM observation failed: {exc}",
                    telemetry={
                        "runtime_tile": _r55_tile_adaptation_from_state(
                            tile_state, gpu_decision, initial_tile
                        )
                    },
                ) from exc

            tile_now = _current_tile()
            check = _r55_runtime_tile_check(
                gpu_decision,
                tile=int(tile_now),
                free_bytes=int(free_vram),
                host_budget_bytes=int(budget),
                bundle=gpu_bundle,
            )
            shrink_attempts: list[dict] = []
            action = "continue"
            if not check["eligible"]:
                # RUN-time pressure: walk the SAME admitted ladder strictly down.
                shrink_attempts, chosen = _r55_runtime_tile_walk(
                    gpu_decision,
                    current_tile=int(tile_now),
                    free_bytes=int(free_vram),
                    host_budget_bytes=int(budget),
                    bundle=gpu_bundle,
                )
                if chosen is None:
                    refused = {
                        "schema": zrt.RUN_CONTROL_SCHEMA,
                        "stage": "run",
                        "unit_index": int(unit_index),
                        "status": "refused",
                        "action": "fallback_cpu",
                        "attempted": True,
                        "completed": False,
                        "discarded": True,
                        "tile_px": int(tile_now),
                        "initial_tile_px": int(initial_tile),
                        "reason_code": (
                            str(check["vram_reason_code"])
                            if not check["device_ok"]
                            else str(renv.REASON_REFUSE_HOST_RAM)
                        ),
                        "detail": (
                            "no smaller declared GPU tile is admissible on the "
                            "fresh facts"
                        ),
                        "host": dict(host_facts),
                        "vram": {
                            "free_bytes": int(free_vram),
                            "percent": int(vram_percent),
                            "budget_bytes": int(check["vram_budget_bytes"]),
                            "device_peak_bytes": int(
                                check["device_peak_bytes"]
                            ),
                            "action": "fallback_cpu",
                            "reason_code": str(check["vram_reason_code"]),
                        },
                        "aggregate_bytes": int(check["host_aggregate_bytes"]),
                        "shrink_attempts": [dict(a) for a in shrink_attempts],
                        "gpu_memory": {
                            "boundary": boundary,
                            "label": zgm.GPU_MEMORY_SNAPSHOT_LABEL,
                        },
                    }
                    tile_state["units"].append(refused)
                    with revalid_lock:
                        revalidations.append(refused)
                        run_checkpoints.append(refused)
                    _refuse_reason = (
                        str(check["vram_reason_code"])
                        if not check["device_ok"]
                        else str(renv.REASON_REFUSE_HOST_RAM)
                    )
                    with tile_lock:
                        tile_state["final_cause"] = str(_refuse_reason)
                    _refusal_diag = _r55_gpu_failure_record(
                        stage="checkpoint",
                        category=(
                            "vram_refusal"
                            if not check["device_ok"]
                            else "host_refusal"
                        ),
                        reason_code=_refuse_reason,
                        detail=(
                            "fresh VRAM/host revalidation refused the GPU "
                            "unit and no smaller declared GPU tile fits"
                        ),
                        exc=None,
                        pipeline=None,
                        tile_state=tile_state,
                        revalidations=revalidations,
                        unit={
                            "unit_index": int(unit_index),
                            "tile_px": int(tile_now),
                            "initial_tile_px": int(initial_tile),
                        },
                        host_facts=host_facts,
                        device_snapshot=boundary.get("after"),
                        extra={
                            "check": dict(check),
                            "shrink_attempts": [
                                dict(a) for a in shrink_attempts
                            ],
                        },
                    )
                    _refusal_path = _r55_write_gpu_failure_diagnostics(
                        output_dir, _refusal_diag
                    )
                    raise _R55GpuFallback(
                        _refuse_reason,
                        "fresh VRAM/host revalidation refused the GPU unit and "
                        "no smaller declared GPU tile fits "
                        f"(free={free_vram} B, tile={tile_now})",
                        telemetry={
                            "runtime_tile": _r55_tile_adaptation_from_state(
                                tile_state, gpu_decision, initial_tile
                            ),
                            "attempts": [dict(a) for a in shrink_attempts],
                            "diagnostics_path": _refusal_path,
                        },
                    )
                # Apply the shrink coherently: mapping -> bundle -> peak -> state.
                probe = replace(
                    gpu_decision.plan_set.chosen, tile_px=int(chosen)
                )
                materialized = _r55_gpu_candidate_lookup(gpu_decision, probe)
                mapping_new = zsched.active_gpu_mapping_for_candidate(
                    materialized,
                    tile_ladder=tuple(
                        int(t) for t in gpu_decision.gpu_context.tile_ladder
                    ),
                    workers_ladder=tuple(
                        int(w) for w in zsched.WORKERS_LADDER
                    ),
                    prefetch_cap=int(_R55E2_PREFETCH_LADDER[-1]),
                )
                bundle_new = _r55_active_gpu_apply(
                    gpu_bundle,
                    mapping=mapping_new,
                    decision=gpu_decision,
                    stage="runtime_tile_shrink",
                    plan_raw_available_bytes=int(raw),
                    plan_factor_num=int(rcontract.ADMIT_FACTOR_NUM),
                    plan_factor_den=int(rcontract.ADMIT_FACTOR_DEN),
                    plan_budget_bytes=int(budget),
                )
                with tile_lock:
                    tile_state["tile"] = int(chosen)
                    tile_state["device_peak"] = int(
                        gpu_decision.device_peak_bytes_by_tile[int(chosen)]
                    )
                    tile_state["mapping"] = dict(mapping_new)
                    tile_state["bundle"] = bundle_new
                    tile_state["attempts"].extend(
                        [dict(a) for a in shrink_attempts]
                    )
                    tile_state["shrink_history"].append(
                        {
                            "schema": _R55E11_TILE_RUNTIME_SCHEMA,
                            "unit_index": int(unit_index),
                            "from_tile_px": int(tile_now),
                            "to_tile_px": int(chosen),
                            "from_device_peak_bytes": int(
                                gpu_decision.device_peak_bytes_by_tile[
                                    int(tile_now)
                                ]
                            ),
                            "to_device_peak_bytes": int(
                                gpu_decision.device_peak_bytes_by_tile[
                                    int(chosen)
                                ]
                            ),
                            "vram_free_bytes": int(free_vram),
                            "vram_percent": int(vram_percent),
                            "host_budget_bytes": int(budget),
                            "host_aggregate_bytes": int(
                                _r55_gpu_host_aggregate_for_tile(
                                    gpu_decision, gpu_bundle, int(chosen)
                                )
                            ),
                            "attempts": [dict(a) for a in shrink_attempts],
                        }
                    )
                tile_now = int(chosen)
                action = "shrink"
                check = _r55_runtime_tile_check(
                    gpu_decision,
                    tile=int(tile_now),
                    free_bytes=int(free_vram),
                    host_budget_bytes=int(budget),
                    bundle=gpu_bundle,
                )

            peak_used = int(gpu_decision.device_peak_bytes_by_tile[int(tile_now)])
            reserved_peak = int(_device_bytes(payload))
            # Keep the per-cell record's bound truthful for the EXECUTED tile.
            try:
                payload["bound"] = int(
                    _estimate_streaming_bytes(
                        len(payload["mem"].patch_ids),
                        payload["patch_shape_hw"],
                        int(tile_now),
                    )
                )
            except Exception:
                pass
            record = {
                "schema": zrt.RUN_CONTROL_SCHEMA,
                "stage": "run",
                "unit_index": int(unit_index),
                "status": "executing",
                "action": str(action),
                "attempted": True,
                "completed": False,
                "discarded": False,
                "tile_px": int(tile_now),
                "initial_tile_px": int(initial_tile),
                "host": dict(host_facts),
                "vram": {
                    "free_bytes": int(free_vram),
                    "percent": int(vram_percent),
                    "budget_bytes": int(check["vram_budget_bytes"]),
                    "device_peak_bytes": int(peak_used),
                    "reserved_device_peak_bytes": int(reserved_peak),
                    "action": str(action),
                    "reason_code": str(check["vram_reason_code"]),
                    "device_ok": bool(check["device_ok"]),
                    "host_ok": bool(check["host_ok"]),
                },
                "aggregate_bytes": int(check["host_aggregate_bytes"]),
                "shrink_attempts": [dict(a) for a in shrink_attempts],
                "gpu_memory": {
                    "label": zgm.GPU_MEMORY_SNAPSHOT_LABEL,
                    "boundary": boundary,
                    "after_unit": None,
                },
            }
            with revalid_lock:
                revalidations.append(record)
                run_checkpoints.append(record)
                deterministic_checkpoints.append(
                    {
                        "stage": "run",
                        "unit_index": int(record["unit_index"]),
                        "host": dict(record["host"]),
                        "vram": {
                            "percent": int(vram_percent),
                            "device_peak_bytes": int(peak_used),
                            "reason_code": str(check["vram_reason_code"]),
                        },
                        "tile_px": int(tile_now),
                        "action": str(action),
                        "aggregate_bytes": int(record["aggregate_bytes"]),
                    }
                )
            t_unit = time.perf_counter()
            payload["_stack_tile_size"] = int(tile_now)
            try:
                result = _consume_stack_cell(payload)
            except BaseException:
                record["status"] = "failed"
                record["action"] = "fallback_cpu"
                record["discarded"] = True
                raise
            finally:
                # BEST-EFFORT boundary cleanup AFTER the unit, never masking the
                # original exception (the helper itself cannot raise).
                record["gpu_memory"]["after_unit"] = zgm.gpu_cell_boundary_cleanup(
                    reason="post_unit"
                )
            record["status"] = "completed"
            record["completed"] = True
            record["stack_seconds"] = float(time.perf_counter() - t_unit)
            with tile_lock:
                tile_state["units"].append(record)
                tile_state["last_host_facts"] = dict(host_facts)
            return result
            facts = _r55_run_checkpoint_sample(disk_root)
            fresh = zrt.validate_checkpoint_facts(facts)
            raw = int(fresh["host"]["raw_available_bytes"])
            budget = int(fresh["host"]["budget_bytes"])
            try:
                free_vram = int(
                    zgr.read_vram_free_bytes(0)
                )
            except zgr.InvalidVramObservation as exc:
                raise _R55GpuFallback(
                    zgr.REASON_GPU_OBSERVATION_FAILED,
                    f"fresh VRAM observation failed: {exc}",
                ) from exc
            vram_decision = zgr.fresh_vram_revalidation(
                policy=gpu_decision.policy,
                current_percent=vram_percent,
                free_bytes=free_vram,
                device_peak_bytes=device_peak,
            )
            record = {
                "schema": zrt.RUN_CONTROL_SCHEMA,
                "stage": "run",
                "unit_index": int(payload.get("_unit_index", 0)),
                "status": "accepted",
                "host": {
                    "raw_available_bytes": int(raw),
                    "budget_bytes": int(budget),
                    "factor_num": int(rcontract.ADMIT_FACTOR_NUM),
                    "factor_den": int(rcontract.ADMIT_FACTOR_DEN),
                },
                "vram": {
                    "free_bytes": int(free_vram),
                    "percent": int(vram_percent),
                    "device_peak_bytes": int(device_peak),
                    "action": str(vram_decision.action),
                    "reason_code": str(vram_decision.reason_code),
                },
                "aggregate_bytes": int(gpu_bundle.envelope.aggregate_bytes),
            }
            with revalid_lock:
                revalidations.append(record)
                run_checkpoints.append(record)
                deterministic_checkpoints.append(
                    {
                        "stage": "run",
                        "unit_index": int(record["unit_index"]),
                        "host": dict(record["host"]),
                        "vram": {
                            "percent": int(vram_percent),
                            "device_peak_bytes": int(device_peak),
                            "reason_code": str(vram_decision.reason_code),
                        },
                        "aggregate_bytes": int(record["aggregate_bytes"]),
                    }
                )
            if vram_decision.action == "fallback_cpu":
                raise _R55GpuFallback(
                    vram_decision.reason_code,
                    "fresh VRAM revalidation refused the GPU unit "
                    f"(free={free_vram} B, peak={device_peak} B)",
                )
            if int(gpu_bundle.envelope.aggregate_bytes) > int(budget):
                raise _R55GpuFallback(
                    renv.REASON_REFUSE_HOST_RAM,
                    "fresh 90 % RUN budget "
                    f"({budget} B) is below the admitted GPU aggregate "
                    f"({gpu_bundle.envelope.aggregate_bytes} B)",
                )
            return _consume_stack_cell(payload)

        # Per-unit index is needed by the consumer record without relying on
        # thread-shared mutable state. Wrap each canonical task once.
        tasks = list(gpu_bundle.cell_tasks)
        indexed = list(enumerate(tasks))
        host_budget_val = int(host_budget)
        pipeline = zpar.BoundedPipeline(
            tasks=indexed,
            produce=lambda item: dict(_produce(item[1]), _unit_index=int(item[0])),
            consume=_consume_guarded,
            release=_release,
            prefetch_units=prefetch,
            host_budget_bytes=int(host_budget_val),
            device_budget_bytes=int(max(device_budget, initial_device_peak)),
            host_bytes=_host_bytes,
            device_bytes=_device_bytes,
            thread_join_s=float(_R55E2_PIPELINE_JOIN_S),
            # No internal per-unit stall budget: honest liveness for a long
            # healthy GPU cell is "the caller is still inside consume". Only a
            # caller/process cancellation (or a producer that never progresses
            # and is explicitly watched) ends it early.
            stall_timeout_s=None,
        )
        t0 = time.perf_counter()
        try:
            results = pipeline.run()
        except _R55GpuFallback as _fallback_exc:
            with tile_lock:
                if tile_state.get("final_cause") in (None, "gpu_completed"):
                    tile_state["final_cause"] = str(
                        _fallback_exc.reason_code
                    )
                for _record in tile_state["units"]:
                    if _record.get("completed"):
                        _record["discarded"] = True
                    elif _record.get("status") == "executing":
                        _record["status"] = "failed"
                        _record["action"] = "fallback_cpu"
                        _record["discarded"] = True
            _fallback_pipeline = pipeline.telemetry()
            if str(_fallback_exc.reason_code) == str(zgr.REASON_REFUSE_VRAM):
                _fallback_category = "vram_refusal"
            elif str(_fallback_exc.reason_code) == str(
                renv.REASON_REFUSE_HOST_RAM
            ):
                _fallback_category = "host_refusal"
            else:
                _fallback_category = "gpu_fallback"
            _fallback_diag = _r55_gpu_failure_record(
                stage="pipeline",
                category=_fallback_category,
                reason_code=_fallback_exc.reason_code,
                detail=_fallback_exc.detail,
                exc=_fallback_exc,
                pipeline=_fallback_pipeline,
                tile_state=tile_state,
                revalidations=revalidations,
                unit={
                    "active_unit_index": _fallback_pipeline.get(
                        "active_unit_index"
                    ),
                    "producer_task_index": _fallback_pipeline.get(
                        "producer_task_index"
                    ),
                },
                host_facts=tile_state.get("last_host_facts"),
                device_snapshot=zgm.device_memory_snapshot(),
                parent_memory=zrt.read_parent_memory(),
            )
            _fallback_path = _r55_write_gpu_failure_diagnostics(
                output_dir, _fallback_diag
            )
            _fallback_exc.telemetry = {
                **_fallback_exc.telemetry,
                "pipeline": _fallback_pipeline,
                "revalidations": list(revalidations),
                "runtime_tile": _r55_tile_adaptation_from_state(
                    tile_state, gpu_decision, initial_tile
                ),
                "diagnostics_path": _fallback_path,
            }
            raise
        except BaseException as exc:  # noqa: BLE001 - exactly one CPU fallback
            # Deterministic cleanup: the pipeline already released its staged
            # units; also clear any per-cell cache left by a failed worker.
            for _t in tasks:
                if _t[4] is not None:
                    _safe_rmtree(_t[3], None)
            with tile_lock:
                for _record in tile_state["units"]:
                    if _record.get("completed"):
                        _record["discarded"] = True
                    elif _record.get("status") == "executing":
                        _record["status"] = "failed"
                        _record["action"] = "fallback_cpu"
                        _record["discarded"] = True
            _generic_reason = (
                "R55_GPU_RUNTIME"
                if zgpu.is_gpu_runtime_error(exc)
                else "R55_GPU_PIPELINE_FAILED"
            )
            with tile_lock:
                tile_state["final_cause"] = str(_generic_reason)
            _generic_pipeline = pipeline.telemetry()
            with failure_lock:
                _captured = dict(failure_capture)
            if _captured.get("stage") == "producer":
                _category = "producer_error"
            elif _captured.get("stage") == "consumer":
                _category = "consumer_error"
            elif _generic_pipeline.get("producer_error"):
                # The pipeline raised after produce (byte callback / credit /
                # accounting), i.e. the producer thread failed AFTER a payload
                # was owned. Distinct from a producer's own science exception.
                _category = "producer_accounting_error"
            elif _generic_pipeline.get("consumer_error"):
                _category = "consumer_error"
            else:
                _category = "pipeline_error"
            # The ORIGINAL exception captured at raise time is the primary
            # record. The pipeline's generic wrapper (when one exists) is kept
            # as ``surfaced_exception`` so both layers remain auditable.
            _primary_exc = _captured.get("exception") or exc
            _surfaced = None
            if _primary_exc is not exc:
                _surfaced = _r55_exception_chain(exc)
            _generic_diag = _r55_gpu_failure_record(
                stage="pipeline",
                category=_category,
                reason_code=_generic_reason,
                detail=f"{type(_primary_exc).__name__}: {_primary_exc}",
                exc=_primary_exc,
                pipeline=_generic_pipeline,
                tile_state=tile_state,
                revalidations=revalidations,
                unit=(
                    _captured.get("unit")
                    or {
                        "active_unit_index": _generic_pipeline.get(
                            "active_unit_index"
                        ),
                        "producer_task_index": _generic_pipeline.get(
                            "producer_task_index"
                        ),
                    }
                ),
                host_facts=tile_state.get("last_host_facts"),
                device_snapshot=zgm.device_memory_snapshot(),
                parent_memory=zrt.read_parent_memory(),
                extra={
                    "capture_stage": _captured.get("stage"),
                    "capture_timestamp": _captured.get("captured_at"),
                    "surfaced_exception": _surfaced,
                    "producer_error": _generic_pipeline.get(
                        "producer_error"
                    ),
                    "consumer_error": _generic_pipeline.get(
                        "consumer_error"
                    ),
                    "ledger": _generic_pipeline.get("ledger"),
                },
            )
            _generic_path = _r55_write_gpu_failure_diagnostics(
                output_dir, _generic_diag
            )
            raise _R55GpuFallback(
                _generic_reason,
                f"{type(_primary_exc).__name__}: {_primary_exc}",
                telemetry={
                    "pipeline": _generic_pipeline,
                    "revalidations": list(revalidations),
                    "runtime_tile": _r55_tile_adaptation_from_state(
                        tile_state, gpu_decision, initial_tile
                    ),
                    "diagnostics_path": _generic_path,
                },
            ) from exc
        wall = time.perf_counter() - t0
        pipeline_telemetry = pipeline.telemetry()
        with tile_lock:
            tile_state["final_cause"] = "gpu_completed"
        # --- R55-E11 effective execution -------------------------------------
        # The runner may have shrunk between cells. Rebuild the EXECUTABLE
        # bundle/mapping for the final tile so the manifest, envelope and
        # final_execution describe what actually ran (never the initial tile).
        effective_tile = int(_current_tile())
        effective_peak = int(_current_peak())
        with tile_lock:
            effective_mapping = dict(tile_state["mapping"])
            host_facts_last = (
                dict(tile_state["last_host_facts"])
                if tile_state["last_host_facts"] is not None
                else None
            )
            runtime_units = [dict(r) for r in tile_state["units"]]
        if effective_tile != initial_tile and host_facts_last is not None:
            effective_bundle = _r55_active_gpu_apply(
                gpu_bundle,
                mapping=effective_mapping,
                decision=gpu_decision,
                stage="runtime_effective",
                plan_raw_available_bytes=int(
                    host_facts_last["raw_available_bytes"]
                ),
                plan_factor_num=int(host_facts_last["factor_num"]),
                plan_factor_den=int(host_facts_last["factor_den"]),
                plan_budget_bytes=int(host_facts_last["budget_bytes"]),
            )
            effective_candidate = _r55_gpu_candidate_lookup(
                gpu_decision,
                replace(
                    gpu_decision.plan_set.chosen, tile_px=int(effective_tile)
                ),
            )
        else:
            effective_bundle = gpu_bundle
            effective_candidate = gpu_decision.plan_set.chosen
        runtime_tile = _r55_tile_adaptation_from_state(
            tile_state, gpu_decision, initial_tile
        )
        runtime_by_index = {
            int(r["unit_index"]): r
            for r in runtime_units
            if r.get("unit_index") is not None
        }
        units_block = []
        for index in range(len(results)):
            record = runtime_by_index.get(int(index))
            if record is None:
                units_block.append(
                    {
                        "index": int(index),
                        "offset": int(index),
                        "unit_units": 1,
                        "tile": None,
                        "status": "empty",
                        "action": "none",
                        "attempted": False,
                        "completed": False,
                        "discarded": False,
                        "prefetch_units": int(prefetch),
                        "device_peak_bytes": 0,
                    }
                )
                continue
            units_block.append(
                {
                    "index": int(index),
                    "offset": int(index),
                    "unit_units": 1,
                    "tile": record.get("tile_px"),
                    "status": str(record.get("status")),
                    "action": str(record.get("action")),
                    "attempted": bool(record.get("attempted")),
                    "completed": bool(record.get("completed")),
                    "discarded": bool(record.get("discarded")),
                    "prefetch_units": int(prefetch),
                    "device_peak_bytes": int(
                        (record.get("vram") or {}).get(
                            "device_peak_bytes", effective_peak
                        )
                    ),
                    "reserved_device_peak_bytes": int(
                        (record.get("vram") or {}).get(
                            "reserved_device_peak_bytes",
                            (record.get("vram") or {}).get(
                                "device_peak_bytes", effective_peak
                            ),
                        )
                    ),
                }
            )
        meta = {
            "executor": "gpu_pipeline_single_owner",
            "wave_executors": ["gpu_pipeline"],
            "parent_daemon": False,
            "workers": 1,
            "tasks": int(len(results)),
            "fallback": False,
            "fallback_reason": None,
            "seconds": float(wall),
            "seconds_per_unit": float(wall) / max(1, int(len(results))),
            "waves": 1,
            "heartbeat_calls": 0,
            "heartbeat_errors": 0,
            "heartbeat_modes": ["pipeline_boundary"],
            "heartbeat_interval_s": float(zrt.RUN_MAX_SAMPLE_INTERVAL_S),
            "pipeline": pipeline_telemetry,
        }
        run_block = {
            "schema": zrt.RUN_DECISION_SCHEMA,
            "mission": _R55E2_GPU_MISSION,
            "policy": {
                "backend": "gpu",
                "single_owner": True,
                "units_in_flight": 1,
                "prefetch_ladder": [int(p) for p in _R55E2_PREFETCH_LADDER],
                "prefetch_units": int(prefetch),
                "queue_max": int(queue_max),
                "vram_percent": int(vram_percent),
                "vram_budget_bytes": int(device_budget),
                "vram_budget_source": "active_r55_policy_70",
                "legacy_probe_vram_budget_bytes": (
                    None
                    if gpu_ctx.get("vram_budget_bytes") is None
                    else int(gpu_ctx["vram_budget_bytes"])
                ),
                "initial_tile_px": int(initial_tile),
                "effective_tile_px": int(effective_tile),
                "tile_growth": "forbidden",
                "disk": {
                    "residency_bytes": int(
                        per_unit_cache * (int(prefetch) + 1)
                    ),
                    "bytes_contract": _R55_CACHE_BYTES_CONTRACT,
                    "domain": (
                        "on_disk_aligned_cache_files_used_as_host_staging_credit"
                    ),
                    "note": (
                        "Exact RGB float32 + support bool + one 128-byte .npy "
                        "header per file; equal to manifest.total_bytes. Not a "
                        "resident-RAM model; never added to worker footprints."
                    ),
                    "free_bytes": int(
                        gpu_decision.facts.get("disk_free", 0)
                    ),
                    "strict_checkpoint": True,
                },
                "checkpoint": "mandatory_before_every_unit",
                "shrink_order": [
                    "prefetch_units",
                    "workers_per_unit",
                    "units_in_flight",
                    "tile_px",
                    "backend",
                ],
                "recovery_growth": "forbidden",
                "fallback": "whole_batch_cpu_once",
            },
            "checkpoints": deterministic_checkpoints,
            "units": units_block,
            "initial_execution": _r55_active_execution(
                gpu_bundle, gpu_mapping
            ),
            "final_execution": _r55_active_execution(
                effective_bundle, effective_mapping
            ),
            "tile_adaptation": runtime_tile,
            "engine_provenance": _r55_engine_provenance(),
            "shrink_count": int(len(_r55_active_trail)),
            "runtime_shrink_count": int(len(runtime_tile["shrink_history"])),
            "examined_shrink_steps": int(
                len(_r55_active_trail) + len(runtime_tile["attempts"])
            ),
            "refusal_count": 0,
            "note": (
                "Deterministic ACTIVE GPU RUN structure only. Wall/RSS/VRAM/"
                "queue observations live in r55_run_telemetry."
            ),
        }
        telemetry = zrt.build_run_telemetry(
            waves=[{
                "index": 0,
                "offset": 0,
                "wave_units": int(len(results)),
                "total_tasks": int(len(results)),
                "units_remaining_after": 0,
                "candidate": zsched.active_candidate_tuple(
                    effective_candidate, allow_gpu=True
                ),
                "mapping": dict(effective_mapping),
                "facts": {
                    "host": {
                        "raw_available_bytes": int(
                            gpu_bundle.envelope.raw_available_bytes
                        ),
                        "budget_bytes": int(
                            gpu_bundle.envelope.budget_bytes
                        ),
                        "factor_num": int(
                            gpu_bundle.envelope.factor_num
                        ),
                        "factor_den": int(
                            gpu_bundle.envelope.factor_den
                        ),
                    },
                    "disk": {
                        "free_bytes": int(
                            gpu_decision.facts.get("disk_free", 0)
                        )
                    },
                    "remaining_units": 0,
                },
                "shrink_steps": [dict(x) for x in _r55_active_trail],
                "wall_s": float(wall),
                "heartbeat": {
                    "heartbeat_calls": 0,
                    "heartbeat_errors": 0,
                    "heartbeat_mode": "pipeline_boundary",
                },
            }],
            checkpoints=list(run_checkpoints),
            shrink_count=int(len(_r55_active_trail)),
            refusal=None,
            parent_end=zrt.read_parent_memory(),
            deterministic_ref=zrt.sha256_payload(run_block),
        )
        telemetry["gpu_pipeline"] = pipeline_telemetry
        telemetry["gpu_revalidations"] = list(revalidations)
        telemetry["gpu_qualification"] = dict(
            gpu_decision.qualification
        )
        telemetry["gpu_tile_adaptation"] = dict(runtime_tile)
        telemetry["gpu_runtime_units"] = list(runtime_units)
        telemetry["gpu_memory_boundaries"] = [
            dict(record.get("gpu_memory") or {})
            for record in runtime_units
            if record.get("gpu_memory")
        ]
        telemetry["gpu_engine_provenance"] = _r55_engine_provenance()
        return (
            results,
            meta,
            run_block,
            telemetry,
            effective_bundle,
            effective_candidate,
            effective_mapping,
            int(len(_r55_active_trail)),
            1,
        )

    runtime_gpu_fallback = None
    _r55_run_block = None
    _r55_run_telemetry = None
    _r55_run_wave_count = 0
    _r55_run_shrink_count = 0
    _r55_gpu_run_fallback = None
    # R55-E11: the ORIGINAL GPU selection/decision survives a runtime fallback.
    # The control-flow variable is cleared (the CPU path owns execution), but the
    # decision, its effective mapping and its runtime adaptation are retained so
    # the manifest can report what the GPU actually attempted.
    _r55_gpu_retained = None
    if scheduler_mode == zsched.MODE_ACTIVE and _r55_active_gpu_decision is not None:
        try:
            (
                results,
                per_cell_meta,
                _r55_run_block,
                _r55_run_telemetry,
                _r55_active_bundle,
                _r55_active_candidate,
                _r55_active_mapping,
                _r55_run_shrink_count,
                _r55_run_wave_count,
            ) = _run_r55_active_gpu_units(
                gpu_bundle=_r55_active_bundle,
                gpu_mapping=_r55_active_mapping,
                gpu_decision=_r55_active_gpu_decision,
                gpu_candidate=_r55_active_candidate,
            )
        except _R55GpuFallback as _gpu_exc:
            # EXACT whole-batch CPU fallback from a CLEAN state. Every GPU
            # result is discarded (none were returned) and every per-cell
            # cache is removed before re-observation; the CPU plan is admitted
            # fresh under the same geometry and identity.
            _r55_gpu_retained = {
                "decision": _r55_active_gpu_decision,
                "mapping": dict(_r55_active_mapping),
                "bundle": _r55_active_bundle,
                "runtime_tile": dict(
                    (_gpu_exc.telemetry or {}).get("runtime_tile") or {}
                ),
                "telemetry": dict(_gpu_exc.telemetry or {}),
            }
            _r55_gpu_run_fallback = {
                "stage": "run",
                "reason_code": _gpu_exc.reason_code,
                "detail": _gpu_exc.detail,
                "telemetry": dict(_gpu_exc.telemetry),
            }
            for _t in _initial_bundle.cell_tasks:
                if _t[4] is not None:
                    _safe_rmtree(_t[3], None)
            gpu_used = False
            science_config = replace(science_config, backend="cpu")
            gpu_ctx = dict(gpu_ctx)
            gpu_ctx["effective"] = False
            gpu_ctx["final_effective"] = False
            gpu_ctx["actually_used"] = False
            gpu_ctx["final_backend"] = "cpu"
            gpu_ctx["runtime_fallback"] = {
                "reason": f"{_gpu_exc.reason_code}: {_gpu_exc.detail}",
                "type": "R55GpuFallback",
            }
            ignored_settings = zin.ignored_settings_present(
                zconfig, gpu_honoured=False
            )
            _diag_path = None
            try:
                _diag_path = (dict(_gpu_exc.telemetry or {})).get(
                    "diagnostics_path"
                )
            except Exception:
                _diag_path = None
            _emit(
                "ZeGrid: R55-E2 ACTIVE GPU RUN failed "
                f"({_gpu_exc.reason_code}: "
                f"{_r55_bounded_text(_gpu_exc.detail, 600)}); all GPU "
                "buffers/results were discarded and cleaned. Retrying the "
                "WHOLE batch exactly once on CPU from a fresh admission (no "
                "mixed result). Immediate diagnostics: "
                f"{_diag_path or 'unavailable'}",
                lvl="WARN", callback=progress_callback,
            )
            _r55_active_gpu_decision = None
            _r55_gpu_fallback_reason = _r55_gpu_run_fallback
            _r55_active_decision = _r55_active_decide(
                bundle=_initial_bundle,
                cell_ctxs=cell_ctxs,
                canvas=canvas,
                halo_px=halo_px,
                cache_root=cache_root,
                output_dir=output_dir,
                frame_ids=_r55_frame_ids,
            )
            (
                _r55_active_bundle,
                _r55_active_candidate,
                _r55_active_mapping,
                _r55_active_adm,
                _r55_active_trail,
            ) = _r55_active_admit(
                initial_bundle=_initial_bundle,
                decision=_r55_active_decision,
            )
            (
                results,
                per_cell_meta,
                _r55_run_block,
                _r55_run_telemetry,
                _r55_active_bundle,
                _r55_active_candidate,
                _r55_active_mapping,
                _r55_run_shrink_count,
                _r55_run_wave_count,
            ) = _run_r55_active_cpu_waves(
                science_cfg=science_config,
                start_bundle=_r55_active_bundle,
                start_candidate=_r55_active_candidate,
                start_mapping=_r55_active_mapping,
            )
        _final = _r55_active_bundle
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
        per_cell_diagnostics["mode"] = chosen_mode
        per_cell_diagnostics["cells_in_flight"] = int(cells_in_flight)
        per_cell_diagnostics["cache_workers_per_cell"] = int(
            cache_workers_per_cell
        )
        per_cell_diagnostics["cpu_streaming_tile"] = int(cpu_streaming_tile)
        per_cell_diagnostics["cpu_streaming"] = dict(cpu_streaming)
        per_cell_diagnostics["cpu_fallback_plan"] = dict(cpu_fallback_info)
        per_cell_diagnostics["ram_contract"] = _ram_contract_diagnostics(
            _adm, _final, _initial_bundle
        )
        per_cell_diagnostics["r55_run_waves"] = int(_r55_run_wave_count)
        per_cell_diagnostics["r55_run_shrink_steps"] = int(
            _r55_run_shrink_count
        )
        if _r55_gpu_run_fallback is not None:
            per_cell_diagnostics["r55_gpu_fallback"] = dict(
                _r55_gpu_run_fallback
            )
        task_cids = [t[0].cell_id for t in cell_tasks]
    elif scheduler_mode == zsched.MODE_ACTIVE and _r55_active_decision is not None:
        (
            results,
            per_cell_meta,
            _r55_run_block,
            _r55_run_telemetry,
            _r55_active_bundle,
            _r55_active_candidate,
            _r55_active_mapping,
            _r55_run_shrink_count,
            _r55_run_wave_count,
        ) = _run_r55_active_cpu_waves(
            science_cfg=science_config,
            start_bundle=_final,
            start_candidate=_r55_active_candidate,
            start_mapping=_r55_active_mapping,
        )
        # The RUN-final bundle is the ONLY source for assembly, diagnostics and
        # the manifest. A changed mapping applied only to the NEXT wave.
        _final = _r55_active_bundle
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
        per_cell_diagnostics["mode"] = chosen_mode
        per_cell_diagnostics["cells_in_flight"] = int(cells_in_flight)
        per_cell_diagnostics["cache_workers_per_cell"] = int(
            cache_workers_per_cell
        )
        per_cell_diagnostics["cpu_streaming_tile"] = int(cpu_streaming_tile)
        per_cell_diagnostics["cpu_streaming"] = dict(cpu_streaming)
        per_cell_diagnostics["cpu_fallback_plan"] = dict(cpu_fallback_info)
        per_cell_diagnostics["ram_contract"] = _ram_contract_diagnostics(
            _adm, _final, _initial_bundle
        )
        per_cell_diagnostics["r55_run_waves"] = int(_r55_run_wave_count)
        per_cell_diagnostics["r55_run_shrink_steps"] = int(
            _r55_run_shrink_count
        )
        task_cids = [t[0].cell_id for t in cell_tasks]
    elif gpu_used:
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
    # R55-E11: keep BOTH budgets distinguishable forever. Up to R55-E10 the
    # manifest's ``gpu.vram_budget_bytes`` was the legacy 50 % probe value even
    # when the ACTIVE 70 % policy executed the GPU. The legacy value is retained
    # under its own explicit key; ``vram_budget_bytes`` now reports the ACTIVE
    # budget whenever an ACTIVE GPU plan exists (executed or attempted).
    _legacy_probe_budget = gpu_ctx.get("vram_budget_bytes")
    gpu_ctx["legacy_probe_vram_budget_bytes"] = (
        None if _legacy_probe_budget is None else int(_legacy_probe_budget)
    )
    gpu_ctx["vram_budget_source"] = "legacy_probe_50"
    gpu_ctx["engine_provenance"] = _r55_engine_provenance()
    if _r55_gpu_retained is not None:
        _ret = _r55_gpu_retained
        _rt = dict(_ret.get("runtime_tile") or {})
        gpu_ctx["gpu_initial_streaming_tile"] = int(
            _ret["mapping"]["gpu_streaming_tile"]
        )
        gpu_ctx["gpu_streaming_tile"] = int(
            _rt.get("effective_tile_px")
            if _rt.get("effective_tile_px") is not None
            else _ret["mapping"]["gpu_streaming_tile"]
        )
        gpu_ctx["gpu_streaming_host_peak_bytes"] = int(
            _ret["bundle"].gpu_streaming_host_peak_bytes
        )
        gpu_ctx["gpu_tile_adaptation"] = _rt
        gpu_ctx["vram_budget_bytes"] = int(
            _ret["decision"].qualification["budget_bytes"]
        )
        gpu_ctx["vram_budget_source"] = "active_r55_policy_70"
    elif _r55_active_gpu_decision is not None:
        gpu_ctx["vram_budget_bytes"] = int(
            _r55_active_gpu_decision.qualification["budget_bytes"]
        )
        gpu_ctx["vram_budget_source"] = "active_r55_policy_70"
        gpu_ctx["gpu_initial_streaming_tile"] = int(gpu_streaming_tile)
    # ZM-ZEGRID-R36: distinguish the GPU streaming tile from the CPU streaming
    # tile (the GPU->CPU fallback uses the CPU tile, never a re-coded 128).
    gpu_ctx["cpu_streaming_tile"] = int(cpu_streaming_tile)
    if _r55_gpu_retained is None:
        gpu_ctx["gpu_streaming_tile"] = (
            int(gpu_streaming_tile) if gpu_streaming_tile is not None else None
        )
        gpu_ctx["gpu_streaming_host_peak_bytes"] = (
            int(gpu_streaming_host_peak_bytes)
            if gpu_streaming_host_peak_bytes is not None
            else None
        )
    # else: R55-E11 retained the EFFECTIVE attempted GPU tile above; the CPU
    # bundle's ``None`` must not erase the evidence of what the GPU ran.
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

    # R55-D2: build the active block AFTER execution so its status and final
    # values are the ones that actually ran. It is deterministic (no timing or
    # observed metric) and is the ONLY scheduler block in active mode.
    if scheduler_mode == zsched.MODE_ACTIVE and (
        _r55_active_decision is not None
        or _r55_active_gpu_decision is not None
    ):
        _r55_active_ok = all(
            r.get("status") in ("complete", "empty") for r in results
        )
        _r55_active_plan_set = (
            _r55_active_decision.plan_set
            if _r55_active_decision is not None
            else _r55_active_gpu_decision.plan_set
        )
        _r55_scheduler_block = _r55_build_active_block(
            plan_set=_r55_active_plan_set,
            final_candidate=_r55_active_candidate,
            mapping=_r55_active_mapping,
            bundle=_final,
            initial_bundle=_initial_bundle,
            preflight_admission=_r55_active_preflight_adm,
            launch_admission=_r55_active_launch_adm,
            preflight_trail=_r55_active_preflight_trail,
            launch_trail=_r55_active_launch_trail,
            status=("executed" if _r55_active_ok else "partial"),
            run=_r55_run_block,
            worker_adaptation=_r55_worker_adaptation_detail(
                (
                    _r55_active_gpu_decision
                    if _r55_active_gpu_decision is not None
                    else _r55_active_decision
                ).facts,
                (
                    _r55_active_gpu_decision
                    if _r55_active_gpu_decision is not None
                    else _r55_active_decision
                ).worker_trail,
            ),
            tile_ladder=(
                _r55_active_gpu_decision.gpu_tile_ladder
                if _r55_active_gpu_decision is not None
                else (
                    _r55_gpu_retained["decision"].gpu_tile_ladder
                    if _r55_gpu_retained is not None
                    else None
                )
            ),
        )
        if _r55_active_gpu_decision is not None:
            _r55_scheduler_block["gpu_execution"] = dict(
                _r55_active_gpu_decision.qualification
            )
            _r55_scheduler_block["gpu_tile_decision"] = (
                _r55_gpu_tile_decision_dict(
                    _r55_active_gpu_decision,
                    _r55_active_mapping,
                    _final,
                    runtime=(_r55_run_telemetry or {}).get(
                        "gpu_tile_adaptation"
                    ),
                    legacy_probe_budget_bytes=gpu_ctx.get(
                        "legacy_probe_vram_budget_bytes"
                    ),
                )
            )
            if _r55_run_telemetry and _r55_run_telemetry.get(
                "gpu_tile_adaptation"
            ):
                _r55_scheduler_block["gpu_tile_runtime"] = dict(
                    _r55_run_telemetry["gpu_tile_adaptation"]
                )
        if (
            _r55_active_gpu_decision is None
            and _r55_gpu_retained is not None
        ):
            # R55-E11: the ORIGINAL GPU decision survives the runtime fallback.
            # Its absence after a fallback was previously unreadable (None meant
            # both "never selected" and "selected then abandoned").
            _ret = _r55_gpu_retained
            _ret_runtime = dict(_ret.get("runtime_tile") or {})
            _r55_scheduler_block["gpu_tile_decision"] = (
                _r55_gpu_tile_decision_dict(
                    _ret["decision"],
                    _ret["mapping"],
                    _ret["bundle"],
                    runtime=(_ret_runtime or None),
                    legacy_probe_budget_bytes=gpu_ctx.get(
                        "legacy_probe_vram_budget_bytes"
                    ),
                )
            )
            _r55_scheduler_block["gpu_original_decision"] = {
                "schema": _R55E11_TILE_RUNTIME_SCHEMA,
                "reason": "runtime_fallback_preserved_selection",
                "selected_tile_px": int(
                    _ret["mapping"]["gpu_streaming_tile"]
                ),
                "effective_tile_px": _ret_runtime.get("effective_tile_px"),
                "declared_ladder": [
                    int(t) for t in _ret["decision"].gpu_tile_ladder
                ],
                "vram_percent": int(
                    _ret["decision"].qualification["percent"]
                ),
                "vram_budget_bytes": int(
                    _ret["decision"].qualification["budget_bytes"]
                ),
                "legacy_probe_vram_budget_bytes": gpu_ctx.get(
                    "legacy_probe_vram_budget_bytes"
                ),
                "engine_provenance": _r55_engine_provenance(),
            }
            _r55_scheduler_block["gpu_tile_runtime"] = _ret_runtime
            _ret_pipeline = (_ret.get("telemetry") or {}).get("pipeline")
            if _ret_pipeline:
                _r55_scheduler_block["gpu_fallback_pipeline"] = dict(
                    _ret_pipeline
                )
        if _r55_gpu_run_fallback is not None:
            _r55_scheduler_block["gpu_fallback"] = dict(
                _r55_gpu_run_fallback
            )
        elif _r55_gpu_fallback_reason is not None:
            _r55_scheduler_block["gpu_fallback"] = dict(
                _r55_gpu_fallback_reason
            )

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
        r55_run_telemetry=_r55_run_telemetry,
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
    r55_run_telemetry=None,
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
                   "predicted_bound_bytes": layout["predicted_bound_bytes"],
                   "layout_origin": layout.get("layout_origin", "auto_refinement"),
                   **({"layout_rescue": layout["layout_rescue"]}
                      if layout.get("layout_rescue") else {})},
        "layout_source": layout.get("layout_source", "auto"),
        # ZM-ZEGRID-R53b: ONE stable top-level provenance block (same shape for
        # global / per_cell / auto outcomes). It is resolved once at the product
        # entry and threaded here explicitly; it never re-reads config.
        **(
            {"zegrid_halo": dict(halo_provenance)}
            if halo_provenance is not None else {}
        ),
        "reproducibility_note": _reproducibility_note(_effective_gauge_mode),
        # ZM-ZEGRID-ACTIVE-OOC-FALLBACK: explicit storage provenance for the
        # ACTIVE product path (legacy/observe keep the pre-existing shape). It
        # records what was REQUESTED, what actually EXECUTED, why dense was
        # refused and which phase triggered the ONE fallback.
        **(
            {"storage": _storage_policy_snapshot()}
            if _STORAGE_ACTIVE_MODE else {}
        ),
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
        # ZM-ZEGRID-R55-E1: ADDITIVE observational RUN telemetry. ``None`` in
        # legacy/observe -> the exact pre-E1 manifest shape is preserved.
        **(
            {"r55_run_telemetry": dict(r55_run_telemetry)}
            if r55_run_telemetry is not None else {}
        ),
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
    storage_profile: str | None = None,
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

    ZM-ZEGRID-R52-OOC Lot C (L-3): this entry point is ONE RUN PER PROCESS and is
    NOT re-entrant. The storage profile is selected once (``storage_profile=``
    else dense) into the run-scoped ``_STACK_STORAGE_PROFILE`` seam, like the
    existing ``_STACK_CONFIG``/``_STACK_TILE_SIZE`` globals; a nested call would
    overwrite its parent's profile. The profile itself is COPIED into every cell
    task (7th element) and into the prepared/record payload, so a worker never
    depends on the parent's global to know which provider/sink to use.

    ZM-ZEGRID-R37-D: ``r37_diagnostic_capture`` (default ``False``) is an explicit,
    diagnostic-only opt-in. When requested it publishes the accepted R37-C strict
    JSON pre-offset/pre-feather artifact into EACH actual output directory and
    records compact capture metadata in the manifest/run log. It never applies a
    correction and never changes the science/aesthetic/coverage outputs; the
    disabled path never imports, builds or writes the diagnostic. Capture requires
    the effective ``cell_gauge_mode`` to be ``per_cell`` (``auto`` is honoured when
    it resolves to per_cell; a global resolution fails loudly).
    """
    global _STACK_STORAGE_PROFILE
    _storage_policy_reset(storage_profile)
    _STACK_STORAGE_PROFILE = _STORAGE_EFFECTIVE
    _emit("ZeGrid engine activated (explicit ZeGrid mode)", callback=progress_callback)
    _emit(
        "ZeGrid: storage policy — requested="
        f"{_STORAGE_REQUESTED} effective={_STORAGE_EFFECTIVE} "
        f"(model={zsp.storage_model_id(_STORAGE_EFFECTIVE)}); "
        + (
            "auto = dense first, ONE bounded retry only on a qualified "
            "host-RAM refusal"
            if _STORAGE_REQUESTED == zsp.STORAGE_PROFILE_AUTO
            else "explicit profile is FORCED (no silent fallback)"
        ),
        callback=progress_callback,
    )

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
    # ZM-ZEGRID-ACTIVE-OOC-FALLBACK: the storage provenance block is published
    # ONLY for the ACTIVE product path (legacy/observe keep their exact shape).
    global _STORAGE_ACTIVE_MODE
    _STORAGE_ACTIVE_MODE = (_scheduler_mode == zsched.MODE_ACTIVE)
    _emit(
        f"ZeGrid: scheduler mode = {_scheduler_mode} "
        + (
            "(R55 observe-only: records a predicted plan; execution remains the "
            "legacy path)"
            if _scheduler_mode == zsched.MODE_OBSERVE
            else (
                "(R55-D2 active: a deterministic mapped CPU plan grants "
                "execution authority before the gauge)"
                if _scheduler_mode == zsched.MODE_ACTIVE
                else "(legacy default: R55 scheduler disabled)"
            )
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
    gpu_requested, gpu_source, gpu_probe, vram_budget = _r55_resolve_gpu_probe(
        use_gpu, zconfig
    )
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
