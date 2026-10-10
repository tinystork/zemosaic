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
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import psutil
from astropy.io import fits
from astropy.wcs import WCS

from . import zemosaic_stack_plan as _stack_plan
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
from .core.zegrid import execution as zxe
from .core.zegrid import file_provider as zfp
from .core.zegrid import final_mosaic_finishing as zfin
from .core.zegrid import geometry as zg
from .core.zegrid import gpu as zgpu
from .core.zegrid import instrumentation as zin
from .core.zegrid import mosaic as zmosaic
from .core.zegrid import observability as zobs
from .core.zegrid import parallel as zpar
from .core.zegrid import photometric as zphot
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
# Fraction of available memory the R4 in-memory bound must fit within to select
# the in-memory path (headroom for decode/cache transients + allocator variance).
INMEM_AVAILABLE_FRAC = 0.80
# Normalization token for the ZeGrid engine when none is explicitly requested.
DEFAULT_NORMALIZATION = "sky_mean"


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
    if reuse_cache and zfp.cache_is_complete(str(cache_dir), frame_ids):
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
        if zfp.cache_is_complete(cache_dir, list(mem.patch_ids)):
            reused.append(cid)
            continue  # reuse: no wipe, no decode
        builders[cid] = zfp.AlignedCacheBuilder(cache_dir)  # wipes THIS cell dir only
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
    """
    worst = None
    for c in _iter_cell_bounds_mode_aware(canvas, footprints, nx, ny, halo_px, tile_size):
        cheaper = min(c["inmem_bound_bytes"], c["stream_bound_bytes"])
        if worst is None or cheaper > worst["cheaper_bound_bytes"]:
            worst = dict(c)
            worst["cheaper_bound_bytes"] = cheaper
    return worst


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
):
    """MODE-AWARE RAM-aware layout: coarsest candidate whose cheaper mode fits.

    Reuses the R4 candidate enumeration (``REFINEMENT_FACTORS``, scientific
    floors, footprint membership) verbatim, but a candidate is FEASIBLE iff the
    CHEAPER-fitting per-cell mode's worst bound fits ``ram_budget``. Deterministic
    (identical enumeration order); returns a dict (layout + per-cell predictions).

    ``pinned_layout`` (``(nx, ny)`` or ``None``) BYPASSES the RAM-adaptive search
    and uses exactly that layout (scientific floors still enforced; an infeasible
    pinned layout raises). ``layout_source`` records ``"pinned"`` vs ``"auto"``.

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
    if observer is not None:
        observer.start(
            zobs.LAYOUT_SUBSTEP_FOOTPRINTS,
            "WCS footprints / layout inputs",
            total=2 * n_frames,
        )

    def _fp_progress(pass_offset):
        def cb(done, total, item_id):
            if observer is not None:
                observer.progress(
                    pass_offset + int(done), item_id=item_id, total=2 * n_frames
                )
        return cb

    mw, mh = zal.median_projected_footprint(frames, canvas, progress=_fp_progress(0))
    if not (mw > 0 and mh > 0):
        if observer is not None:
            observer.end()
        raise zal.LayoutInfeasible("median projected footprint is degenerate")
    footprints = zal._footprints(frames, canvas, progress=_fp_progress(n_frames))
    if observer is not None:
        observer.end()

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
                "Suggest increasing available RAM or pinning a coarser layout via the "
                "'zegrid_layout' config key."
            )
            _log(msg, "ERROR")
            raise zal.LayoutInfeasible(msg)
        cells = list(_iter_cell_bounds_mode_aware(canvas, footprints, nx, ny, halo_px, tile_size))
        max_n = max((c["n"] for c in cells), default=0)
        worst_bound = max((min(c["inmem_bound_bytes"], c["stream_bound_bytes"]) for c in cells), default=0)
        if ram_budget is not None and worst_bound > ram_budget:
            msg = (
                f"pinned layout {nx}x{ny} cannot fit any mode within budget: "
                f"cheaper bound {worst_bound / 2**20:.1f} MiB > budget {ram_budget / 2**20:.1f} MiB. "
                "Suggest increasing available RAM or pinning a coarser layout via the "
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
        return {
            "nx": nx, "ny": ny,
            "cell_count": nx * ny,
            "ram_budget_bytes": ram_budget,
            "available_bytes": available_bytes if available_bytes is not None else ram_budget,
            "max_contributors": max_n,
            "refinement_factor": None,
            "predicted_bound_bytes": int(worst_bound),
            "max_patch_area": max((c["patch_area"] for c in cells), default=0),
            "budget_bound_choice": False,
            "warnings": tuple(warnings),
            "cells": cells,
            "layout_source": "pinned",
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
        memory_ok = (ram_budget is None) or (worst["cheaper_bound_bytes"] <= ram_budget)
        bound_mib = worst["cheaper_bound_bytes"] / 2**20
        budget_mib = (ram_budget / 2**20) if ram_budget is not None else None
        if memory_ok:
            _log(
                f"layout scan: candidate {nx}x{ny} ACCEPTED "
                f"(cheaper-mode bound {bound_mib:.1f} MiB"
                + (f" <= budget {budget_mib:.1f} MiB" if budget_mib is not None else "") + ")"
            )
        else:
            _log(
                f"layout scan: candidate {nx}x{ny} REJECTED — "
                f"cheaper-mode bound {bound_mib:.1f} MiB > budget {budget_mib:.1f} MiB"
            )
        candidates.append({
            "factor": factor, "nx": nx, "ny": ny,
            "bound": worst["cheaper_bound_bytes"], "worst_cell": worst,
            "memory_ok": memory_ok, "geom": geom,
        })

    chosen = None
    chosen_index = None
    for i, cand in enumerate(candidates):
        if cand["memory_ok"]:
            chosen = cand
            chosen_index = i
            break

    if observer is not None:
        observer.end()

    if chosen is None:
        finest = candidates[-1] if candidates else None
        reasons = []
        if ram_budget is not None and finest is not None and finest["bound"] > ram_budget:
            reasons.append(
                f"even the finest floor-feasible layout ({finest['nx']}x{finest['ny']}) "
                f"cheaper-mode bound {finest['bound'] / 2**20:.1f} MiB exceeds budget "
                f"{ram_budget / 2**20:.1f} MiB"
            )
        reasons.append(
            f"min_patch_area floor ({floors.min_patch_area_px} px) / "
            f"max_halo_overhead floor ({floors.max_halo_overhead}) cannot be satisfied "
            f"together with the budget (mode-aware)"
        )
        msg = (
            "RAM budget cannot satisfy scientific floors (mode-aware): "
            + "; ".join(reasons)
            + ". Suggest either increasing available RAM or pinning the layout via "
            "the 'zegrid_layout' config key."
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

    _log(
        f"layout scan: CHOSEN {chosen['nx']}x{chosen['ny']} "
        f"({chosen['nx'] * chosen['ny']} cells, mode-aware, source=auto) "
        f"cheaper-mode bound {chosen['bound'] / 2**20:.1f} MiB"
    )

    return {
        "nx": chosen["nx"], "ny": chosen["ny"],
        "cell_count": chosen["nx"] * chosen["ny"],
        "ram_budget_bytes": ram_budget,
        "available_bytes": available_bytes if available_bytes is not None else ram_budget,
        "max_contributors": max_n,
        "refinement_factor": chosen["factor"],
        "predicted_bound_bytes": int(chosen["bound"]),
        "max_patch_area": chosen["worst_cell"]["patch_area"],
        "budget_bound_choice": constrained,
        "warnings": tuple(warnings),
        "cells": cells,
        "layout_source": "auto",
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


def _init_stack_worker(descs, canvas, science_config, global_gauge, global_frame_ids, global_reference_frame_id, tile_size=STREAM_TILE_SIZE, cache_workers=1):
    """Child/parent initializer: publish the read-only per-cell stack inputs."""
    global _STACK_DESCS, _STACK_CANVAS, _STACK_CONFIG, _STACK_GAUGE
    global _STACK_GLOBAL_FRAME_IDS, _STACK_GLOBAL_REF, _STACK_TILE_SIZE
    global _STACK_CACHE_WORKERS
    _STACK_DESCS = descs
    _STACK_CANVAS = canvas
    _STACK_CONFIG = science_config
    _STACK_GAUGE = global_gauge
    _STACK_GLOBAL_FRAME_IDS = global_frame_ids
    _STACK_GLOBAL_REF = global_reference_frame_id
    _STACK_TILE_SIZE = tile_size
    _STACK_CACHE_WORKERS = cache_workers


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
    if not mem.patch_ids:
        return {"cell_id": cid, "status": "empty",
                "record": {"cell_id": cid, "status": "empty", "mode": None},
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

    # Subset the global gauge to this cell's frames (in cache order).
    cell_provider = zfp.MemmapCanonicalProvider(cache_dir)
    try:
        cell_frame_ids = list(cell_provider.frame_ids)
    finally:
        cell_provider.close()
    cell_fixed = subset_fixed_normalization(
        _STACK_GAUGE, _STACK_GLOBAL_FRAME_IDS, cell_frame_ids
    )

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
        _STACK_GLOBAL_REF, cell_frame_ids, sres.reference_frame_id
    )
    record = {
        "cell_id": cid, "row": cell.row, "col": cell.col, "status": "complete",
        "mode": mode, "n_contributors": len(mem.patch_ids),
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

    return {
        "cell_id": cid, "status": "complete", "core": core, "record": record,
        "backend_used": getattr(_STACK_CONFIG, "backend", "cpu"),
        "n_frames": int(manifest.get("n_frames", 0)),
        "total_bytes": int(manifest.get("total_bytes", 0)),
        "cache_build_s": cache_build_s, "stack_s": stack_s,
        "peak_rss_kib": peak_rss, "cleanup_failed": cleanup_failed,
        "reproject": zxe.get_reproject_path_stats().to_dict(),
        "cache_dir": str(cache_dir),
    }


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


def _reference_provenance(global_reference_frame_id, cell_frame_ids, cell_reference_frame_id):
    """ZM-ZEGRID-R11 L1 (provenance honesty): classify a Cell's reference.

    A Cell's ``reference_frame_id`` is the TRUE photometric anchor only when the
    Cell contains the global reference frame; otherwise it is a BOOKKEEPING
    placeholder (the Cell's highest-weight active frame) and the true anchor is
    ``photometric_gauge.global_reference_frame_id``.

    Returns ``{reference_frame_role, bookkeeping_reference_frame_id}``.
    """
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
):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = zin.Timings()
    start_ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")

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
    _setup_rep.end(
        throughput=_fmt_throughput(len(frames_info), timings.get("setup"), "frames/s")
    )
    global_eta.phase_completed("setup", timings.get("setup"))
    _emit(
        f"ZeGrid: canvas {canvas.width}x{canvas.height} "
        f"(resolution {canvas.resolution_deg:.3g} deg/px)",
        callback=progress_callback,
    )

    # LAYOUT — MODE-AWARE RAM-aware Auto layout (in-memory OR streaming bound),
    # budgeted from psutil (portable). Pinnable via ``zegrid_layout``.
    available = available_memory_bytes()
    _layout_rep = _reporter()
    _layout_rep.start("layout")
    # ZM-ZEGRID-R27: layout sub-step observability — a bounded, throttled reporter
    # for the three stable layout sub-steps (footprints, candidate scan, chosen
    # cells). It emits ONLY log/durable-detail events (never GUI stage/percent),
    # so the current GUI cannot misinterpret a sub-step as a global phase.
    layout_substep = zobs.SubstepReporter(_emit_live, log_line=_log_line)
    with timings.timed("layout"):
        layout = _choose_layout_mode_aware(
            canvas, descs, ram_budget=available, available_bytes=available,
            tile_size=STREAM_TILE_SIZE, pinned_layout=pinned_layout,
            emit=_emit_live, log_line=_log_line, observer=layout_substep,
        )
        nx, ny = layout["nx"], layout["ny"]
        cell_total = nx * ny
        # Chosen-cell membership/context preparation — the final
        # ``build_cell_context`` pass (previously a silent 170-cell loop).
        layout_substep.start(
            zobs.LAYOUT_SUBSTEP_CHOSEN_CELLS,
            "chosen-cell membership/context preparation",
            total=cell_total,
        )
        cell_ctxs = []
        done = 0
        for row, col, _bounds in zg.build_layout(canvas, nx, ny).iter_cells(canvas):
            cell, patch, mem = zsw.build_cell_context(descs, canvas, row, col, nx, ny)
            cell_ctxs.append((row, col, cell, patch, mem))
            done += 1
            layout_substep.progress(done, item_id=cell.cell_id, total=cell_total)
        layout_substep.end()
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
        f"source={layout.get('layout_source', 'auto')}) from RAM budget "
        f"{available / 2**30:.2f} GiB; worst-cell cheaper-mode bound "
        f"{layout['predicted_bound_bytes'] / 2**20:.1f} MiB (maxN={layout['max_contributors']})",
        callback=progress_callback,
    )
    for w in layout["warnings"]:
        _emit(f"ZeGrid: layout warning — {w}", lvl="WARN", callback=progress_callback)

    # Parallel workers (ZM-ZEGRID-R16 adaptive rule: clamp(min(cpu-2, RAM//footprint), 2, 14),
    # per-phase footprint — the gauge reprojects full-canvas frames so it is heavier
    # than the per-cell cache build).
    if workers is None:
        # ZM-ZEGRID-R17 (R16-M1 fix): derive the per-worker footprint from the ACTUAL
        # pixel areas rather than a static estimate. The gauge's pairs pass unions
        # the reference's and the frame's bboxes, which can approach the CANVAS
        # area; the cache build reprojects one cell patch. Using the real areas
        # prevents over-spawning (OOM) on a large mosaic.
        canvas_px = int(canvas.width) * int(canvas.height)
        patch_px = 0
        for _row, _col, _cell, _patch, _mem in cell_ctxs:
            try:
                _ph, _pw = _patch.patch_shape_hw
                patch_px = max(patch_px, int(_ph) * int(_pw))
            except Exception:
                pass
        gauge_footprint = zpar.gauge_footprint_bytes(canvas_px)
        cache_footprint = zpar.cache_footprint_bytes(patch_px or canvas_px)
        # ZM-ZEGRID-R17 (R16-M1): budget on a FRACTION of the available RAM (the
        # workers' peak sits on top of the main process + the product) so the rule
        # cannot admit more workers than the machine can hold.
        worker_budget = int(available_memory_bytes() * zpar.RAM_SAFETY_FRACTION)
        gauge_workers = zpar.choose_workers(None, worker_budget, gauge_footprint)
        cache_workers = zpar.choose_workers(None, worker_budget, cache_footprint)
        _emit(
            f"ZeGrid: parallel workers — gauge={gauge_workers} "
            f"(CPU={os.cpu_count()}, avail={available / 2**30:.2f}GiB, "
            f"budget={worker_budget / 2**30:.2f}GiB, footprint={gauge_footprint / 2**20:.0f}MiB), "
            f"cache_build={cache_workers} (footprint={cache_footprint / 2**20:.0f}MiB)",
            callback=progress_callback,
        )
        workers = gauge_workers
        cache_build_workers = cache_workers
    else:
        cache_build_workers = workers

    cache_root = output_dir / CACHE_DIR_NAME
    cache_root.mkdir(parents=True, exist_ok=True)

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
    # ZM-ZEGRID-R20: per-cell STACKING runs several cells CONCURRENTLY, bounded
    # adaptively by the available RAM. Each cell worker builds its aligned cache
    # SERIALISED (no nested pool), stacks, and deletes its cache — preserving the
    # per-cell temp reuse + bounded-disk behaviour (peak disk is `cells_in_flight`
    # x the largest cell, not the sum). A concurrent-path failure WARNs loudly and
    # degrades to the serial loop via pmap's fail-safe (never crash, never silent).
    available_now = available_memory_bytes()
    per_cell_budget = int(available_now * zpar.RAM_SAFETY_FRACTION)
    cell_specs = []          # (cell, patch, mem, cache_dir, inmem_b | None, stream_b)
    cell_bound_pairs = []    # (inmem_b, stream_b) for the joint planner (non-empty)
    for (row, col, cell, patch, mem) in cell_ctxs:
        cid = cell.cell_id
        idx = row * nx + col
        if not mem.patch_ids:
            cell_specs.append((cell, patch, mem, str(cache_root / cid), None, 0))
            continue
        n = len(mem.patch_ids)
        area = patch.patch.width * patch.patch.height
        inmem_b = int(zal.FITTED_MEMORY_MODEL.predict_bound_bytes(n, area))
        stream_b = _estimate_streaming_bytes(n, patch.patch_shape_hw, STREAM_TILE_SIZE)
        cell_bound_pairs.append((inmem_b, stream_b))
        cell_specs.append((cell, patch, mem, str(cache_root / cid), inmem_b, stream_b))
        _emit(
            f"ZeGrid: cell {cid} ({idx + 1}/{total_cells}) N={n} area={area}px "
            f"inmem={inmem_b / 2**20:.1f}MiB stream={stream_b / 2**20:.1f}MiB",
            callback=progress_callback,
        )

    # ZM-ZEGRID-R22: JOINT mode + concurrency planner (replaces the R20 two-step
    # trap: 'pick inmem per cell, then discover concurrency=1'). One explicit
    # candidate plan compares safe in-memory vs safe streaming concurrency against
    # the CPU RAM AND (when GPU) the single-device owner constraint, and the
    # chosen concurrency is recorded verbatim in the manifest (never a silent
    # serial surprise).
    _plan = zpar.plan_cell_concurrency(
        cell_bound_pairs, os.cpu_count(), per_cell_budget,
        gpu_backend=gpu_used, cache_build_workers=cache_build_workers,
    )
    chosen_mode = _plan["mode"]
    cells_in_flight = int(_plan["cells_in_flight"])
    cache_workers_per_cell = int(_plan["cache_workers_per_cell"])
    _emit(
        f"ZeGrid: joint plan — mode={chosen_mode} cells_in_flight={cells_in_flight} "
        f"cache_workers_per_cell={cache_workers_per_cell} "
        f"gpu_serialized={_plan['gpu_serialized']} — {_plan['chosen_reason']}",
        callback=progress_callback,
    )
    for c in _plan["candidates"]:
        _emit(
            f"ZeGrid: plan candidate mode={c['mode']} cells_in_flight={c['cells_in_flight']} "
            f"max_bound={c['max_bound_bytes'] / 2**20:.1f}MiB "
            f"makespan_rel={c['makespan_rel']:.3f}",
            callback=progress_callback,
        )

    # Build the final cell tasks with the CHOSEN uniform mode + its bound.
    cell_tasks = []
    for (cell, patch, mem, cache_dir, inmem_b, stream_b) in cell_specs:
        if inmem_b is None:
            cell_tasks.append((cell, patch, mem, cache_dir, None, 0))
        elif chosen_mode == "inmem":
            cell_tasks.append((cell, patch, mem, cache_dir, "inmem", inmem_b))
        else:
            cell_tasks.append((cell, patch, mem, cache_dir, "stream", stream_b))

    per_cell_footprint = max((t[5] for t in cell_tasks if t[4] is not None), default=0)

    # ZM-ZEGRID-R22: VRAM-bounded GPU tile size for the stack phase (when the GPU
    # backend is effective). Single GPU owner => stack tasks are serialised; the
    # tile size is a pure function of (max N, worst patch, VRAM budget) so the
    # device never materialises N x full-cell. A degenerate (None) tile degrades
    # LOUDLY to exact CPU (no silent fallback).
    gpu_tile_size = STREAM_TILE_SIZE
    if gpu_used:
        max_n = max((len(m.patch_ids) for (_r, _c, _ce, _p, m) in cell_ctxs if m.patch_ids), default=1)
        worst_hw = (1, 1)
        worst_area = -1
        for (_r, _c, _ce, _p, m) in cell_ctxs:
            if m.patch_ids:
                hw = _p.patch_shape_hw
                area = hw[0] * hw[1]
                if area > worst_area:
                    worst_area = area
                    worst_hw = hw
        _gpu_ts = zgpu.choose_gpu_tile_size(max_n, 3, worst_hw, gpu_ctx.get("vram_budget_bytes"))
        if _gpu_ts is None:
            _emit(
                "ZeGrid: VRAM budget cannot hold even a minimum GPU tile for the "
                "heaviest cell; degrading to exact CPU (no silent fallback)",
                lvl="WARN", callback=progress_callback,
            )
            gpu_used = False
            science_config = replace(science_config, backend="cpu")
            gpu_ctx = dict(gpu_ctx)
            gpu_ctx["effective"] = False
            gpu_ctx["attempted"] = False
            gpu_ctx["final_effective"] = False
            gpu_ctx["actually_used"] = False
            gpu_ctx["final_backend"] = "cpu"
            gpu_ctx["fallback_reason"] = gpu_ctx.get("fallback_reason") or "vram_tile_infeasible"
            ignored_settings = zin.ignored_settings_present(zconfig, gpu_honoured=False)
        else:
            gpu_tile_size = _gpu_ts
    _emit(
        f"ZeGrid: stack tile size={gpu_tile_size} backend={science_config.backend}",
        callback=progress_callback,
    )

    _emit(
        f"ZeGrid: per-cell concurrency — cells_in_flight={cells_in_flight} "
        f"(cpu={os.cpu_count()}, avail={available_now / 2**30:.2f}GiB, "
        f"budget={per_cell_budget / 2**30:.2f}GiB, "
        f"footprint={per_cell_footprint / 2**20:.0f}MiB, "
        f"cache_build_workers={cache_build_workers})",
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
    }
    per_cell_meta = {}

    def _cell_progress(done, total):
        cid = task_cids[int(done) - 1] if 0 < int(done) <= len(task_cids) else None
        _stack_rep.progress(int(done), item_id=cid)

    t_block0 = time.perf_counter()

    def _run_cell_batch(science_cfg, tile_size, *, gpu_attempt):
        """Run the WHOLE per-cell batch once; returns (results, meta).

        ``gpu_attempt=True`` disables pmap's SAME-config serial retry (serial_fallback
        False) so a CuPy OOM/driver error propagates here for a one-shot CPU rerun
        instead of being retried with the same GPU config (which would repeat the
        error and crash).
        """
        _meta = {}
        _res = zpar.pmap(
            _stack_cell, cell_tasks, workers=cells_in_flight,
            progress_callback=_cell_progress, emit=_emit_live, meta=_meta,
            initializer=_init_stack_worker,
            initargs=(descs, canvas, science_cfg, global_gauge, global_frame_ids,
                      global_reference_frame_id, tile_size, cache_workers_per_cell),
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
                # H1: classified GPU runtime failure -> one-shot exact-CPU
                # whole-batch rerun (WARN, recorded; no recursion).
                runtime_gpu_fallback = {"reason": repr(exc), "type": type(exc).__name__}
                _emit(
                    f"ZeGrid: GPU RUNTIME failure ({type(exc).__name__}); "
                    f"degrading to exact CPU whole-batch rerun (no silent fallback): {exc}",
                    lvl="WARN", callback=progress_callback,
                )
                gpu_used = False
                science_config = replace(science_config, backend="cpu")
                gpu_ctx = dict(gpu_ctx)
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
                # Rerun the WHOLE batch exactly once on CPU (no recursion).
                results, per_cell_meta = _run_cell_batch(
                    science_config, STREAM_TILE_SIZE, gpu_attempt=False
                )
            else:
                raise  # arbitrary science/programming error propagates (never swallowed)
    else:
        results, per_cell_meta = _run_cell_batch(
            science_config, gpu_tile_size, gpu_attempt=False
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
    _emit("ZeGrid: assembly (R3 assemble_canvas)", callback=progress_callback)
    _assembly_rep = _reporter()
    _assembly_rep.start("assembly")
    with timings.timed("assembly"):
        assembled = zmosaic.assemble_canvas(canvas, nx, ny, cores)
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

    # Write the immutable pre-finishing assembled science FIRST, so a finishing
    # exception can never destroy the scientific reference.
    _write_raw_science_fits(
        assembled, canvas, raw_science_path,
        related_file=(aesthetic_path.name if aesthetic_path is not None else None),
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
    )

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


def _write_raw_science_fits(assembled, canvas, raw_science_path, role="SCI", related_file=None):
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
    finished_science=None, finishing_info=None, fin_uint16=None,
    gpu_context=None,
    aggregate_peak_rss_kib=None,
    raw_science_path=None,
    raw_science=None,
    aesthetic_path=None,
    export_aesthetic_fits=False,
    scientific_fits_suffix=None,
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
        _write_raw_science_fits(assembled, canvas, raw_science_path)
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
        "reproducibility_note": (
            "output science is GLOBAL-gauge normalized (one reference frame + "
            "per-frame coefficients computed over the full canvas), so the science "
            "is layout-INDEPENDENT under the corrected global gauge "
            "(ZM-ZEGRID-R11 rework-1: Cells lacking the global reference keep its "
            "level, not a re-anchored local level). Pin the layout via the "
            "'zegrid_layout' config key (e.g. '6x5') for reproducible output; "
            "coverage (stack depth) is pure geometry."
        ),
        "photometric_gauge": {
            "mode": "global",
            "global_reference_frame_id": global_reference_frame_id,
            "note": (
                "one reference frame + per-frame sky_mean coefficients computed once "
                "over the full canvas footprint; every Cell reuses the same gauge "
                "(Cells lacking the global reference keep its GLOBAL level)."
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
            ),
        },
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
    }
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
) -> None:
    """Run the NEW ZeGrid engine over a ``stack_plan.csv`` (production entry).

    Normalization defaults to ``sky_mean``;
    weighting/rejection/combine/taper use the frozen ZeGrid science config.
    """
    _emit("ZeGrid engine activated (stack_plan.csv detected)", callback=progress_callback)

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

    csv_path = Path(input_folder).expanduser() / "stack_plan.csv"
    frames_info = _stack_plan.load_stack_plan(csv_path, progress_callback=progress_callback)
    if not frames_info:
        raise RuntimeError("ZeGrid failed: no frames loaded from stack_plan.csv")

    science_config = replace(
        ExecutorConfig().science_config(),
        normalization=_resolve_normalization(stack_norm_method),
        backend=("gpu" if gpu_effective else "cpu"),
        **rej_overrides,
    )
    _emit(
        f"ZeGrid science config: normalization={science_config.normalization} "
        f"(default={DEFAULT_NORMALIZATION}), weighting={science_config.weighting}, "
        f"rejection={science_config.rejection} "
        f"(sigma={science_config.sigma_low:.2f}/{science_config.sigma_high:.2f}, "
        f"winsor={science_config.winsor_limit_low:.3f}/{science_config.winsor_limit_high:.3f}), "
        f"combine={science_config.combine}, taper={science_config.taper}, "
        f"backend={science_config.backend}",
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
                        gpu_context=gpu_context)
        if altz_frames:
            _run_single(altz_frames, input_folder, base_out / "grid_ALTZ",
                        progress_callback=progress_callback,
                        science_config=science_config, zconfig=zconfig,
                        pinned_layout=pinned_layout, sip_mode=sip_mode, workers=workers,
                        ignored_run_args=ignored_run_args,
                        finishing_config=finishing_config,
                        gpu_context=gpu_context)
    else:
        _emit("ZeGrid: mount info missing or homogeneous — single pass",
              callback=progress_callback)
        _run_single(frames_info, input_folder, base_out,
                    progress_callback=progress_callback,
                    science_config=science_config, zconfig=zconfig,
                    pinned_layout=pinned_layout, sip_mode=sip_mode, workers=workers,
                    ignored_run_args=ignored_run_args,
                    finishing_config=finishing_config,
                    gpu_context=gpu_context)
