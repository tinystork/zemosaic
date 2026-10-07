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
from .zemosaic_utils import load_image_with_optional_alpha
from .core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
    subset_fixed_normalization,
)
from .core.zegrid import assembly as za
from .core.zegrid import auto_layout as zal
from .core.zegrid import execution as zxe
from .core.zegrid import file_provider as zfp
from .core.zegrid import geometry as zg
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
    """

    def _log(msg, lvl="INFO"):
        if emit is None:
            return
        try:
            emit(msg, lvl)
        except Exception:
            pass

    floors = floors or zal.ScientificFloors()
    mw, mh = zal.median_projected_footprint(frames, canvas)
    if not (mw > 0 and mh > 0):
        raise zal.LayoutInfeasible("median projected footprint is degenerate")
    footprints = zal._footprints(frames, canvas)

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


def _run_cell_inmem(cache_dir, patch, config, progress_callback, fixed=None):
    """In-memory cell run: aligned frames resident, single-tile streaming executor.

    ZM-ZEGRID-R11: the in-memory path now goes through
    :func:`run_canonical_stack_streaming` with an :class:`InMemoryCanonicalProvider`
    (bit-equal to the engine by the R5/R6 contract) so it can consume the SAME
    fixed photometric gauge as the streaming path. ``fixed`` is the Cell-local
    :class:`FixedNormalization`; when None the per-Cell phase-1 is computed as
    before (legacy behaviour).
    """
    provider = zfp.MemmapCanonicalProvider(cache_dir)
    frame_ids = list(provider.frame_ids)
    images = []
    supports = []
    for i in range(provider.n_frames):
        rgb, sup = provider.get_raw_frame(i)
        images.append(np.array(rgb, dtype=np.float32, copy=True))
        supports.append(np.array(sup, dtype=bool, copy=True))
    # R15: release the memmap handles once the aligned frames are materialised
    # (Windows: the cache dir is deleted right after this cell).
    provider.close()
    inmem_provider = InMemoryCanonicalProvider(images, supports)
    request = zstream.build_streaming_request(config, inmem_provider.n_frames)
    result = run_canonical_stack_streaming(
        inmem_provider, request, tile_size=None, fixed=fixed
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
    lines.append(f"  {zin.describe_gpu_usage()}")
    lines.append("")
    lines.append("Ignored product settings (ZeGrid is CPU-only + no post-stack processing):")
    lines.extend(zin.ignored_settings_warning_lines(ignored_settings))
    lines.append("")
    lines.append("Accepted-but-ignored run_zegrid_mode arguments:")
    lines.extend(zin.describe_ignored_run_args(ignored_run_args or {}))
    lines.append("")
    lines.append(f"peak_rss_kib: {peak_rss_kib}")
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

    # Explicit GPU-usage + ignored-settings surfacing (nothing silently ignored).
    gpu_used = False  # ZeGrid engine is CPU-only.
    ignored_settings = zin.ignored_settings_present(zconfig)
    if ignored_settings:
        for line in zin.ignored_settings_warning_lines(ignored_settings):
            _emit(line, lvl="WARN", callback=progress_callback)
    _emit(
        f"ZeGrid: GPU usage — {zin.describe_gpu_usage()}",
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
    with timings.timed("layout"):
        layout = _choose_layout_mode_aware(
            canvas, descs, ram_budget=available, available_bytes=available,
            tile_size=STREAM_TILE_SIZE, pinned_layout=pinned_layout,
            emit=_emit_live,
        )
        nx, ny = layout["nx"], layout["ny"]
        cell_ctxs = []
        for row, col, _bounds in zg.build_layout(canvas, nx, ny).iter_cells(canvas):
            cell, patch, mem = zsw.build_cell_context(descs, canvas, row, col, nx, ny)
            cell_ctxs.append((row, col, cell, patch, mem))
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

    # Parallel workers (memory-aware: 2-4 by default, clamped by CPU + RAM).
    if workers is None:
        workers = zpar.choose_workers(None, available_memory_bytes())
    _emit(f"ZeGrid: parallel workers={workers} (CPU={os.cpu_count()}, avail={available / 2**30:.2f}GiB)",
          callback=progress_callback)

    cache_root = output_dir / CACHE_DIR_NAME
    cache_root.mkdir(parents=True, exist_ok=True)

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

    with timings.timed("gauge"):
        global_gauge, global_frame_ids = zphot.compute_global_gauge(
            descs, canvas, _gauge_decode, science_config, gauge_cache_dir, workers=workers,
            progress_callback=_gauge_progress,
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
        _safe_rmtree(gauge_cache_dir, progress_callback)

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
    stack_cells_done = 0
    _cache_rep = _reporter()
    _stack_rep = _reporter()
    _cache_rep.start("cache_build", total=cache_total_frames, unit="frames")
    _stack_rep.start("per_cell_stack", total=total_cells, unit="cells")
    for (row, col, cell, patch, mem) in cell_ctxs:
        cid = cell.cell_id
        idx = row * nx + col
        if not mem.patch_ids:
            _emit(f"ZeGrid: cell {cid} ({idx + 1}/{total_cells}) empty — no patch contributors",
                  callback=progress_callback)
            cell_records.append({"cell_id": cid, "status": "empty", "mode": None})
            stack_cells_done += 1
            _stack_rep.progress(stack_cells_done, item_id=cid)
            continue

        n = len(mem.patch_ids)
        area = patch.patch.width * patch.patch.height
        mode, bound = _pick_mode(n, area, patch.patch_shape_hw, available_memory_bytes())
        _emit(
            f"ZeGrid: cell {cid} ({idx + 1}/{total_cells}) N={n} area={area}px "
            f"mode={mode} bound={bound / 2**20:.1f}MiB",
            callback=progress_callback,
        )
        cache_dir = cache_root / cid

        # Build this cell's aligned cache (parallel reproject).
        t0 = time.perf_counter()
        try:
            manifest = _build_one_cell_cache(
                descs, canvas, cell, patch, mem, cache_dir, workers=workers
            )
        except Exception as exc:
            _emit(f"ZeGrid: cell {cid} cache build failed: {exc}", lvl="ERROR", callback=progress_callback)
            raise
        cache_frames_done += int(manifest.get("n_frames", 0))
        _cache_rep.progress(cache_frames_done, item_id=cid)
        timings.add("cache_build", time.perf_counter() - t0)
        cache_total_bytes += int(manifest.get("total_bytes", 0))
        cache_n_frames += int(manifest.get("n_frames", 0))
        cache_peak_bytes = max(cache_peak_bytes, int(manifest.get("total_bytes", 0)))

        # R11: subset the global gauge to THIS cell's frames (in cache order), so
        # the cell reuses the SAME per-frame coefficients/weights as every other
        # cell (a single photometric anchor). Open the provider ONLY long enough
        # to read the frame-id list, then CLOSE it (R15: release the memmap
        # handles before the cache is deleted below).
        cell_provider = zfp.MemmapCanonicalProvider(cache_dir)
        try:
            cell_frame_ids = list(cell_provider.frame_ids)
        finally:
            cell_provider.close()
        cell_fixed = subset_fixed_normalization(
            global_gauge, global_frame_ids, cell_frame_ids
        )

        t0 = time.perf_counter()
        try:
            if mode == "inmem":
                mt, sres = _run_cell_inmem(cache_dir, patch, science_config, progress_callback,
                                           fixed=cell_fixed)
            else:
                mt, sres = _run_cell_stream(cache_dir, patch, science_config, progress_callback,
                                            fixed=cell_fixed)
        except Exception as exc:
            # No silent fallback: a per-cell ZeGrid failure raises.
            _emit(f"ZeGrid: cell {cid} failed: {exc}", lvl="ERROR", callback=progress_callback)
            raise
        timings.add("per_cell_stack", time.perf_counter() - t0)
        stack_cells_done += 1
        _stack_rep.progress(stack_cells_done, item_id=cid)

        cores[cid] = za.crop_all_planes_to_core(mt)
        peak_rss_kib = max(peak_rss_kib, zsw.peak_rss_kib())

        # ZM-ZEGRID-R11 L1 (provenance honesty): the per-cell reference_frame_id is
        # the TRUE photometric anchor only when the cell CONTAINS the global
        # reference frame; otherwise it is a BOOKKEEPING placeholder (the cell's
        # highest-weight active frame) and the true anchor is
        # ``photometric_gauge.global_reference_frame_id``.
        prov = _reference_provenance(
            global_reference_frame_id, cell_frame_ids, sres.reference_frame_id
        )
        cell_records.append(
            {
                "cell_id": cid,
                "row": row,
                "col": col,
                "status": "complete",
                "mode": mode,
                "n_contributors": n,
                "reference_frame_id": sres.reference_frame_id,
                "reference_frame_role": prov["reference_frame_role"],
                "bookkeeping_reference_frame_id": prov["bookkeeping_reference_frame_id"],
                "excluded": [list(e) for e in sres.excluded],
                "bound_bytes": bound,
            }
        )

        # Per-cell temp reuse: delete the cell cache after stacking (bounded disk).
        # R15: best-effort (Windows WinError 32 on open handles -> WARN + continue).
        _safe_rmtree(cache_dir, progress_callback)

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
    }

    # Write legacy-compatible outputs.
    sci_path, cov_path, manifest_path = _write_outputs(
        assembled, canvas, nx, ny, output_dir, descs, {}, cell_records,
        layout, science_config, peak_rss_kib, cache_report, progress_callback,
        rejected=rejected, sip_mode=sip_mode,
        frames_loaded=len(frames_info),
        global_reference_frame_id=global_reference_frame_id,
        timings=timings, gpu_used=gpu_used, ignored_settings=ignored_settings,
        ignored_run_args=ignored_run_args,
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
    )

    _emit(
        f"ZeGrid: done — {sci_path.name} ({assembled.science.shape}) + coverage + run log, "
        f"complete={len(assembled.complete_cells)} incomplete={len(assembled.incomplete_cells)} "
        f"holes={assembled.hole_pixels}px peak_rss={peak_rss_kib}KiB "
        f"cache_total={cache_total_bytes / 2**20:.0f}MiB cache_peak={cache_peak_bytes / 2**20:.0f}MiB",
        lvl="SUCCESS",
        callback=progress_callback,
    )
    return sci_path


def _write_outputs(
    assembled, canvas, nx, ny, output_dir, descs, manifests, cell_records,
    layout, science_config, peak_rss_kib, cache_report, progress_callback,
    rejected=None, sip_mode="keep", frames_loaded=None,
    global_reference_frame_id=None,
    timings=None, gpu_used=False, ignored_settings=None, ignored_run_args=None,
):
    output_dir = Path(output_dir)
    science = np.asarray(assembled.science, dtype=np.float32)  # (H, W, 3)
    stack_depth = np.asarray(assembled.stack_depth, dtype=np.int32)  # (H, W)

    sci_header = _canvas_header(canvas, ndim=3, channels=3)
    sci_data = np.ascontiguousarray(np.moveaxis(science, -1, 0))  # (3, H, W)
    sci_path = output_dir / "mosaic_grid.fits"
    fits.PrimaryHDU(sci_data, header=sci_header).writeto(sci_path, overwrite=True)

    cov_header = _canvas_header(canvas, ndim=2)
    cov_header["BUNIT"] = ("count", "per-pixel stack depth (max over channels)")
    cov_path = output_dir / "mosaic_grid_coverage.fits"
    fits.PrimaryHDU(stack_depth, header=cov_header).writeto(cov_path, overwrite=True)

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
                  "reused_cells": (cache_report or {}).get("reused", []),
                  "rebuilt_cells": (cache_report or {}).get("rebuilt", [])},
        "timings": (timings.to_dict() if timings is not None else {}),
        "gpu": {"used": bool(gpu_used), "note": zin.GPU_USAGE_NOTE},
        "ignored_settings": (ignored_settings or {}),
        "ignored_run_args": (ignored_run_args or {}),
        "peak_rss_kib": peak_rss_kib,
        "outputs": {
            "science": sci_path.name,
            "coverage": cov_path.name,
            "run_log": RUN_LOG_NAME,
        },
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
    save_final_as_uint16: bool = False,
    legacy_rgb_cube: bool = False,
    grid_rgb_equalize: bool | None = True,
    use_gpu: bool | None = None,
    zconfig: object | None = None,
    workers: int | None = None,
) -> None:
    """Run the NEW ZeGrid engine over a ``stack_plan.csv`` (production entry).

    Normalization defaults to ``sky_mean``;
    weighting/rejection/combine/taper use the frozen ZeGrid science config.
    """
    _emit("ZeGrid engine activated (stack_plan.csv detected)", callback=progress_callback)
    if use_gpu:
        _emit(
            "ZeGrid engine is CPU-only (streaming/in-memory canonical); ignoring use_gpu=True",
            lvl="WARN",
            callback=progress_callback,
        )

    # ZM-ZEGRID-R12 F2: surface the run_zegrid_mode arguments ZeGrid accepts for
    # backward compatibility but does NOT honour (frozen science config + standard
    # FITS outputs). Nothing is silently dropped.
    ignored_run_args = {
        "stack_weight_method": stack_weight_method,
        "stack_reject_algo": stack_reject_algo,
        "stack_kappa_low": stack_kappa_low,
        "stack_kappa_high": stack_kappa_high,
        "winsor_limits": list(winsor_limits),
        "stack_final_combine": stack_final_combine,
        "apply_radial_weight": apply_radial_weight,
        "radial_feather_fraction": radial_feather_fraction,
        "radial_shape_power": radial_shape_power,
        "save_final_as_uint16": save_final_as_uint16,
        "legacy_rgb_cube": legacy_rgb_cube,
        "grid_rgb_equalize": grid_rgb_equalize,
        "use_gpu": bool(use_gpu),
    }
    for line in zin.describe_ignored_run_args(ignored_run_args):
        _emit(line, lvl="WARN", callback=progress_callback)

    csv_path = Path(input_folder).expanduser() / "stack_plan.csv"
    frames_info = _stack_plan.load_stack_plan(csv_path, progress_callback=progress_callback)
    if not frames_info:
        raise RuntimeError("ZeGrid failed: no frames loaded from stack_plan.csv")

    science_config = replace(ExecutorConfig().science_config(),
                             normalization=_resolve_normalization(stack_norm_method))
    _emit(
        f"ZeGrid science config: normalization={science_config.normalization} "
        f"(default={DEFAULT_NORMALIZATION}), weighting={science_config.weighting}, "
        f"rejection={science_config.rejection}, combine={science_config.combine}, "
        f"taper={science_config.taper}",
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
                        ignored_run_args=ignored_run_args)
        if altz_frames:
            _run_single(altz_frames, input_folder, base_out / "grid_ALTZ",
                        progress_callback=progress_callback,
                        science_config=science_config, zconfig=zconfig,
                        pinned_layout=pinned_layout, sip_mode=sip_mode, workers=workers,
                        ignored_run_args=ignored_run_args)
    else:
        _emit("ZeGrid: mount info missing or homogeneous — single pass",
              callback=progress_callback)
        _run_single(frames_info, input_folder, base_out,
                    progress_callback=progress_callback,
                    science_config=science_config, zconfig=zconfig,
                    pinned_layout=pinned_layout, sip_mode=sip_mode, workers=workers,
                    ignored_run_args=ignored_run_args)
