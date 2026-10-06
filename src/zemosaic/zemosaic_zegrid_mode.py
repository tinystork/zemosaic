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
from .core.canonical_streaming import run_canonical_stack_streaming
from .core.zegrid import assembly as za
from .core.zegrid import auto_layout as zal
from .core.zegrid import execution as zxe
from .core.zegrid import file_provider as zfp
from .core.zegrid import geometry as zg
from .core.zegrid import mosaic as zmosaic
from .core.zegrid import science_adapter as zs
from .core.zegrid import streaming as zstream
from .core.zegrid import sweep as zsw
from .core.zegrid.executor import ExecutorConfig

logger = logging.getLogger("ZeMosaicWorker").getChild("zegrid_mode")

ProgressCallback = Optional[Callable[[str, object, str], None]]

# Disk cache directory name (under the per-mount output folder). NEVER /tmp.
CACHE_DIR_NAME = "__zegrid_cache__"
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

def _build_frame_descriptors(frames_info, input_folder, progress_callback):
    """Convert legacy ``FrameInfo`` (WCS populated) to ZeGrid ``FrameDescriptor``.

    Each frame's WCS is QUALIFIED (undistorted 2-D celestial TAN). Unqualified
    frames are recorded (not silently dropped) and skipped; if NONE qualify a
    ``RuntimeError`` is raised (no silent degradation).
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
        reason = zg.qualify_wcs(fi.wcs)
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
                wcs_header=zg._serialize_wcs(fi.wcs),
                header_sha256="",
                instrument="",
            )
        )
    for r in rejected:
        _emit(
            f"Rejected frame (unqualified WCS): {r['path']} — {r['reason']}",
            lvl="WARN",
            callback=progress_callback,
        )
    if not descs:
        raise RuntimeError(
            "ZeGrid: no frames with a qualified (undistorted 2-D TAN) WCS; cannot build a mosaic"
        )
    descs.sort(key=lambda d: d.frame_id)
    return descs


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
):
    """MODE-AWARE RAM-aware layout: coarsest candidate whose cheaper mode fits.

    Reuses the R4 candidate enumeration (``REFINEMENT_FACTORS``, scientific
    floors, footprint membership) verbatim, but a candidate is FEASIBLE iff the
    CHEAPER-fitting per-cell mode's worst bound fits ``ram_budget``. Deterministic
    (identical enumeration order); returns a dict (layout + per-cell predictions).

    ``pinned_layout`` (``(nx, ny)`` or ``None``) BYPASSES the RAM-adaptive search
    and uses exactly that layout (scientific floors still enforced; an infeasible
    pinned layout raises). ``layout_source`` records ``"pinned"`` vs ``"auto"``.
    """
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
            raise zal.LayoutInfeasible(
                f"pinned layout {nx}x{ny} violates scientific floors: "
                f"min_patch_area={geom['min_patch_area']} (limit {floors.min_patch_area_px}), "
                f"halo_overhead={geom['halo_overhead']:.4f} (limit {floors.max_halo_overhead})"
            )
        cells = list(_iter_cell_bounds_mode_aware(canvas, footprints, nx, ny, halo_px, tile_size))
        max_n = max((c["n"] for c in cells), default=0)
        worst_bound = max((min(c["inmem_bound_bytes"], c["stream_bound_bytes"]) for c in cells), default=0)
        if ram_budget is not None and worst_bound > ram_budget:
            raise zal.LayoutInfeasible(
                f"pinned layout {nx}x{ny} cannot fit any mode within budget: "
                f"cheaper bound {worst_bound / 2**20:.1f} MiB > budget {ram_budget / 2**20:.1f} MiB"
            )
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
            break  # floors are monotonic in refinement
        worst = _scan_layout_mode_aware(canvas, footprints, nx, ny, halo_px, tile_size)
        memory_ok = (ram_budget is None) or (worst["cheaper_bound_bytes"] <= ram_budget)
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
        raise zal.LayoutInfeasible(
            "RAM budget cannot satisfy scientific floors (mode-aware): " + "; ".join(reasons)
        )

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


def _run_cell_inmem(cache_dir, patch, config, progress_callback):
    provider = zfp.MemmapCanonicalProvider(cache_dir)
    images = []
    supports = []
    for i in range(provider.n_frames):
        rgb, sup = provider.get_raw_frame(i)
        images.append(np.array(rgb, dtype=np.float32, copy=True))
        supports.append(np.array(sup, dtype=bool, copy=True))
    sres = zs.run_minitile_stack(images, supports, list(provider.frame_ids), config)
    mt = za.extract_minitile(patch, sres)
    return mt, sres


def _run_cell_stream(cache_dir, patch, config, progress_callback, tile_size=STREAM_TILE_SIZE):
    provider = zfp.MemmapCanonicalProvider(cache_dir)
    request = zstream.build_streaming_request(config, provider.n_frames)
    result = run_canonical_stack_streaming(provider, request, tile_size=tile_size)
    order = list(provider.frame_ids)
    ref_idx = int(result.provenance["reference"]["index"])
    reference_frame_id = order[ref_idx] if 0 <= ref_idx < len(order) else None
    excluded = tuple(
        (order[idx] if 0 <= idx < len(order) else f"<index {idx}>", stage, reason)
        for idx, stage, reason in result.provenance["excluded_frames"]
    )
    sres = zs.MiniTileScienceResult(
        result=result,
        reference_frame_id=reference_frame_id,
        excluded=excluded,
        frame_order=tuple(order),
    )
    mt = za.extract_minitile(patch, sres)
    return mt, sres


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

def _run_single(
    frames_info,
    input_folder,
    output_dir,
    *,
    progress_callback,
    science_config,
    zconfig,
    pinned_layout=None,
):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    _emit(f"ZeGrid: setup — {len(frames_info)} frame(s) -> {output_dir}", callback=progress_callback)
    descs = _build_frame_descriptors(frames_info, input_folder, progress_callback)
    canvas = zg.build_canvas(descs)
    _emit(
        f"ZeGrid: canvas {canvas.width}x{canvas.height} "
        f"(resolution {canvas.resolution_deg:.3g} deg/px)",
        callback=progress_callback,
    )

    # LAYOUT — MODE-AWARE RAM-aware Auto layout (in-memory OR streaming bound),
    # budgeted from psutil (portable). Pinnable via ``zegrid_layout``.
    available = available_memory_bytes()
    layout = _choose_layout_mode_aware(
        canvas, descs, ram_budget=available, available_bytes=available,
        tile_size=STREAM_TILE_SIZE, pinned_layout=pinned_layout,
    )
    nx, ny = layout["nx"], layout["ny"]
    _emit(
        f"ZeGrid: layout {nx}x{ny} ({layout['cell_count']} cells, "
        f"source={layout.get('layout_source', 'auto')}) from RAM budget "
        f"{available / 2**30:.2f} GiB; worst-cell cheaper-mode bound "
        f"{layout['predicted_bound_bytes'] / 2**20:.1f} MiB (maxN={layout['max_contributors']})",
        callback=progress_callback,
    )
    for w in layout["warnings"]:
        _emit(f"ZeGrid: layout warning — {w}", lvl="WARN", callback=progress_callback)

    cell_ctxs = []
    for row, col, _bounds in zg.build_layout(canvas, nx, ny).iter_cells(canvas):
        cell, patch, mem = zsw.build_cell_context(descs, canvas, row, col, nx, ny)
        cell_ctxs.append((row, col, cell, patch, mem))

    # FRAME-MAJOR decode+cache.
    _emit(f"ZeGrid: decode+cache (frame-major, O(1 frame) memory)", callback=progress_callback)
    cache_root = output_dir / CACHE_DIR_NAME
    cache_dirs, manifests, cache_report = _build_aligned_cache_frame_major(
        descs, canvas, cell_ctxs, cache_root, progress_callback
    )

    # PER-CELL mode policy -> MiniTile -> core planes.
    cores = {}
    cell_records = []
    total_cells = nx * ny
    peak_rss_kib = zsw.peak_rss_kib()
    for (row, col, cell, patch, mem) in cell_ctxs:
        cid = cell.cell_id
        idx = row * nx + col
        if not mem.patch_ids:
            _emit(f"ZeGrid: cell {cid} ({idx + 1}/{total_cells}) empty — no patch contributors",
                  callback=progress_callback)
            cell_records.append({"cell_id": cid, "status": "empty", "mode": None})
            continue

        n = len(mem.patch_ids)
        area = patch.patch.width * patch.patch.height
        mode, bound = _pick_mode(n, area, patch.patch_shape_hw, available_memory_bytes())
        _emit(
            f"ZeGrid: cell {cid} ({idx + 1}/{total_cells}) N={n} area={area}px "
            f"mode={mode} bound={bound / 2**20:.1f}MiB",
            callback=progress_callback,
        )
        cache_dir = cache_dirs.get(cid)
        if cache_dir is None or not (cache_dir / "manifest.json").exists():
            _emit(f"ZeGrid: cell {cid} cache missing; treating as empty", lvl="WARN",
                  callback=progress_callback)
            cell_records.append({"cell_id": cid, "status": "empty", "mode": mode})
            continue

        try:
            if mode == "inmem":
                mt, sres = _run_cell_inmem(cache_dir, patch, science_config, progress_callback)
            else:
                mt, sres = _run_cell_stream(cache_dir, patch, science_config, progress_callback)
        except Exception as exc:
            # No silent fallback: a per-cell ZeGrid failure raises.
            _emit(f"ZeGrid: cell {cid} failed: {exc}", lvl="ERROR", callback=progress_callback)
            raise

        cores[cid] = za.crop_all_planes_to_core(mt)
        peak_rss_kib = max(peak_rss_kib, zsw.peak_rss_kib())
        cell_records.append(
            {
                "cell_id": cid,
                "row": row,
                "col": col,
                "status": "complete",
                "mode": mode,
                "n_contributors": n,
                "reference_frame_id": sres.reference_frame_id,
                "excluded": [list(e) for e in sres.excluded],
                "bound_bytes": bound,
            }
        )

    # ASSEMBLY.
    _emit("ZeGrid: assembly (R3 assemble_canvas)", callback=progress_callback)
    assembled = zmosaic.assemble_canvas(canvas, nx, ny, cores)
    peak_rss_kib = max(peak_rss_kib, zsw.peak_rss_kib())

    # I2 — no covered pixels -> explicit abort (never an all-NaN mosaic).
    if assembled.coverage_pixels == 0:
        _emit(
            "ZeGrid: canvas has NO covered pixels; aborting (would produce an all-NaN mosaic)",
            lvl="ERROR",
            callback=progress_callback,
        )
        raise RuntimeError("ZeGrid: no covered pixels; cannot assemble a mosaic")

    # Write legacy-compatible outputs.
    sci_path, cov_path, manifest_path = _write_outputs(
        assembled, canvas, nx, ny, output_dir, descs, manifests, cell_records,
        layout, science_config, peak_rss_kib, cache_report, progress_callback,
    )
    _emit(
        f"ZeGrid: done — {sci_path.name} ({assembled.science.shape}) + coverage, "
        f"complete={len(assembled.complete_cells)} incomplete={len(assembled.incomplete_cells)} "
        f"holes={assembled.hole_pixels}px peak_rss={peak_rss_kib}KiB",
        lvl="SUCCESS",
        callback=progress_callback,
    )
    return sci_path


def _write_outputs(
    assembled, canvas, nx, ny, output_dir, descs, manifests, cell_records,
    layout, science_config, peak_rss_kib, cache_report, progress_callback,
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

    cache_total_bytes = 0
    cache_files = 0
    for _cid, m in (manifests or {}).items():
        if isinstance(m, dict):
            cache_total_bytes += int(m.get("total_bytes", 0))
            cache_files += int(m.get("n_frames", 0))

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
            "output science depends on the layout (per-cell sky_mean normalization "
            "is computed over the haloed patch); coverage (stack depth) is layout-"
            "invariant. Pin the layout via the 'zegrid_layout' config key (e.g. "
            "'6x5') for reproducible output."
        ),
        "n_frames": len(descs),
        "frame_ids": [d.frame_id.logical_path for d in descs],
        "complete_cells": assembled.complete_cells,
        "incomplete_cells": assembled.incomplete_cells,
        "hole_pixels": assembled.hole_pixels,
        "coverage_pixels": assembled.coverage_pixels,
        "cells": cell_records,
        "cache": {"dir": CACHE_DIR_NAME, "total_bytes": cache_total_bytes,
                  "n_frame_entries": cache_files,
                  "reused_cells": (cache_report or {}).get("reused", []),
                  "rebuilt_cells": (cache_report or {}).get("rebuilt", [])},
        "peak_rss_kib": peak_rss_kib,
        "outputs": {
            "science": sci_path.name,
            "coverage": cov_path.name,
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
                        pinned_layout=pinned_layout)
        if altz_frames:
            _run_single(altz_frames, input_folder, base_out / "grid_ALTZ",
                        progress_callback=progress_callback,
                        science_config=science_config, zconfig=zconfig,
                        pinned_layout=pinned_layout)
    else:
        _emit("ZeGrid: mount info missing or homogeneous — single pass",
              callback=progress_callback)
        _run_single(frames_info, input_folder, base_out,
                    progress_callback=progress_callback,
                    science_config=science_config, zconfig=zconfig,
                    pinned_layout=pinned_layout)
