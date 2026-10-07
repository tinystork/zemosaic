"""ZM-ZEGRID-R11 — global photometric gauge computation.

Computes ONE reference frame + per-frame normalization coefficients/weights over
the frames' FULL valid footprint on the canvas, so every Cell of the mosaic
shares a single photometric anchor. This removes the systematic inter-cell level
steps (the "patchwork") caused by each Cell previously selecting ITS OWN
reference frame and computing its ``sky_mean`` offsets against it.

This module is the GLOBAL half of the gauge. The per-Cell consumption lives in
``canonical_streaming.run_canonical_stack_streaming(..., fixed=...)`` via
:func:`canonical_streaming.compute_fixed_normalization` and
:func:`canonical_streaming.subset_fixed_normalization`.

Design
------
* The gauge is computed by running the EXISTING streaming phase-1
  (``canonical_streaming._phase1``) over a FULL-CANVAS alignment of every frame
  (a "whole-mosaic cell"), so the reference selection, normalization
  coefficients, quality weights and exclusions are all produced by the frozen
  canonical functions verbatim — no re-implementation.
* The full-canvas alignment is built FRAME-MAJOR (decode each frame once, then
  reproject the full source onto the full canvas) into a disk-backed aligned
  cache, reusing the R6 :class:`file_provider.AlignedCacheBuilder` /
  :class:`file_provider.MemmapCanonicalProvider`. Peak memory is O(1 frame); the
  cache is resumable (rebuilt only when missing/incomplete).
* The reference is chosen deterministically by the frozen
  ``select_canonical_reference``: the frame with the greatest valid support over
  the canvas, ties resolved to the lowest sorted-FrameId index (see the R11
  tests).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from astropy.wcs import WCS

from zemosaic.core.canonical_streaming import (
    FixedNormalization,
    compute_fixed_normalization,
)
from zemosaic.core.zegrid import execution as zxe
from zemosaic.core.zegrid import file_provider as zfp
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid.streaming import build_streaming_request

__all__ = ["compute_global_gauge", "gauge_frame_worker"]


def gauge_frame_worker(task):
    """Module-level (picklable) worker: decode + full-canvas reproject one frame.

    Task shape: ``(decode_fn, frame_desc, canvas_wcs_header, canvas_hw,
    rgb_path, sup_path, index, frame_id)``. ``decode_fn`` must be a module-level
    (picklable-by-reference) function of one :class:`FrameDescriptor`. Reproduces
    the serial full-canvas alignment bit-for-bit (decode -> ndim-2 stack ->
    moveaxis -> ``reproject_cropped``).
    """
    (
        decode_fn,
        frame_desc,
        canvas_wcs_header,
        canvas_hw,
        rgb_path,
        sup_path,
        index,
        frame_id,
    ) = task
    hwc = np.asarray(decode_fn(frame_desc), dtype=np.float32)
    if hwc.ndim == 2:
        hwc = np.stack([hwc, hwc, hwc], axis=-1)
    chw = np.ascontiguousarray(np.moveaxis(hwc, -1, 0))
    rgb, geom = zxe.reproject_cropped(
        chw, frame_desc.wcs(), WCS(canvas_wcs_header), tuple(canvas_hw)
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


def compute_global_gauge(
    frames: Sequence[zg.FrameDescriptor],
    canvas: zg.GlobalCanvas,
    decode_fn: Callable[[zg.FrameDescriptor], np.ndarray],
    config,
    cache_dir,
    *,
    reuse_cache: bool = True,
    workers: int = 1,
) -> tuple[FixedNormalization, tuple[str, ...]]:
    """Compute the global photometric gauge over the full canvas.

    Returns ``(gauge, frame_ids)`` where ``frame_ids`` is the frame order the
    gauge covers (sorted by FrameId). ``decode_fn`` decodes one frame descriptor
    to an HWC float32 array (the product decoder is passed in so this module stays
    decoupled from the product layer). ``config`` is the frozen
    :class:`~zemosaic.core.zegrid.science_adapter.MiniTileScienceConfig` (the same
    tokens the Cells use). ``cache_dir`` is a disk-backed aligned cache (never
    ``/tmp``; resumable via ``reuse_cache``).
    """
    ordered = sorted(frames, key=lambda f: f.frame_id)
    frame_ids = [f.frame_id.logical_path for f in ordered]

    if not (reuse_cache and zfp.cache_is_complete(str(cache_dir), frame_ids)):
        cache_dir = Path(str(cache_dir))
        if cache_dir.exists():
            import shutil

            shutil.rmtree(str(cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        canvas_wcs_header = canvas.wcs_header
        canvas_hw = (canvas.height, canvas.width)
        tasks = []
        for idx, f in enumerate(ordered):
            rgb_path = cache_dir / f"frame_{idx:04d}_rgb.npy"
            sup_path = cache_dir / f"frame_{idx:04d}_support.npy"
            tasks.append(
                (
                    decode_fn,
                    f,
                    canvas_wcs_header,
                    canvas_hw,
                    str(rgb_path),
                    str(sup_path),
                    idx,
                    f.frame_id.logical_path,
                )
            )
        results = zpar.pmap(gauge_frame_worker, tasks, workers)
        results.sort(key=lambda r: r["index"])
        meta = zfp._meta_from_hwc(canvas.height, canvas.width, 3)
        zfp._write_manifest(
            cache_dir, meta, results, frame_ids,
            sum(r["rgb_bytes"] + r["support_bytes"] for r in results),
        )

    provider = zfp.MemmapCanonicalProvider(str(cache_dir))
    request = build_streaming_request(config, provider.n_frames)
    gauge = compute_fixed_normalization(provider, request)
    return gauge, tuple(provider.frame_ids)
