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

from typing import Callable, Sequence

import numpy as np

from zemosaic.core.canonical_streaming import (
    FixedNormalization,
    compute_fixed_normalization,
)
from zemosaic.core.zegrid import execution as zxe
from zemosaic.core.zegrid import file_provider as zfp
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid.streaming import build_streaming_request

__all__ = ["compute_global_gauge"]


def compute_global_gauge(
    frames: Sequence[zg.FrameDescriptor],
    canvas: zg.GlobalCanvas,
    decode_fn: Callable[[zg.FrameDescriptor], np.ndarray],
    config,
    cache_dir,
    *,
    reuse_cache: bool = True,
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
        builder = zfp.AlignedCacheBuilder(str(cache_dir))
        for f in ordered:
            hwc = np.asarray(decode_fn(f), dtype=np.float32)
            if hwc.ndim == 2:
                hwc = np.stack([hwc, hwc, hwc], axis=-1)
            chw = np.ascontiguousarray(np.moveaxis(hwc, -1, 0))
            rgb, geom = zxe.reproject_cropped(
                chw, f.wcs(), canvas.wcs(), (canvas.height, canvas.width)
            )
            builder.add(f.frame_id.logical_path, rgb, geom)
            del hwc, chw, rgb, geom
        builder.finish()

    provider = zfp.MemmapCanonicalProvider(str(cache_dir))
    request = build_streaming_request(config, provider.n_frames)
    gauge = compute_fixed_normalization(provider, request)
    return gauge, tuple(provider.frame_ids)
