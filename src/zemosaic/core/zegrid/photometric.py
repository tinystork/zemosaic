"""ZM-ZEGRID-R11/R12 — global photometric gauge computation (bounded streaming).

Computes ONE reference frame + per-frame normalization coefficients/weights over
the frames' FULL valid footprint on the canvas, so every Cell of the mosaic
shares a single photometric anchor. This removes the systematic inter-cell level
steps (the "patchwork") caused by each Cell previously selecting ITS OWN
reference frame and computing its ``sky_mean`` offsets against it.

This module is the GLOBAL half of the gauge. The per-Cell consumption lives in
``canonical_streaming.run_canonical_stack_streaming(..., fixed=...)`` via
:func:`canonical_streaming.compute_fixed_normalization` and
:func:`canonical_streaming.subset_fixed_normalization`.

Design (R12 rework-1 — BOUNDED streaming phase-1, NO disk cache)
----------------------------------------------------------------
R11 materialised ONE FULL-CANVAS aligned plane per frame into ``<cache>/__gauge__``
and only deleted it AFTER use, so its transient peak was ``O(N x canvas)``
(66.8 GiB for Caldwell 11's 306 frames / 142.5 GiB for 653 / 178 GiB for 816 on a
4117x4377 canvas) — larger than the host's free disk, so the real run could not
complete. R12 rework-1 removes that cache entirely and computes the SAME gauge
frame-by-frame over BOUNDED regions, reusing the FROZEN phase-1 functions
verbatim on a 2-frame ``[R, i]`` batch:

* **Per-frame valid counts** (for reference selection) are computed by
  reprojecting each frame to ITS OWN projected-footprint bbox (a bounded,
  ~1/4-canvas region), never a full-canvas plane.
* **Reference selection** reuses the frozen ``select_canonical_reference``
  (greatest valid count, ties to the lowest sorted-FrameId index).
* The **reference** is reprojected ONCE to the full canvas (ONE plane, held in
  memory, shared to workers via fork copy-on-write) so it can be cropped to any
  frame's footprint bbox.
* **Per-frame normalization + weighting** reuses the frozen
  ``prepare_canonical_inputs`` / ``normalize_canonical_images`` /
  ``compute_canonical_quality_weights`` on a 2-frame ``[R_crop, i_crop]`` batch,
  where BOTH are cropped to frame i's footprint bbox. Cropping to i's footprint
  bbox (not just the R-i intersection) is what keeps the QUALITY WEIGHT
  bit-identical: the weight is the noise sigma over frame i's FULL valid
  footprint (the frozen weighting does not restrict to the intersection), while
  the sky_mean offset is computed over the common region (a subset of the bbox).

Peak RAM is O(1-2 planes) (~0.5 GiB: R's full canvas + one footprint bbox), and
peak DISK is ~0. The coefficients/weights/active flags/exclusions are BIT-
IDENTICAL to the R11 full-canvas gauge because every scalar the frozen functions
compute is a deterministic function of a 1-D sequence ``array[valid_mask]``
(row-major), and a bbox crop that is a SUPERSET of the relevant valid region
yields the identical sequence in the identical order.
"""

from __future__ import annotations

import copy
from typing import Callable, Sequence

import numpy as np
from astropy.wcs import WCS

from zemosaic.core.canonical_stacking import (
    CanonicalInputBatch,
    CanonicalStackFailure,
    FrameExclusion,
    compute_canonical_quality_weights,
    normalize_canonical_images,
    prepare_canonical_inputs,
    select_canonical_reference,
)
from zemosaic.core.canonical_streaming import FixedNormalization
from zemosaic.core.zegrid import execution as zxe
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import parallel as zpar

__all__ = ["compute_global_gauge"]

# Canvas-pixel margin around a frame's projected-footprint bbox. This makes the
# bbox a SUPERSET of the reprojected support (bilinear kernel + SIP margin), so
# the bbox count/offset/weight are bit-exact to the full-canvas computation.
_FOOTPRINT_MARGIN_PX = 8

# Module-level globals holding the reference frame's full-canvas alignment,
# handed to the forked parallel workers via copy-on-write inheritance (never
# pickled — they can be large). Set by ``compute_global_gauge`` around the
# per-frame parallel pass and cleared afterwards.
_GAUGE_R_FULL = None
_GAUGE_R_SUP = None


def _footprint_bbox(f, canvas) -> tuple:
    """Canvas bbox ``(y0, y1, x0, x1)`` bounding a frame's footprint + margin."""
    poly = zg._source_polygon(f.shape_hw, f.wcs(), canvas.wcs())
    minx, miny, maxx, maxy = poly.bounds
    m = _FOOTPRINT_MARGIN_PX
    x0 = max(0, int(np.floor(minx)) - m)
    y0 = max(0, int(np.floor(miny)) - m)
    x1 = min(canvas.width, int(np.ceil(maxx)) + m + 1)
    y1 = min(canvas.height, int(np.ceil(maxy)) + m + 1)
    return (y0, y1, x0, x1)


def _bbox_wcs_header(canvas, bbox) -> str:
    """Sub-canvas WCS for a bbox (CRPIX shifted by the bbox origin)."""
    y0, _y1, x0, _x1 = bbox
    w = copy.deepcopy(canvas.wcs())
    w.wcs.crpix -= np.array([x0, y0])
    w.array_shape = (bbox[1] - bbox[0], bbox[3] - bbox[2])
    return w.to_header(relax=True).tostring()


def _decode_to_chw(decode_fn, f) -> np.ndarray:
    hwc = np.asarray(decode_fn(f), dtype=np.float32)
    if hwc.ndim == 2:
        hwc = np.stack([hwc, hwc, hwc], axis=-1)
    return np.ascontiguousarray(np.moveaxis(hwc, -1, 0))


def _reproject_full(decode_fn, f, canvas):
    """Reproject one frame onto the FULL canvas (one plane)."""
    chw = _decode_to_chw(decode_fn, f)
    rgb, geom = zxe.reproject_cropped(
        chw, f.wcs(), canvas.wcs(), (canvas.height, canvas.width)
    )
    return rgb, geom


def _counts_worker(task):
    (decode_fn, f, bbox_wcs_header, bbox_shape) = task
    chw = _decode_to_chw(decode_fn, f)
    rgb, geom = zxe.reproject_cropped(chw, f.wcs(), WCS(bbox_wcs_header), tuple(bbox_shape))
    b = prepare_canonical_inputs([rgb], [geom])
    return int(b.frame_valid_counts[0])


def _pair_worker(task):
    (index, decode_fn, f, bbox, bbox_wcs_header, bbox_shape, norm_token, weight_token) = task
    chw = _decode_to_chw(decode_fn, f)
    rgb, geom = zxe.reproject_cropped(chw, f.wcs(), WCS(bbox_wcs_header), tuple(bbox_shape))
    y0, y1, x0, x1 = bbox
    r_crop = np.ascontiguousarray(_GAUGE_R_FULL[y0:y1, x0:x1])
    r_sup = _GAUGE_R_SUP[y0:y1, x0:x1]

    batch2 = prepare_canonical_inputs([r_crop, rgb], [r_sup, geom])
    norm2 = normalize_canonical_images(batch2, norm_token, reference_index=0)

    norm_exc = []
    for e in norm2.exclusions:
        if e.index == 1:
            norm_exc.append({"stage": e.stage, "reason": e.reason, "detail": e.detail})

    raw_weight = float("nan")
    noise_sigma = float("nan")
    fwhm = float("nan")
    weight_active = bool(norm2.active_frames[1])
    weight_exc = []
    if weight_token != "none" and bool(norm2.active_frames[1]):
        weight2 = compute_canonical_quality_weights(norm2, weight_token)
        raw_weight = float(weight2.raw_weights[1])
        noise_sigma = float(weight2.noise_sigma[1])
        fwhm = float(weight2.fwhm[1])
        weight_active = bool(weight2.active_frames[1])
        if not weight_active:
            for e in weight2.exclusions:
                if e.index == 1 and e.stage == "weighting":
                    weight_exc.append({"stage": e.stage, "reason": e.reason, "detail": e.detail})

    return {
        "index": index,
        "coeff": np.array(norm2.coefficients[1], dtype=np.float64, copy=True),
        "norm_active": bool(norm2.active_frames[1]),
        "norm_exc": norm_exc,
        "raw_weight": raw_weight,
        "noise_sigma": noise_sigma,
        "fwhm": fwhm,
        "weight_active": weight_active,
        "weight_exc": weight_exc,
    }


def compute_global_gauge(
    frames: Sequence[zg.FrameDescriptor],
    canvas: zg.GlobalCanvas,
    decode_fn: Callable[[zg.FrameDescriptor], np.ndarray],
    config,
    cache_dir=None,
    *,
    reuse_cache: bool = True,
    workers: int = 1,
) -> tuple[FixedNormalization, tuple[str, ...]]:
    """Compute the global photometric gauge over the full canvas (bounded).

    Returns ``(gauge, frame_ids)`` where ``frame_ids`` is the frame order the
    gauge covers (sorted by FrameId). ``decode_fn`` decodes one frame descriptor
    to an HWC float32 array (the product decoder is passed in so this module stays
    decoupled from the product layer). ``config`` is the frozen
    :class:`~zemosaic.core.zegrid.science_adapter.MiniTileScienceConfig`.

    ``cache_dir`` / ``reuse_cache`` are accepted for backward compatibility but
    are IGNORED: the R12 rework-1 gauge is a bounded streaming phase-1 with NO
    disk cache (peak disk ~0, peak RAM O(1-2 planes)).
    """
    ordered = sorted(frames, key=lambda f: f.frame_id)
    frame_ids = [f.frame_id.logical_path for f in ordered]
    n = len(ordered)
    c = 3

    norm_token = config.normalization
    weight_token = config.weighting

    # --- per-frame valid counts over each frame's own footprint bbox (bounded) ---
    bboxes = [_footprint_bbox(f, canvas) for f in ordered]
    count_tasks = [
        (
            decode_fn,
            f,
            _bbox_wcs_header(canvas, bboxes[i]),
            (bboxes[i][1] - bboxes[i][0], bboxes[i][3] - bboxes[i][2]),
        )
        for i, f in enumerate(ordered)
    ]
    counts = np.asarray(zpar.pmap(_counts_worker, count_tasks, workers), dtype=np.int64)

    # --- reference selection (reuse the frozen function; exact argmax of counts) ---
    probe = CanonicalInputBatch(
        images=np.empty((0,), dtype=np.float32),
        valid_mask=np.empty((0,), dtype=bool),
        original_mono=False,
        frame_valid_counts=counts,
        n_frames=n,
        height=canvas.height,
        width=canvas.width,
        channels=c,
        original_ndim=3,
        original_shape=(canvas.height, canvas.width, c),
    )
    ref_idx = select_canonical_reference(probe, None)

    # --- reference full-canvas alignment (ONE plane) + reference weighting ---
    r_full, r_sup = _reproject_full(decode_fn, ordered[ref_idx], canvas)
    ref_norm = normalize_canonical_images(
        prepare_canonical_inputs([r_full], [r_sup]), norm_token, reference_index=0
    )
    ref_weight = compute_canonical_quality_weights(ref_norm, weight_token)

    coefficients = np.full((n, c, 2), np.nan, dtype=np.float64)
    norm_active = np.zeros(n, dtype=bool)
    raw_weights = np.full(n, np.nan, dtype=np.float64)
    noise_sigma = np.full(n, np.nan, dtype=np.float64)
    fwhm = np.full(n, np.nan, dtype=np.float64)
    norm_exclusions = []
    weight_exclusions = []

    coefficients[ref_idx, :, :] = (1.0, 0.0)
    norm_active[ref_idx] = bool(counts[ref_idx] > 0)
    if weight_token != "none":
        raw_weights[ref_idx] = float(ref_weight.raw_weights[0])
        noise_sigma[ref_idx] = float(ref_weight.noise_sigma[0])
        fwhm[ref_idx] = float(ref_weight.fwhm[0])

    # --- per-frame normalization + weighting (bounded, parallel) ---
    global _GAUGE_R_FULL, _GAUGE_R_SUP
    _GAUGE_R_FULL = r_full
    _GAUGE_R_SUP = r_sup
    try:
        pair_tasks = [
            (
                i,
                decode_fn,
                ordered[i],
                bboxes[i],
                _bbox_wcs_header(canvas, bboxes[i]),
                (bboxes[i][1] - bboxes[i][0], bboxes[i][3] - bboxes[i][2]),
                norm_token,
                weight_token,
            )
            for i in range(n)
            if i != ref_idx
        ]
        pair_results = zpar.pmap(_pair_worker, pair_tasks, workers)
    finally:
        _GAUGE_R_FULL = None
        _GAUGE_R_SUP = None

    for res in pair_results:
        i = int(res["index"])
        coefficients[i] = res["coeff"]
        norm_active[i] = bool(res["norm_active"])
        for e in res["norm_exc"]:
            norm_exclusions.append(
                FrameExclusion(index=i, stage=e["stage"], reason=e["reason"], detail=e["detail"])
            )
        if weight_token != "none" and norm_active[i]:
            raw_weights[i] = res["raw_weight"]
            noise_sigma[i] = res["noise_sigma"]
            fwhm[i] = res["fwhm"]
            if not res["weight_active"]:
                for e in res["weight_exc"]:
                    weight_exclusions.append(
                        FrameExclusion(
                            index=i, stage=e["stage"], reason=e["reason"], detail=e["detail"]
                        )
                    )

    # --- weighting-level active + weight normalization (reuse _phase1 semantics) ---
    if weight_token == "none":
        weights = np.ones(n, dtype=np.float64)
        weights[~norm_active] = 0.0
        raw_weights[norm_active] = 1.0
        weight_active = norm_active.copy()
    else:
        weight_active = norm_active.copy()
        for e in weight_exclusions:
            weight_active[e.index] = False
        if not weight_active.any():
            raise CanonicalStackFailure("no frame remains after quality weighting")
        max_raw = float(np.max(raw_weights[weight_active]))
        if not (np.isfinite(max_raw) and max_raw > 0.0):
            raise CanonicalStackFailure("no positive quality weight remains after weighting")
        weights = np.zeros(n, dtype=np.float64)
        weights[weight_active] = raw_weights[weight_active] / max_raw

    gauge = FixedNormalization(
        reference_index=int(ref_idx),
        coefficients=np.ascontiguousarray(coefficients),
        norm_active=np.ascontiguousarray(norm_active),
        weights=np.ascontiguousarray(weights),
        weight_active=np.ascontiguousarray(weight_active),
        exclusions=tuple(norm_exclusions) + tuple(weight_exclusions),
    )
    return gauge, tuple(frame_ids)
