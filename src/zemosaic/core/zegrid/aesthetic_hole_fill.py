"""Shared aesthetic hole-fill helper (factored from ``zemosaic_worker``).

Rework-3 (ZM-ZEGRID-R23): the ZeGrid finishing path must reuse the EXISTING
aesthetic hole-fill semantics/settings rather than a divergent copy. This module
holds the single canonical implementation, imported by BOTH:

* ``zemosaic_worker._apply_aesthetic_hole_fill`` (Classic path, thin wrapper), and
* ``core.zegrid.final_mosaic_finishing`` (ZeGrid aesthetic path).

The implementation is a faithful port of the Classic ``_apply_aesthetic_hole_fill``
with ONE deliberate fix (per rework-3 H2):

* **``only_near_seams=true`` must not fill non-target holes.** The original
  helper assigned ``src_nonan`` to the whole plane (the ``else`` branch of the
  final ``np.where``), so every non-target NaN hole — including holes far from
  any seam when ``only_near_seams`` restricts the target mask — got silently
  filled with the smoothed background. Here the non-target branch preserves the
  ORIGINAL pixel value (NaN stays NaN). Only the configured target mask is
  filled. This matches the documented Classic intent ("fill only near seams").

Everything else (blend/feather, radius clamp, protect-stars/detail masking, the
distance-transform target selection) is preserved verbatim.

Aesthetic hole fill is visual-only: it runs ONLY when the caller passes
``enabled=True`` (the existing ``aesthetic_hole_fill_enabled`` setting) and never
touches the raw science reference.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np


def gaussian_blur_2d_float32(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    """Small utility blur for preview-time visual filtering (best-effort).

    Faithful copy of ``zemosaic_worker._gaussian_blur_2d_float32`` (cv2 ->
    scipy.ndimage.gaussian_filter fallback -> passthrough). Kept here so the
    shared hole-fill helper is self-contained.
    """
    try:
        sigma = float(sigma_px)
    except Exception:
        sigma = 0.0
    if sigma <= 1e-6:
        return np.asarray(arr, dtype=np.float32)

    src = np.asarray(arr, dtype=np.float32)

    try:
        import cv2  # type: ignore

        k = max(3, int(2 * round(sigma * 1.5) + 1))
        return cv2.GaussianBlur(src, (k, k), sigmaX=sigma).astype(np.float32, copy=False)
    except Exception:
        pass

    try:
        from scipy.ndimage import gaussian_filter  # type: ignore

        return gaussian_filter(src, sigma=float(sigma), truncate=4.0).astype(np.float32, copy=False)
    except Exception:
        return src


def apply_aesthetic_hole_fill(
    mosaic_hwc: np.ndarray | None,
    *,
    alpha_mask: np.ndarray | None = None,
    coverage_hw: np.ndarray | None = None,
    enabled: bool = False,
    max_radius_px: int = 64,
    blend: float = 0.70,
    only_near_seams: bool = True,
    protect_stars_details: bool = True,
    logger: Any | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Visual-only local hole completion for the aesthetic branch.

    Canonical implementation shared by the Classic worker and the ZeGrid
    finishing path. Returns ``(out_array, info_dict)``; ``out_array`` is the
    input unchanged (or ``None``) when disabled or not RGB.
    """
    info: dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled" if not enabled else "",
        "filled_px": 0,
        "hole_px": 0,
        "max_radius_px": int(max_radius_px),
        "blend": float(blend),
        "only_near_seams": bool(only_near_seams),
        "protect_stars_details": bool(protect_stars_details),
        "protected_frac": 0.0,
    }
    if not enabled or mosaic_hwc is None or not isinstance(mosaic_hwc, np.ndarray):
        return mosaic_hwc, info
    if mosaic_hwc.ndim != 3 or mosaic_hwc.shape[-1] != 3:
        info["reason"] = "non_rgb"
        return mosaic_hwc, info

    try:
        max_radius_px = int(max(4, min(512, int(max_radius_px))))
    except Exception:
        max_radius_px = 64
    try:
        blend = float(max(0.0, min(1.0, float(blend))))
    except Exception:
        blend = 0.70

    rgb = np.asarray(mosaic_hwc, dtype=np.float32)
    out = np.array(rgb, copy=True)

    valid = np.isfinite(rgb).all(axis=-1)
    if isinstance(alpha_mask, np.ndarray):
        try:
            a = np.asarray(alpha_mask)
            if a.ndim == 3 and a.shape[-1] == 1:
                a = a[..., 0]
            elif a.ndim > 2:
                a = np.squeeze(a)
            if a.shape[:2] == valid.shape:
                valid &= (a > 0)
        except Exception:
            pass
    elif isinstance(coverage_hw, np.ndarray):
        try:
            c = np.asarray(coverage_hw, dtype=np.float32)
            if c.ndim > 2:
                c = np.squeeze(c)
            if c.shape[:2] == valid.shape:
                valid &= np.isfinite(c) & (c > 0)
        except Exception:
            pass

    hole = ~valid
    hole_px = int(np.count_nonzero(hole))
    info["hole_px"] = hole_px
    if hole_px <= 0:
        info["reason"] = "no_holes"
        return out, info

    target = hole.copy()
    dist = None
    if only_near_seams:
        try:
            import cv2  # type: ignore
            dist = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3)
            target = hole & (dist <= float(max_radius_px))
        except Exception:
            # fallback: binary dilation-limited mask via blur threshold
            soft = gaussian_blur_2d_float32(
                hole.astype(np.float32), sigma_px=max(1.0, float(max_radius_px) / 6.0)
            )
            target = hole & (soft > 0.05)

    fill_px = int(np.count_nonzero(target))
    info["filled_px"] = fill_px
    if fill_px <= 0:
        info["reason"] = "no_target_after_radius"
        return out, info

    if dist is None:
        # approximate feather where distance transform unavailable
        soft = gaussian_blur_2d_float32(
            target.astype(np.float32), sigma_px=max(1.0, float(max_radius_px) / 5.0)
        )
        feather = np.clip(soft, 0.0, 1.0) * float(blend)
    else:
        feather = np.clip((float(max_radius_px) - dist) / max(1.0, float(max_radius_px)), 0.0, 1.0) * float(blend)
    feather *= target.astype(np.float32)

    if bool(protect_stars_details) and np.any(valid):
        try:
            # ZM-ZEGRID-R24 (A5): a fully-empty RGB hole (all three channels NaN)
            # makes ``np.nanmean`` reduce an empty slice, which emits NumPy's
            # cosmetic ``RuntimeWarning: Mean of empty slice``. The per-pixel
            # luminance is still NaN for those pixels and bit-identical for
            # finite pixels; we only suppress that specific cosmetic warning so
            # the aesthetic helper stays warning-clean under ``-W error``.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                luminance = np.nanmean(out, axis=-1).astype(np.float32, copy=False)
            valid_luma = luminance[valid]
            protect_mask = np.zeros_like(target, dtype=bool)
            if valid_luma.size > 0:
                star_thr = float(np.nanpercentile(valid_luma, 99.5))
                if math.isfinite(star_thr):
                    protect_mask |= np.isfinite(luminance) & (luminance >= star_thr)

            gx, gy = np.gradient(np.where(np.isfinite(luminance), luminance, 0.0).astype(np.float32, copy=False))
            grad = np.hypot(gx, gy).astype(np.float32, copy=False)
            grad_valid = grad[valid]
            if grad_valid.size > 0:
                detail_thr = float(np.nanpercentile(grad_valid, 97.0))
                if math.isfinite(detail_thr):
                    protect_mask |= grad >= detail_thr

            if np.any(protect_mask):
                protect_soft = gaussian_blur_2d_float32(protect_mask.astype(np.float32), sigma_px=1.2)
                protect_soft = np.clip(protect_soft, 0.0, 1.0)
                feather *= (1.0 - 0.85 * protect_soft)
                info["protected_frac"] = float(np.count_nonzero(protect_mask)) / float(protect_mask.size)
        except Exception:
            pass

    for ch in range(3):
        src = out[..., ch]
        if np.any(valid):
            med = float(np.nanmedian(src[valid]))
        else:
            med = 0.0
        seeded = np.where(valid, src, med).astype(np.float32, copy=False)
        sigma_fill = max(2.0, float(max_radius_px) * 0.5)
        smooth = gaussian_blur_2d_float32(seeded, sigma_fill)
        src_nonan = np.where(np.isfinite(src), src, smooth)
        # rework-3 H2 fix: only the configured target mask is filled. The
        # non-target branch preserves the ORIGINAL pixel (NaN stays NaN), so
        # ``only_near_seams=true`` no longer silently fills all non-target holes.
        src[:] = np.where(
            target,
            src_nonan * (1.0 - feather) + smooth * feather,
            src,
        )

    info["applied"] = True
    info["reason"] = "ok"

    if logger is not None:
        logger.info(
            "[AestheticFill] enabled=%s applied=%s hole_px=%d filled_px=%d max_radius_px=%d blend=%.3f only_near_seams=%s protect_stars_details=%s protected_frac=%.5f",
            bool(enabled),
            True,
            hole_px,
            fill_px,
            int(max_radius_px),
            float(blend),
            bool(only_near_seams),
            bool(protect_stars_details),
            float(info.get("protected_frac", 0.0) or 0.0),
        )

    return out, info
