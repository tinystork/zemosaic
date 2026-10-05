"""SCI-05 Gate E4 — canonical RGB equalizer (pure, dependency-light).

Source-port of decision J's existing robust implementation
(``zemosaic_align_stack.equalize_rgb_medians_inplace`` + its
``_parse_percentile_pair``/``_parse_gain_clip_pair`` helpers), as an
**out-of-place** pure-NumPy primitive that never mutates the caller's array.

The math is reproduced **exactly** (see the reference); the only intentional
difference is that the canonical core returns a NEW float32 array instead of
writing in place (the in-place reference's ``write_failed`` edge case therefore
does not exist here).

No ``zemosaic_align_stack`` import (it pulls CuPy + matplotlib and is ~5s to
import); only ``numpy`` + stdlib.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "equalize_rgb_medians_canonical",
    "equalize_rgb_medians_copy",
]

# Stable decision strings (match the reference, minus the in-place-only
# "write_failed" which cannot occur for an out-of-place core).
_DECISION_INVALID_INPUT = "invalid_input"
_DECISION_NO_VALID = "no_valid_pixels"
_DECISION_INVALID_LUMINANCE = "invalid_luminance"
_DECISION_PERCENTILE_ERROR = "percentile_error"
_DECISION_INSUFFICIENT_SAMPLES = "insufficient_samples"
_DECISION_INSUFFICIENT_COVERAGE = "insufficient_coverage"
_DECISION_INVALID_CHANNEL_MEDIANS = "invalid_channel_medians"
_DECISION_INVALID_TARGET = "invalid_target"
_DECISION_APPLIED = "applied"


def _parse_percentile_pair(raw, default):
    """Coerce/validate a (low, high) percentile pair (donor/reference-exact)."""
    lo, hi = default
    try:
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            lo = float(raw[0])
            hi = float(raw[1])
        elif isinstance(raw, str):
            parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
            if len(parts) >= 2:
                lo = float(parts[0])
                hi = float(parts[1])
    except Exception:
        lo, hi = default
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo, hi = default
    lo = float(np.clip(lo, 0.0, 100.0))
    hi = float(np.clip(hi, 0.0, 100.0))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _parse_gain_clip_pair(raw, default):
    """Coerce/validate a (low, high) gain-clip pair (donor/reference-exact)."""
    lo, hi = _parse_percentile_pair(raw, default)
    if lo <= 0.0:
        lo = default[0]
    if hi <= 0.0:
        hi = default[1]
    if lo > hi:
        lo, hi = hi, lo
    return float(lo), float(hi)


def _neutral_info():
    """The reference's neutral info dict (fresh copy per call)."""
    return {
        "applied": False,
        "decision": _DECISION_INVALID_INPUT,
        "samples": 0,
        "mask_coverage": 0.0,
        "target_median": float("nan"),
        "raw_gains": [1.0, 1.0, 1.0],
        "clipped_gains": [1.0, 1.0, 1.0],
        "gain_r": 1.0,
        "gain_g": 1.0,
        "gain_b": 1.0,
    }


def equalize_rgb_medians_canonical(
    arr,
    *,
    gain_clip=(0.95, 1.05),
    bg_percentile=(5.0, 85.0),
    min_samples=5000,
    min_coverage=0.01,
    return_info=False,
):
    """Canonical conservative RGB equalization (out-of-place; never mutates input).

    Returns the equalized float32 array (== ``arr`` unchanged when not applied),
    or ``(equalized_array, info)`` when ``return_info=True``. The info dict is
    JSON-serializable (scalars/lists only).
    """
    if arr is None:
        result = None
        info = _neutral_info()
        return (result, info) if return_info else result

    result = np.array(arr, dtype=np.float32, copy=True)
    if result.ndim != 3 or result.shape[2] != 3:
        info = _neutral_info()  # decision already "invalid_input"
        return (result, info) if return_info else result

    finite = np.isfinite(result)
    positive = result > 0
    valid_rgb = np.all(finite & positive, axis=2)
    valid_count = int(np.count_nonzero(valid_rgb))
    if valid_count <= 0:
        info = _neutral_info()
        info["decision"] = _DECISION_NO_VALID
        return (result, info) if return_info else result

    lo_p, hi_p = _parse_percentile_pair(bg_percentile, (5.0, 85.0))
    clip_lo, clip_hi = _parse_gain_clip_pair(gain_clip, (0.95, 1.05))

    lum = np.nanmedian(result, axis=2)
    lum_vals = lum[valid_rgb & np.isfinite(lum)]
    if lum_vals.size == 0:
        info = _neutral_info()
        info["decision"] = _DECISION_INVALID_LUMINANCE
        return (result, info) if return_info else result

    try:
        p_lo, p_hi = np.nanpercentile(lum_vals, [lo_p, hi_p])
    except Exception:
        info = _neutral_info()
        info["decision"] = _DECISION_PERCENTILE_ERROR
        return (result, info) if return_info else result

    mask = valid_rgb & np.isfinite(lum) & (lum >= p_lo) & (lum <= p_hi)
    samples = int(np.count_nonzero(mask))
    coverage = float(samples / valid_count) if valid_count > 0 else 0.0

    try:
        min_samples_i = max(1, int(min_samples))
    except Exception:
        min_samples_i = 5000
    try:
        min_coverage_f = float(min_coverage)
    except Exception:
        min_coverage_f = 0.01
    if not np.isfinite(min_coverage_f):
        min_coverage_f = 0.01
    min_coverage_f = float(np.clip(min_coverage_f, 0.0, 1.0))

    if samples < min_samples_i:
        info = _neutral_info()
        info.update(samples=samples, mask_coverage=coverage, decision=_DECISION_INSUFFICIENT_SAMPLES)
        return (result, info) if return_info else result
    if coverage < min_coverage_f:
        info = _neutral_info()
        info.update(samples=samples, mask_coverage=coverage, decision=_DECISION_INSUFFICIENT_COVERAGE)
        return (result, info) if return_info else result

    med = np.array([np.nanmedian(result[..., c][mask]) for c in range(3)], dtype=np.float32)
    finite_chan = np.isfinite(med) & (med > 0)
    if not np.any(finite_chan):
        info = _neutral_info()
        info.update(samples=samples, mask_coverage=coverage, decision=_DECISION_INVALID_CHANNEL_MEDIANS)
        return (result, info) if return_info else result

    target = float(np.nanmedian(med[finite_chan]))
    if not np.isfinite(target) or target <= 0:
        info = _neutral_info()
        info.update(samples=samples, mask_coverage=coverage, decision=_DECISION_INVALID_TARGET)
        return (result, info) if return_info else result

    raw_gains = np.ones(3, dtype=np.float32)
    raw_gains[finite_chan] = target / med[finite_chan]
    clipped = np.clip(raw_gains, clip_lo, clip_hi).astype(np.float32)

    # Out-of-place apply (never mutates the caller's array).
    result = (result * clipped).astype(np.float32)

    info = {
        "applied": True,
        "decision": _DECISION_APPLIED,
        "samples": samples,
        "mask_coverage": coverage,
        "target_median": float(target),
        "raw_gains": [float(raw_gains[0]), float(raw_gains[1]), float(raw_gains[2])],
        "clipped_gains": [float(clipped[0]), float(clipped[1]), float(clipped[2])],
        "gain_r": float(clipped[0]),
        "gain_g": float(clipped[1]),
        "gain_b": float(clipped[2]),
    }
    return (result, info) if return_info else result


def equalize_rgb_medians_copy(
    arr,
    *,
    gain_clip=(0.95, 1.05),
    bg_percentile=(5.0, 85.0),
    min_samples=5000,
    min_coverage=0.01,
):
    """Out-of-place canonical RGB equalization; returns ``(new_float32_array, info)``."""
    return equalize_rgb_medians_canonical(
        arr,
        gain_clip=gain_clip,
        bg_percentile=bg_percentile,
        min_samples=min_samples,
        min_coverage=min_coverage,
        return_info=True,
    )
