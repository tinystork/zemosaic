"""ZM-ZEGRID-R18 — final-mosaic finishing (DBE + RGB equalization + uint16).

Post-assembly finishing applied to the ASSEMBLED mosaic BEFORE the outputs are
written. Pure ``numpy``/``scipy`` (no new dependency). Opt-in, fail-safe, and
fully documented. This module does **NOT** touch the per-cell / stacking
science: it only post-processes ``AssembledCanvas.science`` (H, W, 3 float32).

The three features honour the user's previously-ignored settings:

* ``final_mosaic_dbe_*``  — Dynamic Background Extraction on the assembled mosaic.
* ``grid_rgb_equalize``   — per-channel background (sky) equalization.
* ``save_final_as_uint16``— an integer (uint16) copy of the finished science.

When every setting is disabled the input array is returned **unchanged**
(identity) so the written science FITS is bit-equal to the pre-finishing path.
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Documented strength mapping
# ---------------------------------------------------------------------------
#
# ``final_mosaic_dbe_strength`` is a *named* scale factor applied to the
# subtracted background model (it does NOT change the estimation parameters).
# The product GUI exposes ``weak`` / ``normal`` / ``strong``; the lowercase
# aliases ``off`` / ``low`` / ``high`` are also accepted (documented here):
#
#   off    -> 0.0  (no subtraction)
#   weak   -> 0.5  (low;  alias)
#   low    -> 0.5
#   normal -> 1.0
#   strong -> 1.5  (high; alias)
#   high   -> 1.5
#
# Unknown values fall back to ``normal`` (1.0) and are surfaced in the info dict.
STRENGTH_FACTORS: dict[str, float] = {
    "off": 0.0,
    "low": 0.5,
    "weak": 0.5,
    "normal": 1.0,
    "high": 1.5,
    "strong": 1.5,
}
DEFAULT_STRENGTH = "normal"

# Default DBE estimation parameters (mirror the product config defaults).
DEFAULT_DBE_PARAMS: dict[str, float | int] = {
    "obj_k": 3.0,
    "obj_dilate_px": 3,
    "sample_step": 24,
    "smoothing": 0.6,
}


def resolve_strength_factor(strength: Any) -> float:
    """Map a strength name to its documented subtraction factor.

    ``None``/empty -> ``normal`` (1.0). Unknown names -> ``normal`` (1.0).
    """
    key = str(strength or "").strip().lower()
    if not key:
        key = DEFAULT_STRENGTH
    return float(STRENGTH_FACTORS.get(key, STRENGTH_FACTORS[DEFAULT_STRENGTH]))


def _safe_float(value: Any, fallback: float) -> float:
    try:
        if value is None:
            return float(fallback)
        if isinstance(value, str) and not value.strip():
            return float(fallback)
        return float(value)
    except Exception:
        return float(fallback)


def _safe_int(value: Any, fallback: int) -> int:
    try:
        if value is None:
            return int(fallback)
        if isinstance(value, str) and not value.strip():
            return int(fallback)
        return int(float(value))
    except Exception:
        return int(fallback)


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return value not in (0, 0.0)
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off", "none")
    return True


def resolve_dbe_params(zconfig: Any, strength: str | None = None) -> dict:
    """Resolve the DBE estimation parameters from the config object.

    Reads ``final_mosaic_dbe_obj_k`` / ``obj_dilate_px`` / ``sample_step`` /
    ``smoothing`` (with the documented defaults). ``strength`` is resolved to a
    documented factor separately; it never changes the estimation parameters.
    """
    def _get(key: str, fallback: Any) -> Any:
        try:
            return getattr(zconfig, key, fallback)
        except Exception:
            return fallback

    params = {
        "obj_k": max(0.0, _safe_float(_get("final_mosaic_dbe_obj_k", DEFAULT_DBE_PARAMS["obj_k"]),
                                      float(DEFAULT_DBE_PARAMS["obj_k"]))),
        "obj_dilate_px": max(0, _safe_int(_get("final_mosaic_dbe_obj_dilate_px", DEFAULT_DBE_PARAMS["obj_dilate_px"]),
                                          int(DEFAULT_DBE_PARAMS["obj_dilate_px"]))),
        "sample_step": max(1, _safe_int(_get("final_mosaic_dbe_sample_step", DEFAULT_DBE_PARAMS["sample_step"]),
                                        int(DEFAULT_DBE_PARAMS["sample_step"]))),
        "smoothing": max(0.0, _safe_float(_get("final_mosaic_dbe_smoothing", DEFAULT_DBE_PARAMS["smoothing"]),
                                          float(DEFAULT_DBE_PARAMS["smoothing"]))),
    }
    return params


def resolve_finishing_config(
    zconfig: Any,
    *,
    grid_rgb_equalize: bool | None = None,
    save_final_as_uint16: bool | None = None,
) -> dict:
    """Resolve the finishing configuration from the config object + call args.

    Returns a dict with the four booleans/parameters that drive finishing:

    * ``dbe_enabled``   — from ``final_mosaic_dbe_enabled`` (default True).
    * ``dbe_strength``  — from ``final_mosaic_dbe_strength`` (default ``normal``).
    * ``dbe_params``    — the estimation parameters (see :func:`resolve_dbe_params`).
    * ``rgb_equalize``  — from ``grid_rgb_equalize`` (default True).
    * ``save_uint16``   — from ``save_final_as_uint16`` (default False).
    """
    # DBE enable flag: default True (matches the product config default). Explicit
    # False -> DBE off (bit-equal output).
    raw = getattr(zconfig, "final_mosaic_dbe_enabled", None) if zconfig is not None else None
    dbe_enabled = True if raw is None else bool(raw)

    raw_strength = str(getattr(zconfig, "final_mosaic_dbe_strength", DEFAULT_STRENGTH) or DEFAULT_STRENGTH) \
        if zconfig is not None else DEFAULT_STRENGTH
    strength = raw_strength.strip().lower() or DEFAULT_STRENGTH
    if strength not in STRENGTH_FACTORS:
        strength = DEFAULT_STRENGTH

    dbe_params = resolve_dbe_params(zconfig, strength)

    # RGB equalization flag: default True. Honour the explicit argument first,
    # then the config key, then the default.
    if grid_rgb_equalize is not None:
        rgb_equalize = bool(grid_rgb_equalize)
    else:
        cfg_val = getattr(zconfig, "grid_rgb_equalize", None) if zconfig is not None else None
        rgb_equalize = True if cfg_val is None else bool(cfg_val)

    # uint16 flag: default False.
    if save_final_as_uint16 is not None:
        save_uint16 = bool(save_final_as_uint16)
    else:
        cfg_val = getattr(zconfig, "save_final_as_uint16", None) if zconfig is not None else None
        save_uint16 = False if cfg_val is None else bool(cfg_val)

    return {
        "dbe_enabled": bool(dbe_enabled),
        "dbe_strength": strength,
        "dbe_strength_factor": resolve_strength_factor(strength),
        "dbe_params": dbe_params,
        "rgb_equalize": bool(rgb_equalize),
        "save_uint16": bool(save_uint16),
    }


# ---------------------------------------------------------------------------
# Robust statistics + background estimation
# ---------------------------------------------------------------------------

def _robust_sky(values: np.ndarray) -> tuple[float, float]:
    """Robust (median / MAD) sky level and sigma of a 1-D float array."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = float(1.4826 * mad)
    return med, sigma


def _block_median(values: np.ndarray, valid: np.ndarray, block: int) -> np.ndarray:
    """Vectorized per-block median of ``values`` where ``valid`` is True.

    Returns a ``(n_blocks_h, n_blocks_w)`` float32 grid; blocks with no valid
    samples are NaN. ``block`` is the block side in pixels.
    """
    h, w = values.shape
    bh = (h + block - 1) // block
    bw = (w + block - 1) // block
    ph = bh * block - h
    pw = bw * block - w

    vp = np.pad(values, ((0, ph), (0, pw)), mode="constant", constant_values=np.nan)
    mp = np.pad(valid, ((0, ph), (0, pw)), mode="constant", constant_values=False)
    vp = np.asarray(vp, dtype=np.float64)
    vp = vp.reshape(bh, block, bw, block)
    mp = mp.reshape(bh, block, bw, block)
    vp = np.where(mp, vp, np.nan)
    with np.errstate(invalid="ignore"):
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            grid = np.nanmedian(vp, axis=(1, 3))
    return np.asarray(grid, dtype=np.float32)


def _fill_nan_nearest(grid: np.ndarray) -> np.ndarray:
    """Fill NaN cells of a 2-D grid with the nearest finite cell's value."""
    from scipy.ndimage import distance_transform_edt

    grid = np.asarray(grid, dtype=np.float32)
    if not np.isnan(grid).any():
        return grid
    if np.all(np.isnan(grid)):
        return np.zeros_like(grid, dtype=np.float32)
    inds = distance_transform_edt(
        np.isnan(grid), return_distances=False, return_indices=True
    )
    return np.asarray(grid[tuple(inds)], dtype=np.float32)


def estimate_background_channel(
    channel: np.ndarray,
    valid: np.ndarray,
    *,
    sample_step: int,
    obj_k: float,
    obj_dilate_px: int,
    smoothing: float,
) -> tuple[np.ndarray, dict]:
    """Estimate a smooth 2-D background model for a single channel.

    Algorithm (documented):

    1. Robust sky level/sigma (median + 1.4826*MAD) over valid pixels.
    2. Object mask = valid pixels brighter than ``sky + obj_k*sigma``, dilated by
       ``obj_dilate_px`` (scipy binary dilation, 4-connectivity).
    3. Coarse background grid: per-``sample_step``-block median of the
       object-masked channel.
    4. Fill any NaN blocks (holes / fully-object blocks) by nearest-valid.
    5. Smooth the grid with a Gaussian of ``sigma=smoothing`` (grid-pixel units).
    6. Bilinear upsample to full resolution.

    Returns ``(background, info)``.
    """
    from scipy.ndimage import binary_dilation, gaussian_filter, zoom

    info: dict = {"reason": "", "model": "block_median_gaussian"}
    ch = np.asarray(channel, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(ch)

    if not np.any(valid):
        info["reason"] = "no_valid"
        return np.zeros_like(ch), info

    sky, sigma = _robust_sky(ch[valid])
    info["sky"] = float(sky)
    info["sigma"] = float(sigma)
    thr = float(sky + float(obj_k) * sigma)
    info["obj_threshold"] = float(thr)

    obj = valid & (ch > thr)
    if obj_dilate_px > 0:
        obj = binary_dilation(obj, iterations=int(obj_dilate_px))
    bg = valid & (~obj)
    info["obj_frac"] = float(np.count_nonzero(obj) / max(1, int(np.count_nonzero(valid))))

    if not np.any(bg):
        # All valid pixels look like objects; fall back to the full valid set so
        # DBE still has a model (documented fallback, never an unhandled error).
        bg = valid
        info["obj_mask_fallback"] = True

    block = max(1, int(sample_step))
    grid = _block_median(ch, bg, block)
    info["grid_shape"] = [int(grid.shape[0]), int(grid.shape[1])]
    info["grid_nan_before_fill"] = int(np.count_nonzero(np.isnan(grid)))

    grid = _fill_nan_nearest(grid)
    info["smoothing_sigma"] = float(smoothing)
    grid_smooth = gaussian_filter(grid, sigma=float(smoothing))
    info["grid_median"] = float(np.nanmedian(grid_smooth))
    info["grid_std"] = float(np.nanstd(grid_smooth))

    # Bilinear upsample to full resolution.
    zoom_h = ch.shape[0] / float(grid_smooth.shape[0])
    zoom_w = ch.shape[1] / float(grid_smooth.shape[1])
    bg_full = zoom(grid_smooth, (zoom_h, zoom_w), order=1).astype(np.float32)
    # zoom can be off-by-one; crop/pad to the exact shape.
    if bg_full.shape != ch.shape:
        out = np.full(ch.shape, np.nan, dtype=np.float32)
        hh = min(bg_full.shape[0], ch.shape[0])
        ww = min(bg_full.shape[1], ch.shape[1])
        out[:hh, :ww] = bg_full[:hh, :ww]
        bg_full = _fill_nan_nearest(out)

    return bg_full, info


def apply_dbe(
    science: np.ndarray,
    valid: np.ndarray,
    *,
    strength: str,
    strength_factor: float,
    params: dict,
) -> tuple[np.ndarray, dict]:
    """Subtract a smooth background model from each channel.

    Returns ``(corrected, info)``. ``corrected`` is a NEW float32 array (the
    input is never mutated). Object pixels are protected by the estimation mask
    (they are never part of the background samples), so bright sources survive.
    """
    science = np.asarray(science, dtype=np.float32)
    out = science.copy()
    factor = float(strength_factor)
    info: dict = {
        "strength": strength,
        "strength_factor": factor,
        "params": dict(params),
        "channels": [],
        "applied": False,
    }

    if science.ndim != 3 or science.shape[-1] != 3:
        info["reason"] = "non_rgb"
        return out, info

    for c in range(3):
        bg, cinfo = estimate_background_channel(
            science[..., c], valid,
            sample_step=int(params["sample_step"]),
            obj_k=float(params["obj_k"]),
            obj_dilate_px=int(params["obj_dilate_px"]),
            smoothing=float(params["smoothing"]),
        )
        cinfo["channel"] = c
        use = valid & np.isfinite(science[..., c]) & np.isfinite(bg)
        if np.any(use):
            out[..., c][use] = science[..., c][use] - factor * bg[use]
        cinfo["bg_mean_abs"] = float(np.mean(np.abs(bg[use]))) if np.any(use) else 0.0
        cinfo["sky_after"] = float(np.median(out[..., c][use])) if np.any(use) else 0.0
        cinfo["applied"] = bool(np.any(use))
        info["channels"].append(cinfo)

    info["applied"] = bool(any(c.get("applied") for c in info["channels"]))
    return out, info


# ---------------------------------------------------------------------------
# RGB background equalization
# ---------------------------------------------------------------------------

def equalize_rgb(science: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Equalize per-channel background (sky) levels additively.

    Each channel's robust sky median (over the valid coverage) is shifted to the
    three-channel mean, so the channel backgrounds agree while the relative
    per-channel signal (stars) is preserved. Returns ``(equalized, info)``.
    """
    science = np.asarray(science, dtype=np.float32)
    out = science.copy()
    info: dict = {"applied": False, "skies_before": [], "skies_after": [], "target": 0.0}

    if science.ndim != 3 or science.shape[-1] != 3:
        info["reason"] = "non_rgb"
        return out, info

    skies: list[float] = []
    for c in range(3):
        ch = science[..., c]
        v = valid & np.isfinite(ch)
        skies.append(_robust_sky(ch[v])[0] if np.any(v) else 0.0)
    info["skies_before"] = [float(s) for s in skies]

    if not np.any(valid):
        info["reason"] = "no_valid"
        return out, info

    target = float(np.mean(skies))
    info["target"] = float(target)
    for c in range(3):
        offset = float(skies[c] - target)
        v = valid & np.isfinite(science[..., c])
        if np.any(v):
            out[..., c][v] = science[..., c][v] - offset
    info["skies_after"] = [
        float(_robust_sky(out[..., c][valid & np.isfinite(out[..., c])])[0])
        if np.any(valid & np.isfinite(out[..., c])) else 0.0
        for c in range(3)
    ]
    info["applied"] = True
    return out, info


# ---------------------------------------------------------------------------
# uint16 output
# ---------------------------------------------------------------------------

def to_uint16(science: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map the finished float science to a uint16 array with a documented scale.

    ``uint16 = clip(round(65535 * (v - vmin) / (vmax - vmin)), 0, 65535)`` where
    ``vmin`` / ``vmax`` are the 1st / 99.9th percentile of the valid (finite)
    pixels pooled across channels. Invalid (NaN / hole) pixels map to 0. The
    scaling constants are recorded so the mapping is auditable.
    """
    science = np.asarray(science, dtype=np.float32)
    finite = np.isfinite(science)
    info: dict = {"applied": False, "vmin": 0.0, "vmax": 65535.0,
                  "lo_pct": 1.0, "hi_pct": 99.9, "formula": "clip(round(65535*(v-vmin)/(vmax-vmin)),0,65535)"}

    vals = science[finite]
    if vals.size == 0:
        return np.zeros(science.shape[:2] + (3,), dtype=np.uint16), info
    vmin = float(np.percentile(vals, 1.0))
    vmax = float(np.percentile(vals, 99.9))
    if vmax <= vmin:
        vmax = vmin + 1.0
    info["vmin"] = float(vmin)
    info["vmax"] = float(vmax)

    scaled = (science.astype(np.float64) - vmin) / (vmax - vmin) * 65535.0
    scaled = np.clip(np.round(scaled), 0, 65535)
    out = np.zeros(science.shape[:2] + (3,), dtype=np.uint16)
    out[finite] = scaled[finite].astype(np.uint16)
    info["applied"] = True
    return out, info


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class FinishingResult:
    science: np.ndarray            # (H, W, 3) float32 (finished, or the input itself when off)
    uint16: np.ndarray | None      # (H, W, 3) uint16, or None
    info: dict                     # full finishing record for the manifest/log


def apply_final_mosaic_finishing(
    science: np.ndarray,
    coverage: np.ndarray,
    *,
    config: dict | None = None,
) -> FinishingResult:
    """Apply the final-mosaic finishing (DBE -> RGB equalize -> uint16).

    ``config`` is the dict from :func:`resolve_finishing_config`. When ``config``
    is None (or every feature is disabled) the input array is returned by
    identity and no work is done (bit-equal output). The order is documented:
    DBE first, then RGB equalization, then the uint16 render of the result.
    """
    cfg = dict(config or {})
    science = np.asarray(science, dtype=np.float32)
    coverage = np.asarray(coverage)

    info: dict = {
        "enabled": False,
        "failed": False,
        "failure_reason": "",
        "dbe": {"enabled": False, "applied": False},
        "rgb_equalize": {"enabled": False, "applied": False},
        "uint16": {"enabled": False, "written": False},
    }

    # Valid coverage mask (finite + covered). Per-channel finiteness is handled
    # inside each feature; here we build the shared coverage plane.
    valid = (coverage > 0) & np.any(np.isfinite(science), axis=-1)

    dbe_enabled = bool(cfg.get("dbe_enabled", False))
    rgb_equalize = bool(cfg.get("rgb_equalize", False))
    save_uint16 = bool(cfg.get("save_uint16", False))

    info["dbe"]["enabled"] = dbe_enabled
    info["rgb_equalize"]["enabled"] = rgb_equalize
    info["uint16"]["enabled"] = save_uint16
    info["enabled"] = bool(dbe_enabled or rgb_equalize or save_uint16)

    # When nothing is enabled, return the input by identity (bit-equal path).
    if not info["enabled"]:
        return FinishingResult(science=science, uint16=None, info=info)

    out = science

    if dbe_enabled:
        strength = str(cfg.get("dbe_strength", DEFAULT_STRENGTH))
        factor = float(cfg.get("dbe_strength_factor", resolve_strength_factor(strength)))
        params = dict(cfg.get("dbe_params", DEFAULT_DBE_PARAMS))
        out, dbe_info = apply_dbe(
            out, valid, strength=strength, strength_factor=factor, params=params
        )
        info["dbe"].update(dbe_info)
        info["dbe"]["enabled"] = True

    if rgb_equalize:
        out, rgb_info = equalize_rgb(out, valid)
        info["rgb_equalize"].update(rgb_info)
        info["rgb_equalize"]["enabled"] = True

    uint16_data: np.ndarray | None = None
    if save_uint16:
        uint16_data, u16_info = to_uint16(out, valid)
        info["uint16"].update(u16_info)
        info["uint16"]["enabled"] = True
        info["uint16"]["written"] = bool(u16_info.get("applied", False))

    return FinishingResult(science=out, uint16=uint16_data, info=info)
