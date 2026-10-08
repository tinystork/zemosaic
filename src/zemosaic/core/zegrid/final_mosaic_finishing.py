"""ZM-ZEGRID-R23 — final-mosaic finishing (scientific DBE + RGB equalize + uint16).

Post-assembly finishing applied to the ASSEMBLED mosaic BEFORE the outputs are
written. Pure ``numpy``/``scipy`` (no new dependency). Opt-in, fail-safe, and
fully documented. This module does **NOT** touch the per-cell / stacking
science: it only post-processes ``AssembledCanvas.science`` (H, W, 3 float32).

R23 rework (contract ZM-ZEGRID-R23, from the R18 baseline):

* **Scientific honesty.** DBE now corrects only the estimated BACKGROUND
  VARIATION around a robust per-channel reference baseline::

      corrected = ch - (bg - bg_reference)

  (legacy "light DBE" semantics), instead of R18's destructive full-background
  subtraction ``ch - factor*bg`` which drove ~43% negatives on real M106 data.
  Global sky/DC is preserved; negatives are legitimate science and are never
  clamped, inverted, or absoluted.

* **Strength/custom semantics restored.** ``weak`` / ``normal`` / ``strong``
  choose *estimator parameter presets* (object-mask k, dilation, sample step,
  smoothing) — exactly the legacy product presets. ``custom`` reads the explicit
  numeric config fields. There is NO hidden 0.5 / 1.5 model multiplier: the
  subtraction factor is always 1.0. Invalid strength -> documented ``normal``.

* **Diffuse protection.** A multiscale coarse-residual mask excludes broad
  low-contrast structures (galaxies, nebulosity) from the background fit, so the
  smooth block-median model cannot absorb them; compact bright sources are masked
  (and dilated) so they do not imprint dark halos. The correction field is a
  smooth variation surface applied uniformly — no per-pixel masked/uncorrected
  seams.

* **Honest fallback.** When too few background samples remain, no correction is
  applied (the channel is returned unchanged) and the reason is recorded — never
  a silent revert to the destructive full-background subtraction.

The three honoured user settings:

* ``final_mosaic_dbe_*``  — Dynamic Background Extraction on the assembled mosaic.
* ``grid_rgb_equalize``   — per-channel background (sky) equalization.
* ``save_final_as_uint16``— an integer (uint16) render of the finished science.

When every setting is disabled the input array is returned **unchanged**
(identity) so the written science FITS is bit-equal to the pre-finishing path.
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Legacy strength presets (parameter presets, NOT subtraction multipliers)
# ---------------------------------------------------------------------------
#
# ``final_mosaic_dbe_strength`` selects ESTIMATOR PARAMETERS. The product's
# documented legacy presets are reproduced exactly:
#
#   weak   -> {obj_k: 4.0, obj_dilate_px: 2, sample_step: 32, smoothing: 1.0}
#   normal -> {obj_k: 3.0, obj_dilate_px: 3, sample_step: 24, smoothing: 0.6}
#   strong -> {obj_k: 2.2, obj_dilate_px: 4, sample_step: 16, smoothing: 0.25}
#   custom -> the explicit ``final_mosaic_dbe_obj_k`` / ``obj_dilate_px`` /
#             ``sample_step`` / ``smoothing`` config values.
#
# Any other value (including the legacy aliases ``off``/``low``/``high`` and
# empty/None) is treated as the documented ``normal`` preset.
DBE_STRENGTH_PRESETS: dict[str, dict[str, float | int]] = {
    "weak": {"obj_k": 4.0, "obj_dilate_px": 2, "sample_step": 32, "smoothing": 1.0},
    "normal": {"obj_k": 3.0, "obj_dilate_px": 3, "sample_step": 24, "smoothing": 0.6},
    "strong": {"obj_k": 2.2, "obj_dilate_px": 4, "sample_step": 16, "smoothing": 0.25},
}
DEFAULT_STRENGTH = "normal"

# Subtraction factor is always 1.0 (variation-only correction). Kept explicit so
# the contract is auditable and no hidden scalar can creep in.
DBE_SUBTRACTION_FACTOR = 1.0


# Diffuse-protection tuning (internal, documented; not user-facing knobs).
DIFFUSE_COARSE_FACTOR = 8.0   # coarse bg sigma = sample_step * this, for residual
DIFFUSE_DETECT_SIGMA = 24.0   # residual smoothing scale (px) for diffuse detection
DIFFUSE_SIGNIFICANCE = 3.0    # diffuse residual significance (units of pixel noise)
MIN_BG_SAMPLES = 9            # minimum background samples before honest fallback


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


def _config_get(zconfig: Any, key: str, fallback: Any) -> Any:
    if zconfig is None:
        return fallback
    try:
        return getattr(zconfig, key, fallback)
    except Exception:
        return fallback


def resolve_dbe_strength(zconfig: Any) -> dict:
    """Resolve DBE strength into effective estimator params + provenance.

    Returns ``{strength, params_source, params}``.

    * ``strength`` — the effective strength name (``weak``/``normal``/``strong``/
      ``custom``); invalid/unknown -> ``normal``.
    * ``params_source`` — ``preset:<name>`` or ``custom_cfg``.
    * ``params`` — the effective ``{obj_k, obj_dilate_px, sample_step, smoothing}``.
    """
    raw = str(_config_get(zconfig, "final_mosaic_dbe_strength", DEFAULT_STRENGTH)
              or DEFAULT_STRENGTH).strip().lower()
    if raw in DBE_STRENGTH_PRESETS:
        params = dict(DBE_STRENGTH_PRESETS[raw])
        return {"strength": raw, "params_source": f"preset:{raw}", "params": params}

    if raw == "custom":
        normal = DBE_STRENGTH_PRESETS[DEFAULT_STRENGTH]
        params = {
            "obj_k": max(0.0, _safe_float(
                _config_get(zconfig, "final_mosaic_dbe_obj_k", normal["obj_k"]),
                float(normal["obj_k"]))),
            "obj_dilate_px": max(0, _safe_int(
                _config_get(zconfig, "final_mosaic_dbe_obj_dilate_px", normal["obj_dilate_px"]),
                int(normal["obj_dilate_px"]))),
            "sample_step": max(1, _safe_int(
                _config_get(zconfig, "final_mosaic_dbe_sample_step", normal["sample_step"]),
                int(normal["sample_step"]))),
            "smoothing": max(0.0, _safe_float(
                _config_get(zconfig, "final_mosaic_dbe_smoothing", normal["smoothing"]),
                float(normal["smoothing"]))),
        }
        return {"strength": "custom", "params_source": "custom_cfg", "params": params}

    # Invalid / legacy alias -> documented normal.
    params = dict(DBE_STRENGTH_PRESETS[DEFAULT_STRENGTH])
    return {"strength": DEFAULT_STRENGTH, "params_source": "preset:normal", "params": params}


def resolve_dbe_params(zconfig: Any, strength: str | None = None) -> dict:
    """Resolve effective DBE estimation params (back-compat name).

    ``strength`` is accepted for signature compatibility but is ignored: the
    strength always comes from the config (see :func:`resolve_dbe_strength`).
    """
    return resolve_dbe_strength(zconfig)["params"]


def resolve_finishing_config(
    zconfig: Any,
    *,
    grid_rgb_equalize: bool | None = None,
    save_final_as_uint16: bool | None = None,
) -> dict:
    """Resolve the finishing configuration from the config object + call args.

    Returns a dict:

    * ``dbe_enabled``            — from ``final_mosaic_dbe_enabled`` (default True).
    * ``dbe_strength``           — effective strength (weak/normal/strong/custom).
    * ``dbe_params_source``      — ``preset:*`` or ``custom_cfg``.
    * ``dbe_params``             — effective estimator params.
    * ``dbe_subtraction_factor`` — always 1.0 (variation-only correction).
    * ``rgb_equalize``           — from ``grid_rgb_equalize`` (default True).
    * ``save_uint16``            — from ``save_final_as_uint16`` (default False).
    """
    raw = _config_get(zconfig, "final_mosaic_dbe_enabled", None)
    dbe_enabled = True if raw is None else bool(raw)

    strength_info = resolve_dbe_strength(zconfig)

    if grid_rgb_equalize is not None:
        rgb_equalize = bool(grid_rgb_equalize)
    else:
        cfg_val = _config_get(zconfig, "grid_rgb_equalize", None)
        rgb_equalize = True if cfg_val is None else bool(cfg_val)

    if save_final_as_uint16 is not None:
        save_uint16 = bool(save_final_as_uint16)
    else:
        cfg_val = _config_get(zconfig, "save_final_as_uint16", None)
        save_uint16 = False if cfg_val is None else bool(cfg_val)

    return {
        "dbe_enabled": bool(dbe_enabled),
        "dbe_strength": strength_info["strength"],
        "dbe_params_source": strength_info["params_source"],
        "dbe_params": strength_info["params"],
        "dbe_subtraction_factor": 1.0,
        "rgb_equalize": bool(rgb_equalize),
        "save_uint16": bool(save_uint16),
    }


# ---------------------------------------------------------------------------
# Robust statistics
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


def _upsample_bg(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear upsample a low-res grid to ``shape`` (crop/pad exact)."""
    from scipy.ndimage import zoom

    grid = np.asarray(grid, dtype=np.float32)
    zh = shape[0] / float(grid.shape[0])
    zw = shape[1] / float(grid.shape[1])
    bg = zoom(grid, (zh, zw), order=1).astype(np.float32)
    if bg.shape != shape:
        out = np.full(shape, np.nan, dtype=np.float32)
        hh = min(bg.shape[0], shape[0])
        ww = min(bg.shape[1], shape[1])
        out[:hh, :ww] = bg[:hh, :ww]
        bg = _fill_nan_nearest(out)
    return bg


# ---------------------------------------------------------------------------
# Background estimation (scientific DBE, diffuse-safe)
# ---------------------------------------------------------------------------

def estimate_background_channel(
    channel: np.ndarray,
    valid: np.ndarray,
    *,
    sample_step: int,
    obj_k: float,
    obj_dilate_px: int,
    smoothing: float,
) -> tuple[np.ndarray, dict]:
    """Estimate a smooth 2-D background VARIATION model for a single channel.

    Algorithm (documented, deterministic, bounded-RAM):

    1. Robust sky level/sigma (median + 1.4826*MAD) over valid pixels, plus a
       high-pass pixel-noise estimate (robust to gradient/diffuse structure).
    2. Compact-object mask: valid pixels brighter than ``sky + obj_k*sigma``,
       dilated by ``obj_dilate_px`` (protects bright sources -> no dark halos).
    3. Diffuse-object mask (multiscale): a coarse heavily-smoothed background
       captures only the large-scale gradient/vignette; the smoothed residual
       ``channel - coarse_bg`` reveals broad low-contrast structures; pixels
       whose residual exceeds ``DIFFUSE_SIGNIFICANCE * noise`` (then morphologically
       opened to require spatial extension) are excluded from the background fit.
    4. Background grid: per-``sample_step``-block median of the background-only
       samples, NaN-filled, Gaussian-smoothed (``smoothing``), bilinearly upsampled.
    5. Robust reference baseline = median of the background model over the
       background-only samples (preserves global sky/DC).

    Returns ``(bg, info)``. On insufficient background samples the returned
    ``bg`` is a flat field at the sky level (so ``ch - (bg - bg_ref)`` leaves the
    channel unchanged) and ``info`` records the honest fallback.
    """
    from scipy.ndimage import (
        binary_dilation,
        binary_opening,
        gaussian_filter,
        generate_binary_structure,
    )

    info: dict = {
        "reason": "",
        "model": "block_median_variation_diffuse_safe",
        "fallback": "none",
    }
    ch = np.asarray(channel, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(ch)

    if not np.any(valid):
        info["reason"] = "no_valid"
        return np.zeros_like(ch), info

    sky, sigma = _robust_sky(ch[valid])
    info["sky"] = float(sky)
    info["sigma"] = float(sigma)

    # high-pass pixel noise (robust to gradient + diffuse structure)
    sm2 = gaussian_filter(np.nan_to_num(ch, nan=sky), sigma=2.0)
    noise = _robust_sky((ch - sm2)[valid])[1]
    info["noise"] = float(noise)

    # 1. compact-object (star) mask
    star = valid & (ch > sky + float(obj_k) * sigma)
    if obj_dilate_px > 0:
        star = binary_dilation(star, iterations=int(obj_dilate_px))
    info["star_frac"] = float(np.count_nonzero(star) / max(1, int(np.count_nonzero(valid))))

    # 2. diffuse-object mask (multiscale coarse residual)
    filled = np.where(valid & ~star, ch, sky).astype(np.float32)
    coarse_sigma = max(float(sample_step) * DIFFUSE_COARSE_FACTOR, DIFFUSE_DETECT_SIGMA)
    bg_coarse = gaussian_filter(filled, sigma=coarse_sigma)
    resid = ch - bg_coarse
    resid_sm = gaussian_filter(np.nan_to_num(resid, nan=0.0), sigma=DIFFUSE_DETECT_SIGMA)
    thr = float(DIFFUSE_SIGNIFICANCE) * float(noise) if noise > 0 else 0.0
    diffuse = valid & (~star) & (resid_sm > thr)
    if np.any(diffuse) and DIFFUSE_DETECT_SIGMA >= 3:
        diffuse = binary_opening(
            diffuse,
            structure=generate_binary_structure(2, 1),
            iterations=max(1, int(round(DIFFUSE_DETECT_SIGMA))),
        )
    info["diffuse_frac"] = float(np.count_nonzero(diffuse) / max(1, int(np.count_nonzero(valid))))

    # 3. combined object mask, dilated to protect halo shoulders
    obj = star | diffuse
    if obj_dilate_px > 0:
        obj = binary_dilation(obj, iterations=int(obj_dilate_px))
    bg_pixels = valid & (~obj)
    info["obj_frac"] = float(np.count_nonzero(obj) / max(1, int(np.count_nonzero(valid))))

    if int(np.count_nonzero(bg_pixels)) < MIN_BG_SAMPLES:
        info["reason"] = "insufficient_bg"
        info["fallback"] = "no_subtraction"
        return np.full_like(ch, sky), info

    # 4. background model from background-only samples
    block = max(1, int(sample_step))
    grid = _block_median(ch, bg_pixels, block)
    info["grid_shape"] = [int(grid.shape[0]), int(grid.shape[1])]
    info["grid_nan_before_fill"] = int(np.count_nonzero(np.isnan(grid)))

    grid = _fill_nan_nearest(grid)
    info["smoothing_sigma"] = float(smoothing)
    grid = gaussian_filter(grid, sigma=float(smoothing))
    bg = _upsample_bg(grid, ch.shape)

    # 5. robust reference baseline (preserve global sky/DC)
    bg_ref = float(np.median(bg[bg_pixels]))
    info["bg_ref"] = float(bg_ref)
    info["grid_median"] = float(np.nanmedian(grid))
    info["grid_std"] = float(np.nanstd(grid))
    return bg, info


def _channel_stats(ch: np.ndarray, valid: np.ndarray) -> dict:
    vals = ch[valid]
    return {
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
        "median": float(np.nanmedian(vals)),
        "neg_frac": float(np.mean(vals < 0)),
    }


def apply_dbe(
    science: np.ndarray,
    valid: np.ndarray,
    *,
    strength: str,
    params: dict,
) -> tuple[np.ndarray, dict]:
    """Correct per-channel BACKGROUND VARIATION (scientific DBE).

    ``corrected = ch - (bg - bg_ref)`` over valid pixels, preserving global
    sky/DC. Returns ``(corrected, info)``. The input is never mutated. Negatives
    are never clamped / absoluted / inverted.

    ``strength`` / ``params`` / ``params_source`` are recorded (see
    :func:`apply_final_mosaic_finishing` for how the top-level info is assembled).
    """
    science = np.asarray(science, dtype=np.float32)
    out = science.copy()
    info: dict = {
        "strength": strength,
        "params": dict(params),
        "channels": [],
        "applied": False,
        "model": "block_median_variation_diffuse_safe",
    }

    if science.ndim != 3 or science.shape[-1] != 3:
        info["reason"] = "non_rgb"
        return out, info

    for c in range(3):
        ch = science[..., c]
        ch_valid = valid & np.isfinite(ch)
        cinfo = {"channel": int(c), "applied": False}
        cinfo["before"] = _channel_stats(ch, ch_valid) if np.any(ch_valid) else None

        bg, binfo = estimate_background_channel(
            ch, valid,
            sample_step=int(params["sample_step"]),
            obj_k=float(params["obj_k"]),
            obj_dilate_px=int(params["obj_dilate_px"]),
            smoothing=float(params["smoothing"]),
        )
        cinfo.update({k: binfo[k] for k in binfo if k not in ("reason", "fallback")})
        cinfo["fallback"] = binfo.get("fallback", "none")

        use = valid & np.isfinite(ch) & np.isfinite(bg)
        if binfo.get("fallback") == "no_subtraction":
            # Honest fallback: leave the channel unchanged (already == ch - (flat - flat)).
            cinfo["reason"] = "no_subtraction"
            cinfo["applied"] = False
        elif np.any(use):
            bg_ref = float(binfo["bg_ref"])
            out[..., c][use] = ch[use] - (bg[use] - bg_ref)
            cinfo["applied"] = True
            cinfo["bg_mean_abs"] = float(np.mean(np.abs(bg[use] - bg_ref)))

        cinfo["after"] = _channel_stats(out[..., c], ch_valid) if np.any(ch_valid) else None
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
    scaling constants are recorded so the mapping is auditable. This render is
    derived from the finished float science and is never a scientific reference.
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
        params = dict(cfg.get("dbe_params", DBE_STRENGTH_PRESETS[DEFAULT_STRENGTH]))
        out, dbe_info = apply_dbe(out, valid, strength=strength, params=params)
        info["dbe"].update(dbe_info)
        info["dbe"]["enabled"] = True
        info["dbe"]["params_source"] = str(cfg.get("dbe_params_source", "preset:normal"))
        info["dbe"]["subtraction_factor"] = float(cfg.get("dbe_subtraction_factor", 1.0))

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
