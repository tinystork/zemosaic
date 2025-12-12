"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                               ║
║                                                                                   ║
║ Auteur  : Tinystork, seigneur des couteaux à beurre (aka Tristan Nauleau)         ║
║ Partenaire : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System   ║
║              (aka ChatGPT, Grand Master of Code Chiseling)                        ║
║                                                                                   ║
║ Licence : GNU General Public License v3.0 (GPL-3.0)                               ║
║                                                                                   ║
║ Description :                                                                     ║
║   Ce module fournit un cœur de stacking réutilisable pour CPU et GPU,             ║
║   factorisant la logique commune entre Grid mode et pipeline classique.           ║
║   Utilise duck typing pour abstraction CPU/GPU (numpy/cupy).                      ║
║                                                                                   ║
║ Avertissement :                                                                   ║
║   Aucune IA ni aucun couteau à beurre n’a été blessé durant le                    ║
║   développement de ce code.                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import importlib.util
import math

import numpy as np

GPU_AVAILABLE = importlib.util.find_spec("cupy") is not None
if GPU_AVAILABLE:
    import cupy as cp
else:
    cp = None

try:
    from zemosaic_align_stack import _reject_outliers_kappa_sigma
except Exception:
    _reject_outliers_kappa_sigma = None


logger = logging.getLogger("ZeMosaicWorker").getChild("stack_core")

def get_backend_module(backend: str):
    """Return the array module for the given backend ('cpu' or 'gpu')."""
    if backend == 'gpu':
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU backend requested but CuPy not available")
        return cp
    elif backend == 'cpu':
        return np
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _ensure_hwc_tile(tile: np.ndarray) -> np.ndarray:
    """Return ``tile`` as float32 ``H x W x C``.

    Accepts ``H x W`` or ``H x W x C`` (channels last) and best-effort converts
    channels-first inputs (``C x H x W``) when ``C`` is small compared to the
    spatial dimensions. Extra singleton dimensions are squeezed so downstream
    code can uniformly operate on ``HWC`` arrays.
    """

    arr = np.asarray(tile, dtype=np.float32)
    if arr.ndim == 2:
        return arr[..., np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D tile, got shape={arr.shape}")
    if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.moveaxis(arr, 0, -1)
    return np.squeeze(arr, axis=0) if arr.shape[0] == 1 else arr


def _channel_mask(mask: np.ndarray | None, shape_hw: tuple[int, int], channel: int, channels: int) -> np.ndarray | None:
    if mask is None:
        return None
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape[:2] != shape_hw:
        return None
    if mask_arr.ndim == 2:
        return mask_arr
    if mask_arr.ndim == 3:
        if mask_arr.shape[-1] == channels:
            return mask_arr[..., channel]
        if mask_arr.shape[-1] == 1:
            return mask_arr[..., 0]
    return None


def _log_tile_channel_stats(label: str, tile: np.ndarray, *, mask: np.ndarray | None = None) -> None:
    try:
        arr = _ensure_hwc_tile(tile)
    except Exception:
        return
    channels = arr.shape[-1]
    for c in range(channels):
        mask_c = _channel_mask(mask, arr.shape[:2], c, channels)
        valid = np.isfinite(arr[..., c]) if mask_c is None else (np.isfinite(arr[..., c]) & mask_c)
        if not np.any(valid):
            logger.debug("%s channel %s stats: no valid pixels", label, c)
            continue
        vals = arr[..., c][valid]
        mn = float(np.nanmin(vals)) if vals.size else float("nan")
        med = float(np.nanmedian(vals)) if vals.size else float("nan")
        mx = float(np.nanmax(vals)) if vals.size else float("nan")
        logger.debug("%s channel %s stats: min=%.6f median=%.6f max=%.6f", label, c, mn, med, mx)


def compute_tile_photometric_scaling(
    reference_tile: np.ndarray,
    target_tile: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel gain/offset to align ``target_tile`` to ``reference_tile``.

    Both tiles are interpreted as ``H x W x C`` and converted to ``float32``.
    If no valid pixels are found for a channel, ``gain=1`` and ``offset=0`` are
    returned for that channel.
    """

    ref = _ensure_hwc_tile(reference_tile)
    tgt = _ensure_hwc_tile(target_tile)
    if ref.shape != tgt.shape:
        raise ValueError(f"Reference and target tiles must share the same shape; got {ref.shape} vs {tgt.shape}")

    channels = ref.shape[-1]
    gains = np.ones((channels,), dtype=np.float32)
    offsets = np.zeros((channels,), dtype=np.float32)

    _log_tile_channel_stats("[GRIDPHO] reference (pre)", ref, mask=mask)
    _log_tile_channel_stats("[GRIDPHO] target (pre)", tgt, mask=mask)

    for c in range(channels):
        mask_c = _channel_mask(mask, ref.shape[:2], c, channels)
        valid = np.isfinite(ref[..., c]) & np.isfinite(tgt[..., c])
        if mask_c is not None:
            valid &= mask_c
        if not np.any(valid):
            logger.debug("[GRIDPHO] channel %s: no valid pixels; scaling disabled (gain=1 offset=0)", c)
            continue

        ref_vals = ref[..., c][valid]
        tgt_vals = tgt[..., c][valid]
        ref_median = float(np.nanmedian(ref_vals)) if ref_vals.size else float("nan")
        tgt_median = float(np.nanmedian(tgt_vals)) if tgt_vals.size else float("nan")

        if not math.isfinite(ref_median) or not math.isfinite(tgt_median) or tgt_median == 0:
            logger.debug(
                "[GRIDPHO] channel %s: invalid medians (ref=%s target=%s); neutral scaling applied",
                c,
                ref_median,
                tgt_median,
            )
            continue

        gain = ref_median / tgt_median
        offset = ref_median - (gain * tgt_median)

        gain = 1.0 if not math.isfinite(gain) else float(gain)
        offset = 0.0 if not math.isfinite(offset) else float(offset)
        gains[c] = np.float32(gain)
        offsets[c] = np.float32(offset)
        logger.debug(
            "[GRIDPHO] channel %s scaling computed: gain=%.6f offset=%.6f (ref_med=%.6f tgt_med=%.6f)",
            c,
            gain,
            offset,
            ref_median,
            tgt_median,
        )

    gains = np.nan_to_num(gains, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32, copy=False)
    offsets = np.nan_to_num(offsets, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return gains, offsets


def apply_tile_photometric_scaling(
    tile: np.ndarray, gains: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Apply per-channel gain/offset to ``tile`` and return a new ``float32`` array."""

    arr = _ensure_hwc_tile(tile)
    channels = arr.shape[-1]
    gains = np.asarray(gains, dtype=np.float32).reshape((channels,))
    offsets = np.asarray(offsets, dtype=np.float32).reshape((channels,))

    _log_tile_channel_stats("[GRIDPHO] target (pre-apply)", arr)

    scaled = np.array(arr, dtype=np.float32, copy=True)
    valid = np.isfinite(arr)
    if scaled.ndim == 2:
        scaled = scaled[..., np.newaxis]
        valid = valid[..., np.newaxis]
    for c in range(channels):
        channel_mask = valid[..., c]
        if not np.any(channel_mask):
            logger.debug("[GRIDPHO] channel %s: no valid pixels to scale; left untouched", c)
            continue
        scaled[..., c][channel_mask] = (scaled[..., c][channel_mask] * gains[c]) + offsets[c]

    _log_tile_channel_stats("[GRIDPHO] target (post-apply)", scaled)
    return scaled.astype(np.float32, copy=False)

def stack_core(
    images,
    weights=None,
    stack_config=None,
    backend='cpu',
    progress_callback=None
):
    """
    Core stacking function reusable for CPU/GPU, Grid mode / classic pipeline.

    Parameters
    ----------
    images : list of np.ndarray or cp.ndarray
        List of aligned images to stack. Shapes (H,W) or (H,W,C).
    weights : np.ndarray or cp.ndarray or None
        Per-frame weights. Shape (N,) for scalar per frame, or broadcastable to (N,H,W[,C]).
    stack_config : dict or object
        Configuration dict/object with keys like:
        - 'normalize_method': 'none', 'median', 'linear_fit'
        - 'rejection_algorithm': 'none', 'kappa_sigma', 'winsorized_sigma_clip'
        - 'final_combine_method': 'mean', 'median'
        - Other params like sigma_clip_low, sigma_clip_high, etc.
    backend : str
        'cpu' or 'gpu'
    progress_callback : callable or None
        Optional callback for progress/logging.

    Returns
    -------
    stacked : np.ndarray
        Stacked image (H,W) or (H,W,C), dtype float32.
    rejected_pct : float
        Percentage of pixels rejected (0.0 if no rejection).
    """
    xp = get_backend_module(backend)

    # Validate inputs
    if not images:
        raise ValueError("No images provided")
    sample = images[0]
    if sample.ndim not in (2, 3):
        raise ValueError(f"Images must be 2D or 3D; got {sample.ndim}D")

    # Convert images to backend arrays
    images_backend = [xp.asarray(img, dtype=xp.float32) for img in images]
    stacked = xp.stack(images_backend, axis=0)  # (N, H, W[, C])

    # Prepare weights
    weights_backend = None
    if weights is not None:
        weights_backend = xp.asarray(weights, dtype=xp.float32)
        if weights_backend.ndim == 1 and weights_backend.shape[0] == len(images):
            # Broadcast scalar per-frame weights
            extra_dims = (1,) * (stacked.ndim - 1)
            weights_backend = weights_backend.reshape((len(images),) + extra_dims)
        # Ensure broadcastable
        try:
            xp.broadcast_shapes(weights_backend.shape, stacked.shape)
        except ValueError:
            raise ValueError(f"Weights shape {weights_backend.shape} not broadcastable to images shape {stacked.shape}")

    # Get config values
    config = stack_config or {}
    normalize_method = getattr(config, 'normalize_method', config.get('normalize_method', 'none'))
    rejection_algorithm = getattr(config, 'rejection_algorithm', config.get('rejection_algorithm', 'none'))
    final_combine_method = getattr(config, 'final_combine_method', config.get('final_combine_method', 'mean'))
    sigma_clip_low = float(getattr(config, 'sigma_clip_low', config.get('sigma_clip_low', 3.0)))
    sigma_clip_high = float(getattr(config, 'sigma_clip_high', config.get('sigma_clip_high', 3.0)))

    # Normalization
    if normalize_method == 'median':
        median = xp.nanmedian(stacked, axis=0)
        stacked = stacked - median
    elif normalize_method == 'linear_fit':
        # Placeholder: for now, use median. Full linear fit would require per-pixel regression.
        median = xp.nanmedian(stacked, axis=0)
        stacked = stacked - median
    # 'none' : no change

    # Outlier rejection
    rejected_pct = 0.0
    if rejection_algorithm == 'kappa_sigma':
        if _reject_outliers_kappa_sigma is not None:
            if backend == 'gpu':
                data_np = cp.asnumpy(stacked)
                data_rejected, rejection_mask = _reject_outliers_kappa_sigma(data_np, sigma_clip_low, sigma_clip_high, progress_callback=progress_callback)
                stacked = cp.asarray(data_rejected, dtype=cp.float32)
            else:
                data_rejected, rejection_mask = _reject_outliers_kappa_sigma(stacked, sigma_clip_low, sigma_clip_high, progress_callback=progress_callback)
                stacked = data_rejected
            rejected_count = np.sum(~rejection_mask)
            total_count = stacked.size
            rejected_pct = 100.0 * rejected_count / total_count if total_count > 0 else 0.0
        else:
            # Fallback to simple rejection
            med = xp.nanmedian(stacked, axis=0)
            std = xp.nanstd(stacked, axis=0)
            low = med - sigma_clip_low * std
            high = med + sigma_clip_high * std
            mask = (stacked >= low) & (stacked <= high)
            rejected_count = float(xp.sum(~mask))
            total_count = float(xp.prod(xp.array(mask.shape)))
            rejected_pct = 100.0 * rejected_count / total_count if total_count > 0 else 0.0
            stacked = xp.where(mask, stacked, xp.nan)
    elif rejection_algorithm == 'winsorized_sigma_clip':
        # Placeholder: simplified winsorized
        # Full implementation would winsorize first, then clip.
        med = xp.nanmedian(stacked, axis=0)
        std = xp.nanstd(stacked, axis=0)
        low = med - sigma_clip_low * std
        high = med + sigma_clip_high * std
        mask = (stacked >= low) & (stacked <= high)
        rejected_count = float(xp.sum(~mask))
        total_count = float(xp.prod(xp.array(mask.shape)))
        rejected_pct = 100.0 * rejected_count / total_count if total_count > 0 else 0.0
        stacked = xp.where(mask, stacked, xp.nan)
    # 'none' or others: no rejection

    # Final combine
    finite_mask = xp.isfinite(stacked)
    if weights_backend is None:
        weights_backend = xp.ones_like(stacked)
    weight_sum = xp.sum(xp.where(finite_mask, weights_backend, 0.0), axis=0)

    if final_combine_method == 'mean':
        weighted_data = xp.where(finite_mask, stacked * weights_backend, 0.0)
        weighted_sum = xp.sum(weighted_data, axis=0)
        with xp.errstate(divide='ignore', invalid='ignore'):
            result = xp.where(weight_sum > 0, weighted_sum / weight_sum, xp.nan)
    elif final_combine_method == 'median':
        # Median ignores weights
        result = xp.nanmedian(stacked, axis=0)
    else:
        raise ValueError(f"Unsupported combine method: {final_combine_method}")

    # Convert back to numpy
    if backend == 'gpu':
        result = cp.asnumpy(result)
        weight_sum = cp.asnumpy(weight_sum)
    result = np.asarray(result, dtype=np.float32)
    weight_sum = np.asarray(weight_sum, dtype=np.float32)

    return result, rejected_pct, weight_sum
