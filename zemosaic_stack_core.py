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
║   Ce module fournit un cœur de stacking réutilisable pour CPU et GPU,            ║
║   factorisant la logique commune entre Grid mode et pipeline classique.          ║
║   Utilise duck typing pour abstraction CPU/GPU (numpy/cupy).                     ║
║                                                                                   ║
║ Avertissement :                                                                   ║
║   Aucune IA ni aucun couteau à beurre n’a été blessé durant le                    ║
║   développement de ce code.                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import importlib.util

GPU_AVAILABLE = importlib.util.find_spec("cupy") is not None
if GPU_AVAILABLE:
    import cupy as cp
else:
    cp = None

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
