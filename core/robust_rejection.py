from __future__ import annotations

import os
from typing import Any

import numpy as np

WSC_IMPL_PIXINSIGHT = "pixinsight"
WSC_IMPL_LEGACY = "legacy_quantile"
WSC_IMPL_ENV = "ZEMOSAIC_WSC_IMPL"
WSC_PARITY_ENV = "ZEMOSAIC_WSC_PARITY"


def resolve_wsc_impl(zconfig: Any | None = None) -> str:
    """Resolve the winsorized sigma clip implementation (env > config > default)."""

    env = os.environ.get(WSC_IMPL_ENV, "").strip().lower()
    if env in {WSC_IMPL_PIXINSIGHT, WSC_IMPL_LEGACY}:
        return env

    if zconfig is not None:
        for key in ("wsc_impl", "winsor_impl", "stack_winsor_impl"):
            try:
                val = getattr(zconfig, key)
            except Exception:
                val = None
            if val is None and isinstance(zconfig, dict):
                val = zconfig.get(key)
            if isinstance(val, str):
                val_norm = val.strip().lower()
                if val_norm in {WSC_IMPL_PIXINSIGHT, WSC_IMPL_LEGACY}:
                    return val_norm

    return WSC_IMPL_PIXINSIGHT


def resolve_wsc_parity_mode() -> str:
    """Return STRICT (default) or NUMERIC from env."""

    val = os.environ.get(WSC_PARITY_ENV, "").strip().upper()
    if val in {"STRICT", "NUMERIC"}:
        return val
    return "STRICT"


def _broadcast_frame_weights(
    xp: Any,
    weights_block: Any | None,
    target_shape: tuple[int, ...],
) -> Any | None:
    if weights_block is None:
        return None
    try:
        w = xp.asarray(weights_block)
    except Exception:
        return None

    if not target_shape or w.shape[0] != target_shape[0]:
        return None

    if w.ndim == 2 and w.shape[1] == 1:
        w = w.reshape((w.shape[0],))

    if w.ndim == 1:
        extra = (1,) * (len(target_shape) - 1)
        w = w.reshape((w.shape[0],) + extra)
        return xp.broadcast_to(w, target_shape)

    if w.ndim == 3 and len(target_shape) == 4 and w.shape[1:] == (1, 1):
        w = w.reshape((w.shape[0], 1, 1, 1))

    if w.ndim == len(target_shape):
        allowed = True
        for idx in range(1, len(target_shape)):
            dim = w.shape[idx]
            if idx == len(target_shape) - 1 and len(target_shape) == 4:
                if dim not in (1, target_shape[idx]):
                    allowed = False
            else:
                if dim != 1:
                    allowed = False
        if allowed:
            return xp.broadcast_to(w, target_shape)

    return None


def wsc_pixinsight_core(
    xp: Any,
    X_block: Any,
    *,
    sigma_low: float,
    sigma_high: float,
    max_iters: int = 10,
    weights_block: Any | None = None,
    huber: bool = True,
    huber_c: float | None = None,
    return_stats: bool = False,
) -> Any:
    """
    PixInsight-like winsorized sigma clipping core.

    Parameters
    ----------
    xp:
        Backend module: numpy or cupy.
    X_block:
        Input stack shaped (N, H, W) or (N, H, W, C).
    sigma_low/high:
        Lower/upper sigma bounds for winsorization.
    weights_block:
        Optional per-frame weights: (N,), (N,1,1[,1]), or (N,1,1,C).
    huber:
        Enable Huber IRLS scale update.
    return_stats:
        When True, returns (result, stats_dict).

    Notes
    -----
    Internal math runs in float64 for CPU/GPU parity; output is float32.
    """

    X = xp.asarray(X_block)
    if X.ndim not in (3, 4):
        raise ValueError(f"WSC expects (N,H,W[,C]) stack; got shape {X.shape}")

    Xf = X.astype(xp.float64, copy=False)
    valid = xp.isfinite(Xf)
    Xf = xp.where(valid, Xf, xp.nan)

    count_valid = xp.sum(valid, axis=0)
    fallback_mean = xp.nanmean(Xf, axis=0)
    active = count_valid >= 2

    m = xp.nanmedian(Xf, axis=0)
    mad = xp.nanmedian(xp.abs(Xf - m), axis=0)
    sigma = 1.4826 * mad
    sigma = xp.where(sigma <= 0, 1e-10, sigma)

    if huber_c is None:
        huber_c = max(float(sigma_low), float(sigma_high))

    weights = _broadcast_frame_weights(xp, weights_block, X.shape)
    if weights is not None:
        weights = weights.astype(xp.float64, copy=False)
        weights = xp.where(xp.isfinite(weights), weights, 0.0)

    prev_lo = None
    prev_hi = None
    iters_used = 0

    for it in range(int(max_iters)):
        lo = m - (float(sigma_low) * sigma)
        hi = m + (float(sigma_high) * sigma)
        Xw = xp.clip(Xf, lo, hi)
        Xw = xp.where(valid, Xw, xp.nan)

        if weights is None:
            m_new = xp.nanmean(Xw, axis=0)
        else:
            w_masked = xp.where(valid, weights, 0.0)
            Xw_safe = xp.where(valid, Xw, 0.0)
            sum_w = xp.sum(w_masked, axis=0)
            sum_x = xp.sum(Xw_safe * w_masked, axis=0)
            m_new = xp.where(sum_w > 0, sum_x / sum_w, xp.nan)

        r = Xw - m_new
        r_safe = xp.where(valid, r, 0.0)

        if huber:
            sigma_safe = xp.where(sigma <= 0, 1e-10, sigma)
            u = r_safe / sigma_safe
            abs_u = xp.abs(u)
            w_huber = xp.ones_like(abs_u)
            mask = abs_u > huber_c
            try:
                xp.divide(huber_c, abs_u, out=w_huber, where=mask)
            except TypeError:
                w_huber = xp.where(mask, huber_c / abs_u, 1.0)
            w_huber = xp.where(valid, w_huber, 0.0)
            w_total = w_huber if weights is None else (w_huber * w_masked)
            denom = xp.sum(w_total, axis=0)
            numer = xp.sum(w_total * (r_safe ** 2), axis=0)
            sigma_new = xp.sqrt(xp.where(denom > 0, numer / denom, 0.0))
        else:
            if weights is None:
                sigma_new = xp.sqrt(xp.nanmean(r ** 2, axis=0))
            else:
                denom = xp.sum(w_masked, axis=0)
                numer = xp.sum(w_masked * (r_safe ** 2), axis=0)
                sigma_new = xp.sqrt(xp.where(denom > 0, numer / denom, 0.0))

        sigma_new = xp.where(sigma_new <= 0, 1e-10, sigma_new)

        lo_cmp = xp.where(active, lo, 0.0)
        hi_cmp = xp.where(active, hi, 0.0)
        if prev_lo is not None and xp.array_equal(lo_cmp, prev_lo) and xp.array_equal(hi_cmp, prev_hi):
            iters_used = it + 1
            m = m_new
            sigma = sigma_new
            break

        diff_m = xp.abs(m_new - m)
        diff_s = xp.abs(sigma_new - sigma)
        tol_m = 5e-4 * xp.maximum(1.0, xp.abs(m))
        tol_s = 5e-4 * xp.maximum(1.0, xp.abs(sigma))

        diff_m = xp.where(active, diff_m, 0.0)
        diff_s = xp.where(active, diff_s, 0.0)
        tol_m = xp.where(active, tol_m, xp.inf)
        tol_s = xp.where(active, tol_s, xp.inf)

        max_m = float(xp.nanmax(diff_m - tol_m))
        max_s = float(xp.nanmax(diff_s - tol_s))
        m = m_new
        sigma = sigma_new
        prev_lo = lo_cmp
        prev_hi = hi_cmp
        if max_m <= 0.0 and max_s <= 0.0:
            iters_used = it + 1
            break
    else:
        iters_used = int(max_iters)

    output = xp.where(active, m, fallback_mean)
    output = output.astype(xp.float32, copy=False)

    if not return_stats:
        return output

    final_lo = m - (float(sigma_low) * sigma)
    final_hi = m + (float(sigma_high) * sigma)
    total_valid = xp.sum(valid)
    low_mask = valid & (Xf < final_lo)
    high_mask = valid & (Xf > final_hi)
    low_count = xp.sum(low_mask)
    high_count = xp.sum(high_mask)
    total_val = float(total_valid) if float(total_valid) > 0 else 0.0
    low_frac = float(low_count) / total_val if total_val > 0 else 0.0
    high_frac = float(high_count) / total_val if total_val > 0 else 0.0

    stats = {
        "iters_used": int(iters_used),
        "max_iters": int(max_iters),
        "huber": bool(huber),
        "clip_low_frac": float(low_frac),
        "clip_high_frac": float(high_frac),
        "clip_low_count": int(float(low_count)),
        "clip_high_count": int(float(high_count)),
        "valid_count": int(float(total_valid)),
    }
    return output, stats


def wsc_parity_check(
    xp: Any,
    *,
    seed: int = 13,
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    max_iters: int = 10,
) -> tuple[bool, float]:
    """
    Compare CPU vs GPU WSC on a deterministic tiny stack.

    Returns (ok, max_abs_diff) on float32 outputs.
    """

    rng = np.random.default_rng(seed)
    data = rng.normal(loc=100.0, scale=5.0, size=(6, 4, 4, 3)).astype(np.float32)
    data[0, 0, 0, 0] = np.nan
    data[1, 1, 1, 2] = np.inf

    cpu_out = wsc_pixinsight_core(
        np,
        data,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        max_iters=max_iters,
    ).astype(np.float32, copy=False)

    gpu_in = xp.asarray(data)
    gpu_out = wsc_pixinsight_core(
        xp,
        gpu_in,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        max_iters=max_iters,
    )
    # CuPy forbids implicit conversion to NumPy; use an explicit copy for parity checks.
    if hasattr(gpu_out, "get"):
        gpu_out = gpu_out.get()
    gpu_out = np.asarray(gpu_out, dtype=np.float32)
    diff = np.abs(cpu_out.astype(np.float32) - gpu_out.astype(np.float32))
    max_abs = float(np.nanmax(diff)) if diff.size else 0.0

    mode = resolve_wsc_parity_mode()
    if mode == "STRICT":
        ok = max_abs == 0.0
    else:
        ok = max_abs <= 2e-7
    return ok, max_abs
