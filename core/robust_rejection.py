from __future__ import annotations

import os
from typing import Any, Sequence

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
    try:
        writeable = getattr(getattr(Xf, "flags", None), "writeable", True)
        if writeable:
            Xf[~valid] = xp.nan
        else:
            Xf = xp.where(valid, Xf, xp.nan)
    except Exception:
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


def _wsc_stream_sample_indices(n_frames: int, limit: int) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros((0,), dtype=int)
    count = min(max(1, int(limit)), n_frames)
    if count <= 1:
        return np.array([0], dtype=int)
    idx = np.linspace(0, n_frames - 1, num=count)
    return np.unique(idx.astype(int))


def _wsc_stream_stack_block(
    frames: Sequence[np.ndarray],
    indices: Sequence[int],
    rows_slice: slice,
    channel: int | None,
) -> np.ndarray:
    block: list[np.ndarray] = []
    for idx in indices:
        frame = np.asarray(frames[idx])
        if frame.ndim == 3:
            ch = channel if channel is not None else 0
            slice_arr = frame[rows_slice, :, ch]
        else:
            slice_arr = frame[rows_slice, :]
        block.append(np.asarray(slice_arr, dtype=np.float32))
    return np.stack(block, axis=0)


def _wsc_stream_weights_block(
    weights_block: Any,
    indices: Sequence[int],
    channel: int | None,
) -> np.ndarray | None:
    if weights_block is None:
        return None
    try:
        w = np.asarray(weights_block)[list(indices)]
    except Exception:
        return None
    if w.ndim == 4:
        if w.shape[-1] > 1:
            ch = channel if channel is not None else 0
            w = w[..., ch]
        else:
            w = w[..., 0]
    if w.ndim == 3:
        if w.shape[1:] == (1, 1):
            w = w.reshape((w.shape[0], 1, 1))
        else:
            w = np.nanmean(w, axis=(1, 2), keepdims=True)
    elif w.ndim == 2:
        if w.shape[1] == 1:
            w = w.reshape((w.shape[0], 1, 1))
        else:
            if channel is not None and w.shape[1] > channel:
                w = w[:, channel].reshape((w.shape[0], 1, 1))
            else:
                w = np.nanmean(w, axis=1).reshape((w.shape[0], 1, 1))
    elif w.ndim == 1:
        w = w.reshape((w.shape[0], 1, 1))
    else:
        return None
    w = w.astype(np.float64, copy=False)
    return np.where(np.isfinite(w), w, 0.0)


def wsc_pixinsight_core_streaming_numpy(
    frames: Sequence[np.ndarray],
    *,
    rows_slice: slice,
    channel: int | None,
    sigma_low: float,
    sigma_high: float,
    max_iters: int = 10,
    weights_block: Any | None = None,
    sky_offsets: Any | None = None,
    huber: bool = True,
    huber_c: float | None = None,
    sample_limit: int = 256,
    block_size: int = 128,
    return_stats: bool = False,
) -> Any:
    if not frames:
        raise ValueError("WSC streaming expects non-empty frames")

    n_frames = len(frames)
    sample_indices = _wsc_stream_sample_indices(n_frames, sample_limit)
    if sample_indices.size == 0:
        raise ValueError("WSC streaming could not select sample indices")

    sample_block = _wsc_stream_stack_block(frames, sample_indices.tolist(), rows_slice, channel)
    if sky_offsets is not None:
        try:
            off = np.asarray(sky_offsets, dtype=np.float32)[sample_indices]
            off = off.reshape((sample_block.shape[0], 1, 1))
            sample_block = sample_block + off
        except Exception:
            pass
    try:
        np.nan_to_num(sample_block, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    except Exception:
        pass

    sample_block = sample_block.astype(np.float64, copy=False)
    m = np.nanmedian(sample_block, axis=0)
    mad = np.nanmedian(np.abs(sample_block - m), axis=0)
    sigma = 1.4826 * mad
    sigma = np.where(sigma <= 0, 1e-10, sigma)

    if huber_c is None:
        huber_c = max(float(sigma_low), float(sigma_high))

    fallback_sum = None
    fallback_count = None
    active = None
    prev_lo = None
    prev_hi = None
    iters_used = 0

    block = max(1, int(block_size))
    frames_range = range(0, n_frames, block)
    use_weights = weights_block is not None
    if use_weights:
        try:
            probe = _wsc_stream_weights_block(weights_block, [0], channel)
        except Exception:
            probe = None
        if probe is None:
            use_weights = False
    for it in range(int(max_iters)):
        lo = m - (float(sigma_low) * sigma)
        hi = m + (float(sigma_high) * sigma)

        sum_x = np.zeros_like(m, dtype=np.float64)
        count = np.zeros_like(m, dtype=np.float64)
        sum_w = np.zeros_like(m, dtype=np.float64) if use_weights else None

        compute_fallback = fallback_sum is None
        if compute_fallback:
            fallback_sum = np.zeros_like(m, dtype=np.float64)
            fallback_count = np.zeros_like(m, dtype=np.int64)

        for start in frames_range:
            end = min(n_frames, start + block)
            indices = list(range(start, end))
            chunk = _wsc_stream_stack_block(frames, indices, rows_slice, channel)
            if sky_offsets is not None:
                try:
                    off = np.asarray(sky_offsets, dtype=np.float32)[start:end].reshape((chunk.shape[0], 1, 1))
                    chunk = chunk + off
                except Exception:
                    pass
            try:
                np.nan_to_num(chunk, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
            except Exception:
                pass

            valid = np.isfinite(chunk)
            if compute_fallback and fallback_sum is not None and fallback_count is not None:
                fallback_sum += np.nansum(chunk, axis=0, dtype=np.float64)
                fallback_count += valid.sum(axis=0, dtype=np.int64)

            Xf = chunk.astype(np.float64, copy=False)
            Xw = np.clip(Xf, lo, hi)
            Xw_safe = np.where(valid, Xw, 0.0)

            if use_weights and sum_w is not None:
                w_block = _wsc_stream_weights_block(weights_block, indices, channel)
                if w_block is None:
                    w_block = np.ones((chunk.shape[0], 1, 1), dtype=np.float64)
                w_masked = w_block * valid
                sum_w += np.sum(w_masked, axis=0, dtype=np.float64)
                sum_x += np.sum(Xw_safe * w_block, axis=0, dtype=np.float64)
            else:
                sum_x += np.sum(Xw_safe, axis=0, dtype=np.float64)
                count += valid.sum(axis=0, dtype=np.float64)

        if active is None and fallback_count is not None:
            active = fallback_count >= 2

        if not use_weights or sum_w is None:
            m_new = np.where(count > 0, sum_x / count, np.nan)
        else:
            m_new = np.where(sum_w > 0, sum_x / sum_w, np.nan)

        denom = np.zeros_like(m, dtype=np.float64)
        numer = np.zeros_like(m, dtype=np.float64)
        sigma_safe = np.where(sigma <= 0, 1e-10, sigma)

        for start in frames_range:
            end = min(n_frames, start + block)
            indices = list(range(start, end))
            chunk = _wsc_stream_stack_block(frames, indices, rows_slice, channel)
            if sky_offsets is not None:
                try:
                    off = np.asarray(sky_offsets, dtype=np.float32)[start:end].reshape((chunk.shape[0], 1, 1))
                    chunk = chunk + off
                except Exception:
                    pass
            try:
                np.nan_to_num(chunk, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
            except Exception:
                pass

            valid = np.isfinite(chunk)
            Xf = chunk.astype(np.float64, copy=False)
            Xw = np.clip(Xf, lo, hi)
            r_safe = np.where(valid, Xw - m_new, 0.0)

            if huber:
                u = r_safe / sigma_safe
                abs_u = np.abs(u)
                w_huber = np.ones_like(abs_u)
                mask = abs_u > huber_c
                try:
                    np.divide(huber_c, abs_u, out=w_huber, where=mask)
                except TypeError:
                    w_huber = np.where(mask, huber_c / abs_u, 1.0)
                w_huber = np.where(valid, w_huber, 0.0)
                if not use_weights:
                    w_total = w_huber
                else:
                    w_block = _wsc_stream_weights_block(weights_block, indices, channel)
                    if w_block is None:
                        w_block = np.ones((chunk.shape[0], 1, 1), dtype=np.float64)
                    w_total = w_huber * (w_block * valid)
                denom += np.sum(w_total, axis=0, dtype=np.float64)
                numer += np.sum(w_total * (r_safe ** 2), axis=0, dtype=np.float64)
            else:
                if not use_weights:
                    denom += valid.sum(axis=0, dtype=np.float64)
                    numer += np.sum(r_safe ** 2, axis=0, dtype=np.float64)
                else:
                    w_block = _wsc_stream_weights_block(weights_block, indices, channel)
                    if w_block is None:
                        w_block = np.ones((chunk.shape[0], 1, 1), dtype=np.float64)
                    w_masked = w_block * valid
                    denom += np.sum(w_masked, axis=0, dtype=np.float64)
                    numer += np.sum(w_masked * (r_safe ** 2), axis=0, dtype=np.float64)

        sigma_new = np.sqrt(np.where(denom > 0, numer / denom, 0.0))
        sigma_new = np.where(sigma_new <= 0, 1e-10, sigma_new)

        if active is None:
            active = np.ones_like(m_new, dtype=bool)

        lo_cmp = np.where(active, lo, 0.0)
        hi_cmp = np.where(active, hi, 0.0)
        if prev_lo is not None and np.array_equal(lo_cmp, prev_lo) and np.array_equal(hi_cmp, prev_hi):
            iters_used = it + 1
            m = m_new
            sigma = sigma_new
            break

        diff_m = np.abs(m_new - m)
        diff_s = np.abs(sigma_new - sigma)
        tol_m = 5e-4 * np.maximum(1.0, np.abs(m))
        tol_s = 5e-4 * np.maximum(1.0, np.abs(sigma))

        diff_m = np.where(active, diff_m, 0.0)
        diff_s = np.where(active, diff_s, 0.0)
        tol_m = np.where(active, tol_m, np.inf)
        tol_s = np.where(active, tol_s, np.inf)

        max_m = float(np.nanmax(diff_m - tol_m)) if diff_m.size else 0.0
        max_s = float(np.nanmax(diff_s - tol_s)) if diff_s.size else 0.0

        m = m_new
        sigma = sigma_new
        prev_lo = lo_cmp
        prev_hi = hi_cmp
        if max_m <= 0.0 and max_s <= 0.0:
            iters_used = it + 1
            break
    else:
        iters_used = int(max_iters)

    if fallback_sum is None or fallback_count is None:
        fallback_sum = np.zeros_like(m, dtype=np.float64)
        fallback_count = np.zeros_like(m, dtype=np.int64)
        for start in frames_range:
            end = min(n_frames, start + block)
            indices = list(range(start, end))
            chunk = _wsc_stream_stack_block(frames, indices, rows_slice, channel)
            if sky_offsets is not None:
                try:
                    off = np.asarray(sky_offsets, dtype=np.float32)[start:end].reshape((chunk.shape[0], 1, 1))
                    chunk = chunk + off
                except Exception:
                    pass
            try:
                np.nan_to_num(chunk, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
            except Exception:
                pass
            valid = np.isfinite(chunk)
            fallback_sum += np.nansum(chunk, axis=0, dtype=np.float64)
            fallback_count += valid.sum(axis=0, dtype=np.int64)

    with np.errstate(divide="ignore", invalid="ignore"):
        fallback_mean = np.divide(
            fallback_sum,
            fallback_count,
            out=np.zeros_like(fallback_sum, dtype=np.float64),
            where=fallback_count > 0,
        )
    fallback_mean = np.where(fallback_count > 0, fallback_mean, np.nan)
    active = fallback_count >= 2
    output = np.where(active, m, fallback_mean)
    output = output.astype(np.float32, copy=False)

    if not return_stats:
        return output

    final_lo = m - (float(sigma_low) * sigma)
    final_hi = m + (float(sigma_high) * sigma)
    total_valid = float(np.sum(fallback_count)) if fallback_count is not None else 0.0
    low_count = 0.0
    high_count = 0.0

    for start in frames_range:
        end = min(n_frames, start + block)
        indices = list(range(start, end))
        chunk = _wsc_stream_stack_block(frames, indices, rows_slice, channel)
        if sky_offsets is not None:
            try:
                off = np.asarray(sky_offsets, dtype=np.float32)[start:end].reshape((chunk.shape[0], 1, 1))
                chunk = chunk + off
            except Exception:
                pass
        try:
            np.nan_to_num(chunk, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
        except Exception:
            pass
        valid = np.isfinite(chunk)
        Xf = chunk.astype(np.float64, copy=False)
        low_count += float(np.sum(valid & (Xf < final_lo)))
        high_count += float(np.sum(valid & (Xf > final_hi)))

    low_frac = float(low_count) / total_valid if total_valid > 0 else 0.0
    high_frac = float(high_count) / total_valid if total_valid > 0 else 0.0

    stats = {
        "iters_used": int(iters_used),
        "max_iters": int(max_iters),
        "huber": bool(huber),
        "clip_low_frac": float(low_frac),
        "clip_high_frac": float(high_frac),
        "clip_low_count": int(low_count),
        "clip_high_count": int(high_count),
        "valid_count": int(total_valid),
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
