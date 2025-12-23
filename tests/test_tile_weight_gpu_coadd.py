import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import zemosaic_utils as zu


def _weighted_coadd(data_list, input_weights, tile_weights, mode: str = "mean"):
    arrays = [np.asarray(arr, dtype=np.float32) for arr in data_list]
    base_weights: list[np.ndarray] = []
    if input_weights is None:
        base_weights = [np.ones_like(arr, dtype=np.float32) for arr in arrays]
    else:
        for idx, arr in enumerate(arrays):
            try:
                w = np.asarray(input_weights[idx], dtype=np.float32)
            except Exception:
                w = np.ones_like(arr, dtype=np.float32)
            if w.shape != arr.shape:
                w = np.ones_like(arr, dtype=np.float32)
            base_weights.append(np.where(np.isfinite(w), w, 0.0).astype(np.float32))

    weights = base_weights
    if tile_weights is not None:
        weights = []
        for idx, base in enumerate(base_weights):
            try:
                tw = float(tile_weights[idx]) if idx < len(tile_weights) else 1.0
            except Exception:
                tw = 1.0
            weights.append(base * tw)

    stack = np.stack(arrays, axis=0)
    weight_stack = np.stack(weights, axis=0)
    coverage = np.sum(weight_stack, axis=0)

    if mode in {"winsorized", "winsorized_sigma_clip", "winsor"}:
        limits = (0.05, 0.05)
        mosaic = np.zeros_like(arrays[0], dtype=np.float32)
        flat_vals = stack.reshape(stack.shape[0], -1)
        flat_w = weight_stack.reshape(weight_stack.shape[0], -1)
        out_flat = mosaic.reshape(-1)
        cov_flat = coverage.reshape(-1)
        for i in range(flat_vals.shape[1]):
            vals = flat_vals[:, i]
            wts = flat_w[:, i]
            valid = wts > 0
            if not np.any(valid):
                out_flat[i] = 0.0
                cov_flat[i] = 0.0
                continue
            vals_use = vals[valid]
            w_use = wts[valid]
            total_w = float(np.sum(w_use))
            if total_w <= 0.0:
                out_flat[i] = 0.0
                cov_flat[i] = 0.0
                continue
            order = np.argsort(vals_use)
            vals_sorted = vals_use[order]
            w_sorted = w_use[order]
            cumsum = np.cumsum(w_sorted)
            lower_idx = np.searchsorted(cumsum, limits[0] * total_w)
            upper_idx = np.searchsorted(cumsum, max(total_w * (1.0 - limits[1]), 0.0))
            lower_idx = min(lower_idx, len(vals_sorted) - 1)
            upper_idx = min(upper_idx, len(vals_sorted) - 1)
            lower = float(vals_sorted[lower_idx])
            upper = float(vals_sorted[upper_idx])
            clipped = np.clip(vals_use, lower, upper)
            out_flat[i] = float(np.average(clipped, weights=w_use))
            cov_flat[i] = float(np.sum(w_use))
        mosaic = out_flat.reshape(arrays[0].shape).astype(np.float32)
        coverage = cov_flat.reshape(arrays[0].shape).astype(np.float32)
    else:
        weighted_sum = np.sum(stack * weight_stack, axis=0)
        mosaic = np.where(coverage > 0.0, weighted_sum / np.maximum(coverage, 1e-6), 0.0).astype(np.float32)

    return mosaic, coverage.astype(np.float32)


def _max_weight_entry(weight_list) -> float:
    if weight_list is None:
        return 0.0
    max_val = 0.0
    for entry in weight_list:
        if entry is None:
            continue
        arr = np.asarray(entry, dtype=np.float32)
        if arr.size == 0:
            continue
        try:
            local = float(np.nanmax(arr))
        except Exception:
            local = 0.0
        if np.isfinite(local):
            max_val = max(max_val, local)
    return max_val


def test_tile_weight_gpu_coadd_mean_dominates_deep_tile(monkeypatch):
    """GPU coadd should honor tile_weights without inflating input weight maps."""

    rng = np.random.default_rng(123)
    tile_a = rng.normal(loc=0.0, scale=1.0, size=(16, 16)).astype(np.float32)
    tile_b = np.full((16, 16), 100.0, dtype=np.float32) + rng.normal(scale=0.5, size=(16, 16)).astype(np.float32)

    base_weights = [np.ones_like(tile_a, dtype=np.float32), np.ones_like(tile_b, dtype=np.float32)]
    tile_weights = [1.0, 100.0]

    def cpu_stub(inputs, output_proj, shape_out, **kwargs):
        data_only = [arr for arr, _ in inputs]
        weights = kwargs.get("input_weights")
        return _weighted_coadd(data_only, weights, None, mode=str(kwargs.get("stack_reject_algo") or "mean"))

    def gpu_stub(data_list, wcs_list, shape_out, **kwargs):
        iw = kwargs.get("input_weights")
        tw = kwargs.get("tile_weights")
        assert _max_weight_entry(iw) <= 1.5
        mosaic, cov = _weighted_coadd(
            data_list,
            iw,
            tw,
            mode=str(kwargs.get("stack_reject_algo") or kwargs.get("combine_function") or "mean"),
        )
        return mosaic, cov

    monkeypatch.setattr(zu, "gpu_is_available", lambda: True)
    monkeypatch.setattr(zu, "gpu_reproject_and_coadd_impl", gpu_stub)

    cpu_mosaic, cpu_cov = zu.reproject_and_coadd_wrapper(
        data_list=[tile_a, tile_b],
        wcs_list=[object(), object()],
        shape_out=tile_a.shape,
        output_projection="proj",
        use_gpu=False,
        cpu_func=cpu_stub,
        input_weights=base_weights,
        tile_weights=tile_weights,
    )
    gpu_mosaic, gpu_cov = zu.reproject_and_coadd_wrapper(
        data_list=[tile_a, tile_b],
        wcs_list=[object(), object()],
        shape_out=tile_a.shape,
        output_projection="proj",
        use_gpu=True,
        cpu_func=cpu_stub,
        input_weights=base_weights,
        tile_weights=tile_weights,
    )

    assert np.allclose(cpu_cov, gpu_cov)
    assert np.allclose(cpu_mosaic, gpu_mosaic, atol=1e-4)
    np.testing.assert_allclose(np.nanmedian(gpu_cov), sum(tile_weights), rtol=0.05)
    assert np.abs(np.nanmean(cpu_mosaic - tile_b)) < 1.0


def test_tile_weight_gpu_coadd_winsorized_respects_weights(monkeypatch):
    """Winsorized combine on GPU must still honor per-tile weights."""

    rng = np.random.default_rng(321)
    noise_tile = rng.normal(loc=0.0, scale=2.0, size=(12, 12)).astype(np.float32)
    signal_tile = np.full((12, 12), 80.0, dtype=np.float32) + rng.normal(scale=0.3, size=(12, 12)).astype(np.float32)

    base_weights = [np.ones_like(noise_tile, dtype=np.float32), np.ones_like(signal_tile, dtype=np.float32)]
    tile_weights = [1.0, 120.0]

    def cpu_stub(inputs, output_proj, shape_out, **kwargs):
        data_only = [arr for arr, _ in inputs]
        weights = kwargs.get("input_weights")
        return _weighted_coadd(data_only, weights, None, mode=str(kwargs.get("stack_reject_algo") or "winsorized"))

    def gpu_stub(data_list, wcs_list, shape_out, **kwargs):
        iw = kwargs.get("input_weights")
        tw = kwargs.get("tile_weights")
        assert _max_weight_entry(iw) <= 1.5
        return _weighted_coadd(
            data_list,
            iw,
            tw,
            mode=str(kwargs.get("stack_reject_algo") or kwargs.get("combine_function") or "winsorized"),
        )

    monkeypatch.setattr(zu, "gpu_is_available", lambda: True)
    monkeypatch.setattr(zu, "gpu_reproject_and_coadd_impl", gpu_stub)

    cpu_mosaic, cpu_cov = zu.reproject_and_coadd_wrapper(
        data_list=[noise_tile, signal_tile],
        wcs_list=[object(), object()],
        shape_out=noise_tile.shape,
        output_projection="proj",
        use_gpu=False,
        cpu_func=cpu_stub,
        input_weights=base_weights,
        tile_weights=tile_weights,
        stack_reject_algo="winsorized",
    )
    gpu_mosaic, gpu_cov = zu.reproject_and_coadd_wrapper(
        data_list=[noise_tile, signal_tile],
        wcs_list=[object(), object()],
        shape_out=noise_tile.shape,
        output_projection="proj",
        use_gpu=True,
        cpu_func=cpu_stub,
        input_weights=base_weights,
        tile_weights=tile_weights,
        stack_reject_algo="winsorized",
    )

    assert np.allclose(cpu_cov, gpu_cov)
    assert np.allclose(cpu_mosaic, gpu_mosaic, atol=1e-4)
    np.testing.assert_allclose(np.nanmedian(gpu_cov), sum(tile_weights), rtol=0.05)
    assert np.abs(np.nanmean(gpu_mosaic - signal_tile)) < 1.0
