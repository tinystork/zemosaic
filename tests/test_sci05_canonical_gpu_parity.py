"""SCI-05 Gate D — canonical rejection/combine CPU↔GPU parity (physical GPU).

Deterministic, hermetic physical parity witnesses for the backend-neutral
rejection (``none``/``kappa_sigma``/``winsorized_sigma_clip``) and combine
(``mean``/``median``) stages in ``zemosaic.core.canonical_stacking``.

One shared ``xp``-generic algorithm runs on NumPy (CPU, default) or CuPy (GPU,
explicit opt-in). These tests run on the real MX150 CUDA device (they
``pytest.importorskip("cupy")`` + device-count guard so they skip cleanly on
hosts without CuPy/GPU) and assert:

* exact survivor/rejection mask equality (bool, bit-identical),
* exact diagnostic integer equality,
* science / ``estimator_weight_sum`` parity within ``rtol=1e-12`` (float64
  reductions may sum in a different order on the GPU — a ~1e-15 effect; 1e-12 is
  a documented 1000× margin), and
* host-NumPy result contract (shapes/dtypes/ownership unchanged on GPU).

Deterministic float32 corpora only (no randomness with global seeds), no
network, no sleeps.
"""

from __future__ import annotations

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

import zemosaic.core.canonical_stacking as cs

# Tolerance for float64 science/estimator_weight_sum parity (documented): GPU
# tree reductions can differ from NumPy's pairwise summation by ~1 ULP; 1e-12
# relative is a 1000× margin over that while still catching real divergence.
SCIENCE_RTOL = 1e-12

_HAS_GPU = cs.canonical_gpu_available()
requires_gpu = pytest.mark.skipif(not _HAS_GPU, reason="no CUDA device available")


# ---------------------------------------------------------------------------
# Corpus builders (identical inputs fed to both backends)
# ---------------------------------------------------------------------------

def _full_support(h, w):
    return np.ones((h, w), dtype=bool)


def _norm(arrays, masks=None, method="none", reference_index=None):
    if masks is None:
        masks = [_full_support(*a.shape[:2]) for a in arrays]
    batch = cs.prepare_canonical_inputs(arrays, masks)
    return cs.normalize_canonical_images(batch, method, reference_index=reference_index)


def _weighted(norm, method="none"):
    return cs.compute_canonical_quality_weights(norm, method)


def _canonical_wmap(norm, weight, taper=None):
    n = norm.n_frames
    q = np.asarray(weight.weights, dtype=np.float64)
    m = np.asarray(norm.valid_mask, dtype=np.float64)
    if taper is None:
        a = np.ones((n, norm.height, norm.width), dtype=np.float64)
    else:
        a = np.asarray(taper, dtype=np.float64)
    return np.ascontiguousarray(q[:, None, None] * m * a)


def _values_corpus(values):
    return _norm([np.full((1, 1), float(v), dtype=np.float32) for v in values])


def _rgb_corpus(r, g, b):
    arrays = [
        np.array([[[float(r[i]), float(g[i]), float(b[i])]]], dtype=np.float32)
        for i in range(len(r))
    ]
    return _norm(arrays)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _assert_reject_parity(cpu, gpu):
    assert isinstance(cpu.survivor_mask, np.ndarray)
    assert isinstance(gpu.survivor_mask, np.ndarray)
    np.testing.assert_array_equal(cpu.survivor_mask, gpu.survivor_mask)
    np.testing.assert_array_equal(cpu.rejection_mask, gpu.rejection_mask)
    np.testing.assert_array_equal(cpu.active_frames, gpu.active_frames)
    assert cpu.reference_index == gpu.reference_index
    assert cpu.requested_method == cpu.effective_method == gpu.requested_method == gpu.effective_method
    assert cpu.sigma_low == gpu.sigma_low and cpu.sigma_high == gpu.sigma_high
    assert cpu.max_iters == gpu.max_iters
    assert cpu.winsor_limit_low == gpu.winsor_limit_low
    assert cpu.winsor_limit_high == gpu.winsor_limit_high
    assert cpu.iterations_used == gpu.iterations_used
    assert cpu.initial_sample_count == gpu.initial_sample_count
    assert cpu.surviving_sample_count == gpu.surviving_sample_count
    assert cpu.rejected_sample_count == gpu.rejected_sample_count
    assert cpu.rejected_fraction == gpu.rejected_fraction
    assert cpu.low_n_cell_count == gpu.low_n_cell_count
    assert cpu.degenerate_cell_count == gpu.degenerate_cell_count
    # GPU result is host NumPy with the unchanged owned/contiguous contract
    for res in (cpu, gpu):
        for arr in (res.survivor_mask, res.rejection_mask, res.active_frames):
            assert arr.flags["OWNDATA"] and arr.flags["C_CONTIGUOUS"]


def _assert_combine_parity(cpu, gpu):
    assert isinstance(cpu.science, np.ndarray) and isinstance(gpu.science, np.ndarray)
    np.testing.assert_array_equal(cpu.valid_mask, gpu.valid_mask)
    np.testing.assert_array_equal(cpu.surviving_sample_count, gpu.surviving_sample_count)
    np.testing.assert_allclose(
        cpu.science, gpu.science, rtol=SCIENCE_RTOL, atol=0.0, equal_nan=True
    )
    np.testing.assert_allclose(
        cpu.estimator_weight_sum, gpu.estimator_weight_sum,
        rtol=SCIENCE_RTOL, atol=0.0, equal_nan=True,
    )
    assert cpu.valid_output_count == gpu.valid_output_count
    assert cpu.invalid_output_count == gpu.invalid_output_count
    assert cpu.nonfinite_output_count == gpu.nonfinite_output_count
    assert cpu.input_surviving_sample_count == gpu.input_surviving_sample_count
    assert cpu.contributing_sample_count == gpu.contributing_sample_count
    assert cpu.requested_method == cpu.effective_method == gpu.requested_method == gpu.effective_method
    # GPU result is host NumPy with the unchanged owned/contiguous contract
    for res in (cpu, gpu):
        for arr in (res.science, res.estimator_weight_sum, res.valid_mask,
                    res.surviving_sample_count):
            assert arr.flags["OWNDATA"] and arr.flags["C_CONTIGUOUS"]


def _reject_pair(norm, weight, method, **kw):
    cpu = cs.reject_canonical_samples(norm, weight, method, backend="cpu", **kw)
    gpu = cs.reject_canonical_samples(norm, weight, method, backend="gpu", **kw)
    _assert_reject_parity(cpu, gpu)
    return cpu, gpu


def _combine_pair(norm, weight, rej, wmap, method):
    cpu = cs.combine_canonical_samples(norm, weight, rej, wmap, method, backend="cpu")
    gpu = cs.combine_canonical_samples(norm, weight, rej, wmap, method, backend="gpu")
    _assert_combine_parity(cpu, gpu)
    return cpu, gpu


# ---------------------------------------------------------------------------
# Availability / backend validation
# ---------------------------------------------------------------------------

class TestAvailability:
    def test_gpu_available_and_device(self):
        assert cs.canonical_gpu_available() is True
        import cupy as cp
        assert cp.cuda.runtime.getDeviceCount() >= 1
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        # recorded evidence for the report
        print(
            f"GPU_PARITY device={name!r} cupy={cp.__version__} "
            f"devices={cp.cuda.runtime.getDeviceCount()}"
        )

    def test_gpu_unavailable_raises_no_fallback(self, monkeypatch):
        monkeypatch.setattr(
            "zemosaic.core.canonical_stacking.canonical_gpu_available", lambda: False
        )
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        with pytest.raises(cs.CanonicalStackValidationError) as ei:
            cs.reject_canonical_samples(norm, weight, "kappa_sigma", backend="gpu")
        assert "unavailable" in str(ei.value)
        wmap = _canonical_wmap(norm, weight)
        rej = cs.reject_canonical_samples(norm, weight, "none")
        with pytest.raises(cs.CanonicalStackValidationError) as ei:
            cs.combine_canonical_samples(norm, weight, rej, wmap, "mean", backend="gpu")
        assert "unavailable" in str(ei.value)

    def test_unknown_backend_raises(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        wmap = _canonical_wmap(norm, weight)
        rej = cs.reject_canonical_samples(norm, weight, "none")
        # genuinely unknown/non-string backends (case is normalized: "Gpu" is
        # the valid opt-in token "gpu", not an alias of a different science)
        for bad in ("cuda", "CPUs", "", "gpu2", "cudagpu", 123, None, b"gpu"):
            with pytest.raises(cs.CanonicalStackValidationError):
                cs.reject_canonical_samples(norm, weight, "none", backend=bad)
            with pytest.raises(cs.CanonicalStackValidationError):
                cs.combine_canonical_samples(norm, weight, rej, wmap, "mean", backend=bad)
        # case/whitespace normalization is accepted (same token, same science)
        for ok in (" GPU ", "Gpu", "CPU"):
            r = cs.reject_canonical_samples(norm, weight, "none", backend=ok)
            assert r.requested_method == "none"

    def test_explicit_cpu_equals_default(self):
        # backend="cpu" and the default produce byte-identical CPU results
        norm = _values_corpus([10.0] * 9 + [100.0])
        weight = _weighted(norm, "none")
        default = cs.reject_canonical_samples(norm, weight, "kappa_sigma")
        explicit = cs.reject_canonical_samples(norm, weight, "kappa_sigma", backend="cpu")
        np.testing.assert_array_equal(default.survivor_mask, explicit.survivor_mask)
        np.testing.assert_array_equal(default.rejection_mask, explicit.rejection_mask)
        assert default.surviving_sample_count == explicit.surviving_sample_count


# ---------------------------------------------------------------------------
# Rejection parity
# ---------------------------------------------------------------------------

class TestRejectionParity:
    @requires_gpu
    def test_none(self):
        a = np.ones((2, 2), dtype=np.float32)
        norm = _norm([a, a, np.full((2, 2), np.nan, dtype=np.float32)])
        weight = _weighted(norm, "none")
        _reject_pair(norm, weight, "none")

    @requires_gpu
    def test_kappa_gross_outlier(self):
        norm = _values_corpus([10.0] * 9 + [100.0])
        weight = _weighted(norm, "none")
        _reject_pair(norm, weight, "kappa_sigma")

    @requires_gpu
    def test_kappa_asymmetric_sigmas(self):
        norm = _values_corpus([0.0] * 6 + [-100.0, 100.0])
        weight = _weighted(norm, "none")
        _reject_pair(norm, weight, "kappa_sigma", sigma_low=0.5, sigma_high=3.0)
        _reject_pair(norm, weight, "kappa_sigma", sigma_low=3.0, sigma_high=0.5)

    @requires_gpu
    def test_kappa_multi_iteration(self):
        norm = _values_corpus([0.0] * 8 + [40.0, 80.0])
        weight = _weighted(norm, "none")
        _reject_pair(norm, weight, "kappa_sigma")

    @requires_gpu
    def test_kappa_low_n_frozen(self):
        for vals in ([5.0], [5.0, 6.0]):
            norm = _values_corpus(vals)
            weight = _weighted(norm, "none")
            _reject_pair(norm, weight, "kappa_sigma")

    @requires_gpu
    def test_kappa_degenerate_equal(self):
        norm = _values_corpus([7.0] * 5)
        weight = _weighted(norm, "none")
        _reject_pair(norm, weight, "kappa_sigma")

    @requires_gpu
    def test_wsc_gross_outlier(self):
        norm = _values_corpus([10.0] * 9 + [100.0])
        weight = _weighted(norm, "none")
        _reject_pair(norm, weight, "winsorized_sigma_clip")

    @requires_gpu
    def test_wsc_custom_winsor_limits(self):
        norm = _values_corpus([0.0] * 8 + [-100.0, 100.0])
        weight = _weighted(norm, "none")
        _reject_pair(norm, weight, "winsorized_sigma_clip",
                     winsor_limit_low=0.25, winsor_limit_high=0.25)
        _reject_pair(norm, weight, "winsorized_sigma_clip",
                     winsor_limit_low=0.0, winsor_limit_high=0.3)

    @requires_gpu
    def test_wsc_degenerate_equal(self):
        norm = _values_corpus([7.0] * 5)
        weight = _weighted(norm, "none")
        _reject_pair(norm, weight, "winsorized_sigma_clip")

    @requires_gpu
    def test_rgb_channel_specific_survivor(self):
        r = [10.0] * 9 + [100.0]
        g = [10.0] * 10
        b = [10.0] * 10
        norm = _rgb_corpus(r, g, b)
        weight = _weighted(norm, "none")
        _reject_pair(norm, weight, "kappa_sigma")
        _reject_pair(norm, weight, "winsorized_sigma_clip")

    @requires_gpu
    def test_all_invalid_and_mixed_validity(self):
        arrays = []
        for _ in range(4):
            img = np.ones((2, 2), dtype=np.float32)
            img[1, 1] = np.nan  # one all-invalid cell
            arrays.append(img)
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        for method in ("none", "kappa_sigma", "winsorized_sigma_clip"):
            _reject_pair(norm, weight, method)


# ---------------------------------------------------------------------------
# Combine parity
# ---------------------------------------------------------------------------

class TestCombineParity:
    @requires_gpu
    def test_mean_weighted(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = cs.reject_canonical_samples(norm, weight, "none")
        wmap = np.array([[[0.1]], [[0.2]], [[0.3]]], dtype=np.float64)
        _combine_pair(norm, weight, rej, wmap, "mean")

    @requires_gpu
    def test_mean_spatial_weights(self):
        arrays = [
            np.array([[2.0, 6.0]], dtype=np.float32),
            np.array([[4.0, 10.0]], dtype=np.float32),
        ]
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        rej = cs.reject_canonical_samples(norm, weight, "none")
        wmap = np.array([[[0.5, 0.1]], [[0.5, 0.3]]], dtype=np.float64)
        _combine_pair(norm, weight, rej, wmap, "mean")

    @requires_gpu
    def test_mean_weights_bounds(self):
        # weights spanning [tiny, mid, near-1] within [0, 1]
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = cs.reject_canonical_samples(norm, weight, "none")
        wmap = np.array([[[1e-9]], [[0.5]], [[0.9999999999]]], dtype=np.float64)
        _combine_pair(norm, weight, rej, wmap, "mean")

    @requires_gpu
    def test_mean_channel_specific_survivor(self):
        r = [10.0] * 9 + [100.0]
        g = [10.0] * 10
        b = [10.0] * 10
        norm = _rgb_corpus(r, g, b)
        weight = _weighted(norm, "none")
        rej = cs.reject_canonical_samples(norm, weight, "kappa_sigma")
        wmap = _canonical_wmap(norm, weight)
        _combine_pair(norm, weight, rej, wmap, "mean")

    @requires_gpu
    def test_mean_tiny_positive_denominator(self):
        norm = _values_corpus([1.0, 1.0])
        weight = _weighted(norm, "none")
        rej = cs.reject_canonical_samples(norm, weight, "none")
        wmap = np.full((2, 1, 1), 1e-300, dtype=np.float64)
        _combine_pair(norm, weight, rej, wmap, "mean")

    @requires_gpu
    def test_mean_zero_denominator_all_invalid(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = cs.reject_canonical_samples(norm, weight, "none")
        wmap = np.zeros((3, 1, 1), dtype=np.float64)
        _combine_pair(norm, weight, rej, wmap, "mean")

    @requires_gpu
    def test_mean_constant_field_varying_weights(self):
        arrays = [np.full((3, 3), 5.0, dtype=np.float32) for _ in range(4)]
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        rej = cs.reject_canonical_samples(norm, weight, "none")
        wmap = np.array([
            [[1.0, 0.5, 0.2], [0.8, 0.3, 0.1], [0.9, 0.6, 0.4]],
            [[0.1, 0.9, 0.7], [0.4, 1.0, 0.2], [0.3, 0.8, 0.5]],
            [[0.6, 0.2, 0.9], [0.7, 0.5, 1.0], [0.4, 0.1, 0.8]],
            [[0.5, 0.8, 0.1], [0.2, 0.6, 0.9], [1.0, 0.7, 0.3]],
        ], dtype=np.float64)
        _combine_pair(norm, weight, rej, wmap, "mean")

    @requires_gpu
    def test_median_odd_even(self):
        for vals in ([1.0, 3.0, 2.0], [1.0, 4.0, 2.0, 3.0]):
            norm = _values_corpus(vals)
            weight = _weighted(norm, "none")
            rej = cs.reject_canonical_samples(norm, weight, "none")
            wmap = _canonical_wmap(norm, weight)
            _combine_pair(norm, weight, rej, wmap, "median")

    @requires_gpu
    def test_median_magnitude_ignored(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = cs.reject_canonical_samples(norm, weight, "none")
        wmap = np.array([[[1e-9]], [[0.999]], [[1e-6]]], dtype=np.float64)
        _combine_pair(norm, weight, rej, wmap, "median")

    @requires_gpu
    def test_median_zero_weight_gates(self):
        norm = _values_corpus([1.0, 3.0, 2.0])
        weight = _weighted(norm, "none")
        rej = cs.reject_canonical_samples(norm, weight, "none")
        wmap = np.array([[[1.0]], [[1.0]], [[0.0]]], dtype=np.float64)
        _combine_pair(norm, weight, rej, wmap, "median")

    @requires_gpu
    def test_median_channel_specific_survivor(self):
        r = [10.0] * 9 + [100.0]
        g = [10.0] * 10
        b = [10.0] * 10
        norm = _rgb_corpus(r, g, b)
        weight = _weighted(norm, "none")
        rej = cs.reject_canonical_samples(norm, weight, "kappa_sigma")
        wmap = _canonical_wmap(norm, weight)
        _combine_pair(norm, weight, rej, wmap, "median")


# ---------------------------------------------------------------------------
# End-to-end + shape restoration on GPU
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @requires_gpu
    def test_full_pipeline_gpu_matches_cpu(self):
        base = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        frames = [base + float(k) for k in range(9)] + [
            np.full((2, 2), 1000.0, dtype=np.float32)
        ]
        norm = _norm(frames)
        weight = _weighted(norm, "none")
        wmap = _canonical_wmap(norm, weight)
        for reject_method in ("kappa_sigma", "winsorized_sigma_clip"):
            cpu_rej = cs.reject_canonical_samples(norm, weight, reject_method, backend="cpu")
            gpu_rej = cs.reject_canonical_samples(norm, weight, reject_method, backend="gpu")
            _assert_reject_parity(cpu_rej, gpu_rej)
            for combine_method in ("mean", "median"):
                cpu = cs.combine_canonical_samples(
                    norm, weight, cpu_rej, wmap, combine_method, backend="cpu"
                )
                gpu = cs.combine_canonical_samples(
                    norm, weight, gpu_rej, wmap, combine_method, backend="gpu"
                )
                _assert_combine_parity(cpu, gpu)

    @requires_gpu
    def test_shape_restoration_on_gpu(self):
        # HW mono -> (H, W)
        mono = _norm([np.ones((2, 3), dtype=np.float32)] * 3)
        w_mono = _weighted(mono, "none")
        rej_mono = cs.reject_canonical_samples(mono, w_mono, "none")
        wm_mono = _canonical_wmap(mono, w_mono)
        gpu = cs.combine_canonical_samples(mono, w_mono, rej_mono, wm_mono, "mean", backend="gpu")
        assert gpu.science.shape == (2, 3)
        assert gpu.science.dtype == np.float32
        assert gpu.estimator_weight_sum.dtype == np.float64
        assert gpu.valid_mask.dtype == np.bool_
        assert gpu.surviving_sample_count.dtype == np.int64

        # HWC1 stays (H, W, 1)
        hwc1 = _norm([np.ones((2, 3, 1), dtype=np.float32)] * 3)
        w1 = _weighted(hwc1, "none")
        rej1 = cs.reject_canonical_samples(hwc1, w1, "none")
        wm1 = _canonical_wmap(hwc1, w1)
        gpu1 = cs.combine_canonical_samples(hwc1, w1, rej1, wm1, "median", backend="gpu")
        assert gpu1.science.shape == (2, 3, 1)

        # RGB stays (H, W, 3)
        rgb = _norm([np.ones((2, 3, 3), dtype=np.float32)] * 3)
        w3 = _weighted(rgb, "none")
        rej3 = cs.reject_canonical_samples(rgb, w3, "none")
        wm3 = _canonical_wmap(rgb, w3)
        gpu3 = cs.combine_canonical_samples(rgb, w3, rej3, wm3, "mean", backend="gpu")
        assert gpu3.science.shape == (2, 3, 3)


# ---------------------------------------------------------------------------
# R1: bit-exact CuPy quantile vs np.nanquantile (two-sided _lerp)
# ---------------------------------------------------------------------------

class TestQuantileBitExact:
    """The CuPy NaN-aware quantile helper must match ``np.nanquantile`` bit-for-bit
    for **every** interpolation fraction, including ``frac >= 0.5`` (NumPy's
    two-sided ``_lerp``), not just the all-equal/single-value cases.
    """

    @requires_gpu
    def test_quantile_bit_exact_vs_numpy(self):
        import cupy as cp

        rng = np.random.default_rng(1234)
        # adversarial columns: all-equal, single-valid, all-NaN, sorted asc/desc,
        # unsorted, monotone neg->pos, huge magnitudes, tiny magnitudes.
        base_columns = [
            np.array([[7.0], [7.0], [7.0], [7.0], [7.0]]),
            np.array([[3.0], [np.nan], [np.nan], [np.nan], [np.nan]]),
            np.array([[np.nan], [np.nan], [np.nan], [np.nan], [np.nan]]),
            np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
            np.array([[5.0], [4.0], [3.0], [2.0], [1.0]]),
            np.array([[1.0], [3.0], [2.0], [5.0], [4.0]]),
            np.array([[-3.0], [-2.0], [-1.0], [0.0], [1.0]]),
            np.array([[1e300], [2e300], [3e300], [4e300], [5e300]]),
            np.array([[1e-300], [2e-300], [3e-300], [4e-300], [5e-300]]),
        ]
        cols = np.hstack(base_columns).astype(np.float64)
        rand = rng.normal(size=(5, 12)).astype(np.float64)
        rand[rng.random((5, 12)) < 0.3] = np.nan  # NaN masking
        a = np.hstack([cols, rand])

        qs = [0.05, 0.25, 0.5, 0.75, 0.95, 0.1, 0.3, 0.6, 0.9, 0.0, 1.0]
        for q in qs:
            cpu = np.nanquantile(a, q, axis=0, method="linear")
            gpu = cp.asnumpy(cs._nan_axis_quantile_gpu(cp.asarray(a), q, cp))
            # bit-exact (assert_array_equal treats NaN == NaN as equal)
            np.testing.assert_array_equal(cpu, gpu)

    @requires_gpu
    def test_quantile_bit_exact_winsor_limits(self):
        import cupy as cp

        # Real WSC winsor-limit quantile pairs (winsor_limit_low, 1-winsor_limit_high)
        # are the exact q values the rejection stage feeds this helper.
        rng = np.random.default_rng(7)
        a = rng.normal(size=(10, 6)).astype(np.float64)
        a[rng.random((10, 6)) < 0.25] = np.nan
        for q in (0.0, 0.05, 0.25, 0.3, 0.5, 0.7, 0.75, 0.95, 1.0):
            cpu = np.nanquantile(a, q, axis=0, method="linear")
            gpu = cp.asnumpy(cs._nan_axis_quantile_gpu(cp.asarray(a), q, cp))
            np.testing.assert_array_equal(cpu, gpu)
