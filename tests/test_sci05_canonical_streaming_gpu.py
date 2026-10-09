"""ZM-ZEGRID-R22 — streaming CPU↔GPU bit-exactness (physical GPU, opt-in).

Verifies the VRAM-bounded tiled GPU path in
``zemosaic.core.canonical_streaming.run_canonical_stack_streaming(backend="gpu")``
is BIT-IDENTICAL to the CPU path for rejection masks / counts / science / weights
/ support. Skips cleanly without CuPy/GPU.

The GPU path uses the bit-exact host-side reductions (see
``canonical_stacking._nan_axis_mean`` / ``_nan_axis_popstd`` / ``_axis_sum_f64``),
so — unlike the raw CuPy tree reductions — the results are bit-identical, not
merely within 1e-12.
"""

from __future__ import annotations

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

import zemosaic.core.canonical_stacking as cs
from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)

_HAS_GPU = cs.canonical_gpu_available()
requires_gpu = pytest.mark.skipif(not _HAS_GPU, reason="no CUDA device available")


def _full_support(h, w):
    return np.ones((h, w), dtype=bool)


def _corpus(n=9, h=24, w=24, seed=5):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        f = rng.normal(0, 3, (h, w, 3)).astype(np.float32) + np.array([50, 55, 60], np.float32) + 2.0 * i
        f[rng.random((h, w, 3)) < 0.1] = np.nan
        frames.append(f)
    frames[2][1, 1, :] = 1000.0  # outlier
    masks = [_full_support(h, w)] * n
    masks[1][4:8, 4:8] = False
    masks[3][:, 15:] = False
    return frames, masks


def _run(arrays, masks, backend, tile_size, **kw):
    req = CanonicalStackRequest(images=arrays, geometric_support=masks, backend=backend, **kw)
    prov = InMemoryCanonicalProvider(arrays, masks)
    return run_canonical_stack_streaming(prov, req, tile_size=tile_size)


def _assert_bit_exact(cpu, gpu):
    np.testing.assert_array_equal(cpu.science, gpu.science)
    np.testing.assert_array_equal(cpu.estimator_weight_sum, gpu.estimator_weight_sum)
    np.testing.assert_array_equal(cpu.valid_mask, gpu.valid_mask)
    np.testing.assert_array_equal(cpu.surviving_sample_count, gpu.surviving_sample_count)
    np.testing.assert_array_equal(cpu.support_w1, gpu.support_w1)
    np.testing.assert_array_equal(cpu.support_w2, gpu.support_w2)
    np.testing.assert_array_equal(cpu.n_eff_support, gpu.n_eff_support)
    np.testing.assert_array_equal(cpu.rejection_mask, gpu.rejection_mask)
    assert cpu.iterations_used == gpu.iterations_used
    assert cpu.initial_sample_count == gpu.initial_sample_count
    assert cpu.rejected_sample_count == gpu.rejected_sample_count
    assert cpu.rejected_fraction == gpu.rejected_fraction
    assert cpu.low_n_cell_count == gpu.low_n_cell_count
    assert cpu.degenerate_cell_count == gpu.degenerate_cell_count


class TestStreamingGPU:
    @requires_gpu
    def test_streaming_gpu_bit_exact_winsor_mean(self):
        arrays, masks = _corpus()
        cpu = _run(
            arrays, masks, "cpu", 9,
            normalization="sky_mean", weighting="noise_variance",
            rejection="winsorized_sigma_clip", combine="mean", taper="footprint",
        )
        gpu = _run(
            arrays, masks, "gpu", 9,
            normalization="sky_mean", weighting="noise_variance",
            rejection="winsorized_sigma_clip", combine="mean", taper="footprint",
        )
        _assert_bit_exact(cpu, gpu)

    @requires_gpu
    def test_streaming_gpu_bit_exact_kappa_median(self):
        arrays, masks = _corpus()
        cpu = _run(
            arrays, masks, "cpu", 7,
            normalization="none", weighting="none",
            rejection="kappa_sigma", combine="median", taper="none",
        )
        gpu = _run(
            arrays, masks, "gpu", 7,
            normalization="none", weighting="none",
            rejection="kappa_sigma", combine="median", taper="none",
        )
        _assert_bit_exact(cpu, gpu)

    @requires_gpu
    def test_streaming_gpu_bit_exact_tile_invariance(self):
        # Different tile sizes on GPU are still bit-identical to CPU (bounded GPU
        # never materialises N x full-cell).
        arrays, masks = _corpus()
        cpu = _run(
            arrays, masks, "cpu", None,
            normalization="sky_mean", weighting="none",
            rejection="winsorized_sigma_clip", combine="mean", taper="footprint",
        )
        for tile in (5, 11):
            gpu = _run(
                arrays, masks, "gpu", tile,
                normalization="sky_mean", weighting="none",
                rejection="winsorized_sigma_clip", combine="mean", taper="footprint",
            )
            _assert_bit_exact(cpu, gpu)

    def test_gpu_unavailable_raises_no_silent_fallback(self, monkeypatch):
        monkeypatch.setattr(
            "zemosaic.core.canonical_streaming.canonical_gpu_available", lambda: False
        )
        arrays, masks = _corpus()
        with pytest.raises(cs.CanonicalStackValidationError):
            _run(arrays, masks, "gpu", 7, normalization="none", weighting="none",
                 rejection="none", combine="mean", taper="none")

    def test_streaming_gpu_provenance_backend(self):
        if not _HAS_GPU:
            pytest.skip("no CUDA device available")
        arrays, masks = _corpus()
        gpu = _run(arrays, masks, "gpu", 7, normalization="none", weighting="none",
                   rejection="none", combine="mean", taper="none")
        assert gpu.provenance["backend"]["requested"] == "gpu"
        assert gpu.provenance["backend"]["rejection"] == "gpu"
        assert gpu.provenance["backend"]["combine"] == "gpu"
        assert gpu.provenance["backend"]["normalization"] == "cpu"
