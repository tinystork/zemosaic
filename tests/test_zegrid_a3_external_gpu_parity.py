"""A3 — external working-set reduction CPU↔GPU parity (physical GPU, opt-in).

Verifies the disk-spooled external reduction
``zemosaic.core.zegrid.external_stream.run_cell_streaming_external(backend="gpu")``
is bit-identical to the CPU path (masks/counts bit-exact; float64 science within
``rtol=1e-12``, matching the canonical parity contract). Skips cleanly without
CuPy/GPU.

The reduction reuses the bit-exact host-side reductions from
``canonical_stacking`` (``_axis_sum_f64``, ``_nan_axis_median``), so — like the
streaming RAM path — the results are bit-identical, not merely within tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_stacking import CanonicalStackValidationError, canonical_gpu_available
from zemosaic.core.canonical_streaming import InMemoryCanonicalProvider
from zemosaic.core.zegrid import external_stream as x

SCIENCE_RTOL = 1e-12
_HAS_GPU = canonical_gpu_available()
requires_gpu = pytest.mark.skipif(not _HAS_GPU, reason="no CUDA device available")

_PLANES = (
    "science",
    "estimator_weight_sum",
    "valid_mask",
    "surviving_sample_count",
    "support_w1",
    "support_w2",
    "n_eff_support",
)


def _corpus(n=9, h=24, w=24, seed=5):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        f = (
            rng.normal(0, 3, (h, w, 3)).astype(np.float32)
            + np.array([50, 55, 60], np.float32)
            + 2.0 * i
        )
        f[rng.random((h, w, 3)) < 0.1] = np.nan
        frames.append(f)
    frames[2][1, 1, :] = 1000.0  # outlier
    masks = [np.ones((h, w), dtype=bool)] * n
    masks[1][4:8, 4:8] = False
    masks[3][:, 15:] = False
    return frames, masks


def _run(arrays, masks, backend, tmp_path, **kw):
    req = CanonicalStackRequest(
        images=arrays, geometric_support=masks, backend=backend, **kw
    )
    return x.run_cell_streaming_external(
        InMemoryCanonicalProvider(arrays, masks),
        req,
        tile_size=16,
        column_batch=7,
        spool_root=str(tmp_path),
        cell_id="c",
        attempt_id="a",
    )


def _assert_parity(cpu, gpu):
    for name in _PLANES:
        a, b = getattr(cpu, name), getattr(gpu, name)
        if name in ("valid_mask", "surviving_sample_count"):
            np.testing.assert_array_equal(a, b, err_msg=name)
        elif a.dtype.kind in "iu":
            np.testing.assert_array_equal(a, b, err_msg=name)
        else:
            np.testing.assert_allclose(
                a, b, rtol=SCIENCE_RTOL, atol=0.0, equal_nan=True, err_msg=name
            )
    assert cpu.iterations_used == gpu.iterations_used
    assert cpu.initial_sample_count == gpu.initial_sample_count
    assert cpu.rejected_sample_count == gpu.rejected_sample_count
    assert cpu.rejected_fraction == gpu.rejected_fraction
    assert cpu.low_n_cell_count == gpu.low_n_cell_count
    assert cpu.degenerate_cell_count == gpu.degenerate_cell_count


class TestExternalGPU:
    @requires_gpu
    def test_external_gpu_bit_exact_winsor_mean(self, tmp_path):
        arrays, masks = _corpus()
        cpu = _run(arrays, masks, "cpu", tmp_path, normalization="sky_mean",
                   weighting="noise_variance", rejection="winsorized_sigma_clip",
                   combine="mean", taper="footprint")
        gpu = _run(arrays, masks, "gpu", tmp_path, normalization="sky_mean",
                   weighting="noise_variance", rejection="winsorized_sigma_clip",
                   combine="mean", taper="footprint")
        _assert_parity(cpu, gpu)
        assert gpu.provenance["backend"]["rejection"] == "gpu"
        assert gpu.provenance["backend"]["combine"] == "gpu"

    @requires_gpu
    def test_external_gpu_bit_exact_kappa_median(self, tmp_path):
        arrays, masks = _corpus()
        cpu = _run(arrays, masks, "cpu", tmp_path, normalization="sky_mean",
                   weighting="noise_variance", rejection="kappa_sigma",
                   combine="median", taper="none")
        gpu = _run(arrays, masks, "gpu", tmp_path, normalization="sky_mean",
                   weighting="noise_variance", rejection="kappa_sigma",
                   combine="median", taper="none")
        _assert_parity(cpu, gpu)
        assert gpu.provenance["backend"]["combine"] == "gpu"

    @requires_gpu
    def test_external_gpu_no_rejection_mean(self, tmp_path):
        arrays, masks = _corpus()
        cpu = _run(arrays, masks, "cpu", tmp_path, normalization="none",
                   weighting="none", rejection="none", combine="mean", taper="none")
        gpu = _run(arrays, masks, "gpu", tmp_path, normalization="none",
                   weighting="none", rejection="none", combine="mean", taper="none")
        _assert_parity(cpu, gpu)

    def test_gpu_unavailable_raises_no_silent_fallback(self, tmp_path, monkeypatch):
        # ``_resolve_backend`` (imported into external_stream) checks availability
        # in canonical_stacking; patch the source of truth, not the re-export.
        monkeypatch.setattr(
            "zemosaic.core.canonical_stacking.canonical_gpu_available", lambda: False
        )
        arrays, masks = _corpus()
        with pytest.raises(CanonicalStackValidationError):
            _run(arrays, masks, "gpu", tmp_path, normalization="none",
                 weighting="none", rejection="none", combine="mean", taper="none")
