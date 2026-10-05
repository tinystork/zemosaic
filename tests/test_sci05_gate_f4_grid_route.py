"""SCI-05 Gate F4 — Grid route caller convergence onto the canonical engine.

Deterministic tests proving that Grid per-tile stacking (via `_stack_weighted_patches` /
`_stack_grid_via_canonical`) executes the canonical engine when the honest geometric support
(footprints) is threaded, and that the method set/edge cases behave canonically.
"""

from __future__ import annotations

import numpy as np
import pytest

from zemosaic import grid_mode as gm
from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack
from zemosaic.core import canonical_stacking

_H, _W, _C = 48, 48, 3


def _patches(n=3, shape=(_H, _W, _C)):
    rng = np.random.default_rng(0)
    return [rng.normal(100.0, 10.0, shape).astype(np.float32) for _ in range(n)]


def _all_true_footprints(n, shape=(_H, _W)):
    return [np.ones(shape, dtype=bool) for _ in range(n)]


def _shifted_footprints(n, shape=(_H, _W)):
    fps = [np.ones(shape, dtype=bool)]
    for k in range(1, n):
        dy, dx = k * 2, k * 3
        f = np.zeros(shape, dtype=bool)
        f[dy:, dx:] = True
        fps.append(f)
    return fps


def _config(**overrides):
    kw = dict(
        stack_norm_method="none",
        stack_weight_method="none",
        stack_reject_algo="none",
        stack_final_combine="mean",
        coverage_support_taper=True,
    )
    kw.update(overrides)
    return gm.GridModeConfig(**kw)


def _dummy_weights(patches):
    return [np.ones(p.shape, dtype=np.float32) for p in patches]


class TestGridRouteConvergence:
    def test_matches_canonical_engine(self):
        patches = _patches()
        fps = _shifted_footprints(3)
        cfg = _config()
        sci, wsum = gm._stack_weighted_patches(
            patches, _dummy_weights(patches), cfg,
            geometric_support=fps, return_weight_sum=True,
        )
        res = run_canonical_stack(
            CanonicalStackRequest(
                images=patches, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="mean",
                reference_index=0, taper="footprint", backend="cpu",
            )
        )
        np.testing.assert_allclose(sci, res.science, rtol=1e-6, atol=1e-6, equal_nan=True)
        np.testing.assert_allclose(
            wsum, res.estimator_weight_sum.astype(np.float32),
            rtol=1e-6, atol=1e-6, equal_nan=True,
        )

    def test_median_combine_matches(self):
        patches = _patches()
        fps = _all_true_footprints(3)
        cfg = _config(stack_final_combine="median")
        sci = gm._stack_weighted_patches(
            patches, _dummy_weights(patches), cfg, geometric_support=fps
        )
        res = run_canonical_stack(
            CanonicalStackRequest(
                images=patches, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="median",
                reference_index=0, taper="footprint", backend="cpu",
            )
        )
        np.testing.assert_allclose(sci, res.science, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_linear_fit_clip_fails(self):
        patches = _patches()
        fps = _all_true_footprints(3)
        cfg = _config(stack_reject_algo="linear_fit_clip")
        with pytest.raises(ValueError) as ei:
            gm._stack_weighted_patches(
                patches, _dummy_weights(patches), cfg, geometric_support=fps
            )
        assert "unsupported_removed_sci05" in str(ei.value)

    def test_noise_fwhm_without_photutils_fails(self, monkeypatch):
        monkeypatch.setattr(canonical_stacking, "canonical_noise_fwhm_available", lambda: False)
        patches = _patches()
        fps = _all_true_footprints(3)
        cfg = _config(stack_weight_method="noise_fwhm")
        with pytest.raises(canonical_stacking.CanonicalStackValidationError):
            gm._stack_weighted_patches(
                patches, _dummy_weights(patches), cfg, geometric_support=fps
            )

    def test_all_invalid_is_nan(self):
        patches = _patches(2)
        for p in patches:
            p[10, 10, :] = np.nan
        fps = _all_true_footprints(2)
        cfg = _config()
        sci = gm._stack_weighted_patches(
            patches, _dummy_weights(patches), cfg, geometric_support=fps
        )
        assert np.all(np.isnan(sci[10, 10, :]))
        assert np.all(np.isfinite(sci[0, 0, :]))

    def test_taper_on_off_weight_difference(self):
        patches = _patches()
        fps = _shifted_footprints(3)
        res_on = run_canonical_stack(
            CanonicalStackRequest(
                images=patches, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="mean",
                reference_index=0, taper="footprint", backend="cpu",
            )
        )
        res_off = run_canonical_stack(
            CanonicalStackRequest(
                images=patches, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="mean",
                reference_index=0, taper="none", backend="cpu",
            )
        )
        assert not np.allclose(res_on.support_w1, res_off.support_w1, equal_nan=True)

    def test_gpu_path_routes_canonical_on_cpu(self, monkeypatch):
        # Even the GPU entry point routes the canonical stage on CPU when footprints are threaded.
        monkeypatch.setattr(gm, "_CUPY_AVAILABLE", True)
        patches = _patches()
        fps = _all_true_footprints(3)
        cfg = _config(use_gpu=True)
        sci = gm._stack_weighted_patches_gpu(
            patches, _dummy_weights(patches), cfg, geometric_support=fps
        )
        res = run_canonical_stack(
            CanonicalStackRequest(
                images=patches, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="mean",
                reference_index=0, taper="footprint", backend="cpu",
            )
        )
        np.testing.assert_allclose(sci, res.science, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_legacy_radial_inert(self):
        # Grid never applies a radial map.
        from zemosaic import zemosaic_align_stack_gpu as zasgpu
        assert zasgpu._compute_radial_weight_map(16, 16, 3, {"apply_radial_weight": True}, None) is None

    def test_render_never_mutates_science(self):
        from zemosaic.core import canonical_render
        patches = _patches(2)
        fps = _all_true_footprints(2)
        res = run_canonical_stack(
            CanonicalStackRequest(
                images=patches, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="mean",
                reference_index=0, taper="none", backend="cpu",
            )
        )
        before = res.science.copy()
        canonical_render.coverage_aware_render(res.science, res.n_eff_support)
        np.testing.assert_array_equal(res.science, before)


# ---------------------------------------------------------------------------
# R1 — reference selection (supported-first explicit 0 vs zero-support-first auto)
# ---------------------------------------------------------------------------

class TestGridReferenceSelectionR1:
    def test_ref_present_uses_index_0_bit_equal(self):
        patches = _patches()
        fps = _shifted_footprints(3)
        cfg = _config()
        sci = gm._stack_weighted_patches(
            patches, _dummy_weights(patches), cfg, geometric_support=fps
        )
        res = run_canonical_stack(
            CanonicalStackRequest(
                images=patches, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="mean",
                reference_index=0, taper="footprint", backend="cpu",
            )
        )
        np.testing.assert_allclose(sci, res.science, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_zero_support_first_frame_auto_does_not_raise(self):
        patches = _patches(2)
        fps = _all_true_footprints(2)
        fps[0] = np.zeros((_H, _W), dtype=bool)  # frame 0 has zero geometric support
        cfg = _config()
        sci = gm._stack_weighted_patches(
            patches, _dummy_weights(patches), cfg, geometric_support=fps
        )
        assert sci is not None and sci.shape == (_H, _W, _C)
        res = run_canonical_stack(
            CanonicalStackRequest(
                images=patches, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="mean",
                reference_index=None, taper="footprint", backend="cpu",
            )
        )
        np.testing.assert_allclose(sci, res.science, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_auto_selects_supported_frame(self):
        patches = _patches(2)
        fps = _all_true_footprints(2)
        fps[0] = np.zeros((_H, _W), dtype=bool)
        fps[0][:2, :2] = True  # frame 0 has only 4 valid pixels; frame 1 has full support
        res = run_canonical_stack(
            CanonicalStackRequest(
                images=patches, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="mean",
                reference_index=None, taper="none", backend="cpu",
            )
        )
        assert res.provenance["reference"]["mode"] == "auto"
        assert res.provenance["reference"]["index"] == 1
