"""SCI-05 Gate E3 — canonical coverage-aware render (deterministic witnesses).

Deterministic, hermetic tests for the donor-exact preview-only coverage render in
``zemosaic.core.canonical_render``. The render is a cosmetic transform: never
science, never mutating the scientific result/support. No randomness with global
seeds, no network, no filesystem writes, no ZSSS import.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import zemosaic.core.canonical_render as cr
from zemosaic.core.canonical_render import (
    coverage_aware_render,
    coverage_render_event,
    render_preview,
)

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None


def _rand(shape, seed=0, scale=1.0, offset=0.0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=shape) * scale + offset).astype(np.float32)


# ---------------------------------------------------------------------------
# Render math
# ---------------------------------------------------------------------------

class TestRenderMath:
    def test_alpha_zero_high_support_untouched(self):
        # sup >= n_ref -> alpha = 0 -> detail not attenuated (out == sci up to 1 ULP)
        sci = _rand((40, 40), seed=1, scale=10.0, offset=50.0)
        sup = np.full((40, 40), 100.0, dtype=np.float32)  # >= n_ref
        out = coverage_aware_render(sci, sup)
        # The donor formula B + (sci - B) is exact up to float32 rounding (~1 ULP).
        np.testing.assert_allclose(out, sci, rtol=0.0, atol=1e-6)
        assert out.dtype == np.float32

    def test_alpha_intermediate_blend(self):
        # sup = n_ref/2 -> alpha = 0.5 -> out == B + (1-alpha)*D + alpha*Dd
        # (independently reconstructed; verifies the alpha formula).
        sci = _rand((40, 40), seed=2, scale=10.0, offset=50.0)
        sup = np.full((40, 40), 16.0, dtype=np.float32)  # n_ref=32 -> alpha=0.5
        out = coverage_aware_render(sci, sup)
        alpha = np.clip(1.0 - sup / 32.0, 0.0, 1.0).astype(np.float32)
        B = gaussian_filter(sci, sigma=32.0)
        D = sci - B
        Dd = gaussian_filter(D, sigma=2.0)
        expected = (B + (1.0 - alpha) * D + alpha * Dd).astype(np.float32)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-4)

    def test_flat_field_stays_exactly_flat(self):
        # constant sci -> constant out (no brightness gain), bit-exact.
        sci = np.full((40, 40), 5.0, dtype=np.float32)
        sup = _rand((40, 40), seed=3).astype(np.float32)
        out = coverage_aware_render(sci, sup)
        np.testing.assert_array_equal(out, sci)

    def test_no_new_extrema_beyond_input(self):
        # no invented signal: output stays within the input's value range.
        sci = _rand((40, 40), seed=4, scale=10.0, offset=50.0)
        sup = _rand((40, 40), seed=5, scale=1.0).astype(np.float32) * 40.0
        out = coverage_aware_render(sci, sup)
        assert float(out.min()) >= float(sci.min()) - 1e-3
        assert float(out.max()) <= float(sci.max()) + 1e-3

    def test_low_support_detail_attenuated(self):
        # a sharp impulse in a low-support region is smoothed (peak reduced).
        # sup must be positive (else the "no support" early-return fires unchanged).
        sci = np.zeros((64, 64), dtype=np.float32)
        sci[32, 32] = 1.0
        sup = np.full((64, 64), 1.0, dtype=np.float32)  # low support -> alpha ~ 0.97
        out = coverage_aware_render(sci, sup)
        assert float(out[32, 32]) < 1.0  # peak attenuated


# ---------------------------------------------------------------------------
# Validation / no-support / scipy fallback
# ---------------------------------------------------------------------------

class TestValidation:
    def test_n_ref_validation(self):
        sci = np.ones((8, 8), dtype=np.float32)
        sup = np.ones((8, 8), dtype=np.float32)
        for bad in (0.0, -1.0, np.nan, np.inf):
            with pytest.raises(ValueError):
                coverage_aware_render(sci, sup, n_ref=bad)

    def test_sup_3d_rejected(self):
        sci = np.ones((8, 8, 3), dtype=np.float32)
        sup = np.ones((8, 8, 1), dtype=np.float32)  # 3-D
        with pytest.raises(ValueError):
            coverage_aware_render(sci, sup)

    def test_shape_mismatch_rejected(self):
        sci = np.ones((8, 8), dtype=np.float32)
        sup = np.ones((9, 8), dtype=np.float32)
        with pytest.raises(ValueError):
            coverage_aware_render(sci, sup)

    def test_no_support_returns_unchanged(self):
        sci = _rand((16, 16), seed=6)
        for sup in (np.zeros((16, 16), dtype=np.float32),
                    np.full((16, 16), np.nan, dtype=np.float32),
                    np.full((16, 16), -1.0, dtype=np.float32)):
            out = coverage_aware_render(sci, sup)
            np.testing.assert_array_equal(out, sci)

    def test_scipy_unavailable_fallback(self, monkeypatch):
        sci = _rand((16, 16), seed=7)
        sup = np.full((16, 16), 1.0, dtype=np.float32)
        monkeypatch.setattr(cr, "_gaussian_filter", None)
        out = coverage_aware_render(sci, sup)
        monkeypatch.undo()
        np.testing.assert_array_equal(out, sci)  # unchanged


# ---------------------------------------------------------------------------
# Purity / A-B (never mutates science/support)
# ---------------------------------------------------------------------------

class TestPureAB:
    def test_inputs_not_mutated(self):
        sci = _rand((16, 16), seed=8)
        sup = _rand((16, 16), seed=9).astype(np.float32) * 10.0
        sci_before = sci.copy()
        sup_before = sup.copy()
        coverage_aware_render(sci, sup)
        np.testing.assert_array_equal(sci, sci_before)
        np.testing.assert_array_equal(sup, sup_before)

    def test_engine_result_unchanged(self):
        from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack
        arrays = [np.full((8, 8), float(v), dtype=np.float32) for v in (1.0, 2.0, 3.0)]
        req = CanonicalStackRequest(
            images=arrays, geometric_support=[np.ones((8, 8), dtype=bool)] * 3,
            normalization="none", weighting="none", rejection="none", combine="mean",
            taper="none",
        )
        result = run_canonical_stack(req)
        sci_before = result.science.copy()
        w1_before = result.support_w1.copy()
        w2_before = result.support_w2.copy()
        neff_before = result.n_eff_support.copy()
        valid_before = result.valid_mask.copy()
        rendered = render_preview(result)
        # the render returns a NEW array and leaves every result array bit-identical
        assert rendered is not result.science
        np.testing.assert_array_equal(result.science, sci_before)
        np.testing.assert_array_equal(result.support_w1, w1_before)
        np.testing.assert_array_equal(result.support_w2, w2_before)
        np.testing.assert_array_equal(result.n_eff_support, neff_before)
        np.testing.assert_array_equal(result.valid_mask, valid_before)

    def test_render_preview_tuple_form(self):
        sci = _rand((16, 16), seed=10)
        neff = np.full((16, 16), 50.0, dtype=np.float64)  # high support
        out = render_preview((sci, neff))
        assert out.shape == sci.shape
        assert out.dtype == np.float32

    def test_determinism(self):
        sci = _rand((16, 16), seed=11)
        sup = _rand((16, 16), seed=12).astype(np.float32) * 20.0
        a = coverage_aware_render(sci, sup)
        b = coverage_aware_render(sci, sup)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Shapes + event helper
# ---------------------------------------------------------------------------

class TestShapesAndEvent:
    def test_mono_hwc1_rgb_shapes(self):
        sup = np.full((10, 10), 1.0, dtype=np.float32)
        mono = coverage_aware_render(_rand((10, 10), seed=13), sup)
        assert mono.shape == (10, 10) and mono.dtype == np.float32
        hwc1 = coverage_aware_render(_rand((10, 10, 1), seed=14), sup)
        assert hwc1.shape == (10, 10, 1) and hwc1.dtype == np.float32
        rgb = coverage_aware_render(_rand((10, 10, 3), seed=15), sup)
        assert rgb.shape == (10, 10, 3) and rgb.dtype == np.float32

    def test_event_json_serializable(self):
        ev = coverage_render_event(n_ref=32.0, sigma_denoise=2.0, sigma_low=32.0)
        assert ev["render"] == "coverage_aware_render"
        assert ev["n_ref"] == 32.0
        json.dumps(ev)  # must not raise (scalars/strings only)
