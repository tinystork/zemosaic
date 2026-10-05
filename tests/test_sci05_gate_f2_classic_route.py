"""SCI-05 Gate F2 — Classic CPU route caller convergence (footprint threading + canonical engine).

Deterministic, hermetic tests for the F2 convergence: the alignment footprints are surfaced
and threaded, and the Classic CPU stacking route executes the accepted canonical engine for its
supported method set (no silent fallback, Coverage taper consumed, legacy radial inert).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from zemosaic import zemosaic_align_stack as zas
from zemosaic.core import canonical_stacking
from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack

_H, _W, _C = 48, 48, 3


def _frames(n=3, shape=(_H, _W, _C)):
    rng = np.random.default_rng(0)
    return [rng.normal(100.0, 10.0, shape).astype(np.float32) for _ in range(n)]


def _all_true_footprints(n, shape=(_H, _W)):
    return [np.ones(shape, dtype=bool) for _ in range(n)]


def _shifted_footprints(n, shape=(_H, _W)):
    """Reference all-True; the rest are axis-aligned overlap rectangles (a real shift)."""
    fps = [np.ones(shape, dtype=bool)]
    for k in range(1, n):
        dy, dx = (k * 2) % 8, (k * 3) % 8
        fp = np.zeros(shape, dtype=bool)
        y0 = max(0, dy)
        x0 = max(0, dx)
        fp[y0:, x0:] = True
        fps.append(fp)
    return fps


def _zconfig(taper=True):
    return SimpleNamespace(coverage_support_taper=taper)


# ---------------------------------------------------------------------------
# Footprint surfacing (align_images_in_group return_footprints)
# ---------------------------------------------------------------------------

class TestAlignFootprintSurfacing:
    def test_reference_footprint_all_true_and_parallel(self, monkeypatch):
        monkeypatch.setattr(zas, "ASTROALIGN_AVAILABLE", False)
        monkeypatch.setattr(zas, "astroalign_module", None)
        frames = _frames(2)
        aligned, failed, footprints = zas.align_images_in_group(
            frames, reference_image_index=0, propagate_mask=True, return_footprints=True
        )
        assert len(aligned) == 2 and len(failed) == 0 and len(footprints) == 2
        assert np.all(footprints[0])  # reference identity footprint all-True
        for fp in footprints:
            assert fp is not None and fp.dtype == bool and fp.shape == (_H, _W)

    def test_backward_compatible_two_tuple(self, monkeypatch):
        monkeypatch.setattr(zas, "ASTROALIGN_AVAILABLE", False)
        monkeypatch.setattr(zas, "astroalign_module", None)
        result = zas.align_images_in_group(_frames(2), reference_image_index=0, propagate_mask=True)
        assert isinstance(result, tuple) and len(result) == 2  # legacy 2-tuple


# ---------------------------------------------------------------------------
# Route convergence (stack_aligned_images -> run_canonical_stack)
# ---------------------------------------------------------------------------

class TestClassicRouteConvergence:
    def test_matches_canonical_engine(self):
        frames = _frames(3)
        fps = _shifted_footprints(3)
        sci = zas.stack_aligned_images(
            frames,
            normalize_method="none",
            weighting_method="none",
            rejection_algorithm="none",
            final_combine_method="mean",
            geometric_support=fps,
            zconfig=_zconfig(True),
        )
        res = run_canonical_stack(
            CanonicalStackRequest(
                images=frames,
                geometric_support=fps,
                normalization="none",
                weighting="none",
                rejection="none",
                combine="mean",
                reference_index=0,
                taper="footprint",
                backend="cpu",
            )
        )
        assert sci is not None and sci.shape == (_H, _W, _C)
        np.testing.assert_allclose(sci, res.science, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_median_combine_matches(self):
        frames = _frames(3)
        fps = _all_true_footprints(3)
        sci = zas.stack_aligned_images(
            frames,
            normalize_method="none",
            weighting_method="none",
            rejection_algorithm="none",
            final_combine_method="median",
            geometric_support=fps,
            zconfig=_zconfig(True),
        )
        res = run_canonical_stack(
            CanonicalStackRequest(
                images=frames,
                geometric_support=fps,
                normalization="none",
                weighting="none",
                rejection="none",
                combine="median",
                reference_index=0,
                taper="footprint",
                backend="cpu",
            )
        )
        np.testing.assert_allclose(sci, res.science, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_linear_fit_clip_fails(self):
        with pytest.raises(ValueError) as ei:
            zas.stack_aligned_images(
                _frames(3),
                rejection_algorithm="linear_fit_clip",
                geometric_support=_all_true_footprints(3),
                zconfig=_zconfig(True),
            )
        assert "unsupported_removed_sci05" in str(ei.value)

    def test_taper_on_off_weight_difference(self):
        frames = _frames(3)
        fps = _shifted_footprints(3)
        res_on = run_canonical_stack(
            CanonicalStackRequest(
                images=frames, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="mean",
                reference_index=0, taper="footprint", backend="cpu",
            )
        )
        res_off = run_canonical_stack(
            CanonicalStackRequest(
                images=frames, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="mean",
                reference_index=0, taper="none", backend="cpu",
            )
        )
        # The footprint taper softens the support edges, so the positive-support maps differ.
        assert not np.allclose(res_on.support_w1, res_off.support_w1, equal_nan=True)

    def test_all_invalid_is_nan(self):
        frames = _frames(2)
        # A single pixel is invalid (NaN) in every frame; the rest is valid.
        for f in frames:
            f[10, 10, :] = np.nan
        fps = _all_true_footprints(2)
        res = run_canonical_stack(
            CanonicalStackRequest(
                images=frames, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="mean",
                reference_index=0, taper="none", backend="cpu",
            )
        )
        # The all-invalid pixel is NaN (never a zero sentinel); others stay finite.
        assert np.all(np.isnan(res.science[10, 10, :]))
        assert np.all(np.isfinite(res.science[0, 0, :]))

    def test_noise_fwhm_without_photutils_fails(self, monkeypatch):
        monkeypatch.setattr(canonical_stacking, "canonical_noise_fwhm_available", lambda: False)
        with pytest.raises(canonical_stacking.CanonicalStackValidationError):
            run_canonical_stack(
                CanonicalStackRequest(
                    images=_frames(2), geometric_support=_all_true_footprints(2),
                    normalization="none", weighting="noise_fwhm", rejection="none",
                    combine="mean", reference_index=0, taper="none", backend="cpu",
                )
            )


# ---------------------------------------------------------------------------
# Legacy inert (E5b) + render preview-only (E3)
# ---------------------------------------------------------------------------

class TestLegacyInert:
    def test_legacy_radial_stays_inert(self):
        from zemosaic import zemosaic_align_stack_gpu as zasgpu
        params = {"apply_radial_weight": True}
        assert zasgpu._compute_radial_weight_map(16, 16, 3, params, None) is None

    def test_render_never_mutates_science(self):
        from zemosaic.core import canonical_render
        frames = _frames(2)
        fps = _all_true_footprints(2)
        res = run_canonical_stack(
            CanonicalStackRequest(
                images=frames, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="mean",
                reference_index=0, taper="none", backend="cpu",
            )
        )
        before = res.science.copy()
        # preview-only render must not mutate the science/support
        canonical_render.coverage_aware_render(res.science, res.n_eff_support)
        np.testing.assert_array_equal(res.science, before)


# ---------------------------------------------------------------------------
# R1 — reference selection (ref-present explicit 0 vs ref-missing auto)
# ---------------------------------------------------------------------------

class TestReferenceSelectionR1:
    def test_ref_present_uses_index_0_bit_equal(self):
        frames = _frames(3)
        fps = _shifted_footprints(3)
        sci = zas.stack_aligned_images(
            frames,
            normalize_method="none", weighting_method="none",
            rejection_algorithm="none", final_combine_method="mean",
            geometric_support=fps, zconfig=_zconfig(True), reference_index=0,
        )
        res = run_canonical_stack(
            CanonicalStackRequest(
                images=frames, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="mean",
                reference_index=0, taper="footprint", backend="cpu",
            )
        )
        np.testing.assert_allclose(sci, res.science, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_ref_missing_auto_matches(self):
        frames = _frames(3)
        fps = _shifted_footprints(3)
        sci = zas.stack_aligned_images(
            frames,
            normalize_method="none", weighting_method="none",
            rejection_algorithm="none", final_combine_method="mean",
            geometric_support=fps, zconfig=_zconfig(True), reference_index=None,
        )
        res = run_canonical_stack(
            CanonicalStackRequest(
                images=frames, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="mean",
                reference_index=None, taper="footprint", backend="cpu",
            )
        )
        np.testing.assert_allclose(sci, res.science, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_zero_support_first_frame_auto_does_not_raise(self):
        frames = _frames(2)
        fps = _all_true_footprints(2)
        fps[0] = np.zeros((_H, _W), dtype=bool)  # frame 0 has zero geometric support
        sci = zas.stack_aligned_images(
            frames,
            normalize_method="none", weighting_method="none",
            rejection_algorithm="none", final_combine_method="mean",
            geometric_support=fps, zconfig=_zconfig(True), reference_index=None,
        )
        assert sci is not None and sci.shape == (_H, _W, _C)

    def test_auto_selects_max_valid_count_frame(self):
        frames = _frames(2)
        fps = _all_true_footprints(2)
        fps[0] = np.zeros((_H, _W), dtype=bool)
        fps[0][:2, :2] = True  # frame 0 has only 4 valid pixels; frame 1 has full support
        res = run_canonical_stack(
            CanonicalStackRequest(
                images=frames, geometric_support=fps,
                normalization="none", weighting="none", rejection="none", combine="mean",
                reference_index=None, taper="none", backend="cpu",
            )
        )
        # auto-selection picks the max-valid-count frame (frame 1), not the arbitrary frame 0
        assert res.provenance["reference"]["mode"] == "auto"
        assert res.provenance["reference"]["index"] == 1
