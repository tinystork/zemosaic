"""ZM-SDS-R35C-C2 — SDS two-pass level origin, sign-insensitive merge, ALPHA support.

Scope: ONLY the SDS finalisation route.  The historical two-pass merge
(``_apply_two_pass_coverage_renorm_if_requested``) mixed a *signed*, near-zero
first pass with a *pedestal-shifted* second pass and decided per pixel with
``lum2 / (lum1 + eps) >= 0.4`` — a decision driven by the **sign** of a
near-zero luminance.

The SDS route now opts into ``sds_level_origin_fix`` which routes the merge
through ``_merge_two_pass_aligned_level_origin``:

* level origin of the second pass is aligned onto the first pass (per channel,
  robust median over the common valid support) *before* any decision;
* the merge decision is purely support-based and therefore sign-insensitive;
* the coverage map keeps a single coherent unit (the first-pass coverage);
* ``_finalize_sds_global_mosaic`` re-intersects the final ALPHA with the final
  support so that ``coverage <= 0 => ALPHA == 0``.

Invariants guarded below:

* no sign-dependent jump after the corrected merge (positive and negative
  regions both collapse back onto the first pass);
* the coverage map returned is bit-identical to the first-pass coverage;
* the opt-in defaults to ``False`` on both shared functions (Classic untouched);
* ``_finalize_sds_global_mosaic`` enables the opt-in and applies the ALPHA
  intersection;
* ``ALPHA == 0`` is never overwritten to 255 by the intersection.
"""

from __future__ import annotations

import inspect

import numpy as np

from zemosaic import zemosaic_worker as zw


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _signed_first_pass(h: int = 32, w: int = 48) -> np.ndarray:
    """Signed HWC first pass crossing zero in every channel."""

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = xx - float(w) / 2.0
    ch0 = ramp
    ch1 = ramp * 0.5 - 3.0
    ch2 = -ramp + 7.0
    return np.stack([ch0, ch1, ch2], axis=-1).astype(np.float32)


_PEDESTAL = np.array([320.924, 468.506, 466.833], dtype=np.float32)


def _make_pair(h: int = 32, w: int = 48):
    first = _signed_first_pass(h, w)
    second = first + _PEDESTAL  # percentile pedestal, per channel
    cov1 = np.full((h, w), 41.0, dtype=np.float32)
    cov2 = np.full((h, w), 6.5, dtype=np.float32)
    return first, second, cov1, cov2


class _Cfg:
    sds_min_coverage_keep = 0.0


def _pcb(*args, **kwargs):
    return None


# ---------------------------------------------------------------------------
# corrected merge: no sign-dependent jump
# ---------------------------------------------------------------------------

class TestAlignedLevelOriginMerge:
    def test_helper_exists_and_callable(self):
        assert callable(zw._merge_two_pass_aligned_level_origin)
        assert callable(zw._intersect_alpha_with_support)

    def test_reported_offsets_match_per_channel_pedestal(self):
        first, second, cov1, cov2 = _make_pair()
        _, _, info = zw._merge_two_pass_aligned_level_origin(first, cov1, second, cov2)
        assert info["applied"] is True
        assert info["coverage_unit"] == "first_pass"
        offsets = np.asarray(info["level_offsets"], dtype=np.float64)
        assert np.allclose(offsets, _PEDESTAL, rtol=0, atol=1e-3)
        assert info["use_two_pass_fraction"] == 1.0

    def test_no_sign_dependent_jump(self):
        first, second, cov1, cov2 = _make_pair()
        merged, _, info = zw._merge_two_pass_aligned_level_origin(first, cov1, second, cov2)
        assert info["applied"] is True
        resid = merged - first
        # The aligned second pass collapses back onto the first pass everywhere.
        assert float(np.max(np.abs(resid))) < 1e-3

        # Both the positive and the negative branch of the signed signal show the
        # same (null) residual: no sign-dependent discontinuity.
        lum = first.mean(axis=-1)
        pos = lum > 1.0
        neg = lum < -1.0
        assert int(np.count_nonzero(pos)) > 0
        assert int(np.count_nonzero(neg)) > 0
        assert abs(float(np.median(resid[..., 0][pos]))) < 1e-3
        assert abs(float(np.median(resid[..., 0][neg]))) < 1e-3
        assert abs(float(np.median(resid[..., 0][pos]))) == abs(
            float(np.median(resid[..., 0][neg]))
        )

    def test_legacy_ratio_rule_was_sign_dependent(self):
        """Document the defect: the historical rule selects on the sign."""

        first, second, cov1, cov2 = _make_pair()
        lum1 = first.mean(axis=-1)
        lum2 = second.mean(axis=-1)
        eps = max(float(np.nanpercentile(lum1, 5)) * 0.1, 1e-6)
        ratio = lum2 / (lum1 + eps)
        old_use = ratio >= 0.4
        # The legacy rule treats a non-negative luminance (sign 0 included) as
        # "use the second pass"; it is exactly the sign of the first pass.
        sign_nonneg = lum1 >= 0.0
        assert float(np.mean(old_use == sign_nonneg)) == 1.0
        # And the selected set is not the sign-insensitive support set.
        assert not np.array_equal(old_use, np.ones_like(old_use, dtype=bool))

    def test_coverage_map_keeps_single_unit(self):
        first, second, cov1, cov2 = _make_pair()
        merged, cov_out, info = zw._merge_two_pass_aligned_level_origin(first, cov1, second, cov2)
        assert info["applied"] is True
        # Single coherent unit: the first-pass raw-count coverage, never the
        # megatile-count support of the second pass.
        assert np.array_equal(cov_out, cov1)
        assert not np.any(np.isclose(cov_out, cov2))

    def test_second_pass_on_off_only_changes_where_supported(self):
        first, second, cov1, cov2 = _make_pair()
        cov2_gap = cov2.copy()
        cov2_gap[:, :5] = 0.0  # no support for the second pass in the first 5 columns
        merged, cov_out, info = zw._merge_two_pass_aligned_level_origin(
            first, cov1, second, cov2_gap
        )
        assert info["applied"] is True
        # Where the second pass is unsupported, the first pass is kept as-is.
        assert np.allclose(merged[:, :5], first[:, :5])
        # And the coverage map is still the single first-pass unit.
        assert np.array_equal(cov_out, cov1)

    def test_mono_channel_supported(self):
        first = _signed_first_pass(16, 16)[..., 0]
        second = first + np.float32(120.0)
        cov1 = np.full((16, 16), 10.0, dtype=np.float32)
        cov2 = np.full((16, 16), 3.0, dtype=np.float32)
        merged, cov_out, info = zw._merge_two_pass_aligned_level_origin(first, cov1, second, cov2)
        assert info["applied"] is True
        assert float(np.max(np.abs(merged - first))) < 1e-3
        assert np.array_equal(cov_out, cov1)

    def test_no_common_support_is_noop(self):
        first, second, cov1, cov2 = _make_pair()
        merged, cov_out, info = zw._merge_two_pass_aligned_level_origin(
            first, cov1, second, np.zeros_like(cov2)
        )
        assert info["applied"] is False
        assert np.array_equal(merged, first)
        assert np.array_equal(cov_out, cov1)


# ---------------------------------------------------------------------------
# ALPHA <-> support intersection
# ---------------------------------------------------------------------------

class TestAlphaSupportIntersection:
    def test_coverage_zero_forces_alpha_zero(self):
        alpha = np.full((4, 5), 255, dtype=np.uint8)
        cov = np.ones((4, 5), dtype=np.float32)
        cov[1, 2] = 0.0
        cov[3, :] = 0.0
        out = zw._intersect_alpha_with_support(alpha, cov)
        assert out is not None
        assert out[1, 2] == 0
        assert np.all(out[3, :] == 0)
        assert np.all(out[cov > 0] == 255)
        # input is not mutated
        assert alpha[1, 2] == 255

    def test_alpha_zero_is_not_promoted(self):
        alpha = np.full((2, 2), 255, dtype=np.uint8)
        alpha[0, 0] = 0
        cov = np.ones((2, 2), dtype=np.float32)
        out = zw._intersect_alpha_with_support(alpha, cov)
        assert out is not None
        assert out[0, 0] == 0  # ALPHA == 0 survives (never overwritten to 255)
        assert np.all(out[1, :] == 255)

    def test_none_inputs_pass_through(self):
        assert zw._intersect_alpha_with_support(None, np.ones((2, 2), np.float32)) is None
        alpha = np.full((2, 2), 255, dtype=np.uint8)
        assert zw._intersect_alpha_with_support(alpha, None) is alpha

    def test_shape_mismatch_is_noop(self):
        alpha = np.full((2, 2), 255, dtype=np.uint8)
        cov = np.ones((3, 3), dtype=np.float32)
        out = zw._intersect_alpha_with_support(alpha, cov)
        assert out is alpha


# ---------------------------------------------------------------------------
# opt-in wiring: default off for Classic, on only for the SDS finalisation
# ---------------------------------------------------------------------------

class TestOptInWiring:
    def test_two_pass_default_is_off(self):
        sig = inspect.signature(zw._apply_two_pass_coverage_renorm_if_requested)
        assert sig.parameters["sds_level_origin_fix"].default is False

    def test_phase5_default_is_off(self):
        sig = inspect.signature(zw._apply_phase5_post_stack_pipeline)
        assert sig.parameters["sds_level_origin_fix"].default is False

    def test_phase5_forwards_the_flag(self):
        src = inspect.getsource(zw._apply_phase5_post_stack_pipeline)
        assert "sds_level_origin_fix=sds_level_origin_fix" in src

    def test_finalize_sds_enables_the_fix(self):
        src = inspect.getsource(zw._finalize_sds_global_mosaic)
        assert "sds_level_origin_fix=True" in src
        assert "_intersect_alpha_with_support(" in src

    def test_classic_call_site_does_not_enable_the_fix(self):
        # The shared Classic/SDS call site (worker:11763) must not pass the opt-in.
        src = inspect.getsource(zw._run_shared_phase45_phase5_pipeline)
        assert "sds_level_origin_fix" not in src


# ---------------------------------------------------------------------------
# end-to-end finalisation: coverage <= 0 => ALPHA == 0
# ---------------------------------------------------------------------------

class TestSdsFinalizeAlphaInvariant:
    def test_coverage_zero_implies_alpha_zero(self):
        mosaic = np.full((8, 9, 3), 10.0, dtype=np.float32)
        cov = np.full((8, 9), 41.0, dtype=np.float32)
        cov[0, :] = 0.0
        out_m, out_c, out_a = zw._finalize_sds_global_mosaic(
            mosaic, cov, zconfig=_Cfg(), pcb=_pcb
        )
        assert out_m is not None
        assert out_a is not None
        assert out_a.dtype == np.uint8
        assert np.all(out_a[0, :] == 0)  # empty support -> ALPHA == 0
        assert np.all(out_a[cov > 0] == 255)
        assert np.all(out_c[0, :] == 0.0)

    def test_nanized_where_support_empty(self):
        mosaic = np.full((6, 6, 3), 5.0, dtype=np.float32)
        cov = np.full((6, 6), 20.0, dtype=np.float32)
        cov[:, 0] = 0.0
        out_m, _, _ = zw._finalize_sds_global_mosaic(mosaic, cov, zconfig=_Cfg(), pcb=_pcb)
        assert out_m is not None
        assert np.all(np.isnan(out_m[:, 0, :]))
        assert np.all(np.isfinite(out_m[:, 1:, :]))
