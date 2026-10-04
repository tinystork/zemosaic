"""SCI-05 Gate B1 — canonical input + normalization primitives (deterministic witnesses).

Deterministic, hermetic tests for the pure CPU canonical normalization layer in
``zemosaic.core.canonical_stacking``. This is **Gate B1** only: input validation,
reference selection, and per-frame normalization. No weighting (B2), rejection,
combine, coverage, GUI, config, GPU, or production-caller wiring.

Design notes
------------
* Tiny deterministic float32 corpora only; no random data, no sleeps, no network,
  no GPU, no media, no profile/XDG/HOME writes.
* Assertions target stable reason codes, requested==effective method, N-axis/index
  preservation, non-aliasing, and deterministic repeat-run equality.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.stats import sigma_clipped_stats

from zemosaic.core.canonical_stacking import (
    CanonicalInputBatch,
    CanonicalNormalizationResult,
    CanonicalStackFailure,
    CanonicalStackValidationError,
    FrameExclusion,
    _fit_linear_channel,
    _sky_mean_channel,
    normalize_canonical_images,
    prepare_canonical_inputs,
    select_canonical_reference,
)


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------

def _ramp(h, w, base=0.0, step=1.0):
    """Deterministic float32 2-D mono ramp of shape ``(h, w)``."""
    return (base + step * np.arange(h * w, dtype=np.float64)).reshape(h, w).astype(np.float32)


def _rgb_ramp(h, w, base=0.0, step=1.0):
    """Deterministic float32 3-D HWC ramp with identical channels."""
    v = _ramp(h, w, base, step)
    return np.stack([v, v, v], axis=-1)


def _full_support(h, w):
    return np.ones((h, w), dtype=bool)


def _mono_batch(arrays, masks=None):
    if masks is None:
        masks = [_full_support(a.shape[0], a.shape[1]) for a in arrays]
    return prepare_canonical_inputs(arrays, masks)


# ---------------------------------------------------------------------------
# 1. Validation: input / mask / layout errors
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_empty_images_rejected(self):
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs([], [])

    def test_none_images_rejected(self):
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs(None, None)

    def test_missing_geometric_support_rejected(self):
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs([np.ones((4, 4), dtype=np.float32)], None)

    def test_mask_count_mismatch_rejected(self):
        img = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs([img, img], [_full_support(4, 4)])

    def test_non_bool_mask_rejected(self):
        img = np.ones((4, 4), dtype=np.float32)
        numeric_mask = np.ones((4, 4), dtype=np.float32)  # numeric, not bool
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs([img], [numeric_mask])

    def test_int_mask_rejected_not_thresholded(self):
        img = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs([img], [np.ones((4, 4), dtype=np.int32)])

    def test_mask_wrong_shape_rejected(self):
        img = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs([img], [np.ones((3, 4), dtype=bool)])

    def test_mask_not_2d_rejected(self):
        img = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs([img], [np.ones((4, 4, 1), dtype=bool)])

    def test_channel_count_rejected(self):
        img = np.ones((4, 4, 2), dtype=np.float32)  # C=2 not allowed
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs([img], [_full_support(4, 4)])

    def test_empty_spatial_rejected(self):
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs(
                [np.ones((0, 4), dtype=np.float32)], [_full_support(0, 4)]
            )

    def test_mixed_layout_rejected(self):
        mono = np.ones((4, 4), dtype=np.float32)
        color = np.ones((4, 4, 3), dtype=np.float32)
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs([mono, color], [_full_support(4, 4)] * 2)

    def test_mixed_channel_count_rejected(self):
        c1 = np.ones((4, 4, 1), dtype=np.float32)
        c3 = np.ones((4, 4, 3), dtype=np.float32)
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs([c1, c3], [_full_support(4, 4)] * 2)

    def test_mixed_spatial_shape_rejected(self):
        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((5, 4), dtype=np.float32)
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs([a, b], [_full_support(4, 4), _full_support(5, 4)])

    def test_object_dtype_rejected(self):
        obj = np.empty((4, 4), dtype=object)
        obj.fill(1.0)
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs([obj], [_full_support(4, 4)])

    def test_complex_dtype_rejected(self):
        cplx = np.ones((4, 4), dtype=np.complex128)
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs([cplx], [_full_support(4, 4)])

    def test_ndim_rejected(self):
        with pytest.raises(CanonicalStackValidationError):
            prepare_canonical_inputs(
                [np.ones((4, 4, 4, 4), dtype=np.float32)], [_full_support(4, 4)]
            )


# ---------------------------------------------------------------------------
# 2. Validity semantics: zero science, RGB finite rule, NaN/Inf, overflow
# ---------------------------------------------------------------------------

class TestValiditySemantics:
    def test_zero_science_pixels_remain_valid(self):
        # Brightness is never used to derive validity: zeros inside support are valid.
        img = np.zeros((4, 4), dtype=np.float32)
        batch = _mono_batch([img])
        assert batch.frame_valid_counts[0] == 16
        assert np.all(batch.valid_mask[0])

    def test_rgb_all_channels_finite_rule(self):
        # Channel-invariant validity requires ALL channels finite.
        rgb = _rgb_ramp(4, 4).astype(np.float32)
        rgb[1, 1, 0] = np.nan  # only the R channel fails
        batch = prepare_canonical_inputs([rgb], [_full_support(4, 4)])
        assert not batch.valid_mask[0][1, 1]
        assert batch.frame_valid_counts[0] == 15

    def test_nan_and_inf_invalid_set_nan_outside_mask(self):
        img = _ramp(4, 4).astype(np.float32)
        img[0, 0] = np.nan
        img[1, 1] = np.inf
        batch = _mono_batch([img])
        assert not batch.valid_mask[0][0, 0]
        assert not batch.valid_mask[0][1, 1]
        assert np.isnan(batch.images[0, 0, 0, 0])
        assert np.isnan(batch.images[0, 1, 1, 0])

    def test_float_overflow_becomes_invalid(self):
        # float64 1e40 overflows float32 -> inf -> invalid (never valid by accident).
        img = np.full((4, 4), 1e40, dtype=np.float64)
        batch = prepare_canonical_inputs([img], [_full_support(4, 4)])
        assert batch.frame_valid_counts[0] == 0
        assert np.all(np.isnan(batch.images[0]))

    def test_masks_outside_support_nan(self):
        img = _ramp(4, 4).astype(np.float32)
        mask = np.ones((4, 4), dtype=bool)
        mask[2, 2] = False
        batch = prepare_canonical_inputs([img], [mask])
        assert np.isnan(batch.images[0, 2, 2, 0])
        assert not batch.valid_mask[0][2, 2]

    def test_no_input_mutation(self):
        img = _ramp(4, 4).astype(np.float32)
        mask = _full_support(4, 4)
        img_copy = img.copy()
        mask_copy = mask.copy()
        batch = _mono_batch([img], [mask])
        assert np.array_equal(img, img_copy)
        assert np.array_equal(mask, mask_copy)
        assert not np.shares_memory(batch.images, img)
        assert not np.shares_memory(batch.valid_mask, mask)


# ---------------------------------------------------------------------------
# 3. Reference selection
# ---------------------------------------------------------------------------

class TestReferenceSelection:
    def test_auto_largest_count(self):
        a = _ramp(8, 8).astype(np.float32)
        b = _ramp(8, 8).astype(np.float32)
        c = _ramp(8, 8).astype(np.float32)
        masks = [_full_support(8, 8), _full_support(8, 8), _full_support(8, 8)]
        masks[1][4:, :] = False  # 32 valid
        masks[2][2:, :] = False  # 16 valid
        batch = prepare_canonical_inputs([a, b, c], masks)
        assert batch.frame_valid_counts.tolist() == [64, 32, 16]
        assert select_canonical_reference(batch) == 0

    def test_auto_tie_lowest_index(self):
        a = _ramp(8, 8).astype(np.float32)
        b = _ramp(8, 8).astype(np.float32)
        masks = [_full_support(8, 8), _full_support(8, 8)]
        masks[0][4:, :] = False  # 32
        masks[1][:, 4:] = False  # 32
        batch = prepare_canonical_inputs([a, b], masks)
        assert batch.frame_valid_counts.tolist() == [32, 32]
        assert select_canonical_reference(batch) == 0

    def test_explicit_valid(self):
        a = _ramp(8, 8).astype(np.float32)
        b = _ramp(8, 8).astype(np.float32)
        masks = [_full_support(8, 8), _full_support(8, 8)]
        masks[0][4:, :] = False  # frame 0 has 32
        batch = prepare_canonical_inputs([a, b], masks)
        assert select_canonical_reference(batch, reference_index=1) == 1

    def test_explicit_bool_rejected(self):
        a = _ramp(8, 8).astype(np.float32)
        batch = _mono_batch([a])
        with pytest.raises(CanonicalStackValidationError):
            select_canonical_reference(batch, reference_index=True)
        with pytest.raises(CanonicalStackValidationError):
            select_canonical_reference(batch, reference_index=False)

    def test_explicit_out_of_range_rejected(self):
        a = _ramp(8, 8).astype(np.float32)
        batch = _mono_batch([a, a])
        with pytest.raises(CanonicalStackValidationError):
            select_canonical_reference(batch, reference_index=2)
        with pytest.raises(CanonicalStackValidationError):
            select_canonical_reference(batch, reference_index=-1)

    def test_explicit_zero_valid_rejected(self):
        a = _ramp(8, 8).astype(np.float32)
        b = np.full((8, 8), np.nan, dtype=np.float32)
        batch = prepare_canonical_inputs([a, b], [_full_support(8, 8)] * 2)
        assert batch.frame_valid_counts[1] == 0
        with pytest.raises(CanonicalStackValidationError):
            select_canonical_reference(batch, reference_index=1)

    def test_all_invalid_auto_failure(self):
        a = np.full((8, 8), np.nan, dtype=np.float32)
        b = np.full((8, 8), np.nan, dtype=np.float32)
        batch = prepare_canonical_inputs([a, b], [_full_support(8, 8)] * 2)
        with pytest.raises(CanonicalStackFailure):
            select_canonical_reference(batch)


# ---------------------------------------------------------------------------
# 4. none normalization
# ---------------------------------------------------------------------------

class TestNormalizationNone:
    def test_hw_passthrough_identity(self):
        img = _ramp(8, 8).astype(np.float32)
        img[0, 0] = np.nan
        batch = _mono_batch([img])
        res = normalize_canonical_images(batch, "none")
        assert res.requested_method == "none" and res.effective_method == "none"
        assert res.images.shape == (1, 8, 8, 1)
        assert res.images.dtype == np.float32
        # identity coefficients (1, 0) for the active frame
        assert res.coefficients[0, 0].tolist() == [1.0, 0.0]
        # valid samples unchanged; invalid stays NaN
        assert res.images[0, 1, 0, 0] == img[1, 0]
        assert np.isnan(res.images[0, 0, 0, 0])

    def test_rgb_passthrough(self):
        rgb = _rgb_ramp(8, 8).astype(np.float32)
        batch = prepare_canonical_inputs([rgb], [_full_support(8, 8)])
        res = normalize_canonical_images(batch, "none")
        assert res.images.shape == (1, 8, 8, 3)
        assert np.allclose(res.images[0], rgb, equal_nan=True)
        assert res.coefficients.shape == (1, 3, 2)

    def test_multi_frame_none_preserves_n(self):
        a = _ramp(8, 8).astype(np.float32)
        b = _ramp(8, 8).astype(np.float32)
        c = np.full((8, 8), np.nan, dtype=np.float32)  # zero-valid frame
        batch = prepare_canonical_inputs([a, b, c], [_full_support(8, 8)] * 3)
        res = normalize_canonical_images(batch, "none")
        assert res.images.shape[0] == 3  # N preserved
        assert res.active_frames.tolist() == [True, True, False]
        assert any(e.index == 2 and e.reason == "zero_valid_support" for e in res.exclusions)


# ---------------------------------------------------------------------------
# 5. N=1 identity for every method
# ---------------------------------------------------------------------------

class TestN1Identity:
    @pytest.mark.parametrize("method", ["none", "linear_fit", "sky_mean"])
    def test_n1_identity(self, method):
        img = _ramp(8, 8).astype(np.float32)
        batch = _mono_batch([img])
        res = normalize_canonical_images(batch, method)
        # documented identity success, effective method unchanged (no fallback)
        assert res.effective_method == method
        assert res.reference_index == 0
        assert np.all(res.active_frames)
        assert res.exclusions == ()
        assert res.coefficients[0, 0].tolist() == [1.0, 0.0]
        assert np.allclose(res.images[0], batch.images[0], equal_nan=True)


# ---------------------------------------------------------------------------
# 6. linear_fit
# ---------------------------------------------------------------------------

class TestLinearFit:
    def test_exact_known_affine(self):
        # y_ref = 2*x_src + 10 with >=256 common pixels; exact in float32.
        src = _ramp(20, 20).astype(np.float32)  # 0..399
        ref = (2.0 * src + 10.0).astype(np.float32)
        batch = _mono_batch([ref, src])
        res = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        a, b = res.coefficients[1, 0]
        assert a == pytest.approx(2.0, rel=1e-5)
        assert b == pytest.approx(10.0, rel=1e-5)
        # applied: normalized src ~ ref
        assert np.allclose(res.images[1, :, :, 0], ref, rtol=1e-5, atol=1e-3)
        assert res.requested_method == res.effective_method == "linear_fit"

    def test_per_channel_rgb_coefficients(self):
        base = _ramp(20, 20).astype(np.float32)
        src = np.stack(
            [base, 2.0 * base + 3.0, 0.5 * base - 7.0], axis=-1
        ).astype(np.float32)
        ref = np.stack(
            [2.0 * src[:, :, 0] + 1.0, 0.5 * src[:, :, 1] - 2.0, 3.0 * src[:, :, 2] + 4.0],
            axis=-1,
        ).astype(np.float32)
        batch = prepare_canonical_inputs([ref, src], [_full_support(20, 20)] * 2)
        res = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        # per-channel (a, b): (2,1), (0.5,-2), (3,4)
        assert res.coefficients[1, 0, 0] == pytest.approx(2.0, rel=1e-5)
        assert res.coefficients[1, 0, 1] == pytest.approx(1.0, rel=1e-5)
        assert res.coefficients[1, 1, 0] == pytest.approx(0.5, rel=1e-5)
        assert res.coefficients[1, 1, 1] == pytest.approx(-2.0, rel=1e-5)
        assert res.coefficients[1, 2, 0] == pytest.approx(3.0, rel=1e-5)
        assert res.coefficients[1, 2, 1] == pytest.approx(4.0, rel=1e-5)

    def test_apply_beyond_common_overlap(self):
        # ref valid only in a 375-pixel region; src valid everywhere (400).
        src = _ramp(20, 20).astype(np.float32)
        ref = (2.0 * src + 10.0).astype(np.float32)
        ref_mask = np.ones((20, 20), dtype=bool)
        ref_mask[15:20, 15:20] = False  # 25 pixels excluded -> 375 valid
        src_mask = np.ones((20, 20), dtype=bool)
        batch = prepare_canonical_inputs([ref, src], [ref_mask, src_mask])
        # common = 375 >= min_common(256)
        res = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        # The 25 src-valid-but-not-ref-valid pixels are still transformed (finite).
        assert np.all(np.isfinite(res.images[1, 15:20, 15:20, 0]))
        # And they match the fitted affine applied there.
        a, b = res.coefficients[1, 0]
        expected = a * src[15:20, 15:20] + b
        assert np.allclose(res.images[1, 15:20, 15:20, 0], expected, rtol=1e-5, atol=1e-3)

    def test_robust_outlier_discrimination(self):
        # 380 clean points on y=2x+10, plus 20 gross outliers that a naive OLS
        # would chase; the MAD refinement must recover the clean affine.
        src = _ramp(20, 20).astype(np.float32)  # 400 values 0..399
        ref = (2.0 * src + 10.0).astype(np.float32)
        # Deterministic explicit outlier indices (every 20th pixel).
        outlier_idx = np.arange(0, 400, 20)
        assert outlier_idx.size == 20
        ref_flat = ref.copy()
        ref_flat.flat[outlier_idx] += 5000.0  # gross outliers in reference
        ref = ref_flat.astype(np.float32)
        batch = _mono_batch([ref, src])
        res = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        assert res.active_frames[1]
        a, b = res.coefficients[1, 0]
        assert a == pytest.approx(2.0, rel=1e-4)
        assert b == pytest.approx(10.0, rel=1e-4)

    def test_monotonic_residual_mask_fixed_point(self):
        # Direct helper check: the accepted mask is a subset of the common mask
        # AND an actual fixed point — recomputing the MAD keep rule on the
        # accepted mask leaves it unchanged (one further monotonic intersection).
        src = _ramp(20, 20).astype(np.float32)
        ref = (2.0 * src + 10.0).astype(np.float32)
        outlier_idx = np.arange(20)  # first 20 pixels, deterministic
        ref_flat = ref.copy()
        ref_flat.flat[outlier_idx] += 3000.0
        ref = ref_flat.astype(np.float32)
        common = np.ones((20, 20), dtype=bool)
        a, b, reason, mask = _fit_linear_channel(
            ref.astype(np.float64), src.astype(np.float64), common, 256
        )
        assert reason is None
        assert np.all(mask <= common)  # monotonic: never larger than the seed
        assert int(mask.sum()) >= 256
        # Fixed-point assertion: recompute residual median/MAD keep rule from the
        # ACCEPTED (a, b, mask) and verify a further intersection changes nothing.
        resid = ref.astype(np.float64) - (a * src.astype(np.float64) + b)
        rm = resid[mask]
        center = float(np.median(rm))
        mad = float(np.median(np.abs(rm - center)))
        scale = 1.4826 * mad
        if scale > 0.0:
            keep = np.abs(rm - center) <= 3.0 * scale
        else:
            keep = rm == center
        assert np.all(keep)  # every accepted residual survives one more iteration

    def test_zero_mad_case(self):
        # Perfectly affine (exact) data -> residuals all zero -> scale==0 -> keep all.
        src = _ramp(20, 20).astype(np.float32)
        ref = (2.0 * src + 10.0).astype(np.float32)
        batch = _mono_batch([ref, src])
        res = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        assert res.active_frames[1]
        a, b = res.coefficients[1, 0]
        assert a == pytest.approx(2.0, rel=1e-5)
        assert b == pytest.approx(10.0, rel=1e-5)

    def test_insufficient_common_excludes(self):
        # ref full support (400), src only 100 valid pixels -> common < 256.
        ref = _ramp(20, 20).astype(np.float32)
        src = _ramp(20, 20).astype(np.float32)
        src_mask = np.zeros((20, 20), dtype=bool)
        src_mask[:10, :10] = True  # 100 valid
        batch = prepare_canonical_inputs([ref, src], [_full_support(20, 20), src_mask])
        res = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        assert not res.active_frames[1]
        assert any(e.index == 1 and e.reason == "insufficient_common" for e in res.exclusions)
        assert np.all(np.isnan(res.images[1]))
        assert np.all(np.isnan(res.coefficients[1]))

    def test_degenerate_denominator_excludes(self):
        # src constant -> zero variance -> degenerate OLS denominator.
        ref = _ramp(20, 20).astype(np.float32)
        src = np.full((20, 20), 5.0, dtype=np.float32)
        batch = _mono_batch([ref, src])
        res = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        assert not res.active_frames[1]
        assert any(e.index == 1 and e.reason == "degenerate_ols" for e in res.exclusions)
        assert np.all(np.isnan(res.images[1]))

    def test_slope_out_of_range_excludes(self):
        # slope 10 > 4.0 -> excluded, never clipped.
        src = _ramp(20, 20).astype(np.float32)
        ref = (10.0 * src + 1.0).astype(np.float32)
        batch = _mono_batch([ref, src])
        res = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        assert not res.active_frames[1]
        assert any(e.index == 1 and e.reason == "slope_out_of_range" for e in res.exclusions)
        assert np.all(np.isnan(res.images[1]))

    def test_nonfinite_fit_reason_defensive(self):
        # Defensive seam: non-finite reference values (bypassing input finiteness
        # checks) yield a non-finite fit, exercised directly on the helper.
        src = _ramp(20, 20).astype(np.float32)
        ref = src.astype(np.float64).copy()
        ref[:] = 0.0
        ref[0, 0] = np.inf
        common = np.ones((20, 20), dtype=bool)
        a, b, reason, _ = _fit_linear_channel(ref, src.astype(np.float64), common, 256)
        assert reason == "nonfinite_fit"
        assert a is None and b is None

    def test_whole_frame_excluded_when_one_rgb_channel_fails(self):
        # Two channels fit cleanly; the third is degenerate -> whole frame excluded.
        base = _ramp(20, 20).astype(np.float32)
        src = np.stack([base, base, np.full((20, 20), 7.0)], axis=-1).astype(np.float32)
        ref = np.stack(
            [2.0 * src[:, :, 0], 2.0 * src[:, :, 1], src[:, :, 2] + 1.0], axis=-1
        ).astype(np.float32)
        batch = prepare_canonical_inputs([ref, src], [_full_support(20, 20)] * 2)
        res = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        assert not res.active_frames[1]
        assert any(e.index == 1 and e.reason == "degenerate_ols" for e in res.exclusions)
        assert np.all(np.isnan(res.images[1]))
        assert np.all(np.isnan(res.coefficients[1]))

    def test_original_n_and_index_preservation(self):
        # N=3 with one zero-valid frame: indices and N axis preserved.
        a = _ramp(20, 20).astype(np.float32)
        b = _ramp(20, 20).astype(np.float32)
        c = np.full((20, 20), np.nan, dtype=np.float32)
        batch = prepare_canonical_inputs([a, b, c], [_full_support(20, 20)] * 3)
        res = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        assert res.images.shape[0] == 3
        assert res.active_frames.shape == (3,)
        assert res.active_frames.tolist() == [True, True, False]
        reasons = {e.index: e.reason for e in res.exclusions}
        assert reasons[2] == "zero_valid_support"

    def test_overflow_source_only_pixel_invalidated_no_clip(self):
        # EXACT defect reproduction: 399 common pixels fit to accepted (a,b)=(4,0),
        # one source-only pixel at 1e38. Applying the affine casts to Inf float32;
        # that pixel must be invalidated (all-NaN), the rest stays valid/finite,
        # the frame stays active, coefficients retained, and NO clipping/saturation.
        src = _ramp(20, 20).astype(np.float32)  # 0..399
        ref = (4.0 * src).astype(np.float32)  # accepted slope 4, intercept 0
        ref_mask = np.ones((20, 20), dtype=bool)
        ref_mask[0, 0] = False  # source-only pixel excluded from reference
        src[0, 0] = 1e38  # finite float32, but 4*1e38 overflows float32
        batch = prepare_canonical_inputs([ref, src], [ref_mask, _full_support(20, 20)])
        res = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        # frame active, coefficients retained
        assert res.active_frames[1]
        a, b = res.coefficients[1, 0]
        assert a == pytest.approx(4.0, rel=1e-5)
        assert b == pytest.approx(0.0, abs=1e-3)
        # the overflowing pixel is invalidated for all channels, all-NaN
        assert not res.valid_mask[1, 0, 0]
        assert np.all(np.isnan(res.images[1, 0, 0]))
        assert not np.any(np.isinf(res.images[1]))  # no Inf left, no clip to max
        # the rest stays valid, finite, and equals the applied affine (no clipping)
        assert int(res.valid_mask[1].sum()) == 399
        assert np.all(np.isfinite(res.images[1][res.valid_mask[1]]))
        assert np.allclose(res.images[1, 1:, :, 0], ref[1:, :], rtol=1e-5, atol=1e-3)
        assert np.allclose(res.images[1, 0, 1:, 0], ref[0, 1:], rtol=1e-5, atol=1e-3)

    def test_rgb_single_channel_overflow_invalidates_all_channels(self):
        # One RGB channel alone overflows at a source-only pixel; the pixel is
        # invalidated channel-invariantly (all channels NaN, mask false).
        base = _ramp(20, 20).astype(np.float32)
        src = np.stack([base, base, base], axis=-1).astype(np.float32)
        ref = (4.0 * src).astype(np.float32)
        ref_mask = np.ones((20, 20), dtype=bool)
        ref_mask[0, 0] = False
        src[0, 0, 1] = 1e38  # only the G channel overflows on cast
        batch = prepare_canonical_inputs([ref, src], [ref_mask, _full_support(20, 20)])
        res = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        assert res.active_frames[1]
        assert not res.valid_mask[1, 0, 0]  # channel-invariant invalidation
        assert np.all(np.isnan(res.images[1, 0, 0]))  # ALL channels NaN
        assert int(res.valid_mask[1].sum()) == 399

    def test_sky_mean_overflow_shared_application_invalidated(self):
        # Shared application path: the additive sky-mean offset overflows a
        # source-only pixel; it is invalidated (not clipped, not left Inf-valid).
        ref = np.full((20, 20), 2.0e38, dtype=np.float32)  # finite
        src = np.zeros((20, 20), dtype=np.float32)
        src[0, 0] = 2.0e38  # finite, but src+offset ~ 4e38 overflows float32
        ref_mask = np.ones((20, 20), dtype=bool)
        ref_mask[0, 0] = False
        batch = prepare_canonical_inputs([ref, src], [ref_mask, _full_support(20, 20)])
        res = normalize_canonical_images(batch, "sky_mean", reference_index=0)
        assert res.active_frames[1]
        a, b = res.coefficients[1, 0]
        assert a == 1.0
        assert b == pytest.approx(2.0e38, rel=1e-3)
        assert not res.valid_mask[1, 0, 0]
        assert np.all(np.isnan(res.images[1, 0, 0]))
        assert int(res.valid_mask[1].sum()) == 399


# ---------------------------------------------------------------------------
# 7. sky_mean
# ---------------------------------------------------------------------------

def _sigma_clipped_mean_ref(values):
    return float(sigma_clipped_stats(values, sigma_lower=3.0, sigma_upper=3.0, maxiters=5)[0])


class TestSkyMean:
    def test_true_mean_not_median(self):
        # Discriminating corpus: sigma-clipped MEAN != median, and the sky_mean
        # offset uses the MEAN (not the median / a percentile).
        src_vals = np.concatenate([np.full(300, 10.0), np.full(100, 30.0)])
        ref_vals = np.concatenate([np.full(300, 0.0), np.full(100, 60.0)])
        mean_src = _sigma_clipped_mean_ref(src_vals)
        mean_ref = _sigma_clipped_mean_ref(ref_vals)
        median_src = float(np.median(src_vals))
        median_ref = float(np.median(ref_vals))
        mean_offset = mean_ref - mean_src
        median_offset = median_ref - median_src
        # sanity: the corpus discriminates mean from median
        assert mean_offset != pytest.approx(median_offset, abs=1e-9)

        src = src_vals.reshape(20, 20).astype(np.float32)
        ref = ref_vals.reshape(20, 20).astype(np.float32)
        batch = _mono_batch([ref, src])
        res = normalize_canonical_images(batch, "sky_mean", reference_index=0)
        a, b = res.coefficients[1, 0]
        assert a == 1.0
        assert b == pytest.approx(mean_offset, abs=1e-6)
        # additive: normalized src = src + b ~ ref on valid pixels
        assert np.allclose(res.images[1, :, :, 0], src + b, atol=1e-5)

    def test_rgb_per_channel_offsets(self):
        base = _ramp(20, 20).astype(np.float32)
        src = np.stack([base + 10.0, base * 2.0, base - 5.0], axis=-1).astype(np.float32)
        ref = np.stack(
            [src[:, :, 0] + 3.0, src[:, :, 1] - 7.0, src[:, :, 2] + 11.0], axis=-1
        ).astype(np.float32)
        batch = prepare_canonical_inputs([ref, src], [_full_support(20, 20)] * 2)
        res = normalize_canonical_images(batch, "sky_mean", reference_index=0)
        # per-channel a=1, b = mean_ref - mean_src (sigma-clipped means).
        for ch in range(3):
            b = res.coefficients[1, ch, 1]
            expected = _sigma_clipped_mean_ref(ref[:, :, ch].ravel()) - _sigma_clipped_mean_ref(
                src[:, :, ch].ravel()
            )
            assert res.coefficients[1, ch, 0] == 1.0
            assert b == pytest.approx(expected, abs=1e-5)

    def test_common_mask_min_gate_excludes(self):
        ref = _ramp(20, 20).astype(np.float32)
        src = _ramp(20, 20).astype(np.float32)
        src_mask = np.zeros((20, 20), dtype=bool)
        src_mask[:10, :10] = True  # 100 valid < 256
        batch = prepare_canonical_inputs([ref, src], [_full_support(20, 20), src_mask])
        res = normalize_canonical_images(batch, "sky_mean", reference_index=0)
        assert not res.active_frames[1]
        assert any(e.index == 1 and e.reason == "insufficient_common" for e in res.exclusions)

    def test_sky_mean_failed_reason_defensive(self):
        # Defensive seam: non-finite data bypassing input checks yields a NaN mean.
        ref = np.full((20, 20), np.inf)
        src = np.full((20, 20), 1.0)
        common = np.ones((20, 20), dtype=bool)
        offset, reason = _sky_mean_channel(ref, src, common)
        assert reason == "sky_mean_failed"
        assert offset is None

    def test_stats_exception_becomes_structured_exclusion(self, monkeypatch):
        # Public-stage containment: if Astropy sigma-clipped stats raises for a
        # non-reference frame, the public result excludes that frame with reason
        # `sky_mean_failed` (all-NaN/all-false), reference survives, effective
        # method unchanged, and no raw exception leaks out.
        import zemosaic.core.canonical_stacking as cs

        def _raise(*args, **kwargs):
            raise RuntimeError("sigma_clipped_stats exploded")

        monkeypatch.setattr(cs, "sigma_clipped_stats", _raise)
        ref = _ramp(20, 20).astype(np.float32)
        src = _ramp(20, 20).astype(np.float32) + 100.0
        batch = _mono_batch([ref, src])
        res = normalize_canonical_images(batch, "sky_mean", reference_index=0)
        assert res.effective_method == "sky_mean"  # unchanged, no fallback
        assert res.active_frames[0]  # reference survives
        assert not res.active_frames[1]
        assert any(e.index == 1 and e.reason == "sky_mean_failed" for e in res.exclusions)
        assert np.all(np.isnan(res.images[1]))
        assert np.all(~res.valid_mask[1])
        assert np.all(np.isnan(res.coefficients[1]))


# ---------------------------------------------------------------------------
# 8. Common rules: aliasing, determinism, method normalization
# ---------------------------------------------------------------------------

class TestCommonRules:
    def test_result_does_not_alias_batch_or_input(self):
        src = _ramp(20, 20).astype(np.float32)
        ref = (2.0 * src + 10.0).astype(np.float32)
        batch = _mono_batch([ref, src])
        res = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        assert not np.shares_memory(res.images, batch.images)
        assert not np.shares_memory(res.valid_mask, batch.valid_mask)
        assert not np.shares_memory(res.coefficients, batch.images)
        # mutating the result must not touch the batch
        res.images[1, 0, 0, 0] = 12345.0
        assert batch.images[1, 0, 0, 0] != 12345.0

    def test_repeat_run_deterministic(self):
        src = _ramp(20, 20).astype(np.float32)
        ref = (2.0 * src + 10.0).astype(np.float32)
        ref[0, 0] = np.nan
        ref[1, 1] = 9999.0
        batch = _mono_batch([ref, src])
        r1 = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        r2 = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        np.testing.assert_array_equal(r1.images, r2.images)
        np.testing.assert_array_equal(r1.valid_mask, r2.valid_mask)
        np.testing.assert_array_equal(r1.active_frames, r2.active_frames)
        np.testing.assert_array_equal(r1.coefficients, r2.coefficients)
        assert r1.exclusions == r2.exclusions
        assert r1.reference_index == r2.reference_index

    def test_method_strip_lower_normalization(self):
        src = _ramp(8, 8).astype(np.float32)
        batch = _mono_batch([src])
        res = normalize_canonical_images(batch, "  None ")
        assert res.requested_method == "none" and res.effective_method == "none"

    def test_unknown_method_rejected(self):
        batch = _mono_batch([_ramp(8, 8).astype(np.float32)])
        for bad in ("median", "linear-fit", "LINEAR_FIT_CLIP", "sky", ""):
            with pytest.raises(CanonicalStackValidationError):
                normalize_canonical_images(batch, bad)

    def test_non_string_method_rejected(self):
        batch = _mono_batch([_ramp(8, 8).astype(np.float32)])
        with pytest.raises(CanonicalStackValidationError):
            normalize_canonical_images(batch, 123)

    def test_no_hidden_fallback(self):
        # A failing non-reference frame is excluded (all-NaN), never silently
        # converted to identity or another method.
        ref = _ramp(20, 20).astype(np.float32)
        src = np.full((20, 20), 5.0, dtype=np.float32)  # degenerate
        batch = _mono_batch([ref, src])
        res = normalize_canonical_images(batch, "linear_fit", reference_index=0)
        assert res.effective_method == "linear_fit"
        assert not res.active_frames[1]
        assert np.all(np.isnan(res.images[1]))
        assert np.all(~res.valid_mask[1])
