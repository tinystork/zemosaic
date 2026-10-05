"""SCI-05 Gate C2 — canonical combine primitive (deterministic witnesses).

Deterministic, hermetic tests for the pure CPU canonical combine layer in
``zemosaic.core.canonical_stacking``. This is **Gate C2** only: the
``mean``/``median`` combine primitive consuming the B1 normalized originals, the
B2 scalar quality weights, the C1 survivor/rejection masks, and an **explicit**
pre-rejection 2-D canonical estimator-weight map ``w_i = q_i * m_i * a_i``
(``(N, H, W)`` float64). No taper/support accumulator, Coverage, final
request/result orchestration, GUI, config, GPU, or production-caller wiring.

Design notes
------------
* Deterministic float32 corpora only; no random data, no sleeps, no network, no
  GPU, no media, no profile/XDG/HOME writes.
* The combine result uses the ORIGINAL normalized samples (never winsorized
  replacements); weight magnitude never changes median values.
* Assertions target exact output shapes/dtypes, ownership/non-aliasing, the
  exact ``estimator_weight_sum`` denominator semantics, token validation, stage
  agreement/mask-invariant rejection, deterministic repeat-run equality, and the
  constant-field / convex-hull finite invariants.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zemosaic.core.canonical_stacking import (
    CanonicalCombineResult,
    CanonicalStackValidationError,
    FrameExclusion,
    compute_canonical_quality_weights,
    combine_canonical_samples,
    normalize_canonical_images,
    prepare_canonical_inputs,
    reject_canonical_samples,
)


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------

def _full_support(h, w):
    return np.ones((h, w), dtype=bool)


def _norm(arrays, masks=None, method="none", reference_index=None):
    if masks is None:
        masks = [_full_support(*a.shape[:2]) for a in arrays]
    batch = prepare_canonical_inputs(arrays, masks)
    return normalize_canonical_images(batch, method, reference_index=reference_index)


def _weighted(norm, method="none"):
    return compute_canonical_quality_weights(norm, method)


def _rejected(norm, weight, method="none", **kw):
    return reject_canonical_samples(norm, weight, method, **kw)


def _canonical_wmap(norm, weight, taper=None):
    """Explicit pre-rejection ``w = q * m * a`` canonical estimator-weight map.

    ``q`` is the scalar B2 quality weight, ``m`` the channel-invariant valid
    mask, ``a`` the footprint taper (default all-ones within the mask). Returns
    an owned contiguous ``(N, H, W)`` float64 array.
    """
    n = norm.n_frames
    q = np.asarray(weight.weights, dtype=np.float64)
    m = np.asarray(norm.valid_mask, dtype=np.float64)
    if taper is None:
        a = np.ones((n, norm.height, norm.width), dtype=np.float64)
    else:
        a = np.asarray(taper, dtype=np.float64)
    return np.ascontiguousarray(q[:, None, None] * m * a)


def _values_corpus(values):
    """``N`` frames, each a 1x1 mono float32 image; one cell == the value list."""
    return _norm([np.full((1, 1), float(v), dtype=np.float32) for v in values])


# ---------------------------------------------------------------------------
# 1. API: shape / dtype / ownership / nonaliasing / token / type / mismatch /
#    mask invariants
# ---------------------------------------------------------------------------

class TestAPI:
    def test_result_shape_dtype_and_metadata(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        assert isinstance(res, CanonicalCombineResult)
        assert res.science.shape == (1, 1)
        assert res.science.dtype == np.float32
        assert res.estimator_weight_sum.shape == (1, 1)
        assert res.estimator_weight_sum.dtype == np.float64
        assert res.valid_mask.shape == (1, 1)
        assert res.valid_mask.dtype == np.bool_
        assert res.surviving_sample_count.shape == (1, 1)
        assert res.surviving_sample_count.dtype == np.int64
        assert res.requested_method == res.effective_method == "mean"
        assert res.reference_index == 0
        assert res.normalization_method == "none"
        assert res.weighting_method == "none"
        assert res.rejection_method == "none"
        assert res.original_mono is True
        assert (res.n_frames, res.height, res.width, res.channels) == (3, 1, 1, 1)
        assert res.original_ndim == 2 and res.original_shape == (1, 1)
        assert res.exclusions == rej.exclusions

    def test_owned_contiguous_nonaliasing(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        for arr in (res.science, res.estimator_weight_sum, res.valid_mask,
                    res.surviving_sample_count):
            assert arr.flags["OWNDATA"]
            assert arr.flags["C_CONTIGUOUS"]
        assert not np.shares_memory(res.science, norm.images)
        assert not np.shares_memory(res.estimator_weight_sum, wmap)
        assert not np.shares_memory(res.surviving_sample_count, rej.survivor_mask)
        # mutating a result array never touches inputs
        res.science[:] = 0.0
        assert not np.shares_memory(res.science, norm.images)

    def test_token_validation(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        for bad in ("average", "sum", "Mean2", "trimmed_mean", "", "MEDIANX", "weighted"):
            with pytest.raises(CanonicalStackValidationError):
                combine_canonical_samples(norm, weight, rej, wmap, bad)
        for bad in (123, None, 3.14, b"mean"):
            with pytest.raises(CanonicalStackValidationError):
                combine_canonical_samples(norm, weight, rej, wmap, bad)
        # strip/lower accepted; requested == effective exact
        r = combine_canonical_samples(norm, weight, rej, wmap, "  Median ")
        assert r.requested_method == r.effective_method == "median"

    def test_input_type_validation(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(None, weight, rej, wmap, "mean")
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, None, rej, wmap, "mean")
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, weight, None, wmap, "mean")
        # swapped / wrong result types
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(weight, norm, rej, wmap, "mean")
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, weight, norm, wmap, "mean")
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, weight, "not a result", wmap, "mean")

    def test_stage_mismatch_validation(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        # reference index mismatch
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, replace(weight, reference_index=2), rej, wmap, "mean")
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, weight, replace(rej, reference_index=2), wmap, "mean")
        # N mismatch
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, replace(weight, n_frames=2), rej, wmap, "mean")
        # H/W/C mismatch
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, replace(weight, height=2), rej, wmap, "mean")
        # original_mono mismatch
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, replace(weight, original_mono=False), rej, wmap, "mean")
        # original_shape mismatch
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, replace(weight, original_shape=(2, 2)), rej, wmap, "mean")
        # active_frames mismatch (weighting vs rejection)
        bad = replace(weight, active_frames=np.array([True, True, False], dtype=bool))
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, bad, rej, wmap, "mean")
        # weighting reactivated a frame normalization excluded
        bad_norm = replace(norm, active_frames=np.array([True, True, False], dtype=bool))
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(bad_norm, weight, rej, wmap, "mean")

    def test_exclusions_mismatch_validation(self):
        # frame 2 all-NaN -> normalization exclusion (zero_valid_support)
        a = np.ones((1, 1), dtype=np.float32)
        bad = np.full((1, 1), np.nan, dtype=np.float32)
        norm = _norm([a, a, bad])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        assert len(norm.exclusions) == 1
        # weighting/rejection exclusions mismatch
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, replace(weight, exclusions=tuple()), rej, wmap, "mean")
        # normalization exclusions not a prefix of weighting exclusions
        wrong = replace(
            norm,
            exclusions=(FrameExclusion(index=99, stage="normalization",
                                       reason="zero_valid_support"),),
        )
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(wrong, weight, rej, wmap, "mean")

    def test_mutated_mask_overlap_rejected(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        rm = rej.rejection_mask.copy()
        rm[0, 0, 0, 0] = True  # overlaps survivor -> disjoint invariant broken
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, weight, replace(rej, rejection_mask=rm), wmap, "mean")

    def test_mutated_mask_partition_rejected(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        sm = rej.survivor_mask.copy()
        sm[0, 0, 0, 0] = False  # drops a survivor without rejecting it
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, weight, replace(rej, survivor_mask=sm), wmap, "mean")

    def test_survivor_outside_initial_rejected(self):
        # one pixel NaN in every frame -> invalid cell; survivor must stay False
        arrays = [np.array([[1.0, np.nan]], dtype=np.float32) for _ in range(3)]
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        sm = rej.survivor_mask.copy()
        sm[:, 0, 1, 0] = True  # marks the invalid (NaN) cell as survivor
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, weight, replace(rej, survivor_mask=sm), wmap, "mean")

    def test_validation_does_not_mutate_inputs(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        img_before = norm.images.copy()
        mask_before = norm.valid_mask.copy()
        w_before = weight.weights.copy()
        sm_before = rej.survivor_mask.copy()
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, weight, rej, np.full((3, 1, 1), 2.0), "mean")
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, weight, rej, wmap, "bogus")
        assert np.array_equal(norm.images, img_before)
        assert np.array_equal(norm.valid_mask, mask_before)
        assert np.array_equal(weight.weights, w_before)
        assert np.array_equal(rej.survivor_mask, sm_before)


# ---------------------------------------------------------------------------
# 2. Estimator weight validation
# ---------------------------------------------------------------------------

class TestEstimatorWeightValidation:
    def _ctx(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        return norm, _weighted(norm, "none"), _rejected(norm, _weighted(norm, "none"), "none")

    def test_wrong_shape(self):
        norm, weight, rej = self._ctx()
        for bad_shape in ((3, 2, 1), (3, 1), (3, 1, 1, 1), (2, 1, 1), (4, 1, 1)):
            with pytest.raises(CanonicalStackValidationError):
                combine_canonical_samples(norm, weight, rej, np.ones(bad_shape), "mean")

    def test_wrong_dtype(self):
        norm, weight, rej = self._ctx()
        for bad in (np.ones((3, 1, 1), dtype=bool),
                    np.ones((3, 1, 1), dtype=np.complex128),
                    np.ones((3, 1, 1), dtype=object)):
            with pytest.raises(CanonicalStackValidationError):
                combine_canonical_samples(norm, weight, rej, bad, "mean")
        # non-array-convertible
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, weight, rej, "not weights", "mean")

    def test_negative_nonfinite_over_one(self):
        norm, weight, rej = self._ctx()
        base = np.full((3, 1, 1), 0.5, dtype=np.float64)
        for mutate in (-0.1, 1.5, np.nan, np.inf):
            wmap = base.copy()
            wmap[0, 0, 0] = mutate
            with pytest.raises(CanonicalStackValidationError):
                combine_canonical_samples(norm, weight, rej, wmap, "mean")

    def test_positive_on_invalid(self):
        # NaN pixel -> outside valid mask; positive weight there rejected
        arrays = [np.array([[1.0, np.nan]], dtype=np.float32) for _ in range(3)]
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        wmap[0, 0, 1] = 0.5  # positive weight on invalid pixel
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, weight, rej, wmap, "mean")

    def test_positive_on_inactive(self):
        # frame 2 all-NaN -> inactive; positive weight there rejected
        a = np.ones((1, 1), dtype=np.float32)
        bad = np.full((1, 1), np.nan, dtype=np.float32)
        norm = _norm([a, a, bad])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        assert wmap[2, 0, 0] == 0.0  # inactive frame forced zero
        wmap[2, 0, 0] = 0.5
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, weight, rej, wmap, "mean")

    def test_exceeds_q(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        base_weight = _weighted(norm, "none")
        weight = replace(base_weight, weights=np.array([0.5, 0.5, 0.5]))
        rej = _rejected(norm, base_weight, "none")
        wmap = np.full((3, 1, 1), 0.7, dtype=np.float64)  # 0.7 > q = 0.5
        with pytest.raises(CanonicalStackValidationError):
            combine_canonical_samples(norm, weight, rej, wmap, "mean")
        # w == q is allowed
        ok = np.full((3, 1, 1), 0.5, dtype=np.float64)
        res = combine_canonical_samples(norm, weight, rej, ok, "mean")
        assert res.valid_mask[0, 0]

    def test_estimator_weights_input_not_mutated(self):
        norm, weight, rej = self._ctx()
        wmap = np.full((3, 1, 1), 0.5, dtype=np.float64)
        orig = wmap.copy()
        combine_canonical_samples(norm, weight, rej, wmap, "mean")
        assert np.array_equal(wmap, orig)


# ---------------------------------------------------------------------------
# 3. mean analytical semantics
# ---------------------------------------------------------------------------

class TestMean:
    def test_analytical_weighted_mean(self):
        norm = _values_corpus([2.0, 4.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = np.array([[[0.25]], [[0.75]]], dtype=np.float64)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        assert float(res.science[0, 0]) == pytest.approx(3.5, rel=1e-6)
        assert float(res.estimator_weight_sum[0, 0]) == pytest.approx(1.0, rel=1e-12)
        assert res.valid_mask[0, 0]
        assert res.surviving_sample_count[0, 0] == 2

    def test_denominator_is_sum_of_weights(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = np.array([[[0.1]], [[0.2]], [[0.3]]], dtype=np.float64)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        expected = (1.0 * 0.1 + 2.0 * 0.2 + 3.0 * 0.3) / (0.1 + 0.2 + 0.3)
        assert float(res.science[0, 0]) == pytest.approx(expected, rel=1e-6)
        assert float(res.estimator_weight_sum[0, 0]) == pytest.approx(0.6, rel=1e-12)
        assert res.surviving_sample_count[0, 0] == 3

    def test_q_m_a_spatial_map(self):
        arrays = [
            np.array([[2.0, 6.0]], dtype=np.float32),
            np.array([[4.0, 10.0]], dtype=np.float32),
        ]
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = np.array([[[0.5, 0.1]], [[0.5, 0.3]]], dtype=np.float64)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        # pixel 0: (2*0.5 + 4*0.5) / 1.0 = 3.0
        assert float(res.science[0, 0]) == pytest.approx(3.0, rel=1e-6)
        # pixel 1: (6*0.1 + 10*0.3) / 0.4 = 9.0
        assert float(res.science[0, 1]) == pytest.approx(9.0, rel=1e-6)
        assert float(res.estimator_weight_sum[0, 0]) == pytest.approx(1.0, rel=1e-12)
        assert float(res.estimator_weight_sum[0, 1]) == pytest.approx(0.4, rel=1e-12)

    def test_channel_specific_survivor_gives_channel_specific_sum(self):
        r = [10.0] * 9 + [100.0]
        g = [10.0] * 10
        b = [10.0] * 10
        arrays = [
            np.array([[[r[i], g[i], b[i]]]], dtype=np.float32) for i in range(len(r))
        ]
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "kappa_sigma")
        wmap = _canonical_wmap(norm, weight)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        assert float(res.science[0, 0, 0]) == pytest.approx(10.0, rel=1e-6)
        assert float(res.estimator_weight_sum[0, 0, 0]) == pytest.approx(9.0, rel=1e-12)
        assert float(res.science[0, 0, 1]) == pytest.approx(10.0, rel=1e-6)
        assert float(res.estimator_weight_sum[0, 0, 1]) == pytest.approx(10.0, rel=1e-12)
        assert float(res.estimator_weight_sum[0, 0, 2]) == pytest.approx(10.0, rel=1e-12)

    def test_tiny_positive_denominator_no_epsilon(self):
        norm = _values_corpus([1.0, 1.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = np.full((2, 1, 1), 1e-300, dtype=np.float64)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        assert res.valid_mask[0, 0]
        assert float(res.science[0, 0]) == pytest.approx(1.0, rel=1e-6)
        assert float(res.estimator_weight_sum[0, 0]) == pytest.approx(2e-300)

    def test_zero_denominator_all_invalid(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = np.zeros((3, 1, 1), dtype=np.float64)  # all w=0 -> absent
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        assert not res.valid_mask[0, 0]
        assert np.isnan(res.science[0, 0])
        assert float(res.estimator_weight_sum[0, 0]) == 0.0
        assert res.surviving_sample_count[0, 0] == 0
        assert res.valid_output_count == 0
        assert res.invalid_output_count == 1

    def test_n1(self):
        norm = _values_corpus([7.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = np.array([[[1.0]]], dtype=np.float64)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        assert float(res.science[0, 0]) == pytest.approx(7.0)
        assert float(res.estimator_weight_sum[0, 0]) == pytest.approx(1.0)
        assert res.surviving_sample_count[0, 0] == 1


# ---------------------------------------------------------------------------
# 4. Constant-field invariant under varying positive spatial weights / taper
# ---------------------------------------------------------------------------

class TestConstantField:
    def test_mean_constant_under_varying_weights(self):
        arrays = [np.full((3, 3), 5.0, dtype=np.float32) for _ in range(4)]
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = np.array([
            [[1.0, 0.5, 0.2], [0.8, 0.3, 0.1], [0.9, 0.6, 0.4]],
            [[0.1, 0.9, 0.7], [0.4, 1.0, 0.2], [0.3, 0.8, 0.5]],
            [[0.6, 0.2, 0.9], [0.7, 0.5, 1.0], [0.4, 0.1, 0.8]],
            [[0.5, 0.8, 0.1], [0.2, 0.6, 0.9], [1.0, 0.7, 0.3]],
        ], dtype=np.float64)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        assert np.all(res.valid_mask)
        np.testing.assert_allclose(res.science, 5.0, rtol=1e-6, atol=1e-6)

    def test_median_constant_under_varying_weights(self):
        arrays = [np.full((3, 3), 5.0, dtype=np.float32) for _ in range(5)]
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = np.array([
            [[1.0, 0.5, 0.2], [0.8, 0.3, 0.1], [0.9, 0.6, 0.4]],
            [[0.1, 0.9, 0.7], [0.4, 1.0, 0.2], [0.3, 0.8, 0.5]],
            [[0.6, 0.2, 0.9], [0.7, 0.5, 1.0], [0.4, 0.1, 0.8]],
            [[0.5, 0.8, 0.1], [0.2, 0.6, 0.9], [1.0, 0.7, 0.3]],
            [[0.3, 0.4, 0.6], [0.9, 0.7, 0.5], [0.2, 0.5, 0.8]],
        ], dtype=np.float64)
        res = combine_canonical_samples(norm, weight, rej, wmap, "median")
        assert np.all(res.valid_mask)
        np.testing.assert_allclose(res.science, 5.0, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. median semantics
# ---------------------------------------------------------------------------

class TestMedian:
    def test_odd_median(self):
        norm = _values_corpus([1.0, 3.0, 2.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        res = combine_canonical_samples(norm, weight, rej, wmap, "median")
        assert float(res.science[0, 0]) == pytest.approx(2.0)
        assert float(res.estimator_weight_sum[0, 0]) == pytest.approx(3.0)
        assert res.surviving_sample_count[0, 0] == 3

    def test_even_median_average_middle_two(self):
        norm = _values_corpus([1.0, 4.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        res = combine_canonical_samples(norm, weight, rej, wmap, "median")
        assert float(res.science[0, 0]) == pytest.approx(2.5)
        assert res.surviving_sample_count[0, 0] == 4
        assert float(res.estimator_weight_sum[0, 0]) == pytest.approx(4.0)

    def test_positive_weight_magnitude_ignored(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        big = np.array([[[1e-9]], [[0.999]], [[1e-6]]], dtype=np.float64)
        res = combine_canonical_samples(norm, weight, rej, big, "median")
        assert float(res.science[0, 0]) == pytest.approx(2.0)
        # estimator_weight_sum is the COUNT, not the sum of magnitudes
        assert float(res.estimator_weight_sum[0, 0]) == pytest.approx(3.0)
        assert res.surviving_sample_count[0, 0] == 3

    def test_zero_weight_gates_out(self):
        norm = _values_corpus([1.0, 3.0, 2.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = np.array([[[1.0]], [[1.0]], [[0.0]]], dtype=np.float64)
        res = combine_canonical_samples(norm, weight, rej, wmap, "median")
        assert float(res.science[0, 0]) == pytest.approx(2.0)  # median of [1, 3]
        assert float(res.estimator_weight_sum[0, 0]) == pytest.approx(2.0)
        assert res.surviving_sample_count[0, 0] == 2

    def test_channel_specific_survivor(self):
        r = [10.0] * 9 + [100.0]
        g = [10.0] * 10
        b = [10.0] * 10
        arrays = [
            np.array([[[r[i], g[i], b[i]]]], dtype=np.float32) for i in range(len(r))
        ]
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "kappa_sigma")
        wmap = _canonical_wmap(norm, weight)
        res = combine_canonical_samples(norm, weight, rej, wmap, "median")
        assert float(res.science[0, 0, 0]) == pytest.approx(10.0)
        assert res.surviving_sample_count[0, 0, 0] == 9
        assert float(res.estimator_weight_sum[0, 0, 0]) == pytest.approx(9.0)
        assert res.surviving_sample_count[0, 0, 1] == 10
        assert float(res.estimator_weight_sum[0, 0, 1]) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 6. Original sample semantics (WSC combine uses originals, not winsor values)
# ---------------------------------------------------------------------------

class TestOriginalSemantics:
    def test_combine_uses_originals_not_winsor_replacements(self):
        values = [10.0] * 9 + [100.0]
        norm = _values_corpus(values)
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "winsorized_sigma_clip")
        assert not rej.survivor_mask[9, 0, 0, 0]  # outlier rejected
        wmap = _canonical_wmap(norm, weight)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        assert float(res.science[0, 0]) == pytest.approx(10.0, rel=1e-6)
        assert float(res.estimator_weight_sum[0, 0]) == pytest.approx(9.0)
        # source image still holds the ORIGINAL 100 (untouched)
        assert float(norm.images[9, 0, 0, 0]) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 7. Shape restoration
# ---------------------------------------------------------------------------

class TestShapeRestoration:
    def test_hw_mono_restores_hw(self):
        arrays = [np.ones((2, 3), dtype=np.float32) for _ in range(3)]
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        for method in ("mean", "median"):
            res = combine_canonical_samples(norm, weight, rej, wmap, method)
            assert res.science.shape == (2, 3)
            assert res.estimator_weight_sum.shape == (2, 3)
            assert res.valid_mask.shape == (2, 3)
            assert res.surviving_sample_count.shape == (2, 3)
            assert res.science.dtype == np.float32
            assert res.estimator_weight_sum.dtype == np.float64
            assert res.valid_mask.dtype == np.bool_
            assert res.surviving_sample_count.dtype == np.int64

    def test_hwc1_restores_hwc1(self):
        hwc1 = np.ones((2, 3, 1), dtype=np.float32)
        norm = _norm([hwc1] * 3)
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        assert res.science.shape == (2, 3, 1)
        assert res.estimator_weight_sum.shape == (2, 3, 1)
        assert res.valid_mask.shape == (2, 3, 1)
        assert res.surviving_sample_count.shape == (2, 3, 1)

    def test_rgb_restores_hwc(self):
        rgb = np.ones((2, 3, 3), dtype=np.float32)
        norm = _norm([rgb] * 3)
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        assert res.science.shape == (2, 3, 3)
        assert res.estimator_weight_sum.shape == (2, 3, 3)
        assert res.valid_mask.shape == (2, 3, 3)
        assert res.surviving_sample_count.shape == (2, 3, 3)


# ---------------------------------------------------------------------------
# 8. Post-cast finite invariant / no Inf-valid
# ---------------------------------------------------------------------------

class TestPostCastFinite:
    def test_valid_implies_finite_science(self):
        big = np.full((1, 1), 3.0e38, dtype=np.float32)  # near float32 max
        norm = _norm([big.copy() for _ in range(4)])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        for method in ("mean", "median"):
            res = combine_canonical_samples(norm, weight, rej, wmap, method)
            assert res.valid_mask[0, 0]
            assert np.isfinite(res.science[0, 0])
            assert res.nonfinite_output_count == 0
            assert np.all(np.isfinite(res.science[res.valid_mask]))

    def test_no_inf_or_nan_marked_valid(self):
        arrays = [
            np.array([[1.0, 3.0e38]], dtype=np.float32),
            np.array([[2.0, 3.0e38]], dtype=np.float32),
            np.array([[3.0, 3.0e38]], dtype=np.float32),
        ]
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        for method in ("mean", "median"):
            res = combine_canonical_samples(norm, weight, rej, wmap, method)
            assert np.all(np.isfinite(res.science[res.valid_mask]))
            assert res.nonfinite_output_count == 0

    def test_convex_hull_stays_finite(self):
        # The mean/median of finite float32 samples with weights in [0, 1] is a
        # convex combination: it lies within [min, max] of the inputs and so can
        # never overflow float32. The defensive nonfinite-output branch is
        # therefore unreachable for well-formed inputs; this test documents that.
        arrays = [
            np.array([[3.0e38, -3.0e38]], dtype=np.float32),
            np.array([[-3.0e38, 3.0e38]], dtype=np.float32),
        ]
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        for method in ("mean", "median"):
            res = combine_canonical_samples(norm, weight, rej, wmap, method)
            assert np.all(np.isfinite(res.science[res.valid_mask]))
            assert res.nonfinite_output_count == 0


# ---------------------------------------------------------------------------
# 9. Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_exact_counts(self):
        arrays = []
        for _ in range(3):
            img = np.ones((2, 2), dtype=np.float32)
            img[1, 1] = np.nan
            arrays.append(img)
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        assert res.input_surviving_sample_count == 9
        assert res.contributing_sample_count == 9
        assert res.valid_output_count == 3
        assert res.invalid_output_count == 1
        assert res.nonfinite_output_count == 0

    def test_contributing_less_than_input_when_weight_zero(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = np.array([[[1.0]], [[1.0]], [[0.0]]], dtype=np.float64)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        assert res.input_surviving_sample_count == 3
        assert res.contributing_sample_count == 2
        assert res.valid_output_count == 1
        assert res.invalid_output_count == 0

    def test_rejection_reduces_input_surviving(self):
        norm = _values_corpus([10.0] * 9 + [100.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "kappa_sigma")
        wmap = _canonical_wmap(norm, weight)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        assert res.input_surviving_sample_count == 9
        assert res.contributing_sample_count == 9
        assert res.initial_sample_count == 10
        assert res.rejected_sample_count == 1

    def test_deterministic_python_scalar_types(self):
        norm = _values_corpus([1.0, 2.0, 3.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "none")
        wmap = _canonical_wmap(norm, weight)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        for name in ("valid_output_count", "invalid_output_count",
                     "input_surviving_sample_count", "contributing_sample_count",
                     "nonfinite_output_count"):
            assert type(getattr(res, name)) is int
        assert type(res.reference_index) is int
        assert type(res.rejected_fraction) is float

    def test_repeat_run_deterministic_and_no_mutation(self):
        norm = _values_corpus([10.0] * 9 + [100.0])
        weight = _weighted(norm, "none")
        rej = _rejected(norm, weight, "kappa_sigma")
        wmap = _canonical_wmap(norm, weight)
        img_before = norm.images.copy()
        wmap_before = wmap.copy()
        sm_before = rej.survivor_mask.copy()
        rm_before = rej.rejection_mask.copy()
        r1 = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        r2 = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        np.testing.assert_array_equal(r1.science, r2.science)
        np.testing.assert_array_equal(r1.estimator_weight_sum, r2.estimator_weight_sum)
        np.testing.assert_array_equal(r1.valid_mask, r2.valid_mask)
        np.testing.assert_array_equal(r1.surviving_sample_count, r2.surviving_sample_count)
        assert r1.valid_output_count == r2.valid_output_count
        assert np.array_equal(norm.images, img_before)
        assert np.array_equal(wmap, wmap_before)
        assert np.array_equal(rej.survivor_mask, sm_before)
        assert np.array_equal(rej.rejection_mask, rm_before)


# ---------------------------------------------------------------------------
# 10. End-to-end pure stage witness
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_mean_and_median_pipeline_with_rejection(self):
        base = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        frames = [base + float(k) for k in range(9)] + [
            np.full((2, 2), 1000.0, dtype=np.float32)
        ]
        norm = _norm(frames)                            # B1 normalize
        weight = _weighted(norm, "none")                # B2 weights
        wmap = _canonical_wmap(norm, weight)            # explicit w=q*m*a
        rej = _rejected(norm, weight, "kappa_sigma")    # C1 rejection
        expected = base + 4.0
        for method in ("mean", "median"):
            res = combine_canonical_samples(norm, weight, rej, wmap, method)  # C2
            np.testing.assert_allclose(res.science, expected, rtol=1e-6, atol=1e-6)
            assert np.all(res.surviving_sample_count == 9)
            assert np.all(res.valid_mask)
            assert res.input_surviving_sample_count == 36
            assert res.contributing_sample_count == 36
            assert res.valid_output_count == 4
            assert res.invalid_output_count == 0
            assert res.rejected_sample_count == 4
