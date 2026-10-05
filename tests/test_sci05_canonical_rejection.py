"""SCI-05 Gate C1 — canonical rejection primitives (deterministic witnesses).

Deterministic, hermetic tests for the pure CPU canonical rejection layer in
``zemosaic.core.canonical_stacking``. This is **Gate C1** only: rejection masks
(``none``/``kappa_sigma``/``winsorized_sigma_clip``) on top of the B1 normalized
and B2 weighted batch. No combine, coverage, GUI, config, GPU, or
production-caller wiring.

Design notes
------------
* Deterministic float32 corpora only; no random data, no sleeps, no network, no
  GPU, no media, no profile/XDG/HOME writes.
* Analytical corpora are 1x1 (mono) or 1x1x3 (RGB) stacks so a cell's frame-axis
  values are exactly the input list; a tiny independent reference implementation
  (explicit loops + ``np.quantile(method="linear")``) cross-checks every mask.
* Assertions target stable reason/token codes, requested==effective method,
  N-axis/index preservation, non-aliasing, weight-independence, and deterministic
  repeat-run equality.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zemosaic.core.canonical_stacking import (
    CanonicalRejectionResult,
    CanonicalStackValidationError,
    compute_canonical_quality_weights,
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


def _values_corpus(values):
    """``N`` frames, each a 1x1 mono float32 image; one cell == the value list."""
    return _norm([np.full((1, 1), float(v), dtype=np.float32) for v in values])


def _rgb_values_corpus(r, g, b):
    """``N`` frames, each a 1x1x3 RGB float32 image."""
    arrays = [
        np.array([[[float(r[i]), float(g[i]), float(b[i])]]], dtype=np.float32)
        for i in range(len(r))
    ]
    return _norm(arrays)


# ---------------------------------------------------------------------------
# Independent tiny reference (explicit loops; never imports legacy helpers)
# ---------------------------------------------------------------------------

def _reference_reject(images, initial, method, sigma_low, sigma_high, max_iters,
                      winsor_low=0.05, winsor_high=0.05):
    """Reference survivor mask + (iterations_used, degenerate_cell_count).

    ``images`` is ``(N, H, W, C)`` float64, ``initial`` is the same-shape bool.
    Explicit per-cell Python loops; WSC quantiles use exact NumPy
    ``method="linear"`` semantics; statistics float64; monotonic intersection.
    """
    n, h, w, c = images.shape
    survivor = initial.copy()
    iterations = 0
    degenerate_cells = set()
    if method == "none":
        return survivor, 0, 0
    for _ in range(max_iters):
        iterations += 1
        changed = False
        for y in range(h):
            for x in range(w):
                for k in range(c):
                    idx = np.nonzero(survivor[:, y, x, k])[0]
                    if idx.size < 3:
                        continue
                    old = survivor[:, y, x, k].copy()
                    vals = images[idx, y, x, k].astype(np.float64)
                    if method == "kappa_sigma":
                        center = float(np.median(vals))
                        std = float(np.std(vals))  # population ddof=0
                    else:  # winsorized_sigma_clip
                        q_low = float(np.quantile(vals, winsor_low, method="linear"))
                        q_high = float(np.quantile(vals, 1.0 - winsor_high, method="linear"))
                        win = np.clip(vals, q_low, q_high)
                        center = float(np.mean(win))
                        std = float(np.std(win))
                    if std <= 0.0:
                        degenerate_cells.add((y, x, k))
                        keep = images[:, y, x, k] == center
                    else:
                        lo = center - sigma_low * std
                        hi = center + sigma_high * std
                        keep = (images[:, y, x, k] >= lo) & (images[:, y, x, k] <= hi)
                    new = old & keep
                    if not np.array_equal(new, old):
                        changed = True
                    survivor[:, y, x, k] = new
        if not changed:
            break
    return survivor, iterations, len(degenerate_cells)


def _reference_diagnostics(images, initial, method, **kw):
    survivor, iterations, degenerate = _reference_reject(
        images, initial, method,
        sigma_low=kw.get("sigma_low", 3.0),
        sigma_high=kw.get("sigma_high", 3.0),
        max_iters=kw.get("max_iters", 5),
        winsor_low=kw.get("winsor_limit_low", 0.05),
        winsor_high=kw.get("winsor_limit_high", 0.05),
    )
    init_count = int(initial.sum())
    surv_count = int(survivor.sum())
    rej_count = int((initial & ~survivor).sum())
    return {
        "survivor": survivor,
        "iterations_used": iterations,
        "degenerate_cell_count": degenerate,
        "initial_sample_count": init_count,
        "surviving_sample_count": surv_count,
        "rejected_sample_count": rej_count,
        "rejected_fraction": (rej_count / init_count) if init_count > 0 else 0.0,
        "low_n_cell_count": int((initial.sum(axis=0) < 3).sum()),
    }


# ---------------------------------------------------------------------------
# 1. API: shapes / dtypes / owned / nonaliasing / token / parameter / mismatch
# ---------------------------------------------------------------------------

class TestAPI:
    def test_result_shape_dtype_and_metadata(self):
        norm = _values_corpus([1.0, 2.0, 3.0, 4.0, 5.0])
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "kappa_sigma")
        assert isinstance(res, CanonicalRejectionResult)
        assert res.survivor_mask.shape == (5, 1, 1, 1)
        assert res.rejection_mask.shape == (5, 1, 1, 1)
        assert res.survivor_mask.dtype == np.bool_
        assert res.rejection_mask.dtype == np.bool_
        assert res.active_frames.shape == (5,) and res.active_frames.dtype == np.bool_
        assert res.reference_index == 0
        assert res.requested_method == res.effective_method == "kappa_sigma"
        assert res.sigma_low == 3.0 and res.sigma_high == 3.0
        assert res.max_iters == 5
        assert res.winsor_limit_low == 0.05 and res.winsor_limit_high == 0.05
        assert res.original_mono is True
        assert (res.n_frames, res.height, res.width, res.channels) == (5, 1, 1, 1)
        assert res.original_ndim == 2 and res.original_shape == (1, 1)
        assert res.exclusions == weight.exclusions

    def test_owned_contiguous_nonaliasing(self):
        norm = _values_corpus([1.0, 2.0, 3.0, 4.0, 5.0])
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "winsorized_sigma_clip")
        for arr in (res.survivor_mask, res.rejection_mask, res.active_frames):
            assert arr.flags["OWNDATA"]
            assert arr.flags["C_CONTIGUOUS"]
        assert not np.shares_memory(res.survivor_mask, norm.images)
        assert not np.shares_memory(res.rejection_mask, norm.images)
        assert not np.shares_memory(res.active_frames, weight.active_frames)
        # mutating a result array never touches inputs
        res.survivor_mask[:] = False
        assert not np.shares_memory(res.survivor_mask, norm.images)

    def test_token_validation(self):
        norm = _values_corpus([1.0, 2.0, 3.0, 4.0, 5.0])
        weight = _weighted(norm, "none")
        for bad in ("kappa", "winsorized", "KappaSigma", "sigma_clip", "median", "", "NONE2"):
            with pytest.raises(CanonicalStackValidationError):
                reject_canonical_samples(norm, weight, bad)
        for bad in (123, None, 3.14, b"none"):
            with pytest.raises(CanonicalStackValidationError):
                reject_canonical_samples(norm, weight, bad)
        # strip/lower accepted; requested==effective exact
        r = reject_canonical_samples(norm, weight, "  Winsorized_Sigma_Clip ")
        assert r.requested_method == r.effective_method == "winsorized_sigma_clip"

    def test_linear_fit_clip_exact_removed_token(self):
        norm = _values_corpus([1.0, 2.0, 3.0, 4.0, 5.0])
        weight = _weighted(norm, "none")
        with pytest.raises(CanonicalStackValidationError) as ei:
            reject_canonical_samples(norm, weight, "linear_fit_clip")
        assert "unsupported_removed_sci05" in str(ei.value)
        # strip/lower form hits the same stable token
        with pytest.raises(CanonicalStackValidationError) as ei:
            reject_canonical_samples(norm, weight, "  Linear_Fit_Clip ")
        assert "unsupported_removed_sci05" in str(ei.value)
        # a longer unknown token is NOT the removed token (generic unknown failure)
        with pytest.raises(CanonicalStackValidationError) as ei:
            reject_canonical_samples(norm, weight, "linear_fit_clip_extra")
        assert "unsupported_removed_sci05" not in str(ei.value)

    def test_sigma_parameter_validation(self):
        norm = _values_corpus([1.0, 2.0, 3.0, 4.0, 5.0])
        weight = _weighted(norm, "none")
        for bad in (0.0, -1.0, float("nan"), float("inf"), True, "3.0", None):
            with pytest.raises(CanonicalStackValidationError):
                reject_canonical_samples(norm, weight, "kappa_sigma", sigma_low=bad)
            with pytest.raises(CanonicalStackValidationError):
                reject_canonical_samples(norm, weight, "kappa_sigma", sigma_high=bad)
        # valid ints/floats accepted
        r = reject_canonical_samples(norm, weight, "kappa_sigma", sigma_low=2, sigma_high=4)
        assert r.sigma_low == 2.0 and r.sigma_high == 4.0

    def test_max_iters_parameter_validation(self):
        norm = _values_corpus([1.0, 2.0, 3.0, 4.0, 5.0])
        weight = _weighted(norm, "none")
        for bad in (0, 6, -1, True, 2.5, "5", None):
            with pytest.raises(CanonicalStackValidationError):
                reject_canonical_samples(norm, weight, "kappa_sigma", max_iters=bad)
        r = reject_canonical_samples(norm, weight, "kappa_sigma", max_iters=1)
        assert r.max_iters == 1

    def test_winsor_limit_parameter_validation(self):
        norm = _values_corpus([1.0, 2.0, 3.0, 4.0, 5.0])
        weight = _weighted(norm, "none")
        for bad in (-0.1, 0.5, 0.6, float("nan"), float("inf"), True, "0.05", None):
            with pytest.raises(CanonicalStackValidationError):
                reject_canonical_samples(norm, weight, "none", winsor_limit_low=bad)
            with pytest.raises(CanonicalStackValidationError):
                reject_canonical_samples(norm, weight, "none", winsor_limit_high=bad)
        # low + high >= 1 fails
        with pytest.raises(CanonicalStackValidationError):
            reject_canonical_samples(
                norm, weight, "none", winsor_limit_low=0.6, winsor_limit_high=0.6
            )
        # boundary 0.0 allowed
        r = reject_canonical_samples(norm, weight, "none", winsor_limit_low=0.0, winsor_limit_high=0.0)
        assert r.winsor_limit_low == 0.0 and r.winsor_limit_high == 0.0

    def test_input_type_validation(self):
        norm = _values_corpus([1.0, 2.0, 3.0, 4.0, 5.0])
        weight = _weighted(norm, "none")
        with pytest.raises(CanonicalStackValidationError):
            reject_canonical_samples(None, weight, "none")
        with pytest.raises(CanonicalStackValidationError):
            reject_canonical_samples(norm, None, "none")
        # wrong result types (swapped)
        with pytest.raises(CanonicalStackValidationError):
            reject_canonical_samples(weight, norm, "none")
        with pytest.raises(CanonicalStackValidationError):
            reject_canonical_samples("not a result", weight, "none")

    def test_b1_b2_mismatch_validation(self):
        norm = _values_corpus([1.0, 2.0, 3.0, 4.0, 5.0])
        weight = _weighted(norm, "none")
        # reference index mismatch
        bad = replace(weight, reference_index=4)
        with pytest.raises(CanonicalStackValidationError):
            reject_canonical_samples(norm, bad, "none")
        # N mismatch
        bad = replace(weight, n_frames=4)
        with pytest.raises(CanonicalStackValidationError):
            reject_canonical_samples(norm, bad, "none")
        # H/W/C mismatch
        bad = replace(weight, height=2)
        with pytest.raises(CanonicalStackValidationError):
            reject_canonical_samples(norm, bad, "none")
        # original_mono mismatch
        bad = replace(weight, original_mono=False)
        with pytest.raises(CanonicalStackValidationError):
            reject_canonical_samples(norm, bad, "none")
        # original_shape mismatch
        bad = replace(weight, original_shape=(2, 2))
        with pytest.raises(CanonicalStackValidationError):
            reject_canonical_samples(norm, bad, "none")
        # active_frames wrong length
        bad = replace(weight, active_frames=np.array([True, True], dtype=bool))
        with pytest.raises(CanonicalStackValidationError):
            reject_canonical_samples(norm, bad, "none")

    def test_validation_does_not_mutate_inputs(self):
        norm = _values_corpus([1.0, 2.0, 3.0, 4.0, 5.0])
        weight = _weighted(norm, "none")
        img_before = norm.images.copy()
        mask_before = norm.valid_mask.copy()
        w_before = weight.weights.copy()
        with pytest.raises(CanonicalStackValidationError):
            reject_canonical_samples(norm, weight, "kappa_sigma", sigma_low=-1.0)
        with pytest.raises(CanonicalStackValidationError):
            reject_canonical_samples(norm, weight, "bogus")
        assert np.array_equal(norm.images, img_before)
        assert np.array_equal(norm.valid_mask, mask_before)
        assert np.array_equal(weight.weights, w_before)


# ---------------------------------------------------------------------------
# 2. none
# ---------------------------------------------------------------------------

class TestNone:
    def test_none_exact_survivors_no_rejection(self):
        # frame 2 prior-inactive (all-NaN) + a channel-invalid pixel elsewhere
        a = np.ones((3, 3), dtype=np.float32)
        b = np.ones((3, 3), dtype=np.float32)
        bad = np.full((3, 3), np.nan, dtype=np.float32)
        norm = _norm([a, b, bad])
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "none")
        assert res.iterations_used == 0
        assert res.degenerate_cell_count == 0
        assert not res.rejection_mask.any()
        assert res.rejected_sample_count == 0
        assert res.surviving_sample_count == res.initial_sample_count == 2 * 9
        # active_frames inherited: [True, True, False]
        assert res.active_frames.tolist() == [True, True, False]

    def test_prior_invalid_inactive_never_marked_rejected(self):
        a = np.ones((3, 3), dtype=np.float32)
        b = np.ones((3, 3), dtype=np.float32)
        # frame 2 inactive (all NaN); frame 1 has one NaN pixel -> invalid
        b2 = b.copy()
        b2[0, 0] = np.nan
        norm = _norm([a, b2, np.full((3, 3), np.nan, dtype=np.float32)])
        weight = _weighted(norm, "none")
        for method in ("none", "kappa_sigma", "winsorized_sigma_clip"):
            res = reject_canonical_samples(norm, weight, method)
            # rejection_mask is a subset of the initial (active & finite) mask
            initial = (
                weight.active_frames[:, None, None, None]
                & norm.valid_mask[..., None]
                & np.isfinite(norm.images)
            )
            assert np.all(~res.rejection_mask | initial)
            # inactive frame never rejected
            assert not res.rejection_mask[2].any()

    def test_all_invalid_cells_remain_all_false(self):
        # one pixel is NaN in every frame -> all-invalid cell stays all false;
        # no arbitrary science zero is produced for it (combine is C2).
        arrays = []
        for _ in range(4):
            img = np.ones((2, 2), dtype=np.float32)
            img[1, 1] = np.nan
            arrays.append(img)
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "kappa_sigma")
        assert not res.survivor_mask[:, 1, 1, 0].any()
        assert not res.rejection_mask[:, 1, 1, 0].any()
        assert res.survivor_mask[:, 0, 0, 0].all()  # valid cells survive
        assert np.isnan(norm.images[0, 1, 1, 0])  # source NaN untouched

    def test_hw_hwc1_rgb_shapes(self):
        # HW mono
        norm = _norm([np.ones((2, 2), dtype=np.float32)] * 3)
        res = reject_canonical_samples(norm, _weighted(norm, "none"), "none")
        assert res.survivor_mask.shape == (3, 2, 2, 1)
        assert res.original_mono is True and res.channels == 1
        # HWC1
        hwc1 = np.ones((2, 2, 1), dtype=np.float32)
        norm1 = _norm([hwc1] * 3)
        res1 = reject_canonical_samples(norm1, _weighted(norm1, "none"), "none")
        assert res1.survivor_mask.shape == (3, 2, 2, 1)
        assert res1.original_mono is False and res1.channels == 1
        # RGB
        rgb = np.ones((2, 2, 3), dtype=np.float32)
        norm3 = _norm([rgb] * 3)
        res3 = reject_canonical_samples(norm3, _weighted(norm3, "none"), "none")
        assert res3.survivor_mask.shape == (3, 2, 2, 3)
        assert res3.channels == 3


# ---------------------------------------------------------------------------
# 3. Weight independence
# ---------------------------------------------------------------------------

class TestWeightIndependence:
    def test_radically_different_weights_bit_identical(self):
        # outlier corpus so rejection is non-trivial
        norm = _values_corpus([10.0] * 9 + [100.0])
        base = _weighted(norm, "none")  # weights == [1.0]*10
        # same active_frames, radically different positive magnitudes
        big = replace(
            base,
            weights=np.array([1e9, 1e-9, 3.0, 0.25, 7.5e6, 2.0, 1.0, 1e-3, 42.0, 5.0]),
            raw_weights=np.array([1e9, 1e-9, 3.0, 0.25, 7.5e6, 2.0, 1.0, 1e-3, 42.0, 5.0]),
        )
        assert np.array_equal(base.active_frames, big.active_frames)
        for method in ("kappa_sigma", "winsorized_sigma_clip"):
            r_base = reject_canonical_samples(norm, base, method)
            r_big = reject_canonical_samples(norm, big, method)
            np.testing.assert_array_equal(r_base.survivor_mask, r_big.survivor_mask)
            np.testing.assert_array_equal(r_base.rejection_mask, r_big.rejection_mask)
            assert r_base.iterations_used == r_big.iterations_used
            assert r_base.degenerate_cell_count == r_big.degenerate_cell_count
            assert r_base.initial_sample_count == r_big.initial_sample_count
            assert r_base.surviving_sample_count == r_big.surviving_sample_count
            assert r_base.rejected_sample_count == r_big.rejected_sample_count
            assert r_base.rejected_fraction == r_big.rejected_fraction
            assert r_base.low_n_cell_count == r_big.low_n_cell_count


# ---------------------------------------------------------------------------
# 4. kappa_sigma analytical corpora
# ---------------------------------------------------------------------------

class TestKappaSigma:
    def test_defaults_reject_gross_outlier(self):
        norm = _values_corpus([10.0] * 9 + [100.0])
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "kappa_sigma")
        assert res.survivor_mask[:, 0, 0, 0].tolist() == [1] * 9 + [0]
        assert res.rejected_sample_count == 1
        assert res.surviving_sample_count == 9
        assert res.iterations_used == 2
        assert res.degenerate_cell_count == 1  # final all-equal survivor cell

    def test_asymmetric_low_high(self):
        # A negative outlier is only rejected under a tight sigma_low, and a
        # positive outlier only under a tight sigma_high — asymmetric bounds.
        norm = _values_corpus([0.0] * 6 + [-100.0, 100.0])
        weight = _weighted(norm, "none")
        tight_low = reject_canonical_samples(norm, weight, "kappa_sigma", sigma_low=0.5, sigma_high=3.0)
        tight_high = reject_canonical_samples(norm, weight, "kappa_sigma", sigma_low=3.0, sigma_high=0.5)
        symmetric = reject_canonical_samples(norm, weight, "kappa_sigma", sigma_low=3.0, sigma_high=3.0)
        assert not symmetric.rejection_mask.any()  # wide symmetric keeps both
        # tight low rejects only the -100 (frame 6)
        assert not tight_low.survivor_mask[6, 0, 0, 0]
        assert tight_low.survivor_mask[7, 0, 0, 0]
        # tight high rejects only the +100 (frame 7)
        assert tight_high.survivor_mask[6, 0, 0, 0]
        assert not tight_high.survivor_mask[7, 0, 0, 0]

    def test_multi_iteration_discriminating(self):
        norm = _values_corpus([0.0] * 8 + [40.0, 80.0])
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "kappa_sigma")
        assert res.survivor_mask[:, 0, 0, 0].tolist() == [1] * 8 + [0, 0]
        assert res.rejected_sample_count == 2
        assert res.iterations_used == 3

    def test_exact_stable_no_rejection(self):
        # all within bounds -> stable on the first pass, nothing rejected
        norm = _values_corpus([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "kappa_sigma")
        assert not res.rejection_mask.any()
        assert res.survivor_mask.sum() == res.initial_sample_count
        assert res.iterations_used == 1

    def test_low_n_unchanged(self):
        for n_vals in ([5.0], [5.0, 6.0]):
            norm = _values_corpus(n_vals)
            weight = _weighted(norm, "none")
            res = reject_canonical_samples(norm, weight, "kappa_sigma")
            assert not res.rejection_mask.any()
            assert res.low_n_cell_count == 1
            assert res.surviving_sample_count == len(n_vals)

    def test_channel_specific_rgb_rejection(self):
        r = [10.0] * 9 + [100.0]   # outlier in R (frame 9)
        g = [10.0] * 10
        b = [10.0] * 10
        norm = _rgb_values_corpus(r, g, b)
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "kappa_sigma")
        # R channel rejects frame 9; G and B keep every frame
        assert not res.survivor_mask[9, 0, 0, 0]
        assert res.survivor_mask[9, 0, 0, 1] and res.survivor_mask[9, 0, 0, 2]
        assert res.rejection_mask[:, 0, 0, 0].tolist() == [0] * 9 + [1]
        assert not res.rejection_mask[:, 0, 0, 1].any()
        assert not res.rejection_mask[:, 0, 0, 2].any()

    def test_degenerate_equal_samples_survive(self):
        norm = _values_corpus([7.0] * 5)
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "kappa_sigma")
        assert not res.rejection_mask.any()
        assert res.degenerate_cell_count == 1
        assert res.iterations_used == 1

    def test_matches_independent_reference(self):
        cases = [
            ([10.0] * 9 + [100.0], {}),
            ([0.0] * 8 + [40.0, 80.0], {}),
            ([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 50.0], {"sigma_low": 2.0, "sigma_high": 2.0}),
            ([0.0] * 6 + [-100.0, 100.0], {"sigma_low": 0.5, "sigma_high": 0.5}),
            ([5.0, 5.0, 5.0, 5.0, 9.0, 11.0], {}),
        ]
        for values, kw in cases:
            norm = _values_corpus(values)
            weight = _weighted(norm, "none")
            res = reject_canonical_samples(norm, weight, "kappa_sigma", **kw)
            images64 = norm.images.astype(np.float64)
            initial = (
                weight.active_frames[:, None, None, None]
                & norm.valid_mask[..., None]
                & np.isfinite(images64)
            )
            ref = _reference_diagnostics(images64, initial, "kappa_sigma", **kw)
            np.testing.assert_array_equal(res.survivor_mask, ref["survivor"])
            assert res.iterations_used == ref["iterations_used"]
            assert res.degenerate_cell_count == ref["degenerate_cell_count"]


# ---------------------------------------------------------------------------
# 5. winsorized_sigma_clip analytical corpora
# ---------------------------------------------------------------------------

class TestWSC:
    def test_defaults_reject_gross_outlier(self):
        norm = _values_corpus([10.0] * 9 + [100.0])
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "winsorized_sigma_clip")
        assert res.survivor_mask[:, 0, 0, 0].tolist() == [1] * 9 + [0]
        assert res.rejected_sample_count == 1

    def test_single_impulse_sigma_25_rejected(self):
        norm = _values_corpus([0.0, 0.0, 0.0, 0.0, 100.0])
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(
            norm, weight, "winsorized_sigma_clip", sigma_low=2.5, sigma_high=2.5
        )
        assert res.survivor_mask[:, 0, 0, 0].tolist() == [1, 1, 1, 1, 0]
        assert res.rejected_sample_count == 1
        assert res.iterations_used == 2
        assert res.degenerate_cell_count == 1

    def test_winsor_limits_matter(self):
        # Same corpus, same sigma: winsor 0.0 keeps everything; winsor 0.25
        # clips the extreme tails and rejects them via the tighter sigma band.
        norm = _values_corpus([0.0] * 8 + [-100.0, 100.0])
        weight = _weighted(norm, "none")
        no_winsor = reject_canonical_samples(
            norm, weight, "winsorized_sigma_clip",
            winsor_limit_low=0.0, winsor_limit_high=0.0,
        )
        win = reject_canonical_samples(
            norm, weight, "winsorized_sigma_clip",
            winsor_limit_low=0.25, winsor_limit_high=0.25,
        )
        assert not no_winsor.rejection_mask.any()
        assert win.rejected_sample_count >= 2

    def test_sigma_limits_matter(self):
        norm = _values_corpus([0.0] * 8 + [-100.0, 100.0])
        weight = _weighted(norm, "none")
        wide = reject_canonical_samples(
            norm, weight, "winsorized_sigma_clip", sigma_low=10.0, sigma_high=10.0
        )
        tight = reject_canonical_samples(
            norm, weight, "winsorized_sigma_clip", sigma_low=1.0, sigma_high=1.0
        )
        assert not wide.rejection_mask.any()
        assert tight.rejected_sample_count >= 2

    def test_final_mask_on_originals_not_replacements(self):
        # The 100 is winsorized to 59.5 for statistics, but the final mask
        # judges the ORIGINAL 100 (outside the band) -> rejected.
        norm = _values_corpus([10.0] * 9 + [100.0])
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "winsorized_sigma_clip")
        assert not res.survivor_mask[9, 0, 0, 0]
        assert res.rejection_mask[9, 0, 0, 0]
        # surviving samples remain the ORIGINAL normalized values (never replaced)
        assert float(norm.images[0, 0, 0, 0]) == 10.0
        assert float(norm.images[9, 0, 0, 0]) == 100.0

    def test_multi_iteration(self):
        norm = _values_corpus([0.0] * 8 + [40.0, 80.0])
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "winsorized_sigma_clip")
        assert res.survivor_mask[:, 0, 0, 0].tolist() == [1] * 8 + [0, 0]
        assert res.rejected_sample_count == 2
        assert res.iterations_used >= 2

    def test_asymmetric_winsor_limits(self):
        # A tight low winsor collapses the low tail, a tight high winsor the
        # high tail; asymmetric limits produce asymmetric rejection.
        norm = _values_corpus([0.0] * 6 + [-100.0, 100.0])
        weight = _weighted(norm, "none")
        low_only = reject_canonical_samples(
            norm, weight, "winsorized_sigma_clip",
            winsor_limit_low=0.3, winsor_limit_high=0.0,
        )
        high_only = reject_canonical_samples(
            norm, weight, "winsorized_sigma_clip",
            winsor_limit_low=0.0, winsor_limit_high=0.3,
        )
        assert not low_only.survivor_mask[6, 0, 0, 0]   # -100 rejected
        assert low_only.survivor_mask[7, 0, 0, 0]       # +100 kept
        assert high_only.survivor_mask[6, 0, 0, 0]      # -100 kept
        assert not high_only.survivor_mask[7, 0, 0, 0]  # +100 rejected

    def test_low_n_and_degenerate(self):
        # N=2 -> frozen, no rejection
        norm = _values_corpus([5.0, 6.0])
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "winsorized_sigma_clip")
        assert not res.rejection_mask.any()
        assert res.low_n_cell_count == 1
        # all-equal -> degenerate winsorized std == 0, all survive
        norm = _values_corpus([7.0] * 5)
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "winsorized_sigma_clip")
        assert not res.rejection_mask.any()
        assert res.degenerate_cell_count == 1

    def test_channel_specific_rgb(self):
        r = [10.0] * 9 + [100.0]
        g = [10.0] * 10
        b = [10.0] * 10
        norm = _rgb_values_corpus(r, g, b)
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "winsorized_sigma_clip")
        assert not res.survivor_mask[9, 0, 0, 0]
        assert res.survivor_mask[9, 0, 0, 1] and res.survivor_mask[9, 0, 0, 2]

    def test_matches_independent_reference(self):
        cases = [
            ([10.0] * 9 + [100.0], {}),
            ([0.0, 0.0, 0.0, 0.0, 100.0], {"sigma_low": 2.5, "sigma_high": 2.5}),
            ([0.0] * 8 + [-100.0, 100.0], {"winsor_limit_low": 0.25, "winsor_limit_high": 0.25}),
            ([0.0] * 6 + [-100.0, 100.0], {"winsor_limit_low": 0.3, "winsor_limit_high": 0.0}),
            ([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 50.0], {"sigma_low": 2.0, "sigma_high": 2.0}),
        ]
        for values, kw in cases:
            norm = _values_corpus(values)
            weight = _weighted(norm, "none")
            res = reject_canonical_samples(norm, weight, "winsorized_sigma_clip", **kw)
            images64 = norm.images.astype(np.float64)
            initial = (
                weight.active_frames[:, None, None, None]
                & norm.valid_mask[..., None]
                & np.isfinite(images64)
            )
            ref = _reference_diagnostics(images64, initial, "winsorized_sigma_clip", **kw)
            np.testing.assert_array_equal(res.survivor_mask, ref["survivor"])
            assert res.iterations_used == ref["iterations_used"]
            assert res.degenerate_cell_count == ref["degenerate_cell_count"]


# ---------------------------------------------------------------------------
# 6. Distinguish canonical WSC from legacy simplified / percentile meanings
# ---------------------------------------------------------------------------

class TestDistinguishFromLegacy:
    def test_wsc_differs_from_legacy_simplified_and_percentile(self):
        # Single impulse outlier. Canonical WSC (winsor + sigma, mean center)
        # rejects it; the legacy simplified median/σ clip (reconstructed here as
        # kappa_sigma) keeps it; the legacy global-coadd percentile clip only
        # REPLACES it (no rejection). Independent reconstruction, labelled, no
        # legacy-helper import.
        values = [0.0, 0.0, 0.0, 0.0, 100.0]
        norm = _values_corpus(values)
        weight = _weighted(norm, "none")

        wsc = reject_canonical_samples(
            norm, weight, "winsorized_sigma_clip", sigma_low=2.5, sigma_high=2.5
        )
        assert wsc.survivor_mask[:, 0, 0, 0].tolist() == [1, 1, 1, 1, 0]
        assert wsc.rejected_sample_count == 1

        # legacy simplified winsorized == median/σ clip (reconstructed: kappa_sigma)
        simplified = reject_canonical_samples(
            norm, weight, "kappa_sigma", sigma_low=2.5, sigma_high=2.5
        )
        assert simplified.rejected_sample_count == 0
        assert bool(np.all(simplified.survivor_mask))

        # legacy global-coadd percentile clip: replaces, never rejects
        arr = np.asarray(values, dtype=np.float64)
        q_low = float(np.quantile(arr, 0.05, method="linear"))
        q_high = float(np.quantile(arr, 0.95, method="linear"))
        clipped = np.clip(arr, q_low, q_high)
        assert clipped[4] == pytest.approx(80.0)   # value replaced to the bound
        assert clipped[4] != pytest.approx(100.0)  # not the original
        # canonical WSC actually marks the ORIGINAL sample rejected
        assert wsc.rejection_mask[4, 0, 0, 0]


# ---------------------------------------------------------------------------
# 7. Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_exact_counts_and_fraction(self):
        norm = _values_corpus([10.0] * 9 + [100.0])
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "kappa_sigma")
        assert res.initial_sample_count == 10
        assert res.surviving_sample_count == 9
        assert res.rejected_sample_count == 1
        assert res.rejected_fraction == pytest.approx(0.1)
        assert res.low_n_cell_count == 0  # N=10 >= 3
        assert isinstance(res.rejected_fraction, float)

    def test_low_n_cell_count_counts_once_including_zero(self):
        # Build a 2-column corpus: column 0 has 10 valid frames, column 1 only 2
        # valid frames (invalid for frames 2..9) -> its initial count is 2 < 3.
        arrays = []
        for i in range(10):
            col = np.ones((1, 2), dtype=np.float32)
            if i >= 2:
                col[0, 1] = np.nan  # column 1 invalid for frames 2..9
            arrays.append(col)
        norm = _norm(arrays)
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "kappa_sigma")
        assert res.low_n_cell_count == 1  # only column 1 has initial count 2 < 3

    def test_degenerate_cell_count_counted_once_not_per_iteration(self):
        # all-equal values: degenerate every executed iteration but counted once
        norm = _values_corpus([7.0] * 5)
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "kappa_sigma")
        assert res.degenerate_cell_count == 1

    def test_none_diagnostics_zero(self):
        norm = _values_corpus([10.0] * 9 + [100.0])
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "none")
        assert res.iterations_used == 0
        assert res.degenerate_cell_count == 0
        assert res.rejected_sample_count == 0

    def test_deterministic_python_scalar_types(self):
        norm = _values_corpus([10.0] * 9 + [100.0])
        weight = _weighted(norm, "none")
        res = reject_canonical_samples(norm, weight, "winsorized_sigma_clip")
        assert isinstance(res.iterations_used, int)
        assert isinstance(res.initial_sample_count, int)
        assert isinstance(res.surviving_sample_count, int)
        assert isinstance(res.rejected_sample_count, int)
        assert isinstance(res.low_n_cell_count, int)
        assert isinstance(res.degenerate_cell_count, int)
        assert isinstance(res.rejected_fraction, float)
        # not numpy scalars
        assert type(res.iterations_used) is int
        assert type(res.rejected_fraction) is float


# ---------------------------------------------------------------------------
# 8. Determinism / input immutability
# ---------------------------------------------------------------------------

class TestDeterminismOwnership:
    def test_repeat_run_deterministic(self):
        norm = _values_corpus([10.0] * 9 + [100.0])
        weight = _weighted(norm, "none")
        for method in ("none", "kappa_sigma", "winsorized_sigma_clip"):
            r1 = reject_canonical_samples(norm, weight, method)
            r2 = reject_canonical_samples(norm, weight, method)
            np.testing.assert_array_equal(r1.survivor_mask, r2.survivor_mask)
            np.testing.assert_array_equal(r1.rejection_mask, r2.rejection_mask)
            np.testing.assert_array_equal(r1.active_frames, r2.active_frames)
            assert r1.iterations_used == r2.iterations_used
            assert r1.degenerate_cell_count == r2.degenerate_cell_count
            assert r1.rejected_fraction == r2.rejected_fraction

    def test_inputs_unchanged_after_rejection(self):
        norm = _values_corpus([10.0] * 9 + [100.0])
        weight = _weighted(norm, "none")
        img_before = norm.images.copy()
        mask_before = norm.valid_mask.copy()
        active_before = weight.active_frames.copy()
        w_before = weight.weights.copy()
        reject_canonical_samples(norm, weight, "winsorized_sigma_clip")
        assert np.array_equal(norm.images, img_before)
        assert np.array_equal(norm.valid_mask, mask_before)
        assert np.array_equal(weight.active_frames, active_before)
        assert np.array_equal(weight.weights, w_before)
