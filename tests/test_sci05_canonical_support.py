"""SCI-05 Gate E1 — canonical positive support + footprint taper (deterministic).

Deterministic, hermetic witnesses for the donor-exact coverage-support primitives
in ``zemosaic.core.canonical_support``:

* ``make_footprint_taper`` (footprint-following feather, EDT primary + chamfer
  fallback, never radial);
* ``PositiveSupportAccumulator`` / ``accumulate_support_pair`` (atomic
  ``SUP_W1``/``SUP_W2``, derived ``N_eff``);
* ``build_canonical_estimator_weights`` (``w = q*m*a`` glue consumable by the C2
  ``combine_canonical_samples``).

No randomness with global seeds, no network, no filesystem, no GPU, no ZSSS
import. The only ZSSS-donor dependency that is exercised is the local source-port.
"""

from __future__ import annotations

import numpy as np
import pytest

import zemosaic.core.canonical_support as cs
from zemosaic.core.canonical_support import (
    PositiveSupportAccumulator,
    accumulate_support_pair,
    build_canonical_estimator_weights,
    make_footprint_taper,
)


# ---------------------------------------------------------------------------
# A. Footprint taper
# ---------------------------------------------------------------------------

class TestFootprintTaper:
    def test_validation_mask_bool_2d(self):
        with pytest.raises(ValueError):
            make_footprint_taper(np.ones((4, 4)))  # float, not bool
        with pytest.raises(ValueError):
            make_footprint_taper(np.ones((4, 4, 1), dtype=bool))  # 3-D
        with pytest.raises(ValueError):
            make_footprint_taper(np.ones(4, dtype=bool))  # 1-D

    def test_validation_floor_and_feather(self):
        m = np.ones((4, 4), dtype=bool)
        for bad_floor in (1.0, 1.5, -0.1, np.nan, np.inf):
            with pytest.raises(ValueError):
                make_footprint_taper(m, floor=bad_floor)
        for bad_px in (0.0, -1.0, np.nan, np.inf):
            with pytest.raises(ValueError):
                make_footprint_taper(m, feather_px=bad_px)
        # boundary floor=0.9999 is valid
        t = make_footprint_taper(m, floor=0.0)
        assert t.dtype == np.float32

    def test_empty_mask_returns_zeros(self):
        t = make_footprint_taper(np.zeros((5, 5), dtype=bool))
        assert t.shape == (5, 5)
        assert t.dtype == np.float32
        assert not t.any()

    def test_full_mask_interior_one(self):
        h, w = 40, 40
        m = np.ones((h, w), dtype=bool)
        t = make_footprint_taper(m, feather_px=8.0, floor=0.0)
        assert float(t[20, 20]) == pytest.approx(1.0)  # deep interior
        assert 0.0 < float(t[0, 20]) < 1.0  # boundary edge ramps (inside mask)

    def test_outside_zero(self):
        m = np.zeros((20, 20), dtype=bool)
        m[6:14, 6:14] = True
        t = make_footprint_taper(m)
        assert not t[~m].any()  # exactly zero outside the footprint

    def test_ramp_monotone_and_bounded(self):
        # Along a straight line from the interior to the boundary, the taper must
        # be monotonically non-increasing toward the boundary and within [floor, 1].
        h, w = 40, 40
        m = np.ones((h, w), dtype=bool)
        t = make_footprint_taper(m, feather_px=8.0, floor=0.2)
        assert t.min() >= 0.2 - 1e-6
        assert t.max() <= 1.0
        row = t[20, :]
        # non-decreasing from the left edge to the centre
        for j in range(20):
            assert row[j] <= row[j + 1] + 1e-6
        # non-increasing from the centre to the right edge
        for j in range(20, w - 1):
            assert row[j] >= row[j + 1] - 1e-6

    def test_floor_plateau(self):
        h, w = 40, 40
        m = np.ones((h, w), dtype=bool)
        floor = 0.35
        t = make_footprint_taper(m, feather_px=8.0, floor=floor)
        assert float(t.max()) == pytest.approx(1.0)  # deep interior
        assert float(t.min()) >= floor - 1e-6  # never below floor
        assert float(t.min()) < 1.0  # ramp, not constant
        # boundary value is the exact ramp formula at distance 1
        expected = floor + (1.0 - floor) * (1.0 / 8.0)
        assert float(t[0, 0]) == pytest.approx(expected, abs=1e-6)

    def test_padded_boundary_symmetry(self):
        # A footprint filling the whole array feathers symmetrically on all four
        # sides (the array exterior is a support boundary).
        h, w = 40, 41  # odd width to catch off-by-one
        m = np.ones((h, w), dtype=bool)
        t = make_footprint_taper(m, feather_px=8.0)
        np.testing.assert_allclose(t, t[::-1, :], atol=1e-6)  # vertical symmetry
        np.testing.assert_allclose(t, t[:, ::-1], atol=1e-6)  # horizontal symmetry

    def test_translation_invariance_up_to_rasterization(self):
        # A small footprint deep inside a large array gives the same taper wherever
        # it is translated (footprint-relative, never image-centre-relative).
        big = 60
        foot = np.ones((8, 8), dtype=bool)
        def place(r0, c0):
            m = np.zeros((big, big), dtype=bool)
            m[r0:r0 + 8, c0:c0 + 8] = foot
            return make_footprint_taper(m, feather_px=3.0)

        tA = place(10, 10)
        tB = place(35, 30)
        # extract the local neighbourhoods (interior region, away from array edges)
        np.testing.assert_allclose(tA[10:18, 10:18], tB[35:43, 30:38], atol=1e-6)

    def test_rotation_invariance_90(self):
        m = np.zeros((40, 40), dtype=bool)
        m[8:20, 12:28] = True
        m[20:30, 14:24] = True  # asymmetric footprint
        t_orig = make_footprint_taper(m, feather_px=6.0)
        t_rot = make_footprint_taper(np.rot90(m), feather_px=6.0)
        np.testing.assert_allclose(t_rot, np.rot90(t_orig), atol=1e-6)

    def test_irregular_holey_footprint(self):
        m = np.zeros((30, 30), dtype=bool)
        m[3:27, 3:27] = True
        m[10:13, 10:13] = False  # interior hole
        m[20:22, 20:26] = False  # notch
        t = make_footprint_taper(m, feather_px=3.0)
        assert not t[~m].any()  # zero outside
        assert t[m].min() >= 0.0
        assert t[m].max() <= 1.0
        # the interior hole boundary also ramps: a pixel adjacent to the hole
        # (inside the footprint) has taper < 1
        assert float(t[9, 11]) < 1.0
        # deep interior (far from every boundary and hole) reaches 1.0
        assert float(t[17, 15]) == pytest.approx(1.0)

    def test_chamfer_fallback_matches_edt_and_never_radial(self, monkeypatch):
        # Force the chamfer fallback and compare against the EDT path.
        m = np.zeros((40, 40), dtype=bool)
        m[4:36, 4:36] = True
        edt_taper = make_footprint_taper(m, feather_px=8.0)
        monkeypatch.setattr(cs, "_distance_transform_edt", None)
        chamfer_taper = make_footprint_taper(m, feather_px=8.0)
        monkeypatch.undo()

        # same shape/dtype/invariants
        assert chamfer_taper.shape == edt_taper.shape
        assert chamfer_taper.dtype == np.float32
        assert not chamfer_taper[~m].any()  # zero outside
        assert chamfer_taper[m].min() >= 0.0 and chamfer_taper[m].max() <= 1.0
        # matches EDT within the documented chamfer rasterization tolerance
        np.testing.assert_allclose(chamfer_taper, edt_taper, atol=0.30)
        # deep interior still reaches 1.0 (footprint-following, not radial falloff)
        assert float(chamfer_taper[20, 20]) == pytest.approx(1.0)

    def test_never_radial_off_center_footprint(self):
        # An off-centre footprint: the taper's maximum lies within the footprint,
        # not at the image centre (a radial map would peak at the centre).
        m = np.zeros((40, 40), dtype=bool)
        m[2:12, 2:12] = True  # top-left footprint
        t = make_footprint_taper(m, feather_px=4.0)
        # centre of the image is far from the footprint -> zero taper there
        assert float(t[20, 20]) == 0.0
        # centre of the footprint reaches 1.0 (deep interior)
        assert float(t[7, 7]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# B. Positive-support accumulator
# ---------------------------------------------------------------------------

class TestAccumulator:
    def test_exact_w1_w2_known_weights(self):
        acc = PositiveSupportAccumulator((2, 2))
        s0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        s1 = np.array([[0.5, 1.5], [2.5, 3.5]])
        acc.add(s0)
        acc.add(s1)
        np.testing.assert_array_equal(acc.support_w1, s0 + s1)
        np.testing.assert_array_equal(acc.support_w2, s0 * s0 + s1 * s1)

    def test_n_eff_exact_first(self):
        acc = PositiveSupportAccumulator((2, 2))
        s0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        s1 = np.array([[0.5, 1.5], [2.5, 3.5]])
        acc.add(s0)
        acc.add(s1)
        w1 = s0 + s1
        w2 = s0 * s0 + s1 * s1
        expected = w1 * w1 / w2
        np.testing.assert_array_equal(acc.n_eff_support, expected)  # bit-exact

    def test_n_eff_neutral_zero(self):
        acc = PositiveSupportAccumulator((2, 2))
        np.testing.assert_array_equal(acc.n_eff_support, np.zeros((2, 2)))

    def test_n_eff_does_not_mutate_and_owned_copy(self):
        acc = PositiveSupportAccumulator((2, 2))
        acc.add(np.ones((2, 2)))
        w1_before = acc.support_w1
        w2_before = acc.support_w2
        n = acc.n_eff_support
        n[0, 0] = 999.0  # mutating the returned view must not affect state
        assert acc.support_w1[0, 0] == w1_before[0, 0]
        assert acc.support_w2[0, 0] == w2_before[0, 0]
        assert acc.n_eff_support[0, 0] == 1.0  # still the original value
        assert n.flags["OWNDATA"]

    def test_owned_copies(self):
        acc = PositiveSupportAccumulator((2, 2))
        acc.add(np.ones((2, 2)))
        w1 = acc.support_w1
        w1[0, 0] = 123.0
        assert acc.support_w1[0, 0] == 1.0  # not aliased

    def test_float64_default_and_dtype_validation(self):
        acc = PositiveSupportAccumulator((2, 2))
        assert acc.dtype == np.dtype(np.float64)
        assert acc.support_w1.dtype == np.float64
        # float32 supported
        acc32 = PositiveSupportAccumulator((2, 2), dtype=np.float32)
        acc32.add(np.ones((2, 2), dtype=np.float32))
        assert acc32.support_w1.dtype == np.float32
        # invalid dtypes rejected
        for bad in (np.int64, np.float16, np.int32):
            with pytest.raises(TypeError):
                PositiveSupportAccumulator((2, 2), dtype=bad)

    def test_shape_validation(self):
        for bad in ((), (5,), (0, 5), (5, -1), (2, 2, 3)):
            with pytest.raises(ValueError):
                PositiveSupportAccumulator(bad)

    def test_atomic_fail_before_mutation_negative(self):
        acc = PositiveSupportAccumulator((2, 2))
        acc.add(np.ones((2, 2)))
        w1_before = acc.support_w1
        w2_before = acc.support_w2
        with pytest.raises(ValueError):
            acc.add(np.array([[1.0, 1.0], [1.0, -1.0]]))  # negative
        np.testing.assert_array_equal(acc.support_w1, w1_before)
        np.testing.assert_array_equal(acc.support_w2, w2_before)

    def test_atomic_fail_before_mutation_nonfinite(self):
        acc = PositiveSupportAccumulator((2, 2))
        acc.add(np.ones((2, 2)))
        w1_before = acc.support_w1
        w2_before = acc.support_w2
        for bad in (np.nan, np.inf):
            s = np.ones((2, 2))
            s[0, 0] = bad
            with pytest.raises(ValueError):
                acc.add(s)
        np.testing.assert_array_equal(acc.support_w1, w1_before)
        np.testing.assert_array_equal(acc.support_w2, w2_before)

    def test_atomic_fail_before_mutation_shape(self):
        acc = PositiveSupportAccumulator((2, 2))
        w1_before = acc.support_w1
        w2_before = acc.support_w2
        with pytest.raises(ValueError):
            acc.add(np.ones((3, 3)))
        np.testing.assert_array_equal(acc.support_w1, w1_before)
        np.testing.assert_array_equal(acc.support_w2, w2_before)

    def test_atomic_fail_before_mutation_square_overflow(self):
        acc = PositiveSupportAccumulator((1, 1))
        acc.add(np.ones((1, 1)))
        w1_before = acc.support_w1
        w2_before = acc.support_w2
        huge = np.array([[1e200]])  # finite but s**2 -> +Inf
        with pytest.raises(ValueError):
            acc.add(huge)
        np.testing.assert_array_equal(acc.support_w1, w1_before)
        np.testing.assert_array_equal(acc.support_w2, w2_before)

    def test_atomic_fail_before_mutation_cumulative_overflow(self):
        acc = PositiveSupportAccumulator((1, 1))
        s = np.array([[1e154]])
        acc.add(s)  # W1=1e154, W2=1e308 (both finite)
        w1_before = acc.support_w1
        w2_before = acc.support_w2
        with pytest.raises(ValueError):
            acc.add(s)  # would push W2 to 2e308 = +Inf
        np.testing.assert_array_equal(acc.support_w1, w1_before)
        np.testing.assert_array_equal(acc.support_w2, w2_before)

    def test_decomposition_invariance(self):
        rng = np.random.default_rng(12345)
        shape = (4, 4)
        n = 61
        supports = [rng.uniform(0.0, 1.0, size=shape) for _ in range(n)]

        def run_all():
            a = PositiveSupportAccumulator(shape)
            for s in supports:
                a.add(s)
            return a

        def run_partitioned(parts):
            a = PositiveSupportAccumulator(shape)
            i = 0
            for k in parts:
                for _ in range(k):
                    a.add(supports[i])
                    i += 1
            return a

        all_in_one = run_all()
        part_3_17_41 = run_partitioned([3, 17, 41])
        singletons = run_partitioned([1] * n)
        for other in (part_3_17_41, singletons):
            np.testing.assert_array_equal(all_in_one.support_w1, other.support_w1)
            np.testing.assert_array_equal(all_in_one.support_w2, other.support_w2)
            np.testing.assert_array_equal(all_in_one.n_eff_support, other.n_eff_support)

    def test_unit_weight_reduces_to_count(self):
        acc = PositiveSupportAccumulator((3, 3))
        n = 7
        for _ in range(n):
            acc.add(np.ones((3, 3)))
        assert np.all(acc.support_w1 == n)
        assert np.all(acc.support_w2 == n)
        assert np.all(acc.n_eff_support == n)

    def test_n_eff_overflow_resistant_fallback(self):
        acc = PositiveSupportAccumulator((1, 1))
        s = np.array([[7e153]])
        acc.add(s)
        acc.add(s)  # W1=1.4e154, W2=9.8e307; naive W1**2 would overflow
        assert np.all(np.isfinite(acc.support_w1))
        assert np.all(np.isfinite(acc.support_w2))
        n_eff = float(acc.n_eff_support[0, 0])
        assert np.isfinite(n_eff)
        assert np.isclose(n_eff, 2.0, rtol=1e-3, atol=1e-3)  # two equal supports -> 2

    def test_functional_accumulate_support_pair(self):
        w1 = np.zeros((2, 2), dtype=np.float64)
        w2 = np.zeros((2, 2), dtype=np.float64)
        accumulate_support_pair(w1, w2, np.ones((2, 2)))
        accumulate_support_pair(w1, w2, np.ones((2, 2)))
        np.testing.assert_array_equal(w1, np.full((2, 2), 2.0))
        np.testing.assert_array_equal(w2, np.full((2, 2), 2.0))
        # fail-before-mutation on the functional helper too
        w1b = w1.copy()
        w2b = w2.copy()
        with pytest.raises(ValueError):
            accumulate_support_pair(w1, w2, np.full((2, 2), -1.0))
        np.testing.assert_array_equal(w1, w1b)
        np.testing.assert_array_equal(w2, w2b)


# ---------------------------------------------------------------------------
# C. Estimator-weight builder
# ---------------------------------------------------------------------------

class TestBuilder:
    def _ctx(self, n=3, h=4, w=5):
        q = np.array([1.0, 0.5, 0.0])
        vm = np.ones((3, h, w), dtype=bool)
        vm[2] = False  # frame 2 inactive
        return q, vm

    def test_result_shape_dtype_owned(self):
        q, vm = self._ctx()
        w = build_canonical_estimator_weights(q, vm)
        assert w.shape == vm.shape
        assert w.dtype == np.float64
        assert w.flags["OWNDATA"] and w.flags["C_CONTIGUOUS"]

    def test_zero_outside_mask_and_inactive(self):
        q, vm = self._ctx()
        vm[0, 1, 1] = False  # hole in frame 0
        w = build_canonical_estimator_weights(q, vm)
        assert w[0, 1, 1] == 0.0  # outside mask
        assert not w[2].any()  # inactive frame all zero

    def test_bounded_and_le_q(self):
        q, vm = self._ctx()
        w = build_canonical_estimator_weights(q, vm)
        assert w.min() >= 0.0 and w.max() <= 1.0
        # w_i <= q_i per frame
        assert w[0].max() <= 1.0
        assert w[1].max() <= 0.5

    def test_equals_q_when_no_taper_full_mask(self):
        q, vm = self._ctx()
        w = build_canonical_estimator_weights(q, vm)
        # frame 0: q=1, m=1, a=1 -> w == 1; frame 1: q=0.5 -> w == 0.5
        np.testing.assert_array_equal(w[0], np.full(vm[0].shape, 1.0))
        np.testing.assert_array_equal(w[1], np.full(vm[1].shape, 0.5))

    def test_explicit_taper_applied(self):
        q, vm = self._ctx()
        taper = np.ones(vm.shape, dtype=np.float64)
        taper[0] = 0.25  # frame 0 taper = 0.25 everywhere inside mask
        w = build_canonical_estimator_weights(q, vm, taper=taper)
        np.testing.assert_array_equal(w[0], np.full(vm[0].shape, 0.25))  # q=1 * a=0.25
        np.testing.assert_array_equal(w[1], np.full(vm[1].shape, 0.5))  # q=0.5 * a=1
        # taper must still be zero outside mask
        vm2 = vm.copy()
        vm2[0, 0, 0] = False
        taper2 = np.ones(vm.shape)
        w2 = build_canonical_estimator_weights(q, vm2, taper=taper2)
        assert w2[0, 0, 0] == 0.0

    def test_per_frame_sequence_taper(self):
        q, vm = self._ctx()
        frames = [np.full(vm[i].shape, 0.5) for i in range(3)]
        w = build_canonical_estimator_weights(q, vm, taper=frames)
        np.testing.assert_array_equal(w[0], np.full(vm[0].shape, 0.5))
        np.testing.assert_array_equal(w[1], np.full(vm[1].shape, 0.25))  # 0.5*0.5

    def test_generated_taper_via_footprint(self):
        # Large full mask: interior reaches a=1, boundary ramps below 1.
        h, w = 40, 40
        q = np.array([1.0, 1.0])
        vm = np.ones((2, h, w), dtype=bool)
        wmap = build_canonical_estimator_weights(q, vm, taper="footprint", taper_px=8.0)
        expected = make_footprint_taper(vm[0]).astype(np.float64)
        np.testing.assert_array_equal(wmap[0], expected)  # q=1, m=1
        assert float(wmap[0, 20, 20]) == pytest.approx(1.0)  # deep interior
        assert float(wmap[0, 0, 0]) < 1.0  # boundary ramp

    def test_invalid_inputs_raise_before_build(self):
        from zemosaic.core.canonical_stacking import CanonicalStackValidationError
        q, vm = self._ctx()
        # bad q dtype/ndim/range
        with pytest.raises(CanonicalStackValidationError):
            build_canonical_estimator_weights(np.ones((3, 3)), vm)
        with pytest.raises(CanonicalStackValidationError):
            build_canonical_estimator_weights(np.array([1.0, 2.0, 0.0]), vm)  # >1
        # bad valid_mask
        with pytest.raises(CanonicalStackValidationError):
            build_canonical_estimator_weights(q, np.ones((3, 4, 5)))  # float mask
        with pytest.raises(CanonicalStackValidationError):
            build_canonical_estimator_weights(q, np.ones((2, 4, 5), dtype=bool))  # N mismatch
        # bad taper
        with pytest.raises(CanonicalStackValidationError):
            build_canonical_estimator_weights(q, vm, taper="radial")
        with pytest.raises(CanonicalStackValidationError):
            build_canonical_estimator_weights(q, vm, taper=np.full((3, 4, 5), 2.0))  # >1
        with pytest.raises(CanonicalStackValidationError):
            build_canonical_estimator_weights(q, vm, taper=np.ones((3, 3)))  # wrong shape
        # active_frames consistency
        with pytest.raises(CanonicalStackValidationError):
            build_canonical_estimator_weights(
                q, vm, active_frames=np.array([True, True, True])
            )  # frame 2 inactive but q==0 -> inconsistent

    def test_c2_integration_witness(self):
        from zemosaic.core.canonical_stacking import (
            combine_canonical_samples,
            compute_canonical_quality_weights,
            normalize_canonical_images,
            prepare_canonical_inputs,
            reject_canonical_samples,
        )
        arrays = [np.full((2, 2), float(v), dtype=np.float32) for v in (1.0, 2.0, 3.0)]
        batch = prepare_canonical_inputs(
            arrays, [np.ones((2, 2), dtype=bool)] * 3
        )
        norm = normalize_canonical_images(batch, "none")
        weight = compute_canonical_quality_weights(norm, "none")
        rej = reject_canonical_samples(norm, weight, "none")
        wmap = build_canonical_estimator_weights(weight.weights, norm.valid_mask)
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        # mean of [1, 2, 3] = 2.0, denominator = 1+1+1 = 3
        np.testing.assert_allclose(res.science, 2.0, rtol=1e-6)
        assert float(res.estimator_weight_sum[0, 0]) == pytest.approx(3.0)
        assert res.valid_mask.all()
        assert res.surviving_sample_count[0, 0] == 3

    def test_c2_integration_with_taper(self):
        from zemosaic.core.canonical_stacking import (
            combine_canonical_samples,
            compute_canonical_quality_weights,
            normalize_canonical_images,
            prepare_canonical_inputs,
            reject_canonical_samples,
        )
        # Constant field + taper: weighted mean still equals the constant value
        # (constant-field invariant holds even with a spatial taper).
        arrays = [np.full((40, 40), 5.0, dtype=np.float32) for _ in range(3)]
        batch = prepare_canonical_inputs(
            arrays, [np.ones((40, 40), dtype=bool)] * 3
        )
        norm = normalize_canonical_images(batch, "none")
        weight = compute_canonical_quality_weights(norm, "none")
        rej = reject_canonical_samples(norm, weight, "none")
        wmap = build_canonical_estimator_weights(
            weight.weights, norm.valid_mask, taper="footprint", taper_px=8.0
        )
        res = combine_canonical_samples(norm, weight, rej, wmap, "mean")
        np.testing.assert_allclose(res.science, 5.0, rtol=1e-6, atol=1e-6)
