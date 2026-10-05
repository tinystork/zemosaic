"""SCI-05 Gate E2 — canonical engine orchestration (deterministic witnesses).

Deterministic, hermetic tests for the assembled backend-neutral engine
``run_canonical_stack`` in ``zemosaic.core.canonical_engine``, wiring the accepted
B1 → B2 → E1 → C1 → C2 stages into one ``CanonicalStackRequest`` →
``CanonicalStackResult`` entry point with bounded provenance.

No randomness with global seeds, no network, no filesystem writes, no ZSSS import.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import zemosaic.core.canonical_stacking as cs
from zemosaic.core.canonical_engine import (
    CanonicalStackRequest,
    CanonicalStackResult,
    run_canonical_stack,
)
from zemosaic.core.canonical_support import (
    PositiveSupportAccumulator,
    build_canonical_estimator_weights,
)


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------

def _full_support(h, w):
    return np.ones((h, w), dtype=bool)


def _request(arrays, masks=None, **kw):
    if masks is None:
        masks = [_full_support(*a.shape[:2]) for a in arrays]
    defaults = dict(
        normalization="none", weighting="none", rejection="none", combine="mean",
        taper="none",
    )
    defaults.update(kw)
    return CanonicalStackRequest(images=arrays, geometric_support=masks, **defaults)


def _values_corpus(values):
    return [np.full((1, 1), float(v), dtype=np.float32) for v in values]


def _rgb_corpus(r, g, b):
    return [
        np.array([[[float(r[i]), float(g[i]), float(b[i])]]], dtype=np.float32)
        for i in range(len(r))
    ]


def _expected_support(arrays, masks, norm_method, weight_method, taper, taper_px, taper_floor):
    """Independently recompute the E1 support maps (same public stage functions)."""
    batch = cs.prepare_canonical_inputs(arrays, masks)
    norm = cs.normalize_canonical_images(batch, norm_method)
    weight = cs.compute_canonical_quality_weights(norm, weight_method)
    builder_taper = "footprint" if taper == "footprint" else None
    wmap = build_canonical_estimator_weights(
        weight.weights, norm.valid_mask,
        taper=builder_taper, taper_px=taper_px, taper_floor=taper_floor,
        active_frames=weight.active_frames,
    )
    acc = PositiveSupportAccumulator((norm.height, norm.width), dtype=np.float64)
    for i in range(norm.n_frames):
        if weight.active_frames[i]:
            acc.add(wmap[i])
    return acc.support_w1, acc.support_w2, acc.n_eff_support


# ---------------------------------------------------------------------------
# 1. End-to-end determinism
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_mean_and_median(self):
        for combine in ("mean", "median"):
            res = run_canonical_stack(_request(_values_corpus([1.0, 2.0, 3.0]), combine=combine))
            assert isinstance(res, CanonicalStackResult)
            np.testing.assert_allclose(res.science, 2.0, rtol=1e-6)
            assert res.science.dtype == np.float32
            assert res.valid_mask.all()

    def test_rejection_methods(self):
        # outlier corpus: kappa and WSC reject the 100, none keeps it
        arrays = _values_corpus([10.0] * 9 + [100.0])
        res_none = run_canonical_stack(_request(arrays, rejection="none", combine="mean"))
        res_kappa = run_canonical_stack(_request(arrays, rejection="kappa_sigma", combine="mean"))
        res_wsc = run_canonical_stack(_request(arrays, rejection="winsorized_sigma_clip", combine="mean"))
        # none keeps the outlier -> mean > 10
        assert float(res_none.science[0, 0]) > 10.0
        # kappa/wsc reject it -> mean == 10
        assert float(res_kappa.science[0, 0]) == pytest.approx(10.0, rel=1e-6)
        assert float(res_wsc.science[0, 0]) == pytest.approx(10.0, rel=1e-6)
        assert res_kappa.rejected_sample_count == 1
        assert res_wsc.rejected_sample_count == 1

    def test_taper_none_vs_footprint(self):
        arrays = [np.full((40, 40), 5.0, dtype=np.float32)] * 3
        res_none = run_canonical_stack(_request(arrays, taper="none"))
        res_fp = run_canonical_stack(_request(arrays, taper="footprint", taper_px=8.0))
        # constant field: both give constant science 5.0
        np.testing.assert_allclose(res_none.science, 5.0, rtol=1e-6)
        np.testing.assert_allclose(res_fp.science, 5.0, rtol=1e-6)
        # no-taper support_w1 == 3 everywhere (3 active frames, a=1)
        assert np.all(res_none.support_w1 == 3.0)
        # footprint taper support is < 3 near the boundary (a < 1)
        assert res_fp.support_w1[0, 0] < 3.0
        assert res_fp.support_w1[20, 20] == pytest.approx(3.0)

    def test_shape_restoration(self):
        # HW mono
        mono = run_canonical_stack(_request([np.ones((2, 3), dtype=np.float32)] * 3))
        assert mono.science.shape == (2, 3)
        # HWC1
        hwc1 = run_canonical_stack(_request([np.ones((2, 3, 1), dtype=np.float32)] * 3))
        assert hwc1.science.shape == (2, 3, 1)
        # RGB
        rgb = run_canonical_stack(_request([np.ones((2, 3, 3), dtype=np.float32)] * 3))
        assert rgb.science.shape == (2, 3, 3)
        assert rgb.support_w1.shape == (2, 3)  # support is always 2-D (H, W)

    def test_n1_identity(self):
        res = run_canonical_stack(_request(_values_corpus([7.0])))
        assert float(res.science[0, 0]) == pytest.approx(7.0)
        assert res.valid_mask.all()
        assert res.support_w1[0, 0] == pytest.approx(1.0)

    def test_all_invalid_cell(self):
        arrays = []
        for _ in range(3):
            img = np.ones((2, 2), dtype=np.float32)
            img[1, 1] = np.nan  # all-invalid cell
            arrays.append(img)
        res = run_canonical_stack(_request(arrays))
        assert not res.valid_mask[1, 1]
        assert np.isnan(res.science[1, 1])
        assert float(res.estimator_weight_sum[1, 1]) == 0.0
        assert float(res.support_w1[1, 1]) == 0.0  # no support at that pixel
        assert res.valid_mask[0, 0]  # valid pixels remain valid

    def test_mixed_validity(self):
        a = np.ones((3, 3), dtype=np.float32)
        b = np.ones((3, 3), dtype=np.float32)
        b[0, 0] = np.nan
        bad = np.full((3, 3), np.nan, dtype=np.float32)
        res = run_canonical_stack(_request([a, b, bad]))
        assert res.valid_mask[0, 0]  # a valid
        assert not res.valid_mask[0, 0] or res.science[0, 0] == pytest.approx(1.0)
        # frame 2 inactive -> support at valid pixel == 2 (two active frames)
        assert float(res.support_w1[1, 1]) == pytest.approx(2.0)

    def test_rgb_channel_specific_survivor(self):
        r = [10.0] * 9 + [100.0]
        g = [10.0] * 10
        b = [10.0] * 10
        res = run_canonical_stack(_request(_rgb_corpus(r, g, b), rejection="kappa_sigma", combine="mean"))
        assert float(res.science[0, 0, 0]) == pytest.approx(10.0)  # R: outlier rejected
        assert float(res.science[0, 0, 1]) == pytest.approx(10.0)  # G: all kept
        # R channel rejection is channel-specific
        assert res.rejection_mask[9, 0, 0, 0]  # R outlier rejected
        assert not res.rejection_mask[9, 0, 0, 1]  # G kept


# ---------------------------------------------------------------------------
# 2. Support maps
# ---------------------------------------------------------------------------

class TestSupport:
    def test_support_matches_independent_recompute(self):
        arrays = _values_corpus([1.0, 2.0, 3.0, 4.0])
        masks = [_full_support(1, 1)] * 4
        for taper in ("none", "footprint"):
            res = run_canonical_stack(_request(arrays, masks, taper=taper))
            w1, w2, n_eff = _expected_support(arrays, masks, "none", "none", taper, 8.0, 0.0)
            np.testing.assert_array_equal(res.support_w1, w1)
            np.testing.assert_array_equal(res.support_w2, w2)
            np.testing.assert_array_equal(res.n_eff_support, n_eff)

    def test_support_rejection_independent(self):
        arrays = _values_corpus([10.0] * 9 + [100.0])
        res_none = run_canonical_stack(_request(arrays, rejection="none"))
        res_kappa = run_canonical_stack(_request(arrays, rejection="kappa_sigma"))
        res_wsc = run_canonical_stack(_request(arrays, rejection="winsorized_sigma_clip"))
        for other in (res_kappa, res_wsc):
            np.testing.assert_array_equal(res_none.support_w1, other.support_w1)
            np.testing.assert_array_equal(res_none.support_w2, other.support_w2)
            np.testing.assert_array_equal(res_none.n_eff_support, other.n_eff_support)

    def test_inactive_frames_contribute_zero(self):
        a = np.ones((1, 1), dtype=np.float32)
        bad = np.full((1, 1), np.nan, dtype=np.float32)
        res = run_canonical_stack(_request([a, a, bad]))
        # 2 active frames, 1 inactive -> support_w1 == 2 (not 3)
        assert float(res.support_w1[0, 0]) == pytest.approx(2.0)
        assert float(res.support_w2[0, 0]) == pytest.approx(2.0)
        assert float(res.n_eff_support[0, 0]) == pytest.approx(2.0)
        # excluded frame recorded
        assert any(e[1] == "normalization" for e in res.provenance["excluded_frames"])

    def test_unit_weight_count_case(self):
        arrays = [np.ones((3, 3), dtype=np.float32)] * 5
        res = run_canonical_stack(_request(arrays, taper="none"))
        # unit weight (q=1, a=1, m=1): SUP_W1 == SUP_W2 == n == 5, N_eff == 5
        assert np.all(res.support_w1 == 5.0)
        assert np.all(res.support_w2 == 5.0)
        assert np.all(res.n_eff_support == 5.0)


# ---------------------------------------------------------------------------
# 3. Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_requested_effective_and_json_serializable(self):
        res = run_canonical_stack(_request(_values_corpus([1.0, 2.0, 3.0]), combine="median"))
        prov = res.provenance
        for stage in ("normalization", "weighting", "rejection", "combine"):
            assert prov[stage]["requested"] == prov[stage]["effective"]
        assert prov["combine"]["effective"] == "median"
        # per-stage backend: B1/B2/support on CPU, C1/C2 on the requested backend
        assert prov["backend"] == {
            "requested": "cpu",
            "normalization": "cpu",
            "weighting": "cpu",
            "support": "cpu",
            "rejection": "cpu",
            "combine": "cpu",
        }
        assert prov["effective_event"]["backend_requested"] == "cpu"
        assert prov["effective_event"]["backend_normalization"] == "cpu"
        assert prov["effective_event"]["backend_weighting"] == "cpu"
        assert prov["effective_event"]["backend_support"] == "cpu"
        assert prov["effective_event"]["backend_rejection"] == "cpu"
        assert prov["effective_event"]["backend_combine"] == "cpu"
        assert prov["taper"]["kind"] == "none"
        assert prov["equalize_rgb"] == {"requested": False, "applied": False}
        # JSON-serializable (no numpy scalars/arrays)
        json.dumps(prov)  # must not raise

    def test_reference_auto_vs_explicit(self):
        arrays = _values_corpus([1.0, 2.0, 3.0])
        auto = run_canonical_stack(_request(arrays))
        explicit = run_canonical_stack(_request(arrays, reference_index=1))
        assert auto.provenance["reference"] == {"mode": "auto", "index": 0}
        assert explicit.provenance["reference"] == {"mode": "explicit", "index": 1}
        assert auto.provenance["effective_event"]["reference_index"] == 0

    def test_excluded_frames_recorded(self):
        a = np.ones((1, 1), dtype=np.float32)
        bad = np.full((1, 1), np.nan, dtype=np.float32)
        res = run_canonical_stack(_request([a, a, bad]))
        excl = res.provenance["excluded_frames"]
        assert len(excl) == 1
        idx, stage, reason = excl[0]
        assert idx == 2 and stage == "normalization" and reason == "zero_valid_support"

    def test_effective_event_bounded(self):
        res = run_canonical_stack(_request(_values_corpus([1.0, 2.0, 3.0])))
        ev = res.provenance["effective_event"]
        # all values are JSON-serializable scalars/strings/lists (no numpy
        # arrays/scalars); the non-requested equalizer decision is None.
        for v in ev.values():
            assert isinstance(v, (str, int, float, bool, list, type(None)))
        json.dumps(ev)  # must not raise
        json.dumps(ev)


# ---------------------------------------------------------------------------
# 4. Validation / error paths
# ---------------------------------------------------------------------------

class TestValidation:
    def test_unknown_tokens(self):
        base = _values_corpus([1.0, 2.0, 3.0])
        for field in ("normalization", "weighting", "rejection", "combine"):
            with pytest.raises(cs.CanonicalStackValidationError):
                run_canonical_stack(_request(base, **{field: "bogus"}))

    def test_bad_taper(self):
        with pytest.raises(cs.CanonicalStackValidationError):
            run_canonical_stack(_request(_values_corpus([1.0, 2.0]), taper="radial"))

    def test_bad_request_type(self):
        with pytest.raises(cs.CanonicalStackValidationError):
            run_canonical_stack("not a request")

    def test_mismatched_inputs(self):
        # inconsistent spatial shapes
        arrays = [np.ones((2, 2), dtype=np.float32), np.ones((3, 3), dtype=np.float32)]
        with pytest.raises(cs.CanonicalStackValidationError):
            run_canonical_stack(_request(arrays))
        # geometric_support length mismatch
        arrays = [np.ones((2, 2), dtype=np.float32)] * 3
        with pytest.raises(cs.CanonicalStackValidationError):
            run_canonical_stack(_request(arrays, masks=[np.ones((2, 2), bool)] * 2))

    def _rgb_frames(self, n=3, h=100, w=100):
        rng = np.random.default_rng(0)
        offsets = np.array([10.0, 12.0, 14.0], dtype=np.float32)
        return [
            (rng.normal(0, 1, (h, w, 3)).astype(np.float32) + offsets)
            for _ in range(n)
        ]

    def test_equalize_rgb_false_no_equalizer(self):
        arrays = self._rgb_frames()
        res = run_canonical_stack(_request(arrays, combine="mean", equalize_rgb=False))
        assert res.provenance["equalize_rgb"] == {"requested": False, "applied": False}
        assert res.provenance["effective_event"]["equalize_rgb_applied"] is False

    def test_equalize_rgb_changes_only_science(self):
        arrays = self._rgb_frames()
        off = run_canonical_stack(_request(arrays, combine="mean", equalize_rgb=False))
        on = run_canonical_stack(_request(arrays, combine="mean", equalize_rgb=True))
        prov = on.provenance["equalize_rgb"]
        assert prov["requested"] is True
        assert prov["applied"] is True
        assert prov["decision"] == "applied"
        assert len(prov["clipped_gains"]) == 3
        # only science changes; everything else stays bit-identical
        assert not np.array_equal(on.science, off.science)
        np.testing.assert_array_equal(on.estimator_weight_sum, off.estimator_weight_sum)
        np.testing.assert_array_equal(on.valid_mask, off.valid_mask)
        np.testing.assert_array_equal(on.surviving_sample_count, off.surviving_sample_count)
        np.testing.assert_array_equal(on.support_w1, off.support_w1)
        np.testing.assert_array_equal(on.support_w2, off.support_w2)
        np.testing.assert_array_equal(on.n_eff_support, off.n_eff_support)
        # effective_event mirrors it
        ev = on.provenance["effective_event"]
        assert ev["equalize_rgb_applied"] is True
        assert ev["equalize_rgb_decision"] == "applied"
        assert ev["equalize_rgb_gains"] == prov["clipped_gains"]

    def test_equalize_rgb_mono_raises(self):
        req = _request(_values_corpus([1.0, 2.0, 3.0]), equalize_rgb=True)
        with pytest.raises(cs.CanonicalStackValidationError) as ei:
            run_canonical_stack(req)
        assert "RGB" in str(ei.value)

    def test_equalize_rgb_hwc1_raises(self):
        hwc1 = [np.ones((10, 10, 1), dtype=np.float32) * v for v in (1.0, 2.0, 3.0)]
        with pytest.raises(cs.CanonicalStackValidationError):
            run_canonical_stack(_request(hwc1, equalize_rgb=True))

    def test_equalize_rgb_noop_insufficient_samples(self):
        # small RGB corpus -> insufficient samples -> explicit no-op (applied=False,
        # science stays the un-equalized combined values).
        arrays = [np.ones((20, 20, 3), dtype=np.float32) * v for v in (1.0, 2.0, 3.0)]
        off = run_canonical_stack(_request(arrays, combine="mean", equalize_rgb=False))
        on = run_canonical_stack(_request(arrays, combine="mean", equalize_rgb=True))
        prov = on.provenance["equalize_rgb"]
        assert prov["applied"] is False
        assert prov["decision"] == "insufficient_samples"
        np.testing.assert_array_equal(on.science, off.science)

    def test_backend_gpu_unavailable_raises(self, monkeypatch):
        monkeypatch.setattr(
            "zemosaic.core.canonical_stacking.canonical_gpu_available", lambda: False
        )
        req = _request(_values_corpus([1.0, 2.0, 3.0]), backend="gpu")
        with pytest.raises(cs.CanonicalStackValidationError) as ei:
            run_canonical_stack(req)
        assert "unavailable" in str(ei.value)


# ---------------------------------------------------------------------------
# 5. Constant-field invariant
# ---------------------------------------------------------------------------

class TestConstantField:
    def test_constant_science_and_support_sum(self):
        arrays = [np.full((5, 5), 3.0, dtype=np.float32)] * 4
        res = run_canonical_stack(_request(arrays, taper="footprint", taper_px=6.0))
        # constant field -> constant science
        np.testing.assert_allclose(res.science, 3.0, rtol=1e-6, atol=1e-6)
        # support_w1 == sum(s_i) per pixel (independently recomputed)
        w1, w2, n_eff = _expected_support(
            arrays, [_full_support(5, 5)] * 4, "none", "none", "footprint", 6.0, 0.0
        )
        np.testing.assert_array_equal(res.support_w1, w1)
        np.testing.assert_array_equal(res.support_w2, w2)
        np.testing.assert_array_equal(res.n_eff_support, n_eff)


# ---------------------------------------------------------------------------
# 6. Determinism / immutability
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_repeat_run_deterministic(self):
        arrays = _values_corpus([10.0] * 9 + [100.0])
        req = _request(arrays, rejection="kappa_sigma", combine="mean")
        r1 = run_canonical_stack(req)
        r2 = run_canonical_stack(req)
        np.testing.assert_array_equal(r1.science, r2.science)
        np.testing.assert_array_equal(r1.support_w1, r2.support_w1)
        np.testing.assert_array_equal(r1.rejection_mask, r2.rejection_mask)
        assert r1.provenance == r2.provenance

    def test_no_input_mutation(self):
        arrays = _values_corpus([1.0, 2.0, 3.0])
        arrays_before = [a.copy() for a in arrays]
        req = _request(arrays, rejection="kappa_sigma")
        run_canonical_stack(req)
        for a, b in zip(arrays, arrays_before):
            np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# 7. Physical GPU backend parity (opt-in)
# ---------------------------------------------------------------------------

class TestBackendGPU:
    @pytest.mark.skipif(not cs.canonical_gpu_available(), reason="no CUDA device")
    def test_gpu_backend_matches_cpu(self):
        arrays = _values_corpus([10.0] * 9 + [100.0])
        cpu = run_canonical_stack(_request(arrays, rejection="kappa_sigma", combine="mean", backend="cpu"))
        gpu = run_canonical_stack(_request(arrays, rejection="kappa_sigma", combine="mean", backend="gpu"))
        np.testing.assert_array_equal(cpu.rejection_mask, gpu.rejection_mask)
        np.testing.assert_allclose(cpu.science, gpu.science, rtol=1e-12, equal_nan=True)
        np.testing.assert_array_equal(cpu.support_w1, gpu.support_w1)
        # per-stage backend: B1/B2/support stay cpu, C1/C2 honour the request
        assert cpu.provenance["backend"]["rejection"] == "cpu"
        assert cpu.provenance["backend"]["combine"] == "cpu"
        assert gpu.provenance["backend"]["requested"] == "gpu"
        assert gpu.provenance["backend"]["normalization"] == "cpu"
        assert gpu.provenance["backend"]["weighting"] == "cpu"
        assert gpu.provenance["backend"]["support"] == "cpu"
        assert gpu.provenance["backend"]["rejection"] == "gpu"
        assert gpu.provenance["backend"]["combine"] == "gpu"
        ev = gpu.provenance["effective_event"]
        assert ev["backend_requested"] == "gpu"
        assert ev["backend_normalization"] == "cpu"
        assert ev["backend_weighting"] == "cpu"
        assert ev["backend_support"] == "cpu"
        assert ev["backend_rejection"] == "gpu"
        assert ev["backend_combine"] == "gpu"
        json.dumps(gpu.provenance)  # still JSON-serializable
