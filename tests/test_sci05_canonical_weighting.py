"""SCI-05 Gate B2 — canonical scalar quality weighting (deterministic witnesses).

Deterministic, hermetic tests for the pure CPU canonical quality-weighting layer
in ``zemosaic.core.canonical_stacking``. This is **Gate B2** only: scalar quality
weights (``none``/``noise_variance``/``noise_fwhm``) on top of the B1 normalized
batch. No rejection, combine, coverage, GUI, config, GPU, or production-caller
wiring.

Design notes
------------
* Deterministic float32 corpora only; no random data, no sleeps, no network, no
  GPU, no media, no profile/XDG/HOME writes.
* Synthetic Gaussian-star fields use explicit positions and deterministic
  non-random background variation (linear gradient), so the FWHM estimator can be
  checked against the known ``2.35482 * sigma_psf`` circularized-Gaussian FWHM.
* Assertions target stable reason codes, requested==effective method, N-axis/index
  preservation, non-aliasing, and deterministic repeat-run equality.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import SourceCatalog, detect_sources

from zemosaic.core.canonical_stacking import (
    CanonicalStackFailure,
    CanonicalStackValidationError,
    compute_canonical_quality_weights,
    normalize_canonical_images,
    prepare_canonical_inputs,
)


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------

FWHM_GAUSS = 2.35482  # FWHM of a 2-D circular Gaussian = 2*sqrt(2*ln2)*sigma
STARS_5 = [(24, 24), (24, 72), (72, 24), (72, 72), (48, 48)]
STARS_3 = [(24, 24), (48, 48), (72, 72)]


def _ramp(h, w, base=0.0, step=1.0):
    """Deterministic float32 2-D mono ramp of shape ``(h, w)``."""
    return (base + step * np.arange(h * w, dtype=np.float64)).reshape(h, w).astype(np.float32)


def _full_support(h, w):
    return np.ones((h, w), dtype=bool)


def _normalize(arrays, masks=None, method="none", reference_index=None):
    """Build a B1 normalization result from raw frames."""
    if masks is None:
        masks = [_full_support(*a.shape[:2]) for a in arrays]
    batch = prepare_canonical_inputs(arrays, masks)
    return normalize_canonical_images(batch, method, reference_index=reference_index)


def _star_field(h, w, stars, sigma_psf, amp=500.0, bg=10.0, gradient=0.02):
    """Deterministic float32 mono field of circular Gaussian stars on a gradient."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.full((h, w), bg, dtype=np.float64)
    for cy, cx in stars:
        img += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma_psf ** 2))
    if gradient:
        img += gradient * (xx + yy)
    return img.astype(np.float32)


def _elongated_field(h, w, stars, sigma_x, sigma_y, amp=500.0, bg=10.0):
    """Deterministic float32 mono field of elongated (elliptical) Gaussians."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.full((h, w), bg, dtype=np.float64)
    for cy, cx in stars:
        img += amp * np.exp(
            -((xx - cx) ** 2 / (2.0 * sigma_x ** 2))
            - ((yy - cy) ** 2 / (2.0 * sigma_y ** 2))
        )
    return img.astype(np.float32)


def _independent_fwhm(norm, i):
    """Independently re-derive ``(median_fwhm, n_accepted)`` via public Photutils.

    Mirrors the canonical detector (single pass, threshold 3*sigma, n_pixels=5,
    connectivity 8, ``0.8 < fwhm < 20`` and ``ecc <= 0.8``, >=3 accepted) but is
    written inline here as a cross-check witness.
    """
    lum = norm.images[i][:, :, 0].astype(np.float64)
    valid = norm.valid_mask[i]
    stats = sigma_clipped_stats(lum[valid], sigma_lower=3, sigma_upper=3, maxiters=5)
    median = float(stats[1])
    sigma = float(stats[2])
    plane = (lum - median).astype(np.float64)
    plane[~valid] = 0.0
    segm = detect_sources(plane, threshold=3.0 * sigma, n_pixels=5, connectivity=8, mask=~valid)
    if segm is None:
        return None, 0
    catalog = SourceCatalog(plane, segm, mask=~valid, progress_bar=False)
    accepted = []
    for source in catalog:
        f = float(source.fwhm.value)
        e = float(source.eccentricity.value)
        if np.isfinite(f) and np.isfinite(e) and 0.8 < f < 20.0 and e <= 0.8:
            accepted.append(f)
    if len(accepted) < 3:
        return None, len(accepted)
    return float(np.median(accepted)), len(accepted)


# ---------------------------------------------------------------------------
# 1. API shape / token / non-mutation / N-preservation
# ---------------------------------------------------------------------------

class TestAPI:
    def test_result_shape_and_dtype(self):
        norm = _normalize([_ramp(32, 32)], method="none")
        res = compute_canonical_quality_weights(norm, "none")
        assert res.weights.shape == (1,) and res.weights.dtype == np.float64
        assert res.raw_weights.shape == (1,) and res.raw_weights.dtype == np.float64
        assert res.noise_sigma.shape == (1,) and res.noise_sigma.dtype == np.float64
        assert res.fwhm.shape == (1,) and res.fwhm.dtype == np.float64
        assert res.active_frames.shape == (1,) and res.active_frames.dtype == np.bool_
        assert res.reference_index == 0
        assert res.requested_method == res.effective_method == "none"
        assert res.n_frames == 1
        assert res.original_mono is True
        assert res.height == 32 and res.width == 32 and res.channels == 1
        assert res.original_ndim == 2 and res.original_shape == (32, 32)

    def test_n_preserved_and_b1_exclusions_inherited(self):
        a = _ramp(32, 32)
        b = _ramp(32, 32)
        c = np.full((32, 32), np.nan, dtype=np.float32)  # zero-valid -> B1 excluded
        norm = _normalize([a, b, c], method="none")
        assert any(e.index == 2 and e.reason == "zero_valid_support" for e in norm.exclusions)
        res = compute_canonical_quality_weights(norm, "none")
        assert res.weights.shape[0] == 3
        assert res.active_frames.tolist() == [True, True, False]
        assert res.weights.tolist() == [1.0, 1.0, 0.0]
        assert np.isnan(res.raw_weights[2])
        # B1 exclusion survives into the B2 result
        assert any(e.index == 2 and e.reason == "zero_valid_support" for e in res.exclusions)

    def test_no_mutation_of_normalization_input(self):
        norm = _normalize([_ramp(32, 32)], method="none")
        img_before = norm.images.copy()
        mask_before = norm.valid_mask.copy()
        active_before = norm.active_frames.copy()
        exclusions_before = norm.exclusions
        res = compute_canonical_quality_weights(norm, "noise_variance")
        assert res.active_frames[0]
        assert np.array_equal(norm.images, img_before)
        assert np.array_equal(norm.valid_mask, mask_before)
        assert np.array_equal(norm.active_frames, active_before)
        assert norm.exclusions == exclusions_before

    def test_hwc1_and_rgb(self):
        # Genuine (H, W, 1) frame (not a 2-D HW): exercises the HWC1 channel-0
        # luminance path explicitly.
        hw = _ramp(32, 32)
        hwc1 = hw[..., np.newaxis].astype(np.float32)  # (32, 32, 1)
        assert hwc1.shape == (32, 32, 1)
        norm1 = _normalize([hwc1], method="none")
        assert norm1.original_mono is False
        assert norm1.channels == 1
        res1 = compute_canonical_quality_weights(norm1, "noise_variance")
        assert res1.channels == 1
        assert res1.original_mono is False
        assert res1.weights.shape == (1,)
        assert res1.active_frames[0]
        # channel-0 sigma equals a direct Astropy witness on the same data
        direct = float(
            sigma_clipped_stats(
                hw.astype(np.float64).ravel(), sigma_lower=3, sigma_upper=3, maxiters=5
            )[2]
        )
        assert res1.noise_sigma[0] == pytest.approx(direct, rel=1e-12)
        # and equals the equivalent HW frame's sigma (HWC1 channel 0 == HW)
        res_hw = compute_canonical_quality_weights(_normalize([hw], method="none"), "noise_variance")
        assert res1.noise_sigma[0] == pytest.approx(res_hw.noise_sigma[0], rel=1e-12)

        # RGB coverage retained
        rgb = np.stack(
            [_ramp(32, 32), _ramp(32, 32, step=2.0), _ramp(32, 32, step=0.5)], axis=-1
        ).astype(np.float32)
        res3 = compute_canonical_quality_weights(_normalize([rgb], method="none"), "noise_variance")
        assert res3.channels == 3
        assert res3.weights.shape == (1,)
        assert res3.active_frames[0]

    def test_token_validation(self):
        norm = _normalize([_ramp(32, 32)], method="none")
        for bad in ("variance", "noise-variance", "NoiseFwhm", "fwhm", "NONE2", "", "noise_variance_extra"):
            with pytest.raises(CanonicalStackValidationError):
                compute_canonical_quality_weights(norm, bad)
        with pytest.raises(CanonicalStackValidationError):
            compute_canonical_quality_weights(norm, 123)
        with pytest.raises(CanonicalStackValidationError):
            compute_canonical_quality_weights(norm, None)
        # strip/lower accepted; requested==effective exact
        r = compute_canonical_quality_weights(norm, "  Noise_Variance ")
        assert r.requested_method == r.effective_method == "noise_variance"


# ---------------------------------------------------------------------------
# 2. none
# ---------------------------------------------------------------------------

class TestNoneWeights:
    def test_none_exact_1_active_0_inactive_no_exposure_fold(self):
        a = _ramp(32, 32, base=1.0)
        b = _ramp(32, 32, base=100.0)  # 100x brighter -> must still be q=1 (no exposure fold)
        c = np.full((32, 32), np.nan, dtype=np.float32)
        norm = _normalize([a, b, c], method="none")
        res = compute_canonical_quality_weights(norm, "none")
        assert res.weights.tolist() == [1.0, 1.0, 0.0]
        assert res.raw_weights[0] == 1.0 and res.raw_weights[1] == 1.0
        assert np.isnan(res.raw_weights[2])
        assert np.all(np.isnan(res.noise_sigma))
        assert np.all(np.isnan(res.fwhm))
        assert res.requested_method == res.effective_method == "none"
        assert res.exclusions == norm.exclusions  # no new B2 exclusions


# ---------------------------------------------------------------------------
# 3. Rec.709 luminance
# ---------------------------------------------------------------------------

class TestRec709:
    def test_rec709_scalar_sigma_discriminates(self):
        # Per-channel amplitudes chosen so Rec.709, 0.299, and per-channel scalars
        # all differ; the canonical scalar must follow exact Rec.709 luminance.
        unit = _ramp(32, 32)
        rgb = np.stack(
            [3.0 * unit, 1.0 * unit, 0.5 * unit], axis=-1
        ).astype(np.float32)
        norm = _normalize([rgb], method="none")
        res = compute_canonical_quality_weights(norm, "noise_variance")
        sigma = res.noise_sigma[0]

        valid = norm.valid_mask[0]
        r64 = rgb[:, :, 0].astype(np.float64)
        g64 = rgb[:, :, 1].astype(np.float64)
        b64 = rgb[:, :, 2].astype(np.float64)
        lum_rec709 = 0.2126 * r64 + 0.7152 * g64 + 0.0722 * b64
        lum_0299 = 0.299 * r64 + 0.587 * g64 + 0.114 * b64

        expected = float(
            sigma_clipped_stats(lum_rec709[valid], sigma_lower=3, sigma_upper=3, maxiters=5)[2]
        )
        sigma_0299 = float(
            sigma_clipped_stats(lum_0299[valid], sigma_lower=3, sigma_upper=3, maxiters=5)[2]
        )
        sigma_ch0 = float(
            sigma_clipped_stats(r64[valid], sigma_lower=3, sigma_upper=3, maxiters=5)[2]
        )

        assert sigma == pytest.approx(expected, rel=1e-12)
        # discriminates from the old 0.299 formula and from per-channel R
        assert sigma != pytest.approx(sigma_0299, rel=1e-6)
        assert sigma != pytest.approx(sigma_ch0, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. noise_variance
# ---------------------------------------------------------------------------

class TestNoiseVariance:
    def test_raw_inverse_sigma_squared_and_max1(self):
        a = _ramp(32, 32, step=1.0)          # sigma ~ s
        b = (2.0 * a).astype(np.float32)     # sigma ~ 2s
        norm = _normalize([a, b], method="none")
        res = compute_canonical_quality_weights(norm, "noise_variance")
        sa = res.noise_sigma[0]
        sb = res.noise_sigma[1]
        assert sb == pytest.approx(2.0 * sa, rel=1e-6)
        # exact raw formula 1/sigma^2
        assert res.raw_weights[0] == pytest.approx(1.0 / (sa * sa), rel=1e-12)
        assert res.raw_weights[1] == pytest.approx(1.0 / (sb * sb), rel=1e-12)
        # normalized max = 1 (frame 0 has the smaller sigma -> larger raw)
        assert res.weights[0] == pytest.approx(1.0, abs=1e-12)
        assert res.weights[1] == pytest.approx(0.25, rel=1e-6)

    def test_max_computed_after_exclusion(self):
        a = _ramp(32, 32, step=1.0)          # smallest sigma -> largest raw
        b = _ramp(32, 32, step=4.0)          # larger sigma -> smaller raw
        flat = np.full((32, 32), 5.0, dtype=np.float32)  # sigma=0 -> excluded
        norm = _normalize([a, b, flat], method="none")
        res = compute_canonical_quality_weights(norm, "noise_variance")
        assert not res.active_frames[2]
        assert any(e.index == 2 and e.reason == "noise_sigma_failed" for e in res.exclusions)
        assert np.isnan(res.raw_weights[2]) and res.weights[2] == 0.0
        # survivor max excludes the failed frame -> max is raw[0]
        assert res.weights[0] == pytest.approx(1.0, abs=1e-12)
        assert res.weights[1] == pytest.approx(res.raw_weights[1] / res.raw_weights[0], rel=1e-12)

    def test_under_256_valid_excludes_nonreference(self):
        ref = _ramp(32, 32)
        img = _ramp(32, 32)
        mask = np.zeros((32, 32), dtype=bool)
        mask[:10, :20] = True  # 200 valid < 256
        norm = _normalize([ref, img], [_full_support(32, 32), mask], method="none", reference_index=0)
        res = compute_canonical_quality_weights(norm, "noise_variance")
        assert res.active_frames[0]
        assert not res.active_frames[1]
        assert any(e.index == 1 and e.reason == "insufficient_quality_samples" for e in res.exclusions)
        assert res.weights[1] == 0.0 and np.isnan(res.raw_weights[1])

    def test_reference_metric_failure_raises(self):
        flat = np.full((32, 32), 5.0, dtype=np.float32)  # reference sigma=0
        img = _ramp(32, 32)
        norm = _normalize([flat, img], method="none", reference_index=0)
        with pytest.raises(CanonicalStackFailure) as ei:
            compute_canonical_quality_weights(norm, "noise_variance")
        assert "noise_sigma_failed" in str(ei.value)

    def test_raw_overflow_failure_seam(self, monkeypatch):
        # Defensive seam: a pathological tiny-but-positive sigma makes 1/sigma^2
        # overflow to Inf -> raw_weight_failed (no epsilon floor). Not honestly
        # reachable through the float32 pipeline, so the stats helper is patched.
        import zemosaic.core.canonical_stacking as cs

        a = _ramp(32, 32)
        b = _ramp(32, 32)
        norm = _normalize([a, b], method="none", reference_index=0)
        real = cs._noise_stats
        calls = {"n": 0}

        def patched(lum, valid):
            calls["n"] += 1
            if calls["n"] == 1:  # frame 0 (reference) real
                return real(lum, valid)
            return (0.0, 1e-200, None)  # frame 1: tiny sigma -> raw overflow

        monkeypatch.setattr(cs, "_noise_stats", patched)
        res = compute_canonical_quality_weights(norm, "noise_variance")
        assert res.active_frames[0]
        assert not res.active_frames[1]
        assert any(e.index == 1 and e.reason == "raw_weight_failed" for e in res.exclusions)
        assert np.isnan(res.raw_weights[1]) and res.weights[1] == 0.0

    def test_variance_unaffected_by_fwhm_availability(self, monkeypatch):
        import zemosaic.core.canonical_stacking as cs

        monkeypatch.setattr(cs, "canonical_noise_fwhm_available", lambda: False)
        norm = _normalize([_ramp(32, 32)], method="none")
        res = compute_canonical_quality_weights(norm, "noise_variance")
        assert res.effective_method == "noise_variance"
        assert res.active_frames[0]


# ---------------------------------------------------------------------------
# 5. noise_fwhm (deterministic Gaussian-star fields)
# ---------------------------------------------------------------------------

class TestNoiseFwhm:
    def test_median_fwhm_near_known_value(self):
        img = _star_field(96, 96, STARS_5, sigma_psf=2.0)
        norm = _normalize([img], method="none")
        res = compute_canonical_quality_weights(norm, "noise_fwhm")
        assert res.active_frames[0]
        # independent witness: detector obtains >=3 accepted sources
        median_fwhm, n_accepted = _independent_fwhm(norm, 0)
        assert n_accepted >= 3
        assert res.fwhm[0] == pytest.approx(median_fwhm, rel=1e-12)
        # near the known circularized-Gaussian FWHM 2.35482*sigma_psf
        assert res.fwhm[0] == pytest.approx(FWHM_GAUSS * 2.0, rel=0.15)

    def test_raw_includes_fwhm_not_variance_only(self):
        # Two frames, same background/gradient, different star width (sigma_psf).
        a = _star_field(96, 96, STARS_5, sigma_psf=2.0)
        b = _star_field(96, 96, STARS_5, sigma_psf=3.0)
        norm = _normalize([a, b], method="none")
        res = compute_canonical_quality_weights(norm, "noise_fwhm")
        assert res.active_frames[0] and res.active_frames[1]
        sa, sb = res.noise_sigma[0], res.noise_sigma[1]
        fa, fb = res.fwhm[0], res.fwhm[1]
        assert fa != fb  # different star widths
        # exact raw formula 1/(sigma^2 * fwhm^2)
        assert res.raw_weights[0] == pytest.approx(1.0 / (sa * sa * fa * fa), rel=1e-12)
        assert res.raw_weights[1] == pytest.approx(1.0 / (sb * sb * fb * fb), rel=1e-12)
        # NOT variance-only: raw != 1/sigma^2 (fwhm != 1 px)
        assert res.raw_weights[0] != pytest.approx(1.0 / (sa * sa), rel=1e-6)
        assert res.raw_weights[1] != pytest.approx(1.0 / (sb * sb), rel=1e-6)
        # normalized max = 1
        assert max(res.weights[0], res.weights[1]) == pytest.approx(1.0, abs=1e-12)
        assert res.weights[0] > 0.0 and res.weights[1] > 0.0


# ---------------------------------------------------------------------------
# 6. FWHM exclusion / filtering / reference failure
# ---------------------------------------------------------------------------

class TestFwhmExclusions:
    def test_star_free_excludes(self):
        # Smooth gradient only: sigma > 0 but no point sources above 3*sigma.
        yy, xx = np.mgrid[0:96, 0:96].astype(np.float64)
        gradient = (10.0 + 0.05 * (xx + yy)).astype(np.float32)
        ref = _star_field(96, 96, STARS_5, sigma_psf=2.0)
        norm = _normalize([ref, gradient], method="none", reference_index=0)
        res = compute_canonical_quality_weights(norm, "noise_fwhm")
        assert res.active_frames[0]
        assert not res.active_frames[1]
        assert any(e.index == 1 and e.reason == "fwhm_insufficient_sources" for e in res.exclusions)

    def test_under_3_accepted_sources_excludes(self):
        ref = _star_field(96, 96, STARS_5, sigma_psf=2.0)
        two = _star_field(96, 96, [(24, 24), (72, 72)], sigma_psf=2.0)  # only 2 sources
        norm = _normalize([ref, two], method="none", reference_index=0)
        res = compute_canonical_quality_weights(norm, "noise_fwhm")
        assert res.active_frames[0]
        assert not res.active_frames[1]
        assert any(e.index == 1 and e.reason == "fwhm_insufficient_sources" for e in res.exclusions)

    def test_elongated_eccentric_filtered(self):
        # 3 elongated stars detected but all ecc > 0.8 -> filtered -> insufficient.
        ref = _star_field(96, 96, STARS_5, sigma_psf=2.0)
        elongated = _elongated_field(96, 96, STARS_3, sigma_x=3.0, sigma_y=0.8)
        # witness: sources exist but are eccentric
        _, n = _independent_fwhm(_normalize([elongated], method="none"), 0)
        assert n < 3  # independently filtered out too
        norm = _normalize([ref, elongated], method="none", reference_index=0)
        res = compute_canonical_quality_weights(norm, "noise_fwhm")
        assert res.active_frames[0]
        assert not res.active_frames[1]
        assert any(e.index == 1 and e.reason == "fwhm_insufficient_sources" for e in res.exclusions)

    def test_out_of_range_fwhm_filtered(self):
        # 3 broad isolated stars detected with fwhm > 20 px -> filtered.
        ref = _star_field(96, 96, STARS_5, sigma_psf=2.0)
        broad = _star_field(160, 160, [(40, 40), (120, 40), (80, 120)], sigma_psf=10.0, amp=100.0)
        # witness: detected sources have fwhm > 20 (out of accepted range)
        norm_broad = _normalize([broad], method="none")
        fwhm_w, n_w = _independent_fwhm(norm_broad, 0)
        assert n_w < 3  # all filtered for fwhm > 20
        assert fwhm_w is None
        # now as non-reference alongside a good reference
        # (different sizes cannot share a batch, so use a same-size good ref)
        yy, xx = np.mgrid[0:160, 0:160].astype(np.float64)
        good = _star_field(160, 160, [(40, 40), (120, 40), (80, 120), (40, 120), (120, 120)], sigma_psf=2.0)
        norm = _normalize([good, broad], method="none", reference_index=0)
        res = compute_canonical_quality_weights(norm, "noise_fwhm")
        assert res.active_frames[0]
        assert not res.active_frames[1]
        assert any(e.index == 1 and e.reason == "fwhm_insufficient_sources" for e in res.exclusions)

    def test_reference_fwhm_failure_aborts(self):
        ref = _star_field(96, 96, STARS_5, sigma_psf=2.0)
        gradient = (10.0 + 0.05 * np.add.outer(np.arange(96), np.arange(96)).astype(np.float64)).astype(np.float32)
        norm = _normalize([gradient, ref], method="none", reference_index=0)
        with pytest.raises(CanonicalStackFailure) as ei:
            compute_canonical_quality_weights(norm, "noise_fwhm")
        assert "fwhm_insufficient_sources" in str(ei.value)

    def test_nonreference_failure_does_not_affect_survivor_max(self):
        a = _star_field(96, 96, STARS_5, sigma_psf=2.0)   # narrow stars -> larger raw
        b = _star_field(96, 96, STARS_5, sigma_psf=3.0)   # wider stars -> smaller raw
        bad = _star_field(96, 96, [(24, 24), (72, 72)], sigma_psf=2.0)  # 2 sources -> excluded
        norm = _normalize([a, b, bad], method="none", reference_index=0)
        res = compute_canonical_quality_weights(norm, "noise_fwhm")
        assert res.active_frames[0] and res.active_frames[1]
        assert not res.active_frames[2]
        assert any(e.index == 2 and e.reason == "fwhm_insufficient_sources" for e in res.exclusions)
        assert res.weights[2] == 0.0 and np.isnan(res.raw_weights[2])
        # survivor max computed over {0,1} only
        max_raw = max(res.raw_weights[0], res.raw_weights[1])
        assert max(res.weights[0], res.weights[1]) == pytest.approx(1.0, abs=1e-12)
        assert res.weights[0] == pytest.approx(res.raw_weights[0] / max_raw, rel=1e-12)
        assert res.weights[1] == pytest.approx(res.raw_weights[1] / max_raw, rel=1e-12)


# ---------------------------------------------------------------------------
# 7. Photutils availability / exception containment
# ---------------------------------------------------------------------------

class TestPhotutilsAvailability:
    def test_unavailable_before_call_validation_error(self, monkeypatch):
        import zemosaic.core.canonical_stacking as cs

        norm = _normalize([_star_field(96, 96, STARS_5, sigma_psf=2.0)], method="none")
        img_before = norm.images.copy()
        monkeypatch.setattr(cs, "canonical_noise_fwhm_available", lambda: False)
        assert cs.canonical_noise_fwhm_available() is False
        with pytest.raises(CanonicalStackValidationError):
            compute_canonical_quality_weights(norm, "noise_fwhm")
        # no fallback, no mutation of the normalization input
        assert np.array_equal(norm.images, img_before)

    def test_detection_exception_reference_failure(self, monkeypatch):
        import zemosaic.core.canonical_stacking as cs

        def boom(data, threshold, n_pixels, connectivity=8, mask=None):
            raise RuntimeError("detect_sources exploded")

        monkeypatch.setattr(cs, "_detect_sources", boom)
        norm = _normalize([_star_field(96, 96, STARS_5, sigma_psf=2.0)], method="none")
        with pytest.raises(CanonicalStackFailure) as ei:
            compute_canonical_quality_weights(norm, "noise_fwhm")
        assert "fwhm_measurement_failed" in str(ei.value)

    def test_catalog_exception_nonreference_excluded(self, monkeypatch):
        import zemosaic.core.canonical_stacking as cs

        real = cs._SourceCatalog
        calls = {"n": 0}

        class _PatchedCatalog:
            # Preserve the availability signal (``hasattr(..., "fwhm")``) and a
            # signature-compatible ``__init__`` so the preflight passes.
            fwhm = getattr(real, "fwhm", None)

            def __init__(self, data, segmentation_image, mask=None, progress_bar=False, **kwargs):
                calls["n"] += 1
                if calls["n"] == 2:  # second (non-reference) frame explodes
                    raise RuntimeError("SourceCatalog exploded")
                self._inner = real(
                    data, segmentation_image, mask=mask, progress_bar=progress_bar, **kwargs
                )

            def __iter__(self):
                return iter(self._inner)

        monkeypatch.setattr(cs, "_SourceCatalog", _PatchedCatalog)
        ref = _star_field(96, 96, STARS_5, sigma_psf=2.0)
        other = _star_field(96, 96, STARS_5, sigma_psf=2.0)
        norm = _normalize([ref, other], method="none", reference_index=0)
        res = compute_canonical_quality_weights(norm, "noise_fwhm")
        assert res.active_frames[0]
        assert not res.active_frames[1]
        assert any(e.index == 1 and e.reason == "fwhm_measurement_failed" for e in res.exclusions)
        assert np.isnan(res.raw_weights[1]) and res.weights[1] == 0.0

    # -- F1: importable-but-API-incompatible Photutils must be preflighted out --

    def test_legacy_npixels_detect_signature_unavailable(self, monkeypatch):
        import zemosaic.core.canonical_stacking as cs

        calls = []

        def legacy_detect(data, threshold, npixels, connectivity=8, mask=None):
            calls.append(True)
            raise AssertionError("detector must not be called")

        monkeypatch.setattr(cs, "_detect_sources", legacy_detect)
        assert cs.canonical_noise_fwhm_available() is False
        norm = _normalize([_star_field(96, 96, STARS_5, sigma_psf=2.0)], method="none")
        img_before = norm.images.copy()
        with pytest.raises(CanonicalStackValidationError):
            compute_canonical_quality_weights(norm, "noise_fwhm")
        assert calls == []  # validation failed before the detector ran
        assert np.array_equal(norm.images, img_before)

    def test_incompatible_sourcecatalog_signature_unavailable(self, monkeypatch):
        import zemosaic.core.canonical_stacking as cs

        calls = []

        class _BadCatalog:
            fwhm = None  # property present, but signature lacks progress_bar

            def __init__(self, data, segmentation_image, mask=None):
                calls.append(True)
                raise AssertionError("catalog must not be called")

        monkeypatch.setattr(cs, "_SourceCatalog", _BadCatalog)
        assert cs.canonical_noise_fwhm_available() is False
        norm = _normalize([_star_field(96, 96, STARS_5, sigma_psf=2.0)], method="none")
        img_before = norm.images.copy()
        with pytest.raises(CanonicalStackValidationError):
            compute_canonical_quality_weights(norm, "noise_fwhm")
        assert calls == []
        assert np.array_equal(norm.images, img_before)

    def test_missing_fwhm_property_unavailable(self, monkeypatch):
        import zemosaic.core.canonical_stacking as cs

        class _NoFwhmCatalog:
            def __init__(self, data, segmentation_image, mask=None, progress_bar=False):
                pass

        monkeypatch.setattr(cs, "_SourceCatalog", _NoFwhmCatalog)
        assert cs.canonical_noise_fwhm_available() is False
        norm = _normalize([_star_field(96, 96, STARS_5, sigma_psf=2.0)], method="none")
        with pytest.raises(CanonicalStackValidationError):
            compute_canonical_quality_weights(norm, "noise_fwhm")

    def test_missing_segmentation_image_capability_unavailable(self, monkeypatch):
        import zemosaic.core.canonical_stacking as cs

        class _BadSegmentation:
            pass  # no n_labels, no nlabels

        monkeypatch.setattr(cs, "_SegmentationImage", _BadSegmentation)
        assert cs.canonical_noise_fwhm_available() is False
        norm = _normalize([_star_field(96, 96, STARS_5, sigma_psf=2.0)], method="none")
        with pytest.raises(CanonicalStackValidationError):
            compute_canonical_quality_weights(norm, "noise_fwhm")

class TestDeterminismOwnership:
    def test_repeat_run_deterministic(self):
        norm = _normalize([_star_field(96, 96, STARS_5, sigma_psf=2.0)], method="none")
        r1 = compute_canonical_quality_weights(norm, "noise_fwhm")
        r2 = compute_canonical_quality_weights(norm, "noise_fwhm")
        np.testing.assert_array_equal(r1.weights, r2.weights)
        np.testing.assert_array_equal(r1.raw_weights, r2.raw_weights)
        np.testing.assert_array_equal(r1.noise_sigma, r2.noise_sigma)
        np.testing.assert_array_equal(r1.fwhm, r2.fwhm)
        np.testing.assert_array_equal(r1.active_frames, r2.active_frames)
        assert r1.exclusions == r2.exclusions
        assert r1.reference_index == r2.reference_index

    def test_arrays_owned_contiguous_nonaliasing(self):
        norm = _normalize([_star_field(96, 96, STARS_5, sigma_psf=2.0)], method="none")
        res = compute_canonical_quality_weights(norm, "noise_fwhm")
        for arr in (res.weights, res.raw_weights, res.noise_sigma, res.fwhm, res.active_frames):
            assert arr.flags["OWNDATA"]
            assert arr.flags["C_CONTIGUOUS"]
        assert not np.shares_memory(res.weights, norm.images)
        assert not np.shares_memory(res.active_frames, norm.active_frames)
        # mutating a result array must not touch the normalization input
        res.weights[0] = 12345.0
        assert not np.any(norm.images == 12345.0)
