"""ZM-SDS-R35C-C1 — SDS intermediate FITS Linux interoperability.

Scope: ONLY the SDS intermediate export boundary. ``_persist_sds_tile`` now
delegates its FITS write to the module-level ``_write_sds_tile_fits`` helper,
which:

* reuses the existing R26 helper ``save_fits_image(..., sanitize_nonfinite=True)``
  for the normal path, and
* sanitizes NaN/+Inf/-Inf -> 0.0 on a float32 **copy** in the Astropy fallback
  (used when ``zemosaic_utils`` is unavailable), recording coherent
  ``ZNFILL``/``ZNFREPL``/HISTORY cards.

Invariants guarded here:

* primaries are entirely finite after export; every non-finite sample becomes
  exactly ``0.0``;
* finite samples — including valid zeros and negatives — are preserved
  bit-identically by the sanitization step; in-memory inputs are never mutated;
* WCS (primary) and the ALPHA extension are preserved;
* the nominal in-memory science array keeps its signed/NaN values; no-data is
  carried by ALPHA, never inferred from a primary ``0``;
* helper path AND Astropy fallback are exercised in isolation.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from zemosaic import zemosaic_worker as zw


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _wcs_header(shape=(6, 8)):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [12.5, 41.25]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = [-0.001, 0.001]
    w.array_shape = shape
    return w.to_header(relax=True)


def _rgb_hwc(h=6, w=8, with_nonfinite=True):
    data = np.linspace(-4.0, 20.0, h * w * 3, dtype=np.float32).reshape(h, w, 3)
    # A valid zero and a valid negative value are part of the fixture.
    data[1, 1, 0] = np.float32(0.0)
    data[3, 3, 1] = np.float32(-2.5)
    if with_nonfinite:
        data[0, 0, 0] = np.nan
        data[2, 3, 1] = np.inf
        data[5, 7, 2] = -np.inf
    return data


def _alpha_hw(h=6, w=8):
    a = np.full((h, w), 255, dtype=np.uint8)
    a[0, 0] = 0  # no-data pixel (must survive as ALPHA, not as "primary==0")
    return a


def _force_fallback(monkeypatch):
    monkeypatch.setattr(zw, "ZEMOSAIC_UTILS_AVAILABLE", False)
    monkeypatch.setattr(zw, "zemosaic_utils", None)


def _assert_common_metadata(hdul, *, expect_alpha, expected_replaced):
    primary = hdul[0]
    data = np.asarray(primary.data)
    # astropy returns FITS big-endian; assert the 32-bit float format, not the
    # native byte order.
    assert data.dtype.kind == "f"
    assert data.dtype.itemsize == 4
    assert primary.header["BITPIX"] == -32
    assert "BSCALE" not in primary.header
    assert "BZERO" not in primary.header
    assert np.all(np.isfinite(data))
    if expected_replaced:
        assert primary.header["ZNFILL"] == 0.0
        assert primary.header["ZNFREPL"] == expected_replaced
        assert any(
            f"replaced {expected_replaced} non-finite" in str(h)
            for h in primary.header["HISTORY"]
        )
    else:
        assert "ZNFILL" not in primary.header
        assert "ZNFREPL" not in primary.header
    if expect_alpha:
        assert "ALPHA" in hdul
    else:
        assert "ALPHA" not in hdul


# ---------------------------------------------------------------------------
# 1. Helper path (real R26 writer via save_fits_image)
# ---------------------------------------------------------------------------

def test_helper_path_all_finite_metadata_wcs_alpha(tmp_path):
    data = _rgb_hwc()
    alpha = _alpha_hw()
    orig = data.copy()
    out = tmp_path / "sds_tile_0001.fits"

    assert zw._write_sds_tile_fits(data, alpha, _wcs_header(), out) is True

    with fits.open(out) as hdul:
        # The pre-existing baseline floor already maps -Inf -> 0, so the R26
        # sanitizer replaces the remaining 2 samples (NaN, +Inf). The on-disk
        # result is still fully finite, with every non-finite input -> 0.
        _assert_common_metadata(hdul, expect_alpha=True, expected_replaced=2)
        d = np.asarray(hdul[0].data)
        assert d.shape == (3, 6, 8)  # RGB written CHW
        # every non-finite input sample is exactly 0.0 on disk
        assert d[0, 0, 0] == 0.0   # NaN
        assert d[1, 2, 3] == 0.0   # +Inf
        assert d[2, 5, 7] == 0.0   # -Inf
        # WCS preserved on the primary
        assert hdul[0].header["CTYPE1"] == "RA---TAN"
        assert hdul[0].header["CTYPE2"] == "DEC--TAN"
        assert hdul[0].header["CRVAL1"] == pytest.approx(12.5)
        assert hdul[0].header["CRVAL2"] == pytest.approx(41.25)
        # ALPHA preserved verbatim
        assert np.array_equal(np.asarray(hdul["ALPHA"].data), alpha)

    # in-memory science array is NOT mutated (still signed / still NaN)
    assert np.isnan(data[0, 0, 0])
    assert np.isinf(data[2, 3, 1])
    assert np.array_equal(data, orig, equal_nan=True)


def test_helper_path_finite_bits_identical_after_sanitize(tmp_path):
    """The sanitization step never alters finite samples.

    Compare our SDS export against the same R26 writer with
    ``sanitize_nonfinite=False`` (identical baseline-shift path); every finite
    sample must match bit-for-bit.
    """
    from zemosaic import zemosaic_utils as zu

    data = _rgb_hwc()
    alpha = _alpha_hw()
    out_san = tmp_path / "san.fits"
    out_ref = tmp_path / "ref.fits"

    assert zw._write_sds_tile_fits(data, alpha, _wcs_header(), out_san) is True
    zu.save_fits_image(
        image_data=np.ascontiguousarray(data),
        output_path=str(out_ref),
        header=_wcs_header(),
        overwrite=True,
        save_as_float=True,
        legacy_rgb_cube=False,
        progress_callback=None,
        axis_order="HWC",
        alpha_mask=alpha,
        sanitize_nonfinite=False,
    )

    with fits.open(out_san) as hs, fits.open(out_ref) as hr:
        ds = np.asarray(hs[0].data)
        dr = np.asarray(hr[0].data)

    finite = np.isfinite(dr)
    assert not np.all(finite)  # the reference really carries non-finite samples
    assert np.array_equal(ds[finite], dr[finite])  # finite bits untouched
    assert np.all(ds[~finite] == np.float32(0.0))  # replaced with exactly 0


def test_helper_path_mono_hw(tmp_path):
    from zemosaic import zemosaic_utils as zu

    data = np.linspace(-1.0, 1.0, 20, dtype=np.float32).reshape(4, 5)
    data[0, 0] = np.nan
    orig = data.copy()
    out = tmp_path / "mono.fits"
    out_ref = tmp_path / "mono_ref.fits"

    assert zw._write_sds_tile_fits(data, None, None, out) is True
    # Reference write without sanitization (identical baseline-shift path).
    zu.save_fits_image(
        image_data=np.ascontiguousarray(data),
        output_path=str(out_ref),
        header=None,
        overwrite=True,
        save_as_float=True,
        legacy_rgb_cube=False,
        progress_callback=None,
        axis_order="HWC",
        alpha_mask=None,
        sanitize_nonfinite=False,
    )

    with fits.open(out) as hdul, fits.open(out_ref) as href:
        _assert_common_metadata(hdul, expect_alpha=False, expected_replaced=1)
        d = np.asarray(hdul[0].data)
        dr = np.asarray(href[0].data)
        assert d.shape == (4, 5)  # mono/HW stays 2D
        assert d[0, 0] == 0.0
        # finite samples untouched by the sanitization step
        finite = np.isfinite(dr)
        assert np.array_equal(d[finite], dr[finite])

    assert np.isnan(data[0, 0])
    assert np.array_equal(data, orig, equal_nan=True)


def test_helper_path_zero_replacement_keeps_clean_output(tmp_path):
    data = _rgb_hwc(with_nonfinite=False)
    out = tmp_path / "clean.fits"

    assert zw._write_sds_tile_fits(data, _alpha_hw(), _wcs_header(), out) is True

    with fits.open(out) as hdul:
        _assert_common_metadata(hdul, expect_alpha=True, expected_replaced=0)
        assert np.all(np.isfinite(np.asarray(hdul[0].data)))


# ---------------------------------------------------------------------------
# 2. Astropy fallback path (zemosaic_utils unavailable)
# ---------------------------------------------------------------------------

def test_fallback_all_finite_metadata_wcs_alpha(tmp_path, monkeypatch):
    _force_fallback(monkeypatch)
    data = _rgb_hwc()
    alpha = _alpha_hw()
    orig = data.copy()
    out = tmp_path / "sds_tile_0002.fits"

    assert zw._write_sds_tile_fits(data, alpha, _wcs_header(), out) is True

    with fits.open(out) as hdul:
        # The fallback has no baseline floor, so all three non-finite samples
        # are replaced by the sanitizer.
        _assert_common_metadata(hdul, expect_alpha=True, expected_replaced=3)
        d = np.asarray(hdul[0].data)
        assert d.shape == (3, 6, 8)
        assert d[0, 0, 0] == 0.0
        assert d[1, 2, 3] == 0.0
        assert d[2, 5, 7] == 0.0
        assert hdul[0].header["CTYPE1"] == "RA---TAN"
        assert hdul[0].header["CRVAL1"] == pytest.approx(12.5)
        assert np.array_equal(np.asarray(hdul["ALPHA"].data), alpha)
        # fallback has no baseline shift -> finite samples bit-identical to input
        chw_in = np.moveaxis(data, -1, 0)
        finite = np.isfinite(chw_in)
        assert np.array_equal(d[finite], chw_in[finite])
        # valid zero and valid negative survive exactly
        assert d[0, 1, 1] == np.float32(0.0)
        assert d[1, 3, 3] == np.float32(-2.5)

    assert np.array_equal(data, orig, equal_nan=True)


def test_fallback_mono_hw_no_header(tmp_path, monkeypatch):
    _force_fallback(monkeypatch)
    data = np.linspace(-3.0, 3.0, 30, dtype=np.float32).reshape(5, 6)
    data[4, 5] = -np.inf
    orig = data.copy()
    out = tmp_path / "mono_fallback.fits"

    # header=None exercises header synthesis for the ZNFILL/ZNFREPL cards.
    assert zw._write_sds_tile_fits(data, None, None, out) is True

    with fits.open(out) as hdul:
        _assert_common_metadata(hdul, expect_alpha=False, expected_replaced=1)
        d = np.asarray(hdul[0].data)
        assert d.shape == (5, 6)
        assert d[4, 5] == 0.0
        finite = np.isfinite(orig)
        assert np.array_equal(d[finite], orig[finite])

    assert np.array_equal(data, orig, equal_nan=True)


def test_fallback_zero_replacement_adds_no_cards(tmp_path, monkeypatch):
    _force_fallback(monkeypatch)
    data = _rgb_hwc(with_nonfinite=False)
    out = tmp_path / "fallback_clean.fits"

    assert zw._write_sds_tile_fits(data, None, _wcs_header(), out) is True

    with fits.open(out) as hdul:
        _assert_common_metadata(hdul, expect_alpha=False, expected_replaced=0)
        d = np.asarray(hdul[0].data)
        chw_in = np.moveaxis(data, -1, 0)
        assert np.array_equal(d, chw_in)


def test_fallback_failure_returns_false_and_cleans_up(tmp_path, monkeypatch):
    _force_fallback(monkeypatch)
    data = _rgb_hwc()
    target = tmp_path / "as_dir"
    target.mkdir()  # writeto onto a directory must fail cleanly

    assert zw._write_sds_tile_fits(data, None, _wcs_header(), target) is False
    assert target.is_dir()  # no partial file left behind


# ---------------------------------------------------------------------------
# 3. Static non-regression: only the SDS intermediate export boundary changed
# ---------------------------------------------------------------------------

def test_persist_sds_tile_delegates_to_module_helper():
    src = inspect.getsource(zw.assemble_global_mosaic_sds)
    assert "_write_sds_tile_fits(" in src
    # The old raw inline fallback is gone from the SDS export path.
    assert "fits.HDUList(hdus).writeto(str(tile_path)" not in src
    assert "np.moveaxis(tile_data, -1, 0)" not in src


def test_module_helper_uses_r26_opt_in():
    helper_src = inspect.getsource(zw._write_sds_tile_fits)
    assert "sanitize_nonfinite=True" in helper_src
    # fallback sanitizes on a copy, never in place on the caller array
    assert ".copy()" in helper_src
    assert "ZNFREPL" in helper_src
    assert "add_history" in helper_src


def test_sds_tile_writer_is_not_the_global_default():
    """The global writer default stays opt-out so unrelated FITS writes are
    untouched (mirrors the R26 contract)."""
    from zemosaic import zemosaic_utils as zu

    sig = inspect.signature(zu.save_fits_image)
    assert sig.parameters["sanitize_nonfinite"].default is False
