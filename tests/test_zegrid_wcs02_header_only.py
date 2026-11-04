"""ZM-WCS-02 — the frame WCS loader must be strictly header-only.

The beta incident showed ``load_frame_wcs`` materializing ``hdul[0].data``
(~24.9 MB per M45 cube) only to read ``.shape``. Three transient short reads
made Astropy return an empty array and the frames were rejected even though the
files were intact.

These tests pin the bounded fix:

* ``load_frame_wcs`` succeeds using only ``hdul[0].header`` and
  ``hdul[0].shape`` (header-derived ``PrimaryHDU.shape``);
* the primary ``.data`` property is never read — a raising fake proves it;
* the admission gate consumes an immutable shape tuple, never a pixel array;
* every WCS-01 admission/refusal semantic and fail-closed reset is preserved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from astropy.io import fits

from zemosaic import zemosaic_stack_plan as zsp

H, W = 24, 32


def _wcs_core(*, sip: bool) -> fits.Header:
    h = fits.Header()
    h["CTYPE1"] = "RA---TAN-SIP" if sip else "RA---TAN"
    h["CTYPE2"] = "DEC--TAN-SIP" if sip else "DEC--TAN"
    h["CRPIX1"] = W / 2.0
    h["CRPIX2"] = H / 2.0
    h["CRVAL1"] = 10.0
    h["CRVAL2"] = 20.0
    h["CD1_1"] = -2.8e-4
    h["CD1_2"] = 0.0
    h["CD2_1"] = 0.0
    h["CD2_2"] = 2.8e-4
    if sip:
        h["A_ORDER"] = 2
        h["A_2_0"] = 0.0
        h["A_0_2"] = 0.0
        h["B_ORDER"] = 2
        h["B_2_0"] = 0.0
        h["B_0_2"] = 0.0
    return h


def _cube_header(*, sip: bool, naxis3: int = 3) -> fits.Header:
    h = _wcs_core(sip=sip)
    h["NAXIS"] = 3
    h["NAXIS1"] = W
    h["NAXIS2"] = H
    h["NAXIS3"] = int(naxis3)
    return h


class _HeaderOnlyHDU:
    """Primary HDU stand-in whose pixel payload is unreadable on purpose.

    ``.header`` and ``.shape`` are valid until the owning HDU list is closed;
    accessing either after close fails, so a successful load also proves the
    loader copied what it needed before leaving the ``with`` block. ``.data``
    raises immediately and counts every attempt.
    """

    def __init__(self, header: fits.Header, shape: tuple[int, ...] | None) -> None:
        self._header = header
        self._shape = shape
        self.header_reads = 0
        self.shape_reads = 0
        self.data_reads = 0
        self.closed = False

    def _assert_open(self) -> None:
        if self.closed:
            raise AssertionError("HDU accessed after its file was closed")

    @property
    def header(self) -> fits.Header:
        self._assert_open()
        self.header_reads += 1
        return self._header

    @property
    def shape(self) -> tuple[int, ...] | None:
        self._assert_open()
        self.shape_reads += 1
        return self._shape

    @property
    def data(self):
        self.data_reads += 1
        raise AssertionError("load_frame_wcs must never read primary .data")


class _FakeHDUL:
    def __init__(self, hdu: _HeaderOnlyHDU) -> None:
        self._hdu = hdu
        self.closed = False

    def __getitem__(self, index: int) -> _HeaderOnlyHDU:
        assert index == 0
        return self._hdu

    def __enter__(self) -> "_FakeHDUL":
        return self

    def __exit__(self, *exc_info) -> bool:
        self.closed = True
        self._hdu.closed = True
        return False


def _install(monkeypatch, header: fits.Header, shape: tuple[int, ...] | None):
    hdu = _HeaderOnlyHDU(header, shape)
    hdul = _FakeHDUL(hdu)
    monkeypatch.setattr(zsp, "_open_fits_safely", lambda path: hdul)
    return hdu, hdul


def test_2d_primary_is_loaded_without_touching_data(monkeypatch) -> None:
    hdu, hdul = _install(monkeypatch, _wcs_core(sip=False), (H, W))

    fi = zsp.FrameInfo(path=Path("ghost_2d.fits"))
    assert zsp.load_frame_wcs(fi) is True

    assert fi.shape_hw == (H, W)
    assert fi.wcs is not None
    assert fi.wcs.pixel_n_dim == 2
    assert fi.wcs.is_celestial is True
    assert hdul.closed is True
    assert hdu.data_reads == 0
    assert hdu.header_reads == 1
    assert hdu.shape_reads == 1


@pytest.mark.parametrize("sip", [True, False])
def test_rgb_chw_cube_is_loaded_without_touching_data(monkeypatch, sip: bool) -> None:
    hdu, hdul = _install(monkeypatch, _cube_header(sip=sip), (3, H, W))

    fi = zsp.FrameInfo(path=Path("ghost_cube.fits"))
    assert zsp.load_frame_wcs(fi) is True

    assert fi.shape_hw == (H, W)
    assert fi.wcs is not None
    assert fi.wcs.pixel_n_dim == 2
    assert fi.wcs.is_celestial is True
    assert (fi.wcs.sip is not None) is sip
    assert hdul.closed is True
    assert hdu.data_reads == 0
    assert hdu.header_reads == 1
    assert hdu.shape_reads == 1


def test_refused_cube_is_still_header_only_and_fail_closed(monkeypatch) -> None:
    hdu, hdul = _install(monkeypatch, _cube_header(sip=False, naxis3=4), (4, H, W))

    fi = zsp.FrameInfo(path=Path("ghost_four_planes.fits"))
    assert zsp.load_frame_wcs(fi) is False

    assert fi.wcs is None
    assert fi.shape_hw is None
    assert hdul.closed is True
    assert hdu.data_reads == 0
    assert hdu.header_reads == 1
    assert hdu.shape_reads == 1


def test_unreadable_shape_on_3d_primary_is_refused_without_data(monkeypatch) -> None:
    hdu, hdul = _install(monkeypatch, _cube_header(sip=False), None)

    fi = zsp.FrameInfo(path=Path("ghost_unreadable_shape.fits"))
    assert zsp.load_frame_wcs(fi) is False

    assert fi.wcs is None
    assert fi.shape_hw is None
    assert hdul.closed is True
    assert hdu.data_reads == 0


def test_admission_gate_accepts_only_immutable_shape_tuples() -> None:
    header = _cube_header(sip=True)

    assert zsp._admit_rgb_chw_cube(header, (3, H, W)) == (True, None, (H, W))
    assert zsp._admit_rgb_chw_cube(header, (H, W)) == (False, None, None)

    admitted, reason, shape_hw = zsp._admit_rgb_chw_cube(header, (2, H, W))
    assert admitted is False
    assert shape_hw is None
    assert reason is not None and "leading" in reason
