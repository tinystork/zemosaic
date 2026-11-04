"""ZM-WCS-01 — RGB CHW 3-D primaries with a 2-D celestial WCS.

The product loader (:func:`zemosaic.zemosaic_stack_plan.load_frame_wcs`) used to
build ``WCS(header)`` unconditionally. For a prepared RGB CHW cube
(``NAXIS=3``, ``NAXIS3=3``, ``(3, H, W)``, blank ``CTYPE3``) whose celestial WCS
lives on axes 1-2, that call either raises (SIP) or yields a 3-D non-celestial
WCS (plain TAN) — so every real M45 RGB frame was rejected.

These tests pin the bounded fix:

* the 2-D primary path is unchanged;
* ONLY the unambiguous CHW RGB class builds ``WCS(header, naxis=2)``;
* every other 3-D primary is refused explicitly;
* ``qualify_wcs`` still applies the whole TAN/SIP/PV/lookup policy downstream;
* the decoder and descriptor pipeline consume the cube unchanged.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from zemosaic import zemosaic_stack_plan as zsp
from zemosaic.core.zegrid import geometry as zg

H, W = 24, 32


# ---------------------------------------------------------------------------
# Header builders — the five real M45 header signatures (scaled down)
# ---------------------------------------------------------------------------

def _wcs_core(*, sip: bool, pc: bool, wcsaxes: bool, cunit2: bool) -> fits.Header:
    h = fits.Header()
    if wcsaxes:
        h["WCSAXES"] = 2
    h["CTYPE1"] = "RA---TAN-SIP" if sip else "RA---TAN"
    h["CTYPE2"] = "DEC--TAN-SIP" if sip else "DEC--TAN"
    h["CRPIX1"] = W / 2.0
    h["CRPIX2"] = H / 2.0
    h["CRVAL1"] = 10.0
    h["CRVAL2"] = 20.0
    if pc:
        h["PC1_1"] = -1.0
        h["PC1_2"] = 0.0
        h["PC2_1"] = 0.0
        h["PC2_2"] = 1.0
        h["CDELT1"] = 2.8e-4
        h["CDELT2"] = 2.8e-4
    else:
        h["CD1_1"] = -2.8e-4
        h["CD1_2"] = 0.0
        h["CD2_1"] = 0.0
        h["CD2_2"] = 2.8e-4
    if cunit2:
        h["CUNIT1"] = "deg"
        h["CUNIT2"] = "deg"
    if sip:
        # Zero-valued but present SIP terms: enough for astropy to build the
        # Sip object and for the SIP code path to be exercised (as on disk).
        h["A_ORDER"] = 2
        h["A_2_0"] = 0.0
        h["A_0_2"] = 0.0
        h["B_ORDER"] = 2
        h["B_2_0"] = 0.0
        h["B_0_2"] = 0.0
    return h


def _cube_header(
    *,
    sip: bool,
    pc: bool,
    wcsaxes: bool,
    cunit2: bool,
    ctype3: str | None = None,
    naxis1: int = W,
    naxis2: int = H,
    naxis3: int = 3,
) -> fits.Header:
    h = _wcs_core(sip=sip, pc=pc, wcsaxes=wcsaxes, cunit2=cunit2)
    # The third WCS axis is declared as a no-op channel axis via the linear
    # matrix only; CTYPE3 stays blank (exactly the audited corpus).
    if pc:
        h["PC3_3"] = 1.0
        h["CDELT3"] = 1.0
    else:
        h["CD3_3"] = 1.0
    if ctype3 is not None:
        h["CTYPE3"] = ctype3
    h["NAXIS"] = 3
    h["NAXIS1"] = int(naxis1)
    h["NAXIS2"] = int(naxis2)
    h["NAXIS3"] = int(naxis3)
    return h


def _chw(channels: int = 3, h: int = H, w: int = W) -> np.ndarray:
    return np.zeros((channels, h, w), dtype=np.float32)


def _write(path: Path, data: np.ndarray, header: fits.Header) -> Path:
    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)
    return path


# ---------------------------------------------------------------------------
# Historical 2-D path — unchanged semantics
# ---------------------------------------------------------------------------

def test_2d_tan_primary_path_unchanged(tmp_path: Path) -> None:
    data = np.zeros((H, W), dtype=np.float32)
    header = _wcs_core(sip=False, pc=False, wcsaxes=False, cunit2=False)
    path = _write(tmp_path / "mono.fits", data, header)

    fi = zsp.FrameInfo(path=path)
    assert zsp.load_frame_wcs(fi) is True
    assert fi.shape_hw == (H, W)
    assert fi.wcs is not None
    assert fi.wcs.pixel_n_dim == 2
    assert fi.wcs.is_celestial is True
    assert zg.qualify_wcs(fi.wcs) is None


# ---------------------------------------------------------------------------
# Admitted CHW RGB cube class (the actual M45 defect)
# ---------------------------------------------------------------------------

def test_chw_sip_cd_no_wcsaxes_no_cunit2_is_admitted(tmp_path: Path) -> None:
    header = _cube_header(sip=True, pc=False, wcsaxes=False, cunit2=False)
    path = _write(tmp_path / "sip_cd.fits", _chw(), header)

    fi = zsp.FrameInfo(path=path)
    assert zsp.load_frame_wcs(fi) is True
    assert fi.shape_hw == (H, W)  # exactly (H, W) for CHW
    assert fi.wcs is not None
    assert fi.wcs.pixel_n_dim == 2
    assert fi.wcs.is_celestial is True
    assert fi.wcs.sip is not None  # SIP preserved, not stripped
    assert zg.qualify_wcs(fi.wcs) is None


def test_chw_sip_pc_cdelt_wcsaxes2_is_admitted(tmp_path: Path) -> None:
    header = _cube_header(sip=True, pc=True, wcsaxes=True, cunit2=False)
    path = _write(tmp_path / "sip_pc.fits", _chw(), header)

    fi = zsp.FrameInfo(path=path)
    assert zsp.load_frame_wcs(fi) is True
    assert fi.shape_hw == (H, W)
    assert fi.wcs.pixel_n_dim == 2
    assert fi.wcs.is_celestial is True
    assert fi.wcs.sip is not None
    assert zg.qualify_wcs(fi.wcs) is None


def test_chw_tan_cd_no_wcsaxes_is_admitted(tmp_path: Path) -> None:
    header = _cube_header(sip=False, pc=False, wcsaxes=False, cunit2=False)
    path = _write(tmp_path / "tan_cd.fits", _chw(), header)

    fi = zsp.FrameInfo(path=path)
    assert zsp.load_frame_wcs(fi) is True
    assert fi.shape_hw == (H, W)
    assert fi.wcs.pixel_n_dim == 2
    assert fi.wcs.is_celestial is True
    assert fi.wcs.sip is None
    assert zg.qualify_wcs(fi.wcs) is None


def test_chw_tan_pc_cdelt_wcsaxes2_is_admitted(tmp_path: Path) -> None:
    header = _cube_header(sip=False, pc=True, wcsaxes=True, cunit2=True)
    path = _write(tmp_path / "tan_pc.fits", _chw(), header)

    fi = zsp.FrameInfo(path=path)
    assert zsp.load_frame_wcs(fi) is True
    assert fi.shape_hw == (H, W)
    assert fi.wcs.pixel_n_dim == 2
    assert fi.wcs.is_celestial is True
    assert zg.qualify_wcs(fi.wcs) is None


def test_baseline_header_without_naxis2_was_rejected(tmp_path: Path) -> None:
    """Witness that the pre-fix construction fails on the same header class."""

    header = _cube_header(sip=True, pc=False, wcsaxes=False, cunit2=False)
    with pytest.raises(ValueError, match="naxis"):
        WCS(header)
    # And the TAN class silently degraded to a non-celestial 3-D WCS.
    tan_header = _cube_header(sip=False, pc=False, wcsaxes=False, cunit2=False)
    assert WCS(tan_header).is_celestial is False


# ---------------------------------------------------------------------------
# Explicit refusals — no widening to ambiguous cubes
# ---------------------------------------------------------------------------

def test_non_rgb_third_axis_count_is_refused(tmp_path: Path) -> None:
    header = _cube_header(sip=False, pc=False, wcsaxes=False, cunit2=False, naxis3=4)
    path = _write(tmp_path / "four_planes.fits", _chw(channels=4), header)

    fi = zsp.FrameInfo(path=path)
    assert zsp.load_frame_wcs(fi) is False
    assert fi.wcs is None


@pytest.mark.parametrize(
    "ctype3, expected",
    [
        ("WAVE", "spectral"),
        ("UTC", "temporal"),
        ("RA---TAN", "celestial"),
        ("STOKES", "polarization"),
        ("BAND", "other"),
    ],
)
def test_declared_third_axis_ctype3_is_refused(tmp_path: Path, ctype3: str, expected: str) -> None:
    header = _cube_header(sip=False, pc=False, wcsaxes=True, cunit2=True, ctype3=ctype3)
    path = _write(tmp_path / f"ctype3_{ctype3.replace('-', '_')}.fits", _chw(), header)

    fi = zsp.FrameInfo(path=path)
    assert zsp.load_frame_wcs(fi) is False
    assert fi.wcs is None


def test_wcsaxes2_alone_does_not_admit_ambiguous_cube(tmp_path: Path) -> None:
    # WCSAXES=2 + 3 planes + declared spectral third axis: still refused.
    header = _cube_header(
        sip=False, pc=True, wcsaxes=True, cunit2=True, ctype3="WAVE"
    )
    path = _write(tmp_path / "wcsaxes2_spectral.fits", _chw(), header)

    fi = zsp.FrameInfo(path=path)
    assert zsp.load_frame_wcs(fi) is False
    assert fi.wcs is None


@pytest.mark.parametrize(
    "header_mutator, reason_fragment",
    [
        (lambda h: h.__setitem__("NAXIS3", 4), "NAXIS3"),
        (lambda h: h.__setitem__("NAXIS1", W + 1), "contradict"),
        (lambda h: h.__setitem__("NAXIS2", H + 1), "contradict"),
        (lambda h: h.__delitem__("NAXIS"), "NAXIS"),
    ],
)
def test_admission_gate_refuses_contradictory_headers(
    header_mutator, reason_fragment: str
) -> None:
    header = _cube_header(sip=False, pc=False, wcsaxes=False, cunit2=False)
    header_mutator(header)
    admitted, reason, shape_hw = zsp._admit_rgb_chw_cube(header, _chw())
    assert admitted is False
    assert shape_hw is None
    assert reason is not None and reason_fragment in reason


def test_admission_gate_refuses_hwc_shaped_data() -> None:
    # (H, W, 3) pixels under a CHW-looking header: leading axis is not 3.
    header = _cube_header(sip=False, pc=False, wcsaxes=False, cunit2=False)
    hwc = np.zeros((H, W, 3), dtype=np.float32)
    admitted, reason, shape_hw = zsp._admit_rgb_chw_cube(header, hwc)
    assert admitted is False
    assert shape_hw is None
    assert reason is not None and "leading data axis" in reason


def test_admission_gate_refuses_non_3d_axis_count() -> None:
    header = _cube_header(sip=False, pc=False, wcsaxes=False, cunit2=False)
    four_d = np.zeros((3, 2, H, W), dtype=np.float32)
    admitted, reason, shape_hw = zsp._admit_rgb_chw_cube(header, four_d)
    assert admitted is False
    assert shape_hw is None
    assert reason is not None and "dimensionality" in reason


# ---------------------------------------------------------------------------
# Downstream witnesses: decoder + descriptor pipeline
# ---------------------------------------------------------------------------

def test_decode_frame_hwc_on_chw_fits_keeps_channel_order(tmp_path: Path) -> None:
    from zemosaic.zemosaic_zegrid_mode import _decode_frame_hwc

    chw = np.zeros((3, H, W), dtype=np.float32)
    chw[0] = 0.25
    chw[1] = 0.50
    chw[2] = 0.75
    header = _cube_header(sip=False, pc=False, wcsaxes=False, cunit2=False)
    path = _write(tmp_path / "decode.fits", chw, header)

    desc = SimpleNamespace(source_path=str(path))
    hwc = _decode_frame_hwc(desc, None)

    assert hwc.shape == (H, W, 3)
    assert hwc.dtype == np.float32
    assert np.allclose(hwc[..., 0], 0.25)
    assert np.allclose(hwc[..., 1], 0.50)
    assert np.allclose(hwc[..., 2], 0.75)


def test_build_frame_descriptors_serializes_qualified_2d_wcs(tmp_path: Path) -> None:
    from zemosaic.zemosaic_zegrid_mode import _build_frame_descriptors

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    header = _cube_header(sip=True, pc=False, wcsaxes=False, cunit2=False)
    path = _write(in_dir / "frame.fits", _chw(), header)

    fi = zsp.FrameInfo(path=path)
    descs, rejected = _build_frame_descriptors([fi], in_dir, None)

    assert rejected == []
    assert len(descs) == 1
    desc = descs[0]
    assert desc.shape_hw == (H, W)
    reloaded = desc.wcs()
    assert reloaded.pixel_n_dim == 2
    assert reloaded.is_celestial is True
    assert zg.qualify_wcs(reloaded) is None
