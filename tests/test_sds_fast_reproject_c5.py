"""ZM-SDS-R35C-C5 (part A) — SDS CPU reprojection fast path: parity + fallback.

Pins the contract of :mod:`zemosaic.core.sds_reproject`:

* BIT-EXACT parity of :func:`reproject_frame_patch_fast` against the VERBATIM
  historical worker loop (``reproject_interp`` per channel + footprint weight),
  on a synthetic TAN matrix (translation / rotation / scale / edge overhang /
  NaN input / mono C=1) and on a real M106 TAN frame (gated, read-only);
* the ADVERSARIAL fallback: a SIP-distorted frame WCS is routed to
  ``reproject_interp`` (never approximated), the result stays bit-equal and the
  R21 path stats record ``path == "fallback"``;
* the SDS gate (default ON for SDS, opt-out via plan / env kill switch);
* the worker wiring is present and gated (static guard).

All synthetic cases run in the FAST tier; the real-data case is skipped when the
M106 corpus is absent (CI / clean checkout).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS, Sip
from reproject import reproject_interp

from zemosaic.core import sds_reproject as sr
from zemosaic.core.zegrid import execution as zx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk_wcs(crval, crpix, cd, shape, ctype=("RA---TAN", "DEC--TAN")):
    w = WCS(naxis=2)
    w.wcs.ctype = list(ctype)
    w.wcs.crval = list(crval)
    w.wcs.crpix = list(crpix)
    w.wcs.cd = np.array(cd)
    w.array_shape = shape
    return w


def _legacy_reproject(frame, local_wcs, sub_wcs, h_patch, w_patch):
    """VERBATIM historical worker loop (``_reproject_frame`` inner block)."""
    planes: list[np.ndarray] = []
    footprint = None
    for ch in range(frame.shape[-1]):
        plane = frame[..., ch]
        if footprint is None:
            reproj_plane, footprint = reproject_interp(
                (plane, local_wcs), sub_wcs, shape_out=(h_patch, w_patch),
                return_footprint=True, order="bilinear", parallel=False,
            )
        else:
            reproj_plane = reproject_interp(
                (plane, local_wcs), sub_wcs, shape_out=(h_patch, w_patch),
                return_footprint=False, order="bilinear", parallel=False,
            )
        planes.append(np.asarray(reproj_plane, dtype=np.float32))
    if footprint is None:
        return None
    weight = np.clip(np.nan_to_num(footprint, nan=0.0), 0.0, 1.0).astype(np.float32, copy=False)
    patch = np.stack(planes, axis=-1)
    finite_mask = np.all(np.isfinite(patch), axis=-1)
    weight = np.where(finite_mask, weight, 0.0)
    if not np.any(weight > 0):
        return None
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return patch, weight


def _assert_parity(frame, local_wcs, sub_wcs, h, w):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = _legacy_reproject(frame, local_wcs, sub_wcs, h, w)
        fast = sr.reproject_frame_patch_fast(frame, local_wcs, sub_wcs, (h, w))
    if base is None:
        # The legacy loop refuses an empty support; the fast helper may either
        # decline (None -> legacy loop runs) or return it, and the CALLER then
        # rejects it on ``not any(weight > 0)``. Either way the outcome matches.
        assert fast is None or not np.any(fast[1] > 0)
        return
    assert fast is not None
    base_patch, base_weight = base
    fast_patch, fast_weight = fast
    assert fast_patch.shape == base_patch.shape
    assert fast_weight.shape == base_weight.shape
    # The binary footprint weight and the support are identical ...
    assert np.array_equal(base_weight, fast_weight)
    # ... and the science is bit-equal on the common (weight > 0) support.
    common = base_weight > 0
    assert np.array_equal(base_patch[common], fast_patch[common])
    assert float(np.max(np.abs(base_patch[common] - fast_patch[common]))) == 0.0


_BASE = (40, 60)
_W_IN = _mk_wcs([10.0, 30.0], [30.0, 20.0], [[-0.001, 0], [0, 0.001]], _BASE)
_RNG = np.random.default_rng(0)
_CHW = _RNG.standard_normal((3,) + _BASE).astype(np.float32)
_HWC = np.ascontiguousarray(np.moveaxis(_CHW, 0, -1))
_ROT = np.array(
    [[-np.cos(np.deg2rad(27.0)), np.sin(np.deg2rad(27.0))],
     [np.sin(np.deg2rad(27.0)), np.cos(np.deg2rad(27.0))]]
) * 0.001


# ---------------------------------------------------------------------------
# Bit-exact parity matrix (fast path)
# ---------------------------------------------------------------------------

def test_parity_translation():
    _assert_parity(
        _HWC, _W_IN,
        _mk_wcs([10.0, 30.0], [33.5, 17.5], [[-0.001, 0], [0, 0.001]], (40, 60)),
        40, 60,
    )


def test_parity_rotation():
    _assert_parity(
        _HWC, _W_IN, _mk_wcs([10.0, 30.0], [30.0, 20.0], _ROT, (55, 70)), 55, 70,
    )


def test_parity_scale():
    _assert_parity(
        _HWC, _W_IN,
        _mk_wcs([10.0, 30.0], [30.0, 20.0], [[-0.002, 0], [0, 0.002]], (20, 30)),
        20, 30,
    )


def test_parity_edge_overhang():
    _assert_parity(
        _HWC, _W_IN,
        _mk_wcs([10.0, 30.0], [30.0 - 45.0, 20.0 + 10.0], [[-0.001, 0], [0, 0.001]], (40, 60)),
        40, 60,
    )


def test_parity_tan_pc_cdelt():
    # TAN/PC+CDELT is explicitly fast-path-qualified.
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 30.0]
    w.wcs.crpix = [30.0, 20.0]
    w.wcs.pc = [[-1.0, 0.0], [0.0, 1.0]]
    w.wcs.cdelt = [0.001, 0.001]
    w.array_shape = (55, 70)
    assert zx._plain_tan_wcs(w) is True
    _assert_parity(_HWC, _W_IN, w, 55, 70)


def test_parity_nan_input():
    frame = _HWC.copy()
    frame[5:8, 10:12, :] = np.nan
    _assert_parity(frame, _W_IN, _mk_wcs([10.0, 30.0], [30.0, 20.0], _ROT, (55, 70)), 55, 70)


def test_parity_mono_c1():
    mono = np.ascontiguousarray(_HWC[..., :1])
    _assert_parity(mono, _W_IN, _mk_wcs([10.0, 30.0], [30.0, 20.0], _ROT, (55, 70)), 55, 70)


def test_parity_all_nan_output_no_support():
    # Output WCS whose tangent point is 50 deg away -> empty support on both.
    w_sing = _mk_wcs([60.0, 30.0], [30.0, 20.0], [[-0.001, 0], [0, 0.001]], (40, 60))
    _assert_parity(_HWC, _W_IN, w_sing, 40, 60)


# ---------------------------------------------------------------------------
# Adversarial fallback (SIP / non-TAN)
# ---------------------------------------------------------------------------

def _sip_wcs():
    w = _mk_wcs([10.0, 30.0], [30.0, 20.0], [[-0.001, 0], [0, 0.001]], _BASE)
    w.sip = Sip(
        np.array([[1e-5, 0.0], [0.0, 0.0]]),
        np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), [30.0, 20.0],
    )
    return w


def test_sip_frame_wcs_falls_back_and_is_bit_equal():
    win = _sip_wcs()
    assert zx._plain_tan_wcs(win) is False
    zx.reset_reproject_path_stats()
    _assert_parity(_HWC, win, _W_IN, 40, 60)
    stats = zx.get_reproject_path_stats().to_dict()
    assert stats["path"] == "fallback"
    assert stats["fallback_calls"] == 1
    assert stats["fast_calls"] == 0
    assert stats["fallback_reason"]


def test_non_tan_output_falls_back_and_is_bit_equal():
    w_sin = _mk_wcs(
        [10.0, 30.0], [30.0, 20.0], [[-0.001, 0], [0, 0.001]], (40, 60),
        ctype=("RA---SIN", "DEC--SIN"),
    )
    zx.reset_reproject_path_stats()
    _assert_parity(_HWC, _W_IN, w_sin, 40, 60)
    assert zx.get_reproject_path_stats().to_dict()["path"] == "fallback"


def test_fast_path_records_fast_when_qualified():
    zx.reset_reproject_path_stats()
    _assert_parity(_HWC, _W_IN, _mk_wcs([10.0, 30.0], [30.0, 20.0], _ROT, (55, 70)), 55, 70)
    assert zx.get_reproject_path_stats().to_dict()["path"] == "fast"


# ---------------------------------------------------------------------------
# Gate resolution (SDS default ON, opt-out via plan / env kill switch)
# ---------------------------------------------------------------------------

def test_resolve_default_on():
    assert sr.resolve_sds_fast_reproject(None) is True
    assert sr.resolve_sds_fast_reproject({}) is True


def test_resolve_plan_opt_out():
    assert sr.resolve_sds_fast_reproject({"sds_fast_reproject": False}) is False
    assert sr.resolve_sds_fast_reproject({"meta": {"sds_fast_reproject": False}}) is False
    assert sr.resolve_sds_fast_reproject({"sds_fast_reproject": "off"}) is False


def test_resolve_env_kill_switch(monkeypatch):
    monkeypatch.setenv(sr.SDS_REPROJECT_FORCE_FALLBACK_ENV, "1")
    assert sr.sds_fast_reproject_forced_off() is True
    assert sr.resolve_sds_fast_reproject(None) is False
    assert sr.resolve_sds_fast_reproject({"sds_fast_reproject": True}) is False


def test_invalid_frame_layout_returns_none():
    # 2-D input is not a valid HWC frame -> helper declines (legacy loop runs).
    assert sr.reproject_frame_patch_fast(np.zeros((10, 10), np.float32), _W_IN, _W_IN, (10, 10)) is None
    # Degenerate patch shape -> decline.
    assert sr.reproject_frame_patch_fast(_HWC, _W_IN, _W_IN, (0, 10)) is None


# ---------------------------------------------------------------------------
# Worker wiring (static guard)
# ---------------------------------------------------------------------------

def _worker_source() -> str:
    return (
        Path(__file__).resolve().parents[1] / "src" / "zemosaic" / "zemosaic_worker.py"
    ).read_text()


def test_worker_wires_sds_fast_reproject_and_is_gated():
    src = _worker_source()
    assert "zemosaic.core.sds_reproject" in src
    assert "reproject_frame_patch_fast" in src
    assert "resolve_sds_fast_reproject" in src
    # Gated on the SDS plan and opt-out-able (Classic/Legacy path unchanged).
    assert "sds_fast_reproject_enabled" in src
    assert "if sds_fast_reproject_enabled:" in src
    # The historical reproject_interp loop is KEPT as the fallback.
    assert 'parallel=False' in src
    assert 'order="bilinear"' in src
    # Observability of the path actually taken.
    assert "sds_fast_reproject_summary" in src


# ---------------------------------------------------------------------------
# Real M106 TAN frame (gated)
# ---------------------------------------------------------------------------

_M106 = Path("/home/tristan/M106")


def test_real_m106_tan_frame_parity():
    sds = _M106 / "OUT" / "R35_A" / "sds_fixed"
    if not sds.is_dir():
        pytest.skip("M106 SDS output not present")
    cache = sorted((sds / ".zemosaic_img_cache").glob("*230107*.npy"))
    raw = sorted((_M106 / "lights").glob("*230107.fit"))
    if not cache or not raw:
        pytest.skip("M106 frame/cache not present")
    arr = np.load(cache[0], mmap_mode="r").astype(np.float32)
    gw = WCS(fits.getheader(sds / "global_mosaic_wcs.fits")).celestial
    sw = WCS(fits.getheader(raw[0])).celestial
    h, w = arr.shape[:2]
    ra, dec = sw.pixel_to_world_values(w / 2, h / 2)
    cx, cy = gw.world_to_pixel_values(ra, dec)
    x0, y0 = int(cx) - 256, int(cy) - 256
    pw = gw.slice((slice(y0, y0 + 512), slice(x0, x0 + 512)))
    frame = np.ascontiguousarray(arr)
    assert zx._plain_tan_wcs(sw) is True
    zx.reset_reproject_path_stats()
    _assert_parity(frame, sw, pw, 512, 512)
    stats = zx.get_reproject_path_stats().to_dict()
    assert stats["path"] == "fast"
    assert stats["fast_calls"] == 1
