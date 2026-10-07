"""ZM-ZEGRID-R21 targeted tests — fast reprojection path (bit-equality + fallback).

These tests pin the R21 fast path in ``execution.reproject_cropped``:

* the bit-equality matrix (translation / rotation / scale / edge overhang / NaN
  input / tiny patch / non-contiguous input / TAN singularity) — the fast path
  MUST reproduce ``reproject_interp`` bit-for-bit (NaN-aware);
* the fallback path (non-TAN WCS and a forced fast-path error) — the result
  stays bit-equal to ``reproject_interp`` and the fallback is recorded + surfaced;
* the diagnostics field (process-local ``ReprojectPathStats`` + the merge helper).

All cases are synthetic (no corpora), so they run in the FAST tier.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy.wcs import WCS
from reproject import reproject_interp

from zemosaic.core.zegrid import execution as zx


def _make_wcs(crval, crpix, cd, shape, ctype=("RA---TAN", "DEC--TAN")):
    w = WCS(naxis=2)
    w.wcs.ctype = list(ctype)
    w.wcs.crval = list(crval)
    w.wcs.crpix = list(crpix)
    w.wcs.cd = np.array(cd)
    w.array_shape = shape
    return w


def _reference_reproject(chw, wcs_in, wcs_out, shape_out):
    """The pre-R21 behaviour: reproject_interp per channel + footprint -> geom."""
    c = chw.shape[0]
    out = np.empty((shape_out[0], shape_out[1], c), dtype=np.float32)
    support = None
    for ch in range(c):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr, fp = reproject_interp(
                (chw[ch], wcs_in), output_projection=wcs_out, shape_out=shape_out,
                order="bilinear", return_footprint=True,
            )
        out[..., ch] = np.asarray(arr, dtype=np.float32)
        if support is None:
            support = np.asarray(fp, dtype=np.float32)
    geom = support > 0.0
    out[~geom] = np.nan
    return out, geom


def _assert_bit_equal(chw, wcs_in, wcs_out, shape_out):
    ref_rgb, ref_geom = _reference_reproject(chw, wcs_in, wcs_out, shape_out)
    zx.reset_reproject_path_stats()
    rgb, geom = zx.reproject_cropped(chw, wcs_in, wcs_out, shape_out)
    assert np.array_equal(ref_rgb, rgb, equal_nan=True)
    assert np.array_equal(ref_geom, geom)
    stats = zx.get_reproject_path_stats().to_dict()
    assert stats["path"] == "fast"
    assert stats["fast_calls"] == 1
    assert stats["fallback_calls"] == 0


# ---------------------------------------------------------------------------
# Bit-equality matrix
# ---------------------------------------------------------------------------

_BASE = (40, 60)
_W_IN = _make_wcs([10.0, 30.0], [30.0, 20.0], [[-0.001, 0], [0, 0.001]], _BASE)
_RNG = np.random.default_rng(0)
_CHW = _RNG.standard_normal((3,) + _BASE).astype(np.float32)
_ROT = np.array(
    [[-np.cos(np.deg2rad(27.0)), np.sin(np.deg2rad(27.0))],
     [np.sin(np.deg2rad(27.0)), np.cos(np.deg2rad(27.0))]]
) * 0.001


def test_bit_equal_translation():
    _assert_bit_equal(
        _CHW, _W_IN,
        _make_wcs([10.0, 30.0], [33.5, 17.5], [[-0.001, 0], [0, 0.001]], (40, 60)),
        (40, 60),
    )


def test_bit_equal_rotation():
    _assert_bit_equal(
        _CHW, _W_IN,
        _make_wcs([10.0, 30.0], [30.0, 20.0], _ROT, (55, 70)),
        (55, 70),
    )


def test_bit_equal_scale():
    _assert_bit_equal(
        _CHW, _W_IN,
        _make_wcs([10.0, 30.0], [30.0, 20.0], [[-0.002, 0], [0, 0.002]], (20, 30)),
        (20, 30),
    )


def test_bit_equal_edge_overhang():
    _assert_bit_equal(
        _CHW, _W_IN,
        _make_wcs([10.0, 30.0], [30.0 - 45.0, 20.0 + 10.0], [[-0.001, 0], [0, 0.001]], (40, 60)),
        (40, 60),
    )


def test_bit_equal_nan_input():
    chw = _CHW.copy()
    chw[:, 5:8, 10:12] = np.nan
    _assert_bit_equal(
        chw, _W_IN,
        _make_wcs([10.0, 30.0], [30.0, 20.0], _ROT, (55, 70)),
        (55, 70),
    )


def test_bit_equal_tiny_patch():
    _assert_bit_equal(
        _CHW, _W_IN,
        _make_wcs([10.0, 30.0], [30.0, 20.0], [[-0.001, 0], [0, 0.001]], (3, 4)),
        (3, 4),
    )


def test_bit_equal_noncontiguous_input():
    _assert_bit_equal(
        np.asfortranarray(_CHW), _W_IN,
        _make_wcs([10.0, 30.0], [30.0, 20.0], _ROT, (55, 70)),
        (55, 70),
    )


def test_bit_equal_tan_singularity_all_nan():
    # Output WCS whose tangent point is 50 deg away -> every output pixel maps
    # beyond the TAN projection's valid domain (all-NaN output).
    w_sing = _make_wcs([60.0, 30.0], [30.0, 20.0], [[-0.001, 0], [0, 0.001]], (40, 60))
    ref_rgb, ref_geom = _reference_reproject(_CHW, _W_IN, w_sing, (40, 60))
    assert np.all(np.isnan(ref_rgb))
    assert not ref_geom.any()
    zx.reset_reproject_path_stats()
    rgb, geom = zx.reproject_cropped(_CHW, _W_IN, w_sing, (40, 60))
    assert np.array_equal(ref_rgb, rgb, equal_nan=True)
    assert np.array_equal(ref_geom, geom)


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------

def test_fallback_on_non_tan_wcs():
    # A SIN (non-TAN) output WCS forces the fallback; the result must still be
    # bit-equal to reproject_interp and the fallback must be recorded.
    w_sin = _make_wcs(
        [10.0, 30.0], [30.0, 20.0], [[-0.001, 0], [0, 0.001]], (40, 60),
        ctype=("RA---SIN", "DEC--SIN"),
    )
    ref_rgb, ref_geom = _reference_reproject(_CHW, _W_IN, w_sin, (40, 60))
    zx.reset_reproject_path_stats()
    rgb, geom = zx.reproject_cropped(_CHW, _W_IN, w_sin, (40, 60))
    assert np.array_equal(ref_rgb, rgb, equal_nan=True)
    assert np.array_equal(ref_geom, geom)
    stats = zx.get_reproject_path_stats().to_dict()
    assert stats["path"] == "fallback"
    assert stats["fallback_calls"] == 1
    assert stats["fast_calls"] == 0
    assert stats["fallback_reason"]


def test_fallback_on_fast_path_error(monkeypatch):
    # A forced fast-path exception must degrade to reproject_interp (bit-equal)
    # and be surfaced via emit (loud) + recorded with the exception reason.
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic fast-path failure")

    monkeypatch.setattr(zx, "reproject_cropped_fast", boom)
    ref_rgb, ref_geom = _reference_reproject(_CHW, _W_IN, _W_IN, _BASE)

    emitted = []
    zx.reset_reproject_path_stats()
    rgb, geom = zx.reproject_cropped(
        _CHW, _W_IN, _W_IN, _BASE,
        emit=lambda msg, lvl="INFO": emitted.append((msg, lvl)),
    )
    assert np.array_equal(ref_rgb, rgb, equal_nan=True)
    assert np.array_equal(ref_geom, geom)
    stats = zx.get_reproject_path_stats().to_dict()
    assert stats["path"] == "fallback"
    assert stats["fallback_calls"] == 1
    assert "synthetic fast-path failure" in stats["fallback_reason"]
    assert any("FALLBACK" in msg for msg, _lvl in emitted)


# ---------------------------------------------------------------------------
# Diagnostics field
# ---------------------------------------------------------------------------

def test_stats_accumulate_and_reset():
    zx.reset_reproject_path_stats()
    _ = zx.reproject_cropped(_CHW, _W_IN, _W_IN, _BASE)  # fast
    _ = zx.reproject_cropped(_CHW, _W_IN, _W_IN, _BASE)  # fast
    stats = zx.get_reproject_path_stats().to_dict()
    assert stats["fast_calls"] == 2
    assert stats["fallback_calls"] == 0
    assert stats["path"] == "fast"

    zx.reset_reproject_path_stats()
    assert zx.get_reproject_path_stats().to_dict()["path"] == "unused"


def test_merge_reproject_stats():
    merged = zx.merge_reproject_stats(
        {"path": "fast", "fast_calls": 2, "fallback_calls": 0,
         "fast_seconds": 1.0, "fallback_seconds": 0.0, "fallback_reason": None},
        None,
        {"path": "mixed", "fast_calls": 1, "fallback_calls": 1,
         "fast_seconds": 0.5, "fallback_seconds": 0.25, "fallback_reason": "boom"},
    )
    assert merged["path"] == "mixed"
    assert merged["fast_calls"] == 3
    assert merged["fallback_calls"] == 1
    assert merged["fast_seconds"] == pytest.approx(1.5)
    assert merged["fallback_seconds"] == pytest.approx(0.25)
    assert merged["fallback_reason"] == "boom"


def test_plain_tan_wcs_gate():
    assert zx._plain_tan_wcs(_W_IN) is True
    assert zx._plain_tan_wcs(
        _make_wcs([10.0, 30.0], [30.0, 20.0], [[-0.001, 0], [0, 0.001]], (40, 60),
                  ctype=("RA---SIN", "DEC--SIN"))
    ) is False
