import os
import sys

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits

# Ensure repository root is on sys.path so tests can import module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from zemosaic_worker import EMPTY_MASTER_TILE_EPS_SIGNAL, load_image_with_optional_alpha


def _write_master_tile(path, data, alpha, header_updates=None):
    hdu = fits.PrimaryHDU(data=data)
    if header_updates:
        for key, value in header_updates.items():
            hdu.header[key] = value
    alpha_hdu = fits.ImageHDU(alpha, name="ALPHA")
    fits.HDUList([hdu, alpha_hdu]).writeto(path, overwrite=True)


def test_load_empty_master_tile_flag(tmp_path):
    data = np.zeros((4, 4, 3), dtype=np.float32)
    alpha = np.full((4, 4), 255, dtype=np.uint8)
    path = tmp_path / "mt_empty_flag.fits"
    _write_master_tile(
        path,
        data,
        alpha,
        {
            "ZMT_TYPE": "Master Tile",
            "ZMT_ID": 1,
            "ZMT_EMPT": 1,
        },
    )

    data_out, weights, alpha_out = load_image_with_optional_alpha(str(path))

    assert weights is not None
    assert alpha_out is not None
    assert np.all(weights == 0)
    assert np.all(alpha_out == 0)
    assert np.all(np.isnan(data_out))


def test_load_empty_master_tile_fallback(tmp_path):
    data = np.full((4, 4, 3), EMPTY_MASTER_TILE_EPS_SIGNAL * 0.1, dtype=np.float32)
    alpha = np.full((4, 4), 255, dtype=np.uint8)
    path = tmp_path / "mt_empty_fallback.fits"
    _write_master_tile(
        path,
        data,
        alpha,
        {
            "ZMT_TYPE": "Master Tile",
            "ZMT_ID": 2,
        },
    )

    data_out, weights, alpha_out = load_image_with_optional_alpha(str(path))

    assert weights is not None
    assert alpha_out is not None
    assert np.all(weights == 0)
    assert np.all(alpha_out == 0)
    assert np.all(np.isnan(data_out))


def test_load_master_tile_non_empty(tmp_path):
    data = np.zeros((4, 4, 3), dtype=np.float32)
    data[0, 0, 0] = EMPTY_MASTER_TILE_EPS_SIGNAL * 100.0
    alpha = np.full((4, 4), 255, dtype=np.uint8)
    path = tmp_path / "mt_non_empty.fits"
    _write_master_tile(
        path,
        data,
        alpha,
        {
            "ZMT_TYPE": "Master Tile",
            "ZMT_ID": 3,
        },
    )

    data_out, weights, alpha_out = load_image_with_optional_alpha(str(path))

    assert weights is not None
    assert alpha_out is not None
    assert np.any(weights > 0)
    assert np.any(alpha_out > 0)
    assert np.any(np.isfinite(data_out))
