from __future__ import annotations

import numpy as np
import pytest

from zemosaic_utils import save_fits_image

fits = pytest.importorskip("astropy.io.fits")


def _random_rgb(height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((height, width, 3), dtype=np.float32)


def test_rgb_luminance_output(tmp_path):
    data = _random_rgb(10, 12)
    output = tmp_path / "rgb_luma.fits"

    save_fits_image(
        image_data=data,
        output_path=str(output),
        header=None,
        overwrite=True,
        save_as_float=False,
        legacy_rgb_cube=False,
        axis_order="HWC",
    )

    with fits.open(output, do_not_scale_image_data=True) as hdul:
        hdul.verify("exception")
        assert len(hdul) == 4
        primary = hdul[0]
        assert primary.header["BITPIX"] == 16
        assert primary.header["BZERO"] == 32768
        assert primary.header["NAXIS"] == 2
        assert np.issubdtype(primary.data.dtype, np.int16)
        assert primary.data.shape == (10, 12)
        assert primary.header["DATAMIN"] == int(primary.data.min())
        assert primary.header["DATAMAX"] == int(primary.data.max())

        names = [hdu.header.get("EXTNAME") for hdu in hdul[1:]]
        assert names == ["R", "G", "B"]
        for hdu in hdul[1:]:
            assert np.issubdtype(hdu.data.dtype, np.int16)
            assert hdu.data.shape == (10, 12)


def test_monochrome_output(tmp_path):
    data = np.linspace(0, 1, 50, dtype=np.float32).reshape(5, 10)
    output = tmp_path / "mono.fits"

    save_fits_image(
        image_data=data,
        output_path=str(output),
        header=None,
        overwrite=True,
        save_as_float=False,
        axis_order="HWC",
    )

    with fits.open(output, do_not_scale_image_data=True) as hdul:
        hdul.verify("exception")
        assert len(hdul) == 1
        primary = hdul[0]
        assert primary.header["NAXIS"] == 2
        assert primary.header["NAXIS1"] == 10
        assert primary.header["NAXIS2"] == 5
        assert primary.data.shape == (5, 10)
        assert np.issubdtype(primary.data.dtype, np.int16)


def test_legacy_rgb_cube(tmp_path):
    data = _random_rgb(6, 8)
    output = tmp_path / "legacy_cube.fits"

    save_fits_image(
        image_data=data,
        output_path=str(output),
        header=None,
        overwrite=True,
        save_as_float=False,
        legacy_rgb_cube=True,
        axis_order="HWC",
    )

    with fits.open(output, do_not_scale_image_data=True) as hdul:
        hdul.verify("exception")
        assert len(hdul) == 1
        primary = hdul[0]
        assert primary.header["NAXIS"] == 3
        assert primary.header["NAXIS3"] == 3
        assert primary.header.get("CTYPE3") == "RGB"
        assert primary.header.get("EXTNAME") == "RGB"
        assert primary.data.shape == (3, 6, 8)
        assert np.issubdtype(primary.data.dtype, np.int16)


def test_nan_inf_handled(tmp_path):
    data = _random_rgb(4, 4)
    data[0, 0, 0] = np.nan
    data[1, 1, 1] = np.inf
    data[2, 2, 2] = -np.inf
    output = tmp_path / "nan_inf.fits"

    save_fits_image(
        image_data=data,
        output_path=str(output),
        header=None,
        overwrite=True,
        save_as_float=False,
        legacy_rgb_cube=False,
        axis_order="HWC",
    )

    with fits.open(output, do_not_scale_image_data=True) as hdul:
        hdul.verify("exception")
        for hdu in hdul:
            assert np.isfinite(hdu.data).all()


def test_constant_range_produces_black(tmp_path):
    data = np.ones((3, 3, 3), dtype=np.float32)
    output = tmp_path / "constant.fits"

    save_fits_image(
        image_data=data,
        output_path=str(output),
        header=None,
        overwrite=True,
        save_as_float=False,
        legacy_rgb_cube=False,
        axis_order="HWC",
    )

    with fits.open(output, do_not_scale_image_data=True) as hdul:
        primary = hdul[0]
        assert np.all(primary.data == -32768)
        assert primary.header["DATAMIN"] == -32768
        assert primary.header["DATAMAX"] == -32768
        for hdu in hdul[1:]:
            assert np.all(hdu.data == -32768)
