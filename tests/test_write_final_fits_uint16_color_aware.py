from __future__ import annotations

import numpy as np
import pytest

from zemosaic_utils import write_final_fits_uint16_color_aware

fits = pytest.importorskip("astropy.io.fits")


def test_float_data_with_large_range_preserves_structure(tmp_path):
    data = np.linspace(0, 12000, 12, dtype=np.float32).reshape(3, 4)
    output = tmp_path / "mono_uint16.fits"

    write_final_fits_uint16_color_aware(
        str(output),
        data,
        header=None,
        force_rgb_planes=False,
        legacy_rgb_cube=False,
        overwrite=True,
    )

    with fits.open(output, do_not_scale_image_data=True) as hdul:
        hdul.verify("exception")
        primary = hdul[0]
        assert primary.data.shape == (3, 4)
        # Stored values are signed with a 32768 offset.
        assert int(primary.data.min()) == -32768
        assert int(primary.data.max()) == -20768
        header = primary.header
        assert header["BITPIX"] == 16
        assert header["BSCALE"] == 1
        assert header["BZERO"] == 32768
        assert "DATAMIN" not in header
        assert "DATAMAX" not in header


def test_rgb_planes_are_reordered(tmp_path):
    data = np.stack([
        np.full((2, 3), fill_value=1000, dtype=np.float32),
        np.full((2, 3), fill_value=2000, dtype=np.float32),
        np.full((2, 3), fill_value=3000, dtype=np.float32),
    ], axis=-1)
    output = tmp_path / "rgb_uint16.fits"

    write_final_fits_uint16_color_aware(
        str(output),
        data,
        header=None,
        force_rgb_planes=True,
        legacy_rgb_cube=False,
        overwrite=True,
    )

    with fits.open(output, do_not_scale_image_data=True) as hdul:
        hdul.verify("exception")
        primary = hdul[0]
        assert primary.data.shape == (3, 2, 3)
        assert np.all(primary.data[0] == -31768)
        assert np.all(primary.data[1] == -30768)
        assert np.all(primary.data[2] == -29768)
        header = primary.header
        assert header["ZEMORGB"] is True
        assert header["CHANNELS"] == 3
        assert header["BZERO"] == 32768
