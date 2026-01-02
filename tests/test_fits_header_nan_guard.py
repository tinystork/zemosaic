import os
import sys

import pytest

pytest.importorskip("astropy")
from astropy.io import fits

# Ensure repository root is on sys.path so tests can import module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zemosaic_worker import _coerce_finite_float


def test_astropy_header_rejects_nan():
    header = fits.Header()
    with pytest.raises(ValueError):
        header["RGBEQMED"] = float("nan")


def test_coerce_finite_float_skips_nan_in_header():
    header = fits.Header()
    header["RGBGAINR"] = (_coerce_finite_float(float("nan"), 1.0), "gain red")
    target_hdr = _coerce_finite_float(float("nan"), None)
    if target_hdr is not None:
        header["RGBEQMED"] = (target_hdr, "target median")

    assert header["RGBGAINR"] == 1.0
    assert "RGBEQMED" not in header

