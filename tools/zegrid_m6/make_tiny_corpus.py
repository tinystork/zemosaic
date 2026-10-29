#!/usr/bin/env python3
"""Generate the M6 tiny deterministic corpus for the headless CLI checks.

4 mono FITS frames, 256x256, real undistorted TAN WCS, dithered on a 2x2 grid
so the mosaic spans exactly two frames in RA and Dec. The corpus is seeded and
bit-deterministic; the intended pinned layout is ``2x2``.

Usage::

    python tools/zegrid_m6/make_tiny_corpus.py <output_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

N_FRAMES = 4
FRAME_HW = (256, 256)
SCALE_DEG = 0.001
EXPOSURE = 10.0
SEED = 20261029


def _make_wcs(ra_deg: float, dec_deg: float) -> WCS:
    h, w = FRAME_HW
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra_deg, dec_deg]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * SCALE_DEG
    wcs.array_shape = (h, w)
    return wcs


def _frame_data(rng: np.random.Generator, offset: float) -> np.ndarray:
    h, w = FRAME_HW
    yy, xx = np.mgrid[0:h, 0:w]
    base = 100.0 + 20.0 * np.exp(
        -((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2) / (2.0 * 18.0**2)
    )
    data = base + offset + rng.normal(0.0, 0.5, (h, w))
    return data.astype(np.float32)


def make_tiny_corpus(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ra0, dec0 = 12.0, 34.0
    # ~10 % overlap on each axis: the 4-frame footprint is just wide enough
    # for a valid pinned 2x2 layout (patch core >= the 50 000 px scientific
    # floor at halo 8) while every frame keeps real common area with its
    # neighbours (a full-frame step leaves zero overlap and excludes frames).
    d_ra = 0.9 * FRAME_HW[1] * SCALE_DEG
    d_dec = 0.9 * FRAME_HW[0] * SCALE_DEG
    for i in range(N_FRAMES):
        col, row = i % 2, i // 2
        ra = ra0 + col * d_ra
        dec = dec0 - row * d_dec
        wcs = _make_wcs(ra, dec)
        data = _frame_data(rng, float(i))
        hdu = fits.PrimaryHDU(data)
        hdu.header.update(wcs.to_header(relax=True))
        hdu.header["EXPTIME"] = EXPOSURE
        hdu.writeto(out / f"frame_{i:04d}.fits", overwrite=True)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    print(make_tiny_corpus(sys.argv[1]))
