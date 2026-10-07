#!/usr/bin/env python3
"""Generate a tiny DETERMINISTIC on-disk corpus for routine ZeGrid gated checks.

The routine corpus replaces a full M16/M106 pipeline run for the everyday loop
gate: 12 small mono FITS frames with a real undistorted TAN WCS, dithered so
they tile a small patch and produce a genuine mosaic (multi-cell, multi-frame).
Everything is seeded and bit-deterministic, so the resulting mosaic is
reproducible across machines (unlike the frozen M16/M106 corpora, which live
outside git and are the human/release gate).

Usage::

    python tools/zegrid_routine/make_routine_corpus.py <output_dir>

Writes ``<output_dir>/{stack_plan.csv, frame_XXXX.fits, ...}`` — the exact
``stack_plan.csv`` layout the production ``run_zegrid_mode`` entry consumes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

N_FRAMES = 12
FRAME_HW = (384, 384)          # (height, width) px
SCALE_DEG = 0.001              # deg/px
EXPOSURE = 10.0
SEED = 20261007


def _make_wcs(ra_deg: float, dec_deg: float) -> WCS:
    """Undistorted RA---TAN / DEC--TAN WCS (what ``qualify_wcs`` accepts)."""
    h, w = FRAME_HW
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra_deg, dec_deg]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * SCALE_DEG
    wcs.array_shape = (h, w)
    return wcs


def _frame_data(rng: np.random.Generator, offset: float) -> np.ndarray:
    """Deterministic smooth 2-D field (sky gradient + noise), float32."""
    h, w = FRAME_HW
    yy, xx = np.mgrid[0:h, 0:w]
    base = 100.0 + 20.0 * np.exp(
        -((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2) / (2.0 * 24.0**2)
    )
    data = base + offset + rng.normal(0.0, 0.5, (h, w))
    return data.astype(np.float32)


def make_routine_corpus(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    # Dither: a 3x4 grid of ~64 px RA / ~96 px Dec steps -> partial overlap.
    ra0, dec0 = 10.0, 30.0
    d_ra = 192.0 * SCALE_DEG
    d_dec = 192.0 * SCALE_DEG
    offsets = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0]

    names = []
    for i in range(N_FRAMES):
        col, row = i % 4, i // 4
        ra = ra0 + col * d_ra
        dec = dec0 - row * d_dec
        name = f"frame_{i:04d}.fits"
        wcs = _make_wcs(ra, dec)
        data = _frame_data(rng, offsets[i])
        hdu = fits.PrimaryHDU(data)
        hdu.header.update(wcs.to_header(relax=True))
        hdu.header["EXPTIME"] = EXPOSURE
        hdu.writeto(out / name, overwrite=True)
        names.append(name)

    (out / "stack_plan.csv").write_text(
        "file_path,exposure\n" + "".join(f"{n},{EXPOSURE}\n" for n in names),
        encoding="utf-8",
    )
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    print(make_routine_corpus(sys.argv[1]))
