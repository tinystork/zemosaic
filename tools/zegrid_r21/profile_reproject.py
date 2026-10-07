"""ZM-ZEGRID-R21 — reproducible reprojection profiling (BEFORE vs AFTER).

Measures the R21 fast path (``execution.reproject_cropped``) against the pre-R21
path (``reproject_interp``) on REAL M16 / M106 frames, and re-confirms bit-equality
on those real frames.  Run from the repo root:

    python tools/zegrid_r21/profile_reproject.py [--frames N]

Outputs: per-case wall-clock (old per-channel reproject_interp, the old 3-channel
loop cost, the new fast-path cost), the per-pixel costs, the speedup, and the
bit-equality max|diff| (must be 0.0 on the fast path).

No new dependency; scipy/astropy/reproject already used by the engine.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from zemosaic.core.zegrid import execution as zx

M106_DIR = Path("/home/tristan/M106/lights")
M16_DIR = Path("/home/tristan/M16/quick")


def _load_light(dirpath: Path, n: int):
    fits_files = sorted(p for p in dirpath.iterdir() if p.suffix.lower() == ".fit")
    return fits_files[:n]


def _frame_data_wcs(path: Path):
    with fits.open(path) as hdul:
        hdu = hdul[0]
        data = np.asarray(hdu.data, dtype=np.float32)
        w = WCS(hdu.header)
    # mono -> 3 identical channels (CHW) so the fast path sees a 3-channel plane
    chw = np.ascontiguousarray(np.stack([data, data, data], axis=0))
    return chw, w


def _make_patch_wcs(w_in, out_shape, angle_deg=15.0, scale=1.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [float(w_in.wcs.crval[0]), float(w_in.wcs.crval[1])]
    h, wdt = out_shape
    w.wcs.crpix = [wdt / 2.0, h / 2.0]
    a = np.deg2rad(angle_deg)
    base_cd = np.array(
        [[-np.cos(a), np.sin(a)], [np.sin(a), np.cos(a)]]
    )
    # derive the pixel scale from the input WCS (works for CD and PC+CDELT alike)
    s = abs(float(w_in.pixel_scale_matrix[0, 0])) * scale
    w.wcs.cd = base_cd * s
    w.array_shape = out_shape
    return w


def _timeit(fn, n=3):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _old_reproject_single(plane, w_in, w_out, shape_out):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arr, fp = reproject_interp(
            (plane, w_in), output_projection=w_out, shape_out=shape_out,
            order="bilinear", return_footprint=True,
        )
    return arr, fp


def measure_case(name, chw, w_in, w_out, shape_out, n=3):
    plane = chw[0]
    # OLD: per-channel reproject_interp; the old reproject_cropped loops 3 channels.
    t_old_single = _timeit(lambda: _old_reproject_single(plane, w_in, w_out, shape_out), n)
    t_old_3ch = t_old_single * 3.0  # the old loop calls reproject_interp 3x

    # NEW: the R21 fast path (mapping computed once, 3 channels interpolated).
    zx.reset_reproject_path_stats()
    t_new = _timeit(lambda: zx.reproject_cropped(chw, w_in, w_out, shape_out), n)

    # Bit-equality on the real frame: the OLD reproject_cropped casts the
    # reproject_interp float64 result to float32; the NEW fast path does the same.
    # Compare float32-vs-float32 (the actual OLD-vs-NEW output) — must be 0.0.
    ref, _ = _old_reproject_single(plane, w_in, w_out, shape_out)
    rgb, geom = zx.reproject_cropped(chw, w_in, w_out, shape_out)
    fast_plane = np.asarray(rgb[..., 0], dtype=np.float32)
    ref32 = np.asarray(ref, dtype=np.float32)
    equal = np.array_equal(fast_plane, ref32, equal_nan=True)
    maxdiff = float(np.nanmax(np.abs(np.asarray(fast_plane, dtype=np.float64) - np.asarray(ref32, dtype=np.float64)))) if not equal else 0.0

    npx = shape_out[0] * shape_out[1]
    stats = zx.get_reproject_path_stats().to_dict()
    print(f"[{name}] output={shape_out} ({npx/1e6:.2f} Mpx)")
    print(f"  old per-channel reproject_interp : {t_old_single:.4f}s  ({t_old_single/npx*1e6:.3f} us/px)")
    print(f"  old 3-channel loop               : {t_old_3ch:.4f}s")
    print(f"  NEW fast path (3ch)              : {t_new:.4f}s  ({t_new/npx*1e6:.3f} us/px)")
    print(f"  speedup (vs old 3ch loop)        : {t_old_3ch/t_new:.2f}x")
    print(f"  path={stats['path']}  bit-equal max|diff|={maxdiff:.3e}")
    print()
    return dict(old_3ch=t_old_3ch, new=t_new, speedup=t_old_3ch / t_new, maxdiff=maxdiff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--cases", choices=["patch", "canvas", "all"], default="patch")
    args = ap.parse_args()

    print("=== R21 reprojection profiling (real frames) ===\n")

    for label, d in (("M106", M106_DIR), ("M16", M16_DIR)):
        if not d.is_dir():
            print(f"[{label}] corpus not present: {d}")
            continue
        for path in _load_light(d, args.frames):
            chw, w_in = _frame_data_wcs(path)
            print(f"--- {label} {path.name}  frame={chw.shape} wcs={tuple(w_in.wcs.ctype)} ---\n")

            patch_w = _make_patch_wcs(w_in, (1200, 1600), angle_deg=15.0, scale=1.0)
            if args.cases in ("patch", "all"):
                measure_case(f"{label} patch 1200x1600", chw, w_in, patch_w, (1200, 1600))

            if args.cases in ("canvas", "all"):
                canvas_w = _make_patch_wcs(w_in, (3278, 2403), angle_deg=15.0, scale=0.6)
                measure_case(f"{label} canvas 2403x3278", chw, w_in, canvas_w, (3278, 2403))


if __name__ == "__main__":
    main()
