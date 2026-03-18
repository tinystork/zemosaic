#!/usr/bin/env python3
"""Quick CPU vs GPU Phase 5 coadd comparison on synthetic tiles.

By default this uses a lightweight stub of ``reproject_and_coadd_wrapper`` so you
can run it without a GPU or heavy astropy/reproject machinery. It exercises
``assemble_final_mosaic_reproject_coadd`` twice (CPU then GPU) and reports
per-channel medians and max absolute differences so you can see if the green tint
appears right after the first reprojection/coadd.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import zemosaic_worker as zw  # noqa: E402
import zemosaic_utils  # noqa: E402


def _make_wcs():
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.crpix = [2.0, 2.0]
    w.wcs.cdelt = np.array([-0.1, 0.1])
    w.wcs.crval = [0.0, 0.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (4, 4)
    return w


def _write_tile(arr_hwc: np.ndarray, name: str, tmpdir: Path):
    from astropy.io import fits

    fits_data = np.moveaxis(arr_hwc, -1, 0)  # HWC -> CHW for FITS
    path = tmpdir / f"{name}.fits"
    fits.writeto(path, fits_data, header=_make_wcs().to_header(), overwrite=True)
    return str(path), _make_wcs()


def _maybe_patch_wrapper(use_stub: bool):
    if not use_stub:
        return None
    records: list[dict[str, Any]] = []

    def _stub_reproject_and_coadd_wrapper(data_list, wcs_list, shape_out, *, use_gpu=False, tile_weights=None, tile_affine_corrections=None, match_background=True, **kwargs):
        """Minimal combine: apply optional affine, optional background median removal, average."""
        captured = {
            "use_gpu": bool(use_gpu),
            "tile_weights": list(tile_weights) if tile_weights is not None else None,
            "tile_affine": tile_affine_corrections,
            "match_background": bool(match_background),
        }
        records.append(captured)
        arrays = []
        for idx, arr in enumerate(data_list):
            arr_np = np.asarray(arr, dtype=np.float32)
            if tile_affine_corrections is not None:
                try:
                    gain, offset = tile_affine_corrections[idx]
                except Exception:
                    gain, offset = (1.0, 0.0)
                arr_np = arr_np * float(gain) + float(offset)
            if match_background:
                med = float(np.nanmedian(arr_np))
                arr_np = arr_np - med
            arrays.append(arr_np)
        if tile_weights is not None:
            weights = np.asarray(tile_weights, dtype=np.float32)
            weights = np.where(np.isfinite(weights), weights, 1.0)
            weights = np.clip(weights, 1e-6, None)
            weighted = np.stack([a * w for a, w in zip(arrays, weights)], axis=0)
            mosaic = np.sum(weighted, axis=0) / np.sum(weights)
        else:
            mosaic = np.mean(np.stack(arrays, axis=0), axis=0)
        coverage = np.mean([np.isfinite(a).astype(np.float32) for a in arrays], axis=0)
        return mosaic.astype(np.float32), coverage.astype(np.float32)

    zemosaic_utils.reproject_and_coadd_wrapper = _stub_reproject_and_coadd_wrapper
    return records


def _run_once(tile_paths_wcs, *, use_gpu: bool, match_bg: bool, stubbed: bool):
    mosaic, cov, alpha = zw.assemble_final_mosaic_reproject_coadd(
        master_tile_fits_with_wcs_list=tile_paths_wcs,
        final_output_wcs=_make_wcs(),
        final_output_shape_hw=(4, 4),
        progress_callback=None,
        n_channels=3,
        match_bg=match_bg,
        use_gpu=use_gpu,
        collect_tile_data=[] if stubbed else None,
        intertile_photometric_match=False,
    )
    return mosaic, cov, alpha


def _channel_stats(arr: np.ndarray) -> dict[str, Any]:
    med = np.nanmedian(arr, axis=(0, 1))
    mn = np.nanmin(arr, axis=(0, 1))
    mx = np.nanmax(arr, axis=(0, 1))
    return {"median": med, "min": mn, "max": mx}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare CPU vs GPU Phase5 coadd on synthetic tiles.")
    parser.add_argument("--no-match-bg", action="store_true", help="Disable match_background for both paths.")
    parser.add_argument("--use-real-wrapper", action="store_true", help="Use real reproject_and_coadd_wrapper instead of stub (needs GPU/astropy).")
    args = parser.parse_args(argv or sys.argv[1:])

    match_bg = not args.no_match_bg
    stubbed = not args.use_real_wrapper
    records = _maybe_patch_wrapper(use_stub=stubbed)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        # Tile A: neutral with slight gradient; Tile B: green-biased to expose tint differences.
        yy, xx = np.mgrid[0:4, 0:4].astype(np.float32)
        gradient = (xx + yy) * 5.0  # small slope to avoid zeroing when background-matched
        base_a = np.array([500.0, 500.0, 500.0], dtype=np.float32)
        base_b = np.array([400.0, 800.0, 400.0], dtype=np.float32)
        tile_a = base_a + gradient[..., None]
        tile_b = base_b + gradient[..., None]
        t1, w1 = _write_tile(tile_a, "tile_a", tmpdir)
        t2, w2 = _write_tile(tile_b, "tile_b", tmpdir)
        tiles = [(t1, w1), (t2, w2)]

        cpu_mosaic, _, _ = _run_once(tiles, use_gpu=False, match_bg=match_bg, stubbed=stubbed)
        gpu_mosaic, _, _ = _run_once(tiles, use_gpu=True, match_bg=match_bg, stubbed=stubbed)

    diff = np.abs(cpu_mosaic - gpu_mosaic)
    print(f"Match background: {match_bg}")
    print("CPU channel stats:", _channel_stats(cpu_mosaic))
    print("GPU channel stats:", _channel_stats(gpu_mosaic))
    print("Max abs diff per channel:", np.nanmax(diff, axis=(0, 1)))
    print("Overall max abs diff:", float(np.nanmax(diff)))
    if records is not None:
        print("Captured wrapper calls (in order):")
        for idx, rec in enumerate(records, 1):
            print(f"  Call {idx}: {rec}")
    return 0


if __name__ == "__main__":  # pragma: no cover - diagnostic entry point
    raise SystemExit(main())
