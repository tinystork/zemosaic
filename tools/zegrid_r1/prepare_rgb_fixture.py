#!/usr/bin/env python3
"""ZM-ZEGRID-R1 — prepare immutable prepared RGB FITS fixtures (reproducible).

Uses the EXISTING shared decode path (no new science):
    ``zemosaic.grid_mode._load_image_with_optional_alpha``
which internally calls ``zemosaic_utils.load_and_validate_fits`` (BZERO/BSCALE,
axis normalization, nonfinite repair) then ``zemosaic_utils.debayer_image``
(OpenCV Bayer GRBG -> RGB) with full-frame min/max normalization, then rescales
back to ADU. This is the exact production decoder reused verbatim.

Prepared FITS contract (consumed by ``core/zegrid/execution.py``):
* Primary HDU, float32, NAXIS=3, NAXIS1=W, NAXIS2=H, NAXIS3=3 (CHW on disk).
* The raw 2-D celestial WCS is preserved on axes 1-2 (WCS(header).celestial).
* Logical RGB is H x W x 3 = 1920 x 1080 x 3.

Full-frame preparation I/O is SEPARATE and NOT counted as local execution
(R0 report section P option (a)). It reads full raws once; the MiniTile
execution path reads only sections.

Usage:
    PYTHONPATH=src .venv/bin/python tools/zegrid_r1/prepare_rgb_fixture.py \
        --input /home/tristan/M106/lights --output /tmp/zegrid_r1_fixtures \
        [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from zemosaic import grid_mode as gm  # noqa: E402  (existing shared decode path)


def prepare_one(raw_path: Path, out_path: Path) -> dict:
    """Decode one raw frame via the shared path and write a prepared RGB FITS."""
    # Existing shared decoder: full-frame Bayer -> HWC float32 RGB (ADU).
    rgb_hwc, alpha = gm._load_image_with_optional_alpha(raw_path)
    if rgb_hwc.ndim != 3 or rgb_hwc.shape[-1] != 3:
        raise ValueError(f"unexpected decoded shape {rgb_hwc.shape}")
    # CHW on disk so the 2-D celestial WCS stays on axes 1-2.
    chw = np.ascontiguousarray(np.moveaxis(rgb_hwc, -1, 0).astype(np.float32))

    raw_header = fits.getheader(raw_path, 0)
    out_header = raw_header.copy()
    # Preserve the raw WCS verbatim; declare the 3-D channel axis.
    out_header["NAXIS"] = 3
    out_header["NAXIS1"] = int(chw.shape[2])  # W
    out_header["NAXIS2"] = int(chw.shape[1])  # H
    out_header["NAXIS3"] = 3  # channels
    out_header["BITPIX"] = -32
    out_header["BZERO"] = 0.0
    out_header["BSCALE"] = 1.0
    out_header["ZEGRIDAX"] = ("CHW", "channels-first float32 RGB; WCS on axes 1-2")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(chw, header=out_header).writeto(out_path, overwrite=True)

    raw_hash = hashlib.sha256(Path(raw_path).read_bytes()).hexdigest()
    out_hash = hashlib.sha256(Path(out_path).read_bytes()).hexdigest()
    return {
        "raw": raw_path.name,
        "raw_sha256": raw_hash,
        "prepared": str(out_path),
        "prepared_sha256": out_hash,
        "shape_hwc": list(rgb_hwc.shape),
        "dtype": str(rgb_hwc.dtype),
        "alpha_present": alpha is not None,
        "decoder": "zemosaic.grid_mode._load_image_with_optional_alpha "
        "(load_and_validate_fits + debayer_image GRBG->RGB)",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--frames", type=str, default=None,
                    help="comma-separated basenames to prepare (default: all)")
    args = ap.parse_args()

    raw_files = sorted(
        p
        for p in args.input.resolve().rglob("*")
        if p.suffix.lower() in (".fit", ".fits", ".fts")
    )
    if args.frames:
        wanted = {n.strip() for n in args.frames.split(",") if n.strip()}
        raw_files = [p for p in raw_files if p.name in wanted]
    if args.limit is not None:
        raw_files = raw_files[: args.limit]

    out_dir = args.output
    manifest_path = out_dir / "prepared_manifest.json"
    manifest = {"schema": "ZM-ZEGRID-R1-prepared-v1", "frames": []}
    for i, raw in enumerate(raw_files, 1):
        t0 = time.time()
        rec = prepare_one(raw, out_dir / (raw.stem + "_rgb.fits"))
        rec["seconds"] = round(time.time() - t0, 3)
        manifest["frames"].append(rec)
        print(f"[{i}/{len(raw_files)}] {raw.name} -> {rec['prepared_sha256'][:12]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
