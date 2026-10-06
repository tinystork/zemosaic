#!/usr/bin/env python3
"""ZM-ZEGRID-R7 — evidence collector: run the production ZeGrid engine end-to-end
on a REAL small M106 subset and dump all acceptance evidence to a persistent dir.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

SRC = Path("/home/tristan/.openclaw/workspace/worktrees/zemosaic-zegrid-r1/src")
sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zegrid  # noqa: E402

LIGHTS = Path("/home/tristan/M106/lights")
N = 6
OUT = Path("/home/tristan/zegrid_r7_e2e_out")
INPUT = Path("/home/tristan/zegrid_r7_e2e_input")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if INPUT.exists():
        shutil.rmtree(INPUT)
    INPUT.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    frames = sorted(LIGHTS.glob("*.fit"))[:N]
    for p in frames:
        shutil.copy2(p, INPUT / p.name)

    lines = ["file_path,exposure"]
    for p in frames:
        lines.append(f"{p.name},10.0")
    (INPUT / "stack_plan.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    zegrid.run_zegrid_mode(str(INPUT), str(OUT))

    sci = OUT / "mosaic_grid.fits"
    cov = OUT / "mosaic_grid_coverage.fits"
    manifest = json.loads((OUT / "zegrid_manifest.json").read_text())

    with fits.open(sci) as h:
        sci_data = h[0].data
    with fits.open(cov) as h:
        cov_data = h[0].data

    cache_dir = OUT / zegrid.CACHE_DIR_NAME
    cache_total = sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())

    evidence = {
        "frames": [p.name for p in frames],
        "n_frames": N,
        "canvas": manifest["canvas"],
        "layout": manifest["layout"],
        "complete_cells": manifest["complete_cells"],
        "incomplete_cells": manifest["incomplete_cells"],
        "n_cells": manifest["layout"]["nx"] * manifest["layout"]["ny"],
        "cells": manifest["cells"],
        "science_fits_shape": list(sci_data.shape),
        "science_fits_dtype": str(sci_data.dtype),
        "coverage_fits_shape": list(cov_data.shape),
        "coverage_fits_dtype": str(cov_data.dtype),
        "coverage_pixels": int(manifest["coverage_pixels"]),
        "hole_pixels": int(manifest["hole_pixels"]),
        "peak_rss_kib": manifest["peak_rss_kib"],
        "peak_rss_mib": round(manifest["peak_rss_kib"] / 1024, 1),
        "cache_total_bytes": manifest["cache"]["total_bytes"],
        "cache_files_total_bytes_on_disk": cache_total,
        "science_sha256": sha256(sci),
        "coverage_sha256": sha256(cov),
        "manifest_sha256": sha256(OUT / "zegrid_manifest.json"),
        "normalization": manifest["normalization"],
    }

    # finite science over covered area
    science_hwc = np.moveaxis(np.asarray(sci_data, dtype=np.float32), 0, -1)
    covered = np.asarray(cov_data) > 0
    evidence["finite_fraction_over_covered"] = float(
        np.mean(np.isfinite(science_hwc[covered]))
    )
    evidence["science_min_max_over_covered"] = [
        float(np.nanmin(science_hwc[covered])),
        float(np.nanmax(science_hwc[covered])),
    ]

    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
