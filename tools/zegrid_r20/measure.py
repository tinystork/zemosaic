#!/usr/bin/env python3
"""ZM-ZEGRID-R20 measurement — per-cell STACKING concurrency BEFORE vs AFTER.

Runs the production ``run_zegrid_mode`` on the M16 "iteration speed" corpus with
a PINNED layout (so the two runs are directly comparable), once with the serial
per-cell loop (``cells_in_flight=1``, the R19 behaviour) and once with several
cells in flight (the R20 concurrent path). Reports:

* the ``per_cell_stack`` phase time and the end-to-end time (BEFORE vs AFTER),
* the chosen cells-in-flight (and the AUTO value the RAM budget would derive),
* the peak RSS observed,
* the bit-equality evidence (science SHA-256 identical).

The AUTO cells-in-flight is derived from the run's own ``per_cell_diagnostics``
(available RAM, RAM_SAFETY_FRACTION budget, per-cell footprint) so the report
shows what the production path would choose WITHOUT the forced override.

Usage::

    python tools/zegrid_r20/measure.py [corpus_dir] [output_dir]

Defaults: corpus = /home/tristan/M16/quick, output = /home/tristan/zegrid_r20_measure.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zegrid
from zemosaic.core.zegrid import parallel as zpar

M16 = Path("/home/tristan/M16/quick")
OUT = Path("/home/tristan/zegrid_r20_measure")
LAYOUT = "3x3"


def _science_sha256(out_dir: Path) -> str:
    with fits.open(out_dir / "mosaic_grid.fits") as hdul:
        data = np.ascontiguousarray(np.asarray(hdul[0].data, dtype=np.float32))
    return hashlib.sha256(data.tobytes()).hexdigest()


def _run(corpus: Path, out_dir: Path, cells_in_flight: int | None) -> dict:
    """Run once; ``cells_in_flight=None`` uses the AUTO decision, else forced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    orig = zpar.cells_in_flight
    if cells_in_flight is not None:
        zpar.cells_in_flight = lambda *a, **k: int(cells_in_flight)
    try:
        t0 = time.perf_counter()
        zconfig = SimpleNamespace(zegrid_layout=LAYOUT)
        zegrid.run_zegrid_mode(str(corpus), str(out_dir), zconfig=zconfig)
        wall = time.perf_counter() - t0
    finally:
        zpar.cells_in_flight = orig

    manifest = json.loads((out_dir / "zegrid_manifest.json").read_text(encoding="utf-8"))
    timings = manifest.get("timings", {})
    diag = manifest.get("per_cell_diagnostics", {})
    return {
        "wall_s": round(wall, 3),
        "per_cell_stack_s": timings.get("per_cell_stack"),
        "total_s": timings.get("total"),
        "cache_build_s": timings.get("cache_build"),
        "peak_rss_kib": manifest.get("peak_rss_kib"),
        "cells_in_flight": diag.get("cells_in_flight"),
        "executor": diag.get("executor"),
        "per_cell_footprint_bytes": diag.get("per_cell_footprint_bytes"),
        "ram_budget_bytes": diag.get("ram_budget_bytes"),
        "available_bytes": diag.get("available_bytes"),
        "science_sha256": _science_sha256(out_dir),
    }


def main() -> int:
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else M16
    out_root = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    if not (corpus / "stack_plan.csv").exists():
        print(f"no stack_plan.csv in {corpus}", file=sys.stderr)
        return 2

    before = _run(corpus, out_root / "serial", cells_in_flight=1)
    after = _run(corpus, out_root / "concurrent", cells_in_flight=2)

    # AUTO decision (what the production path would choose without an override).
    footprint = int(after.get("per_cell_footprint_bytes") or 0)
    budget = int(after.get("ram_budget_bytes") or 0)
    auto = zpar.cells_in_flight(os.cpu_count(), budget, footprint)

    report = {
        "mission": "ZM-ZEGRID-R20",
        "corpus": str(corpus),
        "layout": LAYOUT,
        "auto_cells_in_flight": auto,
        "before": before,
        "after": after,
        "bit_equal": before["science_sha256"] == after["science_sha256"],
        "per_cell_stack_speedup": (
            round(before["per_cell_stack_s"] / after["per_cell_stack_s"], 2)
            if before["per_cell_stack_s"] and after["per_cell_stack_s"] else None
        ),
        "end_to_end_speedup": (
            round(before["total_s"] / after["total_s"], 2)
            if before["total_s"] and after["total_s"] else None
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
