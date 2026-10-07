#!/usr/bin/env python3
"""ZM-ZEGRID-R11: 2x2 BEFORE (legacy per-cell references) — layout-dependence contrast.

Reuses the already-built 2x2 cell cache. Runs every cell with fixed=None (legacy
per-cell reference/coefficients), assembles, and reports the science hash + seam
metric + per-cell reference ids. Contrasts with the 3x3 BEFORE hash to prove the
legacy science was LAYOUT-DEPENDENT (R7-M1 caveat).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path("/home/tristan/.openclaw/workspace/worktrees/zemosaic-zegrid-r1/src")
sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import assembly as za  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import file_provider as zfp  # noqa: E402
from zemosaic.core.zegrid import mosaic as zmosaic  # noqa: E402
from zemosaic.core.zegrid import seam as zsm  # noqa: E402
from zemosaic.core.zegrid import sweep as zsw  # noqa: E402
from zemosaic.core.zegrid.executor import ExecutorConfig  # noqa: E402

LIGHTS = Path("/home/tristan/M16/quick")
WORK = Path("/home/tristan/zegrid_r11_m16_measure")
NOOP = lambda *a, **k: None


def main() -> int:
    descs, _ = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(descs)
    config = ExecutorConfig().science_config()
    nx, ny = 2, 2

    cell_ctxs = []
    for row, col, _b in zg.build_layout(canvas, nx, ny).iter_cells(canvas):
        cell, patch, mem = zsw.build_cell_context(descs, canvas, row, col, nx, ny)
        cell_ctxs.append((row, col, cell, patch, mem))

    cache_root = WORK / "cache_2x2"
    cores = {}
    ref_ids = {}
    for (row, col, cell, patch, mem) in cell_ctxs:
        cid = cell.cell_id
        if not mem.patch_ids:
            cores[cid] = None
            continue
        cache_dir = cache_root / cid
        mt, sres = zz._run_cell_stream(cache_dir, patch, config, NOOP, fixed=None)
        cores[cid] = za.crop_all_planes_to_core(mt)
        ref_ids[cid] = sres.reference_frame_id

    assembled = zmosaic.assemble_canvas(canvas, nx, ny, cores)
    metric = zsm.compute_boundary_step_metric(canvas, assembled.science, nx, ny, strip_px=6)
    a = np.ascontiguousarray(np.asarray(assembled.science, dtype=np.float32))
    out = {
        "nx": nx,
        "ny": ny,
        "gauge": False,
        "per_cell_reference_frame_ids": ref_ids,
        "seam_median_pct": metric["median_pct"],
        "seam_max_pct": metric["max_pct"],
        "science_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
    }
    print(json.dumps(out, indent=2))
    (WORK / "measure_2x2_legacy.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
