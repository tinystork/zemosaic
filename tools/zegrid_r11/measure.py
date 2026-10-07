#!/usr/bin/env python3
"""ZM-ZEGRID-R11 measurement on M16 (iteration speed): seam metric BEFORE/AFTER
and layout-independence, exercising the production internals directly.

BEFORE = per-cell reference/coefficients (legacy `fixed=None`).
AFTER  = the global photometric gauge (one reference + per-frame coefficients
         computed over the full canvas, subset per cell).

The gauge is LAYOUT-INDEPENDENT (full canvas), so it is computed ONCE and shared
across the two pinned layouts. Cell caches and the gauge cache are disk-backed
and resumable.

Memory discipline: one heavy run at a time; cell caches are disk-backed.
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
from zemosaic.core.canonical_streaming import subset_fixed_normalization  # noqa: E402
from zemosaic.core.zegrid import assembly as za  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import file_provider as zfp  # noqa: E402
from zemosaic.core.zegrid import mosaic as zmosaic  # noqa: E402
from zemosaic.core.zegrid import photometric as zphot  # noqa: E402
from zemosaic.core.zegrid import seam as zsm  # noqa: E402
from zemosaic.core.zegrid import sweep as zsw  # noqa: E402
from zemosaic.core.zegrid.executor import ExecutorConfig  # noqa: E402

LIGHTS = Path("/home/tristan/M16/quick")
WORK = Path("/home/tristan/zegrid_r11_m16_measure")
NOOP = lambda *a, **k: None


def science_hash(science: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(science, dtype=np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()


def build_cell_caches(descs, canvas, nx, ny):
    config = ExecutorConfig().science_config()
    cell_ctxs = []
    for row, col, _b in zg.build_layout(canvas, nx, ny).iter_cells(canvas):
        cell, patch, mem = zsw.build_cell_context(descs, canvas, row, col, nx, ny)
        cell_ctxs.append((row, col, cell, patch, mem))
    cache_root = WORK / f"cache_{nx}x{ny}"
    cache_dirs, manifests, cache_report = zz._build_aligned_cache_frame_major(
        descs, canvas, cell_ctxs, cache_root, NOOP
    )
    return config, cell_ctxs, cache_dirs


def run_cells(descs, canvas, nx, ny, config, cell_ctxs, cache_dirs, gauge, global_frame_ids, use_gauge):
    cores = {}
    ref_ids = {}
    for (row, col, cell, patch, mem) in cell_ctxs:
        cid = cell.cell_id
        if not mem.patch_ids:
            cores[cid] = None
            continue
        cache_dir = cache_dirs[cid]
        provider = zfp.MemmapCanonicalProvider(cache_dir)
        cell_frame_ids = list(provider.frame_ids)
        fixed = None
        if use_gauge:
            fixed = subset_fixed_normalization(gauge, global_frame_ids, cell_frame_ids)
        mt, sres = zz._run_cell_stream(cache_dir, patch, config, NOOP, fixed=fixed)
        cores[cid] = za.crop_all_planes_to_core(mt)
        ref_ids[cid] = sres.reference_frame_id

    assembled = zmosaic.assemble_canvas(canvas, nx, ny, cores)
    metric = zsm.compute_boundary_step_metric(canvas, assembled.science, nx, ny, strip_px=6)
    return {
        "nx": nx,
        "ny": ny,
        "gauge": use_gauge,
        "per_cell_reference_frame_ids": ref_ids,
        "seam_median_pct": metric["median_pct"],
        "seam_max_pct": metric["max_pct"],
        "n_boundaries": metric["n_boundaries"],
        "science_sha256": science_hash(assembled.science),
        "coverage_pixels": assembled.coverage_pixels,
        "hole_pixels": assembled.hole_pixels,
    }


def main() -> int:
    descs, rejected = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(descs)
    print(f"M16: {len(descs)} frames, canvas {canvas.width}x{canvas.height}", flush=True)

    config = ExecutorConfig().science_config()

    # Global gauge — computed ONCE (layout-independent), shared by both layouts.
    gauge_dir = WORK / "__gauge__"
    gauge, global_frame_ids = zphot.compute_global_gauge(
        descs, canvas, lambda f: zz._decode_frame_hwc(f, NOOP), config, gauge_dir
    )
    ref_id = global_frame_ids[int(gauge.reference_index)]
    print(f"global reference = {ref_id!r} (index {gauge.reference_index})", flush=True)

    results = []

    # Layout A (3x3): BEFORE + AFTER.
    cfg_a, ctxs_a, dirs_a = build_cell_caches(descs, canvas, 3, 3)
    results.append(run_cells(descs, canvas, 3, 3, cfg_a, ctxs_a, dirs_a, gauge,
                             global_frame_ids, use_gauge=False))
    results.append(run_cells(descs, canvas, 3, 3, cfg_a, ctxs_a, dirs_a, gauge,
                             global_frame_ids, use_gauge=True))

    # Layout B (2x2): AFTER only, for layout-independence.
    cfg_b, ctxs_b, dirs_b = build_cell_caches(descs, canvas, 2, 2)
    results.append(run_cells(descs, canvas, 2, 2, cfg_b, ctxs_b, dirs_b, gauge,
                             global_frame_ids, use_gauge=True))

    out = {"global_reference_frame_id": ref_id, "results": results}
    print(json.dumps(out, indent=2))
    (WORK / "measure.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
