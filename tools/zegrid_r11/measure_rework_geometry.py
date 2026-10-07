#!/usr/bin/env python3
"""ZM-ZEGRID-R11 rework-1 — M106 5x4 header/geometry: which cells contain the
global reference R (exercises the F1 branch without a full pixel run).

Computes, GEOMETRY-ONLY (no decode/reproject):
  * the global reference R = the frame with the greatest projected footprint area
    on the canvas (ties -> lowest sorted FrameId), matching select_canonical_reference
    semantics over the full canvas;
  * for a 5x4 layout, the cells whose PATCH intersects R's footprint (R present)
    vs. the cells that lack R (the F1 branch).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path("/home/tristan/.openclaw/workspace/worktrees/zemosaic-zegrid-r1/src")
sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import sweep as zsw  # noqa: E402

LIGHTS = Path("/home/tristan/M106/lights")
NX, NY = 5, 4


def main() -> int:
    frames, _ = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(frames)
    ordered = sorted(frames, key=lambda f: f.frame_id)
    canvas_wcs = canvas.wcs()

    # Projected footprint area per frame (greatest valid support proxy).
    areas = []
    for f in ordered:
        poly = zg._source_polygon(f.shape_hw, f.wcs(), canvas_wcs)
        areas.append(float(poly.area))
    ref_idx = int(np.argmax(areas))  # ties -> lowest sorted index (deterministic)

    ref_poly = zg._source_polygon(ordered[ref_idx].shape_hw, ordered[ref_idx].wcs(), canvas_wcs)
    ref_area = areas[ref_idx]
    canvas_area = float(canvas.width * canvas.height)

    # Which 5x4 cells contain R (patch intersects R's footprint)?
    cells_present = []
    cells_absent = []
    for row, col, _b in zg.build_layout(canvas, NX, NY).iter_cells(canvas):
        cell, patch, mem = zsw.build_cell_context(ordered, canvas, row, col, NX, NY)
        patch_rect = zg._rect(patch.patch)
        intersects = ref_poly.intersection(patch_rect).area > zg.INTERSECTION_AREA_EPS
        (cells_present if intersects else cells_absent).append(cell.cell_id)

    out = {
        "n_frames": len(ordered),
        "canvas": f"{canvas.width}x{canvas.height}",
        "layout": f"{NX}x{NY}",
        "global_reference_frame_id": ordered[ref_idx].frame_id.logical_path,
        "reference_footprint_fraction_of_canvas": round(ref_area / canvas_area, 4),
        "cells_containing_R": cells_present,
        "cells_lacking_R": cells_absent,
        "n_cells_lacking_R": len(cells_absent),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
