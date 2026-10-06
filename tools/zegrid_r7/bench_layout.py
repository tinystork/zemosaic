#!/usr/bin/env python3
"""ZM-ZEGRID-R7 rework-1 — mode-aware layout BEFORE/AFTER benchmark.

Computes, on the FULL M106 canvas (66 frames), the chosen layout at RAM budgets
1.0 / 2.0 / 4.0 GiB for BOTH:
  * BEFORE: R4 in-memory-only ``auto_layout.choose_layout`` (the old policy), and
  * AFTER:  mode-aware ``zemosaic_zegrid_mode._choose_layout_mode_aware``.

Geometry-only (no pixels / no heavy decode). Resumable; prints a table + JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path("/home/tristan/.openclaw/workspace/worktrees/zemosaic-zegrid-r1/src")
sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import auto_layout as zal  # noqa: E402
from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402

LIGHTS = Path("/home/tristan/M106/lights")
BUDGETS_GIB = (1.0, 2.0, 4.0)
OUT = Path("/home/tristan/zegrid_r7_layout_bench.json")


def main() -> int:
    frames, _rejected = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(frames)
    print(f"FULL M106: {len(frames)} frames, canvas {canvas.width}x{canvas.height} "
          f"({canvas.width * canvas.height} px)")

    rows = []
    for gib in BUDGETS_GIB:
        budget = int(gib * 2**30)
        before = zal.choose_layout(canvas, frames, ram_budget=budget)
        after = zz._choose_layout_mode_aware(canvas, frames, ram_budget=budget)

        # worst-cell bounds for AFTER (the argmax cheaper cell is the layout's worst)
        worst = max(after["cells"], key=lambda c: min(c["inmem_bound_bytes"], c["stream_bound_bytes"]))
        rows.append({
            "budget_gib": gib,
            "before": {
                "nx": before.nx, "ny": before.ny,
                "cells": before.nx * before.ny,
                "max_patch_area": before.max_patch_area,
                "inmem_bound_bytes": int(before.predicted_bound_bytes),
            },
            "after": {
                "nx": after["nx"], "ny": after["ny"],
                "cells": after["cell_count"],
                "max_patch_area": after["max_patch_area"],
                "worst_cell_id": worst["cell_id"],
                "worst_n": worst["n"],
                "inmem_bound_bytes": worst["inmem_bound_bytes"],
                "stream_bound_bytes": worst["stream_bound_bytes"],
                "cheaper_mode": worst["cheaper_mode"],
                "cheaper_bound_bytes": min(worst["inmem_bound_bytes"], worst["stream_bound_bytes"]),
                "budget_binds": after["budget_bound_choice"],
                "floors_ok": after["floors"],
            },
        })
        # verify: every cell's cheaper-mode bound <= budget (no cell exceeds)
        max_cheaper = max(min(c["inmem_bound_bytes"], c["stream_bound_bytes"]) for c in after["cells"])
        assert max_cheaper <= budget, f"budget {gib} GiB: a cell's cheaper bound exceeds budget"

        b, a = rows[-1]["before"], rows[-1]["after"]
        print(
            f"budget {gib:>4.1f} GiB | "
            f"BEFORE {b['nx']}x{b['ny']}={b['cells']:>4} cells (maxpatch {b['max_patch_area']:>7}, "
            f"inmem {b['inmem_bound_bytes']/2**20:>6.0f} MiB) | "
            f"AFTER {a['nx']}x{a['ny']}={a['cells']:>4} cells (maxpatch {a['max_patch_area']:>7}, "
            f"inmem {a['inmem_bound_bytes']/2**20:>6.0f} MiB / stream "
            f"{a['stream_bound_bytes']/2**20:>6.0f} MiB -> {a['cheaper_mode']}, "
            f"binds={a['budget_binds']})"
        )

    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print("\nJSON:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
