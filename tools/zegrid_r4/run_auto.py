#!/usr/bin/env python3
"""ZM-ZEGRID-R4 — RAM-aware Auto layout runner (Levier 1, OPT-IN).

Chooses the layout from a RAM budget (``--ram-budget-mb``) via
``auto_layout.choose_layout`` instead of a fixed ``--nx/--ny``, then delegates
the ENTIRE cell execution + assembly to the unchanged R3 orchestrator
(``tools/zegrid_r3/run_executor.py``), so the explicit ``--nx/--ny`` R3 path is
left untouched. No production wiring.

The layout DECISION (budget, available RAM, model coefficients, estimated peak,
chosen Nx/Ny, floors, budget-bound flag, per-cell predictions) is written to
``<out>/layout_decision.json`` BEFORE any cell runs (durable provenance), and
after the run ``<out>/layout_comparison.json`` records predicted-vs-measured
peak RSS per cell.

Usage (M106, tight budget):
    PYTHONPATH=src .venv/bin/python tools/zegrid_r4/run_auto.py \
        --lights /home/tristan/M106/lights \
        --fixtures /home/tristan/zegrid_r2_fixtures \
        --out /home/tristan/zegrid_r4_m106_tight \
        --ram-budget-mb 1500

Usage (M106, loose budget comparison):
    PYTHONPATH=src .venv/bin/python tools/zegrid_r4/run_auto.py \
        --lights /home/tristan/M106/lights \
        --fixtures /home/tristan/zegrid_r2_fixtures \
        --out /home/tristan/zegrid_r4_m106_loose \
        --ram-budget-mb 6000

Usage (no budget -> coarsest sensible layout):
    ... --out /home/tristan/zegrid_r4_m106_coarse   (omit --ram-budget-mb)

Optional scientific-floor overrides:
    --min-patch-area-px, --min-contributors, --max-halo-overhead
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src"
TOOLS_R3 = Path(__file__).resolve().parents[1] / "zegrid_r3"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(TOOLS_R3))

import run_executor as r3  # noqa: E402  (R3 orchestrator, unchanged)
from zemosaic.core.zegrid import auto_layout as za  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402

MILLION = 2**20


def _layout_decision_path(out: Path) -> Path:
    return out / "layout_decision.json"


def _comparison_path(out: Path) -> Path:
    return out / "layout_comparison.json"


def write_decision(out: Path, decision: za.LayoutDecision) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _layout_decision_path(out).write_text(json.dumps(decision.to_dict(), indent=2) + "\n")


def _load_decision(out: Path) -> za.LayoutDecision:
    d = json.loads(_layout_decision_path(out).read_text())
    cells = tuple(
        za.LayoutCellPrediction(
            cell_id=c["cell_id"], row=c["row"], col=c["col"],
            n_contributors=c["n_contributors"], patch_area_px=c["patch_area_px"],
            predicted_peak_bytes=c["predicted_peak_bytes"],
            predicted_bound_bytes=c["predicted_bound_bytes"],
        )
        for c in d["cells"]
    )
    return za.LayoutDecision(
        nx=d["nx"], ny=d["ny"], status=d["status"],
        ram_budget_bytes=d["ram_budget_bytes"],
        available_bytes=d["available_bytes"],
        median_footprint_w=d["median_footprint_w"],
        median_footprint_h=d["median_footprint_h"],
        n_upper=d["n_upper"], refinement_factor=d["refinement_factor"],
        predicted_bound_bytes=d["predicted_bound_bytes"],
        predicted_mean_bytes=d["predicted_mean_bytes"],
        max_patch_area=d["max_patch_area"], model=d["model"], floors=d["floors"],
        budget_bound_choice=d["budget_bound_choice"],
        warnings=tuple(d["warnings"]), cells=cells,
    )


def write_comparison(out: Path, decision: za.LayoutDecision) -> None:
    """Join per-cell predicted peaks with the measured peaks from cell records."""
    out = Path(out)
    rows = []
    pred = {c.cell_id: c for c in decision.cells}
    for jp in sorted(out.glob("cell_r*.json")):
        rec = json.loads(jp.read_text())
        cid = rec.get("cell_id")
        c = pred.get(cid)
        meas_bytes = rec.get("peak_rss_kib", 0) * 1024.0
        rows.append({
            "cell_id": cid,
            "status": rec.get("status"),
            "n_patch_contributors": rec.get("n_patch_contributors"),
            "predicted_peak_bytes": c.predicted_peak_bytes if c else None,
            "predicted_bound_bytes": c.predicted_bound_bytes if c else None,
            "measured_peak_bytes": meas_bytes if meas_bytes else None,
            "residual_bytes": (meas_bytes - c.predicted_peak_bytes) if c and meas_bytes else None,
        })
    max_meas = max((r["measured_peak_bytes"] or 0) for r in rows)
    summary = {
        "budget_bytes": decision.ram_budget_bytes,
        "budget_bound_choice": decision.budget_bound_choice,
        "max_measured_peak_bytes": max_meas,
        "budget_obeyed": (
            (max_meas <= decision.ram_budget_bytes) if decision.ram_budget_bytes is not None else None
        ),
        "cells": rows,
    }
    _comparison_path(out).write_text(json.dumps(summary, indent=2) + "\n")


def _augment_manifest(out: Path, decision: za.LayoutDecision) -> None:
    """Record the Auto-layout decision into the R3 assembly manifest provenance."""
    mp = Path(out) / "assembly_manifest.json"
    if not mp.exists():
        return
    manifest = json.loads(mp.read_text())
    manifest["layout_decision"] = decision.to_dict()
    mp.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lights", required=True, type=Path)
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ram-budget-mb", type=int, default=None)
    ap.add_argument("--min-patch-area-px", type=int, default=50_000)
    ap.add_argument("--min-contributors", type=int, default=3)
    ap.add_argument("--max-halo-overhead", type=float, default=0.50)
    args = ap.parse_args()

    out = Path(args.out)

    frames, _ = zg.read_manifest(args.lights)
    canvas = zg.build_canvas(frames)
    floors = za.ScientificFloors(
        min_patch_area_px=args.min_patch_area_px,
        min_contributors=args.min_contributors,
        max_halo_overhead=args.max_halo_overhead,
    )
    budget = int(args.ram_budget_mb * MILLION) if args.ram_budget_mb is not None else None

    print(f"[auto] canvas {canvas.width}x{canvas.height} frames={len(frames)}", flush=True)
    decision = za.choose_layout(canvas, frames, budget, floors=floors)
    write_decision(out, decision)
    print(
        f"[auto] budget={args.ram_budget_mb} MiB -> nx={decision.nx} ny={decision.ny} "
        f"({decision.nx * decision.ny} cells) factor={decision.refinement_factor} "
        f"bound={decision.predicted_bound_bytes / MILLION:.0f} MiB "
        f"mean={decision.predicted_mean_bytes / MILLION:.0f} MiB "
        f"available={decision.available_bytes / 2**30:.2f} GiB "
        f"budget_bound={decision.budget_bound_choice} warnings={list(decision.warnings)}",
        flush=True,
    )
    if decision.warnings:
        print(f"[auto] WARNINGS: {list(decision.warnings)}", flush=True)

    # Delegate the FULL execution + assembly to the unchanged R3 orchestrator
    # with the chosen nx/ny. (Auto is opt-in; the explicit --nx/--ny path is the
    # R3 tool itself and is untouched.)
    ns = argparse.Namespace(
        lights=args.lights, fixtures=args.fixtures, out=str(out),
        nx=decision.nx, ny=decision.ny, cell=None, row=None, col=None,
        assemble_only=False,
    )
    rc = r3._orchestrator(ns)
    write_comparison(out, decision)
    _augment_manifest(out, decision)
    print(f"[auto] DONE comparison={_comparison_path(out)}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
