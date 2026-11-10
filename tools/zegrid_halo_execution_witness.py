#!/usr/bin/env python3
"""ZM-HALO-EXECUTION-DECOUPLING — read-only Caldwell/LP dry witness.

Proves, WITHOUT running the product and WITHOUT touching any output folder, that
a user-requested 128 px halo is no longer refused solely because
``halo_overhead = 0.5363 > max_halo_overhead = 0.5``, and that a truthful
executable plan exists under the observed PLAN budget (2011.4 MiB).

What it does (read-only, side-effect free):

1. reads the FITS *headers* of the LP corpus (``fits.getheader`` only — no pixel
   data, no cache, no output);
2. rebuilds the exact canvas / median projected footprint / factor-1 grid;
3. runs the layout planner twice: historical HARD FLOOR (must refuse with the
   observed 0.5363 > 0.5 halo-conflict rescue path) and ADVISORY explicit-halo
   policy (must produce a memory-admitted plan at the unchanged 128 px halo);
4. with ``--bundle``, builds the COMPLETE execution bundle at the advisory
   layout and reports the physical working-set class, tile, column batch and
   the joint envelope — the truthful executable plan.

Nothing is written anywhere. The durable live run and its runtime slot are
never read, locked or modified.

Usage::

    <venv>/bin/python -E tools/zegrid_halo_execution_witness.py
    <venv>/bin/python -E tools/zegrid_halo_execution_witness.py --bundle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic import zegrid_halo as zhalo  # noqa: E402
from zemosaic.core.zegrid import auto_layout as zal  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import ram_contract as rcontract  # noqa: E402

MIB = 2 ** 20
DEFAULT_LIGHTS = Path("/home/tristan/caldwell11/lights organized/EQ/LP")
DEFAULT_BUDGET_MIB = 2011.4
DEFAULT_HALO = 128


def _layout_scan(canvas, frames, raw, halo_px, policy):
    logs: list[str] = []
    try:
        layout = zz._choose_layout_mode_aware(
            canvas, frames, ram_budget=raw, available_bytes=raw,
            halo_px=halo_px, floors=zal.ScientificFloors(),
            log_line=logs.append, footprints_out=[], allow_external=True,
            halo_policy=policy,
        )
        return {
            "status": "ADMITTED",
            "layout": [int(layout["nx"]), int(layout["ny"])],
            "cell_count": int(layout["cell_count"]),
            "layout_origin": layout["layout_origin"],
            "predicted_bound_mib": round(
                int(layout["predicted_bound_bytes"]) / MIB, 3
            ),
            "cpu_streaming_tile": layout["cpu_streaming"].get(
                "selected_tile_size"
            ),
            "halo_overhead": layout["floors"]["max_halo_overhead"],
            "halo_overhead_policy": layout.get("halo_overhead_policy"),
            "halo_overhead_advisory": layout.get("halo_overhead_advisory"),
            "warnings": list(layout.get("warnings") or ()),
            "layout_rescue": layout.get("layout_rescue"),
            "logs": logs,
            "_layout": layout,
        }
    except zal.LayoutInfeasible as exc:
        return {
            "status": "REFUSED",
            "reason": str(exc),
            "refusal": getattr(exc, "refusal", None),
            "logs": logs,
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lights", type=Path, default=DEFAULT_LIGHTS)
    ap.add_argument("--budget-mib", type=float, default=DEFAULT_BUDGET_MIB)
    ap.add_argument("--halo", type=int, default=DEFAULT_HALO)
    ap.add_argument("--bundle", action="store_true",
                    help="also build the complete execution bundle")
    args = ap.parse_args(argv)

    if not args.lights.is_dir():
        print(json.dumps({"status": "SKIP", "reason": f"missing corpus {args.lights}"}))
        return 0

    frames, rejected = zg.read_manifest(args.lights)
    canvas = zg.build_canvas(frames)
    mw, mh = zal.median_projected_footprint(frames, canvas)
    factor1 = (
        int(-(-canvas.width // mw)),
        int(-(-canvas.height // mh)),
    )
    geom = zal._nominal_geometry(canvas, factor1[0], factor1[1], args.halo)
    raw = int(round(args.budget_mib * MIB))

    result = {
        "mission": "ZM-HALO-EXECUTION-DECOUPLING-20261109",
        "corpus": str(args.lights),
        "n_frames": len(frames),
        "rejected": rejected,
        "canvas": [int(canvas.width), int(canvas.height)],
        "median_projected_footprint": [round(mw, 3), round(mh, 3)],
        "factor1_layout": list(factor1),
        "halo_px": int(args.halo),
        "halo_overhead_factor1": round(float(geom["halo_overhead"]), 6),
        "min_patch_area_factor1": int(geom["min_patch_area"]),
        "plan_budget_mib": float(args.budget_mib),
        "hard_floor": None,
        "advisory": None,
    }

    # 1. Historical hard floor: the observed refusal must still be reproducible.
    hard = _layout_scan(canvas, frames, raw, args.halo, None)
    hard.pop("_layout", None)
    result["hard_floor"] = hard

    # 2. Advisory explicit-halo policy: a truthful plan must exist.
    adv = _layout_scan(
        canvas, frames, raw, args.halo, zhalo.HALO_OVERHEAD_POLICY_ADVISORY
    )
    adv_layout = adv.pop("_layout", None)
    result["advisory"] = adv

    # 3. Optional: the COMPLETE executable bundle at the advisory geometry.
    if args.bundle and adv_layout is not None:
        # The observed 2011.4 MiB is the PLAN budget (B = 4/5 available). Recover
        # the matching RAW observation with exact integer arithmetic so the
        # envelope validator sees its own contract, never a rounded float.
        raw_available = -(-raw * rcontract.PLAN_FACTOR_DEN // rcontract.PLAN_FACTOR_NUM)
        plan_budget = int(rcontract.plan_budget_bytes(int(raw_available)))
        if plan_budget != raw:
            raw_available = raw
            plan_budget = int(rcontract.plan_budget_bytes(int(raw_available)))
        bundle = zz._build_execution_bundle(
            canvas, frames, None,
            raw_available_bytes=int(raw_available), budget_bytes=int(plan_budget),
            cpu_count=4, cache_root=None, halo_px=int(args.halo),
            halo_policy=zhalo.HALO_OVERHEAD_POLICY_ADVISORY,
        )
        ws = bundle.working_set_block or {}
        ext = bundle.external_contract or {}
        result["bundle"] = {
            "layout": [int(bundle.layout["nx"]), int(bundle.layout["ny"])],
            "chosen_mode": bundle.chosen_mode,
            "stack_working_set": bundle.stack_working_set,
            "cpu_streaming_tile": bundle.cpu_streaming_tile,
            "cells_in_flight": bundle.cells_in_flight,
            "cache_workers_per_cell": bundle.cache_workers_per_cell,
            "external_tile": ext.get("tile_size"),
            "external_column_batch": ext.get("column_batch"),
            "envelope_layout": list(bundle.envelope.layout),
            "envelope_refinement_rank": bundle.envelope.refinement_rank,
            "working_set": ws or None,
            "halo_overhead_policy": bundle.layout.get("halo_overhead_policy"),
            "halo_overhead_advisory": bundle.layout.get("halo_overhead_advisory"),
        }

    print(json.dumps(result, indent=1, default=str))
    return 0 if result["advisory"]["status"] == "ADMITTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
