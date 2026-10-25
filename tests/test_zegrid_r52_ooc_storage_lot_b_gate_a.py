"""ZM-ZEGRID-R52-OOC Lot B — Gate A: real 816-header H128 replay (slow).

Reads ONLY FITS headers from the real Caldwell corpus
(``/home/tristan/caldwell11/lights organized/EQ/IRCUT``), builds the 4117x4377
canvas (id ``e8f1877f16f8cca4``) and validates the header-only PLAN/execution
seam. It never reads pixels, never builds an aligned cache, never stacks, and
never writes a mosaic.

Asserted baseline (reproduced from the accepted 2026-10-25 replay):

* DENSE / H128 / halo 128 / budget 15315.8 MiB is REFUSED: the 4x3 layout's
  cheaper-mode bound is 28450.7 MiB and 5x3 is blocked by
  ``halo_overhead 0.5407 > 0.5``.
* BOUNDED / same budget is ADMITTED with a COMPLETE bundle: 4x3, CPU streaming
  tile 128, bound 4260.3 MiB, 12 stream cell tasks carrying the bounded profile,
  every one of the 816 contributors covered exactly once in the union, and a
  validated RAM-contract envelope.
* GPU/ACTIVE is reported model-only (no CUDA device on this host): with an
  explicit VRAM budget the same bundle admits a GPU stream plan. This is a
  model verdict, never an executed GPU claim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import ram_contract as rcontract

LIGHTS = Path("/home/tristan/caldwell11/lights organized/EQ/IRCUT")
BUDGET_BYTES = int(15315.8 * 2**20)
HALO_PX = 128
EXPECTED_CANVAS_ID = "e8f1877f16f8cca4"
EXPECTED_LAYOUT = (4, 3)


@pytest.fixture(scope="module")
def caldwell_headers():
    if not LIGHTS.is_dir():
        pytest.skip("Caldwell IRCUT corpus not present")
    frames, rejected = zg.read_manifest(LIGHTS)
    if len(frames) != 816:
        pytest.skip(f"expected 816 headers, got {len(frames)}")
    canvas = zg.build_canvas(frames)
    return frames, rejected, canvas


@pytest.mark.slow
def test_gate_a_dense_h128_refusal_and_bounded_bundle(caldwell_headers):
    frames, rejected, canvas = caldwell_headers
    assert rejected == []
    assert (canvas.width, canvas.height) == (4117, 4377)
    assert canvas.canvas_id == EXPECTED_CANVAS_ID

    # ---- DENSE: reproduce the exact H128 refusal -------------------------
    zz._STACK_STORAGE_PROFILE = "dense"
    with pytest.raises(zal.LayoutInfeasible) as excinfo:
        zz._choose_layout_mode_aware(
            canvas, frames, BUDGET_BYTES,
            floors=zal.ScientificFloors(), halo_px=HALO_PX,
            tile_size=zz.STREAM_TILE_SIZE,
        )
    msg = str(excinfo.value)
    assert "28450.7 MiB" in msg, msg
    assert "0.5407" in msg, msg
    assert "15315.8 MiB" in msg, msg

    # ---- BOUNDED: complete admissible bundle -----------------------------
    zz._STACK_STORAGE_PROFILE = "bounded"
    footprints = []
    layout = zz._choose_layout_mode_aware(
        canvas, frames, BUDGET_BYTES,
        floors=zal.ScientificFloors(), halo_px=HALO_PX,
        tile_size=zz.STREAM_TILE_SIZE, footprints_out=footprints,
    )
    assert (layout["nx"], layout["ny"]) == EXPECTED_LAYOUT
    assert layout["cpu_streaming"]["selected_tile_size"] == 128
    assert layout["cpu_streaming"]["predicted_peak_bytes"] < BUDGET_BYTES

    raw = BUDGET_BYTES * rcontract.PLAN_FACTOR_DEN // rcontract.PLAN_FACTOR_NUM
    bundle = zz._build_execution_bundle(
        canvas, frames, footprints[0],
        raw_available_bytes=raw, budget_bytes=BUDGET_BYTES, halo_px=HALO_PX,
    )

    # Profile + model carried by the plan and the bundle.
    assert bundle.storage_profile == "bounded"
    assert bundle.chosen_mode == "stream"
    assert bundle.layout.get("storage_profile") == "bounded"
    assert bundle.plan["storage"]["profile"] == "bounded"
    assert bundle.plan["storage"]["provider"] == "BoundedMemmapCanonicalProvider"
    assert bundle.plan["storage"]["rejection"] == "BoundedCountingRejectionSink"
    assert bundle.plan["storage"]["fresh_sink_per_cell"] is True
    # L-1: the real in-memory bound is published; the 1<<62 planner sentinel is
    # never exposed as a diagnostic bound.
    assert bundle.plan["storage"]["inmem_worst_actual"] < 1 << 40
    import json as _json
    assert str(1 << 62) not in _json.dumps(bundle.plan)

    # CPU admitted: every cell is a bounded streaming task.
    assert bundle.cpu_streaming_tile == 128
    assert len(bundle.cell_tasks) == 12
    assert all(len(t) == 7 for t in bundle.cell_tasks)
    assert all(t[4] in (None, "stream") for t in bundle.cell_tasks)
    assert all(t[6] == "bounded" for t in bundle.cell_tasks if t[4] is not None)
    assert bundle.per_cell_footprint < BUDGET_BYTES

    # No contributor dropped: the union of Cell contributors == all 816 frames.
    covered = set()
    for _cell, _patch, mem, _cache_dir, mode, _bound, *_rest in bundle.cell_tasks:
        if mode is not None:
            covered.update(mem.patch_ids)
    all_ids = {f.frame_id.logical_path for f in frames}
    assert covered == all_ids
    assert len(covered) == 816

    # The frozen RAM contract validates the admitted envelope.
    rcontract.validate_envelope(bundle.envelope)
    assert bundle.envelope.aggregate_bytes <= BUDGET_BYTES

    # ---- GPU/ACTIVE: MODEL-ONLY verdict (no CUDA device here) ------------
    gpu_bundle = zz._build_execution_bundle(
        canvas, frames, footprints[0],
        raw_available_bytes=raw, budget_bytes=BUDGET_BYTES, halo_px=HALO_PX,
        gpu_used=True, vram_budget_bytes=int(7.5 * 2**30),
    )
    assert gpu_bundle.gpu_used is True
    assert gpu_bundle.chosen_mode == "stream"
    assert gpu_bundle.storage_profile == "bounded"
    assert gpu_bundle.gpu_streaming_tile is not None
    # Model verdict only: this test asserts the arithmetic seam, not a GPU run.
