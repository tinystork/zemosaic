"""A1b — layout admission must use the FULL joint CPU aggregate.

The A0 witness proved that a layout could be accepted on its stack bound alone
while the joint PLAN (gauge + cache + stack + assembly + export, ONE 15/100
margin) refused it afterwards. These tests pin the unified behaviour:
``_layout_joint_plan_probe`` calls the REAL scheduler path, so a layout is only
accepted when the same aggregate that ADMIT will build already fits.
"""

from __future__ import annotations

import pytest

from zemosaic import zemosaic_zegrid_mode as mode
from zemosaic.core.zegrid.resource_envelope import (
    DEFAULT_MARGIN_DEN,
    DEFAULT_MARGIN_NUM,
)
from zemosaic.core.zegrid.scheduler import REASON_REFUSE_HOST_RAM
from zemosaic.core.zegrid.streaming import (
    STREAMING_BASELINE_KIB,
    estimate_streaming_peak_kib,
)

M33_CANVAS_PX = 75_698_820
M33_PATCH_HW = (659, 502)
M33_N = 21_818
M33_CELLS = 240
M33_LAYOUT = (12, 20)
M33_BUDGET_MIB = 9_931.5


def _added_by_tile(n: int, hw: tuple[int, int]) -> dict[int, int]:
    return {
        int(t): max(
            1,
            int(
                estimate_streaming_peak_kib(
                    n, hw, int(t), 3, storage_profile="bounded"
                )
            )
            * 1024
            - STREAMING_BASELINE_KIB * 1024,
        )
        for t in mode.CPU_STREAMING_TILE_LADDER
    }


def _probe(
    budget_bytes: int,
    *,
    canvas_px: int = M33_CANVAS_PX,
    patch_hw: tuple[int, int] = M33_PATCH_HW,
    n: int = M33_N,
    cells: int = M33_CELLS,
    layout: tuple[int, int] = M33_LAYOUT,
) -> dict:
    return mode._layout_joint_plan_probe(
        canvas_px=int(canvas_px),
        canvas_hw=(1, int(canvas_px)),
        patch_hw=tuple(int(v) for v in patch_hw),
        n_frames=int(n),
        cell_count=int(cells),
        layout=(int(layout[0]), int(layout[1])),
        halo_px=8,
        budget_bytes=int(budget_bytes),
        available_bytes=(int(budget_bytes) * 5) // 4,
        storage_profile="bounded",
        stream_added_by_tile=_added_by_tile(int(n), tuple(patch_hw)),
        refinement_rank=int(cells),
    )


def test_m33_layout_is_refused_by_full_joint_aggregate():
    # Stack-only stream@20 was 9 289.1 MiB <= 9 931.5 MiB (A0). The joint
    # aggregate must refuse anyway, with a numeric reason.
    result = _probe(int(M33_BUDGET_MIB * 2**20))
    assert result["ok"] is False
    assert result["reason_code"] == REASON_REFUSE_HOST_RAM
    assert "margin" in result["detail"]
    assert "budget" in result["detail"]


def test_feasible_layout_is_admitted_with_one_margin():
    result = _probe(
        1 * 2**30, canvas_px=1_000_000, patch_hw=(100, 100), n=100, cells=4, layout=(2, 2)
    )
    assert result["ok"] is True
    assert result["peak_bytes"] <= result["budget_bytes"]
    assert result["tile_px"] in mode.CPU_STREAMING_TILE_LADDER
    raw = int(result["raw_bytes"])
    expected = (
        raw * (DEFAULT_MARGIN_DEN + DEFAULT_MARGIN_NUM) + DEFAULT_MARGIN_DEN - 1
    ) // DEFAULT_MARGIN_DEN
    assert int(result["peak_bytes"]) == expected
    # M1/F6: A1 prices the product 16 B/px assembly convention; the real
    # mode-aware canvas is 49 B/px global / 125 B/px feathered per_cell.
    # feathered canvas is explicitly carried as DEFERRED to A4.
    assert result["assembly_canvas_deferred_to_a4"] is True
    assert int(result["unpriced_assembly_canvas_bytes"]) > 0
    assert "deferred to A4" in result["detail"]


@pytest.mark.parametrize("gib", (1, 2, 4, 8, 16, 32))
def test_q10_same_request_all_budgets(gib):
    result = _probe(int(gib * 2**30))
    if result["ok"]:
        # A layout accepted at this budget must fit the SAME budget, with the
        # aggregate already margin-applied.
        assert int(result["peak_bytes"]) <= int(result["budget_bytes"])
        assert int(result["tile_px"]) in mode.CPU_STREAMING_TILE_LADDER
    else:
        assert result["reason_code"] == REASON_REFUSE_HOST_RAM


def test_budget_drop_between_plan_and_admission_is_not_stale():
    high = _probe(16 * 2**30)
    low = _probe(1 * 2**30)
    # The two calls must not share a cached winner: the low budget is judged on
    # its own numbers.
    if low["ok"]:
        assert int(low["peak_bytes"]) <= int(low["budget_bytes"])
    else:
        assert low["reason_code"] == REASON_REFUSE_HOST_RAM
    if high["ok"]:
        assert int(high["peak_bytes"]) <= int(high["budget_bytes"])


def test_m33_refusal_reports_best_candidate_not_generic():
    result = _probe(int(M33_BUDGET_MIB * 2**20))
    assert result["ok"] is False
    # The detail names the actual aggregate, not "no CPU candidate fits".
    assert result["detail"].startswith("host peak ")
    assert result["detail"] != "no CPU candidate fits the hard constraints"
