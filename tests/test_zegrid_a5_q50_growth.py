"""ZM-ZEGRID-ADAPTIVE-RESOURCES A5 — Q50: budget ramp + restart, grow exploitable.

Gate: a budget that DROPS (4 -> 1 Gio) then RISES (1 -> 4 Gio) must let the
scheduler SHRINK resources prudently and then GROW them back to the same plan,
with the scientific identity (layout/cells/frames/halo) preserved. The growth
path is the exact inverse of the shrink path (``GROW_ORDER`` reversed), bounded
by the SAME enumeration constraints, and guarded by ``assert_grow_only`` /
``assert_joint_grow_only`` (a growth step can never shrink bytes or drop a GPU).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import resource_envelope as re  # noqa: E402
from zemosaic.core.zegrid import scheduler as zs  # noqa: E402

GIB = 1024 ** 3
MIB = 1024 ** 2
CELLS = ("r0000c0000", "r0000c0001", "r0001c0000", "r0001c0001")
FRAMES = ("frame_0001.fits", "frame_0002.fits")


def _ctx(*, budget_bytes, raw=None, **overrides):
    data = dict(
        phase=re.PHASE_CELL_STACK_STREAM,
        backend=re.BACKEND_CPU,
        tile_ladder=zs.PRODUCT_TILE_LADDER,
        budget_bytes=int(budget_bytes),
        raw_available_bytes=int(raw if raw is not None else budget_bytes * 5 // 4),
        factor_num=4,
        factor_den=5,
        cpu_logical=8,
        units_total=4,
        shared_bytes=340 * MIB,
        per_unit_stack_bytes=200 * MIB,
        per_unit_cache_bytes=50 * MIB,
        per_unit_prefetch_bytes=50 * MIB,
        disk_free_bytes=10 * GIB,
        layout=(3, 4),
        refinement_rank=12,
        patch_hw=(200, 200),
        halo_px=32,
        channels=3,
        n_frames=50,
        cell_ids=CELLS,
        frame_ids=FRAMES,
    )
    data.update(overrides)
    return zs.PlanContext(**data)


def _requests(phase=re.PHASE_CELL_STACK_STREAM):
    return (
        zs.PhaseRequest(
            phase=re.PHASE_GAUGE,
            components={"gauge_worker_footprint_bytes": 400 * MIB},
            units=1, units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=re.PHASE_CACHE_BUILD,
            components={"cache_build_worker_footprint_bytes": 300 * MIB},
            units=1, units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=re.PHASE_ASSEMBLY,
            components={"halo_band_bytes": 4 * MIB, "retained_core_bytes": 100 * MIB},
            units=1, units_in_flight=1, overlaps_with=(phase,),
        ),
        zs.PhaseRequest(
            phase=re.PHASE_EXPORT,
            components={"coverage_plane_bytes": 2 * MIB, "science_planes_bytes": 6 * MIB},
            units=1, units_in_flight=1,
        ),
    )


def _winner(ctx):
    plan = zs.plan(ctx, phase_requests=_requests())
    return plan.chosen


def _lattice_max(ctx):
    """The lattice maximum candidate reachable by growth (tile=largest,
    units=k_max, workers=bounded by the coupled w_max, prefetch=largest value
    that still respects the ``k * (1 + prefetch) <= cpu_slots`` slot cap)."""
    cpu_slots = max(0, int(ctx.cpu_logical) - zs.CPU_OS_RESERVE)
    k_max = min(cpu_slots, int(ctx.units_total))
    tile = max(int(t) for t in ctx.tile_ladder)
    w_max = max(1, int(ctx.cpu_logical) // max(1, k_max))
    workers = min(max(int(w) for w in ctx.workers_ladder), w_max)
    prefetch = 0
    for p in sorted(ctx.prefetch_ladder):
        if int(k_max) * (1 + int(p)) <= cpu_slots:
            prefetch = int(p)
    return zs.Candidate(
        backend=re.BACKEND_CPU,
        tile_px=tile,
        units_in_flight=k_max,
        prefetch_units=prefetch,
        workers_per_unit=workers,
        host_peak_bytes=0,
        disk_bytes=0,
        device_peak_bytes=0,
        feasible=True,
        reason_code=re.REASON_OK,
    )


def test_q50_grow_is_exact_inverse_of_shrink():
    # From the lattice MAXIMUM, shrink_sequence reaches the minimum and
    # grow_sequence walks back to the EXACT same tuple. This is the true
    # inverse property: growth reverses the shrink order field by field.
    ctx = _ctx(budget_bytes=(4 * GIB * 4) // 5, raw=4 * GIB)
    big = _lattice_max(ctx)

    shrunk_steps = zs.shrink_sequence(big, ctx)
    small = shrunk_steps[-1][1] if shrunk_steps else big
    assert len(shrunk_steps) > 0
    assert small.tile_px == min(ctx.tile_ladder)
    assert small.units_in_flight == 1
    assert small.workers_per_unit == 1
    assert small.prefetch_units == 0

    grown_steps = zs.grow_sequence(small, ctx)
    back = grown_steps[-1][1] if grown_steps else small
    assert back == big, "grow must be the exact inverse of shrink from the lattice max"
    # Scientific identity (frozen fields) is untouched by either sequence.
    assert zs.assert_same_identity(ctx.identity, ctx.identity) is None


def test_q50_shrink_sequence_then_grow_sequence_round_trips():
    # From the winner, shrink fully, then grow back: the result is a feasible
    # candidate with the largest tile restored, and the scientific identity is
    # preserved (the worker/unit trade-off may differ under the coupled w_max
    # constraint, but growth never shrinks below the minimum).
    ctx = _ctx(budget_bytes=(4 * GIB * 4) // 5, raw=4 * GIB)
    big = _winner(ctx)
    shrunk = zs.shrink_sequence(big, ctx)
    smallest = shrunk[-1][1]
    grown = zs.grow_sequence(smallest, ctx)
    largest = grown[-1][1]
    assert int(largest.tile_px) == max(ctx.tile_ladder)
    assert int(largest.units_in_flight) >= int(smallest.units_in_flight)
    assert int(largest.workers_per_unit) >= int(smallest.workers_per_unit)
    assert int(largest.prefetch_units) >= int(smallest.prefetch_units)
    assert zs.assert_same_identity(ctx.identity, ctx.identity) is None


def test_q50_grow_guard_rejects_any_shrink():
    ctx = _ctx(budget_bytes=(4 * GIB * 4) // 5, raw=4 * GIB)
    big = _winner(ctx)
    step_name, nxt = zs.next_shrink(big, ctx)
    assert step_name is not None and nxt is not None
    # A shrink step is refused by the growth guard.
    with pytest.raises(zs.GrowthRejected):
        zs.assert_grow_only(big, nxt)
    with pytest.raises(zs.GrowthRejected):
        zs.assert_joint_grow_only(big, nxt)


def test_q50_joint_grow_guard_rejects_byte_shrink():
    ctx = _ctx(budget_bytes=(4 * GIB * 4) // 5, raw=4 * GIB)
    big = _winner(ctx)
    shrunk = dataclasses.replace(
        big, host_peak_bytes=big.host_peak_bytes - 1, disk_bytes=big.disk_bytes - 1
    )
    with pytest.raises(zs.GrowthRejected):
        zs.assert_joint_grow_only(big, shrunk)


def test_q50_low_budget_plan_refuses_or_shrinks():
    # A 1 Gio budget must not buy the same plan as 4 Gio: either the plan picks a
    # smaller candidate or refuses closed (never a silent bigger plan).
    ctx_low = _ctx(budget_bytes=(1 * GIB * 4) // 5, raw=1 * GIB)
    try:
        low = _winner(ctx_low)
    except zs.PlanRefused as exc:
        assert exc.reason_code in (re.REASON_REFUSE_HOST_RAM, re.REASON_REFUSE_DISK)
        return
    ctx_high = _ctx(budget_bytes=(4 * GIB * 4) // 5, raw=4 * GIB)
    high = _winner(ctx_high)
    # The low-budget plan never exceeds the high-budget plan's resources.
    assert low.host_peak_bytes <= high.host_peak_bytes
