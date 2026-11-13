"""ZM-ZEGRID-CELL-MEMORY-BOUND — ONE memory reservation per ACTIVE cell.

The defect this test pins: the planner collapsed every cell's bound to a single
plan-wide ``max(...)`` constant (13.55 GiB for a real corpus) and used it for the
concurrency decision. A cell's reservation is now the bound of the mode THAT CELL
executes, floored by the measured envelope of committed peaks on a resume, and
the plan publishes the cell that produced the maximum.

Shape fixtures come from the diagnosed M16 corpus: patch area 906 406 px, the
measured streaming bound 780.6 MiB, and a 2 180 MiB owned peak (2 180 MiB x 1.15
= 2 507 MiB envelope) for the N=40 cell. The tail cell (N=21 807) must stay at
the documented 1.5x extrapolation, never the linear 2.7 TB model.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import auto_layout as zal  # noqa: E402
from zemosaic.core.zegrid import parallel as zpar  # noqa: E402

MIB = 2 ** 20
GIB = 2 ** 30

AREA_PX = 906_406
STREAM_BOUND = int(780.6 * MIB)      # measured streaming bound of the shape
INMEM_HUGE = 1 << 62                 # bounded profile: in-memory is never chosen
PLAN_BUDGET = 16_667 * MIB           # the diagnosed host plan budget
OBS_N = 40
OBS_PEAK_KIB = 2_232_320             # 2 180 MiB OWNED peak -> 2 507 MiB envelope

OBSERVATIONS = [
    {
        "n_contributors": OBS_N,
        "patch_area_px": AREA_PX,
        "owned_peak_bytes": OBS_PEAK_KIB * 1024,
    }
]


def _reservation(n, bound=STREAM_BOUND, observations=OBSERVATIONS):
    return zal.reservation_bytes(
        int(n), AREA_PX, int(bound), observations=observations
    )


def _stream_plan(n_cells, reservations, *, budget=PLAN_BUDGET, cell_id="cell-0"):
    return zpar.plan_cell_concurrency(
        [(INMEM_HUGE, STREAM_BOUND)] * int(n_cells),
        16,
        int(budget),
        shared_bytes=0,
        cache_worker_px=1,
        cell_reservations=[int(r) for r in reservations],
        reservation_source="measured",
        reservation_cell_id=cell_id,
    )


# ---------------------------------------------------------------------------
# (a) measured shape: 2 507 MiB reservation, K >= 4 at 90 % of the plan budget
# ---------------------------------------------------------------------------

def test_measured_shape_reservation_is_2507_mib_and_unlocks_parallelism():
    reservation = _reservation(OBS_N)
    assert abs(reservation - 2_507 * MIB) <= 1 * MIB

    plan = _stream_plan(40, [reservation] * 40, cell_id="cell-40")
    assert plan["mode"] == "stream"
    assert plan["cells_in_flight"] >= 4
    assert plan["reservation"]["source"] == "measured"
    assert plan["reservation"]["reservation_bytes"] == reservation
    assert plan["reservation"]["cell_id"] == "cell-40"
    # the 90 % guard is explicit, and K is a strict int (never a float)
    assert plan["reservation"]["decision_budget_bytes"] == int(
        PLAN_BUDGET * 0.9
    )
    assert isinstance(plan["cells_in_flight"], int)


# ---------------------------------------------------------------------------
# (b) tail cell: 1.5x extrapolation, never the linear 2.7 TB model
# ---------------------------------------------------------------------------

def test_tail_shape_stays_bounded_and_never_the_linear_model():
    reservation = _reservation(21_807)
    assert abs(reservation - 3_760 * MIB) <= 1 * MIB

    linear_bound = int(
        zal.FITTED_MEMORY_MODEL.predict_bound_bytes(21_807, AREA_PX)
    )
    # The defect: the in-memory model prices this tail at ~2.7 TB.
    assert linear_bound > 1_000 * GIB
    assert reservation < 8 * GIB


# ---------------------------------------------------------------------------
# (c) two cells with different N produce different reservations
# ---------------------------------------------------------------------------

def test_per_cell_reservations_replace_the_plan_wide_constant():
    # Dense profile: each cell's own in-memory model bound is its reservation.
    small_bound = int(zal.FITTED_MEMORY_MODEL.predict_bound_bytes(10, AREA_PX))
    large_bound = int(zal.FITTED_MEMORY_MODEL.predict_bound_bytes(60, AREA_PX))
    small = _reservation(10, bound=small_bound, observations=None)
    large = _reservation(60, bound=large_bound, observations=None)
    assert small != large
    assert small < large

    plan = zpar.plan_cell_concurrency(
        [(small_bound, STREAM_BOUND), (large_bound, STREAM_BOUND)],
        16,
        PLAN_BUDGET,
        shared_bytes=0,
        cache_worker_px=1,
        cell_reservations=[small, large],
        reservation_source="model",
        reservation_cell_id="cell-large",
    )
    assert plan["reservation"]["reservation_bytes"] == large
    assert plan["reservation"]["cell_id"] == "cell-large"


# ---------------------------------------------------------------------------
# (d) aggregate above the guard -> serial K == 1 with a refusal reason
# ---------------------------------------------------------------------------

def test_over_budget_reservation_falls_back_to_serial_k1_with_a_reason():
    reservation = 16_000 * MIB  # fits the FULL budget, not the 90 % guard
    plan = _stream_plan(4, [reservation] * 4, cell_id="cell-big")

    assert plan["mode"] == "stream"
    assert plan["cells_in_flight"] == 1
    assert isinstance(plan["cells_in_flight"], int)
    assert plan["cells_in_flight"] >= 1
    assert plan["reservation"]["cells_in_flight"] == 1
    reason = plan["reservation"]["reason"]
    assert reason and "serial K=1" in reason
    assert plan["reservation"]["reservation_bytes"] == reservation
    # the serial plan is still validated against the FULL budget
    assert plan["aggregate_bytes"] <= PLAN_BUDGET


# ---------------------------------------------------------------------------
# (e) an injected observed peak above the model raises the reservation
# ---------------------------------------------------------------------------

def test_observed_peak_above_the_model_raises_the_reservation():
    model_only = _reservation(OBS_N, observations=None)
    assert model_only == STREAM_BOUND

    hot = [dict(OBSERVATIONS[0], owned_peak_bytes=4_000 * MIB)]
    raised = _reservation(OBS_N, observations=hot)
    assert raised > model_only
    assert raised == 4_600 * MIB  # 4 000 MiB x 1.15


# ---------------------------------------------------------------------------
# (f) in-memory cells keep their own model bound unchanged
# ---------------------------------------------------------------------------

def test_in_memory_cells_keep_the_model_bound_unchanged():
    inmem_bound = int(zal.FITTED_MEMORY_MODEL.predict_bound_bytes(OBS_N, AREA_PX))
    assert _reservation(OBS_N, bound=inmem_bound, observations=None) == inmem_bound

    plan = zpar.plan_cell_concurrency(
        [(inmem_bound, STREAM_BOUND)] * 3,
        16,
        PLAN_BUDGET,
        shared_bytes=0,
        cache_worker_px=1,
        cell_reservations=[_reservation(OBS_N)] * 3,
        reservation_source="measured",
        reservation_cell_id="cell-inmem",
    )
    inmem_candidate = next(c for c in plan["candidates"] if c["mode"] == "inmem")
    assert inmem_candidate["max_bound_bytes"] == inmem_bound


# ---------------------------------------------------------------------------
# (3) ONE auditable log line per plan: budget, reservation, source, K, cell
# ---------------------------------------------------------------------------

def test_reservation_log_line_names_budget_reservation_source_k_and_cell():
    line = zpar.reservation_log_line(_stream_plan(40, [_reservation(OBS_N)] * 40,
                                                  cell_id="cell-40"))
    assert line is not None
    assert "budget=16667.0MiB" in line
    assert "reservation=2507.0MiB" in line
    assert "source=measured" in line
    assert "K=5" in line
    assert "cell=cell-40" in line

    # a plan without a reservation block (external pseudo-plan) yields no line
    assert zpar.reservation_log_line({"mode": "external"}) is None
