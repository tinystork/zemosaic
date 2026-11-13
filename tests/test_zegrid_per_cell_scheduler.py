"""ZM-ZEGRID-PER-CELL-SCHEDULER — a per-cell dispatch queue, not a plan-wide K.

The defect: one heavy cell (N=21 807) sized the WHOLE plan to K=1, so 103 light
cells that individually need ~1.5-2 GiB ran one at a time. The dispatcher admits
each cell when ITS OWN reservation fits the remaining decision budget (90 % of
the plan budget), skips a cell that does not fit instead of blocking the queue,
and keeps the serial K=1 fallback when nothing fits.

Byte identity is the hard invariant: the same worker, the same tasks, the same
order — results are collected by task index exactly like ``parallel.pmap``.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import parallel as zpar  # noqa: E402

MIB = 2 ** 20
GIB = 2 ** 30


# --- module-level workers (picklable by reference) -------------------------

def _sleep_worker(task):
    time.sleep(float(task.get("sleep", 0.0)))
    return {"cell": task["cell"], "index": task["index"]}


def _plane_worker(task):
    """Deterministic per-cell plane, hashed: the byte-identity witness."""
    rng = np.random.default_rng(int(task["index"]) + 7)
    plane = rng.normal(0.0, 1.0, (64, 96)).astype(np.float32)
    plane += np.float32(task["index"])
    return {
        "cell": task["cell"],
        "index": int(task["index"]),
        "digest": hashlib.sha256(plane.tobytes()).hexdigest(),
        "nbytes": int(plane.nbytes),
        "dtype": str(plane.dtype),
        "shape": list(plane.shape),
    }


def _tasks(n, sleep=0.0):
    return [
        {"cell": f"cell-{i}", "index": i, "sleep": sleep}
        for i in range(n)
    ]


def _run(reservations, *, budget, cap, tasks=None, sleep=0.0):
    tasks = _tasks(len(reservations), sleep) if tasks is None else tasks
    logs = []
    meta = {}
    out = zpar.pmap_reserved(
        _sleep_worker,
        tasks,
        reservations,
        shared_bytes=0,
        decision_budget_bytes=budget,
        workers=cap,
        cell_ids=[t["cell"] for t in tasks],
        emit=lambda msg, lvl="INFO": logs.append(msg),
        meta=meta,
    )
    return out, meta, logs


# ---------------------------------------------------------------------------
# 1. four light cells that all fit => they overlap, inside the budget
# ---------------------------------------------------------------------------

def test_light_cells_overlap_within_the_decision_budget():
    light = 1 * GIB
    out, meta, logs = _run([light] * 4, budget=6 * GIB, cap=4, sleep=0.6)

    assert [r["index"] for r in out] == [0, 1, 2, 3]
    assert meta["max_concurrent"] == 4          # all four really overlapped
    dispatches = [e for e in meta["events"] if e["event"] == "dispatch"]
    assert len(dispatches) == 4
    assert [d["k"] for d in dispatches] == [1, 2, 3, 4]
    # the running reservation sum never exceeds the decision budget
    for d in dispatches:
        assert d["running_bytes"] <= 6 * GIB
    assert any("k=4" in line for line in logs)


# ---------------------------------------------------------------------------
# 2. one heavy cell => it runs ALONE; nothing else starts while it runs
# ---------------------------------------------------------------------------

def test_heavy_cell_runs_alone_and_lights_wait_for_its_turn():
    reservations = [14 * GIB, 1500 * MIB, 1500 * MIB, 1500 * MIB]
    tasks = _tasks(4)
    tasks[0]["sleep"] = 0.7
    out, meta, logs = _run(
        reservations, budget=15 * GIB, cap=4, tasks=tasks
    )

    assert out[0]["cell"] == "cell-0"
    events = meta["events"]
    first = events[0]
    assert first["event"] == "dispatch" and first["index"] == 0
    assert first["k"] == 1 and first["reservation_bytes"] == 14 * GIB

    # The heavy cell must COMPLETE before any other cell is dispatched.
    done_heavy = next(
        i for i, e in enumerate(events)
        if e["event"] == "done" and e["index"] == 0
    )
    other_dispatches = [
        i for i, e in enumerate(events)
        if e["event"] == "dispatch" and e["index"] != 0
    ]
    assert other_dispatches, "the light cells must still run"
    assert min(other_dispatches) > done_heavy
    # and the heavy dispatch consumed the whole budget on its own
    assert first["running_bytes"] + 1500 * MIB > 15 * GIB


# ---------------------------------------------------------------------------
# 3. nothing fits => serial K=1 with an explicit reason
# ---------------------------------------------------------------------------

def test_nothing_fits_degrades_to_serial_k1_with_a_reason():
    out, meta, logs = _run([2 * GIB, 2 * GIB], budget=1 * GIB, cap=2)

    assert [r["index"] for r in out] == [0, 1]
    first = meta["events"][0]
    assert first["event"] == "dispatch"
    assert first["k"] == 1
    assert first["reason"] == zpar.DISPATCH_REASON_NOTHING_FITS
    assert first["running_bytes"] == 2 * GIB     # above the decision budget
    assert any("serial_k1" in line for line in logs)


# ---------------------------------------------------------------------------
# 4. byte identity: sequential vs scheduled produce identical plane digests
# ---------------------------------------------------------------------------

def test_scheduled_run_is_byte_identical_to_the_sequential_run():
    tasks = _tasks(5)
    calls = {"n": 0}

    def _seq_worker(task):
        calls["n"] += 1
        return _plane_worker(task)

    seq = [
        zpar.pmap(_plane_worker, [t], workers=1)[0] for t in tasks
    ]
    sched_meta = {}
    sched = zpar.pmap_reserved(
        _plane_worker, tasks, [256 * MIB] * 5,
        shared_bytes=0, decision_budget_bytes=4 * GIB, workers=4,
        cell_ids=[t["cell"] for t in tasks], meta=sched_meta,
    )

    assert [s["cell"] for s in sched] == [t["cell"] for t in tasks]
    assert [s["digest"] for s in sched] == [s["digest"] for s in seq]
    assert [s["nbytes"] for s in sched] == [s["nbytes"] for s in seq]
    assert [s["dtype"] for s in sched] == [s["dtype"] for s in seq]
    assert sched_meta["max_concurrent"] > 1


# ---------------------------------------------------------------------------
# 5. deterministic dispatch log line, with every required field
# ---------------------------------------------------------------------------

def test_dispatch_log_line_is_deterministic_and_complete():
    reservations = [1500 * MIB, 1500 * MIB]
    out, meta, logs = _run(reservations, budget=4 * GIB, cap=2)

    assert [r["index"] for r in out] == [0, 1]
    first = logs[0]
    assert first == zpar.dispatch_log_line(
        "cell-0", 1500 * MIB, 1500 * MIB, 1, 4 * GIB
    )
    assert first.startswith("ZeGrid: cell dispatch — cell=cell-0 ")
    for field in ("reservation=", "running=", "k=1", "budget="):
        assert field in first
    # the second cell sees the first one already in flight
    assert "k=2" in logs[1]

    # a plan/intent helper stays a pure function (no globals, no clock)
    assert zpar.dispatch_log_line("c", 1 * MIB, 2 * MIB, 1, 3 * MIB) == (
        "ZeGrid: cell dispatch — cell=c reservation=1.0MiB running=2.0MiB "
        "k=1 budget=3.0MiB"
    )


# ---------------------------------------------------------------------------
# budget helper: the admission predicate itself
# ---------------------------------------------------------------------------

def test_dispatch_fits_predicate_counts_shared_once():
    assert zpar.dispatch_fits(0, 3, 1, 4) is True
    assert zpar.dispatch_fits(1, 3, 1, 4) is False
    assert zpar.dispatch_fits(10 ** 12, 10 ** 12, 0, None) is True
