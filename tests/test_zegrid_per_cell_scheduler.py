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
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import auto_layout as zal  # noqa: E402
from zemosaic.core.zegrid import parallel as zpar  # noqa: E402
from zemosaic.core.zegrid import runtime_control as zrt  # noqa: E402

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


# ---------------------------------------------------------------------------
# Rework-1 (M1): an IN-MEMORY plan can never enter the per-cell queue
# ---------------------------------------------------------------------------

def test_inmem_plan_is_structurally_excluded_from_the_dispatch_queue():
    reservations = {"cell-0": 8 * GIB}
    # M1: dispatching an in-memory plan against stream-sized reservations could
    # oversubscribe RAM. The gate must be False regardless of cap/task count.
    assert zpar.dispatch_is_applicable("inmem", reservations, 8, 40) is False
    assert zpar.dispatch_is_applicable("inmem", reservations, 2, 2) is False
    # …and the reservation itself uses the IN-MEMORY bound for that mode.
    assert zpar.dispatch_reservation_bound(8 * GIB, 2 * GIB, "inmem") == 8 * GIB
    assert zpar.dispatch_reservation_bound(8 * GIB, 2 * GIB, "stream") == 2 * GIB
    # the streaming plan keeps the queue
    assert zpar.dispatch_is_applicable("stream", reservations, 8, 40) is True
    # no reservations / a single cell / a cap of 1 keep the historical path
    assert zpar.dispatch_is_applicable("stream", {}, 8, 40) is False
    assert zpar.dispatch_is_applicable("stream", reservations, 8, 1) is False
    assert zpar.dispatch_is_applicable("stream", reservations, 1, 40) is False


# ---------------------------------------------------------------------------
# Rework-1: a streaming-mode plan still dispatches per cell
# ---------------------------------------------------------------------------

def test_streaming_plan_still_dispatches_per_cell():
    assert zpar.dispatch_is_applicable("stream", {"c": 1 * GIB}, 4, 4)
    out, meta, logs = _run([1 * GIB] * 4, budget=6 * GIB, cap=4, sleep=0.5)
    assert [r["index"] for r in out] == [0, 1, 2, 3]
    assert meta["max_concurrent"] > 1
    assert meta["scheduler"] == "per_cell_reservation"


# ---------------------------------------------------------------------------
# Rework-1 (L2): a THREAD fallback warns ONCE and still completes
# ---------------------------------------------------------------------------

def test_thread_executor_fallback_warns_once_and_completes(monkeypatch):
    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: True)
    tasks = _tasks(3, sleep=0.05)
    logs = []
    meta = {}
    out = zpar.pmap_reserved(
        _sleep_worker, tasks, [1 * GIB] * 3,
        shared_bytes=0, decision_budget_bytes=6 * GIB, workers=3,
        cell_ids=[t["cell"] for t in tasks],
        emit=lambda msg, lvl="INFO": logs.append((lvl, msg)),
        meta=meta,
    )
    assert [r["index"] for r in out] == [0, 1, 2]
    assert meta["executor"] == "thread"
    warnings = [
        msg for lvl, msg in logs
        if lvl == "WARN" and "thread executor" in msg
    ]
    assert warnings == [zpar.THREAD_EXECUTOR_WARNING]


# ---------------------------------------------------------------------------
# Rework-2: the EXTERNAL class (the user's real run) must dispatch per cell
# ---------------------------------------------------------------------------

def _external_block(n_cells=8, reservation=2 * GIB, budget=16667 * MIB):
    return zpar.external_dispatch_reservation(
        reservations_by_cell={f"cell-{i}": reservation for i in range(n_cells)},
        budget_bytes=budget,
        shared_bytes=0,
        cpu_count=16,
        cells_in_flight=1,
        source="model",
    )


def test_external_plan_publishes_a_sound_reservation_block_and_log_line():
    block = _external_block()
    assert block is not None
    assert block["reservation_bytes"] == 2 * GIB
    assert block["decision_budget_bytes"] == int(16667 * MIB * 0.9)
    assert block["dispatch_cap"] > 1
    # rework-3: the cap is the CPU bound (min(cpu 14, 8 cells) = 8), NOT the
    # reservation-derived budget: the budget is enforced per cell at dispatch.
    assert block["dispatch_cap"] == 8
    assert "external dispatch cap" in block["reason"]

    line = zpar.reservation_log_line(
        {"mode": "external", "cells_in_flight": 1, "reservation": block}
    )
    assert line is not None
    assert "dispatch_cap=8" in line
    assert "cell=cell-0" in line
    assert "budget=16667.0MiB" in line

    # no sound reservation => None (the caller keeps the historical serial path)
    assert zpar.external_dispatch_reservation(
        reservations_by_cell={}, budget_bytes=1 * GIB, shared_bytes=0, cpu_count=16
    ) is None
    assert zpar.external_dispatch_reservation(
        reservations_by_cell={"c": 0}, budget_bytes=1 * GIB, shared_bytes=0, cpu_count=16
    ) is None


def test_external_mode_dispatches_per_cell():
    assert zpar.dispatch_is_applicable(
        "external", {"a": 1 * GIB}, 7, 8
    ) is True
    # inmem stays structurally excluded (M1 regression)
    assert zpar.dispatch_is_applicable(
        "inmem", {"a": 1 * GIB}, 7, 8
    ) is False
    out, meta, logs = _run(
        [2 * GIB] * 6, budget=16667 * MIB, cap=6, sleep=0.5
    )
    assert [r["index"] for r in out] == list(range(6))
    assert meta["scheduler"] == "per_cell_reservation"
    assert meta["max_concurrent"] > 1


def test_external_concurrent_reservations_never_exceed_the_decision_budget():
    # ``decision_budget_bytes`` is ALREADY the 90 % figure (the production
    # caller passes 0.9 x plan budget, exactly like the phase planner).
    decision = int(16667 * MIB * 0.9)
    out, meta, logs = _run(
        [2 * GIB] * 8, budget=decision, cap=8, sleep=0.5
    )
    assert [r["index"] for r in out] == list(range(8))
    for event in meta["events"]:
        if event["event"] == "dispatch":
            assert event["running_bytes"] <= decision
    # 7 x 2 GiB fits (14 GiB <= 14.65 GiB), an 8th does not
    assert meta["max_concurrent"] == 7


def test_external_path_is_byte_identical_sequential_vs_dispatched():
    tasks = _tasks(6)
    seq = [zpar.pmap(_plane_worker, [t], workers=1)[0] for t in tasks]
    meta = {}
    sched = zpar.pmap_reserved(
        _plane_worker, tasks, [2 * GIB] * 6,
        shared_bytes=0, decision_budget_bytes=int(16667 * MIB * 0.9), workers=6,
        cell_ids=[t["cell"] for t in tasks], meta=meta,
    )
    assert zpar.dispatch_is_applicable("external", {"x": 1}, 6, 6)
    assert [s["digest"] for s in sched] == [s["digest"] for s in seq]
    assert [s["nbytes"] for s in sched] == [s["nbytes"] for s in seq]
    assert meta["max_concurrent"] > 1


# ---------------------------------------------------------------------------
# Rework-3: HEAVY-TAIL regression — the cap must not be reservation-derived
# ---------------------------------------------------------------------------

def test_heavy_tail_cap_is_cpu_bound_and_lights_still_overlap():
    heavy = 14 * GIB
    light = 2 * GIB
    reservations = [heavy] + [light] * 7
    block = zpar.external_dispatch_reservation(
        reservations_by_cell={
            f"cell-{i}": r for i, r in enumerate(reservations)
        },
        budget_bytes=16667 * MIB,
        shared_bytes=0,
        cpu_count=8,
        cells_in_flight=1,
        n_cells=len(reservations),
    )
    assert block is not None
    # CPU cap = min(8 - 2 OS reserve, 8 cells) = 6 — NOT 1 from the 14 GiB cell.
    assert block["dispatch_cap"] == 6
    assert block["dispatch_cap"] > 1
    # the max reservation and its cell are still published (informative)
    assert block["reservation_bytes"] == heavy
    assert block["cell_id"] == "cell-0"

    tasks = _tasks(8)
    tasks[0]["sleep"] = 0.7
    for t in tasks[1:]:
        t["sleep"] = 0.35
    out, meta, logs = _run(
        reservations, budget=int(16667 * MIB * 0.9), cap=block["dispatch_cap"],
        tasks=tasks,
    )
    assert [r["index"] for r in out] == list(range(8))
    events = meta["events"]
    first = events[0]
    assert first["event"] == "dispatch" and first["index"] == 0
    assert first["k"] == 1                      # the heavy cell runs ALONE
    done_heavy = next(
        i for i, e in enumerate(events)
        if e["event"] == "done" and e["index"] == 0
    )
    light_dispatches = [
        i for i, e in enumerate(events)
        if e["event"] == "dispatch" and e["index"] != 0
    ]
    assert light_dispatches and min(light_dispatches) > done_heavy
    # …then the light cells overlap (up to the CPU cap)
    assert meta["max_concurrent"] >= 2
    assert meta["max_concurrent"] == 6
    # and no dispatch ever exceeded the decision budget
    for e in events:
        if e["event"] == "dispatch":
            assert e["running_bytes"] <= int(16667 * MIB * 0.9)


# ---------------------------------------------------------------------------
# Rework-4: the WAVE must be sliced by the DISPATCH cap, not the plan-wide K
# ---------------------------------------------------------------------------

def test_dispatchable_plan_slices_waves_by_the_dispatch_cap():
    # Production defect #5: the external plan has K=1, so every wave was ONE
    # cell and pmap ran serial (tasks=1). The wave bound is now the cap.
    assert zpar.dispatch_wave_bound("external", 6, 1) == 6
    assert zpar.dispatch_wave_bound("stream", 6, 1) == 6

    total, cap = 12, 6
    offset, waves = 0, []
    while offset < total:
        bound = zpar.dispatch_wave_bound("external", cap, 1)
        start, end = zrt.next_wave_bound(offset, total, bound)
        waves.append((start, end))
        offset = end
    # two waves of six cells, never twelve waves of one
    assert waves == [(0, 6), (6, 12)]
    assert [end - start for start, end in waves] == [6, 6]

    # …and a wave of six really overlaps when dispatched
    out, meta, logs = _run([2 * GIB] * 6, budget=int(16667 * MIB * 0.9), cap=6,
                           sleep=0.4)
    assert [r["index"] for r in out] == list(range(6))
    assert meta["max_concurrent"] > 1


def test_non_dispatchable_plan_keeps_the_historical_wave_bound():
    # inmem: plan-wide K governs (M1 stays closed)
    assert zpar.dispatch_wave_bound("inmem", 6, 1) == 1
    # no cap published / cap of 1: historical K
    assert zpar.dispatch_wave_bound("stream", 0, 3) == 3
    assert zpar.dispatch_wave_bound("external", 1, 4) == 4
    # a legacy/unknown mode keeps the historical K
    assert zpar.dispatch_wave_bound("legacy", 8, 2) == 2


def test_multi_wave_run_is_byte_identical_to_the_sequential_run():
    tasks = _tasks(12)
    seq = [zpar.pmap(_plane_worker, [t], workers=1)[0] for t in tasks]
    out = []
    meta_all = []
    for start in (0, 6):
        meta = {}
        out.extend(
            zpar.pmap_reserved(
                _plane_worker, tasks[start:start + 6], [2 * GIB] * 6,
                shared_bytes=0,
                decision_budget_bytes=int(16667 * MIB * 0.9),
                workers=zpar.dispatch_wave_bound("external", 6, 1),
                cell_ids=[t["cell"] for t in tasks[start:start + 6]],
                meta=meta,
            )
        )
        meta_all.append(meta)
    assert [r["index"] for r in out] == list(range(12))
    assert [r["digest"] for r in out] == [r["digest"] for r in seq]
    assert all(m["max_concurrent"] > 1 for m in meta_all)


# ---------------------------------------------------------------------------
# Rework-5: the durable commit peak lives at payload_meta.record.memory_peak
# ---------------------------------------------------------------------------

def _commit_record(payload_meta):
    return SimpleNamespace(manifest={"payload_meta": payload_meta})


_MANIFEST_ROW = {
    "n_contributors": 40,
    "n_patch_contributors": 40,
    "patch_shape_hw": [901, 1006],
}


def test_observations_read_the_real_commit_field_path():
    # The writer (mode.py ~:6549) puts the peak INSIDE record["memory_peak"].
    obs = zal.memory_observations_from_commit_records([
        _commit_record({
            "record": {
                **_MANIFEST_ROW,
                "memory_peak": {
                    "owned_peak_kib": 2_232_320,     # 2 180 MiB owned
                    "peak_rss_kib": 3_000_000,
                },
            }
        })
    ])
    assert len(obs) == 1
    assert obs[0]["n_contributors"] == 40
    assert obs[0]["patch_area_px"] == 901 * 1006
    assert obs[0]["owned_peak_bytes"] == 2_232_320 * 1024


def test_observations_fall_back_to_peak_rss_when_owned_is_absent():
    # (a) record.memory_peak.peak_rss_kib (no owned decomposition)
    obs = zal.memory_observations_from_commit_records([
        _commit_record({
            "record": {**_MANIFEST_ROW, "memory_peak": {"peak_rss_kib": 1_500_000}},
        })
    ])
    assert obs and obs[0]["owned_peak_bytes"] == 1_500_000 * 1024

    # (b) legacy top-level payload_meta.peak_rss_kib
    obs2 = zal.memory_observations_from_commit_records([
        _commit_record({
            "record": dict(_MANIFEST_ROW),
            "peak_rss_kib": 900_000,
        })
    ])
    assert obs2 and obs2[0]["owned_peak_bytes"] == 900_000 * 1024

    # a record with no peak at all is skipped (never guessed)
    assert zal.memory_observations_from_commit_records([
        _commit_record({"record": dict(_MANIFEST_ROW)})
    ]) == []


def test_measured_observation_tightens_the_admitted_concurrency():
    # The fixed path turns the real 2 180 MiB owned peak into the 2 507 MiB
    # envelope; the deep shape gets 3 760 MiB. That is what the plan missed.
    obs = zal.memory_observations_from_commit_records([
        _commit_record({
            "record": {
                **_MANIFEST_ROW,
                "memory_peak": {"owned_peak_kib": 2_232_320},
            }
        })
    ])
    stream_bound = int(780.6 * MIB)
    reservation = zal.reservation_bytes(
        40, 906_406, stream_bound, observations=obs
    )
    assert abs(reservation - 2_507 * MIB) <= 1 * MIB
    deep = zal.reservation_bytes(21_807, 906_406, stream_bound, observations=obs)
    assert abs(deep - 3_760 * MIB) <= 1 * MIB

    decision = int(16_667 * MIB * 0.9)
    # CPU cap with 16 logical CPUs is 14 (the pool size)…
    block = zpar.external_dispatch_reservation(
        reservations_by_cell={f"c{i}": reservation for i in range(20)},
        budget_bytes=16_667 * MIB, shared_bytes=0, cpu_count=16, n_cells=20,
    )
    assert block["dispatch_cap"] == 14
    # …but the BUDGET admits only 5 cells at 2 507 MiB (6 would exceed it),
    # NOT the 14 the buggy 750 MiB model allowed.
    _out, meta, _logs = _run(
        [reservation] * 20, budget=decision, cap=block["dispatch_cap"], sleep=0.35
    )
    assert meta["max_concurrent"] == 5
    assert meta["max_concurrent"] <= 6

    _out2, meta_old, _logs2 = _run(
        [750 * MIB] * 20, budget=decision, cap=block["dispatch_cap"], sleep=0.35
    )
    assert meta_old["max_concurrent"] == 14   # the defect, for contrast
