"""ZM-CELL-LIVE-OBS (lot 2) — truthful, bounded live per-cell status.

Pinned contracts:

A. states / counts
   * QUEUED, SUBMITTED, RUNNING, COMPUTED, COMMITTING, COMMITTED, FAILED and
     CANCELLED are DISTINCT; SUBMITTED is a real dispatch and RUNNING requires a
     real worker start (never a whole-batch announcement);
   * every plan cell appears EXACTLY once, resumed and new commits are separate
     and the published counts always add up to the plan total;
   * a stale/late event can never rewind a committed cell, and a running worker
     is never relabelled CANCELLED (only a future the parent really cancelled).

B. worker telemetry transport (small, bounded, spawn/Qt-safe)
   * the writer rate-limits and coalesces progress, has a hard byte bound and
     never carries an image/array/dtype;
   * the reader only reads NEW bytes of named cells, validates run/generation/
     cell identity and rejects stale lines;
   * the transport crosses a real SPAWN process boundary (plain path + scalars
     only), and a telemetry failure can never change a result.

C. executor seam
   * pmap/pmap_reserved emit SUBMITTED/COMPUTED and a fatal error is reported
     BEFORE the executor drain, while a slow sibling is still blocked;
   * a sibling WORKER failure is labelled as a worker failure, never as a
     consumer failure.

D. external streaming
   * the true current write/reduce/finalize phase plus tile i/n and block j/m are
     reported (never the historical cumulative START/END only), and the probe
     cannot change a single scientific value;
   * the existing ``phases.log`` breadcrumb is untouched.

E. production seam (real run, real durable store, tiny corpus)
   * while a slow cell is gated inside its worker, the run log shows RUNNING +
     phase for it and committed counts for the fast cells, and it is NOT
     COMMITTED;
   * a resumed run reports resumed commits and new commits separately.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _path in (SRC, TESTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import cell_status as zcs  # noqa: E402
from zemosaic.core.zegrid import parallel as zpar  # noqa: E402

import zegrid_live_commit_harness as GH  # noqa: E402

CORPUS_TOOL = REPO_ROOT / "tools" / "zegrid_m6" / "make_tiny_corpus.py"
ALL_CELLS = ("r0000c0000", "r0000c0001", "r0001c0000", "r0001c0001")


# ---------------------------------------------------------------------------
# A. states / counts
# ---------------------------------------------------------------------------


def test_lifecycle_kinds_are_distinct_and_truthful():
    assert zpar.cell_lifecycle_line("QUEUED", "c1", frames=9, mode="external") == (
        "ZeGrid: cell QUEUED id=c1 frames=9 mode=external"
    )
    assert zpar.cell_lifecycle_line(
        "SUBMITTED", "c1", k=3, reservation_bytes=1024
    ) == "ZeGrid: cell SUBMITTED id=c1 k=3 reservation=1024B"
    assert zpar.cell_lifecycle_line(
        "RUNNING", "c1", phase="reduce", generation=7
    ) == "ZeGrid: cell RUNNING id=c1 phase=reduce gen=7"
    assert zpar.cell_lifecycle_line(
        "COMMITTING", "c1", generation=7
    ) == "ZeGrid: cell COMMITTING id=c1 gen=7"
    assert zpar.cell_lifecycle_line(
        "FAILED", "c1", stage="worker", error=ValueError("boom")
    ) == "ZeGrid: cell FAILED id=c1 stage=worker error=ValueError"
    assert zpar.cell_lifecycle_line("CANCELLED", "c1", generation=7) == (
        "ZeGrid: cell CANCELLED id=c1 gen=7 (never started)"
    )
    # COMPUTED != COMMITTED (the lot-1 contract is preserved byte-for-byte).
    assert zpar.cell_lifecycle_line(
        "COMPUTED", "c1", stack_s=1.25, status="complete"
    ) == "ZeGrid: cell COMPUTED id=c1 stack_s=1.2 status=complete"
    assert zpar.cell_lifecycle_line(
        "COMMITTED", "c1", attempt="a", generation=2, payload_sha="deadbeef" * 4
    ) == "ZeGrid: cell COMMITTED id=c1 attempt=a gen=2 payload=deadbeefdeadbeef"


def test_fatal_line_says_it_does_not_kill_a_started_worker():
    line = zpar.fatal_lifecycle_line(
        RuntimeError("commit lost"), cell_id="c9", stage="consume"
    )
    assert "FATAL consume failure at cell=c9" in line
    assert "drained" in line and "no forced kill" in line


def test_state_counts_are_unique_and_split_resumed_and_new():
    states = zcs.LiveCellStates()
    states.reset(["a", "b", "c", "d"], committed=["a"])
    assert states.snapshot()["committed"] == 1
    assert states.snapshot()["resumed_committed"] == 1
    assert states.snapshot()["new_committed"] == 0
    states.update("b", zcs.RUNNING)
    states.update("c", zcs.RUNNING)
    states.update("c", zcs.COMMITTED)
    snap = states.snapshot()
    assert snap["total"] == 4
    assert sum(
        snap[key]
        for key in (
            "queued", "submitted", "running", "computed", "committing",
            "committed", "failed", "cancelled",
        )
    ) == 4
    assert snap["resumed_committed"] == 1
    assert snap["new_committed"] == 1
    assert snap["running"] == 1 and snap["queued"] == 1
    # Unknown cells are never invented.
    assert states.update("zz", zcs.RUNNING) is False
    assert states.snapshot()["total"] == 4
    line = zcs.live_status_line(snap, generation=3)
    assert "total=4" in line and "resumed=1 new=1" in line and "gen=3" in line


def test_stale_state_cannot_rewind_and_committed_is_terminal():
    states = zcs.LiveCellStates()
    states.reset(["a"])
    assert states.update("a", zcs.RUNNING) is True
    assert states.update("a", zcs.SUBMITTED) is False  # older
    assert states.update("a", zcs.COMMITTED) is True
    assert states.update("a", zcs.RUNNING) is False
    assert states.update("a", zcs.FAILED) is False
    assert states.snapshot()["committed"] == 1
    # CANCEL is accepted only before terminal COMMITTED.
    states.reset(["b"])
    states.update("b", zcs.SUBMITTED)
    assert states.update("b", zcs.CANCELLED) is True
    assert states.snapshot()["cancelled"] == 1


def test_progress_facts_are_attached_without_changing_the_state():
    states = zcs.LiveCellStates()
    states.reset(["a"])
    states.update("a", zcs.RUNNING)
    assert states.note_progress(
        "a", phase="reduce", tile_current=2, tile_total=12,
        block_current=5, block_total=60, elapsed_s=9.5,
    )
    row = states.snapshot()["rows"][0]
    assert row["state"] == zcs.RUNNING
    assert row["tile"] == [2, 12] and row["block"] == [5, 60]
    assert zcs.live_cell_line(row) == (
        "ZeGrid: cell a RUNNING phase=reduce tile=2/12 block=5/60 elapsed_s=9.5"
    )


# ---------------------------------------------------------------------------
# B. writer / reader (rate limit, bounds, identity)
# ---------------------------------------------------------------------------


def _write_lines(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_writer_rate_limits_coalesces_and_records_transitions(tmp_path):
    path = tmp_path / "c1.jsonl"
    writer = zcs.LiveTelemetryWriter(
        path, run_id="run", generation=4, cell_id="c1", min_interval_s=3600.0
    )
    assert writer.state("RUNNING") is True
    # Inside the huge window, identical and different progress is coalesced.
    first = writer.progress("write", tile_current=1, tile_total=10)
    assert first is True
    for tile in range(2, 10):
        assert writer.progress("write", tile_current=tile, tile_total=10) is False
    # An explicit boundary is always visible.
    assert writer.progress(
        "finalize", force=True, elapsed_s=3.0
    ) is True
    payloads = _write_lines(path)
    kinds = [payload["kind"] for payload in payloads]
    assert kinds == ["STATE", "PROGRESS", "PROGRESS"]
    assert payloads[0]["state"] == "RUNNING" and payloads[0]["gen"] == 4
    assert payloads[1]["tile"] == [1, 10]
    assert payloads[2]["phase"] == "finalize"
    assert writer.progress_suppressed >= 8
    # Nothing but scalars/identifiers: no array, no dtype, no science value.
    text = path.read_text(encoding="utf-8")
    assert "ndarray" not in text and "dtype" not in text
    for payload in payloads:
        for value in payload.values():
            assert not isinstance(value, (np.ndarray, bytes, bytearray))


def test_writer_rotates_and_never_stops_at_a_fixed_capacity(tmp_path):
    """R1 (rework-1): a multi-hour cell must stay live, not hit a quota.

    Simulated duration (fake clock), not a sleep: 6000 samples 2.1 s apart is
    ~3.5 h of a single cell, far beyond any single segment and far beyond the
    old 256 KiB permanent cutoff.
    """
    now = [0.0]
    path = tmp_path / "c1.jsonl"
    writer = zcs.LiveTelemetryWriter(
        path, run_id="run", generation=1, cell_id="c1",
        min_interval_s=2.0, max_segment_bytes=2048, max_segments=3,
        clock=lambda: now[0],
    )
    assert writer.state("RUNNING") is True
    written = 1
    for tile in range(1, 4001):
        now[0] += 2.1
        if writer.progress("write", tile_current=tile, tile_total=4000):
            written += 1
    # The stream is STILL alive after thousands of samples.
    assert writer.enabled is True
    assert writer.rotations >= 10
    assert written > 1000
    # Disk stays bounded: the ring, not an ever-growing file.
    segments = sorted(tmp_path.glob("c1.jsonl*"))
    assert len(segments) <= 3
    assert sum(p.stat().st_size for p in segments) <= 3 * 2048 + 1024

    # A LATE finalize (hours later) is still written and readable...
    now[0] += 2.1
    assert writer.progress("finalize", force=True, elapsed_s=1.5) is True
    reader = zcs.LiveTelemetryReader(tmp_path, run_id="run", generation=1)
    events = reader.poll(["c1"])
    # ... and RUNNING is recovered across the rotation (re-emitted).
    assert events["c1"]["state"]["state"] == "RUNNING"
    assert events["c1"]["progress"]["phase"] == "finalize"


@pytest.mark.parametrize("segment_bytes", [512, 1024, 4096, 65536])
def test_rotation_keeps_the_current_tile_visible_for_any_capacity(
    tmp_path, segment_bytes
):
    now = [0.0]
    path = tmp_path / "c.jsonl"
    writer = zcs.LiveTelemetryWriter(
        path, run_id="run", generation=2, cell_id="c", min_interval_s=1.0,
        max_segment_bytes=segment_bytes, max_segments=2, clock=lambda: now[0],
    )
    writer.state("RUNNING")
    for tile in range(1, 2001):
        now[0] += 1.5
        writer.progress("reduce", tile_current=tile, tile_total=4000,
                        block_current=tile, block_total=4000)
    reader = zcs.LiveTelemetryReader(tmp_path, run_id="run", generation=2)
    events = reader.poll(["c"])
    assert events["c"]["progress"]["tile"] == [2000, 4000]
    assert events["c"]["state"]["state"] == "RUNNING"
    # Bounded disk for EVERY capacity (never a single unbounded file).
    assert sum(
        p.stat().st_size for p in sorted(tmp_path.glob("c.jsonl*"))
    ) <= 2 * max(512, segment_bytes) + 1024


def test_lagging_reader_across_many_rotations_still_recovers_running(tmp_path):
    now = [0.0]
    writer = zcs.LiveTelemetryWriter(
        tmp_path / "c.jsonl", run_id="run", generation=3, cell_id="c",
        min_interval_s=0.0, max_segment_bytes=1024, max_segments=2,
        clock=lambda: now[0],
    )
    writer.state("RUNNING")
    writer.progress("reduce", tile_current=1, tile_total=999)
    # The reader does not poll for a very long time (many rotations happen).
    for tile in range(2, 1001):
        now[0] += 1.0
        writer.progress("reduce", tile_current=tile, tile_total=999,
                        force=True)
    reader = zcs.LiveTelemetryReader(tmp_path, run_id="run", generation=3)
    events = reader.poll(["c"])
    assert events["c"]["state"]["state"] == "RUNNING"
    assert events["c"]["progress"]["tile"][0] >= 1
    # A second poll consumes everything already seen (no repeated work).
    assert reader.poll(["c"]) == {}


def test_reader_survives_a_torn_partial_line(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    path = zcs.live_cell_path(live, "c")
    writer = zcs.LiveTelemetryWriter(
        path, run_id="run", generation=0, cell_id="c", min_interval_s=0.0
    )
    writer.progress("write", tile_current=1, tile_total=5, force=True)
    # Simulate an interrupted write: a torn JSON prefix with NO newline.
    with path.open("ab") as handle:
        handle.write(b'{"schema":"zemosaic-zegrid-live-status-v1","kind":"PROG')
    writer.progress("write", tile_current=2, tile_total=5, force=True)
    reader = zcs.LiveTelemetryReader(live, run_id="run", generation=0)
    events = reader.poll(["c"])
    # The torn prefix is skipped and the FOLLOWING valid line is still used.
    assert events["c"]["progress"]["tile"] == [2, 5]
    assert reader.rejected_stale == 0


def test_reader_accepts_only_current_identity_and_returns_the_last_event(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    path = zcs.live_cell_path(live, "c1")
    writer = zcs.LiveTelemetryWriter(
        path, run_id="run", generation=2, cell_id="c1", min_interval_s=0.0
    )
    writer.progress("write", tile_current=1, tile_total=5, force=True)
    reader = zcs.LiveTelemetryReader(live, run_id="run", generation=2)
    events = reader.poll(["c1"])
    assert events["c1"]["progress"]["tile"] == [1, 5]

    # A stale generation and a foreign run id are REJECTED.
    writer.state("RUNNING")  # still gen=2
    reader.retarget(generation=3)
    assert reader.poll(["c1"]) == {}
    assert reader.rejected_stale >= 1

    # A different run reusing the same directory is rejected too.
    other = zcs.LiveTelemetryWriter(
        path, run_id="other-run", generation=3, cell_id="c1",
        min_interval_s=0.0,
    )
    other.progress("reduce", tile_current=2, tile_total=5, force=True)
    assert reader.poll(["c1"]) == {}
    assert reader.rejected_stale >= 2


def test_reader_reads_only_new_bounded_bytes(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    path = zcs.live_cell_path(live, "c1")
    writer = zcs.LiveTelemetryWriter(
        path, run_id="run", generation=0, cell_id="c1", min_interval_s=0.0
    )
    for tile in range(50):
        writer.progress("write", tile_current=tile, tile_total=50, force=True)
    reader = zcs.LiveTelemetryReader(
        live, run_id="run", generation=0, max_read_bytes=400
    )
    first = reader.poll(["c1"])
    assert first, "the first bounded read must still deliver a sample"
    # The first chunk is bounded, so the LATEST tile is not necessarily seen yet.
    assert first["c1"]["progress"]["tile"][0] < 50
    # The first bounded chunk also carried the state transition (if any).
    for _ in range(200):
        events = reader.poll(["c1"])
        if (
            events
            and events["c1"]["progress"]
            and events["c1"]["progress"]["tile"][0] == 49
        ):
            break
    else:  # pragma: no cover - defensive
        raise AssertionError("the reader never caught up on new bytes")
    assert reader.poll(["c1"]) == {}  # fully consumed: no repeated work


def test_apply_events_only_advances_running_and_terminal_states():
    states = zcs.LiveCellStates()
    states.reset(["c1"])
    zcs.apply_events(
        states,
        {"c1": {
            "state": {"schema": zcs.LIVE_STATUS_SCHEMA, "kind": "STATE",
                      "state": "RUNNING", "gen": 1, "cell_id": "c1",
                      "run_id": "r"},
            "progress": None,
        }},
    )
    assert states.snapshot()["running"] == 1
    zcs.apply_events(
        states,
        {"c1": {
            "state": None,
            "progress": {"schema": zcs.LIVE_STATUS_SCHEMA, "kind": "PROGRESS",
                         "phase": "reduce", "tile": [3, 9], "block": [1, 4],
                         "gen": 1, "cell_id": "c1", "run_id": "r"},
        }},
    )
    assert states.snapshot()["rows"][0]["block"] == [1, 4]
    # A worker file can never claim COMMITTED: only the parent transaction can.
    zcs.apply_events(
        states,
        {"c1": {
            "state": {"schema": zcs.LIVE_STATUS_SCHEMA, "kind": "STATE",
                      "state": "COMMITTED", "gen": 1, "cell_id": "c1",
                      "run_id": "r"},
            "progress": None,
        }},
    )
    assert states.snapshot()["committed"] == 0


# ---------------------------------------------------------------------------
# B2. real spawn boundary + telemetry failure neutrality
# ---------------------------------------------------------------------------


def spawn_writer_child(payload):
    """Module-level target for a REAL spawn child (no inherited module state)."""
    writer = zcs.LiveTelemetryWriter(
        payload["path"], run_id=payload["run_id"],
        generation=payload["gen"], cell_id=payload["cell_id"],
        min_interval_s=0.0,
    )
    writer.state("RUNNING")
    writer.progress("cache", tile_current=1, tile_total=1, force=True)
    writer.progress("finalize", force=True, elapsed_s=0.5)
    return 0


def test_transport_crosses_a_real_spawn_process(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    live = tmp_path / "live"
    live.mkdir()
    payload = {
        "path": str(zcs.live_cell_path(live, "c-spawn")),
        "run_id": "run", "gen": 5, "cell_id": "c-spawn",
    }
    proc = ctx.Process(target=spawn_writer_child, args=(payload,))
    proc.start()
    proc.join(timeout=120)
    assert proc.exitcode == 0
    reader = zcs.LiveTelemetryReader(live, run_id="run", generation=5)
    events = reader.poll(["c-spawn"])
    assert events["c-spawn"]["progress"]["phase"] == "finalize"
    assert events["c-spawn"]["state"]["state"] == "RUNNING"
    assert events["c-spawn"]["state"]["gen"] == 5


def _telemetry_worker(task):
    writer = zcs.LiveTelemetryWriter(
        task["path"], run_id="run", generation=1, cell_id=task["cell_id"],
        min_interval_s=0.0,
    )
    writer.state("RUNNING")
    writer.progress("stack", tile_current=1, tile_total=1, force=True)
    if task["boom"]:
        raise ValueError("worker boom")
    return {"cell_id": task["cell_id"], "value": int(task["value"])}


def test_telemetry_failure_never_changes_a_result(tmp_path):
    # The directory is a FILE: every open/mkdir is impossible.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    tasks = [
        {"path": str(blocker / "c1.jsonl"), "cell_id": "c1",
         "value": 41, "boom": False},
        {"path": str(blocker / "c2.jsonl"), "cell_id": "c2",
         "value": 1, "boom": False},
    ]
    out = zpar.pmap(_telemetry_worker, tasks, workers=1)
    assert [row["value"] for row in out] == [41, 1]


# ---------------------------------------------------------------------------
# C. executor seam
# ---------------------------------------------------------------------------


def test_pmap_reserved_reports_submitted_computed_failed_and_run_bound_cancel(
    monkeypatch,
):
    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: True)  # threads

    seen: list[dict] = []
    release = threading.Event()
    started = threading.Event()

    def worker(task):
        if task == "slow":
            started.set()
            assert release.wait(120)
            return "slow"
        if task == "boom":
            raise ValueError("boom")
        return task

    def consume(index, value):
        return value

    # The slow worker is released by a TIMER so the drain can finish; the point
    # of the test is that a STARTED worker is never relabelled CANCELLED.
    timer = threading.Timer(1.0, release.set)
    timer.daemon = True
    timer.start()
    try:
        with pytest.raises(zpar.ParallelTaskError):
            zpar.pmap_reserved(
                worker, ["slow", "boom"], [1, 1],
                workers=2, cell_ids=["c-slow", "c-boom"],
                consume=consume, state_cb=seen.append,
            )
    finally:
        release.set()
        timer.cancel()
    assert started.is_set()
    events = {(event["event"], event.get("cell_id")) for event in seen}
    assert ("submitted", "c-slow") in events
    assert ("submitted", "c-boom") in events
    assert ("failed", "c-boom") in events
    # A STARTED worker is never relabelled CANCELLED.
    assert ("cancelled", "c-slow") not in events


def test_pmap_never_cancels_a_started_worker_but_does_cancel_a_pending_one(
    monkeypatch,
):
    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: True)
    release = threading.Event()
    seen: list[dict] = []

    def worker(task):
        if task == "block":
            assert release.wait(120)
            return "block"
        if task == "boom":
            raise ValueError("boom")
        return task

    timer = threading.Timer(1.0, release.set)
    timer.daemon = True
    timer.start()
    try:
        with pytest.raises(zpar.ParallelTaskError):
            # 2 workers, 3 tasks: 'block' and 'boom' really start, the third one
            # never starts and is therefore the ONLY honest CANCELLED.
            zpar.pmap(
                worker, ["block", "boom", "queued"], workers=2,
                consume=lambda index, value: value, state_cb=seen.append,
            )
    finally:
        release.set()
        timer.cancel()
    failed = [event for event in seen if event["event"] == "failed"]
    assert failed and failed[0]["index"] == 1
    cancelled = [event for event in seen if event["event"] == "cancelled"]
    assert cancelled and all(event["index"] == 2 for event in cancelled)


def test_fatal_is_reported_before_the_drain_while_a_sibling_is_still_blocked(
    monkeypatch,
):
    """Nono LOW-1: the fatal state must be visible BEFORE shutdown(wait=True)."""
    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: True)
    release = threading.Event()
    started = threading.Event()
    fatal = threading.Event()
    reported: dict = {}

    def worker(task):
        if task == "slow":
            started.set()
            # Held for LONGER than any reasonable drain budget.
            assert release.wait(300)
            return "slow"
        return "fast"

    def consume(index, value):
        if value == "fast":
            raise RuntimeError("durable commit refused")
        return value

    def on_fatal(event):
        reported.update(event)
        fatal.set()

    errors: list[BaseException] = []
    done = threading.Event()

    def run():
        try:
            zpar.pmap_reserved(
                worker, ["slow", "fast"], [1, 1],
                workers=2, cell_ids=["c-slow", "c-fast"],
                consume=consume, on_fatal=on_fatal,
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert started.wait(60), "the slow worker never started"
        assert fatal.wait(60), "the fatal state was NOT reported before the drain"
        assert reported.get("stage") == "consume"
        assert reported.get("cell_id") == "c-fast"
        assert isinstance(reported.get("error"), RuntimeError)
        # The slow worker is STILL blocked and the wave has NOT returned: the
        # report really happened before the drain, not after it.
        assert not release.is_set()
        assert not done.is_set()
    finally:
        release.set()
        assert done.wait(120)
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert "durable commit refused" in str(errors[0])


def test_sibling_worker_failure_is_labelled_worker_not_consumer():
    """Nono LOW-2, pinned DIRECTLY on the sibling-collection helper."""
    import concurrent.futures as cf

    worker_failed = cf.Future()
    worker_failed.set_exception(ValueError("sibling boom"))
    worker_ok = cf.Future()
    worker_ok.set_result("ok")
    running = {worker_failed: 0, worker_ok: 1}
    out: list = [None, None]
    meta: dict = {"consume_calls": 0, "consume_failures": []}

    def consume(index, value):
        return value

    zpar._consume_finished_siblings(
        running, ["t0", "t1"], [1, 1], out, consume, meta, lambda index: None
    )
    assert meta["consume_calls"] == 1
    assert meta["consume_failures"] == []
    assert meta["sibling_worker_failures"] == [
        {
            "index": 0,
            "kind": "worker",
            "error": "ValueError: sibling boom",
        }
    ]
    assert out[1] == "ok"


def test_serial_and_thread_paths_emit_state_events(monkeypatch):
    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: True)
    states = zcs.LiveCellStates()
    states.reset(["a", "b"])

    def cb(event):
        if event.get("cell_id") and event["event"] == "computed":
            states.update(event["cell_id"], zcs.COMPUTED)

    def worker(task):
        return task

    out = zpar.pmap_reserved(
        worker, ["a", "b"], [1, 1], workers=1, cell_ids=["a", "b"],
        consume=lambda i, v: v, state_cb=cb,
    )
    assert out == ["a", "b"]
    assert states.snapshot()["computed"] == 2

    thread_events: list[dict] = []
    out = zpar.pmap(
        worker, ["p", "q"], workers=2, state_cb=thread_events.append,
    )
    assert out == ["p", "q"]
    # ``pmap`` has no cell identity of its own: it reports the TASK INDEX, which
    # the caller maps to its own cell id (never a Stringified index).
    computed = {
        event["index"] for event in thread_events
        if event["event"] == "computed"
    }
    assert computed == {0, 1}
    assert all(
        event["cell_id"] is None for event in thread_events
    )


# ---------------------------------------------------------------------------
# D. external streaming probe
# ------------------------------------------------------------------------


def _ext_fixture(tmp_path):
    from zemosaic.core.canonical_engine import CanonicalStackRequest
    from zemosaic.core.zegrid import external_stream as xs
    from zemosaic.core.zegrid import file_provider as fp

    n, h, w, c = 3, 16, 16, 3
    base = np.random.default_rng(5).normal(
        100.0, 2.0, (n, h, w, c)
    ).astype(np.float32)

    def prepare(i):
        return np.ascontiguousarray(base[int(i)]), np.ones((h, w), dtype=bool)

    def prepare_window(i, y0, y1, x0, x1):
        hh, ww = int(y1 - y0), int(x1 - x0)
        src = base[int(i)]
        return (
            np.ascontiguousarray(src[np.arange(hh) % h][:, np.arange(ww) % w]),
            np.ones((hh, ww), dtype=bool),
        )

    provider = fp.WindowedCanonicalProvider(
        prepare, n, expected_hw=(h, w), prepare_window=prepare_window,
        expected_channels=c,
    )
    request = CanonicalStackRequest(
        images=[None] * n, geometric_support=[None] * n,
        normalization="sky_mean", weighting="noise_variance",
        rejection="winsorized_sigma_clip", combine="mean",
        taper="footprint", taper_px=4.0,
    )
    return xs, provider, request


def test_external_stream_reports_true_tile_and_block_progress(tmp_path):
    xs, provider, request = _ext_fixture(tmp_path)
    fields: list[dict] = []
    xs.run_cell_streaming_external(
        provider, request, tile_size=8, column_batch=8,
        spool_root=str(tmp_path), cell_id="c1", attempt_id="a1",
        keep_spool=True, progress_probe=fields.append,
    )
    phases = {entry.get("phase") for entry in fields}
    assert {"write", "reduce", "finalize"} <= phases
    write_tiles = [
        entry["tile_current"] for entry in fields if entry["phase"] == "write"
    ]
    assert write_tiles and write_tiles == sorted(write_tiles)
    total = max(entry.get("tile_total") or 0 for entry in fields)
    assert total > 0
    assert write_tiles[-1] == total
    blocks = [entry for entry in fields if "block_current" in entry]
    assert blocks, "no block-level progress (long reductions stay silent)"
    assert all(entry["block_total"] >= entry["block_current"] for entry in blocks)
    assert blocks[-1]["block_current"] == blocks[-1]["block_total"]
    # Only plain scalars: no image, mask, dtype or ndarray crosses the seam.
    for entry in fields:
        for value in entry.values():
            assert isinstance(value, (int, float, str, bool, type(None)))
    # The historical phases.log breadcrumb is untouched.
    logs = list(Path(tmp_path).rglob(xs.PHASE_LOG_NAME))
    assert logs
    lines = logs[0].read_text(encoding="utf-8").splitlines()
    assert lines[-1].startswith("ZeGrid: cell c1 phase=finalize END")


def test_progress_probe_cannot_change_the_science(tmp_path):
    xs, provider, request = _ext_fixture(tmp_path)
    clean = xs.run_cell_streaming_external(
        provider, request, tile_size=8, column_batch=8,
        spool_root=str(tmp_path), cell_id="c-clean", attempt_id="a0",
        keep_spool=False,
    )
    noisy = xs.run_cell_streaming_external(
        provider, request, tile_size=8, column_batch=8,
        spool_root=str(tmp_path), cell_id="c-noisy", attempt_id="a1",
        keep_spool=False, progress_probe=lambda fields: fields["nope"],
    )
    for key in (
        "science", "estimator_weight_sum", "valid_mask",
        "surviving_sample_count",
    ):
        assert np.array_equal(
            np.asarray(getattr(clean, key)), np.asarray(getattr(noisy, key))
        ), key


# ---------------------------------------------------------------------------
# B3. rework-1 — R2 periodic heartbeat (journal vs GUI) and R3 merge
# ---------------------------------------------------------------------------


def test_reporter_publishes_a_periodic_heartbeat_with_identical_counters():
    """R2: a blocked worker must still produce a visible periodic heartbeat."""
    clock = [0.0]
    lines: list[str] = []
    gui: list[str] = []
    states = zcs.LiveCellStates(clock=lambda: clock[0])
    states.reset(["a"])
    states.update("a", zcs.RUNNING, observed_at=0.0)
    reporter = zcs.LiveReporter(
        states, None, emit_line=lines.append, gui_emit=gui.append,
        interval_s=30.0, clock=lambda: clock[0],
    )
    assert reporter.publish() is True  # first heartbeat
    clock[0] += 5.0
    assert reporter.publish() is False  # not due and counters unchanged
    clock[0] += 26.0  # 31 s since the last heartbeat
    assert reporter.publish() is True
    heartbeats = [line for line in lines if "cells live" in line]
    assert len(heartbeats) == 2
    # The heartbeat is informative WITHOUT inventing progress: it carries the
    # age of the last real worker observation.
    assert "last_worker_observation_age_s=" in heartbeats[-1]
    assert "running=1" in heartbeats[-1]
    # Detail stays change-only (no repeated per-cell spam).
    assert sum("cell a RUNNING" in line for line in lines) == 1
    assert len(gui) == len(lines)  # GUI delivered when allowed


def test_reporter_journal_only_keeps_beating_and_never_calls_gui():
    clock = [0.0]
    journal: list[str] = []
    gui: list[str] = []
    states = zcs.LiveCellStates(clock=lambda: clock[0])
    states.reset(["a"])
    states.update("a", zcs.RUNNING, observed_at=0.0)
    reporter = zcs.LiveReporter(
        states, None, emit_line=journal.append, gui_emit=gui.append,
        interval_s=10.0, clock=lambda: clock[0],
    )
    for _ in range(5):
        clock[0] += 10.0
        reporter.publish(deliver_gui=False)
    assert gui == []
    assert sum("cells live" in line for line in journal) >= 4


def test_serial_monitor_thread_keeps_a_real_journal_heartbeat(monkeypatch):
    """R2: the SERIAL path (monitor thread) must beat in the JOURNAL.

    The monitor thread publishes with ``gui=False``: the journal line appears
    during a long serial cell, while the GUI callback is never touched from that
    thread (thread affinity).
    """
    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: True)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        live = tmp_path / "live"
        live.mkdir()
        journal: list[str] = []
        gui: list[str] = []
        states = zcs.LiveCellStates()
        states.reset(["c1"])
        states.update("c1", zcs.SUBMITTED)
        reader = zcs.LiveTelemetryReader(live, run_id="run", generation=0)
        reporter = zcs.LiveReporter(
            states, reader, emit_line=journal.append, gui_emit=gui.append,
            interval_s=0.05,
        )
        owner = threading.current_thread()
        release = threading.Event()

        def worker(task):
            writer = zcs.LiveTelemetryWriter(
                zcs.live_cell_path(live, "c1"), run_id="run",
                generation=0, cell_id="c1", min_interval_s=0.0,
            )
            writer.state("RUNNING")
            writer.progress("reduce", tile_current=1, tile_total=9, force=True)
            assert release.wait(30)
            return "ok"

        def heartbeat():
            reporter.sample()
            reporter.publish(
                deliver_gui=(threading.current_thread() is owner)
            )

        timer = threading.Timer(0.6, release.set)
        timer.daemon = True
        timer.start()
        try:
            out = zpar.pmap_reserved(
                worker, ["t"], [1], workers=1, cell_ids=["c1"],
                heartbeat=heartbeat, heartbeat_interval_s=0.05,
            )
        finally:
            release.set()
            timer.cancel()
    assert out == ["ok"]
    heartbeats = [line for line in journal if "cells live" in line]
    assert len(heartbeats) >= 3, journal
    assert gui == []  # the monitor thread never delivered to the GUI
    # The journal heartbeat saw the real worker state and the current tile.
    assert any("running=1" in line for line in heartbeats)
    assert any(
        "cell c1 RUNNING" in line and "tile=1/9" in line for line in journal
    )


def test_block_progress_never_erases_the_current_tile():
    """R3: a partial block sample must not blank the tile, nor leak a stale block."""
    states = zcs.LiveCellStates()
    states.reset(["c"])
    states.update("c", zcs.RUNNING)
    states.note_progress("c", phase="reduce", tile_current=4, tile_total=100)
    states.note_progress("c", phase="reduce", block_current=2, block_total=20)
    row = states.snapshot()["rows"][0]
    assert row["tile"] == [4, 100]  # tile is NOT erased
    assert row["block"] == [2, 20]
    # A new tile clears the previous tile's block counter.
    states.note_progress("c", phase="reduce", tile_current=5, tile_total=100)
    row = states.snapshot()["rows"][0]
    assert row["tile"] == [5, 100] and row["block"] is None
    states.note_progress("c", phase="reduce", block_current=7, block_total=20)
    assert states.snapshot()["rows"][0]["block"] == [7, 20]
    # A new phase also clears the block.
    states.note_progress("c", phase="finalize")
    row = states.snapshot()["rows"][0]
    assert row["phase"] == "finalize" and row["block"] is None
    assert row["tile"] == [5, 100]


def _run_probe_pipeline(tmp_path, *, min_interval_s):
    """REAL probe -> writer -> reader -> states -> formatter pipeline."""
    xs, provider, request = _ext_fixture(tmp_path)
    live = tmp_path / "live"
    live.mkdir()
    writer = zcs.LiveTelemetryWriter(
        zcs.live_cell_path(live, "c1"), run_id="run", generation=0,
        cell_id="c1", min_interval_s=min_interval_s,
    )
    reader = zcs.LiveTelemetryReader(live, run_id="run", generation=0)
    states = zcs.LiveCellStates()
    states.reset(["c1"])
    states.update("c1", zcs.RUNNING)
    rows: list[dict] = []
    formatted: list[str] = []
    calls = [0]

    def probe(fields):
        calls[0] += 1
        writer.progress(
            fields.get("phase") or "phase",
            tile_current=fields.get("tile_current"),
            tile_total=fields.get("tile_total"),
            block_current=fields.get("block_current"),
            block_total=fields.get("block_total"),
            force=bool(fields.get("force")),
        )
        zcs.apply_events(states, reader.poll(["c1"]))
        snapshot = states.snapshot()
        rows.append(dict(snapshot["rows"][0]))
        formatted.append(zcs.live_cell_line(snapshot["rows"][0]))

    xs.run_cell_streaming_external(
        provider, request, tile_size=8, column_batch=8,
        spool_root=str(tmp_path), cell_id="c1", attempt_id="a1",
        keep_spool=False, progress_probe=probe,
    )
    return writer, rows, formatted, calls[0]


def test_probe_to_transport_to_formatter_keeps_tile_and_scopes_block(tmp_path):
    writer, rows, formatted, calls = _run_probe_pipeline(
        tmp_path, min_interval_s=0.0
    )
    with_block = [row for row in rows if row.get("block")]
    assert with_block, rows
    # R3: EVERY block sample carries the tile it belongs to.
    assert all(row.get("tile") for row in with_block)
    assert any("block=" in line and "tile=" in line for line in formatted)
    # A tile change always starts with NO block from the previous tile.
    for previous, current in zip(rows, rows[1:]):
        if previous.get("tile") != current.get("tile") and current.get("tile"):
            assert current.get("block") is None, (previous, current)
    # The last state is finalize with no stale block, tile retained.
    assert rows[-1]["phase"] == "finalize"
    assert rows[-1]["block"] is None
    assert rows[-1]["tile"] is not None
    # Coalescing can never write MORE than the probes it received.
    assert writer.progress_written <= calls


def test_pipeline_rate_limit_bounds_writes_across_many_tiles(tmp_path):
    writer, rows, _formatted, calls = _run_probe_pipeline(
        tmp_path, min_interval_s=3600.0
    )
    # Phase/tile/block transitions plus the forced boundaries only; never one
    # write per block, and never an unbounded stream.
    assert writer.progress_written <= 12, writer.progress_written
    assert calls > 100  # the reduction really probed many blocks
    assert rows[-1]["phase"] == "finalize"
    assert writer.progress_suppressed > 0


def test_rotation_retirement_work_is_linear_and_bookkeeping_bounded(tmp_path):
    """R4 (rework-2): O(1) retirement per rotation, never an O(R^2) rescan.

    The historical implementation rescanned every historic segment path on each
    rotation (198/398/798 rotations -> 19306/78606/317206 unlink attempts) and
    retained every closed segment id. This witness drives the SAME shape with
    real files and measures the actual unlink attempts.
    """
    now = [0.0]
    writer = zcs.LiveTelemetryWriter(
        tmp_path / "c.jsonl", run_id="run", generation=0, cell_id="c",
        min_interval_s=0.0, max_segment_bytes=512, max_segments=3,
        clock=lambda: now[0],
    )
    writer.state("RUNNING")
    checkpoints: dict[int, int] = {}
    target = 800
    tile = 0
    while writer.rotations < target:
        tile += 1
        now[0] += 2.1
        writer.progress(
            "reduce", tile_current=tile, tile_total=10 ** 9, force=True
        )
        if writer.rotations in (200, 400, 800) and writer.rotations not in checkpoints:
            checkpoints[writer.rotations] = writer.unlink_attempts
    assert writer.rotations == target
    assert set(checkpoints) == {200, 400, 800}
    # O(statements) per rotation: at most one retired segment + the two bounded
    # retries (none here, since every unlink succeeds).
    assert checkpoints[800] <= 800 + 8, checkpoints
    assert checkpoints[800] - checkpoints[400] <= 408, checkpoints
    assert checkpoints[400] - checkpoints[200] <= 208, checkpoints
    # Exact linear shape (no quadratic tail).
    assert writer.unlink_attempts <= writer.rotations
    assert writer.unlink_attempts >= writer.rotations - writer.max_segments
    # No bookkeeping growth: nothing pending, no historic id set.
    assert writer._retire_pending == {}
    assert not hasattr(writer, "_closed_segments")
    assert writer.segments_retired >= writer.rotations - writer.max_segments
    # Only the ring is live on disk.
    assert len(sorted(tmp_path.glob("c.jsonl*"))) == writer.max_segments


def test_retirement_bookkeeping_stays_bounded_under_permanent_failure(
    tmp_path, monkeypatch
):
    """A permanently un-unlinkable filesystem must not grow work or memory."""

    def _always_busy(self, *args, **kwargs):
        raise PermissionError("simulated permanent busy")

    monkeypatch.setattr(zcs.Path, "unlink", _always_busy)
    now = [0.0]
    writer = zcs.LiveTelemetryWriter(
        tmp_path / "c.jsonl", run_id="run", generation=0, cell_id="c",
        min_interval_s=0.0, max_segment_bytes=512, max_segments=3,
        clock=lambda: now[0],
    )
    writer.state("RUNNING")
    for tile in range(1, 1201):
        now[0] += 2.1
        writer.progress(
            "reduce", tile_current=tile, tile_total=10 ** 9, force=True
        )
    assert writer.rotations > 100
    # Bounded memory: the pending map never exceeds its hard cap.
    assert len(writer._retire_pending) <= zcs.MAX_PENDING_RETIREMENTS
    # Bounded work: at most one new target + RETIRE_RETRIES_PER_ROTATION retries.
    assert writer.unlink_attempts <= writer.rotations * (
        1 + zcs.RETIRE_RETRIES_PER_ROTATION
    )
    assert writer.retire_failures > 0
    assert writer.retire_given_up > 0
    # The TELEMETRY itself is still alive (never a finite quota).
    assert writer.enabled is True
    assert writer.progress_written > 0


def test_transient_unlink_failure_is_recovered_with_bounded_attempts(
    tmp_path, monkeypatch
):
    real_unlink = zcs.Path.unlink
    transient = {"done": False}

    def _flaky(self, *args, **kwargs):
        # Fail exactly ONCE, on the base segment (the first retirement target).
        if self.name.endswith("c.jsonl") and not transient["done"]:
            transient["done"] = True
            raise PermissionError("simulated transient busy")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(zcs.Path, "unlink", _flaky)
    now = [0.0]
    writer = zcs.LiveTelemetryWriter(
        tmp_path / "c.jsonl", run_id="run", generation=0, cell_id="c",
        min_interval_s=0.0, max_segment_bytes=512, max_segments=3,
        clock=lambda: now[0],
    )
    writer.state("RUNNING")
    for tile in range(1, 401):
        now[0] += 2.1
        writer.progress(
            "reduce", tile_current=tile, tile_total=10 ** 9, force=True
        )
    assert writer.retire_failures == 1
    assert writer.retire_given_up == 0
    assert writer._retire_pending == {}
    assert not (tmp_path / "c.jsonl").exists()  # recovered by the bounded retry pass
    # Exactly one extra attempt for the transient failure (no retry storm).
    assert writer.unlink_attempts <= writer.rotations + 1


# ---------------------------------------------------------------------------
# E. production seam
# ---------------------------------------------------------------------------


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "in"
    proc = subprocess.run(
        [sys.executable, str(CORPUS_TOOL), str(corpus)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return corpus


def _spawn(tmp_path: Path, corpus: Path, out: Path, **kwargs):
    store = out / "zegrid-durable"
    status = tmp_path / f"status-{out.name}.json"
    proc = multiprocessing.get_context("fork").Process(
        target=GH.run_gated,
        kwargs={
            "corpus": str(corpus),
            "out": str(out),
            "store_dir": str(store),
            "status_path": str(status),
            **kwargs,
        },
        daemon=False,
    )
    proc.start()
    return proc, status, store


def _wait_until(predicate, timeout_s=120.0, interval_s=0.05):
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval_s)
    return predicate()


def _log(out: Path) -> str:
    path = out / "zegrid_run.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _commit_files(store: Path) -> dict:
    out = {}
    for path in sorted((store / "commits").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out[str(payload.get("cell_id"))] = path
    return out


def test_live_status_shows_a_running_phase_while_the_slow_cell_is_gated(tmp_path):
    corpus = _corpus(tmp_path)
    out = tmp_path / "out"
    release = tmp_path / "release"
    # The gate sits AFTER the real prepare (cache + live telemetry) and BEFORE
    # the stacking reduction: the slow cell really IS running while we read.
    proc, status_path, store = _spawn(
        tmp_path, corpus, out,
        consume_gate_cell="r0000c0000", consume_release_path=str(release),
    )
    try:
        # The heartbeat is bounded to ~5 s: wait for the real RUNNING evidence.
        def _running_line():
            text = _log(out)
            for line in text.splitlines():
                if (
                    "cell r0000c0000 RUNNING" in line
                    and "phase=" in line
                ):
                    return line
            return None

        line = _wait_until(_running_line, timeout_s=120.0)
        assert line, _log(out)[-6000:]
        assert "phase=cache" in line or "phase=write" in line or "phase=stack" in line
        # The aggregate line is truthful: fast cells committed, the slow one
        # still running, and nothing is counted twice. The FIRST aggregate line
        # is published at the wave boundary BEFORE dispatch, so the witness waits
        # for the latest line to report the real RUNNING cell.
        def _aggregate():
            lines = [
                candidate
                for candidate in _log(out).splitlines()
                if "ZeGrid: cells live —" in candidate
            ]
            if not lines:
                return None
            match = re.search(
                r"total=(\d+) committed=(\d+) \(resumed=(\d+) new=(\d+)\) "
                r"running=(\d+)",
                lines[-1],
            )
            if not match or int(match.group(5)) < 1:
                return None
            return lines[-1]

        aggregate = _wait_until(_aggregate, timeout_s=120.0)
        assert aggregate, _log(out)[-6000:]
        match = re.search(
            r"total=(\d+) committed=(\d+) \(resumed=(\d+) new=(\d+)\) "
            r"running=(\d+)",
            aggregate,
        )
        assert match, aggregate
        total, committed, resumed, new, running = (
            int(value) for value in match.groups()
        )
        assert total == len(ALL_CELLS)
        assert resumed == 0
        assert new == committed
        assert running >= 1
        assert committed < total
        # R2 (rework-1): the periodic heartbeat re-publishes the SAME counters
        # while nothing changes — a blocked worker can never look like silence.
        prefix = aggregate.split(" last_worker_observation_age_s=")[0]

        def _periodic_repeat():
            return sum(
                1
                for candidate in _log(out).splitlines()
                if candidate.startswith(prefix)
            )

        assert _wait_until(lambda: _periodic_repeat() >= 2, timeout_s=120.0)
        # The slow cell is NOT committed while it is still inside its worker.
        assert "cell COMMITTED id=r0000c0000" not in _log(out)
    finally:
        release.write_text("go", encoding="utf-8")
        proc.join(timeout=600)
        if proc.is_alive():  # pragma: no cover - defensive
            proc.kill()
            proc.join(timeout=60)
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "done"
    assert set(_commit_files(store)) == set(ALL_CELLS)
    log = _log(out)
    for cid in ALL_CELLS:
        assert f"cell COMMITTED id={cid}" in log


def test_resumed_run_publishes_resumed_and_new_commits_separately(tmp_path):
    corpus = _corpus(tmp_path)
    out = tmp_path / "out"
    release = tmp_path / "release"
    first, status1, store = _spawn(
        tmp_path, corpus, out,
        gate_cell="r0000c0000", release_path=str(release),
    )
    durable: dict = {}
    try:
        durable = _wait_until(lambda: _commit_files(store), timeout_s=120.0)
        assert durable and "r0000c0000" not in durable
    finally:
        # Stop the WHOLE tree exactly like the L1 witnesses (no parent-only
        # crash claim is made by this lot).
        os.killpg(first.pid, 9)
        first.join(timeout=120)
    assert not status1.exists()

    resumed_proc, resumed_status, _ = _spawn(
        tmp_path, corpus, out, resume_mode="auto",
    )
    resumed_proc.join(timeout=900)
    assert not resumed_proc.is_alive()
    assert json.loads(resumed_status.read_text(encoding="utf-8"))["status"] == "done"
    log = _log(out)
    matches = [
        re.search(
            r"total=(\d+) committed=(\d+) \(resumed=(\d+) new=(\d+)\)",
            line,
        )
        for line in log.splitlines()
        if "ZeGrid: cells live —" in line
    ]
    matches = [match for match in matches if match]
    assert matches, log[-6000:]
    total = int(matches[-1].group(1))
    committed = int(matches[-1].group(2))
    resumed = int(matches[-1].group(3))
    new = int(matches[-1].group(4))
    assert total == len(ALL_CELLS)
    assert resumed == len(durable)
    assert committed == total
    assert new == total - resumed
