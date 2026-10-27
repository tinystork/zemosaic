"""ZM-ZEGRID-RESILIENT-RUN-RESUME M3 — pipeline recovery integration.

These tests exercise the REAL bounded producer/consumer pipeline with the M3
release adapter. They are deterministic and CPU-only (no GPU, no Windows):

* a producer blocked waiting for a credit is cancelled and its reservations are
  released before any wait;
* a full queue is drained without losing ownership;
* a consumer in ``COMMITTING`` is never killed mid-transaction (release before
  wait finishes the safe commit first);
* a worker-return IPC payload from a cancelled generation is released and its
  late result is refused by the generation fence;
* the combined credit-block + COMMITTING case finishes without deadlock.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import parallel as zpar  # noqa: E402
from zemosaic.core.zegrid import resource_recovery as rec  # noqa: E402
from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402


def _wait_for(predicate, timeout=5.0, interval=0.01):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _empty_cell(cell_id: str) -> dict:
    return {
        "cell_id": str(cell_id),
        "status": "empty",
        "record": {
            "cell_id": str(cell_id),
            "status": "empty",
            "mode": None,
            "cell_gauge_mode": "global",
            "n_contributors": 0,
            "patch_bounds": [0, 0, 10, 8],
            "patch_shape_hw": [8, 10],
        },
    }


class _PipelineHarness:
    """A real pipeline run on its own thread, with deterministic consumers."""

    def __init__(
        self,
        *,
        tasks,
        prefetch_units,
        host_budget,
        host_bytes,
        consume,
        device_budget=1024,
        device_bytes=lambda payload: 0,
    ):
        self.released: list[object] = []
        self.release_errors: list[str] = []
        self.produced: list[object] = []
        self.errors: list[BaseException] = []

        def produce(task):
            self.produced.append(task)
            return {"task": task, "buffer": bytearray(host_bytes)}

        def release(payload):
            self.released.append(payload.get("task"))

        self.pipeline = zpar.BoundedPipeline(
            tasks=list(tasks),
            produce=produce,
            consume=consume,
            release=release,
            prefetch_units=prefetch_units,
            host_budget_bytes=host_budget,
            device_budget_bytes=device_budget,
            host_bytes=lambda payload: len(payload["buffer"]),
            device_bytes=device_bytes,
            disk_budget_bytes=0,
            thread_join_s=5.0,
        )
        self.thread = threading.Thread(
            target=self._run, name="m3-pipeline-run", daemon=True
        )

    def _run(self):
        try:
            self.pipeline.run()
        except BaseException as exc:  # noqa: BLE001 - asserted by the test
            self.errors.append(exc)

    def start(self):
        self.thread.start()

    def join(self, timeout=10.0):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "pipeline thread did not terminate"


# ---------------------------------------------------------------------------
# Producer blocked on a credit
# ---------------------------------------------------------------------------


def test_producer_blocked_on_credit_is_released_before_waiting():
    consumer_state = {"value": "CONSUMING"}
    state_lock = threading.Lock()
    consume_started = threading.Event()

    def consume(payload):
        consume_started.set()
        while not harness.pipeline.cancelled():
            time.sleep(0.005)
        with state_lock:
            consumer_state["value"] = "IDLE"
        return {"ok": payload["task"]}

    harness = _PipelineHarness(
        tasks=[0, 1, 2],
        prefetch_units=0,
        host_budget=128,
        host_bytes=128,
        consume=consume,
    )
    harness.start()
    assert consume_started.wait(3.0)
    # The producer must be waiting for the credit held by the consumer.
    assert _wait_for(
        lambda: harness.pipeline.ledger_snapshot()["credit_wait_started_events"] >= 1,
        timeout=3.0,
    )

    adapter = rec.PipelineReleaseAdapter(
        harness.pipeline,
        consumer_state=lambda: consumer_state["value"],
        quiesce_grace_s=5.0,
    )
    report = adapter()
    harness.join()
    assert harness.errors and isinstance(
        harness.errors[0], zpar.PipelineCancelled
    )
    assert report.stopped_admission is True
    assert report.credits_balanced is True
    assert report.units_released >= 1
    assert report.host_released_bytes > 0
    ledger = harness.pipeline.ledger_snapshot()
    assert ledger["balanced"] is True
    assert ledger["host_used_bytes"] == 0
    # Every produced payload was released exactly once.
    assert sorted(harness.released) == sorted(harness.produced)


# ---------------------------------------------------------------------------
# Full queue
# ---------------------------------------------------------------------------


def test_full_queue_is_drained_without_ownership_leak():
    consume_started = threading.Event()

    def consume(payload):
        consume_started.set()
        while not harness.pipeline.cancelled():
            time.sleep(0.005)
        return {"ok": payload["task"]}

    harness = _PipelineHarness(
        tasks=list(range(8)),
        prefetch_units=2,
        host_budget=4096,
        host_bytes=256,
        consume=consume,
    )
    harness.start()
    assert consume_started.wait(3.0)
    assert _wait_for(
        lambda: harness.pipeline.telemetry()["producer_backpressure_events"] >= 1,
        timeout=3.0,
    )
    assert harness.pipeline.telemetry()["queue_depth_max"] >= 1

    report = rec.PipelineReleaseAdapter(
        harness.pipeline, consumer_state=lambda: "CONSUMING"
    )()
    harness.join()
    assert report.credits_balanced is True
    assert report.units_released >= 1
    assert harness.pipeline.ledger_snapshot()["balanced"] is True
    assert sorted(harness.released) == sorted(harness.produced)
    assert len(harness.released) == len(set(harness.released))


# ---------------------------------------------------------------------------
# Consumer in COMMITTING
# ---------------------------------------------------------------------------


def test_committing_consumer_is_not_killed_mid_transaction():
    commit_done = threading.Event()
    commit_started = threading.Event()
    state = {"value": "CONSUMING"}
    committed = {"value": False}
    cancelled_at_commit = {"value": None}

    def consume(payload):
        state["value"] = "COMMITTING"
        commit_started.set()
        # A real commit must be allowed to finish (bounded by the adapter).
        while not commit_done.is_set():
            time.sleep(0.005)
        committed["value"] = True
        state["value"] = "IDLE"
        return {"ok": payload["task"]}

    harness = _PipelineHarness(
        tasks=[0, 1, 2],
        prefetch_units=1,
        host_budget=1024,
        host_bytes=128,
        consume=consume,
    )
    harness.start()
    assert commit_started.wait(3.0)
    assert state["value"] == "COMMITTING"

    report_holder: dict = {}

    def run_adapter():
        report_holder["report"] = rec.PipelineReleaseAdapter(
            harness.pipeline,
            consumer_state=lambda: state["value"],
            commit_done_event=commit_done,
            commit_grace_s=5.0,
        )()

    adapter_thread = threading.Thread(target=run_adapter, name="m3-adapter")
    adapter_thread.start()
    time.sleep(0.2)
    # The adapter is waiting for the commit; it must NOT have cancelled yet.
    cancelled_at_commit["value"] = harness.pipeline.cancelled()
    assert cancelled_at_commit["value"] is False
    commit_done.set()
    adapter_thread.join(timeout=5.0)
    assert not adapter_thread.is_alive()
    report = report_holder["report"]
    assert committed["value"] is True
    assert report.consumer_state_before == "COMMITTING"
    assert report.consumer_state_after == "IDLE"
    harness.join()
    assert report.credits_balanced is True
    assert harness.pipeline.ledger_snapshot()["balanced"] is True


# ---------------------------------------------------------------------------
# Worker return IPC + cancelled generation
# ---------------------------------------------------------------------------


def test_worker_ipc_return_from_cancelled_generation_is_released_and_refused():
    ipc: "queue.Queue[dict]" = queue.Queue()
    release_count = {"value": 0}
    payload = {
        "cell_id": "r0000c0000",
        "generation": 0,
        "buffer": bytearray(4096),
        "released": False,
    }
    ipc.put(payload)

    def drain_ipc():
        units = 0
        freed = 0
        while True:
            try:
                item = ipc.get_nowait()
            except queue.Empty:
                break
            if item.get("released"):
                continue
            item["released"] = True
            release_count["value"] += 1
            units += 1
            freed += len(item["buffer"])
        return {
            "cancelled_units": units,
            "host_released_bytes": freed,
            "credits_balanced": True,
        }

    report = rec.PipelineReleaseAdapter(
        None, external_drains=[drain_ipc]
    )()
    assert report.units_released == 1
    assert report.host_released_bytes == 4096
    assert release_count["value"] == 1
    assert ipc.empty()

    registry = zrr.RunResultRegistry(
        run_id="run-ipc",
        expected_cell_ids=["r0000c0000"],
        gauge_mode="global",
    )
    fence = rec.GenerationFence(run_id=registry.run_id, registry=registry)
    old = rec.TaskGeneration(
        run_id=registry.run_id,
        cell_id="r0000c0000",
        attempt_id="worker-ipc",
        plan_generation=0,
    )
    fence.advance(reason="cancel speculative generation")
    outcome = fence.accept(old, _empty_cell("r0000c0000"), backend="cpu")
    assert outcome.accepted is False
    assert outcome.reason == "stale_generation"
    assert registry.counts()["accepted"] == 0
    assert release_count["value"] == 1  # released exactly once, never re-served


# ---------------------------------------------------------------------------
# Combined credit block + COMMITTING: no deadlock
# ---------------------------------------------------------------------------


def test_credit_block_with_committing_consumer_does_not_deadlock():
    commit_done = threading.Event()
    commit_started = threading.Event()
    state = {"value": "COMMITTING"}

    def consume(payload):
        commit_started.set()
        while not commit_done.is_set():
            time.sleep(0.005)
        state["value"] = "IDLE"
        return {"ok": payload["task"]}

    harness = _PipelineHarness(
        tasks=[0, 1, 2],
        prefetch_units=0,
        host_budget=128,
        host_bytes=128,
        consume=consume,
    )
    harness.start()
    assert commit_started.wait(3.0)
    assert _wait_for(
        lambda: harness.pipeline.ledger_snapshot()["credit_wait_started_events"] >= 1,
        timeout=3.0,
    )
    adapter = rec.PipelineReleaseAdapter(
        harness.pipeline,
        consumer_state=lambda: state["value"],
        commit_done_event=commit_done,
        commit_grace_s=5.0,
        quiesce_grace_s=5.0,
    )
    timer = threading.Timer(0.25, commit_done.set)
    timer.start()
    started = time.monotonic()
    try:
        report = adapter()
    finally:
        timer.cancel()
        commit_done.set()
    elapsed = time.monotonic() - started
    harness.join()
    assert elapsed < 8.0
    assert report.credits_balanced is True
    assert harness.pipeline.ledger_snapshot()["balanced"] is True
    # Either the safe commit let the run finish normally, or the one-way
    # cancellation ended it; no accounting/deadlock error is acceptable.
    assert all(
        isinstance(exc, zpar.PipelineCancelled) for exc in harness.errors
    )
