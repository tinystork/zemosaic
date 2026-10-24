"""ZM-ZEGRID-R55-E5 — bounded pipeline liveness on Windows GPU.

Real-witness failure (RTX 3070, commit 15ada5c): the GPU was requested,
available, admitted and executing, yet ``BoundedPipeline.run`` cancelled the
pipeline after the sequential 60 s producer join + 60 s consumer join + 10 s
grace while the FIRST healthy GPU cell was still inside ``consume``. Telemetry
proved the false positive: ``transfers=0``, ``units_consumed=0``,
``device_leases=1`` (a unit was actively in flight), ``producer_alive=False``,
``cancel_reason=thread_join_timeout``.

The fix is a lifecycle change, not timeout inflation: the GPU consumer is the
CALLER thread and the only background thread is the bounded CPU producer. A
healthy unit can take as long as it needs; the finite join bound is applied
only after a terminal state (sentinel, error or cancellation), where a
surviving producer is a genuine stuck thread.

Acceptance coverage here:

* a healthy consume longer than the configured join interval completes;
* ``consume`` provably runs on the caller thread; only ``produce`` is
  asynchronous; ``queue_max == prefetch_units + 1``;
* producer/consumer exceptions keep their primary type and release each staged
  unit exactly once;
* caller cancellation (including ``KeyboardInterrupt``) stays effective and the
  producer join stays finite;
* the optional stall watchdog fails closed instead of hanging forever;
* the ledger/queue/threads are reconciled on every terminal path;
* a read-only parser explains the supplied Windows manifest evidence and is
  skipped unless ``ZM_R55_WINDOWS_MANIFEST`` points at it.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from zemosaic.core.zegrid import parallel as zpar


class _Boom(RuntimeError):
    pass


def _pipe(**kw):
    released: list[int] = []
    params = dict(
        tasks=[0, 1, 2],
        produce=lambda task: {"value": int(task)},
        consume=lambda payload: payload["value"],
        release=lambda payload: released.append(payload["value"]),
        prefetch_units=1,
        host_budget_bytes=64,
        device_budget_bytes=64,
        host_bytes=lambda payload: 8,
        device_bytes=lambda payload: 8,
    )
    params.update(kw)
    return zpar.BoundedPipeline(**params), released


# ---------------------------------------------------------------------------
# 1. The regression: long healthy GPU work is not a deadlock
# ---------------------------------------------------------------------------


def test_healthy_consume_longer_than_join_interval_completes():
    """A unit that outlives ``thread_join_s`` must succeed, not cancel.

    Under the pre-E5 sequential-join lifecycle this is a hard failure: the
    consumer join expires while a healthy unit is still running and the
    pipeline cancels with ``thread_join_timeout``.
    """
    def slow_consume(payload):
        time.sleep(0.35)
        return payload["value"] * 10

    started = time.monotonic()
    pipe, released = _pipe(
        tasks=list(range(3)),
        consume=slow_consume,
        thread_join_s=0.2,
    )
    assert pipe.run() == [0, 10, 20]
    elapsed = time.monotonic() - started
    # Three healthy units (~1.05 s) far exceed the 0.2 s join bound.
    assert elapsed > 0.7
    tel = pipe.telemetry()
    assert tel["cancelled"] is False
    assert tel["cancel_reason"] is None
    assert tel["units_consumed"] == 3
    assert tel["transfers"] == 3
    assert tel["ledger"]["balanced"] is True
    assert tel["threads_joined"] is True
    assert sorted(released) == [0, 1, 2]


def test_single_unit_longer_than_join_interval_is_not_cancelled():
    def slow_consume(payload):
        time.sleep(0.4)
        return payload["value"]

    pipe, _ = _pipe(tasks=[7], consume=slow_consume, thread_join_s=0.1)
    assert pipe.run() == [7]
    tel = pipe.telemetry()
    assert tel["cancelled"] is False
    assert tel["units_consumed"] == 1


# ---------------------------------------------------------------------------
# 2. Ownership: caller thread consumes, only the producer is asynchronous
# ---------------------------------------------------------------------------


def test_consume_runs_on_the_caller_thread_and_producer_is_background():
    caller = threading.current_thread()
    seen = {}

    def produce(task):
        seen["produce_thread"] = threading.current_thread()
        return {"value": int(task)}

    def consume(payload):
        seen["consume_thread"] = threading.current_thread()
        seen["seen_producer_alive"] = any(
            t.name == "r55e2-pipeline-producer" and t.is_alive()
            for t in threading.enumerate()
        )
        return payload["value"]

    pipe = zpar.BoundedPipeline(
        tasks=[0, 1, 2, 3],
        produce=produce,
        consume=consume,
        release=lambda payload: None,
        prefetch_units=3,
        host_budget_bytes=64,
        device_budget_bytes=64,
        host_bytes=lambda payload: 8,
        device_bytes=lambda payload: 8,
    )
    assert pipe.queue_max == 4
    assert pipe.run() == [0, 1, 2, 3]

    assert seen["consume_thread"] is caller
    assert seen["produce_thread"] is not caller
    assert seen["produce_thread"].name == "r55e2-pipeline-producer"
    assert seen["seen_producer_alive"] is True

    tel = pipe.telemetry()
    assert tel["lifecycle"] == "caller_owned_consumer"
    assert tel["consumer_owner"] == "caller_thread"
    assert tel["owner_thread_name"] == caller.name
    assert tel["owner_thread_ident"] == threading.get_ident()
    assert tel["producer_thread_name"] == "r55e2-pipeline-producer"
    assert tel["producer_alive"] is False
    assert tel["queue_max"] == 4
    assert tel["queue_depth_final"] == 0
    assert tel["active_unit_index"] is None


def test_prefetch_bound_is_still_exact():
    for prefetch in (0, 1, 2, 4):
        pipe, _ = _pipe(prefetch_units=prefetch, tasks=list(range(6)))
        assert pipe.queue_max == prefetch + 1
        assert pipe.run() == list(range(6))
        assert pipe.telemetry()["queue_depth_max"] <= prefetch + 1


# ---------------------------------------------------------------------------
# 3. Exceptions: primary type preserved, every unit released exactly once
# ---------------------------------------------------------------------------


def test_consumer_error_after_long_healthy_prefix_keeps_type_and_ledger():
    released: list[int] = []

    def consume(payload):
        if payload["value"] == 2:
            raise _Boom("gpu exploded")
        time.sleep(0.05)
        return payload["value"]

    pipe = zpar.BoundedPipeline(
        tasks=list(range(6)),
        produce=lambda task: {"value": int(task)},
        consume=consume,
        release=lambda payload: released.append(payload["value"]),
        prefetch_units=2,
        host_budget_bytes=64,
        device_budget_bytes=64,
        host_bytes=lambda payload: 8,
        device_bytes=lambda payload: 8,
        thread_join_s=0.2,
    )
    with pytest.raises(_Boom, match="gpu exploded"):
        pipe.run()
    tel = pipe.telemetry()
    assert tel["consumer_error"].startswith("_Boom:")
    assert tel["units_consumed"] == 2
    assert tel["ledger"]["balanced"] is True
    assert tel["ledger"]["host_leases"] == 0
    assert tel["ledger"]["device_leases"] == 0
    assert tel["threads_joined"] is True
    assert tel["queue_depth_final"] == 0
    assert len(released) == len(set(released))
    assert 2 in released  # the failing unit is released exactly once


def test_producer_error_keeps_primary_classification_and_releases_staged():
    released: list[int] = []

    def produce(task):
        if task == 2:
            raise ValueError("cache build failed")
        return {"value": int(task)}

    pipe = zpar.BoundedPipeline(
        tasks=list(range(6)),
        produce=produce,
        consume=lambda payload: payload["value"],
        release=lambda payload: released.append(payload["value"]),
        prefetch_units=1,
        host_budget_bytes=64,
        device_budget_bytes=64,
        host_bytes=lambda payload: 8,
        device_bytes=lambda payload: 8,
    )
    with pytest.raises(zpar.PipelineError, match="producer failed") as exc:
        pipe.run()
    assert isinstance(exc.value.__cause__, ValueError)
    tel = pipe.telemetry()
    assert tel["producer_error"].startswith("ValueError:")
    assert tel["ledger"]["balanced"] is True
    assert tel["threads_joined"] is True
    assert tel["queue_depth_final"] == 0
    assert len(released) == len(set(released))


def test_accounting_error_is_not_wrapped_as_a_producer_failure():
    pipe = zpar.BoundedPipeline(
        tasks=[0],
        produce=lambda task: {"value": task},
        consume=lambda payload: payload["value"],
        release=lambda payload: None,
        prefetch_units=0,
        host_budget_bytes=4,
        device_budget_bytes=4,
        host_bytes=lambda payload: 8,
        device_bytes=lambda payload: 8,
    )
    with pytest.raises(zpar.PipelineAccountingError):
        pipe.run()
    tel = pipe.telemetry()
    assert tel["ledger"]["balanced"] is True
    assert tel["threads_joined"] is True


# ---------------------------------------------------------------------------
# 4. Cancellation stays effective, joins stay finite
# ---------------------------------------------------------------------------


def test_external_cancel_interrupts_a_slow_pipeline_and_cleans_up():
    released: list[int] = []
    entered = threading.Event()

    def consume(payload):
        entered.set()
        time.sleep(0.2)
        return payload["value"]

    pipe = zpar.BoundedPipeline(
        tasks=list(range(20)),
        produce=lambda task: {"value": int(task)},
        consume=consume,
        release=lambda payload: released.append(payload["value"]),
        prefetch_units=2,
        host_budget_bytes=64,
        device_budget_bytes=64,
        host_bytes=lambda payload: 8,
        device_bytes=lambda payload: 8,
        thread_join_s=1.0,
    )

    def canceller():
        assert entered.wait(5.0)
        pipe.cancel("user_cancelled")

    thread = threading.Thread(target=canceller, daemon=True)
    thread.start()
    with pytest.raises(zpar.PipelineCancelled, match="user_cancelled"):
        pipe.run()
    thread.join(5.0)
    tel = pipe.telemetry()
    assert tel["cancelled"] is True
    assert tel["cancel_reason"] == "user_cancelled"
    assert tel["ledger"]["balanced"] is True
    assert tel["threads_joined"] is True
    assert tel["queue_depth_final"] == 0
    assert tel["units_consumed"] < 20
    assert len(released) == len(set(released))  # exactly once per staged unit


def test_keyboard_interrupt_is_propagated_and_cleans_up():
    released: list[int] = []

    def consume(payload):
        if payload["value"] == 1:
            raise KeyboardInterrupt()
        return payload["value"]

    pipe = zpar.BoundedPipeline(
        tasks=list(range(8)),
        produce=lambda task: {"value": int(task)},
        consume=consume,
        release=lambda payload: released.append(payload["value"]),
        prefetch_units=2,
        host_budget_bytes=64,
        device_budget_bytes=64,
        host_bytes=lambda payload: 8,
        device_bytes=lambda payload: 8,
    )
    with pytest.raises(KeyboardInterrupt):
        pipe.run()
    tel = pipe.telemetry()
    # The signal/interrupt type is preserved, and the observable state is clean.
    assert tel["interrupt"].startswith("KeyboardInterrupt")
    assert tel["cancelled"] is True
    assert tel["ledger"]["balanced"] is True
    assert tel["threads_joined"] is True
    assert len(released) == len(set(released))
    assert 1 in released


def test_stuck_producer_fails_closed_with_a_finite_join():
    """The optional stall watchdog is the only no-progress budget."""
    started = threading.Event()
    unblock = threading.Event()

    def produce(task):
        started.set()
        # A genuinely stuck producer: blocks until the test releases it.
        unblock.wait(30.0)
        return {"value": int(task)}

    pipe = zpar.BoundedPipeline(
        tasks=[0],
        produce=produce,
        consume=lambda payload: payload["value"],
        release=lambda payload: None,
        prefetch_units=0,
        host_budget_bytes=64,
        device_budget_bytes=64,
        host_bytes=lambda payload: 8,
        device_bytes=lambda payload: 8,
        thread_join_s=0.2,
        stall_timeout_s=0.2,
    )
    t0 = time.monotonic()
    try:
        with pytest.raises(zpar.PipelineError, match="no producer progress"):
            pipe.run()
        elapsed = time.monotonic() - t0
        # Telemetry at the terminal moment, BEFORE the test releases the stuck
        # producer: liveness must be reported truthfully, not hidden.
        tel = pipe.telemetry()
    finally:
        unblock.set()  # never leave a daemon producer behind for other tests
        if pipe._producer_thread is not None:
            pipe._producer_thread.join(5.0)
    # Bounded: 0.2 s stall + 0.2 s join + 1.0 s floor join, nowhere near 30 s.
    assert elapsed < 5.0
    assert started.is_set()
    assert tel["stalled"] is True
    assert tel["cancel_reason"] == "stall_timeout"
    assert tel["producer_alive"] is True  # reported, never hidden
    assert tel["threads_joined"] is False
    assert tel["cleanup_failures"]
    assert tel["ledger"]["balanced"] is True  # nothing was staged


def test_stall_timeout_validation_is_closed():
    for bad in (0, -1.0, True, "10", 601.0):
        with pytest.raises(zpar.PipelineError, match="stall_timeout_s"):
            _pipe(stall_timeout_s=bad)


def test_stall_watchdog_is_disabled_by_default():
    pipe, _ = _pipe()
    assert pipe.telemetry()["stall_timeout_s"] is None


# ---------------------------------------------------------------------------
# 5. Windows evidence parser (read-only; hermetic by default)
# ---------------------------------------------------------------------------


def diagnose_manifest_fallback(manifest: dict) -> dict:
    """Explain a ZeGrid GPU fallback from the published manifest fields."""
    block = (manifest.get("r55_scheduler") or {})
    fallback = (block.get("gpu_fallback") or {})
    telemetry = (fallback.get("telemetry") or {})
    pipe = (telemetry.get("pipeline") or {})
    ledger = (pipe.get("ledger") or {})
    detail = str(fallback.get("detail") or "")
    reason = str(fallback.get("reason_code") or "")

    join_timeout = (
        pipe.get("cancel_reason") == "thread_join_timeout"
        and "did not terminate" in detail
    )
    unit_in_flight = bool(
        int(pipe.get("units_consumed") or 0) == 0
        and int(pipe.get("transfers") or 0) == 0
        and int(ledger.get("device_leases") or 0) > 0
    )
    return {
        "reason_code": reason,
        "cancel_reason": pipe.get("cancel_reason"),
        "class": (
            "fixed_join_budget_false_timeout"
            if join_timeout and unit_in_flight
            else "other"
        ),
        "gpu_work_was_in_flight": unit_in_flight,
        "units_total": int(pipe.get("units_total") or 0),
        "units_produced": int(pipe.get("units_produced") or 0),
        "units_consumed": int(pipe.get("units_consumed") or 0),
        "transfers": int(pipe.get("transfers") or 0),
        "device_leases": int(ledger.get("device_leases") or 0),
        "producer_alive": bool(pipe.get("producer_alive", False)),
        "consumer_alive_legacy": bool(pipe.get("threads_joined") is False),
    }


def test_evidence_parser_classifies_the_observed_windows_failure():
    manifest = {
        "r55_scheduler": {
            "gpu_fallback": {
                "stage": "run",
                "reason_code": "R55_GPU_PIPELINE_FAILED",
                "detail": (
                    "PipelineError: bounded pipeline thread did not terminate "
                    "(producer_alive=False, consumer_alive=True)"
                ),
                "telemetry": {
                    "pipeline": {
                        "units_total": 6,
                        "units_produced": 4,
                        "units_consumed": 0,
                        "transfers": 0,
                        "queue_max": 3,
                        "queue_depth_last": 3,
                        "cancel_reason": "thread_join_timeout",
                        "producer_alive": False,
                        "threads_joined": False,
                        "ledger": {
                            "device_leases": 1,
                            "balanced": False,
                        },
                    }
                },
            }
        }
    }
    diagnosis = diagnose_manifest_fallback(manifest)
    assert diagnosis["class"] == "fixed_join_budget_false_timeout"
    assert diagnosis["gpu_work_was_in_flight"] is True
    assert diagnosis["transfers"] == 0
    assert diagnosis["units_consumed"] == 0
    assert diagnosis["device_leases"] == 1


def test_evidence_parser_reads_the_real_windows_manifest_when_supplied():
    path = os.environ.get("ZM_R55_WINDOWS_MANIFEST")
    if not path:
        pytest.skip("ZM_R55_WINDOWS_MANIFEST not set (hermetic default)")
    manifest_path = Path(path)
    if not manifest_path.is_file():
        pytest.skip(f"evidence manifest not found: {manifest_path}")
    # Read-only: the evidence directory is never modified.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnosis = diagnose_manifest_fallback(manifest)
    assert diagnosis["class"] == "fixed_join_budget_false_timeout"
    assert diagnosis["gpu_work_was_in_flight"] is True
