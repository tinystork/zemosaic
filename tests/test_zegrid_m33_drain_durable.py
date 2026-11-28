"""ZM-M33-DRAIN-DURABLE — first-lot failure-lifecycle acceptance witnesses.

Contract under test (defined before the implementation):

* on the FIRST worker failure ``pmap``/``pmap_reserved`` cancel the truly
  PENDING tasks (no resubmission, no serial replay) and surface the fatal state
  BEFORE any drain;
* tasks the executor had ALREADY STARTED are drained, and every successful one
  is handed to the parent ``consume`` (durable commit) EXACTLY ONCE while the
  original error is being waited for;
* the ORIGINAL exception is always the one that escapes (no false PASS), and
  secondary worker/consumer failures are explicit in ``meta``;
* a consumer (durable commit) failure is never swallowed nor converted into
  success: after it, the drain stops consuming further results and records them
  as uncommitted (fail-closed store policy);
* heartbeat/progress keep advancing during the drain; receipts are truthful.

These are tiny synthetic workers: no M33 dataset, no big benchmark, no GPU.
Determinism comes from events/barriers/file gates with finite timeouts, never
from a sleep-based race.
"""

from __future__ import annotations

import functools
import json
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
TOOLS_M4 = REPO_ROOT / "tools" / "zegrid_resume_m4"
if str(TOOLS_M4) not in sys.path:
    sys.path.insert(0, str(TOOLS_M4))

import concurrent.futures as cf  # noqa: E402

from zemosaic.core.zegrid import parallel as zpar  # noqa: E402
from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

import m4_resume_harness as M4  # noqa: E402

MIB = 2 ** 20
RUN_ID = "m33-drain-durable"


def _force_thread_executor(monkeypatch):
    """Force pmap's THREAD path so a gate can be an in-process Event."""
    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: True)


def _force_spawn_executor(monkeypatch):
    """Force a real multiprocessing SPAWN pool (the Windows-like start method)."""
    ctx = multiprocessing.get_context("spawn")
    monkeypatch.setattr(
        cf,
        "ProcessPoolExecutor",
        functools.partial(cf.ProcessPoolExecutor, mp_context=ctx),
    )
    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: False)


def _real_store(tmp_path: Path):
    rows = M4.ensure_inputs(tmp_path / "in")
    identity = M4.make_identity(RUN_ID, rows)
    return zrs.RunCellStore.open(
        tmp_path / f"store-{RUN_ID}",
        identity=identity,
        budget=zrs.DiskBudget(durable_cell_bytes=64 * MIB),
        session_id=RUN_ID,
    )


def _registry():
    return zrr.RunResultRegistry(
        run_id=RUN_ID,
        expected_cell_ids=list(M4.CELL_IDS),
        gauge_mode="per_cell",
    )


def _durable_consume(store, registry, receipts, threads):
    """A REAL per-cell durable receipt (accept -> commit -> require_commit)."""

    def consume(index, result):
        threads.append(threading.get_ident())
        cid = str(result["cell_id"])
        attempt = f"drain:{cid}"
        registry.accept(
            result, backend="cpu", plan_generation=0, attempt_id=attempt
        )
        store.commit_and_release(
            registry,
            cid,
            attempt_id=attempt,
            backend="cpu",
            plan_generation=0,
        )
        store.require_commit(cid, attempt)
        receipts.append(cid)
        return {"receipt": True, "cell_id": cid, "committed": True}

    return consume


# --- spawn-safe helpers (module level: picklable by reference) -------------


def _wait_path(path, timeout_s):
    deadline = time.time() + float(timeout_s)
    target = Path(str(path))
    while not target.exists():
        if time.time() > deadline:
            raise RuntimeError(f"drain gate timed out waiting for {target}")
        time.sleep(0.01)


def _append_line(path, payload):
    with open(str(path), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _spawn_drain_worker(task):
    """Tiny synthetic worker driven by picklable task fields (spawn-safe)."""
    index = int(task["index"])
    log = task.get("executed_log")
    if log:
        _append_line(log, {"index": index, "pid": os.getpid()})
    if task.get("die"):
        os._exit(1)
    marker = task.get("started_marker")
    if marker:
        Path(str(marker)).write_text("started", encoding="utf-8")
    wait_started = task.get("wait_for_started")
    if wait_started:
        _wait_path(wait_started, task.get("timeout_s", 60.0))
    wait_for = task.get("wait_for")
    if wait_for:
        _wait_path(wait_for, task.get("timeout_s", 60.0))
    if task.get("fail"):
        raise MemoryError(f"M33 synthetic worker failure at index {index}")
    return {"cell_id": str(task["cell_id"]), "index": index}


def _read_jsonl(path):
    if not Path(path).exists():
        return []
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# 1. pmap THREAD: failing worker + sibling that completes only AFTER on_fatal
# ---------------------------------------------------------------------------


def test_pmap_thread_failure_drains_and_durably_commits_started_sibling(
    monkeypatch, tmp_path
):
    _force_thread_executor(monkeypatch)
    store = _real_store(tmp_path)
    registry = _registry()
    receipts: list[str] = []
    threads: list[int] = []
    fatal_events: list[dict] = []
    meta: dict = {}
    barrier = threading.Barrier(2)
    release_sibling = threading.Event()

    def worker(task):
        barrier.wait(timeout=60)
        if task == 0:
            raise MemoryError("M33 synthetic worker failure")
        assert release_sibling.wait(60), "sibling was never released"
        return M4.compute_cell(1)

    def on_fatal(event):
        fatal_events.append(dict(event))
        # The sibling only completes AFTER the fatal state was surfaced.
        release_sibling.set()

    try:
        with pytest.raises(zpar.ParallelTaskError) as excinfo:
            zpar.pmap(
                worker,
                [0, 1],
                workers=2,
                consume=_durable_consume(store, registry, receipts, threads),
                meta=meta,
                on_fatal=on_fatal,
                serial_fallback=False,
            )
        err = excinfo.value
        assert err.index == 0
        assert isinstance(err.original, MemoryError)
        assert fatal_events and fatal_events[0]["index"] == 0
        # Drained, consumed EXACTLY once, and durably committed before the raise.
        assert receipts == [M4.CELL_IDS[1]], receipts
        assert meta["consume_calls"] == 1
        assert meta["drain"]["started"] is True
        assert meta["drain"]["consumed"] == [1]
        assert meta["uncommitted_outputs"] == []
        # The consumer ran in the CALLER thread, never in a worker.
        assert threads and set(threads) == {threading.get_ident()}
        reader = zrs.RunCellStore.open(store.root, read_only=True, create=False)
        try:
            record = reader.commit_record(M4.CELL_IDS[1])
            assert record.cell_id == M4.CELL_IDS[1]
            with pytest.raises(zrs.StoreNotFoundError):
                reader.commit_record(M4.CELL_IDS[0])
        finally:
            reader.close()
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 2. pmap SPAWN: same guarantee across a real Windows-like process pool
# ---------------------------------------------------------------------------


def test_pmap_spawn_failure_drains_and_persists_sibling_receipt(
    monkeypatch, tmp_path
):
    _force_spawn_executor(monkeypatch)
    release = tmp_path / "release-sibling"
    receipt_path = tmp_path / "receipts.jsonl"
    executed = tmp_path / "executed.jsonl"
    fatal_events: list[dict] = []
    meta: dict = {}

    def on_fatal(event):
        fatal_events.append(dict(event))
        release.write_text("go", encoding="utf-8")

    def consume(index, result):
        _append_line(
            receipt_path,
            {
                "index": int(index),
                "cell_id": result["cell_id"],
                "thread": threading.get_ident(),
            },
        )
        return {"receipt": True, "cell_id": result["cell_id"]}

    tasks = [
        {
            "index": 0,
            "cell_id": "c0",
            "fail": True,
            "executed_log": str(executed),
            "wait_for_started": str(tmp_path / "sibling-started"),
        },
        {
            "index": 1,
            "cell_id": "c1",
            "executed_log": str(executed),
            "started_marker": str(tmp_path / "sibling-started"),
            "wait_for": str(release),
        },
    ]
    with pytest.raises(zpar.ParallelTaskError) as excinfo:
        zpar.pmap(
            _spawn_drain_worker,
            tasks,
            workers=2,
            consume=consume,
            meta=meta,
            on_fatal=on_fatal,
            serial_fallback=False,
        )
    assert isinstance(excinfo.value.original, MemoryError)
    assert fatal_events and fatal_events[0]["index"] == 0
    lines = _read_jsonl(receipt_path)
    assert len(lines) == 1, lines
    assert lines[0]["index"] == 1 and lines[0]["cell_id"] == "c1"
    assert lines[0]["thread"] == threading.get_ident()
    assert meta["drain"]["consumed"] == [1]
    assert meta["uncommitted_outputs"] == []
    assert sorted(entry["index"] for entry in _read_jsonl(executed)) == [0, 1]


# ---------------------------------------------------------------------------
# 3. pmap_reserved: pending tasks NEVER run, started sibling is consumed
# ---------------------------------------------------------------------------


def test_pmap_reserved_thread_never_dispatches_pending_and_consumes_started(
    monkeypatch,
):
    _force_thread_executor(monkeypatch)
    barrier = threading.Barrier(2)
    release = threading.Event()
    executed: list[int] = []
    consumed: list[int] = []
    fatal_events: list[dict] = []
    meta: dict = {}

    def worker(task):
        executed.append(int(task))
        if task == 0:
            barrier.wait(timeout=60)
            raise MemoryError("M33 synthetic worker failure")
        barrier.wait(timeout=60)
        assert release.wait(60)
        return {"cell_id": f"c{task}"}

    def on_fatal(event):
        fatal_events.append(dict(event))
        release.set()

    def consume(index, result):
        consumed.append(int(index))
        return {"receipt": True, "cell_id": result["cell_id"]}

    with pytest.raises(zpar.ParallelTaskError) as excinfo:
        zpar.pmap_reserved(
            worker,
            [0, 1, 2, 3],
            [1, 1, 1, 1],
            workers=2,
            cell_ids=["c0", "c1", "c2", "c3"],
            consume=consume,
            meta=meta,
            on_fatal=on_fatal,
            serial_fallback=False,
        )
    assert isinstance(excinfo.value.original, MemoryError)
    assert fatal_events and fatal_events[0]["index"] == 0
    # Pending tasks were never dispatched (parent-controlled admission).
    assert sorted(executed) == [0, 1], executed
    assert meta["dispatches"] == 2
    assert meta["pending_never_dispatched"] == [2, 3]
    assert consumed == [1]
    assert meta["drain"]["consumed"] == [1]
    assert meta["uncommitted_outputs"] == []


def test_pmap_reserved_spawn_never_dispatches_pending_and_consumes_started(
    monkeypatch, tmp_path
):
    _force_spawn_executor(monkeypatch)
    release = tmp_path / "release-sibling"
    receipt_path = tmp_path / "receipts.jsonl"
    executed = tmp_path / "executed.jsonl"
    meta: dict = {}

    def on_fatal(event):
        release.write_text("go", encoding="utf-8")

    def consume(index, result):
        _append_line(receipt_path, {"index": int(index), "cell_id": result["cell_id"]})
        return {"receipt": True, "cell_id": result["cell_id"]}

    def task(index, **extra):
        payload = {
            "index": index,
            "cell_id": f"c{index}",
            "executed_log": str(executed),
        }
        payload.update(extra)
        return payload

    tasks = [
        task(0, fail=True, wait_for_started=str(tmp_path / "sibling-started")),
        task(1, started_marker=str(tmp_path / "sibling-started"), wait_for=str(release)),
        task(2),
        task(3),
    ]
    with pytest.raises(zpar.ParallelTaskError) as excinfo:
        zpar.pmap_reserved(
            _spawn_drain_worker,
            tasks,
            [1, 1, 1, 1],
            workers=2,
            cell_ids=["c0", "c1", "c2", "c3"],
            consume=consume,
            meta=meta,
            on_fatal=on_fatal,
            serial_fallback=False,
        )
    assert isinstance(excinfo.value.original, MemoryError)
    assert sorted(entry["index"] for entry in _read_jsonl(executed)) == [0, 1]
    assert meta["dispatches"] == 2
    assert meta["pending_never_dispatched"] == [2, 3]
    lines = _read_jsonl(receipt_path)
    assert len(lines) == 1 and lines[0]["index"] == 1, lines


# ---------------------------------------------------------------------------
# 4. secondary worker failure is explicit, original stays primary
# ---------------------------------------------------------------------------


def test_secondary_worker_failure_is_recorded_without_masking_the_original(
    monkeypatch,
):
    _force_thread_executor(monkeypatch)
    barrier = threading.Barrier(2)
    release = threading.Event()
    consumed: list[int] = []
    meta: dict = {}

    def worker(task):
        barrier.wait(timeout=60)
        if task == 0:
            raise MemoryError("primary worker failure")
        assert release.wait(60)
        raise ValueError("secondary worker failure")

    def consume(index, result):  # pragma: no cover - must never be called
        consumed.append(int(index))
        return {"receipt": True}

    with pytest.raises(zpar.ParallelTaskError) as excinfo:
        zpar.pmap(
            worker,
            [0, 1],
            workers=2,
            consume=consume,
            meta=meta,
            on_fatal=lambda event: release.set(),
            serial_fallback=False,
        )
    assert excinfo.value.index == 0
    assert isinstance(excinfo.value.original, MemoryError)
    assert consumed == []
    assert [entry["index"] for entry in meta["sibling_worker_failures"]] == [1]
    assert "ValueError" in meta["sibling_worker_failures"][0]["error"]
    assert meta["drain"]["consumed"] == []


# ---------------------------------------------------------------------------
# 5. consumer (durable commit) failure: primary, never swallowed, no blind
#    consumption of the siblings that follow
# ---------------------------------------------------------------------------


class _StoreCommitBoom(RuntimeError):
    """Stands in for a StoreCommitNotDurable-shaped parent transaction error."""


def test_consumer_failure_stops_further_consumption_and_keeps_original(
    monkeypatch,
):
    _force_thread_executor(monkeypatch)
    barrier = threading.Barrier(2)
    release = threading.Event()
    consumed: list[int] = []
    meta: dict = {}
    boom = _StoreCommitBoom("durable commit refused")

    def worker(task):
        barrier.wait(timeout=60)
        if task == 1:
            assert release.wait(60)
        return {"cell_id": f"c{task}"}

    def consume(index, result):
        consumed.append(int(index))
        if index == 0:
            raise boom
        return {"receipt": True, "cell_id": result["cell_id"]}

    with pytest.raises(_StoreCommitBoom) as excinfo:
        zpar.pmap(
            worker,
            [0, 1],
            workers=2,
            consume=consume,
            meta=meta,
            on_fatal=lambda event: release.set(),
            serial_fallback=False,
        )
    assert excinfo.value is boom
    assert consumed == [0]
    assert meta["consume_policy"].startswith("stopped")
    assert [item["index"] for item in meta["uncommitted_outputs"]] == [1]
    assert meta["drain"]["consumed"] == []


def test_secondary_consumer_failure_during_drain_is_explicit(
    monkeypatch,
):
    _force_thread_executor(monkeypatch)
    barrier = threading.Barrier(3)
    release = threading.Event()
    consumed: list[int] = []
    meta: dict = {}
    secondary = _StoreCommitBoom("secondary durable commit failure")

    def worker(task):
        barrier.wait(timeout=60)
        if task == 0:
            raise MemoryError("primary worker failure")
        assert release.wait(60)
        return {"cell_id": f"c{task}"}

    def consume(index, result):
        consumed.append(int(index))
        if index == 1:
            raise secondary
        return {"receipt": True, "cell_id": result["cell_id"]}

    with pytest.raises(zpar.ParallelTaskError) as excinfo:
        zpar.pmap(
            worker,
            [0, 1, 2],
            workers=3,
            consume=consume,
            meta=meta,
            on_fatal=lambda event: release.set(),
            serial_fallback=False,
        )
    assert excinfo.value.index == 0
    assert isinstance(excinfo.value.original, MemoryError)
    assert consumed == [1]
    assert [entry["index"] for entry in meta["consume_failures"]] == [1]
    assert [item["index"] for item in meta["uncommitted_outputs"]] == [2]
    assert meta["consume_policy"].startswith("stopped")


# ---------------------------------------------------------------------------
# 6. broken pool: explicit, prompt, never replayed
# ---------------------------------------------------------------------------


def test_broken_process_pool_is_explicit_and_never_replays(monkeypatch, tmp_path):
    _force_spawn_executor(monkeypatch)
    executed = tmp_path / "executed.jsonl"
    release = tmp_path / "never-released"
    meta: dict = {}

    tasks = [
        {"index": 0, "cell_id": "c0", "die": True, "executed_log": str(executed)},
        {
            "index": 1,
            "cell_id": "c1",
            "executed_log": str(executed),
            "wait_for": str(release),
            "timeout_s": 30.0,
        },
    ]
    with pytest.raises(zpar.ParallelPoolError) as excinfo:
        zpar.pmap(
            _spawn_drain_worker,
            tasks,
            workers=2,
            meta=meta,
            serial_fallback=True,
        )
    chain = []
    exc = excinfo.value
    while exc is not None and len(chain) < 8:
        chain.append(type(exc).__name__)
        exc = exc.__cause__ or exc.__context__
    assert "BrokenProcessPool" in chain, chain
    # No serial replay: each task was executed at most once.
    indices = [entry["index"] for entry in _read_jsonl(executed)]
    assert len(indices) == len(set(indices)), indices


# ---------------------------------------------------------------------------
# 7. progress / heartbeat continue while the drain waits
# ---------------------------------------------------------------------------


def test_heartbeat_and_progress_advance_while_draining(monkeypatch):
    _force_thread_executor(monkeypatch)
    barrier = threading.Barrier(2)
    fatal_seen = threading.Event()
    drain_heartbeat = threading.Event()
    release = threading.Event()
    samples: list[int] = []
    meta: dict = {}

    def heartbeat():
        if meta.get("drain", {}).get("started"):
            drain_heartbeat.set()

    def worker(task):
        barrier.wait(timeout=60)
        if task == 0:
            raise MemoryError("primary worker failure")
        assert release.wait(60)
        return {"cell_id": f"c{task}"}

    def helper():
        assert fatal_seen.wait(60), "fatal state was never surfaced"
        # Release the sibling only once a drain heartbeat was really emitted.
        assert drain_heartbeat.wait(60), "no heartbeat during the drain"
        release.set()

    thread = threading.Thread(target=helper, daemon=True)
    thread.start()
    try:
        with pytest.raises(zpar.ParallelTaskError):
            zpar.pmap(
                worker,
                [0, 1],
                workers=2,
                consume=lambda index, result: {
                    "receipt": True,
                    "cell_id": result["cell_id"],
                },
                meta=meta,
                heartbeat=heartbeat,
                heartbeat_interval_s=0.05,
                progress_callback=lambda done, total: samples.append(int(done)),
                on_fatal=lambda event: fatal_seen.set(),
                serial_fallback=False,
            )
    finally:
        release.set()
        thread.join(timeout=30)
    assert not thread.is_alive()
    assert meta["drain"]["started"] is True
    assert meta["drain"]["consumed"] == [1]
    assert meta["heartbeat_calls"] >= 2
    assert samples and samples[-1] == 1, samples
    assert samples == sorted(samples)


# ---------------------------------------------------------------------------
# 8. success path + no-replay regressions
# ---------------------------------------------------------------------------


def test_success_path_consumes_each_result_exactly_once(monkeypatch):
    _force_thread_executor(monkeypatch)
    consumed: list[int] = []
    meta: dict = {}

    def worker(task):
        return {"cell_id": f"c{task}", "index": int(task)}

    def consume(index, result):
        consumed.append(int(index))
        return {"receipt": True, "index": int(index)}

    out = zpar.pmap(
        worker, [0, 1, 2], workers=2, consume=consume, meta=meta,
        serial_fallback=False,
    )
    assert [entry["index"] for entry in out] == [0, 1, 2]
    assert sorted(consumed) == [0, 1, 2]
    assert meta["consume_calls"] == 3
    assert meta["drain"]["started"] is False
    assert meta["uncommitted_outputs"] == []


def test_worker_failure_is_never_replayed_in_serial(monkeypatch):
    _force_thread_executor(monkeypatch)
    calls: list[int] = []

    def worker(task):
        calls.append(int(task))
        if task == 1:
            raise ValueError("deterministic worker failure")
        return int(task)

    with pytest.raises(zpar.ParallelTaskError) as excinfo:
        # serial_fallback=True is the historical default: it must NOT replay.
        zpar.pmap(worker, [0, 1, 2], workers=2, serial_fallback=True)
    assert excinfo.value.index == 1
    assert isinstance(excinfo.value.original, ValueError)
    assert len(calls) == len(set(calls)), calls
    assert calls.count(1) == 1


# ---------------------------------------------------------------------------
# 9. rework-1 (F1): a PARTIAL pool.submit failure still drains/consumes
# ---------------------------------------------------------------------------

_REAL_THREAD_POOL = cf.ThreadPoolExecutor


class _SubmitFailsAfterFirst:
    """Real thread pool wrapper: submit #1 succeeds, submit #2 fails at once.

    No sleep and no timing assumption: the failed submission is deterministic,
    and task 0 is either finished or still running — both are drained.
    """

    def __init__(self, max_workers=2, **kwargs):
        self._inner = _REAL_THREAD_POOL(max_workers=max_workers)
        self.submit_calls = 0

    def submit(self, fn, *args, **kwargs):
        self.submit_calls += 1
        if self.submit_calls >= 2:
            raise RuntimeError("injected submit failure")
        return self._inner.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True, cancel_futures=False):
        return self._inner.shutdown(wait=wait, cancel_futures=cancel_futures)


def test_pmap_partial_submit_failure_drains_started_success(monkeypatch):
    _force_thread_executor(monkeypatch)
    monkeypatch.setattr(cf, "ThreadPoolExecutor", _SubmitFailsAfterFirst)
    executed: list[int] = []
    consumed: list[int] = []
    threads: list[int] = []
    meta: dict = {}

    def worker(task):
        executed.append(int(task))
        return {"cell_id": f"c{task}"}

    def consume(index, result):
        consumed.append(int(index))
        threads.append(threading.get_ident())
        return {"receipt": True, "cell_id": result["cell_id"]}

    with pytest.raises(zpar.ParallelPoolError) as excinfo:
        zpar.pmap(
            worker, [0, 1, 2], workers=2, consume=consume, meta=meta,
            serial_fallback=False,
        )
    err = excinfo.value
    assert err.index == 1
    assert isinstance(err.__cause__, RuntimeError)
    # No new submission and no replay: only task 0 ever ran.
    assert executed == [0]
    # The already-submitted success is consumed exactly once, caller thread.
    assert consumed == [0]
    assert set(threads) == {threading.get_ident()}
    assert meta["drain"]["started"] is True
    assert meta["drain"]["consumed"] == [0]
    assert meta["drain"]["primary"]["kind"] == "submit"
    assert meta["submit_failed_index"] == 1
    assert meta["never_submitted"] == [1, 2]
    assert meta["uncommitted_outputs"] == []


def test_pmap_reserved_partial_submit_failure_drains_started_success(monkeypatch):
    _force_thread_executor(monkeypatch)
    monkeypatch.setattr(cf, "ThreadPoolExecutor", _SubmitFailsAfterFirst)
    executed: list[int] = []
    consumed: list[int] = []
    meta: dict = {}

    def worker(task):
        executed.append(int(task))
        return {"cell_id": f"c{task}"}

    def consume(index, result):
        consumed.append(int(index))
        return {"receipt": True, "cell_id": result["cell_id"]}

    with pytest.raises(zpar.ParallelPoolError) as excinfo:
        zpar.pmap_reserved(
            worker, [0, 1, 2, 3], [1, 1, 1, 1], workers=2,
            cell_ids=["c0", "c1", "c2", "c3"], consume=consume, meta=meta,
            serial_fallback=False,
        )
    err = excinfo.value
    assert err.index == 1
    assert isinstance(err.__cause__, RuntimeError)
    assert executed == [0]
    assert consumed == [0]
    assert meta["dispatches"] == 1  # only the SUCCESSFUL submission counts
    assert meta["submit_failures"] == [
        {"index": 1, "error": "RuntimeError: injected submit failure"}
    ]
    assert meta["pending_never_dispatched"] == [2, 3]
    assert meta["drain"]["started"] is True
    assert meta["drain"]["consumed"] == [0]
    assert meta["drain"]["primary"]["kind"] == "submit"


# ---------------------------------------------------------------------------
# 10. rework-1 (F2): one DONE batch never commits a sibling after a failure
# ---------------------------------------------------------------------------


def _force_single_done_batch(monkeypatch):
    """Deterministically deliver ALL futures in one batch (no sleep race)."""

    real_wait = cf.wait

    def _all_completed(futures, timeout=None, return_when=None):
        items = list(futures)
        real_wait(items, timeout=timeout, return_when=cf.ALL_COMPLETED)
        return set(items), set()

    monkeypatch.setattr(cf, "wait", _all_completed)


def test_pmap_same_done_batch_consumer_failure_never_consumes_sibling(
    monkeypatch,
):
    _force_thread_executor(monkeypatch)
    _force_single_done_batch(monkeypatch)
    consumed: list[int] = []
    meta: dict = {}
    boom = _StoreCommitBoom("batch durable commit failure")

    def worker(task):
        return {"cell_id": f"c{task}"}

    def consume(index, result):
        consumed.append(int(index))
        raise boom

    with pytest.raises(_StoreCommitBoom) as excinfo:
        zpar.pmap(
            worker, [0, 1], workers=2, consume=consume, meta=meta,
            serial_fallback=False,
        )
    assert excinfo.value is boom
    assert consumed == [0]
    assert [item["index"] for item in meta["uncommitted_outputs"]] == [1]
    assert meta["consume_policy"].startswith("stopped")


def test_reserved_same_done_batch_consumer_failure_never_consumes_sibling(
    monkeypatch,
):
    _force_thread_executor(monkeypatch)
    _force_single_done_batch(monkeypatch)
    consumed: list[int] = []
    meta: dict = {}
    boom = _StoreCommitBoom("batch durable commit failure")

    def worker(task):
        return {"cell_id": f"c{task}"}

    def consume(index, result):
        consumed.append(int(index))
        raise boom

    with pytest.raises(_StoreCommitBoom) as excinfo:
        zpar.pmap_reserved(
            worker, [0, 1], [1, 1], workers=2, cell_ids=["c0", "c1"],
            consume=consume, meta=meta, serial_fallback=False,
        )
    assert excinfo.value is boom
    assert consumed == [0]
    assert [item["index"] for item in meta["uncommitted_outputs"]] == [1]
    assert meta["consume_policy"].startswith("stopped")
