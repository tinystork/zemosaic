"""ZM-CELL-LIVE-COMMIT (lot 1) — commit each CPU cell at RESULT RECEPTION.

The defect this pins: ``pmap_reserved`` collected into ``out[idx]`` and returned
only after the WHOLE wave, so a cell finished in minutes stayed in RAM (and was
lost on stop) for as long as its slowest peer ran. The durable store, fence and
transaction order are UNCHANGED; only the moment of the parent transaction moves.

Contracts pinned here:

A. executor seam (``pmap_reserved`` / ``pmap`` / serial / fallback):
   * the optional ``consume(index, result)`` runs in the PARENT at receipt, before
     any slower peer of the same batch returns;
   * a consumer failure is FATAL and keeps its ORIGINAL type; there is no serial
     replay and no second consumption of the same task;
   * already finished siblings are consumed before the fatal error propagates;
   * the returned list holds the consumer's LIGHT value, never the payload;
   * out-of-order reception is counted by reception, not by a canonical prefix.

B. ``_cpu_durable_commit`` (the extracted durable step):
   * verification (``require_commit``) happens BEFORE counters/progress;
   * a silent no-op commit is a terminal ``StoreCommitNotDurable`` and the
     payload is NOT released;
   * no configured store -> no durable claim (``None``), and nothing counted.

C. production seam (spawned real run, real durable store, tiny corpus):
   * a fast cell is COMMITTED and readable in the real store while a slow peer is
     still gated inside its worker;
   * killing the run after that commit, then resuming, recomputes ONLY the
     pending cells and produces the SAME science as an uninterrupted baseline;
   * a commit failure is terminal, keeps prior commits, and never emits COMMITTED
     for the failed cell;
   * a worker failure keeps prior commits and reports no false COMMITTED.
"""

from __future__ import annotations

import concurrent.futures as cf
import functools
import gc
import hashlib
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
import weakref
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
from zemosaic.core.zegrid import parallel as zpar  # noqa: E402
from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

import zegrid_live_commit_harness as GH  # noqa: E402

MIB = 2 ** 20
CORPUS_TOOL = REPO_ROOT / "tools" / "zegrid_m6" / "make_tiny_corpus.py"
ALL_CELLS = ("r0000c0000", "r0000c0001", "r0001c0000", "r0001c0001")

TOOLS_M4 = REPO_ROOT / "tools" / "zegrid_resume_m4"
if str(TOOLS_M4) not in sys.path:
    sys.path.insert(0, str(TOOLS_M4))
import m4_resume_harness as M4  # noqa: E402


# ---------------------------------------------------------------------------
# A. executor seam
# ---------------------------------------------------------------------------


def _force_thread_executor(monkeypatch):
    """Force pmap's THREAD path so a gate can be an in-process Event."""
    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: True)


class _LifetimePayload:
    """A worker result whose LIFETIME the R1 witnesses actually measure.

    ``weakref`` needs a real instance (a dict is not weak-referenceable), and the
    payload must stay picklable so the PROCESS executor exercises the same code.
    """

    def __init__(self, cell_id: str, size: int = 2 ** 20) -> None:
        self.cell_id = cell_id
        self.blob = bytearray(int(size))


#: Gates shared with the pool children (``fork`` inherits this dict).
_LIFETIME: dict = {}


def _lifetime_worker(task):
    """Module-level worker: gate the slow cell, return a weakref-able payload."""
    payload = _LifetimePayload(str(task))
    if str(task) == "slow":
        _LIFETIME["started"].set()
        assert _LIFETIME["release"].wait(180)
    return payload


def _weakref_alive(ref):
    return ref() is not None


def test_reserved_consumer_commits_fast_result_while_slow_peer_still_runs(
    monkeypatch,
):
    _force_thread_executor(monkeypatch)
    slow_running = threading.Event()
    release = threading.Event()
    consumed: list[tuple] = []

    def worker(task):
        if task == "slow":
            slow_running.set()
            assert release.wait(60), "slow worker never released"
            return {"cell_id": "slow", "payload": "slow-bytes"}
        assert slow_running.wait(60), "slow worker never started"
        return {"cell_id": "fast", "payload": "fast-bytes"}

    def consume(index, result):
        consumed.append((int(index), result["cell_id"], release.is_set()))
        if result["cell_id"] == "fast":
            # PROOF of receipt-time consumption: the slow peer has NOT returned.
            assert release.is_set() is False
            release.set()
        return {"receipt": True, "cell_id": result["cell_id"]}

    out = zpar.pmap_reserved(
        worker,
        ["slow", "fast"],
        [1, 1],
        workers=2,
        cell_ids=["slow", "fast"],
        consume=consume,
        serial_fallback=False,
    )
    assert consumed[0] == (1, "fast", False), consumed
    assert out[1] == {"receipt": True, "cell_id": "fast"}
    assert out[0] == {"receipt": True, "cell_id": "slow"}


def test_pmap_consumer_commits_fast_result_while_slow_peer_still_runs(monkeypatch):
    _force_thread_executor(monkeypatch)
    slow_running = threading.Event()
    release = threading.Event()
    consumed: list[tuple] = []

    def worker(task):
        if task == "slow":
            slow_running.set()
            assert release.wait(60), "slow worker never released"
            return {"cell_id": "slow"}
        assert slow_running.wait(60), "slow worker never started"
        return {"cell_id": "fast"}

    def consume(index, result):
        consumed.append((int(index), result["cell_id"], release.is_set()))
        if result["cell_id"] == "fast":
            assert release.is_set() is False
            release.set()
        return {"receipt": True, "cell_id": result["cell_id"]}

    out = zpar.pmap(
        worker,
        ["slow", "fast"],
        workers=2,
        consume=consume,
        serial_fallback=False,
    )
    assert consumed[0] == (1, "fast", False), consumed
    assert out == [
        {"receipt": True, "cell_id": "slow"},
        {"receipt": True, "cell_id": "fast"},
    ]


def test_serial_path_consumes_each_result_before_the_next_task_runs():
    order: list[str] = []

    def worker(task):
        order.append(f"work:{task}")
        return {"cell_id": task}

    def consume(index, result):
        order.append(f"consume:{result['cell_id']}")
        return {"receipt": True, "cell_id": result["cell_id"]}

    meta: dict = {}
    out = zpar.pmap_reserved(
        worker,
        ["a", "b"],
        [1, 1],
        workers=1,
        cell_ids=["a", "b"],
        consume=consume,
        meta=meta,
    )
    assert order == ["work:a", "consume:a", "work:b", "consume:b"], order
    assert meta["executor"] == "serial"
    assert meta["consume_calls"] == 2
    assert out == [
        {"receipt": True, "cell_id": "a"},
        {"receipt": True, "cell_id": "b"},
    ]


def test_serial_fallback_still_consumes_after_pool_creation_failure(monkeypatch):
    def _no_pool(*_args, **_kwargs):
        raise OSError("no pool for this test")

    monkeypatch.setattr(cf, "ProcessPoolExecutor", _no_pool)
    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: False)
    consumed: list[tuple] = []

    def worker(task):
        return {"cell_id": task}

    def consume(index, result):
        consumed.append((int(index), result["cell_id"]))
        return {"receipt": True}

    meta: dict = {}
    out = zpar.pmap_reserved(
        worker,
        ["a", "b"],
        [1, 1],
        workers=2,
        cell_ids=["a", "b"],
        consume=consume,
        meta=meta,
        serial_fallback=True,
    )
    assert meta["executor"] == "serial"
    assert meta["fallback"] is True
    assert consumed == [(0, "a"), (1, "b")], consumed
    assert len(out) == 2


class _CommitBoom(RuntimeError):
    """A parent-side transaction failure (stands in for a store failure)."""


def test_consumer_failure_is_fatal_with_its_own_type_and_never_replays(monkeypatch):
    _force_thread_executor(monkeypatch)
    calls: list[str] = []
    consumed: list[str] = []
    boom = _CommitBoom("durable commit failed")

    def worker(task):
        calls.append(task)
        return {"cell_id": task}

    def consume(index, result):
        consumed.append(result["cell_id"])
        if result["cell_id"] == "b":
            raise boom
        return {"receipt": True, "cell_id": result["cell_id"]}

    with pytest.raises(_CommitBoom) as excinfo:
        zpar.pmap_reserved(
            worker,
            ["a", "b"],
            [1, 1],
            workers=2,
            cell_ids=["a", "b"],
            consume=consume,
            serial_fallback=False,
        )
    # The ORIGINAL exception object, not a ParallelPoolError/ParallelTaskError.
    assert excinfo.value is boom
    # No serial replay and no double consumption.
    assert sorted(calls) == ["a", "b"], calls
    assert consumed == ["a", "b"], consumed


def test_consumer_failure_in_pmap_keeps_the_original_type(monkeypatch):
    _force_thread_executor(monkeypatch)
    boom = _CommitBoom("durable commit failed")

    def worker(task):
        return {"cell_id": task}

    def consume(index, result):
        raise boom

    with pytest.raises(_CommitBoom) as excinfo:
        zpar.pmap(
            worker, ["a", "b"], workers=2, consume=consume, serial_fallback=False
        )
    assert excinfo.value is boom


def test_finished_siblings_are_consumed_before_the_error_propagates():
    """A computed peer must never be lost behind a still-running task."""
    futures = [cf.Future(), cf.Future(), cf.Future()]
    futures[0].set_result({"cell_id": "a"})
    futures[1].set_result({"cell_id": "b"})
    # futures[2] stays PENDING: it must not be waited for nor consumed.
    running = {futures[0]: 0, futures[1]: 1, futures[2]: 2}
    out: list = [None, None, None]
    consumed: list[int] = []
    meta = {"consume_failures": []}
    done_idx: list[int] = []

    def consume(index, result):
        consumed.append(int(index))
        return {"receipt": True, "cell_id": result["cell_id"]}

    zpar._consume_finished_siblings(
        running, ["a", "b", "c"], [1, 1, 1], out, consume, meta, done_idx.append
    )
    assert consumed == [0, 1], consumed
    assert out[0] == {"receipt": True, "cell_id": "a"}
    assert out[1] == {"receipt": True, "cell_id": "b"}
    assert out[2] is None
    assert done_idx == [0, 1]
    assert list(running) == [futures[2]]
    assert meta["consume_failures"] == []


def test_consumer_output_never_retains_the_payload(monkeypatch):
    _force_thread_executor(monkeypatch)
    payload = {
        "cell_id": "a",
        "science": np.zeros((64, 64, 3), dtype=np.float32),
    }

    def worker(task):
        return payload

    def consume(index, result):
        assert result is payload
        return {"receipt": True, "cell_id": result["cell_id"]}

    out = zpar.pmap_reserved(
        worker,
        ["a", "b"],
        [1, 1],
        workers=2,
        cell_ids=["a", "b"],
        consume=consume,
        serial_fallback=False,
    )
    for entry in out:
        assert isinstance(entry, dict)
        assert "science" not in entry
        assert entry.get("receipt") is True


@pytest.mark.parametrize("mapper", ["pmap_reserved", "pmap"])
@pytest.mark.parametrize("executor", ["thread", "process"])
def test_consumed_payload_is_released_while_the_slow_peer_is_blocked(
    monkeypatch, mapper, executor
):
    """R1: a CONSUMED payload must be dead while a slow peer is still blocked.

    ``Future._result`` is the real owner of a returned payload; clearing the local
    ``value`` is not enough. This measures the OBJECT LIFETIME with a weakref from
    a parent heartbeat (i.e. after the collection pass and BEFORE the next
    blocking wait), for both executors and both maps.
    """
    if executor == "thread":
        _force_thread_executor(monkeypatch)
    ctx = multiprocessing.get_context("fork")
    started, release = ctx.Event(), ctx.Event()
    _LIFETIME.clear()
    _LIFETIME.update({"started": started, "release": release})
    state = {
        "ref": None,
        "consumed_fast": False,
        "slow_blocked_at_probe": None,
        "alive_at_probe": None,
    }
    probe = {"done": False}

    def consume(index, result):
        state["ref"] = weakref.ref(result)
        if result.cell_id == "fast":
            state["consumed_fast"] = True
        return {"receipt": True, "cell_id": result.cell_id}

    def heartbeat():
        if not state["consumed_fast"] or probe["done"]:
            return
        state["slow_blocked_at_probe"] = bool(
            started.is_set() and not release.is_set()
        )
        gc.collect()
        state["alive_at_probe"] = _weakref_alive(state["ref"])
        probe["done"] = True
        release.set()

    if mapper == "pmap_reserved":
        out = zpar.pmap_reserved(
            _lifetime_worker,
            ["slow", "fast"],
            [1, 1],
            workers=2,
            cell_ids=["slow", "fast"],
            consume=consume,
            heartbeat=heartbeat,
            heartbeat_interval_s=0.01,
            serial_fallback=False,
        )
    else:
        out = zpar.pmap(
            _lifetime_worker,
            ["slow", "fast"],
            workers=2,
            consume=consume,
            heartbeat=heartbeat,
            heartbeat_interval_s=0.01,
            serial_fallback=False,
        )
    assert probe["done"] is True, "the parent heartbeat never probed the payload"
    assert state["slow_blocked_at_probe"] is True
    assert state["alive_at_probe"] is False, (
        "the CONSUMED payload was still alive while the slow peer was blocked: "
        "the completed Future (hence its result) is still referenced"
    )
    assert [entry["cell_id"] for entry in out] == ["slow", "fast"]


@pytest.mark.parametrize("mapper", ["pmap_reserved", "pmap"])
def test_progress_is_reported_at_reception_while_the_first_task_still_runs(
    monkeypatch, mapper
):
    """R3: progress must count RECEPTIONS, not a contiguous recorded prefix."""
    _force_thread_executor(monkeypatch)
    started, gate = threading.Event(), threading.Event()
    samples: list[tuple] = []

    def worker(task):
        if task == "slow":
            started.set()
            assert gate.wait(60), "slow worker never released"
            return {"cell_id": "slow"}
        assert started.wait(60), "slow worker never started"
        return {"cell_id": "fast"}

    def consume(index, result):
        return {"receipt": True, "cell_id": result["cell_id"]}

    def progress(done, total):
        samples.append((int(done), int(total), gate.is_set()))
        if int(done) >= 1:
            gate.set()

    if mapper == "pmap_reserved":
        out = zpar.pmap_reserved(
            worker,
            ["slow", "fast"],
            [1, 1],
            workers=2,
            cell_ids=["slow", "fast"],
            consume=consume,
            progress_callback=progress,
            serial_fallback=False,
        )
    else:
        out = zpar.pmap(
            worker,
            ["slow", "fast"],
            workers=2,
            consume=consume,
            progress_callback=progress,
            serial_fallback=False,
        )
    assert (1, 2, False) in samples, samples
    assert [entry["cell_id"] for entry in out] == ["slow", "fast"]


def test_spawn_executor_consumes_at_receipt(monkeypatch, tmp_path):
    """PORTABLE executor check: a SPAWN pool is plumbed like fork (no fork-only claim)."""
    ctx = multiprocessing.get_context("spawn")
    monkeypatch.setattr(
        cf,
        "ProcessPoolExecutor",
        functools.partial(cf.ProcessPoolExecutor, mp_context=ctx),
    )
    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: False)
    marker = tmp_path / "slow-finished"
    consumed: list[tuple] = []

    def consume(index, result):
        consumed.append((result["cell_id"], marker.exists()))
        return {"receipt": True, "cell_id": result["cell_id"]}

    tasks = [
        {"cell_id": "slow", "delay_s": 0.8, "marker": str(marker)},
        {"cell_id": "fast"},
    ]
    meta: dict = {}
    out = zpar.pmap(
        GH.spawn_probe_worker,
        tasks,
        workers=2,
        consume=consume,
        meta=meta,
        serial_fallback=False,
    )
    assert meta["executor"] == "process"
    assert ("fast", False) in consumed, consumed
    assert [entry["cell_id"] for entry in out] == ["slow", "fast"]


def test_reception_progress_is_out_of_order_counted_not_a_prefix(monkeypatch):
    """Progress counts RECEPTIONS: an out-of-order wave still reaches total/total."""
    _force_thread_executor(monkeypatch)
    samples: list[int] = []
    barrier = threading.Barrier(2)

    def worker(task):
        # task "0" is slower than "1" but both must be collected out of order.
        if task == 0:
            time.sleep(0.05)
        return {"cell_id": str(task)}

    def consume(index, result):
        return {"receipt": True, "cell_id": result["cell_id"]}

    out = zpar.pmap_reserved(
        worker,
        [0, 1],
        [1, 1],
        workers=2,
        cell_ids=["0", "1"],
        consume=consume,
        progress_callback=lambda done, total: samples.append(int(done)),
        serial_fallback=False,
    )
    assert out[0]["cell_id"] == "0" and out[1]["cell_id"] == "1"
    assert samples and samples[-1] == 2, samples


# ---------------------------------------------------------------------------
# B. the extracted durable step
# ---------------------------------------------------------------------------


def _real_store(tmp_path: Path, run_id: str = "live-commit-l1"):
    rows = M4.ensure_inputs(tmp_path / "in")
    identity = M4.make_identity(run_id, rows)
    store = zrs.RunCellStore.open(
        tmp_path / f"store-{run_id}",
        identity=identity,
        budget=zrs.DiskBudget(durable_cell_bytes=64 * MIB),
        session_id=run_id,
    )
    return store


def _registry(run_id: str = "live-commit-l1"):
    return zrr.RunResultRegistry(
        run_id=run_id,
        expected_cell_ids=list(M4.CELL_IDS),
        gauge_mode="per_cell",
    )


def _store_state(store):
    return {
        "store": store,
        "publishing": True,
        "commits": 0,
        "released_cells": 0,
        "released_bytes": 0,
        "degraded_reason": None,
        "commit_errors": [],
    }


def test_durable_step_verifies_before_counting_and_releases_after_commit(tmp_path):
    store = _real_store(tmp_path)
    registry = _registry()
    try:
        cell_id = M4.CELL_IDS[0]
        receipt = registry.accept(
            M4.compute_cell(0), backend="cpu", plan_generation=0, attempt_id="a1"
        )
        state = _store_state(store)
        handoff = zz._cpu_durable_commit(
            state, registry, cell_id, receipt, plan_generation=0
        )
        assert state["commits"] == 1
        assert state["released_cells"] == 1
        assert state["released_bytes"] >= 1
        assert handoff["commit"]["cell_id"] == cell_id
        # Durable: an independent read-only open sees the verified commit.
        reader = zrs.RunCellStore.open(store.root, read_only=True, create=False)
        try:
            record = reader.require_commit(cell_id, "a1")
            assert record.cell_id == cell_id
        finally:
            reader.close()
        # The large payload was released only AFTER the commit.
        meta = registry.metadata(cell_id)
        assert meta["released"] is True
    finally:
        registry.dispose()
        store.close()


def test_silent_commit_no_op_is_terminal_and_never_releases_the_payload(
    tmp_path, monkeypatch
):
    store = _real_store(tmp_path)
    registry = _registry()
    try:
        def _silent(self, registry, cell_id, **kwargs):
            return {"commit": {}, "release": {"released": False}}

        monkeypatch.setattr(zrs.RunCellStore, "commit_and_release", _silent)
        cell_id = M4.CELL_IDS[0]
        receipt = registry.accept(
            M4.compute_cell(0), backend="cpu", plan_generation=0, attempt_id="a1"
        )
        state = _store_state(store)
        with pytest.raises(zrs.StoreCommitNotDurable) as excinfo:
            zz._cpu_durable_commit(
                state, registry, cell_id, receipt, plan_generation=0
            )
        assert excinfo.value.cell_id == cell_id
        assert state["commits"] == 0
        # A silent no-op is caught by VERIFICATION (the store itself did not
        # raise). R2: a store that cannot verify the record it just accepted is
        # marked UNHEALTHY, so no later cell of the same run can ever claim
        # progress without a durable record.
        assert state["publishing"] is False
        assert str(state["degraded_reason"]).startswith("verification:")
        assert state["commit_errors"] and state["commit_errors"][0]["cell_id"] == cell_id
        # Nothing durable was ever claimed and the payload is retained.
        assert registry.metadata(cell_id)["released"] is False
        assert not list((store.root / "commits").glob("*.json"))
    finally:
        registry.dispose()
        store.close()


def test_durable_step_without_a_store_makes_no_durable_claim(tmp_path):
    registry = _registry()
    try:
        receipt = registry.accept(
            M4.compute_cell(0), backend="cpu", plan_generation=0, attempt_id="a1"
        )
        state = _store_state(None)
        handoff = zz._cpu_durable_commit(
            state, registry, M4.CELL_IDS[0], receipt, plan_generation=0
        )
        assert handoff is None
        assert state["commits"] == 0
        assert registry.metadata(M4.CELL_IDS[0])["released"] is False
    finally:
        registry.dispose()


def test_intake_gate_fails_closed_after_the_durable_store_degrades():
    """R2: durable intent is FROZEN; a degraded store can never be M1 progress."""
    healthy = {
        "store": object(),
        "publishing": True,
        "degraded_reason": None,
    }
    assert zz._cpu_cell_intake_refusal(True, healthy, "c-1") is None
    assert zz._cpu_cell_intake_refusal(False, healthy, "c-1") is None

    for state in (
        {"store": object(), "publishing": False, "degraded_reason": "boom"},
        {"store": object(), "publishing": True, "degraded_reason": "boom"},
        {"store": None, "publishing": False, "degraded_reason": None},
    ):
        # Durable intent: refuse (never fall back to memory-only progression).
        refusal = zz._cpu_cell_intake_refusal(True, state, "c-1")
        assert isinstance(refusal, zz._R55DurableStoreUnavailable), state
        assert "never continues as memory-only" in str(refusal)
        # No durable intent (store-less run): the M1 in-memory claim is unchanged.
        assert zz._cpu_cell_intake_refusal(False, state, "c-1") is None


# ---------------------------------------------------------------------------
# C. production seam — spawned real run, real durable store
# ---------------------------------------------------------------------------


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "in"
    proc = subprocess.run(
        [sys.executable, str(CORPUS_TOOL), str(corpus)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return corpus


def _spawn(tmp_path: Path, corpus: Path, out: Path, **kwargs):
    store = out / "zegrid-durable"
    status = tmp_path / f"status-{out.name}.json"
    # FORK: exactly what the production cell dispatcher uses on POSIX (a spawned
    # run would re-import the pytest main module). The gate crosses the process
    # boundary through the FILESYSTEM, never through inherited memory.
    proc = __import__("multiprocessing").get_context("fork").Process(
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


def _wait_until(predicate, timeout_s=300.0, interval_s=0.05):
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval_s)
    return predicate()


def _commit_files(store: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    commits = store / "commits"
    if not commits.is_dir():
        return out
    for path in sorted(commits.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out[str(payload.get("cell_id"))] = path
    return out


def _executed(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        json.loads(line)["cell_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _status(path: Path) -> dict:
    assert path.exists(), "the harness never reported a terminal status"
    return json.loads(path.read_text(encoding="utf-8"))


def _log(out: Path) -> str:
    path = out / "zegrid_run.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _progress_samples(text: str) -> list[int]:
    """Every ``per_cell_stack N/total`` progress sample, in log order."""
    import re

    pattern = re.compile(r"phase PROGRESS: per_cell_stack\s+(\d+)/(\d+)")
    return [int(done) for done, _total in pattern.findall(text)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _product(out: Path) -> Path:
    return out / "mosaic_grid.fits"


def test_fast_cell_is_durably_committed_before_the_slow_peer_returns(tmp_path):
    """REAL seam: cell B is committed AND readable while cell A is still gated."""
    corpus = _corpus(tmp_path)
    out = tmp_path / "out"
    release = tmp_path / "release-a"
    done_marker = tmp_path / "a-returned"
    executed = tmp_path / "executed.log"
    proc, status_path, store = _spawn(
        tmp_path,
        corpus,
        out,
        gate_cell="r0000c0000",
        release_path=str(release),
        done_marker_cell="r0000c0000",
        done_marker_path=str(done_marker),
        executed_log=str(executed),
    )
    try:
        # The FAST peer must become durable while A is still inside its worker.
        committed = _wait_until(lambda: _commit_files(store), timeout_s=300.0)
        assert committed, "no cell was durably committed while the slow peer ran"
        assert not done_marker.exists(), (
            "the gated cell A returned BEFORE the first commit was observed"
        )
        fast_ids = [cid for cid in committed if cid != "r0000c0000"]
        assert fast_ids, committed
        assert "r0000c0000" not in committed

        # Readable from the REAL store, in an independent read-only handle, while
        # the run keeps going and A is still gated.
        reader = zrs.RunCellStore.open(store, read_only=True, create=False)
        try:
            for cid in fast_ids:
                record = reader.require_commit(cid)
                assert record.cell_id == cid
            # The committed bytes themselves are readable (not just an index).
            payload = reader.load_payload(fast_ids[0])
            assert payload["cell_id"] == fast_ids[0]
        finally:
            reader.close()
        assert not done_marker.exists()

        log = _log(out)
        for cid in fast_ids:
            assert f"cell COMMITTED id={cid}" in log, log[-4000:]
        assert "cell COMMITTED id=r0000c0000" not in log
    finally:
        release.write_text("go", encoding="utf-8")
        proc.join(timeout=600)
        if proc.is_alive():  # pragma: no cover - defensive
            proc.kill()
            proc.join(timeout=60)

    assert _status(status_path)["status"] == "done"
    assert set(_commit_files(store)) == set(ALL_CELLS)
    assert sorted(_executed(executed)) == sorted(ALL_CELLS)
    # R2 invariant: every progress sample is backed by a PERSISTED unique commit.
    samples = _progress_samples(_log(out))
    assert samples and max(samples) == len(_commit_files(store)) == 4, samples


def test_kill_after_fast_commit_resumes_only_pending_and_matches_baseline(tmp_path):
    corpus = _corpus(tmp_path)
    # 1. Uninterrupted baseline on the SAME corpus (same identity/plan).
    baseline_out = tmp_path / "baseline"
    base_proc, base_status, base_store = _spawn(
        tmp_path, corpus, baseline_out, executed_log=str(tmp_path / "baseline.log")
    )
    base_proc.join(timeout=900)
    assert not base_proc.is_alive()
    assert _status(base_status)["status"] == "done"
    assert set(_commit_files(base_store)) == set(ALL_CELLS)

    # 2. Interrupted run: kill it while A is still gated, after B is committed.
    out = tmp_path / "interrupted"
    release = tmp_path / "release"
    executed = tmp_path / "interrupted.log"
    proc, status_path, store = _spawn(
        tmp_path,
        corpus,
        out,
        gate_cell="r0000c0000",
        release_path=str(release),
        executed_log=str(executed),
    )
    committed_before_kill: dict[str, Path] = {}
    try:
        committed_before_kill = _wait_until(
            lambda: _commit_files(store), timeout_s=300.0
        )
        assert committed_before_kill, "no durable commit before the kill"
        assert "r0000c0000" not in committed_before_kill
    finally:
        # Kill the WHOLE run tree (the harness run is its own session): the pool
        # workers die with it, exactly like stopping the application.
        os.killpg(proc.pid, signal.SIGKILL)
        proc.join(timeout=120)
    assert not status_path.exists(), "a SIGKILLed run must not write a status"

    durable = set(_commit_files(store))
    assert durable == set(committed_before_kill)

    # 3. Resume: ONLY the pending cells may run.
    resume_executed = tmp_path / "resume.log"
    resume_out = out
    resume_proc, resume_status, _ = _spawn(
        tmp_path,
        corpus,
        resume_out,
        executed_log=str(resume_executed),
        resume_mode="auto",
    )
    resume_proc.join(timeout=900)
    assert not resume_proc.is_alive()
    payload = _status(resume_status)
    assert payload["status"] == "done", payload

    expected_pending = set(ALL_CELLS) - durable
    recomputed = set(_executed(resume_executed))
    assert recomputed == expected_pending, (recomputed, expected_pending)
    assert not (recomputed & durable), "a committed cell was recomputed on resume"
    assert set(_commit_files(store)) == set(ALL_CELLS)

    # 4. Science parity with the uninterrupted baseline.
    assert _sha256(_product(out)) == _sha256(_product(baseline_out))
    assert _sha256(out / "mosaic_grid_coverage.fits") == _sha256(
        baseline_out / "mosaic_grid_coverage.fits"
    )


def test_commit_failure_is_terminal_keeps_prior_commits_and_no_false_commit(
    tmp_path,
):
    corpus = _corpus(tmp_path)
    out = tmp_path / "out"
    release = tmp_path / "release"
    executed = tmp_path / "executed.log"
    proc, status_path, store = _spawn(
        tmp_path,
        corpus,
        out,
        gate_cell="r0000c0000",
        release_path=str(release),
        executed_log=str(executed),
        commit_fail_on_call=2,
    )
    try:
        committed = _wait_until(
            lambda: len(_commit_files(store)) >= 1, timeout_s=300.0
        )
        assert committed, "the first cell should have committed before the failure"
    finally:
        release.write_text("go", encoding="utf-8")
        proc.join(timeout=600)
        if proc.is_alive():  # pragma: no cover - defensive
            proc.kill()
            proc.join(timeout=60)

    payload = _status(status_path)
    assert payload["status"] == "error", payload
    assert payload["type"] == "StoreCommitNotDurable", payload
    # Exactly the first cell stayed durable; the failed one is NOT reported.
    durable = _commit_files(store)
    assert len(durable) == 1, durable
    failed = [cid for cid in ALL_CELLS if cid not in durable]
    log = _log(out)
    for cid in failed:
        assert f"cell COMMITTED id={cid}" not in log, log[-4000:]
    # The failing cell WAS computed (COMPUTED) but never COMMITTED.
    assert "cell COMPUTED id=" in log
    # No cell is committed twice (no end-of-wave second pass).
    assert len(list((store / "commits").glob("*.json"))) == len(durable)
    # R2 invariant: progress never exceeds the PERSISTED unique commits.
    samples = _progress_samples(log)
    assert max(samples) == len(durable) == 1, samples
    assert len([line for line in log.splitlines() if "cell COMMITTED" in line]) == 1


def test_no_progress_or_committed_after_the_first_commit_fails(tmp_path):
    """R2: a durable run whose FIRST commit fails reports NO processed cell.

    The already-computed siblings are intaked AFTER the store degraded (same wait
    batch or drain). They must be refused explicitly instead of being counted as
    M1 progress or emitted as COMMITTED.
    """
    corpus = _corpus(tmp_path)
    out = tmp_path / "out"
    release = tmp_path / "release"
    executed = tmp_path / "executed.log"
    proc, status_path, store = _spawn(
        tmp_path,
        corpus,
        out,
        gate_cell="r0000c0000",
        release_path=str(release),
        executed_log=str(executed),
        commit_fail_on_call=1,
    )
    try:
        # The gated cell A CANNOT return before ``release`` exists, so waiting for
        # the first COMPUTED line guarantees the failing commit (and any sibling
        # intake that follows it) happened while A was still pending.
        assert _wait_until(
            lambda: "cell COMPUTED" in _log(out), timeout_s=300.0
        ), "no cell was ever computed before the injected commit failure"
    finally:
        release.write_text("go", encoding="utf-8")
        proc.join(timeout=600)
        if proc.is_alive():  # pragma: no cover - defensive
            proc.kill()
            proc.join(timeout=60)

    payload = _status(status_path)
    assert payload["status"] == "error", payload
    assert payload["type"] == "StoreCommitNotDurable", payload
    durable = _commit_files(store)
    assert durable == {}, durable
    log = _log(out)
    assert "cell COMMITTED" not in log, log[-4000:]
    assert _progress_samples(log) == [], log[-4000:]
    # At least one cell was really computed (the failure is not a no-op run).
    assert "cell COMPUTED id=" in log


def test_worker_failure_keeps_prior_commits_and_reports_no_false_commit(tmp_path):
    corpus = _corpus(tmp_path)
    out = tmp_path / "out"
    release = tmp_path / "release"
    fail_release = tmp_path / "fail-now"
    executed = tmp_path / "executed.log"
    proc, status_path, store = _spawn(
        tmp_path,
        corpus,
        out,
        gate_cell="r0000c0000",
        release_path=str(release),
        fail_cell="r0001c0001",
        fail_release_path=str(fail_release),
        executed_log=str(executed),
    )
    try:
        committed = _wait_until(lambda: _commit_files(store), timeout_s=300.0)
        assert committed, "no cell committed before the injected worker failure"
        assert "r0001c0001" not in committed
    finally:
        # Trigger the worker failure first, then release the gated peer so the
        # run can drain and terminate (the failure must be observed while A is
        # still pending, which the assertions above already proved).
        fail_release.write_text("boom", encoding="utf-8")
        release.write_text("go", encoding="utf-8")
        proc.join(timeout=600)
        if proc.is_alive():  # pragma: no cover - defensive
            proc.kill()
            proc.join(timeout=60)

    payload = _status(status_path)
    assert payload["status"] == "error", payload
    chain = "\n".join(payload.get("chain") or [])
    assert "ParallelTaskError" in chain, chain
    assert "injected cell worker failure" in chain, chain
    durable = _commit_files(store)
    assert durable, "prior durable success was lost"
    assert "r0001c0001" not in durable
    log = _log(out)
    assert "cell COMMITTED id=r0001c0001" not in log
