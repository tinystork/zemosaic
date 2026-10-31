"""Focused tests for the M3 test-only transparent observers.

These tests prove that each observer:
  * forwards the ORIGINAL callable's result unchanged,
  * writes a real trace entry attributed to the real process/frame,
  * arms the barrier ONLY at the exact safe-wave sampler call site.

No product run, no pressure and no store write happens here except one small
temporary store-free commit receipt mock.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import textwrap
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OBS_PATH = REPO_ROOT / "tools/zegrid_resume_m3/m3_observers.py"
MODE_PATH = REPO_ROOT / "src/zemosaic/zemosaic_zegrid_mode.py"


def _load_observers():
    spec = importlib.util.spec_from_file_location("m3_observers", OBS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


obs = _load_observers()


def _lines(path):
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def test_safe_site_resolves_to_the_cpu_wave_sampler_line():
    """The observer must resolve the CURRENT safe-wave sampler, not a frozen line.

    The contract is: the resolver returns a real 1-based line whose stripped
    source is exactly the sampler anchor, AND that line lives inside the CPU
    wave sampler function (the one that also contains the wave loop head),
    after the loop head. A drifted/removed seam therefore still fails.
    """
    source_text = MODE_PATH.read_text(encoding="utf-8")
    line = obs.resolve_safe_sampler_line(source_text)
    assert line > 0  # sanity bound: a real 1-based source line
    lines = source_text.splitlines()
    assert lines[line - 1].strip() == obs.CPU_WAVE_ANCHOR

    tree = ast.parse(source_text)
    enclosing = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno or node.lineno
            if node.lineno <= line <= end:
                if enclosing is None or node.lineno > enclosing.lineno:
                    enclosing = node
    assert enclosing is not None, "sampler line is not inside any function"
    end = enclosing.end_lineno or enclosing.lineno
    loop_lines = [
        index
        for index, text in enumerate(lines, start=1)
        if enclosing.lineno <= index <= end
        and text.strip() == obs.CPU_WAVE_LOOP_ANCHOR
    ]
    assert loop_lines, "sampler is not inside a function holding the CPU wave loop"
    assert loop_lines[0] < line, "sampler must follow the CPU wave loop head"


def test_execution_observer_forwards_result_and_records_pid(tmp_path):
    calls = []

    def fake_stack_cell(task):
        calls.append(task)
        return {"cell_id": task[0].cell_id, "ok": True}

    module = types.SimpleNamespace(_stack_cell=fake_stack_cell)
    sink = obs.TraceSink(tmp_path / "execution.jsonl")
    observer = obs.ExecutionObserver(sink).install(module)

    task = (types.SimpleNamespace(cell_id="cell-7"), "patch", None, "/tmp/c", "stream", 1, "dense")
    result = module._stack_cell(task)

    assert result == {"cell_id": "cell-7", "ok": True}
    assert calls == [task]
    assert module._stack_cell is not fake_stack_cell
    assert module._stack_cell._m3_original is fake_stack_cell
    starts = [x for x in _lines(sink.path) if x["kind"] == "execution_start"]
    ends = [x for x in _lines(sink.path) if x["kind"] == "execution_end"]
    assert len(starts) == 1 and len(ends) == 1
    assert starts[0]["cell_id"] == "cell-7"
    assert starts[0]["worker_pid"] == os.getpid()
    assert ends[0]["status"] == "ok"
    assert ends[0]["end_monotonic_ns"] >= ends[0]["start_monotonic_ns"]


def test_execution_observer_forwards_exceptions_unchanged(tmp_path):
    class Boom(RuntimeError):
        pass

    def fake_stack_cell(task):
        raise Boom("original failure")

    module = types.SimpleNamespace(_stack_cell=fake_stack_cell)
    sink = obs.TraceSink(tmp_path / "execution.jsonl")
    obs.ExecutionObserver(sink).install(module)

    task = (types.SimpleNamespace(cell_id="cell-err"), None, None, None, None, None, None)
    try:
        module._stack_cell(task)
    except Boom as exc:
        assert str(exc) == "original failure"
    else:  # pragma: no cover - the exception must propagate
        raise AssertionError("exception was not forwarded")
    ends = [x for x in _lines(sink.path) if x["kind"] == "execution_end"]
    assert ends and ends[0]["status"] == "error"


def test_commit_observer_forwards_receipt_unchanged(tmp_path):
    class Receipt:
        cell_id = "c1"
        attempt_id = "cpu-wave:0:0:c1"
        payload_sha256 = "ab" * 32
        stored_bytes = 1234
        logical_bytes = 4096
        idempotent = False
        commit_file = "/store/commits/c1.json"

    receipt = Receipt()

    class Store:
        def commit_cell(self, result, *, attempt_id, backend=None, plan_generation=0):
            self.last = (result, attempt_id, backend, plan_generation)
            return receipt

    sink = obs.TraceSink(tmp_path / "commit.jsonl")
    original = Store.commit_cell
    observer = obs.CommitObserver(sink).install(Store)
    assert Store.commit_cell is not original

    store = Store()
    out = store.commit_cell({"cell_id": "c1"}, attempt_id="cpu-wave:0:0:c1", backend="cpu")
    assert out is receipt
    assert store.last == ({"cell_id": "c1"}, "cpu-wave:0:0:c1", "cpu", 0)
    records = [x for x in _lines(sink.path) if x["kind"] == "commit"]
    assert len(records) == 1
    assert records[0]["cell_id"] == "c1"
    assert records[0]["attempt_id"] == "cpu-wave:0:0:c1"
    assert records[0]["payload_sha256"] == "ab" * 32
    assert records[0]["stored_bytes"] == 1234
    assert observer.durable_commit_count() == 1


def test_wait_observer_captures_the_real_coordinator_journal(tmp_path):
    from zemosaic.core.zegrid import resource_recovery as zrec

    def probe():
        raise AssertionError("probe must be forwarded unchanged, never called here")

    coordinator = zrec.RecoveryCoordinator(
        run_id="run-1",
        release=lambda: zrec.ReleaseReport(
            stopped_admission=True, cancelled_units=0, drained_units=0,
            credits_balanced=True, notes=[],
        ),
        probe=probe,
        stop_event=__import__("threading").Event(),
        profile=zrec.StorageProfilePolicy(requested="auto", effective="bounded"),
        initial_interval_s=zrec.PROBE_INITIAL_INTERVAL_S,
        max_interval_s=zrec.PROBE_MAX_INTERVAL_S,
        required_observations=zrec.REQUIRED_ADMISSIBLE_OBSERVATIONS,
        min_spacing_s=zrec.MIN_ADMISSIBLE_SPACING_S,
    )
    sink = obs.TraceSink(tmp_path / "wait.jsonl")
    observer = obs.WaitObserver(sink)
    observer._capture(coordinator)

    coordinator._log("waiting_resources", minimum_bytes=123)
    coordinator._log("readmitting")

    events = [x for x in _lines(sink.path) if x["kind"] == "wait_event"]
    assert [e["event"] for e in events] == ["waiting_resources", "readmitting"]
    assert events[0]["minimum_bytes"] == 123
    # The real journal still holds the events: forwarding preserved behaviour.
    assert [e["event"] for e in coordinator.journal()] == [
        "waiting_resources", "readmitting",
    ]
    assert observer.coordinators == [coordinator]


def _synthetic_site_module(tmp_path):
    """Build a module whose sampler call sits EXACTLY on the resolved anchor line."""
    loop_line_text = "while offset < total_tasks:"
    anchor_text = "facts = _r55_run_checkpoint_sample(disk_root)"
    header = [
        "def _r55_run_checkpoint_sample(disk_root):",
        "    return SOURCE(disk_root)",
        "",
        "def other_site(disk_root):",
        "    return _r55_run_checkpoint_sample(disk_root)",
        "",
        "def _run_single(offset, total_tasks, disk_root):",
        "    facts = None",
    ]
    # Pad so the anchor lands immediately after the loop marker.
    body = list(header)
    body.append("    " + loop_line_text)
    body.append("        " + anchor_text)
    body.append("        break")
    body.append("    return facts")
    text = "\n".join(body) + "\n"
    path = tmp_path / "synthetic_mode.py"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_barrier_arms_only_at_the_exact_safe_site(tmp_path):
    import threading

    path = _synthetic_site_module(tmp_path)
    spec = importlib.util.spec_from_file_location("synthetic_mode", path)
    module = importlib.util.module_from_spec(spec)

    def source(disk_root):
        return {
            "host": {"raw_available_bytes": 1000, "source": "injected"},
            "sampled_monotonic_ns": 42,
        }

    module.SOURCE = source
    spec.loader.exec_module(module)
    module.SOURCE = source

    safe_line = obs.resolve_safe_sampler_line(path.read_text(encoding="utf-8"))
    sink = obs.TraceSink(tmp_path / "barrier.jsonl")
    hook_calls = []
    observer = obs.BarrierObserver(
        sink,
        mode="experiment",
        hook=lambda: hook_calls.append(1) or {"pressure_applied": True},
        minimum_commits=0,
        source_path=path,
    ).install(module)
    assert observer.safe_line == safe_line
    observer.arm()

    # Negative: a different call site in another function must NOT arm.
    module.other_site("/tmp")
    assert hook_calls == []
    assert not observer.one_shot_consumed

    # Positive: the exact safe-wave call site arms exactly once.
    module._run_single(0, 4, "/tmp")
    module._run_single(0, 4, "/tmp")
    assert hook_calls == [1]
    assert observer.armed_count == 1

    records = _lines(sink.path)
    samplers = [x for x in records if x["kind"] == "sampler"]
    assert [s["attribution"] for s in samplers] == [
        "other_site", "safe_wave_site", "safe_wave_site",
    ]
    hooks = [x for x in records if x["kind"] == "barrier_hook"]
    assert len(hooks) == 1
    assert hooks[0]["safe_site_line"] == safe_line


def test_barrier_requires_k_durable_commits(tmp_path):
    path = _synthetic_site_module(tmp_path)
    spec = importlib.util.spec_from_file_location("synthetic_mode_k", path)
    module = importlib.util.module_from_spec(spec)
    module.SOURCE = lambda root: {
        "host": {"raw_available_bytes": 5, "source": "injected"},
        "sampled_monotonic_ns": 1,
    }
    spec.loader.exec_module(module)
    module.SOURCE = lambda root: {
        "host": {"raw_available_bytes": 5, "source": "injected"},
        "sampled_monotonic_ns": 1,
    }

    sink = obs.TraceSink(tmp_path / "barrier-k.jsonl")
    hook_calls = []
    commits = {"n": 1}
    observer = obs.BarrierObserver(
        sink,
        mode="experiment",
        hook=lambda: hook_calls.append(1) or {"pressure_applied": True},
        durable_commit_count=lambda: commits["n"],
        minimum_commits=2,
        source_path=path,
    ).install(module)
    observer.arm()

    module._run_single(0, 4, "/tmp")
    assert hook_calls == []
    assert not observer.one_shot_consumed

    commits["n"] = 2
    module._run_single(0, 4, "/tmp")
    assert hook_calls == [1]


def test_observe_mode_never_arms(tmp_path):
    path = _synthetic_site_module(tmp_path)
    spec = importlib.util.spec_from_file_location("synthetic_mode_observe", path)
    module = importlib.util.module_from_spec(spec)
    module.SOURCE = lambda root: {
        "host": {"raw_available_bytes": 7, "source": "injected"},
        "sampled_monotonic_ns": 3,
    }
    spec.loader.exec_module(module)
    module.SOURCE = lambda root: {
        "host": {"raw_available_bytes": 7, "source": "injected"},
        "sampled_monotonic_ns": 3,
    }
    sink = obs.TraceSink(tmp_path / "barrier-observe.jsonl")
    hook_calls = []
    observer = obs.BarrierObserver(
        sink, mode="observe", hook=lambda: hook_calls.append(1),
        minimum_commits=0, source_path=path,
    ).install(module)
    observer.arm()
    module._run_single(0, 2, "/tmp")
    assert hook_calls == []
    samplers = [x for x in _lines(sink.path) if x["kind"] == "sampler"]
    assert samplers[0]["attribution"] == "safe_wave_site"
    assert samplers[0]["armed_here"] is False


def test_sampled_facts_are_returned_byte_identical(tmp_path):
    path = _synthetic_site_module(tmp_path)
    spec = importlib.util.spec_from_file_location("synthetic_mode_facts", path)
    module = importlib.util.module_from_spec(spec)
    marker = {"host": {"raw_available_bytes": 321, "source": "injected"}, "sampled_monotonic_ns": 9, "x": [1, 2]}
    module.SOURCE = lambda root: marker
    spec.loader.exec_module(module)
    module.SOURCE = lambda root: marker
    sink = obs.TraceSink(tmp_path / "facts.jsonl")
    obs.BarrierObserver(sink, mode="observe", minimum_commits=0, source_path=path).install(module)
    assert module._run_single(0, 1, "/tmp") is marker
    assert marker == {"host": {"raw_available_bytes": 321, "source": "injected"}, "sampled_monotonic_ns": 9, "x": [1, 2]}
