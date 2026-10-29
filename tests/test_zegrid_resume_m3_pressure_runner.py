"""Focused tests for the M3 Linux pressure runner (guards, IPC, refusal).

The physical-pressure experiment is NEVER executed here. Guards are exercised
with simulated children: injected clocks, memory readers, RSS readers and kill
callables. The allocator/IPC paths are unit-tested without touching the host.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "tools/zegrid_resume_m3/m3_linux_pressure_runner.py"

MIB = 1 << 20
GIB = 1 << 30


def _load_runner():
    spec = importlib.util.spec_from_file_location("m3_linux_pressure_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rn = _load_runner()


def test_runner_refuses_artifact_roots_inside_a_git_checkout(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    inside = repo / "artifacts"
    try:
        rn.ensure_outside_git(inside, what="--out")
    except SystemExit as exc:
        assert "git checkout" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a tracked checkout must be refused")
    outside = tmp_path / "evidence"
    assert rn.ensure_outside_git(outside) == outside.resolve()


def test_relative_artifact_paths_resolve_under_the_base(tmp_path):
    base = tmp_path / "base"
    assert rn.resolve_artifact_path("reference", base) == (base / "reference").resolve()
    absolute = tmp_path / "absolute"
    assert rn.resolve_artifact_path(absolute, base) == absolute.resolve()


def test_mem_available_is_parsed_from_the_labelled_meminfo_line(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text(
        "MemTotal:        8000000 kB\n"
        "MemAvailable:    2583000 kB\n"
        "SwapTotal:       1000 kB\n",
        encoding="utf-8",
    )
    assert rn.read_mem_available_bytes(path) == 2583000 * 1024
    assert rn.read_mem_available_bytes() == rn.read_mem_available_bytes()


def test_guard_accepts_a_healthy_sample():
    decision = rn.evaluate_guard(
        pressure_rss_bytes=480 * MIB,
        mem_available_bytes=2 * GIB,
    )
    assert decision.ok is True


def test_guard_kills_on_pressure_rss_breach():
    decision = rn.evaluate_guard(
        pressure_rss_bytes=513 * MIB,
        mem_available_bytes=2 * GIB,
    )
    assert not decision.ok
    assert decision.code == "PRESSURE_RSS_BREACH"


def test_guard_kills_on_reserve_breach():
    decision = rn.evaluate_guard(
        pressure_rss_bytes=100 * MIB,
        mem_available_bytes=(1 * GIB) - 1,
    )
    assert not decision.ok
    assert decision.code == "RESERVE_BREACH"


def test_guard_rejects_missing_telemetry():
    assert rn.evaluate_guard(pressure_rss_bytes=1, mem_available_bytes=None).code == "MISSING_TELEMETRY"
    assert rn.evaluate_guard(pressure_rss_bytes=None, mem_available_bytes=2 * GIB).code == "MISSING_TELEMETRY"
    assert (
        rn.evaluate_guard(
            pressure_rss_bytes=1,
            mem_available_bytes=2 * GIB,
            product_tree_rss_bytes=None,
            product_ceiling_bytes=10 * MIB,
        ).code
        == "MISSING_TELEMETRY"
    )


def test_guard_rejects_contract_violations_before_probing():
    assert (
        rn.evaluate_guard(pressure_rss_bytes=0, mem_available_bytes=2 * GIB, reserve_bytes=GIB - 1).code
        == "RESERVE_BELOW_CONTRACT"
    )
    assert (
        rn.evaluate_guard(
            pressure_rss_bytes=0, mem_available_bytes=2 * GIB,
            pressure_cap_bytes=512 * MIB + 1,
        ).code
        == "CAP_ABOVE_CONTRACT"
    )


def test_guard_stops_on_product_tree_ceiling_breach():
    decision = rn.evaluate_guard(
        pressure_rss_bytes=10 * MIB,
        mem_available_bytes=2 * GIB,
        product_tree_rss_bytes=200 * MIB,
        product_ceiling_bytes=150 * MIB,
    )
    assert not decision.ok
    assert decision.code == "PRODUCT_CEILING_BREACH"


def test_ceiling_is_calibrated_from_the_reference():
    assert rn.calibrate_product_ceiling(300 * MIB) == 300 * MIB + 64 * MIB
    try:
        rn.calibrate_product_ceiling(None)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("an uncalibrated ceiling must be refused")


def test_supervisor_releases_only_the_owned_pressure_pid():
    killed = []
    rss_state = {"value": 400 * MIB}

    def fake_kill(pid):
        killed.append(pid)
        rss_state["value"] = 0
        return True

    supervisor = rn.PressureSupervisor(
        pressure_pid=4242,
        mem_available_fn=lambda: 2 * GIB,
        pressure_rss_fn=lambda pid: rss_state["value"],
        clock=lambda: time.monotonic(),
        kill_fn=fake_kill,
        sample_interval_s=0.001,
    )
    assert supervisor.sample_once().ok is True
    assert supervisor.release_and_wait(timeout_s=1.0) is True
    assert killed == [4242]


def test_supervisor_reports_guard_breach_and_release_reason():
    events = []

    class Sink:
        def emit(self, kind, **fields):
            events.append((kind, fields))

    supervisor = rn.PressureSupervisor(
        pressure_pid=7,
        mem_available_fn=lambda: GIB - 10,
        pressure_rss_fn=lambda pid: 10 * MIB,
        clock=lambda: time.monotonic(),
        event_sink=Sink(),
        kill_fn=lambda pid: True,
    )
    decision = supervisor.sample_once()
    assert decision.code == "RESERVE_BREACH"
    assert [k for k, _ in events] == ["guard_sample"]
    assert events[0][1]["decision"]["code"] == "RESERVE_BREACH"


def test_ack_timeout_is_not_run_and_pressure_is_released(tmp_path):
    server = rn.HookServer(tmp_path / "hook.sock", ack_deadline_s=0.2)
    started = time.monotonic()
    response = server.serve_once(lambda request: {"pressure_applied": True})
    elapsed = time.monotonic() - started
    server.close()
    assert response["pressure_applied"] is False
    assert response["status"] == "NOT_RUN"
    assert response["reason"] == "ack_timeout_no_request"
    assert elapsed < 2.0


def test_ack_deadline_above_contract_is_refused(tmp_path):
    try:
        rn.HookServer(tmp_path / "hook.sock", ack_deadline_s=2.5)
    except ValueError as exc:
        assert "2 s" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a >2 s ack deadline must be refused")


def test_hook_server_applies_pressure_once_and_acks_measured_effect(tmp_path):
    server = rn.HookServer(tmp_path / "hook.sock", ack_deadline_s=1.0)
    seen = []

    def apply_fn(request):
        seen.append(request)
        return {
            "pressure_applied": True,
            "pressure_rss_at_ack": 480 * MIB,
            "mem_available_at_ack": 2 * GIB,
            "measured_drop_bytes": 470 * MIB,
        }

    result = {}

    def client():
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(tmp_path / "hook.sock"))
        sock.sendall(b'{"wave": 3}\n')
        result.update(json.loads(sock.makefile("r", encoding="utf-8").readline()))
        sock.close()

    thread = threading.Thread(target=client)
    thread.start()
    response = server.serve_once(apply_fn)
    thread.join(timeout=3)
    server.close()

    assert seen == [{"wave": 3}]
    assert response["pressure_applied"] is True
    assert response["pressure_rss_at_ack"] == 480 * MIB
    assert result["pressure_applied"] is True


def test_experiment_refuses_by_default_without_any_child(tmp_path, monkeypatch):
    monkeypatch.delenv("M3_PRESSURE_EXPERIMENT_REVIEWED", raising=False)
    out = tmp_path / "experiment.json"
    args = SimpleNamespace(
        corpus=None, reference=None, out=str(tmp_path / "x"),
        store=str(tmp_path / "store"), json_out=str(out),
        workers=1, k_commits=2, reserve_bytes=GIB,
        pressure_bytes=480 * MIB, pressure_cap_bytes=512 * MIB,
        ack_deadline_s=2.0, max_lifetime_s=900.0, i_am_reviewed=False,
    )
    code = rn.run_experiment(args)
    assert code == rn.EXIT_NOT_RUN
    document = json.loads(out.read_text())
    assert document["status"] == "NOT_RUN"
    assert document["pressure_applied"] is False
    assert document["reason"] == "PRESSURE_EXPERIMENT_NOT_AUTHORIZED"
    assert not (tmp_path / "x").exists()
    assert not (tmp_path / "store").exists()


def test_experiment_refuses_even_with_flag_without_env_grant(tmp_path, monkeypatch):
    monkeypatch.delenv("M3_PRESSURE_EXPERIMENT_REVIEWED", raising=False)
    out = tmp_path / "experiment2.json"
    args = SimpleNamespace(
        corpus=None, reference=None, out=str(tmp_path / "y"),
        store=str(tmp_path / "store"), json_out=str(out),
        workers=1, k_commits=2, reserve_bytes=GIB,
        pressure_bytes=480 * MIB, pressure_cap_bytes=512 * MIB,
        ack_deadline_s=2.0, max_lifetime_s=900.0, i_am_reviewed=True,
    )
    assert rn.run_experiment(args) == rn.EXIT_NOT_RUN
    assert json.loads(out.read_text())["reason"] == "PRESSURE_EXPERIMENT_NOT_AUTHORIZED"


def test_experiment_contract_is_still_enforced_when_granted(tmp_path, monkeypatch):
    monkeypatch.setenv("M3_PRESSURE_EXPERIMENT_REVIEWED", "1")
    out = tmp_path / "experiment3.json"
    args = SimpleNamespace(
        corpus=None, reference=None, out=str(tmp_path / "z"),
        store=str(tmp_path / "store"), json_out=str(out),
        workers=1, k_commits=2, reserve_bytes=GIB,
        pressure_bytes=600 * MIB, pressure_cap_bytes=512 * MIB,
        ack_deadline_s=2.0, max_lifetime_s=900.0, i_am_reviewed=True,
    )
    assert rn.run_experiment(args) == rn.EXIT_USAGE
    document = json.loads(out.read_text())
    assert document["status"] == "FAIL"
    assert document["reason"] == "DOSE_ABOVE_CAP"


def test_reviewed_flow_still_does_not_execute_pressure_in_this_mission(tmp_path, monkeypatch):
    monkeypatch.setenv("M3_PRESSURE_EXPERIMENT_REVIEWED", "1")
    out = tmp_path / "experiment4.json"
    args = SimpleNamespace(
        corpus=None, reference=None, out=str(tmp_path / "w"),
        store=str(tmp_path / "store"), json_out=str(out),
        workers=1, k_commits=2, reserve_bytes=GIB,
        pressure_bytes=480 * MIB, pressure_cap_bytes=512 * MIB,
        ack_deadline_s=2.0, max_lifetime_s=900.0, i_am_reviewed=True,
    )
    assert rn.run_experiment(args) == rn.EXIT_NOT_RUN
    document = json.loads(out.read_text())
    assert document["status"] == "NOT_RUN"
    assert document["reason"] == "REVIEWED_FLOW_NOT_EXECUTED_IN_THIS_MISSION"
    assert document["pressure_applied"] is False


def test_extract_aggregate_records_the_json_path():
    manifest = {
        "r55_scheduler": {
            "execution": {"aggregate_bytes": 999},
            "plan_observation": {"budget_bytes": 5000, "raw_available_bytes": 6000, "layout": [4, 4]},
        },
        "per_cell_diagnostics": {"ram_contract": {"aggregate_bytes": 999}},
    }
    provenance = rn.extract_aggregate(manifest)
    assert provenance["aggregate_bytes"] == 999
    assert provenance["json_path"] == "r55_scheduler.execution.aggregate_bytes"
    assert provenance["plan_budget_bytes"] == 5000
    assert provenance["final_layout"] == [4, 4]
    assert rn.extract_aggregate({})["aggregate_bytes"] is None


def test_pressure_child_allocator_is_bounded_and_releases(tmp_path):
    """Drive the allocator directly (no supervision, no physical experiment)."""
    import subprocess

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"step_bytes": 1 * MIB, "cap_bytes": 2 * MIB}))
    proc = subprocess.Popen(
        [sys.executable, str(RUNNER_PATH), "_pressure-child", str(spec)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    try:
        ready = json.loads(proc.stdout.readline())
        assert ready["ready"] is True
        for _ in range(2):
            proc.stdin.write("+\n")
            proc.stdin.flush()
            assert json.loads(proc.stdout.readline())["ok"] is True
        proc.stdin.write("+\n")  # third step would exceed the 2 MiB cap
        proc.stdin.flush()
        assert json.loads(proc.stdout.readline())["ok"] is False
        proc.stdin.write("q\n")
        proc.stdin.flush()
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    assert proc.returncode == 0
