"""M5-B RSS bench harness tests — fail-closed guards, evidence and timing.

Cheap tests only: no 512/1024 matrix. They exercise process-failure precedence,
typed pre-payload refusal vs generic/post-progress errors, mandatory named
checks, real monitoring evidence, normal-exit races, payload-content
immutability, temporal ordering, sticky matrix failures and the one-real
128x128/4-cell store+guard+parity round trip on the real filesystem.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools" / "zegrid_resume_m5b"
for _p in (str(REPO_ROOT / "src"), str(TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rss_bench as rb  # noqa: E402

TINY = {"name": "tiny", "canvas": (128, 128), "grid": (2, 2), "halo": 4}
WORKSPACE_ROOT = rb.WORKSPACE_ROOT


def _read_result(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _good_run_from(run: dict) -> dict:
    return {
        "returncode": int(run.get("returncode", 0)),
        "timed_out": False,
        "guard_breach": None,
        "samples": int(run.get("samples", 20)),
        "valid_rss_samples": int(run.get("valid_rss_samples", 20)),
        "valid_host_samples": int(run.get("valid_host_samples", 20)),
        "valid_paired_samples": int(run.get("valid_paired_samples", 20)),
        "valid_assembly_samples": int(run.get("valid_assembly_samples", 5)),
        "assembly_window_observed": True,
        "sampled_peak_rss_bytes": 200 * 2 ** 20,
        "sampled_assembly_peak_rss_bytes": 190 * 2 ** 20,
        "sampled_min_available_bytes": rb.MIN_RESERVE_BYTES + 64 * 2 ** 20,
    }


@pytest.fixture(scope="module")
def tiny_store():
    base = Path(tempfile.mkdtemp(prefix=".m5b-rss-pytest-", dir=str(WORKSPACE_ROOT)))
    spec = rb.validate_case(TINY)
    store = base / "store"
    seed_run = rb.run_child(
        rb._case_argv("seed", spec, store, result=str(base / "seed.json")),
        log_path=base / "seed.log",
        timeout_s=60,
    )
    assert seed_run["returncode"] == 0, seed_run
    seed = _read_result(base / "seed.json")
    assert seed["status"] == "SEEDED"
    yield base, spec, store, seed
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(scope="module")
def tiny_measure(tiny_store):
    base, spec, store, _seed = tiny_store
    run = rb.run_child(
        rb._case_argv(
            "measure", spec, store, result=str(base / "measure.json")
        ),
        log_path=base / "measure.log",
        timeout_s=60,
    )
    result = _read_result(base / "measure.json")
    return base, spec, store, run, result


@pytest.fixture
def fresh_store():
    base = Path(tempfile.mkdtemp(prefix=".m5b-rss-mut-", dir=str(WORKSPACE_ROOT)))
    spec = rb.validate_case({"name": "mut", "canvas": (96, 96), "grid": (2, 2), "halo": 4})
    store = base / "store"
    run = rb.run_child(
        rb._case_argv("seed", spec, store, result=str(base / "seed.json")),
        log_path=base / "seed.log",
        timeout_s=60,
    )
    assert run["returncode"] == 0
    yield base, spec, store
    shutil.rmtree(base, ignore_errors=True)


# ---------------------------------------------------------------------------
# Filesystem / limits
# ---------------------------------------------------------------------------


def test_filesystem_guard_rejects_ram_backed_tmp() -> None:
    info = rb._fs_info(Path("/tmp"))
    if str(info.get("fstype") or "").lower() not in rb.MEMORY_FS:
        pytest.skip("/tmp is not RAM-backed on this host")
    with pytest.raises(RuntimeError, match="RAM-backed"):
        rb.assert_real_disk(Path("/tmp"))


def test_workspace_artifact_base_is_real_disk() -> None:
    base = Path(tempfile.mkdtemp(prefix=".m5b-rss-fs-", dir=str(WORKSPACE_ROOT)))
    try:
        info = rb.assert_real_disk(base)
        assert info["real_disk"] is True
        assert str(info.get("fstype") or "").lower() not in rb.MEMORY_FS
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_artifact_base_must_stay_in_workspace() -> None:
    assert rb._ensure_under_workspace(WORKSPACE_ROOT / ".m5b-x") == (
        WORKSPACE_ROOT / ".m5b-x"
    )
    with pytest.raises(RuntimeError, match="outside the owned workspace"):
        rb._ensure_under_workspace(Path("/tmp/m5b-outside"))


def test_unknown_filesystem_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(
        rb, "_fs_info", lambda _path: {"fstype": "", "free_bytes": 10 ** 12}
    )
    with pytest.raises(RuntimeError, match="unknown"):
        rb.assert_real_disk(Path("/home/tristan/.openclaw/workspace"))


def test_case_limits_are_enforced() -> None:
    assert rb.validate_case({"name": "ok", "canvas": (1024, 1024), "grid": (8, 8)})[
        "cells"
    ] == 64
    with pytest.raises(ValueError, match="exceeds 1024"):
        rb.validate_case({"name": "big", "canvas": (2048, 512), "grid": (2, 2)})
    with pytest.raises(ValueError, match="exceeds 64 cells"):
        rb.validate_case({"name": "many", "canvas": (512, 512), "grid": (9, 8)})
    with pytest.raises(ValueError, match="halo"):
        rb.validate_case({"name": "halo", "canvas": (512, 512), "grid": (2, 2), "halo": 0})


# ---------------------------------------------------------------------------
# Pure guard / verdict logic
# ---------------------------------------------------------------------------


def test_monitor_decision_requires_telemetry() -> None:
    assert rb.monitor_decision(rss_bytes=1, available_bytes=rb.MIN_RESERVE_BYTES) is None
    assert "CHILD_RSS_CAP" in rb.monitor_decision(
        rss_bytes=rb.MAX_CHILD_RSS_BYTES + 1,
        available_bytes=rb.MIN_RESERVE_BYTES,
    )
    assert "RESERVE_BREACH" in rb.monitor_decision(
        rss_bytes=1, available_bytes=rb.MIN_RESERVE_BYTES - 1
    )
    assert "TELEMETRY_UNAVAILABLE" in rb.monitor_decision(
        rss_bytes=None, available_bytes=rb.MIN_RESERVE_BYTES
    )
    assert "TELEMETRY_UNAVAILABLE" in rb.monitor_decision(
        rss_bytes=1, available_bytes=None
    )
    assert "TELEMETRY_UNAVAILABLE" in rb.monitor_decision(
        rss_bytes=None, available_bytes=None
    )


def test_classification_requires_every_real_condition(tiny_measure) -> None:
    _base, spec, _store, run, result = tiny_measure
    verdict = rb.classify_measurement(
        run=run,
        result=result,
        store_commit_before=result["digests"]["store_commits_before"],
        store_commit_after=result["digests"]["store_commits_after"],
        expected_case=spec,
    )
    assert verdict["status"] == "PASS", verdict["reasons"]
    assert result["status"] == "PASS"
    assert result["counters"]["max_resident_cells"] == 2
    assert result["counters"]["peak_live_cells"] == 2
    assert all(result["checks"].values())
    assert set(rb.REQUIRED_CHECKS) <= set(result["checks"])
    assert result["digests"]["output"] == result["digests"]["output_after_close"]
    assert result["stats_after_close"]["live_cells"] == 0


def test_nonzero_returncode_beats_a_pass_json(tiny_measure) -> None:
    _base, spec, _store, run, result = tiny_measure
    verdict = rb.classify_measurement(
        run=dict(run, returncode=2),
        result=copy.deepcopy(result),
        store_commit_before=result["digests"]["store_commits_before"],
        store_commit_after=result["digests"]["store_commits_after"],
        expected_case=spec,
    )
    assert verdict["status"] == "FAIL"
    assert any("returncode" in reason for reason in verdict["reasons"])


def test_timeout_beats_a_refused_json(tiny_measure) -> None:
    _base, _spec, _store, run, result = tiny_measure
    refused = {
        "schema": rb.SCHEMA_MEASURE,
        "status": "REFUSED",
        "refusal": {"typed": True},
        "assembly": {"completed": False, "progressed": False},
        "stats": {"loads": 0, "verifications": 0, "live_mappings": 0},
        "checks": {"refused_before_payload": True},
    }
    verdict = rb.classify_measurement(
        run=dict(run, timed_out=True, returncode=2),
        result=refused,
        store_commit_before=None,
        store_commit_after=None,
    )
    assert verdict["status"] == "FAIL"
    assert verdict["status"] != "NOT_RUN"
    assert any("timed out" in reason for reason in verdict["reasons"])


def test_partial_json_is_not_success(tiny_measure) -> None:
    _base, spec, _store, run, result = tiny_measure
    partial = {"schema": result["schema"], "status": "PASS"}
    verdict = rb.classify_measurement(
        run=run,
        result=partial,
        store_commit_before=None,
        store_commit_after=None,
        expected_case=spec,
    )
    assert verdict["status"] == "FAIL"
    assert any("partial" in reason for reason in verdict["reasons"])


def test_nonfinite_measurement_is_rejected(tiny_measure) -> None:
    _base, spec, _store, run, result = tiny_measure
    broken = copy.deepcopy(result)
    broken["assembly"]["elapsed_s"] = math.nan
    verdict = rb.classify_measurement(
        run=run,
        result=broken,
        store_commit_before=None,
        store_commit_after=None,
        expected_case=spec,
    )
    assert verdict["status"] == "FAIL"


def test_changed_commit_digest_is_rejected(tiny_measure) -> None:
    _base, spec, _store, run, result = tiny_measure
    verdict = rb.classify_measurement(
        run=run,
        result=copy.deepcopy(result),
        store_commit_before="a" * 64,
        store_commit_after="b" * 64,
        expected_case=spec,
    )
    assert verdict["status"] == "FAIL"
    assert any("commit digest changed" in reason for reason in verdict["reasons"])


def test_changed_payload_content_digest_is_rejected(tiny_measure) -> None:
    _base, spec, _store, run, result = tiny_measure
    verdict = rb.classify_measurement(
        run=run,
        result=copy.deepcopy(result),
        store_commit_before=None,
        store_commit_after=None,
        store_content_before="a" * 64,
        store_content_after="b" * 64,
        require_content_digest=True,
        expected_case=spec,
    )
    assert verdict["status"] == "FAIL"
    assert any("CONTENT digest changed" in reason for reason in verdict["reasons"])


def test_missing_parent_content_digest_is_rejected(tiny_measure) -> None:
    _base, spec, _store, run, result = tiny_measure
    verdict = rb.classify_measurement(
        run=run,
        result=copy.deepcopy(result),
        store_commit_before=None,
        store_commit_after=None,
        require_content_digest=True,
        expected_case=spec,
    )
    assert verdict["status"] == "FAIL"
    assert any("content digest is missing" in reason for reason in verdict["reasons"])


def test_diagnostic_budget_is_never_qualified(tiny_measure) -> None:
    _base, spec, _store, run, result = tiny_measure
    bad = copy.deepcopy(result)
    bad["qualified_measurement"] = False
    verdict = rb.classify_measurement(
        run=run,
        result=bad,
        store_commit_before=None,
        store_commit_after=None,
        expected_case=spec,
    )
    assert verdict["status"] == "FAIL"


def test_empty_checks_cannot_pass(tiny_measure) -> None:
    _base, spec, _store, run, result = tiny_measure
    broken = copy.deepcopy(result)
    broken["checks"] = {}
    verdict = rb.classify_measurement(
        run=run,
        result=broken,
        store_commit_before=None,
        store_commit_after=None,
        expected_case=spec,
    )
    assert verdict["status"] == "FAIL"
    assert any("checks are missing/empty" in reason for reason in verdict["reasons"])


def test_missing_one_mandatory_check_cannot_pass(tiny_measure) -> None:
    _base, spec, _store, run, result = tiny_measure
    broken = copy.deepcopy(result)
    broken["checks"].pop("store_metadata_unchanged")
    verdict = rb.classify_measurement(
        run=run,
        result=broken,
        store_commit_before=None,
        store_commit_after=None,
        expected_case=spec,
    )
    assert verdict["status"] == "FAIL"
    assert any("missing required checks" in reason for reason in verdict["reasons"])


def test_zero_monitoring_samples_are_not_pass(tiny_measure) -> None:
    _base, spec, _store, run, result = tiny_measure
    bad_run = dict(
        run,
        valid_rss_samples=0,
        valid_host_samples=0,
        valid_paired_samples=0,
        valid_assembly_samples=0,
        assembly_window_observed=False,
        sampled_peak_rss_bytes=0,
        sampled_assembly_peak_rss_bytes=None,
        sampled_min_available_bytes=None,
    )
    verdict = rb.classify_measurement(
        run=bad_run,
        result=copy.deepcopy(result),
        store_commit_before=None,
        store_commit_after=None,
        expected_case=spec,
    )
    assert verdict["status"] == "NOT_RUN"
    assert verdict["status"] != "PASS"
    assert any("insufficient safety/monitoring evidence" in r for r in verdict["reasons"])


def test_valid_typed_refusal_is_not_run(tiny_store) -> None:
    base, spec, store, _seed = tiny_store
    result_path = base / "refused.json"
    run = rb.run_child(
        rb._case_argv(
            "measure", spec, store, result=str(result_path), available_bytes=1
        ),
        log_path=base / "refused.log",
        timeout_s=60,
    )
    assert run["returncode"] == 0
    result = _read_result(result_path)
    assert result["status"] == "REFUSED"
    assert result["refusal"]["typed"] is True
    assert result["refusal"]["payloads_loaded"] == 0
    assert result["checks"]["refused_before_payload"] is True
    verdict = rb.classify_measurement(
        run=run,
        result=result,
        store_commit_before=None,
        store_commit_after=None,
        expected_case=spec,
    )
    assert verdict["status"] == "NOT_RUN"
    assert verdict["status"] != "PASS"


def test_spoofed_refusal_is_failure(tiny_measure) -> None:
    _base, spec, _store, run, result = tiny_measure
    spoof = {
        "schema": rb.SCHEMA_MEASURE,
        "status": "REFUSED",
        "refusal": {"typed": True},
        "assembly": {"completed": False, "progressed": False},
        "stats": {"loads": 2, "verifications": 1, "live_mappings": 1},
        "checks": {"refused_before_payload": False},
    }
    verdict = rb.classify_measurement(
        run=_good_run_from(run),
        result=spoof,
        store_commit_before=None,
        store_commit_after=None,
        expected_case=spec,
    )
    assert verdict["status"] == "FAIL"
    assert verdict["status"] != "NOT_RUN"


def test_timeout_is_failure_not_pass(tmp_path: Path) -> None:
    run = rb.run_child(
        [sys.executable, str(Path(rb.__file__)), "selftest-sleep", "--seconds", "5"],
        log_path=tmp_path / "sleep.log",
        timeout_s=0.4,
        sample_interval_s=0.02,
    )
    assert run["timed_out"] is True
    assert run["returncode"] != 0
    assert run["elapsed_s"] < 3.0
    verdict = rb.classify_measurement(
        run=run, result=None, store_commit_before=None, store_commit_after=None
    )
    assert verdict["status"] == "FAIL"


def test_child_rss_cap_kills_only_the_owned_child(tmp_path: Path) -> None:
    run = rb.run_child(
        [
            sys.executable,
            str(Path(rb.__file__)),
            "selftest-grow",
            "--bytes",
            str(32 * 2 ** 20),
            "--seconds",
            "3",
        ],
        log_path=tmp_path / "grow.log",
        timeout_s=10,
        rss_cap_bytes=16 * 2 ** 20,
    )
    assert run["guard_breach"] is not None
    assert "CHILD_RSS_CAP" in run["guard_breach"]
    assert run["returncode"] != 0
    assert run["elapsed_s"] < 3.0


def test_failing_child_with_pass_json_is_failure(tmp_path: Path) -> None:
    result_path = tmp_path / "fail.json"
    run = rb.run_child(
        [sys.executable, str(Path(rb.__file__)), "selftest-fail", "--result", str(result_path)],
        log_path=tmp_path / "fail.log",
        timeout_s=10,
    )
    assert run["returncode"] == 2
    assert _read_result(result_path)["status"] == "PASS"
    verdict = rb.classify_measurement(
        run=run,
        result=_read_result(result_path),
        store_commit_before=None,
        store_commit_after=None,
    )
    assert verdict["status"] == "FAIL"


def test_normal_exit_race_does_not_false_fail(tmp_path: Path) -> None:
    for index in range(4):
        run = rb.run_child(
            [sys.executable, "-c", "pass"],
            log_path=tmp_path / f"fast-{index}.log",
            timeout_s=10,
            sample_interval_s=0.02,
        )
        assert run["returncode"] == 0
        assert run["guard_breach"] is None


def test_repeated_missing_telemetry_aborts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        rb,
        "_read_proc_mem",
        lambda _pid: {
            "available": False,
            "rss_bytes": None,
            "hwm_rss_bytes": None,
            "virtual_bytes": None,
            "virtual_peak_bytes": None,
        },
    )
    run = rb.run_child(
        [sys.executable, str(Path(rb.__file__)), "selftest-sleep", "--seconds", "2"],
        log_path=tmp_path / "telemetry.log",
        timeout_s=10,
        sample_interval_s=0.02,
    )
    assert run["guard_breach"] is not None
    assert "TELEMETRY_UNAVAILABLE" in run["guard_breach"]
    assert run["elapsed_s"] < 2.0
    assert run["valid_rss_samples"] == 0


# ---------------------------------------------------------------------------
# Real tiny store: measure + error injection + parity + content immutability
# ---------------------------------------------------------------------------


def test_real_tiny_measurement_is_guarded_and_timed(tiny_measure) -> None:
    _base, spec, store, run, result = tiny_measure
    assert result["status"] == "PASS", result.get("checks")
    assert result["qualified_measurement"] is True
    assert result["admission"]["budget_source"] == "ram_contract.admit_budget_bytes"
    assert result["counters"]["max_resident_cells"] == 2
    assert result["counters"]["peak_live_cells"] == 2
    assert result["counters"]["peak_live_mapped_bytes"] <= result["counters"]["max_live_bytes"]
    assert result["model"]["required_bytes"] <= result["admission"]["available_bytes"]
    assert result["stats_after_close"]["closed"] is True
    assert result["stats_after_close"]["live_cells"] == 0
    assert result["stats_after_close"]["live_mapped_bytes"] == 0
    assert result["checks"]["store_metadata_unchanged"] is True
    assert result["timeline"]["assembly_end_monotonic_s"] < result["timeline"]["close_start_monotonic_s"]
    assert result["timeline"]["close_start_monotonic_s"] <= result["timeline"]["post_close_rss_monotonic_s"]
    assert result["timeline"]["post_close_rss_monotonic_s"] <= result["timeline"]["done_monotonic_s"]
    assert result["assembly"]["post_close_current_rss_bytes"] == (
        result["assembly"]["post_close_snapshot"]["current_rss_bytes"]
    )
    assert result["timings"]["digest_before_close_included_in_lifetime_peak"] is True
    assert result["digests"]["output"] == result["digests"]["output_after_close"]
    assert set(rb.OUTPUT_PLANES) <= set(result["digests"]["output"])
    assert rb._timeline_ordered(result)[0] is True
    assert run["valid_rss_samples"] >= rb.MIN_VALID_SAMPLES
    assert run["valid_host_samples"] >= rb.MIN_VALID_SAMPLES
    assert run["valid_assembly_samples"] >= rb.MIN_VALID_SAMPLES
    assert run["sampled_assembly_peak_rss_bytes"] > 0


def test_tiny_store_is_immutable_after_measurement(tiny_measure) -> None:
    _base, _spec, store, _run, result = tiny_measure
    assert rb.commit_digest(store) == result["digests"]["store_commits_before"]
    assert rb.tree_stat_digest(store) == result["digests"]["store_tree_before"]
    assert rb.store_content_digest(store) == rb.store_content_digest(store)


def test_store_content_digest_detects_payload_mutation(fresh_store) -> None:
    _base, _spec, store = fresh_store
    before = rb.store_content_digest(store)
    objects = sorted((store / "objects").rglob("*"))
    payloads = [path for path in objects if path.is_file()]
    assert payloads
    target = payloads[0]
    os.chmod(target, 0o644)
    with open(target, "r+b") as handle:
        original = handle.read(1)
        handle.seek(0)
        handle.write(b"\x00" if original != b"\x00" else b"\xff")
    after = rb.store_content_digest(store)
    assert before != after


def test_generic_exception_in_measure_is_error(tiny_store) -> None:
    base, spec, store, _seed = tiny_store
    result_path = base / "generic-error.json"
    run = rb.run_child(
        rb._case_argv(
            "measure", spec, store, result=str(result_path), inject_error="generic"
        ),
        log_path=base / "generic-error.log",
        timeout_s=60,
    )
    assert run["returncode"] == 3
    result = _read_result(result_path)
    assert result["status"] == "ERROR"
    assert result["assembly"]["error"]["type"] == "RuntimeError"
    assert "RuntimeError" in result["assembly"]["error"]["traceback"]
    verdict = rb.classify_measurement(
        run=run,
        result=result,
        store_commit_before=None,
        store_commit_after=None,
        expected_case=spec,
    )
    assert verdict["status"] == "FAIL"
    assert verdict["status"] != "NOT_RUN"


def test_post_progress_capacity_refusal_is_error(tiny_store) -> None:
    base, spec, store, _seed = tiny_store
    result_path = base / "post-progress.json"
    run = rb.run_child(
        rb._case_argv(
            "measure",
            spec,
            store,
            result=str(result_path),
            inject_capacity_after_load=True,
        ),
        log_path=base / "post-progress.log",
        timeout_s=60,
    )
    assert run["returncode"] == 4
    result = _read_result(result_path)
    assert result["status"] == "ERROR"
    assert result["assembly"]["progressed"] is True
    assert result["stats"]["loads"] >= 1
    verdict = rb.classify_measurement(
        run=run,
        result=result,
        store_commit_before=None,
        store_commit_after=None,
        expected_case=spec,
    )
    assert verdict["status"] == "FAIL"
    assert verdict["status"] != "NOT_RUN"


def test_parity_separate_process_exact(tiny_store) -> None:
    base, spec, store, _seed = tiny_store
    run = rb.run_child(
        rb._case_argv("parity", spec, store, result=str(base / "parity.json")),
        log_path=base / "parity.log",
        timeout_s=60,
    )
    parity = _read_result(base / "parity.json")
    assert run["returncode"] == 0
    assert parity["status"] == "PARITY_PASS"
    assert parity["mismatches"] == []
    assert parity["bounded"]["digests"] == parity["dense"]["digests"]
    assert parity["bounded"]["offsets"] == parity["dense"]["offsets"]


# ---------------------------------------------------------------------------
# Matrix: sticky failure, unique records, no parity after a stop
# ---------------------------------------------------------------------------


def _matrix_args(base: Path, names: str) -> argparse.Namespace:
    return argparse.Namespace(
        base=str(base), cases=names, include_8x8=False, parity=True
    )


def _fake_run(returncode: int = 0, guard_breach=None, timed_out=False) -> dict:
    return {
        "returncode": int(returncode),
        "guard_breach": guard_breach,
        "timed_out": bool(timed_out),
        "elapsed_s": 0.05,
        "samples": 5,
        "valid_rss_samples": 5,
        "valid_host_samples": 5,
        "valid_paired_samples": 5,
        "valid_assembly_samples": 3,
        "assembly_window_observed": True,
        "sampled_peak_rss_bytes": 200 * 2 ** 20,
        "sampled_assembly_peak_rss_bytes": 190 * 2 ** 20,
        "sampled_min_available_bytes": rb.MIN_RESERVE_BYTES + 64 * 2 ** 20,
        "telemetry_unavailable_count": 0,
        "phases": [],
        "phase_timeline": [],
        "last_phase": "done",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_matrix_seed_failure_is_sticky_unique_and_skips_parity(monkeypatch, capsys) -> None:
    base = Path(tempfile.mkdtemp(prefix=".m5b-rss-mx1-", dir=str(WORKSPACE_ROOT)))
    cases = [
        {"name": "m1", "canvas": (96, 96), "grid": (2, 2), "halo": 4},
        {"name": "m2", "canvas": (96, 96), "grid": (2, 2), "halo": 4},
        {"name": "m3", "canvas": (96, 96), "grid": (2, 2), "halo": 4},
    ]
    monkeypatch.setattr(rb, "DEFAULT_CASES", cases)
    calls = []

    def fake_run_child(argv, **_kwargs):
        calls.append(argv)
        return _fake_run(returncode=2)

    monkeypatch.setattr(rb, "run_child", fake_run_child)
    try:
        rc = rb.cmd_matrix(_matrix_args(base, "m1,m2,m3"))
        summary = _read_result(next(base.glob("run-*/matrix.json")))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    assert rc == 1
    assert summary["status"] == "FAIL"
    names = [case["case"]["name"] for case in summary["cases"]]
    assert len(names) == 4  # 3 selected + parity report
    assert len(set(names)) == 4
    assert summary["cases"][0]["status"] == "FAIL"
    assert all(case["status"] == "NOT_RUN" for case in summary["cases"][1:3])
    assert summary["cases"][-1]["status"] == "NOT_RUN"
    assert "NOT launched" in " ".join(summary["cases"][-1]["reasons"])
    assert not any("parity" in argv for argv in calls)
    assert len(calls) == 1


def test_matrix_predict_failure_is_fail_and_skips_parity(monkeypatch) -> None:
    base = Path(tempfile.mkdtemp(prefix=".m5b-rss-mx2-", dir=str(WORKSPACE_ROOT)))
    cases = [
        {"name": "p1", "canvas": (96, 96), "grid": (2, 2), "halo": 4},
        {"name": "p2", "canvas": (96, 96), "grid": (2, 2), "halo": 4},
    ]
    monkeypatch.setattr(rb, "DEFAULT_CASES", cases)
    calls = []

    def fake_run_child(argv, **kwargs):
        calls.append(argv)
        verb = argv[2]
        if verb == "seed":
            store = Path(argv[argv.index("--root") + 1])
            store.mkdir(parents=True, exist_ok=True)
            result = Path(argv[argv.index("--result") + 1])
            _write_json(result, {"status": "SEEDED", "store": {}, "producer": {}})
            return _fake_run(0)
        if verb == "predict":
            return _fake_run(returncode=2)
        raise AssertionError(f"unexpected verb {verb}")

    monkeypatch.setattr(rb, "run_child", fake_run_child)
    try:
        rc = rb.cmd_matrix(_matrix_args(base, "p1,p2"))
        summary = _read_result(next(base.glob("run-*/matrix.json")))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    assert rc == 1
    assert summary["status"] == "FAIL"
    assert summary["cases"][0]["status"] == "FAIL"
    assert summary["cases"][1]["status"] == "NOT_RUN"
    parity = summary["cases"][-1]
    assert parity["status"] == "NOT_RUN"
    assert "NOT launched" in " ".join(parity["reasons"])
    assert not any("parity" in argv for argv in calls)


def test_matrix_guard_stop_is_fail_and_skips_parity(monkeypatch) -> None:
    base = Path(tempfile.mkdtemp(prefix=".m5b-rss-mx3-", dir=str(WORKSPACE_ROOT)))
    cases = [
        {"name": "g1", "canvas": (96, 96), "grid": (2, 2), "halo": 4},
        {"name": "g2", "canvas": (96, 96), "grid": (2, 2), "halo": 4},
    ]
    monkeypatch.setattr(rb, "DEFAULT_CASES", cases)
    calls = []

    def fake_run_child(argv, **kwargs):
        calls.append(argv)
        verb = argv[2]
        store = Path(argv[argv.index("--root") + 1])
        result = Path(argv[argv.index("--result") + 1])
        if verb == "seed":
            store.mkdir(parents=True, exist_ok=True)
            _write_json(result, {"status": "SEEDED", "store": {}, "producer": {}})
            return _fake_run(0)
        if verb == "predict":
            _write_json(result, {"status": "PREDICTED", "plan": {"required_bytes": 1}})
            return _fake_run(0)
        if verb == "measure":
            return _fake_run(returncode=137, guard_breach="CHILD_RSS_CAP (test)")
        if verb == "parity":
            raise AssertionError("parity must not launch after a guard stop")
        raise AssertionError(f"unexpected verb {verb}")

    monkeypatch.setattr(rb, "run_child", fake_run_child)
    try:
        rc = rb.cmd_matrix(_matrix_args(base, "g1,g2"))
        summary = _read_result(next(base.glob("run-*/matrix.json")))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    assert rc == 1
    assert summary["status"] == "FAIL"
    assert summary["cases"][0]["status"] == "FAIL"
    assert "guard breach" in " ".join(summary["cases"][0]["reasons"])
    assert summary["cases"][1]["status"] == "NOT_RUN"
    parity = summary["cases"][-1]
    assert parity["status"] == "NOT_RUN"
    assert "NOT launched" in " ".join(parity["reasons"])
    assert "FAIL preserved" in " ".join(parity["reasons"])
    assert not any("parity" in argv for argv in calls)


# ---------------------------------------------------------------------------
# REWORK-2: parity must validate its seed before launching the parity child
# ---------------------------------------------------------------------------


def _install_mocked_success_case(monkeypatch, calls, state, parity_seed, parity_run=None):
    """Reach the parity branch after one mocked passing measured case."""

    def fake_classify(**_kwargs):
        return {
            "status": "PASS",
            "reasons": [],
            "result": {
                "preassembly": {},
                "assembly": {},
                "baseline_imports": {},
                "model": {},
                "counters": {},
            },
        }

    def fake_run_child(argv, **_kwargs):
        calls.append(argv)
        verb = argv[2]
        store = Path(argv[argv.index("--root") + 1])
        result = Path(argv[argv.index("--result") + 1])
        is_parity = rb.PARITY_CASE["name"] in str(store)
        if verb == "seed":
            store.mkdir(parents=True, exist_ok=True)
            if is_parity:
                return parity_seed(result)
            _write_json(result, {"status": "SEEDED"})
            return _fake_run(0)
        if verb == "predict":
            _write_json(result, {"status": "PREDICTED", "plan": {"required_bytes": 1}})
            return _fake_run(0)
        if verb == "measure":
            _write_json(result, {"status": "PASS"})
            return _fake_run(0)
        if verb == "parity":
            state["parity_called"] = True
            _write_json(result, {"status": "PARITY_PASS", "mismatches": []})
            return parity_run if parity_run is not None else _fake_run(0)
        raise AssertionError(f"unexpected verb {verb}")

    monkeypatch.setattr(rb, "classify_measurement", fake_classify)
    monkeypatch.setattr(rb, "run_child", fake_run_child)


@pytest.mark.parametrize(
    "seed_run,seed_payload,expected_reason",
    [
        (_fake_run(137, guard_breach="HOST_RESERVE"), {"status": "SEEDED"}, "guard breach"),
        (_fake_run(0, timed_out=True), {"status": "SEEDED"}, "timed out"),
        (_fake_run(0), None, "missing/invalid SEEDED"),
        (_fake_run(0), {"status": "WRONG"}, "missing/invalid SEEDED"),
    ],
)
def test_parity_seed_failure_never_launches_parity(
    monkeypatch, seed_run, seed_payload, expected_reason
):
    base = Path(tempfile.mkdtemp(prefix=".m5b-rss-parity-stop-", dir=str(WORKSPACE_ROOT)))
    cases = [{"name": "q1", "canvas": (96, 96), "grid": (2, 2), "halo": 4}]
    monkeypatch.setattr(rb, "DEFAULT_CASES", cases)
    calls = []
    state = {"parity_called": False}

    def parity_seed(result):
        if seed_payload is not None:
            _write_json(result, seed_payload)
        return seed_run

    _install_mocked_success_case(monkeypatch, calls, state, parity_seed)
    try:
        rc = rb.cmd_matrix(_matrix_args(base, "q1"))
        summary = _read_result(next(base.glob("run-*/matrix.json")))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    assert rc == 1
    assert summary["status"] == "FAIL"
    assert summary["cases"][0]["status"] == "PASS"
    parity = summary["cases"][-1]
    assert parity["status"] == "FAIL"
    assert expected_reason in " ".join(parity["reasons"])
    assert parity.get("parity_launched") is False
    assert state["parity_called"] is False
    assert not any("parity" in argv for argv in calls)


def test_parity_success_path_still_launches(monkeypatch):
    base = Path(tempfile.mkdtemp(prefix=".m5b-rss-parity-ok-", dir=str(WORKSPACE_ROOT)))
    cases = [{"name": "s1", "canvas": (96, 96), "grid": (2, 2), "halo": 4}]
    monkeypatch.setattr(rb, "DEFAULT_CASES", cases)
    calls = []
    state = {"parity_called": False}

    def parity_seed(result):
        _write_json(result, {"status": "SEEDED"})
        return _fake_run(0)

    _install_mocked_success_case(monkeypatch, calls, state, parity_seed)
    try:
        rc = rb.cmd_matrix(_matrix_args(base, "s1"))
        summary = _read_result(next(base.glob("run-*/matrix.json")))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    assert rc == 0
    assert summary["status"] == "PASS"
    parity = summary["cases"][-1]
    assert parity["status"] == "PASS"
    assert parity.get("parity_launched") is True
    assert state["parity_called"] is True
    assert any("parity" in argv for argv in calls)


def test_parity_seed_deadline_exhaustion_is_not_run_not_fail(monkeypatch):
    base = Path(tempfile.mkdtemp(prefix=".m5b-rss-parity-dl-", dir=str(WORKSPACE_ROOT)))
    cases = [{"name": "d1", "canvas": (96, 96), "grid": (2, 2), "halo": 4}]
    monkeypatch.setattr(rb, "DEFAULT_CASES", cases)
    calls = []
    state = {"parity_called": False}
    fake_now = [1000.0]

    def fake_monotonic():
        return fake_now[0]

    def parity_seed(result):
        _write_json(result, {"status": "SEEDED"})
        fake_now[0] += 1000.0  # exhaust the real 300 s budget after seeding
        return _fake_run(0)

    monkeypatch.setattr(rb.time, "monotonic", fake_monotonic)
    _install_mocked_success_case(monkeypatch, calls, state, parity_seed)
    try:
        rc = rb.cmd_matrix(_matrix_args(base, "d1"))
        summary = _read_result(next(base.glob("run-*/matrix.json")))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    assert rc == 2
    assert summary["status"] == "PARTIAL"
    parity = summary["cases"][-1]
    assert parity["status"] == "NOT_RUN"
    assert "deadline exhausted after parity seed" in " ".join(parity["reasons"])
    assert parity.get("parity_launched") is False
    assert state["parity_called"] is False
    assert not any("parity" in argv for argv in calls)


@pytest.mark.parametrize(
    "parity_run,expected_reason",
    [
        (_fake_run(0, timed_out=True), "timed out"),
        (_fake_run(0, guard_breach="HOST_RESERVE"), "guard breach"),
    ],
)
def test_parity_result_flags_fail_even_with_pass_json(
    monkeypatch, parity_run, expected_reason
):
    base = Path(tempfile.mkdtemp(prefix=".m5b-rss-parity-run-", dir=str(WORKSPACE_ROOT)))
    cases = [{"name": "r1c", "canvas": (96, 96), "grid": (2, 2), "halo": 4}]
    monkeypatch.setattr(rb, "DEFAULT_CASES", cases)
    calls = []
    state = {"parity_called": False}

    def parity_seed(result):
        _write_json(result, {"status": "SEEDED"})
        return _fake_run(0)

    _install_mocked_success_case(
        monkeypatch, calls, state, parity_seed, parity_run=parity_run
    )
    try:
        rc = rb.cmd_matrix(_matrix_args(base, "r1c"))
        summary = _read_result(next(base.glob("run-*/matrix.json")))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    assert rc == 1
    assert summary["status"] == "FAIL"
    parity = summary["cases"][-1]
    assert parity["status"] == "FAIL"
    assert expected_reason in " ".join(parity["reasons"])
    assert parity.get("parity_launched") is True
    assert state["parity_called"] is True
