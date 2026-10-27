"""M1 REWORK-1 — fail-closed guard tests for the bounded Windows probe.

These run on Linux; they exercise the path/artifact/provenance guards without
requiring Windows or deploying anything.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "tools" / "zegrid_resume_m1" / "m1_windows_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("_m1_probe_guard", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


PROBE = _load_probe()


def test_output_root_must_be_new_and_inside_zgprobe(tmp_path):
    zgprobe = tmp_path / "zgprobe"
    zgprobe.mkdir()
    ok = zgprobe / "m1_probe_new"
    assert PROBE.validate_output_root(ok, zgprobe) == ok

    existing = zgprobe / "existing"
    existing.mkdir()
    with pytest.raises(PROBE.ProbeGuardError):
        PROBE.validate_output_root(existing, zgprobe)

    outside = tmp_path / "outside"
    with pytest.raises(PROBE.ProbeGuardError):
        PROBE.validate_output_root(outside, zgprobe)

    with pytest.raises(PROBE.ProbeGuardError):
        PROBE.validate_output_root(zgprobe, zgprobe)

    with pytest.raises(PROBE.ProbeGuardError):
        PROBE.validate_output_root(Path("relative"), zgprobe)


def test_output_root_symlink_is_refused(tmp_path):
    zgprobe = tmp_path / "zgprobe"
    zgprobe.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    link = zgprobe / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(PROBE.ProbeGuardError):
        PROBE.validate_output_root(link, zgprobe)


def test_directory_bytes_and_watcher_cap(tmp_path):
    root = tmp_path / "out"
    root.mkdir()
    (root / "a.bin").write_bytes(b"x" * 1000)
    assert PROBE.directory_bytes(root) == 1000
    watcher = PROBE._SizeWatcher(root, cap=1500, interval=0.01)
    watcher.start()
    (root / "b.bin").write_bytes(b"y" * 1000)
    # Poll briefly for the watcher to observe the growth.
    for _ in range(100):
        if watcher.exceeded:
            break
        import time

        time.sleep(0.01)
    assert watcher.stop() is True
    watcher2 = PROBE._SizeWatcher(root, cap=10 ** 9, interval=0.01)
    watcher2.start()
    assert watcher2.stop() is False


def test_source_digest_detects_mutation(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    before = PROBE.sha256_file(path)
    path.write_text("VALUE = 2\n", encoding="utf-8")
    after = PROBE.sha256_file(path)
    assert before != after


def test_main_refuses_missing_source_before_creating_output(tmp_path, capsys):
    zgprobe = tmp_path / "zgprobe"
    zgprobe.mkdir()
    out = zgprobe / "probe"
    code = PROBE.main(
        [
            "--source-root",
            str(tmp_path / "does-not-exist"),
            "--output-root",
            str(out),
            "--zgprobe-root",
            str(zgprobe),
        ]
    )
    assert code == 2
    assert not out.exists()
    captured = capsys.readouterr().out
    assert "NOT_RUN" in captured
    assert "source_missing" in captured


def test_main_refuses_existing_output_and_preserves_files(tmp_path, capsys):
    zgprobe = tmp_path / "zgprobe"
    zgprobe.mkdir()
    out = zgprobe / "probe"
    out.mkdir()
    sentinel = out / "probe_summary.json"
    sentinel.write_text("ORIGINAL\n", encoding="utf-8")
    code = PROBE.main(
        [
            "--source-root",
            str(tmp_path),
            "--output-root",
            str(out),
            "--zgprobe-root",
            str(zgprobe),
        ]
    )
    assert code == 2
    captured = capsys.readouterr().out
    assert "invalid_output_root" in captured
    # A rejected path must not be modified or have its summary overwritten.
    assert sentinel.read_text(encoding="utf-8") == "ORIGINAL\n"
    assert sorted(p.name for p in out.iterdir()) == ["probe_summary.json"]


def test_main_refuses_outside_path_without_writing(tmp_path, capsys):
    zgprobe = tmp_path / "zgprobe"
    zgprobe.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "probe_summary.json"
    sentinel.write_text("ORIGINAL\n", encoding="utf-8")
    code = PROBE.main(
        [
            "--source-root",
            str(tmp_path),
            "--output-root",
            str(outside),
            "--zgprobe-root",
            str(zgprobe),
        ]
    )
    assert code == 2
    assert "invalid_output_root" in capsys.readouterr().out
    assert sentinel.read_text(encoding="utf-8") == "ORIGINAL\n"


def test_run_bounded_kills_on_artifact_cap(tmp_path):
    out = tmp_path / "capout"
    out.mkdir()
    script = (
        "import sys,time\n"
        "f=open(sys.argv[1],'wb')\n"
        "for _ in range(400):\n"
        "    f.write(b'x'*65536); f.flush(); time.sleep(0.01)\n"
    )
    outcome = PROBE.run_bounded(
        [sys.executable, "-c", script, str(out / "big.bin")],
        cwd=tmp_path,
        env=dict(os.environ),
        log_path=out / "log.txt",
        err_path=out / "err.txt",
        max_seconds=30,
        max_artifacts_bytes=200_000,
        artifact_root=out,
    )
    assert outcome["cap_exceeded"] is True
    assert outcome["timed_out"] is False
    assert outcome["returncode"] != 0
    assert outcome["elapsed_s"] < 20
    assert outcome["max_artifact_bytes_seen"] > 200_000


def test_run_bounded_kills_on_timeout(tmp_path):
    out = tmp_path / "timeoutout"
    out.mkdir()
    outcome = PROBE.run_bounded(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        cwd=tmp_path,
        env=dict(os.environ),
        log_path=out / "log.txt",
        err_path=out / "err.txt",
        max_seconds=0.5,
        max_artifacts_bytes=10 ** 9,
        artifact_root=out,
    )
    assert outcome["timed_out"] is True
    assert outcome["cap_exceeded"] is False
    assert outcome["returncode"] != 0
    assert outcome["elapsed_s"] < 5


def test_run_bounded_success(tmp_path):
    out = tmp_path / "okout"
    out.mkdir()
    outcome = PROBE.run_bounded(
        [sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        env=dict(os.environ),
        log_path=out / "log.txt",
        err_path=out / "err.txt",
        max_seconds=10,
        max_artifacts_bytes=10 ** 9,
        artifact_root=out,
    )
    assert outcome["returncode"] == 0
    assert outcome["timed_out"] is False
    assert outcome["cap_exceeded"] is False


def _verdict(outcome, parsed, *, missing=None, registry=20):
    return PROBE.classify_probe_outcome(
        outcome,
        parsed,
        required_missing=[] if missing is None else missing,
        registry_passed=registry,
    )


def _all_pass_parsed():
    return {
        "passed": [
            f"{PROBE.CONSERVATION_TEST}::{name}"
            for name in PROBE.REQUIRED_CONSERVATION
        ]
        + [f"{PROBE.REGISTRY_TEST}::test_{i:02d}" for i in range(20)],
        "failed": [],
        "skipped": [],
        "errors": [],
    }


def test_probe_verdict_precedence_matrix():
    ok = _all_pass_parsed()
    # All six required + unit floor -> PASS.
    assert _verdict({"returncode": 0}, ok)["status"] == "PASS"
    # Failed + skipped must be FAIL, never NOT_RUN.
    failed_skip = dict(ok, failed=["x::broken"], skipped=["x::skipped"])
    assert _verdict({"returncode": 0}, failed_skip)["status"] == "FAIL"
    # Error + skipped must be FAIL.
    err_skip = dict(ok, errors=["x::error"], skipped=["x::skipped"])
    assert _verdict({"returncode": 0}, err_skip)["status"] == "FAIL"
    # Nonzero return is FAIL even when every printed line says PASSED.
    assert _verdict({"returncode": 1}, ok)["status"] == "FAIL"
    # Zero return + printed FAILED is contradictory and still FAIL.
    contradictory = dict(ok, failed=["x::contradiction"])
    assert _verdict({"returncode": 0}, contradictory)["status"] == "FAIL"
    # Required case absent -> FAIL (not PASS/NOT_RUN).
    assert _verdict(
        {"returncode": 0}, ok, missing=[PROBE.REQUIRED_TEST_IDS[0]]
    )["status"] == "FAIL"
    # Mandatory skip (no failure) -> NOT_RUN.
    skipped = dict(ok, skipped=[PROBE.REQUIRED_TEST_IDS[0]])
    assert _verdict({"returncode": 0}, skipped)["status"] == "NOT_RUN"
    # Cap/timeouts are unsuccessful terminals, even with failures/skips.
    assert _verdict(
        {"returncode": 0, "cap_exceeded": True}, failed_skip
    )["status"] == "CAP_EXCEEDED"
    assert _verdict(
        {"returncode": 0, "timed_out": True}, skipped
    )["status"] == "TIMEOUT"


def test_probe_required_identity_is_exact_not_substring():
    required = PROBE.REQUIRED_TEST_IDS[0]
    parsed = _all_pass_parsed()
    # Replace the exact required ID with a near-substring superset.
    parsed["passed"] = [
        t for t in parsed["passed"] if required not in t
    ] + [f"{required}_extra"]
    missing = [
        rid
        for rid in PROBE.REQUIRED_TEST_IDS
        if rid not in {PROBE.normalize_test_id(t) for t in parsed["passed"]}
    ]
    assert required in missing
    assert _verdict(
        {"returncode": 0}, parsed, missing=missing
    )["status"] == "FAIL"
    assert PROBE.normalize_test_id(f"/abs/tests/{required}") == required
