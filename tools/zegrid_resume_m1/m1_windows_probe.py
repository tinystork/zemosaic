"""Bounded M1 Windows/source-tree probe with fail-closed guards.

Runs only the M1 tests from an explicit source worktree into a NEW directory
under the authorised ``zgprobe`` root. It never touches an installed runtime and
never writes into a rejected/existing/outside/symlink path.

Control-only CPU: the conservation tests clamp their synthetic memory trajectory
to the LIVE available RAM. If the host cannot admit the corpus, the conservation
cases skip and the probe reports NOT_RUN (pytest exit 0 alone is not PASS).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SCHEMA = "zemosaic-zegrid-resume-m1-windows-probe-v1"
MISSION = "ZM-ZEGRID-RESILIENT-RUN-RESUME-M1-20261027"
DEFAULT_ZGPROBE_ROOT = r"C:\Users\JarvisJunior\Desktop\zgprobe"
DEFAULT_MAX_SECONDS = 900
DEFAULT_MAX_ARTIFACTS_BYTES = 1 * 2 ** 30
MIN_SECONDS = 60
MAX_SECONDS = 1200
#: Live-RAM floor below which the real-orchestration conservation cases are
#: not physically admissible on the 2x2 corpus. Registry unit tests still run.
MIN_CONSERVATION_LIVE_BYTES = 1200 * 2 ** 20

REGISTRY_TEST = "test_zegrid_run_registry_m1.py"
CONSERVATION_TEST = "test_zegrid_resume_m1_conservation.py"
REQUIRED_CONSERVATION = (
    "test_m1_refusal_preserves_accepted_payloads_and_sparse_pending_set",
    "test_m1_cooperative_stop_ack_preserves_accepted_cells",
    "test_m1_gpu_oom_after_accepted_cells_is_conserved",
    "test_m1_gpu_internal_acceptance_failure_is_conserved_and_classified",
    "test_m1_cpu_acceptance_failure_preserves_registry_handle",
    "test_m1_mid_wave_generation_transition_rejects_old_results",
)
REQUIRED_TEST_IDS = tuple(
    f"{CONSERVATION_TEST}::{name}" for name in REQUIRED_CONSERVATION
)
REQUIRED_REGISTRY_MIN = 20


class ProbeGuardError(RuntimeError):
    """A guard refused the probe before/after running pytest."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_output_root(output_root: Path, zgprobe_root: Path) -> Path:
    """Require a NEW, non-symlink directory strictly under zgprobe."""
    out = Path(output_root)
    root = Path(zgprobe_root)
    if not out.is_absolute():
        raise ProbeGuardError("output root must be absolute")
    if not root.is_absolute():
        raise ProbeGuardError("zgprobe root must be absolute")
    if out.is_symlink() or root.is_symlink():
        raise ProbeGuardError("output/zgprobe root must not be a symlink")
    if not _is_within(out, root):
        raise ProbeGuardError(
            f"output root {out} is outside the authorised zgprobe root {root}"
        )
    if out.resolve() == root.resolve():
        raise ProbeGuardError("output root must be a NEW child of zgprobe")
    if out.exists():
        raise ProbeGuardError(f"output root already exists: {out}")
    return out


def directory_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += int(path.stat().st_size)
        except OSError:
            continue
    return int(total)


class _SizeWatcher:
    def __init__(self, root: Path, cap: int, interval: float = 0.1):
        self.root = root
        self.cap = int(cap)
        self.interval = float(interval)
        self.exceeded = False
        self.max_seen = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            size = directory_bytes(self.root)
            self.max_seen = max(self.max_seen, size)
            if size > self.cap:
                self.exceeded = True
                self._stop.set()
                return

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.max_seen = max(self.max_seen, directory_bytes(self.root))
        if self.max_seen > self.cap:
            self.exceeded = True
        return self.exceeded


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _live_available_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        if os.name == "nt":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
            return None
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=15,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_bounded(
    cmd,
    *,
    cwd: Path,
    env,
    log_path: Path,
    err_path: Path,
    max_seconds: float,
    max_artifacts_bytes: int,
    artifact_root: Path,
):
    """Run ``cmd`` with a hard deadline and a LIVE artifact cap.

    The watcher and the deadline both kill the owned process tree immediately;
    they are not post-hoc labels. Returns a dict with the terminal facts.
    """
    started = time.monotonic()
    deadline = started + max(0.05, float(max_seconds))
    stdout_fh = open(log_path, "w", encoding="utf-8")
    stderr_fh = open(err_path, "w", encoding="utf-8")
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=stdout_fh,
            stderr=stderr_fh,
            start_new_session=(os.name != "nt"),
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
        )
        watcher = _SizeWatcher(artifact_root, max_artifacts_bytes)
        watcher.start()
        timed_out = False
        cap_exceeded = False
        while proc.poll() is None:
            if watcher.exceeded:
                cap_exceeded = True
                _kill_tree(proc)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                _kill_tree(proc)
                break
            time.sleep(0.05)
        try:
            proc.wait(timeout=15)
        except Exception:
            _kill_tree(proc)
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        cap_exceeded = watcher.stop() or cap_exceeded
        return {
            "returncode": proc.returncode,
            "timed_out": bool(timed_out),
            "cap_exceeded": bool(cap_exceeded),
            "max_artifact_bytes_seen": int(watcher.max_seen),
            "artifact_bytes": directory_bytes(artifact_root),
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    finally:
        if proc is not None and proc.poll() is None:
            _kill_tree(proc)
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        stdout_fh.close()
        stderr_fh.close()


def _parse_pytest_results(text: str) -> dict:
    passed, failed, skipped, errors = [], [], [], []
    for line in text.splitlines():
        tokens = line.strip().split()
        if not tokens:
            continue
        test_id = tokens[0]
        if "::" not in test_id:
            continue
        if "PASSED" in tokens:
            passed.append(test_id)
        elif "FAILED" in tokens:
            failed.append(test_id)
        elif "SKIPPED" in tokens:
            skipped.append(test_id)
        elif "ERROR" in tokens:
            errors.append(test_id)
    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
    }


def normalize_test_id(test_id: str) -> str:
    """Reduce an absolute pytest node id to ``file.py::test_name``."""
    text = str(test_id).replace("\\", "/")
    return text.rsplit("/", 1)[-1]


def classify_probe_outcome(
    outcome: dict,
    parsed: dict,
    *,
    required_missing,
    registry_passed: int,
    min_registry: int = REQUIRED_REGISTRY_MIN,
) -> dict:
    """Pure terminal verdict with strict failure-before-skip precedence.

    cap/timeout/nonzero return/failed/errors are unsuccessful terminals and are
    evaluated BEFORE skips. A zero exit with a printed FAILED is contradictory
    and still FAILS. Skips are NOT_RUN only when the run itself did not fail.
    """
    if outcome.get("cap_exceeded"):
        return {
            "status": "CAP_EXCEEDED",
            "code": 5,
            "reason": "artifact_cap",
            "detail": "artifact cap exceeded",
        }
    if outcome.get("timed_out"):
        return {
            "status": "TIMEOUT",
            "code": 4,
            "reason": "timeout",
            "detail": "pytest deadline exceeded",
        }
    if int(outcome.get("returncode", 1)) != 0:
        return {
            "status": "FAIL",
            "code": 3,
            "reason": "pytest_failed",
            "detail": f"pytest exit {outcome.get('returncode')}",
        }
    if parsed.get("failed") or parsed.get("errors"):
        return {
            "status": "FAIL",
            "code": 3,
            "reason": "pytest_reported_failure",
            "detail": (
                f"failed={list(parsed.get('failed') or [])}, "
                f"errors={list(parsed.get('errors') or [])}"
            ),
        }
    if parsed.get("skipped"):
        return {
            "status": "NOT_RUN",
            "code": 2,
            "reason": "conservation_not_exercised",
            "detail": f"skipped={list(parsed.get('skipped') or [])}",
        }
    required_missing = list(required_missing or [])
    if required_missing or int(registry_passed) < int(min_registry):
        return {
            "status": "FAIL",
            "code": 3,
            "reason": "required_cases_not_exercised",
            "detail": (
                f"missing={required_missing}, "
                f"registry_passed={int(registry_passed)} < {int(min_registry)}"
            ),
        }
    return {
        "status": "PASS",
        "code": 0,
        "reason": None,
        "detail": "M1 registry + six conservation cases passed",
    }


def _write_summary(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--zgprobe-root", default=DEFAULT_ZGPROBE_ROOT)
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    parser.add_argument(
        "--max-artifacts-bytes", type=int, default=DEFAULT_MAX_ARTIFACTS_BYTES
    )
    args = parser.parse_args(argv)

    started = time.monotonic()
    source = Path(args.source_root).expanduser().resolve()
    out = Path(args.output_root).expanduser()
    python_exe = Path(args.python_exe).expanduser()
    max_seconds = max(MIN_SECONDS, min(MAX_SECONDS, int(args.max_seconds)))
    max_bytes = max(1, int(args.max_artifacts_bytes))
    summary = {
        "schema": SCHEMA,
        "mission": MISSION,
        "mode": "WINDOWS_M1_CONTROL_ONLY_CPU",
        "gpu_qualified": False,
        "installed_runtime_used": False,
        "provenance": "SOURCE_TREE_ONLY_NOT_VERIFIED_INSTALLED_RUNTIME",
        "status": "NOT_RUN",
        "source_root": str(source),
        "output_root": str(out),
        "python_exe": str(python_exe),
        "zgprobe_root": str(args.zgprobe_root),
        "timeout_seconds": int(max_seconds),
        "max_artifacts_bytes": int(max_bytes),
        "started_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)
        ),
        "limits": [
            "Control-only CPU: GPU device executor replaced by the CPU canonical executor.",
            "No installed-runtime comparison or edits.",
            "In-memory conservation only; no durable save/restart claim.",
            "Conservation cases clamp observations to live RAM; skips are NOT_RUN, never PASS.",
            "Missing deps/nonzero/timeout/cap can never yield PASS.",
        ],
    }
    owns_output = False

    def finish(status: str, code: int, detail: str, reason=None):
        summary["status"] = status
        summary["detail"] = detail
        summary["reason"] = reason
        summary["exit_code"] = int(code)
        summary["finished_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        summary["duration_s"] = round(time.monotonic() - started, 3)
        summary["artifact_bytes"] = (
            directory_bytes(out) if owns_output and out.exists() else 0
        )
        # Only write when THIS invocation validated and created the directory.
        if owns_output and out.exists():
            _write_summary(out / "probe_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return int(code)

    try:
        validate_output_root(out, Path(args.zgprobe_root))
    except ProbeGuardError as exc:
        return finish("NOT_RUN", 2, str(exc), "invalid_output_root")

    if not source.is_dir():
        return finish("NOT_RUN", 2, "source root missing", "source_missing")
    tests = [
        source / "tests" / REGISTRY_TEST,
        source / "tests" / CONSERVATION_TEST,
    ]
    modules = [
        source / "src" / "zemosaic" / "zemosaic_zegrid_mode.py",
        source / "src" / "zemosaic" / "core" / "zegrid" / "run_registry.py",
    ]
    if not all(p.is_file() for p in tests + modules):
        return finish(
            "NOT_RUN", 2, "M1 test/product files missing", "source_incomplete"
        )
    if not python_exe.is_file():
        return finish(
            "NOT_RUN",
            2,
            f"python executable missing: {python_exe}",
            "python_missing",
        )

    live = _live_available_bytes()
    summary["live_available_bytes_preflight"] = live
    summary["min_conservation_live_bytes"] = int(
        MIN_CONSERVATION_LIVE_BYTES
    )
    if live is None:
        return finish(
            "NOT_RUN",
            2,
            "live available RAM observation failed",
            "live_ram_unavailable",
        )

    out.mkdir(parents=True, exist_ok=False)
    owns_output = True
    log = out / "pytest.log"
    err = out / "pytest.err.log"
    basetemp = out / "tmp"
    summary["pytest_log"] = str(log)
    summary["pytest_err_log"] = str(err)
    summary["basetemp"] = str(basetemp)
    summary["source_module_sha256_before"] = {
        str(p.relative_to(source)): sha256_file(p) for p in modules
    }

    env = dict(os.environ)
    env["PYTHONPATH"] = str(source / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["ZEMOSAIC_M1_LIVE_AVAILABLE_BYTES"] = str(int(live))

    def remaining():
        return max(0.0, max_seconds - (time.monotonic() - started))

    try:
        dep = subprocess.run(
            [
                str(python_exe),
                "-c",
                "import pytest,numpy,astropy,psutil; print('DEPS_OK')",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=min(60.0, remaining()),
        )
    except subprocess.TimeoutExpired:
        return finish(
            "TIMEOUT", 4, "dependency preflight exceeded the deadline", "timeout"
        )
    if dep.returncode != 0 or "DEPS_OK" not in dep.stdout:
        return finish(
            "NOT_RUN",
            2,
            f"dependencies unavailable: {dep.stderr or dep.stdout}",
            "dependencies_missing",
        )

    try:
        provenance = subprocess.run(
            [
                str(python_exe),
                "-c",
                "import json, zemosaic.zemosaic_zegrid_mode as z, "
                "zemosaic.core.zegrid.run_registry as r; "
                "print(json.dumps({'mode': z.__file__, 'registry': r.__file__}))",
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(source),
            timeout=min(120.0, remaining()),
        )
    except subprocess.TimeoutExpired:
        return finish(
            "TIMEOUT", 4, "provenance preflight exceeded the deadline", "timeout"
        )
    if provenance.returncode != 0:
        return finish(
            "NOT_RUN",
            2,
            f"source module import failed: {provenance.stderr}",
            "source_import_failed",
        )
    try:
        paths = json.loads(provenance.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return finish(
            "NOT_RUN", 2, f"provenance parse failed: {exc}", "provenance_failed"
        )
    if not _is_within(Path(paths["mode"]), source) or not _is_within(
        Path(paths["registry"]), source
    ):
        return finish(
            "FAIL",
            3,
            f"imported modules are outside the source tree: {paths}",
            "provenance_outside_source",
        )
    summary["actual_module_paths"] = paths

    outcome = run_bounded(
        [
            str(python_exe),
            "-m",
            "pytest",
            "-v",
            "-p",
            "no:cacheprovider",
            "--tb=short",
            "-rs",
            "--basetemp",
            str(basetemp),
            str(tests[0]),
            str(tests[1]),
        ],
        cwd=source,
        env=env,
        log_path=log,
        err_path=err,
        max_seconds=remaining(),
        max_artifacts_bytes=max_bytes,
        artifact_root=out,
    )
    summary["pytest_outcome"] = dict(outcome)

    after = {str(p.relative_to(source)): sha256_file(p) for p in modules}
    summary["source_module_sha256_after"] = after
    if after != summary["source_module_sha256_before"]:
        return finish("FAIL", 3, "source module hashes changed", "source_mutated")
    if outcome["cap_exceeded"]:
        return finish(
            "CAP_EXCEEDED",
            5,
            f"artifact cap exceeded ({outcome['max_artifact_bytes_seen']} > {max_bytes})",
            "artifact_cap",
        )
    if outcome["timed_out"]:
        return finish(
            "TIMEOUT", 4, f"pytest exceeded {max_seconds}s", "timeout"
        )

    log_text = ""
    try:
        log_text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    parsed = _parse_pytest_results(log_text)
    summary["pytest_results"] = {
        key: list(value) for key, value in parsed.items()
    }
    passed_norm = {
        normalize_test_id(test_id) for test_id in parsed["passed"]
    }
    required_missing = [
        test_id for test_id in REQUIRED_TEST_IDS if test_id not in passed_norm
    ]
    registry_passed = sum(
        1
        for test_id in parsed["passed"]
        if normalize_test_id(test_id).startswith(f"{REGISTRY_TEST}::")
    )
    summary["registry_tests_passed"] = int(registry_passed)
    summary["required_conservation_missing"] = list(required_missing)
    verdict = classify_probe_outcome(
        outcome,
        parsed,
        required_missing=required_missing,
        registry_passed=registry_passed,
    )
    summary["verdict"] = dict(verdict)
    return finish(
        verdict["status"],
        int(verdict["code"]),
        str(verdict["detail"]),
        verdict["reason"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
