"""Bounded M2 bench: durable run store (source tree only, fail-closed).

Runs the M2 suites from an explicit source worktree into a NEW directory under
an authorised ``zgprobe`` root. It never imports or edits an installed ZeMosaic
runtime and never writes into an existing/symlink/outside path.

Suites (all must run on the REAL OS primitives of the host):
* ``tests/test_zegrid_run_store_m2.py``   — store contract (identity, atomic
  publication, reconstructible index, lock, integrity, disk budget, sparse
  completeness, release hand-off);
* ``tests/test_zegrid_run_store_m2_kill.py`` — kill matrix: a child process is
  killed during each of the four write steps, 5 iterations each;
* ``tests/test_zegrid_resume_m2_store_run.py`` — real-orchestration store seam
  (RAM-gated; below the corpus floor it is NOT_RUN, never PASS);
* M1 non-regression: ``tests/test_zegrid_run_registry_m1.py`` and
  ``tests/test_zegrid_resume_m1_conservation.py``.

Verdict discipline: any pytest failure, timeout, artifact-cap breach, missing
suite or SKIP makes the bench NOT PASS. ``status=PASS`` requires zero failures,
zero errors and zero skips. Exit codes: 0 PASS / 2 NOT_RUN / 3 FAIL /
4 TIMEOUT / 5 CAP_EXCEEDED / 6 GUARD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SCHEMA = "zemosaic-zegrid-resume-m2-bench-v1"
MISSION = "ZM-ZEGRID-RESILIENT-RUN-RESUME-M2-20261027"
DEFAULT_ZGPROBE_ROOT = r"C:\Users\JarvisJunior\Desktop\zgprobe"
DEFAULT_MAX_SECONDS = 1800
DEFAULT_MAX_ARTIFACTS_BYTES = 1 * 2 ** 30
MIN_SECONDS = 60
MAX_SECONDS = 3600

SUITES = (
    ("m2_store_contract", "tests/test_zegrid_run_store_m2.py"),
    ("m2_io_regression", "tests/test_zegrid_run_store_m2_io.py"),
    ("m2_kill_matrix", "tests/test_zegrid_run_store_m2_kill.py"),
    ("m2_real_store_run", "tests/test_zegrid_resume_m2_store_run.py"),
    ("m1_registry", "tests/test_zegrid_run_registry_m1.py"),
    ("m1_conservation", "tests/test_zegrid_resume_m1_conservation.py"),
    ("m0_witness", "tests/test_zegrid_resume_m0_baseline_witness.py"),
    ("m0_harness", "tests/test_zegrid_resume_m0_windows_harness.py"),
)

#: Files the bench fingerprints before/after to prove it never edited sources.
PROVENANCE_FILES = (
    "src/zemosaic/core/zegrid/run_store.py",
    "src/zemosaic/core/zegrid/run_registry.py",
    "src/zemosaic/zemosaic_zegrid_mode.py",
    "tests/test_zegrid_run_store_m2.py",
    "tests/test_zegrid_run_store_m2_io.py",
    "tests/test_zegrid_run_store_m2_kill.py",
    "tests/test_zegrid_resume_m2_store_run.py",
)

REQUIRED_KILL_TESTS = 4 * 5  # 4 write steps x 5 iterations
MIN_STORE_CONTRACT_TESTS = 35


class BenchGuardError(RuntimeError):
    """A guard refused the bench before/after running pytest."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_output_root(output_root: Path, zgprobe_root: Path) -> Path:
    out = Path(output_root)
    root = Path(zgprobe_root)
    if not out.is_absolute():
        raise BenchGuardError("output root must be absolute")
    if not root.is_absolute():
        raise BenchGuardError("zgprobe root must be absolute")
    if out.is_symlink() or root.is_symlink():
        raise BenchGuardError("output/zgprobe root must not be a symlink")
    if not _is_within(out, root):
        raise BenchGuardError(
            f"output root {out} is outside the authorised zgprobe root {root}"
        )
    if out.resolve() == root.resolve():
        raise BenchGuardError("output root must be a NEW child of zgprobe")
    if out.exists():
        raise BenchGuardError(f"output root already exists: {out}")
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(root: Path) -> dict:
    out = {}
    for rel in PROVENANCE_FILES:
        path = root / rel
        out[rel] = sha256_file(path) if path.exists() else None
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
    def __init__(self, root: Path, cap: int, interval: float = 0.25):
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

    def stop(self) -> bool:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.max_seen = max(self.max_seen, directory_bytes(self.root))
        if self.max_seen > self.cap:
            self.exceeded = True
        return self.exceeded


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":  # pragma: no cover - Windows path
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=20,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_pytest(
    suite: str,
    test_path: str,
    *,
    cwd: Path,
    log_path: Path,
    max_seconds: float,
) -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-rs",
        test_path,
    ]
    started = time.time()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"),
        )
        timed_out = False
        try:
            proc.wait(timeout=max(1.0, float(max_seconds)))
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
    elapsed = time.time() - started
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    passed = skipped = failed = errors = 0
    match = re.search(
        r"(\d+) passed", text
    )
    if match:
        passed = int(match.group(1))
    for pattern, setter in (
        (r"(\d+) skipped", "skipped"),
        (r"(\d+) failed", "failed"),
        (r"(\d+) error", "errors"),
    ):
        found = re.search(pattern, text)
        if found:
            if setter == "skipped":
                skipped = int(found.group(1))
            elif setter == "failed":
                failed = int(found.group(1))
            else:
                errors = int(found.group(1))
    return {
        "suite": suite,
        "test_path": test_path,
        "command": " ".join(cmd),
        "returncode": int(proc.returncode) if proc.returncode is not None else None,
        "timed_out": bool(timed_out),
        "elapsed_s": round(elapsed, 3),
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
        "log": str(log_path),
    }


def decide(results: list[dict], *, cap_exceeded: bool) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if cap_exceeded:
        return "CAP_EXCEEDED", ["artifact cap exceeded"]
    if any(r["timed_out"] for r in results):
        return "TIMEOUT", [
            f"timeout in {r['suite']}" for r in results if r["timed_out"]
        ]
    if any(r["returncode"] not in (0,) for r in results):
        return "FAIL", [
            f"{r['suite']} exit={r['returncode']} failed={r['failed']} "
            f"errors={r['errors']}"
            for r in results
            if r["returncode"] not in (0,)
        ]
    if any(r["skipped"] for r in results):
        return "NOT_RUN", [
            f"{r['suite']} skipped {r['skipped']} test(s)"
            for r in results
            if r["skipped"]
        ]
    by_name = {r["suite"]: r for r in results}
    kill = by_name.get("m2_kill_matrix", {})
    if int(kill.get("passed", 0)) < REQUIRED_KILL_TESTS:
        return "NOT_RUN", [
            f"kill matrix executed {kill.get('passed', 0)}/{REQUIRED_KILL_TESTS} "
            "cases"
        ]
    contract = by_name.get("m2_store_contract", {})
    if int(contract.get("passed", 0)) < MIN_STORE_CONTRACT_TESTS:
        return "NOT_RUN", [
            f"store contract suite executed {contract.get('passed', 0)} "
            f"< {MIN_STORE_CONTRACT_TESTS} cases"
        ]
    return "PASS", reasons


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--zgprobe-root", default=DEFAULT_ZGPROBE_ROOT)
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    parser.add_argument(
        "--max-artifacts-bytes", type=int, default=DEFAULT_MAX_ARTIFACTS_BYTES
    )
    args = parser.parse_args(argv)

    source = Path(args.source_root).resolve()
    try:
        if not (source / "src" / "zemosaic" / "core" / "zegrid" / "run_store.py").exists():
            raise BenchGuardError(f"M2 store module missing from {source}")
        if not (source / "tests" / "test_zegrid_run_store_m2_kill.py").exists():
            raise BenchGuardError(f"M2 kill suite missing from {source}")
        output = validate_output_root(
            Path(args.output_root), Path(args.zgprobe_root)
        )
        max_seconds = int(args.max_seconds)
        if not (MIN_SECONDS <= max_seconds <= MAX_SECONDS):
            raise BenchGuardError(
                f"--max-seconds must be within [{MIN_SECONDS}, {MAX_SECONDS}]"
            )
        if int(args.max_artifacts_bytes) <= 0:
            raise BenchGuardError("--max-artifacts-bytes must be positive")
    except BenchGuardError as exc:
        print(f"GUARD: {exc}", file=sys.stderr)
        return 6

    output.mkdir(parents=True, exist_ok=False)
    before = provenance(source)
    watcher = _SizeWatcher(output, int(args.max_artifacts_bytes))
    watcher.start()
    summary_path = output / "probe_summary.json"
    results: list[dict] = []
    deadline = time.time() + max_seconds
    try:
        for suite, rel in SUITES:
            remaining = deadline - time.time()
            if remaining <= 5:
                results.append(
                    {
                        "suite": suite,
                        "test_path": rel,
                        "command": None,
                        "returncode": None,
                        "timed_out": True,
                        "elapsed_s": 0.0,
                        "passed": 0,
                        "skipped": 0,
                        "failed": 0,
                        "errors": 0,
                        "log": None,
                    }
                )
                break
            results.append(
                run_pytest(
                    suite,
                    rel,
                    cwd=source,
                    log_path=output / f"{suite}.log",
                    max_seconds=remaining,
                )
            )
        cap_exceeded = watcher.stop()
    finally:
        watcher.stop()

    # The bench must not have mutated the source tree
    after = provenance(source)
    if before != after:
        status, reasons = "FAIL", [
            "source tree changed while the bench was running (provenance "
            "mismatch)"
        ]
    else:
        status, reasons = decide(results, cap_exceeded=cap_exceeded)

    summary = {
        "schema": SCHEMA,
        "mission": MISSION,
        "status": status,
        "exit_code": {
            "PASS": 0,
            "NOT_RUN": 2,
            "FAIL": 3,
            "TIMEOUT": 4,
            "CAP_EXCEEDED": 5,
        }[status],
        "reasons": reasons,
        "source_root": str(source),
        "output_root": str(output),
        "python": sys.executable,
        "platform": {
            "os": os.name,
            "sys_platform": sys.platform,
            "machine": os.uname().machine if hasattr(os, "uname") else None,
            "release": os.uname().release if hasattr(os, "uname") else None,
        },
        "provenance_before": before,
        "provenance_after": after,
        "artifacts_bytes": directory_bytes(output),
        "max_artifacts_bytes": int(args.max_artifacts_bytes),
        "max_seen_bytes": int(watcher.max_seen),
        "suites": results,
        "totals": {
            "passed": sum(r["passed"] for r in results),
            "failed": sum(r["failed"] for r in results),
            "errors": sum(r["errors"] for r in results),
            "skipped": sum(r["skipped"] for r in results),
        },
        "notes": [
            "A skipped case is NOT_RUN, never qualified.",
            "Kill matrix uses real SIGKILL (POSIX) or TerminateProcess (Windows).",
            "No GPU science/parity claim and no installed-runtime claim.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({k: summary[k] for k in ("status", "totals", "reasons")}))
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
