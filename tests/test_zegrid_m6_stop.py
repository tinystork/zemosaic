"""ZM-ZEGRID-M6 Part C — real SIGTERM cooperative stop at a wave boundary.

A real CLI subprocess opens a durable store, commits at least one cell, then
receives SIGTERM. The engine finishes the current safe boundary, acknowledges a
durable ``STOPPED_RESUMABLE`` and the CLI exits 3 with:

* every durable commit manifest present and parseable;
* the frozen resume state still ``finalized=false``;
* no publication generation and no ``CURRENT.json`` pointer.

Windows signal semantics are NOT_RUN (documented in the CLI); this test is
POSIX-only and skips explicitly elsewhere.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
CORPUS_TOOL = REPO_ROOT / "tools" / "zegrid_m6" / "make_tiny_corpus.py"

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="POSIX signal semantics (Windows NOT_RUN)"
)


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def _commit_manifests(store: Path) -> list[Path]:
    return sorted(store.rglob("commits/*.json"))


def test_sigterm_stops_resumably_with_intact_durable_commits(tmp_path):
    corpus = tmp_path / "in"
    proc = subprocess.run(
        [sys.executable, str(CORPUS_TOOL), str(corpus)],
        env=_env(), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    out = tmp_path / "out"
    store = tmp_path / "store"
    summary_path = tmp_path / "summary.json"
    stdout_path = tmp_path / "cli.stdout"
    stderr_path = tmp_path / "cli.stderr"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        child = subprocess.Popen(
            [
                sys.executable, "-m", "zemosaic.zegrid_cli",
                "--input", str(corpus),
                "--output", str(out),
                "--workers", "1",
                "--layout", "2x2",
                "--halo-px", "8",
                "--gauge-mode", "global",
                "--durable-store-dir", str(store),
                "--json-summary", str(summary_path),
                "--stop-grace-seconds", "60",
            ],
            env=_env(), cwd=str(REPO_ROOT), stdout=stdout, stderr=stderr,
        )
        try:
            deadline = time.monotonic() + 90.0
            first_commit = None
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    break
                manifests = _commit_manifests(store)
                if manifests:
                    first_commit = manifests[0]
                    break
                time.sleep(0.01)
            assert first_commit is not None, (
                "no durable commit appeared before the run ended; cannot "
                "prove a stop at a boundary. stderr tail:\n"
                + stderr_path.read_text(errors="replace")[-3000:]
            )
            child.send_signal(signal.SIGTERM)
            returncode = child.wait(timeout=120)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=30)

    assert returncode == 3, (
        f"expected exit 3 (cooperative stop), got {returncode}\n"
        + stderr_path.read_text(errors="replace")[-3000:]
    )
    stderr_text = stderr_path.read_text(errors="replace")
    assert "cooperative stop requested" in stderr_text

    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "STOPPED_RESUMABLE"
    assert summary["exit_code"] == 3
    stop = summary["stop"]
    assert stop["requested"] is True
    assert stop["signal"] == int(signal.SIGTERM)
    assert stop["status"] == "stopped_resumable"
    assert stop["durable"] is True
    assert stop["durable_commits_reported"] >= 1
    assert stop["durable_commits_found"] >= 1
    assert stop["durable_commits_valid"] == stop["durable_commits_found"]
    assert stop["publication_started"] is False

    # Every commit manifest is intact and names a real cell/attempt.
    manifests = _commit_manifests(store)
    assert len(manifests) >= 1
    for path in manifests:
        payload = json.loads(path.read_text())
        assert payload.get("cell_id")
        assert payload.get("attempt_id")

    # The frozen state exists and is NOT finalized.
    state_files = list(store.rglob("state.json"))
    assert state_files, "the frozen resume state is missing"
    state = json.loads(state_files[0].read_text())
    assert state["finalized"] is False
    assert state["expected_cell_ids"]

    # No half-written publication: no generation and no atomic pointer.
    assert not list(out.rglob("CURRENT.json"))
    assert not (out / ".zegrid_publications").exists()
