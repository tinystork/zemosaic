"""ZM-ZEGRID-RESILIENT-RUN-RESUME M4 — NEW-PROCESS kill/resume proof.

Each case spawns REAL subprocesses on a REAL durable store:

* baseline: an uninterrupted run;
* kill: a new process commits K cells and is terminated with SIGKILL semantics
  (``os._exit(9)``) without any cleanup/assembly;
* resume: a DIFFERENT process opens the store, validates content identity,
  restores the stored policy, plans the sparse pending set, and publishes a
  byte-identical product without recomputing any valid cell.

The producer is a deterministic synthetic plane generator (no FITS/GPU); the
store, the registry validation, the sparse plan and the feathered assembly are
the product code. GPU/FITS/RAM are NOT_RUN, explicitly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "tools" / "zegrid_resume_m4" / "m4_resume_harness.py"
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import run_resume as zres  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

CELL_IDS = ("r0000c0000", "r0000c0001", "r0001c0000", "r0001c0001")


def call(
    tmp: Path,
    *,
    mode: str,
    store: Path,
    run_id: str,
    out: Path,
    log: Path,
    kill_after: int = 0,
    policy: str = "dense",
) -> subprocess.CompletedProcess:
    args = [
        sys.executable, str(HARNESS),
        "--mode", mode,
        "--store", str(store),
        "--input-dir", str(tmp / "inputs"),
        "--out", str(out),
        "--summary", str(out.with_suffix(".summary.json")),
        "--executed-log", str(log),
        "--run-id", run_id,
        "--policy", policy,
    ]
    if kill_after:
        args += ["--kill-after", str(kill_after)]
    return subprocess.run(args, capture_output=True, text=True, timeout=180)


def events(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def executes(log: Path) -> list[str]:
    return [e["cell_id"] for e in events(log) if e["event"] == "execute"]


def summary(out: Path) -> dict:
    return json.loads(out.with_suffix(".summary.json").read_text())


@pytest.fixture
def base(tmp_path):
    """An uninterrupted baseline run used as the byte-identity reference."""
    out = tmp_path / "baseline.json"
    done = call(
        tmp_path, mode="new", store=tmp_path / "baseline-store",
        run_id="run-baseline", out=out, log=tmp_path / "baseline.log",
    )
    assert done.returncode == 0, done.stderr
    return summary(out)


def test_m4_kill_then_new_process_resume_is_byte_identical(base, tmp_path):
    store = tmp_path / "store"
    log = tmp_path / "exec.log"
    killed = call(
        tmp_path, mode="new", store=store, run_id="run-kill",
        out=tmp_path / "killed.json", log=log, kill_after=2,
    )
    assert killed.returncode == 9, (killed.returncode, killed.stderr)
    assert executes(log) == ["r0000c0000", "r0000c0001"]

    resumed = call(
        tmp_path, mode="resume", store=store, run_id="run-kill",
        out=tmp_path / "resumed.json", log=log,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = summary(tmp_path / "resumed.json")
    assert result["product_sha256"] == base["product_sha256"]
    assert (tmp_path / "resumed.json").read_bytes() == (
        tmp_path / "baseline.json"
    ).read_bytes()
    assert result["plan"]["pending"] == ["r0001c0000", "r0001c0001"]
    assert result["plan"]["complete"] == 2
    assert result["plan"]["is_complete"] is False

    e = events(log)
    per_cell_exec = {cid: e.count for cid in CELL_IDS}
    for cid in CELL_IDS:
        per_cell_exec[cid] = sum(1 for item in e
                                 if item["event"] == "execute" and item["cell_id"] == cid)
        assert per_cell_exec[cid] == 1, (cid, per_cell_exec[cid])
    # The resumed process executed ONLY the missing cells.
    resumed_cells = [
        item["cell_id"] for item in e
        if item["event"] == "execute" and item["process"] == "resume"
    ]
    assert resumed_cells == ["r0001c0000", "r0001c0001"]
    # REWORK-1: the expensive phases ran ONCE, in the first process only.
    phases = [item for item in e if item["event"] == "phase"]
    executed_phases = {
        (item["phase"], item["process"])
        for item in phases
        if item["phase"] in ("gauge_compute", "layout_compute")
    }
    assert executed_phases == {
        ("gauge_compute", "new"), ("layout_compute", "new")
    }
    resumed_phases = {item["phase"] for item in phases if item["process"] == "resume"}
    assert resumed_phases == {"gauge_restored", "geometry_restored"}
    assert result["artifacts"]["gauge_restored"] is True
    assert result["artifacts"]["geometry_restored"] is True


def test_m4_sparse_corruption_invalidates_one_cell_only(base, tmp_path):
    store = tmp_path / "store"
    log = tmp_path / "exec.log"
    killed = call(
        tmp_path, mode="new", store=store, run_id="run-sparse",
        out=tmp_path / "killed.json", log=log, kill_after=3,
    )
    assert killed.returncode == 9
    # Corrupt the middle committed cell (r0000c0001), never the others.
    opened = zrs.RunCellStore.open(store, identity=None, read_only=True)
    try:
        record = opened.commit_record("r0000c0001")
        plane = store / record.manifest["planes"][0]["path"]
    finally:
        opened.close()
    os.chmod(plane, 0o600)
    plane.write_bytes(b"corrupted-isolated-cell")

    resumed = call(
        tmp_path, mode="resume", store=store, run_id="run-sparse",
        out=tmp_path / "resumed.json", log=log,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = summary(tmp_path / "resumed.json")
    assert result["plan"]["invalid_isolated"] == ["r0000c0001"]
    assert result["plan"]["pending"] == ["r0000c0001", "r0001c0001"]
    assert result["product_sha256"] == base["product_sha256"]
    resumed_cells = [
        item["cell_id"] for item in events(log)
        if item["event"] == "execute" and item["process"] == "resume"
    ]
    assert resumed_cells == ["r0000c0001", "r0001c0001"]
    # Valid cells r0000c0000 / r0001c0000 were NEVER re-executed.
    assert "r0000c0000" not in resumed_cells
    assert "r0001c0000" not in resumed_cells


def test_m4_auto_incompatible_stops_explicitly(base, tmp_path):
    store = tmp_path / "store"
    log = tmp_path / "exec.log"
    killed = call(
        tmp_path, mode="new", store=store, run_id="run-bad",
        out=tmp_path / "killed.json", log=log, kill_after=2,
    )
    assert killed.returncode == 9
    before = zrs.RunCellStore.open(store, identity=None, read_only=True)
    try:
        commits_before = before.report()["counts"]["cells"]
    finally:
        before.close()
    (tmp_path / "inputs" / "frame_0001.fits").write_bytes(b"mutated-input")

    resumed = call(
        tmp_path, mode="resume", store=store, run_id="run-bad",
        out=tmp_path / "resumed.json", log=log,
    )
    assert resumed.returncode == 3
    refusal = json.loads((tmp_path / "resumed.json").read_text())
    assert refusal["status"] == "refused"
    assert "ResumeIncompatible" in refusal["error"]
    assert "input" in refusal["error"].lower() or "identity" in refusal["error"].lower()
    after = zrs.RunCellStore.open(store, identity=None, read_only=True)
    try:
        assert after.report()["counts"]["cells"] == commits_before
    finally:
        after.close()
    assert executes(log) == ["r0000c0000", "r0000c0001"]


def test_m4_complete_run_is_a_verified_noop(base, tmp_path):
    store = tmp_path / "store"
    log = tmp_path / "exec.log"
    full = call(
        tmp_path, mode="new", store=store, run_id="run-complete",
        out=tmp_path / "full.json", log=log,
    )
    assert full.returncode == 0
    before = len(executes(log))
    assert before == 4

    resumed = call(
        tmp_path, mode="resume", store=store, run_id="run-complete",
        out=tmp_path / "noop.json", log=log,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert len(executes(log)) == before, "a complete run must execute no cell"
    result = summary(tmp_path / "noop.json")
    assert result["plan"]["is_complete"] is True
    assert result["product_sha256"] == base["product_sha256"]
    state = zres.read_state(store)
    assert state["finalized"] is True


def test_m4_never_creates_a_new_generation_without_overwriting(base, tmp_path):
    store = tmp_path / "store"
    first = call(
        tmp_path, mode="new", store=store, run_id="run-first",
        out=tmp_path / "first.json", log=tmp_path / "a.log", kill_after=2,
    )
    assert first.returncode == 9
    roots_before = {
        p for p in [store, *[c for c in store.iterdir() if c.is_dir()]]
        if (p / "run.json").exists()
    }
    assert roots_before, "the first run must exist"
    old_root = next(iter(roots_before))
    old_state = zres.read_state(old_root)
    old_commits = len(list((old_root / "commits").glob("*.json")))

    second = call(
        tmp_path, mode="new", store=store, run_id="run-second",
        out=tmp_path / "second.json", log=tmp_path / "b.log",
    )
    assert second.returncode == 0, second.stderr
    result = summary(tmp_path / "second.json")
    assert Path(result["store_root"]) != old_root
    assert zres.read_state(old_root)["run_id"] == old_state["run_id"]
    assert len(list((old_root / "commits").glob("*.json"))) == old_commits
    assert result["product_sha256"] == base["product_sha256"]


def test_m4_multiple_runs_require_explicit_selection(tmp_path):
    for index, run_id in enumerate(("run-one", "run-two")):
        done = call(
            tmp_path, mode="new", store=tmp_path / "store", run_id=run_id,
            out=tmp_path / f"s{index}.json", log=tmp_path / f"l{index}.log",
            kill_after=1,
        )
        assert done.returncode == 9
    ambiguous = call(
        tmp_path, mode="resume", store=tmp_path / "store", run_id="",
        out=tmp_path / "ambiguous.json", log=tmp_path / "x.log",
    )
    assert ambiguous.returncode == 4
    payload = json.loads((tmp_path / "ambiguous.json").read_text())
    assert payload["status"] == "selection_required"
    assert {c["run_id"] for c in payload["candidates"]} == {"run-one", "run-two"}


def test_m4_corrupt_gauge_artifact_refuses_without_recompute(tmp_path):
    store = tmp_path / "store"
    log = tmp_path / "exec.log"
    killed = call(
        tmp_path, mode="new", store=store, run_id="run-gauge-bad",
        out=tmp_path / "killed.json", log=log, kill_after=2,
    )
    assert killed.returncode == 9
    path = store / "gauge.json"
    doc = json.loads(path.read_text())
    doc["coefficients"][0][0][0] = 42.0
    path.write_text(json.dumps(doc))
    before = [e for e in events(log) if e["event"] == "execute"]

    resumed = call(
        tmp_path, mode="resume", store=store, run_id="run-gauge-bad",
        out=tmp_path / "refused.json", log=log,
    )
    assert resumed.returncode == 3
    refusal = json.loads((tmp_path / "refused.json").read_text())
    assert refusal["status"] == "refused"
    assert "gauge" in refusal["error"].lower()
    assert [e for e in events(log) if e["event"] == "execute"] == before
    assert not any(
        e.get("phase") in ("gauge_compute", "layout_compute")
        and e.get("process") == "resume"
        for e in events(log)
    )


def test_m4_restores_bounded_policy_after_restart(tmp_path):
    store = tmp_path / "store"
    log = tmp_path / "exec.log"
    killed = call(
        tmp_path, mode="new", store=store, run_id="run-policy",
        out=tmp_path / "killed.json", log=log, kill_after=2,
        policy="bounded",
    )
    assert killed.returncode == 9
    resumed = call(
        tmp_path, mode="resume", store=store, run_id="run-policy",
        out=tmp_path / "resumed.json", log=log,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = summary(tmp_path / "resumed.json")
    assert result["policy"]["effective"] == "bounded"
    assert result["policy"]["fallback_count"] == 1
    assert result["policy"]["requested"] == "auto"
