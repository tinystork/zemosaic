"""ZM-ZEGRID-COMMIT-INTEGRITY-20261031 — never report a cell without a commit.

The RESUME_NEW incident: ``per_cell_stack`` advanced 0/20 → 19/20 while
``zegrid-durable/commits/`` stayed EMPTY; the run only died later at assembly
with ``StoreNotFoundError: no durable commit for cell 'r0000c0000'``.

Contracts pinned here:
* ``RunCellStore.require_commit`` raises a typed ``StoreCommitNotDurable``
  (cell, run_id, store root) when a commit is not durable;
* a durable run fails AT THE FIRST CELL when the commit is a silent no-op, and
  the progress counter never advances;
* a durable run with ZERO commits refuses assembly via a typed error naming the
  store root, the expected count and the log-phase count;
* a normal run logs ``durable_commits=0`` at stacking start and
  ``durable_commits=<commits>`` at the end; progress ends at total;
* a resume with N committed starts at N/total and ends with durable_commits=total.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for _path in (SRC, REPO_ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

MIB = 2 ** 20
TINY_CORPUS_TOOL = REPO_ROOT / "tools" / "zegrid_m6" / "make_tiny_corpus.py"
TOOLS_M4 = REPO_ROOT / "tools" / "zegrid_resume_m4"
if str(TOOLS_M4) not in sys.path:
    sys.path.insert(0, str(TOOLS_M4))
import m4_resume_harness as HARNESS  # noqa: E402


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "in"
    if not corpus.exists():
        proc = subprocess.run(
            [sys.executable, str(TINY_CORPUS_TOOL), str(corpus)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
    return corpus


def _config(store: Path, resume_mode: str = "never") -> SimpleNamespace:
    return SimpleNamespace(
        zegrid_scheduler_mode="active",
        zegrid_durable_store_dir=str(store),
        zegrid_resume_mode=str(resume_mode),
        zegrid_resume_run_id=None,
        zegrid_durable_store_budget_bytes=64 * MIB,
        zegrid_layout="2x2",
        zegrid_cell_gauge_mode="global",
        zegrid_resource_recovery_enabled=False,
        zegrid_resource_recovery_deadline_s=None,
        zegrid_sip_mode="keep",
        temporary_work_dir="",
    )


def _run(tmp_path: Path, *, resume_mode="never", lines=None, corpus=None):
    corpus = corpus or _corpus(tmp_path)
    out = tmp_path / "out"
    store = out / "zegrid-durable"
    zz.run_zegrid_mode(
        str(corpus),
        str(out),
        (lambda *arts, **_kw: lines.append(str(arts[0]))) if lines is not None else None,
        zconfig=_config(store, resume_mode),
        zegrid_halo_px=8,
        use_gpu=False,
    )
    return out, store


def _m4_store(tmp_path: Path):
    """A real (empty) durable store built with the M4 harness identity."""
    rows = HARNESS.ensure_inputs(tmp_path / "in")
    identity = HARNESS.make_identity("commit-integrity", rows)
    root = tmp_path / "m4-store"
    store = zrs.RunCellStore.open(
        root,
        identity=identity,
        budget=zrs.DiskBudget(durable_cell_bytes=64 * MIB),
        session_id="commit-integrity",
    )
    return store, identity


def test_require_commit_names_cell_run_and_store(tmp_path):
    store, _identity = _m4_store(tmp_path)
    try:
        with pytest.raises(zrs.StoreCommitNotDurable) as excinfo:
            store.require_commit("r0000c0000", "attempt-x")
        exc = excinfo.value
        assert exc.cell_id == "r0000c0000"
        assert exc.run_id == "commit-integrity"
        assert str(store.root) == exc.store_root
        assert exc.attempt_id == "attempt-x"
        assert "NO durable commit" in str(exc)

        registry = zrr.RunResultRegistry(
            run_id="commit-integrity",
            expected_cell_ids=list(HARNESS.CELL_IDS),
            gauge_mode="per_cell",
        )
        try:
            payload = HARNESS.compute_cell(0)
            receipt = registry.accept(
                payload, backend="cpu", plan_generation=0, attempt_id="ok"
            )
            store.commit_and_release(
                registry,
                HARNESS.CELL_IDS[0],
                attempt_id=receipt.attempt_id,
                backend="cpu",
                plan_generation=0,
            )
        finally:
            registry.dispose()
        record = store.require_commit(HARNESS.CELL_IDS[0], "ok")
        assert record.cell_id == HARNESS.CELL_IDS[0]
    finally:
        store.close()


def test_silent_commit_fails_at_the_first_cell(tmp_path, monkeypatch):
    def _silent(self, registry, cell_id, attempt_id=None, backend="cpu", plan_generation=0):
        return {"released": False}

    monkeypatch.setattr(zrs.RunCellStore, "commit_and_release", _silent)
    out = tmp_path / "out"
    with pytest.raises(zrs.StoreCommitNotDurable) as excinfo:
        _run(tmp_path)
    exc = excinfo.value
    assert exc.cell_id
    assert exc.run_id
    assert exc.store_root
    log = (out / "zegrid_run.log").read_text(encoding="utf-8")
    # The visible counter never advanced: no cell was ever reported processed.
    assert "per_cell_stack 1/4" not in log
    assert "phase PROGRESS: per_cell_stack" not in log
    # The store stayed empty: nothing durable was ever claimed.
    assert not list((out / "zegrid-durable" / "commits").glob("*.json"))


def test_zero_commit_store_refuses_assembly(tmp_path):
    store, _identity = _m4_store(tmp_path)
    try:
        with pytest.raises(zrs.StoreCommitNotDurable) as excinfo:
            zz._require_durable_commits_before_assembly(
                store, expected_cells=4, log_done=3
            )
        message = str(excinfo.value)
        assert "ZERO commits" in message
        assert str(store.root) in message
        assert "4 cells were expected" in message
        assert "reported done=3" in message
    finally:
        store.close()


def test_helper_returns_the_durable_count_when_commits_exist(tmp_path):
    store, _identity = _m4_store(tmp_path)
    registry = zrr.RunResultRegistry(
        run_id="commit-integrity",
        expected_cell_ids=list(HARNESS.CELL_IDS),
        gauge_mode="per_cell",
    )
    try:
        payload = HARNESS.compute_cell(0)
        receipt = registry.accept(
            payload, backend="cpu", plan_generation=0, attempt_id="one"
        )
        store.commit_and_release(
            registry,
            HARNESS.CELL_IDS[0],
            attempt_id=receipt.attempt_id,
            backend="cpu",
            plan_generation=0,
        )
        assert (
            zz._require_durable_commits_before_assembly(
                store, expected_cells=4, log_done=1
            )
            == 1
        )
    finally:
        registry.dispose()
        store.close()


def test_normal_run_reports_durable_commits_and_completes(tmp_path):
    lines: list[str] = []
    out, _store = _run(tmp_path, lines=lines)
    text = "\n".join(lines)
    assert "stacking start — durable_commits=0" in text, lines
    assert "per-cell stacking complete — durable_commits=4" in text, lines
    log = (out / "zegrid_run.log").read_text(encoding="utf-8")
    assert "per_cell_stack 4/4" in log
    commits = list((out / "zegrid-durable" / "commits").glob("*.json"))
    assert len(commits) == 4


def test_resume_reports_the_committed_count_then_total(tmp_path):
    out, store = _run(tmp_path)
    corpus = tmp_path / "in"
    commits = sorted((store / "commits").glob("*.json"))
    assert len(commits) == 4
    for path in commits[2:]:
        path.unlink()
    lines: list[str] = []
    _run(tmp_path, resume_mode="auto", lines=lines, corpus=corpus)
    text = "\n".join(lines)
    assert "stacking start — durable_commits=2" in text, lines
    assert "per-cell stacking complete — durable_commits=4" in text, lines
    log = (out / "zegrid_run.log").read_text(encoding="utf-8")
    progress = [
        line for line in log.splitlines()
        if "phase PROGRESS: per_cell_stack" in line
    ]
    assert progress and "per_cell_stack 2/4" in progress[0], progress
    assert "per_cell_stack 4/4" in log
    assert len(list((store / "commits").glob("*.json"))) == 4
