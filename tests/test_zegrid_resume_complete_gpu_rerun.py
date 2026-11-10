"""ZM-RESUME-COMPLETE-GPU-RERUN — a COMPLETE resume executes ZERO cell stacks.

Production evidence (``crashed before finalization``): a GPU-frozen run whose
20 expected Cells were ALL durably committed reported ``per_cell_stack 20/20``
and then ``21/20`` because the resume admitted the generic ``gpu_used`` batch
after legitimately clearing the ACTIVE GPU decision for an empty pending set.

These tests drive the REAL product entry (``run_zegrid_mode``) over a synthetic
tiny corpus and a REAL ``RunCellStore``. Only the CUDA kernel is simulated: the
lower cell functions run the REAL CPU math under the GPU config, exactly like
``test_zegrid_gpu_durable_integration``. A COMPLETE non-finalized resume must
execute zero stacks, reuse every immutable commit through the bounded
store-backed assembly and publish from there.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import run_publication as zpub  # noqa: E402
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

import test_zegrid_r55e2_active_gpu_prefetch as R55  # noqa: E402

MIB = 2 ** 20
TINY_CORPUS_TOOL = REPO_ROOT / "tools" / "zegrid_m6" / "make_tiny_corpus.py"


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


def _zconfig(store: Path, resume_mode="never") -> SimpleNamespace:
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


def _cpu_math(monkeypatch):
    """SIMULATED GPU KERNEL: real CPU math under the GPU config (no CUDA)."""
    real_inmem = zz._run_cell_inmem
    real_stream = zz._run_cell_stream

    def inmem(cache_dir, patch, config, progress_callback=None, **kw):
        return real_inmem(
            cache_dir,
            patch,
            dataclasses.replace(config, backend="cpu"),
            progress_callback,
            **kw,
        )

    def stream(cache_dir, patch, config, progress_callback=None, **kw):
        return real_stream(
            cache_dir,
            patch,
            dataclasses.replace(config, backend="cpu"),
            progress_callback,
            **kw,
        )

    monkeypatch.setattr(zz, "_run_cell_inmem", inmem)
    monkeypatch.setattr(zz, "_run_cell_stream", stream)
    return [inmem, stream]


def _run(tmp_path, monkeypatch, *, use_gpu, resume_mode="never", corpus=None,
         progress=None):
    if use_gpu:
        R55._fake_gpu_available(monkeypatch)
    corpus = corpus or _corpus(tmp_path)
    out = tmp_path / "out"
    store = out / "zegrid-durable"
    zz.run_zegrid_mode(
        str(corpus), str(out), progress,
        zconfig=_zconfig(store, resume_mode),
        zegrid_halo_px=8,
        use_gpu=bool(use_gpu),
    )
    return out, store


def _commits(store: Path):
    return sorted((store / "commits").glob("*.json"))


def _commit_cid(path: Path) -> str:
    """The cell id of a commit file (``<cell_id>__<attempt-hash>.json``)."""
    return path.stem.split("__")[0]


def _commit_bytes(store: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in _commits(store)}


def _break_finalization(out: Path, store: Path) -> None:
    """Leave the store EXACTLY as a crash before finalization/assembly.

    All Cell commits stay untouched; the durable state loses ``finalized`` and
    the publication (pointer + immutable generations + visible products) is
    removed, so ``verify_finalized`` reports ``no_publication``.
    """
    state = zres.read_state(store)
    zres.write_state(
        store,
        run_id=state["run_id"],
        expected_cell_ids=state["expected_cell_ids"],
        policy=zres.policy_from_snapshot(state.get("policy") or {}),
        plan_generation=int(state["plan_generation"]) + 1,
        finalized=False,
        gauge_sha256=state["gauge_sha256"],
        outputs={},
        created_unix_ns=state["created_unix_ns"],
        note=state.get("note"),
        admission=state.get("admission"),
    )
    for path in (
        out / zpub.POINTER_FILENAME,
        out / zpub.PRODUCTS_JSON_FILENAME,
        out / zpub.PRODUCTS_TEXT_FILENAME,
    ):
        if path.exists():
            path.unlink()
    for name in zpub.PUBLICATION_DIRNAMES:
        target = out / name
        if target.is_dir():
            shutil.rmtree(target)


def _bomb_cell_execution(monkeypatch):
    """Hard-fail every per-cell stack executor (GPU or CPU, any batch size).

    ``_run_cell_batch`` is a closure inside the product entry, so the guard is
    placed on the MODULE-LEVEL cell executors both the GPU pipeline and the
    CPU batch funnel through. Reaching either one is proof that a stack was
    admitted for a COMPLETE resume.
    """
    calls: list[str] = []

    def bomb(*args, **kwargs):
        calls.append("stack")
        raise AssertionError(
            "a COMPLETE resumed run admitted a cell stack executor"
        )

    patched = 0
    for name in (
        "_run_cell_inmem",
        "_run_cell_stream",
        "_run_cell_batch",
        "_run_cell_stack_unit",
    ):
        if hasattr(zz, name):
            monkeypatch.setattr(zz, name, bomb)
            patched += 1
    assert patched >= 2, patched
    return calls


def _spy_commits(monkeypatch):
    """Record every durable commit attempt without altering the contract."""
    calls: list[str] = []
    real = zrs.RunCellStore.commit_and_release

    def spy(self, *args, **kwargs):
        calls.append(str(args[1] if len(args) > 1 else kwargs.get("cell_id")))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(zrs.RunCellStore, "commit_and_release", spy)
    return calls


def _samples(log_text: str, total: int) -> list[int]:
    """Every ``per_cell_stack N/total`` sample, in log order."""
    import re

    pattern = re.compile(r"per_cell_stack\s+(\d+)/(\d+)")
    return [
        int(done)
        for done, denom in pattern.findall(log_text)
        if int(denom) == int(total)
    ]


def _log(out: Path) -> str:
    path = out / "zegrid_run.log"
    assert path.is_file(), sorted(p.name for p in out.iterdir())
    return path.read_text(encoding="utf-8")


def _published_manifest(out: Path) -> dict:
    products = json.loads(
        (out / zpub.PRODUCTS_JSON_FILENAME).read_text(encoding="utf-8")
    )
    gen_dir = Path(products["generation_dir"])
    return json.loads(
        (gen_dir / "zegrid_manifest.json").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# 1 — GPU-frozen COMPLETE non-finalized resume: zero stacks, zero commits
# ---------------------------------------------------------------------------


def test_complete_gpu_frozen_resume_executes_zero_stacks(tmp_path, monkeypatch):
    lines: list[str] = []
    out, store = _run(
        tmp_path, monkeypatch, use_gpu=True,
        progress=lambda msg, *a, **k: lines.append(str(msg)),
    )
    total = len(_commits(store))
    assert total == 4
    before = _commit_bytes(store)
    _break_finalization(out, store)
    assert not (out / zpub.POINTER_FILENAME).exists()

    # --- the resumed attempt must never stack, cache or re-commit a cell -----
    stacks = _bomb_cell_execution(monkeypatch)
    commits = _spy_commits(monkeypatch)
    resumed_lines: list[str] = []
    out2, store2 = _run(
        tmp_path, monkeypatch, use_gpu=True, resume_mode="auto",
        progress=lambda msg, *a, **k: resumed_lines.append(str(msg)),
    )
    assert store2 == store
    assert stacks == [], stacks
    assert commits == [], commits
    assert _commit_bytes(store) == before

    text = "\n".join(resumed_lines)
    assert "ALREADY COMPLETE" in text
    assert "executed_this_attempt=0" in text
    assert "assembly" in text
    assert "done" in text.lower()

    log = _log(out)
    samples = _samples(log, total)
    assert samples, log[-2000:]
    assert max(samples) == total <= total
    assert total + 1 not in samples
    assert f"per_cell_stack {total}/{total}" in log

    manifest = _published_manifest(out)
    gpu = manifest["gpu"]
    assert gpu["actually_used"] is False
    assert gpu["gpu_executed_cells"] == 0
    assert gpu["executed_this_attempt"] == 0
    assert gpu["attempted"] is True
    diag = manifest["per_cell_diagnostics"]
    assert diag["executed_this_attempt"] == 0
    assert diag["reused_durable_cells"] == total
    assert diag["durable_replay"]["source"] == "durable_commits"
    assert diag["r55_resume"]["resumed"] is True
    assert diag["r55_resume"]["pending_cells"] == 0
    # The FROZEN execution contract is untouched: same backend and tile as the
    # admission written before the crash.
    admission = zres.read_state(store)["admission"]
    assert admission["backend"] == "gpu"
    assert int(gpu["gpu_streaming_tile"]) == int(admission["tile_px"])
    # Publication happened from the immutable commits.
    products = json.loads(
        (out / zpub.PRODUCTS_JSON_FILENAME).read_text(encoding="utf-8")
    )
    assert zpub.ROLE_SCIENCE in products["products"]


# ---------------------------------------------------------------------------
# 2 — CPU-frozen COMPLETE non-finalized resume: also zero stacks
# ---------------------------------------------------------------------------


def test_complete_cpu_frozen_resume_executes_zero_stacks(tmp_path, monkeypatch):
    lines: list[str] = []
    out, store = _run(
        tmp_path, monkeypatch, use_gpu=False,
        progress=lambda msg, *a, **k: lines.append(str(msg)),
    )
    total = len(_commits(store))
    assert total == 4
    before = _commit_bytes(store)
    _break_finalization(out, store)

    stacks = _bomb_cell_execution(monkeypatch)
    commits = _spy_commits(monkeypatch)
    resumed_lines: list[str] = []
    _run(
        tmp_path, monkeypatch, use_gpu=False, resume_mode="auto",
        progress=lambda msg, *a, **k: resumed_lines.append(str(msg)),
    )
    assert stacks == [], stacks
    assert commits == [], commits
    assert _commit_bytes(store) == before
    text = "\n".join(resumed_lines)
    assert "executed_this_attempt=0" in text
    log = _log(out)
    samples = _samples(log, total)
    assert max(samples) == total
    assert total + 1 not in samples
    manifest = _published_manifest(out)
    assert manifest["per_cell_diagnostics"]["executed_this_attempt"] == 0
    assert manifest["per_cell_diagnostics"]["reused_durable_cells"] == total


# ---------------------------------------------------------------------------
# 3 — PARTIAL resume still executes ONLY the sparse pending set
# ---------------------------------------------------------------------------


def test_partial_cpu_resume_executes_only_pending_cells(tmp_path, monkeypatch):
    out, store = _run(tmp_path, monkeypatch, use_gpu=False)
    commits = _commits(store)
    assert len(commits) == 4
    removed_ids = [_commit_cid(path) for path in commits[2:]]
    for path in commits[2:]:
        path.unlink()

    new_commits = _spy_commits(monkeypatch)
    resumed_lines: list[str] = []
    zz.run_zegrid_mode(
        str(_corpus(tmp_path)), str(out),
        lambda msg, *a, **k: resumed_lines.append(str(msg)),
        zconfig=_zconfig(store, "auto"), zegrid_halo_px=8, use_gpu=False,
    )
    # ONLY the sparse pending set was executed (no prefix/offset shortcut).
    assert sorted(new_commits) == sorted(removed_ids), (new_commits, removed_ids)
    assert len(_commits(store)) == 4
    assert "2 to stack" in "\n".join(resumed_lines)
    log = _log(out)
    samples = _samples(log, 4)
    assert samples and max(samples) == 4
    assert 5 not in samples
    manifest = _published_manifest(out)
    diag = manifest["per_cell_diagnostics"]
    assert diag["executed_this_attempt"] == 2
    assert diag["reused_durable_cells"] == 2


# ---------------------------------------------------------------------------
# 3b — a GPU-frozen PARTIAL resume keeps its pre-existing fail-closed refusal
# ---------------------------------------------------------------------------


def test_partial_gpu_frozen_resume_keeps_the_existing_refusal(
    tmp_path, monkeypatch
):
    """UNCHANGED policy: a GPU plan is never mixed into a resumed sparse set.

    ``ZM-RESUME-COMPLETE-GPU-RERUN`` must not alter this fail-closed contract:
    the sparse pending set is refused before any cell admission, no commit is
    written and no stack runs.
    """
    out, store = _run(tmp_path, monkeypatch, use_gpu=True)
    assert len(_commits(store)) == 4
    for path in _commits(store)[2:]:
        path.unlink()
    partial = _commit_bytes(store)

    stacks = _bomb_cell_execution(monkeypatch)
    commits_spy = _spy_commits(monkeypatch)
    lines: list[str] = []
    with pytest.raises(RuntimeError) as excinfo:
        _run(
            tmp_path, monkeypatch, use_gpu=True, resume_mode="auto",
            progress=lambda msg, *a, **k: lines.append(str(msg)),
        )
    assert "CPU-frozen" in str(excinfo.value)
    assert stacks == [], stacks
    assert commits_spy == [], commits_spy
    assert _commit_bytes(store) == partial


# ---------------------------------------------------------------------------
# 4 — A fresh run still executes every cell (no accidental skip)
# ---------------------------------------------------------------------------


def test_fresh_run_still_executes_every_cell(tmp_path, monkeypatch):
    lines: list[str] = []
    out, store = _run(
        tmp_path, monkeypatch, use_gpu=True,
        progress=lambda msg, *a, **k: lines.append(str(msg)),
    )
    assert len(_commits(store)) == 4
    log = _log(out)
    samples = _samples(log, 4)
    assert samples and max(samples) == 4
    manifest = _published_manifest(out)
    assert manifest["per_cell_diagnostics"]["executed_this_attempt"] == 4
    assert manifest["per_cell_diagnostics"]["reused_durable_cells"] == 0
    assert manifest["gpu"]["actually_used"] is True


# ---------------------------------------------------------------------------
# 5 — A FINALIZED resume keeps its verified no-op semantics
# ---------------------------------------------------------------------------


def test_finalized_complete_resume_is_a_verified_noop(tmp_path, monkeypatch):
    out, store = _run(tmp_path, monkeypatch, use_gpu=True)
    before = _commit_bytes(store)
    products_before = (out / zpub.PRODUCTS_JSON_FILENAME).read_bytes()

    stacks = _bomb_cell_execution(monkeypatch)
    commits = _spy_commits(monkeypatch)
    lines: list[str] = []
    _run(
        tmp_path, monkeypatch, use_gpu=True, resume_mode="auto",
        progress=lambda msg, *a, **k: lines.append(str(msg)),
    )
    assert stacks == [], stacks
    assert commits == [], commits
    assert _commit_bytes(store) == before
    assert (out / zpub.PRODUCTS_JSON_FILENAME).read_bytes() == products_before
    text = "\n".join(lines)
    assert "verified no-op" in text
    assert "executed_this_attempt=0" not in text
