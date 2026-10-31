"""ZM-ZEGRID-COMMIT-INTEGRITY r4 — REAL full-product ACTIVE GPU integration.

The GPU durable commit is proven through the REAL product entry
(``run_zegrid_mode`` -> ACTIVE GPU orchestration -> ``BoundedPipeline`` ->
``_consume_guarded`` -> real ``RunCellStore``/``RunResultRegistry``).

SIMULATION, stated honestly: the CUDA KERNEL is not executed. Only the LOWER
cell computation functions ``_run_cell_inmem``/``_run_cell_stream`` are wrapped
to run the REAL CPU computation with ``dataclasses.replace(config,
backend="cpu")``. GPU availability/VRAM observations are tiny fake hardware
facts (the existing r55e2 fixture). Everything else — planning, decide/admit,
the pipeline, the consumer, registry acceptance, durable commit/verification,
release, coverage and the outer wrapper — is the REAL product code.
"""

from __future__ import annotations

import dataclasses
import json
import os
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


def _run(tmp_path, monkeypatch, *, lines=None, corpus=None, resume_mode="never"):
    R55._fake_gpu_available(monkeypatch)
    _cpu_math(monkeypatch)
    corpus = corpus or _corpus(tmp_path)
    out = tmp_path / "out"
    store = out / "zegrid-durable"
    cb = (lambda *a, **k: lines.append(str(a[0]))) if lines is not None else None
    zz.run_zegrid_mode(
        str(corpus), str(out), cb,
        zconfig=_zconfig(store, resume_mode),
        zegrid_halo_px=8,
        use_gpu=True,
    )
    return out, store


def _commits(store: Path):
    return sorted((store / "commits").glob("*.json"))


def test_full_product_gpu_run_commits_every_expected_cell(tmp_path, monkeypatch):
    spy = []
    real_commit = zz._gpu_durable_commit

    def wrapped(state, registry, cid, receipt, **kw):
        out = real_commit(state, registry, cid, receipt, **kw)
        spy.append((str(cid), getattr(receipt, "attempt_id", None), out))
        return out

    monkeypatch.setattr(zz, "_gpu_durable_commit", wrapped)
    lines: list[str] = []
    out, store = _run(tmp_path, monkeypatch, lines=lines)

    # r4: the published manifest lives inside the immutable generation; the
    # visible root marker points at it.
    products = json.loads(
        (out / "zemosaic_products.json").read_text(encoding="utf-8")
    )
    gen_dir = Path(products["generation_dir"])
    manifest = json.loads(
        (gen_dir / "zegrid_manifest.json").read_text(encoding="utf-8")
    )
    block = manifest.get("r55_scheduler") or {}
    assert (block.get("mapping") or {}).get("backend") == "gpu", block
    assert (manifest.get("gpu") or {}).get("actually_used") is True

    commits = _commits(store)
    assert len(commits) == 4, [p.name for p in commits]
    # Every expected cell has a durable commit readable by the reader.
    report = zz.zrs.RunCellStore.open(store, identity=None)
    try:
        for task in manifest.get("cells") or []:
            cid = task.get("cell_id") if isinstance(task, dict) else None
            if cid:
                assert report.attempts(str(cid))
    finally:
        report.close()
    assert len(spy) == 4 and all(item[2] for item in spy), spy

    text = "\n".join(lines)
    assert "per-cell stacking complete — durable_commits=4" in text, lines
    log = (out / "zegrid_run.log").read_text(encoding="utf-8")
    assert "per_cell_stack 4/4" in log
    # r4 visible contract still holds.
    assert (out / "zemosaic_products.json").is_file()


def test_full_product_gpu_thrown_commit_failure_is_terminal(tmp_path, monkeypatch):
    R55._fake_gpu_available(monkeypatch)
    _cpu_math(monkeypatch)
    corpus = _corpus(tmp_path)
    out = tmp_path / "out"
    store = out / "zegrid-durable"

    real_commit = zz._gpu_durable_commit
    calls = {"n": 0}

    def injected(self, registry, cell_id, attempt_id=None, backend="cpu", plan_generation=0):
        calls["n"] += 1
        raise zrs.StoreIntegrityError("simulated commit failure")

    monkeypatch.setattr(zrs.RunCellStore, "commit_and_release", injected)
    with pytest.raises(zrs.StoreCommitNotDurable):
        zz.run_zegrid_mode(
            str(corpus), str(out), None,
            zconfig=_zconfig(store),
            zegrid_halo_px=8,
            use_gpu=True,
        )
    assert calls["n"] == 1
    assert not _commits(store)
    log = (out / "zegrid_run.log").read_text(encoding="utf-8")
    assert "per_cell_stack 1/4" not in log
    assert not (out / "zemosaic_products.json").exists()


def test_full_product_gpu_silent_noop_commit_is_terminal(tmp_path, monkeypatch):
    R55._fake_gpu_available(monkeypatch)
    _cpu_math(monkeypatch)
    corpus = _corpus(tmp_path)
    out = tmp_path / "out"
    store = out / "zegrid-durable"

    def noop(self, registry, cell_id, attempt_id=None, backend="cpu", plan_generation=0):
        return {"release": {"released": False}}

    monkeypatch.setattr(zrs.RunCellStore, "commit_and_release", noop)
    with pytest.raises(zrs.StoreCommitNotDurable) as excinfo:
        zz.run_zegrid_mode(
            str(corpus), str(out), None,
            zconfig=_zconfig(store),
            zegrid_halo_px=8,
            use_gpu=True,
        )
    assert excinfo.value.cell_id
    assert not _commits(store)
    log = (out / "zegrid_run.log").read_text(encoding="utf-8")
    assert "per_cell_stack 1/4" not in log
    assert not (out / "zemosaic_products.json").exists()
