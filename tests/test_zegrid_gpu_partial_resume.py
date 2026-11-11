"""ZM-ZEGRID-GPU-PARTIAL-RESUME-20261111 — GPU-frozen sparse resume.

Owner-approved policy: a PARTIAL durable resume whose frozen admission is GPU
may continue, provided it stays under the run's OWN frozen backend contract
(same geometry/halo/working set/science/parity; the execution tile is only
ever allowed to shrink at RUN). It must execute ONLY the sparse pending Cells:

* commits are added only for the pending Cells;
* canonical order is preserved (never a prefix/offset shortcut);
* stack progress is ``offset + executed`` over the FROZEN expected total and
  never reports ``N+1/N``.

A GPU plan injected into a CPU-frozen store stays refused, and a GPU-frozen
run with no GPU is a precise typed refusal (never a silent CPU switch): no
in-tree qualified CPU<->GPU numeric parity contract exists for a product
resume, as pinned by ``test_zegrid_active_resume_frozen_identity`` and by the
recorded ``cpu_reuse="refused_no_numeric_parity_qualification"`` diagnostic.

The product entry (``run_zegrid_mode``) runs over the real tiny corpus and a
REAL ``RunCellStore``; only the CUDA kernel is simulated by running the real
CPU math under the GPU config (the same seam used by the complete-resume
suite). No real dataset, no GUI, no science change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import run_publication as zpub  # noqa: E402
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

import test_zegrid_resume_complete_gpu_rerun as RERUN  # noqa: E402
import test_zegrid_r55e2_active_gpu_prefetch as R55  # noqa: E402


def _stack_spy(monkeypatch):
    """Record the Cells the GPU producer really prepares, in call order."""
    seen: list[str] = []
    real = zz._prepare_stack_cell

    def spy(task):
        seen.append(str(task[0].cell_id))
        return real(task)

    monkeypatch.setattr(zz, "_prepare_stack_cell", spy)
    return seen


def _partial_store(tmp_path, monkeypatch, *, use_gpu=True):
    """A real run, then remove two commits: a sparse pending set.

    Returns ``(out, store, removed_ids, retained_bytes, frozen_total)``. The
    retained snapshot is taken AFTER the removal, so it is exactly the set a
    resume must leave byte-for-byte untouched.
    """
    out, store = RERUN._run(tmp_path, monkeypatch, use_gpu=use_gpu)
    commits = RERUN._commits(store)
    frozen_total = len(commits)
    assert frozen_total == 4
    removed = list(commits[2:])
    removed_ids = sorted(RERUN._commit_cid(path) for path in removed)
    for path in removed:
        path.unlink()
    retained = RERUN._commit_bytes(store)
    return out, store, removed_ids, retained, frozen_total


# ---------------------------------------------------------------------------
# 1 — GPU-frozen PARTIAL resume: only pending Cells, canonical order, commits
# ---------------------------------------------------------------------------


def test_gpu_frozen_partial_resume_executes_only_pending_cells(tmp_path, monkeypatch):
    out, store, removed_ids, before, frozen_total = _partial_store(
        tmp_path, monkeypatch, use_gpu=True
    )
    admission = zres.read_state(store)["admission"]
    assert admission["backend"] == "gpu"
    expected_total = frozen_total

    seen = _stack_spy(monkeypatch)
    commits_spy = RERUN._spy_commits(monkeypatch)
    lines: list[str] = []
    RERUN._run(
        tmp_path, monkeypatch, use_gpu=True, resume_mode="auto",
        progress=lambda msg, *a, **k: lines.append(str(msg)),
    )

    # EVERY GPU producer unit sees exactly the pending ids, in canonical order.
    # Canonical order for this 2x2 store is the sorted cell-id order; the
    # pending subset must be executed in that same order.
    expected_pending = sorted(removed_ids)
    assert seen == expected_pending, (seen, expected_pending)
    assert set(seen).isdisjoint(
        {RERUN._commit_cid(Path(name)) for name in before}
    )

    # Commits are added ONLY for the pending Cells; the rest is byte-identical.
    assert sorted(commits_spy) == removed_ids, commits_spy
    after = RERUN._commit_bytes(store)
    for name, payload in before.items():
        if RERUN._commit_cid(Path(name)) in set(removed_ids):
            continue
        assert after[name] == payload, name
    assert len(RERUN._commits(store)) == expected_total

    # The run FINISHES and publishes; the frozen backend contract is reported.
    manifest = RERUN._published_manifest(out)
    gpu = manifest["gpu"]
    assert gpu["actually_used"] is True
    assert gpu["executed_this_attempt"] == len(removed_ids)
    diag = manifest["per_cell_diagnostics"]
    assert diag["executed_this_attempt"] == len(removed_ids)
    assert diag["reused_durable_cells"] == expected_total - len(removed_ids)
    assert int(gpu["gpu_streaming_tile"]) == int(admission["tile_px"])
    # Same-backend resume: no mixed-backend fallback was recorded.
    assert manifest["gpu"]["final_backend"] == "gpu"
    assert "r55_gpu_fallback" not in diag
    products = json.loads(
        (out / zpub.PRODUCTS_JSON_FILENAME).read_text(encoding="utf-8")
    )
    assert zpub.ROLE_SCIENCE in products["products"]


# ---------------------------------------------------------------------------
# 2 — progress is offset+executed over the FROZEN total (never N+1/N)
# ---------------------------------------------------------------------------


def test_gpu_partial_resume_progress_never_exceeds_the_frozen_total(
    tmp_path, monkeypatch
):
    out, store, removed_ids, before, total = _partial_store(
        tmp_path, monkeypatch, use_gpu=True
    )
    RERUN._run(
        tmp_path, monkeypatch, use_gpu=True, resume_mode="auto",
    )
    log = RERUN._log(out)
    samples = RERUN._samples(log, total)
    assert samples, log[-3000:]
    # offset == already committed Cells, so the FIRST resumed sample is the
    # run-level coverage, never 1/total.
    assert samples[0] == total - len(removed_ids), samples
    assert max(samples) == total
    assert total + 1 not in samples
    assert samples == sorted(samples)


# ---------------------------------------------------------------------------
# 3 — a GPU plan injected into a CPU-frozen store is refused, nothing written
# ---------------------------------------------------------------------------


def _refusal_message(frozen, *, gpu_plan_selected=True, pending_count=2):
    return zz._resume_partial_gpu_refusal(
        frozen,
        gpu_plan_selected=gpu_plan_selected,
        pending_count=pending_count,
    )


def test_partial_gpu_guard_predicate_is_exact():
    assert _refusal_message(None, gpu_plan_selected=False) is None
    assert _refusal_message(None, pending_count=0) is None
    assert _refusal_message({"backend": "gpu", "tile_px": 48}) is None
    cpu = _refusal_message({"backend": "cpu", "tile_px": 96})
    assert "CPU-frozen" in cpu
    assert "never injected" in cpu
    assert "mixing backends" in cpu
    missing = _refusal_message(None)
    assert "pre-parity" in missing
    assert "mixing backends" in missing
    unknown = _refusal_message({"backend": "tpu", "tile_px": 96})
    assert "unknown" in unknown


def test_gpu_plan_into_a_cpu_frozen_store_is_refused_before_writing(
    tmp_path, monkeypatch
):
    out, store, removed_ids, before, _total = _partial_store(
        tmp_path, monkeypatch, use_gpu=False
    )
    state = zres.read_state(store)
    assert state["admission"]["backend"] == "cpu"

    # Force a GPU plan into the CPU-frozen store: the planner normally never
    # does this (_frozen_planning_gpu forces CPU), so this is the injected
    # shape the defensive guard must catch.
    R55._fake_gpu_available(monkeypatch)
    monkeypatch.setattr(
        zz, "_frozen_planning_gpu", lambda frozen, *, gpu_used: True
    )
    stacks = RERUN._bomb_cell_execution(monkeypatch)
    commits = RERUN._spy_commits(monkeypatch)
    with pytest.raises(Exception) as excinfo:
        RERUN._run(tmp_path, monkeypatch, use_gpu=True, resume_mode="auto")
    message = str(excinfo.value)
    # ANY precise fail-closed refusal is acceptable here (injected GPU plan):
    # the frozen-capacity guard, the frozen identity check, or the backend
    # injection guard itself. What matters is that NOTHING was executed or
    # written.
    assert (
        "mixing backends" in message
        or "does not match the stored run identity" in message
        or "cannot be re-derived" in message
    ), message
    assert stacks == [], stacks
    assert commits == [], commits
    assert RERUN._commit_bytes(store) == before


# ---------------------------------------------------------------------------
# 4 — GPU frozen but no GPU: typed refusal (no silent CPU switch)
# ---------------------------------------------------------------------------


def test_gpu_frozen_partial_resume_without_gpu_is_refused(tmp_path, monkeypatch):
    out, store, removed_ids, before, frozen_total = _partial_store(
        tmp_path, monkeypatch, use_gpu=True
    )
    stacks = RERUN._bomb_cell_execution(monkeypatch)
    commits = RERUN._spy_commits(monkeypatch)
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable) as excinfo:
        RERUN._run(tmp_path, monkeypatch, use_gpu=False, resume_mode="auto")
    message = str(excinfo.value)
    assert "backend='gpu'" in message
    assert "requires" in message
    assert "Remedy:" in message
    assert excinfo.value.frozen_backend == "gpu"
    # No CPU fallback, no mixed backend, no Cell admitted, no commit written.
    assert stacks == [], stacks
    assert commits == [], commits
    assert RERUN._commit_bytes(store) == before
