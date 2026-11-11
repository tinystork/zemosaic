"""ZM-ZEGRID-RESTORED-GAUGE-ADMISSION-20261111 — restored gauge is resid-ent.

Production defect: a GPU-frozen partial resume was refused although the run
would only RESTORE the durable global gauge and never execute
``compute_global_gauge``. The ACTIVE admission nevertheless priced the
compute-phase footprint (``footprint_bytes x gauge_workers``), which dominated
the aggregate and refused the resume.

These regressions pin the contract:

* ``gauge_restored=True`` (derived ONLY from a verified restored artifact)
  excludes the gauge COMPUTE term and prices the restored residency explicitly;
* a fresh run, a missing/corrupt artifact, a failed restore or an
  identity/``gauge_sha256`` mismatch keeps today's compute pricing and every
  refusal byte-identical;
* the published aggregate is explainable: max over the overlap groups of the
  phases that WILL run, with the restored gauge residency named, never a
  silently omitted phase.

The decision-level tests use the real ACTIVE planner seam
(``_r55_active_facts`` / ``_r55_active_phase_requests`` / ``_r55_active_decide``)
on a real ``ExecutionBundle``; the artifact tests use a REAL ``RunCellStore``.
No pixels are stacked, no dataset is touched, no science/geometry changes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.canonical_streaming import FixedNormalization  # noqa: E402
from zemosaic.core.zegrid import auto_layout as zal  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import parallel as zpar  # noqa: E402
from zemosaic.core.zegrid import resource_envelope as renv  # noqa: E402
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402
from zemosaic.core.zegrid import scheduler as zsched  # noqa: E402

import test_zegrid_active_external_wiring as WIRE  # noqa: E402
import test_zegrid_resume_complete_gpu_rerun as RERUN  # noqa: E402

MIB = 2 ** 20
GIB = 2 ** 30

#: Large enough to dominate the aggregate at the tight budget below.
GAUGE_COMPUTE_BYTES = 900 * MIB
#: Tight plan budget: the compute-phase gauge alone cannot fit.
TIGHT_RAW = 800 * MIB
TIGHT_B_PLAN = (4 * TIGHT_RAW) // 5


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _restored_gauge_object():
    """The exact shape returned by ``restore_gauge`` (a FixedNormalization)."""
    return FixedNormalization(
        reference_index=0,
        coefficients=np.zeros((4, 3, 2), dtype=np.float64),
        norm_active=np.ones(4, dtype=bool),
        weights=np.ones(4, dtype=np.float64),
        weight_active=np.ones(4, dtype=bool),
        exclusions=(),
    )


def _exact_resident_bytes(gauge) -> int:
    return (
        int(gauge.coefficients.nbytes)
        + int(gauge.norm_active.nbytes)
        + int(gauge.weights.nbytes)
        + int(gauge.weight_active.nbytes)
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    monkeypatch.setattr(zal, "ScientificFloors", WIRE._tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", WIRE._FakeModel(1 << 62))
    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)
    monkeypatch.setattr(zpar, "_CACHE_PER_WORKER_FOOTPRINT_BYTES", 0)
    monkeypatch.setattr(zpar, "cache_footprint_bytes", lambda px: 0)
    # ONE explicit seam for the FROZEN gauge compute footprint: the defect is
    # about WHETHER it is priced, not about re-deriving the model here.
    monkeypatch.setattr(
        zz,
        "_bundle_gauge_footprint_bytes",
        lambda bundle, canvas_px: int(GAUGE_COMPUTE_BYTES),
    )
    monkeypatch.setattr(
        zz,
        "_estimate_streaming_bytes",
        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3,
        storage_profile=None, **_kw: 8 * MIB,
    )
    monkeypatch.setattr(zz.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: TIGHT_RAW)
    monkeypatch.setattr(
        zz.shutil,
        "disk_usage",
        lambda _p: __import__("types").SimpleNamespace(
            total=100 * GIB, used=0, free=100 * GIB
        ),
    )


def _bundle(tmp_path, monkeypatch):
    descs = WIRE._corpus()
    canvas = zg.build_canvas(descs)
    bundle = zz._build_execution_bundle(
        canvas,
        descs,
        None,
        raw_available_bytes=TIGHT_RAW,
        budget_bytes=TIGHT_B_PLAN,
        cpu_count=4,
        cache_root=str(tmp_path / "cache"),
        pinned_layout=(2, 2),
        max_refinement_rank=4,
        halo_px=8,
    )
    return descs, canvas, bundle


def _decide(tmp_path, descs, canvas, bundle, **kwargs):
    return zz._r55_active_decide(
        bundle=bundle,
        cell_ctxs=bundle.cell_ctxs,
        canvas=canvas,
        halo_px=8,
        cache_root=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "out"),
        frame_ids=tuple(d.frame_id.logical_path for d in descs),
        **kwargs,
    )


def _gauge_envelope(plan_set):
    return next(e for e in plan_set.phases if e.phase == renv.PHASE_GAUGE)


def _group_sums(plan_set):
    per_phase = {e.phase: int(e.estimate.peak_bytes) for e in plan_set.phases}
    return {
        tuple(group): sum(per_phase[p] for p in group)
        for group in renv.overlap_groups(plan_set.phases)
    }


# ---------------------------------------------------------------------------
# 1. Restored-gauge resume admits at the budget that refuses a fresh price.
# ---------------------------------------------------------------------------

def test_restored_gauge_resume_admits_at_the_tight_budget(
    tmp_path, monkeypatch
):
    descs, canvas, bundle = _bundle(tmp_path, monkeypatch)
    gauge = _restored_gauge_object()
    resident = zz._restored_gauge_resident_bytes(gauge)
    assert resident == _exact_resident_bytes(gauge) > 0

    # CONTROL: the unproven/compute path at this SAME budget refuses.
    with pytest.raises(zsched.PlanRefused) as refused:
        _decide(tmp_path, descs, canvas, bundle)
    message = str(refused.value)
    assert "phase=gauge components:" in message
    assert f"footprint_bytes={GAUGE_COMPUTE_BYTES}" in message

    # FIX: the verified restored artifact admits and names its residency.
    decision = _decide(
        tmp_path,
        descs,
        canvas,
        bundle,
        gauge_restored=True,
        restored_gauge_resident_bytes=resident,
    )
    gauge_env = _gauge_envelope(decision.plan_set)
    components = dict(gauge_env.estimate.components)
    assert components == {"restored_gauge_resident_bytes": resident}
    assert "gauge_worker_footprint_bytes" not in components
    assert gauge_env.concurrency.units == 1
    assert gauge_env.concurrency.units_in_flight == 1
    assert decision.plan_set.aggregate_host_peak_bytes <= TIGHT_B_PLAN
    # The gauge compute term is gone from the aggregate.
    assert decision.plan_set.aggregate_host_peak_bytes_raw < (
        GAUGE_COMPUTE_BYTES
    )
    assert decision.facts["gauge_restored"] is True
    assert decision.facts["gauge_phase_basis"] == "restored"
    assert decision.facts["gauge_compute_priced"] is False
    assert decision.facts["gauge_phase_bytes"] == resident


def test_restored_gauge_resume_runs_its_pending_cells(tmp_path, monkeypatch):
    """Product seam: a GPU-frozen partial resume with a restored gauge runs.

    The gauge artifact is genuinely restored by the REAL ``RunCellStore``; the
    tight budget is reproduced by making the compute footprint dominate during
    the resume only. The resumed run must execute its pending Cells (not zero,
    not the whole batch) and publish the ``restored`` gauge basis.
    """
    # Initial run at a comfortable budget so the store, gauge, geometry and
    # frozen GPU admission are real.
    monkeypatch.setattr(
        zz, "_bundle_gauge_footprint_bytes", lambda bundle, cpx: 0
    )
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 8 * GIB)
    out, store = RERUN._run(tmp_path, monkeypatch, use_gpu=True)
    commits = RERUN._commits(store)
    assert len(commits) == 4
    removed = list(commits[2:])
    removed_ids = sorted(RERUN._commit_cid(path) for path in removed)
    for path in removed:
        path.unlink()

    # Resume under the defect's shape: the compute footprint alone exceeds the
    # plan budget unless the verified restored gauge is priced as residency.
    monkeypatch.setattr(
        zz,
        "_bundle_gauge_footprint_bytes",
        lambda bundle, cpx: int(GAUGE_COMPUTE_BYTES),
    )
    monkeypatch.setattr(
        zz, "available_memory_bytes", lambda: int(TIGHT_RAW)
    )
    seen = []
    real = zz._prepare_stack_cell

    def spy(task):
        seen.append(str(task[0].cell_id))
        return real(task)

    monkeypatch.setattr(zz, "_prepare_stack_cell", spy)

    RERUN._run(tmp_path, monkeypatch, use_gpu=True, resume_mode="auto")
    assert seen == sorted(removed_ids), (seen, removed_ids)
    manifest = RERUN._published_manifest(out)
    block = manifest["r55_scheduler"]
    assert block["status"] == "executed"
    assert block["worker_adaptation"]["gauge_phase"]["basis"] == "restored"
    assert block["worker_adaptation"]["gauge_phase"]["compute_priced"] is False
    assert (
        block["worker_adaptation"]["gauge_phase"]["restored_resident_bytes"]
        > 0
    )


# ---------------------------------------------------------------------------
# 2. Missing/unproven artifact keeps today's compute pricing and refusal.
# ---------------------------------------------------------------------------

def test_missing_or_unproven_artifact_prices_the_compute_unchanged(
    tmp_path, monkeypatch
):
    descs, canvas, bundle = _bundle(tmp_path, monkeypatch)

    with pytest.raises(zsched.PlanRefused) as baseline:
        _decide(tmp_path, descs, canvas, bundle)

    # A claimed restore WITHOUT a proven residency is fail-closed: the compute
    # phase is priced, and the refusal is byte-identical.
    with pytest.raises(zsched.PlanRefused) as unproven:
        _decide(
            tmp_path,
            descs,
            canvas,
            bundle,
            gauge_restored=True,
            restored_gauge_resident_bytes=0,
        )
    assert str(unproven.value) == str(baseline.value)

    with pytest.raises(zsched.PlanRefused) as missing_residency:
        _decide(
            tmp_path,
            descs,
            canvas,
            bundle,
            gauge_restored=True,
            restored_gauge_resident_bytes=None,
        )
    assert str(missing_residency.value) == str(baseline.value)


# ---------------------------------------------------------------------------
# 3. Fresh pricing is numerically identical.
# ---------------------------------------------------------------------------

def test_fresh_run_prices_the_gauge_compute_exactly_as_today(
    tmp_path, monkeypatch
):
    descs, canvas, bundle = _bundle(tmp_path, monkeypatch)
    # A higher budget admits the compute plan so its numbers can be asserted.
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 4 * GIB)
    high = zz._build_execution_bundle(
        canvas,
        descs,
        None,
        raw_available_bytes=4 * GIB,
        budget_bytes=(4 * (4 * GIB)) // 5,
        cpu_count=4,
        cache_root=str(tmp_path / "cache"),
        pinned_layout=(2, 2),
        max_refinement_rank=4,
        halo_px=8,
    )
    fresh = _decide(tmp_path, descs, canvas, high)
    facts = fresh.facts
    assert facts["gauge_restored"] is False
    assert facts["gauge_phase_basis"] == "compute"
    assert facts["gauge_compute_priced"] is True
    assert facts["gauge_restored_resident_bytes"] is None
    assert facts["gauge_phase_bytes"] == (
        GAUGE_COMPUTE_BYTES * int(facts["gauge_workers"])
    )
    gauge_env = _gauge_envelope(fresh.plan_set)
    assert dict(gauge_env.estimate.components) == {
        "gauge_worker_footprint_bytes": GAUGE_COMPUTE_BYTES
        * int(facts["gauge_workers"])
    }
    assert gauge_env.concurrency.units == int(facts["gauge_workers"])


# ---------------------------------------------------------------------------
# 4. The aggregate is explainable: max over groups, no phase omitted.
# ---------------------------------------------------------------------------

def test_restored_aggregate_is_explainable_and_complete(
    tmp_path, monkeypatch
):
    descs, canvas, bundle = _bundle(tmp_path, monkeypatch)
    gauge = _restored_gauge_object()
    resident = zz._restored_gauge_resident_bytes(gauge)
    decision = _decide(
        tmp_path,
        descs,
        canvas,
        bundle,
        gauge_restored=True,
        restored_gauge_resident_bytes=resident,
    )
    plan_set = decision.plan_set
    # Every required PLAN phase is present, plus exactly one stack phase.
    phases = {e.phase for e in plan_set.phases}
    assert set(renv.REQUIRED_PLAN_PHASES) <= phases
    assert len(phases & set(renv.CELL_STACK_PHASES)) == 1
    # The published raw aggregate EQUALS the max overlap-group sum.
    groups = _group_sums(plan_set)
    assert plan_set.aggregate_host_peak_bytes_raw == max(groups.values())
    # The restored gauge phase is a group on its own with the named residency.
    gauge_group = next(g for g in groups if renv.PHASE_GAUGE in g)
    assert gauge_group == (renv.PHASE_GAUGE,)
    assert groups[gauge_group] == resident
    # The compute footprint appears NOWHERE in the priced components.
    priced = {
        key
        for envelope in plan_set.phases
        for key, _value in envelope.estimate.components
    }
    assert "gauge_worker_footprint_bytes" not in priced
    assert "restored_gauge_resident_bytes" in priced
    # The aggregate is still a real upper bound over the phases that run.
    assert plan_set.aggregate_host_peak_bytes == (
        (max(groups.values()) * 115 + 99) // 100
    )


# ---------------------------------------------------------------------------
# 5. Identity / gauge_sha256 mismatch must never benefit from the exclusion.
# ---------------------------------------------------------------------------

def _store_with_gauge(tmp_path, monkeypatch):
    monkeypatch.setattr(
        zz, "_bundle_gauge_footprint_bytes", lambda bundle, cpx: 0
    )
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 8 * GIB)
    out, store = RERUN._run(tmp_path, monkeypatch, use_gpu=True)
    return out, store


def test_identity_mismatch_never_benefits_from_the_exclusion(
    tmp_path, monkeypatch
):
    out, store = _store_with_gauge(tmp_path, monkeypatch)
    boot = zres.ResumeBootstrap.open(
        store, budget=64 * MIB, session_id=str(out)
    )
    try:
        # CONTROL: the real artifact restores and verifies.
        gauge, _frames, _ref = boot.restore_gauge()
        resident = zz._restored_gauge_resident_bytes(gauge)
        assert resident == _exact_resident_bytes(gauge) > 0

        # Tamper ONLY the frozen state hash: restore must fail closed BEFORE
        # any admission could see gauge_restored=True.
        state = dict(boot.state)
        state["gauge_sha256"] = "0" * 64
        boot.state = state
        with pytest.raises(zres.ResumeIncompatible):
            boot.restore_gauge()
    finally:
        boot.close()


def test_corrupt_gauge_artifact_never_yields_a_restored_signal(
    tmp_path, monkeypatch
):
    out, store = _store_with_gauge(tmp_path, monkeypatch)
    gauge_path = Path(store) / "gauge.json"
    if not gauge_path.exists():
        # The durable gauge artifact lives under the run root.
        candidates = list(Path(store).rglob("gauge.json"))
        assert candidates, "no durable gauge artifact found"
        gauge_path = candidates[0]
    payload = json.loads(gauge_path.read_text(encoding="utf-8"))
    payload["coefficients"] = [[[9.0, 9.0]]]
    gauge_path.write_text(json.dumps(payload), encoding="utf-8")

    boot = zres.ResumeBootstrap.open(
        store, budget=64 * MIB, session_id=str(out)
    )
    try:
        with pytest.raises(zres.ResumeError):
            boot.restore_gauge()
    finally:
        boot.close()

    # Direct admission control: gauge_restored is never passed here, so the
    # compute is priced and the tight-budget refusal is unchanged.
    monkeypatch.setattr(
        zz,
        "_bundle_gauge_footprint_bytes",
        lambda bundle, canvas_px: int(GAUGE_COMPUTE_BYTES),
    )
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: TIGHT_RAW)
    descs, canvas, bundle = _bundle(tmp_path, monkeypatch)
    with pytest.raises(zsched.PlanRefused) as refused:
        _decide(tmp_path, descs, canvas, bundle)
    assert "phase=gauge components:" in str(refused.value)
