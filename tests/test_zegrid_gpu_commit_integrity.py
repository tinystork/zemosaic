"""ZM-ZEGRID-COMMIT-INTEGRITY — COMPONENT tests for the GPU commit helpers.

These are component-level tests: they call ``_gpu_durable_commit`` /
``_r55_durable_cell_coverage`` / the real ``BoundedPipeline`` directly with a real
store and registry. They do NOT drive the full ``_run_single`` GPU orchestration;
that integration proof lives in ``tests/test_zegrid_gpu_durable_integration.py``
(where removing the production call makes the tests fail).

The GPU kernel is never executed (no CUDA). The red-baseline branch below is a
HELPER/coverage probe, not a reproduction of the original orchestration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TOOLS = REPO_ROOT / "tools" / "zegrid_resume_m4"
for _path in (str(SRC), str(REPO_ROOT / "tests"), str(TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

import m4_resume_harness as HARNESS  # noqa: E402

MIB = 2 ** 20


def _store_registry(tmp_path, *, run_id="gpu-commit"):
    rows = HARNESS.ensure_inputs(tmp_path / "in")
    identity = HARNESS.make_identity(run_id, rows)
    root = tmp_path / "store"
    store = zrs.RunCellStore.open(
        root,
        identity=identity,
        budget=zrs.DiskBudget(durable_cell_bytes=64 * MIB),
        session_id="gpu-commit",
    )
    registry = zrr.RunResultRegistry(
        run_id=run_id,
        expected_cell_ids=list(HARNESS.CELL_IDS),
        gauge_mode="per_cell",
    )
    state = {"store": store, "publishing": True, "degraded_reason": None, "commits": 0}
    return store, registry, state


def _accept(registry, index, *, attempt="gpu-unit"):
    cid = HARNESS.CELL_IDS[index]
    payload = HARNESS.compute_cell(index)
    receipt = registry.accept(
        payload, backend="gpu", plan_generation=0, attempt_id=f"{attempt}:{cid}"
    )
    return cid, receipt


def test_gpu_accepted_unit_is_committed_verified_and_released(tmp_path):
    store, registry, state = _store_registry(tmp_path)
    try:
        cid, receipt = _accept(registry, 0)
        # RED baseline: the GPU consumer accepted and returned WITHOUT a commit.
        assert store.attempts(cid) == []
        committed = zz._gpu_durable_commit(
            state,
            registry,
            cid,
            receipt,
            plan_generation=0,
            backend="gpu",
        )
        assert committed is True
        assert state["commits"] == 1
        # The commit is durable and readable by the cheap local lookup.
        record = store.require_commit(cid, receipt.attempt_id)
        assert record.cell_id == cid
        assert record.backend == "gpu"
        # Released from RAM (no retained large array in the registry).
        assert int(registry.retained_bytes) == 0
        assert store.report()["counts"]["attempts"] == 1
    finally:
        registry.dispose()
        store.close()


def test_missing_gpu_commit_is_refused_before_diagnostics(tmp_path):
    store, registry, state = _store_registry(tmp_path)
    try:
        cid, _receipt = _accept(registry, 0)  # accepted, NEVER committed
        ids = list(HARNESS.CELL_IDS)
        coverage = getattr(zz, "_r55_durable_cell_coverage", None)
        if coverage is None:
            # RED BASELINE: the boolean "ready" check passes, so the lazy
            # durable consumer (diagnostics) raises the LATE incident symptom.
            # The test FAILS with that exact symptom as evidence.
            assert zz._r55_durable_coverage_ready(state) is True
            view = zz._bounded_store_results(store, ids, store_state=state)
            try:
                # The incident accessor: the lazy per-cell descriptor, keyed
                # by cell_id (bounded_assembly descriptor -> commit_record).
                view.descriptors[cid]
            except zrs.StoreNotFoundError as late:
                pytest.fail(
                    "baseline reached the LATE missing-commit symptom instead "
                    f"of refusing before diagnostics: {late}"
                )
            pytest.fail("baseline never refused the missing GPU commit")
        # FIXED: complete sparse coverage is validated BEFORE any lazy durable
        # consumer; the refusal names the missing cell.
        with pytest.raises(zrs.StoreCommitNotDurable) as excinfo:
            coverage(state, ids, context="diagnostics")
        exc = excinfo.value
        assert exc.cell_id == cid
        assert exc.run_id == "gpu-commit"
        assert str(store.root) == exc.store_root
        assert "INCOMPLETE" in str(exc)
    finally:
        registry.dispose()
        store.close()


def test_partial_coverage_names_the_missing_cell(tmp_path):
    store, registry, state = _store_registry(tmp_path)
    try:
        committed_cid, receipt = _accept(registry, 0)
        zz._gpu_durable_commit(
            state, registry, committed_cid, receipt, plan_generation=0
        )
        missing_cid = HARNESS.CELL_IDS[1]
        with pytest.raises(zrs.StoreCommitNotDurable) as excinfo:
            zz._r55_durable_cell_coverage(
                state, list(HARNESS.CELL_IDS), context="assembly"
            )
        assert excinfo.value.cell_id == missing_cid
        assert "3/4" in str(excinfo.value)
    finally:
        registry.dispose()
        store.close()


def test_full_coverage_passes_and_returns_the_store(tmp_path):
    store, registry, state = _store_registry(tmp_path)
    try:
        for index in range(len(HARNESS.CELL_IDS)):
            cid, receipt = _accept(registry, index, attempt=f"gpu-{index}")
            zz._gpu_durable_commit(
                state, registry, cid, receipt, plan_generation=0
            )
        assert state["commits"] == len(HARNESS.CELL_IDS)
        assert (
            zz._r55_durable_cell_coverage(
                state, list(HARNESS.CELL_IDS), context="assembly"
            )
            is store
        )
        assert store.report()["counts"]["cells"] == len(HARNESS.CELL_IDS)
    finally:
        registry.dispose()
        store.close()


def test_integrity_failure_is_terminal_and_typed(tmp_path, monkeypatch):
    store, registry, state = _store_registry(tmp_path)
    try:
        cid, receipt = _accept(registry, 0)

        def _broken(*_args, **_kwargs):
            raise zrs.StoreIntegrityError("simulated commit failure")

        monkeypatch.setattr(store, "commit_and_release", _broken)
        with pytest.raises(zrs.StoreCommitNotDurable) as excinfo:
            zz._gpu_durable_commit(
                state, registry, cid, receipt, plan_generation=0
            )
        assert excinfo.value.cell_id == cid
        assert state["publishing"] is False
        # Never reclassified as a GPU fallback (which would trigger a CPU rerun).
        assert not isinstance(excinfo.value, zz._R55GpuFallback)
    finally:
        registry.dispose()
        store.close()


def test_component_commit_is_durable_before_a_stop_boundary(tmp_path):
    store, registry, state = _store_registry(tmp_path)
    try:
        cid, receipt = _accept(registry, 0)
        zz._gpu_durable_commit(state, registry, cid, receipt, plan_generation=0)
        # A stop honoured at the next unit boundary must find this unit durable.
        assert store.require_commit(cid, receipt.attempt_id).cell_id == cid
        assert state["commits"] == 1
    finally:
        registry.dispose()
        store.close()


def test_bounded_pipeline_preserves_the_integrity_error_type():
    """COMPONENT: BoundedPipeline must NOT reclassify a durable-commit failure.

    No GPU is involved: the consumer raises the typed integrity error and the
    pipeline must surface the SAME type (the mission requires it to stay
    terminal instead of becoming a GPU fallback / conserved wrapper).
    """
    from zemosaic.core.zegrid import parallel as zpar

    def _produce(item):
        return {"cell": None, "_unit_index": int(item[0])}

    def _consume(payload):
        raise zrs.StoreCommitNotDurable(
            "injected durable failure",
            run_id="pipeline-run",
            cell_id="r0000c0000",
            store_root="/tmp/store",
        )

    pipeline = zpar.BoundedPipeline(
        tasks=[(0, None)],
        produce=_produce,
        consume=_consume,
        release=lambda payload: None,
        prefetch_units=0,
        host_budget_bytes=64 * MIB,
        device_budget_bytes=64 * MIB,
        disk_budget_bytes=64 * MIB,
        disk_enabled=False,
        host_bytes=lambda payload: 1,
        device_bytes=lambda payload: 1,
        disk_bytes=lambda payload: 1,
    )
    with pytest.raises(zrs.StoreCommitNotDurable) as excinfo:
        pipeline.run()
    assert not isinstance(excinfo.value, zz._R55GpuFallback)


def test_configured_but_unhealthy_store_fails_closed(tmp_path):
    store, registry, state = _store_registry(tmp_path)
    try:
        cid, receipt = _accept(registry, 0)
        state["publishing"] = False
        with pytest.raises(zz._R55DurableStoreUnavailable):
            zz._gpu_durable_commit(
                state, registry, cid, receipt, plan_generation=0
            )
        assert store.attempts(cid) == []
    finally:
        registry.dispose()
        store.close()
