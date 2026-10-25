"""ZM-ZEGRID-R56 REWORK-1 — F1/F2/F4 adversarial gates.

Covers the confirmed review findings:

* F1: credit is acquired BEFORE produce allocates; actual retained bytes never
  exceed the declared bound; an underestimated prediction fails closed; partial
  produce errors and cancellation release exactly once; the ledger stays
  balanced.
* F2: the concurrent producer owner is part of the simultaneous host peak; the
  envelope declares the simultaneous rule; a case where each phase alone fits
  but the sum does not is refused/shrunk, not approved.
* F4: a fresh disk observation may only SHRINK the budget and refuses before any
  production when the volume cannot hold the next staging step.
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import ram_contract as rcontract
from zemosaic.core.zegrid import scheduler as zs

MIB = 2 ** 20
GIB = 2 ** 30


# ---------------------------------------------------------------------------
# F1 — pre-acquisition before allocation
# ---------------------------------------------------------------------------

def test_credit_is_acquired_before_produce_allocates(tmp_path):
    """Inside produce(), the lease must ALREADY be held (Junior 200B witness)."""
    observed = {"used_at_produce": [], "files_at_produce": []}

    def produce(task):
        used = ledger_ref["pipe"].telemetry()["ledger"]["host_used_bytes"]
        files = sum(
            p.stat().st_size for p in tmp_path.iterdir() if p.is_file()
        )
        observed["used_at_produce"].append(int(used))
        observed["files_at_produce"].append(int(files))
        path = tmp_path / f"cache_{task}.bin"
        path.write_bytes(b"x" * 100)
        return {"task": task, "path": path}

    def consume(payload):
        return payload["task"]

    def release(payload):
        try:
            os.unlink(payload["path"])
        except FileNotFoundError:
            pass

    pipe = zpar.BoundedPipeline(
        tasks=[0, 1],
        produce=produce,
        consume=consume,
        release=release,
        prefetch_units=0,
        host_budget_bytes=100,
        device_budget_bytes=100,
        disk_budget_bytes=100,
        host_bytes=lambda _p: 100,
        device_bytes=lambda _p: 0,
        disk_bytes=lambda _p: 100,
        host_reserve_bytes=lambda _t: 100,
        disk_reserve_bytes=lambda _t: 100,
    )
    ledger_ref = {"pipe": pipe}
    pipe.run()
    tel = pipe.telemetry()
    # Both produces saw a FULLY RESERVED lease before they wrote anything.
    assert observed["used_at_produce"] == [100, 100]
    assert observed["files_at_produce"] == [0, 0]
    assert tel["credit_accounting"]["mode"] == "pre_acquire"
    assert tel["ledger"]["disk_peak_bytes"] == 100
    assert tel["ledger"]["balanced"] is True


def test_second_produce_waits_for_released_credit(tmp_path):
    """Only one slot admitted: the second produce must NOT start early.

    Event-synchronised, with bounded waits so a regression FAILS instead of
    deadlocking the test session.
    """
    second_started = threading.Event()
    first_consumed = threading.Event()

    def produce(task):
        if task == 1:
            second_started.set()
        path = tmp_path / f"c{task}.bin"
        path.write_bytes(b"x" * 100)
        return {"task": task, "path": path}

    def consume(payload):
        if payload["task"] == 0:
            # Give the producer every chance to (incorrectly) start task 1.
            assert not second_started.wait(0.3), (
                "second produce started while the only credit slot was retained"
            )
            first_consumed.set()
        return payload["task"]

    def release(payload):
        try:
            os.unlink(payload["path"])
        except FileNotFoundError:
            pass

    pipe = zpar.BoundedPipeline(
        tasks=[0, 1],
        produce=produce,
        consume=consume,
        release=release,
        prefetch_units=0,
        host_budget_bytes=100,
        device_budget_bytes=1,
        disk_budget_bytes=100,
        host_bytes=lambda _p: 100,
        device_bytes=lambda _p: 0,
        disk_bytes=lambda _p: 100,
        host_reserve_bytes=lambda _t: 100,
        disk_reserve_bytes=lambda _t: 100,
    )
    assert pipe.run() == [0, 1]
    assert first_consumed.is_set()
    assert pipe.telemetry()["ledger"]["balanced"] is True


def test_underestimated_prediction_fails_closed():
    pipe = zpar.BoundedPipeline(
        tasks=[0],
        produce=lambda _t: {"bytes": 100},
        consume=lambda p: p["bytes"],
        release=lambda _p: None,
        prefetch_units=0,
        host_budget_bytes=1000,
        device_budget_bytes=1,
        disk_budget_bytes=1000,
        host_bytes=lambda _p: 100,
        device_bytes=lambda _p: 0,
        disk_bytes=lambda _p: 100,
        # Prediction too small: the allocation already happened, so this must
        # FAIL CLOSED rather than approve the extra bytes.
        host_reserve_bytes=lambda _t: 50,
        disk_reserve_bytes=lambda _t: 50,
    )
    with pytest.raises(Exception) as excinfo:
        pipe.run()
    message = str(excinfo.value)
    assert "exceeds the pre-acquired reservation" in message
    tel = pipe.telemetry()
    assert tel["units_produced"] == tel["units_released"] == 1
    assert tel["ledger"]["balanced"] is True
    assert tel["ledger"]["host_used_bytes"] == 0
    assert tel["ledger"]["disk_used_bytes"] == 0


def test_partial_produce_failure_releases_pre_acquired_credit():
    calls = {"n": 0}

    def produce(_task):
        calls["n"] += 1
        raise RuntimeError("cache build exploded")

    pipe = zpar.BoundedPipeline(
        tasks=[0, 1],
        produce=produce,
        consume=lambda p: p,
        release=lambda _p: None,
        prefetch_units=1,
        host_budget_bytes=100,
        device_budget_bytes=1,
        disk_budget_bytes=100,
        host_bytes=lambda _p: 100,
        device_bytes=lambda _p: 0,
        disk_bytes=lambda _p: 100,
        host_reserve_bytes=lambda _t: 100,
        disk_reserve_bytes=lambda _t: 100,
    )
    with pytest.raises(zpar.PipelineError):
        pipe.run()
    tel = pipe.telemetry()
    assert tel["ledger"]["balanced"] is True
    assert tel["ledger"]["host_leases"] == 0
    assert tel["ledger"]["disk_leases"] == 0
    assert tel["units_produced"] == tel["units_released"] == 0
    # The pipeline stops at the first producer failure (no silent retry storm).
    assert calls["n"] == 1


def test_pre_acquisition_cancellation_releases_and_stays_balanced():
    cancel_evt = threading.Event()
    holder = {}

    def produce(task):
        if task == 1:
            holder["p"].cancel("test_cancel")
            cancel_evt.set()
        return {"task": task}

    def consume(payload):
        if payload["task"] == 0:
            assert cancel_evt.wait(2.0)
        return payload["task"]

    def release(payload):
        pass

    pipe = zpar.BoundedPipeline(
        tasks=[0, 1, 2],
        produce=produce,
        consume=consume,
        release=release,
        # ONE extra slot so the producer can run task 1 while task 0 is consumed
        # (with prefetch 0 pre-acquisition correctly blocks the second produce).
        prefetch_units=1,
        host_budget_bytes=200,
        device_budget_bytes=1,
        disk_budget_bytes=200,
        host_bytes=lambda _p: 100,
        device_bytes=lambda _p: 0,
        disk_bytes=lambda _p: 100,
        host_reserve_bytes=lambda _t: 100,
        disk_reserve_bytes=lambda _t: 100,
    )
    holder["p"] = pipe
    with pytest.raises(zpar.PipelineCancelled):
        pipe.run()
    tel = pipe.telemetry()
    assert tel["ledger"]["balanced"] is True
    assert tel["units_produced"] == tel["units_released"]


def test_reconcile_releases_surplus_without_a_gap():
    pipe = zpar.BoundedPipeline(
        tasks=[0],
        produce=lambda _t: {"bytes": 40},
        consume=lambda p: p["bytes"],
        release=lambda _p: None,
        prefetch_units=0,
        host_budget_bytes=100,
        device_budget_bytes=1,
        disk_budget_bytes=100,
        host_bytes=lambda _p: 40,
        device_bytes=lambda _p: 0,
        disk_bytes=lambda _p: 40,
        host_reserve_bytes=lambda _t: 100,
        disk_reserve_bytes=lambda _t: 100,
    )
    pipe.run()
    ledger = pipe.telemetry()["ledger"]
    assert ledger["reconciliations"] >= 2
    assert ledger["surplus_released_bytes"] == 120
    assert ledger["balanced"] is True


# ---------------------------------------------------------------------------
# F4 — fresh disk observation before staging
# ---------------------------------------------------------------------------

def test_fresh_disk_shrink_refuses_before_production():
    state = {"produced": 0}

    def produce(_task):
        state["produced"] += 1
        return {"bytes": 100}

    def before(task, ledger):
        # External consumption: the volume can no longer hold the next slot.
        ledger.reduce_budget("disk", 10)
        if ledger.snapshot()["disk_used_bytes"] + 100 > ledger.disk_budget_bytes:
            raise zpar.PipelineAccountingError("fresh disk too small")

    pipe = zpar.BoundedPipeline(
        tasks=[0],
        produce=produce,
        consume=lambda p: p,
        release=lambda _p: None,
        prefetch_units=0,
        host_budget_bytes=100,
        device_budget_bytes=1,
        disk_budget_bytes=100,
        host_bytes=lambda _p: 100,
        device_bytes=lambda _p: 0,
        disk_bytes=lambda _p: 100,
        host_reserve_bytes=lambda _t: 100,
        disk_reserve_bytes=lambda _t: 100,
        before_produce=before,
    )
    with pytest.raises(zpar.PipelineError):
        pipe.run()
    assert state["produced"] == 0  # refusal happened BEFORE production
    ledger = pipe.telemetry()["ledger"]
    assert ledger["balanced"] is True
    assert ledger["budget_reductions"] == [
        {"kind": "disk", "from_bytes": 100, "to_bytes": 10}
    ]


def test_budget_reduction_refuses_growth():
    ledger = zpar.BufferLedger(
        host_budget_bytes=100, device_budget_bytes=100, disk_budget_bytes=100
    )
    with pytest.raises(zpar.PipelineAccountingError):
        ledger.reduce_budget("disk", 101)
    assert ledger.reduce_budget("disk", 100) == 100


# ---------------------------------------------------------------------------
# F2 — simultaneous ownership model
# ---------------------------------------------------------------------------

def test_simultaneous_rule_counts_shared_once():
    stack = 1000
    cache = 700
    shared = 400
    simultaneous = rcontract.simultaneous_stack_and_cache_bytes(
        stack, cache, shared
    )
    assert simultaneous == 1000 + 700 - 400
    assert simultaneous > max(stack, cache)


def test_envelope_accepts_both_rules_and_refuses_mismatch():
    base = dict(
        phase=rcontract.PHASE_PLAN,
        raw_available_bytes=1000,
        factor_num=4,
        factor_den=5,
        budget_bytes=800,
        layout=(1, 1),
        refinement_rank=1,
        cpu_tile=0,
        gpu_tile=128,
        backend=rcontract.BACKEND_GPU,
        cells_in_flight=1,
        cache_workers_per_cell=2,
        shared_bytes=400,
        stack_total_bytes=1000,
        cache_total_bytes=700,
        pinned=False,
        reason="test",
    )
    max_env = rcontract.PlanEnvelope(
        aggregate_bytes=1000, **base
    )
    rcontract.validate_envelope(max_env)  # default rule, unchanged

    sim_env = rcontract.PlanEnvelope(
        aggregate_bytes=1300,
        aggregate_rule=rcontract.AGGREGATE_RULE_STACK_PLUS_CACHE,
        **base,
    )
    rcontract.validate_envelope(sim_env)
    assert sim_env.to_dict()["aggregate_rule"] == (
        rcontract.AGGREGATE_RULE_STACK_PLUS_CACHE
    )

    with pytest.raises(rcontract.InvalidEnvelope):
        rcontract.validate_envelope(
            rcontract.PlanEnvelope(
                aggregate_bytes=1000,
                aggregate_rule=rcontract.AGGREGATE_RULE_STACK_PLUS_CACHE,
                **base,
            )
        )


def _ctx(**over):
    data = dict(
        phase=zs.PHASE_CELL_STACK_STREAM,
        backend=zs.BACKEND_CPU,
        budget_bytes=1000,
        raw_available_bytes=1250,
        factor_num=4,
        factor_den=5,
        cpu_logical=16,
        units_total=9,
        shared_bytes=100,
        per_unit_stack_bytes=400,
        per_unit_cache_bytes=300,
        per_unit_prefetch_bytes=10,
        disk_free_bytes=10 * GIB,
        layout=(3, 3),
        refinement_rank=1,
        patch_hw=(100, 100),
        halo_px=128,
        channels=3,
        n_frames=10,
        tile_ladder=(128,),
        prefetch_ladder=(0, 1),
        workers_ladder=(1,),
        per_unit_prefetch_disk_bytes=300,
    )
    data.update(over)
    return zs.PlanContext(**data)


def test_concurrent_producer_breaks_the_max_only_model():
    """Each phase alone fits; the simultaneous sum must refuse/shrink.

    stack(100+400) + prefetch(0) = 500 with margin 15% -> 575 <= 1000.
    Adding a concurrent producer owner of 600 -> 1100 with margin -> 1265 > 1000,
    so no candidate is feasible at prefetch 0: the honest model refuses instead
    of publishing a max-only plan.
    """
    without = zs.enumerate_candidates(_ctx())
    assert any(c.feasible for c in without)
    with_owner = zs.enumerate_candidates(_ctx(concurrent_producer_bytes=600))
    assert not any(c.feasible for c in with_owner)
    reasons = {c.reason_code for c in with_owner if not c.feasible}
    assert reasons == {zs.REASON_REFUSE_HOST_RAM}


def test_concurrent_producer_is_added_not_maxed():
    plain = _ctx(per_unit_stack_bytes=800, per_unit_prefetch_bytes=0,
                 prefetch_ladder=(0,))
    owner = _ctx(per_unit_stack_bytes=800, per_unit_prefetch_bytes=0,
                 prefetch_ladder=(0,), concurrent_producer_bytes=250)
    def host(ctx):
        return next(
            c.host_peak_bytes for c in zs.enumerate_candidates(ctx) if c.feasible
        ) if any(c.feasible for c in zs.enumerate_candidates(ctx)) else max(
            c.host_peak_bytes for c in zs.enumerate_candidates(ctx)
        )
    assert host(owner) - host(plain) == 250


def test_n341_and_10k_descriptor_accounting():
    """Bounded resident credit is N-independent; disk scales with N."""
    px = 1625 * 1415
    resident_341 = zz._r55_prefetch_resident_bytes(
        storage_profile="bounded", per_unit_cache_bytes=1, worst_patch_px=px
    )
    resident_10k = zz._r55_prefetch_resident_bytes(
        storage_profile="bounded", per_unit_cache_bytes=1, worst_patch_px=px
    )
    assert resident_341 == resident_10k == px * 13
    cache_341 = zz._r55_cache_unit_bytes(px, 341)
    cache_10k = zz._r55_cache_unit_bytes(px, 10_000)
    assert cache_10k > cache_341 * 20  # disk scales with N
    disk = zs.disk_residency_bytes(
        units_in_flight=1,
        prefetch_units=1,
        per_unit_cache_bytes=cache_10k,
        per_unit_prefetch_bytes=cache_10k,
    )
    assert disk == 2 * cache_10k
    # Tight disk refuses the 10k corpus before any staging/plan admission.
    tight = _ctx(
        units_total=10_000,
        n_frames=10_000,
        disk_free_bytes=cache_10k,  # half = one cache, need two at prefetch1
        per_unit_cache_bytes=cache_10k,
        per_unit_prefetch_disk_bytes=cache_10k,
    )
    candidates = zs.enumerate_candidates(tight)
    assert any(
        c.reason_code == zs.REASON_REFUSE_DISK
        for c in candidates
        if not c.feasible
    )


# ---------------------------------------------------------------------------
# F3 — campaign gates reject null / any-arm evidence
# ---------------------------------------------------------------------------

def _campaign_module():
    import importlib.util  # noqa: PLC0415

    path = (
        zz.__file__.rsplit("/src/", 1)[0]
        + "/tools/zegrid_r56_prefetch_campaign.py"
    )
    import pathlib  # noqa: PLC0415

    tool = pathlib.Path(zz.__file__).resolve().parents[2] / "tools" / (
        "zegrid_r56_prefetch_campaign.py"
    )
    spec = importlib.util.spec_from_file_location("r56_campaign_tool", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arm(**over):
    base = {
        "pid": 1,
        "error": None,
        "outputs": {"mosaic_grid.fits": "a", "mosaic_grid_coverage.fits": "b"},
        "complete_cells": 4,
        "incomplete_cells": 0,
        "frame_ids": ["f1", "f2"],
        "n_frames": 2,
        "layout": [2, 2],
        "normalization": "sky_mean",
        "mapping": {"backend": "gpu"},
        "gpu_execution": {"qualified": True},
        "gpu_fallback": None,
        "run_policy": {"prefetch_units": 1},
        "overlap": {"overlap_proved": True, "intersection_s": 1.0},
        "observation": {
            "process_tree_rss_peak_bytes": 1,
            "vram_samples": 1,
        },
    }
    base.update(over)
    return base


def test_campaign_rejects_null_equality_and_any_arm_overlap():
    mod = _campaign_module()
    null_frames = _arm(frame_ids=None)
    control = _arm(pid=2, run_policy={"prefetch_units": 0})
    gates = mod.evaluate(null_frames, _arm())
    assert gates["frames_identical"] is False
    assert gates["all_passed"] is False

    # Control-only overlap must NOT satisfy the candidate gate.
    control_overlap = _arm(pid=2, run_policy={"prefetch_units": 0})
    candidate_serial = _arm(
        overlap={"overlap_proved": False, "intersection_s": 0.0}
    )
    gates = mod.evaluate(control_overlap, candidate_serial)
    assert gates["candidate_overlap_proved"] is False
    assert gates["all_passed"] is False

    # Candidate must actually run prefetch>=1.
    gates = mod.evaluate(control, _arm(run_policy={"prefetch_units": 0}))
    assert gates["candidate_gpu_prefetch_ge1"] is False

    # Missing digest on either arm fails the scientific gate.
    gates = mod.evaluate(
        _arm(pid=2, outputs={"mosaic_grid.fits": None,
                             "mosaic_grid_coverage.fits": "b"}),
        _arm(),
    )
    assert gates["scientific_outputs_identical"] is False
    assert gates["all_passed"] is False


def test_campaign_accepts_a_complete_candidate_pair():
    mod = _campaign_module()
    gates = mod.evaluate(_arm(pid=2, run_policy={"prefetch_units": 0}), _arm())
    assert gates["all_passed"] is True, gates
