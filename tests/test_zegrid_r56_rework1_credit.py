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

    def before(task, ledger, cancel_event):
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

    # UNIFIED PROJECTION rule: shared + added + (owner if p>=1) + p*resident.
    projected_zero = rcontract.PlanEnvelope(
        aggregate_bytes=1000,
        aggregate_rule=rcontract.AGGREGATE_RULE_PROJECTED_HOST_PEAK,
        prefetch_units=0,
        producer_owner_bytes=600,
        prefetch_resident_per_slot_bytes=10,
        **base,
    )
    rcontract.validate_envelope(projected_zero)  # p=0: owner NOT charged
    projected_one = rcontract.PlanEnvelope(
        aggregate_bytes=1000 + 600 + 10,
        aggregate_rule=rcontract.AGGREGATE_RULE_PROJECTED_HOST_PEAK,
        prefetch_units=1,
        producer_owner_bytes=600,
        prefetch_resident_per_slot_bytes=10,
        **base,
    )
    rcontract.validate_envelope(projected_one)
    with pytest.raises(rcontract.InvalidEnvelope):
        rcontract.validate_envelope(
            rcontract.PlanEnvelope(
                aggregate_bytes=1000 + 600,
                aggregate_rule=rcontract.AGGREGATE_RULE_PROJECTED_HOST_PEAK,
                prefetch_units=1,
                producer_owner_bytes=600,
                prefetch_resident_per_slot_bytes=10,
                **base,
            )
        )


def test_projected_host_peak_is_one_function_for_every_depth():
    """The published projection: serial at p=0, one owner + resident at p>=1."""
    shared, added, owner, resident = 340 * MIB, 900 * MIB, 200 * MIB, 7 * MIB
    assert rcontract.projected_host_peak_bytes(
        shared, added, owner, 0, resident
    ) == shared + added
    for p in (1, 2, 3, 4):
        assert rcontract.projected_host_peak_bytes(
            shared, added, owner, p, resident
        ) == shared + added + owner + p * resident
    # Stack is NEVER re-billed per slot and the owner is charged exactly once.
    assert rcontract.projected_host_peak_bytes(
        shared, added, owner, 2, resident
    ) - rcontract.projected_host_peak_bytes(
        shared, added, owner, 1, resident
    ) == resident
    for bad in (-1, True, 1.5, None, "1"):
        with pytest.raises(rcontract.RamContractError):
            rcontract.projected_host_peak_bytes(
                shared, added, owner, bad, resident
            )
    with pytest.raises(rcontract.RamContractError):
        rcontract.projected_host_peak_bytes(shared, -1, owner, 0, resident)


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


def test_p0_serial_does_not_charge_a_concurrent_producer():
    """A prefetch-0 plan is SERIAL: the producer is blocked, so no owner.

    stack = shared(100) + per_unit(400) = 500 with the 15 % margin = 575 <=
    1000. An owner of 600 is declared but must NOT be charged at p == 0.
    """
    def host(ctx):
        candidates = zs.enumerate_candidates(ctx)
        return next(
            c.host_peak_bytes for c in candidates
            if c.feasible
        ) if any(c.feasible for c in candidates) else max(
            c.host_peak_bytes for c in candidates
        )

    plain = _ctx(
        per_unit_stack_bytes=400, prefetch_ladder=(0,), units_total=1
    )
    owner = _ctx(
        per_unit_stack_bytes=400,
        prefetch_ladder=(0,),
        units_total=1,
        concurrent_producer_bytes=600,
    )
    assert host(plain) == 100 + 400
    assert host(owner) == 100 + 400
    assert all(c.feasible for c in zs.enumerate_candidates(owner))


def test_p1_charges_exactly_one_producer_and_the_resident_credit():
    """At p >= 1 the producer genuinely overlaps and is charged ONCE.

    stack(100+400) + owner(600) + 1 * resident(10) = 1110 with 15 % margin ->
    1277 > 1000, so the overlap candidate is honestly refused instead of being
    approved from a max-only model.
    """
    def host(ctx):
        candidates = zs.enumerate_candidates(ctx)
        return max(c.host_peak_bytes for c in candidates)

    plain = _ctx(
        per_unit_stack_bytes=400,
        per_unit_prefetch_bytes=10,
        prefetch_ladder=(1,),
        units_total=1,
    )
    owner = _ctx(
        per_unit_stack_bytes=400,
        per_unit_prefetch_bytes=10,
        prefetch_ladder=(1,),
        units_total=1,
        concurrent_producer_bytes=600,
    )
    assert host(plain) == 100 + 400 + 1 * 10
    assert host(owner) - host(plain) == 600
    # Each phase alone fits (500*1.15=575); the overlap does NOT (1277 > 1000).
    assert any(c.feasible for c in zs.enumerate_candidates(plain))
    with_owner = zs.enumerate_candidates(owner)
    assert not any(c.feasible for c in with_owner)
    assert {c.reason_code for c in with_owner} == {zs.REASON_REFUSE_HOST_RAM}


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
        "final_backend": "gpu",
        "consumed_cells": 4,
        "complete_cell_count": 4,
        "cells_complete": 4,
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


def test_campaign_adapter_reads_the_real_product_manifest_shape():
    """Integration on the FORME RÉELLE of a successful Windows manifest.

    source: /tmp/r56_big/ab_pinned/arm_candidate/zegrid_manifest.json
    (complete_cells list of 6, incomplete_cells [], frame_ids 12,
    r55_scheduler.run.final_execution.backend == "gpu",
    6 completed gpu_runtime_units, 6 cells with status "complete").
    """
    import json  # noqa: PLC0415
    import pathlib  # noqa: PLC0415

    manifest_path = pathlib.Path(
        "/tmp/r56_big/ab_pinned/arm_candidate/zegrid_manifest.json"
    )
    if not manifest_path.is_file():
        pytest.skip(f"real product manifest absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    mod = _campaign_module()
    fields = mod._arm_manifest_fields(manifest)
    assert fields["final_backend"] == "gpu"
    assert fields["consumed_cells"] == 6
    assert fields["complete_cell_count"] == 6
    assert fields["cells_complete"] == 6
    assert fields["gpu_units_total"] == 6

    arm = _arm(**fields)
    arm["complete_cells"] = manifest["complete_cells"]
    arm["incomplete_cells"] = manifest["incomplete_cells"]
    arm["frame_ids"] = manifest["frame_ids"]
    assert mod._arm_complete(arm) is True
    assert mod._arm_gpu_prefetch(arm) is True

    # A plan that qualified but never executed must NOT qualify.
    not_executed = dict(arm)
    not_executed["consumed_cells"] = 0
    assert mod._arm_complete(not_executed) is False
    assert mod._arm_gpu_prefetch(not_executed) is False

    # A CPU fallback must NOT pass the GPU gate even with complete cells.
    cpu_fallback = dict(arm)
    cpu_fallback["final_backend"] = "cpu"
    assert mod._arm_complete(cpu_fallback) is True
    assert mod._arm_gpu_prefetch(cpu_fallback) is False

    # complete_cells list longer/shorter than the consumed units is refused.
    mismatched = dict(arm)
    mismatched["complete_cells"] = list(manifest["complete_cells"])[:-1]
    assert mod._arm_complete(mismatched) is False


# ---------------------------------------------------------------------------
# F4 adversarials — real pipeline behaviour (wait vs refuse vs cancel)
# ---------------------------------------------------------------------------

def _disk_hook(volume_bytes: int, *, external_delay=None, external_bytes=0):
    """Fresh-volume hook: capacity = (fresh_free + own_retained) // 2."""

    def _hook(task, ledger, cancel_event):
        own = int(ledger.snapshot()["disk_used_bytes"])
        used_external = 0
        if external_delay is not None and int(task) >= int(external_delay):
            used_external = int(external_bytes)
        free = max(0, int(volume_bytes) - own - used_external)
        ledger.reduce_budget("disk", (free + own) // 2)

    return _hook


def test_full_slots_with_unchanged_volume_wait_then_proceed():
    """Own saturation is NOT a refusal: it waits for the consumer release.

    Volume 400 B, policy capacity 200 B, ONE extra slot. The producer fills the
    queue (200 B), then the third pre-acquisition would exceed 200 B: the ledger
    must WAIT (unchanged volume) until the consumer releases the first file.
    """
    volume = 400
    produced = []
    released = []

    def produce(task):
        produced.append(int(task))
        return {"task": int(task), "bytes": 100}

    def consume(payload):
        return payload["task"]

    def release(payload):
        released.append(payload["task"])

    pipe = zpar.BoundedPipeline(
        tasks=[0, 1, 2],
        produce=produce,
        consume=consume,
        release=release,
        prefetch_units=1,
        host_budget_bytes=1000,
        device_budget_bytes=10,
        disk_budget_bytes=200,
        disk_enabled=True,
        host_bytes=lambda _p: 10,
        device_bytes=lambda _p: 0,
        disk_bytes=lambda _p: 100,
        host_reserve_bytes=lambda _t: 10,
        disk_reserve_bytes=lambda _t: 100,
        before_produce=_disk_hook(volume),
    )
    assert pipe.run() == [0, 1, 2]
    ledger = pipe.telemetry()["ledger"]
    assert ledger["balanced"] is True
    assert ledger["credit_wait_events"] >= 1
    assert sorted(produced) == [0, 1, 2]
    assert sorted(released) == [0, 1, 2]
    assert ledger["disk_peak_bytes"] == 200  # never exceeded the policy cap


def test_external_volume_decline_refuses_before_staging():
    """An EXTERNAL writer shrinks the fresh volume -> refuse (no wait forever)."""
    volume = 400
    produced = []

    def produce(task):
        produced.append(int(task))
        return {"task": int(task), "bytes": 100}

    pipe = zpar.BoundedPipeline(
        tasks=[0, 1, 2],
        produce=produce,
        consume=lambda p: p["task"],
        release=lambda _p: None,
        prefetch_units=1,
        host_budget_bytes=1000,
        device_budget_bytes=10,
        disk_budget_bytes=200,
        disk_enabled=True,
        host_bytes=lambda _p: 10,
        device_bytes=lambda _p: 0,
        disk_bytes=lambda _p: 100,
        host_reserve_bytes=lambda _t: 10,
        disk_reserve_bytes=lambda _t: 100,
        # From task 1 onward an external writer leaves only 200 B of volume:
        # capacity = (200 - own + own)//2 = 100 while a single request is 100
        # and the previous own=100 -> 200 > 100 -> refusal (not a wait).
        before_produce=_disk_hook(
            volume, external_delay=1, external_bytes=390
        ),
    )
    with pytest.raises(zpar.PipelineAccountingError):
        pipe.run()
    assert produced == [0]  # the second unit never allocated
    ledger = pipe.telemetry()["ledger"]
    assert ledger["balanced"] is True


def test_capacity_shrink_during_disk_wait_fails_closed():
    """A shrink WHILE waiting must refuse instead of blocking forever."""
    volume = 400
    holder = {}

    def before(task, ledger, cancel_event):
        own = int(ledger.snapshot()["disk_used_bytes"])
        ledger.reduce_budget("disk", (volume - own + own) // 2)
        if int(task) == 2:
            threading.Timer(
                0.2, lambda: ledger.reduce_budget("disk", 50)
            ).start()

    def consume(payload):
        # Hold the consumed unit so both queued slots stay leased while the
        # producer waits for the third disk credit.
        assert holder["p"]._cancel.wait(2.0)
        return payload["task"]

    def produce(task):
        return {"task": int(task), "bytes": 100}

    pipe = zpar.BoundedPipeline(
        tasks=[0, 1, 2],
        produce=produce,
        consume=consume,
        release=lambda _p: None,
        prefetch_units=1,
        host_budget_bytes=1000,
        device_budget_bytes=10,
        disk_budget_bytes=200,
        disk_enabled=True,
        host_bytes=lambda _p: 10,
        device_bytes=lambda _p: 0,
        disk_bytes=lambda _p: 100,
        host_reserve_bytes=lambda _t: 10,
        disk_reserve_bytes=lambda _t: 100,
        before_produce=before,
        stall_timeout_s=None,
    )
    holder["p"] = pipe
    with pytest.raises(zpar.PipelineAccountingError) as excinfo:
        pipe.run()
    assert "shrank" in str(excinfo.value)
    ledger = pipe.telemetry()["ledger"]
    assert ledger["budget_reductions"][-1]["to_bytes"] == 50
    assert ledger["balanced"] is True


def test_cancellation_during_disk_wait_is_clean():
    volume = 400
    holder = {}

    def before(task, ledger, cancel_event):
        own = int(ledger.snapshot()["disk_used_bytes"])
        ledger.reduce_budget("disk", (volume - own + own) // 2)
        if int(task) == 2:
            threading.Timer(
                0.2, lambda: holder["p"].cancel("test_cancel_disk_wait")
            ).start()

    def consume(payload):
        # Hold the consumed unit so both queued slots stay leased while the
        # producer waits for the third disk credit.
        holder["p"]._cancel.wait(2.0)
        return payload["task"]

    pipe = zpar.BoundedPipeline(
        tasks=[0, 1, 2],
        produce=lambda t: {"task": int(t), "bytes": 100},
        consume=consume,
        release=lambda _p: None,
        prefetch_units=1,
        host_budget_bytes=1000,
        device_budget_bytes=10,
        disk_budget_bytes=200,
        disk_enabled=True,
        host_bytes=lambda _p: 10,
        device_bytes=lambda _p: 0,
        disk_bytes=lambda _p: 100,
        host_reserve_bytes=lambda _t: 10,
        disk_reserve_bytes=lambda _t: 100,
        before_produce=before,
    )
    holder["p"] = pipe
    with pytest.raises(zpar.PipelineCancelled):
        pipe.run()
    ledger = pipe.telemetry()["ledger"]
    assert ledger["balanced"] is True


def test_enabled_zero_disk_capacity_refuses_a_nonzero_request():
    produced = []

    def produce(task):
        produced.append(int(task))
        return {"task": int(task), "bytes": 100}

    pipe = zpar.BoundedPipeline(
        tasks=[0],
        produce=produce,
        consume=lambda p: p["task"],
        release=lambda _p: None,
        prefetch_units=0,
        host_budget_bytes=100,
        device_budget_bytes=10,
        disk_budget_bytes=0,
        disk_enabled=True,  # explicit: the dimension is ON with capacity 0
        host_bytes=lambda _p: 10,
        device_bytes=lambda _p: 0,
        disk_bytes=lambda _p: 100,
        host_reserve_bytes=lambda _t: 10,
        disk_reserve_bytes=lambda _t: 100,
    )
    with pytest.raises(zpar.PipelineAccountingError):
        pipe.run()
    assert produced == []


def test_partial_produce_cleanup_deletes_real_files_before_credit_release(
    tmp_path,
):
    partial = tmp_path / "partial_cache.bin"
    observed = {}
    holder = {}

    def produce(_task):
        partial.write_bytes(b"x" * 100)
        raise RuntimeError("cache build exploded mid-write")

    def cleanup(_task, _payload):
        # The partial file is still there AND its credit is STILL HELD: cleanup
        # must delete BEFORE the lease is released, otherwise the ledger frees
        # space while the bytes are still on the volume.
        observed["exists"] = partial.exists()
        observed["disk_used"] = int(
            holder["p"].telemetry()["ledger"]["disk_used_bytes"]
        )
        partial.unlink()

    pipe = zpar.BoundedPipeline(
        tasks=[0],
        produce=produce,
        consume=lambda p: p,
        release=lambda _p: None,
        prefetch_units=0,
        host_budget_bytes=100,
        device_budget_bytes=10,
        disk_budget_bytes=100,
        disk_enabled=True,
        cleanup_task=cleanup,
        host_bytes=lambda _p: 10,
        device_bytes=lambda _p: 0,
        disk_bytes=lambda _p: 100,
        host_reserve_bytes=lambda _t: 10,
        disk_reserve_bytes=lambda _t: 100,
    )
    holder["p"] = pipe
    with pytest.raises(zpar.PipelineError) as excinfo:
        pipe.run()
    # PRIMARY cause preserved (not replaced by an accounting error).
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "cache build exploded" in str(excinfo.value)
    assert observed == {"exists": True, "disk_used": 100}
    assert not partial.exists()
    ledger = pipe.telemetry()["ledger"]
    assert ledger["balanced"] is True
    assert ledger["disk_used_bytes"] == 0
