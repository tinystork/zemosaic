"""ZM-ZEGRID-R56-BOUNDED-PREFETCH — profile-aware prefetch credit + overlap proof.

Root cause pinned here with the REAL Caldwell LP341 numbers extracted from the
Windows R56_SKY_MEAN_LP manifest (halo128, 3x3, sky_mean, GPU tile192, bounded):

* the aligned cache of ONE cell is exactly ``10193216671`` B
  (``341`` frames x (``1625*1415`` px x 13 B + 256 B)). Before this mission a
  prefetch slot was billed that FULL cache size as if it were RESIDENT host RAM,
  so ``prefetch + 1`` slots at tile192 exceeded ``B_plan=14687984025`` and the
  planner could only admit ``prefetch=0``;
* under the ``bounded`` provider no mapping outlives an access, so a queued slot
  retains only its metadata; the largest transient owned copy for one access is
  one worst-patch RGB float32 plane + its bool support plane = ``29891875`` B.
  The exact cache bytes still have to be accounted — but as DISK, not RAM.

The tests below prove that separation in PLAN, ADMIT and RUN, and prove real
producer/consumer overlap from observed work intervals (never from queue depth).
"""

from __future__ import annotations

import dataclasses
import threading
import time

import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import gpu as zgpu
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import ram_contract as rcontract
from zemosaic.core.zegrid import scheduler as zs

MIB = 2 ** 20
GIB = 2 ** 30

# --- Caldwell LP341 witness (read-only extraction from the Windows run) ------
LP_N_FRAMES = 341
LP_WORST_PATCH_HW = (1625, 1415)
LP_WORST_PATCH_PX = LP_WORST_PATCH_HW[0] * LP_WORST_PATCH_HW[1]
LP_PER_UNIT_CACHE = 10193216671
LP_B_PLAN = 14687984025
LP_RAW = 18359980032
LP_VRAM_FREE = 7463763968
LP_VRAM_BUDGET = 5224634777
LP_GPU_TILES = (192, 128, 112, 96, 80, 64, 48, 32)
LP_SHARED = rcontract.PROCESS_BASELINE_BYTES


def _lp_device_peaks():
    return {
        int(t): int(
            zgpu.gpu_tile_workspace_bytes(
                LP_N_FRAMES,
                int(t) + 2 * int(zgpu.GPU_TAPER_HALO_PX),
                int(t) + 2 * int(zgpu.GPU_TAPER_HALO_PX),
                3,
            )
        )
        for t in LP_GPU_TILES
    }


def _lp_gpu_context(*, storage_profile="bounded", bounded_credit=True):
    bounds = {
        int(t): int(
            zz._profile_stream_bytes(
                LP_N_FRAMES, LP_WORST_PATCH_HW, int(t), storage_profile
            )
        )
        for t in LP_GPU_TILES
    }
    host_prefetch = (
        zz._r55_prefetch_resident_bytes(
            storage_profile=storage_profile,
            per_unit_cache_bytes=LP_PER_UNIT_CACHE,
            worst_patch_px=LP_WORST_PATCH_PX,
            channels=3,
        )
        if bounded_credit
        else LP_PER_UNIT_CACHE
    )
    return zs.PlanContext(
        phase=zs.PHASE_CELL_STACK_STREAM,
        backend=zs.BACKEND_GPU,
        budget_bytes=LP_B_PLAN,
        raw_available_bytes=LP_RAW,
        factor_num=4,
        factor_den=5,
        cpu_logical=16,
        units_total=9,
        shared_bytes=LP_SHARED,
        per_unit_stack_bytes=max(bounds.values()),
        per_unit_cache_bytes=LP_PER_UNIT_CACHE,
        per_unit_prefetch_bytes=int(host_prefetch),
        per_unit_prefetch_disk_bytes=(
            LP_PER_UNIT_CACHE if bounded_credit else None
        ),
        disk_free_bytes=200 * GIB,
        layout=(3, 3),
        refinement_rank=1,
        patch_hw=LP_WORST_PATCH_HW,
        halo_px=128,
        channels=3,
        n_frames=LP_N_FRAMES,
        tile_ladder=LP_GPU_TILES,
        prefetch_ladder=(0, 1, 2, 3, 4),
        workers_ladder=zs.WORKERS_LADDER,
        per_tile_stack_bytes=tuple(
            (int(t), max(1, int(bounds[int(t)]) - int(LP_SHARED)))
            for t in LP_GPU_TILES
        ),
        storage_profile=storage_profile,
    )


def _lp_gpu_facts():
    peaks = _lp_device_peaks()
    return zs.GpuFacts(
        requested=True,
        available=True,
        parity_qualified=True,
        vram_total_bytes=8589410304,
        vram_free_bytes=LP_VRAM_FREE,
        vram_budget_bytes=LP_VRAM_BUDGET,
        device_name="NVIDIA GeForce RTX 3070 Laptop GPU",
        device_peak_bytes_by_tile=tuple(sorted(peaks.items())),
    )


def _gpu_candidates(context):
    cpu = zs.PlanContext(
        phase=zs.PHASE_CELL_STACK_STREAM,
        backend=zs.BACKEND_CPU,
        budget_bytes=LP_B_PLAN,
        raw_available_bytes=LP_RAW,
        factor_num=4,
        factor_den=5,
        cpu_logical=16,
        units_total=9,
        shared_bytes=LP_SHARED,
        per_unit_stack_bytes=max(1, LP_PER_UNIT_CACHE // 4),
        per_unit_cache_bytes=LP_PER_UNIT_CACHE,
        per_unit_prefetch_bytes=0,
        disk_free_bytes=200 * GIB,
        layout=(3, 3),
        refinement_rank=1,
        patch_hw=LP_WORST_PATCH_HW,
        halo_px=128,
        channels=3,
        n_frames=LP_N_FRAMES,
        tile_ladder=zz.CPU_STREAMING_TILE_LADDER,
        prefetch_ladder=(0,),
        workers_ladder=zs.WORKERS_LADDER,
    )
    joint = zs.JointPlanInput(
        contexts=(cpu, context), gpu_facts=_lp_gpu_facts()
    )
    return tuple(
        c
        for c in zs.enumerate_joint_candidates(joint)
        if c.backend == zs.BACKEND_GPU
    )


# ---------------------------------------------------------------------------
# 1. The published LP341 numbers are reproduced exactly.
# ---------------------------------------------------------------------------

def test_lp341_cache_credit_matches_the_published_windows_manifest():
    assert (
        zz._r55_cache_unit_bytes(LP_WORST_PATCH_PX, LP_N_FRAMES) == LP_PER_UNIT_CACHE
    )
    # Sanity: the reconstructed geometry is the one that produced that byte count.
    assert LP_WORST_PATCH_PX == 2299375


def test_prefetch_resident_credit_is_profile_aware_and_never_zero():
    bounded = zz._r55_prefetch_resident_bytes(
        storage_profile="bounded",
        per_unit_cache_bytes=LP_PER_UNIT_CACHE,
        worst_patch_px=LP_WORST_PATCH_PX,
        channels=3,
    )
    dense = zz._r55_prefetch_resident_bytes(
        storage_profile="dense",
        per_unit_cache_bytes=LP_PER_UNIT_CACHE,
        worst_patch_px=LP_WORST_PATCH_PX,
        channels=3,
    )
    # One worst-patch RGB float32 plane + one bool support plane.
    assert bounded == LP_WORST_PATCH_PX * (3 * 4 + 1) == 29891875
    assert 0 < bounded < dense
    # Dense keeps the historical value byte for byte.
    assert dense == LP_PER_UNIT_CACHE


def test_prefetch_resident_credit_rejects_unknown_profiles():
    with pytest.raises(ValueError):
        zz._r55_prefetch_resident_bytes(
            storage_profile="nonsense",
            per_unit_cache_bytes=1,
            worst_patch_px=1,
        )


# ---------------------------------------------------------------------------
# 2. PLAN: the old dense billing refuses prefetch>=1, the new model admits it.
# ---------------------------------------------------------------------------

def test_legacy_cache_billing_refuses_prefetch_at_lp341():
    """Root cause: SAME bounded stack bound, ONLY the prefetch credit differs.

    Using the bounded per-tile bounds isolates the prefetch term: with the
    legacy billing (the full aligned cache as resident RAM) prefetch>=1 breaks
    the host budget, while prefetch=0 is feasible. The new profile-aware credit
    (next test) admits it.
    """
    context = _lp_gpu_context(storage_profile="bounded", bounded_credit=False)
    candidates = _gpu_candidates(context)
    assert candidates
    # The published Windows run: tile192, prefetch0, bounded.
    best = min(
        (c for c in candidates if c.feasible),
        key=lambda c: c.ordering_key(),
    )
    assert int(best.tile_px) == 192
    assert int(best.prefetch_units) == 0
    # At the SELECTED tile the legacy billing cannot afford even one slot.
    for candidate in candidates:
        if int(candidate.tile_px) == 192 and int(candidate.prefetch_units) >= 1:
            assert candidate.feasible is False
            assert candidate.reason_code == zs.REASON_REFUSE_HOST_RAM


def test_bounded_credit_admits_prefetch_at_least_one_at_lp341():
    context = _lp_gpu_context(storage_profile="bounded")
    feasible = [c for c in _gpu_candidates(context) if c.feasible]
    assert feasible, "no feasible GPU candidate under the bounded credit"
    with_prefetch = [
        c for c in feasible if int(c.prefetch_units) >= 1
    ]
    assert with_prefetch, (
        "bounded profile must admit at least one prefetch slot; "
        f"max admitted prefetch={max(int(c.prefetch_units) for c in feasible)}"
    )
    # The best feasible candidate is selected by (tile, k, prefetch, workers).
    best = min(feasible, key=lambda c: c.ordering_key())
    assert int(best.prefetch_units) >= 1


def test_plan_separates_disk_residency_from_host_residency():
    """The disk term uses the EXACT cache bytes even when host credit is small."""
    disk = zs.disk_residency_bytes(
        units_in_flight=1,
        prefetch_units=2,
        per_unit_cache_bytes=LP_PER_UNIT_CACHE,
        per_unit_prefetch_bytes=LP_PER_UNIT_CACHE,
    )
    assert disk == (1 + 2) * LP_PER_UNIT_CACHE
    # The host term for the same candidate never sees the disk bytes.
    context = _lp_gpu_context(storage_profile="bounded")
    assert context.prefetch_disk_bytes == LP_PER_UNIT_CACHE
    assert context.per_unit_prefetch_bytes == 29891875


def test_bounded_plan_refuses_on_disk_even_when_host_is_comfortable():
    context = _lp_gpu_context(storage_profile="bounded")
    tight = zs.PlanContext(
        **{
            **{
                field: getattr(context, field)
                for field in context.__dataclass_fields__
            },
            "disk_free_bytes": 4 * LP_PER_UNIT_CACHE,  # half = 2 caches
        }
    )
    candidates = _gpu_candidates(tight)
    assert candidates
    # Any prefetch >= 1 needs (1 + p) caches > disk cap (2 caches).
    for candidate in candidates:
        if int(candidate.prefetch_units) >= 2:
            assert candidate.feasible is False
            assert candidate.reason_code == zs.REASON_REFUSE_DISK
    feasible = [c for c in candidates if c.feasible]
    assert feasible
    assert max(int(c.prefetch_units) for c in feasible) <= 1


def test_default_context_still_conflates_prefetch_when_disk_field_absent():
    """Backwards compatibility: no disk field -> historical single-number model."""
    context = _lp_gpu_context(storage_profile="bounded", bounded_credit=False)
    assert context.per_unit_prefetch_disk_bytes is None
    assert context.prefetch_disk_bytes == context.per_unit_prefetch_bytes


# ---------------------------------------------------------------------------
# 3. RUN: the bounded pipeline books disk separately and stays balanced.
# ---------------------------------------------------------------------------

def _pipeline(*, prefetch, disk_budget, host_per=10, disk_per=1000, sleep=0.0):
    released = []

    def produce(task):
        if sleep:
            time.sleep(sleep)
        return {"task": task}

    def consume(payload):
        if sleep:
            time.sleep(sleep)
        return payload["task"] * 2

    def release(payload):
        released.append(payload["task"])

    return (
        zpar.BoundedPipeline(
            tasks=list(range(8)),
            produce=produce,
            consume=consume,
            release=release,
            prefetch_units=prefetch,
            host_budget_bytes=host_per * 4,
            device_budget_bytes=host_per * 4,
            disk_budget_bytes=disk_budget,
            host_bytes=lambda _p: host_per,
            device_bytes=lambda _p: host_per,
            disk_bytes=lambda _p: disk_per,
        ),
        released,
    )


def test_disk_ledger_is_exact_separate_and_balanced():
    pipe, released = _pipeline(prefetch=1, disk_budget=1000 * 3)
    assert pipe.run() == [i * 2 for i in range(8)]
    tel = pipe.telemetry()
    ledger = tel["ledger"]
    assert tel["schema_v2"] == zpar.PIPELINE_SCHEMA_V2
    assert ledger["schema"] == zpar.PIPELINE_SCHEMA_V2
    assert ledger["balanced"] is True
    assert ledger["host_leases"] == 0
    assert ledger["disk_leases"] == 0
    assert ledger["host_peak_bytes"] == 10 * 3  # queue_max=2 + the consumed unit
    assert ledger["host_peak_bytes"] <= 10 * (1 + 2)
    assert ledger["disk_peak_bytes"] == 1000 * 3  # queue_max=2 + the consumed unit
    assert ledger["disk_peak_bytes"] > ledger["host_peak_bytes"]
    assert ledger["dimensions"]["disk"]["role"] == "on_disk_payload_bytes"
    assert sorted(released) == list(range(8))


def test_disk_credit_fails_closed_on_an_impossible_request():
    """A payload larger than the whole disk budget can never fit: fail closed."""
    pipe, released = _pipeline(prefetch=0, disk_budget=1000, disk_per=2000)
    with pytest.raises(zpar.PipelineError):
        pipe.run()
    tel = pipe.telemetry()
    assert tel["ledger"]["balanced"] is True
    assert tel["ledger"]["disk_used_bytes"] == 0
    # The refused payload was owned then released exactly once.
    assert tel["units_released"] == tel["units_produced"]


def test_zero_disk_budget_keeps_the_dimension_inert():
    pipe, _ = _pipeline(prefetch=1, disk_budget=0)
    pipe.run()
    ledger = pipe.telemetry()["ledger"]
    assert ledger["disk_budget_bytes"] == 0
    assert ledger["disk_used_bytes"] == 0
    assert ledger["disk_peak_bytes"] == 0
    assert ledger["balanced"] is True


# ---------------------------------------------------------------------------
# 4. Observed overlap witness (real intervals, not queue capacity).
# ---------------------------------------------------------------------------

def _overlap_pipeline(*, producer_sleep, consumer_sleep, gate=None):
    def produce(task):
        if gate is not None and task == 1:
            gate.set()
        time.sleep(producer_sleep)
        return {"task": task}

    def consume(payload):
        if gate is not None:
            gate.wait(2.0)
        time.sleep(consumer_sleep)
        return payload["task"]

    return zpar.BoundedPipeline(
        tasks=list(range(4)),
        produce=produce,
        consume=consume,
        release=lambda _p: None,
        prefetch_units=1,
        host_budget_bytes=1024,
        device_budget_bytes=1024,
        host_bytes=lambda _p: 1,
        device_bytes=lambda _p: 1,
    )


def test_overlap_witness_proves_concurrent_work():
    gate = threading.Event()
    pipe = _overlap_pipeline(producer_sleep=0.05, consumer_sleep=0.05, gate=gate)
    pipe.run()
    witness = pipe.telemetry()["observed_overlap"]
    assert witness["producer_work_s"] > 0
    assert witness["consumer_work_s"] > 0
    assert witness["intersection_s"] > 0
    assert witness["overlap_units"] >= 1
    assert witness["overlap_proved"] is True
    assert witness["clock"] == "time.monotonic"


def test_no_overlap_claim_when_consumer_never_coincides():
    """Prefetch 0 + a producer that always finishes first: no intersection."""
    pipe = zpar.BoundedPipeline(
        tasks=list(range(3)),
        produce=lambda task: {"task": task},
        consume=lambda payload: time.sleep(0.02) or payload["task"],
        release=lambda _p: None,
        prefetch_units=0,
        host_budget_bytes=1024,
        device_budget_bytes=1024,
        host_bytes=lambda _p: 1,
        device_bytes=lambda _p: 1,
    )
    pipe.run()
    witness = pipe.telemetry()["observed_overlap"]
    # The intervals MAY still touch for a fast producer; what must never happen
    # is a positive claim without a positive intersection.
    assert witness["overlap_proved"] == (witness["intersection_s"] > 0)


# ---------------------------------------------------------------------------
# 4. FAITHFUL LP341 product witness (real entry/decision, no helper shortcut)
# ---------------------------------------------------------------------------
#
# The historical helper built a PlanContext with ``concurrent_producer_bytes=0``,
# so it could not prove the corrected product planner. This witness drives the
# REAL product decision (``_r55_active_facts`` + ``_r55_active_gpu_decide``) with
# the published LP341 facts:
#   * 341 frames, halo 128, 3x3 layout, sky_mean, bounded storage;
#   * worst patch 1625x1415 px (core 1369x1159 + 2*128 halo);
#   * VRAM free 7 463 763 968 B, B_plan 14 687 984 025 B / raw 18 359 980 032 B.
# The geometry is reconstructed from the published patch/halo/layout (the Windows
# corpus is not present on this host); every number below is produced by the
# product code paths, never by a hand-built PlanContext.

LP_CORE_HW = (1369, 1159)
LP_CANVAS_HW = (LP_CORE_HW[1] * 3, LP_CORE_HW[0] * 3)


def _lp341_cell_ctxs():
    from types import SimpleNamespace

    out = []
    core_h, core_w = LP_CORE_HW
    for row in range(3):
        for col in range(3):
            cell = SimpleNamespace(
                cell_id=f"r{row:04d}c{col:04d}",
                core=SimpleNamespace(
                    x0=col * core_w,
                    y0=row * core_h,
                    x1=(col + 1) * core_w,
                    y1=(row + 1) * core_h,
                ),
            )
            patch = SimpleNamespace(
                patch=SimpleNamespace(
                    width=LP_WORST_PATCH_HW[1], height=LP_WORST_PATCH_HW[0]
                ),
                patch_shape_hw=LP_WORST_PATCH_HW,
            )
            mem = SimpleNamespace(
                patch_ids=tuple(f"lp341_{i}" for i in range(LP_N_FRAMES))
            )
            out.append((row, col, cell, patch, mem))
    return tuple(out)


@dataclasses.dataclass
class _LPBundle:
    envelope: object
    shared_bytes: int
    cache_worker_footprint_bytes: int
    gauge_worker_footprint_bytes: int
    gauge_workers: int
    cache_build_workers: int
    layout: dict
    storage_profile: str = "bounded"
    cell_specs: list = dataclasses.field(default_factory=list)
    cell_tasks: list = dataclasses.field(default_factory=list)
    attempts: list = dataclasses.field(default_factory=list)
    cpu_fallback_info: dict = dataclasses.field(default_factory=dict)
    cpu_streaming: dict = dataclasses.field(default_factory=dict)
    cpu_streaming_tile: int = 0
    gpu_tile_size: int | None = None
    gpu_streaming_tile: int | None = None
    gpu_streaming_host_peak_bytes: int | None = None
    gpu_used: bool = False
    gpu_pre_execution_fallback: object = None
    plan: dict = dataclasses.field(default_factory=dict)
    chosen_mode: str = "stream"
    cells_in_flight: int = 1
    cache_workers_per_cell: int = 14
    per_cell_footprint: int = 0


def _lp341_bundle():
    from types import SimpleNamespace

    canvas_px = int(LP_CANVAS_HW[0]) * int(LP_CANVAS_HW[1])
    return _LPBundle(
        envelope=SimpleNamespace(
            raw_available_bytes=int(LP_RAW),
            factor_num=4,
            factor_den=5,
            budget_bytes=int(LP_B_PLAN),
            layout=(3, 3),
            refinement_rank=1,
            pinned=False,
        ),
        shared_bytes=int(LP_SHARED),
        cache_worker_footprint_bytes=int(
            zz.zpar.cache_footprint_bytes(LP_WORST_PATCH_PX)
        ),
        gauge_worker_footprint_bytes=int(
            zz.zpar.gauge_footprint_bytes(canvas_px)
        ),
        gauge_workers=14,
        cache_build_workers=14,
        layout={"nx": 3, "ny": 3},
        storage_profile="bounded",
    )


def _lp341_decide(monkeypatch):
    from types import SimpleNamespace

    from zemosaic.core.zegrid import gpu_readiness as zgr

    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    monkeypatch.setattr(zz.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(
        zz.shutil,
        "disk_usage",
        lambda _p: SimpleNamespace(total=4 * 2 ** 40, used=0, free=4 * 2 ** 40),
    )
    readiness = zgr.GpuReadiness(
        status=zgr.STATUS_USABLE,
        reason_code=zgr.REASON_GPU_OK,
        available=True,
        dependency_declared=True,
        dependency_importable=True,
        cupy_version="model-only",
        device_count=1,
        device_index=0,
        device_name="NVIDIA GeForce RTX 3070 Laptop GPU",
        vram_total_bytes=8589410304,
        vram_free_bytes=int(LP_VRAM_FREE),
    )
    canvas = SimpleNamespace(
        width=int(LP_CANVAS_HW[1]),
        height=int(LP_CANVAS_HW[0]),
        canvas_id="lp341-faithful",
    )
    return zz._r55_active_gpu_decide(
        bundle=_lp341_bundle(),
        cell_ctxs=_lp341_cell_ctxs(),
        canvas=canvas,
        halo_px=128,
        cache_root="/tmp",
        output_dir="/tmp",
        frame_ids=tuple(f"lp341_{i}" for i in range(LP_N_FRAMES)),
        gpu_probe={
            "available": True,
            "device": "NVIDIA GeForce RTX 3070 Laptop GPU",
            "vram_total_bytes": 8589410304,
            "vram_free_bytes": int(LP_VRAM_FREE),
        },
        readiness=readiness,
    )


def test_lp341_faithful_product_witness_is_admitted_with_an_honest_depth(monkeypatch):
    decision = _lp341_decide(monkeypatch)
    chosen = decision.plan_set.chosen
    assert decision.mapping["backend"] == "gpu"
    # Honest depth from the REAL product decision (no forcing).
    assert int(decision.mapping["gpu_streaming_tile"]) == 192
    assert int(decision.mapping["units_in_flight"]) == 1
    assert int(decision.mapping["workers_per_unit"]) == 14
    assert int(decision.mapping["prefetch_units"]) == 2

    facts = decision.facts
    assert int(facts["per_unit_cache"]) == LP_PER_UNIT_CACHE
    assert int(facts["per_unit_prefetch_disk"]) == LP_PER_UNIT_CACHE
    assert int(facts["per_unit_prefetch_host"]) == LP_WORST_PATCH_PX * 13
    added = int(dict(decision.gpu_host_bound_by_tile)[192])
    assert int(facts["concurrent_producer_bytes"]) == max(
        int(
            rcontract.cache_per_cell_bytes(
                rcontract.PROCESS_BASELINE_BYTES,
                int(decision.cache_workers),
                int(zz.zpar.cache_footprint_bytes(LP_WORST_PATCH_PX)),
            )
        ),
        2 * LP_WORST_PATCH_PX * 13,
    )
    # The cache-build owner (a REAL ram_contract path) is the binding owner
    # term; the N-independent normalization bound is NOT binding here.
    owner = int(facts["concurrent_producer_bytes"])
    assert owner == int(
        rcontract.cache_per_cell_bytes(
            rcontract.PROCESS_BASELINE_BYTES,
            int(decision.cache_workers),
            int(zz.zpar.cache_footprint_bytes(LP_WORST_PATCH_PX)),
        )
    )
    assert 2 * LP_WORST_PATCH_PX * 13 < owner
    # Parity PLAN == RUN: the chosen host peak is the unified projection.
    assert int(chosen.host_peak_bytes) == (
        int(zz.rcontract.PROCESS_BASELINE_BYTES)
        + added
        + owner
        + 2 * int(facts["per_unit_prefetch_host"])
    )
    assert int(decision.plan_set.aggregate_host_peak_bytes) <= LP_B_PLAN
    # The aggregate is 95.4 % of B_plan and its limiting term is the GAUGE
    # phase (14 workers x gauge footprint), NOT the stack nor the prefetch.
    assert int(decision.plan_set.aggregate_host_peak_bytes_raw) == (
        int(decision.gauge_workers) * int(facts["gauge_footprint_bytes"])
    )
    assert int(decision.plan_set.aggregate_host_peak_bytes) == 14016615629
    assert LP_B_PLAN - int(decision.plan_set.aggregate_host_peak_bytes) == 671368396
    # p=2 is the declared E2 target, not a memory limit: p+1 and p+4 still fit
    # the host candidate (disk holds 5 slots within the 4 TiB volume).
    assert int(chosen.host_peak_bytes) + (4 - 2) * int(
        facts["per_unit_prefetch_host"]
    ) < LP_B_PLAN


def test_lp341_plan_admit_run_publish_the_same_projected_host_peak(monkeypatch):
    decision = _lp341_decide(monkeypatch)
    mapping = dict(decision.mapping)
    bundle = _lp341_bundle()
    # Apply needs the frozen cell_specs; the geometry is irrelevant to the
    # aggregate arithmetic, so an empty task list is honest here.
    applied = zz._r55_active_gpu_apply(
        bundle, mapping=mapping, decision=decision, stage="preflight"
    )
    chosen = decision.plan_set.chosen
    assert applied.envelope.aggregate_rule == (
        rcontract.AGGREGATE_RULE_PROJECTED_HOST_PEAK
    )
    assert applied.envelope.prefetch_units == 2
    assert int(applied.envelope.aggregate_bytes) == int(chosen.host_peak_bytes)
    runtime = zz._r55_gpu_host_aggregate_for_tile(
        decision, bundle, 192, prefetch_units=2
    )
    assert int(runtime) == int(chosen.host_peak_bytes)
    # The scheduler phase envelope publishes the same value.
    phase_env = next(
        env
        for env in decision.plan_set.phases
        if env.backend == "gpu" and env.phase == "cell_stack_stream"
    )
    assert int(phase_env.estimate.peak_bytes) == int(chosen.host_peak_bytes)
