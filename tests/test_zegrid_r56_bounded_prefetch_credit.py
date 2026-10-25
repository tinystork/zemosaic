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
