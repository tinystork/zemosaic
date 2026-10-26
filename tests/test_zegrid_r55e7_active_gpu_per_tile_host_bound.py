"""ZM-ZEGRID-R55-E7 — exact per-tile ACTIVE GPU host bounds + truthful refusal.

The valid E6 2x2/halo128 layout exposed a false ACTIVE GPU preflight refusal:
the product collapsed the whole tile ladder to ONE static host bound (the
maximum over the ladder, driven by tile32) and then reported
``R55_REFUSE_VRAM`` because the backend-level trail only said the GPU context
was eligible.

These tests pin the corrected contract:

* every GPU tile candidate uses its OWN exact integer
  ``_estimate_streaming_bytes(max_n, worst_hw, tile)`` bound;
* the published E6 witness numbers (max_n=66, worst patch 1767x1330,
  B_plan≈7326.9 MiB) produce a legal tile128/K=1/prefetch=1 decision while the
  old ladder-max path refuses every candidate;
* host-only exhaustion reports ``R55_REFUSE_HOST_RAM`` with quantitative
  evidence, while genuine device exhaustion stays ``R55_REFUSE_VRAM``;
* incomplete/extra/bool/float/negative/contradictory per-tile maps fail closed,
  and the CPU/static path and hashes are unchanged when no map is supplied.

No pixel stack and no full mosaic are built anywhere in this module.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import calibration as zcal
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import gpu as zgpu
from zemosaic.core.zegrid import gpu_readiness as zgr
from zemosaic.core.zegrid import resource_envelope as re
from zemosaic.core.zegrid import scheduler as zs

MIB = 2 ** 20
GIB = 2 ** 30
LIGHTS = Path("/home/tristan/M106/lights")

TILES = (128, 112, 96, 80, 64, 48, 32)
WITNESS_MAX_N = 66
WITNESS_PATCH_HW = (1330, 1767)
WITNESS_SHARED_MIB = 340
WITNESS_PREFETCH_MIB = 1775.062
WITNESS_RAW_MIB = 9158.684
WITNESS_B_PLAN_MIB = 7326.947198867798
# Root-cause evidence: the ladder is NON-MONOTONE and tile32 is the maximum.
WITNESS_BOUND_MIB = {
    128: 3588.6142578125,
    112: 3621.0751953125,
    96: 3750.7861328125,
    80: 4015.248046875,
    64: 4466.9619140625,
    48: 5553.42578125,
    32: 8817.140625,
}


def _witness_bounds() -> dict[int, int]:
    return {
        int(tile): int(
            zz._estimate_streaming_bytes(
                WITNESS_MAX_N, WITNESS_PATCH_HW, int(tile)
            )
        )
        for tile in TILES
    }


def _witness_budget() -> tuple[int, int]:
    raw = int(WITNESS_RAW_MIB * MIB)
    return raw, (4 * raw) // 5


def _device_peaks(tiles=TILES) -> dict[int, int]:
    return {
        int(tile): int(
            zgpu.gpu_tile_workspace_bytes(
                WITNESS_MAX_N,
                int(tile) + 2 * int(zgpu.GPU_TAPER_HALO_PX),
                int(tile) + 2 * int(zgpu.GPU_TAPER_HALO_PX),
                3,
            )
        )
        for tile in tiles
    }


def _gpu_facts(*, vram_free=7 * GIB, vram_budget=4 * GIB, peaks=None) -> zs.GpuFacts:
    return zs.GpuFacts(
        requested=True,
        available=True,
        parity_qualified=True,
        vram_total_bytes=8 * GIB,
        vram_free_bytes=int(vram_free),
        vram_budget_bytes=int(vram_budget),
        device_name="fake-rtx3070",
        device_peak_bytes_by_tile=tuple(
            sorted((peaks or _device_peaks()).items())
        ),
    )


def _gpu_context(
    *,
    per_tile: bool,
    static_bound: int | None = None,
    budget: int | None = None,
    raw: int | None = None,
    per_tile_map=None,
    tile_ladder=TILES,
    profile=None,
    phase: str = zs.PHASE_CELL_STACK_STREAM,
    **overrides,
):
    witness_raw, witness_budget = _witness_budget()
    bounds = _witness_bounds()
    data = dict(
        phase=phase,
        backend=zs.BACKEND_GPU,
        budget_bytes=int(witness_budget if budget is None else budget),
        raw_available_bytes=int(
            witness_raw if raw is None else raw
        ),
        factor_num=4,
        factor_den=5,
        cpu_logical=16,
        units_total=4,
        shared_bytes=int(WITNESS_SHARED_MIB * MIB),
        per_unit_stack_bytes=int(
            max(bounds.values()) if static_bound is None else static_bound
        ),
        per_unit_cache_bytes=int(WITNESS_PREFETCH_MIB * MIB),
        per_unit_prefetch_bytes=int(WITNESS_PREFETCH_MIB * MIB),
        disk_free_bytes=200 * GIB,
        layout=(2, 2),
        refinement_rank=1,
        patch_hw=WITNESS_PATCH_HW,
        halo_px=128,
        channels=3,
        n_frames=WITNESS_MAX_N,
        tile_ladder=tuple(int(t) for t in tile_ladder),
        prefetch_ladder=(0, 1, 2, 3, 4),
        workers_ladder=zs.WORKERS_LADDER,
        per_tile_stack_bytes=(
            tuple(
                (int(tile), int(bounds[int(tile)]))
                for tile in tile_ladder
            )
            if per_tile and per_tile_map is None
            else per_tile_map
        ),
        profile=profile,
    )
    data.update(overrides)
    return zs.PlanContext(**data)


def _cpu_context() -> zs.PlanContext:
    raw, budget = _witness_budget()
    return zs.PlanContext(
        phase=zs.PHASE_CELL_STACK_STREAM,
        backend=zs.BACKEND_CPU,
        budget_bytes=budget,
        raw_available_bytes=raw,
        factor_num=4,
        factor_den=5,
        cpu_logical=16,
        units_total=4,
        shared_bytes=int(WITNESS_SHARED_MIB * MIB),
        per_unit_stack_bytes=max(_witness_bounds().values()),
        per_unit_cache_bytes=int(WITNESS_PREFETCH_MIB * MIB),
        per_unit_prefetch_bytes=int(WITNESS_PREFETCH_MIB * MIB),
        disk_free_bytes=200 * GIB,
        layout=(2, 2),
        refinement_rank=1,
        patch_hw=WITNESS_PATCH_HW,
        halo_px=128,
        channels=3,
        n_frames=WITNESS_MAX_N,
        tile_ladder=TILES,
        prefetch_ladder=(0, 1, 2, 3, 4),
        workers_ladder=zs.WORKERS_LADDER,
    )


def _joint(gpu: zs.PlanContext, facts: zs.GpuFacts | None = None):
    return zs.JointPlanInput(
        contexts=(_cpu_context(), gpu),
        gpu_facts=facts or _gpu_facts(peaks={
            int(t): int(_device_peaks()[int(t)]) for t in gpu.tile_ladder
        }),
    )


def _gpu_candidates(joint):
    return tuple(
        c
        for c in zs.enumerate_joint_candidates(joint)
        if c.backend == zs.BACKEND_GPU
    )


# ---------------------------------------------------------------------------
# 1. Root cause: the old static ladder-max path refuses EVERY GPU candidate.
# ---------------------------------------------------------------------------

def test_witness_bound_table_is_non_monotone_and_matches_the_log():
    bounds = _witness_bounds()
    for tile, expected in WITNESS_BOUND_MIB.items():
        assert bounds[tile] / MIB == pytest.approx(expected, abs=1e-6)
    assert bounds[128] < bounds[112] < bounds[96] < bounds[80] < bounds[64] < bounds[48] < bounds[32]
    assert max(bounds.values()) == bounds[32]


def test_old_static_ladder_max_refuses_every_gpu_candidate_as_host_ram():
    bounds = _witness_bounds()
    joint = _joint(_gpu_context(per_tile=False, static_bound=max(bounds.values())))
    candidates = _gpu_candidates(joint)
    assert candidates
    assert all(candidate.feasible is False for candidate in candidates)
    assert {candidate.reason_code for candidate in candidates} == {
        zs.REASON_REFUSE_HOST_RAM
    }
    # The exact old product bug: the static bound is the tile32 maximum, so
    # even the tile128 candidate is far above the host budget.
    tile128 = next(c for c in candidates if c.tile_px == 128)
    assert tile128.host_peak_bytes > int(WITNESS_B_PLAN_MIB * MIB)

    reason, detail, evidence = zs.gpu_exhaustion_reason(joint, candidates)
    assert reason == zs.REASON_REFUSE_HOST_RAM
    assert "host RAM" in detail
    assert "VRAM" not in detail
    assert evidence["gpu_candidate_count"] == len(candidates)
    assert evidence["gpu_candidate_reason_counts"] == {
        zs.REASON_REFUSE_HOST_RAM: len(candidates)
    }
    assert evidence["host_min_margin_applied_peak_bytes"] > evidence[
        "host_budget_bytes"
    ]


# ---------------------------------------------------------------------------
# 2. Per-tile exactness: every candidate uses its own integer bound.
# ---------------------------------------------------------------------------

def test_each_gpu_candidate_uses_its_exact_tile_bound():
    bounds = _witness_bounds()
    cache = int(WITNESS_PREFETCH_MIB * MIB)
    shared = int(WITNESS_SHARED_MIB * MIB)
    joint = _joint(_gpu_context(per_tile=True))
    candidates = _gpu_candidates(joint)
    assert candidates
    for candidate in candidates:
        assert candidate.tile_px in bounds
        assert candidate.host_peak_bytes == (
            shared
            + bounds[int(candidate.tile_px)]
            + int(candidate.prefetch_units) * cache
        )


def test_tile128_prefetch1_fits_while_tile32_and_prefetch2_do_not():
    bounds = _witness_bounds()
    budget = _witness_budget()[1]
    joint = _joint(_gpu_context(per_tile=True))
    candidates = _gpu_candidates(joint)

    def one(tile, prefetch):
        return next(
            c
            for c in candidates
            if c.tile_px == tile and c.prefetch_units == prefetch
        )

    assert bounds[128] < budget
    assert bounds[32] > budget
    assert one(128, 1).feasible is True
    assert one(128, 2).feasible is False
    assert one(128, 2).reason_code == zs.REASON_REFUSE_HOST_RAM
    assert one(32, 0).feasible is False
    assert one(32, 0).reason_code == zs.REASON_REFUSE_HOST_RAM
    # The exact margin-applied host peak at the witness decision.
    assert (one(128, 1).host_peak_bytes * 115 + 99) // 100 <= budget


# ---------------------------------------------------------------------------
# 3. Real product decision on the published E6 witness facts.
# ---------------------------------------------------------------------------

@dataclass
class _Bundle:
    gauge_workers: int
    cache_build_workers: int
    layout: dict
    envelope: object
    shared_bytes: int
    cache_worker_footprint_bytes: int
    cell_specs: list
    attempts: list
    cpu_fallback_info: dict
    # Fields written by ``_r55_active_gpu_apply`` via dataclasses.replace.
    cpu_streaming: dict = field(default_factory=dict)
    cpu_streaming_tile: int = 0
    gpu_tile_size: int = 0
    gpu_streaming_tile: int = 0
    gpu_streaming_host_peak_bytes: int = 0
    gpu_used: bool = False
    gpu_pre_execution_fallback: object = None
    cell_tasks: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    chosen_mode: str = "stream"
    cells_in_flight: int = 1
    cache_workers_per_cell: int = 1
    per_cell_footprint: int = 0


def _witness_bundle(*, raw_mib=WITNESS_RAW_MIB, canvas_hw=(4096, 4096)):
    raw = int(raw_mib * MIB)
    budget = (4 * raw) // 5
    envelope = SimpleNamespace(
        refinement_rank=1,
        pinned=False,
        raw_available_bytes=raw,
        factor_num=4,
        factor_den=5,
        budget_bytes=budget,
        layout=(2, 2),
    )
    cells = []
    cell_ctxs = []
    specs = []
    for row in range(2):
        for col in range(2):
            cell_id = f"r{row:04d}c{col:04d}"
            core = SimpleNamespace(
                x0=col * (canvas_hw[0] // 2),
                y0=row * (canvas_hw[1] // 2),
                x1=(col + 1) * (canvas_hw[0] // 2),
                y1=(row + 1) * (canvas_hw[1] // 2),
            )
            cell = SimpleNamespace(cell_id=cell_id, core=core)
            patch = SimpleNamespace(
                patch=SimpleNamespace(
                    width=WITNESS_PATCH_HW[1], height=WITNESS_PATCH_HW[0]
                ),
                patch_shape_hw=WITNESS_PATCH_HW,
            )
            mem = SimpleNamespace(
                patch_ids=tuple(f"f{i}" for i in range(WITNESS_MAX_N))
            )
            cells.append(cell_id)
            cell_ctxs.append((row, col, cell, patch, mem))
            specs.append((cell, patch, mem, Path("/tmp"), 0, 0, 0))
    bundle = _Bundle(
        gauge_workers=4,
        cache_build_workers=4,
        layout={"nx": 2, "ny": 2},
        envelope=envelope,
        shared_bytes=int(WITNESS_SHARED_MIB * MIB),
        cache_worker_footprint_bytes=int(64 * MIB),
        cell_specs=specs,
        attempts=[],
        cpu_fallback_info={},
    )
    cells_tuple = (cells, cell_ctxs, bundle, SimpleNamespace(
        width=canvas_hw[0], height=canvas_hw[1], canvas_id="r55e7-witness"
    ))
    return cells_tuple


def _readiness():
    return zgr.GpuReadiness(
        status=zgr.STATUS_USABLE,
        reason_code=zgr.REASON_GPU_OK,
        available=True,
        dependency_declared=True,
        dependency_importable=True,
        cupy_version="14.1.1",
        device_count=1,
        device_index=0,
        device_name="fake-rtx3070",
        vram_total_bytes=8 * GIB,
        vram_free_bytes=7 * GIB,
    )


def _probe(free=7 * GIB):
    return {
        "available": True,
        "device": "fake-rtx3070",
        "vram_free_bytes": int(free),
        "vram_total_bytes": 8 * GIB,
    }


def _decide(tmp_path, *, raw_mib=WITNESS_RAW_MIB, probe_free=7 * GIB):
    _cells, cell_ctxs, bundle, canvas = _witness_bundle(raw_mib=raw_mib)
    return zz._r55_active_gpu_decide(
        bundle=bundle,
        cell_ctxs=cell_ctxs,
        canvas=canvas,
        halo_px=128,
        cache_root=str(tmp_path),
        output_dir=str(tmp_path),
        frame_ids=tuple(f"f{i}" for i in range(WITNESS_MAX_N)),
        gpu_probe=_probe(probe_free),
        readiness=_readiness(),
    )


def _total_bound(tile: int) -> int:
    """Exact TOTAL streaming bound for ANY (including >128) witness tile."""
    return int(
        zz._estimate_streaming_bytes(WITNESS_MAX_N, WITNESS_PATCH_HW, int(tile))
    )


def test_product_decision_is_largest_feasible_dynamic_tile(tmp_path):
    """R55-E9 + REWORK-1 F2: largest jointly feasible GPU tile.

    The E7 budgets were originally pinned to tile128 because the ACTIVE path
    intersected the VRAM maximum with the CPU ladder. That artificial cap is
    gone. UNIFIED PROJECTION: the selected prefetch is 0, so the queue is full
    during the stack, the producer is BLOCKED and is NOT charged; the largest
    feasible rung is therefore 448 (the historical max-only model charged the
    producer unconditionally and stopped at 384).
    """
    shared = int(WITNESS_SHARED_MIB * MIB)
    _raw, budget = _witness_budget()
    decision = _decide(tmp_path)
    tile = 448

    mapping = decision.mapping
    assert mapping["backend"] == "gpu"
    assert mapping["gpu_streaming_tile"] == tile
    assert tile > 128
    assert int(decision.gpu_tile_max) == 507
    assert tile in decision.gpu_tile_ladder
    assert any(int(t) > 128 for t in decision.gpu_tile_ladder)
    assert mapping["units_in_flight"] == 1
    assert mapping["prefetch_units"] == 0
    assert mapping["single_owner"] is True

    chosen = decision.plan_set.chosen
    assert chosen.tile_px == tile
    assert chosen.units_in_flight == 1
    assert chosen.prefetch_units == 0
    # R55-E8: the streaming estimate ALREADY includes the 340 MiB process
    # baseline that ``shared`` counts once, so the scheduler consumes the ADDED
    # bound (the same contract as ``check_streaming_gate``). Exact integer
    # equality is still asserted, never a range.
    added = _total_bound(tile) - int(zz.rcontract.PROCESS_BASELINE_BYTES)
    # UNIFIED PROJECTION at p == 0: shared + added + (owner if p>=1 else 0)
    # + p*resident. The producer is NOT charged for a serial plan.
    assert int(decision.facts["concurrent_producer_bytes"]) > 0
    assert chosen.host_peak_bytes == shared + added
    assert chosen.host_peak_bytes != shared + max(_witness_bounds().values())

    gpu_phase = next(
        env for env in decision.plan_set.phases if env.backend == zs.BACKEND_GPU
    )
    assert gpu_phase.geometry.tile_px == tile
    assert gpu_phase.estimate.peak_bytes == chosen.host_peak_bytes
    assert gpu_phase.accounting.host_peak_bytes == chosen.host_peak_bytes

    device_peaks = decision.device_peak_bytes_by_tile
    assert chosen.device_peak_bytes == device_peaks[tile]
    assert chosen.device_peak_bytes == _device_peaks((tile,))[tile]
    assert chosen.device_peak_bytes <= decision.gpu_facts.vram_budget_bytes
    assert decision.plan_set.aggregate_host_peak_bytes <= budget
    assert decision.plan_set.aggregate_device_peak_bytes <= decision.gpu_facts.vram_budget_bytes

    # The per-tile contract is explicit and COMPLETE for every declared rung:
    # no interpolation, no shared/static bound reused for another tile.
    actual_map = decision.gpu_context.per_tile_stack_map
    assert set(actual_map) == set(decision.gpu_tile_ladder)
    for ladder_tile in decision.gpu_tile_ladder:
        assert actual_map[int(ladder_tile)] == (
            _total_bound(ladder_tile)
            - int(zz.rcontract.PROCESS_BASELINE_BYTES)
        )


def test_product_apply_uses_the_selected_tile_envelope(tmp_path):
    shared = int(WITNESS_SHARED_MIB * MIB)
    _cells, _cell_ctxs, bundle, _canvas = _witness_bundle()
    decision = _decide(tmp_path)
    tile = int(decision.mapping["gpu_streaming_tile"])
    applied = zz._r55_active_gpu_apply(
        bundle, mapping=decision.mapping, decision=decision, stage="preflight"
    )
    assert applied.envelope.gpu_tile == tile
    assert applied.envelope.backend == "gpu"
    assert applied.envelope.cells_in_flight == 1
    assert applied.envelope.cache_workers_per_cell == decision.mapping[
        "workers_per_unit"
    ]
    # R55-E11 rework-3: the envelope consumes the ADDED bound (shared carries the
    # baseline once); the TOTAL estimate remains the execution diagnostic.
    added = _total_bound(tile) - int(zz.rcontract.PROCESS_BASELINE_BYTES)
    assert applied.envelope.stack_total_bytes == shared + added
    assert applied.plan["per_cell_stack_total_bytes"] == _total_bound(tile)
    assert applied.envelope.stack_total_bytes != shared + _total_bound(tile)
    assert applied.envelope.stack_total_bytes != shared + max(_witness_bounds().values())
    assert applied.envelope.aggregate_bytes <= applied.envelope.budget_bytes
    assert applied.gpu_streaming_tile == tile
    assert applied.gpu_tile_size == tile


# ---------------------------------------------------------------------------
# 4. Truthful closed refusal classification.
# ---------------------------------------------------------------------------

def test_host_only_exhaustion_refuses_host_ram_not_vram(tmp_path):
    with pytest.raises(zz._R55GpuRefused) as exc:
        _decide(tmp_path, raw_mib=2000)
    refused = exc.value
    assert refused.reason_code == zs.REASON_REFUSE_HOST_RAM
    assert "host RAM" in refused.detail
    assert "VRAM" not in refused.detail
    record = refused.record
    assert record["gpu_exhaustion"]["gpu_candidate_count"] > 0
    assert set(record["gpu_exhaustion"]["gpu_candidate_reason_counts"]) == {
        zs.REASON_REFUSE_HOST_RAM
    }
    assert record["b_plan_bytes"] <= int(2000 * MIB)


def test_true_device_exhaustion_stays_vram(tmp_path):
    # Product preflight: no device tile fits the fresh 70 % VRAM budget.
    with pytest.raises(zz._R55GpuRefused) as exc:
        _decide(tmp_path, probe_free=40 * MIB)
    assert exc.value.reason_code == zs.REASON_REFUSE_VRAM

    # Scheduler classification: host fits for every candidate, device does not.
    peaks = {int(tile): 2 * GIB for tile in TILES}
    facts = _gpu_facts(vram_budget=1 * GIB, peaks=peaks)
    raw, _budget = _witness_budget()
    gpu = _gpu_context(per_tile=True, raw=raw)
    joint = _joint(gpu, facts)
    candidates = _gpu_candidates(joint)
    assert candidates
    assert all(c.device_peak_bytes > facts.vram_budget_bytes for c in candidates)
    assert all(c.feasible is False for c in candidates)
    reason, detail, evidence = zs.gpu_exhaustion_reason(joint, candidates)
    assert reason == zs.REASON_REFUSE_VRAM
    assert "device VRAM" in detail
    assert evidence["device_min_peak_bytes"] > evidence["device_budget_bytes"]


# ---------------------------------------------------------------------------
# 5. The per-tile map is a strict contract.
# ---------------------------------------------------------------------------

def _bad_map_cases():
    good = {int(t): int(_witness_bounds()[int(t)]) for t in TILES}
    incomplete = dict(good)
    incomplete.pop(32)
    extra = dict(good)
    extra[16] = 1
    return [
        ("incomplete", incomplete),
        ("extra", extra),
        ("bool", {**good, 128: True}),
        ("float", {**good, 128: 1.5}),
        ("negative", {**good, 128: -1}),
        ("duplicate", ((128, good[128]), (128, good[128] + 1))),
        ("empty", ()),
    ]


@pytest.mark.parametrize("label,value", _bad_map_cases(), ids=lambda v: str(v))
def test_invalid_per_tile_maps_fail_closed(label, value):
    gpu = _gpu_context(per_tile=False, per_tile_map=value)
    with pytest.raises(zs.SchedulerConfigError):
        _joint(gpu)


def test_map_and_calibrated_profile_are_mutually_exclusive():
    profile = zcal.CalibrationProfile(
        calibration_id="cal-e7-test",
        host_id="TINYDEBIAN",
        os="Linux-test",
        cpu_logical=8,
        ram_total_bytes=8 * GIB,
        phase=zs.PHASE_CELL_STACK_STREAM,
        backend=zs.BACKEND_GPU,
        path="product.test.gpu.stream",
        model=zcal.MODEL_LINEAR_COMPONENTS,
        coefficients={
            "process_baseline_bytes": 100 * MIB,
            "tile_workspace_bytes_per_px_ch_frame": 2,
            "tile_fixed_overhead_bytes": 1 * MIB,
            "aligned_input_bytes_per_px_ch_frame": 4,
            "rejection_mask_bytes_per_px_ch_frame": 1,
        },
        valid_ranges={
            "tile_px": (64, 128),
            "n_frames": (1, 100),
            "patch_area_px": (1, 1_000_000),
        },
        measured={},
        margin_num=15,
        margin_den=100,
        product_admissible=True,
    )
    gpu = _gpu_context(per_tile=True, profile=profile)
    with pytest.raises(zs.SchedulerConfigError):
        _joint(gpu)


def test_per_tile_map_is_rejected_for_inmem_phase():
    gpu = _gpu_context(
        per_tile=True, phase=zs.PHASE_CELL_STACK_INMEM
    )
    with pytest.raises(zs.SchedulerConfigError):
        _joint(gpu)


def test_cpu_static_path_is_unchanged_without_a_map():
    joint = _joint(_gpu_context(per_tile=False, static_bound=123456))
    tile128 = next(
        c
        for c in _gpu_candidates(joint)
        if c.tile_px == 128 and c.prefetch_units == 0
    )
    assert tile128.estimate_method == "static_bound"
    assert tile128.host_peak_bytes == (
        int(WITNESS_SHARED_MIB * MIB) + 123456
    )


# ---------------------------------------------------------------------------
# 6. Determinism / hashing / manifest payload.
# ---------------------------------------------------------------------------

def test_context_payload_covers_the_map_without_changing_static_payloads():
    static_payload = zs._joint_context_payload(_gpu_context(per_tile=False))
    assert "per_tile_stack_bytes" not in static_payload

    mapped = _gpu_context(per_tile=True)
    payload_a = zs._joint_context_payload(mapped)
    payload_b = zs._joint_context_payload(dataclasses.replace(mapped))
    assert payload_a == payload_b
    assert "per_tile_stack_bytes" in payload_a
    assert len(payload_a["per_tile_stack_bytes"]) == len(TILES)
    assert json.dumps(payload_a, sort_keys=True) == json.dumps(
        payload_b, sort_keys=True
    )

    changed = _witness_bounds()
    changed[128] += 1
    other = _gpu_context(
        per_tile=False,
        per_tile_map=tuple(sorted(changed.items())),
    )
    payload_c = zs._joint_context_payload(other)
    assert payload_c["per_tile_stack_bytes"] != payload_a["per_tile_stack_bytes"]


def test_plan_hash_is_deterministic_for_identical_maps():
    raw, budget = _witness_budget()

    def build():
        context = _gpu_context(per_tile=True)
        joint = _joint(context)
        chosen = next(
            c
            for c in _gpu_candidates(joint)
            if c.feasible and c.tile_px == 128 and c.prefetch_units == 1
        )
        return zs.build_joint_plan_set(
            joint,
            candidate=chosen,
            phase_requests=(
                zs.PhaseRequest(
                    phase=re.PHASE_GAUGE,
                    components={"gauge_worker_footprint_bytes": 1},
                    units=1,
                    units_in_flight=1,
                    workers_per_unit=1,
                ),
                zs.PhaseRequest(
                    phase=re.PHASE_CACHE_BUILD,
                    components={"cache_build_worker_footprint_bytes": 1},
                    units=1,
                    units_in_flight=1,
                    workers_per_unit=1,
                ),
                zs.PhaseRequest(
                    phase=zs.PHASE_ASSEMBLY,
                    components={"halo_band_bytes": 0, "retained_core_bytes": 1},
                    units=1,
                    units_in_flight=1,
                    workers_per_unit=1,
                    overlaps_with=(),
                ),
                zs.PhaseRequest(
                    phase=re.PHASE_EXPORT,
                    components={"coverage_plane_bytes": 1},
                    units=1,
                    units_in_flight=1,
                    workers_per_unit=1,
                ),
            ),
            mode=zs.MODE_ACTIVE,
        )

    first = build()
    second = build()
    assert first.plan_hash == second.plan_hash
    assert first.plan_hash.startswith("sha256:")
    assert first.chosen.host_peak_bytes == second.chosen.host_peak_bytes


# ---------------------------------------------------------------------------
# 7. Gated REAL M106 descriptor planning probe (header-only, no pixels).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LIGHTS.is_dir(), reason="M106 lights directory absent")
def test_m106_descriptor_planning_probe_admits_dynamic_tile(tmp_path):
    frames, _rejected = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(frames)
    assert len(frames) == 66
    assert (canvas.width, canvas.height) == (2403, 3278)

    raw, budget = _witness_budget()
    layout = zz._choose_layout_mode_aware(
        canvas,
        frames,
        ram_budget=budget,
        available_bytes=raw,
        halo_px=128,
        footprints_out=[],
    )
    assert (layout["nx"], layout["ny"]) == (2, 2)
    assert layout["max_contributors"] == 66
    worst = max(layout["cells"], key=lambda cell: int(cell["patch_area"]))
    assert tuple(int(v) for v in worst["patch_hw"]) == (1767, 1330)
    assert int(layout["predicted_bound_bytes"]) == _witness_bounds()[128]

    # Build an EXACT product decision from the real descriptors (no pixels).
    cell_ctxs = []
    specs = []
    for cell_meta in layout["cells"]:
        row = int(cell_meta["row"])
        col = int(cell_meta["col"])
        h, w = (int(v) for v in cell_meta["patch_hw"])
        core = SimpleNamespace(
            x0=col * canvas.width // int(layout["nx"]),
            y0=row * canvas.height // int(layout["ny"]),
            x1=(col + 1) * canvas.width // int(layout["nx"]),
            y1=(row + 1) * canvas.height // int(layout["ny"]),
        )
        cell = SimpleNamespace(cell_id=str(cell_meta["cell_id"]), core=core)
        patch = SimpleNamespace(
            patch=SimpleNamespace(width=w, height=h),
            patch_shape_hw=(h, w),
        )
        mem = SimpleNamespace(
            patch_ids=tuple(
                f.frame_id.logical_path for f in frames[: int(cell_meta["n"])]
            )
        )
        cell_ctxs.append((row, col, cell, patch, mem))
        specs.append((cell, patch, mem, tmp_path, 0, 0, 0))

    bundle = _Bundle(
        gauge_workers=4,
        cache_build_workers=4,
        layout=layout,
        envelope=SimpleNamespace(
            refinement_rank=1,
            pinned=False,
            raw_available_bytes=raw,
            factor_num=4,
            factor_den=5,
            budget_bytes=budget,
            layout=(int(layout["nx"]), int(layout["ny"])),
        ),
        shared_bytes=int(WITNESS_SHARED_MIB * MIB),
        cache_worker_footprint_bytes=int(64 * MIB),
        cell_specs=specs,
        attempts=[],
        cpu_fallback_info={},
    )
    frame_ids = tuple(f.frame_id.logical_path for f in frames)
    decision = zz._r55_active_gpu_decide(
        bundle=bundle,
        cell_ctxs=cell_ctxs,
        canvas=canvas,
        halo_px=128,
        cache_root=str(tmp_path),
        output_dir=str(tmp_path),
        frame_ids=frame_ids,
        gpu_probe=_probe(),
        readiness=_readiness(),
    )
    tile = int(decision.mapping["gpu_streaming_tile"])
    assert tile == 448
    assert tile > 128
    assert decision.mapping["units_in_flight"] == 1
    assert decision.mapping["prefetch_units"] == 0
    # prefetch_units == 0: the plan is SERIAL, so the host peak is shared +
    # this tile's exact ADDED streaming bound. The declared producer owner is
    # published but NOT charged at p == 0.
    assert decision.plan_set.chosen.host_peak_bytes == (
        int(WITNESS_SHARED_MIB * MIB)
        + (_total_bound(tile) - int(zz.rcontract.PROCESS_BASELINE_BYTES))
    )
    assert int(decision.facts["concurrent_producer_bytes"]) > 0
    assert decision.plan_set.aggregate_host_peak_bytes <= budget
    assert decision.plan_set.aggregate_device_peak_bytes <= decision.gpu_facts.vram_budget_bytes
