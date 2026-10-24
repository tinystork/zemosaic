"""ZM-ZEGRID-R55-E9 — dynamic, device-agnostic ACTIVE GPU streaming-tile ladder.

The post-E8 Windows M106 run was functionally correct but underused the GPU: the
fresh-VRAM maximum (506 px) was intersected with the CPU streaming ladder
``(128, 112, ...)``, so every rung above 128 was discarded and execution always
used tile128. These tests pin the corrected contract:

* ONE pure helper builds a deterministic, strictly descending, finite GPU ladder
  from the exact fresh-VRAM maximum: the exact max, coarse 64-px-aligned rungs
  down to 128, then the canonical CPU tail (<= 128). No model name, no fixed
  device ceiling.
* every rung carries its OWN exact device workspace and ADDED host bound; maps
  are complete and are never interpolated or reused across tiles;
* the recorded Windows M106 facts select tile 506 (prefetch 0) only because the
  existing scheduler says it is jointly feasible under B_plan and the fresh 70 %
  VRAM budget — no performance claim is made;
* small/large synthetic device budgets behave monotonically, and small-VRAM
  cases refuse honestly instead of extrapolating;
* fresh VRAM pressure shrinks along the frozen ladder and can NEVER grow it,
  even when VRAM recovers; host pressure keeps the fixed global shrink order and
  the whole-batch CPU fallback;
* 128 vs a >128 tile are bit-identical through the canonical streaming engine;
* the diagnostics block states the auto max, declared ladder, selected tile and
  both exact peaks.

Model-only: no CuPy call, no pixel mosaic, no GPU performance claim.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.canonical_engine import (
    CanonicalStackRequest,
    run_canonical_stack,
)
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import gpu as zgpu
from zemosaic.core.zegrid import gpu_readiness as zgr
from zemosaic.core.zegrid import scheduler as zs

MIB = 2 ** 20
GIB = 2 ** 30
LIGHTS = Path("/home/tristan/M106/lights")

# Recorded Windows M106 facts (R55-E8 real run).
WITNESS_FRAMES = 66
WITNESS_PATCH_HW = (1767, 1330)  # (h, w) as cell patch_shape_hw
WITNESS_FREE = 7463763968
WITNESS_TOTAL = 8589410304
WITNESS_B_PLAN = 10481732812
WITNESS_RAW = 13102166016
WITNESS_CPU = 16
WITNESS_HALO = 128
WITNESS_VRAM_BUDGET = 5224634777  # floor(free * 70 / 100)

# Exact R55-E9 model outcomes (bytes/px), used as equality assertions.
WITNESS_MAX_TILE = 506
WITNESS_LADDER = (506, 448, 384, 320, 256, 192, 128, 112, 96, 80, 64, 48, 32)
WITNESS_SELECTED_TILE = 506
WITNESS_SELECTED_PREFETCH = 0
WITNESS_SELECTED_WORKERS = 14
WITNESS_SHARED = 356515840
WITNESS_ADDED_506 = 6795085824
WITNESS_HOST_506 = 7151601664
WITNESS_DEVICE_506 = 5223334912
WITNESS_AGGREGATE_HOST = 10305433931
WITNESS_AGGREGATE_DEVICE = 5223334912
WITNESS_ENVELOPE_AGGREGATE = 7508117504

# Canonical CPU tail (unchanged by R55-E9).
CPU_TAIL = (128, 112, 96, 80, 64, 48, 32)


def _readiness(free=WITNESS_FREE, total=WITNESS_TOTAL):
    return zgr.GpuReadiness(
        status=zgr.STATUS_USABLE,
        reason_code=zgr.REASON_GPU_OK,
        available=True,
        dependency_declared=True,
        dependency_importable=True,
        cupy_version="model-only",
        device_count=1,
        device_index=0,
        device_name="simulated-device",
        vram_total_bytes=int(total),
        vram_free_bytes=int(free),
    )


def _probe(free=WITNESS_FREE, total=WITNESS_TOTAL):
    return {
        "available": True,
        "device": "simulated-device",
        "vram_total_bytes": int(total),
        "vram_free_bytes": int(free),
    }


@pytest.fixture(scope="module")
def m106():
    if not LIGHTS.is_dir():
        pytest.skip("M106 lights directory absent")
    frames, _rejected = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(frames)
    assert len(frames) == WITNESS_FRAMES
    return frames, canvas


@pytest.fixture(autouse=True)
def _deterministic_host(monkeypatch):
    monkeypatch.setattr(
        zz.shutil,
        "disk_usage",
        lambda _p: SimpleNamespace(total=4 * 2 ** 40, used=0, free=4 * 2 ** 40),
    )
    monkeypatch.setattr(zz.os, "cpu_count", lambda: WITNESS_CPU)


def _windows_bundle(m106):
    frames, canvas = m106
    return zz._build_execution_bundle(
        canvas,
        frames,
        None,
        raw_available_bytes=WITNESS_RAW,
        budget_bytes=WITNESS_B_PLAN,
        halo_px=WITNESS_HALO,
        cpu_count=WITNESS_CPU,
        gpu_used=True,
        vram_budget_bytes=zgpu.vram_budget_bytes(
            {"available": True, "vram_free_bytes": WITNESS_FREE}
        ),
        cache_root="/tmp",
    )


def _windows_decision(m106, bundle=None):
    frames, canvas = m106
    bundle = _windows_bundle(m106) if bundle is None else bundle
    return zz._r55_active_gpu_decide(
        bundle=bundle,
        cell_ctxs=bundle.cell_ctxs,
        canvas=canvas,
        halo_px=WITNESS_HALO,
        cache_root="/tmp",
        output_dir="/tmp",
        frame_ids=tuple(f.frame_id.logical_path for f in frames),
        gpu_probe=_probe(),
        readiness=_readiness(),
    )


# ---------------------------------------------------------------------------
# A. Ladder policy: deterministic, descending, finite, device-agnostic.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "max_tile,expected",
    [
        (31, ()),
        (32, (32,)),
        (40, (40, 32)),
        (64, (64, 48, 32)),
        (96, (96, 80, 64, 48, 32)),
        (100, (100, 96, 80, 64, 48, 32)),
        (128, CPU_TAIL),
        (129, (129,) + CPU_TAIL),
        (192, (192,) + CPU_TAIL),
        (256, (256, 192) + CPU_TAIL),
        (506, WITNESS_LADDER),
    ],
)
def test_e9_ladder_policy_is_exact_and_descending(max_tile, expected):
    ladder = zz._r55_active_gpu_tile_ladder(max_tile)
    assert ladder == expected
    if ladder:
        assert ladder[0] == max_tile
        assert all(a > b for a, b in zip(ladder, ladder[1:]))
        assert len(set(ladder)) == len(ladder)
        assert max(ladder) <= max_tile
        tail = tuple(t for t in CPU_TAIL if t <= ladder[0])
        assert tail
        assert ladder[-len(tail):] == tail


def test_e9_ladder_rejects_malformed_maximums():
    with pytest.raises(Exception):
        zz._r55_active_gpu_tile_ladder(True)
    with pytest.raises(Exception):
        zz._r55_active_gpu_tile_ladder(0)


def test_e9_ladder_does_not_touch_the_cpu_ladder():
    assert zz.CPU_STREAMING_TILE_LADDER == CPU_TAIL
    assert zz._r55_active_gpu_tile_ladder(128) == zz.CPU_STREAMING_TILE_LADDER


# ---------------------------------------------------------------------------
# A. Windows M106 witness (model-only, exact recorded facts).
# ---------------------------------------------------------------------------

def test_e9_windows_m106_witness_uses_the_vram_max_tile(m106):
    frames, canvas = m106
    assert (canvas.width, canvas.height) == (2403, 3278)
    bundle = _windows_bundle(m106)
    decision = _windows_decision(m106, bundle)

    assert decision.facts["layout"] == (2, 2)
    assert decision.facts["halo_px"] == WITNESS_HALO
    assert int(decision.gpu_tile_max) == WITNESS_MAX_TILE
    assert tuple(int(t) for t in decision.gpu_tile_ladder) == WITNESS_LADDER
    # Meaningful intermediates strictly above the old cap, plus the canonical tail.
    assert [t for t in decision.gpu_tile_ladder if t > 128] == [
        506, 448, 384, 320, 256, 192
    ]
    assert tuple(decision.gpu_tile_ladder[-len(CPU_TAIL):]) == CPU_TAIL

    mapping = decision.mapping
    assert mapping["backend"] == "gpu"
    assert mapping["gpu_streaming_tile"] == WITNESS_SELECTED_TILE
    assert mapping["prefetch_units"] == WITNESS_SELECTED_PREFETCH
    assert mapping["units_in_flight"] == 1
    assert mapping["workers_per_unit"] == WITNESS_SELECTED_WORKERS
    assert mapping["single_owner"] is True

    # Every declared rung carries its OWN exact device + ADDED host bound.
    assert dict(decision.gpu_host_bound_by_tile) == {
        t: max(
            1,
            int(zz._estimate_streaming_bytes(WITNESS_FRAMES, WITNESS_PATCH_HW, t))
            - int(zz.rcontract.PROCESS_BASELINE_BYTES),
        )
        for t in WITNESS_LADDER
    }
    assert dict(decision.device_peak_bytes_by_tile) == {
        t: int(
            zgpu.gpu_tile_workspace_bytes(
                WITNESS_FRAMES,
                t + 2 * int(zgpu.GPU_TAPER_HALO_PX),
                t + 2 * int(zgpu.GPU_TAPER_HALO_PX),
                3,
            )
        )
        for t in WITNESS_LADDER
    }

    chosen = decision.plan_set.chosen
    assert chosen.tile_px == WITNESS_SELECTED_TILE
    assert chosen.prefetch_units == WITNESS_SELECTED_PREFETCH
    assert chosen.units_in_flight == 1
    assert chosen.host_peak_bytes == WITNESS_HOST_506 == (
        WITNESS_SHARED + WITNESS_ADDED_506
    )
    assert chosen.device_peak_bytes == WITNESS_DEVICE_506
    assert dict(decision.gpu_host_bound_by_tile)[WITNESS_SELECTED_TILE] == (
        WITNESS_ADDED_506
    )

    # Host margin is applied exactly ONCE at the plan boundary.
    raw_peak = int(decision.plan_set.aggregate_host_peak_bytes_raw)
    assert decision.plan_set.aggregate_host_peak_bytes == -(-raw_peak * 115 // 100)
    assert decision.plan_set.aggregate_host_peak_bytes == WITNESS_AGGREGATE_HOST
    assert decision.plan_set.aggregate_host_peak_bytes <= WITNESS_B_PLAN
    assert decision.plan_set.aggregate_device_peak_bytes == WITNESS_AGGREGATE_DEVICE
    assert (
        decision.plan_set.aggregate_device_peak_bytes
        <= decision.gpu_facts.vram_budget_bytes
        == WITNESS_VRAM_BUDGET
    )
    assert int(decision.qualification["percent"]) == 70

    # The exact selected tile reaches the envelope and the applied bundle.
    applied = zz._r55_active_gpu_apply(
        bundle, mapping=mapping, decision=decision, stage="preflight"
    )
    assert applied.envelope.gpu_tile == WITNESS_SELECTED_TILE
    assert applied.envelope.backend == "gpu"
    assert applied.envelope.cells_in_flight == 1
    assert applied.envelope.aggregate_bytes == WITNESS_ENVELOPE_AGGREGATE
    assert applied.envelope.aggregate_bytes <= WITNESS_B_PLAN
    assert applied.gpu_tile_size == WITNESS_SELECTED_TILE
    assert applied.gpu_streaming_tile == WITNESS_SELECTED_TILE
    assert applied.gpu_streaming_host_peak_bytes == int(
        zz._estimate_streaming_bytes(
            WITNESS_FRAMES, WITNESS_PATCH_HW, WITNESS_SELECTED_TILE
        )
    )

    # Identity: layout, halo and all 66 frame identities are untouched.
    assert applied.envelope.layout == (2, 2)
    assert int(decision.facts["halo_px"]) == WITNESS_HALO
    assert len(decision.facts["frame_ids"]) == WITNESS_FRAMES
    assert decision.facts["frame_ids"] == tuple(
        f.frame_id.logical_path for f in frames
    )
    # Execution wiring: every non-empty cell task streams at the SELECTED tile.
    for (cell, patch, mem, _cache, mode, bound) in applied.cell_tasks:
        if not mem.patch_ids:
            continue
        assert mode == "stream"
        assert bound == int(
            zz._estimate_streaming_bytes(
                len(mem.patch_ids), patch.patch_shape_hw, WITNESS_SELECTED_TILE
            )
        )
    assert zz._r55_active_execution(applied, mapping)["gpu_streaming_tile"] == (
        WITNESS_SELECTED_TILE
    )


# ---------------------------------------------------------------------------
# B. Synthetic device matrix (no hardware names).
# ---------------------------------------------------------------------------

PATCH_4096 = (4096, 4096)
N_64 = 64


@pytest.mark.parametrize("gib", [2, 4, 8, 16, 24])
def test_e9_synthetic_budget_matrix_exposes_no_tile_above_max(gib):
    budget = gib * GIB
    max_tile = zgpu.choose_gpu_tile_size(N_64, 3, PATCH_4096, budget)
    assert max_tile is not None
    ladder = zz._r55_active_gpu_tile_ladder(max_tile)
    assert ladder[0] == max_tile
    assert max(ladder) == max_tile
    assert all(t <= max_tile for t in ladder)
    assert all(a > b for a, b in zip(ladder, ladder[1:]))


def test_e9_synthetic_budget_matrix_is_monotone_and_adapts_past_3070():
    maxima = {
        gib: zgpu.choose_gpu_tile_size(N_64, 3, PATCH_4096, gib * GIB)
        for gib in (2, 4, 8, 16, 24)
    }
    ordered = [maxima[gib] for gib in (2, 4, 8, 16, 24)]
    assert ordered == sorted(ordered)
    assert ordered[0] >= 128
    # No model-name rule and no fixed 3070/5080 ceiling: 24 GiB exposes a
    # ladder top far above the 3070 witness when the patch permits it.
    assert maxima[24] > WITNESS_MAX_TILE
    assert maxima[24] == 1164
    assert 256 in zz._r55_active_gpu_tile_ladder(maxima[24])


def test_e9_small_budget_produces_a_valid_tail_and_refuses_honestly(m106):
    # 20 MiB free -> 70 % budget is below the policy floor: closed refusal.
    frames, canvas = m106
    bundle = _windows_bundle(m106)
    with pytest.raises(zz._R55GpuRefused) as exc:
        zz._r55_active_gpu_decide(
            bundle=bundle,
            cell_ctxs=bundle.cell_ctxs,
            canvas=canvas,
            halo_px=WITNESS_HALO,
            cache_root="/tmp",
            output_dir="/tmp",
            frame_ids=tuple(f.frame_id.logical_path for f in frames),
            gpu_probe=_probe(free=20 * MIB, total=WITNESS_TOTAL),
            readiness=_readiness(free=20 * MIB, total=WITNESS_TOTAL),
        )
    assert exc.value.reason_code in (
        zgr.REASON_REFUSE_VRAM,
        zgr.REASON_VRAM_BELOW_MINIMUM,
    )

    # A budget that yields max<128 still yields a finite canonical tail.
    max_tile = zgpu.choose_gpu_tile_size(WITNESS_FRAMES, 3, WITNESS_PATCH_HW, 70 * MIB)
    assert 32 <= max_tile < 128
    ladder = zz._r55_active_gpu_tile_ladder(max_tile)
    assert ladder[0] == max_tile
    assert all(t <= max_tile for t in ladder)
    assert set(ladder) <= set(CPU_TAIL) | {max_tile}
    assert all(a > b for a, b in zip(ladder, ladder[1:]))


def _strict_gpu_context(*, per_tile_map, tile_ladder=(506, 448, 256, 192, 128)):
    raw = 13 * GIB
    return zs.PlanContext(
        phase=zs.PHASE_CELL_STACK_STREAM,
        backend=zs.BACKEND_GPU,
        budget_bytes=(4 * raw) // 5,
        raw_available_bytes=raw,
        factor_num=4,
        factor_den=5,
        cpu_logical=16,
        units_total=4,
        shared_bytes=WITNESS_SHARED,
        per_unit_stack_bytes=7 * GIB,
        per_unit_cache_bytes=1 * GIB,
        per_unit_prefetch_bytes=1 * GIB,
        disk_free_bytes=200 * GIB,
        layout=(2, 2),
        refinement_rank=1,
        patch_hw=WITNESS_PATCH_HW,
        halo_px=WITNESS_HALO,
        channels=3,
        n_frames=WITNESS_FRAMES,
        tile_ladder=tuple(tile_ladder),
        prefetch_ladder=(0, 1, 2, 3, 4),
        workers_ladder=zs.WORKERS_LADDER,
        per_tile_stack_bytes=per_tile_map,
    )


def _validated_context(per_tile_map):
    context = _strict_gpu_context(per_tile_map=per_tile_map)
    zs._validate_context(context, allow_gpu=True)
    return context


def test_e9_strict_per_tile_map_contracts_still_reject_bad_maps():
    complete = {506: 6, 448: 5, 256: 3, 192: 2, 128: 1}
    assert _validated_context(complete).per_tile_stack_map == complete

    # Incomplete (a declared rung has no exact bound).
    with pytest.raises(zs.SchedulerConfigError):
        _validated_context({506: 6, 448: 5, 256: 3, 192: 2})
    # Extra tile outside the declared ladder.
    with pytest.raises(zs.SchedulerConfigError):
        _validated_context({**complete, 384: 4})
    # Duplicate / contradictory pair for one tile.
    with pytest.raises(zs.SchedulerConfigError):
        _validated_context(
            ((506, 6), (506, 7), (448, 5), (256, 3), (192, 2), (128, 1))
        )
    # Malformed (bool/float/negative bytes) entries.
    with pytest.raises(zs.SchedulerConfigError):
        _validated_context({**complete, 128: 0})
    with pytest.raises(zs.SchedulerConfigError):
        _validated_context({**complete, 128: 1.5})

    # Non-descending execution ladders are refused: off-ladder tile mapping and
    # any tile growth are rejected by the existing strict contracts.
    cand = zs.Candidate(
        backend=zs.BACKEND_GPU,
        tile_px=256,
        units_in_flight=1,
        prefetch_units=0,
        workers_per_unit=1,
        host_peak_bytes=1,
        disk_bytes=0,
        device_peak_bytes=1,
        feasible=True,
        reason_code=zs.REASON_OK,
    )
    with pytest.raises(zs.ActivePlanRejected):
        zs.active_gpu_mapping_for_candidate(cand, tile_ladder=(128, 112, 96, 80, 64, 48, 32))
    grown = zs.Candidate(**{**cand.__dict__, "tile_px": 448})
    with pytest.raises(zs.GrowthRejected):
        zs.assert_joint_shrink_only(cand, grown)


# ---------------------------------------------------------------------------
# C. Pressure: shrink-only along the frozen ladder, never grow.
# ---------------------------------------------------------------------------

def _admit(decision, bundle, *, raw, vram_values):
    values = list(vram_values)
    calls = {"n": 0}

    def _vram():
        index = min(calls["n"], len(values) - 1)
        calls["n"] += 1
        return int(values[index])

    return zz._r55_active_gpu_admit(
        initial_bundle=bundle,
        decision=decision,
        available_fn=lambda: int(raw),
        vram_free_fn=_vram,
    )


def test_e9_vram_pressure_shrinks_to_next_declared_tile(m106):
    bundle = _windows_bundle(m106)
    decision = _windows_decision(m106, bundle)
    assert decision.mapping["gpu_streaming_tile"] == WITNESS_SELECTED_TILE

    # 5.0 GB free at 70 % = 3.5 GB: tile506 (5.22 GB) and 448 (4.13 GB) cannot
    # fit; the next declared feasible rung is 384 (3.08 GB).
    applied, candidate, mapping, outcome, trail = _admit(
        decision, bundle, raw=WITNESS_RAW, vram_values=[5_000_000_000] * 8
    )
    assert mapping["gpu_streaming_tile"] == 384
    assert mapping["gpu_streaming_tile"] < WITNESS_SELECTED_TILE
    assert candidate.tile_px == 384
    assert applied.gpu_streaming_tile == 384
    assert applied.envelope.gpu_tile == 384
    assert int(decision.device_peak_bytes_by_tile[384]) <= 3_500_000_000
    assert mapping["units_in_flight"] == 1
    assert outcome.replans == 1
    assert any(step["step"] == "tile_px" for step in trail)
    tile_steps = [s for s in trail if s["step"] == "tile_px"]
    assert tile_steps[0]["from"]["tile_px"] == WITNESS_SELECTED_TILE
    assert tile_steps[0]["to"]["tile_px"] == 384
    assert tile_steps[0]["vram_free_bytes"] == 5_000_000_000
    # No step grew any resource.
    for step in trail:
        assert step["to"]["tile_px"] <= step["from"]["tile_px"]
        assert step["to"]["prefetch_units"] <= step["from"]["prefetch_units"]
        assert step["to"]["workers_per_unit"] <= step["from"]["workers_per_unit"]


def test_e9_vram_recovery_never_grows_the_frozen_ladder(m106):
    bundle = _windows_bundle(m106)
    decision = _windows_decision(m106, bundle)
    # First observation is tight, then VRAM recovers hugely. The plan already
    # moved to tile384 and must stay there: no growth after planning.
    applied, candidate, mapping, outcome, _trail = _admit(
        decision,
        bundle,
        raw=WITNESS_RAW,
        vram_values=[5_000_000_000, 64 * GIB, 64 * GIB, 64 * GIB],
    )
    assert candidate.tile_px == 384
    assert mapping["gpu_streaming_tile"] == 384
    assert applied.gpu_streaming_tile == 384
    assert outcome.replans == 1


def test_e9_host_pressure_keeps_global_order_and_whole_batch_fallback(m106):
    bundle = _windows_bundle(m106)
    decision = _windows_decision(m106, bundle)
    with pytest.raises(zz._R55GpuFallback) as exc:
        _admit(
            decision,
            bundle,
            raw=1 * GIB,
            vram_values=[WITNESS_FREE] * 8,
        )
    steps = [s["step"] for s in exc.value.telemetry.get("trail", [])]
    fixed_order = [
        "prefetch_units",
        "workers_per_unit",
        "units_in_flight",
        "tile_px",
        "backend",
    ]
    assert steps
    assert all(step in fixed_order for step in steps)
    assert steps == sorted(steps, key=fixed_order.index)
    assert exc.value.telemetry["trail"][0]["step"] != "tile_px"


# ---------------------------------------------------------------------------
# E. Tile-boundary invariance through the canonical streaming engine.
# ---------------------------------------------------------------------------

def _mono_corpus(n=6, h=300, w=260, seed=11):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.full((h, w), 120.0, dtype=np.float64)
    for k in range(5):
        cy = rng.uniform(0.2, 0.8) * h
        cx = rng.uniform(0.2, 0.8) * w
        base += 35.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0 ** 2))
    frames = []
    for i in range(n):
        f = base * (0.97 + 0.01 * i) + i + rng.normal(0.0, 1.5, (h, w))
        frames.append(f.astype(np.float32))
    frames[1][10:20, 10:20] = np.nan
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[3][:, w // 2:] = False
    return frames, masks


def _assert_bit_exact(a, b):
    np.testing.assert_array_equal(a.science, b.science)
    np.testing.assert_array_equal(a.estimator_weight_sum, b.estimator_weight_sum)
    np.testing.assert_array_equal(a.valid_mask, b.valid_mask)
    np.testing.assert_array_equal(a.surviving_sample_count, b.surviving_sample_count)
    np.testing.assert_array_equal(a.support_w1, b.support_w1)
    np.testing.assert_array_equal(a.support_w2, b.support_w2)
    np.testing.assert_array_equal(a.n_eff_support, b.n_eff_support)
    np.testing.assert_array_equal(a.rejection_mask, b.rejection_mask)
    assert a.iterations_used == b.iterations_used
    assert a.rejected_sample_count == b.rejected_sample_count
    assert a.rejected_fraction == b.rejected_fraction
    assert a.low_n_cell_count == b.low_n_cell_count
    assert a.degenerate_cell_count == b.degenerate_cell_count
    assert a.provenance == b.provenance


def test_e9_tile_boundary_invariance_128_vs_192():
    frames, masks = _mono_corpus()
    request = CanonicalStackRequest(
        images=frames,
        geometric_support=masks,
        normalization="linear_fit",
        weighting="noise_variance",
        rejection="kappa_sigma",
        combine="mean",
        taper="footprint",
        taper_px=8.0,
    )
    ref = run_canonical_stack(request)
    at_128 = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), request, tile_size=128
    )
    at_192 = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), request, tile_size=192
    )
    # Exact contract of the canonical engine: no weakened tolerance.
    _assert_bit_exact(ref, at_128)
    _assert_bit_exact(at_128, at_192)


# ---------------------------------------------------------------------------
# G. Diagnostics: auto max, declared ladder, selected tile, peaks.
# ---------------------------------------------------------------------------

def test_e9_diagnostics_report_exact_tile_and_peaks(m106):
    bundle = _windows_bundle(m106)
    decision = _windows_decision(m106, bundle)
    applied = zz._r55_active_gpu_apply(
        bundle, mapping=decision.mapping, decision=decision, stage="preflight"
    )
    block = zz._r55_gpu_tile_decision_dict(decision, decision.mapping, applied)
    assert block["max_tile_px"] == WITNESS_MAX_TILE
    assert block["declared_ladder"] == list(WITNESS_LADDER)
    assert block["selected_tile_px"] == WITNESS_SELECTED_TILE
    assert block["selected_prefetch_units"] == WITNESS_SELECTED_PREFETCH
    assert block["selected_host_added_bound_bytes"] == WITNESS_ADDED_506
    assert block["selected_host_aggregate_bytes"] == WITNESS_ENVELOPE_AGGREGATE
    assert block["selected_device_peak_bytes"] == WITNESS_DEVICE_506
    assert block["device_peak_bytes_by_tile"]["506"] == WITNESS_DEVICE_506
    assert block["host_added_bound_bytes_by_tile"]["506"] == WITNESS_ADDED_506
    assert block["vram_budget_bytes"] == WITNESS_VRAM_BUDGET
    assert block["vram_percent"] == 70
