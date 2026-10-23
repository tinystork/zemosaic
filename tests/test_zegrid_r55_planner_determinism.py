"""ZM-ZEGRID-R55-B — deterministic CPU planner, calibration guard, shrink-only API.

All tests are CPU-only, integer-only and fast (<10 s).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from zemosaic.core.zegrid import calibration as zcal
from zemosaic.core.zegrid import resource_envelope as re
from zemosaic.core.zegrid import scheduler as zs

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
MIB = 2 ** 20
GIB = 2 ** 30


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _ctx(**overrides):
    data = dict(
        phase=re.PHASE_CELL_STACK_STREAM,
        backend=re.BACKEND_CPU,
        budget_bytes=(4 * 8 * GIB) // 5,
        raw_available_bytes=8 * GIB,
        factor_num=4,
        factor_den=5,
        cpu_logical=8,
        units_total=10,
        shared_bytes=340 * MIB,
        per_unit_stack_bytes=200 * MIB,
        per_unit_cache_bytes=50 * MIB,
        per_unit_prefetch_bytes=50 * MIB,
        disk_free_bytes=10 * GIB,
        layout=(3, 4),
        refinement_rank=12,
        patch_hw=(200, 200),
        halo_px=32,
        channels=3,
        n_frames=50,
        cell_ids=tuple(f"r0000c{i:04d}" for i in range(10)),
        frame_ids=tuple(f"f{i}" for i in range(12)),
    )
    data.update(overrides)
    return zs.PlanContext(**data)


def _requests():
    return (
        zs.PhaseRequest(
            phase=re.PHASE_GAUGE,
            components={"gauge_worker_footprint_bytes": 400 * MIB},
            units=1,
            units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=re.PHASE_CACHE_BUILD,
            components={"cache_build_worker_footprint_bytes": 300 * MIB},
            units=1,
            units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=re.PHASE_ASSEMBLY,
            components={
                "halo_band_bytes": 4 * MIB,
                "retained_core_bytes": 100 * MIB,
            },
            units=1,
            units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=re.PHASE_EXPORT,
            components={
                "coverage_plane_bytes": 2 * MIB,
                "science_planes_bytes": 6 * MIB,
            },
            units=1,
            units_in_flight=1,
        ),
    )


def _profile(**overrides):
    data = dict(
        calibration_id="cal-test-cpu-tinydebian-v1",
        host_id="TINYDEBIAN",
        os="Linux-test",
        cpu_logical=8,
        ram_total_bytes=8 * GIB,
        phase=re.PHASE_CELL_STACK_STREAM,
        backend=re.BACKEND_CPU,
        path="product.test.cpu.stream",
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
    data.update(overrides)
    return zcal.CalibrationProfile(**data)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_candidate_enumeration_is_deterministic_and_canonical():
    ctx = _ctx()
    a = zs.enumerate_candidates(ctx)
    b = zs.enumerate_candidates(ctx)
    assert a == b
    assert len(a) > 1
    keys = [c.ordering_key() for c in a]
    assert keys == sorted(keys)
    # Fixed lexicographic key: (backend_rank, -tile, -K, -P, -W)
    first = a[0]
    assert first.backend == re.BACKEND_CPU
    assert first.tile_px == max(zs.PRODUCT_TILE_LADDER)
    assert first.ordering_key() == (
        0,
        -max(zs.PRODUCT_TILE_LADDER),
        -first.units_in_flight,
        -first.prefetch_units,
        -first.workers_per_unit,
    )


def test_plan_winner_and_hash_are_deterministic():
    p1 = zs.plan(_ctx(), phase_requests=_requests())
    p2 = zs.plan(_ctx(), phase_requests=_requests())
    assert p1.chosen == p2.chosen
    assert p1.plan_hash == p2.plan_hash
    assert p1.to_manifest_dict() == p2.to_manifest_dict()
    assert p1.plan_hash.startswith("sha256:")


def test_phase_request_order_does_not_change_the_plan_hash():
    p1 = zs.plan(_ctx(), phase_requests=_requests())
    p2 = zs.plan(_ctx(), phase_requests=tuple(reversed(_requests())))
    assert p1.plan_hash == p2.plan_hash
    assert [e.phase for e in p1.phases] == [e.phase for e in p2.phases]


def test_shuffled_cell_ids_do_not_change_the_plan():
    cells = [f"r0000c{i:04d}" for i in range(10)]
    p1 = zs.plan(_ctx(cell_ids=tuple(cells)), phase_requests=_requests())
    p2 = zs.plan(
        _ctx(cell_ids=tuple(reversed(cells))), phase_requests=_requests()
    )
    assert p1.plan_hash == p2.plan_hash
    assert p1.chosen == p2.chosen


def test_plan_hash_is_stable_across_process_restarts():
    script = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from zemosaic.core.zegrid import scheduler as zs
from test_zegrid_r55_planner_determinism import _ctx, _requests
print(zs.plan(_ctx(), phase_requests=_requests()).plan_hash)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + str(REPO_ROOT / "tests")
    proc = subprocess.run(
        [sys.executable, "-c", script, str(SRC)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    sub_hash = proc.stdout.strip().splitlines()[-1]
    local = zs.plan(_ctx(), phase_requests=_requests()).plan_hash
    assert sub_hash == local


def test_tie_break_chain_order():
    base = dict(
        backend=re.BACKEND_CPU,
        tile_px=64,
        units_in_flight=1,
        prefetch_units=0,
        workers_per_unit=1,
        host_peak_bytes=100,
        disk_bytes=0,
        device_peak_bytes=0,
        feasible=True,
        reason_code=re.REASON_UNCALIBRATED,
    )
    c_base = zs.Candidate(**base)
    # 1. throughput (lower makespan first)
    c_slow = zs.Candidate(**{**base, "predicted_makespan_ns": 200})
    c_fast = zs.Candidate(**{**base, "predicted_makespan_ns": 100})
    assert zs.objective_key(c_fast) < zs.objective_key(c_slow)
    # uncalibrated ranks behind calibrated
    assert zs.objective_key(c_slow) < zs.objective_key(c_base)
    # 2. lower host peak
    c_host = zs.Candidate(**{**base, "host_peak_bytes": 50})
    assert zs.objective_key(c_host) < zs.objective_key(c_base)
    # 3. lower device peak
    c_dev = zs.Candidate(**{**base, "device_peak_bytes": 0})
    c_dev_hi = zs.Candidate(**{**base, "device_peak_bytes": 10})
    assert zs.objective_key(c_dev) < zs.objective_key(c_dev_hi)
    # 4. CPU before GPU
    c_gpu = zs.Candidate(**{**base, "backend": re.BACKEND_GPU})
    assert zs.objective_key(c_base) < zs.objective_key(c_gpu)
    # 5. larger tile
    c_tile = zs.Candidate(**{**base, "tile_px": 128})
    assert zs.objective_key(c_tile) < zs.objective_key(c_base)
    # 6. lower units_in_flight
    c_k = zs.Candidate(**{**base, "units_in_flight": 2})
    assert zs.objective_key(c_base) < zs.objective_key(c_k)
    # 7. lower prefetch
    c_p = zs.Candidate(**{**base, "prefetch_units": 1})
    assert zs.objective_key(c_base) < zs.objective_key(c_p)
    # 8. lexicographic candidate key (its last element is -workers, so the
    # documented enumeration order prefers the higher worker count).
    c_w = zs.Candidate(**{**base, "workers_per_unit": 2})
    assert zs.objective_key(c_w) < zs.objective_key(c_base)


# ---------------------------------------------------------------------------
# Product ladder and calibration range guard
# ---------------------------------------------------------------------------

def test_product_ladder_only():
    from zemosaic import zemosaic_zegrid_mode as zz

    assert zs.PRODUCT_TILE_LADDER == tuple(zz.CPU_STREAMING_TILE_LADDER)
    assert 16 not in zs.PRODUCT_TILE_LADDER
    assert 24 not in zs.PRODUCT_TILE_LADDER
    assert min(zs.PRODUCT_TILE_LADDER) == zs.MIN_TILE_PX == 32


def test_calibration_in_range_and_out_of_range_never_extrapolates():
    profile = _profile()
    comps = zcal.estimate_components(
        profile, tile_px=64, n_frames=10, patch_area_px=40_000
    )
    assert all(isinstance(v, int) and v >= 0 for v in comps.values())
    assert sum(comps.values()) == zcal.calibrated_estimate(
        profile, tile_px=64, n_frames=10, patch_area_px=40_000
    ).subtotal_bytes
    with pytest.raises(zcal.CalibrationOutOfRange) as exc:
        zcal.estimate_components(
            profile, tile_px=256, n_frames=10, patch_area_px=40_000
        )
    assert exc.value.field_name == "tile_px"
    with pytest.raises(zcal.CalibrationOutOfRange):
        zcal.estimate_components(
            profile, tile_px=64, n_frames=101, patch_area_px=40_000
        )


def test_planner_marks_out_of_range_tiles_ineligible():
    ctx = _ctx(
        tile_ladder=(128, 96, 64, 32),
        profile=_profile(valid_ranges={
            "tile_px": (64, 128),
            "n_frames": (1, 100),
            "patch_area_px": (1, 1_000_000),
        }),
    )
    candidates = zs.enumerate_candidates(ctx)
    off_ladder = [c for c in candidates if c.tile_px == 32]
    assert off_ladder
    assert all(
        (not c.feasible) and c.reason_code == re.REASON_UNCALIBRATED
        for c in off_ladder
    )
    plan = zs.plan(ctx, phase_requests=_requests())
    assert plan.chosen.tile_px in (64, 96, 128)
    assert plan.chosen.reason_code == re.REASON_OK


def test_calibration_registry_scope_and_ooc_refusal():
    profile = _profile()
    registry = zcal.ProfileRegistry([profile])
    assert registry.lookup(
        phase=profile.phase,
        backend=profile.backend,
        host_class=profile.host_id,
        path=profile.path,
    ) is profile
    assert registry.lookup(
        phase=profile.phase,
        backend=profile.backend,
        host_class="OTHERHOST",
        path=profile.path,
    ) is None
    with pytest.raises(zcal.InvalidProfile, match="duplicate"):
        zcal.ProfileRegistry([profile, _profile()])
    ooc = _profile(
        calibration_id="cal-r54-ooc",
        path=zcal.EXPERIMENTAL_PATH,
        product_admissible=False,
    )
    ooc.validate()
    ooc_registry = zcal.ProfileRegistry([ooc])
    with pytest.raises(zcal.CalibrationNotAdmissible):
        ooc_registry.lookup(
            phase=ooc.phase,
            backend=ooc.backend,
            host_class=ooc.host_id,
            path=ooc.path,
        )
    with pytest.raises(zcal.InvalidProfile, match="product-admissible"):
        _profile(path=zcal.EXPERIMENTAL_PATH, product_admissible=True).validate()


def test_float_and_negative_coefficients_are_refused():
    profile = _profile()
    with pytest.raises(Exception):
        _profile(coefficients={**dict(profile.coefficients),
                               "process_baseline_bytes": 1.5}).validate()
    with pytest.raises(Exception):
        _profile(coefficients={**dict(profile.coefficients),
                               "process_baseline_bytes": -1}).validate()


# ---------------------------------------------------------------------------
# Disk refusal (D3)
# ---------------------------------------------------------------------------

def test_disk_shortfall_is_a_hard_refusal_independent_of_tile():
    ctx = _ctx(disk_free_bytes=1)
    candidates = zs.enumerate_candidates(ctx)
    assert candidates
    assert all(c.reason_code == re.REASON_REFUSE_DISK for c in candidates)
    assert all(not c.feasible for c in candidates)
    # A persistent aligned cache is independent of the stack tile: the SAME
    # (K, P) produces the same disk bytes at every tile, so a tile downgrade is
    # not offered as a disk remedy.
    per_key: dict[tuple[int, int], set[int]] = {}
    for c in candidates:
        per_key.setdefault((c.units_in_flight, c.prefetch_units), set()).add(
            c.disk_bytes
        )
    assert all(len(values) == 1 for values in per_key.values())
    with pytest.raises(zs.PlanRefused) as exc:
        zs.plan(ctx, phase_requests=_requests())
    assert exc.value.reason_code == re.REASON_REFUSE_DISK


def test_disk_term_counts_prefetch_buffers_once_per_depth():
    assert zs.disk_residency_bytes(
        units_in_flight=2,
        prefetch_units=1,
        per_unit_cache_bytes=100,
        per_unit_prefetch_bytes=100,
    ) == 300
    assert zs.disk_residency_bytes(
        units_in_flight=2,
        prefetch_units=0,
        per_unit_cache_bytes=100,
        per_unit_prefetch_bytes=100,
    ) == 200
    # Depth is never multiplied by the frame count or the tile count.
    with pytest.raises(re.InvalidEnvelope):
        zs.disk_residency_bytes(
            units_in_flight=-1,
            prefetch_units=0,
            per_unit_cache_bytes=100,
            per_unit_prefetch_bytes=100,
        )


def test_gpu_backend_is_out_of_scope_and_fails_visibly():
    with pytest.raises(zs.SchedulerConfigError, match="CPU backend only"):
        zs.build_observe_plan(
            phase=re.PHASE_CELL_STACK_STREAM,
            backend=re.BACKEND_GPU,
            budget_bytes=1 * GIB,
            raw_available_bytes=1 * GIB,
            factor_num=1,
            factor_den=1,
            cpu_logical=8,
            shared_bytes=1,
            per_unit_stack_bytes=1,
            per_unit_cache_bytes=0,
            per_unit_prefetch_bytes=0,
            disk_free_bytes=1,
            layout=(1, 1),
            refinement_rank=1,
            patch_hw=(1, 1),
            halo_px=0,
            channels=3,
            n_frames=1,
            phase_requests=_requests(),
            units_total=1,
        )


# ---------------------------------------------------------------------------
# Shrink-only API
# ---------------------------------------------------------------------------

def test_shrink_sequence_uses_the_documented_order():
    ctx = _ctx()
    candidate = zs.Candidate(
        backend=re.BACKEND_CPU,
        tile_px=128,
        units_in_flight=3,
        prefetch_units=1,
        workers_per_unit=6,
        host_peak_bytes=1,
        disk_bytes=0,
        device_peak_bytes=0,
        feasible=True,
        reason_code=re.REASON_UNCALIBRATED,
    )
    steps = zs.shrink_sequence(candidate, ctx)
    fields = [name for name, _ in steps]
    # First any prefetch step, then workers, then K, then tile; backend last.
    assert fields[0] == "prefetch_units"
    distinct = []
    for name in fields:
        if not distinct or distinct[-1] != name:
            distinct.append(name)
    assert distinct == ["prefetch_units", "workers_per_unit", "units_in_flight", "tile_px"]
    final = steps[-1][1]
    assert final.prefetch_units == 0
    assert final.workers_per_unit == 1
    assert final.units_in_flight == 1
    assert final.tile_px == 32


def test_growth_is_rejected_on_every_field():
    base = zs.Candidate(
        backend=re.BACKEND_CPU,
        tile_px=64,
        units_in_flight=1,
        prefetch_units=0,
        workers_per_unit=1,
        host_peak_bytes=1,
        disk_bytes=0,
        device_peak_bytes=0,
        feasible=True,
        reason_code=re.REASON_UNCALIBRATED,
    )
    for changed in (
        {"tile_px": 128},
        {"units_in_flight": 2},
        {"prefetch_units": 1},
        {"workers_per_unit": 2},
        {"backend": re.BACKEND_GPU},
    ):
        grown = zs.Candidate(**{**base.__dict__, **changed})
        with pytest.raises(zs.GrowthRejected):
            zs.assert_shrink_only(base, grown)
    # Legitimate degradation is accepted.
    gpu = zs.Candidate(**{**base.__dict__, "backend": re.BACKEND_GPU})
    zs.assert_shrink_only(gpu, base)
    zs.assert_shrink_only(base, zs.Candidate(**{**base.__dict__, "tile_px": 32}))


def test_stream_inmem_transitions_are_growth():
    ctx = _ctx()
    inmem = zs.Candidate(
        backend=re.BACKEND_CPU,
        tile_px=None,
        units_in_flight=1,
        prefetch_units=0,
        workers_per_unit=1,
        host_peak_bytes=1,
        disk_bytes=0,
        device_peak_bytes=0,
        feasible=True,
        reason_code=re.REASON_UNCALIBRATED,
    )
    stream = zs.Candidate(**{**inmem.__dict__, "tile_px": 64})
    with pytest.raises(zs.GrowthRejected):
        zs.assert_shrink_only(inmem, stream)
    with pytest.raises(zs.GrowthRejected):
        zs.assert_shrink_only(stream, inmem)


def test_frozen_identity_rejects_every_scientific_change():
    ident = zs.FrozenIdentity(
        layout=(3, 4),
        cell_ids=("r0000c0000", "r0000c0001"),
        frame_ids=("f0", "f1"),
        halo_px=32,
        refinement_rank=12,
    )
    zs.assert_same_identity(ident, ident)
    for changed in (
        {"layout": (4, 3)},
        {"cell_ids": ("r0000c0000",)},
        {"frame_ids": ("f0",)},
        {"halo_px": 64},
        {"refinement_rank": 11},
    ):
        with pytest.raises(zs.GrowthRejected):
            zs.assert_same_identity(
                ident, zs.FrozenIdentity(**{**ident.__dict__, **changed})
            )


# ---------------------------------------------------------------------------
# Reason-code closure
# ---------------------------------------------------------------------------

def test_reason_codes_are_unique_and_every_emission_is_closed():
    assert len(re.REASON_CODES) == len(set(re.REASON_CODES))
    assert all(code.startswith("R55_") for code in re.REASON_CODES)
    ctx = _ctx()
    for c in zs.enumerate_candidates(ctx):
        assert c.reason_code in re.REASON_CODES
    plan = zs.plan(ctx, phase_requests=_requests())
    for decision in plan.decisions:
        assert decision["reason_code"] in re.REASON_CODES
    with pytest.raises(zs.PlanRefused) as exc:
        zs.plan(_ctx(disk_free_bytes=1), phase_requests=_requests())
    assert exc.value.reason_code in re.REASON_CODES
