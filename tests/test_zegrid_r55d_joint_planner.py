"""ZM-ZEGRID-R55-D — deterministic joint CPU/GPU planner core (D1).

All tests are integer-only, CPU-only and fast. No CuPy/CUDA is imported or
probed; GPU facts are explicit inputs.
"""

from __future__ import annotations

import dataclasses
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
CELLS = tuple(f"r0000c{i:04d}" for i in range(10))
FRAMES = tuple(f"f{i}" for i in range(12))


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
        cell_ids=CELLS,
        frame_ids=FRAMES,
    )
    data.update(overrides)
    return zs.PlanContext(**data)


def _gpu_ctx(**overrides):
    data = dict(backend=re.BACKEND_GPU)
    data.update(overrides)
    return _ctx(**data)


def _requests(phase=re.PHASE_CELL_STACK_STREAM):
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
            overlaps_with=(phase,),
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


def _profile(backend=re.BACKEND_CPU, **overrides):
    data = dict(
        calibration_id=f"cal-test-{backend}-tinydebian-v1",
        host_id="TINYDEBIAN",
        os="Linux-test",
        cpu_logical=8,
        ram_total_bytes=8 * GIB,
        phase=re.PHASE_CELL_STACK_STREAM,
        backend=backend,
        path=f"product.test.{backend}.stream",
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


def _device_peaks(tiles=zs.PRODUCT_TILE_LADDER):
    return {
        int(tile): int(tile) * int(tile) * 3 * 3 * 4
        for tile in tiles
    }


def _gpu_facts(**overrides):
    data = dict(
        requested=True,
        available=True,
        parity_qualified=True,
        vram_total_bytes=8 * GIB,
        vram_free_bytes=7 * GIB,
        vram_budget_bytes=4 * GIB,
        device_name="test-gpu-0",
        device_peak_bytes_by_tile=_device_peaks(),
    )
    data.update(overrides)
    return zs.GpuFacts(**data)


def _joint(*, cpu=None, gpu=None, gpu_facts=None):
    cpu = _ctx() if cpu is None else cpu
    gpu = _gpu_ctx() if gpu is None else gpu
    gpu_facts = _gpu_facts() if gpu_facts is None else gpu_facts
    return zs.JointPlanInput((cpu, gpu), gpu_facts)


def _evidence_fixture():
    cpu = _ctx()
    gpu = _gpu_ctx(
        profile=_profile(re.BACKEND_GPU),
        time_model=lambda tile, k, prefetch: (
            1000 * (129 - int(tile)) + 50 * int(prefetch)
        ),
    )
    return zs.JointPlanInput((gpu, cpu), _gpu_facts()), _requests()


def _find_gpu(joint, **criteria):
    for candidate in zs.enumerate_joint_candidates(joint):
        if candidate.backend != re.BACKEND_GPU:
            continue
        if all(getattr(candidate, key) == value for key, value in criteria.items()):
            return candidate
    raise AssertionError(f"no GPU candidate matching {criteria}")


def _find_cpu(joint, **criteria):
    for candidate in zs.enumerate_joint_candidates(joint):
        if candidate.backend != re.BACKEND_CPU:
            continue
        if all(getattr(candidate, key) == value for key, value in criteria.items()):
            return candidate
    raise AssertionError(f"no CPU candidate matching {criteria}")


# ---------------------------------------------------------------------------
# 1. CPU-only compatibility
# ---------------------------------------------------------------------------

def test_cpu_only_joint_is_byte_identical_to_r55b():
    context = _ctx()
    legacy = zs.plan(context, phase_requests=_requests())
    joint = zs.JointPlanInput((context,))
    joint_plan = zs.plan_joint(joint, phase_requests=_requests())

    assert joint_plan.plan_hash == legacy.plan_hash
    assert joint_plan.phases == legacy.phases
    assert joint_plan.chosen == legacy.chosen
    assert joint_plan.decisions == legacy.decisions
    assert joint_plan.to_manifest_dict() == legacy.to_manifest_dict()
    assert (
        zs.enumerate_joint_candidates(joint)
        == zs.enumerate_candidates(context)
    )


def test_cpu_delegation_via_build_joint_plan_set_is_identical():
    context = _ctx()
    legacy = zs.plan(context, phase_requests=_requests())
    joint = zs.JointPlanInput((context,))
    rebuilt = zs.build_joint_plan_set(
        joint,
        candidate=legacy.chosen,
        phase_requests=_requests(),
    )
    assert rebuilt.plan_hash == legacy.plan_hash
    assert rebuilt.to_manifest_dict() == legacy.to_manifest_dict()


def test_cpu_behavior_unchanged_when_gpu_is_blocked():
    cpu = _ctx(
        profile=_profile(re.BACKEND_CPU),
        time_model=lambda tile, k, prefetch: 5000 * (129 - int(tile)),
    )
    legacy = zs.plan(cpu, phase_requests=_requests())
    for facts in (
        _gpu_facts(requested=False),
        _gpu_facts(available=False),
        _gpu_facts(parity_qualified=False),
    ):
        joint = zs.JointPlanInput((cpu, _gpu_ctx()), facts)
        planned = zs.plan_joint(joint, phase_requests=_requests())
        assert planned.plan_hash == legacy.plan_hash
        assert planned.chosen == legacy.chosen
        assert planned.to_manifest_dict() == legacy.to_manifest_dict()


# ---------------------------------------------------------------------------
# 2. Candidate tuple, canonical order, single owner
# ---------------------------------------------------------------------------

def test_joint_tuple_is_exact_and_ordered_cpu_first():
    candidates = zs.enumerate_joint_candidates(_joint())
    assert candidates
    keys = [candidate.ordering_key() for candidate in candidates]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)

    seen_gpu = False
    for candidate in candidates:
        assert candidate.backend in re.BACKENDS
        if candidate.backend == re.BACKEND_GPU:
            seen_gpu = True
            assert candidate.units_in_flight == 1
            assert candidate.device_peak_bytes > 0
        else:
            assert not seen_gpu, "CPU candidates must be enumerated first"
        # The exact joint tuple is carried by these five fields.
        assert (
            candidate.backend,
            candidate.tile_px,
            candidate.units_in_flight,
            candidate.prefetch_units,
            candidate.workers_per_unit,
        ) == (
            candidate.backend,
            candidate.tile_px,
            1 if candidate.backend == re.BACKEND_GPU else candidate.units_in_flight,
            candidate.prefetch_units,
            candidate.workers_per_unit,
        )


def test_gpu_is_single_owner_for_every_gpu_candidate():
    joint = _joint()
    gpu_candidates = [
        candidate
        for candidate in zs.enumerate_joint_candidates(joint)
        if candidate.backend == re.BACKEND_GPU
    ]
    assert gpu_candidates
    assert {candidate.units_in_flight for candidate in gpu_candidates} == {1}


# ---------------------------------------------------------------------------
# 3. Determinism across input/mapping/cell/process order
# ---------------------------------------------------------------------------

def test_enumeration_independent_of_context_mapping_and_cell_order():
    reversed_cells = tuple(reversed(CELLS))
    peaks = _device_peaks()
    reversed_peaks = {tile: peaks[tile] for tile in reversed(list(peaks))}

    cpu_a = _ctx(cell_ids=CELLS)
    gpu_a = _gpu_ctx()
    cpu_b = _ctx(cell_ids=reversed_cells)
    gpu_b = _gpu_ctx(cell_ids=reversed_cells)

    joint_a = zs.JointPlanInput((cpu_a, gpu_a), _gpu_facts())
    joint_b = zs.JointPlanInput(
        (gpu_b, cpu_b),
        _gpu_facts(device_peak_bytes_by_tile=reversed_peaks),
    )

    assert (
        zs.enumerate_joint_candidates(joint_a)
        == zs.enumerate_joint_candidates(joint_b)
    )
    assert zs.joint_refusal_trail(joint_a) == zs.joint_refusal_trail(joint_b)
    assert zs.joint_enumeration_hash(joint_a) == zs.joint_enumeration_hash(joint_b)
    assert zs.joint_evidence_json(
        joint_a, phase_requests=_requests()
    ) == zs.joint_evidence_json(joint_b, phase_requests=_requests())


def test_joint_evidence_is_stable_across_process_restarts():
    joint, requests = _evidence_fixture()
    local = zs.joint_evidence_json(joint, phase_requests=requests)
    script = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "from test_zegrid_r55d_joint_planner import _evidence_fixture\n"
        "from zemosaic.core.zegrid import scheduler as zs\n"
        "joint, requests = _evidence_fixture()\n"
        "sys.stdout.write(zs.joint_evidence_json("
        "joint, phase_requests=requests))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + str(REPO_ROOT / "tests")
    outputs = []
    for _ in range(2):
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(SRC),
                str(REPO_ROOT / "tests"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        outputs.append(proc.stdout)
    assert outputs[0] == outputs[1] == local
    assert zs.joint_evidence_hash(joint, phase_requests=requests).startswith(
        "sha256:"
    )


def test_enumeration_hash_is_repeatable():
    joint = _joint()
    assert zs.joint_enumeration_hash(joint) == zs.joint_enumeration_hash(joint)
    assert zs.joint_enumeration_payload(joint) == zs.joint_enumeration_payload(
        joint
    )


# ---------------------------------------------------------------------------
# 4. GPU hard constraints / refusals
# ---------------------------------------------------------------------------

def test_gpu_vram_exact_fit_passes_and_one_byte_less_fails():
    peak = _device_peaks()[128]
    exact = _joint(gpu_facts=_gpu_facts(vram_budget_bytes=peak))
    candidate = _find_gpu(
        exact, tile_px=128, prefetch_units=0, workers_per_unit=1
    )
    assert candidate.feasible is True
    assert candidate.reason_code == re.REASON_UNCALIBRATED
    assert candidate.device_peak_bytes == peak

    short = _joint(gpu_facts=_gpu_facts(vram_budget_bytes=peak - 1))
    refused = _find_gpu(
        short, tile_px=128, prefetch_units=0, workers_per_unit=1
    )
    assert refused.feasible is False
    assert refused.reason_code == re.REASON_REFUSE_VRAM
    assert "device peak" in (refused.infeasible_reason or "")


@pytest.mark.parametrize(
    "overrides,expected_reason",
    [
        ({"requested": False}, re.REASON_DEFAULT_OFF),
        ({"available": False}, re.REASON_GPU_UNAVAILABLE),
        ({"parity_qualified": False}, re.REASON_PARITY_FAIL),
        ({"vram_budget_bytes": 0}, re.REASON_REFUSE_VRAM),
    ],
)
def test_gpu_gate_refusals_are_closed_and_explained(
    overrides, expected_reason
):
    joint = _joint(gpu_facts=_gpu_facts(**overrides))
    candidates = zs.enumerate_joint_candidates(joint)
    assert candidates
    assert all(candidate.backend == re.BACKEND_CPU for candidate in candidates)
    assert candidates == zs.enumerate_candidates(joint.cpu_context)
    trail = zs.joint_refusal_trail(joint)
    assert len(trail) == 2
    assert trail[1].backend == re.BACKEND_GPU
    assert trail[1].reason_code == expected_reason
    assert expected_reason in re.REASON_CODES


def test_gpu_with_positive_but_too_small_budget_is_refused_per_candidate():
    joint = _joint(gpu_facts=_gpu_facts(vram_budget_bytes=1))
    gpu_candidates = [
        candidate
        for candidate in zs.enumerate_joint_candidates(joint)
        if candidate.backend == re.BACKEND_GPU
    ]
    assert gpu_candidates
    assert all(
        candidate.feasible is False
        and candidate.reason_code == re.REASON_REFUSE_VRAM
        for candidate in gpu_candidates
    )
    assert zs.select_joint_candidate(joint).backend == re.BACKEND_CPU


def test_disk_shortfall_is_closed_refusal():
    joint = _joint(
        cpu=_ctx(disk_free_bytes=0),
        gpu=_gpu_ctx(disk_free_bytes=0),
    )
    candidates = zs.enumerate_joint_candidates(joint)
    assert candidates
    assert {candidate.reason_code for candidate in candidates} == {
        re.REASON_REFUSE_DISK
    }
    with pytest.raises(zs.PlanRefused) as exercised:
        zs.plan_joint(joint, phase_requests=_requests())
    assert exercised.value.reason_code == re.REASON_REFUSE_DISK


def test_host_shortfall_is_closed_refusal():
    tiny = dict(
        raw_available_bytes=1,
        factor_num=1,
        factor_den=1,
        budget_bytes=1,
    )
    joint = _joint(
        cpu=_ctx(**tiny),
        gpu=_gpu_ctx(**tiny),
    )
    with pytest.raises(zs.PlanRefused) as exercised:
        zs.plan_joint(joint, phase_requests=_requests())
    assert exercised.value.reason_code == re.REASON_REFUSE_HOST_RAM


# ---------------------------------------------------------------------------
# 5. Strict input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "overrides",
    [
        {"requested": 1},
        {"requested": "yes"},
        {"available": 0},
        {"available": None},
        {"parity_qualified": "true"},
        {"parity_qualified": 1},
        {"vram_total_bytes": True},
        {"vram_free_bytes": 1.0},
        {"vram_budget_bytes": -1},
        {"vram_budget_bytes": True},
        {"device_name": ""},
        {"device_name": " test-gpu-0 "},
        {"device_peak_bytes_by_tile": {128: True}},
        {"device_peak_bytes_by_tile": {128: 1.0}},
        {"device_peak_bytes_by_tile": {128: -1}},
        {"device_peak_bytes_by_tile": "not-a-map"},
        {"device_peak_bytes_by_tile": ((128, 1), (128, 2))},
        {"prefetch_overlap_measured": 1},
    ],
)
def test_gpu_facts_reject_bad_types(overrides):
    with pytest.raises(zs.SchedulerConfigError):
        _gpu_facts(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"vram_free_bytes": 8 * GIB + 1},
        {"vram_budget_bytes": 7 * GIB + 1},
    ],
)
def test_gpu_facts_reject_contradictory_bytes(overrides):
    with pytest.raises(zs.SchedulerConfigError):
        _gpu_facts(**overrides)


def test_gpu_device_peak_map_must_cover_exactly_the_declared_tiles():
    peaks = _device_peaks()
    missing = dict(peaks)
    missing.pop(128)
    with pytest.raises(zs.SchedulerConfigError):
        _joint(gpu_facts=_gpu_facts(device_peak_bytes_by_tile=missing))

    extra = dict(peaks)
    extra[256] = 1
    with pytest.raises(zs.SchedulerConfigError):
        _joint(gpu_facts=_gpu_facts(device_peak_bytes_by_tile=extra))


def test_joint_context_requires_exactly_one_cpu_and_sequence():
    with pytest.raises(zs.SchedulerConfigError):
        zs.JointPlanInput((), None)
    with pytest.raises(zs.SchedulerConfigError):
        zs.JointPlanInput("not-a-sequence", None)
    with pytest.raises(zs.SchedulerConfigError):
        zs.JointPlanInput((_gpu_ctx(),), _gpu_facts())
    with pytest.raises(zs.SchedulerConfigError):
        zs.JointPlanInput((_ctx(), _ctx()), None)
    with pytest.raises(zs.SchedulerConfigError):
        zs.JointPlanInput((_ctx(), "gpu"), _gpu_facts())


def test_gpu_context_and_facts_must_be_supplied_together():
    with pytest.raises(zs.SchedulerConfigError):
        zs.JointPlanInput((_ctx(), _gpu_ctx()))
    with pytest.raises(zs.SchedulerConfigError):
        zs.JointPlanInput((_ctx(),), _gpu_facts())


def test_gpu_inmem_stack_is_not_modelled():
    with pytest.raises(zs.SchedulerConfigError):
        zs.JointPlanInput(
            (
                _ctx(phase=re.PHASE_CELL_STACK_INMEM),
                _gpu_ctx(phase=re.PHASE_CELL_STACK_INMEM),
            ),
            _gpu_facts(),
        )


def test_joint_contexts_must_share_host_budget_and_science_identity():
    facts = _gpu_facts()
    with pytest.raises(zs.SchedulerConfigError):
        zs.JointPlanInput(
            (
                _ctx(),
                _gpu_ctx(
                    factor_num=3,
                    factor_den=4,
                    budget_bytes=(3 * 8 * GIB) // 4,
                ),
            ),
            facts,
        )
    with pytest.raises(zs.SchedulerConfigError):
        zs.JointPlanInput(
            (_ctx(), _gpu_ctx(margin_num=10, margin_den=100)), facts
        )
    different_cells = list(CELLS)
    different_cells[-1] = "r9999c9999"
    with pytest.raises(zs.SchedulerConfigError):
        zs.JointPlanInput(
            (_ctx(), _gpu_ctx(cell_ids=tuple(different_cells))), facts
        )
    with pytest.raises(zs.SchedulerConfigError):
        zs.JointPlanInput((_ctx(), _gpu_ctx(run_id="other-run")), facts)
    with pytest.raises(zs.SchedulerConfigError):
        zs.JointPlanInput((_ctx(), _gpu_ctx(cpu_logical=6)), facts)


def test_all_joint_candidate_reasons_are_closed():
    candidates = zs.enumerate_joint_candidates(_joint())
    assert {candidate.reason_code for candidate in candidates} <= re.REASON_CODES
    for entry in zs.joint_refusal_trail(_joint()):
        assert entry.reason_code in re.REASON_CODES


# ---------------------------------------------------------------------------
# 6. Objective / tie-break
# ---------------------------------------------------------------------------

def test_uncalibrated_ranks_behind_calibrated_and_cpu_wins_ties():
    base = dict(
        backend=re.BACKEND_CPU,
        tile_px=128,
        units_in_flight=1,
        prefetch_units=0,
        workers_per_unit=1,
        host_peak_bytes=100,
        disk_bytes=10,
        device_peak_bytes=0,
        feasible=True,
        reason_code=re.REASON_UNCALIBRATED,
    )
    cpu = zs.Candidate(**base)
    gpu = zs.Candidate(**{**base, "backend": re.BACKEND_GPU})
    assert zs.objective_key(cpu) < zs.objective_key(gpu)

    calibrated = zs.Candidate(**{**base, "predicted_makespan_ns": 100})
    uncalibrated = zs.Candidate(**base)
    assert zs.objective_key(calibrated) < zs.objective_key(uncalibrated)


def test_cpu_calibrated_beats_gpu_uncalibrated():
    cpu = _ctx(
        profile=_profile(re.BACKEND_CPU),
        time_model=lambda tile, k, prefetch: 5000 * (129 - int(tile)),
    )
    joint = _joint(cpu=cpu, gpu=_gpu_ctx())
    winner = zs.select_joint_candidate(joint)
    assert winner.backend == re.BACKEND_CPU
    assert winner.predicted_makespan_ns is not None


def test_gpu_calibrated_wins_only_when_its_objective_wins():
    fast_gpu = _gpu_ctx(
        profile=_profile(re.BACKEND_GPU),
        time_model=lambda tile, k, prefetch: 1000 * (129 - int(tile)),
    )
    assert (
        zs.select_joint_candidate(
            _joint(cpu=_ctx(), gpu=fast_gpu)
        ).backend
        == re.BACKEND_GPU
    )

    slow_gpu = _gpu_ctx(
        profile=_profile(re.BACKEND_GPU),
        time_model=lambda tile, k, prefetch: 10 ** 9,
    )
    fast_cpu = _ctx(
        profile=_profile(re.BACKEND_CPU),
        time_model=lambda tile, k, prefetch: 1,
    )
    assert (
        zs.select_joint_candidate(
            _joint(cpu=fast_cpu, gpu=slow_gpu)
        ).backend
        == re.BACKEND_CPU
    )


def test_prefetch_gets_no_speed_credit_without_measured_overlap():
    gpu = _gpu_ctx(
        profile=_profile(re.BACKEND_GPU),
        time_model=lambda tile, k, prefetch: 1000 - 100 * int(prefetch),
    )
    unmeasured = _joint(gpu=gpu, gpu_facts=_gpu_facts())
    measured = _joint(
        gpu=gpu,
        gpu_facts=_gpu_facts(prefetch_overlap_measured=True),
    )
    for joint, expect_different in ((unmeasured, False), (measured, True)):
        zero = _find_gpu(
            joint,
            tile_px=128,
            units_in_flight=1,
            prefetch_units=0,
            workers_per_unit=1,
        )
        one = _find_gpu(
            joint,
            tile_px=128,
            units_in_flight=1,
            prefetch_units=1,
            workers_per_unit=1,
        )
        assert zero.predicted_makespan_ns == 1000 * 10
        if expect_different:
            assert one.predicted_makespan_ns == 900 * 10
        else:
            assert one.predicted_makespan_ns == zero.predicted_makespan_ns


# ---------------------------------------------------------------------------
# 7. Truthful mixed-backend PlanSet
# ---------------------------------------------------------------------------

def test_mixed_backend_plan_set_has_truthful_phase_backends():
    joint, requests = _evidence_fixture()
    plan = zs.plan_joint(joint, phase_requests=requests)
    assert plan.chosen.backend == re.BACKEND_GPU
    assert plan.chosen.units_in_flight == 1
    assert plan.scheduler_version == zs.JOINT_SCHEDULER_VERSION

    phases = {envelope.phase: envelope for envelope in plan.phases}
    stack = phases[re.PHASE_CELL_STACK_STREAM]
    assert stack.backend == re.BACKEND_GPU
    assert stack.device is not None
    assert stack.device.budget_bytes == joint.gpu_facts.vram_budget_bytes
    assert stack.device.vram_total_bytes == joint.gpu_facts.vram_total_bytes
    assert stack.device.vram_free_bytes == joint.gpu_facts.vram_free_bytes
    assert (
        stack.accounting.device_peak_bytes
        == plan.chosen.device_peak_bytes
    )

    for phase in (
        re.PHASE_GAUGE,
        re.PHASE_CACHE_BUILD,
        re.PHASE_ASSEMBLY,
        re.PHASE_EXPORT,
    ):
        envelope = phases[phase]
        assert envelope.backend == re.BACKEND_CPU
        assert envelope.device is None
        assert envelope.accounting.device_peak_bytes == 0

    assert stack.overlaps_with == (re.PHASE_ASSEMBLY,)
    assert phases[re.PHASE_ASSEMBLY].overlaps_with == (
        re.PHASE_CELL_STACK_STREAM,
    )

    # The explicit VRAM budget is authoritative: the host margin is not applied
    # a second time to the device aggregate.
    assert plan.aggregate_device_peak_bytes == plan.chosen.device_peak_bytes
    assert plan.aggregate_host_peak_bytes == re.ceil_margin_div(
        plan.aggregate_host_peak_bytes_raw,
        joint.cpu_context.margin_num,
        joint.cpu_context.margin_den,
    )
    assert plan.aggregate_disk_bytes == re.aggregate_disk_bytes(plan.phases)
    assert plan.plan_hash.startswith("sha256:")
    assert plan.identity == joint.cpu_context.identity

    manifest = plan.to_manifest_dict()
    assert manifest["joint_schema"] == zs.JOINT_SCHEMA
    assert manifest["scheduler_version"] == zs.JOINT_SCHEDULER_VERSION
    assert "R55-D joint" in manifest["note"]


def test_build_joint_plan_set_rejects_tampered_candidate():
    joint, requests = _evidence_fixture()
    winner = zs.select_joint_candidate(joint)
    tampered = dataclasses.replace(winner, units_in_flight=2)
    with pytest.raises(zs.SchedulerConfigError):
        zs.build_joint_plan_set(
            joint, candidate=tampered, phase_requests=requests
        )


def test_gpu_unavailable_still_plans_cpu():
    joint = _joint(gpu_facts=_gpu_facts(available=False))
    plan = zs.plan_joint(joint, phase_requests=_requests())
    assert plan.chosen.backend == re.BACKEND_CPU
    assert (
        zs.joint_refusal_trail(joint)[1].reason_code
        == re.REASON_GPU_UNAVAILABLE
    )


# ---------------------------------------------------------------------------
# 8. Shrink/growth and identity
# ---------------------------------------------------------------------------

def test_joint_shrink_guard_allows_gpu_to_cpu_and_rejects_growth():
    # A true shrink: the CPU static bound is below the calibrated GPU stack, so
    # the GPU->CPU step reduces host, device and disk bytes.
    cpu_context = _ctx(per_unit_stack_bytes=50 * MIB)
    gpu_context = _gpu_ctx(
        profile=_profile(re.BACKEND_GPU),
        time_model=lambda tile, k, prefetch: 1000 * (129 - int(tile)),
    )
    joint = zs.JointPlanInput(
        (cpu_context, gpu_context), _gpu_facts()
    )
    gpu = _find_gpu(
        joint,
        tile_px=128,
        units_in_flight=1,
        prefetch_units=0,
        workers_per_unit=1,
    )
    cpu = _find_cpu(
        joint,
        tile_px=128,
        units_in_flight=1,
        prefetch_units=0,
        workers_per_unit=1,
    )
    zs.assert_joint_shrink_only(gpu, cpu)

    with pytest.raises(zs.GrowthRejected):
        zs.assert_joint_shrink_only(cpu, gpu)

    for changed in (
        dataclasses.replace(cpu, host_peak_bytes=cpu.host_peak_bytes + 1),
        dataclasses.replace(cpu, device_peak_bytes=cpu.device_peak_bytes + 1),
        dataclasses.replace(cpu, disk_bytes=cpu.disk_bytes + 1),
    ):
        with pytest.raises(zs.GrowthRejected):
            zs.assert_joint_shrink_only(cpu, changed)


def test_gpu_to_cpu_is_refused_when_it_would_grow_host_bytes():
    # A calibrated GPU stack can be cheaper in host bytes than the CPU static
    # bound. "GPU->CPU only" is the allowed backend direction, but it is still
    # bound by the no-resource-growth invariant: a fallback must rebuild and
    # re-validate instead of being offered as a legal shrink.
    joint, _requests_value = _evidence_fixture()
    gpu = _find_gpu(
        joint,
        tile_px=128,
        units_in_flight=1,
        prefetch_units=0,
        workers_per_unit=1,
    )
    cpu = _find_cpu(
        joint,
        tile_px=128,
        units_in_flight=1,
        prefetch_units=0,
        workers_per_unit=1,
    )
    assert cpu.host_peak_bytes > gpu.host_peak_bytes
    with pytest.raises(zs.GrowthRejected):
        zs.assert_joint_shrink_only(gpu, cpu)


def test_joint_shrink_sequence_reaches_cpu_backend():
    joint, _requests_value = _evidence_fixture()
    gpu = _find_gpu(
        joint,
        tile_px=128,
        units_in_flight=1,
        prefetch_units=1,
        workers_per_unit=1,
    )
    steps = zs.shrink_sequence(gpu, joint.gpu_context)
    assert any(name == "backend" for name, _ in steps)
    assert steps[-1][1].backend == re.BACKEND_CPU


def test_frozen_identity_is_never_shrinkable():
    identity = _ctx().identity
    changed = dataclasses.replace(identity, layout=(4, 5))
    with pytest.raises(zs.GrowthRejected):
        zs.assert_same_identity(identity, changed)


# ---------------------------------------------------------------------------
# 9. Import laziness
# ---------------------------------------------------------------------------

def test_normal_import_does_not_load_cupy_or_cuda_readiness():
    script = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import zemosaic.core.zegrid.scheduler\n"
        "print('cupy' in sys.modules)\n"
        "print('zemosaic.core.zegrid.gpu_readiness' in sys.modules)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(SRC)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines() == ["False", "False"]


def test_normal_import_is_lazy_even_when_cupy_is_importable(tmp_path):
    # Simulate an installed CuPy: a fake importable module that EXPLODES if the
    # scheduler imports it. The scheduler must stay lazy (it never probes).
    fake = tmp_path / "cupy.py"
    fake.write_text(
        "raise AssertionError('scheduler imported CuPy during normal import')\n",
        encoding="utf-8",
    )
    script = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import zemosaic.core.zegrid.scheduler\n"
        "print('cupy' in sys.modules)\n"
        "print('zemosaic.core.zegrid.gpu_readiness' in sys.modules)\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path) + os.pathsep + str(SRC),
    }
    proc = subprocess.run(
        [sys.executable, "-c", script, str(SRC)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines() == ["False", "False"]
