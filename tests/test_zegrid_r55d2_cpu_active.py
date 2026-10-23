"""ZM-ZEGRID-R55-D2 — ACTIVE CPU integration.

Acceptance exercised here:

* strict mode normalization (``active`` additive, default still ``legacy``);
* exact CPU inmem/stream mapping onto the product's existing knobs;
* GPU / prefetch-positive / off-ladder / stale-tampered winners fail closed;
* PLAN 80 % + fresh ADMIT 90 % semantics with scheduler-produced shrink-only
  pressure corrections on frozen identity;
* zero-gauge/zero-stack refusal witnesses for every fatal active refusal;
* a REAL hermetic product-path active-vs-legacy byte-identity witness whose
  active manifest proves the knobs that actually ran;
* deterministic two-build active evidence and active->legacy rollback;
* scheduler import laziness with CuPy installed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import ram_contract as rcontract
from zemosaic.core.zegrid import resource_envelope as re
from zemosaic.core.zegrid import scheduler as zs

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "zegrid_routine" / "make_routine_corpus.py"
WITNESS_HALO_PX = 32
WITNESS_AVAILABLE_BYTES = 12 * 2 ** 30
MIB = 2 ** 20
GIB = 2 ** 30
CELLS = tuple(f"r0000c{i:04d}" for i in range(6))
FRAMES = tuple(f"f{i}" for i in range(8))
VOLATILE_MANIFEST_KEYS = (
    "timings",
    "peak_rss_kib",
    "aggregate_peak_rss_kib",
    "gauge_diagnostics",
    "per_cell_diagnostics",
    "reprojection",
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

class _SeqMemory:
    """Deterministic available-RAM sequence (first = PLAN observation)."""

    def __init__(self, first, then):
        self.first = int(first)
        self.then = int(then)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.first if self.calls == 1 else self.then


class _ScheduleMemory:
    """Explicit per-call RAM schedule, then a fixed default."""

    def __init__(self, values, *, default):
        self.values = [int(v) for v in values]
        self.default = int(default)
        self.calls = 0

    def __call__(self):
        idx = self.calls
        self.calls += 1
        return self.values[idx] if idx < len(self.values) else self.default


def _recompute_final_hash(block) -> str:
    """Recompute the primary active plan hash from the manifest's own fields."""
    candidate = zs.Candidate(**block["chosen_candidate"])
    ident = block["identity"]
    identity = zs.FrozenIdentity(
        layout=tuple(ident["layout"]),
        cell_ids=tuple(ident["cell_ids"]),
        frame_ids=tuple(ident["frame_ids"]),
        halo_px=int(ident["halo_px"]),
        refinement_rank=int(ident["refinement_rank"]),
    )
    return zs.active_final_plan_hash(
        candidate=candidate,
        mapping=block["mapping"],
        identity=identity,
        envelope=block["final_envelope"],
    )


def _run_product(inp, out, mode, *, memory=WITNESS_AVAILABLE_BYTES):
    if not callable(memory):
        _constant = int(memory)
        memory = lambda: _constant
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(zz, "available_memory_bytes", memory)
        zz.run_zegrid_mode(
            str(inp),
            str(out),
            zegrid_halo_px=WITNESS_HALO_PX,
            zegrid_scheduler_mode=mode,
        )
    finally:
        patch.undo()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(out: Path) -> dict:
    return json.loads((out / "zegrid_manifest.json").read_text())


@pytest.fixture(scope="module")
def routine_input(tmp_path_factory):
    root = tmp_path_factory.mktemp("r55d2_routine")
    corpus = root / "corpus"
    proc = subprocess.run(
        [sys.executable, str(TOOL), str(corpus)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    inp = root / "input"
    inp.mkdir()
    for path in sorted(corpus.glob("*.fits")):
        shutil.copy2(path, inp / path.name)
    plan_csv = corpus / "stack_plan.csv"
    if plan_csv.exists():
        shutil.copy2(plan_csv, inp / "stack_plan.csv")
    return inp


@pytest.fixture(scope="module")
def parity_runs(routine_input, tmp_path_factory):
    root = tmp_path_factory.mktemp("r55d2_parity")
    legacy = root / "legacy"
    active = root / "active"
    active2 = root / "active2"
    _run_product(routine_input, legacy, "legacy")
    _run_product(routine_input, active, "active")
    _run_product(routine_input, active2, "active")
    return {"legacy": legacy, "active": active, "active2": active2}


def _spy_gauge_and_stack(monkeypatch):
    calls = {"gauge": 0, "stack": 0}

    def _gauge(*_a, **_k):
        calls["gauge"] += 1
        raise AssertionError("global gauge must not run for an active refusal")

    def _stack(*_a, **_k):
        calls["stack"] += 1
        raise AssertionError("cell stack must not run for an active refusal")

    monkeypatch.setattr(zz.zphot, "compute_global_gauge", _gauge)
    monkeypatch.setattr(zz, "_stack_cell", _stack)
    return calls


def _spy_gauge_call_through_and_stack(monkeypatch):
    """Let the gauge run and count it; forbid any stack worker."""
    calls = {"gauge": 0, "stack": 0}
    real_gauge = zz.zphot.compute_global_gauge

    def _gauge(*args, **kwargs):
        calls["gauge"] += 1
        return real_gauge(*args, **kwargs)

    def _stack(*_a, **_k):
        calls["stack"] += 1
        raise AssertionError("cell stack must not run for a launch refusal")

    monkeypatch.setattr(zz.zphot, "compute_global_gauge", _gauge)
    monkeypatch.setattr(zz, "_stack_cell", _stack)
    return calls


def _no_science_output(out: Path) -> None:
    assert not (out / "mosaic_grid.fits").exists()
    assert not (out / "mosaic_grid_coverage.fits").exists()
    assert not (out / "zegrid_manifest.json").exists()


# ---------------------------------------------------------------------------
# Mode contract
# ---------------------------------------------------------------------------

def test_active_mode_is_additive_and_default_remains_legacy():
    assert zs.DEFAULT_MODE == "legacy"
    assert zs.MODE_ACTIVE == "active"
    assert "active" in zs.MODES
    assert zs.normalize_scheduler_mode(None) == "legacy"
    assert zs.normalize_scheduler_mode("legacy") == "legacy"
    assert zs.normalize_scheduler_mode("observe") == "observe"
    assert zs.normalize_scheduler_mode("active") == "active"
    for bad in ("", "auto", "ACTIVE", " active", True, 0, 1.0, []):
        with pytest.raises(zs.InvalidSchedulerMode):
            zs.normalize_scheduler_mode(bad)
    # Default/config are untouched.
    from zemosaic import zemosaic_config

    assert zemosaic_config.DEFAULT_CONFIG["zegrid_scheduler_mode"] == "legacy"
    assert inspect.signature(zz.run_zegrid_mode).parameters[
        "zegrid_scheduler_mode"
    ].default is None


def test_unknown_active_mode_fails_before_any_work(tmp_path):
    with pytest.raises(zs.InvalidSchedulerMode):
        zz.run_zegrid_mode(
            str(tmp_path / "missing"),
            str(tmp_path / "out"),
            zegrid_scheduler_mode="ACTIVE",
        )
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# Scheduler mapping / refusal units
# ---------------------------------------------------------------------------

def _context(*, phase=re.PHASE_CELL_STACK_STREAM, per_unit_stack=200 * MIB):
    return zs.PlanContext(
        phase=phase,
        backend=re.BACKEND_CPU,
        budget_bytes=(4 * 8 * GIB) // 5,
        raw_available_bytes=8 * GIB,
        factor_num=4,
        factor_den=5,
        cpu_logical=8,
        units_total=len(CELLS),
        shared_bytes=340 * MIB,
        per_unit_stack_bytes=per_unit_stack,
        per_unit_cache_bytes=50 * MIB,
        per_unit_prefetch_bytes=50 * MIB,
        disk_free_bytes=100 * GIB,
        layout=(3, 2),
        refinement_rank=6,
        patch_hw=(200, 200),
        halo_px=32,
        channels=3,
        n_frames=50,
        prefetch_ladder=(0,),
        cell_ids=CELLS,
        frame_ids=FRAMES,
    )


def _requests():
    return (
        zs.PhaseRequest(
            phase=re.PHASE_GAUGE,
            components={"gauge_worker_footprint_bytes": 100 * MIB},
            units=1,
            units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=re.PHASE_CACHE_BUILD,
            components={"cache_build_worker_footprint_bytes": 100 * MIB},
            units=1,
            units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=re.PHASE_ASSEMBLY,
            components={"halo_band_bytes": 4 * MIB, "retained_core_bytes": 50 * MIB},
            units=1,
            units_in_flight=1,
            overlaps_with=(),
        ),
        zs.PhaseRequest(
            phase=re.PHASE_EXPORT,
            components={"coverage_plane_bytes": 10 * MIB, "science_planes_bytes": 30 * MIB},
            units=1,
            units_in_flight=1,
        ),
    )


def _joint_plan(*, inmem, stream):
    joint = zs.JointPlanInput(contexts=(inmem, stream))
    return zs.plan_joint(joint, phase_requests=_requests(), mode=zs.MODE_ACTIVE)


def _stream_wins_plan():
    return _joint_plan(
        inmem=_context(phase=re.PHASE_CELL_STACK_INMEM, per_unit_stack=500 * MIB),
        stream=_context(phase=re.PHASE_CELL_STACK_STREAM, per_unit_stack=80 * MIB),
    )


def _inmem_wins_plan():
    return _joint_plan(
        inmem=_context(phase=re.PHASE_CELL_STACK_INMEM, per_unit_stack=40 * MIB),
        stream=_context(phase=re.PHASE_CELL_STACK_STREAM, per_unit_stack=300 * MIB),
    )


def test_stream_mapping_is_exact_and_prefetch_is_zero():
    plan = _stream_wins_plan()
    mapping = zs.map_active_cpu_candidate(plan)
    candidate = plan.chosen
    assert mapping["schema"] == zs.ACTIVE_MAPPING_SCHEMA
    assert mapping["backend"] == "cpu"
    assert mapping["phase"] == re.PHASE_CELL_STACK_STREAM
    assert mapping["cell_task_mode"] == "stream"
    assert mapping["cpu_streaming_tile"] in zs.PRODUCT_TILE_LADDER
    assert mapping["cpu_streaming_tile"] == candidate.tile_px
    assert mapping["units_in_flight"] == candidate.units_in_flight
    assert mapping["workers_per_unit"] == candidate.workers_per_unit
    assert mapping["workers_per_unit"] in zs.WORKERS_LADDER
    assert mapping["prefetch_units"] == 0
    assert candidate.prefetch_units == 0


def test_inmem_mapping_is_none_tile_and_inmem_task_mode():
    plan = _inmem_wins_plan()
    mapping = zs.map_active_cpu_candidate(plan)
    assert plan.chosen.tile_px is None
    assert mapping["phase"] == re.PHASE_CELL_STACK_INMEM
    assert mapping["cell_task_mode"] == "inmem"
    assert mapping["cpu_streaming_tile"] is None
    assert mapping["prefetch_units"] == 0


def _candidate(**overrides):
    data = dict(
        backend=re.BACKEND_CPU,
        tile_px=128,
        units_in_flight=2,
        prefetch_units=0,
        workers_per_unit=4,
        host_peak_bytes=1,
        disk_bytes=1,
        device_peak_bytes=0,
        feasible=True,
        reason_code=re.REASON_OK,
    )
    data.update(overrides)
    return zs.Candidate(**data)


def test_active_mapping_rejects_gpu_prefetch_off_ladder_and_bad_workers():
    with pytest.raises(zs.ActivePlanRejected) as exc:
        zs.active_mapping_for_candidate(_candidate(backend=re.BACKEND_GPU))
    assert exc.value.reason_code == re.REASON_UNKNOWN_BACKEND

    with pytest.raises(zs.ActivePlanRejected) as exc:
        zs.active_mapping_for_candidate(_candidate(prefetch_units=1))
    assert exc.value.reason_code == re.REASON_SHRINK_PREFETCH

    with pytest.raises(zs.ActivePlanRejected) as exc:
        zs.active_mapping_for_candidate(_candidate(tile_px=16))
    assert exc.value.reason_code == re.REASON_SHRINK_TILE

    with pytest.raises(zs.ActivePlanRejected) as exc:
        zs.active_mapping_for_candidate(_candidate(workers_per_unit=5))
    assert exc.value.reason_code == re.REASON_SHRINK_WORKERS

    with pytest.raises(zs.ActivePlanRejected):
        zs.active_mapping_for_candidate(_candidate(units_in_flight=0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_ladder": (128.5,)},
        {"tile_ladder": (True, 128)},
        {"tile_ladder": ()},
        {"tile_ladder": (128, 128)},
        {"workers_ladder": (4.5,)},
        {"workers_ladder": (True, 4)},
        {"workers_ladder": ()},
        {"workers_ladder": (4, 4)},
    ],
)
def test_active_mapping_rejects_malformed_custom_ladders(kwargs):
    with pytest.raises(zs.ActivePlanRejected):
        zs.active_mapping_for_candidate(
            _candidate(tile_px=128, workers_per_unit=4), **kwargs
        )


def test_stale_or_tampered_candidate_is_refused_by_the_joint_builder():
    plan = _stream_wins_plan()
    tampered = dataclasses.replace(
        plan.chosen,
        units_in_flight=int(plan.chosen.units_in_flight) + 1,
    )
    joint = zs.JointPlanInput(
        contexts=(
            _context(phase=re.PHASE_CELL_STACK_INMEM, per_unit_stack=500 * MIB),
            _context(phase=re.PHASE_CELL_STACK_STREAM, per_unit_stack=80 * MIB),
        )
    )
    with pytest.raises(zs.SchedulerConfigError):
        zs.build_joint_plan_set(
            joint,
            candidate=tampered,
            phase_requests=_requests(),
            mode=zs.MODE_ACTIVE,
        )


def test_mapping_requires_an_active_plan_with_a_hash():
    plan = _stream_wins_plan()
    observe_plan = dataclasses.replace(plan, mode=zs.MODE_OBSERVE)
    with pytest.raises(zs.ActivePlanRejected):
        zs.map_active_cpu_candidate(observe_plan)
    no_hash = dataclasses.replace(plan, plan_hash="")
    with pytest.raises(zs.ActivePlanRejected) as exc:
        zs.map_active_cpu_candidate(no_hash)
    assert exc.value.reason_code == re.REASON_INTERNAL_ERROR


# ---------------------------------------------------------------------------
# Shrink / growth / identity units
# ---------------------------------------------------------------------------

def test_shrink_is_only_and_growth_is_rejected():
    context = _context()
    candidate = _candidate(workers_per_unit=4, units_in_flight=2)
    name, nxt = zs.next_shrink(candidate, context)
    assert name == "workers_per_unit"
    zs.assert_shrink_only(candidate, nxt)
    with pytest.raises(zs.GrowthRejected):
        zs.assert_shrink_only(nxt, candidate)
    with pytest.raises(zs.GrowthRejected):
        zs.assert_shrink_only(
            _candidate(tile_px=None), _candidate(tile_px=128)
        )
    with pytest.raises(zs.GrowthRejected):
        zs.assert_shrink_only(_candidate(), _candidate(backend=re.BACKEND_GPU))


def test_identity_change_is_rejected():
    base = zs.FrozenIdentity(
        layout=(3, 2), cell_ids=CELLS, frame_ids=FRAMES, halo_px=32,
        refinement_rank=6,
    )
    zs.assert_same_identity(base, base)
    for changed in (
        dataclasses.replace(base, layout=(2, 3)),
        dataclasses.replace(base, cell_ids=CELLS[:-1]),
        dataclasses.replace(base, frame_ids=FRAMES[:-1]),
        dataclasses.replace(base, halo_px=16),
        dataclasses.replace(base, refinement_rank=5),
    ):
        with pytest.raises(zs.GrowthRejected):
            zs.assert_same_identity(base, changed)


# ---------------------------------------------------------------------------
# Candidate materialization (rework-2)
# ---------------------------------------------------------------------------

def _enumerated(ctx, *, tile, k, workers=1):
    return next(
        c for c in zs.enumerate_candidates(ctx)
        if c.tile_px == tile
        and c.units_in_flight == k
        and c.prefetch_units == 0
        and c.workers_per_unit == workers
    )


def test_materialized_k_shrink_recomputes_canonical_projections():
    ctx = _context()
    orig = _enumerated(ctx, tile=128, k=3)
    name, stale = zs.next_shrink(orig, ctx)
    assert name == "units_in_flight"
    # next_shrink preserves the ORIGINAL projections (the defect D2 must not
    # serialize as the final candidate).
    assert stale.host_peak_bytes == orig.host_peak_bytes
    assert stale.disk_bytes == orig.disk_bytes

    canonical_k2 = _enumerated(ctx, tile=128, k=2)
    materialized = zs.materialize_active_candidate(ctx, stale)
    assert materialized == canonical_k2
    assert materialized.feasible is True
    assert materialized.host_peak_bytes == (
        ctx.shared_bytes + 2 * ctx.per_unit_stack_bytes
    )
    assert materialized.host_peak_bytes < stale.host_peak_bytes
    assert materialized.disk_bytes < stale.disk_bytes
    assert materialized.host_peak_bytes != stale.host_peak_bytes
    assert materialized.disk_bytes != stale.disk_bytes


def test_materialized_shrink_rebinds_and_enforces_fresh_feasibility():
    ctx = _context()
    k2 = _enumerated(ctx, tile=128, k=2)
    k1 = _enumerated(ctx, tile=128, k=1)
    tight_raw = 1 * GIB
    tight_budget = (4 * tight_raw) // 5

    infeasible = zs.materialize_active_candidate(
        ctx,
        k2,
        raw_available_bytes=tight_raw,
        factor_num=4,
        factor_den=5,
        budget_bytes=tight_budget,
    )
    assert infeasible.feasible is False
    assert infeasible.reason_code == re.REASON_REFUSE_HOST_RAM

    feasible = zs.materialize_active_candidate(
        ctx,
        k1,
        raw_available_bytes=tight_raw,
        factor_num=4,
        factor_den=5,
        budget_bytes=tight_budget,
    )
    assert feasible.feasible is True
    assert feasible.host_peak_bytes == (
        ctx.shared_bytes + ctx.per_unit_stack_bytes
    )
    # A mismatched rebound budget is refused, not guessed.
    with pytest.raises(zs.SchedulerConfigError):
        zs.materialize_active_candidate(
            ctx,
            k1,
            raw_available_bytes=tight_raw,
            factor_num=4,
            factor_den=5,
            budget_bytes=tight_budget + 1,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "raw_available_bytes": 1000.9,
            "factor_num": 4.2,
            "factor_den": 5.7,
            "budget_bytes": 800.1,
        },
        {
            "raw_available_bytes": True,
            "factor_num": True,
            "factor_den": True,
            "budget_bytes": True,
        },
        {
            "raw_available_bytes": 1000,
            "factor_num": 4,
            "factor_den": 5,
            "budget_bytes": 800.0,
        },
        {
            "raw_available_bytes": -1,
            "factor_num": 4,
            "factor_den": 5,
            "budget_bytes": 800,
        },
        {
            "raw_available_bytes": 1000,
            "factor_num": 0,
            "factor_den": 5,
            "budget_bytes": 800,
        },
        {
            "raw_available_bytes": 1000,
            "factor_num": 4,
            "factor_den": 5,
            "budget_bytes": -1,
        },
        {
            "raw_available_bytes": 1000,
            "factor_num": 4,
            "factor_den": 5,
            "budget_bytes": 801,
        },
    ],
)
def test_materialize_rejects_non_integer_or_mismatched_rebound_values(kwargs):
    ctx = _context()
    candidate = _enumerated(ctx, tile=128, k=2)
    with pytest.raises((zs.ActivePlanRejected, zs.SchedulerConfigError)):
        zs.materialize_active_candidate(ctx, candidate, **kwargs)


def test_materialize_accepts_consistent_integer_rebound_values():
    ctx = _context()
    candidate = _enumerated(ctx, tile=128, k=2)
    raw = 2 * GIB
    budget = (4 * raw) // 5
    out = zs.materialize_active_candidate(
        ctx,
        candidate,
        raw_available_bytes=raw,
        factor_num=4,
        factor_den=5,
        budget_bytes=budget,
    )
    assert isinstance(out, zs.Candidate)
    assert out.feasible is True


@pytest.mark.parametrize(
    "change",
    [
        {"tile_px": True},
        {"tile_px": 128.0},
        {"tile_px": -32},
        {"units_in_flight": True},
        {"units_in_flight": 1.5},
        {"units_in_flight": 0},
        {"prefetch_units": True},
        {"prefetch_units": 1.0},
        {"prefetch_units": -1},
        {"prefetch_units": 1},
        {"workers_per_unit": True},
        {"workers_per_unit": 4.0},
        {"workers_per_unit": 0},
        {"host_peak_bytes": True},
        {"host_peak_bytes": 1.5},
        {"host_peak_bytes": -1},
        {"disk_bytes": True},
        {"disk_bytes": 1.5},
        {"disk_bytes": -1},
        {"device_peak_bytes": True},
        {"device_peak_bytes": -1},
        {"device_peak_bytes": 1},
        {"feasible": 0},
        {"feasible": False},
        {"backend": re.BACKEND_GPU},
        {"reason_code": "BOGUS"},
        {"predicted_makespan_ns": True},
        {"predicted_makespan_ns": -1},
        {"predicted_makespan_ns": 1.5},
    ],
)
def test_active_mapping_and_hash_reject_malformed_or_infeasible_candidates(change):
    candidate = _enumerated(_context(), tile=128, k=2)
    broken = dataclasses.replace(candidate, **change)
    with pytest.raises((zs.ActivePlanRejected, zs.SchedulerConfigError)):
        zs.active_mapping_for_candidate(broken)
    mapping = zs.active_mapping_for_candidate(candidate)
    identity = zs.FrozenIdentity(
        layout=(3, 2),
        cell_ids=CELLS,
        frame_ids=FRAMES,
        halo_px=32,
        refinement_rank=6,
    )
    with pytest.raises((zs.ActivePlanRejected, zs.SchedulerConfigError)):
        zs.active_final_plan_hash(
            candidate=broken,
            mapping=mapping,
            identity=identity,
            envelope={"schema": "test-envelope"},
        )


def test_skipped_infeasible_shrink_trail_is_complete_and_adjacent():
    ctx = _context()
    k3 = _enumerated(ctx, tile=128, k=3)
    decision = types.SimpleNamespace(
        stream_context=ctx, inmem_context=ctx
    )
    # Tight fresh 4/5 window: K=2 is infeasible, K=1 is feasible.
    plan_obs = rcontract.Observation.plan(1 * GIB)
    name, accepted, examined = zz._r55_next_materialized_shrink(
        decision=decision, candidate=k3, plan_obs=plan_obs
    )
    assert name == "units_in_flight"
    assert accepted.units_in_flight == 1
    assert accepted.feasible is True
    assert len(examined) == 2
    first, second = examined
    assert first["step"] == second["step"] == "units_in_flight"
    assert first["accepted"] is False
    assert first["to"]["units_in_flight"] == 2
    assert first["reason_code"] == re.REASON_REFUSE_HOST_RAM
    assert first["detail"]
    assert second["accepted"] is True
    assert second["to"]["units_in_flight"] == 1
    # Adjacent, complete trail: no compressed K=3 -> K=1 jump.
    assert first["from"]["units_in_flight"] == 3
    assert second["from"] == first["to"]


# ---------------------------------------------------------------------------
# Active manifest contract (unit)
# ---------------------------------------------------------------------------

def test_active_manifest_is_deterministic_and_fabricates_nothing():
    plan = _stream_wins_plan()
    mapping = zs.map_active_cpu_candidate(plan)
    block = zs.active_manifest_dict(
        plan,
        mapping=mapping,
        status="executed",
        execution={"units_in_flight": mapping["units_in_flight"]},
        plan_observation={"raw_available_bytes": 8 * GIB},
        admission={"replans": 0},
        note="unit",
    )
    assert block["schema"] == zs.ACTIVE_MANIFEST_SCHEMA
    assert block["mode"] == "active"
    assert block["requested_mode"] == "active"
    assert block["status"] == "executed"
    assert block["plan_hash"] == plan.plan_hash
    assert block["initial_plan_hash"] == plan.plan_hash
    assert block["initial_candidate"] == plan.chosen.to_dict()
    assert block["chosen_candidate"] == plan.chosen.to_dict()
    assert block["final_envelope"] is None
    assert block["plan_hash_schema"].startswith("scheduler_plan_set:")
    assert block["mapping"] == mapping
    assert block["observed_metrics"] is None
    assert block["evidence_hash"].startswith("sha256:")
    assert zs.active_manifest_dict(
        plan,
        mapping=mapping,
        status="executed",
        execution={"units_in_flight": mapping["units_in_flight"]},
        plan_observation={"raw_available_bytes": 8 * GIB},
        admission={"replans": 0},
        note="unit",
    ) == block
    text = json.dumps(block, sort_keys=True)
    for forbidden in (
        "observed_rss", "observed_vram", "observed_peak",
        "throughput_units_per_s",
    ):
        assert forbidden not in text


# ---------------------------------------------------------------------------
# REAL product-path parity + deterministic two-build evidence
# ---------------------------------------------------------------------------

def test_active_vs_legacy_byte_identical_with_truthful_active_evidence(parity_runs):
    legacy = parity_runs["legacy"]
    active = parity_runs["active"]

    for name in ("mosaic_grid.fits", "mosaic_grid_coverage.fits"):
        assert _digest(legacy / name) == _digest(active / name), name

    legacy_manifest = _manifest(legacy)
    active_manifest = _manifest(active)
    assert "r55_scheduler" not in legacy_manifest
    block = active_manifest["r55_scheduler"]
    assert block["schema"] == zs.ACTIVE_MANIFEST_SCHEMA
    assert block["mode"] == "active"
    assert block["requested_mode"] == "active"
    assert block["status"] == "executed"
    assert block["observed_metrics"] is None
    assert block["plan_hash"].startswith("sha256:")
    assert block["evidence_hash"].startswith("sha256:")
    assert block["plan_hash_schema"] == zs.ACTIVE_FINAL_PLAN_SCHEMA
    # Distinct concepts by design: the D1 scheduler PlanSet hash vs the final
    # executable-plan hash. No pressure means the same candidate ran.
    assert block["initial_plan_hash"].startswith("sha256:")
    assert block["initial_plan_hash"] != block["plan_hash"]
    assert block["chosen_candidate"] == block["initial_candidate"]
    assert block["final_envelope"] is not None
    assert _recompute_final_hash(block) == block["plan_hash"]
    assert block["preflight_admission"]["replans"] == 0
    assert block["launch_revalidation"]["replans"] == 0
    assert block["admission"] == block["launch_revalidation"]
    assert (
        block["initial_plan_observation"]["raw_available_bytes"]
        == WITNESS_AVAILABLE_BYTES
    )
    assert (
        block["plan_observation"]["raw_available_bytes"]
        == WITNESS_AVAILABLE_BYTES
    )

    mapping = block["mapping"]
    execution = block["execution"]
    diag = active_manifest["per_cell_diagnostics"]
    assert mapping["backend"] == "cpu"
    assert mapping["prefetch_units"] == 0
    assert mapping["cell_task_mode"] == execution["cell_task_mode"] == diag["mode"]
    assert mapping["units_in_flight"] == execution["units_in_flight"] == diag["cells_in_flight"]
    assert mapping["workers_per_unit"] == execution["workers_per_unit"] == diag["cache_workers_per_cell"]
    assert (
        mapping["cpu_streaming_tile"]
        if mapping["cpu_streaming_tile"] is not None
        else 0
    ) == execution["cpu_streaming_tile"] == diag["cpu_streaming_tile"]
    assert block["chosen_candidate"]["tile_px"] == mapping["cpu_streaming_tile"]
    assert block["chosen_candidate"]["units_in_flight"] == mapping["units_in_flight"]
    assert block["chosen_candidate"]["workers_per_unit"] == mapping["workers_per_unit"]
    assert block["chosen_candidate"]["prefetch_units"] == 0

    plan_obs = block["plan_observation"]
    assert plan_obs["factor_num"] == 4 and plan_obs["factor_den"] == 5
    assert plan_obs["budget_bytes"] == (4 * plan_obs["raw_available_bytes"]) // 5
    admission = block["admission"]
    assert admission["observations"]
    admit_obs = admission["observations"][0]
    assert admit_obs["phase"] == "admit"
    assert admit_obs["factor_num"] == 9 and admit_obs["factor_den"] == 10
    assert admit_obs["budget_bytes"] == (9 * admit_obs["raw_available_bytes"]) // 10
    assert admission["replans"] == 0
    assert block["shrink_trail"] == []

    # Frozen identity: same layout/cells/frames/halo as the legacy run.
    legacy_layout = legacy_manifest["layout"]
    assert block["identity"]["layout"] == [
        int(legacy_layout["nx"]), int(legacy_layout["ny"])
    ]
    assert block["identity"]["halo_px"] == WITNESS_HALO_PX
    assert len(block["identity"]["cell_ids"]) > 1
    assert len(block["identity"]["frame_ids"]) == 12
    assert block["identity"]["refinement_rank"] == (
        int(block["identity"]["layout"][0]) * int(block["identity"]["layout"][1])
    )

    # The SCIENCE fields are unchanged by active mode. Execution provenance
    # (per_cell_diagnostics, ram_contract, cell modes) legitimately differs
    # because active maps different knobs; the additive r55_scheduler block is
    # the only scheduler authority.
    assert active_manifest["layout"] == legacy_manifest["layout"]
    assert active_manifest["complete_cells"] == legacy_manifest["complete_cells"]
    assert active_manifest["incomplete_cells"] == legacy_manifest["incomplete_cells"]
    assert active_manifest["coverage_pixels"] == legacy_manifest["coverage_pixels"]

    # Default config was not mutated by the active run.
    from zemosaic import zemosaic_config

    assert zemosaic_config.DEFAULT_CONFIG["zegrid_scheduler_mode"] == "legacy"
    assert zs.DEFAULT_MODE == "legacy"


def test_active_evidence_is_deterministic_across_two_builds(parity_runs):
    first = _manifest(parity_runs["active"])["r55_scheduler"]
    second = _manifest(parity_runs["active2"])["r55_scheduler"]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["evidence_hash"] == second["evidence_hash"]
    assert first["plan_hash"] == second["plan_hash"]
    # The second run also produced byte-identical science.
    for name in ("mosaic_grid.fits", "mosaic_grid_coverage.fits"):
        assert _digest(parity_runs["active"] / name) == _digest(
            parity_runs["active2"] / name
        ), name


def test_active_to_legacy_rollback_restores_the_baseline_path(parity_runs):
    """Changing only active -> legacy restores the pre-R55 manifest shape."""
    legacy = _manifest(parity_runs["legacy"])
    assert "r55_scheduler" not in legacy
    assert _digest(parity_runs["legacy"] / "mosaic_grid.fits") == _digest(
        parity_runs["active"] / "mosaic_grid.fits"
    )


# ---------------------------------------------------------------------------
# Refusal timing: ZERO gauge / ZERO stack calls
# ---------------------------------------------------------------------------

def test_gpu_product_execution_refuses_before_gauge_and_stack(
    routine_input, tmp_path, monkeypatch
):
    calls = _spy_gauge_and_stack(monkeypatch)
    real_build = zz._build_execution_bundle

    def _gpu_build(*args, **kwargs):
        bundle = real_build(*args, **kwargs)
        return dataclasses.replace(bundle, gpu_used=True)

    monkeypatch.setattr(zz, "_build_execution_bundle", _gpu_build)
    out = tmp_path / "gpu_refused"
    with pytest.raises(RuntimeError, match="GPU"):
        _run_product(routine_input, out, "active")
    assert calls == {"gauge": 0, "stack": 0}
    _no_science_output(out)


def test_plan_refusal_is_fatal_before_gauge_and_stack(
    routine_input, tmp_path, monkeypatch
):
    calls = _spy_gauge_and_stack(monkeypatch)

    def _refuse(**_kwargs):
        raise zs.PlanRefused(
            re.REASON_REFUSE_HOST_RAM, "synthetic active plan refusal"
        )

    monkeypatch.setattr(zz, "_r55_active_decide", _refuse)
    out = tmp_path / "plan_refused"
    with pytest.raises(RuntimeError, match="refused before the gauge/stack"):
        _run_product(routine_input, out, "active")
    assert calls == {"gauge": 0, "stack": 0}
    _no_science_output(out)


def test_exact_mapping_mismatch_is_fatal_before_gauge_and_stack(
    routine_input, tmp_path, monkeypatch
):
    calls = _spy_gauge_and_stack(monkeypatch)
    real_decide = zz._r55_active_decide

    def _tampered_decide(**kwargs):
        decision = real_decide(**kwargs)
        bad = dict(decision.mapping)
        bad["units_in_flight"] = int(bad["units_in_flight"]) + 1
        return dataclasses.replace(decision, mapping=bad)

    monkeypatch.setattr(zz, "_r55_active_decide", _tampered_decide)
    out = tmp_path / "mapping_mismatch"
    with pytest.raises(RuntimeError, match="refused before the gauge/stack"):
        _run_product(routine_input, out, "active")
    assert calls == {"gauge": 0, "stack": 0}
    _no_science_output(out)


def test_admission_exhaustion_is_fatal_before_gauge_and_stack(
    routine_input, tmp_path, monkeypatch
):
    calls = _spy_gauge_and_stack(monkeypatch)
    # PLAN succeeds at 12 GiB; every fresh ADMIT observation is unusable-small.
    memory = _SeqMemory(WITNESS_AVAILABLE_BYTES, 1)
    out = tmp_path / "admit_exhausted"
    with pytest.raises(RuntimeError, match="refused before the gauge/stack"):
        _run_product(routine_input, out, "active", memory=memory)
    assert memory.calls >= 2
    assert calls == {"gauge": 0, "stack": 0}
    _no_science_output(out)


# ---------------------------------------------------------------------------
# Pressure shrink witness (scheduler-produced, shrink-only, frozen identity)
# ---------------------------------------------------------------------------

def test_fresh_pressure_shrinks_only_the_scheduler_plan(
    routine_input, tmp_path, monkeypatch
):
    # Real product path, synthetic PRESSURE MODEL. The routine corpus's stream
    # bound dominates its tiny aligned cache, so a fresh-RAM miss could only
    # shrink workers by inflating the declared cache-worker footprint. That
    # makes the CACHE phase the mapped envelope's dominant term without touching
    # science, layout, geometry or the executor. PLAN stays at 12 GiB; fresh
    # ADMIT is 2.9 GiB, so the initial W=8 mapped plan misses and the scheduler's
    # fixed workers-first shrink must admit W=6 on the SAME frozen identity.
    real_cache_fp = zz.zpar.cache_footprint_bytes

    def _inflated_cache_fp(px):
        return max(int(real_cache_fp(px)), 600 * MIB)

    monkeypatch.setattr(zz.zpar, "cache_footprint_bytes", _inflated_cache_fp)
    # Call 1 = PLAN at 12 GiB; calls 2-3 = PREFLIGHT admit+launch at 2.9 GiB
    # (preflight shrink); calls 4+ = POST-GAUGE launch at 12 GiB (no further
    # correction). This isolates the preflight pressure path.
    memory = _ScheduleMemory(
        [WITNESS_AVAILABLE_BYTES, 2900 * MIB, 2900 * MIB, WITNESS_AVAILABLE_BYTES],
        default=WITNESS_AVAILABLE_BYTES,
    )
    out = tmp_path / "shrunk"
    _run_product(routine_input, out, "active", memory=memory)
    block = _manifest(out)["r55_scheduler"]
    assert block["status"] == "executed"
    assert block["preflight_admission"]["replans"] >= 1
    assert block["launch_revalidation"]["replans"] == 0
    assert block["shrink_trail"], "a fresh admission miss must record shrink steps"
    steps = block["shrink_trail"]
    assert all(item["at"] == "admission_pressure" for item in steps)
    assert all(item["stage"] == "preflight" for item in steps)
    # The trail is shrink-only in the fixed order.
    order = ["prefetch_units", "workers_per_unit", "units_in_flight", "tile_px", "backend"]
    positions = [order.index(item["step"]) for item in steps]
    assert positions == sorted(positions)
    for item in steps:
        assert int(item["to"]["prefetch_units"]) <= int(item["from"]["prefetch_units"])
        assert int(item["to"]["units_in_flight"]) <= int(item["from"]["units_in_flight"])
        assert int(item["to"]["workers_per_unit"]) <= int(item["from"]["workers_per_unit"])
        if item["from"]["tile_px"] is None:
            assert item["to"]["tile_px"] is None
        elif item["to"]["tile_px"] is not None:
            assert int(item["to"]["tile_px"]) <= int(item["from"]["tile_px"])
    # Frozen identity is preserved by the pressure correction.
    initial = block["identity"]
    final = block["execution"]
    assert final["layout"] == initial["layout"]
    assert final["refinement_rank"] == initial["refinement_rank"]
    assert final["aggregate_bytes"] <= block["admission"]["launch_observation"]["budget_bytes"]
    # The final mapped values are exactly what the diagnostics observed.
    diag = _manifest(out)["per_cell_diagnostics"]
    assert final["cell_task_mode"] == diag["mode"]
    assert final["units_in_flight"] == diag["cells_in_flight"]
    assert final["workers_per_unit"] == diag["cache_workers_per_cell"]
    assert final["cpu_streaming_tile"] == diag["cpu_streaming_tile"]
    # rework-2: the exposed fallback diagnostic is rebound to the FINAL
    # envelope, not the stale base budget.
    fallback = diag["cpu_fallback_plan"]
    assert fallback["ram_budget_bytes"] == block["final_envelope"]["budget_bytes"]
    assert fallback["raw_available_bytes"] == block["final_envelope"]["raw_available_bytes"]
    assert fallback["plan_budget_bytes"] == block["final_envelope"]["budget_bytes"]
    # rework-1: the final candidate/hash are truthful, and the window that
    # admitted the correction is the fresh 4/5 observation (not the stale one).
    assert block["plan_hash_schema"] == zs.ACTIVE_FINAL_PLAN_SCHEMA
    assert block["plan_hash"] != block["initial_plan_hash"]
    assert (
        block["chosen_candidate"]["units_in_flight"]
        == block["mapping"]["units_in_flight"]
    )
    assert (
        block["chosen_candidate"]["workers_per_unit"]
        == block["mapping"]["workers_per_unit"]
    )
    assert block["chosen_candidate"]["workers_per_unit"] == final["workers_per_unit"]
    assert (
        block["chosen_candidate"]["workers_per_unit"]
        < block["initial_candidate"]["workers_per_unit"]
    )
    assert block["plan_observation"]["raw_available_bytes"] == 2900 * MIB
    assert block["initial_plan_observation"]["raw_available_bytes"] == WITNESS_AVAILABLE_BYTES
    assert block["final_envelope"]["raw_available_bytes"] == 2900 * MIB
    assert block["final_envelope"]["factor_num"] == 4
    assert block["final_envelope"]["factor_den"] == 5
    assert block["final_envelope"]["budget_bytes"] == (4 * (2900 * MIB)) // 5
    assert (
        block["final_envelope"]["aggregate_bytes"]
        <= block["plan_observation"]["budget_bytes"]
    )
    assert _recompute_final_hash(block) == block["plan_hash"]


# ---------------------------------------------------------------------------
# Post-gauge launch revalidation (rework-1)
# ---------------------------------------------------------------------------

def test_post_gauge_ram_collapse_refuses_before_stack(
    routine_input, tmp_path, monkeypatch
):
    """PLAN + preflight are high; RAM collapses AFTER the gauge.

    The gauge must run exactly once, the launch revalidation must refuse, the
    stack must never start, and no science/manifest may be committed.
    """
    calls = _spy_gauge_call_through_and_stack(monkeypatch)
    memory = _ScheduleMemory(
        [
            WITNESS_AVAILABLE_BYTES,  # PLAN
            WITNESS_AVAILABLE_BYTES,  # preflight admit
            WITNESS_AVAILABLE_BYTES,  # preflight launch
            1,                        # post-gauge admit (collapse)
        ],
        default=1,
    )
    out = tmp_path / "post_gauge_refused"
    with pytest.raises(
        RuntimeError, match="launch revalidation refused before the stack"
    ):
        _run_product(routine_input, out, "active", memory=memory)
    assert calls == {"gauge": 1, "stack": 0}
    _no_science_output(out)


def test_post_gauge_pressure_shrinks_the_actual_stack_knobs(
    routine_input, tmp_path, monkeypatch
):
    """PLAN + preflight are high; post-gauge pressure permits one shrink.

    The final stack must use the shrunk W, the final candidate/hash must be
    truthful, and the final envelope must be bound to the fresh 4/5 observation.
    """
    real_cache_fp = zz.zpar.cache_footprint_bytes

    def _inflated_cache_fp(px):
        return max(int(real_cache_fp(px)), 600 * MIB)

    monkeypatch.setattr(zz.zpar, "cache_footprint_bytes", _inflated_cache_fp)
    memory = _ScheduleMemory(
        [
            WITNESS_AVAILABLE_BYTES,  # PLAN
            WITNESS_AVAILABLE_BYTES,  # preflight admit
            WITNESS_AVAILABLE_BYTES,  # preflight launch
            2900 * MIB,               # post-gauge admit (miss -> shrink)
        ],
        default=2900 * MIB,
    )
    out = tmp_path / "post_gauge_shrunk"
    _run_product(routine_input, out, "active", memory=memory)
    manifest = _manifest(out)
    block = manifest["r55_scheduler"]
    diag = manifest["per_cell_diagnostics"]
    assert block["status"] == "executed"
    assert block["preflight_admission"]["replans"] == 0
    assert block["launch_revalidation"]["replans"] >= 1
    steps = block["shrink_trail"]
    assert steps and all(item["stage"] == "launch" for item in steps)
    assert all(item["at"] == "admission_pressure" for item in steps)
    # Final candidate, mapping, execution and diagnostics agree.
    final = block["execution"]
    assert block["chosen_candidate"]["workers_per_unit"] == final["workers_per_unit"]
    assert block["mapping"]["workers_per_unit"] == final["workers_per_unit"]
    assert diag["cache_workers_per_cell"] == final["workers_per_unit"]
    assert block["mapping"]["units_in_flight"] == final["units_in_flight"]
    assert diag["cells_in_flight"] == final["units_in_flight"]
    assert diag["mode"] == final["cell_task_mode"]
    fallback = diag["cpu_fallback_plan"]
    assert fallback["ram_budget_bytes"] == block["final_envelope"]["budget_bytes"]
    assert fallback["raw_available_bytes"] == block["final_envelope"]["raw_available_bytes"]
    assert fallback["plan_budget_bytes"] == block["final_envelope"]["budget_bytes"]
    # The correction is a real shrink and the final hash is not the initial one.
    assert (
        block["chosen_candidate"]["workers_per_unit"]
        < block["initial_candidate"]["workers_per_unit"]
    )
    assert block["plan_hash"] != block["initial_plan_hash"]
    assert block["plan_hash_schema"] == zs.ACTIVE_FINAL_PLAN_SCHEMA
    assert _recompute_final_hash(block) == block["plan_hash"]
    # The final envelope is bound to the POST-GAUGE 4/5 observation.
    assert block["plan_observation"]["raw_available_bytes"] == 2900 * MIB
    assert block["initial_plan_observation"]["raw_available_bytes"] == WITNESS_AVAILABLE_BYTES
    assert block["final_envelope"]["raw_available_bytes"] == 2900 * MIB
    assert block["final_envelope"]["factor_num"] == 4
    assert block["final_envelope"]["factor_den"] == 5
    assert block["final_envelope"]["budget_bytes"] == (4 * (2900 * MIB)) // 5
    assert (
        block["final_envelope"]["aggregate_bytes"]
        <= block["plan_observation"]["budget_bytes"]
    )
    # Frozen identity survives the post-gauge correction.
    assert final["layout"] == block["identity"]["layout"]
    assert final["refinement_rank"] == block["identity"]["refinement_rank"]


# ---------------------------------------------------------------------------
# Import laziness with CuPy installed
# ---------------------------------------------------------------------------

def test_scheduler_import_does_not_load_cupy_or_probe_cuda():
    code = (
        "import sys\n"
        "from zemosaic.core.zegrid import scheduler as zs\n"
        "assert zs.MODE_ACTIVE == 'active'\n"
        "print('cupy' in sys.modules)\n"
        "print('zemosaic.core.zegrid.gpu' in sys.modules)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env={**__import__("os").environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.splitlines()[-2:] == ["False", "False"]


# ---------------------------------------------------------------------------
# Generic GUI/backend kwargs propagation
# ---------------------------------------------------------------------------

class _RecordingQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_generic_active_mode_propagation_guard(monkeypatch, tmp_path):
    from zemosaic import zemosaic_worker as zw

    real_sig = inspect.signature(zw.run_hierarchical_mosaic)
    captured: dict = {}

    def stub(**final_kwargs):
        captured.update(final_kwargs)

    stub.__signature__ = real_sig  # type: ignore[attr-defined]
    monkeypatch.setattr(zw, "run_hierarchical_mosaic", stub)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    zw.run_hierarchical_mosaic_process(
        _RecordingQueue(),
        solver_settings_dict=None,
        zegrid_scheduler_mode="active",
        crash_breadcrumbs_mode="off",
        input_dir=str(tmp_path / "in"),
        output_dir=str(tmp_path / "out"),
    )
    assert captured.get("zegrid_scheduler_mode") == "active"
    assert not (set(captured) - set(real_sig.parameters))
    assert "zegrid_scheduler_mode" in inspect.signature(zz.run_zegrid_mode).parameters
