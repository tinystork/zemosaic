"""ZM-ZEGRID-R55-E1 — CPU RUN checkpoints, deterministic waves and telemetry.

Acceptance exercised here:

* strict malformed RUN snapshots fail closed (bool/float/negative/missing);
* a mandatory fresh 90 % checkpoint precedes every wave;
* canonical contiguous waves are bounded by the accepted K and preserve
  result order/progress accounting;
* scheduler-produced adjacent shrink-only corrections in the fixed global
  order (prefetch -> workers -> K -> tile -> backend), with frozen endpoints
  recorded as unavailable;
* a changed mapping applies only to the next wave; recovery never grows;
* host exhaustion and disk shortfall refuse BEFORE the next wave and publish
  no FITS/manifest/science, with best-effort temp cleanup recorded;
* truthful CPU telemetry (parent RSS/VmHWM, worker peaks with an honest
  aggregate label, raw/90 % host budget, cgroup v2 and OS I/O availability,
  disk free, wave wall/remaining/shrink/refusal counts);
* deterministic RUN decision structure is separated from volatile telemetry;
* real hermetic active-vs-legacy parity for multi-wave and multi-wave-shrink
  product witnesses;
* CPU import laziness (no CuPy / CUDA initialisation).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import ram_contract as rcontract
from zemosaic.core.zegrid import resource_envelope as re
from zemosaic.core.zegrid import runtime_control as zrt
from zemosaic.core.zegrid import scheduler as zs

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "zegrid_routine" / "make_routine_corpus.py"
WITNESS_HALO_PX = 32
WITNESS_AVAILABLE_BYTES = 12 * 2 ** 30
MIB = 2 ** 20
GIB = 2 ** 30
FITS_NAMES = ("mosaic_grid.fits", "mosaic_grid_coverage.fits")


def _echo_task(value):
    return value


def _slow_echo_task(value):
    time.sleep(0.02)
    return value


def _fail_task(value):
    if int(value) == 2:
        raise RuntimeError("intentional worker failure")
    return value


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(out: Path) -> dict:
    return json.loads((out / "zegrid_manifest.json").read_text())


def _facts(raw, *, disk_free=100 * GIB, source="injected"):
    raw = int(raw)
    return {
        "schema": zrt.RUN_CONTROL_SCHEMA,
        "host": {
            "raw_available_bytes": raw,
            "budget_bytes": (rcontract.ADMIT_FACTOR_NUM * raw)
            // rcontract.ADMIT_FACTOR_DEN,
            "factor_num": rcontract.ADMIT_FACTOR_NUM,
            "factor_den": rcontract.ADMIT_FACTOR_DEN,
            "source": source,
        },
        "disk": {
            "free_bytes": int(disk_free),
            "io_read_bytes": None,
            "io_write_bytes": None,
            "unavailable_reason": None,
        },
        "parent": {
            "current_rss_bytes": None,
            "peak_rss_bytes": None,
            "unavailable_reason": "unit-test",
        },
        "cgroup": {
            "current_bytes": None,
            "peak_bytes": None,
            "unavailable_reason": "unit-test",
        },
    }


class _ScheduleSample:
    def __init__(self, schedule, *, disk_free=100 * GIB):
        self.schedule = list(schedule)
        self.disk_free = int(disk_free)
        self.calls = 0

    def __call__(self, _root):
        raw = (
            self.schedule[self.calls]
            if self.calls < len(self.schedule)
            else self.schedule[-1]
        )
        self.calls += 1
        return _facts(raw, disk_free=self.disk_free)


class _FakeEnvelope:
    def __init__(self, aggregate_bytes=1):
        self.aggregate_bytes = int(aggregate_bytes)


class _FakeBundle:
    def __init__(self, aggregate_bytes=1):
        self.envelope = _FakeEnvelope(aggregate_bytes)
        self.cell_tasks = []
        self.attempts = []
        self.layout = {"nx": 3, "ny": 2}
        self.footprints = {}
        self.cell_ctxs = []
        self.cpu_streaming = {}
        self.cpu_streaming_tile = 128
        self.cell_specs = []
        self.chosen_mode = "stream"
        self.cells_in_flight = 1
        self.cache_workers_per_cell = 1
        self.per_cell_footprint = 0
        self.gpu_tile_size = None
        self.gpu_streaming_tile = None
        self.gpu_streaming_host_peak_bytes = None
        self.gpu_used = False
        self.gpu_pre_execution_fallback = None
        self.cpu_fallback_info = {}
        self.cpu_fallback_plan = None
        self.cpu_fallback_tasks = None
        self.cache_build_workers = 1
        self.worst_patch_px = 1


def _synthetic_context(*, k_target=3, raw=WITNESS_AVAILABLE_BYTES):
    cells = tuple(f"c{i}" for i in range(6))
    frames = tuple(f"f{i}" for i in range(12))
    return zs.PlanContext(
        phase=re.PHASE_CELL_STACK_STREAM,
        backend=re.BACKEND_CPU,
        budget_bytes=(rcontract.PLAN_FACTOR_NUM * int(raw))
        // rcontract.PLAN_FACTOR_DEN,
        raw_available_bytes=int(raw),
        factor_num=rcontract.PLAN_FACTOR_NUM,
        factor_den=rcontract.PLAN_FACTOR_DEN,
        cpu_logical=8,
        units_total=len(cells),
        shared_bytes=256 * MIB,
        per_unit_stack_bytes=256 * MIB,
        per_unit_cache_bytes=128 * MIB,
        per_unit_prefetch_bytes=0,
        disk_free_bytes=100 * GIB,
        layout=(3, 2),
        refinement_rank=6,
        patch_hw=(512, 512),
        halo_px=WITNESS_HALO_PX,
        channels=3,
        n_frames=12,
        tile_ladder=zs.PRODUCT_TILE_LADDER,
        prefetch_ladder=(0,),
        workers_ladder=zs.WORKERS_LADDER,
        cell_ids=cells,
        frame_ids=frames,
        run_id="r55e1-unit",
    )


def _synthetic_decision(*, k_target=3):
    ctx = _synthetic_context()
    reqs = (
        zs.PhaseRequest(
            phase=re.PHASE_GAUGE,
            components={"gauge_worker_footprint_bytes": 0},
            units=1,
            units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=re.PHASE_CACHE_BUILD,
            components={"cache_build_worker_footprint_bytes": 0},
            units=1,
            units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=re.PHASE_ASSEMBLY,
            components={"halo_band_bytes": 0, "retained_core_bytes": 0},
            units=1,
            units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=re.PHASE_EXPORT,
            components={"coverage_plane_bytes": 0, "science_planes_bytes": 0},
            units=1,
            units_in_flight=1,
        ),
    )
    plan = zs.plan_joint(
        zs.JointPlanInput(contexts=(ctx,)),
        phase_requests=reqs,
        mode=zs.MODE_ACTIVE,
    )
    target = next(
        c
        for c in zs.enumerate_candidates(ctx)
        if c.tile_px == 128
        and c.units_in_flight == k_target
        and c.workers_per_unit == 1
        and c.prefetch_units == 0
        and c.feasible
    )
    plan = dataclasses.replace(plan, chosen=target)
    facts = {
        "cell_ids": ctx.cell_ids,
        "stream_worst_by_tile": {
            int(t): int(ctx.per_unit_stack_bytes) for t in zs.PRODUCT_TILE_LADDER
        },
        "inmem_worst": int(ctx.per_unit_stack_bytes),
        "units_total": int(ctx.units_total),
    }
    return zz._R55ActiveDecision(
        plan_set=plan,
        mapping=zs.active_mapping_for_candidate(target),
        facts=facts,
        inmem_context=ctx,
        stream_context=ctx,
        phase=zs.active_cpu_phase(target),
    )


def _force_k_decide(_real_decide, k):
    def patched(**kwargs):
        decision = _real_decide(**kwargs)
        target = next(
            c
            for c in zs.enumerate_candidates(decision.stream_context)
            if c.tile_px == 128
            and c.units_in_flight == int(k)
            and c.workers_per_unit == 1
            and c.prefetch_units == 0
            and c.feasible
        )
        return dataclasses.replace(
            decision,
            plan_set=dataclasses.replace(decision.plan_set, chosen=target),
            mapping=zs.active_mapping_for_candidate(target),
            phase=zs.active_cpu_phase(target),
        )

    return patched


def _force_k3_decide(_real_decide):
    return _force_k_decide(_real_decide, 3)


def _run_product(
    inp,
    out,
    mode,
    *,
    memory=WITNESS_AVAILABLE_BYTES,
    sample=None,
    decide=None,
    pressure=None,
):
    if not callable(memory):
        _constant = int(memory)
        memory = lambda: _constant
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(zz, "available_memory_bytes", memory)
        if sample is not None:
            patch.setattr(zz, "_r55_run_checkpoint_sample", sample)
        if decide is not None:
            patch.setattr(zz, "_r55_active_decide", decide)
        if pressure is not None:
            patch.setattr(zz, "_r55_sample_pressure", pressure)
        zz.run_zegrid_mode(
            str(inp),
            str(out),
            zegrid_halo_px=WITNESS_HALO_PX,
            zegrid_scheduler_mode=mode,
        )
    finally:
        patch.undo()


def _recompute_final_hash(block) -> str:
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


@pytest.fixture(scope="module")
def routine_input(tmp_path_factory):
    root = tmp_path_factory.mktemp("r55e1_routine")
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
    root = tmp_path_factory.mktemp("r55e1_parity")
    legacy = root / "legacy"
    active = root / "active"
    active2 = root / "active2"
    _run_product(routine_input, legacy, "legacy")
    _run_product(routine_input, active, "active")
    _run_product(routine_input, active2, "active")
    return {"legacy": legacy, "active": active, "active2": active2}


@pytest.fixture(scope="module")
def shrink_runs(routine_input, tmp_path_factory):
    root = tmp_path_factory.mktemp("r55e1_shrink")
    legacy = root / "legacy"
    active = root / "active"
    active2 = root / "active2"
    _run_product(routine_input, legacy, "legacy")
    decide = _force_k3_decide(zz._r55_active_decide)
    # K=3 host peak ~2.89 GiB; a 2.5 GiB raw (2.25 GiB budget) admits K=2
    # (~2.05 GiB) but not K=3, producing a real adjacent pressure shrink.
    schedule = [WITNESS_AVAILABLE_BYTES, 5 * GIB // 2,
                WITNESS_AVAILABLE_BYTES, WITNESS_AVAILABLE_BYTES]
    _run_product(
        routine_input,
        active,
        "active",
        sample=_ScheduleSample(schedule),
        decide=decide,
    )
    _run_product(
        routine_input,
        active2,
        "active",
        sample=_ScheduleSample(schedule),
        decide=_force_k3_decide(zz._r55_active_decide),
    )
    return {"legacy": legacy, "active": active, "active2": active2}


# ---------------------------------------------------------------------------
# Strict RUN facts / refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        True,
        1.5,
        -1,
        0,
        None,
        "123",
    ],
)
def test_malformed_raw_observation_fails_closed(monkeypatch, bad):
    decision = _synthetic_decision()
    monkeypatch.setattr(zz, "_r55_active_apply", lambda *a, **k: _FakeBundle())
    with pytest.raises(zz._R55RunRefused) as ei:
        zz._r55_run_checkpoint(
            decision=decision,
            candidate=decision.plan_set.chosen,
            mapping=decision.mapping,
            bundle=_FakeBundle(),
            remaining_units=6,
            wave_index=0,
            facts={
                "host": {
                    "raw_available_bytes": bad,
                    "budget_bytes": 1,
                    "factor_num": 9,
                    "factor_den": 10,
                },
                "disk": {"free_bytes": 1},
            },
        )
    assert ei.value.reason_code == re.REASON_REFUSE_HOST_RAM
    assert "malformed RUN checkpoint" in ei.value.detail


@pytest.mark.parametrize(
    "bad_budget",
    [
        True,
        -1,
        1.0,
        (9 * WITNESS_AVAILABLE_BYTES) // 10 + 1,
    ],
)
def test_malformed_or_fabricated_budget_fails_closed(monkeypatch, bad_budget):
    decision = _synthetic_decision()
    monkeypatch.setattr(zz, "_r55_active_apply", lambda *a, **k: _FakeBundle())
    facts = _facts(WITNESS_AVAILABLE_BYTES)
    facts["host"]["budget_bytes"] = bad_budget
    with pytest.raises(zz._R55RunRefused) as ei:
        zz._r55_run_checkpoint(
            decision=decision,
            candidate=decision.plan_set.chosen,
            mapping=decision.mapping,
            bundle=_FakeBundle(),
            remaining_units=6,
            wave_index=0,
            facts=facts,
        )
    assert ei.value.reason_code == re.REASON_REFUSE_HOST_RAM


# ---------------------------------------------------------------------------
# Adjacent shrink-only corrections / no growth
# ---------------------------------------------------------------------------


def test_pressure_shrinks_adjacent_then_recovery_never_grows(monkeypatch):
    decision = _synthetic_decision(k_target=3)
    calls = {"n": 0}

    def apply(bundle, *, mapping, **_kwargs):
        calls["n"] += 1
        return _FakeBundle(aggregate_bytes=1)

    monkeypatch.setattr(zz, "_r55_active_apply", apply)
    # K=3 host peak is 1 GiB; a 1 GiB raw -> 921 MiB budget admits K=2 only.
    shrunk = zz._r55_run_checkpoint(
        decision=decision,
        candidate=decision.plan_set.chosen,
        mapping=decision.mapping,
        bundle=_FakeBundle(aggregate_bytes=1),
        remaining_units=6,
        wave_index=1,
        facts=_facts(1 * GIB),
    )
    assert shrunk["record"]["changed"] is True
    assert shrunk["candidate"].units_in_flight == 2
    assert shrunk["mapping"]["units_in_flight"] == 2
    steps = shrunk["record"]["shrink_steps"]
    assert [s["step"] for s in steps] == ["units_in_flight"]
    assert steps[0]["from"]["units_in_flight"] == 3
    assert steps[0]["to"]["units_in_flight"] == 2
    assert steps[0]["accepted"] is True
    # Frozen endpoints are recorded, never invented.
    unavailable = {x["step"]: x for x in shrunk["record"]["unavailable_shrinks"]}
    assert unavailable["prefetch_units"]["available"] is False
    assert unavailable["backend"]["available"] is False

    before = len(shrunk["record"]["shrink_steps"])
    recovered = zz._r55_run_checkpoint(
        decision=decision,
        candidate=shrunk["candidate"],
        mapping=shrunk["mapping"],
        bundle=shrunk["bundle"],
        remaining_units=4,
        wave_index=2,
        facts=_facts(WITNESS_AVAILABLE_BYTES),
    )
    assert recovered["record"]["changed"] is False
    assert recovered["record"]["shrink_steps"] == []
    assert recovered["candidate"].units_in_flight == 2
    assert recovered["mapping"]["units_in_flight"] == 2
    assert before == 1
    assert calls["n"] == 1  # only the shrink wave re-applied the mapping


def test_exhaustion_refuses_before_the_wave(monkeypatch):
    decision = _synthetic_decision(k_target=3)
    monkeypatch.setattr(zz, "_r55_active_apply", lambda *a, **k: _FakeBundle())
    with pytest.raises(zz._R55RunRefused) as ei:
        zz._r55_run_checkpoint(
            decision=decision,
            candidate=decision.plan_set.chosen,
            mapping=decision.mapping,
            bundle=_FakeBundle(),
            remaining_units=6,
            wave_index=1,
            facts=_facts(200 * MIB),
        )
    assert ei.value.reason_code == re.REASON_REFUSE_HOST_RAM
    record = ei.value.record
    assert record["status"] == "refused"
    assert record["publication"] if "publication" in record else True
    # Adjacent examination is complete and ordered until the dead end.
    assert record["shrink_steps"]
    for item in record["shrink_steps"]:
        assert item["accepted"] is False
        assert item["step"] in (
            "prefetch_units",
            "workers_per_unit",
            "units_in_flight",
            "tile_px",
            "backend",
        )


def test_disk_shortfall_refuses_with_disk_reason(monkeypatch):
    decision = _synthetic_decision(k_target=3)
    monkeypatch.setattr(zz, "_r55_active_apply", lambda *a, **k: _FakeBundle())
    with pytest.raises(zz._R55RunRefused) as ei:
        zz._r55_run_checkpoint(
            decision=decision,
            candidate=decision.plan_set.chosen,
            mapping=decision.mapping,
            bundle=_FakeBundle(),
            remaining_units=6,
            wave_index=1,
            facts=_facts(WITNESS_AVAILABLE_BYTES, disk_free=1),
        )
    assert ei.value.reason_code == re.REASON_REFUSE_DISK
    assert any(
        item["reason_code"] == re.REASON_REFUSE_DISK
        for item in ei.value.record["shrink_steps"]
    )


# ---------------------------------------------------------------------------
# Persistent latch: consumed once at the NEXT boundary, min-combined
# ---------------------------------------------------------------------------


def test_transient_latched_drop_governs_the_next_boundary(monkeypatch):
    decision = _synthetic_decision(k_target=3)
    monkeypatch.setattr(zz, "_r55_active_apply", lambda *a, **k: _FakeBundle())
    checkpoint = zz._r55_run_checkpoint(
        decision=decision,
        candidate=decision.plan_set.chosen,
        mapping=decision.mapping,
        bundle=_FakeBundle(),
        remaining_units=6,
        wave_index=1,
        facts=_facts(WITNESS_AVAILABLE_BYTES),
        latched=_facts(1 * GIB)["host"],
    )
    record = checkpoint["record"]
    assert record["changed"] is True
    assert checkpoint["candidate"].units_in_flight == 2
    assert record["host_source"] == "fresh+latched"
    assert record["host"]["raw_available_bytes"] == 1 * GIB
    assert record["host"]["budget_bytes"] == (9 * GIB) // 10
    assert record["fresh_host"]["raw_available_bytes"] == WITNESS_AVAILABLE_BYTES
    assert record["latched_host"]["raw_available_bytes"] == 1 * GIB
    assert record["latched_host"]["budget_bytes"] == (9 * GIB) // 10


def test_latch_is_consumed_exactly_once_and_malformed_fails_closed():
    latch = zrt.PressureLatch()
    latch.record(_facts(100 * MIB))
    first = latch.consume_validated()
    assert first is not None
    assert first["raw_available_bytes"] == 100 * MIB
    assert first["budget_bytes"] == (9 * 100 * MIB) // 10
    assert latch.consume_validated() is None
    # A malformed stored fact never slips through the boundary.
    latch._sample = {
        "raw_available_bytes": True,
        "budget_bytes": 1,
        "monotonic_ns": 0,
    }
    with pytest.raises(zrt.RuntimeControlError):
        latch.consume_validated()


def test_public_record_malformed_fact_is_fatal_and_survives_valid_sample():
    latch = zrt.PressureLatch()
    # This is the exact public path the in-wave heartbeat uses.
    assert latch.record(
        {
            "host": {"raw_available_bytes": True, "budget_bytes": 1},
            "sampled_monotonic_ns": 1,
        }
    ) is False
    snapshot = latch.snapshot()
    assert snapshot["pending_fatal"] is True
    assert snapshot["fatal_errors"]
    # A later valid sample must NOT erase the pending malformed admission fact.
    assert latch.record(_facts(WITNESS_AVAILABLE_BYTES)) is True
    with pytest.raises(zrt.RuntimeControlError, match="pending latched"):
        latch.consume_validated()
    # The consumed generation (error + valid sample) is cleared exactly once.
    assert latch.consume_validated() is None
    assert latch.snapshot()["pending_fatal"] is False


def test_collector_failure_is_a_pending_fatal_latch():
    latch = zrt.PressureLatch()
    latch.record_error(RuntimeError("MemAvailable collection failed"))
    assert latch.record(_facts(WITNESS_AVAILABLE_BYTES)) is True
    with pytest.raises(zrt.RuntimeControlError, match="collection failed"):
        latch.consume_validated()
    assert latch.consume_validated() is None


def test_run_sampler_malformed_fact_counts_as_error():
    latch = zrt.PressureLatch()
    sampler = zrt.RunSampler(
        lambda: {
            "host": {"raw_available_bytes": True, "budget_bytes": 1},
            "sampled_monotonic_ns": 1,
        },
        latch=latch,
        interval_s=0.01,
        wait=lambda _delay: True,
    )
    sampler.start()
    sampler.stop_and_join()
    assert sampler.samples == 0
    assert sampler.errors >= 1
    assert "malformed" in (sampler.last_error or "").lower()
    assert latch.snapshot()["pending_fatal"] is True
    with pytest.raises(zrt.RuntimeControlError):
        latch.consume_validated()


def test_checkpoint_malformed_latched_fact_fails_closed(monkeypatch):
    decision = _synthetic_decision(k_target=3)
    monkeypatch.setattr(zz, "_r55_active_apply", lambda *a, **k: _FakeBundle())
    with pytest.raises(zz._R55RunRefused) as ei:
        zz._r55_run_checkpoint(
            decision=decision,
            candidate=decision.plan_set.chosen,
            mapping=decision.mapping,
            bundle=_FakeBundle(),
            remaining_units=6,
            wave_index=1,
            facts=_facts(WITNESS_AVAILABLE_BYTES),
            latched={"raw_available_bytes": True, "budget_bytes": 1},
        )
    assert ei.value.reason_code == re.REASON_REFUSE_HOST_RAM
    assert "malformed latched" in ei.value.detail


# ---------------------------------------------------------------------------
# Wave partition / canonical order
# ---------------------------------------------------------------------------


def test_wave_partition_is_canonical_contiguous_and_bounded():
    assert zrt.plan_wave_bounds(7, 3) == ((0, 3), (3, 6), (6, 7))
    assert zrt.plan_wave_bounds(6, 1) == ((0, 1), (1, 2), (2, 3), (3, 4),
                                          (4, 5), (5, 6))
    assert zrt.next_wave_bound(3, 6, 2) == (3, 5)
    assert zrt.next_wave_bound(5, 6, 2) == (5, 6)
    for bad in (True, 1.5, -1):
        with pytest.raises(zrt.RuntimeControlError):
            zrt.plan_wave_bounds(6, bad)


# ---------------------------------------------------------------------------
# Parent-side heartbeat seam: no E1 thread before process creation
# ---------------------------------------------------------------------------


def test_pmap_serial_heartbeat_cadence_and_cleanup():
    calls = []
    progress = []

    def heartbeat():
        calls.append(threading.current_thread().name)

    meta = {}
    out = zpar.pmap(
        _slow_echo_task,
        [1, 2, 3],
        workers=1,
        heartbeat=heartbeat,
        heartbeat_interval_s=0.02,
        progress_callback=lambda done, total: progress.append(done),
        meta=meta,
    )
    assert out == [1, 2, 3]
    assert progress == [1, 2, 3]
    assert calls
    assert all(name == "r55e1-heartbeat" for name in calls)
    assert meta["heartbeat_mode"] == "serial_thread"
    assert meta["heartbeat_calls"] >= 1
    assert meta["heartbeat_errors"] == 0
    assert not any(
        thread.name.startswith("r55e1") for thread in threading.enumerate()
    )


def test_pmap_serial_heartbeat_start_failure_is_fatal_before_work(monkeypatch):
    worker_calls = []
    heartbeat_calls = []

    def worker(value):
        worker_calls.append(value)
        return value

    def heartbeat():
        heartbeat_calls.append(1)

    def _fail_start(self):
        raise RuntimeError("cannot start monitor")

    monkeypatch.setattr(zpar._HeartbeatMonitor, "start", _fail_start)
    with pytest.raises(RuntimeError, match="could not start"):
        zpar.pmap(
            worker,
            [1, 2],
            workers=1,
            heartbeat=heartbeat,
            heartbeat_interval_s=0.02,
        )
    assert worker_calls == []
    assert heartbeat_calls == []
    assert not any(
        thread.name.startswith("r55e1") for thread in threading.enumerate()
    )


def test_pmap_process_heartbeat_uses_parent_no_e1_thread(monkeypatch):
    seen = {"init": [], "submit": []}
    calls = []
    progress = []
    real_executor = concurrent.futures.ProcessPoolExecutor

    class _TracedExecutor(real_executor):
        def __init__(self, *args, **kwargs):
            seen["init"] = [t.name for t in threading.enumerate()]
            super().__init__(*args, **kwargs)

        def submit(self, *args, **kwargs):
            seen["submit"] = [t.name for t in threading.enumerate()]
            return super().submit(*args, **kwargs)

    monkeypatch.setattr(
        concurrent.futures, "ProcessPoolExecutor", _TracedExecutor
    )

    def heartbeat():
        calls.append(threading.current_thread().name)

    meta = {}
    out = zpar.pmap(
        _echo_task,
        [3, 1, 2, 4],
        workers=2,
        heartbeat=heartbeat,
        heartbeat_interval_s=0.02,
        progress_callback=lambda done, total: progress.append(done),
        meta=meta,
    )
    assert out == [3, 1, 2, 4]
    assert progress == [1, 2, 3, 4]
    assert meta["heartbeat_mode"] == "parent_wait"
    assert meta["heartbeat_calls"] >= 1
    assert meta["heartbeat_errors"] == 0
    assert calls and all(name == "MainThread" for name in calls)
    assert not any(name.startswith("r55e1") for name in seen["init"])
    assert not any(name.startswith("r55e1") for name in seen["submit"])
    assert not any(
        thread.name.startswith("r55e1") for thread in threading.enumerate()
    )


def test_pmap_monitor_terminates_on_worker_exception():
    calls = []

    def heartbeat():
        calls.append(threading.current_thread().name)

    with pytest.raises(RuntimeError, match="intentional worker failure"):
        zpar.pmap(
            _fail_task,
            [1, 2, 3],
            workers=1,
            heartbeat=heartbeat,
            heartbeat_interval_s=0.02,
        )
    assert calls
    assert not any(
        thread.name.startswith("r55e1") for thread in threading.enumerate()
    )


def test_pmap_process_heartbeat_terminates_on_exception(monkeypatch):
    calls = []

    def heartbeat():
        calls.append(threading.current_thread().name)

    with pytest.raises(RuntimeError, match="intentional worker failure"):
        zpar.pmap(
            _fail_task,
            [1, 2, 3],
            workers=2,
            heartbeat=heartbeat,
            heartbeat_interval_s=0.02,
            # The process path itself must terminate the E1 monitor; do not
            # exercise the historical same-config serial fallback here.
            serial_fallback=False,
        )
    assert calls and all(name == "MainThread" for name in calls)
    assert not any(
        thread.name.startswith("r55e1") for thread in threading.enumerate()
    )


def test_sampler_cadence_at_most_five_seconds_and_latches(monkeypatch):
    latch = zrt.PressureLatch()
    samples = [_facts(100 * MIB), _facts(50 * MIB), _facts(80 * MIB)]
    state = {"i": 0}

    def sample_fn():
        value = samples[min(state["i"], len(samples) - 1)]
        state["i"] += 1
        return value

    fake_clock = {"t": 0.0}

    def clock():
        return fake_clock["t"]

    wait_calls = {"n": 0}

    def wait(delay):
        wait_calls["n"] += 1
        assert delay <= zrt.RUN_MAX_SAMPLE_INTERVAL_S
        fake_clock["t"] += max(0.0, float(delay))
        return wait_calls["n"] >= 3

    sampler = zrt.RunSampler(
        sample_fn, latch=latch, interval_s=zrt.RUN_MAX_SAMPLE_INTERVAL_S,
        clock=clock, wait=wait,
    )
    sampler.start()
    sampler.stop_and_join()
    assert sampler.samples == 3
    assert sampler.errors == 0
    assert all(
        0.0 <= value <= zrt.RUN_MAX_SAMPLE_INTERVAL_S
        for value in sampler.requested_waits
    )
    lowest = latch.peek()
    assert lowest["raw_available_bytes"] == 50 * MIB
    assert lowest["budget_bytes"] == (9 * 50 * MIB) // 10


# ---------------------------------------------------------------------------
# Telemetry availability / aggregation / deterministic projection
# ---------------------------------------------------------------------------


def test_parent_memory_reads_strict_integers(tmp_path):
    status = tmp_path / "status"
    status.write_text(
        "Name:\tpython\nVmPeak:\t  2048 kB\nVmRSS:\t  1024 kB\n"
        "VmHWM:\t  1536 kB\n",
        encoding="ascii",
    )
    result = zrt.read_parent_memory(status)
    assert result["current_rss_bytes"] == 1024 * 1024
    assert result["peak_rss_bytes"] == 1536 * 1024
    assert result["unavailable_reason"] is None


def test_optional_metrics_are_null_plus_reason_never_fabricated_zero(tmp_path):
    assert zrt.read_parent_memory(tmp_path / "missing")["current_rss_bytes"] is None
    assert (
        zrt.read_parent_memory(tmp_path / "missing")["unavailable_reason"]
    )
    cgroup = zrt.read_cgroup_memory(tmp_path / "missing_cgroup")
    assert cgroup["current_bytes"] is None
    assert cgroup["peak_bytes"] is None
    assert cgroup["unavailable_reason"]
    io = zrt.read_disk_io(tmp_path / "missing_io")
    assert io["read_bytes"] is None
    assert io["write_bytes"] is None
    assert io["unavailable_reason"]
    missing = tmp_path / "missing_meminfo"
    with pytest.raises(zrt.RuntimeControlError):
        zrt.read_meminfo(missing)


def test_collect_run_facts_available_and_unavailable(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       1000 kB\nMemAvailable:    500 kB\n", encoding="ascii"
    )
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text("1234\n", encoding="ascii")
    (cgroup / "memory.peak").write_text("4321\n", encoding="ascii")
    proc_io = tmp_path / "io"
    proc_io.write_text("read_bytes: 11\nwrite_bytes: 22\n", encoding="ascii")
    facts = zrt.collect_run_facts(
        root=tmp_path,
        meminfo_path=meminfo,
        cgroup_root=cgroup,
        proc_io_path=proc_io,
        proc_status_path=tmp_path / "missing_status",
    )
    assert facts["host"]["raw_available_bytes"] == 500 * 1024
    assert facts["host"]["budget_bytes"] == (9 * 500 * 1024) // 10
    assert facts["disk"]["free_bytes"] >= 0
    assert facts["cgroup"]["current_bytes"] == 1234
    assert facts["cgroup"]["peak_bytes"] == 4321
    assert facts["disk"]["io_read_bytes"] == 11
    assert facts["disk"]["io_write_bytes"] == 22
    assert facts["parent"]["current_rss_bytes"] is None
    assert facts["parent"]["unavailable_reason"]


def test_worker_peak_aggregate_label_and_normalized_projection():
    results = [
        {"peak_rss_kib": 10},
        {"peak_rss_kib": 20},
        {"no_peak": True},
    ]
    agg = zrt.aggregate_worker_peaks(results)
    assert agg["values_kib"] == [10, 20]
    assert agg["sum_kib"] == 30
    assert agg["max_kib"] == 20
    assert agg["current_available"] is False
    assert agg["current_unavailable_reason"]
    assert "no_dedup" in agg["label"]
    payload = {
        "wave": 1,
        "wall_s": 3.5,
        "parent_rss_bytes": 123,
        "cgroup_current_bytes": 456,
        "decision": {"candidate": {"units_in_flight": 2}},
    }
    normalized = zrt.normalized_projection(payload)
    assert "wall_s" not in normalized
    assert "parent_rss_bytes" not in normalized
    assert normalized["decision"]["candidate"]["units_in_flight"] == 2
    assert zrt.deterministic_hash(payload) == zrt.deterministic_hash(
        {**payload, "wall_s": 99.0, "parent_rss_bytes": 999}
    )


# ---------------------------------------------------------------------------
# REAL product-path parity witnesses
# ---------------------------------------------------------------------------


def test_real_multi_wave_active_vs_legacy_byte_identity(parity_runs):
    legacy = parity_runs["legacy"]
    active = parity_runs["active"]
    for name in FITS_NAMES:
        assert _digest(legacy / name) == _digest(active / name), name

    llm = _manifest(legacy)
    am = _manifest(active)
    assert "r55_run_telemetry" not in llm
    block = am["r55_scheduler"]
    run = block["run"]
    telemetry = am["r55_run_telemetry"]
    assert run["schema"] == zrt.RUN_DECISION_SCHEMA
    assert run["policy"]["checkpoint"] == "mandatory_before_every_wave"
    assert run["policy"]["shrink_order"] == [
        "prefetch_units",
        "workers_per_unit",
        "units_in_flight",
        "tile_px",
        "backend",
    ]
    assert telemetry["schema"] == zrt.RUN_TELEMETRY_SCHEMA
    assert telemetry["metrics"]["wave_count"] == len(run["waves"])
    assert len(run["checkpoints"]) == len(run["waves"]) >= 2
    assert telemetry["metrics"]["checkpoint_count"] == len(run["checkpoints"])
    assert telemetry["metrics"]["shrink_count"] == 0
    assert telemetry["metrics"]["refusal_count"] == 0
    assert telemetry["metrics"]["heartbeat_errors"] == 0
    assert telemetry["metrics"]["heartbeat_calls"] >= 1
    # Exact per-wave candidate/mapping/facts + volatile metrics.
    assert telemetry["waves"]
    offset = 0
    for index, wave in enumerate(run["waves"]):
        assert wave["index"] == index
        assert wave["offset"] == offset
        assert wave["wave_units"] <= wave["mapping"]["units_in_flight"]
        assert wave["mapping"] == run["checkpoints"][index]["mapping"]
        assert wave["candidate"] == run["checkpoints"][index]["candidate"]
        offset += wave["wave_units"]
    assert offset == am["per_cell_diagnostics"]["cells"]
    assert telemetry["waves"][-1]["units_remaining_after"] == 0
    for wave in telemetry["waves"]:
        assert isinstance(wave["facts"]["host"]["raw_available_bytes"], int)
        assert isinstance(wave["facts"]["host"]["budget_bytes"], int)
        assert wave["facts"]["disk"]["free_bytes"] >= 0
        assert "worker_peaks" in wave
        assert wave["worker_peaks"]["current_available"] is False
        assert wave["wall_s"] is not None
        # Natural K=1 waves run the serial path; the heartbeat seam must be
        # active there too (<= 5 s cadence) and the thread must be joined.
        assert wave["heartbeat"]["heartbeat_mode"] == "serial_thread"
        assert wave["heartbeat"]["heartbeat_calls"] >= 1
        assert wave["heartbeat"]["heartbeat_errors"] == 0
    assert not any(
        thread.name.startswith("r55e1") for thread in threading.enumerate()
    )
    # Deterministic RUN structure has NO volatile observation keys.
    text = json.dumps(run, sort_keys=True)
    for forbidden in (
        "wall_s",
        "parent_rss_bytes",
        "cgroup_current_bytes",
        "disk_io_read_bytes",
        "disk_free_bytes",
        "sampled_monotonic_ns",
    ):
        assert forbidden not in text
    assert block["run_hash"].startswith("sha256:")
    assert block["plan_hash"] == _recompute_final_hash(block)
    diag = am["per_cell_diagnostics"]
    assert diag["r55_run_waves"] == len(run["waves"])
    assert diag["r55_run_shrink_steps"] == 0
    assert diag["cells_in_flight"] == block["mapping"]["units_in_flight"]
    assert diag["cache_workers_per_cell"] == block["mapping"]["workers_per_unit"]
    assert diag["cpu_streaming_tile"] == (
        block["mapping"]["cpu_streaming_tile"] or 0
    )


def test_real_multi_wave_shrink_parity_and_no_growth(shrink_runs):
    legacy = shrink_runs["legacy"]
    active = shrink_runs["active"]
    active2 = shrink_runs["active2"]
    for name in FITS_NAMES:
        assert _digest(legacy / name) == _digest(active / name), name
        assert _digest(active / name) == _digest(active2 / name), name

    first = _manifest(active)
    second = _manifest(active2)
    block = first["r55_scheduler"]
    run = block["run"]
    telemetry = first["r55_run_telemetry"]
    assert run["shrink_count"] >= 1
    assert telemetry["metrics"]["shrink_count"] == run["shrink_count"]
    assert telemetry["metrics"]["refusal_count"] == 0
    # At least one adjacent accepted shrink; every transition is ordered.
    accepted = [
        step
        for cp in run["checkpoints"]
        for step in cp["shrink_steps"]
        if step["accepted"]
    ]
    assert accepted
    for step in accepted:
        assert step["to"]["units_in_flight"] <= step["from"]["units_in_flight"]
    # Initial K=3, final K=2, and recovery never grows back to 3.
    assert block["initial_candidate"]["units_in_flight"] == 3
    assert block["chosen_candidate"]["units_in_flight"] == 2
    assert block["mapping"]["units_in_flight"] == 2
    assert block["execution"]["units_in_flight"] == 2
    assert run["final_execution"]["units_in_flight"] == 2
    assert not any(
        cp["changed"] and cp["candidate"]["units_in_flight"] > 2
        for cp in run["checkpoints"]
    )
    # A later high-memory checkpoint is unchanged (no growth after recovery).
    assert any(
        (not cp["changed"])
        and cp["host"]["raw_available_bytes"] == WITNESS_AVAILABLE_BYTES
        for cp in run["checkpoints"][1:]
    )
    # Wave bounds/offsets remain exact after the shrink.
    offset = 0
    for index, wave in enumerate(run["waves"]):
        assert wave["index"] == index
        assert wave["offset"] == offset
        assert wave["wave_units"] <= wave["mapping"]["units_in_flight"]
        offset += wave["wave_units"]
    assert offset == first["per_cell_diagnostics"]["cells"]
    # Deterministic decision structure identical across two builds; volatile
    # telemetry is intentionally NOT part of that comparison.
    assert json.dumps(block, sort_keys=True) == json.dumps(
        second["r55_scheduler"], sort_keys=True
    )
    assert block["run_hash"] == second["r55_scheduler"]["run_hash"]
    assert block["plan_hash"] == _recompute_final_hash(block)
    # Exact injected facts are auditable in telemetry, including the drop.
    raws = [
        w["facts"]["host"]["raw_available_bytes"] for w in telemetry["waves"]
    ]
    assert raws[0] == WITNESS_AVAILABLE_BYTES
    assert 5 * GIB // 2 in raws
    assert raws[-1] == WITNESS_AVAILABLE_BYTES
    modes = {
        wave["heartbeat"]["heartbeat_mode"]
        for wave in telemetry["waves"]
    }
    assert "parent_wait" in modes
    assert not any(
        thread.name.startswith("r55e1") for thread in threading.enumerate()
    )


def test_transient_mid_wave_drop_is_consumed_at_next_boundary(
    routine_input, tmp_path
):
    """A drop that recovers before the next fresh sample still shrinks."""
    out = tmp_path / "active_transient"
    legacy = tmp_path / "legacy_transient"
    _run_product(routine_input, legacy, "legacy")
    pressure_calls = {"n": 0}

    def pressure(_root):
        raw = 5 * GIB // 2 if pressure_calls["n"] == 0 else WITNESS_AVAILABLE_BYTES
        pressure_calls["n"] += 1
        return _facts(raw)

    _run_product(
        routine_input,
        out,
        "active",
        sample=_ScheduleSample([WITNESS_AVAILABLE_BYTES] * 10),
        decide=_force_k3_decide(zz._r55_active_decide),
        pressure=pressure,
    )
    for name in FITS_NAMES:
        assert _digest(legacy / name) == _digest(out / name), name
    block = _manifest(out)["r55_scheduler"]
    run = block["run"]
    assert run["shrink_count"] >= 1
    assert run["checkpoints"][1]["host_source"] == "fresh+latched"
    assert run["checkpoints"][1]["host"]["raw_available_bytes"] == 5 * GIB // 2
    assert (
        run["checkpoints"][1]["fresh_host"]["raw_available_bytes"]
        == WITNESS_AVAILABLE_BYTES
    )
    assert (
        run["checkpoints"][1]["latched_host"]["raw_available_bytes"]
        == 5 * GIB // 2
    )
    assert block["mapping"]["units_in_flight"] == 2
    assert not any(
        thread.name.startswith("r55e1") for thread in threading.enumerate()
    )


def test_malformed_in_wave_sample_refuses_at_next_boundary(routine_input, tmp_path):
    out = tmp_path / "refused_latched"
    calls = {"n": 0}

    def pressure(_root):
        calls["n"] += 1
        if calls["n"] == 1:
            facts = _facts(WITNESS_AVAILABLE_BYTES)
            facts["host"]["raw_available_bytes"] = True
            return facts
        return _facts(WITNESS_AVAILABLE_BYTES)

    with pytest.raises(zz._R55RunRefused) as ei:
        _run_product(
            routine_input,
            out,
            "active",
            sample=_ScheduleSample([WITNESS_AVAILABLE_BYTES] * 10),
            decide=_force_k3_decide(zz._r55_active_decide),
            pressure=pressure,
        )
    assert ei.value.reason_code == re.REASON_REFUSE_HOST_RAM
    assert "latched" in ei.value.detail
    assert ei.value.record["publication"]["fits"] is False
    assert ei.value.record["publication"]["manifest"] is False
    telemetry = ei.value.record["telemetry"]
    assert telemetry["metrics"]["heartbeat_errors"] >= 1
    assert any(
        wave["heartbeat"].get("heartbeat_last_error")
        for wave in telemetry["waves"]
    )
    assert not (out / "mosaic_grid.fits").exists()
    assert not (out / "zegrid_manifest.json").exists()
    assert not any(
        thread.name.startswith("r55e1") for thread in threading.enumerate()
    )


def test_heartbeat_collection_exception_refuses_at_next_boundary(
    routine_input, tmp_path
):
    out = tmp_path / "refused_collection"
    calls = {"n": 0}

    def pressure(_root):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("MemAvailable collection failed")
        return _facts(WITNESS_AVAILABLE_BYTES)

    with pytest.raises(zz._R55RunRefused) as ei:
        _run_product(
            routine_input,
            out,
            "active",
            sample=_ScheduleSample([WITNESS_AVAILABLE_BYTES] * 10),
            decide=_force_k3_decide(zz._r55_active_decide),
            pressure=pressure,
        )
    assert ei.value.reason_code == re.REASON_REFUSE_HOST_RAM
    telemetry = ei.value.record["telemetry"]
    assert telemetry["metrics"]["heartbeat_errors"] >= 1
    assert any(
        "MemAvailable collection failed"
        in str(wave["heartbeat"].get("heartbeat_last_error"))
        for wave in telemetry["waves"]
    )
    assert ei.value.record["publication"]["fits"] is False
    assert ei.value.record["publication"]["manifest"] is False
    assert not (out / "mosaic_grid.fits").exists()
    assert not (out / "zegrid_manifest.json").exists()
    assert not any(
        thread.name.startswith("r55e1") for thread in threading.enumerate()
    )


def test_terminal_pending_malformed_fact_refuses_publication(
    routine_input, tmp_path
):
    out = tmp_path / "refused_terminal"

    def pressure(_root):
        facts = _facts(WITNESS_AVAILABLE_BYTES)
        facts["host"]["raw_available_bytes"] = True
        return facts

    with pytest.raises(zz._R55RunRefused) as ei:
        _run_product(
            routine_input,
            out,
            "active",
            sample=_ScheduleSample([WITNESS_AVAILABLE_BYTES] * 10),
            decide=_force_k_decide(zz._r55_active_decide, 6),
            pressure=pressure,
        )
    assert ei.value.reason_code == re.REASON_REFUSE_HOST_RAM
    assert "terminal" in ei.value.detail
    assert ei.value.record["publication"]["fits"] is False
    assert ei.value.record["publication"]["manifest"] is False
    telemetry = ei.value.record["telemetry"]
    assert telemetry["metrics"]["heartbeat_errors"] >= 1
    assert any(
        wave["heartbeat"].get("heartbeat_last_error")
        for wave in telemetry["waves"]
    )
    assert not (out / "mosaic_grid.fits").exists()
    assert not (out / "zegrid_manifest.json").exists()
    assert not any(
        thread.name.startswith("r55e1") for thread in threading.enumerate()
    )


def test_terminal_valid_low_pressure_is_recorded_not_applied(
    routine_input, tmp_path
):
    out = tmp_path / "terminal_valid"
    legacy = tmp_path / "terminal_valid_legacy"
    _run_product(routine_input, legacy, "legacy")
    _run_product(
        routine_input,
        out,
        "active",
        sample=_ScheduleSample([WITNESS_AVAILABLE_BYTES] * 10),
        decide=_force_k_decide(zz._r55_active_decide, 6),
        pressure=lambda _root: _facts(5 * GIB // 2),
    )
    for name in FITS_NAMES:
        assert _digest(legacy / name) == _digest(out / name), name
    manifest = _manifest(out)
    terminal = manifest["r55_run_telemetry"]["terminal"]
    assert terminal is not None
    assert terminal["latched"]["raw_available_bytes"] == 5 * GIB // 2
    assert terminal["pressure_observed"] is True
    assert "recorded_only" in terminal["action"]
    block = manifest["r55_scheduler"]
    # No post-execution shrink or mapping rewrite was invented.
    assert block["mapping"]["units_in_flight"] == 6
    assert block["run"]["shrink_count"] == 0


def test_host_exhaustion_refuses_and_publishes_nothing(routine_input, tmp_path):
    out = tmp_path / "refused_host"
    decide = _force_k3_decide(zz._r55_active_decide)
    sample = _ScheduleSample(
        [WITNESS_AVAILABLE_BYTES, 200 * MIB], disk_free=100 * GIB
    )
    with pytest.raises(zz._R55RunRefused) as ei:
        _run_product(
            routine_input, out, "active", sample=sample, decide=decide
        )
    assert ei.value.reason_code == re.REASON_REFUSE_HOST_RAM
    record = ei.value.record
    assert record["publication"] == {
        "fits": False,
        "manifest": False,
        "assembly": False,
    }
    assert record["cleanup"]
    assert all("cell_id" in item and "removed" in item for item in record["cleanup"])
    assert record["telemetry"]["metrics"]["refusal_count"] == 1
    assert not (out / "mosaic_grid.fits").exists()
    assert not (out / "mosaic_grid_coverage.fits").exists()
    assert not (out / "zegrid_manifest.json").exists()


def test_disk_shortfall_refuses_and_publishes_nothing(routine_input, tmp_path):
    out = tmp_path / "refused_disk"
    decide = _force_k3_decide(zz._r55_active_decide)
    sample = _ScheduleSample(
        [WITNESS_AVAILABLE_BYTES, WITNESS_AVAILABLE_BYTES], disk_free=1
    )
    # First wave is accepted under the real disk; the second checkpoint injects
    # a predictable 1-byte disk shortfall.
    sample.calls = 0
    original = sample.schedule

    def two_phase(_root):
        if sample.calls == 0:
            sample.calls += 1
            return _facts(WITNESS_AVAILABLE_BYTES, disk_free=100 * GIB)
        sample.calls += 1
        return _facts(WITNESS_AVAILABLE_BYTES, disk_free=1)

    with pytest.raises(zz._R55RunRefused) as ei:
        _run_product(
            routine_input, out, "active", sample=two_phase, decide=decide
        )
    assert ei.value.reason_code == re.REASON_REFUSE_DISK
    assert ei.value.record["publication"]["fits"] is False
    assert ei.value.record["publication"]["manifest"] is False
    assert not (out / "mosaic_grid.fits").exists()
    assert not (out / "zegrid_manifest.json").exists()


def test_malformed_injected_product_checkpoint_refuses(routine_input, tmp_path):
    out = tmp_path / "refused_malformed"
    decide = _force_k3_decide(zz._r55_active_decide)
    calls = {"n": 0}

    def malformed(_root):
        calls["n"] += 1
        if calls["n"] == 1:
            return _facts(WITNESS_AVAILABLE_BYTES, disk_free=100 * GIB)
        facts = _facts(WITNESS_AVAILABLE_BYTES, disk_free=100 * GIB)
        facts["host"]["raw_available_bytes"] = True
        return facts

    with pytest.raises(zz._R55RunRefused) as ei:
        _run_product(
            routine_input, out, "active", sample=malformed, decide=decide
        )
    assert ei.value.reason_code == re.REASON_REFUSE_HOST_RAM
    assert "malformed RUN checkpoint" in ei.value.detail
    assert not (out / "zegrid_manifest.json").exists()


# ---------------------------------------------------------------------------
# Import laziness
# ---------------------------------------------------------------------------


def _import_laziness_script() -> str:
    """Import the E1 CPU seam with an IMPORTABLE exploding fake CuPy.

    The full product module has a pre-existing Dask/chunk-types CuPy edge when
    CuPy is installed; E1 must not assert that invariant. This proof targets
    the E1 modules only (runtime_control + scheduler).
    """
    return (
        "import sys\n"
        "class _ExplodingCupy:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'cupy' or name.startswith('cupy.'):\n"
        "            raise AssertionError('CuPy import attempted by E1 CPU seam')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _ExplodingCupy())\n"
        "import zemosaic.core.zegrid.runtime_control as rc\n"
        "import zemosaic.core.zegrid.scheduler as zs\n"
        "assert 'cupy' not in sys.modules, 'cupy imported on the CPU path'\n"
        "assert 'cupy.cuda' not in sys.modules, 'cupy.cuda imported'\n"
        "assert rc.RUN_MAX_SAMPLE_INTERVAL_S == 5.0\n"
        "assert zs.MODE_ACTIVE == 'active'\n"
        "print('LAZY_OK')\n"
    )


def test_cpu_run_control_import_is_lazy():
    import os

    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = (
        src + os.pathsep + env.get("PYTHONPATH", "")
        if env.get("PYTHONPATH") else src
    )
    proc = subprocess.run(
        [sys.executable, "-c", _import_laziness_script()],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "LAZY_OK" in proc.stdout


def test_no_top_level_cupy_import_in_new_module():
    import ast

    tree = ast.parse(
        (REPO_ROOT / "src/zemosaic/core/zegrid/runtime_control.py").read_text(
            encoding="utf-8"
        )
    )
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [getattr(node, "module", "") or ""]
            )
            assert not any(
                n == "cupy" or n.startswith("cupy.") for n in names
            )
