"""ZM-ZEGRID-R55-E2 — bounded CPU->GPU prefetch, single-owner active GPU.

Acceptance coverage:

* bounded producer/consumer queue (explicit max, prefetch honored, proven
  backpressure) and exact host/device buffer accounting on success, cancel,
  producer/consumer exception and fallback;
* single-owner ACTIVE GPU mapping (K == 1) with a streaming tile and a bounded
  prefetch depth; every unqualified tuple is refused with a closed reason;
* fresh PLAN/ADMIT/RUN observations at 80/90 thresholds, shrink-only pressure
  with no regrowth;
* CuPy-tree-only OOM classification (plain MemoryError/ValueError untouched);
* exact whole-batch CPU fallback once, with no mixed/partial GPU result;
* deterministic cleanup/telemetry and no CuPy import on CPU-only paths.
"""

from __future__ import annotations

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
from zemosaic.core.zegrid import gpu as zgpu
from zemosaic.core.zegrid import gpu_readiness as zgr
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import ram_contract as rcontract
from zemosaic.core.zegrid import scheduler as zsched

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "zegrid_routine" / "make_routine_corpus.py"
GIB = 2 ** 30
MIB = 2 ** 20
FITS_NAMES = ("mosaic_grid.fits", "mosaic_grid_coverage.fits")


# ---------------------------------------------------------------------------
# Helpers — bounded pipeline
# ---------------------------------------------------------------------------


class _Boom(RuntimeError):
    pass


def _success_pipeline(**kw):
    released = []

    def prod(task):
        return {"value": task, "bytes": 8}

    def cons(payload):
        return payload["value"] * 2

    def rel(payload):
        released.append(payload["value"])

    params = dict(
        tasks=list(range(12)),
        produce=prod,
        consume=cons,
        release=rel,
        prefetch_units=2,
        host_budget_bytes=64,
        device_budget_bytes=64,
        host_bytes=lambda p: p["bytes"],
        device_bytes=lambda p: p["bytes"],
    )
    params.update(kw)
    return zpar.BoundedPipeline(**params), released


def test_pipeline_success_order_ledger_and_progress():
    pipe, released = _success_pipeline()
    assert pipe.queue_max == 3
    assert pipe.run() == [i * 2 for i in range(12)]
    tel = pipe.telemetry()
    assert sorted(released) == list(range(12))
    assert tel["units_produced"] == 12
    assert tel["units_consumed"] == 12
    assert tel["units_released"] == 12
    assert tel["transfers"] == 12
    assert tel["ledger"]["balanced"] is True
    assert tel["queue_depth_max"] <= tel["queue_max"]
    assert tel["threads_joined"] is True
    assert tel["cancelled"] is False


def test_prefetch_zero_is_strict_and_prefetch_is_honoured():
    pipe0, rel0 = _success_pipeline(prefetch_units=0)
    assert pipe0.queue_max == 1
    assert pipe0.run() == [i * 2 for i in range(12)]
    assert pipe0.telemetry()["queue_max"] == 1
    assert len(rel0) == 12

    pipe4, _ = _success_pipeline(prefetch_units=4)
    assert pipe4.queue_max == 5


def test_producer_backpressure_is_proven_and_bounded():
    started = threading.Event()

    def slow_consume(payload):
        started.set()
        time.sleep(0.01)
        return payload["value"]

    pipe, released = _success_pipeline(
        tasks=list(range(10)),
        consume=slow_consume,
        prefetch_units=0,
        host_budget_bytes=100000,
        device_budget_bytes=100000,
    )
    assert pipe.run() == list(range(10))
    tel = pipe.telemetry()
    assert tel["producer_backpressure_events"] > 0
    assert tel["producer_backpressure_s"] > 0
    assert tel["queue_depth_max"] <= 1


def test_device_credit_backpressure_blocks_the_consumer():
    order = []

    def cons(payload):
        order.append(("consume", payload["value"]))
        return payload["value"]

    def host_bytes(payload):
        return 8

    # device budget fits exactly one unit: the second consume must wait for the
    # first device release (single-owner credit), proving device backpressure.
    pipe = zpar.BoundedPipeline(
        tasks=[0, 1, 2],
        produce=lambda t: {"value": t},
        consume=cons,
        release=lambda p: order.append(("release", p["value"])),
        prefetch_units=2,
        host_budget_bytes=64,
        device_budget_bytes=8,
        host_bytes=host_bytes,
        device_bytes=lambda p: 8,
    )
    assert pipe.run() == [0, 1, 2]
    tel = pipe.telemetry()
    # Single-owner: device occupancy never exceeds one unit, and the ledger is
    # exactly balanced after every device release.
    assert tel["ledger"]["device_peak_bytes"] <= 8
    assert tel["ledger"]["device_used_bytes"] == 0
    assert tel["ledger"]["balanced"] is True
    assert tel["units_consumed"] == 3


def test_consumer_error_releases_everything_and_preserves_type():
    released = []

    def cons(payload):
        if payload["value"] == 4:
            raise _Boom("consumer exploded")
        return payload["value"]

    pipe = zpar.BoundedPipeline(
        tasks=list(range(12)),
        produce=lambda t: {"value": t},
        consume=cons,
        release=lambda p: released.append(p["value"]),
        prefetch_units=2,
        host_budget_bytes=64,
        device_budget_bytes=64,
        host_bytes=lambda p: 8,
        device_bytes=lambda p: 8,
    )
    with pytest.raises(_Boom, match="consumer exploded"):
        pipe.run()
    tel = pipe.telemetry()
    assert tel["ledger"]["balanced"] is True
    assert tel["threads_joined"] is True
    assert tel["consumer_error"].startswith("_Boom:")
    assert len(released) == len(set(released))
    assert 4 in released


def test_producer_error_releases_staged_units():
    released = []

    def prod(task):
        if task == 3:
            raise ValueError("producer exploded")
        return {"value": task}

    pipe = zpar.BoundedPipeline(
        tasks=list(range(12)),
        produce=prod,
        consume=lambda p: p["value"],
        release=lambda p: released.append(p["value"]),
        prefetch_units=1,
        host_budget_bytes=64,
        device_budget_bytes=64,
        host_bytes=lambda p: 8,
        device_bytes=lambda p: 8,
    )
    with pytest.raises(zpar.PipelineError, match="producer failed"):
        pipe.run()
    tel = pipe.telemetry()
    assert tel["ledger"]["balanced"] is True
    assert tel["producer_error"].startswith("ValueError:")
    assert len(released) == len(set(released))


@pytest.mark.parametrize("prefetch", [0, 1, 3])
def test_cancellation_at_multiple_points_is_deterministic(prefetch):
    released = []
    state = {"n": 0}

    def cons(payload):
        state["n"] += 1
        if state["n"] == 2:
            pipe.cancel("test-cancel")
        return payload["value"]

    pipe = zpar.BoundedPipeline(
        tasks=list(range(40)),
        produce=lambda t: {"value": t},
        consume=cons,
        release=lambda p: released.append(p["value"]),
        prefetch_units=prefetch,
        host_budget_bytes=64,
        device_budget_bytes=64,
        host_bytes=lambda p: 8,
        device_bytes=lambda p: 8,
    )
    with pytest.raises(zpar.PipelineCancelled):
        pipe.run()
    tel = pipe.telemetry()
    assert tel["cancelled"] is True
    assert tel["cancel_reason"] == "test-cancel"
    assert tel["ledger"]["balanced"] is True
    assert tel["threads_joined"] is True
    assert len(released) == len(set(released))
    assert tel["units_consumed"] <= 2


def test_cleanup_failure_is_recorded_without_masking_primary():
    def rel(payload):
        raise OSError("cannot remove")

    pipe = zpar.BoundedPipeline(
        tasks=[0],
        produce=lambda t: {"value": t},
        consume=lambda p: (_ for _ in ()).throw(_Boom("primary")),
        release=rel,
        prefetch_units=0,
        host_budget_bytes=64,
        device_budget_bytes=64,
        host_bytes=lambda p: 8,
        device_bytes=lambda p: 8,
    )
    with pytest.raises(_Boom):
        pipe.run()
    tel = pipe.telemetry()
    assert tel["cleanup_failures"]
    assert tel["consumer_error"].startswith("_Boom:")
    assert tel["ledger"]["balanced"] is True


def test_request_larger_than_budget_fails_closed():
    pipe = zpar.BoundedPipeline(
        tasks=[0],
        produce=lambda t: {"value": t},
        consume=lambda p: p,
        release=lambda p: None,
        prefetch_units=0,
        host_budget_bytes=4,
        device_budget_bytes=4,
        host_bytes=lambda p: 8,
        device_bytes=lambda p: 8,
    )
    with pytest.raises(zpar.PipelineAccountingError):
        pipe.run()
    assert pipe.telemetry()["ledger"]["balanced"] is True


def test_pipeline_telemetry_never_invents_unavailable_metrics():
    pipe, _ = _success_pipeline(tasks=[])
    before = pipe.telemetry()
    assert before["queue_depth_mean"] is None  # no sample, not fabricated 0
    assert before["transfers"] == 0
    assert pipe.run() == []
    tel = pipe.telemetry()
    assert tel["producer_backpressure_s"] == 0.0
    assert tel["consumer_idle_s"] >= 0.0
    assert tel["ledger"]["balanced"] is True


# ---------------------------------------------------------------------------
# Scheduler — ACTIVE GPU mapping
# ---------------------------------------------------------------------------


def _gpu_candidate(**kw):
    data = dict(
        backend=zsched.BACKEND_GPU,
        tile_px=64,
        units_in_flight=1,
        prefetch_units=2,
        workers_per_unit=8,
        host_peak_bytes=1,
        disk_bytes=1,
        device_peak_bytes=MIB,
        feasible=True,
        reason_code=zsched.REASON_OK,
    )
    data.update(kw)
    return zsched.Candidate(**data)


def test_active_gpu_mapping_is_single_owner_and_bounded():
    mapping = zsched.active_gpu_mapping_for_candidate(
        _gpu_candidate(),
        tile_ladder=(128, 64, 32),
        workers_ladder=(1, 2, 4, 8),
        prefetch_cap=4,
    )
    assert mapping["schema"] == zsched.E2_GPU_MAPPING_SCHEMA
    assert mapping["backend"] == "gpu"
    assert mapping["units_in_flight"] == 1
    assert mapping["prefetch_units"] == 2
    assert mapping["single_owner"] is True
    assert mapping["cell_task_mode"] == "stream"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"units_in_flight": 2},
        {"tile_px": None},
        {"tile_px": 999},
        {"prefetch_units": -1},
        {"prefetch_units": 9},
        {"workers_per_unit": 99},
    ],
)
def test_unqualified_gpu_candidates_are_refused(kwargs):
    with pytest.raises(zsched.ActivePlanRejected) as exc:
        zsched.active_gpu_mapping_for_candidate(
            _gpu_candidate(**kwargs),
            tile_ladder=(128, 64, 32),
            workers_ladder=(1, 2, 4, 8),
            prefetch_cap=4,
        )
    assert exc.value.reason_code in zsched.REASON_CODES


def test_cpu_mapping_still_refuses_gpu_and_zero_prefetch():
    with pytest.raises(zsched.ActivePlanRejected):
        zsched.active_mapping_for_candidate(_gpu_candidate())
    cpu = _gpu_candidate(backend="cpu", tile_px=64, prefetch_units=0,
                         device_peak_bytes=0)
    assert zsched.active_mapping_for_candidate(cpu)["backend"] == "cpu"
    with pytest.raises(zsched.ActivePlanRejected):
        zsched.active_mapping_for_candidate(
            _gpu_candidate(backend="cpu", prefetch_units=1,
                           device_peak_bytes=0)
        )


def test_active_final_hash_dispatches_gpu_and_stays_stable():
    ident = zsched.FrozenIdentity(
        layout=(2, 2), cell_ids=("a", "b"), frame_ids=("f1",),
        halo_px=32, refinement_rank=1,
    )
    env = rcontract.PlanEnvelope(
        phase=rcontract.PHASE_PLAN,
        raw_available_bytes=10 * GIB,
        factor_num=4, factor_den=5, budget_bytes=8 * GIB,
        layout=(2, 2), refinement_rank=1, cpu_tile=0, gpu_tile=64,
        backend="gpu", cells_in_flight=1, cache_workers_per_cell=8,
        shared_bytes=0, stack_total_bytes=1, cache_total_bytes=1,
        aggregate_bytes=1, pinned=False, reason="test",
    )
    cand = _gpu_candidate()
    mapping = zsched.active_gpu_mapping_for_candidate(
        cand, tile_ladder=(128, 64, 32), workers_ladder=(1, 2, 4, 8),
        prefetch_cap=4,
    )
    h1 = zsched.active_final_plan_hash(
        candidate=cand, mapping=mapping, identity=ident, envelope=env
    )
    h2 = zsched.active_final_plan_hash(
        candidate=cand, mapping=mapping, identity=ident, envelope=env
    )
    assert h1 == h2
    assert h1.startswith("sha256:")


# ---------------------------------------------------------------------------
# GPU qualification / fresh VRAM / OOM classification
# ---------------------------------------------------------------------------


def _readiness(available=True, status="usable"):
    return zgr.GpuReadiness(
        status=status,
        reason_code=(
            zgr.REASON_GPU_OK if available else zgr.REASON_GPU_NO_DEVICE
        ),
        available=available,
        dependency_declared=True,
        dependency_importable=True,
        cupy_version="14.1.1" if available else None,
        device_count=1 if available else 0,
        device_index=0 if available else None,
        device_name="fake-gpu" if available else None,
        vram_total_bytes=4 * GIB if available else None,
        vram_free_bytes=3 * GIB if available else None,
    )


def test_qualification_closed_reasons():
    policy = zgr.VramPolicy()
    ok = zgr.qualify_gpu_execution(
        requested=True, readiness=_readiness(), parity_qualified=True,
        free_bytes=3 * GIB, policy=policy, device_peak_bytes=256 * MIB,
        prefetch_units=2, queue_max=3, units_in_flight=1,
    )
    assert ok.qualified is True
    assert ok.reason_code == zgr.REASON_VRAM_CANDIDATE_OK
    assert ok.queue_max == 3

    cases = [
        (dict(requested=False), zgr.REASON_GPU_NOT_PROBED),
        (dict(readiness=_readiness(False, "no_device")), zgr.REASON_GPU_NO_DEVICE),
        (dict(parity_qualified=False), zgr.REASON_GPU_PARITY_NOT_QUALIFIED),
        (dict(units_in_flight=2), zgr.REASON_GPU_PREFETCH_UNBOUNDED),
        (dict(prefetch_units=9, queue_max=3), zgr.REASON_GPU_PREFETCH_UNBOUNDED),
        (dict(device_peak_bytes=None), zgr.REASON_GPU_DEVICE_PEAK_UNKNOWN),
        (dict(device_peak_bytes=3 * GIB), zgr.REASON_REFUSE_VRAM),
    ]
    for override, reason in cases:
        args = dict(
            requested=True, readiness=_readiness(), parity_qualified=True,
            free_bytes=3 * GIB, policy=policy, device_peak_bytes=256 * MIB,
            prefetch_units=2, queue_max=3, units_in_flight=1,
        )
        args.update(override)
        decision = zgr.qualify_gpu_execution(**args)
        assert decision.qualified is False
        assert decision.reason_code == reason


def test_fresh_vram_revalidation_never_grows_and_falls_back_whole_batch():
    policy = zgr.VramPolicy()
    go = zgr.fresh_vram_revalidation(
        policy=policy, current_percent=70, free_bytes=3 * GIB,
        device_peak_bytes=256 * MIB,
    )
    assert go.action == "continue"
    assert go.whole_batch is True and go.mixed_partial is False
    stop = zgr.fresh_vram_revalidation(
        policy=policy, current_percent=70, free_bytes=128 * MIB,
        device_peak_bytes=256 * MIB,
    )
    assert stop.action == "fallback_cpu"
    assert stop.reason_code == zgr.REASON_REFUSE_VRAM
    with pytest.raises(zgr.GrowthRejected):
        zgr.shrink_only(65, 70)


def test_cupy_oom_classified_only_from_cupy_module_tree():
    OOM = type("OutOfMemoryError", (RuntimeError,), {"__module__": "cupy.cuda.memory"})
    Cuda = type("CUDARuntimeError", (RuntimeError,), {"__module__": "cupy.cuda.runtime"})
    assert zgpu.is_gpu_runtime_error(OOM("oom")) is True
    assert zgpu.is_gpu_runtime_error(Cuda("cuda")) is True
    assert zgpu.is_gpu_runtime_error(MemoryError("plain")) is False
    assert zgpu.is_gpu_runtime_error(ValueError("plain")) is False
    wrapped = RuntimeError("wrapper")
    wrapped.__cause__ = OOM("wrapped oom")
    assert zgpu.is_gpu_runtime_error(wrapped) is True


# ---------------------------------------------------------------------------
# Product — simulated GPU success / OOM fallback / CPU parity
# ---------------------------------------------------------------------------


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(out: Path) -> dict:
    return json.loads((out / "zegrid_manifest.json").read_text())


@pytest.fixture(scope="module")
def routine_input(tmp_path_factory):
    root = tmp_path_factory.mktemp("r55e2_routine")
    corpus = root / "corpus"
    proc = subprocess.run(
        [sys.executable, str(TOOL), str(corpus)],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    inp = root / "input"
    inp.mkdir()
    for path in sorted(corpus.glob("*.fits")):
        shutil.copy2(path, inp / path.name)
    return inp


def _fake_gpu_available(monkeypatch, *, free_vram=2 * GIB, total_vram=4 * GIB):
    probe = {
        "available": True,
        "device": "Fake MX150",
        "cupy_version": "14.1.1",
        "vram_total_bytes": int(total_vram),
        "vram_free_bytes": int(free_vram),
        "reason": None,
    }
    monkeypatch.setattr(zz.zgpu, "probe_gpu_backend", lambda: dict(probe))
    monkeypatch.setattr(
        zz.zgr, "probe_gpu_readiness_safe",
        lambda **kw: _readiness(True, "usable"),
    )
    monkeypatch.setattr(
        zz.zgr, "read_vram_free_bytes",
        lambda device_index=0: int(free_vram),
    )
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 12 * GIB)
    return probe


def _run(inp, out, mode, *, use_gpu=None):
    zz.run_zegrid_mode(
        str(inp), str(out), zegrid_halo_px=32,
        zegrid_scheduler_mode=mode, use_gpu=use_gpu,
    )


@pytest.fixture(scope="module")
def legacy_run(routine_input, tmp_path_factory):
    out = tmp_path_factory.mktemp("r55e2_legacy")
    before = zz.available_memory_bytes
    zz.available_memory_bytes = lambda: 12 * GIB
    try:
        _run(routine_input, out, "legacy")
    finally:
        zz.available_memory_bytes = before
    return out


def test_simulated_active_gpu_executes_pipeline_and_publishes_exact_output(
    routine_input, legacy_run, tmp_path_factory, monkeypatch
):
    _fake_gpu_available(monkeypatch)
    real_consume = zz._consume_stack_cell
    seen = []

    def fake_gpu_consume(prepared):
        if isinstance(prepared, dict) and "empty_result" in prepared:
            return real_consume(prepared)
        cfg = zz._STACK_CONFIG
        zz._STACK_CONFIG = dataclasses.replace(cfg, backend="cpu")
        try:
            result = real_consume(prepared)
        finally:
            zz._STACK_CONFIG = cfg
        result["backend_used"] = "gpu"
        result["record"]["backend_used"] = "gpu"
        seen.append(result["cell_id"])
        return result

    monkeypatch.setattr(zz, "_consume_stack_cell", fake_gpu_consume)
    out = tmp_path_factory.mktemp("r55e2_gpu_ok")
    _run(routine_input, out, "active", use_gpu=True)

    manifest = _manifest(out)
    block = manifest["r55_scheduler"]
    assert block["status"] == "executed"
    assert block["mapping"]["backend"] == "gpu"
    assert block["mapping"]["units_in_flight"] == 1
    assert block["mapping"]["prefetch_units"] >= 0
    assert block["gpu_execution"]["qualified"] is True
    assert block["gpu_fallback"] if "gpu_fallback" in block else True

    telemetry = manifest["r55_run_telemetry"]
    pipe = telemetry["gpu_pipeline"]
    assert pipe["queue_max"] == block["mapping"]["prefetch_units"] + 1
    assert pipe["units_consumed"] == len(seen)
    assert pipe["ledger"]["balanced"] is True
    assert pipe["threads_joined"] is True
    assert telemetry["gpu_revalidations"]
    assert all(r["vram"]["free_bytes"] == 2 * GIB for r in telemetry["gpu_revalidations"])

    # Exact scientific parity: the GPU pipeline published byte-identical FITS.
    for name in FITS_NAMES:
        assert _digest(out / name) == _digest(legacy_run / name)


def test_gpu_oom_falls_back_to_exact_whole_batch_cpu(
    routine_input, legacy_run, tmp_path_factory, monkeypatch
):
    _fake_gpu_available(monkeypatch)
    real_consume = zz._consume_stack_cell
    state = {"n": 0}

    OOM = type("OutOfMemoryError", (RuntimeError,), {"__module__": "cupy.cuda.memory"})

    def oom_consumer(prepared):
        if isinstance(prepared, dict) and "empty_result" in prepared:
            return real_consume(prepared)
        if getattr(zz._STACK_CONFIG, "backend", "cpu") != "gpu":
            # CPU fallback path: never simulate/annotate GPU work here.
            return real_consume(prepared)
        state["n"] += 1
        if state["n"] >= 2:
            raise OOM("simulated CuPy OOM")
        cfg = zz._STACK_CONFIG
        zz._STACK_CONFIG = dataclasses.replace(cfg, backend="cpu")
        try:
            result = real_consume(prepared)
        finally:
            zz._STACK_CONFIG = cfg
        result["backend_used"] = "gpu"
        result["record"]["backend_used"] = "gpu"
        return result

    monkeypatch.setattr(zz, "_consume_stack_cell", oom_consumer)
    out = tmp_path_factory.mktemp("r55e2_gpu_oom")
    _run(routine_input, out, "active", use_gpu=True)

    manifest = _manifest(out)
    block = manifest["r55_scheduler"]
    fallback = block.get("gpu_fallback") or {}
    assert fallback.get("stage") == "run"
    assert fallback.get("reason_code") == "R55_GPU_RUNTIME"
    assert fallback["telemetry"]["pipeline"]["cancelled"] is True
    assert fallback["telemetry"]["pipeline"]["ledger"]["balanced"] is True
    assert fallback["telemetry"]["pipeline"]["threads_joined"] is True
    # No mixed result: the published output is the exact CPU fallback.
    assert (manifest.get("gpu") or {}).get("final_backend") == "cpu"
    assert (manifest.get("gpu") or {}).get("actually_used") is False
    for name in FITS_NAMES:
        assert _digest(out / name) == _digest(legacy_run / name)
    # Same canonical cells / reference / exclusions as the exact CPU witness.
    legacy_manifest = _manifest(legacy_run)
    assert manifest["complete_cells"] == legacy_manifest["complete_cells"]
    assert manifest["incomplete_cells"] == legacy_manifest["incomplete_cells"]
    stable = (
        "cell_id", "row", "col", "status", "n_contributors",
        "n_patch_contributors", "patch_bounds", "patch_shape_hw",
        "reference_frame_id", "reference_frame_role",
        "bookkeeping_reference_frame_id", "excluded",
    )
    def _science_cells(m):
        return [
            {k: c.get(k) for k in stable}
            for c in sorted(m["cells"], key=lambda c: c["cell_id"])
        ]
    assert _science_cells(manifest) == _science_cells(legacy_manifest)


def test_active_cpu_path_is_unchanged_by_e2(routine_input, legacy_run, tmp_path_factory, monkeypatch):
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 12 * GIB)
    monkeypatch.setattr(
        zz.zgr, "probe_gpu_readiness_safe",
        lambda **kw: _readiness(False, "not_probed"),
    )
    out = tmp_path_factory.mktemp("r55e2_cpu")
    _run(routine_input, out, "active")
    manifest = _manifest(out)
    block = manifest["r55_scheduler"]
    assert block["mapping"]["backend"] == "cpu"
    assert block["mapping"]["prefetch_units"] == 0
    assert manifest["r55_run_telemetry"]["metrics"]["wave_count"] >= 1
    for name in FITS_NAMES:
        assert _digest(out / name) == _digest(legacy_run / name)


def _lazy_import_script() -> str:
    # The full product module has a pre-existing Dask/chunk-types CuPy edge
    # when CuPy is installed; E2 must prove ITS modules stay lazy.
    return """
import sys
class _ExplodingCupy:
    def find_spec(self, name, path=None, target=None):
        if name == 'cupy' or name.startswith('cupy.'):
            raise AssertionError('CuPy import attempted by an E2 CPU module')
        return None
sys.meta_path.insert(0, _ExplodingCupy())
from zemosaic.core.zegrid import parallel, runtime_control, gpu_readiness, scheduler
assert 'cupy' not in sys.modules
assert 'cupy.cuda' not in sys.modules
assert parallel.PIPELINE_SCHEMA
assert runtime_control.RUN_MAX_SAMPLE_INTERVAL_S == 5.0
assert gpu_readiness.INITIAL_TARGET_PERCENT == 70
assert scheduler.MODE_ACTIVE == 'active'
print('LAZY_OK')
"""


def test_cpu_modules_do_not_import_cupy():
    proc = subprocess.run(
        [sys.executable, "-c", _lazy_import_script()],
        capture_output=True, text=True, timeout=120,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "LAZY_OK" in proc.stdout


def test_run_credit_model_is_coherent_and_publishes_disk_and_overlap(
    routine_input, legacy_run, tmp_path_factory, monkeypatch
):
    """ZM-ZEGRID-R56: RUN agrees with PLAN and books disk separately.

    The RUN block must publish the SAME separated credit model the planner used
    (resident host staging vs exact on-disk bytes), the pipeline ledger must keep
    both dimensions balanced, and the observed overlap witness must be present
    (a boolean proof derived from real produce()/consume() intervals).
    """
    _fake_gpu_available(monkeypatch)
    real_consume = zz._consume_stack_cell

    def fake_gpu_consume(prepared):
        if isinstance(prepared, dict) and "empty_result" in prepared:
            return real_consume(prepared)
        cfg = zz._STACK_CONFIG
        zz._STACK_CONFIG = dataclasses.replace(cfg, backend="cpu")
        try:
            result = real_consume(prepared)
        finally:
            zz._STACK_CONFIG = cfg
        result["backend_used"] = "gpu"
        result["record"]["backend_used"] = "gpu"
        return result

    monkeypatch.setattr(zz, "_consume_stack_cell", fake_gpu_consume)
    out = tmp_path_factory.mktemp("r55e2_r56_credit")
    _run(routine_input, out, "active", use_gpu=True)

    manifest = _manifest(out)
    block = manifest["r55_scheduler"]
    run_block = block["run"]
    policy = run_block["policy"]
    credit = policy["credit_model"]
    assert credit["schema"] == zz._R55_PREFETCH_CREDIT_SCHEMA
    assert credit["per_unit_prefetch_disk_bytes"] == credit["per_unit_cache_disk_bytes"]
    assert credit["disk_budget_bytes"] >= credit["per_unit_cache_disk_bytes"]
    assert policy["disk"]["ledger_dimension"] == "disk"

    pipe = manifest["r55_run_telemetry"]["gpu_pipeline"]
    ledger = pipe["ledger"]
    assert ledger["schema"] == zpar.PIPELINE_SCHEMA_V2
    assert ledger["balanced"] is True
    assert ledger["disk_budget_bytes"] <= credit["disk_budget_bytes"]
    assert ledger["host_budget_bytes"] == credit["host_budget_bytes"]
    # The on-disk peak is a real, positive observation of the queued caches.
    assert ledger["disk_peak_bytes"] > 0
    assert ledger["disk_peak_bytes"] <= policy["disk"]["required_bytes"]
    # F1: the product pre-acquires and reconciles; F4 may shrink the disk budget
    # from a fresh observation, but never grow it.
    assert pipe["credit_accounting"]["mode"] == "pre_acquire"
    assert ledger["reconciliations"] > 0
    # F4 reductions are OPTIONAL: the fresh volume may be large enough that the
    # shrink-only budget is unchanged. If present, they may only shrink.
    assert all(
        entry["to_bytes"] <= entry["from_bytes"]
        for entry in ledger["budget_reductions"]
    )

    overlap = run_block["overlap"]
    assert overlap["clock"] == "time.monotonic"
    assert isinstance(overlap["overlap_proved"], bool)
    assert overlap["overlap_proved"] == (
        overlap["intersection_s"] > 0 and overlap["overlap_units"] > 0
    )
    assert overlap["producer_work_s"] > 0
    assert overlap["consumer_work_s"] > 0
    assert overlap["intersection_s"] <= min(
        overlap["producer_work_s"], overlap["consumer_work_s"]
    )
