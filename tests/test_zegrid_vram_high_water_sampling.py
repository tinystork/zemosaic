"""ZM-WIN-VRAM-HW — sampled VRAM high-water during ``per_cell_stack``.

CPU-only regressions (no GPU, no CuPy) for the bounded sampler, its honest
cadence degradation on the ``nvidia-smi`` path, the MAX-not-last contract, the
guaranteed stop on success/exception/cancellation, the "GPU OFF -> null +
reason" record and the observer-only wiring in the ZeGrid engine.
"""

from __future__ import annotations

import ast
import threading
import time
from pathlib import Path

import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic import zemosaic_resource_telemetry as rt
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import resource_envelope as re
from zemosaic.core.zegrid import vram_high_water as zvh


@pytest.fixture(autouse=True)
def _clean_registry():
    zvh.stop_all_active_vram_samplers(reason="test_reset")
    yield
    zvh.stop_all_active_vram_samplers(reason="test_teardown")


def _probe_sequence(values, source="ctypes_nvml"):
    """Probe returning ``values`` in order, then repeating the last one."""
    state = {"i": 0}

    def _probe():
        i = min(state["i"], len(values) - 1)
        state["i"] += 1
        value = values[i]
        if value is None:
            return {
                "gpu_used_mb": None,
                "gpu_total_mb": None,
                "gpu_sample_source": None,
                "gpu_unavailable_reason": "all_sources_failed",
            }
        return {
            "gpu_used_mb": float(value),
            "gpu_total_mb": 8192.0,
            "gpu_sample_source": source,
            "gpu_unavailable_reason": None,
        }

    return _probe


def _wait_for_samples(sampler, count, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sampler.snapshot()["vram_high_water_sample_count"] >= count:
            return
        time.sleep(0.01)
    raise AssertionError(
        f"sampler did not reach {count} samples: {sampler.snapshot()}"
    )


def test_sampler_keeps_max_not_last_value():
    sampler = zvh.VramHighWaterSampler(
        interval_sec=0.5, probe=_probe_sequence([100.0, 700.0, 300.0])
    )
    delays: list[float] = []
    sampler._wait = lambda delay: (delays.append(delay) or False) if len(delays) < 2 else True
    sampler.start()
    _wait_for_samples(sampler, 3)
    sampler.stop("test")
    record = sampler.snapshot()

    assert record["vram_high_water_sampled_mb"] == 700.0  # MAX, not 300 (last)
    assert record["vram_high_water_last_sample_mb"] == 300.0
    assert record["vram_high_water_sample_count"] == 3
    assert record["vram_high_water_available"] is True
    assert record["label"] == "sampled_high_water_not_driver_counter"


def test_cadence_uses_requested_interval_on_nvml_path():
    sampler = zvh.VramHighWaterSampler(
        interval_sec=0.5, probe=_probe_sequence([10.0, 20.0, 30.0])
    )
    delays: list[float] = []
    sampler._wait = lambda delay: (delays.append(delay) or False) if len(delays) < 2 else True
    sampler.start()
    _wait_for_samples(sampler, 3)
    sampler.stop("test")

    assert delays and all(d == 0.5 for d in delays), delays
    record = sampler.snapshot()
    assert record["vram_high_water_interval_s"] == 0.5
    assert record["vram_high_water_requested_interval_s"] == 0.5
    assert record["vram_high_water_effective_cadence_degraded"] is False


def test_nvidia_smi_only_degrades_effective_cadence_to_ttl():
    sampler = zvh.VramHighWaterSampler(
        interval_sec=0.5,
        probe=_probe_sequence([123.0, 456.0], source="nvidia_smi"),
    )
    delays: list[float] = []
    sampler._wait = lambda delay: (delays.append(delay) or False) if len(delays) < 1 else True
    sampler.start()
    _wait_for_samples(sampler, 1)
    sampler.stop("test")
    record = sampler.snapshot()

    # The 5 s TTL cache would make a 0.5 s loop repeat stale values. The sampler
    # waits >= 5 s and SAYS SO instead of claiming 0.5 s.
    assert delays and delays[0] >= zvh.NVIDIA_SMI_TTL_S
    assert record["vram_high_water_interval_s"] >= zvh.NVIDIA_SMI_TTL_S
    assert record["vram_high_water_requested_interval_s"] == 0.5
    assert record["vram_high_water_effective_cadence_degraded"] is True
    assert (
        record["vram_high_water_effective_cadence_note"]
        == "effective_cadence_limited_by_nvidia_smi_ttl_cache"
    )


def test_probe_failure_is_best_effort_and_never_raises():
    def _boom():
        raise RuntimeError("driver exploded")

    sampler = zvh.VramHighWaterSampler(interval_sec=0.5, probe=_boom)
    sampler._wait = lambda delay: True
    sampler.start()
    record = sampler.stop("test")  # must not raise

    assert record["vram_high_water_available"] is False
    assert record["vram_high_water_sampled_mb"] is None
    assert record["vram_high_water_sample_count"] == 0
    assert any("probe_exception" in e for e in record["vram_high_water_sample_errors"])


def test_disabled_record_is_null_plus_reason():
    record = zvh.disabled_record("gpu_disabled")
    assert record["vram_high_water_sampled_mb"] is None
    assert record["vram_high_water_sample_count"] == 0
    assert record["vram_high_water_interval_s"] is None
    assert record["vram_high_water_unavailable_reason"] == "gpu_disabled"
    assert record["label"] == "sampled_high_water_not_driver_counter"


def test_gpu_off_and_telemetry_off_reasons_are_explicit():
    class _Tel:
        interval_sec = 1.5

    assert (
        zz._vram_high_water_disabled_reason(None, True, {"requested": True})
        == "resource_telemetry_disabled"
    )
    assert (
        zz._vram_high_water_disabled_reason(_Tel(), False, {"requested": False})
        == "gpu_disabled"
    )
    assert (
        zz._vram_high_water_disabled_reason(_Tel(), False, {"requested": True})
        == "gpu_not_effective"
    )
    assert zz._vram_high_water_disabled_reason(_Tel(), True, {}) is None


def test_stop_on_exception_and_cancellation_via_registry():
    for exc_type in (RuntimeError, KeyboardInterrupt):
        sampler = zvh.start_vram_high_water_sampler(
            interval_sec=0.5, probe=_probe_sequence([10.0])
        )
        assert zvh.active_sampler_count() >= 1
        try:
            try:
                raise exc_type("boom")
            finally:
                zvh.stop_all_active_vram_samplers(reason="run_exit")
        except exc_type:
            pass
        assert sampler.snapshot()["vram_high_water_stopped"] is True
        assert zvh.active_sampler_count() == 0


def test_stop_vram_high_water_sampler_removes_registry_entry():
    sampler = zvh.start_vram_high_water_sampler(
        interval_sec=0.5, probe=_probe_sequence([5.0])
    )
    record = zvh.stop_vram_high_water_sampler(sampler, reason="success")
    assert record is not None
    assert record["vram_high_water_stop_reason"] == "success"
    assert zvh.active_sampler_count() == 0


def test_snapshot_is_thread_safe_during_sampling():
    sampler = zvh.VramHighWaterSampler(
        interval_sec=0.5, probe=_probe_sequence([1.0, 2.0, 3.0, 4.0])
    )
    sampler.start()
    records = []
    threads = [
        threading.Thread(target=lambda: records.append(sampler.snapshot()))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    final = sampler.stop("test")
    assert len(records) == 8
    assert final["vram_high_water_sample_count"] >= 1


def test_engine_wires_sampler_inside_per_cell_stack_and_publishes():
    source = Path(zz.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # `_finish_vram_high_water_record` is called after the stack wall clock.
    text_after_block = source.split("block_wall = time.perf_counter() - t_block0", 1)[1]
    assert "_finish_vram_high_water_record(" in text_after_block
    assert 'per_cell_diagnostics["vram_high_water"]' in text_after_block

    # The run-level guard stops every registered sampler in a `finally`.
    guarded = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_run_single_guarded"
    )
    try_nodes = [node for node in ast.walk(guarded) if isinstance(node, ast.Try)]
    assert try_nodes, "_run_single_guarded must wrap _run_single in try/finally"
    attrs = {n.attr for node in try_nodes for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    assert "stop_all_active_vram_samplers" in attrs


def test_telemetry_fields_and_reservation_are_observer_only():
    class _Decision:
        device_peak_bytes_by_tile = {448: 4 * 1024**3 + 100 * 1024**2}

    reservation = zz._vram_reservation_bytes_from_plan(
        None, _Decision(), {"gpu_streaming_tile": 448}, None
    )
    assert reservation == 4 * 1024**3 + 100 * 1024**2

    record = {
        "label": "sampled_high_water_not_driver_counter",
        "vram_high_water_sampled_mb": 2200.0,
        "vram_high_water_sample_count": 42,
        "vram_high_water_interval_s": 1.5,
        "vram_high_water_source": "ctypes_nvml",
        "vram_high_water_effective_cadence_degraded": False,
        "vram_high_water_unavailable_reason": None,
    }
    fields = zz._vram_high_water_telemetry_fields(record, reservation)
    assert fields["vram_high_water_sampled_mb"] == 2200.0
    assert fields["vram_high_water_sample_count"] == 42
    assert fields["vram_high_water_interval_s"] == 1.5
    assert fields["vram_reservation_mb"] == pytest.approx(4196.0)


def test_telemetry_on_with_sampler_changes_neither_plan_nor_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The observer must not depend on this machine having a GPU.
    monkeypatch.setattr(
        rt, "_sample_runtime_resources_for_telemetry", lambda: {}
    )
    bounds = [(400, 300), (500, 350), (450, 320)]
    plan_before = zpar.plan_cell_concurrency(
        bounds, cpu=8, ram_budget_bytes=1_000_000
    )
    envelope_before = _envelope()

    events = []
    controller = rt.ResourceTelemetryController(
        enabled=True,
        interval_sec=0.5,
        callback=lambda *a, **k: events.append((a, k)),
        csv_path=str(tmp_path / "telemetry.csv"),
    )
    sampler = zvh.start_vram_high_water_sampler(
        interval_sec=1.5,
        probe=_probe_sequence([100.0, 900.0]),
        on_sample=lambda rec: controller.maybe_emit_stats(
            {
                "phase_name": "Per-cell stacking",
                **zz._vram_high_water_telemetry_fields(rec, 4 * 1024**3),
            }
        ),
    )
    _wait_for_samples(sampler, 1)
    sampler.stop("test")
    controller.close()

    plan_after = zpar.plan_cell_concurrency(
        bounds, cpu=8, ram_budget_bytes=1_000_000
    )
    assert repr(plan_before) == repr(plan_after)
    assert repr(envelope_before) == repr(_envelope())
    assert events, "the telemetry observer must actually have emitted"
    assert events[0][0][0] == "STATS_UPDATE"


def _envelope() -> re.ResourceEnvelope:
    estimate = re.make_estimate({"stack_bytes": 300})
    return re.ResourceEnvelope(
        envelope_id="cell_stack_stream.cpu",
        phase="cell_stack_stream",
        backend="cpu",
        host=re.HostEnvelope(
            observation_source="test.available_bytes",
            raw_available_bytes=1000,
            factor_num=4,
            factor_den=5,
            budget_bytes=800,
            reserve_bytes=0,
            process_baseline_bytes=100,
        ),
        device=None,
        geometry=re.GeometryEnvelope(
            layout=(2, 2),
            refinement_rank=3,
            cell_id=None,
            patch_hw=(100, 100),
            tile_px=64,
            halo_px=32,
            channels=3,
        ),
        concurrency=re.ConcurrencyEnvelope(
            units=2, units_in_flight=1, workers_per_unit=1, prefetch_units=0
        ),
        estimate=estimate,
        bound=re.BoundEnvelope(
            kind="upper", margin_num=15, margin_den=100, confidence="static_bound"
        ),
        accounting=re.AccountingEnvelope(
            aggregate_rule=re.AGGREGATE_RULE_MAX_OVER_PHASES,
            host_peak_bytes=estimate.peak_bytes,
            device_peak_bytes=0,
            disk_bytes=0,
            transfer_bytes=0,
        ),
        overlaps_with=(),
        identity=re.IdentityEnvelope(run_id="run", plan_hash="", pinned=False),
        reason_code=re.REASON_UNCALIBRATED,
    )


# ---------------------------------------------------------------------------
# Static guard: the new sampler path must NEVER touch CuPy / CUDA.
# ---------------------------------------------------------------------------

def test_vram_high_water_module_never_touches_cupy():
    source = Path(zvh.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            if getattr(node, "module", None):
                names.append(node.module)
            assert all("cupy" not in n.lower() for n in names)
        if isinstance(node, ast.Name):
            assert "cupy" not in node.id.lower()
        if isinstance(node, ast.Attribute):
            assert "cupy" not in node.attr.lower()
            assert node.attr != "memGetInfo"


def test_engine_high_water_helpers_never_touch_cupy():
    source = Path(zz.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {
        "_vram_reservation_bytes_from_plan",
        "_vram_high_water_disabled_reason",
        "_vram_high_water_telemetry_fields",
        "_finish_vram_high_water_record",
        "_run_single_guarded",
    }
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    }
    assert set(functions) == wanted
    for name, node in functions.items():
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in child.names]
                if getattr(child, "module", None):
                    names.append(child.module)
                assert all("cupy" not in n.lower() for n in names), name
            if isinstance(child, ast.Name):
                assert "cupy" not in child.id.lower(), name
            if isinstance(child, ast.Attribute):
                assert "cupy" not in child.attr.lower(), name
                assert child.attr != "memGetInfo", name
