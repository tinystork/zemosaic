"""ZM-WIN-GPUPROBE — CUDA-free breadcrumb GPU probe fallbacks.

The Windows product run showed a fully measurable VRAM (nvidia-smi answered)
while the breadcrumb published ``gpu_*: null`` because only the optional
``pynvml`` binding was tried. These CPU-only regressions pin the ordered source
cascade, its explicit aggregated failure reason, the ``nvidia-smi`` TTL cache,
the "never touch CuPy from the heartbeat" contract, and the fact that
telemetry being ON cannot change a plan or a resource envelope.
"""

from __future__ import annotations

import ast
import json
import subprocess
import types
from pathlib import Path

import pytest

from zemosaic import zemosaic_resource_telemetry as rt
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import resource_envelope as re


@pytest.fixture(autouse=True)
def _clean_probe_state():
    rt.reset_gpu_probe_caches()
    yield
    rt.reset_gpu_probe_caches()


# ---------------------------------------------------------------------------
# Fake driver binding (no GPU required)
# ---------------------------------------------------------------------------

class _FakeBuffer:
    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.value = b""


class _FakePtr:
    def __init__(self) -> None:
        self.value = None


class _FakeStructure:
    _fields_: tuple = ()


def _fake_ctypes(loader):
    return types.SimpleNamespace(
        Structure=_FakeStructure,
        c_int=int,
        c_uint=int,
        c_ulonglong=int,
        c_void_p=_FakePtr,
        POINTER=lambda _t: object,
        byref=lambda obj: obj,
        create_string_buffer=_FakeBuffer,
        CDLL=loader,
        WinDLL=loader,
    )


def _fake_nvml_lib(
    *,
    total_mib: int = 8192,
    used_mib: int = 329,
    free_mib: int = 7692,
    name: bytes = b"NVIDIA GeForce GTX 1070",
    init_rc: int = 0,
    mem_rc: int = 0,
):
    calls = {"init": 0, "shutdown": 0, "handle": 0, "name": 0}
    lib = types.SimpleNamespace()

    def nvml_init():
        calls["init"] += 1
        return init_rc

    def get_handle(index, out):
        calls["handle"] += 1
        out.value = 1
        return 0

    def memory_info(handle, out):
        mib = 1024 * 1024
        out.total = int(total_mib * mib)
        out.used = int(used_mib * mib)
        out.free = int(free_mib * mib)
        return mem_rc

    def get_name(handle, buf, size):
        calls["name"] += 1
        buf.value = name
        return 0

    def shutdown():
        calls["shutdown"] += 1
        return 0

    lib.nvmlInit_v2 = nvml_init
    lib.nvmlDeviceGetHandleByIndex_v2 = get_handle
    lib.nvmlDeviceGetMemoryInfo = memory_info
    lib.nvmlDeviceGetName = get_name
    lib.nvmlShutdown = shutdown
    return lib, calls


def _pynvml_down(monkeypatch: pytest.MonkeyPatch, token: str = "pynvml_unavailable"):
    monkeypatch.setattr(rt, "_sample_gpu_via_pynvml", lambda: rt._gpu_info_defaults())
    monkeypatch.setattr(rt, "_pynvml_error_token", lambda: token)


def _smi_down(monkeypatch: pytest.MonkeyPatch, *, mode: str = "timeout"):
    monkeypatch.setattr(rt, "_nvidia_smi_executable", lambda: "/usr/bin/nvidia-smi")

    def fake_run(cmd, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(
        rt,
        "subprocess",
        types.SimpleNamespace(run=fake_run, TimeoutExpired=subprocess.TimeoutExpired),
    )


# ---------------------------------------------------------------------------
# Source cascade
# ---------------------------------------------------------------------------

def test_probe_prefers_pynvml_when_it_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rt,
        "_sample_gpu_via_pynvml",
        lambda: dict(
            rt._gpu_info_defaults(),
            gpu_used_mb=10.0,
            gpu_total_mb=100.0,
            gpu_free_mb=90.0,
            gpu_backend="nvml",
            gpu_sample_source="pynvml",
        ),
    )

    def _must_not_run(*_a, **_k):
        raise AssertionError("lower-priority GPU sources must not run")

    monkeypatch.setattr(rt, "_sample_gpu_via_ctypes_nvml", _must_not_run)
    monkeypatch.setattr(rt, "_sample_gpu_via_nvidia_smi", _must_not_run)

    info = rt.sample_gpu_via_nvml()

    assert info["gpu_source"] == "pynvml"
    assert info["gpu_used_mb"] == pytest.approx(10.0)
    assert info["gpu_total_mb"] == pytest.approx(100.0)
    assert info["gpu_unavailable_reason"] is None


def test_probe_falls_back_to_ctypes_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    _pynvml_down(monkeypatch)
    lib, calls = _fake_nvml_lib()
    monkeypatch.setattr(rt, "_CTYPES", _fake_ctypes(lambda _name: lib))

    info = rt.sample_gpu_via_nvml()

    assert info["gpu_source"] == "ctypes_nvml"
    assert info["gpu_backend"] == "nvml"
    assert info["gpu_used_mb"] == pytest.approx(329.0)
    assert info["gpu_total_mb"] == pytest.approx(8192.0)
    assert info["gpu_free_mb"] == pytest.approx(7692.0)
    assert info["gpu_name"] == "NVIDIA GeForce GTX 1070"
    assert info["gpu_unavailable_reason"] is None
    # The driver handle is released (no leaked NVML reference).
    assert calls["init"] == 1
    assert calls["shutdown"] == 1


def test_ctypes_nvml_dll_failure_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    _pynvml_down(monkeypatch)

    def _no_library(name):
        raise OSError(f"cannot load {name}")

    monkeypatch.setattr(rt, "_CTYPES", _fake_ctypes(_no_library))
    _smi_down(monkeypatch, mode="missing")

    info = rt.sample_gpu_via_nvml()

    assert info["gpu_total_mb"] is None
    assert "ctypes_nvml_failed:DLL" in info["gpu_unavailable_reason"]


def test_probe_falls_back_to_nvidia_smi_with_ttl_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pynvml_down(monkeypatch)
    monkeypatch.setattr(rt, "_load_nvml_library", lambda: (None, "ctypes_nvml_failed:DLL"))
    monkeypatch.setattr(rt, "_nvidia_smi_executable", lambda: "/usr/bin/nvidia-smi")
    clock = {"now": 100.0}
    monkeypatch.setattr(rt, "_MONOTONIC", lambda: clock["now"])
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        assert cmd[0] == "/usr/bin/nvidia-smi"
        assert "--query-gpu=memory.used,memory.total,memory.free" in cmd
        assert "--format=csv,noheader,nounits" in cmd
        assert kwargs["timeout"] == rt._NVIDIA_SMI_TIMEOUT_S
        return types.SimpleNamespace(returncode=0, stdout="329, 8192, 7692\n")

    monkeypatch.setattr(
        rt,
        "subprocess",
        types.SimpleNamespace(run=fake_run, TimeoutExpired=subprocess.TimeoutExpired),
    )

    first = rt.sample_gpu_via_nvml()
    second = rt.sample_gpu_via_nvml()

    assert first["gpu_source"] == "nvidia_smi"
    assert first["gpu_used_mb"] == pytest.approx(329.0)
    assert first["gpu_total_mb"] == pytest.approx(8192.0)
    assert first["gpu_free_mb"] == pytest.approx(7692.0)
    assert second["gpu_source"] == "nvidia_smi"
    assert calls["n"] == 1  # TTL: a 2 s heartbeat must not spawn every tick

    clock["now"] += rt._NVIDIA_SMI_TTL_S + 0.1
    rt.sample_gpu_via_nvml()
    assert calls["n"] == 2


def test_all_sources_unavailable_is_null_with_aggregated_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pynvml_down(monkeypatch)
    monkeypatch.setattr(rt, "_load_nvml_library", lambda: (None, "ctypes_nvml_failed:DLL"))
    _smi_down(monkeypatch, mode="timeout")

    info = rt.sample_gpu_via_nvml()

    for key in ("gpu_used_mb", "gpu_total_mb", "gpu_free_mb", "gpu_source"):
        assert info[key] is None
    assert info["gpu_unavailable_reason"] == (
        "pynvml_unavailable;ctypes_nvml_failed:DLL;nvidia_smi_timeout"
    )


def test_probe_never_touches_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom():
        raise AssertionError("CuPy/CUDA must never be touched by the breadcrumb probe")

    monkeypatch.setattr(rt, "_sample_gpu_via_cupy", _boom)
    _pynvml_down(monkeypatch)
    monkeypatch.setattr(rt, "_load_nvml_library", lambda: (None, "ctypes_nvml_failed:DLL"))
    _smi_down(monkeypatch, mode="missing")

    info = rt.sample_gpu_via_nvml()
    assert info["gpu_total_mb"] is None
    assert "gpu_probe" not in str(info.get("gpu_unavailable_reason"))

    # Static guard: the breadcrumb-probe functions must contain NO CuPy code
    # (their docstrings may *say* CuPy is forbidden; that is not a call).
    source = Path(rt.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {
        "sample_gpu_via_nvml",
        "_sample_gpu_via_pynvml",
        "_sample_gpu_via_ctypes_nvml",
        "_sample_gpu_via_nvidia_smi",
        "_query_nvidia_smi",
        "_load_nvml_library",
        "_nvml_device_name",
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


# ---------------------------------------------------------------------------
# in_flight / winsor_workers publication (observability only)
# ---------------------------------------------------------------------------

def test_zegrid_publishes_single_owner_units_in_flight() -> None:
    from zemosaic import zemosaic_zegrid_mode as zz

    zz.publish_runtime_concurrency(None, source=None, unavailable_reason="test_reset")
    zz.publish_runtime_concurrency(1, source="active_gpu_single_owner")
    snap = zz.runtime_concurrency_snapshot()
    assert snap["units_in_flight"] == 1
    assert snap["source"] == "active_gpu_single_owner"
    assert snap["unavailable_reason"] is None

    zz.publish_runtime_concurrency(
        None,
        source="active_gpu_single_owner",
        unavailable_reason="per_cell_stack_finished",
    )
    snap = zz.runtime_concurrency_snapshot()
    assert snap["units_in_flight"] is None
    assert snap["unavailable_reason"] == "per_cell_stack_finished"


def test_worker_bridge_uses_the_published_concurrency(monkeypatch) -> None:
    import sys

    from zemosaic import zemosaic_worker as zw

    published = {
        "units_in_flight": 1,
        "source": "active_gpu_single_owner",
        "unavailable_reason": None,
    }
    fake_engine = types.SimpleNamespace(
        runtime_concurrency_snapshot=lambda: dict(published)
    )
    monkeypatch.setitem(sys.modules, "zemosaic.zemosaic_zegrid_mode", fake_engine)
    assert zw._zegrid_runtime_concurrency() == published

    monkeypatch.delitem(sys.modules, "zemosaic.zemosaic_zegrid_mode", raising=False)
    assert zw._zegrid_runtime_concurrency() is None


# ---------------------------------------------------------------------------
# Telemetry ON must not change the plan or the envelope
# ---------------------------------------------------------------------------

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


def _run_with_telemetry(monkeypatch, tmp_path: Path, enabled: bool):
    monkeypatch.setattr(zpar, "cells_in_flight", lambda *_a, **_k: 2)
    bounds = [(400, 300), (500, 350), (450, 320)]
    events = []
    controller = rt.ResourceTelemetryController(
        enabled=enabled,
        interval_sec=0.5,
        callback=lambda *a, **k: events.append((a, k)),
        csv_path=str(tmp_path / f"telemetry_{enabled}.csv"),
    )
    plan = zpar.plan_cell_concurrency(bounds, cpu=8, ram_budget_bytes=1_000_000)
    envelope = _envelope()
    context = {
        "phase_name": "Per-cell stacking",
        "tiles_done": 2,
        "tiles_total": 3,
    }
    controller.maybe_emit_stats(context)
    controller.close()
    return plan, envelope, events


def test_telemetry_on_changes_neither_plan_nor_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    off_plan, off_env, off_events = _run_with_telemetry(monkeypatch, tmp_path, False)
    on_plan, on_env, on_events = _run_with_telemetry(monkeypatch, tmp_path, True)

    # The observer really was live only in the ON run.
    assert off_events == []
    assert len(on_events) == 1
    assert on_events[0][0][0] == "STATS_UPDATE"
    assert (tmp_path / "telemetry_True.csv").exists()
    assert not (tmp_path / "telemetry_False.csv").exists()

    # ... and it changed NOTHING about the frozen decision/envelope.
    assert json.dumps(off_plan, sort_keys=True) == json.dumps(on_plan, sort_keys=True)
    assert json.dumps(off_env.to_dict(), sort_keys=True) == json.dumps(
        on_env.to_dict(), sort_keys=True
    )
    assert off_env.envelope_id == on_env.envelope_id
