import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import parallel_utils as pu
import zemosaic_gpu_safety as gs

MB = 1024**2
GB = 1024**3


def _make_hybrid_ctx() -> gs.GpuRuntimeContext:
    return gs.GpuRuntimeContext(
        os_name="windows",
        platform_system="Windows",
        gpu_available=True,
        gpu_name="Test GPU",
        gpu_vendor="nvidia",
        vram_total_bytes=8 * GB,
        vram_free_bytes=6 * GB,
        has_battery=True,
        power_plugged=True,
        on_battery=False,
        is_windows=True,
        is_hybrid_graphics=True,
        safe_mode=False,
        reasons=[],
    )


def _make_plan() -> pu.ParallelPlan:
    return pu.ParallelPlan(
        cpu_workers=4,
        use_memmap=False,
        max_chunk_bytes=512 * MB,
        rows_per_chunk=256,
        use_gpu=True,
        gpu_rows_per_chunk=512,
        gpu_max_chunk_bytes=0,
    )


def test_hybrid_guard_on_clamps(monkeypatch):
    ctx = _make_hybrid_ctx()
    monkeypatch.setattr(gs, "probe_gpu_runtime_context", lambda *args, **kwargs: ctx)
    plan = _make_plan()

    clamped_plan, clamped_ctx = gs.apply_gpu_safety_to_parallel_plan(
        plan,
        None,
        {"gpu_hybrid_vram_guard": True},
        operation="test_hybrid_guard_on",
    )

    assert clamped_plan is plan
    assert clamped_plan.gpu_max_chunk_bytes <= 256 * MB
    assert "hybrid_vram_guard" in clamped_ctx.reasons


def test_hybrid_guard_off_no_clamp(monkeypatch):
    ctx = _make_hybrid_ctx()
    monkeypatch.setattr(gs, "probe_gpu_runtime_context", lambda *args, **kwargs: ctx)
    plan = _make_plan()
    original_bytes = plan.gpu_max_chunk_bytes

    clamped_plan, clamped_ctx = gs.apply_gpu_safety_to_parallel_plan(
        plan,
        None,
        {"gpu_hybrid_vram_guard": False},
        operation="test_hybrid_guard_off",
    )

    assert clamped_plan is plan
    assert clamped_plan.gpu_max_chunk_bytes == original_bytes
    assert "hybrid_vram_guard_disabled_by_user" in clamped_ctx.reasons
