import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import parallel_utils as pu

MB = 1024**2
GB = 1024**3
CAPS = pu.detect_parallel_capabilities()

# Tests focus on relative invariants derived from the detected machine capabilities
# so they stay portable across hosts with different RAM/VRAM sizes.


def _bytes_for_frames(height: int, width: int, n_frames: int, bpp: int) -> int:
    return height * width * n_frames * bpp


def test_auto_tune_respects_cpu_limits():
    caps = CAPS
    cfg = {
        "parallel_autotune_enabled": True,
        "parallel_target_cpu_load": 0.5,
        "parallel_max_cpu_workers": 4,
        "parallel_target_ram_fraction": 0.5,
    }
    height, width, n_frames, bpp = 8000, 8000, 64, 4
    plan = pu.auto_tune_parallel_plan(
        kind="global_reproject",
        frame_shape=(height, width),
        n_frames=n_frames,
        bytes_per_pixel=bpp,
        config=cfg,
        caps=caps,
    )

    logical = max(1, caps.cpu_logical_cores)
    expected_workers = max(1, int(round(logical * cfg["parallel_target_cpu_load"])))
    expected_workers = min(expected_workers, cfg["parallel_max_cpu_workers"])
    assert plan.cpu_workers == expected_workers

    bytes_total = _bytes_for_frames(height, width, n_frames, bpp)
    ram_baseline = max(caps.ram_total_bytes, caps.ram_available_bytes, 8 * GB)
    per_worker_upper = int(ram_baseline * cfg["parallel_target_ram_fraction"])
    assert plan.max_chunk_bytes >= 32 * MB
    assert plan.max_chunk_bytes <= per_worker_upper
    if caps.ram_available_bytes:
        assert plan.max_chunk_bytes <= int(caps.ram_available_bytes * 0.99)

    budget_total_lower = plan.max_chunk_bytes * plan.cpu_workers
    if bytes_total <= budget_total_lower:
        assert not plan.use_memmap
    if plan.rows_per_chunk is not None:
        assert 0 < plan.rows_per_chunk <= height
    assert plan.max_chunk_bytes > 0



def test_auto_tune_disables_gpu_when_autotune_off():
    caps = CAPS
    plan = pu.auto_tune_parallel_plan(
        kind="stack_master_tiles",
        frame_shape=(2048, 2048),
        n_frames=8,
        bytes_per_pixel=4,
        config={"parallel_autotune_enabled": False},
        caps=caps,
    )

    assert plan.cpu_workers == max(1, caps.cpu_logical_cores)
    assert not plan.use_gpu
    assert plan.gpu_max_chunk_bytes in (None, 0)


def test_auto_tune_gpu_rows_follow_fraction():
    caps = CAPS
    cfg = {
        "parallel_gpu_vram_fraction": 0.5,
        "parallel_target_cpu_load": 0.75,
    }
    height, width, n_frames, bpp = 50000, 2000, 5, 4
    plan = pu.auto_tune_parallel_plan(
        kind="global_reproject",
        frame_shape=(height, width),
        n_frames=n_frames,
        bytes_per_pixel=bpp,
        config=cfg,
        caps=caps,
    )

    if not (caps.gpu_available and (caps.gpu_vram_total_bytes or caps.gpu_vram_free_bytes)):
        assert not plan.use_gpu
        assert plan.gpu_max_chunk_bytes in (None, 0)
        assert plan.gpu_rows_per_chunk in (None, 0)
        return

    vram_baseline = max(caps.gpu_vram_total_bytes or 0, caps.gpu_vram_free_bytes or 0)
    assert plan.use_gpu
    assert plan.gpu_max_chunk_bytes is not None
    assert plan.gpu_max_chunk_bytes >= 32 * MB
    assert plan.gpu_max_chunk_bytes <= int(vram_baseline * cfg["parallel_gpu_vram_fraction"])
    if caps.gpu_vram_free_bytes:
        assert plan.gpu_max_chunk_bytes <= int(caps.gpu_vram_free_bytes * 0.99)
    if plan.gpu_rows_per_chunk is not None:
        assert 0 < plan.gpu_rows_per_chunk <= height
