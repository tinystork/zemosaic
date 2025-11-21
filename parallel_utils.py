"""
Utilities to probe CPU/RAM/GPU capabilities and derive a parallelization plan.

This module centralizes the heuristics that were previously scattered across the
worker. It stays lightweight (only ``psutil`` + optional ``cupy``) so that both
CLI and GUI entrypoints can import it without triggering heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import logging
import math
import multiprocessing
import os

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil missing on some environments
    psutil = None

try:
    import cupy as cp  # type: ignore

    _CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - GPU libraries missing
    cp = None
    _CUPY_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clamp(value: float, low: float, high: float, default: float) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return float(min(high, max(low, numeric)))


def _extract_config(config: Mapping[str, Any] | None, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    # Accept SimpleNamespace-like objects
    return getattr(config, key, default)


@dataclass
class ParallelCapabilities:
    cpu_cores: int
    cpu_logical_cores: int
    ram_total_bytes: int
    ram_available_bytes: int
    gpu_available: bool
    gpu_name: str | None
    gpu_vram_total_bytes: int | None
    gpu_vram_free_bytes: int | None


@dataclass
class ParallelPlan:
    cpu_workers: int
    use_memmap: bool
    max_chunk_bytes: int
    rows_per_chunk: int | None = None
    tiles_per_chunk: int | None = None
    use_gpu: bool = False
    gpu_rows_per_chunk: int | None = None
    gpu_max_chunk_bytes: int | None = None


def _probe_ram() -> Tuple[int, int]:
    """Return ``(total_bytes, available_bytes)`` for system RAM."""

    if psutil is None:
        # Best-effort fallback using os.sysconf on POSIX
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            total = int(page_size * phys_pages)
            available = int(page_size * avail_pages)
            return max(total, 0), max(available, 0)
        except Exception:
            pass
        return 0, 0
    try:
        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        return 0, 0


def _probe_gpu() -> Tuple[bool, str | None, int | None, int | None]:
    """Return ``(available, name, total_bytes, free_bytes)``."""

    if not _CUPY_AVAILABLE:
        return False, None, None, None
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        return False, None, None, None

    try:
        current_dev = cp.cuda.runtime.getDevice()
    except Exception:
        current_dev = 0

    try:
        props = cp.cuda.runtime.getDeviceProperties(current_dev)
        gpu_name = props.get("name", "").strip() or None
        total_bytes = _safe_int(props.get("totalGlobalMem"), None)
    except Exception:
        gpu_name = None
        total_bytes = None

    free_bytes = None
    try:
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        free_bytes = int(free_bytes)
    except Exception:
        free_bytes = None

    available = True if gpu_name or total_bytes else True
    return available, gpu_name, total_bytes, free_bytes


def detect_parallel_capabilities() -> ParallelCapabilities:
    """Gather base system information used by the auto-tuner."""

    cpu_cores = 0
    cpu_logical = os.cpu_count() or multiprocessing.cpu_count() or 1
    if psutil is not None:
        try:
            cpu_cores = psutil.cpu_count(logical=False) or 0
        except Exception:
            cpu_cores = 0
    if cpu_cores <= 0:
        cpu_cores = max(1, cpu_logical // 2)

    ram_total, ram_available = _probe_ram()
    gpu_available, gpu_name, gpu_total, gpu_free = _probe_gpu()

    return ParallelCapabilities(
        cpu_cores=int(cpu_cores),
        cpu_logical_cores=int(max(1, cpu_logical)),
        ram_total_bytes=int(max(0, ram_total)),
        ram_available_bytes=int(max(0, ram_available)),
        gpu_available=bool(gpu_available),
        gpu_name=gpu_name,
        gpu_vram_total_bytes=gpu_total,
        gpu_vram_free_bytes=gpu_free,
    )


def _estimate_bytes(frame_shape: Tuple[int, int] | None, n_frames: int, bpp: int) -> Tuple[int, int]:
    if not frame_shape or len(frame_shape) < 2:
        return 0, 0
    height = max(0, int(frame_shape[0]))
    width = max(0, int(frame_shape[1]))
    bytes_per_frame = height * width * max(1, bpp)
    bytes_total = bytes_per_frame * max(1, n_frames)
    return bytes_per_frame, bytes_total


def _compute_rows_per_chunk(bytes_budget: int, width: int, n_frames: int, bytes_per_pixel: int) -> int | None:
    if bytes_budget <= 0 or width <= 0 or n_frames <= 0:
        return None
    bytes_per_row = width * max(1, bytes_per_pixel) * n_frames
    if bytes_per_row <= 0:
        return None
    rows = max(1, bytes_budget // bytes_per_row)
    # Keep chunks reasonably large so we do not thrash the scheduler.
    return int(rows)


def auto_tune_parallel_plan(
    kind: str,
    frame_shape: Tuple[int, int] | None,
    n_frames: int,
    bytes_per_pixel: int,
    config: Mapping[str, Any] | None = None,
    caps: ParallelCapabilities | None = None,
) -> ParallelPlan:
    """
    Heuristic tuner for CPU/GPU workers and chunk sizes.

    Parameters
    ----------
    kind:
        Hint describing the pipeline phase (``"stack_master_tiles"``,
        ``"global_reproject"``, ...). Currently informational only but left in
        place so we can refine heuristics later.
    frame_shape:
        Tuple ``(height, width)`` describing the output grid.
    n_frames:
        Number of input tiles/frames participating in the combine.
    bytes_per_pixel:
        Size (in bytes) of each sample inside the working arrays.
    config:
        Mapping with optional overrides such as ``parallel_target_cpu_load``.
    caps:
        Optional pre-probed capabilities; when omitted they are detected on the fly.
    """

    del kind  # Structured for future per-kind heuristics

    caps = caps or detect_parallel_capabilities()
    cfg = config or {}

    autotune_enabled = bool(_extract_config(cfg, "parallel_autotune_enabled", True))
    target_cpu_load = _clamp(
        _extract_config(cfg, "parallel_target_cpu_load", 0.85),
        0.1,
        1.0,
        0.85,
    )
    target_ram_fraction = _clamp(
        _extract_config(cfg, "parallel_target_ram_fraction", 0.8),
        0.1,
        0.95,
        0.8,
    )
    target_gpu_fraction = _clamp(
        _extract_config(cfg, "parallel_gpu_vram_fraction", 0.5),
        0.1,
        0.9,
        0.5,
    )
    max_cpu_workers = _safe_int(_extract_config(cfg, "parallel_max_cpu_workers", 0), 0)

    logical = max(1, caps.cpu_logical_cores)
    if autotune_enabled:
        workers = max(1, int(math.ceil(logical * target_cpu_load)))
    else:
        workers = logical
    if max_cpu_workers > 0:
        workers = min(workers, max_cpu_workers)
    workers = max(1, workers)

    height = int(frame_shape[0]) if frame_shape else 0
    width = int(frame_shape[1]) if frame_shape else 0
    bytes_per_frame, bytes_total = _estimate_bytes(frame_shape, n_frames, max(1, bytes_per_pixel))

    ram_available = caps.ram_available_bytes or caps.ram_total_bytes
    ram_budget = int((ram_available or (8 * 1024**3)) * target_ram_fraction)
    ram_budget = max(ram_budget, 128 * 1024 * 1024)  # never drop below 128 MB per chunk
    use_memmap = bytes_total > ram_budget
    max_chunk_bytes = ram_budget // max(workers, 1)
    max_chunk_bytes = max(32 * 1024 * 1024, max_chunk_bytes)

    rows_per_chunk = None
    if height > 0 and width > 0 and bytes_per_frame > 0:
        rows_per_chunk = _compute_rows_per_chunk(max_chunk_bytes, width, max(1, n_frames), bytes_per_pixel)
        if rows_per_chunk is not None:
            rows_per_chunk = max(1, min(height, rows_per_chunk))

    gpu_rows_per_chunk = None
    gpu_chunk_bytes = None
    use_gpu = bool(caps.gpu_available and autotune_enabled)
    if use_gpu:
        gpu_free = caps.gpu_vram_free_bytes or caps.gpu_vram_total_bytes
        if gpu_free:
            gpu_chunk_bytes = int(max(32 * 1024 * 1024, gpu_free * target_gpu_fraction))
            gpu_rows_per_chunk = _compute_rows_per_chunk(
                gpu_chunk_bytes,
                width,
                max(1, n_frames),
                bytes_per_pixel,
            )
            if gpu_rows_per_chunk is not None and height > 0:
                gpu_rows_per_chunk = max(4, min(height, gpu_rows_per_chunk))
        else:
            use_gpu = False

    plan = ParallelPlan(
        cpu_workers=workers,
        use_memmap=bool(use_memmap),
        max_chunk_bytes=int(max_chunk_bytes),
        rows_per_chunk=rows_per_chunk,
        tiles_per_chunk=None,
        use_gpu=use_gpu,
        gpu_rows_per_chunk=gpu_rows_per_chunk,
        gpu_max_chunk_bytes=gpu_chunk_bytes,
    )

    LOGGER.debug(
        "Parallel plan: workers=%d, memmap=%s, chunk=%s, gpu=%s, gpu_chunk=%s",
        plan.cpu_workers,
        plan.use_memmap,
        plan.rows_per_chunk,
        plan.use_gpu,
        plan.gpu_rows_per_chunk,
    )
    return plan
