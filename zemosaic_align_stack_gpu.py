"""
GPU-accelerated helpers for Phase 3 stacking.

The public API mirrors the CPU stacker enough for ``zemosaic_worker`` to call
it without changing any high-level workflow. All GPU-specific logic and imports
live inside this module so that machines without CUDA/CuPy can continue to run
Phase 3 on CPU without raising import errors.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

import numpy as np

try:  # pragma: no cover - exercised only when CuPy is available
    import cupy as cp  # type: ignore

    _CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - GPU libraries missing on many machines
    cp = None  # type: ignore
    _CUPY_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - hints only
    from parallel_utils import ParallelPlan
else:
    ParallelPlan = Any  # type: ignore

LOGGER = logging.getLogger(__name__)

DEFAULT_GPU_MAX_CHUNK_BYTES = 512 * 1024 * 1024  # 512 MB
DEFAULT_GPU_ROWS_PER_CHUNK = 256
MIN_GPU_ROWS_PER_CHUNK = 32

_GPU_HEALTH_CHECKED = False
_GPU_HEALTHY = False


class GPUStackingError(RuntimeError):
    """Raised when GPU-based stacking is unavailable or fails."""


def _gpu_is_usable(logger: logging.Logger | None = None) -> bool:
    """
    Return True if a CUDA device is available and a tiny CuPy allocation works.

    The result is cached because devices rarely change between calls.
    """

    global _GPU_HEALTH_CHECKED, _GPU_HEALTHY
    if _GPU_HEALTH_CHECKED:
        return _GPU_HEALTHY
    if not _CUPY_AVAILABLE or cp is None:
        _GPU_HEALTH_CHECKED = True
        _GPU_HEALTHY = False
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        cp.zeros((4, 4), dtype=cp.float32)
        _GPU_HEALTHY = True
    except Exception as exc:  # pragma: no cover - GPU failure path
        if logger:
            try:
                logger.debug("CuPy GPU smoke test failed: %s", exc)
            except Exception:
                pass
        _GPU_HEALTHY = False
    finally:
        _GPU_HEALTH_CHECKED = True
    return _GPU_HEALTHY


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Ensure frames are float32, contiguous, and shaped (H, W, C)."""

    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _resolve_rows_per_chunk(
    height: int,
    width: int,
    channels: int,
    n_frames: int,
    plan: ParallelPlan | None,
) -> int:
    max_bytes = DEFAULT_GPU_MAX_CHUNK_BYTES
    if plan is not None:
        try:
            plan_bytes = getattr(plan, "gpu_max_chunk_bytes", None)
            if plan_bytes:
                max_bytes = int(plan_bytes)
        except Exception:
            pass
    rows_hint = None
    if plan is not None:
        try:
            rows_hint = getattr(plan, "gpu_rows_per_chunk", None)
        except Exception:
            rows_hint = None
    if rows_hint:
        rows = int(rows_hint)
    else:
        bytes_per_row = width * channels * 4 * max(1, n_frames)
        rows = max_bytes // bytes_per_row if bytes_per_row > 0 else 0
    if rows <= 0:
        rows = DEFAULT_GPU_ROWS_PER_CHUNK
    rows = max(MIN_GPU_ROWS_PER_CHUNK, rows)
    return min(height, rows)


def _winsorize_chunk(
    data_gpu: "cp.ndarray",
    low_high_limits: tuple[float, float] | Sequence[float] | None,
) -> "cp.ndarray":
    if not low_high_limits or len(low_high_limits) < 2:
        low, high = 0.05, 0.05
    else:
        low = float(low_high_limits[0])
        high = float(low_high_limits[1])
    low_pct = max(0.0, min(100.0, low * 100.0))
    high_pct = max(0.0, min(100.0, 100.0 - high * 100.0))
    lower = cp.percentile(data_gpu, low_pct, axis=0, keepdims=False)
    upper = cp.percentile(data_gpu, high_pct, axis=0, keepdims=False)
    return cp.clip(data_gpu, lower, upper)


def _kappa_clip_chunk(
    data_gpu: "cp.ndarray",
    kappa_low: float,
    kappa_high: float,
) -> "cp.ndarray":
    mean = cp.mean(data_gpu, axis=0)
    std = cp.std(data_gpu, axis=0)
    lower = mean - float(kappa_low) * std
    upper = mean + float(kappa_high) * std
    mask = (data_gpu >= lower) & (data_gpu <= upper)
    return cp.where(mask, data_gpu, cp.nan)


def _combine_chunk(
    data_gpu: "cp.ndarray",
    method: str,
) -> "cp.ndarray":
    method_norm = (method or "mean").strip().lower()
    if method_norm in {"median", "med"}:
        return cp.nanmedian(data_gpu, axis=0)
    if method_norm in {"min"}:
        return cp.nanmin(data_gpu, axis=0)
    if method_norm in {"max"}:
        return cp.nanmax(data_gpu, axis=0)
    return cp.nanmean(data_gpu, axis=0)


def _require_supported_features(stacking_params: Mapping[str, Any]) -> None:
    weight_method = (stacking_params.get("stack_weight_method") or "none").lower()
    if weight_method not in {"none", "unit", "unity"}:
        raise GPUStackingError(f"GPU stack does not yet support weight method '{weight_method}'")
    apply_radial = bool(stacking_params.get("apply_radial_weight"))
    if apply_radial:
        raise GPUStackingError("GPU stack does not yet support radial weighting")


def gpu_stack_from_arrays(
    aligned_images: Sequence[np.ndarray],
    stacking_params: Mapping[str, Any] | None,
    *,
    parallel_plan: ParallelPlan | None = None,
    logger: logging.Logger | None = None,
    pcb_tile: Callable[..., Any] | None = None,
    tile_id: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Stack aligned frames on the GPU (CuPy) and return the stacked array + metadata.

    Parameters
    ----------
    aligned_images:
        Iterable of numpy arrays shaped ``(H, W)`` or ``(H, W, C)`` in float32.
    stacking_params:
        Dict mirroring the parameters currently passed to the CPU stacker.
    parallel_plan:
        Optional ``ParallelPlan`` controlling chunk sizes.
    pcb_tile:
        Optional logging callback mirroring ``_log_and_callback`` in the worker.
    """

    if logger is None:
        logger = LOGGER
    if not aligned_images:
        raise GPUStackingError("No aligned frames provided for GPU stacking")
    if stacking_params is None:
        stacking_params = {}
    if not _gpu_is_usable(logger):
        raise GPUStackingError("GPU acceleration unavailable (CuPy missing or no CUDA device)")

    _require_supported_features(stacking_params)

    normalized_frames: list[np.ndarray] = []
    first_frame = None
    expanded_channels = False
    for idx, frame in enumerate(aligned_images):
        raw = np.asarray(frame)
        norm = _normalize_frame(frame)
        if first_frame is None:
            first_frame = norm
            expanded_channels = raw.ndim == 2
        else:
            if norm.shape != first_frame.shape:
                raise GPUStackingError(f"Input frame {idx} shape {norm.shape} differs from {first_frame.shape}")
        normalized_frames.append(norm)

    if first_frame is None:
        raise GPUStackingError("Failed to normalize input frames for GPU stacking")

    height, width = first_frame.shape[0], first_frame.shape[1]
    channels = first_frame.shape[2]
    drop_channel = bool(expanded_channels and channels == 1)

    rows_per_chunk = _resolve_rows_per_chunk(height, width, channels, len(normalized_frames), parallel_plan)
    rows_per_chunk = max(1, rows_per_chunk)

    start_time = time.perf_counter()
    stacked = np.empty((height, width, channels), dtype=np.float32)

    algo = (stacking_params.get("stack_reject_algo") or "").strip().lower()
    kappa_low = float(stacking_params.get("stack_kappa_low", 3.0))
    kappa_high = float(stacking_params.get("stack_kappa_high", 3.0))
    winsor_limits = stacking_params.get("parsed_winsor_limits", (0.05, 0.05))
    combine_method = stacking_params.get("stack_final_combine", "mean")

    for row_start in range(0, height, rows_per_chunk):
        row_end = min(height, row_start + rows_per_chunk)
        chunk_cpu = np.stack(
            [frame[row_start:row_end, :, :] for frame in normalized_frames],
            axis=0,
        )
        data_gpu = cp.asarray(chunk_cpu, dtype=cp.float32)
        if algo in {"winsorized_sigma_clip", "winsorized", "winsor"}:
            data_gpu = _winsorize_chunk(data_gpu, winsor_limits)
        elif algo in {"kappa_sigma", "sigma_clip"}:
            data_gpu = _kappa_clip_chunk(data_gpu, kappa_low, kappa_high)
        elif algo in {"linear_fit_clip"}:
            raise GPUStackingError("linear_fit_clip is not implemented for GPU stacking yet")
        chunk_result = _combine_chunk(data_gpu, combine_method)
        stacked[row_start:row_end] = cp.asnumpy(chunk_result)

    duration = time.perf_counter() - start_time
    if pcb_tile:
        try:
            pcb_tile(
                "phase3_gpu_chunk_summary",
                prog=None,
                lvl="INFO_DETAIL",
                rows_per_chunk=int(rows_per_chunk),
                total_rows=int(height),
                tile_id=tile_id,
                duration_s=float(duration),
            )
        except Exception:
            pass
    else:
        try:
            logger.debug(
                "Phase 3 GPU stacking completed in %.3fs (rows/chunk=%d, frames=%d)",
                duration,
                rows_per_chunk,
                len(normalized_frames),
            )
        except Exception:
            pass

    if drop_channel:
        stacked = stacked[..., 0]

    stack_metadata: dict[str, Any] = {}
    return stacked, stack_metadata


def gpu_stack_from_paths(
    image_descriptors: Sequence[Any],
    stacking_params: Mapping[str, Any] | None,
    *,
    array_loader: Callable[[Any], np.ndarray] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Convenience wrapper that converts descriptors/paths into numpy arrays.

    ``image_descriptors`` can be any iterable of:
      - numpy arrays
      - dicts with ``"data"`` or ``"array"`` entries
      - paths to ``.npy`` files (loaded via ``np.load``)
    """

    arrays: list[np.ndarray] = []
    loader = array_loader or _default_array_loader
    for desc in image_descriptors:
        arrays.append(loader(desc))
    return gpu_stack_from_arrays(arrays, stacking_params, **kwargs)


def _default_array_loader(descriptor: Any) -> np.ndarray:
    if isinstance(descriptor, np.ndarray):
        return descriptor
    if isinstance(descriptor, (str, bytes, os.PathLike)):
        return np.load(descriptor, allow_pickle=False)
    if hasattr(descriptor, "get"):
        data = descriptor.get("data") or descriptor.get("array")
        if data is not None:
            return np.asarray(data, dtype=np.float32)
        path = descriptor.get("path") or descriptor.get("path_preprocessed_cache")
        if path:
            return np.load(path, allow_pickle=False)
    raise GPUStackingError("Unsupported descriptor type for gpu_stack_from_paths")


__all__ = [
    "GPUStackingError",
    "gpu_stack_from_arrays",
    "gpu_stack_from_paths",
]
