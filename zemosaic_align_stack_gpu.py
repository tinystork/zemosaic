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
from contextlib import nullcontext

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

try:
    import zemosaic_align_stack as _zas  # type: ignore

    _CPU_STACK_HELPERS_AVAILABLE = True
except Exception:  # pragma: no cover - guards GPU path even if CPU helpers fail to import
    _zas = None  # type: ignore
    _CPU_STACK_HELPERS_AVAILABLE = False

try:
    from zemosaic_utils import make_radial_weight_map as _make_radial_weight_map  # type: ignore

    _RADIAL_WEIGHT_AVAILABLE = True
except Exception:  # pragma: no cover - radial weighting optional
    _make_radial_weight_map = None  # type: ignore
    _RADIAL_WEIGHT_AVAILABLE = False

DEFAULT_GPU_MAX_CHUNK_BYTES = 512 * 1024 * 1024  # 512 MB
DEFAULT_GPU_ROWS_PER_CHUNK = 256
MIN_GPU_ROWS_PER_CHUNK = 32
SAFE_MODE_ROWS_CAP = 128
SAFE_MODE_CHUNK_TIMEOUT_SEC = 3.0

_GPU_HEALTH_CHECKED = False
_GPU_HEALTHY = False


class GPUStackingError(RuntimeError):
    """Raised when GPU-based stacking is unavailable or fails."""


def _gpu_safe_mode_enabled() -> bool:
    """Return True if the worker requested GPU safe mode via environment."""

    return os.environ.get("ZEMOSAIC_GPU_SAFE_MODE", "").strip() == "1"


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
    if _gpu_safe_mode_enabled():
        rows = min(rows, SAFE_MODE_ROWS_CAP)
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
    apply_radial = bool(stacking_params.get("apply_radial_weight"))
    needs_weights = weight_method not in {"", "none", "unit", "unity"}
    if (needs_weights or apply_radial) and not _CPU_STACK_HELPERS_AVAILABLE:
        raise GPUStackingError(
            "GPU stack requires CPU weighting helpers to mirror CPU path but they are unavailable"
        )
    if needs_weights and weight_method not in {"noise_variance", "noise_fwhm", "unit", "unity"}:
        LOGGER.warning(
            "GPU stack: unknown weight method '%s' requested; continuing without special handling.",
            weight_method,
        )


def _coerce_config_bool(value: Any, default: bool = True) -> bool:
    """Best-effort boolean coercion mirroring the CPU stacker."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return default
    try:
        return bool(value)
    except Exception:
        return default


def _progress_callback_adapter(
    pcb_tile: Callable[..., Any] | None, logger: logging.Logger | None
) -> Callable[..., None]:
    """Normalize progress callbacks to the signature used in CPU helpers."""

    def _cb(msg_key, prog=None, lvl="INFO_DETAIL", **kwargs):
        if pcb_tile:
            try:
                pcb_tile(msg_key, prog=prog, lvl=lvl, **kwargs)
                return
            except Exception:
                pass
        if logger:
            try:
                level_upper = str(lvl).upper() if isinstance(lvl, str) else ""
                if "ERROR" in level_upper:
                    lvl_num = logging.ERROR
                elif "WARN" in level_upper:
                    lvl_num = logging.WARNING
                elif "INFO" in level_upper:
                    lvl_num = logging.INFO
                else:
                    lvl_num = logging.DEBUG
                logger.log(lvl_num, "%s %s", msg_key, kwargs if kwargs else "")
            except Exception:
                pass

    return _cb


def _broadcast_weight_template(weight_template: Any, target_shape: tuple[int, ...]) -> np.ndarray | None:
    """Expand a compact per-frame weight to match ``target_shape``."""

    if weight_template is None:
        return None
    w = np.asarray(weight_template, dtype=np.float32, copy=False)
    if w.ndim == 0:
        w = w.reshape((1,))
    if w.ndim == 1 and target_shape[-1] == 3 and w.shape[0] == 3:
        w = w.reshape((1, 1, 3))
    try:
        return np.broadcast_to(w, target_shape).astype(np.float32, copy=False)
    except Exception:
        return None


def _compute_radial_weight_map(
    height: int,
    width: int,
    channels: int,
    stacking_params: Mapping[str, Any],
    logger: logging.Logger | None,
) -> np.ndarray | None:
    """Create a per-pixel radial weighting map if requested."""

    apply_radial = bool(stacking_params.get("apply_radial_weight"))
    if not apply_radial:
        return None
    if not _RADIAL_WEIGHT_AVAILABLE or _make_radial_weight_map is None:
        if logger:
            try:
                logger.warning("Radial weighting requested but helper unavailable; continuing without it.")
            except Exception:
                pass
        return None
    try:
        feather = float(stacking_params.get("radial_feather_fraction", 0.8))
        shape_power = float(stacking_params.get("radial_shape_power", 2.0))
    except Exception:
        feather = 0.8
        shape_power = 2.0
    try:
        radial_2d = _make_radial_weight_map(height, width, feather_fraction=feather, shape_power=shape_power)
        radial_2d = np.asarray(radial_2d, dtype=np.float32, copy=False)
        if channels == 1:
            return radial_2d[..., None]
        return np.repeat(radial_2d[..., None], channels, axis=2)
    except Exception as exc:
        if logger:
            try:
                logger.warning("Radial weighting failed (%s); continuing without it.", exc)
            except Exception:
                pass
        return None


def _prepare_frames_and_weights(
    aligned_images: Sequence[np.ndarray],
    stacking_params: Mapping[str, Any],
    *,
    zconfig: Any | None,
    pcb_tile: Callable[..., Any] | None,
    logger: logging.Logger | None,
) -> tuple[list[np.ndarray], np.ndarray | None, str, dict[str, float] | None, bool]:
    """
    Mirror the CPU stacker's preprocessing: normalize frames and compute weights.

    Returns (frames, weights_stack, weight_method_used, weight_stats, expanded_channels_flag).
    """

    if not aligned_images:
        raise GPUStackingError("No aligned frames provided for GPU stacking")

    progress_cb = _progress_callback_adapter(pcb_tile, logger)
    normalized_frames: list[np.ndarray] = []
    first_shape = None
    expanded_channels = False

    # Basic normalization to float32 + shape enforcement
    for idx, frame in enumerate(aligned_images):
        raw = np.asarray(frame)
        norm = _normalize_frame(frame)
        if first_shape is None:
            first_shape = norm.shape
            expanded_channels = raw.ndim == 2
        else:
            if norm.shape != first_shape:
                if logger:
                    try:
                        logger.warning(
                            "GPU stack: dropping frame %d due to shape mismatch (got %s, expected %s)",
                            idx,
                            norm.shape,
                            first_shape,
                        )
                    except Exception:
                        pass
                continue
        normalized_frames.append(norm)

    if not normalized_frames or first_shape is None:
        raise GPUStackingError("No usable frames after normalization")

    # Optional photometric normalization (linear_fit / sky_mean)
    norm_method = (stacking_params.get("stack_norm_method") or "none").strip().lower()
    use_gpu_norm = False
    if zconfig is not None:
        use_gpu_norm = _coerce_config_bool(
            getattr(zconfig, "stack_use_gpu", getattr(zconfig, "use_gpu_stack", getattr(zconfig, "use_gpu", False))),
            False,
        )

    if _CPU_STACK_HELPERS_AVAILABLE and _zas is not None:
        try:
            if norm_method in {"linear_fit", "linear"} and hasattr(_zas, "_normalize_images_linear_fit"):
                normalized_frames = _zas._normalize_images_linear_fit(
                    normalized_frames,
                    progress_callback=progress_cb,
                    use_gpu=use_gpu_norm,
                )
            elif norm_method in {"sky_mean", "skymean"} and hasattr(_zas, "_normalize_images_sky_mean"):
                normalized_frames = _zas._normalize_images_sky_mean(
                    normalized_frames,
                    progress_callback=progress_cb,
                    use_gpu=use_gpu_norm,
                )
        except Exception as exc:
            if logger:
                try:
                    logger.warning("GPU stack: normalization '%s' failed (%s); continuing without it.", norm_method, exc)
                except Exception:
                    pass

    # Filter out None entries post-normalization and ensure numeric data
    filtered_frames: list[np.ndarray] = []
    for frame in normalized_frames:
        if frame is None:
            continue
        frame_f32 = np.asarray(frame, dtype=np.float32, copy=False)
        if not np.all(np.isfinite(frame_f32)):
            frame_f32 = np.nan_to_num(frame_f32, nan=0.0, posinf=0.0, neginf=0.0)
        filtered_frames.append(frame_f32 if frame_f32.flags.c_contiguous else np.ascontiguousarray(frame_f32))

    if not filtered_frames:
        raise GPUStackingError("All frames were invalid after preprocessing")

    height, width, channels = filtered_frames[0].shape
    weight_method = (stacking_params.get("stack_weight_method") or "none").strip().lower()
    weight_method_used = weight_method or "none"
    weight_stats: dict[str, float] | None = None
    quality_weights: list[np.ndarray | None] | None = None

    if _CPU_STACK_HELPERS_AVAILABLE and _zas is not None and hasattr(_zas, "_compute_quality_weights"):
        try:
            q_weights, weight_method_used, weight_stats = _zas._compute_quality_weights(
                filtered_frames,
                weight_method,
                progress_callback=progress_cb,
            )
            quality_weights = q_weights if q_weights else None
        except Exception as exc:
            quality_weights = None
            if logger:
                try:
                    logger.warning("GPU stack: quality weighting failed (%s); continuing without weights.", exc)
                except Exception:
                    pass

    radial_map = _compute_radial_weight_map(height, width, channels, stacking_params, logger)

    # Align with CPU behavior: when no radial map is present, the CPU path drops
    # compact (1x1xC) quality weights due to shape mismatch. Skip them here too
    # to avoid GPU-only weighting that drifts from the CPU reference.
    if radial_map is None:
        quality_weights = None
        weight_method_used = "none"
        weight_stats = None

    weights_stack: np.ndarray | None = None
    if radial_map is not None or quality_weights is not None:
        combined_weights: list[np.ndarray | None] = []
        for idx, frame in enumerate(filtered_frames):
            q_weight = quality_weights[idx] if quality_weights is not None and idx < len(quality_weights) else None
            if q_weight is None and radial_map is None:
                combined_weights.append(None)
                continue
            base = _broadcast_weight_template(q_weight, frame.shape)
            if base is None and q_weight is not None:
                combined_weights.append(None)
                continue
            if radial_map is None:
                combined = base
            elif base is None:
                combined = radial_map
            else:
                combined = radial_map * base
            combined_weights.append(np.asarray(combined, dtype=np.float32, copy=False))

        if combined_weights and all(w is not None for w in combined_weights):
            try:
                weights_stack = np.stack(combined_weights, axis=0)
            except Exception:
                weights_stack = None

    return filtered_frames, weights_stack, weight_method_used, weight_stats, expanded_channels


def _combine_weighted_chunk(
    data_gpu: "cp.ndarray",
    weights_gpu: "cp.ndarray" | None,
    method: str,
) -> "cp.ndarray":
    """Combine a chunk with optional weights, mirroring CPU mean logic."""

    method_norm = (method or "mean").strip().lower()
    if weights_gpu is None or method_norm != "mean":
        return _combine_chunk(data_gpu, method)

    mask = cp.isfinite(data_gpu)
    safe_weights = cp.where(mask, weights_gpu, cp.zeros_like(weights_gpu))
    # Match the CPU path's float64 accumulation to minimize parity drift.
    weighted_sum = cp.nansum(data_gpu * safe_weights, axis=0, dtype=cp.float64)
    weight_sum = cp.sum(safe_weights, axis=0, dtype=cp.float64)
    err_ctx = getattr(cp, "errstate", None)
    with (err_ctx(divide="ignore", invalid="ignore") if callable(err_ctx) else nullcontext()):
        combined = cp.where(weight_sum > 0, weighted_sum / weight_sum, cp.nan)
    needs_fallback = ~cp.isfinite(combined)
    if cp.any(needs_fallback):
        fallback = cp.nanmean(data_gpu.astype(cp.float64, copy=False), axis=0)
        combined = cp.where(needs_fallback, fallback, combined)
    return combined


def _poststack_rgb_equalization(
    stacked: np.ndarray | None,
    stacking_params: Mapping[str, Any] | None = None,
    zconfig: Any | None = None,
) -> dict[str, Any]:
    """
    Apply optional RGB median equalization and return metadata.

    Keeps GPU-only or mixed GPU/CPU runs consistent with the CPU stacker's
    per-stack equalization behavior.
    """

    default_enabled = True
    enabled = default_enabled
    try:
        if stacking_params is not None and hasattr(stacking_params, "get"):
            enabled = _coerce_config_bool(
                stacking_params.get("poststack_equalize_rgb", default_enabled),
                default_enabled,
            )
    except Exception:
        enabled = default_enabled

    if zconfig is not None:
        try:
            enabled = _coerce_config_bool(
                getattr(zconfig, "poststack_equalize_rgb", enabled),
                enabled,
            )
        except Exception:
            pass

    info = {
        "enabled": enabled,
        "applied": False,
        "gain_r": 1.0,
        "gain_g": 1.0,
        "gain_b": 1.0,
        "target_median": float("nan"),
    }

    if not enabled or stacked is None or not isinstance(stacked, np.ndarray):
        return info
    if stacked.ndim != 3 or stacked.shape[2] != 3:
        return info

    try:
        med = np.nanmedian(stacked, axis=(0, 1)).astype(np.float32)
        finite = np.isfinite(med) & (med > 0)
        if not np.any(finite):
            return info

        target = float(np.nanmedian(med[finite]))
        gains = np.ones(3, dtype=np.float32)
        gains[finite] = target / med[finite]
        try:
            if not stacked.flags.writeable:
                try:
                    stacked.setflags(write=True)
                except Exception:
                    pass
            np.multiply(stacked, gains[None, None, :], out=stacked, casting="unsafe")
        except Exception:
            np.multiply(stacked, gains[None, None, :], out=None, casting="unsafe")

        info.update(
            gain_r=float(gains[0]),
            gain_g=float(gains[1]),
            gain_b=float(gains[2]),
            target_median=float(target),
        )
        if np.isfinite(target):
            info["applied"] = True
            LOGGER.info(
                "[RGB-EQ][GPU] Applied per-substack RGB equalization: gains=(%.6f,%.6f,%.6f) target=%.6g",
                gains[0],
                gains[1],
                gains[2],
                target,
            )
    except Exception as exc:
        LOGGER.warning("[RGB-EQ][GPU] Skipped RGB equalization due to error: %s", exc)

    return info


def gpu_stack_from_arrays(
    aligned_images: Sequence[np.ndarray],
    stacking_params: Mapping[str, Any] | None,
    *,
    parallel_plan: ParallelPlan | None = None,
    logger: logging.Logger | None = None,
    pcb_tile: Callable[..., Any] | None = None,
    tile_id: int | None = None,
    zconfig: Any | None = None,
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
    zconfig:
        Optional configuration namespace mirroring the CPU stacker options.
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

    frames_for_stack, weights_stack, weight_method_used, weight_stats, expanded_channels = _prepare_frames_and_weights(
        aligned_images,
        stacking_params,
        zconfig=zconfig,
        pcb_tile=pcb_tile,
        logger=logger,
    )

    if not frames_for_stack:
        raise GPUStackingError("No frames available after preprocessing for GPU stacking")

    height, width, channels = frames_for_stack[0].shape
    drop_channel = bool(expanded_channels and channels == 1)
    gpu_safe_mode = _gpu_safe_mode_enabled()

    rows_per_chunk = _resolve_rows_per_chunk(height, width, channels, len(frames_for_stack), parallel_plan)
    rows_per_chunk = max(1, rows_per_chunk)

    start_time = time.perf_counter()
    stacked = np.empty((height, width, channels), dtype=np.float32)

    algo = (stacking_params.get("stack_reject_algo") or "").strip().lower()
    kappa_low = float(stacking_params.get("stack_kappa_low", 3.0))
    kappa_high = float(stacking_params.get("stack_kappa_high", 3.0))
    winsor_limits = stacking_params.get("parsed_winsor_limits", (0.05, 0.05))
    combine_method = stacking_params.get("stack_final_combine", "mean")
    if weights_stack is not None and str(combine_method).strip().lower() != "mean":
        try:
            logger.warning(
                "GPU stack: weights available but combine method '%s' does not support weighting; using unweighted output.",
                combine_method,
            )
        except Exception:
            pass

    for row_start in range(0, height, rows_per_chunk):
        chunk_start_time = time.perf_counter() if gpu_safe_mode else None
        row_end = min(height, row_start + rows_per_chunk)
        chunk_cpu = np.stack(
            [frame[row_start:row_end, :, :] for frame in frames_for_stack],
            axis=0,
        )
        weights_chunk_gpu = None
        if weights_stack is not None:
            weights_chunk_cpu = weights_stack[:, row_start:row_end, :, :]
            weights_chunk_gpu = cp.asarray(weights_chunk_cpu, dtype=cp.float32)
        data_gpu = cp.asarray(chunk_cpu, dtype=cp.float32)
        if algo in {"winsorized_sigma_clip", "winsorized", "winsor"}:
            data_gpu = _winsorize_chunk(data_gpu, winsor_limits)
        elif algo in {"kappa_sigma", "sigma_clip"}:
            data_gpu = _kappa_clip_chunk(data_gpu, kappa_low, kappa_high)
        elif algo in {"linear_fit_clip"}:
            raise GPUStackingError("linear_fit_clip is not implemented for GPU stacking yet")
        chunk_result = _combine_weighted_chunk(data_gpu, weights_chunk_gpu, combine_method)
        stacked[row_start:row_end] = cp.asnumpy(chunk_result)
        if gpu_safe_mode:
            try:
                cp.cuda.Stream.null.synchronize()
            except Exception:
                pass
            if chunk_start_time is not None:
                duration_chunk = time.perf_counter() - chunk_start_time
                if duration_chunk > SAFE_MODE_CHUNK_TIMEOUT_SEC:
                    raise GPUStackingError(
                        f"GPU chunk exceeded safe timeout ({duration_chunk:.2f}s > {SAFE_MODE_CHUNK_TIMEOUT_SEC:.2f}s)"
                    )

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
                len(frames_for_stack),
            )
        except Exception:
            pass

    if drop_channel:
        stacked = stacked[..., 0]

    # Validate GPU output before returning to the worker. Falling back to CPU when the
    # GPU path produces only NaNs or an all-zero array keeps parity with the reference
    # CPU stacker instead of silently returning unusable data.
    if not np.any(np.isfinite(stacked)):
        raise GPUStackingError("GPU stack produced no finite pixels; falling back to CPU")
    if not np.all(np.isfinite(stacked)):
        stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        max_abs = float(np.nanmax(np.abs(stacked)))
    except Exception:
        max_abs = 0.0
    if max_abs <= 0.0:
        raise GPUStackingError("GPU stack produced a zero-valued image; falling back to CPU")

    stack_metadata: dict[str, Any] = {
        "weight_method": weight_method_used,
        "weight_stats": weight_stats,
    }
    rgb_eq_info = _poststack_rgb_equalization(stacked, stacking_params, zconfig)
    stack_metadata["rgb_equalization"] = rgb_eq_info
    return stacked, stack_metadata


def gpu_stack_from_paths(
    image_descriptors: Sequence[Any],
    stacking_params: Mapping[str, Any] | None,
    *,
    array_loader: Callable[[Any], np.ndarray] | None = None,
    zconfig: Any | None = None,
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
    return gpu_stack_from_arrays(arrays, stacking_params, zconfig=zconfig, **kwargs)


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
