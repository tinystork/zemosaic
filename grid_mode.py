"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                               ║
║                                                                                   ║
║ Auteur  : Tinystork, seigneur des couteaux à beurre (aka Tristan Nauleau)         ║
║ Partenaire : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System   ║
║              (aka ChatGPT, Grand Maître du ciselage de code)                      ║
║                                                                                   ║
║ Licence : GNU General Public License v3.0 (GPL-3.0)                               ║
║                                                                                   ║
║ Description :                                                                     ║
║   Ce programme a été forgé à la lueur des pixels et de la caféine,                ║
║   dans le but noble de transformer des nuages de photons en art                   ║
║   astronomique. Si vous l’utilisez, pensez à dire “merci”,                        ║
║   à lever les yeux vers le ciel, ou à citer Tinystork et J.A.R.V.I.S.             ║
║   (le karma des développeurs en dépend).                                          ║
║                                                                                   ║
║ Avertissement :                                                                   ║
║   Aucune IA ni aucun couteau à beurre n’a été blessé durant le                    ║
║   développement de ce code.                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════╝


╔═══════════════════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                               ║
║                                                                                   ║
║ Author  : Tinystork, Lord of the Butter Knives (aka Tristan Nauleau)              ║
║ Partner : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System      ║
║           (aka ChatGPT, Grand Master of Code Chiseling)                           ║
║                                                                                   ║
║ License : GNU General Public License v3.0 (GPL-3.0)                               ║
║                                                                                   ║
║ Description:                                                                      ║
║   This program was forged under the sacred light of pixels and                    ║
║   caffeine, with the noble intent of turning clouds of photons into               ║
║   astronomical art. If you use it, please consider saying “thanks,”               ║
║   gazing at the stars, or crediting Tinystork and J.A.R.V.I.S. —                  ║
║   developer karma depends on it.                                                  ║
║                                                                                   ║
║ Disclaimer:                                                                       ║
║   No AIs or butter knives were harmed in the making of this code.                 ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
Grid/Survey processing pipeline for ZeMosaic.

All logic here is isolated from the classic worker workflow and is only
invoked when a ``stack_plan.csv`` file is present in the input folder."""

from __future__ import annotations

import copy
import csv
import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # Optional heavy deps – handled gracefully if missing
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    import astropy.units as u

    _ASTROPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    fits = None
    WCS = None
    proj_plane_pixel_scales = None
    u = None
    _ASTROPY_AVAILABLE = False

try:
    from astropy.coordinates import SkyCoord
except Exception:  # pragma: no cover - optional dependency
    SkyCoord = None

try:
    from reproject.mosaicking import find_optimal_celestial_wcs
    from reproject import reproject_interp

    _REPROJECT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    find_optimal_celestial_wcs = None
    reproject_interp = None
    _REPROJECT_AVAILABLE = False

try:
    import scipy.ndimage as ndimage

    _NDIMAGE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    ndimage = None
    _NDIMAGE_AVAILABLE = False

try:
    from zemosaic_align_stack import (
        _reject_outliers_kappa_sigma,
        _reject_outliers_winsorized_sigma_clip,
        equalize_rgb_medians_inplace,
    )
except Exception:  # pragma: no cover - worker remains functional without rejection helpers
    _reject_outliers_kappa_sigma = None
    _reject_outliers_winsorized_sigma_clip = None
    equalize_rgb_medians_inplace = None

logger = logging.getLogger("ZeMosaicWorker").getChild("grid_mode")
# Propagate to the worker logger without attaching a NullHandler so Grid messages
# share the same output stream.
logger.propagate = True

_DEGRADED_WCS_FRAMES: set[str] = set()


ProgressCallback = Optional[Callable[[str, object, str], None]]


def _emit(msg: str, *, lvl: str = "INFO", callback: ProgressCallback = None, **kwargs) -> None:
    """Emit a `[GRID]` log and mirror it to the worker progress callback."""

    tag = f"[GRID] {msg}"
    level = getattr(logging, str(lvl).upper(), logging.INFO)
    try:
        logger.log(level, tag)
    except Exception:
        pass
    if callback:
        try:
            callback(tag, None, str(lvl).upper(), **kwargs)
        except Exception:
            try:
                logger.debug("Progress callback failed for %s", tag, exc_info=True)
            except Exception:
                pass


def _log_wcs_degraded(frame: FrameInfo, *, progress_callback: ProgressCallback = None) -> None:
    """Emit a single degraded-WCS message per frame."""

    key = str(frame.path)
    if key in _DEGRADED_WCS_FRAMES:
        return
    _DEGRADED_WCS_FRAMES.add(key)
    name = frame.path.name if isinstance(frame.path, Path) else str(frame.path)
    _emit(f"WCS convergence degraded for frame {name} (distortion-stripped WCS used)", lvl="WARN", callback=progress_callback)


def _has_wcs_convergence_warning(caught: list[warnings.WarningMessage] | None) -> bool:
    if not caught:
        return False
    for w in caught:
        try:
            text = str(getattr(w, "message", ""))
        except Exception:
            text = ""
        if "all_world2pix" in text or "Reproject encountered WCS" in text:
            return True
    return False


def _open_fits_safely(path: Path):
    """Open a FITS file with the robust settings used across Grid mode."""

    if not (_ASTROPY_AVAILABLE and fits):
        raise RuntimeError("Astropy FITS support unavailable")
    return fits.open(path, memmap=False, do_not_scale_image_data=True)


def _ensure_hwc_array(data: np.ndarray) -> np.ndarray:
    """Return data as HxWxC, squeezing extras and converting 2D to 3D."""

    arr = np.asarray(data)
    if arr.ndim > 3:
        arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    elif arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    return arr.astype(np.float32, copy=False)


@dataclass
class FrameInfo:
    path: Path
    exposure: float = 1.0
    bortle: str | None = None
    filter_name: str | None = None
    batch_id: str | None = None
    order: int = 0
    wcs: object | None = None
    shape_hw: tuple[int, int] | None = None
    footprint: tuple[float, float, float, float] | None = None  # (xmin, xmax, ymin, ymax) in global pixels


@dataclass
class GridTile:
    tile_id: int
    bbox: tuple[int, int, int, int]  # (xmin, xmax, ymin, ymax) in global pixels
    wcs: object
    frames: list[FrameInfo] = field(default_factory=list)
    output_path: Path | None = None


@dataclass
class GridDefinition:
    global_wcs: object
    global_shape_hw: tuple[int, int]
    tile_size_px: int
    overlap_fraction: float
    tiles: list[GridTile]


@dataclass
class GridModeConfig:
    grid_size_factor: float = 1.0
    overlap_fraction: float = 0.1
    stack_norm_method: str = "linear_fit"
    stack_weight_method: str = "noise_variance"
    stack_reject_algo: str = "kappa_sigma"
    stack_kappa_low: float = 3.0
    stack_kappa_high: float = 3.0
    winsor_limits: tuple[float, float] = (0.05, 0.05)
    stack_final_combine: str = "mean"
    stack_chunk_budget_mb: float = 512.0
    apply_radial_weight: bool = False
    radial_feather_fraction: float = 0.8
    radial_shape_power: float = 2.0
    save_final_as_uint16: bool = False
    legacy_rgb_cube: bool = False


@dataclass
class TilePhotometryInfo:
    tile_id: int
    bbox: tuple[int, int, int, int]
    data: np.ndarray
    mask: np.ndarray
    gain: np.ndarray | None = None
    offset: np.ndarray | None = None
    background: np.ndarray | None = None


@dataclass
class TileOverlap:
    tile_a: int
    tile_b: int
    global_bbox: tuple[int, int, int, int]
    slice_a: tuple[slice, slice]
    slice_b: tuple[slice, slice]
    sample_count: int = 0


def compute_valid_mask(data: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Return a boolean mask where True marks finite, non-empty pixels."""

    arr = np.asarray(data)
    finite = np.isfinite(arr)
    if eps <= 0:
        return finite
    try:
        non_empty = np.abs(arr) > eps
    except Exception:
        non_empty = np.ones_like(arr, dtype=bool)
    return finite & non_empty


def _sigma_clipped_median(values: np.ndarray, *, sigma: float = 3.0, max_iters: int = 3) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    for _ in range(max_iters):
        med = float(np.median(arr))
        if not math.isfinite(med):
            break
        diff = arr - med
        std = float(np.std(diff))
        if not (math.isfinite(std) and std > 1e-6):
            break
        keep = np.abs(diff) <= sigma * std
        if np.all(keep):
            break
        arr = arr[keep]
        if arr.size == 0:
            return float("nan")
    return float(np.median(arr)) if arr.size else float("nan")


def estimate_tile_background(tile_data: np.ndarray, tile_mask: np.ndarray, *, sigma: float = 3.0, max_iters: int = 3) -> np.ndarray:
    """Return a sigma-clipped median background per channel."""

    arr = np.asarray(tile_data, dtype=np.float32)
    mask = np.asarray(tile_mask, dtype=bool)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    if mask.ndim == 2 and arr.ndim == 3:
        mask = np.repeat(mask[..., np.newaxis], arr.shape[-1], axis=2)
    backgrounds: list[float] = []
    for c in range(arr.shape[-1]):
        mask_c = mask if mask.ndim == 2 else mask[..., c]
        finite_mask = mask_c & np.isfinite(arr[..., c])
        vals = arr[..., c][finite_mask]
        if vals.size == 0:
            backgrounds.append(float("nan"))
            continue
        bg = _sigma_clipped_median(vals, sigma=sigma, max_iters=max_iters)
        backgrounds.append(bg)
    if not backgrounds:
        return np.array([], dtype=np.float32)
    return np.asarray(backgrounds, dtype=np.float32)


def _smooth_image_for_pyramid(image: np.ndarray) -> np.ndarray:
    """Apply a small blur used when constructing Gaussian pyramids."""

    arr = np.asarray(image, dtype=np.float32)
    if _NDIMAGE_AVAILABLE and ndimage is not None:
        sigma = (1.0, 1.0, 0.0) if arr.ndim == 3 else 1.0
        try:
            return ndimage.gaussian_filter(arr, sigma=sigma, mode="nearest")
        except Exception:
            pass
    if arr.ndim == 2:
        arr_padded = np.pad(arr, 1, mode="edge")
        blurred = (
            arr_padded[:-2, :-2]
            + arr_padded[:-2, 1:-1]
            + arr_padded[:-2, 2:]
            + arr_padded[1:-1, :-2]
            + arr_padded[1:-1, 1:-1]
            + arr_padded[1:-1, 2:]
            + arr_padded[2:, :-2]
            + arr_padded[2:, 1:-1]
            + arr_padded[2:, 2:]
        ) / 9.0
        return blurred.astype(np.float32, copy=False)
    arr_padded = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="edge")
    blurred = (
        arr_padded[:-2, :-2, :]
        + arr_padded[:-2, 1:-1, :]
        + arr_padded[:-2, 2:, :]
        + arr_padded[1:-1, :-2, :]
        + arr_padded[1:-1, 1:-1, :]
        + arr_padded[1:-1, 2:, :]
        + arr_padded[2:, :-2, :]
        + arr_padded[2:, 1:-1, :]
        + arr_padded[2:, 2:, :]
    ) / 9.0
    return blurred.astype(np.float32, copy=False)


def _downsample_image(image: np.ndarray) -> np.ndarray:
    blurred = _smooth_image_for_pyramid(image)
    return blurred[::2, ::2, ...]


def _upsample_image(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    up = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
    h, w = target_shape
    return up[:h, :w, ...]


def build_gaussian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    pyramid: List[np.ndarray] = []
    arr = np.asarray(image, dtype=np.float32)
    pyramid.append(arr)
    for _ in range(1, max(1, levels)):
        prev = pyramid[-1]
        if min(prev.shape[0], prev.shape[1]) < 2:
            break
        pyramid.append(_downsample_image(prev))
    return pyramid


def build_laplacian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    g_pyr = build_gaussian_pyramid(image, levels)
    if not g_pyr:
        return []
    laplacian: List[np.ndarray] = []
    for i in range(len(g_pyr) - 1):
        upsampled = _upsample_image(g_pyr[i + 1], g_pyr[i].shape[:2])
        laplacian.append(g_pyr[i] - upsampled)
    laplacian.append(g_pyr[-1])
    return laplacian


def reconstruct_from_laplacian(pyramid: List[np.ndarray]) -> np.ndarray:
    if not pyramid:
        return np.array([], dtype=np.float32)
    img = pyramid[-1]
    for lev in reversed(pyramid[:-1]):
        img = _upsample_image(img, lev.shape[:2]) + lev
    return img


def detect_grid_mode(input_folder: str | os.PathLike[str]) -> bool:
    """Return True when a stack_plan.csv is present in *input_folder*."""

    try:
        candidate = Path(input_folder).expanduser() / "stack_plan.csv"
    except Exception:
        return False
    return candidate.is_file()


def _parse_float(value: object, default: float = 0.0) -> float:
    try:
        val = float(value)
        if math.isfinite(val):
            return val
    except Exception:
        pass
    return default


def _parse_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _resolve_path(base_dir: Path, text: str | os.PathLike[str]) -> Path:
    try:
        candidate = Path(text)
    except Exception:
        candidate = Path(str(text))
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.expanduser().resolve(strict=False)


def _dialect_from_sample(sample: str) -> csv.Dialect | None:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;|\t ")
    except Exception:
        return None


def load_stack_plan(csv_path: str | os.PathLike[str], *, progress_callback: ProgressCallback = None) -> list[FrameInfo]:
    """Parse ``stack_plan.csv`` and return a list of FrameInfo entries."""

    csv_file = Path(csv_path).expanduser()
    base_dir = csv_file.parent
    if not csv_file.is_file():
        _emit(f"stack_plan.csv not found at {csv_file}", lvl="WARN", callback=progress_callback)
        return []

    try:
        raw_text = csv_file.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        _emit(f"Unable to read stack_plan.csv: {exc}", lvl="ERROR", callback=progress_callback)
        return []

    sample = raw_text[:2048]
    dialect = _dialect_from_sample(sample) or csv.excel
    lines = raw_text.splitlines()
    if not lines:
        _emit("stack_plan.csv is empty", lvl="WARN", callback=progress_callback)
        return []

    reader = csv.DictReader(lines, dialect=dialect)
    frames: list[FrameInfo] = []
    normalized_headers = [h.strip().lower() for h in (reader.fieldnames or [])]
    has_header = bool(normalized_headers)

    def _field(row: dict, keys: Sequence[str], default: str | None = None) -> str | None:
        for key in keys:
            if key in row and row.get(key) not in (None, ""):
                return str(row.get(key)).strip()
        return default

    if not has_header:
        _emit("stack_plan.csv without headers, treating first column as path", lvl="WARN", callback=progress_callback)

    for idx, row in enumerate(reader):
        if not isinstance(row, dict):
            continue
        path_text: str | None
        if has_header:
            path_text = _field(
                row,
                (
                    "file_path",
                    "filepath",
                    "file",
                    "path",
                    "filename",
                    "fits",
                    "raw",
                ),
            )
        else:
            # DictReader with no headers returns None keys; pick the first non-empty value
            values = [v for v in row.values() if v not in (None, "")]
            path_text = str(values[0]).strip() if values else None

        if not path_text:
            _emit(f"Row {idx+1}: missing file path, skipping", lvl="WARN", callback=progress_callback)
            continue

        frame_path = _resolve_path(base_dir, path_text)
        if not frame_path.is_file():
            _emit(f"Row {idx+1}: file not found {frame_path}", lvl="WARN", callback=progress_callback)
            continue

        exposure = _parse_float(
            _field(row, ("exposure", "exp", "exptime", "exposure_s"), default=1.0),
            default=1.0,
        )
        bortle = _field(row, ("bortle", "bortle_class", "sky_quality"))
        filter_name = _field(row, ("filter", "band", "channel"))
        batch_id = _field(row, ("batch_id", "batch", "session", "night"))
        order_val = _field(row, ("order", "seq", "sequence", "index"))
        order = _parse_int(order_val, default=idx)

        frames.append(
            FrameInfo(
                path=frame_path,
                exposure=exposure,
                bortle=bortle,
                filter_name=filter_name,
                batch_id=batch_id,
                order=order,
            )
        )

    _emit(f"Loaded {len(frames)} frame(s) from stack_plan.csv", callback=progress_callback)
    return frames


def _load_frame_wcs(frame: FrameInfo, *, progress_callback: ProgressCallback = None) -> bool:
    """Populate frame.wcs and frame.shape_hw if possible."""

    if not (_ASTROPY_AVAILABLE and fits and WCS):
        _emit("Astropy not available for WCS parsing", lvl="ERROR", callback=progress_callback)
        return False
    try:
        with _open_fits_safely(frame.path) as hdul:
            header = hdul[0].header
            data = hdul[0].data
    except Exception as exc:
        _emit(f"Failed to open FITS {frame.path}: {exc}", lvl="ERROR", callback=progress_callback)
        return False


    try:
        frame.wcs = WCS(header)
    except Exception:
        frame.wcs = None
    if frame.wcs is None or not getattr(frame.wcs, "is_celestial", False):
        _emit(f"No usable WCS in {frame.path}", lvl="WARN", callback=progress_callback)
        return False

    shape_hw = None
    try:
        shape = data.shape if data is not None else None
        if shape is not None:
            if len(shape) == 2:
                shape_hw = (int(shape[0]), int(shape[1]))
            elif len(shape) == 3:
                # Assume either HWC or CHW; pick the last two dims as spatial
                height, width = shape[-2], shape[-1]
                shape_hw = (int(height), int(width))
    except Exception:
        shape_hw = None
    if shape_hw is None:
        _emit(f"Cannot derive image shape for {frame.path}", lvl="WARN", callback=progress_callback)
        return False
    frame.shape_hw = shape_hw
    return True


def _extract_pixel_scale_deg(wcs_obj: object) -> float | None:
    if not (_ASTROPY_AVAILABLE and proj_plane_pixel_scales and u):
        return None
    try:
        scales = proj_plane_pixel_scales(wcs_obj)  # type: ignore[arg-type]
        vals = [abs(s.to_value(u.deg)) for s in scales]
        return float(np.mean(vals))
    except Exception:
        try:
            cdelt = np.asarray(getattr(wcs_obj, "wcs", None).cdelt, dtype=float)
            if cdelt.size >= 2:
                return float(np.mean(np.abs(cdelt)))
        except Exception:
            return None
    return None


def _frame_fov_deg(frame: FrameInfo) -> float | None:
    if frame.wcs is None or frame.shape_hw is None:
        return None
    px_scale = _extract_pixel_scale_deg(frame.wcs)
    if px_scale is None:
        return None
    h, w = frame.shape_hw
    return max(h, w) * px_scale


def _strip_wcs_distortion(wcs_obj: object) -> object:
    """Return a copy of ``wcs_obj`` without SIP/CPDIS/DET2IM distortions."""

    try:
        wcs_copy = copy.deepcopy(wcs_obj)
    except Exception:
        wcs_copy = wcs_obj
    try:
        for attr in ("sip", "cpdis", "det2im", "distortion_lookup_table"):
            if hasattr(wcs_copy, attr):
                try:
                    setattr(wcs_copy, attr, None)
                except Exception:
                    pass
        inner = getattr(wcs_copy, "wcs", None)
        if inner is not None:
            for attr in ("sip", "cpdis", "det2im"):
                if hasattr(inner, attr):
                    try:
                        setattr(inner, attr, None)
                    except Exception:
                        pass
            try:
                inner.set_pv([])
            except Exception:
                pass
    except Exception:
        pass
    return wcs_copy


def _compute_frame_footprint(frame: FrameInfo, global_wcs: object, *, progress_callback: ProgressCallback = None) -> tuple[float, float, float, float] | None:
    if frame.wcs is None or frame.shape_hw is None or not (_ASTROPY_AVAILABLE and SkyCoord):
        return None
    try:
        h, w = frame.shape_hw
        corners_x = np.array([0, w - 1, 0, w - 1], dtype=float)
        corners_y = np.array([0, 0, h - 1, h - 1], dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.filterwarnings("always", category=UserWarning, message=".*all_world2pix.*")
            warnings.filterwarnings("always", category=UserWarning, message=".*Reproject encountered WCS.*")
            sky = frame.wcs.pixel_to_world(corners_x, corners_y)  # type: ignore[call-arg]
            gx, gy = global_wcs.world_to_pixel(sky)  # type: ignore[attr-defined]
        if _has_wcs_convergence_warning(caught):
            _log_wcs_degraded(frame, progress_callback=progress_callback)
        xmin = float(np.nanmin(gx))
        xmax = float(np.nanmax(gx))
        ymin = float(np.nanmin(gy))
        ymax = float(np.nanmax(gy))
        return xmin, xmax, ymin, ymax
    except Exception:
        return None


def _clone_tile_wcs(global_wcs: object, offset_xy: tuple[int, int], shape_hw: tuple[int, int]) -> object:
    tile_wcs = _strip_wcs_distortion(global_wcs)
    try:
        offset_x, offset_y = offset_xy
        crpix = np.asarray(tile_wcs.wcs.crpix, dtype=float)  # type: ignore[attr-defined]
        tile_wcs.wcs.crpix = crpix - np.asarray([offset_x, offset_y], dtype=float)  # type: ignore[attr-defined]
        tile_wcs.wcs.naxis1 = int(shape_hw[1])  # type: ignore[attr-defined]
        tile_wcs.wcs.naxis2 = int(shape_hw[0])  # type: ignore[attr-defined]
        tile_wcs.pixel_shape = (int(shape_hw[1]), int(shape_hw[0]))  # type: ignore[attr-defined]
        tile_wcs.array_shape = (int(shape_hw[0]), int(shape_hw[1]))  # type: ignore[attr-defined]
    except Exception:
        pass
    return tile_wcs


def build_global_grid(
    frames: Iterable[FrameInfo],
    grid_size_factor: float,
    overlap_fraction: float,
    *,
    progress_callback: ProgressCallback = None,
) -> GridDefinition | None:
    """Build global WCS and the regular grid of tiles."""

    if not (_ASTROPY_AVAILABLE and _REPROJECT_AVAILABLE and WCS and u):
        _emit("Astropy/Reproject missing, cannot build grid", lvl="ERROR", callback=progress_callback)
        return None

    usable_frames: list[FrameInfo] = []
    for frame in frames:
        if frame.wcs is None or frame.shape_hw is None:
            _load_frame_wcs(frame, progress_callback=progress_callback)
        if frame.wcs is None or frame.shape_hw is None:
            continue
        usable_frames.append(frame)

    if not usable_frames:
        _emit("No frames with valid WCS found in stack_plan.csv", lvl="ERROR", callback=progress_callback)
        return None

    inputs_for_wcs = []
    fov_candidates_deg: list[float] = []
    pixel_scales_deg: list[float] = []
    for frame in usable_frames:
        if frame.wcs is None or frame.shape_hw is None:
            continue
        inputs_for_wcs.append((frame.shape_hw, frame.wcs))
        fov_val = _frame_fov_deg(frame)
        if fov_val is not None:
            fov_candidates_deg.append(fov_val)
        px_scale = _extract_pixel_scale_deg(frame.wcs)
        if px_scale is not None:
            pixel_scales_deg.append(px_scale)

    if not inputs_for_wcs:
        _emit("No usable WCS/shapes available to compute global grid", lvl="ERROR", callback=progress_callback)
        return None

    target_resolution = None
    if pixel_scales_deg:
        try:
            target_resolution = u.Quantity(np.median(pixel_scales_deg), unit=u.deg)
        except Exception:
            target_resolution = None

    try:
        global_wcs, global_shape_hw = find_optimal_celestial_wcs(
            inputs_for_wcs,
            resolution=target_resolution,
            auto_rotate=True,
            projection="TAN",
        )
    except Exception as exc:
        _emit(f"find_optimal_celestial_wcs failed ({exc}), using first frame WCS", lvl="WARN", callback=progress_callback)
        first = usable_frames[0]
        global_wcs = copy.deepcopy(first.wcs)
        global_shape_hw = first.shape_hw or (0, 0)

    if global_wcs is not None:
        global_wcs = _strip_wcs_distortion(global_wcs)

    if global_wcs is None or global_shape_hw is None:
        _emit("Global WCS calculation failed", lvl="ERROR", callback=progress_callback)
        return None

    pixel_scale_global = _extract_pixel_scale_deg(global_wcs) or (np.median(pixel_scales_deg) if pixel_scales_deg else None)
    if pixel_scale_global is None or pixel_scale_global <= 0:
        pixel_scale_global = 2.0 / 3600.0  # default 2 arcsec/pix

    median_fov_deg = np.median(fov_candidates_deg) if fov_candidates_deg else (pixel_scale_global * max(global_shape_hw))
    tile_size_deg = max(pixel_scale_global, median_fov_deg) / max(grid_size_factor, 1e-3)
    tile_size_px = int(max(32, round(tile_size_deg / pixel_scale_global)))
    overlap_fraction = max(0.0, min(0.9, float(overlap_fraction)))
    step_px = max(1, int(round(tile_size_px * (1.0 - overlap_fraction))))

    global_bounds = []
    for frame in usable_frames:
        fp = _compute_frame_footprint(frame, global_wcs, progress_callback=progress_callback)
        frame.footprint = fp
        if fp is not None:
            global_bounds.append(fp)
    if global_bounds:
        min_x = math.floor(min(b[0] for b in global_bounds))
        max_x = math.ceil(max(b[1] for b in global_bounds))
        min_y = math.floor(min(b[2] for b in global_bounds))
        max_y = math.ceil(max(b[3] for b in global_bounds))
    else:
        min_x = min_y = 0
        max_y, max_x = global_shape_hw

    tiles: list[GridTile] = []
    y0 = min_y
    tile_id = 1
    while y0 < max_y:
        x0 = min_x
        while x0 < max_x:
            bbox_xmin = int(x0)
            bbox_xmax = int(min(x0 + tile_size_px, max_x))
            bbox_ymin = int(y0)
            bbox_ymax = int(min(y0 + tile_size_px, max_y))
            shape_hw = (bbox_ymax - bbox_ymin, bbox_xmax - bbox_xmin)
            if shape_hw[0] <= 0 or shape_hw[1] <= 0:
                x0 += step_px
                continue
            tile_wcs = _clone_tile_wcs(global_wcs, (bbox_xmin, bbox_ymin), shape_hw)
            tiles.append(GridTile(tile_id=tile_id, bbox=(bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax), wcs=tile_wcs))
            tile_id += 1
            x0 += step_px
        y0 += step_px

    _emit(f"Global grid ready: {len(tiles)} tile(s), tile_size_px={tile_size_px}, overlap={overlap_fraction:.3f}", callback=progress_callback)
    return GridDefinition(
        global_wcs=global_wcs,
        global_shape_hw=(int(global_shape_hw[0]), int(global_shape_hw[1])),
        tile_size_px=tile_size_px,
        overlap_fraction=overlap_fraction,
        tiles=tiles,
    )


def assign_frames_to_tiles(frames: Iterable[FrameInfo], tiles: Iterable[GridTile], *, progress_callback: ProgressCallback = None) -> None:
    """Assign frames to tiles based on bounding-box intersection."""

    for frame in frames:
        if frame.footprint is None:
            continue
        fx0, fx1, fy0, fy1 = frame.footprint
        for tile in tiles:
            tx0, tx1, ty0, ty1 = tile.bbox
            if fx1 < tx0 or fx0 > tx1 or fy1 < ty0 or fy0 > ty1:
                continue
            tile.frames.append(frame)
    for tile in tiles:
        _emit(f"Tile {tile.tile_id}: assigned {len(tile.frames)} frame(s)", lvl="DEBUG", callback=progress_callback)


def _load_image_with_optional_alpha(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if not (_ASTROPY_AVAILABLE and fits):
        raise RuntimeError("Astropy FITS support unavailable")
    with _open_fits_safely(path) as hdul:
        data = hdul[0].data
        alpha_hdu = hdul["ALPHA"] if "ALPHA" in hdul else None
        alpha = None
        if alpha_hdu is not None and alpha_hdu.data is not None:
            try:
                alpha = np.asarray(alpha_hdu.data, dtype=np.float32)
            except Exception:
                alpha = None
    arr = _ensure_hwc_array(data)
    weights = None
    if alpha is not None:
        alpha = np.squeeze(alpha)
        if alpha.ndim == 2:
            weights = np.clip(alpha, 0.0, 255.0) / 255.0
        elif alpha.ndim == 3 and alpha.shape[0] in (1, 3):
            weights = np.clip(np.moveaxis(alpha, 0, -1), 0.0, 255.0) / 255.0
    return arr, weights


def _reproject_frame_to_tile(
    frame: FrameInfo,
    tile: GridTile,
    tile_shape_hw: tuple[int, int],
    *,
    progress_callback: ProgressCallback = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not (_REPROJECT_AVAILABLE and reproject_interp):
        return None, None
    if frame.wcs is None:
        return None, None
    try:
        data, alpha_weights = _load_image_with_optional_alpha(frame.path)
    except Exception:
        return None, None

    channels = data.shape[-1] if data.ndim == 3 else 1
    reproj_channels = []
    footprints = []
    degraded = False
    for c in range(channels):
        arr_2d = data[..., c] if data.ndim == 3 else data
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.filterwarnings("always", category=UserWarning, message=".*all_world2pix.*")
                warnings.filterwarnings("always", category=UserWarning, message=".*Reproject encountered WCS.*")
                reproj_arr, footprint = reproject_interp(
                    (arr_2d, frame.wcs),
                    output_projection=tile.wcs,
                    shape_out=(tile_shape_hw[0], tile_shape_hw[1]),
                    return_footprint=True,
                )
            if _has_wcs_convergence_warning(caught):
                degraded = True
        except Exception:
            return None, None
        reproj_channels.append(reproj_arr.astype(np.float32, copy=False))
        footprints.append(footprint.astype(np.float32, copy=False))

    if degraded:
        _log_wcs_degraded(frame, progress_callback=progress_callback)

    reproj_stack = np.stack(reproj_channels, axis=-1)
    footprint_combined = np.nanmax(np.stack(footprints, axis=0), axis=0)
    if alpha_weights is not None:
        try:
            if alpha_weights.ndim == 2:
                footprint_combined *= alpha_weights
            elif alpha_weights.ndim == 3 and alpha_weights.shape[-1] == reproj_stack.shape[-1]:
                footprint_combined *= np.nanmax(alpha_weights, axis=-1)
        except Exception:
            pass
    return reproj_stack, footprint_combined


def _compute_frame_weight(frame: FrameInfo, patch: np.ndarray, footprint: np.ndarray) -> float:
    exposure_w = max(frame.exposure, 1e-3)
    bortle_w = 1.0
    if frame.bortle:
        try:
            bortle_num = float("".join(ch for ch in frame.bortle if (ch.isdigit() or ch == ".")))
            bortle_w = 1.0 / max(1.0, bortle_num)
        except Exception:
            bortle_w = 1.0
    finite = np.isfinite(patch)
    if not np.any(finite):
        return exposure_w * bortle_w
    try:
        median_val = float(np.nanmedian(patch[finite]))
        noise = float(np.nanstd(patch[finite]))
        snr = median_val / max(noise, 1e-6)
    except Exception:
        snr = 1.0
    return exposure_w * bortle_w * max(snr, 1e-3)


def _normalize_patches(
    patches: list[np.ndarray],
    reference_median: float | None = None,
) -> tuple[list[np.ndarray], float]:
    """Scale each patch to a common median.

    When ``reference_median`` is provided it is reused so that multiple chunks
    share the same photometric reference. The value used is returned so callers
    can pass it back on the next invocation.
    """

    if not patches:
        return [], reference_median if reference_median is not None else 1.0
    ref_median = reference_median
    if ref_median is None:
        ref_patch = patches[0]
        finite = np.isfinite(ref_patch)
        ref_median = float(np.nanmedian(ref_patch[finite])) if np.any(finite) else 1.0
        ref_median = ref_median if math.isfinite(ref_median) and ref_median != 0 else 1.0

    normalized: list[np.ndarray] = []
    for patch in patches:
        patch_norm = patch
        finite_patch = np.isfinite(patch_norm)
        med = float(np.nanmedian(patch_norm[finite_patch])) if np.any(finite_patch) else ref_median
        med = med if math.isfinite(med) and med != 0 else ref_median
        try:
            scale = ref_median / med
        except Exception:
            scale = 1.0
        patch_norm = patch_norm * scale
        normalized.append(patch_norm.astype(np.float32, copy=False))
    return normalized, float(ref_median)


def _stack_weighted_patches(
    patches: list[np.ndarray],
    weights: list[np.ndarray],
    config: GridModeConfig,
    *,
    reference_median: float | None = None,
    return_weight_sum: bool = False,
    return_ref_median: bool = False,
) -> np.ndarray | tuple | None:
    """Stack patches with optional sigma clipping and shared photometric anchor."""

    if not patches:
        return None
    # Ensure shapes match and promote to float32
    normalized, ref_median_used = _normalize_patches(patches, reference_median)
    data_stack = np.stack(normalized, axis=0).astype(np.float32, copy=False)
    weight_stack = np.stack(weights, axis=0).astype(np.float32, copy=False)
    data_stack = np.where(weight_stack > 0, data_stack, np.nan)

    rejection = config.stack_reject_algo.lower().strip()
    data_for_combine = data_stack
    if rejection in {"kappa_sigma", "kappa"} and _reject_outliers_kappa_sigma:
        data_for_combine, _ = _reject_outliers_kappa_sigma(
            data_stack,
            config.stack_kappa_low,
            config.stack_kappa_high,
            progress_callback=None,
        )
    elif rejection in {"winsorized_sigma_clip", "winsor"} and _reject_outliers_winsorized_sigma_clip:
        data_for_combine, _ = _reject_outliers_winsorized_sigma_clip(
            data_stack,
            config.winsor_limits,
            config.stack_kappa_low,
            config.stack_kappa_high,
            progress_callback=None,
            max_workers=1,
        )

    data_masked = np.nan_to_num(data_for_combine, nan=0.0)
    finite_mask = np.isfinite(data_for_combine)
    weight_effective = np.where(finite_mask, weight_stack, 0.0)
    weight_sum = np.sum(weight_effective, axis=0)
    valid_positions = np.any(finite_mask, axis=0)
    if not np.any(valid_positions):
        empty = np.zeros(data_for_combine.shape[1:], dtype=np.float32)
        outputs = [empty]
    else:
        if config.stack_final_combine.lower().strip() == "median":
            median_input = np.where(valid_positions[np.newaxis, ...], data_for_combine, 0.0)
            result = np.nanmedian(median_input, axis=0)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.sum(data_masked * weight_effective, axis=0) / np.clip(weight_sum, 1e-6, None)
        outputs = [result.astype(np.float32, copy=False)]

    if return_weight_sum:
        outputs.append(weight_sum.astype(np.float32, copy=False))
    if return_ref_median:
        outputs.append(ref_median_used)
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def process_tile(tile: GridTile, output_dir: Path, config: GridModeConfig, *, progress_callback: ProgressCallback = None) -> Path | None:
    """Process a single tile and write it to disk."""

    if not tile.frames:
        _emit(f"Tile {tile.tile_id}: no frames, skipping", lvl="WARN", callback=progress_callback)
        return None
    if not (_ASTROPY_AVAILABLE and fits):
        _emit(f"Tile {tile.tile_id}: Astropy unavailable, cannot save tile", lvl="ERROR", callback=progress_callback)
        return None

    tile_shape = (tile.bbox[3] - tile.bbox[2], tile.bbox[1] - tile.bbox[0])
    aligned_patches: list[np.ndarray] = []
    weight_maps: list[np.ndarray] = []
    # Keep the working set bounded so very large stacks (thousands of frames)
    # do not explode memory usage.
    try:
        budget_mb = float(getattr(config, "stack_chunk_budget_mb", 512.0) or 512.0)
    except Exception:
        budget_mb = 512.0
    if budget_mb <= 0:
        budget_mb = 512.0
    target_chunk_bytes = max(64.0, budget_mb) * 1024 * 1024
    per_frame_bytes: float | None = None
    chunk_limit = len(tile.frames)
    running_sum: np.ndarray | None = None
    running_weight: np.ndarray | None = None
    reference_median: float | None = None
    chunk_failed = False

    def flush_chunk() -> None:
        """Stack the current chunk and fold it into the running accumulator."""

        nonlocal running_sum, running_weight, reference_median, chunk_failed
        if chunk_failed or not aligned_patches:
            aligned_patches.clear()
            weight_maps.clear()
            return
        res = _stack_weighted_patches(
            aligned_patches,
            weight_maps,
            config,
            reference_median=reference_median,
            return_weight_sum=True,
            return_ref_median=True,
        )
        if not isinstance(res, tuple) or len(res) < 2:
            chunk_failed = True
            _emit(f"Tile {tile.tile_id}: stacking failed for chunk", lvl="ERROR", callback=progress_callback)
        else:
            stacked_chunk = res[0]
            weight_sum = res[1]
            ref_med_out = res[2] if len(res) >= 3 else reference_median
            if stacked_chunk is None or weight_sum is None:
                chunk_failed = True
                _emit(f"Tile {tile.tile_id}: stacking failed for chunk", lvl="ERROR", callback=progress_callback)
            else:
                if ref_med_out is not None:
                    reference_median = ref_med_out
                weighted = stacked_chunk * weight_sum
                if running_sum is None:
                    running_sum = weighted
                    running_weight = weight_sum
                else:
                    running_sum = running_sum + weighted
                    running_weight = running_weight + weight_sum
        aligned_patches.clear()
        weight_maps.clear()

    for frame in tile.frames:
        patch, footprint = _reproject_frame_to_tile(frame, tile, tile_shape, progress_callback=progress_callback)
        if patch is None or footprint is None:
            _emit(f"Tile {tile.tile_id}: reprojection failed for {frame.path.name}", lvl="WARN", callback=progress_callback)
            continue
        weight_scalar = _compute_frame_weight(frame, patch, footprint)
        weight_map = np.clip(np.asarray(footprint, dtype=np.float32), 0.0, 1.0)
        if patch.ndim == 3 and weight_map.ndim == 2:
            weight_map = np.repeat(weight_map[..., np.newaxis], patch.shape[-1], axis=2)
        weight_map *= weight_scalar

        if per_frame_bytes is None:
            per_frame_bytes = float(patch.nbytes + weight_map.nbytes)
            per_frame_bytes = per_frame_bytes if per_frame_bytes > 0 else float(patch.size * 4)
            chunk_limit = int(target_chunk_bytes // max(per_frame_bytes, 1.0))
            if chunk_limit <= 0:
                chunk_limit = 1
            chunk_limit = min(chunk_limit, len(tile.frames))
            # Cap the chunk to avoid stacking too many frames at once even if the budget allows it.
            if len(tile.frames) > 256:
                chunk_limit = min(chunk_limit, 256)
            if len(tile.frames) > chunk_limit:
                est_chunk_mb = (per_frame_bytes * chunk_limit) / (1024 ** 2)
                _emit(
                    f"Tile {tile.tile_id}: chunked stacking enabled "
                    f"({len(tile.frames)} frames -> chunk_size={chunk_limit}, ~{est_chunk_mb:.1f} MB)",
                    callback=progress_callback,
                )
        aligned_patches.append(patch)
        weight_maps.append(weight_map)
        if len(aligned_patches) >= chunk_limit:
            flush_chunk()
            if chunk_failed:
                break

    flush_chunk()
    if chunk_failed or running_sum is None or running_weight is None:
        _emit(f"Tile {tile.tile_id}: no usable patches", lvl="WARN", callback=progress_callback)
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        stacked = running_sum / np.clip(running_weight, 1e-6, None)
    stacked = np.where(np.isfinite(stacked), stacked, 0.0).astype(np.float32, copy=False)

    tiles_dir = output_dir / "tiles"
    try:
        tiles_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    output_path = tiles_dir / f"tile_{tile.tile_id:04d}.fits"
    header = None
    try:
        header = tile.wcs.to_header() if hasattr(tile.wcs, "to_header") else None  # type: ignore[attr-defined]
    except Exception:
        header = None
    try:
        fits.writeto(output_path, stacked.astype(np.float32), header=header, overwrite=True)
    except Exception as exc:
        _emit(f"Tile {tile.tile_id}: failed to write FITS ({exc})", lvl="ERROR", callback=progress_callback)
        return None
    tile.output_path = output_path
    _emit(f"Tile {tile.tile_id}: saved to {output_path}", callback=progress_callback)
    return output_path


def _normalize_background(mosaic: np.ndarray, weight_map: np.ndarray) -> np.ndarray:
    finite = (weight_map > 0) & np.isfinite(mosaic)
    if not np.any(finite):
        return mosaic
    if mosaic.ndim == 3:
        normed = mosaic.copy()
        for c in range(mosaic.shape[-1]):
            channel_mask = finite[..., c] if finite.ndim == mosaic.ndim else finite
            if not np.any(channel_mask):
                continue
            median_val = float(np.nanmedian(mosaic[..., c][channel_mask]))
            if math.isfinite(median_val):
                normed[..., c] -= median_val
        return normed
    median_val = float(np.nanmedian(mosaic[finite]))
    if math.isfinite(median_val):
        return mosaic - median_val
    return mosaic


def grid_post_equalize_rgb(
    mosaic: np.ndarray,
    weight_sum: np.ndarray,
    *,
    background_percentile: float = 30.0,
    progress_callback: ProgressCallback = None,
) -> np.ndarray:
    """Equalize RGB channels using the classic median logic on valid pixels."""

    arr = np.asarray(mosaic, dtype=np.float32)
    weights = np.asarray(weight_sum, dtype=np.float32) if weight_sum is not None else None
    weight_shape = getattr(weights, "shape", None)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        _emit(
            f"RGB equalization: skipped (non-RGB mosaic with shape={arr.shape})",
            lvl="DEBUG",
            callback=progress_callback,
        )
        return arr

    try:
        arr_work = np.array(arr, dtype=np.float32, copy=True)
        weight_mask = None
        if weights is not None:
            try:
                if weights.ndim == 2:
                    weights = weights[..., np.newaxis]
                if weights.ndim == 3 and weights.shape[-1] == 1:
                    weights = np.repeat(weights, 3, axis=2)
                weights = np.broadcast_to(weights, arr_work.shape)
                weight_mask = (weights > 0) & np.isfinite(weights)
            except Exception:
                weight_mask = None

        valid = np.isfinite(arr_work)
        if weight_mask is not None:
            valid = valid & weight_mask
        valid_pixels = int(np.count_nonzero(valid))
        if valid_pixels == 0:
            _emit(
                f"RGB equalization: skipped (no valid pixels, weight_shape={weight_shape})",
                lvl="WARN",
                callback=progress_callback,
            )
            return arr

        arr_work = np.where(valid, arr_work, np.nan)

        medians: list[float] = []
        counts: list[int] = []
        for c in range(3):
            vals = arr_work[..., c]
            vals = vals[np.isfinite(vals)]
            counts.append(int(vals.size))
            medians.append(float(np.median(vals)) if vals.size else float("nan"))

        medians_arr = np.asarray(medians, dtype=np.float32)
        if any(count == 0 for count in counts):
            _emit(
                "RGB equalization: skipped (channel missing valid pixels)",
                lvl="WARN",
                callback=progress_callback,
            )
            return arr
        finite_chan = np.isfinite(medians_arr) & (medians_arr > 0)
        if not np.any(finite_chan):
            _emit(
                "RGB equalization: skipped (no finite positive channel medians)",
                lvl="WARN",
                callback=progress_callback,
            )
            return arr
        target = float(np.nanmedian(medians_arr[finite_chan]))
        if not math.isfinite(target) or abs(target) < 1e-6:
            _emit(
                f"RGB equalization: skipped (invalid target median={target})",
                lvl="WARN",
                callback=progress_callback,
            )
            return arr

        try:
            helper_used = bool(equalize_rgb_medians_inplace)
            if equalize_rgb_medians_inplace:
                equalized = arr_work.copy()
                gain_r, gain_g, gain_b, target_used = equalize_rgb_medians_inplace(equalized)
                gains = np.array([gain_r, gain_g, gain_b], dtype=np.float32)
                if not math.isfinite(target_used):
                    target_used = target
                target = target_used
            else:
                gains = np.ones(3, dtype=np.float32)
                gains[finite_chan] = target / medians_arr[finite_chan]
                equalized = arr_work * gains.reshape((1, 1, 3))
        except Exception as exc:
            _emit(f"RGB equalization: skipped (error during equalization: {exc})", lvl="WARN", callback=progress_callback)
            return arr

        _emit(
            (
                "RGB equalization: applied "
                + ("(classic helper); " if helper_used else "(manual medians); ")
                + f"gains=({gains[0]:.6f},{gains[1]:.6f},{gains[2]:.6f}), "
                + f"medians=({medians_arr[0]:.6g},{medians_arr[1]:.6g},{medians_arr[2]:.6g}), "
                + f"counts={counts}, target={target:.6g}, weight_shape={weight_shape}"
            ),
            lvl="INFO",
            callback=progress_callback,
        )
        return equalized
    except Exception as exc:
        _emit(f"RGB equalization: skipped (unexpected error: {exc})", lvl="WARN", callback=progress_callback)
        return arr


def build_tile_overlap_graph(tiles: List[TilePhotometryInfo], mosaic_shape_hw: tuple[int, int]) -> List[TileOverlap]:
    overlaps: List[TileOverlap] = []
    if not tiles:
        return overlaps
    H_m, W_m = mosaic_shape_hw
    for idx_a, tile_a in enumerate(tiles):
        bbox_a = tile_a.bbox
        for idx_b in range(idx_a + 1, len(tiles)):
            tile_b = tiles[idx_b]
            bbox_b = tile_b.bbox
            x0 = max(bbox_a[0], bbox_b[0], 0)
            y0 = max(bbox_a[2], bbox_b[2], 0)
            x1 = min(bbox_a[1], bbox_b[1], W_m)
            y1 = min(bbox_a[3], bbox_b[3], H_m)
            if x1 <= x0 or y1 <= y0:
                continue
            xa0 = max(0, x0 - bbox_a[0])
            ya0 = max(0, y0 - bbox_a[2])
            xa1 = min(tile_a.data.shape[1], x1 - bbox_a[0])
            ya1 = min(tile_a.data.shape[0], y1 - bbox_a[2])
            xb0 = max(0, x0 - bbox_b[0])
            yb0 = max(0, y0 - bbox_b[2])
            xb1 = min(tile_b.data.shape[1], x1 - bbox_b[0])
            yb1 = min(tile_b.data.shape[0], y1 - bbox_b[2])
            if xa1 <= xa0 or ya1 <= ya0 or xb1 <= xb0 or yb1 <= yb0:
                continue
            overlaps.append(
                TileOverlap(
                    tile_a=tile_a.tile_id,
                    tile_b=tile_b.tile_id,
                    global_bbox=(x0, x1, y0, y1),
                    slice_a=(slice(ya0, ya1), slice(xa0, xa1)),
                    slice_b=(slice(yb0, yb1), slice(xb0, xb1)),
                    sample_count=(xa1 - xa0) * (ya1 - ya0),
                )
            )
    return overlaps


def _fit_overlap_regression(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    min_samples: int = 16,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return per-channel slope/offset fitting B ≈ slope * A + offset."""

    A = np.asarray(patch_a, dtype=np.float32)
    B = np.asarray(patch_b, dtype=np.float32)
    if A.ndim == 2:
        A = A[..., np.newaxis]
    if B.ndim == 2:
        B = B[..., np.newaxis]
    mask_a_arr = np.asarray(mask_a, dtype=bool)
    mask_b_arr = np.asarray(mask_b, dtype=bool)
    if mask_a_arr.ndim == 2 and A.ndim == 3:
        mask_a_arr = np.repeat(mask_a_arr[..., np.newaxis], A.shape[-1], axis=2)
    if mask_b_arr.ndim == 2 and B.ndim == 3:
        mask_b_arr = np.repeat(mask_b_arr[..., np.newaxis], B.shape[-1], axis=2)
    channels = A.shape[-1]
    slopes = np.ones(channels, dtype=np.float32)
    offsets = np.zeros(channels, dtype=np.float32)
    total_samples = 0
    total_valid_pairs = 0
    for c in range(channels):
        valid = mask_a_arr[..., c] & mask_b_arr[..., c] & np.isfinite(A[..., c]) & np.isfinite(B[..., c])
        valid_count = int(np.count_nonzero(valid))
        total_valid_pairs += valid_count
        if valid_count == 0:
            continue
        vals_a = A[..., c][valid]
        vals_b = B[..., c][valid]
        if vals_a.size < min_samples:
            continue
        total_samples += int(vals_a.size)
        x = vals_a
        y = vals_b
        A_mat = np.vstack([x, np.ones_like(x)]).T
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
            slope, intercept = coeffs
            pred = slope * x + intercept
            resid = y - pred
            std = float(np.std(resid))
            if math.isfinite(std) and std > 0:
                keep = np.abs(resid) <= 3.0 * std
                if np.any(~keep):
                    x = x[keep]
                    y = y[keep]
                    if x.size >= min_samples:
                        A_mat = np.vstack([x, np.ones_like(x)]).T
                        coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
                        slope, intercept = coeffs
        except Exception:
            slope, intercept = 1.0, 0.0
        slope = float(slope)
        intercept = float(intercept)
        if not math.isfinite(slope) or abs(slope) < 1e-6:
            slope = 1.0
        if not math.isfinite(intercept):
            intercept = 0.0
        slopes[c] = slope
        offsets[c] = intercept
    return slopes, offsets, total_samples, total_valid_pairs


def _solve_global_gain_offset(
    tiles: List[TilePhotometryInfo],
    overlaps: List[TileOverlap],
    overlap_fits: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int]],
    *,
    max_iters: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    if not tiles:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    channels = tiles[0].data.shape[-1]
    gains = np.ones((len(tiles), channels), dtype=np.float32)
    offsets = np.zeros((len(tiles), channels), dtype=np.float32)
    id_to_idx = {t.tile_id: idx for idx, t in enumerate(tiles)}
    degree = np.zeros(len(tiles), dtype=np.int32)
    for ov in overlaps:
        degree[id_to_idx[ov.tile_a]] += 1
        degree[id_to_idx[ov.tile_b]] += 1
    ref_idx = int(np.argmax(degree)) if len(degree) else 0

    for _ in range(max_iters):
        sum_g = np.zeros_like(gains)
        sum_o = np.zeros_like(offsets)
        counts = np.zeros(len(tiles), dtype=np.float32)
        for ov in overlaps:
            fit = overlap_fits.get((ov.tile_a, ov.tile_b))
            if not fit:
                continue
            slope, intercept, samples = fit
            if samples <= 0:
                continue
            slope = np.asarray(slope, dtype=np.float32)
            intercept = np.asarray(intercept, dtype=np.float32)
            slope = np.nan_to_num(slope, nan=1.0, posinf=1.0, neginf=1.0)
            intercept = np.nan_to_num(intercept, nan=0.0, posinf=0.0, neginf=0.0)
            idx_a = id_to_idx[ov.tile_a]
            idx_b = id_to_idx[ov.tile_b]
            slope_safe = np.where(np.abs(slope) < 1e-3, 1.0, slope)
            weight = max(samples, 1)
            sugg_g_b = gains[idx_a] / slope_safe
            sugg_o_b = offsets[idx_a] - sugg_g_b * intercept
            sum_g[idx_b] += sugg_g_b * weight
            sum_o[idx_b] += sugg_o_b * weight
            counts[idx_b] += weight

            sugg_g_a = gains[idx_b] * slope_safe
            sugg_o_a = offsets[idx_b] + gains[idx_b] * intercept
            sum_g[idx_a] += sugg_g_a * weight
            sum_o[idx_a] += sugg_o_a * weight
            counts[idx_a] += weight

        updated = False
        for idx in range(len(tiles)):
            if idx == ref_idx or counts[idx] <= 0:
                continue
            avg_g = sum_g[idx] / counts[idx]
            avg_o = sum_o[idx] / counts[idx]
            gains[idx] = 0.6 * gains[idx] + 0.4 * avg_g
            offsets[idx] = 0.6 * offsets[idx] + 0.4 * avg_o
            updated = True
        if not updated:
            break

    gains = np.clip(np.nan_to_num(gains, nan=1.0, posinf=1.0, neginf=1.0), 0.01, 100.0)
    offsets = np.nan_to_num(offsets, nan=0.0, posinf=0.0, neginf=0.0)
    return gains, offsets


def _build_overlap_blend_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create smooth masks wA, wB with wA + wB = 1 where data is valid."""

    valid_a = mask_a if mask_a.ndim == 2 else np.any(mask_a, axis=-1)
    valid_b = mask_b if mask_b.ndim == 2 else np.any(mask_b, axis=-1)
    valid_a_f = valid_a.astype(np.float32, copy=False)
    valid_b_f = valid_b.astype(np.float32, copy=False)
    h, w = valid_a_f.shape
    if h == 0 or w == 0:
        return np.zeros_like(valid_a_f), np.zeros_like(valid_b_f)
    if w >= h:
        ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)
        wB = np.broadcast_to(ramp, (h, w)).copy()
    else:
        ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
        wB = np.broadcast_to(ramp, (h, w)).copy()
    wA = 1.0 - wB
    wA *= valid_a_f
    wB *= valid_b_f
    total = wA + wB
    wA = np.where(total > 0, wA / np.clip(total, 1e-6, None), 0.0)
    wB = np.where(total > 0, wB / np.clip(total, 1e-6, None), 0.0)
    wA *= (total > 0)
    wB *= (total > 0)
    return wA.astype(np.float32, copy=False), wB.astype(np.float32, copy=False)


def _blend_overlap_region(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    use_pyramid: bool = True,
    max_levels: int = 4,
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """Blend two overlapping patches; returns blended patch, weight, and pyramid flag."""

    wA, wB = _build_overlap_blend_masks(mask_a, mask_b)
    total_w = wA + wB
    if not np.any(total_w > 0):
        return None, None, False
    A = np.asarray(patch_a, dtype=np.float32)
    B = np.asarray(patch_b, dtype=np.float32)
    if A.ndim == 2:
        A = A[..., np.newaxis]
    if B.ndim == 2:
        B = B[..., np.newaxis]
    if A.shape != B.shape:
        min_h = min(A.shape[0], B.shape[0])
        min_w = min(A.shape[1], B.shape[1])
        A = A[:min_h, :min_w, ...]
        B = B[:min_h, :min_w, ...]
        wA = wA[:min_h, :min_w]
        wB = wB[:min_h, :min_w]
        total_w = total_w[:min_h, :min_w]
    mask_a_full = mask_a if mask_a.ndim == 3 else np.repeat(mask_a[..., np.newaxis], A.shape[-1], axis=2)
    mask_b_full = mask_b if mask_b.ndim == 3 else np.repeat(mask_b[..., np.newaxis], A.shape[-1], axis=2)
    A_clean = np.where(mask_a_full, A, 0.0)
    B_clean = np.where(mask_b_full, B, 0.0)
    used_pyramid = False
    blended = None
    if use_pyramid and min(A.shape[0], A.shape[1]) >= 8 and min(B.shape[0], B.shape[1]) >= 8:
        try:
            LA = build_laplacian_pyramid(A_clean, max_levels)
            LB = build_laplacian_pyramid(B_clean, max_levels)
            wA_pyr = build_gaussian_pyramid(wA[..., np.newaxis], max_levels)
            wB_pyr = build_gaussian_pyramid(wB[..., np.newaxis], max_levels)
            min_len = min(len(LA), len(LB), len(wA_pyr), len(wB_pyr))
            L_blend: List[np.ndarray] = []
            for k in range(min_len):
                wa = wA_pyr[k]
                wb = wB_pyr[k]
                mask_sum = wa + wb
                wa_norm = np.where(mask_sum > 0, wa / np.clip(mask_sum, 1e-6, None), 0.0)
                wb_norm = np.where(mask_sum > 0, wb / np.clip(mask_sum, 1e-6, None), 0.0)
                lb_a = LA[k]
                lb_b = LB[k]
                if lb_b.shape != lb_a.shape:
                    lb_b = lb_b[: lb_a.shape[0], : lb_a.shape[1], ...]
                    wb_norm = wb_norm[: lb_a.shape[0], : lb_a.shape[1], ...]
                    wa_norm = wa_norm[: lb_a.shape[0], : lb_a.shape[1], ...]
                L_blend.append(lb_a * wa_norm + lb_b * wb_norm)
            if L_blend:
                blended = reconstruct_from_laplacian(L_blend)
                used_pyramid = True
        except Exception:
            used_pyramid = False
            blended = None
    if blended is None:
        total_w_expanded = total_w[..., np.newaxis]
        blended = np.where(
            total_w_expanded > 0,
            (A_clean * wA[..., np.newaxis] + B_clean * wB[..., np.newaxis]) / np.clip(total_w_expanded, 1e-6, None),
            0.0,
        )
    weight_out = total_w.astype(np.float32, copy=False)
    return blended.astype(np.float32, copy=False), weight_out, used_pyramid


def assemble_tiles(
    grid: GridDefinition,
    tiles: Iterable[GridTile],
    output_path: Path,
    *,
    save_final_as_uint16: bool = False,
    legacy_rgb_cube: bool = False,
    grid_rgb_equalize: bool = True,
    progress_callback: ProgressCallback = None,
) -> Path | None:
    """Assemble processed tiles into the final mosaic without global reprojection."""

    if not (_ASTROPY_AVAILABLE and fits):
        _emit(
            f"Astropy unavailable, cannot assemble mosaic (fallback to classic?). _ASTROPY_AVAILABLE={_ASTROPY_AVAILABLE}, fits_present={bool(fits)}",
            lvl="ERROR",
            callback=progress_callback,
        )
        return None

    tiles_seq = list(tiles)
    tiles_list = [t for t in tiles_seq if t.output_path and t.output_path.is_file()]
    missing_outputs = max(0, len(tiles_seq) - len(tiles_list))
    _emit(
        (
            "Assembly: scanning tiles for output files "
            f"(received={len(tiles_seq)}, with_files={len(tiles_list)}, missing_outputs={missing_outputs})"
        ),
        lvl="DEBUG",
        callback=progress_callback,
    )
    if not tiles_list:
        sample_paths = [str(t.output_path) for t in tiles_seq if getattr(t, "output_path", None)]
        sample_preview = ", ".join(sample_paths[:3]) if sample_paths else "<no paths>"
        _emit(
            (
                "No tiles to assemble; none of the expected outputs were found. "
                f"Assembly summary: received={len(tiles_seq)}, with_files=0, io_fail=0, "
                f"channel_mismatch=0, empty_mask=0, kept=0. Sample output paths: {sample_preview}"
            ),
            lvl="ERROR",
            callback=progress_callback,
        )
        return None

    tile_infos: list[TilePhotometryInfo] = []
    channels: int | None = None
    io_failures = 0
    channel_mismatches = 0
    empty_masks = 0
    for t in tiles_list:
        try:
            with _open_fits_safely(t.output_path) as hdul:
                raw_data = hdul[0].data
                if raw_data is None:
                    raise ValueError("empty data")
                data = _ensure_hwc_array(raw_data)
        except Exception as exc:
            io_failures += 1
            _emit(
                f"Assembly: failed to read {t.output_path} ({exc})",
                lvl="WARN",
                callback=progress_callback,
            )
            continue

        c = 1 if data.ndim == 2 else data.shape[-1]
        if channels is None:
            channels = c
        if c != channels:
            channel_mismatches += 1
            _emit(
                f"Assembly: tile {t.tile_id} channel mismatch ({c} != {channels}), skipping",
                lvl="WARN",
                callback=progress_callback,
            )
            continue
        mask = compute_valid_mask(data)
        if not np.any(mask):
            empty_masks += 1
            _emit(
                f"Assembly: tile {t.tile_id} has empty valid-mask, skipping",
                lvl="WARN",
                callback=progress_callback,
            )
            continue
        tile_infos.append(TilePhotometryInfo(tile_id=t.tile_id, bbox=t.bbox, data=data, mask=mask))

    if not tile_infos:
        _emit(
            (
                "Unable to read any tile for assembly. "
                f"Assembly summary: attempted={len(tiles_list)}, io_fail={io_failures}, "
                f"channel_mismatch={channel_mismatches}, empty_mask={empty_masks}, kept=0"
            ),
            lvl="ERROR",
            callback=progress_callback,
        )
        return None

    _emit(
        (
            "Assembly summary: "
            f"attempted={len(tiles_list)}, io_fail={io_failures}, channel_mismatch={channel_mismatches}, "
            f"empty_mask={empty_masks}, kept={len(tile_infos)}"
        ),
        callback=progress_callback,
    )

    channels = channels or 1
    mosaic_shape = (grid.global_shape_hw[0], grid.global_shape_hw[1], channels)
    mosaic_sum = np.zeros(mosaic_shape, dtype=np.float32)
    weight_sum = np.zeros(mosaic_shape, dtype=np.float32)
    H_m, W_m, _ = mosaic_sum.shape
    _emit(f"Photometry: loaded {len(tile_infos)} tiles for assembly", callback=progress_callback)

    overlaps = build_tile_overlap_graph(tile_infos, (H_m, W_m))
    _emit(f"Photometry: built overlap graph with {len(overlaps)} edges", callback=progress_callback)
    info_by_id = {info.tile_id: info for info in tile_infos}

    overlap_fits: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int]] = {}
    regression_failures = 0
    for ov in overlaps:
        info_a = info_by_id.get(ov.tile_a)
        info_b = info_by_id.get(ov.tile_b)
        if info_a is None or info_b is None:
            continue
        patch_a = info_a.data[ov.slice_a]
        patch_b = info_b.data[ov.slice_b]
        mask_a = info_a.mask[ov.slice_a]
        mask_b = info_b.mask[ov.slice_b]
        slope, intercept, samples, valid_pairs = _fit_overlap_regression(patch_a, patch_b, mask_a, mask_b)
        if valid_pairs <= 0:
            regression_failures += 1
            _emit(
                f"Overlap {info_a.tile_id}-{info_b.tile_id} skipped (no finite pixels)",
                lvl="WARN",
                callback=progress_callback,
            )
            overlap_fits[(ov.tile_a, ov.tile_b)] = (np.ones(channels, dtype=np.float32), np.zeros(channels, dtype=np.float32), 0)
            continue
        if samples <= 0:
            regression_failures += 1
            _emit(
                f"Photometry: overlap {info_a.tile_id}-{info_b.tile_id} regression skipped (insufficient valid samples)",
                lvl="WARN",
                callback=progress_callback,
            )
            overlap_fits[(ov.tile_a, ov.tile_b)] = (np.ones(channels, dtype=np.float32), np.zeros(channels, dtype=np.float32), 0)
            continue
        slope = np.nan_to_num(slope, nan=1.0, posinf=1.0, neginf=1.0)
        intercept = np.nan_to_num(intercept, nan=0.0, posinf=0.0, neginf=0.0)
        overlap_fits[(ov.tile_a, ov.tile_b)] = (slope, intercept, samples)
    if regression_failures:
        _emit(f"Photometry: {regression_failures} overlap regressions fell back to neutral gains", lvl="WARN", callback=progress_callback)

    gains, offsets = _solve_global_gain_offset(tile_infos, overlaps, overlap_fits)
    _emit(f"Photometry: solved global gain/offset for {len(tile_infos)} tiles", callback=progress_callback)

    for idx, info in enumerate(tile_infos):
        gain = gains[idx] if gains.size else np.ones(channels, dtype=np.float32)
        offset = offsets[idx] if offsets.size else np.zeros(channels, dtype=np.float32)
        info.gain = gain
        info.offset = offset
        info.data = (info.data * gain) + offset
        info.mask = compute_valid_mask(info.data) & info.mask

    backgrounds: list[np.ndarray] = []
    valid_backgrounds: list[np.ndarray] = []
    for info in tile_infos:
        bg = estimate_tile_background(info.data, info.mask)
        info.background = bg
        backgrounds.append(bg)
        if bg is None or bg.size == 0:
            continue
        if not np.any(np.isfinite(bg)):
            continue
        valid_backgrounds.append(bg)
    B_target = None
    try:
        if valid_backgrounds:
            stacked_bg = np.stack(valid_backgrounds, axis=0)
            med_list: list[float] = []
            stacked_bg = np.asarray(stacked_bg, dtype=np.float32)
            if stacked_bg.ndim == 1:
                stacked_bg = stacked_bg[:, np.newaxis]
            for c in range(stacked_bg.shape[1]):
                vals_c = stacked_bg[:, c]
                vals_c = vals_c[np.isfinite(vals_c)]
                if vals_c.size == 0:
                    med_list.append(float("nan"))
                else:
                    med_list.append(float(np.median(vals_c)))
            if med_list:
                B_target = np.asarray(med_list, dtype=np.float32)
        if B_target is not None and np.any(np.isfinite(B_target)):
            for info in tile_infos:
                bg = info.background
                if bg is None or bg.size == 0:
                    continue
                if bg.shape[0] != B_target.shape[0]:
                    continue
                valid_chan = np.isfinite(bg) & np.isfinite(B_target)
                if not np.any(valid_chan):
                    continue
                delta = np.zeros_like(B_target, dtype=np.float32)
                delta[valid_chan] = bg[valid_chan] - B_target[valid_chan]
                info.data = info.data - delta.reshape((1, 1, -1))
                info.mask = compute_valid_mask(info.data) & info.mask
            _emit("Photometry: applied background harmonization across tiles", callback=progress_callback)
        else:
            _emit("Photometry: background harmonization skipped (no valid backgrounds)", lvl="WARN", callback=progress_callback)
    except Exception:
        _emit("Photometry: background harmonization failed, proceeding without it", lvl="WARN", callback=progress_callback)

    overlap_union: Dict[int, np.ndarray] = {
        info.tile_id: np.zeros(info.data.shape[:2], dtype=bool) for info in tile_infos
    }
    overlaps_used = 0
    pyramids_used = 0
    for ov in overlaps:
        info_a = info_by_id.get(ov.tile_a)
        info_b = info_by_id.get(ov.tile_b)
        if info_a is None or info_b is None:
            continue
        patch_a = info_a.data[ov.slice_a]
        patch_b = info_b.data[ov.slice_b]
        mask_a = info_a.mask[ov.slice_a]
        mask_b = info_b.mask[ov.slice_b]
        overlap_union[info_a.tile_id][ov.slice_a] = True
        overlap_union[info_b.tile_id][ov.slice_b] = True
        blended, weight_patch, used_pyr = _blend_overlap_region(patch_a, patch_b, mask_a, mask_b, use_pyramid=True)
        if blended is None or weight_patch is None or not np.any(weight_patch > 0):
            _emit(
                f"Blending: overlap {info_a.tile_id}-{info_b.tile_id} had no valid pixels",
                lvl="WARN",
                callback=progress_callback,
            )
            continue
        overlaps_used += 1
        if used_pyr:
            pyramids_used += 1
        gx0, gx1, gy0, gy1 = ov.global_bbox
        region_h = min(blended.shape[0], gy1 - gy0)
        region_w = min(blended.shape[1], gx1 - gx0)
        if region_h <= 0 or region_w <= 0:
            continue
        gy_slice = slice(gy0, gy0 + region_h)
        gx_slice = slice(gx0, gx0 + region_w)
        weight_expanded = np.repeat(weight_patch[:region_h, :region_w][..., np.newaxis], channels, axis=2)
        mosaic_sum[gy_slice, gx_slice, :] += blended[:region_h, :region_w, :] * weight_expanded
        weight_sum[gy_slice, gx_slice, :] += weight_expanded

    if overlaps_used:
        _emit(
            f"Blending: applied pyramidal blending on {pyramids_used} overlaps (processed {overlaps_used})",
            callback=progress_callback,
        )

    for info in tile_infos:
        tx0, tx1, ty0, ty1 = info.bbox
        x0 = max(0, min(tx0, W_m))
        y0 = max(0, min(ty0, H_m))
        x1 = max(0, min(tx1, W_m))
        y1 = max(0, min(ty1, H_m))
        if x1 <= x0 or y1 <= y0:
            _emit(
                f"Assembly: tile {info.tile_id} bbox outside mosaic, skipping unique region",
                lvl="DEBUG",
                callback=progress_callback,
            )
            continue
        h, w, c = info.data.shape
        off_x = max(0, -tx0)
        off_y = max(0, -ty0)
        used_w = min(w - off_x, x1 - x0)
        used_h = min(h - off_y, y1 - y0)
        if used_h <= 0 or used_w <= 0:
            _emit(
                f"Assembly: tile {info.tile_id} has no overlap after clamping, skipping unique region",
                lvl="DEBUG",
                callback=progress_callback,
            )
            continue
        data_crop = info.data[off_y : off_y + used_h, off_x : off_x + used_w, :]
        mask_crop = info.mask[off_y : off_y + used_h, off_x : off_x + used_w, :]
        overlap_mask_local = overlap_union[info.tile_id][off_y : off_y + used_h, off_x : off_x + used_w]
        valid_mask_2d = np.any(mask_crop, axis=-1) if mask_crop.ndim == 3 else mask_crop
        unique_mask = valid_mask_2d & (~overlap_mask_local)
        if not np.any(unique_mask):
            continue
        channel_mask = mask_crop if mask_crop.ndim == 3 else np.repeat(mask_crop[..., np.newaxis], c, axis=2)
        channel_mask = channel_mask & unique_mask[..., np.newaxis]
        weight_crop = channel_mask.astype(np.float32, copy=False)
        slice_y = slice(y0, y0 + used_h)
        slice_x = slice(x0, x0 + used_w)
        mosaic_sum[slice_y, slice_x, :] += np.where(channel_mask, data_crop, 0.0) * weight_crop
        weight_sum[slice_y, slice_x, :] += weight_crop
        _emit(f"Assembly: placed tile {info.tile_id} unique area", lvl="DEBUG", callback=progress_callback)

    if not np.any(weight_sum > 0):
        try:
            min_x0 = min(info.bbox[0] for info in tile_infos)
            max_x1 = max(info.bbox[1] for info in tile_infos)
            min_y0 = min(info.bbox[2] for info in tile_infos)
            max_y1 = max(info.bbox[3] for info in tile_infos)
            bbox_hint = f"bbox_extent=({min_x0}:{max_x1},{min_y0}:{max_y1})"
        except Exception:
            bbox_hint = "bbox_extent=unavailable"
        _emit(
            (
                "Assembly: no valid tile data written to mosaic. "
                f"Tiles kept={len(tile_infos)}; {bbox_hint}. "
                "All tiles may have been fully masked out or outside the global canvas."
            ),
            lvl="WARN",
            callback=progress_callback,
        )
        _emit("Assembly: attempting salvage assembly with relaxed placement", lvl="WARN", callback=progress_callback)

        salvage_sum = np.zeros_like(mosaic_sum)
        salvage_weight = np.zeros_like(weight_sum)
        salvage_placed = 0
        for info in tile_infos:
            tx0, tx1, ty0, ty1 = info.bbox
            x0 = max(0, min(tx0, W_m))
            y0 = max(0, min(ty0, H_m))
            x1 = max(0, min(tx1, W_m))
            y1 = max(0, min(ty1, H_m))
            if x1 <= x0 or y1 <= y0:
                continue
            h, w, c = info.data.shape
            off_x = max(0, -tx0)
            off_y = max(0, -ty0)
            used_w = min(w - off_x, x1 - x0)
            used_h = min(h - off_y, y1 - y0)
            if used_h <= 0 or used_w <= 0:
                continue
            data_crop = info.data[off_y : off_y + used_h, off_x : off_x + used_w, :]
            mask_crop = compute_valid_mask(data_crop) & info.mask[off_y : off_y + used_h, off_x : off_x + used_w, :]
            if not np.any(mask_crop):
                continue
            channel_mask = mask_crop if mask_crop.ndim == 3 else np.repeat(mask_crop[..., np.newaxis], c, axis=2)
            weight_crop = channel_mask.astype(np.float32, copy=False)
            slice_y = slice(y0, y0 + used_h)
            slice_x = slice(x0, x0 + used_w)
            salvage_sum[slice_y, slice_x, :] += np.where(channel_mask, data_crop, 0.0) * weight_crop
            salvage_weight[slice_y, slice_x, :] += weight_crop
            salvage_placed += 1

        if not np.any(salvage_weight > 0):
            _emit(
                "Assembly: salvage assembly failed (no valid tile data after salvage)",
                lvl="ERROR",
                callback=progress_callback,
            )
            return None

        mosaic_sum = salvage_sum
        weight_sum = salvage_weight
        _emit(
            f"Assembly: salvage assembly succeeded (placed {salvage_placed} tiles)",
            lvl="WARN",
            callback=progress_callback,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        mosaic = np.where(weight_sum > 0, mosaic_sum / np.clip(weight_sum, 1e-6, None), np.nan)
    if grid_rgb_equalize and mosaic.ndim == 3 and mosaic.shape[-1] == 3:
        weight_shape = getattr(weight_sum, "shape", None)
        _emit(
            f"RGB equalization: calling grid_post_equalize_rgb (shape={mosaic.shape}, weight_shape={weight_shape})",
            lvl="DEBUG",
            callback=progress_callback,
        )
        mosaic = grid_post_equalize_rgb(mosaic, weight_sum, progress_callback=progress_callback)
    else:
        _emit(
            f"RGB equalization: skipped (grid_rgb_equalize={grid_rgb_equalize}, shape={mosaic.shape})",
            lvl="DEBUG",
            callback=progress_callback,
        )
    _emit(
        f"Assembly: final mosaic prepared (shape={mosaic.shape}, dtype={mosaic.dtype})",
        callback=progress_callback,
    )
    if mosaic.shape[-1] == 1 and not legacy_rgb_cube:
        mosaic = mosaic[..., 0]
        weight_sum = weight_sum[..., 0]

    header = None
    if _ASTROPY_AVAILABLE and fits:
        try:
            header = grid.global_wcs.to_header() if hasattr(grid.global_wcs, "to_header") else None  # type: ignore[attr-defined]
        except Exception:
            header = None
        output_data = mosaic.astype(np.uint16) if save_final_as_uint16 else mosaic.astype(np.float32)
        try:
            fits.writeto(output_path, output_data, header=header, overwrite=True)
            # Save weight map for diagnostics
            try:
                weight_hdu = fits.ImageHDU(weight_sum.astype(np.float32), name="WEIGHT")
                with fits.open(output_path, mode="append", memmap=False, do_not_scale_image_data=True) as hdul:
                    hdul.append(weight_hdu)
            except Exception:
                pass
        except Exception as exc:
            _emit(f"Failed to write mosaic {output_path} ({exc})", lvl="ERROR", callback=progress_callback)
            return None
    _emit(f"Final mosaic saved to {output_path}", callback=progress_callback)
    return output_path


def _load_config_from_disk() -> dict:
    try:
        import zemosaic_config

        return zemosaic_config.load_config() or {}
    except Exception:
        return {}


def run_grid_mode(
    input_folder: str,
    output_folder: str,
    progress_callback: ProgressCallback = None,
    *,
    stack_norm_method: str = "linear_fit",
    stack_weight_method: str = "noise_variance",
    stack_reject_algo: str = "kappa_sigma",
    stack_kappa_low: float = 3.0,
    stack_kappa_high: float = 3.0,
    winsor_limits: tuple[float, float] = (0.05, 0.05),
    stack_final_combine: str = "mean",
    apply_radial_weight: bool = False,
    radial_feather_fraction: float = 0.8,
    radial_shape_power: float = 2.0,
    save_final_as_uint16: bool = False,
    legacy_rgb_cube: bool = False,
    grid_rgb_equalize: bool | None = True,
) -> None:
    """Main entry point for Grid/Survey mode."""

    def _coerce_bool_flag(value: object) -> bool | None:
        """Interpret truthy/falsy flags from config, UI, or defaults."""

        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return None
            if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
                return True
            if normalized in {"0", "false", "no", "off", "disable", "disabled"}:
                return False
        try:
            return bool(value)
        except Exception:
            return None

    _emit("Grid/Survey mode activated (stack_plan.csv detected)", callback=progress_callback)
    cfg_disk = _load_config_from_disk()
    # Precedence: on-disk config (grid_rgb_equalize/poststack_equalize_rgb) →
    # caller parameter → built-in default (True).
    rgb_source = "default" if grid_rgb_equalize is None else "param"
    try:
        overlap_fraction = float(cfg_disk.get("batch_overlap_pct", 0.0)) / 100.0
    except Exception:
        overlap_fraction = 0.0
    try:
        grid_size_factor = float(cfg_disk.get("grid_size_factor", 1.0))
    except Exception:
        grid_size_factor = 1.0
    try:
        rgb_cfg = _coerce_bool_flag(cfg_disk.get("grid_rgb_equalize"))
        if rgb_cfg is None:
            rgb_cfg = _coerce_bool_flag(cfg_disk.get("poststack_equalize_rgb"))
        if rgb_cfg is not None:
            grid_rgb_equalize = rgb_cfg
            rgb_source = "config"
        if grid_rgb_equalize is None:
            grid_rgb_equalize = True
        else:
            parsed_param = _coerce_bool_flag(grid_rgb_equalize)
            if parsed_param is not None:
                grid_rgb_equalize = parsed_param
    except Exception:
        grid_rgb_equalize = True if grid_rgb_equalize is None else bool(grid_rgb_equalize)

    grid_rgb_equalize = bool(grid_rgb_equalize)
    _emit(
        f"Grid mode RGB equalization: enabled={grid_rgb_equalize} (source={rgb_source})",
        lvl="INFO",
        callback=progress_callback,
    )
    chunk_budget_mb = 512.0
    try:
        gb_val = cfg_disk.get("grid_chunk_ram_gb")
        if gb_val is not None:
            chunk_budget_mb = float(gb_val) * 1024.0
    except Exception:
        pass
    try:
        mb_val = cfg_disk.get("grid_chunk_ram_mb", cfg_disk.get("grid_chunk_ram"))
        if mb_val is not None:
            chunk_budget_mb = float(mb_val)
    except Exception:
        pass
    if chunk_budget_mb <= 0:
        chunk_budget_mb = 512.0

    config = GridModeConfig(
        grid_size_factor=grid_size_factor,
        overlap_fraction=overlap_fraction,
        stack_norm_method=stack_norm_method,
        stack_weight_method=stack_weight_method,
        stack_reject_algo=stack_reject_algo,
        stack_kappa_low=stack_kappa_low,
        stack_kappa_high=stack_kappa_high,
        winsor_limits=winsor_limits,
        stack_final_combine=stack_final_combine,
        stack_chunk_budget_mb=chunk_budget_mb,
        apply_radial_weight=apply_radial_weight,
        radial_feather_fraction=radial_feather_fraction,
        radial_shape_power=radial_shape_power,
        save_final_as_uint16=save_final_as_uint16,
        legacy_rgb_cube=legacy_rgb_cube,
    )

    csv_path = Path(input_folder).expanduser() / "stack_plan.csv"
    frames = load_stack_plan(csv_path, progress_callback=progress_callback)
    if not frames:
        _emit("Grid mode aborted: no frames loaded", lvl="ERROR", callback=progress_callback)
        raise RuntimeError("Grid mode failed: no frames loaded")

    grid = build_global_grid(frames, grid_size_factor, overlap_fraction, progress_callback=progress_callback)
    if grid is None:
        _emit("Grid mode aborted: unable to build grid", lvl="ERROR", callback=progress_callback)
        raise RuntimeError("Grid mode failed: unable to build grid")
    if not grid.tiles:
        _emit("Grid mode aborted: no grid tiles generated", lvl="ERROR", callback=progress_callback)
        raise RuntimeError("Grid mode failed: no tiles to process")

    assign_frames_to_tiles(frames, grid.tiles, progress_callback=progress_callback)
    out_dir = Path(output_folder).expanduser()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    for tile in grid.tiles:
        process_tile(tile, out_dir, config, progress_callback=progress_callback)

    mosaic_path = assemble_tiles(
        grid,
        grid.tiles,
        out_dir / "mosaic_grid.fits",
        save_final_as_uint16=save_final_as_uint16,
        legacy_rgb_cube=legacy_rgb_cube,
        grid_rgb_equalize=grid_rgb_equalize,
        progress_callback=progress_callback,
    )

    if mosaic_path is None:
        _emit("Grid mode aborted: assembly failed", lvl="ERROR", callback=progress_callback)
        raise RuntimeError("Grid mode failed during assembly")

    _emit("Grid/Survey mode completed", lvl="SUCCESS", callback=progress_callback)
