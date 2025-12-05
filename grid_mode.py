"""Grid/Survey processing pipeline for ZeMosaic.

All logic here is isolated from the classic worker workflow and is only
invoked when a ``stack_plan.csv`` file is present in the input folder.
"""

from __future__ import annotations

import copy
import csv
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

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
    from zemosaic_align_stack import _reject_outliers_kappa_sigma, _reject_outliers_winsorized_sigma_clip
except Exception:  # pragma: no cover - worker remains functional without rejection helpers
    _reject_outliers_kappa_sigma = None
    _reject_outliers_winsorized_sigma_clip = None

logger = logging.getLogger("zemosaic.grid_mode")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


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
    apply_radial_weight: bool = False
    radial_feather_fraction: float = 0.8
    radial_shape_power: float = 2.0
    save_final_as_uint16: bool = False
    legacy_rgb_cube: bool = False


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
        with fits.open(frame.path, memmap=False, do_not_scale_image_data=True) as hdul:
            # do_not_scale_image_data=True pour ne pas se faire surprendre par BZERO/BSCALE
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


def _compute_frame_footprint(frame: FrameInfo, global_wcs: object) -> tuple[float, float, float, float] | None:
    if frame.wcs is None or frame.shape_hw is None or not (_ASTROPY_AVAILABLE and SkyCoord):
        return None
    try:
        h, w = frame.shape_hw
        corners_x = np.array([0, w - 1, 0, w - 1], dtype=float)
        corners_y = np.array([0, 0, h - 1, h - 1], dtype=float)
        sky = frame.wcs.pixel_to_world(corners_x, corners_y)  # type: ignore[call-arg]
        gx, gy = global_wcs.world_to_pixel(sky)  # type: ignore[attr-defined]
        xmin = float(np.nanmin(gx))
        xmax = float(np.nanmax(gx))
        ymin = float(np.nanmin(gy))
        ymax = float(np.nanmax(gy))
        return xmin, xmax, ymin, ymax
    except Exception:
        return None


def _clone_tile_wcs(global_wcs: object, offset_xy: tuple[int, int], shape_hw: tuple[int, int]) -> object:
    tile_wcs = copy.deepcopy(global_wcs)
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
        fp = _compute_frame_footprint(frame, global_wcs)
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
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
        data = hdul[0].data
        alpha_hdu = hdul["ALPHA"] if "ALPHA" in hdul else None
        alpha = None
        if alpha_hdu is not None and alpha_hdu.data is not None:
            try:
                alpha = np.asarray(alpha_hdu.data, dtype=np.float32)
            except Exception:
                alpha = None
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    elif arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
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
    for c in range(channels):
        arr_2d = data[..., c] if data.ndim == 3 else data
        try:
            reproj_arr, footprint = reproject_interp(
                (arr_2d, frame.wcs),
                output_projection=tile.wcs,
                shape_out=(tile_shape_hw[0], tile_shape_hw[1]),
                return_footprint=True,
            )
        except Exception:
            return None, None
        reproj_channels.append(reproj_arr.astype(np.float32, copy=False))
        footprints.append(footprint.astype(np.float32, copy=False))

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


def _normalize_patches(patches: list[np.ndarray]) -> list[np.ndarray]:
    if not patches:
        return patches
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
        patch_norm = patch_norm * (ref_median / med)
        normalized.append(patch_norm.astype(np.float32, copy=False))
    return normalized


def _stack_weighted_patches(
    patches: list[np.ndarray],
    weights: list[np.ndarray],
    config: GridModeConfig,
) -> np.ndarray | None:
    if not patches:
        return None
    # Ensure shapes match and promote to float32
    normalized = _normalize_patches(patches)
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
    weight_effective = np.where(np.isfinite(data_for_combine), weight_stack, 0.0)
    weight_sum = np.sum(weight_effective, axis=0)
    if config.stack_final_combine.lower().strip() == "median":
        result = np.nanmedian(data_for_combine, axis=0)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.sum(data_masked * weight_effective, axis=0) / np.clip(weight_sum, 1e-6, None)
    return result.astype(np.float32, copy=False)


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

    for frame in tile.frames:
        patch, footprint = _reproject_frame_to_tile(frame, tile, tile_shape)
        if patch is None or footprint is None:
            _emit(f"Tile {tile.tile_id}: reprojection failed for {frame.path.name}", lvl="WARN", callback=progress_callback)
            continue
        weight_scalar = _compute_frame_weight(frame, patch, footprint)
        weight_map = np.clip(np.asarray(footprint, dtype=np.float32), 0.0, 1.0)
        if patch.ndim == 3 and weight_map.ndim == 2:
            weight_map = np.repeat(weight_map[..., np.newaxis], patch.shape[-1], axis=2)
        weight_map *= weight_scalar
        aligned_patches.append(patch)
        weight_maps.append(weight_map)

    if not aligned_patches:
        _emit(f"Tile {tile.tile_id}: no usable patches", lvl="WARN", callback=progress_callback)
        return None

    stacked = _stack_weighted_patches(aligned_patches, weight_maps, config)
    if stacked is None:
        _emit(f"Tile {tile.tile_id}: stacking failed", lvl="ERROR", callback=progress_callback)
        return None

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


def assemble_tiles(
    grid: GridDefinition,
    tiles: Iterable[GridTile],
    output_path: Path,
    *,
    save_final_as_uint16: bool = False,
    legacy_rgb_cube: bool = False,
    progress_callback: ProgressCallback = None,
) -> Path | None:
    """Assemble processed tiles into the final mosaic without global reprojection."""

    if not (_ASTROPY_AVAILABLE and fits):
        _emit("Astropy unavailable, cannot assemble mosaic", lvl="ERROR", callback=progress_callback)
        return None
    tiles_list = [t for t in tiles if t.output_path and t.output_path.is_file()]
    if not tiles_list:
        _emit("No tiles to assemble", lvl="ERROR", callback=progress_callback)
        return None

    first_tile_data = None
    for t in tiles_list:
        try:
            with fits.open(t.output_path, memmap=True) as hdul:
                first_tile_data = np.asarray(hdul[0].data, dtype=np.float32)
            break
        except Exception:
            continue
    if first_tile_data is None:
        _emit("Unable to read any tile for assembly", lvl="ERROR", callback=progress_callback)
        return None

    channels = 1 if first_tile_data.ndim == 2 else first_tile_data.shape[-1]
    mosaic_shape = (grid.global_shape_hw[0], grid.global_shape_hw[1], channels)
    mosaic_sum = np.zeros(mosaic_shape, dtype=np.float32)
    weight_sum = np.zeros(mosaic_shape, dtype=np.float32)

    for tile in tiles_list:
        try:
            with fits.open(tile.output_path, memmap=True) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float32)
        except Exception as exc:
            _emit(f"Assembly: failed to read {tile.output_path} ({exc})", lvl="WARN", callback=progress_callback)
            continue
        if data.ndim == 2:
            data = data[..., np.newaxis]
        tx0, tx1, ty0, ty1 = tile.bbox
        h, w = data.shape[:2]
        target_h = min(h, ty1 - ty0)
        target_w = min(w, tx1 - tx0)
        if target_h <= 0 or target_w <= 0:
            continue
        slice_y = slice(ty0, ty0 + target_h)
        slice_x = slice(tx0, tx0 + target_w)
        data_crop = data[:target_h, :target_w, :]
        weight_crop = np.ones_like(data_crop, dtype=np.float32)
        mosaic_sum[slice_y, slice_x, :] += data_crop * weight_crop
        weight_sum[slice_y, slice_x, :] += weight_crop
        _emit(f"Assembly: placed tile {tile.tile_id}", lvl="DEBUG", callback=progress_callback)

    with np.errstate(divide="ignore", invalid="ignore"):
        mosaic = np.where(weight_sum > 0, mosaic_sum / np.clip(weight_sum, 1e-6, None), np.nan)
    mosaic = _normalize_background(mosaic, weight_sum)
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
                with fits.open(output_path, mode="append") as hdul:
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
) -> None:
    """Main entry point for Grid/Survey mode."""

    _emit("Grid/Survey mode activated (stack_plan.csv detected)", callback=progress_callback)
    cfg_disk = _load_config_from_disk()
    try:
        overlap_fraction = float(cfg_disk.get("batch_overlap_pct", 0.0)) / 100.0
    except Exception:
        overlap_fraction = 0.0
    try:
        grid_size_factor = float(cfg_disk.get("grid_size_factor", 1.0))
    except Exception:
        grid_size_factor = 1.0

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
        return

    grid = build_global_grid(frames, grid_size_factor, overlap_fraction, progress_callback=progress_callback)
    if grid is None:
        _emit("Grid mode aborted: unable to build grid", lvl="ERROR", callback=progress_callback)
        return

    assign_frames_to_tiles(frames, grid.tiles, progress_callback=progress_callback)
    out_dir = Path(output_folder).expanduser()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    for tile in grid.tiles:
        process_tile(tile, out_dir, config, progress_callback=progress_callback)

    assemble_tiles(
        grid,
        grid.tiles,
        out_dir / "mosaic_grid.fits",
        save_final_as_uint16=save_final_as_uint16,
        legacy_rgb_cube=legacy_rgb_cube,
        progress_callback=progress_callback,
    )

    _emit("Grid/Survey mode completed", lvl="SUCCESS", callback=progress_callback)
