"""stack_plan.csv parsing helpers for the ZeGrid engine.

Relocated verbatim (ZM-ZEGRID-R8) from the now-removed legacy ``grid_mode``
module. The legacy Grid engine was REMOVED from the product and is ARCHIVED at
``origin/archive/zegrid-legacy-grid-5.0.0`` (branch tip 83b94f4). This module
retains ONLY the helpers still required by the NEW ZeGrid engine
(``zemosaic_zegrid_mode``): the ``stack_plan.csv`` presence check, the CSV
loader, the per-frame WCS loader, and their small private helpers.

Nothing here re-implements or depends on the legacy Grid pipeline.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

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


logger = logging.getLogger("ZeMosaicWorker").getChild("stack_plan")
logger.propagate = True

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


def _open_fits_safely(path: Path):
    """Open a FITS file with the robust settings used across Grid mode."""

    if not (_ASTROPY_AVAILABLE and fits):
        raise RuntimeError("Astropy FITS support unavailable")
    return fits.open(path, memmap=False, do_not_scale_image_data=True)


@dataclass
class FrameInfo:
    path: Path
    exposure: float = 1.0
    bortle: str | None = None
    filter_name: str | None = None
    batch_id: str | None = None
    order: int = 0
    mount: str | None = None
    wcs: object | None = None
    shape_hw: tuple[int, int] | None = None
    footprint: tuple[float, float, float, float] | None = None  # (xmin, xmax, ymin, ymax) in global pixels


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


def _normalize_mount(value: object) -> str | None:
    """Normalize mount descriptors to \"EQ\" or \"ALTZ\"."""

    if value is None:
        return None
    try:
        text = str(value).strip().upper()
    except Exception:
        return None
    if not text:
        return None
    if text in {"EQ", "EQUATORIAL", "GEM", "GERMAN"}:
        return "EQ"
    if text in {"ALTZ", "ALT-AZ", "ALT/AZ", "ALT-AZM", "ALTAZ", "ALTITUDE-AZIMUTH"}:
        return "ALTZ"
    return None


def _resolve_path(base_dir: Path, text: str | os.PathLike[str]) -> Path:
    """Resolve frame paths from stack_plan entries.

    Supports native absolute/relative paths and Windows-exported paths reused on
    POSIX hosts (e.g. ``D:\\...\\file.fit``). In cross-platform cases we try
    sane fallbacks under ``base_dir`` before returning a non-existing candidate.
    """
    raw = str(text).strip().strip('"').strip("'")
    candidates: list[Path] = []

    try:
        candidate = Path(raw)
    except Exception:
        candidate = Path(str(raw))

    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(base_dir / candidate)

    # Windows absolute path serialized into CSV and reused on Linux/macOS.
    windows_drive_like = len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha()
    windows_unc_like = raw.startswith("\\")
    if windows_drive_like or windows_unc_like or "\\" in raw:
        posixish = raw.replace("\\", "/")
        # Keep only path tail after drive if present: D:/a/b/file.fit -> a/b/file.fit
        if windows_drive_like and ":" in posixish:
            posixish = posixish.split(":", 1)[1].lstrip("/")
        if posixish:
            candidates.append(base_dir / Path(posixish))
        # Last-resort: basename in current input folder
        try:
            basename = Path(posixish).name if posixish else Path(raw).name
        except Exception:
            basename = ""
        if basename:
            candidates.append(base_dir / basename)

    # Return first existing candidate.
    seen: set[str] = set()
    first_resolved: Path | None = None
    for cand in candidates:
        try:
            resolved = cand.expanduser().resolve(strict=False)
        except Exception:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if first_resolved is None:
            first_resolved = resolved
        try:
            if resolved.is_file():
                return resolved
        except Exception:
            pass

    return first_resolved if first_resolved is not None else (base_dir / Path(raw)).expanduser().resolve(strict=False)


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
        mount_raw = _field(row, ("mount", "eqmode", "mount_mode"))
        mount = _normalize_mount(mount_raw)

        frames.append(
            FrameInfo(
                path=frame_path,
                exposure=exposure,
                bortle=bortle,
                filter_name=filter_name,
                batch_id=batch_id,
                order=order,
                mount=mount,
            )
        )

    _emit(f"Loaded {len(frames)} frame(s) from stack_plan.csv", callback=progress_callback)
    return frames


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


def load_frame_wcs(frame: FrameInfo, *, progress_callback: ProgressCallback = None) -> bool:
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
    except Exception as exc:
        frame.wcs = None
        _emit(f"[GRID] Failed to parse WCS from {frame.path}: {exc}", lvl="WARN", callback=progress_callback)
    if frame.wcs is None or not getattr(frame.wcs, "is_celestial", False):
        _emit(f"[GRID] No usable celestial WCS in {frame.path}", lvl="WARN", callback=progress_callback)
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
    try:
        px_scale = _extract_pixel_scale_deg(frame.wcs)
        if px_scale is None or not math.isfinite(px_scale) or px_scale <= 0:
            _emit(
                f"[GRID] Rejecting frame {frame.path}: invalid pixel scale",
                lvl="WARN",
                callback=progress_callback,
            )
            frame.wcs = None
            frame.shape_hw = None
            return False
    except Exception:
        _emit(f"[GRID] Rejecting frame {frame.path}: pixel scale check failed", lvl="WARN", callback=progress_callback)
        frame.wcs = None
        frame.shape_hw = None
        return False
    return True
