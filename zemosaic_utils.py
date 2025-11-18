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
"""


# --- Standard Library Imports ---
import os
import math
import copy
import json
import logging
import re
import time
import sys
import platform
import tempfile
from functools import lru_cache
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence
import numpy as np
# L'import de astropy.io.fits est géré ci-dessous pour définir le flag
import cv2

import warnings
import traceback
import gc
import importlib.util

# --- Optional Astropy WCS import (lightweight, guarded) ----------------------
ASTROPY_WCS_AVAILABLE_IN_UTILS = False
AstropyWCS = None
proj_plane_pixel_scales = None
try:  # pragma: no cover - optional dependency path
    from astropy.wcs import WCS as AstropyWCS  # type: ignore
    from astropy.wcs.utils import proj_plane_pixel_scales  # type: ignore

    ASTROPY_WCS_AVAILABLE_IN_UTILS = True
except Exception:
    AstropyWCS = None
    proj_plane_pixel_scales = None

# --- GPU/CUDA Availability ----------------------------------------------------
GPU_AVAILABLE = importlib.util.find_spec("cupy") is not None
map_coordinates = None  # Lazily imported when needed


logger = logging.getLogger(__name__)
_PREVIEW_WCS_LINEARIZED_LOGGED = False


EXCLUDED_DIRS = frozenset({"unaligned_by_zemosaic"})


@lru_cache(maxsize=1)
def get_app_base_dir() -> Path:
    """Return the root directory where ZeMosaic resources live."""

    if getattr(sys, "frozen", False):
        frozen_base = getattr(sys, "_MEIPASS", None)
        if frozen_base:
            try:
                base = Path(frozen_base).resolve()
            except Exception:
                base = Path(sys.executable).resolve().parent
        else:
            base = Path(sys.executable).resolve().parent
        candidate = base / "zemosaic"
        return candidate if candidate.is_dir() else base

    try:
        spec = importlib.util.find_spec("zemosaic")
        if spec and spec.origin:
            return Path(spec.origin).resolve().parent
    except Exception:
        pass

    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path(os.getcwd())


def apply_windows_icon_to_window(window, ico_path: Path | str, log_prefix: str = "[Tk]") -> bool:
    """Force-set the Windows taskbar/title icon using Win32 APIs when possible."""

    if platform.system().lower() != "windows":
        return False

    try:
        import ctypes
    except Exception:
        return False

    try:
        icon_path = Path(ico_path)
    except Exception:
        return False
    if not icon_path.is_file():
        return False

    try:
        hwnd = window.winfo_id()
    except Exception:
        hwnd = None
    if not hwnd:
        return False

    LR_LOADFROMFILE = 0x0010
    LR_DEFAULTSIZE = 0x0040
    IMAGE_ICON = 1
    WM_SETICON = 0x0080
    ICON_SMALL = 0
    ICON_BIG = 1

    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        handle = user32.LoadImageW(
            0,
            str(icon_path),
            IMAGE_ICON,
            0,
            0,
            LR_LOADFROMFILE | LR_DEFAULTSIZE,
        )
        if not handle:
            return False
        user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, handle)
        user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, handle)
        return True
    except Exception as exc:
        try:
            print(f"{log_prefix} Unable to refresh Windows icon via Win32: {exc}")
        except Exception:
            pass
        return False


@lru_cache(maxsize=1)
def get_user_config_dir() -> Path:
    """Return the per-user configuration directory for ZeMosaic."""

    home = Path.home()
    system = platform.system().lower()
    if system == "windows":
        base = os.environ.get("APPDATA")
        base_path = Path(base) if base else home / "AppData" / "Roaming"
    elif system == "darwin":
        base_path = home / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_CONFIG_HOME")
        base_path = Path(base) if base else home / ".config"
    return base_path / "ZeMosaic"


def ensure_user_config_dir() -> Path:
    """Create the per-user configuration directory if needed."""

    config_dir = get_user_config_dir()
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return config_dir


@lru_cache(maxsize=1)
def get_runtime_temp_dir() -> Path:
    """Return a shared temporary directory for ZeMosaic runtime artifacts."""

    base = Path(tempfile.gettempdir()) / "zemosaic_runtime"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        return Path(tempfile.gettempdir())
    return base


def is_path_excluded(path: Path | str, excluded_dirs: Iterable[str] | None = None) -> bool:
    """Return True when *path* is located inside a known excluded directory."""

    try:
        candidate = Path(path)
    except Exception:
        return False

    directories = set(EXCLUDED_DIRS)
    if excluded_dirs is not None:
        try:
            directories = set(excluded_dirs)
        except Exception:
            directories = set(EXCLUDED_DIRS)
    if not directories:
        return False

    try:
        resolved = candidate.expanduser().resolve(strict=False)
        parts = set(resolved.parts)
    except Exception:
        try:
            parts = set(candidate.parts)
        except Exception:
            return False
    return any(part in parts for part in directories)


def _extract_cd_matrix_from_wcs(wcs_obj):
    """Return the 2x2 CD matrix for *wcs_obj* or None when unavailable."""
    try:
        cd_attr = getattr(wcs_obj.wcs, "cd", None)
    except Exception:
        cd_attr = None

    if cd_attr is not None:
        cd_arr = np.asarray(cd_attr, dtype=np.float64)
        if cd_arr.ndim == 2 and cd_arr.shape[0] >= 2 and cd_arr.shape[1] >= 2:
            return cd_arr[:2, :2]

    try:
        pc_attr = getattr(wcs_obj.wcs, "pc", None)
        cdelt_attr = getattr(wcs_obj.wcs, "cdelt", None)
    except Exception:
        pc_attr = None
        cdelt_attr = None

    if pc_attr is not None and cdelt_attr is not None:
        pc_arr = np.asarray(pc_attr, dtype=np.float64)
        cdelt_arr = np.asarray(cdelt_attr, dtype=np.float64)
        if (
            pc_arr.ndim == 2
            and pc_arr.shape[0] >= 2
            and pc_arr.shape[1] >= 2
            and cdelt_arr.size >= 2
        ):
            return pc_arr[:2, :2] @ np.diag(cdelt_arr[:2])
    return None


def validate_wcs_header(header, *, require_footprint: bool = True):
    """Validate *header* and return (is_valid, wcs_obj_or_none, failure_reason)."""

    if header is None:
        return False, None, "missing_header"

    def _get(key: str):
        if header is None:
            return None
        try:
            if hasattr(header, "get"):
                return header.get(key)
            return header[key]
        except Exception:
            return None

    def _float_like(value: object) -> bool:
        if value is None:
            return False
        try:
            float(value)
        except Exception:
            return False
        return True

    ctype1 = _get("CTYPE1")
    ctype2 = _get("CTYPE2")
    if not isinstance(ctype1, str) or not isinstance(ctype2, str):
        return False, None, "missing_ctype"
    axis1 = ctype1.strip().upper()
    axis2 = ctype2.strip().upper()
    if not axis1 or not axis2:
        return False, None, "empty_ctype"
    axis1_ok = any(tag in axis1 for tag in ("RA", "LON"))
    axis2_ok = any(tag in axis2 for tag in ("DEC", "LAT"))
    if not (axis1_ok and axis2_ok):
        return False, None, "ctype_not_celestial"

    for key in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"):
        if not _float_like(_get(key)):
            return False, None, f"missing_{key.lower()}"

    cd_terms = ("CD1_1", "CD1_2", "CD2_1", "CD2_2")
    pc_terms = ("PC1_1", "PC1_2", "PC2_1", "PC2_2")
    cdelt_terms = ("CDELT1", "CDELT2")

    has_cd_matrix = all(_float_like(_get(term)) for term in cd_terms)
    has_pc_matrix = all(_float_like(_get(term)) for term in pc_terms) and all(
        _float_like(_get(term)) for term in cdelt_terms
    )

    if not (has_cd_matrix or has_pc_matrix):
        return False, None, "missing_wcs_matrix"

    if not ASTROPY_WCS_AVAILABLE_IN_UTILS or AstropyWCS is None:
        return False, None, "astropy_wcs_unavailable"

    try:
        candidate_wcs = AstropyWCS(header, naxis=2, relax=True)  # type: ignore[call-arg]
    except Exception as exc:
        return False, None, f"astropy_wcs_exception: {exc}"

    if not getattr(candidate_wcs, "is_celestial", False):
        return False, None, "wcs_not_celestial"

    cd_matrix = _extract_cd_matrix_from_wcs(candidate_wcs)
    if cd_matrix is None:
        return False, None, "missing_cd_matrix"
    if not np.all(np.isfinite(cd_matrix)):
        return False, None, "nonfinite_cd_matrix"

    try:
        det = float(np.linalg.det(cd_matrix))
    except Exception as exc_det:  # pragma: no cover - extremely unlikely
        return False, None, f"cd_determinant_failure: {exc_det}"

    if not np.isfinite(det):
        return False, None, "nonfinite_cd_determinant"
    if abs(det) < 1e-16:
        return False, None, "singular_cd_matrix"

    scale_min_arcsec = 0.3
    scale_max_arcsec = 15.0
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales  # type: ignore

        scales_deg = np.asarray(proj_plane_pixel_scales(candidate_wcs), dtype=np.float64)
    except Exception:
        scales_deg = np.sqrt(np.sum(cd_matrix[:2, :2] ** 2, axis=0))
    scales_arcsec = np.abs(scales_deg) * 3600.0
    finite_scales = scales_arcsec[np.isfinite(scales_arcsec)]
    if finite_scales.size == 0:
        return False, None, "pixel_scale_missing"
    min_scale = float(np.nanmin(finite_scales))
    max_scale = float(np.nanmax(finite_scales))
    if min_scale < scale_min_arcsec or max_scale > scale_max_arcsec:
        return False, None, f"pixel_scale_out_of_range[{min_scale:.3f},{max_scale:.3f}]"

    scale_min_deg = scale_min_arcsec / 3600.0
    scale_max_deg = scale_max_arcsec / 3600.0
    det_abs = abs(det)
    if det_abs < (scale_min_deg**2) * 1e-2:
        return False, None, "cd_determinant_too_small"
    if det_abs > (scale_max_deg**2) * 1e2:
        return False, None, "cd_determinant_too_large"

    if require_footprint:
        naxis1 = _get("NAXIS1")
        naxis2 = _get("NAXIS2")
        axes = None
        if _float_like(naxis1) and _float_like(naxis2):
            try:
                axes = (int(float(naxis2)), int(float(naxis1)))
            except Exception:
                axes = None
        if axes and axes[0] > 0 and axes[1] > 0:
            try:
                footprint = candidate_wcs.calc_footprint(axes=axes)
            except Exception as exc_fp:
                return False, None, f"calc_footprint_failed: {exc_fp}"
            footprint_arr = np.asarray(footprint, dtype=np.float64)
            if footprint_arr.ndim != 2 or footprint_arr.shape[1] < 2:
                return False, None, "calc_footprint_bad_shape"
            if not np.all(np.isfinite(footprint_arr)):
                return False, None, "calc_footprint_nonfinite"
            dec_values = footprint_arr[:, 1]
            if np.any((dec_values < -90.5) | (dec_values > 90.5)):
                return False, None, "calc_footprint_dec_out_of_bounds"

    return True, candidate_wcs, None


def has_valid_wcs(header) -> bool:
    """Return True when *header* appears to describe a usable celestial WCS."""

    is_valid, _, _ = validate_wcs_header(header)
    return bool(is_valid)


def _safe_get_item_value(candidate: Any, attr: str, default: Any = None) -> Any:
    """Return attribute or dict value from *candidate*."""

    if candidate is None:
        return default
    if hasattr(candidate, attr):
        try:
            return getattr(candidate, attr)
        except Exception:
            pass
    if isinstance(candidate, dict):
        try:
            return candidate.get(attr, default)
        except Exception:
            return default
    return default


def _infer_shape_from_item(candidate: Any, wcs_obj: Any = None) -> Optional[tuple[int, int]]:
    """Infer (H, W) for *candidate* using shape/header/WCS metadata."""

    shape = _safe_get_item_value(candidate, "shape")
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        try:
            h = int(shape[0])
            w = int(shape[1])
            if h > 0 and w > 0:
                return (h, w)
        except Exception:
            pass

    header = _safe_get_item_value(candidate, "header")
    if header is None:
        header = _safe_get_item_value(candidate, "header_subset")
    if header is not None:
        getter = header.get if hasattr(header, "get") else header.__getitem__
        try:
            naxis1 = int(getter("NAXIS1"))
            naxis2 = int(getter("NAXIS2"))
            if naxis1 > 0 and naxis2 > 0:
                return (naxis2, naxis1)
        except Exception:
            pass

    if wcs_obj is not None:
        for attr in ("pixel_shape", "array_shape"):
            try:
                shp = getattr(wcs_obj, attr, None)
            except Exception:
                shp = None
            if shp is None:
                continue
            try:
                if len(shp) >= 2:
                    ny, nx = int(shp[0]), int(shp[1])
                    if attr == "pixel_shape":
                        nx, ny = int(shp[0]), int(shp[1])
                        return (ny, nx)
                    if ny > 0 and nx > 0:
                        return (ny, nx)
            except Exception:
                continue
    return None


def _orthonormalize_pc_matrix(matrix: np.ndarray | Sequence[Sequence[float]]) -> Optional[np.ndarray]:
    """Return an orthonormal approximation of *matrix* suitable for WCS PC."""

    try:
        mat = np.asarray(matrix, dtype=np.float64)
    except Exception:
        return None
    if mat.ndim != 2 or mat.shape[0] < 2 or mat.shape[1] < 2:
        return None

    col0 = mat[:, 0]
    col1 = mat[:, 1]
    if not np.all(np.isfinite(col0)) or not np.all(np.isfinite(col1)):
        return None

    norm0 = np.linalg.norm(col0)
    if norm0 <= 0:
        return None
    col0 = col0 / norm0
    col1 = col1 - np.dot(col1, col0) * col0
    norm1 = np.linalg.norm(col1)
    if norm1 <= 0:
        col1 = np.array([-col0[1], col0[0]])
        norm1 = np.linalg.norm(col1)
        if norm1 <= 0:
            return None
    col1 = col1 / norm1
    orientation = np.column_stack((col0, col1))
    det = np.linalg.det(orientation)
    if not np.isfinite(det) or det == 0:
        return None
    if det < 0:
        orientation[:, 1] *= -1.0
    return orientation


def _estimate_pc_matrix_from_wcs_list(wcs_objects: Sequence[Any]) -> Optional[np.ndarray]:
    """Approximate a representative PC matrix from *wcs_objects*."""

    matrices: list[np.ndarray] = []
    for wcs_obj in wcs_objects:
        if wcs_obj is None:
            continue
        pc_candidate = None
        try:
            pc_attr = getattr(wcs_obj.wcs, "pc", None)
        except Exception:
            pc_attr = None
        if pc_attr is not None:
            try:
                pc_arr = np.asarray(pc_attr, dtype=np.float64)
                if pc_arr.ndim == 2 and pc_arr.shape[0] >= 2 and pc_arr.shape[1] >= 2:
                    pc_candidate = pc_arr[:2, :2]
            except Exception:
                pc_candidate = None
        if pc_candidate is None:
            cd_mat = _extract_cd_matrix_from_wcs(wcs_obj)
            if cd_mat is not None:
                col0 = cd_mat[:, 0]
                col1 = cd_mat[:, 1]
                norm0 = float(np.linalg.norm(col0))
                norm1 = float(np.linalg.norm(col1))
                if norm0 > 0 and norm1 > 0:
                    pc_candidate = np.column_stack((col0 / norm0, col1 / norm1))
        if pc_candidate is None:
            continue
        matrices.append(pc_candidate)

    if not matrices:
        return None
    median_pc = np.median(np.stack(matrices, axis=0), axis=0)
    return _orthonormalize_pc_matrix(median_pc)


def _compute_pixel_scale_for_wcs(wcs_obj: Any) -> Optional[float]:
    """Return a representative pixel scale (deg/px) for *wcs_obj*."""

    if wcs_obj is None:
        return None
    if proj_plane_pixel_scales is not None:
        try:
            scales = proj_plane_pixel_scales(wcs_obj)
            if scales is not None and len(scales) >= 2:
                values = np.asarray(scales[:2], dtype=np.float64)
                values = values[np.isfinite(values) & (values > 0)]
                if values.size:
                    return float(np.median(values))
        except Exception:
            pass

    cd_mat = _extract_cd_matrix_from_wcs(wcs_obj)
    if cd_mat is not None:
        col0 = cd_mat[:, 0]
        col1 = cd_mat[:, 1]
        norms = [float(np.linalg.norm(col0)), float(np.linalg.norm(col1))]
        norms = [val for val in norms if np.isfinite(val) and val > 0]
        if norms:
            return float(sum(norms) / len(norms))
    return None


def _analyze_ra_samples(ra_values: Sequence[float]) -> dict[str, float | bool]:
    """Return wrap-aware interval metadata for *ra_values* (degrees)."""

    sanitized: list[float] = []
    for val in ra_values:
        try:
            coerced = float(val)
        except Exception:
            continue
        if math.isnan(coerced) or math.isinf(coerced):
            continue
        sanitized.append(coerced % 360.0)

    if not sanitized:
        raise ValueError("No valid RA samples provided")

    sanitized.sort()
    if len(sanitized) == 1:
        return {
            "use_wrap": False,
            "offset_deg": 0.0,
            "min_unwrapped": sanitized[0],
            "max_unwrapped": sanitized[0],
            "span_unwrapped": 0.0,
            "min_raw": sanitized[0],
            "max_raw": sanitized[0],
        }

    diffs = []
    for i in range(len(sanitized) - 1):
        diffs.append(sanitized[i + 1] - sanitized[i])
    diffs.append((sanitized[0] + 360.0) - sanitized[-1])
    max_gap = max(diffs)
    gap_idx = diffs.index(max_gap)
    span_nowrap = sanitized[-1] - sanitized[0]
    span_nowrap = max(0.0, span_nowrap)
    wrap_span = 360.0 - max_gap
    use_wrap = wrap_span + 1e-6 < span_nowrap

    if use_wrap:
        offset = sanitized[(gap_idx + 1) % len(sanitized)]
        unwrapped = [((val - offset) % 360.0) for val in sanitized]
        min_unwrapped = min(unwrapped)
        max_unwrapped = max(unwrapped)
        return {
            "use_wrap": True,
            "offset_deg": offset,
            "min_unwrapped": min_unwrapped,
            "max_unwrapped": max_unwrapped,
            "span_unwrapped": max_unwrapped - min_unwrapped,
            "min_raw": sanitized[0],
            "max_raw": sanitized[-1],
        }

    return {
        "use_wrap": False,
        "offset_deg": 0.0,
        "min_unwrapped": sanitized[0],
        "max_unwrapped": sanitized[-1],
        "span_unwrapped": span_nowrap,
        "min_raw": sanitized[0],
        "max_raw": sanitized[-1],
    }


def parse_global_wcs_resolution_override(value: Any) -> Optional[tuple[int, int]]:
    """Return (W, H) override parsed from *value* (list/tuple/'WxH')."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            w = int(value[0])
            h = int(value[1])
            if w > 0 and h > 0:
                return (w, h)
        except Exception:
            return None
        return None
    if isinstance(value, str):
        match = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", value)
        if match:
            try:
                w = int(match.group(1))
                h = int(match.group(2))
                if w > 0 and h > 0:
                    return (w, h)
            except Exception:
                return None
    return None


def resolve_global_wcs_output_paths(output_dir: str | Path, configured_path: Optional[str]) -> tuple[str, str]:
    """Return absolute (fits_path, json_path) for the global WCS artifacts."""

    base_dir = Path(output_dir or ".").expanduser()
    rel_path = configured_path or "global_mosaic_wcs.fits"
    rel_path = str(rel_path).strip() or "global_mosaic_wcs.fits"
    fits_path = Path(rel_path)
    if not fits_path.is_absolute():
        fits_path = (base_dir / fits_path).expanduser()
    fits_abs = fits_path.resolve(strict=False)
    json_path = fits_abs.with_suffix(".json")
    return str(fits_abs), str(json_path)


def _build_file_records(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect (path, mtime, size) metadata for each unique file."""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        path = entry.get("path")
        if not path:
            continue
        path_obj = Path(str(path)).expanduser()
        try:
            norm_path = path_obj.resolve(strict=False)
        except Exception:
            norm_path = path_obj
        norm_str = str(norm_path)
        if norm_str in seen:
            continue
        seen.add(norm_str)
        record: dict[str, Any] = {"path": norm_str}
        try:
            stat = norm_path.stat()
            record["mtime"] = float(stat.st_mtime)
            record["size"] = int(stat.st_size)
        except Exception:
            pass
        records.append(record)
    return records


def compute_global_wcs_descriptor(
    items: Sequence[Any],
    *,
    pixel_scale_mode: str = "median",
    orientation_mode: str = "north_up",
    padding_percent: float = 2.0,
    resolution_override: Optional[tuple[int, int]] = None,
    logger_override: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Compute a global WCS header + metadata from *items*."""

    log = logger_override or logger
    if not ASTROPY_WCS_AVAILABLE_IN_UTILS or AstropyWCS is None:
        raise RuntimeError("Astropy WCS is required to compute the global mosaic grid")
    if not items:
        raise ValueError("Cannot build a global WCS without input items")

    usable_entries: list[dict[str, Any]] = []
    ra_samples: list[float] = []
    dec_samples: list[float] = []

    for item in items:
        wcs_obj = _safe_get_item_value(item, "wcs")
        if wcs_obj is None or not getattr(wcs_obj, "is_celestial", False):
            continue
        shape = _infer_shape_from_item(item, wcs_obj)
        if not shape:
            continue
        path = _safe_get_item_value(item, "path") or _safe_get_item_value(item, "path_raw")
        path_str = str(path).strip() if path else None
        try:
            axes = (int(shape[1]), int(shape[0]))
            footprint = wcs_obj.calc_footprint(axes=axes)
            footprint_arr = np.asarray(footprint, dtype=np.float64)
            if footprint_arr.ndim != 2 or footprint_arr.shape[1] < 2:
                continue
            ra_samples.extend(footprint_arr[:, 0].tolist())
            dec_samples.extend(footprint_arr[:, 1].tolist())
        except Exception as exc_fp:
            log.debug("Global WCS: calc_footprint failed for %s: %s", path_str, exc_fp)
            continue
        usable_entries.append({"wcs": wcs_obj, "shape": shape, "path": path_str})

    if not usable_entries:
        raise ValueError("No valid WCS entries found to build the global grid")
    if len(ra_samples) < 2 or len(dec_samples) < 2:
        raise ValueError("Insufficient celestial coverage to determine a bounding box")

    ra_info = _analyze_ra_samples(ra_samples)
    dec_values = []
    for dec in dec_samples:
        try:
            val = float(dec)
        except Exception:
            continue
        if math.isnan(val) or math.isinf(val):
            continue
        dec_values.append(val)
    if not dec_values:
        raise ValueError("Unable to derive DEC extent from the provided WCS footprints")
    dec_min = max(-90.0, min(dec_values))
    dec_max = min(90.0, max(dec_values))
    dec_span = max(1e-9, dec_max - dec_min)

    padding_pct = max(0.0, float(padding_percent))
    padding_multiplier = 1.0 + padding_pct / 100.0

    ra_span = max(1e-9, float(ra_info.get("span_unwrapped") or 0.0))
    ra_span_padded = ra_span * padding_multiplier
    dec_span_padded = dec_span * padding_multiplier

    ra_min_unwrapped = float(ra_info.get("min_unwrapped", 0.0))
    ra_max_unwrapped = float(ra_info.get("max_unwrapped", 0.0))
    ra_offset = float(ra_info.get("offset_deg", 0.0))
    ra_pad_delta = (ra_span_padded - ra_span) * 0.5
    ra_min_unwrapped -= ra_pad_delta
    ra_max_unwrapped += ra_pad_delta
    ra_center_unwrapped = 0.5 * (ra_min_unwrapped + ra_max_unwrapped)
    if bool(ra_info.get("use_wrap", False)):
        ra_center = (ra_offset + ra_center_unwrapped) % 360.0
    else:
        ra_center = 0.5 * (float(ra_info.get("min_raw", 0.0)) + float(ra_info.get("max_raw", 0.0)))

    dec_pad_delta = (dec_span_padded - dec_span) * 0.5
    dec_min_padded = max(-90.0, dec_min - dec_pad_delta)
    dec_max_padded = min(90.0, dec_max + dec_pad_delta)
    dec_center = 0.5 * (dec_min_padded + dec_max_padded)

    scale_candidates = [
        _compute_pixel_scale_for_wcs(entry["wcs"]) for entry in usable_entries
    ]
    scale_candidates = [val for val in scale_candidates if val and val > 0]
    if not scale_candidates:
        raise ValueError("Unable to derive pixel scale from the provided WCS objects")

    mode = (pixel_scale_mode or "median").strip().lower()
    if mode not in {"min", "max", "median"}:
        mode = "median"
    if mode == "min":
        scale_deg = float(min(scale_candidates))
    elif mode == "max":
        scale_deg = float(max(scale_candidates))
    else:
        scale_deg = float(np.median(scale_candidates))
    scale_deg = max(scale_deg, 1e-8)
    scale_arcsec = scale_deg * 3600.0

    if resolution_override is not None:
        width_px = int(resolution_override[0])
        height_px = int(resolution_override[1])
    else:
        width_px = int(math.ceil(max(1.0, ra_span_padded) / scale_deg))
        height_px = int(math.ceil(max(1.0, dec_span_padded) / scale_deg))
    width_px = max(8, width_px)
    height_px = max(8, height_px)

    orientation_mode_norm = (orientation_mode or "north_up").strip().lower()
    orientation_matrix = None
    if orientation_mode_norm == "median_pc":
        orientation_matrix = _estimate_pc_matrix_from_wcs_list([entry["wcs"] for entry in usable_entries])
        if orientation_matrix is None:
            log.warning("Global WCS: median_pc orientation unavailable, falling back to north_up")
    if orientation_matrix is None:
        orientation_matrix = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        orientation_mode_norm = "north_up"

    crpix1 = (width_px / 2.0) + 0.5
    crpix2 = (height_px / 2.0) + 0.5
    global_wcs = AstropyWCS(naxis=2)
    global_wcs.wcs.crval = [float(ra_center), float(dec_center)]
    global_wcs.wcs.crpix = [float(crpix1), float(crpix2)]
    global_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    global_wcs.wcs.cunit = ["deg", "deg"]
    global_wcs.wcs.cdelt = np.array([scale_deg, scale_deg], dtype=np.float64)
    global_wcs.wcs.pc = orientation_matrix
    header = global_wcs.to_header(relax=True)
    header["NAXIS"] = 2
    header["NAXIS1"] = width_px
    header["NAXIS2"] = height_px

    files_meta = _build_file_records(usable_entries)
    descriptor = {
        "header": header,
        "width": width_px,
        "height": height_px,
        "pixel_scale_deg_per_px": scale_deg,
        "pixel_scale_as_per_px": scale_arcsec,
        "padding_percent": padding_pct,
        "orientation": orientation_mode_norm,
        "orientation_matrix": orientation_matrix.tolist(),
        "ra_wrap_used": bool(ra_info.get("use_wrap", False)),
        "ra_wrap_offset_deg": ra_offset,
        "ra_span_deg": ra_span_padded,
        "dec_span_deg": dec_span_padded,
        "center_ra_deg": ra_center,
        "center_dec_deg": dec_center,
        "files": files_meta,
        "nb_images": len(usable_entries),
        "resolution_override": bool(resolution_override is not None),
        "pixel_scale_mode": mode,
        "timestamp": time.time(),
        "source": "computed",
    }
    try:
        log.info(
            "Global WCS descriptor built: entries=%d, RA span=%.3f° (%.3f° padded), "
            "DEC span=%.3f° (%.3f° padded), scale=%.5f°/px (%.2f\"/px), size=%dx%d px, padding=%.2f%%",
            len(usable_entries),
            float(ra_span),
            float(ra_span_padded),
            float(dec_span),
            float(dec_span_padded),
            float(scale_deg),
            float(scale_arcsec),
            int(width_px),
            int(height_px),
            float(padding_pct),
        )
    except Exception:
        pass
    return descriptor


def _descriptor_to_json_payload(descriptor: dict[str, Any]) -> dict[str, Any]:
    """Prepare a JSON-serializable payload from *descriptor*."""

    payload = {
        "width": int(descriptor.get("width", 0)),
        "height": int(descriptor.get("height", 0)),
        "pixel_scale_as_per_px": float(descriptor.get("pixel_scale_as_per_px", 0.0)),
        "pixel_scale_deg_per_px": float(descriptor.get("pixel_scale_deg_per_px", 0.0)),
        "padding_percent": float(descriptor.get("padding_percent", 0.0)),
        "orientation": descriptor.get("orientation"),
        "orientation_matrix": descriptor.get("orientation_matrix"),
        "ra_wrap_used": bool(descriptor.get("ra_wrap_used", False)),
        "ra_wrap_offset_deg": float(descriptor.get("ra_wrap_offset_deg", 0.0)),
        "ra_span_deg": float(descriptor.get("ra_span_deg", 0.0)),
        "dec_span_deg": float(descriptor.get("dec_span_deg", 0.0)),
        "center_ra_deg": float(descriptor.get("center_ra_deg", 0.0)),
        "center_dec_deg": float(descriptor.get("center_dec_deg", 0.0)),
        "nb_images": int(descriptor.get("nb_images", 0)),
        "files": descriptor.get("files", []),
        "resolution_override": bool(descriptor.get("resolution_override", False)),
        "pixel_scale_mode": descriptor.get("pixel_scale_mode"),
        "timestamp": float(descriptor.get("timestamp", time.time())),
        "source": descriptor.get("source", "computed"),
    }
    return payload


def write_global_wcs_files(
    descriptor: dict[str, Any],
    fits_path: str,
    json_path: str,
    *,
    logger_override: Optional[logging.Logger] = None,
) -> None:
    """Persist the global WCS header (.fits) and metadata (.json)."""

    log = logger_override or logger
    if fits_module_for_utils is None:
        raise RuntimeError("Astropy FITS is required to persist the global WCS header")
    header = descriptor.get("header")
    if header is None:
        raise ValueError("Descriptor missing FITS header data")

    fits_path_obj = Path(fits_path).expanduser()
    json_path_obj = Path(json_path).expanduser()
    fits_path_obj.parent.mkdir(parents=True, exist_ok=True)
    json_path_obj.parent.mkdir(parents=True, exist_ok=True)
    primary_hdu = fits_module_for_utils.PrimaryHDU(header=header)
    hdul = fits_module_for_utils.HDUList([primary_hdu])
    hdul.writeto(str(fits_path_obj), overwrite=True, output_verify="silentfix")
    meta_payload = _descriptor_to_json_payload(descriptor)
    meta_payload["fits_path"] = str(fits_path_obj)
    meta_payload["json_path"] = str(json_path_obj)
    with json_path_obj.open("w", encoding="utf-8") as handle:
        json.dump(meta_payload, handle, indent=2, sort_keys=True)
    log.info(
        "Global WCS artifacts written: %s (%dx%d)",
        fits_path_obj,
        descriptor.get("width"),
        descriptor.get("height"),
    )


def load_global_wcs_descriptor(
    fits_path: str,
    json_path: Optional[str] = None,
    *,
    logger_override: Optional[logging.Logger] = None,
) -> Optional[dict[str, Any]]:
    """Load an existing global WCS header + optional JSON metadata."""

    log = logger_override or logger
    fits_path_obj = Path(fits_path).expanduser()
    if not fits_path or not fits_path_obj.is_file():
        return None
    if not ASTROPY_WCS_AVAILABLE_IN_UTILS or AstropyWCS is None or fits_module_for_utils is None:
        raise RuntimeError("Astropy WCS/FITS is required to read the global mosaic header")

    header = fits_module_for_utils.getheader(str(fits_path_obj), 0)
    # Build a WCS object from the header even if NAXIS1/2 are missing.
    wcs_obj = AstropyWCS(header, naxis=2)
    # Some FITS writers reset NAXIS to 0 if no data array is attached. Our
    # writer persists header-only descriptors; rely on JSON as a fallback for
    # width/height when NAXIS1/2 are absent or zero.
    width = int(header.get("NAXIS1", 0) or 0)
    height = int(header.get("NAXIS2", 0) or 0)
    px_scale = _compute_pixel_scale_for_wcs(wcs_obj) or 0.0
    meta_payload = None
    json_candidate_obj = Path(json_path).expanduser() if json_path else fits_path_obj.with_suffix(".json")
    try:
        if json_candidate_obj and json_candidate_obj.is_file():
            with json_candidate_obj.open("r", encoding="utf-8") as handle:
                meta_payload = json.load(handle)
    except Exception as exc_json:
        log.warning("Global WCS: failed to read JSON metadata (%s): %s", json_candidate_obj, exc_json)
        meta_payload = None

    # Fallback to JSON width/height if FITS header lacks them
    try:
        if (width <= 0 or height <= 0) and isinstance(meta_payload, dict):
            mw = int(meta_payload.get("width") or 0)
            mh = int(meta_payload.get("height") or 0)
            if mw > 0 and mh > 0:
                width, height = mw, mh
    except Exception:
        pass

    descriptor = {
        "header": header,
        "wcs": wcs_obj,
        "width": width,
        "height": height,
        "pixel_scale_deg_per_px": px_scale,
        "pixel_scale_as_per_px": px_scale * 3600.0 if px_scale else 0.0,
        "files": meta_payload.get("files") if isinstance(meta_payload, dict) else [],
        "nb_images": meta_payload.get("nb_images") if isinstance(meta_payload, dict) else None,
        "json_path": str(json_candidate_obj),
        "fits_path": str(fits_path_obj),
        "source": "existing",
        "metadata": meta_payload,
    }
    return descriptor

# --- Lightweight CuPy helpers -------------------------------------------------
def gpu_is_available() -> bool:
    if not GPU_AVAILABLE:
        return False
    try:
        import cupy as cp  # type: ignore
        return bool(cp.is_available())
    except Exception:
        return False


_GPU_MEM_DEBUG = os.environ.get("ZEMOSAIC_GPU_MEM_DEBUG", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "debug",
}


def _normalize_gpu_safety_fraction(value: float | None) -> float:
    """Clamp ``safety_fraction`` to a reasonable range."""

    default = 0.75
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return float(min(0.9, max(0.1, numeric)))


def ensure_cupy_pool_initialized(device_id: int | None = None) -> None:
    """Idempotently enable CuPy device + memory pools.

    - Sets the current device (optional).
    - Enables a default device memory pool and a pinned host memory pool.
    """
    if not gpu_is_available():
        return
    try:
        ensure_cupy_pool_initialized._done  # type: ignore[attr-defined]
        return
    except AttributeError:
        pass
    import cupy as cp  # type: ignore
    try:
        if device_id is not None:
            cp.cuda.Device(int(device_id)).use()
    except Exception:
        # Ignore invalid ids; rely on current device
        pass
    try:
        mp = cp.cuda.MemoryPool(cp.cuda.malloc)
        cp.cuda.set_allocator(mp.malloc)
    except Exception:
        pass
    try:
        pmp = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(pmp.malloc)
    except Exception:
        pass
    ensure_cupy_pool_initialized._done = True  # type: ignore[attr-defined]


def free_cupy_memory_pools() -> None:
    """Release cached device and pinned host memory held by CuPy pools."""

    if not gpu_is_available():
        return
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return
    try:
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
    except Exception:
        pass
    try:
        pinned_pool = cp.get_default_pinned_memory_pool()
        pinned_pool.free_all_blocks()
    except Exception:
        pass


def gpu_memory_sufficient(estimated_bytes: int, safety_fraction: float = 0.75) -> bool:
    """Return True if the reported free GPU memory largely exceeds ``estimated_bytes``."""

    if not gpu_is_available() or estimated_bytes <= 0:
        return True
    safety = _normalize_gpu_safety_fraction(safety_fraction)
    try:
        import cupy as cp  # type: ignore
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        # Leave some headroom so we do not hit the allocator limit immediately
        threshold = int(max(0, free_bytes * safety))
        allowed = estimated_bytes <= threshold
        if _GPU_MEM_DEBUG and not allowed:
            logger.debug(
                "GPU memory guard: estimated=%d bytes, free=%d bytes, "
                "allowed=%d bytes (safety=%.2f)",
                int(estimated_bytes),
                int(free_bytes),
                int(threshold),
                safety,
            )
        return allowed
    except Exception as exc:
        if _GPU_MEM_DEBUG:
            logger.debug("GPU memory guard: memGetInfo failed (%s); allowing GPU.", exc)
        # If querying memory fails, err on the side of allowing execution
        return True


def _get_gpu_allowed_bytes(safety_fraction: float = 0.75) -> int | None:
    """Return the approximate safe allocation budget in bytes."""

    if not gpu_is_available():
        return None
    try:
        import cupy as cp  # type: ignore

        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        safe_fraction = max(0.1, min(0.9, float(safety_fraction)))
        return int(max(0, free_bytes * safe_fraction))
    except Exception:
        return None


def _format_mebibytes(byte_count: int | None) -> str:
    if not byte_count or byte_count <= 0:
        return "n/a"
    return f"{byte_count / (1024 ** 2):.1f}"


def _estimate_tile_cache_bytes(data_list: Sequence[np.ndarray]) -> int:
    total = 0
    for tile in data_list:
        arr = np.asarray(tile, dtype=np.float32)
        total += arr.nbytes
    # Each tile allocates an image and a mask on the GPU.
    return total * 2


def _log_gpu_event(
    message_key: str,
    level: str = "INFO",
    progress_callback: Optional[Callable] = None,
    **kwargs,
) -> None:
    """Emit a GPU-related log either via callback or the module logger."""

    if progress_callback and callable(progress_callback):
        try:
            progress_callback(message_key, None, level, **kwargs)
            return
        except Exception as exc:
            logger.warning("GPU log callback failed (%s); falling back to module logger.", exc)
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.log(lvl, "[%s] %s", message_key, kwargs if kwargs else "")


def _force_cpu_intertile() -> bool:
    """Return True when GPU usage for intertile helpers is disabled via env."""

    value = os.environ.get("ZEMOSAIC_FORCE_CPU_INTERTILE")
    if value is None:
        return False
    value = value.strip().lower()
    return value not in {"", "0", "false", "no", "off"}

from reproject.mosaicking import reproject_and_coadd as cpu_reproject_and_coadd
try:
    from reproject import reproject_interp
except Exception:
    reproject_interp = None

# --- Définition locale du flag ASTROPY_AVAILABLE et du module fits pour ce fichier ---
ASTROPY_AVAILABLE_IN_UTILS = False
fits_module_for_utils = None # Contiendra le module fits réel ou un mock

# IMPORTS D'ASTROPY POUR LA VISUALISATION
ASTROPY_VISUALIZATION_AVAILABLE = False
ImageNormalize, PercentileInterval, AsinhStretch, LogStretch = None, None, None, None # Pour les type hints et fallback
try:
    from astropy.visualization import (ImageNormalize, PercentileInterval, 
                                       AsinhStretch, LogStretch) # Et d'autres si besoin (SqrtStretch, etc.)
    ASTROPY_VISUALIZATION_AVAILABLE = True
    # print("INFO (zemosaic_utils): astropy.visualization importé.")
except ImportError:
    print("AVERT (zemosaic_utils): astropy.visualization non disponible. L'étirement asinh avancé ne sera pas possible.")

try:
    from astropy.io import fits as actual_fits_for_utils
    from astropy.io.fits.verify import VerifyWarning # Importer ici si Astropy est là
    warnings.filterwarnings("ignore", category=VerifyWarning, message="Keyword name.*is greater than 8 characters.*")
    warnings.filterwarnings("ignore", category=VerifyWarning, message="Keyword name.*contains characters not allowed.*")
    fits_module_for_utils = actual_fits_for_utils
    ASTROPY_AVAILABLE_IN_UTILS = True
    # print("INFO (zemosaic_utils): Astropy (fits) importé avec succès pour ce module.")
except ImportError:
    # Créer un placeholder minimal pour fits si Astropy n'est pas là du tout
    class MockFitsCard: 
        def __init__(self, key, value, comment=''):
            self.keyword = key; self.value = value; self.comment = comment
    class MockFitsHeader:
        def __init__(self): self._cards = {}; self.comments = {}
        def update(self, other):
            if isinstance(other, MockFitsHeader): self._cards.update(other._cards); self.comments.update(other.comments)
            elif isinstance(other, dict):
                for k, v_tuple in other.items():
                    if isinstance(v_tuple, tuple) and len(v_tuple) == 2: self._cards[k] = v_tuple[0]; self.comments[k] = v_tuple[1]
                    else: self._cards[k] = v_tuple
        def copy(self): new_header = MockFitsHeader(); new_header._cards = self._cards.copy(); new_header.comments = self.comments.copy(); return new_header
        def __contains__(self, key): return key in self._cards
        def __delitem__(self, key):
            if key in self._cards: del self._cards[key]
            if key in self.comments: del self.comments[key]
        def __setitem__(self, key, value_comment_tuple):
            if isinstance(value_comment_tuple, tuple) and len(value_comment_tuple) == 2: self._cards[key] = value_comment_tuple[0]; self.comments[key] = value_comment_tuple[1]
            else: self._cards[key] = value_comment_tuple
        def get(self, key, default=None): return self._cards.get(key, default)
        def cards(self):
            for k, v in self._cards.items(): yield MockFitsCard(k,v,self.comments.get(k,''))
  
    class MockPrimaryHDU: # Renommé pour éviter conflit si MockHDU est aussi défini
        def __init__(self, data=None, header=None):
            self.data = data
            self.header = header if header is not None else MockFitsHeader()
            self.is_image = data is not None 
            self.shape = data.shape if hasattr(data, 'shape') else None
            self.name = "PRIMARY"
        def copy(self): return MockPrimaryHDU(self.data.copy() if self.data is not None else None, self.header.copy())
  
    class MockHDU: 
        def __init__(self, data=None, header=None):
            self.data = data
            self.header = header if header is not None else MockFitsHeader()
            self.is_image = data is not None
            self.shape = data.shape if hasattr(data, 'shape') else None
            self.name = "PRIMARY"
        def copy(self): return MockHDU(self.data.copy() if self.data is not None else None, self.header.copy())

    class MockHDUList:
        def __init__(self, hdus=None):
            self.hdus = hdus if hdus is not None else []
        def __getitem__(self, key): return self.hdus[key]
        def __len__(self): return len(self.hdus)
        def writeto(self, output_path, overwrite=True, checksum=False, output_verify='fix'): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): self.close()

    class MockFitsModule:
        Header = MockFitsHeader
        PrimaryHDU = MockPrimaryHDU
        HDUList = MockHDUList
        @staticmethod
        def open(filepath, memmap=False, do_not_scale_image_data=True):
            print(f"MOCK fits_module_for_utils.open CALLED for {filepath} (Astropy not found). Returning minimal mock HDU.")
            # Simuler une HDU minimale pour éviter des crashs dans load_and_validate_fits
            # Ce mock est très basique et ne lira pas réellement le fichier.
            mock_data = np.array([[0,0],[0,0]], dtype=np.int16) # Petite donnée pour avoir un .shape
            mock_header = MockFitsHeader()
            mock_header['NAXIS'] = 2
            mock_header['NAXIS1'] = 2
            mock_header['NAXIS2'] = 2
            return MockHDUList([MockHDU(data=mock_data, header=mock_header)])
        @staticmethod
        def getheader(filepath, ext=0):
            print(f"MOCK fits_module_for_utils.getheader CALLED for {filepath} (Astropy not found).")
            return MockFitsHeader()


def _merge_header_cards(target_header, source_header) -> None:
    """Merge ``source_header`` into ``target_header`` while skipping NAXIS cards."""

    if source_header is None:
        return

    try:
        from astropy.io.fits import Header as FitsHeader  # type: ignore
    except Exception:
        FitsHeader = ()  # type: ignore

    # Handle astropy Header explicitly to preserve comments
    if FitsHeader and isinstance(source_header, FitsHeader):
        for card in source_header.cards:
            keyword = getattr(card, "keyword", None)
            if not keyword:
                continue
            if str(keyword).upper().startswith("NAXIS"):
                continue
            target_header[keyword] = (card.value, card.comment)
        return

    # Generic mapping-like objects (dict, HeaderDict, etc.)
    if hasattr(source_header, "items"):
        try:
            for key, value in source_header.items():
                if str(key).upper().startswith("NAXIS"):
                    continue
                target_header[key] = value
        except Exception:
            pass
        return

    # Fallback: attempt direct ``update``
    try:
        target_header.update(source_header)
    except Exception:
        pass


def write_final_fits_uint16_color_aware(
    out_path: str,
    final_img: "np.ndarray",
    header: "fits_module_for_utils.Header | dict | None" = None,
    *,
    force_rgb_planes: bool,
    legacy_rgb_cube: bool,
    overwrite: bool = True,
):
    """Save final mosaic as uint16 FITS while preserving RGB colour planes."""

    if not ASTROPY_AVAILABLE_IN_UTILS or fits_module_for_utils is None:
        raise RuntimeError("Astropy FITS writer unavailable; cannot save uint16 mosaic.")

    import numpy as np  # Local import for clarity inside helper

    if final_img is None:
        raise ValueError("final_img is None")

    arr = np.asarray(final_img)
    if arr.size == 0:
        raise ValueError("final_img is empty")

    fits_mod = fits_module_for_utils

    # Normalize mono (H, W, 1) inputs
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    is_rgb = bool(force_rgb_planes and arr.ndim == 3 and arr.shape[-1] == 3)

    if np.issubdtype(arr.dtype, np.floating):
        arr_float = _ensure_float32_no_nan(arr)

        finite_mask = np.isfinite(arr_float)
        if not np.any(finite_mask):
            u16 = np.zeros(arr_float.shape, dtype=np.uint16)
        else:
            finite_vals = arr_float[finite_mask]
            vmin = float(np.nanmin(finite_vals))
            vmax = float(np.nanmax(finite_vals))

            if 0.0 <= vmin and vmax <= 1.0 + 1e-6:
                scaled = np.clip(arr_float, 0.0, 1.0) * 65535.0 + 0.5
                u16 = scaled.astype(np.uint16, copy=False)
            elif 0.0 <= vmin and vmax <= 65535.0:
                clipped = np.clip(arr_float, 0.0, 65535.0) + 0.5
                u16 = clipped.astype(np.uint16, copy=False)
            else:
                u16 = _rescale_to_u16(arr_float)
    elif arr.dtype != np.uint16:
        u16 = arr.astype(np.uint16, copy=False)
    else:
        u16 = arr

    i16 = (u16.astype(np.int32, copy=False) - 32768).astype(np.int16, copy=False)

    if is_rgb:
        data = np.moveaxis(i16, -1, 0)
        primary_hdu = fits_mod.PrimaryHDU(data=data)
        hdr = primary_hdu.header
        _merge_header_cards(hdr, header)
        hdr["ZEMORGB"] = (True, "RGB planes present")
        hdr["CHANNELS"] = (3, "Number of color channels")
    else:
        primary_hdu = fits_mod.PrimaryHDU(data=i16)
        hdr = primary_hdu.header
        _merge_header_cards(hdr, header)
        hdr["ZEMORGB"] = (False, "No separate RGB planes")
        hdr["CHANNELS"] = (1, "Number of color channels")

    hdr["ZEMO16"] = (True, "Saved as uint16 via int16 + BZERO")
    hdr["ZEMOLEG"] = (bool(legacy_rgb_cube), "Legacy RGB cube mode")
    hdr["BITPIX"] = 16
    hdr["BSCALE"] = 1
    hdr["BZERO"] = 32768
    if "DATAMIN" in hdr:
        del hdr["DATAMIN"]
    if "DATAMAX" in hdr:
        del hdr["DATAMAX"]

    hdul = fits_mod.HDUList([primary_hdu])

    try:
        hdul.writeto(out_path, overwrite=overwrite, output_verify="fix")
    finally:
        if hasattr(hdul, "close"):
            try:
                hdul.close()
            except Exception:
                pass


def append_alpha_hdu(hdul, alpha_arr):
    """
    Ensure an ALPHA image extension (uint8, 0..255) matches *alpha_arr*.
    Replaces an existing ALPHA HDU while leaving other HDUs untouched.
    """
    if hdul is None or alpha_arr is None:
        return
    if fits_module_for_utils is None:
        raise RuntimeError("append_alpha_hdu requires astropy.io.fits")

    alpha = np.asarray(alpha_arr)
    if alpha.ndim == 3 and alpha.shape[-1] == 1:
        alpha = alpha[..., 0]
    elif alpha.ndim > 2:
        alpha = np.squeeze(alpha)
    if alpha.ndim != 2:
        raise ValueError(f"alpha_arr must be 2D, got shape={alpha.shape}")
    alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
    alpha_u8 = np.clip(alpha, 0, 255).astype(np.uint8, copy=False)

    indices_to_drop: list[int] = []
    for idx in range(1, len(hdul)):
        name = getattr(hdul[idx], "name", "")
        if isinstance(name, str) and name.upper() == "ALPHA":
            indices_to_drop.append(idx)
    for idx in reversed(indices_to_drop):
        del hdul[idx]

    alpha_hdu = fits_module_for_utils.ImageHDU(data=alpha_u8, name="ALPHA")
    alpha_hdu.header["ALPHADSC"] = ("1=opaque(in), 0=transparent(out)", "")
    hdul.append(alpha_hdu)
    try:
        hdul[0].header["ALPHAEXT"] = (1, "Alpha mask ext present")
    except Exception:
        pass
# --- Fin Définition locale ---

warnings.filterwarnings("ignore", category=FutureWarning)


def _extract_luminance_plane(tile_array: np.ndarray) -> np.ndarray:
    """Retourne une vue 2D (float32) utilisée pour les analyses photométriques."""

    if tile_array.ndim == 2:
        return tile_array.astype(np.float32, copy=False)
    if tile_array.ndim == 3 and tile_array.shape[-1] > 1:
        return tile_array.mean(axis=-1, dtype=np.float32)
    if tile_array.ndim == 3:
        return tile_array[..., 0].astype(np.float32, copy=False)
    return tile_array.reshape(tile_array.shape[0], tile_array.shape[1]).astype(np.float32, copy=False)


def estimate_overlap_pairs(
    wcs_list,
    shapes_hw,
    final_output_wcs,
    final_output_shape_hw,
    min_overlap_fraction: float = 0.05,
):
    """Estime les couples de tuiles dont les empreintes WCS se chevauchent."""

    if not wcs_list or not final_output_wcs:
        return []
    try:
        final_w, final_h = int(final_output_shape_hw[1]), int(final_output_shape_hw[0])
    except Exception:
        final_h, final_w = 0, 0
    try:
        header = final_output_wcs.to_header()
        crpix1 = float(header.get("CRPIX1", 0.0))
        crpix2 = float(header.get("CRPIX2", 0.0))
    except Exception:
        header = None
        crpix1 = crpix2 = 0.0

    footprints = []
    for idx, (wcs_obj, shape_hw) in enumerate(zip(wcs_list, shapes_hw)):
        try:
            if wcs_obj is None or not getattr(wcs_obj, "is_celestial", False):
                footprints.append(None)
                continue
            h_i, w_i = int(shape_hw[0]), int(shape_hw[1])
            if h_i <= 0 or w_i <= 0:
                footprints.append(None)
                continue
            corners = np.array(
                [[0.0, 0.0], [w_i - 1.0, 0.0], [0.0, h_i - 1.0], [w_i - 1.0, h_i - 1.0]],
                dtype=np.float64,
            )
            world = wcs_obj.pixel_to_world(corners[:, 0], corners[:, 1])
            px, py = final_output_wcs.world_to_pixel(world)
            if px is None or py is None:
                footprints.append(None)
                continue
            px = np.asarray(px, dtype=np.float64)
            py = np.asarray(py, dtype=np.float64)
            if not np.isfinite(px).any() or not np.isfinite(py).any():
                footprints.append(None)
                continue
            x_min = float(np.nanmin(px))
            x_max = float(np.nanmax(px))
            y_min = float(np.nanmin(py))
            y_max = float(np.nanmax(py))
            if header is not None:
                # Les CRPIX sont indexés à 1 ; limiter les bbox au champ final si possible.
                x_min = max(x_min, -crpix1)
                y_min = max(y_min, -crpix2)
                if final_w > 0:
                    x_max = min(x_max, final_w - crpix1)
                if final_h > 0:
                    y_max = min(y_max, final_h - crpix2)
            if not np.isfinite([x_min, x_max, y_min, y_max]).all():
                footprints.append(None)
                continue
            if x_max <= x_min or y_max <= y_min:
                footprints.append(None)
                continue
            area = max((x_max - x_min) * (y_max - y_min), 1e-6)
            footprints.append((x_min, x_max, y_min, y_max, area))
        except Exception:
            footprints.append(None)

    overlaps = []
    for i in range(len(footprints)):
        box_i = footprints[i]
        if box_i is None:
            continue
        for j in range(i + 1, len(footprints)):
            box_j = footprints[j]
            if box_j is None:
                continue
            x0 = max(box_i[0], box_j[0])
            x1 = min(box_i[1], box_j[1])
            y0 = max(box_i[2], box_j[2])
            y1 = min(box_i[3], box_j[3])
            if x1 <= x0 or y1 <= y0:
                continue
            overlap_area = (x1 - x0) * (y1 - y0)
            min_area = min(box_i[4], box_j[4])
            if min_area <= 0:
                continue
            if overlap_area / min_area < max(0.0, float(min_overlap_fraction)):
                continue
            x0_int = int(max(0, math.floor(x0)))
            y0_int = int(max(0, math.floor(y0)))
            x1_int = int(math.ceil(x1))
            y1_int = int(math.ceil(y1))
            if final_w > 0:
                x1_int = min(x1_int, final_w)
            if final_h > 0:
                y1_int = min(y1_int, final_h)
            if x1_int - x0_int < 4 or y1_int - y0_int < 4:
                continue
            overlaps.append(
                {
                    "i": i,
                    "j": j,
                    "bbox": (x0_int, x1_int, y0_int, y1_int),
                    "weight": overlap_area,
                }
            )
    return overlaps


def robust_affine_fit(x_values: np.ndarray, y_values: np.ndarray, clip_sigma: float = 2.5):
    """Ajuste robuste y ≈ a*x + b avec rejet sigma-clipping."""

    if x_values.size < 16:
        return None
    x = x_values.astype(np.float64, copy=False)
    y = y_values.astype(np.float64, copy=False)
    mask = np.ones_like(x, dtype=bool)
    clip_sigma = max(1.0, float(clip_sigma))
    a_b = (1.0, 0.0)
    for _ in range(6):
        if mask.sum() < 8:
            break
        A = np.vstack([x[mask], np.ones(mask.sum(), dtype=np.float64)]).T
        try:
            sol, *_ = np.linalg.lstsq(A, y[mask], rcond=None)
        except Exception:
            return None
        a_est, b_est = float(sol[0]), float(sol[1])
        residuals = y - (a_est * x + b_est)
        res_sel = residuals[mask]
        sigma = np.std(res_sel)
        a_b = (a_est, b_est)
        if sigma <= 1e-6:
            break
        keep = np.abs(res_sel) <= clip_sigma * sigma
        if keep.all():
            break
        new_mask = np.zeros_like(mask)
        new_mask[np.flatnonzero(mask)[keep]] = True
        if new_mask.sum() < 8:
            break
        mask = new_mask
    return a_b


def compute_sky_statistics(
    image: np.ndarray | None,
    low_percentile: float,
    high_percentile: float,
) -> dict[str, float] | None:
    """Compute simple sky statistics using percentile-based limits."""

    if image is None:
        return None
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    use_gpu = False
    fallback_reason = ""
    if _force_cpu_intertile():
        fallback_reason = "env_override"
    else:
        try:
            if gpu_is_available():
                estimated_bytes = int(arr.nbytes * 2.0)
                if gpu_memory_sufficient(estimated_bytes, safety_fraction=0.75):
                    use_gpu = True
                else:
                    fallback_reason = f"insufficient_memory(est={estimated_bytes})"
            else:
                fallback_reason = "gpu_unavailable"
        except Exception as exc:
            fallback_reason = f"gpu_check_failed:{exc!r}"

    if use_gpu:
        try:
            import cupy as cp  # type: ignore
        except Exception as exc:
            fallback_reason = f"gpu_import_failed:{exc!r}"
            use_gpu = False
        else:
            ensure_cupy_pool_initialized()
            arr_gpu = None
            try:
                logger.debug(
                    "[Intertile][GPU] compute_sky_statistics using CuPy (N=%d, bytes=%d)",
                    arr.size,
                    arr.nbytes,
                )
                arr_gpu = cp.asarray(arr, dtype=cp.float32)
                low = float(cp.asnumpy(cp.nanpercentile(arr_gpu, float(low_percentile))))
                high = float(cp.asnumpy(cp.nanpercentile(arr_gpu, float(high_percentile))))
                median = float(cp.asnumpy(cp.nanmedian(arr_gpu)))
                return {"median": median, "low": low, "high": high}
            except Exception as exc:
                fallback_reason = f"gpu_error:{exc!r}"
            finally:
                if arr_gpu is not None:
                    del arr_gpu
                free_cupy_memory_pools()

    if fallback_reason:
        logger.debug("[Intertile][GPU] Fallback to CPU: reason=%s", fallback_reason)
    low = float(np.nanpercentile(arr, float(low_percentile)))
    high = float(np.nanpercentile(arr, float(high_percentile)))
    median = float(np.nanmedian(arr))
    return {"median": median, "low": low, "high": high}


def estimate_sky_affine_to_ref(
    samples_src: np.ndarray,
    samples_ref: np.ndarray,
    sky_low: float,
    sky_high: float,
    clip_sigma: float,
):
    """Estimate affine parameters matching ``samples_src`` onto ``samples_ref``."""

    if samples_src is None or samples_ref is None:
        return None
    x = np.asarray(samples_src, dtype=np.float64).ravel()
    y = np.asarray(samples_ref, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return None
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return None
    x = x[valid]
    y = y[valid]
    if x.size < 16:
        return None
    x_sel: np.ndarray | None = None
    y_sel: np.ndarray | None = None
    use_gpu = False
    fallback_reason = ""
    if _force_cpu_intertile():
        fallback_reason = "env_override"
    else:
        try:
            if gpu_is_available():
                estimated_bytes = int((x.nbytes + y.nbytes) * 3.0)
                if gpu_memory_sufficient(estimated_bytes, safety_fraction=0.75):
                    use_gpu = True
                else:
                    fallback_reason = f"insufficient_memory(est={estimated_bytes})"
            else:
                fallback_reason = "gpu_unavailable"
        except Exception as exc:
            fallback_reason = f"gpu_check_failed:{exc!r}"

    if use_gpu:
        try:
            import cupy as cp  # type: ignore
        except Exception as exc:
            fallback_reason = f"gpu_import_failed:{exc!r}"
            use_gpu = False
        else:
            ensure_cupy_pool_initialized()
            x_gpu = y_gpu = proxy_gpu = mask_gpu = None
            try:
                logger.debug(
                    "[Intertile][GPU] estimate_sky_affine_to_ref using CuPy for percentiles (N=%d, bytes=%d)",
                    x.size,
                    x.nbytes + y.nbytes,
                )
                x_gpu = cp.asarray(x, dtype=cp.float32)
                y_gpu = cp.asarray(y, dtype=cp.float32)
                proxy_gpu = 0.5 * (x_gpu + y_gpu)
                p_low = float(cp.asnumpy(cp.nanpercentile(proxy_gpu, float(sky_low))))
                p_high = float(cp.asnumpy(cp.nanpercentile(proxy_gpu, float(sky_high))))
                if not math.isfinite(p_low) or not math.isfinite(p_high):
                    return None
                if p_high <= p_low:
                    p_low = float(cp.asnumpy(cp.nanmin(proxy_gpu)))
                    p_high = float(cp.asnumpy(cp.nanmax(proxy_gpu)))
                mask_gpu = (proxy_gpu >= p_low) & (proxy_gpu <= p_high)
                x_sel = cp.asnumpy(x_gpu[mask_gpu])
                y_sel = cp.asnumpy(y_gpu[mask_gpu])
                if x_sel.size < 16:
                    x_sel = cp.asnumpy(x_gpu)
                    y_sel = cp.asnumpy(y_gpu)
            except Exception as exc:
                fallback_reason = f"gpu_error:{exc!r}"
                x_sel = None
                y_sel = None
                use_gpu = False
            finally:
                if mask_gpu is not None:
                    del mask_gpu
                if proxy_gpu is not None:
                    del proxy_gpu
                if x_gpu is not None:
                    del x_gpu
                if y_gpu is not None:
                    del y_gpu
                free_cupy_memory_pools()

    if not use_gpu:
        if fallback_reason:
            logger.debug("[Intertile][GPU] Fallback to CPU: reason=%s", fallback_reason)
        proxy = 0.5 * (x + y)
        try:
            p_low = float(np.nanpercentile(proxy, float(sky_low)))
            p_high = float(np.nanpercentile(proxy, float(sky_high)))
        except Exception:
            p_low = float(np.nanpercentile(proxy, 25.0))
            p_high = float(np.nanpercentile(proxy, 75.0))
        if not np.isfinite(p_low) or not np.isfinite(p_high):
            return None
        if p_high <= p_low:
            p_low = float(np.nanmin(proxy))
            p_high = float(np.nanmax(proxy))
        sky_mask = (proxy >= p_low) & (proxy <= p_high)
        x_sel = x[sky_mask]
        y_sel = y[sky_mask]
        if x_sel.size < 16:
            x_sel = x
            y_sel = y
    fit = robust_affine_fit(x_sel, y_sel, clip_sigma=float(clip_sigma))
    if fit is None:
        return None
    gain, offset = fit
    if not (np.isfinite(gain) and np.isfinite(offset)):
        return None

    # --- Safe photometric fallback (visual stability only) ---
    # Compute correlation between selected samples. If correlation is weak or
    # fitted gain is non-positive, recompute a stable gain/offset from robust
    # statistics to avoid negative-flux master tiles on weak datasets.
    try:
        corr = float(np.corrcoef(x_sel, y_sel)[0, 1])
    except Exception:
        corr = float("nan")

    if float(gain) <= 0.0 or (not np.isfinite(corr)) or abs(corr) < 0.3:
        src_med = float(np.nanmedian(x_sel))
        ref_med = float(np.nanmedian(y_sel))
        src_std = float(np.nanstd(x_sel))
        ref_std = float(np.nanstd(y_sel))
        if src_std > 1e-6:
            safe_gain = max(1e-6, ref_std / src_std)
        else:
            safe_gain = 1.0
        safe_gain = np.float32(safe_gain)
        safe_offset = np.float32(ref_med - float(safe_gain) * src_med)

        # Log explicit safeguard application for diagnostics
        try:
            logger.debug(
                "Photometric safeguard applied (corr=%.3f, gain=%.3f, offset=%.3f, samples=%d)",
                corr if np.isfinite(corr) else float("nan"),
                float(safe_gain),
                float(safe_offset),
                int(x_sel.size),
            )
        except Exception:
            pass

        gain, offset = float(safe_gain), float(safe_offset)

    return float(gain), float(offset), int(x_sel.size)

def _rescale_wcs_for_preview(
    wcs_obj,
    original_shape_hw: tuple[int, int],
    new_shape_hw: tuple[int, int],
    *,
    keep_distortion: bool = True,
):
    """
    Build a preview WCS consistent with the resized preview array.
    - keep_distortion=True  (default): clone original WCS and rescale CRPIX & CD/CDELT,
      preserving SIP/distortion (matches 3.2.4 behavior).
    - keep_distortion=False: fit a local TAN plane at the image center (no SIP),
      then rescale CD to the preview pixel size.
    """
    if not wcs_obj or not getattr(wcs_obj, "is_celestial", False):
        return None
    try:
        orig_h, orig_w = map(float, original_shape_hw)
        new_h, new_w = map(float, new_shape_hw)
    except Exception:
        return None
    if new_h <= 0 or new_w <= 0:
        return None

    scale_y = orig_h / new_h
    scale_x = orig_w / new_w

    # Fast path: preserve SIP/distortion (regression to 3.2.4)
    if keep_distortion:
        try:
            preview_wcs = wcs_obj.deepcopy()
        except Exception:
            preview_wcs = copy.deepcopy(wcs_obj)
        if preview_wcs is None:
            return None
        try:
            if hasattr(preview_wcs, "wcs") and preview_wcs.wcs is not None:
                if preview_wcs.wcs.crpix is not None and preview_wcs.wcs.crpix.size >= 2:
                    preview_wcs.wcs.crpix[0] = (float(preview_wcs.wcs.crpix[0]) - 0.5) / scale_x + 0.5
                    preview_wcs.wcs.crpix[1] = (float(preview_wcs.wcs.crpix[1]) - 0.5) / scale_y + 0.5
                # Rescale CD or CDELT
                try:
                    cd = preview_wcs.wcs.cd
                except Exception:
                    cd = None
                if cd is not None:
                    preview_wcs.wcs.cd[0, :] *= scale_x
                    preview_wcs.wcs.cd[1, :] *= scale_y
                else:
                    try:
                        preview_wcs.wcs.cdelt[0] *= scale_x
                        preview_wcs.wcs.cdelt[1] *= scale_y
                    except Exception:
                        pass
            try:
                preview_wcs.pixel_shape = (int(round(new_w)), int(round(new_h)))
            except Exception:
                pass
            try:
                preview_wcs.fix()
            except Exception:
                pass
            return preview_wcs
        except Exception:
            return None

    # Proper linearization: fit a TAN plane at center, then rescale CD
    if not (ASTROPY_WCS_AVAILABLE_IN_UTILS and AstropyWCS is not None):
        return None
    try:
        # Center in original pixel grid
        cx = (orig_w - 1.0) * 0.5
        cy = (orig_h - 1.0) * 0.5
        # Sample world at center and ±1px to estimate Jacobian
        c0 = wcs_obj.pixel_to_world(cx, cy)
        cx1 = wcs_obj.pixel_to_world(cx + 1.0, cy)
        cy1 = wcs_obj.pixel_to_world(cx, cy + 1.0)
        # Convert small offsets to degrees in a local tangent approx
        ra0 = float(c0.ra.deg)
        dec0 = float(c0.dec.deg)
        cosd = max(1e-6, abs(math.cos(math.radians(dec0))))
        d_ra_dx = (float(cx1.ra.deg) - ra0) * cosd
        d_dec_dx = float(cx1.dec.deg) - dec0
        d_ra_dy  = (float(cy1.ra.deg) - ra0) * cosd
        d_dec_dy = float(cy1.dec.deg) - dec0
        # CD in deg/px at original scale
        cd = np.array([[d_ra_dx, d_ra_dy],
                       [d_dec_dx, d_dec_dy]], dtype=np.float64)
        # Rescale to preview pixels: preview pixel = scale_x * original px (x), scale_y (y)
        cd[0, 0] *= scale_x; cd[0, 1] *= scale_y
        cd[1, 0] *= scale_x; cd[1, 1] *= scale_y
        # Build a clean TAN WCS
        hdr = {}
        hdr["NAXIS"]  = 2
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRVAL1"] = ra0
        hdr["CRVAL2"] = dec0
        hdr["CRPIX1"] = (new_w + 1.0) * 0.5
        hdr["CRPIX2"] = (new_h + 1.0) * 0.5
        hdr["CD1_1"]  = cd[0, 0]; hdr["CD1_2"] = cd[0, 1]
        hdr["CD2_1"]  = cd[1, 0]; hdr["CD2_2"] = cd[1, 1]
        preview_wcs = AstropyWCS(header=hdr, naxis=2, relax=True)
        try:
            preview_wcs.pixel_shape = (int(round(new_w)), int(round(new_h)))
        except Exception:
            pass
        try:
            preview_wcs.fix()
        except Exception:
            pass
        return preview_wcs
    except Exception:
        return None
    
def create_downscaled_luminance_preview(
    image: np.ndarray,
    wcs_obj,
    preview_size: int = 256,
) -> tuple[np.ndarray | None, object | None]:
    """Return a luminance preview and adjusted WCS for quick overlap tests."""

    if image is None:
        return None, None
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[..., 0]
        else:
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            return arr, None
    arr = np.where(np.isfinite(arr), arr, np.nan)
    h, w = arr.shape
    preview = arr
    preview_wcs = None
    if preview_size and max(h, w) > preview_size:
        scale = max(h, w) / float(preview_size)
        new_w = max(8, int(round(w / scale)))
        new_h = max(8, int(round(h / scale)))
        preview = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        preview = preview.astype(np.float32, copy=False)
        preview_wcs = _rescale_wcs_for_preview(wcs_obj, (h, w), (new_h, new_w))
    else:
        preview = arr.astype(np.float32, copy=False)
        if wcs_obj is not None:
            preview_wcs = _rescale_wcs_for_preview(wcs_obj, (h, w), (h, w))
    return preview, preview_wcs


def solve_global_affine(num_tiles: int, pair_entries, anchor_index: int = 0):
    """Résout les gains/offsets globaux à partir des couples (a,b)."""

    if num_tiles <= 0:
        return {}
    if not pair_entries:
        return {i: (1.0, 0.0) for i in range(num_tiles)}

    anchor = int(max(0, min(anchor_index, num_tiles - 1)))
    rows = []
    rhs = []
    for entry in pair_entries:
        i = entry[0]
        j = entry[1]
        a_ij = float(entry[2])
        b_ij = float(entry[3])
        weight = max(1.0, float(entry[4]))
        if not np.isfinite(a_ij) or abs(a_ij) < 1e-6:
            continue
        sqrt_w = math.sqrt(weight)
        row_gain = np.zeros(2 * num_tiles, dtype=np.float64)
        row_gain[i] = -sqrt_w
        row_gain[j] = a_ij * sqrt_w
        rows.append(row_gain)
        rhs.append(0.0)

        row_offset = np.zeros(2 * num_tiles, dtype=np.float64)
        row_offset[num_tiles + i] = sqrt_w
        row_offset[num_tiles + j] = -sqrt_w
        row_offset[j] = -b_ij * sqrt_w
        rows.append(row_offset)
        rhs.append(0.0)

    # Contraintes d'ancrage : gain=1, offset=0
    row_anchor_gain = np.zeros(2 * num_tiles, dtype=np.float64)
    row_anchor_gain[anchor] = 1.0
    rows.append(row_anchor_gain)
    rhs.append(1.0)

    row_anchor_offset = np.zeros(2 * num_tiles, dtype=np.float64)
    row_anchor_offset[num_tiles + anchor] = 1.0
    rows.append(row_anchor_offset)
    rhs.append(0.0)

    if not rows:
        return {i: (1.0, 0.0) for i in range(num_tiles)}

    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=np.float64)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return {i: (1.0, 0.0) for i in range(num_tiles)}

    gains = sol[:num_tiles]
    offsets = sol[num_tiles:]
    result = {}
    for idx in range(num_tiles):
        g = float(gains[idx])
        o = float(offsets[idx])
        if not np.isfinite(g) or abs(g) < 1e-6:
            g = 1.0
        if not np.isfinite(o):
            o = 0.0
        result[idx] = (g, o)
    return result


def compute_intertile_affine_calibration(
    tile_data_with_wcs,
    final_output_wcs,
    final_output_shape_hw,
    preview_size: int = 512,
    min_overlap_fraction: float = 0.05,
    sky_percentile: tuple[float, float] | list[float] = (30.0, 70.0),
    robust_clip_sigma: float = 2.5,
    use_auto_intertile: bool = False,
    logger=None,
    progress_callback=None,
):
    """Calcule des corrections affine (gain/offset) inter-tuiles avant reprojection."""

    if tile_data_with_wcs is None or len(tile_data_with_wcs) < 2:
        return {}
    if reproject_interp is None or not ASTROPY_AVAILABLE_IN_UTILS:
        return {}
    try:
        header_full = final_output_wcs.to_header()
    except Exception:
        return {}

    try:
        h_full = int(final_output_shape_hw[0])
        w_full = int(final_output_shape_hw[1])
    except Exception:
        h_full = w_full = 0

    sky_low, sky_high = 30.0, 70.0
    try:
        if isinstance(sky_percentile, (list, tuple)) and len(sky_percentile) >= 2:
            sky_low = float(sky_percentile[0])
            sky_high = float(sky_percentile[1])
            if sky_low > sky_high:
                sky_low, sky_high = sky_high, sky_low
    except Exception:
        sky_low, sky_high = 30.0, 70.0

    wcs_list = [wcs for _data, wcs in tile_data_with_wcs]
    shapes_hw = []
    luminance_tiles = []
    for data, _wcs in tile_data_with_wcs:
        arr = np.asarray(data)
        if arr.ndim == 3 and arr.shape[-1] == 0:
            luminance_tiles.append(None)
            shapes_hw.append((0, 0))
            continue
        luminance = _extract_luminance_plane(arr)
        luminance_tiles.append(luminance)
        shapes_hw.append((luminance.shape[0], luminance.shape[1]))

    num_tiles = len(tile_data_with_wcs)

    try:
        preview_size = int(preview_size)
    except Exception:
        preview_size = 512
    preview_size = max(128, preview_size)

    try:
        min_overlap_fraction = float(min_overlap_fraction)
    except Exception:
        min_overlap_fraction = 0.05
    if not math.isfinite(min_overlap_fraction):
        min_overlap_fraction = 0.05
    if min_overlap_fraction < 0:
        min_overlap_fraction = 0.0

    def _log_intertile(message: str, level: str = "INFO") -> None:
        prefixed = message if message.startswith("[Intertile]") else f"[Intertile] {message}"
        level_upper = str(level).upper()
        if logger is not None:
            try:
                if level_upper in {"WARN", "WARNING"}:
                    logger.warning(prefixed)
                elif level_upper in {"ERROR", "CRITICAL"}:
                    logger.error(prefixed)
                elif level_upper in {"DEBUG", "DEBUG_DETAIL"}:
                    logger.debug(prefixed)
                else:
                    logger.info(prefixed)
            except Exception:
                pass
        if progress_callback:
            try:
                progress_callback(prefixed, None, level_upper)
            except Exception:
                pass

    if use_auto_intertile and num_tiles > 20:
        tuned_preview = max(1024, preview_size)
        tuned_overlap = min(min_overlap_fraction, 0.01)
        preview_size = int(tuned_preview)
        min_overlap_fraction = float(tuned_overlap)
        _log_intertile(
            f"Auto-tune enabled for {num_tiles} tiles — using preview={preview_size}, min_overlap={min_overlap_fraction:.4f}",
            level="INFO",
        )

    base_min_overlap = float(min_overlap_fraction)

    candidates: list[float] = []
    seen_thresholds: set[float] = set()

    def _add_candidate(value: float) -> None:
        try:
            val = float(value)
        except Exception:
            return
        if not math.isfinite(val):
            return
        if val < 0:
            val = 0.0
        key = round(val, 6)
        if key in seen_thresholds:
            return
        seen_thresholds.add(key)
        candidates.append(val)

    _add_candidate(base_min_overlap)
    for fallback_threshold in (0.03, 0.02, 0.01):
        if fallback_threshold < base_min_overlap - 1e-6:
            _add_candidate(fallback_threshold)
    if not candidates:
        candidates.append(0.0)

    overlaps = []
    effective_min_overlap = base_min_overlap
    for threshold in candidates:
        overlaps = estimate_overlap_pairs(
            wcs_list,
            shapes_hw,
            final_output_wcs,
            (h_full, w_full),
            min_overlap_fraction=threshold,
        )
        _log_intertile(
            f"Overlap pairs at min_overlap={threshold:.4f}: {len(overlaps)}",
            level="INFO",
        )
        if overlaps:
            effective_min_overlap = threshold
            if threshold + 1e-6 < base_min_overlap:
                _log_intertile(
                    f"Using relaxed min_overlap={threshold:.4f} (initial {base_min_overlap:.4f}).",
                    level="INFO",
                )
            break

    if not overlaps:
        _log_intertile(
            "No overlap pairs found after retries — applying GLOBAL fallback (median normalization).",
            level="WARN",
        )
        try:
            use_gpu = False
            xp = np
            cp = None
            try:
                import cupy as _cp  # type: ignore

                if gpu_is_available():
                    xp = _cp
                    cp = _cp
                    use_gpu = True
                    _log_intertile(
                        "CuPy GPU detected — normalization on GPU.",
                        level="INFO_DETAIL",
                    )
            except Exception:
                xp = np
                use_gpu = False

            medians = []
            for data, _ in tile_data_with_wcs:
                if data is None:
                    continue
                arr = xp.asarray(data, dtype=xp.float32)
                med = float(xp.median(arr).get() if use_gpu else np.median(arr))
                if np.isfinite(med) and med > 0:
                    medians.append(med)

            if not medians:
                _log_intertile("No valid medians found for normalization.", level="WARN")
                return {}

            global_median = float(np.median(medians))
            _log_intertile(
                f"Global normalization median reference = {global_median:.4f}",
                level="INFO_DETAIL",
            )

            new_tile_data = []
            for data, wcs_obj in tile_data_with_wcs:
                if data is None:
                    new_tile_data.append((data, wcs_obj))
                    continue
                arr = xp.asarray(data, dtype=xp.float32)
                med = float(xp.median(arr).get() if use_gpu else np.median(arr))
                scale = (global_median / med) if med and np.isfinite(med) and med != 0 else 1.0
                scaled = arr * scale
                if use_gpu and cp is not None:
                    scaled = cp.asnumpy(scaled)  # type: ignore
                new_tile_data.append((scaled, wcs_obj))
            tile_data_with_wcs[:] = new_tile_data

            _log_intertile("Applied global normalization to all master tiles.", level="INFO")

        except Exception as e_norm:
            _log_intertile(f"Global normalization fallback failed: {e_norm}", level="ERROR")

        return {}

    min_overlap_fraction = effective_min_overlap
    _log_intertile(
        f"Using: preview={preview_size}, min_overlap={effective_min_overlap:.4f}, sky=({sky_low:.1f},{sky_high:.1f}), clip={robust_clip_sigma:.2f}, pairs={len(overlaps)}",
        level="INFO",
    )

    try:
        from astropy.wcs import WCS as _WCS
    except Exception:
        return {}

    pair_entries = []
    connectivity = np.zeros(len(tile_data_with_wcs), dtype=np.float64)
    preview_size = max(128, int(preview_size))

    for idx, overlap in enumerate(overlaps, 1):
        i = overlap["i"]
        j = overlap["j"]
        bbox = overlap["bbox"]
        weight = float(overlap.get("weight", 1.0))
        if luminance_tiles[i] is None or luminance_tiles[j] is None:
            continue
        x0, x1, y0, y1 = bbox
        sub_w = max(1, x1 - x0)
        sub_h = max(1, y1 - y0)
        header = header_full.copy()
        header["NAXIS1"] = sub_w
        header["NAXIS2"] = sub_h
        if "CRPIX1" in header:
            header["CRPIX1"] = float(header["CRPIX1"]) - x0
        if "CRPIX2" in header:
            header["CRPIX2"] = float(header["CRPIX2"]) - y0
        try:
            target_wcs = _WCS(header)
        except Exception:
            continue
        try:
            reproj_i, _ = reproject_interp(
                (luminance_tiles[i], wcs_list[i]), target_wcs, shape_out=(sub_h, sub_w)
            )
            reproj_j, _ = reproject_interp(
                (luminance_tiles[j], wcs_list[j]), target_wcs, shape_out=(sub_h, sub_w)
            )
        except Exception:
            continue
        if reproj_i is None or reproj_j is None:
            continue
        arr_i = np.asarray(reproj_i, dtype=np.float32)
        arr_j = np.asarray(reproj_j, dtype=np.float32)
        if arr_i.size == 0 or arr_j.size == 0:
            continue
        max_dim = max(arr_i.shape[0], arr_i.shape[1])
        if max_dim > preview_size:
            scale = preview_size / max_dim
            new_w = max(8, int(round(arr_i.shape[1] * scale)))
            new_h = max(8, int(round(arr_i.shape[0] * scale)))
            arr_i = cv2.resize(arr_i, (new_w, new_h), interpolation=cv2.INTER_AREA)
            arr_j = cv2.resize(arr_j, (new_w, new_h), interpolation=cv2.INTER_AREA)
        valid = np.isfinite(arr_i) & np.isfinite(arr_j)
        if not np.any(valid):
            continue
        sample_i = arr_i[valid]
        sample_j = arr_j[valid]
        if sample_i.size < 64:
            continue
        sky_proxy = 0.5 * (sample_i + sample_j)
        try:
            p_low = np.percentile(sky_proxy, sky_low)
            p_high = np.percentile(sky_proxy, sky_high)
        except Exception:
            p_low = np.nanmin(sky_proxy)
            p_high = np.nanmax(sky_proxy)
        if not np.isfinite(p_low) or not np.isfinite(p_high):
            continue
        if p_high <= p_low:
            p_low = np.nanmin(sky_proxy)
            p_high = np.nanmax(sky_proxy)
        mask = (sky_proxy >= p_low) & (sky_proxy <= p_high)
        if mask.sum() < 32:
            mask = np.ones_like(sky_proxy, dtype=bool)
        x_samples = sample_i[mask]
        y_samples = sample_j[mask]
        fit = robust_affine_fit(x_samples, y_samples, clip_sigma=robust_clip_sigma)
        if fit is None:
            continue
        a_ij, b_ij = fit
        pair_entries.append((i, j, a_ij, b_ij, weight))

        a_ij, b_ij = fit
        pair_entries.append((i, j, a_ij, b_ij, weight))
        connectivity[i] += weight
        connectivity[j] += weight

        # [ETA] Tick fin de traitement de la paire idx
        if progress_callback:
            try:
                progress_callback("phase5_intertile_pairs", int(idx), int(len(overlaps)))
            except Exception:
                pass

        if progress_callback and idx % 5 == 0:
            try:
                progress_callback("phase5_intertile", idx, len(overlaps))
            except Exception:
                pass

    if not pair_entries:
        return {}

    anchor = int(np.argmax(connectivity)) if np.any(connectivity > 0) else 0
    solution = solve_global_affine(len(tile_data_with_wcs), pair_entries, anchor_index=anchor)
    if progress_callback:
        try:
            progress_callback(
                "Ensuring match_background=True for final coadd.",
                None,
                "DEBUG_DETAIL",
            )
        except Exception:
            pass

    return solution
# Les filtres VerifyWarning sont maintenant dans le try/except d'Astropy ci-dessus.







# DANS zemosaic_utils.py

# (Les imports et la définition de ASTROPY_AVAILABLE_IN_UTILS, fits_module_for_utils restent les mêmes)
# ...

def load_and_validate_fits(filepath,
                           normalize_to_float32=True, # Si True, normalise la sortie à [0,1]
                           attempt_fix_nonfinite=True,
                           progress_callback=None):
    filename = Path(filepath).name

    def _log_util(message, level="DEBUG_DETAIL"):
        if progress_callback and callable(progress_callback):
            progress_callback(f"  [ZMU_LoadVal] {message}", None, level)
        else:
            print(f"  [ZMU_LoadVal PRINTFALLBACK] {level}: {message}")

    _log_util(f"Début chargement (V3 - BZERO/BSCALE affiné). Fichier: '{filename}'. NormalizeOutput01={normalize_to_float32}, FixNonFinite={attempt_fix_nonfinite}", "DEBUG")

    data_raw_from_fits = None  # Données telles que lues par fits.open
    header = None
    header_for_fallback = fits_module_for_utils.Header()
    info = {}

    try:
        _log_util(f"Tentative fits_module_for_utils.open('{filepath}', do_not_scale_image_data=True)...", "DEBUG_DETAIL")
        with fits_module_for_utils.open(filepath, memmap=False, do_not_scale_image_data=True) as hdul:
            # ... (logique de sélection de hdu_img inchangée) ...
            _log_util(f"fits_module_for_utils.open OK. Nombre HDUs: {len(hdul) if hdul else 0}", "DEBUG_DETAIL")
            if not hdul:
                _log_util(f"REJET: Fichier FITS vide ou corrompu (hdul est None/vide).", "WARN")
                return None, header_for_fallback, info

            hdu_img = None; img_hdu_idx = -1
            _log_util(f"Recherche HDU image...", "DEBUG_DETAIL")
            for idx, hdu_item in enumerate(hdul):
                is_image_attr = getattr(hdu_item, 'is_image', False)
                has_data_attr = hasattr(hdu_item, 'data')
                if is_image_attr and has_data_attr and hdu_item.data is not None:
                    _log_util(f"  HDU {idx} est image. Shape brute: {hdu_item.data.shape if hasattr(hdu_item.data, 'shape') else 'N/A'}, Dtype brut: {hdu_item.data.dtype if hasattr(hdu_item.data, 'dtype') else 'N/A'}", "DEBUG_DETAIL")
                    hdu_name = getattr(hdu_item, 'name', 'N/A_NAME')
                    if idx == 0 or (isinstance(hdu_name, str) and hdu_name.upper() in ['SCI', 'IMAGE', 'PRIMARY']):
                        hdu_img = hdu_item; img_hdu_idx = idx
                        _log_util(f"  HDU prioritaire {img_hdu_idx} ('{hdu_name}') sélectionnée.", "DEBUG_DETAIL"); break
            if hdu_img is None:
                _log_util(f"Pas de HDU prioritaire. Recherche première HDU image...", "DEBUG_DETAIL")
                for idx, hdu_item in enumerate(hdul):
                    is_image_attr = getattr(hdu_item, 'is_image', False); has_data_attr = hasattr(hdu_item, 'data')
                    if is_image_attr and has_data_attr and hdu_item.data is not None:
                        hdu_img = hdu_item; img_hdu_idx = idx
                        _log_util(f"  Première HDU image {img_hdu_idx} sélectionnée.", "DEBUG_DETAIL"); break
            if hdu_img is None or hdu_img.data is None:
                _log_util(f"REJET: Aucune HDU image valide avec données.", "WARN")
                if ASTROPY_AVAILABLE_IN_UTILS and len(hdul) > 0 and hasattr(hdul[0], 'header') and hdul[0].header:
                    header_for_fallback = hdul[0].header.copy()
                return None, header_for_fallback, info

            data_raw_from_fits = hdu_img.data # Peut être int16, uint16, float32, etc.
            alpha_mask = None
            for hdu_extra in hdul:
                name = getattr(hdu_extra, "name", "")
                if isinstance(name, str) and name.upper() == "ALPHA" and hdu_extra.data is not None:
                    alpha_mask = np.asarray(hdu_extra.data)
                    if alpha_mask.ndim > 2:
                        alpha_mask = alpha_mask[..., 0]
                    alpha_mask = np.asarray(alpha_mask, dtype=np.uint8, copy=False)
                    break
            if alpha_mask is not None:
                info["alpha_mask"] = alpha_mask
            header = hdu_img.header.copy(); header_for_fallback = header.copy()
            
            _log_util(f"Données lues HDU {img_hdu_idx}. Shape brute: {data_raw_from_fits.shape}, Dtype brut: {data_raw_from_fits.dtype}", "DEBUG")
            _log_util(f"  Range brut (depuis FITS): [{np.min(data_raw_from_fits) if data_raw_from_fits.size>0 else 'N/A'}, {np.max(data_raw_from_fits) if data_raw_from_fits.size>0 else 'N/A'}]", "DEBUG")

            # --- Conversion en float64 pour le scaling ADU et autres opérations ---
            data_scaled_f64 = data_raw_from_fits.astype(np.float64)
            _log_util(f"Converti en float64. Range: [{np.min(data_scaled_f64):.1f}, {np.max(data_scaled_f64):.1f}]", "DEBUG_DETAIL")

            # --- Application BZERO/BSCALE si `do_not_scale_image_data=True` a empêché Astropy ---
            # Et si les données d'origine étaient des entiers.
            if data_raw_from_fits.dtype.kind == 'i': 
                bzero = header.get('BZERO', 0.0)
                bscale = header.get('BSCALE', 1.0)
                if abs(bscale - 1.0) > 1e-6 or abs(bzero) > 1e-6:
                    _log_util(f"Application BZERO={bzero}, BSCALE={bscale} (car do_not_scale_image_data=True et dtype entier).", "INFO_DETAIL")
                    # data_scaled_f64 était déjà une copie de data_raw_from_fits en float64
                    data_scaled_f64 = data_scaled_f64 * bscale + bzero 
                    _log_util(f"  Après BZERO/BSCALE: Range [{np.min(data_scaled_f64):.1f}, {np.max(data_scaled_f64):.1f}], Dtype: {data_scaled_f64.dtype}", "DEBUG")
                else:
                    _log_util(f"BZERO/BSCALE triviaux ou absents pour dtype entier. Pas de scaling manuel BZ/BS.", "DEBUG_DETAIL")
            else:
                 _log_util(f"Dtype brut ({data_raw_from_fits.dtype}) non entier. Pas de scaling BZERO/BSCALE manuel appliqué.", "DEBUG_DETAIL")
            # À ce stade, data_scaled_f64 devrait être en float64 et avoir la bonne plage ADU (ex: 0-65535 pour un Seestar FITS)

            # --- Transposition si nécessaire ---
            data_transposed_f64 = data_scaled_f64
            axis_orig = "HWC"
            if data_scaled_f64.ndim == 3:
                if data_scaled_f64.shape[0] in [1, 3, 4] and data_scaled_f64.shape[1] > 4 and data_scaled_f64.shape[2] > 4:
                    _log_util(f"Shape 3D {data_scaled_f64.shape} type CxHxW. Transposition vers HxWxC...", "INFO_DETAIL")
                    data_transposed_f64 = np.moveaxis(data_scaled_f64, 0, -1)
                    axis_orig = "CHW"
                    _log_util(f"  Shape après transposition: {data_transposed_f64.shape}", "DEBUG_DETAIL")
                elif data_scaled_f64.shape[2] in [1, 3, 4] and data_scaled_f64.shape[0] > 4 and data_scaled_f64.shape[1] > 4:
                    _log_util(f"Shape 3D {data_scaled_f64.shape} déjà HxWxC.", "DEBUG_DETAIL")
                    axis_orig = "HWC"
                else:
                    _log_util(f"REJET: Shape 3D non supportée ({data_scaled_f64.shape}).", "WARN"); return None, header, info
            elif data_scaled_f64.ndim != 2:
                _log_util(f"REJET: Shape {data_scaled_f64.ndim}D non supportée.", "WARN"); return None, header, info

            info["axis_order_original"] = axis_orig
            
            _log_util(f"Après transposition (si 3D): Range [{np.min(data_transposed_f64):.1f}, {np.max(data_transposed_f64):.1f}], Dtype: {data_transposed_f64.dtype}", "DEBUG")
            
            # --- Gestion NaN/Inf ---
            data_cleaned_f64 = data_transposed_f64
            if attempt_fix_nonfinite:
                if not np.all(np.isfinite(data_transposed_f64)):
                    _log_util(f"AVERT: Données non finies détectées. Remplacement par 0.0.", "WARN")
                    data_cleaned_f64 = np.nan_to_num(data_transposed_f64, nan=0.0, posinf=0.0, neginf=0.0)
                    _log_util(f"  Après nan_to_num: Range [{np.min(data_cleaned_f64):.1f}, {np.max(data_cleaned_f64):.1f}]", "DEBUG_DETAIL")
            
            # --- Conversion finale en float32 pour la sortie ---
            image_data_final_float32 = data_cleaned_f64.astype(np.float32)
            _log_util(f"Converti en float32 final. Range: [{np.min(image_data_final_float32):.3g}, {np.max(image_data_final_float32):.3g}]", "DEBUG")

            # --- Normalisation optionnelle à [0,1] ---
            if normalize_to_float32:
                _log_util(f"Normalisation 0-1 demandée...", "DEBUG_DETAIL")
                min_val_norm, max_val_norm = np.nanmin(image_data_final_float32), np.nanmax(image_data_final_float32)
                _log_util(f"  Min/Max pour normalisation 0-1: [{min_val_norm:.3g}, {max_val_norm:.3g}]", "DEBUG_DETAIL")
                
                if np.isfinite(min_val_norm) and np.isfinite(max_val_norm) and (max_val_norm > min_val_norm + 1e-9):
                    image_data_final_float32 = (image_data_final_float32 - min_val_norm) / (max_val_norm - min_val_norm)
                    image_data_final_float32 = np.clip(image_data_final_float32, 0.0, 1.0)
                elif np.any(np.isfinite(image_data_final_float32)): # Image constante non-Nan/Inf
                    image_data_final_float32 = np.full_like(image_data_final_float32, 0.5, dtype=np.float32)
                    _log_util(f"  Image constante, normalisée à 0.5.", "DEBUG_DETAIL")
                else: # Tout NaN ou Inf
                    image_data_final_float32 = np.zeros_like(image_data_final_float32, dtype=np.float32)
                    _log_util(f"  Image non-finie, normalisée à 0.0.", "DEBUG_DETAIL")
                _log_util(f"Normalisation 0-1 effectuée. Range après: [{np.nanmin(image_data_final_float32):.3f}, {np.nanmax(image_data_final_float32):.3f}]", "DEBUG_DETAIL")
            else:
                _log_util(f"Pas de normalisation 0-1 (ADU). Range final: [{np.nanmin(image_data_final_float32):.3g}, {np.nanmax(image_data_final_float32):.3g}]", "DEBUG_DETAIL")

            _log_util(
                f"FIN chargement '{filename}'. Shape: {image_data_final_float32.shape}, Dtype: {image_data_final_float32.dtype}, "
                f"Range: [{np.nanmin(image_data_final_float32):.3g} - {np.nanmax(image_data_final_float32):.3g}], Mean: {np.nanmean(image_data_final_float32):.3g}",
                "INFO",
            )
            return image_data_final_float32, header, info

    except FileNotFoundError:
        _log_util(f"ERREUR CRITIQUE: Fichier non trouvé: '{filepath}'", "ERROR")
        return None, header_for_fallback, info
    except MemoryError as me:
        _log_util(f"ERREUR CRITIQUE MÉMOIRE: {me}", "ERROR"); return None, header_for_fallback, info
    except Exception as e:
        _log_util(f"ERREUR INATTENDUE chargement/validation '{filename}': {type(e).__name__} - {e}", "ERROR")
        if progress_callback and hasattr(progress_callback.__self__ if hasattr(progress_callback, '__self__') else progress_callback, 'logger'):
             logger_instance = progress_callback.__self__.logger if hasattr(progress_callback, '__self__') else _log_util.getLogger("ZeMosaicUtilsUnknownContext")
             logger_instance.error(f"Traceback pour load_and_validate_fits (fichier: {filename}):", exc_info=True)
        elif progress_callback:
             progress_callback(f"  [ZMU_LoadVal TRACEBACK] {traceback.format_exc(limit=3)}", None, "ERROR")
        else:
            traceback.print_exc(limit=3)
        return None, header_for_fallback, info





def crop_image_and_wcs(
    image_data_hwc: np.ndarray, 
    wcs_obj, # Type hint générique pour éviter les problèmes avec Pylance si WCS n'est pas toujours AstropyWCSBase
    crop_percentage_per_side: float,
    progress_callback: callable = None
) -> tuple[np.ndarray | None, object | None]: # Type de retour générique pour WCS
    """
    Rogne une image (HWC ou HW) d'un certain pourcentage sur chaque côté
    et ajuste l'objet WCS Astropy correspondant.

    Args:
        image_data_hwc (np.ndarray): Tableau image (H, W, C) ou (H, W).
        wcs_obj (astropy.wcs.WCS): Objet WCS original.
        crop_percentage_per_side (float): Fraction (0.0 à <0.5) à rogner de chaque côté.
        progress_callback (callable, optional): Fonction pour les logs.

    Returns:
        tuple: (cropped_image_data, cropped_wcs_obj) ou (None, None) si erreur,
               ou (image_data_hwc, wcs_obj) si pas de rognage ou pas d'Astropy.
    """
    # Définir un logger local simple pour cette fonction si _internal_logger n'est pas souhaité/disponible
    # ou utiliser progress_callback pour tous les messages.
    # Pour simplifier, j'utilise progress_callback s'il est fourni.
    def _pcb_crop(message, level="DEBUG_DETAIL", **kwargs):
        if progress_callback and callable(progress_callback):
            # Préfixer pour identifier l'origine du log si besoin
            progress_callback(f"[CropUtil] {message}", None, level, **kwargs)
        else:
            # Fallback simple si pas de callback (pourrait arriver si utilisé ailleurs)
            print(f"CROP_UTIL_LOG {level}: {message} {kwargs if kwargs else ''}")

    if image_data_hwc is None:
        _pcb_crop("Erreur: Données image en entrée est None.", lvl="ERROR")
        return None, None
    if wcs_obj is None : #  and ASTROPY_AVAILABLE_IN_UTILS (si wcs_obj peut être autre chose)
        _pcb_crop("Erreur: Objet WCS en entrée est None.", lvl="ERROR")
        return image_data_hwc, None # Retourner l'image, mais pas de WCS

    if not ASTROPY_AVAILABLE_IN_UTILS:
        _pcb_crop("AVERT: Astropy non disponible, impossible d'ajuster le WCS. Rognage de l'image seule effectué si demandé.", lvl="WARN")
        # On pourrait choisir de rogner l'image quand même et retourner un WCS non modifié,
        # ou retourner l'image et WCS originaux. Pour l'instant, on ne touche pas au WCS.
        # Si on rogne l'image, le WCS ne correspondra plus. Il vaut mieux ne pas rogner.
        if crop_percentage_per_side > 1e-4 :
             _pcb_crop(" Rognage annulé car WCS ne peut être ajusté.", lvl="WARN")
        return image_data_hwc, wcs_obj


    if not (0.0 <= crop_percentage_per_side < 0.5):
        if crop_percentage_per_side <= 1e-4 : # Pratiquement pas de rognage demandé
            # _pcb_crop("Pas de rognage demandé (pourcentage nul ou négligeable).", lvl="DEBUG_VERY_DETAIL")
            return image_data_hwc, wcs_obj
        else:
            _pcb_crop(f"Erreur: Pourcentage de rognage ({crop_percentage_per_side*100:.1f}%) hors limites [0, 50).", lvl="ERROR")
            return None, None # Erreur critique si le pourcentage est vraiment invalide


    original_shape = image_data_hwc.shape
    h_orig, w_orig = original_shape[0], original_shape[1]

    dh = int(h_orig * crop_percentage_per_side)
    dw = int(w_orig * crop_percentage_per_side)

    if (2 * dh >= h_orig) or (2 * dw >= w_orig):
        _pcb_crop(f"AVERT: Rognage demandé ({crop_percentage_per_side*100:.1f}%) est trop important pour les dimensions de l'image ({h_orig}x{w_orig}). Rognage annulé.", lvl="WARN")
        return image_data_hwc, wcs_obj

    _pcb_crop(f"Rognage de {dh}px (Haut/Bas) et {dw}px (Gauche/Droite).", lvl="DEBUG_DETAIL")

    if image_data_hwc.ndim == 3: # HWC
        cropped_image_data = image_data_hwc[dh : h_orig - dh, dw : w_orig - dw, :]
    elif image_data_hwc.ndim == 2: # HW
        cropped_image_data = image_data_hwc[dh : h_orig - dh, dw : w_orig - dw]
    else:
        _pcb_crop(f"Erreur: Dimensions d'image non supportées pour le rognage ({image_data_hwc.ndim}D).", lvl="ERROR")
        return None, None # Ou retourner l'original ? Mieux de signaler un échec.
        
    new_h, new_w = cropped_image_data.shape[0], cropped_image_data.shape[1]
    # _pcb_crop(f"Nouvelle shape après rognage: {new_h}x{new_w}", lvl="DEBUG_VERY_DETAIL")

    # Ajuster l'objet WCS
    try:
        # Tenter d'utiliser wcs.slice_like si l'objet WCS le supporte (Astropy >= 5.0)
        # et si l'objet WCS est bien un objet Astropy WCS.
        # Pour cela, il faudrait importer WCS d'Astropy ici.
        # from astropy.wcs import WCS as AstropyWCS (à mettre en haut du fichier utils)
        # if isinstance(wcs_obj, AstropyWCS) and hasattr(wcs_obj, 'slice_like'): # Nécessite l'import
        
        # Pour l'instant, faisons l'ajustement manuel de CRPIX, qui est plus universel
        # mais moins précis pour les WCS complexes.
        
        # Vérifier si wcs_obj est bien un objet WCS d'Astropy avant d'accéder à .wcs
        if not (hasattr(wcs_obj, 'wcs') and hasattr(wcs_obj.wcs, 'crpix')):
            _pcb_crop("AVERT: l'objet WCS ne semble pas être un WCS Astropy standard (manque .wcs.crpix). Ajustement WCS manuel impossible.", lvl="WARN")
            return cropped_image_data, wcs_obj # Retourner l'image rognée, mais le WCS original non modifié

        cropped_wcs_obj = wcs_obj.copy() # Travailler sur une copie

        if cropped_wcs_obj.wcs.crpix is not None:
            # CRPIX est 1-based dans le header FITS, mais l'attribut .wcs.crpix d'un objet WCS Astropy
            # est généralement interprété comme 1-based pour la manipulation via l'API haut niveau.
            # Lorsqu'on soustrait, on soustrait le nombre de pixels rognés du côté "origine" (gauche/bas).
            new_crpix1 = cropped_wcs_obj.wcs.crpix[0] - dw
            new_crpix2 = cropped_wcs_obj.wcs.crpix[1] - dh
            # Clamp to positive values to avoid invalid WCS after heavy cropping
            new_crpix1 = max(new_crpix1, 1.0)
            new_crpix2 = max(new_crpix2, 1.0)
            cropped_wcs_obj.wcs.crpix = [new_crpix1, new_crpix2]
        else:
            # Ce cas est peu probable si c'est un WCS valide, mais par sécurité.
            _pcb_crop("AVERT: wcs_obj.wcs.crpix est None. Impossible d'ajuster CRPIX.", lvl="WARN")
            # On pourrait essayer de retourner wcs_obj original, mais il ne correspondra plus.
            # Renvoyer None pour le WCS est plus sûr pour indiquer un problème.
            return cropped_image_data, None


        # Mettre à jour la taille de l'image de référence dans l'objet WCS
        # pixel_shape est (width, height) et 0-indexed pour l'API Python
        # NAXIS1/2 dans le header sont 1-indexed et (width, height)
        if hasattr(cropped_wcs_obj, 'pixel_shape'):
             cropped_wcs_obj.pixel_shape = (new_w, new_h)
        # Alternativement, si on manipule directement les clés FITS-like dans .wcs:
        if hasattr(cropped_wcs_obj.wcs, 'naxis1'): cropped_wcs_obj.wcs.naxis1 = new_w
        if hasattr(cropped_wcs_obj.wcs, 'naxis2'): cropped_wcs_obj.wcs.naxis2 = new_h
        
        # _pcb_crop("Ajustement WCS terminé.", lvl="DEBUG_DETAIL")
        return cropped_image_data, cropped_wcs_obj

    except Exception as e_wcs_crop:
        _pcb_crop(f"Erreur lors de l'ajustement du WCS: {e_wcs_crop}", lvl="ERROR")
        # Pas de logger global ici, on se fie au progress_callback
        # logger.error(f"Erreur lors de l'ajustement du WCS pour l'image rognée:", exc_info=True)
        return cropped_image_data, None # Retourner l'image rognée mais indiquer échec WCS





def debayer_image(img_norm_01, bayer_pattern="GRBG", progress_callback=None):
    def _log_util_debayer(message, level="DEBUG_DETAIL"):
        if progress_callback and callable(progress_callback): progress_callback(f"  [ZU Debayer] {message}", None, level)
        else: print(f"  [ZU Debayer PRINTFALLBACK] {level}: {message}")
    
    _log_util_debayer(f"Début debayering. Shape entrée: {img_norm_01.shape if hasattr(img_norm_01, 'shape') else 'N/A'}, Pattern: {bayer_pattern}", "DEBUG")
    if not isinstance(img_norm_01, np.ndarray): _log_util_debayer(f"ERREUR: Entrée pas ndarray.", "ERROR"); raise TypeError("Input must be NumPy array")
    if img_norm_01.ndim != 2: _log_util_debayer(f"ERREUR: Attend 2D.", "ERROR"); raise ValueError("Expects 2D image")

    img_uint16 = (np.clip(img_norm_01, 0.0, 1.0) * 65535.0).astype(np.uint16)
    _log_util_debayer(f"Converti en uint16 [0,65535] pour OpenCV.", "DEBUG_DETAIL")
    
    bayer_codes = {"GRBG": cv2.COLOR_BayerGR2RGB, "RGGB": cv2.COLOR_BayerRG2RGB, "GBRG": cv2.COLOR_BayerGB2RGB, "BGGR": cv2.COLOR_BayerBG2RGB}
    bayer_pattern_upper = bayer_pattern.upper()
    if bayer_pattern_upper not in bayer_codes: _log_util_debayer(f"ERREUR: Motif Bayer '{bayer_pattern}' non supporté.", "ERROR"); raise ValueError(f"Bayer pattern '{bayer_pattern}' not supported")
    
    try:
        _log_util_debayer(f"Appel cv2.cvtColor avec code {bayer_codes[bayer_pattern_upper]}...", "DEBUG_DETAIL")
        color_img_bgr_uint16 = cv2.cvtColor(img_uint16, bayer_codes[bayer_pattern_upper])
        color_img_rgb_uint16 = cv2.cvtColor(color_img_bgr_uint16, cv2.COLOR_BGR2RGB)
    except cv2.error as cv_err:
        _log_util_debayer(f"ERREUR OpenCV debayering (pattern: {bayer_pattern_upper}): {cv_err}", "ERROR")
        if progress_callback: progress_callback(f"  [ZU Debayer TRACEBACK] {traceback.format_exc(limit=2)}", None, "ERROR")
        raise ValueError(f"OpenCV error during debayering: {cv_err}")
    
    _log_util_debayer(f"Debayering OK. Conversion retour float32 [0,1]. Shape sortie: {color_img_rgb_uint16.shape}", "DEBUG")
    return color_img_rgb_uint16.astype(np.float32) / 65535.0


def detect_and_correct_hot_pixels(image, threshold=3.0, neighborhood_size=5,
                                  progress_callback=None, save_mask_path=None):
    def _log_util_hp(message, level="DEBUG_DETAIL"):
        if progress_callback and callable(progress_callback): progress_callback(f"  [ZU HotPix] {message}", None, level)
        else: print(f"  [ZU HotPix PRINTFALLBACK] {level}: {message}")

    _log_util_hp(
        f"Début détection/correction HP. Threshold: {threshold}, Neighborhood: {neighborhood_size}",
        "DEBUG",
    )
    if image is None: _log_util_hp("AVERT: Image entrée est None.", "WARN"); return None
    if not isinstance(image, np.ndarray): _log_util_hp(f"ERREUR: Entrée pas ndarray.", "ERROR"); return image 

    if neighborhood_size % 2 == 0: neighborhood_size += 1
    neighborhood_size = max(3, neighborhood_size); ksize = (neighborhood_size, neighborhood_size)

    original_dtype = image.dtype; img_float = image.astype(np.float32, copy=True)
    is_color = img_float.ndim == 3 and img_float.shape[-1] == 3
    _log_util_hp(f"Image {'couleur' if is_color else 'monochrome'}. Dtype original: {original_dtype}.", "DEBUG_DETAIL")
    
    try:
        mask_accum = None
        if is_color:
            mask_accum = np.zeros(img_float.shape, dtype=np.uint8)
            for c in range(img_float.shape[2]):
                channel = img_float[:, :, c]
                median_filtered = cv2.medianBlur(channel, neighborhood_size)
                mean_local = cv2.blur(channel, ksize)
                mean_sq_local = cv2.blur(channel**2, ksize)
                std_dev_local = np.sqrt(np.maximum(mean_sq_local - mean_local**2, 0))
                std_dev_floor = (
                    1e-5
                    if np.issubdtype(channel.dtype, np.floating)
                    else (
                        1.0
                        / (
                            np.iinfo(np.uint16).max
                            if np.max(channel) <= 1
                            else np.iinfo(channel.dtype).max
                            if np.issubdtype(channel.dtype, np.integer)
                            else (2**16 - 1)
                        )
                        if np.max(channel) > 1
                        else 1.0
                    )
                )
                std_dev_local_thresholded = np.maximum(std_dev_local, std_dev_floor)
                hot_pixels_mask = channel > (median_filtered + threshold * std_dev_local_thresholded)
                num_hot = np.sum(hot_pixels_mask)
                if num_hot > 0:
                    _log_util_hp(f"    Canal {c}: {num_hot} pixels chauds corrigés.", "DEBUG_DETAIL")
                channel[hot_pixels_mask] = median_filtered[hot_pixels_mask]
                mask_accum[..., c] = hot_pixels_mask
        else:  # Grayscale
            median_filtered = cv2.medianBlur(img_float, neighborhood_size)
            mean_local = cv2.blur(img_float, ksize)
            mean_sq_local = cv2.blur(img_float**2, ksize)
            std_dev_local = np.sqrt(np.maximum(mean_sq_local - mean_local**2, 0))
            std_dev_floor = (
                1e-5
                if np.issubdtype(img_float.dtype, np.floating)
                else (
                    1.0
                    / (
                        np.iinfo(np.uint16).max
                        if np.max(img_float) <= 1
                        else np.iinfo(img_float.dtype).max
                        if np.issubdtype(img_float.dtype, np.integer)
                        else (2**16 - 1)
                    )
                    if np.max(img_float) > 1
                    else 1.0
                )
            )
            std_dev_local_thresholded = np.maximum(std_dev_local, std_dev_floor)
            hot_pixels_mask = img_float > (median_filtered + threshold * std_dev_local_thresholded)
            num_hot = np.sum(hot_pixels_mask)
            if num_hot > 0:
                _log_util_hp(f"  Image N&B: {num_hot} pixels chauds corrigés.", "DEBUG_DETAIL")
            img_float[hot_pixels_mask] = median_filtered[hot_pixels_mask]
            mask_accum = hot_pixels_mask.astype(np.uint8)
        if save_mask_path:
            try:
                np.save(save_mask_path, mask_accum.astype(np.uint8))
                _log_util_hp(f"Masque HP sauvegardé vers {Path(save_mask_path).name}", "DEBUG_DETAIL")
            except Exception as e_save:
                _log_util_hp(f"ERREUR sauvegarde masque HP: {e_save}", "WARN")
        del mask_accum

        if np.issubdtype(original_dtype, np.integer):
            d_info = np.iinfo(original_dtype)
            corrected_img = np.clip(np.round(img_float), d_info.min, d_info.max).astype(original_dtype)
        else: corrected_img = img_float.astype(original_dtype)
        _log_util_hp(f"Correction HP terminée.", "DEBUG")
        return corrected_img
        
    except cv2.error as cv_err_hp: _log_util_hp(f"ERREUR OpenCV HotPix: {cv_err_hp}", "ERROR"); return image
    except Exception as e_hp: _log_util_hp(f"ERREUR Inattendue HotPix: {e_hp}", "ERROR"); return image


def make_radial_weight_map(height: int, width: int,
                           feather_fraction: float = 0.8,
                           shape_power: float = 2.0,
                           min_weight_floor: float = 0.05, # NOUVEAU PARAMÈTRE, défaut à 0.0 = pas de plancher
                           progress_callback: callable = None) -> np.ndarray: # Ajout de progress_callback pour les logs
    """
    Crée une carte de poids 2D avec une atténuation radiale basée sur une fonction cosinus.
    Le poids est de 1 au centre et décroît jusqu'à 0 (ou min_weight_floor) sur les bords.

    Args:
        height (int): Hauteur de l'image (nombre de lignes).
        width (int): Largeur de l'image (nombre de colonnes).
        feather_fraction (float): Fraction (0.1-1.0) de la demi-diagonale où le poids atteint zéro (ou le plancher).
        shape_power (float): Exposant appliqué à la fonction cosinus (ex: 2.0 pour cos²).
        min_weight_floor (float): Valeur plancher minimale pour les poids (0.0 à <1.0).
                                  Si > 0, les poids ne descendront pas en dessous de cette valeur.
        progress_callback (callable, optional): Fonction pour les logs.

    Returns:
        np.ndarray: Carte de poids 2D de forme (height, width).
    """
    # Alias local pour le callback, si fourni
    _pcb_radial = lambda msg_key, lvl="DEBUG_DETAIL", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else print(f"RADIAL_MAP_LOG {lvl}: {msg_key} {kwargs}")


    if not (0.1 <= feather_fraction <= 1.0):
        _pcb_radial(f"RadialMap: feather_fraction ({feather_fraction}) hors [0.1, 1.0]. Clampe à {np.clip(feather_fraction, 0.1, 1.0)}.", lvl="WARN")
        feather_fraction = np.clip(feather_fraction, 0.1, 1.0)

    if not (0.0 <= min_weight_floor < 1.0):
        _pcb_radial(f"RadialMap: min_weight_floor ({min_weight_floor}) hors [0.0, 1.0). Clampe à {np.clip(min_weight_floor, 0.0, 0.99)}.", lvl="WARN")
        min_weight_floor = np.clip(min_weight_floor, 0.0, 0.99)


    y_coords, x_coords = np.ogrid[:height, :width]
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0

    delta_x = x_coords - center_x
    delta_y = y_coords - center_y
    distance_from_center = np.sqrt(delta_x**2 + delta_y**2)

    max_distance_to_normalize = 0.5 * np.hypot(height, width)
    if max_distance_to_normalize < 1e-6:
        _pcb_radial("RadialMap: Image trop petite, retour poids uniforme 1.0.", lvl="DEBUG_DETAIL")
        return np.ones((height, width), dtype=np.float32)

    normalized_distance = distance_from_center / max_distance_to_normalize
    arg_cos = normalized_distance / feather_fraction
    
    # Calcul de la carte de poids basée sur le cosinus
    weight_map_cos = np.cos(0.5 * np.pi * np.clip(arg_cos, 0.0, 1.0)) ** shape_power
    
    # Application du plancher de poids si spécifié et > 0
    if min_weight_floor > 1e-6: # Utiliser un epsilon pour comparer les floats à zéro
        _pcb_radial(f"RadialMap: Application d'un plancher de poids minimal de {min_weight_floor:.3f}.", lvl="DEBUG_DETAIL")
        final_weight_map = np.maximum(weight_map_cos, min_weight_floor)
    else:
        final_weight_map = weight_map_cos
        
    return final_weight_map.astype(np.float32)

def stretch_auto_asifits_like(img_hwc_adu, p_low=0.5, p_high=99.8, 
                              asinh_a=0.01, apply_wb=True):
    """
    Étirement type ASIFitsViewer avec asinh et auto balance RVB.
    Fallback vers du linéaire si dynamique trop faible.
    """
    # Keep everything strictly in float32 to avoid huge float64 temporaries on large mosaics
    img = np.asarray(img_hwc_adu, dtype=np.float32)
    out = np.empty_like(img, dtype=np.float32)

    a32 = np.float32(asinh_a) if asinh_a is not None else np.float32(0.01)
    inv_asinh_den = np.float32(1.0) / np.arcsinh(np.float32(1.0) / a32)

    for c in range(3):
        chan = img[..., c]
        # percentile returns python floats/float64; cast to float32 to avoid upcasting chan
        vmin_f64, vmax_f64 = np.percentile(chan, [p_low, p_high])
        vmin = np.float32(vmin_f64)
        vmax = np.float32(vmax_f64)
        dv = vmax - vmin
        if not np.isfinite(dv) or dv < np.float32(1e-3):
            out[..., c].fill(0.0)
            continue
        # In-place normalize into out[..., c] to avoid an extra full-size array
        dst = out[..., c]
        np.subtract(chan, vmin, out=dst, dtype=np.float32)
        np.divide(dst, dv, out=dst)
        np.clip(dst, 0.0, 1.0, out=dst)
        # asinh stretch in-place, keeping float32
        # tmp = arcsinh(dst / a32) * inv_arcsinh(1/a)
        np.divide(dst, a32, out=dst)
        np.arcsinh(dst, out=dst)
        dst *= inv_asinh_den
        if not np.isfinite(np.nanmax(dst)) or np.nanmax(dst) < np.float32(0.05):
            # fallback to linear (already in dst)
            pass

    if apply_wb:
        avg_per_chan = np.mean(out, axis=(0, 1)).astype(np.float32)
        norm = np.max(avg_per_chan)
        if norm > 0:
            avg_per_chan /= norm
        else:
            avg_per_chan = np.ones_like(avg_per_chan)
        for c in range(3):
            denom = avg_per_chan[c]
            if denom > 1e-8:
                out[..., c] = (out[..., c] / np.float32(denom)).astype(np.float32, copy=False)

    return np.clip(out, 0, 1)

def stretch_percentile_rgb(img_hwc_adu, p_low=0.5, p_high=99.8, 
                           independent_channels=False, 
                           asinh_a=0.05, # 'a' parameter for AsinhStretch
                           progress_callback: callable = None):
    """
    Applique un stretch par percentiles avec une transformation asinh (via Astropy)
    à une image HWC. Sortie normalisée [0,1] pour affichage.

    Args:
        img_hwc_adu (np.ndarray): Image HWC (ou HW), float32, en ADU.
        p_low (float): Percentile inférieur pour définir le point noir (0-100).
        p_high (float): Percentile supérieur pour définir le point blanc (0-100).
        independent_channels (bool): Si True et image couleur, stretch chaque canal indépendamment.
                                     Si False, calcule les limites sur la luminance et applique
                                     le même vmin/vmax à chaque canal avant leur stretch individuel.
        asinh_a (float): Paramètre 'a' pour AsinhStretch. Contrôle la linéarité pour les faibles
                         signaux. Typiquement entre 0.01 (fort) et 1.0 (doux). 0.1 est un bon début.
        progress_callback (callable, optional): Fonction pour les logs.

    Returns:
        np.ndarray: Image HWC (ou HW) normalisée [0,1] après étirement asinh, float32.
                    Retourne une version basique étirée si Astropy.visualization n'est pas disponible.
    """
    if img_hwc_adu is None:
        if progress_callback:
            progress_callback("stretch_utils_error_input_none", lvl="ERROR")
        return None

    if not ASTROPY_VISUALIZATION_AVAILABLE:
        if progress_callback:
            progress_callback("stretch_utils_warn_astropy_viz_unavailable_fallback_linear", lvl="WARN")
        # Fallback très basique si astropy.visualization n'est pas là :
        try:
            # Utiliser les p_low/p_high comme pourcentages
            min_val, max_val = np.percentile(img_hwc_adu, [p_low, p_high])
            if not (np.isfinite(min_val) and np.isfinite(max_val)) or (max_val - min_val < 1e-5):
                return np.zeros_like(img_hwc_adu, dtype=np.float32)
            return np.clip((img_hwc_adu - min_val) / (max_val - min_val), 0, 1).astype(np.float32)
        except Exception as e_fallback:
            if progress_callback:
                progress_callback(f"stretch_utils_error_fallback_stretch: {e_fallback}", lvl="ERROR")
            return np.zeros_like(img_hwc_adu, dtype=np.float32) # Sécurité ultime

    
    img_float = img_hwc_adu.astype(np.float32, copy=False) 

    stretch = AsinhStretch(a=asinh_a)
    
    if img_float.ndim == 2: # Image monochrome
        # CORRECTION : Ajout de n_samples=None
        interval = PercentileInterval(p_low, p_high, n_samples=None)
        try:
            norm = ImageNormalize(img_float, interval=interval, stretch=stretch, clip=True)
            return norm(img_float).astype(np.float32)
        except Exception as e_norm_mono:
            if progress_callback:
                progress_callback(f"stretch_utils_error_norm_mono: {e_norm_mono}", lvl="ERROR")
            return np.zeros_like(img_float, dtype=np.float32) # Fallback
        
    elif img_float.ndim == 3 and img_float.shape[2] == 3: # Image couleur HWC
        stretched_img_array = np.empty_like(img_float)
        if independent_channels:
            for c in range(3):
                # CORRECTION : Ajout de n_samples=None
                interval = PercentileInterval(p_low, p_high, n_samples=None)
                try:
                    norm = ImageNormalize(img_float[..., c], interval=interval, stretch=stretch, clip=True)
                    stretched_img_array[..., c] = norm(img_float[..., c])
                except Exception as e_norm_color_ind:
                    if progress_callback:
                        progress_callback(f"stretch_utils_error_norm_color_ind_ch{c}: {e_norm_color_ind}", lvl="ERROR")
                    stretched_img_array[..., c] = np.zeros_like(img_float[..., c], dtype=np.float32) # Fallback pour ce canal
        else: # Stretch lié basé sur la luminance pour vmin/vmax, mais stretch asinh par canal
            try:
                luminance = 0.299 * img_float[..., 0] + 0.587 * img_float[..., 1] + 0.114 * img_float[..., 2]
                # CORRECTION : Ajout de n_samples=None
                interval = PercentileInterval(p_low, p_high, n_samples=None)
                vmin, vmax = interval.get_limits(luminance)
                
                if not (np.isfinite(vmin) and np.isfinite(vmax)) or (vmax - vmin < 1e-5) : # Vérifier si les limites sont valides
                    if progress_callback:
                        progress_callback(f"stretch_utils_warn_invalid_lum_limits_linked_stretch: vmin={vmin}, vmax={vmax}. Fallback sur stretch indépendant ou neutre.", lvl="WARN")
                    # Fallback : si les limites de luminance sont mauvaises, on pourrait faire un stretch indépendant prudent.
                    # Pour simplifier, on peut retourner une image neutre ou logguer une erreur plus sévère.
                    # Ici, on va essayer un stretch indépendant comme fallback.
                    for c_fb in range(3):
                        interval_fb = PercentileInterval(p_low, p_high, n_samples=None)
                        norm_fb = ImageNormalize(img_float[..., c_fb], interval=interval_fb, stretch=stretch, clip=True)
                        stretched_img_array[..., c_fb] = norm_fb(img_float[..., c_fb])
                    return stretched_img_array.astype(np.float32)

                for c in range(3):
                    norm = ImageNormalize(img_float[..., c], vmin=vmin, vmax=vmax, stretch=stretch, clip=True)
                    stretched_img_array[..., c] = norm(img_float[..., c])
            except Exception as e_norm_color_linked:
                if progress_callback:
                    progress_callback(f"stretch_utils_error_norm_color_linked: {e_norm_color_linked}", lvl="ERROR")
                # Fallback en cas d'erreur majeure dans le stretch lié
                for c_fb_err in range(3): stretched_img_array[..., c_fb_err] = np.clip(img_float[...,c_fb_err] / (np.max(img_float[...,c_fb_err]) if np.max(img_float[...,c_fb_err]) > 0 else 1.0) ,0,1)


        return stretched_img_array.astype(np.float32)
    else:
        if progress_callback:
            progress_callback("stretch_utils_warn_unsupported_shape_for_stretch", lvl="WARN", shape=str(img_float.shape if hasattr(img_float, 'shape') else 'N/A'))
        return img_float.astype(np.float32) # Retourner en float32 au cas où


def save_numpy_to_fits(image_data: np.ndarray, header, output_path: str, *, axis_order: str = "HWC", overwrite: bool = True) -> None:
    """Write a NumPy array to FITS without any scaling.

    Parameters
    ----------
    image_data : np.ndarray
        Array to save. Can be 2-D or 3-D.
    header : fits.Header or dict
        Header to write with the data.
    output_path : str
        Destination FITS path.
    axis_order : {"HWC", "CHW"}
        Interpretation of 3-D arrays. ``HWC`` means channels last and the array
        will be transposed to ``CxHxW`` for saving.
    overwrite : bool
        Overwrite existing file if True.
    """

    current_fits = fits_module_for_utils
    final_header = current_fits.Header()
    if header is not None:
        try:
            if hasattr(header, "to_header"):
                final_header.update(header.to_header(relax=True))
            else:
                final_header.update(header)
        except Exception:
            pass

    for key in ["SIMPLE", "BITPIX", "NAXIS", "EXTEND", "BSCALE", "BZERO"]:
        if key in final_header:
            try:
                del final_header[key]
            except KeyError:
                pass

    data_to_write = image_data
    if image_data.ndim == 3:
        ao = str(axis_order).upper()
        if ao == "HWC":
            data_to_write = np.moveaxis(image_data, -1, 0)
    hdu = current_fits.PrimaryHDU(data=data_to_write, header=final_header)
    hdul = current_fits.HDUList([hdu])
    hdul.writeto(output_path, overwrite=overwrite)
    if hasattr(hdul, "close"):
        hdul.close()



def _ensure_float32_no_nan(arr: np.ndarray) -> np.ndarray:
    """Return a float32 view/copy of ``arr`` with NaN/Inf replaced by zero."""

    arr_float = arr.astype(np.float32, copy=False)
    if not np.all(np.isfinite(arr_float)):
        arr_float = arr_float.copy()
        np.nan_to_num(arr_float, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr_float


def _rescale_to_u16(arr: np.ndarray) -> np.ndarray:
    """Scale float32 ``arr`` to the uint16 range using gentle percentile clipping."""

    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros(arr.shape, dtype=np.uint16)

    finite_values = arr[finite_mask]
    if finite_values.size == 0:
        return np.zeros(arr.shape, dtype=np.uint16)

    try:
        q_low = float(np.percentile(finite_values, 0.1))
        q_high = float(np.percentile(finite_values, 99.9))
    except Exception:
        q_low = float(np.nanmin(finite_values))
        q_high = float(np.nanmax(finite_values))

    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high <= q_low:
        q_low = float(np.nanmin(finite_values))
        q_high = float(np.nanmax(finite_values))

    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high <= q_low:
        return np.zeros(arr.shape, dtype=np.uint16)

    span = q_high - q_low
    vmin = q_low - 0.02 * span
    vmax = q_high + 0.02 * span

    global_min = float(np.nanmin(finite_values))
    global_max = float(np.nanmax(finite_values))
    if np.isfinite(global_min):
        vmin = max(vmin, global_min)
    if np.isfinite(global_max):
        vmax = min(vmax, global_max)

    if vmax <= vmin:
        vmax = vmin + 1.0

    scale = 65535.0 / (vmax - vmin)
    result = np.zeros(arr.shape, dtype=np.float32)
    result[finite_mask] = (arr[finite_mask] - vmin) * scale
    np.clip(result, 0.0, 65535.0, out=result)
    return result.astype(np.uint16)


def _to_int16_with_bzero(u16_arr: np.ndarray) -> np.ndarray:
    """Convert ``uint16`` samples to FITS-compliant int16 with ``BZERO=32768``."""

    shifted = u16_arr.astype(np.int32, copy=False) - 32768
    return np.clip(shifted, -32768, 32767).astype(np.int16, copy=False)


def _prepare_int16_header(base_header, width: int, height: int, *, extname: str | None = None):
    """Return a header for a 2-D int16 image with BSCALE/BZERO metadata."""

    header = base_header.copy()
    for key in (
        "SIMPLE",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "NAXIS3",
        "NAXIS4",
        "NAXIS5",
        "BSCALE",
        "BZERO",
        "EXTEND",
        "XTENSION",
        "CHECKSUM",
        "DATASUM",
        "PCOUNT",
        "GCOUNT",
        "CTYPE3",
        "CTYPE4",
    ):
        if key in header:
            del header[key]

    for key in ("DATAMIN", "DATAMAX"):
        if key in header:
            del header[key]

    if extname:
        header["EXTNAME"] = extname
    elif "EXTNAME" in header:
        del header["EXTNAME"]

    header["BITPIX"] = 16
    header["BSCALE"] = 1
    header["BZERO"] = 32768
    header["NAXIS"] = 2
    header["NAXIS1"] = int(width)
    header["NAXIS2"] = int(height)
    return header


def _update_dataminmax(header, data: np.ndarray) -> None:
    """Set ``DATAMIN``/``DATAMAX`` cards to match the stored integer range."""

    for key in ("DATAMIN", "DATAMAX"):
        if key in header:
            del header[key]
    try:
        raw_min = int(np.nanmin(data))
        raw_max = int(np.nanmax(data))
    except ValueError:
        raw_min = raw_max = 0
    header["DATAMIN"] = raw_min
    header["DATAMAX"] = raw_max


def _build_rgb_luminance_hdus(
    img_hwc: np.ndarray,
    base_header,
    fits_module,
    log_fn,
):
    """Create a luminance primary HDU and R/G/B extensions."""

    height, width, _ = img_hwc.shape
    r_plane, g_plane, b_plane = np.moveaxis(img_hwc, -1, 0)

    luminance = (0.2126 * r_plane + 0.7152 * g_plane + 0.0722 * b_plane).astype(
        np.float32, copy=False
    )

    u16_l = _rescale_to_u16(luminance)
    i16_l = _to_int16_with_bzero(u16_l)
    header_primary = _prepare_int16_header(base_header, width, height)
    _update_dataminmax(header_primary, i16_l)
    primary_hdu = fits_module.PrimaryHDU(data=i16_l, header=header_primary)
    primary_hdu.header['BSCALE'] = 1
    primary_hdu.header['BZERO'] = 32768
    log_fn(
        f"  SAVE_DEBUG: Luminance primary range [{np.min(i16_l)}, {np.max(i16_l)}]",
        "WARN",
    )

    extensions = []
    for name, plane in (("R", r_plane), ("G", g_plane), ("B", b_plane)):
        u16_plane = _rescale_to_u16(plane)
        i16_plane = _to_int16_with_bzero(u16_plane)
        header_plane = _prepare_int16_header(base_header, width, height, extname=name)
        _update_dataminmax(header_plane, i16_plane)
        image_hdu = fits_module.ImageHDU(data=i16_plane, header=header_plane, name=name)
        image_hdu.header['BSCALE'] = 1
        image_hdu.header['BZERO'] = 32768
        log_fn(
            f"  SAVE_DEBUG: Channel {name} range [{np.min(i16_plane)}, {np.max(i16_plane)}]",
            "WARN",
        )
        extensions.append(image_hdu)

    return primary_hdu, extensions


def _build_legacy_rgb_cube_hdu(
    img_hwc: np.ndarray,
    base_header,
    fits_module,
    log_fn,
):
    """Return a legacy RGB cube PrimaryHDU (channels-first)."""

    height, width, channels = img_hwc.shape
    planes = np.moveaxis(img_hwc, -1, 0)
    scaled_planes = []
    for name, plane in zip(("R", "G", "B"), planes):
        u16_plane = _rescale_to_u16(plane)
        i16_plane = _to_int16_with_bzero(u16_plane)
        log_fn(
            f"  SAVE_DEBUG: Legacy plane {name} range [{np.min(i16_plane)}, {np.max(i16_plane)}]",
            "WARN",
        )
        scaled_planes.append(i16_plane)

    cube_i16 = np.stack(scaled_planes, axis=0)
    header = _prepare_int16_header(base_header, width, height)
    header["NAXIS"] = 3
    header["NAXIS3"] = int(channels)
    header["CTYPE3"] = ("RGB", "Color Format")
    header["EXTNAME"] = "RGB"
    _update_dataminmax(header, cube_i16)
    log_fn(
        f"  SAVE_DEBUG: Legacy cube range [{np.min(cube_i16)}, {np.max(cube_i16)}]",
        "WARN",
    )
    legacy_hdu = fits_module.PrimaryHDU(data=cube_i16, header=header)
    legacy_hdu.header['BSCALE'] = 1
    legacy_hdu.header['BZERO'] = 32768
    return legacy_hdu


def _build_generic_cube_hdu(
    img_hwc: np.ndarray,
    base_header,
    fits_module,
    log_fn,
):
    """Fallback: scale and save an arbitrary multi-channel cube."""

    height, width, channels = img_hwc.shape
    u16_data = _rescale_to_u16(img_hwc)
    i16_data = _to_int16_with_bzero(u16_data)
    data_to_write = np.moveaxis(i16_data, -1, 0)
    header = _prepare_int16_header(base_header, width, height)
    header["NAXIS"] = 3
    header["NAXIS3"] = int(channels)
    _update_dataminmax(header, data_to_write)
    log_fn(
        f"  SAVE_DEBUG: Generic cube range [{np.min(data_to_write)}, {np.max(data_to_write)}]",
        "WARN",
    )
    generic_hdu = fits_module.PrimaryHDU(data=data_to_write, header=header)
    generic_hdu.header['BSCALE'] = 1
    generic_hdu.header['BZERO'] = 32768
    return generic_hdu




def save_fits_image(image_data: np.ndarray,
                    output_path: str,
                    header = None,  # Type hint peut être plus flexible: fits_module_for_utils.Header() | dict
                    overwrite: bool = True,
                    save_as_float: bool = False,
                    legacy_rgb_cube: bool = False,
                    progress_callback: callable = None,
                    axis_order: str = "HWC",
                    alpha_mask: Optional[np.ndarray] = None):
    """
    Sauvegarde des données image NumPy dans un fichier FITS.
    Utilise ASTROPY_AVAILABLE_IN_UTILS défini localement.
    Version avec logs de débogage améliorés et gestion gc.
    L'argument ``axis_order`` indique comment interpréter les tableaux couleur
    en entrée.
    - ``"HWC"`` (défaut) : ``Height x Width x Channels``. Les données sont
      transposées en ``CxHxW`` pour l'écriture FITS.
    - ``"CHW"`` : les données sont déjà dans l'ordre ``Channels x Height x Width``.
    - ``alpha_mask`` : optionally include a uint8 mask (0..255) as ``ALPHA`` extension.
    """

    def _log_util_save(message, level="DEBUG_DETAIL", pcb=progress_callback):
        log_prefix = "  [ZU SaveFITS]"
        if "SAVE_DEBUG" in message: log_prefix = "  [ZU SaveFITS DEBUG]"
        full_message = f"{log_prefix} {message}"
        if pcb and callable(pcb): pcb(full_message, None, level)
        else: print(full_message)

    base_output_filename = Path(output_path).name
    _log_util_save(f"Début sauvegarde FITS vers '{base_output_filename}'. SaveAsFloat={save_as_float}", "INFO")

    # Utiliser le fits_module_for_utils défini globalement dans ce module
    current_fits_module = fits_module_for_utils 
    current_astropy_available_flag = ASTROPY_AVAILABLE_IN_UTILS

    if not current_astropy_available_flag and current_fits_module.__class__.__name__ == "MockFitsModule":
        _log_util_save(f"ERREUR CRITIQUE: Astropy non disponible. Sauvegarde réelle de '{base_output_filename}' impossible (mock actif).", "ERROR")
        return

    if image_data is None: _log_util_save(f"ERREUR: Image data est None pour '{base_output_filename}'. Sauvegarde annulée.", "ERROR"); return
    if not isinstance(image_data, np.ndarray): _log_util_save(f"ERREUR: Input doit être NumPy array, reçu {type(image_data)}.", "ERROR"); return

    _log_util_save(f"SAVE_DEBUG: Données image_data reçues - Shape: {image_data.shape}, Dtype: {image_data.dtype}, Range: [{np.nanmin(image_data):.3g} - {np.nanmax(image_data):.3g}], IsFinite: {np.all(np.isfinite(image_data))}", "WARN")

    final_header_to_write = current_fits_module.Header()
    if header is not None:
        try:
            if hasattr(header, 'to_header') and callable(header.to_header): final_header_to_write.update(header.to_header(relax=True))
            elif isinstance(header, (current_fits_module.Header if current_astropy_available_flag else dict)): final_header_to_write.update(header.copy()) # type: ignore
            elif isinstance(header, dict): final_header_to_write.update(header)
            else: _log_util_save(f"AVERT: Type de header non supporté ({type(header)}). Utilisation header vide.", "WARN")
        except Exception as e_hdr_copy:
             _log_util_save(f"AVERT: Erreur copie/update header: {e_hdr_copy}. Header partiel/vide.", "WARN")
             final_header_to_write = current_fits_module.Header()

    keywords_to_remove_base = ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'BSCALE', 'BZERO']
    for key_k in keywords_to_remove_base:
        if key_k in final_header_to_write:
            try: del final_header_to_write[key_k]
            except KeyError: pass

    data_to_write_temp = None
    hdus_to_write = []
    primary_hdu_object = None
    if save_as_float:
        # Avoid an extra full-size copy if already float32 (important for huge mosaics)
        if isinstance(image_data, np.ndarray) and image_data.dtype == np.float32:
            data_to_write_temp = image_data
        else:
            data_to_write_temp = image_data.astype(np.float32, copy=False)

        # Ensure a zero floor for better viewer auto-stretch (ASI FITS View, etc.).
        # Some viewers expect the black level to be at 0. If our mosaic carries a
        # positive offset (e.g. min ~ 300-500 ADU), auto-stretch may miss the true
        # background. Shift the baseline so the global finite minimum maps to 0.
        baseline_shift_applied = 0.0
        try:
            finite_min = float(np.nanmin(data_to_write_temp))
            if np.isfinite(finite_min) and finite_min > 0.0:
                _log_util_save(
                    f"  SAVE_DEBUG: Positive baseline detected (min={finite_min:.3f}). Shifting to zero.",
                    "INFO_DETAIL",
                )
                # Avoid mutating the caller's array when we didn't already make a copy
                if data_to_write_temp is image_data:
                    data_to_write_temp = data_to_write_temp.copy()
                data_to_write_temp -= np.float32(finite_min)
                # Guard against any numerical underflow
                data_to_write_temp = np.maximum(data_to_write_temp, 0.0)
                baseline_shift_applied = -float(finite_min)
            elif np.isfinite(finite_min) and finite_min < 0.0:
                shift_value = -float(finite_min)
                _log_util_save(
                    f"  SAVE_DEBUG: Negative baseline detected (min={finite_min:.3f}). Raising by {shift_value:.3f}.",
                    "INFO_DETAIL",
                )
                if data_to_write_temp is image_data:
                    data_to_write_temp = data_to_write_temp.copy()
                data_to_write_temp += np.float32(shift_value)
                data_to_write_temp = np.maximum(data_to_write_temp, 0.0)
                baseline_shift_applied = shift_value
        except Exception as _e_minshift:
            # Non-fatal: keep original values if anything goes wrong
            pass

        # For float images keep the header simple; many viewers rely on missing BSCALE/BZERO
        final_header_to_write['BITPIX'] = -32
        if 'BSCALE' in final_header_to_write: del final_header_to_write['BSCALE']
        if 'BZERO' in final_header_to_write: del final_header_to_write['BZERO']
        # Several viewers misinterpret DATAMIN/DATAMAX on float images for their
        # initial auto-stretch. Historically our best compatibility came from NOT
        # writing these two cards for BITPIX=-32. Ensure they are absent.
        if 'DATAMIN' in final_header_to_write: del final_header_to_write['DATAMIN']
        if 'DATAMAX' in final_header_to_write: del final_header_to_write['DATAMAX']
        _log_util_save(f"  SAVE_DEBUG: (Float) data_to_write_temp: Range [{np.nanmin(data_to_write_temp):.3g}, {np.nanmax(data_to_write_temp):.3g}], IsFinite: {np.all(np.isfinite(data_to_write_temp))}", "WARN")

        axis_order_upper = str(axis_order).upper()
        if data_to_write_temp.ndim == 3:
            if axis_order_upper == 'HWC':
                data_for_primary = np.moveaxis(data_to_write_temp, -1, 0)
            elif axis_order_upper == 'CHW':
                data_for_primary = data_to_write_temp
            else:
                _log_util_save(f"Axis order '{axis_order}' non reconnu, utilisation 'HWC'", "WARN")
                data_for_primary = np.moveaxis(data_to_write_temp, -1, 0)
        else:
            data_for_primary = data_to_write_temp

        _log_util_save(
            f"SAVE_DEBUG: Données PRÊTES (float) - Shape: {data_for_primary.shape}, Dtype: {data_for_primary.dtype}, Range: [{np.nanmin(data_for_primary):.3g}, {np.nanmax(data_for_primary):.3g}], IsFinite: {np.all(np.isfinite(data_for_primary))}",
            "WARN",
        )

        header_float = final_header_to_write.copy()
        header_float['BITPIX'] = -32
        if 'BSCALE' in header_float:
            del header_float['BSCALE']
        if 'BZERO' in header_float:
            del header_float['BZERO']
        if 'DATAMIN' in header_float:
            del header_float['DATAMIN']
        if 'DATAMAX' in header_float:
            del header_float['DATAMAX']

        if baseline_shift_applied != 0.0:
            shift_msg = f"Baseline shift applied for FITS float export: {baseline_shift_applied:+.6f} ADU"
            try:
                header_float.add_history(shift_msg)
            except Exception:
                pass
            try:
                final_header_to_write.add_history(shift_msg)
            except Exception:
                pass

        if data_to_write_temp.ndim == 3:
            if axis_order_upper == 'HWC':
                h, w, c = image_data.shape
            elif axis_order_upper == 'CHW':
                c, h, w = image_data.shape
            else:
                h, w, c = image_data.shape
            header_float['NAXIS'] = 3
            header_float['NAXIS1'] = int(w)
            header_float['NAXIS2'] = int(h)
            header_float['NAXIS3'] = int(c)
            if 'CTYPE3' not in header_float:
                header_float['CTYPE3'] = ('RGB', 'Color Format')
            if 'EXTNAME' not in header_float:
                header_float['EXTNAME'] = 'RGB'
        else:
            header_float['NAXIS'] = 2
            header_float['NAXIS1'] = int(data_for_primary.shape[1])
            header_float['NAXIS2'] = int(data_for_primary.shape[0])
            if 'NAXIS3' in header_float:
                del header_float['NAXIS3']
            if 'CTYPE3' in header_float:
                del header_float['CTYPE3']
            if 'EXTNAME' in header_float:
                del header_float['EXTNAME']

        primary_hdu_object = current_fits_module.PrimaryHDU(data=data_for_primary, header=header_float)
        hdus_to_write.append(primary_hdu_object)
    else:
        axis_order_upper = str(axis_order).upper()
        if image_data.ndim == 2:
            sanitized = _ensure_float32_no_nan(image_data)
            u16_plane = _rescale_to_u16(sanitized)
            i16_plane = _to_int16_with_bzero(u16_plane)
            header_primary = _prepare_int16_header(final_header_to_write, sanitized.shape[1], sanitized.shape[0])
            _update_dataminmax(header_primary, i16_plane)
            primary_hdu_object = current_fits_module.PrimaryHDU(data=i16_plane, header=header_primary)
            primary_hdu_object.header['BSCALE'] = 1
            primary_hdu_object.header['BZERO'] = 32768
            hdus_to_write.append(primary_hdu_object)
            _log_util_save(
                f"  SAVE_DEBUG: Monochrome int16 range [{np.min(i16_plane)}, {np.max(i16_plane)}]",
                "WARN",
            )
        elif image_data.ndim == 3:
            if axis_order_upper == 'CHW':
                img_hwc = np.moveaxis(image_data, 0, -1)
            elif axis_order_upper == 'HWC':
                img_hwc = image_data
            else:
                _log_util_save(f"Axis order '{axis_order}' non reconnu, utilisation 'HWC'", "WARN")
                img_hwc = image_data

            img_hwc = _ensure_float32_no_nan(img_hwc)
            channels = img_hwc.shape[-1]
            if channels == 3:
                if legacy_rgb_cube:
                    primary_hdu_object = _build_legacy_rgb_cube_hdu(
                        img_hwc, final_header_to_write, current_fits_module, _log_util_save
                    )
                    hdus_to_write.append(primary_hdu_object)
                else:
                    primary_hdu_object, plane_hdus = _build_rgb_luminance_hdus(
                        img_hwc, final_header_to_write, current_fits_module, _log_util_save
                    )
                    hdus_to_write.append(primary_hdu_object)
                    hdus_to_write.extend(plane_hdus)
            else:
                _log_util_save(
                    f"SAVE_DEBUG: Canaux={channels} (non-RGB). Utilisation du mode cube générique.",
                    "WARN",
                )
                primary_hdu_object = _build_generic_cube_hdu(
                    img_hwc, final_header_to_write, current_fits_module, _log_util_save
                )
                hdus_to_write.append(primary_hdu_object)
        else:
            _log_util_save(
                f"ERREUR: Dimensions d'image non supportées pour '{base_output_filename}' : {image_data.shape}",
                "ERROR",
            )
            return

        if hdus_to_write:
            primary_data = hdus_to_write[0].data
            _log_util_save(
                f"SAVE_DEBUG: Données PRÊTES (int16) - Shape: {primary_data.shape}, Dtype: {primary_data.dtype}, Range: [{np.min(primary_data)} - {np.max(primary_data)}]",
                "WARN",
            )

    if not hdus_to_write:
        _log_util_save(
            f"ERREUR: Aucun HDU généré pour '{base_output_filename}'. Sauvegarde annulée.",
            "ERROR",
        )
        return

    if alpha_mask is not None:
        try:
            alpha_arr = np.asarray(alpha_mask, dtype=np.uint8, copy=False)
            if alpha_arr.ndim > 2:
                alpha_arr = alpha_arr[..., 0]
            alpha_hdu = current_fits_module.ImageHDU(alpha_arr, name="ALPHA")
            alpha_hdu.header["ALPHADSC"] = ("1=opaque(in), 0=transparent(out)", "")
            hdus_to_write.append(alpha_hdu)
        except Exception:
            _log_util_save("SAVE_DEBUG: Unable to append ALPHA extension (shape mismatch)", "WARN")
    hdul = None
    try:
        primary_for_log = primary_hdu_object or hdus_to_write[0]
        primary_data = getattr(primary_for_log, 'data', None)
        if primary_data is not None:
            _log_util_save(
                f"SAVE_DEBUG: AVANT écriture - Min: {np.nanmin(primary_data)}, Max: {np.nanmax(primary_data)}, Mean: {np.nanmean(primary_data)}, Std: {np.nanstd(primary_data)}, Dtype: {primary_data.dtype}, Finite: {np.all(np.isfinite(primary_data))}",
                "ERROR",
            )

        hdul = current_fits_module.HDUList(hdus_to_write)
        _log_util_save(f"Écriture vers '{base_output_filename}' (overwrite={overwrite})...", "DEBUG_DETAIL")

        hdul.writeto(output_path, overwrite=overwrite, checksum=True, output_verify='exception')
        _log_util_save(f"Sauvegarde FITS vers '{base_output_filename}' RÉUSSIE.", "INFO")

    except Exception as e_write:
        _log_util_save(f"ERREUR CRITIQUE lors sauvegarde FITS '{base_output_filename}': {type(e_write).__name__} - {e_write}", "ERROR")
        if progress_callback:
            _log_util_save(f"  [ZU SaveFITS TRACEBACK] {traceback.format_exc(limit=3)}", "ERROR")
    finally:
        if hdul is not None and hasattr(hdul, 'close'):
            try:
                hdul.close()
            except Exception:
                pass

        # Nettoyage explicite pour aider le GC
        if 'data_to_write_temp' in locals() and data_to_write_temp is not None:
            del data_to_write_temp
        if primary_hdu_object is not None and hasattr(primary_hdu_object, 'data') and primary_hdu_object.data is not None:
            del primary_hdu_object.data
        if primary_hdu_object is not None:
            del primary_hdu_object
        if 'hdus_to_write' in locals():
            del hdus_to_write
        if 'hdul' in locals() and hdul is not None:
            del hdul
        gc.collect() # gc doit être importé en haut du fichier zemosaic_utils.py


def gpu_assemble_final_mosaic_reproject_coadd(*args, **kwargs):
    """Deprecated placeholder kept for compatibility.

    The worker now routes GPU usage via ``assemble_final_mosaic_reproject_coadd(..., use_gpu=True)``.
    This function remains for backward compatibility and will raise to signal callers to switch.
    """
    raise NotImplementedError(
        "Call assemble_final_mosaic_reproject_coadd(use_gpu=True) from the worker instead."
    )


def gpu_assemble_final_mosaic_incremental(*args, **kwargs):
    """Deprecated placeholder for incremental GPU path.

    Use the CPU incremental assembly; reprojection dominates and remains CPU-bound for now.
    """
    raise NotImplementedError(
        "Incremental GPU path is not implemented; use CPU incremental assembly."
    )


def _sanitize_winsor_limits(limits) -> tuple[float, float]:
    """Normalize winsorized limits to sane float tuple."""

    if not isinstance(limits, (list, tuple)) or len(limits) < 2:
        return 0.05, 0.05
    try:
        low = float(limits[0])
        high = float(limits[1])
    except Exception:
        return 0.05, 0.05
    low = max(0.0, min(0.49, low))
    high = max(0.0, min(0.49, high))
    return low, high


def _xp_errstate(xp_module, **kwargs):
    errstate = getattr(xp_module, "errstate", None)
    if callable(errstate):
        return errstate(**kwargs)
    return nullcontext()


def _winsorized_weighted_average_chunk(
    stack,
    weights,
    winsor_limits,
    xp_module,
):
    """Winsorize stack along axis=0 using *xp_module* (NumPy or CuPy)."""

    xp = xp_module
    low_frac, high_frac = _sanitize_winsor_limits(winsor_limits)
    low_pct = max(0.0, min(100.0, low_frac * 100.0))
    high_pct = max(0.0, min(100.0, 100.0 - high_frac * 100.0))
    lower = xp.nanpercentile(stack, low_pct, axis=0)
    upper = xp.nanpercentile(stack, high_pct, axis=0)
    clipped = xp.clip(stack, lower, upper)
    weighted = clipped * weights
    chunk_weight = xp.nansum(weights, axis=0)
    with _xp_errstate(xp, invalid="ignore", divide="ignore"):
        chunk_result = xp.nansum(weighted, axis=0) / xp.maximum(chunk_weight, 1e-6)
    chunk_result = xp.nan_to_num(chunk_result, nan=0.0, posinf=0.0, neginf=0.0)
    chunk_weight = xp.nan_to_num(chunk_weight, nan=0.0, posinf=0.0, neginf=0.0)
    return chunk_result, chunk_weight


def _kappa_sigma_clip_chunk(
    stack,
    weights,
    mean_ref,
    std_ref,
    count_ref,
    *,
    kappa: float,
    min_sigma: float,
    xp_module,
):
    """Apply kappa-sigma clipping for a chunk using *xp_module*."""

    xp = xp_module
    thresh = kappa * xp.maximum(std_ref, min_sigma)
    diff = xp.abs(stack - mean_ref)
    accept = diff <= thresh
    accept = xp.where(count_ref <= 1.5, True, accept)
    kept_weights = xp.where(accept, weights, 0.0)
    clip_weight = xp.nansum(kept_weights, axis=0)
    clip_sum = xp.nansum(stack * kept_weights, axis=0)
    with _xp_errstate(xp, invalid="ignore", divide="ignore"):
        clipped = xp.where(
            clip_weight > 0,
            clip_sum / xp.maximum(clip_weight, 1e-6),
            mean_ref,
        )
    clipped = xp.nan_to_num(clipped, nan=0.0, posinf=0.0, neginf=0.0)
    clip_weight = xp.nan_to_num(clip_weight, nan=0.0, posinf=0.0, neginf=0.0)
    return clipped, clip_weight


def gpu_reproject_and_coadd_impl(data_list, wcs_list, shape_out, **kwargs):
    """CuPy-accelerated reprojection and coaddition for a single channel."""

    progress_callback = kwargs.pop("progress_callback", None)
    if not gpu_is_available():
        _log_gpu_event(
            "gpu_fallback_unavailable",
            "WARN",
            progress_callback,
            helper="gpu_reproject",
            reason="gpu_unavailable",
        )
        raise RuntimeError("gpu_unavailable")

    import cupy as cp  # type: ignore
    from cupyx.scipy.ndimage import map_coordinates  # type: ignore

    ensure_cupy_pool_initialized()
    try:
        from astropy.wcs import WCS as _WCS
    except Exception:
        _WCS = None

    output_projection = kwargs.get("output_projection")
    if output_projection is None:
        raise ValueError("gpu_reproject_and_coadd requires 'output_projection' (WCS or header)")
    if _WCS is not None and not hasattr(output_projection, "pixel_to_world"):
        try:
            output_wcs = _WCS(output_projection)
        except Exception as exc:
            raise ValueError(f"Invalid output_projection for WCS: {exc}") from exc
    else:
        output_wcs = output_projection

    H, W = int(shape_out[0]), int(shape_out[1])
    n_inputs = len(data_list)
    if len(wcs_list) != n_inputs:
        raise ValueError("data_list and wcs_list must have the same length")

    combine_function = str(kwargs.get("combine_function") or "mean").strip().lower()
    stack_reject_algo = str(kwargs.get("stack_reject_algo") or combine_function).strip().lower()
    if combine_function == "median":
        combine_mode = "median"
    elif combine_function == "kappa_sigma":
        combine_mode = "kappa_sigma"
    elif combine_function == "winsorized" or stack_reject_algo in {"winsorized", "winsor", "winsorized_sigma_clip"}:
        combine_mode = "winsorized"
    else:
        combine_mode = "mean"

    winsor_limits = _sanitize_winsor_limits(kwargs.get("winsor_limits", (0.05, 0.05)) or (0.05, 0.05))
    kappa_sigma_k = float(kwargs.get("coadd_k", kwargs.get("kappa_sigma_k", 2.0)) or 2.0)
    match_background = bool(kwargs.get("match_background", kwargs.get("match_bg", False)))

    float32_bytes = np.dtype(np.float32).itemsize
    tile_affine = kwargs.get("tile_affine_corrections", None)
    if isinstance(tile_affine, (list, tuple)) and len(tile_affine) == n_inputs:
        normalized: list[tuple[float, float]] = []
        for pair in tile_affine:
            gain_val = 1.0
            offset_val = 0.0
            if isinstance(pair, (list, tuple)) and pair:
                try:
                    gain_val = float(pair[0])
                except Exception:
                    gain_val = 1.0
                if len(pair) > 1:
                    try:
                        offset_val = float(pair[1])
                    except Exception:
                        offset_val = 0.0
            if not np.isfinite(gain_val):
                gain_val = 1.0
            if not np.isfinite(offset_val):
                offset_val = 0.0
            normalized.append((gain_val, offset_val))
        tile_affine = normalized
    else:
        tile_affine = None

    def _prepare_tile_arrays(idx: int) -> tuple[cp.ndarray, cp.ndarray]:
        arr = np.asarray(data_list[idx], dtype=np.float32)
        img = cp.asarray(arr)
        if tile_affine is not None:
            gain_val, offset_val = tile_affine[idx]
            if gain_val != 1.0:
                img = img * cp.float32(gain_val)
            if offset_val != 0.0:
                img = img + cp.float32(offset_val)
        if match_background:
            try:
                offset_val = float(cp.nanmedian(img).get())
            except Exception:
                try:
                    offset_val = float(np.nanmedian(cp.asnumpy(img)))
                except Exception:
                    offset_val = 0.0
            if offset_val != 0.0 and np.isfinite(offset_val):
                img = img - cp.float32(offset_val)
        mask = cp.isfinite(img).astype(cp.float32)
        img = cp.nan_to_num(img, copy=False, nan=0.0)
        return img, mask

    def _build_world_chunk_grid(rows_per_chunk: int) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
        chunk_grid: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        x_coords = np.arange(W, dtype=np.float32)
        rows_per_chunk = max(1, min(rows_per_chunk, H))
        for y0 in range(0, H, rows_per_chunk):
            y1 = min(H, y0 + rows_per_chunk)
            y_coords = np.arange(y0, y1, dtype=np.float32)
            xx_cpu, yy_cpu = np.meshgrid(x_coords, y_coords)
            try:
                ra_chunk, dec_chunk = output_wcs.wcs_pix2world(xx_cpu, yy_cpu, 0)
            except Exception:
                raise RuntimeError("Failed to convert output grid to world coordinates")
            chunk_grid.append((y0, y1, np.asarray(ra_chunk), np.asarray(dec_chunk)))
        return chunk_grid

    def _sample_tile_chunk(
        img_gpu: cp.ndarray,
        mask_gpu: cp.ndarray,
        tile_wcs,
        ra_chunk: np.ndarray,
        dec_chunk: np.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        try:
            x_in, y_in = tile_wcs.wcs_world2pix(ra_chunk, dec_chunk, 0)
        except Exception:
            shape = ra_chunk.shape
            zeros = cp.zeros(shape, dtype=cp.float32)
            return zeros, zeros
        x_in_gpu = cp.asarray(x_in, dtype=cp.float32)
        y_in_gpu = cp.asarray(y_in, dtype=cp.float32)
        h_in, w_in = img_gpu.shape
        valid = (
            (x_in_gpu >= -0.5)
            & (x_in_gpu <= (w_in - 0.5))
            & (y_in_gpu >= -0.5)
            & (y_in_gpu <= (h_in - 0.5))
        )
        coords = cp.stack([y_in_gpu, x_in_gpu], axis=0)
        sampled = map_coordinates(img_gpu, coords, order=1, mode="constant", cval=0.0)
        sampled_mask = map_coordinates(mask_gpu, coords, order=1, mode="constant", cval=0.0)
        sampled = sampled * sampled_mask * valid.astype(cp.float32)
        sampled_mask = sampled_mask * valid.astype(cp.float32)
        return sampled, sampled_mask

    def _finalize_match_background(mosaic_gpu: cp.ndarray) -> cp.ndarray:
        if not match_background:
            return mosaic_gpu
        try:
            p01 = float(cp.percentile(mosaic_gpu, 0.1).get())
            if p01 < 0:
                mosaic_gpu = mosaic_gpu - cp.float32(p01)
        except Exception:
            pass
        return mosaic_gpu

    def _gpu_mean_path() -> tuple[np.ndarray, np.ndarray]:
        accumulator_bytes = 2 * H * W * float32_bytes
        memory_guard = int(accumulator_bytes * 2.5)
        if not gpu_memory_sufficient(memory_guard, safety_fraction=0.8):
            allowed = _get_gpu_allowed_bytes(0.8)
            _log_gpu_event(
                "gpu_fallback_insufficient_memory",
                "WARN",
                progress_callback,
                helper="gpu_reproject",
                estimated_mb=_format_mebibytes(memory_guard),
                allowed_mb=_format_mebibytes(allowed),
            )
            raise RuntimeError("Insufficient GPU memory for reprojection accumulators")
        mosaic_sum_gpu = cp.zeros((H, W), dtype=cp.float32)
        weight_sum_gpu = cp.zeros((H, W), dtype=cp.float32)
        rows_per_chunk = int(kwargs.get("rows_per_chunk", 64))
        rows_per_chunk = max(32, min(rows_per_chunk, H))
        max_chunk_bytes = int(kwargs.get("max_chunk_bytes") or (128 * 1024 * 1024))
        try:
            free_bytes, _ = cp.cuda.runtime.memGetInfo()
            safe_target = max(16 * 1024 * 1024, int(free_bytes * 0.25))
            max_chunk_bytes = min(max_chunk_bytes, safe_target)
        except Exception:
            pass
        bytes_per_row_estimate = max(1, W * float32_bytes * 8)
        adaptive_rows = max(1, max_chunk_bytes // bytes_per_row_estimate)
        rows_per_chunk = max(32, min(rows_per_chunk, adaptive_rows, H))
        chunk_grid = _build_world_chunk_grid(rows_per_chunk)
        for idx_tile, tile_wcs in enumerate(wcs_list):
            img_gpu, mask_gpu = _prepare_tile_arrays(idx_tile)
            for y0, y1, ra_chunk, dec_chunk in chunk_grid:
                sampled, sampled_mask = _sample_tile_chunk(img_gpu, mask_gpu, tile_wcs, ra_chunk, dec_chunk)
                mosaic_sum_gpu[y0:y1, :] += sampled
                weight_sum_gpu[y0:y1, :] += sampled_mask
            del img_gpu, mask_gpu
        eps = cp.float32(1e-6)
        mosaic_gpu = cp.where(weight_sum_gpu > eps, mosaic_sum_gpu / cp.maximum(weight_sum_gpu, eps), 0.0)
        coverage_gpu = cp.clip(weight_sum_gpu / float(max(1, n_inputs)), 0.0, 1.0)
        mosaic_gpu = _finalize_match_background(mosaic_gpu)
        return cp.asnumpy(mosaic_gpu).astype(np.float32), cp.asnumpy(coverage_gpu).astype(np.float32)

    def _gpu_chunkwise_path(mode: str) -> tuple[np.ndarray, np.ndarray]:
        rows_hint = int(kwargs.get("rows_per_chunk", 64))
        chunk_bytes_per_row = max(1, n_inputs * W * float32_bytes * 2)
        max_chunk_bytes = int(kwargs.get("max_chunk_bytes") or (256 * 1024 * 1024))
        try:
            free_bytes, _ = cp.cuda.runtime.memGetInfo()
            safe_target = max(64 * 1024 * 1024, int(free_bytes * 0.4))
            max_chunk_bytes = min(max_chunk_bytes, safe_target)
        except Exception:
            pass
        rows_per_chunk = max(1, min(H, max(1, max_chunk_bytes // chunk_bytes_per_row)))
        rows_per_chunk = max(1, min(rows_hint, rows_per_chunk))
        if rows_per_chunk < rows_hint:
            _log_gpu_event(
                "gpu_chunking_rows",
                "INFO_DETAIL",
                progress_callback,
                helper="gpu_reproject",
                rows=int(rows_per_chunk),
                height=int(H),
            )

        cache_tiles = True
        estimated_cache_bytes = _estimate_tile_cache_bytes(data_list)
        if estimated_cache_bytes > 0 and not gpu_memory_sufficient(
            int(estimated_cache_bytes * 1.1), safety_fraction=0.75
        ):
            cache_tiles = False
            _log_gpu_event(
                "gpu_helper_stream_tiles",
                "INFO_DETAIL",
                progress_callback,
                helper="gpu_reproject",
            )

        tile_cache: list[tuple[cp.ndarray, cp.ndarray]] | None = None
        if cache_tiles:
            tile_cache = []
            try:
                for idx in range(n_inputs):
                    tile_cache.append(_prepare_tile_arrays(idx))
            except Exception as exc:
                tile_cache.clear()
                raise RuntimeError("Failed to upload tiles to GPU for chunked combine") from exc
        mosaic_gpu = cp.zeros((H, W), dtype=cp.float32)
        coverage_gpu = cp.zeros((H, W), dtype=cp.float32)
        chunk_grid = _build_world_chunk_grid(rows_per_chunk)
        for y0, y1, ra_chunk, dec_chunk in chunk_grid:
            chunk_h = y1 - y0
            chunk_cube = cp.full((n_inputs, chunk_h, W), cp.nan, dtype=cp.float32)
            chunk_weight = cp.zeros((n_inputs, chunk_h, W), dtype=cp.float32)
            for idx_tile, tile_wcs in enumerate(wcs_list):
                if tile_cache is not None:
                    img_gpu, mask_gpu = tile_cache[idx_tile]
                else:
                    img_gpu, mask_gpu = _prepare_tile_arrays(idx_tile)
                sampled, sampled_mask = _sample_tile_chunk(img_gpu, mask_gpu, tile_wcs, ra_chunk, dec_chunk)
                chunk_cube[idx_tile, :, :] = cp.where(sampled_mask > 0, sampled, cp.nan)
                chunk_weight[idx_tile, :, :] = sampled_mask
                if tile_cache is None:
                    del img_gpu, mask_gpu
            if mode == "median":
                chunk_result = cp.nanmedian(chunk_cube, axis=0)
                chunk_cov = cp.nansum(chunk_weight, axis=0)
            else:
                chunk_result, chunk_cov = _winsorized_weighted_average_chunk(
                    chunk_cube,
                    chunk_weight,
                    winsor_limits,
                    cp,
                )
            chunk_result = cp.nan_to_num(chunk_result, nan=0.0, posinf=0.0, neginf=0.0)
            chunk_cov = cp.nan_to_num(chunk_cov, nan=0.0, posinf=0.0, neginf=0.0)
            mosaic_gpu[y0:y1, :] = chunk_result
            coverage_gpu[y0:y1, :] = chunk_cov
            del chunk_cube, chunk_weight
        mosaic_gpu = _finalize_match_background(mosaic_gpu)
        return cp.asnumpy(mosaic_gpu).astype(np.float32), cp.asnumpy(coverage_gpu).astype(np.float32)

    def _gpu_kappa_sigma_path() -> tuple[np.ndarray, np.ndarray]:
        rows_hint = int(kwargs.get("rows_per_chunk", 64))
        cache_tiles = True
        estimated_cache_bytes = _estimate_tile_cache_bytes(data_list)
        if estimated_cache_bytes > 0 and not gpu_memory_sufficient(
            int(estimated_cache_bytes * 1.1), safety_fraction=0.75
        ):
            cache_tiles = False
            _log_gpu_event(
                "gpu_helper_stream_tiles",
                "INFO_DETAIL",
                progress_callback,
                helper="gpu_reproject",
            )
        tile_cache: list[tuple[cp.ndarray, cp.ndarray]] | None = None
        if cache_tiles:
            tile_cache = []
            try:
                for idx in range(n_inputs):
                    tile_cache.append(_prepare_tile_arrays(idx))
            except Exception as exc:
                tile_cache.clear()
                raise RuntimeError("Failed to upload tiles for kappa-sigma combine") from exc
        weight_grid = cp.zeros((H, W), dtype=cp.float32)
        count_grid = cp.zeros((H, W), dtype=cp.float32)
        sum_grid = cp.zeros((H, W), dtype=cp.float64)
        sumsq_grid = cp.zeros((H, W), dtype=cp.float64)
        rows_stats = max(32, min(rows_hint, H))
        chunk_stats = _build_world_chunk_grid(rows_stats)
        def _tile_arrays(idx_tile: int) -> tuple[cp.ndarray, cp.ndarray]:
            if tile_cache is not None:
                return tile_cache[idx_tile]
            return _prepare_tile_arrays(idx_tile)

        for idx_tile, tile_wcs in enumerate(wcs_list):
            img_gpu, mask_gpu = _tile_arrays(idx_tile)
            for y0, y1, ra_chunk, dec_chunk in chunk_stats:
                sampled, sampled_mask = _sample_tile_chunk(img_gpu, mask_gpu, tile_wcs, ra_chunk, dec_chunk)
                sum_grid[y0:y1, :] += sampled.astype(cp.float64)
                sumsq_grid[y0:y1, :] += (sampled ** 2).astype(cp.float64)
                weight_grid[y0:y1, :] += sampled_mask
                count_grid[y0:y1, :] += (sampled_mask > 0).astype(cp.float32)
            if tile_cache is None:
                del img_gpu, mask_gpu
        weight_safe = cp.maximum(weight_grid, cp.float32(1e-6))
        mean_map = sum_grid / weight_safe
        second_moment = sumsq_grid / weight_safe
        variance = cp.maximum(second_moment - (mean_map ** 2), 0.0)
        std_map = cp.sqrt(variance, dtype=cp.float64)
        min_sigma = 0.0
        finite_mask = cp.isfinite(std_map)
        if bool(int(cp.any(finite_mask).get())):
            try:
                min_sigma = float(cp.percentile(std_map[finite_mask], 5).get())
            except Exception:
                min_sigma = 0.0
        min_sigma = max(min_sigma, 1e-4)
        chunk_bytes_per_row = max(1, n_inputs * W * float32_bytes * 2)
        max_chunk_bytes = int(kwargs.get("max_chunk_bytes") or (256 * 1024 * 1024))
        try:
            free_bytes, _ = cp.cuda.runtime.memGetInfo()
            safe_target = max(64 * 1024 * 1024, int(free_bytes * 0.4))
            max_chunk_bytes = min(max_chunk_bytes, safe_target)
        except Exception:
            pass
        rows_clip = max(1, min(H, max(1, max_chunk_bytes // chunk_bytes_per_row)))
        rows_clip = max(1, min(rows_hint, rows_clip))
        if rows_clip < rows_hint:
            _log_gpu_event(
                "gpu_chunking_rows",
                "INFO_DETAIL",
                progress_callback,
                helper="gpu_reproject",
                rows=int(rows_clip),
                height=int(H),
            )
        chunk_clip = _build_world_chunk_grid(rows_clip)
        clipped_buffer = cp.zeros((H, W), dtype=cp.float32)
        clip_weight = cp.zeros((H, W), dtype=cp.float32)
        for y0, y1, ra_chunk, dec_chunk in chunk_clip:
            chunk_h = y1 - y0
            chunk_cube = cp.full((n_inputs, chunk_h, W), cp.nan, dtype=cp.float32)
            chunk_weight = cp.zeros((n_inputs, chunk_h, W), dtype=cp.float32)
            for idx_tile, tile_wcs in enumerate(wcs_list):
                img_gpu, mask_gpu = _tile_arrays(idx_tile)
                sampled, sampled_mask = _sample_tile_chunk(img_gpu, mask_gpu, tile_wcs, ra_chunk, dec_chunk)
                chunk_cube[idx_tile, :, :] = cp.where(sampled_mask > 0, sampled, cp.nan)
                chunk_weight[idx_tile, :, :] = sampled_mask
                if tile_cache is None:
                    del img_gpu, mask_gpu
            mean_slice = mean_map[y0:y1, :]
            std_slice = std_map[y0:y1, :]
            count_slice = count_grid[y0:y1, :]
            chunk_result, chunk_cov = _kappa_sigma_clip_chunk(
                chunk_cube,
                chunk_weight,
                mean_slice,
                std_slice,
                count_slice,
                kappa=kappa_sigma_k,
                min_sigma=min_sigma,
                xp_module=cp,
            )
            clipped_buffer[y0:y1, :] = chunk_result.astype(cp.float32)
            clip_weight[y0:y1, :] = chunk_cov.astype(cp.float32)
            del chunk_cube, chunk_weight
        coverage_gpu = cp.where(clip_weight > 0, clip_weight, weight_grid)
        clipped_map = cp.where(clip_weight > 0, clipped_buffer, mean_map.astype(cp.float32))
        clipped_map = _finalize_match_background(clipped_map)
        return cp.asnumpy(clipped_map).astype(np.float32), cp.asnumpy(coverage_gpu).astype(np.float32)

    try:
        if combine_mode == "mean":
            return _gpu_mean_path()
        if combine_mode == "median":
            return _gpu_chunkwise_path("median")
        if combine_mode == "winsorized":
            return _gpu_chunkwise_path("winsorized")
        if combine_mode == "kappa_sigma":
            return _gpu_kappa_sigma_path()
        raise NotImplementedError(f"Unsupported combine_function '{combine_function}' for GPU coadd")
    finally:
        free_cupy_memory_pools()


def _reproject_and_coadd_wrapper_impl(
    data_list,
    wcs_list,
    shape_out,
    *,
    use_gpu: bool = False,
    cpu_func=None,
    allow_cpu_fallback: bool = True,
    progress_callback=None,
    **kwargs,
):
    """Dispatch to GPU or CPU reproject+coadd.

    - GPU path: uses ``gpu_reproject_and_coadd`` (CuPy). Falls back to CPU on any error.
    - CPU path: calls astropy-reproject's ``reproject_and_coadd``.
    """
    gpu_kwargs = dict(kwargs)
    if progress_callback is not None:
        gpu_kwargs["progress_callback"] = progress_callback
    if use_gpu:
        if not gpu_is_available():
            _log_gpu_event(
                "gpu_fallback_unavailable",
                "WARN",
                progress_callback,
                helper="gpu_reproject",
                reason="gpu_unavailable",
            )
            if not allow_cpu_fallback:
                raise RuntimeError("gpu_unavailable")
        else:
            try:
                return gpu_reproject_and_coadd_impl(data_list, wcs_list, shape_out, **gpu_kwargs)
            except Exception as e:  # pragma: no cover - GPU failures
                _log_gpu_event(
                    "gpu_fallback_runtime_error",
                    "WARN",
                    progress_callback,
                    helper="gpu_reproject",
                    error=str(e),
                )
                if not allow_cpu_fallback:
                    raise
    if cpu_func is None:
        cpu_func = cpu_reproject_and_coadd
    # Remove GPU-only extras before CPU call to avoid unexpected kwargs
    gpu_only = {
        "bg_preview_size",
        "intertile_sky_percentile",
        "intertile_robust_clip_sigma",
        "rows_per_chunk",
        "max_chunk_bytes",
        # New GPU-only hints
        "tile_affine_corrections",
    }
    cpu_kwargs = {k: v for k, v in kwargs.items() if k not in gpu_only}
    inputs = list(zip(data_list, wcs_list))
    output_proj = cpu_kwargs.pop("output_projection")
    return cpu_func(inputs, output_proj, shape_out, **cpu_kwargs)



def gpu_reproject_and_coadd(data_list, wcs_list, shape_out, **kwargs):
    """Alias that forwards to the main GPU implementation.

    Some import sites may reference this name; keep it as a thin wrapper.
    """
    return gpu_reproject_and_coadd_impl(data_list, wcs_list, shape_out, **kwargs)


def reproject_and_coadd_wrapper(
    data_list,
    wcs_list,
    shape_out,
    use_gpu=False,
    cpu_func=None,
    allow_cpu_fallback: bool = True,
    progress_callback=None,
    **kwargs,
):
    """Dispatch to CPU or GPU ``reproject_and_coadd`` depending on availability."""
    return _reproject_and_coadd_wrapper_impl(
        data_list,
        wcs_list,
        shape_out,
        use_gpu=use_gpu,
        cpu_func=cpu_func,
        allow_cpu_fallback=allow_cpu_fallback,
        progress_callback=progress_callback,
        **kwargs,
    )


# --- GPU Percentiles, Hot-Pixels, and Background Map -------------------------
def _percentiles_gpu(arr2d: np.ndarray, p_low: float, p_high: float) -> tuple[float, float]:
    """Compute two percentiles on GPU; fall back to CPU if CuPy unavailable."""
    if not gpu_is_available():
        lo, hi = np.percentile(arr2d, [p_low, p_high])
        return float(lo), float(hi)
    import cupy as cp  # type: ignore
    ensure_cupy_pool_initialized()
    arr_gpu = None
    try:
        arr_gpu = cp.asarray(arr2d)
        lo = cp.percentile(arr_gpu, p_low)
        hi = cp.percentile(arr_gpu, p_high)
        return float(lo), float(hi)
    finally:
        if arr_gpu is not None:
            del arr_gpu
        free_cupy_memory_pools()


def detect_and_correct_hot_pixels_gpu(image,
                                      threshold: float = 3.0,
                                      neighborhood_size: int = 5,
                                      progress_callback=None):
    """GPU version of hot-pixel correction using median and local variance.

    Works on HW or HWC images (float32 recommended). Falls back to CPU on error.
    """
    try:
        if not gpu_is_available():
            raise RuntimeError("GPU not available")
        import cupy as cp  # type: ignore
        from cupyx.scipy.ndimage import median_filter, uniform_filter  # type: ignore
        ensure_cupy_pool_initialized()

        if image is None:
            return None
        img = np.asarray(image, dtype=np.float32)
        if img.ndim == 2:
            img = img[:, :, None]
        h, w, c = img.shape
        k = int(neighborhood_size) if neighborhood_size % 2 == 1 else int(neighborhood_size + 1)
        k = max(3, k)

        estimated_bytes = img.size * np.dtype(np.float32).itemsize * 6
        if not gpu_memory_sufficient(int(estimated_bytes), safety_fraction=0.7):
            raise RuntimeError("GPU hot-pixel filter: insufficient memory")

        out = np.empty_like(img, dtype=np.float32)
        for ch in range(c):
            g = cp.asarray(img[..., ch])
            med = median_filter(g, size=k, mode="reflect")
            mean = uniform_filter(g, size=k)
            mean_sq = uniform_filter(g * g, size=k)
            var = cp.maximum(mean_sq - mean * mean, 0.0)
            std = cp.sqrt(var)
            # floor the std to avoid zero-division; choose a small epsilon relative to dynamic range
            eps = cp.maximum(1e-5, 1e-5 * cp.nanmax(cp.abs(g)))
            std = cp.maximum(std, eps)
            hot = g > (med + threshold * std)
            g_corr = cp.where(hot, med, g)
            out[..., ch] = cp.asnumpy(g_corr.astype(cp.float32))

        if out.shape[2] == 1:
            out = out[..., 0]
        return out
    except Exception:
        # Fallback to CPU implementation
        return detect_and_correct_hot_pixels(image, threshold=threshold,
                                             neighborhood_size=neighborhood_size,
                                             progress_callback=progress_callback)
    finally:
        free_cupy_memory_pools()


def estimate_background_map_gpu(image,
                                method: str = "gaussian",
                                sigma: float = 32.0,
                                progress_callback=None) -> np.ndarray:
    """Compute a smooth background map on GPU and return it on CPU.

    - method: currently supports 'gaussian' (cupyx.scipy.ndimage.gaussian_filter)
    - sigma: Gaussian sigma in pixels; typical 16–64 for wide background.
    """
    try:
        if not gpu_is_available():
            raise RuntimeError("GPU not available")
        import cupy as cp  # type: ignore
        from cupyx.scipy.ndimage import gaussian_filter  # type: ignore
        ensure_cupy_pool_initialized()

        img = np.asarray(image, dtype=np.float32)
        is_color = img.ndim == 3 and img.shape[-1] == 3
        estimated_bytes = img.size * np.dtype(np.float32).itemsize * (4 if is_color else 3)
        if not gpu_memory_sufficient(int(estimated_bytes), safety_fraction=0.7):
            raise RuntimeError("GPU background estimation: insufficient memory")
        if not is_color:
            g = cp.asarray(img)
            bg = gaussian_filter(g, sigma=float(sigma))
            return cp.asnumpy(bg.astype(cp.float32))
        else:
            bg = np.empty_like(img, dtype=np.float32)
            for ch in range(3):
                g = cp.asarray(img[..., ch])
                b = gaussian_filter(g, sigma=float(sigma))
                bg[..., ch] = cp.asnumpy(b.astype(cp.float32))
            return bg
    except Exception:
        # CPU fallback using OpenCV Gaussian blur as a coarse approximation
        try:
            k = max(3, int(2 * round(float(sigma) * 1.5) + 1))
            if image.ndim == 2:
                return cv2.GaussianBlur(image.astype(np.float32), (k, k), sigmaX=float(sigma))
            else:
                out = np.empty_like(image, dtype=np.float32)
                for ch in range(image.shape[-1]):
                    out[..., ch] = cv2.GaussianBlur(image[..., ch].astype(np.float32), (k, k), sigmaX=float(sigma))
                return out
        except Exception:
            # As a last resort, return zeros so subtraction is a no-op
            return np.zeros_like(image, dtype=np.float32)
    finally:
        free_cupy_memory_pools()


def stretch_auto_asifits_like_gpu(img_hwc_adu,
                                  p_low: float = 0.5,
                                  p_high: float = 99.8,
                                  asinh_a: float = 0.01,
                                  apply_wb: bool = True) -> np.ndarray:
    """GPU variant of stretch_auto_asifits_like; falls back to CPU on error."""
    try:
        if not gpu_is_available():
            raise RuntimeError("GPU not available")
        import cupy as cp  # type: ignore
        ensure_cupy_pool_initialized()
        img = np.asarray(img_hwc_adu, dtype=np.float32)
        out = np.empty_like(img, dtype=np.float32)
        estimated_bytes = img.size * np.dtype(np.float32).itemsize * 4
        if not gpu_memory_sufficient(int(estimated_bytes), safety_fraction=0.7):
            raise RuntimeError("GPU stretch: insufficient memory")
        for c in range(3):
            chan = cp.asarray(img[..., c])
            vmin = cp.percentile(chan, p_low)
            vmax = cp.percentile(chan, p_high)
            if float(vmax - vmin) < 1e-3:
                out[..., c] = 0.0
                continue
            normed = cp.clip((chan - vmin) / cp.maximum(vmax - vmin, 1e-6), 0, 1)
            stretched = cp.arcsinh(normed / asinh_a) / cp.arcsinh(1.0 / asinh_a)
            if float(cp.nanmax(stretched)) < 0.05:
                stretched = normed
            out[..., c] = cp.asnumpy(stretched.astype(cp.float32))
        if apply_wb:
            avg = out.mean(axis=(0, 1))
            m = float(np.max(avg)) if np.all(np.isfinite(avg)) else 0.0
            if m > 0:
                avg /= m
            else:
                avg = np.ones_like(avg)
            for c in range(3):
                d = float(avg[c]) if np.isfinite(avg[c]) and avg[c] > 1e-8 else 1.0
                out[..., c] = out[..., c] / d
        return np.clip(out, 0, 1).astype(np.float32)
    except Exception:
        return stretch_auto_asifits_like(img_hwc_adu, p_low=p_low, p_high=p_high, asinh_a=asinh_a, apply_wb=apply_wb)
    finally:
        free_cupy_memory_pools()






#####################################################################################################################

