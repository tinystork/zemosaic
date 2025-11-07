"""
╔══════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                  ║
║                                                                      ║
║ Auteur  : Tinystork, seigneur des couteaux à beurre (aka Tristan Nauleau)  
║ Partenaire : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System  
║              (aka ChatGPT, Grand Maître du ciselage de code)         ║
║                                                                      ║
║ Licence : GNU General Public License v3.0 (GPL-3.0)                  ║
║                                                                      ║
║ Description :                                                        ║
║   Ce programme a été forgé à la lueur des pixels et de la caféine,   ║
║   dans le but noble de transformer des nuages de photons en art      ║
║   astronomique. Si vous l’utilisez, pensez à dire “merci”,           ║
║   à lever les yeux vers le ciel, ou à citer Tinystork et J.A.R.V.I.S.║
║   (le karma des développeurs en dépend).                             ║
║                                                                      ║
║ Avertissement :                                                      ║
║   Aucune IA ni aucun couteau à beurre n’a été blessé durant le       ║
║   développement de ce code.                                          ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                  ║
║                                                                      ║
║ Author  : Tinystork, Lord of the Butter Knives (aka Tristan Nauleau) ║
║ Partner : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System  
║           (aka ChatGPT, Grand Master of Code Chiseling)              ║
║                                                                      ║
║ License : GNU General Public License v3.0 (GPL-3.0)                  ║
║                                                                      ║
║ Description:                                                         ║
║   This program was forged under the sacred light of pixels and       ║
║   caffeine, with the noble intent of turning clouds of photons into  ║
║   astronomical art. If you use it, please consider saying “thanks,”  ║
║   gazing at the stars, or crediting Tinystork and J.A.R.V.I.S. —     ║
║   developer karma depends on it.                                     ║
║                                                                      ║
║ Disclaimer:                                                          ║
║   No AIs or butter knives were harmed in the making of this code.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# zemosaic_worker.py

import os
import copy
import shutil
import time
import traceback
import gc
import logging
import inspect  # Pas utilisé directement ici, mais peut être utile pour des introspections futures
import math
import hashlib
from datetime import datetime
import psutil
import tempfile
import glob
import uuid
import multiprocessing
import threading
import itertools
import platform
import importlib.util
from pathlib import Path
from threading import Lock
from dataclasses import dataclass
from typing import Callable, Any, Iterable
from types import SimpleNamespace

import numpy as np


from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED, as_completed
# BrokenProcessPool moved under concurrent.futures.process in modern Python
from concurrent.futures.process import BrokenProcessPool

try:
    from zemosaic_utils import EXCLUDED_DIRS, is_path_excluded
except Exception:
    EXCLUDED_DIRS = frozenset({"unaligned_by_zemosaic"})

    def is_path_excluded(path, excluded_dirs=None):
        import os

        parts = set(os.path.normpath(str(path)).split(os.sep))
        dirs = set(excluded_dirs) if excluded_dirs else set()
        return any(d in parts for d in (dirs or {"unaligned_by_zemosaic"}))


UNALIGNED_DIRNAME = "unaligned_by_zemosaic"
_UNALIGNED_LOCK = Lock()


def _move_to_unaligned_safe(
    src_path: str | os.PathLike,
    input_root: str | os.PathLike,
    *,
    logger: logging.Logger = logging.getLogger(__name__),
) -> tuple[str, Path | None]:
    """Thread-safe move helper for ``unaligned_by_zemosaic``.

    Parameters
    ----------
    src_path : str | os.PathLike
        Path to the source file that should be relocated.
    input_root : str | os.PathLike
        Root input directory that owns the ``unaligned`` folder.
    logger : logging.Logger, optional
        Logger instance used for diagnostics.

    Returns
    -------
    tuple[str, Path | None]
        ``(status, destination_path)`` where ``status`` is one of
        ``{"moved", "skipped_excluded", "missing", "already_moved", "conflict", "failed"}``.
        ``destination_path`` is provided when a target directory is known.
    """

    try:
        src = Path(src_path).expanduser().resolve(strict=False)
    except Exception:
        src = Path(src_path)

    try:
        root = Path(input_root).expanduser().resolve(strict=False)
    except Exception:
        root = Path(input_root)

    # Skip if the source is already part of an excluded directory
    try:
        if is_path_excluded(src, EXCLUDED_DIRS):
            try:
                logger.warning("Skip move: path already excluded: %s", src)
            except Exception:
                pass
            return "skipped_excluded", None
    except Exception:
        if UNALIGNED_DIRNAME in set(src.parts):
            try:
                logger.warning("Skip move: path already excluded: %s", src)
            except Exception:
                pass
            return "skipped_excluded", None

    if not src.exists():
        try:
            logger.debug("Skip move: source missing (likely already moved): %s", src)
        except Exception:
            pass
        return "missing", None

    target_dir = root / UNALIGNED_DIRNAME

    with _UNALIGNED_LOCK:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            try:
                logger.debug("Failed to ensure unaligned dir exists (%s): %s", target_dir, exc)
            except Exception:
                pass

    dst = target_dir / src.name

    try:
        shutil.move(str(src), str(dst))
        try:
            logger.info("Moved to '%s': %s", UNALIGNED_DIRNAME, dst)
        except Exception:
            pass
        return "moved", dst
    except FileExistsError:
        if not src.exists():
            try:
                logger.debug("Move raced (already exists), keeping destination: %s", dst)
            except Exception:
                pass
            return "already_moved", dst
        try:
            logger.debug("Move skipped (destination exists): %s", dst)
        except Exception:
            pass
        return "conflict", dst
    except FileNotFoundError:
        try:
            logger.debug("Move skipped: source vanished before move: %s", src)
        except Exception:
            pass
        return "already_moved", dst
    except Exception as exc:
        try:
            logger.warning(
                "Move to '%s' failed for %s: %s", UNALIGNED_DIRNAME, src, exc
            )
        except Exception:
            pass
        return "failed", dst


_MASTER_TILE_ID_LOCK = Lock()
_MASTER_TILE_ID_REGISTRY: dict[str, str] = {}


def _normalize_tile_path(path: str | os.PathLike | None) -> str | None:
    """Return a stable absolute path for registry keys."""

    if path is None:
        return None
    try:
        return os.path.abspath(os.path.realpath(str(path)))
    except Exception:
        try:
            return os.path.abspath(str(path))
        except Exception:
            return str(path)


def _register_master_tile_identity(path: str | os.PathLike | None, tile_id: str | int | None) -> None:
    """Remember the logical ``tile_id`` associated with a saved tile path."""

    if path is None or tile_id is None:
        return
    normalized = _normalize_tile_path(path)
    if not normalized:
        return
    with _MASTER_TILE_ID_LOCK:
        _MASTER_TILE_ID_REGISTRY[normalized] = str(tile_id)


def _lookup_master_tile_identity(path: str | os.PathLike | None) -> str | None:
    """Fetch a previously registered ``tile_id`` for ``path`` if available."""

    if path is None:
        return None
    normalized = _normalize_tile_path(path)
    if not normalized:
        return None
    with _MASTER_TILE_ID_LOCK:
        return _MASTER_TILE_ID_REGISTRY.get(normalized)


def _resolve_tile_identifier(
    path: str | os.PathLike | None,
    header: Any | None,
    fallback_idx: int | None,
) -> str:
    """Resolve or synthesize a deterministic ``tile_id`` for a tile entry."""

    registered = _lookup_master_tile_identity(path)
    if registered:
        return registered

    candidate = None
    if header is not None:
        try:
            if "ZMT_SUPID" in header:
                candidate = str(header["ZMT_SUPID"])
        except Exception:
            candidate = None
        if candidate is None:
            try:
                if "ZMT_ID" in header:
                    candidate = f"tile:{int(header['ZMT_ID'])}"
            except Exception:
                try:
                    candidate = str(header.get("ZMT_ID")) if header.get("ZMT_ID") is not None else None
                except Exception:
                    candidate = None

    if candidate is None and fallback_idx is not None:
        candidate = f"tile:{int(fallback_idx):04d}"
    elif candidate is None and path:
        base = os.path.splitext(os.path.basename(str(path)))[0]
        candidate = f"path:{base}"
    elif candidate is None:
        candidate = "tile:unknown"

    _register_master_tile_identity(path, candidate)
    return candidate

# Nombre maximum de tentatives d'alignement avant abandon définitif
MAX_ALIGNMENT_RETRY_ATTEMPTS = 3

SYSTEM_NAME = platform.system().lower()
IS_WINDOWS = SYSTEM_NAME == "windows"
CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None and IS_WINDOWS


def _ensure_hwc_master_tile(
    data: np.ndarray,
    tile_label: str | None = None,
) -> np.ndarray:
    """Normalize master tile data to ``H x W x C`` float32 layout.

    Parameters
    ----------
    data : np.ndarray
        Array loaded from FITS (typically via ``hdul[0].data``).
    tile_label : str | None
        Optional identifier used for logging in case of shape issues.

    Returns
    -------
    np.ndarray
        Array with shape ``(H, W, C)`` and dtype ``float32``.

    Raises
    ------
    ValueError
        If the array dimensionality is unsupported or cannot be coerced
        into an ``HWC`` representation.
    """

    arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    elif arr.ndim == 3:
        if arr.shape[-1] in (1, 3):
            pass
        elif arr.shape[0] in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
        else:
            msg = (
                f"Unexpected tile shape for RGB master tile: {arr.shape}"
            )
            if tile_label:
                msg += f" (tile: {tile_label})"
            logger.error(msg)
            raise ValueError(msg)
    else:
        msg = f"Unsupported tile dimensionality: {arr.shape}"
        if tile_label:
            msg += f" (tile: {tile_label})"
        logger.error(msg)
        raise ValueError(msg)

    return np.asarray(arr, dtype=np.float32, order="C")


@dataclass
class _TileAffineSource:
    """Container for intertile photometric calibration inputs."""

    path: str | None
    wcs: Any
    data: np.ndarray | None = None


def _apply_preview_quality_crop(
    tile_array: "np.ndarray",
    crop_settings: dict | None,
) -> "np.ndarray":
    """Apply quality-crop heuristics to candidate previews when available."""

    if tile_array is None or not crop_settings:
        return tile_array
    if not crop_settings.get("enabled", False) or not ANCHOR_AUTOCROP_AVAILABLE:
        return tile_array

    arr = np.asarray(tile_array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    if arr.ndim != 3 or arr.shape[-1] == 0:
        return tile_array

    h, w, c = arr.shape
    if h < 8 or w < 8:
        return tile_array

    if c == 1:
        rgb_view = np.repeat(arr, 3, axis=-1)
    elif c >= 3:
        rgb_view = arr[..., :3]
    else:
        pad_count = 3 - c
        rgb_view = np.concatenate([arr, np.repeat(arr[..., -1:], pad_count, axis=-1)], axis=-1)

    try:
        band_px = max(4, int(crop_settings.get("band_px", 32)))
    except Exception:
        band_px = 32
    try:
        margin_px = max(0, int(crop_settings.get("margin_px", 8)))
    except Exception:
        margin_px = 8
    try:
        k_sigma = float(crop_settings.get("k_sigma", 2.0))
        if not math.isfinite(k_sigma):
            raise ValueError
    except Exception:
        k_sigma = 2.0
    k_sigma = max(0.1, min(k_sigma, 10.0))

    try:
        lum2d = rgb_view.mean(axis=-1)
        R = rgb_view[..., 0]
        G = rgb_view[..., 1]
        B = rgb_view[..., 2]
        y0, x0, y1, x1 = _anchor_detect_autocrop(
            lum2d,
            R,
            G,
            B,
            band_px=band_px,
            k_sigma=k_sigma,
            margin_px=margin_px,
        )
    except Exception:
        return tile_array

    if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
        return tile_array

    crop_area = (y1 - y0) * (x1 - x0)
    full_area = h * w if h > 0 and w > 0 else 0
    if crop_area <= 0 or (full_area > 0 and (crop_area / full_area) >= 0.97):
        return tile_array

    cropped = arr[y0:y1, x0:x1, ...]
    return np.ascontiguousarray(cropped, dtype=np.float32)


def _score_anchor_candidate(
    stats: dict,
    group_median: float,
    deviation_clip: float | None = None,
) -> float:
    """Compute a quality score for an anchor candidate (lower is better)."""

    try:
        median_val = float(stats.get("median", 0.0))
    except Exception:
        median_val = 0.0
    try:
        span_val = float(stats.get("span"))
    except Exception:
        try:
            span_val = float(stats.get("high", 0.0) - stats.get("low", 0.0))
        except Exception:
            span_val = 0.0
    try:
        robust_sigma = float(stats.get("robust_sigma", 0.0))
    except Exception:
        robust_sigma = 0.0

    if not math.isfinite(span_val):
        span_val = 0.0
    if not math.isfinite(robust_sigma):
        robust_sigma = 0.0

    deviation = abs(median_val - group_median)
    if deviation_clip is not None and deviation_clip > 0:
        deviation = min(deviation, float(deviation_clip))

    return float(deviation + 0.7 * max(span_val, 0.0) + 0.3 * max(robust_sigma, 0.0))


def _sanitize_affine_corrections(
    raw_corrections: Any,
    total_tiles: int,
) -> tuple[list[tuple[float, float]] | None, bool]:
    """Normalize gain/offset pairs and detect non-trivial entries."""

    if raw_corrections is None:
        return None, False

    sanitized: list[tuple[float, float]] = []
    nontrivial = False

    for idx in range(max(total_tiles, 0)):
        if isinstance(raw_corrections, dict):
            candidate = raw_corrections.get(idx, (1.0, 0.0))
        else:
            try:
                candidate = raw_corrections[idx]
            except (IndexError, TypeError):
                candidate = (1.0, 0.0)

        try:
            gain_val = float(candidate[0])
        except Exception:
            gain_val = 1.0
        try:
            offset_val = float(candidate[1])
        except Exception:
            offset_val = 0.0

        if not np.isfinite(gain_val):
            gain_val = 1.0
        if not np.isfinite(offset_val):
            offset_val = 0.0

        if not nontrivial and (abs(gain_val - 1.0) > 1e-6 or abs(offset_val) > 1e-6):
            nontrivial = True

        sanitized.append((gain_val, offset_val))

    if not sanitized:
        return None, False

    if not nontrivial:
        return None, False

    return sanitized, True


def _select_affine_log_indices(
    affine_list: list[tuple[float, float]] | None,
) -> set[int]:
    """Pick representative tile indices (1-based) for logging."""

    if not affine_list:
        return set()

    nontrivial_indices = [
        idx + 1
        for idx, (gain_val, offset_val) in enumerate(affine_list)
        if abs(gain_val - 1.0) > 1e-6 or abs(offset_val) > 1e-6
    ]
    if not nontrivial_indices:
        return set()

    sample_span = max(1, len(nontrivial_indices) // 4)
    selected = set(nontrivial_indices[::sample_span])
    selected.add(nontrivial_indices[0])
    selected.add(nontrivial_indices[-1])
    return selected


def _build_affine_lookup_for_tiles(
    tiles: list[dict[str, Any]],
    affine_list: list[tuple[float, float]] | None,
) -> tuple[dict[str, tuple[float, float]] | None, str | None]:
    """Map sanitized affine corrections to ``tile_id`` entries."""

    if not tiles or not affine_list:
        return None, None

    expected = len(tiles)
    if len(affine_list) != expected:
        return None, f"expected={expected}, got={len(affine_list)}"

    lookup: dict[str, tuple[float, float]] = {}
    missing: list[str] = []
    for tile_entry, affine in zip(tiles, affine_list):
        tile_id = tile_entry.get("tile_id") if isinstance(tile_entry, dict) else None
        if not tile_id:
            missing.append(f"idx{len(lookup)}")
            continue
        if tile_id in lookup:
            missing.append(f"dup:{tile_id}")
            continue
        try:
            gain_val = float(affine[0])
        except Exception:
            gain_val = 1.0
        try:
            offset_val = float(affine[1])
        except Exception:
            offset_val = 0.0
        lookup[str(tile_id)] = (gain_val, offset_val)

    if missing:
        preview = ", ".join(missing[:5])
        detail = f"missing_ids={preview}"
        return None, detail

    return lookup or None, None


def _compose_global_anchor_shift(
    affine_list: list[tuple[float, float]] | None,
    total_tiles: int,
    anchor_shift: tuple[float, float] | None,
) -> tuple[list[tuple[float, float]] | None, bool]:
    """Apply a global anchor shift to per-tile affine corrections."""

    if not anchor_shift:
        return affine_list, False
    try:
        gain_shift = float(anchor_shift[0])
    except Exception:
        gain_shift = 1.0
    try:
        offset_shift = float(anchor_shift[1])
    except Exception:
        offset_shift = 0.0
    if not np.isfinite(gain_shift):
        gain_shift = 1.0
    if not np.isfinite(offset_shift):
        offset_shift = 0.0
    if abs(gain_shift - 1.0) < 1e-6 and abs(offset_shift) < 1e-6:
        return affine_list, False

    total_tiles = max(0, int(total_tiles))
    if affine_list is None:
        if total_tiles <= 0:
            return None, False
        composed = [(gain_shift, offset_shift)] * total_tiles
    else:
        composed: list[tuple[float, float]] = []
        for idx in range(total_tiles):
            if idx < len(affine_list):
                raw = affine_list[idx]
            else:
                raw = (1.0, 0.0)
            try:
                g = float(raw[0])
            except Exception:
                g = 1.0
            try:
                o = float(raw[1])
            except Exception:
                o = 0.0
            if not np.isfinite(g):
                g = 1.0
            if not np.isfinite(o):
                o = 0.0
            new_gain = g * gain_shift
            new_offset = o * gain_shift + offset_shift
            composed.append((float(new_gain), float(new_offset)))
        composed.extend([(gain_shift, offset_shift)] * max(0, total_tiles - len(composed)))
    result = composed
    nontrivial = any(abs(g - 1.0) > 1e-6 or abs(o) > 1e-6 for g, o in result) if result else False
    return result, nontrivial


@dataclass
class _InterMasterTile:
    index: int
    path: str
    wcs: Any
    shape_hw: tuple[int, int]
    sky_corners: Any | None = None
    bbox_bounds: tuple[float, float, float, float] | None = None


def _phase45_resolve_tile_shape(path: str | None, tile_wcs: Any) -> tuple[int, int] | None:
    if path is None or not os.path.exists(path):
        return None
    if tile_wcs is not None and getattr(tile_wcs, "pixel_shape", None):
        try:
            px_shape = tile_wcs.pixel_shape
            if px_shape and len(px_shape) >= 2:
                h = int(px_shape[1])
                w = int(px_shape[0])
                if h > 0 and w > 0:
                    return (h, w)
        except Exception:
            pass
    try:
        with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
            data_shape = hdul[0].shape if hdul and hdul[0] is not None else None
            if not data_shape:
                return None
            if len(data_shape) == 2:
                return (int(data_shape[0]), int(data_shape[1]))
            if len(data_shape) == 3:
                return (int(data_shape[0]), int(data_shape[1]))
    except Exception:
        return None
    return None


def _phase45_tile_corners(tile: _InterMasterTile) -> Any | None:
    if tile.sky_corners is not None:
        return tile.sky_corners
    if not (tile.wcs and tile.wcs.is_celestial and tile.shape_hw):
        return None
    try:
        h, w = tile.shape_hw
        # Use pixel-edge coordinates (-0.5 .. width-0.5) so the projected polygon
        # matches the actual coverage footprint instead of pixel centres.
        xs = np.asarray(
            [-0.5, float(w) - 0.5, float(w) - 0.5, -0.5],
            dtype=np.float64,
        )
        ys = np.asarray(
            [-0.5, -0.5, float(h) - 0.5, float(h) - 0.5],
            dtype=np.float64,
        )
        sky = tile.wcs.pixel_to_world(xs, ys)
        tile.sky_corners = sky
        return sky
    except Exception:
        return None


def _phase45_tile_bbox(tile: _InterMasterTile) -> tuple[float, float, float, float] | None:
    if tile.bbox_bounds is not None:
        return tile.bbox_bounds
    corners = _phase45_tile_corners(tile)
    if corners is None:
        return None
    try:
        celestial = corners.icrs
    except Exception:
        celestial = corners
    ra_vals: np.ndarray | None = None
    dec_vals: np.ndarray | None = None
    for attr_name in (("ra", "dec"), ("lon", "lat")):
        ra_attr = getattr(celestial, attr_name[0], None)
        dec_attr = getattr(celestial, attr_name[1], None)
        if ra_attr is None or dec_attr is None:
            continue
        try:
            ra_vals = np.asarray(ra_attr.deg, dtype=np.float64).ravel()
            dec_vals = np.asarray(dec_attr.deg, dtype=np.float64).ravel()
            break
        except Exception:
            continue
    if ra_vals is None or dec_vals is None:
        try:
            ra_vals = np.asarray(celestial.spherical.lon.deg, dtype=np.float64).ravel()
            dec_vals = np.asarray(celestial.spherical.lat.deg, dtype=np.float64).ravel()
        except Exception:
            return None
    mask = np.isfinite(ra_vals) & np.isfinite(dec_vals)
    if not np.any(mask):
        return None
    ra_vals = ra_vals[mask]
    dec_vals = dec_vals[mask]
    bbox = (
        float(np.min(ra_vals)),
        float(np.max(ra_vals)),
        float(np.min(dec_vals)),
        float(np.max(dec_vals)),
    )
    tile.bbox_bounds = bbox
    return bbox


def _phase45_group_bbox(members: list[_InterMasterTile]) -> dict[str, float] | None:
    ra_min: float | None = None
    ra_max: float | None = None
    dec_min: float | None = None
    dec_max: float | None = None
    for tile in members:
        bbox = _phase45_tile_bbox(tile)
        if not bbox:
            continue
        t_ra_min, t_ra_max, t_dec_min, t_dec_max = bbox
        ra_min = t_ra_min if ra_min is None else min(ra_min, t_ra_min)
        ra_max = t_ra_max if ra_max is None else max(ra_max, t_ra_max)
        dec_min = t_dec_min if dec_min is None else min(dec_min, t_dec_min)
        dec_max = t_dec_max if dec_max is None else max(dec_max, t_dec_max)
    if None in (ra_min, ra_max, dec_min, dec_max):
        return None
    return {
        "ra_min": float(ra_min),
        "ra_max": float(ra_max),
        "dec_min": float(dec_min),
        "dec_max": float(dec_max),
    }


def _phase45_project_polygon(corners: Any, reference: Any) -> np.ndarray | None:
    if corners is None or reference is None:
        return None
    try:
        offset_frame = reference.skyoffset_frame()
        projected = corners.transform_to(offset_frame)
        x = np.asarray(projected.lon.arcsec, dtype=np.float64)
        y = np.asarray(projected.lat.arcsec, dtype=np.float64)
        return np.stack([x, y], axis=1)
    except Exception:
        return None


def _phase45_polygon_area(poly: np.ndarray | None) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _phase45_clip_polygon(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    if subject.size == 0 or clipper.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    def _edge_intersection(p1, p2, q1, q2):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = q1
        x4, y4 = q2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-9:
            return p2
        num_x = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        num_y = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        return np.array([num_x / denom, num_y / denom], dtype=np.float64)

    def _inside(point, edge_start, edge_end):
        return (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0]) >= 0

    output = subject
    clip_points = clipper
    for i in range(len(clip_points)):
        next_output = []
        a = clip_points[i]
        b = clip_points[(i + 1) % len(clip_points)]
        if not len(output):
            break
        prev = output[-1]
        for curr in output:
            if _inside(curr, a, b):
                if not _inside(prev, a, b):
                    next_output.append(_edge_intersection(prev, curr, a, b))
                next_output.append(curr)
            elif _inside(prev, a, b):
                next_output.append(_edge_intersection(prev, curr, a, b))
            prev = curr
        output = np.asarray(next_output, dtype=np.float64)
        if output.size == 0:
            break
    return output if output.size else np.empty((0, 2), dtype=np.float64)


def _phase45_overlap_fraction(tile_a: _InterMasterTile, tile_b: _InterMasterTile) -> float:
    if not (ASTROPY_AVAILABLE and SkyCoord and tile_a and tile_b):
        return 0.0
    sky_a = _phase45_tile_corners(tile_a)
    sky_b = _phase45_tile_corners(tile_b)
    if sky_a is None or sky_b is None:
        return 0.0
    try:
        ra_all = np.concatenate([sky_a.ra.deg, sky_b.ra.deg])
        dec_all = np.concatenate([sky_a.dec.deg, sky_b.dec.deg])
        if ra_all.size == 0 or dec_all.size == 0:
            return 0.0
        center = SkyCoord(ra=np.nanmean(ra_all), dec=np.nanmean(dec_all), unit="deg")
    except Exception:
        return 0.0

    poly_a = _phase45_project_polygon(sky_a, center)
    poly_b = _phase45_project_polygon(sky_b, center)
    if poly_a is None or poly_b is None or poly_a.size == 0 or poly_b.size == 0:
        return 0.0
    area_a = _phase45_polygon_area(poly_a)
    area_b = _phase45_polygon_area(poly_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    intersection = _phase45_clip_polygon(poly_a, poly_b)
    inter_area = _phase45_polygon_area(intersection)
    if inter_area <= 0.0:
        return 0.0
    return float(inter_area / max(1e-9, min(area_a, area_b)))


def _phase45_compute_cutout_wcs(final_wcs: Any, final_shape_hw: tuple[int, int] | None, tiles: list[_InterMasterTile]) -> tuple[Any | None, tuple[int, int] | None]:
    if not (final_wcs and final_shape_hw and tiles):
        return None, None
    try:
        wcs_copy = final_wcs.deepcopy() if hasattr(final_wcs, "deepcopy") else copy.deepcopy(final_wcs)
    except Exception:
        wcs_copy = copy.deepcopy(final_wcs)
    if wcs_copy is None:
        return None, None
    height, width = int(final_shape_hw[0]), int(final_shape_hw[1])
    edge_min_x = float("inf")
    edge_min_y = float("inf")
    edge_max_x = float("-inf")
    edge_max_y = float("-inf")
    valid = False
    for tile in tiles:
        sky = _phase45_tile_corners(tile)
        if sky is None:
            continue
        try:
            px, py = wcs_copy.world_to_pixel(sky)
            if px is None or py is None:
                continue
            px = np.asarray(px, dtype=np.float64)
            py = np.asarray(py, dtype=np.float64)
        except Exception:
            continue
        if px.size == 0 or py.size == 0:
            continue
        edge_min_x = min(edge_min_x, float(np.nanmin(px)))
        edge_min_y = min(edge_min_y, float(np.nanmin(py)))
        edge_max_x = max(edge_max_x, float(np.nanmax(px)))
        edge_max_y = max(edge_max_y, float(np.nanmax(py)))
        valid = True
    if not valid:
        return None, None
    # Convert edge coordinates (relative to pixel boundaries) into integer pixel
    # indices that map to the same mosaic grid as the final WCS.
    start_x = max(0, int(math.floor(edge_min_x + 0.5)))
    start_y = max(0, int(math.floor(edge_min_y + 0.5)))
    stop_x = min(width, int(math.ceil(edge_max_x - 0.5)) + 1)
    stop_y = min(height, int(math.ceil(edge_max_y - 0.5)) + 1)
    if stop_x <= start_x:
        stop_x = min(width, start_x + 1)
    if stop_y <= start_y:
        stop_y = min(height, start_y + 1)
    local_w = max(1, int(stop_x - start_x))
    local_h = max(1, int(stop_y - start_y))
    try:
        if hasattr(wcs_copy, "wcs") and hasattr(wcs_copy.wcs, "crpix"):
            wcs_copy.wcs.crpix[0] -= start_x
            wcs_copy.wcs.crpix[1] -= start_y
        if hasattr(wcs_copy, "array_shape"):
            wcs_copy.array_shape = (local_h, local_w)
    except Exception:
        pass
    return wcs_copy, (local_h, local_w)


def _phase45_allocate_stack_storage(
    count: int,
    shape_hw: tuple[int, int],
    channels: int,
    policy: str,
    temp_dir: str | None,
) -> tuple[np.ndarray, str | None]:
    dtype = np.float32
    h, w = shape_hw
    policy_norm = str(policy or "auto").lower()
    use_memmap = False
    if policy_norm == "always":
        use_memmap = True
    elif policy_norm == "auto":
        estimated_bytes = count * h * w * channels * np.dtype(dtype).itemsize
        use_memmap = estimated_bytes > (256 * 1024 * 1024)
    if not use_memmap:
        return np.full((count, h, w, channels), np.nan, dtype=dtype), None
    directory = temp_dir or tempfile.gettempdir()
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"phase45_stack_{uuid.uuid4().hex}.dat")
    storage = np.memmap(file_path, dtype=dtype, mode="w+", shape=(count, h, w, channels))
    storage[:] = np.nan
    return storage, file_path


def _phase45_cleanup_storage(storage: np.ndarray | None, memmap_path: str | None) -> None:
    try:
        if isinstance(storage, np.memmap):
            storage.flush()
    except Exception:
        pass
    if memmap_path and os.path.exists(memmap_path):
        try:
            os.remove(memmap_path)
        except Exception:
            pass


def _run_phase4_5_inter_master_merge(
    master_tiles: list[tuple[str | None, Any]],
    final_output_wcs: Any,
    final_output_shape_hw: tuple[int, int] | None,
    temp_storage_dir: str | None,
    output_folder: str,
    cache_retention_mode: str,
    inter_cfg: dict,
    stack_cfg: dict,
    progress_callback: Callable | None,
    pcb: Callable[..., None],
) -> list[tuple[str | None, Any]]:
    enable = bool(inter_cfg.get("enable", False))
    if not enable:
        return master_tiles
    if len(master_tiles) < 2:
        return master_tiles
    if not (ASTROPY_AVAILABLE and REPROJECT_AVAILABLE and ZEMOSAIC_UTILS_AVAILABLE and ZEMOSAIC_ALIGN_STACK_AVAILABLE and reproject_interp):
        pcb("p45_finished", prog=None, lvl="INFO_DETAIL", tiles_in=len(master_tiles), tiles_out=len(master_tiles))
        return master_tiles

    threshold = float(inter_cfg.get("overlap_threshold", 0.60))
    threshold = max(0.0, min(1.0, threshold))
    min_group = max(2, int(inter_cfg.get("min_group_size", 2)))
    max_group = max(min_group, int(inter_cfg.get("max_group", 64)))
    memmap_policy = str(inter_cfg.get("memmap_policy", "auto")).lower()
    local_scale = str(inter_cfg.get("local_scale", "native")).lower()
    if local_scale not in {"final", "native"}:
        local_scale = "native"

    photometry_intragroup = bool(inter_cfg.get("photometry_intragroup", True))
    photometry_intersuper = bool(inter_cfg.get("photometry_intersuper", True))
    try:
        photometry_clip_sigma = float(inter_cfg.get("photometry_clip_sigma", 3.0))
    except Exception:
        photometry_clip_sigma = 3.0
    if not math.isfinite(photometry_clip_sigma):
        photometry_clip_sigma = 3.0
    photometry_clip_sigma = max(0.1, photometry_clip_sigma)

    pcb("p45_start", prog=None, lvl="INFO")

    def _phase45_gui_emit(key=None, *, level: str = "DEBUG", **kwargs) -> bool:
        """Emit a GUI log safely, falling back silently if the GUI is detached."""
        if not pcb:
            return False
        try:
            pcb(key, prog=None, lvl=level, **kwargs)
            return True
        except Exception:
            return False

    def _phase45_gui_message(message: str, level: str = "DEBUG") -> bool:
        """Send a free-form heartbeat message to the GUI log."""
        return _phase45_gui_emit(message, level=level)

    logger.debug(
        "[P4.5] Starting Inter-Master merge: threshold=%.2f, min_group=%d, max_group=%d, "
        "memmap=%s, local_scale=%s, master_tiles=%d",
        threshold,
        min_group,
        max_group,
        memmap_policy,
        local_scale,
        len(master_tiles),
    )
    if not _phase45_gui_emit(
        "p45_group_info",
        level="DEBUG",
        group=0,
        total=len(master_tiles),
        members=len(master_tiles),
    ):
        _phase45_gui_message(
            f"Phase 4.5: preparing overlap graph (tiles={len(master_tiles)})"
        )

    micro_align_available = hasattr(zemosaic_align_stack, "micro_align_stack")
    photometry_estimator_available = hasattr(zemosaic_align_stack, "estimate_affine_photometry")
    photometry_apply_available = hasattr(zemosaic_align_stack, "apply_affine_photometry")

    tiles: list[_InterMasterTile] = []
    group_super_counts: dict[int, int] = {}
    for idx, (path, wcs_obj) in enumerate(master_tiles):
        shape_hw = _phase45_resolve_tile_shape(path, wcs_obj)
        if not shape_hw:
            continue
        tiles.append(_InterMasterTile(index=idx, path=path, wcs=wcs_obj, shape_hw=shape_hw))

    logger.debug(
        "[P4.5] Candidate tiles retained for overlap graph: %d/%d",
        len(tiles),
        len(master_tiles),
    )
    _phase45_gui_message(f"Phase 4.5: building overlap graph with {len(tiles)} tiles")

    if len(tiles) < min_group:
        pcb("p45_finished", prog=None, lvl="INFO_DETAIL", tiles_in=len(master_tiles), tiles_out=len(master_tiles))
        return master_tiles

    tile_map = {tile.index: tile for tile in tiles}
    adjacency: dict[int, set[int]] = {tile.index: set() for tile in tiles}
    total_pairs = 0
    for i, tile_a in enumerate(tiles):
        for tile_b in tiles[i + 1 :]:
            overlap = _phase45_overlap_fraction(tile_a, tile_b)
            if overlap >= threshold:
                adjacency[tile_a.index].add(tile_b.index)
                adjacency[tile_b.index].add(tile_a.index)
                total_pairs += 1
        if i and i % 10 == 0:
            msg = (
                f"Phase 4.5: overlap scan {i}/{len(tiles)} (edges>={threshold:.2f}: {total_pairs})"
            )
            logger.debug("[P4.5] %s", msg)
            _phase45_gui_message(msg)
    logger.debug(
        "[P4.5] Overlap graph: %d tiles, %d edges >= %.2f",
        len(tiles),
        total_pairs,
        threshold,
    )
    _phase45_gui_message(
        f"Phase 4.5: overlap graph ready ({total_pairs} edges >= {threshold:.2f})"
    )
    visited: set[int] = set()
    groups: list[list[int]] = []
    total_components = 0
    valid_components = 0
    for tile in tiles:
        if tile.index in visited:
            continue
        stack = [tile.index]
        component: list[int] = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(adjacency.get(node, ()))
        total_components += 1
        if len(component) >= min_group:
            groups.append(sorted(component))
            valid_components += 1

    logger.debug(
        "[P4.5] Connected components: total=%d, meeting_min_group=%d (min_group=%d)",
        total_components,
        valid_components,
        min_group,
    )
    if not groups:
        _phase45_gui_message("Phase 4.5: no groups reached the minimum overlap")

    if not groups:
        pcb("p45_finished", prog=None, lvl="INFO_DETAIL", tiles_in=len(master_tiles), tiles_out=len(master_tiles))
        return master_tiles

    _phase45_gui_message(
        f"Phase 4.5: overlap grouping ready ({len(groups)} groups >= {min_group})"
    )
    group_layout_payload: list[dict[str, Any]] = []
    for gid, component in enumerate(groups, start=1):
        member_tiles = [tile_map[idx] for idx in component if idx in tile_map]
        if not member_tiles:
            continue
        repr_idx = min(tile.index for tile in member_tiles)
        bbox = _phase45_group_bbox(member_tiles)
        group_layout_payload.append(
            {
                "group_id": gid,
                "members": [tile.index for tile in member_tiles],
                "repr": repr_idx,
                "bbox": bbox,
            }
        )
    if group_layout_payload:
        logger.debug("[P4.5] Emitting groups layout payload (%d groups)", len(group_layout_payload))
        _phase45_gui_emit(
            "p45_groups_layout",
            level="DEBUG",
            groups=group_layout_payload,
            total_groups=len(groups),
        )

    total_chunks = sum(max(1, math.ceil(len(group) / max_group)) for group in groups)
    logger.debug(
        "[P4.5] Planned processing batches: groups=%d, chunks=%d, max_group=%d",
        len(groups),
        total_chunks,
        max_group,
    )
    _phase45_gui_message(
        f"Phase 4.5: processing {len(groups)} groups / {total_chunks} chunks"
    )
    processed_chunks = 0
    replacements: dict[int, tuple[str | None, Any]] = {}
    consumed_indices: set[int] = set()
    cleanup_paths: list[str] = []

    for group_id, group_indices in enumerate(groups, start=1):
        members = [tile_map[idx] for idx in group_indices if idx in tile_map]
        if not members:
            continue
        group_chunks = max(1, math.ceil(len(members) / max_group))
        member_ids = [tile.index for tile in members]
        group_completed_chunks = 0
        if not _phase45_gui_emit(
            "p45_group_info",
            level="DEBUG",
            group=group_id,
            total=len(groups),
            members=len(members),
            chunks=group_chunks,
        ):
            _phase45_gui_message(
                f"Phase 4.5: group {group_id} queued ({len(members)} tiles)"
            )
        logger.debug(
            "[P4.5][G%03d] Preparing group: members=%d, chunks=%d, ids=%s",
            group_id,
            len(members),
            group_chunks,
            member_ids,
        )
        for chunk_start in range(0, len(members), max_group):
            chunk_tiles = members[chunk_start : chunk_start + max_group]
            processed_chunks += 1
            if progress_callback:
                try:
                    progress_callback("phase4_5", processed_chunks, total_chunks)
                except Exception:
                    pass
            chunk_idx = min(group_chunks, (chunk_start // max_group) + 1)
            _phase45_gui_emit(
                "p45_group_started",
                level="DEBUG",
                group_id=group_id,
                chunk=chunk_idx,
                chunks=group_chunks,
                size=len(chunk_tiles),
            )
            _phase45_gui_emit(
                "p45_group_progress",
                level="ETA_LEVEL",
                group_id=group_id,
                chunk=chunk_idx,
                done=group_completed_chunks,
                total=group_chunks,
                size=len(chunk_tiles),
            )
            if not _phase45_gui_emit(
                "p45_group_info",
                level="DEBUG",
                group=group_id,
                total=len(groups),
                members=len(chunk_tiles),
                chunk=chunk_idx,
                chunks=group_chunks,
                processed=processed_chunks,
                total_chunks=total_chunks,
            ):
                _phase45_gui_message(
                    f"Phase 4.5: processing chunk {chunk_idx}/{group_chunks} "
                    f"(group {group_id}, global {processed_chunks}/{total_chunks})"
                )
            logger.debug(
                "[P4.5][G%03d] Chunk %d/%d (%d tiles) - global chunk %d/%d",
                group_id,
                chunk_idx,
                group_chunks,
                len(chunk_tiles),
                processed_chunks,
                total_chunks,
            )

            reference_tile = chunk_tiles[0]
            if local_scale == "final" and final_output_wcs and final_output_shape_hw:
                local_wcs, local_shape = _phase45_compute_cutout_wcs(final_output_wcs, final_output_shape_hw, chunk_tiles)
                if local_wcs is None or local_shape is None:
                    local_wcs = reference_tile.wcs
                    local_shape = reference_tile.shape_hw
            else:
                local_wcs = reference_tile.wcs
                local_shape = reference_tile.shape_hw

            if not (local_wcs and local_shape):
                continue
            logger.debug(
                "[P4.5][G%03d] Local frame resolved: scale=%s, shape=%s, ref_tile=%d",
                group_id,
                local_scale,
                tuple(local_shape),
                reference_tile.index,
            )
            _phase45_gui_message(
                f"Phase 4.5: group {group_id} local frame {local_shape[0]}x{local_shape[1]}"
            )

            storage = None
            memmap_path = None
            success = True
            frames: list[np.ndarray] = []
            channels = 3
            preloaded: dict[int, np.ndarray] = {}
            try:
                with fits.open(chunk_tiles[0].path, memmap=True, do_not_scale_image_data=True) as hdul_sample:
                    first_arr = _ensure_hwc_master_tile(hdul_sample[0].data, os.path.basename(chunk_tiles[0].path))
                channels = int(first_arr.shape[-1]) if first_arr.ndim == 3 else 1
                if channels <= 0:
                    channels = 1
                if channels not in (1, 3):
                    channels = min(3, max(1, channels))
                    if first_arr.shape[-1] != channels:
                        first_arr = first_arr[..., :channels]
                preloaded[chunk_tiles[0].index] = first_arr.astype(np.float32, copy=False)
            except Exception:
                success = False
                channels = 3
            if not success:
                logger.debug("[P4.5][G%03d] Failed to preload reference tile for chunk %d/%d", group_id, chunk_idx, group_chunks)
                continue
            try:
                storage, memmap_path = _phase45_allocate_stack_storage(len(chunk_tiles), local_shape, channels, memmap_policy, temp_storage_dir)
                allocation_mode = "memmap" if memmap_path else "ram"
                logger.debug(
                    "[P4.5][G%03d] Allocated %s stack storage for chunk %d/%d: "
                    "tiles=%d, shape=%s, channels=%d",
                    group_id,
                    allocation_mode,
                    chunk_idx,
                    group_chunks,
                    len(chunk_tiles),
                    local_shape,
                    channels,
                )
                _phase45_gui_message(
                    f"Phase 4.5: group {group_id} chunk {chunk_idx}/{group_chunks} reprojection ({allocation_mode})"
                )
                for idx_tile, tile in enumerate(chunk_tiles):
                    if tile.index in preloaded:
                        arr = preloaded.pop(tile.index)
                    else:
                        with fits.open(tile.path, memmap=True, do_not_scale_image_data=True) as hdul:
                            arr = _ensure_hwc_master_tile(hdul[0].data, os.path.basename(tile.path))
                            arr = np.asarray(arr, dtype=np.float32)
                    if arr.ndim == 2:
                        arr = arr[..., np.newaxis]
                    if arr.shape[-1] != channels:
                        if channels == 3 and arr.shape[-1] == 1:
                            arr = np.repeat(arr, 3, axis=-1)
                        elif channels == 1:
                            arr = arr[..., :1]
                        else:
                            arr = np.repeat(arr[..., :1], channels, axis=-1)
                    arr = np.asarray(arr, dtype=np.float32, copy=False)
                    reproj = np.full((local_shape[0], local_shape[1], channels), np.nan, dtype=np.float32)
                    for ch in range(channels):
                        plane = arr[..., ch]
                        reproj_plane, footprint = reproject_interp((plane, tile.wcs), local_wcs, shape_out=(local_shape[0], local_shape[1]))
                        if reproj_plane is None:
                            success = False
                            break
                        reproj_plane = np.asarray(reproj_plane, dtype=np.float32)
                        if footprint is not None:
                            mask = np.asarray(footprint) <= 0.0
                            if mask.shape == reproj_plane.shape:
                                reproj_plane[mask] = np.nan
                        reproj[..., ch] = reproj_plane
                    if not success:
                        break
                    storage[idx_tile, :, :, :] = reproj
                    frames.append(storage[idx_tile])
                if not success or not frames:
                    success = False
            except Exception:
                success = False

            if not success:
                logger.debug(
                    "[P4.5][G%03d] Chunk %d/%d aborted during reprojection",
                    group_id,
                    chunk_idx,
                    group_chunks,
                )
                _phase45_cleanup_storage(storage, memmap_path)
                continue

            try:
                kappa_val = float(stack_cfg.get("kappa_low", 3.0))
            except Exception:
                kappa_val = 3.0
            limits_val = stack_cfg.get("winsor_limits", (0.05, 0.05))
            try:
                limits_val = (
                    float(limits_val[0]),
                    float(limits_val[1]),
                )
            except Exception:
                limits_val = (0.05, 0.05)
            try:
                max_pass_val = int(stack_cfg.get("winsor_max_frames_per_pass", 0))
            except Exception:
                max_pass_val = 0
            try:
                worker_limit_val = int(stack_cfg.get("winsor_worker_limit", 1))
            except Exception:
                worker_limit_val = 1
            stack_kwargs = {
                "kappa": kappa_val,
                "winsor_limits": limits_val,
                "winsor_max_frames_per_pass": max_pass_val,
                "winsor_max_workers": max(1, worker_limit_val),
            }
            super_arr = None
            # --- Phase 4.5 (améliorée) : 2.1 → 3.1 à l’intérieur de 4.5 ---
            # Réutiliser les options choisies dans le GUI :
            #   normalize_method, weight_method, reject_algo, final_combine,
            #   kappa/winsor/workers déjà présents dans stack_cfg/stack_kwargs.
            try:
                norm_method = str(stack_cfg.get("normalize_method", stack_cfg.get("stack_norm_method", "none"))).lower()
            except Exception:
                norm_method = "none"
            try:
                weight_method = str(stack_cfg.get("weight_method", stack_cfg.get("stack_weight_method", "none"))).lower()
            except Exception:
                weight_method = "none"
            try:
                reject_algo = str(stack_cfg.get("reject_algo", stack_cfg.get("stack_reject_algo", "winsorized_sigma_clip"))).lower()
            except Exception:
                reject_algo = "winsorized_sigma_clip"
            try:
                final_combine = str(stack_cfg.get("final_combine", stack_cfg.get("stack_final_combine", "mean"))).lower()
            except Exception:
                final_combine = "mean"

            # 4.5.a — micro-alignement résiduel (noop si indisponible)
            try:
                if micro_align_available:
                    logger.debug(
                        "[P4.5][G%03d] Micro-align start: method=phase, frames=%d",
                        group_id,
                        len(frames),
                    )
                    _phase45_gui_message(
                        f"Phase 4.5: group {group_id} micro-align ({len(frames)} frames)"
                    )
                    frames = zemosaic_align_stack.micro_align_stack(
                        frames, method="phase", max_shift_px=8
                    )
                    logger.debug("[P4.5][G%03d] Micro-align done", group_id)
                    _phase45_gui_message(
                        f"Phase 4.5: group {group_id} micro-align done"
                    )
            except Exception as exc:
                logger.debug("[P4.5][G%03d] Micro-align skipped/failed: %s", group_id, exc)

            # 4.5.b — normalisation photométrique intra-groupe (gain/offset)
            if (
                photometry_estimator_available
                and photometry_apply_available
                and photometry_intragroup
                and len(frames) >= 2
            ):
                try:
                    _phase45_gui_emit(
                        "p45_group_photometry_start",
                        level="INFO_DETAIL",
                        group_id=group_id,
                        chunk=chunk_idx,
                        size=len(chunk_tiles),
                        clip=photometry_clip_sigma,
                    )
                    logger.debug(
                        "[P4.5][G%03d] Intra-group photometry start (chunk=%d, tiles=%d, clip=%.2f)",
                        group_id,
                        chunk_idx,
                        len(chunk_tiles),
                        photometry_clip_sigma,
                    )
                    chunk_sources: list[_TileAffineSource] = []
                    for idx_tile, tile in enumerate(chunk_tiles):
                        if idx_tile >= len(frames):
                            break
                        frame_view = np.array(frames[idx_tile], dtype=np.float32, copy=False)
                        chunk_sources.append(
                            _TileAffineSource(
                                path=tile.path,
                                wcs=local_wcs,
                                data=frame_view,
                            )
                        )
                    if len(chunk_sources) >= 2:
                        affine_estimates = zemosaic_align_stack.estimate_affine_photometry(
                            chunk_sources,
                            robust=True,
                            clip_sigma=photometry_clip_sigma,
                            match_background=True,
                        )
                        corrections, nontrivial = _sanitize_affine_corrections(
                            affine_estimates,
                            len(chunk_sources),
                        )
                        if corrections and nontrivial:
                            frames = zemosaic_align_stack.apply_affine_photometry(frames, corrections)
                            log_indices = _select_affine_log_indices(corrections)
                            sample_indices = [
                                chunk_tiles[idx - 1].index
                                for idx in sorted(log_indices)
                                if 0 < idx <= len(chunk_tiles)
                            ]
                            sample_payload = ", ".join(str(val) for val in sample_indices) if sample_indices else "-"
                            _phase45_gui_emit(
                                "p45_group_photometry_applied",
                                level="INFO_DETAIL",
                                group_id=group_id,
                                chunk=chunk_idx,
                                applied=len(corrections),
                                samples=sample_payload,
                            )
                            logger.debug(
                                "[P4.5][G%03d] Intra-group photometry applied (chunk=%d, samples=%s)",
                                group_id,
                                chunk_idx,
                                sample_payload,
                            )
                        else:
                            logger.debug(
                                "[P4.5][G%03d] Intra-group photometry produced no corrections (chunk=%d)",
                                group_id,
                                chunk_idx,
                            )
                    else:
                        logger.debug(
                            "[P4.5][G%03d] Intra-group photometry skipped (insufficient tiles: %d)",
                            group_id,
                            len(chunk_sources),
                        )
                except Exception as exc:
                    logger.debug(
                        "[P4.5][G%03d] Intra-group photometry skipped due to error: %s",
                        group_id,
                        exc,
                    )

            # Héritage: respecter les réglages legacy si la nouvelle photométrie est désactivée.
            if (
                not photometry_intragroup
                and photometry_estimator_available
                and photometry_apply_available
                and norm_method != "none"
                and len(frames) >= 2
            ):
                try:
                    logger.debug(
                        "[P4.5][G%03d] Legacy inter-tile photometric normalization (method=%s, weight=%s)",
                        group_id,
                        norm_method,
                        weight_method,
                    )
                    affine_corr = zemosaic_align_stack.estimate_affine_photometry(
                        frames, method=norm_method, weight_method=weight_method
                    )
                    frames = zemosaic_align_stack.apply_affine_photometry(frames, affine_corr)
                    logger.debug("[P4.5][G%03d] Legacy inter-tile normalization applied", group_id)
                    _phase45_gui_message(
                        f"Phase 4.5: group {group_id} inter-tile photometric normalization"
                    )
                except Exception as exc:
                    logger.warning("[P4.5][G%03d] Legacy inter-tile normalization failed: %s", group_id, exc)

            # 4.5.c — empilement selon les réglages GUI
            logger.debug(
                "[P4.5][G%03d] Stack params: reject=%s, combine=%s, kappa=%.2f, winsor_limits=%s, workers=%d, weight=%s",
                group_id,
                reject_algo,
                final_combine,
                kappa_val,
                limits_val,
                stack_kwargs["winsor_max_workers"],
                weight_method,
            )
            _phase45_gui_message(
                f"Phase 4.5: group {group_id} stacking ({reject_algo}/{final_combine})"
            )
            try:
                if reject_algo in ("winsor", "winsorized_sigma_clip"):
                    result = zemosaic_align_stack.stack_winsorized_sigma_clip(
                        frames, weight_method=weight_method, zconfig=None, **stack_kwargs
                    )
                    super_arr = result[0] if isinstance(result, (tuple, list)) else result
                elif reject_algo == "kappa_sigma":
                    stack_result = None
                    if hasattr(zemosaic_align_stack, "stack_kappa_sigma"):
                        stack_result = zemosaic_align_stack.stack_kappa_sigma(
                            frames,
                            kappa=float(stack_cfg.get("kappa_low", 3.0)),
                            combine=final_combine,
                            weight_method=weight_method,
                        )
                    elif hasattr(zemosaic_align_stack, "stack_kappa_sigma_clip"):
                        stack_result = zemosaic_align_stack.stack_kappa_sigma_clip(
                            frames,
                            weight_method=weight_method,
                            zconfig=None,
                            sigma_low=float(stack_cfg.get("kappa_low", 3.0)),
                            sigma_high=float(stack_cfg.get("kappa_high", stack_cfg.get("kappa_low", 3.0))),
                        )
                    if stack_result is not None:
                        super_arr = stack_result[0] if isinstance(stack_result, (tuple, list)) else stack_result
                elif reject_algo == "linear_fit_clip" and hasattr(zemosaic_align_stack, "stack_linear_fit_clip"):
                    result = zemosaic_align_stack.stack_linear_fit_clip(
                        frames,
                        weight_method=weight_method,
                        zconfig=None,
                        sigma=float(stack_cfg.get("kappa_high", stack_cfg.get("kappa_low", 3.0))),
                    )
                    super_arr = result[0] if isinstance(result, (tuple, list)) else result

                if super_arr is None:
                    stack_np = np.stack(frames, axis=0).astype(np.float32, copy=False)
                    super_arr = (
                        np.nanmedian(stack_np, axis=0).astype(np.float32)
                        if final_combine == "median"
                        else np.nanmean(stack_np, axis=0).astype(np.float32)
                    )
            except Exception as exc:
                logger.debug("[P4.5][G%03d] Stack failed: %s", group_id, exc)
                _phase45_cleanup_storage(storage, memmap_path)
                continue

            _phase45_cleanup_storage(storage, memmap_path)

            if super_arr is None:
                logger.debug(
                    "[P4.5][G%03d] Stack yielded no data; skipping chunk %d/%d",
                    group_id,
                    chunk_idx,
                    group_chunks,
                )
                continue

            try:
                super_arr = _ensure_hwc_master_tile(
                    super_arr,
                    tile_label=f"p45_group_{group_id:03d}_chunk_{chunk_idx}",
                )
            except Exception as exc:
                logger.debug(
                    "[P4.5][G%03d] Super array normalization failed: %s",
                    group_id,
                    exc,
                )
                continue
            if channels == 1 and super_arr.shape[-1] != 1:
                super_arr = super_arr[..., :1]
            elif channels == 3:
                if super_arr.shape[-1] == 1:
                    super_arr = np.repeat(super_arr, 3, axis=-1)
                elif super_arr.shape[-1] > 3:
                    super_arr = super_arr[..., :3]
            super_arr = np.asarray(super_arr, dtype=np.float32, order="C")
            arr_shape = tuple(super_arr.shape)
            logger.debug(
                "[P4.5][G%03d] Super array ready: shape=%s, dtype=%s",
                group_id,
                arr_shape,
                super_arr.dtype,
            )
            _phase45_gui_message(
                f"Phase 4.5: group {group_id} super-tile shape {arr_shape}"
            )
            member_indices = sorted(tile.index for tile in chunk_tiles)
            representative_idx = member_indices[0]
            consumed_indices.update(tile.index for tile in chunk_tiles)

            member_signature = ",".join(str(idx) for idx in member_indices)
            digest = hashlib.sha1(member_signature.encode("utf-8")).hexdigest()[:8]
            super_tile_id = f"super:{representative_idx:04d}_{len(member_indices):02d}_{digest}"

            target_dir = temp_storage_dir or output_folder
            os.makedirs(target_dir, exist_ok=True)
            super_filename = f"super_tile_{representative_idx:03d}_{uuid.uuid4().hex[:8]}.fits"
            super_path = os.path.join(target_dir, super_filename)
            logger.debug("[P4.5][G%03d] Saving super tile to %s", group_id, super_path)
            _phase45_gui_message(
                f"Phase 4.5: group {group_id} writing {super_filename} ({arr_shape})"
            )

            header = fits.Header()
            try:
                header.update(local_wcs.to_header(relax=True))
            except Exception:
                pass
            try:
                header["ZMT_TYPE"] = ("Super Tile", "Inter-Master merged tile")
                header["ZMT_SUPR"] = (len(chunk_tiles), "Tiles merged in Phase 4.5")
            except Exception:
                pass
            try:
                header["ZMT_SUPID"] = (super_tile_id, "Phase 4.5 super tile id")
            except Exception:
                pass
            try:
                header.add_history(f"Inter-Master merge ({len(chunk_tiles)} tiles)")
            except Exception:
                header["HISTORY"] = f"Inter-Master merge ({len(chunk_tiles)} tiles)"

            try:
                zemosaic_utils.save_fits_image(
                    image_data=super_arr,
                    output_path=super_path,
                    header=header,
                    overwrite=True,
                    save_as_float=True,
                    axis_order="HWC" if super_arr.ndim == 3 else None,
                )
            except Exception:
                try:
                    fits.writeto(super_path, super_arr.astype(np.float32), header=header, overwrite=True)
                except Exception:
                    continue

            _register_master_tile_identity(super_path, super_tile_id)

            replacements[representative_idx] = (super_path, local_wcs)
            cleanup_paths.extend(tile.path for tile in chunk_tiles if tile.path)
            group_super_counts[group_id] = group_super_counts.get(group_id, 0) + 1

            snr_gain = math.sqrt(len(chunk_tiles))
            pcb(
                "p45_group_result",
                prog=None,
                lvl="INFO_DETAIL",
                size=len(chunk_tiles),
                out=os.path.basename(super_path),
                snr=f"{snr_gain:.2f}",
                group_id=group_id,
                chunk=chunk_idx,
            )
            logger.debug(
                "[P4.5][G%03d] Super tile saved (%d tiles, SNR x%.2f): %s shape=%s",
                group_id,
                len(chunk_tiles),
                snr_gain,
                super_path,
                arr_shape,
            )
            _phase45_gui_message(
                f"Phase 4.5: group {group_id} super tile saved "
                f"(chunk {chunk_idx}/{group_chunks}, SNR x{snr_gain:.2f})"
            )
            group_completed_chunks = max(group_completed_chunks, chunk_idx)
            _phase45_gui_emit(
                "p45_group_progress",
                level="ETA_LEVEL",
                group_id=group_id,
                chunk=chunk_idx,
                done=group_completed_chunks,
                total=group_chunks,
                size=len(chunk_tiles),
            )

    total_super_tiles = sum(group_super_counts.values())
    if total_super_tiles:
        groups_with_super = len(group_super_counts)
        logger.info(
            "phase4_5: injected %d super-tiles from %d groups",
            total_super_tiles,
            groups_with_super,
        )
        _phase45_gui_message(
            f"phase4_5: injected {total_super_tiles} super-tiles from {groups_with_super} groups",
            level="INFO",
        )

    if cache_retention_mode != "keep":
        removed_count = 0
        for path in cleanup_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                os.remove(path)
                removed_count += 1
            except Exception:
                pass
        if removed_count:
            logger.debug("[P4.5] Removed %d original master tiles post-merge", removed_count)
            _phase45_gui_message(f"Phase 4.5: cleanup removed {removed_count} original tiles")

    # --- [P4.5] Inter-super photometric normalization (gain-only) ---
    try:
        candidate_super_tiles = [
            (tidx, entry)
            for tidx, entry in sorted(replacements.items())
            if entry and entry[0] and os.path.exists(entry[0])
        ]
        if photometry_intersuper and len(candidate_super_tiles) >= 2:
            pcb("p45_norm_start", lvl="INFO_DETAIL", tiles=len(candidate_super_tiles))
            end_payload: dict[str, Any] = {"applied": 0}
            try:
                stats: list[tuple[int, str, np.ndarray, int]] = []
                for tidx, (tpath, _twcs) in candidate_super_tiles:
                    try:
                        with fits.open(
                            tpath,
                            memmap=True,
                            do_not_scale_image_data=True,
                        ) as hdul:
                            raw = hdul[0].data
                            if raw is None:
                                continue
                            arr = _ensure_hwc_master_tile(raw, os.path.basename(tpath))
                    except Exception as exc:
                        logger.debug("[P4.5] Norm stats failed for %s: %s", tpath, exc)
                        continue
                    channels = arr.shape[-1]
                    med = np.full(channels, np.nan, dtype=np.float32)
                    weight = 0
                    for ch in range(channels):
                        plane = arr[..., ch]
                        valid = np.isfinite(plane)
                        valid_count = int(valid.sum())
                        if not valid_count:
                            continue
                        vals = plane[valid].astype(np.float32, copy=False)
                        median_val = float(np.nanmedian(vals))
                        if valid_count > 32 and photometry_clip_sigma > 0:
                            try:
                                q1, q3 = np.nanpercentile(vals, [25.0, 75.0])
                                iqr = float(q3 - q1)
                                if iqr > 0:
                                    sigma_est = iqr / 1.349
                                    clip_width = sigma_est * photometry_clip_sigma
                                    if clip_width > 0:
                                        low = median_val - clip_width
                                        high = median_val + clip_width
                                        clip_mask = (vals >= low) & (vals <= high)
                                        if clip_mask.any():
                                            median_val = float(np.nanmedian(vals[clip_mask]))
                            except Exception:
                                pass
                        med[ch] = median_val
                        weight += valid_count
                    if weight == 0 or not np.all(np.isfinite(med)):
                        continue
                    stats.append((int(tidx), tpath, med, weight))

                valid_stats = [entry for entry in stats if entry[3] > 0 and np.all(np.isfinite(entry[2]))]
                if len(valid_stats) >= 2:
                    meds = np.vstack([entry[2] for entry in valid_stats])
                    weights = np.asarray([float(entry[3]) for entry in valid_stats], dtype=np.float64)
                    total_weight = float(weights.sum())
                    ref_mode = "median"
                    dominant_idx = int(np.argmax(weights))
                    ref_tile = int(valid_stats[dominant_idx][0])
                    ref = np.zeros(meds.shape[1], dtype=np.float32)
                    if total_weight <= 0:
                        ref_mode = "tile"
                        ref = meds[dominant_idx]
                    else:
                        for ch in range(meds.shape[1]):
                            col = meds[:, ch]
                            order = np.argsort(col)
                            sorted_vals = col[order]
                            sorted_weights = weights[order]
                            cutoff = total_weight * 0.5
                            cumsum = np.cumsum(sorted_weights)
                            idx = int(np.searchsorted(cumsum, cutoff, side="left"))
                            idx = min(idx, len(sorted_vals) - 1)
                            ref[ch] = float(sorted_vals[idx])
                    if not np.all(np.isfinite(ref)):
                        ref_mode = "tile"
                        ref = meds[dominant_idx]
                    ref = np.asarray(ref, dtype=np.float32, order="C")

                    gain_clip_cfg = (
                        inter_cfg.get("two_pass_cov_gain_clip")
                        or stack_cfg.get("two_pass_cov_gain_clip")
                    )
                    gmin, gmax = 0.85, 1.18
                    if isinstance(gain_clip_cfg, (list, tuple)) and len(gain_clip_cfg) >= 2:
                        try:
                            gmin = float(gain_clip_cfg[0])
                            gmax = float(gain_clip_cfg[1])
                        except Exception:
                            gmin, gmax = 0.85, 1.18
                    if not math.isfinite(gmin) or not math.isfinite(gmax):
                        gmin, gmax = 0.85, 1.18
                    if gmin > gmax:
                        gmin, gmax = gmax, gmin

                    preview_gains = []
                    for med in meds:
                        gains_preview = np.ones_like(ref, dtype=np.float32)
                        valid_mask = (
                            np.isfinite(med)
                            & np.isfinite(ref)
                            & (med != 0)
                            & (ref != 0)
                        )
                        same_sign = np.signbit(med) == np.signbit(ref)
                        valid_mask &= same_sign
                        gains_preview[valid_mask] = ref[valid_mask] / med[valid_mask]
                        preview_gains.append(np.clip(gains_preview, gmin, gmax))
                    preview_arr = np.vstack(preview_gains)
                    gain_summary = ",".join(
                        f"ch{idx}:{float(np.nanmedian(preview_arr[:, idx])):.3f}"
                        for idx in range(preview_arr.shape[1])
                    )
                    pcb(
                        "p45_norm_stats",
                        lvl="DEBUG",
                        ref="median" if ref_mode == "median" else f"tile_{ref_tile}",
                        target=[float(val) for val in ref.tolist()],
                        gains_summary=gain_summary,
                    )

                    applied_tiles = 0
                    ref_descriptor = "weighted" if ref_mode == "median" else f"tile:{ref_tile}"
                    for tidx, tpath, med, _ in valid_stats:
                        gains = np.ones_like(ref, dtype=np.float32)
                        valid_mask = (
                            np.isfinite(med)
                            & np.isfinite(ref)
                            & (med != 0)
                            & (ref != 0)
                        )
                        same_sign = np.signbit(med) == np.signbit(ref)
                        valid_mask &= same_sign
                        gains[valid_mask] = ref[valid_mask] / med[valid_mask]
                        gains = np.clip(gains, gmin, gmax)
                        if np.all(np.abs(gains - 1.0) <= 1e-6):
                            continue
                        if not tpath or not os.path.exists(tpath):
                            continue
                        try:
                            with fits.open(
                                tpath,
                                mode="update",
                                memmap=False,
                                do_not_scale_image_data=True,
                            ) as hdul:
                                raw = hdul[0].data
                                if raw is None:
                                    continue
                                arr = _ensure_hwc_master_tile(raw, os.path.basename(tpath))
                                arr_corr = np.asarray(arr, dtype=np.float32, order="C")
                                for ch in range(min(arr_corr.shape[-1], gains.size)):
                                    arr_corr[..., ch] *= float(gains[ch])
                                if raw.ndim == 2:
                                    arr_store = arr_corr[..., 0]
                                elif (
                                    raw.ndim == 3
                                    and raw.shape[0] in (1, 3)
                                    and raw.shape[-1] not in (1, 3)
                                ):
                                    arr_store = np.moveaxis(arr_corr, -1, 0)
                                elif raw.ndim == 3 and raw.shape[-1] in (1, 3):
                                    arr_store = arr_corr[..., : raw.shape[-1]]
                                else:
                                    arr_store = arr_corr
                                arr_store = np.asarray(arr_store, dtype=np.float32, order="C")
                                if hdul[0].data.shape == arr_store.shape:
                                    hdul[0].data[...] = arr_store
                                else:
                                    hdul[0].data = arr_store
                                header = hdul[0].header
                                try:
                                    header["ZM45NORM"] = (True, "Phase 4.5 inter-super normalization")
                                except Exception:
                                    pass
                                history_line = (
                                    "P4.5 intersuper norm gain="
                                    + ",".join(f"{val:.6f}" for val in gains.tolist())
                                    + ", offset=0.0, ref="
                                    + ref_descriptor
                                )
                                try:
                                    header.add_history(history_line)
                                except Exception:
                                    try:
                                        header["HISTORY"] = history_line
                                    except Exception:
                                        pass
                                hdul.flush()
                            applied_tiles += 1
                            pcb("p45_norm_apply", lvl="DEBUG", tile=int(tidx), gains=gains.tolist())
                        except Exception as exc:
                            logger.debug("[P4.5] Norm apply failed for %s: %s", tpath, exc)
                    end_payload["applied"] = applied_tiles
                else:
                    logger.debug(
                        "[P4.5] Inter-super normalization skipped: insufficient stats (%d)",
                        len(valid_stats),
                    )
                    end_payload["reason"] = "insufficient_stats"
            finally:
                pcb("p45_norm_end", lvl="INFO_DETAIL", **end_payload)
    except Exception as exc:
        logger.debug("[P4.5] Inter-super normalization skipped due to error: %s", exc)

    # ---- Inter-supertiles photometric normalization (optional) ----
    if (
        photometry_estimator_available
        and photometry_apply_available
        and photometry_intersuper
    ):
        try:
            calib_entries: list[_TileAffineSource] = []
            for idx, original in enumerate(master_tiles):
                entry: tuple[str | None, Any] | None = None
                if idx in replacements:
                    entry = replacements[idx]
                elif idx not in consumed_indices:
                    entry = original
                if not entry:
                    continue
                path, wcs_obj = entry
                if not path or wcs_obj is None:
                    continue
                calib_entries.append(_TileAffineSource(path=path, wcs=wcs_obj))
            if len(calib_entries) >= 2:
                _phase45_gui_emit(
                    "p45_global_photometry_start",
                    level="INFO_DETAIL",
                    size=len(calib_entries),
                    clip=photometry_clip_sigma,
                )
                logger.debug(
                    "[P4.5] Global photometry start: entries=%d, clip=%.2f",
                    len(calib_entries),
                    photometry_clip_sigma,
                )
                global_affine = zemosaic_align_stack.estimate_affine_photometry(
                    calib_entries,
                    robust=True,
                    clip_sigma=photometry_clip_sigma,
                    match_background=True,
                )
                corrections, nontrivial = _sanitize_affine_corrections(
                    global_affine,
                    len(calib_entries),
                )
                if corrections and nontrivial:
                    nontrivial_count = sum(
                        1
                        for gain_val, offset_val in corrections
                        if abs(gain_val - 1.0) > 1e-6 or abs(offset_val) > 1e-6
                    )
                    for entry_idx, src in enumerate(calib_entries):
                        path = src.path
                        if not path or not os.path.exists(path):
                            continue
                        gain_val, offset_val = corrections[entry_idx]
                        if abs(gain_val - 1.0) <= 1e-6 and abs(offset_val) <= 1e-6:
                            continue
                        try:
                            with fits.open(
                                path,
                                mode="update",
                                memmap=True,
                                do_not_scale_image_data=True,
                            ) as hdul:
                                raw_data = hdul[0].data
                                if raw_data is None:
                                    continue
                                arr_hwc = _ensure_hwc_master_tile(
                                    raw_data,
                                    os.path.basename(path),
                                )
                                arr_corr = (
                                    arr_hwc * np.float32(gain_val) + np.float32(offset_val)
                                ).astype(np.float32, copy=False)
                                if raw_data.ndim == 2:
                                    arr_store = arr_corr[..., 0]
                                elif (
                                    raw_data.ndim == 3
                                    and raw_data.shape[0] in (1, 3)
                                    and raw_data.shape[-1] not in (1, 3)
                                ):
                                    arr_store = np.moveaxis(arr_corr, -1, 0)
                                elif raw_data.ndim == 3 and raw_data.shape[-1] in (1, 3):
                                    arr_store = arr_corr[..., : raw_data.shape[-1]]
                                else:
                                    arr_store = arr_corr
                                arr_store = np.asarray(arr_store, dtype=np.float32, order="C")
                                if hdul[0].data.shape == arr_store.shape:
                                    hdul[0].data[...] = arr_store
                                else:
                                    hdul[0].data = arr_store
                                header = hdul[0].header
                                try:
                                    header.add_history("Phase 4.5 inter-super photometric match applied")
                                except Exception:
                                    try:
                                        header["HISTORY"] = "Phase 4.5 inter-super photometric match applied"
                                    except Exception:
                                        pass
                                hdul.flush()
                        except Exception as exc:
                            logger.debug(
                                "[P4.5] Global photometry apply failed for %s: %s",
                                os.path.basename(path) if path else "<memory>",
                                exc,
                            )
                    _phase45_gui_emit(
                        "p45_global_photometry_done",
                        level="INFO_DETAIL",
                        nontrivial=nontrivial_count,
                        size=len(corrections),
                    )
                    logger.debug(
                        "[P4.5] Global photometry applied to %d/%d entries",
                        nontrivial_count,
                        len(corrections),
                    )
                else:
                    _phase45_gui_emit(
                        "p45_global_photometry_done",
                        level="INFO_DETAIL",
                        nontrivial=0,
                        size=len(calib_entries),
                    )
                    logger.debug("[P4.5] Global photometry produced no corrections")
            else:
                logger.debug(
                    "[P4.5] Global photometry skipped (insufficient entries: %d)",
                    len(calib_entries),
                )
        except Exception:
            logger.debug("[P4.5] Inter-super photometry skipped (error).", exc_info=True)

    new_master_tiles: list[tuple[str | None, Any]] = []
    for idx, original in enumerate(master_tiles):
        if idx in replacements:
            new_master_tiles.append(replacements[idx])
        elif idx not in consumed_indices:
            new_master_tiles.append(original)

    logger.debug(
        "[P4.5] Finished Inter-Master merge: tiles_in=%d, tiles_out=%d, replacements=%d",
        len(master_tiles),
        len(new_master_tiles),
        len(replacements),
    )
    pcb(
        "p45_finished",
        prog=None,
        lvl="INFO",
        tiles_in=len(master_tiles),
        tiles_out=len(new_master_tiles),
    )
    return new_master_tiles if new_master_tiles else master_tiles


def _select_quality_anchor(
    ordered_ids: list[int],
    distances: dict[int, float],
    seestar_groups: list[list[dict]],
    anchor_settings: dict,
    center_settings: dict,
    quality_crop_settings: dict | None,
    progress_callback: Callable | None = None,
) -> int | None:
    """Pick the best-quality anchor among the most central tiles."""

    if not ordered_ids or not seestar_groups:
        return None
    if not (
        ZEMOSAIC_UTILS_AVAILABLE
        and hasattr(zemosaic_utils, "create_downscaled_luminance_preview")
        and hasattr(zemosaic_utils, "compute_sky_statistics")
    ):
        return None

    try:
        probe_limit = int(anchor_settings.get("probe_limit", 12))
    except Exception:
        probe_limit = 12
    probe_limit = max(1, probe_limit)

    span_cfg = anchor_settings.get("span_range", (0.02, 6.0))
    if not (isinstance(span_cfg, (list, tuple)) and len(span_cfg) >= 2):
        span_cfg = (0.02, 6.0)
    try:
        span_low = float(span_cfg[0])
    except Exception:
        span_low = 0.02
    try:
        span_high = float(span_cfg[1])
    except Exception:
        span_high = 6.0
    if span_low > span_high:
        span_low, span_high = span_high, span_low

    try:
        median_clip_sigma = float(anchor_settings.get("median_clip_sigma", 2.5))
    except Exception:
        median_clip_sigma = 2.5

    try:
        preview_size = int(center_settings.get("preview_size", 256))
    except Exception:
        preview_size = 256
    sky_percent = center_settings.get("sky_percentile", (25.0, 60.0))
    if not (isinstance(sky_percent, (list, tuple)) and len(sky_percent) >= 2):
        sky_percent = (25.0, 60.0)
    try:
        sky_low = float(sky_percent[0])
    except Exception:
        sky_low = 25.0
    try:
        sky_high = float(sky_percent[1])
    except Exception:
        sky_high = 60.0

    id_to_group_index = {
        ordered_ids[idx]: idx
        for idx in range(min(len(ordered_ids), len(seestar_groups)))
    }
    candidate_ids: list[int] = []
    for tid in ordered_ids:
        if tid in id_to_group_index:
            candidate_ids.append(int(tid))
        if len(candidate_ids) >= probe_limit:
            break
    if not candidate_ids:
        return None

    _log_and_callback(
        "center_anchor_probe_start",
        lvl="INFO",
        callback=progress_callback,
        probe_count=len(candidate_ids),
    )

    candidate_entries: list[dict[str, Any]] = []
    medians: list[float] = []

    for tile_id in candidate_ids:
        stats: dict[str, float] | None = None
        group_idx = id_to_group_index.get(tile_id)
        preview_arr = None
        tile_array = None
        loaded_tile_ref = None
        if group_idx is not None and 0 <= group_idx < len(seestar_groups):
            group_info = seestar_groups[group_idx]
        else:
            group_info = None
        try:
            reference_entry = group_info[0] if group_info else None
            if reference_entry is None:
                raise ValueError("empty_group")
            wcs_obj = reference_entry.get("wcs")
            cache_path = reference_entry.get("path_preprocessed_cache")
            if cache_path and os.path.exists(cache_path):
                try:
                    tile_array = np.load(cache_path, mmap_mode="r")
                except Exception:
                    tile_array = None
            if tile_array is None and reference_entry.get("preprocessed_data") is not None:
                tile_array = reference_entry.get("preprocessed_data")
            if tile_array is None and reference_entry.get("img_data_processed") is not None:
                tile_array = reference_entry.get("img_data_processed")
            if tile_array is None:
                raw_path = reference_entry.get("path_raw") or reference_entry.get("path")
                if (
                    raw_path
                    and os.path.exists(raw_path)
                    and hasattr(zemosaic_utils, "load_and_validate_fits")
                ):
                    try:
                        res = zemosaic_utils.load_and_validate_fits(
                            raw_path,
                            normalize_to_float32=False,
                            attempt_fix_nonfinite=True,
                            progress_callback=None,
                        )
                        tile_array = res[0] if isinstance(res, (tuple, list)) and res else res
                    except Exception:
                        tile_array = None
            if tile_array is None:
                raise ValueError("no_cache")
            loaded_tile_ref = tile_array
            tile_array = _ensure_hwc_master_tile(tile_array, f"anchor_probe#{tile_id}")
            tile_array = _apply_preview_quality_crop(tile_array, quality_crop_settings)
            tile_array = np.asarray(tile_array, dtype=np.float32, order="C")
            preview_arr, _ = zemosaic_utils.create_downscaled_luminance_preview(
                tile_array,
                wcs_obj,
                preview_size,
            )
            stats = zemosaic_utils.compute_sky_statistics(preview_arr, sky_low, sky_high)
            if stats is not None and preview_arr is not None:
                valid = np.asarray(preview_arr, dtype=np.float64)
                valid = valid[np.isfinite(valid)]
                if valid.size > 0:
                    base_median = float(np.median(valid))
                    stats.setdefault("median", base_median)
                    span_val = float(stats.get("high", base_median) - stats.get("low", base_median))
                    stats["span"] = span_val
                    mad = float(np.median(np.abs(valid - base_median)))
                    stats["robust_sigma"] = float(1.4826 * mad) if mad > 0 else 0.0
                    medians.append(float(stats.get("median", base_median)))
                else:
                    stats.setdefault("median", 0.0)
                    stats["span"] = float(stats.get("high", 0.0) - stats.get("low", 0.0))
                    stats["robust_sigma"] = 0.0
            else:
                stats = None
        except Exception as exc:
            logger.debug("Anchor candidate %s preview failed: %s", tile_id, exc, exc_info=True)
            stats = None
        finally:
            if preview_arr is not None:
                del preview_arr
            if loaded_tile_ref is not None and isinstance(loaded_tile_ref, np.memmap):
                mmap_obj = getattr(loaded_tile_ref, "_mmap", None)
                if mmap_obj is not None:
                    try:
                        mmap_obj.close()
                    except Exception:
                        pass
            tile_array = None

        candidate_entries.append(
            {
                "tile_id": tile_id,
                "stats": stats,
                "distance": float(distances.get(tile_id, float("nan"))),
            }
        )

    if not medians:
        return None

    medians_arr = np.asarray(medians, dtype=np.float64)
    medians_arr = medians_arr[np.isfinite(medians_arr)]
    if medians_arr.size == 0:
        return None
    group_median = float(np.median(medians_arr))
    mad = float(np.median(np.abs(medians_arr - group_median)))
    deviation_clip = None
    if mad > 0 and math.isfinite(mad):
        deviation_clip = float(max(0.0, median_clip_sigma) * 1.4826 * mad)

    best_entry: dict[str, Any] | None = None
    best_score = float("inf")
    best_entry_soft: dict[str, Any] | None = None
    best_score_soft = float("inf")

    for entry in candidate_entries:
        stats = entry.get("stats")
        accepted = True
        score = float("inf")
        median_val = float("nan")
        span_val = float("nan")
        raw_score = float("inf")
        if not stats:
            accepted = False
        else:
            try:
                median_val = float(stats.get("median", float("nan")))
            except Exception:
                median_val = float("nan")
            try:
                span_val = float(stats.get("span"))
            except Exception:
                try:
                    span_val = float(stats.get("high", 0.0) - stats.get("low", 0.0))
                except Exception:
                    span_val = float("nan")
            try:
                robust_sigma_val = float(stats.get("robust_sigma", 0.0))
            except Exception:
                robust_sigma_val = 0.0
            try:
                raw_score = _score_anchor_candidate(stats, group_median, deviation_clip)
            except Exception:
                raw_score = float("inf")
            if not (math.isfinite(median_val) and math.isfinite(span_val)):
                accepted = False
            if accepted and not math.isfinite(robust_sigma_val):
                stats["robust_sigma"] = 0.0
                robust_sigma_val = 0.0
            if accepted:
                if span_val < span_low or span_val > span_high:
                    accepted = False
            if accepted and deviation_clip is not None and deviation_clip > 0:
                if abs(median_val - group_median) > deviation_clip:
                    accepted = False
            if accepted:
                score = raw_score
                if score < best_score:
                    best_score = score
                    best_entry = entry
        if math.isfinite(raw_score) and raw_score < best_score_soft:
            best_score_soft = raw_score
            best_entry_soft = entry
        entry["accepted"] = accepted
        entry["score"] = raw_score
        _log_and_callback(
            "center_anchor_probe_candidate",
            lvl="DEBUG_DETAIL",
            callback=progress_callback,
            tile=int(entry["tile_id"]),
            dist_deg=entry.get("distance"),
            median=median_val,
            span=span_val,
            score=score,
            accepted="accepted" if accepted else "rejected",
        )

    if best_entry is None:
        if best_entry_soft is not None:
            best_entry = best_entry_soft
            best_score = best_score_soft
            _log_and_callback(
                "center_anchor_soft_fallback",
                lvl="WARN",
                callback=progress_callback,
            )
        else:
            return None

    _log_and_callback(
        "center_anchor_selected",
        lvl="INFO",
        callback=progress_callback,
        tile=int(best_entry["tile_id"]),
        dist_deg=best_entry.get("distance"),
        score=best_score,
    )

    return int(best_entry["tile_id"])


def _compute_intertile_affine_corrections_from_sources(
    sources: list[_TileAffineSource],
    final_output_wcs,
    final_output_shape_hw: tuple[int, int],
    preview_size: int,
    min_overlap_fraction: float,
    sky_percentile: tuple[float, float] | list[float],
    robust_clip_sigma: float,
    use_auto_intertile: bool,
    logger_obj=None,
    progress_callback: Callable | None = None,
    intertile_global_recenter: bool = False,
    intertile_recenter_clip: tuple[float, float] | list[float] | None = None,
) -> tuple[list[tuple[float, float]] | None, bool, str, str | None]:
    """Common implementation for intertile gain/offset computation.

    Returns
    -------
    (list[(gain, offset)] | None, bool, str, str | None)
        Sanitized affine list (or ``None``), flag indicating whether non-trivial
        corrections were detected, status string (``\"ok\"``, ``\"skipped\"``,
        ``\"preview_failed\"``, ``\"compute_failed\"``) and optional error message.
    """

    total_tiles = len(sources)
    if total_tiles < 2:
        return None, False, "skipped", None

    if not (
        ZEMOSAIC_UTILS_AVAILABLE
        and hasattr(zemosaic_utils, "compute_intertile_affine_calibration")
    ):
        return None, False, "skipped", None

    tile_pairs: list[tuple[np.ndarray, Any]] = []
    preview_arrays: list[np.ndarray | None] = []

    for idx, src in enumerate(sources, 1):
        try:
            tile_arr: np.ndarray
            label = os.path.basename(src.path) if src.path else None
            if src.data is not None:
                tile_arr = _ensure_hwc_master_tile(src.data, label)
            else:
                if not src.path:
                    raise ValueError("Tile data missing and no path provided.")
                with fits.open(src.path, memmap=False) as hdul:
                    tile_arr = _ensure_hwc_master_tile(hdul[0].data, label)
            tile_arr = np.asarray(tile_arr, dtype=np.float32, order="C")
            tile_pairs.append((tile_arr, src.wcs))
            preview_entry = None
            if intertile_global_recenter:
                try:
                    preview_entry, _ = zemosaic_utils.create_downscaled_luminance_preview(
                        tile_arr,
                        src.wcs,
                        preview_size,
                    )
                    if preview_entry is not None:
                        preview_entry = np.asarray(preview_entry, dtype=np.float32)
                except Exception:
                    preview_entry = None
            preview_arrays.append(preview_entry)
        except Exception as exc:
            if logger_obj:
                logger_obj.warning(
                    "Intertile data load failed for %s: %s",
                    src.path or f"tile#{idx}",
                    exc,
                )
                logger_obj.debug("Traceback (intertile data load):", exc_info=True)
            tile_pairs.clear()
            preview_arrays.clear()
            return None, False, "preview_failed", str(exc)
        finally:
            if progress_callback:
                try:
                    progress_callback("phase5_intertile", idx, total_tiles)
                except Exception:
                    pass

    # [ETA Bridge] Wrapper pour intercepter "[Intertile] Using: ... pairs=K"
    def _intertile_progress_bridge(message_or_stage, current, level, **kwargs):
        try:
            # Seed ETA dès que l'on connaît le nombre de paires et la taille de preview
            if isinstance(message_or_stage, str) and message_or_stage.startswith("[Intertile] Using:"):
                # Ex: "[Intertile] Using: preview=512, min_overlap=0.0500, sky=(30.0,70.0), clip=2.50, pairs=353"
                try:
                    parts = message_or_stage.split("Using:", 1)[1]
                    # petite extraction clé=valeur robuste
                    kv = {}
                    for p in parts.split(","):
                        if "=" in p:
                            k, v = p.split("=", 1)
                            kv[k.strip()] = v.strip().strip(",")
                    preview = int(kv.get("preview", "512"))
                    pairs = int(kv.get("pairs", "0"))
                    # Heuristique : coût ~ C * pairs * (preview/512)^2
                    # GPU/CPU : facteur 1.0 (GPU) / 3.0 (CPU) basique (peut être affiné)
                    gpu_factor = 1.0 if bool(getattr(zconfig, "use_gpu_phase5", False)) else 3.0
                    base_cost_per_pair = 0.020  # 20 ms/pair @ preview=512 sur GPU (à ajuster)
                    seconds = gpu_factor * base_cost_per_pair * pairs * (preview / 512.0) ** 2
                    h, rem = divmod(int(seconds), 3600)
                    m, s = divmod(rem, 60)
                    # Pousser une ETA initiale vers le GUI (il se recadrera ensuite avec les ticks "pairs")
                    _log_and_callback(f"ETA_UPDATE:{h:02d}:{m:02d}:{s:02d}", None, "ETA_LEVEL", callback=progress_callback)
                except Exception:
                    pass
        except Exception:
            pass
        # Relais vers le callback GUI d'origine pour l'affichage + ETA fine (via on_worker_progress)
        if progress_callback:
            try:
                progress_callback(message_or_stage, current, level, **kwargs)
            except Exception:
                pass

    try:
        corrections = zemosaic_utils.compute_intertile_affine_calibration(

            tile_pairs,
            final_output_wcs,
            final_output_shape_hw,
            preview_size=preview_size,
            min_overlap_fraction=min_overlap_fraction,
            sky_percentile=sky_percentile,
            robust_clip_sigma=robust_clip_sigma,
            use_auto_intertile=use_auto_intertile,
            logger=logger_obj,
            progress_callback=_intertile_progress_bridge,
        )
    except Exception as exc:
        if logger_obj:
            logger_obj.warning(
                "Intertile photometric calibration failed: %s",
                exc,
            )
            logger_obj.debug("Traceback (intertile failure):", exc_info=True)
        return None, False, "compute_failed", str(exc)
    finally:
        tile_pairs.clear()

    sanitized, nontrivial = _sanitize_affine_corrections(corrections, total_tiles)

    if sanitized and intertile_global_recenter and preview_arrays:
        clip_cfg = intertile_recenter_clip if isinstance(intertile_recenter_clip, (list, tuple)) else None
        if not clip_cfg or len(clip_cfg) < 2:
            clip_cfg = (0.85, 1.18)
        try:
            clip_low = float(clip_cfg[0])
        except Exception:
            clip_low = 0.85
        try:
            clip_high = float(clip_cfg[1])
        except Exception:
            clip_high = 1.18
        if clip_low > clip_high:
            clip_low, clip_high = clip_high, clip_low

        try:
            medians_before: list[float] = []
            for preview_entry, affine in zip(preview_arrays, sanitized):
                if preview_entry is None:
                    medians_before.append(float("nan"))
                    continue
                gain_val, offset_val = affine
                try:
                    corrected = preview_entry * float(gain_val) + float(offset_val)
                    med_val = float(np.nanmedian(corrected)) if corrected.size > 0 else float("nan")
                except Exception:
                    med_val = float("nan")
                medians_before.append(med_val)

            finite_medians = [m for m in medians_before if math.isfinite(m) and m > 0]
            if finite_medians:
                target = float(np.median(finite_medians))
                _log_and_callback(
                    "intertile_recenter_applied",
                    lvl="INFO",
                    callback=progress_callback,
                    target=target,
                    clip_low=clip_low,
                    clip_high=clip_high,
                )
                for idx, med_val in enumerate(medians_before):
                    if not (math.isfinite(med_val) and med_val > 0):
                        continue
                    gain_adj = target / med_val if med_val != 0 else 1.0
                    if not math.isfinite(gain_adj) or gain_adj <= 0:
                        continue
                    if gain_adj < clip_low:
                        gain_adj = clip_low
                    elif gain_adj > clip_high:
                        gain_adj = clip_high
                    gain_val, offset_val = sanitized[idx]
                    sanitized[idx] = (float(gain_val) * gain_adj, float(offset_val) * gain_adj)
                    _log_and_callback(
                        "intertile_recenter_adjust",
                        lvl="INFO_DETAIL",
                        callback=progress_callback,
                        tile=int(idx),
                        median_before=med_val,
                        gain_adj=gain_adj,
                    )
        except Exception as exc:
            if logger_obj:
                logger_obj.warning("Intertile global recenter failed: %s", exc)
                logger_obj.debug("Traceback (intertile recenter):", exc_info=True)
        finally:
            preview_arrays.clear()
    elif intertile_global_recenter and preview_arrays:
        preview_arrays.clear()

    return sanitized, nontrivial, "ok", None

def cluster_seestar_stacks_connected(
    all_raw_files_with_info: list,
    stack_threshold_deg: float,
    progress_callback: callable,
    orientation_split_threshold_deg: float = 0.0,
):
    """Order-invariant clustering of Seestar raws using spherical proximity.

    Builds a proximity graph (edges when separation < threshold) and returns
    connected components. Deterministic across runs when input ordering is
    stable (we sort file paths earlier).
    """
    # Deps imported later in module; they will be available at runtime
    try:
        ok_astropy = ASTROPY_AVAILABLE and (SkyCoord is not None) and (u is not None) and (Angle is not None)
    except NameError:
        ok_astropy = False
    if not ok_astropy:
        _log_and_callback("clusterstacks_error_astropy_unavailable", level="ERROR", callback=progress_callback)
        return []
    if not all_raw_files_with_info:
        _log_and_callback("clusterstacks_warn_no_raw_info", level="WARN", callback=progress_callback)
        return []
    _log_and_callback(
        "clusterstacks_info_start",
        num_files=len(all_raw_files_with_info),
        threshold=stack_threshold_deg,
        level="INFO",
        callback=progress_callback,
    )
    panel_centers_sky = []
    panel_data_for_clustering = []
    panel_orientations_deg = []  # orientation of +X pixel axis on sky, in degrees [0,360)
    for info in all_raw_files_with_info:
        wcs_obj = info.get("wcs")
        if not (wcs_obj and getattr(wcs_obj, "is_celestial", False)):
            continue
        try:
            if getattr(wcs_obj, "pixel_shape", None):
                cx = wcs_obj.pixel_shape[0] / 2.0
                cy = wcs_obj.pixel_shape[1] / 2.0
                center_world = wcs_obj.pixel_to_world(cx, cy)
            elif hasattr(wcs_obj, "wcs") and hasattr(wcs_obj.wcs, "crval"):
                center_world = SkyCoord(
                    ra=float(wcs_obj.wcs.crval[0]) * u.deg,
                    dec=float(wcs_obj.wcs.crval[1]) * u.deg,
                    frame="icrs",
                )
            else:
                continue
            panel_centers_sky.append(center_world)
            panel_data_for_clustering.append(info)
            # Optionally compute orientation of X pixel axis using WCS
            if orientation_split_threshold_deg and float(orientation_split_threshold_deg) > 0:
                try:
                    # Use center pixel + one-pixel step in +X to get position angle
                    if getattr(wcs_obj, "pixel_shape", None):
                        cx = wcs_obj.pixel_shape[0] / 2.0
                        cy = wcs_obj.pixel_shape[1] / 2.0
                    else:
                        cx, cy = 0.0, 0.0
                    c0 = wcs_obj.pixel_to_world(cx, cy)
                    c1 = wcs_obj.pixel_to_world(cx + 1.0, cy)
                    pa = c0.position_angle(c1).to(u.deg).value  # east of north
                    ang = float(pa) % 360.0
                    panel_orientations_deg.append(ang)
                except Exception:
                    panel_orientations_deg.append(None)
            else:
                panel_orientations_deg.append(None)
        except Exception:
            continue
    if not panel_centers_sky:
        _log_and_callback("clusterstacks_warn_no_centers", level="WARN", callback=progress_callback)
        return []
    coords = SkyCoord(
        ra=[c.ra for c in panel_centers_sky],
        dec=[c.dec for c in panel_centers_sky],
        frame="icrs",
    )
    max_sep = Angle(float(stack_threshold_deg), unit=u.deg)
    try:
        idx1, idx2, _, _ = coords.search_around_sky(coords, max_sep)
    except Exception:
        idx1, idx2 = np.array([], dtype=int), np.array([], dtype=int)
    n = len(coords)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    def _circ_delta_deg(a: float, b: float) -> float:
        d = abs(float(a) - float(b))
        if d > 180.0:
            d = 360.0 - d
        return d

    for a, b in zip(idx1, idx2):
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        # If orientation-split is enabled, only connect when |Δangle| <= threshold
        if orientation_split_threshold_deg and float(orientation_split_threshold_deg) > 0:
            oa = panel_orientations_deg[ia] if ia < len(panel_orientations_deg) else None
            ob = panel_orientations_deg[ib] if ib < len(panel_orientations_deg) else None
            if oa is None or ob is None:
                # Cannot compare orientations: do not connect
                continue
            if _circ_delta_deg(oa, ob) > float(orientation_split_threshold_deg):
                continue
        union(ia, ib)
    groups_indices = {}
    for i in range(n):
        r = find(i)
        groups_indices.setdefault(r, []).append(i)
    ordered_roots = sorted(groups_indices.keys(), key=lambda r: min(groups_indices[r]))
    groups = []
    for r in ordered_roots:
        members = groups_indices[r]
        members.sort()
        groups.append([panel_data_for_clustering[i] for i in members])
    _log_and_callback("clusterstacks_info_finished", num_groups=len(groups), level="INFO", callback=progress_callback)
    return groups


# --- Phase 3 center-out helpers -------------------------------------------------


@dataclass
class _CenterPreviewEntry:
    tile_id: int
    preview: "np.ndarray | None"
    wcs: object | None
    stats: dict | None
    mode: str
    gain: float
    offset: float


class CenterOutNormalizationContext:
    def __init__(
        self,
        anchor_tile_original_id: int,
        ordered_tile_ids: list[int],
        tile_distances: dict[int, float] | None,
        settings: dict,
        global_center=None,
        logger_instance=None,
    ):
        self.anchor_original_id = int(anchor_tile_original_id)
        self.ordered_tile_ids = list(ordered_tile_ids)
        self._rank_map = {tid: idx for idx, tid in enumerate(self.ordered_tile_ids)}
        self.tile_distances = dict(tile_distances or {})
        self.settings = settings or {}
        self.global_center = global_center
        self.logger = logger_instance or logger
        self._lock = threading.RLock()
        self._entries: dict[int, _CenterPreviewEntry] = {}
        self.anchor_stats: dict | None = None
        self.anchor_ready_event = threading.Event()

    def get_rank(self, tile_id: int) -> int | None:
        return self._rank_map.get(int(tile_id))

    def get_distance(self, tile_id: int) -> float | None:
        return self.tile_distances.get(int(tile_id))

    def wait_for_anchor(self) -> bool:
        if self.anchor_ready_event.is_set():
            return True
        return self.anchor_ready_event.wait(timeout=60.0)

    def register_tile(
        self,
        tile_id: int,
        preview: "np.ndarray | None",
        preview_wcs,
        stats: dict | None,
        gain: float,
        offset: float,
        mode: str,
    ) -> None:
        entry = _CenterPreviewEntry(
            tile_id=int(tile_id),
            preview=None if preview is None else np.asarray(preview, dtype=np.float32, copy=True),
            wcs=preview_wcs,
            stats=stats.copy() if isinstance(stats, dict) else stats,
            mode=str(mode),
            gain=float(gain),
            offset=float(offset),
        )
        with self._lock:
            self._entries[int(tile_id)] = entry
            if int(tile_id) == self.anchor_original_id:
                self.anchor_stats = entry.stats
                self.anchor_ready_event.set()

    def get_processed_tiles(self, exclude_tile_id: int | None = None) -> list[_CenterPreviewEntry]:
        with self._lock:
            items = [
                entry
                for tid, entry in self._entries.items()
                if exclude_tile_id is None or int(tid) != int(exclude_tile_id)
            ]
        items.sort(key=lambda ent: (self.get_rank(ent.tile_id) if self.get_rank(ent.tile_id) is not None else 1_000_000))
        return items

    def get_anchor_stats(self) -> dict | None:
        with self._lock:
            return self.anchor_stats.copy() if isinstance(self.anchor_stats, dict) else self.anchor_stats


def _extract_group_center_skycoord(group_info_list: list[dict]) -> "SkyCoord | None":
    if not group_info_list:
        return None
    if not (ASTROPY_AVAILABLE and SkyCoord and u):
        return None
    for entry in group_info_list:
        wcs_obj = entry.get("wcs")
        if wcs_obj and getattr(wcs_obj, "is_celestial", False):
            try:
                if getattr(wcs_obj, "pixel_shape", None):
                    width = float(wcs_obj.pixel_shape[0])
                    height = float(wcs_obj.pixel_shape[1])
                else:
                    shape = entry.get("preprocessed_shape")
                    if shape and len(shape) >= 2:
                        height = float(shape[0])
                        width = float(shape[1])
                    else:
                        height = width = 0.0
                cx = width / 2.0
                cy = height / 2.0
                center_world = wcs_obj.pixel_to_world(cx, cy)
                if isinstance(center_world, SkyCoord):
                    return center_world
            except Exception:
                pass
            try:
                if hasattr(wcs_obj, "wcs") and wcs_obj.wcs and wcs_obj.wcs.crval is not None:
                    ra = float(wcs_obj.wcs.crval[0])
                    dec = float(wcs_obj.wcs.crval[1])
                    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            except Exception:
                pass
    for entry in group_info_list:
        header = entry.get("header")
        if header is None:
            continue
        try:
            if hasattr(header, "get"):
                ra_val = header.get("CRVAL1")
                dec_val = header.get("CRVAL2")
            else:
                ra_val = header["CRVAL1"]
                dec_val = header["CRVAL2"]
            if ra_val is None or dec_val is None:
                continue
            return SkyCoord(ra=float(ra_val) * u.deg, dec=float(dec_val) * u.deg, frame="icrs")
        except Exception:
            continue
    return None


def _compute_center_out_order(
    seestar_stack_groups: list[list[dict]],
) -> tuple[list[int], "SkyCoord", dict[int, float]] | None:
    if not seestar_stack_groups:
        return None
    if not (ASTROPY_AVAILABLE and SkyCoord and u):
        return None
    centers = []
    fallback = []
    for idx, group in enumerate(seestar_stack_groups):
        center = _extract_group_center_skycoord(group)
        if center:
            centers.append((idx, center))
        else:
            fallback.append(idx)
    if not centers:
        return None
    arr = np.array([coord.cartesian.xyz.value for _, coord in centers], dtype=np.float64)
    norm = np.linalg.norm(arr, axis=1)
    norm[norm == 0] = 1.0
    arr = arr / norm[:, None]
    vec = arr.mean(axis=0)
    if np.linalg.norm(vec) <= 0:
        return None
    vec_norm = vec / np.linalg.norm(vec)
    spherical = SkyCoord(
        x=vec_norm[0] * u.one,
        y=vec_norm[1] * u.one,
        z=vec_norm[2] * u.one,
        frame="icrs",
        representation_type="cartesian",
    ).spherical
    global_center = SkyCoord(ra=spherical.lon.to(u.deg), dec=spherical.lat.to(u.deg), frame="icrs")
    distances: dict[int, float] = {}
    for idx, coord in centers:
        try:
            distances[idx] = float(coord.separation(global_center).deg)
        except Exception:
            distances[idx] = float("nan")
    ordered = sorted(
        [idx for idx, _ in centers],
        key=lambda tid: (distances.get(tid, float("inf")), tid),
    )
    for idx in fallback:
        if idx not in ordered:
            ordered.append(idx)
    return ordered, global_center, distances


def _compute_master_quality_metrics(
    master_arr: np.ndarray,
    sky_low_pct: float = 25.0,
    sky_high_pct: float = 60.0,
) -> dict[str, float]:
    """Estimate robust quality statistics for a master tile array.

    Parameters
    ----------
    master_arr : np.ndarray
        Master tile data in ``H x W x C`` or ``H x W`` layout.
    sky_low_pct : float, optional
        Lower percentile for sky span estimation.
    sky_high_pct : float, optional
        Upper percentile for sky span estimation.

    Returns
    -------
    dict[str, float]
        Dictionary containing ``median``, ``span``, ``robust_sigma``,
        ``grad_proxy`` and aggregated ``score`` (lower is better).
    """

    if master_arr is None:
        return {
            "median": 0.0,
            "span": 0.0,
            "robust_sigma": 0.0,
            "grad_proxy": 0.0,
            "score": float("inf"),
        }

    lum = master_arr if master_arr.ndim == 2 else np.mean(master_arr, axis=2, dtype=np.float32)
    lum = np.asarray(lum, dtype=np.float32)

    if lum.size == 0:
        return {
            "median": 0.0,
            "span": 0.0,
            "robust_sigma": 0.0,
            "grad_proxy": 0.0,
            "score": float("inf"),
        }

    finite_mask = np.isfinite(lum)
    valid = lum[finite_mask]
    if valid.size == 0:
        median_val = 0.0
        span_val = 0.0
        robust_sigma = 0.0
        lum_clean = np.zeros_like(lum, dtype=np.float32)
    else:
        median_val = float(np.median(valid))
        try:
            lo_val = float(np.percentile(valid, float(sky_low_pct)))
        except Exception:
            lo_val = float(np.percentile(valid, 25.0))
        try:
            hi_val = float(np.percentile(valid, float(sky_high_pct)))
        except Exception:
            hi_val = float(np.percentile(valid, 60.0))
        span_val = max(hi_val - lo_val, 0.0)
        abs_dev = np.abs(valid - median_val)
        robust_sigma = float(1.4826 * np.median(abs_dev)) if abs_dev.size else 0.0
        lum_clean = np.array(lum, copy=True)
        lum_clean[~finite_mask] = median_val

    if not np.isfinite(median_val):
        median_val = 0.0
    if not np.isfinite(span_val):
        span_val = 0.0
    if not np.isfinite(robust_sigma):
        robust_sigma = 0.0

    try:
        from scipy.ndimage import gaussian_filter  # type: ignore

        blurred = gaussian_filter(lum_clean, sigma=3.0, mode="nearest")
    except Exception:
        sigma = 3.0
        radius = max(1, int(round(sigma * 3.0)))
        coords = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (coords ** 2) / float(sigma ** 2))
        kernel_sum = float(kernel.sum())
        if kernel_sum > 0:
            kernel /= kernel_sum
        blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=lum_clean)
        blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=blurred)

    diff = lum_clean - blurred
    grad_proxy = float(np.nanstd(diff)) if diff.size else 0.0
    if not np.isfinite(grad_proxy):
        grad_proxy = 0.0

    eps = 1e-6
    score = (span_val + eps) * 0.7 + robust_sigma * 0.25 + grad_proxy * 0.05

    return {
        "median": float(median_val),
        "span": float(span_val),
        "robust_sigma": float(robust_sigma),
        "grad_proxy": float(grad_proxy),
        "score": float(score),
    }


def _compute_tile_distance_map(
    master_tiles: dict[int, tuple[str | None, Any]]
) -> dict[int, float]:
    """Estimate angular distances of master tiles from the average sky center."""

    if not master_tiles:
        return {}
    if not (ASTROPY_AVAILABLE and SkyCoord and u):
        return {}

    centers: list[tuple[int, "SkyCoord"]] = []
    for tile_id, (_path, wcs_obj) in master_tiles.items():
        if not (wcs_obj and getattr(wcs_obj, "is_celestial", False)):
            continue
        try:
            if getattr(wcs_obj, "pixel_shape", None):
                width = float(wcs_obj.pixel_shape[0])
                height = float(wcs_obj.pixel_shape[1])
            else:
                width = float(getattr(wcs_obj.wcs, "naxis1", 0))
                height = float(getattr(wcs_obj.wcs, "naxis2", 0))
            if width <= 0 or height <= 0:
                continue
            cx = width / 2.0
            cy = height / 2.0
            center_coord = wcs_obj.pixel_to_world(cx, cy)
            if isinstance(center_coord, SkyCoord):
                centers.append((int(tile_id), center_coord))
        except Exception:
            continue

    if not centers:
        return {}

    arr = np.array([coord.cartesian.xyz.value for _, coord in centers], dtype=np.float64)
    norm = np.linalg.norm(arr, axis=1)
    norm[norm == 0] = 1.0
    arr = arr / norm[:, None]
    vec = arr.mean(axis=0)
    if np.linalg.norm(vec) <= 0:
        return {}
    vec_norm = vec / np.linalg.norm(vec)
    global_center = SkyCoord(
        x=vec_norm[0] * u.one,
        y=vec_norm[1] * u.one,
        z=vec_norm[2] * u.one,
        frame="icrs",
        representation_type="cartesian",
    ).spherical
    center_sc = SkyCoord(ra=global_center.lon.to(u.deg), dec=global_center.lat.to(u.deg), frame="icrs")

    distances: dict[int, float] = {}
    for tile_id, coord in centers:
        try:
            distances[int(tile_id)] = float(coord.separation(center_sc).deg)
        except Exception:
            continue
    return distances


def apply_center_out_normalization_p3(
    tile_array: "np.ndarray",
    tile_wcs,
    tile_id: int,
    context: CenterOutNormalizationContext | None,
    settings: dict | None,
    log_func: Callable | None = None,
) -> tuple["np.ndarray", tuple[float, float] | None, str, dict]:
    if tile_array is None or context is None or not settings or not settings.get("enabled", True):
        return tile_array, None, "disabled", {}

    preview_size = int(settings.get("preview_size", 256))
    sky_percent = settings.get("sky_percentile", (25.0, 60.0))
    if not (isinstance(sky_percent, (tuple, list)) and len(sky_percent) >= 2):
        sky_percent = (25.0, 60.0)
    sky_low, sky_high = float(sky_percent[0]), float(sky_percent[1])
    clip_sigma = float(settings.get("clip_sigma", 2.5))
    min_overlap = float(settings.get("min_overlap_fraction", 0.03))

    preview_raw, preview_wcs = zemosaic_utils.create_downscaled_luminance_preview(
        tile_array,
        tile_wcs,
        preview_size,
    )
    tile_stats_raw = zemosaic_utils.compute_sky_statistics(preview_raw, sky_low, sky_high)

    if int(tile_id) == context.anchor_original_id:
        context.register_tile(tile_id, preview_raw, preview_wcs, tile_stats_raw, 1.0, 0.0, "anchor")
        details = {
            "rank": context.get_rank(tile_id),
            "distance": context.get_distance(tile_id),
            "mode": "anchor",
            "samples": int(preview_raw.size if preview_raw is not None else 0),
        }
        return tile_array, (1.0, 0.0), "anchor", details

    context.wait_for_anchor()
    best_gain_offset: tuple[float, float] | None = None
    best_samples = 0
    best_reference = None

    if preview_raw is not None and preview_wcs is not None and reproject_interp:
        for entry in context.get_processed_tiles(exclude_tile_id=tile_id):
            if entry.preview is None or entry.wcs is None:
                continue
            try:
                reproj_src, footprint = reproject_interp(
                    (preview_raw, preview_wcs),
                    entry.wcs,
                    shape_out=entry.preview.shape,
                )
            except Exception:
                continue
            if reproj_src is None or footprint is None:
                continue
            valid = np.isfinite(reproj_src) & np.isfinite(entry.preview) & (footprint > 0.1)
            if not np.any(valid):
                continue
            overlap_fraction = valid.sum() / max(1, entry.preview.size)
            if overlap_fraction < min_overlap:
                continue
            fit = zemosaic_utils.estimate_sky_affine_to_ref(
                reproj_src[valid],
                entry.preview[valid],
                sky_low,
                sky_high,
                clip_sigma,
            )
            if not fit:
                continue
            gain, offset, samples = fit
            if not np.isfinite(gain) or not np.isfinite(offset):
                continue
            if samples > best_samples:
                best_samples = samples
                best_gain_offset = (gain, offset)
                best_reference = entry.tile_id

    mode = "anchor_fallback"
    if best_gain_offset is None:
        anchor_stats = context.get_anchor_stats()
        gain = 1.0
        offset = 0.0
        if anchor_stats and tile_stats_raw:
            anchor_span = float(anchor_stats.get("high", 0.0)) - float(anchor_stats.get("low", 0.0))
            tile_span = float(tile_stats_raw.get("high", 0.0)) - float(tile_stats_raw.get("low", 0.0))
            if np.isfinite(anchor_span) and np.isfinite(tile_span) and tile_span > 1e-6:
                gain = anchor_span / tile_span
            anchor_med = float(anchor_stats.get("median", 0.0))
            tile_med = float(tile_stats_raw.get("median", 0.0))
            offset = anchor_med - gain * tile_med
        best_gain_offset = (gain, offset)
        best_samples = int(tile_stats_raw.get("median", 0) if tile_stats_raw else 0)
    else:
        mode = "overlap"

    gain, offset = best_gain_offset
    if not np.isfinite(gain) or not np.isfinite(offset):
        return tile_array, None, "invalid", {}

    try:
        np.multiply(tile_array, gain, out=tile_array, casting="unsafe")
        np.add(tile_array, offset, out=tile_array, casting="unsafe")
    except Exception:
        tile_array = tile_array * gain + offset

    preview_corrected = None
    if preview_raw is not None:
        preview_corrected = preview_raw * gain + offset
    corrected_stats = zemosaic_utils.compute_sky_statistics(preview_corrected, sky_low, sky_high)
    context.register_tile(tile_id, preview_corrected, preview_wcs, corrected_stats, gain, offset, mode)

    details = {
        "rank": context.get_rank(tile_id),
        "distance": context.get_distance(tile_id),
        "mode": mode if best_reference is None else f"{mode}:{best_reference}",
        "samples": int(best_samples),
        "reference": best_reference,
        "gain": float(gain),
        "offset": float(offset),
    }
    if log_func:
        try:
            log_func(
                "center_out_debug",
                prog=None,
                lvl="DEBUG_DETAIL",
                tile=int(tile_id),
                rank=details.get("rank"),
                distance=f"{details.get('distance'):.4f}" if isinstance(details.get("distance"), float) else None,
                mode=details.get("mode"),
                gain=f"{gain:.6f}",
                offset=f"{offset:.6f}",
                samples=int(best_samples),
            )
        except Exception:
            pass
    return tile_array, (gain, offset), mode, details


def run_poststack_anchor_review(
    master_tiles: dict[int, tuple[str | None, Any]],
    prestack_anchor_id: int | None,
    cfg: dict | SimpleNamespace | None,
    progress_callback: Callable | None = None,
    tile_distances: dict[int, float] | None = None,
) -> tuple[dict[int, tuple[str | None, Any]], tuple[float, float]]:
    """Evaluate master tiles post-stack and optionally re-anchor globally."""

    anchor_shift = (1.0, 0.0)
    if not master_tiles:
        return master_tiles, anchor_shift

    def _cfg_get(key: str, default: Any) -> Any:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        if isinstance(cfg, SimpleNamespace) and hasattr(cfg, key):
            return getattr(cfg, key)
        return default

    try:
        probe_limit = int(_cfg_get("probe_limit", _cfg_get("poststack_anchor_probe_limit", 8)))
    except Exception:
        probe_limit = 8
    probe_limit = max(1, probe_limit)

    span_cfg = _cfg_get("span_range", _cfg_get("poststack_anchor_span_range", (0.004, 10.0)))
    if not (isinstance(span_cfg, (list, tuple)) and len(span_cfg) >= 2):
        span_cfg = (0.004, 10.0)
    try:
        span_low = float(span_cfg[0])
    except Exception:
        span_low = 0.004
    try:
        span_high = float(span_cfg[1])
    except Exception:
        span_high = 10.0
    if span_low > span_high:
        span_low, span_high = span_high, span_low

    try:
        median_clip_sigma = float(_cfg_get("median_clip_sigma", _cfg_get("poststack_anchor_median_clip_sigma", 3.5)))
    except Exception:
        median_clip_sigma = 3.5

    try:
        min_improvement = float(_cfg_get("min_improvement", _cfg_get("poststack_anchor_min_improvement", 0.12)))
    except Exception:
        min_improvement = 0.12
    if not math.isfinite(min_improvement) or min_improvement < 0:
        min_improvement = 0.0

    use_overlap_affine = bool(_cfg_get("use_overlap_affine", _cfg_get("poststack_anchor_use_overlap_affine", True)))

    sky_percent_cfg = _cfg_get("sky_percentile", _cfg_get("p3_center_sky_percentile", (25.0, 60.0)))
    if not (isinstance(sky_percent_cfg, (list, tuple)) and len(sky_percent_cfg) >= 2):
        sky_percent_cfg = (25.0, 60.0)
    try:
        sky_low_pct = float(sky_percent_cfg[0])
    except Exception:
        sky_low_pct = 25.0
    try:
        sky_high_pct = float(sky_percent_cfg[1])
    except Exception:
        sky_high_pct = 60.0
    if sky_low_pct > sky_high_pct:
        sky_low_pct, sky_high_pct = sky_high_pct, sky_low_pct

    _log_and_callback("post_anchor_start", lvl="INFO", callback=progress_callback)

    distances_map: dict[int, float] = {}
    if isinstance(tile_distances, dict):
        for key, value in tile_distances.items():
            try:
                distances_map[int(key)] = float(value)
            except Exception:
                continue
    computed_distances = _compute_tile_distance_map(master_tiles)
    for key, value in computed_distances.items():
        if key not in distances_map and math.isfinite(value):
            distances_map[key] = float(value)

    available_ids = list(master_tiles.keys())
    if not available_ids:
        return master_tiles, anchor_shift

    def _distance_key(tile_id: int) -> tuple[float, int]:
        dist = distances_map.get(int(tile_id), float("inf"))
        return (float(dist), int(tile_id))

    ordered_ids = sorted(available_ids, key=_distance_key)
    candidate_ids: list[int] = ordered_ids[:probe_limit]
    if prestack_anchor_id is not None and prestack_anchor_id in master_tiles and prestack_anchor_id not in candidate_ids:
        candidate_ids.append(int(prestack_anchor_id))

    candidates: list[dict[str, Any]] = []
    medians: list[float] = []

    for tile_id in candidate_ids:
        path, wcs_obj = master_tiles.get(int(tile_id), (None, None))
        arr: np.ndarray | None = None
        metrics: dict[str, float] | None = None
        try:
            if path and os.path.exists(path):
                with fits.open(path, memmap=False, do_not_scale_image_data=True) as hdul_mt:
                    arr = _ensure_hwc_master_tile(hdul_mt[0].data, os.path.basename(path))
            if arr is not None:
                arr = np.asarray(arr, dtype=np.float32)
                metrics = _compute_master_quality_metrics(arr, sky_low_pct=sky_low_pct, sky_high_pct=sky_high_pct)
                if metrics and math.isfinite(metrics.get("median", float("nan"))):
                    medians.append(float(metrics.get("median", 0.0)))
        except Exception:
            arr = None
            metrics = None

        candidates.append(
            {
                "tile_id": int(tile_id),
                "path": path,
                "wcs": wcs_obj,
                "metrics": metrics,
                "array": arr,
                "distance": distances_map.get(int(tile_id)),
            }
        )

    if not medians:
        for entry in candidates:
            entry.pop("array", None)
        return master_tiles, anchor_shift

    medians_arr = np.asarray(medians, dtype=np.float64)
    medians_arr = medians_arr[np.isfinite(medians_arr)]
    if medians_arr.size == 0:
        for entry in candidates:
            entry.pop("array", None)
        return master_tiles, anchor_shift

    group_median = float(np.median(medians_arr))
    mad = float(np.median(np.abs(medians_arr - group_median)))
    deviation_clip = None
    if mad > 0 and math.isfinite(mad):
        deviation_clip = float(max(0.0, median_clip_sigma) * 1.4826 * mad)

    best_entry: dict[str, Any] | None = None
    best_score = float("inf")
    best_entry_soft: dict[str, Any] | None = None
    best_score_soft = float("inf")

    for entry in candidates:
        metrics = entry.get("metrics") or {}
        median_val = float(metrics.get("median", float("nan")))
        span_val = float(metrics.get("span", float("nan")))
        robust_val = float(metrics.get("robust_sigma", 0.0))
        grad_val = float(metrics.get("grad_proxy", 0.0))
        score_val = float(metrics.get("score", float("inf")))
        accepted = metrics is not None and math.isfinite(median_val) and math.isfinite(span_val)
        if accepted and (span_val < span_low or span_val > span_high):
            accepted = False
        if accepted and deviation_clip is not None and deviation_clip > 0:
            if abs(median_val - group_median) > deviation_clip:
                accepted = False
        if accepted:
            if score_val < best_score:
                best_score = score_val
                best_entry = entry
        if math.isfinite(score_val) and score_val < best_score_soft:
            best_score_soft = score_val
            best_entry_soft = entry

        entry["accepted"] = accepted
        entry["score"] = score_val
        _log_and_callback(
            "post_anchor_candidate",
            lvl="DEBUG",
            callback=progress_callback,
            tile=int(entry.get("tile_id", -1)),
            median=median_val,
            span=span_val,
            robust=robust_val,
            grad=grad_val,
            score=score_val,
            accepted="accepted" if accepted else "rejected",
        )

    if best_entry is None:
        if best_entry_soft is not None:
            best_entry = best_entry_soft
            best_score = best_score_soft
            _log_and_callback("center_anchor_soft_fallback", lvl="WARN", callback=progress_callback)
        else:
            for entry in candidates:
                entry.pop("array", None)
            return master_tiles, anchor_shift

    current_anchor_entry = None
    if prestack_anchor_id is not None:
        for entry in candidates:
            if int(entry.get("tile_id", -1)) == int(prestack_anchor_id):
                current_anchor_entry = entry
                break

    best_score = float(best_score)
    if not math.isfinite(best_score):
        for entry in candidates:
            entry.pop("array", None)
        return master_tiles, anchor_shift

    anchor_score = float("inf")
    if current_anchor_entry is not None:
        anchor_score = float(current_anchor_entry.get("score", float("inf")))

    improvement = 0.0
    if math.isfinite(anchor_score) and anchor_score > 0:
        improvement = (anchor_score - best_score) / anchor_score
    elif math.isfinite(best_score):
        improvement = 1.0
    improvement = max(0.0, improvement)

    selected_tile_id = int(best_entry.get("tile_id", -1))
    MIN_MEDIAN_REL_DELTA = 0.01
    median_delta_ok = True
    if current_anchor_entry and current_anchor_entry.get("metrics") and best_entry.get("metrics"):
        old_med = float(current_anchor_entry["metrics"].get("median", 0.0))
        new_med = float(best_entry["metrics"].get("median", 0.0))
        denom = max(abs(old_med), 1e-6)
        median_delta = abs(new_med - old_med) / denom
        median_delta_ok = median_delta >= MIN_MEDIAN_REL_DELTA
    else:
        median_delta = 0.0

    if (
        selected_tile_id == int(prestack_anchor_id)
        or improvement < min_improvement
        or not median_delta_ok
    ):
        _log_and_callback(
            "post_anchor_keep_old",
            lvl="INFO",
            callback=progress_callback,
            impr=float(improvement),
        )
        for entry in candidates:
            entry.pop("array", None)
        return master_tiles, anchor_shift

    if current_anchor_entry is None or not current_anchor_entry.get("metrics"):
        _log_and_callback(
            "post_anchor_selected",
            lvl="INFO",
            callback=progress_callback,
            tile=selected_tile_id,
            impr=float(improvement),
        )
        for entry in candidates:
            entry.pop("array", None)
        return master_tiles, anchor_shift

    def _compute_overlap_shift(old_entry: dict[str, Any], new_entry: dict[str, Any]) -> tuple[float, float] | None:
        if not use_overlap_affine or reproject_interp is None:
            return None
        arr_old = old_entry.get("array")
        arr_new = new_entry.get("array")
        wcs_old = old_entry.get("wcs")
        wcs_new = new_entry.get("wcs")
        if arr_old is None or arr_new is None or wcs_old is None or wcs_new is None:
            return None
        try:
            old_lum = arr_old if arr_old.ndim == 2 else np.mean(arr_old, axis=2, dtype=np.float32)
            new_lum = arr_new if arr_new.ndim == 2 else np.mean(arr_new, axis=2, dtype=np.float32)
            old_lum = np.asarray(old_lum, dtype=np.float32)
            new_lum = np.asarray(new_lum, dtype=np.float32)
            if old_lum.size == 0 or new_lum.size == 0:
                return None
            old_med = float(np.nanmedian(old_lum[np.isfinite(old_lum)])) if np.any(np.isfinite(old_lum)) else 0.0
            new_med = float(np.nanmedian(new_lum[np.isfinite(new_lum)])) if np.any(np.isfinite(new_lum)) else 0.0
            old_clean = np.nan_to_num(old_lum, nan=old_med, posinf=old_med, neginf=old_med)
            new_clean = np.nan_to_num(new_lum, nan=new_med, posinf=new_med, neginf=new_med)
            reproj_new, footprint = reproject_interp((new_clean, wcs_new), wcs_old, shape_out=old_clean.shape)
            if footprint is None:
                return None
            mask = np.isfinite(reproj_new) & np.isfinite(old_clean) & (footprint > 0.25)
            if mask.sum() < max(256, int(old_clean.size * 0.005)):
                return None
            old_vals = old_clean[mask]
            new_vals = reproj_new[mask]
            if old_vals.size == 0 or new_vals.size == 0:
                return None
            lo_old = float(np.percentile(old_vals, sky_low_pct))
            hi_old = float(np.percentile(old_vals, sky_high_pct))
            lo_new = float(np.percentile(new_vals, sky_low_pct))
            hi_new = float(np.percentile(new_vals, sky_high_pct))
            span_old = max(hi_old - lo_old, 1e-6)
            span_new = max(hi_new - lo_new, 0.0)
            gain_val = span_new / span_old if span_old > 1e-6 else 1.0
            if not math.isfinite(gain_val) or gain_val <= 0:
                gain_val = 1.0
            med_old = float(np.median(old_vals))
            med_new = float(np.median(new_vals))
            offset_val = med_new - gain_val * med_old
            if not math.isfinite(offset_val):
                offset_val = 0.0
            return float(gain_val), float(offset_val)
        except Exception:
            return None

    def _compute_metrics_shift(old_metrics: dict[str, float], new_metrics: dict[str, float]) -> tuple[float, float]:
        span_old = float(old_metrics.get("span", 0.0))
        span_new = float(new_metrics.get("span", 0.0))
        if not math.isfinite(span_old) or span_old <= 1e-6:
            span_old = 1.0
        gain_val = span_new / span_old if span_old > 0 else 1.0
        if not math.isfinite(gain_val) or gain_val <= 0:
            gain_val = 1.0
        median_old = float(old_metrics.get("median", 0.0))
        median_new = float(new_metrics.get("median", 0.0))
        offset_val = median_new - gain_val * median_old
        if not math.isfinite(offset_val):
            offset_val = 0.0
        return float(gain_val), float(offset_val)

    gain_shift, offset_shift = (1.0, 0.0)
    overlap_shift = _compute_overlap_shift(current_anchor_entry, best_entry)
    if overlap_shift is not None:
        gain_shift, offset_shift = overlap_shift
    else:
        gain_shift, offset_shift = _compute_metrics_shift(current_anchor_entry["metrics"], best_entry["metrics"])

    anchor_shift = (float(gain_shift), float(offset_shift))

    _log_and_callback(
        "post_anchor_selected",
        lvl="INFO",
        callback=progress_callback,
        tile=selected_tile_id,
        impr=float(improvement),
    )
    _log_and_callback(
        "post_anchor_shift",
        lvl="INFO",
        callback=progress_callback,
        gain=float(gain_shift),
        offset=float(offset_shift),
    )

    for entry in candidates:
        entry.pop("array", None)

    return master_tiles, anchor_shift


# --- Helpers for RAM budget enforcement during stacking ---
def _extract_hw_from_info(raw_info: dict) -> tuple[int, int]:
    """Return (H, W) dimensions inferred from cached metadata."""

    if not isinstance(raw_info, dict):
        return 0, 0

    shape = raw_info.get("preprocessed_shape")
    if shape:
        try:
            # Accept either (H, W) or (H, W, C)
            h = int(shape[0])
            w = int(shape[1]) if len(shape) >= 2 else 0
            if h > 0 and w > 0:
                return h, w
        except Exception:
            pass

    header_obj = raw_info.get("header")
    if header_obj is not None:
        try:
            # fits.Header exposes .get, dict fallback to __getitem__
            get = header_obj.get if hasattr(header_obj, "get") else header_obj.__getitem__
            w = int(get("NAXIS1", 0)) if hasattr(header_obj, "get") else int(get("NAXIS1"))
            h = int(get("NAXIS2", 0)) if hasattr(header_obj, "get") else int(get("NAXIS2"))
            if h > 0 and w > 0:
                return h, w
        except Exception:
            pass

    wcs_obj = raw_info.get("wcs")
    if wcs_obj is not None and getattr(wcs_obj, "pixel_shape", None):
        try:
            w = int(wcs_obj.pixel_shape[0])
            h = int(wcs_obj.pixel_shape[1]) if len(wcs_obj.pixel_shape) > 1 else 0
            if h > 0 and w > 0:
                return h, w
        except Exception:
            pass

    return 0, 0


def _estimate_group_memory_bytes(group: list[dict]) -> tuple[int, int, int, int]:
    """Estimate total memory footprint (bytes) for a stack group.

    Returns ``(total_bytes, per_frame_bytes, max_h, max_w)``.
    ``per_frame_bytes`` follows the simplified model ``H * W * 4``.
    """

    if not group:
        return 0, 0, 0, 0

    max_h = 0
    max_w = 0
    for info in group:
        h, w = _extract_hw_from_info(info)
        max_h = max(max_h, int(h))
        max_w = max(max_w, int(w))

    if max_h <= 0 or max_w <= 0:
        return 0, 0, max_h, max_w

    per_frame_bytes = int(max_h) * int(max_w) * 4
    total_bytes = per_frame_bytes * len(group)
    return total_bytes, per_frame_bytes, max_h, max_w


def _split_group_temporally(group: list[dict], segment_size: int) -> list[list[dict]]:
    """Split ``group`` into contiguous segments of ``segment_size`` (>=1)."""

    if segment_size <= 0:
        return [group]
    return [group[i:i + segment_size] for i in range(0, len(group), segment_size)]


def _estimate_per_frame_cost_mb(
    header_items: list[dict] | None,
    bytes_per_pixel: int = 4,
    overhead_factor: float = 2.0,
    sample_size: int = 32,
) -> dict:
    """Estimate per-frame memory usage from Phase 0 metadata.

    Returns a dictionary containing ``per_frame_mb``, ``max_height`` and
    ``max_width`` along with the inferred ``channels``.
    """

    if not header_items:
        header_items = []

    try:
        overhead_factor = max(1.0, float(overhead_factor))
    except Exception:
        overhead_factor = 2.0

    max_h = 0
    max_w = 0
    max_channels = 0

    if header_items:
        if sample_size > 0 and len(header_items) > sample_size:
            step = max(1, len(header_items) // sample_size)
            sampled_items = [header_items[i] for i in range(0, len(header_items), step)][:sample_size]
        else:
            sampled_items = list(header_items)
    else:
        sampled_items = []

    for item in sampled_items:
        try:
            shape = item.get("shape") if isinstance(item, dict) else None
            if shape:
                h = int(shape[0]) if len(shape) >= 1 else 0
                w = int(shape[1]) if len(shape) >= 2 else 0
                c = int(shape[2]) if len(shape) >= 3 else 1
            else:
                header = item.get("header") if isinstance(item, dict) else None
                h, w = 0, 0
                c = 1
                if header is not None:
                    getter = header.get if hasattr(header, "get") else header.__getitem__
                    try:
                        w = int(getter("NAXIS1", 0)) if hasattr(header, "get") else int(getter("NAXIS1"))
                        h = int(getter("NAXIS2", 0)) if hasattr(header, "get") else int(getter("NAXIS2"))
                    except Exception:
                        h, w = 0, 0
                    try:
                        if hasattr(header, "get"):
                            naxis = int(header.get("NAXIS", 2))
                        else:
                            naxis = int(header["NAXIS"]) if "NAXIS" in header else 2
                    except Exception:
                        naxis = 2
                    if naxis >= 3:
                        try:
                            if hasattr(header, "get"):
                                c = int(header.get("NAXIS3", 1))
                            else:
                                c = int(header.get("NAXIS3", 1)) if hasattr(header, "get") else int(header["NAXIS3"])
                        except Exception:
                            c = 1
                else:
                    h, w, c = 0, 0, 1
            if isinstance(item, dict):
                if "BAYERPAT" in item.get("header", {}):
                    c = max(1, c)
            max_h = max(max_h, int(h))
            max_w = max(max_w, int(w))
            max_channels = max(max_channels, max(1, int(c)))
        except Exception:
            continue

    if max_h <= 0 or max_w <= 0:
        # Conservative fallback for unknown dimensions (~9MP mono sensor)
        max_h = 3000
        max_w = 3000
    if max_channels <= 0:
        max_channels = 1

    per_frame_bytes = max_h * max_w * max_channels * max(1, int(bytes_per_pixel))
    per_frame_mb = (per_frame_bytes / (1024 * 1024)) * overhead_factor

    return {
        "per_frame_mb": float(per_frame_mb),
        "bytes_per_pixel": int(bytes_per_pixel),
        "overhead_factor": float(overhead_factor),
        "max_height": int(max_h),
        "max_width": int(max_w),
        "channels": int(max_channels),
    }


def _probe_system_resources(
    cache_dir: str | None = None,
    *,
    two_pass_enabled: bool | None = None,
    two_pass_sigma_px: int | None = None,
    two_pass_gain_clip: tuple[float, float] | list[float] | None = None,
) -> dict:
    """Collect RAM, disk and GPU availability information."""

    info: dict = {
        "ram_total_mb": None,
        "ram_available_mb": None,
        "usable_ram_mb": None,
        "disk_total_mb": None,
        "disk_free_mb": None,
        "usable_disk_mb": None,
        "gpu_total_mb": None,
        "gpu_free_mb": None,
        "usable_vram_mb": None,
    }

    try:
        if psutil is not None:
            vm = psutil.virtual_memory()
            info["ram_total_mb"] = vm.total / (1024 * 1024)
            info["ram_available_mb"] = vm.available / (1024 * 1024)
            info["usable_ram_mb"] = min(info["ram_total_mb"], info["ram_available_mb"] * 0.6) if info["ram_available_mb"] else None
    except Exception:
        pass

    tp_enabled = bool(two_pass_enabled) if two_pass_enabled is not None else False
    tp_sigma_px = 50
    if two_pass_sigma_px is not None:
        try:
            tp_sigma_px = int(two_pass_sigma_px)
        except (TypeError, ValueError):
            tp_sigma_px = 50
    gain_clip_tuple: tuple[float, float] = (0.85, 1.18)
    if isinstance(two_pass_gain_clip, (list, tuple)) and len(two_pass_gain_clip) >= 2:
        try:
            low = float(two_pass_gain_clip[0])
            high = float(two_pass_gain_clip[1])
            if low > high:
                low, high = high, low
            gain_clip_tuple = (low, high)
        except (TypeError, ValueError):
            gain_clip_tuple = (0.85, 1.18)
    info["two_pass_enabled"] = tp_enabled
    info["two_pass_sigma_px"] = tp_sigma_px
    info["two_pass_gain_clip"] = gain_clip_tuple

    try:
        target_dir = cache_dir if cache_dir and os.path.isdir(cache_dir) else os.getcwd()
        du = shutil.disk_usage(target_dir)
        disk_total_mb = du.total / (1024 * 1024)
        disk_free_mb = du.free / (1024 * 1024)
        info["disk_total_mb"] = disk_total_mb
        info["disk_free_mb"] = disk_free_mb
        info["usable_disk_mb"] = disk_free_mb * 0.7
    except Exception:
        pass

    # GPU detection via CuPy first, then torch
    try:
        if CUPY_AVAILABLE:
            import cupy  # type: ignore

            try:
                cupy.cuda.Device().use()
                free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
                free_mb = free_bytes / (1024 * 1024)
                total_mb = total_bytes / (1024 * 1024)
                info["gpu_total_mb"] = total_mb
                info["gpu_free_mb"] = free_mb
                info["usable_vram_mb"] = free_mb * 0.7
            except Exception:
                pass
        elif importlib.util.find_spec("torch") is not None:
            import torch  # type: ignore

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
                free_mb = torch.cuda.mem_get_info(device)[0] / (1024 * 1024)
                info["gpu_total_mb"] = total_mb
                info["gpu_free_mb"] = free_mb
                info["usable_vram_mb"] = free_mb * 0.7
    except Exception:
        pass

    return info


def _compute_auto_tile_caps(
    resource_info: dict,
    per_frame_info: dict,
    policy_max: int = 50,
    policy_min: int = 8,
    disk_threshold_mb: float = 8192.0,
    user_max_override: int | None = None,
) -> dict:
    """Combine resource probes and per-frame costs into adaptive caps."""

    per_frame_mb = float(per_frame_info.get("per_frame_mb", 0.0) or 0.0)
    usable_ram_mb = float(resource_info.get("usable_ram_mb") or 0.0)
    ram_available_mb = float(resource_info.get("ram_available_mb") or 0.0)

    if user_max_override and user_max_override > 0:
        policy_max = min(policy_max, int(user_max_override))

    frames_by_ram = 0
    if per_frame_mb > 0 and usable_ram_mb > 0:
        frames_by_ram = max(0, int(math.floor(usable_ram_mb / per_frame_mb)))

    cap_candidate = policy_max if policy_max > 0 else frames_by_ram or policy_min
    if frames_by_ram > 0:
        cap_candidate = min(cap_candidate, frames_by_ram)
    cap_candidate = max(policy_min, cap_candidate)

    disk_free_mb = float(resource_info.get("disk_free_mb") or 0.0)
    usable_disk_mb = float(resource_info.get("usable_disk_mb") or 0.0)

    memmap_enabled = False
    memmap_budget_mb = None
    if frames_by_ram < policy_min and disk_free_mb > disk_threshold_mb:
        memmap_enabled = True
        memmap_budget_mb = max(policy_min * per_frame_mb, usable_disk_mb * 0.2 if usable_disk_mb else disk_free_mb * 0.2)

    gpu_hint = None
    usable_vram_mb = float(resource_info.get("usable_vram_mb") or 0.0)
    if per_frame_mb > 0 and usable_vram_mb > 0:
        gpu_hint = max(1, min(cap_candidate, int(math.floor(usable_vram_mb / per_frame_mb))))

    parallel_cap = 1
    if frames_by_ram and cap_candidate > 0:
        parallel_cap = max(1, frames_by_ram // max(1, cap_candidate))
    if memmap_enabled:
        parallel_cap = 1

    return {
        "per_frame_mb": per_frame_mb,
        "frames_by_ram": frames_by_ram,
        "cap": int(cap_candidate),
        "min_cap": int(policy_min),
        "memmap": bool(memmap_enabled),
        "memmap_budget_mb": memmap_budget_mb,
        "gpu_batch_hint": gpu_hint,
        "ram_available_mb": ram_available_mb,
        "parallel_groups": int(parallel_cap),
    }


def _extract_timestamp(info: dict, fallback: float) -> float:
    header = info.get("header") if isinstance(info, dict) else None
    if header is not None:
        for key in ("DATE-OBS", "DATE-AVG", "DATE", "TIME-OBS"):
            try:
                if hasattr(header, "get"):
                    value = header.get(key)
                else:
                    value = header[key] if key in header else None
            except Exception:
                value = None
            if not value:
                continue
            try:
                from astropy.time import Time  # type: ignore

                return float(Time(value, format="isot", scale="utc").unix)
            except Exception:
                try:
                    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                    return dt.timestamp()
                except Exception:
                    continue
    try:
        idx = info.get("phase0_index")
        if idx is not None:
            return float(idx)
    except Exception:
        pass
    return float(fallback)


def _extract_ra_dec_deg(info: dict) -> tuple[float, float] | None:
    wcs_obj = info.get("wcs") if isinstance(info, dict) else None
    if wcs_obj and getattr(wcs_obj, "is_celestial", False):
        try:
            if getattr(wcs_obj, "pixel_shape", None):
                cx = wcs_obj.pixel_shape[0] / 2.0
                cy = wcs_obj.pixel_shape[1] / 2.0
            else:
                cx = cy = 0.0
            center = wcs_obj.pixel_to_world(cx, cy)
            if hasattr(center, "ra") and hasattr(center.ra, "deg"):
                return float(center.ra.deg), float(center.dec.deg)
        except Exception:
            pass

    if isinstance(info, dict):
        phase0_center = info.get("phase0_center")
        if phase0_center is not None:
            try:
                if hasattr(phase0_center, "ra") and hasattr(phase0_center.ra, "deg"):
                    return float(phase0_center.ra.deg), float(phase0_center.dec.deg)
                if isinstance(phase0_center, (list, tuple)) and len(phase0_center) >= 2:
                    return float(phase0_center[0]), float(phase0_center[1])
            except Exception:
                pass

        header = info.get("header")
        if header is not None:
            try:
                getter = header.get if hasattr(header, "get") else header.__getitem__
                ra = getter("CRVAL1", None)
                dec = getter("CRVAL2", None)
                if ra is not None and dec is not None:
                    return float(ra), float(dec)
            except Exception:
                pass
    return None


def _estimate_frame_fov_deg(info: dict) -> float | None:
    if isinstance(info, dict):
        direct = info.get("phase0_fov_deg") or info.get("estimated_fov_deg")
        if direct:
            try:
                return float(direct)
            except Exception:
                pass
    wcs_obj = info.get("wcs") if isinstance(info, dict) else None
    if wcs_obj and getattr(wcs_obj, "is_celestial", False):
        try:
            if getattr(wcs_obj, "pixel_shape", None):
                width = float(wcs_obj.pixel_shape[0])
                height = float(wcs_obj.pixel_shape[1]) if len(wcs_obj.pixel_shape) > 1 else width
            else:
                height, width = _extract_hw_from_info(info)
            if width and height:
                xs = [0.0, width, 0.0, width]
                ys = [0.0, 0.0, height, height]
                corners = wcs_obj.pixel_to_world(xs, ys)
                if SkyCoord is not None and u is not None:
                    sc = SkyCoord(ra=corners.ra, dec=corners.dec)
                    seps = sc[:, None].separation(sc[None, :]).deg
                    return float(np.nanmax(seps)) if np.size(seps) else None
        except Exception:
            pass

    header = info.get("header") if isinstance(info, dict) else None
    if header is not None:
        try:
            getter = header.get if hasattr(header, "get") else header.__getitem__
            cd1 = abs(float(getter("CDELT1", 0)))
            cd2 = abs(float(getter("CDELT2", 0)))
            h, w = _extract_hw_from_info(info)
            if cd1 and cd2 and h and w:
                return math.hypot(cd1 * w, cd2 * h)
        except Exception:
            pass
    return None


def _unit_vector_from_ra_dec(ra_deg: float, dec_deg: float) -> tuple[float, float, float]:
    ra_rad = math.radians(float(ra_deg))
    dec_rad = math.radians(float(dec_deg))
    x = math.cos(dec_rad) * math.cos(ra_rad)
    y = math.cos(dec_rad) * math.sin(ra_rad)
    z = math.sin(dec_rad)
    return x, y, z


def _compute_max_angular_separation_deg(coords: list[tuple[float, float]]) -> float:
    if not coords or len(coords) < 2:
        return 0.0
    if SkyCoord is not None and u is not None:
        try:
            sc = SkyCoord(ra=[c[0] for c in coords] * u.deg, dec=[c[1] for c in coords] * u.deg)
            seps = sc[:, None].separation(sc[None, :]).deg
            return float(np.nanmax(seps)) if np.size(seps) else 0.0
        except Exception:
            pass
    vectors = np.array([_unit_vector_from_ra_dec(*c) for c in coords], dtype=float)
    max_sep = 0.0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dot = float(np.dot(vectors[i], vectors[j]))
            dot = min(1.0, max(-1.0, dot))
            sep = math.degrees(math.acos(dot))
            if sep > max_sep:
                max_sep = sep
    return max_sep


def _cluster_unit_vectors(vectors: 'np.ndarray', k: int, max_iter: int = 25) -> list[int]:
    if k <= 1 or vectors.shape[0] <= 1:
        return [0] * vectors.shape[0]
    k = min(k, vectors.shape[0])
    centers = [vectors[0]]
    for _ in range(1, k):
        distances = 1 - np.dot(vectors, np.stack(centers, axis=0).T)
        min_dist = np.min(distances, axis=1)
        idx = int(np.argmax(min_dist))
        centers.append(vectors[idx])
    centers = np.array(centers, dtype=float)

    assignments = np.zeros(vectors.shape[0], dtype=int)
    for _ in range(max_iter):
        distances = 1 - np.dot(vectors, centers.T)
        new_assignments = np.argmin(distances, axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for ci in range(k):
            members = vectors[assignments == ci]
            if members.size == 0:
                # Reinitialize empty cluster to farthest point
                idx = int(np.argmax(np.min(distances, axis=1)))
                centers[ci] = vectors[idx]
            else:
                center = members.mean(axis=0)
                norm = np.linalg.norm(center)
                if norm > 0:
                    centers[ci] = center / norm
    return assignments.tolist()


def _sort_group_chronologically(group: list[dict]) -> list[dict]:
    ordered = []
    for idx, info in enumerate(group):
        ts = _extract_timestamp(info, idx)
        ordered.append((ts, idx, info))
    ordered.sort(key=lambda x: (x[0], x[1]))
    return [item[2] for item in ordered]


def _chunk_sequence(seq: list[dict], size: int) -> list[list[dict]]:
    if size <= 0:
        return [seq]
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _auto_split_single_group(
    group: list[dict],
    cap: int,
    min_cap: int,
    spatial_fraction: float = 0.25,
) -> tuple[list[list[dict]], dict]:
    n = len(group)
    detail = {
        "original_size": n,
        "segment_sizes": [n],
        "spatial_split": False,
        "reason": "within_cap" if n <= cap else "ram_cap",
        "dispersion_deg": None,
        "fov_deg": None,
    }

    if n <= max(cap, min_cap):
        return [group], detail

    centers = []
    indices = []
    for idx, info in enumerate(group):
        coord = _extract_ra_dec_deg(info)
        if coord:
            centers.append(coord)
            indices.append(idx)

    fov_deg = _estimate_frame_fov_deg(group[0]) if group else None
    dispersion_deg = _compute_max_angular_separation_deg(centers) if centers else 0.0
    detail["dispersion_deg"] = dispersion_deg
    detail["fov_deg"] = fov_deg

    base_clusters: list[list[tuple[int, dict]]] = []
    if (
        centers
        and fov_deg
        and dispersion_deg > float(fov_deg) * float(max(0.0, spatial_fraction))
    ):
        k = max(1, math.ceil(n / max(1, cap)))
        vectors = np.array([_unit_vector_from_ra_dec(*c) for c in centers], dtype=float)
        assignments = _cluster_unit_vectors(vectors, k)
        cluster_map: dict[int, list[tuple[int, dict]]] = {i: [] for i in range(k)}
        for pos, assignment in zip(indices, assignments):
            cluster_map.setdefault(int(assignment), []).append((pos, group[pos]))
        remaining_indices = [i for i in range(n) if i not in indices]
        for idx in remaining_indices:
            target = min(cluster_map.keys(), key=lambda key: (len(cluster_map[key]), key))
            cluster_map[target].append((idx, group[idx]))
        base_clusters = [sorted(items, key=lambda x: x[0]) for items in cluster_map.values() if items]
        if len(base_clusters) > 1:
            detail["spatial_split"] = True
            detail["reason"] = "dispersion"
    if not base_clusters:
        base_clusters = [list(enumerate(group))]

    output_groups: list[list[dict]] = []
    for cluster in base_clusters:
        ordered = _sort_group_chronologically([info for _idx, info in cluster])
        output_groups.extend(_chunk_sequence(ordered, max(min_cap, cap)))

    detail["segment_sizes"] = [len(sub) for sub in output_groups]
    return output_groups, detail


def _auto_split_groups(
    groups: list[list[dict]],
    cap: int,
    min_cap: int,
    progress_callback: Callable | None = None,
    spatial_fraction: float = 0.25,
) -> list[list[dict]]:
    if cap <= 0 or not groups:
        return groups
    new_groups: list[list[dict]] = []
    for idx, group in enumerate(groups, start=1):
        subgroups, detail = _auto_split_single_group(group, cap, min_cap, spatial_fraction)
        new_groups.extend(subgroups)
        if progress_callback:
            try:
                sizes_str = ",".join(str(len(sg)) for sg in subgroups)
                msg = (
                    f"AutoSplit: group #{idx} N={len(group)} -> {len(subgroups)} subgroups "
                    f"[{sizes_str}] (chrono; spatial split={'yes' if detail['spatial_split'] else 'no'}; "
                    f"reason={detail['reason']})"
                )
                _log_and_callback(msg, prog=None, lvl="INFO_DETAIL", callback=progress_callback)
            except Exception:
                pass
    return new_groups


def _group_center_deg(group):
    """Renvoie le centre RA/DEC moyen d'un groupe."""

    ras, decs = [], []
    for info in group:
        ra, dec = info.get("RA"), info.get("DEC")
        if ra is not None and dec is not None:
            ras.append(float(ra))
            decs.append(float(dec))
    if not ras:
        return None
    return (sum(ras) / len(ras), sum(decs) / len(decs))


def _angular_sep_deg(a, b):
    """Distance angulaire simple en degrés (approximation suffisante)."""

    if not a or not b:
        return 9999
    dra = abs(a[0] - b[0])
    ddec = abs(a[1] - b[1])
    return (dra**2 + ddec**2) ** 0.5


def _merge_small_groups(groups, min_size, cap):
    """
    Fusionne les petits groupes (<min_size) avec le plus proche voisin
    si le total reste <= cap (avec marge 10%).
    """

    merged_flags = [False] * len(groups)
    centers = [_group_center_deg(g) for g in groups]

    for i, gi in enumerate(groups):
        if merged_flags[i] or len(gi) >= min_size:
            continue

        best_j, best_d = None, 1e9
        for j, gj in enumerate(groups):
            if i == j or merged_flags[j]:
                continue
            d = _angular_sep_deg(centers[i], centers[j])
            if d < best_d:
                best_d, best_j = d, j

        if best_j is not None and len(groups[best_j]) + len(gi) <= int(cap * 1.1):
            groups[best_j].extend(gi)
            merged_flags[i] = True
            print(
                f"[AutoMerge] Group {i} ({len(gi)} imgs) merged into {best_j} (now {len(groups[best_j])})"
            )

    return [g for k, g in enumerate(groups) if not merged_flags[k]]


def _attempt_recluster_for_budget(
    group: list[dict],
    budget_bytes: int,
    base_threshold_deg: float,
    orientation_split_threshold_deg: float,
    cluster_func: Callable[..., list] = cluster_seestar_stacks_connected,
    max_attempts: int = 6,
) -> tuple[list[list[dict]], float, int] | None:
    """Try to relax clustering threshold until all subgroups fit the RAM budget."""

    if not group or len(group) <= 1:
        return None
    try:
        current_thr = float(base_threshold_deg)
    except Exception:
        return None
    if current_thr <= 0:
        return None

    for attempt in range(1, max_attempts + 1):
        current_thr = max(current_thr * 0.7, 1e-5)
        try:
            reclustered = cluster_func(
                group,
                float(current_thr),
                None,
                orientation_split_threshold_deg=orientation_split_threshold_deg,
            )
        except Exception:
            return None

        if not reclustered or len(reclustered) <= 1:
            continue

        fits_budget = True
        for sub in reclustered:
            total_bytes, _, _, _ = _estimate_group_memory_bytes(sub)
            if budget_bytes > 0 and total_bytes > budget_bytes:
                fits_budget = False
                break
        if fits_budget:
            return reclustered, float(current_thr), attempt

    return None


def _apply_ram_budget_to_groups(
    groups: list[list[dict]],
    budget_bytes: int,
    base_threshold_deg: float,
    orientation_split_threshold_deg: float,
    cluster_func: Callable[..., list] = cluster_seestar_stacks_connected,
) -> tuple[list[list[dict]], list[dict]]:
    """Ensure each stack group fits in the RAM budget by splitting or re-clustering."""

    if budget_bytes is None or budget_bytes <= 0:
        return groups, []

    final_groups: list[list[dict]] = []
    adjustments: list[dict] = []
    queue: list[tuple[int, list[dict]]] = [(idx + 1, grp) for idx, grp in enumerate(groups)]

    while queue:
        group_index, group = queue.pop(0)
        total_bytes, per_frame_bytes, _, _ = _estimate_group_memory_bytes(group)

        if total_bytes <= 0 or total_bytes <= budget_bytes:
            final_groups.append(group)
            continue

        if len(group) == 1:
            # Nothing else can be done; log and proceed.
            adjustments.append(
                {
                    "method": "single_over_budget",
                    "group_index": group_index,
                    "original_frames": len(group),
                    "estimated_mb": total_bytes / (1024 ** 2),
                    "budget_mb": budget_bytes / (1024 ** 2),
                }
            )
            final_groups.append(group)
            continue

        recluster_result = _attempt_recluster_for_budget(
            group,
            budget_bytes,
            base_threshold_deg,
            orientation_split_threshold_deg,
            cluster_func=cluster_func,
        )
        if recluster_result:
            reclustered_groups, new_threshold, attempts = recluster_result
            adjustments.append(
                {
                    "method": "recluster",
                    "group_index": group_index,
                    "original_frames": len(group),
                    "num_subgroups": len(reclustered_groups),
                    "new_threshold_deg": new_threshold,
                    "attempts": attempts,
                    "estimated_mb": total_bytes / (1024 ** 2),
                    "budget_mb": budget_bytes / (1024 ** 2),
                }
            )
            queue = [(group_index, sub) for sub in reclustered_groups] + queue
            continue

        if per_frame_bytes <= 0:
            # Unable to infer size; keep original group.
            final_groups.append(group)
            continue

        max_frames = max(1, int(budget_bytes // per_frame_bytes))
        if max_frames >= len(group):
            final_groups.append(group)
            continue

        segmented = _split_group_temporally(group, max_frames)
        still_over = any(_estimate_group_memory_bytes(seg)[0] > budget_bytes for seg in segmented)
        adjustments.append(
            {
                "method": "split",
                "group_index": group_index,
                "original_frames": len(group),
                "num_subgroups": len(segmented),
                "segment_size": max_frames,
                "estimated_mb": total_bytes / (1024 ** 2),
                "budget_mb": budget_bytes / (1024 ** 2),
                "still_over_budget": still_over,
            }
        )
        queue = [(group_index, seg) for seg in segmented] + queue

    return final_groups, adjustments


# --- Configuration du Logging ---
try:
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zemosaic_worker.log")
except NameError:
    log_file_path = "zemosaic_worker.log"

logger = logging.getLogger("ZeMosaicWorker")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
logger.info("Logging pour ZeMosaicWorker initialisé. Logs écrits dans: %s", log_file_path)

# --- Alignment Warning Tracking ---
# These warnings come from zemosaic_align_stack when an image fails to align.
# We count them here so a summary can be written at the end of a run.
ALIGN_WARNING_SUMMARY = {
    "aligngroup_warn_max_iter_error": "astroalign MaxIterError",
    "aligngroup_warn_shape_mismatch_after_align": "shape mismatch after align",
    "aligngroup_warn_register_returned_none": "astroalign returned None",
    "aligngroup_warn_value_error": "value error during align",
}
ALIGN_WARNING_COUNTS = {key: 0 for key in ALIGN_WARNING_SUMMARY}

# --- Third-Party Library Imports ---
import zarr
from packaging.version import Version

try:
    from zarr.storage import LRUStoreCache
    if Version(zarr.__version__).major >= 3:
        # In zarr>=3 LRUStoreCache was removed. Use a no-op wrapper
        raise ImportError
except Exception:  # pragma: no cover - fallback for zarr>=3 or missing cache
    class LRUStoreCache:
        """Simple pass-through wrapper used when LRUStoreCache is unavailable."""

        def __init__(self, store, max_size=None):
            self.store = store

        def __getattr__(self, name):
            return getattr(self.store, name)

try:
    # Prefer storage module first (zarr < 3)
    from zarr.storage import DirectoryStore
except Exception:
    try:  # pragma: no cover - zarr >= 3 uses LocalStore
        from zarr.storage import LocalStore as DirectoryStore
    except Exception:
        try:
            from zarr.storage import FsspecStore
            import fsspec

            def DirectoryStore(path):
                return FsspecStore(fsspec.filesystem("file").get_mapper(path))
        except Exception:  # pragma: no cover - ultimate fallback
            DirectoryStore = None

# now LRUStoreCache and DirectoryStore are defined


# --- Astropy (critique) ---
ASTROPY_AVAILABLE = False
WCS, SkyCoord, Angle, fits, u = None, None, None, None, None
try:
    from astropy.io import fits as actual_fits
    from astropy.wcs import WCS as actual_WCS
    from astropy.coordinates import SkyCoord as actual_SkyCoord, Angle as actual_Angle
    from astropy import units as actual_u
    fits, WCS, SkyCoord, Angle, u = actual_fits, actual_WCS, actual_SkyCoord, actual_Angle, actual_u
    ASTROPY_AVAILABLE = True
    logger.info("Bibliothèque Astropy importée.")
except ImportError as e_astro_imp: logger.critical(f"Astropy non trouvée: {e_astro_imp}.")
except Exception as e_astro_other_imp: logger.critical(f"Erreur import Astropy: {e_astro_other_imp}", exc_info=True)

# --- Reproject (critique pour la mosaïque) ---
REPROJECT_AVAILABLE = False
find_optimal_celestial_wcs, reproject_and_coadd, reproject_interp = None, None, None
try:
    from reproject.mosaicking import find_optimal_celestial_wcs as actual_find_optimal_wcs
    from reproject.mosaicking import reproject_and_coadd as actual_reproject_coadd
    from reproject import reproject_interp as actual_reproject_interp
    find_optimal_celestial_wcs, reproject_and_coadd, reproject_interp = actual_find_optimal_wcs, actual_reproject_coadd, actual_reproject_interp
    REPROJECT_AVAILABLE = True
    logger.info("Bibliothèque 'reproject' importée.")
except ImportError as e_reproject_final: logger.critical(f"Échec import reproject: {e_reproject_final}.")
except Exception as e_reproject_other_final: logger.critical(f"Erreur import 'reproject': {e_reproject_other_final}", exc_info=True)

# --- Local Project Module Imports ---
zemosaic_utils, ZEMOSAIC_UTILS_AVAILABLE = None, False
zemosaic_astrometry, ZEMOSAIC_ASTROMETRY_AVAILABLE = None, False
zemosaic_align_stack, ZEMOSAIC_ALIGN_STACK_AVAILABLE = None, False
CALC_GRID_OPTIMIZED_AVAILABLE = False
_calculate_final_mosaic_grid_optimized = None

try:
    import zemosaic_utils
    from zemosaic_utils import (
        gpu_assemble_final_mosaic_reproject_coadd,
        gpu_assemble_final_mosaic_incremental,
        reproject_and_coadd_wrapper,
    )
    ZEMOSAIC_UTILS_AVAILABLE = True
    logger.info("Module 'zemosaic_utils' importé.")
except ImportError as e: logger.error(f"Import 'zemosaic_utils.py' échoué: {e}.")
try: import zemosaic_astrometry; ZEMOSAIC_ASTROMETRY_AVAILABLE = True; logger.info("Module 'zemosaic_astrometry' importé.")
except ImportError as e: logger.error(f"Import 'zemosaic_astrometry.py' échoué: {e}.")
try: import zemosaic_align_stack; ZEMOSAIC_ALIGN_STACK_AVAILABLE = True; logger.info("Module 'zemosaic_align_stack' importé.")
except ImportError as e: logger.error(f"Import 'zemosaic_align_stack.py' échoué: {e}.")
try:
    from .solver_settings import SolverSettings  # type: ignore
except ImportError:
    from solver_settings import SolverSettings  # type: ignore

try:
    from .lecropper import detect_autocrop_rgb as _anchor_detect_autocrop
except ImportError:
    try:
        from lecropper import detect_autocrop_rgb as _anchor_detect_autocrop
    except Exception:
        _anchor_detect_autocrop = None

ANCHOR_AUTOCROP_AVAILABLE = callable(_anchor_detect_autocrop)

# Optional configuration import for GPU toggle
try:
    import zemosaic_config
    ZEMOSAIC_CONFIG_AVAILABLE = True
except Exception:
    zemosaic_config = None  # type: ignore
    ZEMOSAIC_CONFIG_AVAILABLE = False

import importlib.util

# Global semaphore to throttle concurrent *.npy cache reads in Phase 3
_CACHE_IO_SEMAPHORE = threading.Semaphore(2 if os.name == 'nt' else 4)

# Global semaphore to limit concurrent Phase 3 (master tile) tasks.
# This allows runtime adaptation when other apps (e.g. a video read) are active.
# It is initialized later inside run_hierarchical_mosaic and can be reassigned
# by the runtime monitor to change the concurrency cap without restarting pools.
_PH3_CONCURRENCY_SEMAPHORE = threading.Semaphore(2 if os.name == 'nt' else 4)

# --- Basic IO throughput probing helpers (Windows-friendly, OS-agnostic) ---
def _measure_sequential_read_mbps(file_path: str, bytes_to_read: int = 16 * 1024 * 1024, block_size: int = 1 * 1024 * 1024) -> float | None:
    """Measure approximate sequential read speed on a single file.

    Returns MB/s or None on failure. Uses small sizes to avoid long stalls.
    """
    try:
        if not (file_path and os.path.exists(file_path)):
            return None
        size_target = max(block_size, bytes_to_read)
        read_total = 0
        t0 = time.perf_counter()
        with open(file_path, 'rb', buffering=0) as f:
            while read_total < size_target:
                chunk = f.read(min(block_size, size_target - read_total))
                if not chunk:
                    break
                read_total += len(chunk)
        dt = max(1e-6, time.perf_counter() - t0)
        return (read_total / (1024 * 1024)) / dt
    except Exception:
        return None


def _measure_sequential_write_mbps(dir_path: str, bytes_to_write: int = 16 * 1024 * 1024, block_size: int = 1 * 1024 * 1024) -> float | None:
    """Measure approximate sequential write speed in a directory.

    Writes and deletes a small temporary file. Returns MB/s or None on failure.
    """
    try:
        if not (dir_path and os.path.isdir(dir_path)):
            return None
        import uuid as _uuid
        tmp_path = os.path.join(dir_path, f"_zemosaic_io_probe_{_uuid.uuid4().hex}.bin")
        size_target = max(block_size, bytes_to_write)
        data = os.urandom(block_size)
        written_total = 0
        t0 = time.perf_counter()
        with open(tmp_path, 'wb', buffering=0) as f:
            while written_total < size_target:
                to_write = min(block_size, size_target - written_total)
                f.write(data[:to_write])
                written_total += to_write
            try:
                f.flush(); os.fsync(f.fileno())
            except Exception:
                pass
        dt = max(1e-6, time.perf_counter() - t0)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return (written_total / (1024 * 1024)) / dt
    except Exception:
        return None


def _categorize_io_speed(mbps: float | None) -> str:
    """Rough IO category string based on MB/s; conservative thresholds.

    very_slow: < 60 MB/s (typical USB HDD or spinning disk behind a hub)
    slow:      < 120 MB/s
    medium:    < 220 MB/s
    fast:      >= 220 MB/s
    """
    if mbps is None or mbps <= 0:
        return "unknown"
    if mbps < 60:
        return "very_slow"
    if mbps < 120:
        return "slow"
    if mbps < 220:
        return "medium"
    return "fast"

def gpu_is_available() -> bool:
    """Return True if CuPy and a CUDA device are available."""
    if not CUPY_AVAILABLE:
        return False
    try:
        import cupy
        return cupy.is_available()
    except Exception:
        return False

# Exposed compatibility flag expected by some tests
ASTROMETRY_SOLVER_AVAILABLE = ZEMOSAIC_ASTROMETRY_AVAILABLE

# progress_callback(stage: str, current: int, total: int)







# DANS zemosaic_worker.py

# ... (imports et logger configuré comme avant) ...

# --- Helper pour log et callback ---
def _log_and_callback(
    message_key_or_raw,
    progress_value=None,
    level="INFO",
    callback=None,
    **kwargs,
):
    """
    Helper pour loguer un message et appeler le callback GUI.
    - Si level est INFO, WARN, ERROR, SUCCESS, message_key_or_raw est traité comme une clé.
    - Sinon (DEBUG, ETA_LEVEL, etc.), message_key_or_raw est loggué tel quel.
    - Les **kwargs sont passés pour le formatage si message_key_or_raw est une clé.
    """
    # Support backwards compatibility for lvl/prog keyword aliases
    if "lvl" in kwargs and level == "INFO":
        level = kwargs.pop("lvl")
    elif "lvl" in kwargs:
        level = kwargs.pop("lvl")
    if "prog" in kwargs and progress_value is None:
        progress_value = kwargs.pop("prog")
    elif "prog" in kwargs:
        progress_value = kwargs.pop("prog")

    # Count alignment warnings for final summary
    if isinstance(message_key_or_raw, str) and message_key_or_raw in ALIGN_WARNING_COUNTS:
        ALIGN_WARNING_COUNTS[message_key_or_raw] += 1
    log_level_map = {
        "INFO": logging.INFO, "DEBUG": logging.DEBUG, "DEBUG_DETAIL": logging.DEBUG,
        "WARN": logging.WARNING, "ERROR": logging.ERROR, "SUCCESS": logging.INFO,
        "INFO_DETAIL": logging.DEBUG, 
        "ETA_LEVEL": logging.DEBUG, # Pour les messages ETA spécifiques
        "CHRONO_LEVEL": logging.DEBUG # Pour les commandes de chrono
    }
    
    level_str = "INFO" # Défaut
    if isinstance(level, str):
        level_str = level.upper()
    elif level is not None:
        logger.warning(f"_log_and_callback: Argument 'level' inattendu (type: {type(level)}, valeur: {level}). Utilisation de INFO par défaut.")

    # Préparer le message pour le logger Python interne
    final_message_for_py_logger = ""
    user_facing_log_levels = ["INFO", "WARN", "ERROR", "SUCCESS"]

    if level_str in user_facing_log_levels:
        # Pour ces niveaux, on s'attend à une clé. Logguer la clé et les args pour le debug interne.
        final_message_for_py_logger = f"[CLÉ_POUR_GUI: {message_key_or_raw}]"
        if kwargs:
            final_message_for_py_logger += f" (Args: {kwargs})"
    else: 
        # Pour les niveaux DEBUG, ETA, CHRONO, on loggue le message brut.
        # Si des kwargs sont passés avec un message brut (ex: debug), on peut essayer de le formater.
        final_message_for_py_logger = str(message_key_or_raw)
        if kwargs:
            try:
                final_message_for_py_logger = final_message_for_py_logger.format(**kwargs)
            except (KeyError, ValueError, IndexError) as fmt_err:
                logger.debug(f"Échec formatage message brut '{message_key_or_raw}' avec kwargs {kwargs} pour logger interne: {fmt_err}")
                # Garder le message brut si le formatage échoue

    logger.log(log_level_map.get(level_str, logging.INFO), final_message_for_py_logger)
    
    # Appel au callback GUI
    if callback and callable(callback):
        try:
            # On envoie la clé (ou le message brut) et les kwargs au callback GUI.
            # La GUI (sa méthode _log_message) sera responsable de faire la traduction
            # et le formatage final en utilisant ces kwargs si message_key_or_raw est une clé.
            #
            # La signature de _log_message dans la GUI doit être :
            # def _log_message(self, message_key_or_raw, progress_value=None, level="INFO", **kwargs):
            callback(message_key_or_raw, progress_value, level if isinstance(level, str) else "INFO", **kwargs)
        except Exception as e_cb:
            # Logguer l'erreur du callback, mais ne pas planter le worker pour ça
            logger.warning(f"Erreur dans progress_callback lors de l'appel depuis _log_and_callback: {e_cb}", exc_info=False)
            # Peut-être afficher la trace pour le debug du callback lui-même
            # logger.debug("Traceback de l'erreur du callback:", exc_info=True)




def _log_memory_usage(progress_callback: callable, context_message: str = ""): # Fonction helper définie ici ou globalement dans le module
    """Logue l'utilisation actuelle de la mémoire du processus et du système."""
    if not progress_callback or not callable(progress_callback):
        return
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)
        vms_mb = mem_info.vms / (1024 * 1024)

        virtual_mem = psutil.virtual_memory()
        available_ram_mb = virtual_mem.available / (1024 * 1024)
        total_ram_mb = virtual_mem.total / (1024 * 1024)
        percent_ram_used = virtual_mem.percent

        swap_mem = psutil.swap_memory()
        used_swap_mb = swap_mem.used / (1024 * 1024)
        total_swap_mb = swap_mem.total / (1024 * 1024)
        percent_swap_used = swap_mem.percent
        
        log_msg = (
            f"Memory Usage ({context_message}): "
            f"Proc RSS: {rss_mb:.1f}MB, VMS: {vms_mb:.1f}MB. "
            f"Sys RAM: Avail {available_ram_mb:.0f}MB / Total {total_ram_mb:.0f}MB ({percent_ram_used}%% used). "
            f"Sys Swap: Used {used_swap_mb:.0f}MB / Total {total_swap_mb:.0f}MB ({percent_swap_used}%% used)."
        )
        _log_and_callback(log_msg, prog=None, lvl="DEBUG", callback=progress_callback)
        
    except Exception as e_mem_log:
        _log_and_callback(f"Erreur lors du logging mémoire ({context_message}): {e_mem_log}", prog=None, lvl="WARN", callback=progress_callback)


def _log_alignment_warning_summary():
    """Write a summary of alignment warnings to the worker log."""
    total = sum(ALIGN_WARNING_COUNTS.values())
    if total == 0:
        logger.info("Alignment summary: no frames ignored due to errors.")
        return

    logger.info("===== Alignment warning summary =====")
    logger.info("Total frames ignored: %d", total)
    for key, count in ALIGN_WARNING_COUNTS.items():
        if count:
            human = ALIGN_WARNING_SUMMARY.get(key, key)
            logger.info("%d frame(s) - %s", count, human)


def _crop_array_to_signal(
    img: np.ndarray,
    coverage: np.ndarray | None = None,
    margin_frac: float = 0.05,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop ``img`` to the bounding box of useful signal.

    Parameters
    ----------
    img : np.ndarray
        2D/3D array containing image data.
    coverage : np.ndarray | None, optional
        Optional coverage map used as mask (>0 considered valid).
    margin_frac : float, optional
        Additional fractional margin added to each side of the bounding box.

    Returns
    -------
    tuple[np.ndarray, tuple[int, int, int, int]]
        Cropped image and bounding box ``(y0, y1, x0, x1)``.
    """

    if img is None:
        return img, (0, 0, 0, 0)

    arr = np.asarray(img)
    if arr.ndim < 2:
        height = int(arr.shape[0]) if arr.ndim >= 1 else 0
        width = int(arr.shape[1]) if arr.ndim > 1 else 0
        return img, (0, height, 0, width)

    height, width = int(arr.shape[0]), int(arr.shape[1])
    default_bbox = (0, height, 0, width)

    mask: np.ndarray | None = None
    if coverage is not None:
        try:
            cov_arr = np.asarray(coverage)
            if cov_arr.shape[0] == height and cov_arr.shape[1] == width:
                mask = cov_arr > 0
        except Exception:
            mask = None

    if mask is None:
        data_arr = np.asarray(img)
        if data_arr.ndim == 3:
            valid_pixels = np.any(np.isfinite(data_arr) & (data_arr != 0), axis=-1)
        else:
            valid_pixels = np.isfinite(data_arr) & (data_arr != 0)
        mask = valid_pixels

    if not np.any(mask):
        return img, default_bbox

    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img, default_bbox

    y_min, y_max = int(rows[0]), int(rows[-1]) + 1
    x_min, x_max = int(cols[0]), int(cols[-1]) + 1

    try:
        margin_frac = float(margin_frac)
    except (TypeError, ValueError):
        margin_frac = 0.0
    margin_frac = max(0.0, margin_frac)

    if margin_frac > 0.0:
        bbox_height = y_max - y_min
        bbox_width = x_max - x_min
        margin_y = int(math.ceil(bbox_height * margin_frac))
        margin_x = int(math.ceil(bbox_width * margin_frac))
        y_min = max(0, y_min - margin_y)
        y_max = min(height, y_max + margin_y)
        x_min = max(0, x_min - margin_x)
        x_max = min(width, x_max + margin_x)

    bbox = (y_min, y_max, x_min, x_max)
    cropped = img[y_min:y_max, x_min:x_max, ...]

    return cropped, bbox


def _auto_crop_mosaic_to_valid_region(
    mosaic: np.ndarray,
    coverage: np.ndarray | None,
    output_wcs,
    log_callback=None,
    threshold: float = 1e-6,
    *,
    follow_signal: bool | None = None,
    margin_frac: float | None = 0.05,
):
    """Crop blank borders from the mosaic using the coverage map.

    Parameters
    ----------
    mosaic : np.ndarray
        Final stacked mosaic with shape ``(H, W, C)``.
    coverage : np.ndarray | None
        Coverage/weight map returned by ``reproject_and_coadd``.
    output_wcs : astropy.wcs.WCS | Any
        WCS object describing the mosaic; will be updated in-place if cropping occurs.
    log_callback : callable | None
        Optional callback used to emit log messages (same signature as ``_pcb``).
    threshold : float
        Minimum coverage value considered as valid data when computing the crop bounds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        Cropped mosaic and coverage arrays. If no cropping is necessary the
        original inputs are returned unchanged.
    """

    if mosaic is None:
        return mosaic, coverage

    mosaic_arr = np.asarray(mosaic)
    if mosaic_arr.ndim < 2:
        return mosaic, coverage

    default_bbox = (0, int(mosaic_arr.shape[0]), 0, int(mosaic_arr.shape[1]))

    if follow_signal is None:
        try:
            import zemosaic_config

            cfg = zemosaic_config.load_config() or {}
            follow_signal = bool(cfg.get("crop_follow_signal", False))
        except Exception:
            follow_signal = False
    else:
        follow_signal = bool(follow_signal)

    try:
        margin_value = 0.05 if margin_frac is None else float(margin_frac)
    except (TypeError, ValueError):
        margin_value = 0.05
    margin_value = max(0.0, margin_value)

    bbox = default_bbox
    cropped_mosaic = mosaic
    cropped_coverage = coverage
    used_signal_crop = False

    if follow_signal:
        try:
            candidate_mosaic, candidate_bbox = _crop_array_to_signal(
                mosaic,
                coverage,
                margin_value,
            )
            if candidate_bbox:
                bbox = candidate_bbox
                used_signal_crop = True
                if bbox != default_bbox:
                    cropped_mosaic = candidate_mosaic
                    if coverage is not None:
                        y0, y1, x0, x1 = bbox
                        cropped_coverage = coverage[y0:y1, x0:x1]
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("follow_signal crop applied, bbox=%s", bbox)
        except Exception:
            bbox = default_bbox
            used_signal_crop = False

    if used_signal_crop and bbox == default_bbox:
        return mosaic, coverage

    if not used_signal_crop:
        if coverage is None:
            return mosaic, coverage

        try:
            cov_array = np.asarray(coverage)
        except Exception:
            cov_array = coverage

        if getattr(cov_array, "ndim", 0) != 2:
            return mosaic, coverage

        try:
            valid_mask = np.asarray(cov_array) > float(threshold)
        except Exception:
            return mosaic, coverage

        if not np.any(valid_mask):
            return mosaic, coverage

        rows = np.where(np.any(valid_mask, axis=1))[0]
        cols = np.where(np.any(valid_mask, axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return mosaic, coverage

        y_min, y_max = int(rows[0]), int(rows[-1]) + 1
        x_min, x_max = int(cols[0]), int(cols[-1]) + 1

        bbox = (y_min, y_max, x_min, x_max)

        if (
            y_min == 0
            and x_min == 0
            and y_max == mosaic.shape[0]
            and x_max == mosaic.shape[1]
        ):
            return mosaic, coverage

        cropped_mosaic = mosaic[y_min:y_max, x_min:x_max, ...]
        cropped_coverage = coverage[y_min:y_max, x_min:x_max]

    if used_signal_crop and bbox == default_bbox:
        # Signal crop requested but no bounding box reduction occurred.
        return mosaic, coverage

    new_shape = tuple(int(v) for v in np.shape(cropped_mosaic))

    y_min, y_max, x_min, x_max = bbox

    if callable(log_callback):
        try:
            log_callback(
                "ASM_REPROJ_COADD: Auto-cropped output to coverage bounds",
                prog=None,
                lvl="INFO_DETAIL",
                y_bounds=f"{y_min}:{y_max}",
                x_bounds=f"{x_min}:{x_max}",
                new_shape=str(new_shape),
            )
        except Exception:
            pass

    try:
        if hasattr(output_wcs, "wcs") and getattr(output_wcs, "wcs") is not None:
            if hasattr(output_wcs.wcs, "crpix") and output_wcs.wcs.crpix is not None:
                output_wcs.wcs.crpix[0] -= float(x_min)
                output_wcs.wcs.crpix[1] -= float(y_min)
            if hasattr(output_wcs.wcs, "naxis1"):
                output_wcs.wcs.naxis1 = int(new_shape[1])
            if hasattr(output_wcs.wcs, "naxis2"):
                output_wcs.wcs.naxis2 = int(new_shape[0])
    except Exception:
        pass

    for attr, val in (
        ("pixel_shape", (int(new_shape[1]), int(new_shape[0]))),
        ("array_shape", (int(new_shape[0]), int(new_shape[1]))),
    ):
        if hasattr(output_wcs, attr):
            try:
                setattr(output_wcs, attr, val)
            except Exception:
                pass

    return cropped_mosaic, cropped_coverage


def _wait_for_memmap_files(prefixes, timeout=10.0):
    """Poll until each prefix.dat and prefix.npy exist and are non-empty."""
    import time, os
    start = time.time()
    while True:
        all_ready = True
        for prefix in prefixes:
            dat_f = prefix + '.dat'
            npy_f = prefix + '.npy'
            if not (os.path.exists(dat_f) and os.path.getsize(dat_f) > 0 and os.path.exists(npy_f) and os.path.getsize(npy_f) > 0):
                all_ready = False
                break
        if all_ready:
            return
        if time.time() - start > timeout:
            raise RuntimeError(f"Memmap file not ready after {timeout}s: {prefix}")


def astap_paths_valid(astap_exe_path: str, astap_data_dir: str) -> bool:
    """Return True if ASTAP executable and data directory look valid."""
    return (
        astap_exe_path
        and os.path.isfile(astap_exe_path)
        and astap_data_dir
        and os.path.isdir(astap_data_dir)
    )


def _write_header_to_fits(file_path: str, header_obj, pcb=None):
    """Safely update ``file_path`` FITS header with ``header_obj`` if possible."""
    if not (ASTROPY_AVAILABLE and fits):
        return
    try:
        with fits.open(file_path, mode="update", memmap=False) as hdul:
            hdul[0].header.update(header_obj)
            hdul.flush()
        if pcb:
            pcb("getwcs_info_header_written", lvl="DEBUG_DETAIL", filename=os.path.basename(file_path))
    except Exception as e_update:
        if pcb:
            pcb("getwcs_warn_header_write_failed", lvl="WARN", filename=os.path.basename(file_path), error=str(e_update))


def solve_with_astrometry(
    image_fits_path: str,
    fits_header,
    settings: dict | None,
    progress_callback=None,
):
    """Attempt plate solving via the Astrometry.net service."""

    if not ASTROMETRY_SOLVER_AVAILABLE:
        return None

    try:
        from . import zemosaic_astrometry
    except Exception:
        return None

    solver_dict = settings or {}
    api_key = solver_dict.get("api_key", "")
    timeout = solver_dict.get("timeout")
    down = solver_dict.get("downsample")

    try:
        return zemosaic_astrometry.solve_with_astrometry_net(
            image_fits_path,
            fits_header,
            api_key=api_key,
            timeout_sec=timeout or 60,
            downsample_factor=down,
            update_original_header_in_place=True,
            progress_callback=progress_callback,
        )
    except Exception as e:
        _log_and_callback(
            f"Astrometry solve error: {e}", prog=None, lvl="WARN", callback=progress_callback
        )
        return None


def solve_with_ansvr(
    image_fits_path: str,
    fits_header,
    settings: dict | None,
    progress_callback=None,
):
    """Attempt plate solving using a local ansvr installation."""

    if not ASTROMETRY_SOLVER_AVAILABLE:
        return None

    try:
        from . import zemosaic_astrometry
    except Exception:
        return None

    solver_dict = settings or {}
    path = solver_dict.get("ansvr_path") or solver_dict.get("astrometry_local_path") or solver_dict.get("local_ansvr_path")
    timeout = solver_dict.get("ansvr_timeout") or solver_dict.get("timeout")

    try:
        return zemosaic_astrometry.solve_with_ansvr(
            image_fits_path,
            fits_header,
            ansvr_config_path=path or "",
            timeout_sec=timeout or 120,
            update_original_header_in_place=True,
            progress_callback=progress_callback,
        )
    except Exception as e:
        _log_and_callback(
            f"Ansvr solve error: {e}", prog=None, lvl="WARN", callback=progress_callback
        )
        return None


# Note: Ancienne fonction _prepare_image_for_astap supprimée. Les images sont
# passées à ASTAP telles quelles pour la résolution (pas de conversion mono).


def reproject_tile_to_mosaic(
    tile_path: str,
    tile_wcs,
    mosaic_wcs,
    mosaic_shape_hw,
    feather: bool = True,
    apply_crop: bool = False,
    crop_percent: float = 0.0,
    tile_affine: tuple[float, float] | None = None,
    gain: float | None = None,
    offset: float | None = None,
    match_background: bool = True,
    nan_fill_value: float = 0.0,
    enforce_positive: bool = True,
):
    """Reprojecte une tuile sur la grille finale et renvoie l'image et sa carte
    de poids ainsi que la bounding box utile.

    Les bornes sont retournées dans l'ordre ``(xmin, xmax, ymin, ymax)`` afin
    de correspondre aux indices ``[ligne, colonne]`` lors de l'incrémentation
    sur la mosaïque.

    ``tile_wcs`` et ``mosaic_wcs`` peuvent être soit des objets :class:`WCS`
    directement, soit des en-têtes FITS (``dict`` ou :class:`~astropy.io.fits.Header``).
    Cela permet d'utiliser cette fonction avec :class:`concurrent.futures.ProcessPoolExecutor`
    où les arguments doivent être sérialisables.
    """
    if not (REPROJECT_AVAILABLE and reproject_interp and ASTROPY_AVAILABLE and fits):
        return None, None, (0, 0, 0, 0)

    # Les objets WCS ne sont pas toujours sérialisables via multiprocessing.
    # Si on reçoit des en-têtes (dict ou fits.Header), reconstruire les WCS ici.
    if ASTROPY_AVAILABLE and WCS:
        if not isinstance(tile_wcs, WCS):
            try:
                tile_wcs = WCS(tile_wcs)
            except Exception:
                return None, None, (0, 0, 0, 0)
        if not isinstance(mosaic_wcs, WCS):
            try:
                mosaic_wcs = WCS(mosaic_wcs)
            except Exception:
                return None, None, (0, 0, 0, 0)

    with fits.open(tile_path, memmap=False) as hdul:
        raw_data = hdul[0].data

    try:
        data = _ensure_hwc_master_tile(raw_data, os.path.basename(tile_path))
    except ValueError:
        return None, None, (0, 0, 0, 0)
    n_channels = data.shape[-1]

    if gain is None or offset is None:
        if tile_affine is not None:
            try:
                gain, offset = tile_affine
            except Exception:
                gain, offset = None, None

    if gain is not None and offset is not None:
        try:
            gain_val = float(gain)
        except Exception:
            gain_val = 1.0
        try:
            offset_val = float(offset)
        except Exception:
            offset_val = 0.0
        if not np.isfinite(gain_val):
            gain_val = 1.0
        if not np.isfinite(offset_val):
            offset_val = 0.0
        if gain_val != 1.0:
            data *= gain_val
        if offset_val != 0.0:
            data += offset_val

    # Optional cropping of the tile before reprojection
    if apply_crop and crop_percent > 1e-3 and ZEMOSAIC_UTILS_AVAILABLE \
            and hasattr(zemosaic_utils, "crop_image_and_wcs"):
        try:
            cropped, cropped_wcs = zemosaic_utils.crop_image_and_wcs(
                data,
                tile_wcs,
                crop_percent / 100.0,
                progress_callback=None,
            )
            if cropped is not None and cropped_wcs is not None:
                data = cropped
                tile_wcs = cropped_wcs
                n_channels = data.shape[-1]
        except Exception:
            pass

    if data.ndim != 3:
        raise ValueError(f"Expected HWC data after normalization, got shape {data.shape}")

    base_weight = np.ones(data.shape[:2], dtype=np.float32)
    if (
        feather
        and ZEMOSAIC_UTILS_AVAILABLE
        and hasattr(zemosaic_utils, "make_radial_weight_map")
    ):
        try:
            base_weight = zemosaic_utils.make_radial_weight_map(
                data.shape[0],
                data.shape[1],
                feather_fraction=0.92,
                min_weight_floor=0.10,
            )
            logger.debug("Feather applied with min_weight_floor=0.10")
        except Exception:
            base_weight = np.ones(data.shape[:2], dtype=np.float32)

    # --- Determine bounding box covered by the tile on the mosaic
    footprint_full, _ = reproject_interp(
        (base_weight, tile_wcs),
        mosaic_wcs,
        shape_out=mosaic_shape_hw,
        order='nearest-neighbor',  # suffit, c'est binaire
        parallel=False,
    )

    j_idx, i_idx = np.where(footprint_full > 0)
    if j_idx.size == 0:
        return None, None, (0, 0, 0, 0)

    j0, j1 = int(j_idx.min()), int(j_idx.max()) + 1
    i0, i1 = int(i_idx.min()), int(i_idx.max()) + 1
    h, w = j1 - j0, i1 - i0

    # Create a WCS for the sub-region
    try:
        sub_wcs = mosaic_wcs.deepcopy()
        sub_wcs.wcs.crpix = [mosaic_wcs.wcs.crpix[0] - i0, mosaic_wcs.wcs.crpix[1] - j0]
    except Exception:
        sub_wcs = mosaic_wcs

    # Allocate arrays only for the useful area
    reproj_img = np.zeros((h, w, n_channels), dtype=np.float32)
    reproj_weight = np.zeros((h, w), dtype=np.float32)

    for c in range(n_channels):
        reproj_c, footprint = reproject_interp(
            (data[..., c], tile_wcs),
            sub_wcs,
            shape_out=(h, w),
            order='bilinear',
            parallel=False,
        )

        w_reproj, _ = reproject_interp(
            (base_weight, tile_wcs),
            sub_wcs,
            shape_out=(h, w),
            order='bilinear',
            parallel=False,
        )

        total_w = footprint * w_reproj
        reproj_img[..., c] = reproj_c.astype(np.float32)
        reproj_weight += total_w.astype(np.float32)

    valid = reproj_weight > 0
    if not np.any(valid):
        return None, None, (0, 0, 0, 0)

    # Normalisation d'arrière-plan optionnelle (match_background)
    if match_background:
        try:
            for c in range(n_channels):
                channel_view = reproj_img[..., c]
                med_c = np.nanmedian(channel_view[valid])
                if np.isfinite(med_c):
                    channel_view -= med_c
        except Exception:
            pass

    if nan_fill_value is not None:
        np.nan_to_num(
            reproj_img,
            copy=False,
            nan=nan_fill_value,
            posinf=nan_fill_value,
            neginf=nan_fill_value,
        )
    if enforce_positive:
        np.clip(reproj_img, 0.0, None, out=reproj_img)

    # Les indices sont retournés dans l'ordre (xmin, xmax, ymin, ymax)
    return reproj_img, reproj_weight, (i0, i1, j0, j1)




# --- Fonctions Utilitaires Internes au Worker ---
def _calculate_final_mosaic_grid(panel_wcs_list: list, panel_shapes_hw_list: list,
                                 drizzle_scale_factor: float = 1.0, progress_callback: callable = None):
    num_initial_inputs = len(panel_wcs_list)
    # Utilisation de clés pour les messages utilisateur
    _log_and_callback("calcgrid_info_start_calc", num_wcs_shapes=num_initial_inputs, scale_factor=drizzle_scale_factor, level="DEBUG_DETAIL", callback=progress_callback)
    
    if not REPROJECT_AVAILABLE:
        _log_and_callback("calcgrid_error_reproject_unavailable", level="ERROR", callback=progress_callback)
        return None, None
    if find_optimal_celestial_wcs is None:
        if CALC_GRID_OPTIMIZED_AVAILABLE and _calculate_final_mosaic_grid_optimized:
            _log_and_callback(
                "calcgrid_warn_find_optimal_celestial_wcs_missing",
                level="WARN",
                callback=progress_callback,
            )
            return _calculate_final_mosaic_grid_optimized(
                panel_wcs_list, panel_shapes_hw_list, drizzle_scale_factor
            )
        _log_and_callback("calcgrid_error_reproject_unavailable", level="ERROR", callback=progress_callback)
        return None, None
    if not (ASTROPY_AVAILABLE and u and Angle):
        _log_and_callback("calcgrid_error_astropy_unavailable", level="ERROR", callback=progress_callback); return None, None
    if num_initial_inputs == 0:
        _log_and_callback("calcgrid_error_no_wcs_shape", level="ERROR", callback=progress_callback); return None, None

    valid_wcs_inputs = []; valid_shapes_inputs_hw = []
    for idx_filt, wcs_filt in enumerate(panel_wcs_list):
        if isinstance(wcs_filt, WCS) and wcs_filt.is_celestial:
            if idx_filt < len(panel_shapes_hw_list):
                shape_filt = panel_shapes_hw_list[idx_filt]
                if isinstance(shape_filt, tuple) and len(shape_filt) == 2 and isinstance(shape_filt[0], int) and shape_filt[0] > 0 and isinstance(shape_filt[1], int) and shape_filt[1] > 0:
                    valid_wcs_inputs.append(wcs_filt); valid_shapes_inputs_hw.append(shape_filt)
                else: _log_and_callback("calcgrid_warn_invalid_shape_skipped", shape=shape_filt, wcs_index=idx_filt, level="WARN", callback=progress_callback)
            else: _log_and_callback("calcgrid_warn_no_shape_for_wcs_skipped", wcs_index=idx_filt, level="WARN", callback=progress_callback)
        else: _log_and_callback("calcgrid_warn_invalid_wcs_skipped", wcs_index=idx_filt, level="WARN", callback=progress_callback)
    
    if not valid_wcs_inputs:
        _log_and_callback("calcgrid_error_no_valid_wcs_shape_after_filter", level="ERROR", callback=progress_callback); return None, None

    panel_wcs_list_to_use = valid_wcs_inputs; panel_shapes_hw_list_to_use = valid_shapes_inputs_hw
    num_valid_inputs = len(panel_wcs_list_to_use)
    _log_and_callback(f"CalcGrid: {num_valid_inputs} WCS/Shapes valides pour calcul.", None, "DEBUG", progress_callback) # Log technique

    inputs_for_optimal_wcs_calc = []
    for i in range(num_valid_inputs):
        wcs_in = panel_wcs_list_to_use[i]
        shape_in_hw = panel_shapes_hw_list_to_use[i] # shape (height, width)
        shape_in_wh_for_wcs_pixel_shape = (shape_in_hw[1], shape_in_hw[0]) # (width, height) for WCS.pixel_shape

        # Ensure WCS.pixel_shape is set for reproject, it might use it internally.
        if wcs_in.pixel_shape is None or wcs_in.pixel_shape != shape_in_wh_for_wcs_pixel_shape:
            try: 
                wcs_in.pixel_shape = shape_in_wh_for_wcs_pixel_shape
                _log_and_callback(f"CalcGrid: WCS {i} pixel_shape set to {shape_in_wh_for_wcs_pixel_shape}", None, "DEBUG_DETAIL", progress_callback)
            except Exception as e_pshape_set: 
                _log_and_callback("calcgrid_warn_set_pixel_shape_failed", wcs_index=i, error=str(e_pshape_set), level="WARN", callback=progress_callback)
        
        # **** LA CORRECTION EST ICI ****
        # find_optimal_celestial_wcs expects a list of (shape, wcs) tuples or HDU objects.
        # The shape should be (height, width).
        inputs_for_optimal_wcs_calc.append((shape_in_hw, wcs_in))
        # *****************************

    if not inputs_for_optimal_wcs_calc:
        _log_and_callback("calcgrid_error_no_wcs_for_optimal_calc", level="ERROR", callback=progress_callback); return None, None
        
    try:
        sum_of_pixel_scales_deg = 0.0; count_of_valid_scales = 0
        # For calculating average input pixel scale, we use panel_wcs_list_to_use (which are just WCS objects)
        for wcs_obj_scale in panel_wcs_list_to_use: 
            if not (wcs_obj_scale and wcs_obj_scale.is_celestial): continue
            try:
                current_pixel_scale_deg = 0.0
                if hasattr(wcs_obj_scale, 'proj_plane_pixel_scales') and callable(wcs_obj_scale.proj_plane_pixel_scales):
                    pixel_scales_angle_tuple = wcs_obj_scale.proj_plane_pixel_scales(); current_pixel_scale_deg = np.mean(np.abs([s.to_value(u.deg) for s in pixel_scales_angle_tuple]))
                elif hasattr(wcs_obj_scale, 'pixel_scale_matrix'): current_pixel_scale_deg = np.sqrt(np.abs(np.linalg.det(wcs_obj_scale.pixel_scale_matrix)))
                else: continue
                if np.isfinite(current_pixel_scale_deg) and current_pixel_scale_deg > 1e-10: sum_of_pixel_scales_deg += current_pixel_scale_deg; count_of_valid_scales += 1
            except Exception: pass # Ignore errors in calculating scale for one WCS
        
        avg_input_pixel_scale_deg = (2.0 / 3600.0) # Fallback 2 arcsec/pix
        if count_of_valid_scales > 0: avg_input_pixel_scale_deg = sum_of_pixel_scales_deg / count_of_valid_scales
        elif num_valid_inputs > 0 : _log_and_callback("calcgrid_warn_scale_fallback", level="WARN", callback=progress_callback)
        
        target_resolution_deg_per_pixel = avg_input_pixel_scale_deg / drizzle_scale_factor
        target_resolution_angle = Angle(target_resolution_deg_per_pixel, unit=u.deg)
        _log_and_callback("calcgrid_info_scales", avg_input_scale_arcsec=avg_input_pixel_scale_deg*3600, target_scale_arcsec=target_resolution_angle.arcsec, level="INFO", callback=progress_callback)
        
        # Now call with inputs_for_optimal_wcs_calc which is a list of (shape_hw, wcs) tuples
        optimal_wcs_out, optimal_shape_hw_out = find_optimal_celestial_wcs(
            inputs_for_optimal_wcs_calc, # This is now a list of (shape_hw, WCS) tuples
            resolution=target_resolution_angle, 
            auto_rotate=True, 
            projection='TAN', 
            reference=None, 
            frame='icrs'
        )
        
        if optimal_wcs_out and optimal_shape_hw_out:
            expected_pixel_shape_wh_for_wcs_out = (optimal_shape_hw_out[1], optimal_shape_hw_out[0])
            if optimal_wcs_out.pixel_shape is None or optimal_wcs_out.pixel_shape != expected_pixel_shape_wh_for_wcs_out:
                try: optimal_wcs_out.pixel_shape = expected_pixel_shape_wh_for_wcs_out
                except Exception: pass
            if not (hasattr(optimal_wcs_out.wcs, 'naxis1') and hasattr(optimal_wcs_out.wcs, 'naxis2')) or not (optimal_wcs_out.wcs.naxis1 > 0 and optimal_wcs_out.wcs.naxis2 > 0) :
                try: optimal_wcs_out.wcs.naxis1 = expected_pixel_shape_wh_for_wcs_out[0]; optimal_wcs_out.wcs.naxis2 = expected_pixel_shape_wh_for_wcs_out[1]
                except Exception: pass
        
        _log_and_callback("calcgrid_info_optimal_grid_calculated", shape=optimal_shape_hw_out, crval=optimal_wcs_out.wcs.crval if optimal_wcs_out and optimal_wcs_out.wcs else 'N/A', level="INFO", callback=progress_callback)
        return optimal_wcs_out, optimal_shape_hw_out
    except ImportError: _log_and_callback("calcgrid_error_find_optimal_wcs_unavailable", level="ERROR", callback=progress_callback); return None, None
    except Exception as e_optimal_wcs_call: 
        _log_and_callback("calcgrid_error_find_optimal_wcs_call", error=str(e_optimal_wcs_call), level="ERROR", callback=progress_callback)
        logger.error("Traceback find_optimal_celestial_wcs:", exc_info=True)
        return None, None


def cluster_seestar_stacks(all_raw_files_with_info: list, stack_threshold_deg: float, progress_callback: callable):
    """Group raw files captured by the Seestar based on their WCS position."""

    if not (ASTROPY_AVAILABLE and SkyCoord and u):
        _log_and_callback("clusterstacks_error_astropy_unavailable", level="ERROR", callback=progress_callback)
        return []

    if not all_raw_files_with_info:
        _log_and_callback("clusterstacks_warn_no_raw_info", level="WARN", callback=progress_callback)
        return []

    _log_and_callback(
        "clusterstacks_info_start",
        num_files=len(all_raw_files_with_info),
        threshold=stack_threshold_deg,
        level="INFO",
        callback=progress_callback,
    )

    panel_centers_sky = []
    panel_data_for_clustering = []

    for i, info in enumerate(all_raw_files_with_info):
        wcs_obj = info["wcs"]
        if not (wcs_obj and wcs_obj.is_celestial):
            continue
        try:
            if wcs_obj.pixel_shape:
                center_world = wcs_obj.pixel_to_world(
                    wcs_obj.pixel_shape[0] / 2.0,
                    wcs_obj.pixel_shape[1] / 2.0,
                )
            elif hasattr(wcs_obj.wcs, "crval"):
                center_world = SkyCoord(
                    ra=wcs_obj.wcs.crval[0] * u.deg,
                    dec=wcs_obj.wcs.crval[1] * u.deg,
                    frame="icrs",
                )
            else:
                continue
            panel_centers_sky.append(center_world)
            panel_data_for_clustering.append(info)
        except Exception:
            continue

    if not panel_centers_sky:
        _log_and_callback("clusterstacks_warn_no_centers", level="WARN", callback=progress_callback)
        return []

    groups = []
    assigned_mask = [False] * len(panel_centers_sky)

    for i in range(len(panel_centers_sky)):
        if assigned_mask[i]:
            continue
        current_group_infos = [panel_data_for_clustering[i]]
        assigned_mask[i] = True
        current_group_center_seed = panel_centers_sky[i]
        for j in range(i + 1, len(panel_centers_sky)):
            if assigned_mask[j]:
                continue
            if current_group_center_seed.separation(panel_centers_sky[j]).deg < stack_threshold_deg:
                current_group_infos.append(panel_data_for_clustering[j])
                assigned_mask[j] = True
        groups.append(current_group_infos)

    _log_and_callback("clusterstacks_info_finished", num_groups=len(groups), level="INFO", callback=progress_callback)
    return groups

def get_wcs_and_pretreat_raw_file(
    file_path: str,
    astap_exe_path: str,
    astap_data_dir: str,
    astap_search_radius: float,
    astap_downsample: int,
    astap_sensitivity: int,
    astap_timeout_seconds: int,
    progress_callback: callable,
    hotpix_mask_dir: str | None = None,
    solver_settings: dict | None = None,
):
    filename = os.path.basename(file_path)
    # Utiliser une fonction helper pour les logs internes à cette fonction si _log_and_callback
    # est trop lié à la structure de run_hierarchical_mosaic
    _pcb_local = lambda msg_key, lvl="DEBUG", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else print(f"GETWCS_LOG {lvl}: {msg_key} {kwargs}")

    try:
        if is_path_excluded(file_path, EXCLUDED_DIRS):
            logger.debug("Skip excluded path: %s", file_path)
            return None, None, None, None
    except Exception:
        if UNALIGNED_DIRNAME in os.path.normpath(file_path).split(os.sep):
            logger.debug("Skip excluded path: %s", file_path)
            return None, None, None, None

    if solver_settings is None:
        solver_settings = {}
    elif not isinstance(solver_settings, dict):
        try:
            solver_settings = dict(solver_settings)
        except Exception:
            solver_settings = getattr(solver_settings, "__dict__", {}) or {}
            if not isinstance(solver_settings, dict):
                solver_settings = {}

    header_precheck = None
    preexisting_wcs_flag = False
    preexisting_wcs_failure_reason = None
    if ASTROPY_AVAILABLE and fits is not None:
        try:
            with fits.open(file_path, mode="readonly", memmap=False) as hdul_hdr:
                header_precheck = hdul_hdr[0].header.copy()
        except Exception:
            header_precheck = None
        else:
            if hasattr(zemosaic_utils, "validate_wcs_header"):
                try:
                    preexisting_wcs_flag, _, failure_reason = zemosaic_utils.validate_wcs_header(header_precheck)
                except Exception as exc_validate:
                    preexisting_wcs_flag = False
                    failure_reason = f"validate_exception: {exc_validate}"
                if not preexisting_wcs_flag:
                    preexisting_wcs_failure_reason = failure_reason
            elif hasattr(zemosaic_utils, "has_valid_wcs"):
                try:
                    preexisting_wcs_flag = bool(zemosaic_utils.has_valid_wcs(header_precheck))
                except Exception:
                    preexisting_wcs_flag = False

    # Charger configuration pour options de prétraitement (si disponible)
    _cfg_pre = {}
    try:
        if ZEMOSAIC_CONFIG_AVAILABLE and zemosaic_config:
            _cfg_pre = zemosaic_config.load_config() or {}
    except Exception:
        _cfg_pre = {}
    _bg_gpu_enabled = bool(_cfg_pre.get("preprocess_remove_background_gpu", False))
    _bg_sigma = float(_cfg_pre.get("preprocess_background_sigma", 24.0))
    force_resolve_existing_wcs_cfg = bool(_cfg_pre.get("force_resolve_existing_wcs", False))
    force_resolve_existing_wcs = bool(
        solver_settings.get("force_resolve_existing_wcs", force_resolve_existing_wcs_cfg)
    )
    try:
        affine_offset_limit_adu = float(solver_settings.get("intertile_offset_limit_adu", 50.0))
    except Exception:
        affine_offset_limit_adu = 50.0
    affine_offset_limit_adu = max(0.0, abs(affine_offset_limit_adu))
    gain_limits_cfg = solver_settings.get("intertile_gain_limits")
    if isinstance(gain_limits_cfg, (list, tuple)) and len(gain_limits_cfg) == 2:
        try:
            gain_limit_min = float(gain_limits_cfg[0])
            gain_limit_max = float(gain_limits_cfg[1])
        except Exception:
            gain_limit_min, gain_limit_max = 0.75, 1.25
    else:
        gain_limit_min, gain_limit_max = 0.75, 1.25
    if gain_limit_min > gain_limit_max:
        gain_limit_min, gain_limit_max = gain_limit_max, gain_limit_min

    _pcb_local(f"GetWCS_Pretreat: Début pour '{filename}'.", lvl="DEBUG_DETAIL") # Niveau DEBUG_DETAIL pour être moins verbeux

    hp_mask_path = None

    if not (ZEMOSAIC_UTILS_AVAILABLE and zemosaic_utils):
        _pcb_local("getwcs_error_utils_unavailable", lvl="ERROR")
        return None, None, None, None
        
    res_load = zemosaic_utils.load_and_validate_fits(
        file_path,
        normalize_to_float32=False,
        attempt_fix_nonfinite=True,
        progress_callback=progress_callback,
    )
    if isinstance(res_load, tuple):
        img_data_raw_adu = res_load[0]
        header_orig = res_load[1] if len(res_load) > 1 else None
    else:
        img_data_raw_adu = res_load
        header_orig = None

    if header_orig is None and header_precheck is not None:
        try:
            header_orig = header_precheck.copy()
        except Exception:
            header_orig = copy.deepcopy(header_precheck)

    if img_data_raw_adu is None or header_orig is None:
        _pcb_local("getwcs_error_load_failed", lvl="ERROR", filename=filename)
        # Le fichier n'a pas pu être chargé, on ne peut pas le déplacer car on ne sait pas s'il existe ou est corrompu.
        # Ou on pourrait essayer de le déplacer s'il existe. Pour l'instant, on retourne None.
        return None, None, None, None

    # ... (log de post-load) ...
    _pcb_local(f"  Post-Load: '{filename}' - Shape: {img_data_raw_adu.shape}, Dtype: {img_data_raw_adu.dtype}", lvl="DEBUG_VERY_DETAIL")

    img_data_processed_adu = img_data_raw_adu.astype(np.float32, copy=True)

    # --- Débayerisation ---
    if img_data_processed_adu.ndim == 2:
        _pcb_local(f"  Monochrome détecté pour '{filename}'. Débayerisation...", lvl="DEBUG_DETAIL")
        bayer_pattern = header_orig.get('BAYERPAT', header_orig.get('CFAIMAGE', 'GRBG'))
        if not isinstance(bayer_pattern, str) or bayer_pattern.upper() not in ['GRBG','RGGB','GBRG','BGGR']: bayer_pattern = 'GRBG'
        else: bayer_pattern = bayer_pattern.upper()
        
        bitpix = header_orig.get('BITPIX', 16)
        # ... (logique de max_val_for_norm_before_debayer inchangée) ...
        max_val_for_norm_before_debayer = (2**abs(bitpix))-1. if bitpix!=0 and np.issubdtype(img_data_processed_adu.dtype,np.integer) else (65535. if np.issubdtype(img_data_processed_adu.dtype,np.unsignedinteger) else 1.)
        if abs(bitpix)>16 and np.issubdtype(img_data_processed_adu.dtype,np.integer): max_val_for_norm_before_debayer=(2**16)-1.
        if max_val_for_norm_before_debayer<=0: max_val_for_norm_before_debayer=1.

        img_norm_for_debayer = np.zeros_like(img_data_processed_adu,dtype=np.float32)
        min_adu_pre_debayer,max_adu_pre_debayer=np.nanmin(img_data_processed_adu),np.nanmax(img_data_processed_adu)
        range_adu_pre_debayer=max_adu_pre_debayer-min_adu_pre_debayer
        if range_adu_pre_debayer>1e-9: img_norm_for_debayer=(img_data_processed_adu-min_adu_pre_debayer)/range_adu_pre_debayer
        elif np.any(np.isfinite(img_data_processed_adu)): img_norm_for_debayer=np.full_like(img_data_processed_adu,0.5)
        img_norm_for_debayer=np.clip(img_norm_for_debayer,0.,1.)
        
        try:
            img_rgb_norm_01 = zemosaic_utils.debayer_image(img_norm_for_debayer, bayer_pattern, progress_callback=progress_callback)
            if range_adu_pre_debayer>1e-9: img_data_processed_adu=(img_rgb_norm_01*range_adu_pre_debayer)+min_adu_pre_debayer
            else: img_data_processed_adu=np.full_like(img_rgb_norm_01,min_adu_pre_debayer if np.isfinite(min_adu_pre_debayer) else 0.)
        except Exception as e_debayer: 
            _pcb_local("getwcs_warn_debayer_failed", lvl="WARN", filename=filename, error=str(e_debayer))
            img_data_processed_adu = np.stack([img_data_processed_adu]*3, axis=-1) # Fallback stack
    
    if img_data_processed_adu.ndim == 2: # Toujours monochrome après tentative de débayerisation
        _pcb_local("getwcs_warn_still_2d_after_debayer_attempt", lvl="WARN", filename=filename)
        img_data_processed_adu = np.stack([img_data_processed_adu]*3, axis=-1)
    
    if img_data_processed_adu.ndim != 3 or img_data_processed_adu.shape[-1] != 3:
        _pcb_local("getwcs_error_shape_after_debayer_final_check", lvl="ERROR", filename=filename, shape=str(img_data_processed_adu.shape))
        return None, None, None, None

    # --- Correction Hot Pixels + optional GPU background smoothing ---
    _pcb_local(f"  Correction HP pour '{filename}'...", lvl="DEBUG_DETAIL")
    if hotpix_mask_dir:
        os.makedirs(hotpix_mask_dir, exist_ok=True)
        hp_mask_path = os.path.join(hotpix_mask_dir, f"hp_mask_{os.path.splitext(filename)[0]}_{uuid.uuid4().hex}.npy")

    img_data_hp_corrected_adu = None
    try:
        # Prefer GPU hot-pixel correction when available
        if hasattr(zemosaic_utils, 'detect_and_correct_hot_pixels_gpu') and zemosaic_utils.gpu_is_available():
            img_data_hp_corrected_adu = zemosaic_utils.detect_and_correct_hot_pixels_gpu(
                img_data_processed_adu,
                threshold=3.0,
                neighborhood_size=5,
                progress_callback=progress_callback,
            )
        else:
            raise RuntimeError('GPU HP not available')
    except Exception:
        if 'save_mask_path' in zemosaic_utils.detect_and_correct_hot_pixels.__code__.co_varnames:
            img_data_hp_corrected_adu = zemosaic_utils.detect_and_correct_hot_pixels(
                img_data_processed_adu,
                3.0,
                5,
                progress_callback=progress_callback,
                save_mask_path=hp_mask_path,
            )
        else:
            img_data_hp_corrected_adu = zemosaic_utils.detect_and_correct_hot_pixels(
                img_data_processed_adu,
                3.0,
                5,
                progress_callback=progress_callback,
            )

    if img_data_hp_corrected_adu is not None:
        img_data_processed_adu = img_data_hp_corrected_adu
    else:
        _pcb_local("getwcs_warn_hp_returned_none_using_previous", lvl="WARN", filename=filename)

    # Optional GPU background smoothing (stabilize inter-batch photometry)
    # IMPORTANT: remove only the low-frequency GRADIENT (bg - median(bg)) to avoid truncating
    # histogram at zero and avoid dark rings around stars. Do NOT hard-clip to 0 here.
    try:
        if _bg_gpu_enabled and hasattr(zemosaic_utils, 'estimate_background_map_gpu') and zemosaic_utils.gpu_is_available():
            bg = zemosaic_utils.estimate_background_map_gpu(img_data_processed_adu, method='gaussian', sigma=_bg_sigma)
            if bg is not None and np.any(np.isfinite(bg)):
                # Use luminance gradient so the subtraction is achromatic
                if bg.ndim == 3 and bg.shape[-1] == 3:
                    lum_bg = 0.299 * bg[..., 0].astype(np.float32) + 0.587 * bg[..., 1].astype(np.float32) + 0.114 * bg[..., 2].astype(np.float32)
                else:
                    lum_bg = bg.astype(np.float32)
                med_lum = np.nanmedian(lum_bg) if np.any(np.isfinite(lum_bg)) else 0.0
                grad = (lum_bg - med_lum).astype(np.float32)
                if img_data_processed_adu.ndim == 3 and img_data_processed_adu.shape[-1] == 3:
                    for c in range(3):
                        img_data_processed_adu[..., c] = img_data_processed_adu[..., c].astype(np.float32) - grad
                else:
                    img_data_processed_adu = img_data_processed_adu.astype(np.float32) - grad
                _pcb_local("  Background luminance gradient removed (achromatic), no hard clipping.", lvl="DEBUG_DETAIL")
    except Exception:
        pass

    header_for_wcs_check = header_orig if header_orig is not None else header_precheck
    skip_solver_due_to_existing_wcs = False
    wcs_validation_reason = None
    preexisting_wcs_obj = None
    if header_for_wcs_check is not None and hasattr(zemosaic_utils, "validate_wcs_header"):
        try:
            valid_wcs, candidate_wcs, failure_reason = zemosaic_utils.validate_wcs_header(header_for_wcs_check)
        except Exception as exc_validate_hdr:
            valid_wcs, candidate_wcs, failure_reason = False, None, f"validate_exception: {exc_validate_hdr}"
        skip_solver_due_to_existing_wcs = bool(valid_wcs)
        if skip_solver_due_to_existing_wcs:
            preexisting_wcs_obj = candidate_wcs
        else:
            wcs_validation_reason = failure_reason
    elif header_for_wcs_check is not None and hasattr(zemosaic_utils, "has_valid_wcs"):
        try:
            skip_solver_due_to_existing_wcs = bool(zemosaic_utils.has_valid_wcs(header_for_wcs_check))
        except Exception:
            skip_solver_due_to_existing_wcs = bool(preexisting_wcs_flag)
    else:
        skip_solver_due_to_existing_wcs = bool(preexisting_wcs_flag)

    if skip_solver_due_to_existing_wcs and preexisting_wcs_obj is None and header_for_wcs_check is not None and ASTROPY_AVAILABLE and WCS:
        try:
            candidate_wcs_hdr = WCS(header_for_wcs_check, naxis=2, relax=True)
            if getattr(candidate_wcs_hdr, "is_celestial", False):
                preexisting_wcs_obj = candidate_wcs_hdr
            else:
                skip_solver_due_to_existing_wcs = False
                if wcs_validation_reason is None:
                    wcs_validation_reason = "wcs_not_celestial"
        except Exception as e_wcs_hdr:
            skip_solver_due_to_existing_wcs = False
            if wcs_validation_reason is None:
                wcs_validation_reason = f"astropy_wcs_exception: {e_wcs_hdr}"
            _pcb_local("getwcs_warn_header_wcs_read_failed", lvl="WARN", filename=filename, error=str(e_wcs_hdr))
            logger.warning("Existing WCS header invalid for '%s': %s", filename, e_wcs_hdr)

    if not skip_solver_due_to_existing_wcs and wcs_validation_reason is None:
        wcs_validation_reason = preexisting_wcs_failure_reason

    if force_resolve_existing_wcs and skip_solver_due_to_existing_wcs:
        _pcb_local(
            "getwcs_info_force_resolve_existing_wcs",
            lvl="INFO",
            filename=filename,
        )
        logger.info("Force resolving existing WCS for '%s' due to configuration override.", filename)
        skip_solver_due_to_existing_wcs = False
        preexisting_wcs_obj = None

    if not skip_solver_due_to_existing_wcs and wcs_validation_reason:
        _pcb_local(
            "getwcs_info_existing_wcs_rejected",
            lvl="WARN",
            filename=filename,
            reason=wcs_validation_reason,
        )
        logger.warning("Existing WCS for '%s' rejected: %s", filename, wcs_validation_reason)

    # --- Résolution WCS ---
    _pcb_local(f"  Résolution WCS pour '{filename}'...", lvl="DEBUG_DETAIL")
    wcs_brute = preexisting_wcs_obj if preexisting_wcs_obj is not None else None
    # Évite d'écrire le header FITS si le WCS est déjà présent dans le fichier d'origine.
    # Nous ne réécrivons le header que si un solver externe (ASTAP/ASTROMETRY/ANSVR)
    # a effectivement injecté/ajusté des clés WCS dans header_orig.
    should_write_header_back = False
    if preexisting_wcs_obj is not None:
        skip_msg = f"Skip WCS solve for '{filename}' (WCS present)."
        _pcb_local(skip_msg, lvl="INFO")
        logger.info(skip_msg)
    if wcs_brute is None and ASTROPY_AVAILABLE and WCS: # S'assurer que WCS est bien l'objet d'Astropy
        try:
            wcs_from_header = WCS(header_orig, naxis=2, relax=True) # Utiliser WCS d'Astropy
            if wcs_from_header.is_celestial and hasattr(wcs_from_header.wcs,'crval') and \
               (hasattr(wcs_from_header.wcs,'cdelt') or hasattr(wcs_from_header.wcs,'cd') or hasattr(wcs_from_header.wcs,'pc')):
                wcs_brute = wcs_from_header
                _pcb_local(f"    WCS trouvé dans header FITS de '{filename}'.", lvl="DEBUG_DETAIL")
                skip_msg = f"Skip WCS solve for '{filename}' (WCS present)."
                _pcb_local(skip_msg, lvl="INFO")
                logger.info(skip_msg)
                # WCS déjà présent => pas besoin de réécrire le header
                should_write_header_back = False
        except Exception as e_wcs_hdr:
            _pcb_local("getwcs_warn_header_wcs_read_failed", lvl="WARN", filename=filename, error=str(e_wcs_hdr))
            wcs_brute = None
            
    solver_choice_effective = (solver_settings or {}).get("solver_choice", "ASTAP")
    api_key_len = len((solver_settings or {}).get("api_key", ""))
    _pcb_local(
        f"Solver choice effective={solver_choice_effective}",
        lvl="DEBUG_DETAIL",
    )
    if wcs_brute is None and ZEMOSAIC_ASTROMETRY_AVAILABLE and zemosaic_astrometry:
        try:
            # Utiliser directement le fichier original sans conversion mono ni FITS minimal
            input_for_solver = file_path

            if solver_choice_effective == "ASTROMETRY":
                _pcb_local("GetWCS: using ASTROMETRY", lvl="DEBUG")
                wcs_brute = solve_with_astrometry(
                    input_for_solver,
                    header_orig,
                    solver_settings or {},
                    progress_callback,
                )
                if not wcs_brute and astap_paths_valid(astap_exe_path, astap_data_dir):
                    _pcb_local("Astrometry failed; fallback to ASTAP", lvl="INFO")
                    _pcb_local("GetWCS: using ASTAP (fallback)", lvl="DEBUG")
                    wcs_brute = zemosaic_astrometry.solve_with_astap(
                        image_fits_path=input_for_solver,
                        original_fits_header=header_orig,
                        astap_exe_path=astap_exe_path,
                        astap_data_dir=astap_data_dir,
                        search_radius_deg=astap_search_radius,
                        downsample_factor=astap_downsample,
                        sensitivity=astap_sensitivity,
                        timeout_sec=astap_timeout_seconds,
                        update_original_header_in_place=True,
                        progress_callback=progress_callback,
                    )
                # Si un solver a réussi, le header_orig a potentiellement été mis à jour
                if wcs_brute:
                    should_write_header_back = True
                if wcs_brute:
                    _pcb_local("getwcs_info_astrometry_solved", lvl="INFO_DETAIL", filename=filename)
            elif solver_choice_effective == "ANSVR":
                _pcb_local("GetWCS: using ANSVR", lvl="DEBUG")
                wcs_brute = solve_with_ansvr(
                    input_for_solver,
                    header_orig,
                    solver_settings or {},
                    progress_callback,
                )
                if not wcs_brute and astap_paths_valid(astap_exe_path, astap_data_dir):
                    _pcb_local("Ansvr failed; fallback to ASTAP", lvl="INFO")
                    _pcb_local("GetWCS: using ASTAP (fallback)", lvl="DEBUG")
                    wcs_brute = zemosaic_astrometry.solve_with_astap(
                        image_fits_path=input_for_solver,
                        original_fits_header=header_orig,
                        astap_exe_path=astap_exe_path,
                        astap_data_dir=astap_data_dir,
                        search_radius_deg=astap_search_radius,
                        downsample_factor=astap_downsample,
                        sensitivity=astap_sensitivity,
                        timeout_sec=astap_timeout_seconds,
                        update_original_header_in_place=True,
                        progress_callback=progress_callback,
                    )
                # Si ANSVR/ASTAP réussit, le header a été mis à jour par le solver
                if wcs_brute:
                    should_write_header_back = True
                if wcs_brute:
                    _pcb_local("getwcs_info_astrometry_solved", lvl="INFO_DETAIL", filename=filename)
            else:
                _pcb_local("GetWCS: using ASTAP", lvl="DEBUG")
                wcs_brute = zemosaic_astrometry.solve_with_astap(
                    image_fits_path=input_for_solver,
                    original_fits_header=header_orig,
                    astap_exe_path=astap_exe_path,
                    astap_data_dir=astap_data_dir,
                    search_radius_deg=astap_search_radius,
                    downsample_factor=astap_downsample,
                    sensitivity=astap_sensitivity,
                    timeout_sec=astap_timeout_seconds,
                    update_original_header_in_place=True,
                    progress_callback=progress_callback,
                )
                # ASTAP a potentiellement mis à jour le header_orig
                if wcs_brute:
                    should_write_header_back = True
                if wcs_brute:
                    _pcb_local("getwcs_info_astap_solved", lvl="INFO_DETAIL", filename=filename)
                else:
                    _pcb_local("getwcs_warn_astap_failed", lvl="WARN", filename=filename)
        except Exception as e_solver_call:
            _pcb_local("getwcs_error_astap_exception", lvl="ERROR", filename=filename, error=str(e_solver_call))
            logger.error(f"Erreur solver pour {filename}", exc_info=True)
            wcs_brute = None
        finally:
            del img_data_raw_adu
            gc.collect()
    elif wcs_brute is None: # Ni header, ni ASTAP n'a fonctionné ou n'était dispo
        _pcb_local("getwcs_warn_no_wcs_source_available_or_failed", lvl="WARN", filename=filename)
        # Action de déplacement sera gérée par le check suivant

    # --- Vérification finale du WCS et action de déplacement si échec ---
    if wcs_brute and wcs_brute.is_celestial:
        # Mettre à jour pixel_shape si nécessaire
        if wcs_brute.pixel_shape is None or not (wcs_brute.pixel_shape[0]>0 and wcs_brute.pixel_shape[1]>0):
            n1_final = header_orig.get('NAXIS1', img_data_processed_adu.shape[1])
            n2_final = header_orig.get('NAXIS2', img_data_processed_adu.shape[0])
            if n1_final > 0 and n2_final > 0:
                try: wcs_brute.pixel_shape = (int(n1_final), int(n2_final))
                except Exception as e_ps_final: 
                    _pcb_local("getwcs_error_set_pixel_shape_final_wcs_invalid", lvl="ERROR", filename=filename, error=str(e_ps_final))
                    # WCS devient invalide ici
                    wcs_brute = None # Forcer le déplacement
            else:
                _pcb_local("getwcs_error_invalid_naxis_for_pixel_shape_wcs_invalid", lvl="ERROR", filename=filename)
                wcs_brute = None # Forcer le déplacement
        
        if wcs_brute and wcs_brute.is_celestial: # Re-vérifier après la tentative de set_pixel_shape
            _pcb_local("getwcs_info_pretreatment_wcs_ok", lvl="DEBUG", filename=filename)
            # Écriture du header uniquement si un solver a réellement mis à jour le header
            if should_write_header_back:
                _write_header_to_fits(file_path, header_orig, _pcb_local)
            return img_data_processed_adu, wcs_brute, header_orig, hp_mask_path
        # else: tombe dans le bloc de déplacement ci-dessous

    # Si on arrive ici, c'est que wcs_brute est None ou non céleste
    _pcb_local("getwcs_action_moving_unsolved_file", lvl="WARN", filename=filename)
    status, destination_path = _move_to_unaligned_safe(
        file_path,
        os.path.dirname(file_path),
        logger=logger,
    )

    if status == "moved" and destination_path is not None:
        _pcb_local(
            f"  Fichier '{filename}' déplacé vers '{destination_path.parent}'.",
            lvl="INFO",
        )
    elif status in {"already_moved", "missing"}:
        _pcb_local(
            f"  Fichier '{filename}' déjà déplacé ou introuvable (course détectée).",
            lvl="DEBUG_DETAIL",
        )
    elif status == "skipped_excluded":
        _pcb_local(
            f"  Fichier '{filename}' déjà dans un dossier exclu, déplacement ignoré.",
            lvl="WARN",
        )
    elif status in {"conflict", "failed"}:
        _pcb_local(
            "getwcs_error_moving_unaligned_file",
            lvl="ERROR",
            filename=filename,
            error=f"status={status}",
        )

    if img_data_processed_adu is not None: del img_data_processed_adu
    gc.collect()
    return None, None, None, None








# Dans zemosaic_worker.py

# ... (vos imports existants : os, shutil, time, traceback, gc, logging, np, astropy, reproject, et les modules zemosaic_...)

def create_master_tile(
    seestar_stack_group_info: list[dict],
    tile_id: int,
    output_temp_dir: str,
    # Paramètres de stacking existants
    stack_norm_method: str,
    stack_weight_method: str, # Ex: "none", "noise_variance", "noise_fwhm", "noise_plus_fwhm"
    stack_reject_algo: str,
    stack_kappa_low: float,
    stack_kappa_high: float,
    parsed_winsor_limits: tuple[float, float],
    stack_final_combine: str,
    poststack_equalize_rgb: bool,
    # --- NOUVEAUX PARAMÈTRES POUR LA PONDÉRATION RADIALE ---
    apply_radial_weight: bool,             # Vient de la GUI/config
    radial_feather_fraction: float,      # Vient de la GUI/config
    radial_shape_power: float,           # Pourrait être une constante ou configurable
    min_radial_weight_floor: float,
    # --- FIN NOUVEAUX PARAMÈTRES ---
    quality_crop_enabled: bool,
    quality_crop_band_px: int,
    quality_crop_k_sigma: float,
    quality_crop_margin_px: int,
    # Paramètres ASTAP (pourraient être enlevés si plus du tout utilisés ici)
    astap_exe_path_global: str, 
    astap_data_dir_global: str, 
    astap_search_radius_global: float,
    astap_downsample_global: int,
    astap_sensitivity_global: int,
    astap_timeout_seconds_global: int,
    winsor_pool_workers: int,
    winsor_max_frames_per_pass: int,
    progress_callback: callable,
    resource_strategy: dict | None = None,
    center_out_context: CenterOutNormalizationContext | None = None,
    center_out_settings: dict | None = None,
    center_out_rank: int | None = None,
):
    """
    Crée une "master tuile" à partir d'un groupe d'images.
    Lit les données image prétraitées depuis un cache disque (.npy).
    Utilise les WCS et Headers déjà résolus et stockés en mémoire.
    Transmet toutes les options de stacking, y compris la pondération radiale.

    Returns
    -------
    tuple[tuple[str | None, object | None], list[list[dict]]]
        - ``(path, wcs)`` du master stack produit (``None`` si échec).
        - Liste de sous-groupes à retraiter (copie des ``raw_info`` pour les images non alignées).
    """
    pcb_tile = lambda msg_key, prog=None, lvl="INFO_DETAIL", **kwargs: _log_and_callback(msg_key, prog, lvl, callback=progress_callback, **kwargs)
    # Load persistent configuration to forward GPU preference
    if ZEMOSAIC_CONFIG_AVAILABLE and zemosaic_config:
        try:
            zconfig = SimpleNamespace(**zemosaic_config.load_config())
        except Exception:
            zconfig = SimpleNamespace()
    else:
        zconfig = SimpleNamespace()
    # Ensure stacking GPU preference mirrors the Phase 5 GPU intent when not explicitly set.
    try:
        phase5_pref = bool(getattr(zconfig, "use_gpu_phase5"))
    except Exception:
        phase5_pref = False
    try:
        stack_pref = getattr(zconfig, "stack_use_gpu")
    except AttributeError:
        stack_pref = None
    except Exception:
        stack_pref = None
    if stack_pref is None:
        setattr(zconfig, "stack_use_gpu", phase5_pref)
    else:
        try:
            setattr(zconfig, "stack_use_gpu", bool(stack_pref))
        except Exception:
            setattr(zconfig, "stack_use_gpu", phase5_pref)
    if not hasattr(zconfig, "use_gpu_stack") or getattr(zconfig, "use_gpu_stack") is None:
        setattr(zconfig, "use_gpu_stack", getattr(zconfig, "stack_use_gpu", phase5_pref))
    try:
        setattr(zconfig, "poststack_equalize_rgb", bool(poststack_equalize_rgb))
    except Exception:
        pass
    # Do not alias Phase‑5 GPU flag onto a generic 'use_gpu' here.
    # Stacking code now honors only explicit stacking flags
    # (e.g., 'stack_use_gpu' / 'use_gpu_stack') or 'use_gpu' if set by user.
    if resource_strategy:
        try:
            if resource_strategy.get('gpu_batch_hint'):
                setattr(zconfig, 'gpu_batch_hint', int(resource_strategy.get('gpu_batch_hint')))
            if 'memmap' in resource_strategy:
                setattr(zconfig, 'stack_memmap_enabled', bool(resource_strategy.get('memmap')))
            if resource_strategy.get('memmap_budget_mb') is not None:
                setattr(zconfig, 'stack_memmap_budget_mb', resource_strategy.get('memmap_budget_mb'))
        except Exception:
            pass
        try:
            pcb_tile(
                f"{func_id_log_base}_autocaps_hint",
                prog=None,
                lvl="INFO_DETAIL",
                cap=resource_strategy.get('cap'),
                memmap=resource_strategy.get('memmap'),
                gpu_hint=resource_strategy.get('gpu_batch_hint'),
            )
        except Exception:
            pass
    func_id_log_base = "mastertile"

    pcb_tile(f"{func_id_log_base}_info_creation_started_from_cache", prog=None, lvl="INFO",
             num_raw=len(seestar_stack_group_info), tile_id=tile_id)
    failed_groups_to_retry: list[list[dict]] = []
    pcb_tile(
        f"    {func_id_log_base}_{tile_id}: Options Stacking - Norm='{stack_norm_method}', "
        f"Weight='{stack_weight_method}' (RadialWeight={apply_radial_weight}), "
        f"Reject='{stack_reject_algo}', Combine='{stack_final_combine}', RGBEqualize={poststack_equalize_rgb}",
        prog=None,
        lvl="DEBUG",
    )

    if not (ZEMOSAIC_UTILS_AVAILABLE and zemosaic_utils and ZEMOSAIC_ALIGN_STACK_AVAILABLE and zemosaic_align_stack and ASTROPY_AVAILABLE and fits): # Ajout de 'fits' pour header_mt_save
        # ... (votre gestion d'erreur de dépendances existante) ...
        if not ZEMOSAIC_UTILS_AVAILABLE: pcb_tile(f"{func_id_log_base}_error_utils_unavailable", prog=None, lvl="ERROR", tile_id=tile_id)
        if not ZEMOSAIC_ALIGN_STACK_AVAILABLE: pcb_tile(f"{func_id_log_base}_error_alignstack_unavailable", prog=None, lvl="ERROR", tile_id=tile_id)
        if not ASTROPY_AVAILABLE or not fits: pcb_tile(f"{func_id_log_base}_error_astropy_unavailable", prog=None, lvl="ERROR", tile_id=tile_id)
        return (None, None), failed_groups_to_retry
        
    if not seestar_stack_group_info: 
        pcb_tile(f"{func_id_log_base}_error_no_images_provided", prog=None, lvl="ERROR", tile_id=tile_id)
        return (None, None), failed_groups_to_retry
    
    # Choix de l'image de référence (généralement la première du groupe après tri ou la plus centrale)
    reference_image_index_in_group = 0 # Pourrait être plus sophistiqué à l'avenir
    if not (0 <= reference_image_index_in_group < len(seestar_stack_group_info)): 
        pcb_tile(f"{func_id_log_base}_error_invalid_ref_index", prog=None, lvl="ERROR", tile_id=tile_id, ref_idx=reference_image_index_in_group, group_size=len(seestar_stack_group_info))
        return (None, None), failed_groups_to_retry
    
    ref_info_for_tile = seestar_stack_group_info[reference_image_index_in_group]
    wcs_for_master_tile = ref_info_for_tile.get('wcs')
    # Le header est un dict venant du cache, il faut le convertir en objet fits.Header si besoin
    header_dict_for_master_tile_base = ref_info_for_tile.get('header') 

    if not (wcs_for_master_tile and wcs_for_master_tile.is_celestial and header_dict_for_master_tile_base):
        pcb_tile(f"{func_id_log_base}_error_invalid_ref_wcs_header", prog=None, lvl="ERROR", tile_id=tile_id)
        return (None, None), failed_groups_to_retry
    
    # Conversion du dict en objet astropy.io.fits.Header pour la sauvegarde
    header_for_master_tile_base = fits.Header(header_dict_for_master_tile_base.cards if hasattr(header_dict_for_master_tile_base,'cards') else header_dict_for_master_tile_base)
    
    ref_path_raw = ref_info_for_tile.get('path_raw', 'UnknownRawRef')
    pcb_tile(f"{func_id_log_base}_info_reference_set", prog=None, lvl="DEBUG_DETAIL", ref_index=reference_image_index_in_group, ref_filename=os.path.basename(ref_path_raw), tile_id=tile_id)

    # Acquire a dynamic Phase 3 I/O concurrency slot to avoid disk stalls
    # when the system is busy (e.g., another app reading video files).
    try:
        _PH3_CONCURRENCY_SEMAPHORE.acquire()
    except Exception:
        pass

    pcb_tile(f"{func_id_log_base}_info_loading_from_cache_started", prog=None, lvl="DEBUG_DETAIL", num_images=len(seestar_stack_group_info), tile_id=tile_id)
    
    tile_images_data_HWC_adu = []
    tile_original_raw_headers = [] # Liste des dictionnaires de header originaux

    for i, raw_file_info in enumerate(seestar_stack_group_info):
        cached_image_file_path = raw_file_info.get('path_preprocessed_cache')
        original_raw_path = raw_file_info.get('path_raw', 'UnknownRawPathForTileImg') # Plus descriptif

        if not (cached_image_file_path and os.path.exists(cached_image_file_path)):
            pcb_tile(f"{func_id_log_base}_warn_cache_file_missing", prog=None, lvl="WARN", filename=os.path.basename(original_raw_path), cache_path=cached_image_file_path, tile_id=tile_id)
            continue
        
        # pcb_tile(f"    {func_id_log_base}_{tile_id}_Img{i}: Lecture cache '{os.path.basename(cached_image_file_path)}'", prog=None, lvl="DEBUG_VERY_DETAIL")
        
        try:
            # Throttle concurrent cache reads and use memory-mapped load to reduce RAM spikes
            with _CACHE_IO_SEMAPHORE:
                img_data_adu = np.load(cached_image_file_path, allow_pickle=False, mmap_mode='r') 
            if not (isinstance(img_data_adu, np.ndarray) and img_data_adu.dtype == np.float32 and img_data_adu.ndim == 3 and img_data_adu.shape[-1] == 3):
                pcb_tile(f"{func_id_log_base}_warn_invalid_cached_data", prog=None, lvl="WARN", filename=os.path.basename(cached_image_file_path), 
                         shape=img_data_adu.shape if hasattr(img_data_adu, 'shape') else 'N/A', 
                         dtype=img_data_adu.dtype if hasattr(img_data_adu, 'dtype') else 'N/A', tile_id=tile_id)
                del img_data_adu; gc.collect(); continue
            # Assurer des buffers C-contigus ET écriturables pour l'aligneur
            # (les memmaps en lecture seule peuvent provoquer des échecs silencieux)
            try:
                img_data_adu = np.array(img_data_adu, dtype=np.float32, order='C', copy=True)
            except Exception:
                # En cas d'exception, forcer une copie contiguë
                img_data_adu = np.ascontiguousarray(img_data_adu, dtype=np.float32)
            
            tile_images_data_HWC_adu.append(img_data_adu)
            # Stocker le dict de header, pas l'objet fits.Header, car c'est ce qui est dans raw_file_info
            tile_original_raw_headers.append(raw_file_info.get('header')) 
        except MemoryError as e_mem_load_cache:
             pcb_tile(f"{func_id_log_base}_error_memory_loading_cache", prog=None, lvl="ERROR", filename=os.path.basename(cached_image_file_path), error=str(e_mem_load_cache), tile_id=tile_id)
             # Release the concurrency slot before aborting
             try:
                 _PH3_CONCURRENCY_SEMAPHORE.release()
             except Exception:
                 pass
             del tile_images_data_HWC_adu, tile_original_raw_headers; gc.collect(); return (None, None), failed_groups_to_retry
        except Exception as e_load_cache:
            pcb_tile(f"{func_id_log_base}_error_loading_cache", prog=None, lvl="ERROR", filename=os.path.basename(cached_image_file_path), error=str(e_load_cache), tile_id=tile_id)
            logger.error(f"Erreur chargement cache {cached_image_file_path} pour tuile {tile_id}", exc_info=True)
            continue
            
    # Release the concurrency slot as soon as disk reads are done for this tile
    try:
        _PH3_CONCURRENCY_SEMAPHORE.release()
    except Exception:
        pass

    if not tile_images_data_HWC_adu:
        pcb_tile(f"{func_id_log_base}_error_no_valid_images_from_cache", prog=None, lvl="ERROR", tile_id=tile_id)
        return (None, None), failed_groups_to_retry
    # pcb_tile(f"{func_id_log_base}_info_loading_from_cache_finished", prog=None, lvl="DEBUG_DETAIL", num_loaded=len(tile_images_data_HWC_adu), tile_id=tile_id)

    pcb_tile(f"{func_id_log_base}_info_intra_tile_alignment_started", prog=None, lvl="DEBUG_DETAIL", num_to_align=len(tile_images_data_HWC_adu), tile_id=tile_id)
    # Limit concurrency during alignment/stacking as well to reduce peak RAM
    try:
        _PH3_CONCURRENCY_SEMAPHORE.acquire()
    except Exception:
        pass
    aligned_images_for_stack, failed_alignment_indices = zemosaic_align_stack.align_images_in_group(
        image_data_list=tile_images_data_HWC_adu,
        reference_image_index=reference_image_index_in_group,
        progress_callback=progress_callback
    )
    if failed_alignment_indices:
        retry_group: list[dict] = []
        for idx_fail in failed_alignment_indices:
            if 0 <= idx_fail < len(seestar_stack_group_info):
                raw_info = seestar_stack_group_info[idx_fail]
                if isinstance(raw_info, dict):
                    info_copy = dict(raw_info)
                    current_retry = int(info_copy.get('retry_attempt', 0))
                    info_copy['retry_attempt'] = current_retry + 1
                    origin_chain = list(info_copy.get('retry_origin_chain', []))
                    origin_chain.append(int(tile_id))
                    info_copy['retry_origin_chain'] = origin_chain
                else:
                    info_copy = raw_info
                retry_group.append(info_copy)
        if retry_group:
            failed_groups_to_retry.append(retry_group)

    del tile_images_data_HWC_adu; gc.collect()

    valid_aligned_images = [img for img in aligned_images_for_stack if img is not None]
    if aligned_images_for_stack:
        del aligned_images_for_stack # Libérer la liste originale après filtrage

    num_actually_aligned_for_header = len(valid_aligned_images)
    pcb_tile(f"{func_id_log_base}_info_intra_tile_alignment_finished", prog=None, lvl="DEBUG_DETAIL", num_aligned=num_actually_aligned_for_header, tile_id=tile_id)
    
    if not valid_aligned_images: 
        pcb_tile(f"{func_id_log_base}_error_no_images_after_alignment", prog=None, lvl="ERROR", tile_id=tile_id)
        try:
            _PH3_CONCURRENCY_SEMAPHORE.release()
        except Exception:
            pass
        return (None, None), failed_groups_to_retry
    
    pcb_tile(f"{func_id_log_base}_info_stacking_started", prog=None, lvl="DEBUG_DETAIL",
             num_to_stack=len(valid_aligned_images), tile_id=tile_id) # Les options sont loggées au début

    stack_metadata: dict[str, Any] = {}

    if stack_reject_algo == "winsorized_sigma_clip":
        master_tile_stacked_HWC, _ = zemosaic_align_stack.stack_winsorized_sigma_clip(
            valid_aligned_images,
            weight_method=stack_weight_method,
            zconfig=zconfig,
            kappa=stack_kappa_low,
            winsor_limits=parsed_winsor_limits,
            apply_rewinsor=True,
            winsor_max_frames_per_pass=int(winsor_max_frames_per_pass) if winsor_max_frames_per_pass is not None else 0,
            winsor_max_workers=int(winsor_pool_workers) if winsor_pool_workers is not None else 1,
            stack_metadata=stack_metadata,
        )
    elif stack_reject_algo == "kappa_sigma":
        master_tile_stacked_HWC, _ = zemosaic_align_stack.stack_kappa_sigma_clip(
            valid_aligned_images,
            weight_method=stack_weight_method,
            zconfig=zconfig,
            sigma_low=stack_kappa_low,
            sigma_high=stack_kappa_high,
            stack_metadata=stack_metadata,
        )
    elif stack_reject_algo == "linear_fit_clip":
        master_tile_stacked_HWC, _ = zemosaic_align_stack.stack_linear_fit_clip(
            valid_aligned_images,
            weight_method=stack_weight_method,
            zconfig=zconfig,
            sigma=stack_kappa_high,
            stack_metadata=stack_metadata,
        )
    else:
        master_tile_stacked_HWC = zemosaic_align_stack.stack_aligned_images(
            aligned_image_data_list=valid_aligned_images,
            normalize_method=stack_norm_method,
            weighting_method=stack_weight_method,
            rejection_algorithm=stack_reject_algo,
            final_combine_method=stack_final_combine,
            sigma_clip_low=stack_kappa_low,
            sigma_clip_high=stack_kappa_high,
            winsor_limits=parsed_winsor_limits,
            minimum_signal_adu_target=0.0,
            apply_radial_weight=apply_radial_weight,
            radial_feather_fraction=radial_feather_fraction,
            radial_shape_power=radial_shape_power,
            winsor_max_workers=winsor_pool_workers,
            progress_callback=progress_callback,
            zconfig=zconfig,
            stack_metadata=stack_metadata,
        )
    
    del valid_aligned_images; gc.collect() # valid_aligned_images a été passé par valeur (copie de la liste)
                                          # mais les arrays NumPy à l'intérieur sont passés par référence.
                                          # stack_aligned_images travaille sur ces arrays.
                                          # Il est bon de del ici.

    if master_tile_stacked_HWC is None:
        pcb_tile(f"{func_id_log_base}_error_stacking_failed", prog=None, lvl="ERROR", tile_id=tile_id)
        try:
            _PH3_CONCURRENCY_SEMAPHORE.release()
        except Exception:
            pass
        return (None, None), failed_groups_to_retry

    pcb_tile(f"{func_id_log_base}_info_stacking_finished", prog=None, lvl="DEBUG_DETAIL", tile_id=tile_id,
             shape=master_tile_stacked_HWC.shape)
             # min_val=np.nanmin(master_tile_stacked_HWC), # Peut être verbeux
             # max_val=np.nanmax(master_tile_stacked_HWC),
             # mean_val=np.nanmean(master_tile_stacked_HWC))

    rgb_eq_info = stack_metadata.get("rgb_equalization", {})
    try:
        gain_r = float(rgb_eq_info.get("gain_r", 1.0))
    except (TypeError, ValueError):
        gain_r = 1.0
    try:
        gain_g = float(rgb_eq_info.get("gain_g", 1.0))
    except (TypeError, ValueError):
        gain_g = 1.0
    try:
        gain_b = float(rgb_eq_info.get("gain_b", 1.0))
    except (TypeError, ValueError):
        gain_b = 1.0
    try:
        target_median_val = float(rgb_eq_info.get("target_median", float("nan")))
    except (TypeError, ValueError):
        target_median_val = float("nan")
    eq_enabled = bool(rgb_eq_info.get("enabled", False))
    eq_applied = bool(rgb_eq_info.get("applied", False))
    target_str = f"{target_median_val:.6g}" if np.isfinite(target_median_val) else "nan"
    history_msg = (
        f"RGB equalized per sub-stack (enabled={str(eq_enabled)}, applied={str(eq_applied)}): "
        f"gains=({gain_r:.6f},{gain_g:.6f},{gain_b:.6f}), target={target_str}"
    )
    pcb_tile(
        f"[RGB-EQ] poststack_equalize_rgb enabled={eq_enabled}, applied={eq_applied}, "
        f"gains=({gain_r:.6f},{gain_g:.6f},{gain_b:.6f}), target={target_str}",
        prog=None,
        lvl="INFO" if eq_enabled else "DEBUG_DETAIL",
    )

    norm_result = None
    norm_mode = "disabled"
    norm_details: dict = {}
    if center_out_context and isinstance(center_out_settings, dict):
        try:
            norm_settings = {
                "enabled": bool(center_out_settings.get("enabled", True)),
                "preview_size": int(center_out_settings.get("preview_size", 256)),
                "sky_percentile": tuple(center_out_settings.get("sky_percentile", (25.0, 60.0))),
                "clip_sigma": float(center_out_settings.get("clip_sigma", 2.5)),
                "min_overlap_fraction": float(center_out_settings.get("min_overlap_fraction", 0.03)),
            }
        except Exception:
            norm_settings = {"enabled": False}
        if norm_settings.get("enabled", False):
            master_tile_stacked_HWC, norm_result, norm_mode, norm_details = apply_center_out_normalization_p3(
                master_tile_stacked_HWC,
                wcs_for_master_tile,
                tile_id,
                center_out_context,
                norm_settings,
                pcb_tile,
            )
            if norm_result:
                pcb_tile(
                    f"{func_id_log_base}_center_out_applied",
                    prog=None,
                    lvl="INFO_DETAIL",
                    tile_id=tile_id,
                    gain=f"{norm_result[0]:.6f}",
                    offset=f"{norm_result[1]:.6f}",
                    mode=norm_mode,
                    samples=norm_details.get("samples"),
                )
            else:
                pcb_tile(
                    f"{func_id_log_base}_center_out_skipped",
                    prog=None,
                    lvl="DEBUG_DETAIL",
                    tile_id=tile_id,
                    reason=norm_mode,
                )

    quality_crop_rect: tuple[int, int, int, int] | None = None
    if quality_crop_enabled:
        try:
            band_px = max(4, int(quality_crop_band_px))
        except Exception:
            band_px = 32
        try:
            margin_px = max(0, int(quality_crop_margin_px))
        except Exception:
            margin_px = 8
        try:
            k_sigma = float(quality_crop_k_sigma)
            if not math.isfinite(k_sigma):
                raise ValueError("k_sigma not finite")
        except Exception:
            k_sigma = 2.0
        k_sigma = max(0.1, min(k_sigma, 10.0))

        data_for_crop = np.asarray(master_tile_stacked_HWC)
        axis_mode = "HWC"
        if data_for_crop.ndim == 3 and data_for_crop.shape[0] == 3 and data_for_crop.shape[-1] != 3:
            data_for_crop = np.moveaxis(data_for_crop, 0, -1)
            axis_mode = "CHW"
        elif data_for_crop.ndim == 2:
            data_for_crop = data_for_crop[..., np.newaxis]
            axis_mode = "HW"

        try:
            if data_for_crop.ndim < 3:
                raise ValueError("insufficient dimensions for quality crop")

            if data_for_crop.shape[-1] >= 3:
                R = data_for_crop[..., 0]
                G = data_for_crop[..., 1]
                B = data_for_crop[..., 2]
            else:
                mono = data_for_crop[..., 0]
                R = G = B = mono

            lum2d = np.nanmean(np.stack([R, G, B], axis=0), axis=0).astype(np.float32)
            R = np.nan_to_num(np.asarray(R, dtype=np.float32), nan=0.0)
            G = np.nan_to_num(np.asarray(G, dtype=np.float32), nan=0.0)
            B = np.nan_to_num(np.asarray(B, dtype=np.float32), nan=0.0)
            lum2d = np.nan_to_num(lum2d, nan=0.0)

            from .lecropper import detect_autocrop_rgb

            y0, x0, y1, x1 = detect_autocrop_rgb(
                lum2d,
                R,
                G,
                B,
                band_px=band_px,
                k_sigma=k_sigma,
                margin_px=margin_px,
            )

            h_lum, w_lum = lum2d.shape
            if not (0 <= y0 < y1 <= h_lum and 0 <= x0 < x1 <= w_lum):
                raise ValueError("invalid crop rectangle")

            crop_area = (y1 - y0) * (x1 - x0)
            full_area = h_lum * w_lum
            if crop_area <= 0 or (crop_area / max(1, full_area)) >= 0.97:
                pcb_tile(
                    f"MT_CROP: quality crop skipped (rect={y0,x0,y1,x1}, area_ratio={crop_area/max(1, full_area):.3f})",
                    prog=None,
                    lvl="WARN",
                )
            else:
                cropped = data_for_crop[y0:y1, x0:x1, ...]
                if axis_mode == "CHW":
                    master_tile_stacked_HWC = np.moveaxis(cropped, -1, 0)
                elif axis_mode == "HW":
                    master_tile_stacked_HWC = cropped[..., 0]
                else:
                    master_tile_stacked_HWC = cropped
                quality_crop_rect = (int(y0), int(x0), int(y1), int(x1))

                if wcs_for_master_tile is not None:
                    try:
                        if hasattr(wcs_for_master_tile, "deepcopy"):
                            wcs_cropped = wcs_for_master_tile.deepcopy()
                        else:
                            wcs_cropped = copy.deepcopy(wcs_for_master_tile)
                    except Exception:
                        wcs_cropped = None

                    if wcs_cropped is not None and hasattr(wcs_cropped, "wcs"):
                        try:
                            wcs_cropped.wcs.crpix[0] -= x0
                            wcs_cropped.wcs.crpix[1] -= y0
                        except Exception as e_crpix:
                            pcb_tile(
                                f"MT_CROP: quality-based WCS shift failed: {e_crpix}",
                                prog=None,
                                lvl="WARN",
                            )
                        else:
                            wcs_for_master_tile = wcs_cropped
                    elif wcs_cropped is not None:
                        wcs_for_master_tile = wcs_cropped

                    if wcs_for_master_tile is wcs_cropped and hasattr(wcs_cropped, "pixel_shape"):
                        try:
                            new_h, new_w = master_tile_stacked_HWC.shape[:2]
                            wcs_cropped.pixel_shape = (new_w, new_h)
                            if hasattr(wcs_cropped, "array_shape"):
                                wcs_cropped.array_shape = (new_h, new_w)
                        except Exception:
                            pass

                pcb_tile(
                    f"MT_CROP: quality-based rect={quality_crop_rect} (band={band_px}, k={k_sigma:.2f}, margin={margin_px})",
                    prog=None,
                    lvl="INFO_DETAIL",
                )
        except Exception as e_crop:
            pcb_tile(
                f"MT_CROP: quality-based crop failed ({e_crop})",
                prog=None,
                lvl="WARN",
            )

    # pcb_tile(f"{func_id_log_base}_info_saving_started", prog=None, lvl="DEBUG_DETAIL", tile_id=tile_id)
    temp_fits_filename = f"master_tile_{tile_id:03d}.fits"
    temp_fits_filepath = os.path.join(output_temp_dir,temp_fits_filename)

    try:
        # Créer un nouvel objet Header pour la sauvegarde
        header_mt_save = fits.Header()
        if wcs_for_master_tile:
            try: 
                # S'assurer que wcs_for_master_tile a les NAXIS bien définis pour to_header
                # La shape de master_tile_stacked_HWC est (H, W, C)
                # Pour le WCS 2D, on a besoin de (W, H)
                if master_tile_stacked_HWC.ndim >= 2:
                    h_final, w_final = master_tile_stacked_HWC.shape[:2]
                    # Mettre à jour les attributs NAXIS du WCS si nécessaire,
                    # car to_header les utilise.
                    # wcs_for_master_tile.wcs.naxis1 = w_final # Ne pas modifier l'objet WCS original directement ici
                    # wcs_for_master_tile.wcs.naxis2 = h_final # car il est partagé/réutilisé.
                    # Créer une copie du WCS pour modification locale avant to_header si besoin.
                    # Cependant, save_fits_image devrait gérer les NAXIS en fonction des données.
                    pass

                header_mt_save.update(wcs_for_master_tile.to_header(relax=True))
            except Exception as e_wcs_hdr: 
                pcb_tile(f"{func_id_log_base}_warn_wcs_header_error_saving", prog=None, lvl="WARN", tile_id=tile_id, error=str(e_wcs_hdr))
        
        
        
        header_mt_save['ZMT_TYPE']=('Master Tile','ZeMosaic Processed Tile'); header_mt_save['ZMT_ID']=(tile_id,'Master Tile ID')
        header_mt_save['ZMT_NRAW']=(len(seestar_stack_group_info),'Raw frames in this tile group')
        header_mt_save['ZMT_NALGN']=(num_actually_aligned_for_header,'Successfully aligned frames for stack')
        header_mt_save['ZMT_NORM'] = (str(stack_norm_method), 'Normalization method')
        header_mt_save['ZMT_WGHT'] = (str(stack_weight_method), 'Weighting method')
        if apply_radial_weight: # Log des paramètres radiaux
            header_mt_save['ZMT_RADW'] = (True, 'Radial weighting applied')
            header_mt_save['ZMT_RADF'] = (radial_feather_fraction, 'Radial feather fraction')
            header_mt_save['ZMT_RADP'] = (radial_shape_power, 'Radial shape power')
        else:
            header_mt_save['ZMT_RADW'] = (False, 'Radial weighting applied')

        header_mt_save['RGBGAINR'] = (gain_r, 'RGB equalization gain (red)')
        header_mt_save['RGBGAING'] = (gain_g, 'RGB equalization gain (green)')
        header_mt_save['RGBGAINB'] = (gain_b, 'RGB equalization gain (blue)')
        header_mt_save['RGBEQMED'] = (target_median_val, 'RGB equalization target median')
        try:
            header_mt_save.add_history(history_msg)
        except Exception:
            header_mt_save['HISTORY'] = history_msg

        header_mt_save['ZMT_REJ'] = (str(stack_reject_algo), 'Rejection algorithm')
        if stack_reject_algo == "kappa_sigma":
            header_mt_save['ZMT_KAPLO'] = (stack_kappa_low, 'Kappa Sigma Low threshold')
            header_mt_save['ZMT_KAPHI'] = (stack_kappa_high, 'Kappa Sigma High threshold')
        elif stack_reject_algo == "winsorized_sigma_clip":
            header_mt_save['ZMT_WINLO'] = (parsed_winsor_limits[0], 'Winsor Lower limit %')
            header_mt_save['ZMT_WINHI'] = (parsed_winsor_limits[1], 'Winsor Upper limit %')
            # Les paramètres Kappa sont aussi pertinents pour Winsorized
            header_mt_save['ZMT_KAPLO'] = (stack_kappa_low, 'Kappa Low for Winsorized')
            header_mt_save['ZMT_KAPHI'] = (stack_kappa_high, 'Kappa High for Winsorized')
        header_mt_save['ZMT_COMB'] = (str(stack_final_combine), 'Final combine method')

        if center_out_context and center_out_settings:
            header_mt_save['ZMT_ANCH'] = (
                int(center_out_context.anchor_original_id),
                'Anchor tile id (original index)'
            )
            if norm_result:
                header_mt_save['ZMT_P3CO'] = (1, 'Phase 3 center-out normalization applied')
                header_mt_save['ZMT_AGAIN'] = (float(norm_result[0]), 'Phase 3 center-out gain')
                header_mt_save['ZMT_AOFF'] = (float(norm_result[1]), 'Phase 3 center-out offset')
            else:
                header_mt_save['ZMT_P3CO'] = (0, 'Phase 3 center-out normalization applied')
                header_mt_save['ZMT_AGAIN'] = (1.0, 'Phase 3 center-out gain')
                header_mt_save['ZMT_AOFF'] = (0.0, 'Phase 3 center-out offset')
        else:
            header_mt_save['ZMT_P3CO'] = (0, 'Phase 3 center-out normalization applied')
            header_mt_save['ZMT_AGAIN'] = (1.0, 'Phase 3 center-out gain')
            header_mt_save['ZMT_AOFF'] = (0.0, 'Phase 3 center-out offset')
            header_mt_save['ZMT_ANCH'] = (-1, 'Anchor tile id (original index)')

        if header_for_master_tile_base: # C'est déjà un objet fits.Header
            ref_path_raw_for_hdr = seestar_stack_group_info[reference_image_index_in_group].get('path_raw', 'UnknownRef')
            header_mt_save['ZMT_REF'] = (os.path.basename(ref_path_raw_for_hdr), 'Reference raw frame for this tile WCS')
            keys_from_ref = ['OBJECT','DATE-AVG','FILTER','INSTRUME','FOCALLEN','XPIXSZ','YPIXSZ', 'GAIN', 'OFFSET'] # Ajout GAIN, OFFSET
            for key_h in keys_from_ref:
                if key_h in header_for_master_tile_base:
                    try: 
                        # Tenter d'obtenir la valeur et le commentaire
                        card = header_for_master_tile_base.cards[key_h]
                        header_mt_save[key_h] = (card.value, card.comment)
                    except (KeyError, AttributeError): # Si la carte n'a pas de commentaire ou si ce n'est pas un objet CardImage
                        header_mt_save[key_h] = header_for_master_tile_base[key_h]
            
            total_exposure_tile = 0.
            num_exposure_summed = 0
            for hdr_raw_item_dict in tile_original_raw_headers: # Ce sont des dicts
                if hdr_raw_item_dict is None: continue
                try: 
                    exposure_val = hdr_raw_item_dict.get('EXPTIME', hdr_raw_item_dict.get('EXPOSURE', 0.0))
                    total_exposure_tile += float(exposure_val if exposure_val is not None else 0.0)
                    num_exposure_summed +=1
                except (TypeError, ValueError) : pass
            header_mt_save['EXPTOTAL']=(round(total_exposure_tile,2),'[s] Sum of EXPTIME for this tile')
            header_mt_save['NEXP_SUM']=(num_exposure_summed,'Number of exposures summed for EXPTOTAL')


        if quality_crop_rect:
            header_mt_save['ZMT_QCRO'] = (True, 'Quality-based crop applied')
            header_mt_save['ZMT_QBOX'] = (
                "{},{},{},{}".format(*quality_crop_rect),
                'Quality crop rectangle (y0,x0,y1,x1)',
            )
        else:
            header_mt_save['ZMT_QCRO'] = (False, 'Quality-based crop applied')

        zemosaic_utils.save_fits_image(
            image_data=master_tile_stacked_HWC,
            output_path=temp_fits_filepath,
            header=header_mt_save,
            overwrite=True,
            save_as_float=True,
            progress_callback=progress_callback,
            axis_order="HWC",
        )
        try:
            _register_master_tile_identity(temp_fits_filepath, f"tile:{int(tile_id):04d}")
        except Exception:
            _register_master_tile_identity(temp_fits_filepath, tile_id)
        pcb_tile(f"{func_id_log_base}_info_saved", prog=None, lvl="INFO_DETAIL", tile_id=tile_id, format_type='float32', filename=os.path.basename(temp_fits_filepath))
        # pcb_tile(f"{func_id_log_base}_info_saving_finished", prog=None, lvl="DEBUG_DETAIL", tile_id=tile_id)
        try:
            _PH3_CONCURRENCY_SEMAPHORE.release()
        except Exception:
            pass
        return (temp_fits_filepath, wcs_for_master_tile), failed_groups_to_retry
        
    except Exception as e_save_mt:
        pcb_tile(f"{func_id_log_base}_error_saving", prog=None, lvl="ERROR", tile_id=tile_id, error=str(e_save_mt))
        logger.error(f"Traceback pour {func_id_log_base}_{tile_id} sauvegarde:", exc_info=True)
        try:
            _PH3_CONCURRENCY_SEMAPHORE.release()
        except Exception:
            pass
        return (None, None), failed_groups_to_retry
    finally:
        if 'master_tile_stacked_HWC' in locals() and master_tile_stacked_HWC is not None: 
            del master_tile_stacked_HWC
        gc.collect()



# Dans zemosaic_worker.py

# ... (s'assurer que zemosaic_utils est importé et ZEMOSAIC_UTILS_AVAILABLE est défini)
# ... (s'assurer que WCS, fits d'Astropy sont importés, ainsi que reproject_interp)
# ... (définition de logger, _log_and_callback, etc.)



def assemble_final_mosaic_incremental(
    master_tile_fits_with_wcs_list: list,
    final_output_wcs: WCS,
    final_output_shape_hw: tuple,
    progress_callback: callable,
    n_channels: int = 3,
    dtype_accumulator: np.dtype = np.float64,
    dtype_norm: np.dtype = np.float32,
    apply_crop: bool = False,
    crop_percent: float = 0.0,
    processing_threads: int = 0,
    memmap_dir: str | None = None,
    cleanup_memmap: bool = True,
    intertile_photometric_match: bool = False,
    intertile_preview_size: int = 512,
    intertile_overlap_min: float = 0.05,
    intertile_sky_percentile: tuple[float, float] | list[float] = (30.0, 70.0),
    intertile_robust_clip_sigma: float = 2.5,
    intertile_global_recenter: bool = True,
    intertile_recenter_clip: tuple[float, float] | list[float] = (0.85, 1.18),
    use_auto_intertile: bool = False,
    match_background: bool = True,
    feather_parity: bool = False,
    use_radial_feather: bool | None = None,
    two_pass_coverage_renorm: bool = False,
    tile_affine_corrections: list[tuple[float, float]] | None = None,
    enforce_positive: bool | None = None,
    base_progress_phase5: float | None = None,
    progress_weight_phase5: float | None = None,
    start_time_total_run: float | None = None,
    global_anchor_shift: tuple[float, float] | None = None,
):
    """Assemble les master tiles par co-addition sur disque."""
    import time
    # Marquer le début de la phase 5 incrémentale
    start_time_inc = time.monotonic()
    total_tiles = len(master_tile_fits_with_wcs_list)
    FLUSH_BATCH_SIZE = 10  # nombre de tuiles entre chaque flush sur le memmap
    pcb_asm = lambda msg_key, prog=None, lvl="INFO_DETAIL", **kwargs: _log_and_callback(
        msg_key, prog, lvl, callback=progress_callback, **kwargs
    )

    pending_affine_list: list[tuple[float, float]] | None = None
    nontrivial_detected = False
    if tile_affine_corrections:
        pending_affine_list, nontrivial_detected = _sanitize_affine_corrections(
            tile_affine_corrections,
            len(master_tile_fits_with_wcs_list),
        )

    # Default to radial feathering for parity with the legacy pipeline unless explicitly disabled.
    use_feather = True if use_radial_feather is None else bool(use_radial_feather)
    parity_mode = bool(intertile_photometric_match and match_background and feather_parity)
    if feather_parity:
        if use_feather:
            use_feather = False
        pcb_asm("run_warn_incremental_feather_parity_enabled", prog=None, lvl="WARN")

    if enforce_positive is None:
        enforce_positive_flag = not parity_mode
    else:
        enforce_positive_flag = bool(enforce_positive)

    if progress_weight_phase5 is None:
        progress_weight_phase5 = globals().get("PROGRESS_WEIGHT_PHASE5_ASSEMBLY", 0.0)
    if base_progress_phase5 is None:
        base_progress_phase5 = 0.0

    pcb_asm(
        f"ASM_INC: Début. Options rognage - Appliquer: {apply_crop}, %: {crop_percent if apply_crop else 'N/A'}",
        lvl="DEBUG_DETAIL",
    )

    if not (REPROJECT_AVAILABLE and reproject_interp and ASTROPY_AVAILABLE and fits):
        missing_deps = []
        if not REPROJECT_AVAILABLE or not reproject_interp:
            missing_deps.append("Reproject (reproject_interp)")
        if not ASTROPY_AVAILABLE or not fits:
            missing_deps.append("Astropy (fits)")
        pcb_asm(
            "assemble_error_core_deps_unavailable_incremental",
            prog=None,
            lvl="ERROR",
            missing=", ".join(missing_deps),
        )
        return None, None

    if not master_tile_fits_with_wcs_list:
        pcb_asm("assemble_error_no_tiles_provided_incremental", prog=None, lvl="ERROR")
        return None, None

    # ``final_output_shape_hw`` MUST be provided in ``(height, width)`` order.
    if (
        not isinstance(final_output_shape_hw, (tuple, list))
        or len(final_output_shape_hw) != 2
    ):
        pcb_asm(
            "assemble_error_invalid_final_shape_inc",
            prog=None,
            lvl="ERROR",
            shape=str(final_output_shape_hw),
        )
        return None, None

    h, w = map(int, final_output_shape_hw)

    # --- Extra validation to help catch swapped width/height ---
    try:
        w_wcs = int(getattr(final_output_wcs, "pixel_shape", (w, h))[0])
        h_wcs = int(getattr(final_output_wcs, "pixel_shape", (w, h))[1])
    except Exception:
        w_wcs = int(getattr(final_output_wcs.wcs, "naxis1", w)) if hasattr(final_output_wcs, "wcs") else w
        h_wcs = int(getattr(final_output_wcs.wcs, "naxis2", h)) if hasattr(final_output_wcs, "wcs") else h

    expected_hw = (h_wcs, w_wcs)
    if (h, w) != expected_hw:
        if (w, h) == expected_hw:
            pcb_asm(
                "assemble_warn_swapped_final_shape_inc",
                prog=None,
                lvl="WARN",
                provided=str(final_output_shape_hw),
                expected=str(expected_hw),
            )
            h, w = expected_hw
        else:
            pcb_asm(
                "assemble_error_mismatch_final_shape_inc",
                prog=None,
                lvl="ERROR",
                provided=str(final_output_shape_hw),
                expected=str(expected_hw),
            )
            return None, None

    if match_background:
        pcb_asm("run_info_incremental_match_background", prog=None, lvl="INFO_DETAIL")

    if (
        intertile_photometric_match
        and pending_affine_list is None
        and total_tiles >= 2
        and ZEMOSAIC_UTILS_AVAILABLE
        and hasattr(zemosaic_utils, "create_downscaled_luminance_preview")
        and hasattr(zemosaic_utils, "compute_intertile_affine_calibration")
    ):
        affine_start = time.monotonic()
        pcb_asm(
            "run_info_incremental_affine_start",
            prog=None,
            lvl="INFO",
            num_tiles=total_tiles,
        )
        tile_sources = [
            _TileAffineSource(path=tile_path, wcs=tile_wcs)
            for tile_path, tile_wcs in master_tile_fits_with_wcs_list
        ]
        pending_affine_list, nontrivial_detected, affine_status, affine_error = (
            _compute_intertile_affine_corrections_from_sources(
                sources=tile_sources,
                final_output_wcs=final_output_wcs,
                final_output_shape_hw=final_output_shape_hw,
                preview_size=int(intertile_preview_size),
                min_overlap_fraction=float(intertile_overlap_min),
                sky_percentile=intertile_sky_percentile,
                robust_clip_sigma=float(intertile_robust_clip_sigma),
                use_auto_intertile=use_auto_intertile,
                logger_obj=logger,
                progress_callback=progress_callback,
                intertile_global_recenter=bool(intertile_global_recenter),
                intertile_recenter_clip=intertile_recenter_clip,
            )
        )
        if affine_status == "preview_failed":
            pcb_asm(
                "assemble_warn_intertile_photometric_failed",
                prog=None,
                lvl="WARN",
                error="preview_failed",
            )
            pending_affine_list = None
            nontrivial_detected = False
        elif affine_status == "compute_failed":
            pcb_asm(
                "assemble_warn_intertile_photometric_failed",
                prog=None,
                lvl="WARN",
                error=str(affine_error),
            )
            pending_affine_list = None
            nontrivial_detected = False
        affine_elapsed = time.monotonic() - affine_start
        if nontrivial_detected:
            try:
                pcb_asm(
                    "assemble_info_intertile_photometric_applied",
                    prog=None,
                    lvl="INFO_DETAIL",
                    num_tiles=sum(
                        1
                        for g_val, o_val in pending_affine_list or []
                        if abs(g_val - 1.0) > 1e-6 or abs(o_val) > 1e-6
                    ),
                )
            except Exception:
                pass
        pcb_asm(
            "run_info_incremental_affine_done",
            prog=None,
            lvl="INFO",
            elapsed=f"{affine_elapsed:.2f}",
            num_tiles=len(pending_affine_list or []),
        )
    pending_affine_list, anchor_shift_applied = _compose_global_anchor_shift(
        pending_affine_list,
        total_tiles,
        global_anchor_shift,
    )
    if anchor_shift_applied:
        nontrivial_detected = True

    affine_log_indices = _select_affine_log_indices(pending_affine_list)

    sum_shape = (h, w, n_channels)
    weight_shape = (h, w)

    internal_temp_dir = False
    if memmap_dir is None:
        memmap_dir = tempfile.mkdtemp(prefix="zemosaic_memmap_")
        internal_temp_dir = True
    else:
        os.makedirs(memmap_dir, exist_ok=True)
    sum_path = os.path.join(memmap_dir, "SOMME.fits")
    weight_path = os.path.join(memmap_dir, "WEIGHT.fits")

    try:
        fits.writeto(sum_path, np.zeros(sum_shape, dtype=dtype_accumulator), overwrite=True)
        fits.writeto(weight_path, np.zeros(weight_shape, dtype=dtype_norm), overwrite=True)
    except Exception as e_create:
        pcb_asm("assemble_error_memmap_write_failed_inc", prog=None, lvl="ERROR", error=str(e_create))
        logger.error("Failed to create memmap FITS", exc_info=True)
        return None, None


    try:
        req_workers = int(processing_threads)
    except Exception:
        req_workers = 0
    if req_workers > 0:
        max_procs = req_workers
    else:
        max_procs = min(os.cpu_count() or 1, len(master_tile_fits_with_wcs_list))
    pcb_asm(f"ASM_INC: Using {max_procs} process workers", lvl="DEBUG_DETAIL")

    parent_is_daemon = multiprocessing.current_process().daemon
    Executor = ThreadPoolExecutor if parent_is_daemon else ProcessPoolExecutor


    try:
        with Executor(max_workers=max_procs) as ex, \
                fits.open(sum_path, mode="update", memmap=True) as hsum, \
                fits.open(weight_path, mode="update", memmap=True) as hwei:
            fsum = hsum[0].data
            fwei = hwei[0].data

            tiles_since_flush = 0

            future_map = {}
            for tile_idx, (tile_path, tile_wcs) in enumerate(master_tile_fits_with_wcs_list, 1):
                pcb_asm(
                    "assemble_info_processing_tile",
                    prog=None,
                    lvl="INFO_DETAIL",
                    tile_num=tile_idx,
                    total_tiles=len(master_tile_fits_with_wcs_list),
                    filename=os.path.basename(tile_path),
                )
                # Les objets WCS peuvent poser problème lors de la sérialisation.
                # On transmet donc leurs en-têtes et ils seront reconstruits dans le worker.
                tile_wcs_hdr = tile_wcs.to_header() if hasattr(tile_wcs, "to_header") else tile_wcs
                output_wcs_hdr = final_output_wcs.to_header() if hasattr(final_output_wcs, "to_header") else final_output_wcs
                gain_val = None
                offset_val = None
                if pending_affine_list and (tile_idx - 1) < len(pending_affine_list):
                    raw_affine = pending_affine_list[tile_idx - 1]
                    try:
                        gain_val = float(raw_affine[0])
                    except Exception:
                        gain_val = 1.0
                    try:
                        offset_val = float(raw_affine[1])
                    except Exception:
                        offset_val = 0.0
                    if not np.isfinite(gain_val):
                        gain_val = 1.0
                    if not np.isfinite(offset_val):
                        offset_val = 0.0
                    if (
                        affine_log_indices
                        and tile_idx in affine_log_indices
                        and (abs(gain_val - 1.0) > 1e-6 or abs(offset_val) > 1e-6)
                    ):
                        try:
                            pcb_asm(
                                "run_info_incremental_apply_gain_offset",
                                prog=None,
                                lvl="INFO_DETAIL",
                                tile_num=tile_idx,
                                gain=f"{gain_val:.6f}",
                                offset=f"{offset_val:.6f}",
                            )
                        except Exception:
                            pass
                future = ex.submit(
                    reproject_tile_to_mosaic,
                    tile_path,
                    tile_wcs_hdr,
                    output_wcs_hdr,
                    final_output_shape_hw,
                    feather=use_feather,
                    apply_crop=apply_crop,
                    crop_percent=crop_percent,
                    tile_affine=None,
                    gain=gain_val,
                    offset=offset_val,
                    match_background=match_background,
                    nan_fill_value=0.0,
                    enforce_positive=enforce_positive_flag,
                )
                future_map[future] = tile_idx

            processed = 0
            total_steps = len(future_map)
            start_time_iter = time.time()
            last_time = start_time_iter
            step_times = []
            for fut in as_completed(future_map):
                idx = future_map[fut]
                try:
                    # reproject_tile_to_mosaic renvoie les bornes de la tuile
                    # sous la forme (xmin, xmax, ymin, ymax) afin de
                    # correspondre aux indices de colonne puis de ligne.
                    I_tile, W_tile, (xmin, xmax, ymin, ymax) = fut.result()
                except MemoryError as e_mem:
                    pcb_asm(
                        "assemble_error_memory_tile_reprojection_inc",
                        prog=None,
                        lvl="ERROR",
                        tile_num=idx,
                        error=str(e_mem),
                    )
                    logger.error(
                        f"MemoryError reproject_tile_to_mosaic tuile {idx}",
                        exc_info=True,
                    )
                    processed += 1
                    continue
                except BrokenProcessPool as bpp:
                    pcb_asm(
                        "assemble_error_broken_process_pool_incremental",
                        prog=None,
                        lvl="ERROR",
                        tile_num=idx,
                        error=str(bpp),
                    )
                    logger.error(
                        "BrokenProcessPool during tile reprojection",
                        exc_info=True,
                    )
                    return None, None
                except Exception as e_reproj:
                    pcb_asm(
                        "assemble_error_tile_reprojection_failed_inc",
                        prog=None,
                        lvl="ERROR",
                        tile_num=idx,
                        error=str(e_reproj),
                    )
                    logger.error(
                        f"Erreur reproject_tile_to_mosaic tuile {idx}",
                        exc_info=True,
                    )
                    processed += 1
                    continue

                if I_tile is not None and W_tile is not None:
                    mask = W_tile > 0
                    tgt_sum = fsum[ymin:ymax, xmin:xmax]
                    tgt_wgt = fwei[ymin:ymax, xmin:xmax]
                    for c in range(n_channels):
                        tgt_sum[..., c][mask] += I_tile[..., c][mask] * W_tile[mask]
                    tgt_wgt[mask] += W_tile[mask]
                    tiles_since_flush += 1
                    if tiles_since_flush >= FLUSH_BATCH_SIZE:
                        hsum.flush()
                        hwei.flush()
                        tiles_since_flush = 0

                processed += 1
                now = time.time()
                step_times.append(now - last_time)
                last_time = now
                if progress_callback:
                    try:
                        progress_callback("phase5_incremental", processed, total_steps)
                    except Exception:
                        pass
                if processed % FLUSH_BATCH_SIZE == 0 or processed == total_tiles:
                    pcb_asm(
                        "assemble_progress_tiles_processed_inc",
                        prog=None,
                        lvl="INFO_DETAIL",
                        num_done=processed,
                        total_num=total_tiles,
                    )

                    # --- Calcul et mise à jour de l’ETA global ---
                    elapsed_inc = time.monotonic() - start_time_inc
                    time_per_tile = elapsed_inc / processed
                    eta_tiles_sec = (total_tiles - processed) * time_per_tile

                    if (
                        progress_weight_phase5
                        and start_time_total_run is not None
                        and total_tiles > 0
                    ):
                        try:
                            current_progress_pct = base_progress_phase5 + (
                                (processed / total_tiles) * progress_weight_phase5
                            )
                            current_progress_pct = max(
                                current_progress_pct,
                                base_progress_phase5 + 0.01,
                            )
                            elapsed_total = time.monotonic() - start_time_total_run
                            sec_per_pct = (
                                elapsed_total / current_progress_pct
                                if current_progress_pct > 0
                                else 0.0
                            )
                            total_eta_sec = eta_tiles_sec + (
                                max(0.0, 100 - current_progress_pct) * sec_per_pct
                            )
                            update_gui_eta(total_eta_sec)
                        except Exception:
                            pass

            if tiles_since_flush > 0:
                hsum.flush()
                hwei.flush()
                tiles_since_flush = 0
    except Exception as e_pool:
        pcb_asm("assemble_error_incremental_pool_failed", prog=None, lvl="ERROR", error=str(e_pool))
        logger.error("Error during incremental assembly", exc_info=True)
        return None, None

    with fits.open(sum_path, memmap=True) as hsum, fits.open(weight_path, memmap=True) as hwei:
        sum_data = hsum[0].data.astype(np.float32)
        weight_data = hwei[0].data.astype(np.float32)
        mosaic = np.zeros_like(sum_data, dtype=np.float32)
        np.divide(sum_data, weight_data[..., None], out=mosaic, where=weight_data[..., None] > 0)

    if step_times:
        avg_step = sum(step_times) / len(step_times)
        total_elapsed = time.time() - start_time_iter
        pcb_asm(
            "assemble_debug_incremental_timing",
            prog=None,
            lvl="DEBUG_DETAIL",
            avg=f"{avg_step:.2f}",
            total=f"{total_elapsed:.2f}",
        )

    pcb_asm("assemble_info_finished_incremental", prog=None, lvl="INFO", shape=str(mosaic.shape))

    # Harmonize incremental output with reproject/coadd by removing empty borders
    try:
        mosaic, weight_data = _auto_crop_mosaic_to_valid_region(
            mosaic,
            weight_data,
            final_output_wcs,
            log_callback=pcb_asm,
        )
        pcb_asm(
            "assemble_info_incremental_autocrop_done",
            prog=None,
            lvl="INFO_DETAIL",
            shape=str(getattr(mosaic, "shape", None)),
        )
    except Exception as exc_crop:
        pcb_asm(
            "assemble_warn_incremental_autocrop_failed",
            prog=None,
            lvl="WARN",
            error=str(exc_crop),
        )

    if mosaic.ndim != 3:
        raise ValueError(f"Expected incremental mosaic in HWC order, got {mosaic.shape}")
    logger.debug("Mosaic shape (HWC): %s", mosaic.shape)
    if logger.isEnabledFor(logging.DEBUG):
        mask = weight_data > 0
        if np.any(mask):
            stats = []
            for c in range(mosaic.shape[-1]):
                vals = mosaic[..., c][mask]
                if vals.size:
                    stats.append((float(np.mean(vals)), float(np.std(vals))))
                else:
                    stats.append((float("nan"), float("nan")))
            logger.debug(
                "Incremental overlap stats (mean/std per channel): %s",
                stats,
            )

    if cleanup_memmap:
        for p in (sum_path, weight_path):
            try:
                os.remove(p)
            except OSError:
                pass

        if internal_temp_dir:
            try:
                os.rmdir(memmap_dir)
            except OSError:
                pass


    return mosaic, weight_data

def _reproject_and_coadd_channel_worker(channel_data_list, output_wcs_header, output_shape_hw, match_bg, mm_sum_prefix=None, mm_cov_prefix=None):
    """Worker function to run reproject_and_coadd in a separate process."""
    from astropy.wcs import WCS
    from reproject import reproject_interp
    import numpy as np

    final_wcs = WCS(output_wcs_header)
    data_list = []
    wcs_list = []
    for arr, hdr in channel_data_list:
        data_list.append(arr)
        wcs_list.append(WCS(hdr))




    # The memmap prefixes are produced by other workers. Ensure they exist before
    # reading if provided. Wait here until both files are fully written.

    import inspect
    sig = inspect.signature(reproject_and_coadd)
    bg_kw = "match_background" if "match_background" in sig.parameters else (
        "match_bg" if "match_bg" in sig.parameters else None
    )

    kwargs = {
        "output_projection": final_wcs,
        "shape_out": output_shape_hw,
        "reproject_function": reproject_interp,
        "combine_function": "mean",
    }
    if bg_kw:
        kwargs[bg_kw] = match_bg

    stacked, coverage = reproject_and_coadd_wrapper(
        data_list=data_list,
        wcs_list=wcs_list,
        shape_out=output_shape_hw,
        output_projection=final_wcs,
        use_gpu=False,
        cpu_func=reproject_and_coadd,
        **kwargs,
    )

    if mm_sum_prefix and mm_cov_prefix:
        _wait_for_memmap_files([mm_sum_prefix, mm_cov_prefix])
    return stacked.astype(np.float32), coverage.astype(np.float32)


def assemble_final_mosaic_reproject_coadd(
    master_tile_fits_with_wcs_list: list,
    final_output_wcs: WCS,
    final_output_shape_hw: tuple,
    progress_callback: callable,
    n_channels: int = 3,
    match_bg: bool = True,
    apply_crop: bool = False,
    crop_percent: float = 0.0,
    use_memmap: bool = False,
    memmap_dir: str | None = None,
    cleanup_memmap: bool = True,
    assembly_process_workers: int = 0,
    re_solve_cropped_tiles: bool = False,
    solver_settings: dict | None = None,
    solver_instance=None,
    use_gpu: bool = False,
    base_progress_phase5: float | None = None,
    progress_weight_phase5: float | None = None,
    start_time_total_run: float | None = None,
    intertile_photometric_match: bool = False,
    intertile_preview_size: int = 512,
    intertile_overlap_min: float = 0.05,
    intertile_sky_percentile: tuple[float, float] | list[float] = (30.0, 70.0),
    intertile_robust_clip_sigma: float = 2.5,
    intertile_global_recenter: bool = True,
    intertile_recenter_clip: tuple[float, float] | list[float] | None = (0.85, 1.18),
    use_auto_intertile: bool = False,
    collect_tile_data: list | None = None,
    tile_affine_corrections: list[tuple[float, float]] | None = None,
    global_anchor_shift: tuple[float, float] | None = None,
    phase45_enabled: bool = False,
):
    """Assemble les master tiles en utilisant ``reproject_and_coadd``."""
    _pcb = lambda msg_key, prog=None, lvl="INFO_DETAIL", **kwargs: _log_and_callback(
        msg_key, prog, lvl, callback=progress_callback, **kwargs
    )

    _log_memory_usage(progress_callback, "Début assemble_final_mosaic_reproject_coadd")
    _pcb(
        f"ASM_REPROJ_COADD: Options de rognage - Appliquer: {apply_crop}, Pourcentage: {crop_percent if apply_crop else 'N/A'}",
        lvl="DEBUG_DETAIL",
    )

    start_time_phase = time.monotonic()

    # Emit ETA during the preparation phase (before channels start)
    def _update_eta_prepare(done_tiles: int, total_tiles_local: int):
        if (
            base_progress_phase5 is None
            or progress_weight_phase5 is None
            or start_time_total_run is None
        ):
            return
        try:
            prep_fraction = 0.0
            if total_tiles_local > 0:
                prep_fraction = max(0.0, min(1.0, float(done_tiles) / float(total_tiles_local)))
            # Use a small pseudo progress for ETA only to avoid 0%% division
            current_progress_pct = base_progress_phase5 + (0.1 * prep_fraction) * progress_weight_phase5
            current_progress_pct = max(current_progress_pct, base_progress_phase5 + 0.01)
            elapsed_phase_local = time.monotonic() - start_time_phase
            eta_pre_sec = 0.0
            if done_tiles > 0 and total_tiles_local > 0:
                time_per_tile = elapsed_phase_local / float(done_tiles)
                eta_pre_sec = max(0.0, (total_tiles_local - done_tiles) * time_per_tile)
            elapsed_total = time.monotonic() - start_time_total_run
            sec_per_pct = elapsed_total / max(1.0, current_progress_pct)
            total_eta_sec = eta_pre_sec + (100 - current_progress_pct) * sec_per_pct
            h, rem = divmod(int(total_eta_sec), 3600)
            m, s = divmod(rem, 60)
            _pcb(f"ETA_UPDATE:{h:02d}:{m:02d}:{s:02d}", prog=None, lvl="ETA_LEVEL")
        except Exception:
            pass

    def _update_eta(completed_channels: int):
        if (
            base_progress_phase5 is not None
            and progress_weight_phase5 is not None
            and start_time_total_run is not None
            and completed_channels > 0
        ):
            elapsed_phase = time.monotonic() - start_time_phase
            time_per_ch = elapsed_phase / completed_channels
            eta_ch_sec = (n_channels - completed_channels) * time_per_ch
            current_progress_pct = base_progress_phase5 + (
                completed_channels / n_channels
            ) * progress_weight_phase5
            elapsed_total = time.monotonic() - start_time_total_run
            # Avoid zero-division at early stage; use at least 1%% of run for denominator
            sec_per_pct = elapsed_total / max(1.0, current_progress_pct)
            total_eta_sec = eta_ch_sec + (100 - current_progress_pct) * sec_per_pct
            h, rem = divmod(int(total_eta_sec), 3600)
            m, s = divmod(rem, 60)
            _pcb(
                f"ETA_UPDATE:{h:02d}:{m:02d}:{s:02d}",
                prog=None,
                lvl="ETA_LEVEL",
            )

    # Ensure wrapper uses the possibly monkeypatched CPU implementation
    try:
        zemosaic_utils.cpu_reproject_and_coadd = reproject_and_coadd
    except Exception:
        pass


    if not (REPROJECT_AVAILABLE and reproject_and_coadd and ASTROPY_AVAILABLE and fits):
        missing_deps = []
        if not REPROJECT_AVAILABLE or not reproject_and_coadd:
            missing_deps.append("Reproject")
        if not ASTROPY_AVAILABLE or not fits:
            missing_deps.append("Astropy (fits)")
        _pcb(
            "assemble_error_core_deps_unavailable_reproject_coadd",
            prog=None,
            lvl="ERROR",
            missing=", ".join(missing_deps),
        )
        return None, None

    if not master_tile_fits_with_wcs_list:
        _pcb("assemble_error_no_tiles_provided_reproject_coadd", prog=None, lvl="ERROR")
        return None, None

    if (
        not isinstance(final_output_shape_hw, (tuple, list))
        or len(final_output_shape_hw) != 2
    ):
        _pcb(
            "assemble_error_invalid_final_shape_reproj_coadd",
            prog=None,
            lvl="ERROR",
            shape=str(final_output_shape_hw),
        )
        return None, None

    h, w = map(int, final_output_shape_hw)

    try:
        w_wcs = int(getattr(final_output_wcs, "pixel_shape", (w, h))[0])
        h_wcs = int(getattr(final_output_wcs, "pixel_shape", (w, h))[1])
    except Exception:
        w_wcs = int(getattr(final_output_wcs.wcs, "naxis1", w)) if hasattr(final_output_wcs, "wcs") else w
        h_wcs = int(getattr(final_output_wcs.wcs, "naxis2", h)) if hasattr(final_output_wcs, "wcs") else h

    expected_hw = (h_wcs, w_wcs)
    if (h, w) != expected_hw:
        if (w, h) == expected_hw:
            _pcb(
                "assemble_warn_swapped_final_shape_reproj_coadd",
                prog=None,
                lvl="WARN",
                provided=str(final_output_shape_hw),
                expected=str(expected_hw),
            )
            h, w = expected_hw
            final_output_shape_hw = (h, w)
        else:
            _pcb(
                "assemble_error_mismatch_final_shape_reproj_coadd",
                prog=None,
                lvl="ERROR",
                provided=str(final_output_shape_hw),
                expected=str(expected_hw),
            )
            return None, None

    # Convertir la sortie WCS en header FITS si possible une seule fois
    output_header = (
        final_output_wcs.to_header()
        if hasattr(final_output_wcs, "to_header")
        else final_output_wcs
    )


    effective_tiles: list[dict[str, Any]] = []
    hdr_for_output = None
    total_tiles_for_prep = len(master_tile_fits_with_wcs_list)
    for idx, (tile_path, tile_wcs) in enumerate(master_tile_fits_with_wcs_list, 1):
        tile_header = None
        with fits.open(tile_path, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float32)
            try:
                tile_header = hdul[0].header.copy()
            except Exception:
                tile_header = None

        # Master tiles saved via ``save_fits_image`` use the ``HWC`` axis order
        # which stores color images in ``C x H x W`` within the FITS file. When
        # reading them back for final assembly we expect ``H x W x C``.
        if data.ndim == 3 and data.shape[0] in (1, 3) and data.shape[-1] != data.shape[0]:
            data = np.moveaxis(data, 0, -1)
        if data.ndim == 2:
            data = data[..., np.newaxis]

        if (
            apply_crop
            and crop_percent > 1e-3
            and ZEMOSAIC_UTILS_AVAILABLE
            and hasattr(zemosaic_utils, "crop_image_and_wcs")
        ):
            try:
                cropped, cropped_wcs = zemosaic_utils.crop_image_and_wcs(
                    data,
                    tile_wcs,
                    crop_percent / 100.0,
                    progress_callback=None,
                )
                if cropped is not None and cropped_wcs is not None:
                    data = cropped
                    tile_wcs = cropped_wcs
            except Exception:
                pass

        if re_solve_cropped_tiles and solver_instance is not None:
            try:
                hdr = fits.Header()
                hdr['BITPIX'] = -32
                if 'BSCALE' in hdr:
                    del hdr['BSCALE']
                if 'BZERO' in hdr:
                    del hdr['BZERO']
                use_hints = solver_settings.get("use_radec_hints", False) if solver_settings else False
                if use_hints and hasattr(tile_wcs, "wcs"):
                    cx = tile_wcs.pixel_shape[0] / 2
                    cy = tile_wcs.pixel_shape[1] / 2
                    ra_dec = tile_wcs.wcs_pix2world([[cx, cy]], 0)[0]
                    hdr["RA"] = ra_dec[0]
                    hdr["DEC"] = ra_dec[1]
                solver_instance.solve(
                    str(tile_path), hdr, solver_settings or {}, update_header_with_solution=True
                )
                hdr_for_output = hdr
            except Exception:
                pass

        tile_entry = {
            "data": data,
            "wcs": tile_wcs,
            "path": tile_path,
            "tile_id": _resolve_tile_identifier(tile_path, tile_header, idx - 1),
        }
        effective_tiles.append(tile_entry)

        if idx % 10 == 0 or idx == len(master_tile_fits_with_wcs_list):
            _pcb(
                "assemble_progress_tiles_processed_inc",
                prog=None,
                lvl="INFO_DETAIL",
                num_done=idx,
                total_num=len(master_tile_fits_with_wcs_list),
            )

        # Keep ETA responsive during preparation
        if idx == 1 or (idx % 5 == 0) or (idx == total_tiles_for_prep):
            _update_eta_prepare(idx, total_tiles_for_prep)



    # Optional inter-tile photometric (gain/offset) calibration
    pending_affine_list, nontrivial_affine = _sanitize_affine_corrections(
        tile_affine_corrections,
        len(effective_tiles),
    )

    if (
        pending_affine_list is None
        and intertile_photometric_match
        and len(effective_tiles) >= 2
    ):
        tile_sources = []
        for entry in effective_tiles:
            tile_sources.append(
                _TileAffineSource(
                    path=entry.get("path"),
                    wcs=entry.get("wcs"),
                    data=entry.get("data"),
                )
            )

        pending_affine_list, nontrivial_affine, affine_status, affine_error = (
            _compute_intertile_affine_corrections_from_sources(
                sources=tile_sources,
                final_output_wcs=final_output_wcs,
                final_output_shape_hw=final_output_shape_hw,
                preview_size=int(intertile_preview_size),
                min_overlap_fraction=float(intertile_overlap_min),
                sky_percentile=intertile_sky_percentile,
                robust_clip_sigma=float(intertile_robust_clip_sigma),
                use_auto_intertile=use_auto_intertile,
                logger_obj=logger,
                progress_callback=progress_callback,
                intertile_global_recenter=bool(intertile_global_recenter),
                intertile_recenter_clip=intertile_recenter_clip,
            )
        )

        if affine_status == "preview_failed":
            _pcb(
                "assemble_warn_intertile_photometric_failed",
                prog=None,
                lvl="WARN",
                error="preview_failed",
            )
            pending_affine_list = None
            nontrivial_affine = False
        elif affine_status == "compute_failed":
            _pcb(
                "assemble_warn_intertile_photometric_failed",
                prog=None,
                lvl="WARN",
                error=str(affine_error),
            )
            pending_affine_list = None
            nontrivial_affine = False

    total_tiles_prepared = len(effective_tiles)
    pending_affine_list, anchor_shift_applied = _compose_global_anchor_shift(
        pending_affine_list,
        total_tiles_prepared,
        global_anchor_shift,
    )
    if anchor_shift_applied:
        nontrivial_affine = True

    affine_by_id: dict[str, tuple[float, float]] | None = None
    if pending_affine_list:
        affine_by_id, affine_mismatch_detail = _build_affine_lookup_for_tiles(
            effective_tiles,
            pending_affine_list,
        )
        if affine_mismatch_detail:
            logger.warning(
                "affine_list_mismatch: expected %d corrections for %d tiles, got %d — skip photometric corrections and rely on match_background",
                len(effective_tiles),
                len(effective_tiles),
                len(pending_affine_list or []),
            )
            logger.warning("affine_list_mismatch detail: %s", affine_mismatch_detail)
            pending_affine_list = None
            affine_by_id = None
            nontrivial_affine = False
        else:
            if phase45_enabled:
                logger.info(
                    "intertile: recomputed affine corrections on effective set: tiles=%d",
                    len(effective_tiles),
                )
            logger.info(
                "apply_photometric: using affine_by_id for %d tiles",
                len(effective_tiles),
            )
    else:
        nontrivial_affine = False
        pending_affine_list = None

    if pending_affine_list and affine_by_id:
        if use_gpu:
            nontrivial_affine = True
            for entry in effective_tiles:
                tile_id = entry.get("tile_id")
                if not tile_id:
                    continue
                gain_val, offset_val = affine_by_id.get(tile_id, (1.0, 0.0))
                logger.info(
                    "apply_photometric: tile=%s gain=%.5f offset=%.5f",
                    tile_id,
                    gain_val,
                    offset_val,
                )
        else:
            corrected_tiles = 0
            for tile_entry in effective_tiles:
                tile_id = tile_entry.get("tile_id")
                if not tile_id or tile_entry.get("data") is None:
                    continue
                gain_val, offset_val = affine_by_id.get(tile_id, (1.0, 0.0))
                try:
                    arr_np = np.asarray(tile_entry["data"], dtype=np.float32, order="C")
                    gain_to_apply = float(gain_val)
                    offset_to_apply = float(offset_val)
                    if match_bg:
                        gain_before = gain_to_apply
                        offset_before = offset_to_apply
                        if gain_to_apply < gain_limit_min:
                            gain_to_apply = gain_limit_min
                        elif gain_to_apply > gain_limit_max:
                            gain_to_apply = gain_limit_max
                        if affine_offset_limit_adu > 0.0:
                            if abs(offset_to_apply) > affine_offset_limit_adu:
                                offset_to_apply = 0.0
                            else:
                                offset_to_apply = max(-affine_offset_limit_adu, min(offset_to_apply, affine_offset_limit_adu))
                        if gain_to_apply != gain_before or offset_to_apply != offset_before:
                            try:
                                _pcb(
                                    "assemble_warn_affine_clamped",
                                    prog=None,
                                    lvl="INFO_DETAIL",
                                    tile_id=tile_id,
                                    gain_before=gain_before,
                                    gain_after=gain_to_apply,
                                    offset_before=offset_before,
                                    offset_after=offset_to_apply,
                                )
                            except Exception:
                                pass
                    if gain_to_apply != 1.0:
                        np.multiply(arr_np, gain_to_apply, out=arr_np, casting="unsafe")
                    if offset_to_apply != 0.0:
                        np.add(arr_np, offset_to_apply, out=arr_np, casting="unsafe")
                    tile_entry["data"] = arr_np
                    corrected_tiles += 1
                    logger.info(
                        "apply_photometric: tile=%s gain=%.5f offset=%.5f",
                        tile_id,
                        gain_to_apply,
                        offset_to_apply,
                    )
                except Exception:
                    continue
            if corrected_tiles:
                try:
                    _pcb(
                        "assemble_info_intertile_photometric_applied",
                        prog=None,
                        lvl="INFO_DETAIL",
                        num_tiles=corrected_tiles,
                    )
                except Exception:
                    pass
                nontrivial_affine = True
            else:
                nontrivial_affine = False
                pending_affine_list = None
    else:
        nontrivial_affine = False
        pending_affine_list = None

    if collect_tile_data is not None:
        try:
            collect_tile_data.clear()
        except Exception:
            collect_tile_data[:] = []
        for entry in effective_tiles:
            arr = entry.get("data") if isinstance(entry, dict) else None
            tile_wcs = entry.get("wcs") if isinstance(entry, dict) else None
            if arr is None:
                continue
            try:
                collect_tile_data.append((np.array(arr, copy=True), tile_wcs))
            except Exception:
                try:
                    collect_tile_data.append((arr.copy(), tile_wcs))
                except Exception:
                    collect_tile_data.append((np.asarray(arr, dtype=np.float32), tile_wcs))


    # Build kwargs dynamically to remain compatible with different reproject versions
    reproj_kwargs = {}
    try:
        import inspect
        sig = inspect.signature(reproject_and_coadd)
        if "match_background" in sig.parameters:
            reproj_kwargs["match_background"] = match_bg
        elif "match_bg" in sig.parameters:
            reproj_kwargs["match_bg"] = match_bg
        if "process_workers" in sig.parameters:
            reproj_kwargs["process_workers"] = assembly_process_workers
        if "use_memmap" in sig.parameters:
            reproj_kwargs["use_memmap"] = use_memmap
        elif "intermediate_memmap" in sig.parameters:
            reproj_kwargs["intermediate_memmap"] = use_memmap
        if "memmap_dir" in sig.parameters:
            reproj_kwargs["memmap_dir"] = memmap_dir
        if "cleanup_memmap" in sig.parameters:
            reproj_kwargs["cleanup_memmap"] = False
    except Exception:
        # If introspection fails just fall back to basic arguments
        reproj_kwargs = {"match_background": match_bg}

    # Provide GPU-only tuning hints (safely ignored by CPU via wrapper filtering)
    try:
        reproj_kwargs["bg_preview_size"] = int(max(128, int(intertile_preview_size)))
    except Exception:
        reproj_kwargs["bg_preview_size"] = 512
    try:
        reproj_kwargs["intertile_sky_percentile"] = (
            tuple(intertile_sky_percentile)
            if isinstance(intertile_sky_percentile, (list, tuple)) and len(intertile_sky_percentile) >= 2
            else (30.0, 70.0)
        )
    except Exception:
        reproj_kwargs["intertile_sky_percentile"] = (30.0, 70.0)
    try:
        reproj_kwargs["intertile_robust_clip_sigma"] = float(intertile_robust_clip_sigma)
    except Exception:
        reproj_kwargs["intertile_robust_clip_sigma"] = 2.5

    # If we are going to use the GPU, pass the precomputed affine corrections down
    # so they are applied inside the GPU reprojection (parity with CPU path).
    if use_gpu and pending_affine_list is not None:
        reproj_kwargs["tile_affine_corrections"] = pending_affine_list
        try:
            _pcb(
                f"ASM_REPROJ_COADD: Passing intertile affine corrections to GPU (n={len(pending_affine_list)})",
                lvl="DEBUG_DETAIL",
            )
        except Exception:
            pass


    # Prepare output containers: either RAM lists or disk-backed memmaps
    mosaic_channels = []
    coverage = None
    mosaic_memmap = None
    coverage_memmap = None
    mosaic_mm_path = None
    coverage_mm_path = None
    if use_memmap:
        try:
            mm_dir = memmap_dir or tempfile.mkdtemp(prefix="zemosaic_coadd_")
            os.makedirs(mm_dir, exist_ok=True)
            mosaic_mm_path = os.path.join(mm_dir, f"mosaic_{h}x{w}x{n_channels}.dat")
            coverage_mm_path = os.path.join(mm_dir, f"coverage_{h}x{w}.dat")
            mosaic_memmap = np.memmap(mosaic_mm_path, dtype=np.float32, mode='w+', shape=(h, w, n_channels))
            coverage_memmap = np.memmap(coverage_mm_path, dtype=np.float32, mode='w+', shape=(h, w))
            _pcb("assemble_debug_memmap_paths", prog=None, lvl="DEBUG_DETAIL", mosaic_path=mosaic_mm_path, coverage_path=coverage_mm_path)
        except Exception as e_mm:
            mosaic_memmap = None
            coverage_memmap = None
            _pcb("assemble_warn_memmap_create_failed", prog=None, lvl="WARN", error=str(e_mm))
    try:
        total_steps = n_channels
        start_time_loop = time.time()
        last_time = start_time_loop
        step_times = []
        if use_gpu:
            try:
                _pcb(
                    f"ASM_REPROJ_COADD: GPU background match enabled (preview={reproj_kwargs.get('bg_preview_size','N/A')}, sky={reproj_kwargs.get('intertile_sky_percentile','N/A')}, clip={reproj_kwargs.get('intertile_robust_clip_sigma','N/A')})",
                    lvl="DEBUG_DETAIL",
                )
            except Exception:
                pass
        for ch in range(n_channels):
            valid_entries: list[dict[str, Any]] = []
            for entry in effective_tiles:
                arr = entry.get("data") if isinstance(entry, dict) else None
                if arr is None:
                    continue
                if arr.ndim != 3:
                    raise ValueError(
                        f"Master tile data must be HWC before channel slicing, got {arr.shape}"
                    )
                if ch >= arr.shape[-1]:
                    raise ValueError(
                        f"Channel index {ch} out of bounds for tile shape {arr.shape}"
                    )
                valid_entries.append(entry)

            data_list = [entry.get("data")[..., ch] for entry in valid_entries]
            wcs_list = [entry.get("wcs") for entry in valid_entries]

            reproj_call_kwargs = dict(reproj_kwargs)
            if use_gpu:
                for unsupported_kw in ("intertile_global_recenter",):
                    if unsupported_kw in reproj_call_kwargs:
                        reproj_call_kwargs.pop(unsupported_kw, None)
                        logger.debug(
                            "[GPU Reproject] Ignoring unsupported kwarg: %s",
                            unsupported_kw,
                        )

            def _invoke_reproject(local_kwargs: dict):
                return reproject_and_coadd_wrapper(
                    data_list=data_list,
                    wcs_list=wcs_list,
                    shape_out=final_output_shape_hw,
                    output_projection=output_header,
                    use_gpu=use_gpu,
                    cpu_func=reproject_and_coadd,
                    reproject_function=reproject_interp,
                    combine_function="mean",
                    **local_kwargs,
                )

            try:
                chan_mosaic, chan_cov = _invoke_reproject(reproj_call_kwargs)
            except TypeError as gpu_kw_err:
                if use_gpu:
                    logger.warning(
                        "[GPU Reproject] Unexpected kwargs triggered TypeError: %s",
                        gpu_kw_err,
                    )
                    retry_kwargs = reproj_call_kwargs.copy()
                    removed_after_error: list[str] = []
                    err_msg = str(gpu_kw_err)
                    for key in list(retry_kwargs.keys()):
                        if f"'{key}'" in err_msg:
                            retry_kwargs.pop(key, None)
                            removed_after_error.append(key)
                    if removed_after_error:
                        logger.debug(
                            "[GPU Reproject] Retrying without kwargs: %s",
                            ", ".join(removed_after_error),
                        )
                        chan_mosaic, chan_cov = _invoke_reproject(retry_kwargs)
                    else:
                        raise
                else:
                    raise
            # Store channel result to memmap if enabled, else keep in RAM list
            ch_f32 = chan_mosaic.astype(np.float32)
            if mosaic_memmap is not None:
                mosaic_memmap[..., ch] = ch_f32
                mosaic_memmap.flush()
                del ch_f32
            else:
                mosaic_channels.append(ch_f32)

            if coverage is None:
                cov_f32 = chan_cov.astype(np.float32)
                if coverage_memmap is not None:
                    coverage_memmap[:] = cov_f32
                    coverage_memmap.flush()
                    coverage = coverage_memmap
                else:
                    coverage = cov_f32
            now = time.time()
            step_times.append(now - last_time)
            last_time = now
            if progress_callback:
                try:
                    progress_callback("phase5_reproject", ch + 1, total_steps)
                except Exception:
                    pass
            _update_eta(ch + 1)
            _log_memory_usage(progress_callback, f"Phase5 Reproject: mémoire après canal {ch+1}")
    except Exception as e_reproject:
        _pcb("assemble_error_reproject_coadd_call_failed", lvl="ERROR", error=str(e_reproject))
        logger.error(
            "Erreur fatale lors de l'appel à reproject_and_coadd:",
            exc_info=True,
        )
        return None, None

    if mosaic_memmap is not None:
        mosaic_data = mosaic_memmap
    else:
        mosaic_data = np.stack(mosaic_channels, axis=-1)
    if step_times:
        avg_step = sum(step_times) / len(step_times)
        total_elapsed = time.time() - start_time_loop
        _pcb(
            "assemble_debug_reproject_timing",
            prog=None,
            lvl="DEBUG_DETAIL",
            avg=f"{avg_step:.2f}",
            total=f"{total_elapsed:.2f}",
        )
    if re_solve_cropped_tiles and solver_instance is not None and hdr_for_output is not None:
        try:
            fits.writeto("final_mosaic.fits", mosaic_data.astype(np.float32), hdr_for_output, overwrite=True)
        except Exception:
            pass

    mosaic_data, coverage = _auto_crop_mosaic_to_valid_region(
        mosaic_data,
        coverage,
        final_output_wcs,
        log_callback=_pcb,
    )

    # Defer memmap cleanup to Phase 6 after final save

    _log_memory_usage(progress_callback, "Fin assemble_final_mosaic_reproject_coadd")
    _pcb(
        "assemble_info_finished_reproject_coadd",
        prog=None,
        lvl="INFO",
        shape=mosaic_data.shape if mosaic_data is not None else "N/A",
    )

    if mosaic_data is not None:
        if mosaic_data.ndim != 3:
            raise ValueError(f"Expected final mosaic in HWC order, got {mosaic_data.shape}")
        logger.debug("Mosaic shape (HWC): %s", mosaic_data.shape)
        if isinstance(coverage, np.ndarray) and logger.isEnabledFor(logging.DEBUG):
            mask = coverage > 0
            if np.any(mask):
                stats = []
                for c in range(mosaic_data.shape[-1]):
                    vals = mosaic_data[..., c][mask]
                    if vals.size:
                        stats.append((float(np.mean(vals)), float(np.std(vals))))
                    else:
                        stats.append((float("nan"), float("nan")))
                logger.debug(
                    "Reproject overlap stats (mean/std per channel): %s",
                    stats,
                )

    _update_eta(n_channels)

    return mosaic_data.astype(np.float32), coverage.astype(np.float32)


def _load_master_tiles_for_two_pass(
    master_tile_fits_with_wcs_list: list,
    *,
    apply_crop: bool,
    crop_percent: float,
    logger=None,
):
    """Load master tiles from disk for the coverage renormalization pass."""
    tiles: list[np.ndarray] = []
    tiles_wcs: list[Any] = []
    if not master_tile_fits_with_wcs_list:
        return tiles, tiles_wcs
    for tile_path, tile_wcs in master_tile_fits_with_wcs_list:
        if not tile_path or not os.path.exists(tile_path) or tile_wcs is None:
            continue
        try:
            with fits.open(tile_path, memmap=False) as hdul:
                data = hdul[0].data.astype(np.float32)
        except Exception as exc:
            if logger:
                logger.warning(
                    "[TwoPass] Failed to load master tile %s: %s",
                    os.path.basename(tile_path),
                    exc,
                )
            continue
        if data.ndim == 3 and data.shape[0] in (1, 3) and data.shape[-1] != data.shape[0]:
            data = np.moveaxis(data, 0, -1)
        if data.ndim == 2:
            data = data[..., np.newaxis]
        current_wcs = tile_wcs
        if (
            apply_crop
            and crop_percent > 1e-3
            and ZEMOSAIC_UTILS_AVAILABLE
            and hasattr(zemosaic_utils, "crop_image_and_wcs")
        ):
            try:
                cropped_img, cropped_wcs = zemosaic_utils.crop_image_and_wcs(
                    data,
                    copy.deepcopy(tile_wcs),
                    float(crop_percent) / 100.0,
                    progress_callback=None,
                )
                if cropped_img is not None and cropped_wcs is not None:
                    data = cropped_img
                    current_wcs = cropped_wcs
            except Exception as exc:
                if logger:
                    logger.debug(
                        "[TwoPass] Crop failed for %s: %s",
                        os.path.basename(tile_path),
                        exc,
                    )
        tiles.append(np.asarray(data, dtype=np.float32))
        tiles_wcs.append(current_wcs)
        if logger and len(tiles) <= 5:
            logger.debug(
                "[TwoPass] Loaded tile %s with shape=%s", os.path.basename(tile_path), np.asarray(data).shape
            )
    return tiles, tiles_wcs


def compute_per_tile_gains_from_coverage(
    tiles: list[np.ndarray],
    tiles_wcs: list[Any],
    final_wcs: Any,
    coverage_p1: np.ndarray,
    *,
    sigma_px: int,
    gain_clip: tuple[float, float],
    logger=None,
    use_gpu: bool = False,
) -> list[float]:
    """Compute multiplicative gains for each tile using the blurred coverage map."""
    if logger:
        logger.debug(
            "[TwoPass] compute_per_tile_gains_from_coverage start: tiles=%d, coverage_shape=%s, sigma=%s, clip=%s",
            len(tiles) if tiles else 0,
            getattr(coverage_p1, "shape", None),
            sigma_px,
            gain_clip,
        )
    if coverage_p1 is None or coverage_p1.ndim != 2:
        raise ValueError("Coverage map must be 2D for gain estimation")
    coverage = np.asarray(coverage_p1, dtype=np.float32)
    if coverage.size == 0:
        raise ValueError("Coverage map is empty")
    if logger:
        logger.debug(
            "[TwoPass] Coverage stats before blur: shape=%s, min=%.4f, max=%.4f, mean=%.4f",
            coverage.shape,
            float(np.nanmin(coverage)),
            float(np.nanmax(coverage)),
            float(np.nanmean(coverage)),
        )
    sigma_px = int(max(0, sigma_px))
    gain_min, gain_max = map(float, gain_clip)
    if gain_min > gain_max:
        gain_min, gain_max = gain_max, gain_min
    if sigma_px > 0:
        blurred = None
        blur_source = "identity"
        if use_gpu and CUPY_AVAILABLE:
            try:
                import cupy as cp  # type: ignore
                from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter  # type: ignore

                cov_gpu = cp.asarray(coverage, dtype=cp.float32)
                blurred_gpu = gpu_gaussian_filter(cov_gpu, float(sigma_px))
                blurred = cp.asnumpy(blurred_gpu)
                blur_source = "cupy"
            except Exception as exc_gpu:
                if logger:
                    logger.debug(
                        "[TwoPass] cupy gaussian_filter failed (%s), falling back to CPU",
                        exc_gpu,
                    )
        if blurred is None:
            try:
                from scipy.ndimage import gaussian_filter  # type: ignore

                blurred = gaussian_filter(coverage, sigma=float(sigma_px))
                blur_source = "scipy"
            except Exception as exc:
                if logger:
                    logger.debug(
                        "[TwoPass] scipy gaussian_filter failed (%s), trying cv2 fallback",
                        exc,
                    )
                try:
                    import cv2  # type: ignore

                    k = max(3, int(2 * round(float(sigma_px) * 1.5) + 1))
                    blurred = cv2.GaussianBlur(coverage, (k, k), sigmaX=float(sigma_px))
                    blur_source = "cv2"
                except Exception as exc_cv:
                    if logger:
                        logger.warning("[TwoPass] Coverage blur fallback failed: %s", exc_cv)
        cov_blur = (
            np.asarray(blurred, dtype=np.float32)
            if blurred is not None
            else coverage.copy()
        )
        if logger:
            logger.debug(
                "[TwoPass] Coverage blur applied with sigma=%d using %s", sigma_px, blur_source
            )
    else:
        cov_blur = coverage.copy()
        if logger:
            logger.debug("[TwoPass] Coverage blur skipped (sigma=0)")
    eps = np.finfo(np.float32).eps
    scale_map = cov_blur / np.maximum(coverage, eps)
    scale_map = np.clip(scale_map, 0.5, 2.0)
    if logger:
        logger.debug(
            "[TwoPass] Scale map stats: min=%.4f, max=%.4f, mean=%.4f",
            float(np.nanmin(scale_map)),
            float(np.nanmax(scale_map)),
            float(np.nanmean(scale_map)),
        )
    if not tiles or not tiles_wcs or len(tiles) != len(tiles_wcs):
        raise ValueError("Tile data and WCS lists must be aligned and non-empty")
    gains: list[float] = []
    for idx, (tile, tile_wcs) in enumerate(zip(tiles, tiles_wcs)):
        if tile is None or tile_wcs is None:
            gains.append(1.0)
            continue
        shape = np.asarray(tile).shape
        if shape[0] <= 0 or shape[1] <= 0:
            gains.append(1.0)
            continue
        mask = np.ones(shape[:2], dtype=np.float32)
        try:
            reproj_mask, _ = reproject_interp(
                (mask, tile_wcs),
                final_wcs,
                shape_out=coverage.shape,
            )
        except Exception as exc:
            if logger:
                logger.warning("[TwoPass] Mask reprojection failed for tile %d: %s", idx, exc)
            gains.append(1.0)
            continue
        valid = (reproj_mask > 0.1) & (coverage > 0.0)
        if not np.any(valid):
            gains.append(1.0)
            if logger and idx < 10:
                logger.debug(
                    "[TwoPass] Tile %d has no valid overlap (mask>0.1 count=%d)",
                    idx,
                    int(np.count_nonzero(reproj_mask > 0.1)),
                )
            continue
        med_gain = float(np.median(scale_map[valid]))
        gains.append(float(np.clip(med_gain, gain_min, gain_max)))
        if logger and idx < 10:
            logger.debug(
                "[TwoPass] Tile %d gain=%.4f (raw=%.4f, valid_pix=%d)",
                idx,
                gains[-1],
                med_gain,
                int(np.count_nonzero(valid)),
            )
    return gains


def run_second_pass_coverage_renorm(
    tiles: list[np.ndarray],
    tiles_wcs: list[Any],
    final_wcs_p1: Any,
    coverage_p1: np.ndarray,
    shape_out: tuple[int, int],
    *,
    sigma_px: int,
    gain_clip: tuple[float, float],
    logger=None,
    use_gpu_two_pass: bool | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Apply coverage-based gains to tiles and reproject them for a second pass."""
    if logger:
        logger.debug(
            "[TwoPass] run_second_pass_coverage_renorm start: tiles=%d, wcs=%d, coverage_shape=%s, sigma=%s, clip=%s",
            len(tiles) if tiles else 0,
            len(tiles_wcs) if tiles_wcs else 0,
            getattr(coverage_p1, "shape", None),
            sigma_px,
            gain_clip,
        )
    if not tiles or not tiles_wcs or coverage_p1 is None:
        if logger:
            logger.warning(
                "[TwoPass] Missing inputs for second pass (tiles=%s, wcs=%s, coverage=%s)",
                bool(tiles),
                bool(tiles_wcs),
                coverage_p1 is not None,
            )
        return None
    if not (REPROJECT_AVAILABLE and reproject_and_coadd and reproject_interp):
        if logger:
            logger.warning("[TwoPass] Reproject dependencies unavailable; skipping second pass")
        return None
    if not (ZEMOSAIC_UTILS_AVAILABLE and zemosaic_utils):
        if logger:
            logger.warning("[TwoPass] zemosaic_utils unavailable; skipping second pass")
        return None
    use_gpu = bool(use_gpu_two_pass)
    try:
        gains = compute_per_tile_gains_from_coverage(
            tiles,
            tiles_wcs,
            final_wcs_p1,
            coverage_p1,
            sigma_px=sigma_px,
            gain_clip=gain_clip,
            logger=logger,
            use_gpu=use_gpu,
        )
    except Exception as exc:
        if logger:
            logger.warning("[TwoPass] Gain computation failed: %s", exc, exc_info=True)
        return None
    if logger:
        finite_gains = [g for g in gains if np.isfinite(g)]
        logger.debug(
            "[TwoPass] Computed gains count=%d, finite=%d, min=%.4f, max=%.4f",
            len(gains),
            len(finite_gains),
            float(np.min(finite_gains)) if finite_gains else float("nan"),
            float(np.max(finite_gains)) if finite_gains else float("nan"),
        )
    corrected_tiles: list[np.ndarray] = []
    for arr, gain in zip(tiles, gains):
        tile_arr = np.asarray(arr, dtype=np.float32)
        if tile_arr.ndim == 2:
            tile_arr = tile_arr[..., np.newaxis]
        corrected_tiles.append(tile_arr * float(gain))
    try:
        output_projection = (
            final_wcs_p1.to_header(relax=True)
            if hasattr(final_wcs_p1, "to_header")
            else final_wcs_p1
        )
    except Exception:
        output_projection = final_wcs_p1

    reproj_kwargs: dict[str, Any] = {
        "output_projection": output_projection,
        "reproject_function": reproject_interp,
        "combine_function": "mean",
    }
    try:
        sig = inspect.signature(reproject_and_coadd)
    except Exception:
        sig = None
    if sig:
        if "match_background" in sig.parameters:
            reproj_kwargs["match_background"] = True
        elif "match_bg" in sig.parameters:
            reproj_kwargs["match_bg"] = True
        if "use_memmap" in sig.parameters:
            reproj_kwargs["use_memmap"] = False
        elif "intermediate_memmap" in sig.parameters:
            reproj_kwargs["intermediate_memmap"] = False
    else:
        reproj_kwargs["match_background"] = True

    n_channels = corrected_tiles[0].shape[-1] if corrected_tiles[0].ndim == 3 else 1
    mosaic_channels: list[np.ndarray] = []
    coverage_result: np.ndarray | None = None
    shape_out_hw = tuple(map(int, shape_out))
    for ch in range(n_channels):
        if logger:
            logger.debug(
                "[TwoPass] Reproject channel %d/%d with %d tiles (shape_out=%s)",
                ch + 1,
                n_channels,
                len(corrected_tiles),
                shape_out_hw,
            )
        data_list = [tile[..., ch] if tile.ndim == 3 else tile[..., 0] for tile in corrected_tiles]
        try:
            chan_mosaic, chan_cov = zemosaic_utils.reproject_and_coadd_wrapper(
                data_list=data_list,
                wcs_list=tiles_wcs,
                shape_out=shape_out_hw,
                use_gpu=use_gpu,
                cpu_func=reproject_and_coadd,
                **reproj_kwargs,
            )
        except Exception as exc:
            if logger:
                logger.warning(
                    "[TwoPass] Reprojection failed on channel %d: %s",
                    ch,
                    exc,
                    exc_info=True,
                )
            return None
        chan_mosaic_np = np.asarray(chan_mosaic, dtype=np.float32)
        mosaic_channels.append(chan_mosaic_np)
        if coverage_result is None:
            coverage_result = np.asarray(chan_cov, dtype=np.float32)
        if logger:
            logger.debug(
                "[TwoPass] Channel %d done: mosaic_shape=%s, coverage_shape=%s, cov_stats=(min=%.4f, max=%.4f)",
                ch + 1,
                getattr(chan_mosaic_np, "shape", None),
                getattr(coverage_result, "shape", None),
                float(np.nanmin(coverage_result)) if coverage_result is not None else float("nan"),
                float(np.nanmax(coverage_result)) if coverage_result is not None else float("nan"),
            )
    mosaic = (
        mosaic_channels[0][..., np.newaxis]
        if n_channels == 1
        else np.stack(mosaic_channels, axis=-1)
    )
    if coverage_result is None:
        return None
    return mosaic.astype(np.float32), coverage_result.astype(np.float32)


# Backwards compatibility alias expected by tests
assemble_final_mosaic_with_reproject_coadd = assemble_final_mosaic_reproject_coadd


def prepare_tiles_and_calc_grid(
    tiles_with_wcs: list,
    crop_percent: float = 0.0,
    re_solve_cropped_tiles: bool = False,
    solver_settings: dict | None = None,
    solver_instance=None,
    drizzle_scale_factor: float = 1.0,
    progress_callback: Callable | None = None,
):
    wcs_list = []
    shape_list = []
    for path, w in tiles_with_wcs:
        current_wcs = w
        if re_solve_cropped_tiles and solver_instance is not None:
            try:
                solved = solver_instance.solve(path, w.to_header(), solver_settings or {}, update_header_with_solution=True)
                if solved:
                    current_wcs = solved
            except Exception:
                pass
        wcs_list.append(current_wcs)
        if hasattr(current_wcs, "pixel_shape"):
            shape_list.append((current_wcs.pixel_shape[1], current_wcs.pixel_shape[0]))
        else:
            shape_list.append((0, 0))
    return _calculate_final_mosaic_grid(wcs_list, shape_list, drizzle_scale_factor, progress_callback)




def run_hierarchical_mosaic(
    input_folder: str,
    output_folder: str,
    astap_exe_path: str,
    astap_data_dir_param: str,
    astap_search_radius_config: float,
    astap_downsample_config: int,
    astap_sensitivity_config: int,
    cluster_threshold_config: float,
    cluster_target_groups_config: int,
    cluster_orientation_split_deg_config: float,
    progress_callback: callable,
    stack_ram_budget_gb_config: float,
    stack_norm_method: str,
    stack_weight_method: str,
    stack_reject_algo: str,
    stack_kappa_low: float,
    stack_kappa_high: float,
    parsed_winsor_limits: tuple[float, float],
    stack_final_combine: str,
    poststack_equalize_rgb_config: bool,
    apply_radial_weight_config: bool,
    radial_feather_fraction_config: float,
    radial_shape_power_config: float,
    min_radial_weight_floor_config: float,
    final_assembly_method_config: str,
    inter_master_merge_enable_config: bool,
    inter_master_overlap_threshold_config: float,
    inter_master_min_group_size_config: int,
    inter_master_stack_method_config: str,
    inter_master_memmap_policy_config: str,
    inter_master_local_scale_config: str,
    inter_master_max_group_config: int,
    num_base_workers_config: int,
    # --- ARGUMENTS POUR LE ROGNAGE ---
    apply_master_tile_crop_config: bool,
    master_tile_crop_percent_config: float,
    quality_crop_enabled_config: bool,
    quality_crop_band_px_config: int,
    quality_crop_k_sigma_config: float,
    quality_crop_margin_px_config: int,
    save_final_as_uint16_config: bool,
    legacy_rgb_cube_config: bool,

    coadd_use_memmap_config: bool,
    coadd_memmap_dir_config: str,
    coadd_cleanup_memmap_config: bool,
    assembly_process_workers_config: int,
    auto_limit_frames_per_master_tile_config: bool,
    winsor_max_frames_per_pass_config: int,
    winsor_worker_limit_config: int,
    max_raw_per_master_tile_config: int,
    intertile_photometric_match_config: bool = True,
    intertile_preview_size_config: int = 512,
    intertile_overlap_min_config: float = 0.05,
    intertile_sky_percentile_config: tuple[float, float] | list[float] = (30.0, 70.0),
    intertile_robust_clip_sigma_config: float = 2.5,
    intertile_global_recenter_config: bool = True,
    intertile_recenter_clip_config: tuple[float, float] | list[float] = (0.85, 1.18),
    use_auto_intertile_config: bool = False,
    match_background_for_final_config: bool = True,
    incremental_feather_parity_config: bool = False,
    two_pass_coverage_renorm_config: bool = False,
    two_pass_cov_sigma_px_config: int = 50,
    two_pass_cov_gain_clip_config: tuple[float, float] | list[float] = (0.85, 1.18),
    center_out_normalization_p3_config: bool = True,
    p3_center_sky_percentile_config: tuple[float, float] | list[float] = (25.0, 60.0),
    p3_center_robust_clip_sigma_config: float = 2.5,
    p3_center_preview_size_config: int = 256,
    p3_center_min_overlap_fraction_config: float = 0.03,
    center_out_anchor_mode_config: str = "auto_central_quality",
    anchor_quality_probe_limit_config: int = 12,
    anchor_quality_span_range_config: tuple[float, float] | list[float] = (0.02, 6.0),
    anchor_quality_median_clip_sigma_config: float = 2.5,
    enable_poststack_anchor_review_config: bool = True,
    poststack_anchor_probe_limit_config: int = 8,
    poststack_anchor_span_range_config: tuple[float, float] | list[float] = (0.004, 10.0),
    poststack_anchor_median_clip_sigma_config: float = 3.5,
    poststack_anchor_min_improvement_config: float = 0.12,
    poststack_anchor_use_overlap_affine_config: bool = True,
    use_gpu_phase5: bool = False,
    gpu_id_phase5: int | None = None,
    logging_level_config: str = "INFO",
    solver_settings: dict | None = None,
    skip_filter_ui: bool = False,
    # New optional integration points when filter ran in GUI
    filter_invoked: bool = False,
    filter_overrides: dict | None = None,
    filtered_header_items: list[dict] | None = None,
    early_filter_enabled: bool | None = None,
):
    """
    Orchestre le traitement de la mosaïque hiérarchique.

    Parameters
    ----------
    winsor_max_frames_per_pass_config : int
        Limite du nombre d'images traitées simultanément par le rejet Winsorized (0 = illimité).
    winsor_worker_limit_config : int
        Nombre maximal de workers pour la phase de rejet Winsorized.
    stack_ram_budget_gb_config : float
        Budget RAM (en Gio) autorisé pour le chargement d'un groupe de stacking (0 = illimité).
    """
    worker_config_cache: dict[str, Any] = {}
    if ZEMOSAIC_CONFIG_AVAILABLE and zemosaic_config:
        try:
            worker_config_cache = zemosaic_config.load_config() or {}
        except Exception:
            worker_config_cache = {}

    pcb = lambda msg_key, prog=None, lvl="INFO", **kwargs: _log_and_callback(msg_key, prog, lvl, callback=progress_callback, **kwargs)

    # Cache retention policy (Phase 1 preprocessed cache cleanup)
    cache_retention_mode = "run_end"
    allowed_cache_modes = {"run_end", "per_tile", "keep"}
    if worker_config_cache:
        try:
            cache_retention_mode = str(worker_config_cache.get("cache_retention", "run_end")).strip().lower()
        except Exception:
            cache_retention_mode = "run_end"
    if cache_retention_mode not in allowed_cache_modes:
        cache_retention_mode = "run_end"
    logger.info("Cache retention mode: %s", cache_retention_mode)
    try:
        pcb("run_info_cache_retention_mode", prog=None, lvl="INFO_DETAIL", mode=cache_retention_mode)
    except Exception:
        pass

    # --- Apply logging level from GUI/config ---
    try:
        level_map = {
            "ERROR": logging.ERROR,
            "WARN": logging.WARNING,
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
        }
        lvl = level_map.get(str(logging_level_config).upper(), logging.INFO)
        logger.setLevel(lvl)
        for h in logger.handlers:
            try:
                h.setLevel(lvl)
            except Exception:
                pass
        logger.info("Worker logging level set to %s", str(logging.getLevelName(lvl)))
    except Exception:
        pass

    # --- Harmoniser les méthodes de pondération issues du GUI / CLI / fallback config ---
    requested_stack_weight_method = stack_weight_method
    stack_weight_method_normalized = str(stack_weight_method or "").lower().strip()
    if not stack_weight_method_normalized:
        stack_weight_method_normalized = ""
        if ZEMOSAIC_CONFIG_AVAILABLE and zemosaic_config:
            try:
                cfg_weight = zemosaic_config.load_config() or {}
                stack_weight_method_normalized = str(
                    cfg_weight.get("stacking_weighting_method", "")
                ).lower().strip()
            except Exception:
                stack_weight_method_normalized = ""
    if not stack_weight_method_normalized:
        stack_weight_method_normalized = "none"
    if stack_weight_method_normalized not in {"none", "noise_variance", "noise_fwhm"}:
        stack_weight_method_normalized = "none"
    if str(requested_stack_weight_method or "").lower().strip() != stack_weight_method_normalized:
        _log_and_callback(
            f"[Worker] stack_weight_method fallback -> '{stack_weight_method_normalized}'",
            lvl="INFO",
            callback=progress_callback,
        )
    stack_weight_method = stack_weight_method_normalized

    # Reset alignment warning counters at start of run
    for k in ALIGN_WARNING_COUNTS:
        ALIGN_WARNING_COUNTS[k] = 0
    
    def update_gui_eta(eta_seconds_total):
        if progress_callback and callable(progress_callback):
            eta_str = "--:--:--"
            if eta_seconds_total is not None and eta_seconds_total >= 0:
                h, rem = divmod(int(eta_seconds_total), 3600); m, s = divmod(rem, 60)
                eta_str = f"{h:02d}:{m:02d}:{s:02d}"
            pcb(f"ETA_UPDATE:{eta_str}", prog=None, lvl="ETA_LEVEL")


    resource_probe_info = _probe_system_resources(
        output_folder,
        two_pass_enabled=two_pass_coverage_renorm_config,
        two_pass_sigma_px=two_pass_cov_sigma_px_config,
        two_pass_gain_clip=two_pass_cov_gain_clip_config,
    )
    two_pass_enabled = bool(resource_probe_info.get("two_pass_enabled", False))
    try:
        two_pass_sigma_px = int(resource_probe_info.get("two_pass_sigma_px", 50) or 50)
    except (TypeError, ValueError):
        two_pass_sigma_px = 50
    gain_clip_raw = resource_probe_info.get("two_pass_gain_clip")
    gain_clip_tuple: tuple[float, float]
    if isinstance(gain_clip_raw, (list, tuple)) and len(gain_clip_raw) >= 2:
        try:
            low = float(gain_clip_raw[0])
            high = float(gain_clip_raw[1])
            if low > high:
                low, high = high, low
            gain_clip_tuple = (low, high)
        except (TypeError, ValueError):
            gain_clip_tuple = (0.85, 1.18)
    else:
        gain_clip_tuple = (0.85, 1.18)
    try:
        if (
            isinstance(intertile_recenter_clip_config, (list, tuple))
            and len(intertile_recenter_clip_config) >= 2
        ):
            clip_low = float(intertile_recenter_clip_config[0])
            clip_high = float(intertile_recenter_clip_config[1])
            if clip_low > clip_high:
                clip_low, clip_high = clip_high, clip_low
            intertile_recenter_clip_tuple = (clip_low, clip_high)
        else:
            intertile_recenter_clip_tuple = (0.85, 1.18)
    except Exception:
        intertile_recenter_clip_tuple = (0.85, 1.18)
    auto_caps_info: dict | None = None
    auto_resource_strategy: dict = {}
    phase0_header_items: list[dict] = []
    phase0_lookup: dict[str, dict] = {}
    preplan_groups_override_paths: list[list[str]] | None = None
    intertile_match_flag = bool(intertile_photometric_match_config)
    match_background_flag = (
        True
        if match_background_for_final_config is None
        else bool(match_background_for_final_config)
    )
    feather_parity_flag = bool(incremental_feather_parity_config)

    try:
        if isinstance(intertile_sky_percentile_config, (list, tuple)) and len(intertile_sky_percentile_config) >= 2:
            intertile_sky_percentile_tuple = (
                float(intertile_sky_percentile_config[0]),
                float(intertile_sky_percentile_config[1]),
            )
        else:
            intertile_sky_percentile_tuple = (30.0, 70.0)
    except Exception:
        intertile_sky_percentile_tuple = (30.0, 70.0)

    def _normalize_path_for_matching(path_value: str | None) -> str | None:
        if not path_value:
            return None
        try:
            return os.path.normcase(os.path.abspath(path_value))
        except Exception:
            try:
                return os.path.normcase(str(path_value))
            except Exception:
                return None


    # Seuil de clustering : valeur de repli à 0.05° si l'option est absente ou non positive
    try:
        cluster_threshold = float(cluster_threshold_config or 0)
    except (TypeError, ValueError):
        cluster_threshold = 0
    SEESTAR_STACK_CLUSTERING_THRESHOLD_DEG = (
        cluster_threshold if cluster_threshold > 0 else 0.05

    )
    # Orientation split threshold (degrees). 0 disables orientation filtering
    try:
        orientation_split_thr = float(cluster_orientation_split_deg_config or 0)
    except (TypeError, ValueError):
        orientation_split_thr = 0.0
    ORIENTATION_SPLIT_THRESHOLD_DEG = orientation_split_thr if orientation_split_thr > 0 else 0.0
    try:
        stack_ram_budget_gb = float(stack_ram_budget_gb_config or 0.0)
    except (TypeError, ValueError):
        stack_ram_budget_gb = 0.0
    STACK_RAM_BUDGET_BYTES = int(stack_ram_budget_gb * (1024 ** 3)) if stack_ram_budget_gb > 0 else 0
    PROGRESS_WEIGHT_PHASE1_RAW_SCAN = 30; PROGRESS_WEIGHT_PHASE2_CLUSTERING = 5
    PROGRESS_WEIGHT_PHASE3_MASTER_TILES = 35; PROGRESS_WEIGHT_PHASE4_GRID_CALC = 5
    PROGRESS_WEIGHT_PHASE4_5_INTER_MASTER = 6
    PROGRESS_WEIGHT_PHASE5_ASSEMBLY = 9; PROGRESS_WEIGHT_PHASE6_SAVE = 8
    PROGRESS_WEIGHT_PHASE7_CLEANUP = 2

    DEFAULT_PHASE_WORKER_RATIO = 1.0
    ALIGNMENT_PHASE_WORKER_RATIO = 0.5  # Limit aggressive phases to 50% of base workers

    if use_gpu_phase5 and gpu_id_phase5 is not None and CUPY_AVAILABLE:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id_phase5)
        try:
            import cupy
            cupy.cuda.Device(0).use()
        except Exception as e:
            pcb(
                "run_error_gpu_init_failed",
                prog=None,
                lvl="ERROR",
                error=str(e),
            )
            use_gpu_phase5 = False
    else:
        for v in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
            os.environ.pop(v, None)

    # Determine final GPU usage flag only if a valid NVIDIA GPU is selected
    use_gpu_phase5_flag = (
        use_gpu_phase5
        and gpu_id_phase5 is not None
        and CUPY_AVAILABLE
        and gpu_is_available()
    )
    if use_gpu_phase5_flag and ZEMOSAIC_UTILS_AVAILABLE and zemosaic_utils:
        try:
            # Initialize CuPy memory pools on the selected device (index 0 under the mask)
            if hasattr(zemosaic_utils, 'ensure_cupy_pool_initialized'):
                zemosaic_utils.ensure_cupy_pool_initialized(0)
        except Exception:
            pass
    def _cleanup_per_tile_cache(cache_paths: Iterable[str]) -> tuple[int, int]:
        """Remove preprocessed cache files for a completed master tile."""

        removed_count = 0
        removed_bytes = 0
        seen_paths: set[str] = set()

        for path in cache_paths or ():
            if not isinstance(path, str):
                continue
            try:
                norm_path = os.path.abspath(path)
            except Exception:
                norm_path = path
            if norm_path in seen_paths:
                continue
            seen_paths.add(norm_path)
            if not isinstance(norm_path, str) or not norm_path.lower().endswith(".npy"):
                continue
            if not os.path.isfile(norm_path):
                continue

            file_size = 0
            try:
                file_size = os.path.getsize(norm_path)
            except OSError:
                file_size = 0

            try:
                os.remove(norm_path)
                removed_count += 1
                removed_bytes += file_size
                logger.debug("Removed per-tile cache file: %s", norm_path)
            except FileNotFoundError:
                continue
            except OSError as exc_remove:
                logger.warning("Failed to remove per-tile cache file %s: %s", norm_path, exc_remove)

        return removed_count, removed_bytes

    def _compute_phase_workers(base_workers: int, num_tasks: int, ratio: float = DEFAULT_PHASE_WORKER_RATIO) -> int:
        workers = max(1, int(base_workers * ratio))
        if num_tasks > 0:
            workers = min(workers, num_tasks)
        return max(1, workers)
    current_global_progress = 0
    
    error_messages_deps = []
    if not (ASTROPY_AVAILABLE and WCS and SkyCoord and Angle and fits and u): error_messages_deps.append("Astropy")
    if not (REPROJECT_AVAILABLE and find_optimal_celestial_wcs and reproject_and_coadd and reproject_interp): error_messages_deps.append("Reproject")
    if not (ZEMOSAIC_UTILS_AVAILABLE and zemosaic_utils): error_messages_deps.append("zemosaic_utils")
    if not (ZEMOSAIC_ASTROMETRY_AVAILABLE and zemosaic_astrometry): error_messages_deps.append("zemosaic_astrometry")
    if not (ZEMOSAIC_ALIGN_STACK_AVAILABLE and zemosaic_align_stack): error_messages_deps.append("zemosaic_align_stack")
    try: import psutil
    except ImportError: error_messages_deps.append("psutil")
    if error_messages_deps:
        pcb("run_error_critical_deps_missing", prog=None, lvl="ERROR", modules=", ".join(error_messages_deps)); return

    start_time_total_run = time.monotonic()
    pcb("CHRONO_START_REQUEST", prog=None, lvl="CHRONO_LEVEL")
    _log_memory_usage(progress_callback, "Début Run Hierarchical Mosaic")
    pcb("run_info_processing_started", prog=current_global_progress, lvl="INFO")
    _log_and_callback(
        (
            f"Options Stacking (Master Tiles): Norm='{stack_norm_method}', "
            f"Weight='{stack_weight_method}', Reject='{stack_reject_algo}', "
            f"Combine='{stack_final_combine}'"
        ),
        lvl="INFO",
        callback=progress_callback,
    )
    pcb(f"  Config ASTAP: Exe='{os.path.basename(astap_exe_path) if astap_exe_path else 'N/A'}', Data='{os.path.basename(astap_data_dir_param) if astap_data_dir_param else 'N/A'}', Radius={astap_search_radius_config}deg, Downsample={astap_downsample_config}, Sens={astap_sensitivity_config}", prog=None, lvl="DEBUG_DETAIL")
    pcb(f"  Config Workers (GUI): Base demandé='{num_base_workers_config}' (0=auto)", prog=None, lvl="DEBUG_DETAIL")
    pcb(
        f"  Options Stacking (Master Tuiles): Norm='{stack_norm_method}', Weight='{stack_weight_method}', Reject='{stack_reject_algo}', "
        f"Combine='{stack_final_combine}', RGBEqualize={poststack_equalize_rgb_config}, RadialWeight={apply_radial_weight_config} "
        f"(Feather={radial_feather_fraction_config if apply_radial_weight_config else 'N/A'}, "
        f"Power={radial_shape_power_config if apply_radial_weight_config else 'N/A'}, "
        f"Floor={min_radial_weight_floor_config if apply_radial_weight_config else 'N/A'})",
        prog=None,
        lvl="DEBUG_DETAIL",
    )
    pcb(f"  Options Assemblage Final: Méthode='{final_assembly_method_config}'", prog=None, lvl="DEBUG_DETAIL")

    time_per_raw_file_wcs = None; time_per_master_tile_creation = None
    cache_dir_name = ".zemosaic_img_cache"; temp_image_cache_dir = os.path.join(output_folder, cache_dir_name)
    try:
        if os.path.exists(temp_image_cache_dir): shutil.rmtree(temp_image_cache_dir)
        os.makedirs(temp_image_cache_dir, exist_ok=True)
    except OSError as e_mkdir_cache:
        pcb("run_error_cache_dir_creation_failed", prog=None, lvl="ERROR", directory=temp_image_cache_dir, error=str(e_mkdir_cache)); return
    try:
        cache_probe = _probe_system_resources(
            temp_image_cache_dir,
            two_pass_enabled=two_pass_coverage_renorm_config,
            two_pass_sigma_px=two_pass_cov_sigma_px_config,
            two_pass_gain_clip=two_pass_cov_gain_clip_config,
        )
        for key, value in cache_probe.items():
            if value is not None:
                resource_probe_info[key] = value
    except Exception:
        pass

# --- Phase 1 (Prétraitement et WCS) ---
    base_progress_phase1 = current_global_progress
    _log_memory_usage(progress_callback, "Début Phase 1 (Prétraitement)")
    pcb("run_info_phase1_started_cache", prog=base_progress_phase1, lvl="INFO")
    pcb("PHASE_UPDATE:1", prog=None, lvl="ETA_LEVEL")
    
    fits_file_paths = []
    # Scan des fichiers FITS dans le dossier d'entrée et ses sous-dossiers
    for root_dir_iter, dirnames_iter, files_in_dir_iter in os.walk(input_folder):
        # Exclure les dossiers interdits dès la descente
        try:
            dirnames_iter[:] = [
                d
                for d in dirnames_iter
                if not is_path_excluded(Path(root_dir_iter) / d, EXCLUDED_DIRS)
            ]
        except Exception:
            dirnames_iter[:] = [
                d
                for d in dirnames_iter
                if UNALIGNED_DIRNAME
                not in os.path.normpath(os.path.join(root_dir_iter, d)).split(os.sep)
            ]

        # Assurer un ordre déterministe quelle que soit la plateforme/FS
        try:
            files_in_dir_iter = sorted(files_in_dir_iter, key=lambda s: s.lower())
        except Exception:
            files_in_dir_iter = list(files_in_dir_iter)

        for file_name_iter in files_in_dir_iter:
            if file_name_iter.lower().endswith((".fit", ".fits")):
                full_path = os.path.join(root_dir_iter, file_name_iter)
                try:
                    if is_path_excluded(full_path, EXCLUDED_DIRS):
                        continue
                except Exception:
                    if UNALIGNED_DIRNAME in os.path.normpath(full_path).split(os.sep):
                        continue
                fits_file_paths.append(full_path)
    # Tri global déterministe
    try:
        fits_file_paths.sort(key=lambda p: p.lower())
    except Exception:
        fits_file_paths.sort()
    
    if not fits_file_paths: 
        pcb("run_error_no_fits_found_input", prog=current_global_progress, lvl="ERROR")
        return # Sortie anticipée si aucun fichier FITS n'est trouvé

    num_total_raw_files = len(fits_file_paths)
    pcb("run_info_found_potential_fits", prog=base_progress_phase1, lvl="INFO_DETAIL", num_files=num_total_raw_files)
    # Kick off a stage progress stream so the GUI progress bar animates
    try:
        if progress_callback and callable(progress_callback):
            progress_callback("phase1_scan", 0, int(num_total_raw_files))
        # Also update a dedicated raw files counter in the GUI
        pcb(f"RAW_FILE_COUNT_UPDATE:0/{num_total_raw_files}", prog=None, lvl="ETA_LEVEL")
    except Exception:
        pass

    # --- Phase 0 (Header-only scan + early filter) ---
    # Preserve GUI-provided filter context arguments
    filter_invoked_arg = filter_invoked
    filter_overrides_arg = filter_overrides
    filtered_header_items_arg = filtered_header_items

    skip_filter_ui = bool(skip_filter_ui)
    # Resolve early filter enable policy: explicit argument takes precedence,
    # otherwise load from config, then apply skip_filter_ui override.
    if early_filter_enabled is None:
        early_filter_enabled = True
        try:
            if ZEMOSAIC_CONFIG_AVAILABLE and zemosaic_config:
                cfg0 = zemosaic_config.load_config() or {}
                early_filter_enabled = bool(cfg0.get("enable_early_filter", True))
        except Exception:
            early_filter_enabled = True
    else:
        early_filter_enabled = bool(early_filter_enabled)

    if skip_filter_ui:
        early_filter_enabled = False
        pcb("log_filter_ui_skipped", prog=None, lvl="INFO_DETAIL")

    if ASTROPY_AVAILABLE and fits is not None:
        header_items_for_filter: list[dict] = []
        filtered_items: list[dict] | None = None
        # If caller provided overrides or prior filter state, adopt them
        filter_overrides = filter_overrides_arg if isinstance(filter_overrides_arg, dict) else None
        filter_accepted = False
        filter_invoked = bool(filter_invoked_arg)
        streaming_filter_success = False

        launch_filter_interface_fn = None
        if early_filter_enabled:
            try:
                from zemosaic_filter_gui import launch_filter_interface as launch_filter_interface_fn  # type: ignore
            except ImportError:
                launch_filter_interface_fn = None
                pcb("Phase 0: filter GUI not available", prog=None, lvl="DEBUG_DETAIL")

        def _parse_filter_result(ret_obj):
            filt_items = None
            accepted_flag = False
            overrides_obj = None
            if isinstance(ret_obj, tuple) and len(ret_obj) >= 1:
                filt_items = ret_obj[0]
                if len(ret_obj) >= 2:
                    try:
                        accepted_flag = bool(ret_obj[1])
                    except Exception:
                        accepted_flag = False
                if len(ret_obj) >= 3:
                    overrides_obj = ret_obj[2]
            elif isinstance(ret_obj, list):
                filt_items = ret_obj
                accepted_flag = True
            return filt_items, accepted_flag, overrides_obj

        initial_filter_overrides = None
        try:
            initial_filter_overrides = {
                "cluster_panel_threshold": float(cluster_threshold_config),
                "cluster_target_groups": int(cluster_target_groups_config),
                "cluster_orientation_split_deg": float(cluster_orientation_split_deg_config),
            }
        except Exception:
            initial_filter_overrides = None

        # If the GUI already provided a filtered list, adopt it directly and
        # mark the streaming path as successful to avoid relaunching the UI.
        if isinstance(filtered_header_items_arg, list) and filtered_header_items_arg:
            try:
                header_items_for_filter = filtered_header_items_arg
            except Exception:
                header_items_for_filter = list(filtered_header_items_arg)
            filter_invoked = True
            filter_accepted = True
            streaming_filter_success = True

        solver_payload_for_filter = solver_settings if isinstance(solver_settings, dict) else None
        config_payload_for_filter = {
            "astap_executable_path": astap_exe_path,
            "astap_data_directory_path": astap_data_dir_param,
            "astap_default_search_radius": astap_search_radius_config,
            "astap_default_downsample": astap_downsample_config,
            "astap_default_sensitivity": astap_sensitivity_config,
        }

        if launch_filter_interface_fn is not None:
            try:
                filter_invoked = True
                filter_ret = launch_filter_interface_fn(
                    input_folder,
                    initial_filter_overrides,
                    stream_scan=True,
                    scan_recursive=True,
                    batch_size=200,
                    preview_cap=1500,
                    solver_settings_dict=solver_payload_for_filter,
                    config_overrides=config_payload_for_filter,
                )
                filtered_items, filter_accepted, filter_overrides = _parse_filter_result(filter_ret)
                # If the user cancelled from the filter UI, abort the run cleanly
                if isinstance(filter_overrides, dict) and filter_overrides.get("filter_cancelled"):
                    pcb("run_warn_phase0_filter_cancelled", prog=None, lvl="WARN")
                    pcb("log_key_processing_cancelled", prog=None, lvl="WARN")
                    return
                # In streaming mode the UI returns the final filtered list, not
                # the header pre-scan items. Consider the streaming path a success
                # whenever the UI was invoked without raising.
                streaming_filter_success = True
                if isinstance(filtered_items, list):
                    header_items_for_filter = filtered_items
                pcb(
                    "Phase 0: streaming filter UI completed",
                    prog=None,
                    lvl="INFO_DETAIL",
                )
            except Exception as e_filter:
                # If we fail to invoke the streaming UI, fall back to header scan.
                filter_invoked = False
                header_items_for_filter = []
                filtered_items = None
                filter_overrides = None
                filter_accepted = False
                pcb(f"Phase 0 streaming filter failed: {e_filter}", prog=None, lvl="WARN")

        if not streaming_filter_success:
            pcb("Phase 0: header scan start", prog=None, lvl="INFO_DETAIL")
            t0_hscan = time.monotonic()
            header_items_for_filter = []
            num_scanned = 0
            for idx_file, fpath in enumerate(fits_file_paths):
                hdr = None
                wcs0 = None
                shp_hw = None
                center_sc = None
                try:
                    hdr = fits.getheader(fpath, 0)
                    try:
                        nax1 = int(hdr.get("NAXIS1", 0))
                        nax2 = int(hdr.get("NAXIS2", 0))
                        if nax1 > 0 and nax2 > 0:
                            shp_hw = (nax2, nax1)
                    except Exception:
                        shp_hw = None
                    try:
                        w = WCS(hdr, naxis=2, relax=True) if WCS is not None else None
                        if w and getattr(w, "is_celestial", False):
                            wcs0 = w
                    except Exception:
                        wcs0 = None
                    if wcs0 is None:
                        try:
                            if (
                                ZEMOSAIC_ASTROMETRY_AVAILABLE
                                and zemosaic_astrometry
                                and hasattr(zemosaic_astrometry, "extract_center_from_header")
                            ):
                                center_sc = zemosaic_astrometry.extract_center_from_header(hdr)
                        except Exception:
                            center_sc = None
                    item = {
                        "path": fpath,
                        "header": hdr,
                        "index": idx_file,
                    }
                    if shp_hw:
                        item["shape"] = shp_hw
                    if wcs0 is not None:
                        item["wcs"] = wcs0
                    if center_sc is not None:
                        item["center"] = center_sc
                    header_items_for_filter.append(item)
                    num_scanned += 1
                except Exception:
                    header_items_for_filter.append({"path": fpath, "index": idx_file})
                    num_scanned += 1
            t1_hscan = time.monotonic()
            avg_t = (t1_hscan - t0_hscan) / max(1, num_scanned)
            pcb(
                f"Phase 0: header scan finished — files={num_scanned}, avg={avg_t:.4f}s/header",
                prog=None,
                lvl="DEBUG",
            )

            if launch_filter_interface_fn is not None and not filter_invoked:
                try:
                    filter_invoked = True
                    filter_ret = launch_filter_interface_fn(header_items_for_filter, initial_filter_overrides)
                    filtered_items, filter_accepted, filter_overrides = _parse_filter_result(filter_ret)
                    if isinstance(filter_overrides, dict) and filter_overrides.get("filter_cancelled"):
                        pcb("run_warn_phase0_filter_cancelled", prog=None, lvl="WARN")
                        pcb("log_key_processing_cancelled", prog=None, lvl="WARN")
                        return
                except Exception as e_filter:
                    filter_invoked = False
                    filtered_items = None
                    filter_overrides = None
                    filter_accepted = False
                    pcb(f"Phase 0 filter UI failed: {e_filter}", prog=None, lvl="WARN")
            elif not early_filter_enabled:
                pcb("Phase 0: header scan completed (filter UI disabled)", prog=None, lvl="DEBUG_DETAIL")

        phase0_header_items = header_items_for_filter

        if filter_invoked:
            if filter_overrides:
                try:
                    if "cluster_panel_threshold" in filter_overrides:
                        cluster_threshold_config = filter_overrides["cluster_panel_threshold"]
                        pcb(
                            "clusterstacks_info_override_threshold",
                            prog=None,
                            lvl="INFO_DETAIL",
                            value=cluster_threshold_config,
                        )
                    if "cluster_target_groups" in filter_overrides:
                        cluster_target_groups_config = filter_overrides["cluster_target_groups"]
                        pcb(
                            "clusterstacks_info_override_target_groups",
                            prog=None,
                            lvl="INFO_DETAIL",
                            value=cluster_target_groups_config,
                        )
                    if "cluster_orientation_split_deg" in filter_overrides:
                        cluster_orientation_split_deg_config = filter_overrides["cluster_orientation_split_deg"]
                        pcb(
                            "clusterstacks_info_override_orientation_split",
                            prog=None,
                            lvl="INFO_DETAIL",
                            value=cluster_orientation_split_deg_config,
                        )
                except Exception:
                    pass
                try:
                    raw_groups_override = (
                        filter_overrides.get("preplan_master_groups")
                        if isinstance(filter_overrides, dict)
                        else None
                    )
                    if isinstance(raw_groups_override, list):
                        mapped_groups: list[list[str]] = []
                        for group in raw_groups_override:
                            if not isinstance(group, (list, tuple)):
                                continue
                            normalized_group: list[str] = []
                            for item in group:
                                path_val = None
                                if isinstance(item, dict):
                                    path_val = item.get("path") or item.get("path_raw")
                                elif isinstance(item, str):
                                    path_val = item
                                norm_path = _normalize_path_for_matching(path_val)
                                if norm_path:
                                    normalized_group.append(norm_path)
                            if normalized_group:
                                mapped_groups.append(normalized_group)
                        if mapped_groups:
                            preplan_groups_override_paths = mapped_groups
                            pcb(
                                f"Phase 0 filter provided {len(mapped_groups)} preplanned group(s).",
                                prog=None,
                                lvl="INFO_DETAIL",
                            )
                except Exception as e_preplan:
                    pcb(
                        f"Phase 0 filter preplan override failed: {e_preplan}",
                        prog=None,
                        lvl="DEBUG_DETAIL",
                    )

            if not filter_accepted:
                pcb("run_warn_phase0_filter_cancelled", prog=None, lvl="WARN")
                pcb("Phase 0: filter cancelled -> proceeding with all files", prog=None, lvl="INFO_DETAIL")
            if filter_accepted and isinstance(filtered_items, list):
                new_paths = [
                    item.get("path")
                    for item in filtered_items
                    if isinstance(item, dict) and item.get("path")
                ]
                filtered_paths: list[str] = []
                for candidate_path in new_paths:
                    try:
                        if is_path_excluded(candidate_path, EXCLUDED_DIRS):
                            continue
                    except Exception:
                        if UNALIGNED_DIRNAME in os.path.normpath(str(candidate_path)).split(os.sep):
                            continue
                    filtered_paths.append(candidate_path)

                fits_file_paths = filtered_paths
                pcb(
                    f"Phase 0: selection after filter = {len(fits_file_paths)} files",
                    prog=None,
                    lvl="INFO_DETAIL",
                )
                if fits_file_paths:
                    try:
                        fits_file_paths.sort(key=lambda p: p.lower())
                    except Exception:
                        fits_file_paths.sort()
            elif filter_accepted and not filtered_items:
                pcb("Phase 0: filter returned no items", prog=None, lvl="WARN")
    else:
        phase0_header_items = []
        pcb("Phase 0: header scan unavailable (Astropy missing)", prog=None, lvl="WARN")

    phase0_lookup = {item["path"]: item for item in phase0_header_items if isinstance(item, dict) and item.get("path")}
    per_frame_info = _estimate_per_frame_cost_mb(phase0_header_items)
    auto_caps_info = _compute_auto_tile_caps(
        resource_probe_info,
        per_frame_info,
        policy_max=50,
        policy_min=8,
        user_max_override=int(max_raw_per_master_tile_config) if max_raw_per_master_tile_config else None,
    )
    try:
        msg = (
            "AutoCaps: per_frame≈{pf:.1f} MB, RAM_free≈{rf:.0f} MB → "
            "frames_by_ram={fbr}, cap={cap}, memmap={mm}, GPUHint={gpu}, parallel={par}".format(
                pf=auto_caps_info.get("per_frame_mb", 0.0),
                rf=resource_probe_info.get("ram_available_mb", 0.0) or 0.0,
                fbr=auto_caps_info.get("frames_by_ram", 0),
                cap=auto_caps_info.get("cap"),
                mm="on" if auto_caps_info.get("memmap") else "off",
                gpu=auto_caps_info.get("gpu_batch_hint") or "n/a",
                par=auto_caps_info.get("parallel_groups", 1),
            )
        )
        _log_and_callback(msg, prog=None, lvl="INFO_DETAIL", callback=progress_callback)
    except Exception:
        pass
    auto_resource_strategy = {
        "cap": auto_caps_info.get("cap"),
        "min_cap": auto_caps_info.get("min_cap"),
        "memmap": auto_caps_info.get("memmap"),
        "memmap_budget_mb": auto_caps_info.get("memmap_budget_mb"),
        "gpu_batch_hint": auto_caps_info.get("gpu_batch_hint"),
        "parallel_groups": auto_caps_info.get("parallel_groups"),
        "per_frame_mb": auto_caps_info.get("per_frame_mb"),
    }

    
    # --- Détermination du nombre de workers de BASE ---
    effective_base_workers = 0
    num_logical_processors = os.cpu_count() or 1 
    
    if num_base_workers_config <= 0: # Mode automatique (0 de la GUI)
        desired_auto_ratio = 0.75
        effective_base_workers = max(1, int(np.ceil(num_logical_processors * desired_auto_ratio)))
        pcb(f"WORKERS_CONFIG: Mode Auto. Base de workers calculée: {effective_base_workers} ({desired_auto_ratio*100:.0f}% de {num_logical_processors} processeurs logiques)", prog=None, lvl="INFO_DETAIL")
    else: # Mode manuel
        effective_base_workers = min(num_base_workers_config, num_logical_processors)
        if effective_base_workers < num_base_workers_config:
             pcb(f"WORKERS_CONFIG: Demande GUI ({num_base_workers_config}) limitée à {effective_base_workers} (total processeurs logiques: {num_logical_processors}).", prog=None, lvl="WARN")
        pcb(f"WORKERS_CONFIG: Mode Manuel. Base de workers: {effective_base_workers}", prog=None, lvl="INFO_DETAIL")
    
    if effective_base_workers <= 0: # Fallback
        effective_base_workers = 1
        pcb(f"WORKERS_CONFIG: AVERT - effective_base_workers était <= 0, forcé à 1.", prog=None, lvl="WARN")

    # Calcul du nombre de workers pour la Phase 1
    actual_num_workers_ph1 = _compute_phase_workers(
        effective_base_workers,
        num_total_raw_files,
        DEFAULT_PHASE_WORKER_RATIO,
    )
    pcb(
        f"WORKERS_PHASE1: Utilisation de {actual_num_workers_ph1} worker(s). (Base: {effective_base_workers}, Fichiers: {num_total_raw_files})",
        prog=None,
        lvl="INFO",
    )  # Log mis à jour pour plus de clarté
    
    start_time_phase1 = time.monotonic()
    all_raw_files_processed_info_dict = {} # Pour stocker les infos des fichiers traités avec succès
    files_processed_count_ph1 = 0      # Compteur pour les fichiers soumis au ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=actual_num_workers_ph1, thread_name_prefix="ZeMosaic_Ph1_") as executor_ph1:
        batch_size = 200
        for i in range(0, len(fits_file_paths), batch_size):
            batch = fits_file_paths[i:i+batch_size]
            future_to_filepath_ph1 = {
                executor_ph1.submit(
                    get_wcs_and_pretreat_raw_file,
                    f_path,
                    astap_exe_path,
                    astap_data_dir_param,
                    astap_search_radius_config,
                    astap_downsample_config,
                    astap_sensitivity_config,
                    180,
                    progress_callback,
                    temp_image_cache_dir,
                    solver_settings
                ): f_path for f_path in batch
            }

            for future in as_completed(future_to_filepath_ph1):
                file_path_original = future_to_filepath_ph1[future]
                files_processed_count_ph1 += 1  # Incrémenter pour chaque future terminée

                # Update GUI stage progress (files read / total)
                try:
                    if progress_callback and callable(progress_callback):
                        progress_callback("phase1_scan", int(files_processed_count_ph1), int(num_total_raw_files))
                    # Mirror the count so the GUI can show X/N files
                    pcb(f"RAW_FILE_COUNT_UPDATE:{files_processed_count_ph1}/{num_total_raw_files}", prog=None, lvl="ETA_LEVEL")
                except Exception:
                    pass

                prog_step_phase1 = base_progress_phase1 + int(
                    PROGRESS_WEIGHT_PHASE1_RAW_SCAN * (files_processed_count_ph1 / max(1, num_total_raw_files))
                )

                try:
                    # Récupérer le résultat de la tâche
                    img_data_adu, wcs_obj_solved, header_obj_updated, hp_mask_path = future.result()

                    # Si la tâche a réussi (ne retourne pas que des None)
                    if (
                        img_data_adu is not None
                        and wcs_obj_solved is not None
                        and header_obj_updated is not None
                    ):
                        # Sauvegarder les données prétraitées en .npy
                        cache_file_basename = f"preprocessed_{os.path.splitext(os.path.basename(file_path_original))[0]}_{files_processed_count_ph1}.npy"
                        cached_image_path = os.path.join(temp_image_cache_dir, cache_file_basename)
                        try:
                            np.save(cached_image_path, img_data_adu)
                            # Stocker les informations pour les phases suivantes
                            entry = {
                                'path_raw': file_path_original,
                                'path_preprocessed_cache': cached_image_path,
                                'path_hotpix_mask': hp_mask_path,
                                'wcs': wcs_obj_solved,
                                'header': header_obj_updated,
                                'preprocessed_shape': tuple(int(dim) for dim in getattr(img_data_adu, 'shape', []) or ()),
                            }
                            meta = phase0_lookup.get(file_path_original)
                            if isinstance(meta, dict):
                                if 'index' in meta:
                                    entry['phase0_index'] = meta.get('index')
                                if 'center' in meta:
                                    entry['phase0_center'] = meta.get('center')
                                if 'shape' in meta:
                                    entry['phase0_shape'] = meta.get('shape')
                                if 'wcs' in meta and 'wcs' not in entry:
                                    entry['phase0_wcs'] = meta.get('wcs')
                            all_raw_files_processed_info_dict[file_path_original] = entry
                        except Exception as e_save_npy:
                            pcb(
                                "run_error_phase1_save_npy_failed",
                                prog=prog_step_phase1,
                                lvl="ERROR",
                                filename=os.path.basename(file_path_original),
                                error=str(e_save_npy),
                            )
                            logger.error(f"Erreur sauvegarde NPY pour {file_path_original}:", exc_info=True)
                        finally:
                            # Libérer la mémoire des données image dès que possible
                            del img_data_adu
                            gc.collect()
                    else:
                        # Le fichier a échoué (ex: WCS non résolu et déplacé)
                        # get_wcs_and_pretreat_raw_file a déjà loggué l'échec spécifique.
                        pcb(
                            "run_warn_phase1_wcs_pretreat_failed_or_skipped_thread",
                            prog=prog_step_phase1,
                            lvl="WARN",
                            filename=os.path.basename(file_path_original),
                        )
                        if img_data_adu is not None:
                            del img_data_adu
                            gc.collect()

                except Exception as exc_thread:
                    # Erreur imprévue dans la future elle-même
                    pcb(
                        "run_error_phase1_thread_exception",
                        prog=prog_step_phase1,
                        lvl="ERROR",
                        filename=os.path.basename(file_path_original),
                        error=str(exc_thread),
                    )
                    logger.error(
                        f"Exception non gérée dans le thread Phase 1 pour {file_path_original}:",
                        exc_info=True,
                    )

                # Log de mémoire et ETA
                if (
                    files_processed_count_ph1 % max(1, num_total_raw_files // 10) == 0
                    or files_processed_count_ph1 == num_total_raw_files
                ):
                    _log_memory_usage(
                        progress_callback,
                        f"Phase 1 - Traité {files_processed_count_ph1}/{num_total_raw_files}",
                    )

                elapsed_phase1 = time.monotonic() - start_time_phase1
                if files_processed_count_ph1 > 0:
                    time_per_raw_file_wcs = elapsed_phase1 / files_processed_count_ph1
                    eta_phase1_sec = (num_total_raw_files - files_processed_count_ph1) * time_per_raw_file_wcs
                    current_progress_in_run_percent = base_progress_phase1 + (
                        files_processed_count_ph1 / max(1, num_total_raw_files)
                    ) * PROGRESS_WEIGHT_PHASE1_RAW_SCAN
                    time_per_percent_point_global = (
                        (time.monotonic() - start_time_total_run) / max(1, current_progress_in_run_percent)
                        if current_progress_in_run_percent > 0
                        else (time.monotonic() - start_time_total_run)
                    )
                    total_eta_sec = eta_phase1_sec + (
                        100 - current_progress_in_run_percent
                    ) * time_per_percent_point_global
                    update_gui_eta(total_eta_sec)

    # Construire la liste finale des informations des fichiers traités avec succès
    all_raw_files_processed_info = [
        all_raw_files_processed_info_dict[fp] 
        for fp in fits_file_paths 
        if fp in all_raw_files_processed_info_dict
    ]
    
    if not all_raw_files_processed_info: 
        pcb("run_error_phase1_no_valid_raws_after_cache", prog=(base_progress_phase1 + PROGRESS_WEIGHT_PHASE1_RAW_SCAN), lvl="ERROR")
        return # Sortie anticipée si aucun fichier n'a pu être traité avec succès

    current_global_progress = base_progress_phase1 + PROGRESS_WEIGHT_PHASE1_RAW_SCAN
    _log_memory_usage(progress_callback, "Fin Phase 1 (Prétraitement)")
    pcb("run_info_phase1_finished_cache", prog=current_global_progress, lvl="INFO", num_valid_raws=len(all_raw_files_processed_info))
    # --- Optional interactive filtering between Phase 1 and Phase 2 ---
    try:
        raw_files_with_wcs = all_raw_files_processed_info
        try:
            raw_files_with_wcs = raw_files_with_wcs
            # Keep the same variable name used by subsequent phases
            all_raw_files_processed_info = raw_files_with_wcs
        except ImportError:
            # Optional module not present: silently skip
            pass
        except Exception as e_opt:
            logger.warning(f"Filtrage facultatif désactivé suite à une erreur : {e_opt}")
    except Exception as e_hook:
        # Any unexpected issue in the hook wrapper: continue unchanged
        logger.warning(f"Filtrage facultatif non appliqué: {e_hook}")
    if time_per_raw_file_wcs: 
        pcb(f"    Temps moyen/brute (P1): {time_per_raw_file_wcs:.2f}s", prog=None, lvl="DEBUG")

    # --- Phase 2 (Clustering) ---
    base_progress_phase2 = current_global_progress
    _log_memory_usage(progress_callback, "Début Phase 2 (Clustering)")
    pcb("run_info_phase2_started", prog=base_progress_phase2, lvl="INFO")
    pcb("PHASE_UPDATE:2", prog=None, lvl="ETA_LEVEL")
    # Use order-invariant connected-components clustering for robustness
    preplan_groups_active = False
    if preplan_groups_override_paths:
        try:
            path_lookup = {
                _normalize_path_for_matching(info.get("path_raw") or info.get("path")): info
                for info in all_raw_files_processed_info
                if isinstance(info, dict)
            }
            used_paths: set[str] = set()
            mapped_info_groups: list[list[dict]] = []
            missing_preplan: list[str] = []
            for group_paths in preplan_groups_override_paths:
                current_group: list[dict] = []
                for path_norm in group_paths:
                    if not path_norm:
                        continue
                    info = path_lookup.get(path_norm)
                    if info is not None:
                        current_group.append(info)
                        used_paths.add(path_norm)
                    else:
                        missing_preplan.append(path_norm)
                if current_group:
                    mapped_info_groups.append(current_group)
            if mapped_info_groups:
                leftovers = [
                    info
                    for info in all_raw_files_processed_info
                    if _normalize_path_for_matching(info.get("path_raw") or info.get("path")) not in used_paths
                ]
                if leftovers:
                    mapped_info_groups.append(leftovers)
                seestar_stack_groups = mapped_info_groups
                preplan_groups_active = True
                _log_and_callback(
                    f"Phase 2: using {len(mapped_info_groups)} preplanned group(s) from filter UI.",
                    prog=None,
                    lvl="INFO_DETAIL",
                    callback=progress_callback,
                )
                if missing_preplan:
                    try:
                        preview = ", ".join(os.path.basename(p) for p in missing_preplan[:5] if p)
                    except Exception:
                        preview = ""
                    _log_and_callback(
                        "Phase 2: some preplanned paths were not found after preprocessing: "
                        + (preview if preview else str(len(missing_preplan))),
                        prog=None,
                        lvl="WARN",
                        callback=progress_callback,
                    )
        except Exception as e_preplan_map:
            _log_and_callback(
                f"Phase 2: failed to map preplanned groups ({e_preplan_map}). Falling back to clustering.",
                prog=None,
                lvl="WARN",
                callback=progress_callback,
            )
            preplan_groups_active = False

    if not preplan_groups_active:
        seestar_stack_groups = cluster_seestar_stacks_connected(
            all_raw_files_processed_info,
            SEESTAR_STACK_CLUSTERING_THRESHOLD_DEG,
            progress_callback,
            orientation_split_threshold_deg=ORIENTATION_SPLIT_THRESHOLD_DEG,
        )
        if STACK_RAM_BUDGET_BYTES > 0 and seestar_stack_groups:
            seestar_stack_groups, ram_budget_adjustments = _apply_ram_budget_to_groups(
                seestar_stack_groups,
                STACK_RAM_BUDGET_BYTES,
                float(SEESTAR_STACK_CLUSTERING_THRESHOLD_DEG),
                float(ORIENTATION_SPLIT_THRESHOLD_DEG),
            )
            for adj in ram_budget_adjustments:
                method = adj.get("method")
                if method == "recluster":
                    _log_and_callback(
                        "clusterstacks_warn_ram_budget_recluster",
                        prog=None,
                        lvl="WARN",
                        callback=progress_callback,
                        group_index=adj.get("group_index"),
                        original_frames=adj.get("original_frames"),
                        num_subgroups=adj.get("num_subgroups"),
                        new_threshold_deg=adj.get("new_threshold_deg"),
                        attempts=adj.get("attempts"),
                        estimated_mb=adj.get("estimated_mb"),
                        budget_mb=adj.get("budget_mb"),
                    )
                elif method == "split":
                    _log_and_callback(
                        "clusterstacks_warn_ram_budget_split",
                        prog=None,
                        lvl="WARN",
                        callback=progress_callback,
                        group_index=adj.get("group_index"),
                        original_frames=adj.get("original_frames"),
                        num_subgroups=adj.get("num_subgroups"),
                        segment_size=adj.get("segment_size"),
                        estimated_mb=adj.get("estimated_mb"),
                        budget_mb=adj.get("budget_mb"),
                    )
                    if adj.get("still_over_budget"):
                        _log_and_callback(
                            "clusterstacks_warn_ram_budget_split_still_over",
                            prog=None,
                            lvl="WARN",
                            callback=progress_callback,
                            group_index=adj.get("group_index"),
                            segment_size=adj.get("segment_size"),
                            budget_mb=adj.get("budget_mb"),
                        )
                elif method == "single_over_budget":
                    _log_and_callback(
                        "clusterstacks_warn_ram_budget_single_over",
                        prog=None,
                        lvl="WARN",
                        callback=progress_callback,
                        group_index=adj.get("group_index"),
                        estimated_mb=adj.get("estimated_mb"),
                        budget_mb=adj.get("budget_mb"),
                    )
    # Diagnostic: nearest-neighbor separation percentiles to help tune eps
    try:
        panel_centers_sky_dbg = []
        for info in all_raw_files_processed_info:
            wcs_obj = info.get("wcs")
            if not (wcs_obj and getattr(wcs_obj, "is_celestial", False)):
                continue
            try:
                if getattr(wcs_obj, "pixel_shape", None):
                    cx = wcs_obj.pixel_shape[0] / 2.0
                    cy = wcs_obj.pixel_shape[1] / 2.0
                    center_world = wcs_obj.pixel_to_world(cx, cy)
                elif hasattr(wcs_obj, "wcs") and hasattr(wcs_obj.wcs, "crval"):
                    center_world = SkyCoord(
                        ra=float(wcs_obj.wcs.crval[0]) * u.deg,
                        dec=float(wcs_obj.wcs.crval[1]) * u.deg,
                        frame="icrs",
                    )
                else:
                    continue
                panel_centers_sky_dbg.append(center_world)
            except Exception:
                continue
        if len(panel_centers_sky_dbg) >= 2:
            coords_dbg = SkyCoord(ra=[c.ra for c in panel_centers_sky_dbg], dec=[c.dec for c in panel_centers_sky_dbg], frame="icrs")
            try:
                _, sep_nn, _ = coords_dbg.match_to_catalog_sky(coords_dbg, nthneighbor=1)
                nn = np.asarray(sep_nn.deg, dtype=float)
                p10 = float(np.nanpercentile(nn, 10.0))
                p50 = float(np.nanpercentile(nn, 50.0))
                p90 = float(np.nanpercentile(nn, 90.0))
                _log_and_callback(
                    f"Cluster NN stats (deg): P10={p10:.4f}, P50={p50:.4f}, P90={p90:.4f}",
                    prog=None,
                    lvl="DEBUG_DETAIL",
                    callback=progress_callback,
                )
            except Exception:
                pass
    except Exception:
        pass
    # If clustering is pathologically conservative (almost one group per image),
    # auto-relax the threshold based on nearest-neighbor distances to avoid
    # producing hundreds of master tiles for tightly-dithered panels.
    try:
        total_inputs_for_cluster = len(all_raw_files_processed_info)
        groups_initial = len(seestar_stack_groups)
        if total_inputs_for_cluster > 2 and groups_initial >= max(3, int(0.9 * total_inputs_for_cluster)):
            # Compute a robust suggested threshold from the 90th percentile of
            # nearest-neighbor separations between panel centers.
            # Rebuild centers the same way as clustering helpers do.
            panel_centers_sky = []
            for info in all_raw_files_processed_info:
                wcs_obj = info.get("wcs")
                if not (wcs_obj and getattr(wcs_obj, "is_celestial", False)):
                    continue
                try:
                    if getattr(wcs_obj, "pixel_shape", None):
                        cx = wcs_obj.pixel_shape[0] / 2.0
                        cy = wcs_obj.pixel_shape[1] / 2.0
                        center_world = wcs_obj.pixel_to_world(cx, cy)
                    elif hasattr(wcs_obj, "wcs") and hasattr(wcs_obj.wcs, "crval"):
                        center_world = SkyCoord(
                            ra=float(wcs_obj.wcs.crval[0]) * u.deg,
                            dec=float(wcs_obj.wcs.crval[1]) * u.deg,
                            frame="icrs",
                        )
                    else:
                        continue
                    panel_centers_sky.append(center_world)
                except Exception:
                    continue

            if len(panel_centers_sky) >= 2:
                coords = SkyCoord(
                    ra=[c.ra for c in panel_centers_sky],
                    dec=[c.dec for c in panel_centers_sky],
                    frame="icrs",
                )
                try:
                    # Nearest neighbor (excluding self). Astropy handles wrap.
                    _, sep2d, _ = coords.match_to_catalog_sky(coords, nthneighbor=1)
                    nn_deg = np.asarray(sep2d.deg, dtype=float)
                    # Robust high-quantile of dithers; add a small headroom.
                    p90 = float(np.nanpercentile(nn_deg, 90.0)) if nn_deg.size else 0.0
                    # Propose a relaxed threshold within sane bounds.
                    thr_initial = float(SEESTAR_STACK_CLUSTERING_THRESHOLD_DEG)
                    thr_candidate = max(thr_initial, p90 * 1.2)
                    thr_candidate = float(min(max(thr_candidate, 0.01), 1.0))  # clamp 0.01°..1.0°

                    if thr_candidate > thr_initial:
                        _log_and_callback(
                            f"Cluster AUTO: threshold {thr_initial:.3f}° too conservative -> {groups_initial}/{total_inputs_for_cluster} groups.",
                            prog=None,
                            lvl="INFO_DETAIL",
                            callback=progress_callback,
                        )
                        _log_and_callback(
                            f"Cluster AUTO: relaxing to {thr_candidate:.3f}° (≈1.2×P90 NN={p90:.3f}°) and re-clustering...",
                            prog=None,
                            lvl="INFO_DETAIL",
                            callback=progress_callback,
                        )
                        seestar_stack_groups = cluster_seestar_stacks_connected(
                            all_raw_files_processed_info, thr_candidate, progress_callback
                        )
                        groups_after = len(seestar_stack_groups)
                        _log_and_callback(
                            f"Cluster AUTO: re-clustered into {groups_after} groups (was {groups_initial}).",
                            prog=None,
                            lvl="INFO_DETAIL",
                            callback=progress_callback,
                        )
                except Exception as e_auto_relax:
                    _log_and_callback(
                        f"Cluster AUTO: failed to compute NN-based relax: {e_auto_relax}",
                        prog=None,
                        lvl="DEBUG_DETAIL",
                        callback=progress_callback,
                    )
    except Exception as e_cluster_guard:
        _log_and_callback(
            f"Cluster AUTO: guard exception: {e_cluster_guard}", prog=None, lvl="DEBUG_DETAIL", callback=progress_callback
        )

    # Optional: drive clustering to a target number of groups by relaxing
    # the threshold via a bounded search. Disabled when target <= 0.
    try:
        target_groups = int(cluster_target_groups_config or 0)
    except Exception:
        target_groups = 0
    if (not preplan_groups_active) and target_groups > 0 and len(seestar_stack_groups) != target_groups:
        try:
            # Build coordinates array
            panel_centers_sky = []
            for info in all_raw_files_processed_info:
                wcs_obj = info.get("wcs")
                if not (wcs_obj and getattr(wcs_obj, "is_celestial", False)):
                    continue
                try:
                    if getattr(wcs_obj, "pixel_shape", None):
                        cx = wcs_obj.pixel_shape[0] / 2.0
                        cy = wcs_obj.pixel_shape[1] / 2.0
                        center_world = wcs_obj.pixel_to_world(cx, cy)
                    elif hasattr(wcs_obj, "wcs") and hasattr(wcs_obj.wcs, "crval"):
                        center_world = SkyCoord(
                            ra=float(wcs_obj.wcs.crval[0]) * u.deg,
                            dec=float(wcs_obj.wcs.crval[1]) * u.deg,
                            frame="icrs",
                        )
                    else:
                        continue
                    panel_centers_sky.append(center_world)
                except Exception:
                    continue

            if len(panel_centers_sky) >= 2:
                coords = SkyCoord(
                    ra=[c.ra for c in panel_centers_sky],
                    dec=[c.dec for c in panel_centers_sky],
                    frame="icrs",
                )
                # Establish an upper bound big enough that all panels connect
                # (max pairwise separation). Clamp to 5 degrees to avoid
                # pathological values.
                try:
                    sep_mat_deg = coords.separation(coords).deg
                    max_pair_deg = float(np.nanmax(sep_mat_deg)) if np.size(sep_mat_deg) else 0.5
                except Exception:
                    max_pair_deg = 0.5
                thr_current = float(SEESTAR_STACK_CLUSTERING_THRESHOLD_DEG)
                def _count_groups(thr: float) -> tuple[int, list]:
                    g = cluster_seestar_stacks_connected(
                        all_raw_files_processed_info,
                        float(thr),
                        None,
                        orientation_split_threshold_deg=ORIENTATION_SPLIT_THRESHOLD_DEG,
                    )
                    return len(g), g
                cnt_cur = len(seestar_stack_groups)
                # Direction: if too many groups, increase threshold; if too few, decrease.
                if cnt_cur > target_groups:
                    lo = thr_current
                    hi = float(min(max(max_pair_deg, lo * 2.0, 0.05), 5.0))
                    cnt_hi, groups_hi = _count_groups(hi)
                    # Expand hi until we get <= target (fewer groups) or cap
                    expand_iter = 0
                    while cnt_hi > target_groups and hi < 5.0 and expand_iter < 8:
                        hi = min(hi * 1.5 + 1e-6, 5.0)
                        cnt_hi, groups_hi = _count_groups(hi)
                        expand_iter += 1
                    best_thr = hi
                    best_groups = groups_hi
                    for _ in range(14):
                        mid = 0.5 * (lo + hi)
                        cnt_mid, groups_mid = _count_groups(mid)
                        if cnt_mid > target_groups:
                            lo = mid
                        else:
                            hi = mid
                            best_thr = mid
                            best_groups = groups_mid
                else:
                    # Need more groups ⇒ lower the threshold
                    hi = thr_current
                    lo = max(1e-6, hi / 2.0)
                    cnt_lo, groups_lo = _count_groups(lo)
                    shrink_iter = 0
                    while cnt_lo < target_groups and lo > 1e-6 and shrink_iter < 12:
                        hi = lo
                        lo = max(1e-6, lo / 1.5)
                        cnt_lo, groups_lo = _count_groups(lo)
                        shrink_iter += 1
                    best_thr = lo
                    best_groups = groups_lo
                    # Binary search upward to approach target from the high side (more stable)
                    for _ in range(14):
                        mid = 0.5 * (lo + hi)
                        cnt_mid, groups_mid = _count_groups(mid)
                        if cnt_mid < target_groups:
                            # still too few groups ⇒ lower threshold more
                            hi = mid
                        else:
                            lo = mid
                            best_thr = mid
                            best_groups = groups_mid
                _log_and_callback(
                    f"Cluster AUTO Target: threshold -> {best_thr:.4f}° for ≈{len(best_groups)} groups (target {target_groups}).",
                    prog=None,
                    lvl="INFO_DETAIL",
                    callback=progress_callback,
                )
                seestar_stack_groups = best_groups
        except Exception as e_target:
            _log_and_callback(
                f"Cluster AUTO Target: search failed: {e_target}", prog=None, lvl="DEBUG_DETAIL", callback=progress_callback
            )
    if not seestar_stack_groups:
        pcb("run_error_phase2_no_groups", prog=(base_progress_phase2 + PROGRESS_WEIGHT_PHASE2_CLUSTERING), lvl="ERROR")
        return
    if (not preplan_groups_active) and auto_caps_info and seestar_stack_groups:
        try:
            cap_value = int(auto_caps_info.get("cap", 0))
            min_value = int(auto_caps_info.get("min_cap", 8))
        except Exception:
            cap_value = 0
            min_value = 8
        if cap_value > 0:
            original_count = len(seestar_stack_groups)
            seestar_stack_groups = _auto_split_groups(
                seestar_stack_groups,
                cap_value,
                min_value,
                progress_callback=progress_callback,
            )
            if len(seestar_stack_groups) != original_count:
                try:
                    _log_and_callback(
                        f"AutoSplit summary: {original_count} -> {len(seestar_stack_groups)} subgroup(s) (cap={cap_value})",
                        prog=None,
                        lvl="INFO_DETAIL",
                        callback=progress_callback,
                    )
                except Exception:
                    pass
            if min_value > 0:
                seestar_stack_groups = _merge_small_groups(
                    seestar_stack_groups,
                    min_size=min_value,
                    cap=cap_value,
                )

    # Do not subdivide groups if a target group count is set; respect clustering first.
    if (
        not preplan_groups_active
        and (cluster_target_groups_config is None or int(cluster_target_groups_config) <= 0)
        and max_raw_per_master_tile_config
        and max_raw_per_master_tile_config > 0
    ):
        new_groups = []
        for g in seestar_stack_groups:
            for i in range(0, len(g), max_raw_per_master_tile_config):
                new_groups.append(g[i:i + max_raw_per_master_tile_config])
        if len(new_groups) != len(seestar_stack_groups):
            pcb(
                "clusterstacks_info_groups_split_manual_limit",
                prog=None,
                lvl="INFO_DETAIL",
                original=len(seestar_stack_groups),
                new=len(new_groups),
                limit=max_raw_per_master_tile_config,
            )
        seestar_stack_groups = new_groups
    cpu_total = os.cpu_count() or 1
    winsor_worker_limit = max(1, min(int(winsor_worker_limit_config), cpu_total))
    winsor_max_frames_per_pass = max(0, int(winsor_max_frames_per_pass_config))
    pcb(
        f"Winsor worker limit set to {winsor_worker_limit}" + (
            " (ProcessPoolExecutor enabled)" if winsor_worker_limit > 1 else ""
        ),
        prog=None,
        lvl="INFO",
    )
    if winsor_max_frames_per_pass > 0:
        pcb(
            f"Winsor streaming limit set to {winsor_max_frames_per_pass} frame(s) per pass",
            prog=None,
            lvl="INFO_DETAIL",
        )
    manual_limit = max_raw_per_master_tile_config
    if (
        not preplan_groups_active
        and (cluster_target_groups_config is None or int(cluster_target_groups_config) <= 0)
        and auto_limit_frames_per_master_tile_config
    ):
        try:
            sample_path = seestar_stack_groups[0][0].get('path_preprocessed_cache')
            sample_arr = np.load(sample_path, mmap_mode='r')
            bytes_per_frame = sample_arr.nbytes
            sample_shape = sample_arr.shape
            sample_arr = None
            available_bytes = psutil.virtual_memory().available
            expected_workers = max(1, int(effective_base_workers * ALIGNMENT_PHASE_WORKER_RATIO))
            # Be more conservative: align/stack create extra buffers; use a larger safety factor
            limit = max(
                1,
                int(
                    available_bytes // (expected_workers * bytes_per_frame * 12)
                ),
            )
            # Clamp to a reasonable upper bound if no manual cap is set
            if manual_limit <= 0:
                limit = min(limit, 100)
            if manual_limit > 0:
                limit = min(limit, manual_limit)
            winsor_worker_limit = min(winsor_worker_limit, limit)
            new_groups = []
            for g in seestar_stack_groups:
                for i in range(0, len(g), limit):
                    new_groups.append(g[i:i+limit])
            if len(new_groups) != len(seestar_stack_groups):
                pcb(
                    "clusterstacks_info_groups_split_auto_limit",
                    prog=None,
                    lvl="INFO_DETAIL",
                    original=len(seestar_stack_groups),
                    new=len(new_groups),
                    limit=limit,
                    shape=str(sample_shape),
                )
            seestar_stack_groups = new_groups
            if manual_limit > 0 and limit != manual_limit:
                logger.info(
                    "Manual frame limit (%d) is lower than auto limit, using manual value.",
                    manual_limit,
                )
        except Exception as e_auto:
            pcb("clusterstacks_warn_auto_limit_failed", prog=None, lvl="WARN", error=str(e_auto))
    current_global_progress = base_progress_phase2 + PROGRESS_WEIGHT_PHASE2_CLUSTERING
    num_seestar_stacks_to_process = len(seestar_stack_groups)
    _log_memory_usage(progress_callback, "Fin Phase 2"); pcb("run_info_phase2_finished", prog=current_global_progress, lvl="INFO", num_groups=num_seestar_stacks_to_process)


    # --- IO-aware adaptation (bench read speed on cache + write speed on output) ---
    io_read_mbps, io_write_mbps = None, None
    io_read_cat, io_write_cat = "unknown", "unknown"
    try:
        sample_cache_for_read = None
        # Try to pick a representative cached image path from the first group
        if seestar_stack_groups and seestar_stack_groups[0]:
            sample_cache_for_read = seestar_stack_groups[0][0].get('path_preprocessed_cache')
        if sample_cache_for_read and os.path.exists(sample_cache_for_read):
            io_read_mbps = _measure_sequential_read_mbps(sample_cache_for_read)
            io_read_cat = _categorize_io_speed(io_read_mbps)
        # Write speed on output folder
        if output_folder and os.path.isdir(output_folder):
            io_write_mbps = _measure_sequential_write_mbps(output_folder)
            io_write_cat = _categorize_io_speed(io_write_mbps)
        pcb(
            f"IO_BENCH: read {io_read_mbps:.1f} MB/s ({io_read_cat}), write {io_write_mbps:.1f} MB/s ({io_write_cat})"
            if (io_read_mbps is not None and io_write_mbps is not None)
            else f"IO_BENCH: read={io_read_mbps}, write={io_write_mbps}"
            ,
            prog=None,
            lvl="DEBUG",
        )
    except Exception as e_io_bench:
        pcb(f"IO_BENCH: failed ({e_io_bench})", prog=None, lvl="WARN")

    # Derive conservative caps from read speed (dominant in Phase 3) on Windows/slow disks
    io_ph3_cap = None
    io_cache_read_slots = None
    new_winsor_limit = winsor_worker_limit
    if os.name == 'nt':
        if io_read_cat == "very_slow":
            io_ph3_cap = 1
            io_cache_read_slots = 1
            new_winsor_limit = min(new_winsor_limit, 1)
        elif io_read_cat == "slow":
            io_ph3_cap = 2
            io_cache_read_slots = 1
            new_winsor_limit = min(new_winsor_limit, 1)
        elif io_read_cat == "medium":
            io_ph3_cap = 3
            io_cache_read_slots = 2
            new_winsor_limit = min(new_winsor_limit, 2)
        elif io_read_cat == "fast":
            io_ph3_cap = 4
            io_cache_read_slots = 2
            # Keep winsor limit as computed
        # Apply winsor limit adjustment if changed
        if new_winsor_limit != winsor_worker_limit:
            pcb(
                f"IO_ADAPT: winsor_worker_limit reduced {winsor_worker_limit} -> {new_winsor_limit} due to IO ({io_read_cat})",
                prog=None,
                lvl="INFO_DETAIL",
            )
            winsor_worker_limit = new_winsor_limit
        # Adjust cache IO semaphore (controls concurrent npy reads)
        try:
            if io_cache_read_slots and io_cache_read_slots > 0:
                global _CACHE_IO_SEMAPHORE
                _CACHE_IO_SEMAPHORE = threading.Semaphore(int(io_cache_read_slots))
                pcb(
                    f"IO_ADAPT: cache read slots set to {io_cache_read_slots}",
                    prog=None,
                    lvl="INFO_DETAIL",
                )
        except Exception:
            pass


    try:
        setattr(zconfig, "winsor_worker_limit", int(winsor_worker_limit))
    except Exception:
        pass
    try:
        setattr(zconfig, "winsor_max_frames_per_pass", int(winsor_max_frames_per_pass))
    except Exception:
        pass



    # --- Phase 3 (Création Master Tuiles) ---
    base_progress_phase3 = current_global_progress
    _log_memory_usage(progress_callback, "Début Phase 3 (Master Tuiles)")
    pcb("run_info_phase3_started_from_cache", prog=base_progress_phase3, lvl="INFO")
    pcb("PHASE_UPDATE:3", prog=None, lvl="ETA_LEVEL")
    temp_master_tile_storage_dir = os.path.join(output_folder, "zemosaic_temp_master_tiles")
    try:
        if os.path.exists(temp_master_tile_storage_dir): shutil.rmtree(temp_master_tile_storage_dir)
        os.makedirs(temp_master_tile_storage_dir, exist_ok=True)
    except OSError as e_mkdir_mt: 
        pcb("run_error_phase3_mkdir_failed", prog=current_global_progress, lvl="ERROR", directory=temp_master_tile_storage_dir, error=str(e_mkdir_mt)); return
        
    master_tiles_results_list_temp = {}
    start_time_phase3 = time.monotonic()

    tile_id_order = list(range(len(seestar_stack_groups)))
    center_out_context: CenterOutNormalizationContext | None = None
    global_anchor_shift: tuple[float, float] = (1.0, 0.0)
    prestack_anchor_tile_id: int | None = None
    center_out_settings = {
        "enabled": bool(center_out_normalization_p3_config),
        "sky_percentile": tuple((p3_center_sky_percentile_config or (25.0, 60.0))[:2]) if isinstance(p3_center_sky_percentile_config, (list, tuple)) else (25.0, 60.0),
        "clip_sigma": float(p3_center_robust_clip_sigma_config),
        "preview_size": int(p3_center_preview_size_config),
        "min_overlap_fraction": float(p3_center_min_overlap_fraction_config),
    }
    anchor_mode_value = str(center_out_anchor_mode_config or "auto_central_quality").strip()
    anchor_mode_lower = anchor_mode_value.lower()
    anchor_quality_settings = {
        "probe_limit": anchor_quality_probe_limit_config,
        "span_range": anchor_quality_span_range_config,
        "median_clip_sigma": anchor_quality_median_clip_sigma_config,
    }
    try:
        anchor_crop_band = max(4, int(quality_crop_band_px_config))
    except Exception:
        anchor_crop_band = 32
    try:
        anchor_crop_margin = max(0, int(quality_crop_margin_px_config))
    except Exception:
        anchor_crop_margin = 8
    try:
        anchor_crop_sigma = float(quality_crop_k_sigma_config)
        if not math.isfinite(anchor_crop_sigma):
            raise ValueError
    except Exception:
        anchor_crop_sigma = 2.0
    anchor_crop_sigma = max(0.1, min(anchor_crop_sigma, 10.0))
    anchor_crop_settings = {
        "enabled": bool(ANCHOR_AUTOCROP_AVAILABLE and anchor_mode_lower == "auto_central_quality"),
        "band_px": anchor_crop_band,
        "margin_px": anchor_crop_margin,
        "k_sigma": anchor_crop_sigma,
    }
    if center_out_settings["enabled"] and seestar_stack_groups:
        order_info = _compute_center_out_order(seestar_stack_groups)
        distances = {}
        global_center_coord = None
        if order_info:
            ordered_indices, global_center_coord, distances = order_info
            try:
                seestar_stack_groups = [seestar_stack_groups[i] for i in ordered_indices]
                tile_id_order = ordered_indices
            except Exception:
                tile_id_order = list(range(len(seestar_stack_groups)))
        else:
            distances = {}
            global_center_coord = None

        anchor_original_id: int | None = tile_id_order[0] if tile_id_order else None
        if tile_id_order and anchor_mode_lower == "auto_central_quality":
            selected_anchor = _select_quality_anchor(
                tile_id_order,
                distances,
                seestar_stack_groups,
                anchor_quality_settings,
                center_out_settings,
                anchor_crop_settings,
                progress_callback,
            )
            if selected_anchor is not None and selected_anchor in tile_id_order:
                if selected_anchor != tile_id_order[0]:
                    try:
                        sel_index = tile_id_order.index(selected_anchor)
                        tile_id_order.insert(0, tile_id_order.pop(sel_index))
                        seestar_stack_groups.insert(0, seestar_stack_groups.pop(sel_index))
                    except Exception:
                        pass
                anchor_original_id = int(selected_anchor)
            else:
                _log_and_callback(
                    "center_anchor_fallback_central_only",
                    lvl="WARN",
                    callback=progress_callback,
                )
        elif tile_id_order:
            anchor_original_id = int(tile_id_order[0])

        try:
            pcb(
                "phase3_center_out_plan",
                prog=None,
                lvl="INFO_DETAIL",
                anchor=int(anchor_original_id) if anchor_original_id is not None else None,
                center_ra=f"{global_center_coord.ra.deg:.6f}" if global_center_coord else None,
                center_dec=f"{global_center_coord.dec.deg:.6f}" if global_center_coord else None,
            )
        except Exception:
            pass

        if tile_id_order:
            anchor_id_final = int(anchor_original_id) if anchor_original_id is not None else int(tile_id_order[0])
            center_out_context = CenterOutNormalizationContext(
                anchor_tile_original_id=anchor_id_final,
                ordered_tile_ids=tile_id_order,
                tile_distances=distances,
                settings=center_out_settings,
                global_center=global_center_coord,
                logger_instance=logger,
            )
            prestack_anchor_tile_id = anchor_id_final
    else:
        center_out_settings["enabled"] = False
    
    # Calcul des workers pour la Phase 3 (alignement/stacking des groupes)
    actual_num_workers_ph3 = _compute_phase_workers(
        effective_base_workers,
        num_seestar_stacks_to_process,
        ALIGNMENT_PHASE_WORKER_RATIO,
    )
    if auto_caps_info:
        try:
            parallel_cap = int(auto_caps_info.get("parallel_groups", 0))
        except Exception:
            parallel_cap = 0
        if parallel_cap > 0:
            prev_workers = actual_num_workers_ph3
            actual_num_workers_ph3 = max(1, min(actual_num_workers_ph3, parallel_cap))
            if actual_num_workers_ph3 != prev_workers:
                try:
                    _log_and_callback(
                        f"AutoCaps: Phase 3 worker cap {prev_workers} -> {actual_num_workers_ph3} (parallel limit)",
                        prog=None,
                        lvl="INFO_DETAIL",
                        callback=progress_callback,
                    )
                except Exception:
                    pass
    # On Windows, cap Phase 3 concurrency to reduce I/O + CPU contention
    if os.name == 'nt':
        actual_num_workers_ph3 = max(1, min(actual_num_workers_ph3, 4))
    # Apply IO-based cap if available
    try:
        if io_ph3_cap is not None:
            prev_workers = actual_num_workers_ph3
            actual_num_workers_ph3 = max(1, min(actual_num_workers_ph3, int(io_ph3_cap)))
            if actual_num_workers_ph3 != prev_workers:
                pcb(
                    f"IO_ADAPT: Phase 3 workers {prev_workers} -> {actual_num_workers_ph3} due to IO ({io_read_cat})",
                    prog=None,
                    lvl="INFO_DETAIL",
                )
    except Exception:
        pass
    # RAM-aware cap for Phase 3: estimate per-job footprint and clamp concurrency
    try:
        avail_bytes = int(psutil.virtual_memory().available)
        # Determine per-frame bytes via a sample cached image when possible
        per_frame_bytes = None
        try:
            sample_cache = None
            if seestar_stack_groups and seestar_stack_groups[0]:
                sample_cache = seestar_stack_groups[0][0].get('path_preprocessed_cache')
            if sample_cache and os.path.exists(sample_cache):
                _arr = np.load(sample_cache, mmap_mode='r')
                per_frame_bytes = int(_arr.size) * int(_arr.dtype.itemsize)
                _arr = None
        except Exception:
            per_frame_bytes = None
        if per_frame_bytes is None or per_frame_bytes <= 0:
            # Conservative default (Seestar 1080x1920 RGB float32)
            per_frame_bytes = 1080 * 1920 * 3 * 4
        frames_per_pass = int(winsor_max_frames_per_pass) if winsor_max_frames_per_pass and int(winsor_max_frames_per_pass) > 0 else 256
        fudge = 2.0
        per_job_bytes = int(max(1, frames_per_pass) * per_frame_bytes * fudge)
        allowed = int(avail_bytes * 0.6)
        max_by_ram = max(1, allowed // max(1, per_job_bytes))
        prev_workers = actual_num_workers_ph3
        actual_num_workers_ph3 = max(1, min(actual_num_workers_ph3, int(max_by_ram)))
        if actual_num_workers_ph3 != prev_workers:
            try:
                mb_per_job = per_job_bytes / (1024.0 * 1024.0)
                pcb(
                    f"RAM_CAP: Phase 3 workers {prev_workers} -> {actual_num_workers_ph3} (frames/pass={frames_per_pass}, per-job~{mb_per_job:.1f}MB)",
                    prog=None,
                    lvl="INFO_DETAIL",
                )
            except Exception:
                pass
    except Exception:
        pass
    pcb(
        f"WORKERS_PHASE3: Utilisation de {actual_num_workers_ph3} worker(s). (Base: {effective_base_workers}, Ratio {ALIGNMENT_PHASE_WORKER_RATIO*100:.0f}%, Groupes: {num_seestar_stacks_to_process})",
        prog=None,
        lvl="INFO",
    )  # Log mis à jour pour clarté

    # Initialize adaptive concurrency controls for Phase 3 (I/O + tasks)
    try:
        global _PH3_CONCURRENCY_SEMAPHORE
        _PH3_CONCURRENCY_SEMAPHORE = threading.Semaphore(int(actual_num_workers_ph3))
    except Exception:
        pass

    # Start a lightweight real-time monitor to adapt concurrency while Phase 3 runs
    monitor_stop_evt = threading.Event()

    def _rt_adapt_concurrency():
        try:
            import psutil as _ps
        except Exception:
            return  # psutil absent; skip runtime adaptation
        current_ph3_limit = int(actual_num_workers_ph3)
        current_cache_slots = None
        default_cache_slots = 2 if os.name == 'nt' else 3
        last_io = None
        last_t = None
        try:
            last_io = _ps.disk_io_counters()
            last_t = time.perf_counter()
        except Exception:
            last_io, last_t = None, None
        while not monitor_stop_evt.is_set():
            time.sleep(1.25)
            # CPU snapshot
            try:
                cpu_pct = _ps.cpu_percent(interval=None)
            except Exception:
                cpu_pct = None
            # Disk read throughput MB/s
            read_mbps = None
            try:
                if last_io is not None:
                    now_io = _ps.disk_io_counters()
                    now_t = time.perf_counter()
                    dt = max(1e-3, (now_t - (last_t or now_t)))
                    read_mbps = (max(0, now_io.read_bytes - last_io.read_bytes) / dt) / (1024 * 1024)
                    last_io, last_t = now_io, now_t
            except Exception:
                pass

            new_ph3_limit = current_ph3_limit
            new_cache_slots = current_cache_slots if current_cache_slots is not None else default_cache_slots

            if read_mbps is not None:
                if os.name == 'nt':
                    if read_mbps >= 120:
                        new_ph3_limit = 1
                        new_cache_slots = 1
                    elif read_mbps >= 80:
                        new_ph3_limit = min(new_ph3_limit, 2)
                        new_cache_slots = 1
                    elif read_mbps >= 40:
                        new_cache_slots = 2
                    else:
                        new_cache_slots = default_cache_slots
                else:
                    if read_mbps >= 200:
                        new_ph3_limit = max(1, min(new_ph3_limit, 2))
                        new_cache_slots = 2
                    elif read_mbps >= 120:
                        new_cache_slots = 2
                    else:
                        new_cache_slots = default_cache_slots

            if cpu_pct is not None:
                if cpu_pct >= 90:
                    new_ph3_limit = max(1, min(new_ph3_limit, 2 if os.name == 'nt' else 3))
                elif cpu_pct <= 45:
                    new_ph3_limit = max(new_ph3_limit, min(int(actual_num_workers_ph3), 3 if os.name == 'nt' else int(actual_num_workers_ph3)))

            new_ph3_limit = max(1, min(int(actual_num_workers_ph3), int(new_ph3_limit)))
            new_cache_slots = max(1, int(new_cache_slots))

            try:
                if new_ph3_limit != current_ph3_limit:
                    current_ph3_limit = new_ph3_limit
                    try:
                        global _PH3_CONCURRENCY_SEMAPHORE
                        _PH3_CONCURRENCY_SEMAPHORE = threading.Semaphore(int(current_ph3_limit))
                        pcb(f"IO_ADAPT_RT: ph3_workers -> {current_ph3_limit}", prog=None, lvl="INFO_DETAIL")
                    except Exception:
                        pass
                if (current_cache_slots is None) or (new_cache_slots != current_cache_slots):
                    current_cache_slots = new_cache_slots
                    try:
                        global _CACHE_IO_SEMAPHORE
                        _CACHE_IO_SEMAPHORE = threading.Semaphore(int(current_cache_slots))
                        pcb(f"IO_ADAPT_RT: cache_read_slots -> {current_cache_slots}", prog=None, lvl="INFO_DETAIL")
                    except Exception:
                        pass
            except Exception:
                pass

    monitor_thread = threading.Thread(target=_rt_adapt_concurrency, name="ZeMosaic_Ph3_RTAdapt", daemon=True)
    monitor_thread.start()

    tiles_processed_count_ph3 = 0
    # Envoyer l'info initiale avant la boucle
    if num_seestar_stacks_to_process > 0:
        pcb(f"MASTER_TILE_COUNT_UPDATE:{tiles_processed_count_ph3}/{num_seestar_stacks_to_process}", prog=None, lvl="ETA_LEVEL")
    
    executor_ph3 = ThreadPoolExecutor(max_workers=actual_num_workers_ph3, thread_name_prefix="ZeMosaic_Ph3_")

    future_to_tile_id: dict = {}
    tile_input_cache_paths: dict[int, list[str]] = {}
    pending_futures: set = set()
    next_dynamic_tile_id = num_seestar_stacks_to_process

    def _submit_master_tile_group(group_info_list: list[dict], assigned_tile_id: int, processing_rank: int | None = None) -> None:
        future = executor_ph3.submit(
            create_master_tile,
            group_info_list,
            assigned_tile_id,
            temp_master_tile_storage_dir,
            stack_norm_method, stack_weight_method, stack_reject_algo,
            stack_kappa_low, stack_kappa_high, parsed_winsor_limits,
            stack_final_combine,
            poststack_equalize_rgb_config,
            apply_radial_weight_config, radial_feather_fraction_config,
            radial_shape_power_config, min_radial_weight_floor_config,
            quality_crop_enabled_config, quality_crop_band_px_config,
            quality_crop_k_sigma_config, quality_crop_margin_px_config,
            astap_exe_path, astap_data_dir_param, astap_search_radius_config,
            astap_downsample_config, astap_sensitivity_config, 180,
            winsor_worker_limit,
            winsor_max_frames_per_pass,
            progress_callback,
            resource_strategy=auto_resource_strategy,
            center_out_context=center_out_context,
            center_out_settings=center_out_settings if center_out_context else None,
            center_out_rank=processing_rank,
        )
        future_to_tile_id[future] = assigned_tile_id
        pending_futures.add(future)
        if cache_retention_mode == "per_tile":
            cache_paths: list[str] = []
            for raw_entry in group_info_list or []:
                if not isinstance(raw_entry, dict):
                    continue
                cache_path = raw_entry.get('path_preprocessed_cache')
                if isinstance(cache_path, str):
                    cache_paths.append(cache_path)
            tile_input_cache_paths[assigned_tile_id] = cache_paths

    for proc_idx, sg_info_list in enumerate(seestar_stack_groups):
        assigned_tile_id = tile_id_order[proc_idx] if proc_idx < len(tile_id_order) else proc_idx
        rank = center_out_context.get_rank(assigned_tile_id) if center_out_context else proc_idx
        _submit_master_tile_group(sg_info_list, assigned_tile_id, rank)

    start_time_loop_ph3 = time.time()
    last_time_loop_ph3 = start_time_loop_ph3
    step_times_ph3 = []

    while pending_futures:
        done_futures, _ = wait(pending_futures, return_when=FIRST_COMPLETED)
        for future in done_futures:
            pending_futures.discard(future)
            tile_id_for_future = future_to_tile_id.pop(future, None)
            if tile_id_for_future is None:
                continue
            tiles_processed_count_ph3 += 1
            cache_paths_for_tile: list[str] = []
            if cache_retention_mode == "per_tile":
                cache_paths_for_tile = tile_input_cache_paths.pop(tile_id_for_future, [])

            pcb(f"MASTER_TILE_COUNT_UPDATE:{tiles_processed_count_ph3}/{num_seestar_stacks_to_process}", prog=None, lvl="ETA_LEVEL")

            prog_step_phase3 = base_progress_phase3 + int(
                PROGRESS_WEIGHT_PHASE3_MASTER_TILES * (tiles_processed_count_ph3 / max(1, num_seestar_stacks_to_process))
            )
            if progress_callback:
                try:
                    progress_callback("phase3_master_tiles", tiles_processed_count_ph3, num_seestar_stacks_to_process)
                except Exception:
                    pass

            now = time.time()
            step_times_ph3.append(now - last_time_loop_ph3)
            last_time_loop_ph3 = now

            try:
                main_result, retry_groups = future.result()
                mt_result_path, mt_result_wcs = (main_result or (None, None))
                if mt_result_path and mt_result_wcs:
                    master_tiles_results_list_temp[tile_id_for_future] = (mt_result_path, mt_result_wcs)
                    if cache_retention_mode == "per_tile" and cache_paths_for_tile:
                        removed_count, removed_bytes = _cleanup_per_tile_cache(cache_paths_for_tile)
                        freed_mb = removed_bytes / (1024 * 1024) if removed_bytes else 0.0
                        logger.debug(
                            "Per-tile cache cleanup for tile %s: removed %d file(s), freed %.3f MiB",
                            tile_id_for_future,
                            removed_count,
                            freed_mb,
                        )
                        try:
                            pcb(
                                "run_debug_cache_per_tile_cleanup",
                                prog=None,
                                lvl="DEBUG_DETAIL",
                                tile_id=int(tile_id_for_future),
                                removed=int(removed_count),
                                freed_mib=f"{freed_mb:.3f}",
                            )
                        except Exception:
                            pass
                else:
                    pcb(
                        "run_warn_phase3_master_tile_creation_failed_thread",
                        prog=prog_step_phase3,
                        lvl="WARN",
                        stack_num=int(tile_id_for_future) + 1,
                    )
                if retry_groups:
                    for retry_group in retry_groups:
                        if not retry_group:
                            continue
                        filtered_retry_group: list[dict] = []
                        dropped_infos: list[dict] = []
                        for raw_info in retry_group:
                            if isinstance(raw_info, dict):
                                attempts = int(raw_info.get('retry_attempt', 0))
                                if attempts > MAX_ALIGNMENT_RETRY_ATTEMPTS:
                                    dropped_infos.append(raw_info)
                                    continue
                            filtered_retry_group.append(raw_info)
                        for dropped in dropped_infos:
                            try:
                                filename = os.path.basename(dropped.get('path_raw', 'UnknownRaw'))
                            except Exception:
                                filename = str(dropped)
                            pcb(
                                "run_warn_phase3_alignment_retry_abandoned",
                                prog=None,
                                lvl="WARN",
                                tile_id=int(tile_id_for_future),
                                filename=filename,
                                attempts=int(dropped.get('retry_attempt', 0)) if isinstance(dropped, dict) else None,
                            )
                        if not filtered_retry_group:
                            continue
                        new_tile_id = next_dynamic_tile_id
                        next_dynamic_tile_id += 1
                        num_seestar_stacks_to_process += 1
                        pcb(
                            "run_info_phase3_retry_submitted",
                            prog=None,
                            lvl="INFO_DETAIL",
                            origin_tile=int(tile_id_for_future),
                            new_tile=new_tile_id,
                            frames=len(filtered_retry_group),
                        )
                        retry_rank = center_out_context.get_rank(new_tile_id) if center_out_context else None
                        _submit_master_tile_group(filtered_retry_group, new_tile_id, retry_rank)
                        pcb(
                            f"MASTER_TILE_COUNT_UPDATE:{tiles_processed_count_ph3}/{num_seestar_stacks_to_process}",
                            prog=None,
                            lvl="ETA_LEVEL",
                        )
                        if progress_callback:
                            try:
                                progress_callback("phase3_master_tiles", tiles_processed_count_ph3, num_seestar_stacks_to_process)
                            except Exception:
                                pass
            except Exception as exc_thread_ph3:
                pcb(
                    "run_error_phase3_thread_exception",
                    prog=prog_step_phase3,
                    lvl="ERROR",
                    stack_num=int(tile_id_for_future) + 1,
                    error=str(exc_thread_ph3),
                )
                logger.error(f"Exception Phase 3 pour stack {int(tile_id_for_future) + 1}:", exc_info=True)
            finally:
                # Aggressively free CuPy memory pools between tiles to avoid device/pinned host growth
                try:
                    if ZEMOSAIC_UTILS_AVAILABLE and zemosaic_utils and hasattr(zemosaic_utils, "free_cupy_memory_pools"):
                        zemosaic_utils.free_cupy_memory_pools()
                except Exception:
                    pass
                try:
                    gc.collect()
                except Exception:
                    pass

            if tiles_processed_count_ph3 % max(1, num_seestar_stacks_to_process // 5) == 0 or tiles_processed_count_ph3 == num_seestar_stacks_to_process:
                 _log_memory_usage(progress_callback, f"Phase 3 - Traité {tiles_processed_count_ph3}/{num_seestar_stacks_to_process} tuiles")

            elapsed_phase3 = time.monotonic() - start_time_phase3
            time_per_master_tile_creation = elapsed_phase3 / max(1, tiles_processed_count_ph3)
            eta_phase3_sec = (num_seestar_stacks_to_process - tiles_processed_count_ph3) * time_per_master_tile_creation
            current_progress_in_run_percent_ph3 = base_progress_phase3 + (tiles_processed_count_ph3 / max(1, num_seestar_stacks_to_process)) * PROGRESS_WEIGHT_PHASE3_MASTER_TILES
            time_per_percent_point_global_ph3 = (time.monotonic() - start_time_total_run) / max(1, current_progress_in_run_percent_ph3) if current_progress_in_run_percent_ph3 > 0 else (time.monotonic() - start_time_total_run)
            total_eta_sec_ph3 = eta_phase3_sec + (100 - current_progress_in_run_percent_ph3) * time_per_percent_point_global_ph3
            update_gui_eta(total_eta_sec_ph3)

    # Toutes les futures sont terminées → fermeture propre
    # Stop the runtime adaptation monitor for Phase 3
    try:
        monitor_stop_evt.set()
        if monitor_thread and monitor_thread.is_alive():
            monitor_thread.join(timeout=2.0)
    except Exception:
        pass
    executor_ph3.shutdown(wait=True)

    if enable_poststack_anchor_review_config and master_tiles_results_list_temp:
        post_review_cfg = {
            "probe_limit": poststack_anchor_probe_limit_config,
            "span_range": poststack_anchor_span_range_config,
            "median_clip_sigma": poststack_anchor_median_clip_sigma_config,
            "min_improvement": poststack_anchor_min_improvement_config,
            "use_overlap_affine": poststack_anchor_use_overlap_affine_config,
            "sky_percentile": center_out_settings.get("sky_percentile"),
        }
        try:
            master_tiles_results_list_temp, anchor_shift_candidate = run_poststack_anchor_review(
                master_tiles_results_list_temp,
                prestack_anchor_tile_id,
                post_review_cfg,
                progress_callback,
                tile_distances=getattr(center_out_context, "tile_distances", None) if center_out_context else None,
            )
            if isinstance(anchor_shift_candidate, tuple):
                global_anchor_shift = anchor_shift_candidate
        except Exception:
            logger.warning("Post-stack anchor review failed", exc_info=True)

    master_tiles_results_list = [master_tiles_results_list_temp[i] for i in sorted(master_tiles_results_list_temp.keys())]
    del master_tiles_results_list_temp; gc.collect()
    if not master_tiles_results_list:
        pcb("run_error_phase3_no_master_tiles_created", prog=(base_progress_phase3 + PROGRESS_WEIGHT_PHASE3_MASTER_TILES), lvl="ERROR"); return

    current_global_progress = base_progress_phase3 + PROGRESS_WEIGHT_PHASE3_MASTER_TILES
    _log_memory_usage(progress_callback, "Fin Phase 3");
    if step_times_ph3:
        avg_step = sum(step_times_ph3) / len(step_times_ph3)
        total_elapsed = time.time() - start_time_loop_ph3
        pcb(
            "phase3_debug_timing",
            prog=None,
            lvl="DEBUG_DETAIL",
            avg=f"{avg_step:.2f}",
            total=f"{total_elapsed:.2f}",
        )
    pcb("run_info_phase3_finished_from_cache", prog=current_global_progress, lvl="INFO", num_master_tiles=len(master_tiles_results_list))
    
    # Assurer que le compteur final est bien affiché (au cas où la dernière itération n'aurait pas été exactement le total)
    # Bien que la logique dans la boucle devrait déjà le faire. Peut être redondant mais ne fait pas de mal.
    pcb(f"MASTER_TILE_COUNT_UPDATE:{tiles_processed_count_ph3}/{num_seestar_stacks_to_process}", prog=None, lvl="ETA_LEVEL")

    logger.info("All master tiles complete, entering Phase 5 (reproject & coadd)")
    if progress_callback:
        try:
            progress_callback("run_info_phase3_finished", None, "INFO", num_master_tiles=len(master_tiles_results_list))
        except Exception:
            logger.warning("progress_callback failed for phase3 finished", exc_info=True)




    
    
    # --- Phase 4 (Calcul Grille Finale) ---
    base_progress_phase4 = current_global_progress
    _log_memory_usage(progress_callback, "Début Phase 4 (Calcul Grille)")
    pcb("run_info_phase4_started", prog=base_progress_phase4, lvl="INFO")
    pcb("PHASE_UPDATE:4", prog=None, lvl="ETA_LEVEL")
    wcs_list_for_final_grid = []; shapes_list_for_final_grid_hw = []
    start_time_loop_ph4 = time.time(); last_time_loop_ph4 = start_time_loop_ph4; step_times_ph4 = []
    total_steps_ph4 = len(master_tiles_results_list)
    for idx_loop, (mt_path_iter,mt_wcs_iter) in enumerate(master_tiles_results_list, 1):
        # ... (logique de récupération shape, inchangée) ...
        if not (mt_path_iter and os.path.exists(mt_path_iter) and mt_wcs_iter and mt_wcs_iter.is_celestial): pcb("run_warn_phase4_invalid_master_tile_for_grid", prog=None, lvl="WARN", path=os.path.basename(mt_path_iter if mt_path_iter else "N/A_path")); continue
        try:
            h_mt_loc,w_mt_loc=0,0
            if mt_wcs_iter.pixel_shape and mt_wcs_iter.pixel_shape[0] > 0 and mt_wcs_iter.pixel_shape[1] > 0 : h_mt_loc,w_mt_loc=mt_wcs_iter.pixel_shape[1],mt_wcs_iter.pixel_shape[0] 
            else: 
                with fits.open(mt_path_iter,memmap=True, do_not_scale_image_data=True) as hdul_mt_s:
                    if hdul_mt_s[0].data is None: pcb("run_warn_phase4_no_data_in_tile_fits", prog=None, lvl="WARN", path=os.path.basename(mt_path_iter)); continue
                    data_shape = hdul_mt_s[0].shape
                    if len(data_shape) == 3:
                        # data_shape == (height, width, channels)
                        h_mt_loc,w_mt_loc = data_shape[0],data_shape[1]
                    elif len(data_shape) == 2: h_mt_loc,w_mt_loc = data_shape[0],data_shape[1]
                    else: pcb("run_warn_phase4_unhandled_tile_shape", prog=None, lvl="WARN", path=os.path.basename(mt_path_iter), shape=data_shape); continue 
                    if mt_wcs_iter and mt_wcs_iter.is_celestial and mt_wcs_iter.pixel_shape is None:
                        try: mt_wcs_iter.pixel_shape=(w_mt_loc,h_mt_loc)
                        except Exception as e_set_ps: pcb("run_warn_phase4_failed_set_pixel_shape", prog=None, lvl="WARN", path=os.path.basename(mt_path_iter), error=str(e_set_ps))
            if h_mt_loc > 0 and w_mt_loc > 0: shapes_list_for_final_grid_hw.append((int(h_mt_loc),int(w_mt_loc))); wcs_list_for_final_grid.append(mt_wcs_iter)
            else: pcb("run_warn_phase4_zero_dimensions_tile", prog=None, lvl="WARN", path=os.path.basename(mt_path_iter))
            now = time.time(); step_times_ph4.append(now - last_time_loop_ph4); last_time_loop_ph4 = now
            if progress_callback:
                try:
                    progress_callback("phase4_grid", idx_loop, total_steps_ph4)
                except Exception:
                    pass
        except Exception as e_read_tile_shape: pcb("run_error_phase4_reading_tile_shape", prog=None, lvl="ERROR", path=os.path.basename(mt_path_iter), error=str(e_read_tile_shape)); logger.error(f"Erreur lecture shape tuile {os.path.basename(mt_path_iter)}:", exc_info=True); continue
    if not wcs_list_for_final_grid or not shapes_list_for_final_grid_hw or len(wcs_list_for_final_grid) != len(shapes_list_for_final_grid_hw): pcb("run_error_phase4_insufficient_tile_info", prog=(base_progress_phase4 + PROGRESS_WEIGHT_PHASE4_GRID_CALC), lvl="ERROR"); return
    final_mosaic_drizzle_scale = 1.0 
    final_output_wcs, final_output_shape_hw = _calculate_final_mosaic_grid(wcs_list_for_final_grid, shapes_list_for_final_grid_hw, final_mosaic_drizzle_scale, progress_callback)
    if not final_output_wcs or not final_output_shape_hw: pcb("run_error_phase4_grid_calc_failed", prog=(base_progress_phase4 + PROGRESS_WEIGHT_PHASE4_GRID_CALC), lvl="ERROR"); return
    current_global_progress = base_progress_phase4 + PROGRESS_WEIGHT_PHASE4_GRID_CALC
    _log_memory_usage(progress_callback, "Fin Phase 4");
    if step_times_ph4:
        avg_step = sum(step_times_ph4) / len(step_times_ph4)
        total_elapsed = time.time() - start_time_loop_ph4
        pcb(
            "phase4_debug_timing",
            prog=None,
            lvl="DEBUG_DETAIL",
            avg=f"{avg_step:.2f}",
            total=f"{total_elapsed:.2f}",
        )
    pcb("run_info_phase4_finished", prog=current_global_progress, lvl="INFO", shape=final_output_shape_hw, crval=final_output_wcs.wcs.crval if final_output_wcs.wcs else 'N/A')

    base_progress_phase4_5 = current_global_progress
    phase45_active_flag = bool(inter_master_merge_enable_config)
    if inter_master_merge_enable_config:
        pcb("PHASE_UPDATE:4.5", prog=None, lvl="ETA_LEVEL")
        _log_memory_usage(progress_callback, "Début Phase 4.5 (Inter-Master merge)")
        inter_cfg_phase45 = {
            "enable": bool(inter_master_merge_enable_config),
            "overlap_threshold": float(inter_master_overlap_threshold_config),
            "min_group_size": int(inter_master_min_group_size_config),
            "stack_method": str(inter_master_stack_method_config).lower(),
            "memmap_policy": str(inter_master_memmap_policy_config).lower(),
            "local_scale": str(inter_master_local_scale_config).lower(),
            "max_group": int(inter_master_max_group_config),
        }
        try:
            photometry_clip_cfg = float(worker_config_cache.get("inter_master_photometry_clip_sigma", 3.0))
        except Exception:
            photometry_clip_cfg = 3.0
        if not math.isfinite(photometry_clip_cfg):
            photometry_clip_cfg = 3.0
        inter_cfg_phase45.update(
            {
                "photometry_intragroup": bool(worker_config_cache.get("inter_master_photometry_intragroup", True)),
                "photometry_intersuper": bool(worker_config_cache.get("inter_master_photometry_intersuper", True)),
                "photometry_clip_sigma": max(0.1, photometry_clip_cfg),
            }
        )
        gain_clip_cfg = worker_config_cache.get("two_pass_cov_gain_clip")
        if isinstance(gain_clip_cfg, (list, tuple)) and len(gain_clip_cfg) >= 2:
            try:
                gmin = float(gain_clip_cfg[0])
                gmax = float(gain_clip_cfg[1])
                if math.isfinite(gmin) and math.isfinite(gmax):
                    if gmin > gmax:
                        gmin, gmax = gmax, gmin
                    inter_cfg_phase45["two_pass_cov_gain_clip"] = (gmin, gmax)
            except Exception:
                pass
        stack_cfg_phase45 = {
            "kappa_low": float(stack_kappa_low),
            "kappa_high": float(stack_kappa_high),
            "winsor_limits": parsed_winsor_limits,
            "winsor_max_frames_per_pass": winsor_max_frames_per_pass_config,
            "winsor_worker_limit": winsor_worker_limit_config,
            # Phase 4.5 uses GUI stacking options directly; include canonical and legacy keys
            "normalize_method": stack_norm_method,
            "weight_method": stack_weight_method,
            "reject_algo": stack_reject_algo,
            "final_combine": stack_final_combine,
            "stack_norm_method": stack_norm_method,
            "stack_weight_method": stack_weight_method,
            "stack_reject_algo": stack_reject_algo,
            "stack_final_combine": stack_final_combine,
        }
        master_tiles_results_list = _run_phase4_5_inter_master_merge(
            master_tiles_results_list,
            final_output_wcs,
            final_output_shape_hw,
            temp_master_tile_storage_dir,
            output_folder,
            cache_retention_mode,
            inter_cfg_phase45,
            stack_cfg_phase45,
            progress_callback,
            pcb,
        )
        current_global_progress = base_progress_phase4_5 + PROGRESS_WEIGHT_PHASE4_5_INTER_MASTER
        _log_memory_usage(progress_callback, "Fin Phase 4.5")
    else:
        current_global_progress = base_progress_phase4_5 + PROGRESS_WEIGHT_PHASE4_5_INTER_MASTER

# --- Phase 5 (Assemblage Final) ---
    base_progress_phase5 = current_global_progress
    pcb("PHASE_UPDATE:5", prog=None, lvl="ETA_LEVEL")
    USE_INCREMENTAL_ASSEMBLY = (final_assembly_method_config == "incremental")
    apply_crop_for_assembly = bool(apply_master_tile_crop_config and not quality_crop_enabled_config)
    _log_memory_usage(
        progress_callback,
        (
            "Début Phase 5 (Méthode: "
            f"{final_assembly_method_config}, "
            f"Rognage MT Appliqué: {apply_crop_for_assembly}, "
            f"QualityCrop: {quality_crop_enabled_config}, "
            f"%Rognage: {master_tile_crop_percent_config if apply_crop_for_assembly else 'N/A'})"
        ),
    )
    
    incremental_parity_active = (
        USE_INCREMENTAL_ASSEMBLY
        and intertile_match_flag
        and match_background_flag
        and feather_parity_flag
    )
    if incremental_parity_active and two_pass_enabled:
        two_pass_enabled = False
        pcb("run_info_incremental_two_pass_parity_disabled", prog=None, lvl="INFO_DETAIL")

    valid_master_tiles_for_assembly = []
    for mt_p, mt_w in master_tiles_results_list:
        if mt_p and os.path.exists(mt_p) and mt_w and mt_w.is_celestial: 
            valid_master_tiles_for_assembly.append((mt_p, mt_w))
        else:
            pcb("run_warn_phase5_invalid_tile_skipped_for_assembly", prog=None, lvl="WARN", filename=os.path.basename(mt_p if mt_p else 'N/A')) # Clé de log plus spécifique
            
    if not valid_master_tiles_for_assembly: 
        pcb("run_error_phase5_no_valid_tiles_for_assembly", prog=(base_progress_phase5 + PROGRESS_WEIGHT_PHASE5_ASSEMBLY), lvl="ERROR")
        # Nettoyage optionnel ici avant de retourner si besoin
        return

    final_mosaic_data_HWC, final_mosaic_coverage_HW = None, None
    collected_tiles_for_second_pass: list[tuple[np.ndarray, Any]] | None = (
        [] if two_pass_enabled and not USE_INCREMENTAL_ASSEMBLY else None
    )
    log_key_phase5_failed, log_key_phase5_finished = "", ""

    # Vérification de la disponibilité des fonctions d'assemblage
    # (Tu pourrais les importer en haut du module pour éviter le check 'in globals()' à chaque fois)
    reproject_coadd_available = ('assemble_final_mosaic_reproject_coadd' in globals() and callable(assemble_final_mosaic_reproject_coadd))
    incremental_available = ('assemble_final_mosaic_incremental' in globals() and callable(assemble_final_mosaic_incremental))

    if USE_INCREMENTAL_ASSEMBLY:
        if not incremental_available: 
            pcb("run_error_phase5_inc_func_missing", prog=None, lvl="CRITICAL"); return
        pcb("run_info_phase5_started_incremental", prog=base_progress_phase5, lvl="INFO")
        inc_memmap_dir = temp_master_tile_storage_dir or output_folder
        if use_gpu_phase5_flag:
            try:
                import cupy
                cupy.cuda.Device(0).use()
                # Incremental GPU path not implemented; use CPU incremental assembly.
                final_mosaic_data_HWC, final_mosaic_coverage_HW = assemble_final_mosaic_incremental(
                    master_tile_fits_with_wcs_list=valid_master_tiles_for_assembly,
                    final_output_wcs=final_output_wcs,
                    final_output_shape_hw=final_output_shape_hw,
                    progress_callback=progress_callback,
                    n_channels=3,
                    apply_crop=apply_crop_for_assembly,
                    crop_percent=master_tile_crop_percent_config,
                    processing_threads=assembly_process_workers_config,
                    memmap_dir=inc_memmap_dir,
                    cleanup_memmap=True,
                    intertile_photometric_match=intertile_match_flag,
                    intertile_preview_size=int(intertile_preview_size_config),
                    intertile_overlap_min=float(intertile_overlap_min_config),
                    intertile_sky_percentile=intertile_sky_percentile_tuple,
                    intertile_robust_clip_sigma=float(intertile_robust_clip_sigma_config),
                    intertile_global_recenter=bool(intertile_global_recenter_config),
                    intertile_recenter_clip=intertile_recenter_clip_tuple,
                    use_auto_intertile=bool(use_auto_intertile_config),
                    match_background=match_background_flag,
                    feather_parity=feather_parity_flag,
                    two_pass_coverage_renorm=bool(two_pass_coverage_renorm_config),
                    base_progress_phase5=base_progress_phase5,
                    progress_weight_phase5=PROGRESS_WEIGHT_PHASE5_ASSEMBLY,
                    start_time_total_run=start_time_total_run,
                    global_anchor_shift=global_anchor_shift,
                )
            except Exception as e_gpu:
                logger.warning("GPU incremental assembly failed, falling back to CPU: %s", e_gpu)
                final_mosaic_data_HWC, final_mosaic_coverage_HW = assemble_final_mosaic_incremental(
                    master_tile_fits_with_wcs_list=valid_master_tiles_for_assembly,
                    final_output_wcs=final_output_wcs,
                    final_output_shape_hw=final_output_shape_hw,
                    progress_callback=progress_callback,
                    n_channels=3,
                    apply_crop=apply_crop_for_assembly,
                    crop_percent=master_tile_crop_percent_config,
                    processing_threads=assembly_process_workers_config,
                    memmap_dir=inc_memmap_dir,
                    cleanup_memmap=True,
                    intertile_photometric_match=intertile_match_flag,
                    intertile_preview_size=int(intertile_preview_size_config),
                    intertile_overlap_min=float(intertile_overlap_min_config),
                    intertile_sky_percentile=intertile_sky_percentile_tuple,
                    intertile_robust_clip_sigma=float(intertile_robust_clip_sigma_config),
                    intertile_global_recenter=bool(intertile_global_recenter_config),
                    intertile_recenter_clip=intertile_recenter_clip_tuple,
                    use_auto_intertile=bool(use_auto_intertile_config),
                    match_background=match_background_flag,
                    feather_parity=feather_parity_flag,
                    two_pass_coverage_renorm=bool(two_pass_coverage_renorm_config),
                    base_progress_phase5=base_progress_phase5,
                    progress_weight_phase5=PROGRESS_WEIGHT_PHASE5_ASSEMBLY,
                    start_time_total_run=start_time_total_run,
                    global_anchor_shift=global_anchor_shift,
                )
        else:
            final_mosaic_data_HWC, final_mosaic_coverage_HW = assemble_final_mosaic_incremental(
                master_tile_fits_with_wcs_list=valid_master_tiles_for_assembly,
                final_output_wcs=final_output_wcs,
                final_output_shape_hw=final_output_shape_hw,
                progress_callback=progress_callback,
                n_channels=3,
                apply_crop=apply_crop_for_assembly,
                crop_percent=master_tile_crop_percent_config,
                processing_threads=assembly_process_workers_config,
                memmap_dir=inc_memmap_dir,
                cleanup_memmap=True,
                intertile_photometric_match=intertile_match_flag,
                intertile_preview_size=int(intertile_preview_size_config),
                intertile_overlap_min=float(intertile_overlap_min_config),
                intertile_sky_percentile=intertile_sky_percentile_tuple,
                intertile_robust_clip_sigma=float(intertile_robust_clip_sigma_config),
                intertile_global_recenter=bool(intertile_global_recenter_config),
                intertile_recenter_clip=intertile_recenter_clip_tuple,
                use_auto_intertile=bool(use_auto_intertile_config),
                match_background=match_background_flag,
                feather_parity=feather_parity_flag,
                two_pass_coverage_renorm=bool(two_pass_coverage_renorm_config),
                base_progress_phase5=base_progress_phase5,
                progress_weight_phase5=PROGRESS_WEIGHT_PHASE5_ASSEMBLY,
                start_time_total_run=start_time_total_run,
                global_anchor_shift=global_anchor_shift,
            )
        log_key_phase5_failed = "run_error_phase5_assembly_failed_incremental"
        log_key_phase5_finished = "run_info_phase5_finished_incremental"
    else: # Méthode Reproject & Coadd
        if not reproject_coadd_available: 
            pcb("run_error_phase5_reproject_coadd_func_missing", prog=None, lvl="CRITICAL"); return
        pcb("run_info_phase5_started_reproject_coadd", prog=base_progress_phase5, lvl="INFO")

        if use_gpu_phase5_flag:
            try:
                import cupy

                cupy.cuda.Device(0).use()
                # Use the internal CPU/GPU wrapper with use_gpu=True
                final_mosaic_data_HWC, final_mosaic_coverage_HW = assemble_final_mosaic_reproject_coadd(
                    master_tile_fits_with_wcs_list=valid_master_tiles_for_assembly,
                    final_output_wcs=final_output_wcs,
                    final_output_shape_hw=final_output_shape_hw,
                    progress_callback=progress_callback,
                    n_channels=3,
                    match_bg=True,
                    apply_crop=apply_crop_for_assembly,
                    crop_percent=master_tile_crop_percent_config,
                    use_gpu=True,
                    base_progress_phase5=base_progress_phase5,
                    progress_weight_phase5=PROGRESS_WEIGHT_PHASE5_ASSEMBLY,
                    start_time_total_run=start_time_total_run,
                    intertile_photometric_match=intertile_match_flag,
                    intertile_preview_size=int(intertile_preview_size_config),
                    intertile_overlap_min=float(intertile_overlap_min_config),
                    intertile_sky_percentile=intertile_sky_percentile_tuple,
                    intertile_robust_clip_sigma=float(intertile_robust_clip_sigma_config),
                    intertile_global_recenter=bool(intertile_global_recenter_config),
                    intertile_recenter_clip=intertile_recenter_clip_tuple,
                    use_auto_intertile=bool(use_auto_intertile_config),
                    collect_tile_data=collected_tiles_for_second_pass,
                    global_anchor_shift=global_anchor_shift,
                    phase45_enabled=phase45_active_flag,
                )
            except Exception as e_gpu:
                logger.warning("GPU reproject_coadd failed, falling back to CPU: %s", e_gpu)
                final_mosaic_data_HWC, final_mosaic_coverage_HW = assemble_final_mosaic_reproject_coadd(
                    master_tile_fits_with_wcs_list=valid_master_tiles_for_assembly,
                    final_output_wcs=final_output_wcs,
                    final_output_shape_hw=final_output_shape_hw,
                    progress_callback=progress_callback,
                    n_channels=3,
                    match_bg=True,
                    apply_crop=apply_crop_for_assembly,
                    crop_percent=master_tile_crop_percent_config,
                    use_gpu=False,
                    use_memmap=bool(coadd_use_memmap_config),
                    memmap_dir=(coadd_memmap_dir_config or output_folder),
                    cleanup_memmap=False,
                    base_progress_phase5=base_progress_phase5,
                    progress_weight_phase5=PROGRESS_WEIGHT_PHASE5_ASSEMBLY,
                    start_time_total_run=start_time_total_run,
                    intertile_photometric_match=intertile_match_flag,
                    intertile_preview_size=int(intertile_preview_size_config),
                    intertile_overlap_min=float(intertile_overlap_min_config),
                    intertile_sky_percentile=intertile_sky_percentile_tuple,
                    intertile_robust_clip_sigma=float(intertile_robust_clip_sigma_config),
                    intertile_global_recenter=bool(intertile_global_recenter_config),
                    intertile_recenter_clip=intertile_recenter_clip_tuple,
                    use_auto_intertile=bool(use_auto_intertile_config),
                    collect_tile_data=collected_tiles_for_second_pass,
                    global_anchor_shift=global_anchor_shift,
                    phase45_enabled=phase45_active_flag,
                )
        else:
            final_mosaic_data_HWC, final_mosaic_coverage_HW = assemble_final_mosaic_reproject_coadd(
                master_tile_fits_with_wcs_list=valid_master_tiles_for_assembly,
                final_output_wcs=final_output_wcs,
                final_output_shape_hw=final_output_shape_hw,
                progress_callback=progress_callback,
                n_channels=3,
                match_bg=True,
                apply_crop=apply_crop_for_assembly,
                crop_percent=master_tile_crop_percent_config,
                use_gpu=use_gpu_phase5_flag,
                use_memmap=bool(coadd_use_memmap_config),
                memmap_dir=(coadd_memmap_dir_config or output_folder),
                cleanup_memmap=False,
                base_progress_phase5=base_progress_phase5,
                progress_weight_phase5=PROGRESS_WEIGHT_PHASE5_ASSEMBLY,
                start_time_total_run=start_time_total_run,
                intertile_photometric_match=intertile_match_flag,
                intertile_preview_size=int(intertile_preview_size_config),
                intertile_overlap_min=float(intertile_overlap_min_config),
                intertile_sky_percentile=intertile_sky_percentile_tuple,
                intertile_robust_clip_sigma=float(intertile_robust_clip_sigma_config),
                intertile_global_recenter=bool(intertile_global_recenter_config),
                intertile_recenter_clip=intertile_recenter_clip_tuple,
                use_auto_intertile=bool(use_auto_intertile_config),
                collect_tile_data=collected_tiles_for_second_pass,
                global_anchor_shift=global_anchor_shift,
                phase45_enabled=phase45_active_flag,
            )

        log_key_phase5_failed = "run_error_phase5_assembly_failed_reproject_coadd"
        log_key_phase5_finished = "run_info_phase5_finished_reproject_coadd"

    if final_mosaic_data_HWC is None: 
        pcb(log_key_phase5_failed, prog=(base_progress_phase5 + PROGRESS_WEIGHT_PHASE5_ASSEMBLY), lvl="ERROR")
        # Nettoyage optionnel ici
        return
        
    current_global_progress = base_progress_phase5 + PROGRESS_WEIGHT_PHASE5_ASSEMBLY

    if two_pass_enabled and final_mosaic_coverage_HW is not None:
        if logger:
            logger.info(
                "[TwoPass] Second pass requested (sigma=%s, clip=%s); coverage shape=%s",
                two_pass_sigma_px,
                gain_clip_tuple,
                getattr(final_mosaic_coverage_HW, "shape", None),
            )
        try:
            tiles_for_second_pass: list[np.ndarray] = []
            wcs_for_second_pass: list[Any] = []
            if collected_tiles_for_second_pass:
                for arr, twcs in collected_tiles_for_second_pass:
                    if arr is None or twcs is None:
                        continue
                    tiles_for_second_pass.append(np.asarray(arr, dtype=np.float32))
                    wcs_for_second_pass.append(twcs)
            elif USE_INCREMENTAL_ASSEMBLY:
                tiles_for_second_pass, wcs_for_second_pass = _load_master_tiles_for_two_pass(
                    valid_master_tiles_for_assembly,
                    apply_crop=apply_crop_for_assembly,
                    crop_percent=master_tile_crop_percent_config,
                    logger=logger,
                )
            if logger:
                logger.debug(
                    "[TwoPass] Prepared %d tiles for second pass (collected=%s, incremental=%s)",
                    len(tiles_for_second_pass),
                    bool(collected_tiles_for_second_pass),
                    USE_INCREMENTAL_ASSEMBLY,
                )
            if tiles_for_second_pass and wcs_for_second_pass:
                result = run_second_pass_coverage_renorm(
                    tiles_for_second_pass,
                    wcs_for_second_pass,
                    final_output_wcs,
                    final_mosaic_coverage_HW,
                    final_output_shape_hw,
                    sigma_px=two_pass_sigma_px,
                    gain_clip=gain_clip_tuple,
                    logger=logger,
                    use_gpu_two_pass=use_gpu_phase5_flag,
                )
                if result is not None:
                    final_mosaic_data_HWC, final_mosaic_coverage_HW = result
                    logger.info(
                        "[TwoPass] coverage-renorm OK (σ=%s, clip=[%.3f, %.3f])",
                        two_pass_sigma_px,
                        gain_clip_tuple[0],
                        gain_clip_tuple[1],
                    )
                else:
                    logger.warning("[TwoPass] renorm failed → keeping first-pass outputs")
            else:
                logger.warning("[TwoPass] No tiles available for coverage renorm; keeping first-pass outputs")
        except Exception:
            logger.exception("[TwoPass] renorm exception → keeping first-pass outputs")
        finally:
            if collected_tiles_for_second_pass is not None:
                collected_tiles_for_second_pass.clear()
            tiles_for_second_pass = []
            wcs_for_second_pass = []

    _log_memory_usage(progress_callback, "Fin Phase 5 (Assemblage)")
    pcb(
        log_key_phase5_finished,
        prog=current_global_progress,
        lvl="INFO",
        shape=final_mosaic_data_HWC.shape if final_mosaic_data_HWC is not None else "N/A",
    )
    

    # --- Phase 6 (Sauvegarde) ---
    base_progress_phase6 = current_global_progress
    pcb("PHASE_UPDATE:6", prog=None, lvl="ETA_LEVEL")
    _log_memory_usage(progress_callback, "Début Phase 6 (Sauvegarde)")
    pcb("run_info_phase6_started", prog=base_progress_phase6, lvl="INFO")
    output_base_name = f"zemosaic_MT{len(master_tiles_results_list)}_R{len(all_raw_files_processed_info)}"
    final_fits_path = os.path.join(output_folder, f"{output_base_name}.fits")
    
    final_header = fits.Header() 
    if final_output_wcs:
        try: final_header.update(final_output_wcs.to_header(relax=True))
        except Exception as e_hdr_wcs: pcb("run_warn_phase6_wcs_to_header_failed", error=str(e_hdr_wcs), lvl="WARN")
    
    final_header['SOFTWARE']=('ZeMosaic v3.2.9','Mosaic Software') # Incrémente la version 
    final_header['NMASTILE']=(len(master_tiles_results_list),"Master Tiles combined")
    final_header['NRAWINIT']=(num_total_raw_files,"Initial raw images found")
    final_header['NRAWPROC']=(len(all_raw_files_processed_info),"Raw images with WCS processed")
    # ... (autres clés de config comme ASTAP, Stacking, etc.) ...
    final_header['STK_NORM'] = (str(stack_norm_method), 'Stacking: Normalization Method')
    final_header['STK_WGHT'] = (str(stack_weight_method), 'Stacking: Weighting Method')
    if apply_radial_weight_config:
        final_header['STK_RADW'] = (True, 'Stacking: Radial Weighting Applied')
        final_header['STK_RADFF'] = (radial_feather_fraction_config, 'Stacking: Radial Feather Fraction')
        final_header['STK_RADPW'] = (radial_shape_power_config, 'Stacking: Radial Weight Shape Power')
        final_header['STK_RADFLR'] = (min_radial_weight_floor_config, 'Stacking: Min Radial Weight Floor')
    else:
        final_header['STK_RADW'] = (False, 'Stacking: Radial Weighting Applied')
    final_header['STK_REJ'] = (str(stack_reject_algo), 'Stacking: Rejection Algorithm')
    # ... (kappa, winsor si pertinent pour l'algo de rejet) ...
    final_header['STK_COMB'] = (str(stack_final_combine), 'Stacking: Final Combine Method')
    final_header['ZMASMBMTH'] = (final_assembly_method_config, 'Final Assembly Method')
    final_header['ZM_WORKERS'] = (num_base_workers_config, 'GUI: Base workers config (0=auto)')

    try:
        if not (ZEMOSAIC_UTILS_AVAILABLE and zemosaic_utils): 
            raise RuntimeError("zemosaic_utils non disponible pour sauvegarde FITS.")
        legacy_rgb_flag = bool(legacy_rgb_cube_config)
        if bool(save_final_as_uint16_config) and not legacy_rgb_flag:
            if not hasattr(zemosaic_utils, "write_final_fits_uint16_color_aware"):
                raise RuntimeError("write_final_fits_uint16_color_aware unavailable in zemosaic_utils")
            is_rgb = (
                isinstance(final_mosaic_data_HWC, np.ndarray)
                and final_mosaic_data_HWC.ndim == 3
                and final_mosaic_data_HWC.shape[-1] == 3
            )
            zemosaic_utils.write_final_fits_uint16_color_aware(
                final_fits_path,
                final_mosaic_data_HWC,
                header=final_header,
                force_rgb_planes=is_rgb,
                legacy_rgb_cube=legacy_rgb_flag,
                overwrite=True,
            )
            if is_rgb:
                pcb(
                    "run_info_phase6_saved_uint16_rgb_planes",
                    prog=None,
                    lvl="INFO_DETAIL",
                    filename=os.path.basename(final_fits_path),
                )
        else:
            zemosaic_utils.save_fits_image(
                image_data=final_mosaic_data_HWC,
                output_path=final_fits_path,
                header=final_header,
                overwrite=True,
                save_as_float=not save_final_as_uint16_config,
                legacy_rgb_cube=legacy_rgb_flag,
                progress_callback=progress_callback,
                axis_order="HWC",
            )

            if (
                ZEMOSAIC_UTILS_AVAILABLE
                and hasattr(zemosaic_utils, "write_final_fits_uint16_color_aware")
            ):
                viewer_fits_path = os.path.join(output_folder, f"{output_base_name}_viewer.fits")
                try:
                    zemosaic_utils.write_final_fits_uint16_color_aware(
                        viewer_fits_path,
                        final_mosaic_data_HWC,
                        header=final_header,
                        force_rgb_planes=isinstance(final_mosaic_data_HWC, np.ndarray)
                        and final_mosaic_data_HWC.ndim == 3
                        and final_mosaic_data_HWC.shape[-1] == 3,
                        legacy_rgb_cube=legacy_rgb_flag,
                        overwrite=True,
                    )
                    pcb(
                        "run_info_phase6_viewer_fits_saved",
                        prog=None,
                        lvl="INFO_DETAIL",
                        filename=os.path.basename(viewer_fits_path),
                    )
                except Exception as e_viewer:
                    pcb(
                        "run_warn_phase6_viewer_fits_failed",
                        prog=None,
                        lvl="WARN",
                        error=str(e_viewer),
                    )
        
        if final_mosaic_coverage_HW is not None and np.any(final_mosaic_coverage_HW):
            coverage_path = os.path.join(output_folder, f"{output_base_name}_coverage.fits")
            cov_hdr = fits.Header() 
            if ASTROPY_AVAILABLE and final_output_wcs: 
                try: cov_hdr.update(final_output_wcs.to_header(relax=True))
                except: pass 
            cov_hdr['EXTNAME']=('COVERAGE','Coverage Map') 
            cov_hdr['BUNIT']=('count','Pixel contributions or sum of weights')
            zemosaic_utils.save_fits_image(
                final_mosaic_coverage_HW,
                coverage_path,
                header=cov_hdr,
                overwrite=True,
                save_as_float=True,
                progress_callback=progress_callback,
                axis_order="HWC",
            )
            pcb("run_info_coverage_map_saved", prog=None, lvl="INFO_DETAIL", filename=os.path.basename(coverage_path))
        
        current_global_progress = base_progress_phase6 + PROGRESS_WEIGHT_PHASE6_SAVE
        pcb("run_success_mosaic_saved", prog=current_global_progress, lvl="SUCCESS", filename=os.path.basename(final_fits_path))
    except Exception as e_save_m: 
        pcb("run_error_phase6_save_failed", prog=(base_progress_phase6 + PROGRESS_WEIGHT_PHASE6_SAVE), lvl="ERROR", error=str(e_save_m))
        logger.error("Erreur sauvegarde FITS final:", exc_info=True)
        # En cas d'échec de sauvegarde, on ne peut pas générer de preview car final_mosaic_data_HWC pourrait être le problème.
        # On essaie quand même de nettoyer avant de retourner.
        if 'final_mosaic_data_HWC' in locals() and final_mosaic_data_HWC is not None: del final_mosaic_data_HWC
        if 'final_mosaic_coverage_HW' in locals() and final_mosaic_coverage_HW is not None: del final_mosaic_coverage_HW
        gc.collect()
        return

    _log_memory_usage(progress_callback, "Fin Sauvegarde FITS (avant preview)")

    # --- MODIFIÉ : Génération de la Preview PNG avec stretch_auto_asifits_like ---
    if final_mosaic_data_HWC is not None and ZEMOSAIC_UTILS_AVAILABLE and zemosaic_utils:
        pcb("run_info_preview_stretch_started_auto_asifits", prog=None, lvl="INFO_DETAIL") # Log mis à jour
        try:
            # Downscale extremely large mosaics for preview to avoid OOM
            try:
                h_prev, w_prev = int(final_mosaic_data_HWC.shape[0]), int(final_mosaic_data_HWC.shape[1])
                max_preview_dim = 4000  # cap the longest side for preview
                step_h = max(1, h_prev // max_preview_dim)
                step_w = max(1, w_prev // max_preview_dim)
                step = max(step_h, step_w)
                if step > 1:
                    preview_view = final_mosaic_data_HWC[::step, ::step, :]
                    pcb("run_info_preview_downscale", prog=None, lvl="INFO_DETAIL", downscale_step=step, src_shape=str(final_mosaic_data_HWC.shape), preview_shape=str(preview_view.shape))
                else:
                    preview_view = final_mosaic_data_HWC
            except Exception:
                preview_view = final_mosaic_data_HWC

            # Vérifier si la fonction stretch_auto_asifits_like existe dans zemosaic_utils
            if hasattr(zemosaic_utils, 'stretch_auto_asifits_like') and callable(zemosaic_utils.stretch_auto_asifits_like):
                
                # Paramètres pour stretch_auto_asifits_like (à ajuster si besoin)
                # Ces valeurs sont des exemples, tu devras peut-être les affiner
                # ou les rendre configurables plus tard.
                preview_p_low = 2.5  # Percentile pour le point noir (plus élevé que pour asinh seul)
                preview_p_high = 99.8 # Percentile pour le point blanc initial
                                      # Facteur 'a' pour le stretch asinh après la normalisation initiale
                                      # Pour un stretch plus "doux" similaire à ASIFitsView, 'a' peut être plus grand.
                                      # ASIFitsView utilise souvent un 'midtones balance' (gamma-like) aussi.
                                      # Un 'a' de 10 comme dans ton code de test est très doux. Essayons 0.5 ou 1.0.
                preview_asinh_a = 20.0 # Test avec une valeur plus douce pour le 'a' de asinh

                # Prefer GPU stretch when GPU is enabled/available
                if use_gpu_phase5_flag and hasattr(zemosaic_utils, 'stretch_auto_asifits_like_gpu'):
                    m_stretched = zemosaic_utils.stretch_auto_asifits_like_gpu(
                        preview_view,
                        p_low=preview_p_low,
                        p_high=preview_p_high,
                        asinh_a=preview_asinh_a,
                        apply_wb=True,
                    )
                else:
                    m_stretched = zemosaic_utils.stretch_auto_asifits_like(
                        preview_view,
                        p_low=preview_p_low,
                        p_high=preview_p_high,
                        asinh_a=preview_asinh_a,
                        apply_wb=True  # Applique une balance des blancs automatique
                    )

                if m_stretched is not None:
                    img_u8 = (
                        np.nan_to_num(
                            np.clip(m_stretched.astype(np.float32), 0, 1)
                        )
                        * 255
                    ).astype(np.uint8)
                    png_path = os.path.join(output_folder, f"{output_base_name}_preview.png")
                    try: 
                        import cv2 # Importer cv2 seulement si nécessaire
                        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
                        if cv2.imwrite(png_path, img_bgr): 
                            pcb("run_success_preview_saved_auto_asifits", prog=None, lvl="SUCCESS", filename=os.path.basename(png_path))
                        else: 
                            pcb("run_warn_preview_imwrite_failed_auto_asifits", prog=None, lvl="WARN", filename=os.path.basename(png_path))
                    except ImportError: 
                        pcb("run_warn_preview_opencv_missing_for_auto_asifits", prog=None, lvl="WARN")
                    except Exception as e_cv2_prev: 
                        pcb("run_error_preview_opencv_failed_auto_asifits", prog=None, lvl="ERROR", error=str(e_cv2_prev))
                else:
                    pcb("run_error_preview_stretch_auto_asifits_returned_none", prog=None, lvl="ERROR")
            else:
                pcb("run_warn_preview_stretch_auto_asifits_func_missing", prog=None, lvl="WARN")
                # Fallback sur l'ancienne méthode si stretch_auto_asifits_like n'est pas trouvée
                # (Tu peux supprimer ce fallback si tu es sûr que la fonction existe)
                pcb("run_info_preview_fallback_to_simple_asinh", prog=None, lvl="DEBUG_DETAIL")
                if hasattr(zemosaic_utils, 'stretch_percentile_rgb') and zemosaic_utils.ASTROPY_VISUALIZATION_AVAILABLE:
                     m_stretched_fallback = zemosaic_utils.stretch_percentile_rgb(final_mosaic_data_HWC, p_low=0.5, p_high=99.9, independent_channels=False, asinh_a=0.01 )
                     if m_stretched_fallback is not None:
                        img_u8_fb = (np.clip(m_stretched_fallback.astype(np.float32), 0, 1) * 255).astype(np.uint8)
                        png_path_fb = os.path.join(output_folder, f"{output_base_name}_preview_fallback.png")
                        try:
                            import cv2
                            img_bgr_fb = cv2.cvtColor(img_u8_fb, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(png_path_fb, img_bgr_fb)
                            pcb("run_success_preview_saved_fallback", prog=None, lvl="INFO_DETAIL", filename=os.path.basename(png_path_fb))
                        except: pass # Ignorer erreur fallback

        except Exception as e_stretch_main: 
            pcb("run_error_preview_stretch_unexpected_main", prog=None, lvl="ERROR", error=str(e_stretch_main))
            logger.error("Erreur imprévue lors de la génération de la preview:", exc_info=True)
            
    if 'final_mosaic_data_HWC' in locals() and final_mosaic_data_HWC is not None: del final_mosaic_data_HWC
    if 'final_mosaic_coverage_HW' in locals() and final_mosaic_coverage_HW is not None: del final_mosaic_coverage_HW
    gc.collect()

    # Cleanup memmap .dat files now that arrays are released (Windows requires handles closed)
    try:
        if bool(coadd_use_memmap_config) and bool(coadd_cleanup_memmap_config) and coadd_memmap_dir_config:
            for _name in os.listdir(coadd_memmap_dir_config):
                name_l = _name.lower()
                if name_l.endswith('.dat') and (name_l.startswith('mosaic_') or name_l.startswith('coverage_') or name_l.startswith('zemosaic_')):
                    try:
                        os.remove(os.path.join(coadd_memmap_dir_config, _name))
                    except OSError:
                        pass
    except Exception:
        pass



    # --- Phase 7 (Nettoyage) ---
    # ... (contenu Phase 7 inchangé) ...
    base_progress_phase7 = current_global_progress
    _log_memory_usage(progress_callback, "Début Phase 7 (Nettoyage)")
    pcb("run_info_phase7_cleanup_starting", prog=base_progress_phase7, lvl="INFO")
    pcb("PHASE_UPDATE:7", prog=None, lvl="ETA_LEVEL")
    try:
        if cache_retention_mode == "keep":
            if os.path.exists(temp_image_cache_dir):
                pcb(
                    "run_info_temp_preprocessed_cache_kept",
                    prog=None,
                    lvl="INFO_DETAIL",
                    directory=temp_image_cache_dir,
                )
        else:
            if os.path.exists(temp_image_cache_dir):
                shutil.rmtree(temp_image_cache_dir)
                pcb(
                    "run_info_temp_preprocessed_cache_cleaned",
                    prog=None,
                    lvl="INFO_DETAIL",
                    directory=temp_image_cache_dir,
                )
        if (
            not two_pass_enabled
            and os.path.exists(temp_master_tile_storage_dir)
        ):
            shutil.rmtree(temp_master_tile_storage_dir)
            pcb("run_info_temp_master_tiles_fits_cleaned", prog=None, lvl="INFO_DETAIL", directory=temp_master_tile_storage_dir)
        elif two_pass_enabled and os.path.exists(temp_master_tile_storage_dir):
            pcb(
                "run_info_temp_master_tiles_kept_two_pass",
                prog=None,
                lvl="INFO_DETAIL",
                directory=temp_master_tile_storage_dir,
            )
    except Exception as e_clean_final: pcb("run_warn_phase7_cleanup_failed", prog=None, lvl="WARN", error=str(e_clean_final))
    current_global_progress = base_progress_phase7 + PROGRESS_WEIGHT_PHASE7_CLEANUP; current_global_progress = min(100, current_global_progress)
    _log_memory_usage(progress_callback, "Fin Phase 7"); pcb("CHRONO_STOP_REQUEST", prog=None, lvl="CHRONO_LEVEL"); update_gui_eta(0)
    total_duration_sec = time.monotonic() - start_time_total_run
    pcb("run_success_processing_completed", prog=current_global_progress, lvl="SUCCESS", duration=f"{total_duration_sec:.2f}")
    gc.collect(); _log_memory_usage(progress_callback, "Fin Run Hierarchical Mosaic (après GC final)")
    _log_alignment_warning_summary()
    logger.info(f"===== Run Hierarchical Mosaic COMPLETED in {total_duration_sec:.2f}s =====")
################################################################################
################################################################################
####

def run_hierarchical_mosaic_process(
    progress_queue,
    *args,
    solver_settings_dict=None,
    **kwargs,
):
    """Wrapper for running :func:`run_hierarchical_mosaic` in a separate process."""

    # progress_callback(stage: str, current: int, total: int)

    def queue_callback(*cb_args, **cb_kwargs):
        """Proxy callback used inside the worker process.

        It supports both legacy logging calls and the new progress
        reporting style ``progress_callback(stage, current, total)``.

        Legacy calls are forwarded unchanged as
        ``(message_key_or_raw, progress_value, level, kwargs)`` tuples.
        Stage updates are sent with ``"STAGE_PROGRESS"`` as the message key.
        """
        if (
            len(cb_args) == 3
            and not cb_kwargs
            and isinstance(cb_args[0], str)
            and isinstance(cb_args[1], int)
            and isinstance(cb_args[2], int)
        ):
            stage, current, total = cb_args
            progress_queue.put(("STAGE_PROGRESS", stage, current, {"total": total}))
            return

        message_key_or_raw = cb_args[0] if cb_args else ""
        progress_value = cb_args[1] if len(cb_args) > 1 else None
        level = cb_args[2] if len(cb_args) > 2 else cb_kwargs.pop("level", "INFO")
        if "lvl" in cb_kwargs:
            level = cb_kwargs.pop("lvl")
        # Only forward user-facing or control messages to the GUI queue
        lvl_str = str(level).upper() if isinstance(level, str) else "INFO"
        if lvl_str not in {"INFO", "WARN", "ERROR", "SUCCESS", "ETA_LEVEL", "CHRONO_LEVEL"}:
            return
        progress_queue.put((message_key_or_raw, progress_value, level, cb_kwargs))

    # Insert the process queue callback in the expected position (after
    # cluster threshold, target group count, and orientation split parameter).
    # With the current signature, progress_callback is the 11th positional arg.
    if len(args) > 10:
        candidate = args[10]
        if callable(candidate):
            # Replace the provided callback without disturbing other
            # positional arguments.
            full_args = args[:10] + (queue_callback,) + args[11:]
        else:
            # No callback was supplied: insert ours in the expected slot so
            # that subsequent parameters keep their intended positions.
            full_args = args[:10] + (queue_callback,) + args[10:]
    else:
        # Safety fallback: if the caller did not provide enough positional
        # arguments to reach the callback slot, append ours so the worker
        # still runs (mainly for CLI/debug scenarios).
        full_args = args + (queue_callback,)
    try:
        run_hierarchical_mosaic(*full_args, solver_settings=solver_settings_dict, **kwargs)
    except Exception as e_proc:
        progress_queue.put(("PROCESS_ERROR", None, "ERROR", {"error": str(e_proc)}))
    finally:
        progress_queue.put(("PROCESS_DONE", None, "INFO", {}))

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="ZeMosaic worker")
    parser.add_argument("input_folder", help="Folder with input FITS")
    parser.add_argument("output_folder", help="Destination folder")
    parser.add_argument("--config", default=None, help="Optional config JSON")
    parser.add_argument("--coadd_use_memmap", action="store_true",
                        help="Write sum/cov arrays to disk via numpy.memmap")
    parser.add_argument("--coadd_memmap_dir", default=None,
                        help="Directory to store *.dat blocks")
    parser.add_argument("--coadd_cleanup_memmap", action="store_true",
                        default=True,
                        help="Delete *.dat blocks when the run finishes")
    parser.add_argument("--no_auto_limit_frames", action="store_true",
                        help="Disable automatic frame limit per master tile")
    parser.add_argument("--assembly_process_workers", type=int, default=None,
                        help="Number of processes for final assembly (0=auto)")
    parser.add_argument("-W", "--winsor-workers", type=int, default=None,
                        help="Process workers for Winsorized rejection (1-16)")
    parser.add_argument("--max-raw-per-master-tile", type=int, default=None,
                        help="Cap raw frames per master tile (0=auto)")
    parser.add_argument("--solver-settings", default=None,
                        help="Path to solver settings JSON")
    args = parser.parse_args()

    cfg = {}
    if ZEMOSAIC_CONFIG_AVAILABLE and zemosaic_config:
        cfg.update(zemosaic_config.load_config())
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass

    solver_cfg = {}
    if args.solver_settings:
        try:
            solver_cfg = SolverSettings.load(args.solver_settings).__dict__
        except Exception:
            solver_cfg = {}
    else:
        try:
            solver_cfg = SolverSettings.load_default().__dict__
        except Exception:
            solver_cfg = SolverSettings().__dict__

  
