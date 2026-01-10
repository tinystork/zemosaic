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



# Initial Qt filter dialog for ZeMosaic.
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import datetime
import logging
import importlib.util
import os
from os import path as ospath
from pathlib import Path
import math
import csv
import threading
import time
import json
import copy
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import numpy as np

from core.path_helpers import casefold_path


_pyside_spec = importlib.util.find_spec("PySide6")
if _pyside_spec is None:  # pragma: no cover - import guard
    raise ImportError(
        "PySide6 is required to use the ZeMosaic Qt filter interface. "
        "Install PySide6 or use the Tk interface instead."
    )

from PySide6.QtCore import (
    QObject,
    Qt,
    QThread,
    Signal,
    Slot,
    QTimer,
    QRect,
    QByteArray,
    QPoint,
)
from PySide6.QtGui import QIcon, QMovie
from PySide6.QtWidgets import (  # noqa: E402  - imported after availability check
    QApplication,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
    QVBoxLayout,
    QPlainTextEdit,
    QCheckBox,
    QFileDialog,
    QAbstractItemView,
    QTabWidget,
    QHeaderView,
    QMenu,
)


def _reset_filter_log() -> None:
    """Supprime le log du filtre au chargement pour repartir d'un fichier propre."""

    try:
        log_path = Path(__file__).with_name("zemosaic_filter.log")
        if log_path.exists():
            log_path.unlink()
    except Exception:
        # On ignore toute erreur : le démarrage de l'UI ne doit pas échouer pour un log.
        pass


_reset_filter_log()


def _ensure_filter_file_logger() -> None:
    """Ensure a file handler is attached to the root logger for filter logs.

    Key points:
    - Avoid duplicate FileHandlers pointing to the same file.
    - Make sure INFO/DEBUG actually reach the file (set levels).
    - Do not crash UI startup if logging setup fails.
    """
    try:
        log_path = Path(__file__).with_name("zemosaic_filter.log")
        target_path = log_path.resolve()

        root_logger = logging.getLogger()  # root

        # If a FileHandler already targets this file, we're done.
        for handler in list(root_logger.handlers):
            if not isinstance(handler, logging.FileHandler):
                continue
            try:
                handler_path = Path(getattr(handler, "baseFilename", "")).resolve()
            except Exception:
                continue
            if handler_path == target_path:
                return

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")

        # Ensure messages aren't filtered out by handler/root default levels.
        file_handler.setLevel(logging.DEBUG)
        if root_logger.level in (logging.NOTSET, 0) or root_logger.level > logging.DEBUG:
            root_logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler.setFormatter(formatter)

        root_logger.addHandler(file_handler)
        root_logger.info("Filter log file handler enabled: %s", str(log_path))
    except Exception:
        # UI startup must not fail if file logging fails
        pass


_ensure_filter_file_logger()


def _resolve_tristate_flag() -> Qt.ItemFlag:
    candidates = ("ItemIsTristate", "ItemIsAutoTristate", "ItemIsUserTristate")
    for name in candidates:
        flag = getattr(Qt, name, None)
        if flag is not None:
            return flag
        item_flag = getattr(Qt, "ItemFlag", None)
        if item_flag is not None:
            derived = getattr(item_flag, name, None)
            if derived is not None:
                return derived
    raise AttributeError("Qt is missing all tristate item flags")


_QT_TRISTATE_FLAG = _resolve_tristate_flag()


FITS_EXTENSIONS = {".fit", ".fits"}


def _expand_to_path(value: Any) -> Optional[Path]:
    """Expand environment variables and ``~`` in ``value`` and return a Path."""

    if value is None:
        return None
    if isinstance(value, Path):
        return value
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text:
        return None
    try:
        expanded = ospath.expandvars(ospath.expanduser(text))
    except Exception:
        expanded = text
    try:
        return Path(expanded)
    except Exception:
        return None


def _path_display_name(path: Any, *, default: str = "") -> str:
    """Return a filesystem-friendly display name."""

    candidate = _expand_to_path(path)
    if candidate:
        name = candidate.name
        return name or str(candidate)
    return str(path or default or "<unknown>")


def _path_is_file(value: Any) -> bool:
    path = _expand_to_path(value)
    return bool(path and path.is_file())


def _path_is_dir(value: Any) -> bool:
    path = _expand_to_path(value)
    return bool(path and path.is_dir())


def _load_zemosaic_qicon() -> QIcon | None:
    try:
        icon_dir = get_app_base_dir() / "icon"
    except Exception:
        return None

    candidates = [
        icon_dir / "zemosaic.ico",
        icon_dir / "zemosaic_64x64.png",
        icon_dir / "zemosaic_icon.png",
        icon_dir / "zemosaic.png",
    ]

    for path in candidates:
        try:
            if not path.is_file():
                continue
            icon = QIcon(str(path))
            if not icon.isNull():
                return icon
        except Exception:
            continue

    print(f"[QtFilter] Aucune icône ZeMosaic trouvée dans {icon_dir}")
    return None


PREVIEW_REFRESH_INTERVAL_SEC = 0.15
PREVIEW_DRAW_THROTTLE_SEC = 0.30
PREVIEW_HARD_LIMIT = 1500
PREVIEW_LEGEND_MAX_GROUPS = 30

if importlib.util.find_spec("locales.zemosaic_localization") is not None:
    from locales.zemosaic_localization import ZeMosaicLocalization  # type: ignore
else:  # pragma: no cover - optional dependency guard
    ZeMosaicLocalization = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for metadata extraction
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
except Exception:  # pragma: no cover - optional dependency guard
    fits = None  # type: ignore[assignment]
    WCS = None  # type: ignore[assignment]
    SkyCoord = None  # type: ignore[assignment]
    u = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import RectangleSelector
    # Monkey-patch matplotlib's _draw_idle to handle deleted canvases without error
    if FigureCanvasQTAgg is not None:
        original_draw_idle = FigureCanvasQTAgg._draw_idle
        def patched_draw_idle(self):
            try:
                return original_draw_idle(self)
            except RuntimeError as e:
                if "already deleted" in str(e):
                    return  # Silently ignore draw calls on deleted canvases
                raise
        FigureCanvasQTAgg._draw_idle = patched_draw_idle
except Exception:  # pragma: no matplotlib optional
    FigureCanvasQTAgg = None  # type: ignore[assignment]
    LineCollection = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]
    Rectangle = None  # type: ignore[assignment]
    RectangleSelector = None  # type: ignore[assignment]
    to_rgba = None  # type: ignore[assignment]
    Line2D = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    from zemosaic_astrometry import (
        compute_astap_recommended_max_instances,
        set_astap_max_concurrent_instances,
        solve_with_astap,
    )
except Exception:  # pragma: no cover - optional dependency guard
    compute_astap_recommended_max_instances = None  # type: ignore[assignment]
    solve_with_astap = None  # type: ignore[assignment]
    set_astap_max_concurrent_instances = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    import zemosaic_worker as _zemosaic_worker  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    _zemosaic_worker = None  # type: ignore[assignment]

if _zemosaic_worker is not None:
    _CLUSTER_CONNECTED = getattr(_zemosaic_worker, "cluster_seestar_stacks_connected", None)
    _AUTOSPLIT_GROUPS = getattr(_zemosaic_worker, "_auto_split_groups", None)
    _COMPUTE_MAX_SEPARATION = getattr(_zemosaic_worker, "_compute_max_angular_separation_deg", None)
else:  # pragma: no cover - helper fallback
    _CLUSTER_CONNECTED = None
    _AUTOSPLIT_GROUPS = None
    _COMPUTE_MAX_SEPARATION = None

try:  # pragma: no cover - optional dependency guard
    from zemosaic_filter_gui import (  # type: ignore
        _merge_small_groups as _tk_merge_small_groups,
        _split_group_by_orientation as _tk_split_group_by_orientation,
        _circular_dispersion_deg as _tk_circular_dispersion_deg,
    )
except Exception:  # pragma: no cover - helper fallback
    _tk_merge_small_groups = None  # type: ignore[assignment]
    _tk_split_group_by_orientation = None  # type: ignore[assignment]
    _tk_circular_dispersion_deg = None  # type: ignore[assignment]

if _tk_circular_dispersion_deg is None:  # pragma: no cover - fallback copy
    def _tk_circular_dispersion_deg(values: Iterable[float]) -> float:
        values_list = [float(v) for v in values if v is not None]
        if not values_list:
            return 0.0
        mean_angle = math.atan2(
            sum(math.sin(math.radians(v)) for v in values_list),
            sum(math.cos(math.radians(v)) for v in values_list),
        )
        deviations = [
            math.degrees(math.acos(math.cos(math.radians(v) - mean_angle))) for v in values_list
        ]
        return max(deviations) if deviations else 0.0

if _tk_split_group_by_orientation is None:  # pragma: no cover - fallback copy
    def _tk_split_group_by_orientation(group: list[dict], threshold_deg: float) -> list[list[dict]]:
        if threshold_deg <= 0.0 or not group:
            return [group]
        buckets: list[list[dict]] = []
        for entry in group:
            pa = entry.get("PA_DEG")
            try:
                pa_value = float(pa)
            except Exception:
                pa_value = None
            if pa_value is None:
                buckets.append([entry])
                continue
            matched = False
            for bucket in buckets:
                ref = bucket[0].get("PA_DEG")
                try:
                    ref_value = float(ref)
                except Exception:
                    ref_value = None
                if ref_value is None:
                    continue
                delta = abs(pa_value - ref_value) % 360.0
                delta = min(delta, 360.0 - delta)
                if delta <= threshold_deg:
                    bucket.append(entry)
                    matched = True
                    break
            if not matched:
                buckets.append([entry])
        return buckets

def _split_group_by_orientation_key(
    group: list[dict[str, Any]],
    threshold_deg: float,
    pa_key: str,
) -> list[list[dict[str, Any]]]:
    if threshold_deg <= 0.0 or not group:
        return [group]
    buckets: list[list[dict[str, Any]]] = []
    for entry in group:
        pa = entry.get(pa_key)
        try:
            pa_value = float(pa)
        except Exception:
            pa_value = None
        if pa_value is None:
            buckets.append([entry])
            continue
        matched = False
        for bucket in buckets:
            if not bucket:
                continue
            ref = bucket[0].get(pa_key)
            try:
                ref_value = float(ref)
            except Exception:
                ref_value = None
            if ref_value is None:
                continue
            delta = abs(pa_value - ref_value) % 360.0
            delta = min(delta, 360.0 - delta)
            if delta <= threshold_deg:
                bucket.append(entry)
                matched = True
                break
        if not matched:
            buckets.append([entry])
    return buckets


def split_clusters_by_orientation(
    groups: list[list[dict[str, Any]]],
    orientation_split_deg: float,
    pa_key: str = "PA_DEG",
) -> list[list[dict[str, Any]]]:
    threshold = _sanitize_angle_value(orientation_split_deg, 0.0)
    if threshold <= 0.0 or not groups:
        return groups
    result: list[list[dict[str, Any]]] = []
    for group in groups:
        if not group:
            result.append(group)
            continue
        has_valid_pa = False
        for entry in group:
            pa = entry.get(pa_key)
            if pa is None:
                continue
            try:
                float(pa)
            except Exception:
                continue
            has_valid_pa = True
            break
        if not has_valid_pa:
            result.append(group)
            continue
        if pa_key == "PA_DEG" and _tk_split_group_by_orientation is not None:
            try:
                subgroups = _tk_split_group_by_orientation(group, threshold)
            except Exception:
                subgroups = [group]
        else:
            subgroups = _split_group_by_orientation_key(group, threshold, pa_key)
        if subgroups:
            result.extend(subgroups)
        else:
            result.append(group)
    return result

if _tk_merge_small_groups is None:  # pragma: no cover - fallback copy
    def _tk_merge_small_groups(
        groups: list[list[dict]],
        min_size: int,
        cap: int,
        *,
        cap_allowance: int | None = None,
        compute_dispersion: Callable[[list[tuple[float, float]]], float] | None = None,
        max_dispersion_deg: float | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> list[list[dict]]:
        if not groups or min_size <= 0 or cap <= 0:
            return groups
        merged = [False] * len(groups)
        centers: list[tuple[float, float] | None] = []
        for group in groups:
            coords = []
            for entry in group:
                ra = entry.get("RA")
                dec = entry.get("DEC")
                if ra is None or dec is None:
                    continue
                try:
                    coords.append((float(ra), float(dec)))
                except Exception:
                    continue
            if coords:
                avg_ra = sum(pt[0] for pt in coords) / len(coords)
                avg_dec = sum(pt[1] for pt in coords) / len(coords)
                centers.append((avg_ra, avg_dec))
            else:
                centers.append(None)

        allowance = cap_allowance if cap_allowance and cap_allowance > 0 else cap
        allowance = max(allowance, cap)

        def _distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
            if a is None or b is None:
                return float("inf")
            dra = abs(a[0] - b[0])
            ddec = abs(a[1] - b[1])
            return math.hypot(dra, ddec)

        for idx, group in enumerate(groups):
            if merged[idx] or len(group) >= min_size:
                continue
            nearest = None
            nearest_dist = float("inf")
            for other_idx, other in enumerate(groups):
                if idx == other_idx or merged[other_idx]:
                    continue
                dist = _distance(centers[idx], centers[other_idx])
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = other_idx
            if nearest is None:
                continue
            if len(groups[nearest]) + len(group) > allowance:
                continue
            if compute_dispersion is not None and max_dispersion_deg is not None:
                coords = []
                for entry in groups[nearest] + group:
                    ra = entry.get("RA")
                    dec = entry.get("DEC")
                    if ra is None or dec is None:
                        continue
                    try:
                        coords.append((float(ra), float(dec)))
                    except Exception:
                        continue
                try:
                    dispersion = compute_dispersion(coords)
                except Exception:
                    dispersion = None
                if dispersion is not None and dispersion > max_dispersion_deg:
                    continue
            groups[nearest].extend(group)
            merged[idx] = True
            if log_fn:
                log_fn(f"[AutoMerge] Group {idx} merged into {nearest}")
        return [group for idx, group in enumerate(groups) if not merged[idx]]


class _FallbackWCS:
    """Minimal WCS-like object built from a SkyCoord center."""

    is_celestial = True

    def __init__(self, center_coord: SkyCoord) -> None:
        self._center = center_coord
        self.pixel_shape = (1, 1)
        self.array_shape = (1, 1)

        class _Inner:
            def __init__(self, center: SkyCoord) -> None:
                self.crval = (
                    float(center.ra.to(u.deg).value),
                    float(center.dec.to(u.deg).value),
                )
                self.crpix = (0.5, 0.5)

        self.wcs = _Inner(center_coord)

    def pixel_to_world(self, _x: float, _y: float) -> SkyCoord:
        return self._center


def _classify_mount_mode_from_header(header: Any) -> str:
    """Return 'EQ', 'ALT_AZ', 'EQMODE_<N>', or 'UNKNOWN' from header['EQMODE']."""  # noqa: E501

    if not header:
        return "UNKNOWN"
    try:
        raw_value = header.get("EQMODE")  # type: ignore[attr-defined]
    except Exception:
        raw_value = None
    try:
        value_int = int(raw_value)
    except Exception:
        try:
            value_int = int(float(str(raw_value).strip()))
        except Exception:
            return "UNKNOWN"
    if value_int == 1:
        return "EQ"
    if value_int == 0:
        return "ALT_AZ"
    return f"EQMODE_{value_int}"


def _split_group_by_mount_mode(group: list[dict]) -> list[list[dict]]:
    """Split group into EQ / ALT_AZ buckets; UNKNOWN entries follow the majority."""

    if not group:
        return [group]

    def _normalize_mode(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return str(value)
        except Exception:
            return "UNKNOWN"

    modes = [_normalize_mode(entry.get("MOUNT_MODE", "UNKNOWN")) for entry in group]

    def _is_eq(mode: str) -> bool:
        return mode == "EQ" or mode == "EQMODE_1"

    def _is_altaz(mode: str) -> bool:
        return mode == "ALT_AZ"

    eq_count = sum(1 for mode in modes if _is_eq(mode))
    altaz_count = sum(1 for mode in modes if _is_altaz(mode))

    if eq_count == 0 and altaz_count == 0:
        return [group]
    if eq_count == 0 or altaz_count == 0:
        return [group]

    majority = "EQ" if eq_count >= altaz_count else "ALT_AZ"
    eq_entries: list[dict] = []
    altaz_entries: list[dict] = []
    for entry, mode in zip(group, modes):
        target = None
        if _is_eq(mode):
            target = "EQ"
        elif _is_altaz(mode):
            target = "ALT_AZ"
        else:
            target = majority
        if target == "EQ":
            eq_entries.append(entry)
        elif target == "ALT_AZ":
            altaz_entries.append(entry)

    return [eq_entries, altaz_entries]


ANGLE_SPLIT_DEFAULT_DEG = 5.0
AUTO_ANGLE_DETECT_DEFAULT_DEG = 10.0


def _sanitize_angle_value(value: Any, default: float) -> float:
    try:
        val = float(value)
    except Exception:
        return default
    if math.isnan(val) or math.isinf(val):
        return default
    if val < 0.0:
            return 0.0
    if val > 180.0:
        return 180.0
    return val


def _apply_borrowing_per_mount_mode(
    final_groups: list[list[dict]],
    logger: logging.Logger,
) -> tuple[list[list[dict]], dict[str, Any]]:
    """Apply borrowing per mount-mode bucket to avoid EQ/ALT_AZ mixing."""

    groups = final_groups or []
    if not isinstance(groups, list):
        groups = []
    groups = list(groups)

    def _empty_stats() -> dict[str, Any]:
        return {
            "executed": False,
            "borrowed_total_assignments": 0,
            "borrowed_unique_images": 0,
            "borrow_attempts_total": 0,
            "borrow_success_total": 0,
            "border_candidate_images_total": 0,
            "per_group": [],
            "examples": [],
            "valid_image_centers": 0,
            "valid_group_centers": 0,
        }

    borrow_func = apply_borrowing_v1
    if borrow_func is None:
        return groups, _empty_stats()

    if len(groups) < 2:
        return borrow_func(groups, None, logger=logger)

    def _normalize_mode(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return str(value)
        except Exception:
            return "UNKNOWN"

    def _group_mode(group: list[dict]) -> str:
        eq_count = 0
        altaz_count = 0
        for entry in group or []:
            try:
                mode_raw = entry.get("MOUNT_MODE", "UNKNOWN")
            except Exception:
                mode_raw = "UNKNOWN"
            mode_norm = _normalize_mode(mode_raw)
            if mode_norm == "EQ" or mode_norm == "EQMODE_1":
                eq_count += 1
            elif mode_norm == "ALT_AZ":
                altaz_count += 1
        if eq_count == 0 and altaz_count == 0:
            return "UNKNOWN"
        if eq_count > 0 and altaz_count == 0:
            return "EQ"
        if altaz_count > 0 and eq_count == 0:
            return "ALT_AZ"
        return "EQ" if eq_count >= altaz_count else "ALT_AZ"

    modes_by_group = [_group_mode(group) for group in groups]
    has_eq_mode = any(mode == "EQ" for mode in modes_by_group)
    has_altaz_mode = any(mode == "ALT_AZ" for mode in modes_by_group)
    if not (has_eq_mode and has_altaz_mode):
        return borrow_func(groups, None, logger=logger)

    buckets: dict[str, list[list[dict]]] = {
        "EQ": [],
        "ALT_AZ": [],
        "UNKNOWN": [],
    }
    for group, mode in zip(groups, modes_by_group):
        if mode == "EQ":
            buckets["EQ"].append(group)
        elif mode == "ALT_AZ":
            buckets["ALT_AZ"].append(group)
        else:
            buckets["UNKNOWN"].append(group)

    def _merge_stats(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        if not incoming:
            return base
        if not base:
            return copy.deepcopy(incoming)
        for key, value in incoming.items():
            if key not in base:
                base[key] = copy.deepcopy(value)
                continue
            existing = base.get(key)
            if key == "executed":
                base[key] = bool(existing) or bool(value)
            elif isinstance(existing, (int, float)) and isinstance(value, (int, float)):
                base[key] = existing + value
            elif isinstance(existing, list) and isinstance(value, list):
                base[key].extend(value)
            else:
                base[key] = value
        return base

    aggregated_stats: dict[str, Any] = {}
    combined_groups: list[list[dict]] = []
    for label in ("EQ", "ALT_AZ", "UNKNOWN"):
        bucket_groups = buckets.get(label) or []
        if not bucket_groups:
            continue
        bucket_result, bucket_stats = borrow_func(bucket_groups, None, logger=logger)
        combined_groups.extend(bucket_result or [])
        aggregated_stats = _merge_stats(aggregated_stats, bucket_stats or {})

    if "executed" in aggregated_stats:
        aggregated_stats["executed"] = bool(aggregated_stats.get("executed"))
    else:
        aggregated_stats["executed"] = False

    mixed_groups = 0
    for group in combined_groups:
        has_eq = False
        has_altaz = False
        for entry in group or []:
            mode_value = _normalize_mode(entry.get("MOUNT_MODE", "UNKNOWN"))
            if mode_value == "EQ" or mode_value == "EQMODE_1":
                has_eq = True
            elif mode_value == "ALT_AZ":
                has_altaz = True
        if has_eq and has_altaz:
            mixed_groups += 1
    if mixed_groups > 0:
        logger.warning(
            "Mount-mode guard violation: %d group(s) contain EQ and ALT_AZ after borrowing.",
            mixed_groups,
        )

    return combined_groups, aggregated_stats

try:  # pragma: no cover - optional dependency guard
    from zemosaic_config import DEFAULT_CONFIG as _DEFAULT_GUI_CONFIG  # type: ignore
    from zemosaic_config import load_config as _load_gui_config  # type: ignore
    from zemosaic_config import save_config as _save_gui_config  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    _DEFAULT_GUI_CONFIG = {}
    _load_gui_config = None  # type: ignore[assignment]
    _save_gui_config = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    from zemosaic_utils import (  # type: ignore
        EXCLUDED_DIRS,
        apply_borrowing_v1,
        get_app_base_dir,
        get_user_config_dir,
        ensure_user_config_dir,
        compute_global_wcs_descriptor,
        is_path_excluded,
        load_global_wcs_descriptor,
        parse_global_wcs_resolution_override,
        resolve_global_wcs_output_paths,
        write_global_wcs_files,
    )
except Exception:  # pragma: no cover - optional dependency guard
    EXCLUDED_DIRS = frozenset({"unaligned_by_zemosaic"})  # type: ignore[assignment]
    def get_app_base_dir() -> Path:  # type: ignore
        return Path(__file__).resolve().parent
    def get_user_config_dir() -> Path:  # type: ignore
        return Path.home() / "ZeMosaic"
    def ensure_user_config_dir() -> Path:  # type: ignore
        path = get_user_config_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path
    compute_global_wcs_descriptor = None  # type: ignore[assignment]
    is_path_excluded = None  # type: ignore[assignment]
    load_global_wcs_descriptor = None  # type: ignore[assignment]
    parse_global_wcs_resolution_override = None  # type: ignore[assignment]
    resolve_global_wcs_output_paths = None  # type: ignore[assignment]
    write_global_wcs_files = None  # type: ignore[assignment]
    apply_borrowing_v1 = None  # type: ignore[assignment]


try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


logger = logging.getLogger(__name__)

QT_FILTER_WINDOW_GEOMETRY_KEY = "qt_filter_window_geometry"

_DEFAULT_GUI_CONFIG_MAP: dict[str, Any] = {}
if isinstance(_DEFAULT_GUI_CONFIG, dict):
    try:
        _DEFAULT_GUI_CONFIG_MAP.update(_DEFAULT_GUI_CONFIG)
    except Exception:
        _DEFAULT_GUI_CONFIG_MAP = {}

DEFAULT_FILTER_CONFIG: dict[str, Any] = dict(_DEFAULT_GUI_CONFIG_MAP)
DEFAULT_FILTER_CONFIG.setdefault("auto_detect_seestar", True)
DEFAULT_FILTER_CONFIG.setdefault("force_seestar_mode", False)
DEFAULT_FILTER_CONFIG.setdefault("sds_mode_default", False)
DEFAULT_FILTER_CONFIG.setdefault("sds_min_batch_size", 5)
DEFAULT_FILTER_CONFIG.setdefault("sds_target_batch_size", 10)
DEFAULT_FILTER_CONFIG.setdefault("batch_overlap_pct", 40)
DEFAULT_FILTER_CONFIG.setdefault("allow_batch_duplication", True)
DEFAULT_FILTER_CONFIG.setdefault("min_safe_stack", 3)
DEFAULT_FILTER_CONFIG.setdefault("target_stack", 5)
DEFAULT_FILTER_CONFIG.setdefault("cluster_panel_threshold", 0.0)
DEFAULT_FILTER_CONFIG.setdefault("max_raw_per_master_tile", 50)
DEFAULT_FILTER_CONFIG.setdefault("global_coadd_method", "kappa_sigma")
DEFAULT_FILTER_CONFIG.setdefault("global_coadd_k", 2.0)
DEFAULT_FILTER_CONFIG.setdefault(QT_FILTER_WINDOW_GEOMETRY_KEY, None)


class _FallbackLocalizer:
    """Very small localization shim when the full helper is unavailable."""

    def __init__(self, language_code: str = "en") -> None:
        self.language_code = language_code

    def get(self, key: str, default_text: str | None = None, **_: Any) -> str:
        return default_text if default_text is not None else key

    def set_language(self, language_code: str) -> None:
        self.language_code = language_code


@dataclass(slots=True)
class _NormalizedItem:
    """Container describing a row displayed by the Qt dialog."""

    original: Any
    display_name: str
    file_path: str
    has_wcs: bool
    instrument: str | None
    group_label: str | None
    include_by_default: bool = True
    center_ra_deg: float | None = None
    center_dec_deg: float | None = None
    cluster_index: int | None = None
    footprint_radec: List[Tuple[float, float]] | None = None
    header_cache: Any | None = None
    wcs_cache: Any | None = None


_GroupKey = tuple[int | None, str | None]
_GroupOutline = tuple[int | None, float, float, float, float]


def _sanitize_footprint_radec(payload: Any) -> List[Tuple[float, float]] | None:
    """Return a normalised list of (RA, Dec) tuples for a footprint.

    The Tk filter can emit several shapes for footprint metadata (list of
    tuples, list of dicts with 'RA'/'DEC' or 'ra'/'dec', or a mapping with a
    ``corners``/``footprint`` key).  This helper mirrors that flexibility so
    the Qt dialog can reuse pre-computed footprints when available.
    """

    if payload is None:
        return None

    # Unwrap common container keys used by the Tk filter / worker.
    if isinstance(payload, dict):
        for key in ("footprint_radec", "corners", "footprint"):
            if key in payload:
                payload = payload.get(key)
                break

    if not isinstance(payload, (list, tuple)):
        return None

    points: list[Tuple[float, float]] = []
    for entry in payload:
        ra_val: Any = None
        dec_val: Any = None
        if isinstance(entry, dict):
            ra_val = entry.get("ra")
            if ra_val is None:
                ra_val = entry.get("RA")
            dec_val = entry.get("dec")
            if dec_val is None:
                dec_val = entry.get("DEC")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            ra_val, dec_val = entry[0], entry[1]
        else:
            continue

        try:
            ra = float(ra_val)
            dec = float(dec_val)
        except Exception:
            continue
        points.append((ra, dec))

    return points or None


_HEADER_CACHE_KEYS: tuple[str, ...] = (
    "NAXIS1",
    "NAXIS2",
    "EQMODE",
    "CRVAL1",
    "CRVAL2",
    "CRPIX1",
    "CRPIX2",
    "CTYPE1",
    "CTYPE2",
    "CDELT1",
    "CDELT2",
    "CD1_1",
    "CD1_2",
    "CD2_1",
    "CD2_2",
    "PC1_1",
    "PC1_2",
    "PC2_1",
    "PC2_2",
    "DATE-OBS",
    "EXPTIME",
    "FILTER",
    "OBJECT",
    "CREATOR",
    "INSTRUME",
)


def _coerce_header_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if np is not None:
        try:
            if isinstance(value, np.generic):
                return value.item()
        except Exception:
            pass
    try:
        return value.item()  # type: ignore[call-arg]
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        pass
    try:
        return str(value)
    except Exception:
        return None


def _sanitize_header_subset(header_obj: Any) -> dict[str, Any] | None:
    """Extract a lightweight header mapping similar to the Tk filter."""

    if header_obj is None:
        return None

    def _lookup(key: str) -> Any:
        if isinstance(header_obj, dict):
            return header_obj.get(key)
        try:
            return header_obj[key]
        except Exception:
            try:
                return header_obj.get(key)  # type: ignore[call-arg]
            except Exception:
                return None

    subset: dict[str, Any] = {}
    for key in _HEADER_CACHE_KEYS:
        value = _coerce_header_scalar(_lookup(key))
        if value is None:
            continue
        subset[key] = value
    if "NAXIS1" in subset and "NAXIS2" in subset:
        try:
            subset["shape"] = (int(subset["NAXIS2"]), int(subset["NAXIS1"]))
        except Exception:
            pass
    return subset or None


def _format_sizes_histogram(sizes: list[int], max_buckets: int = 6) -> str:
    """Return a compact histogram string for group sizes (Tk parity)."""

    if not sizes:
        return "[]"

    counter = Counter(sizes)
    pairs = sorted(counter.items(), key=lambda kv: (-kv[1], -kv[0]))
    head = ", ".join(f"{size}×{count}" for size, count in pairs[:max_buckets])
    tail = len(pairs) - max_buckets
    return head + (f", +{tail} more" if tail > 0 else "")


def _iter_normalized_entries(
    payload: Any,
    initial_overrides: Any,
    *,
    scan_recursive: bool,
) -> Iterator[_NormalizedItem]:
    """Yield ``_NormalizedItem`` instances without eagerly materialising them."""

    excluded_paths: Sequence[str] = ()
    if isinstance(initial_overrides, dict):
        candidate = initial_overrides.get("excluded_paths")
        if isinstance(candidate, (list, tuple, set)):
            excluded_paths = tuple(casefold_path(p) for p in candidate)

    def _should_exclude(path: Path | str) -> bool:
        try:
            path_obj = Path(path)
        except Exception:
            path_obj = Path(str(path))

        # Honour explicit per-path exclusions from overrides first.
        norm = casefold_path(path_obj)
        for entry in excluded_paths:
            if norm == entry:
                return True

        # Then apply global directory-level exclusions for stream-scan parity.
        try:
            if is_path_excluded is not None and EXCLUDED_DIRS:
                if is_path_excluded(path_obj, EXCLUDED_DIRS):
                    return True
        except Exception:
            pass

        return False

    def _build_from_mapping(obj: dict) -> _NormalizedItem:
        original = obj
        instrument = None
        for key in ("instrument", "INSTRUME", "instrument_name"):
            if key in obj and obj[key]:
                try:
                    instrument = str(obj[key])
                except Exception:
                    instrument = None
                if instrument:
                    break
        group_label: str | None = None
        for key in (
            "group_label",
            "group",
            "group_id",
            "cluster",
            "cluster_id",
            "tile_group",
            "tile_group_id",
            "master_tile_id",
        ):
            if key in obj and obj[key] not in (None, ""):
                try:
                    group_label = str(obj[key])
                except Exception:
                    group_label = None
                if group_label:
                    break
        if group_label is None and "grouping" in obj:
            grouping = obj.get("grouping")
            if isinstance(grouping, dict):
                for key in ("label", "id", "name"):
                    candidate = grouping.get(key)
                    if candidate not in (None, ""):
                        try:
                            group_label = str(candidate)
                        except Exception:
                            group_label = None
                        if group_label:
                            break
        file_path_raw = (
            obj.get("file_path")
            or obj.get("path")
            or obj.get("filepath")
            or obj.get("filename")
            or obj.get("full_path")
            or obj.get("fullpath")
            or obj.get("name")
            or obj.get("src")
            or ""
        )
        file_path_path = _expand_to_path(file_path_raw)
        file_path = str(file_path_path) if file_path_path else str(file_path_raw or "")
        if not file_path and "header" in obj:
            try:
                file_path = str(obj["header"].get("FILENAME", ""))
            except Exception:
                file_path = ""
        has_wcs = False
        for key in ("has_wcs", "wcs_solved", "has_valid_wcs"):
            if obj.get(key):
                has_wcs = True
                break
        if not has_wcs and "wcs" in obj:
            has_wcs = obj["wcs"] is not None
        display_name = _path_display_name(file_path, default=instrument or "Item")
        include = not _should_exclude(file_path)
        ra_deg = None
        dec_deg = None
        for key in ("center_ra_deg", "ra_deg", "ra"):
            if key in obj:
                try:
                    ra_deg = float(obj[key])
                except Exception:
                    ra_deg = None
                if ra_deg is not None:
                    break
        for key in ("center_dec_deg", "dec_deg", "dec"):
            if key in obj:
                try:
                    dec_deg = float(obj[key])
                except Exception:
                    dec_deg = None
                if dec_deg is not None:
                    break
        if (ra_deg is None or dec_deg is None) and "header" in obj:
            try:
                header_obj = obj["header"]
            except Exception:
                header_obj = None
            if header_obj is not None:
                ra_from_header, dec_from_header = _extract_center_from_header(header_obj)
                ra_deg = ra_deg if ra_deg is not None else ra_from_header
                dec_deg = dec_deg if dec_deg is not None else dec_from_header
        header_cache = obj.get("header") or obj.get("header_subset")
        header_cache = _sanitize_header_subset(header_cache)
        footprint = None
        for key in ("footprint_radec", "footprint", "corners"):
            if key in obj:
                footprint = _sanitize_footprint_radec(obj.get(key))
                if footprint:
                    break
        return _NormalizedItem(
            original=original,
            display_name=display_name,
            file_path=file_path,
            has_wcs=has_wcs,
            instrument=instrument,
            group_label=group_label,
            include_by_default=include,
            center_ra_deg=ra_deg,
            center_dec_deg=dec_deg,
            footprint_radec=footprint,
            header_cache=header_cache,
            wcs_cache=obj.get("wcs"),
        )

    def _is_supported_fits(path_obj: Path) -> bool:
        try:
            suffix = path_obj.suffix.lower()
        except Exception:
            suffix = ""
        return suffix in FITS_EXTENSIONS

    def _build_from_path(path_obj: Path) -> _NormalizedItem | None:
        if not _is_supported_fits(path_obj):
            return None
        file_path = str(path_obj)
        display_name = path_obj.name or file_path
        include = not _should_exclude(path_obj)
        original_payload: dict[str, Any] = {
            "path": file_path,
            "path_raw": file_path,
            "include_by_default": include,
        }
        return _NormalizedItem(
            original=original_payload,
            display_name=display_name,
            file_path=file_path,
            has_wcs=False,
            instrument=None,
            group_label=None,
            include_by_default=include,
        )

    if isinstance(payload, (str, os.PathLike)):
        directory = Path(payload)
        if directory.is_dir():
            if _should_exclude(directory):
                return
            pattern = "**/*" if scan_recursive else "*"
            for candidate in directory.glob(pattern):
                if candidate.is_file():
                    if not _is_supported_fits(candidate):
                        continue
                    if _should_exclude(candidate):
                        continue
                    entry = _build_from_path(candidate)
                    if entry is not None:
                        yield entry
        elif Path(payload).is_file():
            candidate = Path(payload)
            if not _should_exclude(candidate):
                entry = _build_from_path(candidate)
                if entry is not None:
                    yield entry
    elif isinstance(payload, Iterable):
        for element in payload:
            if isinstance(element, dict):
                yield _build_from_mapping(element)
            elif isinstance(element, (str, os.PathLike)):
                path_obj = Path(element)
                entry = _build_from_path(path_obj)
                if entry is not None:
                    yield entry
            else:
                yield _NormalizedItem(
                    original=element,
                    display_name=str(element),
                    file_path=str(element),
                    has_wcs=False,
                    instrument=None,
                    group_label=None,
                    include_by_default=True,
                )
    else:
        yield _NormalizedItem(
            original=payload,
            display_name=str(payload),
            file_path=str(payload),
            has_wcs=False,
            instrument=None,
            group_label=None,
            include_by_default=True,
        )


SIP_COEFF_PREFIXES = ("A_", "B_", "AP_", "BP_")
SIP_META_KEYS = {"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"}


def _header_contains_sip_terms(header_obj: Any) -> bool:
    """Return True when the header exposes SIP distortion keywords."""

    if header_obj is None:
        return False
    try:
        keys_iter = list(header_obj.keys())  # type: ignore[arg-type]
    except Exception:
        try:
            keys_iter = list(header_obj)
        except Exception:
            return False
    for raw_key in keys_iter:
        if raw_key is None:
            continue
        key_upper = str(raw_key).strip().upper()
        if not key_upper:
            continue
        if key_upper in SIP_META_KEYS:
            return True
        for prefix in SIP_COEFF_PREFIXES:
            if key_upper.startswith(prefix):
                return True
    return False


def _ensure_sip_suffix_inplace(header_obj: Any):
    """Append '-SIP' to CTYPE axes when SIP terms are present."""

    if header_obj is None:
        return None
    try:
        has_sip = _header_contains_sip_terms(header_obj)
    except Exception:
        has_sip = False
    if not has_sip:
        return header_obj

    def _maybe_update(key: str) -> None:
        try:
            value = header_obj[key]
        except Exception:
            try:
                value = header_obj.get(key)
            except Exception:
                value = None
        if not isinstance(value, str):
            return
        cleaned = value.strip()
        if not cleaned or cleaned.upper().endswith("-SIP"):
            return
        new_value = f"{cleaned}-SIP"
        try:
            header_obj[key] = new_value
        except Exception:
            try:
                header_obj.set(key, new_value)
            except Exception:
                pass

    _maybe_update("CTYPE1")
    _maybe_update("CTYPE2")
    return header_obj


def _build_wcs_from_header(header_obj: Any):
    """Instantiate a WCS while normalizing SIP metadata when needed."""

    if header_obj is None or WCS is None:
        return None
    try:
        hdr = _ensure_sip_suffix_inplace(header_obj)
    except Exception:
        hdr = header_obj
    try:
        return WCS(hdr, naxis=2, relax=True)
    except Exception:
        return None


def _header_has_wcs(header: Any) -> bool:
    """Return ``True`` when the FITS header already contains a valid WCS."""

    if header is None:
        return False
    try:
        wcs_obj = _build_wcs_from_header(header)
        if getattr(wcs_obj, "is_celestial", False):
            return True
    except Exception:
        pass
    for key in ("CTYPE1", "CTYPE2", "CD1_1", "CD2_2", "PC1_1", "PC2_2"):
        if header.get(key):
            return True
    return False


def _write_header_to_fits_local(file_path: str, header_obj: Any) -> None:
    """Persist ``header_obj`` into the primary HDU header of ``file_path``.

    Mirrors the Tk filter helper so that ASTAP solutions can be written back
    to disk when the user enables the corresponding option in the Qt GUI.
    """

    if fits is None or header_obj is None:
        return
    path_obj = _expand_to_path(file_path)
    if not path_obj or not path_obj.is_file():
        return
    try:
        with fits.open(str(path_obj), mode="update", memmap=False) as hdul:  # type: ignore[call-arg]
            hdul[0].header.update(header_obj)  # type: ignore[index]
            hdul.flush()
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        try:
            logger.warning("Qt Filter: failed to write WCS header to '%s': %s", file_path, exc)
        except Exception:
            pass


def _persist_wcs_header_if_requested(path: str, header_obj: Any, write_inplace: bool) -> None:
    """Persist a solved WCS header to disk when enabled."""

    if not write_inplace:
        return
    if not path or header_obj is None:
        return
    path_obj = _expand_to_path(path)
    if not path_obj:
        return
    display_name = _path_display_name(path_obj)
    try:
        _write_header_to_fits_local(str(path_obj), header_obj)
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        try:
            logger.warning("Qt Filter: failed to persist WCS for '%s': %s", display_name, exc)
        except Exception:
            pass
    else:
        try:
            logger.info("Qt Filter: WCS header written for '%s'", display_name)
        except Exception:
            pass


def _detect_instrument_from_header(header: Any) -> str | None:
    """Mimic the Tk GUI heuristics to detect the instrument label."""

    if header is None:
        return None
    try:
        creator_raw = str(header.get("CREATOR", "")).strip()
        instrume_raw = str(header.get("INSTRUME", "")).strip()
    except Exception:
        creator_raw = ""
        instrume_raw = ""
    creator = creator_raw.upper()
    instrume = instrume_raw.upper()
    if "SEESTAR S50" in creator:
        return "Seestar S50"
    if "SEESTAR S30" in creator:
        return "Seestar S30"
    if "ASIAIR" in creator:
        return instrume_raw or "ASIAIR"
    if instrume.startswith("ZWO ASI") or "ASI" in instrume:
        return instrume_raw or "ASI (Unknown)"
    if instrume_raw:
        return instrume_raw
    return creator_raw or None


def _sexagesimal_to_degrees(raw: str, is_ra: bool) -> float | None:
    """Convert sexagesimal strings to decimal degrees."""

    text = raw.strip().lower().replace("h", ":").replace("m", ":").replace("s", "")
    parts = [segment for segment in text.replace("°", ":").replace("'", ":").split(":") if segment]
    if not parts:
        return None

    try:
        numbers = [float(part) for part in parts]
    except Exception:
        return None

    if is_ra:
        hours = numbers[0]
        minutes = numbers[1] if len(numbers) > 1 else 0.0
        seconds = numbers[2] if len(numbers) > 2 else 0.0
        return (hours + minutes / 60.0 + seconds / 3600.0) * 15.0

    sign = -1.0 if text.startswith("-") else 1.0
    degrees = abs(numbers[0])
    arcmin = numbers[1] if len(numbers) > 1 else 0.0
    arcsec = numbers[2] if len(numbers) > 2 else 0.0
    return sign * (degrees + arcmin / 60.0 + arcsec / 3600.0)


def _extract_center_from_header(header: Any) -> Tuple[float | None, float | None]:
    """Attempt to extract the approximate pointing centre from a FITS header."""

    if header is None:
        return None, None

    def _coerce(value: Any, *, is_ra: bool) -> float | None:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            text = str(value).strip()
        except Exception:
            return None
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            pass
        if ":" in text or any(sep in text for sep in ("h", "m", "°", "'")):
            return _sexagesimal_to_degrees(text, is_ra)
        return None

    ra_keys = ("CRVAL1", "OBJCTRA", "RA", "RA_DEG", "OBJRA", "OBJRA_DEG")
    dec_keys = ("CRVAL2", "OBJCTDEC", "DEC", "DEC_DEG", "OBJDEC", "OBJDEC_DEG")

    ra_value = None
    for key in ra_keys:
        if key in header:
            ra_value = _coerce(header.get(key), is_ra=True)
            if ra_value is not None:
                break

    dec_value = None
    for key in dec_keys:
        if key in header:
            dec_value = _coerce(header.get(key), is_ra=False)
            if dec_value is not None:
                break

    return ra_value, dec_value


def _force_phase45_disabled(mapping: Any) -> None:
    """Ensure Phase 4.5 / super-tiles stay disabled in config dicts."""

    if not isinstance(mapping, dict):
        return
    try:
        mapping["inter_master_merge_enable"] = False
    except Exception:
        pass


class _DirectoryScanWorker(QObject):
    """Background worker resolving missing WCS entries with ASTAP."""

    progress_changed = Signal(int, str)
    row_updated = Signal(int, dict)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        items: List[_NormalizedItem],
        solver_settings: dict | None,
        localizer: Any,
        astap_overrides: dict | None = None,
    ) -> None:
        super().__init__()
        self._items = items
        self._solver_settings = solver_settings or {}
        self._localizer = localizer
        self._overrides = astap_overrides or {}
        self._stop_requested = False
        self._write_wcs_to_file = bool(self._overrides.get("write_wcs_to_file", False))

    @Slot()
    def run(self) -> None:
        total = len(self._items)
        if total == 0:
            self.progress_changed.emit(0, self._localizer.get("filter.scan.empty", "No files to scan."))
            self.finished.emit()
            return

        astap_cfg = self._prepare_astap_configuration()
        if astap_cfg is None:
            message = self._localizer.get(
                "filter.scan.astap_missing",
                "ASTAP configuration is incomplete; skipping WCS solving.",
            )
            self.error.emit(message)

        concurrency_value = max(1, int(astap_cfg.get("concurrency", 1))) if astap_cfg else 1
        executor: ThreadPoolExecutor | None = None
        astap_semaphore: threading.BoundedSemaphore | None = None
        pending_futures: list[Any] = []
        processed_count = 0

        if astap_cfg is not None:
            executor = ThreadPoolExecutor(
                max_workers=concurrency_value,
                thread_name_prefix="FilterASTAP",
            )
            astap_semaphore = threading.BoundedSemaphore(value=concurrency_value)

        def _solve_astap_target(
            idx: int,
            entry: _NormalizedItem,
            image_path: str,
            header_obj: Any,
            row_snapshot: dict[str, Any],
            payload: dict | None,
            config: dict[str, Any],
            write_inplace: bool,
            semaphore: threading.BoundedSemaphore | None,
        ) -> tuple[int, dict[str, Any]]:
            display_name = _path_display_name(image_path)
            wcs_result = None
            timeout_val = config.get("timeout") or 180
            try:
                if semaphore is not None:
                    semaphore.acquire()
                wcs_result = solve_with_astap(
                    image_path,
                    header_obj,
                    config["exe"],
                    config["data"],
                    search_radius_deg=config.get("radius"),
                    downsample_factor=config.get("downsample"),
                    sensitivity=config.get("sensitivity"),
                    astap_drizzled_fallback_enabled=config.get(
                        "astap_drizzled_fallback_enabled", False
                    ),
                    timeout_sec=timeout_val,
                    update_original_header_in_place=write_inplace,
                )
            except Exception as exc:
                row_snapshot["error"] = str(exc)
            finally:
                if semaphore is not None:
                    try:
                        semaphore.release()
                    except Exception:
                        pass

            if wcs_result is not None and getattr(wcs_result, "is_celestial", False):
                row_snapshot["solver"] = "ASTAP"
                row_snapshot["has_wcs"] = True
                if payload is not None:
                    payload["has_wcs"] = True
                try:
                    entry.wcs_cache = wcs_result
                except Exception:
                    pass
                if write_inplace and header_obj is not None:
                    try:
                        _persist_wcs_header_if_requested(image_path, header_obj, True)
                    except Exception:
                        pass
            else:
                if "error" not in row_snapshot:
                    row_snapshot["error"] = self._localizer.get(
                        "filter.scan.astap_failed",
                        "ASTAP failed for {name}.",
                        name=display_name,
                    )
            return idx, row_snapshot

        try:
            for index, item in enumerate(self._items):
                if self._stop_requested:
                    break
                path = item.file_path
                base_payload = item.original if isinstance(item.original, dict) else None
                if base_payload is not None and path:
                    base_payload.setdefault("path", path)
                    base_payload.setdefault("path_raw", path)
                inspect_message = self._localizer.get(
                    "filter.scan.inspecting",
                    "Inspecting {name}…",
                    name=_path_display_name(path) if path else item.display_name,
                )
                self.progress_changed.emit(self._progress_percent(processed_count, total), inspect_message)
                row_update: dict[str, Any] = {}

                if not path or not _path_is_file(path):
                    row_update["error"] = self._localizer.get(
                        "filter.scan.missing",
                        "File missing or not accessible.",
                    )
                    self.row_updated.emit(index, row_update)
                    processed_count += 1
                    completion_msg = self._localizer.get(
                        "filter.scan.progress",
                        "Processed {done}/{total} files.",
                        done=processed_count,
                        total=total,
                    )
                    self.progress_changed.emit(self._progress_percent(processed_count, total), completion_msg)
                    continue

                header = None
                if fits is not None:
                    try:
                        header = fits.getheader(path, ignore_missing_end=True)
                    except Exception as exc:  # pragma: no cover - passthrough IO error
                        row_update["error"] = str(exc)
                else:
                    row_update["error"] = self._localizer.get(
                        "filter.scan.astropy_missing",
                        "Astropy is not installed; cannot inspect headers.",
                    )
                sanitized_header = _sanitize_header_subset(header)
                if sanitized_header is not None:
                    if base_payload is not None:
                        base_payload["header"] = sanitized_header
                    try:
                        item.header_cache = sanitized_header
                    except Exception:
                        pass

                has_wcs = _header_has_wcs(header) if header is not None else False
                if base_payload is not None:
                    base_payload["has_wcs"] = bool(has_wcs)
                row_update["has_wcs"] = has_wcs

                instrument = _detect_instrument_from_header(header)
                if instrument:
                    if base_payload is not None:
                        base_payload["instrument"] = instrument
                    row_update["instrument"] = instrument

                if header is not None:
                    ra_deg, dec_deg = _extract_center_from_header(header)
                    if ra_deg is not None and dec_deg is not None:
                        row_update["center_ra_deg"] = ra_deg
                        row_update["center_dec_deg"] = dec_deg
                        if base_payload is not None:
                            base_payload["center_ra_deg"] = float(ra_deg)
                            base_payload["center_dec_deg"] = float(dec_deg)
                            base_payload["RA"] = float(ra_deg)
                            base_payload["DEC"] = float(dec_deg)

                needs_astap = (
                    astap_cfg is not None
                    and not has_wcs
                    and solve_with_astap is not None
                    and header is not None
                    and executor is not None
                )
                if needs_astap:
                    solving_message = self._localizer.get(
                        "filter.scan.solving",
                        "Solving WCS with ASTAP…",
                    )
                    self.progress_changed.emit(self._progress_percent(processed_count, total), solving_message)
                    if executor is not None:
                        future = executor.submit(
                            _solve_astap_target,
                            index,
                            item,
                            path,
                            header,
                            row_update.copy(),
                            base_payload,
                            astap_cfg,
                            bool(self._write_wcs_to_file),
                            astap_semaphore,
                        )
                        pending_futures.append(future)
                        continue

                self.row_updated.emit(index, row_update)
                processed_count += 1
                completion_msg = self._localizer.get(
                    "filter.scan.progress",
                    "Processed {done}/{total} files.",
                    done=processed_count,
                    total=total,
                )
                self.progress_changed.emit(self._progress_percent(processed_count, total), completion_msg)

            if pending_futures:
                for future in as_completed(pending_futures):
                    if self._stop_requested:
                        break
                    try:
                        idx, solved_row = future.result()
                    except Exception as exc:
                        try:
                            self.error.emit(str(exc))
                        except Exception:
                            pass
                        continue
                    self.row_updated.emit(idx, solved_row)
                    processed_count += 1
                    completion_msg = self._localizer.get(
                        "filter.scan.progress",
                        "Processed {done}/{total} files.",
                        done=processed_count,
                        total=total,
                    )
                    self.progress_changed.emit(self._progress_percent(processed_count, total), completion_msg)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        self.finished.emit()

    def request_stop(self) -> None:
        self._stop_requested = True

    def _prepare_astap_configuration(self) -> dict[str, Any] | None:
        exe_candidates = [
            self._solver_settings.get("astap_executable_path"),
            self._solver_settings.get("astap_executable"),
            self._overrides.get("astap_executable_path"),
        ]
        data_candidates = [
            self._solver_settings.get("astap_data_directory_path"),
            self._solver_settings.get("astap_data_dir"),
            self._overrides.get("astap_data_directory_path"),
        ]
        exe_path_obj = _expand_to_path(next((p for p in exe_candidates if p), None))
        data_dir_obj = _expand_to_path(next((p for p in data_candidates if p), None))
        if not (exe_path_obj and exe_path_obj.is_file()):
            return None
        exe_path = str(exe_path_obj)

        radius = self._coerce_float(
            self._solver_settings.get("astap_search_radius_deg"),
            self._overrides.get("astap_search_radius_deg"),
            default=None,
        )
        downsample = self._coerce_int(
            self._solver_settings.get("astap_downsample"),
            self._overrides.get("astap_downsample"),
            default=None,
        )
        sensitivity = self._coerce_int(
            self._solver_settings.get("astap_sensitivity"),
            self._overrides.get("astap_sensitivity"),
            default=None,
        )
        timeout = self._coerce_int(
            self._solver_settings.get("astap_timeout_sec"),
            self._overrides.get("astap_timeout_sec"),
            default=180,
        )
        fallback_enabled = self._overrides.get("astap_drizzled_fallback_enabled")
        if fallback_enabled is None:
            fallback_enabled = self._solver_settings.get(
                "astap_drizzled_fallback_enabled", False
            )
        fallback_enabled = bool(fallback_enabled)

        concurrency = self._coerce_int(
            self._solver_settings.get("astap_max_instances"),
            self._overrides.get("astap_max_instances"),
            default=1,
        )
        concurrency_value = max(1, int(concurrency or 1))
        if set_astap_max_concurrent_instances is not None:
            try:
                set_astap_max_concurrent_instances(concurrency_value)
            except Exception:  # pragma: no cover - runtime guard
                pass
        data_dir = str(data_dir_obj) if data_dir_obj else ""

        return {
            "exe": exe_path,
            "data": data_dir,
            "radius": radius,
            "downsample": downsample,
            "sensitivity": sensitivity,
            "timeout": timeout,
            "concurrency": concurrency_value,
            "astap_drizzled_fallback_enabled": fallback_enabled,
        }

    @staticmethod
    def _progress_percent(current: int, total: int) -> int:
        if total <= 0:
            return 0
        try:
            return min(100, max(0, int((current / total) * 100)))
        except Exception:  # pragma: no cover - defensive
            return 0

    @staticmethod
    def _coerce_float(*values: Any, default: float | None = None) -> float | None:
        for value in values:
            if value in (None, "", False):
                continue
            try:
                return float(value)
            except Exception:
                continue
        return default

    @staticmethod
    def _coerce_int(*values: Any, default: int | None = None) -> int | None:
        for value in values:
            if value in (None, "", False):
                continue
            try:
                return int(value)
            except Exception:
                continue
        return default


class _StreamIngestWorker(QObject):
    """Background worker that yields normalized entries in small batches."""

    batch_ready = Signal(list)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        payload: Any,
        initial_overrides: Any,
        *,
        scan_recursive: bool,
        batch_size: int,
    ) -> None:
        super().__init__()
        self._payload = payload
        self._initial_overrides = initial_overrides
        self._runtime_overrides: dict[str, Any] = {}
        _force_phase45_disabled(self._initial_overrides)
        self._scan_recursive = bool(scan_recursive)
        self._batch_size = max(1, int(batch_size))
        self._stop_requested = False

    @Slot()
    def run(self) -> None:
        try:
            iterator = _iter_normalized_entries(
                self._payload,
                self._initial_overrides,
                scan_recursive=self._scan_recursive,
            )
            batch: list[_NormalizedItem] = []
            for entry in iterator:
                if self._stop_requested:
                    break
                batch.append(entry)
                if len(batch) >= self._batch_size:
                    self.batch_ready.emit(batch)
                    batch = []
            if batch:
                self.batch_ready.emit(batch)
        except Exception as exc:  # pragma: no cover - defensive guard
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def request_stop(self) -> None:
        self._stop_requested = True


class FilterQtDialog(QDialog):
    """Small Qt dialog allowing users to review candidate files.

    The dialog intentionally focuses on the essentials for now:

    * present the list of candidate files that will be used for the mosaic;
    * allow end users to include or exclude individual entries;
    * expose OK/Cancel semantics equivalent to the Tk filter interface.

    The dialog stores the input payload so that future iterations can add
    richer analysis (clustering, previews, stream scanning, ...).
    """

    _async_log_signal = Signal(str, str)
    _auto_group_finished_signal = Signal(object, object)
    _auto_group_stage_signal = Signal(str)

    def __init__(
        self,
        raw_files_with_wcs_or_dir: Any,
        initial_overrides: Any = None,
        *,
        stream_scan: bool = False,
        scan_recursive: bool = True,
        batch_size: int = 100,
        preview_cap: int = 200,
        solver_settings_dict: dict | None = None,
        config_overrides: dict | None = None,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        icon = _load_zemosaic_qicon()
        if icon is not None:
            self.setWindowIcon(icon)
        self._input_payload = raw_files_with_wcs_or_dir
        self._initial_overrides = initial_overrides
        self._runtime_overrides: dict[str, Any] = {}
        _force_phase45_disabled(self._initial_overrides)
        self._stream_scan = stream_scan
        self._scan_recursive = scan_recursive
        self._batch_size = batch_size
        self._preview_cap = preview_cap
        self._solver_settings = solver_settings_dict
        overrides: dict[str, Any] = {}
        if isinstance(config_overrides, dict):
            overrides.update(config_overrides)
        else:
            try:
                overrides.update(dict(config_overrides or {}))
            except Exception:
                pass
        self._config_overrides = overrides
        _force_phase45_disabled(self._config_overrides)
        if _load_gui_config is not None:
            try:
                loaded_cfg = _load_gui_config()
                if isinstance(loaded_cfg, dict):
                    for key, value in loaded_cfg.items():
                        if key not in self._config_overrides:
                            self._config_overrides[key] = value
            except Exception:
                pass
        raw_cluster_value = self._config_value("cluster_panel_threshold", 0.0)
        cluster_candidate = 0.0
        try:
            cluster_candidate = float(raw_cluster_value)
        except Exception:
            cluster_candidate = 0.0
        if not math.isfinite(cluster_candidate):
            cluster_candidate = 0.0
        if cluster_candidate > 0.0:
            cluster_candidate = min(1.0, cluster_candidate)
        else:
            cluster_candidate = 0.0
        self._cluster_threshold_value = cluster_candidate

        raw_max_raw = self._config_value("max_raw_per_master_tile", 50)
        max_raw_candidate = 50
        try:
            max_raw_candidate = int(raw_max_raw)
        except Exception:
            max_raw_candidate = 50
        max_raw_candidate = max(0, min(500, max_raw_candidate))
        self._max_raw_per_tile_value = max_raw_candidate

        if isinstance(self._config_overrides, dict):
            if self._cluster_threshold_value > 0.0:
                self._config_overrides["cluster_panel_threshold"] = float(self._cluster_threshold_value)
            else:
                self._config_overrides.pop("cluster_panel_threshold", None)
            # Keep the explicit max-raw cap (0 = unlimited) so autosplit respects the choice.
            self._config_overrides["max_raw_per_master_tile"] = int(self._max_raw_per_tile_value)
        self._accepted = False

        self._localizer = self._load_localizer()
        self._normalized_items: list[_NormalizedItem] = []
        self._stream_thread: QThread | None = None
        self._stream_worker: _StreamIngestWorker | None = None
        self._streaming_active = False
        self._streaming_completed = not self._stream_scan
        self._stream_loaded_count = 0
        self._preview_empty_logged = False
        try:
            self._batch_size = max(1, int(batch_size))
        except Exception:
            self._batch_size = 100
        self._dialog_button_box: QDialogButtonBox | None = None
        self._async_log_signal.connect(self._append_log_from_signal)
        self._auto_group_finished_signal.connect(self._handle_auto_group_finished)
        self._auto_group_stage_signal.connect(self._handle_auto_group_stage_update)

        if self._stream_scan:
            self._normalized_items = []
        else:
            self._normalized_items = self._normalize_items(raw_files_with_wcs_or_dir, initial_overrides)

        overrides = config_overrides or {}
        self._auto_group_requested = bool(overrides.get("filter_auto_group", False))
        self._seestar_priority = bool(overrides.get("filter_seestar_priority", False))

        self._tree = QTreeWidget(self)
        self._summary_label = QLabel(self)
        self._status_label = QLabel(self)
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._run_analysis_btn: QPushButton | None = None
        self._export_csv_btn: QPushButton | None = None
        self._toolbar_maximize_btn: QPushButton | None = None
        self._saved_geometry: QRect | None = None
        self._maximized_state = False
        self._distance_spin: QDoubleSpinBox | None = None
        self._instrument_combo: QComboBox | None = None
        self._instrument_unknown_token = "__unknown__"
        self._scan_thread: QThread | None = None
        self._scan_worker: _DirectoryScanWorker | None = None
        self._auto_group_checkbox: QCheckBox | None = None
        self._seestar_checkbox: QCheckBox | None = None
        self._sds_checkbox: QCheckBox | None = None
        self._coverage_checkbox: QCheckBox | None = None
        self._overcap_spin: QSpinBox | None = None
        self._overlap_spin: QSpinBox | None = None
        self._cluster_threshold_spin: QDoubleSpinBox | None = None
        self._max_raw_per_tile_spin: QSpinBox | None = None
        self._auto_angle_checkbox: QCheckBox | None = None
        self._angle_split_spin: QDoubleSpinBox | None = None
        self._astap_instances_combo: QComboBox | None = None
        self._astap_instances_value = self._resolve_initial_astap_instances()
        self._preview_canvas: FigureCanvasQTAgg | None = None
        self._preview_axes = None
        self._preview_hint_label = QLabel(self)
        self._coverage_canvas: FigureCanvasQTAgg | None = None
        self._coverage_axes = None
        self._preview_tabs: QTabWidget | None = None
        self._preview_default_hint = ""
        self._preview_refresh_pending = False
        self._preview_last_refresh = 0.0
        self._preview_refresh_interval = PREVIEW_REFRESH_INTERVAL_SEC
        self._preview_draw_throttle = PREVIEW_DRAW_THROTTLE_SEC
        self._last_preview_draw_ts = 0.0
        self._preview_draw_attempts = 0
        self._cluster_groups: list[list[_NormalizedItem]] = []
        self._cluster_threshold_used: float | None = None
        self._cluster_refresh_pending = False
        self._auto_group_button: QPushButton | None = None
        self._auto_group_summary_label: QLabel | None = None
        self._solve_overlay: QWidget | None = None
        self._solve_overlay_label: QLabel | None = None
        self._solve_animation: QMovie | None = None
        self._auto_group_running = False
        self._auto_group_override_groups: list[list[dict[str, Any]]] | None = None
        self._auto_group_stage_text = ""
        self._auto_group_started_at: float | None = None
        self._auto_group_elapsed_timer: QTimer | None = None
        self._header_cache: dict[str, Any] = {}
        self._last_eqmode_summary: dict[str, Any] | None = None
        self._last_auto_group_result: dict[str, Any] | None = None
        self._global_wcs_state: dict[str, Any] = {
            "descriptor": None,
            "meta": None,
            "fits_path": None,
            "json_path": None,
        }
        self._coverage_first_enabled_flag = self._coerce_bool(
            self._config_value("filter_enable_coverage_first", True),
            True,
        )
        self._overcap_percent_value = self._clamp_overcap_percent(
            self._config_value("filter_overcap_allowance_pct", 10)
        )
        self._batch_overlap_percent_value = self._clamp_overlap_percent(
            self._config_value("batch_overlap_pct", 40)
        )
        base_angle = _sanitize_angle_value(self._config_value("cluster_orientation_split_deg", 0.0), 0.0)
        if base_angle <= 0.0:
            base_angle = ANGLE_SPLIT_DEFAULT_DEG
        self._auto_angle_enabled = True
        self._angle_split_value = float(base_angle)
        self._group_outline_bounds: list[_GroupOutline] = []
        self._group_outline_collection: LineCollection | None = None
        self._entry_check_state: list[bool] = [bool(item.include_by_default) for item in self._normalized_items]
        self._entry_items: list[QTreeWidgetItem | None] = [None] * len(self._normalized_items)
        self._group_item_map: dict[_GroupKey, QTreeWidgetItem] = {}
        self._group_entries: dict[_GroupKey, list[int]] = {}
        self._cluster_index_to_group_key: dict[int, _GroupKey] = {}
        self._tree_signal_guard = False
        self._rectangle_selector: RectangleSelector | None = None
        self._selected_group_keys: set[_GroupKey] = set()
        self._selection_bounds: tuple[float, float, float, float] | None = None
        self._selection_check_snapshot: list[bool] | None = None
        self._preview_color_cycle: tuple[str, ...] = (
            "#3f7ad6",
            "#d64b3f",
            "#3fd65d",
            "#d6a63f",
            "#8a3fd6",
            "#3fc6d6",
            "#d63fb8",
            "#6ed63f",
        )

        self._preview_canvas = self._create_preview_canvas()
        self._activity_log_output: QPlainTextEdit | None = None
        self._scan_recursive_checkbox: QCheckBox | None = None
        self._draw_group_outlines_checkbox: QCheckBox | None = None
        self._color_by_group_checkbox: QCheckBox | None = None
        self._write_wcs_checkbox: QCheckBox | None = None
        self._sds_mode_initial = self._coerce_bool(
            (initial_overrides or {}).get("sds_mode")
            if isinstance(initial_overrides, dict)
            else None,
            self._coerce_bool(
                (config_overrides or {}).get("sds_mode_default")
                if isinstance(config_overrides, dict)
                else None,
                bool(DEFAULT_FILTER_CONFIG.get("sds_mode_default", False)),
            ),
        )
        self._cache_csv_path: str | None = None
        if self._stream_scan:
            candidate_dir = _expand_to_path(raw_files_with_wcs_or_dir)
            if candidate_dir and candidate_dir.is_dir():
                self._cache_csv_path = str(candidate_dir / "headers_cache.csv")
        self._build_ui()
        self._build_processing_overlay()
        cache_loaded = False
        if self._cache_csv_path:
            cache_path_obj = _expand_to_path(self._cache_csv_path)
            if cache_path_obj and cache_path_obj.is_file():
                self._cache_csv_path = str(cache_path_obj)
                cache_loaded = True
        if self._stream_scan:
            self._prepare_streaming_mode(raw_files_with_wcs_or_dir, initial_overrides)
            self._debug_log(
                f"stream_mode=True; pending_start={not self._streaming_completed} "
                f"csv_loaded={cache_loaded} input_source={self._describe_input_source()}"
            )
        else:
            self._populate_tree()
            self._update_summary_label()
            self._schedule_preview_refresh()
            self._schedule_cluster_refresh()
            self._debug_log(
                f"stream_mode=False; pending_start=False csv_loaded={cache_loaded} "
                f"entries_loaded={len(self._normalized_items)} input_source={self._describe_input_source()}"
            )
            # If the caller provided pre-planned master groups, apply them so
            # that the user immediately sees the same grouping as in Tk.
            preplanned = None
            if isinstance(self._initial_overrides, dict):
                preplanned = self._initial_overrides.get("preplan_master_groups")
            if isinstance(preplanned, list) and preplanned:
                try:
                    sizes = [len(gr) for gr in preplanned if isinstance(gr, list)]
                    payload = {"final_groups": preplanned, "sizes": sizes}
                    self._apply_auto_group_result(payload)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Helpers - localization and normalization
    # ------------------------------------------------------------------
    def _load_localizer(self) -> Any:
        language = None
        try:
            language = str(self._config_overrides.get("language"))
        except Exception:
            language = None

        if ZeMosaicLocalization is not None:
            try:
                loc = ZeMosaicLocalization()
                if language:
                    loc.set_language(language)
                return loc
            except Exception:
                pass
        fallback = _FallbackLocalizer(language or "en")
        return fallback

    def _normalize_items(
        self,
        payload: Any,
        initial_overrides: Any,
    ) -> List[_NormalizedItem]:
        """Materialise the input payload into ``_NormalizedItem`` instances."""

        return list(
            _iter_normalized_entries(
                payload,
                initial_overrides,
                scan_recursive=self._scan_recursive,
            )
        )

    # ------------------------------------------------------------------
    # UI creation
    # ------------------------------------------------------------------
    def _create_preview_canvas(self) -> FigureCanvasQTAgg | None:
        """Initialise the Matplotlib preview canvas if Matplotlib is available."""

        self._preview_hint_label.setWordWrap(True)
        unavailable_text = self._localizer.get(
            "filter.preview.unavailable",
            "Matplotlib is not available; preview disabled.",
        )
        if Figure is None or FigureCanvasQTAgg is None:
            self._preview_default_hint = unavailable_text
            self._preview_hint_label.setText(unavailable_text)
            return None

        figure = Figure(figsize=(5, 3))
        canvas = FigureCanvasQTAgg(figure)
        try:
            canvas.destroyed.connect(self._on_preview_canvas_destroyed)  # type: ignore[attr-defined]
        except Exception:
            pass
        canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        canvas.customContextMenuRequested.connect(self._on_preview_context_menu)  # type: ignore[arg-type]
        axes = figure.add_subplot(111)
        self._preview_axes = axes
        axes.set_xlabel(self._localizer.get("filter.preview.ra", "Right Ascension (°)"))
        axes.set_ylabel(self._localizer.get("filter.preview.dec", "Declination (°)"))
        axes.set_title(self._localizer.get("filter.preview.title", "Sky preview"))
        axes.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        self._preview_default_hint = self._localizer.get(
            "filter.preview.hint",
            "Preview shows the approximate pointing of selected frames (limited by preview cap).",
        )
        self._preview_hint_label.setText(self._preview_default_hint)
        if RectangleSelector is not None:
            try:
                selector = RectangleSelector(
                    axes,
                    self._on_preview_rectangle_selected,
                    useblit=True,
                    button=[1],
                    minspanx=0.0,
                    minspany=0.0,
                    spancoords="data",
                    interactive=False,
                    drag_from_anywhere=False,
                )
                try:
                    selector.set_props(
                        dict(facecolor=(0.2, 0.4, 0.9, 0.2), edgecolor="#3478d6", linewidth=1.2)
                    )
                except Exception:
                    try:
                        selector.rectprops.update(  # type: ignore[attr-defined]
                            facecolor=(0.2, 0.4, 0.9, 0.2),
                            edgecolor="#3478d6",
                            linewidth=1.2,
                        )
                    except Exception:
                        pass
                selector.set_active(True)
                self._rectangle_selector = selector
            except Exception:
                self._rectangle_selector = None
        return canvas

    def _create_coverage_canvas(self) -> FigureCanvasQTAgg | None:
        """Initialise the coverage-map Matplotlib canvas (global WCS plane)."""

        if Figure is None or FigureCanvasQTAgg is None:
            return None

        figure = Figure(figsize=(5, 3))
        canvas = FigureCanvasQTAgg(figure)
        try:
            canvas.destroyed.connect(self._on_coverage_canvas_destroyed)  # type: ignore[attr-defined]
        except Exception:
            pass
        axes = figure.add_subplot(111)
        self._coverage_axes = axes
        axes.set_xlabel(self._localizer.get("filter_axis_cov_x", "X [px]"))
        axes.set_ylabel(self._localizer.get("filter_axis_cov_y", "Y [px]"))
        axes.set_aspect("equal")
        axes.grid(True, linestyle=":", linewidth=0.6)
        self._coverage_canvas = canvas
        return canvas

    def _create_toolbar_widget(self) -> QWidget:
        """Build the top toolbar with status indicators and action buttons."""

        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        self._status_label.setWordWrap(True)
        self._status_label.setText(
            self._localizer.get("filter_status_ready", "Ready — click Analyse to scan.")
        )
        layout.addWidget(self._status_label, 1)

        self._progress_bar.setTextVisible(False)
        layout.addWidget(self._progress_bar, 0)

        button_row = QHBoxLayout()
        analyse_label = self._localizer.get("filter_btn_analyse", "Analyse")
        self._run_analysis_btn = QPushButton(analyse_label, self)
        self._run_analysis_btn.clicked.connect(self._on_run_analysis)  # type: ignore[arg-type]
        button_row.addWidget(self._run_analysis_btn)

        export_label = self._localizer.get("filter_btn_export_csv", "Export CSV")
        self._export_csv_btn = QPushButton(export_label, self)
        self._export_csv_btn.clicked.connect(self._on_export_csv)  # type: ignore[arg-type]
        button_row.addWidget(self._export_csv_btn)

        max_normal = self._localizer.get("filter_btn_maximize", "Maximize")
        self._toolbar_maximize_btn = QPushButton(max_normal, self)
        self._toolbar_maximize_btn.clicked.connect(self._toggle_maximize_restore)  # type: ignore[arg-type]
        button_row.addWidget(self._toolbar_maximize_btn)

        layout.addLayout(button_row)
        return container

    def _create_exclusion_box(self) -> QGroupBox:
        """Create the 'Exclude by distance to center' controls."""

        title = self._localizer.get(
            "filter_exclude_by_distance_title",
            "Exclude by distance to center",
        )
        group = QGroupBox(title, self)
        layout = QHBoxLayout(group)
        label = QLabel(self._localizer.get("filter_distance_label", "Distance (deg):"), group)
        layout.addWidget(label)
        spin = QDoubleSpinBox(group)
        spin.setDecimals(1)
        spin.setRange(0.1, 180.0)
        spin.setSingleStep(0.5)
        spin.setValue(5.0)
        spin.setSuffix(" °")
        self._distance_spin = spin
        layout.addWidget(spin)
        button = QPushButton(
            self._localizer.get("filter_exclude_gt_x", "Exclude > X°"),
            group,
        )
        button.clicked.connect(self._on_exclude_distance)  # type: ignore[arg-type]
        layout.addWidget(button)
        return group

    def _create_instrument_row(self) -> QWidget:
        """Create the instrument selection dropdown row."""

        container = QGroupBox(
            self._localizer.get("filter.instrument.group", "Instrument filter"),
            self,
        )
        layout = QHBoxLayout(container)
        label = QLabel(self._localizer.get("filter.instrument.label", "Instrument:"), container)
        layout.addWidget(label)

        self._instrument_combo = QComboBox(container)
        self._instrument_combo.currentIndexChanged.connect(  # type: ignore[arg-type]
            self._apply_instrument_filter
        )
        layout.addWidget(self._instrument_combo, 1)
        self._refresh_instrument_options()
        return container

    def _create_wcs_controls_box(self) -> QGroupBox:
        """Create the WCS / master-tile / SDS controls."""

        title = self._localizer.get("filter.group.wcs", "WCS / Master tile controls")
        box = QGroupBox(title, self)
        layout = QGridLayout(box)

        resolve_btn = QPushButton(
            self._localizer.get("filter_btn_resolve_wcs", "Resolve missing WCS"),
            box,
        )
        resolve_btn.clicked.connect(self._on_resolve_wcs_clicked)  # type: ignore[arg-type]
        layout.addWidget(resolve_btn, 0, 0)

        self._auto_group_button = QPushButton(
            self._localizer.get("filter_btn_auto_group", "Auto-organize Master Tiles"),
            box,
        )
        self._auto_group_button.clicked.connect(self._on_auto_group_clicked)  # type: ignore[arg-type]
        helpers_ready = bool(_CLUSTER_CONNECTED is not None and _AUTOSPLIT_GROUPS is not None)
        tooltip = None
        if not helpers_ready:
            tooltip = self._localizer.get(
                "filter.cluster.helpers_missing",
                "Auto-organisation helpers are unavailable; install the worker module.",
            )
        self._auto_group_button.setEnabled(helpers_ready)
        if tooltip:
            self._auto_group_button.setToolTip(tooltip)
        layout.addWidget(self._auto_group_button, 0, 1)

        manual_label = self._localizer.get(
            "filter_btn_manual_group",
            "Manual-organize Master Tiles",
        )
        self._manual_group_button = QPushButton(manual_label, box)
        self._manual_group_button.clicked.connect(self._on_manual_group_clicked)  # type: ignore[arg-type]
        self._manual_group_button.setEnabled(helpers_ready)
        if tooltip:
            self._manual_group_button.setToolTip(tooltip)
        layout.addWidget(self._manual_group_button, 0, 2)

        # Summary label mirroring Tk's "Prepared N group(s), sizes: …"
        self._auto_group_summary_label = QLabel("", box)
        self._auto_group_summary_label.setWordWrap(True)
        initial_summary = self._format_group_summary(0, "[]")
        self._auto_group_summary_label.setText(initial_summary)
        self._auto_group_summary_label.setToolTip(initial_summary)
        layout.addWidget(self._auto_group_summary_label, 0, 3, 11, 1)
        layout.setColumnStretch(3, 1)

        astap_label = QLabel(
            self._localizer.get("filter_label_astap_instances", "Max ASTAP instances"),
            box,
        )
        layout.addWidget(astap_label, 1, 0)
        self._astap_instances_combo = QComboBox(box)
        self._populate_astap_instances_combo()
        layout.addWidget(self._astap_instances_combo, 1, 1)

        self._draw_group_outlines_checkbox = QCheckBox(
            self._localizer.get("filter_chk_draw_footprints", "Draw group WCS outlines"),
            box,
        )
        self._draw_group_outlines_checkbox.setChecked(True)
        self._draw_group_outlines_checkbox.toggled.connect(  # type: ignore[arg-type]
            lambda _checked: self._schedule_preview_refresh()
        )
        layout.addWidget(self._draw_group_outlines_checkbox, 2, 0)

        self._write_wcs_checkbox = QCheckBox(
            self._localizer.get("filter_chk_write_wcs", "Write WCS to file"),
            box,
        )
        self._write_wcs_checkbox.setChecked(True)
        layout.addWidget(self._write_wcs_checkbox, 2, 1)

        self._coverage_checkbox = QCheckBox(
            self._localizer.get(
                "ui_coverage_first",
                "Coverage-first clustering (may exceed Max raws/tile)",
            ),
            box,
        )
        self._coverage_checkbox.setChecked(bool(self._coverage_first_enabled_flag))
        self._coverage_checkbox.toggled.connect(self._on_coverage_first_toggled)  # type: ignore[arg-type]
        layout.addWidget(self._coverage_checkbox, 3, 0, 1, 2)

        overcap_label = QLabel(
            self._localizer.get("ui_overcap_allowance_pct", "Over-cap allowance (%)"),
            box,
        )
        layout.addWidget(overcap_label, 4, 0)
        self._overcap_spin = QSpinBox(box)
        self._overcap_spin.setRange(0, 50)
        self._overcap_spin.setSingleStep(5)
        self._overcap_spin.setValue(int(self._resolve_overcap_percent()))
        self._overcap_spin.valueChanged.connect(self._on_overcap_changed)  # type: ignore[arg-type]
        layout.addWidget(self._overcap_spin, 4, 1)

        overlap_label = QLabel(
            self._localizer.get("ui_batch_overlap_pct", "Overlap between batches (%)"),
            box,
        )
        layout.addWidget(overlap_label, 5, 0)
        self._overlap_spin = QSpinBox(box)
        self._overlap_spin.setRange(0, 70)
        self._overlap_spin.setSingleStep(5)
        self._overlap_spin.setValue(int(self._resolve_overlap_percent()))
        self._overlap_spin.valueChanged.connect(self._on_overlap_changed)  # type: ignore[arg-type]
        layout.addWidget(self._overlap_spin, 5, 1)

        cluster_label = QLabel(
            self._localizer.get("filter.wcs.cluster_threshold.label", "Cluster threshold (deg)"),
            box,
        )
        layout.addWidget(cluster_label, 6, 0)
        self._cluster_threshold_spin = QDoubleSpinBox(box)
        self._cluster_threshold_spin.setDecimals(3)
        self._cluster_threshold_spin.setRange(0.0, 1.0)
        self._cluster_threshold_spin.setSingleStep(0.005)
        self._cluster_threshold_spin.setValue(float(self._cluster_threshold_value))
        self._cluster_threshold_spin.setToolTip(
            "Angular clustering threshold in degrees. 0.0 = auto-detect from coverage. "
            "Smaller values = more groups, larger values = fewer, bigger groups."
        )
        self._cluster_threshold_spin.valueChanged.connect(self._on_cluster_threshold_changed)  # type: ignore[arg-type]
        layout.addWidget(self._cluster_threshold_spin, 6, 1)

        max_raw_label = QLabel(
            self._localizer.get(
                "filter.wcs.max_raw_per_tile.label",
                "Max raw frames per master tile",
            ),
            box,
        )
        layout.addWidget(max_raw_label, 7, 0)
        self._max_raw_per_tile_spin = QSpinBox(box)
        self._max_raw_per_tile_spin.setRange(0, 500)
        self._max_raw_per_tile_spin.setSingleStep(5)
        self._max_raw_per_tile_spin.setValue(int(self._max_raw_per_tile_value))
        self._max_raw_per_tile_spin.setToolTip(
            "Hard cap on the number of raw frames per master tile. 0 = unlimited (Qt Filter does not enforce a split cap). "
            "The worker may still split tiles based on memory constraints."
        )
        self._max_raw_per_tile_spin.valueChanged.connect(self._on_max_raw_per_tile_changed)  # type: ignore[arg-type]
        layout.addWidget(self._max_raw_per_tile_spin, 7, 1)

        self._auto_angle_checkbox = QCheckBox(
            self._localizer.get("ui_auto_angle_split", "Auto split by orientation"),
            box,
        )
        self._auto_angle_checkbox.setChecked(True)
        self._auto_angle_checkbox.toggled.connect(self._on_auto_angle_toggled)  # type: ignore[arg-type]
        layout.addWidget(self._auto_angle_checkbox, 8, 0, 1, 2)

        angle_label = QLabel(
            self._localizer.get("ui_angle_split_threshold", "Orientation split (deg)"),
            box,
        )
        layout.addWidget(angle_label, 9, 0)
        self._angle_split_spin = QDoubleSpinBox(box)
        self._angle_split_spin.setRange(0.0, 180.0)
        self._angle_split_spin.setSingleStep(0.5)
        self._angle_split_spin.setDecimals(1)
        self._angle_split_spin.setValue(float(self._angle_split_value))
        self._angle_split_spin.valueChanged.connect(self._on_angle_split_changed)  # type: ignore[arg-type]
        layout.addWidget(self._angle_split_spin, 9, 1)

        self._sds_checkbox = QCheckBox(
            self._localizer.get("filter_chk_sds_mode", "Enable ZeSupaDupStack (SDS)"),
            box,
        )
        self._sds_checkbox.setChecked(bool(self._sds_mode_initial))
        try:
            self._sds_checkbox.toggled.connect(self._on_sds_toggled)  # type: ignore[arg-type]
        except Exception:
            pass
        layout.addWidget(self._sds_checkbox, 10, 0, 1, 2)

        return box

    def _on_coverage_first_toggled(self, checked: bool) -> None:
        self._coverage_first_enabled_flag = bool(checked)
        self._runtime_overrides["filter_enable_coverage_first"] = self._coverage_first_enabled_flag

    def _on_overcap_changed(self, value: int) -> None:
        clamped = self._clamp_overcap_percent(value)
        self._overcap_percent_value = clamped
        self._runtime_overrides["filter_overcap_allowance_pct"] = clamped

    def _on_overlap_changed(self, value: int) -> None:
        clamped = self._clamp_overlap_percent(value)
        self._batch_overlap_percent_value = clamped
        self._runtime_overrides["batch_overlap_pct"] = clamped

    def _on_cluster_threshold_changed(self, value: float) -> None:
        try:
            candidate = float(value)
        except Exception:
            candidate = 0.0
        if not math.isfinite(candidate):
            candidate = 0.0
        # 0.0 ⇒ AUTO; drop the explicit override to keep the original detection logic.
        if candidate <= 0.0:
            self._cluster_threshold_value = 0.0
            if isinstance(self._runtime_overrides, dict):
                self._runtime_overrides.pop("cluster_panel_threshold", None)
            if isinstance(self._config_overrides, dict):
                self._config_overrides.pop("cluster_panel_threshold", None)
            self._debug_log("Cluster threshold reverted to AUTO (0.0°).")
        else:
            sanitized = min(1.0, candidate)
            self._cluster_threshold_value = sanitized
            if isinstance(self._runtime_overrides, dict):
                self._runtime_overrides["cluster_panel_threshold"] = float(sanitized)
            if isinstance(self._config_overrides, dict):
                self._config_overrides["cluster_panel_threshold"] = float(sanitized)
            self._debug_log(f"Cluster threshold override set to {sanitized:.3f}°.")
        self._persist_qt_filter_config()
        self._cluster_refresh_pending = True

    def _on_max_raw_per_tile_changed(self, value: int) -> None:
        try:
            value_int = int(value)
        except Exception:
            value_int = 0
        value_int = max(0, min(500, value_int))
        self._max_raw_per_tile_value = value_int
        if isinstance(self._runtime_overrides, dict):
            self._runtime_overrides["max_raw_per_master_tile"] = value_int
        if isinstance(self._config_overrides, dict):
            # 0 = unlimited, keep the explicit cap so autosplit respects it.
            self._config_overrides["max_raw_per_master_tile"] = value_int
        self._debug_log(f"Max raw frames per master tile set to {value_int} (0 = unlimited).")
        self._persist_qt_filter_config()

    def _on_auto_angle_toggled(self, checked: bool) -> None:
        self._auto_angle_enabled = bool(checked)
        if self._auto_angle_enabled:
            self._runtime_overrides.pop("cluster_orientation_split_deg", None)
        else:
            self._runtime_overrides["cluster_orientation_split_deg"] = float(self._angle_split_value)

    def _on_angle_split_changed(self, value: float) -> None:
        sanitized = _sanitize_angle_value(value, ANGLE_SPLIT_DEFAULT_DEG)
        self._angle_split_value = sanitized
        if not self._auto_angle_enabled:
            self._runtime_overrides["cluster_orientation_split_deg"] = float(sanitized)

    def _refresh_instrument_options(self) -> None:
        combo = self._instrument_combo
        if combo is None:
            return
        instruments: set[str] = set()
        has_unknown = False
        for entry in self._normalized_items:
            instrument = (entry.instrument or "").strip()
            if instrument:
                instruments.add(instrument)
            else:
                has_unknown = True

        all_label = self._localizer.get("filter.instrument.all", "All")
        unknown_label = self._localizer.get("filter.instrument.unknown", "Unknown")

        current_data = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(all_label, None)
        for name in sorted(instruments):
            combo.addItem(name, name)
        if has_unknown:
            combo.addItem(unknown_label, self._instrument_unknown_token)
        combo.blockSignals(False)

        # Restore selection when possible
        index_to_restore = 0
        if current_data is not None:
            for idx in range(combo.count()):
                if combo.itemData(idx) == current_data:
                    index_to_restore = idx
                    break
        combo.setCurrentIndex(index_to_restore)

    def _apply_instrument_filter(self) -> None:
        combo = self._instrument_combo
        if combo is None:
            return
        target = combo.currentData()

        excluded = 0
        changed = False
        for row, entry in enumerate(self._normalized_items):
            instrument_label = (entry.instrument or "").strip()
            keep = True
            if target is None:
                keep = True
            elif target == self._instrument_unknown_token:
                keep = instrument_label == ""
            elif isinstance(target, str):
                keep = instrument_label.lower() == target.lower()
            if not keep:
                if self._set_entry_checked(row, False):
                    excluded += 1
                    changed = True
            else:
                if self._set_entry_checked(row, True):
                    changed = True
        if changed:
            self._after_selection_changed()
        if excluded:
            message = self._localizer.get(
                "filter.instrument.filtered",
                "Instrument filter excluded {count} frame(s).",
            )
            try:
                message = message.format(count=excluded)
            except Exception:
                pass
            self._append_log(message)
            self._status_label.setText(message)
        self._update_summary_label()
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()

    def _on_resolve_wcs_clicked(self) -> None:
        message = self._localizer.get("filter.scan.starting", "Starting analysis…")
        self._status_label.setText(message)
        self._append_log(message)
        self._on_run_analysis()

    def _on_manual_group_clicked(self) -> None:
        self._run_manual_master_tile_organisation()

    def _run_manual_master_tile_organisation(self) -> None:
        self._start_master_tile_organisation(mode="manual")

    def _on_auto_group_clicked(self) -> None:
        self._start_master_tile_organisation(mode="auto")

    def _start_master_tile_organisation(self, mode: str = "auto") -> None:
        """Common entry point for manual and auto master-tile generation."""

        normalized_mode = "manual" if str(mode).lower().startswith("manual") else "auto"
        if self._auto_group_running:
            return
        helpers_ready = _CLUSTER_CONNECTED is not None and _AUTOSPLIT_GROUPS is not None
        if not helpers_ready:
            message = self._localizer.get(
                "filter.cluster.helpers_missing",
                "Auto-organisation helpers are unavailable; install the worker module.",
            )
            self._append_log(message, level="WARN")
            self._status_label.setText(message)
            return

        self._set_group_buttons_enabled(False)

        selected_indices = self._collect_selected_indices()
        if not selected_indices:
            self._handle_auto_group_empty_selection()
            return
        filtered_indices = self._filter_indices_by_selection_bounds(selected_indices)
        if self._selection_bounds is not None:
            total = len(selected_indices)
            kept = len(filtered_indices)
            log_template = self._localizer.get(
                "qt_filter_bbox_log",
                "Bounding box active: kept {kept} of {total} frame(s) for auto-organize.",
            )
            try:
                log_message = log_template.format(kept=kept, total=total)
            except Exception:
                log_message = f"Bounding box active: kept {kept} / {total} frame(s) for auto-organize."
            self._append_log(log_message)
            if kept == 0:
                warning_text = self._localizer.get(
                    "qt_filter_bbox_empty_selection",
                    "No frames found inside the current selection bounding box.",
                )
                self._append_log(warning_text, level="WARN")
                self._status_label.setText(warning_text)
                try:
                    QMessageBox.warning(self, self.windowTitle() or "ZeMosaic Filter", warning_text)
                except Exception:
                    pass
                self._handle_auto_group_empty_selection()
                return
            selected_indices = filtered_indices

        log_key = "filter.cluster.manual_refresh"
        log_default = "Manual master-tile organisation requested."
        if normalized_mode == "auto":
            log_key = "filter.cluster.auto_refresh"
            log_default = "Auto master-tile organisation requested."
        self._auto_group_running = True
        self._append_log(self._localizer.get(log_key, log_default))
        self._status_label.setText(
            self._localizer.get("filter.cluster.running", "Preparing master-tile groups…")
        )
        self._start_auto_group_elapsed_timer(
            self._localizer.get("filter.cluster.running", "Preparing master-tile groups…")
        )

        optimize_flag = normalized_mode == "auto"
        max_raw_snapshot = int(getattr(self, "_max_raw_per_tile_value", 0) or 0)
        min_safe_snapshot = int(self._config_value("min_safe_stack", 3) or 3)
        target_stack_snapshot = int(self._config_value("target_stack", 5) or 5)
        overlap_snapshot = int(self._resolve_overlap_percent())
        try:
            thread = threading.Thread(
                target=self._auto_group_background_task,
                args=(
                    selected_indices,
                    int(self._resolve_overcap_percent()),
                    bool(self._coverage_first_enabled()),
                    normalized_mode,
                    optimize_flag,
                    max_raw_snapshot,
                    min_safe_snapshot,
                    target_stack_snapshot,
                    overlap_snapshot,
                ),
                daemon=True,
            )
            thread.start()
            self._show_processing_overlay()
        except Exception as exc:
            self._hide_processing_overlay()
            self._stop_auto_group_elapsed_timer()
            self._auto_group_running = False
            self._set_group_buttons_enabled(True)
            message = self._localizer.get(
                "filter.cluster.failed",
                "Auto-organisation failed: {error}",
            )
            try:
                message = message.format(error=exc)
            except Exception:
                message = f"Auto-organisation failed: {exc}"
            self._append_log(message, level="ERROR")
            self._status_label.setText(message)

    def _set_group_buttons_enabled(self, enabled: bool) -> None:
        for button in (self._auto_group_button, getattr(self, "_manual_group_button", None)):
            if button is None:
                continue
            try:
                button.setEnabled(bool(enabled))
            except Exception:
                pass

    def _start_auto_group_elapsed_timer(self, stage: str) -> None:
        self._auto_group_started_at = time.perf_counter()
        self._auto_group_stage_text = stage
        if self._auto_group_elapsed_timer is None:
            timer = QTimer(self)
            timer.setInterval(1000)
            timer.timeout.connect(self._update_auto_group_elapsed_label)
            self._auto_group_elapsed_timer = timer
        if self._auto_group_elapsed_timer is not None and not self._auto_group_elapsed_timer.isActive():
            self._auto_group_elapsed_timer.start()
        self._update_auto_group_elapsed_label(force=True)

    def _stop_auto_group_elapsed_timer(self) -> None:
        self._auto_group_started_at = None
        if self._auto_group_elapsed_timer is not None:
            try:
                self._auto_group_elapsed_timer.stop()
            except Exception:
                pass

    def _update_auto_group_elapsed_label(self, *, force: bool = False) -> None:
        if not getattr(self, "_auto_group_running", False):
            self._stop_auto_group_elapsed_timer()
            return
        if self._auto_group_started_at is None:
            if force:
                self._status_label.setText(self._auto_group_stage_text)
            return
        elapsed = max(0, int(time.perf_counter() - self._auto_group_started_at))
        stage_text = self._auto_group_stage_text or self._localizer.get(
            "filter.cluster.running",
            "Preparing master-tile groups…",
        )
        try:
            self._status_label.setText(f"{stage_text} (elapsed: {elapsed}s)")
        except Exception:
            self._status_label.setText(stage_text)

    @Slot(str)
    def _handle_auto_group_stage_update(self, stage: str) -> None:
        self._auto_group_stage_text = stage
        self._update_auto_group_elapsed_label(force=True)

    def _handle_auto_group_empty_selection(self) -> None:
        self._auto_group_running = False
        self._group_outline_bounds = []
        self._schedule_preview_refresh()
        summary_text = self._format_group_summary(0, "[]")
        self._append_log(summary_text)
        self._update_auto_group_summary_display(summary_text, summary_text)
        self._status_label.setText(summary_text)
        self._stop_auto_group_elapsed_timer()
        self._set_group_buttons_enabled(True)
        self._hide_processing_overlay()

    def _auto_group_background_task(
        self,
        selected_indices: list[int],
        overcap_pct: int,
        coverage_enabled_flag: bool,
        mode: str,
        optimize_flag: bool,
        max_raw_cap: int,
        min_safe_stack: int,
        target_stack_size: int,
        overlap_percent: int,
    ) -> None:
        normalized_mode = "manual" if str(mode).lower().startswith("manual") else "auto"
        messages: list[str | tuple[str, str]] = []
        result_payload: dict[str, Any] | None = None
        start_time = time.perf_counter()
        timings: dict[str, float] = {}
        try:
            try:
                self._auto_group_stage_signal.emit("Clustering frames…")
                self._async_log_signal.emit("Stage: clustering connected groups", "INFO")
                result_payload = self._compute_auto_groups(
                    selected_indices,
                    overcap_pct,
                    coverage_enabled_flag,
                    messages,
                )
                if isinstance(result_payload, dict):
                    timings = result_payload.get("timings") or {}
                    self._auto_group_stage_signal.emit("Post-processing groups…")
                    self._async_log_signal.emit("Stage: post-processing clusters", "INFO")
                    result_payload["mode"] = normalized_mode
                    result_payload["auto_optimised"] = bool(optimize_flag)
                    if optimize_flag:
                        self._auto_group_stage_signal.emit("Auto-optimiser…")
                        self._async_log_signal.emit("Stage: running auto-optimiser", "INFO")
                        t_start_optimiser = time.perf_counter()
                        self._optimize_auto_group_result(
                            result_payload,
                            max_raw_cap=int(max_raw_cap),
                            min_safe_stack=int(min_safe_stack),
                            target_stack_size=int(target_stack_size),
                            overlap_percent=int(overlap_percent),
                            messages=messages,
                        )
                        timings["auto_optimiser"] = time.perf_counter() - t_start_optimiser
            except Exception as exc:  # pragma: no cover - defensive guard
                error_text = self._localizer.get(
                    "filter.cluster.failed",
                    "Auto-organisation failed: {error}",
                )
                try:
                    formatted = error_text.format(error=exc)
                except Exception:
                    formatted = f"Auto-organisation failed: {exc}"
                messages.append((formatted, "ERROR"))
            if result_payload and result_payload.get("coverage_first"):
                groups_count = len(result_payload.get("final_groups") or [])
                messages.append(
                    self._format_message(
                        "log_covfirst_done",
                        "Coverage-first preplan ready: {N} groups written to overrides_state.preplan_master_groups",
                        N=groups_count,
                    )
                )
            elapsed_total = time.perf_counter() - start_time
            timings["total"] = elapsed_total
            timing_summary = " ".join(f"{k}={v:.2f}s" for k, v in timings.items())
            logger.info("[AutoGroupTiming] %s", timing_summary)
            messages.append(f"Timings: {timing_summary}")
            messages.append(
                self._format_message(
                    "auto_group_total_time",
                    "Auto-group completed in {DT:.1f}s",
                    DT=elapsed_total,
                )
            )
        finally:
            self._auto_group_stage_signal.emit("Finalizing…")
            self._async_log_signal.emit(f"Stage: finalize (elapsed {time.perf_counter() - start_time:.1f}s)", "INFO")
            self._auto_group_finished_signal.emit(result_payload, list(messages))

    @Slot(object, object)
    def _handle_auto_group_finished(
        self,
        result_payload: object,
        messages_payload: object,
    ) -> None:
        try:
            entries: list[Any] = []
            if isinstance(messages_payload, list):
                entries = list(messages_payload)
            elif messages_payload:
                entries = [messages_payload]
            for entry in entries:
                if isinstance(entry, tuple):
                    text = entry[0]
                    level = entry[1] if len(entry) > 1 else "INFO"
                    self._append_log(str(text), level=str(level))
                else:
                    self._append_log(str(entry))
            if not isinstance(result_payload, dict) or not result_payload:
                if not entries:
                    fallback = self._localizer.get(
                        "filter.cluster.failed_generic",
                        "Unable to prepare master-tile groups.",
                    )
                    self._append_log(fallback, level="WARN")
                self._status_label.setText(
                    self._localizer.get(
                        "filter.cluster.failed_short",
                        "Auto-organisation failed.",
                    )
                )
                return
            self._apply_auto_group_result(result_payload)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._append_log(f"Auto-group apply failed: {exc}", level="ERROR")
            self._status_label.setText(
                self._localizer.get("filter.cluster.failed_short", "Auto-organisation failed.")
            )
        finally:
            self._auto_group_running = False
            self._set_group_buttons_enabled(True)
            self._stop_auto_group_elapsed_timer()
            self._hide_processing_overlay()

    def _coerce_eqmode_mode(self, raw_value: Any) -> tuple[str | None, int | None]:
        if raw_value is None:
            return None, None
        value_int: int | None = None
        try:
            value_int = int(raw_value)
        except Exception:
            try:
                value_int = int(float(str(raw_value).strip()))
            except Exception:
                value_int = None
        if value_int == 1:
            return "EQ", 1
        if value_int == 0:
            return "ALT_AZ", 0
        if value_int is not None:
            return f"EQMODE_{value_int}", value_int
        return None, None

    def _extract_eqmode_from_entry(self, entry: dict[str, Any]) -> str | None:
        if not isinstance(entry, dict):
            return None
        existing_mount = entry.get("MOUNT_MODE")
        if isinstance(existing_mount, str):
            return existing_mount
        cached = entry.get("_eqmode_mode")
        if isinstance(cached, str):
            entry.setdefault("MOUNT_MODE", cached)
            return cached
        raw_value = entry.get("EQMODE")
        header = entry.get("header") or entry.get("header_subset")
        if raw_value is None:
            path_val = None
            for key in ("path", "path_raw", "path_preprocessed_cache", "file", "filename"):
                candidate = entry.get(key)
                if candidate:
                    path_val = candidate
                    break
            if header is None and path_val:
                header = self._load_header(str(path_val))
                if header is not None:
                    entry.setdefault("header", header)
            if header is not None:
                try:
                    raw_value = header.get("EQMODE")
                except Exception:
                    raw_value = None
        mode, _value_int = self._coerce_eqmode_mode(raw_value)
        mount_mode = mode
        if (not mount_mode or mount_mode == "UNKNOWN") and header is not None:
            mount_mode = _classify_mount_mode_from_header(header)
        if isinstance(mount_mode, str) and mount_mode:
            entry["MOUNT_MODE"] = mount_mode
        if mode:
            entry["_eqmode_mode"] = mode
        return entry.get("MOUNT_MODE")

    def _split_group_by_eqmode(
        self,
        group: list[dict[str, Any]],
        log_fn: Callable[[str], None] | None = None,
    ) -> list[list[dict[str, Any]]]:
        if not group:
            return [group]
        eq_count = 0
        altaz_count = 0
        unknown_count = 0
        for entry in group:
            mode = self._extract_eqmode_from_entry(entry)
            if mode == "EQ" or mode == "EQMODE_1":
                eq_count += 1
            elif mode == "ALT_AZ":
                altaz_count += 1
            else:
                unknown_count += 1
        subgroups = _split_group_by_mount_mode(group)
        if len(subgroups) > 1:
            message = (
                "eqmode_split: group mixed (EQ=%d ALT_AZ=%d UNKNOWN=%d) -> split"
                % (eq_count, altaz_count, unknown_count)
            )
            logger.info(message)
            if log_fn is not None:
                log_fn(message)
            return subgroups
        return [group]

    def _group_eqmode_signature(self, group: Sequence[dict[str, Any]]) -> str:
        has_eq = False
        has_altz = False
        for entry in group or []:
            mode = self._extract_eqmode_from_entry(entry)
            if mode == "EQ" or mode == "EQMODE_1":
                has_eq = True
            elif mode == "ALT_AZ":
                has_altz = True
        if has_eq and has_altz:
            return "MIXED"
        if has_eq:
            return "EQ"
        if has_altz:
            return "ALT_AZ"
        return "UNKNOWN"

    def _prefetch_eqmode_for_candidates(
        self,
        candidate_infos: Sequence[dict[str, Any]],
        messages: list[str | tuple[str, str]],
    ) -> dict[str, Any] | None:
        if not candidate_infos:
            return None

        start_time = time.perf_counter()

        def _resolve_entry_path(entry_obj: dict[str, Any]) -> str | None:
            for key in ("path", "path_raw", "path_preprocessed_cache", "file", "filename"):
                candidate = entry_obj.get(key)
                if candidate:
                    return str(candidate)
            return None

        def _stat_file(path_value: str) -> tuple[int | None, float | None]:
            size_val: int | None = None
            mtime_val: float | None = None
            try:
                size_val = int(ospath.getsize(path_value))
            except Exception:
                size_val = None
            try:
                mtime_val = float(ospath.getmtime(path_value))
            except Exception:
                mtime_val = None
            return size_val, mtime_val

        def _prime_entry_mode(entry_obj: dict[str, Any]) -> bool:
            cached_mode = entry_obj.get("_eqmode_mode")
            if isinstance(cached_mode, str):
                entry_obj.setdefault("MOUNT_MODE", cached_mode)
                return True
            existing_mount = entry_obj.get("MOUNT_MODE")
            if isinstance(existing_mount, str):
                mount_norm = existing_mount.strip().upper()
                if mount_norm and mount_norm != "UNKNOWN":
                    if mount_norm == "EQMODE_1":
                        entry_obj.setdefault("_eqmode_mode", "EQ")
                    elif mount_norm in ("EQ", "ALT_AZ"):
                        entry_obj.setdefault("_eqmode_mode", mount_norm)
                    entry_obj["MOUNT_MODE"] = mount_norm
                    return True
            mode_from_value, _mode_int = self._coerce_eqmode_mode(entry_obj.get("EQMODE"))
            if mode_from_value:
                entry_obj["_eqmode_mode"] = mode_from_value
                entry_obj.setdefault("MOUNT_MODE", mode_from_value)
                return True
            header_obj = entry_obj.get("header_subset") or entry_obj.get("header")
            if header_obj is not None:
                try:
                    header_value = header_obj.get("EQMODE")  # type: ignore[attr-defined]
                except Exception:
                    header_value = None
                mode_from_header, _ = self._coerce_eqmode_mode(header_value)
                if mode_from_header:
                    entry_obj["_eqmode_mode"] = mode_from_header
                    entry_obj.setdefault("MOUNT_MODE", mode_from_header)
                    return True
                mount_mode_header = _classify_mount_mode_from_header(header_obj)
                if mount_mode_header and mount_mode_header != "UNKNOWN":
                    entry_obj["MOUNT_MODE"] = mount_mode_header
                    return True
            return False

        def _resolve_cache_path(entries: Sequence[dict[str, Any]]) -> str | None:
            directories: list[str] = []
            for info in entries:
                if not isinstance(info, dict):
                    continue
                candidate_path = _resolve_entry_path(info)
                if not candidate_path:
                    continue
                try:
                    directories.append(ospath.dirname(ospath.abspath(candidate_path)))
                except Exception:
                    continue
            if not directories:
                return None
            common_dir: str | None = None
            try:
                common_dir = ospath.commonpath(directories)
            except Exception:
                common_dir = directories[0]
            if not common_dir:
                return None
            return ospath.join(common_dir, ".zemosaic_eqmode_cache.json")

        cache_path = _resolve_cache_path(candidate_infos)
        cache_store: dict[str, dict[str, Any]] = {}
        if cache_path and ospath.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    for key, payload in loaded.items():
                        if isinstance(key, str) and isinstance(payload, dict):
                            cache_store[key] = dict(payload)
            except Exception:
                cache_store = {}

        cache_dirty = False
        cache_hits = 0
        cache_miss = 0
        reads_header = 0
        worker_count = 0

        needs: list[dict[str, Any]] = []

        for entry in candidate_infos:
            if not isinstance(entry, dict):
                continue
            if _prime_entry_mode(entry):
                continue
            entry_path = _resolve_entry_path(entry)
            if not entry_path:
                continue
            norm_path = casefold_path(entry_path)
            size_val, mtime_val = _stat_file(entry_path)
            cache_entry = cache_store.get(norm_path)
            cache_valid = False
            mode_from_cache = None
            if cache_entry and size_val is not None and mtime_val is not None:
                try:
                    cached_size = int(cache_entry.get("size"))  # type: ignore[arg-type]
                except Exception:
                    cached_size = None
                try:
                    cached_mtime = float(cache_entry.get("mtime"))  # type: ignore[arg-type]
                except Exception:
                    cached_mtime = None
                if cached_size == size_val and cached_mtime == mtime_val:
                    cache_valid = True
                    eq_value = cache_entry.get("eqmode")
                    eq_int: int | None
                    try:
                        eq_int = int(eq_value)
                    except Exception:
                        eq_int = None
                    if eq_int == 1:
                        mode_from_cache = "EQ"
                    elif eq_int == 0:
                        mode_from_cache = "ALT_AZ"
                else:
                    cache_valid = False
            if cache_valid:
                cache_hits += 1
                if mode_from_cache:
                    entry["_eqmode_mode"] = mode_from_cache
                    entry.setdefault("MOUNT_MODE", mode_from_cache)
                continue
            cache_miss += 1
            if cache_entry and cache_path:
                cache_store.pop(norm_path, None)
                cache_dirty = True
            needs.append(
                {
                    "entry": entry,
                    "path": entry_path,
                    "key": norm_path,
                    "size": size_val,
                    "mtime": mtime_val,
                }
            )

        def _read_from_path(path_value: str) -> tuple[str | None, int | None]:
            return self._read_eqmode_from_path(path_value)

        if needs and fits is not None:
            worker_count = min(16, max(4, int(os.cpu_count() or 8)))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(_read_from_path, payload["path"]): payload for payload in needs
                }
                for future in as_completed(future_map):
                    payload = future_map[future]
                    reads_header += 1
                    mode_value: str | None = None
                    mode_int: int | None = None
                    try:
                        mode_value, mode_int = future.result()
                    except Exception:
                        mode_value = None
                        mode_int = None
                    if mode_value:
                        payload["entry"]["_eqmode_mode"] = mode_value
                        payload["entry"].setdefault("MOUNT_MODE", mode_value)
                    if (payload["size"] is None or payload["mtime"] is None) and payload["path"]:
                        payload["size"], payload["mtime"] = _stat_file(payload["path"])
                    if cache_path and payload["size"] is not None and payload["mtime"] is not None:
                        cache_store[payload["key"]] = {
                            "eqmode": mode_int if mode_int in (0, 1) else None,
                            "size": int(payload["size"]),
                            "mtime": float(payload["mtime"]),
                        }
                        cache_dirty = True

        if cache_path and cache_dirty:
            try:
                cache_dir = ospath.dirname(cache_path)
                if cache_dir and not ospath.isdir(cache_dir):
                    os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as handle:
                    json.dump(cache_store, handle)
            except Exception:
                pass

        eq_count = 0
        altaz_count = 0
        unknown_count = 0
        for entry in candidate_infos:
            if not isinstance(entry, dict):
                continue
            mode = entry.get("MOUNT_MODE") or entry.get("_eqmode_mode")
            if mode == "EQ" or mode == "EQMODE_1":
                eq_count += 1
            elif mode == "ALT_AZ":
                altaz_count += 1
            else:
                unknown_count += 1

        total_count = eq_count + altaz_count + unknown_count
        eqmode_summary = {
            "eq_count": int(eq_count),
            "altaz_count": int(altaz_count),
            "unknown_count": int(unknown_count),
            "total": int(total_count),
            "cache_hits": int(cache_hits),
            "cache_miss": int(cache_miss),
            "reads_header": int(reads_header),
            "source": "qt_prefetch_eqmode",
        }
        self._last_eqmode_summary = eqmode_summary

        duration = time.perf_counter() - start_time
        cache_label = cache_path or "none"
        summary = (
            "FILTER_EQMODE_SUMMARY: eq=%d altaz=%d unknown=%d total=%d "
            "(cache_hit=%d cache_miss=%d read=%d workers=%d dt=%.2fs cache=%s)"
            % (
                eq_count,
                altaz_count,
                unknown_count,
                total_count,
                cache_hits,
                cache_miss,
                reads_header,
                worker_count,
                duration,
                cache_label,
            )
        )
        logger.info(summary)
        if isinstance(messages, list):
            messages.append(summary)
        return eqmode_summary

    def _read_eqmode_from_path(self, path: str) -> tuple[str | None, int | None]:
        if not path or fits is None:
            return None, None
        raw_value: Any = None
        try:
            raw_value = fits.getval(path, "EQMODE", ext=0, ignore_missing_end=True)
        except Exception:
            try:
                header = fits.getheader(path, ext=0, ignore_missing_end=True)
            except Exception:
                header = None
            if header is not None:
                try:
                    raw_value = header.get("EQMODE")
                except Exception:
                    raw_value = None
        return self._coerce_eqmode_mode(raw_value)

    def _compute_auto_groups(
        self,
        selected_indices: list[int],
        overcap_pct: int,
        coverage_requested: bool,
        messages: list[str | tuple[str, str]],
    ) -> dict[str, Any]:
        timings: dict[str, float] = {}
        sds_mode = bool(self._sds_checkbox.isChecked()) if self._sds_checkbox is not None else False
        coverage_enabled = bool(coverage_requested)
        if sds_mode and compute_global_wcs_descriptor is not None:
            success, _meta = self._ensure_global_wcs_for_indices(selected_indices)
            if success:
                threshold = self._coerce_float(self._config_value("sds_coverage_threshold", 0.92), 0.92)
                threshold = max(0.10, min(0.99, threshold))
                min_batch_size = self._coerce_int(
                    self._config_value("sds_min_batch_size", 5),
                    default=5,
                ) or 5
                min_batch_size = max(1, int(min_batch_size))
                target_batch_size = self._coerce_int(
                    self._config_value("sds_target_batch_size", 10),
                    default=10,
                ) or 10
                target_batch_size = max(min_batch_size, int(target_batch_size))
                sds_groups = self._build_sds_batches_for_indices(
                    selected_indices,
                    coverage_threshold=threshold,
                    min_batch_size=min_batch_size,
                    target_batch_size=target_batch_size,
                )
                if sds_groups:
                    sizes = [len(group) for group in sds_groups]
                    log_text = self._format_message(
                        "filter.cluster.sds_preview_summary",
                        "SDS preview: thr={threshold:.2f}, min={min_size}, target={target_size} -> {count} batch(es) {sizes}.",
                        threshold=threshold,
                        min_size=min_batch_size,
                        target_size=target_batch_size,
                        count=len(sds_groups),
                        sizes=sizes,
                    )
                    messages.append(log_text)
                    try:
                        self._async_log_signal.emit(log_text, "INFO")
                    except Exception:
                        pass
                    messages.append(
                        self._format_message(
                            "filter.cluster.sds_ready",
                            "ZeSupaDupStack: prepared {count} coverage batch(es) for preview.",
                            count=len(sds_groups),
                        )
                    )
                    return {
                        "final_groups": sds_groups,
                        "sizes": sizes,
                        "coverage_first": True,
                        "threshold_used": 0.0,
                        "angle_split": 0.0,
                        "timings": timings,
                    }
                else:
                    messages.append(
                        (
                            self._localizer.get(
                                "filter.cluster.sds_no_batches",
                                "ZeSupaDupStack auto-group fallback: coverage batches could not be built.",
                            ),
                            "WARN",
                        )
                    )
            else:
                messages.append(
                    (
                        self._localizer.get(
                            "filter.cluster.sds_wcs_unavailable",
                            "ZeSupaDupStack auto-group fallback: global WCS descriptor unavailable.",
                        ),
                        "WARN",
                    )
                )
        eqmode_summary: dict[str, Any] | None = None
        t_start = time.perf_counter()
        candidate_infos: list[dict[str, Any]] = []
        coord_samples: list[tuple[float, float]] = []
        for idx in selected_indices:
            if idx < 0 or idx >= len(self._normalized_items):
                continue
            payload, coords = self._build_candidate_payload(self._normalized_items[idx])
            if payload is None:
                continue
            candidate_infos.append(payload)
            if coords and None not in coords:
                coord_samples.append((float(coords[0]), float(coords[1])))
        timings["build_candidates"] = time.perf_counter() - t_start

        if not candidate_infos:
            raise RuntimeError("No candidate entries were usable for grouping.")
        if WCS is None and SkyCoord is None:
            raise RuntimeError("Astropy is required to compute WCS-based clusters.")

        t_start = time.perf_counter()
        eqmode_summary = self._prefetch_eqmode_for_candidates(candidate_infos, messages)
        timings["prefetch_eqmode"] = time.perf_counter() - t_start

        threshold_override = self._resolve_cluster_threshold_override()
        threshold_heuristic = self._estimate_threshold_from_coords(coord_samples)
        threshold_initial = threshold_override if threshold_override and threshold_override > 0 else threshold_heuristic
        if threshold_initial <= 0:
            threshold_initial = 0.1

        auto_angle_enabled = getattr(self, "_auto_angle_enabled", True)
        manual_angle_mode = not auto_angle_enabled
        orientation_threshold = self._resolve_orientation_split_threshold()
        angle_split_candidate = self._resolve_angle_split_candidate()
        auto_angle_detect = self._resolve_auto_angle_detect_threshold()
        orientation_threshold_worker = float(
            max(0.0, orientation_threshold if manual_angle_mode else 0.0)
        )
        cap_effective, min_cap = self._resolve_autosplit_caps()
        overcap_pct = max(0, min(50, int(overcap_pct)))
        if coverage_enabled:
            messages.append(
                self._format_message(
                    "log_covfirst_start",
                    "Coverage-first clustering: start (threshold={TH} deg, cap={CAP}, min_cap={MIN})",
                    TH=f"{threshold_initial:.3f}",
                    CAP=int(cap_effective),
                    MIN=int(min_cap),
                )
            )

        t_start = time.perf_counter()
        groups_initial = _CLUSTER_CONNECTED(
            candidate_infos,
            float(threshold_initial),
            None,
            orientation_split_threshold_deg=orientation_threshold_worker,
        )
        timings["clustering"] = time.perf_counter() - t_start
        if not groups_initial:
            raise RuntimeError("Worker clustering returned no groups.")

        threshold_used = float(threshold_initial)
        groups_used: list[list[dict[str, Any]]] = list(groups_initial)
        candidate_count = len(candidate_infos)
        ratio = (len(groups_initial) / float(candidate_count)) if candidate_count else 0.0
        pathological = candidate_count > 0 and (
            len(groups_initial) >= candidate_count or ratio >= 0.6
        )
        if coverage_enabled and pathological and len(coord_samples) >= 5:
            p90_value: float | None = None
            if SkyCoord is not None and u is not None:
                try:
                    sc = SkyCoord(
                        ra=[c[0] for c in coord_samples] * u.deg,
                        dec=[c[1] for c in coord_samples] * u.deg,
                        frame="icrs",
                    )
                    sep_matrix = sc[:, None].separation(sc[None, :]).deg
                    if sep_matrix.size:
                        with np.errstate(invalid="ignore"):
                            np.fill_diagonal(sep_matrix, np.nan)
                        nearest = np.nanmin(sep_matrix, axis=1)
                        finite = nearest[np.isfinite(nearest) & (nearest > 0)]
                        if finite.size:
                            p90_value = float(np.nanpercentile(finite, 90))
                except Exception:
                    p90_value = None
            if p90_value and math.isfinite(p90_value):
                threshold_relaxed = max(threshold_used, float(p90_value) * 1.1)
                if threshold_relaxed > threshold_used * 1.001:
                    t_start = time.perf_counter()
                    relaxed_groups = _CLUSTER_CONNECTED(
                        candidate_infos,
                        float(threshold_relaxed),
                        None,
                        orientation_split_threshold_deg=orientation_threshold_worker,
                    )
                    timings["clustering"] += time.perf_counter() - t_start
                    if relaxed_groups and len(relaxed_groups) < len(groups_initial):
                        messages.append(
                            self._format_message(
                                "log_covfirst_relax",
                                "Relaxed epsilon using P90 NN: old={OLD} deg -> new={NEW} deg",
                                OLD=f"{threshold_used:.3f}",
                                NEW=f"{threshold_relaxed:.3f}",
                            )
                        )
                        groups_used = relaxed_groups
                        threshold_used = threshold_relaxed

        mode_splits = 0
        mode_guarded: list[list[dict[str, Any]]] = []
        for group in groups_used:
            subgroups = _split_group_by_mount_mode(group)
            if len(subgroups) > 1:
                mode_splits += 1
            mode_guarded.extend(subgroups)
        if mode_splits > 0:
            msg_text = self._format_message(
                "filter_log_mount_mode_split",
                "Mount-mode guard: split {N} group(s) by EQMODE / MOUNT_MODE.",
                N=int(mode_splits),
            )
            messages.append(msg_text)
            try:
                self._async_log_signal.emit(msg_text, "INFO")
            except Exception:
                pass
            logger.info(
                "Mount-mode guard: split %d group(s) by EQMODE / MOUNT_MODE.",
                mode_splits,
            )
        groups_used = mode_guarded

        altaz_ori_split_enabled = (
            auto_angle_enabled
            and not manual_angle_mode
            and angle_split_candidate > 0.0
        )
        if altaz_ori_split_enabled:
            altaz_in = 0
            altaz_out = 0
            split_groups: list[list[dict[str, Any]]] = []
            for group in groups_used:
                mode_norm = "UNKNOWN"
                if group:
                    try:
                        mode_raw = group[0].get("MOUNT_MODE", "UNKNOWN")
                    except Exception:
                        mode_raw = "UNKNOWN"
                    try:
                        mode_norm = str(mode_raw).strip().upper()
                    except Exception:
                        mode_norm = "UNKNOWN"
                if mode_norm == "EQMODE_1":
                    mode_norm = "EQ"
                if mode_norm == "ALT_AZ":
                    altaz_in += 1
                    subgroups = split_clusters_by_orientation(
                        [group],
                        float(angle_split_candidate),
                        pa_key="PA_DEG",
                    )
                    if not subgroups:
                        subgroups = [group]
                    altaz_out += len(subgroups)
                    split_groups.extend(subgroups)
                else:
                    split_groups.append(group)
            groups_used = split_groups
            if altaz_in > 0 and altaz_out > altaz_in:
                msg = (
                    f"[ALT-AZ ORI_SPLIT] {altaz_in} clusters → {altaz_out} subclusters "
                    f"(threshold={float(angle_split_candidate):.1f}°)"
                )
                messages.append(msg)
                try:
                    self._async_log_signal.emit(msg, "INFO")
                except Exception:
                    pass
                logger.info("%s", msg)

        angle_split_effective = float(orientation_threshold if orientation_threshold > 0 else 0.0)
        if manual_angle_mode:
            angle_split_effective = angle_split_candidate

        auto_angle_triggered = False
        if (
            auto_angle_enabled
            and not manual_angle_mode
            and not altaz_ori_split_enabled
            and angle_split_effective <= 0.0
            and angle_split_candidate > 0.0
            and _tk_split_group_by_orientation is not None
            and _tk_circular_dispersion_deg is not None
        ):
            oriented_groups: list[list[dict[str, Any]]] = []
            triggered = 0
            for group in groups_used:
                pa_values: list[float] = []
                for info in group or []:
                    try:
                        pa = info.get("PA_DEG")
                        if pa is None:
                            continue
                        pa_values.append(float(pa))
                    except Exception:
                        continue
                if not pa_values:
                    oriented_groups.append(group)
                    continue
                dispersion_val = _tk_circular_dispersion_deg(pa_values)
                if dispersion_val > auto_angle_detect:
                    oriented_groups.extend(_tk_split_group_by_orientation(group, angle_split_candidate))
                    triggered += 1
                else:
                    oriented_groups.append(group)
            if triggered > 0:
                auto_angle_triggered = True
                angle_split_effective = angle_split_candidate
                groups_used = oriented_groups
                messages.append(
                    self._format_message(
                        "filter_log_orientation_autosplit",
                        "Orientation auto-split enabled: threshold={TH}° for {N} group(s)",
                        TH=f"{angle_split_effective:.1f}",
                        N=triggered,
                    )
                )
                self._update_orientation_override_value(angle_split_effective)

        if not auto_angle_triggered:
            self._update_orientation_override_value(angle_split_effective)

        groups_after_autosplit = groups_used
        final_groups: list[list[dict[str, Any]]]
        overlap_pct = self._resolve_overlap_percent()
        overlap_fraction = max(0.0, min(0.7, float(overlap_pct) / 100.0))
        t_start_autosplit = time.perf_counter()
        if cap_effective > 0:
            if overlap_fraction <= 0.0 and _AUTOSPLIT_GROUPS is not None:
                groups_after_autosplit = _AUTOSPLIT_GROUPS(
                    groups_used,
                    cap=int(max(1, cap_effective)),
                    min_cap=int(max(1, min_cap)),
                    progress_callback=None,
                )
            else:
                groups_after_autosplit = []
                for group in groups_used:
                    ordered_group = self._sort_group_for_overlap(group)
                    batches = self._make_overlapping_batches(
                        ordered_group,
                        cap=int(max(1, cap_effective)),
                        overlap_fraction=overlap_fraction,
                        min_cap=int(max(1, min_cap)),
                    )
                    if batches:
                        groups_after_autosplit.extend(batches)
                if not groups_after_autosplit:
                    groups_after_autosplit = groups_used
            if coverage_enabled:
                messages.append(
                    self._format_message(
                        "log_covfirst_autosplit",
                        "Autosplit applied: cap={CAP}, min_cap={MIN}, overlap={OVER}%, groups_in={IN}, groups_out={OUT}",
                        CAP=int(cap_effective),
                        MIN=int(min_cap),
                        OVER=int(overlap_pct),
                        IN=len(groups_used),
                        OUT=len(groups_after_autosplit),
                    )
                )

            cap_allowance = max(int(cap_effective), int(cap_effective * (1 + overcap_pct / 100.0)))
            max_dispersion = None
            if _COMPUTE_MAX_SEPARATION is not None and threshold_used > 0:
                max_dispersion = float(threshold_used) * 1.05

            merge_fn = _tk_merge_small_groups or (
                lambda groups, min_size, cap, **_: groups  # type: ignore[misc]
            )
            final_groups = []
            signature_buckets: dict[str, list[list[dict[str, Any]]]] = {}
            for group in groups_after_autosplit:
                signature = self._group_eqmode_signature(group)
                signature_buckets.setdefault(signature, []).append(group)
            for grouped in signature_buckets.values():
                final_groups.extend(
                    merge_fn(
                        grouped,
                        min_size=int(max(1, min_cap)),
                        cap=int(max(1, cap_effective)),
                        cap_allowance=cap_allowance,
                        compute_dispersion=_COMPUTE_MAX_SEPARATION,
                        max_dispersion_deg=max_dispersion,
                        log_fn=messages.append,
                    )
                )
            if coverage_enabled:
                messages.append(
                    self._format_message(
                        "log_covfirst_merge",
                        "Merged small groups with over-cap allowance={ALLOW}%, final_groups={N}",
                        ALLOW=int(overcap_pct),
                        N=len(final_groups),
                    )
                )
        else:
            if coverage_enabled:
                messages.append(
                    self._format_message(
                        "log_covfirst_autosplit_unlimited",
                        "Autosplit skipped: cap=0 (unlimited), groups_in={IN}",
                        IN=len(groups_used),
                    )
                )
            final_groups = groups_after_autosplit
        timings["autosplit"] = time.perf_counter() - t_start_autosplit

        self._log_batching_summary(
            final_groups,
            cap=int(cap_effective),
            overlap_fraction=overlap_fraction if cap_effective > 0 else 0.0,
            messages=messages,
        )

        for group in final_groups:
            for info in group:
                if info.pop("_fallback_wcs_used", False):
                    info.pop("wcs", None)
        
        t_start_borrow = time.perf_counter()
        if coverage_enabled:
            try:
                self._auto_group_stage_signal.emit("Borrowing v1…")
                self._async_log_signal.emit("Stage: borrowing coverage batches", "INFO")
            except Exception:
                pass
            final_groups, _borrow_stats = _apply_borrowing_per_mount_mode(
                final_groups,
                logger=logger,
            )
            borrowed_unique = 0
            borrowed_total = 0
            if isinstance(_borrow_stats, dict):
                try:
                    borrowed_unique = int(_borrow_stats.get("borrowed_unique_images", 0))
                except Exception:
                    borrowed_unique = 0
                try:
                    borrowed_total = int(_borrow_stats.get("borrowed_total_assignments", 0))
                except Exception:
                    borrowed_total = 0
            logger.info(
                "Borrowing v1 applied: groups=%d borrowed_unique=%d borrowed_total=%d",
                len(final_groups) if isinstance(final_groups, list) else 0,
                borrowed_unique,
                borrowed_total,
            )
        timings["borrowing"] = time.perf_counter() - t_start_borrow

        result = {
            "final_groups": final_groups,
            "sizes": [len(group) for group in final_groups],
            "coverage_first": bool(coverage_enabled),
            "threshold_used": threshold_used,
            "angle_split": angle_split_effective,
        }
        if isinstance(eqmode_summary, dict):
            result["eqmode_summary"] = eqmode_summary
        result["timings"] = timings
        return result

    def _optimize_auto_group_result(
        self,
        payload: dict[str, Any],
        *,
        max_raw_cap: int,
        min_safe_stack: int,
        target_stack_size: int,
        overlap_percent: int,
        messages: list[str | tuple[str, str]] | None = None,
    ) -> None:
        groups = payload.get("final_groups")
        if not isinstance(groups, list) or not groups:
            return
        max_cap = max(0, int(max_raw_cap))
        min_safe = max(1, int(min_safe_stack))
        target_stack = max(min_safe, int(target_stack_size))
        overlap_fraction = max(0.0, min(0.7, float(overlap_percent) / 100.0))
        threshold = payload.get("threshold_used")
        try:
            threshold_float = float(threshold)
        except Exception:
            threshold_float = 0.0
        if not math.isfinite(threshold_float) or threshold_float <= 0:
            threshold_float = 0.18
        dispersion_limit = max(0.15, threshold_float * 1.5)
        cap_limit = max_cap if max_cap > 0 else None
        desired_target = cap_limit if cap_limit else target_stack
        path_map = self._build_path_to_entry_map()
        records = self._build_auto_group_records(groups, path_map)
        if not records:
            return
        sizes_text = ", ".join(str(len(group)) for group in groups if isinstance(group, list))
        if messages is not None:
            messages.append(
                self._format_message(
                    "auto_optimiser_start",
                    "[AutoOptimiser] start cap={CAP}, min_safe={MIN}, target={TARGET}, overlap={OVER}% (sizes={SIZES})",
                    CAP=int(max_cap),
                    MIN=int(min_safe),
                    TARGET=int(target_stack),
                    OVER=int(overlap_fraction * 100),
                    SIZES=sizes_text or "[]",
                )
            )

        records, merge_count = self._merge_group_records_for_auto(
            records,
            cap_limit=cap_limit,
            dispersion_limit=dispersion_limit,
            desired_target=desired_target,
        )
        merged_groups = [record["entries"] for record in records]
        split_groups, split_count = self._split_large_groups_for_auto(
            merged_groups,
            cap_limit=cap_limit,
            overlap_fraction=overlap_fraction,
            min_cap=min_safe,
        )
        final_records = self._build_auto_group_records(split_groups, path_map)
        final_records, merge_after_split = self._merge_group_records_for_auto(
            final_records,
            cap_limit=cap_limit,
            dispersion_limit=dispersion_limit,
            desired_target=desired_target,
        )
        optimized_groups = [
            record["entries"] for record in final_records if isinstance(record.get("entries"), list) and record["entries"]
        ]
        if not optimized_groups:
            optimized_groups = merged_groups
        payload["final_groups"] = optimized_groups
        payload["sizes"] = [len(group) for group in optimized_groups]

        if messages is not None:
            final_sizes = ", ".join(str(len(group)) for group in optimized_groups) or "[]"
            messages.append(
                self._format_message(
                    "auto_optimiser_result",
                    "[AutoOptimiser] final groups={GROUPS}, merges={MERGES}, splits={SPLITS}, sizes={SIZES}",
                    GROUPS=len(optimized_groups),
                    MERGES=int(merge_count + merge_after_split),
                    SPLITS=int(split_count),
                    SIZES=final_sizes,
                )
            )

    def _build_path_to_entry_map(self) -> dict[str, _NormalizedItem]:
        path_map: dict[str, _NormalizedItem] = {}
        for entry in self._normalized_items:
            if entry.file_path:
                path_map[casefold_path(entry.file_path)] = entry
        return path_map

    def _build_auto_group_records(
        self,
        groups: list[list[dict[str, Any]]],
        path_map: dict[str, _NormalizedItem],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for group in groups or []:
            if not group:
                continue
            coords = self._gather_group_coordinates(group, path_map)
            center, radius = self._compute_center_and_radius(coords)
            dispersion = self._compute_dispersion_for_coords(coords)
            eqmode_sig = self._group_eqmode_signature(group)
            records.append(
                {
                    "entries": list(group),
                    "coords": coords,
                    "center": center,
                    "radius": radius,
                    "size": len(group),
                    "dispersion": dispersion,
                    "eqmode_sig": eqmode_sig,
                }
            )
        return records

    def _gather_group_coordinates(
        self,
        group: list[dict[str, Any]],
        path_map: dict[str, _NormalizedItem],
    ) -> list[tuple[float, float]]:
        coords: list[tuple[float, float]] = []
        for info in group or []:
            try:
                center_ra, center_dec = self._resolve_group_entry_center(info, path_map)
            except Exception:
                center_ra = center_dec = None
            if center_ra is not None and center_dec is not None:
                try:
                    ra_val = float(center_ra)
                    dec_val = float(center_dec)
                except Exception:
                    ra_val = dec_val = None
                if ra_val is not None and dec_val is not None and math.isfinite(ra_val) and math.isfinite(dec_val):
                    coords.append((ra_val, dec_val))
                    continue
            ra_raw = None
            dec_raw = None
            if isinstance(info, dict):
                ra_raw = info.get("RA")
                dec_raw = info.get("DEC")
            if ra_raw is None or dec_raw is None:
                continue
            try:
                ra_val = float(ra_raw)
                dec_val = float(dec_raw)
            except Exception:
                continue
            if math.isfinite(ra_val) and math.isfinite(dec_val):
                coords.append((ra_val, dec_val))
        return coords

    @staticmethod
    def _compute_center_from_coords(coords: list[tuple[float, float]]) -> tuple[float, float] | None:
        if not coords:
            return None
        sum_ra = sum(point[0] for point in coords)
        sum_dec = sum(point[1] for point in coords)
        count = len(coords)
        if count <= 0:
            return None
        return (sum_ra / count, sum_dec / count)

    def _compute_center_and_radius(
        self,
        coords: list[tuple[float, float]],
    ) -> tuple[tuple[float, float] | None, float]:
        center = self._compute_center_from_coords(coords)
        if center is None:
            return None, 0.0
        if not coords:
            return center, 0.0
        max_dist_sq = 0.0
        for point in coords:
            dist_sq = (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
        return center, math.sqrt(max_dist_sq)

    def _compute_dispersion_for_coords(self, coords: list[tuple[float, float]]) -> float:
        if not coords or len(coords) < 2:
            return 0.0
        if _COMPUTE_MAX_SEPARATION is not None:
            try:
                dispersion = float(_COMPUTE_MAX_SEPARATION(coords))
                if math.isfinite(dispersion):
                    return dispersion
            except Exception:
                pass
        return self._approximate_dispersion(coords)

    @staticmethod
    def _angular_distance(
        a: tuple[float, float] | None,
        b: tuple[float, float] | None,
    ) -> float:
        if a is None or b is None:
            return float("inf")
        dra = float(a[0]) - float(b[0])
        ddec = float(a[1]) - float(b[1])
        return math.hypot(dra, ddec)

    def _merge_group_records_for_auto(
        self,
        records: list[dict[str, Any]],
        *,
        cap_limit: int | None,
        dispersion_limit: float,
        desired_target: int,
    ) -> tuple[list[dict[str, Any]], int]:
        if not records:
            return records, 0
        desired_target = max(1, int(desired_target))
        merges = 0
        max_iterations = max(1, len(records) * 4)
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            merged_this_round = False
            records.sort(key=lambda rec: int(rec.get("size", 0)))
            for idx in range(len(records)):
                partner = self._find_auto_merge_partner(
                    records,
                    idx,
                    cap_limit=cap_limit,
                    dispersion_limit=dispersion_limit,
                    desired_target=desired_target,
                )
                if partner is None:
                    continue
                partner_idx, coords, dispersion_upper = partner
                keep_idx = min(idx, partner_idx)
                drop_idx = max(idx, partner_idx)
                keep = records[keep_idx]
                drop = records[drop_idx]
                combined_entries = list(keep.get("entries") or []) + list(drop.get("entries") or [])
                if not combined_entries:
                    continue
                if cap_limit is not None and len(combined_entries) > cap_limit:
                    continue

                center, radius = self._compute_center_and_radius(coords or [])
                keep["entries"] = combined_entries
                keep["coords"] = coords or []
                keep["center"] = center
                keep["radius"] = radius
                keep["size"] = len(combined_entries)
                keep["dispersion"] = dispersion_upper
                records.pop(drop_idx)
                merges += 1
                merged_this_round = True
                break
            if not merged_this_round:
                break
        return records, merges

    def _find_auto_merge_partner(
        self,
        records: list[dict[str, Any]],
        idx: int,
        *,
        cap_limit: int | None,
        dispersion_limit: float,
        desired_target: int,
    ) -> tuple[int, list[tuple[float, float]], float] | None:
        if idx < 0 or idx >= len(records):
            return None
        source = records[idx]
        source_coords = list(source.get("coords") or [])
        source_center = source.get("center")
        source_sig = str(source.get("eqmode_sig") or "UNKNOWN")
        source_radius = float(source.get("radius", 0.0))
        source_dispersion = float(source.get("dispersion", 0.0))

        if not source_coords or source_center is None:
            return None

        source_size = int(source.get("size", len(source_coords)))
        neighbors: list[tuple[float, int, int]] = []
        for other_idx, other in enumerate(records):
            if other_idx == idx:
                continue
            other_sig = str(other.get("eqmode_sig") or "UNKNOWN")
            if other_sig != source_sig:
                continue

            other_coords = other.get("coords") or []
            other_center = other.get("center")
            if not other_coords or other_center is None:
                continue

            other_size = int(other.get("size", len(other_coords)))
            combined_size = source_size + other_size
            if cap_limit is not None and combined_size > cap_limit:
                continue

            distance = self._angular_distance(source_center, other_center)
            neighbors.append((distance, other_idx, combined_size))

        if not neighbors:
            return None

        neighbors.sort(key=lambda item: item[0])
        neighbor_cap = self._config_value("auto_optimiser_neighbor_cap", 64)
        if neighbor_cap > 0:
            neighbors = neighbors[: int(neighbor_cap)]

        best_idx: int | None = None
        best_coords: list[tuple[float, float]] | None = None
        best_dispersion_upper = 0.0
        best_score = float("inf")
        best_distance = float("inf")

        for distance, other_idx, combined_size in neighbors:
            other = records[other_idx]
            other_radius = float(other.get("radius", 0.0))
            other_dispersion = float(other.get("dispersion", 0.0))

            # Estimate dispersion upper bound without full computation
            cross_upper = distance + source_radius + other_radius
            disp_upper = max(source_dispersion, other_dispersion, cross_upper)

            if dispersion_limit > 0 and disp_upper > dispersion_limit:
                continue

            size_score = abs(desired_target - combined_size)
            if (
                size_score < best_score - 1e-6
                or (abs(size_score - best_score) <= 1e-6 and disp_upper < best_dispersion_upper - 1e-6)
                or (
                    abs(size_score - best_score) <= 1e-6
                    and abs(disp_upper - best_dispersion_upper) <= 1e-6
                    and distance < best_distance
                )
            ):
                combined_coords = source_coords + list(other.get("coords") or [])
                if not combined_coords:
                    continue

                best_idx = other_idx
                best_coords = combined_coords
                best_dispersion_upper = disp_upper
                best_score = size_score
                best_distance = distance

        if best_idx is None or best_coords is None:
            return None

        return best_idx, best_coords, best_dispersion_upper

    def _split_large_groups_for_auto(
        self,
        groups: list[list[dict[str, Any]]],
        *,
        cap_limit: int | None,
        overlap_fraction: float,
        min_cap: int,
    ) -> tuple[list[list[dict[str, Any]]], int]:
        if not groups:
            return [], 0
        if cap_limit is None or cap_limit <= 0:
            return [list(group) for group in groups], 0
        overlap_fraction = max(0.0, min(0.7, float(overlap_fraction)))
        min_cap_effective = max(1, int(min_cap))
        split_groups: list[list[dict[str, Any]]] = []
        split_count = 0
        for group in groups:
            entries = list(group)
            if len(entries) <= cap_limit:
                split_groups.append(entries)
                continue
            ordered = self._sort_group_for_overlap(entries)
            batches = self._make_overlapping_batches(
                ordered,
                cap=int(cap_limit),
                overlap_fraction=overlap_fraction,
                min_cap=min_cap_effective,
            )
            if batches:
                split_groups.extend(batches)
                split_count += max(0, len(batches) - 1)
            else:
                split_groups.append(entries)
        return split_groups, split_count

    def _apply_auto_group_result(self, payload: dict[str, Any]) -> None:
        self._last_auto_group_result = payload if isinstance(payload, dict) else None
        eqmode_summary_payload = payload.get("eqmode_summary") if isinstance(payload, dict) else None
        if isinstance(eqmode_summary_payload, dict):
            self._last_eqmode_summary = eqmode_summary_payload
        groups = payload.get("final_groups") or []
        if not isinstance(groups, list):
            groups = []
        self._auto_group_override_groups = groups if groups else None
        self._cluster_threshold_used = payload.get("threshold_used")

        assignment: dict[str, tuple[int, str]] = {}
        label_template = self._localizer.get("filter.preview.group_label", "Group {index}")
        for idx, group in enumerate(groups, start=1):
            try:
                label = label_template.format(index=idx)
            except Exception:
                label = f"Group {idx}"
            for entry in group:
                path = entry.get("path") or entry.get("file_path")
                if not path:
                    continue
                assignment[casefold_path(path)] = (idx - 1, label)

        for normalized in self._normalized_items:
            key = casefold_path(normalized.file_path or "")
            if key in assignment:
                group_idx, label = assignment[key]
                normalized.cluster_index = group_idx
                normalized.group_label = label
            else:
                normalized.cluster_index = None
                normalized.group_label = None

        new_groups: list[list[_NormalizedItem]] = [[] for _ in groups]
        if new_groups:
            for normalized in self._normalized_items:
                key = casefold_path(normalized.file_path or "")
                if key in assignment:
                    group_idx = assignment[key][0]
                    if 0 <= group_idx < len(new_groups):
                        new_groups[group_idx].append(normalized)
        self._cluster_groups = [group for group in new_groups if group]

        sizes_payload = payload.get("sizes")
        sizes: list[int] = []
        if isinstance(sizes_payload, list):
            for val in sizes_payload:
                try:
                    size_int = int(val)
                except Exception:
                    continue
                if size_int > 0:
                    sizes.append(size_int)
        if not sizes and groups:
            try:
                sizes = [len(group) for group in groups]
            except Exception:
                sizes = []

        if groups:
            self._status_label.setText(
                self._localizer.get(
                    "filter.cluster.complete",
                    "Master-tile organisation complete.",
                )
            )
        else:
            self._status_label.setText(
                self._localizer.get(
                    "filter.cluster.no_groups",
                    "No master-tile groups could be prepared.",
                )
            )

        # Summarise group sizes both in the log (full list) and next to the
        # button (compact histogram) exactly like the Tk dialog.
        hist_text = _format_sizes_histogram(sizes) if sizes else "[]"
        if sizes:
            full_sizes_text = ", ".join(str(val) for val in sizes)
        else:
            full_sizes_text = "[]"
        log_summary = self._format_group_summary(len(groups), full_sizes_text)
        self._append_log(log_summary)
        label_summary = self._format_group_summary(len(groups), hist_text)
        self._update_auto_group_summary_display(label_summary, log_summary)

        self._group_outline_bounds = self._compute_group_outline_bounds(groups)
        if self._coverage_axes is not None and self._coverage_canvas is not None:
            try:
                self._update_coverage_plot(groups, bool(payload.get("coverage_first")))
            except Exception:
                pass
        self._update_summary_label()
        self._schedule_preview_refresh()

    def _build_candidate_payload(
        self,
        entry: _NormalizedItem,
    ) -> tuple[dict[str, Any] | None, tuple[float | None, float | None] | None]:
        payload: dict[str, Any] = {}
        original = entry.original
        if isinstance(original, dict):
            payload.update(original)
        path = entry.file_path
        if not path:
            return None, None
        payload.setdefault("path", path)
        if original and isinstance(original, dict):
            raw = original.get("path_raw")
            if raw:
                payload.setdefault("path_raw", raw)
        header = payload.get("header") or payload.get("header_subset")
        if header is None:
            header_cache = getattr(entry, "header_cache", None)
            if header_cache is not None:
                header = header_cache
                payload["header"] = header_cache
        if header is None:
            header = self._load_header(path)
            if header is not None:
                payload["header"] = header
        if "MOUNT_MODE" not in payload:
            payload["MOUNT_MODE"] = _classify_mount_mode_from_header(header)
        wcs_obj = payload.get("wcs")
        if wcs_obj is None:
            wcs_cache = getattr(entry, "wcs_cache", None)
            if wcs_cache is not None:
                wcs_obj = wcs_cache
                payload["wcs"] = wcs_cache
        if wcs_obj is None and header is not None and WCS is not None:
            try:
                wcs_obj = _build_wcs_from_header(header)
            except Exception:
                wcs_obj = None
            if wcs_obj is not None:
                payload["wcs"] = wcs_obj

        pa_value: float | None = None
        try:
            existing_pa = payload.get("PA_DEG")
            if existing_pa is not None and existing_pa != "":
                pa_value = float(existing_pa)
                if not math.isfinite(pa_value):
                    pa_value = None
        except Exception:
            pa_value = None

        if pa_value is None:
            try:
                def _pa_from_components(cd11: float | None, cd21: float | None) -> float | None:
                    if cd11 is None or cd21 is None:
                        return None
                    if not (math.isfinite(cd11) and math.isfinite(cd21)):
                        return None
                    if cd11 == 0.0 and cd21 == 0.0:
                        return None
                    pa = (math.degrees(math.atan2(cd21, cd11)) % 360.0)
                    return float(pa) if math.isfinite(pa) else None

                def _coerce_float(val: Any) -> float | None:
                    try:
                        parsed = float(val)
                    except Exception:
                        return None
                    return parsed if math.isfinite(parsed) else None

                wcs_inner = getattr(wcs_obj, "wcs", None) if wcs_obj is not None else None
                if wcs_inner is not None:
                    cd_mat = getattr(wcs_inner, "cd", None)
                    if cd_mat is not None:
                        try:
                            pa_value = _pa_from_components(
                                _coerce_float(cd_mat[0][0]),
                                _coerce_float(cd_mat[1][0]),
                            )
                        except Exception:
                            pa_value = None
                    if pa_value is None:
                        pc_mat = getattr(wcs_inner, "pc", None)
                        cdelt_vec = getattr(wcs_inner, "cdelt", None)
                        if pc_mat is not None:
                            try:
                                cdelt1 = _coerce_float(cdelt_vec[0]) if cdelt_vec is not None else None
                                cdelt2 = _coerce_float(cdelt_vec[1]) if cdelt_vec is not None else None
                                if cdelt1 is None:
                                    cdelt1 = 1.0
                                if cdelt2 is None:
                                    cdelt2 = 1.0
                                pa_value = _pa_from_components(
                                    _coerce_float(pc_mat[0][0]) * cdelt1,
                                    _coerce_float(pc_mat[1][0]) * cdelt2,
                                )
                            except Exception:
                                pa_value = None

                if pa_value is None and header is not None:
                    def _header_get(key: str) -> Any:
                        if isinstance(header, dict):
                            return header.get(key)
                        try:
                            return header[key]
                        except Exception:
                            try:
                                return header.get(key)  # type: ignore[call-arg]
                            except Exception:
                                return None

                    pa_value = _pa_from_components(
                        _coerce_float(_header_get("CD1_1")),
                        _coerce_float(_header_get("CD2_1")),
                    )
                    if pa_value is None:
                        cdelt1 = _coerce_float(_header_get("CDELT1"))
                        cdelt2 = _coerce_float(_header_get("CDELT2"))
                        if cdelt1 is None:
                            cdelt1 = 1.0
                        if cdelt2 is None:
                            cdelt2 = 1.0
                        pa_value = _pa_from_components(
                            _coerce_float(_header_get("PC1_1")) * cdelt1,
                            _coerce_float(_header_get("PC2_1")) * cdelt2,
                        )
            except Exception:
                pa_value = None

        payload["PA_DEG"] = pa_value
        ra_deg = entry.center_ra_deg
        dec_deg = entry.center_dec_deg
        if (ra_deg is None or dec_deg is None) and header is not None:
            header_ra, header_dec = _extract_center_from_header(header)
            if ra_deg is None:
                ra_deg = header_ra
            if dec_deg is None:
                dec_deg = header_dec
        if ra_deg is not None:
            payload.setdefault("RA", float(ra_deg))
        if dec_deg is not None:
            payload.setdefault("DEC", float(dec_deg))
        center_coord = None
        if (
            ra_deg is not None
            and dec_deg is not None
            and SkyCoord is not None
            and u is not None
        ):
            try:
                center_coord = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
            except Exception:
                center_coord = None
        if payload.get("wcs") is None and center_coord is not None:
            payload["wcs"] = _FallbackWCS(center_coord)
            payload["_fallback_wcs_used"] = True
        if center_coord is not None:
            payload.setdefault("center", center_coord)
        if entry.instrument and not payload.get("instrument"):
            payload["instrument"] = entry.instrument
        return payload, (ra_deg, dec_deg)

    def _load_header(self, path: str) -> Any | None:
        if not path or fits is None:
            return None
        norm = casefold_path(path)
        if norm in self._header_cache:
            return self._header_cache[norm]
        try:
            header = fits.getheader(path, ignore_missing_end=True)
        except Exception:
            header = None
        sanitized = _sanitize_header_subset(header)
        self._header_cache[norm] = sanitized
        return sanitized

    def _resolve_group_entry_center(
        self,
        info: Any,
        path_map: dict[str, _NormalizedItem],
    ) -> tuple[float | None, float | None]:
        entry_obj: _NormalizedItem | None = None
        if isinstance(info, _NormalizedItem):
            entry_obj = info
        elif isinstance(info, dict):
            path_val = None
            for key in ("path", "path_raw", "path_preprocessed_cache"):
                candidate = info.get(key)
                if candidate:
                    path_val = candidate
                    break
            if path_val:
                entry_obj = path_map.get(casefold_path(path_val))

        center_ra: float | None = None
        center_dec: float | None = None
        if entry_obj is not None:
            center_ra = entry_obj.center_ra_deg
            center_dec = entry_obj.center_dec_deg
            if center_ra is None or center_dec is None:
                center_ra, center_dec = self._ensure_entry_coordinates(entry_obj)

        if (center_ra is None or center_dec is None) and isinstance(info, dict):
            try:
                ra_val = info.get("RA")
                center_ra = float(ra_val) if ra_val is not None else None
            except Exception:
                center_ra = None
            try:
                dec_val = info.get("DEC")
                center_dec = float(dec_val) if dec_val is not None else None
            except Exception:
                center_dec = None
        return center_ra, center_dec

    def _footprint_extent_degrees(
        self,
        footprint: List[Tuple[float, float]] | None,
    ) -> tuple[float, float] | None:
        if not footprint:
            return None
        ra_vals: list[float] = []
        dec_vals: list[float] = []
        for ra_val, dec_val in footprint:
            if ra_val is None or dec_val is None:
                continue
            try:
                ra = float(ra_val)
                dec = float(dec_val)
            except Exception:
                continue
            if not (math.isfinite(ra) and math.isfinite(dec)):
                continue
            ra_vals.append(ra)
            dec_vals.append(dec)
        if not ra_vals or not dec_vals:
            return None
        ra_min, ra_max = self._normalize_ra_span(ra_vals)
        width = max(0.0, ra_max - ra_min)
        dec_min = min(dec_vals)
        dec_max = max(dec_vals)
        height = max(0.0, dec_max - dec_min)
        return width, height

    def _compute_group_outline_bounds(self, groups: list[list[Any]]) -> list[_GroupOutline]:
        if not groups:
            return []
        path_map: dict[str, _NormalizedItem] = {}
        for entry in self._normalized_items:
            if entry.file_path:
                path_map[casefold_path(entry.file_path)] = entry
        outlines: list[_GroupOutline] = []
        for group_idx, group in enumerate(groups):
            ra_vals: list[float] = []
            dec_vals: list[float] = []
            reference_width: float | None = None
            reference_height: float | None = None
            for info in group or []:
                footprint = self._resolve_group_entry_footprint(
                    info,
                    path_map,
                    allow_header_lookup=reference_width is None,
                )
                if footprint:
                    for ra_deg, dec_deg in footprint:
                        if ra_deg is None or dec_deg is None:
                            continue
                        try:
                            ra_val = float(ra_deg)
                            dec_val = float(dec_deg)
                        except Exception:
                            continue
                        if not (math.isfinite(ra_val) and math.isfinite(dec_val)):
                            continue
                        ra_vals.append(ra_val)
                        dec_vals.append(dec_val)
                    if reference_width is None:
                        dims = self._footprint_extent_degrees(footprint)
                        if dims is not None:
                            reference_width, reference_height = dims
                    continue

                center_ra, center_dec = self._resolve_group_entry_center(info, path_map)
                if center_ra is not None and center_dec is not None:
                    if math.isfinite(center_ra) and math.isfinite(center_dec):
                        ra_vals.append(float(center_ra))
                        dec_vals.append(float(center_dec))
            if not ra_vals or not dec_vals:
                continue
            ra_min_raw, ra_max_raw = self._normalize_ra_span(ra_vals)
            dec_min_raw = min(dec_vals)
            dec_max_raw = max(dec_vals)

            center_ra = (ra_min_raw + ra_max_raw) / 2.0
            center_dec = (dec_min_raw + dec_max_raw) / 2.0

            width = reference_width
            height = reference_height
            if width is None or width <= 0 or height is None or height <= 0:
                width = ra_max_raw - ra_min_raw
                height = dec_max_raw - dec_min_raw
                if width <= 0:
                    width = 0.10
                if height <= 0:
                    height = 0.10
            width = max(width, 0.01)
            height = max(height, 0.01)

            ra_min = center_ra - width / 2.0
            ra_max = center_ra + width / 2.0
            dec_min = center_dec - height / 2.0
            dec_max = center_dec + height / 2.0

            outlines.append((group_idx, ra_min, ra_max, dec_min, dec_max))
        return outlines

    def _update_coverage_plot(
        self,
        groups: list[list[dict[str, Any]]] | None,
        coverage_first: bool | None = None,
    ) -> None:
        """Draw a simple coverage map in global WCS pixel space."""

        if self._coverage_axes is None or self._coverage_canvas is None:
            return

        axes = self._coverage_axes
        axes.clear()

        descriptor = self._global_wcs_state.get("descriptor")
        if not isinstance(descriptor, dict) or WCS is None or SkyCoord is None or u is None:
            self._safe_draw_coverage_canvas()
            return

        plan_wcs = descriptor.get("wcs")
        if plan_wcs is None:
            header_obj = descriptor.get("header")
            if header_obj is None and isinstance(descriptor.get("metadata"), dict):
                header_obj = descriptor["metadata"].get("header")
            if header_obj is not None:
                try:
                    plan_wcs = WCS(header_obj)
                except Exception:
                    plan_wcs = None
        if plan_wcs is None:
            self._safe_draw_coverage_canvas()
            return

        try:
            width = int(descriptor.get("width") or 0)
            height = int(descriptor.get("height") or 0)
        except Exception:
            width = height = 0
        if width <= 0 or height <= 0:
            self._safe_draw_coverage_canvas()
            return

        # Build a lookup from path to normalized entry, reusing the same helper
        # logic as the sky-preview group outlines.
        path_map: dict[str, _NormalizedItem] = {}
        for entry in self._normalized_items:
            if entry.file_path:
                path_map[casefold_path(entry.file_path)] = entry

        rectangles: list[tuple[float, float, float, float]] = []
        for group in groups or []:
            xs: list[float] = []
            ys: list[float] = []
            for info in group or []:
                footprint = self._resolve_group_entry_footprint(info, path_map)
                if not footprint:
                    continue
                ra_vals: list[float] = []
                dec_vals: list[float] = []
                for ra_deg, dec_deg in footprint:
                    try:
                        ra_vals.append(float(ra_deg))
                        dec_vals.append(float(dec_deg))
                    except Exception:
                        continue
                if not ra_vals or not dec_vals:
                    continue
                try:
                    sky = SkyCoord(ra=ra_vals * u.deg, dec=dec_vals * u.deg, frame="icrs")
                    x_pix, y_pix = plan_wcs.world_to_pixel(sky)
                except Exception:
                    continue
                try:
                    xs.extend(float(v) for v in np.asarray(x_pix, dtype=float).ravel().tolist())
                    ys.extend(float(v) for v in np.asarray(y_pix, dtype=float).ravel().tolist())
                except Exception:
                    continue
            if not xs or not ys:
                continue
            try:
                x_min = float(np.nanmin(xs))
                x_max = float(np.nanmax(xs))
                y_min = float(np.nanmin(ys))
                y_max = float(np.nanmax(ys))
            except Exception:
                continue
            if not (np.isfinite(x_min) and np.isfinite(x_max) and np.isfinite(y_min) and np.isfinite(y_max)):
                continue
            if x_max <= x_min or y_max <= y_min:
                continue
            rectangles.append((x_min, x_max, y_min, y_max))

        # Draw rectangles for each group coverage in pixel coordinates.
        global_x_min = 0.0
        global_y_min = 0.0
        global_x_max = float(width)
        global_y_max = float(height)

        for idx, (x_min, x_max, y_min, y_max) in enumerate(rectangles):
            color = "#3f7ad6"
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1.0,
                linestyle="--",
                edgecolor=color,
                facecolor="none",
                alpha=0.75,
            )
            axes.add_patch(rect)
            global_x_min = min(global_x_min, x_min)
            global_y_min = min(global_y_min, y_min)
            global_x_max = max(global_x_max, x_max)
            global_y_max = max(global_y_max, y_max)

        # Outer mosaic extent (dashed red box) similar to Tk.
        try:
            border = Rectangle(
                (0.0, 0.0),
                float(width),
                float(height),
                linewidth=1.6,
                linestyle="--",
                edgecolor="#d64b3f",
                facecolor="none",
                alpha=0.9,
            )
            axes.add_patch(border)
        except Exception:
            pass

        if global_x_max <= global_x_min or global_y_max <= global_y_min:
            axes.set_xlim(0, width)
            axes.set_ylim(height, 0)
        else:
            margin_x = max(8.0, 0.02 * float(width))
            margin_y = max(8.0, 0.02 * float(height))
            xmin = max(0.0, global_x_min - margin_x)
            xmax = min(float(width), global_x_max + margin_x)
            ymin = max(0.0, global_y_min - margin_y)
            ymax = min(float(height), global_y_max + margin_y)
            axes.set_xlim(xmin, xmax)
            axes.set_ylim(ymax, ymin)  # invert Y for image-like convention

        axes.set_xlabel(self._localizer.get("filter_axis_cov_x", "X [px]"))
        axes.set_ylabel(self._localizer.get("filter_axis_cov_y", "Y [px]"))
        axes.grid(True, linestyle=":", linewidth=0.6)

        self._safe_draw_coverage_canvas()

    def _resolve_group_entry_footprint(
        self,
        info: Any,
        path_map: dict[str, _NormalizedItem],
        *,
        allow_header_lookup: bool = True,
    ) -> List[Tuple[float, float]] | None:
        entry: _NormalizedItem | None = None
        if isinstance(info, _NormalizedItem):
            entry = info
        elif isinstance(info, dict):
            for key in ("path", "path_raw", "path_preprocessed_cache"):
                value = info.get(key)
                if value:
                    entry = path_map.get(casefold_path(value))
                    if entry is not None:
                        break
        if entry is not None:
            footprint = entry.footprint_radec
            if footprint:
                return footprint
            if allow_header_lookup:
                footprint = self._ensure_entry_footprint(entry)
                if footprint:
                    return footprint
        if isinstance(info, dict):
            footprint_payload = _sanitize_footprint_radec(info.get("footprint_radec"))
            if footprint_payload:
                return footprint_payload
            if allow_header_lookup:
                wcs_obj = info.get("wcs")
                if wcs_obj is not None:
                    return self._footprint_from_wcs_object(wcs_obj)
        return None

    @staticmethod
    def _footprint_from_wcs_object(wcs_obj: Any) -> List[Tuple[float, float]] | None:
        if wcs_obj is None or not getattr(wcs_obj, "is_celestial", False):
            return None
        try:
            nx = None
            ny = None
            if getattr(wcs_obj, "pixel_shape", None):
                nx, ny = wcs_obj.pixel_shape  # type: ignore[attr-defined]
            elif getattr(wcs_obj, "array_shape", None):
                ny, nx = wcs_obj.array_shape  # type: ignore[attr-defined]
            if nx is None or ny is None:
                return None
            corners_pix = [
                (0.5, 0.5),
                (float(nx) - 0.5, 0.5),
                (float(nx) - 0.5, float(ny) - 0.5),
                (0.5, float(ny) - 0.5),
            ]
            footprint: list[Tuple[float, float]] = []
            for x_pix, y_pix in corners_pix:
                try:
                    sky = wcs_obj.pixel_to_world(x_pix, y_pix)
                    ra_deg = float(getattr(sky, "ra").deg)
                    dec_deg = float(getattr(sky, "dec").deg)
                except Exception:
                    continue
                footprint.append((ra_deg, dec_deg))
            return footprint or None
        except Exception:
            return None

    @staticmethod
    def _normalize_ra_span(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        ref = float(np.median(values))
        adjusted = []
        for value in values:
            delta = (value - ref + 180.0) % 360.0 - 180.0
            adjusted.append(ref + delta)
        return min(adjusted), max(adjusted)

    def _format_message(self, key: str, default: str, **kwargs: Any) -> str:
        template = self._localizer.get(key, default)
        try:
            return template.format(**kwargs)
        except Exception:
            try:
                return default.format(**kwargs)
            except Exception:
                return default

    def _format_group_summary(self, groups_count: int, sizes_text: str) -> str:
        template = self._localizer.get(
            "filter_log_groups_summary",
            "Prepared {g} group(s), sizes: {sizes}.",
        )
        try:
            return template.format(g=int(groups_count), sizes=sizes_text)
        except Exception:
            return f"Prepared {groups_count} group(s), sizes: {sizes_text}."

    def _update_auto_group_summary_display(self, label_text: str, tooltip_text: str | None = None) -> None:
        if self._auto_group_summary_label is None:
            return
        self._auto_group_summary_label.setText(label_text)
        self._auto_group_summary_label.setToolTip(tooltip_text or label_text)

    def _update_orientation_override_value(self, value: float) -> None:
        sanitized = _sanitize_angle_value(value, 0.0)
        if sanitized > 0:
            self._runtime_overrides["cluster_orientation_split_deg"] = float(sanitized)
        else:
            self._runtime_overrides.pop("cluster_orientation_split_deg", None)

    def _resolve_cluster_threshold_override(self) -> float | None:
        candidates = (
            self._safe_lookup(self._config_overrides, "panel_clustering_threshold_deg"),
            self._safe_lookup(self._initial_overrides, "panel_clustering_threshold_deg"),
            self._safe_lookup(self._solver_settings, "panel_clustering_threshold_deg"),
            self._safe_lookup(self._config_overrides, "cluster_panel_threshold"),
            self._safe_lookup(self._initial_overrides, "cluster_panel_threshold"),
            self._safe_lookup(self._solver_settings, "cluster_panel_threshold"),
        )
        for candidate in candidates:
            try:
                value = float(candidate)
            except Exception:
                continue
            if value > 0:
                return value
        return None

    def _estimate_threshold_from_coords(self, coords: list[tuple[float, float]]) -> float:
        dispersion = 0.0
        if coords:
            if _COMPUTE_MAX_SEPARATION is not None:
                try:
                    dispersion = float(_COMPUTE_MAX_SEPARATION(coords))
                except Exception:
                    dispersion = 0.0
            if not dispersion:
                dispersion = self._approximate_dispersion(coords)
        if dispersion <= 0.12:
            threshold = 0.10
        elif dispersion <= 0.30:
            threshold = 0.15
        else:
            threshold = 0.05 if dispersion <= 0.60 else 0.20
        return min(0.20, max(0.08, threshold))

    @staticmethod
    def _approximate_dispersion(coords: list[tuple[float, float]]) -> float:
        if len(coords) < 2:
            return 0.0
        max_sep = 0.0
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dra = abs(coords[i][0] - coords[j][0])
                ddec = abs(coords[i][1] - coords[j][1])
                max_sep = max(max_sep, math.hypot(dra, ddec))
        return max_sep

    def _resolve_orientation_split_threshold(self) -> float:
        if hasattr(self, "_auto_angle_enabled") and not getattr(self, "_auto_angle_enabled", True):
            return max(0.0, float(getattr(self, "_angle_split_value", 0.0) or 0.0))
        candidates = (
            self._safe_lookup(self._config_overrides, "cluster_orientation_split_deg"),
            self._safe_lookup(self._initial_overrides, "cluster_orientation_split_deg"),
            self._safe_lookup(self._solver_settings, "cluster_orientation_split_deg"),
        )
        for candidate in candidates:
            value = _sanitize_angle_value(candidate, 0.0)
            if value > 0:
                return value
        return 0.0

    def _resolve_angle_split_candidate(self) -> float:
        value = getattr(self, "_angle_split_value", None)
        if value is not None:
            return float(value)
        raw = self._config_value("cluster_orientation_split_deg", 0.0)
        return _sanitize_angle_value(raw, 0.0)

    def _resolve_auto_angle_detect_threshold(self) -> float:
        candidates = (
            self._safe_lookup(self._config_overrides, "auto_angle_detect_deg"),
            self._safe_lookup(self._initial_overrides, "auto_angle_detect_deg"),
            self._safe_lookup(self._solver_settings, "auto_angle_detect_deg"),
        )
        for candidate in candidates:
            value = _sanitize_angle_value(candidate, AUTO_ANGLE_DETECT_DEFAULT_DEG)
            if value > 0:
                return value
        return AUTO_ANGLE_DETECT_DEFAULT_DEG

    def _resolve_autosplit_caps(self) -> tuple[int, int]:
        def _coerce_int(value: Any) -> int | None:
            try:
                parsed = int(value)
            except Exception:
                return None
            return parsed
        def _has_explicit_value(source: Any) -> bool:
            if not isinstance(source, dict):
                return False
            if "max_raw_per_master_tile" not in source:
                return False
            value = source.get("max_raw_per_master_tile")
            if value is None:
                return False
            if isinstance(value, str) and not value.strip():
                return False
            return True

        default_cap = 50
        base_cap_raw = _coerce_int(self._config_value("max_raw_per_master_tile", default_cap))
        explicit_config_value = any(
            _has_explicit_value(source)
            for source in (self._runtime_overrides, self._config_overrides, self._initial_overrides)
        )
        if base_cap_raw is None:
            cap_effective = default_cap
        elif base_cap_raw > 0:
            cap_effective = base_cap_raw
        elif explicit_config_value:
            cap_effective = 0  # 0 => unlimited cap enforced by the Filter Qt
        else:
            cap_effective = default_cap
        cap_candidates = (
            self._safe_lookup(self._solver_settings, "max_raw_per_master_tile"),
            self._safe_lookup(self._runtime_overrides, "max_raw_per_master_tile"),
            self._safe_lookup(self._config_overrides, "max_raw_per_master_tile"),
            self._safe_lookup(self._initial_overrides, "max_raw_per_master_tile"),
        )
        for candidate in cap_candidates:
            value = _coerce_int(candidate)
            if value is None:
                continue
            if value <= 0:
                cap_effective = 0
                break
            cap_effective = value
            break

        base_min_guess = min(8, cap_effective) if cap_effective > 0 else 0
        default_min = _coerce_int(self._config_value("autosplit_min_cap", base_min_guess))
        if cap_effective > 0:
            min_cap_effective = (
                max(1, min(default_min, cap_effective)) if default_min and default_min > 0 else base_min_guess
            )
        else:
            min_cap_effective = max(0, default_min) if default_min is not None else 0
        min_candidates = (
            self._safe_lookup(self._solver_settings, "autosplit_min_cap"),
            self._safe_lookup(self._runtime_overrides, "autosplit_min_cap"),
            self._safe_lookup(self._config_overrides, "autosplit_min_cap"),
            self._safe_lookup(self._initial_overrides, "autosplit_min_cap"),
        )
        for candidate in min_candidates:
            value = _coerce_int(candidate)
            if value is None:
                continue
            if cap_effective > 0 and value > 0:
                min_cap_effective = max(1, min(value, cap_effective))
                break
            if cap_effective == 0:
                min_cap_effective = max(0, value)
                break

        if cap_effective > 0:
            min_cap_effective = max(1, min(min_cap_effective, cap_effective))
        else:
            min_cap_effective = max(0, min_cap_effective)
        # When cap_effective == 0 the Filter Qt does not impose a split cap.
        return cap_effective, min_cap_effective

    def _resolve_overcap_percent(self) -> int:
        value = getattr(self, "_overcap_percent_value", None)
        if value is not None:
            return int(value)
        return self._clamp_overcap_percent(self._config_value("filter_overcap_allowance_pct", 10))

    def _resolve_overlap_percent(self) -> int:
        value = getattr(self, "_batch_overlap_percent_value", None)
        if value is not None:
            return int(value)
        return self._clamp_overlap_percent(self._config_value("batch_overlap_pct", 40))

    @staticmethod
    def _sort_group_for_overlap(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not group:
            return []

        def _safe_number(value: Any) -> float:
            try:
                number = float(value)
            except Exception:
                return math.inf
            return number if math.isfinite(number) else math.inf

        ordered = sorted(
            ((idx, info) for idx, info in enumerate(group)),
            key=lambda item: (_safe_number(item[1].get("RA")), _safe_number(item[1].get("DEC")), item[0]),
        )
        return [info for _idx, info in ordered]

    @staticmethod
    def _make_overlapping_batches(
        group: list[dict[str, Any]], *, cap: int, overlap_fraction: float, min_cap: int
    ) -> list[list[dict[str, Any]]]:
        if not group:
            return []
        if cap <= 0 or len(group) <= cap:
            return [group]

        step = max(1, int(cap * (1.0 - overlap_fraction)))
        batches: list[list[dict[str, Any]]] = []
        start = 0
        n = len(group)

        while start < n:
            end = min(n, start + cap)
            batch = group[start:end]
            if batch:
                if len(batch) < min_cap and batches and len(batches[-1]) < cap:
                    batches[-1].extend(batch)
                else:
                    batches.append(batch)
            if end == n:
                break
            start += step

        if min_cap > 1 and len(batches) > 1:
            merged: list[list[dict[str, Any]]] = []
            for batch in batches:
                if len(batch) < min_cap and merged:
                    merged[-1].extend(batch)
                else:
                    merged.append(batch)
            batches = merged
        return batches

    def _log_batching_summary(
        self,
        batches: list[list[dict[str, Any]]],
        *,
        cap: int,
        overlap_fraction: float,
        messages: list[str | tuple[str, str]] | None,
    ) -> None:
        if messages is None:
            return
        try:
            effective_step = max(1, int(cap * (1.0 - overlap_fraction))) if cap > 0 else 0
            overlap_pct = max(0.0, min(1.0, overlap_fraction))
            sizes_text = ", ".join(str(len(batch)) for batch in batches) if batches else ""
            messages.append(
                self._format_message(
                    "log_batching_params",
                    "[Batching] cap={CAP}, overlap={OVER:.2f}, effective step={STEP}",
                    CAP=int(cap),
                    OVER=float(overlap_pct),
                    STEP=int(effective_step),
                )
            )
            messages.append(
                self._format_message(
                    "log_batching_sizes",
                    "[Batching] Created {COUNT} batches, sizes: {SIZES}",
                    COUNT=len(batches),
                    SIZES=sizes_text or "<none>",
                )
            )
            previous_paths: set[str] = set()
            for idx, batch in enumerate(batches, start=1):
                paths = {
                    str(
                        info.get("path")
                        or info.get("file_path")
                        or info.get("path_raw")
                        or f"entry_{id(info)}"
                    )
                    for info in batch
                }
                overlap_with_prev = len(previous_paths.intersection(paths)) if previous_paths else 0
                messages.append(
                    self._format_message(
                        "log_batching_batch_detail",
                        "[Batching] Batch {IDX}: size={SIZE}, overlap_with_prev={OVERLAP}",
                        IDX=int(idx),
                        SIZE=len(batch),
                        OVERLAP=int(overlap_with_prev),
                    )
                )
                previous_paths = paths
        except Exception:
            return

    def _coverage_first_enabled(self) -> bool:
        return bool(self._coverage_first_enabled_flag)

    def _ensure_global_wcs_for_indices(
        self,
        selected_indices: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        success, meta_payload, _ = self._ensure_global_wcs_for_selection(True, selected_indices)
        return success, meta_payload

    def _build_sds_batches_with_policy(
        self,
        entry_infos: list[dict[str, Any]],
        grid_h: int,
        grid_w: int,
        *,
        coverage_threshold: float,
        min_batch_size: int,
        target_batch_size: int,
    ) -> list[list[dict[str, Any]]]:
        if not entry_infos or grid_h <= 0 or grid_w <= 0:
            return []
        total_cells = grid_h * grid_w
        batches: list[list[dict[str, Any]]] = []
        current_batch: list[dict[str, Any]] = []
        coverage_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        coverage_cells = 0
        for info in entry_infos:
            bbox = info.get("grid_bbox")
            if not isinstance(bbox, tuple) or len(bbox) != 4:
                continue
            gy0, gy1, gx0, gx1 = bbox
            if gy1 <= gy0 or gx1 <= gx0:
                continue
            region = coverage_grid[gy0:gy1, gx0:gx1]
            region_before = int(region.sum())
            region_cells = (gy1 - gy0) * (gx1 - gx0)
            region[...] = 1
            gain = max(0, region_cells - region_before)
            coverage_cells += gain
            current_batch.append(info)
            batch_len = len(current_batch)
            coverage_fraction = (coverage_cells / total_cells) if total_cells else 1.0
            if (batch_len >= min_batch_size and coverage_fraction >= coverage_threshold) or (
                batch_len >= target_batch_size
            ):
                batches.append(current_batch)
                coverage_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
                coverage_cells = 0
                current_batch = []
        if current_batch:
            if len(current_batch) < min_batch_size and batches:
                batches[-1].extend(current_batch)
            else:
                batches.append(current_batch)
        return batches

    def _build_sds_batches_for_indices(
        self,
        selected_indices: list[int],
        *,
        coverage_threshold: float = 0.92,
        min_batch_size: int = 5,
        target_batch_size: int = 10,
    ) -> list[list[dict[str, Any]]] | None:
        descriptor = self._global_wcs_state.get("descriptor")
        if not descriptor or WCS is None:
            return None
        plan_wcs = descriptor.get("wcs")
        if plan_wcs is None:
            header_obj = descriptor.get("header")
            if header_obj is None and isinstance(descriptor.get("metadata"), dict):
                header_obj = descriptor["metadata"].get("header")
            if header_obj is not None:
                try:
                    plan_wcs = WCS(header_obj)
                except Exception:
                    plan_wcs = None
        if plan_wcs is None:
            return None
        try:
            width = int(descriptor.get("width") or 0)
            height = int(descriptor.get("height") or 0)
        except Exception:
            width = height = 0
        if width <= 0 or height <= 0:
            return None

        entry_infos: list[dict[str, Any]] = []
        for idx in selected_indices:
            if not (0 <= idx < len(self._normalized_items)):
                continue
            entry = self._normalized_items[idx]
            if not self._entry_is_seestar(entry):
                continue
            payload, _coords = self._build_candidate_payload(entry)
            if payload is None:
                continue
            local_wcs = payload.get("wcs")
            if local_wcs is None or not getattr(local_wcs, "is_celestial", False):
                continue
            shape_hw = self._coerce_entry_shape(entry, payload)
            if (
                shape_hw is None
                or shape_hw[0] is None
                or shape_hw[1] is None
                or shape_hw[0] <= 0
                or shape_hw[1] <= 0
            ):
                continue
            try:
                rows = np.array([0, 0, shape_hw[0] - 1, shape_hw[0] - 1], dtype=float)
                cols = np.array([0, shape_hw[1] - 1, 0, shape_hw[1] - 1], dtype=float)
                world_coords = local_wcs.pixel_to_world(cols, rows)
                g_cols, g_rows = plan_wcs.world_to_pixel(world_coords)
            except Exception:
                continue
            if g_cols is None or g_rows is None:
                continue
            try:
                x0 = int(np.floor(np.nanmin(g_cols)))
                x1 = int(np.ceil(np.nanmax(g_cols))) + 1
                y0 = int(np.floor(np.nanmin(g_rows)))
                y1 = int(np.ceil(np.nanmax(g_rows))) + 1
            except Exception:
                continue
            if any(np.isnan(val) for val in (x0, x1, y0, y1)):
                continue
            x0 = max(0, min(width, x0))
            x1 = max(x0 + 1, min(width, x1))
            y0 = max(0, min(height, y0))
            y1 = max(y0 + 1, min(height, y1))
            entry_infos.append({"payload": payload, "bbox": (y0, y1, x0, x1)})

        if not entry_infos:
            return None

        grid_h = max(1, min(512, height))
        grid_w = max(1, min(512, width))
        scale_y = grid_h / float(height)
        scale_x = grid_w / float(width)
        for info in entry_infos:
            y0, y1, x0, x1 = info["bbox"]
            g_y0 = max(0, min(grid_h, int(math.floor(y0 * scale_y))))
            g_y1 = max(g_y0 + 1, min(grid_h, int(math.ceil(y1 * scale_y))))
            g_x0 = max(0, min(grid_w, int(math.floor(x0 * scale_x))))
            g_x1 = max(g_x0 + 1, min(grid_w, int(math.ceil(x1 * scale_x))))
            info["grid_bbox"] = (g_y0, g_y1, g_x0, g_x1)

        coverage_threshold = max(0.10, min(0.99, float(coverage_threshold or 0.92)))
        try:
            min_batch_size = max(1, int(min_batch_size))
        except Exception:
            min_batch_size = 5
        try:
            target_batch_size = max(min_batch_size, int(target_batch_size))
        except Exception:
            target_batch_size = max(min_batch_size, 10)
        batches = self._build_sds_batches_with_policy(
            entry_infos,
            grid_h,
            grid_w,
            coverage_threshold=coverage_threshold,
            min_batch_size=min_batch_size,
            target_batch_size=target_batch_size,
        )
        if not batches:
            return None
        final_groups: list[list[dict[str, Any]]] = []
        for batch in batches:
            group_payload: list[dict[str, Any]] = []
            for info in batch:
                payload = info.get("payload")
                if isinstance(payload, dict):
                    group_payload.append(dict(payload))
            if group_payload:
                final_groups.append(group_payload)
        return final_groups if final_groups else None

    def _entry_is_seestar(self, entry: _NormalizedItem) -> bool:
        label = self._normalize_string(entry.instrument)
        if label and any(token in label.lower() for token in ("seestar", "s50", "s30")):
            return True
        original = entry.original
        header = None
        if isinstance(original, dict):
            header = original.get("header") or original.get("header_subset")
        if header is None:
            header = self._load_header(entry.file_path)
        if header is None:
            return False
        detected = _detect_instrument_from_header(header)
        return bool(detected and "seestar" in detected.lower())

    def _coerce_entry_shape(
        self,
        entry: _NormalizedItem,
        payload: dict[str, Any],
    ) -> tuple[int, int] | None:
        shape_candidate = payload.get("shape")
        if isinstance(shape_candidate, (list, tuple)) and len(shape_candidate) >= 2:
            try:
                h = int(shape_candidate[0])
                w = int(shape_candidate[1])
                if h > 0 and w > 0:
                    return (h, w)
            except Exception:
                pass
        wcs_obj = payload.get("wcs")
        if getattr(wcs_obj, "pixel_shape", None):
            try:
                ny, nx = wcs_obj.pixel_shape  # type: ignore[attr-defined]
                h = int(ny)
                w = int(nx)
                if h > 0 and w > 0:
                    return (h, w)
            except Exception:
                pass
        header = None
        if isinstance(entry.original, dict):
            header = entry.original.get("header") or entry.original.get("header_subset")
        if header is None:
            header = self._load_header(entry.file_path)
        if header is not None:
            try:
                naxis1 = int(header.get("NAXIS1"))
                naxis2 = int(header.get("NAXIS2"))
                if naxis2 > 0 and naxis1 > 0:
                    return (naxis2, naxis1)
            except Exception:
                pass
        return None

    def _format_message(self, key: str, default: str, **kwargs: Any) -> str:
        template = self._localizer.get(key, default)
        try:
            return template.format(**kwargs)
        except Exception:
            try:
                return default.format(**kwargs)
            except Exception:
                return default

    def _compute_global_center(self) -> Tuple[float | None, float | None]:
        coords: list[Tuple[float, float]] = []
        for row, entry in enumerate(self._normalized_items):
            if not self._entry_is_checked(row):
                continue
            ra, dec = entry.center_ra_deg, entry.center_dec_deg
            if ra is None or dec is None:
                ra, dec = self._ensure_entry_coordinates(entry)
            if ra is None or dec is None:
                continue
            coords.append((ra, dec))

        if not coords:
            for entry in self._normalized_items:
                ra, dec = entry.center_ra_deg, entry.center_dec_deg
                if ra is None or dec is None:
                    ra, dec = self._ensure_entry_coordinates(entry)
                if ra is None or dec is None:
                    continue
                coords.append((ra, dec))
            if not coords:
                return None, None

        sum_x = sum_y = sum_z = 0.0
        for ra_deg, dec_deg in coords:
            ra_rad = math.radians(ra_deg)
            dec_rad = math.radians(dec_deg)
            sum_x += math.cos(dec_rad) * math.cos(ra_rad)
            sum_y += math.cos(dec_rad) * math.sin(ra_rad)
            sum_z += math.sin(dec_rad)

        count = len(coords)
        if count == 0:
            return None, None

        sum_x /= count
        sum_y /= count
        sum_z /= count

        ra = math.degrees(math.atan2(sum_y, sum_x))
        if ra < 0:
            ra += 360.0
        hyp = math.hypot(sum_x, sum_y)
        dec = math.degrees(math.atan2(sum_z, hyp))
        return ra, dec

    def _on_exclude_distance(self) -> None:
        if self._distance_spin is None:
            return
        threshold = float(self._distance_spin.value())
        center_ra, center_dec = self._compute_global_center()
        if center_ra is None or center_dec is None:
            message = self._localizer.get(
                "filter.exclude.no_center",
                "Cannot exclude by distance because no celestial center is available.",
            )
            self._append_log(message, level="WARN")
            self._status_label.setText(message)
            return

        excluded = 0
        for row, entry in enumerate(self._normalized_items):
            ra_deg, dec_deg = entry.center_ra_deg, entry.center_dec_deg
            if ra_deg is None or dec_deg is None:
                ra_deg, dec_deg = self._ensure_entry_coordinates(entry)
            if ra_deg is None or dec_deg is None:
                continue
            distance = self._angular_distance_deg(ra_deg, dec_deg, center_ra, center_dec)
            if distance > threshold:
                if self._set_entry_checked(row, False):
                    excluded += 1
        summary = self._localizer.get(
            "filter.exclude.result",
            "Excluded {count} frame(s) beyond {threshold:.1f}°.",
        )
        try:
            summary = summary.format(count=excluded, threshold=threshold)
        except Exception:
            pass
        self._append_log(summary)
        self._status_label.setText(summary)
        self._update_summary_label()
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()

    def _toggle_maximize_restore(self) -> None:
        button = self._toolbar_maximize_btn
        window = self.window() or self
        if not self._maximized_state:
            self._saved_geometry = window.geometry()
            window.showMaximized()
            self._maximized_state = True
            if button is not None:
                button.setText(self._localizer.get("filter_btn_restore", "Restore"))
        else:
            window.showNormal()
            if self._saved_geometry is not None:
                window.setGeometry(self._saved_geometry)
            self._maximized_state = False
            if button is not None:
                button.setText(self._localizer.get("filter_btn_maximize", "Maximize"))

    def _on_export_csv(self) -> None:
        rows = self._gather_csv_rows()
        if not rows:
            message = self._localizer.get(
                "filter.export.empty",
                "No rows available for CSV export.",
            )
            self._append_log(message, level="WARN")
            self._status_label.setText(message)
            return

        destination = self._cache_csv_path or self._prompt_csv_path()
        if not destination:
            return

        success = self._write_csv_file(destination, rows)
        if success:
            self._cache_csv_path = destination
            message = self._localizer.get(
                "filter.export.success",
                "Exported {count} entry(ies) to {path}.",
            )
            try:
                message = message.format(count=len(rows), path=destination)
            except Exception:
                pass
            self._append_log(message)
            self._status_label.setText(message)
        else:
            message = self._localizer.get(
                "filter.export.error",
                "Failed to export CSV file.",
            )
            self._append_log(message, level="ERROR")
            self._status_label.setText(message)

    def _prompt_csv_path(self) -> str | None:
        caption = self._localizer.get("filter.export.dialog_title", "Export filter CSV")
        csv_filter = self._localizer.get("filter.export.csv_filter", "CSV files (*.csv)")
        try:
            default_dir = ensure_user_config_dir()
        except Exception:
            try:
                default_dir = get_user_config_dir()
            except Exception:
                default_dir = Path.home()
        default_path = str(Path(default_dir) / "zemosaic_filter.csv")
        path, _filter = QFileDialog.getSaveFileName(
            self,
            caption,
            default_path,
            f"{csv_filter};;All Files (*)",
        )
        return path or None

    def _gather_csv_rows(self) -> List[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for entry in self._normalized_items:
            payload = self._build_csv_row(entry)
            if payload:
                rows.append(payload)
        return rows

    def _build_csv_row(self, entry: _NormalizedItem) -> dict[str, Any] | None:
        path = entry.file_path
        if not path:
            return None
        header = self._load_header_for_entry(entry)
        record: dict[str, Any] = {"path": path}
        for key in ("NAXIS1", "NAXIS2", "CRVAL1", "CRVAL2", "CREATOR", "INSTRUME", "DATE-OBS", "EXPTIME", "FILTER", "OBJECT"):
            record[key] = header.get(key) if header else None

        if record.get("CRVAL1") is None or record.get("CRVAL2") is None:
            if entry.center_ra_deg is not None and entry.center_dec_deg is not None:
                record["CRVAL1"] = entry.center_ra_deg
                record["CRVAL2"] = entry.center_dec_deg

        footprint = entry.footprint_radec or self._ensure_entry_footprint(entry)
        for idx in range(1, 5):
            ra_key = f"FP_RA{idx}"
            dec_key = f"FP_DEC{idx}"
            record[ra_key] = ""
            record[dec_key] = ""
        if footprint:
            for idx, (ra_deg, dec_deg) in enumerate(footprint[:4], start=1):
                record[f"FP_RA{idx}"] = ra_deg
                record[f"FP_DEC{idx}"] = dec_deg

        return record

    def _load_header_for_entry(self, entry: _NormalizedItem) -> Any:
        if isinstance(entry.original, dict):
            header_candidate = entry.original.get("header") or entry.original.get("header_subset")
            header_candidate = _sanitize_header_subset(header_candidate)
            if header_candidate:
                return header_candidate
        path = entry.file_path
        if not path or not _path_is_file(path):
            return None
        return self._load_header(str(_expand_to_path(path) or path))

    def _write_csv_file(self, path: str, rows: List[dict[str, Any]]) -> bool:
        fieldnames = [
            "path",
            "NAXIS1",
            "NAXIS2",
            "CRVAL1",
            "CRVAL2",
            "FP_RA1",
            "FP_DEC1",
            "FP_RA2",
            "FP_DEC2",
            "FP_RA3",
            "FP_DEC3",
            "FP_RA4",
            "FP_DEC4",
            "CREATOR",
            "INSTRUME",
            "DATE-OBS",
            "EXPTIME",
            "FILTER",
            "OBJECT",
        ]
        try:
            path_obj = _expand_to_path(path)
            if path_obj is None:
                raise ValueError("invalid CSV path")
            if path_obj.parent:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
            with path_obj.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            return True
        except Exception as exc:
            self._append_log(f"CSV export failed: {exc}", level="ERROR")
            return False

    def _build_ui(self) -> None:
        title = self._localizer.get("filter.dialog.title", "Filter raw frames")
        self.setWindowTitle(title)
        self.resize(800, 500)

        main_layout = QVBoxLayout(self)

        description = self._localizer.get(
            "filter.dialog.description",
            "Review the frames that will be used before launching the mosaic process.",
        )
        description_label = QLabel(description, self)
        description_label.setWordWrap(True)
        main_layout.addWidget(description_label)

        content_splitter = QSplitter(Qt.Horizontal, self)
        content_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(content_splitter, 1)

        preview_group = QGroupBox(self._localizer.get("filter.preview.header", "Sky preview"), self)
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(6)
        if self._preview_canvas is not None:
            tabs = QTabWidget(preview_group)
            self._preview_tabs = tabs

            sky_container = QWidget(preview_group)
            sky_layout = QVBoxLayout(sky_container)
            sky_layout.setContentsMargins(0, 0, 0, 0)
            sky_layout.setSpacing(0)
            sky_layout.addWidget(self._preview_canvas, 1)
            controls_row = QWidget(sky_container)
            controls_layout = QHBoxLayout(controls_row)
            controls_layout.setContentsMargins(6, 4, 6, 4)
            controls_layout.setSpacing(8)
            color_label = self._localizer.get(
                "filter.preview.colorize_groups",
                "Color footprints by group",
            )
            color_checkbox = QCheckBox(color_label, controls_row)
            color_checkbox.setChecked(True)
            color_checkbox.toggled.connect(lambda _checked: self._schedule_preview_refresh())  # type: ignore[arg-type]
            controls_layout.addWidget(color_checkbox)
            controls_layout.addStretch(1)
            sky_layout.addWidget(controls_row, 0)
            self._color_by_group_checkbox = color_checkbox
            tabs.addTab(
                sky_container,
                self._localizer.get("filter_tab_sky_preview", "Sky Preview"),
            )

            coverage_canvas = self._create_coverage_canvas()
            if coverage_canvas is not None:
                coverage_container = QWidget(preview_group)
                coverage_layout = QVBoxLayout(coverage_container)
                coverage_layout.setContentsMargins(0, 0, 0, 0)
                coverage_layout.setSpacing(0)
                coverage_layout.addWidget(coverage_canvas, 1)
                tabs.addTab(
                    coverage_container,
                    self._localizer.get("filter_tab_coverage_map", "Coverage Map"),
                )

            preview_layout.addWidget(tabs, 1)
        preview_layout.addWidget(self._preview_hint_label)
        content_splitter.addWidget(preview_group)

        controls_container = QWidget(self)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        content_splitter.addWidget(controls_container)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)

        toolbar_widget = self._create_toolbar_widget()
        controls_layout.addWidget(toolbar_widget)

        exclusion_box = self._create_exclusion_box()
        controls_layout.addWidget(exclusion_box)

        instrument_widget = self._create_instrument_row()
        controls_layout.addWidget(instrument_widget)

        activity_group = QGroupBox(
            self._localizer.get("filter_log_panel_title", "Activity log"),
            self,
        )
        activity_layout = QVBoxLayout(activity_group)
        activity_layout.setContentsMargins(8, 8, 8, 8)
        self._activity_log_output = QPlainTextEdit(activity_group)
        self._activity_log_output.setReadOnly(True)
        self._activity_log_output.setMaximumBlockCount(500)
        self._activity_log_output.setPlaceholderText(
            self._localizer.get(
                "filter.activity.placeholder",
                "Recent filter activity will be listed here.",
            )
        )
        activity_layout.addWidget(self._activity_log_output)
        controls_layout.addWidget(activity_group, 1)

        self._tree.setColumnCount(3)
        headers = [
            self._localizer.get("filter.column.file", "File"),
            self._localizer.get("filter.column.wcs", "WCS"),
            self._localizer.get("filter.column.instrument", "Instrument"),
        ]
        self._tree.setHeaderLabels(headers)
        self._tree.setExpandsOnDoubleClick(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tree.setSelectionMode(QAbstractItemView.NoSelection)
        self._tree.setDragDropMode(QAbstractItemView.NoDragDrop)
        self._tree.setDragEnabled(False)
        self._tree.setAcceptDrops(False)
        header = self._tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._tree.itemChanged.connect(self._handle_tree_item_changed)

        images_group = QGroupBox(
            self._localizer.get("filter_images_check_to_keep", "Images (check to keep)"),
            self,
        )
        images_layout = QVBoxLayout(images_group)
        images_layout.setContentsMargins(8, 8, 8, 8)
        images_layout.setSpacing(6)
        tree_controls = QHBoxLayout()
        tree_controls.setContentsMargins(0, 0, 0, 0)
        tree_controls.setSpacing(6)
        expand_btn = QPushButton(self._localizer.get("filter.expand_all", "Expand all"), images_group)
        expand_btn.setToolTip(self._localizer.get("filter.expand_all.tooltip", "Expand all groups"))
        expand_btn.clicked.connect(self._expand_all_groups)  # type: ignore[arg-type]
        collapse_btn = QPushButton(self._localizer.get("filter.collapse_all", "Collapse all"), images_group)
        collapse_btn.setToolTip(self._localizer.get("filter.collapse_all.tooltip", "Collapse all groups"))
        collapse_btn.clicked.connect(self._collapse_all_groups)  # type: ignore[arg-type]
        tree_controls.addWidget(expand_btn)
        tree_controls.addWidget(collapse_btn)
        tree_controls.addStretch(1)
        images_layout.addLayout(tree_controls)
        images_layout.addWidget(self._tree, 1)

        controls_layout.addWidget(images_group, 2)

        wcs_box = self._create_wcs_controls_box()
        controls_layout.addWidget(wcs_box)

        options_box = self._create_options_box()
        controls_layout.addWidget(options_box)

        actions_widget = QWidget(self)
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(8)
        actions_layout.addWidget(self._summary_label, 1)
        actions_layout.addStretch(1)
        select_all_btn = QPushButton(
            self._localizer.get("filter.button.select_all", "Select all"), self
        )
        select_all_btn.clicked.connect(self._select_all)  # type: ignore[arg-type]
        actions_layout.addWidget(select_all_btn, 0)
        select_none_btn = QPushButton(
            self._localizer.get("filter.button.select_none", "Select none"), self
        )
        select_none_btn.clicked.connect(self._select_none)  # type: ignore[arg-type]
        actions_layout.addWidget(select_none_btn, 0)
        controls_layout.addWidget(actions_widget)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        controls_layout.addWidget(button_box)
        self._dialog_button_box = button_box

        self._apply_saved_window_geometry()

    def _build_processing_overlay(self) -> None:
        overlay = QWidget(self)
        overlay.setObjectName("filterProcessingOverlay")
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        overlay.hide()
        overlay.setStyleSheet(
            "#filterProcessingOverlay { background-color: rgba(0, 0, 0, 0); }"
            "#filterProcessingOverlayContainer {"
            " background-color: rgba(0, 0, 0, 150);"
            " border-radius: 16px;"
            " padding: 16px;"
            "}"
            "QLabel#filterProcessingOverlayLabel { color: white; }"
        )
        layout = QVBoxLayout(overlay)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignCenter)

        container = QWidget(overlay)
        container.setObjectName("filterProcessingOverlayContainer")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(24, 24, 24, 24)
        container_layout.setSpacing(8)

        label = QLabel(container)
        label.setObjectName("filterProcessingOverlayLabel")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        animation_path = self._resolve_overlay_animation_path()
        movie: QMovie | None = None
        if animation_path:
            try:
                movie = QMovie(animation_path)
            except Exception:
                movie = None
            if movie is not None and not movie.isValid():
                movie = None
        if movie is not None:
            label.setMovie(movie)
        else:
            label.setText(
                self._localizer.get("filter.overlay.solving", "Solving WCS, please wait…")
            )

        container_layout.addWidget(label, 0, Qt.AlignCenter)
        layout.addWidget(container, 0, Qt.AlignCenter)

        self._solve_overlay = overlay
        self._solve_overlay_label = label
        self._solve_animation = movie
        self._update_overlay_geometry()

    def _resolve_overlay_animation_path(self) -> str | None:
        """Return the first existing wait-animation GIF path.

        Some environments ship the GIF with duplicate extensions, so we probe
        several reasonable filenames before falling back to a directory scan.
        """

        try:
            base_dir = get_app_base_dir()
        except Exception:
            base_dir = None

        candidate_names = (
            "wait_animation.gif",
            "wait_animation.GIF",
            "wait_animation.gif.gif",
            "wait_animation.GIF.GIF",
        )

        search_roots: list[Path] = []
        if base_dir:
            search_roots.append(base_dir / "gif")
        try:
            search_roots.append(Path.cwd() / "gif")
        except Exception:
            pass

        for root in search_roots:
            if not root or not root.is_dir():
                continue
            for name in candidate_names:
                candidate = root / name
                if candidate.is_file():
                    return str(candidate)
            try:
                for entry in root.iterdir():
                    lower_name = entry.name.lower()
                    if not lower_name.startswith("wait_") or not lower_name.endswith(".gif"):
                        continue
                    if entry.is_file():
                        return str(entry)
            except Exception:
                continue
        return None

    def _update_overlay_geometry(self) -> None:
        if self._solve_overlay is None:
            return
        try:
            self._solve_overlay.setGeometry(self.rect())
        except Exception:
            pass

    def _show_processing_overlay(self) -> None:
        overlay = self._solve_overlay
        if overlay is None:
            return
        self._update_overlay_geometry()
        try:
            overlay.raise_()
        except Exception:
            pass
        overlay.show()
        if self._solve_animation is not None:
            try:
                self._solve_animation.start()
            except Exception:
                pass

    def _hide_processing_overlay(self) -> None:
        if self._solve_animation is not None:
            try:
                self._solve_animation.stop()
            except Exception:
                pass
        overlay = self._solve_overlay
        if overlay is not None:
            overlay.hide()

    def _load_saved_window_geometry(self) -> tuple[int, int, int, int] | None:
        if _load_gui_config is None:
            return None
        try:
            config = _load_gui_config()
        except Exception:
            return None
        return self._normalize_geometry_value(config.get(QT_FILTER_WINDOW_GEOMETRY_KEY))

    def _capture_current_window_geometry(self) -> tuple[int, int, int, int] | None:
        rect = self.normalGeometry() if self.isMaximized() else self.geometry()
        return self._normalize_geometry_value(
            (rect.x(), rect.y(), rect.width(), rect.height())
        )

    def _apply_saved_window_geometry(self) -> None:
        geometry = self._load_saved_window_geometry()
        if geometry is None:
            return
        x, y, width, height = geometry
        try:
            self.setGeometry(x, y, width, height)
        except Exception:
            pass

    def _persist_window_geometry(self) -> None:
        if _save_gui_config is None:
            return
        geometry = self._capture_current_window_geometry()
        if geometry is None:
            return
        try:
            config = _load_gui_config() if _load_gui_config else {}
        except Exception:
            config = {}
        config[QT_FILTER_WINDOW_GEOMETRY_KEY] = list(geometry)
        try:
            _save_gui_config(config)
        except Exception:
            pass

    @staticmethod
    def _normalize_geometry_value(value: Any) -> tuple[int, int, int, int] | None:
        if not isinstance(value, (list, tuple)):
            return None
        if len(value) < 4:
            return None
        try:
            coords = [int(float(entry)) for entry in value[:4]]
        except Exception:
            return None
        width = max(1, coords[2])
        height = max(1, coords[3])
        if width <= 0 or height <= 0:
            return None
        return coords[0], coords[1], width, height

    def _prepare_streaming_mode(self, payload: Any, initial_overrides: Any) -> None:
        """Enable incremental ingestion when the dialog is opened in stream mode."""

        self._stream_loaded_count = 0
        self._streaming_active = True
        self._streaming_completed = False
        self._tree.clear()
        self._entry_check_state = []
        self._entry_items = []
        self._group_item_map = {}
        self._group_entries = {}
        self._cluster_index_to_group_key = {}
        self._selected_group_keys.clear()
        self._discard_selection_bounds_state()
        self._update_summary_label()
        loading_text = self._localizer.get("filter.stream.starting", "Discovering frames…")
        self._status_label.setText(loading_text)
        self._progress_bar.setRange(0, 0)
        self._append_log(loading_text)
        if self._run_analysis_btn is not None:
            self._run_analysis_btn.setEnabled(False)
        ok_button = None
        if self._dialog_button_box is not None:
            ok_button = self._dialog_button_box.button(QDialogButtonBox.Ok)
        if ok_button is not None:
            ok_button.setEnabled(False)
        self._start_stream_worker(payload, initial_overrides)

    def _start_stream_worker(self, payload: Any, initial_overrides: Any) -> None:
        self._stop_stream_worker()
        self._stream_worker = _StreamIngestWorker(
            payload,
            initial_overrides,
            scan_recursive=self._scan_recursive,
            batch_size=self._batch_size,
        )
        self._stream_thread = QThread(self)
        self._stream_worker.moveToThread(self._stream_thread)
        self._stream_thread.started.connect(self._stream_worker.run)
        self._stream_worker.batch_ready.connect(self._on_stream_batch)
        self._stream_worker.error.connect(self._on_stream_error)
        self._stream_worker.finished.connect(self._on_stream_finished)
        self._stream_worker.finished.connect(self._stream_thread.quit)
        self._stream_worker.finished.connect(self._stream_worker.deleteLater)
        self._stream_thread.finished.connect(self._on_stream_thread_finished)
        self._stream_thread.finished.connect(self._stream_thread.deleteLater)
        self._stream_thread.start()

    def _on_stream_batch(self, entries: list[_NormalizedItem]) -> None:
        if not entries:
            return
        self._normalized_items.extend(entries)
        self._append_rows(entries)
        self._stream_loaded_count += len(entries)
        message_template = self._localizer.get(
            "filter.stream.loading",
            "Discovered {count} frame(s)…",
            count=self._stream_loaded_count,
        )
        try:
            resolved = message_template.format(count=self._stream_loaded_count)
        except Exception:
            resolved = f"Discovered {self._stream_loaded_count} frame(s)…"
        self._status_label.setText(resolved)
        self._append_log(resolved)

    def _on_stream_finished(self) -> None:
        self._streaming_active = False
        self._streaming_completed = True
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        total_loaded = len(self._normalized_items)
        message_template = self._localizer.get(
            "filter.stream.completed",
            "Finished loading {count} frame(s).",
            count=total_loaded,
        )
        try:
            resolved = message_template.format(count=total_loaded)
        except Exception:
            resolved = f"Finished loading {total_loaded} frame(s)."
        self._status_label.setText(resolved)
        self._append_log(resolved)
        if self._run_analysis_btn is not None:
            self._run_analysis_btn.setEnabled(True)
        if self._dialog_button_box is not None:
            ok_button = self._dialog_button_box.button(QDialogButtonBox.Ok)
            if ok_button is not None:
                ok_button.setEnabled(True)
        self._update_summary_label()
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()

    def _on_stream_error(self, message: str) -> None:
        if not message:
            return
        try:
            self._status_label.setText(message)
        except Exception:
            pass
        self._append_log(message, level="ERROR")

    def _on_stream_thread_finished(self) -> None:
        self._stream_thread = None
        self._stream_worker = None

    def _stop_scan_worker(self) -> None:
        worker = self._scan_worker
        thread = self._scan_thread
        self._scan_worker = None
        self._scan_thread = None
        if worker is not None:
            worker.request_stop()
        if thread is not None:
            if thread.isRunning():
                thread.quit()
                if not thread.wait(5000):
                    try:
                        thread.terminate()
                    except Exception:
                        pass
                    thread.wait(1000)
            try:
                thread.deleteLater()
            except Exception:
                pass

    def _stop_stream_worker(self) -> None:
        worker = self._stream_worker
        thread = self._stream_thread
        was_active = self._streaming_active
        self._stream_worker = None
        self._stream_thread = None
        self._streaming_active = False
        if worker is not None:
            worker.request_stop()
        if thread is not None:
            if thread.isRunning():
                thread.quit()
                if not thread.wait(5000):
                    try:
                        thread.terminate()
                    except Exception:
                        pass
                    thread.wait(1000)
            try:
                thread.deleteLater()
            except Exception:
                pass
        if was_active:
            self._streaming_completed = False

    def _populate_tree(self) -> None:
        self._ensure_entry_state_capacity()
        self._rebuild_tree(preserve_expansion=False)
        self._refresh_instrument_options()

    def _append_rows(self, entries: Sequence[_NormalizedItem]) -> None:
        if not entries:
            return
        self._ensure_entry_state_capacity()
        self._rebuild_tree(preserve_expansion=True)
        self._update_summary_label()
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()
        self._refresh_instrument_options()

    def _ensure_entry_state_capacity(self) -> None:
        target = len(self._normalized_items)
        if len(self._entry_check_state) > target:
            self._entry_check_state = self._entry_check_state[:target]
        while len(self._entry_check_state) < target:
            entry = self._normalized_items[len(self._entry_check_state)]
            self._entry_check_state.append(bool(entry.include_by_default))

    def _group_key_for_entry(self, entry: _NormalizedItem) -> _GroupKey:
        cluster_idx = entry.cluster_index if isinstance(entry.cluster_index, int) else None
        label = entry.group_label.strip() if isinstance(entry.group_label, str) else None
        if label:
            return (cluster_idx, label)
        if cluster_idx is not None:
            return (cluster_idx, None)
        return (None, None)

    def _group_label_for_key(self, key: _GroupKey) -> str:
        cluster_idx, label = key
        if label:
            return label
        if cluster_idx is not None:
            template = self._localizer.get("filter.preview.group_label", "Group {index}")
            try:
                return template.format(index=cluster_idx + 1)
            except Exception:
                return f"Group {cluster_idx + 1}"
        return self._localizer.get("filter.value.group_unassigned", "Unassigned")

    def _rebuild_tree(self, preserve_expansion: bool = True) -> None:
        if self._tree is None:
            return
        self._ensure_entry_state_capacity()
        expanded_keys: set[_GroupKey] = set()
        if preserve_expansion:
            for key, item in self._group_item_map.items():
                if item is not None and item.isExpanded():
                    expanded_keys.add(key)

        self._tree_signal_guard = True
        self._tree.blockSignals(True)
        self._tree.clear()
        self._entry_items = [None] * len(self._normalized_items)
        self._group_item_map = {}
        self._group_entries = {}
        self._cluster_index_to_group_key = {}

        groups: dict[_GroupKey, list[int]] = {}
        for idx, entry in enumerate(self._normalized_items):
            key = self._group_key_for_entry(entry)
            groups.setdefault(key, []).append(idx)

        for key, indices in groups.items():
            if not indices:
                continue
            label = self._group_label_for_key(key)
            group_item = QTreeWidgetItem(self._tree)
            group_item.setText(0, label)
            group_item.setFlags(
                Qt.ItemIsUserCheckable | _QT_TRISTATE_FLAG | Qt.ItemIsEnabled | Qt.ItemIsSelectable
            )
            group_item.setCheckState(0, Qt.Unchecked)
            group_item.setData(0, Qt.UserRole, {"group_key": key})
            self._group_item_map[key] = group_item
            self._group_entries[key] = list(indices)
            cluster_idx, _ = key
            if isinstance(cluster_idx, int):
                self._cluster_index_to_group_key[cluster_idx] = key
            for idx in indices:
                self._create_entry_item(group_item, idx)
            self._refresh_group_check_state(group_item)
            if key in expanded_keys or len(groups) <= 4:
                group_item.setExpanded(True)

        self._tree.blockSignals(False)
        self._tree_signal_guard = False
        if self._selected_group_keys:
            self._selected_group_keys = {
                key for key in self._selected_group_keys if key in self._group_item_map
            }

    def _current_group_entries(self) -> dict[_GroupKey, list[int]]:
        if self._group_entries:
            return self._group_entries
        fallback: dict[_GroupKey, list[int]] = {}
        for idx, entry in enumerate(self._normalized_items):
            key = self._group_key_for_entry(entry)
            fallback.setdefault(key, []).append(idx)
        self._group_entries = {key: indices for key, indices in fallback.items() if indices}
        return self._group_entries

    def _create_entry_item(self, parent: QTreeWidgetItem, index: int) -> None:
        entry = self._normalized_items[index]
        item = QTreeWidgetItem(parent)
        item.setText(0, entry.display_name)
        item.setText(1, self._entry_wcs_text(entry))
        item.setText(2, self._entry_instrument_text(entry))
        item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        state = Qt.Checked if self._entry_check_state[index] else Qt.Unchecked
        item.setCheckState(0, state)
        item.setData(0, Qt.UserRole, index)
        self._entry_items[index] = item

    def _refresh_entry_row(self, index: int) -> None:
        if not (0 <= index < len(self._entry_items)):
            return
        item = self._entry_items[index]
        if item is None:
            return
        entry = self._normalized_items[index]
        item.setText(1, self._entry_wcs_text(entry))
        item.setText(2, self._entry_instrument_text(entry))

    def _entry_wcs_text(self, entry: _NormalizedItem) -> str:
        return (
            self._localizer.get("filter.value.wcs_present", "Yes")
            if entry.has_wcs
            else self._localizer.get("filter.value.wcs_missing", "No")
        )

    def _entry_instrument_text(self, entry: _NormalizedItem) -> str:
        return entry.instrument or self._localizer.get("filter.value.unknown", "Unknown")

    def _refresh_group_check_state(self, group_item: QTreeWidgetItem | None) -> None:
        if group_item is None:
            return
        total = group_item.childCount()
        if total == 0:
            self._tree_signal_guard = True
            group_item.setCheckState(0, Qt.Unchecked)
            self._tree_signal_guard = False
            return
        checked = 0
        for idx in range(total):
            state = group_item.child(idx).checkState(0)
            if state == Qt.Checked:
                checked += 1
        if checked == 0:
            state = Qt.Unchecked
        elif checked == total:
            state = Qt.Checked
        else:
            state = Qt.PartiallyChecked
        self._tree_signal_guard = True
        group_item.setCheckState(0, state)
        self._tree_signal_guard = False

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------
    def _handle_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0 or self._tree_signal_guard:
            return
        if item.parent() is None:
            self._handle_group_item_changed(item)
        else:
            self._handle_entry_item_changed(item)

    def _handle_group_item_changed(self, item: QTreeWidgetItem) -> None:
        state = item.checkState(0)
        if state == Qt.PartiallyChecked:
            return
        desired = state == Qt.Checked
        for idx in range(item.childCount()):
            entry_idx = item.child(idx).data(0, Qt.UserRole)
            if isinstance(entry_idx, int):
                self._set_entry_checked(entry_idx, desired)
        self._after_selection_changed()

    def _handle_entry_item_changed(self, item: QTreeWidgetItem) -> None:
        entry_idx = item.data(0, Qt.UserRole)
        if not isinstance(entry_idx, int):
            return
        checked = item.checkState(0) == Qt.Checked
        self._set_entry_checked(entry_idx, checked)
        self._after_selection_changed()

    def _after_selection_changed(self) -> None:
        self._update_summary_label()
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()

    def _entry_is_checked(self, index: int) -> bool:
        return 0 <= index < len(self._entry_check_state) and self._entry_check_state[index]

    def _set_entry_checked(self, index: int, checked: bool, update_snapshot: bool = True) -> bool:
        if not (0 <= index < len(self._entry_check_state)):
            return False
        blocked_by_selection = False
        if checked and not self._entry_inside_selection_bounds(self._normalized_items[index]):
            checked = False
            blocked_by_selection = True
        current = self._entry_check_state[index]
        item = self._entry_items[index]
        if current == checked:
            if item is not None:
                parent = item.parent()
                self._tree_signal_guard = True
                item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                self._tree_signal_guard = False
                self._refresh_group_check_state(parent)
            return False
        self._entry_check_state[index] = checked
        if item is not None:
            parent = item.parent()
            self._tree_signal_guard = True
            item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
            self._tree_signal_guard = False
            self._refresh_group_check_state(parent)
        if update_snapshot and not blocked_by_selection:
            self._update_selection_snapshot_entry(index, checked)
        return True

    def _expand_all_groups(self) -> None:
        if self._tree is None:
            return
        for idx in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(idx).setExpanded(True)

    def _collapse_all_groups(self) -> None:
        if self._tree is None:
            return
        for idx in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(idx).setExpanded(False)

    def _select_all(self) -> None:
        self._toggle_all(True)

    def _select_none(self) -> None:
        self._toggle_all(False)

    def _toggle_all(self, enabled: bool) -> None:
        changed = False
        for idx in range(len(self._normalized_items)):
            if self._set_entry_checked(idx, enabled):
                changed = True
        if changed:
            self._after_selection_changed()

    def _collect_selected_indices(self) -> list[int]:
        return [idx for idx, checked in enumerate(self._entry_check_state) if checked]

    def _update_summary_label(self) -> None:
        total = len(self._normalized_items)
        selected = sum(1 for flag in self._entry_check_state if flag)
        summary_text = self._localizer.get(
            "filter.summary.selected",
            "{selected} of {total} frames selected",
            selected=selected,
            total=total,
        )
        try:
            summary_formatted = summary_text.format(selected=selected, total=total)
        except Exception:
            summary_formatted = f"{selected} / {total}"

        cluster_count = sum(1 for group in self._cluster_groups if group)
        if cluster_count:
            cluster_fragment = self._localizer.get("filter.summary.groups", "{groups} group(s)")
            try:
                cluster_formatted = cluster_fragment.format(groups=cluster_count)
            except Exception:
                cluster_formatted = f"{cluster_count} group(s)"
            summary_display = f"{summary_formatted} – {cluster_formatted}"
        else:
            summary_display = summary_formatted

        self._summary_label.setText(summary_display)

    def _on_sds_toggled(self, checked: bool) -> None:
        # Reset any previous auto-organisation overrides so the user can
        # recompute groups after changing SDS mode, and keep the button usable.
        self._auto_group_override_groups = None
        self._group_outline_bounds = []
        if not self._auto_group_running:
            self._set_group_buttons_enabled(True)
        # Refresh cluster and preview so the UI reflects the new mode.
        self._schedule_cluster_refresh()
        self._update_summary_label()
        self._schedule_preview_refresh()

    def _create_options_box(self):
        box = QGroupBox(self._localizer.get("filter.group.options", "Filter options"), self)
        layout = QHBoxLayout(box)

        self._auto_group_checkbox = QCheckBox(
            self._localizer.get("filter.option.auto_group", "Auto-group master tiles"),
            box,
        )
        self._auto_group_checkbox.setChecked(self._auto_group_requested)
        layout.addWidget(self._auto_group_checkbox)

        self._seestar_checkbox = QCheckBox(
            self._localizer.get("filter.option.seestar_mode", "Apply Seestar heuristics"),
            box,
        )
        self._seestar_checkbox.setChecked(self._seestar_priority)
        layout.addWidget(self._seestar_checkbox)

        self._scan_recursive_checkbox = QCheckBox(
            self._localizer.get("filter.option.scan_recursive", "Scan subfolders (recursive)"),
            box,
        )
        self._scan_recursive_checkbox.setChecked(self._scan_recursive)
        self._scan_recursive_checkbox.toggled.connect(self._on_scan_recursive_toggled)  # type: ignore[arg-type]
        layout.addWidget(self._scan_recursive_checkbox)

        layout.addStretch(1)
        return box

    @Slot(str, str)
    def _append_log_from_signal(self, message: str, level: str) -> None:
        self._append_log(message, level=level)

    def _append_log(self, message: str, level: str = "INFO") -> None:
        if not message:
            return
        text = str(message)
        try:
            level_upper = str(level or "INFO").upper()
        except Exception:
            level_upper = "INFO"
        try:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        except Exception:
            timestamp = "--:--:--"
        formatted = f"[{timestamp}] [{level_upper}] {text}"
        if self._activity_log_output is not None:
            try:
                self._activity_log_output.appendPlainText(formatted)
            except Exception:
                pass
        try:
            print(formatted)
        except Exception:
            pass
        try:
            log_method = getattr(logger, level_upper.lower(), logger.info)
        except Exception:
            log_method = logger.info
        try:
            log_method(text)
        except Exception:
            try:
                logger.info(text)
            except Exception:
                pass

    def _debug_log(self, message: str, level: str = "INFO") -> None:
        if not message:
            return
        try:
            stamp = datetime.datetime.now().strftime("%H:%M:%S")
        except Exception:
            stamp = "--:--:--"
        self._append_log(f"[FilterUI {stamp}] {message}", level=level)

    def _describe_input_source(self) -> str:
        payload = self._input_payload
        if isinstance(payload, (str, bytes)):
            return str(payload)
        try:
            if isinstance(payload, os.PathLike):
                return os.fspath(payload)
        except Exception:
            pass
        candidate = None
        for attr in ("input_dir", "path", "directory", "root"):
            try:
                value = getattr(payload, attr, None)
            except Exception:
                continue
            if value:
                candidate = value
                break
        if candidate:
            try:
                return os.fspath(candidate)
            except Exception:
                return str(candidate)
        return "<memory payload>"

    def _on_scan_recursive_toggled(self, checked: bool) -> None:
        self._scan_recursive = bool(checked)

    def _on_run_analysis(self) -> None:
        self._debug_log("Analyse clicked — preparing scan…")
        if self._streaming_active and self._stream_thread is not None and self._stream_thread.isRunning():
            wait_text = self._localizer.get(
                "filter.stream.wait",
                "Please wait for frame discovery to finish before running analysis.",
            )
            self._status_label.setText(wait_text)
            return

        if self._scan_thread is not None and self._scan_thread.isRunning():
            return

        self._status_label.setText(
            self._localizer.get("filter.scan.starting", "Starting analysis…")
        )
        self._progress_bar.setValue(0)
        if self._run_analysis_btn is not None:
            self._run_analysis_btn.setEnabled(False)

        solver_settings = self._solver_settings or {}
        astap_overrides: dict[str, Any] = {}
        if isinstance(self._config_overrides, dict):
            try:
                astap_overrides.update(self._config_overrides)
            except Exception:
                astap_overrides = dict(self._config_overrides)
        write_wcs_flag = False
        if self._write_wcs_checkbox is not None:
            try:
                write_wcs_flag = bool(self._write_wcs_checkbox.isChecked())
            except Exception:
                write_wcs_flag = False
        astap_overrides["write_wcs_to_file"] = write_wcs_flag
        self._scan_worker = _DirectoryScanWorker(
            self._normalized_items,
            solver_settings,
            self._localizer,
            astap_overrides=astap_overrides,
        )
        self._scan_thread = QThread(self)
        self._scan_worker.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(self._scan_worker.run)
        self._scan_worker.progress_changed.connect(self._on_scan_progress)
        self._scan_worker.row_updated.connect(self._on_scan_row_update)
        self._scan_worker.error.connect(self._on_scan_error)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.finished.connect(self._scan_thread.quit)
        self._scan_worker.finished.connect(self._scan_worker.deleteLater)
        self._scan_thread.finished.connect(self._scan_thread.deleteLater)
        self._debug_log("[Filter] Analyse clicked — starting directory crawl…")
        self._scan_thread.start()
        self._show_processing_overlay()

    def _resolve_preview_cap(self) -> int | None:
        """Return the integer cap applied to preview points or ``None`` for unlimited."""

        try:
            if self._preview_cap in (None, ""):
                return 200
            cap = int(self._preview_cap)
            if cap <= 0:
                return None
            return cap
        except Exception:
            return 200

    def _should_draw_group_outlines(self) -> bool:
        checkbox = self._draw_group_outlines_checkbox
        if checkbox is None:
            return False
        try:
            return bool(checkbox.isChecked())
        except Exception:
            return False

    def _should_color_footprints_by_group(self) -> bool:
        checkbox = self._color_by_group_checkbox
        if checkbox is None:
            return True
        try:
            return bool(checkbox.isChecked())
        except Exception:
            return True

    def _resolve_group_color(self, group_idx: int | None) -> str:
        if not isinstance(group_idx, int):
            return "#3f7ad6"
        cycle = self._preview_color_cycle
        if not cycle:
            return "#3f7ad6"
        return cycle[group_idx % len(cycle)]

    def _ensure_entry_coordinates(self, entry: _NormalizedItem) -> Tuple[float | None, float | None]:
        if entry.center_ra_deg is not None and entry.center_dec_deg is not None:
            return entry.center_ra_deg, entry.center_dec_deg
        header_subset = getattr(entry, "header_cache", None)
        if header_subset is None:
            path = entry.file_path
            if not path or not _path_is_file(path):
                return None, None
            header_subset = self._load_header(str(_expand_to_path(path) or path))
            if header_subset is not None:
                try:
                    entry.header_cache = header_subset
                except Exception:
                    pass
        if header_subset is None:
            return None, None

        ra_deg, dec_deg = _extract_center_from_header(header_subset)
        entry.center_ra_deg = ra_deg
        entry.center_dec_deg = dec_deg
        if isinstance(entry.original, dict):
            if ra_deg is not None and dec_deg is not None:
                entry.original["center_ra_deg"] = ra_deg
                entry.original["center_dec_deg"] = dec_deg
                entry.original["RA"] = ra_deg
                entry.original["DEC"] = dec_deg
            entry.original["header"] = header_subset
        return ra_deg, dec_deg

    def _ensure_entry_footprint(self, entry: _NormalizedItem) -> List[Tuple[float, float]] | None:
        """Compute and cache a WCS footprint for ``entry`` when possible."""

        if entry.footprint_radec:
            return entry.footprint_radec
        if WCS is None:
            return None
        header = getattr(entry, "header_cache", None)
        if header is None:
            path = entry.file_path
            if not path or not _path_is_file(path):
                return None
            header = self._load_header(str(_expand_to_path(path) or path))
            if header is not None:
                try:
                    entry.header_cache = header
                except Exception:
                    pass
        if header is None:
            return None
        try:
            wcs_obj = _build_wcs_from_header(header)
        except Exception:
            wcs_obj = None
        if wcs_obj is None or not getattr(wcs_obj, "is_celestial", False):
            return None

        nx: int | None = None
        ny: int | None = None
        try:
            nax1 = header.get("NAXIS1")
            nax2 = header.get("NAXIS2")
            if isinstance(nax1, (int, float)) and isinstance(nax2, (int, float)):
                nx = int(nax1)
                ny = int(nax2)
        except Exception:
            nx = ny = None
        if not nx or not ny:
            shape = getattr(wcs_obj, "pixel_shape", None)
            if shape is not None and len(shape) >= 2:
                try:
                    nx = int(shape[0])
                    ny = int(shape[1])
                except Exception:
                    nx = ny = None
        if not nx or not ny:
            shape = getattr(wcs_obj, "array_shape", None)
            if shape is not None and len(shape) >= 2:
                try:
                    ny = int(shape[0])
                    nx = int(shape[1])
                except Exception:
                    nx = ny = None

        if not nx or not ny or nx <= 0 or ny <= 0:
            return None

        corners_pix = [
            (0.5, 0.5),
            (nx - 0.5, 0.5),
            (nx - 0.5, ny - 0.5),
            (0.5, ny - 0.5),
        ]
        footprint: list[Tuple[float, float]] = []
        for x_pix, y_pix in corners_pix:
            try:
                sky = wcs_obj.pixel_to_world(x_pix, y_pix)  # type: ignore[attr-defined]
                ra_deg = float(getattr(sky, "ra").deg)  # type: ignore[attr-defined]
                dec_deg = float(getattr(sky, "dec").deg)  # type: ignore[attr-defined]
            except Exception:
                continue
            footprint.append((ra_deg, dec_deg))

        if not footprint:
            return None
        entry.footprint_radec = footprint
        if isinstance(entry.original, dict):
            entry.original["footprint_radec"] = footprint
        return footprint

    def _collect_preview_points(self) -> List[Tuple[int, float, float, int | None]]:
        if self._preview_canvas is None:
            return []
        limit = self._resolve_preview_cap()
        points: list[Tuple[int, float, float, int | None]] = []
        for row, entry in enumerate(self._normalized_items):
            if not self._entry_is_checked(row):
                continue
            ra_deg, dec_deg = entry.center_ra_deg, entry.center_dec_deg
            if ra_deg is None or dec_deg is None:
                ra_deg, dec_deg = self._ensure_entry_coordinates(entry)
            if ra_deg is None or dec_deg is None:
                continue
            cluster_idx = entry.cluster_index if isinstance(entry.cluster_index, int) else None
            points.append((row, ra_deg, dec_deg, cluster_idx))
            if limit is not None and len(points) >= limit:
                break
        return points

    def _resolve_preview_hard_limit(self) -> int:
        try:
            value = self._config_value("preview_hard_limit", PREVIEW_HARD_LIMIT)
            if value in (None, ""):
                return PREVIEW_HARD_LIMIT
            return max(200, int(value))
        except Exception:
            return PREVIEW_HARD_LIMIT

    def _build_preview_geometry(
        self,
        collected_points: Sequence[tuple[int, float, float, int | None]] | None = None,
    ) -> dict[str, Any]:
        colorize = self._should_color_footprints_by_group()
        preview_cap = self._resolve_preview_cap()
        hard_limit = self._resolve_preview_hard_limit()
        points_by_group: dict[int | None, list[Tuple[float, float]]] = {}
        highlight_points: list[Tuple[float, float]] = []
        footprints: list[list[Tuple[float, float]]] = []
        footprint_colors: list[Any] = []
        total_entries = 0
        points_added = 0
        highlight_cap = preview_cap if preview_cap is not None else hard_limit
        footprint_mode = "footprints"
        selected_keys = set(self._selected_group_keys)
        collected_by_row: dict[int, tuple[float, float, int | None]] = {}

        if collected_points:
            for row_idx, ra_deg, dec_deg, group_idx in collected_points:
                try:
                    ra_val = float(ra_deg)
                    dec_val = float(dec_deg)
                except Exception:
                    continue
                collected_by_row[int(row_idx)] = (ra_val, dec_val, group_idx)
                points_by_group.setdefault(group_idx, []).append((ra_val, dec_val))
                points_added += 1

        for row, entry in enumerate(self._normalized_items):
            if not self._entry_is_checked(row):
                continue
            total_entries += 1
            if hard_limit is not None and total_entries > hard_limit and footprint_mode != "centroid_only":
                footprint_mode = "centroid_only"
                footprints = []
                footprint_colors = []
            cached_coords = collected_by_row.get(row)
            group_idx: int | None = None
            if cached_coords is not None:
                ra_deg, dec_deg, group_idx = cached_coords
            else:
                ra_deg, dec_deg = entry.center_ra_deg, entry.center_dec_deg
                if ra_deg is None or dec_deg is None:
                    ra_deg, dec_deg = self._ensure_entry_coordinates(entry)
                if ra_deg is None or dec_deg is None:
                    continue
                group_idx = entry.cluster_index if isinstance(entry.cluster_index, int) else None
                if preview_cap is None or points_added < preview_cap:
                    points_by_group.setdefault(group_idx, []).append((ra_deg, dec_deg))
                    points_added += 1

            if selected_keys and self._group_key_for_entry(entry) in selected_keys:
                if highlight_cap is None or len(highlight_points) < highlight_cap:
                    try:
                        highlight_points.append((float(ra_deg), float(dec_deg)))
                    except Exception:
                        pass

            if footprint_mode == "centroid_only":
                continue
            if hard_limit is not None and len(footprints) >= hard_limit:
                footprint_mode = "thin_lines"
                continue

            footprint = entry.footprint_radec or self._ensure_entry_footprint(entry)
            if not footprint:
                continue

            segment: list[Tuple[float, float]] = []
            for point_ra, point_dec in footprint:
                try:
                    ra_val = float(point_ra)
                    dec_val = float(point_dec)
                except Exception:
                    continue
                if not (math.isfinite(ra_val) and math.isfinite(dec_val)):
                    continue
                segment.append((ra_val, dec_val))
            if len(segment) < 2:
                continue
            if segment[0] != segment[-1]:
                segment.append(segment[0])
            footprints.append(segment)
            color_value: Any = (
                self._resolve_group_color(group_idx) if colorize else "#3f7ad6"
            )
            if to_rgba is not None:
                try:
                    color_value = to_rgba(color_value)
                except Exception:
                    pass
            footprint_colors.append(color_value)
            if hard_limit is not None and len(footprints) >= hard_limit:
                footprint_mode = "thin_lines"

        if hard_limit is not None and total_entries > hard_limit:
            footprint_mode = "centroid_only"
            footprints = []
            footprint_colors = []

        group_count = sum(1 for key in points_by_group.keys() if isinstance(key, int))
        preview_count = points_added
        return {
            "points_by_group": points_by_group,
            "highlight_points": highlight_points,
            "footprints": footprints,
            "footprint_colors": footprint_colors,
            "footprint_mode": footprint_mode,
            "total_entries": total_entries,
            "group_count": group_count,
            "preview_count": preview_count,
        }

    def _render_sky_preview_fast(
        self,
        axes: Any,
        geometry: dict[str, Any],
        *,
        selection_bounds: tuple[float, float, float, float] | None,
        outline_bounds: list[_GroupOutline],
        should_draw_outlines: bool,
        timing: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        points_by_group: dict[int | None, list[Tuple[float, float]]] = geometry.get("points_by_group") or {}
        highlight_points: list[Tuple[float, float]] = geometry.get("highlight_points") or []
        footprints: list[list[Tuple[float, float]]] = geometry.get("footprints") or []
        footprint_colors: list[Any] = geometry.get("footprint_colors") or []
        footprint_mode = str(geometry.get("footprint_mode") or "footprints")
        preview_count = int(geometry.get("preview_count") or 0)
        total_entries = int(geometry.get("total_entries") or 0)
        group_count = int(geometry.get("group_count") or 0)
        if timing is not None:
            timing["build_arrays_dt"] = 0.0
            timing["add_artists_dt"] = 0.0

        axes.clear()
        if self._group_outline_collection is not None:
            try:
                self._group_outline_collection.remove()
            except Exception:
                pass
            self._group_outline_collection = None

        if not outline_bounds:
            candidate_groups: list[list[Any]] | None = None
            if self._auto_group_override_groups:
                candidate_groups = self._auto_group_override_groups
            elif self._cluster_groups:
                candidate_groups = self._cluster_groups
            if candidate_groups:
                try:
                    outline_bounds = self._compute_group_outline_bounds(candidate_groups)
                except Exception:
                    outline_bounds = []
        self._group_outline_bounds = outline_bounds

        axes.set_xlabel(self._localizer.get("filter.preview.ra", "Right Ascension (°)"))
        axes.set_ylabel(self._localizer.get("filter.preview.dec", "Declination (°)"))
        axes.set_title(self._localizer.get("filter.preview.title", "Sky preview"))
        axes.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

        if preview_count <= 0 or not any(points_by_group.values()):
            message = self._localizer.get(
                "filter.preview.empty",
                "No WCS information available for the current selection.",
            )
            if not self._preview_empty_logged:
                hint = self._localizer.get(
                    "filter.preview.empty_hint",
                    "No WCS/center information available; the sky preview will remain empty but you can still select files.",
                )
                self._append_log(hint, level="WARN")
                self._preview_empty_logged = True
            axes.text(0.5, 0.5, message, ha="center", va="center", transform=axes.transAxes)
            self._preview_hint_label.setText(self._preview_default_hint)
            return {
                "legend": False,
                "footprint_mode": footprint_mode,
                "preview_count": preview_count,
                "total_entries": total_entries,
            }
        self._preview_empty_logged = False

        colorize = self._should_color_footprints_by_group()
        legend_cap = max(3, PREVIEW_LEGEND_MAX_GROUPS)
        legend_allowed = colorize and group_count > 0 and group_count <= legend_cap
        legend_hidden_hint = colorize and group_count > legend_cap
        ra_values: list[float] = []
        dec_values: list[float] = []
        scatter_ra: list[float] = []
        scatter_dec: list[float] = []
        scatter_colors: list[Any] = []
        legend_labels: list[tuple[str, Any]] = []
        outline_segments: list[list[tuple[float, float]]] = []
        outline_colors: list[Any] = []
        outline_corners: list[tuple[float, float]] = []
        build_arrays_start = time.monotonic()
        default_color = "#3f7ad6"
        for group_idx, coords in points_by_group.items():
            if not coords:
                continue
            color = self._resolve_group_color(group_idx) if colorize else default_color
            appended = 0
            for coord in coords:
                try:
                    ra_val = float(coord[0])
                    dec_val = float(coord[1])
                except Exception:
                    continue
                scatter_ra.append(ra_val)
                scatter_dec.append(dec_val)
                scatter_colors.append(color)
                ra_values.append(ra_val)
                dec_values.append(dec_val)
                appended += 1
            if legend_allowed and appended:
                label = None
                if isinstance(group_idx, int):
                    label_template = self._localizer.get("filter.preview.group_label", "Group {index}")
                    try:
                        label = label_template.format(index=group_idx + 1)
                    except Exception:
                        label = f"Group {group_idx + 1}"
                elif len(points_by_group) > 1:
                    label = self._localizer.get("filter.preview.group_unassigned", "Unassigned")
                if label:
                    legend_labels.append((label, color))
        colorize_outlines = self._should_color_footprints_by_group()
        default_outline_color: Any = to_rgba("red", 0.9) if to_rgba is not None else "#d64b3f"
        if should_draw_outlines and outline_bounds:
            for outline_entry in outline_bounds:
                group_idx: int | None = None
                coords: tuple[float, float, float, float] | None = None
                if isinstance(outline_entry, (list, tuple)):
                    if len(outline_entry) == 5:
                        idx_candidate = outline_entry[0]
                        if isinstance(idx_candidate, int):
                            group_idx = idx_candidate
                        try:
                            ra_min = float(outline_entry[1])
                            ra_max = float(outline_entry[2])
                            dec_min = float(outline_entry[3])
                            dec_max = float(outline_entry[4])
                            coords = (ra_min, ra_max, dec_min, dec_max)
                        except Exception:
                            coords = None
                    elif len(outline_entry) == 4:
                        try:
                            ra_min = float(outline_entry[0])
                            ra_max = float(outline_entry[1])
                            dec_min = float(outline_entry[2])
                            dec_max = float(outline_entry[3])
                            coords = (ra_min, ra_max, dec_min, dec_max)
                        except Exception:
                            coords = None
                if coords is None:
                    continue
                ra_min, ra_max, dec_min, dec_max = coords
                width = float(ra_max) - float(ra_min)
                height = float(dec_max) - float(dec_min)
                if width <= 0 or height <= 0:
                    continue
                corners = [
                    (float(ra_min), float(dec_min)),
                    (float(ra_max), float(dec_min)),
                    (float(ra_max), float(dec_max)),
                    (float(ra_min), float(dec_max)),
                    (float(ra_min), float(dec_min)),
                ]
                outline_segments.append(corners)
                outline_corners.extend(corners)
                outline_color = default_outline_color
                if colorize_outlines and isinstance(group_idx, int):
                    outline_color = (
                        to_rgba(self._resolve_group_color(group_idx), 0.95)
                        if to_rgba is not None
                        else self._resolve_group_color(group_idx)
                    )
                outline_colors.append(outline_color)
        build_arrays_dt = time.monotonic() - build_arrays_start
        if timing is not None:
            timing["build_arrays_dt"] = build_arrays_dt

        add_artists_start = time.monotonic()
        if scatter_ra and scatter_dec:
            axes.scatter(
                scatter_ra,
                scatter_dec,
                c=scatter_colors or default_color,
                s=24,
                alpha=0.85,
                edgecolors="none",
                zorder=2,
            )

        if selection_bounds and Rectangle is not None:
            try:
                ra_min, ra_max, dec_min, dec_max = selection_bounds
                width = float(ra_max) - float(ra_min)
                height = float(dec_max) - float(dec_min)
                if width > 0 and height > 0:
                    selection_rect = Rectangle(
                        (float(ra_min), float(dec_min)),
                        width,
                        height,
                        linewidth=1.4,
                        edgecolor="#1f54d6",
                        facecolor=(0.1, 0.35, 0.85, 0.18),
                        linestyle="-",
                        zorder=4,
                    )
                    axes.add_patch(selection_rect)
                    ra_values.extend([float(ra_min), float(ra_max)])
                    dec_values.extend([float(dec_min), float(dec_max)])
            except Exception:
                pass

        if highlight_points:
            axes.scatter(
                [pt[0] for pt in highlight_points],
                [pt[1] for pt in highlight_points],
                s=80,
                facecolors="none",
                edgecolors="#f5f542",
                linewidths=1.5,
                zorder=6,
            )

        if footprints and LineCollection is not None and footprint_mode != "centroid_only":
            footprint_collection = LineCollection(
                footprints,
                colors=footprint_colors or "#3f7ad6",
                linewidths=0.8 if footprint_mode == "thin_lines" else 1.2,
                linestyles="-",
                alpha=0.65 if footprint_mode == "thin_lines" else 0.9,
                zorder=3,
            )
            axes.add_collection(footprint_collection)
            try:
                for segment in footprints:
                    for ra_val, dec_val in segment:
                        ra_values.append(float(ra_val))
                        dec_values.append(float(dec_val))
            except Exception:
                pass

        if outline_segments and LineCollection is not None:
            coll = LineCollection(
                outline_segments,
                colors=outline_colors or [default_outline_color],
                linewidths=1.6,
                linestyles="--",
                alpha=0.9,
                zorder=5,
            )
            axes.add_collection(coll)
            self._group_outline_collection = coll
        else:
            self._group_outline_collection = None

        legend_shown = False
        if legend_allowed and legend_labels and Line2D is not None:
            handles: list[Any] = []
            seen_labels: set[str] = set()
            for label, color in legend_labels:
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                try:
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            linestyle="",
                            markersize=6,
                            markerfacecolor=color,
                            markeredgecolor=color,
                            alpha=0.85,
                        )
                    )
                except Exception:
                    continue
            if handles:
                axes.legend(handles=handles, loc="upper right", fontsize="small")
                legend_shown = True
        add_artists_dt = time.monotonic() - add_artists_start
        if timing is not None:
            timing["add_artists_dt"] = add_artists_dt

        for ra_corner, dec_corner in outline_corners:
            ra_values.append(ra_corner)
            dec_values.append(dec_corner)

        if ra_values and dec_values:
            ra_min, ra_max = min(ra_values), max(ra_values)
            dec_min, dec_max = min(dec_values), max(dec_values)
            if ra_min == ra_max:
                ra_margin = max(1.0, abs(ra_min) * 0.01 + 0.5)
            else:
                ra_margin = (ra_max - ra_min) * 0.05
            if dec_min == dec_max:
                dec_margin = max(1.0, abs(dec_min) * 0.01 + 0.5)
            else:
                dec_margin = (dec_max - dec_min) * 0.05
            axes.set_xlim(ra_max + ra_margin, ra_min - ra_margin)
            axes.set_ylim(dec_min - dec_margin, dec_max + dec_margin)
            axes.set_aspect("equal", adjustable="datalim")

        preview_cap_value = self._resolve_preview_cap()
        summary_text = self._localizer.get(
            "filter.preview.summary",
            "Showing {count} frame(s) in preview (cap {cap}).",
            count=preview_count,
            cap=preview_cap_value or "∞",
        )
        clusters_present = group_count
        if clusters_present:
            cluster_fragment = self._localizer.get("filter.preview.groups_hint", "{groups} cluster(s)")
            try:
                cluster_formatted = cluster_fragment.format(groups=clusters_present)
            except Exception:
                cluster_formatted = f"{clusters_present} cluster(s)"
            summary_text = f"{summary_text} – {cluster_formatted}"
        mode_fragment = f"mode={footprint_mode}"
        summary_text = f"{summary_text} [{mode_fragment}]"
        if legend_hidden_hint:
            legend_hint_text = self._localizer.get(
                "filter.preview.legend_hidden_hint",
                "Legend hidden (too many groups)",
            )
            summary_text = f"{summary_text} – {legend_hint_text}"
        try:
            self._preview_hint_label.setText(
                summary_text.format(count=preview_count, cap=preview_cap_value or "∞")
            )
        except Exception:
            self._preview_hint_label.setText(summary_text)

        return {
            "legend": legend_shown and legend_allowed,
            "legend_hidden": legend_hidden_hint,
            "footprint_mode": footprint_mode,
            "preview_count": preview_count,
            "total_entries": total_entries,
            "group_count": group_count,
        }

    def _on_preview_rectangle_selected(self, eclick, erelease) -> None:
        if eclick is None or erelease is None:
            return
        if eclick.xdata is None or erelease.xdata is None or eclick.ydata is None or erelease.ydata is None:
            return
        try:
            ra1 = float(eclick.xdata)
            ra2 = float(erelease.xdata)
            dec1 = float(eclick.ydata)
            dec2 = float(erelease.ydata)
        except Exception:
            return
        if not all(math.isfinite(value) for value in (ra1, ra2, dec1, dec2)):
            return
        ra_min, ra_max = (ra1, ra2) if ra1 <= ra2 else (ra2, ra1)
        dec_min, dec_max = (dec1, dec2) if dec1 <= dec2 else (dec2, dec1)
        if self._selection_bounds is None:
            self._capture_selection_snapshot()
        else:
            self._restore_selection_state_for_reselection()
        self._selection_bounds = (ra_min, ra_max, dec_min, dec_max)
        selected_keys = self._resolve_groups_in_bounds(ra_min, ra_max, dec_min, dec_max)
        self._apply_group_selection(selected_keys)
        self._apply_selection_bounds_constraints()
        if selected_keys:
            message_template = self._localizer.get(
                "filter.preview.selection_compact",
                "Selected {count} group(s) from sky preview.",
            )
            try:
                message = message_template.format(count=len(selected_keys))
            except Exception:
                message = f"Selected {len(selected_keys)} group(s)."
            self._status_label.setText(message)
            self._append_log(message)
        else:
            cleared_message = self._localizer.get(
                "filter.preview.selection_cleared",
                "No groups intersected the selected region.",
            )
            self._status_label.setText(cleared_message)
            self._append_log(cleared_message)
        self._schedule_preview_refresh()

    def _on_preview_context_menu(self, point: QPoint) -> None:
        if self._preview_canvas is None:
            return
        menu = QMenu(self._preview_canvas)
        label = self._localizer.get(
            "qt_filter_clear_bbox",
            "Clear selection bounding box",
        )
        clear_action = menu.addAction(label)
        clear_action.setEnabled(self._selection_bounds is not None)
        clear_action.triggered.connect(lambda: self._clear_preview_selection())  # type: ignore[arg-type]
        menu.exec(self._preview_canvas.mapToGlobal(point))

    def _clear_preview_selection(self) -> None:
        if self._selection_bounds is None:
            return
        self._selection_bounds = None
        self._restore_selection_snapshot()
        if self._selected_group_keys:
            self._selected_group_keys.clear()
        cleared_message = self._localizer.get(
            "filter.preview.selection_cleared",
            "Selection bounding box cleared.",
        )
        self._status_label.setText(cleared_message)
        self._append_log(cleared_message)
        self._schedule_preview_refresh()

    def _resolve_groups_in_bounds(
        self,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
    ) -> set[_GroupKey]:
        candidates = self._group_entries
        if not candidates:
            candidates = self._current_group_entries()
            if not candidates:
                return set()
        selected: set[_GroupKey] = set()
        for key, indices in candidates.items():
            if not indices:
                continue
            if self._group_intersects_bounds(indices, ra_min, ra_max, dec_min, dec_max):
                selected.add(key)
        return selected

    def _group_intersects_bounds(
        self,
        indices: Sequence[int],
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
    ) -> bool:
        for idx in indices:
            if not (0 <= idx < len(self._normalized_items)):
                continue
            entry = self._normalized_items[idx]
            if self._entry_intersects_bounds(entry, ra_min, ra_max, dec_min, dec_max):
                return True
        return False

    def _entry_intersects_bounds(
        self,
        entry: _NormalizedItem,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
    ) -> bool:
        ra_deg, dec_deg = entry.center_ra_deg, entry.center_dec_deg
        if ra_deg is None or dec_deg is None:
            ra_deg, dec_deg = self._ensure_entry_coordinates(entry)
        if (
            ra_deg is not None
            and dec_deg is not None
            and self._point_in_bounds(ra_deg, dec_deg, ra_min, ra_max, dec_min, dec_max)
        ):
            return True
        footprint = entry.footprint_radec or self._ensure_entry_footprint(entry)
        if footprint:
            for point_ra, point_dec in footprint:
                if self._point_in_bounds(point_ra, point_dec, ra_min, ra_max, dec_min, dec_max):
                    return True
            ra_values = [pt[0] for pt in footprint]
            dec_values = [pt[1] for pt in footprint]
            if ra_values and dec_values:
                f_ra_min = min(ra_values)
                f_ra_max = max(ra_values)
                f_dec_min = min(dec_values)
                f_dec_max = max(dec_values)
                if self._rectangles_intersect(
                    ra_min, ra_max, dec_min, dec_max, f_ra_min, f_ra_max, f_dec_min, f_dec_max
                ):
                    return True
        return False

    @staticmethod
    def _point_in_bounds(
        ra: float,
        dec: float,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
    ) -> bool:
        return ra_min <= ra <= ra_max and dec_min <= dec <= dec_max

    @staticmethod
    def _rectangles_intersect(
        ra_min_a: float,
        ra_max_a: float,
        dec_min_a: float,
        dec_max_a: float,
        ra_min_b: float,
        ra_max_b: float,
        dec_min_b: float,
        dec_max_b: float,
    ) -> bool:
        return not (
            ra_max_b < ra_min_a
            or ra_min_b > ra_max_a
            or dec_max_b < dec_min_a
            or dec_min_b > dec_max_a
        )

    def _apply_group_selection(self, selected_keys: set[_GroupKey]) -> None:
        if not selected_keys:
            if self._selected_group_keys:
                self._selected_group_keys.clear()
                self._schedule_preview_refresh()
            return
        changed = False
        for key in selected_keys:
            entries = self._group_entries.get(key)
            if not entries:
                continue
            group_item = self._group_item_map.get(key)
            if group_item is not None:
                group_item.setExpanded(True)
            for idx in entries:
                if self._set_entry_checked(idx, True):
                    changed = True
        self._selected_group_keys = set(selected_keys)
        if changed:
            self._after_selection_changed()
        else:
            self._schedule_preview_refresh()

    def _capture_selection_snapshot(self) -> None:
        if self._selection_check_snapshot is None:
            self._selection_check_snapshot = list(self._entry_check_state)

    def _restore_selection_state_for_reselection(self) -> None:
        snapshot = self._selection_check_snapshot
        if not snapshot or len(snapshot) != len(self._entry_check_state):
            return
        changed = False
        for idx, desired in enumerate(snapshot):
            if self._set_entry_checked(idx, bool(desired), update_snapshot=False):
                changed = True
        if changed:
            self._after_selection_changed()

    def _restore_selection_snapshot(self) -> None:
        snapshot = self._selection_check_snapshot
        self._selection_check_snapshot = None
        if not snapshot or len(snapshot) != len(self._entry_check_state):
            return
        changed = False
        for idx, desired in enumerate(snapshot):
            if self._set_entry_checked(idx, bool(desired)):
                changed = True
        if changed:
            self._after_selection_changed()
        else:
            self._schedule_preview_refresh()

    def _update_selection_snapshot_entry(self, index: int, checked: bool) -> None:
        if self._selection_check_snapshot is None:
            return
        if not (0 <= index < len(self._selection_check_snapshot)):
            return
        self._selection_check_snapshot[index] = bool(checked)

    def _discard_selection_bounds_state(self) -> None:
        self._selection_bounds = None
        self._selection_check_snapshot = None

    def _entry_inside_selection_bounds(
        self,
        entry: _NormalizedItem,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> bool:
        selection_bounds = bounds if bounds is not None else self._selection_bounds
        if selection_bounds is None:
            return True
        ra_val = entry.center_ra_deg
        dec_val = entry.center_dec_deg
        if ra_val is None or dec_val is None:
            ra_val, dec_val = self._ensure_entry_coordinates(entry)
        if ra_val is None or dec_val is None:
            return False
        ra_min, ra_max, dec_min, dec_max = selection_bounds
        if ra_min <= ra_max:
            ra_ok = ra_min <= ra_val <= ra_max
        else:
            ra_ok = ra_val >= ra_min or ra_val <= ra_max
        dec_ok = dec_min <= dec_val <= dec_max
        return ra_ok and dec_ok

    def _apply_selection_bounds_constraints(self) -> None:
        if self._selection_bounds is None:
            return
        changed = False
        for idx, entry in enumerate(self._normalized_items):
            if not self._entry_inside_selection_bounds(entry):
                # Skip snapshot updates so entries outside the bbox can be restored later.
                if self._set_entry_checked(idx, False, update_snapshot=False):
                    changed = True
                continue
            # Ensure the snapshot keeps track of entries that remain inside the bbox.
            if self._selection_check_snapshot is not None:
                self._update_selection_snapshot_entry(idx, self._entry_check_state[idx])
        if changed:
            self._after_selection_changed()

    def _filter_indices_by_selection_bounds(self, indices: Sequence[int]) -> list[int]:
        if self._selection_bounds is None:
            return list(indices)
        filtered: list[int] = []
        for idx in indices:
            if 0 <= idx < len(self._normalized_items):
                entry = self._normalized_items[idx]
                if self._entry_inside_selection_bounds(entry):
                    filtered.append(idx)
        return filtered

    def _schedule_preview_refresh(self) -> None:
        if self._preview_canvas is None or self._preview_refresh_pending:
            return
        now = time.monotonic()
        interval = max(0.0, self._preview_refresh_interval)
        elapsed = now - self._preview_last_refresh
        delay_ms = 0
        if elapsed < interval:
            delay_ms = int((interval - elapsed) * 1_000)
        self._preview_refresh_pending = True
        QTimer.singleShot(delay_ms, self._update_preview_plot)

    def _dispose_preview_canvas(self) -> None:
        canvas = self._preview_canvas
        self._preview_canvas = None
        self._preview_axes = None
        self._group_outline_collection = None
        self._preview_refresh_pending = False
        self._preview_last_refresh = 0.0
        if self._rectangle_selector is not None:
            try:
                self._rectangle_selector.disconnect_events()
            except Exception:
                pass
            try:
                self._rectangle_selector.set_active(False)
            except Exception:
                pass
            self._rectangle_selector = None
        if canvas is not None:
            try:
                canvas.setParent(None)
            except Exception:
                pass
            try:
                canvas.deleteLater()
            except Exception:
                pass
            try:
                canvas.hide()
            except Exception:
                pass

    def _dispose_coverage_canvas(self) -> None:
        canvas = self._coverage_canvas
        self._coverage_canvas = None
        self._coverage_axes = None
        if canvas is not None:
            try:
                canvas.setParent(None)
            except Exception:
                pass
            try:
                canvas.deleteLater()
            except Exception:
                pass


    def _on_preview_canvas_destroyed(self, _obj: QObject | None = None) -> None:  # pragma: no cover - Qt signal glue
        self._dispose_preview_canvas()

    def _on_coverage_canvas_destroyed(self, _obj: QObject | None = None) -> None:  # pragma: no cover - Qt signal glue
        self._dispose_coverage_canvas()

    def _safe_draw_preview_canvas(self, *, force: bool = False) -> bool:
        if force:
            self._last_preview_draw_ts = 0.0
        return self._safe_draw_canvas("preview")

    def _safe_draw_coverage_canvas(self) -> bool:
        return self._safe_draw_canvas("coverage")

    def _safe_draw_canvas(self, canvas_type: str) -> bool:
        canvas = self._preview_canvas if canvas_type == "preview" else self._coverage_canvas
        if canvas is None:
            return False
        try:
            canvas.draw_idle()
            if canvas_type == "preview":
                self._preview_draw_attempts += 1
        except RuntimeError as exc:
            message = str(exc).lower()
            if "already deleted" not in message:
                self._append_log(f"{canvas_type.capitalize()} canvas redraw failed: {exc}", level="WARN")
            if canvas_type == "preview":
                self._dispose_preview_canvas()
            else:
                self._dispose_coverage_canvas()
            return False
        return True

    def _throttled_preview_draw(self, *, force: bool = False) -> bool:
        if self._preview_canvas is None:
            return False
        now = time.monotonic()
        throttle = max(0.0, float(self._preview_draw_throttle or 0.0))
        if not force and self._last_preview_draw_ts and (now - self._last_preview_draw_ts) < throttle:
            return False
        drawn = self._safe_draw_preview_canvas(force=force)
        if drawn:
            self._last_preview_draw_ts = time.monotonic()
        return drawn

    def _schedule_cluster_refresh(self) -> None:
        if self._cluster_refresh_pending:
            return
        self._cluster_refresh_pending = True
        QTimer.singleShot(0, self._update_cluster_assignments)

    def _update_cluster_assignments(self) -> None:
        self._cluster_refresh_pending = False
        groups, threshold = self._compute_cluster_groups()
        self._cluster_groups = groups
        self._cluster_threshold_used = threshold
        if groups:
            try:
                self._group_outline_bounds = self._compute_group_outline_bounds(groups)
            except Exception:
                self._group_outline_bounds = []
        else:
            self._group_outline_bounds = []

        assigned_ids: set[int] = set()
        label_template = self._localizer.get("filter.group.auto_label", "Group {index}")
        for idx, group in enumerate(groups, start=1):
            try:
                label = label_template.format(index=idx)
            except Exception:
                label = f"Group {idx}"
            for entry in group:
                entry.group_label = label
                entry.cluster_index = idx - 1
                assigned_ids.add(id(entry))

        for entry in self._normalized_items:
            if id(entry) not in assigned_ids:
                entry.group_label = None
                entry.cluster_index = None

        self._rebuild_tree(preserve_expansion=True)

        if groups and (self._scan_thread is None or not self._scan_thread.isRunning()):
            threshold_value = threshold if isinstance(threshold, (int, float)) else 0.0
            summary_template = self._localizer.get(
                "filter.cluster.summary",
                "Auto-grouped {groups} cluster(s) (threshold {threshold:.2f}°)",
            )
            try:
                summary_text = summary_template.format(
                    groups=len(groups),
                    threshold=threshold_value,
                )
            except Exception:
                summary_text = f"Auto-grouped {len(groups)} clusters"
            self._status_label.setText(summary_text)

        self._update_summary_label()
        self._schedule_preview_refresh()

    def _compute_cluster_groups(self) -> Tuple[List[List[_NormalizedItem]], float | None]:
        threshold = self._resolve_cluster_threshold()
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            return [], threshold

        groups: list[list[_NormalizedItem]] = []
        centroids: list[Tuple[float, float]] = []
        for row, entry in enumerate(self._normalized_items):
            if not self._entry_is_checked(row):
                continue
            ra_deg, dec_deg = entry.center_ra_deg, entry.center_dec_deg
            if ra_deg is None or dec_deg is None:
                ra_deg, dec_deg = self._ensure_entry_coordinates(entry)
            if ra_deg is None or dec_deg is None:
                continue

            assigned_idx = None
            for idx, (cen_ra, cen_dec) in enumerate(centroids):
                distance = self._angular_distance_deg(ra_deg, dec_deg, cen_ra, cen_dec)
                if distance <= threshold:
                    assigned_idx = idx
                    break

            if assigned_idx is None:
                groups.append([entry])
                centroids.append((ra_deg, dec_deg))
            else:
                groups[assigned_idx].append(entry)
                count = len(groups[assigned_idx])
                cen_ra, cen_dec = centroids[assigned_idx]
                new_ra = cen_ra + (ra_deg - cen_ra) / count
                new_dec = cen_dec + (dec_deg - cen_dec) / count
                centroids[assigned_idx] = (new_ra, new_dec)

        non_empty = [group for group in groups if group]
        return non_empty, threshold

    def _resolve_cluster_threshold(self) -> float:
        candidates: tuple[Any, ...] = (
            self._safe_lookup(self._config_overrides, "cluster_panel_threshold"),
            self._safe_lookup(self._initial_overrides, "cluster_panel_threshold"),
            self._safe_lookup(self._solver_settings, "cluster_panel_threshold"),
            self._safe_lookup(self._config_overrides, "panel_clustering_threshold_deg"),
            self._safe_lookup(self._solver_settings, "panel_clustering_threshold_deg"),
        )
        for candidate in candidates:
            try:
                value = float(candidate)
            except Exception:
                continue
            if value > 0:
                return max(0.02, min(5.0, value))
        return 0.25

    @staticmethod
    def _angular_distance_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
        dra = abs(float(ra1) - float(ra2)) % 360.0
        if dra > 180.0:
            dra = 360.0 - dra
        ddec = abs(float(dec1) - float(dec2))
        return math.hypot(dra, ddec)

    def _update_preview_plot(self) -> None:
        total_start = time.monotonic()
        self._preview_refresh_pending = False
        self._preview_last_refresh = time.monotonic()
        if self._preview_axes is None or self._preview_canvas is None:
            return

        axes = self._preview_axes
        start_draws = self._preview_draw_attempts
        collect_start = time.monotonic()
        collected_points = self._collect_preview_points()
        collect_dt = time.monotonic() - collect_start

        build_start = time.monotonic()
        geometry = self._build_preview_geometry(collected_points)
        outline_bounds = self._group_outline_bounds
        should_draw_outlines = self._should_draw_group_outlines()
        build_dt = time.monotonic() - build_start

        render_timing: dict[str, float] = {}
        render_result = self._render_sky_preview_fast(
            axes,
            geometry,
            selection_bounds=self._selection_bounds,
            outline_bounds=outline_bounds or [],
            should_draw_outlines=should_draw_outlines,
            timing=render_timing,
        )
        build_arrays_dt = build_dt + float(render_timing.get("build_arrays_dt", 0.0))
        add_artists_dt = float(render_timing.get("add_artists_dt", 0.0))

        draw_start = time.monotonic()
        force_draw = not getattr(self, "_auto_group_running", False)
        drew = self._throttled_preview_draw(force=force_draw)
        draw_dt = time.monotonic() - draw_start
        redraw_delta = self._preview_draw_attempts - start_draws
        if not drew:
            delay_ms = int(max(0.0, float(self._preview_draw_throttle or 0.0)) * 1_000)
            QTimer.singleShot(delay_ms, lambda: self._throttled_preview_draw(force=True))
        try:
            total_dt = time.monotonic() - total_start
            perf_message = (
                "sky_preview_perf: N=%s groups=%s collect=%.3fs build=%.3fs artists=%.3fs draw=%.3fs "
                "total=%.3fs redraws=%s mode=%s"
            ) % (
                render_result.get("preview_count"),
                render_result.get("group_count", geometry.get("group_count", 0)),
                collect_dt,
                build_arrays_dt,
                add_artists_dt,
                draw_dt,
                total_dt,
                redraw_delta,
                render_result.get("footprint_mode", "footprints"),
            )
            logger.info(
                "%s total=%s",
                perf_message,
                render_result.get("total_entries"),
            )
            self._append_log(perf_message)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # QDialog API
    # ------------------------------------------------------------------
    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_overlay_geometry()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._persist_window_geometry()
        self._stop_stream_worker()
        self._stop_scan_worker()
        self._hide_processing_overlay()
        self._dispose_preview_canvas()
        self._dispose_coverage_canvas()
        self._safe_shutdown_matplotlib()
        super().closeEvent(event)

    def _safe_shutdown_matplotlib(self) -> None:
        """Safely close all Matplotlib figures during shutdown."""
        if plt is None:
            return
        try:
            plt.close("all")
        except Exception:
            pass


    def accept(self) -> None:  # noqa: D401 - inherit docstring
        self._accepted = True
        super().accept()

    def reject(self) -> None:  # noqa: D401 - inherit docstring
        self._accepted = False
        super().reject()

    # ------------------------------------------------------------------
    # Public helpers queried by ``launch_filter_interface_qt``
    # ------------------------------------------------------------------
    def _serialize_entry_for_worker(
        self,
        entry: _NormalizedItem,
        row_index: int | None = None,
        *,
        include_header: bool = True,
    ) -> dict[str, Any]:
        """Return a Tk-compatible dict payload for the given table entry."""

        original = entry.original
        if isinstance(original, dict):
            try:
                payload: dict[str, Any] = dict(original)
            except Exception:
                payload = {key: original[key] for key in original.keys()}
            payload.pop("wcs", None)
        else:
            payload = {}
            if isinstance(original, (str, os.PathLike)):
                try:
                    payload["path"] = os.fspath(original)
                except Exception:
                    payload["path"] = str(original)

        path_value = entry.file_path or payload.get("path") or payload.get("file_path")
        if path_value:
            payload["path"] = str(path_value)
            payload.setdefault("file_path", payload["path"])
        if row_index is not None:
            payload.setdefault("index", row_index)

        payload["has_wcs"] = bool(entry.has_wcs or payload.get("wcs"))

        if entry.instrument and not payload.get("instrument"):
            payload["instrument"] = entry.instrument
        if entry.group_label and not payload.get("group_label"):
            payload["group_label"] = entry.group_label

        # Persist inclusion flag so Tk/worker heuristics behave identically.
        if "include_by_default" not in payload:
            payload["include_by_default"] = bool(entry.include_by_default)

        # Attach header / WCS metadata when available so the worker can skip
        # its own header scan just like it does when Tk provides this list.
        header_obj = (
            payload.get("header")
            or payload.get("header_subset")
            or getattr(entry, "header_cache", None)
        )
        if include_header and header_obj is not None:
            payload["header"] = header_obj

        # Surface RA/DEC + footprint metadata for downstream grouping parity.
        ra_deg, dec_deg = entry.center_ra_deg, entry.center_dec_deg
        if ra_deg is not None:
            payload.setdefault("center_ra_deg", float(ra_deg))
            payload["RA"] = float(ra_deg)
        if dec_deg is not None:
            payload.setdefault("center_dec_deg", float(dec_deg))
            payload["DEC"] = float(dec_deg)
        footprint = entry.footprint_radec or payload.get("footprint_radec")
        if footprint:
            payload["footprint_radec"] = footprint

        if entry.cluster_index is not None:
            payload["cluster_index"] = entry.cluster_index

        return payload

    def _should_include_header_for_entry(self, entry: _NormalizedItem) -> bool:
        """Return True when we already have cached metadata for ``entry``."""

        if getattr(entry, "header_cache", None) is not None:
            return True
        if getattr(entry, "wcs_cache", None) is not None:
            return True
        return not self._stream_scan

    def selected_items(self) -> List[Any]:
        results: list[Any] = []
        for row, entry in enumerate(self._normalized_items):
            if self._entry_is_checked(row):
                include_header = self._should_include_header_for_entry(entry)
                results.append(
                    self._serialize_entry_for_worker(entry, row, include_header=include_header)
                )
        return results

    def was_accepted(self) -> bool:
        return self._accepted

    def _resolved_wcs_count(self) -> int:
        return sum(1 for entry in self._normalized_items if entry.has_wcs)

    @staticmethod
    def _normalize_string(value: Any) -> str | None:
        if value is None:
            return None
        try:
            text = str(value).strip()
        except Exception:
            return None
        return text or None

    def _config_value(self, key: str, default: Any = None) -> Any:
        sources: tuple[Any, ...] = (
            self._runtime_overrides if isinstance(self._runtime_overrides, dict) else None,
            self._config_overrides if isinstance(self._config_overrides, dict) else None,
            self._initial_overrides if isinstance(self._initial_overrides, dict) else None,
            DEFAULT_FILTER_CONFIG,
        )
        for mapping in sources:
            if not isinstance(mapping, dict):
                continue
            if key not in mapping:
                continue
            value = mapping.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                normalized = value.strip()
                if not normalized:
                    continue
                return normalized
            return value
        return default

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value in (None, ""):
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        try:
            text = str(value).strip().lower()
        except Exception:
            return default
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _clamp_overcap_percent(value: Any, default: int = 10) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        parsed = max(0, min(50, parsed))
        return parsed

    @staticmethod
    def _clamp_overlap_percent(value: Any, default: int = 40) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        parsed = max(0, min(70, parsed))
        return parsed

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            result = float(value)
        except Exception:
            return default
        if math.isnan(result) or math.isinf(result):
            return default
        return result

    @staticmethod
    def _coerce_int(value: Any, default: int | None = None) -> int | None:
        try:
            return int(value)
        except Exception:
            return default

    def _detect_primary_instrument_label(self) -> str | None:
        for entry in self._normalized_items:
            label = self._normalize_string(entry.instrument)
            if label:
                return label

        if fits is not None:
            header_checks = 0
            for entry in self._normalized_items:
                if not entry.file_path or not _path_is_file(entry.file_path):
                    continue
                try:
                    header = fits.getheader(str(_expand_to_path(entry.file_path) or entry.file_path), ignore_missing_end=True)
                except Exception:
                    header = None
                if header is None:
                    continue
                label = _detect_instrument_from_header(header)
                if label:
                    return label
                header_checks += 1
                if header_checks >= 5:
                    break

        payload = self._input_payload
        if isinstance(payload, (list, tuple)):
            for candidate in payload[:5]:
                label = None
                if hasattr(candidate, "instrument"):
                    try:
                        label = self._normalize_string(getattr(candidate, "instrument"))
                    except Exception:
                        label = None
                if label:
                    return label
                if isinstance(candidate, dict):
                    label = self._normalize_string(
                        candidate.get("instrument")
                        or candidate.get("INSTRUME")
                        or candidate.get("instrument_name")
                    )
                    if label:
                        return label
                    header_obj = candidate.get("header") or candidate.get("header_subset")
                    if header_obj:
                        header_label = _detect_instrument_from_header(header_obj)
                        if header_label:
                            return header_label
        return None

    def _determine_workflow_mode(self, instrument_label: str | None) -> str:
        if self._coerce_bool(self._config_value("force_seestar_mode", False), False):
            return "seestar"
        auto_detect = self._coerce_bool(self._config_value("auto_detect_seestar", True), True)
        if auto_detect and instrument_label:
            lowered = instrument_label.lower()
            if any(token in lowered for token in ("seestar", "s50", "s30")):
                return "seestar"
        return "classic"

    def _build_metadata_overrides(self, existing: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        instrument_label = self._normalize_string(existing.get("workflow_instrument"))
        if not instrument_label:
            instrument_label = self._detect_primary_instrument_label()
        instrument_for_mode = instrument_label
        metadata["workflow_instrument"] = instrument_label or "Unknown"

        mode_value = self._normalize_string(existing.get("mode"))
        if mode_value:
            normalized_mode = "seestar" if mode_value.lower().startswith("seestar") else "classic"
        else:
            normalized_mode = self._determine_workflow_mode(instrument_for_mode)
        metadata["mode"] = normalized_mode

        coadd_method = self._normalize_string(existing.get("global_coadd_method"))
        if not coadd_method:
            method_candidate = self._config_value("global_coadd_method", "kappa_sigma")
            coadd_method = self._normalize_string(method_candidate) or "kappa_sigma"
        metadata["global_coadd_method"] = coadd_method

        coadd_k_candidate = existing.get("global_coadd_k")
        if coadd_k_candidate is None:
            coadd_k_candidate = self._config_value("global_coadd_k", 2.0)
        metadata["global_coadd_k"] = self._coerce_float(coadd_k_candidate, 2.0)

        sds_value = existing.get("sds_mode")
        if sds_value is None:
            sds_value = self._config_value("sds_mode_default", False)
        sds_flag = self._coerce_bool(sds_value, False)
        metadata["sds_mode"] = sds_flag

        plan_override_existing = existing.get("global_wcs_plan_override")
        plan_override_payload: dict[str, Any] | None = None
        if isinstance(plan_override_existing, dict):
            plan_override_payload = dict(plan_override_existing)
        if sds_flag:
            if plan_override_payload is None:
                plan_override_payload = {}
            plan_override_payload["sds_mode"] = True
            plan_override_payload.setdefault("enabled", True)
        elif plan_override_payload is not None:
            plan_override_payload.pop("sds_mode", None)
            if not plan_override_payload:
                plan_override_payload = None
        if plan_override_payload is not None:
            metadata["global_wcs_plan_override"] = plan_override_payload
        elif "global_wcs_plan_override" in existing:
            metadata["global_wcs_plan_override"] = None

        return metadata

    @staticmethod
    def _build_global_wcs_meta(descriptor: dict[str, Any], fits_path: str, json_path: str) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        descriptor_meta = descriptor.get("metadata")
        if isinstance(descriptor_meta, dict):
            payload.update(descriptor_meta)

        keys = [
            "width",
            "height",
            "pixel_scale_as_per_px",
            "pixel_scale_deg_per_px",
            "padding_percent",
            "orientation",
            "orientation_matrix",
            "ra_wrap_used",
            "ra_wrap_offset_deg",
            "ra_span_deg",
            "dec_span_deg",
            "center_ra_deg",
            "center_dec_deg",
            "files",
            "nb_images",
            "resolution_override",
            "pixel_scale_mode",
            "timestamp",
            "source",
        ]
        for key in keys:
            if key in descriptor and key not in payload:
                payload[key] = descriptor[key]

        payload["fits_path"] = fits_path
        payload["json_path"] = json_path
        if "nb_images" not in payload or payload["nb_images"] is None:
            payload["nb_images"] = descriptor.get("nb_images")
        payload["files"] = payload.get("files") or descriptor.get("files") or []
        if "source" not in payload:
            payload["source"] = descriptor.get("source", "computed")
        return payload

    def _ensure_global_wcs_for_selection(
        self,
        require_plan: bool,
        selected_indices: list[int] | None = None,
    ) -> tuple[bool, dict[str, Any] | None, dict[str, Any] | None]:
        if not require_plan:
            return False, None, None

        try:
            from astropy.io import fits as _fits  # type: ignore
        except Exception:
            return False, None, None
        if WCS is None:
            return False, None, None

        output_dir_path = _expand_to_path(self._config_value("output_dir", ""))
        if not output_dir_path:
            return False, None, None

        if resolve_global_wcs_output_paths is None or parse_global_wcs_resolution_override is None:
            return False, None, None

        wcs_output_cfg = self._config_value("global_wcs_output_path", "global_mosaic_wcs.fits")
        pixelscale_mode = str(self._config_value("global_wcs_pixelscale_mode", "median") or "median")
        orientation_mode = str(self._config_value("global_wcs_orientation", "north_up") or "north_up")
        padding_raw = self._config_value("global_wcs_padding_percent", 2.0)
        try:
            padding_percent = float(padding_raw if padding_raw is not None else 2.0)
        except Exception:
            padding_percent = 2.0
        res_override_raw = self._config_value("global_wcs_res_override", None)
        res_override = parse_global_wcs_resolution_override(res_override_raw)

        try:
            fits_path, json_path = resolve_global_wcs_output_paths(str(output_dir_path), wcs_output_cfg)
        except Exception as exc:
            try:
                logger.warning("Global WCS (Qt): unable to resolve output path: %s", exc)
            except Exception:
                pass
            return False, None, None

        descriptor: dict[str, Any] | None = None
        if load_global_wcs_descriptor is not None:
            try:
                descriptor = load_global_wcs_descriptor(fits_path, json_path, logger_override=logger)
            except Exception:
                descriptor = None

        if descriptor is None:
            if compute_global_wcs_descriptor is None:
                return False, None, None

            selected_entries: list[_NormalizedItem] = []
            if selected_indices is None:
                for row, entry in enumerate(self._normalized_items):
                    if self._entry_is_checked(row):
                        selected_entries.append(entry)
            else:
                for idx in selected_indices:
                    if 0 <= idx < len(self._normalized_items):
                        selected_entries.append(self._normalized_items[idx])

            if not selected_entries:
                return False, None, None

            seestar_items: list[dict[str, Any]] = []
            fallback_items: list[dict[str, Any]] = []

            for entry in selected_entries:
                path = entry.file_path
                if not path or not _path_is_file(path):
                    continue
                try:
                    header = _fits.getheader(str(_expand_to_path(path) or path), ignore_missing_end=True)
                except Exception:
                    header = None
                if header is None:
                    continue
                try:
                    hdr = _ensure_sip_suffix_inplace(header)
                except Exception:
                    hdr = header
                try:
                    wcs_obj = WCS(hdr, naxis=2, relax=True)
                except Exception:
                    wcs_obj = None
                if wcs_obj is None or not getattr(wcs_obj, "is_celestial", False):
                    continue

                shape_hw = None
                try:
                    nax1 = header.get("NAXIS1")
                    nax2 = header.get("NAXIS2")
                    if isinstance(nax1, (int, float)) and isinstance(nax2, (int, float)):
                        shape_hw = (int(nax2), int(nax1))
                except Exception:
                    shape_hw = None

                item_payload: dict[str, Any] = {"path": path, "wcs": wcs_obj}
                if shape_hw is not None:
                    item_payload["shape"] = shape_hw

                is_seestar = False
                label = entry.instrument
                if not label:
                    label = _detect_instrument_from_header(header)
                try:
                    if label and any(token in str(label).lower() for token in ("seestar", "s50", "s30")):
                        is_seestar = True
                except Exception:
                    is_seestar = False

                if is_seestar:
                    seestar_items.append(item_payload)
                else:
                    fallback_items.append(item_payload)

            items_for_descriptor = seestar_items or (seestar_items + fallback_items)
            if not items_for_descriptor:
                try:
                    logger.warning("Global WCS (Qt): no usable entries found for descriptor computation.")
                except Exception:
                    pass
                return False, None, None

            try:
                descriptor = compute_global_wcs_descriptor(
                    items_for_descriptor,
                    pixel_scale_mode=pixelscale_mode,
                    orientation_mode=orientation_mode,
                    padding_percent=padding_percent,
                    resolution_override=res_override,
                    logger_override=logger,
                )
            except Exception as exc:
                try:
                    logger.error("Global WCS (Qt): computation failed: %s", exc, exc_info=True)
                except Exception:
                    pass
                descriptor = None

            if descriptor is None:
                return False, None, None

            if write_global_wcs_files is not None:
                try:
                    write_global_wcs_files(descriptor, fits_path, json_path, logger_override=logger)
                except Exception as exc:
                    try:
                        logger.error("Global WCS (Qt): failed to write descriptor: %s", exc, exc_info=True)
                    except Exception:
                        pass

        if not isinstance(descriptor, dict):
            return False, None, None

        try:
            self._global_wcs_state["descriptor"] = descriptor
            self._global_wcs_state["fits_path"] = fits_path
            self._global_wcs_state["json_path"] = json_path
            meta_state = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), dict) else None
            if meta_state is not None:
                self._global_wcs_state["meta"] = meta_state
        except Exception:
            pass

        meta_payload = self._build_global_wcs_meta(descriptor, fits_path, json_path)
        return True, meta_payload, {"fits_path": fits_path, "json_path": json_path}

    def overrides(self) -> Any:
        if self._cluster_refresh_pending:
            self._update_cluster_assignments()

        overrides: dict[str, Any] = {}
        if isinstance(self._initial_overrides, dict):
            try:
                overrides.update(self._initial_overrides)
            except Exception:
                overrides = dict(self._initial_overrides)

        if self._auto_group_checkbox is not None:
            overrides["filter_auto_group"] = bool(self._auto_group_checkbox.isChecked())
        if self._seestar_checkbox is not None:
            overrides["filter_seestar_priority"] = bool(self._seestar_checkbox.isChecked())
        if self._astap_instances_value:
            overrides["astap_max_instances"] = int(self._astap_instances_value)
        if self._auto_group_override_groups:
            overrides["preplan_master_groups"] = self._auto_group_override_groups
            overrides.pop("autosplit_cap", None)
        elif self._cluster_groups:
            serialized_groups: list[list[dict[str, Any]]] = []
            for group in self._cluster_groups:
                entries_payload: list[dict[str, Any]] = []
                for entry in group:
                    if not entry.file_path:
                        continue
                    payload = {
                        "path": entry.file_path,
                        "has_wcs": bool(entry.has_wcs),
                    }
                    if entry.instrument:
                        payload["instrument"] = entry.instrument
                    if entry.center_ra_deg is not None and entry.center_dec_deg is not None:
                        payload["RA"] = entry.center_ra_deg
                        payload["DEC"] = entry.center_dec_deg
                    entries_payload.append(payload)
                if entries_payload:
                    serialized_groups.append(entries_payload)
            if serialized_groups:
                overrides["preplan_master_groups"] = serialized_groups
        else:
            overrides.pop("preplan_master_groups", None)
        if isinstance(self._cluster_threshold_used, (int, float)) and self._cluster_threshold_used > 0:
            overrides["cluster_panel_threshold"] = float(self._cluster_threshold_used)

        overrides["max_raw_per_master_tile"] = int(self._max_raw_per_tile_value)
        resolved_count = self._resolved_wcs_count()
        if resolved_count:
            overrides["resolved_wcs_count"] = int(resolved_count)

        if self._sds_checkbox is not None:
            overrides["sds_mode"] = bool(self._sds_checkbox.isChecked())
        overrides["filter_overcap_allowance_pct"] = int(self._resolve_overcap_percent())
        overrides["batch_overlap_pct"] = int(self._resolve_overlap_percent())
        overrides["allow_batch_duplication"] = bool(self._config_value("allow_batch_duplication", True))
        overrides["min_safe_stack"] = int(self._config_value("min_safe_stack", 3))
        overrides["target_stack"] = int(self._config_value("target_stack", 5))
        overrides["filter_enable_coverage_first"] = bool(self._coverage_first_enabled_flag)
        if hasattr(self, "_auto_angle_enabled") and not getattr(self, "_auto_angle_enabled", True):
            overrides["cluster_orientation_split_deg"] = float(self._angle_split_value)

        excluded_indices = [idx for idx, checked in enumerate(self._entry_check_state) if not checked]
        if excluded_indices:
            overrides["filter_excluded_indices"] = excluded_indices

        metadata_update = self._build_metadata_overrides(overrides)

        eqmode_summary: dict[str, Any] | None = None
        if isinstance(self._last_auto_group_result, dict):
            eqmode_summary = self._last_auto_group_result.get("eqmode_summary")
        if not isinstance(eqmode_summary, dict):
            eqmode_summary = self._last_eqmode_summary
        if not isinstance(eqmode_summary, dict):
            eqmode_summary = None
        if eqmode_summary:
            self._last_eqmode_summary = eqmode_summary
            metadata_update["eqmode_summary"] = eqmode_summary
            plan_override_payload = metadata_update.get("global_wcs_plan_override")
            if not isinstance(plan_override_payload, dict):
                plan_override_payload = {}
            plan_override_payload["eqmode_summary"] = eqmode_summary
            metadata_update["global_wcs_plan_override"] = plan_override_payload

        sds_flag = bool(metadata_update.get("sds_mode"))
        mode_value = str(metadata_update.get("mode") or "").strip().lower()
        require_global_plan = sds_flag or mode_value == "seestar"

        success = False
        meta_payload: dict[str, Any] | None = None
        path_payload: dict[str, Any] | None = None
        if require_global_plan:
            success, meta_payload, path_payload = self._ensure_global_wcs_for_selection(True, None)

        if success and meta_payload and path_payload:
            if eqmode_summary:
                meta_payload = dict(meta_payload)
                meta_payload["eqmode_summary"] = eqmode_summary
            overrides["global_wcs_meta"] = meta_payload
            overrides["global_wcs_path"] = meta_payload.get("fits_path") or path_payload.get("fits_path")
            overrides["global_wcs_json"] = meta_payload.get("json_path") or path_payload.get("json_path")
            metadata_update["mode"] = "seestar"
            if sds_flag:
                plan_override = metadata_update.get("global_wcs_plan_override")
                if not isinstance(plan_override, dict):
                    plan_override = {}
                plan_override["sds_mode"] = True
                plan_override.setdefault("enabled", True)
                metadata_update["global_wcs_plan_override"] = plan_override
        elif require_global_plan:
            metadata_update["mode"] = "classic"
            plan_override = metadata_update.get("global_wcs_plan_override")
            if isinstance(plan_override, dict):
                plan_override.pop("sds_mode", None)
                if not plan_override:
                    metadata_update["global_wcs_plan_override"] = None
                else:
                    metadata_update["global_wcs_plan_override"] = plan_override

        for key, value in metadata_update.items():
            if key == "global_wcs_plan_override" and value is None:
                overrides.pop(key, None)
                continue
            overrides[key] = value

        result_overrides = overrides or None
        _force_phase45_disabled(result_overrides)
        return result_overrides

    def input_items(self) -> List[Any]:
        """Return a shallow copy of the original payload list."""

        if self._stream_scan and not self._streaming_completed:
            serialized_entries: list[dict[str, Any]] = []
            for idx, entry in enumerate(
                _iter_normalized_entries(
                    self._input_payload,
                    self._initial_overrides,
                    scan_recursive=self._scan_recursive,
                )
            ):
                serialized_entries.append(
                    self._serialize_entry_for_worker(entry, idx, include_header=False)
                )
            return serialized_entries

        return [
            self._serialize_entry_for_worker(
                entry,
                row,
                include_header=self._should_include_header_for_entry(entry),
            )
            for row, entry in enumerate(self._normalized_items)
        ]

    # ------------------------------------------------------------------
    # Scan worker slots
    # ------------------------------------------------------------------
    def _on_scan_progress(self, percent: int, message: str) -> None:
        try:
            self._progress_bar.setValue(percent)
        except Exception:
            pass
        if message:
            try:
                self._status_label.setText(message)
            except Exception:
                pass
            self._append_log(message)

    def _on_scan_row_update(self, row: int, payload: dict) -> None:
        if not (0 <= row < len(self._normalized_items)):
            return
        entry = self._normalized_items[row]
        has_wcs = payload.get("has_wcs")
        if isinstance(has_wcs, bool):
            entry.has_wcs = has_wcs
        instrument_changed = False
        instrument = payload.get("instrument")
        if instrument:
            if entry.instrument != instrument:
                entry.instrument = instrument
                instrument_changed = True
        ra_deg = payload.get("center_ra_deg")
        dec_deg = payload.get("center_dec_deg")
        if isinstance(ra_deg, (int, float)) and isinstance(dec_deg, (int, float)):
            entry.center_ra_deg = float(ra_deg)
            entry.center_dec_deg = float(dec_deg)
        self._refresh_entry_row(row)
        self._update_summary_label()
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()
        if instrument_changed:
            self._refresh_instrument_options()

    def _on_scan_error(self, message: str) -> None:
        if not message:
            return
        try:
            self._status_label.setText(message)
        except Exception:
            pass
        self._append_log(message, level="ERROR")

    def _on_scan_finished(self) -> None:
        self._hide_processing_overlay()
        if self._run_analysis_btn is not None:
            self._run_analysis_btn.setEnabled(True)
        self._scan_thread = None
        self._scan_worker = None
        completed_text = self._localizer.get("filter.scan.completed", "Analysis completed.")
        self._status_label.setText(completed_text)
        self._append_log(completed_text)
        self._progress_bar.setValue(100)
        self._update_summary_label()
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()
        self._refresh_instrument_options()

    def _resolve_initial_astap_instances(self) -> int:
        sources: tuple[Any, ...] = (
            self._safe_lookup(self._config_overrides, "astap_max_instances"),
            self._safe_lookup(self._solver_settings, "astap_max_instances"),
            os.environ.get("ZEMOSAIC_ASTAP_MAX_PROCS"),
        )
        for candidate in sources:
            try:
                if candidate in (None, "", False):
                    continue
                value = int(candidate)
                if value > 0:
                    return max(1, value)
            except Exception:
                continue
        return 1

    @staticmethod
    def _safe_lookup(mapping: Any, key: str) -> Any:
        if isinstance(mapping, dict):
            return mapping.get(key)
        try:
            return getattr(mapping, key)
        except Exception:
            return None

    def _persist_qt_filter_config(self) -> None:
        if _save_gui_config is None or _load_gui_config is None:
            return
        try:
            existing = _load_gui_config()
            if not isinstance(existing, dict):
                existing = dict(DEFAULT_FILTER_CONFIG)
        except Exception:
            existing = dict(DEFAULT_FILTER_CONFIG)
        cfg = dict(existing)
        if self._cluster_threshold_value > 0.0:
            cfg["cluster_panel_threshold"] = float(self._cluster_threshold_value)
        else:
            cfg.pop("cluster_panel_threshold", None)
        cfg["max_raw_per_master_tile"] = int(self._max_raw_per_tile_value)
        try:
            _save_gui_config(cfg)
        except Exception:
            pass

    def _compute_astap_cap(self) -> int:
        if compute_astap_recommended_max_instances is not None:
            try:
                return int(compute_astap_recommended_max_instances())
            except Exception:
                pass
        cpu_count = os.cpu_count() or 2
        return max(1, cpu_count // 2)

    def _populate_astap_instances_combo(self) -> None:
        if self._astap_instances_combo is None:
            return
        combo = self._astap_instances_combo
        combo.blockSignals(True)
        combo.clear()
        try:
            cap = self._compute_astap_cap()
        except Exception:
            cpu_count = os.cpu_count() or 2
            cap = max(1, cpu_count // 2)
        options = {str(i): i for i in range(1, cap + 1)}
        current = self._astap_instances_value
        options[str(current)] = current
        for text, value in sorted(options.items(), key=lambda item: int(item[0])):
            combo.addItem(text, value)
        index = combo.findData(current)
        if index < 0:
            index = combo.findText(str(current))
        if index < 0:
            combo.addItem(str(current), current)
            index = combo.findData(current)
        combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)
        combo.currentIndexChanged.connect(self._on_astap_instances_changed)  # type: ignore[arg-type]
        self._apply_astap_instances_choice(current, warn=False)

    def _on_astap_instances_changed(self, index: int) -> None:
        if self._astap_instances_combo is None:
            return
        value = self._astap_instances_combo.itemData(index)
        if value is None:
            try:
                value = int(self._astap_instances_combo.currentText())
            except Exception:
                value = self._astap_instances_value
        if not self._apply_astap_instances_choice(int(value), warn=True):
            self._astap_instances_combo.blockSignals(True)
            target_idx = self._astap_instances_combo.findData(self._astap_instances_value)
            if target_idx >= 0:
                self._astap_instances_combo.setCurrentIndex(target_idx)
            self._astap_instances_combo.blockSignals(False)

    def _apply_astap_instances_choice(self, value: int, *, warn: bool) -> bool:
        parsed = max(1, min(self._compute_astap_cap(), int(value)))
        if warn and parsed > 1 and parsed != self._astap_instances_value:
            title = self._localizer.get("filter_astap_multi_warning_title", "ASTAP Concurrency Warning")
            message = self._localizer.get(
                "filter_astap_multi_warning_message",
                (
                    "Running more than one ASTAP instance can trigger the \"Access violation\" "
                    "popup you saw earlier. Only continue if you are ready to dismiss "
                    "those warnings and understand this mode is not officially supported."
                ),
            )
            continue_label = self._localizer.get(
                "filter_astap_multi_warning_continue", "Continue anyway"
            )
            cancel_label = self._localizer.get(
                "filter_astap_multi_warning_cancel", "Cancel"
            )
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle(title)
            box.setText(message)
            continue_button = box.addButton(continue_label, QMessageBox.AcceptRole)
            cancel_button = box.addButton(cancel_label, QMessageBox.RejectRole)
            box.setDefaultButton(cancel_button)
            box.exec()
            if box.clickedButton() is not continue_button:
                return False

        self._astap_instances_value = parsed
        if isinstance(self._solver_settings, dict):
            self._solver_settings["astap_max_instances"] = parsed
        if isinstance(self._config_overrides, dict):
            self._config_overrides["astap_max_instances"] = parsed
        self._set_astap_concurrency_runtime(parsed)
        return True

    def _set_astap_concurrency_runtime(self, value: int) -> None:
        try:
            clamped = max(1, int(value))
        except Exception:
            clamped = 1
        try:
            clamped = min(clamped, self._compute_astap_cap())
        except Exception:
            pass
        os.environ["ZEMOSAIC_ASTAP_MAX_PROCS"] = str(clamped)
        if set_astap_max_concurrent_instances is not None:
            try:
                set_astap_max_concurrent_instances(clamped)
            except Exception:
                pass


def launch_filter_interface_qt(
    raw_files_with_wcs_or_dir,
    initial_overrides=None,
    *,
    stream_scan=False,
    scan_recursive=True,
    batch_size=100,
    preview_cap=200,
    solver_settings_dict=None,
    config_overrides=None,
    **kwargs,
):
    """Launch the Qt-based filter dialog and return the user selection.

    Returns a tuple ``(filtered_list, accepted, overrides)`` mirroring the
    semantics of the Tk filter dialog. When the user validates the selection,
    ``filtered_list`` contains the chosen entries and ``overrides`` includes
    any optional tweaks gathered in the dialog. Cancelling the dialog returns
    the original input list, ``accepted`` set to ``False`` and ``overrides``
    set to ``None``.
    """

    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(["zemosaic-filter-qt"])
        owns_app = True

    dialog = FilterQtDialog(
        raw_files_with_wcs_or_dir,
        initial_overrides,
        stream_scan=stream_scan,
        scan_recursive=scan_recursive,
        batch_size=batch_size,
        preview_cap=preview_cap,
        solver_settings_dict=solver_settings_dict,
        config_overrides=config_overrides,
    )
    dialog.exec()

    try:
        selected = dialog.selected_items()
        overrides = dialog.overrides()
        all_items = dialog.input_items()
        accepted = dialog.was_accepted()
    finally:
        try:
            dialog.deleteLater()
        except Exception:
            pass
        if owns_app:
            app.quit()

    if accepted:
        return selected, True, overrides

    return all_items, False, None


__all__ = ["FilterQtDialog", "launch_filter_interface_qt"]
