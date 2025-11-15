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
import logging
import importlib.util
import os
from pathlib import Path
import math
import csv
import threading
from typing import Any, Callable, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

import numpy as np


_pyside_spec = importlib.util.find_spec("PySide6")
if _pyside_spec is None:  # pragma: no cover - import guard
    raise ImportError(
        "PySide6 is required to use the ZeMosaic Qt filter interface. "
        "Install PySide6 or use the Tk interface instead."
    )

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot, QTimer, QRect
from PySide6.QtGui import QIcon
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
    QTableWidget,
    QTableWidgetItem,
    QWidget,
    QVBoxLayout,
    QPlainTextEdit,
    QCheckBox,
    QFileDialog,
    QTabWidget,
)


def _load_zemosaic_qicon() -> QIcon | None:
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        icon_dir = os.path.join(base_path, "icon")
    except Exception:
        return None

    candidates = [
        os.path.join(icon_dir, "zemosaic.ico"),
        os.path.join(icon_dir, "zemosaic_64x64.png"),
        os.path.join(icon_dir, "zemosaic_icon.png"),
    ]

    for path in candidates:
        try:
            if os.path.exists(path):
                return QIcon(path)
        except Exception:
            continue
    return None


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
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
except Exception:  # pragma: no cover - matplotlib optional
    FigureCanvasQTAgg = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]
    Rectangle = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    from zemosaic_astrometry import solve_with_astap, set_astap_max_concurrent_instances
except Exception:  # pragma: no cover - optional dependency guard
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

try:  # pragma: no cover - optional dependency guard
    from zemosaic_config import DEFAULT_CONFIG as _DEFAULT_GUI_CONFIG  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    _DEFAULT_GUI_CONFIG = {}

try:  # pragma: no cover - optional dependency guard
    from zemosaic_utils import (  # type: ignore
        EXCLUDED_DIRS,
        compute_global_wcs_descriptor,
        is_path_excluded,
        load_global_wcs_descriptor,
        parse_global_wcs_resolution_override,
        resolve_global_wcs_output_paths,
        write_global_wcs_files,
    )
except Exception:  # pragma: no cover - optional dependency guard
    EXCLUDED_DIRS = frozenset({"unaligned_by_zemosaic"})  # type: ignore[assignment]
    compute_global_wcs_descriptor = None  # type: ignore[assignment]
    is_path_excluded = None  # type: ignore[assignment]
    load_global_wcs_descriptor = None  # type: ignore[assignment]
    parse_global_wcs_resolution_override = None  # type: ignore[assignment]
    resolve_global_wcs_output_paths = None  # type: ignore[assignment]
    write_global_wcs_files = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

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
DEFAULT_FILTER_CONFIG.setdefault("global_coadd_method", "kappa_sigma")
DEFAULT_FILTER_CONFIG.setdefault("global_coadd_k", 2.0)


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
            excluded_paths = [os.fspath(p) for p in candidate]

    def _should_exclude(path: Path | str) -> bool:
        try:
            path_obj = Path(path)
        except Exception:
            path_obj = Path(str(path))

        # Honour explicit per-path exclusions from overrides first.
        try:
            norm = os.path.normcase(os.fspath(path_obj))
        except Exception:
            norm = os.path.normcase(str(path_obj))
        for entry in excluded_paths:
            try:
                if norm == os.path.normcase(entry):
                    return True
            except Exception:
                continue

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
        file_path = str(
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
        display_name = file_path or instrument or "Item"
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
        )

    def _build_from_path(path_obj: Path) -> _NormalizedItem:
        file_path = str(path_obj)
        display_name = path_obj.name or file_path
        include = not _should_exclude(path_obj)
        return _NormalizedItem(
            original=file_path,
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
                    if _should_exclude(candidate):
                        continue
                    yield _build_from_path(candidate)
        elif Path(payload).is_file():
            candidate = Path(payload)
            if not _should_exclude(candidate):
                yield _build_from_path(candidate)
    elif isinstance(payload, Iterable):
        for element in payload:
            if isinstance(element, dict):
                yield _build_from_mapping(element)
            elif isinstance(element, (str, os.PathLike)):
                yield _build_from_path(Path(element))
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
    if not isinstance(file_path, str) or not file_path:
        return
    if not os.path.isfile(file_path):
        return
    try:
        with fits.open(file_path, mode="update", memmap=False) as hdul:  # type: ignore[call-arg]
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
    display_name = os.path.basename(path) or path
    try:
        _write_header_to_fits_local(path, header_obj)
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

        for index, item in enumerate(self._items):
            if self._stop_requested:
                break
            path = item.file_path
            message = self._localizer.get(
                "filter.scan.inspecting",
                "Inspecting {name}…",
                name=os.path.basename(path) if path else item.display_name,
            )
            self.progress_changed.emit(self._progress_percent(index, total), message)
            row_update: dict[str, Any] = {}
            if not path or not os.path.isfile(path):
                row_update["error"] = self._localizer.get(
                    "filter.scan.missing",
                    "File missing or not accessible.",
                )
                self.row_updated.emit(index, row_update)
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

            has_wcs = _header_has_wcs(header) if header is not None else False
            instrument = _detect_instrument_from_header(header)

            if astap_cfg is not None and not has_wcs and solve_with_astap is not None and header is not None:
                solve_message = self._localizer.get(
                    "filter.scan.solving",
                    "Solving WCS with ASTAP…",
                )
                self.progress_changed.emit(self._progress_percent(index, total), solve_message)

                def _solver_callback(text: str, *_: Any) -> None:
                    try:
                        message_text = str(text)
                    except Exception:
                        message_text = ""
                    if message_text:
                        self.progress_changed.emit(
                            self._progress_percent(index, total),
                            message_text,
                        )

                try:
                    wcs_result = solve_with_astap(
                        path,
                        header,
                        astap_cfg["exe"],
                        astap_cfg["data"],
                        search_radius_deg=astap_cfg["radius"],
                        downsample_factor=astap_cfg["downsample"],
                        sensitivity=astap_cfg["sensitivity"],
                        timeout_sec=astap_cfg["timeout"],
                        update_original_header_in_place=self._write_wcs_to_file,
                        progress_callback=_solver_callback,
                    )
                except Exception as exc:  # pragma: no cover - solver failure path
                    row_update["error"] = str(exc)
                else:
                    if wcs_result is not None and getattr(wcs_result, "is_celestial", False):
                        has_wcs = True
                        row_update["solver"] = "ASTAP"
                        if self._write_wcs_to_file and header is not None:
                            _persist_wcs_header_if_requested(path, header, True)

            row_update["has_wcs"] = has_wcs
            if instrument:
                row_update["instrument"] = instrument
            if header is not None:
                ra_deg, dec_deg = _extract_center_from_header(header)
                if ra_deg is not None and dec_deg is not None:
                    row_update["center_ra_deg"] = ra_deg
                    row_update["center_dec_deg"] = dec_deg
            self.row_updated.emit(index, row_update)
            completed_message = self._localizer.get(
                "filter.scan.progress",
                "Processed {done}/{total} files.",
                done=index + 1,
                total=total,
            )
            self.progress_changed.emit(self._progress_percent(index + 1, total), completed_message)

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
        exe_path = next((str(p) for p in exe_candidates if p), "")
        data_dir = next((str(p) for p in data_candidates if p), "")
        if not exe_path or not os.path.isfile(exe_path):
            return None

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

        if set_astap_max_concurrent_instances is not None:
            concurrency = self._coerce_int(
                self._solver_settings.get("astap_max_instances"),
                self._overrides.get("astap_max_instances"),
                default=None,
            )
            if concurrency:
                try:
                    set_astap_max_concurrent_instances(concurrency)
                except Exception:  # pragma: no cover - runtime guard
                    pass

        return {
            "exe": exe_path,
            "data": data_dir,
            "radius": radius,
            "downsample": downsample,
            "sensitivity": sensitivity,
            "timeout": timeout,
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
        _force_phase45_disabled(self._initial_overrides)
        self._stream_scan = stream_scan
        self._scan_recursive = scan_recursive
        self._batch_size = batch_size
        self._preview_cap = preview_cap
        self._solver_settings = solver_settings_dict
        self._config_overrides = config_overrides or {}
        _force_phase45_disabled(self._config_overrides)
        self._accepted = False

        self._localizer = self._load_localizer()
        self._normalized_items: list[_NormalizedItem] = []
        self._stream_thread: QThread | None = None
        self._stream_worker: _StreamIngestWorker | None = None
        self._streaming_active = False
        self._streaming_completed = not self._stream_scan
        self._stream_loaded_count = 0
        try:
            self._batch_size = max(1, int(batch_size))
        except Exception:
            self._batch_size = 100
        self._dialog_button_box: QDialogButtonBox | None = None

        if self._stream_scan:
            self._normalized_items = []
        else:
            self._normalized_items = self._normalize_items(raw_files_with_wcs_or_dir, initial_overrides)

        overrides = config_overrides or {}
        self._auto_group_requested = bool(overrides.get("filter_auto_group", False))
        self._seestar_priority = bool(overrides.get("filter_seestar_priority", False))

        self._table = QTableWidget(self)
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
        self._cluster_groups: list[list[_NormalizedItem]] = []
        self._cluster_threshold_used: float | None = None
        self._cluster_refresh_pending = False
        self._auto_group_button: QPushButton | None = None
        self._auto_group_summary_label: QLabel | None = None
        self._auto_group_running = False
        self._auto_group_override_groups: list[list[dict[str, Any]]] | None = None
        self._header_cache: dict[str, Any] = {}
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
        base_angle = _sanitize_angle_value(self._config_value("cluster_orientation_split_deg", 0.0), 0.0)
        if base_angle <= 0.0:
            base_angle = ANGLE_SPLIT_DEFAULT_DEG
        self._auto_angle_enabled = True
        self._angle_split_value = float(base_angle)
        self._group_outline_bounds: list[tuple[float, float, float, float]] = []
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
        self._log_output: QPlainTextEdit | None = None
        self._scan_recursive_checkbox: QCheckBox | None = None
        self._draw_footprints_checkbox: QCheckBox | None = None
        self._write_wcs_checkbox: QCheckBox | None = None
        self._sds_mode_initial = self._coerce_bool(
            (initial_overrides or {}).get("sds_mode")
            if isinstance(initial_overrides, dict)
            else None,
            self._coerce_bool(
                (config_overrides or {}).get("sds_mode")
                if isinstance(config_overrides, dict)
                else None,
                self._coerce_bool(self._config_value("sds_mode_default", False), False),
            ),
        )
        self._cache_csv_path: str | None = None
        if self._stream_scan:
            candidate_dir = None
            if isinstance(raw_files_with_wcs_or_dir, (str, os.PathLike)):
                candidate_dir = Path(raw_files_with_wcs_or_dir)
            elif isinstance(raw_files_with_wcs_or_dir, Path):
                candidate_dir = raw_files_with_wcs_or_dir
            if candidate_dir is not None:
                try:
                    if candidate_dir.is_dir():
                        self._cache_csv_path = str(candidate_dir / "headers_cache.csv")
                except Exception:
                    self._cache_csv_path = None
        self._build_ui()
        if self._stream_scan:
            self._prepare_streaming_mode(raw_files_with_wcs_or_dir, initial_overrides)
        else:
            self._populate_table()
            self._update_summary_label()
            self._schedule_preview_refresh()
            self._schedule_cluster_refresh()
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
        return canvas

    def _create_coverage_canvas(self) -> FigureCanvasQTAgg | None:
        """Initialise the coverage-map Matplotlib canvas (global WCS plane)."""

        if Figure is None or FigureCanvasQTAgg is None:
            return None

        figure = Figure(figsize=(5, 3))
        canvas = FigureCanvasQTAgg(figure)
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
        layout.addWidget(self._auto_group_button, 0, 1)

        # Summary label mirroring Tk's "Prepared N group(s), sizes: …"
        self._auto_group_summary_label = QLabel("", box)
        self._auto_group_summary_label.setWordWrap(True)
        layout.addWidget(self._auto_group_summary_label, 0, 2, 7, 1)
        layout.setColumnStretch(2, 1)

        astap_label = QLabel(
            self._localizer.get("filter_label_astap_instances", "Max ASTAP instances"),
            box,
        )
        layout.addWidget(astap_label, 1, 0)
        self._astap_instances_combo = QComboBox(box)
        self._populate_astap_instances_combo()
        layout.addWidget(self._astap_instances_combo, 1, 1)

        self._draw_footprints_checkbox = QCheckBox(
            self._localizer.get("filter_chk_draw_footprints", "Draw WCS footprints"),
            box,
        )
        self._draw_footprints_checkbox.setChecked(True)
        self._draw_footprints_checkbox.toggled.connect(  # type: ignore[arg-type]
            lambda _checked: self._schedule_preview_refresh()
        )
        layout.addWidget(self._draw_footprints_checkbox, 2, 0)

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

        self._auto_angle_checkbox = QCheckBox(
            self._localizer.get("ui_auto_angle_split", "Auto split by orientation"),
            box,
        )
        self._auto_angle_checkbox.setChecked(True)
        self._auto_angle_checkbox.toggled.connect(self._on_auto_angle_toggled)  # type: ignore[arg-type]
        layout.addWidget(self._auto_angle_checkbox, 5, 0, 1, 2)

        angle_label = QLabel(
            self._localizer.get("ui_angle_split_threshold", "Orientation split (deg)"),
            box,
        )
        layout.addWidget(angle_label, 6, 0)
        self._angle_split_spin = QDoubleSpinBox(box)
        self._angle_split_spin.setRange(0.0, 180.0)
        self._angle_split_spin.setSingleStep(0.5)
        self._angle_split_spin.setDecimals(1)
        self._angle_split_spin.setValue(float(self._angle_split_value))
        self._angle_split_spin.valueChanged.connect(self._on_angle_split_changed)  # type: ignore[arg-type]
        layout.addWidget(self._angle_split_spin, 6, 1)

        self._sds_checkbox = QCheckBox(
            self._localizer.get("filter_chk_sds_mode", "Enable ZeSupaDupStack (SDS)"),
            box,
        )
        self._sds_checkbox.setChecked(bool(self._sds_mode_initial))
        try:
            self._sds_checkbox.toggled.connect(self._on_sds_toggled)  # type: ignore[arg-type]
        except Exception:
            pass
        layout.addWidget(self._sds_checkbox, 7, 0, 1, 2)

        return box

    def _on_coverage_first_toggled(self, checked: bool) -> None:
        self._coverage_first_enabled_flag = bool(checked)
        self._runtime_overrides["filter_enable_coverage_first"] = self._coverage_first_enabled_flag

    def _on_overcap_changed(self, value: int) -> None:
        clamped = self._clamp_overcap_percent(value)
        self._overcap_percent_value = clamped
        self._runtime_overrides["filter_overcap_allowance_pct"] = clamped

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

        self._table.blockSignals(True)
        excluded = 0
        for row, entry in enumerate(self._normalized_items):
            item = self._table.item(row, 0)
            if item is None:
                continue
            instrument_label = (entry.instrument or "").strip()
            keep = True
            if target is None:
                keep = True
            elif target == self._instrument_unknown_token:
                keep = instrument_label == ""
            elif isinstance(target, str):
                keep = instrument_label.lower() == target.lower()
            if not keep:
                if item.checkState() != Qt.Unchecked:
                    item.setCheckState(Qt.Unchecked)
                    excluded += 1
            else:
                if item.checkState() != Qt.Checked:
                    item.setCheckState(Qt.Checked)
        self._table.blockSignals(False)
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

    def _on_auto_group_clicked(self) -> None:
        """Trigger the worker-style clustering pipeline used by the Tk filter."""

        if self._auto_group_running:
            return
        if _CLUSTER_CONNECTED is None or _AUTOSPLIT_GROUPS is None:
            message = self._localizer.get(
                "filter.cluster.helpers_missing",
                "Auto-organisation helpers are unavailable; install the worker module.",
            )
            self._append_log(message)
            self._status_label.setText(message)
            return

        selected_indices = self._collect_selected_indices()
        if not selected_indices:
            message = self._localizer.get(
                "filter.cluster.no_selection",
                "Select at least one frame before requesting auto-organisation.",
            )
            self._append_log(message)
            self._status_label.setText(message)
            return

        self._auto_group_running = True
        if self._auto_group_button is not None:
            self._auto_group_button.setEnabled(False)
        self._append_log(
            self._localizer.get(
                "filter.cluster.manual_refresh",
                "Manual master-tile organisation requested.",
            )
        )
        self._status_label.setText(
            self._localizer.get("filter.cluster.running", "Preparing master-tile groups…")
        )

        try:
            thread = threading.Thread(
                target=self._auto_group_background_task,
                args=(selected_indices,),
                daemon=True,
            )
            thread.start()
        except Exception as exc:
            # If the background thread cannot be started, fall back to a safe state
            # and report the issue instead of leaving the button permanently disabled.
            self._auto_group_running = False
            if self._auto_group_button is not None:
                try:
                    self._auto_group_button.setEnabled(True)
                except Exception:
                    pass
            message = self._localizer.get(
                "filter.cluster.failed",
                "Auto-organisation failed: {error}",
            )
            try:
                message = message.format(error=exc)
            except Exception:
                message = f"Auto-organisation failed: {exc}"
            self._append_log(message)
            self._status_label.setText(message)

    def _auto_group_background_task(self, selected_indices: list[int]) -> None:
        messages: list[str] = []
        result_payload: dict[str, Any] | None = None
        try:
            result_payload = self._compute_auto_groups(selected_indices, messages)
        except Exception as exc:  # pragma: no cover - defensive guard
            error_text = self._localizer.get(
                "filter.cluster.failed",
                "Auto-organisation failed: {error}",
            )
            try:
                formatted = error_text.format(error=exc)
            except Exception:
                formatted = f"Auto-organisation failed: {exc}"
            messages.append(formatted)
        if result_payload and result_payload.get("coverage_first"):
            groups_count = len(result_payload.get("final_groups") or [])
            messages.append(
                self._format_message(
                    "log_covfirst_done",
                    "Coverage-first preplan ready: {N} group(s).",
                    N=groups_count,
                )
            )

        def _finalize() -> None:
            self._auto_group_running = False
            if self._auto_group_button is not None:
                self._auto_group_button.setEnabled(True)
            for line in messages:
                self._append_log(line)
            if result_payload is None:
                if not messages:
                    fallback = self._localizer.get(
                        "filter.cluster.failed_generic",
                        "Unable to prepare master-tile groups.",
                    )
                    self._append_log(fallback)
                self._status_label.setText(
                    self._localizer.get(
                        "filter.cluster.failed_short",
                        "Auto-organisation failed.",
                    )
                )
                return
            self._apply_auto_group_result(result_payload)

        QTimer.singleShot(0, _finalize)

    def _compute_auto_groups(
        self,
        selected_indices: list[int],
        messages: list[str],
    ) -> dict[str, Any]:
        sds_mode = bool(self._sds_checkbox.isChecked()) if self._sds_checkbox is not None else False
        coverage_enabled = self._coverage_first_enabled()
        workflow_mode = self._determine_workflow_mode(self._detect_primary_instrument_label())
        if workflow_mode == "seestar":
            coverage_enabled = True
        if sds_mode and compute_global_wcs_descriptor is not None:
            success, _meta = self._ensure_global_wcs_for_indices(selected_indices)
            if success:
                threshold = self._coerce_float(self._config_value("sds_coverage_threshold", 0.92), 0.92)
                threshold = max(0.10, min(0.99, threshold))
                sds_groups = self._build_sds_batches_for_indices(
                    selected_indices,
                    coverage_threshold=threshold,
                )
                if sds_groups:
                    template = self._localizer.get(
                        "filter.cluster.sds_ready",
                        "ZeSupaDupStack prepared {count} coverage batch(es).",
                    )
                    try:
                        messages.append(template.format(count=len(sds_groups)))
                    except Exception:
                        messages.append(f"ZeSupaDupStack prepared {len(sds_groups)} coverage batch(es).")
                    return {
                        "final_groups": sds_groups,
                        "sizes": [len(group) for group in sds_groups],
                        "coverage_first": True,
                        "threshold_used": 0.0,
                        "angle_split": 0.0,
                    }
                else:
                    messages.append(
                        self._localizer.get(
                            "filter.cluster.sds_no_batches",
                            "ZeSupaDupStack auto-group fallback: coverage batches could not be built.",
                        )
                    )
            else:
                messages.append(
                    self._localizer.get(
                        "filter.cluster.sds_wcs_unavailable",
                        "ZeSupaDupStack auto-group fallback: global WCS descriptor unavailable.",
                    )
                )
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

        if not candidate_infos:
            raise RuntimeError("No candidate entries were usable for grouping.")
        if WCS is None and SkyCoord is None:
            raise RuntimeError("Astropy is required to compute WCS-based clusters.")

        threshold_override = self._resolve_cluster_threshold_override()
        threshold_heuristic = self._estimate_threshold_from_coords(coord_samples)
        threshold_initial = threshold_override if threshold_override and threshold_override > 0 else threshold_heuristic
        if threshold_initial <= 0:
            threshold_initial = 0.1

        orientation_threshold = self._resolve_orientation_split_threshold()
        angle_split_candidate = self._resolve_angle_split_candidate()
        auto_angle_detect = self._resolve_auto_angle_detect_threshold()
        cap_effective, min_cap = self._resolve_autosplit_caps()
        overcap_pct = self._resolve_overcap_percent()
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

        groups_initial = _CLUSTER_CONNECTED(
            candidate_infos,
            float(threshold_initial),
            None,
            orientation_split_threshold_deg=float(max(0.0, orientation_threshold)),
        )
        if not groups_initial:
            raise RuntimeError("Worker clustering returned no groups.")

        threshold_used = float(threshold_initial)
        groups_used = groups_initial
        angle_split_effective = 0.0

        auto_angle_triggered = False
        if angle_split_candidate > 0.0 and _tk_split_group_by_orientation is not None and _tk_circular_dispersion_deg is not None:
            oriented_groups: list[list[dict[str, Any]]] = []
            triggered = 0
            for group in groups_used:
                pa_values: list[float] = []
                for info in group:
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
                    oriented_groups.extend(
                        _tk_split_group_by_orientation(group, angle_split_candidate)
                    )
                    triggered += 1
                else:
                    oriented_groups.append(group)
            if triggered > 0:
                auto_angle_triggered = True
                angle_split_effective = angle_split_candidate
                groups_used = oriented_groups

        if not auto_angle_triggered and angle_split_candidate > 0.0:
            angle_split_effective = angle_split_candidate

        groups_after_autosplit = _AUTOSPLIT_GROUPS(
            groups_used,
            cap=int(max(1, cap_effective)),
            min_cap=int(max(1, min_cap)),
            progress_callback=None,
        )
        if coverage_enabled:
            messages.append(
                self._format_message(
                    "log_covfirst_autosplit",
                    "Autosplit applied: cap={CAP}, min_cap={MIN}, groups_in={IN}, groups_out={OUT}",
                    CAP=int(cap_effective),
                    MIN=int(min_cap),
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
        final_groups = merge_fn(
            groups_after_autosplit,
            min_size=int(max(1, min_cap)),
            cap=int(max(1, cap_effective)),
            cap_allowance=cap_allowance,
            compute_dispersion=_COMPUTE_MAX_SEPARATION,
            max_dispersion_deg=max_dispersion,
            log_fn=messages.append,
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

        for group in final_groups:
            for info in group:
                if info.pop("_fallback_wcs_used", False):
                    info.pop("wcs", None)

        summary_template = self._localizer.get(
            "filter.cluster.summary",
            "Auto-grouped {groups} cluster(s) (threshold {threshold:.2f}°)",
        )
        try:
            summary_text = summary_template.format(
                groups=len(final_groups),
                threshold=threshold_used,
            )
        except Exception:
            summary_text = f"Auto-grouped {len(final_groups)} cluster(s)"
        messages.append(summary_text)

        return {
            "final_groups": final_groups,
            "sizes": [len(group) for group in final_groups],
            "coverage_first": True,
            "threshold_used": threshold_used,
            "angle_split": angle_split_effective,
        }

    def _apply_auto_group_result(self, payload: dict[str, Any]) -> None:
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
                assignment[os.path.normcase(str(path))] = (idx - 1, label)

        for normalized in self._normalized_items:
            key = os.path.normcase(normalized.file_path or "")
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
                key = os.path.normcase(normalized.file_path or "")
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

        # Summarise group sizes both in the log and next to the button,
        # mirroring Tk's "Prepared {g} group(s), sizes: …" behaviour.
        hist = _format_sizes_histogram(sizes) if sizes else "[]"
        summary_template = self._localizer.get(
            "filter_log_groups_summary",
            "Prepared {g} group(s), sizes: {sizes}.",
        )
        try:
            summary_text = summary_template.format(g=len(groups), sizes=hist)
        except Exception:
            summary_text = f"Prepared {len(groups)} group(s), sizes: {hist}."
        self._append_log(summary_text)
        if self._auto_group_summary_label is not None:
            self._auto_group_summary_label.setText(summary_text)

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
            header = self._load_header(path)
            if header is not None:
                payload["header"] = header
        wcs_obj = payload.get("wcs")
        if wcs_obj is None and header is not None and WCS is not None:
            try:
                wcs_obj = _build_wcs_from_header(header)
            except Exception:
                wcs_obj = None
            if wcs_obj is not None:
                payload["wcs"] = wcs_obj
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
        norm = os.path.normcase(path)
        if norm in self._header_cache:
            return self._header_cache[norm]
        try:
            header = fits.getheader(path, ignore_missing_end=True)
        except Exception:
            header = None
        self._header_cache[norm] = header
        return header

    def _compute_group_outline_bounds(self, groups: list[list[dict[str, Any]]]) -> list[tuple[float, float, float, float]]:
        if not groups:
            return []
        path_map: dict[str, _NormalizedItem] = {}
        for entry in self._normalized_items:
            if entry.file_path:
                path_map[os.path.normcase(entry.file_path)] = entry
        outlines: list[tuple[float, float, float, float]] = []
        for group in groups:
            ra_vals: list[float] = []
            dec_vals: list[float] = []
            for info in group or []:
                footprint = self._resolve_group_entry_footprint(info, path_map)
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
                    continue

                # Fallback: approximate from group entry centre when no footprint is available.
                center_ra: float | None = None
                center_dec: float | None = None
                entry_obj: _NormalizedItem | None = None
                if isinstance(info, dict):
                    path_val = None
                    for key in ("path", "path_raw", "path_preprocessed_cache"):
                        candidate = info.get(key)
                        if candidate:
                            path_val = candidate
                            break
                    if path_val:
                        entry_obj = path_map.get(os.path.normcase(str(path_val)))
                    if entry_obj is not None:
                        center_ra = entry_obj.center_ra_deg
                        center_dec = entry_obj.center_dec_deg
                        if center_ra is None or center_dec is None:
                            center_ra, center_dec = self._ensure_entry_coordinates(entry_obj)
                    if (center_ra is None or center_dec is None) and isinstance(info, dict):
                        try:
                            center_ra = float(info.get("RA")) if info.get("RA") is not None else None
                            center_dec = float(info.get("DEC")) if info.get("DEC") is not None else None
                        except Exception:
                            center_ra = None
                            center_dec = None
                if center_ra is not None and center_dec is not None:
                    if math.isfinite(center_ra) and math.isfinite(center_dec):
                        ra_vals.append(float(center_ra))
                        dec_vals.append(float(center_dec))
            if not ra_vals or not dec_vals:
                continue
            ra_min_raw, ra_max_raw = self._normalize_ra_span(ra_vals)
            dec_min_raw = min(dec_vals)
            dec_max_raw = max(dec_vals)

            # Ensure non-zero extent so rectangles remain visible even for tiny groups.
            if ra_min_raw == ra_max_raw:
                margin_ra = 0.05
                ra_min = ra_min_raw - margin_ra
                ra_max = ra_max_raw + margin_ra
            else:
                ra_min, ra_max = ra_min_raw, ra_max_raw

            if dec_min_raw == dec_max_raw:
                margin_dec = 0.05
                dec_min = dec_min_raw - margin_dec
                dec_max = dec_max_raw + margin_dec
            else:
                dec_min, dec_max = dec_min_raw, dec_max_raw

            outlines.append((ra_min, ra_max, dec_min, dec_max))
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
            self._coverage_canvas.draw_idle()
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
            self._coverage_canvas.draw_idle()
            return

        try:
            width = int(descriptor.get("width") or 0)
            height = int(descriptor.get("height") or 0)
        except Exception:
            width = height = 0
        if width <= 0 or height <= 0:
            self._coverage_canvas.draw_idle()
            return

        # Build a lookup from path to normalized entry, reusing the same helper
        # logic as the sky-preview group outlines.
        path_map: dict[str, _NormalizedItem] = {}
        for entry in self._normalized_items:
            if entry.file_path:
                path_map[os.path.normcase(entry.file_path)] = entry

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

        self._coverage_canvas.draw_idle()

    def _resolve_group_entry_footprint(
        self,
        info: Any,
        path_map: dict[str, _NormalizedItem],
    ) -> List[Tuple[float, float]] | None:
        entry: _NormalizedItem | None = None
        if isinstance(info, dict):
            for key in ("path", "path_raw", "path_preprocessed_cache"):
                value = info.get(key)
                if value:
                    entry = path_map.get(os.path.normcase(str(value)))
                    if entry is not None:
                        break
        if entry is not None:
            footprint = entry.footprint_radec or self._ensure_entry_footprint(entry)
            if footprint:
                return footprint
        if isinstance(info, dict):
            footprint_payload = _sanitize_footprint_radec(info.get("footprint_radec"))
            if footprint_payload:
                return footprint_payload
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
        try:
            cap = int(self._config_value("max_raw_per_master_tile", 50))
        except Exception:
            cap = 50
        cap = max(1, min(50, cap))
        try:
            min_cap = int(self._config_value("autosplit_min_cap", min(8, cap)))
        except Exception:
            min_cap = min(8, cap)
        min_cap = max(1, min(min_cap, cap))
        return cap, min_cap

    def _resolve_overcap_percent(self) -> int:
        value = getattr(self, "_overcap_percent_value", None)
        if value is not None:
            return int(value)
        return self._clamp_overcap_percent(self._config_value("filter_overcap_allowance_pct", 10))

    def _coverage_first_enabled(self) -> bool:
        return bool(self._coverage_first_enabled_flag)

    def _ensure_global_wcs_for_indices(
        self,
        selected_indices: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        success, meta_payload, _ = self._ensure_global_wcs_for_selection(True, selected_indices)
        return success, meta_payload

    def _build_sds_batches_for_indices(
        self,
        selected_indices: list[int],
        *,
        coverage_threshold: float = 0.92,
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
        total_cells = grid_h * grid_w
        remaining = entry_infos
        batches: list[list[dict[str, Any]]] = []
        while remaining:
            coverage_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
            coverage_cells = 0
            used_indices: list[int] = []
            batch_entries: list[dict[str, Any]] = []
            for idx, info in enumerate(remaining):
                gy0, gy1, gx0, gx1 = info.get("grid_bbox", (0, 0, 0, 0))
                if gy1 <= gy0 or gx1 <= gx0:
                    continue
                region = coverage_grid[gy0:gy1, gx0:gx1]
                already = int(region.sum())
                region_cells = (gy1 - gy0) * (gx1 - gx0)
                region[...] = 1
                gain = max(0, region_cells - already)
                if gain <= 0 and batch_entries:
                    continue
                coverage_cells += gain
                batch_entries.append(info)
                used_indices.append(idx)
                fraction = (coverage_cells / total_cells) if total_cells else 1.0
                if fraction >= coverage_threshold and batch_entries:
                    break
            if not batch_entries:
                batch_entries.append(remaining[0])
                used_indices = [0]
            batches.append(batch_entries)
            used = set(used_indices)
            remaining = [info for idx, info in enumerate(remaining) if idx not in used]

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
            item = self._table.item(row, 0)
            if item is None or item.checkState() != Qt.Checked:
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
            self._append_log(message)
            self._status_label.setText(message)
            return

        excluded = 0
        self._table.blockSignals(True)
        for row, entry in enumerate(self._normalized_items):
            item = self._table.item(row, 0)
            if item is None:
                continue
            ra_deg, dec_deg = entry.center_ra_deg, entry.center_dec_deg
            if ra_deg is None or dec_deg is None:
                ra_deg, dec_deg = self._ensure_entry_coordinates(entry)
            if ra_deg is None or dec_deg is None:
                continue
            distance = self._angular_distance_deg(ra_deg, dec_deg, center_ra, center_dec)
            if distance > threshold and item.checkState() != Qt.Unchecked:
                item.setCheckState(Qt.Unchecked)
                excluded += 1
        self._table.blockSignals(False)
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
            self._append_log(message)
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
            self._append_log(message)
            self._status_label.setText(message)

    def _prompt_csv_path(self) -> str | None:
        caption = self._localizer.get("filter.export.dialog_title", "Export filter CSV")
        csv_filter = self._localizer.get("filter.export.csv_filter", "CSV files (*.csv)")
        path, _filter = QFileDialog.getSaveFileName(
            self,
            caption,
            os.path.expanduser("~/zemosaic_filter.csv"),
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
            header_candidate = entry.original.get("header") or entry.original.get("fits_header")
            if header_candidate:
                return header_candidate
        if fits is None:
            return None
        path = entry.file_path
        if not path or not os.path.isfile(path):
            return None
        try:
            return fits.getheader(path, ignore_missing_end=True)
        except Exception:
            return None

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
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            return True
        except Exception as exc:
            self._append_log(f"CSV export failed: {exc}")
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

        self._table.setColumnCount(4)
        headers = [
            self._localizer.get("filter.column.file", "File"),
            self._localizer.get("filter.column.wcs", "WCS"),
            self._localizer.get("filter.column.group", "Group"),
            self._localizer.get("filter.column.instrument", "Instrument"),
        ]
        self._table.setHorizontalHeaderLabels(headers)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemChanged.connect(self._handle_item_changed)

        images_group = QGroupBox(
            self._localizer.get("filter_images_check_to_keep", "Images (check to keep)"),
            self,
        )
        images_layout = QVBoxLayout(images_group)
        images_layout.setContentsMargins(8, 8, 8, 8)
        images_layout.setSpacing(6)
        images_layout.addWidget(self._table, 1)

        controls_layout.addWidget(images_group, 2)

        wcs_box = self._create_wcs_controls_box()
        controls_layout.addWidget(wcs_box)

        options_box = self._create_options_box()
        controls_layout.addWidget(options_box)

        log_group_box = QGroupBox(
            self._localizer.get("filter.group.log", "Scan / grouping log"), self
        )
        log_layout = QVBoxLayout(log_group_box)
        self._log_output = QPlainTextEdit(log_group_box)
        self._log_output.setReadOnly(True)
        self._log_output.setMaximumBlockCount(500)
        placeholder = self._localizer.get(
            "filter.log.placeholder",
            "Scan, clustering, and WCS messages will appear here.",
        )
        self._log_output.setPlaceholderText(placeholder)
        log_layout.addWidget(self._log_output)
        controls_layout.addWidget(log_group_box, 1)

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

        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)

    def _prepare_streaming_mode(self, payload: Any, initial_overrides: Any) -> None:
        """Enable incremental ingestion when the dialog is opened in stream mode."""

        self._stream_loaded_count = 0
        self._streaming_active = True
        self._streaming_completed = False
        self._table.setRowCount(0)
        self._update_summary_label()
        loading_text = self._localizer.get("filter.stream.starting", "Discovering frames…")
        self._status_label.setText(loading_text)
        self._progress_bar.setRange(0, 0)
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
            self._status_label.setText(message_template.format(count=self._stream_loaded_count))
        except Exception:
            self._status_label.setText(f"Discovered {self._stream_loaded_count} frame(s)…")

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
            self._status_label.setText(message_template.format(count=total_loaded))
        except Exception:
            self._status_label.setText(f"Finished loading {total_loaded} frame(s).")
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

    def _on_stream_thread_finished(self) -> None:
        self._stream_thread = None
        self._stream_worker = None

    def _stop_stream_worker(self) -> None:
        worker = self._stream_worker
        thread = self._stream_thread
        was_running = False
        if worker is not None:
            worker.request_stop()
        if thread is not None:
            was_running = thread.isRunning()
            thread.quit()
            thread.wait(2000)
        self._stream_worker = None
        self._stream_thread = None
        self._streaming_active = False
        if was_running:
            self._streaming_completed = False

    def _populate_table(self) -> None:
        items = self._normalized_items
        self._table.blockSignals(True)
        self._table.setRowCount(len(items))
        for row, entry in enumerate(items):
            self._populate_row(row, entry)
        self._table.blockSignals(False)
        self._refresh_instrument_options()

    def _populate_row(self, row: int, entry: _NormalizedItem) -> None:
        name_item = QTableWidgetItem(entry.display_name)
        name_item.setFlags(name_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        name_item.setCheckState(Qt.Checked if entry.include_by_default else Qt.Unchecked)
        self._table.setItem(row, 0, name_item)

        has_wcs_text = (
            self._localizer.get("filter.value.wcs_present", "Yes")
            if entry.has_wcs
            else self._localizer.get("filter.value.wcs_missing", "No")
        )
        wcs_item = QTableWidgetItem(has_wcs_text)
        wcs_item.setFlags(wcs_item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, 1, wcs_item)

        group_label = entry.group_label or self._localizer.get("filter.value.group_unassigned", "Unassigned")
        group_item = QTableWidgetItem(group_label)
        group_item.setFlags(group_item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, 2, group_item)

        instrument = entry.instrument or self._localizer.get("filter.value.unknown", "Unknown")
        instrument_item = QTableWidgetItem(instrument)
        instrument_item.setFlags(instrument_item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, 3, instrument_item)

    def _append_rows(self, entries: Sequence[_NormalizedItem]) -> None:
        if not entries:
            return
        start_row = self._table.rowCount()
        self._table.blockSignals(True)
        self._table.setRowCount(start_row + len(entries))
        for offset, entry in enumerate(entries):
            self._populate_row(start_row + offset, entry)
        self._table.blockSignals(False)
        self._update_summary_label()
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()
        self._refresh_instrument_options()

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------
    def _handle_item_changed(self, _: QTableWidgetItem) -> None:
        self._update_summary_label()
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()

    def _select_all(self) -> None:
        self._toggle_all(True)

    def _select_none(self) -> None:
        self._toggle_all(False)

    def _toggle_all(self, enabled: bool) -> None:
        state = Qt.Checked if enabled else Qt.Unchecked
        self._table.blockSignals(True)
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None:
                item.setCheckState(state)
        self._table.blockSignals(False)
        self._update_summary_label()
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()

    def _collect_selected_indices(self) -> list[int]:
        indices: list[int] = []
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is None:
                continue
            if item.checkState() == Qt.Checked:
                indices.append(row)
        return indices

    def _update_summary_label(self) -> None:
        total = self._table.rowCount()
        selected = 0
        for row in range(total):
            item = self._table.item(row, 0)
            if item and item.checkState() == Qt.Checked:
                selected += 1
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
        self._append_log(summary_display)

    def _on_sds_toggled(self, checked: bool) -> None:
        # Reset any previous auto-organisation overrides so the user can
        # recompute groups after changing SDS mode, and keep the button usable.
        self._auto_group_override_groups = None
        self._group_outline_bounds = []
        if not self._auto_group_running and self._auto_group_button is not None:
            try:
                self._auto_group_button.setEnabled(True)
            except Exception:
                pass
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

    def _append_log(self, message: str) -> None:
        if not message:
            return
        text = str(message)
        if self._activity_log_output is not None:
            try:
                self._activity_log_output.appendPlainText(text)
            except Exception:
                pass
        if self._log_output is None:
            return
        try:
            self._log_output.appendPlainText(text)
        except Exception:
            pass

    def _on_scan_recursive_toggled(self, checked: bool) -> None:
        self._scan_recursive = bool(checked)

    def _on_run_analysis(self) -> None:
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
        self._scan_thread.start()

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

    def _should_draw_footprints(self) -> bool:
        checkbox = self._draw_footprints_checkbox
        if checkbox is None:
            return False
        try:
            return bool(checkbox.isChecked())
        except Exception:
            return False

    def _ensure_entry_coordinates(self, entry: _NormalizedItem) -> Tuple[float | None, float | None]:
        if entry.center_ra_deg is not None and entry.center_dec_deg is not None:
            return entry.center_ra_deg, entry.center_dec_deg
        if fits is None:
            return None, None
        path = entry.file_path
        if not path or not os.path.isfile(path):
            return None, None
        try:
            header = fits.getheader(path, ignore_missing_end=True)
        except Exception:
            return None, None
        ra_deg, dec_deg = _extract_center_from_header(header)
        entry.center_ra_deg = ra_deg
        entry.center_dec_deg = dec_deg
        return ra_deg, dec_deg

    def _ensure_entry_footprint(self, entry: _NormalizedItem) -> List[Tuple[float, float]] | None:
        """Compute and cache a WCS footprint for ``entry`` when possible."""

        if entry.footprint_radec:
            return entry.footprint_radec
        if fits is None or WCS is None:
            return None
        path = entry.file_path
        if not path or not os.path.isfile(path):
            return None
        try:
            header = fits.getheader(path, ignore_missing_end=True)
        except Exception:
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
        return footprint

    def _collect_preview_points(self) -> List[Tuple[float, float, int | None]]:
        if self._preview_canvas is None:
            return []
        limit = self._resolve_preview_cap()
        points: list[Tuple[float, float, int | None]] = []
        for row, entry in enumerate(self._normalized_items):
            item = self._table.item(row, 0)
            if item is None or item.checkState() != Qt.Checked:
                continue
            ra_deg, dec_deg = entry.center_ra_deg, entry.center_dec_deg
            if ra_deg is None or dec_deg is None:
                ra_deg, dec_deg = self._ensure_entry_coordinates(entry)
            if ra_deg is None or dec_deg is None:
                continue
            cluster_idx = entry.cluster_index if isinstance(entry.cluster_index, int) else None
            points.append((ra_deg, dec_deg, cluster_idx))
            if limit is not None and len(points) >= limit:
                break
        return points

    def _schedule_preview_refresh(self) -> None:
        if self._preview_canvas is None or self._preview_refresh_pending:
            return
        self._preview_refresh_pending = True
        QTimer.singleShot(0, self._update_preview_plot)

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

        unassigned_text = self._localizer.get("filter.value.group_unassigned", "Unassigned")
        self._table.blockSignals(True)
        for row, entry in enumerate(self._normalized_items):
            item = self._table.item(row, 2)
            if item is None:
                continue
            label = entry.group_label or unassigned_text
            item.setText(label)
        self._table.blockSignals(False)

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
            item = self._table.item(row, 0)
            if item is None or item.checkState() != Qt.Checked:
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
        self._preview_refresh_pending = False
        if self._preview_axes is None or self._preview_canvas is None:
            return

        points = self._collect_preview_points()
        axes = self._preview_axes
        axes.clear()
        axes.set_xlabel(self._localizer.get("filter.preview.ra", "Right Ascension (°)"))
        axes.set_ylabel(self._localizer.get("filter.preview.dec", "Declination (°)"))
        axes.set_title(self._localizer.get("filter.preview.title", "Sky preview"))
        axes.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

        if not points:
            message = self._localizer.get(
                "filter.preview.empty",
                "No WCS information available for the current selection.",
            )
            axes.text(0.5, 0.5, message, ha="center", va="center", transform=axes.transAxes)
            self._preview_hint_label.setText(self._preview_default_hint)
            self._preview_canvas.draw_idle()
            return

        grouped_points: dict[int | None, list[Tuple[float, float]]] = {}
        for ra_deg, dec_deg, group_idx in points:
            grouped_points.setdefault(group_idx, []).append((ra_deg, dec_deg))

        legend_needed = False
        for group_idx, coords in grouped_points.items():
            ra_coords = [coord[0] for coord in coords]
            dec_coords = [coord[1] for coord in coords]
            if isinstance(group_idx, int):
                color = self._preview_color_cycle[group_idx % len(self._preview_color_cycle)]
                label_template = self._localizer.get("filter.preview.group_label", "Group {index}")
                try:
                    label = label_template.format(index=group_idx + 1)
                except Exception:
                    label = f"Group {group_idx + 1}"
                legend_needed = True
            else:
                color = "#3f7ad6"
                label = None
                if len(grouped_points) > 1:
                    label = self._localizer.get("filter.preview.group_unassigned", "Unassigned")
                    legend_needed = True

            scatter_kwargs = dict(c=color, s=24, alpha=0.85, edgecolors="none")
            if label:
                scatter_kwargs["label"] = label
            axes.scatter(ra_coords, dec_coords, **scatter_kwargs)

        footprints_for_preview: list[Tuple[list[float], list[float], int | None]] = []
        if self._should_draw_footprints():
            footprint_cap = self._resolve_preview_cap()
            used = 0
            for row, entry in enumerate(self._normalized_items):
                item = self._table.item(row, 0)
                if item is None or item.checkState() != Qt.Checked:
                    continue
                fp = entry.footprint_radec or self._ensure_entry_footprint(entry)
                if not fp:
                    continue
                ra_fp = [p[0] for p in fp]
                dec_fp = [p[1] for p in fp]
                if not ra_fp or not dec_fp:
                    continue
                cluster_idx = entry.cluster_index if isinstance(entry.cluster_index, int) else None
                footprints_for_preview.append((ra_fp, dec_fp, cluster_idx))
                used += 1
                if footprint_cap is not None and used >= footprint_cap:
                    break

            for ra_fp, dec_fp, cluster_idx in footprints_for_preview:
                if len(ra_fp) < 2 or len(dec_fp) < 2:
                    continue
                if isinstance(cluster_idx, int):
                    color = self._preview_color_cycle[cluster_idx % len(self._preview_color_cycle)]
                else:
                    color = "#7b6fd6"
                xs = list(ra_fp)
                ys = list(dec_fp)
                xs.append(xs[0])
                ys.append(ys[0])
                axes.plot(xs, ys, color=color, linewidth=0.9, alpha=0.8)

        if Rectangle is not None and self._group_outline_bounds:
            for ra_min, ra_max, dec_min, dec_max in self._group_outline_bounds:
                width = max(0.0, ra_max - ra_min)
                height = max(0.0, dec_max - dec_min)
                if width <= 0 or height <= 0:
                    continue
                rect = Rectangle(
                    (ra_min, dec_min),
                    width,
                    height,
                    linewidth=1.2,
                    linestyle="--",
                    edgecolor="#d64b3f",
                    facecolor="none",
                    alpha=0.9,
                )
                axes.add_patch(rect)

        if legend_needed and len(grouped_points) > 1:
            axes.legend(loc="upper right", fontsize="small")

        ra_values = [pt[0] for pt in points]
        dec_values = [pt[1] for pt in points]
        for ra_fp, dec_fp, _ in footprints_for_preview:
            ra_values.extend(ra_fp)
            dec_values.extend(dec_fp)
        if not ra_values or not dec_values:
            self._preview_canvas.draw_idle()
            return
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
            count=len(points),
            cap=preview_cap_value or "∞",
        )
        clusters_present = sum(1 for key in grouped_points.keys() if isinstance(key, int))
        if clusters_present:
            cluster_fragment = self._localizer.get("filter.preview.groups_hint", "{groups} cluster(s)")
            try:
                cluster_formatted = cluster_fragment.format(groups=clusters_present)
            except Exception:
                cluster_formatted = f"{clusters_present} cluster(s)"
            summary_text = f"{summary_text} – {cluster_formatted}"
        try:
            self._preview_hint_label.setText(
                summary_text.format(count=len(points), cap=preview_cap_value or "∞")
            )
        except Exception:
            self._preview_hint_label.setText(summary_text)
        self._preview_canvas.draw_idle()

    # ------------------------------------------------------------------
    # QDialog API
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_stream_worker()
        if self._scan_worker is not None:
            self._scan_worker.request_stop()
        if self._scan_thread is not None:
            self._scan_thread.quit()
            self._scan_thread.wait(2000)
            self._scan_thread = None
        self._scan_worker = None
        super().closeEvent(event)

    def accept(self) -> None:  # noqa: D401 - inherit docstring
        self._accepted = True
        super().accept()

    def reject(self) -> None:  # noqa: D401 - inherit docstring
        self._accepted = False
        super().reject()

    # ------------------------------------------------------------------
    # Public helpers queried by ``launch_filter_interface_qt``
    # ------------------------------------------------------------------
    def selected_items(self) -> List[Any]:
        results: list[Any] = []
        for row, entry in enumerate(self._normalized_items):
            item = self._table.item(row, 0)
            if item and item.checkState() == Qt.Checked:
                results.append(entry.original)
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
    def _coerce_float(value: Any, default: float) -> float:
        try:
            result = float(value)
        except Exception:
            return default
        if math.isnan(result) or math.isinf(result):
            return default
        return result

    def _detect_primary_instrument_label(self) -> str | None:
        for entry in self._normalized_items:
            label = self._normalize_string(entry.instrument)
            if label:
                return label

        if fits is not None:
            header_checks = 0
            for entry in self._normalized_items:
                if not entry.file_path or not os.path.isfile(entry.file_path):
                    continue
                try:
                    header = fits.getheader(entry.file_path, ignore_missing_end=True)
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

        output_dir_raw = self._config_value("output_dir", "")
        output_dir = str(output_dir_raw or "").strip()
        if not output_dir:
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
            fits_path, json_path = resolve_global_wcs_output_paths(output_dir, wcs_output_cfg)
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
                    item = self._table.item(row, 0)
                    if item and item.checkState() == Qt.Checked:
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
                if not path or not os.path.isfile(path):
                    continue
                try:
                    header = _fits.getheader(path, ignore_missing_end=True)
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
        resolved_count = self._resolved_wcs_count()
        if resolved_count:
            overrides["resolved_wcs_count"] = int(resolved_count)

        if self._sds_checkbox is not None:
            overrides["sds_mode"] = bool(self._sds_checkbox.isChecked())
        overrides["filter_overcap_allowance_pct"] = int(self._resolve_overcap_percent())
        overrides["filter_enable_coverage_first"] = bool(self._coverage_first_enabled_flag)
        if hasattr(self, "_auto_angle_enabled") and not getattr(self, "_auto_angle_enabled", True):
            overrides["cluster_orientation_split_deg"] = float(self._angle_split_value)

        excluded_indices: list[int] = []
        try:
            for idx, _entry in enumerate(self._normalized_items):
                item = self._table.item(idx, 0)
                if item is None:
                    continue
                if item.checkState() != Qt.Checked:
                    excluded_indices.append(idx)
        except Exception:
            excluded_indices = []
        if excluded_indices:
            overrides["filter_excluded_indices"] = excluded_indices

        metadata_update = self._build_metadata_overrides(overrides)

        sds_flag = bool(metadata_update.get("sds_mode"))
        mode_value = str(metadata_update.get("mode") or "").strip().lower()
        require_global_plan = sds_flag or mode_value == "seestar"

        success = False
        meta_payload: dict[str, Any] | None = None
        path_payload: dict[str, Any] | None = None
        if require_global_plan:
            success, meta_payload, path_payload = self._ensure_global_wcs_for_selection(True, None)

        if success and meta_payload and path_payload:
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
            return [
                entry.original
                for entry in _iter_normalized_entries(
                    self._input_payload,
                    self._initial_overrides,
                    scan_recursive=self._scan_recursive,
                )
            ]
        return [entry.original for entry in self._normalized_items]

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
            item = self._table.item(row, 1)
            if item is not None:
                text = (
                    self._localizer.get("filter.value.wcs_present", "Yes")
                    if has_wcs
                    else self._localizer.get("filter.value.wcs_missing", "No")
                )
                item.setText(text)
        instrument_changed = False
        instrument = payload.get("instrument")
        if instrument:
            if entry.instrument != instrument:
                entry.instrument = instrument
                instrument_changed = True
            item = self._table.item(row, 3)
            if item is not None:
                item.setText(str(instrument))
        ra_deg = payload.get("center_ra_deg")
        dec_deg = payload.get("center_dec_deg")
        if isinstance(ra_deg, (int, float)) and isinstance(dec_deg, (int, float)):
            entry.center_ra_deg = float(ra_deg)
            entry.center_dec_deg = float(dec_deg)
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
        self._append_log(message)

    def _on_scan_finished(self) -> None:
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

    def _populate_astap_instances_combo(self) -> None:
        if self._astap_instances_combo is None:
            return
        combo = self._astap_instances_combo
        combo.blockSignals(True)
        combo.clear()
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
        parsed = max(1, int(value))
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
        if owns_app:
            app.quit()

    if accepted:
        return selected, True, overrides

    return all_items, False, None


__all__ = ["FilterQtDialog", "launch_filter_interface_qt"]
