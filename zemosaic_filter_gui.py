# zemosaic_filter_gui.py
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

from typing import List, Dict, Any, Optional, Callable
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
import os
import sys
import shutil
import datetime
import importlib
import importlib.util
import time
import traceback
import threading
import queue
import json
import logging
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from zemosaic_utils import EXCLUDED_DIRS, is_path_excluded


logger = logging.getLogger(__name__)


# --- Instrument detection helpers -------------------------------------------
def _detect_instrument_from_header(header: dict | None) -> str:
    if not header:
        return "Unknown"

    def _clean(value: Any) -> str:
        try:
            return str(value).strip()
        except Exception:
            return ""

    creator_raw = _clean(header.get("CREATOR"))
    creator = creator_raw.upper()
    instrume_raw = _clean(header.get("INSTRUME"))
    instrume = instrume_raw.upper()

    if "SEESTAR S50" in creator:
        return "Seestar S50"
    if "SEESTAR S30" in creator:
        return "Seestar S30"
    if "ASIAIR" in creator:
        # Prefer the more specific INSTRUME label when available (e.g. ZWO ASIAIR Plus)
        return instrume_raw or "ASIAIR"
    if instrume.startswith("ZWO ASI") or "ASI" in instrume:
        return instrume_raw or "ASI (Unknown)"
    if instrume_raw:
        # Fall back to the raw INSTRUME label for other devices
        return instrume_raw
    return "Unknown"


def _group_center_deg(group: list[dict]) -> Optional[tuple[float, float]]:
    """Return the average RA/DEC (degrees) of a group if available."""

    ras: list[float] = []
    decs: list[float] = []
    for info in group:
        ra = info.get("RA")
        dec = info.get("DEC")
        if ra is not None and dec is not None:
            try:
                ras.append(float(ra))
                decs.append(float(dec))
            except Exception:
                continue
    if not ras:
        return None
    return (sum(ras) / len(ras), sum(decs) / len(decs))


def _angular_sep_deg(a: Optional[tuple[float, float]], b: Optional[tuple[float, float]]) -> float:
    """Approximate angular separation in degrees between two (RA, DEC) tuples."""

    if not a or not b:
        return 9999.0
    dra = abs(a[0] - b[0])
    ddec = abs(a[1] - b[1])
    return (dra ** 2 + ddec ** 2) ** 0.5


def _merge_small_groups(
    groups: list[list[dict]],
    min_size: int,
    cap: int,
    *,
    cap_allowance: Optional[int] = None,
    compute_dispersion: Optional[Callable[[list[tuple[float, float]]], float]] = None,
    max_dispersion_deg: Optional[float] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[list[dict]]:
    """Merge undersized groups with their nearest neighbour when safe.

    Parameters
    ----------
    groups : list[list[dict]]
        Groups to examine.
    min_size : int
        Minimum size below which a group becomes a merge candidate.
    cap : int
        Hard cap (without allowance) used as base reference.
    cap_allowance : Optional[int]
        Optional absolute cap allowing temporary overflows.
    compute_dispersion : Optional[Callable]
        Callable returning the maximum angular separation (deg) for coordinates.
    max_dispersion_deg : Optional[float]
        Reject merges that would push dispersion beyond this threshold.
    log_fn : Optional[Callable[[str], None]]
        Optional logging callback invoked for each successful merge.
    """

    if not groups or min_size <= 0 or cap <= 0:
        return groups

    cap_limit = int(cap_allowance) if cap_allowance and cap_allowance > 0 else int(cap)
    cap_limit = max(cap_limit, int(cap))

    merged_flags = [False] * len(groups)
    centers = [_group_center_deg(g) for g in groups]

    def _collect_coords(payload: list[dict]) -> list[tuple[float, float]]:
        coords: list[tuple[float, float]] = []
        for info in payload:
            ra = info.get("RA")
            dec = info.get("DEC")
            if ra is None or dec is None:
                continue
            try:
                coords.append((float(ra), float(dec)))
            except Exception:
                continue
        return coords

    for i, group in enumerate(groups):
        if merged_flags[i] or len(group) >= min_size:
            continue

        best_j: Optional[int] = None
        best_dist = float("inf")
        for j, neighbour in enumerate(groups):
            if i == j or merged_flags[j]:
                continue
            dist = _angular_sep_deg(centers[i], centers[j])
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j is None:
            continue

        candidate_size = len(groups[best_j]) + len(group)
        if candidate_size > cap_limit:
            continue

        if compute_dispersion is not None and max_dispersion_deg is not None and max_dispersion_deg > 0:
            coords_combined = _collect_coords(groups[best_j]) + _collect_coords(group)
            if coords_combined:
                try:
                    dispersion_val = float(compute_dispersion(coords_combined))
                except Exception:
                    dispersion_val = None
                if dispersion_val is not None and dispersion_val > max_dispersion_deg:
                    continue

        groups[best_j].extend(group)
        merged_flags[i] = True
        centers[best_j] = _group_center_deg(groups[best_j])
        if log_fn is not None:
            try:
                log_fn(
                    f"Merged group {i} ({len(group)} imgs) into {best_j} (size={len(groups[best_j])})"
                )
            except Exception:
                pass

    return [grp for idx, grp in enumerate(groups) if not merged_flags[idx]]


def _format_sizes_histogram(sizes: list[int], max_buckets: int = 6) -> str:
    """Return a compact histogram string for group sizes."""

    if not sizes:
        return "[]"

    counter = Counter(sizes)
    pairs = sorted(counter.items(), key=lambda kv: (-kv[1], -kv[0]))
    head = ", ".join(f"{size}×{count}" for size, count in pairs[:max_buckets])
    tail = len(pairs) - max_buckets
    return head + (f", +{tail} more" if tail > 0 else "")


def _compute_dynamic_footprint_budget(
    total_items: int,
    preview_cap: Any,
    *,
    max_footprints: int,
) -> int:
    """Return the footprint budget capped dynamically for preview rendering.

    Parameters
    ----------
    total_items : int
        Number of catalogue entries that could be rendered.
    preview_cap : Any
        User-provided preview cap (may be ``None`` or a string).
    max_footprints : int
        Hard upper bound derived from configuration/CLI arguments.
    """

    try:
        base_cap = int(max_footprints)
    except Exception:
        base_cap = 0

    preview_limit: Optional[int] = None
    try:
        coerced_preview = int(preview_cap)
        if coerced_preview <= 0:
            return 0
        preview_limit = coerced_preview
    except Exception:
        preview_limit = None

    if preview_limit is not None:
        base_cap = min(base_cap or preview_limit, preview_limit)

    base_cap = max(50, base_cap) if base_cap else 50

    if total_items >= 8000:
        cap = min(200, base_cap)
    elif total_items >= 5000:
        cap = 0
    elif total_items >= 2000:
        cap = min(600, base_cap)
    elif total_items >= 1500:
        cap = min(400, base_cap)
    else:
        cap = min(base_cap, 1500)

    if preview_limit is not None:
        cap = min(cap, preview_limit)

    cap = min(cap, base_cap) if base_cap else cap

    try:
        return max(0, int(cap))
    except Exception:
        return 1


def launch_filter_interface(
    raw_files_with_wcs_or_dir,
    initial_overrides: Optional[Dict[str, Any]] = None,
    *,
    stream_scan: bool = False,
    scan_recursive: bool = True,
    batch_size: int = 100,
    preview_cap: int = 200,
    solver_settings_dict: Optional[Dict[str, Any]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Display an optional Tkinter GUI to filter WCS-resolved images.

    Parameters
    ----------
    raw_files_with_wcs_or_dir : list[dict] | str
        Either the legacy list of metadata dictionaries or, when ``stream_scan``
        is True, a directory path to crawl lazily for FITS files.
        Each dict should ideally include:
          - 'path' (or 'path_raw') : absolute path to the FITS file
          - 'wcs' : astropy.wcs.WCS object
          - 'shape' (optional) : (H, W)
          - 'index' (optional) : processing index
          - 'center' (optional) : astropy.coordinates.SkyCoord
          - 'header' (optional) : astropy.io.fits.Header (used to infer shape)

    stream_scan : bool, optional
        Enable lazy directory crawling instead of receiving a pre-computed
        metadata list.  When enabled, ``raw_files_with_wcs_or_dir`` must be a
        directory path.
    scan_recursive : bool, optional
        When ``stream_scan`` is active, decide whether to descend into
        sub-directories (default: True).
    batch_size : int, optional
        Number of files to accumulate before pushing a batch to the UI while
        streaming (default: 100).
    preview_cap : int, optional
        Maximum number of footprints drawn on the Matplotlib preview to avoid
        locking the UI for very large directories (default: 200).
    solver_settings_dict : dict, optional
        Dictionary of solver configuration selected in the main GUI. When
        provided, values such as ASTAP paths and search radius override the
        defaults loaded from configuration files.
    config_overrides : dict, optional
        Additional configuration values collected from the main GUI (for
        example the ASTAP data directory or clustering caps). These override
        the defaults detected by this module.

    Returns
    -------
    tuple[list[dict], bool, dict | None]
        (filtered_list, accepted, overrides) where accepted is True only when
        Validate is clicked; False when Cancel is clicked or the window is closed.
        overrides can contain optional metadata collected in the filter GUI, such
        as pre-computed master tile groupings or counters for resolved WCS files:
          {
             "preplan_master_groups": list[list[dict]],
             "autosplit_cap": int,
             "filter_excluded_indices": list[int],
             "resolved_wcs_count": int,
          }
        Keys are included only when relevant actions were triggered.
    """
    # Early validation and fail-safe behavior
    stream_mode = False
    input_dir: Optional[str] = None
    input_dir_path: Optional[Path] = None
    excluded_input_dir = False
    pending_startup_messages: list[tuple[str, str]] = []
    startup_status_messages: list[str] = []
    exclusion_notified_paths: set[str] = set()
    excluded_paths_pending: list[Path] = []

    # Robust detection: if a directory path is provided, always use streaming
    # mode regardless of the explicit flag. This makes callers resilient to
    # mismatched parameters and guarantees the presence of the Analyse button.
    if isinstance(raw_files_with_wcs_or_dir, str):
        # Normalize user-provided path for robust directory detection
        candidate = str(raw_files_with_wcs_or_dir).strip().strip('"').strip("'")
        candidate = os.path.expanduser(os.path.expandvars(candidate))
        if os.path.isdir(candidate):
            stream_mode = True
            input_dir = candidate
        elif stream_scan:
            # Caller requested streaming but provided a non-directory path →
            # fail fast with a safe default.
            return [], False, None

    if not stream_mode:
        if not isinstance(raw_files_with_wcs_or_dir, list) or not raw_files_with_wcs_or_dir:
            return raw_files_with_wcs_or_dir, False, None

    raw_items_input = raw_files_with_wcs_or_dir

    if input_dir:
        try:
            input_dir_path = Path(input_dir).expanduser()
        except Exception:
            input_dir_path = Path(input_dir)
        try:
            if is_path_excluded(input_dir_path, EXCLUDED_DIRS):
                excluded_input_dir = True
        except Exception:
            pass

    # --- Simple debug logger (console + buffer flushed to UI log later) ---
    _dbg_msgs: list[str] = []
    def _dbg(msg: str) -> None:
        try:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
        except Exception:
            ts = "??:??:??"
        line = f"[FilterUI {ts}] {msg}"
        try:
            print(line)
        except Exception:
            pass
        _dbg_msgs.append(line)

    def _queue_startup_message(level: str, message: str) -> None:
        if not message:
            return
        pending_startup_messages.append((level, message))
        try:
            log_method = getattr(logger, level.lower(), logger.info)
        except Exception:
            log_method = logger.info
        try:
            log_method(message)
        except Exception:
            logger.info(message)

    def _remember_exclusion(path: Path, message: str) -> None:
        key = str(path)
        if key in exclusion_notified_paths:
            return
        exclusion_notified_paths.add(key)
        try:
            logger.debug("Skipping excluded path: %s", path)
        except Exception:
            pass
        display_msg = message
        try:
            display_msg = f"{message} ({path})"
        except Exception:
            pass
        _queue_startup_message("WARN", display_msg)

    try:
        # --- Optional localization support (autonomous fallback) ---
        # If running inside the ZeMosaic project folder structure, try to use
        # the existing localization system and the language set in main GUI.
        localizer = None
        cfg_defaults: Dict[str, Any] = {
            "astap_executable_path": "",
            "astap_data_directory_path": "",
            "astap_default_search_radius": 0.0,
            "astap_default_downsample": 0,
            "astap_default_sensitivity": 100,
            "auto_limit_frames_per_master_tile": True,
            "max_raw_per_master_tile": 0,
            "apply_master_tile_crop": False,
            "master_tile_crop_percent": 0.0,
        }
        cfg: Dict[str, Any] | None = None
        solver_settings_payload: Dict[str, Any] = {}
        lang_code = "en"

        # Ensure project directory and parent are on sys.path to import project modules
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        for candidate in (base_dir, project_root):
            if candidate and candidate not in sys.path:
                sys.path.insert(0, candidate)

        pkg_prefix = globals().get("__package__") or ""

        def _import_optional_module(*module_names: str):
            last_error: str | None = None
            for mod_name in module_names:
                spec = importlib.util.find_spec(mod_name)
                if spec is None:
                    last_error = f"Module not found: {mod_name}"
                    continue
                try:
                    return importlib.import_module(mod_name), None
                except Exception as exc:
                    last_error = f"{mod_name}: {exc}"
            return None, last_error

        # Try to load localization support from either legacy or packaged paths
        localization_candidates = [
            "locales.zemosaic_localization",
            "zemosaic_localization",
        ]
        if pkg_prefix:
            localization_candidates.insert(0, f"{pkg_prefix}.zemosaic_localization")
            localization_candidates.insert(0, f"{pkg_prefix}.locales.zemosaic_localization")
        localizer_cls = None
        localization_errors: list[str] = []
        for candidate in localization_candidates:
            module, import_error = _import_optional_module(candidate)
            if module is None:
                if import_error:
                    localization_errors.append(import_error)
                continue
            localizer_cls = getattr(module, "ZeMosaicLocalization", None)
            if localizer_cls is not None:
                break

        # Load persistent configuration if available
        config_candidates = []
        if pkg_prefix:
            config_candidates.append(f"{pkg_prefix}.zemosaic_config")
        config_candidates.append("zemosaic_config")
        zconfig_module, config_error = _import_optional_module(*config_candidates)

        if zconfig_module is not None:
            try:
                cfg = zconfig_module.load_config()
                if isinstance(cfg, dict):
                    lang_code = str(cfg.get("language", lang_code))
                    cfg_defaults.update({
                        "astap_executable_path": cfg.get("astap_executable_path", cfg_defaults["astap_executable_path"]),
                        "astap_data_directory_path": cfg.get("astap_data_directory_path", cfg_defaults["astap_data_directory_path"]),
                        "astap_default_search_radius": cfg.get("astap_default_search_radius", cfg_defaults["astap_default_search_radius"]),
                        "astap_default_downsample": cfg.get("astap_default_downsample", cfg_defaults["astap_default_downsample"]),
                        "astap_default_sensitivity": cfg.get("astap_default_sensitivity", cfg_defaults["astap_default_sensitivity"]),
                        "auto_limit_frames_per_master_tile": cfg.get("auto_limit_frames_per_master_tile", cfg_defaults["auto_limit_frames_per_master_tile"]),
                        "max_raw_per_master_tile": cfg.get("max_raw_per_master_tile", cfg_defaults["max_raw_per_master_tile"]),
                        "apply_master_tile_crop": cfg.get("apply_master_tile_crop", cfg_defaults["apply_master_tile_crop"]),
                        "master_tile_crop_percent": cfg.get("master_tile_crop_percent", cfg_defaults["master_tile_crop_percent"]),
                    })
            except Exception as exc:
                print(f"WARNING (Filter GUI): failed to load configuration: {exc}")
        elif config_error:
            print(f"WARNING (Filter GUI): configuration module unavailable: {config_error}")

        # Load solver settings (either provided by caller or defaults)
        solver_cls = None
        solver_candidates = []
        if pkg_prefix:
            solver_candidates.append(f"{pkg_prefix}.solver_settings")
        solver_candidates.append("solver_settings")
        solver_module, solver_error = _import_optional_module(*solver_candidates)
        if solver_module is not None:
            solver_cls = getattr(solver_module, "SolverSettings", None)
        elif solver_error:
            print(f"WARNING (Filter GUI): solver settings module unavailable: {solver_error}")

        if isinstance(solver_settings_dict, dict):
            solver_settings_payload.update(solver_settings_dict)
        elif solver_cls is not None:
            try:
                solver_settings = solver_cls.load_default()
            except Exception:
                solver_settings = solver_cls()
            try:
                solver_settings_payload.update(asdict(solver_settings))
            except Exception:
                pass

        if solver_settings_payload:
            exe_path = solver_settings_payload.get("astap_executable_path")
            data_path = solver_settings_payload.get("astap_data_directory_path")
            search_radius = solver_settings_payload.get("astap_search_radius_deg")
            downsample = solver_settings_payload.get("astap_downsample")
            sensitivity = solver_settings_payload.get("astap_sensitivity")

            if isinstance(exe_path, str) and exe_path:
                exe_path = os.path.expanduser(exe_path)
                cfg_defaults["astap_executable_path"] = exe_path
            if isinstance(data_path, str) and data_path:
                data_path = os.path.expanduser(data_path)
                cfg_defaults["astap_data_directory_path"] = data_path
            if search_radius is not None:
                cfg_defaults["astap_default_search_radius"] = search_radius
            if downsample is not None:
                cfg_defaults["astap_default_downsample"] = downsample
            if sensitivity is not None:
                cfg_defaults["astap_default_sensitivity"] = sensitivity

        combined_solver_settings: Dict[str, Any] = {}
        if solver_settings_payload:
            combined_solver_settings.update(solver_settings_payload)
        if isinstance(config_overrides, dict):
            combined_solver_settings.update(config_overrides)

        if localizer_cls is not None:
            try:
                localizer = localizer_cls(language_code=lang_code)
            except Exception as exc:
                print(f"WARNING (Filter GUI): failed to initialise localization: {exc}")
                localizer = None
        elif localization_errors:
            print(
                "WARNING (Filter GUI): localization module not available; proceeding with defaults. "
                f"Details: {localization_errors[-1]}"
            )

        if isinstance(config_overrides, dict):
            try:
                cfg_defaults.update(config_overrides)
            except Exception:
                pass

        def _tr(key: str, default_text: Optional[str] = None, **kwargs) -> str:
            if localizer is not None:
                try:
                    # Pass default_text so JSON keys are optional
                    return localizer.get(key, default_text, **kwargs)
                except Exception:
                    pass
            # Fallback: return default_text or key placeholder
            return default_text if default_text is not None else f"_{key}_"

        if excluded_input_dir:
            msg = _tr(
                "FILTER_MOVE_FAILED_INPUT",
                "Input path points to an excluded folder. Please select another folder.",
            )
            startup_status_messages.append(msg)
            _queue_startup_message("WARN", msg)

        if excluded_paths_pending:
            exclusion_msg = _tr(
                "FILTER_EXCLUDED_DIR",
                'The folder "unaligned_by_zemosaic" is excluded and will not be scanned.',
            )
            for pending_path in list(excluded_paths_pending):
                _remember_exclusion(pending_path, exclusion_msg)
            excluded_paths_pending.clear()

        # Imports kept inside to avoid import-time errors affecting pipeline
        import tkinter as tk
        from tkinter import ttk, messagebox, scrolledtext

        from core.tk_safe import patch_tk_variables

        patch_tk_variables()
        heavy_import_error: ImportError | None = None
        try:
            import numpy as np
            from astropy.coordinates import SkyCoord
            from astropy.io import fits
            from astropy.wcs import WCS
            from astropy.wcs.utils import pixel_to_skycoord
            import astropy.units as u
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.collections import LineCollection
            from matplotlib.colors import to_rgba
        except ImportError as exc:
            heavy_import_error = exc

        if heavy_import_error is not None:
            msg = (
                "Missing optional dependency (numpy/matplotlib/astropy).\n"
                "Install the packages and retry."
            )
            detail = f" Details: {heavy_import_error}"
            try:
                messagebox.showerror("ZeMosaic Filter", msg + detail)
            except Exception:
                print(f"ERROR (Filter GUI): {msg}{detail}")
            return raw_items_input, False, None

        def _write_header_to_fits_local(file_path: str, header_obj) -> None:
            """Safely persist ``header_obj`` into ``file_path`` FITS header."""

            if header_obj is None:
                return
            try:
                with fits.open(file_path, mode="update", memmap=False) as hdul:
                    hdul[0].header.update(header_obj)
                    hdul.flush()
            except Exception as exc:
                logger.debug("[FilterUI] Failed to write header for %s: %s", file_path, exc)

        def _has_celestial_wcs(header_obj) -> bool:
            """Return True when ``header_obj`` contains a valid celestial WCS."""

            if header_obj is None:
                return False
            try:
                wcs_obj = WCS(header_obj, naxis=2, relax=True)
            except Exception:
                return False
            return bool(getattr(wcs_obj, "is_celestial", False))

        def footprint_wh_deg(wcs_obj: Any) -> tuple[float, float]:
            """Return footprint width/height in degrees for a given WCS."""

            try:
                ny: Optional[int] = None
                nx: Optional[int] = None
                if getattr(wcs_obj, "array_shape", None):
                    ny, nx = wcs_obj.array_shape  # type: ignore[attr-defined]
                elif getattr(wcs_obj, "pixel_shape", None):
                    nx, ny = wcs_obj.pixel_shape  # type: ignore[attr-defined]
                if ny is None or nx is None:
                    return (float("nan"), float("nan"))

                px = np.array([[0, 0], [nx, 0], [nx, ny], [0, ny]], dtype=float)
                sky = pixel_to_skycoord(px[:, 0], px[:, 1], wcs_obj)
                ra = sky.ra.deg
                dec = sky.dec.deg
                ra0 = float(np.nanmedian(ra))
                dec0 = float(np.nanmedian(dec))
                cos_dec0 = float(np.cos(np.deg2rad(dec0))) if np.isfinite(dec0) else 1.0
                if abs(cos_dec0) < 1e-6:
                    cos_dec0 = 1e-6 if cos_dec0 >= 0 else -1e-6
                x = (ra - ra0) * cos_dec0
                y = dec - dec0
                width = float(np.nanmax(x) - np.nanmin(x))
                height = float(np.nanmax(y) - np.nanmin(y))
                return width, height
            except Exception:
                return (float("nan"), float("nan"))

        astrometry_mod = None
        worker_mod = None
        _import_failures: Dict[str, list[str]] = {}

        def _record_import_failure(key: str, module_name: str, exc: BaseException) -> None:
            message = f"{module_name}: {exc}"
            try:
                tb = traceback.format_exception_only(type(exc), exc)
                if tb:
                    message = f"{module_name}: {tb[-1].strip()}"
            except Exception:
                pass
            _import_failures.setdefault(key, []).append(message)

        def _import_module_with_fallback(mod_name: str):
            """Attempt to import a module regardless of package layout."""

            candidates = [mod_name]
            pkg = globals().get("__package__") or ""
            if pkg:
                candidates.append(f"{pkg}.{mod_name}")
            if not mod_name.startswith("zemosaic"):
                candidates.append(f"zemosaic.{mod_name}")

            for candidate in candidates:
                try:
                    return importlib.import_module(candidate)
                except Exception as exc:
                    _record_import_failure(mod_name, candidate, exc)
            return None

        astrometry_mod = _import_module_with_fallback('zemosaic_astrometry')
        worker_mod = _import_module_with_fallback('zemosaic_worker')
        worker_import_failure_summary = "; ".join(_import_failures.get('zemosaic_worker', []))

        solve_with_astap = getattr(astrometry_mod, 'solve_with_astap', None) if astrometry_mod else None
        extract_center_from_header_fn = getattr(astrometry_mod, 'extract_center_from_header', None) if astrometry_mod else None
        astap_fits_module = getattr(astrometry_mod, 'fits', None) if astrometry_mod else None
        astap_astropy_available = bool(getattr(astrometry_mod, 'ASTROPY_AVAILABLE_ASTROMETRY', False)) if astrometry_mod else False
        cluster_func = getattr(worker_mod, 'cluster_seestar_stacks_connected', None) if worker_mod else None
        autosplit_func = getattr(worker_mod, '_auto_split_groups', None) if worker_mod else None
        compute_dispersion_func = getattr(worker_mod, '_compute_max_angular_separation_deg', None) if worker_mod else None
        solve_with_astrometry = getattr(worker_mod, 'solve_with_astrometry', None) if worker_mod else None
        solve_with_ansvr = getattr(worker_mod, 'solve_with_ansvr', None) if worker_mod else None

        max_footprints_override = None
        if isinstance(config_overrides, dict):
            try:
                override_val = int(config_overrides.get("footprints_max"))
                if override_val > 0:
                    max_footprints_override = override_val
            except Exception:
                max_footprints_override = None

        MAX_FOOTPRINTS = max_footprints_override or (int(preview_cap or 0) or 3000)
        cache_csv_path: Optional[str] = None
        if stream_mode and input_dir:
            cache_csv_path = os.path.join(input_dir, "headers_cache.csv")

        stream_queue: Optional[queue.Queue] = None
        # Stop flag for the streaming worker to support instant cancel/close
        stream_stop_event: Optional[threading.Event] = None
        stream_state = {
            "done": not stream_mode,
            "running": False,
            "pending_start": False,
            "status_message": None,
            "spawn_worker": None,
            "csv_loaded": False,
        }
        if cache_csv_path:
            stream_state["csv_path"] = cache_csv_path

        def _iter_fits_paths(root: str, recursive: bool = True):
            exclusion_msg = _tr(
                "FILTER_EXCLUDED_DIR",
                'The folder "unaligned_by_zemosaic" is excluded and will not be scanned.',
            )
            try:
                root_candidate = Path(root)
            except Exception:
                root_candidate = Path(str(root))

            if is_path_excluded(root_candidate, EXCLUDED_DIRS):
                _remember_exclusion(root_candidate, exclusion_msg)
                if not startup_status_messages:
                    msg = _tr(
                        "FILTER_MOVE_FAILED_INPUT",
                        "Input path points to an excluded folder. Please select another folder.",
                    )
                    startup_status_messages.append(msg)
                return

            def _should_skip(path_value: Path) -> bool:
                if not is_path_excluded(path_value, EXCLUDED_DIRS):
                    return False
                _remember_exclusion(path_value, exclusion_msg)
                return True

            if not recursive:
                try:
                    with os.scandir(root) as it:
                        for entry in it:
                            try:
                                entry_path = Path(entry.path)
                            except Exception:
                                entry_path = Path(str(entry.path))
                            if entry.is_dir():
                                if _should_skip(entry_path):
                                    continue
                                continue
                            if entry.is_file() and entry.name.lower().endswith((".fit", ".fits")):
                                if _should_skip(entry_path):
                                    continue
                                yield entry.path
                except Exception:
                    return
            else:
                for r, dirs, files in os.walk(root):
                    try:
                        current_dir = Path(r)
                    except Exception:
                        current_dir = Path(str(r))
                    if _should_skip(current_dir):
                        continue
                    filtered_dirs = []
                    for d in list(dirs):
                        child = current_dir / d
                        if _should_skip(child):
                            continue
                        filtered_dirs.append(d)
                    dirs[:] = filtered_dirs
                    for fn in files:
                        if not fn.lower().endswith((".fit", ".fits")):
                            continue
                        candidate = current_dir / fn
                        if _should_skip(candidate):
                            continue
                        yield str(candidate)

        def _compute_footprint_from_wcs(
            wcs_obj: Any,
            shape_hw: Optional[tuple[int, int]] = None,
        ) -> Optional[list[tuple[float, float]]]:
            if wcs_obj is None or not getattr(wcs_obj, "is_celestial", False):
                return None
            try:
                if shape_hw is not None and len(shape_hw) >= 2:
                    h, w = float(shape_hw[0]), float(shape_hw[1])
                else:
                    px_shape = getattr(wcs_obj, "pixel_shape", None)
                    if px_shape and len(px_shape) >= 2:
                        w = float(px_shape[0])
                        h = float(px_shape[1])
                    else:
                        arr_shape = getattr(wcs_obj, "array_shape", None)
                        if arr_shape and len(arr_shape) >= 2:
                            h = float(arr_shape[0])
                            w = float(arr_shape[1])
                        else:
                            return None
                if h <= 1.0 or w <= 1.0:
                    return None
                corners = (
                    (0.0, 0.0),
                    (w - 1.0, 0.0),
                    (w - 1.0, h - 1.0),
                    (0.0, h - 1.0),
                )
                result: list[tuple[float, float]] = []
                for x, y in corners:
                    sky = wcs_obj.pixel_to_world(float(x), float(y))
                    ra_v = float(sky.ra.to(u.deg).value)
                    dec_v = float(sky.dec.to(u.deg).value)
                    result.append((ra_v, dec_v))
                return result
            except Exception:
                return None

        def _minimal_header_payload(fpath: str) -> Dict[str, Any]:
            payload: Dict[str, Any] = {"path": fpath}
            try:
                hdr = fits.getheader(fpath, 0)
            except Exception:
                return payload

            nax1 = hdr.get("NAXIS1")
            nax2 = hdr.get("NAXIS2")
            if isinstance(nax1, (int, float)) and isinstance(nax2, (int, float)):
                payload["shape"] = (int(nax2), int(nax1))

            ra = hdr.get("CRVAL1")
            dec = hdr.get("CRVAL2")
            if ra is not None and dec is not None:
                try:
                    payload["center"] = (float(ra), float(dec))
                except Exception:
                    pass

            try:
                w = WCS(hdr, naxis=2, relax=True)
                if w is not None and getattr(w, "is_celestial", False):
                    payload["wcs"] = w
                    fp = _compute_footprint_from_wcs(w, payload.get("shape"))
                    if fp:
                        payload["footprint_radec"] = fp
            except Exception:
                pass

            keep_keys = {
                key: hdr.get(key)
                for key in (
                    "NAXIS1",
                    "NAXIS2",
                    "CRVAL1",
                    "CRVAL2",
                    "CREATOR",
                    "INSTRUME",
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
                )
                if key in hdr
            }
            if keep_keys:
                payload["header"] = keep_keys

            return payload

        def _move_problematic_file(path_value: str) -> Dict[str, Any]:
            info: Dict[str, Any] = {}
            if not path_value:
                return info
            try:
                src_path = Path(path_value)
            except Exception:
                return info
            if not src_path.exists():
                return info
            if is_path_excluded(src_path, EXCLUDED_DIRS):
                info["skipped"] = True
                return info

            target_dir_name = next(iter(EXCLUDED_DIRS), "unaligned_by_zemosaic")
            base_dir = input_dir_path or src_path.parent
            try:
                base_dir = Path(base_dir)
            except Exception:
                base_dir = Path(str(base_dir))
            try:
                base_dir = base_dir.expanduser().resolve(strict=False)
            except Exception:
                pass
            destination_dir = base_dir / target_dir_name
            try:
                destination_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            destination_path = destination_dir / src_path.name
            if destination_path.exists():
                timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
                stem = src_path.stem
                suffix = src_path.suffix
                counter = 1
                while destination_path.exists() and counter <= 50:
                    destination_path = destination_dir / f"{stem}{timestamp}_{counter}{suffix}"
                    counter += 1

            try:
                logger.debug("Moving problematic file %s -> %s", src_path, destination_path)
            except Exception:
                pass

            try:
                src_path.replace(destination_path)
                moved = True
            except Exception:
                try:
                    shutil.move(str(src_path), str(destination_path))
                    moved = True
                except Exception as exc_move:
                    info["error"] = str(exc_move)
                    try:
                        logger.error(
                            "Failed to move problematic file %s -> %s", src_path, destination_path, exc_info=True
                        )
                    except Exception:
                        pass
                    return info

            if moved:
                info.update(
                    {
                        "moved": True,
                        "source": str(src_path),
                        "destination": str(destination_path),
                        "filename": destination_path.name,
                    }
                )
                try:
                    logger.info("Moved problematic file to %s", destination_path)
                except Exception:
                    pass
            return info

        def _load_csv_bootstrap(path_csv: str) -> list[Dict[str, Any]]:
            rows: list[Dict[str, Any]] = []
            try:
                import csv

                with open(path_csv, "r", newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        entry: Dict[str, Any] = {"path": row.get("path", "")}
                        nax1 = row.get("NAXIS1")
                        nax2 = row.get("NAXIS2")
                        if nax1 and nax2:
                            try:
                                entry["shape"] = (int(nax2), int(nax1))
                            except Exception:
                                pass
                        ra_val = row.get("CRVAL1")
                        dec_val = row.get("CRVAL2")
                        if ra_val and dec_val:
                            try:
                                entry["center"] = (float(ra_val), float(dec_val))
                            except Exception:
                                pass
                        # Try to read persisted WCS footprint corners (if present)
                        try:
                            fp_ra_vals: list[float] = []
                            fp_dec_vals: list[float] = []
                            for k in ("1", "2", "3", "4"):
                                v_ra = row.get(f"FP_RA{k}")
                                v_dec = row.get(f"FP_DEC{k}")
                                if v_ra is not None and v_dec is not None and str(v_ra) != "" and str(v_dec) != "":
                                    fp_ra_vals.append(float(v_ra))
                                    fp_dec_vals.append(float(v_dec))
                            if fp_ra_vals and len(fp_ra_vals) == len(fp_dec_vals) and len(fp_ra_vals) >= 3:
                                entry["footprint_radec"] = [
                                    (fp_ra_vals[i], fp_dec_vals[i]) for i in range(len(fp_ra_vals))
                                ]
                        except Exception:
                            pass
                        header_subset = {
                            key: row.get(key)
                            for key in (
                                "NAXIS1",
                                "NAXIS2",
                                "CRVAL1",
                                "CRVAL2",
                                "CREATOR",
                                "INSTRUME",
                                "DATE-OBS",
                                "EXPTIME",
                                "FILTER",
                                "OBJECT",
                            )
                            if row.get(key) is not None
                        }
                        if header_subset:
                            entry["header"] = header_subset
                        rows.append(entry)
            except Exception:
                return []
            return rows

        def _export_csv(path_csv: str, items_for_csv: list[Dict[str, Any]]) -> None:
            try:
                import csv

                fieldnames = [
                    "path",
                    "NAXIS1",
                    "NAXIS2",
                    "CRVAL1",
                    "CRVAL2",
                    # Persist up to 4 WCS footprint corners (deg)
                    "FP_RA1", "FP_DEC1",
                    "FP_RA2", "FP_DEC2",
                    "FP_RA3", "FP_DEC3",
                    "FP_RA4", "FP_DEC4",
                    "CREATOR",
                    "INSTRUME",
                    "DATE-OBS",
                    "EXPTIME",
                    "FILTER",
                    "OBJECT",
                ]
                with open(path_csv, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for item in items_for_csv:
                        header_payload = item.get("header") or {}
                        shape_payload = item.get("shape")
                        nax1 = header_payload.get("NAXIS1")
                        nax2 = header_payload.get("NAXIS2")
                        if (nax1 is None or nax2 is None) and isinstance(shape_payload, (list, tuple)) and len(shape_payload) >= 2:
                            nax2 = shape_payload[0]
                            nax1 = shape_payload[1]
                        center_payload = item.get("center")
                        if hasattr(center_payload, "ra") and hasattr(center_payload, "dec"):
                            try:
                                center_tuple = (
                                    float(center_payload.ra.deg),
                                    float(center_payload.dec.deg),
                                )
                            except Exception:
                                center_tuple = (None, None)
                        else:
                            center_tuple = None
                            if isinstance(center_payload, (list, tuple)) and len(center_payload) >= 2:
                                try:
                                    center_tuple = (float(center_payload[0]), float(center_payload[1]))
                                except Exception:
                                    center_tuple = None
                        # Determine footprint corners, if available
                        fp_cols = {f"FP_RA{i}": "" for i in range(1, 5)}
                        fp_cols.update({f"FP_DEC{i}": "" for i in range(1, 5)})
                        fp_src = None
                        try:
                            pre = item.get("footprint_radec") or item.get("_precomp_fp")
                            if isinstance(pre, (list, tuple)) and len(pre) >= 3:
                                fp_src = [(float(p[0]), float(p[1])) for p in pre]
                        except Exception:
                            fp_src = None
                        if fp_src is None:
                            # Try computing from WCS if present
                            wcs_obj = item.get("wcs") if isinstance(item, dict) else None
                            if wcs_obj is not None and isinstance(shape_payload, (list, tuple)) and len(shape_payload) >= 2:
                                try:
                                    h = int(shape_payload[0]); w = int(shape_payload[1])
                                    corners = [
                                        (0.0, 0.0),
                                        (w - 1.0, 0.0),
                                        (w - 1.0, h - 1.0),
                                        (0.0, h - 1.0),
                                    ]
                                    fp_tmp: list[tuple[float, float]] = []
                                    for (x, y) in corners:
                                        sc = wcs_obj.pixel_to_world(x, y)
                                        ra = float(sc.ra.to(u.deg).value)
                                        dec = float(sc.dec.to(u.deg).value)
                                        fp_tmp.append((ra, dec))
                                    fp_src = fp_tmp
                                except Exception:
                                    fp_src = None
                        if isinstance(fp_src, list) and fp_src:
                            for i in range(min(4, len(fp_src))):
                                fp_cols[f"FP_RA{i+1}"] = fp_src[i][0]
                                fp_cols[f"FP_DEC{i+1}"] = fp_src[i][1]
                        writer.writerow(
                            {
                                "path": item.get("path", ""),
                                "NAXIS1": nax1 if nax1 is not None else "",
                                "NAXIS2": nax2 if nax2 is not None else "",
                                "CRVAL1": center_tuple[0] if center_tuple else "",
                                "CRVAL2": center_tuple[1] if center_tuple else "",
                                **fp_cols,
                                "CREATOR": header_payload.get("CREATOR", ""),
                                "INSTRUME": header_payload.get("INSTRUME", ""),
                                "DATE-OBS": header_payload.get("DATE-OBS", ""),
                                "EXPTIME": header_payload.get("EXPTIME", ""),
                                "FILTER": header_payload.get("FILTER", ""),
                                "OBJECT": header_payload.get("OBJECT", ""),
                            }
                        )
            except Exception as exc:
                _log_message(f"[CSV] Export failed: {exc}", level="WARN")

        initial_batches: list[list[Dict[str, Any]]] = []

        if stream_mode and input_dir:

            def _crawl_worker(target_queue: "queue.Queue[list[Dict[str, Any]] | None]",
                               stop_event: threading.Event) -> None:
                batch: list[Dict[str, Any]] = []
                minimum_batch = max(1, int(batch_size) if isinstance(batch_size, int) else 100)
                for idx, fpath in enumerate(_iter_fits_paths(input_dir, recursive=scan_recursive)):
                    # Allow cooperative, responsive cancellation
                    if stop_event.is_set():
                        break
                    item = _minimal_header_payload(fpath)
                    item["index"] = idx
                    batch.append(item)
                    if len(batch) >= minimum_batch:
                        if stop_event.is_set():
                            break
                        target_queue.put(batch)
                        batch = []
                if not stop_event.is_set():
                    if batch:
                        target_queue.put(batch)
                # Always signal completion to unblock consumers
                target_queue.put(None)
                stream_state["running"] = False

            def _spawn_worker() -> None:
                nonlocal stream_queue
                nonlocal stream_stop_event
                if stream_state.get("running"):
                    return
                if excluded_input_dir:
                    msg = _tr(
                        "FILTER_MOVE_FAILED_INPUT",
                        "Input path points to an excluded folder. Please select another folder.",
                    )
                    stream_state["running"] = False
                    stream_state["done"] = True
                    stream_state["status_message"] = msg
                    _queue_startup_message("WARN", msg)
                    return
                stream_queue = queue.Queue()
                stream_stop_event = threading.Event()
                stream_state["running"] = True
                stream_state["done"] = False
                stream_state["status_message"] = None
                threading.Thread(
                    target=_crawl_worker,
                    args=(stream_queue, stream_stop_event),
                    daemon=True,
                ).start()

            stream_state["spawn_worker"] = _spawn_worker
            stream_state["done"] = True
            if cache_csv_path and os.path.isfile(cache_csv_path):
                bootstrap_rows = _load_csv_bootstrap(cache_csv_path)
                if bootstrap_rows:
                    initial_batches.append(bootstrap_rows)
                    stream_state["csv_loaded"] = True
                    stream_state["status_message"] = _tr(
                        "filter_status_ready_csv",
                        "Loaded from CSV. Click Analyse to refresh.",
                    )
                else:
                    stream_state["pending_start"] = True
            else:
                stream_state["pending_start"] = True
        else:
            normalized: list[Dict[str, Any]] = []
            next_index = 0
            for entry in raw_items_input or []:
                try:
                    data = dict(entry)
                except Exception:
                    data = {"path": entry}
                path_candidate = (
                    data.get("path")
                    or data.get("path_raw")
                    or data.get("path_preprocessed_cache")
                )
                if path_candidate:
                    try:
                        candidate_path = Path(path_candidate)
                        if is_path_excluded(candidate_path, EXCLUDED_DIRS):
                            excluded_paths_pending.append(candidate_path)
                            continue
                    except Exception:
                        pass
                data["index"] = next_index
                normalized.append(data)
                next_index += 1
            if not normalized:
                return raw_items_input, False, None
            initial_batches.append(normalized)

        # Attempt to read astropy WCS only when needed
        # (objects are provided by caller; we don't import WCS explicitly here)

        # Normalize entries to a consistent structure the GUI can use
        class Item:
            def __init__(self, src: Dict[str, Any], idx: int):
                self.src = src
                self.index: int = int(src.get("index", idx))
                # Prefer 'path', then 'path_raw', then fallback to cached path
                self.path: str = (
                    src.get("path")
                    or src.get("path_raw")
                    or src.get("path_preprocessed_cache")
                    or f"<unknown_{idx}>"
                )
                self.wcs = src.get("wcs")
                header_payload = src.get("header")
                if header_payload is None and src.get("header_subset") is not None:
                    header_payload = src.get("header_subset")
                self.header = header_payload
                detected = _detect_instrument_from_header(self.header)
                self.instrument: str = detected.strip() if isinstance(detected, str) else "Unknown"
                if not self.instrument:
                    self.instrument = "Unknown"
                self.shape: Optional[tuple[int, int]] = None
                self.center: Optional[SkyCoord] = None
                self.phase0_center: Optional[SkyCoord] = None
                # Footprint computations are quite expensive for thousands of
                # entries.  Compute them lazily only when the UI actually needs
                # the polygon to be drawn.
                self._footprint_cache: Optional[np.ndarray] = None
                self._footprint_ready: bool = False
                self.refresh_geometry()

            def _coerce_skycoord(self, value: Any) -> Optional[SkyCoord]:
                if value is None:
                    return None
                if isinstance(value, SkyCoord):
                    return value
                try:
                    if isinstance(value, (list, tuple)) and len(value) >= 2:
                        ra_deg = float(value[0])
                        dec_deg = float(value[1])
                        return SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
                    if isinstance(value, dict):
                        ra_val = value.get("ra") or value.get("RA")
                        dec_val = value.get("dec") or value.get("DEC")
                        if ra_val is not None and dec_val is not None:
                            return SkyCoord(ra=float(ra_val) * u.deg, dec=float(dec_val) * u.deg, frame="icrs")
                except Exception:
                    return None
                return None

            def _infer_shape(self) -> Optional[tuple[int, int]]:
                shp = self.src.get("shape")
                if isinstance(shp, (list, tuple)) and len(shp) >= 2:
                    try:
                        h = int(shp[0])
                        w = int(shp[1])
                        if h > 0 and w > 0:
                            return (h, w)
                    except Exception:
                        pass
                header_obj = self.header
                if header_obj is not None:
                    try:
                        getter = header_obj.get if hasattr(header_obj, "get") else header_obj.__getitem__
                        naxis1 = int(getter("NAXIS1"))
                        naxis2 = int(getter("NAXIS2"))
                        if naxis1 > 0 and naxis2 > 0:
                            return (naxis2, naxis1)
                    except Exception:
                        pass
                if getattr(self.wcs, "pixel_shape", None):
                    try:
                        nx, ny = self.wcs.pixel_shape  # type: ignore[attr-defined]
                        h = int(ny)
                        w = int(nx)
                        if h > 0 and w > 0:
                            return (h, w)
                    except Exception:
                        pass
                if getattr(self.wcs, "array_shape", None):
                    try:
                        ny, nx = self.wcs.array_shape  # type: ignore[attr-defined]
                        h = int(ny)
                        w = int(nx)
                        if h > 0 and w > 0:
                            return (h, w)
                    except Exception:
                        pass
                return None

            def _center_from_wcs(self, shape_hw: Optional[tuple[int, int]]) -> Optional[SkyCoord]:
                if self.wcs is None:
                    return None
                try:
                    if shape_hw is not None:
                        h, w = shape_hw
                        xc = (w - 1) / 2.0
                        yc = (h - 1) / 2.0
                    else:
                        crpix = getattr(self.wcs.wcs, "crpix", None)
                        if crpix is not None and len(crpix) >= 2:
                            xc, yc = float(crpix[0]), float(crpix[1])
                        else:
                            xc, yc = 1023.5, 1023.5
                    sky = self.wcs.pixel_to_world(xc, yc)
                    ra = float(sky.ra.to(u.deg).value)
                    dec = float(sky.dec.to(u.deg).value)
                    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
                except Exception:
                    return None

            def _center_from_header(self) -> Optional[SkyCoord]:
                if extract_center_from_header_fn and self.header is not None:
                    try:
                        return extract_center_from_header_fn(self.header)
                    except Exception:
                        return None
                return None

            def _build_footprint(self, shape_hw: Optional[tuple[int, int]]) -> Optional[np.ndarray]:
                if self.wcs is None or shape_hw is None:
                    return None
                try:
                    h, w = shape_hw
                    corners = [
                        (0.0, 0.0),
                        (w - 1.0, 0.0),
                        (w - 1.0, h - 1.0),
                        (0.0, h - 1.0),
                    ]
                    ras = []
                    decs = []
                    for (x, y) in corners:
                        sc = self.wcs.pixel_to_world(x, y)
                        ras.append(float(sc.ra.to(u.deg).value))
                        decs.append(float(sc.dec.to(u.deg).value))
                    return np.column_stack([np.array(ras), np.array(decs)])
                except Exception:
                    return None

            def _apply_precomputed_footprint_if_available(self) -> None:
                """Load a precomputed footprint polygon from the source dict.

                The CSV bootstrap can store footprint corners as a list of
                (RA, Dec) tuples under the ``footprint_radec`` key. When this
                data is present, use it directly so the preview can display the
                blue frames without requiring a WCS object.
                """
                try:
                    pts = self.src.get("footprint_radec")
                    if isinstance(pts, (list, tuple)) and len(pts) >= 3:
                        arr = []
                        for p in pts:
                            try:
                                arr.append([float(p[0]), float(p[1])])
                            except Exception:
                                arr = []
                                break
                        if arr:
                            self._footprint_cache = np.array(arr, dtype=float)
                            self._footprint_ready = True
                            return
                except Exception:
                    pass
                # Optional fallback: FP_RA*/FP_DEC* scalars if present
                try:
                    vals: list[list[float]] = []
                    for k in ("1", "2", "3", "4"):
                        ra = self.src.get(f"FP_RA{k}")
                        dec = self.src.get(f"FP_DEC{k}")
                        if ra is None or dec is None or str(ra) == "" or str(dec) == "":
                            continue
                        vals.append([float(ra), float(dec)])
                    if len(vals) >= 3:
                        self._footprint_cache = np.array(vals, dtype=float)
                        self._footprint_ready = True
                        return
                except Exception:
                    pass

            def get_cached_footprint(self) -> Optional[np.ndarray]:
                """Return footprint if it has already been computed."""

                if self._footprint_ready:
                    return self._footprint_cache
                return None

            def ensure_footprint(self) -> Optional[np.ndarray]:
                """Compute and cache the footprint when needed."""

                if not self._footprint_ready:
                    self._footprint_cache = self._build_footprint(self.shape)
                    # Mark as ready even when None so we don't retry endlessly
                    self._footprint_ready = True
                return self._footprint_cache

            def refresh_geometry(self) -> None:
                self.shape = self._infer_shape()
                if self.shape and self.wcs is not None and getattr(self.wcs, "pixel_shape", None) is None:
                    try:
                        self.wcs.pixel_shape = (self.shape[1], self.shape[0])
                    except Exception:
                        pass
                self.phase0_center = self._coerce_skycoord(self.src.get("phase0_center"))
                direct_center = self._coerce_skycoord(self.src.get("center"))
                center = direct_center or self.phase0_center or self._center_from_wcs(self.shape) or self._center_from_header()
                self.center = center
                if self.center is not None:
                    self.src["center"] = self.center
                if self.shape is not None:
                    self.src["shape"] = self.shape
                # Heavy footprint computations are deferred until explicitly
                # requested by the UI via ``ensure_footprint``.
                self._footprint_cache = None
                self._footprint_ready = False
                # But if a precomputed polygon is available (from CSV), use it
                # so the preview can immediately display blue frames.
                self._apply_precomputed_footprint_if_available()

        raw_files_with_wcs: list[Dict[str, Any]] = []
        items: list[Item] = []
        instruments_found: set[str] = set()

        def _materialize_batch(batch: list[Dict[str, Any]]) -> None:
            for entry in batch:
                raw_files_with_wcs.append(entry)
                items.append(Item(entry, len(raw_files_with_wcs) - 1))

        for batch in initial_batches:
            _materialize_batch(batch)
        overrides_state: Dict[str, Any] = {}
        if isinstance(initial_overrides, dict):
            try:
                overrides_state.update(initial_overrides)
            except Exception:
                overrides_state = {}

        # Determine available sky coordinates.  Older projects or raw header
        # scans may not provide any RA/Dec information, in which case we still
        # want to display the file list even if the sky preview cannot show any
        # overlay.  Fall back to a neutral reference center so downstream logic
        # continues to work (thresholds, wrapping helpers, etc.).
        has_explicit_centers = False

        # Compute robust global center via unit-vector average
        def average_skycoord(coords: list[SkyCoord]) -> SkyCoord:
            arr = np.array([c.cartesian.xyz.value for c in coords])
            vec = arr.mean(axis=0)
            vec_norm = vec / np.linalg.norm(vec)
            sc = SkyCoord(x=vec_norm[0] * u.one, y=vec_norm[1] * u.one, z=vec_norm[2] * u.one, frame="icrs", representation_type="cartesian").spherical
            return SkyCoord(ra=sc.lon.to(u.deg), dec=sc.lat.to(u.deg), frame="icrs")

        global_center: SkyCoord = SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, frame="icrs")
        ref_ra = float(global_center.ra.to(u.deg).value)
        ref_dec = float(global_center.dec.to(u.deg).value)

        # RA wrapping helper around a reference RA
        def wrap_ra_deg(ra_deg: float, ref_deg: float) -> float:
            x = ra_deg
            r = ref_deg
            d = x - r
            # map difference to [-180, 180)
            d = (d + 180.0) % 360.0 - 180.0
            return r + d

        # Prepare Tkinter window (use Toplevel if a main Tk exists)
        parent_root = getattr(tk, "_default_root", None)
        root_is_toplevel = False
        if parent_root is not None:
            try:
                root = tk.Toplevel(parent_root)
                root_is_toplevel = True
                try:
                    root.transient(parent_root)
                except Exception:
                    pass
                try:
                    root.grab_set()  # modal
                except Exception:
                    pass
            except Exception:
                root = tk.Tk()
        else:
            root = tk.Tk()

        root.title(_tr(
            "filter_window_title",
            "ZeMosaic - Filtrer les images WCS (optionnel)" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "ZeMosaic - Filter WCS images (optional)"
        ))
        # S'assurer que la fenêtre apparaît au premier plan et prend le focus
        try:
            root.lift()
            root.attributes("-topmost", True)
            root.after(200, lambda: root.attributes("-topmost", False))
            root.focus_force()
        except Exception:
            pass
        try:
            root.protocol("WM_DELETE_WINDOW", root.destroy)
        except Exception:
            pass
        # Top-level layout: left plot, right checkboxes/actions
        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(1, weight=1)

        # Status strip (top): crawling indicator + CSV helpers
        status = ttk.Frame(main)
        status.grid(row=0, column=0, columnspan=2, sticky="ew")
        status.columnconfigure(1, weight=1)
        status_var = tk.StringVar(master=root, value=_tr("filter_status_crawling", "Crawling files… please wait"))
        ttk.Label(status, textvariable=status_var).grid(row=0, column=0, padx=6, pady=4, sticky="w")
        pb = ttk.Progressbar(status, mode="indeterminate", length=180)
        pb.grid(row=0, column=1, padx=6, pady=4, sticky="e")
        btn_bar = ttk.Frame(status)
        btn_bar.grid(row=0, column=2, padx=(0, 6), pady=4, sticky="e")
        analyse_btn = ttk.Button(
            btn_bar,
            text=_tr("filter_btn_analyse", "Analyse"),
            width=10,
        )
        export_btn = ttk.Button(
            btn_bar,
            text=_tr("filter_btn_export_csv", "Export CSV"),
            width=12,
        )
        analyse_btn.grid(row=0, column=0, padx=(0, 6))
        export_btn.grid(row=0, column=1)
        total_initial_entries = len(items)

        if startup_status_messages:
            try:
                status_var.set(startup_status_messages[-1])
            except Exception:
                pass
            try:
                pb.stop()
            except Exception:
                pass

        if stream_mode:
            if stream_state.get("status_message"):
                status_var.set(stream_state["status_message"])
                try:
                    pb.stop()
                except Exception:
                    pass
            elif stream_state.get("pending_start"):
                # Idle: wait for explicit click on Analyse
                try:
                    pb.stop()
                except Exception:
                    pass
                try:
                    status_var.set(
                        _tr(
                            "filter_status_click_analyse",
                            "Cliquez sur Analyse pour démarrer l'exploration." if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Ready — click Analyse to scan.",
                        )
                    )
                except Exception:
                    pass
            else:
                try:
                    pb.stop()
                except Exception:
                    pass
            if not cache_csv_path:
                try:
                    export_btn.state(["disabled"])
                except Exception:
                    pass
            _dbg(f"stream_mode=True; pending_start={stream_state.get('pending_start')} csv_loaded={stream_state.get('csv_loaded')} input_dir={input_dir}")
        else:
            if total_initial_entries > 400:
                try:
                    pb.start(80)
                except Exception:
                    pass
                status_var.set(
                    _tr(
                        "filter_status_populating",
                        "Preparing list… {current}/{total}",
                    ).format(current=0, total=total_initial_entries)
                )
            else:
                try:
                    pb.stop()
                except Exception:
                    pass
                status_var.set(_tr("filter_status_ready", "Crawling done."))
            try:
                analyse_btn.state(["disabled"])
            except Exception:
                pass
            try:
                export_btn.state(["disabled"])
            except Exception:
                pass

        # Matplotlib figure
        # Use constrained_layout to reduce internal padding and let the
        # figure adapt to the available space.
        fig = Figure(figsize=(7.0, 5.0), dpi=100, constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.set_xlabel(_tr(
            "filter_axis_ra_deg",
            "AD [deg]" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "RA [deg]"
        ))
        ax.set_ylabel(_tr(
            "filter_axis_dec_deg",
            "Dec [deg]"
        ))
        ax.set_aspect("equal", adjustable="datalim")
        # For sky-like view, invert RA axis (optional)
        ax.invert_xaxis()

        canvas = FigureCanvasTkAgg(fig, master=main)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=1, column=0, sticky="nsew")

        # Make the Matplotlib figure follow the widget size to avoid
        # large empty borders around the plot.
        def _apply_resize():
            try:
                w = max(50, int(canvas_widget.winfo_width()))
                h = max(50, int(canvas_widget.winfo_height()))
                # Resize figure in pixels -> inches
                fig.set_size_inches(w / fig.dpi, h / fig.dpi, forward=True)
                # Maximize axes area inside the figure when compatible with the
                # active Matplotlib layout engine.  Newer Matplotlib versions
                # (>=3.8) expose layout engines that reject ``subplots_adjust``
                # calls.  Skip the adjustment in that case to avoid runtime
                # warnings.
                try:
                    layout_engine = None
                    if hasattr(fig, "get_layout_engine"):
                        layout_engine = fig.get_layout_engine()
                    if layout_engine is None:
                        fig.subplots_adjust(left=0.06, right=0.995, bottom=0.08, top=0.98)
                except Exception:
                    pass
                canvas.draw_idle()
            except Exception:
                pass

        _resize_job = {"id": None}

        def _on_canvas_configure(_event=None):
            # Throttle frequent resizes during interactive dragging
            try:
                if _resize_job["id"] is not None:
                    canvas_widget.after_cancel(_resize_job["id"])  # type: ignore[arg-type]
                _resize_job["id"] = canvas_widget.after(60, lambda: (_apply_resize(), _resize_job.update({"id": None})))
            except Exception:
                pass

        # Bind and trigger once to sync initial size
        try:
            canvas_widget.bind("<Configure>", _on_canvas_configure)
        except Exception:
            pass
        try:
            root.update_idletasks()
            _apply_resize()
        except Exception:
            pass

        # Enable mouse-wheel zoom on the plot for easier selection
        def _setup_wheel_zoom(ax):
            base_scale = 1.2  # zoom factor per wheel notch

            def _orient_limits(lims):
                a, b = lims
                inv = a > b
                mn, mx = (b, a) if inv else (a, b)
                return mn, mx, inv

            def _apply_limits(ax, xmin, xmax, xinv, ymin, ymax, yinv):
                if xinv:
                    ax.set_xlim(xmax, xmin)
                else:
                    ax.set_xlim(xmin, xmax)
                if yinv:
                    ax.set_ylim(ymax, ymin)
                else:
                    ax.set_ylim(ymin, ymax)

            def on_scroll(event):
                try:
                    if event is None or event.inaxes is None:
                        return
                    ax_ = event.inaxes
                    xdata = event.xdata if event.xdata is not None else sum(ax_.get_xlim()) / 2.0
                    ydata = event.ydata if event.ydata is not None else sum(ax_.get_ylim()) / 2.0

                    xmin0, xmax0, xinv = _orient_limits(ax_.get_xlim())
                    ymin0, ymax0, yinv = _orient_limits(ax_.get_ylim())
                    width = max(1e-9, (xmax0 - xmin0))
                    height = max(1e-9, (ymax0 - ymin0))

                    # Choose scale direction
                    if getattr(event, 'button', 'up') in ('up', 4):
                        # zoom in
                        scale = 1.0 / base_scale
                    else:
                        # zoom out
                        scale = base_scale

                    new_w = width * scale
                    new_h = height * scale

                    # Compute relative position of mouse within current view
                    # using oriented (min->max) extents
                    relx = (xdata - xmin0) / max(1e-12, width)
                    rely = (ydata - ymin0) / max(1e-12, height)
                    relx = min(max(relx, 0.0), 1.0)
                    rely = min(max(rely, 0.0), 1.0)

                    xmin = xdata - relx * new_w
                    xmax = xdata + (1.0 - relx) * new_w
                    ymin = ydata - rely * new_h
                    ymax = ydata + (1.0 - rely) * new_h

                    # Avoid zero-span
                    if (xmax - xmin) < 1e-9:
                        pad = 5e-10
                        xmin -= pad; xmax += pad
                    if (ymax - ymin) < 1e-9:
                        pad = 5e-10
                        ymin -= pad; ymax += pad

                    _apply_limits(ax_, xmin, xmax, xinv, ymin, ymax, yinv)
                    canvas.draw_idle()
                except Exception:
                    pass

            try:
                canvas.mpl_connect('scroll_event', on_scroll)
            except Exception:
                pass

        _setup_wheel_zoom(ax)

        # Lazy recomputation of global center once we have some items
        _center_ready = {"ok": False}

        def _maybe_update_global_center():
            nonlocal global_center, has_explicit_centers, ref_ra, ref_dec
            coords = [it.center for it in items if it.center is not None]
            has_explicit_centers = bool(coords)
            if not coords:
                return
            global_center = coords[0] if len(coords) == 1 else average_skycoord(coords)
            ref_ra = float(global_center.ra.to(u.deg).value)
            ref_dec = float(global_center.dec.to(u.deg).value)
            _center_ready["ok"] = True

        _maybe_update_global_center()

        # Right panel with controls
        right = ttk.Frame(main)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        # The scrollable list lives at row=3 (row=1 reserved for threshold & filters)
        try:
            right.rowconfigure(3, weight=1)
        except Exception:
            right.rowconfigure(2, weight=1)
        # Reserve space for the aggregate controls container (row=4)
        # Ensure a reasonable minimum height so controls are always visible.
        try:
            right.rowconfigure(4, weight=0, minsize=180)
        except Exception:
            try:
                right.rowconfigure(4, minsize=180)
            except Exception:
                pass

        instrument_var = tk.StringVar(master=root, value="All")
        instrument_combo: Optional[ttk.Combobox] = None

        def _apply_instrument_filter(*_: object) -> None:
            chosen = instrument_var.get()
            if chosen == "All":
                for idx, it in enumerate(items):
                    _set_selected(idx, True)
            else:
                for idx, it in enumerate(items):
                    _set_selected(idx, it.instrument == chosen)
            update_visuals()
            try:
                canvas.draw_idle()
            except Exception:
                pass

        # Threshold controls
        thresh_frame = ttk.LabelFrame(
            right,
            text=_tr(
                "filter_exclude_by_distance_title",
                "Exclure par distance au centre" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Exclude by distance to center",
            ),
        )
        thresh_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ttk.Label(
            thresh_frame,
            text=_tr(
                "filter_distance_label",
                "Distance (deg) :" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Distance (deg):",
            ),
        ).grid(row=0, column=0, padx=4, pady=4)
        thresh_var = tk.StringVar(master=root, value="5.0")
        thresh_entry = ttk.Entry(thresh_frame, textvariable=thresh_var, width=8)
        thresh_entry.grid(row=0, column=1, padx=4, pady=4)
        def apply_threshold():
            try:
                thr = float(thresh_var.get())
            except Exception:
                messagebox.showwarning(
                    _tr(
                        "filter_invalid_value_title",
                        "Valeur invalide" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Invalid value",
                    ),
                    _tr(
                        "filter_invalid_value_message",
                        "Veuillez entrer une distance en degrés (nombre)." if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Please enter a distance in degrees (number).",
                    ),
                )
                return
            for idx, it in enumerate(items):
                if it.center is None:
                    continue
                sep = it.center.separation(global_center).to(u.deg).value
                if sep > thr:
                    _set_selected(idx, False)
            update_visuals()
        thresh_button = ttk.Button(
            thresh_frame,
            text=_tr(
                "filter_apply_threshold_button",
                "Exclure > X°" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Exclude > X°",
            ),
            command=apply_threshold,
        )
        thresh_button.grid(row=0, column=2, padx=4, pady=4)

        def _refresh_instrument_options() -> None:
            values: list[str] = ["All"]
            extras = sorted(x for x in instruments_found if x and x != "Unknown")
            if extras:
                values.extend(extras)
            else:
                values.append("Unknown")
            if instrument_combo is not None:
                try:
                    instrument_combo["values"] = tuple(values)
                except Exception:
                    pass
            if instrument_var.get() not in values:
                instrument_var.set("All")

        filter_frame = ttk.Frame(right)
        filter_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        filter_frame.columnconfigure(1, weight=1)

        ttk.Label(filter_frame, text="Instrument:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        try:
            instrument_combo = ttk.Combobox(
                filter_frame,
                textvariable=instrument_var,
                state="readonly",
                width=20,
            )
            instrument_combo.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
            instrument_combo["values"] = ("All",)
            instrument_combo.bind("<<ComboboxSelected>>", _apply_instrument_filter)
        except Exception:
            instrument_combo = None

        if not has_explicit_centers:
            try:
                thresh_entry.state(["disabled"])
            except Exception:
                thresh_entry.configure(state=tk.DISABLED)
            try:
                thresh_button.state(["disabled"])
            except Exception:
                pass

        # Logging pane
        log_frame = ttk.LabelFrame(
            right,
            text=_tr(
                "filter_log_panel_title",
                "Activity log" if 'fr' not in str(locals().get('lang_code', 'en')).lower() else "Journal d'activité",
            ),
        )
        log_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=(0, 5))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        log_widget = scrolledtext.ScrolledText(log_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        log_widget.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        async_events: "queue.Queue[tuple[Any, ...]]" = queue.Queue()
        resolve_state = {"running": False}
        _progress_log_state = {"last": 0.0}
        resolve_now_state: Dict[str, Any] = {
            "running": False,
            "executor": None,
            "futures": [],
            "total": 0,
            "done": 0,
            "success": 0,
            "log_path": None,
            "log_dir": None,
            "button": None,
            "progressbar": None,
            "status_var": None,
        }

        MAX_UI_MSG = 150
        UI_BUDGET_MS = 35.0
        LOG_FLUSH_INTERVAL = 0.2

        log_buffer: list[tuple[str, str]] = []
        last_log_flush = {"ts": 0.0}

        pending_visual_refresh: set[int] = set()
        visual_refresh_job = {"handle": None}
        footprints_restore_state = {"needs_restore": False, "value": True, "indices": set()}
        pending_outline_state: Dict[str, Any] = {"groups": None}

        def _enqueue_event(kind: str, *payload: Any) -> None:
            try:
                async_events.put_nowait((kind, *payload))
            except Exception:
                pass

        def _log_async(message: str, level: str = "INFO") -> None:
            if message is None:
                return
            try:
                lvl = str(level or "INFO")
            except Exception:
                lvl = "INFO"
            _enqueue_event("log", str(message), lvl)

        def _write_log_entries(entries: list[tuple[str, str]]) -> None:
            if not entries:
                return
            try:
                log_widget.configure(state=tk.NORMAL)
                for level, message in entries:
                    lvl = str(level or "INFO").upper()
                    text = str(message)
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    log_widget.insert(tk.END, f"[{timestamp}] [{lvl}] {text}\n")
                log_widget.configure(state=tk.DISABLED)
                log_widget.see(tk.END)
            except Exception:
                pass

        # Flush any buffered debug messages from early startup
        try:
            if _dbg_msgs:
                _write_log_entries([("INFO", msg) for msg in _dbg_msgs])
                _dbg_msgs.clear()
        except Exception:
            pass

        try:
            if pending_startup_messages:
                _write_log_entries(list(pending_startup_messages))
                pending_startup_messages.clear()
        except Exception:
            pass

        def _flush_log_buffer(force: bool = False) -> None:
            if not log_buffer:
                return
            now = time.monotonic()
            if not force and now - last_log_flush["ts"] < LOG_FLUSH_INTERVAL:
                return
            entries = list(log_buffer)
            log_buffer.clear()
            last_log_flush["ts"] = now
            _write_log_entries(entries)

        def _flush_visual_refresh() -> None:
            visual_refresh_job["handle"] = None
            if not pending_visual_refresh:
                return
            targets = sorted(pending_visual_refresh)
            pending_visual_refresh.clear()
            for idx in targets:
                try:
                    if idx < 0 or idx >= len(items):
                        continue
                    _ensure_wrapped_capacity(idx)
                    footprint_wrapped[idx] = None
                    centroid_wrapped[idx] = None
                    _prepare_visual_payload(idx)
                except Exception:
                    pass
            _schedule_visual_build(full=True)

        def _schedule_visual_refresh_flush() -> None:
            if visual_refresh_job.get("handle") is not None:
                return
            try:
                visual_refresh_job["handle"] = root.after(200, _flush_visual_refresh)
            except Exception:
                visual_refresh_job["handle"] = None
                _flush_visual_refresh()

        def _canvas_draw_idle() -> None:
            try:
                canvas.draw_idle()
            except Exception:
                try:
                    canvas.draw()
                except Exception:
                    pass

        def _flush_deferred_wcs_results(force: bool = False) -> None:
            buffer = wcs_async_state.get("buffer")
            if not isinstance(buffer, list) or not buffer:
                return
            chunk_size = max(1, int(DEFER_DRAW_CHUNK))
            flushed_any = False
            while buffer and (force or len(buffer) >= chunk_size):
                chunk = buffer[:chunk_size]
                del buffer[:chunk_size]
                pending_visual_refresh.update(chunk)
                _schedule_visual_refresh_flush()
                flushed_any = True
            if flushed_any:
                _canvas_draw_idle()

        def _flush_live_updates(force: bool = False) -> None:
            live_buffer = wcs_async_state.get("live_buffer")
            if not isinstance(live_buffer, list) or not live_buffer:
                return
            threshold = max(1, int(LIVE_DRAW_THROTTLE))
            if not force and len(live_buffer) < threshold:
                return
            targets = list(live_buffer)
            live_buffer.clear()
            pending_visual_refresh.update(targets)
            _schedule_visual_refresh_flush()
            _canvas_draw_idle()

        def _log_message(message: str, level: str = "INFO") -> None:
            _write_log_entries([(level, message)])

        def _handle_wcs_result(payload: Dict[str, Any]) -> None:
            if not isinstance(payload, dict):
                return

            idx_raw = payload.get("idx", payload.get("index"))
            try:
                idx_val = int(idx_raw)
            except Exception:
                idx_val = None

            ok = bool(payload.get("ok"))
            wcs_async_state["done"] = int(wcs_async_state.get("done") or 0) + 1
            if ok:
                wcs_async_state["success"] = int(wcs_async_state.get("success") or 0) + 1

            total = int(wcs_async_state.get("total") or 0)
            done_now = int(wcs_async_state.get("done") or 0)
            try:
                if total > 0:
                    progress_text = f"{done_now}/{total} solved…"
                else:
                    progress_text = f"{done_now} solved…"
                summary_var.set(_apply_summary_hint(progress_text))
            except Exception:
                pass

            header_obj = payload.get("header")
            wcs_obj = payload.get("wcs")
            footprint_payload = payload.get("corners") or payload.get("footprint")

            if isinstance(idx_val, int) and 0 <= idx_val < len(items):
                item = items[idx_val]
                if header_obj is not None:
                    try:
                        item.header = header_obj
                        item.src["header"] = header_obj
                    except Exception:
                        pass
                if wcs_obj is not None and getattr(wcs_obj, "is_celestial", False):
                    try:
                        item.wcs = wcs_obj
                        item.src["wcs"] = wcs_obj
                    except Exception:
                        pass
                sanitized_fp: list[tuple[float, float]] = []
                if isinstance(footprint_payload, (list, tuple)):
                    for entry in footprint_payload:
                        try:
                            ra_val, dec_val = entry
                            sanitized_fp.append((float(ra_val), float(dec_val)))
                        except Exception:
                            continue
                if sanitized_fp:
                    try:
                        item.src["footprint_radec"] = sanitized_fp
                    except Exception:
                        pass
                if wcs_obj is not None or sanitized_fp:
                    try:
                        item.refresh_geometry()
                    except Exception:
                        pass
                if ok and footprints_restore_state.get("needs_restore"):
                    indices_cache = footprints_restore_state.get("indices")
                    if not isinstance(indices_cache, set):
                        indices_cache = set()
                        footprints_restore_state["indices"] = indices_cache
                    indices_cache.add(idx_val)
                has_visual_update = bool(sanitized_fp) or (wcs_obj is not None)
                if has_visual_update:
                    if defer_overlay_var.get():
                        buffer = wcs_async_state.get("buffer")
                        if isinstance(buffer, list):
                            buffer.append(idx_val)
                            _flush_deferred_wcs_results()
                    else:
                        live_buffer = wcs_async_state.get("live_buffer")
                        if isinstance(live_buffer, list):
                            live_buffer.append(idx_val)
                            _flush_live_updates()

            log_path = wcs_async_state.get("log_path")
            path_value = payload.get("path")
            solver_value = payload.get("solver")
            error_value = payload.get("error")
            log_entry: Dict[str, Any] = {}
            if isinstance(path_value, str) and path_value:
                log_entry["path"] = path_value
            log_entry["ok"] = ok
            if solver_value:
                log_entry["solver"] = str(solver_value)
            if error_value:
                log_entry["error"] = str(error_value)
            if footprint_payload:
                corners_list: list[list[float]] = []
                if isinstance(footprint_payload, (list, tuple)):
                    for entry in footprint_payload:
                        try:
                            ra_val, dec_val = entry
                            corners_list.append([float(ra_val), float(dec_val)])
                        except Exception:
                            continue
                if corners_list:
                    log_entry["corners"] = corners_list
            if not ok:
                move_error = payload.get("move_error")
                if move_error:
                    _log_message(f"[FilterUI] Failed to move problematic file: {move_error}", level="ERROR")
                if payload.get("moved") and isinstance(idx_val, int):
                    moved_filename = payload.get("move_filename")
                    if not moved_filename and isinstance(path_value, str):
                        moved_filename = os.path.basename(path_value)
                    info_msg = _tr(
                        "FILTER_FILE_MOVED_UNALIGNED",
                        "Moved problematic file to 'unaligned_by_zemosaic': {filename}",
                    ).format(filename=moved_filename or "?")
                    _log_message(info_msg, level="INFO")
                    try:
                        status_var.set(
                            _tr(
                                "FILTER_WCS_FAIL_MOVED",
                                "WCS solve failed; file moved to 'unaligned_by_zemosaic'.",
                            )
                        )
                    except Exception:
                        pass
                    _remove_item(idx_val)
                    refresh_msg = _tr(
                        "FILTER_REFRESH_AFTER_MOVE",
                        "List refreshed after excluding unaligned files.",
                    )
                    summary_var.set(_apply_summary_hint(refresh_msg))
                    _log_message(refresh_msg, level="INFO")

            if log_entry.get("path") and log_path:
                try:
                    with open(str(log_path), "a", encoding="utf-8") as fw:
                        fw.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass

        def _poll_wcs_queue() -> None:
            wcs_async_state["poll_job"] = None
            q = wcs_async_state.get("ui_queue")
            if not isinstance(q, queue.Queue):
                return

            drained: list[Any] = []
            while True:
                try:
                    drained.append(q.get_nowait())
                except queue.Empty:
                    break
                except Exception:
                    break

            for payload in drained:
                if isinstance(payload, dict):
                    _handle_wcs_result(payload)

            running = bool(wcs_async_state.get("running"))
            total = int(wcs_async_state.get("total") or 0)
            done = int(wcs_async_state.get("done") or 0)
            pending_list = wcs_async_state.get("pending")
            still_running = False
            if isinstance(pending_list, list):
                for fut in list(pending_list):
                    try:
                        if fut is not None and not fut.done():
                            still_running = True
                            break
                    except Exception:
                        continue

            if running and (done >= total > 0 or (not still_running and q.empty())):
                _finalize_wcs_run()
                running = False

            if running:
                try:
                    wcs_async_state["poll_job"] = root.after(120, _poll_wcs_queue)
                except Exception:
                    wcs_async_state["poll_job"] = None
                    _finalize_wcs_run()
            else:
                _flush_deferred_wcs_results(force=True)
                _flush_live_updates(force=True)

        def _finalize_wcs_run() -> None:
            if not wcs_async_state.get("running"):
                return
            wcs_async_state["running"] = False
            _flush_deferred_wcs_results(force=True)
            _flush_live_updates(force=True)
            executor = wcs_async_state.get("executor")
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=False)
                except Exception:
                    pass
            wcs_async_state["executor"] = None
            pending_list = wcs_async_state.get("pending")
            if isinstance(pending_list, list):
                pending_list.clear()
            q = wcs_async_state.get("ui_queue")
            if isinstance(q, queue.Queue):
                try:
                    while True:
                        q.get_nowait()
                except queue.Empty:
                    pass
                except Exception:
                    pass
            success = int(wcs_async_state.get("success") or 0)
            try:
                _enqueue_event("resolve_done", success)
            except Exception:
                pass

        def _cancel_wcs_executor() -> None:
            wcs_async_state["stop"] = True
            poll_job = wcs_async_state.get("poll_job")
            if poll_job is not None:
                try:
                    root.after_cancel(poll_job)
                except Exception:
                    pass
                wcs_async_state["poll_job"] = None
            executor = wcs_async_state.get("executor")
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            pending_list = wcs_async_state.get("pending")
            if isinstance(pending_list, list):
                for fut in pending_list:
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                pending_list.clear()
            q = wcs_async_state.get("ui_queue")
            if isinstance(q, queue.Queue):
                try:
                    while True:
                        q.get_nowait()
                except queue.Empty:
                    pass
                except Exception:
                    pass
            wcs_async_state["running"] = False

        def _resolve_now_log_base(paths: list[str]) -> Optional[Path]:
            if input_dir_path is not None:
                return input_dir_path
            sanitized: list[str] = []
            for entry in paths:
                if not entry:
                    continue
                try:
                    sanitized.append(os.path.abspath(entry))
                except Exception:
                    continue
            if not sanitized:
                return None
            try:
                common = Path(os.path.commonpath(sanitized))
            except Exception:
                try:
                    common = Path(sanitized[0]).parent
                except Exception:
                    return None
            if common.is_file():
                try:
                    common = common.parent
                except Exception:
                    return None
            return common

        def _resolve_now_append_log(result: Dict[str, Any]) -> None:
            log_path = resolve_now_state.get("log_path")
            if not log_path:
                return
            path_val = result.get("path")
            log_dir = resolve_now_state.get("log_dir")
            try:
                if log_dir is not None and path_val:
                    rel_path = os.path.relpath(path_val, str(log_dir))
                elif path_val:
                    rel_path = os.path.basename(path_val)
                else:
                    rel_path = "<unknown>"
            except Exception:
                rel_path = os.path.basename(path_val) if path_val else "<unknown>"
            status = "OK" if result.get("ok") else "FAIL"
            if result.get("skipped"):
                status = "SKIP"
            code_val = result.get("return_code")
            code_txt = "-" if code_val is None else str(code_val)
            duration = result.get("duration")
            try:
                duration_txt = f"{float(duration):.2f}"
            except Exception:
                duration_txt = "-"
            timestamp = datetime.datetime.now().isoformat(timespec="seconds") + "Z"
            line = f"{timestamp}\t{status}\t{rel_path}\tcode={code_txt}\ttime_s={duration_txt}"
            err_msg = result.get("error")
            if err_msg:
                line += f"\tmsg={err_msg}"
            try:
                with open(str(log_path), "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except Exception:
                pass

        def _update_resolve_now_ui() -> None:
            total = int(resolve_now_state.get("total") or 0)
            done = int(resolve_now_state.get("done") or 0)
            success = int(resolve_now_state.get("success") or 0)
            pb_widget = resolve_now_state.get("progressbar")
            if pb_widget is not None:
                try:
                    pb_widget.config(mode="determinate", maximum=max(total, 1))
                    pb_widget["value"] = min(done, total)
                except Exception:
                    pass
            status_var_local = resolve_now_state.get("status_var")
            if status_var_local is not None:
                try:
                    status_text = _tr(
                        "filter_resolve_now_progress",
                        "{done}/{total} processed ({ok} solved)",
                        done=done,
                        total=total,
                        ok=success,
                    )
                except Exception:
                    status_text = f"{done}/{total} ({success})"
                try:
                    status_var_local.set(status_text)
                except Exception:
                    pass
            try:
                if total > 0:
                    summary_text = _tr(
                        "filter_resolve_now_summary",
                        "Resolving WCS: {done}/{total} processed ({ok} success).",
                        done=done,
                        total=total,
                        ok=success,
                    )
                else:
                    summary_text = _tr("filter_resolve_now_summary_idle", "Resolving WCS: idle.")
                summary_var.set(_apply_summary_hint(summary_text))
            except Exception:
                pass

        def _finalize_resolve_now_run() -> None:
            if not resolve_now_state.get("running"):
                return
            resolve_now_state["running"] = False
            executor = resolve_now_state.get("executor")
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=False)
                except Exception:
                    pass
            resolve_now_state["executor"] = None
            futures_list = resolve_now_state.get("futures")
            if isinstance(futures_list, list):
                futures_list.clear()
            btn_widget = resolve_now_state.get("button")
            if btn_widget is not None:
                try:
                    btn_widget.state(["!disabled"])
                except Exception:
                    try:
                        btn_widget.configure(state=tk.NORMAL)
                    except Exception:
                        pass
            total = int(resolve_now_state.get("total") or 0)
            success = int(resolve_now_state.get("success") or 0)
            try:
                final_msg = _tr(
                    "filter_resolve_now_done",
                    "Resolve & write WCS completed: {ok}/{total} success.",
                    ok=success,
                    total=total,
                )
            except Exception:
                final_msg = f"Resolve & write WCS completed: {success}/{total}"
            _update_resolve_now_ui()
            try:
                status_var.set(final_msg)
            except Exception:
                pass
            try:
                summary_var.set(_apply_summary_hint(final_msg))
            except Exception:
                pass
            try:
                _flush_log_buffer(force=True)
            except Exception:
                pass

        def _process_resolve_now_result(result: Dict[str, Any]) -> None:
            if not isinstance(result, dict):
                return
            resolve_now_state["done"] = int(resolve_now_state.get("done") or 0) + 1
            ok = bool(result.get("ok"))
            skipped = bool(result.get("skipped"))
            if ok and not skipped:
                resolve_now_state["success"] = int(resolve_now_state.get("success") or 0) + 1
                resolved_counter["count"] += 1
                overrides_state["resolved_wcs_count"] = resolved_counter["count"]
            idx_val = result.get("idx")
            path_val = result.get("path")
            header_obj = result.get("header")
            wcs_obj = result.get("wcs")
            if isinstance(idx_val, int) and 0 <= idx_val < len(items):
                item = items[idx_val]
                if header_obj is not None:
                    try:
                        item.header = header_obj
                        item.src["header"] = header_obj
                    except Exception:
                        pass
                if wcs_obj is not None and getattr(wcs_obj, "is_celestial", False):
                    item.wcs = wcs_obj
                    item.src["wcs"] = wcs_obj
                    try:
                        item.refresh_geometry()
                    except Exception:
                        pass
                    try:
                        _ensure_wrapped_capacity(idx_val)
                        footprint_wrapped[idx_val] = None
                        centroid_wrapped[idx_val] = None
                    except Exception:
                        pass
                    pending_visual_refresh.add(idx_val)
                    _schedule_visual_refresh_flush()
                    _canvas_draw_idle()
            name = os.path.basename(path_val) if isinstance(path_val, str) else "<unknown>"
            if ok:
                msg = _tr(
                    "filter_resolve_now_log_ok",
                    "WCS resolved for {name}.",
                    name=name,
                )
                _log_message(msg, level="INFO")
            else:
                err_msg = result.get("error") or "unknown error"
                msg = _tr(
                    "filter_resolve_now_log_fail",
                    "Failed to resolve WCS for {name}: {error}",
                    name=name,
                    error=err_msg,
                )
                _log_message(msg, level="WARN" if not skipped else "INFO")
            _resolve_now_append_log(result)
            _update_resolve_now_ui()
            total = int(resolve_now_state.get("total") or 0)
            done = int(resolve_now_state.get("done") or 0)
            if done >= total:
                _finalize_resolve_now_run()

        def _dispatch_resolve_now_future(idx: int, fut) -> None:
            def _deliver() -> None:
                try:
                    payload = fut.result()
                except Exception as exc:
                    path_guess = None
                    if isinstance(idx, int) and 0 <= idx < len(items):
                        path_guess = getattr(items[idx], "path", None)
                    payload = {
                        "idx": idx,
                        "path": path_guess,
                        "ok": False,
                        "error": str(exc),
                        "return_code": None,
                        "duration": 0.0,
                    }
                _process_resolve_now_result(payload)

            try:
                root.after(0, _deliver)
            except Exception:
                _deliver()

        def _on_pre_solve_wcs_clicked() -> None:
            if resolve_now_state.get("running"):
                return
            if not (astap_available and solve_with_astap is not None):
                warn_msg = _tr(
                    "filter_resolve_now_no_astap",
                    "ASTAP solver is not configured or unavailable.",
                )
                _log_message(warn_msg, level="WARN")
                try:
                    status_var.set(warn_msg)
                except Exception:
                    pass
                status_local = resolve_now_state.get("status_var")
                if status_local is not None:
                    try:
                        status_local.set(warn_msg)
                    except Exception:
                        pass
                return

            targets: list[tuple[int, str, Any]] = []
            for idx, item in enumerate(items):
                path_val = getattr(item, "path", None)
                if not isinstance(path_val, str) or not os.path.isfile(path_val):
                    continue
                try:
                    path_obj = Path(path_val)
                except Exception:
                    path_obj = Path(str(path_val))
                try:
                    if is_path_excluded(path_obj, EXCLUDED_DIRS):
                        continue
                except Exception:
                    continue
                header_obj = getattr(item, "header", None)
                has_item_wcs = bool(item.wcs is not None and getattr(item.wcs, "is_celestial", False))
                if has_item_wcs:
                    continue
                if header_obj is not None and _has_celestial_wcs(header_obj):
                    continue
                targets.append((idx, path_val, header_obj))

            if not targets:
                idle_msg = _tr(
                    "filter_resolve_now_idle",
                    "All listed files already include a celestial WCS.",
                )
                try:
                    summary_var.set(_apply_summary_hint(idle_msg))
                    status_var.set(idle_msg)
                except Exception:
                    pass
                status_local = resolve_now_state.get("status_var")
                if status_local is not None:
                    try:
                        status_local.set(idle_msg)
                    except Exception:
                        pass
                return

            def _coerce_float(value: Any) -> Optional[float]:
                try:
                    if value is None or value == "":
                        return None
                    return float(value)
                except Exception:
                    return None

            def _coerce_int(value: Any) -> Optional[int]:
                try:
                    if value is None or value == "":
                        return None
                    return int(float(value))
                except Exception:
                    return None

            search_radius_val = (
                combined_solver_settings.get("astap_search_radius_deg")
                if isinstance(combined_solver_settings, dict)
                else None
            )
            if search_radius_val is None:
                search_radius_val = combined_solver_settings.get("search_radius_deg") if isinstance(combined_solver_settings, dict) else None
            search_radius = _coerce_float(search_radius_val)
            if search_radius is not None and search_radius <= 0:
                search_radius = None

            downsample_candidate = None
            if isinstance(combined_solver_settings, dict):
                downsample_candidate = combined_solver_settings.get("astap_downsample")
                if downsample_candidate is None:
                    downsample_candidate = combined_solver_settings.get("downsample")
            downsample_val = _coerce_int(downsample_candidate)
            if downsample_val is not None and downsample_val < 0:
                downsample_val = None

            sensitivity_candidate = None
            if isinstance(combined_solver_settings, dict):
                sensitivity_candidate = combined_solver_settings.get("astap_sensitivity")
            sensitivity_val = _coerce_int(sensitivity_candidate)

            timeout_base = None
            timeout_astap = None
            if isinstance(combined_solver_settings, dict):
                timeout_base = _coerce_int(combined_solver_settings.get("timeout_sec"))
                if timeout_base is None:
                    timeout_base = _coerce_int(combined_solver_settings.get("timeout"))
                timeout_astap = _coerce_int(combined_solver_settings.get("astap_timeout_sec"))
            if timeout_base is None or timeout_base <= 0:
                timeout_base = 120
            timeout_base = max(5, timeout_base)
            if timeout_astap is None or timeout_astap <= 0:
                timeout_astap = max(180, timeout_base)

            resolve_now_state["running"] = True
            resolve_now_state["total"] = len(targets)
            resolve_now_state["done"] = 0
            resolve_now_state["success"] = 0
            futures_list = resolve_now_state.get("futures")
            if isinstance(futures_list, list):
                futures_list.clear()

            log_base = _resolve_now_log_base([path for _, path, _ in targets])
            if log_base is not None:
                try:
                    log_base.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                log_path = log_base / WCS_LOG_NAME
                resolve_now_state["log_dir"] = log_base
                resolve_now_state["log_path"] = log_path
                try:
                    with open(str(log_path), "a", encoding="utf-8") as handle:
                        handle.write(
                            f"# {datetime.datetime.now().isoformat(timespec='seconds')}Z Resolve & write WCS now\n"
                        )
                except Exception:
                    pass
            else:
                resolve_now_state["log_dir"] = None
                resolve_now_state["log_path"] = None

            btn_widget = resolve_now_state.get("button")
            if btn_widget is not None:
                try:
                    btn_widget.state(["disabled"])
                except Exception:
                    try:
                        btn_widget.configure(state=tk.DISABLED)
                    except Exception:
                        pass
            try:
                status_var.set(
                    _tr(
                        "filter_resolve_now_running",
                        "Resolving WCS in background…",
                    )
                )
            except Exception:
                pass
            _update_resolve_now_ui()

            max_workers = min(8, max(2, (os.cpu_count() or 4)))
            try:
                executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="FilterWCS")
            except Exception as exc:
                resolve_now_state["running"] = False
                resolve_now_state["executor"] = None
                if btn_widget is not None:
                    try:
                        btn_widget.state(["!disabled"])
                    except Exception:
                        try:
                            btn_widget.configure(state=tk.NORMAL)
                        except Exception:
                            pass
                _log_message(f"Failed to start WCS resolve threads: {exc}", level="ERROR")
                return
            resolve_now_state["executor"] = executor

            code_pattern = re.compile(r"code\\s+(-?\\d+)", re.IGNORECASE)

            def _solve_target(idx: int, path_val: str, header_obj) -> Dict[str, Any]:
                start_ts = time.monotonic()
                result: Dict[str, Any] = {
                    "idx": idx,
                    "path": path_val,
                    "ok": False,
                    "return_code": None,
                    "duration": 0.0,
                }
                header_payload = header_obj
                if header_payload is not None:
                    try:
                        header_payload = header_payload.copy()
                    except Exception:
                        pass
                if header_payload is None:
                    try:
                        header_payload = fits.getheader(path_val, 0)
                    except Exception as exc:
                        result["error"] = f"header load failed: {exc}"
                        result["duration"] = time.monotonic() - start_ts
                        return result
                if _has_celestial_wcs(header_payload):
                    result["ok"] = True
                    result["skipped"] = True
                    try:
                        result["wcs"] = WCS(header_payload, naxis=2, relax=True)
                    except Exception:
                        pass
                    result["header"] = header_payload
                    result["return_code"] = 0
                    result["duration"] = time.monotonic() - start_ts
                    return result

                progress_state = {"return_code": None}

                def _progress_cb(message, *_args) -> None:
                    if not isinstance(message, str):
                        return
                    match = code_pattern.search(message)
                    if match:
                        try:
                            progress_state["return_code"] = int(match.group(1))
                        except Exception:
                            progress_state["return_code"] = None

                try:
                    wcs_obj = solve_with_astap(
                        path_val,
                        header_payload,
                        astap_exe_path,
                        astap_data_dir,
                        search_radius_deg=search_radius,
                        downsample_factor=downsample_val,
                        sensitivity=sensitivity_val,
                        timeout_sec=timeout_astap,
                        update_original_header_in_place=True,
                        progress_callback=_progress_cb,
                    )
                except Exception as exc:
                    result["error"] = str(exc)
                    result["return_code"] = progress_state.get("return_code")
                    result["duration"] = time.monotonic() - start_ts
                    return result

                result["return_code"] = progress_state.get("return_code")
                result["duration"] = time.monotonic() - start_ts
                if wcs_obj is not None and getattr(wcs_obj, "is_celestial", False):
                    try:
                        _write_header_to_fits_local(path_val, header_payload)
                    except Exception:
                        pass
                    result["ok"] = True
                    result["wcs"] = wcs_obj
                    result["header"] = header_payload
                else:
                    result["error"] = "no solution"
                return result

            futures_collected: list[tuple[int, Any]] = []
            for idx, path_val, header_obj in targets:
                fut = executor.submit(_solve_target, idx, path_val, header_obj)
                futures_collected.append((idx, fut))
            resolve_now_state["futures"] = [f for _, f in futures_collected]
            for idx, fut in futures_collected:
                try:
                    fut.add_done_callback(lambda f, i=idx: _dispatch_resolve_now_future(i, f))
                except Exception:
                    _dispatch_resolve_now_future(idx, fut)

        def _paint_from_log_clicked() -> None:
            if wcs_async_state.get("running"):
                _log_message("Resolve running — please wait for completion before painting from log.", level="WARN")
                return

            def _normalized(path_value: str) -> str:
                try:
                    return os.path.normcase(os.path.abspath(os.path.expanduser(path_value)))
                except Exception:
                    return str(path_value)

            candidates: list[Path] = []
            log_path_candidate = wcs_async_state.get("log_path")
            if isinstance(log_path_candidate, str) and log_path_candidate:
                candidates.append(Path(log_path_candidate))
            if input_dir:
                try:
                    candidates.append(Path(input_dir) / WCS_LOG_NAME)
                except Exception:
                    pass
            for it in items:
                candidate_path = getattr(it, "path", None)
                if isinstance(candidate_path, str) and candidate_path:
                    try:
                        candidates.append(Path(candidate_path).expanduser().resolve().parent / WCS_LOG_NAME)
                    except Exception:
                        continue

            chosen_path: Optional[Path] = None
            seen: set[str] = set()
            for cand in candidates:
                try:
                    resolved = cand.expanduser()
                except Exception:
                    continue
                key = resolved.as_posix()
                if key in seen:
                    continue
                seen.add(key)
                if resolved.exists():
                    chosen_path = resolved
                    break

            if chosen_path is None and candidates:
                chosen_path = candidates[0]

            if chosen_path is None or not chosen_path.exists():
                _log_message("No WCS log file found for the current selection.", level="WARN")
                return

            wcs_async_state["log_path"] = str(chosen_path)

            try:
                with chosen_path.open("r", encoding="utf-8") as handle:
                    lines = handle.readlines()
            except Exception as exc:
                _log_message(f"Failed to read WCS log: {exc}", level="WARN")
                return

            if not lines:
                _log_message("WCS log is empty.", level="INFO")
                return

            lookup: Dict[str, int] = {}
            for idx, it in enumerate(items):
                for key in (getattr(it, "path", None), it.src.get("path_raw"), it.src.get("path_preprocessed_cache")):
                    if isinstance(key, str) and key:
                        lookup[_normalized(key)] = idx

            updated: list[int] = []
            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                if not isinstance(data, dict) or not data.get("ok"):
                    continue
                path_value = data.get("path")
                if not isinstance(path_value, str):
                    continue
                idx_target = lookup.get(_normalized(path_value))
                if idx_target is None:
                    continue
                corners = data.get("corners") or data.get("footprint")
                if not isinstance(corners, (list, tuple)) or len(corners) < 3:
                    continue
                sanitized: list[tuple[float, float]] = []
                for entry in corners:
                    try:
                        ra_val, dec_val = entry
                        sanitized.append((float(ra_val), float(dec_val)))
                    except Exception:
                        continue
                if not sanitized:
                    continue
                item = items[idx_target]
                try:
                    item.src["footprint_radec"] = sanitized
                except Exception:
                    pass
                try:
                    item.refresh_geometry()
                except Exception:
                    pass
                if idx_target not in updated:
                    updated.append(idx_target)

            if not updated:
                _log_message("WCS log did not contain matching resolved entries.", level="INFO")
                return

            pending_visual_refresh.update(updated)
            _schedule_visual_refresh_flush()
            _canvas_draw_idle()

            summary_msg = f"Painted {len(updated)} entries from log."
            summary_var.set(_apply_summary_hint(summary_msg))
            _log_message(summary_msg, level="INFO")

        if not has_explicit_centers:
            _log_message(
                _tr(
                    "filter_warn_no_displayable_wcs",
                    "Aucune information WCS/centre disponible ; l'aperçu restera vide mais vous pouvez sélectionner les fichiers." if 'fr' in str(locals().get('lang_code', 'en')).lower() else "No WCS/center information available; the sky preview will remain empty but you can still select files.",
                ),
                level="WARN",
            )

        def _progress_callback(
            msg: Any,
            progress: Any = None,
            lvl: str | None = None,
            **kwargs: Any,
        ) -> None:
            """Forward worker log messages without assuming a strict signature.

            Older worker callbacks only forwarded ``(message, progress, level)``
            while newer worker builds may pass additional keyword arguments used
            for GUI formatting.  Accepting ``**kwargs`` prevents ``TypeError``
            exceptions from background threads and keeps the logging pane
            functional.
            """

            if msg is None:
                return

            level = (lvl or "INFO")
            text = str(msg)

            detail_parts: list[str] = []
            if progress not in (None, ""):
                detail_parts.append(f"progress={progress}")
            if kwargs:
                detail_parts.extend(f"{key}={value}" for key, value in kwargs.items())

            if detail_parts:
                text = f"{text} ({', '.join(detail_parts)})"

            now = time.monotonic()
            level_upper = str(level).upper()
            if level_upper not in {"ERROR", "WARN"}:
                if now - _progress_log_state["last"] < 0.5:
                    return
            _progress_log_state["last"] = now
            _enqueue_event("log", text, level)

        def _sanitize_path(value: Any) -> str:
            """Normalize user-provided filesystem paths.

            Configuration values may include surrounding quotes, unresolved
            environment variables (e.g. ``%PROGRAMFILES%`` on Windows) or
            user-home shortcuts.  Normalising here avoids false negatives
            when checking for ASTAP availability.
            """

            try:
                if value is None:
                    return ""
                path_str = str(value).strip()
            except Exception:
                return ""

            # Drop wrapping quotes that Windows file dialogs can preserve
            if path_str.startswith(('"', "'")) and path_str.endswith(('"', "'")) and len(path_str) >= 2:
                path_str = path_str[1:-1]

            expanded = os.path.expanduser(os.path.expandvars(path_str))
            return expanded

        astap_exe_path_raw = cfg_defaults.get('astap_executable_path', '')
        astap_data_dir_raw = cfg_defaults.get('astap_data_directory_path', '')
        astap_exe_path = _sanitize_path(astap_exe_path_raw)
        astap_data_dir = _sanitize_path(astap_data_dir_raw)

        ansvr_path_candidates = [
            combined_solver_settings.get('ansvr_path'),
            combined_solver_settings.get('astrometry_local_path'),
            combined_solver_settings.get('local_ansvr_path'),
        ]
        ansvr_path = next(
            (
                _sanitize_path(candidate)
                for candidate in ansvr_path_candidates
                if isinstance(candidate, str) and candidate.strip()
            ),
            "",
        )
        if ansvr_path:
            combined_solver_settings['ansvr_path'] = ansvr_path

        astrometry_api_key = str(combined_solver_settings.get('api_key') or "").strip()

        # Keep the sanitized values in the defaults so downstream callers (log
        # messages, resolver invocations) see consistent paths.
        if astap_exe_path:
            cfg_defaults['astap_executable_path'] = astap_exe_path
        if astap_data_dir:
            cfg_defaults['astap_data_directory_path'] = astap_data_dir
        search_radius_default = cfg_defaults.get('astap_default_search_radius', 0.0)
        downsample_default = cfg_defaults.get('astap_default_downsample', 0)
        autosplit_cap_cfg = cfg_defaults.get('max_raw_per_master_tile', 0)
        try:
            autosplit_cap_cfg_int = int(autosplit_cap_cfg)
        except Exception:
            autosplit_cap_cfg_int = 0
        autosplit_cap = autosplit_cap_cfg_int if autosplit_cap_cfg_int > 0 else 50
        autosplit_cap = max(1, min(50, autosplit_cap))
        autosplit_min_cap = min(8, autosplit_cap)

        def _astap_path_available(path: str) -> bool:
            """Return True when the configured ASTAP location looks valid."""

            if not path:
                return False

            # Direct file check
            if os.path.isfile(path):
                return True

            # macOS packages are directories ending with ``.app``
            if sys.platform == "darwin" and path.lower().endswith(".app") and os.path.isdir(path):
                return True

            # Accept directories that contain the ASTAP binary
            if os.path.isdir(path):
                exe_name = "astap.exe" if os.name == "nt" else "astap"
                candidate = os.path.join(path, exe_name)
                if os.path.isfile(candidate):
                    return True

            # As a generic fallback try resolving via PATH
            resolved = shutil.which(path) or shutil.which(os.path.basename(path))
            if resolved:
                return True

            # As a generic fallback accept any existing executable entry.
            try:
                return os.path.exists(path) and os.access(path, os.X_OK)
            except Exception:
                return False

        astap_available = bool(
            solve_with_astap is not None
            and astap_astropy_available
            and _astap_path_available(astap_exe_path)
        )

        # Scrollable selection list (checkboxes for small sets, listbox for large)
        list_frame = ttk.LabelFrame(
            right,
            text=_tr(
                "filter_images_check_to_keep",
                "Images (cocher pour garder)" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Images (check to keep)",
            ),
        )
        list_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        CHECKBOX_HARD_LIMIT = 2000
        use_listbox_mode = stream_mode or total_initial_entries > CHECKBOX_HARD_LIMIT
        listbox_widget: Optional[tk.Listbox] = None
        canvas_list: Optional[tk.Canvas] = None
        inner: Optional[ttk.Frame] = None
        list_scroll_target: Optional[Any] = None

        if use_listbox_mode:
            listbox_widget = tk.Listbox(
                list_frame,
                selectmode=tk.MULTIPLE,
                exportselection=False,
                activestyle="none",
            )
            listbox_widget.grid(row=0, column=0, sticky="nsew")
            vsb = ttk.Scrollbar(list_frame, orient="vertical", command=listbox_widget.yview)
            vsb.grid(row=0, column=1, sticky="ns")
            listbox_widget.configure(yscrollcommand=vsb.set)
            list_scroll_target = listbox_widget
        else:
            canvas_list = tk.Canvas(list_frame, borderwidth=0, highlightthickness=0)
            vsb = ttk.Scrollbar(list_frame, orient="vertical", command=canvas_list.yview)
            inner = ttk.Frame(canvas_list)
            inner.bind("<Configure>", lambda e: canvas_list.configure(scrollregion=canvas_list.bbox("all")))
            canvas_list.create_window((0, 0), window=inner, anchor="nw")
            canvas_list.configure(yscrollcommand=vsb.set)
            canvas_list.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            list_scroll_target = canvas_list

        # Enable mouse-wheel scrolling over the right list (Windows/Linux/macOS)
        def _on_list_mousewheel(event):
            target = list_scroll_target
            if target is None:
                return
            try:
                if getattr(event, 'num', None) == 4:  # Linux scroll up
                    target.yview_scroll(-1, "units")
                elif getattr(event, 'num', None) == 5:  # Linux scroll down
                    target.yview_scroll(1, "units")
                else:  # Windows / macOS
                    delta = int(-1 * (event.delta / 120)) if getattr(event, 'delta', 0) else 0
                    if delta != 0:
                        target.yview_scroll(delta, "units")
            except Exception:
                pass

        def _bind_list_mousewheel(event):
            target = list_scroll_target if list_scroll_target is not None else getattr(event, "widget", None)
            if target is None:
                return
            try:
                target.bind_all("<MouseWheel>", _on_list_mousewheel)
                target.bind_all("<Button-4>", _on_list_mousewheel)
                target.bind_all("<Button-5>", _on_list_mousewheel)
            except Exception:
                pass

        def _unbind_list_mousewheel(event):
            target = list_scroll_target if list_scroll_target is not None else getattr(event, "widget", None)
            if target is None:
                return
            try:
                target.unbind_all("<MouseWheel>")
                target.unbind_all("<Button-4>")
                target.unbind_all("<Button-5>")
            except Exception:
                pass

        if list_scroll_target is not None:
            list_scroll_target.bind("<Enter>", _bind_list_mousewheel)
            list_scroll_target.bind("<Leave>", _unbind_list_mousewheel)

        selection_state: list[bool] = []
        listbox_selection_cache: set[int] = set()
        listbox_programmatic = {"active": False}

        summary_var = tk.StringVar(master=root, value="")
        sizes_details_state = {"full_sizes": "[]", "log_text": ""}

        # Early helper: may be called before footprint/draw state exists.
        def _apply_summary_hint(text: str) -> str:
            base_text = str(text or "")
            hint = _tr(
                "filter_preview_points_hint",
                "Preview uses a reduced set of footprints for performance. Zoom or filter to reveal more footprints.",
            )
            # Draw toggle may not exist yet during early startup
            try:
                wants_footprints = bool(draw_footprints_var.get())
            except Exception:
                wants_footprints = False

            # Budget/total may not be initialized yet; be safe
            budget = 0
            total = 0
            try:
                budget = int((footprint_budget_state.get("budget") or 0))  # type: ignore[name-defined]
                total = int((footprint_budget_state.get("total_items") or 0))  # type: ignore[name-defined]
            except Exception:
                budget = 0
                total = 0

            if hint and hint in base_text:
                base_text = base_text.replace(f" - {hint}", "")
                base_text = base_text.replace(hint, "").strip()
                if base_text.endswith("-"):
                    base_text = base_text[:-1].rstrip()

            if not wants_footprints or total <= 0 or budget >= total:
                try:
                    preview_hint_state["active"] = False  # type: ignore[name-defined]
                except Exception:
                    pass
                return base_text

            try:
                preview_hint_state["active"] = True  # type: ignore[name-defined]
            except Exception:
                pass
            if not base_text:
                return hint
            return f"{base_text} - {hint}"
        resolved_counter = {"count": int(overrides_state.get("resolved_wcs_count", 0) or 0)}

        if not has_explicit_centers:
            summary_var.set(
                _apply_summary_hint(
                    _tr(
                        "filter_summary_no_centers",
                        "Centres WCS indisponibles — sélection manuelle uniquement." if 'fr' in str(locals().get('lang_code', 'en')).lower() else "WCS centers unavailable — manual selection only.",
                    )
                )
            )

        # Aggregate controls container (ensures persistent visibility)
        controls = ttk.Frame(right)
        controls.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        controls.columnconfigure(0, weight=1)

        operations = ttk.Frame(controls)
        operations.pack(fill=tk.X, expand=False)
        operations.columnconfigure(2, weight=1)
        operations.columnconfigure(3, weight=0)

        # Build operations area defensively: never let a single error hide the panel
        draw_footprints_var = tk.BooleanVar(master=root, value=True)
        write_wcs_var = tk.BooleanVar(master=root, value=True)
        coverage_first_var = tk.BooleanVar(master=root, value=True)
        overcap_percent_var = tk.IntVar(master=root, value=10)

        WCS_LOG_NAME = "zemosaic_wcs.log"
        DEFER_DRAW_CHUNK = 200
        LIVE_DRAW_THROTTLE = 20

        defer_overlay_var = tk.BooleanVar(master=root, value=True)

        wcs_async_state: Dict[str, Any] = {
            "executor": None,
            "pending": [],
            "ui_queue": queue.Queue(),
            "buffer": [],
            "live_buffer": [],
            "log_path": None,
            "running": False,
            "stop": False,
            "done": 0,
            "total": 0,
            "success": 0,
            "poll_job": None,
        }

        resolve_btn = None
        auto_btn = None
        footprints_chk = None
        write_wcs_chk = None
        details_btn = None
        try:
            resolve_btn = ttk.Button(
                operations,
                text=_tr("filter_btn_resolve_wcs", "Resolve missing WCS"),
            )
            resolve_btn.grid(row=0, column=0, padx=4, pady=2, sticky="w")
        except Exception as e:
            _log_message(f"[FilterUI] Resolve button init failed: {e}", level="WARN")
        try:
            auto_btn = ttk.Button(
                operations,
                text=_tr("filter_btn_auto_group", "Auto-organize Master Tiles"),
            )
            auto_btn.grid(row=0, column=1, padx=4, pady=2, sticky="w")
        except Exception as e:
            _log_message(f"[FilterUI] Auto-group button init failed: {e}", level="WARN")

        try:
            ttk.Label(
                operations,
                textvariable=summary_var,
                anchor="w",
                justify="left",
                wraplength=260,
            ).grid(row=0, column=2, padx=4, pady=2, sticky="w")
        except Exception as e:
            _log_message(f"[FilterUI] Summary label failed: {e}", level="WARN")

        try:
            ttk.Checkbutton(
                operations,
                text="Defer overlay (faster, no freeze)",
                variable=defer_overlay_var,
            ).grid(row=1, column=0, columnspan=2, padx=4, pady=2, sticky="w")
        except Exception as e:
            _log_message(f"[FilterUI] Defer overlay checkbox failed: {e}", level="WARN")

        try:
            ttk.Button(
                operations,
                text="Paint from log",
                command=lambda: _paint_from_log_clicked(),
            ).grid(row=1, column=2, padx=4, pady=2, sticky="w")
        except Exception as e:
            _log_message(f"[FilterUI] Paint-from-log button failed: {e}", level="WARN")

        def _show_sizes_details():
            try:
                win = tk.Toplevel(root)
                win.title("Master-tile sizes")
                win.geometry("420x300")
                import tkinter.scrolledtext as st

                box = st.ScrolledText(win, wrap="word")
                box.pack(fill="both", expand=True)
                text = sizes_details_state.get("full_sizes") or sizes_details_state.get("log_text") or ""
                try:
                    box.insert("1.0", text)
                except Exception:
                    box.insert("1.0", str(text))
                box.configure(state="disabled")
                copy_btn = ttk.Button(
                    win,
                    text="Copy",
                    command=lambda: (win.clipboard_clear(), win.clipboard_append(text)),
                )
                copy_btn.pack(pady=4)
            except Exception as _exc:
                _log_message(f"[FilterUI] Failed to open sizes popup: {_exc}", level="WARN")

        try:
            details_btn = ttk.Button(operations, text="Details…", command=_show_sizes_details)
            details_btn.grid(row=0, column=3, padx=4, sticky="ne")
            details_btn.grid_remove()
        except Exception as e:
            _log_message(f"[FilterUI] Details button failed: {e}", level="WARN")
            details_btn = None

        try:
            coverage_chk = ttk.Checkbutton(
                operations,
                text=_tr(
                    "ui_coverage_first",
                    "Coverage-first clustering (may exceed Max raws/tile)",
                ),
                variable=coverage_first_var,
            )
            coverage_chk.grid(row=2, column=0, columnspan=3, padx=4, pady=(6, 0), sticky="w")
        except Exception as e:
            _log_message(f"[FilterUI] Coverage-first checkbox failed: {e}", level="WARN")

        overcap_spin = None
        try:
            overcap_label = ttk.Label(
                operations,
                text=_tr("ui_overcap_allowance_pct", "Over-cap allowance (%)"),
            )
            overcap_label.grid(row=3, column=0, padx=4, pady=(2, 0), sticky="w")
            overcap_spin = ttk.Spinbox(
                operations,
                from_=0,
                to=50,
                increment=5,
                textvariable=overcap_percent_var,
                width=5,
                justify="center",
            )
            overcap_spin.grid(row=3, column=1, padx=4, pady=(2, 0), sticky="w")
        except Exception as e:
            _log_message(f"[FilterUI] Over-cap spinbox failed: {e}", level="WARN")

        try:
            footprints_chk = ttk.Checkbutton(
                operations,
                text=_tr("filter_chk_draw_footprints", "Draw WCS footprints"),
                variable=draw_footprints_var,
            )
            footprints_chk.grid(row=1, column=0, columnspan=2, padx=4, pady=(2, 0), sticky="w")
        except Exception as e:
            _log_message(f"[FilterUI] Footprints checkbox failed: {e}", level="WARN")

        try:
            write_wcs_chk = ttk.Checkbutton(
                operations,
                text=_tr("filter_chk_write_wcs", "Write WCS to file"),
                variable=write_wcs_var,
            )
            write_wcs_chk.grid(row=1, column=2, padx=4, pady=(2, 0), sticky="w")
        except Exception as e:
            _log_message(f"[FilterUI] Write-WCS checkbox failed: {e}", level="WARN")

        try:
            astrometry_solver_available = bool(solve_with_astrometry is not None and astrometry_api_key)
            ansvr_solver_available = bool(solve_with_ansvr is not None and ansvr_path)
            any_solver_available = astap_available or astrometry_solver_available or ansvr_solver_available

            if not any_solver_available and resolve_btn is not None:
                resolve_btn.state(["disabled"])
                try:
                    if write_wcs_chk is not None:
                        write_wcs_chk.state(["disabled"])
                except Exception:
                    pass
                _log_message(
                    _tr(
                        "filter_warn_no_solvers",
                        "No WCS solver configured (ASTAP/Astrometry/ANSVR).",
                    ),
                    level="WARN",
                )

            if not (cluster_func and autosplit_func) and auto_btn is not None:
                auto_btn.state(["disabled"])
                if worker_import_failure_summary:
                    _log_message(
                        _tr(
                            "filter_warn_worker_missing",
                            "Auto-grouping disabled because the processing worker module failed to load: {error}",
                            error=worker_import_failure_summary,
                        ),
                        level="ERROR",
                    )
        except Exception as e:
            _log_message(f"[FilterUI] Operations availability checks failed: {e}", level="WARN")

        def _resolve_missing_wcs_inplace() -> None:
            if resolve_state["running"]:
                return

            astap_enabled = bool(astap_available and solve_with_astap is not None)
            astrometry_enabled = bool(solve_with_astrometry is not None and astrometry_api_key)
            ansvr_enabled = bool(solve_with_ansvr is not None and ansvr_path)

            if not (astap_enabled or astrometry_enabled or ansvr_enabled):
                _log_message(
                    _tr(
                        "filter_warn_no_solvers",
                        "No WCS solver configured (ASTAP/Astrometry/ANSVR).",
                    ),
                    level="WARN",
                )
                return

            pending = [
                (idx, item)
                for idx, item in enumerate(items)
                if item.wcs is None and isinstance(item.path, str) and os.path.isfile(item.path)
            ]
            if not pending:
                msg = _tr("filter_log_no_missing_wcs", "All listed files already include a WCS solution.")
                summary_var.set(_apply_summary_hint(msg))
                _log_message(msg, level="INFO")
                return

            write_inplace = bool(write_wcs_var.get())
            resolve_state["running"] = True
            resolve_btn.state(["disabled"])
            summary_msg = _tr(
                "filter_log_resolving_wide",
                "Resolving missing WCS (ASTAP/Astrometry/ANSVR)…",
            )
            summary_var.set(_apply_summary_hint(summary_msg))
            _log_message(summary_msg, level="INFO")

            footprints_restore_state["value"] = bool(draw_footprints_var.get())
            footprints_restore_state["needs_restore"] = True
            indices_cache = footprints_restore_state.get("indices")
            if isinstance(indices_cache, set):
                indices_cache.clear()
            else:
                footprints_restore_state["indices"] = set()
            try:
                draw_footprints_var.set(False)
            except Exception:
                pass
            if footprints_restore_state["value"]:
                pending_visual_refresh.update(range(len(items)))
                _schedule_visual_refresh_flush()
            try:
                footprints_chk.state(["disabled"])
            except Exception:
                try:
                    footprints_chk.configure(state=tk.DISABLED)
                except Exception:
                    pass

            solver_settings_local: Dict[str, Any] = dict(combined_solver_settings)
            if astrometry_api_key:
                solver_settings_local["api_key"] = astrometry_api_key
            if ansvr_path:
                solver_settings_local["ansvr_path"] = ansvr_path
            solver_settings_local["update_original_header_in_place"] = write_inplace

            def _coerce_float(val: Any) -> Optional[float]:
                try:
                    if val is None:
                        return None
                    value = float(val)
                except Exception:
                    return None
                return value

            def _coerce_int(val: Any) -> Optional[int]:
                try:
                    if val is None or val == "":
                        return None
                    value = int(float(val))
                except Exception:
                    return None
                return value

            search_radius_val = solver_settings_local.get(
                "astap_search_radius_deg",
                cfg_defaults.get("astap_default_search_radius"),
            )
            srch_radius = _coerce_float(search_radius_val)
            if srch_radius is not None and srch_radius <= 0:
                srch_radius = None
            if srch_radius is not None:
                solver_settings_local["astap_search_radius_deg"] = srch_radius

            downsample_candidate = solver_settings_local.get("astap_downsample")
            if downsample_candidate is None:
                downsample_candidate = solver_settings_local.get("downsample")
            if downsample_candidate is None:
                downsample_candidate = downsample_default
            downsample_val = _coerce_int(downsample_candidate)
            if downsample_val is not None and downsample_val < 0:
                downsample_val = None
            if downsample_val is not None:
                solver_settings_local["downsample"] = downsample_val

            sensitivity_candidate = solver_settings_local.get(
                "astap_sensitivity",
                cfg_defaults.get("astap_default_sensitivity"),
            )
            sensitivity_val = _coerce_int(sensitivity_candidate)
            if sensitivity_val is not None:
                solver_settings_local["astap_sensitivity"] = sensitivity_val

            timeout_candidate = solver_settings_local.get("timeout")
            if timeout_candidate in (None, ""):
                timeout_candidate = solver_settings_local.get("ansvr_timeout")
            timeout_val = _coerce_int(timeout_candidate)
            if timeout_val is None or timeout_val <= 0:
                timeout_val = 60
            timeout_sec = max(5, timeout_val)
            solver_settings_local["timeout"] = timeout_sec
            solver_settings_local["ansvr_timeout"] = timeout_sec

            # ASTAP handling in the worker always allows up to 180 seconds before
            # timing out. Mirror that behaviour here so the filter UI matches the
            # main pipeline resilience on slow solves.
            astap_timeout_sec = max(180, timeout_sec)

            astrometry_direct = getattr(astrometry_mod, "solve_with_astrometry_net", None) if astrometry_mod else None
            ansvr_direct = getattr(astrometry_mod, "solve_with_ansvr", None) if astrometry_mod else None

            def _log_solver_event(key: str, default: str, level: str, **fmt: Any) -> None:
                try:
                    message = _tr(key, default, **fmt)
                except Exception:
                    try:
                        message = default.format(**fmt)
                    except Exception:
                        message = default
                _enqueue_event("log", message, level)

            def _solve_one_image_blocking(idx: int, item: Any) -> Dict[str, Any]:
                result: Dict[str, Any] = {
                    "idx": idx,
                    "path": getattr(item, "path", None),
                    "ok": False,
                }
                if wcs_async_state.get("stop"):
                    result["error"] = "cancelled"
                    return result

                path = getattr(item, "path", None)
                if not isinstance(path, str) or not os.path.isfile(path):
                    result["error"] = "missing file"
                    return result

                file_name = os.path.basename(path)
                header_obj = getattr(item, "header", None)

                def _record_failure(error_message: Optional[str]) -> Dict[str, Any]:
                    if error_message:
                        result["error"] = str(error_message)
                    move_info = _move_problematic_file(path)
                    if move_info.get("moved"):
                        result["moved"] = True
                        result["moved_path"] = move_info.get("destination")
                        result["move_filename"] = move_info.get("filename")
                    if move_info.get("error"):
                        result["move_error"] = move_info.get("error")
                    return result

                try:
                    if header_obj is not None:
                        _enqueue_event("header_loaded", idx, header_obj)
                    elif astap_fits_module is not None and astap_astropy_available:
                        try:
                            with astap_fits_module.open(path) as hdul_hdr:
                                header_obj = hdul_hdr[0].header
                        except Exception as exc:
                            _log_solver_event(
                                "filter_log_solver_failed",
                                "Failed to load FITS header for {name} ({err}).",
                                "ERROR",
                                solver="Header",
                                name=file_name,
                                err=exc,
                            )
                        else:
                            if header_obj is not None:
                                _enqueue_event("header_loaded", idx, header_obj)
                    if header_obj is None:
                        try:
                            header_obj = fits.getheader(path, 0)
                        except Exception as exc:
                            _log_solver_event(
                                "filter_log_solver_failed",
                                "Failed to load FITS header for {name} ({err}).",
                                "ERROR",
                                solver="Header",
                                name=file_name,
                                err=exc,
                            )
                            return _record_failure(str(exc))
                        else:
                            if header_obj is not None:
                                _enqueue_event("header_loaded", idx, header_obj)

                    def _shape_from_header() -> Optional[tuple[int, int]]:
                        shape_hw = getattr(item, "shape", None)
                        if shape_hw is not None:
                            return shape_hw
                        if header_obj is None:
                            return None
                        try:
                            nax1 = header_obj.get("NAXIS1")
                            nax2 = header_obj.get("NAXIS2")
                            if nax1 and nax2:
                                w_px = int(float(nax1))
                                h_px = int(float(nax2))
                                if w_px > 0 and h_px > 0:
                                    return (h_px, w_px)
                        except Exception:
                            return getattr(item, "shape", None)
                        return getattr(item, "shape", None)

                    def _on_success(solver_name: str, wcs_obj: Any) -> Dict[str, Any]:
                        _log_solver_event(
                            "filter_log_solver_ok",
                            "{solver} solved {name}.",
                            "INFO",
                            solver=solver_name,
                            name=file_name,
                        )
                        result["ok"] = True
                        result["solver"] = solver_name
                        result["header"] = header_obj
                        result["wcs"] = wcs_obj
                        footprint_pts = None
                        try:
                            footprint_pts = _compute_footprint_from_wcs(wcs_obj, _shape_from_header())
                        except Exception:
                            footprint_pts = None
                        if footprint_pts:
                            result["corners"] = footprint_pts
                        return result

                    try:
                        existing = WCS(header_obj, naxis=2, relax=True)
                        if existing is not None and getattr(existing, "is_celestial", False):
                            return _on_success("Header", existing)
                    except Exception:
                        pass

                    last_error: Optional[str] = None

                    if wcs_async_state.get("stop"):
                        result["error"] = "cancelled"
                        return result

                    if astap_enabled:
                        _log_solver_event(
                            "filter_log_solver_attempt",
                            "Trying {solver} for {name}…",
                            "INFO",
                            solver="ASTAP",
                            name=file_name,
                        )
                        try:
                            astap_wcs = solve_with_astap(
                                path,
                                header_obj,
                                astap_exe_path,
                                astap_data_dir,
                                search_radius_deg=srch_radius,
                                downsample_factor=downsample_val,
                                sensitivity=sensitivity_val,
                                timeout_sec=astap_timeout_sec,
                                update_original_header_in_place=write_inplace,
                                progress_callback=_progress_callback,
                            )
                        except Exception as exc:
                            last_error = str(exc)
                            _log_solver_event(
                                "filter_log_solver_failed",
                                "{solver} failed for {name} ({err}).",
                                "ERROR",
                                solver="ASTAP",
                                name=file_name,
                                err=exc,
                            )
                        else:
                            if astap_wcs and getattr(astap_wcs, "is_celestial", False):
                                return _on_success("ASTAP", astap_wcs)
                            _log_solver_event(
                                "filter_log_solver_failed",
                                "{solver} failed for {name} ({err}).",
                                "WARN",
                                solver="ASTAP",
                                name=file_name,
                                err="no solution",
                            )

                    if wcs_async_state.get("stop"):
                        result["error"] = "cancelled"
                        return result

                    if astrometry_enabled:
                        _log_solver_event(
                            "filter_log_solver_attempt",
                            "Trying {solver} for {name}…",
                            "INFO",
                            solver="Astrometry.net",
                            name=file_name,
                        )
                        skip_failure = False
                        try:
                            if write_inplace and solve_with_astrometry:
                                astrometry_wcs = solve_with_astrometry(
                                    path,
                                    header_obj,
                                    solver_settings_local,
                                    progress_callback=_progress_callback,
                                )
                            elif not write_inplace and astrometry_direct:
                                astrometry_wcs = astrometry_direct(
                                    path,
                                    header_obj,
                                    api_key=solver_settings_local.get("api_key", ""),
                                    timeout_sec=timeout_sec,
                                    downsample_factor=solver_settings_local.get("downsample"),
                                    update_original_header_in_place=False,
                                    progress_callback=_progress_callback,
                                )
                            else:
                                astrometry_wcs = None
                                if not write_inplace:
                                    _log_solver_event(
                                        "filter_log_solver_failed",
                                        "{solver} failed for {name} ({err}).",
                                        "INFO",
                                        solver="Astrometry.net",
                                        name=file_name,
                                        err="disabled (write-off)",
                                    )
                                    skip_failure = True
                        except Exception as exc:
                            last_error = str(exc)
                            _log_solver_event(
                                "filter_log_solver_failed",
                                "{solver} failed for {name} ({err}).",
                                "ERROR",
                                solver="Astrometry.net",
                                name=file_name,
                                err=exc,
                            )
                        else:
                            if astrometry_wcs and getattr(astrometry_wcs, "is_celestial", False):
                                return _on_success("Astrometry.net", astrometry_wcs)
                            if not skip_failure:
                                _log_solver_event(
                                    "filter_log_solver_failed",
                                    "{solver} failed for {name} ({err}).",
                                    "WARN",
                                    solver="Astrometry.net",
                                    name=file_name,
                                    err="no solution",
                                )

                    if wcs_async_state.get("stop"):
                        result["error"] = "cancelled"
                        return result

                    if ansvr_enabled:
                        _log_solver_event(
                            "filter_log_solver_attempt",
                            "Trying {solver} for {name}…",
                            "INFO",
                            solver="ANSVR",
                            name=file_name,
                        )
                        skip_failure = False
                        try:
                            if write_inplace and solve_with_ansvr:
                                ansvr_wcs = solve_with_ansvr(
                                    path,
                                    header_obj,
                                    solver_settings_local,
                                    progress_callback=_progress_callback,
                                )
                            elif not write_inplace and ansvr_direct:
                                ansvr_wcs = ansvr_direct(
                                    path,
                                    header_obj,
                                    ansvr_config_path=solver_settings_local.get("ansvr_path", ""),
                                    timeout_sec=timeout_sec,
                                    update_original_header_in_place=False,
                                    progress_callback=_progress_callback,
                                )
                            else:
                                ansvr_wcs = None
                                if not write_inplace:
                                    _log_solver_event(
                                        "filter_log_solver_failed",
                                        "{solver} failed for {name} ({err}).",
                                        "INFO",
                                        solver="ANSVR",
                                        name=file_name,
                                        err="disabled (write-off)",
                                    )
                                    skip_failure = True
                        except Exception as exc:
                            last_error = str(exc)
                            _log_solver_event(
                                "filter_log_solver_failed",
                                "{solver} failed for {name} ({err}).",
                                "ERROR",
                                solver="ANSVR",
                                name=file_name,
                                err=exc,
                            )
                        else:
                            if ansvr_wcs and getattr(ansvr_wcs, "is_celestial", False):
                                return _on_success("ANSVR", ansvr_wcs)
                            if not skip_failure:
                                _log_solver_event(
                                    "filter_log_solver_failed",
                                    "{solver} failed for {name} ({err}).",
                                    "WARN",
                                    solver="ANSVR",
                                    name=file_name,
                                    err="no solution",
                                )

                    if wcs_async_state.get("stop"):
                        result["error"] = "cancelled"
                        return result

                    if last_error:
                        result["error"] = last_error
                    else:
                        result["error"] = "no solution"
                    return _record_failure(result.get("error"))
                except Exception as exc:
                    _enqueue_event("log", f"Unexpected resolver error: {exc}", "ERROR")
                    return _record_failure(str(exc))

            try:
                _cancel_wcs_executor()
                queue_obj = wcs_async_state.get("ui_queue")
                if isinstance(queue_obj, queue.Queue):
                    try:
                        while True:
                            queue_obj.get_nowait()
                    except queue.Empty:
                        pass
                    except Exception:
                        pass
                buffer_list = wcs_async_state.get("buffer")
                if isinstance(buffer_list, list):
                    buffer_list.clear()
                live_list = wcs_async_state.get("live_buffer")
                if isinstance(live_list, list):
                    live_list.clear()
                wcs_async_state["done"] = 0
                wcs_async_state["success"] = 0
                wcs_async_state["total"] = len(pending)
                wcs_async_state["stop"] = False
                wcs_async_state["running"] = True

                log_dir: Optional[Path] = None
                for _idx, _item in pending:
                    candidate_path = getattr(_item, "path", None)
                    if isinstance(candidate_path, str) and candidate_path:
                        try:
                            log_dir = Path(candidate_path).expanduser().resolve().parent
                        except Exception:
                            log_dir = Path(os.path.dirname(candidate_path))
                        break
                if log_dir is None and input_dir:
                    try:
                        log_dir = Path(input_dir).expanduser().resolve()
                    except Exception:
                        log_dir = Path(input_dir)

                log_path: Optional[Path] = None
                if log_dir is not None:
                    log_path = log_dir / WCS_LOG_NAME
                    wcs_async_state["log_path"] = str(log_path)
                    try:
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    try:
                        log_path.write_text("", encoding="utf-8")
                    except Exception:
                        pass
                else:
                    wcs_async_state["log_path"] = None

                max_workers = max(1, min(4, os.cpu_count() or 2))
                executor = ThreadPoolExecutor(max_workers=max_workers)
                wcs_async_state["executor"] = executor
                pending_futures: list[Any] = []
                wcs_async_state["pending"] = pending_futures

                for idx, item in pending:
                    if wcs_async_state.get("stop"):
                        break
                    future = executor.submit(_solve_one_image_blocking, idx, item)
                    pending_futures.append(future)

                    def _on_done(fut: Any, *, _idx: int = idx, _item: Any = item) -> None:
                        try:
                            res = fut.result()
                        except Exception as exc:
                            _enqueue_event("log", f"Unexpected resolver error: {exc}", "ERROR")
                            res = {
                                "idx": _idx,
                                "path": getattr(_item, "path", None),
                                "ok": False,
                                "error": str(exc),
                            }
                        try:
                            wcs_async_state["ui_queue"].put_nowait(res)
                        except Exception:
                            pass

                    future.add_done_callback(_on_done)

                try:
                    wcs_async_state["poll_job"] = root.after(100, _poll_wcs_queue)
                except Exception:
                    wcs_async_state["poll_job"] = None
                    _poll_wcs_queue()

            except Exception as exc:
                wcs_async_state["running"] = False
                resolve_state["running"] = False
                try:
                    resolve_btn.state(["!disabled"])
                except Exception:
                    pass
                if footprints_restore_state.get("needs_restore"):
                    original_value = bool(footprints_restore_state.get("value", True))
                    try:
                        draw_footprints_var.set(original_value)
                    except Exception:
                        pass
                    if original_value:
                        pending_visual_refresh.update(range(len(items)))
                    if pending_visual_refresh:
                        _schedule_visual_refresh_flush()
                    try:
                        footprints_chk.state(["!disabled"])
                    except Exception:
                        try:
                            footprints_chk.configure(state=tk.NORMAL)
                        except Exception:
                            pass
                    footprints_restore_state["needs_restore"] = False
                    indices_cache = footprints_restore_state.get("indices")
                    if isinstance(indices_cache, set):
                        indices_cache.clear()
                _log_message(f"[FilterUI] Unable to start WCS executor: {exc}", level="ERROR")
                return

        # Coverage-first auto-organization state and helpers
        auto_cluster_state = {"running": False}

        def _tr_safe(key: str, default: str, **kwargs: Any) -> str:
            try:
                return _tr(key, default, **kwargs)
            except Exception:
                try:
                    return default.format(**kwargs)
                except Exception:
                    return default

        def _auto_organize_master_tiles() -> None:
            if not (cluster_func and autosplit_func):
                return
            if auto_cluster_state.get("running"):
                return
            try:
                auto_btn.state(["disabled"])
            except Exception:
                pass

            try:
                selected_indices = list(_get_selected_indices())
            except Exception:
                selected_indices = []

            if not selected_indices:
                summary_text = _tr_safe(
                    "filter_log_groups_summary",
                    "Prepared {g} group(s), sizes: {sizes}.",
                    g=0,
                    sizes="[]",
                )
                summary_var.set(_apply_summary_hint(summary_text))
                _log_message(summary_text, level="INFO")
                sizes_details_state["full_sizes"] = "[]"
                sizes_details_state["log_text"] = summary_text
                if details_btn is not None:
                    try:
                        details_btn.grid_remove()
                    except Exception:
                        pass
                try:
                    auto_btn.state(["!disabled"])
                except Exception:
                    pass
                return

            try:
                overcap_pct_value = int(overcap_percent_var.get())
            except Exception:
                overcap_pct_value = 10
            overcap_pct_value = max(0, min(50, overcap_pct_value))
            coverage_enabled_flag = bool(coverage_first_var.get())

            auto_cluster_state["running"] = True

            def _finalize_ui() -> None:
                auto_cluster_state["running"] = False
                try:
                    auto_btn.state(["!disabled"])
                except Exception:
                    pass

            def _notify_no_groups(level: str) -> None:
                summary_text_local = _tr_safe(
                    "filter_log_groups_summary",
                    "Prepared {g} group(s), sizes: {sizes}.",
                    g=0,
                    sizes="[]",
                )
                summary_var.set(_apply_summary_hint(summary_text_local))
                _log_message(summary_text_local, level=level)
                sizes_details_state["full_sizes"] = "[]"
                sizes_details_state["log_text"] = summary_text_local
                if details_btn is not None:
                    try:
                        details_btn.grid_remove()
                    except Exception:
                        pass
                _finalize_ui()

            def _handle_error(message: str) -> None:
                _log_message(message, level="ERROR")
                _finalize_ui()

            def _apply_result(result: Dict[str, Any]) -> None:
                try:
                    final_groups = result.get("final_groups") or []
                    if isinstance(overrides_state, dict):
                        overrides_state["preplan_master_groups"] = final_groups
                        try:
                            overrides_state.pop("autosplit_cap", None)
                        except Exception:
                            pass
                    sizes = result.get("sizes")
                    if not isinstance(sizes, list):
                        sizes = [len(gr) for gr in final_groups]
                    full_sizes_str = ", ".join(map(str, sizes)) if sizes else "[]"
                    summary_text_for_log = _tr_safe(
                        "filter_log_groups_summary",
                        "Prepared {g} group(s), sizes: {sizes}.",
                        g=len(final_groups),
                        sizes=full_sizes_str,
                    )
                    _log_message(summary_text_for_log, level="INFO")
                    sizes_details_state["full_sizes"] = full_sizes_str
                    sizes_details_state["log_text"] = summary_text_for_log

                    hist_str = _format_sizes_histogram(sizes)
                    summary_text_compact = _tr_safe(
                        "filter_log_groups_summary",
                        "Prepared {g} group(s), sizes: {sizes}.",
                        g=len(final_groups),
                        sizes=hist_str,
                    )
                    summary_var.set(_apply_summary_hint(summary_text_compact))
                    if details_btn is not None:
                        try:
                            if len(sizes) > 60:
                                details_btn.grid()
                            else:
                                details_btn.grid_remove()
                        except Exception:
                            pass
                    if result.get("coverage_first"):
                        done_msg = _tr_safe(
                            "log_covfirst_done",
                            "Coverage-first preplan ready: {N} groups written to overrides_state.preplan_master_groups",
                            N=len(final_groups),
                        )
                        _log_message(done_msg, level="INFO")
                    pending_outline_state["groups"] = final_groups if final_groups else None
                    if not resolve_state.get("running"):
                        try:
                            _draw_group_outlines(final_groups)
                        except Exception:
                            pass
                except Exception as exc:
                    _log_message(f"[FilterUI] Failed to apply clustering result: {exc}", level="ERROR")
                finally:
                    _finalize_ui()

            def _worker(selected_snapshot: list[int], overcap_pct: int, coverage_enabled: bool) -> None:
                try:
                    class _FallbackWCS:
                        is_celestial = True

                        def __init__(self, center_coord: SkyCoord):
                            self._center = center_coord
                            self.pixel_shape = (1, 1)
                            self.array_shape = (1, 1)

                            class _Inner:
                                def __init__(self, center: SkyCoord):
                                    self.crval = (
                                        float(center.ra.to(u.deg).value),
                                        float(center.dec.to(u.deg).value),
                                    )
                                    self.crpix = (0.5, 0.5)

                            self.wcs = _Inner(center_coord)

                        def pixel_to_world(self, _x: float, _y: float):
                            return self._center

                    candidate_infos: list[dict] = []
                    coord_samples: list[tuple[float, float]] = []
                    for idx in selected_snapshot:
                        if idx < 0 or idx >= len(items):
                            continue
                        item = items[idx]
                        entry = dict(item.src)
                        if "path" not in entry:
                            entry["path"] = item.path
                        if "path_raw" not in entry and item.src.get("path_raw"):
                            entry["path_raw"] = item.src.get("path_raw")
                        if "header" not in entry:
                            entry["header"] = item.header
                        if item.shape and "shape" not in entry:
                            entry["shape"] = item.shape
                        center_obj = item.center or item.phase0_center
                        if center_obj is None and extract_center_from_header_fn and item.header is not None:
                            try:
                                center_obj = extract_center_from_header_fn(item.header)
                            except Exception:
                                center_obj = None
                        if item.wcs is None and center_obj is not None:
                            entry["wcs"] = _FallbackWCS(center_obj)
                            entry["_fallback_wcs_used"] = True
                            entry.setdefault("phase0_center", center_obj)
                            entry.setdefault("center", center_obj)
                        elif item.wcs is not None:
                            entry["wcs"] = item.wcs
                            if center_obj is not None:
                                entry.setdefault("center", center_obj)
                        if center_obj is not None:
                            try:
                                coord_samples.append(
                                    (
                                        float(center_obj.ra.to(u.deg).value),
                                        float(center_obj.dec.to(u.deg).value),
                                    )
                                )
                                entry.setdefault("RA", coord_samples[-1][0])
                                entry.setdefault("DEC", coord_samples[-1][1])
                            except Exception:
                                pass
                        candidate_infos.append(entry)

                    if not candidate_infos:
                        root.after(0, lambda: _notify_no_groups("WARN"))
                        return

                    if coord_samples and compute_dispersion_func:
                        try:
                            dispersion_deg = float(compute_dispersion_func(coord_samples))
                        except Exception:
                            dispersion_deg = 0.0
                    else:
                        dispersion_deg = 0.0

                    if dispersion_deg <= 0.12:
                        threshold_heuristic = 0.10
                    elif dispersion_deg <= 0.30:
                        threshold_heuristic = 0.15
                    else:
                        threshold_heuristic = 0.05 if dispersion_deg <= 0.60 else 0.20
                    threshold_heuristic = min(0.20, max(0.08, threshold_heuristic))

                    threshold_override = None
                    threshold_candidates = [
                        combined_solver_settings.get("panel_clustering_threshold_deg"),
                        combined_solver_settings.get("cluster_panel_threshold"),
                    ]
                    if isinstance(overrides_state, dict):
                        threshold_candidates.extend(
                            [
                                overrides_state.get("panel_clustering_threshold_deg"),
                                overrides_state.get("cluster_panel_threshold"),
                            ]
                        )
                    for candidate in threshold_candidates:
                        try:
                            val = float(candidate)
                        except Exception:
                            continue
                        if val > 0:
                            threshold_override = val
                            break

                    threshold_initial = float(threshold_override) if threshold_override and threshold_override > 0 else float(threshold_heuristic)
                    if threshold_initial <= 0:
                        threshold_initial = float(threshold_heuristic) if threshold_heuristic > 0 else 0.10

                    orientation_threshold = 0.0
                    orientation_candidates = [combined_solver_settings.get("cluster_orientation_split_deg")]
                    if isinstance(overrides_state, dict):
                        orientation_candidates.append(overrides_state.get("cluster_orientation_split_deg"))
                    for candidate in orientation_candidates:
                        try:
                            val = float(candidate)
                        except Exception:
                            continue
                        if val > 0:
                            orientation_threshold = val
                            break

                    cap_effective = int(autosplit_cap)
                    cap_candidates = [combined_solver_settings.get("max_raw_per_master_tile")]
                    if isinstance(overrides_state, dict):
                        cap_candidates.append(overrides_state.get("max_raw_per_master_tile"))
                    for candidate in cap_candidates:
                        try:
                            val = int(candidate)
                        except Exception:
                            continue
                        if val > 0:
                            cap_effective = max(1, min(50, val))
                            break

                    min_cap_effective = int(autosplit_min_cap)
                    min_candidates = [combined_solver_settings.get("autosplit_min_cap")]
                    if isinstance(overrides_state, dict):
                        min_candidates.append(overrides_state.get("autosplit_min_cap"))
                    for candidate in min_candidates:
                        try:
                            val = int(candidate)
                        except Exception:
                            continue
                        if val > 0:
                            min_cap_effective = max(1, min(val, cap_effective))
                            break
                    min_cap_effective = max(1, min(min_cap_effective, cap_effective))

                    if coverage_enabled:
                        start_msg = _tr_safe(
                            "log_covfirst_start",
                            "Coverage-first clustering: start (threshold={TH} deg, cap={CAP}, min_cap={MIN})",
                            TH=f"{threshold_initial:.3f}",
                            CAP=int(cap_effective),
                            MIN=int(min_cap_effective),
                        )
                        _log_async(start_msg, "INFO")

                    threshold_for_cluster = float(threshold_initial)
                    groups_initial = cluster_func(
                        candidate_infos,
                        threshold_for_cluster,
                        _progress_callback,
                        orientation_split_threshold_deg=float(orientation_threshold),
                    )
                    if not groups_initial:
                        root.after(0, lambda: _notify_no_groups("WARN"))
                        return

                    groups_used = groups_initial
                    threshold_used = threshold_for_cluster
                    candidate_count = len(candidate_infos)
                    ratio = (len(groups_initial) / float(candidate_count)) if candidate_count else 0.0
                    pathological = candidate_count > 0 and (len(groups_initial) >= candidate_count or ratio >= 0.6)

                    if coverage_enabled and pathological and len(coord_samples) >= 5:
                        p90_value = None
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
                        if p90_value and np.isfinite(p90_value):
                            threshold_relaxed = max(threshold_for_cluster, float(p90_value) * 1.1)
                            if threshold_relaxed > threshold_for_cluster * 1.001:
                                groups_relaxed = cluster_func(
                                    candidate_infos,
                                    threshold_relaxed,
                                    _progress_callback,
                                    orientation_split_threshold_deg=float(orientation_threshold),
                                )
                                if groups_relaxed and len(groups_relaxed) < len(groups_initial):
                                    relax_msg = _tr_safe(
                                        "log_covfirst_relax",
                                        "Relaxed epsilon using P90 NN: old={OLD} deg -> new={NEW} deg",
                                        OLD=f"{threshold_for_cluster:.3f}",
                                        NEW=f"{threshold_relaxed:.3f}",
                                    )
                                    _log_async(relax_msg, "INFO")
                                    groups_used = groups_relaxed
                                    threshold_used = threshold_relaxed

                    groups_after_autosplit = autosplit_func(
                        groups_used,
                        cap=int(cap_effective),
                        min_cap=int(min_cap_effective),
                        progress_callback=_progress_callback,
                    )
                    if coverage_enabled:
                        autosplit_msg = _tr_safe(
                            "log_covfirst_autosplit",
                            "Autosplit applied: cap={CAP}, min_cap={MIN}, groups_in={IN}, groups_out={OUT}",
                            CAP=int(cap_effective),
                            MIN=int(min_cap_effective),
                            IN=len(groups_used),
                            OUT=len(groups_after_autosplit),
                        )
                        _log_async(autosplit_msg, "INFO")

                    cap_with_allowance = max(int(cap_effective), int(cap_effective * (1 + overcap_pct / 100.0)))
                    max_dispersion_limit = None
                    if compute_dispersion_func and threshold_used > 0:
                        max_dispersion_limit = float(threshold_used) * 1.05

                    merge_log_fn = (lambda msg: _log_async(msg, "DEBUG_DETAIL")) if coverage_enabled else None
                    final_groups = _merge_small_groups(
                        groups_after_autosplit,
                        min_size=int(min_cap_effective),
                        cap=int(cap_effective),
                        cap_allowance=cap_with_allowance,
                        compute_dispersion=compute_dispersion_func if compute_dispersion_func else None,
                        max_dispersion_deg=max_dispersion_limit,
                        log_fn=merge_log_fn,
                    )
                    if coverage_enabled:
                        merge_msg = _tr_safe(
                            "log_covfirst_merge",
                            "Merged small groups with over-cap allowance={ALLOW}%, final_groups={N}",
                            ALLOW=int(overcap_pct),
                            N=len(final_groups),
                        )
                        _log_async(merge_msg, "INFO")

                    for grp in final_groups:
                        for info in grp:
                            if info.pop("_fallback_wcs_used", False):
                                info.pop("wcs", None)

                    result_payload = {
                        "final_groups": final_groups,
                        "sizes": [len(gr) for gr in final_groups],
                        "coverage_first": coverage_enabled,
                        "threshold_used": threshold_used,
                    }
                    root.after(0, lambda res=result_payload: _apply_result(res))
                except Exception as exc:
                    error_msg = f"[FilterUI] Coverage-first clustering failed: {exc}"
                    root.after(0, lambda msg=error_msg: _handle_error(msg))

            try:
                threading.Thread(
                    target=_worker,
                    args=(selected_indices, overcap_pct_value, coverage_enabled_flag),
                    daemon=True,
                ).start()
            except Exception as exc:
                _handle_error(f"[FilterUI] Unable to start coverage-first clustering: {exc}")

        try:
            if resolve_btn is not None:
                resolve_btn.configure(command=_resolve_missing_wcs_inplace)
        except Exception as e:
            _log_message(f"[FilterUI] Resolve button hook failed: {e}", level="WARN")
        try:
            if auto_btn is not None:
                auto_btn.configure(command=_auto_organize_master_tiles)
        except Exception as e:
            _log_message(f"[FilterUI] Auto-group button hook failed: {e}", level="WARN")

        # If initial overrides already include preplanned groups, draw them now
        try:
            preplanned = overrides_state.get("preplan_master_groups") if isinstance(overrides_state, dict) else None
            if preplanned:
                _draw_group_outlines(preplanned)
        except Exception:
            pass

        # Selection helpers
        actions = ttk.Frame(controls)
        try:
            actions.pack(fill=tk.X, expand=False, pady=5)
        except Exception:
            actions.pack(fill=tk.X, expand=False)
        def select_all():
            if use_listbox_mode:
                if selection_state:
                    listbox_programmatic["active"] = True
                    try:
                        selection_state[:] = [True] * len(selection_state)
                        listbox_selection_cache.clear()
                        listbox_selection_cache.update(range(len(selection_state)))
                        if listbox_widget is not None:
                            listbox_widget.selection_set(0, tk.END)
                    finally:
                        listbox_programmatic["active"] = False
                else:
                    pass
            else:
                for v in check_vars:
                    v.set(True)
            update_visuals()

        def select_none():
            if use_listbox_mode:
                if selection_state:
                    listbox_programmatic["active"] = True
                    try:
                        selection_state[:] = [False] * len(selection_state)
                        listbox_selection_cache.clear()
                        if listbox_widget is not None:
                            listbox_widget.selection_clear(0, tk.END)
                    finally:
                        listbox_programmatic["active"] = False
            else:
                for v in check_vars:
                    v.set(False)
            update_visuals()
        ttk.Button(
            actions,
            text=_tr(
                "filter_select_all",
                "Tout sélectionner" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Select all",
            ),
            command=select_all,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            actions,
            text=_tr(
                "filter_select_none",
                "Tout désélectionner" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Deselect all",
            ),
            command=select_none,
        ).pack(side=tk.LEFT, padx=4)

        # Confirm/cancel buttons
        bottom = ttk.Frame(controls)
        try:
            bottom.pack(fill=tk.X, expand=False)
        except Exception:
            bottom.pack(fill=tk.X)
        try:
            right.update_idletasks()
            _dbg("Right panel constructed; awaiting controls visibility check…")
        except Exception:
            pass

        def _ensure_controls_visible_later():
            try:
                # If controls area didn't map yet (themes/geometry lag), give
                # those rows a larger minimum size and try again shortly.
                if not bottom.winfo_ismapped() or bottom.winfo_height() < 5:
                    try:
                        right.rowconfigure(3, minsize=180)
                    except Exception:
                        pass
                    try:
                        _dbg("Controls not visible yet; increasing row minsize and retrying…")
                    except Exception:
                        pass
                    root.after(150, _ensure_controls_visible_later)
                else:
                    try:
                        _dbg(f"Controls visible: operations={operations.winfo_height()} actions={actions.winfo_height()} bottom={bottom.winfo_height()}")
                    except Exception:
                        pass
            except Exception:
                pass

        try:
            root.after(200, _ensure_controls_visible_later)
        except Exception as e:
            _log_message(f"[FilterUI] Scheduling visibility check failed: {e}", level="WARN")

        # Keep controls row visible across resizes or heavy redraws (ASTAP/zoom)
        def _enforce_right_grid(_event: Any | None = None) -> None:
            try:
                # Ensure the scrollable list (row=2) is the only stretchable row
                right.rowconfigure(0, weight=0)
                right.rowconfigure(1, weight=0)
                right.rowconfigure(2, weight=1)
                # Maintain a sane minimum height for the bottom controls
                desired = 0
                try:
                    desired = int(operations.winfo_reqheight()) + int(actions.winfo_reqheight()) + int(bottom.winfo_reqheight()) + 12
                except Exception:
                    desired = 180
                right.rowconfigure(3, weight=0, minsize=max(160, desired))
            except Exception:
                # Last resort: at least ensure row 3 doesn't collapse fully
                try:
                    right.rowconfigure(3, minsize=180)
                except Exception:
                    pass

        try:
            right.bind("<Configure>", _enforce_right_grid)
            root.after_idle(_enforce_right_grid)
        except Exception:
            pass
        result: dict[str, Any] = {
            "accepted": None,
            "selected_indices": None,
            "overrides": None,
            "cancelled": False,
        }

        def _cancel_overrides_payload() -> dict[str, Any]:
            payload: dict[str, Any] = {}
            if isinstance(overrides_state, dict) and overrides_state:
                payload.update(overrides_state)
            current = result.get("overrides")
            if isinstance(current, dict) and current:
                payload.update(current)
            payload["filter_cancelled"] = True
            return payload

        def on_validate():
            _drain_stream_queue_non_blocking(mark_done=True)
            sel = _get_selected_indices()
            result["accepted"] = True
            result["selected_indices"] = sel
            result["overrides"] = overrides_state if overrides_state else None
            result["cancelled"] = False
            # If running as a Toplevel, do not quit the main GUI loop
            if root_is_toplevel:
                try:
                    root.destroy()
                except Exception:
                    pass
            else:
                try:
                    root.quit()
                except Exception:
                    pass
                try:
                    root.destroy()
                except Exception:
                    pass
        def on_cancel():
            _drain_stream_queue_non_blocking(mark_done=True)
            result["accepted"] = False
            result["selected_indices"] = None
            result["cancelled"] = True
            result["overrides"] = _cancel_overrides_payload()
            if root_is_toplevel:
                try:
                    root.destroy()
                except Exception:
                    pass
            else:
                try:
                    root.quit()
                except Exception:
                    pass
                try:
                    root.destroy()
                except Exception:
                    pass
        resolve_now_frame = ttk.Frame(bottom)
        try:
            resolve_now_frame.pack(side=tk.LEFT, padx=4, pady=4)
        except Exception:
            resolve_now_frame.pack(side=tk.LEFT, padx=4)
        resolve_now_status = tk.StringVar(master=root, value="")
        resolve_now_state["status_var"] = resolve_now_status
        try:
            resolve_now_btn_widget = ttk.Button(
                resolve_now_frame,
                text=_tr("filter_resolve_now_button", "Resolve & write WCS now"),
                command=_on_pre_solve_wcs_clicked,
            )
            resolve_now_btn_widget.pack(fill=tk.X, expand=False)
            resolve_now_state["button"] = resolve_now_btn_widget
        except Exception:
            resolve_now_state["button"] = None
        try:
            resolve_now_progress = ttk.Progressbar(resolve_now_frame, mode="determinate", length=160)
            resolve_now_progress.pack(fill=tk.X, expand=False, pady=(4, 0))
            resolve_now_state["progressbar"] = resolve_now_progress
        except Exception:
            resolve_now_state["progressbar"] = None
        try:
            ttk.Label(resolve_now_frame, textvariable=resolve_now_status).pack(fill=tk.X, expand=False, pady=(2, 0))
        except Exception:
            pass

        ttk.Button(
            bottom,
            text=_tr(
                "filter_validate",
                "Valider" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Validate",
            ),
            command=on_validate,
        ).pack(side=tk.RIGHT, padx=4)
        ttk.Button(
            bottom,
            text=_tr(
                "filter_cancel",
                "Annuler" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Cancel",
            ),
            command=on_cancel,
        ).pack(side=tk.RIGHT, padx=4)

        # Populate checkboxes and draw footprints (supports streaming additions)
        check_vars: list[tk.BooleanVar] = []
        checkbuttons: list[Any] = []
        item_labels: list[str] = []
        footprint_wrapped: list[Optional[np.ndarray]] = []
        centroid_wrapped: list[Optional[tuple[float, float]]] = []
        footprints_state: Dict[str, Any] = {
            "collection": None,
            "segments": [],
            "colors": [],
            "indices": [],
        }
        centroids_state: Dict[str, Any] = {
            "collection": None,
            "offsets": [],
            "colors": [],
            "indices": [],
            "sizes": [],
        }
        footprint_budget_state = {"budget": 0, "total_items": 0}
        preview_hint_state = {"active": False}
        visual_build_state = {"indices": [], "cursor": 0, "job": None, "full": True, "footprints_used": 0}

        def _ensure_visual_collections() -> None:
            if footprints_state["collection"] is None:
                lc = LineCollection([], linewidths=1.0, colors=[], alpha=1.0)
                lc.set_picker(True)
                try:
                    lc.set_pickradius(5)
                except Exception:
                    pass
                ax.add_collection(lc)
                footprints_state["collection"] = lc
            if centroids_state["collection"] is None:
                sc = ax.scatter([], [], s=20, picker=True, alpha=0.9, c="tab:blue")
                sc.set_offsets(np.empty((0, 2)))
                sc.set_facecolors(np.empty((0, 4)))
                sc.set_edgecolors(np.empty((0, 4)))
                centroids_state["collection"] = sc

        def _clear_visual_datasets() -> None:
            footprints_state["segments"] = []
            footprints_state["colors"] = []
            footprints_state["indices"] = []
            centroids_state["offsets"] = []
            centroids_state["colors"] = []
            centroids_state["indices"] = []
            centroids_state["sizes"] = []

        def _update_line_collection() -> None:
            _ensure_visual_collections()
            lc: Optional[LineCollection] = footprints_state.get("collection")
            if lc is None:
                return
            lc.set_segments(footprints_state["segments"])
            if footprints_state["colors"]:
                lc.set_colors(footprints_state["colors"])
            else:
                lc.set_colors([])

        def _update_centroid_collection() -> None:
            _ensure_visual_collections()
            coll = centroids_state.get("collection")
            if coll is None:
                return
            offsets = centroids_state["offsets"]
            if offsets:
                coll.set_offsets(np.asarray(offsets, dtype=float))
            else:
                coll.set_offsets(np.empty((0, 2)))
            colors = centroids_state["colors"]
            sizes = centroids_state["sizes"]
            if colors:
                coll.set_facecolors(colors)
                coll.set_edgecolors(colors)
            else:
                coll.set_facecolors(np.empty((0, 4)))
                coll.set_edgecolors(np.empty((0, 4)))
            if sizes:
                coll.set_sizes(np.asarray(sizes, dtype=float))
            else:
                coll.set_sizes(np.empty((0,), dtype=float))

        def _recompute_footprint_budget() -> int:
            cap = _compute_dynamic_footprint_budget(
                len(items),
                preview_cap,
                max_footprints=MAX_FOOTPRINTS,
            )
            footprint_budget_state["budget"] = cap
            footprint_budget_state["total_items"] = len(items)
            try:
                current_summary = summary_var.get()
            except Exception:
                current_summary = ""
            try:
                summary_var.set(_apply_summary_hint(str(current_summary)))
            except Exception:
                pass
            return footprint_budget_state["budget"]

        VISUAL_CHUNK = 300

        def _ensure_wrapped_capacity(idx: int) -> None:
            while idx >= len(footprint_wrapped):
                footprint_wrapped.append(None)
            while idx >= len(centroid_wrapped):
                centroid_wrapped.append(None)

        def _prepare_visual_payload(idx: int) -> None:
            _ensure_wrapped_capacity(idx)
            item = items[idx]
            fp = item.get_cached_footprint()
            wrapped_fp: Optional[np.ndarray] = None
            if fp is not None and len(fp) >= 3:
                try:
                    ra_vals = [wrap_ra_deg(float(v), ref_ra) for v in fp[:, 0].tolist()]
                    dec_vals = [float(v) for v in fp[:, 1].tolist()]
                    coords = np.column_stack([np.asarray(ra_vals, dtype=float), np.asarray(dec_vals, dtype=float)])
                    wrapped_fp = coords
                except Exception:
                    wrapped_fp = None
            footprint_wrapped[idx] = wrapped_fp
            center_val: Optional[tuple[float, float]] = None
            if item.center is not None:
                try:
                    ra_c = wrap_ra_deg(float(item.center.ra.to(u.deg).value), ref_ra)
                    dec_c = float(item.center.dec.to(u.deg).value)
                    center_val = (ra_c, dec_c)
                except Exception:
                    center_val = None
            centroid_wrapped[idx] = center_val

        def _append_visual_entry(idx: int) -> None:
            selected = _is_selected(idx)
            color = to_rgba("tab:blue" if selected else "0.7", 0.9 if selected else 0.3)
            draw_fp = _should_draw_footprints()
            footprint_budget = footprint_budget_state["budget"]
            wrapped_fp = footprint_wrapped[idx]
            if (
                draw_fp
                and wrapped_fp is not None
                and visual_build_state["footprints_used"] < footprint_budget
            ):
                try:
                    closed = np.vstack([wrapped_fp, wrapped_fp[0]])
                except Exception:
                    closed = None
                if closed is not None:
                    footprints_state["segments"].append(closed)
                    footprints_state["colors"].append(color)
                    footprints_state["indices"].append(idx)
                    visual_build_state["footprints_used"] += 1
                    return
            center_val = centroid_wrapped[idx]
            if center_val is not None:
                centroids_state["offsets"].append(center_val)
                centroids_state["colors"].append(color)
                centroids_state["indices"].append(idx)
                centroids_state["sizes"].append(28.0 if selected else 18.0)

        def _process_visual_build_chunk() -> None:
            visual_build_state["job"] = None
            if not visual_build_state["indices"]:
                return
            start = visual_build_state["cursor"]
            end = min(len(visual_build_state["indices"]), start + VISUAL_CHUNK)
            if start == 0 and visual_build_state.get("full", True):
                _clear_visual_datasets()
                visual_build_state["footprints_used"] = 0
                _recompute_footprint_budget()
            for local_idx in range(start, end):
                idx = visual_build_state["indices"][local_idx]
                if idx < 0 or idx >= len(items):
                    continue
                _prepare_visual_payload(idx)
                _append_visual_entry(idx)
            _update_line_collection()
            _update_centroid_collection()
            if end < len(visual_build_state["indices"]):
                visual_build_state["cursor"] = end
                try:
                    visual_build_state["job"] = root.after(15, _process_visual_build_chunk)
                except Exception:
                    _process_visual_build_chunk()
            else:
                visual_build_state["indices"] = []
                visual_build_state["cursor"] = 0
                visual_build_state["full"] = True
                visual_build_state["footprints_used"] = 0
                _schedule_axes_update()

        def _schedule_visual_build(indices: Optional[Iterable[int]] = None, *, full: bool = False) -> None:
            if indices is None or full:
                target = list(range(len(items)))
                full = True
            else:
                uniq: list[int] = []
                seen: set[int] = set()
                for raw in indices:
                    try:
                        idx = int(raw)
                    except Exception:
                        continue
                    if idx < 0 or idx >= len(items) or idx in seen:
                        continue
                    uniq.append(idx)
                    seen.add(idx)
                target = uniq
                if len(target) >= len(items):
                    full = True
                    target = list(range(len(items)))
            visual_build_state["indices"] = target
            visual_build_state["cursor"] = 0
            visual_build_state["full"] = bool(full)
            visual_build_state["footprints_used"] = 0
            if visual_build_state["job"] is None:
                try:
                    visual_build_state["job"] = root.after(10, _process_visual_build_chunk)
                except Exception:
                    _process_visual_build_chunk()
        # Master Tile outlines (group-level overlays)
        group_outline_state: Dict[str, Any] = {"collection": None, "segments": []}

        def _clear_group_outlines() -> None:
            coll = group_outline_state.get("collection")
            if coll is not None:
                try:
                    coll.set_segments([])
                except Exception:
                    try:
                        coll.remove()
                    except Exception:
                        pass
                    group_outline_state["collection"] = None
            group_outline_state["segments"] = []

        def _draw_group_outlines(groups_payload: Optional[list[list[dict]]]) -> None:
            """Draw red Master Tile contours using celestial coordinates."""

            pending_outline_state["groups"] = groups_payload if groups_payload else None

            if not groups_payload:
                _clear_group_outlines()
                try:
                    ax.relim()
                except Exception:
                    pass
                try:
                    ax.autoscale_view()
                except Exception:
                    pass
                try:
                    canvas.draw_idle()
                except Exception:
                    pass
                return

            segments: list[np.ndarray] = []
            try:
                def _wcs_corners_deg(wcs_obj: Any) -> Optional[np.ndarray]:
                    try:
                        ny: Optional[int] = None
                        nx: Optional[int] = None
                        if getattr(wcs_obj, "array_shape", None):
                            ny, nx = wcs_obj.array_shape  # type: ignore[attr-defined]
                        elif getattr(wcs_obj, "pixel_shape", None):
                            nx, ny = wcs_obj.pixel_shape  # type: ignore[attr-defined]
                        if ny is None or nx is None:
                            return None
                        px = np.array(
                            [[0.0, 0.0], [float(nx), 0.0], [float(nx), float(ny)], [0.0, float(ny)]],
                            dtype=float,
                        )
                        sky = pixel_to_skycoord(px[:, 0], px[:, 1], wcs_obj)
                        ra = np.array(sky.ra.deg, dtype=float)
                        dec = np.array(sky.dec.deg, dtype=float)
                        if ra.size != 4 or dec.size != 4:
                            return None
                        return np.column_stack((ra, dec))
                    except Exception:
                        return None

                def _extend_from_points(
                    points: Any,
                    widths_acc: list[float],
                    heights_acc: list[float],
                    ra_acc: list[float],
                    dec_acc: list[float],
                ) -> None:
                    try:
                        arr = np.asarray(points, dtype=float)
                    except Exception:
                        return
                    if arr.ndim != 2 or arr.shape[1] < 2:
                        return
                    ra_vals = arr[:, 0]
                    dec_vals = arr[:, 1]
                    if ra_vals.size == 0 or dec_vals.size == 0:
                        return
                    mask = np.isfinite(ra_vals) & np.isfinite(dec_vals)
                    if not np.any(mask):
                        return
                    ra_vals = ra_vals[mask]
                    dec_vals = dec_vals[mask]
                    ra_acc.extend(float(v) for v in ra_vals.tolist())
                    dec_acc.extend(float(v) for v in dec_vals.tolist())

                    ref_ra_local = float(np.nanmedian(ra_vals))
                    ref_dec_local = float(np.nanmedian(dec_vals))
                    if not np.isfinite(ref_ra_local) or not np.isfinite(ref_dec_local):
                        return
                    cos_local = float(np.cos(np.deg2rad(ref_dec_local))) if np.isfinite(ref_dec_local) else 1.0
                    if abs(cos_local) < 1e-6:
                        cos_local = 1e-6 if cos_local >= 0 else -1e-6

                    try:
                        x_vals = [
                            (wrap_ra_deg(float(rv), ref_ra_local) - ref_ra_local) * cos_local for rv in ra_vals
                        ]
                        y_vals = [float(dv) - ref_dec_local for dv in dec_vals]
                    except Exception:
                        return

                    if x_vals:
                        span_x = float(np.nanmax(x_vals) - np.nanmin(x_vals))
                        if np.isfinite(span_x) and span_x > 0:
                            widths_acc.append(span_x)
                    if y_vals:
                        span_y = float(np.nanmax(y_vals) - np.nanmin(y_vals))
                        if np.isfinite(span_y) and span_y > 0:
                            heights_acc.append(span_y)

                for grp in groups_payload:
                    centers_wrapped: list[float] = []
                    centers_dec: list[float] = []
                    widths: list[float] = []
                    heights: list[float] = []
                    corner_ra_samples: list[float] = []
                    corner_dec_samples: list[float] = []

                    for info in (grp or []):
                        wcs_candidates = []
                        idx_mapped = None
                        path_val = (
                            (info.get("path") or info.get("path_raw") or info.get("path_preprocessed_cache"))
                            if isinstance(info, dict)
                            else None
                        )
                        if path_val:
                            key = _path_key(path_val)
                            idx_mapped = known_path_index.get(key)
                        if idx_mapped is not None and 0 <= idx_mapped < len(items) and getattr(items[idx_mapped], "wcs", None) is not None:
                            wcs_candidates.append(items[idx_mapped].wcs)
                        if isinstance(info, dict) and info.get("wcs") is not None:
                            wcs_candidates.append(info["wcs"])
                        for wcs_obj in wcs_candidates:
                            w_deg, h_deg = footprint_wh_deg(wcs_obj)
                            if np.isfinite(w_deg) and np.isfinite(h_deg) and w_deg > 0 and h_deg > 0:
                                widths.append(w_deg)
                                heights.append(h_deg)
                            corners = _wcs_corners_deg(wcs_obj)
                            if corners is not None:
                                _extend_from_points(
                                    corners,
                                    widths,
                                    heights,
                                    corner_ra_samples,
                                    corner_dec_samples,
                                )

                        ra_c = None
                        dec_c = None
                        if idx_mapped is not None and 0 <= idx_mapped < len(items) and items[idx_mapped].center is not None:
                            try:
                                ra_c = float(items[idx_mapped].center.ra.to(u.deg).value)
                                dec_c = float(items[idx_mapped].center.dec.to(u.deg).value)
                            except Exception:
                                ra_c = None
                                dec_c = None
                        if (ra_c is None or dec_c is None) and isinstance(info, dict):
                            c = info.get("center")
                            try:
                                if c is not None and hasattr(c, "ra") and hasattr(c, "dec"):
                                    ra_c = float(c.ra.to(u.deg).value)
                                    dec_c = float(c.dec.to(u.deg).value)
                                elif isinstance(c, (list, tuple)) and len(c) >= 2:
                                    ra_c = float(c[0])
                                    dec_c = float(c[1])
                                elif isinstance(c, dict):
                                    ra_v = c.get("ra") or c.get("RA")
                                    dec_v = c.get("dec") or c.get("DEC")
                                    if ra_v is not None and dec_v is not None:
                                        ra_c = float(ra_v)
                                        dec_c = float(dec_v)
                            except Exception:
                                ra_c = None
                                dec_c = None
                        if (ra_c is None or dec_c is None) and isinstance(info, dict):
                            try:
                                ra_v = info.get("RA")
                                dec_v = info.get("DEC")
                                if ra_v is not None and dec_v is not None:
                                    ra_c = float(ra_v)
                                    dec_c = float(dec_v)
                            except Exception:
                                ra_c = None
                                dec_c = None

                        if ra_c is not None and dec_c is not None:
                            centers_wrapped.append(wrap_ra_deg(ra_c, ref_ra))
                            centers_dec.append(float(dec_c))

                        footprint_points: Optional[Any] = None
                        if idx_mapped is not None and 0 <= idx_mapped < len(items):
                            try:
                                fp_cached = items[idx_mapped].get_cached_footprint()
                                footprint_points = fp_cached
                            except Exception:
                                footprint_points = None
                        if footprint_points is None and isinstance(info, dict):
                            footprint_points = info.get("footprint_radec") or info.get("_precomp_fp")
                            if footprint_points is None:
                                vals: list[tuple[float, float]] = []
                                try:
                                    for k in ("1", "2", "3", "4"):
                                        ra_val = info.get(f"FP_RA{k}")
                                        dec_val = info.get(f"FP_DEC{k}")
                                        if ra_val is None or dec_val is None:
                                            continue
                                        if str(ra_val) == "" or str(dec_val) == "":
                                            continue
                                        vals.append((float(ra_val), float(dec_val)))
                                except Exception:
                                    vals = []
                                if vals:
                                    footprint_points = vals
                        if footprint_points is not None:
                            _extend_from_points(
                                footprint_points,
                                widths,
                                heights,
                                corner_ra_samples,
                                corner_dec_samples,
                            )

                    if not centers_wrapped:
                        continue

                    tile_w = float(np.median(widths)) if widths else 0.2
                    tile_h = float(np.median(heights)) if heights else 0.2

                    grp_ra = float(np.median(centers_wrapped))
                    grp_dec = float(np.median(centers_dec))

                    cos_dec = float(np.cos(np.deg2rad(grp_dec))) if np.isfinite(grp_dec) else 1.0
                    if abs(cos_dec) < 1e-6:
                        cos_dec = 1e-6 if cos_dec >= 0 else -1e-6

                    if corner_ra_samples and corner_dec_samples:
                        try:
                            x_offsets: list[float] = []
                            y_offsets: list[float] = []
                            for ra_val, dec_val in zip(corner_ra_samples, corner_dec_samples):
                                delta_ra = wrap_ra_deg(float(ra_val), grp_ra) - grp_ra
                                x_offsets.append(delta_ra * cos_dec)
                                y_offsets.append(float(dec_val) - grp_dec)
                            if x_offsets and y_offsets:
                                x_min = float(np.nanmin(x_offsets))
                                x_max = float(np.nanmax(x_offsets))
                                y_min = float(np.nanmin(y_offsets))
                                y_max = float(np.nanmax(y_offsets))
                                if np.isfinite(x_min) and np.isfinite(x_max) and np.isfinite(y_min) and np.isfinite(y_max):
                                    x_center = 0.5 * (x_min + x_max)
                                    y_center = 0.5 * (y_min + y_max)
                                    if abs(x_center) > 1e-9:
                                        grp_ra = wrap_ra_deg(grp_ra + (x_center / cos_dec), ref_ra)
                                    if abs(y_center) > 1e-9:
                                        grp_dec = grp_dec + y_center
                                    cos_dec = float(np.cos(np.deg2rad(grp_dec))) if np.isfinite(grp_dec) else 1.0
                                    if abs(cos_dec) < 1e-6:
                                        cos_dec = 1e-6 if cos_dec >= 0 else -1e-6
                                    x_offsets = []
                                    y_offsets = []
                                    for ra_val, dec_val in zip(corner_ra_samples, corner_dec_samples):
                                        delta_ra = wrap_ra_deg(float(ra_val), grp_ra) - grp_ra
                                        x_offsets.append(delta_ra * cos_dec)
                                        y_offsets.append(float(dec_val) - grp_dec)
                                    x_span = float(np.nanmax(x_offsets) - np.nanmin(x_offsets)) if x_offsets else 0.0
                                    y_span = float(np.nanmax(y_offsets) - np.nanmin(y_offsets)) if y_offsets else 0.0
                                    if np.isfinite(x_span) and x_span > 0:
                                        tile_w = max(tile_w, x_span)
                                    if np.isfinite(y_span) and y_span > 0:
                                        tile_h = max(tile_h, y_span)
                        except Exception:
                            pass

                    dx = tile_w / 2.0 / cos_dec
                    dy = tile_h / 2.0
                    ra_corners = [
                        wrap_ra_deg(grp_ra - dx, ref_ra),
                        wrap_ra_deg(grp_ra + dx, ref_ra),
                        wrap_ra_deg(grp_ra + dx, ref_ra),
                        wrap_ra_deg(grp_ra - dx, ref_ra),
                    ]
                    dec_corners = [grp_dec - dy, grp_dec - dy, grp_dec + dy, grp_dec + dy]
                    rect_pts = list(zip(ra_corners, dec_corners))
                    rect = np.array(rect_pts + [rect_pts[0]], dtype=float)
                    segments.append(rect)

                group_outline_state["segments"] = segments
                coll = group_outline_state.get("collection")
                if coll is None:
                    coll = LineCollection([], linewidths=1.6, colors=["red"], linestyle="--", alpha=0.9, zorder=5)
                    ax.add_collection(coll)
                    group_outline_state["collection"] = coll
                coll.set_segments(segments)
                if segments:
                    coll.set_colors([to_rgba("red", 0.9) for _ in segments])
                coll.set_linestyle("--")
                coll.set_linewidth(1.6)

            except Exception:
                _clear_group_outlines()
            finally:
                try:
                    ax.relim()
                except Exception:
                    pass
                try:
                    ax.autoscale_view()
                except Exception:
                    pass
                try:
                    canvas.draw_idle()
                except Exception:
                    pass
        known_path_index: dict[str, int] = {}

        def _is_selected(idx: int) -> bool:
            if idx < 0:
                return False
            if use_listbox_mode:
                if idx < len(selection_state):
                    return bool(selection_state[idx])
                return False
            if idx < len(check_vars):
                try:
                    return bool(check_vars[idx].get())
                except Exception:
                    return False
            return False

        def _set_selected(idx: int, value: bool, *, sync_widget: bool = True) -> None:
            val = bool(value)
            if use_listbox_mode:
                if idx < 0:
                    return
                while idx >= len(selection_state):
                    selection_state.append(True)
                    listbox_selection_cache.add(len(selection_state) - 1)
                selection_state[idx] = val
                if not sync_widget:
                    if val:
                        listbox_selection_cache.add(idx)
                    else:
                        listbox_selection_cache.discard(idx)
                    return
                if listbox_widget is None:
                    if val:
                        listbox_selection_cache.add(idx)
                    else:
                        listbox_selection_cache.discard(idx)
                    return
                listbox_programmatic["active"] = True
                try:
                    if val:
                        listbox_widget.selection_set(idx)
                        listbox_selection_cache.add(idx)
                    else:
                        listbox_widget.selection_clear(idx)
                        listbox_selection_cache.discard(idx)
                except Exception:
                    pass
                finally:
                    listbox_programmatic["active"] = False
            else:
                if 0 <= idx < len(check_vars):
                    try:
                        check_vars[idx].set(val)
                    except Exception:
                        pass

        def _get_selected_indices() -> list[int]:
            if use_listbox_mode:
                return [i for i, flag in enumerate(selection_state) if flag]
            return [i for i, var in enumerate(check_vars) if var.get()]

        def _update_item_label(idx: int, text: str) -> None:
            if idx < 0 or idx >= len(item_labels):
                return
            item_labels[idx] = text
            if use_listbox_mode:
                if listbox_widget is None:
                    return
                listbox_programmatic["active"] = True
                try:
                    listbox_widget.delete(idx)
                    listbox_widget.insert(idx, text)
                    if _is_selected(idx):
                        listbox_widget.selection_set(idx)
                        listbox_selection_cache.add(idx)
                    else:
                        listbox_widget.selection_clear(idx)
                        listbox_selection_cache.discard(idx)
                except Exception:
                    pass
                finally:
                    listbox_programmatic["active"] = False
            else:
                if 0 <= idx < len(checkbuttons):
                    try:
                        checkbuttons[idx].configure(text=text)
                    except Exception:
                        pass

        def _path_key(value: Any) -> str:
            try:
                if isinstance(value, str) and value:
                    return os.path.normcase(os.path.abspath(value))
            except Exception:
                pass
            try:
                if isinstance(value, str) and value:
                    return os.path.normcase(value)
            except Exception:
                pass
            return str(value) if value is not None else ""

        def _should_draw_footprints() -> bool:
            try:
                if not bool(draw_footprints_var.get()):
                    return False
            except Exception:
                return False
            return footprint_budget_state.get("budget", 0) > 0

        # (moved earlier — see early definition near summary_var)

        axes_update_pending = {"pending": False}

        # Define axes update helpers early so any earlier population calls can use them safely
        def _recompute_axes_limits() -> None:
            ra_vals: list[float] = []
            dec_vals: list[float] = []
            for it in items:
                footprint = it.get_cached_footprint()
                if footprint is not None:
                    for ra in footprint[:, 0].tolist():
                        ra_vals.append(wrap_ra_deg(float(ra), ref_ra))
                    dec_vals.extend(footprint[:, 1].tolist())
                elif it.center is not None:
                    ra_vals.append(wrap_ra_deg(float(it.center.ra.to(u.deg).value), ref_ra))
                    dec_vals.append(float(it.center.dec.to(u.deg).value))
            if ra_vals and dec_vals:
                ra_min, ra_max = min(ra_vals), max(ra_vals)
                dec_min, dec_max = min(dec_vals), max(dec_vals)
                ra_pad = max(1e-3, (ra_max - ra_min) * 0.05 + 0.2)
                dec_pad = max(1e-3, (dec_max - dec_min) * 0.05 + 0.2)
                ax.set_xlim(ra_max + ra_pad, ra_min - ra_pad)
                ax.set_ylim(dec_min - dec_pad, dec_max + dec_pad)
                try:
                    canvas.draw_idle()
                except Exception:
                    pass

        def _schedule_axes_update() -> None:
            if axes_update_pending["pending"]:
                return

            def _do_update() -> None:
                axes_update_pending["pending"] = False
                _recompute_axes_limits()

            axes_update_pending["pending"] = True
            try:
                root.after_idle(_do_update)
            except Exception:
                _do_update()
        population_state = {
            "next_index": 0,
            "total": total_initial_entries,
            "finalized": False,
        }
        populate_indicator = {"running": False}

        def _update_population_indicator(processed: int, *, done: bool = False) -> None:
            if stream_mode:
                return
            if done:
                try:
                    pb.stop()
                except Exception:
                    pass
                populate_indicator["running"] = False
                status_var.set(_tr("filter_status_ready", "Crawling done."))
                return
            if population_state["total"] <= 0:
                return
            if not populate_indicator["running"]:
                try:
                    pb.start(80)
                except Exception:
                    pass
                populate_indicator["running"] = True
            status_var.set(
                _tr(
                    "filter_status_populating",
                    "Preparing list… {current}/{total}",
                ).format(current=min(processed, population_state["total"]), total=population_state["total"])
            )

        def _remove_item(idx: int) -> None:
            if idx < 0 or idx >= len(items):
                return

            try:
                removed_item = items.pop(idx)
            except Exception:
                return

            if idx < len(raw_files_with_wcs):
                try:
                    raw_files_with_wcs.pop(idx)
                except Exception:
                    pass

            if idx < len(item_labels):
                try:
                    item_labels.pop(idx)
                except Exception:
                    pass

            if idx < len(footprint_wrapped):
                try:
                    footprint_wrapped.pop(idx)
                except Exception:
                    pass
            if idx < len(centroid_wrapped):
                try:
                    centroid_wrapped.pop(idx)
                except Exception:
                    pass

            if use_listbox_mode:
                if idx < len(selection_state):
                    try:
                        selection_state.pop(idx)
                    except Exception:
                        pass
                adjusted_cache: set[int] = set()
                for pos in listbox_selection_cache:
                    if pos == idx:
                        continue
                    adjusted_cache.add(pos - 1 if pos > idx else pos)
                listbox_selection_cache.clear()
                listbox_selection_cache.update(adjusted_cache)
                if listbox_widget is not None:
                    try:
                        listbox_widget.delete(idx)
                    except Exception:
                        pass
            else:
                if idx < len(check_vars):
                    try:
                        check_vars.pop(idx)
                    except Exception:
                        pass
                if idx < len(checkbuttons):
                    btn = checkbuttons.pop(idx)
                    if btn is not None:
                        try:
                            btn.destroy()
                        except Exception:
                            pass

            known_path_index.clear()
            for pos, entry in enumerate(items):
                key = _path_key(getattr(entry, "path", None))
                if key:
                    known_path_index[key] = pos

            instruments_found.clear()
            for entry in items:
                instruments_found.add((entry.instrument or "Unknown").strip() or "Unknown")
            try:
                _refresh_instrument_options()
            except Exception:
                pass

            pending_visual_refresh.clear()
            _clear_visual_datasets()
            _recompute_footprint_budget()
            _schedule_visual_build(full=True)
            _schedule_axes_update()

            population_state["total"] = len(items)
            if population_state["next_index"] > len(items):
                population_state["next_index"] = len(items)

            if not items:
                try:
                    _draw_group_outlines(None)
                except Exception:
                    pass

            _dbg(f"Removed problematic item: idx={idx}, remaining={len(items)} path={getattr(removed_item, 'path', '?')}")

        def _add_item_row(item: Item) -> None:
            idx = len(item_labels)
            base = os.path.basename(item.path)
            instrument_name = (item.instrument or "").strip()
            tag = ""
            if instrument_name and instrument_name != "Unknown":
                tag = f" [{instrument_name}]"
            sep_txt = ""
            if item.center is not None:
                try:
                    sep_deg = item.center.separation(global_center).to(u.deg).value
                    sep_txt = f"  ({sep_deg:.2f}°)"
                except Exception:
                    sep_txt = ""

            display_text = base + tag + sep_txt
            item_labels.append(display_text)

            if use_listbox_mode:
                selection_state.append(True)
                listbox_selection_cache.add(idx)
                if listbox_widget is not None:
                    listbox_programmatic["active"] = True
                    try:
                        listbox_widget.insert(tk.END, display_text)
                        listbox_widget.selection_set(idx)
                    except Exception:
                        pass
                    finally:
                        listbox_programmatic["active"] = False
            else:
                var = tk.BooleanVar(master=root, value=True)
                check_vars.append(var)
                if inner is not None:
                    cb = ttk.Checkbutton(inner, text=display_text, variable=var, command=lambda i=idx: update_visuals(i))
                    cb.pack(anchor="w", fill="x", pady=1)
                    checkbuttons.append(cb)
                else:
                    checkbuttons.append(None)

            path_key = _path_key(item.path)
            if path_key:
                known_path_index[path_key] = idx

            previous_count = len(instruments_found)
            instruments_found.add(instrument_name or "Unknown")
            if len(instruments_found) != previous_count:
                _refresh_instrument_options()
            current_choice = instrument_var.get()
            if current_choice != "All" and item.instrument != current_choice:
                _set_selected(idx, False)

            _ensure_wrapped_capacity(idx)
            footprint_wrapped[idx] = None
            centroid_wrapped[idx] = None
        ax.grid(True, which="both", linestyle=":", linewidth=0.6)

        def update_visuals(changed_index: Optional[Iterable[int] | int] = None) -> None:
            """Refresh aggregated matplotlib artists to match selection state."""

            if changed_index is None:
                target = set(range(len(items)))
            elif isinstance(changed_index, Iterable) and not isinstance(changed_index, (str, bytes)):
                target = set()
                for raw_idx in changed_index:
                    try:
                        idx_val = int(raw_idx)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx_val < len(items):
                        target.add(idx_val)
            else:
                target = set()
                try:
                    idx_val = int(changed_index)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    idx_val = -1
                if 0 <= idx_val < len(items):
                    target.add(idx_val)

            if not target:
                target = set(range(len(items)))

            updated = False
            lc: Optional[LineCollection] = footprints_state.get("collection")
            if lc is not None and footprints_state["indices"]:
                colors = list(footprints_state["colors"])
                for pos, idx in enumerate(footprints_state["indices"]):
                    if idx not in target:
                        continue
                    selected = _is_selected(idx)
                    colors[pos] = to_rgba("tab:blue" if selected else "0.7", 0.9 if selected else 0.3)
                    updated = True
                if updated:
                    footprints_state["colors"] = colors
                    lc.set_colors(colors)

            coll = centroids_state.get("collection")
            if coll is not None and centroids_state["indices"]:
                colors = list(centroids_state["colors"])
                sizes = list(centroids_state["sizes"])
                centroid_updated = False
                for pos, idx in enumerate(centroids_state["indices"]):
                    if idx not in target:
                        continue
                    selected = _is_selected(idx)
                    colors[pos] = to_rgba("tab:blue" if selected else "0.7", 0.9 if selected else 0.3)
                    sizes[pos] = 28.0 if selected else 18.0
                    centroid_updated = True
                if centroid_updated:
                    centroids_state["colors"] = colors
                    centroids_state["sizes"] = sizes
                    coll.set_facecolors(colors)
                    coll.set_edgecolors(colors)
                    coll.set_sizes(np.asarray(sizes, dtype=float))
                    updated = True

            if updated:
                try:
                    canvas.blit(ax.bbox)
                except Exception:
                    canvas.draw_idle()

        if use_listbox_mode and listbox_widget is not None:
            def _on_listbox_select(_event: Any) -> None:
                if listbox_programmatic.get("active"):
                    return
                try:
                    current = {int(idx) for idx in listbox_widget.curselection()}
                except Exception:
                    return
                added = current - listbox_selection_cache
                removed = listbox_selection_cache - current
                if not added and not removed:
                    return
                for idx in added:
                    if 0 <= idx < len(selection_state):
                        selection_state[idx] = True
                for idx in removed:
                    if 0 <= idx < len(selection_state):
                        selection_state[idx] = False
                listbox_selection_cache.clear()
                listbox_selection_cache.update(current)
                update_visuals(list(added | removed))

            listbox_widget.bind("<<ListboxSelect>>", _on_listbox_select)

        def _finalize_initial_population() -> None:
            if population_state["finalized"]:
                return
            population_state["finalized"] = True
            try:
                _refresh_instrument_options()
            except Exception:
                pass
            update_visuals()
            _schedule_axes_update()
            _update_population_indicator(population_state["total"], done=True)

        def _populate_initial_chunk() -> None:
            total = population_state["total"]
            if total <= 0:
                _finalize_initial_population()
                return
            start = population_state["next_index"]
            if start >= total:
                _finalize_initial_population()
                return
            # Load a larger synchronous chunk first, then smaller async batches
            chunk = 0
            if start == 0:
                chunk = min(total, 400)
            else:
                chunk = min(total - start, 200)
            end = start + chunk
            for idx in range(start, end):
                _add_item_row(items[idx])
            if end > start:
                _schedule_visual_build(full=True)
            population_state["next_index"] = end
            _update_population_indicator(end, done=end >= total)
            try:
                canvas.draw_idle()
            except Exception:
                pass
            if end >= total:
                _finalize_initial_population()
            else:
                # Yield to Tkinter before continuing with the next batch
                try:
                    root.after(15, _populate_initial_chunk)
                except Exception:
                    # If scheduling fails, continue synchronously to avoid losing items
                    _populate_initial_chunk()

        _populate_initial_chunk()

        def _recompute_axes_limits() -> None:
            ra_vals: list[float] = []
            dec_vals: list[float] = []
            for it in items:
                footprint = it.get_cached_footprint()
                if footprint is not None:
                    for ra in footprint[:, 0].tolist():
                        ra_vals.append(wrap_ra_deg(float(ra), ref_ra))
                    dec_vals.extend(footprint[:, 1].tolist())
                elif it.center is not None:
                    ra_vals.append(wrap_ra_deg(float(it.center.ra.to(u.deg).value), ref_ra))
                    dec_vals.append(float(it.center.dec.to(u.deg).value))
            if ra_vals and dec_vals:
                ra_min, ra_max = min(ra_vals), max(ra_vals)
                dec_min, dec_max = min(dec_vals), max(dec_vals)
                ra_pad = max(1e-3, (ra_max - ra_min) * 0.05 + 0.2)
                dec_pad = max(1e-3, (dec_max - dec_min) * 0.05 + 0.2)
                ax.set_xlim(ra_max + ra_pad, ra_min - ra_pad)
                ax.set_ylim(dec_min - dec_pad, dec_max + dec_pad)
                canvas.draw_idle()

        def _schedule_axes_update() -> None:
            if axes_update_pending["pending"]:
                return

            def _do_update() -> None:
                axes_update_pending["pending"] = False
                _recompute_axes_limits()

            axes_update_pending["pending"] = True
            try:
                root.after_idle(_do_update)
            except Exception:
                _do_update()

        def _ingest_batch(batch: list[Dict[str, Any]]) -> None:
            if not batch:
                return
            new_indices: list[int] = []
            visuals_dirty = False
            exclusion_msg = _tr(
                "FILTER_EXCLUDED_DIR",
                'The folder "unaligned_by_zemosaic" is excluded and will not be scanned.',
            )
            for entry in batch:
                candidate_path = entry.get("path") or entry.get("path_raw") or entry.get("path_preprocessed_cache")
                if candidate_path:
                    try:
                        candidate_obj = Path(candidate_path)
                    except Exception:
                        candidate_obj = Path(str(candidate_path))
                    if is_path_excluded(candidate_obj, EXCLUDED_DIRS):
                        _remember_exclusion(candidate_obj, exclusion_msg)
                        continue
                key = _path_key(candidate_path)
                if key and key in known_path_index:
                    existing_idx = known_path_index[key]
                    entry.setdefault("index", existing_idx)
                    try:
                        raw_files_with_wcs[existing_idx].update(entry)
                    except Exception:
                        raw_files_with_wcs[existing_idx] = dict(entry)
                    try:
                        items[existing_idx].src.update(entry)
                    except Exception:
                        items[existing_idx].src = dict(entry)
                    try:
                        items[existing_idx].index = existing_idx
                        items[existing_idx].refresh_geometry()
                    except Exception:
                        pass
                    try:
                        base_name = os.path.basename(items[existing_idx].path)
                        sep_txt = ""
                        if items[existing_idx].center is not None:
                            sep_deg = items[existing_idx].center.separation(global_center).to(u.deg).value
                            sep_txt = f"  ({sep_deg:.2f}°)"
                        _update_item_label(existing_idx, base_name + sep_txt)
                    except Exception:
                        pass
                    try:
                        _refresh_item_visual(existing_idx, schedule_axes=False, trigger_build=False)
                    except Exception:
                        pass
                    visuals_dirty = True
                    continue
                raw_files_with_wcs.append(entry)
                new_index = len(raw_files_with_wcs) - 1
                entry.setdefault("index", new_index)
                new_item = Item(entry, new_index)
                items.append(new_item)
                _add_item_row(new_item)
                new_indices.append(new_index)
                visuals_dirty = True
            if new_indices:
                update_visuals(new_indices)
            if not _center_ready["ok"]:
                _maybe_update_global_center()
                if _center_ready["ok"] and has_explicit_centers:
                    try:
                        thresh_entry.state(["!disabled"])
                    except Exception:
                        thresh_entry.configure(state=tk.NORMAL)
                    try:
                        thresh_button.state(["!disabled"])
                    except Exception:
                        pass
            if visuals_dirty:
                _schedule_visual_build(full=True)
                _schedule_axes_update()

        def _consume_ui_queue() -> None:
            if stream_queue is None or stream_state.get("done"):
                return
            try:
                while True:
                    batch = stream_queue.get_nowait()
                    if batch is None:
                        stream_state["done"] = True
                        stream_state["running"] = False
                        try:
                            pb.stop()
                            status_var.set(_tr("filter_status_ready", "Crawling done."))
                        except Exception:
                            pass
                        try:
                            _log_message(
                                _tr(
                                    "filter_log_analysis_complete",
                                    "Analyse terminee." if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Analysis complete.",
                                ),
                                level="INFO",
                            )
                        except Exception:
                            pass
                        stream_state["status_message"] = _tr("filter_status_ready", "Crawling done.")
                        try:
                            analyse_btn.state(["!disabled"])
                        except Exception:
                            pass
                        break
                    if batch:
                        _ingest_batch(batch)
            except queue.Empty:
                pass
            finally:
                if not stream_state.get("done"):
                    try:
                        root.after(40, _consume_ui_queue)
                    except Exception:
                        stream_state["done"] = True

        if stream_queue is not None and not stream_state.get("done"):
            try:
                root.after(40, _consume_ui_queue)
                # Opportunistic non-blocking pass in case scheduling is delayed
                _drain_stream_queue_non_blocking(mark_done=False)
            except Exception:
                stream_state["done"] = True

        def _drain_stream_queue_non_blocking(mark_done: bool = False) -> None:
            """Drain any queued batches without waiting.

            If ``mark_done`` is True, also request the worker to stop and
            prevent re-scheduling of consumers. This makes Cancel/close
            instantaneous even on slow USB drives.
            """
            nonlocal stream_stop_event
            if stream_queue is None:
                return
            if mark_done:
                stream_state["done"] = True
                if stream_stop_event is not None:
                    try:
                        stream_stop_event.set()
                    except Exception:
                        pass
            drained_any = False
            while True:
                try:
                    batch = stream_queue.get_nowait()
                except queue.Empty:
                    break
                drained_any = True
                if batch is None:
                    stream_state["done"] = True
                    stream_state["running"] = False
                    try:
                        pb.stop()
                        status_var.set(_tr("filter_status_ready", "Crawling done."))
                    except Exception:
                        pass
                    try:
                        _log_message(
                            _tr(
                                "filter_log_analysis_complete",
                                "Analyse terminee." if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Analysis complete.",
                            ),
                            level="INFO",
                        )
                    except Exception:
                        pass
                    stream_state["status_message"] = _tr("filter_status_ready", "Crawling done.")
                    try:
                        analyse_btn.state(["!disabled"])
                    except Exception:
                        pass
                    break
                if batch:
                    _ingest_batch(batch)
            # If we consumed something and we're not marked done, schedule a light pass
            if drained_any and not stream_state.get("done"):
                try:
                    root.after(40, _consume_ui_queue)
                except Exception:
                    stream_state["done"] = True

        def _refresh_item_visual(idx: int, *, schedule_axes: bool = True, trigger_build: bool = True) -> None:
            if idx < 0 or idx >= len(items):
                return
            _ensure_wrapped_capacity(idx)
            footprint_wrapped[idx] = None
            centroid_wrapped[idx] = None
            _prepare_visual_payload(idx)
            if trigger_build:
                _schedule_visual_build(full=True)
            if schedule_axes:
                _schedule_axes_update()

        def _on_draw_mode_change() -> None:
            _schedule_visual_build(full=True)

        try:
            footprints_chk.configure(command=_on_draw_mode_change)
        except Exception:
            pass

        def _trigger_stream_start(force: bool = False) -> None:
            try:
                _dbg(f"_trigger_stream_start called; stream_mode={stream_mode}, pending_start={stream_state.get('pending_start')}, running={stream_state.get('running')}")
            except Exception:
                pass
            if not stream_mode and not callable(stream_state.get("spawn_worker")):
                return
            spawn_callable = stream_state.get("spawn_worker")
            if not callable(spawn_callable):
                return
            if stream_state.get("running"):
                return
            if not force and not stream_state.get("pending_start"):
                return
            stream_state["pending_start"] = False
            stream_state["done"] = False
            stream_state["status_message"] = _tr("filter_status_crawling", "Crawling files… please wait")
            try:
                status_var.set(stream_state["status_message"])
            except Exception:
                pass
            try:
                pb.start(80)
            except Exception:
                pass
            try:
                _log_message("[Filter] Analyse clicked — starting directory crawl…", level="INFO")
            except Exception:
                pass
            try:
                analyse_btn.state(["disabled"])
            except Exception:
                pass
            spawn_callable()
            try:
                root.after(40, _consume_ui_queue)
            except Exception:
                stream_state["done"] = True

        def _on_analyse() -> None:
            try:
                _dbg("Analyse button pressed")
            except Exception:
                pass
            # Always surface an Activity log entry for clarity
            try:
                _log_message(_tr("filter_log_analyse_clicked", "Analyse clicked — preparing scan…"), level="INFO")
            except Exception:
                pass
            # If the spawn worker is not set yet but we have an input directory,
            # build a minimal fallback spawner so Analyse still works.
            if not callable(stream_state.get("spawn_worker")):
                try:
                    if isinstance(input_dir, str) and os.path.isdir(input_dir):
                        def _fallback_crawl_worker(target_queue: "queue.Queue[list[Dict[str, Any]] | None]", stop_event: threading.Event) -> None:
                            batch: list[Dict[str, Any]] = []
                            minimum_batch = max(1, int(batch_size) if isinstance(batch_size, int) else 100)
                            for idx, fpath in enumerate(_iter_fits_paths(input_dir, recursive=scan_recursive)):
                                if stop_event.is_set():
                                    break
                                item = _minimal_header_payload(fpath)
                                item["index"] = idx
                                batch.append(item)
                                if len(batch) >= minimum_batch:
                                    if stop_event.is_set():
                                        break
                                    target_queue.put(batch)
                                    batch = []
                            if not stop_event.is_set():
                                if batch:
                                    target_queue.put(batch)
                            target_queue.put(None)
                        def _fallback_spawn() -> None:
                            nonlocal stream_queue, stream_stop_event
                            if stream_state.get("running"):
                                return
                            stream_queue = queue.Queue()
                            stream_stop_event = threading.Event()
                            stream_state["running"] = True
                            stream_state["done"] = False
                            stream_state["status_message"] = None
                            threading.Thread(target=_fallback_crawl_worker, args=(stream_queue, stream_stop_event), daemon=True).start()
                        stream_state["spawn_worker"] = _fallback_spawn
                except Exception:
                    pass
            if not stream_mode and not callable(stream_state.get("spawn_worker")):
                return
            _trigger_stream_start(force=True)

        def _on_export() -> None:
            path_csv = stream_state.get("csv_path") if isinstance(stream_state.get("csv_path"), str) else cache_csv_path
            if not path_csv:
                return
            try:
                os.makedirs(os.path.dirname(path_csv), exist_ok=True)
            except Exception:
                pass
            payload: list[Dict[str, Any]] = []
            for it in items:
                data = dict(it.src)
                if it.shape is not None:
                    data.setdefault("shape", it.shape)
                if it.center is not None:
                    data.setdefault("center", it.center)
                # If we have a computed footprint (from WCS or CSV), pass it
                # along explicitly so the exporter can persist it.
                try:
                    fp = it.get_cached_footprint()
                    if fp is None:
                        fp_pts = it.src.get("footprint_radec") or it.src.get("_precomp_fp")
                        if isinstance(fp_pts, (list, tuple)) and len(fp_pts) >= 3:
                            try:
                                fp = np.asarray([[float(p[0]), float(p[1])] for p in fp_pts], dtype=float)
                            except Exception:
                                fp = None
                    if fp is not None:
                        coords = np.asarray(fp, dtype=float)
                        data.setdefault("_precomp_fp", [(float(r), float(d)) for r, d in coords.tolist()])
                except Exception:
                    pass
                payload.append(data)
            _export_csv(path_csv, payload)
            stream_state["csv_loaded"] = True
            stream_state["status_message"] = _tr(
                "filter_status_ready_csv",
                "Loaded from CSV. Click Analyse to refresh.",
            )
            try:
                status_var.set(stream_state["status_message"])
            except Exception:
                pass
            _log_message(f"[CSV] Exported {len(payload)} items -> {path_csv}", level="INFO")

        analyse_btn.configure(command=_on_analyse)
        try:
            analyse_btn.bind('<Button-1>', lambda e: (_dbg('Analyse <Button-1> event'), None)[1])
        except Exception:
            pass
        export_btn.configure(command=_on_export)
        if stream_mode:
            try:
                analyse_btn.state(["!disabled"])
            except Exception:
                pass
            try:
                analyse_btn.configure(state=tk.NORMAL)
            except Exception:
                pass
            if stream_state.get("pending_start"):
                # Do not auto-start; wait for user to click Analyse
                try:
                    pb.stop()
                    status_var.set(
                        _tr(
                            "filter_status_click_analyse",
                            "Cliquez sur Analyse pour démarrer l'exploration." if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Ready — click Analyse to scan.",
                        )
                    )
                except Exception:
                    pass
            elif stream_state.get("status_message"):
                try:
                    analyse_btn.state(["!disabled"])
                except Exception:
                    pass

        def _process_async_events() -> None:
            start = time.monotonic()
            processed = 0
            time_budget_exhausted = False
            try:
                while processed < MAX_UI_MSG:
                    if (time.monotonic() - start) * 1000.0 >= UI_BUDGET_MS:
                        time_budget_exhausted = True
                        break
                    try:
                        event = async_events.get_nowait()
                    except queue.Empty:
                        break
                    processed += 1
                    kind = event[0]
                    if kind == "log":
                        _, message, level = event
                        log_buffer.append((str(level or "INFO"), str(message)))
                    elif kind == "header_loaded":
                        _, idx, header_obj = event
                        if (
                            isinstance(idx, int)
                            and 0 <= idx < len(items)
                            and header_obj is not None
                        ):
                            item = items[idx]
                            item.header = header_obj
                            item.src["header"] = header_obj
                    elif kind == "resolved_item":
                        footprint_pts = event[4] if len(event) >= 5 else None
                        _, idx, header_obj, wcs_obj = event[:4]
                        if isinstance(idx, int) and 0 <= idx < len(items):
                            item = items[idx]
                            if header_obj is not None:
                                item.header = header_obj
                                item.src["header"] = header_obj
                            if footprint_pts:
                                try:
                                    item.src["footprint_radec"] = footprint_pts
                                except Exception:
                                    pass
                            if wcs_obj is not None and getattr(wcs_obj, "is_celestial", False):
                                item.src["wcs"] = wcs_obj
                                item.wcs = wcs_obj
                                try:
                                    item.refresh_geometry()
                                except Exception:
                                    pass
                            pending_visual_refresh.add(idx)
                            _schedule_visual_refresh_flush()
                            if footprints_restore_state.get("needs_restore"):
                                indices_cache = footprints_restore_state.get("indices")
                                if not isinstance(indices_cache, set):
                                    indices_cache = set()
                                    footprints_restore_state["indices"] = indices_cache
                                indices_cache.add(idx)
                    elif kind == "resolve_done":
                        _, resolved_now = event
                        try:
                            resolved_count = int(resolved_now)
                        except Exception:
                            resolved_count = 0
                        if resolved_count >= 0:
                            resolved_counter["count"] += resolved_count
                            if resolved_count > 0:
                                overrides_state["resolved_wcs_count"] = resolved_counter["count"]
                        summary_msg = _tr(
                            "filter_log_resolved_n",
                            "Resolved WCS for {n} files.",
                            n=resolved_count,
                        )
                        summary_var.set(_apply_summary_hint(summary_msg))
                        log_buffer.append(("INFO", summary_msg))
                        resolve_state["running"] = False
                        try:
                            resolve_btn.state(["!disabled"])
                        except Exception:
                            pass
                        if not resolve_state.get("running") and pending_outline_state.get("groups") is not None:
                            try:
                                _draw_group_outlines(pending_outline_state.get("groups"))
                            except Exception:
                                pass
                        if footprints_restore_state.get("needs_restore"):
                            original_value = bool(footprints_restore_state.get("value", True))
                            try:
                                draw_footprints_var.set(original_value)
                            except Exception:
                                pass
                            indices_cache = footprints_restore_state.get("indices")
                            if isinstance(indices_cache, set) and indices_cache:
                                pending_visual_refresh.update(indices_cache)
                            if original_value and not pending_visual_refresh:
                                pending_visual_refresh.update(range(len(items)))
                            if pending_visual_refresh:
                                _schedule_visual_refresh_flush()
                            try:
                                footprints_chk.state(["!disabled"])
                            except Exception:
                                try:
                                    footprints_chk.configure(state=tk.NORMAL)
                                except Exception:
                                    pass
                            footprints_restore_state["needs_restore"] = False
                            if isinstance(indices_cache, set):
                                indices_cache.clear()
                    else:
                        pass
            finally:
                now = time.monotonic()
                has_more = False
                if time_budget_exhausted or processed >= MAX_UI_MSG:
                    has_more = True
                if not async_events.empty():
                    has_more = True
                flush_due = now - last_log_flush["ts"] >= LOG_FLUSH_INTERVAL
                _flush_log_buffer(force=flush_due)
                if log_buffer and not flush_due:
                    has_more = True
                try:
                    root.after(60 if has_more else 150, _process_async_events)
                except Exception:
                    pass

        _process_async_events()

        # Click-to-select/deselect via matplotlib pick events
        def _on_pick(event):
            try:
                artist = getattr(event, 'artist', None)
                if artist is None:
                    return
                indices: list[int] = []
                if artist is footprints_state.get("collection"):
                    picked = getattr(event, "ind", None)
                    if picked is None:
                        return
                    if isinstance(picked, (list, tuple, np.ndarray)):
                        candidates = picked
                    else:
                        candidates = [picked]
                    if not candidates:
                        return
                    pos = int(candidates[0])
                    if 0 <= pos < len(footprints_state["indices"]):
                        indices.append(footprints_state["indices"][pos])
                elif artist is centroids_state.get("collection"):
                    picked = getattr(event, "ind", None)
                    if picked is None:
                        return
                    if isinstance(picked, (list, tuple, np.ndarray)):
                        candidates = picked
                    else:
                        candidates = [picked]
                    if not candidates:
                        return
                    pos = int(candidates[0])
                    if 0 <= pos < len(centroids_state["indices"]):
                        indices.append(centroids_state["indices"][pos])
                else:
                    return
                for i in indices:
                    curr = _is_selected(i)
                    _set_selected(i, not curr)
                if indices:
                    update_visuals(indices)
            except Exception:
                pass

        try:
            canvas.mpl_connect('pick_event', _on_pick)
        except Exception:
            pass

        # On window close: treat as cancel (keep all)
        def on_close():
            _cancel_wcs_executor()
            _drain_stream_queue_non_blocking(mark_done=True)
            if result.get("accepted") is None:
                result["accepted"] = False
            result["selected_indices"] = None
            result["cancelled"] = True
            result["overrides"] = _cancel_overrides_payload()
            root.destroy()
        root.protocol("WM_DELETE_WINDOW", on_close)

        # Start modal loop (use wait_window for Toplevel)
        try:
            if root_is_toplevel:
                parent = parent_root if parent_root is not None else root
                try:
                    parent.wait_window(root)
                except Exception:
                    # Fallback to simple update loop
                    try:
                        root.update()
                        root.update_idletasks()
                    except Exception:
                        pass
            else:
                root.mainloop()
        except KeyboardInterrupt:
            # If interrupted, keep default behavior (keep all)
            pass
        except Exception:
            # Some environments (notably on Windows when running from a
            # multiprocessing daemon) occasionally fail to enter Tk's
            # mainloop, leaving the window unresponsive.  Fall back to a
            # manual event pump so the user can still interact with the UI.
            try:
                deadline = time.monotonic() + 3600.0  # 1 hour safety cap
                while True:
                    try:
                        root.update_idletasks()
                        root.update()
                    except Exception:
                        break
                    # Exit when the window is destroyed or a decision was made
                    if not root.winfo_exists() or result.get("accepted") is not None:
                        break
                    if time.monotonic() > deadline:
                        break
                    time.sleep(0.05)
                try:
                    if root.winfo_exists():
                        root.destroy()
                except Exception:
                    pass
            except Exception:
                pass

        # Return selection (and move unselected files into 'filtered_by_user')
        accepted_flag = result.get("accepted")
        if accepted_flag is None:
            accepted_flag = False
        cancelled_flag = bool(result.get("cancelled"))
        if cancelled_flag:
            overrides_payload: dict[str, Any] | None
            overrides_payload = None
            overrides_result = result.get("overrides")
            if isinstance(overrides_result, dict):
                overrides_payload = dict(overrides_result)
            elif isinstance(overrides_state, dict) and overrides_state:
                overrides_payload = dict(overrides_state)
            if overrides_payload is None:
                overrides_payload = {}
            overrides_payload["filter_cancelled"] = True
            return raw_files_with_wcs, False, overrides_payload
        if accepted_flag and isinstance(result.get("selected_indices"), list):
            sel = result["selected_indices"]  # type: ignore[assignment]

            # Compute unselected indices
            total_n = len(raw_files_with_wcs)
            unselected_indices = [i for i in range(total_n) if i not in sel]
            if unselected_indices:
                overrides_state["filter_excluded_indices"] = unselected_indices

            # Prepare destination folder under the common input directory
            def _preferred_src_path(entry: Dict[str, Any]) -> Optional[str]:
                # Prefer raw/original paths if available
                p = entry.get("path_raw") or entry.get("path")
                if isinstance(p, str):
                    return p
                return None

            excluded_paths: list[str] = []
            all_src_dirs: list[str] = []
            for i in unselected_indices:
                p = _preferred_src_path(raw_files_with_wcs[i])
                if p and os.path.isfile(p):
                    excluded_paths.append(p)
                    all_src_dirs.append(os.path.dirname(p))

            dest_base: Optional[str] = None
            if all_src_dirs:
                try:
                    dest_base = os.path.commonpath(all_src_dirs)
                except Exception:
                    dest_base = all_src_dirs[0]

            # If we have a base, move excluded files to '<base>/filtered_by_user'
            if dest_base is not None and excluded_paths:
                dest_dir = os.path.join(dest_base, "filtered_by_user")
                try:
                    os.makedirs(dest_dir, exist_ok=True)
                except Exception:
                    dest_dir = None  # will fallback to per-file folder

                def _unique_dest(path_dir: str, filename: str) -> str:
                    base, ext = os.path.splitext(filename)
                    candidate = os.path.join(path_dir, filename)
                    if not os.path.exists(candidate):
                        return candidate
                    # Append timestamp, then counter if still colliding
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    candidate = os.path.join(path_dir, f"{base}_{ts}{ext}")
                    if not os.path.exists(candidate):
                        return candidate
                    k = 1
                    while True:
                        candidate = os.path.join(path_dir, f"{base}_{ts}_{k}{ext}")
                        if not os.path.exists(candidate):
                            return candidate
                        k += 1

                # Perform moves
                for src_path in excluded_paths:
                    try:
                        target_dir = dest_dir
                        if target_dir is None:
                            # As fallback, move next to its source directory under a local 'filtered_by_user'
                            local_dir = os.path.join(os.path.dirname(src_path), "filtered_by_user")
                            os.makedirs(local_dir, exist_ok=True)
                            target_dir = local_dir
                        dest_path = _unique_dest(target_dir, os.path.basename(src_path))
                        shutil.move(src_path, dest_path)
                    except Exception as e:
                        # Non-fatal: keep going
                        print(f"WARN filter_gui: Failed to move '{src_path}' -> filtered_by_user: {e}")

            return [raw_files_with_wcs[i] for i in sel], True, (overrides_state if overrides_state else None)
        else:
            # Keep all if canceled or closed
            return raw_files_with_wcs, False, None

    except ImportError as exc:
        # Optional dependency missing — report clearly and keep all
        try:
            print(f"[FilterUI] ImportError: {exc}")
            import traceback as _tb
            print(_tb.format_exc())
        except Exception:
            pass
        # If a Tk root exists (launched from main GUI), surface a messagebox
        try:
            import tkinter as _tk
            from tkinter import messagebox as _mb
            if getattr(_tk, "_default_root", None) is not None:
                _mb.showerror(
                    "Filter UI",
                    "Missing optional dependency (numpy/matplotlib/astropy).\nInstall the packages and retry.",
                )
        except Exception:
            pass
        return raw_files_with_wcs, False, None
    except Exception as exc:
        # Unexpected error — report and fail safe
        try:
            print(f"[FilterUI] Error while building filter UI: {exc}")
            import traceback as _tb
            print(_tb.format_exc())
        except Exception:
            pass
        try:
            import tkinter as _tk
            from tkinter import messagebox as _mb
            if getattr(_tk, "_default_root", None) is not None:
                _mb.showerror(
                    "Filter UI",
                    f"Could not open filter window.\n{exc}",
                )
        except Exception:
            pass
        return raw_files_with_wcs, False, None


__all__ = ["launch_filter_interface"]

if __name__ == "__main__":
    # Minimal CLI to launch the filter window for a directory.
    import sys as _sys, os as _os
    args = _sys.argv[1:]
    if not args:
        print("Usage: python zemosaic_filter_gui.py <input_dir>")
        print("       python -m zemosaic_filter_gui <input_dir>")
        _sys.exit(1)
    inp = _os.path.expanduser(_os.path.expandvars(args[0]))
    if not (_os.path.isdir(inp)):
        print(f"Error: '{inp}' is not a directory")
        _sys.exit(2)
    try:
        res = launch_filter_interface(
            inp,
            None,
            stream_scan=True,
            scan_recursive=True,
            batch_size=200,
            preview_cap=1500,
            solver_settings_dict=None,
            config_overrides=None,
        )
        # Print a short summary to the console so CLI runs are visible
        if isinstance(res, tuple) and len(res) >= 2:
            kept = len(res[0]) if isinstance(res[0], list) else 0
            accepted = bool(res[1])
            print(f"[FilterUI] Closed. accepted={accepted} kept={kept}")
        else:
            kept = len(res) if isinstance(res, list) else 0
            print(f"[FilterUI] Closed. kept={kept}")
    except KeyboardInterrupt:
        pass
    except Exception as _exc:
        print(f"[FilterUI] Unhandled error: {_exc}")
        import traceback as _tb
        print(_tb.format_exc())

