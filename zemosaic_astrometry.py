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


import os
import numpy as np
import warnings
import time
import tempfile
import traceback
import subprocess
import shutil
import gc
import logging
import psutil
import threading
import hashlib
import queue
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Optional

import multiprocessing

from core.path_helpers import expand_to_path, safe_path_exists, safe_path_getsize

try:
    from zemosaic_utils import get_runtime_temp_dir  # type: ignore
except Exception:  # pragma: no cover - standalone usage
    def get_runtime_temp_dir() -> Path:
        root = Path(tempfile.gettempdir()) / "zemosaic_runtime"
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception:
            root = Path(tempfile.gettempdir())
        return root


def _expand_path(pathish: str | os.PathLike[str] | Path | None, *, expanduser: bool = True) -> Path | None:
    """Return a Path from *pathish* with optional user/env expansion."""

    return expand_to_path(pathish, expanduser=expanduser, expandvars=expanduser)


def _path_display_name(pathish: str | os.PathLike[str] | Path | None) -> str:
    """Return a human-friendly basename for *pathish*."""

    if pathish is None:
        return "<inconnu>"
    try:
        name = Path(pathish).name
        return name or str(pathish)
    except Exception:
        return str(pathish)

try:
    import msvcrt
except ImportError:  # pragma: no cover - only available on Windows
    msvcrt = None

try:
    import fcntl
except ImportError:  # pragma: no cover - not available on Windows
    fcntl = None

try:
    import ctypes
except Exception:  # pragma: no cover - very limited Python builds
    ctypes = None


def _env_flag_enabled(key: str) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "on", "yes"}


ProgressCallback = Callable[[str, Any, str], None]

_KEEP_DIALOGS = _env_flag_enabled("KEEP_DIALOGS")

if os.name == "nt" and ctypes is not None and not _KEEP_DIALOGS:
    try:
        SEM_FAILCRITICALERRORS = 0x0001
        SEM_NOGPFAULTERRORBOX = 0x0002
        SEM_NOOPENFILEERRORBOX = 0x8000
        ctypes.windll.kernel32.SetErrorMode(
            SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
        )
    except Exception:
        pass


logger = logging.getLogger("ZeMosaicAstrometry")
# ... (pas besoin de reconfigurer le logger ici s'il hérite du worker)

try:
    from astropy.io import fits
    from astropy.wcs import WCS as AstropyWCS, FITSFixedWarning 
    from astropy.utils.exceptions import AstropyWarning
    from astropy import units as u # Nécessaire pour _update_fits_header_with_wcs_za
    ASTROPY_AVAILABLE_ASTROMETRY = True
    warnings.filterwarnings('ignore', category=FITSFixedWarning)
    warnings.filterwarnings('ignore', category=AstropyWarning)
except ImportError:
    logger.error("Astropy non installée. Certaines fonctionnalités de zemosaic_astrometry seront limitées.")
    ASTROPY_AVAILABLE_ASTROMETRY = False
    class AstropyWCS: pass
    class FITSFixedWarning(Warning): pass
    u = None
    
try:
    # Optional import placed outside the main astropy try to avoid failing the module import
    # if coordinates submodule is unavailable. We degrade gracefully to None.
    from astropy.coordinates import SkyCoord as AstropySkyCoord
except Exception:
    AstropySkyCoord = None

# --- Dépendances pour Astrometry.net web ---
ASTROQUERY_AVAILABLE_ASTROMETRY = False
AstrometryNet = None
try:
    from astroquery.astrometry_net import AstrometryNet as _AstrometryNetClass
    AstrometryNet = _AstrometryNetClass
    ASTROQUERY_AVAILABLE_ASTROMETRY = True
except Exception:
    logger.warning(
        "AstrometrySolver: astroquery non installée. Plate-solving web Astrometry.net désactivé."
    )

_ASTAP_CONCURRENCY_ENV_KEYS = (
    "ZEMOSAIC_ASTAP_MAX_PROCS",
    "ZEMO_ASTAP_MAX_PROCS",
    "ZEMO_FILTER_ASTAP_MAX_PROCS",
)


def _resolve_astap_max_procs(default: int = 1) -> int:
    """Return the configured ASTAP concurrency cap."""
    for key in _ASTAP_CONCURRENCY_ENV_KEYS:
        raw = os.environ.get(key)
        if raw is None:
            continue
        raw_str = str(raw).strip()
        if not raw_str:
            continue
        try:
            value = int(float(raw_str))
        except Exception:
            logger.debug("ASTAP concurrency: ignoring invalid %s=%r", key, raw)
            continue
        if value >= 1:
            return value
    return max(1, default)


_ASTAP_MAX_PROCS = _resolve_astap_max_procs()
_ASTAP_SEMAPHORE_LOCK = threading.Lock()
_ASTAP_SLOT_SEMAPHORE = threading.BoundedSemaphore(value=_ASTAP_MAX_PROCS)
logger.debug(
    "ASTAP concurrency limited to %s process(es) (env keys: %s)",
    _ASTAP_MAX_PROCS,
    ", ".join(_ASTAP_CONCURRENCY_ENV_KEYS),
)


@contextmanager
def _astap_slot_guard(image_label: str = ""):
    """Serialize ASTAP runs across threads to avoid concurrent crashes."""
    semaphore = _ASTAP_SLOT_SEMAPHORE
    if semaphore is None:
        yield
        return
    if image_label:
        logger.debug("ASTAP slot wait -> %s", image_label)
    semaphore.acquire()
    try:
        if image_label:
            logger.debug("ASTAP slot acquired -> %s", image_label)
        yield
    finally:
        if image_label:
            logger.debug("ASTAP slot release -> %s", image_label)
        semaphore.release()


def _env_flag_enabled(key: str) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "on", "yes"}


_ASTAP_IPC_LOCK_DISABLED = _env_flag_enabled("ZEMOSAIC_ASTAP_DISABLE_IPC_LOCK")
_TEMP_BASE_DIR = get_runtime_temp_dir()
_ASTAP_IPC_LOCK_DIR = (
    _expand_path(os.environ.get("ZEMOSAIC_ASTAP_LOCK_DIR"))
    or (_TEMP_BASE_DIR / "zemosaic_astap_slots")
)
_ASTAP_IPC_LOCK_SUPPORTED = bool(msvcrt or fcntl)
_ASTAP_IPC_WAIT_LOG_THRESHOLD_SEC = 2.0
_ASTAP_IPC_WAIT_LOG_INTERVAL_SEC = 12.0
_ASTAP_IPC_LOCK_WARNING_EMITTED = False
_PER_FILE_LOCK_DIR = _TEMP_BASE_DIR / "zemo_astap_filelocks"
try:
    _PER_FILE_LOCK_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
_ASTAP_RATE_MIN_INTERVAL = max(
    0.0, float(os.environ.get("ZEMOSAIC_ASTAP_RATE_SEC", "0.75") or 0.75)
)
_ASTAP_RATE_BURST = max(1, int(os.environ.get("ZEMOSAIC_ASTAP_RATE_BURST", "2") or 2))
_ASTAP_LAST_LAUNCH_TIMES: list[float] = []
_ASTAP_RATE_LOCK = threading.Lock()


def _initialize_slot_file(pathish: str | Path) -> None:
    """Ensure the lock file exists and has at least one byte."""

    path_obj = Path(pathish)
    try:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    try:
        if not safe_path_exists(path_obj):
            with path_obj.open("wb") as handle:
                handle.write(b"0")
                handle.flush()
        elif safe_path_getsize(path_obj) == 0:
            with path_obj.open("ab") as handle:
                handle.write(b"0")
                handle.flush()
    except Exception:
        pass


def _open_slot_handle(pathish: str | Path):
    path_obj = Path(pathish)
    try:
        return path_obj.open("r+b")
    except FileNotFoundError:
        _initialize_slot_file(path_obj)
        try:
            return path_obj.open("r+b")
        except Exception:
            return None
    except Exception:
        return None


def _try_lock_handle(handle) -> bool:
    if handle is None:
        return False
    if msvcrt is not None and os.name == "nt":
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    if fcntl is not None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (BlockingIOError, OSError):
            return False
    return False


def _unlock_handle(handle) -> None:
    if handle is None:
        return
    try:
        if msvcrt is not None and os.name == "nt":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        elif fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


@contextmanager
def _astap_ipc_slot_guard(image_label: str = "", progress_callback: callable = None):
    """Cooperate across ZeMosaic/ZeSeestarStacker processes to avoid ASTAP popups."""
    global _ASTAP_IPC_LOCK_WARNING_EMITTED

    if _ASTAP_IPC_LOCK_DISABLED or not _ASTAP_IPC_LOCK_SUPPORTED:
        yield
        return

    slot_count = max(1, get_astap_max_concurrent_instances())
    try:
        _ASTAP_IPC_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        if not _ASTAP_IPC_LOCK_WARNING_EMITTED:
            logger.warning(
                "ASTAP IPC lock disabled (cannot create %s): %s",
                _ASTAP_IPC_LOCK_DIR,
                exc,
            )
            _ASTAP_IPC_LOCK_WARNING_EMITTED = True
        yield
        return

    slot_paths: list[Path] = []
    for idx in range(slot_count):
        slot_path = _ASTAP_IPC_LOCK_DIR / f"slot_{idx}.lock"
        slot_paths.append(slot_path)
        _initialize_slot_file(slot_path)

    wait_start = time.monotonic()
    last_wait_log = wait_start - _ASTAP_IPC_WAIT_LOG_INTERVAL_SEC
    acquired_handle = None
    slot_name = None

    while acquired_handle is None:
        for slot_path in slot_paths:
            candidate = _open_slot_handle(slot_path)
            if candidate is None:
                continue
            if _try_lock_handle(candidate):
                acquired_handle = candidate
                slot_name = slot_path.name
                break
            candidate.close()
        if acquired_handle is not None:
            break
        now = time.monotonic()
        waited = now - wait_start
        if progress_callback and (
            waited >= _ASTAP_IPC_WAIT_LOG_THRESHOLD_SEC
            and (now - last_wait_log) >= _ASTAP_IPC_WAIT_LOG_INTERVAL_SEC
        ):
            progress_callback(
                "  ASTAP Solve: attente qu'une autre instance libère ASTAP (verrou global).",
                None,
                "DEBUG",
            )
            last_wait_log = now
        time.sleep(0.2)

    if slot_name:
        logger.debug("ASTAP IPC slot '%s' acquired for %s", slot_name, image_label or "?")

    try:
        yield
    finally:
        if acquired_handle:
            _unlock_handle(acquired_handle)
            acquired_handle.close()
            if slot_name:
                logger.debug(
                    "ASTAP IPC slot '%s' released for %s", slot_name, image_label or "?"
                )


@contextmanager
def _temp_cwd_isolated(prefix: str = "zemo_astap_run"):
    """Create a dedicated temporary directory used as cwd for an ASTAP invocation."""
    root = Path(tempfile.gettempdir()) / (
        f"{prefix}_{os.getpid()}_{threading.get_ident()}_{int(time.time()*1000)}"
    )
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield str(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


@contextmanager
def _lock_for_image(
    image_path: str, image_label: str = "", progress_callback: callable = None
):
    """
    Cross-process lock preventing concurrent solves of the same FITS path.
    """
    image_path_obj = _expand_path(image_path)
    if image_path_obj is None:
        yield
        return
    try:
        digest_source = str(image_path_obj.resolve(strict=False))
    except Exception:
        digest_source = str(image_path_obj)
    digest = hashlib.sha1(digest_source.encode("utf-8", "ignore")).hexdigest()[:16]
    lock_path = _PER_FILE_LOCK_DIR / f"{digest}.lock"
    handle = None
    acquired = False
    wait_notified = False
    start = time.monotonic()
    label = image_label or image_path_obj.name or str(image_path_obj)
    try:
        handle = lock_path.open("a+b")
        try:
            if safe_path_getsize(lock_path) == 0:
                handle.write(b"0")
                handle.flush()
        except Exception:
            pass
        while not acquired:
            try:
                if msvcrt is not None and os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                elif fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                else:
                    acquired = True
            except (OSError, BlockingIOError):
                pass
            if not acquired:
                if not wait_notified and progress_callback:
                    progress_callback(
                        f"  ASTAP Solve: attente verrou fichier pour '{label}'.",
                        None,
                        "DEBUG_DETAIL",
                    )
                    wait_notified = True
                if time.monotonic() - start > 120:
                    raise TimeoutError(f"image-level lock timeout for '{label}'")
                time.sleep(0.1)
        if progress_callback:
            progress_callback(
                f"  ASTAP Solve: verrou fichier acquis pour '{label}'.",
                None,
                "DEBUG_DETAIL",
            )
        yield
    finally:
        try:
            if handle:
                if msvcrt is not None and os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                elif fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
        except Exception:
            pass
        if progress_callback and acquired:
            progress_callback(
                f"  ASTAP Solve: verrou fichier libéré pour '{label}'.",
                None,
                "DEBUG_DETAIL",
            )


def _rate_limit_before_launch(progress_callback: callable = None):
    """Simple token bucket to avoid ASTAP bursts that trigger access violations."""
    if _ASTAP_RATE_MIN_INTERVAL <= 0:
        return
    attempts = 0
    while True:
        with _ASTAP_RATE_LOCK:
            now = time.monotonic()
            while _ASTAP_LAST_LAUNCH_TIMES and (
                now - _ASTAP_LAST_LAUNCH_TIMES[0] > _ASTAP_RATE_MIN_INTERVAL
            ):
                _ASTAP_LAST_LAUNCH_TIMES.pop(0)
            if len(_ASTAP_LAST_LAUNCH_TIMES) < _ASTAP_RATE_BURST:
                _ASTAP_LAST_LAUNCH_TIMES.append(now)
                return
            wait_time = max(
                0.0,
                _ASTAP_RATE_MIN_INTERVAL
                - (now - (_ASTAP_LAST_LAUNCH_TIMES[0] if _ASTAP_LAST_LAUNCH_TIMES else now)),
            )
        attempts += 1
        if progress_callback:
            progress_callback(
                f"  ASTAP Solve: limitation de débit active, attente {wait_time:.2f}s.",
                None,
                "DEBUG_DETAIL" if attempts < 3 else "INFO",
            )
        time.sleep(max(0.05, wait_time))


def set_astap_max_concurrent_instances(count: int) -> int:
    """Update the ASTAP concurrency cap at runtime."""
    global _ASTAP_MAX_PROCS, _ASTAP_SLOT_SEMAPHORE
    try:
        desired = int(count)
    except Exception:
        desired = _ASTAP_MAX_PROCS
    desired = max(1, desired)
    with _ASTAP_SEMAPHORE_LOCK:
        if desired == _ASTAP_MAX_PROCS:
            return _ASTAP_MAX_PROCS
        _ASTAP_MAX_PROCS = desired
        _ASTAP_SLOT_SEMAPHORE = threading.BoundedSemaphore(value=_ASTAP_MAX_PROCS)
        logger.info("ASTAP concurrency cap updated to %s via runtime setter.", _ASTAP_MAX_PROCS)
        return _ASTAP_MAX_PROCS


def get_astap_max_concurrent_instances() -> int:
    """Return the current ASTAP concurrency cap."""
    return _ASTAP_MAX_PROCS


def _log_memory_usage(progress_callback: callable, context_message: str = ""):
    """Logue l'utilisation mémoire du processus courant."""
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
            f"Sys RAM: Avail {available_ram_mb:.0f}MB / Total {total_ram_mb:.0f}MB ({percent_ram_used}% used). "
            f"Sys Swap: Used {used_swap_mb:.0f}MB / Total {total_swap_mb:.0f}MB ({percent_swap_used}% used)."
        )
        progress_callback(log_msg, None, "DEBUG")
    except Exception as e_mem_log:
        progress_callback(f"Erreur lors du logging mémoire ({context_message}): {e_mem_log}", None, "WARN")
    


def extract_center_from_header(original_fits_header) -> "AstropySkyCoord | None":
    """Extract a SkyCoord center from a FITS header.

    Reuses the same interpretation as used for ASTAP hints:
    - Prefer CRVAL1/2 (degrees, WCS)
    - Fallback to RA/DEC, OBJCTRA/OBJCTDEC, TELRA/TELDEC
    - Handle HH:MM:SS formats for RA and DD:MM:SS for Dec
    - Respect RAUNIT/CUNIT1 when RA is numeric (hours vs degrees)

    Returns None if Astropy is unavailable or no valid coordinates are found.
    """
    if not (ASTROPY_AVAILABLE_ASTROMETRY and AstropySkyCoord and u):
        return None
    if original_fits_header is None:
        return None

    try:
        def _safe_float(x):
            try:
                return float(x)
            except Exception:
                return None

        ra_candidates = [
            original_fits_header.get("CRVAL1"),
            original_fits_header.get("RA"),
            original_fits_header.get("OBJCTRA"),
            original_fits_header.get("TELRA"),
        ]
        dec_candidates = [
            original_fits_header.get("CRVAL2"),
            original_fits_header.get("DEC"),
            original_fits_header.get("OBJCTDEC"),
            original_fits_header.get("TELDEC"),
        ]

        crval1_val = original_fits_header.get("CRVAL1")
        ra_val = next((v for v in ra_candidates if v is not None), None)
        dec_val = next((v for v in dec_candidates if v is not None), None)

        ra_deg = None
        dec_deg = None

        # RA handling (deg or hours)
        if isinstance(ra_val, (int, float)):
            try:
                same_as_crval1 = (crval1_val is not None and float(ra_val) == float(crval1_val))
            except Exception:
                same_as_crval1 = False
            if same_as_crval1:
                ra_deg = float(ra_val)
            else:
                try:
                    unit = (original_fits_header.get("RAUNIT") or original_fits_header.get("CUNIT1") or "").strip().lower()
                except Exception:
                    unit = ""
                if unit.startswith("h"):
                    ra_deg = float(ra_val) * 15.0
                else:
                    ra_deg = float(ra_val)
        elif isinstance(ra_val, str):
            s = ra_val.strip().lower().replace("h", ":").replace("m", ":").replace("s", "").replace(" ", ":")
            parts = [p for p in s.split(":") if p != ""]
            if 1 <= len(parts) <= 3 and all(p.replace(".", "", 1).lstrip("-+").isdigit() for p in parts):
                hh = _safe_float(parts[0]) or 0.0
                mm = _safe_float(parts[1]) or 0.0 if len(parts) > 1 else 0.0
                ss = _safe_float(parts[2]) or 0.0 if len(parts) > 2 else 0.0
                ra_deg = (hh + mm / 60.0 + ss / 3600.0) * 15.0
            else:
                val = _safe_float(ra_val)
                if val is not None:
                    ra_deg = float(val)

        # Dec handling (always degrees)
        if isinstance(dec_val, (int, float)):
            dec_deg = float(dec_val)
        elif isinstance(dec_val, str):
            s = dec_val.strip().lower().replace("d", ":").replace(" ", ":")
            parts = [p for p in s.split(":") if p != ""]
            if 1 <= len(parts) <= 3 and all(p.replace(".", "", 1).lstrip("-+").isdigit() for p in parts):
                dd = _safe_float(parts[0]) or 0.0
                mm = _safe_float(parts[1]) or 0.0 if len(parts) > 1 else 0.0
                ss = _safe_float(parts[2]) or 0.0 if len(parts) > 2 else 0.0
                sign = -1.0 if str(parts[0]).strip().startswith("-") else 1.0
                dec_deg = sign * (abs(dd) + mm / 60.0 + ss / 3600.0)
            else:
                val = _safe_float(dec_val)
                if val is not None:
                    dec_deg = float(val)

        if ra_deg is not None and dec_deg is not None:
            return AstropySkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    except Exception:
        return None
    return None


def _run_astap_once(
    cmd_list: list,
    timeout_sec: float,
    progress_callback: Optional[ProgressCallback] = None,
    cwd: Optional[str] = None,
    proc_ready_callback: Optional[Callable[[subprocess.Popen], None]] = None,
):
    """Launch ASTAP once, streaming stdout for logging, and return its rc."""
    if progress_callback:
        progress_callback("  ASTAP: lancement du solveur...", None, "DEBUG")
    creationflags = 0
    startupinfo = None
    if os.name == "nt":
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
    start_time = time.monotonic()
    proc = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd or None,
        creationflags=creationflags,
        startupinfo=startupinfo,
        text=True,
        encoding="utf-8",
        errors="ignore",
        bufsize=1,
    )
    reader_done = threading.Event()
    output_queue: "queue.SimpleQueue[str]" = queue.SimpleQueue()

    def _drain_stdout():
        try:
            if proc.stdout:
                for raw_line in proc.stdout:
                    output_queue.put(raw_line.rstrip("\r\n"))
        finally:
            reader_done.set()

    reader_thread = None
    if proc.stdout:
        reader_thread = threading.Thread(target=_drain_stdout, daemon=True)
        reader_thread.start()
    else:
        reader_done.set()

    if proc_ready_callback is not None:
        try:
            proc_ready_callback(proc)
        except Exception:
            logger.debug("ASTAP proc hook raised an exception", exc_info=True)

    try:
        while True:
            while True:
                try:
                    decoded = output_queue.get_nowait()
                except queue.Empty:
                    break
                if decoded and progress_callback:
                    progress_callback(f"  ASTAP> {decoded}", None, "DEBUG_DETAIL")
            if proc.poll() is not None and reader_done.is_set() and output_queue.empty():
                break
            if time.monotonic() - start_time > timeout_sec:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
                raise subprocess.TimeoutExpired(cmd_list, timeout_sec)
            time.sleep(0.05)
        return proc.returncode or 0
    finally:
        if reader_thread is not None:
            try:
                reader_thread.join(timeout=1.0)
            except Exception:
                pass
        if proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass


def _calculate_pixel_scale_from_header(header: fits.Header, progress_callback: callable = None) -> float | None:
    # ... (corps de la fonction inchangé, il semble correct)
    if not header:
        return None
    focal_len_mm = None
    pixel_size_um = None
    focal_keys = ['FOCALLEN', 'FOCAL', 'FLENGTH']
    for key in focal_keys:
        if key in header and isinstance(header[key], (int, float)) and header[key] > 0:
            focal_len_mm = float(header[key])
            if progress_callback: progress_callback(f"  ASTAP ScaleCalc: Trouvé {key}={focal_len_mm} mm", None, "DEBUG_DETAIL")
            break
    if focal_len_mm is None:
        if progress_callback: progress_callback("  ASTAP ScaleCalc: FOCALLEN non trouvée ou invalide dans le header.", None, "DEBUG_DETAIL")
        return None
    pix_size_keys = ['XPIXSZ', 'PIXSIZE', 'PIXELSIZE', 'PIXSCAL1', 'SCALE']
    for key in pix_size_keys:
        if key in header and isinstance(header[key], (int, float)) and header[key] > 0:
            if key.upper() == 'PIXSCAL1':
                unit_key = f"CUNIT{key[-1]}" if key[-1].isdigit() else None
                if unit_key and unit_key in header and str(header[unit_key]).lower() in ['arcsec', 'asec', '"']:
                    if progress_callback: progress_callback(f"  ASTAP ScaleCalc: Trouvé {key}={header[key]} arcsec/pix directement.", None, "DEBUG_DETAIL")
                    return float(header[key])
            pixel_size_um = float(header[key])
            if progress_callback: progress_callback(f"  ASTAP ScaleCalc: Trouvé {key}={pixel_size_um} µm", None, "DEBUG_DETAIL")
            break
    if pixel_size_um is None:
        if progress_callback: progress_callback("  ASTAP ScaleCalc: XPIXSZ (ou équivalent) non trouvé ou invalide.", None, "DEBUG_DETAIL")
        return None
    try:
        calculated_scale_arcsec_pix = (pixel_size_um / focal_len_mm) * 206.264806
        if progress_callback: progress_callback(f"  ASTAP ScaleCalc: Échelle calculée: {calculated_scale_arcsec_pix:.3f} arcsec/pix", None, "INFO_DETAIL")
        return calculated_scale_arcsec_pix
    except ZeroDivisionError:
        if progress_callback: progress_callback("  ASTAP ScaleCalc ERREUR: Division par zéro (FOCALLEN nulle ?).", None, "WARN")
        return None

def _parse_wcs_file_content_za(wcs_file_path, image_shape_hw, progress_callback=None):
    # ... (corps de la fonction inchangé, il semble correct)
    path_obj = _expand_path(wcs_file_path, expanduser=False)
    filename_log = _path_display_name(path_obj or wcs_file_path)
    open_target = path_obj or wcs_file_path
    if progress_callback:
        progress_callback(
            f"  ASTAP WCS Parse: Tentative parsing '{filename_log}' pour shape {image_shape_hw}",
            None,
            "DEBUG_DETAIL",
        )
    if not (safe_path_exists(open_target) and safe_path_getsize(open_target) > 0):
        if progress_callback:
            progress_callback(
                f"    ASTAP WCS Parse ERREUR: Fichier WCS '{filename_log}' non trouvé ou vide.",
                None,
                "WARN",
            )
        return None
    if not ASTROPY_AVAILABLE_ASTROMETRY:
        if progress_callback: progress_callback("    ASTAP WCS Parse ERREUR: Astropy non disponible pour parser WCS.", None, "ERROR")
        return None
    try:
        # Read header text and normalize newlines. Some .wcs files contain invalid
        # FITS CONTINUE cards (not attached to string values) that astropy.io.fits
        # refuses to serialize. Passing the raw header text to AstropyWCS avoids
        # FITS Card verification. If it still fails, strip CONTINUE lines as fallback.
        with open(open_target, 'r', errors='replace') as f:
            wcs_text_raw = f.read()
        wcs_text_norm = wcs_text_raw.replace('\r\n', '\n').replace('\r', '\n')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            try:
                # Prefer passing header string directly to bypass FITS Card checks
                wcs_obj = AstropyWCS(wcs_text_norm, naxis=2, relax=True)
            except Exception:
                # Fallback: strip any invalid CONTINUE cards and retry
                lines = wcs_text_norm.split('\n')
                lines_no_continue = [ln for ln in lines if not ln.lstrip().upper().startswith('CONTINUE')]
                cleaned_text = '\n'.join(lines_no_continue)
                wcs_obj = AstropyWCS(cleaned_text, naxis=2, relax=True)
        if wcs_obj and wcs_obj.is_celestial:
            if image_shape_hw and image_shape_hw[0] > 0 and image_shape_hw[1] > 0:
                try:
                    wcs_obj.pixel_shape = (image_shape_hw[1], image_shape_hw[0])
                except Exception as e_ps_parse:
                    if progress_callback: progress_callback(f"    ASTAP WCS Parse AVERT: Échec set pixel_shape sur WCS parsé: {e_ps_parse}", None, "WARN")
            if progress_callback: progress_callback(f"    ASTAP WCS Parse: Objet WCS parsé avec succès depuis '{filename_log}'.", None, "DEBUG_DETAIL")
            return wcs_obj
        else:
            if progress_callback: progress_callback(f"    ASTAP WCS Parse ERREUR: Échec création WCS valide/céleste depuis '{filename_log}'.", None, "WARN")
            return None
    except Exception as e:
        if progress_callback: progress_callback(f"    ASTAP WCS Parse ERREUR: Exception lors du parsing WCS '{filename_log}': {e}", None, "ERROR")
        logger.error(f"Erreur parsing WCS '{wcs_file_path}': {e}", exc_info=True)
        return None


def _parse_wcs_file_content_za_v2(wcs_file_path, image_shape_hw, progress_callback=None):
    """Parse un .wcs ASTAP de maniere robuste (encodage, CONTINUE, etc.).

    - Lis en binaire puis nettoie les caracteres non-ASCII (ASTAP peut ecrire des degres/alpha/delta).
    - Supprime les cartes CONTINUE orphelines.
    - Tente WCS via chaine brute puis via fits.Header.
    - Force pixel_shape si possible et sanitise image_shape_hw.
    """
    path_obj = _expand_path(wcs_file_path, expanduser=False)
    filename_log = _path_display_name(path_obj or wcs_file_path)
    open_target = path_obj or wcs_file_path
    if progress_callback:
        progress_callback(
            f"  ASTAP WCS Parse: Tentative parsing (v2) '{filename_log}' pour shape {image_shape_hw}",
            None,
            "DEBUG_DETAIL",
        )
    if not (safe_path_exists(open_target) and safe_path_getsize(open_target) > 0):
        if progress_callback:
            progress_callback(
                f"    ASTAP WCS Parse ERREUR: Fichier WCS '{filename_log}' non trouve ou vide.",
                None,
                "WARN",
            )
        return None
    if not ASTROPY_AVAILABLE_ASTROMETRY:
        if progress_callback:
            progress_callback(
                "    ASTAP WCS Parse ERREUR: Astropy non disponible pour parser WCS.",
                None,
                "ERROR",
            )
        return None

    # Sanitise shape -> ints
    sane_shape_hw = None
    try:
        if image_shape_hw and len(image_shape_hw) >= 2:
            h = int(image_shape_hw[0])
            w = int(image_shape_hw[1])
            sane_shape_hw = (h, w)
    except Exception:
        sane_shape_hw = None

    try:
        with open(open_target, 'rb') as f:
            raw_bytes = f.read()
        if raw_bytes.startswith(b"\xEF\xBB\xBF"):
            raw_bytes = raw_bytes[3:]

        try:
            header_text = raw_bytes.decode('ascii')
        except UnicodeDecodeError:
            try:
                header_text = raw_bytes.decode('latin-1')
            except UnicodeDecodeError:
                header_text = raw_bytes.decode(errors='ignore')

        header_text_ascii = header_text.encode('ascii', errors='ignore').decode('ascii')
        wcs_text_norm = header_text_ascii.replace('\r\n', '\n').replace('\r', '\n')
        lines = wcs_text_norm.split('\n')
        lines_no_continue = [ln for ln in lines if not ln.lstrip().upper().startswith('CONTINUE')]
        cleaned_text = '\n'.join(lines_no_continue)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs_obj = None
            last_err = None
            try:
                wcs_obj = AstropyWCS(cleaned_text, naxis=2, relax=True)
            except Exception as e1:
                last_err = e1
                try:
                    hdr = fits.Header.fromstring(cleaned_text, sep='\n')
                    wcs_obj = AstropyWCS(hdr, naxis=2, relax=True)
                except Exception as e2:
                    last_err = e2

        # If still failing, try manual key/value parse to build a minimal WCS header
        if (not wcs_obj) or (not getattr(wcs_obj, 'is_celestial', False)):
            try:
                import re
                kv = {}
                for raw in lines_no_continue:
                    line = raw.strip()
                    if not line:
                        continue
                    up8 = line[:8].strip().upper()
                    if up8 in {"COMMENT", "HISTORY", "END", "CONTINUE"}:
                        continue
                    # Try KEY = VALUE / comment
                    m = re.match(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$", line)
                    if not m:
                        continue
                    key = m.group(1).upper()
                    rest = m.group(2)
                    # Strip inline comment
                    if "/" in rest:
                        rest = rest.split("/", 1)[0].rstrip()
                    rest = rest.strip()
                    # Unquote strings
                    if (len(rest) >= 2) and ((rest[0] == rest[-1]) and rest[0] in ("'", '"')):
                        val: object = rest[1:-1]
                    else:
                        # Convert FITS-like booleans and numbers (handle D exponents)
                        if rest.upper() in ("T", "F"):
                            val = True if rest.upper() == "T" else False
                        else:
                            rest_num = re.sub(r"([0-9])[dD]([+-]?[0-9]+)", r"\1E\2", rest)
                            try:
                                if "." in rest_num or "E" in rest_num.upper() or "-" in rest_num or "+" in rest_num:
                                    val = float(rest_num)
                                else:
                                    val = int(rest_num)
                            except Exception:
                                val = rest
                    kv[key] = val

                # Build a minimal FITS header with WCS keys only
                wcs_keys = [
                    "WCSAXES","CRPIX1","CRPIX2","CRVAL1","CRVAL2",
                    "CTYPE1","CTYPE2","CUNIT1","CUNIT2",
                    "CD1_1","CD1_2","CD2_1","CD2_2",
                    "PC1_1","PC1_2","PC2_1","PC2_2",
                    "CDELT1","CDELT2","CROTA2","CROTA1",
                    "LONPOLE","LATPOLE","EQUINOX","RADESYS",
                ]
                hdr_min = fits.Header()
                hdr_min["SIMPLE"] = True
                hdr_min["NAXIS"] = 2
                if sane_shape_hw:
                    hdr_min["NAXIS1"] = int(sane_shape_hw[1])
                    hdr_min["NAXIS2"] = int(sane_shape_hw[0])
                for k in wcs_keys:
                    if k in kv:
                        try:
                            hdr_min[k] = kv[k]
                        except Exception:
                            pass
                # If neither CD nor PC/CDELT present, try to infer from SCALE or PIXSCAL1
                if ("CD1_1" not in hdr_min) and ("PC1_1" not in hdr_min) and ("CDELT1" not in hdr_min):
                    for fallback_scale_key in ("SCALE", "PIXSCAL1"):
                        if fallback_scale_key in kv:
                            try:
                                s = float(kv[fallback_scale_key])
                                # assume square pixels, degrees per pixel if units unknown -> convert arcsec to deg if large
                                if s > 1e-3:  # likely arcsec/pix
                                    sdeg = s / 3600.0
                                else:
                                    sdeg = s
                                hdr_min["CDELT1"] = -sdeg
                                hdr_min["CDELT2"] = sdeg
                                hdr_min["CTYPE1"] = hdr_min.get("CTYPE1", "RA---TAN")
                                hdr_min["CTYPE2"] = hdr_min.get("CTYPE2", "DEC--TAN")
                            except Exception:
                                pass
                            break

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    wcs_obj = AstropyWCS(hdr_min, naxis=2, relax=True)
            except Exception as e_build:
                last_err = e_build

        if wcs_obj and getattr(wcs_obj, 'is_celestial', False):
            if sane_shape_hw and sane_shape_hw[0] > 0 and sane_shape_hw[1] > 0:
                try:
                    wcs_obj.pixel_shape = (int(sane_shape_hw[1]), int(sane_shape_hw[0]))
                except Exception as e_ps:
                    if progress_callback:
                        progress_callback(
                            f"    ASTAP WCS Parse AVERT: echec set pixel_shape: {e_ps}",
                            None,
                            "WARN",
                        )
            if progress_callback:
                progress_callback(
                    f"    ASTAP WCS Parse: Objet WCS parse (v2) avec succes depuis '{filename_log}'.",
                    None,
                    "DEBUG_DETAIL",
                )
            return wcs_obj

        if progress_callback:
            detail = str(last_err) if 'last_err' in locals() and last_err else 'WCS non valide'
            progress_callback(
                f"    ASTAP WCS Parse ERREUR: Echec creation WCS valide/celeste (v2) depuis '{filename_log}'. Detail: {detail}",
                None,
                "WARN",
            )
        return None
    except Exception as e:
        if progress_callback:
            progress_callback(
                f"    ASTAP WCS Parse ERREUR: Exception (v2) lors du parsing WCS '{filename_log}': {e}",
                None,
                "ERROR",
            )
        logger.error(f"Erreur parsing WCS v2 '{wcs_file_path}': {e}", exc_info=True)
        return None


def _update_fits_header_with_wcs_za(fits_header_to_update: fits.Header, 
                                   wcs_object_solution: AstropyWCS, 
                                   solver_name="ASTAP_ZeMosaic", 
                                   progress_callback=None):
    if not (fits_header_to_update is not None and wcs_object_solution and wcs_object_solution.is_celestial):
        if progress_callback: progress_callback("  ASTAP HeaderUpdate: MàJ header annulée: header/WCS invalide.", None, "WARN")
        return False 
    if progress_callback: progress_callback(f"  ASTAP HeaderUpdate: MàJ header FITS avec solution WCS de {solver_name}...", None, "DEBUG_DETAIL")
    if not ASTROPY_AVAILABLE_ASTROMETRY:
        if progress_callback: progress_callback("  ASTAP HeaderUpdate ERREUR: Astropy non disponible pour MàJ header.", None, "ERROR")
        return False
    try:
        wcs_keys_to_remove = [
            'WCSAXES', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 
            'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
            'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
            'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
            'CDELT1', 'CDELT2', 'CROTA1', 'CROTA2', 
            'LONPOLE', 'LATPOLE', 'EQUINOX', 'RADESYS',
            'PV1_0', 'PV1_1', 'PV1_2', 'PV2_0', 'PV2_1', 'PV2_2' 
        ]
        for key_del in wcs_keys_to_remove:
            if key_del in fits_header_to_update:
                try:
                    del fits_header_to_update[key_del]
                except KeyError:
                    pass
        
        # Correction de la coquille ici :
        new_wcs_header_cards = wcs_object_solution.to_header(relax=True) # Utiliser relax=True est plus simple et robuste
        
        fits_header_to_update.update(new_wcs_header_cards)
        fits_header_to_update[f'{solver_name.upper()}_SOLVED'] = (True, f'{solver_name} solution')
        
        if u is not None: # S'assurer que astropy.units est importé
            try:
                if hasattr(wcs_object_solution, 'proj_plane_pixel_scales') and callable(wcs_object_solution.proj_plane_pixel_scales):
                    scales_deg = wcs_object_solution.proj_plane_pixel_scales()
                    pixscale_arcsec = np.mean(np.abs(scales_deg.to_value(u.arcsec)))
                    fits_header_to_update[f'{solver_name.upper()}_PSCALE'] = (float(f"{pixscale_arcsec:.4f}"), f'[asec/pix] Scale from {solver_name} WCS')
            except Exception:
                pass
            
        if progress_callback: progress_callback("  ASTAP HeaderUpdate: Header FITS MàJ avec WCS.", None, "DEBUG_DETAIL")
        return True
    except Exception as e_upd:
        if progress_callback: progress_callback(f"  ASTAP HeaderUpdate ERREUR: {e_upd}", None, "ERROR")
        logger.error(f"Erreur MàJ header FITS avec WCS: {e_upd}", exc_info=True) # Log le traceback complet
        return False



# DANS zemosaic_astrometry.py

def solve_with_astap(
    image_fits_path: str,
    original_fits_header: fits.Header,
    astap_exe_path: str,
    astap_data_dir: str,
    search_radius_deg: float | None = None,    # Depuis GUI
    downsample_factor: int | None = None,      # Depuis GUI (pour -z)
    sensitivity: int | None = None,            # Depuis GUI (pour -sens)
    timeout_sec: int = 120,
    update_original_header_in_place: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    proc_hook: Optional[Callable[[subprocess.Popen], None]] = None,
):
    """Invoke ASTAP to solve the provided FITS file and stream its logging output.

    Parameters
    ----------
    proc_hook :
        Optional callable invoked with the created ``subprocess.Popen`` instance so
        external watchers (e.g., Qt helpers) can control or inspect the running solver.
    """

    if not ASTROPY_AVAILABLE_ASTROMETRY:
        if progress_callback: progress_callback("ASTAP Solve ERREUR: Astropy non disponible, ASTAP solve annulé.", None, "ERROR")
        return None

    if (
        os.environ.get("ZEMOSAIC_GUI_MODE") == "1"
        and threading.current_thread() is threading.main_thread()
    ):
        msg = (
            "ASTAP Solve ERREUR: appel depuis le thread GUI principal détecté. "
            "Le solver doit être exécuté dans un worker pour éviter tout gel."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    image_fits_path = _expand_path(image_fits_path)
    astap_exe_path = _expand_path(astap_exe_path)
    astap_data_dir = _expand_path(astap_data_dir)

    img_basename_log = _path_display_name(image_fits_path)
    if progress_callback:
        progress_callback(f"ASTAP Solve: Début pour '{img_basename_log}'", None, "INFO_DETAIL")
    logger.debug(f"ASTAP Solve params (entrée fonction): image='{img_basename_log}', radius={search_radius_deg}, "
                 f"downsample={downsample_factor}, sensitivity={sensitivity}")

    if not (astap_exe_path and astap_exe_path.is_file()):
        if progress_callback:
            progress_callback(
                f"ASTAP Solve ERREUR: Chemin ASTAP exe invalide: '{astap_exe_path}'.",
                None,
                "ERROR",
            )
        return None
    if not (image_fits_path and image_fits_path.is_file()):
        if progress_callback:
            progress_callback(
                f"ASTAP Solve ERREUR: Chemin image FITS invalide: '{image_fits_path}'.",
                None,
                "ERROR",
            )
        return None
    if not (astap_data_dir and astap_data_dir.is_dir()):
        if progress_callback:
            progress_callback(
                f"ASTAP Solve AVERT: Chemin ASTAP data non spécifié ou invalide: '{astap_data_dir}'. ASTAP pourrait ne pas trouver ses bases.",
                None,
                "WARN",
            )
    if original_fits_header is None: # Should not happen if called from worker
        if progress_callback: progress_callback("ASTAP Solve ERREUR: Header FITS original non fourni.", None, "ERROR")
        return None

    current_image_dir = image_fits_path.parent
    base_image_name_no_ext = image_fits_path.stem
    expected_wcs_file_path = current_image_dir / f"{base_image_name_no_ext}.wcs"
    expected_ini_file_path = current_image_dir / f"{base_image_name_no_ext}.ini"
    astap_log_file_path = current_image_dir / f"{base_image_name_no_ext}.log"
    files_to_cleanup_by_astap = [expected_wcs_file_path, expected_ini_file_path]

    for f_to_clean in files_to_cleanup_by_astap:
        try:
            if f_to_clean.exists():
                f_to_clean.unlink()
        except Exception as e_del_pre:
            if progress_callback:
                progress_callback(
                    f"  ASTAP Solve AVERT: Échec nettoyage pré-ASTAP '{_path_display_name(f_to_clean)}': {e_del_pre}",
                    None,
                    "WARN",
                )
    try:
        if astap_log_file_path.exists():
            astap_log_file_path.unlink()
    except Exception as e_del_log_pre:
        if progress_callback:
            progress_callback(
                f"  ASTAP Solve AVERT: Échec nettoyage pré-ASTAP log '{_path_display_name(astap_log_file_path)}': {e_del_log_pre}",
                None,
                "WARN",
            )

    cmd_list_astap = [str(astap_exe_path), "-f", str(image_fits_path), "-log", "-wcs"]
    if astap_data_dir and astap_data_dir.is_dir():
        cmd_list_astap.extend(["-d", str(astap_data_dir)])

    # Rétablit l'indication d'échelle ou FOV pour aider ASTAP
    calculated_px_scale = _calculate_pixel_scale_from_header(original_fits_header, progress_callback)
    if calculated_px_scale and 0.01 < float(calculated_px_scale) < 50.0:
        cmd_list_astap.extend(["-pxscale", f"{float(calculated_px_scale):.4f}"])
        if progress_callback:
            progress_callback(
                f"  ASTAP Solve: Utilisation -pxscale {float(calculated_px_scale):.4f} arcsec/pix (dérivé du header).",
                None,
                "DEBUG",
            )
    else:
        # Blind solve: demande à ASTAP d'estimer le FOV automatiquement
        cmd_list_astap.extend(["-fov", "0"])
        if progress_callback:
            progress_callback(
                "  ASTAP Solve: -pxscale indisponible. Utilisation -fov 0 (auto).",
                None,
                "DEBUG",
            )


    # Gestion du downsampling (-z)
    if downsample_factor is not None and isinstance(downsample_factor, int) and downsample_factor >= 0:
        cmd_list_astap.extend(["-z", str(downsample_factor)])
        if progress_callback: progress_callback(f"  ASTAP Solve: Utilisation -z {downsample_factor} (configuré).", None, "DEBUG")
    else:
        if progress_callback: progress_callback(f"  ASTAP Solve: Downsample non spécifié ou invalide ({downsample_factor}). ASTAP utilisera son défaut pour -z.", None, "DEBUG_DETAIL")

    # Sensibilité: l'option "-sens" n'est pas documentée côté CLI ASTAP.
    # Par sécurité, on ignore ce paramètre pour éviter un échec immédiat du binaire.
    if sensitivity is not None and progress_callback:
        progress_callback(
            f"  ASTAP Solve: Paramètre 'sensibilité' reçu ({sensitivity}) mais ignoré (option CLI non supportée).",
            None,
            "DEBUG_DETAIL",
        )

    # Ajoute les hints RA/Dec si disponibles dans le header FITS
    has_pos_hint = False
    try:
        ra_hours_hint = None
        spd_deg_hint = None

        def _safe_float(x):
            try:
                return float(x)
            except Exception:
                return None

        # RA/DEC candidats (ordre de priorité)
        ra_candidates = [
            original_fits_header.get("CRVAL1"),
            original_fits_header.get("RA"),
            original_fits_header.get("OBJCTRA"),
            original_fits_header.get("TELRA"),
        ]
        dec_candidates = [
            original_fits_header.get("CRVAL2"),
            original_fits_header.get("DEC"),
            original_fits_header.get("OBJCTDEC"),
            original_fits_header.get("TELDEC"),
        ]

        # Keep CRVAL1 to disambiguate units: CRVAL1 is always in degrees (WCS)
        crval1_val = original_fits_header.get("CRVAL1")
        ra_val = next((v for v in ra_candidates if v is not None), None)
        dec_val = next((v for v in dec_candidates if v is not None), None)

        ra_deg = None
        dec_deg = None

        # Essaie d'interpréter RA
        if isinstance(ra_val, (int, float)):
            # If value comes from CRVAL1 (WCS), it is degrees even if <= 24
            try:
                same_as_crval1 = (crval1_val is not None and float(ra_val) == float(crval1_val))
            except Exception:
                same_as_crval1 = False
            if same_as_crval1:
                ra_deg = float(ra_val)
            else:
                # Otherwise: convert to degrees only if explicit unit is hours
                try:
                    unit = (original_fits_header.get("RAUNIT") or original_fits_header.get("CUNIT1") or "").strip().lower()
                except Exception:
                    unit = ""
                if unit.startswith("h"):
                    ra_deg = float(ra_val) * 15.0
                else:
                    # Default: treat numeric RA as degrees
                    ra_deg = float(ra_val)
        elif isinstance(ra_val, str):
            # Essaye format HH:MM:SS[.s] => heures
            s = ra_val.strip().lower().replace("h", ":").replace("m", ":").replace("s", "").replace(" ", ":")
            parts = [p for p in s.split(":") if p != ""]
            if 1 <= len(parts) <= 3 and all(p.replace(".", "", 1).lstrip("-+").isdigit() for p in parts):
                hh = _safe_float(parts[0]) or 0.0
                mm = _safe_float(parts[1]) or 0.0 if len(parts) > 1 else 0.0
                ss = _safe_float(parts[2]) or 0.0 if len(parts) > 2 else 0.0
                ra_deg = (hh + mm / 60.0 + ss / 3600.0) * 15.0
            else:
                # Dernier recours: nombre brut
                val = _safe_float(ra_val)
                if val is not None:
                    ra_deg = float(val)

        # Essaie d'interpréter DEC (toujours en degrés)
        if isinstance(dec_val, (int, float)):
            dec_deg = float(dec_val)
        elif isinstance(dec_val, str):
            s = dec_val.strip().lower().replace("d", ":").replace(" ", ":")
            parts = [p for p in s.split(":") if p != ""]
            if 1 <= len(parts) <= 3 and all(p.replace(".", "", 1).lstrip("-+").isdigit() for p in parts):
                dd = _safe_float(parts[0]) or 0.0
                mm = _safe_float(parts[1]) or 0.0 if len(parts) > 1 else 0.0
                ss = _safe_float(parts[2]) or 0.0 if len(parts) > 2 else 0.0
                sign = -1.0 if str(parts[0]).strip().startswith("-") else 1.0
                dec_deg = sign * (abs(dd) + mm / 60.0 + ss / 3600.0)
            else:
                val = _safe_float(dec_val)
                if val is not None:
                    dec_deg = float(val)

        if ra_deg is not None and dec_deg is not None:
            # Convertit en -ra (heures) et -spd (dec+90)
            ra_hours_hint = ra_deg / 15.0
            spd_deg_hint = dec_deg + 90.0
            # Contrainte domaines attendus
            if ra_hours_hint < 0.0:
                ra_hours_hint = (ra_hours_hint % 24.0)
            if spd_deg_hint < 0.0:
                spd_deg_hint = 0.0
            if spd_deg_hint > 180.0:
                spd_deg_hint = 180.0

        if ra_hours_hint is not None and spd_deg_hint is not None:
            has_pos_hint = True
            cmd_list_astap.extend(["-ra", f"{ra_hours_hint:.6f}", "-spd", f"{spd_deg_hint:.6f}"])
            if progress_callback:
                progress_callback(
                    f"  ASTAP Solve: Hints -ra {ra_hours_hint:.6f} h, -spd {spd_deg_hint:.6f}° (depuis header).",
                    None,
                    "DEBUG",
                )
        else:
            if progress_callback:
                progress_callback(
                    "  ASTAP Solve: Hints RA/Dec absents ou invalides dans le header (résolution 'blind').",
                    None,
                    "DEBUG_DETAIL",
                )
    except Exception as e_hints:
        if progress_callback:
            progress_callback(f"  ASTAP Solve: Erreur extraction RA/Dec du header: {e_hints}", None, "WARN")

    # Gestion du rayon de recherche (-r) uniquement s'il y a un hint de position
    if has_pos_hint and search_radius_deg is not None and search_radius_deg > 0:
        cmd_list_astap.extend(["-r", str(search_radius_deg)])
        if progress_callback:
            progress_callback(
                f"  ASTAP Solve: Utilisation -r {search_radius_deg}° (autour du hint RA/Dec).",
                None,
                "DEBUG",
            )
    else:
        if progress_callback:
            progress_callback(
                "  ASTAP Solve: Aucun hint de position fiable; omission de -r pour un blind-solve.",
                None,
                "DEBUG_DETAIL",
            )


    if progress_callback: progress_callback(f"  ASTAP Solve: Commande: {' '.join(cmd_list_astap)}", None, "DEBUG")
    logger.info(f"Executing ASTAP for {img_basename_log}: {' '.join(cmd_list_astap)}")
    wcs_solved_obj = None
    astap_success = False

    try:
        max_retries = max(0, int(os.environ.get("ZEMOSAIC_ASTAP_RETRIES", "2") or 2))
    except Exception:
        max_retries = 2
    try:
        base_backoff = max(
            0.1, float(os.environ.get("ZEMOSAIC_ASTAP_BACKOFF_SEC", "0.8") or 0.8)
        )
    except Exception:
        base_backoff = 0.8
    total_attempts = max_retries + 1

    try:
        rc_astap = None
        with _lock_for_image(image_fits_path, img_basename_log, progress_callback):
            attempt = 0
            last_error: Exception | None = None
            while attempt < total_attempts:
                attempt += 1
                try:
                    if total_attempts > 1 and progress_callback:
                        progress_callback(
                            f"  ASTAP Solve: tentative {attempt}/{total_attempts}.",
                            None,
                            "DEBUG_DETAIL",
                        )
                    with _astap_ipc_slot_guard(img_basename_log, progress_callback):
                        with _astap_slot_guard(img_basename_log):
                            _rate_limit_before_launch(progress_callback)
                            with _temp_cwd_isolated() as isolated_cwd:
                                rc_astap = _run_astap_once(
                                cmd_list_astap,
                                timeout_sec,
                                progress_callback,
                                cwd=isolated_cwd,
                                proc_ready_callback=proc_hook,
                            )
                    if rc_astap == 0:
                        break
                    raise RuntimeError(f"ASTAP return code {rc_astap}")
                except subprocess.TimeoutExpired as exc_timeout:
                    last_error = exc_timeout
                    if attempt >= total_attempts:
                        raise
                    wait = base_backoff * attempt
                    if progress_callback:
                        progress_callback(
                            f"  ASTAP Solve: timeout (tentative {attempt}/{total_attempts}). "
                            f"Nouvel essai dans {wait:.1f}s.",
                            None,
                            "WARN",
                        )
                    time.sleep(wait)
                except Exception as exc_generic:
                    last_error = exc_generic
                    if attempt >= total_attempts:
                        raise
                    wait = base_backoff * attempt
                    if progress_callback:
                        progress_callback(
                            f"  ASTAP Solve: échec tentative {attempt}/{total_attempts} ({exc_generic}). "
                            f"Nouvel essai dans {wait:.1f}s.",
                            None,
                            "WARN",
                        )
                    time.sleep(wait)
            if rc_astap is None and last_error:
                raise last_error

        logger.debug("ASTAP return code: %s", rc_astap)
        gc.collect()
        _log_memory_usage(progress_callback, "Après GC post-ASTAP")

        if rc_astap == 0:
            if safe_path_exists(expected_wcs_file_path) and safe_path_getsize(expected_wcs_file_path) > 0:
                if progress_callback: progress_callback(f"  ASTAP Solve: Résolution OK (code 0). Fichier WCS '{_path_display_name(expected_wcs_file_path)}' trouvé.", None, "INFO_DETAIL")
                img_height = original_fits_header.get('NAXIS2', 0)
                img_width = original_fits_header.get('NAXIS1', 0)
                if img_height == 0 or img_width == 0:
                    try:
                        with fits.open(image_fits_path) as hdul_shape:
                            shape_from_file = hdul_shape[0].shape
                            if len(shape_from_file) >=2 :
                                img_height = shape_from_file[-2]
                                img_width = shape_from_file[-1]
                    except Exception as e_shape_read:
                         if progress_callback: progress_callback(f"  ASTAP Solve AVERT: Impossible de lire NAXIS1/2 du header ou du fichier FITS: {e_shape_read}. WCS parsing pourrait échouer.", None, "WARN")
                if img_height > 0 and img_width > 0:
                    # Utiliser le parseur robuste v2 (gestion encodage/CONTINUE)
                    wcs_solved_obj = _parse_wcs_file_content_za_v2(
                        expected_wcs_file_path, (img_height, img_width), progress_callback
                    )
                else:
                    if progress_callback: progress_callback(f"  ASTAP Solve ERREUR: Dimensions image (NAXIS1/2) non trouvées pour '{img_basename_log}'. WCS non parsé.", None, "ERROR")
                if wcs_solved_obj and wcs_solved_obj.is_celestial:
                    astap_success = True
                    if progress_callback: progress_callback(f"  ASTAP Solve: Objet WCS créé et céleste pour '{img_basename_log}'.", None, "INFO")
                    if update_original_header_in_place and original_fits_header is not None:
                        if _update_fits_header_with_wcs_za(original_fits_header, wcs_solved_obj, progress_callback=progress_callback):
                             if progress_callback: progress_callback(f"  ASTAP Solve: Header FITS original mis à jour avec WCS pour '{img_basename_log}'.", None, "DEBUG_DETAIL")
                        else:
                             if progress_callback: progress_callback(f"  ASTAP Solve AVERT: Échec MàJ header FITS original avec WCS pour '{img_basename_log}'.", None, "WARN")
                else:
                    if progress_callback: progress_callback(f"  ASTAP Solve ERREUR: WCS parsé non valide ou non céleste pour '{img_basename_log}'.", None, "ERROR")
                    wcs_solved_obj = None
            else:
                if progress_callback: progress_callback(f"  ASTAP Solve ERREUR: Code 0 mais fichier .wcs manquant/vide ('{_path_display_name(expected_wcs_file_path)}').", None, "ERROR")
        else:
            error_msg = f"ASTAP Solve Échec (code {rc_astap}) pour '{img_basename_log}'."
            if rc_astap == 1: error_msg += " (No solution found)."
            elif rc_astap == 2: error_msg += " (ASTAP FITS read error - vérifiez format/corruption)."
            elif rc_astap == 10: error_msg += " (ASTAP database not found - vérifiez -d)."
            if progress_callback: progress_callback(f"  {error_msg}", None, "WARN")
            logger.warning(error_msg)

    except subprocess.TimeoutExpired:
        if progress_callback: progress_callback(f"ASTAP Solve ERREUR: Timeout ({timeout_sec}s) pour '{img_basename_log}'.", None, "ERROR")
        logger.error(f"ASTAP command timed out for {img_basename_log}", exc_info=False)
    except FileNotFoundError:
        if progress_callback: progress_callback(f"ASTAP Solve ERREUR: Exécutable ASTAP '{astap_exe_path}' non trouvé.", None, "ERROR")
        logger.error(f"ASTAP executable not found at '{astap_exe_path}'.", exc_info=False)
    except Exception as e_astap_glob:
        if progress_callback: progress_callback(f"ASTAP Solve ERREUR Inattendue: {e_astap_glob}", None, "ERROR")
        logger.error(f"Unexpected error during ASTAP execution for {img_basename_log}: {e_astap_glob}", exc_info=True)
    finally:
        if progress_callback: progress_callback(f"  ASTAP Solve: Nettoyage post-exécution (sauf log si échec) pour '{img_basename_log}'...", None, "DEBUG_DETAIL")
        for f_clean_post in files_to_cleanup_by_astap: # .wcs, .ini
            if safe_path_exists(f_clean_post):
                try:
                    if f_clean_post == expected_wcs_file_path and astap_success and not update_original_header_in_place:
                        if progress_callback: progress_callback(f"    ASTAP Clean: Conservation du .wcs: {_path_display_name(f_clean_post)} (succès, pas de MàJ header en place)", None, "DEBUG_DETAIL")
                        continue
                    os.remove(f_clean_post)
                    if progress_callback: progress_callback(f"    ASTAP Clean: Fichier '{_path_display_name(f_clean_post)}' supprimé.", None, "DEBUG_DETAIL")
                except Exception as e_del_post:
                    if progress_callback: progress_callback(f"    ASTAP Clean AVERT: Échec nettoyage '{_path_display_name(f_clean_post)}': {e_del_post}", None, "WARN")

        if astap_success and safe_path_exists(astap_log_file_path):
            try:
                os.remove(astap_log_file_path)
                if progress_callback: progress_callback(f"    ASTAP Clean: Fichier log ASTAP '{_path_display_name(astap_log_file_path)}' supprimé (succès).", None, "DEBUG_DETAIL")
            except Exception as e_del_log_succ:
                if progress_callback: progress_callback(f"    ASTAP Clean AVERT: Échec nettoyage log ASTAP (succès) '{_path_display_name(astap_log_file_path)}': {e_del_log_succ}", None, "WARN")
        elif not astap_success and safe_path_exists(astap_log_file_path):
             if progress_callback: progress_callback(f"    ASTAP Clean: CONSERVATION du log ASTAP '{_path_display_name(astap_log_file_path)}' (échec solve).", None, "INFO_DETAIL")

        gc.collect()

    if wcs_solved_obj:
        if progress_callback: progress_callback(f"ASTAP Solve: WCS trouvé pour {img_basename_log}.", None, "INFO_DETAIL")
    else:
        if progress_callback: progress_callback(f"ASTAP Solve: Pas de WCS final pour {img_basename_log}.", None, "WARN")
    return wcs_solved_obj


def solve_with_ansvr(
    image_fits_path: str,
    original_fits_header: fits.Header,
    ansvr_config_path: str,
    timeout_sec: int = 120,
    update_original_header_in_place: bool = False,
    progress_callback: callable = None,
):
    """Solve WCS using a local ansvr installation."""

    _pcb = (
        lambda msg, lvl="INFO": progress_callback(msg, None, lvl)
        if progress_callback
        else None
    )

    if not (ASTROPY_AVAILABLE_ASTROMETRY and AstropyWCS):
        if _pcb:
            _pcb("Ansvr solve unavailable (missing deps).", "ERROR")
        return None

    image_path = _expand_path(image_fits_path)
    ansvr_config = _expand_path(ansvr_config_path)

    if not (image_path and image_path.is_file() and ansvr_config and ansvr_config.exists()):
        if _pcb:
            _pcb("Ansvr solve input invalid or path missing.", "ERROR")
        return None

    tmp_dir = Path(tempfile.mkdtemp(prefix="ansvr_"))
    output_fits = tmp_dir / "solution.new"

    cmd = [
        "solve-field",
        "--no-plots",
        "--overwrite",
        "--config",
        str(ansvr_config),
        "--dir",
        str(tmp_dir),
        "--new-fits",
        str(output_fits),
        str(image_path),
    ]

    if original_fits_header:
        ra = original_fits_header.get("RA", original_fits_header.get("CRVAL1"))
        dec = original_fits_header.get("DEC", original_fits_header.get("CRVAL2"))
        if isinstance(ra, (int, float)) and isinstance(dec, (int, float)):
            cmd.extend(["--ra", str(ra), "--dec", str(dec)])

    if _pcb:
        _pcb(f"Ansvr: cmd={' '.join(cmd)}", "DEBUG")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_sec,
            check=False,
        )
    except Exception as e_run:
        if _pcb:
            _pcb(f"Ansvr: execution error {e_run}", "ERROR")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    if result.returncode != 0 or not safe_path_exists(output_fits):
        if _pcb:
            _pcb("Ansvr: solve-field failed", "WARN")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    try:
        with fits.open(output_fits, memmap=False) as hdul:
            wcs_header = hdul[0].header
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs_obj = AstropyWCS(wcs_header, naxis=2)
    except Exception as e_parse:
        if _pcb:
            _pcb(f"Ansvr: parse error {e_parse}", "ERROR")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    if wcs_obj and wcs_obj.is_celestial and update_original_header_in_place and original_fits_header is not None:
        _update_fits_header_with_wcs_za(
            original_fits_header,
            wcs_obj,
            solver_name="Ansvr",
            progress_callback=progress_callback,
        )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return wcs_obj




@dataclass(frozen=True)
class AstapTask:
    """Parameters describing a single ASTAP solve request."""

    image_fits_path: str
    original_header: Optional[fits.Header]
    astap_executable_path: str
    astap_data_directory: str
    search_radius_deg: Optional[float] = None
    downsample_factor: Optional[int] = None
    sensitivity: Optional[int] = None
    timeout_sec: int = 120
    update_header_in_place: bool = False
    progress_callback: Optional[ProgressCallback] = None
    completion_callback: Optional[Callable[[Any | None, Optional[Exception]], None]] = None


class AstapTaskManager:
    """Queue multiple ASTAP solves and run them sequentially on background threads."""

    def __init__(self) -> None:
        self._pending: Deque[AstapTask] = deque()
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._current_proc: Optional[subprocess.Popen] = None
        self._shutdown = False

    def submit(self, task: AstapTask) -> None:
        with self._lock:
            self._pending.append(task)
            worker_alive = self._worker_thread is not None and self._worker_thread.is_alive()
        if not worker_alive:
            self._start_next()

    def cancel(self) -> None:
        with self._lock:
            self._pending.clear()
            proc = self._current_proc
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            self._pending.clear()
            worker = self._worker_thread
            proc = self._current_proc
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        if worker is not None:
            worker.join(timeout=0.5)

    def _start_next(self) -> None:
        with self._lock:
            if self._shutdown or self._worker_thread is not None and self._worker_thread.is_alive():
                return
            if not self._pending:
                return
            task = self._pending.popleft()
            thread = threading.Thread(target=self._run_task, args=(task,), daemon=True)
            self._worker_thread = thread
        thread.start()

    def _run_task(self, task: AstapTask) -> None:
        result: Any | None = None
        error: Optional[Exception] = None

        def _capture_proc(proc: subprocess.Popen) -> None:
            with self._lock:
                self._current_proc = proc

        try:
            result = solve_with_astap(
                task.image_fits_path,
                task.original_header,
                task.astap_executable_path,
                task.astap_data_directory,
                search_radius_deg=task.search_radius_deg,
                downsample_factor=task.downsample_factor,
                sensitivity=task.sensitivity,
                timeout_sec=task.timeout_sec,
                update_original_header_in_place=task.update_header_in_place,
                progress_callback=task.progress_callback,
                proc_hook=_capture_proc,
            )
        except Exception as exc:  # pragma: no cover - defensive
            error = exc
        finally:
            with self._lock:
                self._current_proc = None
                self._worker_thread = None
            callback = task.completion_callback
            if callback is not None:
                try:
                    callback(result, error)
                except Exception:
                    logger.exception("AstapTask completion callback raised an exception")
            if not self._shutdown:
                self._start_next()


_ASTAP_TASK_MANAGER = AstapTaskManager()


def submit_astap_task(task: AstapTask) -> None:
    """Schedule an ASTAP solve to run as soon as the previous task completes."""

    _ASTAP_TASK_MANAGER.submit(task)


def cancel_astap_tasks() -> None:
    """Cancel any pending ASTAP solves and kill the running process."""

    _ASTAP_TASK_MANAGER.cancel()


def shutdown_astap_task_manager() -> None:
    """Stop the ASTAP task manager and ensure no background threads remain."""

    _ASTAP_TASK_MANAGER.shutdown()


def solve_with_astrometry_net(
    image_fits_path: str,
    original_fits_header: fits.Header,
    api_key: str,
    timeout_sec: int = 60,
    scale_est_arcsec_per_pix: float | None = None,
    scale_tolerance_percent: float = 20.0,
    downsample_factor: int | None = None,
    update_original_header_in_place: bool = False,
    progress_callback: callable = None,
):
    """Solve WCS using the astrometry.net web service."""

    _pcb = (
        lambda msg, lvl="INFO": progress_callback(msg, None, lvl)
        if progress_callback
        else None
    )

    if not (
        ASTROPY_AVAILABLE_ASTROMETRY
        and ASTROQUERY_AVAILABLE_ASTROMETRY
        and AstrometryNet
        and AstropyWCS
    ):
        if _pcb:
            missing = []
            if not (ASTROPY_AVAILABLE_ASTROMETRY and AstropyWCS):
                missing.append("astropy")
            if not (ASTROQUERY_AVAILABLE_ASTROMETRY and AstrometryNet):
                missing.append("astroquery")
            joined = ", ".join(missing) if missing else "unknown"
            _pcb(
                f"Astrometry.net solve unavailable (missing deps: {joined}).",
                "ERROR",
            )
        return None

    if not api_key:
        api_key = os.environ.get("ASTROMETRY_API_KEY", "")
    image_path = _expand_path(image_fits_path)
    if not (image_path and image_path.is_file()) or not api_key:
        if _pcb:
            path_ok = bool(image_path and image_path.is_file())
            key_len = len(api_key) if isinstance(api_key, str) else 0

            preview = (
                f"{api_key[:4]}..." if isinstance(api_key, str) and key_len > 4 else api_key
            )
            _pcb(
                "Astrometry.net solve input invalid or API key missing. "
                f"Path ok={path_ok} Key len={key_len} Preview='{preview}'",

                "ERROR",
            )
        return None

    img_basename_log = _path_display_name(image_path)
    if _pcb:
        _pcb(f"WebANET: Début solving pour '{img_basename_log}'", "INFO_DETAIL")

    ast = AstrometryNet()
    ast.api_key = api_key
    original_timeout = None
    if timeout_sec and timeout_sec > 0:
        try:
            original_timeout = getattr(ast, "TIMEOUT", None)
            ast.TIMEOUT = timeout_sec
            if _pcb:
                _pcb(f"WebANET: Timeout configuré à {timeout_sec}s", "DEBUG")
        except Exception as e:
            if _pcb:
                _pcb(f"WebANET: Erreur configuration timeout: {e}", "WARN")

    temp_path = None
    wcs_header = None
    try:
        with fits.open(str(image_path), memmap=False) as hdul:
            data = hdul[0].data

        if data is None:
            if _pcb:
                _pcb("WebANET: Données image manquantes", "ERROR")
            return None

        if not np.all(np.isfinite(data)):
            data = np.nan_to_num(data)

        if data.ndim == 3:
            # Convert color images to a single channel luminance image expected by
            # astrometry.net.  Some FITS files store the RGB channels in the first
            # axis (3, H, W) while others use the last axis (H, W, 3).  Handle both
            # cases gracefully.
            if data.shape[0] == 3:
                data = np.moveaxis(data, 0, -1)

            if data.shape[-1] == 3:
                data = np.sum(
                    data
                    * np.array([0.299, 0.587, 0.114], dtype=np.float32).reshape(
                        1, 1, 3
                    ),
                    axis=2,
                )
            else:
                # Fallback: take the first plane if the dimensionality is
                # unexpected.  This avoids passing a 3D array to photutils which
                # would raise an error.
                data = data[..., 0]

        if downsample_factor and isinstance(downsample_factor, int) and downsample_factor > 1:
            data = data[::downsample_factor, ::downsample_factor]

        min_v, max_v = np.min(data), np.max(data)
        norm_float = (data - min_v) / (max_v - min_v) if max_v > min_v else np.zeros_like(data)
        data_uint16 = (np.clip(norm_float, 0.0, 1.0) * 65535.0).astype(np.uint16)
        data_int16 = (data_uint16.astype(np.int32) - 32768).astype(np.int16)

        header_tmp = fits.Header()
        header_tmp["SIMPLE"] = True
        header_tmp["BITPIX"] = 16
        header_tmp["BSCALE"] = 1
        header_tmp["BZERO"] = 32768
        header_tmp["NAXIS"] = 2
        header_tmp["NAXIS1"] = data_int16.shape[1]
        header_tmp["NAXIS2"] = data_int16.shape[0]
        for key in ["OBJECT", "DATE-OBS", "EXPTIME", "FILTER", "INSTRUME", "TELESCOP"]:
            if original_fits_header and key in original_fits_header:
                header_tmp[key] = original_fits_header[key]

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False, mode="wb") as tf:
            temp_path = Path(tf.name)
        fits.writeto(
            str(temp_path),
            data_int16,
            header=header_tmp,
            overwrite=True,
            output_verify="silentfix",
        )

        if header_tmp.get("BITPIX") == 16:
            with fits.open(str(temp_path), mode="update", memmap=False) as hdul_fix:
                hd0 = hdul_fix[0]
                hd0.header["BSCALE"] = 1
                hd0.header["BZERO"] = 32768
                hdul_fix.flush()

        solve_args = {
            "allow_commercial_use": "n",
            "allow_modifications": "n",
            "publicly_visible": "n",
        }

        if scale_est_arcsec_per_pix and scale_est_arcsec_per_pix > 0:
            try:
                est_val = float(scale_est_arcsec_per_pix)
                tol_val = float(scale_tolerance_percent)
                solve_args["scale_units"] = "arcsecperpix"
                solve_args["scale_lower"] = est_val * (1.0 - tol_val / 100.0)
                solve_args["scale_upper"] = est_val * (1.0 + tol_val / 100.0)
                if _pcb:
                    _pcb(
                        f"WebANET: Échelle contraint [{solve_args['scale_lower']:.2f}-{solve_args['scale_upper']:.2f}] asec/pix",
                        "DEBUG",
                    )
            except Exception as e:
                if _pcb:
                    _pcb(f"WebANET: Erreur config échelle: {e}", "WARN")

        if _pcb:
            _pcb("WebANET: Soumission du job...", "INFO")

            _pcb("WebANET: Contacting nova.astrometry.net", "DEBUG")
        wcs_header = ast.solve_from_image(str(temp_path), **solve_args)

        if _pcb:
            if wcs_header:
                _pcb("WebANET: Solving RÉUSSI", "INFO")
            else:
                _pcb("WebANET: Solving ÉCHOUÉ", "WARN")

    except Exception as e:
        if _pcb:
            msg = f"WebANET: ERREUR solving: {e}"
            if "timeout" in str(e).lower():
                _pcb(msg, "ERROR")
            else:
                _pcb(msg, "ERROR")
        logger.error("Astrometry.net solve exception", exc_info=True)
        wcs_header = None
    finally:
        if temp_path:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
        if original_timeout is not None:
            try:
                ast.TIMEOUT = original_timeout
            except Exception:
                pass

    if not isinstance(wcs_header, fits.Header):
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs_obj = AstropyWCS(wcs_header)
    except Exception as e:
        if _pcb:
            _pcb(f"WebANET: ERREUR conversion WCS: {e}", "ERROR")
        return None

    if (
        wcs_obj
        and wcs_obj.is_celestial
        and update_original_header_in_place
        and original_fits_header is not None
    ):
        _update_fits_header_with_wcs_za(
            original_fits_header,
            wcs_obj,
            solver_name="AstrometryNet",
            progress_callback=progress_callback,
        )

    return wcs_obj
