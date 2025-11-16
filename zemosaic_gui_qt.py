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


PySide6-based ZeMosaic main window.

The Qt interface is optional and can be launched from the regular
``run_zemosaic.py`` entry point by either setting the environment
variable ``ZEMOSAIC_GUI_BACKEND=qt`` or by passing the ``--qt-gui``
command-line flag. When neither of those are supplied the application
falls back to the classic Tk interface, ensuring existing workflows
continue to operate without PySide6. Passing ``--tk-gui`` explicitly
selects the Tk interface even if the environment variable requests Qt.
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import multiprocessing
import queue
import threading
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

try:
    from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
    from PySide6.QtGui import (
        QBrush,
        QCloseEvent,
        QColor,
        QIcon,
        QPainter,
        QPalette,
        QPen,
        QResizeEvent,
        QTextCharFormat,
        QTextCursor,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFrame,
        QGraphicsScene,
        QGraphicsView,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Unable to import PySide6 which is required for the ZeMosaic Qt interface. "
        "Install the optional dependency with `pip install PySide6` or continue "
        "using the Tk interface."
    ) from exc

SYSTEM_NAME = platform.system().lower()
IS_WINDOWS = SYSTEM_NAME == "windows"

if IS_WINDOWS:
    try:  # pragma: no cover - optional dependency for GPU detection
        import wmi  # type: ignore
    except ImportError:  # pragma: no cover - wmi unavailable on non-Windows
        wmi = None  # type: ignore[assignment]
else:
    wmi = None  # type: ignore[assignment]

CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None

if importlib.util.find_spec("zemosaic_config") is not None:
    import zemosaic_config  # type: ignore
else:  # pragma: no cover - optional dependency guard
    zemosaic_config = None  # type: ignore[assignment]

if importlib.util.find_spec("solver_settings") is not None:
    from solver_settings import SolverSettings  # type: ignore
else:  # pragma: no cover - optional dependency guard
    SolverSettings = None  # type: ignore[assignment]

if importlib.util.find_spec("zemosaic_worker") is not None:
    from zemosaic_worker import run_hierarchical_mosaic_process  # type: ignore
else:  # pragma: no cover - optional dependency guard
    run_hierarchical_mosaic_process = None  # type: ignore[assignment]

if importlib.util.find_spec("locales.zemosaic_localization") is not None:
    from locales.zemosaic_localization import ZeMosaicLocalization  # type: ignore
else:  # pragma: no cover - optional dependency guard
    ZeMosaicLocalization = None  # type: ignore[assignment]

if importlib.util.find_spec("zemosaic_astrometry") is not None:
    from zemosaic_astrometry import set_astap_max_concurrent_instances  # type: ignore
else:  # pragma: no cover - optional dependency guard
    set_astap_max_concurrent_instances = None  # type: ignore[assignment]


# Phase 4.5 / super-tiles configuration UI is intentionally hidden for this release.
# The worker-side implementation and overlays remain available, but users must not
# be able to toggle Phase 4.5 from the Qt main window.
ENABLE_PHASE45_UI = False

LANGUAGE_OPTION_DEFINITIONS = [
    ("en", "language_name_en", "English (EN)"),
    ("fr", "language_name_fr", "Français (FR)"),
    ("es", "language_name_es", "Español (ES)"),
    ("pl", "language_name_pl", "Polski (PL)"),
]


def _load_zemosaic_qicon() -> QIcon | None:
    """Return a QIcon for ZeMosaic using the best available icon file."""
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


def _coerce_gpu_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        if not normalized:
            return default
    if value is None:
        return default
    return bool(value)


def _ensure_legacy_gpu_defaults_on_config_module() -> None:
    if zemosaic_config is None:
        return
    defaults = getattr(zemosaic_config, "DEFAULT_CONFIG", None)
    if not isinstance(defaults, dict):
        return

    defaults["use_gpu_phase5"] = _coerce_gpu_bool(defaults.get("use_gpu_phase5"), False)
    canonical = defaults["use_gpu_phase5"]
    for key in ("stack_use_gpu", "use_gpu_stack"):
        defaults[key] = _coerce_gpu_bool(defaults.get(key, canonical), canonical)


_ensure_legacy_gpu_defaults_on_config_module()


class _FallbackLocalizer:
    """Very small localization shim when the real helper is unavailable."""

    def __init__(self, language_code: str = "en") -> None:
        self.language_code = language_code

    def get(self, key: str, default_text: str | None = None, **_: Any) -> str:
        return default_text if default_text is not None else key

    def set_language(self, language_code: str) -> None:
        self.language_code = language_code


class _WorkerQueueListener(QObject):
    """Background queue listener that runs inside a dedicated ``QThread``."""

    payload_received = Signal(object)
    finished = Signal()

    def __init__(
        self,
        queue_obj: multiprocessing.Queue,
        process_alive_checker,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._queue = queue_obj
        self._process_alive_checker = process_alive_checker
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _drain_remaining(self) -> None:
        while not self._stop_requested:
            try:
                payload = self._queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            else:
                self.payload_received.emit(payload)

    def run(self) -> None:
        try:
            while not self._stop_requested:
                try:
                    payload = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if not self._process_alive_checker():
                        self._drain_remaining()
                        break
                    continue
                except Exception:
                    break
                self.payload_received.emit(payload)
        finally:
            self.finished.emit()


class ZeMosaicQtWorker(QObject):
    """Manage the background ZeMosaic worker process using a queue listener thread."""

    log_message_emitted = Signal(str, object, dict)
    progress_changed = Signal(float)
    stage_progress = Signal(str, int, int)
    phase_changed = Signal(str, dict)
    gpu_helper_event = Signal(str, dict)
    stats_updated = Signal(dict)
    phase45_event = Signal(str, dict, str)
    eta_updated = Signal(str)
    chrono_control = Signal(str)
    raw_file_count_updated = Signal(int, int)
    master_tile_count_updated = Signal(int, int)
    cluster_override = Signal(dict)
    finished = Signal(bool, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._queue: multiprocessing.Queue | None = None
        self._process: multiprocessing.Process | None = None
        self._listener_thread: QThread | None = None
        self._listener: _WorkerQueueListener | None = None
        self._stop_requested = False
        self._had_error = False
        self._last_error: str = ""
        self._finished_emitted = False
        self._cancelled = False

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        return bool(self._process and self._process.is_alive())

    def start(
        self,
        worker_args: Sequence[Any],
        worker_kwargs: Dict[str, Any],
    ) -> bool:
        spawn_result = self.spawn_worker_process(worker_args, worker_kwargs)
        if spawn_result is None:
            return False
        queue_obj, process = spawn_result
        self.finalize_spawn(queue_obj, process)
        return True

    def spawn_worker_process(
        self,
        worker_args: Sequence[Any],
        worker_kwargs: Dict[str, Any],
    ) -> tuple[multiprocessing.Queue, multiprocessing.Process] | None:
        if run_hierarchical_mosaic_process is None:
            raise RuntimeError("Worker backend is unavailable")
        if self.is_running():
            return None

        queue_obj: multiprocessing.Queue | None = None
        process: multiprocessing.Process | None = None
        try:
            queue_obj = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=run_hierarchical_mosaic_process,
                args=(queue_obj, *worker_args),
                kwargs=worker_kwargs,
                daemon=True,
                name="ZeMosaicWorkerProcessQt",
            )
            process.start()
        except Exception as exc:
            if process is not None and process.is_alive():
                try:
                    process.terminate()
                except Exception:
                    pass
            if queue_obj is not None:
                try:
                    queue_obj.close()
                except Exception:
                    pass
            raise RuntimeError(str(exc)) from exc
        return queue_obj, process

    def finalize_spawn(
        self,
        queue_obj: multiprocessing.Queue,
        process: multiprocessing.Process,
    ) -> None:
        self._queue = queue_obj
        self._process = process
        self._stop_requested = False
        self._had_error = False
        self._last_error = ""
        self._finished_emitted = False
        self._cancelled = False

        listener_thread = QThread()
        listener = _WorkerQueueListener(
            queue_obj,
            lambda: bool(self._process and self._process.is_alive()),
        )
        listener.moveToThread(listener_thread)
        listener.payload_received.connect(self._handle_payload)  # type: ignore[arg-type]
        listener.finished.connect(self._on_listener_finished)  # type: ignore[arg-type]
        listener_thread.started.connect(listener.run)  # type: ignore[arg-type]

        try:
            listener_thread.start()
        except Exception as exc:
            try:
                listener_thread.quit()
                listener_thread.wait(200)
            except Exception:
                pass
            listener.deleteLater()
            if self._process and self._process.is_alive():
                try:
                    self._process.terminate()
                    self._process.join(timeout=0.5)
                except Exception:
                    pass
            if self._queue is not None:
                try:
                    self._queue.close()
                except Exception:
                    pass
            self._queue = None
            self._process = None
            raise RuntimeError(str(exc)) from exc

        self._listener_thread = listener_thread
        self._listener = listener

    def stop(self) -> None:
        self._stop_requested = True
        if self._listener is not None:
            try:
                self._listener.request_stop()
            except Exception:
                pass
        proc = self._process
        if proc and proc.is_alive():
            try:
                proc.terminate()
                proc.join(timeout=0.5)
            except Exception:
                pass
        # Finalization happens once the listener thread finishes and emits its signal.
        if self._listener_thread is None:
            self._finalize(success=False, message="qt_log_processing_cancelled")

    # ------------------------------------------------------------------
    # Queue processing
    # ------------------------------------------------------------------
    def _handle_payload(self, payload: object) -> None:
        try:
            msg_key, prog, lvl, kwargs = payload  # type: ignore[misc]
        except Exception:
            return
        kwargs = kwargs or {}
        if isinstance(msg_key, str) and msg_key == "log_key_processing_cancelled":
            self._cancelled = True

        if msg_key == "STAGE_PROGRESS":
            stage_name = str(prog)
            try:
                current_val = int(lvl)
            except Exception:
                current_val = 0
            try:
                total_val = int(kwargs.get("total", 0))
            except Exception:
                total_val = 0
            if total_val > 0:
                percent = max(0.0, min(100.0, (current_val / float(total_val)) * 100.0))
            else:
                percent = 0.0
            self.progress_changed.emit(percent)
            self.phase_changed.emit(stage_name, {"current": current_val, "total": total_val})
            self.stage_progress.emit(stage_name, current_val, total_val)
            return

        if msg_key == "PROCESS_ERROR":
            error_text = str(kwargs.get("error") or prog or "")
            self._had_error = True
            self._last_error = error_text or "qt_worker_error_generic"
            level = str(lvl or "ERROR")
            payload: Dict[str, Any] = {}
            if isinstance(kwargs, dict):
                payload.update(kwargs)
            if error_text and "error" not in payload:
                payload["error"] = error_text
            # For Qt, treat PROCESS_ERROR as a raw error string for display,
            # while still exposing the structured payload for logs or tooling.
            message_key_or_raw: Any = error_text or "qt_worker_error_generic"
            self.log_message_emitted.emit(level, message_key_or_raw, payload)
            return

        if msg_key == "PROCESS_DONE":
            # Process will terminate shortly; finalization occurs in the listener finished handler.
            return

        # High-priority control and counter messages mirrored from Tk GUI.
        if isinstance(msg_key, str):
            if msg_key.startswith("ETA_UPDATE:"):
                eta_str = msg_key.split(":", 1)[1].strip() if ":" in msg_key else ""
                self.eta_updated.emit(eta_str)
                return
            if msg_key == "CHRONO_START_REQUEST":
                self.chrono_control.emit("start")
                return
            if msg_key == "CHRONO_STOP_REQUEST":
                self.chrono_control.emit("stop")
                return
            if msg_key.startswith("RAW_FILE_COUNT_UPDATE:"):
                counts = msg_key.split(":", 1)[1] if ":" in msg_key else ""
                cur_val = 0
                tot_val = 0
                try:
                    cur_str, tot_str = counts.split("/", 1)
                    cur_val = int(cur_str.strip())
                    tot_val = int(tot_str.strip())
                except Exception:
                    cur_val = 0
                    tot_val = 0
                self.raw_file_count_updated.emit(cur_val, tot_val)
                return
            if msg_key.startswith("MASTER_TILE_COUNT_UPDATE:"):
                counts = msg_key.split(":", 1)[1] if ":" in msg_key else ""
                cur_val = 0
                tot_val = 0
                try:
                    cur_str, tot_str = counts.split("/", 1)
                    cur_val = int(cur_str.strip())
                    tot_val = int(tot_str.strip())
                except Exception:
                    cur_val = 0
                    tot_val = 0
                self.master_tile_count_updated.emit(cur_val, tot_val)
                return
            if msg_key.startswith("CLUSTER_OVERRIDE:"):
                payload_str = msg_key.split(":", 1)[1] if ":" in msg_key else ""
                overrides: Dict[str, Any] = {}
                try:
                    parts = [segment.strip() for segment in payload_str.split(";") if segment.strip()]
                    for part in parts:
                        if part.startswith("panel="):
                            try:
                                overrides["cluster_panel_threshold"] = float(part.split("=", 1)[1])
                            except Exception:
                                pass
                        elif part.startswith("target="):
                            try:
                                overrides["cluster_target_groups"] = int(part.split("=", 1)[1])
                            except Exception:
                                pass
                except Exception:
                    overrides = {}
                self.cluster_override.emit(overrides)
                return

        if isinstance(msg_key, str):
            if msg_key == "global_coadd_info_helper_path":
                payload_dict: Dict[str, Any] = {}
                if isinstance(kwargs, dict):
                    payload_dict.update(kwargs)
                # Mirror Tk behavior: always surface helper start payload;
                # downstream UI can interpret the helper name as needed.
                self.gpu_helper_event.emit("start", payload_dict)
            elif msg_key == "p4_global_coadd_finished":
                payload_dict = {}
                if isinstance(kwargs, dict):
                    payload_dict.update(kwargs)
                # Mirror Tk behavior: always surface helper finish payload.
                self.gpu_helper_event.emit("finish", payload_dict)
            elif msg_key in {"global_coadd_warn_helper_fallback", "global_coadd_warn_helper_unavailable"}:
                payload_dict = {}
                if isinstance(kwargs, dict):
                    payload_dict.update(kwargs)
                helper_name = str(payload_dict.get("helper") or "")
                if helper_name and "gpu" in helper_name.lower():
                    payload_dict.setdefault("reason_key", msg_key)
                    self.gpu_helper_event.emit("abort", payload_dict)

        if isinstance(msg_key, str) and msg_key.startswith("p45_"):
            payload_dict: Dict[str, Any] = {}
            if isinstance(kwargs, dict):
                payload_dict.update(kwargs)
            if prog not in (None, "") and "message" not in payload_dict:
                payload_dict["message"] = prog
            level_text = str(lvl or "INFO")
            try:
                self.phase45_event.emit(msg_key, payload_dict, level_text)
            except Exception:
                pass
            return

        if isinstance(msg_key, str) and msg_key.upper() == "STATS_UPDATE":
            if isinstance(kwargs, dict):
                self.stats_updated.emit(kwargs)
            return

        level = str(lvl or "INFO")
        payload: Dict[str, Any] = {}
        if isinstance(kwargs, dict):
            payload.update(kwargs)
        if isinstance(msg_key, str):
            message_key_or_raw: Any = msg_key
        else:
            message_key_or_raw = self._stringify_message(msg_key, prog, payload)
        self.log_message_emitted.emit(level, message_key_or_raw, payload)

    def _on_listener_finished(self) -> None:
        success = not self._had_error and not self._stop_requested and not self._cancelled
        if success:
            message = ""
        else:
            if not self._last_error:
                message = "qt_log_processing_cancelled"
            else:
                message = self._last_error
        self._finalize(success=success, message=message)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _cleanup_listener(self) -> None:
        listener = self._listener
        thread = self._listener_thread
        self._listener = None
        self._listener_thread = None

        if thread is not None:
            if listener is not None:
                try:
                    listener.request_stop()
                except Exception:
                    pass
            if thread.isRunning():
                try:
                    thread.quit()
                    thread.wait(500)
                except Exception:
                    pass
            if listener is not None:
                listener.deleteLater()
            thread.deleteLater()
        elif listener is not None:
            try:
                listener.request_stop()
            except Exception:
                pass
            listener.deleteLater()

    def _finalize(self, *, success: bool, message: str) -> None:
        self._cleanup_listener()
        if self._process is not None:
            try:
                if self._process.is_alive():
                    self._process.join(timeout=0.5)
            except Exception:
                pass
        self._process = None
        if self._queue is not None:
            try:
                self._queue.close()
            except Exception:
                pass
        self._queue = None
        if not self._finished_emitted:
            self._finished_emitted = True
            self.finished.emit(success, message)

    def _stringify_message(self, msg_key: Any, prog: Any, kwargs: Dict[str, Any]) -> str:
        parts: List[str] = []
        if isinstance(msg_key, str):
            parts.append(msg_key)
        elif msg_key is not None:
            parts.append(str(msg_key))
        if prog not in (None, ""):
            parts.append(str(prog))
        if kwargs:
            kv_pairs = ", ".join(f"{key}={value}" for key, value in kwargs.items())
            if kv_pairs:
                parts.append(kv_pairs)
        return " | ".join(parts) if parts else "qt_worker_generic_update"

class ZeMosaicQtMainWindow(QMainWindow):
    """Initial Qt main window skeleton with placeholder panels."""

    def __init__(self) -> None:
        super().__init__()
        icon = _load_zemosaic_qicon()
        if icon is not None:
            self.setWindowIcon(icon)

        self._config_path: str | None = self._determine_config_path()
        self._config_load_notes: List[Tuple[str, str, str, Dict[str, Any]]] = []
        self._loaded_config_snapshot: Dict[str, Any] = {}
        self._persisted_config_keys: set[str] = set()
        self.language_combo: QComboBox | None = None
        self.backend_combo: QComboBox | None = None
        self._default_config_values: Dict[str, Any] = self._baseline_default_config()
        self.config: Dict[str, Any] = self._load_config()
        # Phase 4.5 (inter-master merge) must not be user-activable from Qt.
        # Force the flag off regardless of persisted config so the worker never
        # receives an enabled state from this GUI.
        self._disable_phase45_config()
        self.localizer = self._create_localizer(self.config.get("language", "en"))
        self.setWindowTitle(
            self._tr("qt_window_title_preview", "ZeMosaic (Qt Preview)")
        )
        self._gpu_devices: List[Tuple[str, int | None]] = self._detect_gpus()
        if self._gpu_devices:
            self.config.setdefault("gpu_selector", self._gpu_devices[0][0])
        else:
            self.config.setdefault("gpu_selector", "CPU (no GPU)")
        self._initialize_log_level_prefixes()
        for key, fallback in self._default_config_values.items():
            self.config.setdefault(key, fallback)
        self.config.setdefault("qt_theme_mode", "system")
        self._config_fields: Dict[str, Dict[str, Any]] = {}
        self.solver_choice_combo: QComboBox | None = None
        self._solver_panels: Dict[str, QWidget] = {}
        self._solver_none_hint: QLabel | None = None
        self.theme_mode_combo: QComboBox | None = None
        self.tab_widget: QTabWidget | None = None
        self._tab_layouts: Dict[str, QVBoxLayout] = {}
        self._legacy_layout: QVBoxLayout | None = None

        self._last_filter_overrides: Dict[str, Any] | None = None
        self._last_filtered_header_items: List[Any] | None = None
        self._worker_start_thread: threading.Thread | None = None
        self._worker_start_result: tuple[bool, str | None] | None = None

        self._stage_aliases = {
            "phase1_scan": "phase1",
            "phase2_cluster": "phase2",
            "phase3_master_tiles": "phase3",
            "phase4_grid": "phase4",
            "phase4_5": "phase4_5",
            "phase5_intertile": "phase5",
            "phase5_incremental": "phase5",
            "phase5_reproject": "phase5",
        }
        self._stage_order = [
            "phase1",
            "phase2",
            "phase3",
            "phase4",
            "phase4_5",
            "phase5",
            "phase6",
            "phase7",
        ]
        self._stage_weights = {
            "phase1": 30.0,
            "phase2": 5.0,
            "phase3": 35.0,
            "phase4": 5.0,
            "phase4_5": 6.0,
            "phase5": 9.0,
            "phase6": 8.0,
            "phase7": 2.0,
        }
        self._stage_progress_values: Dict[str, float] = {
            key: 0.0 for key in self._stage_order
        }
        self._stage_times: Dict[str, Dict[str, Any]] = {}
        self._progress_start_time: float | None = None
        self._last_global_progress: float = 0.0
        self._eta_seconds_smoothed: float | None = None
        self._weighted_progress_active = False

        self._gpu_eta_override: Dict[str, Any] | None = None
        self._gpu_helper_active: Dict[str, Any] | None = None
        gpu_eta_profiles = self.config.get("gpu_eta_profiles")
        if not isinstance(gpu_eta_profiles, dict):
            gpu_eta_profiles = {}
        self._gpu_eta_profiles: Dict[str, Any] = gpu_eta_profiles
        self.config["gpu_eta_profiles"] = self._gpu_eta_profiles
        try:
            default_gpu_rate = float(self.config.get("gpu_eta_default_rate", 0.85))
            if default_gpu_rate <= 0:
                raise ValueError
        except Exception:
            default_gpu_rate = 0.85
            self.config["gpu_eta_default_rate"] = default_gpu_rate
        self._gpu_eta_default_rate: float = default_gpu_rate

        self._gpu_eta_timer = QTimer(self)
        self._gpu_eta_timer.setInterval(1000)
        self._gpu_eta_timer.timeout.connect(self._tick_gpu_eta_override)  # type: ignore[arg-type]

        self._phase45_groups: Dict[int, Dict[str, Any]] = {}
        self._phase45_group_progress: Dict[int, Dict[str, Any]] = {}
        self._phase45_active: Optional[int] = None
        self._phase45_last_out: Optional[str] = None
        self._phase45_overlay_enabled: bool = False
        self.phase45_status_label: Optional[QLabel] = None
        self.phase45_overlay_scene: Optional[QGraphicsScene] = None
        self.phase45_overlay_view: Optional[QGraphicsView] = None
        self.phase45_overlay_toggle: Optional[QCheckBox] = None

        self.is_processing = False

        self._setup_ui()
        self._apply_theme(self.config.get("qt_theme_mode", "system"))
        self._emit_config_notes()
        self.worker_controller = ZeMosaicQtWorker(self)
        self.worker_controller.log_message_emitted.connect(self._on_worker_log_message)  # type: ignore[arg-type]
        self.worker_controller.progress_changed.connect(self._on_worker_progress_changed)  # type: ignore[arg-type]
        self.worker_controller.stage_progress.connect(self._on_worker_stage_progress)  # type: ignore[arg-type]
        self.worker_controller.phase_changed.connect(self._on_worker_phase_changed)  # type: ignore[arg-type]
        self.worker_controller.stats_updated.connect(self._on_worker_stats_updated)  # type: ignore[arg-type]
        self.worker_controller.phase45_event.connect(self._on_worker_phase45_event)  # type: ignore[arg-type]
        self.worker_controller.gpu_helper_event.connect(self._on_worker_gpu_helper_event)  # type: ignore[arg-type]
        self.worker_controller.eta_updated.connect(self._on_worker_eta_updated)  # type: ignore[arg-type]
        self.worker_controller.chrono_control.connect(self._on_worker_chrono_control)  # type: ignore[arg-type]
        self.worker_controller.raw_file_count_updated.connect(self._on_worker_raw_file_count_updated)  # type: ignore[arg-type]
        self.worker_controller.master_tile_count_updated.connect(self._on_worker_master_tile_count_updated)  # type: ignore[arg-type]
        self.worker_controller.cluster_override.connect(self._on_worker_cluster_override)  # type: ignore[arg-type]
        self.worker_controller.finished.connect(self._on_worker_finished)  # type: ignore[arg-type]

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._on_elapsed_timer_tick)  # type: ignore[arg-type]
        self._run_started_monotonic: float | None = None

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(12, 12, 12, 12)
        outer_layout.setSpacing(10)

        scroll = QScrollArea(central_widget)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer_layout.addWidget(scroll, 1)

        tabs_container = QWidget(scroll)
        scroll.setWidget(tabs_container)
        tab_container_layout = QVBoxLayout(tabs_container)
        tab_container_layout.setContentsMargins(0, 0, 0, 0)
        tab_container_layout.setSpacing(10)

        self.tab_widget = QTabWidget(tabs_container)
        tab_container_layout.addWidget(self.tab_widget)

        self._initialize_tab_pages()

        self._legacy_layout = None
        legacy_widget = QWidget(tabs_container)
        legacy_layout = QVBoxLayout(legacy_widget)
        legacy_layout.setContentsMargins(0, 0, 0, 0)
        legacy_layout.setSpacing(10)
        self._legacy_layout = legacy_layout
        if self._populate_legacy_placeholder_sections():
            tab_container_layout.addWidget(legacy_widget)
        else:
            legacy_widget.deleteLater()
            self._legacy_layout = None

        button_row = self._build_command_row()
        outer_layout.addLayout(button_row)

    def _build_command_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        self.filter_button = QPushButton(self._tr("qt_button_filter", "Filter…"))
        self.filter_button.clicked.connect(self._on_filter_clicked)  # type: ignore[attr-defined]
        self.start_button = QPushButton(self._tr("qt_button_start", "Start"))
        self.stop_button = QPushButton(self._tr("qt_button_stop", "Stop"))
        self.start_button.clicked.connect(self._on_start_clicked)  # type: ignore[attr-defined]
        self.stop_button.clicked.connect(self._on_stop_clicked)  # type: ignore[attr-defined]
        row.addWidget(self.filter_button)
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        return row

    def _initialize_tab_pages(self) -> None:
        if self.tab_widget is None:
            return
        self._tab_layouts = {}
        tab_definitions = [
            ("main", "qt_tab_main_title", "Main"),
            ("solver", "qt_tab_solver_title", "Solver"),
            ("system", "qt_tab_system_title", "System"),
            ("advanced", "qt_tab_advanced_title", "Advanced"),
            ("skin", "qt_tab_skin_title", "Skin"),
            ("language", "qt_tab_language_title", "Language"),
        ]
        for key, label_key, fallback in tab_definitions:
            tab = QWidget(self.tab_widget)
            layout = QVBoxLayout(tab)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(10)
            self._tab_layouts[key] = layout
            self.tab_widget.addTab(tab, self._tr(label_key, fallback))
        self._populate_main_tab(self._tab_layouts["main"])
        self._populate_solver_tab(self._tab_layouts["solver"])
        self._populate_system_tab(self._tab_layouts["system"])
        self._populate_advanced_tab(self._tab_layouts["advanced"])
        self._populate_skin_tab(self._tab_layouts["skin"])
        self._populate_language_tab(self._tab_layouts["language"])

    def _populate_main_tab(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._create_folders_group())
        layout.addWidget(self._create_instrument_group())
        layout.addWidget(self._create_mosaic_group())
        layout.addWidget(self._create_final_assembly_group())
        layout.addStretch(1)

    def _populate_solver_tab(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._create_astap_group())
        layout.addStretch(1)

    def _populate_system_tab(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._create_system_resources_group())
        layout.addWidget(self._create_gpu_group())
        layout.addWidget(self._create_logging_group())
        layout.addStretch(1)

    def _populate_advanced_tab(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._create_quality_group())
        layout.addWidget(self._create_stacking_group())
        layout.addStretch(1)

    def _populate_skin_tab(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._create_skin_group())
        layout.addWidget(self._create_backend_group())
        layout.addStretch(1)

    def _populate_language_tab(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._create_language_group())
        layout.addStretch(1)

    def _add_placeholder_to_tab(self, tab_key: str) -> None:
        layout = self._tab_layouts.get(tab_key)
        if layout is None:
            return
        placeholder_key = f"qt_tab_placeholder_{tab_key}"
        placeholder_text = self._tr(
            placeholder_key,
            "This tab is under construction.",
        )
        placeholder = QLabel(placeholder_text, self.tab_widget)
        placeholder.setWordWrap(True)
        layout.addWidget(placeholder)
        layout.addStretch(1)

    def _populate_legacy_placeholder_sections(self) -> bool:
        """Temporarily keep unmigrated groups visible under the tab widget."""
        if self._legacy_layout is None:
            return False
        builders = []
        added_any = False
        for builder in builders:
            try:
                widget = builder()
            except Exception:
                continue
            self._legacy_layout.addWidget(widget)
            added_any = True
        if added_any:
            self._legacy_layout.addStretch(1)
        return added_any

    def _resolve_available_languages(self) -> List[str]:
        available_langs: List[str] = ["en", "fr"]
        locales_dir = getattr(self.localizer, "locales_dir_abs_path", None)
        if isinstance(locales_dir, str) and os.path.isdir(locales_dir):
            try:
                entries = sorted(
                    name
                    for name in os.listdir(locales_dir)
                    if name.endswith(".json")
                    and os.path.isfile(os.path.join(locales_dir, name))
                )
                detected = [os.path.splitext(name)[0] for name in entries]
                if detected:
                    available_langs = detected
            except Exception:
                available_langs = ["en", "fr"]
        return available_langs

    def _create_folders_group(self) -> QGroupBox:
        group = QGroupBox(self._tr("qt_group_folders", "Folders"), self)
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._register_directory_picker(
            "input_dir",
            layout,
            self._tr("qt_field_input_dir", "Input folder"),
            dialog_title=self._tr(
                "qt_dialog_select_input_dir", "Select Input Folder (raw files)"
            ),
        )
        self._register_directory_picker(
            "output_dir",
            layout,
            self._tr("qt_field_output_dir", "Output folder"),
            dialog_title=self._tr(
                "qt_dialog_select_output_dir", "Select Output Folder"
            ),
        )
        self._register_line_edit(
            "global_wcs_output_path",
            layout,
            self._tr("qt_field_global_wcs_output", "Global WCS output path"),
        )
        return group

    def _create_instrument_group(self) -> QGroupBox:
        group = QGroupBox(
            self._tr("qt_group_instrument", "Instrument / Seestar"),
            self,
        )
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._register_checkbox(
            "auto_detect_seestar",
            layout,
            self._tr("qt_field_auto_detect_seestar", "Auto-detect Seestar frames"),
        )
        self._register_checkbox(
            "force_seestar_mode",
            layout,
            self._tr("qt_field_force_seestar", "Force Seestar workflow"),
        )
        self._register_checkbox(
            "sds_mode_default",
            layout,
            self._tr("qt_field_sds_mode", "Enable SDS mode by default"),
        )
        self._register_double_spinbox(
            "sds_coverage_threshold",
            layout,
            self._tr("qt_field_sds_threshold", "SDS coverage threshold"),
            minimum=0.0,
            maximum=1.0,
            single_step=0.01,
            decimals=2,
        )

        return group

    def _create_astap_group(self) -> QGroupBox:
        group = QGroupBox(self._tr("qt_group_solver", "Plate solving"), self)
        outer_layout = QVBoxLayout(group)
        outer_layout.setContentsMargins(8, 8, 8, 8)
        outer_layout.setSpacing(10)

        selection_box = QGroupBox(
            self._tr("qt_group_solver_choice", "Solver selection"), group
        )
        selection_layout = QFormLayout(selection_box)
        selection_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        solver_combo = QComboBox(selection_box)
        solver_options = [
            ("ASTAP", self._tr("qt_solver_astap", "ASTAP (recommended)")),
            ("ASTROMETRY", self._tr("qt_solver_astrometry", "Astrometry.net")),
            ("ANSVR", self._tr("qt_solver_ansvr", "ANSVR (local server)")),
            ("NONE", self._tr("qt_solver_none", "None (WCS already present)")),
        ]
        for value, label in solver_options:
            solver_combo.addItem(label, value)
        current_solver = str(self.config.get("solver_method", "ASTAP") or "ASTAP").upper()
        solver_index = next(
            (idx for idx, (value, _label) in enumerate(solver_options) if value == current_solver),
            0,
        )
        solver_combo.setCurrentIndex(solver_index)
        resolved_solver = solver_options[solver_index][0]
        self.config["solver_method"] = resolved_solver
        selection_layout.addRow(
            QLabel(self._tr("qt_field_solver_choice", "Preferred solver"), selection_box),
            solver_combo,
        )
        self.solver_choice_combo = solver_combo
        self._config_fields["solver_method"] = {
            "kind": "combobox",
            "widget": solver_combo,
            "type": str,
            "value_getter": solver_combo.currentData,
        }

        outer_layout.addWidget(selection_box)

        astap_box = QGroupBox(
            self._tr("qt_group_astap", "ASTAP configuration"), group
        )
        astap_layout = QFormLayout(astap_box)
        astap_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._register_line_edit(
            "astap_executable_path",
            astap_layout,
            self._tr("qt_field_astap_executable", "ASTAP executable"),
            browse_action="file",
            dialog_title=self._tr(
                "qt_dialog_select_astap_executable", "Select ASTAP Executable"
            ),
        )
        self._register_line_edit(
            "astap_data_directory_path",
            astap_layout,
            self._tr("qt_field_astap_data_dir", "ASTAP data directory"),
            browse_action="directory",
            dialog_title=self._tr(
                "qt_dialog_select_astap_data_dir", "Select ASTAP Data Directory"
            ),
        )
        self._register_double_spinbox(
            "astap_default_search_radius",
            astap_layout,
            self._tr("qt_field_astap_search_radius", "Default search radius (°)"),
            minimum=0.1,
            maximum=180.0,
            single_step=0.1,
        )
        self._register_spinbox(
            "astap_default_downsample",
            astap_layout,
            self._tr("qt_field_astap_downsample", "Default downsample"),
            minimum=0,
            maximum=4,
        )
        self._register_spinbox(
            "astap_default_sensitivity",
            astap_layout,
            self._tr("qt_field_astap_sensitivity", "Default sensitivity"),
            minimum=-25,
            maximum=500,
        )
        self._register_spinbox(
            "astap_max_instances",
            astap_layout,
            self._tr("qt_field_astap_max_instances", "Max ASTAP instances"),
            minimum=1,
            maximum=16,
        )

        outer_layout.addWidget(astap_box)

        astrometry_box = QGroupBox(
            self._tr("qt_group_astrometry", "Astrometry.net configuration"), group
        )
        astrometry_layout = QFormLayout(astrometry_box)
        astrometry_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._register_line_edit(
            "astrometry_api_key",
            astrometry_layout,
            self._tr("qt_field_astrometry_api_key", "API key"),
        )
        self._register_spinbox(
            "astrometry_timeout",
            astrometry_layout,
            self._tr("qt_field_astrometry_timeout", "Timeout (s)"),
            minimum=10,
            maximum=600,
        )
        self._register_spinbox(
            "astrometry_downsample",
            astrometry_layout,
            self._tr("qt_field_astrometry_downsample", "Blind-solve downsample"),
            minimum=1,
            maximum=16,
        )

        outer_layout.addWidget(astrometry_box)

        ansvr_box = QGroupBox(
            self._tr("qt_group_ansvr", "ANSVR notes"),
            group,
        )
        ansvr_layout = QVBoxLayout(ansvr_box)
        ansvr_layout.setContentsMargins(8, 8, 8, 8)
        ansvr_hint = QLabel(
            self._tr(
                "qt_ansvr_hint",
                "ANSVR uses the local Ansvr service (Astrometry.net server).\n"
                "Ensure it is installed and running on your system.",
            ),
            ansvr_box,
        )
        ansvr_hint.setWordWrap(True)
        ansvr_layout.addWidget(ansvr_hint)

        outer_layout.addWidget(ansvr_box)

        none_hint = QLabel(
            self._tr(
                "qt_solver_none_hint",
                "No plate solving will be performed. Existing WCS headers must be present.",
            ),
            group,
        )
        none_hint.setWordWrap(True)
        outer_layout.addWidget(none_hint)

        self._solver_panels = {
            "ASTAP": astap_box,
            "ASTROMETRY": astrometry_box,
            "ANSVR": ansvr_box,
        }
        self._solver_none_hint = none_hint

        solver_combo.currentIndexChanged.connect(self._on_solver_choice_changed)  # type: ignore[arg-type]
        self._update_solver_visibility(resolved_solver)

        return group

    def _on_solver_choice_changed(self) -> None:
        if self.solver_choice_combo is None:
            return
        data = self.solver_choice_combo.currentData()
        if data is None:
            data = self.solver_choice_combo.currentText()
        if not isinstance(data, str):
            data = str(data)
        normalized = data.upper().strip()
        if not normalized:
            normalized = "ASTAP"
        self.config["solver_method"] = normalized
        self._update_solver_visibility(normalized)

    def _update_solver_visibility(self, solver_choice: str | None = None) -> None:
        if solver_choice is None:
            solver_choice = str(self.config.get("solver_method", "ASTAP") or "ASTAP")
        normalized = solver_choice.upper().strip()
        if normalized not in self._solver_panels and normalized != "NONE":
            normalized = "ASTAP"
            self.config["solver_method"] = normalized
        for key, panel in self._solver_panels.items():
            panel.setVisible(normalized == key)
        if self._solver_none_hint is not None:
            self._solver_none_hint.setVisible(normalized == "NONE")

    def _create_mosaic_group(self) -> QGroupBox:
        group = QGroupBox(self._tr("qt_group_mosaic", "Mosaic / clustering"), self)
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._register_double_spinbox(
            "cluster_panel_threshold",
            layout,
            self._tr("qt_field_cluster_threshold", "Cluster threshold"),
            minimum=0.01,
            maximum=5.0,
            single_step=0.01,
            decimals=2,
        )
        self._register_spinbox(
            "cluster_target_groups",
            layout,
            self._tr("qt_field_cluster_target", "Target groups"),
            minimum=0,
            maximum=999,
        )
        self._register_double_spinbox(
            "cluster_orientation_split_deg",
            layout,
            self._tr("qt_field_cluster_orientation", "Orientation split (°)"),
            minimum=0.0,
            maximum=180.0,
            single_step=1.0,
            decimals=1,
        )

        if ENABLE_PHASE45_UI:
            phase45_box = QGroupBox(
                self._tr("qt_group_phase45", "Phase 4.5 / Super-tiles"),
                group,
            )
            phase45_layout = QFormLayout(phase45_box)
            phase45_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

            self._register_checkbox(
                "inter_master_merge_enable",
                phase45_layout,
                self._tr("qt_checkbox_phase45_enable", "Enable inter-master merge"),
            )

            overlap_spinbox = QDoubleSpinBox()
            overlap_spinbox.setRange(0.0, 100.0)
            overlap_spinbox.setDecimals(1)
            overlap_spinbox.setSingleStep(0.5)
            overlap_fraction = float(self.config.get("inter_master_overlap_threshold", 0.60))
            if not 0.0 <= overlap_fraction <= 1.0:
                overlap_fraction = 0.60
                self.config["inter_master_overlap_threshold"] = overlap_fraction
            overlap_spinbox.setValue(overlap_fraction * 100.0)
            phase45_layout.addRow(
                QLabel(self._tr("qt_field_phase45_overlap", "Overlap ≥ (%)")),
                overlap_spinbox,
            )
            self._config_fields["inter_master_overlap_threshold"] = {
                "kind": "double_spinbox",
                "widget": overlap_spinbox,
                "type": float,
                "value_getter": overlap_spinbox.value,
                "postprocess": lambda value: max(0.0, min(1.0, float(value) / 100.0)),
            }

            method_combo = QComboBox()
            method_options = [
                ("winsor", self._tr("qt_phase45_method_winsor", "Winsorized")),
                ("mean", self._tr("qt_phase45_method_mean", "Mean")),
                ("median", self._tr("qt_phase45_method_median", "Median")),
            ]
            for value, label in method_options:
                method_combo.addItem(label, value)
            current_method = str(self.config.get("inter_master_stack_method", "winsor")).lower()
            method_index = next(
                (idx for idx, (value, _label) in enumerate(method_options) if value == current_method),
                0,
            )
            method_combo.setCurrentIndex(method_index)
            phase45_layout.addRow(
                QLabel(self._tr("qt_field_phase45_method", "Stack method")),
                method_combo,
            )
            self._config_fields["inter_master_stack_method"] = {
                "kind": "combobox",
                "widget": method_combo,
                "type": str,
            }

            self._register_spinbox(
                "inter_master_min_group_size",
                phase45_layout,
                self._tr("qt_field_phase45_min_group", "Minimum group size"),
                minimum=2,
                maximum=512,
            )
            self._register_spinbox(
                "inter_master_max_group",
                phase45_layout,
                self._tr("qt_field_phase45_max_group", "Maximum group size"),
                minimum=2,
                maximum=2048,
            )

            memmap_combo = QComboBox()
            memmap_options = [
                ("auto", self._tr("qt_phase45_memmap_auto", "Auto")),
                ("always", self._tr("qt_phase45_memmap_always", "Always")),
                ("never", self._tr("qt_phase45_memmap_never", "Never")),
            ]
            for value, label in memmap_options:
                memmap_combo.addItem(label, value)
            current_policy = str(self.config.get("inter_master_memmap_policy", "auto")).lower()
            memmap_index = next(
                (idx for idx, (value, _label) in enumerate(memmap_options) if value == current_policy),
                0,
            )
            memmap_combo.setCurrentIndex(memmap_index)
            phase45_layout.addRow(
                QLabel(self._tr("qt_field_phase45_memmap", "Memmap policy")),
                memmap_combo,
            )
            self._config_fields["inter_master_memmap_policy"] = {
                "kind": "combobox",
                "widget": memmap_combo,
                "type": str,
            }

            scale_combo = QComboBox()
            scale_options = [
                ("final", self._tr("qt_phase45_scale_final", "Final scale")),
                ("native", self._tr("qt_phase45_scale_native", "Native scale")),
            ]
            for value, label in scale_options:
                scale_combo.addItem(label, value)
            current_scale = str(self.config.get("inter_master_local_scale", "final")).lower()
            scale_index = next(
                (idx for idx, (value, _label) in enumerate(scale_options) if value == current_scale),
                0,
            )
            scale_combo.setCurrentIndex(scale_index)
            phase45_layout.addRow(
                QLabel(self._tr("qt_field_phase45_local_scale", "Local scale")),
                scale_combo,
            )
            self._config_fields["inter_master_local_scale"] = {
                "kind": "combobox",
                "widget": scale_combo,
                "type": str,
            }

            self._register_checkbox(
                "inter_master_photometry_intragroup",
                phase45_layout,
                self._tr("qt_field_phase45_intragroup", "Intra-group photometry"),
            )
            self._register_checkbox(
                "inter_master_photometry_intersuper",
                phase45_layout,
                self._tr("qt_field_phase45_intersuper", "Inter-super photometry"),
            )
            self._register_double_spinbox(
                "inter_master_photometry_clip_sigma",
                phase45_layout,
                self._tr("qt_field_phase45_clip_sigma", "Photometry clip σ"),
                minimum=0.1,
                maximum=10.0,
                single_step=0.1,
                decimals=2,
            )

            layout.addRow(phase45_box)

        return group

    def _create_quality_group(self) -> QGroupBox:
        group = QGroupBox(
            self._tr("qt_group_quality", "Cropping / quality / Alt-Az"),
            self,
        )
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        tile_crop_group = QGroupBox(
            self._tr("qt_group_master_tile_crop", "Master tile crop"),
            group,
        )
        tile_crop_layout = QFormLayout(tile_crop_group)
        tile_crop_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self._register_checkbox(
            "apply_master_tile_crop",
            tile_crop_layout,
            self._tr("qt_field_apply_master_tile_crop", "Apply master tile crop"),
        )
        self._register_double_spinbox(
            "master_tile_crop_percent",
            tile_crop_layout,
            self._tr("qt_field_master_tile_crop_percent", "Crop percent per edge"),
            minimum=0.0,
            maximum=25.0,
            single_step=0.5,
            decimals=1,
        )
        layout.addWidget(tile_crop_group)

        crop_group = QGroupBox(
            self._tr("qt_group_quality_crop", "Quality crop"),
            group,
        )
        crop_layout = QFormLayout(crop_group)
        crop_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self._register_checkbox(
            "quality_crop_enabled",
            crop_layout,
            self._tr("qt_field_quality_crop_enabled", "Enable quality crop"),
        )
        self._register_spinbox(
            "quality_crop_band_px",
            crop_layout,
            self._tr("qt_field_quality_crop_band", "Band width (px)"),
            minimum=4,
            maximum=256,
            single_step=4,
        )
        self._register_double_spinbox(
            "quality_crop_k_sigma",
            crop_layout,
            self._tr("qt_field_quality_crop_k_sigma", "K-sigma"),
            minimum=0.5,
            maximum=5.0,
            single_step=0.1,
            decimals=1,
        )
        self._register_spinbox(
            "quality_crop_min_run",
            crop_layout,
            self._tr("qt_field_quality_crop_min_run", "Minimum run"),
            minimum=1,
            maximum=32,
        )
        self._register_spinbox(
            "quality_crop_margin_px",
            crop_layout,
            self._tr("qt_field_quality_crop_margin", "Margin (px)"),
            minimum=0,
            maximum=64,
        )
        self._register_checkbox(
            "crop_follow_signal",
            crop_layout,
            self._tr("qt_field_crop_follow_signal", "Follow signal when cropping"),
        )
        layout.addWidget(crop_group)

        altaz_group = QGroupBox(
            self._tr("qt_group_altaz_cleanup", "Alt-Az cleanup"),
            group,
        )
        altaz_layout = QFormLayout(altaz_group)
        altaz_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self._register_checkbox(
            "altaz_cleanup_enabled",
            altaz_layout,
            self._tr("qt_field_altaz_enabled", "Enable Alt-Az cleanup"),
        )
        self._register_double_spinbox(
            "altaz_margin_percent",
            altaz_layout,
            self._tr("qt_field_altaz_margin", "Margin (%)"),
            minimum=0.0,
            maximum=50.0,
            single_step=0.5,
            decimals=1,
        )
        self._register_double_spinbox(
            "altaz_decay",
            altaz_layout,
            self._tr("qt_field_altaz_decay", "Decay"),
            minimum=0.0,
            maximum=2.0,
            single_step=0.05,
            decimals=2,
        )
        self._register_checkbox(
            "altaz_nanize",
            altaz_layout,
            self._tr("qt_field_altaz_nanize", "Convert Alt-Az gaps to NaN"),
        )
        layout.addWidget(altaz_group)

        quality_gate_group = QGroupBox(
            self._tr("qt_group_quality_gate", "Master tile quality gate"),
            group,
        )
        quality_gate_layout = QFormLayout(quality_gate_group)
        quality_gate_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self._register_checkbox(
            "quality_gate_enabled",
            quality_gate_layout,
            self._tr("qt_field_quality_gate_enabled", "Enable quality gate"),
        )
        self._register_checkbox(
            "quality_gate_move_rejects",
            quality_gate_layout,
            self._tr("qt_field_quality_gate_move", "Move rejected frames to subfolder"),
        )
        self._register_double_spinbox(
            "quality_gate_threshold",
            quality_gate_layout,
            self._tr("qt_field_quality_gate_threshold", "Threshold"),
            minimum=0.0,
            maximum=1.0,
            single_step=0.01,
            decimals=2,
        )
        self._register_spinbox(
            "quality_gate_edge_band_px",
            quality_gate_layout,
            self._tr("qt_field_quality_gate_edge", "Edge band (px)"),
            minimum=0,
            maximum=4096,
        )
        self._register_double_spinbox(
            "quality_gate_k_sigma",
            quality_gate_layout,
            self._tr("qt_field_quality_gate_k_sigma", "K-sigma"),
            minimum=0.0,
            maximum=10.0,
            single_step=0.1,
            decimals=1,
        )
        self._register_spinbox(
            "quality_gate_erode_px",
            quality_gate_layout,
            self._tr("qt_field_quality_gate_erode", "Erode (px)"),
            minimum=0,
            maximum=512,
        )
        layout.addWidget(quality_gate_group)

        coverage_group = QGroupBox(
            self._tr("qt_group_two_pass", "Two-pass coverage renormalization"),
            group,
        )
        coverage_layout = QFormLayout(coverage_group)
        coverage_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self._register_checkbox(
            "two_pass_coverage_renorm",
            coverage_layout,
            self._tr(
                "qt_field_two_pass_enabled",
                "Enable two-pass coverage renormalization",
            ),
        )
        self._register_spinbox(
            "two_pass_cov_sigma_px",
            coverage_layout,
            self._tr("qt_field_two_pass_sigma", "Sigma radius (px)"),
            minimum=0,
            maximum=512,
            single_step=5,
        )

        gain_widget = QWidget(coverage_group)
        gain_layout = QHBoxLayout(gain_widget)
        gain_layout.setContentsMargins(0, 0, 0, 0)
        gain_layout.setSpacing(6)

        gain_min_spin = QDoubleSpinBox(gain_widget)
        gain_min_spin.setRange(0.1, 5.0)
        gain_min_spin.setSingleStep(0.01)
        gain_min_spin.setDecimals(2)
        gain_max_spin = QDoubleSpinBox(gain_widget)
        gain_max_spin.setRange(0.1, 5.0)
        gain_max_spin.setSingleStep(0.01)
        gain_max_spin.setDecimals(2)

        gain_clip = self.config.get("two_pass_cov_gain_clip", [0.85, 1.18])
        if not (
            isinstance(gain_clip, (list, tuple))
            and len(gain_clip) >= 2
        ):
            gain_clip = [0.85, 1.18]
            self.config["two_pass_cov_gain_clip"] = list(gain_clip)
        gain_min_spin.setValue(float(gain_clip[0]))
        gain_max_spin.setValue(float(gain_clip[1]))

        gain_layout.addWidget(gain_min_spin)
        gain_layout.addWidget(QLabel("→", gain_widget))
        gain_layout.addWidget(gain_max_spin)

        coverage_layout.addRow(
            QLabel(self._tr("qt_field_two_pass_gain", "Gain clip range")),
            gain_widget,
        )
        self._config_fields["two_pass_cov_gain_clip"] = {
            "kind": "composite",
            "widget": (gain_min_spin, gain_max_spin),
            "type": list,
            "value_getter": lambda: [
                float(gain_min_spin.value()),
                float(gain_max_spin.value()),
            ],
        }

        layout.addWidget(coverage_group)

        return group

    def _create_stacking_group(self) -> QGroupBox:
        group = QGroupBox(
            self._tr("stacking_options_frame_title", "Stacking Options (Master Tiles)"),
            self,
        )
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        norm_combo = QComboBox(group)
        norm_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        norm_options = [
            ("none", self._tr("norm_method_none", "None")),
            ("linear_fit", self._tr("norm_method_linear_fit", "Linear Fit (Sky)")),
            ("sky_mean", self._tr("norm_method_sky_mean", "Sky Mean Subtraction")),
        ]
        for value, label in norm_options:
            norm_combo.addItem(label, value)
        current_norm = str(self.config.get("stacking_normalize_method", "linear_fit"))
        norm_index = next(
            (idx for idx, (value, _label) in enumerate(norm_options) if value == current_norm),
            0,
        )
        norm_combo.setCurrentIndex(norm_index)
        layout.addRow(QLabel(self._tr("stacking_norm_method_label", "Normalization:")), norm_combo)
        self._config_fields["stacking_normalize_method"] = {
            "kind": "combobox",
            "widget": norm_combo,
            "type": str,
            "value_getter": norm_combo.currentData,
        }

        weight_combo = QComboBox(group)
        weight_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        weight_options = [
            ("none", self._tr("weight_method_none", "None")),
            ("noise_variance", self._tr("weight_method_noise_variance", "Noise Variance (1/σ²)")),
            ("noise_fwhm", self._tr("weight_method_noise_fwhm", "Noise + FWHM")),
        ]
        for value, label in weight_options:
            weight_combo.addItem(label, value)
        current_weight = str(self.config.get("stacking_weighting_method", "noise_variance"))
        weight_index = next(
            (idx for idx, (value, _label) in enumerate(weight_options) if value == current_weight),
            0,
        )
        weight_combo.setCurrentIndex(weight_index)
        layout.addRow(QLabel(self._tr("stacking_weight_method_label", "Weighting:")), weight_combo)
        self._config_fields["stacking_weighting_method"] = {
            "kind": "combobox",
            "widget": weight_combo,
            "type": str,
            "value_getter": weight_combo.currentData,
        }

        reject_combo = QComboBox(group)
        reject_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        reject_options = [
            ("none", self._tr("reject_algo_none", "None")),
            ("kappa_sigma", self._tr("reject_algo_kappa_sigma", "Kappa-Sigma Clip")),
            (
                "winsorized_sigma_clip",
                self._tr("reject_algo_winsorized_sigma_clip", "Winsorized Sigma Clip"),
            ),
            ("linear_fit_clip", self._tr("reject_algo_linear_fit_clip", "Linear Fit Clip")),
        ]
        for value, label in reject_options:
            reject_combo.addItem(label, value)
        current_reject = str(
            self.config.get("stacking_rejection_algorithm", "winsorized_sigma_clip")
        )
        reject_index = next(
            (idx for idx, (value, _label) in enumerate(reject_options) if value == current_reject),
            0,
        )
        reject_combo.setCurrentIndex(reject_index)
        layout.addRow(QLabel(self._tr("stacking_reject_algo_label", "Rejection Algorithm:")), reject_combo)
        self._config_fields["stacking_rejection_algorithm"] = {
            "kind": "combobox",
            "widget": reject_combo,
            "type": str,
            "value_getter": reject_combo.currentData,
        }

        self._register_double_spinbox(
            "stacking_kappa_low",
            layout,
            self._tr("stacking_kappa_low_label", "Kappa Low:"),
            minimum=0.1,
            maximum=10.0,
            single_step=0.1,
            decimals=2,
        )
        self._register_double_spinbox(
            "stacking_kappa_high",
            layout,
            self._tr("stacking_kappa_high_label", "Kappa High:"),
            minimum=0.1,
            maximum=10.0,
            single_step=0.1,
            decimals=2,
        )

        winsor_value = self.config.get("stacking_winsor_limits", (0.05, 0.05))

        def _format_winsor_pair(low: float, high: float) -> str:
            def _fmt(value: float) -> str:
                text = f"{value:.3f}"
                if "." in text:
                    text = text.rstrip("0").rstrip(".")
                return text or "0"

            return f"{_fmt(low)},{_fmt(high)}"

        def _parse_winsor_value(value: Any) -> Tuple[float, float]:
            if isinstance(value, str):
                parts = [segment.strip() for segment in value.split(",") if segment.strip()]
                if len(parts) >= 2:
                    try:
                        return float(parts[0]), float(parts[1])
                    except ValueError:
                        return (0.05, 0.05)
                return (0.05, 0.05)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                try:
                    return float(value[0]), float(value[1])
                except (TypeError, ValueError):
                    return (0.05, 0.05)
            return (0.05, 0.05)

        parsed_winsor = _parse_winsor_value(winsor_value)
        self.config["stacking_winsor_limits"] = _format_winsor_pair(
            parsed_winsor[0], parsed_winsor[1]
        )

        winsor_container = QWidget(group)
        winsor_layout = QHBoxLayout(winsor_container)
        winsor_layout.setContentsMargins(0, 0, 0, 0)
        winsor_layout.setSpacing(6)

        winsor_low = QDoubleSpinBox(winsor_container)
        winsor_low.setRange(0.0, 0.49)
        winsor_low.setDecimals(3)
        winsor_low.setSingleStep(0.01)
        winsor_low.setValue(parsed_winsor[0])

        winsor_high = QDoubleSpinBox(winsor_container)
        winsor_high.setRange(0.0, 0.49)
        winsor_high.setDecimals(3)
        winsor_high.setSingleStep(0.01)
        winsor_high.setValue(parsed_winsor[1])

        winsor_layout.addWidget(winsor_low)
        winsor_layout.addWidget(QLabel("→", winsor_container))
        winsor_layout.addWidget(winsor_high)

        layout.addRow(
            QLabel(self._tr("stacking_winsor_limits_label", "Winsor Limits (low,high):")),
            winsor_container,
        )

        winsor_note = QLabel(
            self._tr("stacking_winsor_note", "(e.g., 0.05,0.05 for 5% each side)"),
            group,
        )
        winsor_note.setWordWrap(True)
        layout.addRow(QLabel(""), winsor_note)

        self._config_fields["stacking_winsor_limits"] = {
            "kind": "composite",
            "widget": (winsor_low, winsor_high),
            "type": str,
            "value_getter": lambda: _format_winsor_pair(
                float(winsor_low.value()), float(winsor_high.value())
            ),
        }

        combine_combo = QComboBox(group)
        combine_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        combine_options = [
            ("mean", self._tr("combine_method_mean", "Mean")),
            ("median", self._tr("combine_method_median", "Median")),
        ]
        for value, label in combine_options:
            combine_combo.addItem(label, value)
        current_combine = str(self.config.get("stacking_final_combine_method", "mean"))
        combine_index = next(
            (idx for idx, (value, _label) in enumerate(combine_options) if value == current_combine),
            0,
        )
        combine_combo.setCurrentIndex(combine_index)
        layout.addRow(
            QLabel(self._tr("stacking_final_combine_label", "Final Combine:")),
            combine_combo,
        )
        self._config_fields["stacking_final_combine_method"] = {
            "kind": "combobox",
            "widget": combine_combo,
            "type": str,
            "value_getter": combine_combo.currentData,
        }

        self._register_checkbox(
            "poststack_equalize_rgb",
            layout,
            self._tr("stacking_post_equalize_rgb_label", "Equalize RGB (per sub-stack):"),
        )

        radial_checkbox = QCheckBox(
            self._tr("stacking_apply_radial_label", "Apply Radial Weighting:"),
            group,
        )
        radial_checkbox.setChecked(bool(self.config.get("apply_radial_weight", False)))
        layout.addRow(radial_checkbox)
        self._config_fields["apply_radial_weight"] = {
            "kind": "checkbox",
            "widget": radial_checkbox,
            "type": bool,
        }

        radial_feather = QDoubleSpinBox(group)
        radial_feather.setRange(0.1, 1.0)
        radial_feather.setSingleStep(0.05)
        radial_feather.setDecimals(2)
        radial_feather.setValue(float(self.config.get("radial_feather_fraction", 0.8)))
        layout.addRow(
            QLabel(
                self._tr(
                    "stacking_radial_feather_label",
                    "Radial Feather Fraction (0.1-1.0):",
                )
            ),
            radial_feather,
        )
        self._config_fields["radial_feather_fraction"] = {
            "kind": "double_spinbox",
            "widget": radial_feather,
            "type": float,
        }

        radial_floor = QDoubleSpinBox(group)
        radial_floor.setRange(0.0, 0.5)
        radial_floor.setSingleStep(0.01)
        radial_floor.setDecimals(2)
        radial_floor.setValue(float(self.config.get("min_radial_weight_floor", 0.0)))
        layout.addRow(
            QLabel(
                self._tr(
                    "stacking_min_radial_floor_label",
                    "Min Radial Weight Floor (0.0-0.5):",
                )
            ),
            radial_floor,
        )
        radial_floor_note = QLabel(
            self._tr("stacking_min_radial_floor_note", "(0.0 = no floor)"),
            group,
        )
        radial_floor_note.setWordWrap(True)
        layout.addRow(QLabel(""), radial_floor_note)
        self._config_fields["min_radial_weight_floor"] = {
            "kind": "double_spinbox",
            "widget": radial_floor,
            "type": float,
        }

        def _update_radial_controls(enabled: bool) -> None:
            radial_feather.setEnabled(enabled)
            radial_floor.setEnabled(enabled)
            radial_floor_note.setEnabled(enabled)

        _update_radial_controls(radial_checkbox.isChecked())
        radial_checkbox.toggled.connect(_update_radial_controls)  # type: ignore[arg-type]

        return group

    def _create_final_assembly_group(self) -> QGroupBox:
        group = QGroupBox(
            self._tr("qt_group_final_assembly", "Final assembly & output"),
            self,
        )
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        general_box = QGroupBox(
            self._tr("qt_group_final_general", "General output options"),
            group,
        )
        general_layout = QFormLayout(general_box)
        general_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        method_combo = QComboBox(general_box)
        method_options = [
            (
                "reproject_coadd",
                self._tr("qt_final_method_reproject", "Reproject co-add"),
            ),
            (
                "incremental",
                self._tr("qt_final_method_incremental", "Incremental assembly"),
            ),
        ]
        for value, label in method_options:
            method_combo.addItem(label, value)
        current_method = str(self.config.get("final_assembly_method", "reproject_coadd"))
        method_index = next(
            (idx for idx, (value, _label) in enumerate(method_options) if value == current_method),
            0,
        )
        method_combo.setCurrentIndex(method_index)
        general_layout.addRow(
            QLabel(self._tr("qt_field_final_assembly_method", "Assembly method")),
            method_combo,
        )
        self._config_fields["final_assembly_method"] = {
            "kind": "combobox",
            "widget": method_combo,
            "type": str,
            "value_getter": method_combo.currentData,
        }

        self._register_checkbox(
            "match_background_for_final",
            general_layout,
            self._tr("qt_field_match_background", "Match background in final mosaic"),
        )
        self._register_checkbox(
            "save_final_as_uint16",
            general_layout,
            self._tr("qt_field_save_final_uint16", "Save final mosaic as uint16"),
        )
        self._register_checkbox(
            "legacy_rgb_cube",
            general_layout,
            self._tr("qt_field_legacy_rgb_cube", "Legacy RGB cube layout"),
        )
        self._register_checkbox(
            "incremental_feather_parity",
            general_layout,
            self._tr("qt_field_incremental_parity", "Force incremental feather parity"),
        )
        self._register_checkbox(
            "auto_limit_frames_per_master_tile",
            general_layout,
            self._tr("qt_field_auto_limit_frames", "Auto-limit frames per master tile"),
        )
        num_workers_label = QLabel(
            self._tr("num_workers_label", "Processing Threads:"),
            general_box,
        )
        num_workers_spinbox = QSpinBox(general_box)
        cpu_cores = os.cpu_count() or 0
        max_spin_workers = 16
        if cpu_cores:
            max_spin_workers = max(1, cpu_cores) * 2
            if max_spin_workers > 32:
                max_spin_workers = 32
        num_workers_spinbox.setRange(0, max_spin_workers)
        raw_num_workers = self.config.get("num_processing_workers", 0)
        try:
            parsed_num_workers = int(raw_num_workers)
        except (TypeError, ValueError):
            parsed_num_workers = 0
        if parsed_num_workers < 0:
            parsed_num_workers = 0
        if parsed_num_workers > max_spin_workers:
            parsed_num_workers = max_spin_workers
        num_workers_spinbox.setValue(parsed_num_workers)
        self.config["num_processing_workers"] = parsed_num_workers
        general_layout.addRow(num_workers_label, num_workers_spinbox)
        num_workers_note = QLabel(
            self._tr(
                "num_workers_note", "(0 = auto, based on CPU cores)"
            ),
            general_box,
        )
        num_workers_note.setWordWrap(True)
        general_layout.addRow(QLabel(""), num_workers_note)
        self._config_fields["num_processing_workers"] = {
            "kind": "spinbox",
            "widget": num_workers_spinbox,
            "type": int,
        }

        self._register_spinbox(
            "assembly_process_workers",
            general_layout,
            self._tr("qt_field_assembly_workers", "Assembly workers (0 = auto)"),
            minimum=0,
            maximum=64,
        )
        self._register_spinbox(
            "winsor_max_frames_per_pass",
            general_layout,
            self._tr("qt_field_winsor_max_frames", "Winsor max frames / pass (0 = auto)"),
            minimum=0,
            maximum=9999,
        )
        self._register_spinbox(
            "winsor_worker_limit",
            general_layout,
            self._tr("qt_field_winsor_worker_limit", "Winsor worker limit"),
            minimum=1,
            maximum=64,
        )
        self._register_spinbox(
            "max_raw_per_master_tile",
            general_layout,
            self._tr("qt_field_max_raw_per_tile", "Max raw frames per master tile (0 = unlimited)"),
            minimum=0,
            maximum=9999,
        )

        layout.addWidget(general_box)

        intertile_box = QGroupBox(
            self._tr("qt_group_intertile", "Intertile blending"),
            group,
        )
        intertile_layout = QFormLayout(intertile_box)
        intertile_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._register_checkbox(
            "intertile_photometric_match",
            intertile_layout,
            self._tr("qt_field_intertile_match", "Photometric match between tiles"),
        )
        self._register_checkbox(
            "use_auto_intertile",
            intertile_layout,
            self._tr("qt_field_use_auto_intertile", "Auto-adjust intertile parameters"),
        )
        self._register_spinbox(
            "intertile_preview_size",
            intertile_layout,
            self._tr("qt_field_intertile_preview", "Preview size (px)"),
            minimum=128,
            maximum=4096,
            single_step=64,
        )
        self._register_double_spinbox(
            "intertile_overlap_min",
            intertile_layout,
            self._tr("qt_field_intertile_overlap", "Minimum overlap"),
            minimum=0.0,
            maximum=1.0,
            single_step=0.01,
            decimals=2,
        )
        self._register_double_pair(
            "intertile_sky_percentile",
            intertile_layout,
            self._tr("qt_field_intertile_sky", "Sky percentile range"),
            minimum=0.0,
            maximum=100.0,
            single_step=0.5,
            decimals=1,
            default=(30.0, 70.0),
        )
        self._register_double_spinbox(
            "intertile_robust_clip_sigma",
            intertile_layout,
            self._tr("qt_field_intertile_clip", "Robust clip σ"),
            minimum=0.1,
            maximum=10.0,
            single_step=0.1,
            decimals=2,
        )
        self._register_checkbox(
            "intertile_global_recenter",
            intertile_layout,
            self._tr("qt_field_intertile_recenter", "Global recenter after stacking"),
        )
        self._register_double_pair(
            "intertile_recenter_clip",
            intertile_layout,
            self._tr("qt_field_intertile_recenter_clip", "Recenter clip range"),
            minimum=0.1,
            maximum=5.0,
            single_step=0.01,
            decimals=2,
            default=(0.85, 1.18),
        )

        layout.addWidget(intertile_box)

        center_box = QGroupBox(
            self._tr("qt_group_center_out", "Center-out normalization"),
            group,
        )
        center_layout = QFormLayout(center_box)
        center_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._register_checkbox(
            "center_out_normalization_p3",
            center_layout,
            self._tr("qt_field_center_out_enable", "Enable center-out normalization"),
        )
        self._register_spinbox(
            "p3_center_preview_size",
            center_layout,
            self._tr("qt_field_center_preview", "Preview size (px)"),
            minimum=64,
            maximum=2048,
            single_step=32,
        )
        self._register_double_spinbox(
            "p3_center_min_overlap_fraction",
            center_layout,
            self._tr("qt_field_center_overlap", "Minimum overlap fraction"),
            minimum=0.0,
            maximum=1.0,
            single_step=0.01,
            decimals=2,
        )
        self._register_double_pair(
            "p3_center_sky_percentile",
            center_layout,
            self._tr("qt_field_center_sky", "Sky percentile range"),
            minimum=0.0,
            maximum=100.0,
            single_step=1.0,
            decimals=1,
            default=(25.0, 60.0),
        )
        self._register_double_spinbox(
            "p3_center_robust_clip_sigma",
            center_layout,
            self._tr("qt_field_center_clip", "Robust clip σ"),
            minimum=0.1,
            maximum=10.0,
            single_step=0.1,
            decimals=2,
        )

        anchor_mode_combo = QComboBox(center_box)
        anchor_mode_options = [
            (
                "auto_central_quality",
                self._tr("qt_center_anchor_auto", "Auto central quality"),
            ),
            (
                "central_only",
                self._tr("qt_center_anchor_central", "Central only"),
            ),
        ]
        for value, label in anchor_mode_options:
            anchor_mode_combo.addItem(label, value)
        current_anchor_mode = str(self.config.get("center_out_anchor_mode", "auto_central_quality"))
        anchor_mode_index = next(
            (idx for idx, (value, _label) in enumerate(anchor_mode_options) if value == current_anchor_mode),
            0,
        )
        anchor_mode_combo.setCurrentIndex(anchor_mode_index)
        center_layout.addRow(
            QLabel(self._tr("qt_field_center_anchor_mode", "Anchor mode")),
            anchor_mode_combo,
        )
        self._config_fields["center_out_anchor_mode"] = {
            "kind": "combobox",
            "widget": anchor_mode_combo,
            "type": str,
            "value_getter": anchor_mode_combo.currentData,
        }

        self._register_spinbox(
            "anchor_quality_probe_limit",
            center_layout,
            self._tr("qt_field_anchor_probe_limit", "Anchor probe limit"),
            minimum=1,
            maximum=50,
        )
        self._register_double_pair(
            "anchor_quality_span_range",
            center_layout,
            self._tr("qt_field_anchor_span", "Anchor span range"),
            minimum=0.0,
            maximum=20.0,
            single_step=0.01,
            decimals=3,
            default=(0.02, 6.0),
        )
        self._register_double_spinbox(
            "anchor_quality_median_clip_sigma",
            center_layout,
            self._tr("qt_field_anchor_clip", "Anchor median clip σ"),
            minimum=0.1,
            maximum=10.0,
            single_step=0.1,
            decimals=2,
        )

        layout.addWidget(center_box)

        post_box = QGroupBox(
            self._tr("qt_group_poststack", "Post-stack anchor review"),
            group,
        )
        post_layout = QFormLayout(post_box)
        post_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._register_checkbox(
            "enable_poststack_anchor_review",
            post_layout,
            self._tr("qt_field_poststack_enable", "Enable post-stack review"),
        )
        self._register_spinbox(
            "poststack_anchor_probe_limit",
            post_layout,
            self._tr("qt_field_poststack_probe", "Probe limit"),
            minimum=1,
            maximum=20,
        )
        self._register_double_pair(
            "poststack_anchor_span_range",
            post_layout,
            self._tr("qt_field_poststack_span", "Span range"),
            minimum=0.0,
            maximum=20.0,
            single_step=0.01,
            decimals=3,
            default=(0.004, 10.0),
        )
        self._register_double_spinbox(
            "poststack_anchor_median_clip_sigma",
            post_layout,
            self._tr("qt_field_poststack_clip", "Median clip σ"),
            minimum=0.1,
            maximum=10.0,
            single_step=0.1,
            decimals=2,
        )
        self._register_double_spinbox(
            "poststack_anchor_min_improvement",
            post_layout,
            self._tr("qt_field_poststack_min_improve", "Min. improvement"),
            minimum=0.0,
            maximum=1.0,
            single_step=0.01,
            decimals=2,
        )
        self._register_checkbox(
            "poststack_anchor_use_overlap_affine",
            post_layout,
            self._tr("qt_field_poststack_use_overlap", "Use overlap affine adjustment"),
        )

        layout.addWidget(post_box)

        return group

    def _create_system_resources_group(self) -> QGroupBox:
        group = QGroupBox(
            self._tr("qt_group_system_resources", "System resources & cache"),
            self,
        )
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._register_checkbox(
            "coadd_use_memmap",
            layout,
            self._tr("qt_field_coadd_use_memmap", "Use memory-mapped coadd intermediates"),
        )
        self._register_checkbox(
            "coadd_cleanup_memmap",
            layout,
            self._tr("qt_field_coadd_cleanup", "Clean up memmap files after run"),
        )
        self._register_directory_picker(
            "coadd_memmap_dir",
            layout,
            self._tr("qt_field_coadd_memmap_dir", "Memmap directory"),
            dialog_title=self._tr(
                "qt_dialog_select_memmap_dir", "Select memmap directory"
            ),
        )

        cache_combo = QComboBox(group)
        cache_options = [
            ("run_end", self._tr("qt_cache_retention_run_end", "Release caches at run end")),
            ("per_tile", self._tr("qt_cache_retention_per_tile", "Clear caches after each tile")),
            ("keep", self._tr("qt_cache_retention_keep", "Keep caches between runs")),
        ]
        for value, label in cache_options:
            cache_combo.addItem(label, value)
        current_cache_mode = str(self.config.get("cache_retention", "run_end")).lower()
        cache_index = next(
            (idx for idx, (value, _label) in enumerate(cache_options) if value == current_cache_mode),
            0,
        )
        cache_combo.setCurrentIndex(cache_index)
        layout.addRow(
            QLabel(self._tr("qt_field_cache_retention", "Cache retention")),
            cache_combo,
        )
        self._config_fields["cache_retention"] = {
            "kind": "combobox",
            "widget": cache_combo,
            "type": str,
        }

        return group

    def _create_skin_group(self) -> QGroupBox:
        group = QGroupBox(
            self._tr("qt_group_skin_theme", "Theme"),
            self,
        )
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        combo = QComboBox(group)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        options = [
            ("system", self._tr("qt_theme_option_system", "System default")),
            ("dark", self._tr("qt_theme_option_dark", "Dark")),
            ("light", self._tr("qt_theme_option_light", "Light")),
        ]
        for value, label in options:
            combo.addItem(label, value)
        current_mode = str(self.config.get("qt_theme_mode", "system") or "system").lower()
        index = next(
            (idx for idx, (value, _label) in enumerate(options) if value == current_mode),
            0,
        )
        combo.setCurrentIndex(index)
        layout.addRow(
            QLabel(self._tr("qt_field_theme_mode", "Theme mode"), group),
            combo,
        )
        self.theme_mode_combo = combo
        combo.currentIndexChanged.connect(self._on_theme_mode_changed)  # type: ignore[arg-type]
        self._config_fields["qt_theme_mode"] = {
            "kind": "combobox",
            "widget": combo,
            "type": str,
            "value_getter": combo.currentData,
        }

        return group

    def _create_backend_group(self) -> QGroupBox:
        group = QGroupBox(
            self._tr("qt_group_backend_title", "Preferred GUI backend"),
            self,
        )
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        backend_options = [
            ("tk", self._tr("backend_option_tk", "Classic Tk GUI (stable)")),
            ("qt", self._tr("backend_option_qt", "Qt GUI (preview)")),
        ]
        combo = QComboBox(group)
        for value, label in backend_options:
            combo.addItem(label, value)

        current_backend = str(self.config.get("preferred_gui_backend", "tk")).strip().lower()
        index = next(
            (idx for idx, (value, _label) in enumerate(backend_options) if value == current_backend),
            0,
        )
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

        layout.addRow(
            QLabel(self._tr("qt_field_preferred_backend", "Preferred backend"), group),
            combo,
        )

        notice_label = QLabel(
            self._tr(
                "backend_change_notice",
                "Backend change will take effect next time you launch ZeMosaic.",
            ),
            group,
        )
        notice_label.setWordWrap(True)
        layout.addRow(notice_label)

        combo.currentIndexChanged.connect(self._on_backend_selection_changed)  # type: ignore[arg-type]
        self.backend_combo = combo

        return group

    def _create_language_group(self) -> QGroupBox:
        group = QGroupBox(
            self._tr("qt_group_language_title", "Language"),
            self,
        )
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        language_label = QLabel(self._tr("language_selector_label", "Language:"), group)

        combo = QComboBox(group)
        available_langs = set(self._resolve_available_languages())
        option_entries: List[Tuple[str, str]] = []
        for code, key, fallback in LANGUAGE_OPTION_DEFINITIONS:
            if code in available_langs or not available_langs:
                option_entries.append((code, self._tr(key, fallback)))
        if not option_entries:
            option_entries = [
                ("en", self._tr("language_name_en", "English (EN)")),
                ("fr", self._tr("language_name_fr", "Français (FR)")),
            ]
        for code, label_text in option_entries:
            combo.addItem(label_text, code)

        current_lang = str(self.config.get("language", "en"))
        option_codes = [code for code, _ in option_entries]
        if current_lang not in option_codes:
            current_lang = option_codes[0]
            self.config["language"] = current_lang

        combo.blockSignals(True)
        combo.setCurrentIndex(option_codes.index(current_lang))
        combo.blockSignals(False)

        layout.addRow(language_label, combo)

        notice_label = QLabel(
            self._tr(
                "language_change_notice",
                "Language also applies to the classic Tk interface and will be remembered.",
            ),
            group,
        )
        notice_label.setWordWrap(True)
        layout.addRow(notice_label)

        combo.currentIndexChanged.connect(self._on_language_combo_changed)  # type: ignore[arg-type]
        self.language_combo = combo

        return group

    def _create_gpu_group(self) -> QGroupBox:
        group = QGroupBox(
            self._tr("qt_group_gpu", "GPU and acceleration"),
            self,
        )
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._register_checkbox(
            "use_gpu_phase5",
            layout,
            self._tr("qt_field_use_gpu_phase5", "Use GPU acceleration when available"),
        )
        checkbox_binding = self._config_fields.get("use_gpu_phase5")
        checkbox_widget: QCheckBox | None = None
        if isinstance(checkbox_binding, dict):
            widget_candidate = checkbox_binding.get("widget")
            if isinstance(widget_candidate, QCheckBox):
                checkbox_widget = widget_candidate
        self._register_gpu_selector(
            layout,
            self._tr("qt_field_gpu_selector", "GPU selector"),
            checkbox=checkbox_widget,
        )

        return group

    def _on_theme_mode_changed(self, index: int) -> None:
        if self.theme_mode_combo is None:
            return
        data = self.theme_mode_combo.itemData(index)
        if not isinstance(data, str):
            data = str(data or "")
        mode = data.strip().lower() or "system"
        self.config["qt_theme_mode"] = mode
        self._apply_theme(mode)

    def _on_backend_selection_changed(self, index: int) -> None:
        if self.backend_combo is None:
            return
        data = self.backend_combo.itemData(index)
        backend = str(data or "").strip().lower()
        if backend not in {"tk", "qt"}:
            return
        current = str(self.config.get("preferred_gui_backend", "tk")).strip().lower()
        if backend == current:
            return
        self.config["preferred_gui_backend"] = backend
        self.config["preferred_gui_backend_explicit"] = True
        self._save_config()

    def _on_language_combo_changed(self, index: int) -> None:
        if self.language_combo is None:
            return
        data = self.language_combo.itemData(index)
        lang = str(data or "").strip()
        if not lang or lang == str(self.config.get("language", "en")).strip():
            return
        self._apply_language_selection(lang)

    def _apply_theme(self, mode: str | None) -> None:
        mode = (mode or "system").strip().lower()
        palette = QApplication.instance().style().standardPalette()
        stylesheet = ""
        if mode == "dark":
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(45, 45, 45))
            palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
            palette.setColor(QPalette.Base, QColor(35, 35, 35))
            palette.setColor(QPalette.AlternateBase, QColor(55, 55, 55))
            palette.setColor(QPalette.Text, QColor(220, 220, 220))
            palette.setColor(QPalette.Button, QColor(55, 55, 55))
            palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
            palette.setColor(QPalette.Highlight, QColor(100, 150, 255))
            palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
            palette.setColor(QPalette.Link, QColor(80, 140, 255))
        elif mode == "light":
            palette = QApplication.instance().style().standardPalette()
            palette.setColor(QPalette.Window, QColor(250, 250, 250))
            palette.setColor(QPalette.Base, QColor(255, 255, 255))
            palette.setColor(QPalette.Button, QColor(245, 245, 245))
            palette.setColor(QPalette.Highlight, QColor(80, 120, 255))
            palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        QApplication.setPalette(palette)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(stylesheet)

    def _create_logging_group(self) -> QGroupBox:
        group = QGroupBox(self._tr("qt_group_logging", "Logging / progress"), self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        options_layout = QHBoxLayout()
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(6)

        log_level_label = QLabel(self._tr("qt_logging_level", "Logging level:"))
        options_layout.addWidget(log_level_label)

        log_level_combo = QComboBox(group)
        level_options = [
            ("ERROR", self._tr("logging_level_error", "Error")),
            ("WARN", self._tr("logging_level_warn", "Warn")),
            ("INFO", self._tr("logging_level_info", "Info")),
            ("DEBUG", self._tr("logging_level_debug", "Debug")),
        ]
        for value, label in level_options:
            log_level_combo.addItem(label, value)
        current_level = str(self.config.get("logging_level", "INFO")).upper()
        current_index = next(
            (idx for idx, (value, _label) in enumerate(level_options) if value == current_level),
            2,
        )
        log_level_combo.setCurrentIndex(current_index)

        def _on_level_changed(index: int) -> None:
            selected = log_level_combo.itemData(index)
            if not selected:
                selected = "INFO"
            self.config["logging_level"] = str(selected)

        log_level_combo.currentIndexChanged.connect(_on_level_changed)  # type: ignore[arg-type]
        options_layout.addWidget(log_level_combo)
        options_layout.addStretch(1)

        clear_log_button = QPushButton(self._tr("qt_button_clear_log", "Clear log"), group)
        clear_log_button.clicked.connect(self._clear_log)  # type: ignore[arg-type]
        options_layout.addWidget(clear_log_button)

        layout.addLayout(options_layout)

        self.progress_bar = QProgressBar(group)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        stats_layout = QGridLayout()
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setHorizontalSpacing(12)
        stats_layout.setVerticalSpacing(4)

        eta_label = QLabel(self._tr("qt_progress_eta_label", "ETA:"), group)
        self.eta_value_label = QLabel(self._tr("qt_progress_placeholder", "—"), group)
        stats_layout.addWidget(eta_label, 0, 0)
        stats_layout.addWidget(self.eta_value_label, 0, 1)

        elapsed_label = QLabel(self._tr("qt_progress_elapsed_label", "Elapsed:"), group)
        self.elapsed_value_label = QLabel(self._tr("qt_progress_placeholder", "—"), group)
        stats_layout.addWidget(elapsed_label, 0, 2)
        stats_layout.addWidget(self.elapsed_value_label, 0, 3)

        tiles_label = QLabel(self._tr("qt_progress_tiles_label", "Tiles:"), group)
        self.tiles_value_label = QLabel(
            self._tr("qt_progress_count_placeholder", "0 / 0"),
            group,
        )
        stats_layout.addWidget(tiles_label, 0, 4)
        stats_layout.addWidget(self.tiles_value_label, 0, 5)

        files_label = QLabel(
            self._tr("qt_progress_files_label", "Files remaining:"),
            group,
        )
        self.files_value_label = QLabel("", group)
        stats_layout.addWidget(files_label, 1, 0)
        stats_layout.addWidget(self.files_value_label, 1, 1)

        phase_label = QLabel(self._tr("qt_progress_phase_label", "Phase:"), group)
        self.phase_value_label = QLabel(self._tr("qt_progress_placeholder", "Idle"), group)
        stats_layout.addWidget(phase_label, 1, 2)
        stats_layout.addWidget(self.phase_value_label, 1, 3)

        layout.addLayout(stats_layout)

        self.log_output = QPlainTextEdit(group)
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText(
            self._tr(
                "qt_log_placeholder",
                "Worker output will appear here once the integration is complete.",
            )
        )
        self.log_output.setMinimumHeight(180)
        layout.addWidget(self.log_output)

        self.log_level_combo = log_level_combo
        self.clear_log_button = clear_log_button
        self._config_fields["logging_level"] = {
            "kind": "combobox",
            "widget": log_level_combo,
            "type": str,
            "value_getter": log_level_combo.currentData,
        }

        self._phase45_overlay_enabled = False
        self._phase45_reset_overlay()

        return group

    # ------------------------------------------------------------------
    # Configuration handling
    # ------------------------------------------------------------------
    def _detect_gpus(self) -> List[Tuple[str, int | None]]:
        """Return detected GPU devices as ``(label, index)`` tuples."""

        controllers: List[str] = []
        if IS_WINDOWS and wmi is not None:
            try:  # pragma: no cover - hardware/OS specific code
                wmi_client = wmi.WMI()  # type: ignore[attr-defined]
                controllers = [str(entry.Name) for entry in wmi_client.Win32_VideoController()]
            except Exception:
                controllers = []

        if not controllers and shutil.which("nvidia-smi"):
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                controllers = [line.strip() for line in output.splitlines() if line.strip()]
            except Exception:
                controllers = []

        cuda_names: List[str] = []
        if CUPY_AVAILABLE:
            try:  # pragma: no cover - optional dependency
                import cupy  # type: ignore[import]
                from cupy.cuda.runtime import getDeviceCount, getDeviceProperties  # type: ignore[attr-defined]

                for idx in range(getDeviceCount()):
                    props = getDeviceProperties(idx)
                    name = props.get("name")
                    if isinstance(name, bytes):
                        name = name.decode(errors="ignore")
                    cuda_names.append(str(name))
            except Exception:
                cuda_names = []

        def _simplify(name: str) -> str:
            return name.lower().replace("laptop gpu", "").strip()

        simple_cuda = [_simplify(name) for name in cuda_names]
        gpus: List[Tuple[str, int | None]] = []

        for display in controllers:
            simplified = _simplify(display)
            index = simple_cuda.index(simplified) if simplified in simple_cuda else None
            gpus.append((display, index))

        if not gpus and cuda_names:
            gpus = [(name, idx) for idx, name in enumerate(cuda_names)]

        cpu_label = "CPU (no GPU)"
        gpus.insert(0, (cpu_label, None))

        unique: List[Tuple[str, int | None]] = []
        seen: set[Tuple[str, int | None]] = set()
        for entry in gpus:
            if entry in seen:
                continue
            seen.add(entry)
            unique.append(entry)
        return unique

    def _register_directory_picker(
        self,
        key: str,
        layout: QFormLayout,
        label_text: str,
        *,
        dialog_title: str,
    ) -> None:
        container = QWidget()
        row_layout = QHBoxLayout(container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        line_edit = QLineEdit()
        current_value = str(self.config.get(key) or "")
        line_edit.setText(current_value)

        browse_button = QPushButton(self._tr("qt_button_browse", "Browse…"))

        def _on_browse_clicked() -> None:
            start_dir = line_edit.text().strip() or current_value or os.getcwd()
            selected_dir = QFileDialog.getExistingDirectory(
                self,
                dialog_title,
                start_dir,
            )
            if selected_dir:
                line_edit.setText(selected_dir)
                self.config[key] = selected_dir

        browse_button.clicked.connect(_on_browse_clicked)  # type: ignore[arg-type]

        row_layout.addWidget(line_edit)
        row_layout.addWidget(browse_button)
        layout.addRow(QLabel(label_text), container)

        self._config_fields[key] = {
            "kind": "line_edit",
            "widget": line_edit,
            "type": str,
        }

    def _register_line_edit(
        self,
        key: str,
        layout: QFormLayout,
        label_text: str,
        *,
        browse_action: str | None = None,
        dialog_title: str | None = None,
    ) -> None:
        widget = QLineEdit()
        current_value = self.config.get(key)
        widget.setText("" if current_value is None else str(current_value))

        if browse_action is not None:
            container = QWidget()
            row_layout = QHBoxLayout(container)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            row_layout.addWidget(widget)
            browse_button = QPushButton(self._tr("qt_button_browse", "Browse…"))

            def _on_browse() -> None:
                start_path = widget.text().strip() or str(current_value or "")
                if not start_path:
                    start_path = os.getcwd()
                if browse_action == "file":
                    selected = QFileDialog.getOpenFileName(
                        self,
                        dialog_title
                        or self._tr("qt_dialog_select_file", "Select File"),
                        start_path,
                    )[0]
                else:
                    selected = QFileDialog.getExistingDirectory(
                        self,
                        dialog_title
                        or self._tr("qt_dialog_select_directory", "Select Directory"),
                        start_path,
                    )
                if selected:
                    widget.setText(selected)
                    self.config[key] = selected

            browse_button.clicked.connect(_on_browse)  # type: ignore[arg-type]
            row_layout.addWidget(browse_button)
            layout.addRow(QLabel(label_text), container)
        else:
            layout.addRow(QLabel(label_text), widget)

        type_source = self._default_config_values.get(key, current_value)
        if type_source is None:
            value_type = str
        else:
            value_type = type(type_source)
        self._config_fields[key] = {
            "kind": "line_edit",
            "widget": widget,
            "type": value_type,
        }

    def _register_checkbox(self, key: str, layout: QFormLayout, label_text: str) -> None:
        checkbox = QCheckBox(label_text)
        checkbox.setChecked(bool(self.config.get(key, False)))
        layout.addRow(checkbox)
        self._config_fields[key] = {
            "kind": "checkbox",
            "widget": checkbox,
            "type": bool,
        }

    def _register_spinbox(
        self,
        key: str,
        layout: QFormLayout,
        label_text: str,
        *,
        minimum: int,
        maximum: int,
        single_step: int | None = None,
    ) -> None:
        spinbox = QSpinBox()
        spinbox.setRange(minimum, maximum)
        if single_step is not None:
            spinbox.setSingleStep(single_step)
        current_value = int(self.config.get(key, minimum))
        spinbox.setValue(current_value)
        layout.addRow(QLabel(label_text), spinbox)
        self._config_fields[key] = {
            "kind": "spinbox",
            "widget": spinbox,
            "type": int,
        }

    def _register_double_spinbox(
        self,
        key: str,
        layout: QFormLayout,
        label_text: str,
        *,
        minimum: float,
        maximum: float,
        single_step: float,
        decimals: int | None = None,
    ) -> None:
        spinbox = QDoubleSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setSingleStep(single_step)
        if decimals is not None:
            spinbox.setDecimals(decimals)
        else:
            spinbox.setDecimals(1 if single_step < 1 else 0)
        current_value = float(self.config.get(key, minimum))
        spinbox.setValue(current_value)
        layout.addRow(QLabel(label_text), spinbox)
        self._config_fields[key] = {
            "kind": "double_spinbox",
            "widget": spinbox,
            "type": float,
        }

    def _register_double_pair(
        self,
        key: str,
        layout: QFormLayout,
        label_text: str,
        *,
        minimum: float,
        maximum: float,
        single_step: float,
        decimals: int = 2,
        default: Sequence[float] | None = None,
        separator_text: str = "→",
    ) -> None:
        container = QWidget()
        row_layout = QHBoxLayout(container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        first_spin = QDoubleSpinBox(container)
        first_spin.setRange(minimum, maximum)
        first_spin.setSingleStep(single_step)
        first_spin.setDecimals(decimals)

        second_spin = QDoubleSpinBox(container)
        second_spin.setRange(minimum, maximum)
        second_spin.setSingleStep(single_step)
        second_spin.setDecimals(decimals)

        pair_value = self.config.get(key, default if default is not None else (minimum, minimum))
        if not (
            isinstance(pair_value, (list, tuple))
            and len(pair_value) >= 2
        ):
            pair_value = default if default is not None else (minimum, minimum)
            self.config[key] = list(pair_value)

        first_spin.setValue(float(pair_value[0]))
        second_spin.setValue(float(pair_value[1]))

        row_layout.addWidget(first_spin)
        if separator_text:
            row_layout.addWidget(QLabel(separator_text, container))
        row_layout.addWidget(second_spin)

        layout.addRow(QLabel(label_text), container)

        self._config_fields[key] = {
            "kind": "composite",
            "widget": (first_spin, second_spin),
            "type": list,
            "value_getter": lambda fs=first_spin, ss=second_spin: [
                float(fs.value()),
                float(ss.value()),
            ],
        }

    def _register_gpu_selector(
        self,
        layout: QFormLayout,
        label_text: str,
        *,
        checkbox: QCheckBox | None = None,
    ) -> None:
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)

        cpu_canonical = "CPU (no GPU)"
        entries: List[Dict[str, Any]] = []
        devices = self._gpu_devices or [(cpu_canonical, None)]

        seen: set[Tuple[str, int | None]] = set()
        for display, idx in devices:
            canonical = display or cpu_canonical
            display_text = display
            if idx is None and display.strip().lower() == cpu_canonical.lower():
                display_text = self._tr("qt_gpu_option_cpu", cpu_canonical)
                canonical = cpu_canonical
            key = (canonical, idx)
            if key in seen:
                continue
            seen.add(key)
            entries.append({"display": display_text, "selector": canonical, "id": idx})

        for entry in entries:
            combo.addItem(entry["display"], entry)

        stored_selector = str(self.config.get("gpu_selector") or "").strip()
        stored_id = self.config.get("gpu_id_phase5")
        target_index = -1
        for row in range(combo.count()):
            payload = combo.itemData(row)
            if isinstance(payload, dict):
                selector = str(payload.get("selector") or "")
                if stored_selector and selector == stored_selector:
                    target_index = row
                    break
        if target_index < 0 and isinstance(stored_id, int):
            for row in range(combo.count()):
                payload = combo.itemData(row)
                if isinstance(payload, dict) and payload.get("id") == stored_id:
                    target_index = row
                    break
        if target_index < 0 and combo.count() > 0:
            target_index = 0
        if target_index >= 0:
            combo.setCurrentIndex(target_index)
            payload = combo.itemData(target_index)
            if isinstance(payload, dict) and payload.get("selector"):
                self.config.setdefault("gpu_selector", str(payload["selector"]))

        label = QLabel(label_text)
        layout.addRow(label, combo)

        def _selector_getter(combo_ref: QComboBox = combo) -> str:
            payload = combo_ref.currentData()
            if isinstance(payload, dict) and payload.get("selector") is not None:
                return str(payload["selector"])
            text = combo_ref.currentText().strip()
            return text

        def _gpu_id_getter(combo_ref: QComboBox = combo) -> int | None:
            payload = combo_ref.currentData()
            if isinstance(payload, dict):
                value = payload.get("id")
                if isinstance(value, int):
                    return value
                if value in (None, ""):
                    return None
            text = combo_ref.currentText().strip()
            if not text:
                return None
            try:
                return int(text)
            except ValueError:
                return None

        self._config_fields["gpu_selector"] = {
            "kind": "combobox",
            "widget": combo,
            "type": str,
            "value_getter": _selector_getter,
        }
        self._config_fields["gpu_id_phase5"] = {
            "kind": "combobox",
            "widget": combo,
            "type": int,
            "value_getter": _gpu_id_getter,
        }

        def _update_enabled(state: bool) -> None:
            combo.setEnabled(state)
            label.setEnabled(state)

        if checkbox is not None:
            _update_enabled(bool(checkbox.isChecked()))
            checkbox.toggled.connect(_update_enabled)  # type: ignore[arg-type]
        else:
            _update_enabled(True)

    @staticmethod
    def _normalize_config_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            if not normalized:
                return default
        if value is None:
            return default
        return bool(value)

    def _synchronize_gpu_config_keys(
        self, mapping: MutableMapping[str, Any] | None = None
    ) -> bool:
        target = mapping if mapping is not None else getattr(self, "config", None)
        if target is None:
            return False

        canonical_value: Any = target.get("use_gpu_phase5")
        if canonical_value in (None, ""):
            for legacy_key in ("stack_use_gpu", "use_gpu_stack"):
                legacy_val = target.get(legacy_key)
                if legacy_val not in (None, ""):
                    canonical_value = legacy_val
                    break

        gpu_enabled = self._normalize_config_bool(canonical_value, False)
        changed = False
        for key in ("use_gpu_phase5", "stack_use_gpu", "use_gpu_stack"):
            previous = target.get(key)
            if previous != gpu_enabled or not isinstance(previous, bool):
                changed = True
            target[key] = gpu_enabled
        return changed

    def _disable_phase45_config(
        self, mapping: MutableMapping[str, Any] | None = None
    ) -> None:
        """
        Ensure the Qt GUI never enables Phase 4.5 / super-tiles.

        Even if a config file or legacy Tk session left ``inter_master_merge_enable``
        enabled, the Qt backend must force it off so the worker never receives a
        True value from this interface (Task O regression guard).
        """
        target = mapping if mapping is not None else getattr(self, "config", None)
        if target is None:
            return
        target["inter_master_merge_enable"] = False

    def _collect_config_from_widgets(self) -> None:
        for key, binding in self._config_fields.items():
            kind = binding["kind"]
            widget = binding["widget"]
            expected_type = binding["type"]
            value_getter = binding.get("value_getter")
            if value_getter is not None:
                raw_value = value_getter()
            elif kind == "checkbox":
                raw_value = bool(widget.isChecked())
            elif kind == "spinbox":
                raw_value = int(widget.value())
            elif kind == "double_spinbox":
                raw_value = float(widget.value())
            elif kind == "combobox":
                data = widget.currentData()
                raw_value = data if data is not None else widget.currentText()
            else:
                raw_text = widget.text().strip()
                if expected_type in {int, float}:
                    try:
                        raw_value = expected_type(raw_text)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        raw_value = self.config.get(key)
                elif expected_type is bool:
                    normalized = raw_text.lower()
                    if normalized in {"1", "true", "yes", "on"}:
                        raw_value = True
                    elif normalized in {"0", "false", "no", "off"}:
                        raw_value = False
                    else:
                        raw_value = self.config.get(key, False)
                else:
                    raw_value = raw_text

            postprocess = binding.get("postprocess")
            if callable(postprocess):
                try:
                    raw_value = postprocess(raw_value)
                except Exception:
                    raw_value = self.config.get(key, raw_value)

            self.config[key] = raw_value
        self._synchronize_gpu_config_keys()
        self._disable_phase45_config()

    def _baseline_default_config(self) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        if zemosaic_config is not None and hasattr(zemosaic_config, "DEFAULT_CONFIG"):
            try:
                defaults.update(dict(getattr(zemosaic_config, "DEFAULT_CONFIG")))
            except Exception:
                defaults = {}
        fallback_defaults: Dict[str, Any] = {
            "input_dir": "",
            "output_dir": "",
            "global_wcs_output_path": "global_mosaic_wcs.fits",
            "coadd_memmap_dir": "",
            "coadd_use_memmap": True,
            "coadd_cleanup_memmap": True,
            "auto_detect_seestar": True,
            "force_seestar_mode": False,
            "sds_mode_default": False,
            "sds_coverage_threshold": 0.92,
            "solver_method": "ansvr",
            "astrometry_api_key": "",
            "astrometry_timeout": 60,
            "astrometry_downsample": 2,
            "astap_executable_path": "",
            "astap_data_directory_path": "",
            "astap_default_search_radius": 3.0,
            "astap_default_downsample": 2,
            "astap_default_sensitivity": 100,
            "astap_max_instances": 1,
            "stacking_normalize_method": "linear_fit",
            "stacking_weighting_method": "noise_variance",
            "stacking_rejection_algorithm": "winsorized_sigma_clip",
            "stacking_kappa_low": 3.0,
            "stacking_kappa_high": 3.0,
            "stacking_winsor_limits": "0.05,0.05",
            "stacking_final_combine_method": "mean",
            "poststack_equalize_rgb": True,
            "apply_radial_weight": False,
            "radial_feather_fraction": 0.8,
            "min_radial_weight_floor": 0.0,
            "radial_shape_power": 2.0,
            "cluster_panel_threshold": 0.05,
            "cluster_target_groups": 0,
            "cluster_orientation_split_deg": 0.0,
            "inter_master_merge_enable": False,
            "inter_master_overlap_threshold": 0.60,
            "inter_master_stack_method": "winsor",
            "inter_master_min_group_size": 2,
            "inter_master_max_group": 64,
            "inter_master_memmap_policy": "auto",
            "inter_master_local_scale": "final",
            "inter_master_photometry_intragroup": True,
            "inter_master_photometry_intersuper": True,
            "inter_master_photometry_clip_sigma": 3.0,
            "cache_retention": "run_end",
            "quality_crop_enabled": False,
            "quality_crop_band_px": 32,
            "quality_crop_k_sigma": 2.0,
            "quality_crop_margin_px": 8,
            "quality_crop_min_run": 2,
            "crop_follow_signal": True,
            "altaz_cleanup_enabled": False,
            "altaz_margin_percent": 5.0,
            "altaz_decay": 0.15,
            "altaz_nanize": True,
            "quality_gate_enabled": False,
            "quality_gate_threshold": 0.48,
            "quality_gate_edge_band_px": 64,
            "quality_gate_k_sigma": 2.5,
            "quality_gate_erode_px": 3,
            "quality_gate_move_rejects": True,
            "two_pass_coverage_renorm": False,
            "two_pass_cov_sigma_px": 50,
            "two_pass_cov_gain_clip": [0.85, 1.18],
            "apply_master_tile_crop": True,
            "master_tile_crop_percent": 3.0,
            "match_background_for_final": True,
            "incremental_feather_parity": False,
            "intertile_photometric_match": True,
            "intertile_preview_size": 512,
            "intertile_overlap_min": 0.05,
            "intertile_sky_percentile": [30.0, 70.0],
            "intertile_robust_clip_sigma": 2.5,
            "intertile_global_recenter": True,
            "intertile_recenter_clip": [0.85, 1.18],
            "use_auto_intertile": False,
            "center_out_normalization_p3": True,
            "p3_center_preview_size": 256,
            "p3_center_min_overlap_fraction": 0.03,
            "p3_center_sky_percentile": [25.0, 60.0],
            "p3_center_robust_clip_sigma": 2.5,
            "center_out_anchor_mode": "auto_central_quality",
            "anchor_quality_probe_limit": 12,
            "anchor_quality_span_range": [0.02, 6.0],
            "anchor_quality_median_clip_sigma": 2.5,
            "enable_poststack_anchor_review": True,
            "poststack_anchor_probe_limit": 8,
            "poststack_anchor_span_range": [0.004, 10.0],
            "poststack_anchor_median_clip_sigma": 3.5,
            "poststack_anchor_min_improvement": 0.12,
            "poststack_anchor_use_overlap_affine": True,
            "save_final_as_uint16": False,
            "legacy_rgb_cube": False,
            "assembly_process_workers": 0,
            "auto_limit_frames_per_master_tile": True,
            "winsor_max_frames_per_pass": 0,
            "winsor_worker_limit": 10,
            "max_raw_per_master_tile": 0,
            "use_gpu_phase5": False,
            "stack_use_gpu": False,
            "use_gpu_stack": False,
            "gpu_selector": "CPU (no GPU)",
            "gpu_id_phase5": 0,
            "logging_level": "INFO",
        }
        defaults.update(fallback_defaults)
        defaults.setdefault("language", "en")
        defaults.setdefault("qt_theme_mode", "system")
        return defaults

    def _determine_config_path(self) -> str | None:
        if zemosaic_config is None:
            return None
        get_path = getattr(zemosaic_config, "get_config_path", None)
        if callable(get_path):
            try:
                path = get_path()
            except Exception:
                return None
            return str(path)
        return None

    def _load_config(self) -> Dict[str, Any]:
        config_data: Dict[str, Any] = {}
        notes: List[Tuple[str, str, str, Dict[str, Any]]] = []
        config_path = self._config_path

        if zemosaic_config is not None and hasattr(zemosaic_config, "load_config"):
            load_func = getattr(zemosaic_config, "load_config")
            try:
                loaded_config = load_func()
            except Exception as exc:  # pragma: no cover - defensive guard
                notes.append(
                    (
                        "error",
                        "qt_log_config_load_failed",
                        "Failed to read configuration: {error}",
                        {"error": str(exc)},
                    )
                )
                loaded_config = {}
            else:
                if not isinstance(loaded_config, dict):
                    notes.append(
                        (
                            "warning",
                            "qt_log_config_invalid_type",
                            "Configuration file contained unexpected data. Using defaults.",
                            {},
                        )
                    )
                    loaded_config = {}
            config_data.update(loaded_config)
        else:  # pragma: no cover - minimal fallback
            notes.append(
                (
                    "warning",
                    "qt_log_config_fallback",
                    "Using in-memory defaults because zemosaic_config.load_config is unavailable.",
                    {},
                )
            )

        if config_path:
            if os.path.exists(config_path):
                notes.append(
                    (
                        "info",
                        "qt_log_config_loaded_from",
                        "Loaded configuration from {path}",
                        {"path": config_path},
                    )
                )
            else:
                notes.append(
                    (
                        "warning",
                        "qt_log_config_missing",
                        "No configuration file found at {path}; defaults will be used until saved.",
                        {"path": config_path},
                    )
                )

        merged_config = dict(self._default_config_values)
        merged_config.update(config_data)
        merged_config.setdefault("language", "en")

        canonical_gpu_pref: Any = merged_config.get("use_gpu_phase5")
        if canonical_gpu_pref in (None, ""):
            for legacy_key in ("stack_use_gpu", "use_gpu_stack"):
                if legacy_key in merged_config:
                    canonical_gpu_pref = merged_config[legacy_key]
                    break
        gpu_enabled = self._normalize_config_bool(canonical_gpu_pref, False)
        merged_config["use_gpu_phase5"] = gpu_enabled
        merged_config["stack_use_gpu"] = self._normalize_config_bool(
            merged_config.get("stack_use_gpu"), gpu_enabled
        )
        merged_config["use_gpu_stack"] = self._normalize_config_bool(
            merged_config.get("use_gpu_stack"), gpu_enabled
        )
        self._synchronize_gpu_config_keys(merged_config)
        self._disable_phase45_config(merged_config)

        loaded_snapshot = dict(config_data)
        self._synchronize_gpu_config_keys(loaded_snapshot)
        self._disable_phase45_config(loaded_snapshot)
        self._loaded_config_snapshot = loaded_snapshot
        self._persisted_config_keys = set(config_data.keys()) | set(
            self._default_config_values.keys()
        )

        self._config_load_notes = notes
        return merged_config

    def _emit_config_notes(self) -> None:
        if not self._config_load_notes:
            return
        try:
            for level, key, fallback, params in self._config_load_notes:
                message = self._tr(key, fallback)
                try:
                    formatted = message.format(**params)
                except Exception:
                    formatted = message
                self._append_log(formatted, level=level)
        finally:
            self._config_load_notes = []

    def _json_safe_config_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if hasattr(value, "__fspath__"):
            try:
                return os.fspath(value)
            except TypeError:
                pass
        if isinstance(value, dict):
            safe_dict: Dict[str, Any] = {}
            for key, nested_value in value.items():
                safe_key = str(key)
                safe_dict[safe_key] = self._json_safe_config_value(nested_value)
            return safe_dict
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe_config_value(item) for item in value]
        return str(value)

    def _serialize_config_for_save(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}

        self._synchronize_gpu_config_keys()

        keys_to_consider: set[str] = set(self._persisted_config_keys)
        keys_to_consider.update(self.config.keys())
        keys_to_consider.update(self._loaded_config_snapshot.keys())

        for key in keys_to_consider:
            if key in self.config:
                snapshot[key] = self._json_safe_config_value(self.config[key])
            elif key in self._loaded_config_snapshot:
                snapshot[key] = self._json_safe_config_value(
                    self._loaded_config_snapshot[key]
                )
        self._synchronize_gpu_config_keys(snapshot)
        # Phase 4.5 remains disabled for this release; never persist an enabled
        # state from the Qt GUI, even if an older config file contained True.
        self._disable_phase45_config(snapshot)
        return snapshot

    def _save_config(self) -> None:
        snapshot = self._serialize_config_for_save()
        if not snapshot:
            return

        saved = False
        last_error: str | None = None

        if zemosaic_config is not None and hasattr(zemosaic_config, "save_config"):
            try:
                result = zemosaic_config.save_config(snapshot)
            except Exception as exc:  # pragma: no cover - log instead of raising
                last_error = str(exc)
            else:
                saved = bool(result) or result is None

        if not saved and self._config_path:
            try:
                with open(self._config_path, "w", encoding="utf-8") as handle:
                    json.dump(snapshot, handle, indent=4, ensure_ascii=False)
            except Exception as exc:  # pragma: no cover - disk errors are rare
                last_error = str(exc)
            else:
                saved = True

        if saved:
            self.config.update(snapshot)
            normalized_snapshot = dict(snapshot)
            self._synchronize_gpu_config_keys(normalized_snapshot)
            self._loaded_config_snapshot = normalized_snapshot
            self._persisted_config_keys = set(snapshot.keys())
        else:
            if last_error is None and not self._config_path:
                last_error = self._tr(
                    "qt_log_config_path_missing",
                    "configuration path is not available",
                )
            template = self._tr(
                "qt_log_save_config_error",
                "Failed to save configuration: {error}",
            )
            self._append_log(template.format(error=last_error or "unknown"), level="error")

    def _create_localizer(self, language_code: str) -> Any:
        if ZeMosaicLocalization is not None:
            return ZeMosaicLocalization(language_code=language_code)
        return _FallbackLocalizer(language_code=language_code)

    def _tr(self, key: str, fallback: str) -> str:
        return self.localizer.get(key, fallback)

    def _initialize_log_level_prefixes(self) -> None:
        self._log_level_prefixes = {
            "debug": self._tr("qt_log_prefix_debug", "[DEBUG] "),
            "info": self._tr("qt_log_prefix_info", "[INFO] "),
            "success": self._tr("qt_log_prefix_success", "[SUCCESS] "),
            "warning": self._tr("qt_log_prefix_warning", "[WARNING] "),
            "error": self._tr("qt_log_prefix_error", "[ERROR] "),
        }

    def _refresh_translated_ui(self) -> None:
        self.setWindowTitle(
            self._tr("qt_window_title_preview", "ZeMosaic (Qt Preview)")
        )
        previous_log = ""
        if hasattr(self, "log_output"):
            try:
                previous_log = self.log_output.toPlainText()
            except Exception:
                previous_log = ""
        old_widget = self.takeCentralWidget()
        if old_widget is not None:
            old_widget.deleteLater()
        self._config_fields = {}
        self.language_combo = None
        self.backend_combo = None
        self._setup_ui()
        if previous_log and hasattr(self, "log_output"):
            try:
                self.log_output.setPlainText(previous_log)
            except Exception:
                pass
        self._initialize_log_level_prefixes()

    def _apply_language_selection(self, lang: str) -> None:
        if self.is_processing:
            return
        if hasattr(self.localizer, "set_language"):
            try:
                self.localizer.set_language(lang)
            except Exception:
                pass
        self.config["language"] = lang
        self._refresh_translated_ui()

    # ------------------------------------------------------------------
    # Events & callbacks
    # ------------------------------------------------------------------
    def _append_log(
        self,
        message: str,
        level: str = "info",
        *,
        gpu_highlight: bool = False,
    ) -> None:
        normalized_level = level.lower().strip()
        prefix = self._log_level_prefixes.get(normalized_level)
        if prefix is None and normalized_level:
            prefix = f"[{normalized_level.upper()}] "
        formatted_message = f"{prefix}{message}" if prefix else message
        if hasattr(self, "log_output"):
            if gpu_highlight:
                cursor = self.log_output.textCursor()
                cursor.movePosition(QTextCursor.End)
                self.log_output.setTextCursor(cursor)
                fmt = QTextCharFormat()
                fmt.setForeground(QColor("#40E0D0"))
                cursor.insertText(formatted_message + "\n", fmt)
                self.log_output.setTextCursor(cursor)
            else:
                self.log_output.appendPlainText(formatted_message)
        else:  # pragma: no cover - initialization guard
            print(formatted_message)

    def _clear_log(self) -> None:
        if hasattr(self, "log_output"):
            self.log_output.clear()

    # ------------------------------------------------------------------
    # GPU helper + ETA override helpers
    # ------------------------------------------------------------------
    def _is_gpu_eta_override_active(self) -> bool:
        return bool(self._gpu_eta_override)

    def _set_eta_display(self, text: str, *, force: bool = False) -> None:
        label = getattr(self, "eta_value_label", None)
        if label is None:
            return
        if not force and self._is_gpu_eta_override_active():
            return
        label.setText(text)

    def _format_eta_string(self, seconds: float, *, prefix: str = "") -> str:
        try:
            total = float(seconds)
        except Exception:
            total = 0.0
        prefix_value = prefix
        if total < 0:
            prefix_value = prefix_value or "+"
            total = abs(total)
        eta_h, eta_rem = divmod(int(total + 0.5), 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{prefix_value}{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"

    def _set_eta_label_from_seconds(self, seconds: float, *, prefix: str = "") -> None:
        try:
            formatted = self._format_eta_string(seconds, prefix=prefix)
        except Exception:
            formatted = self._tr("qt_progress_placeholder", "—")
        self._set_eta_display(formatted, force=True)

    def _start_gpu_eta_override(self, seconds: float, helper_name: str) -> None:
        try:
            predicted = float(seconds)
        except Exception:
            return
        if predicted <= 0:
            return
        self._gpu_eta_override = {
            "helper": helper_name,
            "predicted": predicted,
            "started": time.monotonic(),
        }
        if not self._gpu_eta_timer.isActive():
            self._gpu_eta_timer.start()
        self._tick_gpu_eta_override()

    def _tick_gpu_eta_override(self) -> None:
        state = self._gpu_eta_override
        if not state:
            if self._gpu_eta_timer.isActive():
                self._gpu_eta_timer.stop()
            return
        started = float(state.get("started", time.monotonic()))
        predicted = float(state.get("predicted", 0.0))
        elapsed = max(0.0, time.monotonic() - started)
        remaining = predicted - elapsed
        if remaining >= 0:
            self._set_eta_label_from_seconds(remaining)
        else:
            self._set_eta_label_from_seconds(abs(remaining), prefix="+")

    def _stop_gpu_eta_override(self) -> None:
        self._gpu_eta_override = None
        if self._gpu_eta_timer.isActive():
            self._gpu_eta_timer.stop()

    def _estimate_gpu_eta_seconds(
        self, helper_name: str, frames: int, channels: int, width: int, height: int
    ) -> Optional[float]:
        try:
            width_val = float(width)
            height_val = float(height)
        except Exception:
            return None
        if width_val <= 0 or height_val <= 0:
            return None
        grid_mp = max(1.0, (width_val * height_val) / 1_000_000.0)
        try:
            frames_val = int(frames or 0)
        except Exception:
            frames_val = 0
        frames_val = max(1, frames_val)
        try:
            channels_val = int(channels or 0)
        except Exception:
            channels_val = 0
        channels_val = max(1, channels_val)
        units = grid_mp * frames_val * max(1.0, channels_val / 3.0)
        if units <= 0:
            return None
        profile = self._gpu_eta_profiles.get(helper_name)
        rate: float | None = None
        if isinstance(profile, dict):
            try:
                rate = float(profile.get("avg_rate") or 0.0)
            except Exception:
                rate = None
        if not rate or rate <= 0:
            rate = float(self._gpu_eta_default_rate or 0.85)
        seconds = rate * units
        return max(10.0, seconds)

    def _record_gpu_eta_sample(
        self,
        helper_name: str,
        frames: int,
        channels: int,
        width: int,
        height: int,
        elapsed_seconds: float,
    ) -> None:
        try:
            elapsed = float(elapsed_seconds)
        except Exception:
            return
        if elapsed <= 0:
            return
        try:
            width_val = float(width)
            height_val = float(height)
        except Exception:
            return
        if width_val <= 0 or height_val <= 0:
            return
        grid_mp = max(1.0, (width_val * height_val) / 1_000_000.0)
        try:
            frames_val = int(frames or 0)
        except Exception:
            frames_val = 0
        frames_val = max(1, frames_val)
        try:
            channels_val = int(channels or 0)
        except Exception:
            channels_val = 0
        channels_val = max(1, channels_val)
        units = grid_mp * frames_val * max(1.0, channels_val / 3.0)
        if units <= 0:
            return
        sample_rate = elapsed / units
        profile = self._gpu_eta_profiles.setdefault(
            helper_name, {"avg_rate": sample_rate, "samples": 0}
        )
        samples = int(profile.get("samples", 0))
        existing_rate = profile.get("avg_rate")
        alpha = 0.35
        if existing_rate and samples > 0:
            new_rate = (1 - alpha) * float(existing_rate) + alpha * sample_rate
        else:
            new_rate = sample_rate
        profile["avg_rate"] = float(new_rate)
        profile["samples"] = min(50, samples + 1)
        profile["last_seconds"] = float(elapsed)
        profile["last_units"] = float(units)
        if zemosaic_config is not None and hasattr(zemosaic_config, "save_config"):
            try:
                self.config["gpu_eta_profiles"] = self._gpu_eta_profiles
                zemosaic_config.save_config(self.config)
            except Exception:
                pass

    def _handle_gpu_helper_start(self, payload: Dict[str, Any]) -> None:
        data = payload or {}
        helper_name = str(data.get("helper") or "gpu_reproject")
        frames_val = data.get("frames") or data.get("images") or 0
        channels_val = data.get("channels") or 0
        width_val = data.get("grid_w") or data.get("W") or 0
        height_val = data.get("grid_h") or data.get("H") or 0
        try:
            frames_int = int(frames_val)
        except Exception:
            frames_int = 0
        try:
            channels_int = int(channels_val)
        except Exception:
            channels_int = 0
        try:
            width_int = int(width_val)
        except Exception:
            width_int = 0
        try:
            height_int = int(height_val)
        except Exception:
            height_int = 0
        self._gpu_helper_active = {
            "helper": helper_name,
            "frames": frames_int,
            "channels": channels_int,
            "grid_w": width_int,
            "grid_h": height_int,
        }
        predicted = self._estimate_gpu_eta_seconds(
            helper_name, frames_int, channels_int, width_int, height_int
        )
        if predicted:
            self._start_gpu_eta_override(predicted, helper_name)

    def _handle_gpu_helper_finish(self, payload: Dict[str, Any]) -> None:
        data = payload or {}
        active = self._gpu_helper_active or {}
        helper_ref = data.get("helper") or active.get("helper")
        helper_name = str(helper_ref) if helper_ref else None
        frames_val = (
            data.get("images")
            or data.get("frames")
            or active.get("frames")
            or 0
        )
        channels_val = data.get("channels") or active.get("channels") or 0
        width_val = data.get("W") or data.get("grid_w") or active.get("grid_w") or 0
        height_val = data.get("H") or data.get("grid_h") or active.get("grid_h") or 0
        elapsed_val = data.get("elapsed_s")
        self._stop_gpu_eta_override()
        self._gpu_helper_active = None
        if helper_name and width_val and height_val and elapsed_val is not None:
            self._record_gpu_eta_sample(
                helper_name,
                frames_val,
                channels_val,
                width_val,
                height_val,
                elapsed_val,
            )

    def _handle_gpu_helper_abort(self) -> None:
        self._stop_gpu_eta_override()
        self._gpu_helper_active = None

    def _on_worker_gpu_helper_event(self, event: str, payload: Dict[str, Any]) -> None:
        event_type = (event or "").lower()
        data = payload if isinstance(payload, dict) else {}
        if event_type == "start":
            self._handle_gpu_helper_start(dict(data))
        elif event_type == "finish":
            self._handle_gpu_helper_finish(dict(data))
        elif event_type == "abort":
            self._handle_gpu_helper_abort()

    # ------------------------------------------------------------------
    # Phase 4.5 overlay helpers
    # ------------------------------------------------------------------
    def _phase45_log_translated(self, key: str, payload: Dict[str, Any], level: str) -> None:
        template = self._tr(key, key)
        try:
            message = template.format(**payload)
        except Exception:
            message = template
        self._append_log(message, level=level)

    def _phase45_reset_overlay(self, clear_groups: bool = True) -> None:
        if clear_groups:
            self._phase45_groups = {}
            self._phase45_last_out = None
        self._phase45_group_progress = {}
        self._phase45_active = None
        self._update_phase45_status()
        self._redraw_phase45_overlay()

    def _update_phase45_status(self, status_override: Optional[str] = None) -> None:
        label = self.phase45_status_label
        if label is None:
            return
        text = status_override
        if not text:
            total_groups = len(self._phase45_groups)
            if total_groups == 0:
                text = self._tr("phase45_status_idle", "Phase 4.5 idle")
            else:
                parts: List[str] = [f"Groups: {total_groups}"]
                if self._phase45_active is not None:
                    progress = self._phase45_group_progress.get(self._phase45_active)
                    if progress and progress.get("total"):
                        parts.append(
                            f"Active: G{self._phase45_active} ({progress.get('done', 0)}/{progress.get('total', 0)})"
                        )
                    else:
                        parts.append(f"Active: G{self._phase45_active}")
                if self._phase45_last_out:
                    parts.append(f"Super: {self._phase45_last_out}")
                text = " | ".join(parts)
        label.setText(text)

    def _phase45_show_message(self, message: str) -> None:
        scene = self.phase45_overlay_scene
        view = self.phase45_overlay_view
        if scene is None or view is None:
            return
        width = max(view.viewport().width(), 10)
        height = max(view.viewport().height(), 10)
        scene.setSceneRect(0, 0, width, height)
        scene.clear()
        text_item = scene.addText(message)
        text_item.setDefaultTextColor(QColor("#6f7177"))
        rect = text_item.boundingRect()
        text_item.setPos((width - rect.width()) / 2.0, (height - rect.height()) / 2.0)

    def _redraw_phase45_overlay(self) -> None:
        scene = self.phase45_overlay_scene
        view = self.phase45_overlay_view
        if scene is None or view is None:
            return
        width = max(view.viewport().width(), 10)
        height = max(view.viewport().height(), 10)
        scene.setSceneRect(0, 0, width, height)
        scene.clear()
        if not self._phase45_overlay_enabled:
            self._phase45_show_message(
                self._tr("phase45_overlay_hidden", "Overlay hidden")
            )
            return
        groups = self._phase45_groups or {}
        if not groups:
            self._phase45_show_message(
                self._tr("phase45_overlay_waiting", "Waiting for Phase 4.5...")
            )
            return
        bboxes = [entry.get("bbox") for entry in groups.values() if entry.get("bbox")]
        if not bboxes:
            self._phase45_show_message(
                self._tr("phase45_overlay_no_geo", "No WCS footprints")
            )
            return
        try:
            ra_min = min(float(entry["ra_min"]) for entry in bboxes)
            ra_max = max(float(entry["ra_max"]) for entry in bboxes)
            dec_min = min(float(entry["dec_min"]) for entry in bboxes)
            dec_max = max(float(entry["dec_max"]) for entry in bboxes)
        except Exception:
            self._phase45_show_message(
                self._tr("phase45_overlay_no_geo", "No WCS footprints")
            )
            return
        ra_span = max(ra_max - ra_min, 1e-6)
        dec_span = max(dec_max - dec_min, 1e-6)
        pad = 8.0
        drawable_w = max(width - (pad * 2.0), 1.0)
        drawable_h = max(height - (pad * 2.0), 1.0)
        for gid in sorted(groups):
            entry = groups[gid]
            bbox = entry.get("bbox")
            if not bbox:
                continue
            try:
                ra0 = float(bbox["ra_min"])
                ra1 = float(bbox["ra_max"])
                dec0 = float(bbox["dec_min"])
                dec1 = float(bbox["dec_max"])
            except Exception:
                continue
            x0 = pad + ((ra0 - ra_min) / ra_span) * drawable_w
            x1 = pad + ((ra1 - ra_min) / ra_span) * drawable_w
            y0 = pad + ((dec_max - dec1) / dec_span) * drawable_h
            y1 = pad + ((dec_max - dec0) / dec_span) * drawable_h
            rect_w = max(1.0, x1 - x0)
            rect_h = max(1.0, y1 - y0)
            is_active = gid == self._phase45_active
            pen = QPen(QColor("#E53935" if is_active else "#8E44AD"))
            pen.setWidth(2 if is_active else 1)
            if not is_active:
                pen.setStyle(Qt.DashLine)
            brush = QBrush(QColor("#FFCDD2")) if is_active else QBrush(Qt.NoBrush)
            rect_item = scene.addRect(x0, y0, rect_w, rect_h, pen, brush)
            if is_active:
                rect_item.setBrush(brush)
            label = f"G{gid}"
            progress = self._phase45_group_progress.get(gid)
            if is_active and progress and progress.get("total"):
                label = f"G{gid} {progress.get('done', 0)}/{progress.get('total', 0)}"
            text_item = scene.addText(label)
            text_item.setDefaultTextColor(QColor("#2E2E2E" if is_active else "#424242"))
            font = text_item.font()
            if is_active:
                font.setBold(True)
            else:
                font.setBold(False)
            text_item.setFont(font)
            text_item.setPos(x0 + 4.0, y0 + 4.0)

    def _phase45_handle_groups_layout(self, payload: Dict[str, Any], level: str) -> None:
        groups_payload = payload.get("groups") or []
        new_groups: Dict[int, Dict[str, Any]] = {}
        for entry in groups_payload:
            try:
                gid = int(entry.get("group_id"))
            except Exception:
                continue
            new_groups[gid] = {
                "bbox": entry.get("bbox"),
                "members": entry.get("members") or [],
                "repr": entry.get("repr"),
            }
        self._phase45_groups = new_groups
        self._phase45_group_progress = {}
        self._phase45_active = None
        self._phase45_last_out = None
        total = payload.get("total_groups", len(new_groups))
        self._update_phase45_status()
        self._redraw_phase45_overlay()
        self._append_log(
            f"[P4.5] Overlap layout ready ({len(new_groups)}/{total})",
            level=level,
        )

    def _phase45_handle_group_started(self, payload: Dict[str, Any], level: str) -> None:
        gid = payload.get("group_id")
        try:
            gid_int = int(gid)
        except Exception:
            gid_int = None
        chunk = payload.get("chunk")
        chunks = payload.get("chunks") or payload.get("total")
        size = payload.get("size")
        if gid_int is not None:
            try:
                total_chunks = int(chunks) if chunks is not None else 0
            except Exception:
                total_chunks = 0
            self._phase45_active = gid_int
            self._phase45_group_progress[gid_int] = {
                "done": int(payload.get("done", 0) or 0),
                "total": total_chunks,
                "size": size,
            }
        self._update_phase45_status()
        self._redraw_phase45_overlay()
        if gid_int is not None:
            chunk_txt = (
                f"{chunk}/{chunks}" if chunk and chunks else str(chunk) if chunk else "-"
            )
            size_txt = f"{size} tiles" if size else "tiles"
            self._append_log(
                f"[P4.5] Group G{gid_int} started chunk {chunk_txt} ({size_txt})",
                level=level,
            )

    def _phase45_handle_group_progress(self, payload: Dict[str, Any], level: str) -> None:
        gid = payload.get("group_id")
        try:
            gid_int = int(gid)
        except Exception:
            gid_int = None
        done = payload.get("done", 0)
        total = payload.get("total", 0)
        size = payload.get("size")
        try:
            done_int = int(done)
        except Exception:
            done_int = 0
        try:
            total_int = int(total)
        except Exception:
            total_int = 0
        if gid_int is not None:
            progress_entry = self._phase45_group_progress.setdefault(
                gid_int,
                {"done": 0, "total": total_int, "size": size},
            )
            progress_entry["done"] = done_int
            if total_int:
                progress_entry["total"] = total_int
            if size is not None:
                progress_entry["size"] = size
            self._phase45_group_progress[gid_int] = progress_entry
            self._phase45_active = gid_int
        self._update_phase45_status()
        self._redraw_phase45_overlay()
        if gid_int is not None and total_int:
            self._append_log(
                f"[P4.5] Group G{gid_int} progress {done_int}/{total_int}",
                level=level,
            )

    def _phase45_handle_group_result(self, payload: Dict[str, Any]) -> None:
        last_out = payload.get("out")
        if last_out:
            self._phase45_last_out = os.path.basename(str(last_out))
        gid = payload.get("group_id")
        try:
            gid_int = int(gid)
        except Exception:
            gid_int = None
        if gid_int is not None:
            self._phase45_active = gid_int
        self._update_phase45_status()
        self._redraw_phase45_overlay()

    def _phase45_handle_finish(self, payload: Dict[str, Any]) -> None:
        completion_text = self._tr(
            "phase45_status_complete",
            "Phase 4.5 complete",
        )
        if self._phase45_last_out:
            completion_text = f"{completion_text} | Super: {self._phase45_last_out}"
        self._phase45_active = None
        self._update_phase45_status(status_override=completion_text)
        self._redraw_phase45_overlay()

    def _on_phase45_overlay_toggled(self, checked: bool) -> None:
        self._phase45_overlay_enabled = bool(checked)
        self._redraw_phase45_overlay()

    def _reset_progress_tracking(self) -> None:
        self._stage_progress_values = {key: 0.0 for key in self._stage_order}
        self._stage_times.clear()
        self._progress_start_time = None
        self._last_global_progress = 0.0
        self._eta_seconds_smoothed = None
        self._weighted_progress_active = False

    def _set_processing_state(self, running: bool) -> None:
        self._stop_gpu_eta_override()
        self._gpu_helper_active = None
        if running:
            self._reset_progress_tracking()
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.filter_button.setEnabled(not running)
        if running:
            self.progress_bar.setValue(0)
            self.phase_value_label.setText(self._tr("qt_progress_placeholder", "Idle"))
            self._set_eta_display(self._tr("qt_progress_placeholder", "—"), force=True)
            self.elapsed_value_label.setText("00:00:00")
            self.files_value_label.setText("")
            self.tiles_value_label.setText(self._tr("qt_progress_count_placeholder", "0 / 0"))
        else:
            self.progress_bar.setValue(0)
            self.phase_value_label.setText(self._tr("qt_progress_placeholder", "Idle"))
            self._set_eta_display(self._tr("qt_progress_placeholder", "—"), force=True)
            self.elapsed_value_label.setText(self._tr("qt_progress_placeholder", "—"))
            self.files_value_label.setText("")
            self.tiles_value_label.setText(self._tr("qt_progress_count_placeholder", "0 / 0"))
            self._reset_progress_tracking()
        if self.language_combo is not None:
            self.language_combo.setEnabled(not running)

    def _normalize_log_level(self, level: str | None) -> str:
        if not level:
            return "info"
        normalized = level.strip().lower()
        mapping = {
            "warn": "warning",
            "warning": "warning",
            "error": "error",
            "err": "error",
            "success": "success",
            "debug": "debug",
        }
        return mapping.get(normalized, "info")

    def _is_gpu_log_entry(
        self, key_candidate: Any, log_text: str, params: Dict[str, Any] | None
    ) -> bool:
        def _contains_gpu(value: Any) -> bool:
            return isinstance(value, str) and "gpu" in value.lower()

        if _contains_gpu(key_candidate):
            return True
        if isinstance(log_text, str) and _contains_gpu(log_text):
            return True
        if isinstance(params, dict):
            for val in params.values():
                if _contains_gpu(val):
                    return True
        return False

    def _translate_worker_message(
        self,
        message_key_or_raw: Any,
        params: Dict[str, Any] | None,
        level: str | None,
    ) -> str:
        params = params or {}
        # Preserve existing behaviour for raw error strings.
        if isinstance(message_key_or_raw, str) and level and level.upper() == "ERROR":
            if not message_key_or_raw.startswith("log_key_") and " " in message_key_or_raw:
                return message_key_or_raw

        level_str = level.upper() if isinstance(level, str) else "INFO"
        user_facing_levels = {
            "INFO",
            "WARN",
            "ERROR",
            "SUCCESS",
            "CRITICAL",
            "INFO_DETAIL",
            "DEBUG_DETAIL",
        }

        if level_str in user_facing_levels:
            if isinstance(message_key_or_raw, str):
                text = self.localizer.get(
                    message_key_or_raw,
                    message_key_or_raw,
                    **params,
                )
                if text == f"_{message_key_or_raw}_" and not params:
                    text = message_key_or_raw
            else:
                text = str(message_key_or_raw)
        else:
            text = str(message_key_or_raw)
            if params:
                try:
                    text = text.format(**params)
                except Exception:
                    pass

        if isinstance(text, str) and (
            text.startswith("  [Z") or text.startswith("      [Z")
        ):
            try:
                text = text.split("] ", 1)[1]
            except Exception:
                pass
        return text

    def _on_worker_log_message(
        self, level: str, message_key_or_raw: Any, params: Dict[str, Any]
    ) -> None:
        level_str = str(level) if isinstance(level, str) else str(level)
        translated_message = self._translate_worker_message(
            message_key_or_raw, params, level_str
        )
        normalized_level = self._normalize_log_level(level_str)
        gpu_highlight = self._is_gpu_log_entry(
            message_key_or_raw, translated_message, params
        )
        self._append_log(translated_message, level=normalized_level, gpu_highlight=gpu_highlight)

    def _on_worker_progress_changed(self, percent: float) -> None:
        if self._weighted_progress_active:
            return
        bounded = int(max(0.0, min(100.0, percent)))
        self._last_global_progress = float(bounded)
        self.progress_bar.setValue(bounded)
        if bounded >= 100:
            self._set_eta_display("00:00:00")

    def _on_worker_stage_progress(self, stage: str, current: int, total: int) -> None:
        stage_label = self._format_stage_name(stage)
        self.phase_value_label.setText(stage_label)
        self._update_stage_progress(stage, current, total)

    def _update_stage_progress(self, stage: str, current: int, total: int) -> None:
        self._weighted_progress_active = True
        now = time.monotonic()
        stage_key = self._stage_aliases.get(stage, stage)
        if stage_key not in self._stage_progress_values:
            self._stage_progress_values[stage_key] = 0.0

        try:
            current_val = int(current)
        except (TypeError, ValueError):
            current_val = 0
        try:
            total_val = int(total)
        except (TypeError, ValueError):
            total_val = 0

        if total_val < 0:
            total_val = 0
        if total_val > 0:
            current_val = max(0, min(current_val, total_val))
        else:
            current_val = max(0, current_val)

        timings = self._stage_times.get(stage_key)
        if timings is None or current_val <= 1:
            timings = {
                "start": now,
                "last": now,
                "steps": [],
                "last_count": current_val,
            }
            self._stage_times[stage_key] = timings
        else:
            last_count = int(timings.get("last_count", 0))
            if current_val > last_count:
                delta = now - float(timings.get("last", now))
                if delta >= 0:
                    steps = timings.setdefault("steps", [])
                    steps.append(delta)
                    if len(steps) > 120:
                        del steps[: len(steps) - 120]
                timings["last"] = now
                timings["last_count"] = current_val
            else:
                timings["last"] = now
                timings["last_count"] = current_val

        if total_val > 0:
            self.tiles_value_label.setText(f"{current_val} / {total_val}")

        if self._progress_start_time is None:
            self._progress_start_time = now

        stage_weight = self._stage_weights.get(stage_key)
        if stage_weight is None:
            percent = (current_val / float(total_val) * 100.0) if total_val else 0.0
            percent = max(0.0, min(100.0, percent))
            percent = max(self._last_global_progress, percent)
            self._last_global_progress = percent
            self.progress_bar.setValue(int(percent))
            steps = timings.get("steps") or []
            if steps and total_val:
                avg = sum(steps) / len(steps)
                remaining_steps = max(0, total_val - current_val)
                remaining = max(0.0, remaining_steps * avg)
                if self._eta_seconds_smoothed is None:
                    smoothed_remaining = remaining
                else:
                    alpha = 0.3
                    smoothed_remaining = (
                        alpha * remaining
                        + (1 - alpha) * float(self._eta_seconds_smoothed)
                    )
                self._eta_seconds_smoothed = smoothed_remaining
                eta_h, eta_rem = divmod(int(smoothed_remaining + 0.5), 3600)
                eta_m, eta_s = divmod(eta_rem, 60)
                self._set_eta_display(f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}")
            return

        if stage_key in self._stage_order:
            stage_index = self._stage_order.index(stage_key)
            for previous_stage in self._stage_order[:stage_index]:
                if self._stage_progress_values.get(previous_stage, 0.0) < 1.0:
                    self._stage_progress_values[previous_stage] = 1.0

        stage_fraction = (current_val / float(total_val)) if total_val else 0.0
        stage_fraction = max(0.0, min(1.0, stage_fraction))
        self._stage_progress_values[stage_key] = stage_fraction

        global_progress = 0.0
        for key in self._stage_order:
            weight = self._stage_weights.get(key, 0.0)
            if weight <= 0.0:
                continue
            fraction = self._stage_progress_values.get(key, 0.0)
            if fraction <= 0.0:
                continue
            if fraction > 1.0:
                fraction = 1.0
            global_progress += weight * fraction

        global_progress = max(self._last_global_progress, min(100.0, global_progress))
        self._last_global_progress = global_progress
        self.progress_bar.setValue(int(global_progress))

        if global_progress >= 99.9:
            self._eta_seconds_smoothed = 0.0
            self._set_eta_display("00:00:00")
            return

        if self._progress_start_time is None:
            return
        elapsed = now - self._progress_start_time
        if global_progress > 0.0 and elapsed >= 0.0:
            fraction_complete = max(1e-6, global_progress / 100.0)
            estimated_total = elapsed / fraction_complete
            eta_seconds = max(0.0, estimated_total - elapsed)
            if self._eta_seconds_smoothed is None:
                smoothed = eta_seconds
            else:
                alpha = 0.3
                smoothed = (alpha * eta_seconds) + (
                    (1 - alpha) * float(self._eta_seconds_smoothed)
                )
            self._eta_seconds_smoothed = smoothed
            eta_h, eta_rem = divmod(int(smoothed + 0.5), 3600)
            eta_m, eta_s = divmod(eta_rem, 60)
            self._set_eta_display(f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}")

    def _on_worker_phase45_event(self, key: str, payload: Dict[str, Any], level: str) -> None:
        data = payload if isinstance(payload, dict) else {}
        normalized_level = self._normalize_log_level(level)
        if key == "p45_start":
            self._phase45_reset_overlay()
            self._phase45_log_translated(key, data, normalized_level)
            return
        if key == "p45_groups_layout":
            self._phase45_handle_groups_layout(data, normalized_level)
            return
        if key == "p45_group_started":
            self._phase45_handle_group_started(data, normalized_level)
            return
        if key == "p45_group_progress":
            self._phase45_handle_group_progress(data, normalized_level)
            return
        if key == "p45_group_result":
            self._phase45_handle_group_result(data)
            return
        if key == "p45_finished":
            self._phase45_handle_finish(data)
            self._phase45_log_translated(key, data, normalized_level)
            return
        self._phase45_log_translated(key, data, normalized_level)

    def _on_worker_phase_changed(self, stage: str, payload: Dict[str, Any]) -> None:
        stage_label = self._format_stage_name(stage)
        self.phase_value_label.setText(stage_label)
        current = payload.get("current") if isinstance(payload, dict) else None
        total = payload.get("total") if isinstance(payload, dict) else None
        if isinstance(current, int) and isinstance(total, int) and total > 0:
            self.tiles_value_label.setText(f"{current} / {total}")

    def _on_worker_stats_updated(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        files_done = payload.get("files_done")
        files_total = payload.get("files_total")
        if isinstance(files_done, int) and isinstance(files_total, int) and files_total >= 0:
            done = max(0, files_done)
            remaining = max(0, files_total - done)
            self.files_value_label.setText(str(remaining))
        tiles_done = payload.get("tiles_done")
        tiles_total = payload.get("tiles_total")
        if isinstance(tiles_done, int) and isinstance(tiles_total, int) and tiles_total >= 0:
            self.tiles_value_label.setText(f"{tiles_done} / {tiles_total}")
        eta_seconds = payload.get("eta_seconds")
        if isinstance(eta_seconds, (int, float)) and eta_seconds >= 0:
            eta_h, eta_rem = divmod(int(eta_seconds + 0.5), 3600)
            eta_m, eta_s = divmod(eta_rem, 60)
            self._set_eta_display(f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}")

    def _on_worker_eta_updated(self, eta_text: str) -> None:
        if not isinstance(eta_text, str):
            return
        text = eta_text.strip()
        if not text:
            placeholder = self._tr("qt_progress_placeholder", "—")
            self._set_eta_display(placeholder, force=True)
            return
        self._set_eta_display(text)

    def _on_worker_chrono_control(self, action: str) -> None:
        if not isinstance(action, str):
            return
        normalized = action.strip().lower()
        if normalized == "start":
            self._run_started_monotonic = time.monotonic()
            self.elapsed_value_label.setText(
                self._tr("initial_elapsed_time", "00:00:00")
            )
            if not self._elapsed_timer.isActive():
                self._elapsed_timer.start()
        elif normalized == "stop":
            self._elapsed_timer.stop()

    def _on_worker_raw_file_count_updated(self, current: int, total: int) -> None:
        try:
            cur = int(current)
        except Exception:
            cur = 0
        try:
            tot = int(total)
        except Exception:
            tot = 0
        if tot < 0:
            tot = 0
        if cur < 0:
            cur = 0
        if tot > 0 and cur > tot:
            cur = tot
        remaining = max(0, tot - cur)
        self.files_value_label.setText(str(remaining))

    def _on_worker_master_tile_count_updated(self, current: int, total: int) -> None:
        try:
            cur = int(current)
        except Exception:
            cur = 0
        try:
            tot = int(total)
        except Exception:
            tot = 0
        if tot < 0:
            tot = 0
        if tot > 0:
            cur = max(0, min(cur, tot))
        else:
            cur = max(0, cur)
        self.tiles_value_label.setText(f"{cur} / {tot}")

    def _on_worker_cluster_override(self, overrides: Dict[str, Any]) -> None:
        if not isinstance(overrides, dict):
            return
        self._apply_filter_overrides_to_config(overrides)

    def _on_worker_finished(self, success: bool, message: str) -> None:
        self.is_processing = False
        self._elapsed_timer.stop()
        self._run_started_monotonic = None
        self._set_processing_state(False)
        self.stop_button.setEnabled(False)

        # Distinguish clean completion from user cancellation and errors.
        is_cancel = (not success) and (
            not message
            or message == "qt_log_processing_cancelled"
            or message == "log_key_processing_cancelled"
        )

        if is_cancel:
            # Mirror Tk semantics: treat user-triggered stops as a warning-style
            # cancellation using the shared log key, reset the progress panel,
            # and avoid hard-error dialogs.
            cancel_key = "log_key_processing_cancelled"
            translated = self._translate_worker_message(cancel_key, {}, "WARN")
            self._append_log(translated, level="warning")
            return

        if success:
            completion_message = self._tr(
                "qt_log_processing_completed", "Processing completed successfully."
            )
            self._append_log(completion_message, level="success")
            if self.isVisible():
                QMessageBox.information(
                    self,
                    self._tr("qt_processing_finished_title", "Processing complete"),
                    completion_message,
                )
            # Optional post-run prompt to open the output folder, mirroring Tk.
            output_dir = str(self.config.get("output_dir", "") or "").strip()
            if (
                output_dir
                and os.path.isdir(output_dir)
                and self.isVisible()
            ):
                title = self.localizer.get(
                    "q_open_output_folder_title",
                    "Open Output Folder?",
                )
                prompt = self.localizer.get(
                    "q_open_output_folder_msg",
                    "Do you want to open the output folder '{folder}'?",
                    folder=output_dir,
                )
                answer = QMessageBox.question(
                    self,
                    title,
                    prompt,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if answer == QMessageBox.Yes:
                    try:
                        if os.name == "nt":
                            os.startfile(output_dir)  # type: ignore[arg-type]
                        elif sys.platform == "darwin":
                            subprocess.Popen(["open", output_dir])
                        else:
                            subprocess.Popen(["xdg-open", output_dir])
                    except Exception as exc:
                        log_msg = self.localizer.get(
                            "log_key_error_opening_folder",
                            "Error opening output folder: {error}",
                            error=str(exc),
                        )
                        self._append_log(log_msg, level="error")
                        error_title = self.localizer.get(
                            "error_title",
                            "Error",
                        )
                        error_body = self.localizer.get(
                            "error_cannot_open_folder",
                            "Could not open output folder:\n{error}",
                            error=str(exc),
                        )
                        QMessageBox.critical(self, error_title, error_body)
            return

        # Non-success, non-cancellation path: treat as an error reported by the worker.
        message_key = message or "qt_worker_error_generic"
        translated = self._translate_worker_message(message_key, {}, "ERROR")
        self._append_log(translated, level="error")
        if self.isVisible():
            QMessageBox.warning(
                self,
                self._tr("qt_processing_stopped_title", "Processing stopped"),
                translated,
            )

    def _on_elapsed_timer_tick(self) -> None:
        if not self.is_processing or self._run_started_monotonic is None:
            self._elapsed_timer.stop()
            return
        elapsed = max(0.0, time.monotonic() - self._run_started_monotonic)
        elapsed_h, remainder = divmod(int(elapsed + 0.5), 3600)
        elapsed_m, elapsed_s = divmod(remainder, 60)
        self.elapsed_value_label.setText(f"{elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}")

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._redraw_phase45_overlay()

    def _format_stage_name(self, stage: str) -> str:
        if not stage:
            return self._tr("qt_progress_placeholder", "Idle")
        key = f"qt_stage_{stage.lower()}"
        fallback = stage.replace("_", " ").strip().title()
        return self._tr(key, fallback)

    def _build_worker_invocation(
        self, *, skip_filter_ui: bool
    ) -> tuple[Tuple[Any, ...], Dict[str, Any]]:
        cfg = self.config
        self._disable_phase45_config(cfg)

        def _coerce_str(value: Any, default: str = "") -> str:
            if value is None:
                return default
            return str(value)

        def _coerce_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _coerce_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _coerce_bool(value: Any, default: bool = False) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value != 0
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "on"}:
                    return True
                if normalized in {"0", "false", "no", "off"}:
                    return False
                if not normalized:
                    return default
            return bool(value) if value is not None else default

        def _coerce_pair(value: Any, default: Sequence[float]) -> List[float]:
            if isinstance(value, str):
                parts = [segment.strip() for segment in value.split(",") if segment.strip()]
                if len(parts) >= 2:
                    try:
                        return [float(parts[0]), float(parts[1])]
                    except Exception:
                        return list(default)
            if isinstance(value, Iterable):
                items = list(value)
                if len(items) >= 2:
                    try:
                        return [float(items[0]), float(items[1])]
                    except Exception:
                        return list(default)
            return list(default)

        input_dir = _coerce_str(cfg.get("input_dir", "")).strip()
        output_dir = _coerce_str(cfg.get("output_dir", "")).strip()
        if not input_dir or not output_dir:
            raise ValueError(
                self._tr(
                    "qt_error_missing_paths",
                    "Input and output folders must be configured before starting.",
                )
            )

        astap_exe = _coerce_str(cfg.get("astap_executable_path", ""))
        astap_data = _coerce_str(cfg.get("astap_data_directory_path", ""))
        astap_radius = _coerce_float(cfg.get("astap_default_search_radius", 3.0), 3.0)
        astap_downsample = _coerce_int(cfg.get("astap_default_downsample", 2), 2)
        astap_sensitivity = _coerce_int(cfg.get("astap_default_sensitivity", 100), 100)

        cluster_threshold = _coerce_float(cfg.get("cluster_panel_threshold", 0.05), 0.05)
        cluster_target_groups = _coerce_int(cfg.get("cluster_target_groups", 0), 0)
        cluster_orientation = _coerce_float(cfg.get("cluster_orientation_split_deg", 0.0), 0.0)

        stack_ram_budget = _coerce_float(cfg.get("stack_ram_budget_gb", 0.0), 0.0)
        stack_norm_method = _coerce_str(cfg.get("stacking_normalize_method", "linear_fit"))
        stack_weight_method = _coerce_str(cfg.get("stacking_weighting_method", "noise_variance"))
        stack_reject_algo = _coerce_str(cfg.get("stacking_rejection_algorithm", "winsorized_sigma_clip"))
        stack_kappa_low = _coerce_float(cfg.get("stacking_kappa_low", 3.0), 3.0)
        stack_kappa_high = _coerce_float(cfg.get("stacking_kappa_high", 3.0), 3.0)
        winsor_limits = _coerce_pair(cfg.get("stacking_winsor_limits", (0.05, 0.05)), (0.05, 0.05))
        stack_final_combine = _coerce_str(cfg.get("stacking_final_combine_method", "mean"))
        poststack_equalize_rgb = _coerce_bool(cfg.get("poststack_equalize_rgb", True), True)
        apply_radial_weight = _coerce_bool(cfg.get("apply_radial_weight", False), False)
        radial_feather_fraction = _coerce_float(cfg.get("radial_feather_fraction", 0.8), 0.8)
        radial_shape_power = _coerce_float(cfg.get("radial_shape_power", 2.0), 2.0)
        min_radial_weight_floor = _coerce_float(cfg.get("min_radial_weight_floor", 0.0), 0.0)
        final_assembly_method = _coerce_str(cfg.get("final_assembly_method", "reproject_coadd"))

        inter_master_merge = _coerce_bool(cfg.get("inter_master_merge_enable", False), False)
        overlap_fraction = _coerce_float(cfg.get("inter_master_overlap_threshold", 0.60), 0.60)
        overlap_fraction = max(0.0, min(1.0, overlap_fraction))
        inter_master_min_group = max(2, _coerce_int(cfg.get("inter_master_min_group_size", 2), 2))
        inter_master_method = _coerce_str(cfg.get("inter_master_stack_method", "winsor"))
        inter_master_memmap_policy = _coerce_str(cfg.get("inter_master_memmap_policy", "auto"))
        inter_master_local_scale = _coerce_str(cfg.get("inter_master_local_scale", "final"))
        inter_master_max_group = max(
            inter_master_min_group,
            _coerce_int(cfg.get("inter_master_max_group", 64), 64),
        )
        num_base_workers = _coerce_int(cfg.get("num_processing_workers", 0), 0)
        if num_base_workers < 0:
            num_base_workers = 0

        apply_master_tile_crop = _coerce_bool(cfg.get("apply_master_tile_crop", True), True)
        master_tile_crop_percent = _coerce_float(cfg.get("master_tile_crop_percent", 3.0), 3.0)
        quality_crop_enabled = _coerce_bool(cfg.get("quality_crop_enabled", False), False)
        quality_crop_band = _coerce_int(cfg.get("quality_crop_band_px", 32), 32)
        quality_crop_k_sigma = _coerce_float(cfg.get("quality_crop_k_sigma", 2.0), 2.0)
        quality_crop_margin = _coerce_int(cfg.get("quality_crop_margin_px", 8), 8)
        quality_crop_min_run = _coerce_int(cfg.get("quality_crop_min_run", 2), 2)
        altaz_cleanup_enabled = _coerce_bool(cfg.get("altaz_cleanup_enabled", False), False)
        altaz_margin_percent = _coerce_float(cfg.get("altaz_margin_percent", 5.0), 5.0)
        altaz_decay = _coerce_float(cfg.get("altaz_decay", 0.15), 0.15)
        altaz_nanize = _coerce_bool(cfg.get("altaz_nanize", True), True)
        quality_gate_enabled = _coerce_bool(cfg.get("quality_gate_enabled", False), False)
        quality_gate_threshold = _coerce_float(cfg.get("quality_gate_threshold", 0.48), 0.48)
        quality_gate_edge_band = _coerce_int(cfg.get("quality_gate_edge_band_px", 64), 64)
        quality_gate_k_sigma = _coerce_float(cfg.get("quality_gate_k_sigma", 2.5), 2.5)
        quality_gate_erode_px = _coerce_int(cfg.get("quality_gate_erode_px", 3), 3)
        quality_gate_move_rejects = _coerce_bool(cfg.get("quality_gate_move_rejects", True), True)

        save_final_uint16 = _coerce_bool(cfg.get("save_final_as_uint16", False), False)
        legacy_rgb_cube = _coerce_bool(cfg.get("legacy_rgb_cube", False), False)
        use_memmap = _coerce_bool(cfg.get("coadd_use_memmap", True), True)
        memmap_dir = _coerce_str(cfg.get("coadd_memmap_dir", ""))
        cleanup_memmap = _coerce_bool(cfg.get("coadd_cleanup_memmap", True), True)
        assembly_workers = _coerce_int(cfg.get("assembly_process_workers", 0), 0)
        auto_limit_frames = _coerce_bool(cfg.get("auto_limit_frames_per_master_tile", True), True)
        winsor_max_frames = _coerce_int(cfg.get("winsor_max_frames_per_pass", 0), 0)
        winsor_worker_limit = _coerce_int(cfg.get("winsor_worker_limit", 10), 10)
        max_raw_per_tile = _coerce_int(cfg.get("max_raw_per_master_tile", 0), 0)

        intertile_match = _coerce_bool(cfg.get("intertile_photometric_match", True), True)
        intertile_preview_size = _coerce_int(cfg.get("intertile_preview_size", 512), 512)
        intertile_overlap_min = _coerce_float(cfg.get("intertile_overlap_min", 0.05), 0.05)
        intertile_sky_percentile = _coerce_pair(cfg.get("intertile_sky_percentile", (30.0, 70.0)), (30.0, 70.0))
        intertile_clip_sigma = _coerce_float(cfg.get("intertile_robust_clip_sigma", 2.5), 2.5)
        intertile_global_recenter = _coerce_bool(cfg.get("intertile_global_recenter", True), True)
        intertile_recenter_clip = _coerce_pair(cfg.get("intertile_recenter_clip", (0.85, 1.18)), (0.85, 1.18))
        use_auto_intertile = _coerce_bool(cfg.get("use_auto_intertile", False), False)
        match_background_for_final = _coerce_bool(cfg.get("match_background_for_final", True), True)
        incremental_feather_parity = _coerce_bool(cfg.get("incremental_feather_parity", False), False)

        two_pass_cov = _coerce_bool(cfg.get("two_pass_coverage_renorm", False), False)
        two_pass_sigma = _coerce_int(cfg.get("two_pass_cov_sigma_px", 50), 50)
        two_pass_gain_clip = _coerce_pair(cfg.get("two_pass_cov_gain_clip", (0.85, 1.18)), (0.85, 1.18))

        center_out_normalization = _coerce_bool(cfg.get("center_out_normalization_p3", True), True)
        p3_center_sky = _coerce_pair(cfg.get("p3_center_sky_percentile", (25.0, 60.0)), (25.0, 60.0))
        p3_center_clip_sigma = _coerce_float(cfg.get("p3_center_robust_clip_sigma", 2.5), 2.5)
        p3_center_preview_size = _coerce_int(cfg.get("p3_center_preview_size", 256), 256)
        p3_center_overlap_fraction = _coerce_float(cfg.get("p3_center_min_overlap_fraction", 0.03), 0.03)
        center_anchor_mode = _coerce_str(cfg.get("center_out_anchor_mode", "auto_central_quality"))
        anchor_probe_limit = _coerce_int(cfg.get("anchor_quality_probe_limit", 12), 12)
        anchor_span_range = _coerce_pair(cfg.get("anchor_quality_span_range", (0.02, 6.0)), (0.02, 6.0))
        anchor_median_clip_sigma = _coerce_float(cfg.get("anchor_quality_median_clip_sigma", 2.5), 2.5)

        poststack_review = _coerce_bool(cfg.get("enable_poststack_anchor_review", True), True)
        poststack_probe_limit = _coerce_int(cfg.get("poststack_anchor_probe_limit", 8), 8)
        poststack_span_range = _coerce_pair(cfg.get("poststack_anchor_span_range", (0.004, 10.0)), (0.004, 10.0))
        poststack_median_clip_sigma = _coerce_float(cfg.get("poststack_anchor_median_clip_sigma", 3.5), 3.5)
        poststack_min_improvement = _coerce_float(cfg.get("poststack_anchor_min_improvement", 0.12), 0.12)
        poststack_min_improvement = max(0.0, min(1.0, poststack_min_improvement))
        poststack_use_overlap = _coerce_bool(cfg.get("poststack_anchor_use_overlap_affine", True), True)

        use_gpu_phase5 = _coerce_bool(cfg.get("use_gpu_phase5", False), False)
        gpu_id_raw = cfg.get("gpu_id_phase5")
        gpu_id = None
        if gpu_id_raw not in (None, ""):
            try:
                gpu_id = int(gpu_id_raw)
            except (TypeError, ValueError):
                gpu_id = None

        logging_level = _coerce_str(cfg.get("logging_level", "INFO"))

        solver_settings_dict = self._build_solver_settings_dict()

        worker_args: Tuple[Any, ...] = (
            input_dir,
            output_dir,
            astap_exe,
            astap_data,
            astap_radius,
            astap_downsample,
            astap_sensitivity,
            cluster_threshold,
            cluster_target_groups,
            cluster_orientation,
            stack_ram_budget,
            stack_norm_method,
            stack_weight_method,
            stack_reject_algo,
            stack_kappa_low,
            stack_kappa_high,
            tuple(winsor_limits),
            stack_final_combine,
            poststack_equalize_rgb,
            apply_radial_weight,
            radial_feather_fraction,
            radial_shape_power,
            min_radial_weight_floor,
            final_assembly_method,
            inter_master_merge,
            overlap_fraction,
            inter_master_min_group,
            inter_master_method,
            inter_master_memmap_policy,
            inter_master_local_scale,
            inter_master_max_group,
            num_base_workers,
            apply_master_tile_crop,
            master_tile_crop_percent,
            quality_crop_enabled,
            quality_crop_band,
            quality_crop_k_sigma,
            quality_crop_margin,
            quality_crop_min_run,
            altaz_cleanup_enabled,
            altaz_margin_percent,
            altaz_decay,
            altaz_nanize,
            quality_gate_enabled,
            quality_gate_threshold,
            quality_gate_edge_band,
            quality_gate_k_sigma,
            quality_gate_erode_px,
            quality_gate_move_rejects,
            save_final_uint16,
            legacy_rgb_cube,
            use_memmap,
            memmap_dir,
            cleanup_memmap,
            assembly_workers,
            auto_limit_frames,
            winsor_max_frames,
            winsor_worker_limit,
            max_raw_per_tile,
            intertile_match,
            intertile_preview_size,
            intertile_overlap_min,
            intertile_sky_percentile,
            intertile_clip_sigma,
            intertile_global_recenter,
            intertile_recenter_clip,
            use_auto_intertile,
            match_background_for_final,
            incremental_feather_parity,
            two_pass_cov,
            two_pass_sigma,
            two_pass_gain_clip,
            center_out_normalization,
            p3_center_sky,
            p3_center_clip_sigma,
            p3_center_preview_size,
            p3_center_overlap_fraction,
            center_anchor_mode,
            anchor_probe_limit,
            anchor_span_range,
            anchor_median_clip_sigma,
            poststack_review,
            poststack_probe_limit,
            poststack_span_range,
            poststack_median_clip_sigma,
            poststack_min_improvement,
            poststack_use_overlap,
            use_gpu_phase5,
            gpu_id,
            logging_level,
        )

        worker_kwargs: Dict[str, Any] = {"solver_settings_dict": solver_settings_dict}
        if skip_filter_ui:
            worker_kwargs["skip_filter_ui"] = True
            if self._last_filter_overrides is not None or self._last_filtered_header_items is not None:
                worker_kwargs["filter_invoked"] = True
            if isinstance(self._last_filter_overrides, dict):
                worker_kwargs["filter_overrides"] = self._last_filter_overrides
            if isinstance(self._last_filtered_header_items, list):
                worker_kwargs["filtered_header_items"] = self._last_filtered_header_items
            worker_kwargs["early_filter_enabled"] = False
        return worker_args, worker_kwargs

    def _build_solver_settings_dict(self) -> Dict[str, Any]:
        astap_exe = str(self.config.get("astap_executable_path", "") or "")
        astap_data = str(self.config.get("astap_data_directory_path", "") or "")
        search_radius = float(self.config.get("astap_default_search_radius", 3.0) or 3.0)
        astap_downsample = int(self.config.get("astap_default_downsample", 2) or 2)
        astap_sensitivity = int(self.config.get("astap_default_sensitivity", 100) or 100)
        astap_max_instances = self._resolve_astap_max_instances()
        solver_choice = str(self.config.get("solver_method", "ASTAP") or "ASTAP")
        api_key = str(self.config.get("astrometry_api_key", "") or "")

        if SolverSettings is None:
            return {
                "solver_choice": solver_choice,
                "api_key": api_key,
                "timeout": int(self.config.get("astrometry_timeout", 60) or 60),
                "downsample": int(self.config.get("astrometry_downsample", astap_downsample) or astap_downsample),
                "force_lum": False,
                "astap_executable_path": astap_exe,
                "astap_data_directory_path": astap_data,
                "astap_search_radius_deg": search_radius,
                "astap_downsample": astap_downsample,
                "astap_sensitivity": astap_sensitivity,
                "astap_max_instances": astap_max_instances,
            }

        try:
            settings = SolverSettings.load_default()
        except Exception:
            settings = SolverSettings()

        settings.solver_choice = solver_choice or settings.solver_choice
        settings.api_key = api_key or settings.api_key
        timeout_value = self.config.get("astrometry_timeout")
        if timeout_value is not None:
            try:
                settings.timeout = int(timeout_value)
            except (TypeError, ValueError):
                pass
        downsample_value = self.config.get("astrometry_downsample")
        if downsample_value is not None:
            try:
                settings.downsample = int(downsample_value)
            except (TypeError, ValueError):
                pass
        settings.astap_executable_path = astap_exe or settings.astap_executable_path
        settings.astap_data_directory_path = astap_data or settings.astap_data_directory_path
        settings.astap_search_radius_deg = search_radius
        settings.astap_downsample = astap_downsample
        settings.astap_sensitivity = astap_sensitivity
        payload = asdict(settings)
        payload["astap_max_instances"] = astap_max_instances
        return payload

    def _on_start_clicked(self) -> None:
        self._start_processing(skip_filter_prompt=False)

    def _on_stop_clicked(self) -> None:
        if not self.is_processing:
            self._append_log(
                self._tr("qt_log_stop_ignored", "No processing is currently running."),
                level="warning",
            )
            return

        self._append_log(
            self._tr("qt_log_stop_requested", "Stop requested."),
            level="warning",
        )
        try:
            self.worker_controller.stop()
        except Exception as exc:  # pragma: no cover - defensive
            template = self._tr(
                "qt_log_stop_failure", "Failed to stop worker cleanly: {error}"
            )
            self._append_log(template.format(error=exc), level="error")
        self.stop_button.setEnabled(False)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self._collect_config_from_widgets()
        self._apply_astap_concurrency_setting()
        self._save_config()
        if self.is_processing:
            try:
                self.worker_controller.stop()
            except Exception:
                pass
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Filter integration helpers
    # ------------------------------------------------------------------
    def _start_processing(
        self,
        *,
        skip_filter_prompt: bool,
        predecided_skip_filter_ui: bool | None = None,
    ) -> None:
        if self.is_processing:
            QMessageBox.warning(
                self,
                self._tr("qt_warning_already_running_title", "Processing in progress"),
                self._tr(
                    "qt_warning_already_running_message",
                    "A processing run is already active.",
                ),
            )
            return

        if run_hierarchical_mosaic_process is None:
            QMessageBox.critical(
                self,
                self._tr("qt_error_worker_missing_title", "Worker unavailable"),
                self._tr(
                    "qt_error_worker_missing_message",
                    "The processing backend is not available. Ensure zemosaic_worker is installed.",
                ),
            )
            return

        self._collect_config_from_widgets()

        skip_filter_ui_for_run = bool(predecided_skip_filter_ui) if predecided_skip_filter_ui is not None else False

        if not skip_filter_prompt and predecided_skip_filter_ui is None:
            answer = QMessageBox.question(
                self,
                self._tr("qt_filter_prompt_title", "Filter range and set clustering?"),
                self._tr(
                    "qt_filter_prompt_message",
                    "Do you want to open the filter window to adjust the range and clustering before processing?",
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.No:
                skip_filter_ui_for_run = True
                self._clear_filter_results()
            else:
                filter_result = self._launch_filter_dialog()
                if filter_result is True:
                    skip_filter_ui_for_run = True
                    self._collect_config_from_widgets()
                elif filter_result is False:
                    return
                else:
                    skip_filter_ui_for_run = True

        self._save_config()

        try:
            worker_args, worker_kwargs = self._build_worker_invocation(skip_filter_ui=skip_filter_ui_for_run)
        except ValueError as exc:
            QMessageBox.warning(self, self._tr("qt_error_invalid_config_title", "Invalid configuration"), str(exc))
            return
        except Exception as exc:  # pragma: no cover - unexpected preparation failure
            message = self._tr("qt_error_prepare_worker", "Unable to prepare worker arguments.")
            self._append_log(f"{message} {exc}", level="error")
            QMessageBox.critical(self, self._tr("qt_error_prepare_worker_title", "Worker preparation failed"), str(exc))
            return

        self._begin_async_worker_start(worker_args, worker_kwargs)

    def _begin_async_worker_start(
        self,
        worker_args: Sequence[Any],
        worker_kwargs: Dict[str, Any],
    ) -> None:
        if self._worker_start_thread is not None:
            return
        self._worker_start_result = None
        self.start_button.setEnabled(False)
        self.filter_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self._append_log(
            self._tr("qt_log_start_worker_pending", "Spawning worker process…"),
            level="info",
        )

        def _runner() -> None:
            try:
                spawn_result = self.worker_controller.spawn_worker_process(worker_args, worker_kwargs)
            except Exception as exc:  # pragma: no cover - start failures are rare
                self._worker_start_result = (False, str(exc))
            else:
                if spawn_result is None:
                    self._worker_start_result = (False, None)
                else:
                    self._worker_start_result = (True, spawn_result)

        thread = threading.Thread(target=_runner, daemon=True)
        self._worker_start_thread = thread
        thread.start()
        QTimer.singleShot(50, self._poll_worker_start_result)

    def _poll_worker_start_result(self) -> None:
        thread = self._worker_start_thread
        if thread is None:
            return
        if thread.is_alive():
            QTimer.singleShot(100, self._poll_worker_start_result)
            return
        self._worker_start_thread = None
        result = self._worker_start_result or (False, self._tr("qt_error_start_worker_generic", "Failed to start worker process."))
        self._worker_start_result = None
        started, payload = result
        if started:
            queue_obj, process = payload  # type: ignore[misc]
            try:
                self.worker_controller.finalize_spawn(queue_obj, process)
            except Exception as exc:
                self._handle_worker_start_failure(str(exc))
                return
            self._finalize_successful_worker_start()
        else:
            error_message = payload
            self._handle_worker_start_failure(error_message if isinstance(error_message, str) else None)

    def _finalize_successful_worker_start(self) -> None:
        self.is_processing = True
        self._run_started_monotonic = time.monotonic()
        self._elapsed_timer.start()
        self._set_processing_state(True)
        self._append_log(
            self._tr("qt_log_processing_started", "Processing started."),
            level="info",
        )

    def _handle_worker_start_failure(self, error_message: str | None) -> None:
        self.start_button.setEnabled(True)
        self.filter_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if error_message:
            log_template = self._tr(
                "qt_log_start_worker_failure", "Failed to start worker: {error}"
            )
            self._append_log(log_template.format(error=error_message), level="error")
            message_template = self._tr(
                "qt_error_start_worker_generic", "Failed to start worker process."
            )
            QMessageBox.critical(
                self,
                self._tr("qt_error_start_worker_title", "Unable to start worker"),
                f"{message_template}\n{error_message}",
            )
        else:
            self._append_log(
                self._tr(
                    "qt_log_worker_running", "Worker is already running."
                ),
                level="warning",
            )

    def _on_filter_clicked(self) -> None:
        if self.is_processing:
            QMessageBox.warning(
                self,
                self._tr("qt_warning_already_running_title", "Processing in progress"),
                self._tr(
                    "qt_warning_already_running_message",
                    "A processing run is already active.",
                ),
            )
            return

        self._collect_config_from_widgets()
        result = self._launch_filter_dialog()
        if result is True:
            self._collect_config_from_widgets()
            self._start_processing(skip_filter_prompt=True, predecided_skip_filter_ui=True)

    def _launch_filter_dialog(self) -> bool | None:
        input_dir = str(self.config.get("input_dir", "") or "").strip()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(
                self,
                self._tr("qt_error_invalid_input_dir", "Please select a valid input folder before filtering."),
                self._tr("qt_error_invalid_input_dir", "Please select a valid input folder before filtering."),
            )
            return None

        if not self._input_dir_contains_fits(input_dir):
            QMessageBox.warning(
                self,
                self._tr("qt_error_no_fits", "No FITS files found in the selected input folder."),
                self._tr("qt_error_no_fits", "No FITS files found in the selected input folder."),
            )
            return None

        try:
            from zemosaic_filter_gui_qt import launch_filter_interface_qt
        except ImportError as exc:
            QMessageBox.warning(
                self,
                self._tr("qt_error_filter_unavailable", "Qt filter interface is not available."),
                str(exc),
            )
            return None

        initial_overrides: Dict[str, Any] | None = None
        try:
            initial_overrides = {
                "cluster_panel_threshold": float(self.config.get("cluster_panel_threshold", 0.05)),
                "cluster_target_groups": int(self.config.get("cluster_target_groups", 0)),
                "cluster_orientation_split_deg": float(self.config.get("cluster_orientation_split_deg", 0.0)),
            }
        except Exception:
            initial_overrides = None

        solver_payload = self._build_solver_settings_dict()
        try:
            global_coadd_k_value = float(self.config.get("global_coadd_k", 2.0) or 2.0)
        except Exception:
            global_coadd_k_value = 2.0
        try:
            global_wcs_padding_value = float(self.config.get("global_wcs_padding_percent", 2.0) or 2.0)
        except Exception:
            global_wcs_padding_value = 2.0

        config_overrides = {
            "astap_executable_path": str(self.config.get("astap_executable_path", "") or ""),
            "astap_data_directory_path": str(self.config.get("astap_data_directory_path", "") or ""),
            "astap_default_search_radius": float(self.config.get("astap_default_search_radius", 3.0) or 3.0),
            "astap_default_downsample": int(self.config.get("astap_default_downsample", 2) or 2),
            "astap_default_sensitivity": int(self.config.get("astap_default_sensitivity", 100) or 100),
            "output_dir": str(self.config.get("output_dir", "") or ""),
            "auto_detect_seestar": bool(self.config.get("auto_detect_seestar", True)),
            "force_seestar_mode": bool(self.config.get("force_seestar_mode", False)),
            "sds_mode_default": bool(self.config.get("sds_mode_default", False)),
            "global_coadd_method": str(self.config.get("global_coadd_method", "kappa_sigma") or "kappa_sigma"),
            "global_coadd_k": global_coadd_k_value,
            "global_wcs_output_path": str(
                self.config.get("global_wcs_output_path", "global_mosaic_wcs.fits")
                or "global_mosaic_wcs.fits"
            ),
            "global_wcs_pixelscale_mode": str(self.config.get("global_wcs_pixelscale_mode", "median") or "median"),
            "global_wcs_padding_percent": global_wcs_padding_value,
            "global_wcs_orientation": str(self.config.get("global_wcs_orientation", "north_up") or "north_up"),
            "global_wcs_res_override": self.config.get("global_wcs_res_override"),
            "apply_master_tile_crop": bool(self.config.get("apply_master_tile_crop", True)),
            "master_tile_crop_percent": float(self.config.get("master_tile_crop_percent", 3.0) or 3.0),
            "quality_crop_enabled": bool(self.config.get("quality_crop_enabled", False)),
            "quality_crop_band_px": int(self.config.get("quality_crop_band_px", 32) or 32),
            "quality_crop_k_sigma": float(self.config.get("quality_crop_k_sigma", 2.0) or 2.0),
            "quality_crop_margin_px": int(self.config.get("quality_crop_margin_px", 8) or 8),
            "quality_crop_min_run": int(self.config.get("quality_crop_min_run", 2) or 2),
            "altaz_cleanup_enabled": bool(self.config.get("altaz_cleanup_enabled", False)),
            "altaz_margin_percent": float(self.config.get("altaz_margin_percent", 5.0) or 5.0),
            "altaz_decay": float(self.config.get("altaz_decay", 0.15) or 0.15),
            "altaz_nanize": bool(self.config.get("altaz_nanize", True)),
            "quality_gate_enabled": bool(self.config.get("quality_gate_enabled", False)),
            "quality_gate_threshold": float(self.config.get("quality_gate_threshold", 0.48) or 0.48),
            "quality_gate_edge_band_px": int(self.config.get("quality_gate_edge_band_px", 64) or 64),
            "quality_gate_k_sigma": float(self.config.get("quality_gate_k_sigma", 2.5) or 2.5),
            "quality_gate_erode_px": int(self.config.get("quality_gate_erode_px", 3) or 3),
            "quality_gate_move_rejects": bool(self.config.get("quality_gate_move_rejects", True)),
            "auto_limit_frames_per_master_tile": bool(self.config.get("auto_limit_frames_per_master_tile", True)),
            "max_raw_per_master_tile": int(self.config.get("max_raw_per_master_tile", 0) or 0),
        }

        try:
            filter_result = launch_filter_interface_qt(
                input_dir,
                initial_overrides,
                stream_scan=True,
                scan_recursive=True,
                batch_size=200,
                preview_cap=1500,
                solver_settings_dict=solver_payload,
                config_overrides=config_overrides,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            template = self._tr(
                "qt_log_filter_error", "Filter UI error: {error}"
            )
            self._append_log(template.format(error=exc), level="warning")
            QMessageBox.warning(
                self,
                self._tr("qt_error_filter_launch", "The filter interface could not be opened."),
                str(exc),
            )
            return None

        filtered_list: Any = None
        accepted = True
        overrides: Dict[str, Any] | None = None
        if isinstance(filter_result, tuple) and len(filter_result) >= 1:
            filtered_list = filter_result[0]
            if len(filter_result) >= 2:
                try:
                    accepted = bool(filter_result[1])
                except Exception:
                    accepted = True
            if len(filter_result) >= 3 and isinstance(filter_result[2], dict):
                overrides = filter_result[2]
        else:
            filtered_list = filter_result

        if not accepted:
            self._append_log(self._tr("qt_log_filter_cancelled", "Filter cancelled by user."), level="warning")
            self._clear_filter_results()
            return False

        kept_count = 0
        total_count = None
        if isinstance(filtered_list, list):
            kept_count = len(filtered_list)
        if isinstance(overrides, dict):
            try:
                total_count = int(overrides.get("resolved_wcs_count"))
            except Exception:
                total_count = None

        summary = self._tr("qt_log_filter_summary", "Filter validated: kept {kept} files.")
        try:
            if total_count is not None:
                summary = self._tr(
                    "qt_log_filter_summary_total",
                    "Filter validated: kept {kept} of {total} files.",
                ).format(kept=kept_count, total=total_count)
            else:
                summary = summary.format(kept=kept_count)
        except Exception:
            pass
        self._append_log(summary, level="info")

        self._last_filter_overrides = overrides if isinstance(overrides, dict) else None
        self._last_filtered_header_items = filtered_list if isinstance(filtered_list, list) else None
        self._apply_filter_overrides_to_config(self._last_filter_overrides)

        return True

    def _apply_filter_overrides_to_config(self, overrides: Dict[str, Any] | None) -> None:
        if not overrides:
            return
        for key in (
            "cluster_panel_threshold",
            "cluster_target_groups",
            "cluster_orientation_split_deg",
        ):
            if key in overrides:
                self._update_widget_from_config(key, overrides[key])
        if "astap_max_instances" in overrides:
            self._update_widget_from_config("astap_max_instances", overrides["astap_max_instances"])
            self._apply_astap_concurrency_setting()

    def _update_widget_from_config(self, key: str, value: Any) -> None:
        self.config[key] = value
        binding = self._config_fields.get(key)
        if not binding:
            return
        widget = binding.get("widget")
        kind = binding.get("kind")
        try:
            if kind == "checkbox":
                widget.setChecked(bool(value))
            elif kind == "spinbox":
                widget.setValue(int(value))
            elif kind == "double_spinbox":
                widget.setValue(float(value))
            elif kind == "combobox":
                idx = widget.findData(value)
                if idx < 0:
                    idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                if key == "solver_method":
                    self._update_solver_visibility(str(value))
            else:
                widget.setText(str(value))
        except Exception:
            pass

    def _clear_filter_results(self) -> None:
        self._last_filter_overrides = None
        self._last_filtered_header_items = None

    @staticmethod
    def _input_dir_contains_fits(input_dir: str) -> bool:
        try:
            for root_dir, _dirs, files in os.walk(input_dir):
                for filename in files:
                    if filename.lower().endswith((".fit", ".fits")):
                        return True
                break
        except Exception:
            return False
        return False

    def _resolve_astap_max_instances(self) -> int:
        try:
            value = int(self.config.get("astap_max_instances", 1) or 1)
        except Exception:
            value = 1
        return max(1, value)

    def _apply_astap_concurrency_setting(self) -> None:
        instances = self._resolve_astap_max_instances()
        os.environ["ZEMOSAIC_ASTAP_MAX_PROCS"] = str(instances)
        if set_astap_max_concurrent_instances is not None:
            try:
                set_astap_max_concurrent_instances(instances)
            except Exception:
                pass


def run_qt_main() -> int:
    """Launch the Qt main window, creating a QApplication if needed."""

    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True

    window = ZeMosaicQtMainWindow()
    window.show()

    if owns_app:
        return app.exec()

    # If another part of the application owns the QApplication, keep the
    # existing event loop running and signal success to the caller.
    return 0


__all__ = ["run_qt_main", "ZeMosaicQtMainWindow"]
