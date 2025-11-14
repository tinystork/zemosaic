"""Minimal PySide6-based GUI entry point for ZeMosaic."""
from __future__ import annotations

import importlib.util
import os
import sys
import multiprocessing
import queue
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    from PySide6.QtCore import QObject, QTimer, Signal
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
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
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "PySide6 is required to use the ZeMosaic Qt interface. "
        "Install PySide6 or use the Tk interface instead."
    ) from exc

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


class _FallbackLocalizer:
    """Very small localization shim when the real helper is unavailable."""

    def __init__(self, language_code: str = "en") -> None:
        self.language_code = language_code

    def get(self, key: str, default_text: str | None = None, **_: Any) -> str:
        return default_text if default_text is not None else key

    def set_language(self, language_code: str) -> None:
        self.language_code = language_code


class ZeMosaicQtWorker(QObject):
    """Manage the background ZeMosaic worker process and emit Qt signals."""

    log_message_emitted = Signal(str, str)
    progress_changed = Signal(float)
    stage_progress = Signal(str, int, int)
    phase_changed = Signal(str, dict)
    stats_updated = Signal(dict)
    finished = Signal(bool, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._queue: multiprocessing.Queue | None = None
        self._process: multiprocessing.Process | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._poll_queue)  # type: ignore[arg-type]
        self._stop_requested = False
        self._had_error = False
        self._last_error: str = ""
        self._finished_emitted = False

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
        if run_hierarchical_mosaic_process is None:
            raise RuntimeError("Worker backend is unavailable")
        if self.is_running():
            return False

        queue_obj: multiprocessing.Queue | None = None
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
            # Clean up any partially created resources
            try:
                if queue_obj is not None:
                    queue_obj.close()
            except Exception:
                pass
            raise RuntimeError(f"Failed to start worker process: {exc}") from exc

        self._queue = queue_obj
        self._process = process
        self._timer.start()
        self._stop_requested = False
        self._had_error = False
        self._last_error = ""
        self._finished_emitted = False
        return True

    def stop(self) -> None:
        self._stop_requested = True
        proc = self._process
        if proc and proc.is_alive():
            try:
                proc.terminate()
                proc.join(timeout=0.5)
            except Exception:
                pass
        self._finalize(success=False, message=self.tr("Processing cancelled"))

    # ------------------------------------------------------------------
    # Queue polling
    # ------------------------------------------------------------------
    def _poll_queue(self) -> None:
        queue_obj = self._queue
        proc = self._process
        if queue_obj is None or proc is None:
            self._finalize(success=not self._had_error and not self._stop_requested, message=self._last_error)
            return

        drained_any = False
        while True:
            try:
                payload = queue_obj.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break

            drained_any = True
            try:
                msg_key, prog, lvl, kwargs = payload
            except Exception:
                continue
            kwargs = kwargs or {}

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
                continue

            if msg_key == "PROCESS_ERROR":
                error_text = str(kwargs.get("error") or prog or "")
                self._had_error = True
                self._last_error = error_text
                level = str(lvl or "ERROR")
                if error_text:
                    self.log_message_emitted.emit(level, error_text)
                else:
                    self.log_message_emitted.emit(level, "Worker error")
                continue

            if msg_key == "PROCESS_DONE":
                # Let the process exit naturally; finalization happens once it stops.
                continue

            if isinstance(msg_key, str) and msg_key.startswith("p45_"):
                # Advanced Phase 4.5 feedback is not yet visualized in Qt.
                continue

            if isinstance(msg_key, str) and msg_key.upper() == "STATS_UPDATE":
                if isinstance(kwargs, dict):
                    self.stats_updated.emit(kwargs)
                continue

            level = str(lvl or "INFO")
            message = self._stringify_message(msg_key, prog, kwargs)
            self.log_message_emitted.emit(level, message)

        if proc is not None and not proc.is_alive():
            success = not self._had_error and not self._stop_requested
            message = "" if success else (self._last_error or self.tr("Processing cancelled"))
            self._finalize(success=success, message=message)
            return

        if self._stop_requested and not drained_any:
            self._finalize(success=False, message=self.tr("Processing cancelled"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _finalize(self, *, success: bool, message: str) -> None:
        if self._timer.isActive():
            self._timer.stop()
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
        return " | ".join(parts) if parts else "(worker update)"

class ZeMosaicQtMainWindow(QMainWindow):
    """Initial Qt main window skeleton with placeholder panels."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ZeMosaic (Qt Preview)")

        self.config: Dict[str, Any] = self._load_config()
        self.localizer = self._create_localizer(self.config.get("language", "en"))
        self._log_level_prefixes = {
            "debug": self._tr("qt_log_prefix_debug", "[DEBUG] "),
            "info": self._tr("qt_log_prefix_info", "[INFO] "),
            "success": self._tr("qt_log_prefix_success", "[SUCCESS] "),
            "warning": self._tr("qt_log_prefix_warning", "[WARNING] "),
            "error": self._tr("qt_log_prefix_error", "[ERROR] "),
        }
        self._default_config_values: Dict[str, Any] = {
            "input_dir": "",
            "output_dir": "",
            "global_wcs_output_path": "global_mosaic_wcs.fits",
            "coadd_memmap_dir": "",
            "astap_executable_path": "",
            "astap_data_directory_path": "",
            "astap_default_search_radius": 3.0,
            "astap_default_downsample": 2,
            "astap_default_sensitivity": 100,
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
            "logging_level": "INFO",
        }
        for key, fallback in self._default_config_values.items():
            self.config.setdefault(key, fallback)
        self._config_fields: Dict[str, Dict[str, Any]] = {}

        self._setup_ui()

        self.is_processing = False
        self.worker_controller = ZeMosaicQtWorker(self)
        self.worker_controller.log_message_emitted.connect(self._on_worker_log_message)  # type: ignore[arg-type]
        self.worker_controller.progress_changed.connect(self._on_worker_progress_changed)  # type: ignore[arg-type]
        self.worker_controller.stage_progress.connect(self._on_worker_stage_progress)  # type: ignore[arg-type]
        self.worker_controller.phase_changed.connect(self._on_worker_phase_changed)  # type: ignore[arg-type]
        self.worker_controller.stats_updated.connect(self._on_worker_stats_updated)  # type: ignore[arg-type]
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
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        main_layout.addWidget(self._create_folders_group())
        main_layout.addWidget(self._create_astap_group())
        main_layout.addWidget(self._create_mosaic_group())
        main_layout.addWidget(self._create_quality_group())
        main_layout.addWidget(self._create_logging_group())

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.start_button = QPushButton(self._tr("qt_button_start", "Start"))
        self.stop_button = QPushButton(self._tr("qt_button_stop", "Stop"))
        self.start_button.clicked.connect(self._on_start_clicked)  # type: ignore[attr-defined]
        self.stop_button.clicked.connect(self._on_stop_clicked)  # type: ignore[attr-defined]
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        main_layout.addLayout(button_row)

        self.setCentralWidget(central_widget)

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
        self._register_directory_picker(
            "coadd_memmap_dir",
            layout,
            self._tr("qt_field_coadd_memmap_dir", "Memmap directory"),
            dialog_title=self._tr(
                "qt_dialog_select_memmap_dir", "Select Memmap Folder"
            ),
        )

        return group

    def _create_astap_group(self) -> QGroupBox:
        group = QGroupBox(self._tr("qt_group_astap", "ASTAP configuration"), self)
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._register_line_edit(
            "astap_executable_path",
            layout,
            self._tr("qt_field_astap_executable", "ASTAP executable"),
            browse_action="file",
            dialog_title=self._tr(
                "qt_dialog_select_astap_executable", "Select ASTAP Executable"
            ),
        )
        self._register_line_edit(
            "astap_data_directory_path",
            layout,
            self._tr("qt_field_astap_data_dir", "ASTAP data directory"),
            browse_action="directory",
            dialog_title=self._tr(
                "qt_dialog_select_astap_data_dir", "Select ASTAP Data Directory"
            ),
        )
        self._register_double_spinbox(
            "astap_default_search_radius",
            layout,
            self._tr("qt_field_astap_search_radius", "Default search radius (°)"),
            minimum=0.1,
            maximum=180.0,
            single_step=0.1,
        )
        self._register_spinbox(
            "astap_default_downsample",
            layout,
            self._tr("qt_field_astap_downsample", "Default downsample"),
            minimum=0,
            maximum=4,
        )
        self._register_spinbox(
            "astap_default_sensitivity",
            layout,
            self._tr("qt_field_astap_sensitivity", "Default sensitivity"),
            minimum=-25,
            maximum=500,
        )

        return group

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

        cache_combo = QComboBox()
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

        phase_label = QLabel(self._tr("qt_progress_phase_label", "Phase:"), group)
        self.phase_value_label = QLabel(self._tr("qt_progress_placeholder", "Idle"), group)
        stats_layout.addWidget(phase_label, 0, 0)
        stats_layout.addWidget(self.phase_value_label, 0, 1)

        eta_label = QLabel(self._tr("qt_progress_eta_label", "ETA:"), group)
        self.eta_value_label = QLabel(self._tr("qt_progress_placeholder", "—"), group)
        stats_layout.addWidget(eta_label, 0, 2)
        stats_layout.addWidget(self.eta_value_label, 0, 3)

        elapsed_label = QLabel(self._tr("qt_progress_elapsed_label", "Elapsed:"), group)
        self.elapsed_value_label = QLabel(self._tr("qt_progress_placeholder", "—"), group)
        stats_layout.addWidget(elapsed_label, 0, 4)
        stats_layout.addWidget(self.elapsed_value_label, 0, 5)

        files_label = QLabel(self._tr("qt_progress_files_label", "Files:"), group)
        self.files_value_label = QLabel(
            self._tr("qt_progress_count_placeholder", "0 / 0"),
            group,
        )
        stats_layout.addWidget(files_label, 1, 0)
        stats_layout.addWidget(self.files_value_label, 1, 1)

        tiles_label = QLabel(self._tr("qt_progress_tiles_label", "Tiles:"), group)
        self.tiles_value_label = QLabel(
            self._tr("qt_progress_count_placeholder", "0 / 0"),
            group,
        )
        stats_layout.addWidget(tiles_label, 1, 2)
        stats_layout.addWidget(self.tiles_value_label, 1, 3)

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

        return group

    # ------------------------------------------------------------------
    # Configuration handling
    # ------------------------------------------------------------------
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

    def _load_config(self) -> Dict[str, Any]:
        if zemosaic_config is not None and hasattr(zemosaic_config, "load_config"):
            config = zemosaic_config.load_config()
        else:  # pragma: no cover - minimal fallback
            config = {
                "language": "en",
                "astap_executable_path": "",
                "astap_data_directory_path": "",
                "astap_default_search_radius": 3.0,
                "astap_default_downsample": 2,
                "astap_default_sensitivity": 100,
                "cluster_panel_threshold": 0.05,
                "cluster_target_groups": 0,
                "cluster_orientation_split_deg": 0.0,
                "quality_crop_enabled": False,
                "quality_crop_band_px": 32,
                "altaz_cleanup_enabled": False,
                "global_wcs_output_path": "global_mosaic_wcs.fits",
                "coadd_memmap_dir": "",
            }
        config.setdefault("language", "en")
        return dict(config)

    def _save_config(self) -> None:
        if zemosaic_config is not None and hasattr(zemosaic_config, "save_config"):
            try:
                zemosaic_config.save_config(self.config)
            except Exception as exc:  # pragma: no cover - log instead of raising
                self._append_log(
                    f"Failed to save configuration: {exc}", level="error"
                )

    def _create_localizer(self, language_code: str) -> Any:
        if ZeMosaicLocalization is not None:
            return ZeMosaicLocalization(language_code=language_code)
        return _FallbackLocalizer(language_code=language_code)

    def _tr(self, key: str, fallback: str) -> str:
        return self.localizer.get(key, fallback)

    # ------------------------------------------------------------------
    # Events & callbacks
    # ------------------------------------------------------------------
    def _append_log(self, message: str, level: str = "info") -> None:
        normalized_level = level.lower().strip()
        prefix = self._log_level_prefixes.get(normalized_level)
        if prefix is None and normalized_level:
            prefix = f"[{normalized_level.upper()}] "
        formatted_message = f"{prefix}{message}" if prefix else message
        if hasattr(self, "log_output"):
            self.log_output.appendPlainText(formatted_message)
        else:  # pragma: no cover - initialization guard
            print(formatted_message)

    def _clear_log(self) -> None:
        if hasattr(self, "log_output"):
            self.log_output.clear()

    def _set_processing_state(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if running:
            self.progress_bar.setValue(0)
            self.phase_value_label.setText(self._tr("qt_progress_placeholder", "Idle"))
            self.eta_value_label.setText(self._tr("qt_progress_placeholder", "—"))
            self.elapsed_value_label.setText("00:00:00")
            self.files_value_label.setText(self._tr("qt_progress_count_placeholder", "0 / 0"))
            self.tiles_value_label.setText(self._tr("qt_progress_count_placeholder", "0 / 0"))
        else:
            self.progress_bar.setValue(0)
            self.phase_value_label.setText(self._tr("qt_progress_placeholder", "Idle"))
            self.eta_value_label.setText(self._tr("qt_progress_placeholder", "—"))
            self.elapsed_value_label.setText(self._tr("qt_progress_placeholder", "—"))
            self.files_value_label.setText(self._tr("qt_progress_count_placeholder", "0 / 0"))
            self.tiles_value_label.setText(self._tr("qt_progress_count_placeholder", "0 / 0"))

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

    def _translate_worker_message(self, message: str) -> str:
        if message.startswith("log_key_") or message.startswith("qt_"):
            translated = self._tr(message, message)
            if translated:
                return translated
        return message

    def _on_worker_log_message(self, level: str, message: str) -> None:
        normalized_level = self._normalize_log_level(level)
        translated_message = self._translate_worker_message(str(message))
        self._append_log(translated_message, level=normalized_level)

    def _on_worker_progress_changed(self, percent: float) -> None:
        bounded = int(max(0.0, min(100.0, percent)))
        self.progress_bar.setValue(bounded)

    def _on_worker_stage_progress(self, stage: str, current: int, total: int) -> None:
        stage_label = self._format_stage_name(stage)
        self.phase_value_label.setText(stage_label)
        percent = 0
        if total > 0:
            percent = int(max(0.0, min(100.0, (current / float(total)) * 100.0)))
        self.progress_bar.setValue(percent)
        if total > 0:
            self.tiles_value_label.setText(f"{current} / {total}")

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
            self.files_value_label.setText(f"{files_done} / {files_total}")
        tiles_done = payload.get("tiles_done")
        tiles_total = payload.get("tiles_total")
        if isinstance(tiles_done, int) and isinstance(tiles_total, int) and tiles_total >= 0:
            self.tiles_value_label.setText(f"{tiles_done} / {tiles_total}")
        eta_seconds = payload.get("eta_seconds")
        if isinstance(eta_seconds, (int, float)) and eta_seconds >= 0:
            eta_h, eta_rem = divmod(int(eta_seconds + 0.5), 3600)
            eta_m, eta_s = divmod(eta_rem, 60)
            self.eta_value_label.setText(f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}")

    def _on_worker_finished(self, success: bool, message: str) -> None:
        self.is_processing = False
        self._elapsed_timer.stop()
        self._run_started_monotonic = None
        self._set_processing_state(False)
        self.stop_button.setEnabled(False)
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
        else:
            if message:
                translated = self._translate_worker_message(message)
            else:
                translated = self._tr(
                    "qt_log_processing_cancelled", "Processing cancelled."
                )
            level = "error" if message else "warning"
            self._append_log(translated, level=level)
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

    def _format_stage_name(self, stage: str) -> str:
        if not stage:
            return self._tr("qt_progress_placeholder", "Idle")
        key = f"qt_stage_{stage.lower()}"
        fallback = stage.replace("_", " ").strip().title()
        return self._tr(key, fallback)

    def _build_worker_invocation(self) -> tuple[Tuple[Any, ...], Dict[str, Any]]:
        cfg = self.config

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
        return worker_args, worker_kwargs

    def _build_solver_settings_dict(self) -> Dict[str, Any]:
        astap_exe = str(self.config.get("astap_executable_path", "") or "")
        astap_data = str(self.config.get("astap_data_directory_path", "") or "")
        search_radius = float(self.config.get("astap_default_search_radius", 3.0) or 3.0)
        astap_downsample = int(self.config.get("astap_default_downsample", 2) or 2)
        astap_sensitivity = int(self.config.get("astap_default_sensitivity", 100) or 100)
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
        return asdict(settings)

    def _on_start_clicked(self) -> None:
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
        self._save_config()

        try:
            worker_args, worker_kwargs = self._build_worker_invocation()
        except ValueError as exc:
            QMessageBox.warning(self, self._tr("qt_error_invalid_config_title", "Invalid configuration"), str(exc))
            return
        except Exception as exc:  # pragma: no cover - unexpected preparation failure
            message = self._tr("qt_error_prepare_worker", "Unable to prepare worker arguments.")
            self._append_log(f"{message} {exc}", level="error")
            QMessageBox.critical(self, self._tr("qt_error_prepare_worker_title", "Worker preparation failed"), str(exc))
            return

        try:
            started = self.worker_controller.start(worker_args, worker_kwargs)
        except Exception as exc:  # pragma: no cover - start failures are rare
            self._append_log(f"Failed to start worker: {exc}", level="error")
            QMessageBox.critical(
                self,
                self._tr("qt_error_start_worker_title", "Unable to start worker"),
                str(exc),
            )
            return

        if not started:
            self._append_log("Worker is already running", level="warning")
            return

        self.is_processing = True
        self._run_started_monotonic = time.monotonic()
        self._elapsed_timer.start()
        self._set_processing_state(True)
        self._append_log(
            self._tr("qt_log_processing_started", "Processing started."),
            level="info",
        )

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
            self._append_log(f"Failed to stop worker cleanly: {exc}", level="error")
        self.stop_button.setEnabled(False)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self._collect_config_from_widgets()
        self._save_config()
        if self.is_processing:
            try:
                self.worker_controller.stop()
            except Exception:
                pass
        super().closeEvent(event)


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
