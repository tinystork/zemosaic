"""Minimal PySide6-based GUI entry point for ZeMosaic."""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict

try:
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

    def _on_start_clicked(self) -> None:
        self._collect_config_from_widgets()
        self._save_config()
        self._append_log(
            self._tr(
                "qt_log_start_placeholder",
                "Start requested (worker integration pending).",
            ),
            level="info",
        )

    def _on_stop_clicked(self) -> None:
        self._append_log(
            self._tr("qt_log_stop_placeholder", "Stop requested (no worker yet)."),
            level="warning",
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self._collect_config_from_widgets()
        self._save_config()
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
