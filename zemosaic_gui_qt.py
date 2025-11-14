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
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
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
        self._default_config_values: Dict[str, Any] = {
            "input_dir": "",
            "output_dir": "",
            "global_wcs_output_path": "global_mosaic_wcs.fits",
            "coadd_memmap_dir": "",
            "astap_executable_path": "",
            "astap_data_directory_path": "",
            "astap_default_search_radius": 3.0,
            "astap_default_downsample": 2,
            "cluster_panel_threshold": 0.05,
            "cluster_target_groups": 0,
            "cluster_orientation_split_deg": 0.0,
            "quality_crop_enabled": False,
            "quality_crop_band_px": 32,
            "altaz_cleanup_enabled": False,
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
        self._register_line_edit(
            "coadd_memmap_dir",
            layout,
            self._tr("qt_field_coadd_memmap_dir", "Memmap directory"),
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
        )
        self._register_line_edit(
            "astap_data_directory_path",
            layout,
            self._tr("qt_field_astap_data_dir", "ASTAP data directory"),
        )
        self._register_line_edit(
            "astap_default_search_radius",
            layout,
            self._tr("qt_field_astap_search_radius", "Default search radius"),
        )
        self._register_line_edit(
            "astap_default_downsample",
            layout,
            self._tr("qt_field_astap_downsample", "Default downsample"),
        )

        return group

    def _create_mosaic_group(self) -> QGroupBox:
        group = QGroupBox(self._tr("qt_group_mosaic", "Mosaic / clustering"), self)
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._register_line_edit(
            "cluster_panel_threshold",
            layout,
            self._tr("qt_field_cluster_threshold", "Cluster threshold"),
        )
        self._register_line_edit(
            "cluster_target_groups",
            layout,
            self._tr("qt_field_cluster_target", "Target groups"),
        )
        self._register_line_edit(
            "cluster_orientation_split_deg",
            layout,
            self._tr("qt_field_cluster_orientation", "Orientation split (°)"),
        )

        return group

    def _create_quality_group(self) -> QGroupBox:
        group = QGroupBox(
            self._tr("qt_group_quality", "Cropping / quality / Alt-Az"),
            self,
        )
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._register_checkbox(
            "quality_crop_enabled",
            layout,
            self._tr("qt_field_quality_crop_enabled", "Enable quality crop"),
        )
        self._register_line_edit(
            "quality_crop_band_px",
            layout,
            self._tr("qt_field_quality_crop_band", "Quality crop band (px)"),
        )
        self._register_checkbox(
            "altaz_cleanup_enabled",
            layout,
            self._tr("qt_field_altaz_enabled", "Enable Alt-Az cleanup"),
        )

        return group

    def _create_logging_group(self) -> QGroupBox:
        group = QGroupBox(self._tr("qt_group_logging", "Logging / progress"), self)
        layout = QVBoxLayout(group)

        self.progress_bar = QProgressBar(group)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.log_output = QPlainTextEdit(group)
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText(
            self._tr(
                "qt_log_placeholder",
                "Worker output will appear here once the integration is complete.",
            )
        )
        layout.addWidget(self.log_output)

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

    def _register_line_edit(self, key: str, layout: QFormLayout, label_text: str) -> None:
        widget = QLineEdit()
        current_value = self.config.get(key)
        widget.setText("" if current_value is None else str(current_value))
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

    def _collect_config_from_widgets(self) -> None:
        for key, binding in self._config_fields.items():
            kind = binding["kind"]
            widget = binding["widget"]
            expected_type = binding["type"]
            if kind == "checkbox":
                self.config[key] = bool(widget.isChecked())
                continue

            raw_text = widget.text().strip()
            if expected_type in {int, float}:
                try:
                    converted: Any = expected_type(raw_text)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    converted = self.config.get(key)
            elif expected_type is bool:
                normalized = raw_text.lower()
                if normalized in {"1", "true", "yes", "on"}:
                    converted = True
                elif normalized in {"0", "false", "no", "off"}:
                    converted = False
                else:
                    converted = self.config.get(key, False)
            else:
                converted = raw_text
            self.config[key] = converted

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
                self._append_log(f"Failed to save configuration: {exc}")

    def _create_localizer(self, language_code: str) -> Any:
        if ZeMosaicLocalization is not None:
            return ZeMosaicLocalization(language_code=language_code)
        return _FallbackLocalizer(language_code=language_code)

    def _tr(self, key: str, fallback: str) -> str:
        return self.localizer.get(key, fallback)

    # ------------------------------------------------------------------
    # Events & callbacks
    # ------------------------------------------------------------------
    def _append_log(self, message: str) -> None:
        if hasattr(self, "log_output"):
            self.log_output.appendPlainText(message)
        else:  # pragma: no cover - initialization guard
            print(message)

    def _on_start_clicked(self) -> None:
        self._collect_config_from_widgets()
        self._save_config()
        self._append_log(
            self._tr(
                "qt_log_start_placeholder",
                "Start requested (worker integration pending).",
            )
        )

    def _on_stop_clicked(self) -> None:
        self._append_log(
            self._tr("qt_log_stop_placeholder", "Stop requested (no worker yet).")
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
