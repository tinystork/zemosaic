#!/usr/bin/env python3
"""PySide6 GUI for master_tile_overlap_diagnostic.py.

This provides a simple graphical interface to configure and run the
master tile overlap diagnostic tool.
"""

from __future__ import annotations

import sys
import shlex
from pathlib import Path
from typing import Any, List, Optional

try:
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
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
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise ImportError(
        "Unable to import PySide6 which is required for the GUI. "
        "Install with `pip install PySide6`."
    ) from exc

from master_tile_overlap_diagnostic import main as diagnostic_main, parse_args


class DiagnosticWorker(QThread):
    """Worker thread to run the diagnostic in the background."""

    progress_updated = Signal(str)
    finished = Signal(int)

    def __init__(self, args: List[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.args = args

    def run(self) -> None:
        try:
            # Capture stdout and stderr
            import io
            from contextlib import redirect_stderr, redirect_stdout

            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exit_code = diagnostic_main(self.args)

            output = stdout_capture.getvalue() + stderr_capture.getvalue()
            self.progress_updated.emit(output)
            self.finished.emit(exit_code)
        except Exception as e:
            self.progress_updated.emit(f"Error: {e}")
            self.finished.emit(1)


class MasterTileOverlapDiagnosticGUI(QMainWindow):
    """Main GUI window for the diagnostic tool."""

    def __init__(self) -> None:
        super().__init__()
        self.worker: Optional[DiagnosticWorker] = None
        self.setWindowTitle("ZeMosaic Master Tile Overlap Diagnostic")
        self.setGeometry(100, 100, 600, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Input section
        input_group = QGroupBox("Input")
        input_layout = QFormLayout(input_group)

        input_container = QWidget()
        input_row = QHBoxLayout(input_container)
        input_row.setContentsMargins(0, 0, 0, 0)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Paths to FITS files or directories")
        input_row.addWidget(self.input_edit)

        browse_input_button = QPushButton("Browse...")
        browse_input_button.clicked.connect(self.browse_input)
        input_row.addWidget(browse_input_button)

        input_layout.addRow("Inputs:", input_container)

        self.recursive_check = QCheckBox("Recursive search in directories")
        self.recursive_check.setChecked(True)
        input_layout.addRow(self.recursive_check)

        layout.addWidget(input_group)

        # Output section
        output_group = QGroupBox("Output")
        output_layout = QFormLayout(output_group)

        self.outdir_edit = QLineEdit("tile_overlap_diag")
        output_layout.addRow("Output directory:", self.outdir_edit)

        self.prefix_edit = QLineEdit("master_tiles")
        output_layout.addRow("Filename prefix:", self.prefix_edit)

        self.min_overlap_spin = QDoubleSpinBox()
        self.min_overlap_spin.setRange(0.0, 1.0)
        self.min_overlap_spin.setSingleStep(0.01)
        self.min_overlap_spin.setValue(0.0)
        output_layout.addRow("Min overlap fraction:", self.min_overlap_spin)

        layout.addWidget(output_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Diagnostic")
        self.run_button.clicked.connect(self.run_diagnostic)
        button_layout.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_diagnostic)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Log output
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("Output will appear here...")
        layout.addWidget(self.log_edit)

    def run_diagnostic(self) -> None:
        inputs = self.input_edit.text().strip()
        if not inputs:
            QMessageBox.warning(self, "Error", "Please specify input paths.")
            return

        try:
            # Use shell-like parsing so quoted paths containing spaces are preserved
            # (e.g. /media/tristan/X10 Pro/...).
            args = shlex.split(inputs)
        except ValueError as exc:
            QMessageBox.warning(self, "Error", f"Invalid input paths: {exc}")
            return

        # Normalize accidental extra quotes from manual edits.
        args = [a.strip().strip('"').strip("'") for a in args if a.strip()]
        if self.recursive_check.isChecked():
            args.insert(0, "--recursive")

        args.extend(["--outdir", self.outdir_edit.text()])
        args.extend(["--prefix", self.prefix_edit.text()])
        min_overlap = self.min_overlap_spin.value()
        if min_overlap > 0:
            args.extend(["--min-overlap-frac", str(min_overlap)])

        self.log_edit.clear()
        self.progress_bar.setVisible(True)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.worker = DiagnosticWorker(args)
        self.worker.progress_updated.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def stop_diagnostic(self) -> None:
        if self.worker:
            self.worker.terminate()
            self.worker.wait()
            self.on_finished(1)

    def on_progress(self, text: str) -> None:
        self.log_edit.appendPlainText(text)

    def browse_input(self) -> None:
        # Allow selecting multiple files and directories
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select FITS files or directories",
            "",
            "FITS files (*.fits *.fit *.fts *.fz);;All files (*)",
        )
        if files:
            current = self.input_edit.text().strip()
            if current:
                current += " "
            current += " ".join(f'"{path}"' for path in files)
            self.input_edit.setText(current)

        # Also allow selecting directories
        dir_path = QFileDialog.getExistingDirectory(self, "Select directory")
        if dir_path:
            current = self.input_edit.text().strip()
            if current:
                current += " "
            current += f'"{dir_path}"'
            self.input_edit.setText(current)

    def on_finished(self, exit_code: int) -> None:
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.worker:
            self.worker = None

        if exit_code == 0:
            QMessageBox.information(self, "Success", "Diagnostic completed successfully.")
        else:
            QMessageBox.warning(self, "Error", f"Diagnostic failed with exit code {exit_code}.")


def main() -> int:
    """Start the application event loop."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = MasterTileOverlapDiagnosticGUI()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
