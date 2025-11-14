"""Minimal PySide6-based GUI entry point for ZeMosaic."""
from __future__ import annotations

import sys

try:
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "PySide6 is required to use the ZeMosaic Qt interface. "
        "Install PySide6 or use the Tk interface instead."
    ) from exc


class ZeMosaicQtMainWindow(QMainWindow):
    """Minimal placeholder window for the upcoming Qt interface."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ZeMosaic (Qt Preview)")

        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(QLabel("ZeMosaic Qt interface is under construction."))
        self.setCentralWidget(central_widget)


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
