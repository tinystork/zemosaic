"""Stub for the ZeMosaic Qt-based filter interface."""
from __future__ import annotations

try:
    from PySide6.QtWidgets import QWidget  # noqa: F401  # placeholder import for future use
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "PySide6 is required to use the ZeMosaic Qt filter interface. "
        "Install PySide6 or use the Tk interface instead."
    ) from exc


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
    """Placeholder implementation of the Qt filter GUI.

    Returns the input data unchanged until the Qt dialog is implemented.
    """

    # TODO: implement the Qt filter GUI
    return raw_files_with_wcs_or_dir, False, None


__all__ = ["launch_filter_interface_qt"]
