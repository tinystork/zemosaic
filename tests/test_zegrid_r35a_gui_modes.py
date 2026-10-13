"""ZM-ZEGRID-R35A GUI tests: explicit pipeline modes + Filter round-trip.

Uses an offscreen Qt platform; no real processing is started.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:  # pragma: no cover - Qt not installed
        pytest.skip("PySide6 unavailable")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_main_window(tmp_path, monkeypatch, config: dict):
    import zemosaic.zemosaic_gui_qt as gui

    cfg_path = tmp_path / "zemosaic_config.json"
    fake_cfg = SimpleNamespace(
        load_config=lambda: dict(config),
        save_config=lambda data: None,
        get_config_path=lambda: cfg_path,
        DEFAULT_CONFIG={},
    )
    monkeypatch.setattr(gui, "zemosaic_config", fake_cfg)
    monkeypatch.setattr(gui, "_detect_analysis_backend", lambda: ("none", None, None))
    win = gui.ZeMosaicQtMainWindow()
    return win


def test_main_checkboxes_are_mutually_exclusive(tmp_path, monkeypatch):
    win = _make_main_window(tmp_path, monkeypatch, {})
    boxes = win._mode_checkboxes
    sds = boxes["sds"]
    zegrid = boxes["zegrid"]
    master = boxes["existing_master_tiles"]

    sds.setChecked(True)
    assert win.config["sds_mode_default"] is True
    assert win.config["zegrid_mode_enabled"] is False
    assert win.config["use_existing_master_tiles"] is False
    assert win.config["processing_mode"] == "sds"

    zegrid.setChecked(True)
    assert win.config["zegrid_mode_enabled"] is True
    assert win.config["sds_mode_default"] is False
    assert win.config["use_existing_master_tiles"] is False
    assert win.config["processing_mode"] == "zegrid"

    master.setChecked(True)
    assert win.config["use_existing_master_tiles"] is True
    assert win.config["zegrid_mode_enabled"] is False
    assert win.config["sds_mode_default"] is False
    assert win.config["processing_mode"] == "existing_master_tiles"

    master.setChecked(False)
    assert win.config["processing_mode"] == "classic"


def test_main_window_hides_global_wcs_field_and_shows_mode_boxes(tmp_path, monkeypatch):
    win = _make_main_window(tmp_path, monkeypatch, {})
    # The Global WCS output path is strictly UX debt: no widget, key preserved.
    assert "global_wcs_output_path" not in win._config_fields
    assert win.config.get("global_wcs_output_path", "global_mosaic_wcs.fits") == "global_mosaic_wcs.fits"
    # The three pipeline choices + the temp dir are all present.
    for key in ("use_existing_master_tiles", "sds_mode_default", "zegrid_mode_enabled", "temporary_work_dir"):
        assert key in win._config_fields


def test_system_tab_uses_plain_language_and_drives_legacy_cleanup(tmp_path, monkeypatch):
    win = _make_main_window(tmp_path, monkeypatch, {})
    # Legacy coadd memmap dir no longer exposed; the single temp dir is.
    assert "coadd_memmap_dir" not in win._config_fields
    assert "coadd_cleanup_memmap" not in win._config_fields
    binding = win._config_fields["cleanup_temp_artifacts"]
    assert callable(binding.get("postprocess"))
    binding["postprocess"](False)
    assert win.config["coadd_cleanup_memmap"] is False
    binding["postprocess"](True)
    assert win.config["coadd_cleanup_memmap"] is True


def test_filter_zegrid_and_sds_are_mutually_exclusive(tmp_path):
    import zemosaic.zemosaic_filter_gui_qt as filt

    dialog = filt.FilterQtDialog(str(tmp_path), {"zegrid_mode": True})
    try:
        assert dialog._zegrid_checkbox.isChecked() is True
        assert dialog._sds_checkbox.isChecked() is False
        dialog._sds_checkbox.setChecked(True)
        assert dialog._zegrid_checkbox.isChecked() is False
        dialog._zegrid_checkbox.setChecked(True)
        assert dialog._sds_checkbox.isChecked() is False
    finally:
        dialog.deleteLater()


def test_filter_overrides_round_trip_mode(tmp_path):
    import zemosaic.zemosaic_filter_gui_qt as filt

    d1 = filt.FilterQtDialog(str(tmp_path), {"zegrid_mode": True})
    try:
        ov = d1.overrides() or {}
        assert ov.get("zegrid_mode") is True
        assert ov.get("sds_mode") is False
    finally:
        d1.deleteLater()

    d2 = filt.FilterQtDialog(str(tmp_path), {"sds_mode": True})
    try:
        ov = d2.overrides() or {}
        assert ov.get("sds_mode") is True
        assert ov.get("zegrid_mode") is False
    finally:
        d2.deleteLater()


def test_main_reflects_filter_selection(tmp_path, monkeypatch):
    win = _make_main_window(tmp_path, monkeypatch, {})
    win._apply_filter_overrides_to_config({"zegrid_mode": True, "sds_mode": False})
    assert win.config["zegrid_mode_enabled"] is True
    assert win.config["sds_mode_default"] is False
    assert win.config["processing_mode"] == "zegrid"

    win._apply_filter_overrides_to_config({"sds_mode": True, "zegrid_mode": False})
    assert win.config["sds_mode_default"] is True
    assert win.config["zegrid_mode_enabled"] is False
    assert win.config["processing_mode"] == "sds"


def test_filter_initial_state_reflects_main_window(tmp_path, monkeypatch):
    import zemosaic.zemosaic_filter_gui_qt as filt

    dialog = filt.FilterQtDialog(
        str(tmp_path), None, config_overrides={"zegrid_mode_enabled": True}
    )
    try:
        assert dialog._zegrid_checkbox.isChecked() is True
        assert dialog._sds_checkbox.isChecked() is False
    finally:
        dialog.deleteLater()
