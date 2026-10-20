"""ZM-ZEGRID-R53C — Qt/GUI, persistence, locales and GUI->worker->engine proof.

The Qt tests run offscreen. The final test uses a real main window config, the
real worker process wrapper, the real ``run_hierarchical_mosaic`` ZeGrid
dispatch and the real engine on a tiny deterministic corpus.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zegrid_halo as zhalo  # noqa: E402

LOCALES = ("en", "fr", "de", "es", "is", "nl", "pl")
HALO_KEYS = (
    "qt_field_zegrid_halo_label",
    "qt_field_zegrid_halo_tip",
    "qt_zegrid_halo_opt_8",
    "qt_zegrid_halo_opt_16",
    "qt_zegrid_halo_opt_32",
    "qt_zegrid_halo_opt_64",
    "qt_zegrid_halo_opt_128",
    "qt_zegrid_halo_opt_256",
    "qt_zegrid_halo_invalid_value",
)


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


def _make_window(tmp_path, monkeypatch, config: dict, save_store: dict | None = None):
    import zemosaic.zemosaic_gui_qt as gui

    cfg_path = tmp_path / "zemosaic_config.json"
    merged = dict(config)

    def _save(data):
        if save_store is not None:
            save_store.clear()
            save_store.update(data)
        return None

    fake_cfg = SimpleNamespace(
        load_config=lambda: dict(merged),
        save_config=_save,
        get_config_path=lambda: cfg_path,
        DEFAULT_CONFIG={},
    )
    monkeypatch.setattr(gui, "zemosaic_config", fake_cfg)
    monkeypatch.setattr(gui, "_detect_analysis_backend", lambda: ("none", None, None))
    return gui.ZeMosaicQtMainWindow()


# ---------------------------------------------------------------------------
# A. Offscreen control structure
# ---------------------------------------------------------------------------

def test_a_combo_structure_uses_shared_contract_with_integer_data(tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch, {})
    combo = win._zegrid_halo_combo

    assert combo.count() == len(zhalo.ZEGRID_HALO_ALLOWED_PX) == 6
    assert tuple(combo.itemData(i) for i in range(combo.count())) == tuple(
        int(v) for v in zhalo.ZEGRID_HALO_ALLOWED_PX
    )
    assert all(isinstance(combo.itemData(i), int) for i in range(combo.count()))
    assert combo.isEditable() is False
    from PySide6.QtWidgets import QComboBox

    assert combo.insertPolicy() == QComboBox.NoInsert
    assert combo.currentData() == zhalo.ZEGRID_HALO_DEFAULT_PX == 64
    assert win.config["zegrid_halo_px"] == 64

    label = win._zegrid_halo_label
    assert label.text() == "Anti-checkerboard smoothing"
    assert label.buddy() is combo
    assert "output-canvas pixels" in combo.toolTip()
    assert "256 px is experimental" in combo.toolTip()
    assert "Experimental" in combo.itemText(combo.findData(256))
    assert win._config_fields["zegrid_halo_px"]["kind"] == "combobox"
    assert win._config_fields["zegrid_halo_px"]["type"] is int


# ---------------------------------------------------------------------------
# B. Mode / running matrix
# ---------------------------------------------------------------------------

def test_b_mode_and_running_matrix_preserves_value(tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch, {"zegrid_halo_px": 128})
    combo = win._zegrid_halo_combo

    # Classic (none checked in a fresh window) is disabled.
    assert win._selected_processing_mode() == "classic"
    assert combo.isEnabled() is False

    for mode in ("existing_master_tiles", "sds"):
        win._mode_checkboxes[mode].setChecked(True)
        assert win._selected_processing_mode() == mode
        assert combo.isEnabled() is False
        assert combo.currentData() == 128

    win._mode_checkboxes["zegrid"].setChecked(True)
    assert win._selected_processing_mode() == "zegrid"
    assert combo.isEnabled() is True
    assert combo.currentData() == 128

    # Running disables both the label and the combo, regardless of is_processing
    # bookkeeping order in the caller.
    win._set_processing_state(True)
    assert combo.isEnabled() is False
    assert win._zegrid_halo_label.isEnabled() is False
    win._set_processing_state(False)
    assert combo.isEnabled() is True
    assert combo.currentData() == 128

    # Stop/failure only re-enables when ZeGrid remains the effective mode.
    win._mode_checkboxes["sds"].setChecked(True)
    win._set_processing_state(False)
    assert combo.isEnabled() is False
    assert combo.currentData() == 128

    # Switching back preserves the selected integer item data.
    win._mode_checkboxes["zegrid"].setChecked(True)
    assert combo.isEnabled() is True
    assert combo.currentData() == 128
    assert win.config["zegrid_halo_px"] == 128


# ---------------------------------------------------------------------------
# C. Persistence / migration matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [8, 16, 32, 64, 128, 256])
def test_c_each_allowed_value_round_trips(tmp_path, monkeypatch, value):
    saved: dict = {}
    win = _make_window(tmp_path, monkeypatch, {"zegrid_halo_px": value}, saved)
    assert win._zegrid_halo_combo.currentData() == value
    assert win.config["zegrid_halo_px"] == value
    win._collect_config_from_widgets()
    win._save_config()
    assert saved["zegrid_halo_px"] == value
    assert type(saved["zegrid_halo_px"]) is int


def test_c_absent_key_becomes_64_and_legacy_string_is_normalized(tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch, {})
    assert win.config["zegrid_halo_px"] == 64
    assert win._zegrid_halo_combo.currentData() == 64

    win2 = _make_window(tmp_path, monkeypatch, {"zegrid_halo_px": "128"})
    assert win2.config["zegrid_halo_px"] == 128
    assert type(win2.config["zegrid_halo_px"]) is int
    assert win2._zegrid_halo_combo.currentData() == 128

    win3 = _make_window(tmp_path, monkeypatch, {"zegrid_halo_px": 8})
    assert win3.config["zegrid_halo_px"] == 8
    assert win3._zegrid_halo_combo.currentData() == 8


@pytest.mark.parametrize("bad", [True, False, 64.0, "64.0", "unknown", 0, 7, 512])
def test_c_invalid_values_are_surfaced_not_rounded(tmp_path, monkeypatch, bad):
    win = _make_window(
        tmp_path, monkeypatch, {"zegrid_halo_px": bad, "zegrid_mode_enabled": True}
    )
    combo = win._zegrid_halo_combo
    assert combo.currentIndex() == -1
    assert combo.currentData() is None
    assert win.config["zegrid_halo_px"] == bad
    assert combo.property("zegrid_halo_invalid") is True
    assert "Invalid ZeGrid anti-checkerboard smoothing value" in win.log_output.toPlainText()

    # Start is refused before any worker creation.
    import zemosaic.zemosaic_gui_qt as gui

    warnings: list[str] = []
    started: list[object] = []
    monkeypatch.setattr(gui, "run_hierarchical_mosaic_process", object())
    monkeypatch.setattr(gui.QMessageBox, "warning", lambda *a, **k: warnings.append(str(a[2])))
    monkeypatch.setattr(win, "_begin_async_worker_start", lambda *a, **k: started.append((a, k)))
    win._start_processing(skip_filter_prompt=True, predecided_skip_filter_ui=True)
    assert started == []
    assert warnings and "Invalid" in warnings[-1]

    # Selecting a valid item clears the invalid state and stores a plain int.
    combo.setCurrentIndex(combo.findData(128))
    assert win.config["zegrid_halo_px"] == 128
    assert type(win.config["zegrid_halo_px"]) is int
    assert combo.property("zegrid_halo_invalid") is False
    assert win._validate_zegrid_halo_for_start() is True


# ---------------------------------------------------------------------------
# D. Filter round-trip
# ---------------------------------------------------------------------------

def test_d_filter_round_trip_preserves_halo_and_has_no_duplicate_control(tmp_path, monkeypatch):
    win = _make_window(
        tmp_path,
        monkeypatch,
        {"zegrid_halo_px": 128, "zegrid_mode_enabled": True},
    )
    assert win._zegrid_halo_combo.currentData() == 128

    win._apply_filter_overrides_to_config({"zegrid_mode": True, "sds_mode": False})
    assert win.config["zegrid_halo_px"] == 128
    assert win._zegrid_halo_combo.currentData() == 128

    win._apply_filter_overrides_to_config({"sds_mode": True, "zegrid_mode": False})
    assert win.config["zegrid_halo_px"] == 128
    assert win._zegrid_halo_combo.currentData() == 128
    assert win._zegrid_halo_combo.isEnabled() is False

    import zemosaic.zemosaic_filter_gui_qt as filt

    dialog = filt.FilterQtDialog(
        str(tmp_path),
        {"zegrid_mode": True},
        config_overrides={"zegrid_mode_enabled": True, "zegrid_halo_px": 128},
    )
    try:
        overrides = dialog.overrides() or {}
        assert "zegrid_halo_px" not in overrides
        assert not hasattr(dialog, "_zegrid_halo_combo")
    finally:
        dialog.deleteLater()


# ---------------------------------------------------------------------------
# E. Localization
# ---------------------------------------------------------------------------

def test_e_all_locales_have_every_halo_key_and_fr_exact_label():
    for code in LOCALES:
        path = SRC / "zemosaic" / "locales" / f"{code}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in HALO_KEYS:
            assert key in data, f"{code}: missing {key}"
            assert isinstance(data[key], str) and data[key].strip()
        # Machine values remain integers in the widget, never translated text.
        assert data["qt_zegrid_halo_opt_8"].split()[0] != "8"  # display text only

    fr = json.loads((SRC / "zemosaic" / "locales" / "fr.json").read_text(encoding="utf-8"))
    assert fr["qt_field_zegrid_halo_label"] == "Lissage anti-damier"
    assert fr["qt_zegrid_halo_opt_64"].startswith("Recommandé")
    assert fr["qt_zegrid_halo_opt_256"] == "Expérimental"


def test_e_language_change_preserves_integer_item_data(tmp_path, monkeypatch):
    win = _make_window(
        tmp_path,
        monkeypatch,
        {"language": "en", "zegrid_halo_px": 32, "zegrid_mode_enabled": True},
    )
    combo = win._zegrid_halo_combo
    assert tuple(combo.itemData(i) for i in range(combo.count())) == tuple(
        zhalo.ZEGRID_HALO_ALLOWED_PX
    )
    assert combo.currentData() == 32

    win.localizer.set_language("fr")
    win._refresh_translated_ui()
    combo = win._zegrid_halo_combo
    assert win._zegrid_halo_label.text() == "Lissage anti-damier"
    assert tuple(combo.itemData(i) for i in range(combo.count())) == tuple(
        zhalo.ZEGRID_HALO_ALLOWED_PX
    )
    assert combo.currentData() == 32
    assert win.config["zegrid_halo_px"] == 32
    assert "128 px" in combo.itemText(combo.findData(128))


# ---------------------------------------------------------------------------
# F. GUI -> worker -> engine/output
# ---------------------------------------------------------------------------

def _write_smoke_corpus(root: Path) -> Path:
    from astropy.io import fits
    from astropy.wcs import WCS

    root.mkdir(parents=True, exist_ok=True)
    for i, (ra, dec) in enumerate(((10.0, 30.0), (10.005, 30.0))):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [ra, dec]
        wcs.wcs.crpix = [256.0, 256.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * 0.001
        wcs.array_shape = (512, 512)
        rng = np.random.default_rng(i)
        data = (100.0 + rng.normal(0.0, 0.5, (512, 512))).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header.update(wcs.to_header(relax=True))
        hdu.writeto(root / f"f{i:02d}.fits", overwrite=True)
    return root


def _fill_worker_required(sig, emitted: dict) -> dict:
    def _default(param):
        ann = str(param.annotation)
        if "bool" in ann:
            return False
        if "int" in ann:
            return 0
        if "float" in ann:
            return 0.0
        if "callable" in ann:
            return lambda *a, **k: None
        if "tuple" in ann:
            return (0.05, 0.05)
        return ""

    filled = {}
    for name, param in sig.parameters.items():
        if name in emitted:
            filled[name] = emitted[name]
        elif param.default is inspect._empty:
            filled[name] = _default(param)
    return filled


def test_f_gui_halo_reaches_worker_and_real_engine_output(tmp_path, monkeypatch):
    from astropy.io import fits

    import zemosaic.zemosaic_gui_qt as gui
    import zemosaic.zemosaic_worker as zw
    import zemosaic.zemosaic_zegrid_mode as zz

    inp = _write_smoke_corpus(tmp_path / "in")
    out = tmp_path / "out"
    base_config = {
        "input_dir": str(inp),
        "output_dir": str(out),
        "zegrid_mode_enabled": True,
        "zegrid_halo_px": 32,
        "zegrid_layout": "1x1",
        "cell_gauge_mode": "global",
        "export_aesthetic_fits": False,
        "final_mosaic_dbe_enabled": False,
        "save_final_as_uint16": False,
        "grid_rgb_equalize": False,
        "stack_ram_budget_gb": 0,
    }
    win = _make_window(tmp_path, monkeypatch, base_config)
    assert win._zegrid_halo_combo.currentData() == 32
    win._collect_config_from_widgets()
    _args, worker_kwargs = win._build_worker_invocation(skip_filter_ui=True)
    assert worker_kwargs["zegrid_halo_px"] == 32
    assert worker_kwargs["zegrid_mode_enabled"] is True

    # Real process wrapper: prove the key survives signature filtering.
    real_run = zw.run_hierarchical_mosaic
    captured: dict = {}

    def _capture_stub(**kwargs):
        captured.update(kwargs)

    _capture_stub.__signature__ = inspect.signature(real_run)  # type: ignore[attr-defined]
    monkeypatch.setattr(zw, "run_hierarchical_mosaic", _capture_stub)
    wrapper_kwargs = dict(worker_kwargs)
    wrapper_kwargs.pop("solver_settings_dict", None)
    zw.run_hierarchical_mosaic_process(
        SimpleNamespace(put=lambda *_: None),
        solver_settings_dict=None,
        crash_breadcrumbs_mode="off",
        **wrapper_kwargs,
    )
    assert captured["zegrid_halo_px"] == 32

    # Real worker dispatch -> real engine on the controlled corpus.
    monkeypatch.setattr(zw, "run_hierarchical_mosaic", real_run)
    full_kwargs = _fill_worker_required(inspect.signature(real_run), captured)
    full_kwargs.update(
        input_folder=str(inp),
        output_folder=str(out),
        progress_callback=lambda *a, **k: None,
        stack_norm_method="linear_fit",
        stack_weight_method="noise_variance",
        stack_reject_algo="winsorized_sigma_clip",
        stack_final_combine="mean",
        parsed_winsor_limits=(0.05, 0.05),
        stack_kappa_low=3.0,
        stack_kappa_high=3.0,
        zegrid_halo_px=32,
    )
    monkeypatch.setattr(
        zw,
        "zemosaic_config",
        SimpleNamespace(load_config=lambda: dict(base_config)),
    )
    monkeypatch.setattr(zw, "ZEMOSAIC_CONFIG_AVAILABLE", True)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 8 * 2 ** 30)

    real_run(**full_kwargs)

    manifest = json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    halo = manifest["zegrid_halo"]
    assert halo["effective_px"] == 32
    assert halo["source"] == "explicit_argument"
    assert halo["requested_px"] == 32
    header = fits.getheader(out / "mosaic_grid.fits")
    assert int(header["ZGHALO"]) == 32


# ---------------------------------------------------------------------------
# G. Generic propagation guard
# ---------------------------------------------------------------------------

def test_g_zegrid_halo_is_signature_accepted_and_not_dropped(tmp_path, monkeypatch):
    import zemosaic.zemosaic_worker as zw

    assert "zegrid_halo_px" in inspect.signature(zw.run_hierarchical_mosaic).parameters

    captured: dict = {}
    real_sig = inspect.signature(zw.run_hierarchical_mosaic)

    def _capture(**kwargs):
        captured.update(kwargs)

    _capture.__signature__ = real_sig  # type: ignore[attr-defined]
    monkeypatch.setattr(zw, "run_hierarchical_mosaic", _capture)
    zw.run_hierarchical_mosaic_process(
        SimpleNamespace(put=lambda *_: None),
        solver_settings_dict=None,
        crash_breadcrumbs_mode="off",
        input_dir="/in",
        output_dir="/out",
        zegrid_mode_enabled=True,
        zegrid_halo_px=128,
    )
    assert captured["zegrid_halo_px"] == 128
    assert "zegrid_halo_px" not in {
        k for k in captured if k.endswith("_config") and k != "zegrid_halo_px"
    }
