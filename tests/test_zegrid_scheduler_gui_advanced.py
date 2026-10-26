"""ZM-GUI-SCHEDULER-ADVANCED — selector moved to Advanced; ``active`` default.

Product decisions locked here (Tristan, after the bounded qualification run):

* the ZeGrid scheduler selector lives in the Advanced tab, not in Main; the
  anti-checkerboard halo stays in Main;
* "no explicit choice" means ``active`` (qualified on a synthetic 5,000-image
  corpus, one cell, bounded profile, GPU, 0 exclusions, 0 fallback,
  ``PASS_NO_FALLBACK`` in 2,501.8 s);
* an explicit ``legacy``/``observe`` value always wins, end to end, and is
  never silently migrated;
* an unknown value still fails visibly (``normalize_scheduler_mode`` never
  coerces).

All Qt tests run offscreen, following the patterns of the existing GUI tests
(``test_zegrid_r53c_gui_halo.py``, ``test_zegrid_r55e4_gui_gpu_toggle.py``).
The propagation test drives the REAL GUI invocation payload through the REAL
worker wrapper and ``run_hierarchical_mosaic`` ZeGrid branch into the engine
boundary (``run_zegrid_mode`` is captured, so no corpus and no GPU is needed).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"
for _path in (SRC, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from zemosaic import zemosaic_config  # noqa: E402
from zemosaic import zemosaic_worker as zw  # noqa: E402
from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import scheduler as zsched  # noqa: E402

# Reuse the exact minimal worker payload used by the GPU-toggle GUI test so the
# real worker/ZeGrid branch is reached without inventing a second contract.
from test_zegrid_r55e4_gui_gpu_toggle import (  # noqa: E402
    _RecordingQueue,
    _worker_required_kwargs,
)

LOCALES = ("en", "fr", "de", "es", "is", "nl", "pl")


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
        # Real product defaults: a fresh config must behave like production.
        DEFAULT_CONFIG=dict(zemosaic_config.DEFAULT_CONFIG),
    )
    monkeypatch.setattr(gui, "zemosaic_config", fake_cfg)
    monkeypatch.setattr(gui, "_detect_analysis_backend", lambda: ("none", None, None))
    return gui.ZeMosaicQtMainWindow()


def _tab_page(win, key: str):
    layout = win._tab_layouts[key]
    return layout.parentWidget()


def _ancestors(widget) -> list:
    chain = []
    current = widget
    while current is not None:
        chain.append(current)
        parent = current.parentWidget()
        if parent is current:
            break
        current = parent
    return chain


def _prepare_zegrid(win, tmp_path) -> None:
    win.config["input_dir"] = str(tmp_path / "in")
    win.config["output_dir"] = str(tmp_path / "out")
    win._mode_checkboxes["zegrid"].setChecked(True)
    assert win._selected_processing_mode() == "zegrid"


# ---------------------------------------------------------------------------
# 1. Fresh config -> active (both config default and GUI)
# ---------------------------------------------------------------------------

def test_fresh_config_uses_active_for_config_and_gui(tmp_path, monkeypatch):
    assert zemosaic_config.DEFAULT_CONFIG["zegrid_scheduler_mode"] == "active"

    win = _make_window(tmp_path, monkeypatch, {})
    combo = win._zegrid_scheduler_combo
    assert win.config["zegrid_scheduler_mode"] == "active"
    assert combo.currentData() == "active"
    assert combo.currentIndex() == combo.findData("active")

    binding = win._config_fields["zegrid_scheduler_mode"]
    assert binding["kind"] == "combobox"
    assert binding["widget"] is combo
    assert binding["type"] is str

    # The widget is the source of truth when collected, and it saves as active.
    win._collect_config_from_widgets()
    assert win.config["zegrid_scheduler_mode"] == "active"
    assert win._serialize_config_for_save()["zegrid_scheduler_mode"] == "active"
    # active is displayed with its qualification warning, never silently.
    assert win._zegrid_scheduler_warning.isHidden() is False


# ---------------------------------------------------------------------------
# 2. Explicit legacy/observe survives load -> widget -> save
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token", ["legacy", "observe"])
def test_explicit_token_is_preserved_end_to_end(tmp_path, monkeypatch, token):
    save_store: dict = {}
    win = _make_window(
        tmp_path, monkeypatch, {"zegrid_scheduler_mode": token}, save_store=save_store
    )

    assert win.config["zegrid_scheduler_mode"] == token
    assert win._zegrid_scheduler_combo.currentData() == token

    win._collect_config_from_widgets()
    assert win.config["zegrid_scheduler_mode"] == token

    snapshot = win._serialize_config_for_save()
    assert snapshot["zegrid_scheduler_mode"] == token
    win._save_config()
    assert save_store["zegrid_scheduler_mode"] == token
    # Reloading the persisted token from config keeps it verbatim: no coercion
    # towards the new default.
    assert win._sync_zegrid_scheduler_combo_from_config() is True
    assert win._zegrid_scheduler_combo.currentData() == token
    assert win.config["zegrid_scheduler_mode"] == token


# ---------------------------------------------------------------------------
# 3. Unknown values still fail explicitly
# ---------------------------------------------------------------------------

def test_unknown_scheduler_value_still_fails_explicitly(tmp_path, monkeypatch):
    for bad in ("", "auto", "ACTIVE", "OBSERVE", " active", "legacy ", True, 0, 1.0):
        with pytest.raises(zsched.InvalidSchedulerMode):
            zsched.normalize_scheduler_mode(bad)

    win = _make_window(tmp_path, monkeypatch, {"zegrid_scheduler_mode": "BOGUS"})
    assert win.config["zegrid_scheduler_mode"] == "BOGUS"
    assert win._zegrid_scheduler_combo.currentIndex() == -1
    assert win._zegrid_scheduler_combo.property("zegrid_scheduler_invalid") is True

    # The raw invalid value is preserved, never replaced by the default.
    win._collect_config_from_widgets()
    assert win.config["zegrid_scheduler_mode"] == "BOGUS"


# ---------------------------------------------------------------------------
# 4. config -> combo -> config round-trip after the move
# ---------------------------------------------------------------------------

def test_config_combo_config_round_trip_after_move(tmp_path, monkeypatch):
    save_store: dict = {}
    win = _make_window(tmp_path, monkeypatch, {}, save_store=save_store)

    for token in zsched.MODES:
        win._zegrid_scheduler_combo.setCurrentIndex(
            win._zegrid_scheduler_combo.findData(token)
        )
        assert win.config["zegrid_scheduler_mode"] == token

        win._collect_config_from_widgets()
        assert win.config["zegrid_scheduler_mode"] == token
        assert win._serialize_config_for_save()["zegrid_scheduler_mode"] == token

        assert win._sync_zegrid_scheduler_combo_from_config() is True
        assert win._zegrid_scheduler_combo.currentData() == token

        win._save_config()
        assert save_store["zegrid_scheduler_mode"] == token


# ---------------------------------------------------------------------------
# 5. The selected token reaches the engine call (GUI -> worker -> engine)
# ---------------------------------------------------------------------------

def _drive_to_engine(win, tmp_path, monkeypatch, token: str) -> dict:
    _prepare_zegrid(win, tmp_path)
    win._zegrid_scheduler_combo.setCurrentIndex(
        win._zegrid_scheduler_combo.findData(token)
    )
    win._collect_config_from_widgets()
    assert win.config["zegrid_scheduler_mode"] == token
    win._save_config()

    _args, worker_kwargs = win._build_worker_invocation(skip_filter_ui=True)
    assert worker_kwargs["zegrid_scheduler_mode"] == token

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(zz, "run_zegrid_mode", _capture)

    required = _worker_required_kwargs(str(tmp_path / "in"), str(tmp_path / "out"))
    required.update(worker_kwargs)
    required["filter_overrides"] = {"processing_mode": "zegrid"}
    required["crash_breadcrumbs_mode"] = "off"
    required.pop("solver_settings_dict", None)

    zw.run_hierarchical_mosaic_process(
        _RecordingQueue(), solver_settings_dict=None, **required
    )
    assert captured, "ZeGrid engine was not reached"
    return captured


def test_selected_mode_reaches_the_engine_call(tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch, {"zegrid_scheduler_mode": "legacy"})
    assert _drive_to_engine(win, tmp_path, monkeypatch, "active")[
        "zegrid_scheduler_mode"
    ] == "active"

    # An explicit legacy choice still wins after the default change.
    win._zegrid_scheduler_combo.setCurrentIndex(
        win._zegrid_scheduler_combo.findData("legacy")
    )
    assert win.config["zegrid_scheduler_mode"] == "legacy"
    assert _drive_to_engine(win, tmp_path, monkeypatch, "legacy")[
        "zegrid_scheduler_mode"
    ] == "legacy"


# ---------------------------------------------------------------------------
# 6. Structural placement: Advanced, not Main; halo stays in Main
# ---------------------------------------------------------------------------

def test_selector_is_in_advanced_tab_and_halo_stays_in_main(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QComboBox

    win = _make_window(tmp_path, monkeypatch, {})
    main_tab = _tab_page(win, "main")
    advanced_tab = _tab_page(win, "advanced")
    assert main_tab is not None and advanced_tab is not None
    assert main_tab is not advanced_tab

    scheduler_chain = _ancestors(win._zegrid_scheduler_combo)
    assert advanced_tab in scheduler_chain
    assert main_tab not in scheduler_chain

    halo_chain = _ancestors(win._zegrid_halo_combo)
    assert main_tab in halo_chain
    assert advanced_tab not in halo_chain

    # Exactly one scheduler widget exists: the move must not duplicate it.
    scheduler_combos = [
        combo
        for combo in win.findChildren(QComboBox)
        if tuple(combo.itemData(i) for i in range(combo.count())) == zsched.MODES
    ]
    assert scheduler_combos == [win._zegrid_scheduler_combo]


# ---------------------------------------------------------------------------
# 7. Locales: group key present, qualification wording, no stale claim
# ---------------------------------------------------------------------------

def test_locales_carry_the_qualification_wording():
    for lang in LOCALES:
        path = REPO_ROOT / "src" / "zemosaic" / "locales" / f"{lang}.json"
        data = json.loads(path.read_text(encoding="utf-8"))

        assert "qt_group_zegrid_scheduler" in data, lang
        warning = data["qt_zegrid_scheduler_active_warning"]
        assert "5 000" in warning or "5,000" in warning or "5.000" in warning, lang
        assert "NOT qualified" not in warning, lang
        assert "Windows/RTX" not in warning, lang
        assert "gpu" in warning.lower(), lang
        # The no-promise clause is localized, so accept each locale's wording.
        assert any(
            word in warning.lower()
            for word in (
                "performance",
                "leistung",
                "rendimiento",
                "afkasta",
                "prestatie",
                "wydajno",
            )
        ), lang

        tip = data["qt_field_zegrid_scheduler_tip"]
        assert "NOT qualified" not in tip, lang
        assert "5 000" in tip or "5,000" in tip or "5.000" in tip, lang


def test_widget_texts_follow_the_selected_locale(tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch, {"language": "fr"})
    assert win._zegrid_scheduler_combo.parentWidget() is not None
    group = win._zegrid_scheduler_combo.parentWidget()
    assert group.title() == "Ordonnanceur ZeGrid"
    assert "5 000" in win._zegrid_scheduler_warning.text()
