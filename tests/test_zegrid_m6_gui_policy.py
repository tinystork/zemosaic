"""ZM-ZEGRID-M6-QT-POLICY — six editable ZeGrid policy keys in Qt.

The Advanced controls must load/edit/save/reload the EXACT declared values
(``False`` / ``None`` / ``"keep"`` are all meaningful) and the real GUI
invocation must carry them through the real worker/dispatcher into the engine
``zconfig`` boundary. The common validator stays the authority: unknown values
are refused visibly, never turned into a default.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"
for _path in (SRC, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from zemosaic import zemosaic_config as zc  # noqa: E402
from zemosaic import zemosaic_worker as zw  # noqa: E402
from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402

from test_zegrid_scheduler_gui_advanced import (  # noqa: E402
    _make_window,
    _prepare_zegrid,
)
from test_zegrid_r55e4_gui_gpu_toggle import (  # noqa: E402
    _RecordingQueue,
    _worker_required_kwargs,
)

POLICY_KEYS = list(zc.ZEGRID_RUN_POLICY_KEYS)
DEFAULTS = {
    "zegrid_resource_recovery_enabled": False,
    "zegrid_resource_recovery_deadline_s": None,
    "zegrid_durable_store_budget_bytes": None,
    "zegrid_cell_gauge_mode": None,
    "zegrid_layout": None,
    "zegrid_sip_mode": "keep",
}
EXPLICIT = {
    "zegrid_resource_recovery_enabled": True,
    "zegrid_resource_recovery_deadline_s": 12.5,
    "zegrid_durable_store_budget_bytes": 6442450944,  # exactly 6 GiB
    "zegrid_cell_gauge_mode": "per_cell",
    "zegrid_layout": "3x2",
    "zegrid_sip_mode": "strip",
}
# Deliberately DIFFERENT from EXPLICIT: if the worker silently reloads the
# persisted/global config instead of honouring the explicit transport, the
# propagation test fails on these values.
DISK_VALUES = {
    "zegrid_resource_recovery_enabled": False,
    "zegrid_resource_recovery_deadline_s": 99.0,
    "zegrid_durable_store_budget_bytes": 1024,
    "zegrid_cell_gauge_mode": "global",
    "zegrid_layout": "1x1",
    "zegrid_sip_mode": "keep",
}

CORPUS_TOOL = REPO_ROOT / "tools" / "zegrid_m6" / "make_tiny_corpus.py"


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


def _no_warning(monkeypatch):
    import zemosaic.zemosaic_gui_qt as gui

    calls: list = []
    monkeypatch.setattr(
        gui.QMessageBox,
        "warning",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    return calls


def _edit_text(win, key: str, value) -> None:
    """Simulate a DELIBERATE user edit (setText alone is not an edit)."""
    edit = {
        "zegrid_resource_recovery_deadline_s": win._zegrid_recovery_deadline_edit,
        "zegrid_durable_store_budget_bytes": win._zegrid_store_budget_edit,
        "zegrid_layout": win._zegrid_layout_edit,
    }[key]
    edit.setText("" if value is None else str(value))
    win._on_zegrid_policy_text_edited(key, user_edit=True)


def _set_widgets(win, values: dict) -> None:
    if "zegrid_resource_recovery_enabled" in values:
        win._zegrid_recovery_checkbox.setChecked(
            bool(values["zegrid_resource_recovery_enabled"])
        )
    if "zegrid_resource_recovery_deadline_s" in values:
        _edit_text(
            win,
            "zegrid_resource_recovery_deadline_s",
            values["zegrid_resource_recovery_deadline_s"],
        )
    if "zegrid_durable_store_budget_bytes" in values:
        _edit_text(
            win,
            "zegrid_durable_store_budget_bytes",
            values["zegrid_durable_store_budget_bytes"],
        )
    if "zegrid_cell_gauge_mode" in values:
        combo = win._zegrid_cell_gauge_combo
        raw = values["zegrid_cell_gauge_mode"]
        index = 0 if raw is None else win._zegrid_combo_index_for_data(combo, raw)
        combo.setCurrentIndex(index)
    if "zegrid_layout" in values:
        _edit_text(win, "zegrid_layout", values["zegrid_layout"])
    if "zegrid_sip_mode" in values:
        combo = win._zegrid_sip_combo
        index = win._zegrid_combo_index_for_data(
            combo, values["zegrid_sip_mode"]
        )
        combo.setCurrentIndex(index)


# ---------------------------------------------------------------------------
# 1. Defaults are explicitly represented, not implicitly coerced
# ---------------------------------------------------------------------------

def test_defaults_are_represented_and_round_trip(tmp_path, monkeypatch):
    save_store: dict = {}
    win = _make_window(tmp_path, monkeypatch, {}, save_store=save_store)
    assert win._zegrid_recovery_checkbox.isChecked() is False
    assert win._zegrid_recovery_deadline_edit.text() == ""
    assert win._zegrid_store_budget_edit.text() == ""
    assert win._zegrid_layout_edit.text() == ""
    gauge = win._zegrid_cell_gauge_combo
    assert gauge.currentIndex() == 0
    assert gauge.itemData(0) is None  # auto/default sentinel lives in itemData
    assert win._zegrid_sip_combo.currentData() == "keep"
    assert win._zegrid_policy_error_label.isHidden() is True

    win._collect_config_from_widgets()
    for key, expected in DEFAULTS.items():
        assert win.config[key] == expected, key
    snapshot = win._serialize_config_for_save()
    for key, expected in DEFAULTS.items():
        assert snapshot[key] == expected, key

    # Disabled outside ZeGrid (values untouched), enabled in ZeGrid.
    assert win._zegrid_recovery_checkbox.isEnabled() is False
    win._mode_checkboxes["zegrid"].setChecked(True)
    assert win._zegrid_recovery_checkbox.isEnabled() is True
    # Recovery OFF keeps the deadline control disabled; the stored value is
    # still None and nothing is erased.
    assert win._zegrid_recovery_deadline_edit.isEnabled() is False
    assert win.config["zegrid_resource_recovery_deadline_s"] is None


# ---------------------------------------------------------------------------
# 2. Explicit values survive load -> widgets -> save -> reload
# ---------------------------------------------------------------------------

def test_explicit_values_round_trip_through_save_and_reload(tmp_path, monkeypatch):
    save_store: dict = {}
    win = _make_window(tmp_path, monkeypatch, dict(EXPLICIT), save_store=save_store)
    assert win._zegrid_recovery_checkbox.isChecked() is True
    assert win._zegrid_recovery_deadline_edit.text() == "12.5"
    assert win._zegrid_store_budget_edit.text() == "6442450944"
    assert win._zegrid_cell_gauge_combo.currentData() == "per_cell"
    assert win._zegrid_layout_edit.text() == "3x2"
    assert win._zegrid_sip_combo.currentData() == "strip"

    win._collect_config_from_widgets()
    for key, expected in EXPLICIT.items():
        assert win.config[key] == expected, key
    win._save_config()
    for key, expected in EXPLICIT.items():
        assert save_store[key] == expected, key

    reloaded = _make_window(tmp_path, monkeypatch, dict(save_store))
    assert reloaded._zegrid_recovery_checkbox.isChecked() is True
    assert reloaded._zegrid_recovery_deadline_edit.text() == "12.5"
    assert reloaded._zegrid_store_budget_edit.text() == "6442450944"
    assert reloaded._zegrid_cell_gauge_combo.currentData() == "per_cell"
    assert reloaded._zegrid_layout_edit.text() == "3x2"
    assert reloaded._zegrid_sip_combo.currentData() == "strip"
    reloaded._collect_config_from_widgets()
    for key, expected in EXPLICIT.items():
        assert reloaded.config[key] == expected, key


# ---------------------------------------------------------------------------
# 3. Null clearing is safe and typed
# ---------------------------------------------------------------------------

def test_null_clearing_round_trips_as_none(tmp_path, monkeypatch):
    save_store: dict = {}
    win = _make_window(tmp_path, monkeypatch, dict(EXPLICIT), save_store=save_store)
    _set_widgets(
        win,
        {
            "zegrid_resource_recovery_enabled": False,
            "zegrid_resource_recovery_deadline_s": None,
            "zegrid_durable_store_budget_bytes": None,
            "zegrid_cell_gauge_mode": None,
            "zegrid_layout": None,
        },
    )
    win._collect_config_from_widgets()
    assert win.config["zegrid_resource_recovery_enabled"] is False
    assert win.config["zegrid_resource_recovery_deadline_s"] is None
    assert win.config["zegrid_durable_store_budget_bytes"] is None
    assert win.config["zegrid_cell_gauge_mode"] is None
    assert win.config["zegrid_layout"] is None
    # SIP has no null UI: its documented default is the literal "keep".
    assert win.config["zegrid_sip_mode"] == "strip"
    win._save_config()
    assert save_store["zegrid_resource_recovery_deadline_s"] is None
    assert save_store["zegrid_durable_store_budget_bytes"] is None
    assert save_store["zegrid_cell_gauge_mode"] is None
    assert save_store["zegrid_layout"] is None


# ---------------------------------------------------------------------------
# 4. Multi-GiB budget and fractional deadline are exact
# ---------------------------------------------------------------------------

def test_multi_gib_budget_is_exact_and_not_int32_limited(tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch, {})
    for raw, expected in (
        ("6442450944", 6442450944),
        ("17179869184", 17179869184),  # 16 GiB, beyond int32
        ("5368709120", 5368709120),  # 5 GiB
    ):
        _edit_text(win, "zegrid_durable_store_budget_bytes", raw)
        win._collect_config_from_widgets()
        assert win.config["zegrid_durable_store_budget_bytes"] == expected
        assert isinstance(win.config["zegrid_durable_store_budget_bytes"], int)


def test_fractional_deadline_is_exact(tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch, {})
    for raw, expected in (("12.5", 12.5), ("0.25", 0.25), ("90", 90.0)):
        _edit_text(win, "zegrid_resource_recovery_deadline_s", raw)
        win._collect_config_from_widgets()
        assert win.config["zegrid_resource_recovery_deadline_s"] == expected


# ---------------------------------------------------------------------------
# 4b. Gauge: None sentinel and explicit "auto" are distinct
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,index,data",
    [
        (None, 0, None),
        ("auto", 1, "auto"),
        ("global", 2, "global"),
        ("per_cell", 3, "per_cell"),
    ],
)
def test_gauge_sentinel_and_explicit_auto_are_distinct(
    tmp_path, monkeypatch, raw, index, data
):
    save_store: dict = {}
    win = _make_window(
        tmp_path, monkeypatch, {"zegrid_cell_gauge_mode": raw}, save_store=save_store
    )
    combo = win._zegrid_cell_gauge_combo
    assert combo.currentIndex() == index
    assert combo.currentData() == data
    assert combo.property("zegrid_policy_invalid") is False
    assert win._zegrid_policy_error_label.isHidden() is True
    _prepare_zegrid(win, tmp_path)
    win._collect_config_from_widgets()
    assert win.config["zegrid_cell_gauge_mode"] == raw
    assert win._validate_zegrid_policy_for_start() is True
    win._save_config()
    assert save_store["zegrid_cell_gauge_mode"] == raw

    reloaded = _make_window(tmp_path, monkeypatch, dict(save_store))
    assert reloaded._zegrid_cell_gauge_combo.currentData() == data
    reloaded._collect_config_from_widgets()
    assert reloaded.config["zegrid_cell_gauge_mode"] == raw


# ---------------------------------------------------------------------------
# 4c. Persisted numeric strings are NOT repaired by passive actions
# ---------------------------------------------------------------------------

def test_persisted_numeric_strings_are_not_repaired_by_passive_actions(
    tmp_path, monkeypatch
):
    _no_warning(monkeypatch)
    save_store: dict = {}
    win = _make_window(
        tmp_path,
        monkeypatch,
        {
            "zegrid_resource_recovery_deadline_s": "12.5",
            "zegrid_durable_store_budget_bytes": "6442450944",
        },
        save_store=save_store,
    )
    assert win._zegrid_policy_error_label.isHidden() is False

    _prepare_zegrid(win, tmp_path)
    win._collect_config_from_widgets()
    assert win.config["zegrid_resource_recovery_deadline_s"] == "12.5"
    assert type(win.config["zegrid_resource_recovery_deadline_s"]) is str
    assert win.config["zegrid_durable_store_budget_bytes"] == "6442450944"
    assert type(win.config["zegrid_durable_store_budget_bytes"]) is str

    win._save_config()
    assert save_store["zegrid_resource_recovery_deadline_s"] == "12.5"
    assert save_store["zegrid_durable_store_budget_bytes"] == "6442450944"

    # Mode round-trip must not repair them.
    win._mode_checkboxes["zegrid"].setChecked(False)
    win._mode_checkboxes["zegrid"].setChecked(True)
    assert win.config["zegrid_resource_recovery_deadline_s"] == "12.5"
    assert win.config["zegrid_durable_store_budget_bytes"] == "6442450944"

    # Language rebuild must not repair them.
    win._apply_language_selection("fr")
    assert win.config["zegrid_resource_recovery_deadline_s"] == "12.5"
    assert win.config["zegrid_durable_store_budget_bytes"] == "6442450944"
    assert win._zegrid_recovery_deadline_edit.text() == "12.5"
    assert win._zegrid_store_budget_edit.text() == "6442450944"
    assert win._zegrid_policy_error_label.isHidden() is False

    # Building the invocation must not repair them.
    _args, worker_kwargs = win._build_worker_invocation(skip_filter_ui=True)
    assert worker_kwargs["zegrid_resource_recovery_deadline_s"] == "12.5"
    assert worker_kwargs["zegrid_durable_store_budget_bytes"] == "6442450944"
    assert win._validate_zegrid_policy_for_start() is False

    # A deliberate edit into valid typed values is allowed and clears the block.
    _edit_text(win, "zegrid_resource_recovery_deadline_s", "15.5")
    _edit_text(win, "zegrid_durable_store_budget_bytes", "6442450945")
    win._collect_config_from_widgets()
    assert win.config["zegrid_resource_recovery_deadline_s"] == 15.5
    assert type(win.config["zegrid_resource_recovery_deadline_s"]) is float
    assert win.config["zegrid_durable_store_budget_bytes"] == 6442450945
    assert type(win.config["zegrid_durable_store_budget_bytes"]) is int
    assert win._zegrid_policy_error_label.isHidden() is True
    assert win._validate_zegrid_policy_for_start() is True


@pytest.mark.parametrize(
    "key,raw",
    [
        ("zegrid_resource_recovery_deadline_s", "  "),
        ("zegrid_resource_recovery_deadline_s", ""),
        ("zegrid_resource_recovery_deadline_s", True),
        ("zegrid_resource_recovery_deadline_s", "abc"),
        ("zegrid_durable_store_budget_bytes", "  "),
        ("zegrid_durable_store_budget_bytes", ""),
        ("zegrid_durable_store_budget_bytes", True),
        ("zegrid_durable_store_budget_bytes", "abc"),
    ],
)
def test_persisted_non_numeric_types_stay_raw_and_block(
    tmp_path, monkeypatch, key, raw
):
    calls = _no_warning(monkeypatch)
    win = _make_window(tmp_path, monkeypatch, {key: raw})
    _prepare_zegrid(win, tmp_path)
    win._collect_config_from_widgets()
    assert win.config[key] == raw
    assert type(win.config[key]) is type(raw)
    assert win._zegrid_policy_error_label.isHidden() is False
    assert win._validate_zegrid_policy_for_start() is False
    assert calls


def test_real_strict_loader_recovery_qtest_edit_and_save_reload(
    tmp_path, monkeypatch
):
    """Persistent C1 regression: real loader refusal -> real recovery.

    Ported from Junior's external probe. It exercises the actual
    ``zc.load_config`` failure, the GUI's optional-path sanitised recovery, the
    real ``textEdited`` signal via ``QTest.keyClicks``, a language rebuild, and
    a real temp-file save/load. It never writes the user's real config.
    """
    from PySide6.QtTest import QTest

    import zemosaic.zemosaic_gui_qt as gui

    cfg_path = tmp_path / "zemosaic_config.json"
    raw = {
        "preferred_gui_backend": "tk",
        "stack_use_gpu": False,
        "zegrid_cell_gauge_mode": "auto",
        "zegrid_resource_recovery_enabled": True,
        "zegrid_resource_recovery_deadline_s": "12.5",
        "zegrid_durable_store_budget_bytes": "6442450944",
    }
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")
    original = cfg_path.read_bytes()

    monkeypatch.setattr(zc, "get_config_path", lambda: cfg_path)
    monkeypatch.setattr(gui, "zemosaic_config", zc)
    monkeypatch.setattr(gui, "_detect_analysis_backend", lambda: ("none", None, None))
    monkeypatch.setattr(gui.QMessageBox, "warning", lambda *args, **kwargs: None)

    # Track ONLY the temp files created by the recovery helper, so cleanup is
    # asserted on our own paths rather than by scanning/deleting tmp_path.
    created_paths: list[Path] = []
    real_mkstemp = gui.tempfile.mkstemp

    def _tracking_mkstemp(*args, **kwargs):
        handle, path = real_mkstemp(*args, **kwargs)
        created_paths.append(Path(path))
        return handle, path

    monkeypatch.setattr(gui.tempfile, "mkstemp", _tracking_mkstemp)

    # The REAL strict loader must refuse the persisted numeric strings.
    with pytest.raises(ValueError):
        zc.load_config()

    win = gui.ZeMosaicQtMainWindow()
    try:
        _prepare_zegrid(win, tmp_path)
        win._collect_config_from_widgets()

        # Recovery ran through the real optional-path loader; the unrelated
        # migrations are still active.
        assert created_paths, "recovery did not create a sanitised temp file"
        assert not any(path.exists() for path in created_paths)
        assert win.config["preferred_gui_backend"] == "qt"  # legacy tk -> qt
        assert win.config["use_gpu_phase5"] is False  # explicit legacy GPU false
        assert win.config["zegrid_cell_gauge_mode"] == "auto"
        assert win._zegrid_cell_gauge_combo.currentData() == "auto"
        assert win.config["zegrid_resource_recovery_deadline_s"] == "12.5"
        assert type(win.config["zegrid_resource_recovery_deadline_s"]) is str
        assert win.config["zegrid_durable_store_budget_bytes"] == "6442450944"
        assert type(win.config["zegrid_durable_store_budget_bytes"]) is str
        assert win._zegrid_policy_error_label.isHidden() is False
        assert win._validate_zegrid_policy_for_start() is False

        # No auto-save happened: the source file is byte-identical before a
        # deliberate edit.
        assert cfg_path.read_bytes() == original

        # Real QTest typing exercises textEdited (not a manual handler call).
        deadline = win._zegrid_recovery_deadline_edit
        deadline.selectAll()
        QTest.keyClicks(deadline, "15.5")
        budget = win._zegrid_store_budget_edit
        budget.selectAll()
        QTest.keyClicks(budget, "6442450945")
        win._collect_config_from_widgets()
        assert win.config["zegrid_resource_recovery_deadline_s"] == 15.5
        assert type(win.config["zegrid_resource_recovery_deadline_s"]) is float
        assert win.config["zegrid_durable_store_budget_bytes"] == 6442450945
        assert type(win.config["zegrid_durable_store_budget_bytes"]) is int
        assert win._zegrid_policy_error_label.isHidden() is True
        assert win._validate_zegrid_policy_for_start() is True

        # Language rebuild preserves the corrected types and explicit auto.
        win._apply_language_selection("fr")
        assert win.config["zegrid_resource_recovery_deadline_s"] == 15.5
        assert win.config["zegrid_durable_store_budget_bytes"] == 6442450945
        assert win._zegrid_cell_gauge_combo.currentData() == "auto"
        assert win._zegrid_recovery_deadline_edit.text() == "15.5"

        # REAL temp-file save/load (isolated path) preserves the corrected
        # types and the distinct explicit auto value.
        win._save_config()
        loaded = zc.load_config()
        assert loaded["zegrid_resource_recovery_deadline_s"] == 15.5
        assert type(loaded["zegrid_resource_recovery_deadline_s"]) is float
        assert loaded["zegrid_durable_store_budget_bytes"] == 6442450945
        assert type(loaded["zegrid_durable_store_budget_bytes"]) is int
        assert loaded["zegrid_cell_gauge_mode"] == "auto"
    finally:
        # Do NOT call close(): closeEvent auto-saves. deleteLater only.
        win.deleteLater()


# ---------------------------------------------------------------------------
# 5. Adversarial values are visible and block only the ZeGrid mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "values",
    [
        {"zegrid_resource_recovery_deadline_s": "abc"},
        {"zegrid_resource_recovery_deadline_s": "0"},
        {"zegrid_resource_recovery_deadline_s": "-2.5"},
        {"zegrid_durable_store_budget_bytes": "12.5"},
        {"zegrid_durable_store_budget_bytes": "0"},
        {"zegrid_durable_store_budget_bytes": "abc"},
        {"zegrid_layout": "abc"},
        {"zegrid_layout": "2x0"},
        {"zegrid_layout": "2x2x2"},
        {"zegrid_layout": 2},
        {"zegrid_layout": True},
        {"zegrid_cell_gauge_mode": "per-cell"},
        {"zegrid_cell_gauge_mode": True},
        {"zegrid_sip_mode": "stripped"},
        {"zegrid_sip_mode": True},
        {"zegrid_resource_recovery_enabled": "yes"},
    ],
)
def test_invalid_values_block_zegrid_but_not_classic(
    tmp_path, monkeypatch, values
):
    calls = _no_warning(monkeypatch)
    win = _make_window(tmp_path, monkeypatch, dict(values))
    _prepare_zegrid(win, tmp_path)
    win._collect_config_from_widgets()

    # The invalid persisted value is not silently repaired to a valid typed
    # value or a default: its raw type/value is preserved until a user edit.
    for key, raw in values.items():
        assert win.config[key] == raw, key
        assert type(win.config[key]) is type(raw), key
    if "zegrid_cell_gauge_mode" in values:
        assert win._zegrid_cell_gauge_combo.currentIndex() == -1
        assert (
            win._zegrid_cell_gauge_combo.property("zegrid_policy_invalid") is True
        )
    if "zegrid_sip_mode" in values:
        assert win._zegrid_sip_combo.currentIndex() == -1
        assert win._zegrid_sip_combo.property("zegrid_policy_invalid") is True
    if "zegrid_resource_recovery_enabled" in values:
        assert (
            win._zegrid_recovery_checkbox.property("zegrid_policy_invalid") is True
        )
    assert win._zegrid_policy_error_label.isHidden() is False

    assert win._validate_zegrid_policy_for_start() is False
    assert calls, "the user must see a blocking warning"
    assert "Advanced" in str(calls[0][0][2]) or "Avanc" in str(calls[0][0][2])

    # Classic is unaffected: the same invalid value does not block a non-ZeGrid
    # run (it is still logged once).
    win._mode_checkboxes["zegrid"].setChecked(False)
    assert win._selected_processing_mode() == "classic"
    assert win._validate_zegrid_policy_for_start() is True


# ---------------------------------------------------------------------------
# 6. Mode switching / running disables controls without erasing values
# ---------------------------------------------------------------------------

def test_mode_switch_and_running_preserve_persisted_values(tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch, dict(EXPLICIT))
    win._mode_checkboxes["zegrid"].setChecked(True)
    assert win._zegrid_store_budget_edit.isEnabled() is True
    assert win._zegrid_recovery_deadline_edit.isEnabled() is True

    win._mode_checkboxes["zegrid"].setChecked(False)
    assert win._zegrid_store_budget_edit.isEnabled() is False
    assert win._zegrid_recovery_checkbox.isEnabled() is False
    assert win._zegrid_recovery_deadline_edit.isEnabled() is False
    # Nothing was erased.
    assert win._zegrid_store_budget_edit.text() == "6442450944"
    assert win._zegrid_layout_edit.text() == "3x2"
    win._collect_config_from_widgets()
    for key, expected in EXPLICIT.items():
        assert win.config[key] == expected, key

    win._mode_checkboxes["zegrid"].setChecked(True)
    assert win._zegrid_store_budget_edit.isEnabled() is True
    win.is_processing = True
    win._apply_zegrid_halo_control_state(running=True)
    assert win._zegrid_store_budget_edit.isEnabled() is False
    assert win._zegrid_recovery_deadline_edit.isEnabled() is False
    assert win._zegrid_store_budget_edit.text() == "6442450944"
    win.is_processing = False


def test_recovery_off_disables_only_the_deadline(tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch, dict(EXPLICIT))
    win._mode_checkboxes["zegrid"].setChecked(True)
    assert win._zegrid_recovery_deadline_edit.isEnabled() is True
    win._zegrid_recovery_checkbox.setChecked(False)
    assert win._zegrid_recovery_deadline_edit.isEnabled() is False
    assert win._zegrid_recovery_deadline_edit.text() == "12.5"
    assert win._zegrid_recovery_deadline_edit.isEnabled() is False
    assert win._zegrid_store_budget_edit.isEnabled() is True
    # The persisted deadline is still the valid value.
    win._collect_config_from_widgets()
    assert win.config["zegrid_resource_recovery_deadline_s"] == 12.5
    win._zegrid_recovery_checkbox.setChecked(True)
    assert win._zegrid_recovery_deadline_edit.isEnabled() is True


# ---------------------------------------------------------------------------
# 6b. Guard rails required by Junior r0: actual binding + None at engine
# ---------------------------------------------------------------------------

def test_built_invocation_binds_the_real_worker_signature(tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch, dict(EXPLICIT))
    worker_kwargs = _worker_kwargs_from_gui(win, tmp_path)

    signature = inspect.signature(zw.run_hierarchical_mosaic)
    bindable = {
        key: value
        for key, value in worker_kwargs.items()
        if key in signature.parameters
    }
    bound = signature.bind_partial(**bindable)
    for key, expected in EXPLICIT.items():
        assert key in bound.arguments, key
        assert bound.arguments[key] == expected, key

    # The process entry point accepts the full built payload via **kwargs and
    # filters it against the real signature above.
    process_params = inspect.signature(zw.run_hierarchical_mosaic_process).parameters
    assert any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in process_params.values()
    )


def test_none_clearing_reaches_engine_over_non_null_disk(tmp_path, monkeypatch):
    null_values = {
        "zegrid_resource_recovery_enabled": False,
        "zegrid_resource_recovery_deadline_s": None,
        "zegrid_durable_store_budget_bytes": None,
        "zegrid_cell_gauge_mode": None,
        "zegrid_layout": None,
        "zegrid_sip_mode": "keep",
    }
    non_null_disk = {
        "zegrid_resource_recovery_enabled": True,
        "zegrid_resource_recovery_deadline_s": 99.0,
        "zegrid_durable_store_budget_bytes": 6442450944,
        "zegrid_cell_gauge_mode": "per_cell",
        "zegrid_layout": "3x2",
        "zegrid_sip_mode": "strip",
    }
    win = _make_window(tmp_path, monkeypatch, dict(null_values))
    worker_kwargs = _worker_kwargs_from_gui(win, tmp_path)
    assert worker_kwargs["zegrid_resource_recovery_enabled"] is False
    assert worker_kwargs["zegrid_resource_recovery_deadline_s"] is None
    assert worker_kwargs["zegrid_durable_store_budget_bytes"] is None
    assert worker_kwargs["zegrid_cell_gauge_mode"] is None
    assert worker_kwargs["zegrid_layout"] is None
    assert worker_kwargs["zegrid_sip_mode"] == "keep"

    captured = _drive(
        monkeypatch, tmp_path, worker_kwargs, _disk_config(**non_null_disk)
    )
    zconfig = captured["zconfig"]
    assert zconfig.zegrid_resource_recovery_enabled is False
    assert zconfig.zegrid_resource_recovery_deadline_s is None
    assert zconfig.zegrid_durable_store_budget_bytes is None
    assert zconfig.zegrid_cell_gauge_mode is None
    assert zconfig.zegrid_layout is None
    assert zconfig.zegrid_sip_mode == "keep"


@pytest.mark.parametrize("gauge", [None, "auto", "global", "per_cell"])
def test_every_gauge_value_reaches_engine_exactly(
    tmp_path, monkeypatch, gauge
):
    win = _make_window(tmp_path, monkeypatch, {"zegrid_cell_gauge_mode": gauge})
    worker_kwargs = _worker_kwargs_from_gui(win, tmp_path)
    assert worker_kwargs["zegrid_cell_gauge_mode"] == gauge
    assert type(worker_kwargs["zegrid_cell_gauge_mode"]) is type(gauge)
    disk_gauge = "global" if gauge != "global" else "per_cell"
    captured = _drive(
        monkeypatch,
        tmp_path,
        worker_kwargs,
        _disk_config(zegrid_cell_gauge_mode=disk_gauge),
    )
    assert captured["zconfig"].zegrid_cell_gauge_mode == gauge
    assert type(captured["zconfig"].zegrid_cell_gauge_mode) is type(gauge)


def test_process_dispatcher_passes_policy_values_to_worker_function(
    tmp_path, monkeypatch
):
    win = _make_window(tmp_path, monkeypatch, dict(EXPLICIT))
    worker_kwargs = _worker_kwargs_from_gui(win, tmp_path)

    captured: dict = {}
    real_worker = zw.run_hierarchical_mosaic

    @functools.wraps(real_worker)
    def _capture(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(zw, "run_hierarchical_mosaic", _capture)
    required = _worker_required_kwargs(str(tmp_path / "in"), str(tmp_path / "out"))
    required.update(worker_kwargs)
    required["filter_overrides"] = {"processing_mode": "zegrid"}
    required["crash_breadcrumbs_mode"] = "off"
    required.pop("solver_settings_dict", None)
    zw.run_hierarchical_mosaic_process(
        _RecordingQueue(), solver_settings_dict=None, **required
    )
    assert captured, "the real dispatcher did not call the worker function"
    # Strong signature contract: bind the ACTUAL downstream kwargs against the
    # unmodified worker signature, without prefiltering the assertion.
    bound = inspect.signature(real_worker).bind_partial(**captured)
    for key, expected in EXPLICIT.items():
        assert bound.arguments[key] == expected, key


# ---------------------------------------------------------------------------
# 7. GUI -> worker -> engine: exact typed values, disk config different
# ---------------------------------------------------------------------------

def _worker_kwargs_from_gui(win, tmp_path):
    _prepare_zegrid(win, tmp_path)
    win._collect_config_from_widgets()
    win._save_config()
    _args, worker_kwargs = win._build_worker_invocation(skip_filter_ui=True)
    return worker_kwargs


def _disk_config(**overrides):
    disk = dict(zc.DEFAULT_CONFIG)
    disk.update(
        {
            "input_dir": "/nonexistent",
            "output_dir": "/nonexistent",
            "zegrid_mode_enabled": True,
            "processing_mode": "zegrid",
            "zegrid_resume_mode": "never",
        }
    )
    disk.update(overrides)
    return disk


def _drive(monkeypatch, tmp_path, worker_kwargs, disk):
    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(zw.zemosaic_config, "load_config", lambda: dict(disk))
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


def test_gui_values_reach_engine_even_when_disk_config_differs(
    tmp_path, monkeypatch
):
    win = _make_window(tmp_path, monkeypatch, dict(EXPLICIT))
    worker_kwargs = _worker_kwargs_from_gui(win, tmp_path)
    for key, expected in EXPLICIT.items():
        assert worker_kwargs[key] == expected, key

    captured = _drive(monkeypatch, tmp_path, worker_kwargs, _disk_config(**DISK_VALUES))
    zconfig = captured["zconfig"]
    for key, expected in EXPLICIT.items():
        assert getattr(zconfig, key) == expected, key
    # The worker did not fall back to the different persisted values.
    for key, wrong in DISK_VALUES.items():
        assert getattr(zconfig, key) != wrong, key


def test_absent_transport_keeps_persisted_config(tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch, dict(EXPLICIT))
    worker_kwargs = _worker_kwargs_from_gui(win, tmp_path)
    for key in POLICY_KEYS:
        worker_kwargs.pop(key, None)

    captured = _drive(monkeypatch, tmp_path, worker_kwargs, _disk_config(**DISK_VALUES))
    zconfig = captured["zconfig"]
    for key, expected in DISK_VALUES.items():
        assert getattr(zconfig, key) == expected, key


def test_signature_and_kwargs_contract():
    params = inspect.signature(zw.run_hierarchical_mosaic).parameters
    for key in POLICY_KEYS:
        assert key in params, key
        assert params[key].default is zw._ZEGRID_POLICY_UNSET, key
    assert set(zw._ZEGRID_POLICY_TRANSPORT_KEYS) == set(POLICY_KEYS)
    # The engine contract is zconfig; the six keys are read from it.
    assert "zconfig" in inspect.signature(zz.run_zegrid_mode).parameters
    process_params = inspect.signature(zw.run_hierarchical_mosaic_process).parameters
    assert any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in process_params.values()
    )


# ---------------------------------------------------------------------------
# 8. Real first-run plumbing smoke through GUI -> dispatcher -> engine
# ---------------------------------------------------------------------------

def _make_corpus(path: Path) -> Path:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(
        [sys.executable, str(CORPUS_TOOL), str(path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return path


def test_real_tiny_run_uses_gui_policy_values(tmp_path, monkeypatch):
    corpus = _make_corpus(tmp_path / "in")
    out = tmp_path / "out"
    config = {
        "input_dir": str(corpus),
        "output_dir": str(out),
        "zegrid_mode_enabled": True,
        "processing_mode": "zegrid",
        "zegrid_halo_px": 8,
        "zegrid_scheduler_mode": "legacy",
        "zegrid_layout": "2x2",
        "zegrid_cell_gauge_mode": "global",
        "zegrid_sip_mode": "keep",
        "zegrid_resource_recovery_enabled": False,
        "zegrid_resource_recovery_deadline_s": None,
        "zegrid_durable_store_budget_bytes": None,
        "use_gpu_phase5": False,
    }
    win = _make_window(tmp_path, monkeypatch, config)
    _prepare_zegrid(win, tmp_path)
    win.config["input_dir"] = str(corpus)
    win.config["output_dir"] = str(out)
    win._collect_config_from_widgets()
    win._save_config()
    _args, worker_kwargs = win._build_worker_invocation(skip_filter_ui=True)
    worker_kwargs["input_folder"] = str(corpus)
    worker_kwargs["output_folder"] = str(out)

    monkeypatch.setattr(
        zw.zemosaic_config,
        "load_config",
        lambda: _disk_config(
            input_dir=str(corpus),
            output_dir=str(out),
            zegrid_halo_px=8,
            zegrid_layout="1x1",
            zegrid_scheduler_mode="legacy",
        ),
    )

    required = _worker_required_kwargs(str(corpus), str(out))
    required.update(worker_kwargs)
    required["filter_overrides"] = {"processing_mode": "zegrid"}
    required["crash_breadcrumbs_mode"] = "off"
    required.pop("solver_settings_dict", None)
    zw.run_hierarchical_mosaic_process(
        _RecordingQueue(), solver_settings_dict=None, **required
    )

    manifest_path = out / "zegrid_manifest.json"
    assert manifest_path.is_file(), sorted(p.name for p in out.iterdir())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["engine"] == "zegrid"
    assert manifest["n_frames"] == 4
    assert manifest["layout"]["nx"] == 2 and manifest["layout"]["ny"] == 2
    science = Path(manifest["outputs"]["science"])
    assert (out / science).is_file() or science.is_file()
