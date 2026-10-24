"""ZM-ZEGRID-R55-E4 rework-1 — real load path honors legacy-only GPU intent.

R0 fixed the Qt synchronization but claimed that a legacy config carrying only
``use_gpu_grid`` kept its intent. That was false through ``load_config()``:
``DEFAULT_CONFIG`` (``use_gpu_phase5=True``) was merged BEFORE normalization, so
the canonical key was never genuinely absent and a persisted legacy false/off
value was overwritten by the default.

This module locks the real path (not just a raw helper):

* ``load_config()`` preserves explicit persisted-key presence, so a legacy-only
  true OR false/off intent is honoured before defaults mask absence;
* an explicit canonical ``use_gpu_phase5`` remains authoritative in both
  directions;
* ``save_config()`` persists a coherent four-key state and a reload does not
  resurrect defaults;
* the same precedence is observed by the raw normalizer, ``load_config``, the
  Qt loader and the ZeGrid engine resolver.
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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_config as zconfig_mod  # noqa: E402
from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402

GPU_KEYS = ("use_gpu_phase5", "use_gpu_grid", "stack_use_gpu", "use_gpu_stack")


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


def _config_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "zemosaic_config.json"
    monkeypatch.setattr(zconfig_mod, "get_config_path", lambda: path)
    return path


def _assert_coherent(cfg: dict, expected: bool) -> None:
    for key in GPU_KEYS:
        assert cfg.get(key) is expected, (key, cfg)


# ---------------------------------------------------------------------------
# 1. Real load_config() matrix: legacy-only true AND false, bool AND string
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"use_gpu_grid": False}, False),
        ({"use_gpu_grid": "off"}, False),
        ({"use_gpu_grid": "0"}, False),
        ({"use_gpu_grid": True}, True),
        ({"use_gpu_grid": "1"}, True),
        ({"use_gpu_grid": "on"}, True),
        ({"stack_use_gpu": False}, False),
        ({"stack_use_gpu": "no"}, False),
        ({"stack_use_gpu": True}, True),
        ({"use_gpu_stack": "0"}, False),
        ({"use_gpu_stack": "yes"}, True),
    ],
)
def test_load_config_honours_legacy_only_intent(tmp_path, monkeypatch, raw, expected):
    path = _config_path(tmp_path, monkeypatch)
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = zconfig_mod.load_config()
    _assert_coherent(loaded, expected)


@pytest.mark.parametrize(
    "raw, expected",
    [
        # explicit canonical wins over contradictory legacy values (both senses)
        (
            {
                "use_gpu_phase5": False,
                "use_gpu_grid": True,
                "stack_use_gpu": "yes",
                "use_gpu_stack": 1,
            },
            False,
        ),
        (
            {
                "use_gpu_phase5": "true",
                "use_gpu_grid": "off",
                "stack_use_gpu": False,
                "use_gpu_stack": "0",
            },
            True,
        ),
        ({"use_gpu_phase5": False}, False),
        ({"use_gpu_phase5": True}, True),
    ],
)
def test_load_config_canonical_key_is_authoritative(tmp_path, monkeypatch, raw, expected):
    path = _config_path(tmp_path, monkeypatch)
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = zconfig_mod.load_config()
    _assert_coherent(loaded, expected)


def test_load_config_without_any_gpu_key_keeps_default(tmp_path, monkeypatch):
    """Absence of user intent must not silently disable the default GPU policy."""
    path = _config_path(tmp_path, monkeypatch)
    path.write_text(json.dumps({"stack_norm_method": "sky_mean"}), encoding="utf-8")

    loaded = zconfig_mod.load_config()
    _assert_coherent(loaded, True)


# ---------------------------------------------------------------------------
# 2. Raw normalizer / load_config / Qt loader / engine all agree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"use_gpu_grid": False}, False),
        ({"use_gpu_grid": "off"}, False),
        ({"stack_use_gpu": False}, False),
        ({"use_gpu_stack": "0"}, False),
        ({"use_gpu_grid": True}, True),
        ({"use_gpu_phase5": False, "use_gpu_grid": True}, False),
        ({"use_gpu_phase5": True, "stack_use_gpu": False}, True),
    ],
)
def test_precedence_is_consistent_across_layers(tmp_path, monkeypatch, raw, expected):
    # Raw normalizer (in-memory dict: every present key is explicit)
    normalized = dict(raw)
    zconfig_mod._normalize_gpu_flags(normalized)
    _assert_coherent(normalized, expected)

    # Real persisted load
    path = _config_path(tmp_path, monkeypatch)
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = zconfig_mod.load_config()
    _assert_coherent(loaded, expected)

    # Engine resolver sees the same coherent object
    requested, _source = zz.resolve_gpu_preference(None, SimpleNamespace(**loaded))
    assert requested is expected


# ---------------------------------------------------------------------------
# 3. save_config round-trip: no default resurrection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"use_gpu_grid": False}, False),
        ({"use_gpu_grid": "off"}, False),
        ({"stack_use_gpu": False}, False),
        ({"use_gpu_phase5": False, "use_gpu_grid": True}, False),
        ({"use_gpu_grid": True}, True),
    ],
)
def test_save_config_round_trip_keeps_migrated_gpu_state(
    tmp_path, monkeypatch, raw, expected
):
    path = _config_path(tmp_path, monkeypatch)
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = zconfig_mod.load_config()
    assert zconfig_mod.save_config(loaded) in (True, None)

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    _assert_coherent(on_disk, expected)

    # A second load must not resurrect DEFAULT_CONFIG's GPU value.
    reloaded = zconfig_mod.load_config()
    _assert_coherent(reloaded, expected)


# ---------------------------------------------------------------------------
# 4. Qt load/save interaction on a legacy-only config
# ---------------------------------------------------------------------------

def _qt_window(tmp_path, monkeypatch):
    import zemosaic.zemosaic_gui_qt as gui

    _config_path(tmp_path, monkeypatch)
    monkeypatch.setattr(gui, "_detect_analysis_backend", lambda: ("none", None, None))
    return gui.ZeMosaicQtMainWindow()


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"use_gpu_grid": False}, False),
        ({"use_gpu_grid": "off"}, False),
        ({"stack_use_gpu": False}, False),
        ({"use_gpu_grid": True}, True),
    ],
)
def test_qt_load_then_save_round_trips_legacy_intent(
    tmp_path, monkeypatch, raw, expected
):
    path = _config_path(tmp_path, monkeypatch)
    path.write_text(json.dumps(raw), encoding="utf-8")

    win = _qt_window(tmp_path, monkeypatch)
    checkbox = win._config_fields["use_gpu_phase5"]["widget"]
    assert checkbox.isChecked() is expected
    for key in GPU_KEYS:
        assert win.config[key] is expected, (key, win.config.get(key))

    win._save_config()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    _assert_coherent(on_disk, expected)

    # Re-open: the migrated state survives without falling back to defaults.
    win2 = _qt_window(tmp_path, monkeypatch)
    assert win2._config_fields["use_gpu_phase5"]["widget"].isChecked() is expected
    for key in GPU_KEYS:
        assert win2.config[key] is expected, (key, win2.config.get(key))
