"""ZM-ZEGRID-GAUGE-IMPORT-GUI — the verified engine option, reachable by users.

Offscreen Qt only. Every transport assertion captures the ACTUAL worker kwargs
returned by ``_build_worker_invocation`` (not a helper's return value), and the
CLI assertion captures the real ``run_zegrid_mode`` kwargs.
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

import zemosaic.zemosaic_gui_qt as gui  # noqa: E402
import zemosaic.zegrid_cli as zcli  # noqa: E402


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
    cfg_path = tmp_path / "zemosaic_config.json"

    def _save(data):
        if save_store is not None:
            save_store.clear()
            save_store.update(data)
        return None

    fake_cfg = SimpleNamespace(
        load_config=lambda: dict(config),
        save_config=_save,
        get_config_path=lambda: cfg_path,
        DEFAULT_CONFIG={},
    )
    monkeypatch.setattr(gui, "zemosaic_config", fake_cfg)
    monkeypatch.setattr(gui, "_detect_analysis_backend", lambda: ("none", None, None))
    return gui.ZeMosaicQtMainWindow()


def _file(path: Path, payload: dict | None = None) -> Path:
    path.write_text(json.dumps(payload or {}), encoding="utf-8")
    return path


def _options(tmp_path):
    gauge = _file(tmp_path / "gauge.json")
    origin = _file(tmp_path / "run.json")
    return gauge, origin


# ---------------------------------------------------------------------------
# a) actual worker kwargs: exact mapping when enabled, NO key when disabled
# ---------------------------------------------------------------------------

def test_gui_transports_the_exact_mapping_and_off_sends_no_key(tmp_path, monkeypatch):
    gauge, origin = _options(tmp_path)
    enabled = _make_window(tmp_path, monkeypatch, {
        "zegrid_gauge_import_enabled": True,
        "zegrid_gauge_import_path": str(gauge),
        "zegrid_gauge_origin_run_json": str(origin),
    })
    _args, kwargs = enabled._build_worker_invocation(skip_filter_ui=True)
    assert kwargs["zegrid_gauge_import"] == {
        "gauge_path": str(gauge),
        "origin_run_json": str(origin),
    }

    disabled = _make_window(tmp_path, monkeypatch, {})
    _args2, kwargs2 = disabled._build_worker_invocation(skip_filter_ui=True)
    assert "zegrid_gauge_import" not in kwargs2


# ---------------------------------------------------------------------------
# b) pre-launch refusals: empty / directory / unreadable, field named
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["empty_gauge", "directory", "missing", "unreadable"])
def test_gui_refuses_bad_paths_and_starts_no_worker(tmp_path, monkeypatch, case):
    gauge, origin = _options(tmp_path)
    if case == "empty_gauge":
        gauge_path, origin_path = "", str(origin)
        expected = "Gauge artifact"
    elif case == "directory":
        gauge_path, origin_path = str(tmp_path), str(origin)
        expected = "folder"
    elif case == "missing":
        gauge_path, origin_path = str(tmp_path / "nope.json"), str(origin)
        expected = "does not exist"
    else:
        gauge_path, origin_path = str(gauge), str(tmp_path / "missing-run.json")
        expected = "Origin run identity"
    output_dir = tmp_path / "out"
    warnings: list = []
    monkeypatch.setattr(
        gui.QMessageBox, "warning",
        staticmethod(lambda *args, **_kw: warnings.append(str(args[-1]))),
    )
    win = _make_window(tmp_path, monkeypatch, {
        "zegrid_gauge_import_enabled": True,
        "zegrid_gauge_import_path": gauge_path,
        "zegrid_gauge_origin_run_json": origin_path,
        "output_dir": str(output_dir),
    })
    win._mode_checkboxes["zegrid"].setChecked(True)
    assert win._validate_zegrid_policy_for_start() is False
    assert warnings and expected in warnings[-1]
    assert not output_dir.exists()
    assert getattr(win, "is_processing", False) is False


# ---------------------------------------------------------------------------
# c) settings round trip; absent keys default to disabled
# ---------------------------------------------------------------------------

def test_settings_round_trip_and_absent_defaults(tmp_path, monkeypatch):
    gauge, origin = _options(tmp_path)
    saved: dict = {}
    win = _make_window(tmp_path, monkeypatch, {}, save_store=saved)
    assert win._zegrid_gauge_import_checkbox.isChecked() is False
    assert win._zegrid_gauge_import_edit.text() == ""

    win._zegrid_gauge_import_checkbox.setChecked(True)
    win._on_zegrid_gauge_import_toggled()
    win._zegrid_gauge_import_edit.setText(str(gauge))
    win._zegrid_gauge_origin_edit.setText(str(origin))
    win._on_zegrid_policy_text_edited("zegrid_gauge_import_path", user_edit=True)
    win._on_zegrid_policy_text_edited(
        "zegrid_gauge_origin_run_json", user_edit=True
    )
    serialized = win._serialize_config_for_save()
    assert serialized["zegrid_gauge_import_enabled"] is True
    assert serialized["zegrid_gauge_import_path"] == str(gauge)
    assert serialized["zegrid_gauge_origin_run_json"] == str(origin)

    restored = _make_window(tmp_path, monkeypatch, serialized)
    assert restored._zegrid_gauge_import_checkbox.isChecked() is True
    assert restored._zegrid_gauge_import_edit.text() == str(gauge)
    assert restored._zegrid_gauge_origin_edit.text() == str(origin)
    assert restored._zegrid_gauge_import_edit.isEnabled() is True


# ---------------------------------------------------------------------------
# e) the snapshot and the GUI log declare the import, naming the path
# ---------------------------------------------------------------------------

def test_snapshot_and_log_declare_the_import(tmp_path, monkeypatch):
    gauge, origin = _options(tmp_path)
    win = _make_window(tmp_path, monkeypatch, {
        "zegrid_gauge_import_enabled": True,
        "zegrid_gauge_import_path": str(gauge),
        "zegrid_gauge_origin_run_json": str(origin),
    })
    logs: list = []
    monkeypatch.setattr(
        win, "_append_log",
        lambda message, level="info", **_kw: logs.append(str(message)),
    )
    snapshot = win._snapshot_for_run()
    assert snapshot["gauge_source"] == "imported_artifact"
    assert snapshot["zegrid_gauge_import"]["gauge_path"] == str(gauge)
    win._build_worker_invocation(skip_filter_ui=True)
    assert any(
        "gauge_source=imported_artifact" in line and str(gauge) in line
        for line in logs
    )


# ---------------------------------------------------------------------------
# d) CLI: mapping, refusals, untouched invocation when omitted
# ---------------------------------------------------------------------------

def _run_cli(monkeypatch, argv):
    import zemosaic.zemosaic_zegrid_mode as zz

    calls: list = []

    def fake_run(*args, **kwargs):
        calls.append(dict(kwargs))
        return str(Path(argv[argv.index("--output") + 1]) / "science.fits")

    monkeypatch.setattr(zz, "run_zegrid_mode", fake_run)
    code = zcli.main(argv)
    return code, calls


def _cli_base(tmp_path):
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    # The CLI validates that the input folder contains FITS frames before it
    # reaches the engine; the engine itself is stubbed in this test.
    (inp / "frame_0001.fits").write_bytes(b"")
    return [
        "--input", str(inp), "--output", str(out),
        "--json-summary", str(tmp_path / "summary.json"),
    ]


def test_cli_maps_the_option_and_refuses_bad_paths(tmp_path, monkeypatch):
    gauge, origin = _options(tmp_path)
    base = _cli_base(tmp_path)

    code, calls = _run_cli(monkeypatch, base + [
        "--gauge-import", str(gauge), "--gauge-origin", str(origin),
    ])
    assert calls and calls[0]["zegrid_gauge_import"] == {
        "gauge_path": str(gauge), "origin_run_json": str(origin),
    }

    code, calls = _run_cli(monkeypatch, base)
    assert calls and calls[0].get("zegrid_gauge_import") is None

    code, calls = _run_cli(monkeypatch, base + ["--gauge-import", str(gauge)])
    assert code == zcli.EXIT_REFUSED and calls == []

    code, calls = _run_cli(monkeypatch, base + [
        "--gauge-import", str(tmp_path / "missing.json"),
        "--gauge-origin", str(origin),
    ])
    assert code == zcli.EXIT_REFUSED and calls == []

    code, calls = _run_cli(monkeypatch, base + [
        "--gauge-import", str(tmp_path), "--gauge-origin", str(origin),
    ])
    assert code == zcli.EXIT_REFUSED and calls == []
