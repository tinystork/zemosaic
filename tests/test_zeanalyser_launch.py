"""Focused tests for the optional ZeAnalyser launch integration (lot ZA-M1-3A).

ZeAnalyser is optional for ZeMosaic.  Discovery goes exclusively through
``importlib.metadata`` (installed distribution + declared ``gui_scripts``
entry point — interop rules 2, 6, 19, 20) and launch goes through the current
environment's entry-point script or the documented ``python -m zeanalyser``
module entry point.  No sibling checkout is ever inspected, no ``sys.path``
mutation happens and ``zeanalyser`` is never imported.

These tests never require a real ZeAnalyser installation: they monkeypatch the
metadata functions, ``sysconfig.get_path`` and ``subprocess.Popen``.  Widgets
are created offscreen and dialogs are recorded, so no real GUI interaction is
needed.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from zemosaic import zemosaic_gui_qt as gui  # noqa: E402


# ---------------------------------------------------------------------------
# Metadata fakes
# ---------------------------------------------------------------------------


class _FakeDistribution:
    """Stand-in for an installed distribution metadata record."""


class _FakeEntryPoint:
    name = "zeanalyser"
    group = "gui_scripts"
    value = "zeanalyser.analyse_gui_qt:main"


def _distribution_ok(name: str):
    assert name == gui.ZEANALYSER_DISTRIBUTION
    return _FakeDistribution()


def _distribution_missing(name: str):
    raise importlib_metadata.PackageNotFoundError(name)


def _entry_points_ok(group=None, name=None):
    return [_FakeEntryPoint()]


def _entry_points_empty(group=None, name=None):
    return []


@pytest.fixture()
def isolated_base_dir(tmp_path, monkeypatch):
    """A ZeMosaic base dir with no seestar/beforehand sibling on disk."""
    base = tmp_path / "zemosaic"
    base.mkdir()
    monkeypatch.setattr(gui, "get_app_base_dir", lambda: base)
    return base


# ---------------------------------------------------------------------------
# Discovery (importlib.metadata only, never crashes)
# ---------------------------------------------------------------------------


def test_detect_absent_distribution_returns_none(monkeypatch, isolated_base_dir):
    monkeypatch.setattr(importlib_metadata, "distribution", _distribution_missing)
    backend, root, diagnostic = gui._detect_analysis_backend()
    assert backend == "none"
    assert root is None
    assert diagnostic is None


def test_detect_available_distribution_returns_zeanalyser(monkeypatch, isolated_base_dir):
    monkeypatch.setattr(importlib_metadata, "distribution", _distribution_ok)
    monkeypatch.setattr(importlib_metadata, "entry_points", _entry_points_ok)
    backend, root, diagnostic = gui._detect_analysis_backend()
    assert backend == "zeanalyser"
    assert root is None  # no checkout path must ever be produced
    assert diagnostic is None


def test_discovery_pins_public_launch_contract(monkeypatch, isolated_base_dir):
    # The discovery must ask for exactly the documented distribution and the
    # documented gui_scripts entry point (never anything else).
    captured = {}

    def fake_distribution(name):
        captured["distribution"] = name
        return _FakeDistribution()

    def fake_entry_points(group=None, name=None):
        captured["group"] = group
        captured["name"] = name
        return [_FakeEntryPoint()]

    monkeypatch.setattr(importlib_metadata, "distribution", fake_distribution)
    monkeypatch.setattr(importlib_metadata, "entry_points", fake_entry_points)

    backend, _, diagnostic = gui._detect_analysis_backend()
    assert backend == "zeanalyser"
    assert diagnostic is None
    assert captured == {
        "distribution": "ZeAnalyser",
        "group": "gui_scripts",
        "name": "zeanalyser",
    }


def test_detect_installed_but_entry_point_missing_is_unavailable(
    monkeypatch, isolated_base_dir
):
    monkeypatch.setattr(importlib_metadata, "distribution", _distribution_ok)
    monkeypatch.setattr(importlib_metadata, "entry_points", _entry_points_empty)
    backend, root, diagnostic = gui._detect_analysis_backend()
    assert backend == "none"
    assert root is None
    assert diagnostic is not None
    assert "zeanalyser" in diagnostic
    assert "gui_scripts" in diagnostic


def test_detect_distribution_inspection_exception_no_crash(
    monkeypatch, isolated_base_dir
):
    def broken_distribution(name):
        raise RuntimeError("metadata boom")

    monkeypatch.setattr(importlib_metadata, "distribution", broken_distribution)
    backend, root, diagnostic = gui._detect_analysis_backend()
    assert backend == "none"
    assert root is None
    assert diagnostic is not None
    assert "metadata boom" in diagnostic


def test_detect_entry_points_exception_no_crash(monkeypatch, isolated_base_dir):
    def broken_entry_points(group=None, name=None):
        raise OSError("entry points boom")

    monkeypatch.setattr(importlib_metadata, "distribution", _distribution_ok)
    monkeypatch.setattr(importlib_metadata, "entry_points", broken_entry_points)
    backend, root, diagnostic = gui._detect_analysis_backend()
    assert backend == "none"
    assert root is None
    assert diagnostic is not None
    assert "entry points boom" in diagnostic


def test_detect_never_uses_sibling_zeanalyser_checkout(monkeypatch, tmp_path):
    # The old forbidden architecture: a sibling ``zeanalyser/`` checkout must
    # never be detected, even when it exists right next to ZeMosaic.
    base = tmp_path / "zemosaic"
    base.mkdir()
    sibling = tmp_path / "zeanalyser"
    sibling.mkdir()
    (sibling / "analyse_gui_qt.py").write_text("# fake checkout\n", encoding="utf-8")
    monkeypatch.setattr(gui, "get_app_base_dir", lambda: base)
    monkeypatch.setattr(importlib_metadata, "distribution", _distribution_missing)

    backend, root, diagnostic = gui._detect_analysis_backend()
    assert backend == "none"
    assert root is None
    assert diagnostic is None


def test_detect_beforehand_branch_unchanged(monkeypatch, tmp_path):
    base = tmp_path / "zemosaic"
    base.mkdir()
    beforehand_dir = base.parent / "seestar" / "beforehand"
    beforehand_dir.mkdir(parents=True)
    monkeypatch.setattr(gui, "get_app_base_dir", lambda: base)
    monkeypatch.setattr(importlib_metadata, "distribution", _distribution_missing)

    backend, root, diagnostic = gui._detect_analysis_backend()
    assert backend == "beforehand"
    assert root == beforehand_dir
    assert diagnostic is None


def test_detect_installed_zeanalyser_wins_over_beforehand(monkeypatch, tmp_path):
    base = tmp_path / "zemosaic"
    base.mkdir()
    (base.parent / "seestar" / "beforehand").mkdir(parents=True)
    monkeypatch.setattr(gui, "get_app_base_dir", lambda: base)
    monkeypatch.setattr(importlib_metadata, "distribution", _distribution_ok)
    monkeypatch.setattr(importlib_metadata, "entry_points", _entry_points_ok)

    backend, _, diagnostic = gui._detect_analysis_backend()
    assert backend == "zeanalyser"
    assert diagnostic is None


def test_detect_unhealthy_zeanalyser_falls_back_to_beforehand(monkeypatch, tmp_path):
    # Alternative backend is preserved (interop rule 6) while the ZeAnalyser
    # diagnostic is still reported.
    base = tmp_path / "zemosaic"
    base.mkdir()
    beforehand_dir = base.parent / "seestar" / "beforehand"
    beforehand_dir.mkdir(parents=True)
    monkeypatch.setattr(gui, "get_app_base_dir", lambda: base)
    monkeypatch.setattr(importlib_metadata, "distribution", _distribution_ok)
    monkeypatch.setattr(importlib_metadata, "entry_points", _entry_points_empty)

    backend, root, diagnostic = gui._detect_analysis_backend()
    assert backend == "beforehand"
    assert root == beforehand_dir
    assert diagnostic is not None
    assert "zeanalyser" in diagnostic


# ---------------------------------------------------------------------------
# Launch command resolution
# ---------------------------------------------------------------------------


def _script_name():
    return "zeanalyser.exe" if gui.IS_WINDOWS else "zeanalyser"


def test_resolve_launch_command_uses_scripts_entry_point(monkeypatch, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    entry = scripts / _script_name()
    entry.write_text("#!/bin/sh\n", encoding="utf-8")
    entry.chmod(0o755)
    monkeypatch.setattr(
        sysconfig, "get_path", lambda name: str(scripts) if name == "scripts" else ""
    )

    cmd = gui._resolve_zeanalyser_launch_command()
    assert cmd == [str(entry)]


def test_resolve_launch_command_script_not_executable_falls_back(
    monkeypatch, tmp_path
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    entry = scripts / _script_name()
    entry.write_text("#!/bin/sh\n", encoding="utf-8")
    entry.chmod(0o644)
    monkeypatch.setattr(
        sysconfig, "get_path", lambda name: str(scripts) if name == "scripts" else ""
    )

    cmd = gui._resolve_zeanalyser_launch_command()
    assert cmd == [sys.executable, "-m", "zeanalyser"]


def test_resolve_launch_command_missing_script_falls_back(monkeypatch, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    monkeypatch.setattr(
        sysconfig, "get_path", lambda name: str(scripts) if name == "scripts" else ""
    )

    cmd = gui._resolve_zeanalyser_launch_command()
    assert cmd == [sys.executable, "-m", "zeanalyser"]


def test_resolve_launch_command_broken_sysconfig_falls_back(monkeypatch):
    def broken_get_path(name):
        raise RuntimeError("sysconfig boom")

    monkeypatch.setattr(sysconfig, "get_path", broken_get_path)
    cmd = gui._resolve_zeanalyser_launch_command()
    assert cmd == [sys.executable, "-m", "zeanalyser"]


# ---------------------------------------------------------------------------
# Launch behavior (bare window instance, recorded dialogs)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _bare_window(backend="none", root=None, diagnostic=None):
    # Skip the full GUI __init__ (config load, GPU detection, widget tree):
    # create the C++ widget through the sanctioned Shiboken __new__ path and
    # set only the attributes the launch path reads.
    window = gui.ZeMosaicQtMainWindow.__new__(gui.ZeMosaicQtMainWindow)
    window.analysis_backend = backend
    window.analysis_backend_root = root
    window.analysis_backend_diagnostic = diagnostic
    window._log_level_prefixes = {}
    return window


@pytest.fixture()
def message_boxes(monkeypatch):
    recorded = {"information": [], "critical": [], "warning": []}

    def record(bucket):
        def fake(parent, title, text, *args, **kwargs):
            bucket.append((parent, title, text))

        return fake

    monkeypatch.setattr(QMessageBox, "information", record(recorded["information"]))
    monkeypatch.setattr(QMessageBox, "critical", record(recorded["critical"]))
    monkeypatch.setattr(QMessageBox, "warning", record(recorded["warning"]))
    return recorded


@pytest.fixture()
def popen_recorder(monkeypatch):
    calls = []

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            calls.append({"cmd": list(cmd), "kwargs": kwargs})
            self.pid = 12345
            self.returncode = None

        def poll(self):  # pragma: no cover - not used by the tests
            return None

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    return calls


def test_launch_zeanalyser_uses_scripts_entry_point(
    qapp, popen_recorder, message_boxes, monkeypatch, tmp_path
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    entry = scripts / _script_name()
    entry.write_text("#!/bin/sh\n", encoding="utf-8")
    entry.chmod(0o755)
    monkeypatch.setattr(
        sysconfig, "get_path", lambda name: str(scripts) if name == "scripts" else ""
    )

    window = _bare_window(backend="zeanalyser")
    window._launch_analysis_backend()

    assert len(popen_recorder) == 1
    assert popen_recorder[0]["cmd"] == [str(entry)]
    kwargs = popen_recorder[0]["kwargs"]
    assert "cwd" not in kwargs  # never a checkout working directory
    assert kwargs.get("shell") is False
    assert not message_boxes["critical"]
    assert not message_boxes["information"]


def test_launch_zeanalyser_falls_back_to_module_invocation(
    qapp, popen_recorder, message_boxes, monkeypatch, tmp_path
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()  # no zeanalyser script inside
    monkeypatch.setattr(
        sysconfig, "get_path", lambda name: str(scripts) if name == "scripts" else ""
    )

    window = _bare_window(backend="zeanalyser")
    window._launch_analysis_backend()

    assert len(popen_recorder) == 1
    assert popen_recorder[0]["cmd"] == [sys.executable, "-m", "zeanalyser"]
    assert "cwd" not in popen_recorder[0]["kwargs"]
    assert not message_boxes["critical"]


def test_launch_popen_failure_shows_critical_no_crash(qapp, message_boxes, monkeypatch):
    def failing_popen(cmd, **kwargs):
        raise OSError("spawn denied")

    monkeypatch.setattr(subprocess, "Popen", failing_popen)

    window = _bare_window(backend="zeanalyser")
    window._launch_analysis_backend()  # must not raise

    assert len(message_boxes["critical"]) == 1
    title, text = (
        message_boxes["critical"][0][1],
        message_boxes["critical"][0][2],
    )
    assert title == "Analysis launch failed"
    assert "spawn denied" in text


def test_launch_no_backend_shows_info(qapp, message_boxes):
    window = _bare_window(backend="none")
    window._launch_analysis_backend()
    assert len(message_boxes["information"]) == 1
    assert "No analysis backend is available" in message_boxes["information"][0][2]
    assert not message_boxes["critical"]


def test_launch_no_backend_with_diagnostic_reports_it(qapp, message_boxes):
    window = _bare_window(
        backend="none", diagnostic="installed but broken: entry point missing"
    )
    window._launch_analysis_backend()
    assert len(message_boxes["information"]) == 1
    assert "entry point missing" in message_boxes["information"][0][2]


def test_launch_beforehand_keeps_legacy_message(qapp, message_boxes):
    window = _bare_window(backend="beforehand", root=Path("/tmp/beforehand"))
    window._launch_analysis_backend()
    assert len(message_boxes["information"]) == 1
    title, text = (
        message_boxes["information"][0][1],
        message_boxes["information"][0][2],
    )
    assert title == "Beforehand detected"
    assert "/tmp/beforehand" in text
    assert not message_boxes["critical"]


def test_launch_beforehand_without_root_stays_graceful(qapp, message_boxes):
    window = _bare_window(backend="beforehand", root=None)
    window._launch_analysis_backend()
    assert len(message_boxes["information"]) == 1
    assert "No analysis backend is available" in message_boxes["information"][0][2]


# ---------------------------------------------------------------------------
# Analyse button visibility
# ---------------------------------------------------------------------------


def test_analyse_button_not_created_when_backend_none(qapp):
    window = _bare_window(backend="none")
    window._tr = lambda key, fallback=None: fallback
    window._build_command_row()
    assert window.analysis_button is None


def test_analyse_button_created_when_zeanalyser_detected(qapp):
    window = _bare_window(backend="zeanalyser")
    window._tr = lambda key, fallback=None: fallback
    window._build_command_row()
    assert window.analysis_button is not None
    assert window.analysis_button.text() == "Analyse"
    assert "ZeAnalyser" in window.analysis_button.toolTip()


def test_analyse_button_created_when_beforehand_detected(qapp):
    window = _bare_window(backend="beforehand", root=Path("/tmp/beforehand"))
    window._tr = lambda key, fallback=None: fallback
    window._build_command_row()
    assert window.analysis_button is not None
    assert "Beforehand" in window.analysis_button.toolTip()
