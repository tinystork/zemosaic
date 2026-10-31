"""ZM-ZEGRID-M6-QT-STOP-RESUME — GUI stop-and-save + durable resume wiring.

Bounded vertical slice only:

* the GUI resolves a deterministic durable store (or an explicit one), keeps an
  explicit per-launch Resume / Start new / Cancel decision, and transports it to
  the REAL spawned worker/engine;
* Stop and Save Progress uses a spawn-safe cooperative channel and reports only
  what the engine acknowledged;
* discovery/classification is read-only and never blesses corrupt, complete or
  live-locked generations.

The last test is a bounded synthetic integration proof on a tiny deterministic
corpus (never the user's datasets): a GUI-built invocation -> real spawned
worker -> real engine stop -> NEW spawned process resume.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import queue as queue_mod
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _path in (SRC, TESTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import zemosaic.zemosaic_config as zc  # noqa: E402
import zemosaic.zemosaic_gui_qt as gui  # noqa: E402
import zemosaic.zemosaic_worker as zw  # noqa: E402
import zemosaic.zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402
from zemosaic.core.zegrid import run_publication as zpub  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

import test_zegrid_resume_m4 as m4  # noqa: E402
from test_zegrid_m6_gui_policy import (  # noqa: E402
    _drive,
    _disk_config,
)
from test_zegrid_scheduler_gui_advanced import (  # noqa: E402
    _make_window,
    _prepare_zegrid,
)

DURABLE_KEYS = (
    "zegrid_durable_store_dir",
    "zegrid_resume_mode",
    "zegrid_resume_run_id",
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:  # pragma: no cover - Qt missing
        pytest.skip("PySide6 unavailable")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _window(tmp_path, monkeypatch, extra=None):
    win = _make_window(tmp_path, monkeypatch, dict(extra or {}))
    _prepare_zegrid(win, tmp_path)
    win._collect_config_from_widgets()
    # ``_collect_config_from_widgets`` legitimately mirrors widget values; the
    # folder selection is not a widget in these headless windows.
    win.config["input_dir"] = str(tmp_path / "in")
    win.config["output_dir"] = str(tmp_path / "out")
    return win


def _tree_digest(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _make_complete(root: Path, *, run_id: str):
    m4.seeded_store(
        root, identity=m4.make_identity(run_id=run_id), cell_indexes=(0, 1)
    )
    state = zres.read_state(root)
    zres.update_policy(root, state["policy"], finalized=True)
    return root


# ---------------------------------------------------------------------------
# 1. Defaults + deterministic store resolution
# ---------------------------------------------------------------------------


def test_checkpoint_opt_in_default_is_off_and_engine_defaults_untouched():
    assert zc.DEFAULT_CONFIG["zegrid_qt_save_checkpoints_enabled"] is False
    assert zc.DEFAULT_CONFIG["zegrid_resume_mode"] == "never"
    assert zc.DEFAULT_CONFIG["zegrid_resume_run_id"] is None
    assert zc.DEFAULT_CONFIG["zegrid_durable_store_dir"] is None


def test_store_root_is_deterministic_under_output_or_explicit(tmp_path, monkeypatch):
    win = _window(
        tmp_path,
        monkeypatch,
        {"zegrid_qt_save_checkpoints_enabled": True},
    )
    expected_default = str(tmp_path / "out" / win.ZEGRID_DEFAULT_STORE_DIRNAME)
    assert win._zegrid_checkpoint_store_root() == expected_default
    assert win.config["zegrid_durable_store_dir"] is None

    explicit = tmp_path / "explicit-store"
    win._zegrid_checkpoint_store_edit.setText(str(explicit))
    win._on_zegrid_checkpoint_store_edited(user_edit=True)
    win._collect_config_from_widgets()
    assert win.config["zegrid_durable_store_dir"] == str(explicit)
    assert win._zegrid_checkpoint_store_root() == str(explicit)

    # No output folder and no explicit store -> no invented path.
    win.config["output_dir"] = ""
    win.config["zegrid_durable_store_dir"] = None
    win._zegrid_checkpoint_store_edit.setText("")
    win._on_zegrid_checkpoint_store_edited()
    win._collect_config_from_widgets()
    assert win._zegrid_checkpoint_store_root() is None


# ---------------------------------------------------------------------------
# 2. Read-only classification (zero/one/multiple/complete/corrupt/legacy/locked)
# ---------------------------------------------------------------------------


def test_candidate_classification_is_read_only_and_honest(tmp_path, monkeypatch):
    root = tmp_path / "store"
    root.mkdir()

    # zero: a merely existing directory is NOT a candidate
    win = _window(tmp_path, monkeypatch, {})
    assert win._classify_zegrid_durable_candidates(str(root))["candidates"] == []

    # one incomplete + one complete generation
    m4.seeded_store(
        root / "partial",
        identity=m4.make_identity(run_id="run-partial"),
        cell_indexes=(0, 1),
    )
    _make_complete(root / "done", run_id="run-done")

    # legacy: valid run.json but no frozen state.json
    legacy = root / "legacy"
    m4.seeded_store(
        legacy, identity=m4.make_identity(run_id="run-legacy"), cell_indexes=(0,)
    )
    (legacy / "state.json").unlink()

    # corrupt identity: not readable JSON
    corrupt = root / "corrupt"
    corrupt.mkdir()
    (corrupt / "run.json").write_text("{not json", encoding="utf-8")

    # live lock: hold the store open in THIS process
    locked = root / "locked"
    m4.seeded_store(
        locked, identity=m4.make_identity(run_id="run-locked"), cell_indexes=(0,)
    )
    held = zrs.RunCellStore.open(locked, identity=None)
    try:
        before = _tree_digest(root)
        info = win._classify_zegrid_durable_candidates(str(root))
        after = _tree_digest(root)
    finally:
        held.close()

    assert before == after, "classification modified the store bytes"
    statuses = {c["run_id"]: c["status"] for c in info["candidates"]}
    assert statuses == {
        "run-partial": "resumable",
        "run-done": "complete",
        "run-legacy": "corrupt",
        None: "corrupt",  # unreadable identity is reported, never offered
        "run-locked": "locked",
    }
    resumable = [c for c in info["candidates"] if c["status"] == "resumable"]
    assert [c["run_id"] for c in resumable] == ["run-partial"]


def test_discovery_error_is_not_a_candidate(tmp_path, monkeypatch):
    win = _window(tmp_path, monkeypatch, {})
    store = tmp_path / "not-a-store"
    store.write_text("not a directory", encoding="utf-8")
    info = win._classify_zegrid_durable_candidates(str(store))
    assert info["candidates"] == []
    assert info["error"]


# ---------------------------------------------------------------------------
# 3. Explicit selection (no newest heuristic) + per-launch plan
# ---------------------------------------------------------------------------


class _FakeButton:
    def __init__(self, token):
        self.token = token


class _FakeMessageBox:
    Question = 1
    AcceptRole = 2
    ActionRole = 3
    Cancel = 4

    click_token = "resume"  # test-controlled

    def __init__(self, *args, **kwargs):
        self._buttons: dict[str, _FakeButton] = {}
        self._clicked = None

    def setIcon(self, *_a, **_k):
        pass

    def setWindowTitle(self, *_a, **_k):
        pass

    def setText(self, text):
        self.text = text

    def setDefaultButton(self, *_a, **_k):
        pass

    def addButton(self, *args):
        if isinstance(args[0], str):
            token = "fresh" if "resume" in self._buttons else "resume"
        else:
            token = "cancel"
        button = _FakeButton(token)
        self._buttons[token] = button
        return button

    def exec(self):
        self._clicked = self._buttons.get(self.click_token)

    def clickedButton(self):
        return self._clicked


def test_multiple_generations_require_explicit_selection(tmp_path, monkeypatch):
    win = _window(tmp_path, monkeypatch, {})
    candidates = [
        {"run_id": "run-a", "status": "resumable", "expected_cells": 4},
        {"run_id": "run-b", "status": "resumable", "expected_cells": 4},
    ]

    monkeypatch.setattr(gui, "QMessageBox", _FakeMessageBox)
    _FakeMessageBox.click_token = "resume"
    seen: dict = {}

    def _get_item(parent, title, label, labels, current, editable):
        seen["labels"] = list(labels)
        return (labels[1], True)

    monkeypatch.setattr(gui.QInputDialog, "getItem", _get_item)
    choice, run_id = win._ask_zegrid_resume_choice(
        candidates, new_run_safe=True, unsafe_reason=None
    )
    assert choice == "resume"
    assert run_id == "run-b"  # the explicitly selected one, not the first
    assert len(seen["labels"]) == 2

    # Cancelling the explicit selector is a cancel, never a silent default.
    monkeypatch.setattr(
        gui.QInputDialog, "getItem", lambda *a, **k: (a[3][0], False)
    )
    assert win._ask_zegrid_resume_choice(
        candidates, new_run_safe=True, unsafe_reason=None
    ) == ("cancel", None)

    # New-run safety failure removes the "Start new" escape hatch.
    _FakeMessageBox.click_token = "cancel"
    assert win._ask_zegrid_resume_choice(
        candidates, new_run_safe=False, unsafe_reason="unsafe"
    ) == ("cancel", None)


def test_resume_plan_reaches_worker_without_persisting_decisions(
    tmp_path, monkeypatch
):
    win = _window(tmp_path, monkeypatch, {})
    store = tmp_path / "store"
    m4.seeded_store(
        store, identity=m4.make_identity(run_id="run-partial"), cell_indexes=(0,)
    )
    win.config["zegrid_durable_store_dir"] = str(store)
    win._zegrid_checkpoint_store_edit.setText(str(store))
    win._on_zegrid_checkpoint_store_edited(user_edit=True)
    # Do NOT call _collect_config_from_widgets here: the folder selection is
    # not a widget in these headless windows and collecting would clear the
    # output folder (which the resume preflight legitimately checks).
    before_config = dict(win.config)
    warnings = _warning_calls(monkeypatch)

    seen: dict = {}

    def _choose(*args, **kwargs):
        seen["asked"] = True
        return ("resume", "run-partial")

    monkeypatch.setattr(win, "_ask_zegrid_resume_choice", _choose)
    assert win._prepare_zegrid_resume_for_start() is True
    assert seen.get("asked") is True
    assert warnings == []
    assert win._pending_zegrid_resume == {
        "mode": "auto",
        "run_id": "run-partial",
        "store_dir": str(store),
    }
    # No launch decision is written into the config (no stale auto later).
    assert win.config["zegrid_resume_mode"] == before_config["zegrid_resume_mode"]
    assert win.config["zegrid_resume_run_id"] == before_config["zegrid_resume_run_id"]

    _args, worker_kwargs = win._build_worker_invocation(skip_filter_ui=True)
    assert worker_kwargs["zegrid_durable_store_dir"] == str(store)
    assert worker_kwargs["zegrid_resume_mode"] == "auto"
    assert worker_kwargs["zegrid_resume_run_id"] == "run-partial"


def test_stale_persisted_auto_never_overrides_a_fresh_start(tmp_path, monkeypatch):
    win = _window(
        tmp_path,
        monkeypatch,
        {
            "zegrid_resume_mode": "auto",
            "zegrid_resume_run_id": "stale-run",
            "zegrid_durable_store_dir": str(tmp_path / "out" / ".zegrid-durable"),
            "zegrid_qt_save_checkpoints_enabled": False,
        },
    )
    # A complete generation exists: it is not offered as unfinished.
    _make_complete(
        Path(win.config["zegrid_durable_store_dir"]), run_id="stale-run"
    )
    assert win._prepare_zegrid_resume_for_start() is True
    assert win._pending_zegrid_resume == {
        "mode": "never",
        "run_id": None,
        "store_dir": None,
    }
    _args, worker_kwargs = win._build_worker_invocation(skip_filter_ui=True)
    assert worker_kwargs["zegrid_resume_mode"] == "never"
    assert worker_kwargs["zegrid_resume_run_id"] is None
    assert worker_kwargs["zegrid_durable_store_dir"] is None


def test_cancel_prevents_any_spawn_plan(tmp_path, monkeypatch):
    win = _window(tmp_path, monkeypatch, {})
    store = tmp_path / "store"
    m4.seeded_store(
        store, identity=m4.make_identity(run_id="run-partial"), cell_indexes=(0,)
    )
    win.config["zegrid_durable_store_dir"] = str(store)
    monkeypatch.setattr(
        win, "_ask_zegrid_resume_choice", lambda *a, **k: ("cancel", None)
    )
    assert win._prepare_zegrid_resume_for_start() is False


# ---------------------------------------------------------------------------
# 4. Signature / transport contract
# ---------------------------------------------------------------------------


def test_durable_signature_and_transport_contract():
    import inspect

    sig = inspect.signature(zw.run_hierarchical_mosaic)
    for key in DURABLE_KEYS:
        assert key in sig.parameters, key
        assert sig.parameters[key].default is zw._ZEGRID_POLICY_UNSET, key
    assert "zegrid_stop_channel" in sig.parameters
    assert sig.parameters["zegrid_stop_channel"].default is None

    process_sig = inspect.signature(zw.run_hierarchical_mosaic_process)
    assert any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in process_sig.parameters.values()
    )


def test_explicit_durable_transport_reaches_engine_over_disk_config(
    tmp_path, monkeypatch
):
    win = _window(
        tmp_path, monkeypatch, {"zegrid_qt_save_checkpoints_enabled": True}
    )
    assert win._prepare_zegrid_resume_for_start() is True
    _args, worker_kwargs = win._build_worker_invocation(skip_filter_ui=True)
    disk = _disk_config(
        zegrid_durable_store_dir="/nonexistent/disk-store",
        zegrid_resume_mode="auto",
        zegrid_resume_run_id="disk-run",
    )
    captured = _drive(monkeypatch, tmp_path, worker_kwargs, disk)
    zconfig = captured["zconfig"]
    # ZM-ZEGRID-VISIBLE-ARTIFACTS-20261031: the default store folder is VISIBLE.
    assert zconfig.zegrid_durable_store_dir == str(
        tmp_path / "out" / "zegrid-durable"
    )
    assert zconfig.zegrid_resume_mode == "never"
    assert zconfig.zegrid_resume_run_id is None
    assert captured["stop_channel"] is None  # no channel for an in-process call


# ---------------------------------------------------------------------------
# 5. Real spawn transport + controller state machine
# ---------------------------------------------------------------------------


def _child_ack(channel):
    """Child side: wait for the request, then acknowledge it."""
    channel.wait_request(10.0)
    channel.acknowledge("STOPPED_RESUMABLE", "child ack")


def test_spawn_stop_channel_crosses_a_real_spawn_boundary():
    ctx = multiprocessing.get_context("spawn")
    channel = zrr.SpawnStopChannel.create(ctx)
    process = ctx.Process(target=_child_ack, args=(channel,), daemon=True)
    process.start()
    try:
        assert channel.request("parent request") is True
        assert channel.wait_ack(15.0) is True
        snap = channel.snapshot()
        assert snap["transport"] == "spawn"
        assert snap["requested"] is True
        assert snap["ack_state"] == "STOPPED_RESUMABLE"
        assert snap["ack_detail"] == "child ack"
    finally:
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)


class _FakeChannel:
    def __init__(self):
        self.requests = 0
        self.state = None

    def request(self, reason="stop"):
        self.requests += 1
        return self.requests == 1

    def snapshot(self):
        return self.state


def test_controller_request_is_idempotent_and_never_terminates(tmp_path, monkeypatch):
    win = _window(tmp_path, monkeypatch, {})
    controller = win.worker_controller
    fake = _FakeChannel()
    controller._stop_channel = fake
    monkeypatch.setattr(controller, "is_running", lambda: True)

    assert controller.request_cooperative_stop() == "requested"
    assert controller.request_cooperative_stop() == "already_requested"
    assert fake.requests == 2  # both calls reach the channel; only the first saves

    controller._stop_channel = None
    assert controller.request_cooperative_stop() == "unavailable"


def _save_stop_window(tmp_path, monkeypatch, ack_state, *, alive=True):
    win = _window(tmp_path, monkeypatch, {})
    win.is_processing = True
    win._active_run_is_zegrid = True
    win._pending_run_is_zegrid = True
    logs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win, "_append_log", lambda text, level="info": logs.append((level, text))
    )
    snaps = {
        "value": {
            "requested": True,
            "acknowledged": ack_state is not None,
            "ack_state": ack_state or "",
            "ack_detail": "detail",
        }
    }
    monkeypatch.setattr(
        win.worker_controller,
        "cooperative_stop_snapshot",
        lambda: snaps["value"],
    )
    monkeypatch.setattr(
        win.worker_controller, "is_running", lambda: alive
    )
    monkeypatch.setattr(
        win.worker_controller,
        "request_cooperative_stop",
        lambda *a, **k: "requested",
    )
    return win, logs, snaps


def test_stop_and_save_reports_only_the_acknowledged_state(
    tmp_path, monkeypatch
):
    win, logs, snaps = _save_stop_window(
        tmp_path, monkeypatch, "STOPPED_RESUMABLE"
    )
    win._on_quit_save_clicked()
    assert win._save_stop_state in ("stopping", "resumable")
    win._poll_zegrid_save_stop()
    assert win._save_stop_state == "resumable"
    assert any("committed and resumable" in text for _lvl, text in logs)
    assert not any("not saved" in text.lower() for _lvl, text in logs)

    win, logs, snaps = _save_stop_window(
        tmp_path, monkeypatch, "STOPPED_IN_MEMORY"
    )
    win._on_quit_save_clicked()
    win._poll_zegrid_save_stop()
    assert win._save_stop_state == "not_resumable"
    assert any("NOT resumable" in text for _lvl, text in logs)


def test_exit_without_ack_is_reported_as_not_saved(tmp_path, monkeypatch):
    win, logs, snaps = _save_stop_window(tmp_path, monkeypatch, None, alive=False)
    win._save_stop_state = "stopping"
    win._save_stop_ack_state = None
    win._poll_zegrid_save_stop()
    assert win._save_stop_state == "error"
    assert any("NOT saved" in text for _lvl, text in logs)


def test_unavailable_channel_never_claims_a_stop(tmp_path, monkeypatch):
    win = _window(tmp_path, monkeypatch, {})
    win.is_processing = True
    win._active_run_is_zegrid = True
    logs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win, "_append_log", lambda text, level="info": logs.append((level, text))
    )
    monkeypatch.setattr(
        win.worker_controller, "request_cooperative_stop", lambda *a, **k: "unavailable"
    )
    win._on_quit_save_clicked()
    assert win._save_stop_state != "resumable"
    assert any(lvl == "error" for lvl, _t in logs)
    assert not any("saved" in text.lower() and "not" not in text.lower() for _l, text in logs)


def _close_harness(tmp_path, monkeypatch, choice):
    from PySide6.QtGui import QCloseEvent

    win = _window(tmp_path, monkeypatch, {})
    win.is_processing = True
    win._active_run_is_zegrid = True
    win._pending_run_is_zegrid = True
    win._save_stop_state = "stopping"
    logs: list[tuple[str, str]] = []
    stops: list[bool] = []
    waits: list[float] = []
    discarded: list[bool] = []
    monkeypatch.setattr(
        win, "_append_log", lambda text, level="info": logs.append((level, text))
    )
    monkeypatch.setattr(win, "_collect_config_from_widgets", lambda: None)
    monkeypatch.setattr(win, "_save_config", lambda: None)
    monkeypatch.setattr(win, "_record_splitter_states", lambda: None)
    monkeypatch.setattr(win, "_record_window_geometry", lambda: None)
    monkeypatch.setattr(win, "_apply_astap_concurrency_setting", lambda: None)
    monkeypatch.setattr(
        win, "_discard_pending_worker_start", lambda: discarded.append(True)
    )
    monkeypatch.setattr(win, "_ask_zegrid_close_choice", lambda: choice)
    monkeypatch.setattr(
        win.worker_controller,
        "stop",
        lambda *, graceful=False, wait_timeout=4.0: stops.append(graceful),
    )
    monkeypatch.setattr(
        win.worker_controller,
        "wait_for_exit",
        lambda timeout: waits.append(float(timeout)) or False,
    )
    monkeypatch.setattr(win.worker_controller, "is_running", lambda: True)
    monkeypatch.setattr(
        win.worker_controller,
        "cooperative_stop_snapshot",
        lambda: {
            "requested": True,
            "acknowledged": False,
            "ack_state": "",
            "ack_detail": "",
        },
    )
    return win, QCloseEvent, logs, stops, waits, discarded


def test_close_during_zegrid_save_defers_without_blocking_or_killing(
    tmp_path, monkeypatch
):
    win, QCloseEvent, logs, stops, waits, discarded = _close_harness(
        tmp_path, monkeypatch, "save"
    )
    event = QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted() is False
    assert win._close_after_safe_stop is True
    assert win._closing is False  # the window is still alive/responsive
    assert waits == []  # NO blocking join on the GUI thread
    assert stops == []  # NO automatic kill
    assert any("Close deferred" in text for _lvl, text in logs)


def test_deferred_close_waits_for_ack_and_worker_exit(tmp_path, monkeypatch):
    win, _QCloseEvent, _logs, stops, waits, _disc = _close_harness(
        tmp_path, monkeypatch, "save"
    )
    closed: list[bool] = []
    monkeypatch.setattr(win, "close", lambda: closed.append(True))
    win._close_after_safe_stop = True
    win._save_stop_ack_state = "STOPPED_RESUMABLE"
    running = {"value": True}
    monkeypatch.setattr(
        win.worker_controller, "is_running", lambda: running["value"]
    )
    # Ack arrived but the worker is still alive: still wait, never kill.
    win._maybe_complete_deferred_close()
    win._complete_deferred_close()
    assert closed == []
    assert stops == []
    assert waits == []
    running["value"] = False
    win._complete_deferred_close()
    assert closed == [True]
    assert win._close_after_safe_stop is False


def test_deferred_close_without_ack_is_not_saved_but_closes(tmp_path, monkeypatch):
    win, _Q, logs, stops, waits, _disc = _close_harness(
        tmp_path, monkeypatch, "save"
    )
    closed: list[bool] = []
    monkeypatch.setattr(win, "close", lambda: closed.append(True))
    monkeypatch.setattr(win.worker_controller, "is_running", lambda: False)
    win._save_stop_ack_state = None
    win._close_after_safe_stop = True
    win._save_stop_state = "stopping"
    win.is_processing = True
    win._poll_zegrid_save_stop()
    assert win._save_stop_state == "error"
    assert any("NOT saved" in text for _lvl, text in logs)
    win._complete_deferred_close()
    assert closed == [True]
    assert stops == [] and waits == []


def test_close_force_choice_is_explicit_and_never_claims_saved(
    tmp_path, monkeypatch
):
    win, QCloseEvent, logs, stops, _waits, discarded = _close_harness(
        tmp_path, monkeypatch, "force"
    )
    event = QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted() is True
    assert stops == [False]  # explicit force, distinct from Stop and Save
    assert discarded == [True]
    assert win._closing is True
    assert any("NOT reported as saved" in text for _lvl, text in logs)


def test_close_cancel_keeps_the_run_alive(tmp_path, monkeypatch):
    win, QCloseEvent, logs, stops, waits, _disc = _close_harness(
        tmp_path, monkeypatch, "cancel"
    )
    event = QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted() is False
    assert stops == [] and waits == []
    assert win._closing is False
    assert win.is_processing is True
    assert any("Close cancelled" in text for _lvl, text in logs)


def test_close_legacy_mode_keeps_graceful_sigterm_path(tmp_path, monkeypatch):
    win = _window(tmp_path, monkeypatch, {})
    win.is_processing = True
    win._active_run_is_zegrid = False
    win._pending_run_is_zegrid = False
    stops: list[bool] = []
    discarded: list[bool] = []
    for name in (
        "_collect_config_from_widgets",
        "_save_config",
        "_record_splitter_states",
        "_record_window_geometry",
        "_apply_astap_concurrency_setting",
    ):
        monkeypatch.setattr(win, name, lambda: None)
    monkeypatch.setattr(
        win, "_discard_pending_worker_start", lambda: discarded.append(True)
    )
    monkeypatch.setattr(
        win.worker_controller,
        "stop",
        lambda *, graceful=False, wait_timeout=4.0: stops.append(graceful),
    )
    from PySide6.QtGui import QCloseEvent

    event = QCloseEvent()
    win.closeEvent(event)
    assert stops == [True]
    assert discarded == [True]
    assert event.isAccepted() is True
    assert win._closing is True


def test_pending_legacy_spawn_does_not_arm_zegrid_stop(tmp_path, monkeypatch):
    win = _window(tmp_path, monkeypatch, {})
    win.is_processing = False
    win._worker_start_pending = True
    win._pending_run_is_zegrid = False
    logs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win, "_append_log", lambda text, level="info": logs.append((level, text))
    )
    win._on_quit_save_clicked()
    assert win._save_stop_when_spawned is False
    assert any("No processing" in text for _lvl, text in logs)


# ---------------------------------------------------------------------------
# 6. Bounded SYNTHETIC integration proof (tiny corpus; never the user's data)
# ---------------------------------------------------------------------------
#
# GUI-built invocation -> REAL spawned worker -> real engine cooperative stop ->
# a NEW spawned process resumes the SAME store. The proof is deliberately
# synthetic: a tiny deterministic 8-frame/512 px corpus. It does NOT qualify the
# user's real datasets, Windows, or any GPU path.


def _spawn_real_worker(ctx, kwargs, channel=None):
    queue_obj = ctx.Queue()
    kw = dict(kwargs)
    kw["crash_breadcrumbs_mode"] = "off"
    if channel is not None:
        kw["zegrid_stop_channel"] = channel
    process = ctx.Process(
        target=zw.run_hierarchical_mosaic_process,
        args=(queue_obj,),
        kwargs=kw,
        daemon=False,
    )
    process.start()
    return process, queue_obj


def _drain_worker(process, queue_obj, timeout):
    messages = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            item = queue_obj.get(timeout=0.2)
        except queue_mod.Empty:
            if not process.is_alive():
                break
            continue
        messages.append(item)
        if item[0] == "PROCESS_DONE":
            break
    process.join(timeout=20.0)
    return messages


def test_synthetic_gui_built_stop_then_new_process_resume(tmp_path, monkeypatch):
    import re

    import test_zegrid_resume_m0_baseline_witness as m0

    corpus = tmp_path / "corpus"
    inp, names, _hashes = m0._write_corpus(corpus)
    store = tmp_path / "durable-store"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    win = _window(
        tmp_path,
        monkeypatch,
        {
            "zegrid_qt_save_checkpoints_enabled": True,
            "zegrid_durable_store_dir": str(store),
            "zegrid_durable_store_budget_bytes": 2 * 2 ** 30,
            "zegrid_layout": "2x2",
            "zegrid_halo_px": 8,
            "zegrid_cell_gauge_mode": "per_cell",
        },
    )
    # Use the REAL m0 corpus paths for this launch.
    win.config["input_dir"] = str(inp)
    win.config["output_dir"] = str(out_dir)
    assert win._prepare_zegrid_resume_for_start() is True
    _args, worker_kwargs = win._build_worker_invocation(skip_filter_ui=True)
    worker_kwargs["filter_overrides"] = {"processing_mode": "zegrid"}

    assert worker_kwargs["zegrid_durable_store_dir"] == str(store)
    assert worker_kwargs["zegrid_resume_mode"] == "never"
    assert worker_kwargs["zegrid_resume_run_id"] is None

    ctx = multiprocessing.get_context("spawn")
    channel = zrr.SpawnStopChannel.create(ctx)
    process, queue_obj = _spawn_real_worker(ctx, worker_kwargs, channel)

    commits_seen: dict[str, int] = {}

    def _watch_first_commit():
        commits_dir = store / "commits"
        deadline = time.time() + 240
        while time.time() < deadline:
            try:
                count = len(list(commits_dir.iterdir())) if commits_dir.is_dir() else 0
            except OSError:
                count = 0
            if count >= 1:
                commits_seen["n"] = count
                channel.request("synthetic integration: stop after first commit")
                return
            if not process.is_alive():
                return
            time.sleep(0.02)

    watcher = threading.Thread(target=_watch_first_commit, daemon=True)
    watcher.start()
    messages = _drain_worker(process, queue_obj, timeout=300)
    watcher.join(timeout=5.0)

    errors = [m for m in messages if m[0] == "PROCESS_ERROR"]
    assert not errors, errors[:1]
    snap = channel.snapshot()
    assert snap["requested"] is True, "the stop was never requested"
    assert snap["acknowledged"] is True, "the engine never acknowledged the stop"
    assert snap["ack_state"] == "STOPPED_RESUMABLE", snap
    assert commits_seen.get("n"), "no durable commit was ever observed"

    commit_dirs = sorted((store / "commits").iterdir())
    commits_at_stop = len(commit_dirs)
    assert 1 <= commits_at_stop <= 3, commits_at_stop  # partial, not the whole run
    state = json.loads((store / "state.json").read_text(encoding="utf-8"))
    assert state["finalized"] is False
    assert not list(out_dir.glob(".zegrid_publications/*/CURRENT.json")), (
        "publication happened despite the stop"
    )

    run_id = json.loads((store / "run.json").read_text(encoding="utf-8"))["run_id"]
    objects_before = _tree_digest(store / "objects")

    def _committed_cells():
        out = {}
        for path in sorted((store / "commits").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            out[str(payload["cell_id"])] = {
                "attempt_id": payload.get("attempt_id"),
                "created_unix_ns": payload.get("created_unix_ns"),
            }
        return out

    committed_at_stop = _committed_cells()
    expected_ids = set(state["expected_cell_ids"])
    pending_ids = expected_ids - set(committed_at_stop)
    assert len(committed_at_stop) == commits_at_stop

    # --- EXECUTION-LEVEL observer proof (transparent forwarding) -------------
    # The engine's own progress labels are not sufficient evidence. Run the
    # REAL engine resume once in-process on a COPY of the partial store with a
    # wrapper around the real per-cell worker that records the cell id and then
    # forwards the call UNCHANGED. This proves committed cells are not executed.
    import shutil as _shutil

    obs_store = tmp_path / "observed-store"
    obs_out = tmp_path / "observed-out"
    _shutil.copytree(store, obs_store)
    _shutil.copytree(out_dir, obs_out)
    observed: list[str] = []
    real_stack_cell = zz._stack_cell

    def _observing_stack_cell(task, *args, **kwargs):
        try:
            observed.append(str(task[0].cell_id))
        except Exception:
            pass
        return real_stack_cell(task, *args, **kwargs)

    # Use the REAL production dispatcher (the same function the spawned worker
    # runs) with the GUI-built kwargs, so the science contract is identical to
    # the actual launch. No identity check is weakened and no science parameter
    # is reconstructed by hand. The only change is output/store pointing at the
    # COPY, and an explicit resume of the same generation.
    obs_kwargs = dict(worker_kwargs)
    obs_kwargs["input_dir"] = str(inp)
    obs_kwargs["output_dir"] = str(obs_out)
    obs_kwargs["zegrid_durable_store_dir"] = str(obs_store)
    obs_kwargs["zegrid_resume_mode"] = "auto"
    obs_kwargs["zegrid_resume_run_id"] = run_id
    obs_kwargs["filter_overrides"] = {"processing_mode": "zegrid"}
    obs_kwargs["crash_breadcrumbs_mode"] = "off"

    class _NullQueue:
        def put(self, *_args, **_kwargs):
            return None

    try:
        zz._stack_cell = _observing_stack_cell
        zw.run_hierarchical_mosaic_process(_NullQueue(), **obs_kwargs)
    finally:
        zz._stack_cell = real_stack_cell
    assert observed, "the observer never reached the real per-cell worker"
    assert set(observed) == pending_ids, (observed, pending_ids)
    assert observed and set(observed).isdisjoint(committed_at_stop)

    # --- GUI discovery + selection -> SECOND REAL SPAWNED WORKER -------------
    # REOPEN: a fresh window performs the REAL read-only discovery + explicit
    # selection, and the GUI builds the second worker invocation (no manual
    # resume kwargs).
    win2 = _window(
        tmp_path,
        monkeypatch,
        {
            "zegrid_qt_save_checkpoints_enabled": True,
            "zegrid_durable_store_dir": str(store),
            "zegrid_durable_store_budget_bytes": 2 * 2 ** 30,
            "zegrid_layout": "2x2",
            "zegrid_halo_px": 8,
            "zegrid_cell_gauge_mode": "per_cell",
        },
    )
    win2.config["input_dir"] = str(inp)
    win2.config["output_dir"] = str(out_dir)

    info = win2._classify_zegrid_durable_candidates(str(store))
    resumable = [c for c in info["candidates"] if c["status"] == "resumable"]
    assert [c["run_id"] for c in resumable] == [run_id]

    # Drive the REAL dialog code: click Resume… and pick the discovered run.
    monkeypatch.setattr(gui, "QMessageBox", _FakeMessageBox)
    _FakeMessageBox.click_token = "resume"
    monkeypatch.setattr(
        gui.QInputDialog,
        "getItem",
        lambda parent, title, label, labels, current, editable: (labels[0], True),
    )
    assert win2._prepare_zegrid_resume_for_start() is True
    assert win2._pending_zegrid_resume == {
        "mode": "auto",
        "run_id": run_id,
        "store_dir": str(store),
    }
    _args2, resume_kwargs = win2._build_worker_invocation(skip_filter_ui=True)
    resume_kwargs["filter_overrides"] = {"processing_mode": "zegrid"}
    assert resume_kwargs["zegrid_resume_mode"] == "auto"
    assert resume_kwargs["zegrid_resume_run_id"] == run_id
    assert resume_kwargs["zegrid_durable_store_dir"] == str(store)

    process2, queue_obj2 = _spawn_real_worker(ctx, resume_kwargs)
    messages2 = _drain_worker(process2, queue_obj2, timeout=300)
    errors2 = [m for m in messages2 if m[0] == "PROCESS_ERROR"]
    assert not errors2, errors2[:1]

    plan_msgs = [
        str(m[0]) for m in messages2 if "resume plan" in str(m[0])
    ]
    assert plan_msgs, messages2[-5:]
    match = re.search(r"complete=(\d+).*pending=(\d+)", plan_msgs[0])
    assert match, plan_msgs[0]
    complete, pending = int(match.group(1)), int(match.group(2))
    assert complete == commits_at_stop
    assert pending == len(pending_ids)

    # The pre-existing commit records are untouched (same attempt/time).
    committed_after = _committed_cells()
    for cell_id, record in committed_at_stop.items():
        assert committed_after.get(cell_id) == record, cell_id

    # Committed payload bytes are untouched (no recomputation/rewrite).
    objects_after = _tree_digest(store / "objects")
    for rel, digest in objects_before.items():
        assert objects_after.get(rel) == digest, rel

    # Final publication is valid and complete (generation-resolved, verified).
    generation = zpub.resolve_current(str(out_dir), run_id)
    science = generation.path(zpub.ROLE_SCIENCE)
    assert science.is_file()
    manifest = json.loads(
        generation.path(zpub.ROLE_MANIFEST).read_text(encoding="utf-8")
    )
    assert manifest["engine"] == "zegrid"
    assert manifest["complete_cells"]
    final_state = json.loads((store / "state.json").read_text(encoding="utf-8"))
    assert final_state["finalized"] is True


def test_stop_and_save_while_spawn_is_pending_is_armed_not_lost(
    tmp_path, monkeypatch
):
    win = _window(tmp_path, monkeypatch, {})
    win.is_processing = False
    win._worker_start_pending = True
    win._pending_run_is_zegrid = True
    calls: list[str] = []
    logs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win,
        "_append_log",
        lambda text, level="info": logs.append((level, text)),
    )
    monkeypatch.setattr(
        win.worker_controller,
        "request_cooperative_stop",
        lambda *a, **k: calls.append("request") or "requested",
    )
    win._on_quit_save_clicked()
    assert win._save_stop_when_spawned is True
    assert calls == []  # nothing is running yet; nothing is terminated either

    # Finalization of the pending spawn delivers the armed request.
    win._pending_run_is_zegrid = True
    monkeypatch.setattr(win.worker_controller, "is_running", lambda: True)
    monkeypatch.setattr(
        win.worker_controller, "cooperative_stop_snapshot", lambda: None
    )
    win._finalize_successful_worker_start()
    assert calls == ["request"]
    assert win._save_stop_when_spawned is False
    assert win._save_stop_state == "stopping"
    win._save_stop_timer.stop()


# ---------------------------------------------------------------------------
# 7. Fail-closed fresh-launch preflight (F2 store, F3 output/publication)
# ---------------------------------------------------------------------------


def _warning_calls(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        gui.QMessageBox,
        "warning",
        lambda *a, **k: calls.append(str(a[-1]) if a else ""),
    )
    return calls


def _prepare_fresh(win, monkeypatch, *, enabled=True, choice=("cancel", None)):
    # The toggled signal is the real widget -> config path. Do NOT call
    # _collect_config_from_widgets here: the folder selection is not a widget
    # in these headless windows and collecting would clear it.
    win._zegrid_save_checkpoints_checkbox.setChecked(bool(enabled))
    assert win.config["zegrid_qt_save_checkpoints_enabled"] is bool(enabled)
    win._pending_zegrid_resume = None
    calls = _warning_calls(monkeypatch)
    monkeypatch.setattr(win, "_ask_zegrid_resume_choice", lambda *a, **k: choice)
    ok = win._prepare_zegrid_resume_for_start()
    return ok, calls


def _set_store(win, path):
    win.config["zegrid_durable_store_dir"] = str(path)
    win._zegrid_checkpoint_store_edit.setText(str(path))
    win._on_zegrid_checkpoint_store_edited(user_edit=True)


def test_fresh_checkpoint_store_preflight_blocks_unsafe_paths(tmp_path, monkeypatch):
    win = _window(tmp_path, monkeypatch, {})

    # Fixtures for the refusal cases are created FIRST; the digest below is the
    # state a refusal must leave byte-identical.
    out_file = tmp_path / "out-file"
    out_file.write_text("x", encoding="utf-8")
    file_store = tmp_path / "store-file"
    file_store.write_text("x", encoding="utf-8")
    real = tmp_path / "real-store"
    real.mkdir()
    link = tmp_path / "store-link"
    link.symlink_to(real, target_is_directory=True)
    before = _tree_digest(tmp_path)

    # store == output
    out = tmp_path / "out"
    _set_store(win, out)
    ok, calls = _prepare_fresh(win, monkeypatch)
    assert ok is False and calls and win._pending_zegrid_resume is None

    # store inside the input folder
    inp = tmp_path / "in"
    _set_store(win, inp / "store")
    ok, calls = _prepare_fresh(win, monkeypatch)
    assert ok is False and calls and win._pending_zegrid_resume is None

    # existing file as the store path
    _set_store(win, file_store)
    ok, calls = _prepare_fresh(win, monkeypatch)
    assert ok is False and calls and win._pending_zegrid_resume is None

    # symlinked store root
    _set_store(win, link)
    ok, calls = _prepare_fresh(win, monkeypatch)
    assert ok is False and calls and win._pending_zegrid_resume is None

    # output path is an existing FILE
    _set_store(win, tmp_path / "fine-store")
    win.config["output_dir"] = str(out_file)
    ok, calls = _prepare_fresh(win, monkeypatch)
    assert ok is False and calls and win._pending_zegrid_resume is None

    assert _tree_digest(tmp_path) == before, "a refusal created/modified files"


def test_disabled_checkpoint_does_not_block_on_store_but_blocks_bad_output(
    tmp_path, monkeypatch
):
    win = _window(tmp_path, monkeypatch, {})
    out = tmp_path / "out"
    out.mkdir()
    _set_store(win, out)  # would be store==output, but checkpoint saving is OFF
    ok, calls = _prepare_fresh(win, monkeypatch, enabled=False)
    assert ok is True, "disabled checkpoint path must not be judged as a store"
    assert win._pending_zegrid_resume == {
        "mode": "never",
        "run_id": None,
        "store_dir": None,
    }

    # ... but an unsafe OUTPUT still blocks, even with the checkbox OFF.
    out_file = tmp_path / "out-file"
    out_file.write_text("x", encoding="utf-8")
    win.config["output_dir"] = str(out_file)
    ok, calls = _prepare_fresh(win, monkeypatch, enabled=False)
    assert ok is False and calls


def test_fresh_launch_refuses_previous_publication_and_foreign_content(
    tmp_path, monkeypatch
):
    win = _window(tmp_path, monkeypatch, {})
    out = tmp_path / "out"
    out.mkdir()

    # A previous ZeGrid publication generation.
    pub = out / ".zegrid_publications" / "run-old"
    (pub / "gen-1").mkdir(parents=True)
    (pub / "CURRENT.json").write_text(
        json.dumps({"run_id": "run-old", "generation": "gen-1"}),
        encoding="utf-8",
    )
    before = _tree_digest(tmp_path)
    ok, calls = _prepare_fresh(win, monkeypatch, enabled=True)
    assert ok is False and calls and win._pending_zegrid_resume is None
    ok, calls = _prepare_fresh(win, monkeypatch, enabled=False)
    assert ok is False and calls and win._pending_zegrid_resume is None
    assert _tree_digest(tmp_path) == before

    # Foreign user content only (no publication evidence).
    import shutil

    shutil.rmtree(out / ".zegrid_publications")
    (out / "foreign.txt").write_text("mine", encoding="utf-8")
    ok, calls = _prepare_fresh(win, monkeypatch, enabled=True)
    assert ok is False and calls
    ok, calls = _prepare_fresh(win, monkeypatch, enabled=False)
    assert ok is False and calls
    assert (out / "foreign.txt").read_text(encoding="utf-8") == "mine"


def test_gui_owned_output_artifacts_do_not_block_a_retry(tmp_path, monkeypatch):
    win = _window(tmp_path, monkeypatch, {})
    out = tmp_path / "out"
    (out / "__zegrid_cache__").mkdir(parents=True)
    (out / "zegrid_run.log").write_text("log", encoding="utf-8")
    (out / "run_config_snapshot.json").write_text("{}", encoding="utf-8")
    (out / "resource_telemetry.csv").write_text("t", encoding="utf-8")
    (out / "worker_last_state.json").write_text("{}", encoding="utf-8")
    (out / "worker_crash_breadcrumbs.jsonl").write_text("", encoding="utf-8")
    (out / win.ZEGRID_DEFAULT_STORE_DIRNAME).mkdir()  # our default store folder

    ok, calls = _prepare_fresh(win, monkeypatch, enabled=True)
    assert ok is True, calls
    assert win._pending_zegrid_resume == {
        "mode": "never",
        "run_id": None,
        "store_dir": str(out / win.ZEGRID_DEFAULT_STORE_DIRNAME),
    }


def test_complete_corrupt_locked_candidates_cannot_become_overwriting_fresh(
    tmp_path, monkeypatch
):
    win = _window(tmp_path, monkeypatch, {})
    store = tmp_path / "store"
    _make_complete(store, run_id="run-done")
    out = tmp_path / "out"
    out.mkdir()
    pub = out / ".zegrid_publications" / "run-old"
    (pub / "gen-1").mkdir(parents=True)
    (pub / "CURRENT.json").write_text(
        json.dumps({"run_id": "run-old", "generation": "gen-1"}),
        encoding="utf-8",
    )
    _set_store(win, store)
    before = _tree_digest(tmp_path)
    ok, calls = _prepare_fresh(win, monkeypatch)
    assert ok is False and calls
    assert win._pending_zegrid_resume is None
    assert _tree_digest(tmp_path) == before

    # The same complete candidate WITHOUT an output publication is a safe
    # fresh generation target (the engine never overwrites a generation).
    out2 = tmp_path / "out2"
    out2.mkdir()
    win.config["output_dir"] = str(out2)
    ok, calls = _prepare_fresh(win, monkeypatch)
    assert ok is True, calls
    assert win._pending_zegrid_resume["mode"] == "never"


def test_legacy_root_and_container_generation_allocation_never_overwrites(
    tmp_path,
):
    from zemosaic import zemosaic_zegrid_mode as zz

    # Legacy single-run root -> a NEW child generation is allocated.
    legacy = tmp_path / "legacy"
    m4.seeded_store(
        legacy, identity=m4.make_identity(run_id="run-legacy"), cell_indexes=(0,)
    )
    target = zz._resume_generation_root(legacy, "run-new")
    assert target != legacy
    assert legacy in target.parents
    assert not target.exists()

    # Container of generations -> the allocator never reuses a generation path.
    container = tmp_path / "container"
    gen = container / "gen-a"
    m4.seeded_store(
        gen, identity=m4.make_identity(run_id="run-a"), cell_indexes=(0,)
    )
    target2 = zz._resume_generation_root(container, "run-b")
    # A container without a root run.json legitimately opens a new legacy root
    # run; it never reuses/overwrites the existing generation path.
    assert target2 == container
    assert target2 != gen


def test_repeated_stop_and_save_click_does_not_double_request(
    tmp_path, monkeypatch
):
    win = _window(tmp_path, monkeypatch, {})
    win.is_processing = True
    win._active_run_is_zegrid = True
    win._save_stop_state = "stopping"
    win._save_stop_ack_state = None
    requests: list[str] = []
    logs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win, "_append_log", lambda text, level="info": logs.append((level, text))
    )
    monkeypatch.setattr(
        win.worker_controller,
        "request_cooperative_stop",
        lambda *a, **k: requests.append("request") or "requested",
    )
    win._on_quit_save_clicked()
    assert requests == []
    assert any("already pending" in text for _lvl, text in logs)
    win._save_stop_timer.stop()


def test_classic_stop_and_save_keeps_the_legacy_graceful_path(
    tmp_path, monkeypatch
):
    win = _window(tmp_path, monkeypatch, {})
    win.is_processing = True
    win._active_run_is_zegrid = False
    win._pending_run_is_zegrid = False
    stops: list[bool] = []
    requests: list[str] = []
    monkeypatch.setattr(
        win.worker_controller,
        "stop",
        lambda *, graceful=False, wait_timeout=4.0: stops.append(graceful),
    )
    monkeypatch.setattr(
        win.worker_controller,
        "request_cooperative_stop",
        lambda *a, **k: requests.append("request") or "requested",
    )
    monkeypatch.setattr(win, "_append_log", lambda *a, **k: None)
    win._on_quit_save_clicked()
    assert stops == [True]
    assert requests == []
    assert win._save_stop_state == "idle"


# ---------------------------------------------------------------------------
# 8. REWORK-2 R1 — output safety applies to RESUME too
# ---------------------------------------------------------------------------


def _seed_partial(store, run_id="run-partial"):
    m4.seeded_store(
        store, identity=m4.make_identity(run_id=run_id), cell_indexes=(0,)
    )
    return run_id


def _resume_win(tmp_path, monkeypatch, store, run_id, *, enabled=False, out=None):
    win = _window(tmp_path, monkeypatch, {})
    if enabled:
        win._zegrid_save_checkpoints_checkbox.setChecked(True)
    win.config["zegrid_durable_store_dir"] = str(store)
    win._zegrid_checkpoint_store_edit.setText(str(store))
    win._on_zegrid_checkpoint_store_edited(user_edit=True)
    if out is not None:
        win.config["output_dir"] = str(out)
    warnings = _warning_calls(monkeypatch)
    monkeypatch.setattr(
        win, "_ask_zegrid_resume_choice", lambda *a, **k: ("resume", run_id)
    )
    return win, warnings


def test_resume_allows_matching_incomplete_store_in_clean_output(
    tmp_path, monkeypatch
):
    store = tmp_path / "store"
    run_id = _seed_partial(store)
    out = tmp_path / "out"
    out.mkdir()
    (out / "zegrid_run.log").write_text("log", encoding="utf-8")
    win, warnings = _resume_win(tmp_path, monkeypatch, store, run_id, out=out)
    assert win._prepare_zegrid_resume_for_start() is True
    assert warnings == []
    assert win._pending_zegrid_resume == {
        "mode": "auto",
        "run_id": run_id,
        "store_dir": str(store),
    }


def test_resume_refuses_foreign_content_and_preserves_bytes(
    tmp_path, monkeypatch
):
    store = tmp_path / "store"
    run_id = _seed_partial(store)
    out = tmp_path / "out"
    out.mkdir()
    (out / "foreign.txt").write_text("mine", encoding="utf-8")
    before = _tree_digest(tmp_path)
    win, warnings = _resume_win(tmp_path, monkeypatch, store, run_id, out=out)
    assert win._prepare_zegrid_resume_for_start() is False
    assert warnings and "not a ZeGrid publication" in warnings[0]
    assert win._pending_zegrid_resume is None
    assert _tree_digest(tmp_path) == before


def test_resume_checkbox_off_still_checks_output(tmp_path, monkeypatch):
    store = tmp_path / "store"
    run_id = _seed_partial(store)
    out = tmp_path / "out"
    out.mkdir()
    (out / "foreign.txt").write_text("mine", encoding="utf-8")
    win, warnings = _resume_win(
        tmp_path, monkeypatch, store, run_id, enabled=False, out=out
    )
    assert win._prepare_zegrid_resume_for_start() is False
    assert warnings
    assert win._pending_zegrid_resume is None


def test_resume_refuses_unrelated_run_publication(tmp_path, monkeypatch):
    store = tmp_path / "store"
    run_id = _seed_partial(store, "run-selected")
    out = tmp_path / "out"
    other_root = zpub.run_publication_root(out, "run-other")
    (other_root).mkdir(parents=True)
    (other_root / "CURRENT.json").write_text(
        json.dumps({"run_id": "run-other", "generation": "gen-1"}),
        encoding="utf-8",
    )
    before = _tree_digest(tmp_path)
    win, warnings = _resume_win(
        tmp_path, monkeypatch, store, "run-selected", out=out
    )
    assert win._prepare_zegrid_resume_for_start() is False
    # REWORK-4 F2: ownership is checked before pointer parsing, so the refusal
    # may name the unrelated root rather than the pointer; both are refusals.
    assert warnings and (
        "another run" in warnings[0] or "does not belong" in warnings[0]
    )
    assert win._pending_zegrid_resume is None
    assert _tree_digest(tmp_path) == before


def _publication_ram():
    import psutil

    return int(psutil.virtual_memory().available)


def _real_publication(out, run_id):
    import test_zegrid_publication_m5b as pubtest

    pub = pubtest._begin(out, run_id=run_id, ram=_publication_ram())
    generation, *_ = pubtest._write_generation(out, pub)
    return pubtest, pub, generation


def _staged_only_publication(out, run_id, *, dead_pid=2147483647):
    import test_zegrid_publication_m5b as pubtest

    pub = pubtest._begin(out, run_id=run_id, ram=_publication_ram())
    pubtest._stage_generation(out, pub)
    run_root = zpub.run_publication_root(out, run_id)
    markers = list(run_root.glob("*/.staging.json"))
    assert len(markers) == 1
    payload = json.loads(markers[0].read_text(encoding="utf-8"))
    payload["pid"] = int(dead_pid)
    markers[0].write_text(json.dumps(payload), encoding="utf-8")
    assert not list(run_root.glob("CURRENT.json"))
    return pubtest, pub, markers[0], run_root


def test_resume_allows_real_verified_same_run_publication_and_refuses_corruption(
    tmp_path, monkeypatch
):
    store = tmp_path / "store"
    run_id = _seed_partial(store, "run-partial")
    out = tmp_path / "out"
    out.mkdir()
    pubtest, pub, generation = _real_publication(out, run_id)
    verified = zpub.resolve_current(str(out), run_id)  # REAL verifier, no mock
    assert Path(verified.generation_dir) == Path(generation.generation_dir)

    win, warnings = _resume_win(tmp_path, monkeypatch, store, run_id, out=out)
    before = _tree_digest(tmp_path)
    assert win._prepare_zegrid_resume_for_start() is True, warnings
    assert warnings == []
    assert win._pending_zegrid_resume["mode"] == "auto"
    assert _tree_digest(tmp_path) == before, "preflight mutated the output"

    # REAL corruption of a real staged member -> GUI refusal via resolve_current.
    science = Path(verified.generation_dir) / pub.members[zpub.ROLE_SCIENCE]
    science.write_bytes(science.read_bytes() + b"CORRUPTION")
    before = _tree_digest(tmp_path)
    win2, warnings2 = _resume_win(
        tmp_path, monkeypatch, store, run_id, out=out
    )
    assert win2._prepare_zegrid_resume_for_start() is False
    assert warnings2 and "could not be verified" in warnings2[0]
    assert win2._pending_zegrid_resume is None
    assert _tree_digest(tmp_path) == before


def test_resume_allows_same_run_staged_only_dead_owner_crash(
    tmp_path, monkeypatch
):
    store = tmp_path / "store"
    run_id = _seed_partial(store, "run-partial")
    out = tmp_path / "out"
    out.mkdir()
    pubtest, pub, marker, run_root = _staged_only_publication(out, run_id)
    assert not zrs.pid_alive(2147483647)
    # Engine authority still says: NOT published, staged bytes are not COMPLETE.
    verdict = zpub.verify_finalized(out, store, run_id)
    assert verdict.published is False and verdict.reason == "no_publication"

    info = zpub.inspect_same_run_staging(out, run_id)
    assert info["ok"] is True, info
    assert info["generations"]

    before = _tree_digest(tmp_path)
    win, warnings = _resume_win(tmp_path, monkeypatch, store, run_id, out=out)
    assert win._prepare_zegrid_resume_for_start() is True, warnings
    assert warnings == []
    assert win._pending_zegrid_resume["mode"] == "auto"
    assert _tree_digest(tmp_path) == before

    # A complete-looking record WITHOUT a pointer stays staged (never COMPLETE)
    # and is still resumable when the marker proves same-run dead ownership.
    (Path(marker).parent / "publication.json").write_text("{}", encoding="utf-8")
    assert zpub.verify_finalized(out, store, run_id).published is False
    info2 = zpub.inspect_same_run_staging(out, run_id)
    assert info2["ok"] is True, info2


def test_resume_refuses_unsafe_staging_negatives_without_mutation(
    tmp_path, monkeypatch
):
    import os as _os

    def _fresh(tmp_name):
        base = tmp_path / tmp_name
        base.mkdir()
        store = base / "store"
        run_id = _seed_partial(store, "run-partial")
        out = base / "out"
        out.mkdir()
        pubtest, pub, marker, run_root = _staged_only_publication(out, run_id)
        return base, store, run_id, out, pub, marker, run_root

    # live owner
    base, store, run_id, out, pub, marker, root = _fresh("live")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["pid"] = _os.getpid()
    marker.write_text(json.dumps(payload), encoding="utf-8")
    before = _tree_digest(base)
    win, warnings = _resume_win(base, monkeypatch, store, run_id, out=out)
    assert win._prepare_zegrid_resume_for_start() is False
    assert warnings and "LIVE" in warnings[0]
    assert _tree_digest(base) == before

    # missing staging marker
    base, store, run_id, out, pub, marker, root = _fresh("nomarker")
    marker.unlink()
    assert zpub.inspect_same_run_staging(out, run_id)["ok"] is False
    before = _tree_digest(base)
    win, warnings = _resume_win(base, monkeypatch, store, run_id, out=out)
    assert win._prepare_zegrid_resume_for_start() is False
    assert warnings and "proven" in warnings[0]
    assert _tree_digest(base) == before

    # foreign content inside the staging generation
    base, store, run_id, out, pub, marker, root = _fresh("foreign")
    (Path(marker).parent / "foreign.txt").write_text("x", encoding="utf-8")
    before = _tree_digest(base)
    win, warnings = _resume_win(base, monkeypatch, store, run_id, out=out)
    assert win._prepare_zegrid_resume_for_start() is False
    assert warnings and "foreign content" in warnings[0]
    assert _tree_digest(base) == before

    # foreign sibling in the run publication root
    base, store, run_id, out, pub, marker, root = _fresh("sibling")
    (root / "notes.txt").write_text("x", encoding="utf-8")
    assert zpub.inspect_same_run_staging(out, run_id)["ok"] is False

    # another run's publication coexisting under .zegrid_publications
    base, store, run_id, out, pub, marker, root = _fresh("otherrun")
    other = zpub.discover_publication_root(out)[0] / "run-other"
    other.mkdir()
    info = zpub.inspect_same_run_staging(out, run_id)
    assert info["ok"] is False and "unrelated run" in info["reason"]

    # symlinked declared member
    base, store, run_id, out, pub, marker, root = _fresh("link")
    gen = Path(marker).parent
    member = gen / pub.members[zpub.ROLE_SCIENCE]
    target = base / "outside-science.fits"
    target.write_bytes(member.read_bytes())
    member.unlink()
    member.symlink_to(target)
    info = zpub.inspect_same_run_staging(out, run_id)
    assert info["ok"] is False and "plain file" in info["reason"]


def test_nested_store_rejects_hidden_unrelated_publication(tmp_path):
    ancestor = tmp_path / "a"
    (ancestor / "b" / "store").mkdir(parents=True)
    assert (
        gui.ZeMosaicQtMainWindow._zegrid_subtree_only_store(
            ancestor, ("b", "store")
        )
        is True
    )
    foreign = ancestor / "b" / ".zegrid_publications"
    foreign.mkdir()
    (foreign / "foreign.txt").write_text("not this run", encoding="utf-8")
    assert (
        gui.ZeMosaicQtMainWindow._zegrid_subtree_only_store(
            ancestor, ("b", "store")
        )
        is False
    )


def test_resume_refuses_root_level_legacy_publication(tmp_path, monkeypatch):
    store = tmp_path / "store"
    run_id = _seed_partial(store)
    out = tmp_path / "out"
    out.mkdir()
    (out / "zegrid_manifest.json").write_text(
        json.dumps({"engine": "zegrid"}), encoding="utf-8"
    )
    (out / "mosaic_grid_science.fits").write_bytes(b"fits")
    win, warnings = _resume_win(tmp_path, monkeypatch, store, run_id, out=out)
    assert win._prepare_zegrid_resume_for_start() is False
    assert warnings
    assert (out / "mosaic_grid_science.fits").read_bytes() == b"fits"


def test_nested_store_ancestor_does_not_hide_unrelated_siblings(
    tmp_path, monkeypatch
):
    out = tmp_path / "out"
    store = out / "a" / "b" / "store"
    run_id = _seed_partial(store)
    (out / "a" / "foreign.txt").write_text("mine", encoding="utf-8")

    # Resume refuses: the ancestor "a" carries unrelated content.
    win, warnings = _resume_win(tmp_path, monkeypatch, store, run_id, out=out)
    assert win._prepare_zegrid_resume_for_start() is False
    assert warnings and "a" in warnings[0]
    assert win._pending_zegrid_resume is None

    # With ONLY the store chain, the same nested store is legitimate.
    (out / "a" / "foreign.txt").unlink()
    win2, warnings2 = _resume_win(tmp_path, monkeypatch, store, run_id, out=out)
    assert win2._prepare_zegrid_resume_for_start() is True
    assert warnings2 == []

    # Fresh checkpoint-enabled launch: same distinction.
    (out / "a" / "foreign.txt").write_text("mine", encoding="utf-8")
    win3 = _window(tmp_path, monkeypatch, {})
    _set_store(win3, store)
    win3.config["output_dir"] = str(out)
    ok, calls = _prepare_fresh(
        win3, monkeypatch, enabled=True, choice=("fresh", None)
    )
    assert ok is False and calls
    (out / "a" / "foreign.txt").unlink()
    win4 = _window(tmp_path, monkeypatch, {})
    _set_store(win4, store)
    win4.config["output_dir"] = str(out)
    ok, calls = _prepare_fresh(
        win4, monkeypatch, enabled=True, choice=("fresh", None)
    )
    assert ok is True, calls


# ---------------------------------------------------------------------------
# 9. REWORK-2 R2 — pending-spawn save/close latch + ack survival
# ---------------------------------------------------------------------------


def test_pending_zeGrid_close_save_latches_instead_of_losing_intent(
    tmp_path, monkeypatch
):
    from PySide6.QtGui import QCloseEvent

    win = _window(tmp_path, monkeypatch, {})
    win.is_processing = False
    win._worker_start_pending = True
    win._pending_run_is_zegrid = True
    win._active_run_is_zegrid = False
    logs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win, "_append_log", lambda text, level="info": logs.append((level, text))
    )
    monkeypatch.setattr(win, "_ask_zegrid_close_choice", lambda: "save")
    monkeypatch.setattr(
        win.worker_controller, "is_running", lambda: False
    )
    requests: list[str] = []
    monkeypatch.setattr(
        win.worker_controller,
        "request_cooperative_stop",
        lambda *a, **k: requests.append("request") or "unavailable",
    )
    event = QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted() is False
    assert win._close_after_safe_stop is True
    assert win._save_stop_when_spawned is True
    assert requests == []  # no request before the channel exists
    assert any("as soon as it is running" in text for _lvl, text in logs)


def test_pending_close_request_applied_once_when_spawn_attaches(
    tmp_path, monkeypatch
):
    win = _window(tmp_path, monkeypatch, {})
    win.is_processing = False
    win._worker_start_pending = True
    win._pending_run_is_zegrid = True
    win._save_stop_when_spawned = True
    win._close_after_safe_stop = True
    win._worker_start_pending = False
    requests: list[str] = []
    monkeypatch.setattr(win, "_append_log", lambda *a, **k: None)
    monkeypatch.setattr(win.worker_controller, "is_running", lambda: True)
    monkeypatch.setattr(
        win.worker_controller,
        "request_cooperative_stop",
        lambda *a, **k: requests.append("request") or "requested",
    )
    monkeypatch.setattr(
        win.worker_controller,
        "cooperative_stop_snapshot",
        lambda: {
            "requested": True,
            "acknowledged": False,
            "ack_state": "",
            "ack_detail": "",
        },
    )
    win._finalize_successful_worker_start()
    win._set_processing_state = lambda *_a, **_k: None
    assert requests == ["request"]
    assert win._save_stop_when_spawned is False
    assert win._save_stop_state == "stopping"
    assert win._close_after_safe_stop is True
    win._save_stop_timer.stop()


def test_pending_close_spawn_failure_cleans_flags_and_completes_close(
    tmp_path, monkeypatch
):
    win = _window(tmp_path, monkeypatch, {})
    win._worker_start_pending = True
    win._pending_run_is_zegrid = True
    win._save_stop_when_spawned = True
    win._close_after_safe_stop = True
    scheduled: list[str] = []
    monkeypatch.setattr(
        gui.QTimer,
        "singleShot",
        lambda _ms, fn: scheduled.append(getattr(fn, "__name__", "callable")),
    )
    monkeypatch.setattr(win, "_append_log", lambda *a, **k: None)
    win._handle_worker_start_failure(None)
    assert win._save_stop_when_spawned is False
    assert win._close_after_safe_stop is False
    assert scheduled and scheduled[0] == "close"


def test_ack_snapshot_survives_fast_finalize_and_is_reported_saved(
    tmp_path, monkeypatch
):
    win = _window(tmp_path, monkeypatch, {})
    controller = win.worker_controller
    ack = {
        "transport": "spawn",
        "requested": True,
        "acknowledged": True,
        "ack_state": "STOPPED_RESUMABLE",
        "ack_detail": "stop honoured before CPU wave 1",
    }

    class _Channel:
        def snapshot(self):
            return dict(ack)

    controller._stop_channel = _Channel()
    controller._process = None
    controller._queue = None
    controller._finished_emitted = True
    controller._cleanup_listener = lambda: None
    controller._finalize(success=False, message="log_key_processing_cancelled")
    assert controller.cooperative_stop_snapshot()["ack_state"] == "STOPPED_RESUMABLE"

    win.is_processing = True
    win._active_run_is_zegrid = True
    win._save_stop_state = "stopping"
    win._save_stop_ack_state = None
    logs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win, "_append_log", lambda text, level="info": logs.append((level, text))
    )
    monkeypatch.setattr(controller, "is_running", lambda: False)
    win._poll_zegrid_save_stop()
    assert win._save_stop_state == "resumable"
    assert any("committed and resumable" in text for _lvl, text in logs)
    assert not any("NOT saved" in text for _lvl, text in logs)


# ---------------------------------------------------------------------------
# 10. REWORK-2 C — bounded durable-store fail-closed behaviour
# ---------------------------------------------------------------------------


def test_durable_coverage_guard_refuses_degraded_store_object():
    with pytest.raises(zz._R55DurableStoreUnavailable):
        zz._r55_durable_coverage_ready(
            {"store": object(), "publishing": False, "degraded_reason": "nan"}
        )
    with pytest.raises(zz._R55DurableStoreUnavailable):
        zz._r55_durable_coverage_ready(
            {"store": object(), "publishing": True, "degraded_reason": "late"}
        )
    assert zz._r55_durable_coverage_ready(
        {"store": object(), "publishing": True, "degraded_reason": None}
    )
    assert zz._r55_durable_coverage_ready(
        {"store": None, "publishing": False, "degraded_reason": None}
    ) is False


def test_frozen_artifact_write_failure_refuses_before_any_cell(
    tmp_path, monkeypatch
):
    import test_zegrid_resume_m0_baseline_witness as m0

    base = tmp_path / "m0"
    inp, _names, _hashes = m0._write_corpus(base)
    out = tmp_path / "out"
    store = tmp_path / "store"
    messages: list[str] = []

    def _cb(*args, **kwargs):
        try:
            messages.append(str(args[0]))
        except Exception:
            pass

    def _fail_write_state(*args, **kwargs):
        raise zrs.StoreFormatError(
            "value is not strict JSON: Out of range float values are not JSON "
            "compliant: nan"
        )

    executed: list[str] = []
    real_stack = zz._stack_cell

    def _observe(task, *a, **k):
        try:
            executed.append(str(task[0].cell_id))
        except Exception:
            pass
        return real_stack(task, *a, **k)

    monkeypatch.setattr(zz.zres, "write_state", _fail_write_state)
    monkeypatch.setattr(zz, "_stack_cell", _observe)
    cfg = SimpleNamespace(
        zegrid_scheduler_mode="active",
        zegrid_layout="2x2",
        zegrid_halo_px=8,
        zegrid_durable_store_dir=str(store),
        zegrid_durable_store_budget_bytes=2 * 2 ** 30,
        zegrid_resume_mode="never",
        zegrid_resume_run_id=None,
    )
    try:
        with pytest.raises(zz._R55DurableStoreInitError) as exc:
            zz.run_zegrid_mode(
                str(inp),
                str(out),
                progress_callback=_cb,
                zconfig=cfg,
                zegrid_halo_px=8,
                use_gpu=False,
                cell_gauge_mode="per_cell",
            )
    except (zz.zal.LayoutInfeasible, zz.zpar.ModeInfeasible) as resource_exc:
        pytest.skip(f"NOT_RUN: host resource preflight refused ({resource_exc})")
    finally:
        zz._stack_cell = real_stack
    assert "not strict JSON" in str(exc.value)
    assert executed == [], executed  # zero cells admitted
    assert not list(out.glob("mosaic_grid*.fits"))
    assert not any("durable run store ACTIVE" in m for m in messages)
    assert any("FAILED" in m and "before any cell admission" in m for m in messages)
    # The lock/handles were released: a second writer can acquire the store.
    reopened = zrs.RunCellStore.open(store, identity=None)
    reopened.close()


def test_real_routine_12x384_nan_gauge_fails_closed_before_cells(tmp_path):
    """Real tiny 12x384 reproducer (the corpus that exposed the NaN artifact).

    If the host cannot satisfy the engine's own resource preflight, this is
    reported as NOT_RUN (skip) rather than forced.
    """
    import subprocess

    tool = REPO_ROOT / "tools" / "zegrid_routine" / "make_routine_corpus.py"
    corpus = tmp_path / "corpus"
    proc = subprocess.run(
        [sys.executable, str(tool), str(corpus)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    inp = tmp_path / "in"
    inp.mkdir()
    for path in corpus.glob("*.fits"):
        (inp / path.name).write_bytes(path.read_bytes())
    out = tmp_path / "out"
    store = tmp_path / "store"
    executed: list[str] = []
    real_stack = zz._stack_cell

    def _observe(task, *a, **k):
        try:
            executed.append(str(task[0].cell_id))
        except Exception:
            pass
        return real_stack(task, *a, **k)

    cfg = SimpleNamespace(
        zegrid_scheduler_mode="active",
        zegrid_layout="2x2",
        zegrid_halo_px=8,
        zegrid_durable_store_dir=str(store),
        zegrid_durable_store_budget_bytes=2 * 2 ** 30,
        zegrid_resume_mode="never",
        zegrid_resume_run_id=None,
    )
    try:
        zz._stack_cell = _observe
        with pytest.raises(zz._R55DurableStoreInitError) as exc:
            zz.run_zegrid_mode(
                str(inp),
                str(out),
                zconfig=cfg,
                zegrid_halo_px=8,
                use_gpu=False,
                cell_gauge_mode="per_cell",
            )
    except (zz.zal.LayoutInfeasible, zz.zpar.ModeInfeasible) as resource_exc:
        pytest.skip(f"NOT_RUN: host resource preflight refused ({resource_exc})")
    finally:
        zz._stack_cell = real_stack
    assert "not strict JSON" in str(exc.value)
    assert executed == []
    assert not list(out.glob("mosaic_grid*.fits"))
    reopened = zrs.RunCellStore.open(store, identity=None)
    reopened.close()


# ---------------------------------------------------------------------------
# 11. REWORK-4 F1/F2 — ownership is not completeness; both branches check roots
# ---------------------------------------------------------------------------


def _begin_only_dead_staging(out, run_id):
    import test_zegrid_publication_m5b as pubtest

    pub = pubtest._begin(out, run_id=run_id, ram=_publication_ram())
    root = zpub.run_publication_root(out, run_id)
    markers = list(root.glob("*/.staging.json"))
    assert len(markers) == 1
    payload = json.loads(markers[0].read_text(encoding="utf-8"))
    payload["pid"] = 2147483647
    markers[0].write_text(json.dumps(payload), encoding="utf-8")
    return pub, markers[0], root


def test_resume_allows_same_run_staging_before_any_member(tmp_path, monkeypatch):
    store = tmp_path / "store"
    run_id = _seed_partial(store, "run-partial")
    out = tmp_path / "out"
    out.mkdir()
    pub, marker, root = _begin_only_dead_staging(out, run_id)
    assert not list(root.glob("CURRENT.json"))
    assert zpub.verify_finalized(out, store, run_id).reason == "no_publication"

    info = zpub.inspect_same_run_staging(out, run_id)
    assert info["ok"] is True, info
    assert info["present_members"][marker.parent.name] == 0

    before = _tree_digest(tmp_path)
    win, warnings = _resume_win(tmp_path, monkeypatch, store, run_id, out=out)
    assert win._prepare_zegrid_resume_for_start() is True, warnings
    assert warnings == []
    assert win._pending_zegrid_resume["mode"] == "auto"
    assert _tree_digest(tmp_path) == before, "preflight mutated the crash staging"


def test_resume_allows_partial_and_truncated_staged_member(tmp_path, monkeypatch):
    store = tmp_path / "store"
    run_id = _seed_partial(store, "run-partial")
    out = tmp_path / "out"
    out.mkdir()
    pub, marker, root = _begin_only_dead_staging(out, run_id)
    gen = marker.parent
    # A single declared member, zero bytes (truncated mid-write), nothing else.
    pub.path_for(zpub.ROLE_SCIENCE).write_bytes(b"")
    info = zpub.inspect_same_run_staging(out, run_id)
    assert info["ok"] is True, info
    assert info["present_members"][gen.name] == 1
    assert zpub.verify_finalized(out, store, run_id).reason == "no_publication"

    before = _tree_digest(tmp_path)
    win, warnings = _resume_win(tmp_path, monkeypatch, store, run_id, out=out)
    assert win._prepare_zegrid_resume_for_start() is True, warnings
    assert _tree_digest(tmp_path) == before

    # An EXTRA undeclared file is still foreign, even with the subset present.
    (gen / "extra.txt").write_text("x", encoding="utf-8")
    assert zpub.inspect_same_run_staging(out, run_id)["ok"] is False
    before = _tree_digest(tmp_path)
    win2, warnings2 = _resume_win(tmp_path, monkeypatch, store, run_id, out=out)
    assert win2._prepare_zegrid_resume_for_start() is False
    assert warnings2
    assert _tree_digest(tmp_path) == before


def test_publication_record_with_missing_members_is_not_trusted(
    tmp_path, monkeypatch
):
    store = tmp_path / "store"
    run_id = _seed_partial(store, "run-partial")
    out = tmp_path / "out"
    out.mkdir()
    pub, marker, root = _begin_only_dead_staging(out, run_id)
    (marker.parent / "publication.json").write_text("{}", encoding="utf-8")
    info = zpub.inspect_same_run_staging(out, run_id)
    assert info["ok"] is False
    assert "not trusted" in info["reason"]
    before = _tree_digest(tmp_path)
    win, warnings = _resume_win(tmp_path, monkeypatch, store, run_id, out=out)
    assert win._prepare_zegrid_resume_for_start() is False
    assert warnings
    assert _tree_digest(tmp_path) == before


def test_other_run_staging_beside_valid_current_is_refused(
    tmp_path, monkeypatch
):
    import shutil

    import test_zegrid_publication_m5b as pubtest

    store = tmp_path / "store"
    run_id = _seed_partial(store, "run-partial")
    out = tmp_path / "out"
    out.mkdir()
    _real_publication(out, run_id)
    assert zpub.resolve_current(str(out), run_id)  # real selected publication

    pubtest._begin(out, run_id="unrelated-run", ram=_publication_ram())
    ownership = zpub.inspect_publication_root_ownership(out, run_id)
    assert ownership["ok"] is False
    assert "unrelated" in ownership["reason"] or "foreign" in ownership["reason"]

    before = _tree_digest(tmp_path)
    win, warnings = _resume_win(tmp_path, monkeypatch, store, run_id, out=out)
    assert win._prepare_zegrid_resume_for_start() is False
    assert warnings and "does not belong" in warnings[0]
    assert win._pending_zegrid_resume is None
    assert _tree_digest(tmp_path) == before

    # Removing the unrelated run makes the selected verified CURRENT valid again.
    shutil.rmtree(zpub.run_publication_root(out, "unrelated-run"))
    win2, warnings2 = _resume_win(tmp_path, monkeypatch, store, run_id, out=out)
    assert win2._prepare_zegrid_resume_for_start() is True, warnings2
    assert win2._pending_zegrid_resume["mode"] == "auto"
