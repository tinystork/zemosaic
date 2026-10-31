"""ZM-UNIFIED-STOP-SAVE-CAPABILITY — ONE mode-aware Stop and Save capability.

Bounded, focused proof of the capability split:

* ``ZEGRID_DURABLE`` (ZeGrid + scheduler active + checkpoint saving ON) keeps
  the durable plan/store/cooperative-stop path unchanged;
* ``LEGACY_GRACEFUL`` (Classic/SDS/existing-master, ZeGrid legacy/observe,
  ZeGrid active with checkpoints OFF) keeps exactly the historical
  ``worker_controller.stop(graceful=True)`` route, never reads/creates the
  durable store, never offers a resume prompt and never claims a saved or
  resumable run;
* the capability is resolved ONCE before the resume preflight/spawn, frozen
  with the run, and transported to the REAL child process;
* the engine refuses a durable resume request on a non-active scheduler BEFORE
  opening/creating any store or lock (fail-closed, no silent fallback).

No user corpus, no full fast tier, no heavy synthetic run.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import queue as queue_mod
import sys
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
from zemosaic.core.zegrid import scheduler as zsched  # noqa: E402
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402

import test_zegrid_resume_m4 as m4  # noqa: E402
from test_zegrid_scheduler_gui_advanced import _make_window  # noqa: E402

DURABLE = "zegrid_durable"
LEGACY = "legacy_graceful"


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


def _window(tmp_path, monkeypatch, *, mode, scheduler=None, checkpoints=False,
            extra=None, save_store=None):
    config = dict(extra or {})
    if scheduler is not None:
        config["zegrid_scheduler_mode"] = scheduler
    config["zegrid_qt_save_checkpoints_enabled"] = checkpoints
    win = _make_window(tmp_path, monkeypatch, config, save_store=save_store)
    for name, checkbox in win._mode_checkboxes.items():
        checkbox.setChecked(name == mode)
    win._sync_mode_config_keys()
    win.config["input_dir"] = str(tmp_path / "in")
    win.config["output_dir"] = str(tmp_path / "out")
    if scheduler is not None:
        win.config["zegrid_scheduler_mode"] = scheduler
        combo = getattr(win, "_zegrid_scheduler_combo", None)
        if combo is not None:
            index = combo.findData(scheduler)
            if index >= 0:
                combo.setCurrentIndex(index)
        win.config["zegrid_scheduler_mode"] = scheduler
    checkbox = getattr(win, "_zegrid_save_checkpoints_checkbox", None)
    if checkbox is not None:
        checkbox.blockSignals(True)
        checkbox.setChecked(checkpoints)
        checkbox.blockSignals(False)
    win.config["zegrid_qt_save_checkpoints_enabled"] = checkpoints
    return win


def _freeze(win, capability):
    """Mirror the production freeze performed when a spawn attaches."""
    win._pending_stop_capability = capability
    win._pending_run_is_zegrid = capability == win.STOP_CAPABILITY_ZEGRID_DURABLE
    win._active_stop_capability = capability
    win._active_run_is_zegrid = win._pending_run_is_zegrid
    win.is_processing = True
    win._save_stop_state = "idle"
    win._save_stop_ack_state = None


def _tree_digest(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _click_and_capture(win, monkeypatch):
    stops: list[bool] = []
    requests: list[str] = []
    logs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win, "_append_log", lambda text, level="info": logs.append((level, text))
    )
    monkeypatch.setattr(
        win.worker_controller,
        "stop",
        lambda *, graceful=False, wait_timeout=4.0: stops.append(graceful),
    )
    monkeypatch.setattr(
        win.worker_controller,
        "request_cooperative_stop",
        lambda reason="": (requests.append(reason) or "requested"),
    )
    win._on_quit_save_clicked()
    return stops, requests, logs


# ---------------------------------------------------------------------------
# 1. Capability resolution matrix
# ---------------------------------------------------------------------------

_CASES = (
    ("classic", None, False, LEGACY),
    ("classic", None, True, LEGACY),
    ("sds", None, False, LEGACY),
    ("sds", None, True, LEGACY),
    ("existing_master_tiles", None, False, LEGACY),
    ("zegrid", "legacy", False, LEGACY),
    ("zegrid", "legacy", True, LEGACY),
    ("zegrid", "observe", False, LEGACY),
    ("zegrid", "observe", True, LEGACY),
    ("zegrid", "active", False, LEGACY),
    ("zegrid", "active", True, DURABLE),
)


@pytest.mark.parametrize("mode,scheduler,checkpoints,expected", _CASES)
def test_capability_matrix(tmp_path, monkeypatch, mode, scheduler, checkpoints, expected):
    win = _window(
        tmp_path,
        monkeypatch,
        mode=mode,
        scheduler=scheduler,
        checkpoints=checkpoints,
    )
    capability = win._resolve_stop_capability()
    expected_const = (
        win.STOP_CAPABILITY_ZEGRID_DURABLE
        if expected == DURABLE
        else win.STOP_CAPABILITY_LEGACY_GRACEFUL
    )
    assert capability == expected_const


# ---------------------------------------------------------------------------
# 2. The single button routes by the frozen capability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode,scheduler,checkpoints,expected", _CASES)
def test_single_button_routes_with_accurate_text(
    tmp_path, monkeypatch, mode, scheduler, checkpoints, expected
):
    win = _window(
        tmp_path,
        monkeypatch,
        mode=mode,
        scheduler=scheduler,
        checkpoints=checkpoints,
    )
    _freeze(win, win._resolve_stop_capability())
    stops, requests, logs = _click_and_capture(win, monkeypatch)

    if expected == DURABLE:
        assert requests and not stops
        assert "ZeGrid stop requested" in " ".join(text for _lvl, text in logs)
        assert not any("resumable" in text.lower() for _lvl, text in logs)
    else:
        # Exactly the historical generic route: SIGTERM, no channel request.
        assert stops == [True] and not requests
        joined = " ".join(text for _lvl, text in logs).lower()
        assert "saved" not in joined
        assert "resumable" not in joined


def test_capability_is_frozen_against_widget_edits_during_the_run(tmp_path, monkeypatch):
    win = _window(tmp_path, monkeypatch, mode="zegrid", scheduler="active",
                  checkpoints=True)
    _freeze(win, win._resolve_stop_capability())
    assert win._active_capability_is_durable() is True

    # Edit everything that would flip the resolver while the run is live.
    for name, checkbox in win._mode_checkboxes.items():
        checkbox.blockSignals(True)
        checkbox.setChecked(name == "classic")
        checkbox.blockSignals(False)
    win._sync_mode_config_keys()
    win.config["zegrid_scheduler_mode"] = "legacy"
    checkbox = win._zegrid_save_checkpoints_checkbox
    checkbox.blockSignals(True)
    checkbox.setChecked(False)
    checkbox.blockSignals(False)
    win.config["zegrid_qt_save_checkpoints_enabled"] = False

    # A NEW launch would now be generic, but the live run is still durable.
    assert win._resolve_stop_capability() == win.STOP_CAPABILITY_LEGACY_GRACEFUL
    stops, requests, _logs = _click_and_capture(win, monkeypatch)
    assert requests and not stops

    # And the reverse: a frozen generic run stays generic even if widgets would
    # now select the durable capability.
    win2 = _window(tmp_path, monkeypatch, mode="classic")
    _freeze(win2, win2._resolve_stop_capability())
    for name, checkbox in win2._mode_checkboxes.items():
        checkbox.blockSignals(True)
        checkbox.setChecked(name == "zegrid")
        checkbox.blockSignals(False)
    win2._sync_mode_config_keys()
    win2.config["zegrid_scheduler_mode"] = "active"
    win2.config["zegrid_qt_save_checkpoints_enabled"] = True
    assert win2._active_capability_is_durable() is False
    stops2, requests2, _logs2 = _click_and_capture(win2, monkeypatch)
    assert stops2 and not requests2


# ---------------------------------------------------------------------------
# 3. Frozen capability reaches the REAL child process (A2)
# ---------------------------------------------------------------------------


def _echo_worker_kwargs(queue_obj, **kwargs):
    queue_obj.put(kwargs)


def test_legacy_capability_transport_reaches_a_real_child(tmp_path, monkeypatch):
    store = tmp_path / "stale-store"
    win = _window(
        tmp_path,
        monkeypatch,
        mode="zegrid",
        scheduler="legacy",
        checkpoints=True,
        extra={"zegrid_durable_store_dir": str(store)},
    )
    win._pending_stop_capability = win.STOP_CAPABILITY_LEGACY_GRACEFUL
    win._pending_run_is_zegrid = False
    # A stale per-launch plan must never leak through a generic capability.
    win._pending_zegrid_resume = {
        "mode": "auto",
        "run_id": "stale-run",
        "store_dir": str(store),
    }
    _args, worker_kwargs = win._build_worker_invocation(skip_filter_ui=True)
    assert worker_kwargs["zegrid_stop_capability"] == LEGACY
    assert worker_kwargs["zegrid_resume_mode"] == "never"
    assert worker_kwargs["zegrid_resume_run_id"] is None
    assert worker_kwargs["zegrid_durable_store_dir"] is None

    ctx = multiprocessing.get_context("spawn")
    queue_obj = ctx.Queue()
    process = ctx.Process(
        target=_echo_worker_kwargs, args=(queue_obj,), kwargs=worker_kwargs
    )
    process.start()
    try:
        received = queue_obj.get(timeout=30)
    except queue_mod.Empty:  # pragma: no cover - spawn failure
        process.join(timeout=5)
        pytest.fail("child process did not echo its kwargs")
    finally:
        process.join(timeout=30)
        if process.is_alive():  # pragma: no cover - defensive
            process.terminate()
            process.join(timeout=5)
    assert received["zegrid_stop_capability"] == LEGACY
    assert received["zegrid_resume_mode"] == "never"
    assert received["zegrid_resume_run_id"] is None
    assert received["zegrid_durable_store_dir"] is None


def test_durable_capability_transport_is_unchanged(tmp_path, monkeypatch):
    store = tmp_path / "store"
    win = _window(
        tmp_path,
        monkeypatch,
        mode="zegrid",
        scheduler="active",
        checkpoints=True,
        extra={"zegrid_durable_store_dir": str(store)},
    )
    win._pending_stop_capability = win.STOP_CAPABILITY_ZEGRID_DURABLE
    win._pending_run_is_zegrid = True
    win._pending_zegrid_resume = {
        "mode": "auto",
        "run_id": "run-partial",
        "store_dir": str(store),
    }
    _args, worker_kwargs = win._build_worker_invocation(skip_filter_ui=True)
    assert worker_kwargs["zegrid_stop_capability"] == DURABLE
    assert worker_kwargs["zegrid_resume_mode"] == "auto"
    assert worker_kwargs["zegrid_resume_run_id"] == "run-partial"
    assert worker_kwargs["zegrid_durable_store_dir"] == str(store)


# ---------------------------------------------------------------------------
# 4. Negative isolation: a pre-existing .zegrid-durable is never touched
# ---------------------------------------------------------------------------


def _drive_start(win, monkeypatch):
    """Run the real launch flow up to (but never into) the worker spawn."""
    monkeypatch.setattr(gui, "run_hierarchical_mosaic_process", object())
    monkeypatch.setattr(win, "_collect_config_from_widgets", lambda: None)
    monkeypatch.setattr(win, "_save_config", lambda: None)
    monkeypatch.setattr(win, "_log_run_config_snapshot", lambda snapshot: None)
    started: list[dict] = []
    monkeypatch.setattr(
        win,
        "_begin_async_worker_start",
        lambda args, kwargs: started.append(kwargs),
    )
    win._start_processing(skip_filter_prompt=True, predecided_skip_filter_ui=True)
    return started


@pytest.mark.parametrize(
    "mode,scheduler,checkpoints",
    [
        ("classic", None, True),
        ("sds", None, True),
        ("zegrid", "legacy", True),
        ("zegrid", "legacy", False),
        ("zegrid", "observe", True),
        ("zegrid", "observe", False),
        ("zegrid", "active", False),
    ],
)
def test_preexisting_store_is_never_read_or_mutated(
    tmp_path, monkeypatch, mode, scheduler, checkpoints
):
    out_dir = tmp_path / "out"
    store = out_dir / ".zegrid-durable"
    store.mkdir(parents=True)
    (store / "unrelated.bin").write_bytes(b"pre-existing durable bytes")
    before = _tree_digest(store)

    win = _window(
        tmp_path,
        monkeypatch,
        mode=mode,
        scheduler=scheduler,
        checkpoints=checkpoints,
    )
    win.config["output_dir"] = str(out_dir)

    preflight_calls: list[bool] = []
    prompt_calls: list[bool] = []
    reads: list[str] = []
    monkeypatch.setattr(
        win,
        "_prepare_zegrid_resume_for_start",
        lambda: (preflight_calls.append(True) or True),
    )
    monkeypatch.setattr(
        win,
        "_ask_zegrid_resume_choice",
        lambda *a, **k: (prompt_calls.append(True) or ("cancel", None)),
    )
    monkeypatch.setattr(
        win,
        "_classify_zegrid_durable_candidates",
        lambda root: (reads.append(str(root)) or {"candidates": [], "error": None}),
    )

    started = _drive_start(win, monkeypatch)

    assert preflight_calls == []
    assert prompt_calls == []
    assert reads == []
    assert _tree_digest(store) == before
    assert started, "the launch flow must still reach the spawn"
    kwargs = started[-1]
    assert kwargs["zegrid_stop_capability"] == LEGACY
    assert kwargs["zegrid_resume_mode"] == "never"
    assert kwargs["zegrid_resume_run_id"] is None
    assert kwargs["zegrid_durable_store_dir"] is None


def test_durable_launch_still_runs_the_resume_preflight(tmp_path, monkeypatch):
    win = _window(tmp_path, monkeypatch, mode="zegrid", scheduler="active",
                  checkpoints=True)
    win.config["output_dir"] = str(tmp_path / "out")
    calls: list[bool] = []
    monkeypatch.setattr(
        win,
        "_prepare_zegrid_resume_for_start",
        lambda: (calls.append(True) or True),
    )
    started = _drive_start(win, monkeypatch)
    assert calls == [True]
    kwargs = started[-1]
    assert kwargs["zegrid_stop_capability"] == DURABLE


def test_checkpoint_with_legacy_or_observe_is_explicit_in_the_log(tmp_path, monkeypatch):
    for scheduler in ("legacy", "observe"):
        win = _window(
            tmp_path,
            monkeypatch,
            mode="zegrid",
            scheduler=scheduler,
            checkpoints=True,
        )
        logs: list[tuple[str, str]] = []
        monkeypatch.setattr(
            win, "_append_log", lambda text, level="info": logs.append((level, text))
        )
        capability = win._resolve_stop_capability()
        assert capability == win.STOP_CAPABILITY_LEGACY_GRACEFUL
        win._log_stop_capability(capability)
        joined = " ".join(text for _lvl, text in logs)
        assert scheduler in joined
        assert "requires scheduler 'active'" in joined
        assert "graceful" in joined
        assert "saved/resumable" in joined  # stated as NOT claimed
        assert not any("resumable" in text.lower() and "never" not in text.lower()
                       for _lvl, text in logs)


# ---------------------------------------------------------------------------
# 5. Engine guard: durable resume on a non-active scheduler fails closed
# ---------------------------------------------------------------------------


def _seeded_store(tmp_path) -> Path:
    store = tmp_path / "durable-store"
    m4.seeded_store(
        store, identity=m4.make_identity(run_id="run-partial"), cell_indexes=(0,)
    )
    return store


def _call_run_single(tmp_path, *, scheduler, resume_mode, resume_run_id=None,
                     store_dir=None):
    return zz_run_single(
        [],
        str(tmp_path / "in"),
        tmp_path / "out",
        progress_callback=lambda *a, **k: None,
        science_config=None,
        zconfig=SimpleNamespace(zegrid_durable_store_dir=store_dir),
        scheduler_mode=scheduler,
        resume_mode=resume_mode,
        resume_run_id=resume_run_id,
    )


def zz_run_single(*args, **kwargs):
    import zemosaic.zemosaic_zegrid_mode as zz

    return zz._run_single(*args, **kwargs)


@pytest.mark.parametrize(
    "scheduler,resume_mode,resume_run_id",
    [
        ("legacy", "auto", None),
        ("observe", "auto", None),
        ("legacy", "never", "run-partial"),
        ("observe", "never", "run-partial"),
    ],
)
def test_engine_refuses_durable_resume_without_active_scheduler(
    tmp_path, monkeypatch, scheduler, resume_mode, resume_run_id
):
    store = _seeded_store(tmp_path)
    before = _tree_digest(store)
    with pytest.raises(zres.ResumeSchedulerUnsupported) as excinfo:
        _call_run_single(
            tmp_path,
            scheduler=scheduler,
            resume_mode=resume_mode,
            resume_run_id=resume_run_id,
            store_dir=str(store),
        )
    message = str(excinfo.value)
    assert "requires zegrid_scheduler_mode=active" in message
    assert scheduler in message
    assert "Refusing BEFORE opening or creating any durable store" in message
    # Fail-closed: nothing was opened, locked or mutated.
    assert _tree_digest(store) == before
    assert not (store / "run-partial" / "LOCK").exists()


def test_engine_guard_does_not_create_a_missing_store(tmp_path):
    store = tmp_path / "never-created"
    with pytest.raises(zres.ResumeSchedulerUnsupported):
        _call_run_single(
            tmp_path,
            scheduler="legacy",
            resume_mode="auto",
            store_dir=str(store),
        )
    assert not store.exists()


def test_engine_active_path_keeps_its_existing_error(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        _call_run_single(
            tmp_path,
            scheduler="active",
            resume_mode="auto",
            store_dir=None,
        )
    assert not isinstance(excinfo.value, zres.ResumeSchedulerUnsupported)
    assert "requires an existing durable store directory" in str(excinfo.value)


def test_engine_guard_never_fires_for_resume_never_without_run_id(tmp_path):
    # legacy + never (no run id) must NOT be refused by the new guard; it fails
    # later for the ordinary missing-frame reasons, not with a resume refusal.
    with pytest.raises(Exception) as excinfo:
        _call_run_single(tmp_path, scheduler="legacy", resume_mode="never")
    assert not isinstance(excinfo.value, zres.ResumeSchedulerUnsupported)


# ---------------------------------------------------------------------------
# 6. A4: absent key -> active; saving never rewrites explicit legacy/observe
# ---------------------------------------------------------------------------


def test_a4_absent_key_is_active(tmp_path):
    cfg_path = tmp_path / "zemosaic_config.json"
    cfg_path.write_text(json.dumps({"language": "en"}), encoding="utf-8")
    loaded = zc.load_config(cfg_path)
    assert loaded["zegrid_scheduler_mode"] == zsched.MODE_ACTIVE


def test_a4_saving_never_persists_active_over_explicit_legacy(tmp_path, monkeypatch):
    saved: dict = {}
    win = _make_window(
        tmp_path,
        monkeypatch,
        {"zegrid_scheduler_mode": "legacy"},
        save_store=saved,
    )
    assert win.config["zegrid_scheduler_mode"] == "legacy"
    assert win._validate_zegrid_scheduler_for_start() is True
    assert win.config["zegrid_scheduler_mode"] == "legacy"
    win._collect_config_from_widgets = lambda: None
    win._save_config()
    assert saved["zegrid_scheduler_mode"] == "legacy"
    assert win._serialize_config_for_save()["zegrid_scheduler_mode"] == "legacy"
