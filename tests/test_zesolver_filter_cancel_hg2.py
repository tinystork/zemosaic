"""HG2 corrective tests: Filter dialog cancellation never destroys a live QThread.

Covers the bounded corrective mission ZM-ZS-FILTER-HG2-CANCEL.  These tests
exercise the *real* ``FilterQtDialog`` (not the ``__new__`` shell used by the
handoff tests) so that ``super().reject()`` / ``super().accept()`` /
``closeEvent`` / ``result()`` have a real ``QDialog`` backing.

They never require a real ZeSolver installation or a real ASTAP binary: a fake
blocking ``zesolver.api.v1`` module is injected whose ``Session.solve()`` blocks
until its cancellation token is cancelled (i.e. until the adapter's
``cancel_active_solve`` runs) and then returns ``CANCELLED``.  The ASTAP
callable is monkeypatched to record invocations so the tests can prove that no
fallback runs after a cancellation.
"""

from __future__ import annotations

import enum
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import zemosaic.zemosaic_filter_gui_qt as qt_filter  # noqa: E402
from zemosaic.zemosaic_filter_gui_qt import FilterQtDialog  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app
    except Exception:  # pragma: no cover - PySide6 absent
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fits_no_wcs(path: Path) -> None:
    import numpy as np
    from astropy.io import fits

    hdu = fits.PrimaryHDU(data=np.zeros((10, 10), dtype=np.float32))
    hdu.writeto(str(path), overwrite=True)


def _install_blocking_zesolver(monkeypatch) -> types.SimpleNamespace:
    """Install a fake blocking ``zesolver.api.v1`` and return an instrumented recorder."""
    rec = types.SimpleNamespace(
        solve_calls=0,
        solve_started=threading.Event(),
        solve_return_statuses=[],
        last_cancellation=None,
        token_cancel_calls=0,
        cancel_active_calls=0,
        runtime_close_calls=0,
        session_close_calls=0,
    )

    class _NetworkPolicy(enum.Enum):
        DISABLED = "disabled"

    class _GpuPolicy(enum.Enum):
        AUTO = "auto"
        DISABLED = "disabled"
        REQUIRED = "required"

    class _BackendPolicy(enum.Enum):
        AUTO = "auto"
        NEAR_ONLY = "near_only"
        BLIND_ONLY = "blind_only"

    class _SolveStatus(enum.Enum):
        SOLVED = "solved"
        FAILED = "failed"
        CANCELLED = "cancelled"
        SKIPPED_EXISTING_WCS = "skipped_existing_wcs"

    class _FailureCode(enum.Enum):
        NO_SOLUTION = "no_solution"

    class _CancellationToken:
        def __init__(self):
            self._cancelled = threading.Event()

        def cancel(self):
            rec.token_cancel_calls += 1
            self._cancelled.set()

        def is_cancelled(self):
            return self._cancelled.is_set()

    class _WcsHeader:
        def to_fits_header(self):
            return {
                "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
                "CRVAL1": 10.0, "CRVAL2": 20.0,
                "CRPIX1": 5.0, "CRPIX2": 5.0,
            }

        def to_astropy_wcs(self):
            return types.SimpleNamespace(is_celestial=True)

    class _SolveResult:
        def __init__(self, status, wcs_header=None, failure_code=None, message=None):
            self.status = status
            self.wcs_header = wcs_header
            self.failure_code = failure_code
            self.message = message

    class _Session:
        def solve(self, request, cancellation=None, progress=None):
            rec.solve_calls += 1
            rec.last_cancellation = cancellation
            rec.solve_started.set()
            # Block until the cooperative cancellation token is set (or a hard
            # timeout), mirroring a long-running near solve.
            deadline = time.time() + 30.0
            while True:
                if cancellation is not None and cancellation.is_cancelled():
                    break
                if time.time() > deadline:
                    break
                time.sleep(0.01)
            if cancellation is not None and cancellation.is_cancelled():
                rec.solve_return_statuses.append("cancelled")
                return _SolveResult(_SolveStatus.CANCELLED, message="cancelled")
            rec.solve_return_statuses.append("solved")
            return _SolveResult(_SolveStatus.SOLVED, wcs_header=_WcsHeader())

        def close(self):
            rec.session_close_calls += 1

    class _Runtime:
        def create_session(self):
            return _Session()

        def close(self):
            rec.runtime_close_calls += 1

    v1 = types.ModuleType("zesolver.api.v1")
    v1.API_VERSION = "1.0"
    v1.API_MAJOR = 1

    def get_api_info():
        return types.SimpleNamespace(
            product_version="1.2.3",
            supported_capabilities=("near_solve", "blind_solve", "wcs_write", "gpu", "cancel"),
        )

    v1.get_api_info = get_api_info

    def probe(**kw):
        return types.SimpleNamespace(capabilities=())

    v1.probe = probe
    v1.SolveHints = lambda **kw: None
    v1.SolveOptions = lambda **kw: None
    v1.SolveRequest = lambda input_path, hints=None, options=None: None
    v1.CanonicalWcsHeader = _WcsHeader
    v1.SolveResult = _SolveResult
    v1.SolveStatus = _SolveStatus
    v1.FailureCode = _FailureCode
    v1.NetworkPolicy = _NetworkPolicy
    v1.GpuPolicy = _GpuPolicy
    v1.BackendPolicy = _BackendPolicy
    v1.CancellationToken = _CancellationToken
    v1.create_solver_runtime = lambda **kw: _Runtime()

    pkg = types.ModuleType("zesolver")
    pkg.__path__ = []
    api = types.ModuleType("zesolver.api")
    api.__path__ = []
    monkeypatch.setitem(sys.modules, "zesolver", pkg)
    monkeypatch.setitem(sys.modules, "zesolver.api", api)
    monkeypatch.setitem(sys.modules, "zesolver.api.v1", v1)

    # Record ``cancel_active_solve`` invocations on the *real* adapter class so
    # the tests can prove the active solve received the cooperative cancel.
    adapter_module = qt_filter._zesolver_adapter
    original_cancel = adapter_module.ZeSolverAdapter.cancel_active_solve

    def _cancel_wrapper(self):
        rec.cancel_active_calls += 1
        return original_cancel(self)

    monkeypatch.setattr(adapter_module.ZeSolverAdapter, "cancel_active_solve", _cancel_wrapper)
    return rec


def _make_dialog(monkeypatch, tmp_path, fits_paths, *, write_wcs=False):
    """Build a real FilterQtDialog configured for ZeSolver + ASTAP-fallback."""
    # Isolate config I/O so geometry persistence never touches the user config.
    monkeypatch.setattr(qt_filter, "_load_gui_config", lambda: {})
    monkeypatch.setattr(qt_filter, "_save_gui_config", lambda cfg: None)

    astap_exe = tmp_path / "astap"
    astap_exe.write_bytes(b"#!/bin/sh\n")

    astap_calls: list = []

    def _fake_solve_with_astap(*args, **kwargs):
        astap_calls.append((args, kwargs))
        return types.SimpleNamespace(is_celestial=True)

    monkeypatch.setattr(qt_filter, "solve_with_astap", _fake_solve_with_astap)

    solver_settings = {
        "solver_choice": "ZESOLVER",
        "astap_executable_path": str(astap_exe),
        "astap_data_directory_path": str(tmp_path),
    }
    dialog = FilterQtDialog(
        [str(p) for p in fits_paths],
        solver_settings_dict=solver_settings,
    )
    if dialog._write_wcs_checkbox is not None:
        dialog._write_wcs_checkbox.setChecked(write_wcs)
    dialog.show()
    return dialog, astap_calls


def _wait_for(event: threading.Event, app, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while not event.is_set():
        if time.time() > deadline:
            raise AssertionError("timed out waiting for event")
        app.processEvents()
        time.sleep(0.01)


def _wait_until(predicate, app, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        app.processEvents()
        time.sleep(0.01)


def _cleanup_dialog(dialog, app, thread) -> None:
    """Cooperative, non-crashing cleanup that never leaves a live thread behind."""
    try:
        worker = dialog._scan_worker
        if worker is not None:
            worker.request_stop()
    except Exception:
        pass
    try:
        if thread is not None:
            deadline = time.time() + 15.0
            while thread.isRunning() and time.time() < deadline:
                app.processEvents()
                time.sleep(0.01)
    except Exception:
        pass
    try:
        dialog.deleteLater()
        app.processEvents()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# A-G. reject() during an active ZeSolver scan is deferred and finalized safely
# ---------------------------------------------------------------------------


def test_reject_during_active_scan_is_safe(monkeypatch, tmp_path, _qapp):
    if _qapp is None:
        pytest.skip("PySide6 unavailable")
    from PySide6.QtWidgets import QDialog, QDialogButtonBox

    app = _qapp
    rec = _install_blocking_zesolver(monkeypatch)

    fits1 = tmp_path / "a.fits"
    fits2 = tmp_path / "b.fits"
    _write_fits_no_wcs(fits1)
    _write_fits_no_wcs(fits2)
    original1 = fits1.read_bytes()
    original2 = fits2.read_bytes()

    dialog, astap_calls = _make_dialog(
        monkeypatch, tmp_path, [fits1, fits2], write_wcs=False
    )

    events: list[str] = []
    dialog_done: list[int] = []
    rows: list[tuple[int, dict]] = []
    retained_at_worker_finished: list[object] = []
    dialog.rejected.connect(lambda: events.append("dialog.rejected"))
    dialog.finished.connect(lambda r: dialog_done.append(r))

    thread = None
    try:
        dialog._on_run_analysis()
        worker = dialog._scan_worker
        thread = dialog._scan_thread
        assert worker is not None and thread is not None
        worker.row_updated.connect(lambda idx, row: rows.append((idx, row)))
        worker.finished.connect(
            lambda: retained_at_worker_finished.append(dialog._scan_thread)
        )
        worker.finished.connect(lambda: events.append("worker.finished"))
        thread.finished.connect(lambda: events.append("thread.finished"))

        # Wait until the first (blocking) solve is actually in flight.
        _wait_for(rec.solve_started, app)
        assert rec.solve_calls == 1

        # A. reject() must not immediately destroy/close the dialog or thread.
        dialog.reject()
        assert dialog._reject_pending is True
        assert dialog_done == []                 # not finalized yet
        assert dialog.isVisible() is True        # dialog still open
        assert thread.isRunning() is True        # thread still running

        # A2. Cancel (reject), Ok (accept) and Analyse are all disabled so no
        # further interaction can complete/destroy the dialog mid-scan.
        cancel_btn = dialog._dialog_button_box.button(QDialogButtonBox.Cancel)
        ok_btn = dialog._dialog_button_box.button(QDialogButtonBox.Ok)
        assert cancel_btn is not None and cancel_btn.isEnabled() is False
        assert ok_btn is not None and ok_btn.isEnabled() is False
        assert dialog._run_analysis_btn is not None
        assert dialog._run_analysis_btn.isEnabled() is False

        # B. cooperative stop flag set on the worker.
        assert worker._stop_requested is True

        # C. cancellation reached the active solve.
        assert rec.cancel_active_calls >= 1
        assert rec.token_cancel_calls >= 1

        # Drive to completion.
        _wait_until(lambda: dialog_done != [], app)

        # C/D. the active solve returned CANCELLED; no second solve was scheduled.
        assert rec.solve_calls == 1
        assert rec.solve_return_statuses == ["cancelled"]
        assert len(rows) == 1
        assert rows[0][0] == 0
        assert "cancel" in str(rows[0][1].get("error", "")).lower()

        # E. no ASTAP fallback.
        assert astap_calls == []

        # F. worker finished before the dialog was rejected.
        assert events.index("worker.finished") < events.index("dialog.rejected")
        assert dialog_done[0] == int(QDialog.DialogCode.Rejected)

        # F2. worker.finished precedes the real QThread.finished event.
        assert events.index("worker.finished") < events.index("thread.finished")

        # F3. the dialog retained the *captured* thread reference through
        # worker.finished (the worker emits ``finished`` before the thread
        # finishes); references are cleared only at/after thread.finished.
        assert retained_at_worker_finished == [thread]
        assert dialog._scan_thread is None
        assert dialog._scan_worker is None

        # G. the QThread is no longer running when the dialog is destroyed.
        _wait_until(lambda: thread.isFinished(), app)
        assert thread.isRunning() is False

        dialog.deleteLater()
        app.processEvents()
    finally:
        _cleanup_dialog(dialog, app, thread)

    # I. Write-WCS-OFF originals remain byte-identical.
    assert fits1.read_bytes() == original1
    assert fits2.read_bytes() == original2


# ---------------------------------------------------------------------------
# I. repeated reject/close/accept during a pending cancellation cannot finalize
# ---------------------------------------------------------------------------


def test_repeated_reject_and_close_cannot_finalize_early(monkeypatch, tmp_path, _qapp):
    if _qapp is None:
        pytest.skip("PySide6 unavailable")
    from PySide6.QtWidgets import QDialog

    app = _qapp
    rec = _install_blocking_zesolver(monkeypatch)

    fits1 = tmp_path / "a.fits"
    _write_fits_no_wcs(fits1)

    dialog, astap_calls = _make_dialog(
        monkeypatch, tmp_path, [fits1], write_wcs=False
    )

    dialog_done: list[int] = []
    dialog.finished.connect(lambda r: dialog_done.append(r))

    thread = None
    try:
        dialog._on_run_analysis()
        worker = dialog._scan_worker
        thread = dialog._scan_thread
        assert worker is not None and thread is not None
        captured = thread

        _wait_for(rec.solve_started, app)

        # Repeated reject/close/accept during the pending window must never
        # finalize the dialog early.
        dialog.reject()
        dialog.close()
        dialog.reject()
        dialog.accept()
        assert dialog._reject_pending is True
        assert dialog._close_pending is True
        assert dialog._accept_pending is True
        assert dialog_done == []                 # still not finalized
        assert dialog.isVisible() is True        # dialog still open
        assert dialog._scan_thread is captured    # reference retained
        assert captured.isRunning() is True       # thread still running
        assert rec.cancel_active_calls >= 1       # cooperative cancel requested

        _wait_until(lambda: dialog_done != [], app)

        # Exactly one finalization; reject wins over close/accept.
        assert len(dialog_done) == 1
        assert dialog_done[0] == int(QDialog.DialogCode.Rejected)
        assert rec.solve_calls == 1
        assert rec.solve_return_statuses == ["cancelled"]
        assert astap_calls == []

        _wait_until(lambda: captured.isFinished(), app)
        assert captured.isRunning() is False
        assert dialog._scan_thread is None
        assert dialog._scan_worker is None

        dialog.deleteLater()
        app.processEvents()
    finally:
        _cleanup_dialog(dialog, app, thread)


# ---------------------------------------------------------------------------
# J. deterministic fake-thread lifecycle ordering (worker.finished ->
#    QThread.finished -> dialog rejection)
# ---------------------------------------------------------------------------


def test_fake_thread_lifecycle_ordering(monkeypatch, tmp_path, _qapp):
    if _qapp is None:
        pytest.skip("PySide6 unavailable")
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QDialog

    app = _qapp
    fits1 = tmp_path / "a.fits"
    _write_fits_no_wcs(fits1)

    dialog, _ = _make_dialog(monkeypatch, tmp_path, [fits1], write_wcs=False)

    # Reproduce the scan connection topology with a bare (empty) QThread and an
    # opaque worker placeholder, so the lifecycle is fully deterministic: we
    # drive worker.finished and QThread.finished by hand.
    thread = QThread(dialog)
    dialog._scan_thread = thread
    dialog._scan_worker = object()  # identity placeholder only
    dialog._reject_pending = True
    dialog._disable_dialog_actions()

    thread.finished.connect(lambda t=thread: dialog._on_scan_thread_finished(t))
    thread.finished.connect(thread.deleteLater)

    order: list[str] = []
    done: list[int] = []
    dialog.rejected.connect(lambda: order.append("dialog.rejected"))
    dialog.finished.connect(lambda r: done.append(int(r)))

    captured = thread

    thread.start()
    app.processEvents()
    assert thread.isRunning() is True

    # 1) worker.finished fires while the thread is still running.
    dialog._on_scan_finished()
    assert dialog._scan_thread is captured       # retained through worker.finished
    assert dialog._scan_worker is not None        # worker ref retained too
    assert done == [] and order == []             # nothing finalized yet
    assert dialog._run_analysis_btn is not None
    assert dialog._run_analysis_btn.isEnabled() is False  # not re-enabled while pending

    # 2) worker.finished -> thread.quit(); wait for the real thread.finished.
    thread.quit()
    _wait_until(lambda: dialog._scan_thread is None, app)

    # 3) thread.finished handler cleared refs then finalized the pending reject.
    assert dialog._scan_worker is None
    assert captured.isRunning() is False          # thread no longer running at finalization
    assert int(QDialog.DialogCode.Rejected) in done
    assert "dialog.rejected" in order
    assert dialog.result() == int(QDialog.DialogCode.Rejected)

    dialog.deleteLater()
    app.processEvents()


# ---------------------------------------------------------------------------
# H. window X (closeEvent) during an active scan follows the same safe lifecycle
# ---------------------------------------------------------------------------


def test_close_during_active_scan_is_safe(monkeypatch, tmp_path, _qapp):
    if _qapp is None:
        pytest.skip("PySide6 unavailable")
    from PySide6.QtWidgets import QDialog

    app = _qapp
    rec = _install_blocking_zesolver(monkeypatch)

    fits1 = tmp_path / "a.fits"
    fits2 = tmp_path / "b.fits"
    _write_fits_no_wcs(fits1)
    _write_fits_no_wcs(fits2)
    original1 = fits1.read_bytes()
    original2 = fits2.read_bytes()

    dialog, astap_calls = _make_dialog(
        monkeypatch, tmp_path, [fits1, fits2], write_wcs=False
    )

    dialog_done: list[int] = []
    dialog.finished.connect(lambda r: dialog_done.append(r))

    thread = None
    try:
        dialog._on_run_analysis()
        worker = dialog._scan_worker
        thread = dialog._scan_thread
        assert worker is not None and thread is not None

        _wait_for(rec.solve_started, app)

        # H. close() (window X) during an active scan must defer, not destroy.
        dialog.close()
        assert dialog._close_pending is True
        assert dialog_done == []                 # not finalized yet
        assert dialog.isVisible() is True        # close ignored, still open
        assert thread.isRunning() is True
        assert worker._stop_requested is True

        _wait_until(lambda: dialog_done != [], app)

        assert rec.solve_calls == 1
        assert rec.solve_return_statuses == ["cancelled"]
        assert astap_calls == []
        assert dialog_done[0] == int(QDialog.DialogCode.Rejected)

        _wait_until(lambda: thread.isFinished(), app)
        assert thread.isRunning() is False

        dialog.deleteLater()
        app.processEvents()
    finally:
        _cleanup_dialog(dialog, app, thread)

    assert fits1.read_bytes() == original1
    assert fits2.read_bytes() == original2
