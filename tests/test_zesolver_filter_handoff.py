"""Focused tests for the ZeSolver Filter -> Phase-1 WCS handoff and lifecycle guards.

Covers the corrective mission objectives:

* Phase-1 reuses a valid Filter-resolved WCS/header (Write WCS OFF) instead of
  re-solving the original FITS (never trusting ``has_wcs`` blindly).
* Cancellation never moves the original FITS nor falls back to ASTAP.
* Qt GUI async startup/shutdown lifecycle is guarded against a pending spawn
  completing after close.
* Filter scan worker exception lifecycle + fallback visibility diagnostics.

These tests never require a real ZeSolver installation or a real ASTAP binary:
they inject fake modules / monkeypatch the heavy worker internals, consistent
with the repo's existing source-inspection + fake-module style.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

WORKER = SRC / "zemosaic" / "zemosaic_worker.py"
GUI = SRC / "zemosaic" / "zemosaic_gui_qt.py"
FILTER = SRC / "zemosaic" / "zemosaic_filter_gui_qt.py"


def _load_worker():
    return importlib.import_module("zemosaic.zemosaic_worker")


def _noop_pcb(*_args, **_kwargs):
    return None


# ---------------------------------------------------------------------------
# Source-inspection: structural wiring
# ---------------------------------------------------------------------------


def test_worker_phase1_passes_pre_resolved_header():
    src = WORKER.read_text(encoding="utf-8")
    assert "pre_resolved_header=" in src
    assert "_pre_resolved_header_for_path(" in src
    assert "pre_resolved_header=None" in src
    # Both the classic-legacy and the grid/SDS Phase-1 loops must pass it.
    assert src.count("pre_resolved_header=_pre_resolved_header_for_path(f_path, phase0_lookup)") == 2


def test_worker_cancellation_guard_present():
    src = WORKER.read_text(encoding="utf-8")
    assert "_solver_outcome_is_cancelled" in src
    assert "solve_cancelled" in src
    assert "getwcs_info_solve_cancelled_no_move" in src
    # CANCELLED is never collapsed to a move-to-unaligned failure.
    assert "if solve_cancelled:" in src


def test_gui_lifecycle_guards_present():
    src = GUI.read_text(encoding="utf-8")
    assert "self._closing: bool = False" in src
    assert "self._worker_start_pending: bool = False" in src
    assert "self._closing = True" in src
    assert "_discard_spawned_worker" in src


def test_filter_worker_fallback_diagnostics_present():
    src = FILTER.read_text(encoding="utf-8")
    assert "filter.scan.zesolver_unavailable" in src
    assert "filter.scan.zesolver_failed_fallback" in src
    assert "logger.exception(\"Qt Filter scan worker failed\"" in src


# ---------------------------------------------------------------------------
# Worker helper units (dynamic)
# ---------------------------------------------------------------------------


def test_solver_outcome_is_cancelled(monkeypatch):
    worker = _load_worker()
    from zemosaic.solver_port import SolveStatus

    assert worker._solver_outcome_is_cancelled(None) is False
    assert worker._solver_outcome_is_cancelled(types.SimpleNamespace(status=None)) is False
    assert (
        worker._solver_outcome_is_cancelled(
            types.SimpleNamespace(status=SolveStatus.CANCELLED)
        )
        is True
    )
    assert (
        worker._solver_outcome_is_cancelled(
            types.SimpleNamespace(status=SolveStatus.FAILED)
        )
        is False
    )
    # Plain-string status must also be recognised defensively.
    assert (
        worker._solver_outcome_is_cancelled(
            types.SimpleNamespace(status=types.SimpleNamespace(value="cancelled"))
        )
        is True
    )


def test_pre_resolved_header_for_path(monkeypatch):
    worker = _load_worker()
    header = {"CTYPE1": "RA---TAN"}
    lookup = {"a.fits": {"path": "a.fits", "header": header}}
    assert worker._pre_resolved_header_for_path("a.fits", lookup) is header
    assert worker._pre_resolved_header_for_path("missing.fits", lookup) is None
    assert worker._pre_resolved_header_for_path("a.fits", None) is None
    # header_subset fallback
    lookup2 = {"b.fits": {"path": "b.fits", "header_subset": header}}
    assert worker._pre_resolved_header_for_path("b.fits", lookup2) is header


def test_build_pre_resolved_wcs_valid_and_invalid(monkeypatch):
    worker = _load_worker()

    class _FakeWCS(types.SimpleNamespace):
        is_celestial = True

    def _validate(header):
        if isinstance(header, dict) and header.get("CTYPE1"):
            return True, _FakeWCS(), None
        return False, None, "missing_ctype"

    monkeypatch.setattr(worker, "zemosaic_utils", types.SimpleNamespace(validate_wcs_header=_validate), raising=False)
    assert worker._build_pre_resolved_wcs(None) is None
    assert worker._build_pre_resolved_wcs({"NAXIS1": 10}) is None
    wcs = worker._build_pre_resolved_wcs({"CTYPE1": "RA---TAN", "CRVAL1": 1.0})
    assert wcs is not None and wcs.is_celestial is True


def test_merge_pre_resolved_wcs_cards_skips_non_fits_keys(monkeypatch):
    worker = _load_worker()
    target = {}
    pre = {"CTYPE1": "RA---TAN", "CRVAL1": 10.0, "shape": (10, 10), "lowercase": 1, "TOOLONGKEY": 2}
    worker._merge_pre_resolved_wcs_cards(target, pre)
    assert target.get("CTYPE1") == "RA---TAN"
    assert target.get("CRVAL1") == 10.0
    assert "shape" not in target
    assert "lowercase" not in target
    assert "TOOLONGKEY" not in target


# ---------------------------------------------------------------------------
# get_wcs_and_pretreat_raw_file integration (A, B, F, G)
# ---------------------------------------------------------------------------


class _FakeWCS(types.SimpleNamespace):
    def __init__(self, celestial=True, pixel_shape=(10, 10)):
        super().__init__(is_celestial=celestial, pixel_shape=pixel_shape)


class _FakeHDU:
    def __init__(self, header):
        self.header = header


class _FakeHDUList:
    def __init__(self, header):
        self._hdu = _FakeHDU(header)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getitem__(self, idx):
        return self._hdu


class _FakeFits:
    def __init__(self, header):
        self._header = header

    def open(self, path, mode="readonly", memmap=False):
        return _FakeHDUList(self._header)


def _run_getwcs(monkeypatch, tmp_path, *, pre_resolved_header=None, outcome=None):
    """Run get_wcs_and_pretreat_raw_file against a fully fake environment."""
    worker = _load_worker()
    from zemosaic.solver_port import SolverOutcome, SolveStatus

    src_path = tmp_path / "img.fits"
    src_path.write_bytes(b"ORIGINAL")

    on_disk_header = {"NAXIS1": 10, "NAXIS2": 10}

    def _validate(header):
        if isinstance(header, dict) and header.get("CTYPE1"):
            return True, _FakeWCS(), None
        return False, None, "missing_ctype"

    def _load_and_validate_fits(path, **kw):
        return np.zeros((10, 10, 3), dtype=np.float32), dict(on_disk_header)

    def _detect_and_correct_hot_pixels(img, a, b, progress_callback=None, save_mask_path=None):
        return img

    fake_utils = types.SimpleNamespace(
        validate_wcs_header=_validate,
        has_valid_wcs=lambda h: bool(isinstance(h, dict) and h.get("CTYPE1")),
        load_and_validate_fits=_load_and_validate_fits,
        detect_and_correct_hot_pixels=_detect_and_correct_hot_pixels,
        gpu_is_available=lambda: False,
    )

    calls = types.SimpleNamespace(solver=[], move=[], write=[])

    def _solve(**kwargs):
        calls.solver.append(kwargs.get("image_fits_path"))
        if outcome is None:
            return SolverOutcome(status=SolveStatus.FAILED, wcs=None, backend_used="astap")
        return outcome

    def _move(*args, **kwargs):
        calls.move.append(args)
        return "moved", Path(str(args[0])) / "moved.fits"

    def _write(*args, **kwargs):
        calls.write.append(args)

    monkeypatch.setattr(worker, "fits", _FakeFits(on_disk_header), raising=False)
    monkeypatch.setattr(worker, "zemosaic_utils", fake_utils, raising=False)
    monkeypatch.setattr(worker, "zemosaic_config", types.SimpleNamespace(load_config=lambda: {}), raising=False)
    monkeypatch.setattr(worker, "ZEMOSAIC_CONFIG_AVAILABLE", False, raising=False)
    monkeypatch.setattr(worker, "ASTROPY_AVAILABLE", True, raising=False)
    monkeypatch.setattr(worker, "ZEMOSAIC_UTILS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(worker, "WCS", lambda *a, **k: _FakeWCS(), raising=False)
    monkeypatch.setattr(worker, "_solve_through_solver_port", _solve, raising=False)
    monkeypatch.setattr(worker, "_move_to_unaligned_safe", _move, raising=False)
    monkeypatch.setattr(worker, "_write_header_to_fits", _write, raising=False)

    result = worker.get_wcs_and_pretreat_raw_file(
        str(src_path),
        "astap_exe", "astap_data", 3.0, 2, 100, 180,
        _noop_pcb,
        None,
        {},
        pre_resolved_header=pre_resolved_header,
    )
    return src_path, result, calls


def test_a_valid_presolved_wcs_reused_no_solver_no_move(monkeypatch, tmp_path):
    from zemosaic.solver_port import SolverOutcome, SolveStatus

    pre = {
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRVAL1": 10.0, "CRVAL2": 20.0,
        "CRPIX1": 5.0, "CRPIX2": 5.0,
        "CD1_1": -1e-4, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 1e-4,
        "NAXIS1": 10, "NAXIS2": 10,
    }
    src_path, result, calls = _run_getwcs(monkeypatch, tmp_path, pre_resolved_header=pre)
    img_data, wcs, header, hp_mask = result
    assert img_data is not None
    assert wcs is not None
    assert header is not None
    # No solver dispatch (ZeSolver or ASTAP), no move, no header write-back.
    assert calls.solver == []
    assert calls.move == []
    assert calls.write == []
    # Original FITS is byte-identical (never modified).
    assert src_path.read_bytes() == b"ORIGINAL"


def test_b_invalid_presolved_wcs_not_trusted_dispatches_solver(monkeypatch, tmp_path):
    pre = {"NAXIS1": 10, "NAXIS2": 10}  # no WCS cards -> invalid
    src_path, result, calls = _run_getwcs(monkeypatch, tmp_path, pre_resolved_header=pre)
    # Invalid pre-resolved header must not be blindly trusted: solver dispatch
    # runs normally (and, here, the FAILED outcome leads to the unaligned move).
    assert len(calls.solver) == 1
    assert calls.solver[0] == str(src_path)


def test_f_cancelled_solve_no_move_source_present(monkeypatch, tmp_path):
    from zemosaic.solver_port import SolverOutcome, SolveStatus

    outcome = SolverOutcome(status=SolveStatus.CANCELLED, wcs=None, backend_used="zesolver")
    src_path, result, calls = _run_getwcs(monkeypatch, tmp_path, outcome=outcome)
    img_data, wcs, header, hp_mask = result
    assert img_data is None and wcs is None and header is None
    # CANCELLED != SOLVE_FAILED: no move to unaligned, no header write.
    assert calls.move == []
    assert calls.write == []
    assert src_path.exists() and src_path.read_bytes() == b"ORIGINAL"


def test_g_genuine_failure_still_moves_to_unaligned(monkeypatch, tmp_path):
    from zemosaic.solver_port import SolverOutcome, SolveStatus

    outcome = SolverOutcome(status=SolveStatus.FAILED, wcs=None, backend_used="astap")
    src_path, result, calls = _run_getwcs(monkeypatch, tmp_path, outcome=outcome)
    img_data, wcs, header, hp_mask = result
    assert img_data is None and wcs is None
    # Genuine failure outside cancellation preserves historical unaligned move.
    assert len(calls.move) == 1


# ---------------------------------------------------------------------------
# Qt GUI async startup/shutdown lifecycle (D, E)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _qapp():
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app
    except Exception:  # pragma: no cover - PySide6 absent
        return None


def _gui_module():
    return importlib.import_module("zemosaic.zemosaic_gui_qt")


def _make_window(_qapp):
    gui = _gui_module()
    return gui.ZeMosaicQtMainWindow()


class _DoneThread:
    def is_alive(self):
        return False


class _FakeProcess:
    def __init__(self):
        self.terminated = False
        self.joined = False

    def is_alive(self):
        return True

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        self.joined = True


class _FakeQueue:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_e_normal_startup_finalizes(monkeypatch, _qapp):
    if _qapp is None:
        pytest.skip("PySide6 unavailable")
    w = _make_window(_qapp)
    try:
        w._closing = False
        w._worker_start_pending = True
        w._worker_start_thread = _DoneThread()
        proc = _FakeProcess()
        queue = _FakeQueue()
        w._worker_start_result = (True, (queue, proc))

        finalized = []
        monkeypatch.setattr(w.worker_controller, "finalize_spawn", lambda q, p: finalized.append((q, p)), raising=False)
        monkeypatch.setattr(w, "_finalize_successful_worker_start", lambda: finalized.append("success"), raising=False)
        monkeypatch.setattr(w, "_handle_worker_start_failure", lambda e: finalized.append(("fail", e)), raising=False)

        w._poll_worker_start_result()
        assert finalized == [(queue, proc), "success"]
        assert w._worker_start_thread is None
        assert w._worker_start_pending is False
    finally:
        w.deleteLater()


def test_d_closing_discards_pending_spawn(monkeypatch, _qapp):
    if _qapp is None:
        pytest.skip("PySide6 unavailable")
    w = _make_window(_qapp)
    try:
        w._closing = True
        w._worker_start_pending = True
        w._worker_start_thread = _DoneThread()
        proc = _FakeProcess()
        queue = _FakeQueue()
        w._worker_start_result = (True, (queue, proc))

        events = []
        monkeypatch.setattr(w.worker_controller, "finalize_spawn", lambda q, p: events.append("finalize"), raising=False)
        monkeypatch.setattr(w, "_finalize_successful_worker_start", lambda: events.append("success"), raising=False)
        monkeypatch.setattr(w, "_handle_worker_start_failure", lambda e: events.append("fail"), raising=False)

        w._poll_worker_start_result()
        # Never finalized/attached; process terminated; queue closed; no active run.
        assert events == []
        assert proc.terminated is True
        assert queue.closed is True
        assert w._worker_start_thread is None
        assert w._worker_start_pending is False
        assert w.is_processing is False
    finally:
        w.deleteLater()


def test_d_start_processing_guarded_when_closing(monkeypatch, _qapp):
    if _qapp is None:
        pytest.skip("PySide6 unavailable")
    w = _make_window(_qapp)
    try:
        w._closing = True
        # Must return early without touching worker start.
        w._start_processing(skip_filter_prompt=True, predecided_skip_filter_ui=True)
        assert w._worker_start_thread is None
        assert w.is_processing is False
    finally:
        w.deleteLater()


# ---------------------------------------------------------------------------
# Filter scan worker exception lifecycle + fallback visibility (H, I)
# ---------------------------------------------------------------------------


def _filter_module():
    return importlib.import_module("zemosaic.zemosaic_filter_gui_qt")


def _make_filter_item(path):
    m = _filter_module()
    return m._NormalizedItem(
        original={"path": path, "path_raw": path},
        display_name=Path(path).name,
        file_path=path,
        has_wcs=False,
        instrument=None,
        group_label=None,
    )


def _write_fits_no_wcs(path):
    from astropy.io import fits

    hdu = fits.PrimaryHDU(data=np.zeros((10, 10), dtype=np.float32))
    hdu.writeto(str(path), overwrite=True)


def test_h_run_unexpected_exception_cleans_up_and_finishes(monkeypatch, tmp_path):
    m = _filter_module()
    fits_path = tmp_path / "img.fits"
    _write_fits_no_wcs(fits_path)
    astap_exe = tmp_path / "astap"
    astap_exe.write_bytes(b"")

    def _boom(header):
        raise RuntimeError("boom")

    monkeypatch.setattr(m, "_sanitize_header_subset", _boom, raising=False)

    worker = m._DirectoryScanWorker(
        [_make_filter_item(str(fits_path))],
        {
            "solver_choice": "ASTAP",
            "astap_executable_path": str(astap_exe),
            "astap_data_directory_path": str(tmp_path),
        },
        m._FallbackLocalizer(),
        astap_overrides={"write_wcs_to_file": False},
    )
    closed = []

    class _FakeAdapter:
        def close(self):
            closed.append(1)

        def cancel_active_solve(self):
            pass

    worker._zesolver_adapter = _FakeAdapter()

    errors = []
    finished = []
    worker.error.connect(lambda msg: errors.append(msg))
    worker.finished.connect(lambda: finished.append(1))
    worker.run()

    assert finished == [1]  # finished always emitted
    assert len(errors) == 1 and "boom" in errors[0]  # unexpected exception reported
    assert closed == [1]  # adapter cleanup always happens


def test_i_run_zesolver_unavailable_emits_one_fallback_diagnostic(monkeypatch, tmp_path):
    m = _filter_module()
    from zemosaic import solver_port as sp

    fits_path = tmp_path / "img.fits"
    _write_fits_no_wcs(fits_path)
    astap_exe = tmp_path / "astap"
    astap_exe.write_bytes(b"")

    def _fake_astap(*a, **k):
        return types.SimpleNamespace(is_celestial=True)

    monkeypatch.setattr(m, "solve_with_astap", _fake_astap, raising=False)

    worker = m._DirectoryScanWorker(
        [_make_filter_item(str(fits_path))],
        {
            "solver_choice": "ZESOLVER",
            "astap_executable_path": str(astap_exe),
            "astap_data_directory_path": str(tmp_path),
        },
        m._FallbackLocalizer(),
        astap_overrides={"write_wcs_to_file": False},
    )
    # Force a run-scoped discovery failure (avoids the real editable ZeSolver).
    monkeypatch.setattr(
        worker,
        "_get_zesolver_discovery",
        lambda: types.SimpleNamespace(state=sp.DiscoveryState.NOT_INSTALLED, message="no zesolver"),
        raising=False,
    )

    msgs = []
    worker.progress_changed.connect(lambda p, msg: msgs.append(msg))
    rows = []
    errors = []
    worker.row_updated.connect(lambda idx, row: rows.append((idx, row)))
    worker.error.connect(lambda e: errors.append(e))
    worker.run()

    diag = [msg for msg in msgs if "falling back to ASTAP" in msg]
    assert len(diag) == 1  # exactly one run-level diagnostic
    assert rows and rows[0][1]["solver"] == "ASTAP"  # ASTAP fallback still used
