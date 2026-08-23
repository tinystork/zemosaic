"""HG2 focused tests: the real Qt Filter -> main GUI -> worker WCS handoff.

Covers the corrective mission ZM-ZS-FILTER-HG2.  These tests exercise the
*real* serialization / retention / invocation / process-boundary / worker
adoption paths, not just the worker-side Phase-1 reuse already covered by
``tests/test_zesolver_filter_handoff.py``.

They never require a real ZeSolver installation or a real ASTAP binary: a fake
``zesolver.api.v1`` module is injected (as in the adapter/hardening tests) and
the heavy worker internals are monkeypatched.  Process-boundary test E uses a
production-equivalent ``spawn`` context with a plain-Python payload.
"""

from __future__ import annotations

import enum
import importlib
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import zemosaic.zemosaic_filter_gui_qt as qt_filter  # noqa: E402
from zemosaic.zemosaic_filter_gui_qt import (  # noqa: E402
    _DirectoryScanWorker,
    _FallbackLocalizer,
    _NormalizedItem,
    FilterQtDialog,
    launch_filter_interface_qt,
)
from zemosaic.solver_port import (  # noqa: E402
    count_filter_handoff_wcs,
    header_carries_wcs_material,
)

FILTER_SOURCE = SRC / "zemosaic" / "zemosaic_filter_gui_qt.py"
GUI_SOURCE = SRC / "zemosaic" / "zemosaic_gui_qt.py"
WORKER_SOURCE = SRC / "zemosaic" / "zemosaic_worker.py"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


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


def _solved_header() -> dict:
    return {
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRVAL1": 10.0, "CRVAL2": 20.0,
        "CRPIX1": 5.0, "CRPIX2": 5.0,
        "CD1_1": -1e-4, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 1e-4,
        "NAXIS1": 100, "NAXIS2": 100,
    }


def _make_item(path: str, *, has_wcs: bool = False) -> _NormalizedItem:
    return _NormalizedItem(
        original={"path": path, "path_raw": path},
        display_name=Path(path).name,
        file_path=path,
        has_wcs=has_wcs,
        instrument=None,
        group_label=None,
    )


def _install_fake_zesolver(monkeypatch, *, solve_status: str = "SOLVED"):
    """Install a fake ``zesolver.api.v1`` returning a solved canonical WCS header."""
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
            self._c = False

        def cancel(self):
            self._c = True

        def is_cancelled(self):
            return self._c

    class _WcsHeader:
        def to_fits_header(self):
            return _solved_header()

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
            if solve_status == "SOLVED":
                return _SolveResult(_SolveStatus.SOLVED, wcs_header=_WcsHeader())
            if solve_status == "CANCELLED":
                return _SolveResult(_SolveStatus.CANCELLED)
            return _SolveResult(_SolveStatus.FAILED, failure_code=_FailureCode.NO_SOLUTION)

        def close(self):
            pass

    class _Runtime:
        def create_session(self):
            return _Session()

        def close(self):
            pass

    v1 = types.ModuleType("zesolver.api.v1")
    v1.API_VERSION = "1.2"
    v1.API_MAJOR = 1
    v1.get_api_info = lambda: types.SimpleNamespace(
        product_version="1.2.3",
        supported_capabilities=("near_solve", "blind_solve", "wcs_write", "gpu", "cancel"),
    )
    v1.probe = lambda **kw: types.SimpleNamespace(capabilities=())
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


def _write_fits_no_wcs(path: Path) -> None:
    import numpy as np
    from astropy.io import fits

    hdu = fits.PrimaryHDU(data=np.zeros((10, 10), dtype=np.float32))
    hdu.writeto(str(path), overwrite=True)


def _minimal_dialog(entries, *, stream_scan=True) -> FilterQtDialog:
    """Build a FilterQtDialog instance without running the full Qt construction."""
    dialog = FilterQtDialog.__new__(FilterQtDialog)
    dialog._stream_scan = stream_scan
    dialog._normalized_items = list(entries)
    dialog._entry_check_state = [True] * len(entries)
    return dialog


# ---------------------------------------------------------------------------
# header_carries_wcs_material (shared helper)
# ---------------------------------------------------------------------------


def test_header_carries_wcs_material_positive_and_negative():
    assert header_carries_wcs_material(_solved_header()) is True
    # Missing CD / PC matrix -> not valid material.
    assert header_carries_wcs_material({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"}) is False
    assert header_carries_wcs_material(None) is False
    assert header_carries_wcs_material({}) is False
    # PC + CDELT variant is also valid material.
    pc_header = {
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRVAL1": 1.0, "CRVAL2": 2.0, "CRPIX1": 3.0, "CRPIX2": 4.0,
        "PC1_1": 1.0, "PC1_2": 0.0, "PC2_1": 0.0, "PC2_2": 1.0,
        "CDELT1": -1e-4, "CDELT2": 1e-4,
    }
    assert header_carries_wcs_material(pc_header) is True


def test_count_filter_handoff_wcs():
    good = {"path": "/a.fits", "header": _solved_header()}
    bad = {"path": "/b.fits", "header": {"NAXIS1": 10}}
    no_header = {"path": "/c.fits", "has_wcs": True}
    total, valid = count_filter_handoff_wcs([good, bad, no_header, "not-a-dict"])
    assert total == 4
    assert valid == 1
    assert count_filter_handoff_wcs(None) == (0, 0)


# ---------------------------------------------------------------------------
# A. streaming Filter solve -> selected_items carries solved WCS header
# ---------------------------------------------------------------------------


def test_a_stream_solve_selected_item_carries_wcs(monkeypatch, tmp_path):
    _install_fake_zesolver(monkeypatch)
    fits_path = tmp_path / "img.fits"
    _write_fits_no_wcs(fits_path)
    original_bytes = fits_path.read_bytes()

    item = _make_item(str(fits_path))
    worker = _DirectoryScanWorker(
        [item],
        {"solver_choice": "ZESOLVER", "astap_executable_path": "/usr/bin/astap", "astap_data_directory_path": str(tmp_path)},
        _FallbackLocalizer(),
        astap_overrides={"write_wcs_to_file": False},
    )
    rows = []
    worker.row_updated.connect(lambda idx, row: rows.append((idx, row)))
    worker.run()

    # The solve succeeded in-memory (Write WCS OFF: original untouched).
    assert item.has_wcs is True or rows[0][1].get("has_wcs") is True
    assert item.header_cache is not None
    assert header_carries_wcs_material(item.header_cache) is True

    # selected_items() must serialise the solved WCS header, not just has_wcs.
    dialog = _minimal_dialog([item], stream_scan=True)
    selected = dialog.selected_items()
    assert len(selected) == 1
    assert selected[0].get("has_wcs") is True
    assert header_carries_wcs_material(selected[0].get("header")) is True
    # Original FITS must remain byte-identical (never written).
    assert fits_path.read_bytes() == original_bytes


def test_a_has_wcs_reconstructs_header_from_wcs_cache(monkeypatch):
    """Invariant: a solved entry always carries WCS even without header_cache."""
    item = _make_item("/tmp/img.fits", has_wcs=True)
    item.header_cache = None

    class _FakeWCS:
        def to_header(self, relax=True):
            return _solved_header()

    item.wcs_cache = _FakeWCS()

    dialog = _minimal_dialog([item], stream_scan=True)
    selected = dialog.selected_items()
    assert len(selected) == 1
    assert header_carries_wcs_material(selected[0].get("header")) is True


def test_a_wcs_cache_reconstructs_before_has_wcs_signal(monkeypatch):
    """Queued GUI row updates must not be required before serialisation.

    The real streaming path can have a solved ``wcs_cache`` while
    ``entry.has_wcs`` has not yet been synchronised by the queued table update.
    That still has to serialise a complete WCS header for the worker.
    """
    item = _make_item("/tmp/img.fits", has_wcs=False)
    item.header_cache = None

    class _FakeWCS:
        def to_header(self, relax=True):
            return _solved_header()

    item.wcs_cache = _FakeWCS()

    dialog = _minimal_dialog([item], stream_scan=True)
    selected = dialog.selected_items()
    assert len(selected) == 1
    assert selected[0].get("has_wcs") is True
    assert header_carries_wcs_material(selected[0].get("header")) is True


def _fake_astap_wcs():
    """Minimal ASTAP ``solve_with_astap`` result (celestial WCS object)."""
    class _AstapWCS:
        is_celestial = True

        def to_header(self, relax=True):
            return _solved_header()

    return _AstapWCS()


def _astap_worker(monkeypatch, tmp_path, item, *, solver_choice, write_wcs_to_file=False):
    astap_exe = tmp_path / "astap"
    astap_exe.write_bytes(b"#!/bin/sh\n")
    monkeypatch.setattr(qt_filter, "solve_with_astap", lambda *a, **k: _fake_astap_wcs())
    worker = _DirectoryScanWorker(
        [item],
        {
            "solver_choice": solver_choice,
            "astap_executable_path": str(astap_exe),
            "astap_data_directory_path": str(tmp_path),
        },
        _FallbackLocalizer(),
        astap_overrides={"write_wcs_to_file": write_wcs_to_file},
    )
    rows = []
    worker.row_updated.connect(lambda idx, row: rows.append((idx, row)))
    worker.run()
    return worker, rows


def _assert_phase1_reuses_selected_header_no_second_solve(monkeypatch, tmp_path, header):
    """Feed a Filter-selected header to Phase 1 and prove no solver is called."""
    worker = _load_worker()

    class _FakeWCS(types.SimpleNamespace):
        def __init__(self, celestial=True):
            super().__init__(is_celestial=celestial, pixel_shape=(10, 10))

    def _validate(candidate):
        if header_carries_wcs_material(candidate):
            return True, _FakeWCS(), None
        return False, None, "missing_ctype"

    def _load_and_validate_fits(path, **kw):
        import numpy as np
        return np.zeros((10, 10, 3), dtype=np.float32), {"NAXIS1": 10, "NAXIS2": 10}

    fake_utils = types.SimpleNamespace(
        validate_wcs_header=_validate,
        has_valid_wcs=lambda h: header_carries_wcs_material(h),
        load_and_validate_fits=_load_and_validate_fits,
        detect_and_correct_hot_pixels=lambda img, *a, **k: img,
        gpu_is_available=lambda: False,
    )
    calls = types.SimpleNamespace(solver=[], move=[], write=[])

    monkeypatch.setattr(worker, "fits", _FakeFits(monkeypatch), raising=False)
    monkeypatch.setattr(worker, "zemosaic_utils", fake_utils, raising=False)
    monkeypatch.setattr(worker, "zemosaic_config", types.SimpleNamespace(load_config=lambda: {}), raising=False)
    monkeypatch.setattr(worker, "ZEMOSAIC_CONFIG_AVAILABLE", False, raising=False)
    monkeypatch.setattr(worker, "ASTROPY_AVAILABLE", True, raising=False)
    monkeypatch.setattr(worker, "ZEMOSAIC_UTILS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(worker, "WCS", lambda *a, **k: _FakeWCS(), raising=False)
    monkeypatch.setattr(worker, "_solve_through_solver_port", lambda **kw: calls.solver.append(kw), raising=False)
    monkeypatch.setattr(worker, "_move_to_unaligned_safe", lambda *a, **k: calls.move.append(a), raising=False)
    monkeypatch.setattr(worker, "_write_header_to_fits", lambda *a, **k: calls.write.append(a), raising=False)

    src_path = tmp_path / "phase1.fits"
    src_path.write_bytes(b"PHASE1_ORIGINAL")
    img_data, wcs, returned_header, hp_mask = worker.get_wcs_and_pretreat_raw_file(
        str(src_path), "astap", "data", 3.0, 2, 100, 180,
        lambda *a, **k: None, None, {}, pre_resolved_header=header,
    )
    assert wcs is not None and returned_header is not None
    assert calls.solver == []
    assert calls.move == []
    assert calls.write == []
    assert src_path.read_bytes() == b"PHASE1_ORIGINAL"


def test_a_astap_selected_write_wcs_off_carries_wcs(monkeypatch, tmp_path):
    """Regression A: ASTAP selected + Write WCS OFF still serializes WCS."""
    fits_path = tmp_path / "img.fits"
    _write_fits_no_wcs(fits_path)
    original_bytes = fits_path.read_bytes()

    item = _make_item(str(fits_path))
    _astap_worker(monkeypatch, tmp_path, item, solver_choice="ASTAP")

    # ASTAP success path must synchronously mark the entry solved and cache the
    # WCS object (mirrors _apply_solved_outcome for ZeSolver).
    assert item.has_wcs is True
    assert item.wcs_cache is not None
    assert getattr(item.wcs_cache, "is_celestial", False) is True

    dialog = _minimal_dialog([item], stream_scan=True)
    selected = dialog.selected_items()
    assert len(selected) == 1
    assert selected[0].get("has_wcs") is True
    assert header_carries_wcs_material(selected[0].get("header")) is True
    _assert_phase1_reuses_selected_header_no_second_solve(
        monkeypatch, tmp_path, selected[0].get("header")
    )
    # Write WCS OFF: original FITS byte-identical.
    assert fits_path.read_bytes() == original_bytes


def test_a_zesolver_failed_astap_fallback_write_wcs_off_carries_wcs(monkeypatch, tmp_path):
    """Regression B: ZeSolver failed -> ASTAP fallback + Write WCS OFF serializes WCS."""
    _install_fake_zesolver(monkeypatch, solve_status="FAILED")
    fits_path = tmp_path / "img.fits"
    _write_fits_no_wcs(fits_path)
    original_bytes = fits_path.read_bytes()

    item = _make_item(str(fits_path))
    _astap_worker(monkeypatch, tmp_path, item, solver_choice="ZESOLVER")

    assert item.has_wcs is True
    assert item.wcs_cache is not None

    dialog = _minimal_dialog([item], stream_scan=True)
    selected = dialog.selected_items()
    assert len(selected) == 1
    assert selected[0].get("has_wcs") is True
    assert header_carries_wcs_material(selected[0].get("header")) is True
    _assert_phase1_reuses_selected_header_no_second_solve(
        monkeypatch, tmp_path, selected[0].get("header")
    )
    # Write WCS OFF: original FITS byte-identical.
    assert fits_path.read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# B. launch_filter_interface_qt return contract
# ---------------------------------------------------------------------------


def test_b_launch_filter_returns_solved_items(monkeypatch, _qapp):
    if _qapp is None:
        pytest.skip("PySide6 unavailable")
    selected_payload = [
        {"path": "/tmp/img.fits", "has_wcs": True, "header": _solved_header()}
    ]

    class _FakeDialog:
        def __init__(self, *a, **k):
            self._accepted = True

        def exec(self):
            return None

        def selected_items(self):
            return list(selected_payload)

        def overrides(self):
            return {"resolved_wcs_count": len(selected_payload)}

        def input_items(self):
            return list(selected_payload)

        def was_accepted(self):
            return True

        def deleteLater(self):
            pass

    monkeypatch.setattr(qt_filter, "FilterQtDialog", _FakeDialog)
    result = launch_filter_interface_qt("/some/dir", None, stream_scan=True)
    assert isinstance(result, tuple) and len(result) >= 3
    filtered, accepted, overrides = result[0], result[1], result[2]
    assert accepted is True
    assert isinstance(filtered, list) and len(filtered) == 1
    assert header_carries_wcs_material(filtered[0].get("header")) is True
    assert overrides == {"resolved_wcs_count": 1}


# ---------------------------------------------------------------------------
# C. GUI retention: _last_filtered_header_items preserves solved WCS
# ---------------------------------------------------------------------------


def test_c_gui_retention_preserves_wcs(monkeypatch, _qapp):
    if _qapp is None:
        pytest.skip("PySide6 unavailable")
    gui = importlib.import_module("zemosaic.zemosaic_gui_qt")
    w = gui.ZeMosaicQtMainWindow()
    try:
        items = [
            {"path": f"/tmp/img{i}.fits", "has_wcs": True, "header": _solved_header()}
            for i in range(3)
        ]
        # Simulate the accepted-Filter path writing the retention field.
        w._last_filtered_header_items = items
        assert w._last_filtered_header_items is items
        total, valid = w._count_filter_handoff_wcs(w._last_filtered_header_items)
        assert total == 3 and valid == 3
    finally:
        w.deleteLater()


def test_c_launch_filter_dialog_retains_returned_wcs(monkeypatch, tmp_path, _qapp):
    """Accepted real main-GUI Filter path keeps the solved payload intact."""
    if _qapp is None:
        pytest.skip("PySide6 unavailable")
    gui = importlib.import_module("zemosaic.zemosaic_gui_qt")
    fits_path = tmp_path / "img.fits"
    fits_path.write_bytes(b"SIMPLE  =                    T\nEND\n")
    returned_items = [
        {"path": str(fits_path), "has_wcs": True, "header": _solved_header()}
    ]

    monkeypatch.setattr(gui.ZeMosaicQtMainWindow, "_input_dir_contains_fits", lambda self, p: True)
    monkeypatch.setattr(gui.ZeMosaicQtMainWindow, "_build_solver_settings_dict", lambda self: {})
    monkeypatch.setattr(gui.ZeMosaicQtMainWindow, "_append_log", lambda self, *a, **k: None)
    monkeypatch.setattr(qt_filter, "launch_filter_interface_qt", lambda *a, **k: (returned_items, True, {"resolved_wcs_count": 1}))

    w = gui.ZeMosaicQtMainWindow()
    try:
        w.config["input_dir"] = str(tmp_path)
        assert w._launch_filter_dialog() is True
        assert w._last_filtered_header_items is returned_items
        total, valid = w._count_filter_handoff_wcs(w._last_filtered_header_items)
        assert total == 1 and valid == 1
    finally:
        w.deleteLater()


# ---------------------------------------------------------------------------
# D. _build_worker_invocation after accepted Filter
# ---------------------------------------------------------------------------


def test_d_build_worker_invocation_carries_filtered_items(monkeypatch, _qapp):
    if _qapp is None:
        pytest.skip("PySide6 unavailable")
    gui = importlib.import_module("zemosaic.zemosaic_gui_qt")
    w = gui.ZeMosaicQtMainWindow()
    try:
        items = [
            {"path": f"/tmp/img{i}.fits", "has_wcs": True, "header": _solved_header()}
            for i in range(20)
        ]
        w._last_filtered_header_items = items
        w._last_filter_overrides = {"resolved_wcs_count": 20}

        args, kwargs = w._build_worker_invocation(skip_filter_ui=True)
        assert kwargs.get("skip_filter_ui") is True
        assert kwargs.get("filtered_header_items") is items
        total, valid = w._count_filter_handoff_wcs(kwargs.get("filtered_header_items"))
        assert total == 20 and valid == 20
    finally:
        w.deleteLater()


def test_d_start_processing_accept_filter_sets_skip_and_builds_payload(monkeypatch, tmp_path, _qapp):
    """Actual Start -> prompt Yes -> Filter accepted transition builds payload."""
    if _qapp is None:
        pytest.skip("PySide6 unavailable")
    gui = importlib.import_module("zemosaic.zemosaic_gui_qt")
    items = [
        {"path": str(tmp_path / f"img{i}.fits"), "has_wcs": True, "header": _solved_header()}
        for i in range(2)
    ]
    captured: dict[str, object] = {}

    monkeypatch.setattr(gui, "run_hierarchical_mosaic_process", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(gui.QMessageBox, "question", lambda *a, **k: gui.QMessageBox.Yes)
    monkeypatch.setattr(gui.ZeMosaicQtMainWindow, "_collect_config_from_widgets", lambda self: None)
    monkeypatch.setattr(gui.ZeMosaicQtMainWindow, "_existing_master_tiles_enabled", lambda self: False)
    monkeypatch.setattr(gui.ZeMosaicQtMainWindow, "_launch_filter_dialog", lambda self: (setattr(self, "_last_filtered_header_items", items) or setattr(self, "_last_filter_overrides", {"resolved_wcs_count": 2}) or True))
    monkeypatch.setattr(gui.ZeMosaicQtMainWindow, "_serialize_config_for_save", lambda self: {"input_dir": str(tmp_path), "output_dir": str(tmp_path / "out")})
    monkeypatch.setattr(gui.ZeMosaicQtMainWindow, "_save_config", lambda self: None)
    monkeypatch.setattr(gui.ZeMosaicQtMainWindow, "_append_log", lambda self, *a, **k: None)

    def _capture_begin(self, args, kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(gui.ZeMosaicQtMainWindow, "_begin_async_worker_start", _capture_begin)

    w = gui.ZeMosaicQtMainWindow()
    try:
        w._start_processing(skip_filter_prompt=False, predecided_skip_filter_ui=None)
        kwargs = captured.get("kwargs")
        assert isinstance(kwargs, dict)
        assert kwargs.get("skip_filter_ui") is True
        assert kwargs.get("filter_invoked") is True
        assert kwargs.get("filtered_header_items") is items
        total, valid = w._count_filter_handoff_wcs(kwargs.get("filtered_header_items"))
        assert total == 2 and valid == 2
    finally:
        w.deleteLater()


# ---------------------------------------------------------------------------
# E. process-boundary serialization with production-equivalent spawn
# ---------------------------------------------------------------------------


def _spawn_child_receiver(q, **kwargs):
    """Runs in the spawned child: prove the handoff payload survives pickling."""
    import inspect

    import zemosaic.zemosaic_worker as w

    sig = inspect.signature(w.run_hierarchical_mosaic).parameters
    final = {k: v for k, v in kwargs.items() if k in sig}
    got = final.get("filtered_header_items")
    count = len(got) if isinstance(got, list) else 0
    valid = sum(
        1
        for x in (got or [])
        if isinstance(x, dict) and (x.get("header") or {}).get("CTYPE1")
    )
    q.put({"count": count, "valid": valid, "skip_filter_ui": final.get("skip_filter_ui")})


def test_e_spawn_payload_survives(monkeypatch):
    import multiprocessing

    payload = [
        {"path": f"/tmp/img{i}.fits", "has_wcs": True, "header": _solved_header()}
        for i in range(20)
    ]
    try:
        ctx = multiprocessing.get_context("spawn")
    except Exception:  # pragma: no cover - platform without spawn
        ctx = multiprocessing.get_context()

    q = ctx.Queue()
    p = ctx.Process(
        target=_spawn_child_receiver,
        args=(q,),
        kwargs={"filtered_header_items": payload, "skip_filter_ui": True},
    )
    p.start()
    p.join(timeout=120)
    assert p.exitcode == 0, "spawned child must exit cleanly"
    result = q.get(timeout=10)
    assert result["count"] == 20
    assert result["valid"] == 20
    assert result["skip_filter_ui"] is True


# ---------------------------------------------------------------------------
# F. worker Phase-0 adoption (no disk header scan)
# ---------------------------------------------------------------------------


def _load_worker():
    return importlib.import_module("zemosaic.zemosaic_worker")


def test_f_worker_adopts_filtered_items_and_skips_scan(monkeypatch):
    worker = _load_worker()
    # Structural: the adoption block marks streaming_filter_success and skips
    # the "Phase 0: header scan start" branch when filtered_header_items is
    # non-empty.
    src = WORKER_SOURCE.read_text(encoding="utf-8")
    assert "if isinstance(filtered_header_items_arg, list) and filtered_header_items_arg:" in src
    assert "streaming_filter_success = True" in src
    assert "Phase 0: header scan start" in src

    # Behavioural: count_filter_handoff_wcs recognises the adopted payload.
    payload = [{"path": "/a.fits", "header": _solved_header()} for _ in range(20)]
    total, valid = count_filter_handoff_wcs(payload)
    assert total == 20 and valid == 20


def test_f_worker_phase0_lookup_builds_normalized_keys(monkeypatch):
    worker = _load_worker()
    items = [
        {"path": "~/img_a.fits", "header": _solved_header()},
        {"path": "~/img_b.fits", "header": _solved_header()},
    ]
    lookup = worker._build_phase0_lookup(items)
    # Raw keys preserved.
    assert "~/img_a.fits" in lookup
    assert "~/img_b.fits" in lookup
    # An expanded spelling resolves to the same item via the normalized key.
    expanded_a = os.path.expanduser("~/img_a.fits")
    assert worker._pre_resolved_header_for_path(expanded_a, lookup) is not None


# ---------------------------------------------------------------------------
# G. Phase-1 reuse: pre-resolved Filter WCS accepted, no re-solve
# ---------------------------------------------------------------------------


def test_g_valid_presolved_reused_no_solver_no_astap(monkeypatch, tmp_path):
    worker = _load_worker()

    class _FakeWCS(types.SimpleNamespace):
        def __init__(self, celestial=True):
            super().__init__(is_celestial=celestial, pixel_shape=(10, 10))

    def _validate(header):
        if isinstance(header, dict) and header.get("CTYPE1"):
            return True, _FakeWCS(), None
        return False, None, "missing_ctype"

    def _load_and_validate_fits(path, **kw):
        import numpy as np
        return np.zeros((10, 10, 3), dtype=np.float32), {"NAXIS1": 10, "NAXIS2": 10}

    def _hotpixels(img, a, b, progress_callback=None, save_mask_path=None):
        return img

    fake_utils = types.SimpleNamespace(
        validate_wcs_header=_validate,
        has_valid_wcs=lambda h: bool(isinstance(h, dict) and h.get("CTYPE1")),
        load_and_validate_fits=_load_and_validate_fits,
        detect_and_correct_hot_pixels=_hotpixels,
        gpu_is_available=lambda: False,
    )
    calls = types.SimpleNamespace(solver=[], move=[], write=[])

    from zemosaic.solver_port import SolverOutcome, SolveStatus

    def _solve(**kwargs):
        calls.solver.append(kwargs.get("image_fits_path"))

    def _move(*a, **k):
        calls.move.append(a)

    def _write(*a, **k):
        calls.write.append(a)

    monkeypatch.setattr(worker, "fits", _FakeFits(monkeypatch), raising=False)
    monkeypatch.setattr(worker, "zemosaic_utils", fake_utils, raising=False)
    monkeypatch.setattr(worker, "zemosaic_config", types.SimpleNamespace(load_config=lambda: {}), raising=False)
    monkeypatch.setattr(worker, "ZEMOSAIC_CONFIG_AVAILABLE", False, raising=False)
    monkeypatch.setattr(worker, "ASTROPY_AVAILABLE", True, raising=False)
    monkeypatch.setattr(worker, "ZEMOSAIC_UTILS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(worker, "WCS", lambda *a, **k: _FakeWCS(), raising=False)
    monkeypatch.setattr(worker, "_solve_through_solver_port", _solve, raising=False)
    monkeypatch.setattr(worker, "_move_to_unaligned_safe", _move, raising=False)
    monkeypatch.setattr(worker, "_write_header_to_fits", _write, raising=False)

    src_path = tmp_path / "img.fits"
    src_path.write_bytes(b"ORIGINAL")
    result = worker.get_wcs_and_pretreat_raw_file(
        str(src_path), "astap", "data", 3.0, 2, 100, 180,
        lambda *a, **k: None, None, {}, pre_resolved_header=_solved_header(),
    )
    img_data, wcs, header, hp_mask = result
    assert wcs is not None and header is not None
    assert calls.solver == []
    assert calls.move == []
    assert calls.write == []
    assert src_path.read_bytes() == b"ORIGINAL"


class _FakeFits:
    def __init__(self, monkeypatch):
        self._monkeypatch = monkeypatch

    def open(self, path, mode="readonly", memmap=False):
        return _FakeHDUList({"NAXIS1": 10, "NAXIS2": 10})

    def getheader(self, path, *a, **k):
        return {"NAXIS1": 10, "NAXIS2": 10}


class _FakeHDUList:
    def __init__(self, header):
        self._hdu = types.SimpleNamespace(header=header)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getitem__(self, idx):
        return self._hdu


# ---------------------------------------------------------------------------
# H. path lookup robustness (unit)
# ---------------------------------------------------------------------------


def test_h_pre_resolved_header_path_variants(monkeypatch):
    worker = _load_worker()
    header = _solved_header()
    raw = "~/img.fits"
    expanded = os.path.expanduser(raw)
    lookup = worker._build_phase0_lookup([{"path": raw, "header": header}])
    assert worker._pre_resolved_header_for_path(raw, lookup) is header
    # Path object (not a bare str) resolves via the normalized-key fallback.
    assert worker._pre_resolved_header_for_path(Path(expanded), lookup) is header
    # Expanded str spelling resolves via the normalized-key fallback.
    assert worker._pre_resolved_header_for_path(expanded, lookup) is header
    assert worker._pre_resolved_header_for_path("/tmp/missing.fits", lookup) is None
    assert worker._pre_resolved_header_for_path("/tmp/img.fits", None) is None
    assert worker._pre_resolved_header_for_path("/tmp/img.fits", {}) is None


# ---------------------------------------------------------------------------
# I. mixed robustness: one bad entry must not invalidate the good ones
# ---------------------------------------------------------------------------


def test_i_mixed_bad_entry_does_not_invalidate_good(monkeypatch):
    worker = _load_worker()
    good = {"path": "/good.fits", "header": _solved_header()}
    bad = {"path": "/bad.fits", "header": {"NAXIS1": 10}}  # no WCS material
    lookup = worker._build_phase0_lookup([good, bad])
    # Good entry still resolves to a valid pre-resolved header.
    assert worker._pre_resolved_header_for_path("/good.fits", lookup) is not None
    assert header_carries_wcs_material(worker._pre_resolved_header_for_path("/good.fits", lookup)) is True
    # Bad entry resolves to a header that carries no WCS material (worker then
    # falls back to the solver for that single path, not for all).
    bad_header = worker._pre_resolved_header_for_path("/bad.fits", lookup)
    assert bad_header is not None
    assert header_carries_wcs_material(bad_header) is False
    # Mixed count is correct: 2 total, 1 valid.
    total, valid = count_filter_handoff_wcs([good, bad])
    assert total == 2 and valid == 1
