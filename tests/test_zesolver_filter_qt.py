"""Focused tests for ZeSolver integration in the Qt Filter workflow.

These tests never require a real ZeSolver installation or a real ASTAP binary:
they inject fake ``zesolver.api.v1`` modules (as the adapter tests do), monkeypatch
the ASTAP callable, and exercise the ``_DirectoryScanWorker`` dispatch logic,
write-WCS semantics, cancellation and cleanup paths.
"""

from __future__ import annotations

import enum
import re
import sys
import threading
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
)
from zemosaic.solver_port import SolverOutcome, SolveStatus  # noqa: E402

FILTER_SOURCE = SRC / "zemosaic" / "zemosaic_filter_gui_qt.py"


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    """Ensure a QCoreApplication exists for QObject/signal use in tests."""
    try:
        from PySide6.QtCore import QCoreApplication

        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication([])
        return app
    except Exception:  # pragma: no cover - PySide6 absent
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wcs():
    return types.SimpleNamespace(is_celestial=True)


def _make_item(path: str) -> _NormalizedItem:
    return _NormalizedItem(
        original={"path": path, "path_raw": path},
        display_name=Path(path).name,
        file_path=path,
        has_wcs=False,
        instrument=None,
        group_label=None,
    )


def _worker(settings, *, write_wcs=False):
    return _DirectoryScanWorker(
        [],
        settings,
        _FallbackLocalizer(),
        astap_overrides={"write_wcs_to_file": write_wcs},
    )


def _astap_cfg() -> dict:
    return {
        "exe": "/usr/bin/astap",
        "data": "/data",
        "radius": 3.0,
        "downsample": 2,
        "sensitivity": 100,
        "timeout": 180,
        "astap_drizzled_fallback_enabled": False,
    }


def _write_fits_without_wcs(path: Path) -> None:
    import numpy as np
    from astropy.io import fits

    hdu = fits.PrimaryHDU(data=np.zeros((10, 10), dtype=np.float32))
    hdu.writeto(str(path), overwrite=True)


def _remove_zesolver(monkeypatch) -> None:
    for key in list(sys.modules):
        if key == "zesolver" or key.startswith("zesolver"):
            monkeypatch.delitem(sys.modules, key, raising=False)


def _install_fake_zesolver(
    monkeypatch,
    *,
    api_major: int = 1,
    probe_raises: bool = False,
    status: str = "solved",
    supported_capabilities=("near_solve", "blind_solve", "wcs_write", "gpu", "cancel"),
    probe_capabilities=None,
    solve_blocks=False,
):
    """Install a fake ``zesolver.api.v1`` and return an instrumented recorder."""
    rec = types.SimpleNamespace(
        solve_calls=0,
        solve_started=threading.Event(),
        release_solve=threading.Event(),
        last_cancellation=None,
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
            self._cancelled = False

        def cancel(self):
            self._cancelled = True

        def is_cancelled(self):
            return self._cancelled

    class _WcsHeader:
        def to_fits_header(self):
            return {"CRVAL1": 10.0, "CRVAL2": 20.0}

        def to_astropy_wcs(self):
            return "FAKE_WCS"

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
            rec.last_progress = progress
            if solve_blocks:
                rec.solve_started.set()
                rec.release_solve.wait(timeout=10)
                if cancellation is not None and cancellation.is_cancelled():
                    return _SolveResult(_SolveStatus.CANCELLED, message="cancelled")
            if status == "solved":
                return _SolveResult(_SolveStatus.SOLVED, wcs_header=_WcsHeader())
            return _SolveResult(
                _SolveStatus.FAILED,
                failure_code=_FailureCode.NO_SOLUTION,
                message="no stars",
            )

        def close(self):
            rec.session_close_calls += 1

    class _Runtime:
        def create_session(self):
            return _Session()

        def close(self):
            rec.runtime_close_calls += 1

    v1 = types.ModuleType("zesolver.api.v1")
    v1.API_VERSION = f"{api_major}.0"
    v1.API_MAJOR = api_major

    def get_api_info():
        return types.SimpleNamespace(
            product_version="1.2.3",
            supported_capabilities=supported_capabilities,
        )

    v1.get_api_info = get_api_info

    if probe_raises:
        def probe(**kw):
            raise RuntimeError("boom")
        v1.probe = probe
    else:
        def probe(**kw):
            caps = tuple(probe_capabilities) if probe_capabilities is not None else ()
            return types.SimpleNamespace(capabilities=caps)
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
    return rec


# ---------------------------------------------------------------------------
# Optional-dependency integrity + public-API boundary
# ---------------------------------------------------------------------------


def test_filter_import_does_not_pull_zesolver():
    # Importing the Filter module must never import the optional zesolver package.
    assert "zesolver" not in sys.modules
    assert not any(k == "zesolver" or k.startswith("zesolver.") for k in sys.modules)
    # The internal adapter module is present and the boundary flags are set.
    assert qt_filter._ZESOLVER_ADAPTER_AVAILABLE is True
    assert qt_filter.SOLVER_CHOICE_ZESOLVER == "ZESOLVER"


def test_filter_source_uses_only_public_zesolver_boundary():
    src = FILTER_SOURCE.read_text(encoding="utf-8")
    lowered = src.lower()
    for token in (
        "zealfie",
        "imagesolver",
        "solveconfig",
        "imagesolveresult",
        "zeblindsolver",
        "zewcs290",
        "gpu_support",
        "..zesolver",
    ):
        assert token not in lowered, f"forbidden token {token!r} in filter source"
    # No direct import of the external zesolver package (the internal
    # ``zemosaic.zesolver_adapter`` module is the only bridge, imported
    # relatively).
    assert re.search(r"\bimport zesolver\b", src) is None
    assert re.search(r"\bfrom zesolver\b", src) is None
    # No sys.path manipulation or sibling-checkout resolution.
    assert "sys.path.insert" not in src
    assert "sys.path.append" not in src


# ---------------------------------------------------------------------------
# Solver choice resolution
# ---------------------------------------------------------------------------


def test_resolve_solver_choice_defaults_to_astap():
    assert _worker({})._resolve_solver_choice() == "ASTAP"
    assert _worker({"solver_choice": "astap"})._resolve_solver_choice() == "ASTAP"


def test_resolve_solver_choice_zesolver():
    assert _worker({"solver_choice": "ZESOLVER"})._resolve_solver_choice() == "ZESOLVER"
    assert _worker({"solver_choice": "zesolver"})._resolve_solver_choice() == "ZESOLVER"


# ---------------------------------------------------------------------------
# ZeSolver discovery gating (active-for-run)
# ---------------------------------------------------------------------------


def test_zesolver_active_for_run_available(monkeypatch):
    _install_fake_zesolver(monkeypatch, status="solved")
    worker = _worker({"solver_choice": "ZESOLVER"})
    assert worker._zesolver_active_for_run() is True


def test_zesolver_active_for_run_not_installed(monkeypatch):
    _remove_zesolver(monkeypatch)
    worker = _worker({"solver_choice": "ZESOLVER"})
    assert worker._zesolver_active_for_run() is False


def test_zesolver_active_for_run_incompatible(monkeypatch):
    _install_fake_zesolver(monkeypatch, api_major=2)
    worker = _worker({"solver_choice": "ZESOLVER"})
    assert worker._zesolver_active_for_run() is False


def test_zesolver_active_for_run_unhealthy(monkeypatch):
    _install_fake_zesolver(monkeypatch, probe_raises=True)
    worker = _worker({"solver_choice": "ZESOLVER"})
    assert worker._zesolver_active_for_run() is False


# ---------------------------------------------------------------------------
# Per-file dispatch: ZeSolver with ASTAP fallback
# ---------------------------------------------------------------------------


def test_zesolver_solved_uses_zesolver_and_not_astap(monkeypatch):
    astap_calls = []
    monkeypatch.setattr(
        qt_filter,
        "solve_with_astap",
        lambda *a, **k: astap_calls.append(1) or _make_wcs(),
    )
    worker = _worker({"solver_choice": "ZESOLVER"})
    monkeypatch.setattr(
        worker,
        "_run_zesolver_solve",
        lambda image_path, header_obj, write_inplace: SolverOutcome(
            status=SolveStatus.SOLVED,
            wcs=_make_wcs(),
            header={"CRVAL1": 10.0},
            backend_used="ZESOLVER",
        ),
    )
    entry = _make_item("/tmp/img.fits")
    payload: dict = {}
    idx, row = worker._solve_zesolver_with_fallback(
        0, entry, "/tmp/img.fits", {"NAXIS1": 100}, {}, payload, _astap_cfg()
    )
    assert row["solver"] == "ZESOLVER"
    assert row["has_wcs"] is True
    assert payload["solver"] == "ZESOLVER"
    assert payload["has_wcs"] is True
    assert entry.wcs_cache is not None
    assert entry.header_cache is not None
    assert astap_calls == []


def test_zesolver_failed_falls_back_to_astap(monkeypatch):
    astap_calls = []
    monkeypatch.setattr(
        qt_filter,
        "solve_with_astap",
        lambda *a, **k: astap_calls.append(1) or _make_wcs(),
    )
    worker = _worker({"solver_choice": "ZESOLVER"})
    monkeypatch.setattr(
        worker,
        "_run_zesolver_solve",
        lambda image_path, header_obj, write_inplace: SolverOutcome(
            status=SolveStatus.FAILED, backend_used="ZESOLVER"
        ),
    )
    entry = _make_item("/tmp/img.fits")
    payload: dict = {}
    idx, row = worker._solve_zesolver_with_fallback(
        0, entry, "/tmp/img.fits", {"NAXIS1": 100}, {}, payload, _astap_cfg()
    )
    assert row["solver"] == "ASTAP"
    assert row["has_wcs"] is True
    assert payload["solver"] == "ASTAP"
    assert len(astap_calls) == 1


def test_zesolver_failed_no_astap_when_unconfigured(monkeypatch):
    def fake_astap(*a, **k):
        raise AssertionError("ASTAP must not be called when unconfigured")

    monkeypatch.setattr(qt_filter, "solve_with_astap", fake_astap)
    worker = _worker({"solver_choice": "ZESOLVER"})
    monkeypatch.setattr(
        worker,
        "_run_zesolver_solve",
        lambda image_path, header_obj, write_inplace: SolverOutcome(
            status=SolveStatus.FAILED, backend_used="ZESOLVER"
        ),
    )
    entry = _make_item("/tmp/img.fits")
    idx, row = worker._solve_zesolver_with_fallback(
        0, entry, "/tmp/img.fits", {"NAXIS1": 100}, {}, {}, None
    )
    assert "solver" not in row or row.get("solver") is None
    assert "error" in row


def test_zesolver_cancelled_does_not_fallback_to_astap(monkeypatch):
    def fake_astap(*a, **k):
        raise AssertionError("ASTAP must not be called after cancellation")

    monkeypatch.setattr(qt_filter, "solve_with_astap", fake_astap)
    worker = _worker({"solver_choice": "ZESOLVER"})
    monkeypatch.setattr(
        worker,
        "_run_zesolver_solve",
        lambda image_path, header_obj, write_inplace: SolverOutcome(
            status=SolveStatus.CANCELLED, backend_used="ZESOLVER"
        ),
    )
    entry = _make_item("/tmp/img.fits")
    idx, row = worker._solve_zesolver_with_fallback(
        0, entry, "/tmp/img.fits", {"NAXIS1": 100}, {}, {}, _astap_cfg()
    )
    assert "error" in row
    assert "cancelled" in row["error"].lower()


def test_zesolver_stop_requested_no_fallback(monkeypatch):
    def fake_astap(*a, **k):
        raise AssertionError("ASTAP must not be called after stop")

    monkeypatch.setattr(qt_filter, "solve_with_astap", fake_astap)
    worker = _worker({"solver_choice": "ZESOLVER"})
    worker._stop_requested = True
    monkeypatch.setattr(
        worker,
        "_run_zesolver_solve",
        lambda image_path, header_obj, write_inplace: SolverOutcome(
            status=SolveStatus.FAILED, backend_used="ZESOLVER"
        ),
    )
    entry = _make_item("/tmp/img.fits")
    idx, row = worker._solve_zesolver_with_fallback(
        0, entry, "/tmp/img.fits", {"NAXIS1": 100}, {}, {}, _astap_cfg()
    )
    assert "cancelled" in row.get("error", "").lower()


# ---------------------------------------------------------------------------
# Write-WCS semantics (temp-copy vs in-place)
# ---------------------------------------------------------------------------


def test_zesolver_write_off_solves_temp_copy_and_preserves_original(monkeypatch, tmp_path):
    src = tmp_path / "img.fits"
    src.write_bytes(b"ORIGINAL")
    received = {}

    class _FakeAdapter:
        def solve(self, **kwargs):
            received["image_fits_path"] = kwargs["image_fits_path"]
            return SolverOutcome(
                status=SolveStatus.SOLVED,
                wcs=_make_wcs(),
                header={"CRVAL1": 1.0},
            )

        def close(self):
            pass

        def cancel_active_solve(self):
            pass

    worker = _worker({"solver_choice": "ZESOLVER"}, write_wcs=False)
    worker._zesolver_adapter = _FakeAdapter()
    outcome = worker._run_zesolver_solve(str(src), {"NAXIS1": 10}, write_inplace=False)

    solved_path = received["image_fits_path"]
    assert str(solved_path) != str(src)
    # Original never mutated.
    assert src.read_bytes() == b"ORIGINAL"
    # Temp copy cleaned up.
    assert not Path(str(solved_path)).exists()
    assert outcome.status is SolveStatus.SOLVED


def test_zesolver_write_on_solves_in_place(monkeypatch, tmp_path):
    src = tmp_path / "img.fits"
    src.write_bytes(b"ORIGINAL")
    received = {}

    class _FakeAdapter:
        def solve(self, **kwargs):
            received["image_fits_path"] = kwargs["image_fits_path"]
            return SolverOutcome(
                status=SolveStatus.SOLVED,
                wcs=_make_wcs(),
                header={"CRVAL1": 1.0},
            )

        def close(self):
            pass

        def cancel_active_solve(self):
            pass

    worker = _worker({"solver_choice": "ZESOLVER"}, write_wcs=True)
    worker._zesolver_adapter = _FakeAdapter()
    worker._run_zesolver_solve(str(src), {"NAXIS1": 10}, write_inplace=True)
    assert str(received["image_fits_path"]) == str(src)


def test_make_and_cleanup_collision_safe_temp_copy(tmp_path):
    src = tmp_path / "img.fits"
    src.write_bytes(b"DATA")
    worker = _worker({"solver_choice": "ASTAP"})
    tmp = worker._make_collision_safe_temp_copy(str(src))
    assert tmp is not None
    assert Path(tmp).exists()
    assert Path(tmp).read_bytes() == b"DATA"
    assert Path(tmp).suffix == ".fits"
    assert Path(tmp) != src
    worker._cleanup_temp_path(tmp)
    assert not Path(tmp).exists()


def test_adapter_close_on_request_stop(monkeypatch):
    closed = []

    class _FakeAdapter:
        def close(self):
            closed.append(1)

        def cancel_active_solve(self):
            pass

    worker = _worker({"solver_choice": "ZESOLVER"})
    worker._zesolver_adapter = _FakeAdapter()
    worker.request_stop()
    assert worker._stop_requested is True
    # request_stop cancels the active solve but does not close (run() finally does).
    assert closed == []


# ---------------------------------------------------------------------------
# End-to-end run() dispatch through the worker
# ---------------------------------------------------------------------------


def _run_worker(worker):
    rows = []
    errors = []
    worker.row_updated.connect(lambda idx, row: rows.append((idx, row)))
    worker.error.connect(lambda msg: errors.append(msg))
    worker.run()
    return rows, errors


def test_run_astap_selected_uses_astap(monkeypatch, tmp_path):
    _remove_zesolver(monkeypatch)
    astap_calls = []
    monkeypatch.setattr(
        qt_filter,
        "solve_with_astap",
        lambda *a, **k: astap_calls.append(1) or _make_wcs(),
    )
    fits_path = tmp_path / "img.fits"
    _write_fits_without_wcs(fits_path)
    astap_exe = tmp_path / "astap"
    astap_exe.write_bytes(b"")

    worker = _DirectoryScanWorker(
        [_make_item(str(fits_path))],
        {
            "solver_choice": "ASTAP",
            "astap_executable_path": str(astap_exe),
            "astap_data_directory_path": str(tmp_path),
        },
        _FallbackLocalizer(),
        astap_overrides={"write_wcs_to_file": False},
    )
    rows, errors = _run_worker(worker)
    assert len(rows) == 1
    assert rows[0][1]["solver"] == "ASTAP"
    assert rows[0][1]["has_wcs"] is True
    assert astap_calls == [1]


def test_run_zesolver_selected_solves_with_zesolver(monkeypatch, tmp_path):
    _install_fake_zesolver(monkeypatch, status="solved")
    astap_calls = []
    monkeypatch.setattr(
        qt_filter,
        "solve_with_astap",
        lambda *a, **k: astap_calls.append(1) or _make_wcs(),
    )
    fits_path = tmp_path / "img.fits"
    _write_fits_without_wcs(fits_path)
    astap_exe = tmp_path / "astap"
    astap_exe.write_bytes(b"")

    items = [_make_item(str(fits_path))]
    worker = _DirectoryScanWorker(
        items,
        {
            "solver_choice": "ZESOLVER",
            "astap_executable_path": str(astap_exe),
            "astap_data_directory_path": str(tmp_path),
        },
        _FallbackLocalizer(),
        astap_overrides={"write_wcs_to_file": False},
    )
    rows, errors = _run_worker(worker)
    assert len(rows) == 1
    assert rows[0][1]["solver"] == "ZESOLVER"
    assert rows[0][1]["has_wcs"] is True
    assert astap_calls == []
    assert items[0].wcs_cache == "FAKE_WCS"


def test_run_zesolver_not_installed_falls_back_to_astap(monkeypatch, tmp_path):
    _remove_zesolver(monkeypatch)
    astap_calls = []
    monkeypatch.setattr(
        qt_filter,
        "solve_with_astap",
        lambda *a, **k: astap_calls.append(1) or _make_wcs(),
    )
    fits_path = tmp_path / "img.fits"
    _write_fits_without_wcs(fits_path)
    astap_exe = tmp_path / "astap"
    astap_exe.write_bytes(b"")

    worker = _DirectoryScanWorker(
        [_make_item(str(fits_path))],
        {
            "solver_choice": "ZESOLVER",
            "astap_executable_path": str(astap_exe),
            "astap_data_directory_path": str(tmp_path),
        },
        _FallbackLocalizer(),
        astap_overrides={"write_wcs_to_file": False},
    )
    rows, errors = _run_worker(worker)
    assert len(rows) == 1
    assert rows[0][1]["solver"] == "ASTAP"
    assert astap_calls == [1]


def test_run_zesolver_incompatible_falls_back_to_astap(monkeypatch, tmp_path):
    _install_fake_zesolver(monkeypatch, api_major=2)
    astap_calls = []
    monkeypatch.setattr(
        qt_filter,
        "solve_with_astap",
        lambda *a, **k: astap_calls.append(1) or _make_wcs(),
    )
    fits_path = tmp_path / "img.fits"
    _write_fits_without_wcs(fits_path)
    astap_exe = tmp_path / "astap"
    astap_exe.write_bytes(b"")

    worker = _DirectoryScanWorker(
        [_make_item(str(fits_path))],
        {
            "solver_choice": "ZESOLVER",
            "astap_executable_path": str(astap_exe),
            "astap_data_directory_path": str(tmp_path),
        },
        _FallbackLocalizer(),
        astap_overrides={"write_wcs_to_file": False},
    )
    rows, errors = _run_worker(worker)
    assert len(rows) == 1
    assert rows[0][1]["solver"] == "ASTAP"
    assert astap_calls == [1]


def test_run_zesolver_unhealthy_falls_back_to_astap(monkeypatch, tmp_path):
    _install_fake_zesolver(monkeypatch, probe_raises=True)
    astap_calls = []
    monkeypatch.setattr(
        qt_filter,
        "solve_with_astap",
        lambda *a, **k: astap_calls.append(1) or _make_wcs(),
    )
    fits_path = tmp_path / "img.fits"
    _write_fits_without_wcs(fits_path)
    astap_exe = tmp_path / "astap"
    astap_exe.write_bytes(b"")

    worker = _DirectoryScanWorker(
        [_make_item(str(fits_path))],
        {
            "solver_choice": "ZESOLVER",
            "astap_executable_path": str(astap_exe),
            "astap_data_directory_path": str(tmp_path),
        },
        _FallbackLocalizer(),
        astap_overrides={"write_wcs_to_file": False},
    )
    rows, errors = _run_worker(worker)
    assert len(rows) == 1
    assert rows[0][1]["solver"] == "ASTAP"
    assert astap_calls == [1]


def test_run_zesolver_missing_capability_falls_back_to_astap(monkeypatch, tmp_path):
    # Missing required ``wcs_write`` capability -> INCOMPATIBLE -> ASTAP fallback.
    _install_fake_zesolver(
        monkeypatch,
        supported_capabilities=("near_solve", "blind_solve", "gpu", "cancel"),
    )
    astap_calls = []
    monkeypatch.setattr(
        qt_filter,
        "solve_with_astap",
        lambda *a, **k: astap_calls.append(1) or _make_wcs(),
    )
    fits_path = tmp_path / "img.fits"
    _write_fits_without_wcs(fits_path)
    astap_exe = tmp_path / "astap"
    astap_exe.write_bytes(b"")

    worker = _DirectoryScanWorker(
        [_make_item(str(fits_path))],
        {
            "solver_choice": "ZESOLVER",
            "astap_executable_path": str(astap_exe),
            "astap_data_directory_path": str(tmp_path),
        },
        _FallbackLocalizer(),
        astap_overrides={"write_wcs_to_file": False},
    )
    rows, errors = _run_worker(worker)
    assert len(rows) == 1
    assert rows[0][1]["solver"] == "ASTAP"
    assert astap_calls == [1]


def test_run_zesolver_cancellation_stops_and_no_astap(monkeypatch, tmp_path):
    rec = _install_fake_zesolver(monkeypatch, status="solved", solve_blocks=True)
    astap_calls = []
    monkeypatch.setattr(
        qt_filter,
        "solve_with_astap",
        lambda *a, **k: astap_calls.append(1) or _make_wcs(),
    )
    fits_path = tmp_path / "img.fits"
    _write_fits_without_wcs(fits_path)
    astap_exe = tmp_path / "astap"
    astap_exe.write_bytes(b"")

    worker = _DirectoryScanWorker(
        [_make_item(str(fits_path))],
        {
            "solver_choice": "ZESOLVER",
            "astap_executable_path": str(astap_exe),
            "astap_data_directory_path": str(tmp_path),
        },
        _FallbackLocalizer(),
        astap_overrides={"write_wcs_to_file": False},
    )
    rows = []
    from PySide6.QtCore import Qt

    worker.row_updated.connect(
        lambda idx, row: rows.append((idx, row)),
        Qt.ConnectionType.DirectConnection,
    )

    t = threading.Thread(target=worker.run)
    t.start()
    assert rec.solve_started.wait(timeout=10)

    worker.request_stop()
    assert rec.last_cancellation is not None
    assert rec.last_cancellation.is_cancelled() is True

    rec.release_solve.set()
    t.join(timeout=10)
    assert not t.is_alive()

    assert astap_calls == []
    assert len(rows) == 1
    assert "cancelled" in rows[0][1].get("error", "").lower()
