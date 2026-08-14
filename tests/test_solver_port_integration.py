"""Integration wiring tests: worker dispatch + GUI option + packaging contract.

These are static source-inspection tests (consistent with the repo's existing
style) because importing ``zemosaic.zemosaic_worker`` pulls the whole heavy
scientific stack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

WORKER = SRC / "zemosaic" / "zemosaic_worker.py"
GUI = SRC / "zemosaic" / "zemosaic_gui_qt.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _worker_source() -> str:
    return WORKER.read_text(encoding="utf-8")


def _gui_source() -> str:
    return GUI.read_text(encoding="utf-8")


def test_worker_routes_dispatch_through_solver_port():
    src = _worker_source()
    assert "_solve_through_solver_port" in src
    assert "solver_port" in src
    assert "SOLVER_CHOICE_ZESOLVER" in src
    # The legacy inline dispatch must be gone (now lives in LegacySolverAdapter).
    assert "solver_choice_effective == \"ASTROMETRY\"" not in src
    assert "solver_choice_effective == \"ANSVR\"" not in src


def test_worker_preserves_legacy_solver_functions():
    src = _worker_source()
    # The legacy entry points are still present and reused by the adapter.
    assert "def solve_with_astrometry(" in src
    assert "def solve_with_ansvr(" in src
    assert "astap_paths_valid" in src
    assert "LegacySolverAdapter" in src
    assert "ZeSolverAdapter" in src


def test_worker_has_no_forbidden_zesolver_imports():
    src = _worker_source()
    # The worker must never import ZeSolver directly (it delegates to the
    # internal zesolver_adapter module).
    assert "from zesolver" not in src
    assert "import zesolver." not in src
    for token in (
        "SolverPipeline",
        "ProductSettings",
        "RuntimeOptions",
        "SolverCatalogResources",
        "zesolver.zeblindsolver",
        "zesolver.zewcs290",
        "zesolver.gpu_support",
        "sys.path.insert",
        "..ZeSolver",
    ):
        assert token not in src, f"forbidden token {token!r} in zemosaic_worker.py"


def test_gui_exposes_zesolver_option():
    src = _gui_source()
    assert "qt_solver_zesolver" in src
    assert '("ZESOLVER"' in src
    assert '"ZESOLVER": zesolver_box' in src
    assert "zesolver_resources_path" in src
    assert "zesolver_gpu_policy" in src


def test_gui_does_not_make_zesolver_default():
    src = _gui_source()
    # The persisted/fallback default remains a legacy solver, never ZeSolver.
    assert '"solver_method": "ansvr"' in src
    # ZeSolver must not appear as the default solver_method value.
    assert '"solver_method": "zesolver"' not in src


def test_pyproject_does_not_require_zesolver():
    src = PYPROJECT.read_text(encoding="utf-8")
    assert "ZeSolver" not in src
    assert "zesolver" not in src


def test_solver_settings_default_is_not_zesolver():
    settings_src = (SRC / "zemosaic" / "solver_settings.py").read_text(encoding="utf-8")
    assert 'solver_choice: str = "ASTAP"' in settings_src


# ---------------------------------------------------------------------------
# Dynamic lifecycle tests: worker adapter factory fingerprint + run-end close.
# These import the heavy worker module lazily (as in production the factory is
# exercised through ``_get_zesolver_adapter``).
# ---------------------------------------------------------------------------


def _load_worker():
    import importlib

    return importlib.import_module("zemosaic.zemosaic_worker")


def test_worker_factory_rebuilds_adapter_on_settings_fingerprint_change(monkeypatch):
    import types

    worker = _load_worker()

    created = []
    closed = []

    class FakeAdapter:
        def __init__(self, *, resources_path=None, gpu_policy=None, network_policy=None):
            self.resources_path = resources_path
            self.gpu_policy = gpu_policy
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True
            closed.append(self)

    fake_module = types.SimpleNamespace(ZeSolverAdapter=FakeAdapter)
    monkeypatch.setattr(worker, "_zesolver_adapter", fake_module)
    monkeypatch.setattr(worker, "_ZESOLVER_ADAPTER_INSTANCE", None)
    monkeypatch.setattr(worker, "_ZESOLVER_ADAPTER_FINGERPRINT", None)

    s1 = {
        "zesolver_resources_path": "/a",
        "zesolver_gpu_policy": "auto",
        "zesolver_backend_policy": "auto",
    }
    a1 = worker._get_zesolver_adapter(s1)
    assert len(created) == 1
    assert created[0].resources_path == "/a"
    assert created[0].gpu_policy == "auto"

    # Identical settings -> cached adapter is reused, nothing closed/rebuilt.
    assert worker._get_zesolver_adapter(s1) is a1
    assert len(created) == 1
    assert closed == []

    # Runtime-affecting settings changed -> stale adapter closed, fresh built.
    s2 = {
        "zesolver_resources_path": "/b",
        "zesolver_gpu_policy": "disabled",
        "zesolver_backend_policy": "near_only",
    }
    a2 = worker._get_zesolver_adapter(s2)
    assert a2 is not a1
    assert len(created) == 2
    assert closed == [a1]
    assert a2.resources_path == "/b"
    assert a2.gpu_policy == "disabled"

    # Same new settings -> reuse the fresh adapter.
    assert worker._get_zesolver_adapter(s2) is a2

    # Run-end hook closes and invalidates the cached adapter.
    worker._close_zesolver_adapter()
    assert a2.closed is True
    assert worker._ZESOLVER_ADAPTER_INSTANCE is None
    assert worker._ZESOLVER_ADAPTER_FINGERPRINT is None

    # A later run builds yet another fresh adapter.
    a3 = worker._get_zesolver_adapter(s2)
    assert a3 is not a2
    assert len(created) == 3


def test_worker_fingerprint_covers_required_settings():
    worker = _load_worker()
    fp = worker._zesolver_settings_fingerprint

    assert fp(
        {
            "zesolver_resources_path": "/x",
            "zesolver_gpu_policy": "auto",
            "zesolver_backend_policy": "auto",
        }
    ) == (
        "zesolver_resources_path=/x",
        "zesolver_gpu_policy=auto",
        "zesolver_backend_policy=auto",
    )
    # Missing settings normalize to empty strings (stable fingerprint).
    assert fp({}) == (
        "zesolver_resources_path=",
        "zesolver_gpu_policy=",
        "zesolver_backend_policy=",
    )
    # Each runtime-affecting key changes the fingerprint.
    assert fp({"zesolver_gpu_policy": "auto"}) != fp({"zesolver_gpu_policy": "disabled"})
    assert fp({"zesolver_resources_path": "/x"}) != fp(
        {"zesolver_resources_path": "/y"}
    )
    assert fp({"zesolver_backend_policy": "auto"}) != fp(
        {"zesolver_backend_policy": "near_only"}
    )


def test_worker_run_entrypoints_are_decorated_with_close_hook():
    worker = _load_worker()
    import inspect

    for fn in (
        worker.run_hierarchical_mosaic,
        worker.run_hierarchical_mosaic_classic_legacy,
    ):
        assert getattr(fn, "__wrapped__", None) is not None, (
            f"{fn.__name__} is not wrapped by the run-exit close hook"
        )
        # The public signature must remain unchanged for the process wrapper.
        assert "solver_settings" in inspect.signature(fn).parameters
