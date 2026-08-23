"""Focused tests for the optional ZeSolver API v1 adapter.

These tests never require a real ZeSolver installation: they inject a fake
``zesolver.api.v1`` module into ``sys.modules`` via monkeypatch and exercise the
discovery, request mapping, result conversion and failure handling logic.
"""

from __future__ import annotations

import enum
import importlib
import re
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.zesolver_adapter import (  # noqa: E402
    ZeSolverAdapter,
    discover_zesolver,
)
from zemosaic.solver_port import DiscoveryState, SolveStatus  # noqa: E402


# ---------------------------------------------------------------------------
# Fake ``zesolver.api.v1`` helpers
# ---------------------------------------------------------------------------


class _FakeNetworkPolicy(enum.Enum):
    DISABLED = "disabled"
    ALLOWED = "allowed"


class _FakeGpuPolicy(enum.Enum):
    AUTO = "auto"
    DISABLED = "disabled"
    REQUIRED = "required"


class _FakeBackendPolicy(enum.Enum):
    AUTO = "auto"
    NEAR_ONLY = "near_only"
    BLIND_ONLY = "blind_only"


class _FakeSolveStatus(enum.Enum):
    SOLVED = "solved"
    SKIPPED_EXISTING_WCS = "skipped_existing_wcs"
    FAILED = "failed"
    CANCELLED = "cancelled"


class _FakeFailureCode(enum.Enum):
    NO_SOLUTION = "no_solution"
    MISSING_RESOURCE = "missing_resource"


class _FakeCancellationToken:
    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self):
        return self._cancelled


def _install_package_stubs(monkeypatch, v1: types.ModuleType) -> None:
    pkg = types.ModuleType("zesolver")
    pkg.__path__ = []
    api = types.ModuleType("zesolver.api")
    api.__path__ = []
    monkeypatch.setitem(sys.modules, "zesolver", pkg)
    monkeypatch.setitem(sys.modules, "zesolver.api", api)
    monkeypatch.setitem(sys.modules, "zesolver.api.v1", v1)


def _make_v1(
    *,
    api_version="1.0",
    api_major=1,
    probe=None,
    product_version="1.2.3",
    supported_capabilities=("near_solve", "blind_solve", "wcs_write", "gpu", "cancel"),
):
    v1 = types.ModuleType("zesolver.api.v1")
    v1.API_VERSION = api_version
    v1.API_MAJOR = api_major
    v1.probe = probe if probe is not None else (lambda **kw: None)

    def get_api_info():
        return types.SimpleNamespace(
            product_version=product_version,
            supported_capabilities=supported_capabilities,
        )

    v1.get_api_info = get_api_info
    return v1


def _remove_zesolver(monkeypatch) -> None:
    for key in list(sys.modules):
        if key == "zesolver" or key.startswith("zesolver"):
            monkeypatch.delitem(sys.modules, key, raising=False)


def _block_zesolver_public_import(monkeypatch) -> None:
    real_import_module = importlib.import_module

    def _blocked_import(name, *args, **kwargs):
        if name == "zesolver.api.v1":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _blocked_import)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_not_installed(monkeypatch):
    _remove_zesolver(monkeypatch)
    _block_zesolver_public_import(monkeypatch)
    discovery = discover_zesolver()
    assert discovery.state is DiscoveryState.NOT_INSTALLED


def test_discover_available_v1(monkeypatch):
    v1 = _make_v1(api_version="1.0", api_major=1)
    _install_package_stubs(monkeypatch, v1)
    discovery = discover_zesolver()
    assert discovery.state is DiscoveryState.AVAILABLE
    assert discovery.api_version == "1.0"
    assert discovery.product_version == "1.2.3"


def test_discover_incompatible_v2(monkeypatch):
    v1 = _make_v1(api_version="2.0", api_major=2)
    _install_package_stubs(monkeypatch, v1)
    discovery = discover_zesolver()
    assert discovery.state is DiscoveryState.INCOMPATIBLE


def test_discover_incompatible_missing_major_parses_from_version(monkeypatch):
    v1 = _make_v1(api_version="3.1")
    # No API_MAJOR attribute -> must parse "3" from API_VERSION and reject.
    del v1.API_MAJOR
    _install_package_stubs(monkeypatch, v1)
    discovery = discover_zesolver()
    assert discovery.state is DiscoveryState.INCOMPATIBLE


def test_discover_unhealthy_when_probe_raises(monkeypatch):
    def probe(**kwargs):
        raise RuntimeError("boom")

    v1 = _make_v1(api_version="1.0", api_major=1, probe=probe)
    _install_package_stubs(monkeypatch, v1)
    discovery = discover_zesolver()
    assert discovery.state is DiscoveryState.UNHEALTHY


# ---------------------------------------------------------------------------
# Adapter request mapping + result conversion (full fake v1)
# ---------------------------------------------------------------------------


def _make_full_v1():
    rec = types.SimpleNamespace(
        solve_options_kwargs=None,
        solve_request_input=None,
        solve_request_hints=None,
        create_runtime_kwargs=None,
        create_runtime_calls=0,
        create_session_calls=0,
        runtime_close_calls=0,
        session_close_calls=0,
        solve_calls=0,
        result=None,
        solve_exc=None,
        runtimes=[],
    )

    class FakeSolveHints:
        def __init__(self, **kwargs):
            rec.solve_hints_kwargs = kwargs

    class FakeSolveOptions:
        def __init__(self, **kwargs):
            rec.solve_options_kwargs = kwargs

    class FakeSolveRequest:
        def __init__(self, input_path, hints=None, options=None):
            rec.solve_request_input = input_path
            rec.solve_request_hints = hints
            rec.solve_request_options = options

    class FakeCanonicalWcsHeader:
        def to_fits_header(self):
            return {"CRVAL1": 10.0}

        def to_astropy_wcs(self):
            return "FAKE_WCS"

    class FakeSolveResult:
        def __init__(self, status, wcs_header=None, failure_code=None, message=None):
            self.status = status
            self.wcs_header = wcs_header
            self.failure_code = failure_code
            self.message = message

    class FakeSession:
        def solve(self, request, cancellation=None, progress=None):
            rec.solve_calls += 1
            rec.last_cancellation = cancellation
            rec.last_progress = progress
            if rec.solve_exc is not None:
                raise rec.solve_exc
            return rec.result

        def close(self):
            rec.session_close_calls += 1

    class FakeRuntime:
        def __init__(self):
            self._session = FakeSession()

        def create_session(self):
            rec.create_session_calls += 1
            return self._session

        def close(self):
            rec.runtime_close_calls += 1

    def create_solver_runtime(**kwargs):
        rec.create_runtime_calls += 1
        rec.create_runtime_kwargs = kwargs
        runtime = FakeRuntime()
        rec.runtimes.append(runtime)
        rec.runtime = runtime
        return runtime

    v1 = types.ModuleType("zesolver.api.v1")
    v1.SolveHints = FakeSolveHints
    v1.SolveOptions = FakeSolveOptions
    v1.SolveRequest = FakeSolveRequest
    v1.CanonicalWcsHeader = FakeCanonicalWcsHeader
    v1.SolveResult = FakeSolveResult
    v1.SolveStatus = _FakeSolveStatus
    v1.FailureCode = _FakeFailureCode
    v1.NetworkPolicy = _FakeNetworkPolicy
    v1.GpuPolicy = _FakeGpuPolicy
    v1.BackendPolicy = _FakeBackendPolicy
    v1.CancellationToken = _FakeCancellationToken
    v1.create_solver_runtime = create_solver_runtime
    v1.FakeRuntime = FakeRuntime
    return v1, rec


def test_adapter_maps_request_and_converts_solved_result(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = v1.SolveResult(
        _FakeSolveStatus.SOLVED,
        wcs_header=v1.CanonicalWcsHeader(),
    )
    _install_package_stubs(monkeypatch, v1)

    adapter = ZeSolverAdapter()
    outcome = adapter.solve(
        image_fits_path="/tmp/img.fits",
        fits_header={"NAXIS1": 100},
        settings={"zesolver_ra_deg": 12.0, "zesolver_dec_deg": 34.0},
        progress_callback=None,
    )

    assert rec.solve_request_input == Path("/tmp/img.fits")
    assert rec.solve_hints_kwargs == {"ra_deg": 12.0, "dec_deg": 34.0}
    # Network policy is always disabled.
    assert rec.solve_options_kwargs["network_policy"] is _FakeNetworkPolicy.DISABLED

    assert outcome.status is SolveStatus.SOLVED
    assert outcome.wcs == "FAKE_WCS"
    assert outcome.header == {"NAXIS1": 100, "CRVAL1": 10.0}
    # ZeSolver (OVERWRITE_INPUT) already wrote the WCS -> no second write.
    assert outcome.should_write_header_back is False
    assert outcome.backend_used == "ZESOLVER"


def test_adapter_runtime_and_session_lifecycle(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = v1.SolveResult(_FakeSolveStatus.SOLVED, wcs_header=v1.CanonicalWcsHeader())
    _install_package_stubs(monkeypatch, v1)

    adapter = ZeSolverAdapter(gpu_policy="disabled")
    for _ in range(3):
        adapter.solve(
            image_fits_path="/tmp/img.fits",
            fits_header={},
            settings={},
            progress_callback=None,
        )

    # One runtime per adapter instance (per process), one session per thread.
    assert rec.create_runtime_calls == 1
    assert rec.create_runtime_kwargs["gpu_policy"] is _FakeGpuPolicy.DISABLED
    assert rec.create_session_calls == 1
    assert rec.solve_calls == 3


def test_adapter_close_is_idempotent_and_recreates_runtime(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = v1.SolveResult(
        _FakeSolveStatus.SOLVED, wcs_header=v1.CanonicalWcsHeader()
    )
    _install_package_stubs(monkeypatch, v1)

    adapter = ZeSolverAdapter()
    adapter.solve(
        image_fits_path="/tmp/img_a.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert rec.create_runtime_calls == 1
    assert rec.create_session_calls == 1
    assert rec.runtime_close_calls == 0

    # close() tears down the runtime and its session exactly once.
    adapter.close()
    assert rec.runtime_close_calls == 1
    assert rec.session_close_calls == 1

    # Idempotent: a second close() is a no-op on the (already closed) runtime.
    adapter.close()
    assert rec.runtime_close_calls == 1
    assert rec.session_close_calls == 1

    # A subsequent solve must create a fresh runtime and a fresh session.
    adapter.solve(
        image_fits_path="/tmp/img_b.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert rec.create_runtime_calls == 2
    assert rec.create_session_calls == 2
    # The original runtime was closed exactly once; the new one is still open.
    assert rec.runtime_close_calls == 1


def test_adapter_close_without_runtime_is_noop(monkeypatch):
    v1, rec = _make_full_v1()
    _install_package_stubs(monkeypatch, v1)

    adapter = ZeSolverAdapter()
    # Closing an adapter that never solved must not raise nor touch the API.
    adapter.close()
    adapter.close()
    assert rec.create_runtime_calls == 0
    assert rec.runtime_close_calls == 0
    assert rec.session_close_calls == 0


def test_adapter_expected_failure_returns_failed(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = v1.SolveResult(
        _FakeSolveStatus.FAILED,
        failure_code=_FakeFailureCode.NO_SOLUTION,
        message="no stars",
    )
    _install_package_stubs(monkeypatch, v1)

    outcome = ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert outcome.status is SolveStatus.FAILED
    assert outcome.failure_code == "no_solution"
    assert outcome.message == "no stars"
    assert outcome.wcs is None
    assert outcome.should_write_header_back is False


def test_adapter_unexpected_exception_returns_failed_not_raises(monkeypatch):
    v1, rec = _make_full_v1()
    rec.solve_exc = RuntimeError("engine exploded")
    _install_package_stubs(monkeypatch, v1)

    outcome = ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert outcome.status is SolveStatus.FAILED
    assert outcome.failure_code == "unexpected_error"
    assert "engine exploded" in outcome.message


# ---------------------------------------------------------------------------
# Public-import-only contract (static source inspection)
# ---------------------------------------------------------------------------


FORBIDDEN_TOKENS = (
    "SolverPipeline",
    "ProductSettings",
    "RuntimeOptions",
    "SolverCatalogResources",
    "zeblindsolver",
    "zewcs290",
    "gpu_support",
    "sys.path",
    "..ZeSolver",
    "profiles.",
)


def test_adapter_source_uses_only_public_api():
    source = (SRC / "zemosaic" / "zesolver_adapter.py").read_text(encoding="utf-8")
    for token in FORBIDDEN_TOKENS:
        assert token not in source, f"forbidden token {token!r} in zesolver_adapter.py"
    # Only the public v1 module is referenced.
    assert "zesolver.api.v1" in source
    # Every ``zesolver.<submodule>`` reference must be the public ``api`` path.
    for submodule in re.findall(r"zesolver\.([A-Za-z0-9_]+)", source):
        assert submodule == "api", f"unexpected zesolver reference: zesolver.{submodule}"


def test_solver_port_source_has_no_zesolver_import():
    source = (SRC / "zemosaic" / "solver_port.py").read_text(encoding="utf-8")
    assert "import zesolver" not in source
    assert "from zesolver" not in source
