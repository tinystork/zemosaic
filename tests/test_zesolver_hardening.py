"""Phase B hardening tests for the optional ZeSolver v1 adapter + worker wiring.

Covers the seven corrections + lifecycle: WCS overwrite flag mapping, single
FITS write, discovery caching, capability negotiation, broken-vs-absent install
distinction, real progress forwarding, cooperative cancellation, multithread
lifecycle, and the fingerprint double-checked-locking race.  These tests inject
fake ``zesolver.api.v1`` modules (no real ZeSolver needed).
"""

from __future__ import annotations

import enum
import importlib
import sys
import threading
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.zesolver_adapter import ZeSolverAdapter, discover_zesolver  # noqa: E402
from zemosaic.solver_port import (  # noqa: E402
    DiscoveryState,
    OPTIONAL_ZESOLVER_CAPABILITIES,
    REQUIRED_ZESOLVER_CAPABILITIES,
    SolveStatus,
    SolverDiscovery,
    ZESOLVER_SOLVE_BACKEND_CAPABILITIES,
)


# ---------------------------------------------------------------------------
# Fake ``zesolver.api.v1`` building blocks
# ---------------------------------------------------------------------------


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


class _Status(enum.Enum):
    SOLVED = "solved"
    SKIPPED_EXISTING_WCS = "skipped_existing_wcs"
    FAILED = "failed"
    CANCELLED = "cancelled"


class _FailureCode(enum.Enum):
    NO_SOLUTION = "no_solution"
    MISSING_RESOURCE = "missing_resource"


class _Phase(enum.Enum):
    PREPARING = "preparing"
    SOLVING = "solving"
    WRITING = "writing"
    FINALIZING = "finalizing"


class _ProgressEvent:
    def __init__(self, phase, message=None):
        self.phase = phase
        self.message = message


class _CancellationToken:
    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    def is_cancelled(self):
        return self._event.is_set()


class _WcsHeader:
    def to_fits_header(self):
        return {"CRVAL1": 10.0}

    def to_astropy_wcs(self):
        return "FAKE_WCS"


def _install_stubs(monkeypatch, v1: types.ModuleType) -> None:
    pkg = types.ModuleType("zesolver")
    pkg.__path__ = []
    api = types.ModuleType("zesolver.api")
    api.__path__ = []
    monkeypatch.setitem(sys.modules, "zesolver", pkg)
    monkeypatch.setitem(sys.modules, "zesolver.api", api)
    monkeypatch.setitem(sys.modules, "zesolver.api.v1", v1)


def _remove_zesolver(monkeypatch) -> None:
    for key in list(sys.modules):
        if key == "zesolver" or key.startswith("zesolver"):
            monkeypatch.delitem(sys.modules, key, raising=False)


def _make_basic_v1(
    *,
    api_version="1.0",
    api_major=1,
    probe=None,
    product_version="1.2.3",
    supported_capabilities=("near_solve", "blind_solve", "wcs_write", "gpu", "cancel"),
    probe_capabilities=None,
):
    """A fake v1 whose get_api_info / probe honour capability negotiation."""
    v1 = types.ModuleType("zesolver.api.v1")
    v1.API_VERSION = api_version
    v1.API_MAJOR = api_major

    def get_api_info():
        return types.SimpleNamespace(
            product_version=product_version,
            supported_capabilities=supported_capabilities,
        )

    v1.get_api_info = get_api_info

    if probe is not None:
        v1.probe = probe
    else:
        def default_probe(**kw):
            caps = []
            if probe_capabilities is not None:
                caps = probe_capabilities
            return types.SimpleNamespace(capabilities=tuple(caps) if caps else ())

        v1.probe = default_probe
    return v1


def _make_result(status, wcs_header=None, failure_code=None, message=None):
    return types.SimpleNamespace(
        status=status,
        wcs_header=wcs_header,
        failure_code=failure_code,
        message=message,
    )


def _make_full_v1():
    """Instrumented fake v1 used for adapter solve/request/result mapping tests."""
    rec = types.SimpleNamespace(
        solve_options_kwargs=None,
        solve_hints_kwargs=None,
        create_runtime_calls=0,
        create_session_calls=0,
        runtime_close_calls=0,
        session_close_calls=0,
        solve_calls=0,
        result=None,
        solve_exc=None,
        last_cancellation=None,
        last_progress=None,
    )

    class FakeSolveHints:
        def __init__(self, **kwargs):
            rec.solve_hints_kwargs = kwargs

    class FakeSolveOptions:
        def __init__(self, **kwargs):
            rec.solve_options_kwargs = kwargs

    class FakeSolveRequest:
        def __init__(self, input_path, hints=None, options=None):
            pass

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
        def create_session(self):
            rec.create_session_calls += 1
            return FakeSession()

        def close(self):
            rec.runtime_close_calls += 1

    def create_solver_runtime(**kwargs):
        rec.create_runtime_calls += 1
        return FakeRuntime()

    v1 = types.ModuleType("zesolver.api.v1")
    v1.SolveHints = FakeSolveHints
    v1.SolveOptions = FakeSolveOptions
    v1.SolveRequest = FakeSolveRequest
    v1.CanonicalWcsHeader = _WcsHeader
    v1.SolveStatus = _Status
    v1.FailureCode = _FailureCode
    v1.NetworkPolicy = _NetworkPolicy
    v1.GpuPolicy = _GpuPolicy
    v1.BackendPolicy = _BackendPolicy
    v1.CancellationToken = _CancellationToken
    v1.create_solver_runtime = create_solver_runtime
    return v1, rec


# ---------------------------------------------------------------------------
# 1) WCS overwrite flag mapping (force_resolve_existing_wcs -> overwrite_existing_wcs)
# ---------------------------------------------------------------------------


def test_build_request_overwrite_false_by_default(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = _make_result(_Status.SOLVED, wcs_header=_WcsHeader())
    _install_stubs(monkeypatch, v1)

    ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert rec.solve_options_kwargs["overwrite_existing_wcs"] is False


def test_build_request_overwrite_true_when_forced(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = _make_result(_Status.SOLVED, wcs_header=_WcsHeader())
    _install_stubs(monkeypatch, v1)

    ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={"force_resolve_existing_wcs": True},
        progress_callback=None,
    )
    assert rec.solve_options_kwargs["overwrite_existing_wcs"] is True


def test_skipped_existing_wcs_is_not_a_false_skip(monkeypatch):
    """A SKIPPED_EXISTING_WCS must surface as a real failure, never a silent skip."""
    v1, rec = _make_full_v1()
    rec.result = _make_result(_Status.SKIPPED_EXISTING_WCS, message="has wcs")
    _install_stubs(monkeypatch, v1)

    outcome = ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert outcome.status is SolveStatus.FAILED
    assert outcome.failure_code == "existing_wcs_not_overwritten"
    assert outcome.wcs is None


def test_unknown_status_fails_explicitly_not_skipped(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = _make_result("some_future_status")
    _install_stubs(monkeypatch, v1)

    outcome = ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert outcome.status is SolveStatus.FAILED
    assert outcome.failure_code == "unexpected_status"


# ---------------------------------------------------------------------------
# 2) Single FITS write: ZeSolver OVERWRITE_INPUT already writes; no 2nd write.
# ---------------------------------------------------------------------------


def test_solved_result_triggers_exactly_one_effective_write(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = _make_result(_Status.SOLVED, wcs_header=_WcsHeader())
    _install_stubs(monkeypatch, v1)

    outcome = ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={"NAXIS1": 100},
        settings={},
        progress_callback=None,
    )
    assert outcome.status is SolveStatus.SOLVED
    # ZeSolver's OVERWRITE_INPUT is the single effective WCS write; ZeMosaic must
    # NOT write the header back a second time.
    assert outcome.should_write_header_back is False
    # Simulate the worker's write gate verbatim: zero extra writes.
    header_writes = 1 if outcome.should_write_header_back else 0
    assert header_writes == 0


# ---------------------------------------------------------------------------
# 4) Capability negotiation
# ---------------------------------------------------------------------------


def test_capability_constants_are_centralized_and_pinned():
    from zemosaic import solver_port

    assert solver_port.REQUIRED_ZESOLVER_CAPABILITIES == ("wcs_write",)
    assert set(ZESOLVER_SOLVE_BACKEND_CAPABILITIES) == {"near_solve", "blind_solve"}
    assert "cancel" in OPTIONAL_ZESOLVER_CAPABILITIES
    assert "gpu" in OPTIONAL_ZESOLVER_CAPABILITIES


def test_discover_available_with_required_and_backend(monkeypatch):
    v1 = _make_basic_v1(
        supported_capabilities=("near_solve", "blind_solve", "wcs_write", "gpu", "cancel")
    )
    _install_stubs(monkeypatch, v1)
    assert discover_zesolver().state is DiscoveryState.AVAILABLE


def test_discover_required_capability_absent_incompatible(monkeypatch):
    v1 = _make_basic_v1(
        supported_capabilities=("near_solve", "blind_solve", "gpu", "cancel")
    )
    _install_stubs(monkeypatch, v1)
    discovery = discover_zesolver()
    assert discovery.state is DiscoveryState.INCOMPATIBLE
    assert "wcs_write" in discovery.message


def test_discover_optional_capability_absent_is_usable(monkeypatch):
    # No ``cancel`` / ``gpu`` -> still AVAILABLE (optional capabilities never block).
    v1 = _make_basic_v1(supported_capabilities=("near_solve", "wcs_write"))
    _install_stubs(monkeypatch, v1)
    assert discover_zesolver().state is DiscoveryState.AVAILABLE


def test_discover_no_solve_backend_incompatible(monkeypatch):
    v1 = _make_basic_v1(supported_capabilities=("wcs_write", "gpu", "cancel"))
    _install_stubs(monkeypatch, v1)
    discovery = discover_zesolver()
    assert discovery.state is DiscoveryState.INCOMPATIBLE
    assert "solve backend" in discovery.message


def test_discover_required_capability_probe_unavailable(monkeypatch):
    class _Avail(enum.Enum):
        AVAILABLE = "available"
        UNAVAILABLE = "unavailable"
        NOT_CHECKED = "not_checked"

    caps = [
        types.SimpleNamespace(id="wcs_write", availability=_Avail.UNAVAILABLE),
        types.SimpleNamespace(id="near_solve", availability=_Avail.NOT_CHECKED),
    ]
    v1 = _make_basic_v1(
        supported_capabilities=("near_solve", "wcs_write"),
        probe_capabilities=caps,
    )
    _install_stubs(monkeypatch, v1)
    discovery = discover_zesolver()
    assert discovery.state is DiscoveryState.UNHEALTHY
    assert "unavailable" in discovery.message


# ---------------------------------------------------------------------------
# 5) Installed-but-broken != not installed
# ---------------------------------------------------------------------------


def test_discover_broken_internal_import_is_unhealthy(monkeypatch, tmp_path):
    _remove_zesolver(monkeypatch)
    root = tmp_path / "fakepkg"
    (root / "zesolver" / "api" / "v1").mkdir(parents=True)
    (root / "zesolver" / "__init__.py").write_text("", encoding="utf-8")
    (root / "zesolver" / "api" / "__init__.py").write_text("", encoding="utf-8")
    (root / "zesolver" / "api" / "v1" / "__init__.py").write_text(
        "import zesolver.missing_internal_dep\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(root))

    discovery = discover_zesolver()
    # The public module was found but its internal import failed -> UNHEALTHY,
    # never NOT_INSTALLED.
    assert discovery.state is DiscoveryState.UNHEALTHY


def test_discover_not_installed_is_not_installed(monkeypatch):
    _remove_zesolver(monkeypatch)
    assert discover_zesolver().state is DiscoveryState.NOT_INSTALLED


# ---------------------------------------------------------------------------
# 6) Progress forwarding (real phases, no invented percentage / ETA)
# ---------------------------------------------------------------------------


def test_progress_forwarder_maps_real_phases():
    adapter = ZeSolverAdapter()
    received = []

    def cb(key, prog, lvl, **kwargs):
        received.append((key, prog, lvl, kwargs))

    forward = adapter._make_progress_forwarder(cb)
    for phase in (_Phase.PREPARING, _Phase.SOLVING, _Phase.WRITING, _Phase.FINALIZING):
        forward(_ProgressEvent(phase, message=f"msg {phase.value}"))

    assert [r[0] for r in received] == [
        "GetWCS: ZESOLVER preparing",
        "GetWCS: ZESOLVER solving",
        "GetWCS: ZESOLVER writing WCS",
        "GetWCS: ZESOLVER finalizing",
    ]
    # No invented percentage and no ETA: prog stays None, level is debug detail.
    assert all(r[1] is None for r in received)
    assert all(r[2] == "DEBUG_DETAIL" for r in received)
    assert all(r[3]["phase"] in ("preparing", "solving", "writing", "finalizing") for r in received)


def test_solve_wires_progress_to_session(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = _make_result(_Status.SOLVED, wcs_header=_WcsHeader())
    _install_stubs(monkeypatch, v1)

    received = []
    ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={},
        progress_callback=lambda key, prog, lvl, **kw: received.append(key),
    )
    assert callable(rec.last_progress)
    # Invoking the wired forwarder reaches the ZeMosaic callback.
    rec.last_progress(_ProgressEvent(_Phase.SOLVING))
    assert received == ["GetWCS: ZESOLVER solving"]


def test_solve_no_progress_callback_passes_none(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = _make_result(_Status.SOLVED, wcs_header=_WcsHeader())
    _install_stubs(monkeypatch, v1)

    ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert rec.last_progress is None


# ---------------------------------------------------------------------------
# 7) Cooperative cancellation
# ---------------------------------------------------------------------------


def _make_cancel_v1():
    rec = types.SimpleNamespace(
        solve_started=threading.Event(),
        release_solve=threading.Event(),
        last_cancellation=None,
        last_progress=None,
        session_close_calls=0,
        runtime_close_calls=0,
    )

    class FakeSession:
        def solve(self, request, cancellation=None, progress=None):
            rec.last_cancellation = cancellation
            rec.last_progress = progress
            rec.solve_started.set()
            rec.release_solve.wait(timeout=10)
            if cancellation is not None and cancellation.is_cancelled():
                return _make_result(_Status.CANCELLED, message="cancelled")
            return _make_result(_Status.SOLVED, wcs_header=_WcsHeader())

        def close(self):
            rec.session_close_calls += 1

    class FakeRuntime:
        def create_session(self):
            return FakeSession()

        def close(self):
            rec.runtime_close_calls += 1

    v1 = types.ModuleType("zesolver.api.v1")
    v1.SolveHints = lambda **kw: None
    v1.SolveOptions = lambda **kw: None
    v1.SolveRequest = lambda input_path, hints=None, options=None: None
    v1.CanonicalWcsHeader = _WcsHeader
    v1.SolveStatus = _Status
    v1.FailureCode = _FailureCode
    v1.NetworkPolicy = _NetworkPolicy
    v1.GpuPolicy = _GpuPolicy
    v1.BackendPolicy = _BackendPolicy
    v1.CancellationToken = _CancellationToken
    v1.create_solver_runtime = lambda **kw: FakeRuntime()
    return v1, rec


def test_cancel_before_solve_is_noop(monkeypatch):
    v1, rec = _make_cancel_v1()
    _install_stubs(monkeypatch, v1)
    adapter = ZeSolverAdapter()
    adapter.cancel_active_solve()  # no active token -> no-op, no raise
    adapter.close()
    assert rec.runtime_close_calls == 0


def test_tokens_cleared_after_solve_and_close(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = _make_result(_Status.SOLVED, wcs_header=_WcsHeader())
    _install_stubs(monkeypatch, v1)

    adapter = ZeSolverAdapter()
    adapter.solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    # Token is registered during solve and unregistered in the finally path.
    assert len(adapter._active_tokens) == 0

    adapter.close()
    assert len(adapter._active_tokens) == 0


def test_cancel_during_solve_produces_cancelled(monkeypatch):
    v1, rec = _make_cancel_v1()
    _install_stubs(monkeypatch, v1)
    adapter = ZeSolverAdapter()

    holder = {}

    def run():
        holder["outcome"] = adapter.solve(
            image_fits_path="/tmp/img.fits",
            fits_header={},
            settings={},
            progress_callback=None,
        )

    t = threading.Thread(target=run)
    t.start()
    assert rec.solve_started.wait(timeout=10)

    # The token actually handed to ZeSolver must be the one we cancel.
    assert rec.last_cancellation is not None
    assert rec.last_cancellation.is_cancelled() is False

    adapter.cancel_active_solve()
    assert rec.last_cancellation.is_cancelled() is True

    rec.release_solve.set()
    t.join(timeout=10)
    assert not t.is_alive()

    outcome = holder["outcome"]
    assert outcome.status is SolveStatus.CANCELLED
    assert outcome.failure_code == "cancelled"

    # Cleanup: closing the adapter tears down the session and runtime.
    adapter.close()
    assert rec.session_close_calls == 1
    assert rec.runtime_close_calls == 1


# ---------------------------------------------------------------------------
# 8) Multithread lifecycle (1 runtime, <=N sessions, N*M solves)
# ---------------------------------------------------------------------------


def _make_multithread_v1():
    rec = types.SimpleNamespace(
        create_runtime_lock=threading.Lock(),
        create_runtime_calls=0,
        create_session_events=[],
        solve_events=[],
        runtime_close_events=[],
        session_close_events=[],
    )
    session_counter = {"n": 0}

    class FakeSession:
        def __init__(self, session_id):
            self.session_id = session_id

        def solve(self, request, cancellation=None, progress=None):
            rec.solve_events.append((threading.get_ident(), self.session_id))
            return _make_result(_Status.SOLVED, wcs_header=_WcsHeader())

        def close(self):
            rec.session_close_events.append(1)

    class FakeRuntime:
        def create_session(self):
            session_counter["n"] += 1
            rec.create_session_events.append(session_counter["n"])
            return FakeSession(session_counter["n"])

        def close(self):
            rec.runtime_close_events.append(1)

    def create_solver_runtime(**kwargs):
        with rec.create_runtime_lock:
            rec.create_runtime_calls += 1
        return FakeRuntime()

    v1 = types.ModuleType("zesolver.api.v1")
    v1.SolveHints = lambda **kw: None
    v1.SolveOptions = lambda **kw: None
    v1.SolveRequest = lambda input_path, hints=None, options=None: None
    v1.CanonicalWcsHeader = _WcsHeader
    v1.SolveStatus = _Status
    v1.FailureCode = _FailureCode
    v1.NetworkPolicy = _NetworkPolicy
    v1.GpuPolicy = _GpuPolicy
    v1.BackendPolicy = _BackendPolicy
    v1.CancellationToken = _CancellationToken
    v1.create_solver_runtime = create_solver_runtime
    return v1, rec


def test_multithread_lifecycle_one_runtime_sessions_reused(monkeypatch):
    v1, rec = _make_multithread_v1()
    _install_stubs(monkeypatch, v1)
    adapter = ZeSolverAdapter()

    N_THREADS = 4
    N_IMAGES_PER_THREAD = 25
    barrier = threading.Barrier(N_THREADS)
    errors = []

    def worker():
        try:
            barrier.wait(timeout=10)
            for i in range(N_IMAGES_PER_THREAD):
                outcome = adapter.solve(
                    image_fits_path=f"/tmp/img_{threading.get_ident()}_{i}.fits",
                    fits_header={},
                    settings={},
                    progress_callback=None,
                )
                assert outcome.status is SolveStatus.SOLVED
        except Exception as exc:  # pragma: no cover - only on bug
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert rec.create_runtime_calls == 1
    assert len(rec.solve_events) == N_THREADS * N_IMAGES_PER_THREAD
    # One session per worker thread (<= number of threads), never per image.
    assert 1 <= len(rec.create_session_events) <= N_THREADS

    # Successive images on the same thread reuse the same session object.
    sessions_by_thread = {}
    for tid, sid in rec.solve_events:
        sessions_by_thread.setdefault(tid, set()).add(sid)
    assert len(sessions_by_thread) == N_THREADS
    for sid_set in sessions_by_thread.values():
        assert sid_set == {min(sid_set)}  # exactly one session per thread

    # Closing tears down every tracked session and the single runtime.
    adapter.close()
    assert len(rec.session_close_events) == len(rec.create_session_events)
    assert len(rec.runtime_close_events) == 1


# ---------------------------------------------------------------------------
# 9) Fingerprint double-checked-locking race (worker factory)
# ---------------------------------------------------------------------------


def _load_worker():
    return importlib.import_module("zemosaic.zemosaic_worker")


def test_worker_factory_concurrent_same_fingerprint_single_adapter(monkeypatch):
    worker = _load_worker()

    created = []
    closed = []

    class FakeAdapter:
        def __init__(self, *, resources_path=None, gpu_policy=None, network_policy=None):
            created.append(self)

        def close(self):
            closed.append(self)

    fake_module = types.SimpleNamespace(ZeSolverAdapter=FakeAdapter)
    monkeypatch.setattr(worker, "_zesolver_adapter", fake_module)
    monkeypatch.setattr(worker, "_ZESOLVER_ADAPTER_INSTANCE", None)
    monkeypatch.setattr(worker, "_ZESOLVER_ADAPTER_FINGERPRINT", None)

    settings = {
        "zesolver_resources_path": "/race",
        "zesolver_gpu_policy": "auto",
        "zesolver_backend_policy": "auto",
    }

    N = 16
    barrier = threading.Barrier(N)
    results = []

    def run():
        barrier.wait(timeout=10)
        results.append(worker._get_zesolver_adapter(settings))

    threads = [threading.Thread(target=run) for _ in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # Exactly one adapter despite N threads racing on the same new fingerprint.
    assert len(created) == 1
    assert closed == []
    assert all(r is created[0] for r in results)


def test_worker_factory_rebuilds_once_on_fingerprint_change(monkeypatch):
    worker = _load_worker()

    created = []
    closed = []

    class FakeAdapter:
        def __init__(self, *, resources_path=None, gpu_policy=None, network_policy=None):
            self.gpu_policy = gpu_policy
            created.append(self)

        def close(self):
            closed.append(self)

    fake_module = types.SimpleNamespace(ZeSolverAdapter=FakeAdapter)
    monkeypatch.setattr(worker, "_zesolver_adapter", fake_module)
    monkeypatch.setattr(worker, "_ZESOLVER_ADAPTER_INSTANCE", None)
    monkeypatch.setattr(worker, "_ZESOLVER_ADAPTER_FINGERPRINT", None)

    s1 = {"zesolver_gpu_policy": "auto", "zesolver_resources_path": "/a", "zesolver_backend_policy": "auto"}
    a1 = worker._get_zesolver_adapter(s1)
    assert len(created) == 1

    s2 = {"zesolver_gpu_policy": "disabled", "zesolver_resources_path": "/a", "zesolver_backend_policy": "auto"}
    a2 = worker._get_zesolver_adapter(s2)
    assert a2 is not a1
    assert len(created) == 2
    assert closed == [a1]
    assert a2.gpu_policy == "disabled"


# ---------------------------------------------------------------------------
# 3) Discovery caching (worker level)
# ---------------------------------------------------------------------------


def test_worker_discovery_cached_once_and_invalidated(monkeypatch):
    worker = _load_worker()

    calls = []

    def fake_discover():
        calls.append(1)
        return SolverDiscovery(state=DiscoveryState.AVAILABLE)

    fake_module = types.SimpleNamespace(discover_zesolver=fake_discover)
    monkeypatch.setattr(worker, "_zesolver_adapter", fake_module)
    monkeypatch.setattr(worker, "_ZESOLVER_DISCOVERY_CACHE", None)

    for _ in range(100):
        assert worker._get_zesolver_discovery().state is DiscoveryState.AVAILABLE
    assert len(calls) == 1  # 100 images -> 1 discovery, not 100

    # Explicit invalidation (new run) re-discovers once.
    worker._invalidate_zesolver_discovery_cache()
    assert worker._get_zesolver_discovery().state is DiscoveryState.AVAILABLE
    assert len(calls) == 2


def test_worker_close_invalidates_discovery_cache(monkeypatch):
    worker = _load_worker()

    calls = []

    def fake_discover():
        calls.append(1)
        return SolverDiscovery(state=DiscoveryState.AVAILABLE)

    fake_adapter = types.SimpleNamespace(
        discover_zesolver=fake_discover,
        ZeSolverAdapter=type(
            "FakeAdapter",
            (),
            {
                "__init__": lambda self, **kw: None,
                "close": lambda self: None,
            },
        ),
    )
    monkeypatch.setattr(worker, "_zesolver_adapter", fake_adapter)
    monkeypatch.setattr(worker, "_ZESOLVER_DISCOVERY_CACHE", None)
    monkeypatch.setattr(worker, "_ZESOLVER_ADAPTER_INSTANCE", None)
    monkeypatch.setattr(worker, "_ZESOLVER_ADAPTER_FINGERPRINT", None)

    assert worker._get_zesolver_discovery().state is DiscoveryState.AVAILABLE
    assert len(calls) == 1

    worker._close_zesolver_adapter()  # run-exit hook invalidates the cache
    assert worker._get_zesolver_discovery().state is DiscoveryState.AVAILABLE
    assert len(calls) == 2
