"""Optional ZeSolver API v1 integration adapter (public-import-only contract).

This module is the *only* place in ZeMosaic that talks to ZeSolver, and it does
so strictly through the public, stable API::

    from zesolver.api import v1          # or
    from zesolver.api.v1 import ...

It never imports ZeSolver internals or private modules (nothing outside the
public ``zesolver.api.v1`` surface), never searches the sibling ZeSolver
checkout, and never mutates the module search path.

ZeSolver is optional: absence, incompatibility or an unhealthy installation must
never break importing ZeMosaic or running the legacy solvers.
"""

from __future__ import annotations

import importlib
import logging
import threading
from pathlib import Path
from typing import Any

from .solver_port import (
    DiscoveryState,
    SOLVER_CHOICE_ZESOLVER,
    SolverDiscovery,
    SolverOutcome,
    SolveStatus,
    _basename,
)

logger = logging.getLogger("zemosaic.zesolver_adapter")

_ZESOLVER_API_MODULE = "zesolver.api.v1"
_ZESOLVER_SUPPORTED_API_MAJOR = 1

# Options that map directly onto public SolveHints fields (all optional).
_HINTS_KEY_MAP = (
    ("zesolver_ra_deg", "ra_deg"),
    ("zesolver_dec_deg", "dec_deg"),
    ("zesolver_radius_deg", "radius_deg"),
    ("zesolver_pixel_scale_arcsec", "pixel_scale_arcsec"),
    ("zesolver_fov_deg", "fov_deg"),
    ("zesolver_focal_length_mm", "focal_length_mm"),
    ("zesolver_pixel_size_um", "pixel_size_um"),
)


def _parse_major(version: Any) -> int | None:
    if version is None:
        return None
    try:
        return int(str(version).split(".")[0])
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def discover_zesolver() -> SolverDiscovery:
    """Lazily discover and health-check the installed ZeSolver public API v1.

    Compatibility is based exclusively on the public ``API_VERSION`` /
    ``API_MAJOR`` (major == 1), never on Git branch or product version.  The
    health check is intentionally cheap: ``probe`` is called without catalog
    scan, without GPU/CuPy import and without network access.
    """
    try:
        v1 = importlib.import_module(_ZESOLVER_API_MODULE)
    except Exception as exc:  # pragma: no cover - exercised via fake modules
        return SolverDiscovery(
            state=DiscoveryState.NOT_INSTALLED,
            message=f"{type(exc).__name__}: {exc}",
        )

    api_version = getattr(v1, "API_VERSION", None)
    api_major = getattr(v1, "API_MAJOR", None)
    if api_major is None:
        api_major = _parse_major(api_version)

    product_version = None
    try:
        info_fn = getattr(v1, "get_api_info", None)
        if info_fn is not None:
            product_version = info_fn().product_version
    except Exception:  # pragma: no cover - defensive
        product_version = None

    if api_major != _ZESOLVER_SUPPORTED_API_MAJOR:
        return SolverDiscovery(
            state=DiscoveryState.INCOMPATIBLE,
            api_version=str(api_version),
            product_version=product_version,
            message=(
                f"unsupported ZeSolver API major version {api_major!r} "
                f"(expected {_ZESOLVER_SUPPORTED_API_MAJOR})"
            ),
        )

    probe_fn = getattr(v1, "probe", None)
    if probe_fn is not None:
        try:
            probe_fn(check_catalogs=False, check_gpu=False)
        except Exception as exc:  # pragma: no cover - exercised via fake modules
            return SolverDiscovery(
                state=DiscoveryState.UNHEALTHY,
                api_version=str(api_version),
                product_version=product_version,
                message=f"probe failed: {type(exc).__name__}: {exc}",
            )

    return SolverDiscovery(
        state=DiscoveryState.AVAILABLE,
        api_version=str(api_version),
        product_version=product_version,
    )


class ZeSolverAdapter:
    """Plate solving through the public ``zesolver.api.v1`` API.

    Lifecycle: one :class:`SolverRuntime` per adapter instance (i.e. per
    batch/process) and one :class:`SolverSession` per worker thread.  The runtime
    is created lazily on first use and shared; sessions are kept in
    ``threading.local`` storage and created on demand.  :meth:`close` tears the
    runtime (and every tracked session) down; the adapter stays usable and the
    next :meth:`solve` transparently recreates a fresh runtime/session.
    """

    name = SOLVER_CHOICE_ZESOLVER

    def __init__(
        self,
        *,
        resources_path: str | Path | None = None,
        gpu_policy: str | None = None,
        network_policy: str | None = None,
    ) -> None:
        self._resources_path = resources_path
        self._gpu_policy = gpu_policy
        self._network_policy = network_policy
        self._runtime = None
        self._runtime_lock = threading.Lock()
        self._sessions = threading.local()
        self._session_registry = set()
        self._session_registry_lock = threading.Lock()

    # -- public ------------------------------------------------------------

    def solve(
        self,
        *,
        image_fits_path: str,
        fits_header,
        settings,
        progress_callback,
        log=None,
        **kwargs,
    ) -> SolverOutcome:
        """Solve one file and return a :class:`SolverOutcome`.

        Expected operational failures are returned as a ``FAILED`` outcome;
        unexpected public API errors are logged and returned as a ``FAILED``
        outcome (failing that single file, never crashing the import path).
        """
        filename = _basename(image_fits_path)
        try:
            v1 = self._import_api()
            request = self._build_request(v1, image_fits_path, settings or {})
            cancellation = v1.CancellationToken()
            result = self._session().solve(request, cancellation=cancellation)
            return self._convert_result(v1, result, fits_header, filename)
        except Exception as exc:  # noqa: BLE001 - fail this file, never the import
            if log is not None:
                try:
                    log(
                        "getwcs_warn_zesolver_failed",
                        lvl="WARN",
                        filename=filename,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                except Exception:  # pragma: no cover - logging must never raise
                    pass
            log_message = f"{type(exc).__name__}: {exc}"
            logger.error("ZeSolver solve failed for '%s': %s", filename, log_message)
            return SolverOutcome(
                status=SolveStatus.FAILED,
                failure_code="unexpected_error",
                message=log_message,
                backend_used=self.name,
            )

    def close(self) -> None:
        """Close the public runtime and every tracked thread-local session.

        Idempotent and safe to call from any thread: closing twice (or closing
        an adapter whose runtime was never created) is a no-op.  After close the
        adapter remains usable — the next :meth:`solve` lazily creates a new
        runtime and a new session.
        """
        # Invalidate the current thread's cached session first so a later
        # solve() on this thread never reuses a closed session.
        try:
            self._sessions.session = None
        except Exception:  # pragma: no cover - defensive
            pass

        sessions_to_close = []
        with self._session_registry_lock:
            sessions_to_close = list(self._session_registry)
            self._session_registry.clear()

        for session in sessions_to_close:
            try:
                close_session = getattr(session, "close", None)
                if callable(close_session):
                    close_session()
            except Exception:  # pragma: no cover - close must never raise
                logger.debug("ZeSolver session close failed", exc_info=True)

        runtime = None
        with self._runtime_lock:
            runtime = self._runtime
            self._runtime = None
        if runtime is not None:
            try:
                close_runtime = getattr(runtime, "close", None)
                if callable(close_runtime):
                    close_runtime()
            except Exception:  # pragma: no cover - close must never raise
                logger.debug("ZeSolver runtime close failed", exc_info=True)

    # -- internal ----------------------------------------------------------

    def _import_api(self):
        return importlib.import_module(_ZESOLVER_API_MODULE)

    def _ensure_runtime(self):
        if self._runtime is not None:
            return self._runtime
        with self._runtime_lock:
            if self._runtime is None:
                v1 = self._import_api()
                create = v1.create_solver_runtime
                kwargs: dict[str, Any] = {}
                if self._resources_path is not None:
                    kwargs["resources_path"] = Path(self._resources_path)
                if self._gpu_policy is not None:
                    kwargs["gpu_policy"] = self._resolve_gpu_policy(self._gpu_policy, v1)
                if self._network_policy is not None:
                    kwargs["network_policy"] = self._resolve_network_policy(
                        self._network_policy, v1
                    )
                self._runtime = create(**kwargs)
        return self._runtime

    def _session(self):
        session = getattr(self._sessions, "session", None)
        if session is None:
            session = self._ensure_runtime().create_session()
            self._sessions.session = session
            with self._session_registry_lock:
                self._session_registry.add(session)
        return session

    def _build_request(self, v1, image_fits_path: str, settings: dict):
        hints_kwargs: dict[str, Any] = {}
        for settings_key, hint_field in _HINTS_KEY_MAP:
            value = settings.get(settings_key)
            if value is not None:
                try:
                    hints_kwargs[hint_field] = float(value)
                except (TypeError, ValueError):
                    continue
        hints = v1.SolveHints(**hints_kwargs)

        options_kwargs: dict[str, Any] = {
            # Network is always disabled (v1 default and ZeMosaic policy).
            "network_policy": v1.NetworkPolicy.DISABLED,
        }
        backend_policy = settings.get("zesolver_backend_policy")
        if backend_policy:
            try:
                options_kwargs["backend_policy"] = self._resolve_backend_policy(
                    backend_policy, v1
                )
            except Exception:  # pragma: no cover - defensive
                pass
        timeout_s = settings.get("zesolver_timeout_s")
        if timeout_s is not None:
            try:
                options_kwargs["timeout_s"] = float(timeout_s)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass

        options = v1.SolveOptions(**options_kwargs)
        return v1.SolveRequest(input_path=Path(image_fits_path), hints=hints, options=options)

    def _convert_result(self, v1, result, fits_header, filename) -> SolverOutcome:
        if result.status == v1.SolveStatus.SOLVED and result.wcs_header is not None:
            try:
                header = result.wcs_header.to_fits_header()
                wcs = result.wcs_header.to_astropy_wcs()
            except Exception as exc:
                return SolverOutcome(
                    status=SolveStatus.FAILED,
                    failure_code="wcs_invalid",
                    message=f"{type(exc).__name__}: {exc}",
                    backend_used=self.name,
                )
            merged_header = self._merge_header(fits_header, header)
            return SolverOutcome(
                status=SolveStatus.SOLVED,
                wcs=wcs,
                header=merged_header,
                should_write_header_back=True,
                backend_used=self.name,
            )

        if result.status == v1.SolveStatus.FAILED:
            failure_code = getattr(result.failure_code, "value", result.failure_code)
            return SolverOutcome(
                status=SolveStatus.FAILED,
                failure_code=failure_code,
                message=result.message,
                backend_used=self.name,
            )

        if result.status == v1.SolveStatus.CANCELLED:
            return SolverOutcome(
                status=SolveStatus.FAILED,
                failure_code="cancelled",
                message=result.message,
                backend_used=self.name,
            )

        # SKIPPED_EXISTING_WCS (or any other non-solved status): no new WCS.
        return SolverOutcome(
            status=SolveStatus.SKIPPED,
            message=result.message,
            backend_used=self.name,
        )

    def _merge_header(self, fits_header, canonical_header):
        """Merge canonical WCS cards into the original header (in-memory)."""
        if fits_header is None:
            return canonical_header
        try:
            merged = fits_header.copy()
            merged.update(canonical_header)
            return merged
        except Exception:  # pragma: no cover - defensive
            return canonical_header

    @staticmethod
    def _resolve_gpu_policy(value, v1):
        text = str(value).strip().lower()
        mapping = {
            "auto": v1.GpuPolicy.AUTO,
            "disabled": v1.GpuPolicy.DISABLED,
            "required": v1.GpuPolicy.REQUIRED,
        }
        return mapping[text]

    @staticmethod
    def _resolve_backend_policy(value, v1):
        text = str(value).strip().lower()
        mapping = {
            "auto": v1.BackendPolicy.AUTO,
            "near_only": v1.BackendPolicy.NEAR_ONLY,
            "blind_only": v1.BackendPolicy.BLIND_ONLY,
        }
        return mapping[text]

    @staticmethod
    def _resolve_network_policy(value, v1):
        text = str(value).strip().lower()
        mapping = {
            "disabled": v1.NetworkPolicy.DISABLED,
            "allowed": v1.NetworkPolicy.ALLOWED,
        }
        return mapping[text]
