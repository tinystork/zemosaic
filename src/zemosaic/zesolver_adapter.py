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
    REQUIRED_ZESOLVER_CAPABILITIES,
    SOLVER_CHOICE_ZESOLVER,
    SolverDiscovery,
    SolverOutcome,
    SolveStatus,
    ZESOLVER_SOLVE_BACKEND_CAPABILITIES,
    _basename,
)

logger = logging.getLogger("zemosaic.zesolver_adapter")

_ZESOLVER_API_MODULE = "zesolver.api.v1"
_ZESOLVER_SUPPORTED_API_MAJOR = 1

# Progress phase -> ZeMosaic progress message key (real phases only, no invented
# percentage and no ETA).  The value is emitted verbatim as a DEBUG_DETAIL
# progress line through the ZeMosaic progress callback.
_PROGRESS_PHASE_KEYS = {
    "preparing": "GetWCS: ZESOLVER preparing",
    "solving": "GetWCS: ZESOLVER solving",
    "writing": "GetWCS: ZESOLVER writing WCS",
    "finalizing": "GetWCS: ZESOLVER finalizing",
}

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


def _is_zesolver_module_absent(exc: BaseException) -> bool:
    """Return True when ``exc`` means the ZeSolver public module itself is absent.

    A :class:`ModuleNotFoundError` whose ``name`` is the public ``zesolver`` /
    ``zesolver.api`` / ``zesolver.api.v1`` chain means "not installed".  A
    ``ModuleNotFoundError`` naming any *other* module means the public module was
    found but one of its internal imports failed, i.e. "installed but broken".
    """
    name = getattr(exc, "name", None)
    if not isinstance(name, str) or not name:
        return False
    return name == "zesolver" or name.startswith("zesolver.api")


def discover_zesolver() -> SolverDiscovery:
    """Lazily discover and health-check the installed ZeSolver public API v1.

    Compatibility is based exclusively on the public ``API_VERSION`` /
    ``API_MAJOR`` (major == 1) plus the declared/negotiated capabilities, never
    on Git branch or product version.  The health check is intentionally cheap:
    ``probe`` is called without catalog scan, without GPU/CuPy import and
    without network access.

    The state distinguishes "not installed" from "installed but broken": a
    genuinely absent public module reports :data:`DiscoveryState.NOT_INSTALLED`,
    while a present-but-failing import/probe reports
    :data:`DiscoveryState.UNHEALTHY` (never a corrupted ZeSolver masquerading as
    "not installed").
    """
    try:
        v1 = importlib.import_module(_ZESOLVER_API_MODULE)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via fake modules
        if _is_zesolver_module_absent(exc):
            return SolverDiscovery(
                state=DiscoveryState.NOT_INSTALLED,
                message=f"{type(exc).__name__}: {exc}",
            )
        # Public module found but one of its internal imports failed.
        return SolverDiscovery(
            state=DiscoveryState.UNHEALTHY,
            message=f"import failed: {type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # pragma: no cover - exercised via fake modules
        # Found but raised a non-import error while importing -> broken install.
        return SolverDiscovery(
            state=DiscoveryState.UNHEALTHY,
            message=f"import failed: {type(exc).__name__}: {exc}",
        )

    api_version = getattr(v1, "API_VERSION", None)
    api_major = getattr(v1, "API_MAJOR", None)
    if api_major is None:
        api_major = _parse_major(api_version)

    product_version = None
    declared_capabilities: tuple[str, ...] = ()
    try:
        info_fn = getattr(v1, "get_api_info", None)
        if info_fn is not None:
            info = info_fn()
            product_version = getattr(info, "product_version", None)
            declared = getattr(info, "supported_capabilities", None)
            if isinstance(declared, (tuple, list)):
                declared_capabilities = tuple(str(c) for c in declared)
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

    # Capability challenge against the *declared* static set.  A required
    # capability that is missing makes the backend unusable on the v1 path.
    declared_set = set(declared_capabilities)
    missing_required = [
        cap for cap in REQUIRED_ZESOLVER_CAPABILITIES if cap not in declared_set
    ]
    if missing_required:
        return SolverDiscovery(
            state=DiscoveryState.INCOMPATIBLE,
            api_version=str(api_version),
            product_version=product_version,
            message=(
                "ZeSolver is missing required capability(ies) "
                f"{', '.join(missing_required)} (declared: "
                f"{', '.join(declared_capabilities) or 'none'})"
            ),
        )

    # "solve" challenge: at least one solve backend must be declared.
    if not (declared_set & set(ZESOLVER_SOLVE_BACKEND_CAPABILITIES)):
        return SolverDiscovery(
            state=DiscoveryState.INCOMPATIBLE,
            api_version=str(api_version),
            product_version=product_version,
            message=(
                "ZeSolver declares no solve backend (expected at least one of "
                f"{', '.join(ZESOLVER_SOLVE_BACKEND_CAPABILITIES)})"
            ),
        )

    # Negotiated availability: required capabilities must not be reported
    # UNAVAILABLE (NOT_CHECKED is tolerated — see solver_port constants).
    probe_fn = getattr(v1, "probe", None)
    if probe_fn is not None:
        try:
            probe_result = probe_fn(check_catalogs=False, check_gpu=False)
        except Exception as exc:  # pragma: no cover - exercised via fake modules
            return SolverDiscovery(
                state=DiscoveryState.UNHEALTHY,
                api_version=str(api_version),
                product_version=product_version,
                message=f"probe failed: {type(exc).__name__}: {exc}",
            )
        negotiated = {}
        for cap_state in getattr(probe_result, "capabilities", ()) or ():
            cap_id = getattr(cap_state, "id", None)
            availability = getattr(cap_state, "availability", None)
            if cap_id is not None:
                negotiated[str(cap_id)] = getattr(availability, "value", availability)
        for cap in REQUIRED_ZESOLVER_CAPABILITIES:
            if negotiated.get(cap) == "unavailable":
                return SolverDiscovery(
                    state=DiscoveryState.UNHEALTHY,
                    api_version=str(api_version),
                    product_version=product_version,
                    message=(
                        f"required capability {cap!r} reported unavailable"
                    ),
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
        self._active_tokens = set()
        self._active_tokens_lock = threading.Lock()

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
            progress = self._make_progress_forwarder(progress_callback)
            self._register_active_token(cancellation)
            try:
                result = self._session().solve(
                    request, cancellation=cancellation, progress=progress
                )
            finally:
                self._unregister_active_token(cancellation)
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

        # Cancel and clear any in-flight tokens (cleanup must cover cancellation
        # and run-abort paths, not just the happy path).
        self.cancel_active_solve()
        with self._active_tokens_lock:
            self._active_tokens.clear()

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

    def cancel_active_solve(self) -> None:
        """Cooperatively cancel any in-flight ZeSolver solve(s).

        Thread-safe and idempotent: it cancels every active token tracked by
        this adapter (one per worker thread with an in-flight solve) and never
        raises.  The process-level cleanup (run-exit close / subprocess kill)
        remains the safety net; this is the first, cooperative cancellation
        level exposed to the worker.
        """
        with self._active_tokens_lock:
            tokens = list(self._active_tokens)
        for token in tokens:
            try:
                cancel = getattr(token, "cancel", None)
                if callable(cancel):
                    cancel()
            except Exception:  # pragma: no cover - cancel must never raise
                logger.debug("ZeSolver token cancel failed", exc_info=True)

    # -- internal ----------------------------------------------------------

    def _import_api(self):
        return importlib.import_module(_ZESOLVER_API_MODULE)

    def _register_active_token(self, token) -> None:
        with self._active_tokens_lock:
            self._active_tokens.add(token)

    def _unregister_active_token(self, token) -> None:
        with self._active_tokens_lock:
            self._active_tokens.discard(token)

    def _make_progress_forwarder(self, progress_callback):
        """Build a ZeSolver progress callback mapping real phases to ZeMosaic.

        ZeSolver calls ``progress(ProgressEvent(phase=..., message=...))``.  We
        forward the *real* phase (PREPARING / SOLVING / WRITING / FINALIZING) to
        the ZeMosaic progress callback using its ``(key, prog, lvl, **kwargs)``
        shape — no invented percentage (``prog=None``) and no ETA.
        """
        if progress_callback is None:
            return None

        def forward(event) -> None:
            phase = getattr(event, "phase", None)
            phase_value = getattr(phase, "value", None)
            if phase_value is None:
                phase_value = str(phase)
            message = getattr(event, "message", None)
            key = _PROGRESS_PHASE_KEYS.get(
                phase_value, "GetWCS: ZESOLVER progress"
            )
            kwargs = {"phase": phase_value}
            if message:
                kwargs["detail"] = str(message)
            try:
                progress_callback(key, None, "DEBUG_DETAIL", **kwargs)
            except Exception:  # pragma: no cover - progress must never raise
                pass

        return forward

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
        # ZeMosaic decides whether to enter the solve (worker's
        # ``force_resolve_existing_wcs``, settings key of the same name).  Map
        # that decision onto ZeSolver's ``overwrite_existing_wcs`` so a forced
        # resolve actually re-solves instead of returning SKIPPED_EXISTING_WCS.
        # We never add a skip here: the worker's existing validation is the only
        # place that decides to skip.
        options_kwargs["overwrite_existing_wcs"] = bool(
            settings.get("force_resolve_existing_wcs")
        )
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
                # ZeSolver (WritePolicy.OVERWRITE_INPUT) already wrote the WCS
                # into the FITS; ZeMosaic must not write it a second time.
                should_write_header_back=False,
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
                status=SolveStatus.CANCELLED,
                failure_code="cancelled",
                message=result.message,
                backend_used=self.name,
            )

        # SKIPPED_EXISTING_WCS means ZeSolver found an existing valid WCS it did
        # not overwrite.  The worker only reaches the adapter after deciding to
        # resolve (absent/invalid/forced WCS), so a skip here would be a silent
        # "unresolved" for a file ZeMosaic already rejected.  Surface it as an
        # explicit failure instead of a false SKIPPED.
        if result.status == v1.SolveStatus.SKIPPED_EXISTING_WCS:
            return SolverOutcome(
                status=SolveStatus.FAILED,
                failure_code="existing_wcs_not_overwritten",
                message=(
                    result.message
                    or "existing valid WCS present and overwrite not requested"
                ),
                backend_used=self.name,
            )

        # Any other unrecognised non-solved status: fail explicitly, never skip.
        return SolverOutcome(
            status=SolveStatus.FAILED,
            failure_code="unexpected_status",
            message=(
                f"unexpected solve status "
                f"{getattr(result.status, 'value', result.status)}"
            ),
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
        # API 1.0 is strictly local-only: ``NetworkPolicy`` exposes no member
        # other than ``DISABLED``.  There is no "allowed" policy in the public
        # v1 contract, so any value other than "disabled" is a caller error.
        if text != "disabled":
            raise ValueError(
                f"network_policy must be 'disabled' (API v1 is local-only), got {value!r}"
            )
        return v1.NetworkPolicy.DISABLED
