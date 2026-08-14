"""Internal SolverPort boundary for plate solving in ZeMosaic.

This module defines the *internal* boundary that :mod:`zemosaic.zemosaic_worker`
uses to dispatch the Phase 1 WCS solve.  It is deliberately decoupled from any
concrete solver so that:

* the existing ASTAP / Astrometry.net / ANSVR / NONE behaviour is preserved
  through :class:`LegacySolverAdapter` (a thin, faithful wrapper around the
  current worker solve functions), and
* the optional ZeSolver integration (see :mod:`zemosaic.zesolver_adapter`) can
  be added without touching scientific processing semantics.

This module imports only the standard library.  In particular it must never
import ``zesolver`` (or any of its submodules) at import time: ZeSolver is an
optional dependency and its discovery is intentionally lazy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Solver choice identifiers (kept compatible with GUI + config strings).
# ---------------------------------------------------------------------------

SOLVER_CHOICE_ASTAP = "ASTAP"
SOLVER_CHOICE_ASTROMETRY = "ASTROMETRY"
SOLVER_CHOICE_ANSVR = "ANSVR"
SOLVER_CHOICE_NONE = "NONE"
SOLVER_CHOICE_ZESOLVER = "ZESOLVER"

# The legacy choices whose behaviour must be preserved byte-for-byte.
LEGACY_SOLVER_CHOICES = (
    SOLVER_CHOICE_ASTAP,
    SOLVER_CHOICE_ASTROMETRY,
    SOLVER_CHOICE_ANSVR,
    SOLVER_CHOICE_NONE,
)

KNOWN_SOLVER_CHOICES = LEGACY_SOLVER_CHOICES + (SOLVER_CHOICE_ZESOLVER,)


# ---------------------------------------------------------------------------
# Outcome model shared by every adapter.
# ---------------------------------------------------------------------------


class SolveStatus(str, Enum):
    """Outcome status of a single solve attempt (internal, not ZeSolver's)."""

    SOLVED = "solved"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"


@dataclass
class SolverOutcome:
    """Transport-neutral result returned by every :class:`SolverAdapter`.

    ``wcs`` is an Astropy :class:`~astropy.wcs.WCS` (or ``None``).
    ``header``, when not ``None``, is an Astropy :class:`~astropy.io.fits.Header`
    carrying the canonical WCS cards to use/return downstream (used by the
    ZeSolver adapter, which receives cards instead of an in-place header update).
    ``should_write_header_back`` mirrors the legacy "an external solver injected
    WCS keys" flag so the caller keeps the exact same header-write behaviour.
    """

    status: SolveStatus = SolveStatus.UNAVAILABLE
    wcs: Any = None
    header: Any = None
    should_write_header_back: bool = False
    failure_code: str | None = None
    message: str | None = None
    backend_used: str | None = None


# ---------------------------------------------------------------------------
# ZeSolver discovery model (state lives here so the boundary owns the contract).
# ---------------------------------------------------------------------------


class DiscoveryState(str, Enum):
    """Lazy-discovery state for the optional ZeSolver integration."""

    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    INCOMPATIBLE = "incompatible"
    UNHEALTHY = "unhealthy"


@dataclass
class SolverDiscovery:
    """Result of a lazy ZeSolver discovery/health check."""

    state: DiscoveryState
    api_version: str | None = None
    product_version: str | None = None
    message: str | None = None


# ---------------------------------------------------------------------------
# ZeSolver capability negotiation.
#
# ZeSolver exposes two capability signals through its public v1 API:
#   * ``get_api_info().supported_capabilities`` — the *static, declared* set of
#     capability IDs the installed API implementation supports (never changes
#     at runtime, cheap, no I/O); and
#   * ``probe(...).capabilities`` — the *negotiated* per-capability availability
#     (AVAILABLE / UNAVAILABLE / NOT_CHECKED).
#
# The adapter challenges only the IDs ZeSolver actually publishes through its
# public ``supported_capabilities`` metadata (near_solve, blind_solve,
# wcs_write, gpu, cancel).  There is no literal ``"solve"`` ID — the "can this
# API solve at all" requirement is expressed as "at least one solve backend
# (near_solve or blind_solve) is declared".
# ---------------------------------------------------------------------------

# Hard requirement for the v1 adapter path.  ZeSolver's default write policy is
# ``WritePolicy.OVERWRITE_INPUT`` (it writes the solved WCS back into the input
# FITS itself) and the adapter consumes the returned canonical header.  Without
# ``wcs_write`` the adapter cannot consume a solved result, so a ZeSolver that
# does not declare ``wcs_write`` is reported unavailable/incompatible.
REQUIRED_ZESOLVER_CAPABILITIES: tuple[str, ...] = ("wcs_write",)

# At least one of these solve-backend IDs must be declared.  Their *runtime*
# availability (catalog presence) is intentionally NOT a discovery blocker: the
# cheap probe reports them NOT_CHECKED and the real availability is negotiated
# lazily at solve time (ZeSolver returns MISSING_RESOURCE / BACKEND_UNAVAILABLE
# then).  Declared support is all discovery needs to establish.
ZESOLVER_SOLVE_BACKEND_CAPABILITIES: tuple[str, ...] = ("near_solve", "blind_solve")

# Optional capabilities: their absence (or runtime unavailability) never blocks
# the adapter.  ``cancel`` enables cooperative cancellation (best-effort);
# ``gpu`` enables GPU-accelerated solving (policy-dependent).
OPTIONAL_ZESOLVER_CAPABILITIES: tuple[str, ...] = ("cancel", "gpu")


# ---------------------------------------------------------------------------
# Adapter protocol.
# ---------------------------------------------------------------------------


class SolverAdapter(Protocol):
    """Minimal protocol implemented by :class:`LegacySolverAdapter` and
    :class:`zesolver_adapter.ZeSolverAdapter`."""

    name: str

    def solve(self, **kwargs) -> SolverOutcome: ...


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------


def _basename(path: Any) -> str:
    try:
        return os.path.basename(os.path.normpath(str(path)))
    except Exception:  # pragma: no cover - defensive
        return str(path)


# ---------------------------------------------------------------------------
# Legacy adapter: faithful preservation of the pre-port dispatch.
# ---------------------------------------------------------------------------


class LegacySolverAdapter:
    """Preserve the exact ASTAP / Astrometry.net / ANSVR / NONE behaviour.

    This adapter does **not** reimplement any solving logic.  It delegates to
    the callables it was constructed with and reproduces the historical dispatch
    semantics, including:

    * ``ASTROMETRY`` -> ``solve_with_astrometry`` with an ASTAP fallback,
    * ``ANSVR`` -> ``solve_with_ansvr`` with an ASTAP fallback,
    * anything else (``ASTAP``, ``NONE`` and unknown values) -> direct ASTAP
      (this is the pre-port fall-through behaviour and is intentionally kept).
    """

    name = "legacy"

    def __init__(
        self,
        *,
        astrometry_fn=None,
        ansvr_fn=None,
        astap_fn=None,
        astap_paths_valid_fn=None,
        log=None,
    ) -> None:
        self._astrometry_fn = astrometry_fn
        self._ansvr_fn = ansvr_fn
        self._astap_fn = astap_fn
        self._astap_paths_valid_fn = astap_paths_valid_fn or (lambda *a, **k: False)
        self._log = log

    def _emit(self, msg_key: str, lvl: str = "DEBUG", **kwargs) -> None:
        if self._log is None:
            return
        try:
            self._log(msg_key, lvl=lvl, **kwargs)
        except Exception:  # pragma: no cover - logging must never raise
            pass

    def _astap_solve(
        self,
        image_fits_path,
        fits_header,
        progress_callback,
        astap_exe_path,
        astap_data_dir,
        astap_search_radius,
        astap_downsample,
        astap_sensitivity,
        astap_timeout_seconds,
        astap_drizzled_fallback_enabled,
    ):
        if self._astap_fn is None:
            return None
        return self._astap_fn(
            image_fits_path=image_fits_path,
            original_fits_header=fits_header,
            astap_exe_path=astap_exe_path,
            astap_data_dir=astap_data_dir,
            search_radius_deg=astap_search_radius,
            downsample_factor=astap_downsample,
            sensitivity=astap_sensitivity,
            astap_drizzled_fallback_enabled=astap_drizzled_fallback_enabled,
            timeout_sec=astap_timeout_seconds,
            update_original_header_in_place=True,
            progress_callback=progress_callback,
        )

    def solve(
        self,
        *,
        solver_choice: str,
        image_fits_path: str,
        fits_header,
        settings,
        progress_callback,
        astap_exe_path: str,
        astap_data_dir: str,
        astap_search_radius: float,
        astap_downsample: int,
        astap_sensitivity: int,
        astap_timeout_seconds: int,
        astap_drizzled_fallback_enabled: bool,
        **kwargs,
    ) -> SolverOutcome:
        filename = _basename(image_fits_path)

        if solver_choice == SOLVER_CHOICE_ASTROMETRY:
            self._emit("GetWCS: using ASTROMETRY", lvl="DEBUG")
            wcs = None
            if self._astrometry_fn is not None:
                wcs = self._astrometry_fn(
                    image_fits_path, fits_header, settings or {}, progress_callback
                )
            if not wcs and self._astap_paths_valid_fn(astap_exe_path, astap_data_dir):
                self._emit("Astrometry failed; fallback to ASTAP", lvl="INFO")
                self._emit("GetWCS: using ASTAP (fallback)", lvl="DEBUG")
                wcs = self._astap_solve(
                    image_fits_path,
                    fits_header,
                    progress_callback,
                    astap_exe_path,
                    astap_data_dir,
                    astap_search_radius,
                    astap_downsample,
                    astap_sensitivity,
                    astap_timeout_seconds,
                    astap_drizzled_fallback_enabled,
                )
            if wcs:
                self._emit(
                    "getwcs_info_astrometry_solved", lvl="INFO_DETAIL", filename=filename
                )
                return SolverOutcome(
                    status=SolveStatus.SOLVED,
                    wcs=wcs,
                    should_write_header_back=True,
                    backend_used="astrometry",
                )
            return SolverOutcome(
                status=SolveStatus.FAILED, backend_used="astrometry"
            )

        if solver_choice == SOLVER_CHOICE_ANSVR:
            self._emit("GetWCS: using ANSVR", lvl="DEBUG")
            wcs = None
            if self._ansvr_fn is not None:
                wcs = self._ansvr_fn(
                    image_fits_path, fits_header, settings or {}, progress_callback
                )
            if not wcs and self._astap_paths_valid_fn(astap_exe_path, astap_data_dir):
                self._emit("Ansvr failed; fallback to ASTAP", lvl="INFO")
                self._emit("GetWCS: using ASTAP (fallback)", lvl="DEBUG")
                wcs = self._astap_solve(
                    image_fits_path,
                    fits_header,
                    progress_callback,
                    astap_exe_path,
                    astap_data_dir,
                    astap_search_radius,
                    astap_downsample,
                    astap_sensitivity,
                    astap_timeout_seconds,
                    astap_drizzled_fallback_enabled,
                )
            if wcs:
                self._emit(
                    "getwcs_info_astrometry_solved", lvl="INFO_DETAIL", filename=filename
                )
                return SolverOutcome(
                    status=SolveStatus.SOLVED,
                    wcs=wcs,
                    should_write_header_back=True,
                    backend_used="ansvr",
                )
            return SolverOutcome(status=SolveStatus.FAILED, backend_used="ansvr")

        # ASTAP, NONE and any unknown value: preserved pre-port fall-through.
        self._emit("GetWCS: using ASTAP", lvl="DEBUG")
        wcs = self._astap_solve(
            image_fits_path,
            fits_header,
            progress_callback,
            astap_exe_path,
            astap_data_dir,
            astap_search_radius,
            astap_downsample,
            astap_sensitivity,
            astap_timeout_seconds,
            astap_drizzled_fallback_enabled,
        )
        if wcs:
            self._emit("getwcs_info_astap_solved", lvl="INFO_DETAIL", filename=filename)
            return SolverOutcome(
                status=SolveStatus.SOLVED,
                wcs=wcs,
                should_write_header_back=True,
                backend_used="astap",
            )
        self._emit("getwcs_warn_astap_failed", lvl="WARN", filename=filename)
        return SolverOutcome(
            status=SolveStatus.FAILED,
            backend_used="astap",
            message="astap solve failed",
        )
