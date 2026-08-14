"""Focused tests for the internal SolverPort boundary (legacy preservation)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import solver_port  # noqa: E402
from zemosaic.solver_port import (  # noqa: E402
    DiscoveryState,
    KNOWN_SOLVER_CHOICES,
    LegacySolverAdapter,
    SOLVER_CHOICE_ANSVR,
    SOLVER_CHOICE_ASTAP,
    SOLVER_CHOICE_ASTROMETRY,
    SOLVER_CHOICE_NONE,
    SOLVER_CHOICE_ZESOLVER,
    SolveStatus,
    SolverOutcome,
)


# ---------------------------------------------------------------------------
# Boundary / choice model
# ---------------------------------------------------------------------------


def test_known_choices_include_zesolver_and_legacy():
    assert SOLVER_CHOICE_ZESOLVER == "ZESOLVER"
    for choice in (
        SOLVER_CHOICE_ASTAP,
        SOLVER_CHOICE_ASTROMETRY,
        SOLVER_CHOICE_ANSVR,
        SOLVER_CHOICE_NONE,
        SOLVER_CHOICE_ZESOLVER,
    ):
        assert choice in KNOWN_SOLVER_CHOICES


def test_importing_solver_port_does_not_import_zesolver():
    # The boundary must never pull in the optional ZeSolver dependency.
    assert "zesolver" not in sys.modules
    assert not any(k.startswith("zesolver") for k in sys.modules)


def test_solver_outcome_defaults():
    outcome = SolverOutcome()
    assert outcome.status is SolveStatus.UNAVAILABLE
    assert outcome.wcs is None
    assert outcome.should_write_header_back is False


# ---------------------------------------------------------------------------
# LegacySolverAdapter: faithful preservation of the pre-port dispatch.
# ---------------------------------------------------------------------------


def _adapter(astrometry_fn=None, ansvr_fn=None, astap_fn=None, paths_valid=None, log=None):
    return LegacySolverAdapter(
        astrometry_fn=astrometry_fn,
        ansvr_fn=ansvr_fn,
        astap_fn=astap_fn,
        astap_paths_valid_fn=paths_valid,
        log=log,
    )


def _solve_kwargs(solver_choice="ASTAP"):
    return dict(
        solver_choice=solver_choice,
        image_fits_path="/tmp/raw/image_001.fits",
        fits_header={"NAXIS1": 1080, "NAXIS2": 1920},
        settings={},
        progress_callback=None,
        astap_exe_path="/opt/astap/astap",
        astap_data_dir="/opt/astap",
        astap_search_radius=3.0,
        astap_downsample=2,
        astap_sensitivity=100,
        astap_timeout_seconds=60,
        astap_drizzled_fallback_enabled=False,
    )


def test_legacy_astap_success():
    sentinel = object()
    calls = {}

    def astap_fn(**kwargs):
        calls.update(kwargs)
        return sentinel

    outcome = _adapter(astap_fn=astap_fn).solve(**_solve_kwargs("ASTAP"))
    assert outcome.status is SolveStatus.SOLVED
    assert outcome.wcs is sentinel
    assert outcome.should_write_header_back is True
    assert outcome.backend_used == "astap"
    # Exact ASTAP call contract is preserved.
    assert calls["update_original_header_in_place"] is True
    assert calls["timeout_sec"] == 60
    assert calls["astap_exe_path"] == "/opt/astap/astap"
    assert calls["astap_data_dir"] == "/opt/astap"


def test_legacy_astap_failure():
    def astap_fn(**kwargs):
        return None

    outcome = _adapter(astap_fn=astap_fn).solve(**_solve_kwargs("ASTAP"))
    assert outcome.status is SolveStatus.FAILED
    assert outcome.wcs is None
    assert outcome.should_write_header_back is False
    assert outcome.backend_used == "astap"


def test_legacy_astrometry_fallback_to_astap():
    sentinel = object()
    astrometry_called = []
    astap_called = []

    def astrometry_fn(image_fits_path, fits_header, settings, progress_callback):
        astrometry_called.append(image_fits_path)
        return None

    def astap_fn(**kwargs):
        astap_called.append(kwargs["image_fits_path"])
        return sentinel

    outcome = _adapter(
        astrometry_fn=astrometry_fn,
        astap_fn=astap_fn,
        paths_valid=lambda exe, data: True,
    ).solve(**_solve_kwargs("ASTROMETRY"))

    assert astrometry_called == ["/tmp/raw/image_001.fits"]
    assert astap_called == ["/tmp/raw/image_001.fits"]
    assert outcome.status is SolveStatus.SOLVED
    assert outcome.wcs is sentinel
    assert outcome.backend_used == "astrometry"


def test_legacy_astrometry_no_fallback_when_paths_invalid():
    def astrometry_fn(image_fits_path, fits_header, settings, progress_callback):
        return None

    def astap_fn(**kwargs):
        raise AssertionError("ASTAP must not be called")

    outcome = _adapter(
        astrometry_fn=astrometry_fn,
        astap_fn=astap_fn,
        paths_valid=lambda exe, data: False,
    ).solve(**_solve_kwargs("ASTROMETRY"))
    assert outcome.status is SolveStatus.FAILED
    assert outcome.backend_used == "astrometry"


def test_legacy_ansvr_fallback_to_astap():
    sentinel = object()

    def ansvr_fn(image_fits_path, fits_header, settings, progress_callback):
        return None

    def astap_fn(**kwargs):
        return sentinel

    outcome = _adapter(
        ansvr_fn=ansvr_fn,
        astap_fn=astap_fn,
        paths_valid=lambda exe, data: True,
    ).solve(**_solve_kwargs("ANSVR"))
    assert outcome.status is SolveStatus.SOLVED
    assert outcome.backend_used == "ansvr"


def test_legacy_none_falls_through_to_astap():
    # Preserved pre-port behaviour: NONE (and any unknown value) routes to ASTAP.
    sentinel = object()
    astap_called = []

    def astap_fn(**kwargs):
        astap_called.append(True)
        return sentinel

    outcome = _adapter(astap_fn=astap_fn).solve(**_solve_kwargs("NONE"))
    assert astap_called == [True]
    assert outcome.status is SolveStatus.SOLVED
    assert outcome.backend_used == "astap"


def test_legacy_unknown_choice_falls_through_to_astap():
    astap_called = []

    def astap_fn(**kwargs):
        astap_called.append(True)
        return object()

    outcome = _adapter(astap_fn=astap_fn).solve(**_solve_kwargs("UNKNOWN"))
    assert astap_called == [True]
    assert outcome.backend_used == "astap"


def test_legacy_emits_expected_progress_messages():
    messages = []

    def log(msg_key, lvl="DEBUG", **kwargs):
        messages.append((msg_key, lvl))

    def astrometry_fn(*a, **k):
        return None

    _adapter(
        astrometry_fn=astrometry_fn,
        astap_fn=lambda **k: object(),
        paths_valid=lambda exe, data: True,
        log=log,
    ).solve(**_solve_kwargs("ASTROMETRY"))

    keys = [m[0] for m in messages]
    assert "GetWCS: using ASTROMETRY" in keys
    assert "Astrometry failed; fallback to ASTAP" in keys
    assert "GetWCS: using ASTAP (fallback)" in keys
    assert "getwcs_info_astrometry_solved" in keys


def test_discovery_state_values():
    assert DiscoveryState.AVAILABLE.value == "available"
    assert DiscoveryState.NOT_INSTALLED.value == "not_installed"
    assert DiscoveryState.INCOMPATIBLE.value == "incompatible"
    assert DiscoveryState.UNHEALTHY.value == "unhealthy"
