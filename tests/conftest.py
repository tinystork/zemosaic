"""Pytest bootstrap: make the src-layout ``zemosaic`` package importable.

Also hosts SESSION-SCOPED fixtures that build/load the reusable gated corpora
(the M16 aligned cache / M106 manifest) ONCE per session, so the gated tests do
not each re-read the manifest + rebuild the canvas per file. See TESTING.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic import eta_service as zeta  # noqa: E402

M106_LIGHTS = Path("/home/tristan/M106/lights")
M16_LIGHTS = Path("/home/tristan/M16/quick")


@pytest.fixture(scope="session", autouse=True)
def _isolate_eta_history(tmp_path_factory):
    """Force EVERY test's ETA history into pytest temp storage.

    ZM-ETA-INTEGRITY-R32: the real defect was synthetic/direct engine tests
    appending to the user's ``~/.zemosaic_eta_history.json`` (95 of 120 ZeGrid
    records were 4-frame CPU junk). This session-scoped autouse guard sets the
    single documented override seam (``ZEMOSAIC_ETA_HISTORY_PATH``) BEFORE any
    test runs, so both the eta service and the legacy worker history code write
    to a per-session temp file. It never mutates ``HOME`` and never disables
    history (production paths are untouched when pytest is not running).

    Intentional history unit tests can still pass an EXPLICIT temp path to
    ``load_eta_history`` / ``append_*``; the explicit argument always wins.
    """
    target = tmp_path_factory.mktemp("eta_history") / ".zemosaic_eta_history.json"
    previous = os.environ.get(zeta.ETA_HISTORY_ENV_VAR)
    os.environ[zeta.ETA_HISTORY_ENV_VAR] = str(target)
    try:
        yield target
    finally:
        if previous is None:
            os.environ.pop(zeta.ETA_HISTORY_ENV_VAR, None)
        else:
            os.environ[zeta.ETA_HISTORY_ENV_VAR] = previous


@pytest.fixture(autouse=True)
def _isolate_zegrid_storage_profile():
    """Undo cross-test leakage of the module-level storage profile.

    A few R52 witnesses assign ``zemosaic_zegrid_mode._STACK_STORAGE_PROFILE``
    DIRECTLY (not through ``monkeypatch``). That value then survived into an
    unrelated test and made R52 Lot B/C leak failures depend on collection
    order. Snapshot/restore here so every test starts from the value it found,
    without weakening any expectation.
    """
    try:
        from zemosaic import zemosaic_zegrid_mode as _zz
    except Exception:  # pragma: no cover - import-time platform guards
        yield
        return
    saved = getattr(_zz, "_STACK_STORAGE_PROFILE", None)
    try:
        yield
    finally:
        if saved is not None:
            _zz._STACK_STORAGE_PROFILE = saved


@pytest.fixture(scope="session")
def m106_corpus():
    """M106 manifest + canvas, read ONCE per session.

    Shared by the gated tests (R1/R2/R3/R4/R7/R14/R15) that each used to read
    the 89-frame M106 manifest and rebuild the canvas per file. Skipped cleanly
    when the corpus is absent (CI / clean checkout).
    """
    if not M106_LIGHTS.is_dir():
        pytest.skip("M106 lights directory not present")
    frames, rejected = zg.read_manifest(M106_LIGHTS)
    canvas = zg.build_canvas(frames)
    return frames, rejected, canvas


@pytest.fixture(scope="session")
def m16_corpus():
    """M16 manifest + canvas, read ONCE per session (R12 instrumentation tests).

    Skipped cleanly when the corpus is absent (CI / clean checkout).
    """
    if not M16_LIGHTS.is_dir():
        pytest.skip("M16 lights directory not present")
    descs, _ = zg.read_manifest(M16_LIGHTS)
    canvas = zg.build_canvas(descs)
    return descs, canvas
