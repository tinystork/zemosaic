"""Pytest bootstrap: make the src-layout ``zemosaic`` package importable.

Also hosts SESSION-SCOPED fixtures that build/load the reusable gated corpora
(the M16 aligned cache / M106 manifest) ONCE per session, so the gated tests do
not each re-read the manifest + rebuild the canvas per file. See TESTING.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import geometry as zg  # noqa: E402

M106_LIGHTS = Path("/home/tristan/M106/lights")
M16_LIGHTS = Path("/home/tristan/M16/quick")


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
