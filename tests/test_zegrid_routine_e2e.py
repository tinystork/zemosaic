"""ROUTINE (fast-tier) end-to-end gated check on the small deterministic corpus.

This is the everyday loop-gate version of the full M16/M106 end-to-end runs: it
exercises the complete production ``run_zegrid_mode`` pipeline (manifest ->
canvas -> layout -> per-cell aligned cache -> streaming stack -> assembly ->
FITS outputs) on a tiny, bit-deterministic 12-frame corpus generated on disk by
``tools/zegrid_routine/make_routine_corpus.py``.

The heavy real-corpus end-to-end tests stay available (marked ``slow``) for the
human/release gate. This routine check needs NO local corpora, so it runs in CI
and the implementer's loop gate.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from zemosaic import zemosaic_zegrid_mode as zegrid

TOOL = (
    Path(__file__).resolve().parents[1]
    / "tools" / "zegrid_routine" / "make_routine_corpus.py"
)
N_FRAMES = 12


@pytest.fixture(scope="module")
def routine_corpus(tmp_path_factory):
    """Generate the deterministic 12-frame corpus ONCE per module (on disk)."""
    out = tmp_path_factory.mktemp("routine_corpus")
    proc = subprocess.run(
        [sys.executable, str(TOOL), str(out)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return out


def test_routine_end_to_end_full_pipeline(tmp_path, routine_corpus):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(routine_corpus.glob("*.fits")):
        (input_dir / p.name).write_bytes(p.read_bytes())
    (input_dir / "stack_plan.csv").write_text(
        routine_corpus.joinpath("stack_plan.csv").read_text(), encoding="utf-8"
    )

    out = tmp_path / "out"
    zegrid.run_zegrid_mode(str(input_dir), str(out))

    sci_path = out / "mosaic_grid.fits"
    cov_path = out / "mosaic_grid_coverage.fits"
    manifest_path = out / "zegrid_manifest.json"
    assert sci_path.exists(), "mosaic_grid.fits not produced"
    assert cov_path.exists(), "coverage FITS not produced"
    assert manifest_path.exists(), "manifest not produced"

    with fits.open(cov_path) as hdul:
        cov = hdul[0].data
    covered = np.asarray(cov) > 0
    assert int(np.count_nonzero(covered)) > 0, "no coverage produced"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["engine"] == "zegrid"
    assert manifest["n_frames"] == N_FRAMES
    assert manifest["outputs"]["science"] == "mosaic_grid.fits"
    assert manifest["complete_cells"], "no complete cells in the assembled canvas"
    assert manifest["coverage_pixels"] == int(np.count_nonzero(covered))
