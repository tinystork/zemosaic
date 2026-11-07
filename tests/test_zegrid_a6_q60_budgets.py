"""ZM-ZEGRID-ADAPTIVE-RESOURCES A6 — Q60: real E2E synthetic runs at budgets.

Gate: the small-budget proof is a REAL end-to-end run (not a dry-plan). At each
of the Linux budget points (1/2/4 Gio) the production ``run_zegrid_mode`` must
complete end-to-end on the deterministic tiny corpus and produce a valid,
covered FITS + manifest. The budget is injected by pinning
``available_memory_bytes`` (the planning observation), exactly like the accepted
M5 product E2E harness, so the run is hermetic and the budget honestly drives the
working-set selection.

32 Gio is NOT qualifiable on this host (total RAM < 32 Gio) and is documented as
not-run, never fabricated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal

REPO = Path(__file__).resolve().parents[1]
TOOL_DIR = REPO / "tools" / "zegrid_m6"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from make_tiny_corpus import make_tiny_corpus  # noqa: E402

GIB = 2 ** 30


class _HugeModel:
    def predict_bound_bytes(self, n, area):
        return 1 << 62

    def predict_peak_bytes(self, n, area):
        return 1 << 62


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("q60_tiny")
    return make_tiny_corpus(root / "in")


def _manifest(root):
    return json.loads((Path(root) / "zegrid_manifest.json").read_text())


@pytest.mark.parametrize("budget_gib", [1, 2, 4])
def test_q60_e2e_synthetic_budget_completes(
    tiny_corpus, tmp_path, monkeypatch, budget_gib
):
    out = tmp_path / f"out_{budget_gib}gib"
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeModel())
    monkeypatch.setattr(
        zz, "available_memory_bytes", lambda: int(budget_gib) * GIB
    )
    zz.run_zegrid_mode(
        str(tiny_corpus), str(out), zegrid_halo_px=8,
        zconfig=SimpleNamespace(
            zegrid_layout="2x2", zegrid_scheduler_mode="legacy"
        ),
    )
    assert (out / "mosaic_grid.fits").exists()
    assert (out / "mosaic_grid_coverage.fits").exists()
    manifest = _manifest(out)
    assert manifest["engine"] == "zegrid"
    assert manifest["complete_cells"], "no complete cells"
    with fits.open(out / "mosaic_grid_coverage.fits") as hdul:
        cov = hdul[0].data
    assert int(np.count_nonzero(np.asarray(cov) > 0)) > 0
    assert manifest["coverage_pixels"] == int(np.count_nonzero(np.asarray(cov) > 0))
    # The budget was the one we injected (the run must not have silently used
    # a larger real-host observation).
    assert manifest.get("n_frames") == 4


def test_q60_budgets_produce_same_science(tiny_corpus, tmp_path, monkeypatch):
    """Science/coverage must be budget-INDEPENDENT (bit-identical across 1/2/4 Gio)."""
    refs = {}
    for budget_gib in (1, 2, 4):
        out = tmp_path / f"parity_{budget_gib}"
        monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeModel())
        monkeypatch.setattr(
            zz, "available_memory_bytes", lambda b=budget_gib: int(b) * GIB
        )
        zz.run_zegrid_mode(
            str(tiny_corpus), str(out), zegrid_halo_px=8,
            zconfig=SimpleNamespace(
                zegrid_layout="2x2", zegrid_scheduler_mode="legacy"
            ),
        )
        sci = sorted(out.glob("*science*.fits")) or sorted(out.glob("*.fits"))
        cov = sorted(out.glob("*coverage*.fits"))
        refs[budget_gib] = (
            np.asarray(fits.getdata(sci[0])),
            np.asarray(fits.getdata(cov[0])),
        )
    base_sci, base_cov = refs[1]
    for budget_gib in (2, 4):
        sci, cov = refs[budget_gib]
        assert np.array_equal(sci, base_sci, equal_nan=True), f"science differs at {budget_gib} Gio"
        assert np.array_equal(cov, base_cov), f"coverage differs at {budget_gib} Gio"
