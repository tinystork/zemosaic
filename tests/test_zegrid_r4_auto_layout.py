"""ZM-ZEGRID-R4 targeted tests — RAM-aware Auto layout (Levier 1).

Tests:
* determinism (same inputs + budget -> same layout),
* budget monotonicity (tighter budget -> smaller or equal cells),
* floors trigger an explicit failure (never silent degradation),
* memory model prediction within a stated, justified tolerance on the R3 data,
* existing R1/R2/R3 tests stay green (run separately).

The geometry/memory-model tests use the REAL M106 manifest (header-only, no
pixel stacking) and are skipped when the M106 lights dir is absent. The memory
model is derived (not invented) — validated here against the persisted R3
per-cell records.

Artifact note: the Auto-layout run evidence lives OUTSIDE git under
``/home/tristan/zegrid_r4_*`` (see the run tool docstring).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zemosaic.core.zegrid import auto_layout as za
from zemosaic.core.zegrid import geometry as zg

LIGHTS = Path("/home/tristan/M106/lights")
R3_M106_OUT = Path("/home/tristan/zegrid_r3_m106_outputs")

MILLION = 2**20


def _m106_manifest():
    if not LIGHTS.exists():
        pytest.skip(f"M106 lights dir absent: {LIGHTS}")
    return zg.read_manifest(LIGHTS)


@pytest.fixture(scope="module")
def manifest():
    frames, _ = _m106_manifest()
    return frames


@pytest.fixture(scope="module")
def canvas(manifest):
    return zg.build_canvas(manifest)


def _r3_records():
    recs = []
    for p in sorted(R3_M106_OUT.glob("cell_r*.json")):
        d = json.loads(p.read_text())
        if d.get("status") != "complete":
            continue
        recs.append(d)
    return recs


# ---------------------------------------------------------------------------
# Memory model is DERIVED (not invented) and validated on the R3 data.
# ---------------------------------------------------------------------------

def test_memory_model_fit_matches_r3_records():
    """The fitted model is derived from the R3 records; refit must match the
    frozen FITTED_MEMORY_MODEL coefficients (bytes), proving it is measured not
    invented."""
    recs = _r3_records()
    if len(recs) < 10:
        pytest.skip("R3 per-cell records absent (run tools/zegrid_r3/run_executor.py)")
    model, stats = za.fit_memory_model(recs)
    assert model.coeff_bytes_per_n_px == pytest.approx(
        za.FITTED_MEMORY_MODEL.coeff_bytes_per_n_px, rel=1e-3
    )
    assert model.baseline_bytes == pytest.approx(
        za.FITTED_MEMORY_MODEL.baseline_bytes, rel=1e-3
    )
    assert stats["r2"] > 0.85, f"R^2 = {stats['r2']} too low (model not linear in N*area)"
    assert model.coeff_bytes_per_n_px > 0
    assert model.baseline_bytes > 0


def test_memory_model_prediction_tolerance_on_r3():
    """Mean prediction must be within a stated, justified tolerance of the
    measured R3 peak, and the CONSERVATIVE bound must cover every cell (safety
    margin from the residual distribution)."""
    recs = _r3_records()
    if len(recs) < 10:
        pytest.skip("R3 per-cell records absent")
    model = za.FITTED_MEMORY_MODEL
    max_rel = 0.0
    max_pos_rel = -1.0
    for d in recs:
        p = d["patch"]
        area = (p[2] - p[0]) * (p[3] - p[1])
        n = d["n_patch_contributors"]
        meas = d["peak_rss_kib"] * 1024.0
        pred = model.predict_peak_bytes(n, area)
        bound = model.predict_bound_bytes(n, area)
        rel = (meas - pred) / pred
        max_rel = max(max_rel, abs(rel))
        max_pos_rel = max(max_pos_rel, rel)
        # Conservative bound must never under-predict (the point of the margin).
        assert bound >= meas, (
            f"{d['cell_id']}: bound {bound/2**20:.1f} MiB < measured {meas/2**20:.1f} MiB"
        )
    # Stated tolerance on the MEAN: within +-30% (max |rel| residual observed is
    # ~29.3%, dominated by the two low-N corner cells which the linear model
    # OVER-predicts = the safe direction). The DANGEROUS direction
    # (under-prediction, measured > predicted) is bounded by +12.6%, which the
    # 15% safety margin covers with headroom.
    assert max_rel <= 0.30, f"max |rel residual| = {max_rel:.3f} exceeds stated 0.30"
    assert max_pos_rel <= za.DEFAULT_SAFETY_MARGIN_FRAC, (
        f"max positive (dangerous) rel residual = {max_pos_rel:.3f} exceeds "
        f"safety margin {za.DEFAULT_SAFETY_MARGIN_FRAC}"
    )


# ---------------------------------------------------------------------------
# Determinism + monotonicity + floors
# ---------------------------------------------------------------------------

def test_choose_layout_deterministic(manifest, canvas):
    d1 = za.choose_layout(canvas, manifest, int(1500 * MILLION))
    d2 = za.choose_layout(canvas, manifest, int(1500 * MILLION))
    assert (d1.nx, d1.ny) == (d2.nx, d2.ny)
    assert d1.refinement_factor == d2.refinement_factor
    assert d1.predicted_bound_bytes == d2.predicted_bound_bytes


def test_choose_layout_permutation_invariant(manifest, canvas):
    base = za.choose_layout(canvas, manifest, int(1500 * MILLION))
    rev = za.choose_layout(zg.build_canvas(list(reversed(manifest))), list(reversed(manifest)), int(1500 * MILLION))
    assert (base.nx, base.ny) == (rev.nx, rev.ny)


def test_budget_monotonicity_tighter_is_finer(manifest, canvas):
    """Tighter budget -> smaller-or-equal cells -> more-or-equal Nx/Ny."""
    loose = za.choose_layout(canvas, manifest, int(6000 * MILLION))
    tight = za.choose_layout(canvas, manifest, int(1500 * MILLION))
    assert tight.nx >= loose.nx
    assert tight.ny >= loose.ny
    assert tight.nx * tight.ny >= loose.nx * loose.ny
    # Cell area (from footprint / factor) is smaller for the tighter budget.
    assert tight.max_patch_area <= loose.max_patch_area


def test_loose_no_budget_picks_coarsest(manifest, canvas):
    """No budget -> coarsest sensible layout (Cell ~= median projected footprint)."""
    d = za.choose_layout(canvas, manifest, None)
    assert d.refinement_factor == 1.0
    # The coarsest layout is strictly coarser than the tight-budget one.
    tight = za.choose_layout(canvas, manifest, int(1500 * MILLION))
    assert d.nx <= tight.nx and d.ny <= tight.ny


def test_budget_respected_bound(manifest, canvas):
    """The chosen layout's conservative peak bound must fit the budget."""
    for budget_mb in (2000, 3000, 4000, 6000):
        d = za.choose_layout(canvas, manifest, int(budget_mb * MILLION))
        assert d.predicted_bound_bytes <= budget_mb * MILLION
        assert d.budget_bound_choice


def test_floors_trigger_explicit_failure(manifest, canvas):
    """A budget that cannot honour the floors raises LayoutInfeasible (never
    silently degrades science)."""
    with pytest.raises(za.LayoutInfeasible):
        za.choose_layout(canvas, manifest, int(1000 * MILLION))


def test_min_patch_area_floor_blocks_fine_layout(manifest, canvas):
    """A very high min_patch_area floor with a tight budget is infeasible."""
    floors = za.ScientificFloors(min_patch_area_px=400_000)
    with pytest.raises(za.LayoutInfeasible):
        za.choose_layout(canvas, manifest, int(1500 * MILLION), floors=floors)


def test_halo_overhead_floor_reported(manifest, canvas):
    d = za.choose_layout(canvas, manifest, int(2000 * MILLION))
    assert d.floors["max_halo_overhead"]["ok"] is True
    assert d.floors["min_patch_area_px"]["ok"] is True
    assert d.floors["min_contributors"]["ok"] is True
    assert d.floors["min_contributors"]["value"] >= 3


def test_provenance_fields_present(manifest, canvas):
    d = za.choose_layout(canvas, manifest, int(1500 * MILLION))
    dd = d.to_dict()
    for key in ("ram_budget_bytes", "available_bytes", "model", "floors",
                "predicted_bound_bytes", "predicted_mean_bytes", "nx", "ny",
                "budget_bound_choice", "max_patch_area", "refinement_factor"):
        assert key in dd, f"provenance missing {key}"
    assert "safety_margin_frac" in dd["model"]["model"]
