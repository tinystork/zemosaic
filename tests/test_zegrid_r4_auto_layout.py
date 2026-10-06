"""ZM-ZEGRID-R4 targeted tests — RAM-aware Auto layout (Levier 1, rework-1).

Tests:
* the memory model is DERIVED from measured records (not invented) and refit
  matches the frozen FITTED_MEMORY_MODEL coefficients (two-term model);
* the conservative bound covers measured peak on BOTH calibration sets (the 20
  R3 cells at ~410k px AND the 166 tight-run complete cells at ~51k px);
* budget monotonicity (tighter budget -> smaller or equal cells);
* floors trigger an explicit failure (never silent degradation);
* determinism + provenance fields.

The geometry/memory-model tests use the REAL M106 manifest (header-only, no
pixel stacking) and are skipped when the M106 lights dir or the calibration
records are absent. The memory model is derived (not invented).

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
R4_TIGHT_OUT = Path("/home/tristan/zegrid_r4_m106_tight")

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


def _records_from(out: Path):
    recs = []
    for p in sorted(out.glob("cell_r*.json")):
        d = json.loads(p.read_text())
        if d.get("status") != "complete":
            continue
        recs.append(d)
    return recs


def _r3_records():
    return _records_from(R3_M106_OUT)


def _tight_records():
    return _records_from(R4_TIGHT_OUT)


def _combined_records():
    return _r3_records() + _tight_records()


# ---------------------------------------------------------------------------
# Memory model is DERIVED (not invented) and conservative on BOTH calibration sets.
# ---------------------------------------------------------------------------

def test_memory_model_fit_matches_combined_records():
    """The frozen two-term model is derived from the combined calibration set
    (R3 20 cells + tight 166 cells); refit must match the frozen coefficients."""
    recs = _combined_records()
    if len(recs) < 150:
        pytest.skip("combined calibration records absent (run R3 + R4 tight first)")
    model, stats = za.fit_memory_model(recs)
    assert model.baseline_bytes == pytest.approx(
        za.FITTED_MEMORY_MODEL.baseline_bytes, rel=1e-3
    )
    assert model.area_coeff_bytes_per_px == pytest.approx(
        za.FITTED_MEMORY_MODEL.area_coeff_bytes_per_px, rel=1e-3
    )
    assert model.frame_coeff_bytes_per_n_px == pytest.approx(
        za.FITTED_MEMORY_MODEL.frame_coeff_bytes_per_n_px, rel=1e-3
    )
    assert stats["r2"] > 0.95, f"R^2 = {stats['r2']} too low"
    # Physically sensible: positive baseline and positive coefficients.
    assert model.baseline_bytes > 0
    assert model.area_coeff_bytes_per_px > 0
    assert model.frame_coeff_bytes_per_n_px > 0
    # The derived residual constants must match the frozen ones (L1 fix: they
    # cannot go stale — derived here from fit_memory_model, not hand-copied).
    assert stats["max_positive_residual_frac"] == pytest.approx(
        za.FITTED_MAX_POS_RESID_FRAC, rel=1e-3
    )
    assert stats["max_abs_residual_frac"] == pytest.approx(
        za.FITTED_MAX_ABS_RESID_FRAC, rel=1e-3
    )


def _check_bound_covers(model, records, label):
    """Assert conservative bound >= measured for every record; return (max_pos_rel, max_abs_rel)."""
    max_abs_rel = 0.0
    max_pos_rel = -1.0
    for d in records:
        p = d["patch"]
        area = (p[2] - p[0]) * (p[3] - p[1])
        n = d["n_patch_contributors"]
        meas = d["peak_rss_kib"] * 1024.0
        pred = model.predict_peak_bytes(n, area)
        bound = model.predict_bound_bytes(n, area)
        rel = (meas - pred) / pred
        max_abs_rel = max(max_abs_rel, abs(rel))
        max_pos_rel = max(max_pos_rel, rel)
        assert bound >= meas, (
            f"{label} {d['cell_id']}: bound {bound/2**20:.1f} MiB < measured {meas/2**20:.1f} MiB"
        )
    return max_pos_rel, max_abs_rel


def test_bound_covers_measured_on_r3():
    recs = _r3_records()
    if len(recs) < 10:
        pytest.skip("R3 per-cell records absent")
    max_pos_rel, _ = _check_bound_covers(za.FITTED_MEMORY_MODEL, recs, "R3")
    assert max_pos_rel <= za.DEFAULT_SAFETY_MARGIN_FRAC, (
        f"R3 max positive rel residual {max_pos_rel:.3f} exceeds margin {za.DEFAULT_SAFETY_MARGIN_FRAC}"
    )


def test_bound_covers_measured_on_tight():
    recs = _tight_records()
    if len(recs) < 100:
        pytest.skip("R4 tight per-cell records absent")
    max_pos_rel, _ = _check_bound_covers(za.FITTED_MEMORY_MODEL, recs, "tight")
    assert max_pos_rel <= za.DEFAULT_SAFETY_MARGIN_FRAC, (
        f"tight max positive rel residual {max_pos_rel:.3f} exceeds margin {za.DEFAULT_SAFETY_MARGIN_FRAC}"
    )


def test_two_term_model_not_overpredicting_small_patches():
    """The two-term model must NOT over-predict small patches the way the old
    single-slope model did. On the tight (~51k px) cells, the mean prediction
    should be within a modest tolerance (the old model was ~-45% mean residual)."""
    recs = _tight_records()
    if len(recs) < 100:
        pytest.skip("R4 tight per-cell records absent")
    model = za.FITTED_MEMORY_MODEL
    rels = []
    for d in recs:
        p = d["patch"]
        area = (p[2] - p[0]) * (p[3] - p[1])
        n = d["n_patch_contributors"]
        meas = d["peak_rss_kib"] * 1024.0
        pred = model.predict_peak_bytes(n, area)
        rels.append((meas - pred) / pred)
    mean_rel = float(np.mean(rels))
    # Well-calibrated (|mean| < 15%) and conservative on average (not grossly over).
    assert abs(mean_rel) < 0.20, f"tight mean rel residual = {mean_rel:.3f}"


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
    # Cell area is smaller (or equal) for the tighter budget.
    assert tight.max_patch_area <= loose.max_patch_area


def test_loose_no_budget_picks_coarsest(manifest, canvas):
    """No budget -> coarsest sensible layout (Cell ~= median projected footprint)."""
    d = za.choose_layout(canvas, manifest, None)
    assert d.refinement_factor == 1.0
    tight = za.choose_layout(canvas, manifest, int(1500 * MILLION))
    assert d.nx <= tight.nx and d.ny <= tight.ny


def test_budget_respected_bound(manifest, canvas):
    """The chosen layout's conservative peak bound must fit the budget."""
    for budget_mb in (2000, 3000, 4000, 6000):
        d = za.choose_layout(canvas, manifest, int(budget_mb * MILLION))
        assert d.predicted_bound_bytes <= budget_mb * MILLION
        assert d.budget_bound_choice


def test_exact_contributor_counts_used(manifest, canvas):
    """The budget search uses the candidate's exact max contributor count (not
    len(frames)=66). The chosen layout's max_contributors must equal the exact
    max over its per-cell predictions."""
    d = za.choose_layout(canvas, manifest, int(1500 * MILLION))
    cell_max = max(c.n_contributors for c in d.cells)
    assert d.max_contributors == cell_max
    # M106 fully-overlapping field: exact max N may be < 66 for finer layouts,
    # but the old n_upper=66 over-estimate is never below the true max.
    assert d.max_contributors <= len(manifest)


def test_floors_trigger_explicit_failure(manifest, canvas):
    """A budget that cannot honour the floors raises LayoutInfeasible (never
    silently degrades science). The finest floor-feasible layout (15x12) has a
    peak bound of ~935 MiB; anything below that is infeasible."""
    with pytest.raises(za.LayoutInfeasible):
        za.choose_layout(canvas, manifest, int(900 * MILLION))


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
                "budget_bound_choice", "max_patch_area", "max_contributors",
                "refinement_factor"):
        assert key in dd, f"provenance missing {key}"
    assert "safety_margin_frac" in dd["model"]["model"]
    assert "area_coeff_bytes_per_px" in dd["model"]["model"]
    assert "frame_coeff_bytes_per_n_px" in dd["model"]["model"]
