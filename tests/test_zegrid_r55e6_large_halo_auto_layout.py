"""ZM-ZEGRID-R55-E6 — bounded COARSER rescue for a large user-selected halo.

The real Windows witness: 66 M106 frames, canvas 2403x3278, GPU on, GUI halo
128 px, PLAN budget 10958.30 MiB. The historical factor-1 candidate (3x2) is
floor-feasible on patch area but violates ``max_halo_overhead`` (0.5257 > 0.5).
Finer candidates are WORSE for that floor, so the old "floors are monotonic in
refinement — stopping" path aborted with a misleading RAM/finer message even
though a COARSER 2x2 layout fits both floors and the actual mode-aware budget.

These tests pin the corrected contract:

* a 3x2 halo-floor rejection triggers a bounded coarser rescue that chooses the
  nearest coarser grid (2x2) when it satisfies the floors AND the real
  mode-aware RAM budget;
* the rescue is REFUSED when a bounded refinement-rank window forbids
  coarsening (pinned / already-committed runs never relayout);
* a rescue whose nearest floor-feasible candidate exceeds RAM fails truthfully
  (halo direction, no "choose finer" advice, no bogus "RAM budget" headline);
* an ordinary halo-64 run keeps the historical choice and NEVER enters the
  rescue path;
* pinned layouts keep their exact geometry and provenance;
* the gated real-corpus probe proves halo128 -> 2x2 and halo64 -> 3x2 at the
  exact observed Windows budget (skipped cleanly without the corpus).

No pixels are stacked; no full mosaic is built.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import ram_contract as rc

MIB = 2 ** 20
LIGHTS = Path("/home/tristan/M106/lights")

# Exact Windows breadcrumb observation (R55 witness). Host-specific values live
# ONLY in this test; the product never hardcodes them (production uses the raw
# available-RAM observation).
WITNESS_RAM_TOTAL_MIB = 32469.49609375
WITNESS_RAM_USED_MIB = 18771.62109375
WITNESS_RAW_AVAILABLE_MIB = 13697.875
WITNESS_PLAN_BUDGET_MIB = 10958.29999923706


# ---------------------------------------------------------------------------
# Synthetic corpus (3x3 frame grid; deterministic, no user data)
# ---------------------------------------------------------------------------

def _make_descriptor(i, j, h=1000, w=600, scale=0.001, ra_step=0.45, dec_step=0.4):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0 + i * ra_step, 30.0 + j * dec_step]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
    wcs.array_shape = (h, w)
    fid = f"e6_{i}_{j}.fits"
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid), source_path=f"/x/{fid}", shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(),
        header_sha256="", instrument="",
    )


def _build_3x3_corpus():
    return [_make_descriptor(i, j) for i in range(3) for j in range(3)]


class _HugeInmemModel:
    """In-memory bound that never fits: forces the mode-aware streaming path."""

    def predict_bound_bytes(self, n, area):
        return 10.0 ** 15

    def predict_peak_bytes(self, n, area):
        return 10.0 ** 15

    def to_dict(self):
        return {"stub": "huge-inmem"}


def _force_streaming(monkeypatch, stream_bytes):
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeInmemModel())

    def fake_est(n, hw, tile=zz.STREAM_TILE_SIZE, channels=3):
        return int(stream_bytes)

    monkeypatch.setattr(zz, "_estimate_streaming_bytes", fake_est)


def _choose(canvas, descs, **kwargs):
    logs = []
    layout = zz._choose_layout_mode_aware(
        canvas, descs, log_line=logs.append, footprints_out=[], **kwargs
    )
    return layout, logs


# ---------------------------------------------------------------------------
# 1. 3x2 halo-floor rejection -> nearest coarser 2x2 rescue
# ---------------------------------------------------------------------------

def test_large_halo_3x2_rejected_and_2x2_rescued(monkeypatch):
    descs = _build_3x3_corpus()
    canvas = zg.build_canvas(descs)
    mw, mh = zal.median_projected_footprint(descs, canvas)
    # Sanity on the synthetic geometry: factor-1 is exactly 3x2.
    assert (math.ceil(canvas.width / mw), math.ceil(canvas.height / mh)) == (3, 2)
    _force_streaming(monkeypatch, 1000)

    layout, logs = _choose(
        canvas, descs, ram_budget=10 ** 9, halo_px=80,
        floors=zal.ScientificFloors(),
    )

    assert (layout["nx"], layout["ny"]) == (2, 2), layout
    assert layout["cell_count"] == 4
    assert layout["layout_source"] == "auto"
    assert layout["layout_origin"] == "auto_coarser_rescue"
    rescue = layout["layout_rescue"]
    assert rescue["applied"] is True and rescue["reason"] == "max_halo_overhead"
    assert rescue["trigger"]["nx"] == 3 and rescue["trigger"]["ny"] == 2
    assert rescue["trigger"]["memory_evaluated"] is False
    # The user halo is EXACT: the trigger overhead is the halo-80 3x2 value and
    # the chosen 2x2 overhead is the halo-80 value (not the halo-64 value).
    assert rescue["trigger"]["halo_overhead"] == pytest.approx(0.585135, abs=1e-5)
    assert layout["floors"]["max_halo_overhead"]["value"] == pytest.approx(0.449611, abs=1e-5)
    assert layout["floors"]["max_halo_overhead"]["limit"] == 0.5
    assert layout["floors"]["max_halo_overhead"]["ok"] is True
    assert layout["floors"]["min_patch_area_px"]["limit"] == 50_000
    assert layout["floors"]["min_patch_area_px"]["ok"] is True
    # Exactly one rescue candidate was needed (nearest coarser first).
    assert [ (t["nx"], t["ny"]) for t in rescue["candidates_evaluated"] ] == [(2, 2)]
    assert rescue["candidates_evaluated"][0]["memory_ok"] is True
    # Logs tell the truth about provenance.
    joined = "\n".join(logs)
    assert "REJECTED SOLELY by max_halo_overhead" in joined
    assert "attempting bounded COARSER rescue" in joined
    assert "origin=auto_coarser_rescue" in joined


def test_coarser_rescue_family_is_deterministic_and_nearest_first():
    descs = _build_3x3_corpus()
    canvas = zg.build_canvas(descs)
    mw, mh = zal.median_projected_footprint(descs, canvas)
    fam_a = zal.coarser_rescue_grids(canvas, mw, mh, 3, 2)
    fam_b = zal.coarser_rescue_grids(canvas, mw, mh, 3, 2)
    assert fam_a == fam_b
    assert fam_a[0] == (2, 2)
    assert all(nx * ny < 6 for nx, ny in fam_a)
    assert fam_a[-1] == (1, 1)
    # Bounded: never an unbounded scan.
    assert len(fam_a) <= len(zal.COARSER_RESCUE_FACTORS) + 3


# ---------------------------------------------------------------------------
# 2. Rescue refused by a bounded refinement-rank window
# ---------------------------------------------------------------------------

def test_rescue_refused_by_rank_floor(monkeypatch):
    descs = _build_3x3_corpus()
    canvas = zg.build_canvas(descs)
    _force_streaming(monkeypatch, 1000)

    with pytest.raises(zal.LayoutInfeasible) as exc:
        zz._choose_layout_mode_aware(
            canvas, descs, ram_budget=10 ** 9, halo_px=80,
            floors=zal.ScientificFloors(), min_refinement_rank=6,
        )
    msg = str(exc.value)
    assert "refinement-rank window" in msg
    assert "forbids" in msg and "coarsening" in msg
    assert "Lower the anti-checkerboard halo" in msg
    # Never the wrong direction for a halo-floor conflict.
    assert "A FINER layout (more cells) is the direction" not in msg
    # No bogus RAM-budget headline when no memory candidate was evaluated.
    assert not msg.startswith("RAM budget cannot satisfy scientific floors")


# ---------------------------------------------------------------------------
# 3. Rescue RAM failure -> truthful direction (no "choose finer")
# ---------------------------------------------------------------------------

def test_rescue_ram_failure_reports_truthful_direction(monkeypatch):
    descs = _build_3x3_corpus()
    canvas = zg.build_canvas(descs)
    _force_streaming(monkeypatch, 1000)

    with pytest.raises(zal.LayoutInfeasible) as exc:
        zz._choose_layout_mode_aware(
            canvas, descs, ram_budget=500, halo_px=80,
            floors=zal.ScientificFloors(),
        )
    msg = str(exc.value)
    assert "nearest floor-feasible COARSER rescue 2x2" in msg
    assert "Coarsening further enlarges the per-cell patch area" in msg
    assert "Lower the anti-checkerboard halo" in msg
    assert "A FINER layout (more cells) is the direction" not in msg
    assert "RAM budget cannot satisfy scientific floors" not in msg


# ---------------------------------------------------------------------------
# 4. Ordinary (halo 64) parity: no rescue, historical choice preserved
# ---------------------------------------------------------------------------

def test_ordinary_halo_choice_unchanged_and_never_rescued(monkeypatch):
    descs = _build_3x3_corpus()
    canvas = zg.build_canvas(descs)
    _force_streaming(monkeypatch, 1000)

    layout, logs = _choose(
        canvas, descs, ram_budget=10 ** 9, halo_px=64,
        floors=zal.ScientificFloors(),
    )
    assert (layout["nx"], layout["ny"]) == (3, 2)
    assert layout["layout_origin"] == "auto_refinement"
    assert layout["layout_source"] == "auto"
    assert "layout_rescue" not in layout
    assert "rescue" not in "\n".join(logs).lower()
    # The factor-1 candidate still carries the historical refinement factor.
    assert layout["refinement_factor"] == 1.0


def test_ram_failure_with_finer_blocked_by_halo_never_advises_finer(monkeypatch):
    """First candidate passes the halo floor but fails RAM; the next finer
    candidate is blocked by halo. The error must not recommend FINER."""
    descs = _build_3x3_corpus()
    canvas = zg.build_canvas(descs)
    _force_streaming(monkeypatch, 1000)
    with pytest.raises(zal.LayoutInfeasible) as exc:
        zz._choose_layout_mode_aware(
            canvas, descs, ram_budget=500, halo_px=64,
            floors=zal.ScientificFloors(),
        )
    msg = str(exc.value)
    assert "Further refinement is BLOCKED by max_halo_overhead" in msg
    assert "A FINER layout (more cells) is the direction" not in msg
    assert "lower the anti-checkerboard halo" in msg


# ---------------------------------------------------------------------------
# 5. Pinned layouts: exact geometry, no rescue, truthful halo advice
# ---------------------------------------------------------------------------

def test_pinned_large_halo_refuses_without_rescue(monkeypatch):
    descs = _build_3x3_corpus()
    canvas = zg.build_canvas(descs)
    _force_streaming(monkeypatch, 1000)
    logs = []
    with pytest.raises(zal.LayoutInfeasible) as exc:
        zz._choose_layout_mode_aware(
            canvas, descs, ram_budget=10 ** 9, halo_px=80,
            floors=zal.ScientificFloors(), pinned_layout=(3, 2),
            log_line=logs.append,
        )
    msg = str(exc.value)
    assert "pinned layout 3x2 violates scientific floors" in msg
    assert "Lower the anti-checkerboard halo" in msg
    # Never recommend pinning FINER for a halo-floor conflict.
    assert "pinning a FINER layout" not in msg
    assert "zegrid_layout" not in msg
    assert "attempting bounded COARSER rescue" not in "\n".join(logs)


def test_pinned_feasible_layout_keeps_exact_geometry(monkeypatch):
    descs = _build_3x3_corpus()
    canvas = zg.build_canvas(descs)
    _force_streaming(monkeypatch, 1000)
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=10 ** 9, halo_px=80,
        floors=zal.ScientificFloors(), pinned_layout=(2, 2),
        footprints_out=[],
    )
    assert (layout["nx"], layout["ny"]) == (2, 2)
    assert layout["layout_source"] == "pinned"
    assert layout["layout_origin"] == "pinned"
    assert "layout_rescue" not in layout


# ---------------------------------------------------------------------------
# 6. Gated real M106 planning probe (read-only; no pixels, no mosaic)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def m106_witness():
    if not LIGHTS.is_dir():
        pytest.skip(f"M106 lights dir absent: {LIGHTS}")
    frames, _rejected = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(frames)
    # The witness geometry is part of the evidence, not an assumption.
    assert len(frames) == 66
    assert (canvas.width, canvas.height) == (2403, 3278)
    total = int(round(WITNESS_RAM_TOTAL_MIB * MIB))
    used = int(round(WITNESS_RAM_USED_MIB * MIB))
    raw_available = total - used
    budget = rc.plan_budget_bytes(raw_available)
    return frames, canvas, raw_available, budget


def test_witness_budget_arithmetic_is_exact(m106_witness):
    _frames, _canvas, raw_available, budget = m106_witness
    assert raw_available / MIB == pytest.approx(WITNESS_RAW_AVAILABLE_MIB, abs=1e-9)
    # Diagnostic inference only: production uses the raw available observation.
    # B_plan = (4 * available) // 5, integer-exact.
    assert budget == (4 * raw_available) // 5
    assert budget / MIB == pytest.approx(WITNESS_PLAN_BUDGET_MIB, abs=1e-6)


def test_witness_halo128_rescues_to_2x2(m106_witness):
    frames, canvas, raw_available, budget = m106_witness
    logs = []
    layout = zz._choose_layout_mode_aware(
        canvas, frames, ram_budget=budget, available_bytes=raw_available,
        halo_px=128, log_line=logs.append, footprints_out=[],
    )
    assert (layout["nx"], layout["ny"]) == (2, 2)
    assert layout["layout_origin"] == "auto_coarser_rescue"
    assert layout["layout_source"] == "auto"
    assert layout["cpu_streaming"]["selected_tile_size"] == 128
    assert layout["predicted_bound_bytes"] / MIB == pytest.approx(3588.6142578125, abs=0.01)
    assert layout["floors"]["max_halo_overhead"]["ok"] is True
    assert layout["floors"]["max_halo_overhead"]["limit"] == 0.5
    assert layout["floors"]["max_halo_overhead"]["value"] == pytest.approx(0.402437, abs=1e-5)
    rescue = layout["layout_rescue"]
    assert rescue["trigger"]["halo_overhead"] == pytest.approx(0.525713, abs=1e-5)
    assert rescue["chosen"] == {
        "nx": 2, "ny": 2, "cell_count": 4,
        "effective_coarsening_factor": rescue["chosen"]["effective_coarsening_factor"],
    }
    # Exact requested halo, not a silent 64 clamp.
    assert layout["floors"]["max_halo_overhead"]["value"] < rescue["trigger"]["halo_overhead"]
    joined = "\n".join(logs)
    assert "origin=auto_coarser_rescue" in joined
    assert "candidate 2x2 (4 cells) ACCEPTED" in joined


def test_witness_halo64_keeps_historical_3x2(m106_witness):
    frames, canvas, raw_available, budget = m106_witness
    layout = zz._choose_layout_mode_aware(
        canvas, frames, ram_budget=budget, available_bytes=raw_available,
        halo_px=64, footprints_out=[],
    )
    assert (layout["nx"], layout["ny"]) == (3, 2)
    assert layout["layout_origin"] == "auto_refinement"
    assert "layout_rescue" not in layout
    assert layout["cpu_streaming"]["selected_tile_size"] == 128
    assert layout["predicted_bound_bytes"] / MIB == pytest.approx(2646.4306640625, abs=0.01)
    assert layout["floors"]["max_halo_overhead"]["ok"] is True
