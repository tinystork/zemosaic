"""ZM-HALO-EXECUTION-DECOUPLING — explicit halo is science, overhead is cost.

Pins the policy/mechanism split introduced by
``ZM-HALO-EXECUTION-DECOUPLING-20261109``:

* ``max_halo_overhead`` is an EFFICIENCY ceiling, never a scientific feasibility
  rule. Under the ADVISORY policy (a user-requested halo) an exceedance is
  reported and accepted; it never rejects a layout, never arms the coarser
  rescue and never coarsens the logical Cells.
* product-default / internal halos and every direct/unit caller keep the
  historical HARD FLOOR byte for byte.
* the halo width itself, memberships, WCS projections, the patch-area floor and
  the 15 % / joint resource envelopes stay fail-closed under BOTH policies.

The real LP witness (341 Caldwell LP frames, canvas 3477x4106, factor-1 4x3,
halo-128 overhead 0.5363, PLAN budget 2011.4 MiB) is gated on the read-only
input corpus. No pixels are stacked; nothing is written to any output folder.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.wcs import WCS

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic import zegrid_halo as zhalo
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg

MIB = 2 ** 20
LP_LIGHTS = Path("/home/tristan/caldwell11/lights organized/EQ/LP")

# Exact observed LP_fix halo-128 facts (real run, 2026-11-09).
LP_CANVAS = (3477, 4106)
LP_FRAMES = 341
LP_BUDGET_MIB = 2011.4
LP_FACTOR1 = (4, 3)
LP_HALO128_OVERHEAD = 0.5363


# ---------------------------------------------------------------------------
# Synthetic corpus: 5x4 frame grid -> factor-1 grid 4x3, halo-128 overhead ~0.98
# ---------------------------------------------------------------------------

def _make_descriptor(i, j, h=1000, w=600, scale=0.001, ra_step=0.45, dec_step=0.4):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0 + i * ra_step, 30.0 + j * dec_step]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
    wcs.array_shape = (h, w)
    fid = f"hdec_{i}_{j}.fits"
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid), source_path=f"/x/{fid}", shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(),
        header_sha256="", instrument="",
    )


def _corpus():
    return [_make_descriptor(i, j) for i in range(5) for j in range(4)]


class _HugeInmemModel:
    """In-memory bound that never fits: forces the mode-aware streaming path."""

    def predict_bound_bytes(self, n, area):
        return 10.0 ** 15

    def predict_peak_bytes(self, n, area):
        return 10.0 ** 15

    def to_dict(self):
        return {"stub": "huge-inmem"}


def _force_area_streaming(monkeypatch, bytes_per_px):
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeInmemModel())

    def fake_est(n, hw, tile=zz.STREAM_TILE_SIZE, channels=3, storage_profile=None):
        return int(int(hw[0]) * int(hw[1]) * int(bytes_per_px))

    monkeypatch.setattr(zz, "_estimate_streaming_bytes", fake_est)


def _force_no_streaming(monkeypatch, bound=10 ** 13):
    def fake_est(n, hw, tile=zz.STREAM_TILE_SIZE, channels=3, storage_profile=None):
        return int(bound)

    monkeypatch.setattr(zz, "_estimate_streaming_bytes", fake_est)


def _choose(canvas, descs, **kwargs):
    logs = []
    layout = zz._choose_layout_mode_aware(
        canvas, descs, log_line=logs.append, footprints_out=[], **kwargs
    )
    return layout, logs


@pytest.fixture(scope="module")
def synth():
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    return descs, canvas


# ---------------------------------------------------------------------------
# 1. The observed real geometry is exactly the historical floor's victim
# ---------------------------------------------------------------------------

def test_real_lp_factor1_nominal_geometry_reproduces_observed_overhead():
    canvas = SimpleNamespace(width=LP_CANVAS[0], height=LP_CANVAS[1])
    geom = zal._nominal_geometry(canvas, LP_FACTOR1[0], LP_FACTOR1[1], 128)
    assert geom["halo_overhead"] == pytest.approx(LP_HALO128_OVERHEAD, abs=5e-5)
    assert geom["halo_overhead"] > 0.5
    assert geom["min_patch_area"] > 50_000


# ---------------------------------------------------------------------------
# 2. Policy provenance mapping (product default keeps the hard floor)
# ---------------------------------------------------------------------------

def test_policy_for_source_keeps_default_and_internal_hard_floor():
    assert (
        zhalo.halo_overhead_policy_for_source(zhalo.ZEGRID_HALO_SOURCE_EXPLICIT)
        == zhalo.HALO_OVERHEAD_POLICY_ADVISORY
    )
    assert (
        zhalo.halo_overhead_policy_for_source(zhalo.ZEGRID_HALO_SOURCE_CONFIG)
        == zhalo.HALO_OVERHEAD_POLICY_ADVISORY
    )
    assert (
        zhalo.halo_overhead_policy_for_source(
            zhalo.ZEGRID_HALO_SOURCE_PRODUCT_DEFAULT
        )
        == zhalo.HALO_OVERHEAD_POLICY_HARD_FLOOR
    )
    assert (
        zhalo.halo_overhead_policy_for_source(
            zhalo.ZEGRID_HALO_SOURCE_INTERNAL_ONLY
        )
        == zhalo.HALO_OVERHEAD_POLICY_HARD_FLOOR
    )
    # An omitted policy is the historical hard floor, never advisory.
    assert (
        zhalo.normalize_halo_overhead_policy(None)
        == zhalo.HALO_OVERHEAD_POLICY_HARD_FLOOR
    )
    with pytest.raises(zhalo.ZeGridHaloError):
        zhalo.normalize_halo_overhead_policy("nope")


# ---------------------------------------------------------------------------
# 3. Advisory: the factor-1 explicit-halo layout is admitted, no rescue
# ---------------------------------------------------------------------------

def test_advisory_explicit_halo_admits_factor1_layout(synth):
    descs, canvas = synth
    layout, logs = _choose(
        canvas, descs, ram_budget=10 * 2 ** 30, halo_px=128,
        floors=zal.ScientificFloors(),
        halo_policy=zhalo.HALO_OVERHEAD_POLICY_ADVISORY,
    )
    assert (layout["nx"], layout["ny"]) == (4, 3)
    assert layout["layout_origin"] == "auto_refinement"
    assert "layout_rescue" not in layout
    assert layout["halo_overhead_policy"] == zhalo.HALO_OVERHEAD_POLICY_ADVISORY
    adv = layout["halo_overhead_advisory"]
    assert adv["halo_px"] == 128
    assert adv["threshold"] == 0.5
    assert adv["value"] > 0.5
    assert adv["state"] == "exceeded_advisory_accepted"
    assert adv["geometry_changed"] is False
    assert layout["floors"]["max_halo_overhead"]["ok"] is False
    assert layout["floors"]["max_halo_overhead"]["state"] == (
        "exceeded_advisory_accepted"
    )
    joined = "\n".join(logs)
    assert "ADVISORY" in joined
    assert "rescue" not in joined.lower()


def test_hard_floor_still_rejects_the_same_geometry(synth):
    """The SAME corpus/budget under the historical policy must not change."""
    descs, canvas = synth
    layout, logs = _choose(
        canvas, descs, ram_budget=10 * 2 ** 30, halo_px=128,
        floors=zal.ScientificFloors(),  # halo_policy omitted -> hard floor
    )
    assert layout["layout_origin"] == "auto_coarser_rescue"
    assert layout["cell_count"] < 12
    assert layout["floors"]["max_halo_overhead"]["ok"] is True
    joined = "\n".join(logs)
    assert "REJECTED SOLELY by max_halo_overhead" in joined
    assert "advisory" not in joined.lower()


# ---------------------------------------------------------------------------
# 4. Advisory: finer candidates continue to be scanned and are selected
# ---------------------------------------------------------------------------

def test_advisory_scans_finer_memory_feasible_candidates(monkeypatch, synth):
    descs, canvas = synth
    _force_area_streaming(monkeypatch, 512)
    geom4 = zal._nominal_geometry(canvas, 4, 3, 128)
    budget = int(geom4["max_patch_area"]) * 512 // 2

    layout, logs = _choose(
        canvas, descs, ram_budget=budget, halo_px=128,
        floors=zal.ScientificFloors(), enforce_joint=False,
        halo_policy=zhalo.HALO_OVERHEAD_POLICY_ADVISORY,
    )
    assert layout["cell_count"] > 12
    assert layout["layout_origin"] == "auto_refinement"
    assert "layout_rescue" not in layout
    assert layout["predicted_bound_bytes"] <= budget
    assert layout["floors"]["max_halo_overhead"]["ok"] is False
    joined = "\n".join(logs)
    assert "candidate 4x3 REJECTED — cheaper-mode bound" in joined


# ---------------------------------------------------------------------------
# 5. Pinned explicit halo: exact geometry despite advisory exceedance
# ---------------------------------------------------------------------------

def test_pinned_advisory_keeps_exact_geometry(synth):
    descs, canvas = synth
    layout, logs = _choose(
        canvas, descs, ram_budget=10 * 2 ** 30, halo_px=128,
        floors=zal.ScientificFloors(), pinned_layout=(4, 3),
        halo_policy=zhalo.HALO_OVERHEAD_POLICY_ADVISORY,
    )
    assert (layout["nx"], layout["ny"]) == (4, 3)
    assert layout["layout_source"] == "pinned"
    assert layout["layout_origin"] == "pinned"
    assert layout["floors"]["max_halo_overhead"]["ok"] is False
    assert layout["floors"]["max_halo_overhead"]["policy"] == (
        zhalo.HALO_OVERHEAD_POLICY_ADVISORY
    )
    assert any("MAX_HALO_OVERHEAD_ADVISORY" in w for w in layout["warnings"])
    assert "ADVISORY" in "\n".join(logs)


def test_pinned_hard_floor_keeps_historical_refusal(synth):
    descs, canvas = synth
    with pytest.raises(zal.LayoutInfeasible) as exc:
        zz._choose_layout_mode_aware(
            canvas, descs, ram_budget=10 * 2 ** 30, halo_px=128,
            floors=zal.ScientificFloors(), pinned_layout=(4, 3),
            footprints_out=[],
        )
    msg = str(exc.value)
    assert "violates scientific floors" in msg
    assert "ADVISORY" not in msg


# ---------------------------------------------------------------------------
# 6. External working set at the SAME pinned geometry when no RAM mode fits
# ---------------------------------------------------------------------------

def test_pinned_explicit_halo_keeps_geometry_for_external_class(monkeypatch, synth):
    descs, canvas = synth
    _force_no_streaming(monkeypatch)
    logs = []
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=1 * MIB, halo_px=128,
        floors=zal.ScientificFloors(), pinned_layout=(4, 3),
        allow_external=True, log_line=logs.append, footprints_out=[],
        halo_policy=zhalo.HALO_OVERHEAD_POLICY_ADVISORY,
    )
    assert (layout["nx"], layout["ny"]) == (4, 3)
    assert layout["layout_source"] == "pinned"
    assert "M5" in "\n".join(logs)


# ---------------------------------------------------------------------------
# 7. Scientific invariants: exact halo, patch geometry, memberships, one pass
# ---------------------------------------------------------------------------

def test_advisory_preserves_halo_patch_geometry_and_membership(synth):
    descs, canvas = synth
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=10 * 2 ** 30, halo_px=128,
        floors=zal.ScientificFloors(), footprints_out=[],
        halo_policy=zhalo.HALO_OVERHEAD_POLICY_ADVISORY,
    )
    nx, ny = int(layout["nx"]), int(layout["ny"])
    grid = zg.build_layout(canvas, nx, ny)
    for row, col, bounds in grid.iter_cells(canvas):
        cid = zg.cell_id(row, col)
        cell = zg.ZeGridCell(cid, canvas.canvas_id, grid.layout_id, row, col, bounds)
        patch = zg.build_patch(canvas, cell, 128)
        assert patch.halo_px == 128
        # The patch is the core expanded by EXACTLY the requested halo on every
        # side that exists, clipped only by the canvas border.
        exp_x0 = max(0, int(cell.core.x0) - 128)
        exp_y0 = max(0, int(cell.core.y0) - 128)
        exp_x1 = min(canvas.width, int(cell.core.x1) + 128)
        exp_y1 = min(canvas.height, int(cell.core.y1) + 128)
        assert (patch.patch.x0, patch.patch.y0) == (exp_x0, exp_y0)
        assert (patch.patch.x1, patch.patch.y1) == (exp_x1, exp_y1)
    # The layout's exact contributor counts are the SAME membership primitive.
    fps = zal.project_footprints(descs, canvas)
    cells = list(
        zz._iter_cell_bounds_mode_aware(canvas, fps, nx, ny, 128, zz.STREAM_TILE_SIZE)
    )
    assert max(c["n"] for c in cells) == layout["max_contributors"]
    assert layout["floors"]["min_contributors"]["value"] == layout["max_contributors"]


def test_advisory_layout_projects_each_frame_exactly_once(monkeypatch, synth):
    descs, canvas = synth
    calls = {"n": 0}
    real = zg._source_polygon

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(zg, "_source_polygon", counted)
    zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=10 * 2 ** 30, halo_px=128,
        floors=zal.ScientificFloors(), footprints_out=[],
        halo_policy=zhalo.HALO_OVERHEAD_POLICY_ADVISORY,
    )
    assert calls["n"] == len(descs)


# ---------------------------------------------------------------------------
# 8. Envelopes stay fail-closed and truthful (no halo-masquerading refusal)
# ---------------------------------------------------------------------------

def test_advisory_tiny_budget_refuses_as_host_ram_not_halo(monkeypatch, synth):
    descs, canvas = synth
    _force_no_streaming(monkeypatch)
    with pytest.raises(zal.LayoutInfeasible) as exc:
        zz._choose_layout_mode_aware(
            canvas, descs, ram_budget=1, halo_px=128,
            floors=zal.ScientificFloors(), footprints_out=[],
            halo_policy=zhalo.HALO_OVERHEAD_POLICY_ADVISORY,
        )
    msg = str(exc.value)
    refusal = getattr(exc.value, "refusal", {})
    assert refusal.get("kind") == "host_ram"
    assert "requested halo" not in msg
    assert "max_halo_overhead" not in msg
    assert "RAM budget" in msg


# ---------------------------------------------------------------------------
# 9. Manifest publication: advisory is additive, hard floor is unchanged
# ---------------------------------------------------------------------------

def test_manifest_layout_block_publishes_advisory_only_when_advisory(synth):
    descs, canvas = synth
    advisory_layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=10 * 2 ** 30, halo_px=128,
        floors=zal.ScientificFloors(), footprints_out=[],
        halo_policy=zhalo.HALO_OVERHEAD_POLICY_ADVISORY,
    )
    block = zz._manifest_layout_block(
        advisory_layout["nx"], advisory_layout["ny"], advisory_layout
    )
    assert block["halo_overhead_policy"] == zhalo.HALO_OVERHEAD_POLICY_ADVISORY
    assert block["halo_overhead_advisory"]["value"] > 0.5
    assert block["halo_overhead_advisory"]["geometry_changed"] is False

    hard_layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=10 * 2 ** 30, halo_px=128,
        floors=zal.ScientificFloors(), footprints_out=[],
    )
    hard_block = zz._manifest_layout_block(
        hard_layout["nx"], hard_layout["ny"], hard_layout
    )
    assert "halo_overhead_policy" not in hard_block
    assert "halo_overhead_advisory" not in hard_block


# ---------------------------------------------------------------------------
# 10. Gated REAL LP witness (read-only corpus; no output folder, no product run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lp_witness():
    if not LP_LIGHTS.is_dir():
        pytest.skip(f"LP lights dir absent: {LP_LIGHTS}")
    frames, rejected = zg.read_manifest(LP_LIGHTS)
    canvas = zg.build_canvas(frames)
    assert rejected == []
    assert len(frames) == LP_FRAMES
    assert (canvas.width, canvas.height) == LP_CANVAS
    return frames, canvas


def test_lp_witness_factor1_is_4x3_and_halo128_overhead_5363(lp_witness):
    frames, canvas = lp_witness
    mw, mh = zal.median_projected_footprint(frames, canvas)
    assert (
        math.ceil(canvas.width / mw), math.ceil(canvas.height / mh)
    ) == LP_FACTOR1
    geom = zal._nominal_geometry(canvas, 4, 3, 128)
    assert geom["halo_overhead"] == pytest.approx(LP_HALO128_OVERHEAD, abs=5e-5)


def test_lp_witness_advisory_produces_a_truthful_plan(lp_witness):
    frames, canvas = lp_witness
    budget = int(LP_BUDGET_MIB * MIB)
    logs = []
    layout = zz._choose_layout_mode_aware(
        canvas, frames, ram_budget=budget, available_bytes=budget,
        halo_px=128, floors=zal.ScientificFloors(), log_line=logs.append,
        footprints_out=[], allow_external=True,
        halo_policy=zhalo.HALO_OVERHEAD_POLICY_ADVISORY,
    )
    # The explicit halo is NOT changed and no coarser rescue was used.
    assert "layout_rescue" not in layout
    assert layout["layout_origin"] == "auto_refinement"
    assert layout["halo_overhead_advisory"]["halo_px"] == 128
    assert layout["halo_overhead_advisory"]["exceeded"] is True
    # A truthful, memory-admitted plan exists at this budget.
    assert layout["predicted_bound_bytes"] <= budget
    assert layout["cell_count"] > 12
    assert layout["floors"]["min_patch_area_px"]["ok"] is True
    assert layout["cpu_streaming"]["selected_tile_size"] is not None
    joined = "\n".join(logs)
    assert "origin=auto_refinement" in joined
    assert "rescue" not in joined.lower()
