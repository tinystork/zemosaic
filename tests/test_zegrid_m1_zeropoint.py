"""ZM-ANTICHESSBOARD-ZEROPOINT M1 — anchor-independent additive zero point.

Fast, synthetic. Covers the M1 scientific contract:

* the zero-point gauge is a pure per-channel constant: pairwise offset
  differences, edge observations and relative photometry are preserved;
* the assembled science is invariant to the CHOICE OF SOLVER ANCHOR (same
  graph/patches, two anchors, tight numeric tolerance);
* the assembled background is stable under an EQUIVALENT RE-PARTITION of the
  same additive field (2x2 vs 1x4, linear ramp: support-weighted mean is the
  field value at the canvas centroid for equal-area Cells);
* Cell iteration order does not matter;
* explicit adverse policies: zero/negative/non-finite weights, huge weights,
  non-finite offsets (fail-closed), zero-support Cells, disconnected graphs;
* the manifest block is strict JSON (no NaN/Infinity), deterministic, exposes
  the policy/constant/weights/status, and ``offsets`` are the EFFECTIVE ones;
* ``cell_gauge_mode=global`` remains untouched.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from zemosaic.core.zegrid import cell_feather as zcf
from zemosaic.core.zegrid import cell_offset as zco
from zemosaic.core.zegrid import geometry as zg


CORE = 32
HALO = 8
LEVELS_2X2 = {
    "r0000c0000": 100.0, "r0000c0001": 104.0,
    "r0001c0000": 96.0, "r0001c0001": 110.0,
}
LEVELS_3X3 = {
    f"r{r:04d}c{c:04d}": 100.0 + 7.0 * r - 5.0 * c + 2.0 * r * c
    for r in range(3) for c in range(3)
}


def _canvas(nx, ny, core=CORE, *, height=None, width=None):
    return zg.GlobalCanvas(
        canvas_id=f"m1-{nx}x{ny}", wcs_header="",
        width=int(nx * core if width is None else width),
        height=int(ny * core if height is None else height),
        resolution_deg=0.001,
    )


def _cid(row, col):
    return f"r{row:04d}c{col:04d}"


def _make_patches(nx, ny, levels=None, *, core=CORE, core_h=None, core_w=None,
                  halo=HALO, sigma=0.3,
                  channel_levels=None, field=None, seed=0, half_core_cell=None,
                  canvas=None, extra_level=None):
    """CellPatch grid, either from per-Cell levels or from a global field."""
    rng = np.random.default_rng(seed)
    ch = int(core if core_h is None else core_h)
    cw = int(core if core_w is None else core_w)
    if canvas is None:
        canvas = _canvas(nx, ny, core)
    patches = {}
    for row in range(ny):
        for col in range(nx):
            cid = _cid(row, col)
            cy0, cy1 = row * ch, (row + 1) * ch
            cx0, cx1 = col * cw, (col + 1) * cw
            py0, py1 = max(0, cy0 - halo), min(canvas.height, cy1 + halo)
            px0, px1 = max(0, cx0 - halo), min(canvas.width, cx1 + halo)
            yy, xx = np.mgrid[py0:py1, px0:px1]
            if field is not None:
                sci = np.asarray(field(yy, xx), dtype=np.float64)
                if sci.ndim == 2:
                    sci = np.repeat(sci[:, :, None], 3, axis=2)
                if extra_level is not None:
                    # Per-CELL additive gauge level (the unknown the offset graph
                    # must remove), independent of the continuous field.
                    sci = sci + float(extra_level(
                        row, col, 0.5 * (cy0 + cy1 - 1), 0.5 * (cx0 + cx1 - 1)
                    ))
            else:
                if channel_levels is not None:
                    lv = np.asarray(channel_levels[cid], dtype=np.float64)
                else:
                    lv = np.full(3, float(levels[cid]))
                sci = lv[None, None, :] + rng.normal(0.0, sigma, (yy.shape + (3,)))
            valid = np.ones(sci.shape, dtype=bool)
            if half_core_cell == cid:
                # Unequal gauge support: half of the CORE is invalid, halo stays
                # valid so the edge observations are still fully defined.
                ncore_w = cx1 - cx0
                valid[cy0 - py0:cy1 - py0, cx0 - px0:cx0 - px0 + ncore_w // 2, :] = False
            core_slice = (cy0 - py0, cy1 - py0, cx0 - px0, cx1 - px0)
            patches[cid] = zcf.CellPatch(
                cell_id=cid, bounds=(py0, py1, px0, px1), core_slice=core_slice,
                science=sci.astype(np.float32), valid_mask=valid,
            )
    return canvas, patches


def _offsets(graph):
    return {cid: np.asarray(graph.offsets[i], np.float64)
            for i, cid in enumerate(graph.cell_ids)}


def _assemble(canvas, nx, ny, patches, graph):
    return zcf.assemble_canvas_feathered(canvas, nx, ny, patches, _offsets(graph))


def _pairwise(offsets):
    ids = sorted(offsets)
    return {
        (a, b): offsets[b] - offsets[a]
        for i, a in enumerate(ids) for b in ids[i + 1:]
    }


# ---------------------------------------------------------------------------
# A. Anchor invariance (same graph/patches, two anchors)
# ---------------------------------------------------------------------------

def test_anchor_choice_does_not_change_science_or_pairwise_offsets():
    canvas, patches = _make_patches(3, 3, LEVELS_3X3)
    g_center = zcf.build_cell_offset_graph(canvas, 3, 3, patches)
    g_corner = zcf.build_cell_offset_graph(
        canvas, 3, 3, patches, anchor="r0000c0000"
    )
    assert g_center.anchor == "r0001c0001"
    assert g_corner.anchor == "r0000c0000"

    off_c, off_k = _offsets(g_center), _offsets(g_corner)
    # The anchor change is EXACTLY one per-channel constant: all pairwise
    # offset differences are unchanged (tight, solver-level tolerance).
    for key, delta in _pairwise(off_c).items():
        assert np.allclose(delta, _pairwise(off_k)[key], rtol=0.0, atol=1e-9), key
    # Same relative physics: weighted residuals identical.
    np.testing.assert_allclose(
        g_center.weighted_residual_rms, g_corner.weighted_residual_rms,
        rtol=0.0, atol=1e-9,
    )
    # Same disconnected policy / edge set.
    assert g_center.edges_used == g_corner.edges_used
    assert g_center.zeropoint_status == zco.ZEROPOINT_STATUS_APPLIED
    assert g_center.zeropoint_policy == zco.ZEROPOINT_POLICY_WEIGHTED

    # The assembled science and coverage are identical to float32 resolution
    # (offsets differ only by solver noise ~1e-13; float32 ulp at level 100 is
    # ~7.6e-6, so 1e-4 is a strict, justified bound).
    a_c = _assemble(canvas, 3, 3, patches, g_center)
    a_k = _assemble(canvas, 3, 3, patches, g_corner)
    np.testing.assert_allclose(a_c.science, a_k.science, rtol=0.0, atol=1e-4)
    np.testing.assert_array_equal(a_c.valid_fraction, a_k.valid_fraction)
    np.testing.assert_array_equal(a_c.owner, a_k.owner)


def test_anchor_invariance_holds_with_unequal_core_support():
    canvas, patches = _make_patches(
        2, 2, LEVELS_2X2, half_core_cell="r0001c0001",
    )
    g_center = zcf.build_cell_offset_graph(canvas, 2, 2, patches)
    g_corner = zcf.build_cell_offset_graph(
        canvas, 2, 2, patches, anchor="r0000c0001"
    )
    # The weights are NOT uniform: the half-masked Cell contributes less.
    assert g_center.zeropoint_total_weight[0] < 4 * CORE * CORE
    for key, delta in _pairwise(_offsets(g_center)).items():
        assert np.allclose(delta, _pairwise(_offsets(g_corner))[key],
                           rtol=0.0, atol=1e-9), key
    a_c = _assemble(canvas, 2, 2, patches, g_center)
    a_k = _assemble(canvas, 2, 2, patches, g_corner)
    np.testing.assert_allclose(a_c.science, a_k.science, rtol=0.0, atol=1e-4)


def test_patch_mapping_order_does_not_change_offsets_or_science():
    canvas, patches = _make_patches(2, 2, LEVELS_2X2)
    reordered = {cid: patches[cid] for cid in reversed(sorted(patches))}
    g_a = zcf.build_cell_offset_graph(canvas, 2, 2, patches)
    g_b = zcf.build_cell_offset_graph(canvas, 2, 2, reordered)
    assert g_a.cell_ids == g_b.cell_ids
    np.testing.assert_array_equal(g_a.offsets, g_b.offsets)
    a_a = _assemble(canvas, 2, 2, patches, g_a)
    a_b = _assemble(canvas, 2, 2, reordered, g_b)
    np.testing.assert_array_equal(a_a.science, a_b.science)


# ---------------------------------------------------------------------------
# B. Equivalent re-partition of the same additive field
# ---------------------------------------------------------------------------

def test_equivalent_partition_background_is_stable():
    h = w = 64
    # Linear additive field: for equal-area Cells the support-weighted mean of
    # the Cell levels equals the field at the canvas centroid (31.5, 31.5).
    def field(yy, xx):
        return 100.0 + 0.5 * xx + 0.25 * yy

    expected = 100.0 + 0.5 * 31.5 + 0.25 * 31.5
    # Per-Cell additive gauge levels (unknowns): a LINEAR function of the Cell
    # centre, zero at the canvas centroid. Its support-weighted mean is exactly 0
    # for any uniform partition, so the expected zero point stays the field value
    # at the centroid — no hard-coded assembly constant.
    def extra_level(row, col, cy, cx):
        return 3.0 * (cx - 31.5) / 32.0 + 6.0 * (cy - 31.5) / 32.0

    canvas_2x2 = _canvas(2, 2, 32, height=h, width=w)
    _, p_2x2 = _make_patches(2, 2, core=32, field=field, sigma=0.0,
                             canvas=canvas_2x2, extra_level=extra_level)
    g_2x2 = zcf.build_cell_offset_graph(canvas_2x2, 2, 2, p_2x2)
    a_2x2 = _assemble(canvas_2x2, 2, 2, p_2x2, g_2x2)

    canvas_4x1 = _canvas(4, 1, 16, height=h, width=w)
    _, p_4x1 = _make_patches(4, 1, core_h=64, core_w=16, field=field, sigma=0.0,
                             canvas=canvas_4x1, extra_level=extra_level)
    g_4x1 = zcf.build_cell_offset_graph(canvas_4x1, 4, 1, p_4x1)
    a_4x1 = _assemble(canvas_4x1, 4, 1, p_4x1, g_4x1)

    # Compare the assembled background away from the outer border band.
    roi = (slice(12, 52), slice(12, 52))
    m2 = float(np.median(np.asarray(a_2x2.science)[roi]))
    m1 = float(np.median(np.asarray(a_4x1.science)[roi]))
    assert abs(m2 - expected) < 0.2, (m2, expected)
    assert abs(m1 - expected) < 0.2, (m1, expected)
    # ... and therefore repartitioning changes the zero point by << 1 ADU.
    assert abs(m2 - m1) < 0.2, (m2, m1)
    # Per-channel gauge constants are recorded and finite.
    assert np.all(np.isfinite(g_2x2.zeropoint_constant))
    assert np.all(np.isfinite(g_4x1.zeropoint_constant))


# ---------------------------------------------------------------------------
# Gauge unit contract (pure)
# ---------------------------------------------------------------------------

def test_gauge_is_one_common_constant_per_channel():
    off = np.array([[4.0, 40.0], [10.0, 70.0], [-2.0, 10.0]])
    out, g = zco.recenter_offsets_to_zero_point(
        off, ["a", "b", "c"], weights=np.array([1.0, 2.0, 1.0]),
    )
    expected_c0 = (4.0 * 1 + 10.0 * 2 + -2.0 * 1) / 4.0
    expected_c1 = (40.0 * 1 + 70.0 * 2 + 10.0 * 1) / 4.0
    np.testing.assert_allclose(g.constant, [expected_c0, expected_c1])
    np.testing.assert_allclose(out + g.constant, off)  # one constant, both channels
    # Input never mutated.
    np.testing.assert_array_equal(off, [[4.0, 40.0], [10.0, 70.0], [-2.0, 10.0]])
    assert g.status == zco.ZEROPOINT_STATUS_APPLIED
    assert g.policy == zco.ZEROPOINT_POLICY_WEIGHTED
    np.testing.assert_array_equal(g.weighted_cells, [3, 3])


def test_gauge_zero_weight_falls_back_to_uniform_mean():
    off = np.array([[1.0], [5.0], [9.0]])
    out, g = zco.recenter_offsets_to_zero_point(
        off, ["a", "b", "c"], weights=np.zeros((3, 1)),
    )
    np.testing.assert_allclose(g.constant, [5.0])
    np.testing.assert_allclose(out.ravel(), [-4.0, 0.0, 4.0])
    assert g.status == zco.ZEROPOINT_STATUS_FALLBACK
    assert g.policy == zco.ZEROPOINT_POLICY_UNIFORM
    assert g.fallback_channels == (0,)
    assert g.reason == "zero_total_weight"


def test_gauge_huge_weights_overflow_to_uniform_fallback():
    off = np.array([[1.0], [3.0]])
    out, g = zco.recenter_offsets_to_zero_point(
        off, ["a", "b"], weights=np.array([1e308, 1e308]),
    )
    assert np.all(np.isfinite(g.constant))
    np.testing.assert_allclose(g.constant, [2.0])
    assert g.status == zco.ZEROPOINT_STATUS_FALLBACK


def test_gauge_invalid_weights_are_excluded_and_counted():
    off = np.array([[1.0], [5.0], [9.0]])
    out, g = zco.recenter_offsets_to_zero_point(
        off, ["a", "b", "c"], weights=np.array([np.nan, -3.0, 7.0]),
    )
    assert g.invalid_weight_entries == 2
    np.testing.assert_allclose(g.constant, [9.0])  # only cell c contributes
    np.testing.assert_allclose(out.ravel(), [-8.0, -4.0, 0.0])
    np.testing.assert_array_equal(g.weighted_cells, [1])


def test_gauge_zero_support_cell_is_shifted_but_does_not_weigh():
    off = np.array([[2.0], [8.0]])
    out, g = zco.recenter_offsets_to_zero_point(
        off, ["a", "b"], weights=np.array([0.0, 1.0]),
    )
    np.testing.assert_allclose(g.constant, [8.0])
    # The zero-support Cell still receives the SAME common constant.
    np.testing.assert_allclose(out.ravel(), [-6.0, 0.0])
    np.testing.assert_array_equal(g.weighted_cells, [1])


def test_gauge_non_finite_offsets_fail_closed():
    for bad in (np.nan, np.inf, -np.inf):
        off = np.array([[1.0], [bad]])
        with pytest.raises(ValueError):
            zco.recenter_offsets_to_zero_point(off, ["a", "b"])


def test_gauge_eligible_only_shifts_and_weighs_solved_component():
    off = np.array([[4.0], [10.0], [20.0]])
    out, g = zco.recenter_offsets_to_zero_point(
        off, ["a", "b", "c"], weights=np.array([1.0, 1.0, 1.0]),
        eligible=np.array([True, True, False]),
    )
    np.testing.assert_allclose(g.constant, [7.0])
    np.testing.assert_allclose(out.ravel(), [-3.0, 3.0, 20.0])
    assert g.shifted_cells == ("a", "b")
    assert g.fallback_channels == ()


def test_gauge_empty_eligible_or_bad_shapes_raise():
    off = np.ones((2, 1))
    with pytest.raises(ValueError):
        zco.recenter_offsets_to_zero_point(
            off, ["a", "b"], eligible=np.zeros(2, dtype=bool),
        )
    with pytest.raises(ValueError):
        zco.recenter_offsets_to_zero_point(off, ["a", "b"], weights=np.ones((2, 3)))
    with pytest.raises(ValueError):
        zco.recenter_offsets_to_zero_point(off, ["a", "a"])
    with pytest.raises(ValueError):
        zco.recenter_offsets_to_zero_point(np.ones((2, 2, 2)))


# ---------------------------------------------------------------------------
# Disconnected graph: pinned Cells are neither weighed nor shifted
# ---------------------------------------------------------------------------

def test_disconnected_zero_policy_keeps_pinned_cell_at_zero():
    levels = {"r0000c0000": 10.0, "r0000c0001": 20.0, "r0000c0002": 30.0}
    canvas, patches = _make_patches(3, 1, levels, core=16)
    del patches["r0000c0001"]
    graph = zcf.build_cell_offset_graph(
        canvas, 3, 1, patches, on_disconnected="zero"
    )
    assert graph.disconnected == ("r0000c0002",)
    idx = graph.cell_ids.index("r0000c0002")
    assert graph.offsets[idx, 0] == 0.0


# ---------------------------------------------------------------------------
# F. Manifest/log strictness + determinism
# ---------------------------------------------------------------------------

def test_wired_info_exposes_effective_offsets_and_strict_manifest_block():
    from zemosaic import zemosaic_zegrid_mode as zz

    canvas, patches = _make_patches(2, 2, LEVELS_2X2)
    results = {
        cid: {
            "cell_id": cid, "status": "complete",
            "patch": {
                "bounds": cp.bounds, "core_slice": cp.core_slice,
                "science": cp.science, "valid_mask": cp.valid_mask,
                "surviving": None,
            },
        }
        for cid, cp in patches.items()
    }
    _, info_a = zz._assemble_per_cell_feathered(canvas, 2, 2, results, halo_px=HALO)
    _, info_b = zz._assemble_per_cell_feathered(canvas, 2, 2, results, halo_px=HALO)
    assert info_a == info_b  # deterministic, JSON-native values

    zp = info_a["zeropoint"]
    assert zp["policy"] == zco.ZEROPOINT_POLICY_WEIGHTED
    assert zp["status"] == zco.ZEROPOINT_STATUS_APPLIED
    assert len(zp["constant"]) == 3
    assert all(np.isfinite(zp["constant"]))
    assert zp["invalid_weight_entries"] == 0
    assert zp["weights"] == "core_valid_finite_support_per_cell_per_channel"

    block = zz._cell_feather_manifest_block(info_a)
    # Strict JSON: no NaN/Infinity tokens are ever emitted.
    text = json.dumps(block, allow_nan=False)
    round_trip = json.loads(text)
    assert round_trip["zeropoint"] == zp
    assert round_trip["offsets"] == info_a["offsets"]
    # Effective offsets: applying them to the patch core flattens every Cell.
    for cid, cp in patches.items():
        off = np.asarray(round_trip["offsets"][cid], np.float64)
        y0, y1, x0, x1 = cp.core_slice
        core = np.asarray(cp.science, np.float64)[y0:y1, x0:x1] + off
        assert np.allclose(core, np.mean(core), atol=1.5), cid

    deterministic = zz._cell_feather_manifest_block(info_b)
    assert json.dumps(deterministic, sort_keys=True, allow_nan=False) == json.dumps(
        block, sort_keys=True, allow_nan=False
    )


def test_manifest_block_refuses_non_finite_offsets():
    from zemosaic import zemosaic_zegrid_mode as zz

    info = {
        "anchor": "r0000c0000", "channels": 1, "halo_px": 8,
        "edges_used": 0, "edges_built": 0, "weighted_residual_rms": [0.0],
        "disconnected": [], "policy": "zero",
        "offsets": {"r0000c0000": [float("nan")]},
    }
    with pytest.raises(ValueError):
        zz._cell_feather_manifest_block(info)


def test_manifest_block_absent_zeropoint_keeps_legacy_shape():
    from zemosaic import zemosaic_zegrid_mode as zz

    info = {
        "anchor": "r0000c0000", "channels": 1, "halo_px": 8,
        "edges_used": 0, "edges_built": 0, "weighted_residual_rms": [0.0],
        "disconnected": [], "policy": "zero", "offsets": {"r0000c0000": [0.0]},
    }
    block = zz._cell_feather_manifest_block(info)
    assert "zeropoint" not in block
    assert block["enabled"] is True
    assert json.loads(json.dumps(block, allow_nan=False))["offsets"] == info["offsets"]


# ---------------------------------------------------------------------------
# C. global mode untouched (no gauge, no feather, no new info)
# ---------------------------------------------------------------------------

def test_global_mode_never_returns_feather_or_zeropoint_info():
    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid import mosaic as zmosaic

    canvas = _canvas(2, 2)
    cores = {
        _cid(r, c): {
            "science_core": np.full((CORE, CORE, 3), float(r * 2 + c), np.float32),
            "surviving_sample_count_core": np.ones((CORE, CORE, 3), np.int32),
            "n_eff_support_core": np.ones((CORE, CORE), np.float64),
            "support_w1_core": np.ones((CORE, CORE), np.float64),
            "support_w2_core": np.ones((CORE, CORE), np.float64),
            "valid_mask_core": np.ones((CORE, CORE, 3), bool),
        }
        for r in range(2) for c in range(2)
    }
    ref = zmosaic.assemble_canvas(canvas, 2, 2, cores)
    got, info = zz._assemble_cells(canvas, 2, 2, "global", cores, {})
    assert info is None
    np.testing.assert_array_equal(ref.science, got.science)
    np.testing.assert_array_equal(ref.stack_depth, got.stack_depth)
    np.testing.assert_array_equal(ref.valid_fraction, got.valid_fraction)
