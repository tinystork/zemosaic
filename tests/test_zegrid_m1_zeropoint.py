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


def _results(patches):
    """Wrap CellPatches as per-cell worker results (wired path)."""
    return {
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
# C5 — repartition under the SAME non-uniform physical support + cost witness
# ---------------------------------------------------------------------------

def _physical_support_and_field():
    """One shared 64x64 support mask (non-uniform) and one shared field."""
    h = w = 64
    support = np.ones((h, w), dtype=bool)
    support[:3, :] = False
    support[-3:, :] = False
    support[:, :3] = False
    support[:, -3:] = False
    support[10:18, 18:30] = False
    support[40:52, 34:44] = False
    support[5:25, 50:54] = False

    def field(yy, xx):
        return 100.0 + 0.7 * xx - 0.3 * yy + 0.004 * xx * yy

    return support, field


def _support_partition_patches(canvas, nx, ny, core_h, core_w, support, field):
    """Cells whose level is the support-weighted mean of ONE underlying field.

    ``science = field + L_cell`` with ``L_cell = mean(field | support & core)``.
    No arranged level constants: the levels follow from the SAME field and the
    SAME physical support in every partition.
    """
    halo = HALO
    patches = {}
    levels = {}
    for row in range(ny):
        for col in range(nx):
            cid = _cid(row, col)
            cy0, cy1 = row * core_h, (row + 1) * core_h
            cx0, cx1 = col * core_w, (col + 1) * core_w
            core_support = support[cy0:cy1, cx0:cx1]
            if not core_support.any():
                raise AssertionError(f"empty physical core support for {cid}")
            yy_c, xx_c = np.nonzero(core_support)
            level = float(np.mean(field(yy_c + cy0, xx_c + cx0)))
            levels[cid] = level
            py0, py1 = max(0, cy0 - halo), min(canvas.height, cy1 + halo)
            px0, px1 = max(0, cx0 - halo), min(canvas.width, cx1 + halo)
            yy, xx = np.mgrid[py0:py1, px0:px1]
            plane = field(yy, xx) + level
            valid = support[py0:py1, px0:px1]
            sci = np.where(valid[:, :, None], plane[:, :, None], np.nan)
            sci = np.repeat(sci, 3, axis=2)          # test builder only
            valid3 = np.repeat(valid[:, :, None], 3, axis=2)
            patches[cid] = zcf.CellPatch(
                cell_id=cid, bounds=(py0, py1, px0, px1),
                core_slice=(cy0 - py0, cy1 - py0, cx0 - px0, cx1 - px0),
                science=sci.astype(np.float32), valid_mask=valid3,
            )
    return patches, levels


def _raw_anchor_pinned_offsets(graph):
    """Pre-M1 offsets: undo the production gauge constant (no reimplementation)."""
    return {
        cid: np.asarray(graph.offsets[i], np.float64)
        + np.asarray(graph.zeropoint_constant, np.float64)
        for i, cid in enumerate(graph.cell_ids)
    }


def test_repartition_with_shared_nonuniform_support_closes_c5():
    h = w = 64
    support, field = _physical_support_and_field()
    yy, xx = np.mgrid[0:h, 0:w]
    F = field(yy, xx)
    global_mean = float(F[support].mean())
    expected = F + global_mean

    canvas_2x2 = _canvas(2, 2, 32, height=h, width=w)
    p_2x2, levels_2x2 = _support_partition_patches(
        canvas_2x2, 2, 2, 32, 32, support, field)
    canvas_4x1 = _canvas(4, 1, 16, height=h, width=w)
    p_4x1, levels_4x1 = _support_partition_patches(
        canvas_4x1, 4, 1, 64, 16, support, field)

    w_2x2 = zcf.cell_core_support(p_2x2, sorted(p_2x2), 3)
    w_4x1 = zcf.cell_core_support(p_4x1, sorted(p_4x1), 3)
    # Same channels everywhere.
    for w in (w_2x2, w_4x1):
        np.testing.assert_array_equal(w[:, 0], w[:, 1])
        np.testing.assert_array_equal(w[:, 0], w[:, 2])
    # Total PHYSICAL support is coherent across the two partitions.
    assert int(w_2x2[:, 0].sum()) == int(support.sum())
    assert int(w_4x1[:, 0].sum()) == int(support.sum())
    # Cell weights are genuinely unequal, and the partitions differ.
    assert len(np.unique(w_2x2[:, 0])) > 1
    assert len(np.unique(w_4x1[:, 0])) > 1
    assert not np.array_equal(np.sort(w_2x2[:, 0]), np.sort(w_4x1[:, 0]))

    g_2x2 = zcf.build_cell_offset_graph(canvas_2x2, 2, 2, p_2x2)
    g_4x1 = zcf.build_cell_offset_graph(canvas_4x1, 4, 1, p_4x1)
    assert not g_2x2.disconnected and not g_4x1.disconnected
    np.testing.assert_allclose(g_2x2.zeropoint_total_weight, w_2x2.sum(axis=0))
    np.testing.assert_allclose(g_4x1.zeropoint_total_weight, w_4x1.sum(axis=0))

    a_2x2 = _assemble(canvas_2x2, 2, 2, p_2x2, g_2x2)
    a_4x1 = _assemble(canvas_4x1, 4, 1, p_4x1, g_4x1)
    # The assembled science is the SAME field plus the SAME global zero.
    for a in (a_2x2, a_4x1):
        np.testing.assert_allclose(
            np.asarray(a.science)[support, 0], expected[support], atol=1e-3
        )
    np.testing.assert_allclose(
        np.asarray(a_2x2.science), np.asarray(a_4x1.science),
        rtol=0.0, atol=1e-4,
    )

    # Anchor change on EACH connected graph: science and relative physics fixed.
    g_2x2_b = zcf.build_cell_offset_graph(
        canvas_2x2, 2, 2, p_2x2, anchor="r0001c0001")
    g_4x1_b = zcf.build_cell_offset_graph(
        canvas_4x1, 4, 1, p_4x1, anchor="r0000c0003")
    for ga, gb in ((g_2x2, g_2x2_b), (g_4x1, g_4x1_b)):
        assert ga.anchor != gb.anchor
        np.testing.assert_allclose(
            ga.weighted_residual_rms, gb.weighted_residual_rms,
            rtol=0.0, atol=1e-9,
        )
        for key, delta in _pairwise(_offsets(ga)).items():
            assert np.allclose(delta, _pairwise(_offsets(gb))[key],
                               rtol=0.0, atol=1e-9), key
        a_a = _assemble(canvas_2x2 if ga is g_2x2 else canvas_4x1,
                        2 if ga is g_2x2 else 4,
                        2 if ga is g_2x2 else 1,
                        p_2x2 if ga is g_2x2 else p_4x1, ga)
        a_b = _assemble(canvas_2x2 if gb is g_2x2_b else canvas_4x1,
                        2 if gb is g_2x2_b else 4,
                        2 if gb is g_2x2_b else 1,
                        p_2x2 if gb is g_2x2_b else p_4x1, gb)
        np.testing.assert_allclose(a_a.science, a_b.science, rtol=0.0, atol=1e-4)

    # Gauge is a pure constant: pairwise offset differences match the raw ones.
    for graph in (g_2x2, g_4x1):
        raw = _raw_anchor_pinned_offsets(graph)
        eff = _offsets(graph)
        for key, delta in _pairwise(raw).items():
            assert np.allclose(delta, _pairwise(eff)[key], rtol=0.0, atol=1e-9)

    # NEGATIVE CONTROL: without the M1 recentering, the same partitions differ.
    raw_2x2 = _raw_anchor_pinned_offsets(g_2x2)
    raw_4x1 = _raw_anchor_pinned_offsets(g_4x1)
    a_2x2_raw = zcf.assemble_canvas_feathered(
        canvas_2x2, 2, 2, p_2x2, raw_2x2, halo_px=HALO)
    a_4x1_raw = zcf.assemble_canvas_feathered(
        canvas_4x1, 4, 1, p_4x1, raw_4x1, halo_px=HALO)
    raw_partition_gap = float(np.nanmax(np.abs(
        np.asarray(a_2x2_raw.science) - np.asarray(a_4x1_raw.science)
    )))
    assert raw_partition_gap > 1.0, raw_partition_gap
    raw_2x2_anchor_b = zcf.assemble_canvas_feathered(
        canvas_2x2, 2, 2, p_2x2,
        _raw_anchor_pinned_offsets(g_2x2_b), halo_px=HALO)
    raw_anchor_gap = float(np.nanmax(np.abs(
        np.asarray(a_2x2_raw.science) - np.asarray(raw_2x2_anchor_b.science)
    )))
    assert raw_anchor_gap > 1.0, raw_anchor_gap


# ---------------------------------------------------------------------------
# Synthetic cost witness (report-only timing; no flaky threshold)
# ---------------------------------------------------------------------------

def test_support_scan_cost_witness(monkeypatch):
    import time

    h = w = 1024
    rng = np.random.default_rng(0)
    sci = rng.normal(100.0, 1.0, (h, w, 3)).astype(np.float32)
    val = np.ones((h, w), dtype=bool)
    patches = {"c": zcf.CellPatch("c", (0, h, 0, w), (0, h, 0, w), sci, val)}

    real_isfinite = np.isfinite
    shapes = []

    def _spy(x, *a, **k):
        shapes.append(tuple(np.shape(x)))
        return real_isfinite(x, *a, **k)

    monkeypatch.setattr(zcf.np, "isfinite", _spy)
    t0 = time.perf_counter()
    got = zcf.cell_core_support(patches, ["c"], 3)
    elapsed = time.perf_counter() - t0

    cap = int(zcf.SUPPORT_SCAN_MAX_ELEMENTS)
    assert got.shape == (1, 3)
    assert np.all(got == float(h * w))
    assert shapes and all(len(shape) == 2 for shape in shapes)
    assert max(int(np.prod(shape)) for shape in shapes) <= cap
    bw = min(w, cap)
    bh = max(1, cap // bw)
    expected_calls = ((h + bh - 1) // bh) * ((w + bw - 1) // bw) * 3
    assert len(shapes) == expected_calls

    peak_bound = 3 * cap  # isfinite + and-mask + optional bool tile conversion
    print(
        "SUPPORT_SCAN_COST "
        f"dims={h}x{w} elements={h*w} channels=3 calls={len(shapes)} "
        f"tile_shape_max={max(shapes, key=lambda s: int(np.prod(s)))} "
        f"max_tile_elements={max(int(np.prod(s)) for s in shapes)} "
        f"cap={cap} peak_bound_bytes={peak_bound} wall_s={elapsed:.6f}"
    )


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
    assert g.fallback_reasons == ("zero_total_weight",)


def test_gauge_huge_weights_overflow_to_uniform_fallback():
    off = np.array([[1.0], [3.0]])
    out, g = zco.recenter_offsets_to_zero_point(
        off, ["a", "b"], weights=np.array([1e308, 1e308]),
    )
    assert np.all(np.isfinite(g.constant))
    np.testing.assert_allclose(g.constant, [2.0])
    assert g.status == zco.ZEROPOINT_STATUS_FALLBACK
    # Overflow is NOT reported as a zero weight: the reason is exact.
    assert g.fallback_reasons == ("nonfinite_total_weight",)


def test_gauge_mixed_fallback_reasons_per_channel():
    """Channel 0 zero weight, channel 1 overflow: reasons stay per-channel."""
    off = np.array([[1.0, 10.0], [3.0, 30.0]])
    weights = np.array([[0.0, 1e308], [0.0, 1e308]])
    out, g = zco.recenter_offsets_to_zero_point(off, ["a", "b"], weights=weights)
    assert g.fallback_channels == (0, 1)
    assert g.fallback_reasons == ("zero_total_weight", "nonfinite_total_weight")
    assert g.status == zco.ZEROPOINT_STATUS_FALLBACK
    np.testing.assert_allclose(g.constant, [2.0, 20.0])
    np.testing.assert_allclose(out, [[-1.0, -10.0], [1.0, 10.0]])


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
# F1 — bounded, copy-free support scan (memory contract)
# ---------------------------------------------------------------------------

class _GuardedPlane:
    """Array-like that refuses float64 conversion / full-array copy requests.

    ``np.asarray(plane)`` (dtype=None, copy=None) yields the SOURCE array view;
    any ``np.asarray(..., dtype=float64)`` or whole-array copy raises, which is
    exactly the regression this test locks out.
    """

    def __init__(self, arr):
        self._arr = arr
        self.calls = []

    def __array__(self, dtype=None, copy=None):
        self.calls.append((dtype, copy))
        if dtype is not None and np.dtype(dtype) == np.dtype(np.float64):
            raise AssertionError("full-patch float64 conversion requested")
        if copy:
            raise AssertionError("full-patch copy requested")
        return self._arr


def _plain_support(sci, val, core_slice, channels):
    """Independent reference count (small synthetic arrays only)."""
    y0, y1, x0, x1 = core_slice
    s = np.asarray(sci, dtype=np.float64)
    v = np.asarray(val, dtype=bool)
    if s.ndim == 2:
        s = s[:, :, None]
    if v.ndim == 2:
        v = v[:, :, None]
    if s.shape[2] == 1:
        s = np.repeat(s, channels, axis=2)
    if v.shape[2] == 1:
        v = np.repeat(v, channels, axis=2)
    core = v[y0:y1, x0:x1] & np.isfinite(s[y0:y1, x0:x1])
    return np.count_nonzero(core, axis=(0, 1)).astype(np.float64)


def test_cell_core_support_never_repeats_or_converts_whole_patch(monkeypatch):
    canvas, patches = _make_patches(2, 2, LEVELS_2X2)
    cid = "r0000c0000"
    cp = patches[cid]
    guarded_sci = _GuardedPlane(cp.science)
    guarded_val = _GuardedPlane(cp.valid_mask)
    guarded = zcf.CellPatch(
        cid, cp.bounds, cp.core_slice, guarded_sci, guarded_val, cp.surviving,
    )
    patches_guarded = dict(patches)
    patches_guarded[cid] = guarded

    def _no_repeat(*_a, **_k):
        raise AssertionError("np.repeat is forbidden on the support path")

    monkeypatch.setattr(zcf.np, "repeat", _no_repeat)
    got = zcf.cell_core_support(patches_guarded, sorted(patches_guarded), 3)
    expected = np.vstack([
        _plain_support(cp_.science, cp_.valid_mask, cp_.core_slice, 3)
        for cp_ in (patches[k] for k in sorted(patches))
    ])
    np.testing.assert_array_equal(got, expected)
    # The guarded planes were materialised as views only (no dtype/copy ask).
    assert guarded_sci.calls and all(dt is None for dt, _c in guarded_sci.calls)
    assert guarded_val.calls and all(dt is None for dt, _c in guarded_val.calls)


def test_cell_core_support_tiles_isfinite_and_never_builds_3d(monkeypatch):
    canvas, patches = _make_patches(1, 1, {"r0000c0000": 100.0}, core=10)
    cp = patches["r0000c0000"]
    # Force tiling with a small documented cap and record every isfinite call.
    monkeypatch.setattr(zcf, "SUPPORT_SCAN_MAX_ELEMENTS", 32)
    real_isfinite = np.isfinite
    shapes = []

    def _spy(x, *a, **k):
        shapes.append(tuple(np.shape(x)))
        return real_isfinite(x, *a, **k)

    monkeypatch.setattr(zcf.np, "isfinite", _spy)
    got = zcf.cell_core_support(patches, ["r0000c0000"], 3)
    assert shapes, "the support scan must actually run isfinite"
    for shape in shapes:
        assert len(shape) == 2, shape          # never a 3-D full-patch bool
        assert int(np.prod(shape)) <= 32, shape
    assert got.shape == (1, 3)
    assert np.all(got == float(10 * 10))


def test_cell_core_support_broadcast_parity_and_nonfinite():
    h = w = 12
    rng = np.random.default_rng(7)
    mono = rng.normal(100.0, 1.0, (h, w)).astype(np.float32)
    mono[0, 0] = np.nan
    mono[1, 1] = np.inf
    valid2 = np.ones((h, w), bool)
    valid2[2, 2] = False
    sci3 = np.repeat(mono[:, :, None], 3, axis=2)
    valid3 = np.repeat(valid2[:, :, None], 3, axis=2)
    core = (0, h, 0, w)

    variants = {
        "mono2d": (mono, valid2),
        "mono3d": (mono[:, :, None], valid2),
        "rgb_2dvalid": (sci3, valid2),
        "rgb_3dvalid": (sci3, valid3),
    }
    weights = {}
    for name, (s, v) in variants.items():
        patches = {"c": zcf.CellPatch("c", (0, h, 0, w), core, s, v)}
        weights[name] = zcf.cell_core_support(patches, ["c"], 3)[0]
    expected = _plain_support(sci3, valid3, core, 3)
    for name, got in weights.items():
        np.testing.assert_array_equal(got, expected, err_msg=str(name))
    # NaN/Inf and invalid pixels are excluded, not fabricated.
    assert expected[0] == h * w - 3
    # Inputs untouched.
    assert np.isnan(mono[0, 0]) and np.isinf(mono[1, 1])


def test_cell_core_support_noncontiguous_planes_are_correct():
    h = w = 16
    base_sci = np.zeros((h, w * 2, 3), np.float32)
    base_val = np.ones((h, w * 2, 3), bool)
    base_sci[:, 0::2, :] = 5.0
    base_sci[3, 0, 0] = np.nan
    base_val[:, 0::2, :] = True
    base_val[4, 0, 1] = False
    sci = base_sci[:, 0::2, :]   # non-contiguous view
    val = base_val[:, 0::2, :]   # non-contiguous view
    assert not sci.flags["C_CONTIGUOUS"]
    patches = {"c": zcf.CellPatch("c", (0, h, 0, w), (0, h, 0, w), sci, val)}
    got = zcf.cell_core_support(patches, ["c"], 3)[0]
    expected = _plain_support(sci, val, (0, h, 0, w), 3)
    np.testing.assert_array_equal(got, expected)
    assert expected[0] == h * w - 1
    assert expected[1] == h * w - 1


def test_cell_core_support_rejects_incoherent_shapes():
    good = np.ones((4, 4, 3), np.float32)
    val = np.ones((4, 4), bool)
    core = (0, 4, 0, 4)
    with pytest.raises(ValueError):
        zcf.cell_core_support(
            {"c": zcf.CellPatch("c", (0, 4, 0, 4), core,
                                np.ones((4, 4, 2, 2), np.float32), val)}, ["c"], 3)
    with pytest.raises(ValueError):
        zcf.cell_core_support(
            {"c": zcf.CellPatch("c", (0, 4, 0, 4), core,
                                good, np.ones((4, 5), bool))}, ["c"], 3)
    with pytest.raises(ValueError):
        zcf.cell_core_support(
            {"c": zcf.CellPatch("c", (0, 4, 0, 4), core,
                                np.ones((4, 4, 2), np.float32), val)}, ["c"], 3)
    with pytest.raises(ValueError):
        zcf.cell_core_support(
            {"c": zcf.CellPatch("c", (0, 4, 0, 4), (0, 9, 0, 4),
                                good, val)}, ["c"], 3)


def test_cell_core_support_mono_science_uses_each_mask_channel():
    """F3 exact repro: 2-D science + divergent RGB mask -> [16, 0, 8]."""
    sci = np.ones((4, 4), np.float32)
    val = np.zeros((4, 4, 3), dtype=bool)
    val[:, :, 0] = True       # 16 valid
    val[0:2, :, 2] = True     # 8 valid
    patches = {"c": zcf.CellPatch("c", (0, 4, 0, 4), (0, 4, 0, 4), sci, val)}
    got = zcf.cell_core_support(patches, ["c"], 3)
    np.testing.assert_array_equal(got, [[16.0, 0.0, 8.0]])


def test_cell_core_support_exhaustive_science_mask_matrix():
    """3x3 matrix: science {2-D, 1-ch, C} x mask {2-D, 1-ch, C}.

    Divergent RGB masks + per-channel NaN/Inf + non-contiguous views; every
    combination must match the independent per-channel reference.
    """
    h = w = 6
    base_sci = np.ones((h, 2 * w, 3), np.float32)
    base_sci[0, 0, 0] = np.nan        # ch0 lost
    base_sci[1, 2, 1] = np.inf        # ch1 lost (mask is already all-False)
    base_sci[2, 4, 2] = -np.inf       # ch2 lost
    base_val = np.ones((h, 2 * w, 3), dtype=bool)
    base_val[:, :, 1] = False         # ch1 fully invalid
    base_val[3, 2, 2] = False         # ch2 extra loss
    base_val[4, 6, 0] = False         # ch0 extra loss
    base_val[5, 8, 2] = False         # ch2 extra loss (diverges from ch0)

    sci3 = base_sci[:, 0::2, :]       # non-contiguous views
    val3 = base_val[:, 0::2, :]
    sci1 = sci3[:, :, 0:1]
    val1 = val3[:, :, 0:1]
    sci2 = sci3[:, :, 0]
    val2 = val3[:, :, 0]
    assert not sci3.flags["C_CONTIGUOUS"] and not val3.flags["C_CONTIGUOUS"]

    core = (0, h, 0, w)
    sciences = {"2d": sci2, "1ch": sci1, "3ch": sci3}
    masks = {"2d": val2, "1ch": val1, "3ch": val3}
    seen = {}
    for sname, s in sciences.items():
        for vname, v in masks.items():
            patches = {"c": zcf.CellPatch("c", (0, h, 0, w), core, s, v)}
            got = zcf.cell_core_support(patches, ["c"], 3)[0]
            expected = _plain_support(s, v, core, 3)
            np.testing.assert_array_equal(
                got, expected, err_msg=f"{sname}/{vname}"
            )
            seen[(sname, vname)] = tuple(got)
    # A 2-D / 1-channel mask is logically broadcast: all channels agree.
    for sname in sciences:
        for vname in ("2d", "1ch"):
            row = seen[(sname, vname)]
            assert row[0] == row[1] == row[2], (sname, vname, row)
    # A 3-channel mask is honored per channel: ch1 is empty, ch0 != ch2.
    for sname in sciences:
        row = seen[(sname, "3ch")]
        assert row[1] == 0.0, (sname, row)
        assert row[0] != row[2], (sname, row)


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


def _two_component_patches():
    """2x3 grid minus its middle column -> two vertical 2-Cell components."""
    levels = {
        "r0000c0000": 10.0, "r0000c0001": 20.0, "r0000c0002": 100.0,
        "r0001c0000": 14.0, "r0001c0001": 24.0, "r0001c0002": 200.0,
    }
    canvas, patches = _make_patches(3, 2, levels, core=16)
    del patches["r0000c0001"]
    del patches["r0001c0001"]
    return canvas, patches


def test_disconnected_provenance_is_component_scoped_and_honest():
    from zemosaic import zemosaic_zegrid_mode as zz

    canvas, patches = _two_component_patches()
    g_left = zcf.build_cell_offset_graph(
        canvas, 3, 2, patches, anchor="r0000c0000", on_disconnected="zero"
    )
    g_right = zcf.build_cell_offset_graph(
        canvas, 3, 2, patches, anchor="r0000c0002", on_disconnected="zero"
    )
    assert g_left.disconnected and g_right.disconnected
    assert (not bool(g_left.disconnected)) is False
    assert zz._anchor_invariant_scope(g_left.disconnected) == (
        zz.ANCHOR_INVARIANT_SCOPE_COMPONENT
    )
    # Inter-component anchor change really changes the science: the global
    # invariant claim would be a lie, and must not be published.
    a_left = _assemble(canvas, 3, 2, patches, g_left)
    a_right = _assemble(canvas, 3, 2, patches, g_right)
    assert not np.allclose(
        a_left.science, a_right.science, rtol=0.0, atol=1.0, equal_nan=True
    )

    _, info = zz._assemble_per_cell_feathered(
        canvas, 3, 2, _results(patches), halo_px=HALO
    )
    zp = info["zeropoint"]
    assert zp["anchor_invariant"] is False
    assert zp["anchor_invariant_scope"] == zz.ANCHOR_INVARIANT_SCOPE_COMPONENT
    block = zz._cell_feather_manifest_block(info)
    assert block["zeropoint"]["anchor_invariant"] is False
    assert block["zeropoint"]["anchor_invariant_scope"] == (
        zz.ANCHOR_INVARIANT_SCOPE_COMPONENT
    )
    assert "anchor_connected_component_only" in block["note"]
    assert "whole graph" not in block["note"]
    # Strict JSON + deterministic.
    text = json.dumps(block, sort_keys=True, allow_nan=False)
    again = zz._cell_feather_manifest_block(info)
    assert json.dumps(again, sort_keys=True, allow_nan=False) == text


def test_connected_provenance_claims_whole_graph():
    from zemosaic import zemosaic_zegrid_mode as zz

    canvas, patches = _make_patches(2, 2, LEVELS_2X2)
    _, info = zz._assemble_per_cell_feathered(
        canvas, 2, 2, _results(patches), halo_px=HALO
    )
    zp = info["zeropoint"]
    assert zp["anchor_invariant"] is True
    assert zp["anchor_invariant_scope"] == zz.ANCHOR_INVARIANT_SCOPE_WHOLE
    block = zz._cell_feather_manifest_block(info)
    assert block["zeropoint"]["anchor_invariant_scope"] == "whole_graph"
    assert "whole_graph" in block["note"]
    assert "component" not in block["note"]
    json.dumps(block, allow_nan=False)


def test_manifest_block_validates_channel_diagnostics_and_scope():
    from zemosaic import zemosaic_zegrid_mode as zz

    def _info(channels=3, reasons=(None, None, None), fb_channels=(),
              invariant=True, scope="whole_graph"):
        return {
            "anchor": "r0000c0000", "channels": channels, "halo_px": 8,
            "edges_used": 0, "edges_built": 0,
            "weighted_residual_rms": [0.0] * channels,
            "disconnected": [], "policy": "zero",
            "offsets": {"r0000c0000": [0.0] * channels},
            "zeropoint": {
                "policy": zco.ZEROPOINT_POLICY_WEIGHTED,
                "status": zco.ZEROPOINT_STATUS_APPLIED,
                "constant": [0.0] * channels,
                "total_weight": [1.0] * channels,
                "weighted_cells": [1] * channels,
                "invalid_weight_entries": 0,
                "fallback_channels": list(fb_channels),
                "fallback_reasons": list(reasons),
                "anchor_invariant": invariant,
                "anchor_invariant_scope": scope,
                "weights": "core_valid_finite_support_per_cell_per_channel",
            },
        }

    zz._cell_feather_manifest_block(_info())  # valid
    with pytest.raises(ValueError):
        zz._cell_feather_manifest_block(
            _info(channels=3, reasons=(None, None))
        )
    with pytest.raises(ValueError):
        zz._cell_feather_manifest_block(
            _info(channels=3, reasons=(None, None, None), fb_channels=(0,))
        )
    with pytest.raises(ValueError):
        zz._cell_feather_manifest_block(
            _info(channels=3, reasons=("zero_total_weight", None, None),
                  fb_channels=(0, 1))
        )
    with pytest.raises(ValueError):
        zz._cell_feather_manifest_block(
            _info(channels=1, reasons=(None,), invariant=False, scope="whole_graph")
        )


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
    assert zp["fallback_channels"] == []
    assert zp["fallback_reasons"] == [None, None, None]
    assert zp["anchor_invariant"] is True
    assert zp["anchor_invariant_scope"] == zz.ANCHOR_INVARIANT_SCOPE_WHOLE
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


def test_manifest_block_mixed_fallback_reasons_is_strict_and_exact():
    from zemosaic import zemosaic_zegrid_mode as zz

    info = {
        "anchor": "r0000c0000", "channels": 2, "halo_px": 8,
        "edges_used": 1, "edges_built": 1, "weighted_residual_rms": [0.0, 0.0],
        "disconnected": [], "policy": "zero",
        "offsets": {"r0000c0000": [0.0, 0.0], "r0000c0001": [1.0, 1.0]},
        "zeropoint": {
            "policy": zco.ZEROPOINT_POLICY_UNIFORM,
            "status": zco.ZEROPOINT_STATUS_FALLBACK,
            "constant": [0.5, 0.5],
            "total_weight": [2.0, 2.0],
            "weighted_cells": [2, 2],
            "invalid_weight_entries": 0,
            "fallback_channels": [0, 1],
            "fallback_reasons": ["zero_total_weight", "nonfinite_total_weight"],
            "anchor_invariant": True,
            "weights": "core_valid_finite_support_per_cell_per_channel",
        },
    }
    block = zz._cell_feather_manifest_block(info)
    text = json.dumps(block, allow_nan=False)
    assert json.loads(text)["zeropoint"]["fallback_reasons"] == [
        "zero_total_weight", "nonfinite_total_weight"
    ]


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
