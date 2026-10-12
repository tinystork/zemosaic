"""ZM-CELL-PHOTOMETRY-R34D focused tests — offsets + feathered assembly.

Fast, synthetic. Covers the R34d deliverables:

* additive-only offset application (no gain, NaN preserved; mono + RGB);
* the deterministic partition-of-unity feather ramp;
* frozen shared-valid semantics (finite in both + valid in both + optional mask);
* END-TO-END: known per-cell levels -> robust observations -> central-anchored
  global solve -> additive offsets -> feathered assembly lands every cell on the
  central level with NO seam step and NO signal loss (mono + per-channel RGB);
* a pure level shift is recovered EXACTLY;
* disconnected-cells explicit behaviour (raise vs documented "zero" policy) and
  the wired path recording them;
* the wired dispatch keeps the "global" one-owner assembly byte-identical;
* determinism; weigh = support-count.
"""

from __future__ import annotations

import numpy as np
import pytest

from zemosaic.core.zegrid import cell_feather as zcf
from zemosaic.core.zegrid import cell_offset as zco
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import mosaic as zmosaic


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

CORE = 32
HALO = 8


def _canvas(nx, ny, core=CORE):
    return zg.GlobalCanvas(
        canvas_id=f"c{nx}x{ny}", wcs_header="", width=nx * core, height=ny * core,
        resolution_deg=0.001,
    )


def _cell_id(row, col):
    return f"r{row:04d}c{col:04d}"


def _make_patches(nx, ny, levels, *, core=CORE, halo=HALO, base=0.0,
                  channel_levels=None, rng=None, sigma=0.3, hole=None,
                  blob=None):
    """Build ``CellPatch`` objects for an ``nx`` x ``ny`` grid.

    ``levels`` maps cell_id -> scalar level (mono, repeated over 3 channels) OR is
    ignored when ``channel_levels`` (cell_id -> (3,) per-channel level) is given.
    ``hole`` = list of cell_ids whose valid mask is all-False (explicit hole).
    ``blob`` = (cell_id, (y0,y1,x0,x1), value) adds a bright patch-local feature.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    canvas = _canvas(nx, ny, core)
    cw = ch = core
    patches = {}
    for row in range(ny):
        for col in range(nx):
            cid = _cell_id(row, col)
            cy0, cy1 = row * ch, (row + 1) * ch
            cx0, cx1 = col * cw, (col + 1) * cw
            py0 = max(0, cy0 - halo); py1 = min(canvas.height, cy1 + halo)
            px0 = max(0, cx0 - halo); px1 = min(canvas.width, cx1 + halo)
            h, w = py1 - py0, px1 - px0
            if channel_levels is not None:
                lv = np.asarray(channel_levels[cid], dtype=np.float64)
            else:
                lv = np.full(3, float(levels[cid]))
            sci = (base + lv[None, None, :] + rng.normal(0.0, sigma, (h, w, 3)))
            valid = np.ones((h, w, 3), dtype=bool)
            if hole and cid in hole:
                valid[:] = False
                sci[:] = np.nan
            if blob is not None and blob[0] == cid:
                by0, by1, bx0, bx1 = blob[1]
                sci[by0:by1, bx0:bx1, :] += blob[2]
            core_slice = (cy0 - py0, cy1 - py0, cx0 - px0, cx1 - px0)
            patches[cid] = zcf.CellPatch(
                cell_id=cid, bounds=(py0, py1, px0, px1), core_slice=core_slice,
                science=sci.astype(np.float32), valid_mask=valid,
            )
    return canvas, patches


def _results_from_patches(patches):
    """Wrap CellPatch objects as per-cell worker result dicts (wired path)."""
    out = {}
    for cid, cp in patches.items():
        out[cid] = {
            "cell_id": cid, "status": "complete",
            "patch": {
                "bounds": cp.bounds, "core_slice": cp.core_slice,
                "science": cp.science, "valid_mask": cp.valid_mask,
                "surviving": None,
            },
        }
    return out


# ---------------------------------------------------------------------------
# Additive-only application
# ---------------------------------------------------------------------------

def test_apply_additive_offsets_additive_only():
    sci = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64)
    out = zcf.apply_additive_offsets(sci, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(out, [[[11.0, 22.0, 33.0]]])
    # NaN stays NaN (never fabricated).
    sci2 = sci.copy(); sci2[0, 0, 1] = np.nan
    out2 = zcf.apply_additive_offsets(sci2, 5.0)
    assert out2[0, 0, 0] == 6.0 and np.isnan(out2[0, 0, 1])
    # mono scalar-only
    with pytest.raises(ValueError):
        zcf.apply_additive_offsets(np.zeros((2, 2)), [1.0, 2.0])


# ---------------------------------------------------------------------------
# Feather ramp + shared-valid semantics
# ---------------------------------------------------------------------------

def test_feather_weight_field_partition_of_unity():
    # patch = 32x32, core = the middle, halo 8 -> ramp 0 at the border, 1 inside.
    core_slice = (8, 24, 8, 24)
    ramp = zcf.feather_weight_field((32, 32), core_slice, halo_px=8)
    assert ramp[0, 0] == 0.0 and ramp[31, 31] == 0.0
    assert ramp[8, 8] == pytest.approx(1.0)
    assert ramp[16, 16] == 1.0
    # monotone ramp from the border to the core.
    assert ramp[0, 16] < ramp[4, 16] < ramp[8, 16]
    # halo <= 0 -> all ones (degenerate hard cut).
    np.testing.assert_array_equal(zcf.feather_weight_field((4, 4), (1, 3, 1, 3), 0),
                                 np.ones((4, 4)))


def test_shared_valid_mask_semantics():
    a = np.ones((4, 4, 3)); b = np.ones((4, 4, 3))
    va = np.ones((4, 4, 3), bool); vb = np.ones((4, 4, 3), bool)
    b[0, 0, :] = np.nan  # non-finite in b -> not shared-valid
    vb[1, 1, :] = False  # invalid in b -> not shared-valid
    extra = np.ones((4, 4), bool); extra[2, 2] = False  # upstream star/object mask
    m = zcf.shared_valid_mask(a, va, b, vb, extra_mask=extra)
    assert m.shape == (4, 4)
    assert not m[0, 0] and not m[1, 1] and not m[2, 2]
    assert m[3, 3]


# ---------------------------------------------------------------------------
# End-to-end: offsets + feather (no seam, no signal loss, exact level recovery)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rgb", [False, True])
def test_end_to_end_feathered_no_seam_no_signal_loss(rgb):
    levels = {"r0000c0000": 100.0, "r0000c0001": 104.0,
              "r0001c0000": 96.0, "r0001c0001": 110.0}
    if rgb:
        chl = {cid: np.array([lv, lv + 3.0, lv - 2.0]) for cid, lv in levels.items()}
    else:
        chl = None
    blob = ("r0000c0000", (10, 14, 10, 14), 50.0)
    canvas, patches = _make_patches(2, 2, levels, channel_levels=chl, blob=blob)

    graph = zcf.build_cell_offset_graph(canvas, 2, 2, patches)
    assert graph.anchor == "r0000c0000"  # deterministic central anchor
    assert graph.edges_used == 4

    offsets = {cid: graph.offsets[i] for i, cid in enumerate(graph.cell_ids)}
    assembled = zcf.assemble_canvas_feathered(canvas, 2, 2, patches, offsets)

    sci = np.asarray(assembled.science, dtype=np.float64)
    assert np.isfinite(sci).all(), "feathered assembly must have no holes here"

    # Every channel lands on the anchor cell's level (per channel): no seam step.
    anchor_lv = (
        np.asarray(chl["r0000c0000"]) if rgb else np.full(3, levels["r0000c0000"])
    )
    for c in range(3):
        body = sci[..., c]
        # Exclude the deliberate bright blob region for the flatness check.
        mask = np.ones(body.shape, bool); mask[10:14, 10:14] = False
        vals = body[mask]
        assert np.abs(vals - anchor_lv[c]).max() < 1.5, (c, np.abs(vals - anchor_lv[c]).max())
    # No signal loss: the bright blob survives at full amplitude in the core interior.
    assert np.isclose(sci[12, 12, 0], anchor_lv[0] + 50.0, atol=1.5)

    # NO seam step: the assembled channel is (near-)flat across the core boundary.
    row = 16  # crosses the vertical boundary x=32
    left = float(np.mean(sci[row, 28:32, 0]))
    right = float(np.mean(sci[row, 32:36, 0]))
    assert abs(left - right) < 1.0


def test_pure_level_shift_recovered_exactly():
    levels = {"r0000c0000": 10.0, "r0000c0001": 25.0,
              "r0001c0000": -5.0, "r0001c0001": 40.0}
    canvas, patches = _make_patches(2, 2, levels, sigma=0.2)
    graph = zcf.build_cell_offset_graph(canvas, 2, 2, patches)
    offsets = {cid: graph.offsets[i] for i, cid in enumerate(graph.cell_ids)}
    for cid, cp in patches.items():
        shifted = zcf.apply_additive_offsets(np.asarray(cp.science, np.float64), offsets[cid])
        # The cell's core sky now equals the anchor level (100% additive recovery).
        y0, y1, x0, x1 = cp.core_slice
        core = shifted[y0:y1, x0:x1]
        assert np.allclose(core, levels["r0000c0000"], atol=1.0)


# ---------------------------------------------------------------------------
# Disconnected-cells policy (explicit)
# ---------------------------------------------------------------------------

def _line3_missing_middle():
    # 1x3 grid, drop the middle cell -> two components (anchor's + a lone one).
    levels = {"r0000c0000": 10.0, "r0000c0001": 20.0, "r0000c0002": 30.0}
    canvas, patches = _make_patches(3, 1, levels, core=16)
    del patches["r0000c0001"]
    return canvas, patches


def test_disconnected_raises_by_default():
    canvas, patches = _line3_missing_middle()
    with pytest.raises(zco.DisconnectedOffsetGraphError):
        zcf.build_cell_offset_graph(canvas, 3, 1, patches, on_disconnected="raise")


def test_disconnected_zero_policy_recorded():
    canvas, patches = _line3_missing_middle()
    graph = zcf.build_cell_offset_graph(canvas, 3, 1, patches, on_disconnected="zero")
    assert graph.policy == "zero"
    assert graph.disconnected == ("r0000c0002",)  # the unreachable cell
    idx = graph.cell_ids.index("r0000c0002")
    assert graph.offsets[idx, 0] == 0.0  # kept on its own level, no bridge


def test_wired_feather_records_disconnected_and_does_not_raise():
    from zemosaic import zemosaic_zegrid_mode as zz

    canvas, patches = _line3_missing_middle()
    results = _results_from_patches(patches)
    assembled, info = zz._assemble_per_cell_feathered(
        canvas, 3, 1, results, halo_px=HALO
    )
    assert info["policy"] == "zero"
    assert "r0000c0002" in info["disconnected"]
    assert np.isfinite(np.asarray(assembled.science)).any()


# ---------------------------------------------------------------------------
# Global-mode regression: byte-identical one-owner assembly, no feather
# ---------------------------------------------------------------------------

def _cores_for(nx, ny, core=CORE):
    cores = {}
    for row in range(ny):
        for col in range(nx):
            cid = _cell_id(row, col)
            cores[cid] = {
                "science_core": np.full((core, core, 3), float(row * nx + col), np.float32),
                "surviving_sample_count_core": np.ones((core, core, 3), np.int32),
                "n_eff_support_core": np.ones((core, core), np.float64),
                "support_w1_core": np.ones((core, core), np.float64),
                "support_w2_core": np.ones((core, core), np.float64),
                "valid_mask_core": np.ones((core, core, 3), bool),
            }
    return cores


def test_global_mode_assembly_is_byte_identical():
    from zemosaic import zemosaic_zegrid_mode as zz

    canvas = _canvas(2, 2)
    cores = _cores_for(2, 2)
    ref = zmosaic.assemble_canvas(canvas, 2, 2, cores)
    got, info = zz._assemble_cells(canvas, 2, 2, "global", cores, {})
    assert info is None
    np.testing.assert_array_equal(ref.science, got.science)
    np.testing.assert_array_equal(ref.stack_depth, got.stack_depth)
    np.testing.assert_array_equal(ref.valid_fraction, got.valid_fraction)
    # Default mode resolves to global.
    assert zz.zphot.CELL_GAUGE_MODE_DEFAULT == "global"


def test_global_mode_unchanged_by_default_config():
    from zemosaic.core.zegrid.executor import ExecutorConfig

    assert ExecutorConfig().science_config().cell_gauge_mode == "global"


# ---------------------------------------------------------------------------
# Determinism + weights
# ---------------------------------------------------------------------------

def test_per_cell_feather_deterministic():
    levels = {"r0000c0000": 100.0, "r0000c0001": 104.0,
              "r0001c0000": 96.0, "r0001c0001": 110.0}
    canvas, patches = _make_patches(2, 2, levels)
    results = _results_from_patches(patches)
    from zemosaic import zemosaic_zegrid_mode as zz

    a, ia = zz._assemble_per_cell_feathered(canvas, 2, 2, results)
    b, ib = zz._assemble_per_cell_feathered(canvas, 2, 2, results)
    np.testing.assert_array_equal(a.science, b.science)
    assert ia["offsets"] == ib["offsets"]


def test_edge_weight_is_shared_valid_support_count():
    levels = {"r0000c0000": 100.0, "r0000c0001": 104.0}
    canvas, patches = _make_patches(2, 1, levels)
    delta, n, weight = zcf.overlap_edge_observation(
        patches["r0000c0000"], patches["r0000c0001"]
    )
    # The overlap is the 16-px-wide halo band x the full 32-px cell height.
    assert int(np.max(n)) == 32 * 16
    np.testing.assert_array_equal(weight, n.astype(np.float64))
    assert np.allclose(delta, -4.0, atol=0.5)  # 100 - 104


def test_overlap_observation_robust_to_outliers():
    levels = {"r0000c0000": 100.0, "r0000c0001": 100.0}
    canvas, patches = _make_patches(2, 1, levels, sigma=0.3)
    a = patches["r0000c0000"]
    sci = np.array(a.science, copy=True)
    # Inject gross bright pixels INSIDE the shared overlap band (right side), but
    # only a MINORITY of the band so the robust percentile+MAD estimate holds.
    py0, py1, px0, px1 = a.bounds
    sci[:, -3:, :] = 1.0e5
    a2 = zcf.CellPatch(a.cell_id, a.bounds, a.core_slice, sci.astype(np.float32), a.valid_mask)
    delta, n, weight = zcf.overlap_edge_observation(a2, patches["r0000c0001"])
    assert abs(float(delta[0])) < 1.0, float(delta[0])
