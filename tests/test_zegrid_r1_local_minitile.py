"""ZM-ZEGRID-R1 targeted tests — local MiniTile implementation (one real Cell).

These are isolated R1 tests over the frozen M106 geometry + prepared RGB
fixtures. They do NOT touch production dispatch. Fixture precondition lives
outside the git tree (``/tmp/zegrid_r1_fixtures``); tests that need it are
skipped with a clear reason when it is absent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import execution as zx
from zemosaic.core.zegrid import science_adapter as zs
from zemosaic.core.zegrid import assembly as za
from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack

# --- Frozen R0 oracle (geometry simulator) + frozen JSON ---------------------
_MAIN_REPO = Path("/home/tristan/.openclaw/workspace/projects/zemosaic")
_R0_GEOM = _MAIN_REPO / "tools/zegrid_r0/geometry.py"
_FROZEN_JSON = _MAIN_REPO / "docs/refactor/zegrid_r0/M106_geometry.json"
_LIGHTS = Path("/home/tristan/M106/lights")
_FIXTURES = Path("/tmp/zegrid_r1_fixtures")

# The frozen R0 oracle + JSON live in the MAIN repo (not the git tree); on a
# clean checkout / CI runner they are absent, so the whole module skips cleanly
# instead of failing at import (the tests are slow-tier and excluded there anyway).
if not (_R0_GEOM.is_file() and _FROZEN_JSON.is_file()):
    pytest.skip("R1 frozen R0 oracle not present (main repo)", allow_module_level=True)


def _load_r0_sim():
    spec = importlib.util.spec_from_file_location("zegrid_r0_geometry", _R0_GEOM)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["zegrid_r0_geometry"] = mod
    spec.loader.exec_module(mod)
    return mod


R0 = _load_r0_sim()
_FROZEN = json.loads(_FROZEN_JSON.read_text())


def _frozen_wcs() -> WCS:
    return WCS(_FROZEN["canvas"]["wcs_header"])


@pytest.fixture(scope="module")
def manifest(m106_corpus):
    frames, rejected, _canvas = m106_corpus
    return frames, rejected


@pytest.fixture(scope="module")
def canvas(m106_corpus):
    _frames, _rejected, canvas = m106_corpus
    return canvas


# ---------------------------------------------------------------------------
# Geometry: exact R0 reproduction
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_canvas_reproduces_frozen_exactly(canvas):
    assert canvas.width == _FROZEN["canvas"]["width"] == 2403
    assert canvas.height == _FROZEN["canvas"]["height"] == 3278
    assert canvas.resolution_deg == pytest.approx(
        _FROZEN["canvas"]["resolution_deg"], abs=1e-18
    )
    frozen = _frozen_wcs()
    mine = canvas.wcs()
    np.testing.assert_allclose(mine.wcs.crpix, frozen.wcs.crpix, atol=1e-9)
    np.testing.assert_allclose(mine.wcs.crval, frozen.wcs.crval, atol=1e-9)
    np.testing.assert_allclose(mine.wcs.cdelt, frozen.wcs.cdelt, atol=1e-12)
    np.testing.assert_allclose(mine.wcs.get_pc(), frozen.wcs.get_pc(), atol=1e-12)


@pytest.mark.slow
def test_cell_r0000c0000_core_patch(manifest, canvas):
    frames, _ = manifest
    layout = zg.build_layout(canvas, 5, 4)
    cell = zg.ZeGridCell(
        zg.cell_id(0, 0), canvas.canvas_id, layout.layout_id, 0, 0,
        layout.cell_bounds(0, 0, canvas),
    )
    assert cell.core == zg.GlobalBounds(0, 0, 480, 819)
    patch = zg.build_patch(canvas, cell, 8)
    assert patch.patch == zg.GlobalBounds(0, 0, 488, 827)
    # core slice in patch: y=0:819, x=0:480
    assert patch.core_slice == zg.PatchBounds(0, 0, 480, 819)
    assert patch.patch_shape_hw == (827, 488)


@pytest.mark.slow
def test_membership_matches_frozen(manifest, canvas):
    frames, _ = manifest
    layout = zg.build_layout(canvas, 5, 4)
    cell = zg.ZeGridCell(
        zg.cell_id(0, 0), canvas.canvas_id, layout.layout_id, 0, 0,
        layout.cell_bounds(0, 0, canvas),
    )
    patch = zg.build_patch(canvas, cell, 8)
    mem = zg.compute_membership(frames, canvas, cell, patch)

    frozen_cell = next(
        c
        for L in _FROZEN["layouts"]
        if L["nx"] == 5 and L["ny"] == 4 and L["halo_target_px"] == 8
        for c in L["cells"]
        if c["cell_id"] == "r0000c0000"
    )
    assert list(mem.core_ids) == frozen_cell["frames"]
    assert list(mem.patch_ids) == frozen_cell["patch_frames"]
    assert len(mem.core_ids) == 5
    assert len(mem.patch_ids) == 7


@pytest.mark.slow
def test_source_roi_plans_match_r0(manifest, canvas):
    frames, _ = manifest
    layout = zg.build_layout(canvas, 5, 4)
    cell = zg.ZeGridCell(
        zg.cell_id(0, 0), canvas.canvas_id, layout.layout_id, 0, 0,
        layout.cell_bounds(0, 0, canvas),
    )
    patch = zg.build_patch(canvas, cell, 8)
    mem = zg.compute_membership(frames, canvas, cell, patch)
    by_id = {f.frame_id.logical_path: f for f in frames}

    # Reconstruct the R0 source crops for the same patch contributors.
    r0_frames, _ = R0.read_frames(_LIGHTS)
    r0_canvas = R0.make_canvas(r0_frames)
    r0_by_key = {f.key: f for f in r0_frames}
    r0_patch_rect = R0.rect((0, 0, 488, 827))

    for key in mem.patch_ids:
        my = zg.plan_source_roi(by_id[key], canvas, patch, margin_px=2)
        r0f = r0_by_key[key]
        inter = R0.polygon(r0f, r0_canvas.wcs).intersection(r0_patch_rect)
        r0crop = R0.source_crop(r0f, r0_canvas, inter, 2)
        assert my is not None
        assert tuple(my.source_bounds.__dict__.values()) == tuple(r0crop)


@pytest.mark.slow
def test_layout_partition_disjoint_and_exhaustive(manifest, canvas):
    frames, _ = manifest
    layout = zg.build_layout(canvas, 5, 4)
    seen = np.zeros((canvas.height, canvas.width), dtype=int)
    for row, col, bounds in layout.iter_cells(canvas):
        seen[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1] += 1
    assert np.all(seen == 1)


# ---------------------------------------------------------------------------
# Permutation determinism
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_permutation_determinism(manifest):
    frames, _ = manifest
    rng = np.random.default_rng(42)
    base_canvas = zg.build_canvas(frames)
    base_layout = zg.build_layout(base_canvas, 5, 4)
    base_cell = zg.ZeGridCell(
        zg.cell_id(0, 0), base_canvas.canvas_id, base_layout.layout_id, 0, 0,
        base_layout.cell_bounds(0, 0, base_canvas),
    )
    base_patch = zg.build_patch(base_canvas, base_cell, 8)
    base_mem = zg.compute_membership(frames, base_canvas, base_cell, base_patch)
    base_plans = {
        k: zg.plan_source_roi(
            {f.frame_id.logical_path: f for f in frames}[k], base_canvas, base_patch
        )
        for k in base_mem.patch_ids
    }

    for name, order in [
        ("normal", list(frames)),
        ("reverse", list(reversed(frames))),
        ("random", [frames[i] for i in rng.permutation(len(frames))]),
    ]:
        c = zg.build_canvas(order)
        l = zg.build_layout(c, 5, 4)
        cl = zg.ZeGridCell(zg.cell_id(0, 0), c.canvas_id, l.layout_id, 0, 0, l.cell_bounds(0, 0, c))
        p = zg.build_patch(c, cl, 8)
        m = zg.compute_membership(order, c, cl, p)
        by = {f.frame_id.logical_path: f for f in order}
        plans = {k: zg.plan_source_roi(by[k], c, p) for k in m.patch_ids}
        assert c.canvas_id == base_canvas.canvas_id, name
        assert l.layout_id == base_layout.layout_id, name
        assert cl.core == base_cell.core, name
        assert p.patch == base_patch.patch, name
        assert m.core_ids == base_mem.core_ids, name
        assert m.patch_ids == base_mem.patch_ids, name
        for k in base_plans:
            assert plans[k].source_bounds == base_plans[k].source_bounds, (name, k)


# ---------------------------------------------------------------------------
# WCS qualification (reject unsupported projections)
# ---------------------------------------------------------------------------

def _synthetic_frame(key="a", ra=10.0, dec=30.0, angle=0.0, shape=(40, 60)):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    a = np.deg2rad(angle)
    w.wcs.cd = np.array([[-np.cos(a), np.sin(a)], [np.sin(a), np.cos(a)]]) * 0.001
    w.array_shape = shape
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(key),
        source_path=key,
        shape_hw=shape,
        wcs_header=zg._serialize_wcs(w),
    )


def test_qualify_rejects_sin_and_pv_and_lookup():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    assert "TAN" in zg.qualify_wcs(w)
    w2 = WCS(naxis=2)
    w2.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w2.wcs.crval = [10, 30]
    w2.wcs.crpix = [10, 10]
    w2.wcs.cd = np.eye(2) * 0.001
    w2.wcs.set_pv([(2, 1, 10.0)])
    assert "PV" in zg.qualify_wcs(w2)


def test_qualify_axis_layout_rejects_ambiguous_3d():
    assert zg.qualify_axis_layout((1920, 1080), "mono") is None
    assert zg.qualify_axis_layout((3, 1920, 1080), "CHW") is None
    assert zg.qualify_axis_layout((3, 1920, 1080), "") is not None
    assert zg.qualify_axis_layout((1920, 1080, 3), "HWC") is None


def test_edge_corner_patch_clipping_rotated_tan():
    # Synthetic rotated-TAN single frame: verify patch clipping at canvas corners.
    f = _synthetic_frame(angle=27.0, shape=(200, 200))
    canvas = zg.build_canvas([f])
    layout = zg.build_layout(canvas, 3, 3)
    for row, col, bounds in layout.iter_cells(canvas):
        cell = zg.ZeGridCell(zg.cell_id(row, col), canvas.canvas_id, layout.layout_id,
                             row, col, bounds)
        patch = zg.build_patch(canvas, cell, 8)
        # Patch stays within canvas.
        assert 0 <= patch.patch.x0 <= patch.patch.x1 <= canvas.width
        assert 0 <= patch.patch.y0 <= patch.patch.y1 <= canvas.height
        # Core slice within patch.
        cs = patch.core_slice
        assert 0 <= cs.x0 <= cs.x1 <= patch.patch.width
        assert 0 <= cs.y0 <= cs.y1 <= patch.patch.height
        # Every core pixel is inside the patch (patch = core + halo).
        assert patch.patch.x0 <= cell.core.x0 and cell.core.x1 <= patch.patch.x1
        assert patch.patch.y0 <= cell.core.y0 and cell.core.y1 <= patch.patch.y1


# ---------------------------------------------------------------------------
# Execution: locality + local-vs-full reprojection
# ---------------------------------------------------------------------------

def _prepared_path(key: str) -> Path:
    return _FIXTURES / (Path(key).stem + "_rgb.fits")


@pytest.fixture(scope="module")
def execution_context(manifest, canvas):
    frames, _ = manifest
    layout = zg.build_layout(canvas, 5, 4)
    cell = zg.ZeGridCell(
        zg.cell_id(0, 0), canvas.canvas_id, layout.layout_id, 0, 0,
        layout.cell_bounds(0, 0, canvas),
    )
    patch = zg.build_patch(canvas, cell, 8)
    mem = zg.compute_membership(frames, canvas, cell, patch)
    by_id = {f.frame_id.logical_path: f for f in frames}
    patch_frames = [by_id[k] for k in mem.patch_ids]
    prepared_paths = {k: str(_prepared_path(k)) for k in mem.patch_ids}
    crop_plans = {
        f.frame_id.logical_path: zg.plan_source_roi(f, canvas, patch)
        for f in patch_frames
    }
    return patch_frames, prepared_paths, crop_plans, patch, by_id


@pytest.mark.slow
def test_section_reads_are_local(execution_context):
    patch_frames, prepared_paths, crop_plans, patch, by_id = execution_context
    canvas = zg.build_canvas(list(by_id.values()))
    tracker = zx.SectionReadTracker()
    contribs = zx.build_patch_contributors(
        patch_frames, prepared_paths, canvas, patch, crop_plans, tracker=tracker
    )
    assert len(contribs) == 7
    assert len(tracker.records) == 7
    for rec in tracker.records:
        # Every read is a strict sub-rectangle of the full frame (never full .data).
        assert rec.n_pixels_read < rec.full_frame_pixels
        assert rec.source_bounds.width * rec.source_bounds.height * 3 == rec.n_pixels_read
        assert rec.axis_layout == "CHW"


@pytest.mark.slow
def test_section_read_never_touches_full_data(monkeypatch, execution_context):
    patch_frames, prepared_paths, crop_plans, patch, by_id = execution_context
    canvas = zg.build_canvas(list(by_id.values()))
    import astropy.io.fits.hdu.image as imghdu

    class Boom(Exception):
        pass

    def raiser(self):
        raise Boom("full .data read!")

    orig = imghdu._ImageBaseHDU.data
    imghdu._ImageBaseHDU.data = property(raiser)
    try:
        tracker = zx.SectionReadTracker()
        contribs = zx.build_patch_contributors(
            patch_frames, prepared_paths, canvas, patch, crop_plans, tracker=tracker
        )
        assert len(contribs) == 7
    finally:
        imghdu._ImageBaseHDU.data = orig


@pytest.mark.slow
def test_local_vs_full_reprojection_covers_every_valid_pixel(execution_context):
    patch_frames, prepared_paths, crop_plans, patch, by_id = execution_context
    canvas = zg.build_canvas(list(by_id.values()))
    tracker = zx.SectionReadTracker()
    contribs = zx.build_patch_contributors(
        patch_frames, prepared_paths, canvas, patch, crop_plans, tracker=tracker
    )
    patch_wcs = patch.patch_wcs()
    for c in contribs:
        fd = by_id[c.frame_id]
        full_rgb, full_geom = zx.reproject_full_source(
            fd, prepared_paths[c.frame_id], patch_wcs, patch.patch_shape_hw
        )
        # Every pixel where the full source is valid must also be covered by the
        # cropped source, and the support maps must match exactly.
        assert np.array_equal(c.geometric_support, full_geom)
        diff = np.abs(c.rgb - full_rgb)
        covered = full_geom & np.isfinite(full_rgb).all(axis=-1)
        if covered.any():
            np.testing.assert_allclose(
                diff[covered], 0.0, rtol=1e-6, atol=1e-4
            )


@pytest.mark.slow
def test_source_roi_covers_every_requested_valid_target_pixel(execution_context):
    patch_frames, prepared_paths, crop_plans, patch, by_id = execution_context
    canvas = zg.build_canvas(list(by_id.values()))
    patch_wcs = patch.patch_wcs()
    for f in patch_frames:
        plan = crop_plans[f.frame_id.logical_path]
        assert plan is not None
        # Every patch pixel centre inside the source polygon must inverse-map to a
        # source coordinate within the crop (+ margin respected).
        poly = zg._source_polygon(f.shape_hw, f.wcs(), canvas.wcs())
        xx, yy = np.meshgrid(
            np.arange(patch.patch.x0, patch.patch.x1) + 0.5,
            np.arange(patch.patch.y0, patch.patch.y1) + 0.5,
        )
        from shapely import contains_xy

        inside = contains_xy(poly, xx.ravel(), yy.ravel()).reshape(xx.shape)
        if not inside.any():
            continue
        xy = zg._project_points(
            np.column_stack((xx[inside], yy[inside])), canvas.wcs(), f.wcs()
        )
        b = plan.source_bounds
        assert np.all(xy[:, 0] >= b.x0 - 0.5 - 1e-8)
        assert np.all(xy[:, 0] <= b.x1 - 0.5 + 1e-8)
        assert np.all(xy[:, 1] >= b.y0 - 0.5 - 1e-8)
        assert np.all(xy[:, 1] <= b.y1 - 0.5 + 1e-8)


# ---------------------------------------------------------------------------
# Science adapter: canonical oracle + determinism + assembly
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_adapter_matches_direct_engine_exactly(execution_context):
    patch_frames, prepared_paths, crop_plans, patch, by_id = execution_context
    canvas = zg.build_canvas(list(by_id.values()))
    tracker = zx.SectionReadTracker()
    contribs = zx.build_patch_contributors(
        patch_frames, prepared_paths, canvas, patch, crop_plans, tracker=tracker
    )
    cfg = zs.MiniTileScienceConfig()
    images = [c.rgb for c in contribs]
    supports = [c.geometric_support for c in contribs]
    order = [c.frame_id for c in contribs]
    sres = zs.run_minitile_stack(images, supports, order, cfg)

    direct = run_canonical_stack(
        CanonicalStackRequest(
            images=list(images),
            geometric_support=list(supports),
            normalization=cfg.normalization,
            weighting=cfg.weighting,
            rejection=cfg.rejection,
            combine=cfg.combine,
            reference_index=None,
            taper=cfg.taper,
            taper_px=cfg.taper_px,
            taper_floor=cfg.taper_floor,
            backend=cfg.backend,
            equalize_rgb=False,
        )
    )
    # Arrays bit-identical; support W1/W2 within tight float64 tolerance.
    assert np.array_equal(sres.result.science, direct.science, equal_nan=True)
    assert np.array_equal(sres.result.estimator_weight_sum, direct.estimator_weight_sum)
    np.testing.assert_allclose(sres.result.support_w1, direct.support_w1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sres.result.support_w2, direct.support_w2, rtol=1e-12, atol=1e-12)
    assert np.array_equal(sres.result.n_eff_support, direct.n_eff_support)
    assert np.array_equal(sres.result.valid_mask, direct.valid_mask)
    assert np.array_equal(sres.result.surviving_sample_count, direct.surviving_sample_count)
    # Reference/exclusion identity resolved to stable FrameId.
    assert sres.reference_frame_id is not None
    assert sres.reference_frame_id == order[int(sres.result.provenance["reference"]["index"])]


@pytest.mark.slow
def test_science_permutation_determinism(execution_context):
    patch_frames, prepared_paths, crop_plans, patch, by_id = execution_context
    canvas = zg.build_canvas(list(by_id.values()))
    cfg = zs.MiniTileScienceConfig()

    def run(frames_ordered):
        tracker = zx.SectionReadTracker()
        contribs = zx.build_patch_contributors(
            frames_ordered, prepared_paths, canvas, patch, crop_plans, tracker=tracker
        )
        return zs.run_minitile_stack(
            [c.rgb for c in contribs],
            [c.geometric_support for c in contribs],
            [c.frame_id for c in contribs],
            cfg,
        )

    base = run(list(patch_frames))
    rev = run(list(reversed(patch_frames)))
    rng = np.random.default_rng(7)
    rand = run([patch_frames[i] for i in rng.permutation(len(patch_frames))])

    # ``build_patch_contributors`` sorts by FrameId, so science is permutation
    # invariant: identical reference ID, exclusions, masks, counts.
    for other in (rev, rand):
        assert other.reference_frame_id == base.reference_frame_id
        assert other.excluded == base.excluded
        assert tuple(other.frame_order) == tuple(base.frame_order)
        assert np.array_equal(other.result.valid_mask, base.result.valid_mask)
        assert np.array_equal(
            other.result.surviving_sample_count, base.result.surviving_sample_count
        )
        assert np.array_equal(other.result.rejection_mask, base.result.rejection_mask)


@pytest.mark.slow
def test_assembly_extracts_core_slice(execution_context):
    patch_frames, prepared_paths, crop_plans, patch, by_id = execution_context
    canvas = zg.build_canvas(list(by_id.values()))
    tracker = zx.SectionReadTracker()
    contribs = zx.build_patch_contributors(
        patch_frames, prepared_paths, canvas, patch, crop_plans, tracker=tracker
    )
    cfg = zs.MiniTileScienceConfig()
    sres = zs.run_minitile_stack(
        [c.rgb for c in contribs], [c.geometric_support for c in contribs],
        [c.frame_id for c in contribs], cfg,
    )
    mt = za.extract_minitile(patch, sres)
    cores = za.crop_all_planes_to_core(mt)
    assert cores["science_core"].shape == (819, 480, 3)
    # Channel-shaped planes (estimator sum, valid mask, survivor count) keep C=3;
    # support maps are channel-invariant 2-D.
    assert cores["estimator_weight_sum_core"].shape == (819, 480, 3)
    assert cores["valid_mask_core"].shape == (819, 480, 3)
    assert cores["surviving_sample_count_core"].shape == (819, 480, 3)
    assert cores["support_w1_core"].shape == (819, 480)
    assert cores["n_eff_support_core"].shape == (819, 480)
    assert mt.cell_id == "r0000c0000"
    assert mt.core_slice == (0, 819, 0, 480)
