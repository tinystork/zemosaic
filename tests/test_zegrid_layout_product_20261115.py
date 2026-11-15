"""ZM-LAYOUT-PRODUCT-20261115 — geometry-only candidate scan, exact footprint
index and chosen-scan reuse in the REAL mode-aware chooser.

What this mission changed (product, phase 2 only):

* the per-Cell patch of a candidate scan is computed with the GEOMETRY-ONLY
  :func:`zemosaic.core.zegrid.geometry.patch_geometry` seam instead of building a
  full :class:`~zemosaic.core.zegrid.geometry.ProcessingPatch` (deep-copied WCS +
  serialized FITS header) that the scan never reads;
* the contributor predicate is evaluated over a bounding-box SUPERSET produced by
  an exactness-preserving :class:`~zemosaic.core.zegrid.auto_layout.FootprintIndex`
  built ONCE per layout choice;
* the chosen grid's per-Cell dicts are reused from the scan that admitted it
  instead of re-running the identical pass.

The acceptance contract pinned here is PARITY FIRST: the same
memberships/counts, memory reservations, candidate ordering/rank, chosen grid,
log lines, refusals and published layout dict, on the REAL chooser (never a
replica), with the index ON versus OFF. Every test is synthetic and fast (no
user data, no pixels, no multiprocessing); the read-only corpus probes live in
``docs/qualification/zm-layout-product-20261115/``.
"""

from __future__ import annotations

import math
import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.wcs import WCS
from shapely.geometry import Polygon
from shapely.geometry import box as sbox

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic import zegrid_halo as zhalo
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import sweep as zsw

MIB = 2 ** 20


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

def _frame(i, h, w, scale, ra, dec, tag="d"):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
    wcs.array_shape = (h, w)
    fid = f"{tag}{i:05d}.fits"
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid), source_path=f"/x/{fid}", shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(),
        header_sha256="", instrument="",
    )


def _halo_corpus():
    """5x4 frames -> factor-1 4x3, halo-128 overhead ~0.98 (rescue/advisory)."""
    descs = [_frame(i * 4 + j, 1000, 600, 0.001, 10.0 + i * 0.45, 30.0 + j * 0.4, "h")
             for i in range(5) for j in range(4)]
    return descs, zg.build_canvas(descs)


def _sparse_corpus(n=180, grid=18, side=90, step=0.03):
    descs = []
    for i in range(n):
        gx = i % grid
        gy = (i // grid) % grid
        descs.append(_frame(i, side, side, 0.001,
                            10.0 + gx * step, 30.0 + gy * step, "s"))
    return descs, zg.build_canvas(descs)


def _compact_corpus():
    """Tiny canvas: the whole refinement ladder is cheap to scan to its floor."""
    descs = [_frame(i, 60, 60, 0.001, 10.0 + (i % 3) * 0.02, 30.0 + (i // 3) * 0.02, "c")
             for i in range(9)]
    return descs, zg.build_canvas(descs)


def _load_corpus(core_n=12, core_step=0.012, core_reps=6):
    """The imbalanced corpus of the load-balance suite (pass TRIGGERS)."""
    descs, i = [], 0
    for ix in range(core_n):
        for iy in range(core_n):
            for r in range(core_reps):
                descs.append(_frame(i, 80, 80, 0.001,
                                    10.0 + (ix - core_n / 2) * core_step + 2e-4 * r,
                                    30.0 + (iy - core_n / 2) * core_step + 2e-4 * r,
                                    "l"))
                i += 1
    canvas = zg.build_canvas(descs)
    return descs, zg.GlobalCanvas(
        canvas_id=canvas.canvas_id + "-pad13",
        wcs_header=canvas.wcs_header,
        width=int(canvas.width * 1.3), height=int(canvas.height * 1.3),
        resolution_deg=canvas.resolution_deg,
    )


def _floors(min_patch_area=1, min_contributors=1, max_halo=1e9):
    return zal.ScientificFloors(
        min_patch_area_px=min_patch_area,
        min_contributors=min_contributors,
        max_halo_overhead=max_halo,
    )


# ---------------------------------------------------------------------------
# Independent reference: the FULL historical patch + explicit brute-force scan
# ---------------------------------------------------------------------------

def _reference_cells(canvas, footprints, nx, ny, halo_px, tile_size):
    layout = zg.build_layout(canvas, nx, ny)
    out = []
    for row, col, bounds in layout.iter_cells(canvas):
        cell = zg.ZeGridCell(zg.cell_id(row, col), canvas.canvas_id,
                             layout.layout_id, row, col, bounds)
        patch = zg.build_patch(canvas, cell, halo_px)  # FULL patch (WCS + header)
        rect = zg._rect(patch.patch)
        n = 0
        for _f, poly in footprints:                    # explicit, no product helper
            if poly.intersection(rect).area > zg.INTERSECTION_AREA_EPS:
                n += 1
        area = patch.patch.width * patch.patch.height
        hw = patch.patch_shape_hw
        inmem = zal.FITTED_MEMORY_MODEL.predict_bound_bytes(n, area)
        stream = zz._profile_stream_bytes(
            n, hw, tile_size, zz._STACK_STORAGE_PROFILE
        )
        out.append({
            "cell_id": zg.cell_id(row, col), "row": row, "col": col, "n": n,
            "patch_area": area, "patch_hw": list(hw),
            "inmem_bound_bytes": int(inmem), "stream_bound_bytes": int(stream),
            "cheaper_mode": "inmem" if inmem <= stream else "stream",
        })
    return out


def _choose(canvas, descs, **kw):
    logs = []
    layout = zz._choose_layout_mode_aware(
        canvas, descs, log_line=logs.append, footprints_out=[], **kw
    )
    return layout, logs


def _choose_or_refusal(canvas, descs, **kw):
    logs = []
    try:
        layout = zz._choose_layout_mode_aware(
            canvas, descs, log_line=logs.append, footprints_out=[], **kw
        )
        return layout, None, logs
    except zal.LayoutInfeasible as exc:
        return None, {"message": str(exc), "refusal": getattr(exc, "refusal", None)}, logs


# ---------------------------------------------------------------------------
# 1. The geometry-only seam is the SAME rectangle as a real patch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("halo", [0, 1, 8, 64, 1000])
@pytest.mark.parametrize("grid", [(1, 1), (3, 2), (7, 5), (13, 11)])
def test_patch_geometry_equals_build_patch(halo, grid):
    descs, _ = _sparse_corpus(n=6, grid=3, side=60, step=0.02)
    canvas = zg.build_canvas(descs)
    layout = zg.build_layout(canvas, grid[0], grid[1])
    for row, col, bounds in layout.iter_cells(canvas):
        cell = zg.ZeGridCell(zg.cell_id(row, col), canvas.canvas_id,
                             layout.layout_id, row, col, bounds)
        full = zg.build_patch(canvas, cell, halo)
        geom = zg.patch_geometry(bounds, canvas, halo)
        assert geom.patch == full.patch
        assert geom.core_slice == full.core_slice
        assert geom.patch_shape_hw == full.patch_shape_hw
        assert geom.halo_px == halo
    # The core slice still extracts the core from a patch-shaped plane.
    first = next(iter(layout.iter_cells(canvas)))
    geom = zg.patch_geometry(first[2], canvas, halo)
    assert (geom.core_slice.width, geom.core_slice.height) == (
        first[2].width, first[2].height
    )


def test_patch_geometry_rejects_negative_halo():
    descs, _ = _sparse_corpus(n=4, grid=2, side=50, step=0.02)
    canvas = zg.build_canvas(descs)
    bounds = next(iter(zg.build_layout(canvas, 1, 1).iter_cells(canvas)))[2]
    with pytest.raises(ValueError):
        zg.patch_geometry(bounds, canvas, -1)


# ---------------------------------------------------------------------------
# 2. Scan cells == independent full-patch brute-force reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid", [(1, 1), (2, 2), (4, 3), (9, 7)])
@pytest.mark.parametrize("use_index", [False, True])
def test_scan_cells_match_independent_reference(grid, use_index, monkeypatch):
    descs, canvas = _halo_corpus()
    footprints = zal.project_footprints(descs, canvas)
    index = zal.FootprintIndex(footprints.polygons) if use_index else None
    halo = zz.zsw.HALO_PX
    got = list(zz._iter_cell_bounds_mode_aware(
        canvas, footprints, grid[0], grid[1], halo, zz.STREAM_TILE_SIZE, index=index
    ))
    ref = _reference_cells(
        canvas, footprints, grid[0], grid[1], halo, zz.STREAM_TILE_SIZE
    )
    assert got == ref
    if use_index:
        assert index.query_calls == len(ref)


def test_cells_out_is_row_major_and_equals_the_generator(monkeypatch):
    descs, canvas = _halo_corpus()
    footprints = zal.project_footprints(descs, canvas)
    index = zal.FootprintIndex(footprints.polygons)
    out = []
    worst = zz._scan_layout_mode_aware(
        canvas, footprints, 4, 3, zz.zsw.HALO_PX, zz.STREAM_TILE_SIZE,
        index=index, cells_out=out,
    )
    assert out == list(zz._iter_cell_bounds_mode_aware(
        canvas, footprints, 4, 3, zz.zsw.HALO_PX, zz.STREAM_TILE_SIZE, index=index
    ))
    assert [c["cell_id"] for c in out] == [
        zg.cell_id(r, c) for r in range(3) for c in range(4)
    ]
    # The scan's worst cell remains the max of the cheaper bound (unchanged).
    assert worst["cheaper_bound_bytes"] == max(
        min(c["inmem_bound_bytes"], c["stream_bound_bytes"]) for c in out
    )
    # And the aggregate load is the exact sum over those same cells.
    assert worst["membership_load"].total_slots == sum(c["n"] for c in out)


def test_scan_never_builds_a_wcs_header():
    """The candidate scan must not serialize a WCS header per Cell."""
    descs, canvas = _halo_corpus()
    footprints = zal.project_footprints(descs, canvas)
    original = zg._serialize_wcs

    def _boom(*a, **k):  # pragma: no cover - only on regression
        raise AssertionError("geometry-only scan built a WCS header")

    zg._serialize_wcs = _boom
    try:
        worst = zz._scan_layout_mode_aware(
            canvas, footprints, 4, 3, zz.zsw.HALO_PX, zz.STREAM_TILE_SIZE
        )
    finally:
        zg._serialize_wcs = original
    assert worst["cheaper_bound_bytes"] > 0


# ---------------------------------------------------------------------------
# 3. Index exactness (adversarial geometry, all backends)
# ---------------------------------------------------------------------------

def _adversarial_polygons():
    return [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),           # ordinary
        Polygon([(10, 0), (20, 0), (20, 10), (10, 10)]),          # touching right
        Polygon([(0, 10), (10, 10), (10, 20), (0, 20)]),          # touching below
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),            # exact duplicate
        Polygon([(30, 30), (40, 30), (40, 40), (30, 40)]),        # disjoint
        Polygon([(5, 5), (5.0 + 1e-6, 5), (5.0 + 1e-6, 5 + 1e-6)]),  # EPS sliver
        Polygon([(50, 50), (50.0005, 50), (50.0005, 50.0005)]),   # tiny but > EPS
        sbox(-100, -100, 100, 100),                                # huge
    ]


@pytest.mark.parametrize("backend", ["strtree", "sorted", "bruteforce"])
def test_index_is_an_exact_superset_over_adversarial_rects(backend):
    polys = _adversarial_polygons()
    idx = zal.FootprintIndex(polys, backend=backend)
    rects = [
        zg.GlobalBounds(0, 0, 10, 10), zg.GlobalBounds(10, 0, 20, 10),
        zg.GlobalBounds(0, 10, 10, 20), zg.GlobalBounds(5, 5, 6, 6),
        zg.GlobalBounds(30, 30, 40, 40), zg.GlobalBounds(100, 100, 110, 110),
        zg.GlobalBounds(49, 49, 51, 51), zg.GlobalBounds(0, 0, 1, 1),
    ]
    for bounds in rects:
        rect = zg._rect(bounds)
        exact = [
            i for i, poly in enumerate(polys)
            if poly.intersection(rect).area > zg.INTERSECTION_AREA_EPS
        ]
        q = idx.query(rect)
        assert q == sorted(q), "query must return ascending indices"
        assert set(exact) <= set(q), "index must never drop a contributor"
        assert idx.count(bounds) == len(exact)
        assert idx.count(bounds) == zal.count_contributors_exact(polys, bounds)


def test_eps_sliver_is_excluded_identically():
    """A sliver whose intersection area is <= EPS must never count."""
    polys = [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
             Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])]
    # A second polygon that only touches the first by an edge: 0-area contact.
    touching = [polys[0], Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])]
    rect = zg._rect(zg.GlobalBounds(0, 0, 10, 10))
    assert touching[1].intersection(rect).area == 0.0
    for backend in ("strtree", "sorted", "bruteforce"):
        idx = zal.FootprintIndex(touching, backend=backend)
        assert idx.count(zg.GlobalBounds(0, 0, 10, 10)) == 1
        assert idx.count(zg.GlobalBounds(0, 0, 10, 10)) == (
            zal.count_contributors_exact(touching, zg.GlobalBounds(0, 0, 10, 10))
        )


def test_empty_and_nonfinite_polygons_keep_historical_behaviour():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nan_poly = Polygon([(0, 0), (float("nan"), 1), (1, 1)])
    polys = [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
             Polygon(),        # empty
             nan_poly]         # non-finite vertex
    bounds = zg.GlobalBounds(0, 0, 10, 10)
    # The historical predicate RAISES on the non-finite polygon: that behaviour
    # must be preserved, never masked by the index.
    with pytest.raises(Exception) as ref_exc:
        zal.count_contributors_exact(polys, bounds)
    for backend in ("strtree", "sorted", "bruteforce"):
        idx = zal.FootprintIndex(polys, backend=backend)
        if backend == "bruteforce":
            assert idx.uncertain == 0          # nothing is indexed at all
        else:
            assert idx.uncertain == 2, "empty + non-finite must be re-checked"
        with pytest.raises(type(ref_exc.value)):
            idx.count(bounds)


def test_finite_invalid_polygon_scope_overlapping_and_bbox_disjoint():
    """rework-1 L3: the exact scope of the error/behaviour guarantee.

    A bowtie is FINITE, has a usable bbox and is INVALID. With the GEOS in use:
      * when the bbox OVERLAPS the rect, both paths raise the same GEOS error
        (the index returns the polygon, the predicate raises, `count` re-runs the
        historical sweep which raises);
      * when the bbox is DISJOINT, neither path raises (GEOS short-circuits a
        bbox-disjoint overlay to an empty geometry) and both count 0 for it.
    The guarantee claimed in the product is the exact COUNT plus the preserved
    error path for non-finite/uncertain geometry - not "no engine can ever raise".
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bowtie = Polygon([(0, 0), (1, 1), (0, 1), (1, 0)])        # invalid, finite
    assert not bowtie.is_valid and all(math.isfinite(v) for v in bowtie.bounds)
    square = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])

    # (a) bbox OVERLAPS: identical exception on every backend and on the reference.
    polys = [square, bowtie]
    overlap = zg.GlobalBounds(0, 0, 1, 1)
    with pytest.raises(Exception) as ref:
        zal.count_contributors_exact(polys, overlap)
    for backend in ("strtree", "sorted", "bruteforce"):
        idx = zal.FootprintIndex(polys, backend=backend)
        with pytest.raises(type(ref.value)):
            idx.count(overlap)

    # (b) bbox DISJOINT from the square's rect: count-exact, no raise on either path.
    far = zg.GlobalBounds(10, 10, 20, 20)
    ref_count = zal.count_contributors_exact(polys, far)
    assert ref_count == 1
    for backend in ("strtree", "sorted", "bruteforce"):
        idx = zal.FootprintIndex(polys, backend=backend)
        assert idx.count(far) == ref_count


def test_duplicate_polygons_are_counted_once_each():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    polys = [poly, poly, poly]
    for backend in ("strtree", "sorted", "bruteforce"):
        idx = zal.FootprintIndex(polys, backend=backend)
        assert idx.count(zg.GlobalBounds(0, 0, 10, 10)) == 3


# ---------------------------------------------------------------------------
# 4. Optional / fallback behaviour of the index
# ---------------------------------------------------------------------------

def _small_corpus():
    descs, canvas = _sparse_corpus(n=12, grid=3, side=70, step=0.03)
    footprints = zal.project_footprints(descs, canvas)
    return canvas, footprints


def test_strtree_build_failure_falls_back_to_sorted_exactly(monkeypatch):
    canvas, footprints = _small_corpus()
    bounds = [b for _r, _c, b in zg.build_layout(canvas, 4, 4).iter_cells(canvas)]

    def _boom(*a, **k):
        raise MemoryError("simulated accelerator failure")

    monkeypatch.setattr(zal, "_strtree", _boom)
    idx = zal.FootprintIndex(footprints.polygons)
    assert idx.backend == "sorted"
    assert idx.build_fallback and "MemoryError" in idx.build_fallback
    for b in bounds:
        assert idx.count(b) == zal.count_contributors_exact(footprints.polygons, b)


def test_query_failure_is_counted_and_counts_stay_exact(monkeypatch):
    canvas, footprints = _small_corpus()
    idx = zal.FootprintIndex(footprints.polygons)

    def _boom(rect):
        raise RuntimeError("simulated query failure")

    monkeypatch.setattr(idx, "_tree", SimpleNamespace(query=_boom))
    bounds = [b for _r, _c, b in zg.build_layout(canvas, 4, 4).iter_cells(canvas)]
    for b in bounds:
        assert idx.count(b) == zal.count_contributors_exact(footprints.polygons, b)
    assert idx.query_failures == len(bounds)
    assert idx.stats()["query_failures"] == len(bounds)


def test_index_refuses_a_foreign_footprint_artifact():
    canvas_a, fp_a = _small_corpus()
    descs_b, canvas_b = _sparse_corpus(n=8, grid=2, side=70, step=0.05)
    fp_b = zal.project_footprints(descs_b, canvas_b)
    idx_a = zal.FootprintIndex(fp_a.polygons)
    bounds = next(iter(zg.build_layout(canvas_b, 1, 1).iter_cells(canvas_b)))[2]
    with pytest.raises(ValueError):
        zal._cell_contributor_count(fp_b, canvas_b, bounds, index=idx_a)


def test_index_threshold_is_honoured_by_the_chooser(monkeypatch):
    """Below the threshold no index is built (the exact full scan is kept)."""
    descs, canvas = _halo_corpus()          # 20 frames
    built = []
    real_cls = zal.FootprintIndex

    class _Recorder(real_cls):
        def __init__(self, *a, **k):
            built.append((len(a[0]) if a else None, k))
            super().__init__(*a, **k)

    monkeypatch.setattr(zal, "FootprintIndex", _Recorder)
    assert 20 < zz.FOOTPRINT_INDEX_MIN_POLYGONS
    _choose(canvas, descs, ram_budget=10 * 2 ** 30, floors=_floors())
    assert built == []

    monkeypatch.setattr(zz, "FOOTPRINT_INDEX_MIN_POLYGONS", 1)
    _choose(canvas, descs, ram_budget=10 * 2 ** 30, floors=_floors())
    assert len(built) == 1, "exactly ONE index per layout choice"


# ---------------------------------------------------------------------------
# 5. REAL chooser parity: index ON vs OFF over every reachable path
# ---------------------------------------------------------------------------

def _parity(monkeypatch, canvas, descs, **kw):
    """Run the REAL chooser twice (index ON / OFF) and return both results."""
    monkeypatch.setattr(zz, "FOOTPRINT_INDEX_MIN_POLYGONS", 10 ** 9)
    off = _choose_or_refusal(canvas, descs, **kw)
    monkeypatch.setattr(zz, "FOOTPRINT_INDEX_MIN_POLYGONS", 0)
    on = _choose_or_refusal(canvas, descs, **kw)
    return off, on


def _decision_logs(logs):
    """Log lines EXCLUDING the dedicated (non-decision) index diagnostics."""
    return [l for l in logs if not l.startswith(zz.FOOTPRINT_INDEX_LOG_PREFIX)]


def _index_logs(logs):
    return [l for l in logs if l.startswith(zz.FOOTPRINT_INDEX_LOG_PREFIX)]


def _assert_same_decision(off, on):
    lay_off, err_off, logs_off = off
    lay_on, err_on, logs_on = on
    # The DECISION lines (layout scan: / layout load: / refusals) are verbatim
    # identical. The dedicated "layout index:" diagnostics are non-decision and
    # exist only when an index was built, so they are compared separately.
    assert _decision_logs(logs_on) == _decision_logs(logs_off)
    assert err_on == err_off
    assert lay_on == lay_off
    if lay_on is not None:
        assert (
            lay_on["nx"], lay_on["ny"], lay_on["layout_origin"],
            lay_on["cpu_streaming"], lay_on["floors"],
        ) == (
            lay_off["nx"], lay_off["ny"], lay_off["layout_origin"],
            lay_off["cpu_streaming"], lay_off["floors"],
        )


def test_parity_unconstrained_auto(monkeypatch):
    descs, canvas = _halo_corpus()
    off, on = _parity(monkeypatch, canvas, descs,
                      ram_budget=10 * 2 ** 30, floors=_floors())
    _assert_same_decision(off, on)
    assert on[0]["layout_origin"] == "auto_refinement"


def test_parity_tight_budget_forces_finer_grid(monkeypatch):
    descs, canvas = _halo_corpus()
    base, _, _ = _choose_or_refusal(
        canvas, descs, ram_budget=10 * 2 ** 30, floors=_floors())
    tight = max(1, int(base["predicted_bound_bytes"]) - 1)
    off, on = _parity(monkeypatch, canvas, descs, ram_budget=tight, floors=_floors())
    _assert_same_decision(off, on)
    assert on[0]["budget_bound_choice"] is True


def test_parity_pinned_layout(monkeypatch):
    descs, canvas = _halo_corpus()
    off, on = _parity(monkeypatch, canvas, descs, ram_budget=10 * 2 ** 30,
                      floors=_floors(), pinned_layout=(4, 3))
    _assert_same_decision(off, on)
    lay = on[0]
    assert lay["layout_source"] == "pinned" and lay["layout_origin"] == "pinned"
    # The pinned per-Cell predictions are the exact reference cells.
    footprints = zal.project_footprints(descs, canvas)
    assert lay["cells"] == _reference_cells(
        canvas, footprints, 4, 3, zz.zsw.HALO_PX, zz.STREAM_TILE_SIZE
    )


def test_parity_refinement_rank_window(monkeypatch):
    descs, canvas = _halo_corpus()
    off, on = _parity(monkeypatch, canvas, descs, ram_budget=10 * 2 ** 30,
                      floors=_floors(), min_refinement_rank=9,
                      max_refinement_rank=20)
    _assert_same_decision(off, on)
    assert on[0] is not None
    assert 9 <= on[0]["nx"] * on[0]["ny"] <= 20


def test_parity_coarser_rescue_large_halo(monkeypatch):
    descs, canvas = _halo_corpus()
    off, on = _parity(monkeypatch, canvas, descs, ram_budget=10 * 2 ** 30,
                      halo_px=128, floors=_floors(max_halo=0.5),
                      halo_policy=zhalo.HALO_OVERHEAD_POLICY_HARD_FLOOR)
    _assert_same_decision(off, on)
    lay, err, _logs = on
    assert (lay is None) != (err is None), "exactly one outcome"
    if lay is not None:
        assert lay["layout_origin"] == "auto_coarser_rescue"
        assert "layout_rescue" in lay


def test_parity_advisory_explicit_halo(monkeypatch):
    descs, canvas = _halo_corpus()
    off, on = _parity(monkeypatch, canvas, descs, ram_budget=10 * 2 ** 30,
                      halo_px=128, floors=_floors(max_halo=0.5),
                      halo_policy=zhalo.HALO_OVERHEAD_POLICY_ADVISORY)
    _assert_same_decision(off, on)
    lay = on[0]
    assert lay["halo_overhead_policy"] == zhalo.HALO_OVERHEAD_POLICY_ADVISORY
    assert lay["halo_overhead_advisory"]["geometry_changed"] is False


def test_parity_bounded_profile_joint(monkeypatch):
    descs, canvas = _halo_corpus()
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    off, on = _parity(monkeypatch, canvas, descs, ram_budget=6 * 2 ** 30,
                      floors=_floors())
    _assert_same_decision(off, on)


def test_parity_external_fallback(monkeypatch):
    descs, canvas = _compact_corpus()
    off, on = _parity(monkeypatch, canvas, descs, ram_budget=1,
                      floors=_floors(), allow_external=True)
    _assert_same_decision(off, on)
    assert on[0] is not None, "external keeps the finest floor-feasible grid"


def test_parity_infeasible_refusal(monkeypatch):
    descs, canvas = _compact_corpus()
    off, on = _parity(monkeypatch, canvas, descs, ram_budget=1, floors=_floors())
    _assert_same_decision(off, on)
    assert on[0] is None and on[1] is not None
    assert on[1]["refusal"]["kind"] == "host_ram"


def test_parity_load_aware_selection(monkeypatch):
    descs, canvas = _load_corpus()
    off, on = _parity(monkeypatch, canvas, descs, ram_budget=int(1e15),
                      floors=_floors())
    _assert_same_decision(off, on)
    lay = on[0]
    assert lay["membership_load"]["applied"] is True
    # The published provenance is JSON-native: no private cells leaked into it.
    assert "chosen_cells" not in lay["membership_load"]
    assert "chosen_state" not in lay["membership_load"]
    # ...and the manifest publication seam accepts it unchanged.
    block = zz._manifest_layout_block(lay["nx"], lay["ny"], lay)
    assert json.loads(json.dumps(block))["membership_load"]["applied"] is True


# ---------------------------------------------------------------------------
# 6. The chosen scan is REUSED (no redundant final pass), and only then
# ---------------------------------------------------------------------------

def _count_scans(monkeypatch):
    counts = {"scan": 0, "iter": 0}
    real_scan = zz._scan_layout_mode_aware
    real_iter = zz._iter_cell_bounds_mode_aware

    def scan(*a, **k):
        counts["scan"] += 1
        return real_scan(*a, **k)

    def it(*a, **k):
        counts["iter"] += 1
        return real_iter(*a, **k)

    monkeypatch.setattr(zz, "_scan_layout_mode_aware", scan)
    monkeypatch.setattr(zz, "_iter_cell_bounds_mode_aware", it)
    return counts


def test_chosen_grid_is_not_rescanned(monkeypatch):
    descs, canvas = _halo_corpus()
    counts = _count_scans(monkeypatch)
    lay, _logs = _choose(canvas, descs, ram_budget=10 * 2 ** 30, floors=_floors())
    assert counts["scan"] == 1 and counts["iter"] == 1
    assert (lay["nx"], lay["ny"]) == (4, 3)


def test_pinned_grid_is_scanned_once(monkeypatch):
    descs, canvas = _halo_corpus()
    counts = _count_scans(monkeypatch)
    _choose(canvas, descs, ram_budget=10 * 2 ** 30, floors=_floors(),
            pinned_layout=(2, 2))
    assert counts["scan"] == 0 and counts["iter"] == 1


def test_load_aware_pass_does_not_rescan_the_chosen_grid(monkeypatch):
    descs, canvas = _load_corpus()
    counts = _count_scans(monkeypatch)
    lay, _logs = _choose(canvas, descs, ram_budget=int(1e15), floors=_floors())
    assert lay["membership_load"]["applied"] is True
    # Every forward/family candidate is scanned exactly once, and the chosen
    # grid is NOT scanned a second time by the final cells pass.
    assert counts["iter"] == counts["scan"]


def test_load_aware_kept_grid_also_reuses_the_forward_scan(monkeypatch):
    descs, canvas = _halo_corpus()
    counts = _count_scans(monkeypatch)
    lay, _logs = _choose(canvas, descs, ram_budget=10 * 2 ** 30, floors=_floors())
    assert "membership_load" not in lay           # neutral signal: historical path
    assert counts["iter"] == counts["scan"] == 1


# ---------------------------------------------------------------------------
# 7. The prefilter really removes exact intersection operations
# ---------------------------------------------------------------------------

def test_index_reduces_exact_intersections_on_a_sparse_corpus():
    descs, canvas = _sparse_corpus(n=180, grid=18, side=90, step=0.03)
    footprints = zal.project_footprints(descs, canvas)
    index = zal.FootprintIndex(footprints.polygons)
    n_cells = 0
    ops = 0
    for row in range(9):
        for col in range(9):
            layout = zg.build_layout(canvas, 9, 9)
            bounds = layout.cell_bounds(row, col, canvas)
            geom = zg.patch_geometry(bounds, canvas, zz.zsw.HALO_PX)
            ops += len(index.query(zg._rect(geom.patch)))
            n_cells += 1
            assert zal._cell_contributor_count(
                footprints, canvas, geom.patch, index=index
            ) == zal._cell_contributor_count(footprints, canvas, geom.patch)
    brute_ops = n_cells * len(footprints)
    assert ops * 2 <= brute_ops, (ops, brute_ops)
    assert index.query_failures == 0


# ---------------------------------------------------------------------------
# 7b. Production OBSERVABILITY of the index (rework-1 L1)
# ---------------------------------------------------------------------------

def _run_with_index_logs(canvas, descs, monkeypatch, **kw):
    monkeypatch.setattr(zz, "FOOTPRINT_INDEX_MIN_POLYGONS", 0)
    logs: list = []
    emitted: list = []

    def _emit(msg, lvl="INFO"):
        emitted.append((msg, lvl))

    lay = zz._choose_layout_mode_aware(
        canvas, descs, log_line=logs.append, emit=_emit, footprints_out=[], **kw
    )
    return lay, logs, emitted


def test_index_diagnostic_is_emitted_for_build_and_completion(monkeypatch):
    descs, canvas = _halo_corpus()                      # 20 frames
    lay, logs, _emitted = _run_with_index_logs(
        canvas, descs, monkeypatch, ram_budget=10 * 2 ** 30, floors=_floors()
    )
    idx_lines = _index_logs(logs)
    assert len(idx_lines) == 2, idx_lines            # build + done
    assert any(" build " in l and "backend=strtree" in l for l in idx_lines)
    assert any(" done " in l and "queries=" in l for l in idx_lines)
    # It is NOT a decision line and it is NOT in the decision dict.
    assert not any(l.startswith(("layout scan:", "layout load:")) for l in idx_lines)
    assert "footprint_index" not in lay and "index" not in lay
    # Decision lines are exactly the historical ones (no index wording leaked).
    assert not any("index" in l for l in _decision_logs(logs))


def test_index_diagnostic_reports_build_fallback(monkeypatch):
    descs, canvas = _halo_corpus()

    def _boom(*a, **k):
        raise MemoryError("simulated accelerator failure")

    monkeypatch.setattr(zal, "_strtree", _boom)
    lay, logs, emitted = _run_with_index_logs(
        canvas, descs, monkeypatch, ram_budget=10 * 2 ** 30, floors=_floors()
    )
    build = next(l for l in _index_logs(logs) if " build " in l)
    assert "backend=sorted" in build
    assert "fallback=strtree: MemoryError" in build
    # A degraded backend is published as WARN, not as an INFO line.
    levels = {msg: lvl for msg, lvl in emitted
              if msg.startswith(zz.FOOTPRINT_INDEX_LOG_PREFIX) and " build " in msg}
    assert levels.get(build) == "WARN", levels
    assert lay["nx"] and lay["ny"]               # the decision still succeeded


def test_index_diagnostic_reports_query_failures_and_keeps_the_decision(monkeypatch):
    descs, canvas = _halo_corpus()
    real_cls = zal.FootprintIndex

    class _Flaky(real_cls):
        """First query fails (falls back to the full exact sweep), then normal."""

        def _query_impl(self, rect):
            if not getattr(self, "_failed_once", False):
                self._failed_once = True
                raise RuntimeError("simulated query failure")
            return super()._query_impl(rect)

    monkeypatch.setattr(zal, "FootprintIndex", _Flaky)
    lay, logs, emitted = _run_with_index_logs(
        canvas, descs, monkeypatch, ram_budget=10 * 2 ** 30, floors=_floors()
    )
    done = next(l for l in _index_logs(logs) if " done " in l)
    assert "query_failures=1" in done
    levels = {msg: lvl for msg, lvl in emitted
              if msg.startswith(zz.FOOTPRINT_INDEX_LOG_PREFIX) and " done " in msg}
    assert levels.get(done) == "WARN", levels
    # The failing query fell back to the exact full sweep: same decision.
    monkeypatch.setattr(zal, "FootprintIndex", real_cls)
    monkeypatch.setattr(zz, "FOOTPRINT_INDEX_MIN_POLYGONS", 10 ** 9)
    off = _choose_or_refusal(canvas, descs, ram_budget=10 * 2 ** 30, floors=_floors())
    assert off[0] == lay


def test_index_diagnostic_is_emitted_on_refusal(monkeypatch):
    descs, canvas = _compact_corpus()
    monkeypatch.setattr(zz, "FOOTPRINT_INDEX_MIN_POLYGONS", 0)
    logs = []
    with pytest.raises(zal.LayoutInfeasible):
        zz._choose_layout_mode_aware(
            canvas, descs, ram_budget=1, floors=_floors(), log_line=logs.append
        )
    idx_lines = _index_logs(logs)
    assert len(idx_lines) == 2, idx_lines
    assert " refusal " in idx_lines[-1] and "queries=" in idx_lines[-1]


# ---------------------------------------------------------------------------
# 8. Product propagation: the published cells drive the REAL cell contexts
# ---------------------------------------------------------------------------

def test_chosen_cells_propagate_to_the_cell_contexts():
    descs, canvas = _sparse_corpus(n=60, grid=6, side=90, step=0.03)
    fp_out = []
    lay = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=10 * 2 ** 30, floors=_floors(),
        footprints_out=fp_out,
    )
    assert len(fp_out) == 1
    ctxs = zsw.build_cell_contexts_from_footprints(
        fp_out[0], canvas, lay["nx"], lay["ny"], zz.zsw.HALO_PX
    )
    assert len(ctxs) == lay["nx"] * lay["ny"] == len(lay["cells"])
    for (row, col, cell, patch, mem), pred in zip(ctxs, lay["cells"]):
        assert (row, col) == (pred["row"], pred["col"])
        geom = zg.patch_geometry(cell.core, canvas, zz.zsw.HALO_PX)
        assert patch.patch == geom.patch
        assert patch.patch_shape_hw == tuple(pred["patch_hw"])
        assert len(mem.patch_ids) == pred["n"]
        assert pred["patch_area"] == patch.patch.width * patch.patch.height
    # The chosen layout is a real, reproducible decision: re-running it on the
    # same inputs and reusing the same once-projected artifact is identical.
    again = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=10 * 2 ** 30, floors=_floors(),
        footprints_in=fp_out[0],
    )
    assert again == lay
