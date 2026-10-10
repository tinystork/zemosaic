"""ZM-ZEGRID-R28 targeted tests — layout performance (single footprint pass +
early-stop candidate scan) without changing science.

All tests are SYNTHETIC and fast (no user data, no M16/M106 fixtures, no pixels,
no multiprocessing). They prove, at the production seam:

* each frame's WCS footprint is projected EXACTLY ONCE per layout run (N, not
  2*N and not N*cells) for both the auto and pinned paths, and the final
  chosen-cell memberships reuse those SAME polygons;
* exact public parity + core_ids/patch_ids order parity against a test-local
  exhaustive (pre-R28) reference on multiple TAN corpora and a synthetic SIP
  (curved-footprint) corpus;
* the candidate scan stops at the FIRST floor-feasible + memory-admissible
  candidate, retains prior rejected candidates for ``budget_bound_choice``,
  keeps the no-fit error informative, and preserves the floor break;
* no-observer/default API compatibility, no private geometry objects in the
  public layout dict, the honest-N footprint progress, a direct chosen-cells
  START/END/timing assertion, and the unchanged generic
  ``compute_membership``/``build_cell_context`` path.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from shapely.geometry.base import BaseGeometry

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import observability as zobs
from zemosaic.core.zegrid import sweep as zsw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.emitted = []
        self.log_lines = []

    def emit(self, msg, lvl="INFO"):
        self.emitted.append((msg, lvl))

    def log_line(self, line):
        self.log_lines.append(line)


def _tiny_floors():
    # Relax the scientific floors so a tiny synthetic canvas is feasible (the
    # floors are irrelevant to what these tests assert — geometry/perf parity).
    return zal.ScientificFloors(
        min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9
    )


def _make_tan_descriptor(i, h, w, scale, ra_step):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0 + i * ra_step, 30.0]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
    wcs.array_shape = (h, w)
    fid = f"d{i}.fits"
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid),
        source_path=f"/x/{fid}",
        shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(),
        header_sha256="",
        instrument="",
    )


def _make_sip_descriptor(i, h, w, scale, ra_step):
    hd = fits.Header()
    hd["NAXIS"] = 2
    hd["NAXIS1"] = w
    hd["NAXIS2"] = h
    hd["CTYPE1"] = "RA---TAN-SIP"
    hd["CTYPE2"] = "DEC--TAN-SIP"
    hd["CRVAL1"] = 10.0 + i * ra_step
    hd["CRVAL2"] = 30.0
    hd["CRPIX1"] = w / 2.0
    hd["CRPIX2"] = h / 2.0
    hd["CD1_1"] = -scale
    hd["CD1_2"] = 0.0
    hd["CD2_1"] = 0.0
    hd["CD2_2"] = scale
    # 2nd-order SIP distortion (A_ORDER=B_ORDER=2) -> curved projected edges.
    hd["A_ORDER"] = 2
    hd["B_ORDER"] = 2
    hd["A_0_0"] = 0.0
    hd["B_0_0"] = 0.0
    hd["A_0_2"] = 1e-4
    hd["A_2_0"] = 1e-4
    hd["B_0_2"] = 1e-4
    hd["B_2_0"] = 1e-4
    wcs = WCS(hd)
    fid = f"s{i}.fits"
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid),
        source_path=f"/x/{fid}",
        shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(),
        header_sha256="",
        instrument="",
    )


def _build_tan_corpus(n=4, h=40, w=40, scale=0.001, ra_step=0.004):
    return [_make_tan_descriptor(i, h, w, scale, ra_step) for i in range(n)]


def _build_sip_corpus(n=4, h=40, w=40, scale=0.001, ra_step=0.004):
    return [_make_sip_descriptor(i, h, w, scale, ra_step) for i in range(n)]


def _tan_corpora():
    """Several distinct synthetic TAN corpora for the parity sweep."""
    return [
        _build_tan_corpus(n=4, h=40, w=40, scale=0.001, ra_step=0.004),
        _build_tan_corpus(n=5, h=64, w=48, scale=0.002, ra_step=0.005),
        _build_tan_corpus(n=6, h=50, w=60, scale=0.0015, ra_step=0.003),
    ]


def _spy_source_polygon(monkeypatch):
    """Monkeypatch ``zg._source_polygon`` and return a call counter list."""
    real = zg._source_polygon
    calls = []

    def spy(shape_hw, src_wcs, tgt_wcs):
        calls.append(1)
        return real(shape_hw, src_wcs, tgt_wcs)

    monkeypatch.setattr(zg, "_source_polygon", spy)
    return calls


def _reference_exhaustive_layout(canvas, frames, ram_budget, floors):
    """Test-local reference reproducing the PRE-R28 exhaustive scan semantics.

    Projects footprints TWICE (``median_projected_footprint`` + ``_footprints``,
    both still present) and scans ALL refinement factors without early stop, then
    picks the first memory-admissible candidate. Used only to prove the R28 path
    returns the identical decision.
    """
    mw, mh = zal.median_projected_footprint(frames, canvas)
    footprints = zal._footprints(frames, canvas)
    candidates = []
    seen = set()
    for factor in zal.REFINEMENT_FACTORS:
        nx = max(1, int(math.ceil(canvas.width / (mw / factor))))
        ny = max(1, int(math.ceil(canvas.height / (mh / factor))))
        nx = min(nx, canvas.width)
        ny = min(ny, canvas.height)
        if (nx, ny) in seen:
            continue
        seen.add((nx, ny))
        geom = zal._nominal_geometry(canvas, nx, ny, zsw.HALO_PX)
        if not (geom["min_patch_area"] >= floors.min_patch_area_px and
                geom["halo_overhead"] <= floors.max_halo_overhead):
            break
        worst = zz._scan_layout_mode_aware(canvas, footprints, nx, ny, zsw.HALO_PX, zz.STREAM_TILE_SIZE)
        memory_ok = (ram_budget is None) or (worst["cheaper_bound_bytes"] <= ram_budget)
        candidates.append({"factor": factor, "nx": nx, "ny": ny,
                           "bound": worst["cheaper_bound_bytes"], "memory_ok": memory_ok})

    chosen = None
    chosen_index = None
    for i, c in enumerate(candidates):
        if c["memory_ok"]:
            chosen = c
            chosen_index = i
            break
    if chosen is None:
        return None

    constrained = False
    if ram_budget is not None and chosen_index is not None:
        for c in candidates[:chosen_index]:
            if not c["memory_ok"]:
                constrained = True
                break
    return {
        "nx": chosen["nx"], "ny": chosen["ny"], "factor": chosen["factor"],
        "bound": chosen["bound"], "constrained": constrained,
    }


def _walk(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def _is_geometry_object(v):
    return isinstance(v, (BaseGeometry, WCS))


# ---------------------------------------------------------------------------
# 1. Projection count: exactly N (not 2N, not N*cells), auto + pinned
# ---------------------------------------------------------------------------

def test_projection_count_exact_n_auto_and_pinned(monkeypatch):
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)  # canvas built BEFORE the spy (not counted)
    calls = _spy_source_polygon(monkeypatch)

    # Auto path through the real layout + final-context seam.
    out = []
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1e15), floors=_tiny_floors(), footprints_out=out,
    )
    assert len(out) == 1, "footprints_out must receive exactly one artifact"
    ctxs = zsw.build_cell_contexts_from_footprints(out[0], canvas, layout["nx"], layout["ny"])
    assert len(ctxs) == layout["nx"] * layout["ny"]
    assert len(calls) == len(descs), "auto layout projected footprints != exactly N"

    # Pinned path also projects exactly N (no per-cell re-projection).
    calls.clear()
    out2 = []
    layout2 = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1e15), floors=_tiny_floors(),
        pinned_layout=(2, 2), footprints_out=out2,
    )
    ctxs2 = zsw.build_cell_contexts_from_footprints(out2[0], canvas, layout2["nx"], layout2["ny"])
    assert len(ctxs2) == layout2["nx"] * layout2["ny"]
    assert len(calls) == len(descs), "pinned layout projected footprints != exactly N"


def test_sip_projection_count_exact_n(monkeypatch):
    descs = _build_sip_corpus()
    canvas = zg.build_canvas(descs)
    calls = _spy_source_polygon(monkeypatch)
    out = []
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1e15), floors=_tiny_floors(), footprints_out=out,
    )
    zsw.build_cell_contexts_from_footprints(out[0], canvas, layout["nx"], layout["ny"])
    assert len(calls) == len(descs), "SIP layout projected footprints != exactly N"


# ---------------------------------------------------------------------------
# 2. Exact public parity + membership order parity (TAN corpora, SIP corpus)
# ---------------------------------------------------------------------------

def test_exact_parity_against_exhaustive_reference():
    for descs in _tan_corpora():
        canvas = zg.build_canvas(descs)
        floors = _tiny_floors()
        unconstrained = zz._choose_layout_mode_aware(
            canvas, descs, ram_budget=int(1e15), floors=floors, footprints_out=[],
        )
        # Unconstrained + constrained (budget just below the coarsest bound).
        budgets = [int(1e15), max(1, unconstrained["predicted_bound_bytes"] - 1)]
        for budget in budgets:
            out = []
            layout = zz._choose_layout_mode_aware(
                canvas, descs, ram_budget=budget, floors=floors, footprints_out=out,
            )
            ref = _reference_exhaustive_layout(canvas, descs, budget, floors)
            assert ref is not None
            # Public layout data identical to the exhaustive reference.
            assert (layout["nx"], layout["ny"]) == (ref["nx"], ref["ny"])
            assert layout["refinement_factor"] == ref["factor"]
            assert layout["predicted_bound_bytes"] == int(ref["bound"])
            assert layout["budget_bound_choice"] == ref["constrained"]

            # core_ids/patch_ids order identical to the exhaustive
            # build_cell_context path (which re-projects WCS).
            ctxs = zsw.build_cell_contexts_from_footprints(out[0], canvas, layout["nx"], layout["ny"])
            ref_ctxs = []
            for row, col, _b in zg.build_layout(canvas, layout["nx"], layout["ny"]).iter_cells(canvas):
                ref_ctxs.append(zsw.build_cell_context(descs, canvas, row, col, layout["nx"], layout["ny"]))
            assert len(ctxs) == len(ref_ctxs)
            for (nr, nc, ncell, npatch, nmem), (rcell, rpatch, rmem) in zip(ctxs, ref_ctxs):
                assert (nr, nc) == (rcell.row, rcell.col)
                assert ncell.cell_id == rcell.cell_id
                assert nmem.cell_id == rmem.cell_id
                assert nmem.core_ids == rmem.core_ids
                assert nmem.patch_ids == rmem.patch_ids
            # Per-cell contributor counts in the layout dict == membership.
            assert [c["n"] for c in layout["cells"]] == [
                len(m.patch_ids) for (_r, _c, _ce, _p, m) in ctxs
            ]


def test_sip_curved_footprint_parity():
    descs = _build_sip_corpus()
    canvas = zg.build_canvas(descs)
    out = []
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1e15), floors=_tiny_floors(), footprints_out=out,
    )
    fp = out[0]
    # The SIP footprint path produced curved (non-4-corner) polygons.
    assert any(len(p.exterior.coords) > 4 for p in fp.polygons), \
        "SIP footprint did not exercise the curved-edge sampling path"

    ctxs = zsw.build_cell_contexts_from_footprints(fp, canvas, layout["nx"], layout["ny"])
    ref_ctxs = []
    for row, col, _b in zg.build_layout(canvas, layout["nx"], layout["ny"]).iter_cells(canvas):
        ref_ctxs.append(zsw.build_cell_context(descs, canvas, row, col, layout["nx"], layout["ny"]))
    assert len(ctxs) == len(ref_ctxs)
    for (nr, nc, ncell, npatch, nmem), (rcell, rpatch, rmem) in zip(ctxs, ref_ctxs):
        assert ncell.cell_id == rcell.cell_id
        assert nmem.core_ids == rmem.core_ids
        assert nmem.patch_ids == rmem.patch_ids


# ---------------------------------------------------------------------------
# 3. Early stop + budget_bound_choice + no-fit + floors
# ---------------------------------------------------------------------------

def test_early_stop_at_first_admissible_candidate(monkeypatch):
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    real_scan = zz._scan_layout_mode_aware
    scanned = []

    def spy(canvas, footprints, nx, ny, halo_px, tile_size):
        scanned.append((nx, ny))
        return real_scan(canvas, footprints, nx, ny, halo_px, tile_size)

    monkeypatch.setattr(zz, "_scan_layout_mode_aware", spy)

    # Unconstrained: the coarsest candidate is admissible -> exactly one scan.
    layout = zz._choose_layout_mode_aware(canvas, descs, ram_budget=int(1e15), floors=_tiny_floors())
    assert len(scanned) == 1, "unconstrained scan did not stop at the first admissible candidate"
    assert scanned[0] == (layout["nx"], layout["ny"])

    # Constrained: coarsest rejected, scan continues only to the first admissible.
    scanned.clear()
    unconstrained = zz._choose_layout_mode_aware(canvas, descs, ram_budget=int(1e15), floors=_tiny_floors())
    tight = max(1, unconstrained["predicted_bound_bytes"] - 1)
    layout2 = zz._choose_layout_mode_aware(canvas, descs, ram_budget=tight, floors=_tiny_floors())
    assert len(scanned) >= 2, "constrained scan did not evaluate the rejected coarser candidates"
    assert scanned[-1] == (layout2["nx"], layout2["ny"]), "last scanned is not the chosen candidate"


def test_budget_bound_choice_constrained_flag_retains_prior_rejects():
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    floors = _tiny_floors()

    unconstrained = zz._choose_layout_mode_aware(canvas, descs, ram_budget=int(1e15), floors=floors)
    assert unconstrained["budget_bound_choice"] is False

    tight = max(1, unconstrained["predicted_bound_bytes"] - 1)
    msgs = []
    constrained = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=tight, floors=floors,
        emit=lambda m, lvl="INFO": msgs.append(m),
    )
    assert constrained["budget_bound_choice"] is True
    assert constrained["cell_count"] >= unconstrained["cell_count"]

    scan = [m for m in msgs if m.startswith("layout scan:")]
    rejected = [m for m in scan if "REJECTED" in m]
    accepted = [m for m in scan if "ACCEPTED" in m]
    assert rejected, "prior rejected candidate was not retained/logged"
    assert any("bound" in m for m in rejected), "prior rejection was not memory-bound"
    assert len(accepted) == 1, "scan did not stop after the first admissible candidate"


def test_no_fit_error_informative():
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    with pytest.raises(zal.LayoutInfeasible) as exc:
        zz._choose_layout_mode_aware(canvas, descs, ram_budget=1, floors=_tiny_floors())
    msg = str(exc.value)
    assert "MiB" in msg
    assert "zegrid_layout" in msg
    assert "budget" in msg.lower()


def test_floors_break_behavior_preserved():
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    msgs = []
    with pytest.raises(zal.LayoutInfeasible):
        zz._choose_layout_mode_aware(
            canvas, descs, ram_budget=int(1e15),
            emit=lambda m, lvl="INFO": msgs.append(m),
        )
    scan = [m for m in msgs if m.startswith("layout scan:")]
    assert scan, "no layout scan lines emitted"
    assert any("min_patch_area" in m for m in scan), "floor rejection reason not logged"
    assert any("stopping" in m for m in scan), "monotonic floor stop not logged"


# ---------------------------------------------------------------------------
# 4. No-observer/default API compatibility + no private geometry leak
# ---------------------------------------------------------------------------

def test_no_observer_default_api_compatibility():
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    layout = zz._choose_layout_mode_aware(canvas, descs, ram_budget=int(1e15), floors=_tiny_floors())
    for key in (
        "nx", "ny", "cell_count", "ram_budget_bytes", "available_bytes",
        "max_contributors", "refinement_factor", "predicted_bound_bytes",
        "max_patch_area", "budget_bound_choice", "warnings", "cells",
        "layout_source", "floors",
    ):
        assert key in layout, f"missing public key {key}"

    # Observer/emit/log_line seam must not change the returned public layout.
    msgs = []
    log_lines = []
    obs = zobs.SubstepReporter(msgs.append, log_line=log_lines.append)
    observed = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1e15), floors=_tiny_floors(),
        emit=lambda m, lvl="INFO": msgs.append(m),
        log_line=log_lines.append, observer=obs,
    )
    assert observed == layout


def test_no_private_geometry_leaks_into_public_dict():
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    out = []
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1e15), floors=_tiny_floors(), footprints_out=out,
    )
    # The public layout dict is fully JSON-serializable (no Shapely/WCS objects).
    json.dumps(layout)
    for value in _walk(layout):
        assert not _is_geometry_object(value), f"private geometry object leaked: {type(value)!r}"
    assert len(out) == 1
    assert "footprints" not in layout and "polygons" not in layout


# ---------------------------------------------------------------------------
# 5. R27 observability compatibility: honest-N + chosen_cells production seam
# ---------------------------------------------------------------------------

def test_layout_substep_progress_honest_n():
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    rec = _Recorder()
    obs = zobs.SubstepReporter(rec.emit, log_line=rec.log_line)
    zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1e15), floors=_tiny_floors(),
        emit=rec.emit, log_line=rec.log_line, observer=obs,
    )
    starts = [m for m, lvl in rec.emitted if m.startswith("substep START:")]
    fp_start = [s for s in starts if zobs.LAYOUT_SUBSTEP_FOOTPRINTS in s][0]
    assert f"total={len(descs)}" in fp_start
    assert f"total={2 * len(descs)}" not in fp_start, "footprint total still reports 2*N"
    fp_progress = [
        m for m, lvl in rec.emitted
        if m.startswith("substep PROGRESS:") and zobs.LAYOUT_SUBSTEP_FOOTPRINTS in m
    ]
    assert fp_progress, "no footprint progress emitted"


def test_chosen_cells_production_seam_start_end_timing():
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    out = []
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1e15), floors=_tiny_floors(), footprints_out=out,
    )
    rec = _Recorder()
    obs = zobs.SubstepReporter(rec.emit, log_line=rec.log_line)
    ctxs = zz._build_chosen_cell_contexts(out[0], canvas, layout["nx"], layout["ny"], obs)
    assert len(ctxs) == layout["nx"] * layout["ny"]

    t = obs.timings()
    assert zobs.LAYOUT_SUBSTEP_CHOSEN_CELLS in t
    assert t[zobs.LAYOUT_SUBSTEP_CHOSEN_CELLS] >= 0.0

    starts = [m for m, lvl in rec.emitted if m.startswith("substep START:")]
    ends = [m for m, lvl in rec.emitted if m.startswith("substep END:")]
    assert any(zobs.LAYOUT_SUBSTEP_CHOSEN_CELLS in s for s in starts)
    assert any(zobs.LAYOUT_SUBSTEP_CHOSEN_CELLS in e for e in ends)
    assert any("elapsed=" in e for e in ends)

    progress = [m for m, lvl in rec.emitted if m.startswith("substep PROGRESS:")]
    total = layout["nx"] * layout["ny"]
    assert any(f"{total}/{total}" in p for p in progress), "chosen-cell final sample missing"


def test_stable_substep_ids_and_no_stage_progress_regression():
    assert zobs.LAYOUT_SUBSTEP_FOOTPRINTS == "layout:footprints"
    assert zobs.LAYOUT_SUBSTEP_CANDIDATES == "layout:candidates"
    assert zobs.LAYOUT_SUBSTEP_CHOSEN_CELLS == "layout:chosen_cells"

    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    rec = _Recorder()
    obs = zobs.SubstepReporter(rec.emit, log_line=rec.log_line)
    zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1e15), floors=_tiny_floors(),
        emit=rec.emit, log_line=rec.log_line, observer=obs,
    )
    # Substep reporter never drives the GUI stage/percent (no STAGE_PROGRESS).
    for msg, lvl in rec.emitted:
        assert not msg.startswith("phase PROGRESS:")
        assert "STAGE_PROGRESS" not in msg


# ---------------------------------------------------------------------------
# 6. Generic compute_membership / build_cell_context remain unchanged
# ---------------------------------------------------------------------------

def test_generic_compute_membership_and_build_cell_context_unchanged(monkeypatch):
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    calls = _spy_source_polygon(monkeypatch)

    layout = zg.build_layout(canvas, 2, 2)
    cell = zg.ZeGridCell("r0000c0000", canvas.canvas_id, layout.layout_id, 0, 0,
                         layout.cell_bounds(0, 0, canvas))
    patch = zg.build_patch(canvas, cell, zsw.HALO_PX)
    mem = zg.compute_membership(descs, canvas, cell, patch)
    # The generic path STILL projects every frame's WCS (unchanged).
    assert len(calls) == len(descs)

    calls.clear()
    cell2, patch2, mem2 = zsw.build_cell_context(descs, canvas, 0, 0, 2, 2)
    assert len(calls) == len(descs), "build_cell_context no longer re-projects (behavior changed)"
    assert mem.core_ids == mem2.core_ids
    assert mem.patch_ids == mem2.patch_ids
