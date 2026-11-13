"""ZM-ZEGRID-LAYOUT-LOAD-BALANCE — membership-load aware layout choice.

The historical Auto policy ranks candidate grids by PEAK MEMORY and takes the
FIRST (coarsest) admissible one. It never looks at how many frame MEMBERSHIPS a
Cell carries, so a layout can be memory-optimal and load-pathological: on the
real M33 geometry (21 818 frames, 170 Cells) the measured load was 298 893
membership slots (amplification 13.7x) with mean 1 758 slots/Cell and a maximum
of 21 804 (12.4x the mean). Each membership is reprojected and reduced, and the
external lane writes/re-reads one spool per membership, so the tail Cells set
the makespan.

These tests are SYNTHETIC and fast (no user data, no pixels, no multiprocessing).
They pin:

* the pure scoring rule (a pinned M33-shaped candidate table: a dominating grid
  is selected, a tail-costly grid and a total-costly grid are refused);
* the production-path load pass (pinned per-candidate tables, the chosen reason,
  the published log lines, a truthful no-op, deterministic replay, input-order
  independence);
* the fail-closed guards: no grid below ``min_patch_area``, no grid above the
  halo floor, no grid outside the RAM envelope, and the tail never grows;
* R28 (exactly N footprint projections, also when the load pass engages);
* legacy parity: a neutral load signal keeps the historical grid byte for byte
  and publishes NOTHING new; a pinned layout is untouched;
* additive, JSON-native publication inside the EXISTING ``layout`` block.
"""

from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest
from astropy.wcs import WCS
from shapely.geometry.base import BaseGeometry

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg


# ---------------------------------------------------------------------------
# Synthetic corpora (no user data, no pixels)
# ---------------------------------------------------------------------------

def _frame(i, h, w, scale, ra, dec):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
    wcs.array_shape = (h, w)
    fid = f"d{i:05d}.fits"
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid),
        source_path=f"/x/{fid}",
        shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(),
        header_sha256="",
        instrument="",
    )


def _padded(canvas, factor):
    """Canvas larger than the sky union (empty Cells are REAL Cells).

    The real M33 layout has Cells with zero memberships (measured min = 0), so a
    canvas whose union covers only a fraction of it is the representative case,
    not an artefact.
    """
    return zg.GlobalCanvas(
        canvas_id=f"{canvas.canvas_id}-pad{factor}",
        wcs_header=canvas.wcs_header,
        width=int(canvas.width * factor),
        height=int(canvas.height * factor),
        resolution_deg=canvas.resolution_deg,
    )


def _core_corpus(core_n, core_step, core_reps, halo_n, halo_step, halo_reps, pad):
    """A dense deep core plus a sparse halo — the M33 shape.

    The core is a quasi-regular dither lattice (like the real nights) and the
    halo places frames across the rest of the canvas so the load is spread.
    """
    descs = []
    i = 0
    for ix in range(core_n):
        for iy in range(core_n):
            for r in range(core_reps):
                descs.append(_frame(
                    i, 80, 80, 0.001,
                    10.0 + (ix - core_n / 2) * core_step + 2e-4 * r,
                    30.0 + (iy - core_n / 2) * core_step + 2e-4 * r,
                ))
                i += 1
    for ix in range(halo_n):
        for iy in range(halo_n):
            for r in range(halo_reps):
                descs.append(_frame(
                    i, 80, 80, 0.001,
                    10.0 + (ix - halo_n / 2) * halo_step + 3e-4 * r,
                    30.0 + (iy - halo_n / 2) * halo_step + 3e-4 * r,
                ))
                i += 1
    canvas = _padded(zg.build_canvas(descs), pad)
    return descs, canvas


def _floors():
    # Floors relaxed (this file is about load policy, not floor policy); the
    # floor guards are exercised explicitly in their own test.
    return zal.ScientificFloors(
        min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9
    )


_CORPUS_PARAMS = {
    # 208 frames, baseline 4x4 (16 Cells): tail pinned at ALL frames, total
    # reducible -> the production path flattens the AGGREGATE load.
    "A": dict(core_n=8, core_step=0.010, core_reps=3, halo_n=4, halo_step=0.03,
              halo_reps=1, pad=1.6),
    # 1380 frames, baseline 6x7 (42 Cells): the tail is NOT pinned, and a finer
    # neighbour shortens it within the published exchange rate.
    "BIG": dict(core_n=16, core_step=0.010, core_reps=5, halo_n=10, halo_step=0.03,
                halo_reps=1, pad=1.4),
    # 636 frames, baseline 5x5: every alternative pays more than the tail it
    # saves -> the historical grid is kept, truthfully, with a reason.
    "C": dict(core_n=10, core_step=0.012, core_reps=6, halo_n=6, halo_step=0.04,
              halo_reps=1, pad=1.4),
}


@pytest.fixture(scope="module")
def corpus_a():
    return _core_corpus(**_CORPUS_PARAMS["A"])


@pytest.fixture(scope="module")
def corpus_big():
    return _core_corpus(**_CORPUS_PARAMS["BIG"])


@pytest.fixture(scope="module")
def corpus_c():
    return _core_corpus(**_CORPUS_PARAMS["C"])


@pytest.fixture(scope="module")
def tiny_corpus():
    descs = [_frame(i, 40, 40, 0.001, 10.0 + i * 0.004, 30.0) for i in range(4)]
    return descs, zg.build_canvas(descs)


def _run(canvas, descs, *, budget=int(1e15), floors=None, emit=None, **kw):
    return zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=budget, floors=floors or _floors(),
        emit=emit, **kw,
    )


# ---------------------------------------------------------------------------
# 1. The rule itself — a pinned M33-shaped candidate table
# ---------------------------------------------------------------------------

def test_rule_selects_dominating_candidate_on_pinned_m33_table():
    """Area-balanced grid has a huge tail; the dominator wins on BOTH metrics.

    Numbers are the measured M33 signature (21 818 frames, 170 Cells, 298 893
    slots, max 21 804, mean 1 758 -> max/mean 12.40) against a candidate that
    carries fewer total slots AND a shorter tail.
    """
    # Exact measured shape: 170 Cells, 298 893 slots, one Cell at 21 804 (the
    # whole field), the rest summing to 277 089.
    counts = [21804] + [1640] * 98 + [1639] * 71
    baseline = zal.membership_load(12, 20, counts)
    assert baseline.cells == 170
    assert baseline.total_slots == 298893
    assert baseline.max_slots == 21804
    assert baseline.mean_slots == pytest.approx(298893 / 170, abs=1e-9)
    assert baseline.mean_slots == pytest.approx(1758.19411765, abs=1e-6)
    assert baseline.imbalance == pytest.approx(12.4014, abs=1e-4)
    assert baseline.score() == 298893 + 4 * 21804 == 386109

    dominating = zal.MembershipLoad(
        nx=14, ny=18, cells=252, total_slots=291600, max_slots=15900,
        mean_slots=291600 / 252,
    )
    assert dominating.total_slots < baseline.total_slots
    assert dominating.max_slots < baseline.max_slots
    assert zal.better_load(dominating, baseline) is True
    assert dominating.score() == 291600 + 4 * 15900 == 355200

    # REFUSED: a shorter tail paid for with too much aggregate work (the price
    # exceeds the published 4:1 exchange: 340 000 > 298 893 + 4*(21 804-12 000)).
    expensive_tail = zal.MembershipLoad(
        nx=16, ny=18, cells=288, total_slots=340000, max_slots=12000,
        mean_slots=340000 / 288,
    )
    assert zal.load_candidate_is_acceptable(expensive_tail, baseline) is False
    assert zal.better_load(expensive_tail, baseline) is False

    # REFUSED: a cheaper aggregate that LENGTHENS the tail.
    longer_tail = zal.MembershipLoad(
        nx=8, ny=16, cells=128, total_slots=280000, max_slots=23000,
        mean_slots=280000 / 128,
    )
    assert zal.better_load(longer_tail, baseline) is False

    # REFUSED: an identical grid is not an improvement.
    assert zal.better_load(baseline, baseline) is False


def test_pure_load_helpers_are_exact_and_pure():
    load = zal.membership_load(3, 4, [5, 0, 9, 0])
    assert (load.nx, load.ny, load.cells) == (3, 4, 4)
    assert load.total_slots == 14
    assert load.max_slots == 9
    assert load.mean_slots == pytest.approx(3.5)
    assert load.imbalance == pytest.approx(9 / 3.5)
    assert isinstance(load.score(), int) and load.score() == 14 + 4 * 9

    empty = zal.membership_load(1, 1, [0])
    assert empty.imbalance == 0.0
    assert empty.score() == 0
    assert zal.load_signal_is_material(empty) is False

    # Neutral signal: every Cell sees every frame (the legacy / tiny corpus).
    flat = zal.membership_load(2, 2, [7, 7, 7, 7])
    assert flat.imbalance == 1.0
    assert zal.load_signal_is_material(flat) is False

    # Material: > LOAD_IMBALANCE_TRIGGER.
    skewed = zal.membership_load(2, 2, [40, 2, 1, 1])
    assert zal.load_signal_is_material(skewed) is True
    assert zal.LOAD_IMBALANCE_TRIGGER == 2.0
    assert zal.LOAD_TAIL_WEIGHT == 4

    # Deterministic payload: JSON-native, no geometry objects.
    payload = json.loads(json.dumps(skewed.to_dict()))
    assert payload["total_slots"] == 44 and payload["max_slots"] == 40


# ---------------------------------------------------------------------------
# 2. Production path — aggregate flattened at a pinned tail
# ---------------------------------------------------------------------------

def test_production_path_reduces_total_slots_at_pinned_tail(corpus_a):
    descs, canvas = corpus_a
    msgs = []
    layout = _run(canvas, descs, emit=lambda m, lvl="INFO": msgs.append(m))
    block = layout["membership_load"]

    # Published baseline = the historical (first-admissible) grid, 4x4.
    assert block["baseline"] == {
        "nx": 4, "ny": 4, "cells": 16, "total_slots": 1164, "max_slots": 208,
        "mean_slots": 72.75, "max_over_mean": 2.859107, "score": 1996,
    }
    # Chosen: 4x3 — 16% fewer slots, the SAME (input-ceiling) tail.
    assert block["chosen"] == {
        "nx": 4, "ny": 3, "cells": 12, "total_slots": 975, "max_slots": 208,
        "mean_slots": 81.25, "max_over_mean": 2.56, "score": 1807,
    }
    assert block["applied"] is True
    assert block["reason"] == "load_balanced_lower_total_slots"
    assert block["n_frames"] == 208
    # A Cell is a member of EVERY frame -> the tail is at its structural floor
    # for ANY partition, so the only reducible quantity is the aggregate.
    assert block["tail_pinned_at_all_frames"] is True

    assert (layout["nx"], layout["ny"]) == (4, 3)
    assert layout["cell_count"] == 12
    # The historical log contract is untouched: still exactly one ACCEPTED scan
    # line (load lines use their own prefix).
    accepted = [m for m in msgs if m.startswith("layout scan:") and "ACCEPTED" in m]
    assert len(accepted) == 1

    # Per-candidate table published, exactly the bounded family.
    rows = [(r["nx"], r["ny"], r["cells"], r["total_slots"], r["max_slots"],
             r["mean_slots"], r["admissible"], r["acceptable"], r["selected"])
            for r in block["candidates"]]
    assert rows == [
        (4, 5, 20, 1353, 208, 67.65, True, False, False),
        (5, 4, 20, 1296, 208, 64.8, True, False, False),
        (4, 3, 12, 975, 208, 81.25, True, True, True),
        (3, 4, 12, 984, 208, 82.0, True, True, False),
        (5, 6, 30, 1696, 208, 56.533333, True, False, False),
        (6, 6, 36, 2007, 208, 55.75, True, False, False),
    ]
    assert len(block["candidates"]) <= zal.LOAD_MAX_EXTRA_CANDIDATES


# ---------------------------------------------------------------------------
# 3. Production path — the tail shortened inside the published exchange rate
# ---------------------------------------------------------------------------

def test_production_path_reduces_tail_within_exchange_rate(corpus_big):
    descs, canvas = corpus_big
    layout = _run(canvas, descs)
    block = layout["membership_load"]
    base, chosen = block["baseline"], block["chosen"]

    assert (base["nx"], base["ny"], base["cells"]) == (6, 7, 42)
    assert (base["total_slots"], base["max_slots"]) == (7351, 1086)
    assert base["max_over_mean"] == pytest.approx(6.20487, abs=1e-5)
    assert base["score"] == 11695

    assert (chosen["nx"], chosen["ny"], chosen["cells"]) == (6, 8, 48)
    assert (chosen["total_slots"], chosen["max_slots"]) == (7928, 940)
    assert chosen["max_over_mean"] == pytest.approx(5.691221, abs=1e-5)
    assert chosen["score"] == 11688
    assert block["applied"] is True
    assert block["reason"] == "load_balanced_lower_tail"
    assert block["tail_pinned_at_all_frames"] is False
    assert (layout["nx"], layout["ny"]) == (6, 8)

    # The tail really is shorter, and the aggregate price is INSIDE the
    # published bound (total <= base_total + 4 * tail_saved).
    assert chosen["max_slots"] < base["max_slots"]
    assert chosen["total_slots"] <= (
        base["total_slots"] + 4 * (base["max_slots"] - chosen["max_slots"])
    )
    # ...and every published candidate obeys the same two guards.
    for row in block["candidates"]:
        if row["acceptable"]:
            assert row["max_slots"] <= base["max_slots"]
            assert row["total_slots"] <= (
                base["total_slots"] + 4 * (base["max_slots"] - row["max_slots"])
            )


# ---------------------------------------------------------------------------
# 4. Truthful no-op: no candidate beats the historical grid
# ---------------------------------------------------------------------------

def test_no_improvement_keeps_historical_grid_and_states_reason(
    corpus_c, monkeypatch
):
    descs, canvas = corpus_c
    layout = _run(canvas, descs)
    block = layout["membership_load"]
    assert block["applied"] is False
    assert block["reason"] == "no_candidate_improved_score"
    assert block["chosen"] == block["baseline"]
    assert (layout["nx"], layout["ny"]) == (5, 5)

    # Parity: with the load pass disabled the chosen grid is EXACTLY the same.
    monkeypatch.setattr(zz, "_load_aware_select", lambda *a, **k: None)
    neutral = _run(canvas, descs)
    assert (neutral["nx"], neutral["ny"]) == (layout["nx"], layout["ny"])
    assert "membership_load" not in neutral


# ---------------------------------------------------------------------------
# 5. Determinism + input-order independence
# ---------------------------------------------------------------------------

def test_deterministic_same_inputs_same_grid_twice(corpus_a):
    descs, canvas = corpus_a
    first = _run(canvas, descs)
    second = _run(canvas, descs)
    assert (first["nx"], first["ny"]) == (second["nx"], second["ny"])
    assert json.dumps(first["membership_load"], sort_keys=True) == json.dumps(
        second["membership_load"], sort_keys=True
    )


def test_input_order_does_not_change_the_grid(corpus_a):
    descs, canvas = corpus_a
    shuffled = list(descs)
    random.Random(1234).shuffle(shuffled)
    a = _run(canvas, descs)
    b = _run(canvas, shuffled)
    assert (a["nx"], a["ny"]) == (b["nx"], b["ny"])
    assert json.dumps(a["membership_load"], sort_keys=True) == json.dumps(
        b["membership_load"], sort_keys=True
    )


# ---------------------------------------------------------------------------
# 6. Fail-closed guards: floors, halo, RAM envelope
# ---------------------------------------------------------------------------

def test_family_never_generates_below_the_patch_floor(corpus_a):
    _descs, canvas = corpus_a
    floors = zal.ScientificFloors(
        min_patch_area_px=8000, min_contributors=1, max_halo_overhead=1e9
    )
    family = zz._load_candidate_family(
        canvas, 80.0, 80.0, 4, 4, floors, zz.zsw.HALO_PX, halo_advisory=False
    )
    assert family, "expected coarser floor-feasible candidates"
    for nx, ny in family:
        geom = zal._nominal_geometry(canvas, nx, ny, zz.zsw.HALO_PX)
        assert geom["min_patch_area"] >= floors.min_patch_area_px
        # Coarsening only (the baseline 4x4 itself is below the floor).
        assert nx * ny < 16


def test_load_pass_cannot_adopt_a_floor_violating_grid(corpus_a, monkeypatch):
    descs, canvas = corpus_a
    # Deliberately hand the load pass a floor-violating grid: the admission
    # guard must refuse it even though its load would score better.
    floors = zal.ScientificFloors(
        min_patch_area_px=5000, min_contributors=1, max_halo_overhead=1e9
    )
    monkeypatch.setattr(zz, "_load_candidate_family", lambda *a, **k: ((8, 8),))
    layout = _run(canvas, descs, floors=floors)
    block = layout["membership_load"]
    assert (layout["nx"], layout["ny"]) == (4, 4)
    assert block["applied"] is False
    assert [r["floors_ok"] for r in block["candidates"]] == [False]
    assert [r["selected"] for r in block["candidates"]] == [False]


def test_load_pass_cannot_adopt_an_inadmissible_grid(corpus_a, monkeypatch):
    descs, canvas = corpus_a
    real = zz._admit_load_candidate

    def _refuse(*args, **kwargs):
        state = real(*args, **kwargs)
        state["admissible"] = False
        return state

    monkeypatch.setattr(zz, "_admit_load_candidate", _refuse)
    layout = _run(canvas, descs)
    block = layout["membership_load"]
    assert (layout["nx"], layout["ny"]) == (4, 4)
    assert block["applied"] is False
    assert all(r["selected"] is False for r in block["candidates"])


def test_chosen_grid_always_fits_the_ram_envelope(corpus_a):
    descs, canvas = corpus_a
    for budget_mib in (630, 700, 1200):
        budget = int(budget_mib * 2 ** 20)
        layout = _run(canvas, descs, budget=budget)
        assert layout["predicted_bound_bytes"] <= budget
        block = layout["membership_load"]
        for row in block["candidates"]:
            if row["selected"]:
                assert row["admissible"] is True
                assert row["acceptable"] is True


# ---------------------------------------------------------------------------
# 7. R28 — exactly N footprint projections, load pass engaged
# ---------------------------------------------------------------------------

def test_r28_exactly_n_projections_with_load_pass(corpus_a, monkeypatch):
    descs, canvas = corpus_a
    real = zg._source_polygon
    calls = []

    def spy(shape_hw, src_wcs, tgt_wcs):
        calls.append(1)
        return real(shape_hw, src_wcs, tgt_wcs)

    monkeypatch.setattr(zg, "_source_polygon", spy)
    out = []
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1e15), floors=_floors(), footprints_out=out,
    )
    assert layout["membership_load"]["applied"] is True
    assert len(calls) == len(descs), "load pass triggered a second projection"
    assert len(out) == 1

    # The chosen Cells are built from the SAME polygons (no re-projection).
    from zemosaic.core.zegrid import sweep as zsw
    ctxs = zsw.build_cell_contexts_from_footprints(
        out[0], canvas, layout["nx"], layout["ny"]
    )
    assert len(ctxs) == layout["nx"] * layout["ny"]
    assert len(calls) == len(descs)


# ---------------------------------------------------------------------------
# 8. Legacy parity: neutral signal + pinned layout publish nothing new
# ---------------------------------------------------------------------------

def test_neutral_signal_keeps_historical_grid_and_publishes_nothing(
    tiny_corpus, monkeypatch
):
    descs, canvas = tiny_corpus
    scanned = []
    real_scan = zz._scan_layout_mode_aware

    def spy(canvas, footprints, nx, ny, halo_px, tile_size):
        scanned.append((nx, ny))
        return real_scan(canvas, footprints, nx, ny, halo_px, tile_size)

    monkeypatch.setattr(zz, "_scan_layout_mode_aware", spy)
    layout = _run(canvas, descs)
    # Perfectly covered corpus: max == mean, the signal is NOT material, so the
    # historical single-candidate scan is byte-for-byte unchanged.
    assert (layout["nx"], layout["ny"]) == (2, 2)
    assert len(scanned) == 1
    assert "membership_load" not in layout


def test_pinned_layout_is_untouched(corpus_a):
    descs, canvas = corpus_a
    layout = _run(canvas, descs, pinned_layout=(2, 2))
    assert (layout["nx"], layout["ny"]) == (2, 2)
    assert layout["layout_source"] == "pinned"
    assert layout["layout_origin"] == "pinned"
    assert "membership_load" not in layout


# ---------------------------------------------------------------------------
# 9. Publication: additive, inside the existing layout block, JSON-native
# ---------------------------------------------------------------------------

def test_manifest_layout_block_is_additive(corpus_a, tiny_corpus):
    descs, canvas = corpus_a
    layout = _run(canvas, descs)
    block = zz._manifest_layout_block(layout["nx"], layout["ny"], layout)
    assert "membership_load" in block
    # Historical keys are all still present and unchanged.
    assert block["nx"] == layout["nx"] and block["ny"] == layout["ny"]
    assert block["max_contributors"] == layout["max_contributors"]
    assert block["predicted_bound_bytes"] == layout["predicted_bound_bytes"]
    assert block["layout_origin"] == "auto_refinement"
    # JSON-native (no geometry objects, no int keys).
    assert json.loads(json.dumps(block))["membership_load"]["chosen"]["nx"] == 4

    tdescs, tcanvas = tiny_corpus
    tlayout = _run(tcanvas, tdescs)
    tblock = zz._manifest_layout_block(tlayout["nx"], tlayout["ny"], tlayout)
    assert "membership_load" not in tblock
    assert set(tblock) == {
        "nx", "ny", "ram_budget_bytes", "available_bytes", "max_contributors",
        "predicted_bound_bytes", "layout_origin",
    }

    # Nothing private leaked into the public layout/block.
    def _walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                yield from _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from _walk(v)
        else:
            yield obj

    assert not [v for v in _walk(block) if isinstance(v, BaseGeometry)]


def test_log_publishes_the_per_candidate_table(corpus_a):
    descs, canvas = corpus_a
    lines = []
    _run(canvas, descs, emit=lambda m, lvl="INFO": lines.append(m))
    load_lines = [m for m in lines if m.startswith("layout load:")]
    baseline_line = load_lines[0]
    assert "baseline 4x4" in baseline_line
    assert "total_slots=1164" in baseline_line
    assert "max_slots=208" in baseline_line
    assert "mean_slots=72.8" in baseline_line
    assert "max/mean=2.86" in baseline_line
    assert "CHOSEN 4x3" in baseline_line
    assert "reason: load_balanced_lower_total_slots" in baseline_line
    assert "tail_pinned_at_all_frames=True" in baseline_line
    # One line per published candidate, each carrying the full tuple.
    for row in _run(canvas, descs)["membership_load"]["candidates"]:
        needle = f"candidate {row['nx']}x{row['ny']} cells={row['cells']}"
        assert any(needle in m for m in load_lines)
    assert sum(1 for m in load_lines if "candidate " in m) == len(
        _run(canvas, descs)["membership_load"]["candidates"]
    )
