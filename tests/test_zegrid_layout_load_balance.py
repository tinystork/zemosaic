"""ZM-ZEGRID-LAYOUT-LOAD-BALANCE — membership-load aware layout choice.

Hardening pass (ZM-ZEGRID-HARDENING-PASS-20261113) rewrote the rank key (H1) and
the trigger denominator (H2) after review; H3 pins legacy byte-identity.

What the policy does
--------------------
The historical Auto policy ranks candidate grids by PEAK MEMORY and takes the
FIRST (coarsest) admissible one. It never looked at how many frame MEMBERSHIPS a
Cell carries, so a layout could be memory-optimal and load-pathological: on the
real M33 geometry (21 818 frames, 170 Cells) the load is 298 893 membership slots
(amplification 13.7x), mean 1 758 slots/Cell, max 21 804. Every membership is
reprojected and reduced, and the external lane spools per membership.

Rank (H1): the ACTUAL makespan proxy, lower is better::

    proxy = max(ceil(total_slots / WORKERS), max_slots)      WORKERS = 4
    rank_key = (proxy, total_slots, max_slots, cells, nx, ny)

($total + 4*max$ is NOT that proxy: total=100, max=10 gives 140, not 25.)

Acceptance vs the historical grid: the tail may never grow AND the proxy may
never worsen. Trigger (H2): ``max_slots / mean(NON-EMPTY Cells) > 2.0`` — empty
Cells carry no work and must not deflate the mean.

Every test here is SYNTHETIC and fast (no user data, no pixels, no
multiprocessing).
"""

from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import pytest
from astropy.wcs import WCS
from shapely.geometry.base import BaseGeometry

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg


# ---------------------------------------------------------------------------
# Synthetic corpora
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


def _padded(canvas, factor, tag=""):
    """Canvas larger than the sky union (empty Cells are REAL Cells).

    The real M33 layout has Cells with zero memberships (measured min = 0), so a
    canvas whose union covers only part of it is the representative case.
    """
    return zg.GlobalCanvas(
        canvas_id=f"{canvas.canvas_id}-pad{factor}{tag}",
        wcs_header=canvas.wcs_header,
        width=int(canvas.width * factor), height=int(canvas.height * factor),
        resolution_deg=canvas.resolution_deg,
    )


def _core_corpus(core_n, core_step, core_reps, halo_n, halo_step, halo_reps, pad):
    descs, i = [], 0
    for ix in range(core_n):
        for iy in range(core_n):
            for r in range(core_reps):
                descs.append(_frame(
                    i, 80, 80, 0.001,
                    10.0 + (ix - core_n / 2) * core_step + 2e-4 * r,
                    30.0 + (iy - core_n / 2) * core_step + 2e-4 * r))
                i += 1
    for ix in range(halo_n):
        for iy in range(halo_n):
            for r in range(halo_reps):
                descs.append(_frame(
                    i, 80, 80, 0.001,
                    10.0 + (ix - halo_n / 2) * halo_step + 3e-4 * r,
                    30.0 + (iy - halo_n / 2) * halo_step + 3e-4 * r))
                i += 1
    return descs, _padded(zg.build_canvas(descs), pad)


def _floors():
    return zal.ScientificFloors(
        min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9
    )


_PARAMS = {
    # 208 frames, 16 Cells of which 9 are occupied: the ALL-cells ratio is 2.86
    # (the historical trigger) but the NON-EMPTY ratio is 1.61 -> the pass must
    # NOT trigger (H2).
    "A": dict(core_n=8, core_step=0.010, core_reps=3, halo_n=4, halo_step=0.03,
              halo_reps=1, pad=1.6),
    # 864 frames, all 16 Cells occupied, genuinely imbalanced: triggers under
    # BOTH denominators and picks the SAME grid (H2 equivalence), lowering the
    # total AND the tail.
    "CORE": dict(core_n=12, core_step=0.012, core_reps=6, halo_n=0, halo_step=0.03,
                 halo_reps=0, pad=1.3),
    # 1380 frames: the proxy is aggregate-bound, so the rank now lowers the
    # total at a pinned tail.
    "BIG": dict(core_n=16, core_step=0.010, core_reps=5, halo_n=10, halo_step=0.03,
                halo_reps=1, pad=1.4),
    # 636 frames: a candidate with a LOWER total and a LOWER proxy but a LONGER
    # tail is refused by the tail guard -> the historical grid is kept.
    "C": dict(core_n=10, core_step=0.012, core_reps=6, halo_n=6, halo_step=0.04,
              halo_reps=1, pad=1.4),
}


@pytest.fixture(scope="module")
def corpus_a():
    return _core_corpus(**_PARAMS["A"])


@pytest.fixture(scope="module")
def corpus_core():
    return _core_corpus(**_PARAMS["CORE"])


@pytest.fixture(scope="module")
def corpus_big():
    return _core_corpus(**_PARAMS["BIG"])


@pytest.fixture(scope="module")
def corpus_c():
    return _core_corpus(**_PARAMS["C"])


@pytest.fixture(scope="module")
def tiny_corpus():
    descs = [_frame(i, 40, 40, 0.001, 10.0 + i * 0.004, 30.0, "t") for i in range(4)]
    return descs, zg.build_canvas(descs)


def _run(canvas, descs, *, budget=int(1e15), floors=None, **kw):
    return zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=budget, floors=floors or _floors(), **kw
    )


# ---------------------------------------------------------------------------
# H1 — the rank is the ACTUAL makespan proxy, not a weighted sum
# ---------------------------------------------------------------------------

def _load(total, mx, cells=4, nx=2, ny=2):
    return zal.MembershipLoad(
        nx=nx, ny=ny, cells=cells, total_slots=total, max_slots=mx,
        mean_slots=total / cells, nonempty_cells=cells,
        mean_nonempty_slots=total / cells,
    )


def test_h1_old_weighted_score_and_true_proxy_disagree():
    """Pinned counter-example: `total + 4*max` is NOT `max(total/4, max)`.

    baseline: total=400, max=100 -> old 800, proxy 100.
    candidate: total=401, max=50 -> old 601 (the OLD score prefers it), proxy 101
    (the TRUE makespan is worse). The new rank keeps the baseline: the lower
    makespan grid wins.
    """
    baseline = _load(400, 100)
    candidate = _load(401, 50)
    assert baseline.score() == baseline.makespan_proxy_slots() == 100
    assert candidate.score() == 101
    # The old formula really does flip the decision...
    assert candidate.total_slots + 4 * candidate.max_slots < (
        baseline.total_slots + 4 * baseline.max_slots
    )
    # ...and the new rank does not.
    assert zal.better_load(candidate, baseline) is False
    # A candidate that improves BOTH the tail and the proxy is accepted.
    good = _load(380, 50)
    assert good.makespan_proxy_slots() == max(-(-380 // 4), 50) == 95
    assert good.total_slots + 4 * good.max_slots < 800  # old agrees here
    assert zal.better_load(good, baseline) is True
    assert good.rank_key() == (95, 380, 50, 4, 2, 2)


def test_h1_rank_key_is_a_total_order_and_deterministic():
    loads = [_load(400, 100, 4, 2, 2), _load(380, 50, 4, 2, 2),
             _load(380, 50, 6, 3, 2), _load(400, 100, 4, 2, 3)]
    keys = [load.rank_key() for load in loads]
    # The tuples are totally ordered and pairwise distinct (a STRICT total order).
    assert len(set(keys)) == len(keys)
    assert sorted(keys) == [
        (95, 380, 50, 4, 2, 2),
        (95, 380, 50, 6, 3, 2),
        (100, 400, 100, 4, 2, 2),
        (100, 400, 100, 4, 2, 3),
    ]
    assert min(keys) == (95, 380, 50, 4, 2, 2)
    # Antisymmetry: better_load is a strict order.
    a, b = loads[0], loads[1]
    assert zal.better_load(b, a) is True
    assert zal.better_load(a, b) is False


def test_h1_workers_constant_and_proxy_arithmetic():
    assert zal.LOAD_WORKERS == 4
    assert _load(401, 10).makespan_proxy_slots() == 101  # ceil, not floor
    assert _load(400, 10).makespan_proxy_slots() == 100
    assert _load(0, 0).makespan_proxy_slots() == 0
    # The proxy is never below the tail (a Cell is an indivisible serial unit).
    assert _load(8, 50).makespan_proxy_slots() == 50


# ---------------------------------------------------------------------------
# H2 — the trigger denominator is the NON-EMPTY Cell count
# ---------------------------------------------------------------------------

def test_h2_empty_cells_do_not_arm_the_pass(corpus_a):
    descs, canvas = corpus_a
    footprints = zal.project_footprints(descs, canvas)
    load = zz._scan_layout_mode_aware(
        canvas, footprints, 4, 4, zz.zsw.HALO_PX, zz.STREAM_TILE_SIZE
    )["membership_load"]

    # Pinned geometry of the historical grid.
    assert load.to_dict() == {
        "nx": 4, "ny": 4, "cells": 16, "nonempty_cells": 9,
        "total_slots": 1164, "max_slots": 208,
        "mean_slots": 72.75, "mean_nonempty_slots": 129.333333,
        "max_over_mean_nonempty": 1.608247,
        "max_over_mean_all_cells": 2.859107,
        "score": 291, "makespan_proxy_slots": 291, "workers": 4,
        "rank_key": [291, 1164, 208, 16, 4, 4],
    }
    # The historical (all-Cells) statistic WOULD have triggered; the non-empty
    # one must not, because 7 empty Cells carry no load at all.
    assert load.imbalance_all_cells > zal.LOAD_IMBALANCE_TRIGGER
    assert load.imbalance <= zal.LOAD_IMBALANCE_TRIGGER
    assert zal.load_signal_is_material(load) is False

    layout = _run(canvas, descs)
    assert "membership_load" not in layout
    assert (layout["nx"], layout["ny"]) == (4, 4)


def test_h2_occupied_corpus_triggers_under_both_denominators_and_picks_same_grid(
    corpus_core, monkeypatch
):
    descs, canvas = corpus_core
    new = _run(canvas, descs)
    block = new["membership_load"]
    # All 16 Cells are occupied, so both denominators agree by construction.
    assert block["baseline"]["cells"] == block["baseline"]["nonempty_cells"] == 16
    assert block["baseline"]["max_over_mean_nonempty"] == pytest.approx(
        block["baseline"]["max_over_mean_all_cells"]
    )

    # Force the LEGACY (all-Cells) trigger: the chosen grid must be identical.
    def all_cells_trigger(load, trigger=zal.LOAD_IMBALANCE_TRIGGER):
        if load.cells <= 1 or load.total_slots <= 0:
            return False
        return float(load.imbalance_all_cells) > float(trigger)

    monkeypatch.setattr(zal, "load_signal_is_material", all_cells_trigger)
    legacy_run = _run(canvas, descs)
    assert (legacy_run["nx"], legacy_run["ny"]) == (new["nx"], new["ny"])
    assert legacy_run["membership_load"]["chosen"] == block["chosen"]
    assert (new["nx"], new["ny"]) == (4, 3)


# ---------------------------------------------------------------------------
# Production path — pinned decisions under the corrected rank
# ---------------------------------------------------------------------------

def test_production_path_lowers_total_and_tail(corpus_core):
    descs, canvas = corpus_core
    layout = _run(canvas, descs)
    block = layout["membership_load"]
    assert block["baseline"] == {
        "nx": 4, "ny": 4, "cells": 16, "nonempty_cells": 16,
        "total_slots": 4872, "max_slots": 864,
        "mean_slots": 304.5, "mean_nonempty_slots": 304.5,
        "max_over_mean_nonempty": 2.837438,
        "max_over_mean_all_cells": 2.837438,
        "score": 1218, "makespan_proxy_slots": 1218, "workers": 4,
        "rank_key": [1218, 4872, 864, 16, 4, 4],
    }
    assert block["chosen"] == {
        "nx": 4, "ny": 3, "cells": 12, "nonempty_cells": 12,
        "total_slots": 4031, "max_slots": 792,
        "mean_slots": 335.916667, "mean_nonempty_slots": 335.916667,
        "max_over_mean_nonempty": 2.357728,
        "max_over_mean_all_cells": 2.357728,
        "score": 1008, "makespan_proxy_slots": 1008, "workers": 4,
        "rank_key": [1008, 4031, 792, 12, 4, 3],
    }
    assert block["applied"] is True
    assert block["reason"] == "load_balanced_lower_total_and_tail"
    assert block["trigger_denominator"] == "non_empty_cells"
    assert block["workers"] == 4
    assert block["n_frames"] == 864
    assert block["tail_pinned_at_all_frames"] is True
    assert (layout["nx"], layout["ny"]) == (4, 3)

    rows = [(r["nx"], r["ny"], r["cells"], r["nonempty_cells"], r["total_slots"],
             r["max_slots"], r["makespan_proxy_slots"], r["acceptable"],
             r["selected"]) for r in block["candidates"]]
    assert rows == [
        (4, 5, 20, 16, 5394, 720, 1349, False, False),
        (5, 4, 20, 20, 5432, 792, 1358, False, False),
        (4, 3, 12, 12, 4031, 792, 1008, True, True),
        (3, 4, 12, 12, 4200, 864, 1050, True, False),
        (5, 6, 30, 25, 6822, 726, 1706, False, False),
        (6, 7, 42, 30, 9120, 792, 2280, False, False),
    ]
    assert len(block["candidates"]) <= zal.LOAD_MAX_EXTRA_CANDIDATES


def test_production_path_proxy_driven_total_reduction(corpus_big):
    descs, canvas = corpus_big
    block = _run(canvas, descs)["membership_load"]
    base, chosen = block["baseline"], block["chosen"]
    assert (base["nx"], base["ny"], base["cells"], base["nonempty_cells"]) == (
        6, 7, 42, 30)
    assert (base["total_slots"], base["max_slots"], base["score"]) == (
        7351, 1086, 1838)
    assert base["max_over_mean_nonempty"] == pytest.approx(4.43205, abs=1e-5)
    assert base["max_over_mean_all_cells"] == pytest.approx(6.20487, abs=1e-5)
    # The proxy is aggregate-bound (ceil(7351/4)=1838 > 1086), so the rank buys
    # a lower total at the SAME pinned tail.
    assert (chosen["nx"], chosen["ny"], chosen["cells"]) == (6, 6, 36)
    assert (chosen["total_slots"], chosen["max_slots"], chosen["score"]) == (
        6937, 1086, 1735)
    assert block["applied"] is True
    assert block["reason"] == "load_balanced_lower_total_slots"
    assert chosen["max_slots"] <= base["max_slots"]
    assert chosen["makespan_proxy_slots"] <= base["makespan_proxy_slots"]


def test_tail_guard_refuses_a_lower_total_and_lower_proxy_but_longer_tail(corpus_c):
    descs, canvas = corpus_c
    layout = _run(canvas, descs)
    block = layout["membership_load"]
    base = block["baseline"]
    assert (base["total_slots"], base["max_slots"], base["score"]) == (
        3255, 565, 814)
    assert block["applied"] is False
    assert block["reason"] == "no_candidate_improved_score"
    assert (layout["nx"], layout["ny"]) == (5, 5)

    row = next(r for r in block["candidates"] if (r["nx"], r["ny"]) == (5, 4))
    # This candidate is better on BOTH the rank (proxy 704 < 814) and the total
    # (2814 < 3255) -- and is still refused, because its tail is longer.
    assert row["total_slots"] < base["total_slots"]
    assert row["makespan_proxy_slots"] < base["makespan_proxy_slots"]
    assert row["max_slots"] > base["max_slots"]
    assert row["acceptable"] is False and row["selected"] is False


# ---------------------------------------------------------------------------
# Determinism + input-order independence
# ---------------------------------------------------------------------------

def test_deterministic_same_inputs_same_grid_twice(corpus_core):
    descs, canvas = corpus_core
    first, second = _run(canvas, descs), _run(canvas, descs)
    assert (first["nx"], first["ny"]) == (second["nx"], second["ny"])
    assert json.dumps(first["membership_load"], sort_keys=True) == json.dumps(
        second["membership_load"], sort_keys=True)


def test_input_order_does_not_change_the_grid(corpus_core):
    descs, canvas = corpus_core
    shuffled = list(descs)
    random.Random(1234).shuffle(shuffled)
    a, b = _run(canvas, descs), _run(canvas, shuffled)
    assert (a["nx"], a["ny"]) == (b["nx"], b["ny"])
    assert json.dumps(a["membership_load"], sort_keys=True) == json.dumps(
        b["membership_load"], sort_keys=True)


# ---------------------------------------------------------------------------
# Fail-closed guards: floors, halo, RAM envelope
# ---------------------------------------------------------------------------

def test_family_never_generates_below_the_patch_floor(corpus_core):
    _descs, canvas = corpus_core
    floors = zal.ScientificFloors(
        min_patch_area_px=8000, min_contributors=1, max_halo_overhead=1e9)
    family = zz._load_candidate_family(
        canvas, 80.0, 80.0, 4, 4, floors, zz.zsw.HALO_PX, halo_advisory=False)
    assert family
    for nx, ny in family:
        geom = zal._nominal_geometry(canvas, nx, ny, zz.zsw.HALO_PX)
        assert geom["min_patch_area"] >= floors.min_patch_area_px
        assert nx * ny < 16


def test_load_pass_cannot_adopt_a_floor_violating_grid(corpus_core, monkeypatch):
    descs, canvas = corpus_core
    floors = zal.ScientificFloors(
        min_patch_area_px=5000, min_contributors=1, max_halo_overhead=1e9)
    monkeypatch.setattr(zz, "_load_candidate_family", lambda *a, **k: ((12, 12),))
    layout = _run(canvas, descs, floors=floors)
    block = layout["membership_load"]
    assert (layout["nx"], layout["ny"]) == (4, 4)
    assert block["applied"] is False
    assert [r["floors_ok"] for r in block["candidates"]] == [False]
    assert [r["selected"] for r in block["candidates"]] == [False]


def test_load_pass_cannot_adopt_an_inadmissible_grid(corpus_core, monkeypatch):
    descs, canvas = corpus_core
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


def test_chosen_grid_always_fits_the_ram_envelope(corpus_core):
    descs, canvas = corpus_core
    for budget_mib in (700, 900, 1400):
        budget = int(budget_mib * 2 ** 20)
        layout = _run(canvas, descs, budget=budget)
        assert layout["predicted_bound_bytes"] <= budget
        for row in layout["membership_load"]["candidates"]:
            if row["selected"]:
                assert row["admissible"] is True and row["acceptable"] is True


# ---------------------------------------------------------------------------
# R28 — exactly N footprint projections, load pass engaged
# ---------------------------------------------------------------------------

def test_r28_exactly_n_projections_with_load_pass(corpus_core, monkeypatch):
    descs, canvas = corpus_core
    real = zg._source_polygon
    calls = []

    def spy(shape_hw, src_wcs, tgt_wcs):
        calls.append(1)
        return real(shape_hw, src_wcs, tgt_wcs)

    monkeypatch.setattr(zg, "_source_polygon", spy)
    out = []
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1e15), floors=_floors(), footprints_out=out)
    assert layout["membership_load"]["applied"] is True
    assert len(calls) == len(descs)
    assert len(out) == 1
    from zemosaic.core.zegrid import sweep as zsw
    ctxs = zsw.build_cell_contexts_from_footprints(
        out[0], canvas, layout["nx"], layout["ny"])
    assert len(ctxs) == layout["nx"] * layout["ny"]
    assert len(calls) == len(descs)


# ---------------------------------------------------------------------------
# Legacy parity — neutral signal and pinned layout publish nothing new
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
    assert (layout["nx"], layout["ny"]) == (2, 2)
    assert len(scanned) == 1
    assert "membership_load" not in layout


def test_pinned_layout_is_untouched(corpus_core):
    descs, canvas = corpus_core
    layout = _run(canvas, descs, pinned_layout=(2, 2))
    assert (layout["nx"], layout["ny"]) == (2, 2)
    assert layout["layout_source"] == "pinned"
    assert layout["layout_origin"] == "pinned"
    assert "membership_load" not in layout


# ---------------------------------------------------------------------------
# Publication — additive, inside the existing layout block, JSON-native
# ---------------------------------------------------------------------------

def test_manifest_layout_block_is_additive(corpus_core, tiny_corpus):
    descs, canvas = corpus_core
    layout = _run(canvas, descs)
    block = zz._manifest_layout_block(layout["nx"], layout["ny"], layout)
    assert "membership_load" in block
    assert block["nx"] == layout["nx"] and block["ny"] == layout["ny"]
    assert block["max_contributors"] == layout["max_contributors"]
    assert block["predicted_bound_bytes"] == layout["predicted_bound_bytes"]
    assert block["layout_origin"] == "auto_refinement"
    assert json.loads(json.dumps(block))["membership_load"]["chosen"]["nx"] == 4

    tdescs, tcanvas = tiny_corpus
    tlayout = _run(tcanvas, tdescs)
    tblock = zz._manifest_layout_block(tlayout["nx"], tlayout["ny"], tlayout)
    assert "membership_load" not in tblock
    assert set(tblock) == {
        "nx", "ny", "ram_budget_bytes", "available_bytes", "max_contributors",
        "predicted_bound_bytes", "layout_origin",
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

    assert not [v for v in _walk(block) if isinstance(v, BaseGeometry)]


def test_log_publishes_the_per_candidate_table(corpus_core):
    descs, canvas = corpus_core
    lines = []
    _run(canvas, descs, emit=lambda m, lvl="INFO": lines.append(m))
    load_lines = [m for m in lines if m.startswith("layout load:")]
    first = load_lines[0]
    assert "baseline 4x4" in first and "16 non-empty" in first
    assert "total_slots=4872" in first
    assert "max_slots=864" in first
    assert "mean/nonempty=304.5" in first
    assert "proxy=1218" in first
    assert "CHOSEN 4x3" in first
    assert "reason: load_balanced_lower_total_and_tail" in first
    assert "tail_pinned_at_all_frames=True" in first
    for row in _run(canvas, descs)["membership_load"]["candidates"]:
        needle = f"candidate {row['nx']}x{row['ny']} cells={row['cells']}"
        assert any(needle in m for m in load_lines)
        assert any(
            f"proxy={row['makespan_proxy_slots']}" in m and needle in m
            for m in load_lines
        )
    # The historical scan log contract is untouched (one ACCEPTED line).
    assert len([m for m in lines
                if m.startswith("layout scan:") and "ACCEPTED" in m]) == 1


# ---------------------------------------------------------------------------
# H3 — legacy byte-identity proven, not asserted
# ---------------------------------------------------------------------------
#
# Each legacy/neutral path is serialised to CANONICAL JSON:
#   {"manifest": <_manifest_layout_block>, "logs": <run-log lines>}
# and compared BYTE FOR BYTE (sha256) with a fixture captured from the
# PRE-CHANGE behaviour (commit c63c810, before the hardening pass) — see the
# report for the exact capture command. The fixture is the four pinned digests
# below; a deliberate comment records that they were produced from the baseline
# tree, not from the current code, so a mismatch is a real regression.

_LEGACY_SHA256 = {
    "neutral": "99f3445863d337af270785f4774e0abd472cfa46e2737ac8aeb5ea21f52f1216",
    "pinned": "b24b01d3204368ca5d430799c46ae438289db748815659eaa415ae71f6856481",
    "window": "589e2915633d4c7017131985b89e3952467c2e78ffa4b3cb074ea5e768d00c6b",
    "rescue": "17c64e2cf73db3c0c3930a2b44e3789faa25d10fc34c6f64ff6de9859143013b",
}


class _HugeModel:
    def predict_bound_bytes(self, n, area):
        return 10.0 ** 15

    def predict_peak_bytes(self, n, area):
        return 10.0 ** 15

    def to_dict(self):
        return {"stub": "huge"}


def _rescue_corpus():
    out = []
    for i in range(3):
        for j in range(3):
            out.append(_frame(i * 10 + j, 1000, 600, 0.001, 10.0 + i * 0.45,
                              30.0 + j * 0.4, "e6_"))
    return out


def _canonical(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _payload(canvas, descs, **kw):
    logs = []
    layout = zz._choose_layout_mode_aware(
        canvas, descs, log_line=logs.append, footprints_out=[], **kw)
    block = zz._manifest_layout_block(layout["nx"], layout["ny"], layout)
    return layout, {"manifest": block, "logs": logs}


def test_h3_legacy_paths_are_byte_identical(monkeypatch):
    tiny = [_frame(i, 40, 40, 0.001, 10.0 + i * 0.004, 30.0, "t")
            for i in range(4)]
    tiny_canvas = zg.build_canvas(tiny)
    floors = zal.ScientificFloors(
        min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9)

    # Record where the load pass runs: neutral -> once and returns None;
    # pinned / rescue / refinement-window -> NEVER.
    real_select = zz._load_aware_select
    calls = {"neutral": 0, "pinned": 0, "window": 0, "rescue": 0}
    active = {"name": None}

    def spy(*args, **kwargs):
        calls[active["name"]] += 1
        return real_select(*args, **kwargs)

    monkeypatch.setattr(zz, "_load_aware_select", spy)

    payloads = {}
    active["name"] = "neutral"
    layout, payloads["neutral"] = _payload(
        tiny_canvas, tiny, ram_budget=int(1e15), floors=floors)
    assert layout["layout_origin"] == "auto_refinement"

    active["name"] = "pinned"
    layout, payloads["pinned"] = _payload(
        tiny_canvas, tiny, ram_budget=int(1e15), floors=floors,
        pinned_layout=(2, 2))
    assert layout["layout_origin"] == "pinned"

    active["name"] = "window"
    layout, payloads["window"] = _payload(
        tiny_canvas, tiny, ram_budget=int(1e15), floors=floors,
        min_refinement_rank=6, max_refinement_rank=6)
    assert layout["layout_origin"] == "auto_refinement"

    rd = _rescue_corpus()
    rc = zg.build_canvas(rd)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeModel())
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes",
        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3: 1000)
    active["name"] = "rescue"
    layout, payloads["rescue"] = _payload(
        rc, rd, ram_budget=10 ** 9, halo_px=80, floors=zal.ScientificFloors())
    assert layout["layout_origin"] == "auto_coarser_rescue"

    # The pass ran ONLY on the neutral path (where it correctly said "no").
    assert calls == {"neutral": 1, "pinned": 0, "window": 0, "rescue": 0}

    for name, payload in payloads.items():
        # Nothing additive leaked into a legacy path.
        assert "membership_load" not in payload["manifest"], name
        digest = _canonical(payload)
        assert digest == _LEGACY_SHA256[name], (
            f"{name} legacy path is not byte-identical to the pre-change "
            f"fixture: got {digest}, expected {_LEGACY_SHA256[name]}"
        )
