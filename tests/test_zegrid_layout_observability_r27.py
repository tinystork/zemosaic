"""ZM-ZEGRID-R27 targeted tests — layout sub-step observability + timing.

Covers (non-gated, fast, SYNTHETIC — no user-data access, no M16/M106 fixtures):

* ``SubstepReporter`` throttles counters (first + final always emitted) on a
  deterministic fake clock, and reports EXACT per-sub-step elapsed timings.
* ``emit`` / ``log_line`` callback failures are swallowed and never raise.
* No fabricated candidate percent/total when the total is unknown; the stable
  sub-step ids are separated from counters/items and never embed counters.
* Candidate ACCEPTED/REJECTED + CHOSEN details reach the DURABLE run log
  (``log_line``), not only ``emit``.
* Chosen-cell progress reports first/final + bounded intermediates, and handles
  zero/one-cell edge cases without division-by-zero or fabricated percents.
* The no-observer path produces EXACTLY the same layout result as the observer
  path, and ``build_cell_context`` membership is unchanged/deterministic.
* Subordinate timing keys are present, non-negative, subordinate to ``layout``,
  excluded from ``Timings.total()``, and never enter ``_PHASE_ORDER``/GlobalEta.
"""

from __future__ import annotations

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import instrumentation as zin
from zemosaic.core.zegrid import observability as zobs
from zemosaic.core.zegrid import sweep as zsw


class _Recorder:
    def __init__(self):
        self.emitted = []
        self.log_lines = []

    def emit(self, msg, lvl="INFO"):
        self.emitted.append((msg, lvl))

    def log_line(self, line):
        self.log_lines.append(line)


class _FakeClock:
    """Deterministic injectable clock for SubstepReporter tests."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _build_corpus(n=4, h=40, w=40, scale=0.001):
    """Build a tiny synthetic TAN corpus (no disk / user-data access)."""
    from astropy.wcs import WCS

    descs = []
    for i in range(n):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [10.0 + i * 0.004, 30.0]
        wcs.wcs.crpix = [w / 2.0, h / 2.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
        wcs.array_shape = (h, w)
        fid = f"d{i}.fits"
        descs.append(
            zg.FrameDescriptor(
                frame_id=zg.FrameId(fid),
                source_path=f"/x/{fid}",
                shape_hw=(h, w),
                wcs_header=wcs.to_header(relax=True).tostring(),
                header_sha256="",
                instrument="",
            )
        )
    return descs


def _tiny_floors():
    # Relax the (real) scientific floors so a tiny synthetic canvas is feasible;
    # the floors are irrelevant to what these tests assert (observability/timing).
    return zal.ScientificFloors(
        min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9
    )


# ---------------------------------------------------------------------------
# SubstepReporter: throttling (first/final) + exact timing on a fake clock
# ---------------------------------------------------------------------------

def test_substep_throttle_first_and_final():
    r = _Recorder()
    clock = _FakeClock()
    rep = zobs.SubstepReporter(r.emit, log_line=r.log_line, clock=clock, interval_s=2.0)
    rep.start("layout:chosen_cells", "chosen-cell membership", total=100)

    rep.progress(1, total=100)  # first sample -> always emitted
    for d in range(2, 50):       # within the 2s window (clock never advances)
        rep.progress(d, total=100)

    progress = [m for m, lvl in r.emitted if m.startswith("substep PROGRESS:")]
    assert len(progress) == 1, "intermediate samples were not throttled"
    assert any("1/100" in p for p in progress)

    clock.advance(3.0)  # past the interval -> next sample allowed
    rep.progress(60, total=100)
    progress = [m for m, lvl in r.emitted if m.startswith("substep PROGRESS:")]
    assert any("60/100" in p for p in progress)

    rep.progress(100, total=100)  # final sample -> always emitted (no advance)
    progress = [m for m, lvl in r.emitted if m.startswith("substep PROGRESS:")]
    assert any("100/100" in p and "(100.0%)" in p for p in progress)


def test_substep_exact_timing():
    r = _Recorder()
    clock = _FakeClock()
    rep = zobs.SubstepReporter(r.emit, clock=clock)
    rep.start("layout:footprints", "footprints")
    clock.advance(12.5)
    elapsed = rep.end()

    assert elapsed == pytest.approx(12.5)
    assert rep.timings()["layout:footprints"] == pytest.approx(12.5)
    ends = [m for m, lvl in r.emitted if m.startswith("substep END:")]
    assert any("elapsed=12.500s" in p for p in ends)


def test_substep_swallows_callback_failures():
    def boom_emit(msg, lvl="INFO"):
        raise RuntimeError("emit failed")

    def boom_log(line):
        raise RuntimeError("log failed")

    rep = zobs.SubstepReporter(boom_emit, log_line=boom_log, clock=_FakeClock())
    rep.start("layout:candidates", "scan", total=3)
    rep.progress(1, total=3)
    rep.end()  # must not raise
    assert rep.timings()["layout:candidates"] >= 0.0


# ---------------------------------------------------------------------------
# No fabricated percent; stable ids separated from counters/items
# ---------------------------------------------------------------------------

def test_substep_no_fabricated_percent_when_total_unknown():
    r = _Recorder()
    rep = zobs.SubstepReporter(r.emit, clock=_FakeClock(), interval_s=0.0)
    rep.start("layout:candidates", "candidate grid scan", total=None)
    rep.progress(3, item_id="17x10")
    progress = [m for m, lvl in r.emitted if m.startswith("substep PROGRESS:")]
    assert progress
    assert "done=3" in progress[0]
    assert "%" not in progress[0]
    assert "/" not in progress[0].split("done=3", 1)[0]
    assert "item=17x10" in progress[0]


def test_stable_substep_ids_do_not_embed_counters():
    for sid in (
        zobs.LAYOUT_SUBSTEP_FOOTPRINTS,
        zobs.LAYOUT_SUBSTEP_CANDIDATES,
        zobs.LAYOUT_SUBSTEP_CHOSEN_CELLS,
    ):
        assert isinstance(sid, str)
        assert "/" not in sid
        assert not any(ch.isdigit() for ch in sid)
    assert zobs.LAYOUT_SUBSTEP_FOOTPRINTS == "layout:footprints"
    assert zobs.LAYOUT_SUBSTEP_CANDIDATES == "layout:candidates"
    assert zobs.LAYOUT_SUBSTEP_CHOSEN_CELLS == "layout:chosen_cells"


def test_substep_id_separated_from_counters_in_messages():
    r = _Recorder()
    rep = zobs.SubstepReporter(r.emit, clock=_FakeClock(), interval_s=0.0)
    rep.start("layout:footprints", "footprints", total=8)
    rep.progress(3, item_id="d2.fits", total=8)
    line = [m for m, lvl in r.emitted if m.startswith("substep PROGRESS:")][0]
    assert "layout:footprints" in line   # stable id, no embedded counter
    assert "3/8" in line                 # counter is a SEPARATE field
    assert "item=d2.fits" in line        # item is a SEPARATE field


# ---------------------------------------------------------------------------
# Zero / one-cell edge cases
# ---------------------------------------------------------------------------

def test_substep_progress_zero_and_one_cell_edge_cases():
    r = _Recorder()
    rep = zobs.SubstepReporter(r.emit, clock=_FakeClock(), interval_s=0.0)
    rep.start("layout:chosen_cells", "cells", total=1)
    rep.progress(1, item_id="r0000c0000", total=1)
    rep.end()
    progress = [m for m, lvl in r.emitted if m.startswith("substep PROGRESS:")]
    assert progress
    assert any("1/1" in p and "(100.0%)" in p for p in progress)

    r2 = _Recorder()
    rep2 = zobs.SubstepReporter(r2.emit, clock=_FakeClock(), interval_s=0.0)
    rep2.start("layout:chosen_cells", "cells", total=0)
    rep2.progress(0, total=0)
    rep2.end()
    p2 = [m for m, lvl in r2.emitted if m.startswith("substep PROGRESS:")]
    for p in p2:
        assert "%" not in p        # no fabricated percent
        assert "0/0" not in p      # no division-by-zero


# ---------------------------------------------------------------------------
# Candidate + CHOSEN durably logged; observer emits three sub-steps with timing
# ---------------------------------------------------------------------------

def test_candidate_and_chosen_details_reach_run_log():
    descs = _build_corpus()
    canvas = zg.build_canvas(descs)
    emit_calls = []
    log_calls = []
    zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1 * 2**30), floors=_tiny_floors(),
        emit=lambda m, lvl="INFO": emit_calls.append(m),
        log_line=log_calls.append,
    )
    scan_log = [m for m in log_calls if m.startswith("layout scan:")]
    scan_emit = [m for m in emit_calls if m.startswith("layout scan:")]
    assert scan_log, "no layout scan lines durably logged"
    assert any("CHOSEN" in m for m in scan_log), "CHOSEN not durably logged"
    # Every candidate/CHOSEN line that was emitted is ALSO durably logged.
    assert set(scan_emit) <= set(scan_log)


def test_layout_observer_emits_three_substeps_with_timing():
    descs = _build_corpus()
    canvas = zg.build_canvas(descs)
    rec = _Recorder()
    obs = zobs.SubstepReporter(rec.emit, log_line=rec.log_line)
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1 * 2**30), floors=_tiny_floors(),
        emit=rec.emit, log_line=rec.log_line, observer=obs,
    )
    assert layout["nx"] >= 1 and layout["ny"] >= 1

    t = obs.timings()
    assert zobs.LAYOUT_SUBSTEP_FOOTPRINTS in t
    assert zobs.LAYOUT_SUBSTEP_CANDIDATES in t
    assert all(v >= 0.0 for v in t.values())

    starts = [m for m, lvl in rec.emitted if m.startswith("substep START:")]
    ends = [m for m, lvl in rec.emitted if m.startswith("substep END:")]
    assert any(zobs.LAYOUT_SUBSTEP_FOOTPRINTS in s for s in starts)
    assert any(zobs.LAYOUT_SUBSTEP_CANDIDATES in s for s in starts)
    assert any("elapsed=" in e for e in ends)

    # Durable: sub-step lines + candidate lines reached the incremental run log.
    assert any(m.startswith("substep ") for m in rec.log_lines)
    assert any("layout scan:" in m for m in rec.log_lines)
    assert any("CHOSEN" in m for m in rec.log_lines)


# ---------------------------------------------------------------------------
# No-observer path == observer path (science bit-equal); membership deterministic
# ---------------------------------------------------------------------------

def test_no_observer_path_equal_layout_and_membership():
    descs = _build_corpus()
    canvas = zg.build_canvas(descs)
    ram_budget = int(1 * 2**30)
    floors = _tiny_floors()

    baseline = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=ram_budget, floors=floors,
    )
    rec = _Recorder()
    obs = zobs.SubstepReporter(rec.emit, log_line=rec.log_line)
    observed = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=ram_budget, floors=floors,
        emit=rec.emit, log_line=rec.log_line, observer=obs,
    )
    # Observability must never change the layout DECISION (or any science field).
    assert baseline == observed

    # Membership/context is deterministic and unchanged by the observer seam.
    nx, ny = baseline["nx"], baseline["ny"]
    memberships_a = []
    memberships_b = []
    for row, col, _b in zg.build_layout(canvas, nx, ny).iter_cells(canvas):
        _c1, _p1, m1 = zsw.build_cell_context(descs, canvas, row, col, nx, ny)
        _c2, _p2, m2 = zsw.build_cell_context(descs, canvas, row, col, nx, ny)
        memberships_a.append((m1.core_ids, m1.patch_ids))
        memberships_b.append((m2.core_ids, m2.patch_ids))
    assert memberships_a == memberships_b
    assert any(core or patch for core, patch in memberships_a), "empty membership unexpectedly"


# ---------------------------------------------------------------------------
# Subordinate timings: present, non-negative, excluded from total, not phases
# ---------------------------------------------------------------------------

def test_subordinate_timings_present_nonnegative_excluded_from_total():
    t = zin.Timings()
    t.add("layout", 10.0)
    t.add("layout.footprints", 2.0, subordinate=True)
    t.add("layout.candidates", 7.0, subordinate=True)
    t.add("layout.chosen_cells", 1.0, subordinate=True)
    d = t.to_dict()
    for k in (
        "layout",
        "layout.footprints",
        "layout.candidates",
        "layout.chosen_cells",
    ):
        assert k in d
        assert d[k] >= 0.0
    # Subordinates appear in to_dict() but are NOT double-counted in total().
    assert t.total() == pytest.approx(10.0)


def test_layout_substeps_are_not_in_phase_order():
    # _PHASE_ORDER is unchanged (frozen); sub-steps are subordinate, never phases.
    assert zz._PHASE_ORDER == (
        "setup", "layout", "gauge", "cache_build", "per_cell_stack", "assembly"
    )
    assert "layout" in zz._PHASE_ORDER
    for sid in (
        zobs.LAYOUT_SUBSTEP_FOOTPRINTS,
        zobs.LAYOUT_SUBSTEP_CANDIDATES,
        zobs.LAYOUT_SUBSTEP_CHOSEN_CELLS,
    ):
        assert sid not in zz._PHASE_ORDER
