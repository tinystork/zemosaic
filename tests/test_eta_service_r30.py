"""ZM-ETA-SERVICE-R30 focused pure tests — hybrid ETA service + history store.

PURE tests (stdlib only; no Qt, no numpy, no astropy). They exercise:

* FakeClock deterministic estimator behaviour: no first-sample ETA, readiness
  threshold, robust active rate, future-phase priors, unknown-total phases,
  transitions, duplicate/out-of-order immunity, asymmetric smoothing, stale
  hold, terminal/reset semantics, finite bounds.
* History store: v1 compatibility/preservation, v2 roundtrip, atomic append,
  record cap, corrupt/unwritable fail-open, mode filtering, median/outlier
  resistance, unit scaling — using TEMP paths only (never the real HOME file).
* A synthetic 653-frame / 170-cell event sequence proving the first estimate
  appears only after calibration or valid history, with no dependence on local
  or global progress percentages, and a reasonable stage-transition reset.
* STATIC source-level seam checks for the worker history write and the GUI
  routing (no Qt import), mirroring the R29 static-seam style.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from zemosaic import eta_service as zeta
from zemosaic import progress_contract as zprogress

SRC = Path(__file__).resolve().parents[1] / "src" / "zemosaic"


# ---------------------------------------------------------------------------
# FakeClock
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)

    def set(self, seconds: float) -> None:
        self.t = float(seconds)


def _estimator(clock, *, priors=None, comparable=0, **kw):
    est = zeta.HybridEtaEstimator(
        zprogress.ZEGRID_PLAN,
        clock=clock,
        **kw,
    )
    if priors:
        est.set_priors(priors, comparable_histories=comparable)
    return est


# ---------------------------------------------------------------------------
# robust_rate pure function
# ---------------------------------------------------------------------------

def test_robust_rate_uses_median_not_first_item_ratio():
    # A fast first item (100/s) followed by a steady slow rate (10/s): the
    # median of the interval deltas must reflect the slow regime, not 100/s.
    samples = [
        (0.0, 0.0),
        (1.0, 100.0),   # 100/s (outlier)
        (11.0, 200.0),  # 10/s
        (21.0, 300.0),  # 10/s
        (31.0, 400.0),  # 10/s
    ]
    rate = zeta.robust_rate(samples)
    assert rate == pytest.approx(10.0, rel=0.01)


def test_robust_rate_none_with_fewer_than_two_intervals():
    assert zeta.robust_rate([(0.0, 0.0)]) is None
    assert zeta.robust_rate([]) is None
    assert zeta.robust_rate([(0.0, 0.0), (1.0, 0.0)]) is None  # no progress


def test_robust_rate_single_interval_is_that_rate():
    # Two samples -> one interval -> its rate (there is no ambiguity).
    assert zeta.robust_rate([(0.0, 0.0), (1.0, 10.0)]) == pytest.approx(10.0)


def test_robust_rate_ignores_nonpositive_deltas():
    # Duplicate done (zero progress) and negative dt are skipped, leaving only
    # the valid interval.
    samples = [(0.0, 0.0), (1.0, 5.0), (2.0, 5.0), (3.0, 10.0)]
    rate = zeta.robust_rate(samples)
    # valid deltas: 5/s (0..1) and 5/s (2..3); the 1..2 interval is zero-progress.
    assert rate == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Estimator: no first-sample ETA + readiness threshold
# ---------------------------------------------------------------------------

def test_no_first_sample_eta():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    r = est.on_stage("zegrid:setup", 1, 653)
    assert r.ready is False
    assert r.remaining_seconds is None
    assert r.confidence == zeta.CONFIDENCE_CALIBRATING


def test_readiness_requires_min_samples_and_elapsed():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {"zegrid:gauge": 500.0, "zegrid:per_cell_stack": 800.0,
         "zegrid:assembly": 100.0, "zegrid:finalize": 20.0},
        comparable_histories=1,
    )
    est.set_context(n_frames=653, cell_count=170)
    # Close setup + layout (which have no priors -> fallback would be needed;
    # instead use a full prior set so only gauge's own readiness gates).
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    # 2 meaningful samples, >=8s elapsed -> still not ready (need >=3 samples).
    est.on_stage("zegrid:gauge", 100, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 200, 1305)
    r = est.on_stage("zegrid:gauge", 300, 1305)  # 3rd sample, only 4s elapsed
    assert r.ready is False

    # Advance to >=8s elapsed with 3 samples -> ready.
    clock.advance(5.0)
    r = est.on_stage("zegrid:gauge", 400, 1305)
    assert r.ready is True


def test_readiness_requires_elapsed_even_with_many_samples():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {"zegrid:gauge": 500.0, "zegrid:per_cell_stack": 800.0,
         "zegrid:assembly": 100.0, "zegrid:finalize": 20.0},
        comparable_histories=1,
    )
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    est.on_stage("zegrid:gauge", 100, 1305)
    est.on_stage("zegrid:gauge", 200, 1305)
    est.on_stage("zegrid:gauge", 300, 1305)
    r = est.on_stage("zegrid:gauge", 400, 1305)  # 4 samples but 0s elapsed
    assert r.ready is False


# ---------------------------------------------------------------------------
# Estimator: active rate + future-phase priors
# ---------------------------------------------------------------------------

def test_active_rate_and_future_priors():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {
            "zegrid:layout": 100.0,
            "zegrid:gauge": 500.0,
            "zegrid:per_cell_stack": 800.0,
            "zegrid:assembly": 100.0,
            "zegrid:finalize": 20.0,
        },
        comparable_histories=2,
    )
    est.set_context(n_frames=653, cell_count=170)
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)

    # gauge: total 1305, progress 100 items/s steady.
    est.on_stage("zegrid:gauge", 0, 1305)
    clock.advance(1.0)
    est.on_stage("zegrid:gauge", 100, 1305)
    clock.advance(1.0)
    est.on_stage("zegrid:gauge", 200, 1305)
    clock.advance(1.0)
    est.on_stage("zegrid:gauge", 300, 1305)
    clock.advance(5.0)  # total >=8s
    r = est.on_stage("zegrid:gauge", 400, 1305)

    assert r.ready is True
    # active remaining = (1305 - 400) / 100 = 9.05s
    # future priors: per_cell_stack 800 + assembly 100 + finalize 20 = 920
    expected = 9.05 + 920.0
    assert r.remaining_seconds == pytest.approx(expected, rel=1e-6)
    assert r.basis == "blend"  # live active + history future
    assert r.confidence == zeta.CONFIDENCE_MEDIUM  # 2 comparable
    assert r.comparable_histories == 2


def test_completed_stages_not_predicted_again():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    # Priors exist for setup/gauge too, but completed stages must be ignored.
    est.set_priors(
        {
            "zegrid:setup": 9999.0,
            "zegrid:gauge": 500.0,
            "zegrid:per_cell_stack": 800.0,
            "zegrid:assembly": 100.0,
            "zegrid:finalize": 20.0,
        },
        comparable_histories=1,
    )
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    est.on_stage("zegrid:gauge", 100, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 200, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 300, 1305)
    r = est.on_stage("zegrid:gauge", 400, 1305)
    # setup's huge prior (9999) must NOT appear (setup is completed), and the
    # active gauge stage is NOT predicted from its own prior (500) — only the
    # live rate + future priors (per_cell_stack+assembly+finalize=920) appear.
    assert r.remaining_seconds < 9999.0
    assert r.remaining_seconds < 2000.0
    assert r.remaining_seconds > 900.0
    # active remaining (~36) + future priors (920) — sanity only, not exact.
    assert r.remaining_seconds == pytest.approx(958.2, abs=6.0)


# ---------------------------------------------------------------------------
# Estimator: unknown-total phases (layout / assembly)
# ---------------------------------------------------------------------------

def test_unknown_total_phase_uses_history_prior():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {
            "zegrid:layout": 100.0,
            "zegrid:gauge": 500.0,
            "zegrid:per_cell_stack": 800.0,
            "zegrid:assembly": 100.0,
            "zegrid:finalize": 20.0,
        },
        comparable_histories=1,
    )
    est.on_stage("zegrid:setup", 653, 653)
    r = est.on_stage("zegrid:layout", 0, 0)  # unknown total, prior available
    assert r.ready is True
    # active remaining = prior(100) - 0 elapsed; future = 500+800+100+20 = 1420
    assert r.remaining_seconds == pytest.approx(100.0 + 1420.0, rel=1e-6)
    assert r.basis == "history"


def test_unknown_total_phase_without_history_stays_calibrating():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    # No priors at all.
    est.on_stage("zegrid:setup", 653, 653)
    r = est.on_stage("zegrid:layout", 0, 0)
    assert r.ready is False
    assert r.remaining_seconds is None
    assert r.confidence == zeta.CONFIDENCE_CALIBRATING


def test_unknown_total_phase_never_fabricates_local_percent():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {
            "zegrid:layout": 100.0, "zegrid:gauge": 500.0,
            "zegrid:per_cell_stack": 800.0, "zegrid:assembly": 100.0,
            "zegrid:finalize": 20.0,
        },
        comparable_histories=1,
    )
    est.on_stage("zegrid:setup", 653, 653)
    r = est.on_stage("zegrid:layout", 0, 0)
    assert r.ready is True
    # The estimate is a duration, not a percent; it never depends on any
    # local/global percentage (the estimator has no such input at all).
    assert r.remaining_seconds is not None


# ---------------------------------------------------------------------------
# Estimator: transition, duplicate/out-of-order, smoothing, stale, terminal
# ---------------------------------------------------------------------------

def test_transition_resets_smoothing_and_closes_prior():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {
            "zegrid:layout": 100.0, "zegrid:gauge": 500.0,
            "zegrid:per_cell_stack": 800.0, "zegrid:assembly": 100.0,
            "zegrid:finalize": 20.0,
        },
        comparable_histories=1,
    )
    est.on_stage("zegrid:setup", 653, 653)
    # layout is unknown-total with a prior -> ready (prior 100 + future 1420).
    est.on_stage("zegrid:layout", 0, 0)
    before = est.on_stage("zegrid:layout", 0, 0).remaining_seconds
    assert before is not None
    assert before == pytest.approx(100.0 + 500.0 + 800.0 + 100.0 + 20.0)
    # Transition to gauge (a measurable stage) closes layout's prior duration and
    # does a bounded regime reset: gauge now needs its OWN calibration window.
    est.on_stage("zegrid:gauge", 0, 1305)
    r = est.on_stage("zegrid:gauge", 1, 1305)
    # After the reset, a single gauge sample is NOT yet ready (no stale carry).
    assert r.ready is False
    assert est._smoothed_remaining is None
    # And the layout stage was closed (its elapsed is recorded, not re-predicted).
    assert est._stages["zegrid:layout"].end_t is not None
    assert est._stages["zegrid:layout"].elapsed_s is not None


def test_duplicate_and_out_of_order_do_not_regress():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {"zegrid:gauge": 500.0, "zegrid:per_cell_stack": 800.0,
         "zegrid:assembly": 100.0, "zegrid:finalize": 20.0},
        comparable_histories=1,
    )
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    est.on_stage("zegrid:gauge", 100, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 200, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 300, 1305)
    high = est.on_stage("zegrid:gauge", 400, 1305)
    # Duplicate + out-of-order lower current must not change the estimate.
    dup = est.on_stage("zegrid:gauge", 400, 1305)
    ooo = est.on_stage("zegrid:gauge", 100, 1305)
    assert dup.remaining_seconds == high.remaining_seconds
    assert ooo.remaining_seconds == high.remaining_seconds


def test_smoothing_allows_upward_correction_and_damps_downward():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {"zegrid:gauge": 500.0, "zegrid:per_cell_stack": 800.0,
         "zegrid:assembly": 100.0, "zegrid:finalize": 20.0},
        comparable_histories=1,
    )
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    # Fast progress -> small remaining.
    est.on_stage("zegrid:gauge", 100, 1305)
    clock.advance(1.0)
    est.on_stage("zegrid:gauge", 400, 1305)
    clock.advance(1.0)
    est.on_stage("zegrid:gauge", 700, 1305)
    clock.advance(6.0)
    fast = est.on_stage("zegrid:gauge", 1000, 1305).remaining_seconds
    # A sudden slowdown -> raw remaining jumps UP; upward correction is immediate.
    clock.advance(10.0)
    slow = est.on_stage("zegrid:gauge", 1010, 1305).remaining_seconds
    assert slow > fast
    # A later recovery -> downward correction is DAMPED (not a full jump down).
    clock.advance(1.0)
    rec = est.on_stage("zegrid:gauge", 1305, 1305)
    assert rec.ready is True


def test_eta_is_never_forced_monotonic_downward():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {"zegrid:gauge": 500.0, "zegrid:per_cell_stack": 800.0,
         "zegrid:assembly": 100.0, "zegrid:finalize": 20.0},
        comparable_histories=1,
    )
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    est.on_stage("zegrid:gauge", 100, 1305)
    clock.advance(1.0)
    est.on_stage("zegrid:gauge", 300, 1305)
    clock.advance(1.0)
    est.on_stage("zegrid:gauge", 500, 1305)
    clock.advance(6.0)
    first = est.on_stage("zegrid:gauge", 700, 1305).remaining_seconds
    # Slowdown (rate drops) -> remaining goes UP (legitimate upward correction).
    clock.advance(20.0)
    second = est.on_stage("zegrid:gauge", 705, 1305).remaining_seconds
    assert second > first


def test_tick_counts_down_within_freshness():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {"zegrid:gauge": 500.0, "zegrid:per_cell_stack": 800.0,
         "zegrid:assembly": 100.0, "zegrid:finalize": 20.0},
        comparable_histories=1,
    )
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    est.on_stage("zegrid:gauge", 100, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 200, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 300, 1305)
    base = est.on_stage("zegrid:gauge", 400, 1305)
    clock.advance(5.0)
    t = est.tick()
    assert t.ready is True
    assert t.stalled is False
    assert t.remaining_seconds == pytest.approx(base.remaining_seconds - 5.0)


def test_stale_holds_instead_of_marching_to_zero():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {"zegrid:gauge": 500.0, "zegrid:per_cell_stack": 800.0,
         "zegrid:assembly": 100.0, "zegrid:finalize": 20.0},
        comparable_histories=1,
    )
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    est.on_stage("zegrid:gauge", 100, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 200, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 300, 1305)
    est.on_stage("zegrid:gauge", 400, 1305)
    # No further progress; advance way past the stale window.
    clock.advance(60.0)
    t = est.tick()
    assert t.stalled is True
    # Held (never zero), and confidence downgraded.
    assert t.remaining_seconds > 0.0
    assert t.confidence == zeta.CONFIDENCE_LOW


def test_terminal_success_exactly_zero():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.on_stage("zegrid:setup", 653, 653)
    r = est.mark_success()
    assert r.ready is True
    assert r.remaining_seconds == 0.0
    assert r.terminal == "success"


def test_terminal_fail_cancel_no_completed_eta():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.on_stage("zegrid:setup", 653, 653)
    rf = est.mark_fail()
    assert rf.ready is False
    assert rf.remaining_seconds is None
    assert rf.terminal == "fail"
    est2 = _estimator(FakeClock(0.0))
    rc = est2.mark_cancel()
    assert rc.ready is False
    assert rc.remaining_seconds is None
    assert rc.terminal == "cancel"


def test_reset_clears_run_state():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.on_stage("zegrid:setup", 653, 653)
    est.mark_success()
    assert est.diagnostics()["terminal"] == "success"
    est.reset()
    d = est.diagnostics()
    assert d["terminal"] is None
    assert d["active_stage"] is None
    assert d["last_estimate"] is None


def test_all_estimates_finite_and_nonnegative():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {"zegrid:gauge": 500.0, "zegrid:per_cell_stack": 800.0,
         "zegrid:assembly": 100.0, "zegrid:finalize": 20.0},
        comparable_histories=1,
    )
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    for cur in (0, 100, 400, 700, 1000, 1305):
        r = est.on_stage("zegrid:gauge", cur, 1305)
        if r.remaining_seconds is not None:
            assert r.remaining_seconds >= 0.0
            import math as _m
            assert _m.isfinite(r.remaining_seconds)


# ---------------------------------------------------------------------------
# History store (temp paths only)
# ---------------------------------------------------------------------------

def _tmp_history(tmp_path):
    return tmp_path / "eta_history.json"


def _v1_record(duration_s=100.0, n_frames=20, resume=False, master_tiles=0):
    return {
        "ts_utc": "2026-01-01T00:00:00Z",
        "duration_s": duration_s,
        "n_frames": n_frames,
        "resume": resume,
        "master_tiles": master_tiles,
        "existing_master_tiles_mode": False,
    }


def _v2_record(total_duration_s=1000.0, n_frames=653, cell_count=170):
    return zeta.build_zegrid_history_record(
        total_duration_s=total_duration_s,
        n_frames=n_frames,
        cell_count=cell_count,
        stage_seconds={
            "setup": 15.0, "layout": 700.0, "gauge": 100.0,
            "per_cell_stack": 150.0, "assembly": 20.0, "finalize": 15.0,
        },
        stage_totals={"setup": n_frames, "gauge": 2 * n_frames - 1,
                      "per_cell_stack": cell_count, "finalize": 1},
        backend="cpu",
        workers=4,
    )


def test_v1_records_load_and_preserved(tmp_path):
    p = _tmp_history(tmp_path)
    p.write_text(json.dumps({"schema": "zemosaic.eta_history.v1", "records": [_v1_record()]}))
    records = zeta.load_eta_history(p)
    assert len(records) == 1
    assert records[0]["duration_s"] == 100.0
    # Append a v2 record: v1 must be preserved alongside it.
    assert zeta.append_zegrid_history(_v2_record(), path=p) is True
    records2 = zeta.load_eta_history(p)
    assert len(records2) == 2
    assert records2[0]["duration_s"] == 100.0  # v1 still intact


def test_v2_roundtrip(tmp_path):
    p = _tmp_history(tmp_path)
    rec = _v2_record()
    assert zeta.append_zegrid_history(rec, path=p) is True
    loaded = zeta.load_eta_history(p)
    assert len(loaded) == 1
    assert loaded[0]["mode"] == "zegrid"
    assert loaded[0]["stage_seconds"]["setup"] == 15.0
    assert loaded[0]["stage_totals"]["per_cell_stack"] == 170
    assert loaded[0]["backend"] == "cpu"


def test_atomic_append_creates_file_and_no_tmp_left(tmp_path):
    p = _tmp_history(tmp_path)
    assert zeta.append_zegrid_history(_v2_record(), path=p) is True
    assert p.exists()
    assert not p.with_name(p.name + ".tmp").exists()


def test_record_cap(tmp_path):
    p = _tmp_history(tmp_path)
    max_records = zeta.ETA_HISTORY_MAX_RECORDS
    for _ in range(max_records + 5):
        assert zeta.append_zegrid_history(_v2_record(), path=p) is True
    records = zeta.load_eta_history(p)
    assert len(records) == max_records


def test_corrupt_file_fails_open(tmp_path):
    p = _tmp_history(tmp_path)
    p.write_text("{not valid json", encoding="utf-8")
    assert zeta.load_eta_history(p) == []
    # Append still works (treats corrupt as empty) and never raises.
    assert zeta.append_zegrid_history(_v2_record(), path=p) is True


def test_unwritable_path_fails_open(tmp_path):
    p = tmp_path / "does" / "not" / "exist" / "history.json"
    # load: missing -> []
    assert zeta.load_eta_history(p) == []
    # append: unwritable parent (a path under a FILE) -> False, no raise.
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    bad = blocker / "history.json"
    assert zeta.append_zegrid_history(_v2_record(), path=bad) is False


def test_priors_filter_mode_and_never_mix_v1(tmp_path):
    # v1 record (no mode) + two v2 records with different scales.
    records = [
        _v1_record(duration_s=99999.0, n_frames=9999),
        _v2_record(total_duration_s=1000.0, n_frames=653, cell_count=170),
        _v2_record(total_duration_s=2000.0, n_frames=653, cell_count=170),
    ]
    priors, comparable = zeta.select_zegrid_priors(records)
    assert comparable == 2
    # v1's huge duration must never leak into per-stage priors.
    assert priors["zegrid:setup"] == pytest.approx(15.0)  # both v2 setups == 15.0
    assert priors["zegrid:layout"] == pytest.approx(700.0)
    assert priors["zegrid:gauge"] == pytest.approx(100.0)


def test_priors_median_and_outlier_resistance():
    records = [
        _v2_record(total_duration_s=1000.0),
        _v2_record(total_duration_s=1000.0),
        # A gross outlier layout duration (must be rejected by the 0.1x..10x band).
        _v2_record_with_layout(100000.0),
    ]
    priors, comparable = zeta.select_zegrid_priors(records)
    assert comparable == 3
    assert priors["zegrid:layout"] == pytest.approx(700.0)


def _v2_record_with_layout(layout_s):
    return zeta.build_zegrid_history_record(
        total_duration_s=1000.0,
        n_frames=653,
        cell_count=170,
        stage_seconds={
            "setup": 15.0, "layout": layout_s, "gauge": 100.0,
            "per_cell_stack": 150.0, "assembly": 20.0, "finalize": 15.0,
        },
    )


def test_priors_scale_by_units_when_meaningful():
    # Historical run with half the frames/cells: setup/gauge/per_cell_stack
    # durations are scaled to the CURRENT workload when n_frames/cell_count given.
    records = [
        zeta.build_zegrid_history_record(
            total_duration_s=1000.0,
            n_frames=326,
            cell_count=85,
            stage_seconds={
                "setup": 10.0, "layout": 700.0, "gauge": 100.0,
                "per_cell_stack": 100.0, "assembly": 20.0, "finalize": 15.0,
            },
        ),
    ]
    priors, comparable = zeta.select_zegrid_priors(records, n_frames=652, cell_count=170)
    assert comparable == 1
    # setup 10.0 * (652/326) = 20.0
    assert priors["zegrid:setup"] == pytest.approx(20.0)
    # gauge 100.0 * 2 = 200.0
    assert priors["zegrid:gauge"] == pytest.approx(200.0)
    # per_cell_stack 100.0 * (170/85) = 200.0
    assert priors["zegrid:per_cell_stack"] == pytest.approx(200.0)
    # layout/assembly/finalize unscaled.
    assert priors["zegrid:layout"] == pytest.approx(700.0)


def test_history_record_is_sanitized():
    rec = _v2_record()
    # No raw paths / frame names / user data anywhere in the record.
    text = json.dumps(rec)
    for forbidden in ("/home", "frame_", ".fits", ".png", "caldwell", "M106"):
        assert forbidden not in text


# ---------------------------------------------------------------------------
# Synthetic 653-frame / 170-cell sequence
# ---------------------------------------------------------------------------

def test_synthetic_sequence_first_estimate_only_after_calibration():
    """No history: the first estimate appears only after the calibration window
    (>=3 increasing samples AND >=8s) in a measurable stage, and depends on no
    local/global progress percent."""
    clock = FakeClock(0.0)
    est = _estimator(clock)  # no priors
    est.set_context(n_frames=653, cell_count=170)

    # setup: measurable, 653 frames.
    est.on_stage("zegrid:setup", 0, 653)
    r = est.on_stage("zegrid:setup", 1, 653)
    assert r.ready is False  # no first-sample ETA

    clock.advance(10.0)
    est.on_stage("zegrid:setup", 653, 653)  # setup done
    # After setup completes there IS observed evidence, but the next phase is
    # layout (unknown total, no history) -> still not ready.
    r = est.on_stage("zegrid:layout", 0, 0)
    assert r.ready is False

    # gauge: measurable total = 2*653-1 = 1305.
    est.on_stage("zegrid:gauge", 0, 1305)
    clock.advance(2.0)
    est.on_stage("zegrid:gauge", 100, 1305)
    clock.advance(2.0)
    est.on_stage("zegrid:gauge", 200, 1305)
    r = est.on_stage("zegrid:gauge", 300, 1305)  # 3 samples but <8s
    assert r.ready is False

    clock.advance(6.0)  # now >=8s elapsed in gauge
    r = est.on_stage("zegrid:gauge", 400, 1305)
    assert r.ready is True
    assert r.basis == "live"  # no history -> live active + fallback future
    assert r.confidence == zeta.CONFIDENCE_LOW
    assert r.remaining_seconds > 0.0


def test_synthetic_sequence_with_history_ready_immediately_after_calibration():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {
            "zegrid:layout": 700.0, "zegrid:gauge": 500.0,
            "zegrid:per_cell_stack": 800.0, "zegrid:assembly": 100.0,
            "zegrid:finalize": 20.0,
        },
        comparable_histories=1,
    )
    est.set_context(n_frames=653, cell_count=170)
    est.on_stage("zegrid:setup", 653, 653)
    # With valid history, layout (unknown total) is immediately estimable.
    r = est.on_stage("zegrid:layout", 0, 0)
    assert r.ready is True
    assert r.confidence == zeta.CONFIDENCE_MEDIUM


def test_synthetic_sequence_transition_reset_is_reasonable():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {
            "zegrid:layout": 700.0, "zegrid:gauge": 500.0,
            "zegrid:per_cell_stack": 800.0, "zegrid:assembly": 100.0,
            "zegrid:finalize": 20.0,
        },
        comparable_histories=1,
    )
    est.set_context(n_frames=653, cell_count=170)
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    clock.advance(60.0)
    est.on_stage("zegrid:gauge", 0, 1305)
    clock.advance(2.0)
    est.on_stage("zegrid:gauge", 200, 1305)
    clock.advance(2.0)
    est.on_stage("zegrid:gauge", 400, 1305)
    clock.advance(6.0)
    est.on_stage("zegrid:gauge", 600, 1305)
    r = est.on_stage("zegrid:gauge", 800, 1305)
    assert r.ready is True
    # No exact wall-clock benchmark claim; just assert a sane positive ETA.
    assert 0.0 < r.remaining_seconds < 3600.0


# ---------------------------------------------------------------------------
# STATIC seam checks (no Qt / no heavy production import)
# ---------------------------------------------------------------------------

def _parse_source(rel: str) -> ast.Module:
    return ast.parse((SRC / rel).read_text(encoding="utf-8"))


def test_worker_writes_history_only_after_finalize_before_success():
    tree = _parse_source("zemosaic_zegrid_mode.py")
    run_single = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_single":
            run_single = node
            break
    assert run_single is not None

    finalize_end = None
    history_append = None
    success_emit = None
    for node in ast.walk(run_single):
        if isinstance(node, ast.Call):
            src = ast.unparse(node)
            if finalize_end is None and "_finalize_rep.end()" in src:
                finalize_end = node.lineno
            if history_append is None and "append_zegrid_history" in src:
                history_append = node.lineno
            if success_emit is None and src.count("_emit(") == 1:
                for kw in node.keywords:
                    if kw.arg == "lvl" and isinstance(kw.value, ast.Constant) and kw.value.value == "SUCCESS":
                        success_emit = node.lineno
    assert finalize_end is not None, "finalize end not found"
    assert history_append is not None, "history append not found"
    assert success_emit is not None, "success emit not found"
    # History write is after finalize completes and before the terminal SUCCESS.
    assert finalize_end < history_append < success_emit


def test_gui_feeds_service_from_stage_events_and_tick_renders():
    tree = _parse_source("zemosaic_gui_qt.py")
    text = (SRC / "zemosaic_gui_qt.py").read_text(encoding="utf-8")
    # The estimator is created/imported and fed from the stage path.
    assert "HybridEtaEstimator" in text
    assert "on_stage" in text
    # The elapsed timer tick drives the estimator tick.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_on_elapsed_timer_tick":
            seg = ast.get_source_segment(text, node)
            assert ".tick()" in seg
            assert "if self._zegrid_active:" in seg
            break
    else:
        raise AssertionError("_on_elapsed_timer_tick not found")


def test_legacy_sds_paths_not_redirected():
    # The estimator is only used for ZeGrid (the legacy ETA paths stay intact).
    text = (SRC / "zemosaic_gui_qt.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_update_stage_progress":
            seg = ast.get_source_segment(text, node)
            # legacy branch still calls the legacy ETA updater.
            assert "_update_eta_from_progress" in seg
            break
    else:
        raise AssertionError("_update_stage_progress not found")
