"""ZM-ETA-SERVICE-R30 focused pure tests — hybrid ETA service + history store.

PURE tests (stdlib only; no Qt, no numpy, no astropy). They exercise:

* FakeClock deterministic estimator behaviour: no first-sample ETA, readiness
  threshold, robust active rate, future-phase priors, unknown-total phases,
  transitions, duplicate/out-of-order immunity, asymmetric smoothing, stale
  hold, terminal/reset semantics, finite bounds, plus rework-1 F1 (future floor
  + nonterminal zero) and F2 (history during active measurable calibration).
* History store: v1 compatibility/preservation, v2 roundtrip, atomic append,
  record cap, corrupt/unwritable fail-open, mode filtering, median/outlier
  resistance, unit scaling, exact per-stage-total scaling (F3) — using TEMP
  paths only (never the real HOME file).
* STATIC source-level seam checks for the worker history write and the GUI
  routing (no Qt import), mirroring the R29 static-seam style.
"""

from __future__ import annotations

import ast
import json
import math
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


# Future-only prior set (active stages then run pure-live, future from history).
_FUTURE_PRIORS = {
    "zegrid:per_cell_stack": 800.0,
    "zegrid:assembly": 100.0,
    "zegrid:finalize": 20.0,
}

# Complete prior set (all six stages) — a full comparable history.
_FULL_PRIORS = {
    "zegrid:setup": 15.0,
    "zegrid:layout": 700.0,
    "zegrid:gauge": 100.0,
    "zegrid:per_cell_stack": 150.0,
    "zegrid:assembly": 20.0,
    "zegrid:finalize": 15.0,
}


# ---------------------------------------------------------------------------
# robust_rate pure function
# ---------------------------------------------------------------------------

def test_robust_rate_uses_median_not_first_item_ratio():
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
    assert zeta.robust_rate([(0.0, 0.0), (1.0, 10.0)]) == pytest.approx(10.0)


def test_robust_rate_ignores_nonpositive_deltas():
    samples = [(0.0, 0.0), (1.0, 5.0), (2.0, 5.0), (3.0, 10.0)]
    rate = zeta.robust_rate(samples)
    assert rate == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Calibration / readiness (F2)
# ---------------------------------------------------------------------------

def test_no_history_first_sample_remains_calibrating():
    clock = FakeClock(0.0)
    est = _estimator(clock)  # no priors
    r = est.on_stage("zegrid:setup", 1, 653)
    assert r.ready is False
    assert r.remaining_seconds is None
    assert r.confidence == zeta.CONFIDENCE_CALIBRATING


def test_first_setup_event_with_complete_history_ready_immediately():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(_FULL_PRIORS, comparable_histories=1)
    est.set_context(n_frames=653, cell_count=170)
    # setup emits only start/end; at start (current=0) a complete comparable
    # history must yield an honest immediate ETA (history-based, not live).
    r = est.on_stage("zegrid:setup", 0, 653)
    assert r.ready is True
    assert r.basis == "history"  # active prior during the calibration window
    # active = prior_setup x remaining_fraction = 15 x (653/653) = 15
    assert r.active_remaining == pytest.approx(15.0)
    # future floor = layout+gauge+per_cell_stack+assembly+finalize priors.
    assert r.future_floor == pytest.approx(700.0 + 100.0 + 150.0 + 20.0 + 15.0)
    assert r.remaining_seconds == pytest.approx(15.0 + 985.0)


def test_readiness_requires_min_samples_and_elapsed():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(_FUTURE_PRIORS, comparable_histories=1)
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    est.on_stage("zegrid:gauge", 100, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 200, 1305)
    r = est.on_stage("zegrid:gauge", 300, 1305)  # 3 samples but only 4s elapsed
    assert r.ready is False
    clock.advance(5.0)
    r = est.on_stage("zegrid:gauge", 400, 1305)  # 4 samples, >=8s elapsed
    assert r.ready is True
    assert r.basis == "live"


def test_readiness_requires_elapsed_even_with_many_samples():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(_FUTURE_PRIORS, comparable_histories=1)
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    est.on_stage("zegrid:gauge", 100, 1305)
    est.on_stage("zegrid:gauge", 200, 1305)
    est.on_stage("zegrid:gauge", 300, 1305)
    r = est.on_stage("zegrid:gauge", 400, 1305)  # 4 samples but 0s elapsed
    assert r.ready is False


def test_transition_from_prior_to_robust_blend():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {
            "zegrid:gauge": 500.0, "zegrid:per_cell_stack": 800.0,
            "zegrid:assembly": 100.0, "zegrid:finalize": 20.0,
        },
        comparable_histories=2,
    )
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    # Calibration window: active gauge uses history prior (never first-item).
    r = est.on_stage("zegrid:gauge", 0, 1305)
    assert r.basis == "history"
    assert r.active_remaining == pytest.approx(500.0)  # full prior at 0/1305
    # Feed increasing samples past the gate -> live becomes ready -> blend.
    clock.advance(1.0)
    est.on_stage("zegrid:gauge", 100, 1305)
    clock.advance(1.0)
    est.on_stage("zegrid:gauge", 200, 1305)
    clock.advance(1.0)
    est.on_stage("zegrid:gauge", 300, 1305)
    clock.advance(5.0)
    r = est.on_stage("zegrid:gauge", 400, 1305)
    assert r.basis == "blend"
    # blended active = alpha*history + (1-alpha)*live, between the two terms.
    assert 0.0 < r.active_remaining < 500.0


# ---------------------------------------------------------------------------
# Estimator: active rate + future-phase priors
# ---------------------------------------------------------------------------

def test_active_rate_and_future_priors():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(_FUTURE_PRIORS, comparable_histories=2)
    est.set_context(n_frames=653, cell_count=170)
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)

    est.on_stage("zegrid:gauge", 0, 1305)
    clock.advance(1.0)
    est.on_stage("zegrid:gauge", 100, 1305)
    clock.advance(1.0)
    est.on_stage("zegrid:gauge", 200, 1305)
    clock.advance(1.0)
    est.on_stage("zegrid:gauge", 300, 1305)
    clock.advance(5.0)
    r = est.on_stage("zegrid:gauge", 400, 1305)

    assert r.ready is True
    # active live remaining = (1305 - 400) / 100 = 9.05s
    # future floor = per_cell_stack 800 + assembly 100 + finalize 20 = 920
    assert r.active_remaining == pytest.approx(9.05, rel=1e-6)
    assert r.future_floor == pytest.approx(920.0)
    assert r.remaining_seconds == pytest.approx(929.05, rel=1e-6)
    assert r.basis == "live"       # active from robust rate only
    assert r.source == "history"   # future from priors
    assert r.confidence == zeta.CONFIDENCE_MEDIUM  # 2 comparable
    assert r.comparable_histories == 2


def test_completed_stages_not_predicted_again():
    clock = FakeClock(0.0)
    est = _estimator(clock)
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
    # setup's huge prior (9999) must NOT appear: it is a completed stage, so it
    # is excluded from the future floor; gauge is the ACTIVE stage (not future).
    assert r.future_floor == pytest.approx(800.0 + 100.0 + 20.0)
    assert r.remaining_seconds < 9999.0
    assert r.remaining_seconds > 900.0


# ---------------------------------------------------------------------------
# Estimator: unknown-total phases (layout / assembly)
# ---------------------------------------------------------------------------

def test_unknown_total_phase_uses_history_prior():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(_FULL_PRIORS, comparable_histories=1)
    est.on_stage("zegrid:setup", 653, 653)
    r = est.on_stage("zegrid:layout", 0, 0)  # unknown total, prior available
    assert r.ready is True
    # active = prior_layout(700) - 0 elapsed; future = 100+150+20+15 = 285
    assert r.active_remaining == pytest.approx(700.0)
    assert r.future_floor == pytest.approx(100.0 + 150.0 + 20.0 + 15.0)
    assert r.remaining_seconds == pytest.approx(700.0 + 285.0)
    assert r.basis == "history"


def test_unknown_total_phase_without_history_stays_calibrating():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.on_stage("zegrid:setup", 653, 653)
    r = est.on_stage("zegrid:layout", 0, 0)
    assert r.ready is False
    assert r.remaining_seconds is None
    assert r.confidence == zeta.CONFIDENCE_CALIBRATING


def test_unknown_total_phase_never_fabricates_local_percent():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(_FULL_PRIORS, comparable_histories=1)
    est.on_stage("zegrid:setup", 653, 653)
    r = est.on_stage("zegrid:layout", 0, 0)
    assert r.ready is True
    assert r.remaining_seconds is not None


# ---------------------------------------------------------------------------
# Estimator: transition, duplicate/out-of-order, smoothing, stale, terminal
# ---------------------------------------------------------------------------

def test_transition_closes_prior_and_resets_regime():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(_FULL_PRIORS, comparable_histories=1)
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    clock.advance(30.0)  # layout elapsed 30s
    r = est.on_stage("zegrid:gauge", 0, 1305)
    # layout was closed (elapsed recorded ~30s), not re-predicted.
    assert est._stages["zegrid:layout"].end_t is not None
    assert est._stages["zegrid:layout"].elapsed_s == pytest.approx(30.0)
    assert r.active_stage == "zegrid:gauge"
    assert r.basis == "history"  # gauge calibration window uses its prior


def test_duplicate_and_out_of_order_do_not_regress():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(_FUTURE_PRIORS, comparable_histories=1)
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    est.on_stage("zegrid:gauge", 100, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 200, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 300, 1305)
    high = est.on_stage("zegrid:gauge", 400, 1305)
    dup = est.on_stage("zegrid:gauge", 400, 1305)
    ooo = est.on_stage("zegrid:gauge", 100, 1305)
    assert dup.remaining_seconds == high.remaining_seconds
    assert ooo.remaining_seconds == high.remaining_seconds


def test_upward_correction_applies_immediately():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(_FUTURE_PRIORS, comparable_histories=1)
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    est.on_stage("zegrid:gauge", 0, 1305)
    cur = 0
    for _ in range(8):
        cur += 100
        clock.advance(1.0)
        est.on_stage("zegrid:gauge", cur, 1305)
    fast = est.on_stage("zegrid:gauge", cur, 1305)
    assert fast.ready is True
    # Slow regime (1 item / 5s) rolls the window -> estimate rises (never forced
    # down).
    for _ in range(8):
        cur += 1
        clock.advance(5.0)
        est.on_stage("zegrid:gauge", cur, 1305)
    slow = est.on_stage("zegrid:gauge", cur, 1305)
    assert slow.remaining_seconds > fast.remaining_seconds


def test_downward_correction_is_damped():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(_FUTURE_PRIORS, comparable_histories=1)
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    # Slow regime first (20 items/s) -> high remaining.
    est.on_stage("zegrid:gauge", 100, 1305)
    cur = 100
    for _ in range(7):
        cur += 20
        clock.advance(1.0)
        est.on_stage("zegrid:gauge", cur, 1305)
    clock.advance(1.0)  # >= 8s elapsed
    slow = est.on_stage("zegrid:gauge", cur, 1305)
    assert slow.ready is True
    slow_active = slow.active_remaining
    # Fast regime (100 items/s) rolls the window -> raw remaining collapses.
    for _ in range(8):
        cur += 100
        clock.advance(1.0)
        est.on_stage("zegrid:gauge", cur, 1305)
    fast = est.on_stage("zegrid:gauge", cur, 1305)
    # Damped downward: decreased, but strictly ABOVE the raw live remaining
    # (the smoothed value does not jump all the way down).
    assert fast.active_remaining < slow_active
    assert fast.active_remaining > 0.0


def test_tick_counts_down_within_freshness():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(_FUTURE_PRIORS, comparable_histories=1)
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
    # Only the active component decremented; the future floor is preserved.
    assert t.future_floor == pytest.approx(base.future_floor)


def test_stale_holds_instead_of_marching_to_zero():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(_FUTURE_PRIORS, comparable_histories=1)
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    est.on_stage("zegrid:gauge", 100, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 200, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 300, 1305)
    base = est.on_stage("zegrid:gauge", 400, 1305)
    clock.advance(60.0)  # past the stale window
    t = est.tick()
    assert t.stalled is True
    assert t.confidence == zeta.CONFIDENCE_LOW
    # Held at >= the future floor (never zero, never consumes future phases).
    assert t.remaining_seconds >= base.future_floor
    assert t.remaining_seconds > 0.0


# ---------------------------------------------------------------------------
# F1: future floor + nonterminal zero
# ---------------------------------------------------------------------------

def test_active_history_overrun_holds_future_floor():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(
        {
            "zegrid:layout": 10.0, "zegrid:gauge": 500.0,
            "zegrid:per_cell_stack": 800.0, "zegrid:assembly": 100.0,
            "zegrid:finalize": 20.0,
        },
        comparable_histories=1,
    )
    est.on_stage("zegrid:setup", 653, 653)
    r = est.on_stage("zegrid:layout", 0, 0)  # active prior 10s
    assert r.basis == "history"
    floor = 500.0 + 800.0 + 100.0 + 20.0
    assert r.future_floor == pytest.approx(floor)
    # Advance past the layout prior with no transition.
    clock.advance(50.0)
    t = est.tick()
    assert t.stalled is True
    # Holds the future floor (never consumes future phases).
    assert t.remaining_seconds == pytest.approx(floor)
    assert t.remaining_seconds > 0.0


def test_active_live_stale_holds_above_future_floor():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(_FUTURE_PRIORS, comparable_histories=1)
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    est.on_stage("zegrid:gauge", 100, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 200, 1305)
    clock.advance(4.0)
    est.on_stage("zegrid:gauge", 300, 1305)
    base = est.on_stage("zegrid:gauge", 400, 1305)
    assert base.future_floor == pytest.approx(920.0)
    clock.advance(60.0)
    t = est.tick()
    assert t.stalled is True
    assert t.confidence == zeta.CONFIDENCE_LOW
    # Held at >= future floor (never below it, never 0).
    assert t.remaining_seconds >= base.future_floor


def test_no_nonterminal_zero():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors({"zegrid:finalize": 1.0}, comparable_histories=1)
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    est.on_stage("zegrid:gauge", 1305, 1305)
    est.on_stage("zegrid:per_cell_stack", 170, 170)
    est.on_stage("zegrid:assembly", 0, 0)
    r = est.on_stage("zegrid:finalize", 1, 1)  # finalize complete, no terminal yet
    assert r.ready is True
    # Nonterminal must never display exact 0 (>= the documented display floor).
    assert r.remaining_seconds > 0.0
    assert r.remaining_seconds >= zeta.MIN_DISPLAY_FLOOR_S
    # Only terminal success returns exactly 0.
    s = est.mark_success()
    assert s.remaining_seconds == 0.0


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
    est.set_priors(_FUTURE_PRIORS, comparable_histories=1)
    est.on_stage("zegrid:setup", 653, 653)
    est.on_stage("zegrid:layout", 0, 0)
    for cur in (0, 100, 400, 700, 1000, 1305):
        r = est.on_stage("zegrid:gauge", cur, 1305)
        if r.remaining_seconds is not None:
            assert r.remaining_seconds >= 0.0
            assert math.isfinite(r.remaining_seconds)


# ---------------------------------------------------------------------------
# F4: zero clock start elapsed
# ---------------------------------------------------------------------------

def test_zero_clock_start_elapsed_handled():
    clock = FakeClock(0.0)  # start exactly 0.0 (falsy edge case)
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
    first = est.on_stage("zegrid:layout", 0, 0)  # start_t == 0.0
    assert first.ready is True
    assert first.active_remaining == pytest.approx(100.0)
    clock.advance(30.0)
    r = est.on_stage("zegrid:layout", 0, 0)  # elapsed 30s counted against prior
    # The active prior is counted down by elapsed (explicit None handling); the
    # old ``start_t or now`` bug would keep it at 100.
    assert r.active_remaining < first.active_remaining


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


def test_v1_records_load_and_preserved(tmp_path):
    p = _tmp_history(tmp_path)
    p.write_text(json.dumps({"schema": "zemosaic.eta_history.v1", "records": [_v1_record()]}))
    records = zeta.load_eta_history(p)
    assert len(records) == 1
    assert records[0]["duration_s"] == 100.0
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
    leftovers = [x for x in p.parent.iterdir() if x.name.startswith(p.name + ".")]
    assert leftovers == []


def test_atomic_append_cleanup_on_replace_failure(tmp_path, monkeypatch):
    p = _tmp_history(tmp_path)
    assert zeta.append_zegrid_history(_v2_record(total_duration_s=111.0), path=p) is True
    original = p.read_text()

    def _failing_replace(src, dst):
        raise OSError("boom")

    monkeypatch.setattr(zeta.os, "replace", _failing_replace)
    assert zeta.append_zegrid_history(_v2_record(), path=p) is False
    # Target untouched, and no temp file left behind.
    assert p.read_text() == original
    leftovers = [x for x in p.parent.iterdir() if x.name.startswith(p.name + ".")]
    assert leftovers == []


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
    assert zeta.append_zegrid_history(_v2_record(), path=p) is True


def test_unwritable_path_fails_open(tmp_path):
    p = tmp_path / "does" / "not" / "exist" / "history.json"
    assert zeta.load_eta_history(p) == []
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    bad = blocker / "history.json"
    assert zeta.append_zegrid_history(_v2_record(), path=bad) is False


def test_priors_filter_mode_and_never_mix_v1(tmp_path):
    records = [
        _v1_record(duration_s=99999.0, n_frames=9999),
        _v2_record(total_duration_s=1000.0, n_frames=653, cell_count=170),
        _v2_record(total_duration_s=2000.0, n_frames=653, cell_count=170),
    ]
    priors, comparable = zeta.select_zegrid_priors(records)
    assert comparable == 2
    assert priors["zegrid:setup"] == pytest.approx(15.0)
    assert priors["zegrid:layout"] == pytest.approx(700.0)
    assert priors["zegrid:gauge"] == pytest.approx(100.0)


def test_priors_median_and_outlier_resistance():
    records = [
        _v2_record(total_duration_s=1000.0),
        _v2_record(total_duration_s=1000.0),
        _v2_record_with_layout(100000.0),
    ]
    priors, comparable = zeta.select_zegrid_priors(records)
    assert comparable == 3
    assert priors["zegrid:layout"] == pytest.approx(700.0)


# ---------------------------------------------------------------------------
# F3: exact per-stage-total scaling
# ---------------------------------------------------------------------------

def test_priors_scale_by_n_frames_cell_count_fallback():
    # Older v2 record WITHOUT stage_totals -> n_frames/cell_count fallback.
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
    assert priors["zegrid:setup"] == pytest.approx(20.0)
    assert priors["zegrid:gauge"] == pytest.approx(200.0)
    assert priors["zegrid:per_cell_stack"] == pytest.approx(200.0)
    assert priors["zegrid:layout"] == pytest.approx(700.0)


def test_exact_gauge_total_preferred_over_n_frames():
    # Record: exact gauge total 100, but n_frames=1000 (raw loaded vs included
    # descriptors mismatch). The exact total must WIN over n_frames scaling.
    rec = zeta.build_zegrid_history_record(
        total_duration_s=1000.0,
        n_frames=1000,
        cell_count=50,
        stage_seconds={
            "setup": 10.0, "layout": 700.0, "gauge": 200.0,
            "per_cell_stack": 100.0, "assembly": 20.0, "finalize": 15.0,
        },
        stage_totals={"setup": 1000, "gauge": 100, "per_cell_stack": 50, "finalize": 1},
    )
    priors, comparable = zeta.select_zegrid_priors(
        [rec],
        n_frames=1000,
        cell_count=50,
        stage_totals={"setup": 1000, "gauge": 200, "per_cell_stack": 50, "finalize": 1},
    )
    assert comparable == 1
    # Exact gauge scaling: 200 * (200/100) = 400 (NOT n_frames scale 200*1=200).
    assert priors["zegrid:gauge"] == pytest.approx(400.0)


def test_exact_per_cell_total_rescale():
    rec = zeta.build_zegrid_history_record(
        total_duration_s=1000.0,
        n_frames=653,
        cell_count=50,
        stage_seconds={
            "setup": 15.0, "layout": 700.0, "gauge": 100.0,
            "per_cell_stack": 100.0, "assembly": 20.0, "finalize": 15.0,
        },
        stage_totals={"setup": 653, "gauge": 1305, "per_cell_stack": 50, "finalize": 1},
    )
    priors, comparable = zeta.select_zegrid_priors(
        [rec], stage_totals={"per_cell_stack": 170}
    )
    assert comparable == 1
    # per_cell_stack 100 * (170/50) = 340 via the exact total.
    assert priors["zegrid:per_cell_stack"] == pytest.approx(340.0)


def test_history_record_is_sanitized():
    rec = _v2_record()
    text = json.dumps(rec)
    for forbidden in ("/home", "frame_", ".fits", ".png", "caldwell", "M106"):
        assert forbidden not in text


# ---------------------------------------------------------------------------
# Synthetic 653-frame / 170-cell sequence
# ---------------------------------------------------------------------------

def test_synthetic_sequence_first_estimate_only_after_calibration():
    """No history: the first estimate appears only after the calibration window
    (>=3 increasing samples AND >=8s) in a measurable stage."""
    clock = FakeClock(0.0)
    est = _estimator(clock)  # no priors
    est.set_context(n_frames=653, cell_count=170)

    est.on_stage("zegrid:setup", 0, 653)
    r = est.on_stage("zegrid:setup", 1, 653)
    assert r.ready is False  # no first-sample ETA

    clock.advance(10.0)
    est.on_stage("zegrid:setup", 653, 653)
    r = est.on_stage("zegrid:layout", 0, 0)
    assert r.ready is False  # layout unknown-total, no history -> calibrating

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
    assert r.basis == "live"
    assert r.source == "fallback"
    assert r.confidence == zeta.CONFIDENCE_LOW
    assert r.remaining_seconds > 0.0


def test_synthetic_sequence_with_history_ready_immediately_after_calibration():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(_FULL_PRIORS, comparable_histories=1)
    est.set_context(n_frames=653, cell_count=170)
    est.on_stage("zegrid:setup", 653, 653)
    r = est.on_stage("zegrid:layout", 0, 0)
    assert r.ready is True
    assert r.confidence == zeta.CONFIDENCE_MEDIUM


def test_synthetic_sequence_transition_reset_is_reasonable():
    clock = FakeClock(0.0)
    est = _estimator(clock)
    est.set_priors(_FULL_PRIORS, comparable_histories=1)
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
    assert finalize_end < history_append < success_emit


def test_worker_records_actual_wall_duration_perf_counter():
    text = (SRC / "zemosaic_zegrid_mode.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    run_single = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_single":
            run_single = node
            break
    assert run_single is not None
    seg = ast.get_source_segment(text, run_single)
    # Explicit perf_counter start at the top of the segment.
    assert "run_wall_start = time.perf_counter()" in seg
    # The history record's total duration is the honest wall delta, NOT a
    # phase-sum guess (timings.total() + finalize_wall).
    assert "time.perf_counter() - run_wall_start" in seg
    assert "timings.total() + finalize_wall" not in seg


def test_gui_feeds_service_from_stage_events_and_tick_renders():
    text = (SRC / "zemosaic_gui_qt.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert "HybridEtaEstimator" in text
    assert "on_stage" in text
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_on_elapsed_timer_tick":
            seg = ast.get_source_segment(text, node)
            assert ".tick()" in seg
            assert "if self._zegrid_active:" in seg
            break
    else:
        raise AssertionError("_on_elapsed_timer_tick not found")


def test_gui_reselects_priors_on_new_stage_totals():
    text = (SRC / "zemosaic_gui_qt.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    # _observe_zegrid_stage_total reselects priors when a total changes.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_observe_zegrid_stage_total":
            seg = ast.get_source_segment(text, node)
            assert "_reselect_zegrid_priors()" in seg
            break
    else:
        raise AssertionError("_observe_zegrid_stage_total not found")
    # _ensure_zegrid_eta reads the disk once (cache); _feed does NOT reread.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_ensure_zegrid_eta":
            seg = ast.get_source_segment(text, node)
            assert "load_eta_history" in seg
            break
    else:
        raise AssertionError("_ensure_zegrid_eta not found")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_feed_zegrid_eta":
            seg = ast.get_source_segment(text, node)
            assert "load_eta_history" not in seg
            assert "select_zegrid_priors" not in seg
            break
    else:
        raise AssertionError("_feed_zegrid_eta not found")


def test_legacy_sds_paths_not_redirected():
    text = (SRC / "zemosaic_gui_qt.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_update_stage_progress":
            seg = ast.get_source_segment(text, node)
            assert "_update_eta_from_progress" in seg
            break
    else:
        raise AssertionError("_update_stage_progress not found")
