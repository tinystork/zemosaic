"""ZM-ETA-ALLMODES-R31 focused tests — all-mode hybrid ETA + generic history.

PURE tests (stdlib only; no Qt, no numpy, no astropy) plus STATIC source-level
GUI seam checks (no Qt import), mirroring the R29/R30 style. They exercise:

* Per-mode ETA fallback cost models (legacy/SDS) — normalized, positive, and
  DISTINCT from the R29 UI progress weights.
* Generic history APIs: mode isolation, v1 legacy bootstrap ONLY for legacy,
  v2 preference over v1, partial-record tolerance, exact stage-total scaling,
  robust median/outlier resistance, and backward-compatible ZeGrid wrappers.
* Success observation export: closes the active stage timing and returns only
  sanitized observed stage seconds.
* Deterministic synthetic legacy sequence (PHASE_UPDATE + phase1/3/5 progress):
  no first-item rate, immediate v1 bootstrap prior, phases without totals
  handled, transition/future-floor/stale/terminal.
* Deterministic synthetic SDS sequence: no cross-mode priors, honest first-run
  calibration, later v2 priors, phase-4 throughput, terminal/reset.
* Static GUI seams: PHASE_UPDATE 1..7/4.5 mapping, legacy/SDS feed routing,
  mode-switch reset, old ETA writer guards, success-only persistence, and
  Roman locale keys/formatter.
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


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)

    def set(self, seconds: float) -> None:
        self.t = float(seconds)


def _legacy_est(clock, *, priors=None, comparable=0):
    est = zeta.HybridEtaEstimator(zprogress.LEGACY_PLAN, clock=clock)
    if priors:
        est.set_priors(priors, comparable_histories=comparable)
    return est


def _sds_est(clock, *, priors=None, comparable=0):
    est = zeta.HybridEtaEstimator(zprogress.SDS_PLAN, clock=clock)
    if priors:
        est.set_priors(priors, comparable_histories=comparable)
    return est


# ---------------------------------------------------------------------------
# A. Cost models validate/normalize and differ from UI weights
# ---------------------------------------------------------------------------

def test_cost_models_normalize_to_one():
    for model in (zeta.LEGACY_COST_MODEL, zeta.SDS_COST_MODEL):
        norm = zeta.normalize_cost_model(model)
        assert norm
        assert math.isclose(sum(norm.values()), 1.0, rel_tol=1e-9)
        assert all(v > 0.0 for v in norm.values())


def test_legacy_cost_model_differs_from_ui_weights():
    # UI progress weights (R29) are 30/5/35/5/6/9/8/2; the ETA fallback cost
    # model is a completely different distribution (runtime shares). They must
    # never be conflated.
    ui = {
        "phase1": 30.0, "phase2": 5.0, "phase3": 35.0, "phase4": 5.0,
        "phase4_5": 6.0, "phase5": 9.0, "phase6": 8.0, "phase7": 2.0,
    }
    cost = zeta.LEGACY_COST_MODEL
    # Distinct values at the dominant phases.
    assert cost["phase1"] != ui["phase1"]
    assert cost["phase3"] != ui["phase3"]
    assert cost["phase5"] != ui["phase5"]
    # Normalized UI shares must differ from the normalized cost shares.
    ui_norm = zeta.normalize_cost_model(ui)
    cost_norm = zeta.normalize_cost_model(cost)
    assert ui_norm != cost_norm


def test_sds_cost_model_differs_from_equal_ui_weights():
    # SDS UI weights are EQUAL (100/7 each); the cost model is explicit
    # conservative phase shares (global coadd dominant).
    ui = {name: 100.0 / 7.0 for name in zeta.SDS_STAGE_NAMES}
    cost = zeta.SDS_COST_MODEL
    assert cost["sds_phase_4"] != ui["sds_phase_4"]
    ui_norm = zeta.normalize_cost_model(ui)
    cost_norm = zeta.normalize_cost_model(cost)
    assert ui_norm != cost_norm


def test_normalize_cost_model_drops_nonpositive_and_nan():
    norm = zeta.normalize_cost_model({"a": 1.0, "b": 0.0, "c": -1.0, "d": float("nan")})
    assert set(norm) == {"a"}
    assert norm["a"] == pytest.approx(1.0)


def test_default_cost_model_is_mode_aware():
    assert zeta.default_cost_model(zprogress.LEGACY_PLAN) is zeta.LEGACY_COST_MODEL
    assert zeta.default_cost_model(zprogress.SDS_PLAN) is zeta.SDS_COST_MODEL
    assert zeta.default_cost_model(zprogress.ZEGRID_PLAN) is zeta.ZEGRID_COST_MODEL


# ---------------------------------------------------------------------------
# A3. Generic history APIs: mode isolation, bootstrap, v2 preference, scaling
# ---------------------------------------------------------------------------

def _tmp(tmp_path):
    return tmp_path / "h.json"


def _v1(duration_s=1000.0, n_frames=100):
    return {
        "ts_utc": "2026-01-01T00:00:00Z",
        "duration_s": duration_s,
        "n_frames": n_frames,
        "resume": False,
        "master_tiles": 0,
    }


def _legacy_v2(duration=1000.0, n_frames=100, master_tiles=40):
    return zeta.build_mode_history_record(
        mode="legacy",
        total_duration_s=duration,
        stage_seconds={
            "phase1": 50.0, "phase2": 5.0, "phase3": 200.0, "phase4": 50.0,
            "phase4_5": 100.0, "phase5": 500.0, "phase6": 60.0, "phase7": 20.0,
        },
        stage_totals={"phase1": n_frames, "phase3": master_tiles},
        n_frames=n_frames,
        master_tiles=master_tiles,
    )


def _sds_v2(duration=500.0):
    return zeta.build_mode_history_record(
        mode="sds",
        total_duration_s=duration,
        stage_seconds={"sds_phase_1": 50.0, "sds_phase_4": 200.0},
        stage_totals={"sds_phase_1": 10, "sds_phase_4": 20},
    )


def test_mode_isolation_legacy_sds_zegrid():
    # A legacy v2 record must NOT contribute to SDS or ZeGrid priors; an SDS
    # record must NOT contribute to legacy or ZeGrid.
    rec = _legacy_v2()
    assert zeta.select_mode_priors([rec], mode="sds") == ({}, 0)
    assert zeta.select_mode_priors([rec], mode="zegrid") == ({}, 0)
    srec = _sds_v2()
    assert zeta.select_mode_priors([srec], mode="legacy")[0] == {}
    assert zeta.select_mode_priors([srec], mode="zegrid") == ({}, 0)


def test_v1_bootstrap_legacy_only_never_sds_or_zegrid():
    v1 = _v1(duration_s=1000.0, n_frames=100)
    priors, comparable = zeta.select_mode_priors([v1], mode="legacy", n_frames=200)
    assert comparable == 0  # bootstrap, not per-stage calibration
    assert set(priors) == set(zeta.LEGACY_STAGE_NAMES)
    # Scaled total 1000 * (200/100) = 2000, distributed by LEGACY_COST_MODEL.
    assert priors["phase5"] == pytest.approx(2000.0 * 0.52)
    assert priors["phase1"] == pytest.approx(2000.0 * 0.04)
    # Never for SDS or ZeGrid.
    assert zeta.select_mode_priors([v1], mode="sds") == ({}, 0)
    assert zeta.select_mode_priors([v1], mode="zegrid") == ({}, 0)


def test_v2_preference_avoids_double_counting_v1():
    v1 = _v1(duration_s=99999.0, n_frames=999)
    v2 = _legacy_v2(duration=1000.0, n_frames=100, master_tiles=40)
    priors, comparable = zeta.select_mode_priors([v1, v2], mode="legacy", n_frames=100)
    # Once a v2 legacy record exists, ONLY v2 contributes (v1 never mixed).
    assert comparable == 1
    assert priors["phase1"] == pytest.approx(50.0)
    assert priors["phase5"] == pytest.approx(500.0)


def test_exact_legacy_stage_total_scaling():
    rec = _legacy_v2(n_frames=100, master_tiles=40)
    priors, comparable = zeta.select_mode_priors(
        [rec], mode="legacy", n_frames=200, master_tiles=80
    )
    assert comparable == 1
    # phase1 50 * (200/100) = 100; phase3 200 * (80/40) = 400.
    assert priors["phase1"] == pytest.approx(100.0)
    assert priors["phase3"] == pytest.approx(400.0)
    # Non-scalable stages verbatim.
    assert priors["phase5"] == pytest.approx(500.0)


def test_partial_record_tolerated():
    # A record missing some stage_seconds still contributes the stages it has;
    # other stages get no prior.
    rec = zeta.build_mode_history_record(
        mode="sds",
        total_duration_s=300.0,
        stage_seconds={"sds_phase_4": 200.0},
    )
    priors, comparable = zeta.select_mode_priors([rec], mode="sds")
    assert comparable == 1
    assert "sds_phase_4" in priors
    assert "sds_phase_1" not in priors


def test_robust_median_and_outlier_resistance():
    recs = [
        _legacy_v2(duration=1000.0),
        _legacy_v2(duration=1000.0),
        zeta.build_mode_history_record(
            mode="legacy",
            total_duration_s=1000.0,
            stage_seconds={
                "phase1": 50.0, "phase2": 5.0, "phase3": 200.0, "phase4": 50.0,
                "phase4_5": 100.0, "phase5": 100000.0, "phase6": 60.0, "phase7": 20.0,
            },
        ),
    ]
    priors, comparable = zeta.select_mode_priors(recs, mode="legacy")
    assert comparable == 3
    # The 100000s outlier is rejected by the robust median.
    assert priors["phase5"] == pytest.approx(500.0)


def test_wrappers_backward_compatible(tmp_path):
    p = _tmp(tmp_path)
    zrec = zeta.build_zegrid_history_record(
        total_duration_s=1000.0,
        n_frames=653,
        cell_count=170,
        stage_seconds={
            "setup": 15.0, "layout": 700.0, "gauge": 100.0,
            "per_cell_stack": 150.0, "assembly": 20.0, "finalize": 15.0,
        },
        stage_totals={"setup": 653, "gauge": 1305, "per_cell_stack": 170, "finalize": 1},
    )
    assert zeta.append_zegrid_history(zrec, path=p) is True
    loaded = zeta.load_eta_history(p)
    assert len(loaded) == 1
    assert loaded[0]["mode"] == "zegrid"
    # select_zegrid_priors still returns stable zegrid:<name> ids.
    priors, comparable = zeta.select_zegrid_priors(loaded, n_frames=653, cell_count=170)
    assert comparable == 1
    assert "zegrid:setup" in priors
    assert "zegrid:gauge" in priors


def test_generic_append_preserves_v1(tmp_path):
    p = _tmp(tmp_path)
    p.write_text(json.dumps({"schema": "zemosaic.eta_history.v1", "records": [_v1()]}))
    assert zeta.append_mode_history(_legacy_v2(), path=p) is True
    loaded = zeta.load_eta_history(p)
    assert len(loaded) == 2
    assert loaded[0]["duration_s"] == 1000.0  # v1 intact
    assert loaded[1]["mode"] == "legacy"


def test_history_record_is_sanitized():
    rec = _legacy_v2()
    text = json.dumps(rec)
    for forbidden in ("/home", "frame_", ".fits", ".png", "caldwell", "M106"):
        assert forbidden not in text


# ---------------------------------------------------------------------------
# A4. Success observation export
# ---------------------------------------------------------------------------

def test_export_observed_closes_active_and_returns_sanitized():
    ck = FakeClock(0.0)
    est = _legacy_est(ck)
    est.on_stage("phase1", 0, 653)
    ck.advance(10.0)
    est.on_stage("phase1", 653, 653)
    est.on_stage("phase2", 0, 0)
    ck.advance(5.0)
    est.on_stage("phase3", 0, 40)
    ck.advance(30.0)
    est.on_stage("phase3", 40, 40)
    est.on_stage("phase4", 0, 0)
    ck.advance(2.0)
    est.on_stage("phase5", 10, 100)
    ck.advance(20.0)
    observed = est.export_observed_stage_seconds()
    # Active phase5 is closed by the export; all seen phases have a positive
    # elapsed; no unseen phase (phase6/phase7) is present.
    assert observed == {
        "phase1": pytest.approx(10.0),
        "phase2": pytest.approx(5.0),
        "phase3": pytest.approx(30.0),
        "phase4": pytest.approx(2.0),
        "phase5": pytest.approx(20.0),
    }
    assert "phase6" not in observed
    assert "phase7" not in observed


def test_export_observed_empty_when_nothing_seen():
    est = _legacy_est(FakeClock(0.0))
    assert est.export_observed_stage_seconds() == {}


def test_export_observed_idempotent():
    ck = FakeClock(0.0)
    est = _legacy_est(ck)
    est.on_stage("phase1", 0, 10)
    ck.advance(3.0)
    est.on_stage("phase1", 10, 10)
    first = est.export_observed_stage_seconds()
    second = est.export_observed_stage_seconds()
    assert first == second


# ---------------------------------------------------------------------------
# B. Synthetic legacy sequence
# ---------------------------------------------------------------------------

def test_legacy_no_first_item_rate():
    ck = FakeClock(0.0)
    est = _legacy_est(ck)  # no priors
    est.on_stage("phase1", 0, 653)
    r = est.on_stage("phase1", 1, 653)
    assert r.ready is False
    assert r.remaining_seconds is None
    assert r.confidence == zeta.CONFIDENCE_CALIBRATING


def test_legacy_v1_bootstrap_prior_immediate():
    ck = FakeClock(0.0)
    est = _legacy_est(ck)
    # A sane v1 bootstrap prior (complete, distributed by LEGACY_COST_MODEL)
    # makes the active measurable phase honest IMMEDIATELY (history-based, never
    # first-item throughput).
    priors, comparable = zeta.select_mode_priors(
        [_v1(duration_s=1000.0, n_frames=100)], mode="legacy", n_frames=100
    )
    assert comparable == 0  # bootstrap, not per-stage calibration
    est.set_priors(priors, comparable_histories=comparable)
    r = est.on_stage("phase1", 0, 653)
    assert r.ready is True
    assert r.basis == "history"
    # active = prior_phase1 x remaining_fraction = (1000*0.04) x (653/653) = 40
    assert r.active_remaining == pytest.approx(40.0)
    # future floor = sum of phase2..phase7 bootstrap priors.
    assert r.future_floor == pytest.approx(1000.0 - 40.0)


def test_legacy_phases_without_totals_handled():
    ck = FakeClock(0.0)
    est = _legacy_est(ck)
    est.set_priors(
        {"phase2": 5.0, "phase3": 200.0, "phase4": 50.0, "phase4_5": 100.0,
         "phase5": 500.0, "phase6": 60.0, "phase7": 20.0},
        comparable_histories=1,
    )
    est.on_stage("phase1", 653, 653)
    # phase2 has NO STAGE_PROGRESS (only a PHASE_UPDATE transition) -> unknown
    # total -> estimated from its prior.
    r = est.on_stage("phase2", 0, 0)
    assert r.ready is True
    assert r.basis == "history"
    assert r.active_remaining == pytest.approx(5.0)


def test_legacy_transition_future_floor_and_terminal():
    ck = FakeClock(0.0)
    est = _legacy_est(ck)
    est.set_priors(
        {"phase2": 5.0, "phase3": 200.0, "phase4": 50.0, "phase4_5": 100.0,
         "phase5": 500.0, "phase6": 60.0, "phase7": 20.0},
        comparable_histories=2,
    )
    est.on_stage("phase1", 653, 653)
    r = est.on_stage("phase2", 0, 0)
    # Future floor = phase3..phase7 priors.
    assert r.future_floor == pytest.approx(200.0 + 50.0 + 100.0 + 500.0 + 60.0 + 20.0)
    # Transition proof closes phase2.
    ck.advance(3.0)
    est.on_stage("phase3", 0, 40)
    assert est._stages["phase2"].end_t is not None
    # Terminal success -> exactly 0.
    s = est.mark_success()
    assert s.remaining_seconds == 0.0
    assert s.terminal == "success"


def test_legacy_stale_holds_and_reset():
    ck = FakeClock(0.0)
    est = _legacy_est(ck)
    # Future-only priors (all phases after the active phase3) so the future
    # floor is computable; the active phase3 uses the pure live rate.
    est.set_priors(
        {"phase4": 50.0, "phase4_5": 100.0, "phase5": 500.0, "phase6": 60.0,
         "phase7": 20.0},
        comparable_histories=1,
    )
    est.on_stage("phase1", 653, 653)
    est.on_stage("phase2", 0, 0)
    est.on_stage("phase3", 0, 40)
    ck.advance(1.0)
    est.on_stage("phase3", 10, 40)
    ck.advance(1.0)
    est.on_stage("phase3", 20, 40)
    ck.advance(6.0)
    base = est.on_stage("phase3", 30, 40)
    assert base.ready is True
    assert base.basis == "live"
    assert base.future_floor == pytest.approx(50.0 + 100.0 + 500.0 + 60.0 + 20.0)
    ck.advance(60.0)
    t = est.tick()
    assert t.stalled is True
    assert t.remaining_seconds >= base.future_floor
    est.reset()
    d = est.diagnostics()
    assert d["terminal"] is None
    assert d["active_stage"] is None


# ---------------------------------------------------------------------------
# C. Synthetic SDS sequence
# ---------------------------------------------------------------------------

def test_sds_no_cross_mode_priors():
    ck = FakeClock(0.0)
    est = _sds_est(ck)
    # Legacy v1 total record MUST NOT yield SDS priors.
    priors, comparable = zeta.select_mode_priors([_v1()], mode="sds")
    assert priors == {}
    assert comparable == 0
    est.set_priors(priors, comparable_histories=comparable)
    r = est.on_stage("sds_phase_1", 0, 100)
    assert r.ready is False  # honest first-run calibration


def test_sds_first_run_calibrates_honestly():
    ck = FakeClock(0.0)
    est = _sds_est(ck)  # no priors (first run)
    est.on_stage("sds_phase_1", 0, 100)
    r = est.on_stage("sds_phase_1", 10, 100)
    assert r.ready is False  # no first-item ETA
    ck.advance(2.0)
    est.on_stage("sds_phase_1", 50, 100)
    ck.advance(2.0)
    est.on_stage("sds_phase_1", 90, 100)
    ck.advance(6.0)
    r = est.on_stage("sds_phase_1", 100, 100)  # live rate ready, but first phase
    # Honest: the FIRST phase has no completed predecessor and no prior, so the
    # future floor cannot be accounted yet -> still 'estimation en cours'.
    assert r.ready is False
    # Phase1 completes; later phases carry fallback evidence from its elapsed.
    est.on_stage("sds_phase_2", 0, 0)
    ck.advance(1.0)
    est.on_stage("sds_phase_3", 0, 0)
    est.on_stage("sds_phase_4", 0, 100)
    ck.advance(1.0)
    est.on_stage("sds_phase_4", 10, 100)
    ck.advance(1.0)
    est.on_stage("sds_phase_4", 20, 100)
    ck.advance(1.0)
    est.on_stage("sds_phase_4", 30, 100)
    ck.advance(6.0)
    r = est.on_stage("sds_phase_4", 40, 100)
    assert r.ready is True
    assert r.source == "fallback"  # future phases from completed-phase evidence


def test_sds_later_v2_priors_give_immediate_estimate():
    ck = FakeClock(0.0)
    est = _sds_est(ck)
    est.set_priors(
        {"sds_phase_1": 40.0, "sds_phase_2": 10.0, "sds_phase_3": 30.0,
         "sds_phase_4": 200.0, "sds_phase_5": 60.0, "sds_phase_6": 20.0,
         "sds_phase_7": 5.0},
        comparable_histories=2,
    )
    r = est.on_stage("sds_phase_1", 0, 100)
    assert r.ready is True
    assert r.basis == "history"
    assert r.active_remaining == pytest.approx(40.0)


def test_sds_phase4_throughput():
    ck = FakeClock(0.0)
    est = _sds_est(ck)
    # Future-only priors (phases 5/6/7 after the active phase4) so the active
    # coadd phase uses the pure live rate (no prior damping on the active stage).
    est.set_priors(
        {"sds_phase_5": 60.0, "sds_phase_6": 20.0, "sds_phase_7": 5.0},
        comparable_histories=1,
    )
    est.on_stage("sds_phase_1", 0, 100)
    est.on_stage("sds_phase_2", 0, 0)
    est.on_stage("sds_phase_3", 0, 0)
    est.on_stage("sds_phase_4", 0, 100)
    ck.advance(1.0)
    est.on_stage("sds_phase_4", 10, 100)
    ck.advance(1.0)
    est.on_stage("sds_phase_4", 20, 100)
    ck.advance(1.0)
    est.on_stage("sds_phase_4", 30, 100)
    ck.advance(6.0)
    r = est.on_stage("sds_phase_4", 40, 100)
    assert r.ready is True
    # active live remaining = (100-40)/10 = 6s; future = phase5+6+7 priors.
    assert r.active_remaining == pytest.approx(6.0, rel=1e-6)
    assert r.future_floor == pytest.approx(60.0 + 20.0 + 5.0)


def test_sds_terminal_and_reset():
    ck = FakeClock(0.0)
    est = _sds_est(ck)
    est.on_stage("sds_phase_1", 0, 100)
    s = est.mark_success()
    assert s.remaining_seconds == 0.0
    assert s.terminal == "success"
    est.reset()
    assert est.diagnostics()["terminal"] is None
    # Fail/cancel -> no completed ETA.
    est2 = _sds_est(FakeClock(0.0))
    assert est2.mark_fail().remaining_seconds is None
    assert est2.mark_cancel().remaining_seconds is None


# ---------------------------------------------------------------------------
# Phase-id mapping (pure)
# ---------------------------------------------------------------------------

def test_legacy_phase_id_to_stage_mapping():
    assert zprogress.legacy_phase_id_to_stage("1") == "phase1"
    assert zprogress.legacy_phase_id_to_stage("2") == "phase2"
    assert zprogress.legacy_phase_id_to_stage("3") == "phase3"
    assert zprogress.legacy_phase_id_to_stage("4") == "phase4"
    assert zprogress.legacy_phase_id_to_stage("4.5") == "phase4_5"
    assert zprogress.legacy_phase_id_to_stage("4_5") == "phase4_5"
    assert zprogress.legacy_phase_id_to_stage("5") == "phase5"
    assert zprogress.legacy_phase_id_to_stage("6") == "phase6"
    assert zprogress.legacy_phase_id_to_stage("7") == "phase7"
    assert zprogress.legacy_phase_id_to_stage("8") is None
    assert zprogress.legacy_phase_id_to_stage("") is None
    assert zprogress.legacy_phase_id_to_stage("x") is None


def test_sds_phase_id_to_stage_mapping():
    for n in range(1, 8):
        assert zprogress.sds_phase_id_to_stage(n) == f"sds_phase_{n}"
    assert zprogress.sds_phase_id_to_stage(0) is None
    assert zprogress.sds_phase_id_to_stage(8) is None
    assert zprogress.sds_phase_id_to_stage(None) is None
    assert zprogress.sds_phase_id_to_stage("x") is None


# ---------------------------------------------------------------------------
# Roman locale keys + formatter
# ---------------------------------------------------------------------------

def test_roman_locale_keys_present():
    for lang in ("en", "fr"):
        data = json.loads((SRC / "locales" / f"{lang}.json").read_text(encoding="utf-8"))
        assert data["phase_label_format"] == "{roman} — {operation}"
        for n in range(1, 8):
            assert data[f"qt_stage_phase{n}"]
        assert data["qt_stage_phase4_5"]
        for n in range(1, 8):
            assert data[f"qt_stage_sds_phase{n}"]


def test_roman_ordinals_cover_legacy_and_sds():
    assert zprogress.roman_ordinal(1) == "I"
    assert zprogress.roman_ordinal(5) == "V"
    assert zprogress.roman_ordinal(7) == "VII"
    assert zprogress.roman_ordinal(8) == "VIII"


# ---------------------------------------------------------------------------
# Static GUI seams (no Qt import)
# ---------------------------------------------------------------------------

def _method_src(name: str) -> str:
    text = (SRC / "zemosaic_gui_qt.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(text, node)
            assert seg is not None, f"no source segment for {name}"
            return seg
    raise AssertionError(f"method {name!r} not found")


def test_phase_update_feeds_correct_mode():
    src = _method_src("_on_worker_phase_changed")
    # SDS branch feeds the SDS authority; legacy branch feeds legacy authority.
    assert "_feed_sds_phase_transition" in src
    assert "_feed_legacy_phase_transition" in src


def test_legacy_stage_feeds_service_once_no_percent_eta():
    src = _method_src("_update_stage_progress")
    assert "_feed_legacy_eta" in src
    # The legacy structured branch must NOT use the percent-derived ETA.
    assert "_update_eta_from_progress" not in src


def test_sds_routes_feed_service():
    src = _method_src("_on_worker_stats_updated")
    assert "_feed_sds_phase_transition" in src
    assert "_feed_sds_stage" in src
    # SDS ETA is never derived from the equal global percent.
    log_src = _method_src("_on_worker_log_message")
    assert "_feed_sds_phase_complete" in log_src


def test_mode_switch_legacy_to_sds_resets():
    # _maybe_detect_sds triggers the reset on transition; _ensure_sds_eta also
    # discards any provisional legacy authority (never blend).
    src = _method_src("_maybe_detect_sds")
    assert "_on_sds_detected" in src
    ensure_src = _method_src("_ensure_sds_eta")
    assert "_discard_legacy_eta" in ensure_src


def test_old_eta_writers_guarded_telemetry_live():
    # The percent-derived writer is guarded for legacy/SDS hybrid active.
    src = _method_src("_update_eta_from_progress")
    assert "_legacy_eta_active" in src
    assert "_sds_eta_active" in src
    # GPU helper override is guarded too.
    gpu_src = _method_src("_start_gpu_eta_override")
    assert "_legacy_eta_active" in gpu_src
    # Textual ETA_UPDATE guarded.
    txt_src = _method_src("_on_worker_eta_updated")
    assert "_legacy_eta_active" in txt_src
    # STATS eta_seconds guarded.
    stats_src = _method_src("_on_worker_stats_updated")
    assert "_legacy_eta_active" in stats_src
    # Resource telemetry still collected (CPU/RAM/GPU parts) — same method.
    assert "cpu_percent" in stats_src


def test_success_only_persistence_legacy_sds_not_zegrid():
    src = _method_src("_on_worker_finished")
    # Wall duration captured BEFORE clearing the run start.
    assert "_persist_hybrid_history_success" in src
    # SDS + legacy success paths each call persistence once.
    assert '_persist_hybrid_history_success("sds"' in src
    assert '_persist_hybrid_history_success("legacy"' in src
    persist_src = _method_src("_persist_hybrid_history_success")
    # Exactly once per run (write-once flag); never for ZeGrid; no record on
    # cancel/fail (guarded by mark_cancel/mark_fail in the finished handler).
    assert "_legacy_eta_written" in persist_src
    assert "_sds_eta_written" in persist_src
    assert "export_observed_stage_seconds" in persist_src
    assert "append_mode_history" in persist_src
    assert "mode not in" in persist_src


def test_wall_duration_captured_before_clear():
    src = _method_src("_on_worker_finished")
    # The run start is captured into wall_duration before being set to None.
    idx_clear = src.index("self._run_started_monotonic = None")
    idx_capture = src.index("wall_duration")
    assert idx_capture < idx_clear


def test_reset_clears_all_mode_eta_state():
    src = _method_src("_reset_progress_tracking")
    for token in (
        "_legacy_eta = None",
        "_legacy_eta_records = None",
        "_legacy_eta_written = False",
        "_sds_eta = None",
        "_sds_eta_records = None",
        "_sds_eta_written = False",
        "_sds_phase_floor = 0",
        "_legacy_phase_floor = 0",
    ):
        assert token in src


def test_roman_phase_label_monotonic_floor():
    src = _method_src("_set_roman_phase_label")
    assert "roman_ordinal" in src
    assert "PHASE_LABEL_FORMAT" in src
    assert "never move the label backwards" in src
