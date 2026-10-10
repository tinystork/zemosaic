"""ZM-ETA-SERVICE-R30 — pure, history-calibrated HYBRID ETA service (lot 4a).

Stdlib-only (no Qt, no numpy, no astropy) so it can be imported by the worker,
the GUI and the tests without pulling in the decode/reproject stack. It is the
ZeGrid ETA authority: the first estimate never extrapolates from the first
frame/sample, and UI progress weights (R29) are NEVER used as ETA costs.

Two cooperating pieces live here:

* :class:`HybridEtaEstimator` — a deterministic estimator with an injectable
  monotonic clock. It consumes STABLE R29 stage events ``(stage_id, current,
  total)`` and produces an immutable :class:`EtaResult`. The estimate is a
  HYBRID: the ACTIVE stage's remaining work is derived from a robust recent
  wall-clock throughput (median + MAD filter, never the first-item ratio) and/or
  a historical phase prior; FUTURE stages are accounted from historical priors
  (or, in the absence of any history, from a conservative fallback that only
  allocates ALREADY-OBSERVED completed-phase evidence). COMPLETED stages are
  never predicted again.

* History store helpers — a backward-compatible v2 record shape on the existing
  ``~/.zemosaic_eta_history.json`` path. Legacy v1 ``{duration_s, n_frames, ...}``
  records still load safely and are preserved (never mixed into ZeGrid
  per-stage priors), atomic append (unique same-dir temp + ``os.replace`` with
  guaranteed best-effort cleanup), bounded count, and fail-open on malformed/
  corrupt/unwritable files.

Design rules (see the R30 mission + rework-1):

* Stable ids, never counters or translated strings or progress percentages as
  model keys (normalized through the R29 progress contract).
* Startup/calibration window: no live-rate ETA before ``MIN_RATE_SAMPLES``
  meaningful increasing samples AND ``MIN_ACTIVE_ELAPSED_S`` seconds elapsed in
  the active measurable stage. DURING that window a comparable/sane history
  prior for the active stage is used IMMEDIATELY (history-based remaining =
  ``prior x remaining-unit-fraction``), never first-item throughput. Once the
  robust live rate is ready it is BLENDED with the active prior
  (``ACTIVE_HISTORY_BLEND_ALPHA``); with no history the live gate still applies.
* The estimate is split into an ACTIVE component and a FUTURE floor
  (not-yet-started stages). ``tick`` decrements ONLY the active component and
  never consumes the future floor. A live stage goes stale after
  ``STALE_WINDOW_S`` (HOLD at >= future floor, ``stalled``, downgraded); an
  unknown-total history stage, when its prior is exhausted without a transition,
  HOLDs the future floor and flags ``stalled`` (re-evaluation).
* Nonterminal estimates never display exactly 0: a documented
  ``MIN_DISPLAY_FLOOR_S`` (1 s) applies. Only terminal success returns exactly 0.
* Smoothing is asymmetric: legitimate UPWARD corrections apply immediately;
  downward corrections are damped so the display does not oscillate. The ETA is
  NEVER forced monotonically downward. Phase transitions reset component/floor
  state cleanly.
* Terminal success -> exactly ``0``; fail/cancel -> no completed ETA.
* Always finite, nonnegative, no divide-by-zero.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants (documented + testable; injectable via constructor kwargs)
# ---------------------------------------------------------------------------

# Minimum number of meaningful INCREASING samples before a live-rate ETA is
# produced for the active measurable stage. A "meaningful" sample strictly
# increases the stage's completed count (duplicates/out-of-order are ignored).
MIN_RATE_SAMPLES = 3

# Minimum elapsed seconds in the active measurable stage before a live-rate ETA
# is produced (the startup/calibration window).
MIN_ACTIVE_ELAPSED_S = 8.0

# Recent-window size (samples) for the robust active-throughput estimate.
RATE_WINDOW_SAMPLES = 8

# MAD outlier-rejection constant for the robust rate (median +/- K * MAD).
MAD_OUTLIER_K = 3.0

# Bounded freshness window (seconds): ``tick`` may count down between fresh
# live-rate samples only within this window; beyond it the ACTIVE component is
# HELD (stalled) at >= the future floor (never marched falsely to zero).
STALE_WINDOW_S = 30.0

# Asymmetric smoothing: downward corrections move only this fraction toward the
# new (lower) value per recompute, so the display does not oscillate. Upward
# corrections are applied immediately (see :meth:`HybridEtaEstimator._smooth`).
SMOOTH_DOWN_ALPHA = 0.5

# Nonterminal estimates never display exactly 0: minimum display floor (seconds).
# Only terminal success returns exactly 0.
MIN_DISPLAY_FLOOR_S = 1.0

# Blend weight given to the ACTIVE stage's history prior once the robust live
# rate is ready::
#   active = ALPHA * history_remaining + (1 - ALPHA) * live_remaining
# where ``history_remaining = prior x remaining_unit_fraction`` (bounded) and
# ``live_remaining = (total - done) / robust_rate``. History is a stabilising
# term; the live rate still dominates the blend (never first-item throughput).
ACTIVE_HISTORY_BLEND_ALPHA = 0.35

# Relative WORK-COST shares for the six ZeGrid stages, used ONLY as a
# conservative FALLBACK to allocate ALREADY-OBSERVED completed-phase evidence
# when no history exists. These are NOT calibrated durations and are NOT the
# R29 UI progress weights (which are a separate, unrelated quantity).
ZEGRID_COST_MODEL: Dict[str, float] = {
    "zegrid:setup": 1.0,
    "zegrid:layout": 8.0,
    "zegrid:gauge": 16.0,
    "zegrid:per_cell_stack": 50.0,
    "zegrid:assembly": 4.0,
    "zegrid:finalize": 1.0,
}

# Ordered ZeGrid stage names (bare, no ``zegrid:`` prefix) for the compact
# history-record ``stage_seconds`` / ``stage_totals`` dictionaries.
ZEGRID_STAGE_NAMES = (
    "setup", "layout", "gauge", "per_cell_stack", "assembly", "finalize",
)

# ZM-ETA-ALLMODES-R31: per-mode ETA fallback cost models. These are WORK-COST
# SHARES used ONLY as a conservative fallback to allocate ALREADY-OBSERVED
# completed-phase evidence when no history exists, and (legacy only) to
# distribute a v1-total bootstrap prior. They are NOT calibrated durations and
# are NOT the R29 UI progress weights (a separate, unrelated quantity).
#
# LEGACY_COST_MODEL derives from the worker's historical conservative RUNTIME
# shares (documented; no accuracy claim), NOT from the R29 UI weights
# (30/5/35/5/6/9/8/2). The shares are normalized safely at use.
LEGACY_COST_MODEL: Dict[str, float] = {
    "phase1": 0.04,     # scan / preprocess (fast per frame; long tail)
    "phase2": 0.01,     # clustering
    "phase3": 0.20,     # master tiles
    "phase4": 0.05,     # geometry / final grid
    "phase4_5": 0.10,   # inter-master harmonization
    "phase5": 0.52,     # final assembly / reproject+coadd (dominant)
    "phase6": 0.06,     # output saving
    "phase7": 0.02,     # cleanup / finalization
}

# SDS explicit conservative phase shares (documented rationale; no accuracy
# claim). Separate from the SDS UI equal weights (100/7 each). Global coadd
# dominates; preprocess and polish are material; cluster/save/cleanup are small.
SDS_COST_MODEL: Dict[str, float] = {
    "sds_phase_1": 0.15,   # preprocess
    "sds_phase_2": 0.05,   # cluster
    "sds_phase_3": 0.20,   # master tiles / batches
    "sds_phase_4": 0.35,   # global coadd
    "sds_phase_5": 0.15,   # polish
    "sds_phase_6": 0.07,   # save
    "sds_phase_7": 0.03,   # cleanup
}

# Ordered canonical stage ids for legacy / SDS (== the bare names used as the
# compact history-record ``stage_seconds`` / ``stage_totals`` keys).
LEGACY_STAGE_NAMES = (
    "phase1", "phase2", "phase3", "phase4", "phase4_5",
    "phase5", "phase6", "phase7",
)
SDS_STAGE_NAMES = (
    "sds_phase_1", "sds_phase_2", "sds_phase_3", "sds_phase_4",
    "sds_phase_5", "sds_phase_6", "sds_phase_7",
)


def stage_names_for(mode: str) -> Tuple[str, ...]:
    """Return the ordered stage names for a mode (``legacy`` | ``sds`` | ``zegrid``)."""
    if mode == "legacy":
        return LEGACY_STAGE_NAMES
    if mode == "sds":
        return SDS_STAGE_NAMES
    return ZEGRID_STAGE_NAMES


def normalize_cost_model(model: Mapping[str, float]) -> Dict[str, float]:
    """Normalize positive cost shares to sum to 1.0 (drop non-positive/NaN).

    Scale-invariant, so normalizing never changes the estimator math; it only
    makes the shares comparable and documents them as a normalized distribution.
    """
    cleaned: Dict[str, float] = {}
    for k, v in (model or {}).items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f > 0.0:
            cleaned[str(k)] = f
    total = sum(cleaned.values())
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in cleaned.items()}


def default_cost_model(plan) -> Dict[str, float]:
    """Return the per-mode fallback cost model for a plan (legacy/sds/zegrid)."""
    mode = getattr(plan, "mode", None)
    if mode == "legacy":
        return LEGACY_COST_MODEL
    if mode == "sds":
        return SDS_COST_MODEL
    return ZEGRID_COST_MODEL

# Stages whose prior may be scaled by their EXACT per-stage total when both the
# current run's and the record's ``stage_totals`` are valid. ``setup``/``gauge``
# fall back to ``n_frames`` and ``per_cell_stack`` to ``cell_count`` for older
# v2 records lacking exact totals.
SCALABLE_STAGES = ("setup", "gauge", "per_cell_stack")

CONFIDENCE_CALIBRATING = "calibrating"
CONFIDENCE_LOW = "low"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_HIGH = "high"


# ---------------------------------------------------------------------------
# Immutable snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EtaResult:
    """Immutable snapshot of one ETA estimate.

    Attributes:
        remaining_seconds:    Remaining wall-clock seconds, or ``None`` when no
                              honest estimate exists (``ready=False``). For a
                              nonterminal ready estimate this is >= the future
                              floor (and >= ``MIN_DISPLAY_FLOOR_S``), never 0.
        ready:                ``True`` when the estimate is usable/displayable.
        confidence:           ``calibrating`` | ``low`` | ``medium`` | ``high``.
        active_stage:         Stable id of the active stage (``None`` if none).
        basis:                ACTIVE-stage mechanism: ``history`` | ``live`` |
                              ``blend`` | ``calibrating`` | ``terminal``.
        source:               FUTURE-stage mechanism: ``history`` | ``fallback``
                              | ``blend`` | ``none`` | ``calibrating`` |
                              ``terminal`` (diagnostic).
        active_remaining:     The ACTIVE component of the estimate (diagnostic;
                              post-smoothing / post-tick-decrement).
        future_floor:         The FUTURE (not-yet-started) phase floor
                              (diagnostic; never consumed by ``tick``).
        stalled:              ``True`` when progress went stale or the active
                              prior was exhausted and the estimate is HELD.
        sample_count:         Live-rate samples for the active stage (diagnostic).
        comparable_histories: Number of comparable ZeGrid history records used
                              for the priors (diagnostic).
        terminal:             ``success`` | ``fail`` | ``cancel`` | ``None``.
    """

    remaining_seconds: Optional[float]
    ready: bool
    confidence: str
    active_stage: Optional[str]
    basis: str
    stalled: bool = False
    source: str = "none"
    sample_count: int = 0
    comparable_histories: int = 0
    terminal: Optional[str] = None
    active_remaining: Optional[float] = None
    future_floor: float = 0.0


# ---------------------------------------------------------------------------
# Robust rate (pure function, unit-testable)
# ---------------------------------------------------------------------------

def robust_rate(
    samples: Sequence[Tuple[float, float]],
    *,
    window: int = RATE_WINDOW_SAMPLES,
    mad_k: float = MAD_OUTLIER_K,
) -> Optional[float]:
    """Median (MAD-filtered) throughput from ``(t, done)`` samples.

    ``samples`` are strictly increasing in ``done`` (the estimator guarantees
    this). The rate is computed over the RECENT window of per-interval deltas
    ``(done1 - done0) / (t1 - t0)`` — NEVER the first-item ratio — and is
    outlier-resistant (median, then drop samples beyond ``mad_k * 1.4826 * MAD``
    and re-median). Returns ``None`` when there are fewer than two usable
    intervals or no positive progress.
    """
    seq = list(samples)[-int(window):]
    if len(seq) < 2:
        return None
    deltas: List[float] = []
    for (t0, d0), (t1, d1) in zip(seq, seq[1:]):
        dt = t1 - t0
        dd = d1 - d0
        if dt > 0.0 and dd > 0.0:
            deltas.append(dd / dt)
    if not deltas:
        return None
    med = statistics.median(deltas)
    if len(deltas) >= 3:
        mad = statistics.median([abs(d - med) for d in deltas])
        if mad > 0.0:
            kept = [d for d in deltas if abs(d - med) <= mad_k * 1.4826 * mad]
            if kept:
                med = statistics.median(kept)
    if not math.isfinite(med) or med <= 0.0:
        return None
    return med


def _median_outlier_resistant(values: Sequence[float]) -> float:
    vals = [v for v in values if math.isfinite(v) and v > 0.0]
    if not vals:
        return 0.0
    med = statistics.median(vals)
    if len(vals) >= 3:
        kept = [v for v in vals if 0.1 * med <= v <= 10.0 * med]
        if kept:
            med = statistics.median(kept)
    return float(med)


# ---------------------------------------------------------------------------
# History store
# ---------------------------------------------------------------------------

ETA_HISTORY_SCHEMA_V2 = "zemosaic.eta_history.v2"
ETA_HISTORY_MAX_RECORDS = 120


def eta_history_path() -> Path:
    """Return the shared history path (compatible with the legacy v1 writer)."""
    try:
        return Path.home() / ".zemosaic_eta_history.json"
    except Exception:
        return Path(".zemosaic_eta_history.json")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_eta_history(path: Optional[Path] = None) -> List[dict]:
    """Load history records (v1 AND v2) — fail-open, never raises.

    Returns the raw list of record dicts regardless of the payload schema, so
    legacy v1 records (``duration_s`` / ``n_frames`` / ...) and new v2 records
    (``mode`` / ``stage_seconds`` / ...) both load safely. Malformed/corrupt/
    missing files return ``[]``.
    """
    try:
        p = Path(path) if path is not None else eta_history_path()
        if not p.exists():
            return []
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return []
        records = payload.get("records")
        if not isinstance(records, list):
            return []
        return [rec for rec in records if isinstance(rec, dict)]
    except Exception:
        return []


def select_zegrid_priors(
    records: Sequence[dict],
    *,
    n_frames: Optional[int] = None,
    cell_count: Optional[int] = None,
    stage_totals: Optional[Mapping[str, int]] = None,
) -> Tuple[Dict[str, float], int]:
    """Select comparable ZeGrid per-stage priors (median, outlier-resistant).

    Only ``mode == "zegrid"`` v2 records with a positive finite total duration
    and a ``stage_seconds`` dict contribute. Legacy v1 total records are NEVER
    mixed into ZeGrid per-stage priors.

    Scaling precedence (per scalable stage ``setup`` / ``gauge`` /
    ``per_cell_stack``):

    1. EXACT per-stage total: ``record.stage_totals[stage]`` -> current
       ``stage_totals[stage]`` when both are valid (>0).
    2. Fallback (older v2 records missing exact totals): ``setup``/``gauge`` by
       ``n_frames``, ``per_cell_stack`` by ``cell_count``.
    3. Otherwise the record's duration is used verbatim.

    Gauge is NEVER scaled by raw/top-level ``n_frames`` when an exact gauge
    total exists. Returns ``(priors, comparable)`` where ``priors`` maps stable
    ``zegrid:<name>`` ids to median seconds and ``comparable`` is the number of
    contributing records.
    """
    per_stage: Dict[str, List[float]] = {name: [] for name in ZEGRID_STAGE_NAMES}
    comparable = 0
    try:
        cur_n = int(n_frames) if n_frames else 0
    except (TypeError, ValueError):
        cur_n = 0
    try:
        cur_c = int(cell_count) if cell_count else 0
    except (TypeError, ValueError):
        cur_c = 0
    cur_totals = stage_totals if isinstance(stage_totals, Mapping) else {}

    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("mode") != "zegrid":
            continue
        stages = rec.get("stage_seconds")
        if not isinstance(stages, dict) or not stages:
            continue
        try:
            total = float(rec.get("total_duration_s"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(total) or total <= 0.0:
            continue
        rec_n = _safe_int(rec.get("n_frames"))
        rec_c = _safe_int(rec.get("cell_count"))
        rec_totals = rec.get("stage_totals")
        rec_totals = rec_totals if isinstance(rec_totals, Mapping) else {}
        contributed = False
        for name in ZEGRID_STAGE_NAMES:
            raw = stages.get(name)
            if raw is None:
                continue
            try:
                sec = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(sec) or sec <= 0.0:
                continue
            scaled = sec
            if name in SCALABLE_STAGES:
                rec_exact = _safe_int(rec_totals.get(name))
                cur_exact = _safe_int(cur_totals.get(name))
                if cur_exact > 0 and rec_exact > 0:
                    # EXACT per-stage total scaling (preferred).
                    scaled = sec * (cur_exact / float(rec_exact))
                elif name in ("setup", "gauge") and cur_n > 0 and rec_n > 0:
                    scaled = sec * (cur_n / float(rec_n))
                elif name == "per_cell_stack" and cur_c > 0 and rec_c > 0:
                    scaled = sec * (cur_c / float(rec_c))
            if math.isfinite(scaled) and scaled > 0.0:
                per_stage[name].append(scaled)
                contributed = True
        if contributed:
            comparable += 1

    priors: Dict[str, float] = {}
    for name in ZEGRID_STAGE_NAMES:
        vals = per_stage[name]
        if vals:
            priors[f"zegrid:{name}"] = _median_outlier_resistant(vals)
    return priors, comparable


def select_mode_priors(
    records: Sequence[dict],
    *,
    mode: str,
    n_frames: Optional[int] = None,
    cell_count: Optional[int] = None,
    master_tiles: Optional[int] = None,
    stage_totals: Optional[Mapping[str, int]] = None,
) -> Tuple[Dict[str, float], int]:
    """Select comparable per-stage priors for ONE mode (legacy|sds|zegrid).

    A generic front door on top of the per-mode selectors. Records are selected
    ONLY within the matching mode (never blended across modes), and legacy v1
    total records are NEVER used for SDS or ZeGrid. See the per-mode selectors
    for scaling/bootstrapping details. Returns ``(priors, comparable)``.
    """
    if mode == "legacy":
        return select_legacy_priors(
            records,
            n_frames=n_frames,
            master_tiles=master_tiles,
            stage_totals=stage_totals,
        )
    if mode == "sds":
        return select_sds_priors(records, stage_totals=stage_totals)
    return select_zegrid_priors(
        records,
        n_frames=n_frames,
        cell_count=cell_count,
        stage_totals=stage_totals,
    )


def _select_v2_mode_priors(
    records: Sequence[dict],
    *,
    mode: str,
    names: Tuple[str, ...],
    scale_by: Mapping[str, str],
    n_frames: Optional[int] = None,
    master_tiles: Optional[int] = None,
    stage_totals: Optional[Mapping[str, int]] = None,
) -> Tuple[Dict[str, float], int]:
    """Shared v2 per-stage prior selection (median, outlier-resistant).

    Only ``mode``-matching v2 records with a positive finite total duration and
    a ``stage_seconds`` dict contribute. Exact per-stage-total scaling is applied
    when both the record's and the current run's ``stage_totals[stage]`` are
    valid (>0), else the optional context fallback ``scale_by`` key is applied.
    Legacy v1 total records (``duration_s`` without ``mode``) NEVER contribute
    here. Returns ``(priors, comparable)``.
    """
    per_stage: Dict[str, List[float]] = {name: [] for name in names}
    comparable = 0
    try:
        cur_n = int(n_frames) if n_frames else 0
    except (TypeError, ValueError):
        cur_n = 0
    try:
        cur_m = int(master_tiles) if master_tiles else 0
    except (TypeError, ValueError):
        cur_m = 0
    cur_totals = stage_totals if isinstance(stage_totals, Mapping) else {}

    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("mode") != mode:
            continue
        stages = rec.get("stage_seconds")
        if not isinstance(stages, dict) or not stages:
            continue
        try:
            total = float(rec.get("total_duration_s"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(total) or total <= 0.0:
            continue
        rec_n = _safe_int(rec.get("n_frames"))
        rec_m = _safe_int(rec.get("master_tiles"))
        rec_totals = rec.get("stage_totals")
        rec_totals = rec_totals if isinstance(rec_totals, Mapping) else {}
        contributed = False
        for name in names:
            raw = stages.get(name)
            if raw is None:
                continue
            try:
                sec = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(sec) or sec <= 0.0:
                continue
            scaled = sec
            ctx_key = scale_by.get(name)
            if ctx_key is not None:
                rec_exact = _safe_int(rec_totals.get(name))
                cur_exact = _safe_int(cur_totals.get(name))
                if cur_exact > 0 and rec_exact > 0:
                    # EXACT per-stage total scaling (preferred).
                    scaled = sec * (cur_exact / float(rec_exact))
                elif ctx_key == "n_frames" and cur_n > 0 and rec_n > 0:
                    scaled = sec * (cur_n / float(rec_n))
                elif ctx_key == "master_tiles" and cur_m > 0 and rec_m > 0:
                    scaled = sec * (cur_m / float(rec_m))
            if math.isfinite(scaled) and scaled > 0.0:
                per_stage[name].append(scaled)
                contributed = True
        if contributed:
            comparable += 1

    priors: Dict[str, float] = {}
    for name in names:
        vals = per_stage[name]
        if vals:
            priors[name] = _median_outlier_resistant(vals)
    return priors, comparable


def select_legacy_priors(
    records: Sequence[dict],
    *,
    n_frames: Optional[int] = None,
    master_tiles: Optional[int] = None,
    stage_totals: Optional[Mapping[str, int]] = None,
) -> Tuple[Dict[str, float], int]:
    """Select legacy per-stage priors (v2 preferred, v1 bootstrap fallback).

    When ANY v2 legacy records (``mode == "legacy"`` with ``stage_seconds``)
    exist, only those calibrated priors are used (v1 never double-counted).
    ``phase1`` scales by ``n_frames`` and ``phase3`` by ``master_tiles`` with
    exact per-stage totals preferred.

    When NO v2 legacy records exist, a LOW/MEDIUM-confidence bootstrap is
    derived from the v1 total records (median total duration, scaled by
    ``n_frames`` when known, then distributed by :data:`LEGACY_COST_MODEL`).
    This bootstrap is a total-duration prior, NOT per-stage calibration, and is
    never used for SDS or ZeGrid.
    """
    priors, comparable = _select_v2_mode_priors(
        records,
        mode="legacy",
        names=LEGACY_STAGE_NAMES,
        scale_by={"phase1": "n_frames", "phase3": "master_tiles"},
        n_frames=n_frames,
        master_tiles=master_tiles,
        stage_totals=stage_totals,
    )
    if comparable > 0:
        return priors, comparable
    return _legacy_v1_bootstrap(records, n_frames=n_frames)


def _legacy_v1_bootstrap(
    records: Sequence[dict],
    *,
    n_frames: Optional[int] = None,
) -> Tuple[Dict[str, float], int]:
    """LOW/MEDIUM-confidence legacy bootstrap from v1 total records.

    Uses the robust median of v1 ``duration_s`` totals, scaled by ``n_frames``
    when both a current ``n_frames`` and a robust median of the records'
    ``n_frames`` are known, then distributes the result by
    :data:`LEGACY_COST_MODEL`. This is a total-duration prior, not per-stage
    calibration. ``comparable`` is returned 0 so the estimator confidence stays
    at most MEDIUM for a bootstrap. Never used for SDS/ZeGrid.
    """
    durations: List[float] = []
    frame_counts: List[float] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("mode"):
            continue  # skip any v2 (typed) record — v1 records have no mode
        try:
            dur = float(rec.get("duration_s"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(dur) or dur <= 0.0:
            continue
        durations.append(dur)
        nf = _safe_int(rec.get("n_frames"))
        if nf > 0:
            frame_counts.append(float(nf))
    if not durations:
        return {}, 0

    median_total = _median_outlier_resistant(durations)
    if median_total <= 0.0:
        return {}, 0
    try:
        cur_n = int(n_frames) if n_frames else 0
    except (TypeError, ValueError):
        cur_n = 0
    if cur_n > 0 and frame_counts:
        median_frames = _median_outlier_resistant(frame_counts)
        if median_frames > 0.0:
            median_total = median_total * (cur_n / median_frames)

    model = normalize_cost_model(LEGACY_COST_MODEL)
    priors: Dict[str, float] = {}
    for name in LEGACY_STAGE_NAMES:
        share = model.get(name, 0.0)
        if share > 0.0:
            priors[name] = median_total * share
    return priors, 0


def select_sds_priors(
    records: Sequence[dict],
    *,
    stage_totals: Optional[Mapping[str, int]] = None,
) -> Tuple[Dict[str, float], int]:
    """Select SDS per-stage priors (v2 only; never v1, never cross-mode).

    Only ``mode == "sds"`` v2 records contribute. Exact per-stage-total scaling
    applies when both record and current totals are valid. Legacy v1 records are
    NEVER used for SDS. First successful Qt SDS runs have no v2 records yet, so
    they return ``({}, 0)`` and the GUI shows ``estimation en cours`` until live
    evidence accrues, then writes a v2 SDS record for future runs.
    """
    return _select_v2_mode_priors(
        records,
        mode="sds",
        names=SDS_STAGE_NAMES,
        scale_by={},
        stage_totals=stage_totals,
    )


def append_mode_history(
    record: dict,
    path: Optional[Path] = None,
) -> bool:
    """Atomically append one v2 mode record, bounded, fail-open (generic).

    Existing records (v1 and v2) are preserved. The record count is capped to
    :data:`ETA_HISTORY_MAX_RECORDS` (oldest dropped). A UNIQUE same-directory
    temp file is created via :func:`tempfile.mkstemp`; on ANY write/replace
    failure the temp file is removed (best-effort) and the existing target is
    left untouched. Returns ``False`` and NEVER raises on any error.
    """
    p = Path(path) if path is not None else eta_history_path()
    tmp: Optional[Path] = None
    try:
        records = load_eta_history(p)
        records.append(dict(record))
        if len(records) > ETA_HISTORY_MAX_RECORDS:
            records = records[-ETA_HISTORY_MAX_RECORDS:]
        payload = {
            "schema": ETA_HISTORY_SCHEMA_V2,
            "updated_utc": _utcnow_iso(),
            "records": records,
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=p.name + ".", suffix=".tmp", dir=str(p.parent)
        )
        tmp = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2, ensure_ascii=False))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        return True
    except Exception:
        return False
    finally:
        if tmp is not None:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass


def append_zegrid_history(
    record: dict,
    path: Optional[Path] = None,
) -> bool:
    """Backward-compatible ZeGrid history append (alias of the generic append)."""
    return append_mode_history(record, path=path)


def build_mode_history_record(
    *,
    mode: str,
    total_duration_s: float,
    stage_seconds: Mapping[str, float],
    stage_totals: Optional[Mapping[str, int]] = None,
    n_frames: Optional[int] = None,
    cell_count: Optional[int] = None,
    master_tiles: Optional[int] = None,
    workers: Optional[int] = None,
    backend: Optional[str] = None,
) -> dict:
    """Build a sanitized, compact v2 mode history record (generic).

    Only numeric totals/durations, per-stage seconds/totals, and (optionally)
    workload context + effective backend/worker concurrency are recorded — never
    raw paths, frame names, or user data.
    """
    names = stage_names_for(mode)
    rec: dict = {
        "schema_v": 2,
        "mode": mode,
        "ts_utc": _utcnow_iso(),
        "total_duration_s": round(float(total_duration_s), 3),
        "stage_seconds": {
            name: round(float(stage_seconds.get(name, 0.0)), 3)
            for name in names
        },
    }
    if n_frames is not None:
        try:
            rec["n_frames"] = int(n_frames)
        except (TypeError, ValueError):
            pass
    if cell_count is not None:
        try:
            rec["cell_count"] = int(cell_count)
        except (TypeError, ValueError):
            pass
    if master_tiles is not None:
        try:
            rec["master_tiles"] = int(master_tiles)
        except (TypeError, ValueError):
            pass
    if stage_totals:
        rec["stage_totals"] = {
            name: int(stage_totals[name])
            for name in names
            if stage_totals.get(name) is not None
        }
    if workers is not None:
        try:
            rec["workers"] = int(workers)
        except (TypeError, ValueError):
            pass
    if backend:
        rec["backend"] = str(backend)
    return rec


def build_zegrid_history_record(
    *,
    total_duration_s: float,
    n_frames: int,
    cell_count: int,
    stage_seconds: Mapping[str, float],
    stage_totals: Optional[Mapping[str, int]] = None,
    workers: Optional[int] = None,
    backend: Optional[str] = None,
) -> dict:
    """Build a sanitized, compact v2 ZeGrid history record.

    Only numeric totals/durations and (optionally) the effective backend /
    worker concurrency are recorded — never raw paths, frame names, or user
    data.
    """
    rec: dict = {
        "schema_v": 2,
        "mode": "zegrid",
        "ts_utc": _utcnow_iso(),
        "total_duration_s": round(float(total_duration_s), 3),
        "n_frames": int(n_frames),
        "cell_count": int(cell_count),
        "stage_seconds": {
            name: round(float(stage_seconds.get(name, 0.0)), 3)
            for name in ZEGRID_STAGE_NAMES
        },
    }
    if stage_totals:
        rec["stage_totals"] = {
            name: int(stage_totals[name])
            for name in ZEGRID_STAGE_NAMES
            if stage_totals.get(name) is not None
        }
    if workers is not None:
        try:
            rec["workers"] = int(workers)
        except (TypeError, ValueError):
            pass
    if backend:
        rec["backend"] = str(backend)
    return rec


def _safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Estimator internal stage state
# ---------------------------------------------------------------------------

@dataclass
class _StageState:
    samples: List[Tuple[float, float]] = field(default_factory=list)
    last_current: int = 0
    total: int = 0
    start_t: Optional[float] = None
    end_t: Optional[float] = None
    elapsed_s: Optional[float] = None


# ---------------------------------------------------------------------------
# Hybrid ETA estimator
# ---------------------------------------------------------------------------

class HybridEtaEstimator:
    """Deterministic hybrid ETA estimator for ONE mode plan (legacy|sds|zegrid).

    The estimator is PLAN-AGNOSTIC: it consumes STABLE stage events from any
    :class:`ModePlan` (``LEGACY_PLAN`` / ``SDS_PLAN`` / ``ZEGRID_PLAN``) and
    applies an INJECTED per-mode ETA fallback cost model (distinct from the R29
    UI progress weights). Feed stable stage events with :meth:`on_stage`, let
    wall time advance with :meth:`tick`, finish with :meth:`mark_success` /
    :meth:`mark_fail` / :meth:`mark_cancel`, and export sanitized observed
    stage timings with :meth:`export_observed_stage_seconds`. Historical
    per-stage priors are injected via :meth:`set_priors` (the caller uses
    :func:`select_mode_priors` to build them). The injectable ``clock`` (default
    ``time.monotonic``) makes the estimator fully deterministic in tests.
    """

    def __init__(
        self,
        plan,
        *,
        cost_model: Optional[Mapping[str, float]] = None,
        clock: Optional[Callable[[], float]] = None,
        min_rate_samples: int = MIN_RATE_SAMPLES,
        min_active_elapsed_s: float = MIN_ACTIVE_ELAPSED_S,
        rate_window_samples: int = RATE_WINDOW_SAMPLES,
        mad_k: float = MAD_OUTLIER_K,
        stale_window_s: float = STALE_WINDOW_S,
        smooth_down_alpha: float = SMOOTH_DOWN_ALPHA,
        blend_history_alpha: float = ACTIVE_HISTORY_BLEND_ALPHA,
        display_floor_s: float = MIN_DISPLAY_FLOOR_S,
    ) -> None:
        self._plan = plan
        self._cost_model: Dict[str, float] = normalize_cost_model(
            cost_model if cost_model is not None else default_cost_model(plan)
        )
        self._clock = clock if clock is not None else time.monotonic
        self._min_rate_samples = int(min_rate_samples)
        self._min_active_elapsed_s = float(min_active_elapsed_s)
        self._rate_window_samples = int(rate_window_samples)
        self._mad_k = float(mad_k)
        self._stale_window_s = float(stale_window_s)
        self._smooth_down_alpha = float(smooth_down_alpha)
        self._blend_history_alpha = float(blend_history_alpha)
        self._display_floor_s = float(display_floor_s)
        self._priors: Dict[str, float] = {}
        self._comparable = 0
        self.reset()

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Clear all run state (stages, terminal, smoothing, priors kept)."""
        self._stages: Dict[str, _StageState] = {}
        self._active_stage: Optional[str] = None
        self._active_position: int = -1
        self._terminal: Optional[str] = None
        self._last_live_sample_t: Optional[float] = None
        self._last_active_basis: Optional[str] = None
        self._last_future_source: Optional[str] = None
        self._active_smoothed: Optional[float] = None
        self._future_floor: float = 0.0
        self._last_raw_active: Optional[float] = None
        self._last_estimate: Optional[EtaResult] = None
        self._last_estimate_t: Optional[float] = None
        self._n_frames: Optional[int] = None
        self._cell_count: Optional[int] = None
        self._master_tiles: Optional[int] = None
        self._workers: Optional[int] = None
        self._backend: Optional[str] = None

    # -- configuration ------------------------------------------------------

    def set_priors(self, priors: Mapping[str, float], *, comparable_histories: int = 0) -> None:
        """Inject historical per-stage priors (stable ``zegrid:<name>`` -> seconds)."""
        cleaned: Dict[str, float] = {}
        for k, v in (priors or {}).items():
            try:
                sec = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(sec) and sec > 0.0:
                cleaned[str(k)] = sec
        self._priors = cleaned
        self._comparable = int(comparable_histories)

    def set_context(
        self,
        *,
        n_frames: Optional[int] = None,
        cell_count: Optional[int] = None,
        master_tiles: Optional[int] = None,
        workers: Optional[int] = None,
        backend: Optional[str] = None,
    ) -> None:
        """Record optional workload context (diagnostic; not required for math)."""
        if n_frames is not None:
            self._n_frames = int(n_frames)
        if cell_count is not None:
            self._cell_count = int(cell_count)
        if master_tiles is not None:
            self._master_tiles = int(master_tiles)
        if workers is not None:
            self._workers = int(workers)
        if backend is not None:
            self._backend = str(backend)

    # -- events -------------------------------------------------------------

    def on_stage(self, stage_id: str, current: int, total: int) -> EtaResult:
        """Feed one structured stage event; returns the current estimate.

        Duplicate/out-of-order events never regress or double-count. A later
        stage closes the previous stage's elapsed duration (a phase transition
        proof) and performs a bounded regime reset.
        """
        now = self._clock()
        if self._terminal is not None:
            return self._terminal_result()
        sid = self._plan.normalize_id(stage_id)
        desc = self._plan.stage(sid)
        if desc is None:
            return self._estimate(now)
        try:
            cur = int(current)
        except (TypeError, ValueError):
            cur = 0
        try:
            tot = int(total)
        except (TypeError, ValueError):
            tot = 0
        if cur < 0:
            cur = 0
        if tot < 0:
            tot = 0

        position = int(desc.position)

        if self._active_stage is not None and position < self._active_position:
            # Out-of-order (earlier stage after a later one): ignore, no regress.
            return self._estimate(now)

        if self._active_stage is None or position > self._active_position:
            # Phase transition: close every prior open stage at ``now``.
            self._close_prior_stages(position, now)
            self._active_stage = sid
            self._active_position = position
            # Bounded regime reset (active component/floor/smoothing restart).
            self._active_smoothed = None
            self._future_floor = 0.0
            self._last_raw_active = None
            self._last_estimate = None
            self._last_estimate_t = None
            self._last_active_basis = None
            self._last_future_source = None
            self._last_live_sample_t = None

        st = self._stages.setdefault(sid, _StageState())
        if st.start_t is None:
            st.start_t = now
        if tot > 0:
            st.total = max(st.total, tot)
        # Only a meaningful INCREASE records a sample; duplicates never regress.
        if cur > st.last_current:
            st.last_current = cur
            st.samples.append((now, cur))
            if tot > 0:
                self._last_live_sample_t = now

        return self._estimate(now)

    def tick(self) -> EtaResult:
        """Advance the ACTIVE-component countdown between fresh samples.

        The FUTURE floor is NEVER consumed. Live/blend stages count down within
        ``STALE_WINDOW_S`` of the last fresh live sample, then HOLD at >= the
        future floor (``stalled``, downgraded). History (unknown-total) stages
        count their prior down; when the prior is exhausted without a
        transition, the future floor is HELD and the result is flagged
        ``stalled`` (re-evaluation). Nonterminal remaining never reaches 0.
        """
        now = self._clock()
        if self._terminal is not None:
            return self._terminal_result()
        est = self._last_estimate
        if est is None or not est.ready or self._active_smoothed is None:
            return self._estimate(now)

        active = self._active_smoothed
        future_floor = self._future_floor
        basis = self._last_active_basis
        stalled = False
        confidence = est.confidence

        if basis in ("live", "blend"):
            ref_t = (
                self._last_live_sample_t
                if self._last_live_sample_t is not None
                else self._last_estimate_t
            )
            since = max(0.0, now - (ref_t if ref_t is not None else now))
            if since > self._stale_window_s:
                # HOLD: countdown capped at the freshness window.
                active = max(0.0, active - self._stale_window_s)
                stalled = True
                confidence = CONFIDENCE_LOW
            else:
                active = max(0.0, active - since)
        elif basis == "history":
            # Count the active prior down by elapsed wall time.
            age = max(0.0, now - (self._last_estimate_t if self._last_estimate_t is not None else now))
            active = max(0.0, active - age)
            if active <= 0.0:
                # Prior exhausted without a transition: HOLD the future floor.
                active = 0.0
                stalled = True
        else:
            age = max(0.0, now - (self._last_estimate_t if self._last_estimate_t is not None else now))
            active = max(0.0, active - age)

        display = self._display_remaining(active, future_floor)
        return replace(
            est,
            remaining_seconds=display,
            active_remaining=active,
            future_floor=future_floor,
            stalled=stalled,
            confidence=confidence,
        )

    def mark_success(self) -> EtaResult:
        if self._terminal is None:
            self._terminal = "success"
        return self._terminal_result()

    def mark_fail(self) -> EtaResult:
        if self._terminal is None:
            self._terminal = "fail"
        return self._terminal_result()

    def mark_cancel(self) -> EtaResult:
        if self._terminal is None:
            self._terminal = "cancel"
        return self._terminal_result()

    # -- success-only observation export ------------------------------------

    def export_observed_stage_seconds(self) -> Dict[str, float]:
        """Close active timing and return sanitized observed stage seconds.

        Call at terminal success to export the per-stage wall durations the
        estimator ACTUALLY observed (stages with a positive closed elapsed
        duration), keyed by stable canonical stage id. This is the pure,
        sanitized observation source for the GUI's success-only v2 history
        record — no GUI private-state scraping. Stages never seen (no event, or
        zero/unknown elapsed) are omitted. Safe to call repeatedly (idempotent).
        """
        now = self._clock()
        # Close the ACTIVE stage too (position + 1 closes all <= active).
        if self._active_stage is not None and self._active_position >= 0:
            self._close_prior_stages(self._active_position + 1, now)
        observed: Dict[str, float] = {}
        for d in self._plan.stages:
            st = self._stages.get(d.id)
            if st is None:
                continue
            sec = st.elapsed_s
            if sec is not None and math.isfinite(sec) and sec > 0.0:
                observed[d.id] = float(sec)
        return observed

    # -- estimate internals -------------------------------------------------

    def _terminal_result(self) -> EtaResult:
        if self._terminal == "success":
            return EtaResult(
                remaining_seconds=0.0,
                ready=True,
                confidence=CONFIDENCE_HIGH,
                active_stage=self._active_stage,
                basis="terminal",
                source="terminal",
                sample_count=self._active_sample_count(),
                comparable_histories=self._comparable,
                terminal="success",
                active_remaining=0.0,
                future_floor=0.0,
            )
        return EtaResult(
            remaining_seconds=None,
            ready=False,
            confidence=CONFIDENCE_CALIBRATING,
            active_stage=self._active_stage,
            basis="terminal",
            source="terminal",
            sample_count=self._active_sample_count(),
            comparable_histories=self._comparable,
            terminal=self._terminal,
        )

    def _not_ready(self, now: float) -> EtaResult:
        result = EtaResult(
            remaining_seconds=None,
            ready=False,
            confidence=CONFIDENCE_CALIBRATING,
            active_stage=self._active_stage,
            basis="calibrating",
            source="none",
            sample_count=self._active_sample_count(),
            comparable_histories=self._comparable,
        )
        self._last_estimate = result
        self._last_estimate_t = now
        self._active_smoothed = None
        self._future_floor = 0.0
        self._last_raw_active = None
        self._last_active_basis = None
        self._last_future_source = None
        return result

    def _estimate(self, now: float) -> EtaResult:
        if self._active_stage is None:
            return self._not_ready(now)
        desc = self._plan.stage(self._active_stage)
        st = self._stages.get(self._active_stage)
        if desc is None or st is None:
            return self._not_ready(now)

        prior = self._priors.get(self._active_stage)
        has_prior = prior is not None and prior > 0.0 and math.isfinite(prior)

        active_remaining: Optional[float] = None
        active_basis: Optional[str] = None

        total_known = st.total > 0
        live_rate: Optional[float] = None
        if total_known and self._live_ready(st, now):
            live_rate = robust_rate(
                st.samples, window=self._rate_window_samples, mad_k=self._mad_k
            )

        if total_known:
            if live_rate is not None:
                live_remaining = max(0.0, (st.total - st.last_current) / live_rate)
                if has_prior:
                    history_remaining = max(0.0, prior * self._remaining_fraction(st))
                    active_remaining = (
                        self._blend_history_alpha * history_remaining
                        + (1.0 - self._blend_history_alpha) * live_remaining
                    )
                    active_basis = "blend"
                else:
                    active_remaining = live_remaining
                    active_basis = "live"
            elif has_prior:
                # Calibration window: history-based remaining (remaining-unit
                # fraction), NEVER first-item throughput.
                active_remaining = max(0.0, prior * self._remaining_fraction(st))
                active_basis = "history"
            else:
                return self._not_ready(now)
        else:
            # Unknown-total stage (layout/assembly): historical prior only.
            if has_prior:
                elapsed = (now - st.start_t) if st.start_t is not None else 0.0
                active_remaining = max(0.0, prior - elapsed)
                active_basis = "history"
            else:
                return self._not_ready(now)

        future_floor, future_source = self._compute_future_floor(desc.position, now)
        if future_floor is None:
            return self._not_ready(now)

        if not math.isfinite(active_remaining) or active_remaining < 0.0:
            active_remaining = 0.0
        active_remaining = float(active_remaining)

        active_smoothed = self._smooth_active(active_remaining)
        self._active_smoothed = active_smoothed
        self._future_floor = future_floor
        self._last_active_basis = active_basis
        self._last_future_source = future_source

        display = self._display_remaining(active_smoothed, future_floor)
        confidence = self._confidence(active_basis, future_source)

        result = EtaResult(
            remaining_seconds=display,
            ready=True,
            confidence=confidence,
            active_stage=self._active_stage,
            basis=active_basis,
            source=future_source,
            sample_count=len(st.samples),
            comparable_histories=self._comparable,
            active_remaining=active_smoothed,
            future_floor=future_floor,
        )
        self._last_estimate = result
        self._last_estimate_t = now
        return result

    def _remaining_fraction(self, st: _StageState) -> float:
        total = max(1, st.total)
        return max(0.0, (total - st.last_current) / float(total))

    def _compute_future_floor(
        self, active_position: int, now: float
    ) -> Tuple[Optional[float], Optional[str]]:
        """Future not-yet-started stage cost (priors, else conservative fallback).

        Returns ``(floor, source)`` where ``source`` is ``history`` | ``fallback``
        | ``blend`` | ``none``, or ``(None, None)`` when the floor cannot yet be
        accounted honestly (no prior for a future stage and no observed evidence
        for a fallback).
        """
        floor = 0.0
        uses_history = False
        uses_fallback = False
        for d in self._plan.stages:
            if d.position <= active_position:
                continue
            prior = self._priors.get(d.id)
            if prior is not None and prior > 0.0 and math.isfinite(prior):
                floor += prior
                uses_history = True
            else:
                unit = self._fallback_unit()
                if unit is None:
                    return None, None
                cost = self._cost_model.get(d.id, 0.0)
                if cost <= 0.0:
                    return None, None
                floor += unit * cost
                uses_fallback = True
        if uses_history and uses_fallback:
            source = "blend"
        elif uses_history:
            source = "history"
        elif uses_fallback:
            source = "fallback"
        else:
            source = "none"
        return float(floor), source

    def _display_remaining(self, active: float, future_floor: float) -> float:
        total = max(0.0, active + future_floor)
        return max(self._display_floor_s, total)

    def _live_ready(self, st: _StageState, now: float) -> bool:
        if len(st.samples) < self._min_rate_samples:
            return False
        if st.start_t is None:
            return False
        return (now - st.start_t) >= self._min_active_elapsed_s

    def _smooth_active(self, raw_active: float) -> float:
        if self._active_smoothed is None:
            self._last_raw_active = raw_active
            return raw_active
        if raw_active == self._last_raw_active:
            # No NEW raw evidence (duplicate/out-of-order event): keep the
            # current smoothed value — a duplicate must never drift the estimate.
            return self._active_smoothed
        self._last_raw_active = raw_active
        prev = self._active_smoothed
        if raw_active >= prev:
            # Legitimate upward correction: apply immediately (never hide it).
            return raw_active
        return prev - self._smooth_down_alpha * (prev - raw_active)

    def _confidence(self, active_basis: Optional[str], future_source: Optional[str]) -> str:
        uses_history = active_basis in ("history", "blend") or future_source in (
            "history",
            "blend",
        )
        if not uses_history:
            return CONFIDENCE_LOW
        return CONFIDENCE_HIGH if self._comparable >= 3 else CONFIDENCE_MEDIUM

    def _fallback_unit(self) -> Optional[float]:
        observed_cost = 0.0
        observed_time = 0.0
        for d in self._plan.stages:
            st = self._stages.get(d.id)
            if st is None or st.elapsed_s is None or st.elapsed_s <= 0.0:
                continue
            cost = self._cost_model.get(d.id, 0.0)
            if cost > 0.0:
                observed_cost += cost
                observed_time += st.elapsed_s
        if observed_cost <= 0.0 or observed_time <= 0.0:
            return None
        return observed_time / observed_cost

    def _close_prior_stages(self, position: int, now: float) -> None:
        for d in self._plan.stages:
            if d.position >= position:
                continue
            st = self._stages.get(d.id)
            if st is None or st.end_t is not None:
                continue
            st.end_t = now
            start = st.start_t if st.start_t is not None else now
            st.elapsed_s = max(0.0, now - start)

    def _active_sample_count(self) -> int:
        st = self._stages.get(self._active_stage or "")
        return len(st.samples) if st is not None else 0

    # -- diagnostics --------------------------------------------------------

    def diagnostics(self) -> dict:
        """Human/log/debug state (no per-event production logging)."""
        return {
            "active_stage": self._active_stage,
            "terminal": self._terminal,
            "priors": dict(self._priors),
            "comparable_histories": self._comparable,
            "n_frames": self._n_frames,
            "cell_count": self._cell_count,
            "master_tiles": self._master_tiles,
            "workers": self._workers,
            "backend": self._backend,
            "active_basis": self._last_active_basis,
            "future_source": self._last_future_source,
            "future_floor": self._future_floor,
            "last_estimate": (
                None
                if self._last_estimate is None
                else {
                    "remaining_seconds": self._last_estimate.remaining_seconds,
                    "ready": self._last_estimate.ready,
                    "confidence": self._last_estimate.confidence,
                    "basis": self._last_estimate.basis,
                    "stalled": self._last_estimate.stalled,
                }
            ),
        }
