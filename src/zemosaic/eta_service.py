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
  wall-clock throughput (median + MAD filter, never the first-item ratio) when
  the stage total is measurable, or from a historical phase prior when the
  total is unknown; FUTURE stages are accounted from historical priors (or, in
  the absence of any history, from a conservative fallback that only allocates
  ALREADY-OBSERVED completed-phase evidence). COMPLETED stages are never
  predicted again.

* History store helpers — a backward-compatible v2 record shape on the existing
  ``~/.zemosaic_eta_history.json`` path. Legacy v1 ``{duration_s, n_frames, ...}``
  records still load safely and are preserved (never mixed into ZeGrid
  per-stage priors), atomic append (temp + ``os.replace``), bounded count, and
  fail-open on malformed/corrupt/unwritable files.

Design rules (see the R30 mission):

* Stable ids, never counters or translated strings or progress percentages as
  model keys (normalized through the R29 progress contract).
* Startup/calibration window: no live-rate ETA before ``MIN_RATE_SAMPLES``
  meaningful increasing samples AND ``MIN_ACTIVE_ELAPSED_S`` seconds elapsed in
  the active measurable stage (constants documented + testable).
* Smoothing is asymmetric: legitimate UPWARD corrections apply immediately;
  downward corrections are damped so the display does not oscillate. The ETA is
  NEVER forced monotonically downward. Phase transitions do a bounded regime
  reset.
* ``tick`` counts down between fresh samples only within ``STALE_WINDOW_S``;
  when progress goes stale the estimate is HELD (never marched falsely to
  zero) and flagged ``stalled``.
* Terminal success -> exactly ``0``; fail/cancel -> no completed ETA.
* Always finite, nonnegative, no divide-by-zero.
"""

from __future__ import annotations

import json
import math
import os
import statistics
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
# samples only within this window; beyond it the estimate is HELD (stalled).
STALE_WINDOW_S = 30.0

# Asymmetric smoothing: downward corrections move only this fraction toward the
# new (lower) value per recompute, so the display does not oscillate. Upward
# corrections are applied immediately (see :meth:`HybridEtaEstimator._smooth`).
SMOOTH_DOWN_ALPHA = 0.5

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
                              honest estimate exists (``ready=False``).
        ready:                ``True`` when the estimate is usable/displayable.
        confidence:           ``calibrating`` | ``low`` | ``medium`` | ``high``.
        active_stage:         Stable id of the active stage (``None`` if none).
        basis:                ``history`` | ``live`` | ``blend`` | ``fallback`` |
                              ``calibrating`` | ``terminal`` — the dominant
                              mechanism behind the estimate.
        stalled:              ``True`` when progress went stale and the estimate
                              is HELD (not counting down).
        source:               Same vocabulary as ``basis`` (diagnostic alias).
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
) -> Tuple[Dict[str, float], int]:
    """Select comparable ZeGrid per-stage priors (median, outlier-resistant).

    Only ``mode == "zegrid"`` v2 records with a positive finite total duration
    and a ``stage_seconds`` dict contribute. Legacy v1 total records are NEVER
    mixed into ZeGrid per-stage priors. Per-stage durations are scaled by units
    when meaningful (setup/gauge by ``n_frames``, per_cell_stack by
    ``cell_count``) and otherwise used verbatim. Returns ``(priors, comparable)``
    where ``priors`` maps stable ``zegrid:<name>`` ids to median seconds and
    ``comparable`` is the number of contributing records.
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
            if name in ("setup", "gauge") and cur_n and rec_n:
                scaled = sec * (cur_n / max(1, rec_n))
            elif name == "per_cell_stack" and cur_c and rec_c:
                scaled = sec * (cur_c / max(1, rec_c))
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


def append_zegrid_history(
    record: dict,
    path: Optional[Path] = None,
) -> bool:
    """Atomically append one v2 record (temp + ``os.replace``), bounded, fail-open.

    Existing records (v1 and v2) are preserved. The record count is capped to
    :data:`ETA_HISTORY_MAX_RECORDS` (oldest dropped). Any I/O/JSON error returns
    ``False`` and NEVER raises (a history write cannot change science/UI
    success). The new record is sanitized by the caller (no raw paths/frame
    names/user data).
    """
    try:
        p = Path(path) if path is not None else eta_history_path()
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
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp, p)
        return True
    except Exception:
        return False


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
    """Deterministic hybrid ETA estimator for one mode plan (ZeGrid).

    Feed stable stage events with :meth:`on_stage`, let wall time advance with
    :meth:`tick`, and finish with :meth:`mark_success` / :meth:`mark_fail` /
    :meth:`mark_cancel`. Historical per-stage priors are injected via
    :meth:`set_priors` (the caller uses :func:`select_zegrid_priors` to build
    them). The injectable ``clock`` (default ``time.monotonic``) makes the
    estimator fully deterministic in tests.
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
    ) -> None:
        self._plan = plan
        self._cost_model: Dict[str, float] = dict(
            cost_model if cost_model is not None else ZEGRID_COST_MODEL
        )
        self._clock = clock if clock is not None else time.monotonic
        self._min_rate_samples = int(min_rate_samples)
        self._min_active_elapsed_s = float(min_active_elapsed_s)
        self._rate_window_samples = int(rate_window_samples)
        self._mad_k = float(mad_k)
        self._stale_window_s = float(stale_window_s)
        self._smooth_down_alpha = float(smooth_down_alpha)
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
        self._smoothed_remaining: Optional[float] = None
        self._last_raw: Optional[float] = None
        self._last_estimate: Optional[EtaResult] = None
        self._last_estimate_t: Optional[float] = None
        self._n_frames: Optional[int] = None
        self._cell_count: Optional[int] = None
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
        workers: Optional[int] = None,
        backend: Optional[str] = None,
    ) -> None:
        """Record optional workload context (diagnostic; not required for math)."""
        if n_frames is not None:
            self._n_frames = int(n_frames)
        if cell_count is not None:
            self._cell_count = int(cell_count)
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
            # Bounded regime reset (smoothing/countdown restart for this phase).
            self._smoothed_remaining = None
            self._last_estimate = None
            self._last_estimate_t = None
            self._last_active_basis = None

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
        """Advance the countdown between fresh samples (call ~1/s from the GUI).

        Within :data:`STALE_WINDOW_S` of the last fresh live-rate sample the
        remaining time counts down smoothly. When the active stage is live-rate
        and progress has gone stale, the estimate is HELD (never marched falsely
        to zero) and flagged ``stalled``. History-prior (unknown-total) stages
        count down by wall time without a staleness gate (elapsed IS the
        progress signal there).
        """
        now = self._clock()
        if self._terminal is not None:
            return self._terminal_result()
        if self._last_estimate is None or not self._last_estimate.ready:
            return self._estimate(now)
        if (
            self._last_active_basis == "live"
            and self._last_live_sample_t is not None
            and (now - self._last_live_sample_t) > self._stale_window_s
        ):
            held = max(0.0, (self._smoothed_remaining or 0.0) - self._stale_window_s)
            return replace(
                self._last_estimate,
                remaining_seconds=held,
                stalled=True,
                confidence=CONFIDENCE_LOW,
            )
        dt = now - (self._last_estimate_t or now)
        remaining = max(0.0, (self._smoothed_remaining or 0.0) - dt)
        return replace(self._last_estimate, remaining_seconds=remaining, stalled=False)

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
        self._smoothed_remaining = None
        return result

    def _estimate(self, now: float) -> EtaResult:
        if self._active_stage is None:
            return self._not_ready(now)
        desc = self._plan.stage(self._active_stage)
        st = self._stages.get(self._active_stage)
        if desc is None or st is None:
            return self._not_ready(now)

        active_remaining: Optional[float] = None
        active_basis: Optional[str] = None

        total_known = st.total > 0
        if total_known:
            # Measurable stage: use ONLY the robust live rate (gated by the
            # calibration window). A history prior is NEVER substituted for the
            # active measurable stage — the first estimate must not extrapolate
            # from the first sample/frame.
            if self._live_ready(st, now):
                rate = robust_rate(st.samples, window=self._rate_window_samples, mad_k=self._mad_k)
                if rate is not None:
                    active_remaining = max(0.0, (st.total - st.last_current) / rate)
                    active_basis = "live"
        else:
            # Unknown-total stage (layout/assembly): use the historical phase
            # prior when available; otherwise stay calibrating (no fabricate).
            prior = self._priors.get(self._active_stage)
            if prior is not None and prior > 0.0:
                elapsed = now - (st.start_t or now)
                active_remaining = max(0.0, prior - elapsed)
                active_basis = "history"
        if active_remaining is None:
            return self._not_ready(now)

        # Future stages: historical priors, else conservative fallback that only
        # allocates already-observed completed-phase evidence.
        future_remaining = 0.0
        future_uses_history = False
        future_uses_fallback = False
        for d in self._plan.stages:
            if d.position <= desc.position:
                continue
            prior = self._priors.get(d.id)
            if prior is not None and prior > 0.0:
                future_remaining += prior
                future_uses_history = True
            else:
                unit = self._fallback_unit()
                if unit is None:
                    return self._not_ready(now)
                cost = self._cost_model.get(d.id, 0.0)
                if cost <= 0.0:
                    return self._not_ready(now)
                future_remaining += unit * cost
                future_uses_fallback = True

        raw = active_remaining + future_remaining
        if not math.isfinite(raw) or raw < 0.0:
            return self._not_ready(now)
        raw = float(raw)

        source = self._source(active_basis, future_uses_history, future_uses_fallback)
        confidence = self._confidence(source)

        smoothed = self._smooth(raw)
        self._smoothed_remaining = smoothed
        self._last_active_basis = active_basis

        result = EtaResult(
            remaining_seconds=smoothed,
            ready=True,
            confidence=confidence,
            active_stage=self._active_stage,
            basis=source,
            source=source,
            sample_count=len(st.samples),
            comparable_histories=self._comparable,
        )
        self._last_estimate = result
        self._last_estimate_t = now
        return result

    def _live_ready(self, st: _StageState, now: float) -> bool:
        if len(st.samples) < self._min_rate_samples:
            return False
        if st.start_t is None:
            return False
        return (now - st.start_t) >= self._min_active_elapsed_s

    def _smooth(self, raw: float) -> float:
        if self._smoothed_remaining is None:
            self._last_raw = raw
            return raw
        if raw == self._last_raw:
            # No NEW raw evidence (duplicate/out-of-order event): keep the
            # current smoothed value — a duplicate must never drift the estimate.
            return self._smoothed_remaining
        self._last_raw = raw
        prev = self._smoothed_remaining
        if raw >= prev:
            # Legitimate upward correction: apply immediately (never hide it).
            return raw
        return prev - self._smooth_down_alpha * (prev - raw)

    def _source(
        self,
        active_basis: Optional[str],
        future_history: bool,
        future_fallback: bool,
    ) -> str:
        if active_basis == "live" and future_history:
            return "blend"
        if active_basis == "live":
            return "live"
        if active_basis == "history" and future_history:
            return "history"
        if active_basis == "history" and future_fallback:
            return "blend"
        return "fallback"

    def _confidence(self, source: str) -> str:
        if source in ("history", "blend"):
            return CONFIDENCE_HIGH if self._comparable >= 3 else CONFIDENCE_MEDIUM
        return CONFIDENCE_LOW

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
            "workers": self._workers,
            "backend": self._backend,
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
