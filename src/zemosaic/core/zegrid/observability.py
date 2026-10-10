"""ZM-ZEGRID-R14 — live observability helpers (phase logging, progress, ETA).

Pure observation, no science, stdlib-only and import-safe so it can be reused by
the production orchestrator and by tests without pulling in the heavy decode/
reproject stack.

Provides:

* :class:`PhaseEta`      — per-phase ETA from observed throughput samples
                          (explicit ``n/a yet`` until enough samples exist).
* :class:`GlobalEta`     — global ETA once phase weights (completed-phase
                          durations) are known.
* :class:`PhaseReporter` — START/END lines + throttled intra-phase progress with
                          percent, current item id, per-phase ETA and global ETA,
                          plus a breadcrumb-stage callback.
* :class:`SubstepReporter` — nested sub-step progress + timing (throttled
                          counters, no GUI stage/percent), used by the layout
                          phase's three stable sub-steps.
* :func:`format_eta`     — human-readable ETA (``n/a yet`` for ``None``).

Cadence contract: intra-phase progress is emitted at most every
``interval_s`` seconds (plus always the first and the final sample), so the log
is chatty enough to diagnose a slow phase without spamming.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default bounded cadence for intra-phase progress (seconds).
DEFAULT_PROGRESS_INTERVAL_S = 2.0
# Minimum number of (elapsed, done) samples before a per-phase ETA is produced.
DEFAULT_ETA_MIN_SAMPLES = 2
# Recent-window size (samples) for the per-phase ETA rate estimate (I1).
DEFAULT_ETA_WINDOW_SAMPLES = 6

# Stable layout SUB-STEP ids (ZM-ZEGRID-R27). Separated from the human-readable
# display text and from counters/items — the ids never embed counters (unlike the
# legacy ``zegrid:<phase>:<done>/<total>`` stage string). They double as the
# subordinate TIMING names: the ``layout:`` prefix marks each as a child of the
# top-level ``layout`` phase (never a new top-level phase for the global ETA).
LAYOUT_SUBSTEP_FOOTPRINTS = "layout:footprints"
LAYOUT_SUBSTEP_CANDIDATES = "layout:candidates"
LAYOUT_SUBSTEP_CHOSEN_CELLS = "layout:chosen_cells"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_eta(seconds: Optional[float]) -> str:
    """Human-readable ETA; ``"n/a yet"`` when not yet computable (``None``)."""
    if seconds is None:
        return "n/a yet"
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "n/a yet"
    if not (s == s):  # NaN guard (must precede the max() clamp)
        return "n/a yet"
    s = max(0.0, s)
    if s < 1.0:
        return "<1s"
    total = int(round(s))
    minutes, secs = divmod(total, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Per-phase ETA
# ---------------------------------------------------------------------------

class PhaseEta:
    """Estimate the remaining time of a phase from observed throughput samples.

    A sample is ``(elapsed_seconds, done_items)``. The rate is computed over a
    RECENT WINDOW of the last ``window`` samples (``(done_last - done_first) /
    (elapsed_last - elapsed_first)``) rather than the whole-phase cumulative
    average, so a mid-phase slowdown is reflected promptly instead of being
    masked by the earlier fast portion. The window keeps the estimate stable
    (no single-sample jitter). Returns ``None`` (``n/a yet``) until ``min_samples``
    samples exist and the window shows real progress.
    """

    def __init__(
        self,
        min_samples: int = DEFAULT_ETA_MIN_SAMPLES,
        window: int = DEFAULT_ETA_WINDOW_SAMPLES,
    ) -> None:
        self.min_samples = int(min_samples)
        self.window = max(2, int(window))
        self.samples: list[tuple[float, float]] = []

    def observe(self, elapsed_s: float, done: float) -> None:
        self.samples.append((float(elapsed_s), float(done)))

    def reset(self) -> None:
        self.samples = []

    def eta_seconds(self, total: Optional[int]) -> Optional[float]:
        """Remaining seconds, or ``None`` when not yet computable."""
        if total is None or total <= 0:
            return None
        if len(self.samples) < self.min_samples:
            return None
        window = self.samples[-self.window:]
        t0, d0 = window[0]
        t1, d1 = window[-1]
        dt = t1 - t0
        dd = d1 - d0
        if dt <= 0.0 or dd <= 0.0:
            return None
        rate = dd / dt
        if rate <= 0.0:
            return None
        remaining = max(0.0, float(total) - d1)
        if remaining <= 0.0:
            return 0.0
        return remaining / rate


# ---------------------------------------------------------------------------
# Global ETA
# ---------------------------------------------------------------------------

class GlobalEta:
    """Global ETA once phase weights (completed-phase durations) are known.

    The remaining time is estimated as the current phase's remaining ETA plus the
    expected duration of every not-yet-started phase. A not-yet-started phase uses
    its learned duration when available (it has completed at least once this run),
    otherwise the mean of the completed phases' durations as the default weight.
    Returns ``None`` (``n/a yet``) until at least one phase has completed (no
    phase weights yet).
    """

    def __init__(self, phase_order: list[str] | tuple[str, ...]) -> None:
        self.phase_order = list(phase_order)
        self.run_t0 = time.perf_counter()
        self.durations: dict[str, float] = {}

    def phase_completed(self, name: str, elapsed_s: float) -> None:
        self.durations[name] = float(elapsed_s)

    def estimate(self, current_remaining_s: Optional[float] = None) -> Optional[float]:
        """Global remaining seconds, or ``None`` when phase weights are unknown."""
        if not self.durations:
            return None
        remaining = 0.0
        if current_remaining_s is not None and current_remaining_s > 0.0:
            remaining += float(current_remaining_s)
        default = sum(self.durations.values()) / len(self.durations)
        seen = set(self.durations)
        for phase in self.phase_order:
            if phase in seen:
                continue
            remaining += self.durations.get(phase, default)
        return remaining


# ---------------------------------------------------------------------------
# Phase reporter
# ---------------------------------------------------------------------------

class PhaseReporter:
    """Emit START/END lines and throttled intra-phase progress with ETA + stage.

    ``emit``     — ``callable(message: str, lvl: str = "INFO")`` routed to the
                   run's logger + ``progress_callback`` (GUI log/breadcrumbs).
    ``log_line`` — ``callable(line: str)`` that appends + flushes one line to the
                   incremental run log (so it is readable DURING the run).
    ``stage``    — ``callable(stage: str, current: int, total: int)`` that updates
                   the worker crash-breadcrumb ``stage``/``tile_progress`` via the
                   worker's 3-int ``progress_callback`` form.
    ``global_eta``— ``callable() -> Optional[float]`` returning the global
                   remaining seconds (or ``None``), included in progress lines.
    """

    def __init__(
        self,
        emit: Callable[[str, str], None],
        *,
        stage: Optional[Callable[[str, int, int], None]] = None,
        log_line: Optional[Callable[[str], None]] = None,
        global_eta: Optional[Callable[[], Optional[float]]] = None,
        interval_s: float = DEFAULT_PROGRESS_INTERVAL_S,
    ) -> None:
        self.emit = emit
        self.stage = stage
        self.log_line = log_line
        self.global_eta = global_eta
        self.interval_s = float(interval_s)
        self.name: Optional[str] = None
        self.total: Optional[int] = None
        self.unit: str = "items"
        self._t0: Optional[float] = None
        self._eta = PhaseEta()
        self._last_report: Optional[tuple[float, float]] = None

    # -- lifecycle ----------------------------------------------------------

    def start(self, name: str, total: Optional[int] = None, unit: str = "items") -> None:
        self.name = name
        self.total = total
        self.unit = unit
        self._t0 = time.perf_counter()
        self._eta.reset()
        self._last_report = None
        total_s = f" (total={total})" if total else ""
        self._out(f"phase START: {name}{total_s}")
        self._set_stage(0, total or 0)

    def new_pass(self, label: str) -> None:
        """Reset ETA samples for a new intra-phase pass (keep the phase open)."""
        self._eta.reset()
        self._last_report = None

    def end(self, elapsed_s: Optional[float] = None, throughput: Optional[str] = None) -> float:
        """Emit the END line with elapsed + optional throughput; returns elapsed."""
        if self._t0 is None:
            return 0.0
        elapsed = elapsed_s if elapsed_s is not None else time.perf_counter() - self._t0
        tp = f" throughput={throughput}" if throughput else ""
        self._out(f"phase END: {self.name} elapsed={elapsed:.3f}s{tp}")
        if self.total is not None:
            self._set_stage(self.total, self.total)
        self._t0 = None
        return float(elapsed)

    # -- progress -----------------------------------------------------------

    def progress(
        self,
        done: int,
        item_id: Optional[str] = None,
        *,
        force: bool = False,
        total: Optional[int] = None,
    ) -> None:
        """Report intra-phase progress at a bounded cadence (throttled).

        ``done`` is the monotonic count of completed items; ``item_id`` is the
        current item (frame id / cell id). ``total`` overrides the phase total
        (used when a single phase has multiple sub-passes with different totals).
        Always records an ETA sample (even when the emit is throttled) so the
        ETA stays fresh.
        """
        if self._t0 is None:
            return
        total = int(total) if total is not None else self.total
        if total is None or total <= 0:
            return
        done = int(done)
        now = time.perf_counter()
        elapsed = now - self._t0
        self._eta.observe(elapsed, done)
        if not force and self._last_report is not None:
            if (now - self._last_report[0]) < self.interval_s and done < total:
                return
        self._last_report = (now, done)
        pct = 100.0 * done / total
        eta = self._eta.eta_seconds(total)
        g_eta = None
        if self.global_eta is not None:
            try:
                g_eta = self.global_eta(eta)
            except Exception:
                g_eta = None
        item = f" item={item_id}" if item_id is not None else ""
        g = f" global_eta={format_eta(g_eta)}" if self.global_eta is not None else ""
        self._out(
            f"phase PROGRESS: {self.name} {done}/{total} "
            f"({pct:.1f}%){item} eta={format_eta(eta)}{g}"
        )
        self._set_stage(done, total)

    # -- internals ----------------------------------------------------------

    def _out(self, msg: str) -> None:
        try:
            self.emit(msg, "INFO")
        except Exception:
            pass
        if self.log_line is not None:
            try:
                self.log_line(msg)
            except Exception:
                pass

    def _set_stage(self, current: int, total: int) -> None:
        if self.stage is None or self.name is None:
            return
        # ZM-PROGRESS-CONTRACT-R29: emit a STABLE stage id (``zegrid:<phase>``)
        # and keep the counters in the separate current/total callback fields.
        # The id NEVER embeds ``<done>/<total>`` (unlike the legacy form) so the
        # GUI progress contract can key on a stable machine identity.
        stage_str = f"zegrid:{self.name}"
        try:
            self.stage(stage_str, int(current), int(total))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Sub-step reporter (ZM-ZEGRID-R27)
# ---------------------------------------------------------------------------

class SubstepReporter:
    """Bounded, throttled progress + timing for a nested SUB-STEP of a phase.

    Unlike :class:`PhaseReporter` (which drives the enclosing phase's
    ``stage``/GUI progress), this emits only log/durable-detail events — it never
    emits STAGE_PROGRESS percentages or stage ids — so the current GUI cannot
    misinterpret a layout sub-step as a global phase.

    Contracts:

    * The STABLE ``substep_id`` is separated from the human-readable display text
      and from counters/items; the id NEVER embeds counters (unlike the legacy
      ``zegrid:<phase>:<done>/<total>`` stage string).
    * Progress counters are throttled (``interval_s`` plus always the FIRST and
      the FINAL sample). When ``total`` is unknown (``None``) no percent/total is
      fabricated — only the completed count is emitted.
    * ``emit`` / ``log_line`` failures are swallowed: observability must never
      change or abort science.
    * The clock is injectable for deterministic tests.
    """

    def __init__(
        self,
        emit: Callable[[str, str], None],
        *,
        log_line: Optional[Callable[[str], None]] = None,
        clock: Optional[Callable[[], float]] = None,
        interval_s: float = DEFAULT_PROGRESS_INTERVAL_S,
    ) -> None:
        self._emit = emit
        self._log_line = log_line
        self._clock = clock if clock is not None else time.perf_counter
        self._interval_s = float(interval_s)
        self._id: Optional[str] = None
        self._t0: Optional[float] = None
        self._last_report: Optional[tuple[float, int]] = None
        self._elapsed: dict[str, float] = {}

    # -- lifecycle ----------------------------------------------------------

    def start(self, substep_id: str, display: str, *, total: Optional[int] = None) -> None:
        self._id = str(substep_id)
        self._t0 = self._clock()
        self._last_report = None
        total_s = f" (total={int(total)})" if total is not None else ""
        self._out(f"substep START: {self._id} — {display}{total_s}")

    def progress(
        self,
        done: int,
        *,
        item_id: Optional[str] = None,
        total: Optional[int] = None,
        force: bool = False,
    ) -> None:
        """Report throttled intra-sub-step progress (first + final always emitted).

        When ``total`` is ``None`` or ``<= 0``, only the completed count is emitted
        — no fabricated percent/total (used for the candidate scan, whose total
        candidate count is not known up-front).
        """
        if self._t0 is None:
            return
        done = int(done)
        now = self._clock()
        is_final = total is not None and int(total) > 0 and done >= int(total)
        if not force and not is_final and self._last_report is not None:
            if (now - self._last_report[0]) < self._interval_s:
                return
        self._last_report = (now, done)
        if total is not None and int(total) > 0:
            msg = (
                f"substep PROGRESS: {self._id} {done}/{int(total)} "
                f"({100.0 * done / int(total):.1f}%)"
            )
        else:
            msg = f"substep PROGRESS: {self._id} done={done}"
        if item_id is not None:
            msg += f" item={item_id}"
        self._out(msg)

    def end(self) -> float:
        """Emit the END line with elapsed; record + return the elapsed seconds."""
        if self._t0 is None:
            return 0.0
        elapsed = self._clock() - self._t0
        self._elapsed[self._id] = float(elapsed)
        self._out(f"substep END: {self._id} elapsed={elapsed:.3f}s")
        self._t0 = None
        return float(elapsed)

    def timings(self) -> dict[str, float]:
        """Return ``{substep_id: elapsed_s}`` recorded by ``end()``.

        Intended to be recorded as SUBORDINATE timings under the enclosing phase
        (the caller adds them to the run's ``Timings`` with a subordinate flag).
        """
        return dict(self._elapsed)

    # -- internals ----------------------------------------------------------

    def _out(self, msg: str) -> None:
        try:
            self._emit(msg, "INFO")
        except Exception:
            pass
        if self._log_line is not None:
            try:
                self._log_line(msg)
            except Exception:
                pass
