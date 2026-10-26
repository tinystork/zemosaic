"""ZM-ZEGRID-R41 — reusable host-RAM contract + planner primitives (Stage 1).

This module is a PURE, dependency-light contract layer:

* exact host budgets derived from a RAW available-RAM observation with integer
  arithmetic only (never a float, never ``0.8 * B_plan``);
* immutable/auditable, JSON-safe records for observations and the final plan
  envelope;
* a bounded admission controller that freshly observes before every admission,
  rebuilds under the PLAN budget from THAT SAME observation, ENFORCES that the
  rebuilt candidate really matches that raw/B_plan and is envelope-consistent,
  and refuses to grow the plan (refinement rank, tiles, backend, concurrency,
  pinned geometry) while it tries;
* a reusable envelope validator;
* phase-accounting math helpers used by :mod:`parallel` in opt-in mode.

It deliberately imports NO psutil and NO product mode (``zemosaic_zegrid_mode``):
it must stay unit-testable and reusable.

Two budgets, two observations
------------------------------
Tristan's accepted amendment (2026-10-16) replaces the historical single 80 %
factor with two explicit thresholds applied to the RAM *available* (not the
physical total):

* ``B_plan  = (4 * available_plan) // 5``  — the plan-BUILD budget;
* ``B_admit = (9 * available_now)  // 10`` — the admission CEILING for an
  already-built plan, re-measured just before execution.

Each factor is applied to its OWN raw observation EXACTLY once. The 90 % ceiling
is an ADMISSION-ONLY tolerance: it must never be handed to a rebuild (which
always receives ``B_plan``). The plan is rebuilt under ``B_plan`` computed from
the very observation that triggered the miss — never ``(9 // 10) * B_plan`` and
never a stale observation. The controller does NOT TRUST the rebuild callback: it
validates the returned envelope against the raw observation and ``B_plan`` it was
given (see :func:`validate_rebuilt_candidate`) before any later 90 % admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

__all__ = [
    "PLAN_FACTOR_NUM",
    "PLAN_FACTOR_DEN",
    "ADMIT_FACTOR_NUM",
    "ADMIT_FACTOR_DEN",
    "PROCESS_BASELINE_BYTES",
    "PHASE_PLAN",
    "PHASE_ADMIT",
    "PHASE_LAUNCH",
    "PHASES",
    "BACKEND_CPU",
    "BACKEND_GPU",
    "RamContractError",
    "InvalidObservation",
    "InvalidEnvelope",
    "GrowthRejected",
    "AdmissionExhausted",
    "PlanInfeasible",
    "Observation",
    "PlanEnvelope",
    "AdmissionOutcome",
    "AdmissionController",
    "plan_budget_bytes",
    "admit_budget_bytes",
    "requires_tolerance",
    "require_nonneg_int",
    "validate_envelope",
    "validate_rebuilt_candidate",
    "assert_no_growth",
    "stack_phase_total_bytes",
    "cache_per_cell_bytes",
    "cache_phase_total_bytes",
    "aggregate_phase_bytes",
    "AGGREGATE_RULE_MAX_OVER_PHASES",
    "AGGREGATE_RULE_STACK_PLUS_CACHE",
    "AGGREGATE_RULE_PROJECTED_HOST_PEAK",
    "AGGREGATE_RULES",
    "projected_host_peak_bytes",
    "feasible_concurrency",
]

# ---------------------------------------------------------------------------
# Factors (integer, exact)
# ---------------------------------------------------------------------------

PLAN_FACTOR_NUM = 4
PLAN_FACTOR_DEN = 5
ADMIT_FACTOR_NUM = 9
ADMIT_FACTOR_DEN = 10

# ZM-ZEGRID-R56 REWORK-1 (F2): explicit aggregation rules for the plan envelope.
AGGREGATE_RULE_MAX_OVER_PHASES = "max_over_phases"
AGGREGATE_RULE_STACK_PLUS_CACHE = "simultaneous_stack_plus_cache_minus_shared"
# ZM-ZEGRID-R56 UNIFIED PLAN/ADMIT/RUN projection: the ONE depth-aware host
# peak (serial at p=0, exactly one overlapping producer at p>=1).
AGGREGATE_RULE_PROJECTED_HOST_PEAK = "projected_host_peak_v1"
AGGREGATE_RULES = (
    AGGREGATE_RULE_MAX_OVER_PHASES,
    AGGREGATE_RULE_STACK_PLUS_CACHE,
    AGGREGATE_RULE_PROJECTED_HOST_PEAK,
)

# The SINGLE authoritative process floor shared by the host phase accounting.
# ``parallel._PER_WORKER_BASELINE_BYTES`` is assigned FROM this constant so the
# two can never drift (L3).
PROCESS_BASELINE_BYTES = 340 * 1024 * 1024

PHASE_PLAN = "plan"
PHASE_ADMIT = "admit"
PHASE_LAUNCH = "launch"
PHASES = (PHASE_PLAN, PHASE_ADMIT, PHASE_LAUNCH)

BACKEND_CPU = "cpu"
BACKEND_GPU = "gpu"
_BACKENDS = (BACKEND_CPU, BACKEND_GPU)


# ---------------------------------------------------------------------------
# Exceptions (fail closed, specifically)
# ---------------------------------------------------------------------------

class RamContractError(Exception):
    """Base class for the R41 host-RAM contract errors."""


class InvalidObservation(RamContractError, ValueError):
    """A raw available-RAM observation is missing, non-integer, zero or negative.

    ``None`` is an explicit 'unbounded' sentinel only in the LOW-LEVEL
    backward-compatible planner APIs (e.g. ``plan_cell_concurrency`` with
    ``ram_budget_bytes=None``); it is NEVER a valid production observation and
    is rejected here. A production observation must be a strictly positive int.
    """


class InvalidEnvelope(RamContractError):
    """A plan envelope is structurally invalid or lies about its raw/B_plan."""


class GrowthRejected(RamContractError):
    """A rebuild tried to GROW the plan (coarser layout, bigger tile, more
    concurrency), change its backend, or mutate a pinned layout identity."""


class AdmissionExhausted(RamContractError):
    """The plan still exceeded the admission ceiling after ``max_replans``.

    Carries a QUANTITATIVE trail: every raw observation, factor, budget and
    aggregate that was tried.
    """

    def __init__(self, message: str, *, attempts: tuple = ()):
        super().__init__(message)
        self.attempts = tuple(attempts)


class PlanInfeasible(RamContractError):
    """A single cell cannot fit even alone under the budget (never clamp to 1)."""


# ---------------------------------------------------------------------------
# Fail-closed numeric validation (M1)
# ---------------------------------------------------------------------------

def require_nonneg_int(value, name: str) -> int:
    """Public fail-closed validator: reject bool / non-int / negative (M1)."""
    return _require_nonneg_int(value, name)


def _require_nonneg_int(value, name: str) -> int:
    """Return ``value`` as an int, refusing bool / non-int / negative inputs.

    Bool is refused even though it is an ``int`` subclass: ``True`` silently
    meaning ``1`` bytes is exactly the kind of optimistic input that must fail
    closed.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise RamContractError(
            f"{name} must be a non-negative int, got "
            f"{type(value).__name__}={value!r}"
        )
    if value < 0:
        raise RamContractError(f"{name} must be non-negative, got {value}")
    return int(value)


def _require_optional_budget(value, name: str):
    """``None`` = explicit unbounded; otherwise a non-negative int (reject negative)."""
    if value is None:
        return None
    return _require_nonneg_int(value, name)


def _require_positive_observation(available_bytes) -> int:
    if available_bytes is None:
        raise InvalidObservation(
            "raw RAM observation is None: None is only valid as an explicit "
            "'unbounded' sentinel in low-level backward-compatible planner APIs, "
            "never as a production observation"
        )
    if isinstance(available_bytes, bool) or not isinstance(available_bytes, int):
        raise InvalidObservation(
            f"raw RAM observation must be a positive int, got "
            f"{type(available_bytes).__name__}={available_bytes!r}"
        )
    if available_bytes <= 0:
        raise InvalidObservation(
            f"raw RAM observation must be strictly positive, got {available_bytes}"
        )
    return int(available_bytes)


# ---------------------------------------------------------------------------
# Exact budgets
# ---------------------------------------------------------------------------

def plan_budget_bytes(available_bytes: int) -> int:
    """``B_plan = (4 * available) // 5`` — exact integer arithmetic."""
    a = _require_positive_observation(available_bytes)
    return (PLAN_FACTOR_NUM * a) // PLAN_FACTOR_DEN


def admit_budget_bytes(available_bytes: int) -> int:
    """``B_admit = (9 * available) // 10`` — exact integer arithmetic."""
    a = _require_positive_observation(available_bytes)
    return (ADMIT_FACTOR_NUM * a) // ADMIT_FACTOR_DEN


def requires_tolerance(aggregate_bytes: int, available_bytes: int) -> bool:
    """True iff the plan sits in the 80–90 % ADMISSION-ONLY tolerance band.

    ``(4 * available) // 5 < aggregate <= (9 * available) // 10``.

    This flag is DESCRIPTIVE (it records that the plan leans on the admission
    tolerance). It must never be used to enlarge a plan, a tile, or the
    concurrency.
    """
    return plan_budget_bytes(available_bytes) < int(aggregate_bytes) <= admit_budget_bytes(
        available_bytes
    )


# ---------------------------------------------------------------------------
# Immutable, auditable records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """One raw RAM observation with ITS factor and derived budget (applied once).

    ``factor_num`` / ``factor_den`` and the exact integer ``budget_bytes`` are the
    AUTHORITATIVE description of the factor. No serialized float factor is emitted
    (L4): a downstream consumer cannot reintroduce float drift.
    """

    phase: str
    raw_available_bytes: int
    factor_num: int
    factor_den: int
    budget_bytes: int

    def to_dict(self) -> dict:
        return {
            "phase": str(self.phase),
            "raw_available_bytes": int(self.raw_available_bytes),
            "factor_num": int(self.factor_num),
            "factor_den": int(self.factor_den),
            "budget_bytes": int(self.budget_bytes),
        }

    @classmethod
    def plan(cls, raw_available_bytes: int) -> "Observation":
        a = _require_positive_observation(raw_available_bytes)
        return cls(PHASE_PLAN, a, PLAN_FACTOR_NUM, PLAN_FACTOR_DEN, plan_budget_bytes(a))

    @classmethod
    def admit(cls, raw_available_bytes: int) -> "Observation":
        a = _require_positive_observation(raw_available_bytes)
        return cls(PHASE_ADMIT, a, ADMIT_FACTOR_NUM, ADMIT_FACTOR_DEN, admit_budget_bytes(a))

    @classmethod
    def launch(cls, raw_available_bytes: int) -> "Observation":
        """The pre-launch observation: 9/10 factor, tagged ``launch`` (L1)."""
        a = _require_positive_observation(raw_available_bytes)
        return cls(PHASE_LAUNCH, a, ADMIT_FACTOR_NUM, ADMIT_FACTOR_DEN, admit_budget_bytes(a))


@dataclass(frozen=True)
class PlanEnvelope:
    """The auditable projection of a planner result under one budget.

    Fields cover the whole decision: the RAW observation and its factor/budget,
    the scientific shape (layout + refinement rank + CPU/GPU tile), the backend,
    the concurrency + cache workers, and the SIMULTANEOUS phase totals
    (stack/cache/shared/aggregate) that the budget must hold. JSON-safe via
    :meth:`to_dict`.
    """

    phase: str
    raw_available_bytes: int
    factor_num: int
    factor_den: int
    budget_bytes: int
    layout: tuple
    refinement_rank: int
    cpu_tile: int
    gpu_tile: int | None
    backend: str
    cells_in_flight: int
    cache_workers_per_cell: int
    shared_bytes: int
    stack_total_bytes: int
    cache_total_bytes: int
    aggregate_bytes: int
    pinned: bool
    reason: str
    # ZM-ZEGRID-R56 REWORK-1 (F2): which aggregation rule produced
    # ``aggregate_bytes``. The default preserves every existing envelope; only
    # the ACTIVE GPU bundle declares the projected rule.
    aggregate_rule: str = AGGREGATE_RULE_MAX_OVER_PHASES
    # ZM-ZEGRID-R56 UNIFIED PROJECTION: the published depth inputs of the ONE
    # projected host peak (PLAN/ADMIT/RUN). They are only consumed by
    # AGGREGATE_RULE_PROJECTED_HOST_PEAK; defaults keep older envelopes valid.
    prefetch_units: int = 0
    producer_owner_bytes: int = 0
    prefetch_resident_per_slot_bytes: int = 0

    @property
    def layout_identity(self) -> tuple:
        return tuple(self.layout)

    def to_dict(self) -> dict:
        return {
            "phase": str(self.phase),
            "raw_available_bytes": int(self.raw_available_bytes),
            "factor_num": int(self.factor_num),
            "factor_den": int(self.factor_den),
            "budget_bytes": int(self.budget_bytes),
            "layout": [int(self.layout[0]), int(self.layout[1])],
            "refinement_rank": int(self.refinement_rank),
            "cpu_tile": int(self.cpu_tile),
            "gpu_tile": (None if self.gpu_tile is None else int(self.gpu_tile)),
            "backend": str(self.backend),
            "cells_in_flight": int(self.cells_in_flight),
            "cache_workers_per_cell": int(self.cache_workers_per_cell),
            "shared_bytes": int(self.shared_bytes),
            "stack_total_bytes": int(self.stack_total_bytes),
            "cache_total_bytes": int(self.cache_total_bytes),
            "aggregate_bytes": int(self.aggregate_bytes),
            "aggregate_rule": str(self.aggregate_rule),
            "prefetch_units": int(self.prefetch_units),
            "producer_owner_bytes": int(self.producer_owner_bytes),
            "prefetch_resident_per_slot_bytes": int(
                self.prefetch_resident_per_slot_bytes
            ),
            "pinned": bool(self.pinned),
            "reason": str(self.reason),
        }


@dataclass(frozen=True)
class AdmissionOutcome:
    """Result of :meth:`AdmissionController.run` (JSON-safe history included)."""

    plan: PlanEnvelope
    observations: tuple
    history: tuple
    replans: int
    tolerance: bool
    final_observation: Observation

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "observations": [o.to_dict() for o in self.observations],
            "history": [p.to_dict() for p in self.history],
            "replans": int(self.replans),
            "tolerance": bool(self.tolerance),
            "final_observation": self.final_observation.to_dict(),
        }


# ---------------------------------------------------------------------------
# Envelope validation (H1) — the controller does NOT trust the rebuild callback
# ---------------------------------------------------------------------------

def validate_envelope(envelope: PlanEnvelope) -> None:
    """Structurally validate a plan envelope (reusable; raises :class:`InvalidEnvelope`).

    Checks (all fail closed):

    * it IS a :class:`PlanEnvelope`;
    * ``phase`` is one of :data:`PHASES`;
    * ``layout`` is EXACTLY two positive integer dimensions;
    * ``factor_num`` / ``factor_den`` are positive ints and
      ``budget_bytes == (factor_num * raw) // factor_den`` (exact integer floor);
    * every numeric field is a non-negative int (bool refused): raw, budget,
      refinement rank, CPU tile, concurrency, cache workers, shared, stack total,
      cache total, aggregate, and a non-``None`` GPU tile;
    * ``aggregate_bytes == max(stack_total_bytes, cache_total_bytes)``;
    * ``stack_total_bytes >= shared_bytes`` and ``cache_total_bytes >= shared_bytes``;
    * ``backend`` is a known backend (``cpu`` / ``gpu``);
    * ``pinned`` is a bool.
    """
    if not isinstance(envelope, PlanEnvelope):
        raise InvalidEnvelope(
            f"expected a PlanEnvelope, got {type(envelope).__name__}"
        )
    if envelope.phase not in PHASES:
        raise InvalidEnvelope(f"unknown envelope phase {envelope.phase!r} (expected {PHASES})")
    layout = envelope.layout
    if (
        not isinstance(layout, (tuple, list))
        or len(layout) != 2
        or any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in layout)
    ):
        raise InvalidEnvelope(
            f"layout must be exactly two POSITIVE integer dimensions, got {layout!r}"
        )
    for name in ("factor_num", "factor_den"):
        d = getattr(envelope, name)
        if isinstance(d, bool) or not isinstance(d, int) or d <= 0:
            raise InvalidEnvelope(f"{name} must be a positive int, got {d!r}")
    raw = _require_positive_observation(envelope.raw_available_bytes)
    budget = _require_nonneg_int(envelope.budget_bytes, "budget_bytes")
    expected_budget = (int(envelope.factor_num) * raw) // int(envelope.factor_den)
    if budget != expected_budget:
        raise InvalidEnvelope(
            f"budget_bytes {budget} != (factor_num*raw)//factor_den "
            f"{expected_budget} (factor {envelope.factor_num}/{envelope.factor_den}, "
            f"raw {raw})"
        )
    for name in (
        "refinement_rank", "cpu_tile", "cells_in_flight",
        "cache_workers_per_cell", "shared_bytes", "stack_total_bytes",
        "cache_total_bytes", "aggregate_bytes", "prefetch_units",
        "producer_owner_bytes", "prefetch_resident_per_slot_bytes",
    ):
        _require_nonneg_int(getattr(envelope, name), name)
    if envelope.gpu_tile is not None:
        _require_nonneg_int(envelope.gpu_tile, "gpu_tile")
    if envelope.aggregate_rule not in AGGREGATE_RULES:
        raise InvalidEnvelope(
            f"unknown aggregate_rule {envelope.aggregate_rule!r} "
            f"(expected {AGGREGATE_RULES})"
        )
    if int(envelope.stack_total_bytes) < int(envelope.shared_bytes):
        raise InvalidEnvelope("stack_total_bytes < shared_bytes")
    if int(envelope.cache_total_bytes) < int(envelope.shared_bytes):
        raise InvalidEnvelope("cache_total_bytes < shared_bytes")
    if envelope.aggregate_rule == AGGREGATE_RULE_MAX_OVER_PHASES:
        expected_aggregate = max(
            int(envelope.stack_total_bytes), int(envelope.cache_total_bytes)
        )
    elif envelope.aggregate_rule == AGGREGATE_RULE_PROJECTED_HOST_PEAK:
        expected_aggregate = projected_host_peak_bytes(
            int(envelope.shared_bytes),
            int(envelope.stack_total_bytes) - int(envelope.shared_bytes),
            int(envelope.producer_owner_bytes),
            int(envelope.prefetch_units),
            int(envelope.prefetch_resident_per_slot_bytes),
        )
    else:
        expected_aggregate = simultaneous_stack_and_cache_bytes(
            int(envelope.stack_total_bytes),
            int(envelope.cache_total_bytes),
            int(envelope.shared_bytes),
        )
    if int(envelope.aggregate_bytes) != int(expected_aggregate):
        raise InvalidEnvelope(
            f"aggregate_bytes {envelope.aggregate_bytes} != {expected_aggregate} "
            f"under aggregate_rule {envelope.aggregate_rule!r} "
            f"(stack_total {envelope.stack_total_bytes}, cache_total "
            f"{envelope.cache_total_bytes}, shared {envelope.shared_bytes}, "
            f"prefetch {envelope.prefetch_units}, owner "
            f"{envelope.producer_owner_bytes}, resident/slot "
            f"{envelope.prefetch_resident_per_slot_bytes})"
        )
    if envelope.backend not in _BACKENDS:
        raise InvalidEnvelope(f"unknown backend {envelope.backend!r} (expected {_BACKENDS})")
    if not isinstance(envelope.pinned, bool):
        raise InvalidEnvelope("pinned must be a bool")


def validate_rebuilt_candidate(
    candidate: PlanEnvelope,
    *,
    raw_available_bytes: int,
    factor_num: int,
    factor_den: int,
    budget_bytes: int,
) -> None:
    """Enforce that a rebuilt candidate really corresponds to the supplied raw/B_plan.

    Required (raises :class:`InvalidEnvelope` otherwise), AFTER the structural
    :func:`validate_envelope`:

    * ``phase == PHASE_PLAN``;
    * ``raw_available_bytes`` equals the miss observation's raw EXACTLY;
    * factor equals the supplied factor (4/5 for a plan rebuild) EXACTLY;
    * ``budget_bytes == (factor_num * raw) // factor_den`` EXACTLY;
    * ``aggregate_bytes <= budget_bytes`` (the 80 % PLAN budget);
    * ``aggregate_bytes == max(stack_total_bytes, cache_total_bytes)`` (also in
      :func:`validate_envelope`, re-checked here so the failure names the rebuild).
    """
    validate_envelope(candidate)
    if candidate.phase != PHASE_PLAN:
        raise InvalidEnvelope(
            f"rebuilt candidate phase must be {PHASE_PLAN!r}, got {candidate.phase!r}"
        )
    if int(candidate.raw_available_bytes) != int(raw_available_bytes):
        raise InvalidEnvelope(
            f"rebuilt candidate raw {candidate.raw_available_bytes} != miss raw "
            f"{raw_available_bytes}"
        )
    if (int(candidate.factor_num), int(candidate.factor_den)) != (
        int(factor_num), int(factor_den)
    ):
        raise InvalidEnvelope(
            f"rebuilt candidate factor {candidate.factor_num}/{candidate.factor_den} "
            f"!= {factor_num}/{factor_den}"
        )
    expected_budget = (int(factor_num) * int(raw_available_bytes)) // int(factor_den)
    if int(candidate.budget_bytes) != int(budget_bytes) or int(budget_bytes) != expected_budget:
        raise InvalidEnvelope(
            f"rebuilt candidate budget {candidate.budget_bytes} != its B_plan "
            f"{budget_bytes} / exact {(4 * int(raw_available_bytes)) // 5}"
        )
    if int(candidate.aggregate_bytes) > int(budget_bytes):
        raise InvalidEnvelope(
            f"rebuilt candidate aggregate {candidate.aggregate_bytes} > B_plan "
            f"{budget_bytes} (the 80 % plan budget)"
        )


# ---------------------------------------------------------------------------
# No-growth guard (bounded replanning invariants)
# ---------------------------------------------------------------------------

def assert_no_growth(previous: PlanEnvelope, candidate: PlanEnvelope) -> None:
    """Reject any rebuild that GROWS or mutates the plan.

    Invariants (a later observation may NEVER buy a bigger plan):

    * refinement rank cannot decrease (no coarsening);
    * CPU tile cannot increase; GPU tile cannot increase (when both known);
    * GPU tile may only go value -> ``None`` (a degradation), never ``None`` ->
      value (a new GPU allocation);
    * concurrency cannot increase;
    * ``pinned`` cannot change, and a pinned layout identity cannot change;
    * the BACKEND may only degrade ``gpu`` -> ``cpu``; ``cpu`` -> ``gpu`` and any
      unknown backend / transition is refused (L2). This still allows the
      legitimate future GPU -> CPU fallback.

    Raises :class:`GrowthRejected` with every violation listed.
    """
    problems = []
    if int(candidate.refinement_rank) < int(previous.refinement_rank):
        problems.append(
            f"refinement rank decreased/coarsened ({previous.refinement_rank} -> "
            f"{candidate.refinement_rank})"
        )
    if int(candidate.cpu_tile) > int(previous.cpu_tile):
        problems.append(f"cpu tile increased ({previous.cpu_tile} -> {candidate.cpu_tile})")
    if previous.gpu_tile is None and candidate.gpu_tile is not None:
        problems.append(
            f"gpu tile introduced (None -> {candidate.gpu_tile}); a rebuild cannot "
            f"allocate a GPU tile"
        )
    elif previous.gpu_tile is not None and candidate.gpu_tile is not None:
        if int(candidate.gpu_tile) > int(previous.gpu_tile):
            problems.append(
                f"gpu tile increased ({previous.gpu_tile} -> {candidate.gpu_tile})"
            )
    if int(candidate.cells_in_flight) > int(previous.cells_in_flight):
        problems.append(
            f"concurrency increased ({previous.cells_in_flight} -> "
            f"{candidate.cells_in_flight})"
        )
    if bool(candidate.pinned) != bool(previous.pinned):
        problems.append(f"pinned flag changed ({previous.pinned} -> {candidate.pinned})")
    if previous.pinned and tuple(candidate.layout) != tuple(previous.layout):
        problems.append(
            f"pinned layout identity changed ({tuple(previous.layout)} -> "
            f"{tuple(candidate.layout)})"
        )
    # Backend monotonicity (L2): only gpu -> cpu degradation is allowed.
    prev_be, cand_be = previous.backend, candidate.backend
    if prev_be not in _BACKENDS or cand_be not in _BACKENDS:
        problems.append(f"unknown backend transition ({prev_be!r} -> {cand_be!r})")
    elif prev_be == BACKEND_CPU and cand_be == BACKEND_GPU:
        problems.append("backend upgraded cpu -> gpu (a rebuild cannot introduce a GPU)")
    if problems:
        raise GrowthRejected("; ".join(problems))


# ---------------------------------------------------------------------------
# Phase accounting (integer arithmetic; fail-closed on unsafe inputs)
# ---------------------------------------------------------------------------

def stack_phase_total_bytes(
    shared_bytes: int, cells_in_flight: int, stack_per_cell_bound_bytes: int
) -> int:
    """``stack_total = shared + K * stack_per_cell_bound``.

    ``shared`` is UNCAPPED and counted ONCE (never per cell). All inputs must be
    non-negative ints (bool refused) — a negative input would silently produce an
    OPTIMISTIC total, so it fails closed.
    """
    shared = _require_nonneg_int(shared_bytes, "shared_bytes")
    k = _require_nonneg_int(cells_in_flight, "cells_in_flight")
    bound = _require_nonneg_int(stack_per_cell_bound_bytes, "stack_per_cell_bound_bytes")
    return shared + k * bound


def cache_per_cell_bytes(
    process_baseline_bytes: int,
    cache_workers_per_cell: int,
    cache_worker_footprint_bytes: int,
) -> int:
    """Conservative per-cell cache footprint.

    Cache workers run INSIDE each active cell process, so the cell's own process
    baseline is counted ONCE and each of the ``W`` workers contributes only its
    INCREMENTAL footprint over that baseline::

        cache_per_cell = PROCESS_BASELINE + W * max(0, cache_worker_footprint - PROCESS_BASELINE)

    This is the correction for the discarded r2 undercount (``K*W*delta+shared``),
    which omitted one process baseline per active cell. The ``max(0, ...)`` floors
    a footprint BELOW the baseline (that is a deliberate clamp, not masked
    negativity); the scalar inputs themselves must be non-negative.
    """
    pb = _require_nonneg_int(process_baseline_bytes, "process_baseline_bytes")
    w = _require_nonneg_int(cache_workers_per_cell, "cache_workers_per_cell")
    cwf = _require_nonneg_int(cache_worker_footprint_bytes, "cache_worker_footprint_bytes")
    delta = max(0, cwf - pb)
    return pb + w * delta


def cache_phase_total_bytes(
    shared_bytes: int, cells_in_flight: int, cache_per_cell_bytes_: int
) -> int:
    """``cache_total = shared + K * cache_per_cell`` (shared counted once)."""
    shared = _require_nonneg_int(shared_bytes, "shared_bytes")
    k = _require_nonneg_int(cells_in_flight, "cells_in_flight")
    cpc = _require_nonneg_int(cache_per_cell_bytes_, "cache_per_cell_bytes")
    return shared + k * cpc


def aggregate_phase_bytes(stack_total_bytes: int, cache_total_bytes: int) -> int:
    """The SIMULTANEOUS host footprint = ``max(stack_total, cache_total)``."""
    st = _require_nonneg_int(stack_total_bytes, "stack_total_bytes")
    ct = _require_nonneg_int(cache_total_bytes, "cache_total_bytes")
    return max(st, ct)


# ---------------------------------------------------------------------------
# ZM-ZEGRID-R56 REWORK-1 (F2): true simultaneous ownership
# ---------------------------------------------------------------------------


def simultaneous_stack_and_cache_bytes(
    stack_total_bytes: int, cache_total_bytes: int, shared_bytes: int
) -> int:
    """Host bytes when the stack and the cache-build set COEXIST.

    Both phase totals already count ``shared_bytes`` once, so the simultaneous
    total is ``stack + cache - shared`` (shared counted exactly once). This is
    the ACTIVE GPU pipeline case: the producer's cache-build worker set runs
    while the consumer stacks the previous Cell. It is NOT a blanket sum of all
    phases: only owners that genuinely coexist are added here.
    """
    st = _require_nonneg_int(stack_total_bytes, "stack_total_bytes")
    ct = _require_nonneg_int(cache_total_bytes, "cache_total_bytes")
    sh = _require_nonneg_int(shared_bytes, "shared_bytes")
    if st < sh or ct < sh:
        raise InvalidEnvelope(
            "simultaneous aggregate requires stack_total and cache_total >= "
            f"shared (stack={st}, cache={ct}, shared={sh})"
        )
    return int(st) + int(ct) - int(sh)


def projected_host_peak_bytes(
    shared_bytes: int,
    stack_added_bytes: int,
    producer_owner_bytes: int,
    prefetch_units: int,
    prefetch_resident_per_slot_bytes: int,
) -> int:
    """THE single PLAN/ADMIT/RUN projected host peak (depth-aware).

    ::

        projected = shared
                  + stack_added
                  + (producer_owner if prefetch >= 1 else 0)
                  + prefetch * prefetch_resident_per_slot

    Semantics (established, not re-derived here):

    * ``shared`` is ``PROCESS_BASELINE`` counted exactly ONCE — never per cell
      and never twice.
    * ``stack_added`` is the ADDED streaming bound above that baseline for the
      active stack (``K * per-tile added bound``), NOT the total estimate.
    * ``producer_owner`` is the ONE producer phase that genuinely coexists with
      the consumer (``max(cache build set, cell normalization)``, already
      reduced to one owner by the caller). It is charged ONLY at ``prefetch >=
      1``: at ``prefetch == 0`` the queue is full during the stack, so the
      producer is blocked and no concurrent owner exists.
    * ``prefetch_resident_per_slot`` is the resident host credit of ONE queued
      slot. The on-disk bytes stay in their own ledger dimension.
    * the initial global gauge is a phase BEFORE the pipeline and is never
      summed here.

    Every input must be a non-negative plain int (bool refused); an invalid
    input fails closed instead of producing an optimistic peak.
    """
    shared = _require_nonneg_int(shared_bytes, "shared_bytes")
    added = _require_nonneg_int(stack_added_bytes, "stack_added_bytes")
    owner = _require_nonneg_int(producer_owner_bytes, "producer_owner_bytes")
    units = _require_nonneg_int(prefetch_units, "prefetch_units")
    resident = _require_nonneg_int(
        prefetch_resident_per_slot_bytes, "prefetch_resident_per_slot_bytes"
    )
    total = shared + added + units * resident
    if units >= 1:
        total += owner
    return int(total)


def feasible_concurrency(
    *,
    budget_bytes,
    shared_bytes: int,
    stack_per_cell_bound_bytes: int,
    process_baseline_bytes: int,
    cache_worker_footprint_bytes: int,
    cache_workers_for_k: Callable[[int], int],
    max_k: int,
) -> tuple:
    """Largest ``K`` in ``[1, max_k]`` whose phase aggregate fits ``budget_bytes``.

    ``budget_bytes is None`` means EXPLICIT UNBOUNDED. A finite budget must be a
    non-negative int (a negative finite budget is refused); ``0`` is a real,
    finite budget that refuses any positive aggregate.

    Reduces K from ``max_k`` down to 1, recomputing the cache worker count (which
    depends on K) each time. Returns ``(k, stack_total, cache_total, aggregate,
    cache_per_cell, cache_workers)`` for the largest feasible K; returns all-zero /
    ``None`` when even ``K == 1`` is infeasible (ONE-CELL REFUSAL — never a
    clamp-to-1 rescue). A ``cache_workers_for_k`` callback returning a negative
    value is REJECTED (never masked with ``max(0, ...)``).
    """
    budget = _require_optional_budget(budget_bytes, "budget_bytes")
    shared = _require_nonneg_int(shared_bytes, "shared_bytes")
    bound = _require_nonneg_int(stack_per_cell_bound_bytes, "stack_per_cell_bound_bytes")
    pb = _require_nonneg_int(process_baseline_bytes, "process_baseline_bytes")
    cwf = _require_nonneg_int(cache_worker_footprint_bytes, "cache_worker_footprint_bytes")
    mk = _require_nonneg_int(max_k, "max_k")
    for k in range(mk, 0, -1):
        w = cache_workers_for_k(k)
        if isinstance(w, bool) or not isinstance(w, int) or w < 0:
            raise RamContractError(
                f"cache_workers_for_k({k}) must return a non-negative int, got {w!r}"
            )
        cpc = cache_per_cell_bytes(pb, int(w), cwf)
        st = stack_phase_total_bytes(shared, k, bound)
        ct = cache_phase_total_bytes(shared, k, cpc)
        agg = aggregate_phase_bytes(st, ct)
        if budget is None or agg <= budget:
            return k, st, ct, agg, cpc, int(w)
    return 0, None, None, None, None, None


# ---------------------------------------------------------------------------
# Bounded admission controller (pure; no psutil, no product import)
# ---------------------------------------------------------------------------

class AdmissionController:
    """Bounded, auditable admission loop over an already-built plan.

    Contract
    --------
    * a FRESH ``measure_available()`` happens before EVERY admission attempt;
    * an unchanged plan PASSES when ``aggregate <= B_admit`` of that observation;
    * on a miss, ``rebuild(B_plan, previous)`` is called with ``B_plan`` computed
      from THAT SAME raw observation (never the 90 % ceiling, never a stale one);
    * the rebuilt candidate is NOT trusted: it must be ``phase=plan``, carry the
      SAME raw, the SAME 4/5 factor, that exact ``B_plan`` and
      ``aggregate <= B_plan``, and be envelope-consistent; otherwise a specific
      :class:`InvalidEnvelope` is raised BEFORE any later 90 % admission;
    * before launch the machine is observed AGAIN (a ``launch``-tagged 9/10
      observation) and re-checked; that observation defines the final tolerance;
    * at most ``max_replans`` (default 2) replans, then a QUANTITATIVE
      :class:`AdmissionExhausted`;
    * every accepted rebuild is additionally subjected to :func:`assert_no_growth`.

    ``measure_available`` returns a strictly positive int (bytes). ``None``,
    zero, negative or non-integer fail closed with :class:`InvalidObservation`.
    ``rebuild`` returns a new :class:`PlanEnvelope`.
    """

    def __init__(self, max_replans: int = 2):
        if isinstance(max_replans, bool) or not isinstance(max_replans, int):
            raise ValueError("max_replans must be an int")
        if max_replans < 0:
            raise ValueError("max_replans must be >= 0")
        self.max_replans = int(max_replans)

    @staticmethod
    def _observe(measure_available: Callable[[], int], phase: str) -> Observation:
        raw = measure_available()
        if phase == PHASE_PLAN:
            return Observation.plan(raw)
        if phase == PHASE_LAUNCH:
            return Observation.launch(raw)
        return Observation.admit(raw)

    def run(
        self,
        initial_plan: PlanEnvelope,
        measure_available: Callable[[], int],
        rebuild: Callable[[int, PlanEnvelope], PlanEnvelope],
    ) -> AdmissionOutcome:
        validate_envelope(initial_plan)
        observations = []
        history = [initial_plan]
        plan = initial_plan
        replans = 0

        while True:
            # Fresh observation BEFORE every admission attempt.
            admit_obs = self._observe(measure_available, PHASE_ADMIT)
            observations.append(admit_obs)

            if int(plan.aggregate_bytes) <= int(admit_obs.budget_bytes):
                # Admitted. Observe AGAIN immediately before launch (tagged launch).
                launch_obs = self._observe(measure_available, PHASE_LAUNCH)
                observations.append(launch_obs)
                if int(plan.aggregate_bytes) <= int(launch_obs.budget_bytes):
                    validate_envelope(plan)
                    return AdmissionOutcome(
                        plan=plan,
                        observations=tuple(observations),
                        history=tuple(history),
                        replans=replans,
                        tolerance=requires_tolerance(
                            plan.aggregate_bytes, launch_obs.raw_available_bytes
                        ),
                        final_observation=launch_obs,
                    )
                miss_obs = launch_obs
            else:
                miss_obs = admit_obs

            # Miss: rebuild under B_plan from THAT SAME raw observation.
            if replans >= self.max_replans:
                raise AdmissionExhausted(
                    "plan still exceeds the admission ceiling after "
                    f"{self.max_replans} replan(s): aggregate "
                    f"{plan.aggregate_bytes / 2**20:.1f} MiB > B_admit "
                    f"{miss_obs.budget_bytes / 2**20:.1f} MiB (raw available "
                    f"{miss_obs.raw_available_bytes / 2**20:.1f} MiB)",
                    attempts=tuple(observations),
                )
            plan_obs = Observation.plan(miss_obs.raw_available_bytes)
            observations.append(plan_obs)
            candidate = rebuild(plan_obs.budget_bytes, plan)
            # ENFORCE the rebuilt contract — never trust the callback (H1).
            validate_rebuilt_candidate(
                candidate,
                raw_available_bytes=plan_obs.raw_available_bytes,
                factor_num=PLAN_FACTOR_NUM,
                factor_den=PLAN_FACTOR_DEN,
                budget_bytes=plan_obs.budget_bytes,
            )
            assert_no_growth(plan, candidate)
            plan = candidate
            history.append(plan)
            replans += 1
