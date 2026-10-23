"""ZM-ZEGRID-R55-B — deterministic CPU admission planner (observe-only).

Scope of this module:

* bounded, deterministic candidate enumeration over the PRODUCT tile ladder
  (``128..32``; R54's ``t16``/``t24`` are NOT members);
* integer-only hard constraints: host RAM, disk residency, CPU slots;
* a fixed objective/tie-break chain and a stable canonical plan hash;
* a shrink-only API (prefetch -> workers -> units_in_flight -> tile -> backend)
  with explicit growth rejection. R55-B implements the CONTRACT only: it never
  mutates a RUN;
* :class:`PlanSet` assembly: one :class:`ResourceEnvelope` per required phase,
  margins applied once at the aggregate boundary.

GPU enumeration is deliberately absent from R55-B: requesting a GPU plan raises
a configuration error rather than silently degrading. The product observe path
always plans the CPU backend it actually runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from . import calibration as zcal
from .resource_envelope import (
    AGGREGATE_RULE_MAX_OVER_PHASES,
    BACKEND_CPU,
    BACKEND_GPU,
    BACKENDS,
    BoundEnvelope,
    CELL_STACK_PHASES,
    ConcurrencyEnvelope,
    DEFAULT_MARGIN_DEN,
    DEFAULT_MARGIN_NUM,
    DeviceEnvelope,
    EstimateEnvelope,
    GeometryEnvelope,
    HostEnvelope,
    IdentityEnvelope,
    InvalidEnvelope,
    InvalidOverlap,
    AccountingEnvelope,
    PHASE_CELL_STACK_INMEM,
    PHASE_CELL_STACK_STREAM,
    PHASE_GUI_PREVIEW,
    PHASES,
    REASON_CODES,
    REASON_OK,
    REASON_REFUSE_DISK,
    REASON_REFUSE_HOST_RAM,
    REASON_SHRINK_CONCURRENCY,
    REASON_SHRINK_PREFETCH,
    REASON_SHRINK_TILE,
    REASON_SHRINK_WORKERS,
    REASON_UNCALIBRATED,
    REQUIRED_PLAN_PHASES,
    ResourceEnvelope,
    aggregate_device_peak_bytes,
    aggregate_disk_bytes,
    aggregate_host_peak_bytes,
    ceil_margin_div,
    compute_plan_hash,
    make_estimate,
    require_int,
    validate_envelope_set,
)

__all__ = [
    "SCHEDULER_VERSION",
    "MANIFEST_SCHEMA",
    "MODES",
    "MODE_LEGACY",
    "MODE_OBSERVE",
    "DEFAULT_MODE",
    "MIN_TILE_PX",
    "PRODUCT_TILE_LADDER",
    "PREFETCH_LADDER",
    "WORKERS_LADDER",
    "CPU_OS_RESERVE",
    "SHRINK_ORDER",
    "SHRINK_REASON_CODES",
    "SchedulerError",
    "SchedulerConfigError",
    "InvalidSchedulerMode",
    "PlanRefused",
    "GrowthRejected",
    "FrozenIdentity",
    "PlanContext",
    "Candidate",
    "PhaseRequest",
    "PlanSet",
    "normalize_scheduler_mode",
    "disk_residency_bytes",
    "enumerate_candidates",
    "objective_key",
    "build_plan_set",
    "plan",
    "build_observe_plan",
    "next_shrink",
    "shrink_sequence",
    "assert_shrink_only",
    "assert_same_identity",
]

SCHEDULER_VERSION = "r55-b0"
MANIFEST_SCHEMA = "zemosaic-r55-manifest-v1"

MODE_LEGACY = "legacy"
MODE_OBSERVE = "observe"
MODES = (MODE_LEGACY, MODE_OBSERVE)
DEFAULT_MODE = MODE_LEGACY

MIN_TILE_PX = 32
# The PRODUCT ladder, in the product's preferred (descending) order. R54's
# t16/t24 are experimental OOC tiles and are intentionally absent.
PRODUCT_TILE_LADDER = (128, 112, 96, 80, 64, 48, 32)
PREFETCH_LADDER = (0, 1)
WORKERS_LADDER = (1, 2, 3, 4, 6, 8, 12, 14)
CPU_OS_RESERVE = 2

SHRINK_ORDER = (
    "prefetch_units",
    "workers_per_unit",
    "units_in_flight",
    "tile_px",
    "backend",
)
SHRINK_REASON_CODES = {
    "prefetch_units": REASON_SHRINK_PREFETCH,
    "workers_per_unit": REASON_SHRINK_WORKERS,
    "units_in_flight": REASON_SHRINK_CONCURRENCY,
    "tile_px": REASON_SHRINK_TILE,
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SchedulerError(Exception):
    """Base class for R55 scheduler errors."""


class SchedulerConfigError(SchedulerError, ValueError):
    """The planning inputs are structurally invalid or out of R55-B scope."""


class InvalidSchedulerMode(SchedulerConfigError):
    """``zegrid_scheduler_mode`` is unknown — never silently coerced."""


class GrowthRejected(SchedulerError):
    """A rebuild or shrink step tried to GROW/freeze-violate the plan."""


class PlanRefused(SchedulerError):
    """No candidate satisfies the hard constraints.

    Carries the closed reason code and the candidate trail; the caller records a
    refusal block instead of pretending a plan exists.
    """

    def __init__(self, reason_code: str, detail: str, *, candidates: Sequence = ()):
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = str(reason_code)
        self.detail = str(detail)
        self.candidates = tuple(candidates)

    def refusal_manifest_dict(
        self, *, mode: str = MODE_OBSERVE, run_id: str = ""
    ) -> dict:
        return {
            "schema": MANIFEST_SCHEMA,
            "scheduler_version": SCHEDULER_VERSION,
            "mode": "observe_only" if mode == MODE_OBSERVE else "active",
            "requested_mode": mode,
            "run_id": str(run_id),
            "refused": True,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "plans": [
                c.to_dict() for c in self.candidates if isinstance(c, Candidate)
            ],
        }


def normalize_scheduler_mode(value, *, default: str = DEFAULT_MODE) -> str:
    """Resolve ``zegrid_scheduler_mode``.

    ``None`` (absent) means ``legacy``. An explicit value must be the exact
    lowercase string ``legacy`` or ``observe``. Anything else — including
    ``True``, ``"LEGACY"``, ``"auto"`` or ``""`` — raises
    :class:`InvalidSchedulerMode` so an unknown value can never be silently
    coerced to the default.
    """
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, str):
        raise InvalidSchedulerMode(
            "zegrid_scheduler_mode must be a string "
            f"({MODE_LEGACY!r} or {MODE_OBSERVE!r}), got "
            f"{type(value).__name__}={value!r}"
        )
    if value not in MODES:
        raise InvalidSchedulerMode(
            f"unknown zegrid_scheduler_mode {value!r}; expected one of {MODES}"
        )
    return value


# ---------------------------------------------------------------------------
# Frozen identity and distance helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrozenIdentity:
    """The scientific identity that no shrink may change."""

    layout: tuple[int, int]
    cell_ids: tuple[str, ...]
    frame_ids: tuple[str, ...]
    halo_px: int
    refinement_rank: int

    def to_dict(self) -> dict:
        return {
            "layout": [int(self.layout[0]), int(self.layout[1])],
            "cell_ids": list(self.cell_ids),
            "frame_ids": list(self.frame_ids),
            "halo_px": int(self.halo_px),
            "refinement_rank": int(self.refinement_rank),
        }


def assert_same_identity(previous: FrozenIdentity, candidate: FrozenIdentity) -> None:
    """Raise :class:`GrowthRejected` unless every frozen field is identical."""
    if not isinstance(previous, FrozenIdentity) or not isinstance(candidate, FrozenIdentity):
        raise SchedulerConfigError("identity comparison requires FrozenIdentity values")
    problems = []
    if tuple(previous.layout) != tuple(candidate.layout):
        problems.append(f"layout changed {tuple(previous.layout)} -> {tuple(candidate.layout)}")
    if tuple(previous.cell_ids) != tuple(candidate.cell_ids):
        problems.append("cell identity/membership changed")
    if tuple(previous.frame_ids) != tuple(candidate.frame_ids):
        problems.append("frame set changed")
    if int(previous.halo_px) != int(candidate.halo_px):
        problems.append(f"halo changed {previous.halo_px} -> {candidate.halo_px}")
    if int(previous.refinement_rank) != int(candidate.refinement_rank):
        problems.append(
            f"refinement rank changed {previous.refinement_rank} -> "
            f"{candidate.refinement_rank}"
        )
    if problems:
        raise GrowthRejected("; ".join(problems))


def disk_residency_bytes(
    *,
    units_in_flight: int,
    prefetch_units: int,
    per_unit_cache_bytes: int,
    per_unit_prefetch_bytes: int,
) -> int:
    """Disk residency under the D2 prefetch rule.

    Each of the ``K`` in-flight units holds exactly ONE aligned cache. Prefetch
    depth ``P`` holds ``P`` concurrently resident buffers: depth 1 therefore
    counts its shared reusable buffer ONCE, and depth N counts each of the N
    resident buffers exactly once. The buffer is NEVER multiplied by the tile
    count or by the frame count.
    """
    k = require_int(units_in_flight, "units_in_flight")
    p = require_int(prefetch_units, "prefetch_units")
    cache = require_int(per_unit_cache_bytes, "per_unit_cache_bytes")
    prefetch = require_int(per_unit_prefetch_bytes, "per_unit_prefetch_bytes")
    return k * cache + p * prefetch


# ---------------------------------------------------------------------------
# Plan context and candidate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanContext:
    """Every integer input the CPU planner needs. No hidden defaults for bytes."""

    phase: str
    backend: str
    budget_bytes: int
    raw_available_bytes: int
    factor_num: int
    factor_den: int
    cpu_logical: int
    units_total: int
    shared_bytes: int
    per_unit_stack_bytes: int
    per_unit_cache_bytes: int
    per_unit_prefetch_bytes: int
    disk_free_bytes: int
    layout: tuple[int, int]
    refinement_rank: int
    patch_hw: tuple[int, int] | None
    halo_px: int
    channels: int
    n_frames: int
    disk_fraction_num: int = 1
    disk_fraction_den: int = 2
    tile_ladder: tuple[int, ...] = PRODUCT_TILE_LADDER
    prefetch_ladder: tuple[int, ...] = PREFETCH_LADDER
    workers_ladder: tuple[int, ...] = WORKERS_LADDER
    workers_cap: int | None = None
    margin_num: int = DEFAULT_MARGIN_NUM
    margin_den: int = DEFAULT_MARGIN_DEN
    profile: zcal.CalibrationProfile | None = None
    time_model: Callable[[int | None, int, int], int] | None = None
    cell_ids: tuple[str, ...] = ()
    frame_ids: tuple[str, ...] = ()
    observation_source: str = "plan.available_bytes"
    run_id: str = ""
    pinned: bool = False

    @property
    def patch_area_px(self) -> int:
        if self.patch_hw is None:
            return 0
        return int(self.patch_hw[0]) * int(self.patch_hw[1])

    @property
    def identity(self) -> FrozenIdentity:
        return FrozenIdentity(
            layout=tuple(int(v) for v in self.layout),
            cell_ids=tuple(sorted(self.cell_ids)),
            frame_ids=tuple(sorted(self.frame_ids)),
            halo_px=int(self.halo_px),
            refinement_rank=int(self.refinement_rank),
        )


@dataclass(frozen=True)
class Candidate:
    """One deterministic candidate tuple with its integer projections."""

    backend: str
    tile_px: int | None
    units_in_flight: int
    prefetch_units: int
    workers_per_unit: int
    host_peak_bytes: int
    disk_bytes: int
    device_peak_bytes: int
    feasible: bool
    reason_code: str
    infeasible_reason: str | None = None
    predicted_makespan_ns: int | None = None
    estimate_method: str = "static_bound"
    calibration_id: str | None = None

    def ordering_key(self) -> tuple:
        """The documented lexicographic ENUMERATION key.

        ``(backend_rank, -tile_px, -units_in_flight, -prefetch_units, -workers)``
        — descending resources, CPU first.
        """
        backend_rank = 0 if self.backend == BACKEND_CPU else 1
        tile = 0 if self.tile_px is None else int(self.tile_px)
        return (
            backend_rank,
            -tile,
            -int(self.units_in_flight),
            -int(self.prefetch_units),
            -int(self.workers_per_unit),
        )

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "tile_px": (None if self.tile_px is None else int(self.tile_px)),
            "units_in_flight": int(self.units_in_flight),
            "prefetch_units": int(self.prefetch_units),
            "workers_per_unit": int(self.workers_per_unit),
            "host_peak_bytes": int(self.host_peak_bytes),
            "disk_bytes": int(self.disk_bytes),
            "device_peak_bytes": int(self.device_peak_bytes),
            "feasible": bool(self.feasible),
            "reason_code": self.reason_code,
            "infeasible_reason": self.infeasible_reason,
            "predicted_makespan_ns": self.predicted_makespan_ns,
            "estimate_method": self.estimate_method,
            "calibration_id": self.calibration_id,
        }


@dataclass(frozen=True)
class PhaseRequest:
    """One non-stack phase envelope request (gauge/cache/assembly/export/...)."""

    phase: str
    components: Mapping[str, int]
    units: int
    units_in_flight: int
    workers_per_unit: int = 1
    prefetch_units: int = 0
    tile_px: int | None = None
    disk_bytes: int = 0
    overlaps_with: tuple[str, ...] = ()
    estimate_method: str = "static_bound"
    calibration_id: str | None = None


@dataclass(frozen=True)
class PlanSet:
    """A deterministic plan: one envelope per required phase plus its hash."""

    mode: str
    budget_bytes: int
    raw_available_bytes: int
    aggregate_host_peak_bytes: int
    aggregate_host_peak_bytes_raw: int
    aggregate_device_peak_bytes: int
    aggregate_disk_bytes: int
    chosen: Candidate
    phases: tuple[ResourceEnvelope, ...]
    decisions: tuple[dict, ...]
    plan_hash: str
    identity: FrozenIdentity
    run_id: str = ""
    scheduler_version: str = SCHEDULER_VERSION

    def phase_map(self) -> dict[str, ResourceEnvelope]:
        return {e.phase: e for e in self.phases}

    def to_manifest_dict(self) -> dict:
        """Additive, deterministic ``r55_scheduler`` manifest block.

        No observed metric is fabricated: predicted envelopes only. The legacy
        ``ram_contract``/``cpu_streaming``/``cache``/``gpu`` blocks are untouched.
        """
        order = {p: i for i, p in enumerate(PHASES)}
        phases = {}
        for env in sorted(self.phases, key=lambda e: order[e.phase]):
            phases[env.phase] = {"envelope": env.to_dict()}
        calibrations = sorted(
            {
                e.estimate.calibration_id
                for e in self.phases
                if e.estimate.calibration_id
            }
        )
        return {
            "schema": MANIFEST_SCHEMA,
            "scheduler_version": self.scheduler_version,
            "mode": "observe_only" if self.mode == MODE_OBSERVE else "active",
            "requested_mode": self.mode,
            "run_id": self.run_id,
            "plan_hash": self.plan_hash,
            "budget_bytes": int(self.budget_bytes),
            "raw_available_bytes": int(self.raw_available_bytes),
            "aggregate_host_peak_bytes": int(self.aggregate_host_peak_bytes),
            "aggregate_host_peak_bytes_raw": int(self.aggregate_host_peak_bytes_raw),
            "aggregate_device_peak_bytes": int(self.aggregate_device_peak_bytes),
            "aggregate_disk_bytes": int(self.aggregate_disk_bytes),
            "margin": {
                "num": int(self.phases[0].bound.margin_num),
                "den": int(self.phases[0].bound.margin_den),
                "applied_once_at": "aggregate",
            },
            "calibrations": calibrations,
            "phases": phases,
            "decisions": [dict(d) for d in self.decisions],
            "fallbacks": [],
            "identity": self.identity.to_dict(),
            "observed_metrics": None,
            "note": (
                "R55-B observe-only: these are PREDICTED integer envelopes. "
                "Execution remains the legacy path; no observed metric is recorded "
                "and no RUN decision was taken."
            ),
        }


# ---------------------------------------------------------------------------
# Context validation and enumeration
# ---------------------------------------------------------------------------

def _validate_context(context: PlanContext) -> None:
    if not isinstance(context, PlanContext):
        raise SchedulerConfigError(
            f"expected PlanContext, got {type(context).__name__}"
        )
    if context.backend not in BACKENDS:
        raise SchedulerConfigError(f"unknown backend {context.backend!r}")
    if context.backend == BACKEND_GPU:
        raise SchedulerConfigError(
            "R55-B enumerates the CPU backend only; GPU admission is not part of "
            "this mission"
        )
    if context.phase not in CELL_STACK_PHASES:
        raise SchedulerConfigError(
            f"R55-B plans a cell_stack_* phase, got {context.phase!r}"
        )
    if context.phase != PHASE_CELL_STACK_STREAM and context.profile is not None:
        raise SchedulerConfigError(
            "a calibrated profile is only modelled for cell_stack_stream in R55-B"
        )
    raw = require_int(context.raw_available_bytes, "raw_available_bytes", minimum=1)
    num = require_int(context.factor_num, "factor_num", minimum=1)
    den = require_int(context.factor_den, "factor_den", minimum=1)
    budget = require_int(context.budget_bytes, "budget_bytes")
    expected = (num * raw) // den
    if budget != expected:
        raise SchedulerConfigError(
            f"budget_bytes {budget} != (factor*raw)//den {expected}"
        )
    require_int(context.cpu_logical, "cpu_logical", minimum=1)
    require_int(context.units_total, "units_total", minimum=1)
    require_int(context.shared_bytes, "shared_bytes")
    require_int(context.per_unit_stack_bytes, "per_unit_stack_bytes", minimum=1)
    require_int(context.per_unit_cache_bytes, "per_unit_cache_bytes")
    require_int(context.per_unit_prefetch_bytes, "per_unit_prefetch_bytes")
    require_int(context.disk_free_bytes, "disk_free_bytes")
    lx, ly = context.layout
    if int(lx) <= 0 or int(ly) <= 0:
        raise SchedulerConfigError(f"layout must be positive, got {context.layout!r}")
    require_int(context.refinement_rank, "refinement_rank")
    require_int(context.halo_px, "halo_px")
    require_int(context.channels, "channels", minimum=1)
    require_int(context.n_frames, "n_frames", minimum=1)
    if context.patch_hw is not None:
        ph, pw = context.patch_hw
        if int(ph) <= 0 or int(pw) <= 0:
            raise SchedulerConfigError(
                f"patch_hw must be positive, got {context.patch_hw!r}"
            )
    dfn = require_int(context.disk_fraction_num, "disk_fraction_num", minimum=1)
    dfd = require_int(context.disk_fraction_den, "disk_fraction_den", minimum=1)
    if dfn > dfd:
        raise SchedulerConfigError(
            f"disk fraction {dfn}/{dfd} must not exceed 1"
        )
    mnum = require_int(context.margin_num, "margin_num")
    mden = require_int(context.margin_den, "margin_den", minimum=1)
    if mnum > mden:
        raise SchedulerConfigError(f"margin {mnum}/{mden} must not exceed 1")
    if not context.tile_ladder:
        raise SchedulerConfigError("tile_ladder must not be empty")
    seen = set()
    for tile in context.tile_ladder:
        t = require_int(tile, "tile_ladder entry", minimum=MIN_TILE_PX)
        if t in seen:
            raise SchedulerConfigError(f"duplicate tile {t} in ladder")
        seen.add(t)
    if tuple(context.tile_ladder) != tuple(
        sorted(context.tile_ladder, reverse=True)
    ):
        raise SchedulerConfigError(
            "tile_ladder must be strictly descending (canonical enumeration)"
        )
    if tuple(context.prefetch_ladder) != tuple(
        sorted(set(context.prefetch_ladder))
    ):
        raise SchedulerConfigError(
            "prefetch_ladder must be ascending and unique"
        )
    if not context.prefetch_ladder:
        raise SchedulerConfigError("prefetch_ladder must not be empty")
    for p in context.prefetch_ladder:
        require_int(p, "prefetch_ladder entry")
    if not context.workers_ladder:
        raise SchedulerConfigError("workers_ladder must not be empty")
    for w in context.workers_ladder:
        require_int(w, "workers_ladder entry", minimum=1)
    if tuple(context.workers_ladder) != tuple(sorted(set(context.workers_ladder))):
        raise SchedulerConfigError("workers_ladder must be ascending and unique")
    if context.workers_cap is not None:
        require_int(context.workers_cap, "workers_cap", minimum=1)
    if context.cell_ids and len(context.cell_ids) != int(context.units_total):
        raise SchedulerConfigError(
            f"units_total {context.units_total} != len(cell_ids) {len(context.cell_ids)}"
        )
    if context.profile is not None:
        context.profile.validate()
        if context.profile.phase != context.phase:
            raise SchedulerConfigError(
                f"profile phase {context.profile.phase!r} != context phase "
                f"{context.phase!r}"
            )
        if context.profile.backend != context.backend:
            raise SchedulerConfigError(
                f"profile backend {context.profile.backend!r} != context backend "
                f"{context.backend!r}"
            )


def _candidate_stack_bytes(
    context: PlanContext, tile_px: int | None, *, calibrated: bool
) -> tuple[int, str, str | None]:
    """Return ``(per_unit_stack_bytes, estimate_method, calibration_id)``.

    A calibrated profile's linear model is authoritative ONLY for an in-range
    tile. Outside ``valid_ranges`` the scheduler never extrapolates: it uses the
    caller's declared static bound and marks the candidate ``R55_UNCALIBRATED``
    (the caller then makes the out-of-range candidate ineligible).
    """
    if context.profile is None or not calibrated:
        return int(context.per_unit_stack_bytes), "static_bound", None
    assert tile_px is not None
    components = zcal.estimate_components(
        context.profile,
        tile_px=int(tile_px),
        n_frames=int(context.n_frames),
        patch_area_px=int(context.patch_area_px),
        channels=int(context.channels),
    )
    return sum(components.values()), "calibrated", context.profile.calibration_id


def enumerate_candidates(context: PlanContext) -> tuple[Candidate, ...]:
    """Exhaustively enumerate the bounded CPU candidate set, canonically sorted.

    Enumeration is exhaustive over the declared ladders and independent of the
    order of ``context.cell_ids`` (they are sorted before use). Every candidate
    carries its integer projections and a closed reason code, even when
    infeasible — refusals must be explainable.
    """
    _validate_context(context)
    cpu_slots = max(0, int(context.cpu_logical) - CPU_OS_RESERVE)
    k_max = min(cpu_slots, int(context.units_total))
    disk_cap = (
        int(context.disk_free_bytes) * int(context.disk_fraction_num)
    ) // int(context.disk_fraction_den)

    tiles: tuple[int | None, ...]
    if context.phase == PHASE_CELL_STACK_INMEM:
        tiles = (None,)
        in_range_tiles = {None}
    else:
        tiles = tuple(int(t) for t in context.tile_ladder)
        in_range_tiles = set(tiles)
        if context.profile is not None:
            lo, hi = context.profile.range_map["tile_px"]
            in_range_tiles = {t for t in tiles if lo <= t <= hi}

    out: list[Candidate] = []
    for tile in tiles:
        in_range = tile in in_range_tiles
        per_unit, method, cal_id = _candidate_stack_bytes(
            context, tile, calibrated=in_range
        )
        for k in range(1, k_max + 1):
            for prefetch in context.prefetch_ladder:
                if k * (1 + int(prefetch)) > cpu_slots:
                    continue
                w_max = max(1, int(context.cpu_logical) // max(1, k))
                if context.workers_cap is not None:
                    w_max = min(w_max, int(context.workers_cap))
                for workers in context.workers_ladder:
                    if int(workers) > w_max:
                        continue
                    host_peak = (
                        int(context.shared_bytes)
                        + k * per_unit
                        + int(prefetch) * int(context.per_unit_prefetch_bytes)
                    )
                    disk = disk_residency_bytes(
                        units_in_flight=k,
                        prefetch_units=int(prefetch),
                        per_unit_cache_bytes=int(context.per_unit_cache_bytes),
                        per_unit_prefetch_bytes=int(context.per_unit_prefetch_bytes),
                    )

                    if not in_range:
                        feasible = False
                        reason = REASON_UNCALIBRATED
                        detail = (
                            f"tile {tile} is outside the profile's declared "
                            f"valid_ranges[tile_px]={context.profile.range_map['tile_px']}; "
                            "refusing to extrapolate"
                        )
                    elif disk > disk_cap:
                        feasible = False
                        reason = REASON_REFUSE_DISK
                        detail = (
                            f"disk residency {disk} B > budget {disk_cap} B "
                            f"(free {context.disk_free_bytes} x "
                            f"{context.disk_fraction_num}/{context.disk_fraction_den})"
                        )
                    elif ceil_margin_div(host_peak, context.margin_num, context.margin_den) > int(
                        context.budget_bytes
                    ):
                        feasible = False
                        reason = REASON_REFUSE_HOST_RAM
                        detail = (
                            f"host peak {host_peak} B with margin "
                            f"{context.margin_num}/{context.margin_den} exceeds budget "
                            f"{context.budget_bytes} B"
                        )
                    else:
                        feasible = True
                        reason = REASON_OK if method == "calibrated" else REASON_UNCALIBRATED
                        detail = None

                    makespan = None
                    if feasible and method == "calibrated" and context.time_model is not None:
                        wave = (int(context.units_total) + k - 1) // k
                        per_unit_ns = int(
                            context.time_model(tile, k, int(prefetch))
                        )
                        if per_unit_ns < 0:
                            raise SchedulerConfigError(
                                "time_model returned a negative duration"
                            )
                        makespan = wave * per_unit_ns
                        if int(prefetch) > 0:
                            # Never credit unmeasured overlap: the caller's model
                            # must already include its own penalty.
                            pass

                    out.append(
                        Candidate(
                            backend=context.backend,
                            tile_px=(None if tile is None else int(tile)),
                            units_in_flight=int(k),
                            prefetch_units=int(prefetch),
                            workers_per_unit=int(workers),
                            host_peak_bytes=int(host_peak),
                            disk_bytes=int(disk),
                            device_peak_bytes=0,
                            feasible=bool(feasible),
                            reason_code=reason,
                            infeasible_reason=detail,
                            predicted_makespan_ns=makespan,
                            estimate_method=method,
                            calibration_id=cal_id,
                        )
                    )

    out.sort(key=lambda c: c.ordering_key())
    keys = [c.ordering_key() for c in out]
    if len(set(keys)) != len(keys):
        raise SchedulerConfigError(
            "candidate enumeration produced duplicate keys; ladders must be unique"
        )
    return tuple(out)


def objective_key(candidate: Candidate) -> tuple:
    """The fixed objective/tie-break chain (lower is better).

    1. higher conservative throughput (lower predicted makespan); uncalibrated
       candidates rank behind every calibrated one;
    2. lower host peak;
    3. lower device peak;
    4. CPU before GPU;
    5. larger tile;
    6. lower units_in_flight;
    7. lower prefetch_units;
    8. lexicographic candidate key.
    """
    if not isinstance(candidate, Candidate):
        raise SchedulerConfigError("objective_key expects a Candidate")
    has_time = candidate.predicted_makespan_ns is not None
    tile = -1 if candidate.tile_px is None else int(candidate.tile_px)
    return (
        0 if has_time else 1,
        int(candidate.predicted_makespan_ns or 0),
        int(candidate.host_peak_bytes),
        int(candidate.device_peak_bytes),
        0 if candidate.backend == BACKEND_CPU else 1,
        -tile,
        int(candidate.units_in_flight),
        int(candidate.prefetch_units),
        candidate.ordering_key(),
    )


# ---------------------------------------------------------------------------
# PlanSet assembly
# ---------------------------------------------------------------------------

def _host_envelope(context: PlanContext) -> HostEnvelope:
    return HostEnvelope(
        observation_source=context.observation_source,
        raw_available_bytes=int(context.raw_available_bytes),
        factor_num=int(context.factor_num),
        factor_den=int(context.factor_den),
        budget_bytes=int(context.budget_bytes),
        reserve_bytes=0,
        process_baseline_bytes=int(context.shared_bytes),
    )


def _geometry(context: PlanContext, *, tile_px: int | None, cell_id: str | None = None) -> GeometryEnvelope:
    return GeometryEnvelope(
        layout=tuple(int(v) for v in context.layout),
        refinement_rank=int(context.refinement_rank),
        cell_id=cell_id,
        patch_hw=(None if context.patch_hw is None else tuple(int(v) for v in context.patch_hw)),
        tile_px=(None if tile_px is None else int(tile_px)),
        halo_px=int(context.halo_px),
        channels=int(context.channels),
    )


def _bound(context: PlanContext, method: str) -> BoundEnvelope:
    return BoundEnvelope(
        kind="upper",
        margin_num=int(context.margin_num),
        margin_den=int(context.margin_den),
        confidence=("measured_residual" if method == "calibrated" else "static_bound"),
    )


def _stack_envelope(context: PlanContext, candidate: Candidate) -> ResourceEnvelope:
    components = {
        "prefetch_residency_bytes": (
            int(candidate.prefetch_units) * int(context.per_unit_prefetch_bytes)
        ),
        "shared_bytes": int(context.shared_bytes),
        "stack_in_flight_bytes": (
            int(candidate.units_in_flight) * int(context.per_unit_stack_bytes)
            if candidate.estimate_method == "static_bound"
            else int(candidate.units_in_flight)
            * _candidate_stack_bytes(
                context,
                candidate.tile_px,
                calibrated=(candidate.estimate_method == "calibrated"),
            )[0]
        ),
    }
    estimate = make_estimate(
        components,
        method=candidate.estimate_method,
        calibration_id=candidate.calibration_id,
    )
    return ResourceEnvelope(
        envelope_id=f"{context.phase}.{context.backend}",
        phase=context.phase,
        backend=context.backend,
        host=_host_envelope(context),
        device=None,
        geometry=_geometry(context, tile_px=candidate.tile_px),
        concurrency=ConcurrencyEnvelope(
            units=int(context.units_total),
            units_in_flight=int(candidate.units_in_flight),
            workers_per_unit=int(candidate.workers_per_unit),
            prefetch_units=int(candidate.prefetch_units),
        ),
        estimate=estimate,
        bound=_bound(context, candidate.estimate_method),
        accounting=AccountingEnvelope(
            aggregate_rule=AGGREGATE_RULE_MAX_OVER_PHASES,
            host_peak_bytes=int(estimate.peak_bytes),
            device_peak_bytes=0,
            disk_bytes=int(candidate.disk_bytes),
            transfer_bytes=0,
        ),
        overlaps_with=(),
        identity=IdentityEnvelope(run_id=context.run_id, plan_hash="", pinned=bool(context.pinned)),
        reason_code=candidate.reason_code,
    )


def _request_envelope(context: PlanContext, request: PhaseRequest) -> ResourceEnvelope:
    if not isinstance(request, PhaseRequest):
        raise SchedulerConfigError("phase_requests entries must be PhaseRequest")
    estimate = make_estimate(
        request.components,
        method=request.estimate_method,
        calibration_id=request.calibration_id,
    )
    return ResourceEnvelope(
        envelope_id=f"{request.phase}.{context.backend}",
        phase=request.phase,
        backend=context.backend,
        host=_host_envelope(context),
        device=None,
        geometry=_geometry(context, tile_px=request.tile_px),
        concurrency=ConcurrencyEnvelope(
            units=int(request.units),
            units_in_flight=int(request.units_in_flight),
            workers_per_unit=int(request.workers_per_unit),
            prefetch_units=int(request.prefetch_units),
        ),
        estimate=estimate,
        bound=_bound(context, request.estimate_method),
        accounting=AccountingEnvelope(
            aggregate_rule=AGGREGATE_RULE_MAX_OVER_PHASES,
            host_peak_bytes=int(estimate.peak_bytes),
            device_peak_bytes=0,
            disk_bytes=int(request.disk_bytes),
            transfer_bytes=0,
        ),
        overlaps_with=tuple(request.overlaps_with),
        identity=IdentityEnvelope(run_id=context.run_id, plan_hash="", pinned=bool(context.pinned)),
        reason_code=(
            REASON_OK if request.estimate_method == "calibrated" else REASON_UNCALIBRATED
        ),
    )


def build_plan_set(
    context: PlanContext,
    *,
    candidate: Candidate,
    phase_requests: Sequence[PhaseRequest],
    mode: str = MODE_OBSERVE,
) -> PlanSet:
    """Assemble and validate a PlanSet for one candidate.

    Raises :class:`InvalidEnvelope`/:class:`InvalidOverlap` on any contract
    violation and ``PlanRefused(R55_REFUSE_HOST_RAM)`` when the margin-applied
    aggregate exceeds the budget.
    """
    _validate_context(context)
    if mode not in MODES:
        raise InvalidSchedulerMode(f"unknown plan mode {mode!r}")
    if not isinstance(candidate, Candidate):
        raise SchedulerConfigError("candidate must be a Candidate")
    if not candidate.feasible:
        raise PlanRefused(
            candidate.reason_code,
            f"cannot build a plan from an infeasible candidate: "
            f"{candidate.infeasible_reason}",
            candidates=[candidate],
        )
    requests = tuple(phase_requests)
    phases_present = {r.phase for r in requests}
    missing = [p for p in REQUIRED_PLAN_PHASES if p not in phases_present]
    if missing:
        raise SchedulerConfigError(
            f"phase_requests is missing required PLAN phase(s): {missing}"
        )
    if context.phase in phases_present:
        raise SchedulerConfigError(
            f"phase_requests must not include the stack phase {context.phase!r}: "
            "it is derived from the chosen candidate"
        )
    stacks = [p for p in phases_present if p in CELL_STACK_PHASES]
    if stacks:
        raise SchedulerConfigError(
            f"phase_requests must contain no cell_stack_* phase, got {stacks}"
        )

    envelopes = [_request_envelope(context, r) for r in requests]
    envelopes.append(_stack_envelope(context, candidate))
    validated = validate_envelope_set(envelopes, headless=True)
    raw_peak, peak = aggregate_host_peak_bytes(validated)
    if peak > int(context.budget_bytes):
        raise PlanRefused(
            REASON_REFUSE_HOST_RAM,
            f"aggregate host peak {peak} B (raw {raw_peak} B + margin "
            f"{context.margin_num}/{context.margin_den}) > budget "
            f"{context.budget_bytes} B",
            candidates=[candidate],
        )

    plan_hash = compute_plan_hash(validated)
    # The hash covers immutable decision fields only; stamping it does not feed
    # back into the hash.
    stamped = tuple(
        ResourceEnvelope(
            envelope_id=e.envelope_id,
            phase=e.phase,
            backend=e.backend,
            host=e.host,
            device=e.device,
            geometry=e.geometry,
            concurrency=e.concurrency,
            estimate=e.estimate,
            bound=e.bound,
            accounting=e.accounting,
            overlaps_with=e.overlaps_with,
            identity=IdentityEnvelope(
                run_id=e.identity.run_id,
                plan_hash=plan_hash,
                pinned=e.identity.pinned,
            ),
            reason_code=e.reason_code,
        )
        for e in validated
    )

    device_raw, device_peak = aggregate_device_peak_bytes(stamped)
    del device_raw
    chosen_env = next(e for e in stamped if e.phase == context.phase)
    decision = {
        "at": "plan",
        "chosen": chosen_env.envelope_id,
        "reason_code": candidate.reason_code,
        "candidate": {
            "backend": candidate.backend,
            "tile_px": candidate.tile_px,
            "units_in_flight": candidate.units_in_flight,
            "prefetch_units": candidate.prefetch_units,
            "workers_per_unit": candidate.workers_per_unit,
        },
        "host_peak_bytes": int(candidate.host_peak_bytes),
        "disk_bytes": int(candidate.disk_bytes),
        "budget_bytes": int(context.budget_bytes),
        "aggregate_host_peak_bytes": int(peak),
        "plan_hash": plan_hash,
    }
    if candidate.reason_code not in REASON_CODES:
        raise SchedulerConfigError(
            f"reason_code {candidate.reason_code!r} outside the closed set"
        )

    return PlanSet(
        mode=mode,
        budget_bytes=int(context.budget_bytes),
        raw_available_bytes=int(context.raw_available_bytes),
        aggregate_host_peak_bytes=int(peak),
        aggregate_host_peak_bytes_raw=int(raw_peak),
        aggregate_device_peak_bytes=int(device_peak),
        aggregate_disk_bytes=int(aggregate_disk_bytes(stamped)),
        chosen=candidate,
        phases=stamped,
        decisions=(decision,),
        plan_hash=plan_hash,
        identity=context.identity,
        run_id=context.run_id,
    )


def plan(
    context: PlanContext,
    *,
    phase_requests: Sequence[PhaseRequest],
    mode: str = MODE_OBSERVE,
) -> PlanSet:
    """Choose and build the deterministic winning PlanSet.

    Feasible candidates are ordered by :func:`objective_key`; the first whose
    validated aggregate fits the budget wins. If none does, ``PlanRefused``
    carries the closed reason code (disk shortfall is a hard refusal).
    """
    candidates = enumerate_candidates(context)
    feasible = [c for c in candidates if c.feasible]
    if not feasible:
        disk_only = bool(candidates) and all(
            c.reason_code == REASON_REFUSE_DISK for c in candidates
        )
        reason = REASON_REFUSE_DISK if disk_only else REASON_REFUSE_HOST_RAM
        detail = (
            "no CPU candidate fits the hard constraints"
            + (" (disk shortfall: persistent aligned cache does not shrink with "
               "the stack tile, so no tile downgrade is offered as a remedy)"
               if disk_only else "")
        )
        raise PlanRefused(reason, detail, candidates=candidates)

    last: PlanRefused | None = None
    for candidate in sorted(feasible, key=objective_key):
        try:
            return build_plan_set(
                context,
                candidate=candidate,
                phase_requests=phase_requests,
                mode=mode,
            )
        except PlanRefused as exc:
            last = exc
            continue
    assert last is not None
    raise last


def build_observe_plan(
    *,
    phase: str,
    budget_bytes: int,
    raw_available_bytes: int,
    factor_num: int,
    factor_den: int,
    cpu_logical: int,
    shared_bytes: int,
    per_unit_stack_bytes: int,
    per_unit_cache_bytes: int,
    per_unit_prefetch_bytes: int,
    disk_free_bytes: int,
    layout: tuple[int, int],
    refinement_rank: int,
    patch_hw: tuple[int, int] | None,
    halo_px: int,
    channels: int,
    n_frames: int,
    phase_requests: Sequence[PhaseRequest],
    units_total: int | None = None,
    cell_ids: Sequence[str] = (),
    frame_ids: Sequence[str] = (),
    backend: str = BACKEND_CPU,
    mode: str = MODE_OBSERVE,
    run_id: str = "",
    pinned: bool = False,
    tile_ladder: tuple[int, ...] = PRODUCT_TILE_LADDER,
    prefetch_ladder: tuple[int, ...] = PREFETCH_LADDER,
    workers_ladder: tuple[int, ...] = WORKERS_LADDER,
    workers_cap: int | None = None,
    margin_num: int = DEFAULT_MARGIN_NUM,
    margin_den: int = DEFAULT_MARGIN_DEN,
    disk_fraction_num: int = 1,
    disk_fraction_den: int = 2,
    profile: zcal.CalibrationProfile | None = None,
    time_model: Callable[[int | None, int, int], int] | None = None,
    observation_source: str = "plan.available_bytes",
) -> PlanSet:
    """Product adapter: build the PlanContext and choose a plan in one call."""
    if mode not in MODES:
        raise InvalidSchedulerMode(f"unknown plan mode {mode!r}")
    cells = tuple(str(c) for c in cell_ids)
    if units_total is None:
        if not cells:
            raise SchedulerConfigError(
                "units_total is required when cell_ids is not supplied"
            )
        units_total = len(cells)
    context = PlanContext(
        phase=phase,
        backend=backend,
        budget_bytes=int(budget_bytes),
        raw_available_bytes=int(raw_available_bytes),
        factor_num=int(factor_num),
        factor_den=int(factor_den),
        cpu_logical=int(cpu_logical),
        units_total=int(units_total),
        shared_bytes=int(shared_bytes),
        per_unit_stack_bytes=int(per_unit_stack_bytes),
        per_unit_cache_bytes=int(per_unit_cache_bytes),
        per_unit_prefetch_bytes=int(per_unit_prefetch_bytes),
        disk_free_bytes=int(disk_free_bytes),
        layout=tuple(int(v) for v in layout),
        refinement_rank=int(refinement_rank),
        patch_hw=(None if patch_hw is None else tuple(int(v) for v in patch_hw)),
        halo_px=int(halo_px),
        channels=int(channels),
        n_frames=int(n_frames),
        disk_fraction_num=int(disk_fraction_num),
        disk_fraction_den=int(disk_fraction_den),
        tile_ladder=tuple(int(t) for t in tile_ladder),
        prefetch_ladder=tuple(int(p) for p in prefetch_ladder),
        workers_ladder=tuple(int(w) for w in workers_ladder),
        workers_cap=workers_cap,
        margin_num=int(margin_num),
        margin_den=int(margin_den),
        profile=profile,
        time_model=time_model,
        cell_ids=cells,
        frame_ids=tuple(str(f) for f in frame_ids),
        observation_source=str(observation_source),
        run_id=str(run_id),
        pinned=bool(pinned),
    )
    return plan(context, phase_requests=phase_requests, mode=mode)


# ---------------------------------------------------------------------------
# Shrink-only API (contract only; R55-B never mutates a RUN)
# ---------------------------------------------------------------------------

def next_shrink(
    candidate: Candidate, context: PlanContext
) -> tuple[str | None, Candidate | None]:
    """Return the next legal shrink step in the FIXED order, or ``(None, None)``.

    Order: ``prefetch_units -> workers_per_unit -> units_in_flight -> tile_px ->
    backend``. Each call performs at most ONE step, so a RUN checkpoint can
    re-check feasibility after every step.
    """
    if not isinstance(candidate, Candidate) or not isinstance(context, PlanContext):
        raise SchedulerConfigError("next_shrink requires Candidate + PlanContext")

    if candidate.prefetch_units > 0:
        return "prefetch_units", _replace_candidate(
            candidate, prefetch_units=candidate.prefetch_units - 1
        )

    workers_values = [int(w) for w in context.workers_ladder]
    if candidate.workers_per_unit > 1:
        lower = [w for w in workers_values if w < candidate.workers_per_unit]
        next_w = max(lower) if lower else candidate.workers_per_unit - 1
        return "workers_per_unit", _replace_candidate(candidate, workers_per_unit=next_w)

    if candidate.units_in_flight > 1:
        return "units_in_flight", _replace_candidate(
            candidate, units_in_flight=candidate.units_in_flight - 1
        )

    if candidate.backend == BACKEND_GPU:
        return "backend", _replace_candidate(candidate, backend=BACKEND_CPU)

    if candidate.tile_px is not None:
        ladder = [int(t) for t in context.tile_ladder]
        if candidate.tile_px in ladder:
            idx = ladder.index(candidate.tile_px)
            if idx + 1 < len(ladder):
                return "tile_px", _replace_candidate(
                    candidate, tile_px=ladder[idx + 1]
                )
    return None, None


def _replace_candidate(candidate: Candidate, **changes) -> Candidate:
    data = {
        "backend": candidate.backend,
        "tile_px": candidate.tile_px,
        "units_in_flight": candidate.units_in_flight,
        "prefetch_units": candidate.prefetch_units,
        "workers_per_unit": candidate.workers_per_unit,
        "host_peak_bytes": candidate.host_peak_bytes,
        "disk_bytes": candidate.disk_bytes,
        "device_peak_bytes": candidate.device_peak_bytes,
        "feasible": candidate.feasible,
        "reason_code": candidate.reason_code,
        "infeasible_reason": candidate.infeasible_reason,
        "predicted_makespan_ns": candidate.predicted_makespan_ns,
        "estimate_method": candidate.estimate_method,
        "calibration_id": candidate.calibration_id,
    }
    data.update(changes)
    return Candidate(**data)


def shrink_sequence(
    candidate: Candidate, context: PlanContext
) -> tuple[tuple[str, Candidate], ...]:
    """Apply the fixed shrink order repeatedly until no step remains."""
    steps: list[tuple[str, Candidate]] = []
    current = candidate
    while True:
        field_name, nxt = next_shrink(current, context)
        if field_name is None or nxt is None:
            return tuple(steps)
        assert_shrink_only(current, nxt)
        steps.append((field_name, nxt))
        current = nxt


def assert_shrink_only(previous: Candidate, candidate: Candidate) -> None:
    """Reject ANY growth or unsupported transition (shrink-only contract)."""
    if not isinstance(previous, Candidate) or not isinstance(candidate, Candidate):
        raise SchedulerConfigError("shrink comparison requires two Candidates")
    problems: list[str] = []
    if previous.backend not in BACKENDS or candidate.backend not in BACKENDS:
        problems.append(
            f"unknown backend transition {previous.backend!r} -> {candidate.backend!r}"
        )
    elif previous.backend == BACKEND_CPU and candidate.backend == BACKEND_GPU:
        problems.append("backend upgraded cpu -> gpu (a shrink cannot introduce a GPU)")
    if previous.tile_px is None and candidate.tile_px is not None:
        problems.append(
            f"tile introduced None -> {candidate.tile_px} (inmem -> stream is a growth)"
        )
    elif previous.tile_px is not None and candidate.tile_px is None:
        problems.append(
            f"tile removed {previous.tile_px} -> None (stream -> inmem is a growth)"
        )
    elif (
        previous.tile_px is not None
        and candidate.tile_px is not None
        and int(candidate.tile_px) > int(previous.tile_px)
    ):
        problems.append(
            f"tile increased {previous.tile_px} -> {candidate.tile_px}"
        )
    if int(candidate.units_in_flight) > int(previous.units_in_flight):
        problems.append(
            f"units_in_flight increased {previous.units_in_flight} -> "
            f"{candidate.units_in_flight}"
        )
    if int(candidate.prefetch_units) > int(previous.prefetch_units):
        problems.append(
            f"prefetch_units increased {previous.prefetch_units} -> "
            f"{candidate.prefetch_units}"
        )
    if int(candidate.workers_per_unit) > int(previous.workers_per_unit):
        problems.append(
            f"workers_per_unit increased {previous.workers_per_unit} -> "
            f"{candidate.workers_per_unit}"
        )
    if problems:
        raise GrowthRejected("; ".join(problems))
