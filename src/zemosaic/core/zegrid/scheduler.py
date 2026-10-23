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

R55-D adds an ADDITIVE joint CPU/GPU core (:class:`JointPlanInput`,
:func:`enumerate_joint_candidates`, :func:`plan_joint`) over explicit plain-
integer GPU facts. It does not grant execution authority, does not probe CuPy or
CUDA, and leaves every R55-B CPU path and product execution path unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
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
    PHASE_ASSEMBLY,
    PHASE_CELL_STACK_INMEM,
    PHASE_CELL_STACK_STREAM,
    PHASE_GUI_PREVIEW,
    PHASES,
    REASON_CODES,
    REASON_DEFAULT_OFF,
    REASON_GPU_UNAVAILABLE,
    REASON_INTERNAL_ERROR,
    REASON_OK,
    REASON_PARITY_FAIL,
    REASON_REFUSE_DISK,
    REASON_REFUSE_HOST_RAM,
    REASON_REFUSE_VRAM,
    REASON_SHRINK_CONCURRENCY,
    REASON_SHRINK_PREFETCH,
    REASON_SHRINK_TILE,
    REASON_SHRINK_WORKERS,
    REASON_UNCALIBRATED,
    REASON_UNKNOWN_BACKEND,
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
    "JOINT_SCHEMA",
    "JOINT_EVIDENCE_SCHEMA",
    "JOINT_SCHEDULER_VERSION",
    "MODES",
    "MODE_LEGACY",
    "MODE_OBSERVE",
    "MODE_ACTIVE",
    "DEFAULT_MODE",
    "ACTIVE_MANIFEST_SCHEMA",
    "ACTIVE_MAPPING_SCHEMA",
    "ACTIVE_FINAL_PLAN_SCHEMA",
    "ACTIVE_EXECUTABLE_PHASES",
    "ActivePlanRejected",
    "active_cpu_phase",
    "active_mapping_for_candidate",
    "map_active_cpu_candidate",
    "active_final_plan_hash",
    "active_manifest_dict",
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
    "GpuFacts",
    "JointRefusal",
    "JointPlanInput",
    "normalize_scheduler_mode",
    "disk_residency_bytes",
    "enumerate_candidates",
    "enumerate_joint_candidates",
    "joint_refusal_trail",
    "select_joint_candidate",
    "objective_key",
    "build_plan_set",
    "plan",
    "build_joint_plan_set",
    "plan_joint",
    "joint_input_payload",
    "joint_enumeration_payload",
    "joint_enumeration_hash",
    "joint_evidence_payload",
    "joint_evidence_json",
    "joint_evidence_hash",
    "build_observe_plan",
    "next_shrink",
    "shrink_sequence",
    "assert_shrink_only",
    "assert_joint_shrink_only",
    "assert_same_identity",
]

SCHEDULER_VERSION = "r55-b0"
MANIFEST_SCHEMA = "zemosaic-r55-manifest-v1"
# R55-D additive joint planner/evidence schema. The R55-B SCHEDULER_VERSION
# is deliberately NOT bumped: CPU-only manifests must stay byte-identical.
JOINT_SCHEMA = "zemosaic-r55d-joint-planner-v2"
JOINT_EVIDENCE_SCHEMA = "zemosaic-r55d-joint-evidence-v2"
JOINT_SCHEDULER_VERSION = "r55-d1"

MODE_LEGACY = "legacy"
MODE_OBSERVE = "observe"
MODE_ACTIVE = "active"
MODES = (MODE_LEGACY, MODE_OBSERVE, MODE_ACTIVE)
DEFAULT_MODE = MODE_LEGACY

# R55-D2 active CPU manifest/mapping schemas. These are ADDITIVE: the R55-B
# manifest schema/version is deliberately unchanged so legacy/observe bytes stay
# identical.
ACTIVE_MANIFEST_SCHEMA = "zemosaic-r55d2-active-manifest-v1"
ACTIVE_MAPPING_SCHEMA = "zemosaic-r55d2-active-mapping-v1"
# R55-D2 rework-1: the PRIMARY active plan hash identifies the FINAL
# EXECUTABLE decision (final candidate + exact mapping + frozen identity +
# validated product PlanEnvelope). It is deliberately schema-labelled so it is
# never mistaken for the D1 scheduler-envelope hash.
ACTIVE_FINAL_PLAN_SCHEMA = "zemosaic-r55d2-final-executable-plan-v1"
ACTIVE_EXECUTABLE_PHASES = (
    PHASE_CELL_STACK_INMEM,
    PHASE_CELL_STACK_STREAM,
)

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
        block = {
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
        return block


def normalize_scheduler_mode(value, *, default: str = DEFAULT_MODE) -> str:
    """Resolve ``zegrid_scheduler_mode``.

    ``None`` (absent) means ``legacy``. An explicit value must be the exact
    lowercase string ``legacy``, ``observe`` or ``active``. Anything else —
    including ``True``, ``"LEGACY"``, ``"auto"`` or ``""`` — raises
    :class:`InvalidSchedulerMode` so an unknown value can never be silently
    coerced to the default. R55-D2 adds ``active`` additively; the default and
    every observe/legacy semantic are unchanged.
    """
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, str):
        raise InvalidSchedulerMode(
            "zegrid_scheduler_mode must be a string "
            f"({MODE_LEGACY!r}, {MODE_OBSERVE!r} or {MODE_ACTIVE!r}), got "
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

    def _manifest_note(self) -> str:
        if self.scheduler_version == SCHEDULER_VERSION:
            return (
                "R55-B observe-only: these are PREDICTED integer envelopes. "
                "Execution remains the legacy path; no observed metric is recorded "
                "and no RUN decision was taken."
            )
        return (
            "R55-D joint observe-only: these are PREDICTED integer envelopes. "
            "GPU is modelled ONLY for the stack envelope; gauge/cache/assembly/"
            "export remain CPU. Execution remains the legacy path; no observed "
            "metric is recorded and no RUN decision was taken."
        )

    def to_manifest_dict(
        self,
        *,
        include_gpu_readiness: bool = False,
        gpu_readiness=None,
        requested_backend: str | None = None,
        effective_backend: str | None = None,
    ) -> dict:
        """Additive, deterministic ``r55_scheduler`` manifest block.

        No observed metric is fabricated: predicted envelopes only. The legacy
        ``ram_contract``/``cpu_streaming``/``cache``/``gpu`` blocks are untouched.

        ``include_gpu_readiness`` is DEFAULT-OFF so every R55-B caller keeps a
        byte-identical output. When explicitly enabled it appends the additive
        ``gpu_readiness`` observe-only block (R55-C); it never probes hardware
        and never changes execution authority.
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
        block = {
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
            "note": self._manifest_note(),
        }
        if self.scheduler_version != SCHEDULER_VERSION:
            block["joint_schema"] = JOINT_SCHEMA
        if include_gpu_readiness:
            from .gpu_readiness import readiness_manifest_dict

            block["gpu_readiness"] = readiness_manifest_dict(
                gpu_readiness,
                requested_backend=requested_backend,
                effective_backend=effective_backend,
            )
        return block


# ---------------------------------------------------------------------------
# Context validation and enumeration
# ---------------------------------------------------------------------------

def _validate_context(context: PlanContext, *, allow_gpu: bool = False) -> None:
    if not isinstance(context, PlanContext):
        raise SchedulerConfigError(
            f"expected PlanContext, got {type(context).__name__}"
        )
    if context.backend not in BACKENDS:
        raise SchedulerConfigError(f"unknown backend {context.backend!r}")
    if context.backend == BACKEND_GPU and not allow_gpu:
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
    if isinstance(context.layout, (str, bytes)) or not isinstance(
        context.layout, (tuple, list)
    ) or len(context.layout) != 2:
        raise SchedulerConfigError(f"layout must be two ints, got {context.layout!r}")
    lx = require_int(context.layout[0], "layout[0]", minimum=1)
    ly = require_int(context.layout[1], "layout[1]", minimum=1)
    require_int(context.refinement_rank, "refinement_rank")
    require_int(context.halo_px, "halo_px")
    require_int(context.channels, "channels", minimum=1)
    require_int(context.n_frames, "n_frames", minimum=1)
    if context.patch_hw is not None:
        if isinstance(context.patch_hw, (str, bytes)) or not isinstance(
            context.patch_hw, (tuple, list)
        ) or len(context.patch_hw) != 2:
            raise SchedulerConfigError(
                f"patch_hw must be two ints, got {context.patch_hw!r}"
            )
        require_int(context.patch_hw[0], "patch_hw[0]", minimum=1)
        require_int(context.patch_hw[1], "patch_hw[1]", minimum=1)
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
# R55-D — additive joint CPU/GPU planner core
# ---------------------------------------------------------------------------
#
# The joint planner is an ADDITIVE layer over R55-B. It never changes the CPU
# enumeration/winner/hash path; a CPU winner delegates back to build_plan_set.
# GPU facts are explicit integers, never probed: no CuPy/CUDA import, no driver
# query, no clock. A GPU candidate is a single owner (K == 1) and exists only
# when the caller explicitly requests an available, parity-qualified device
# with a complete per-tile device-peak model.

def _joint_bool(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise SchedulerConfigError(
            f"{name} must be a bool, got {type(value).__name__}={value!r}"
        )
    return bool(value)


def _joint_int(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchedulerConfigError(
            f"{name} must be an int, got {type(value).__name__}={value!r}"
        )
    if value < minimum:
        raise SchedulerConfigError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


@dataclass(frozen=True)
class GpuFacts:
    """Explicit, integer-only GPU facts supplied by the caller.

    ``device_peak_bytes_by_tile`` must cover EXACTLY the GPU context's declared
    tile ladder. An incomplete, extra, negative, bool, float or contradictory
    model is rejected, never guessed. ``prefetch_overlap_measured`` is the only
    way prefetch may receive speed credit on the GPU path.
    """

    requested: bool
    available: bool
    parity_qualified: bool
    vram_total_bytes: int
    vram_free_bytes: int
    vram_budget_bytes: int
    device_name: str
    device_peak_bytes_by_tile: tuple[tuple[int, int], ...] = ()
    prefetch_overlap_measured: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "requested", _joint_bool(self.requested, "gpu.requested")
        )
        object.__setattr__(
            self, "available", _joint_bool(self.available, "gpu.available")
        )
        object.__setattr__(
            self,
            "parity_qualified",
            _joint_bool(self.parity_qualified, "gpu.parity_qualified"),
        )
        total = _joint_int(self.vram_total_bytes, "gpu.vram_total_bytes")
        free = _joint_int(self.vram_free_bytes, "gpu.vram_free_bytes")
        budget = _joint_int(self.vram_budget_bytes, "gpu.vram_budget_bytes")
        if free > total:
            raise SchedulerConfigError(
                f"gpu.vram_free_bytes {free} > vram_total_bytes {total}"
            )
        if budget > free:
            raise SchedulerConfigError(
                f"gpu.vram_budget_bytes {budget} > vram_free_bytes {free}"
            )
        if not isinstance(self.device_name, str) or not self.device_name.strip():
            raise SchedulerConfigError("gpu.device_name must be a non-empty string")
        if self.device_name != self.device_name.strip():
            raise SchedulerConfigError(
                "gpu.device_name must not have leading/trailing whitespace"
            )
        raw = self.device_peak_bytes_by_tile
        if isinstance(raw, Mapping):
            items = tuple(raw.items())
        elif isinstance(raw, (tuple, list)):
            items = tuple(raw)
        else:
            raise SchedulerConfigError(
                "gpu.device_peak_bytes_by_tile must be a mapping or a sequence of "
                f"(tile_px, bytes) pairs, got {type(raw).__name__}"
            )
        normalized: list[tuple[int, int]] = []
        for index, item in enumerate(items):
            if isinstance(item, (str, bytes)) or not isinstance(item, (tuple, list)):
                raise SchedulerConfigError(
                    "gpu.device_peak_bytes_by_tile entries must be (tile_px, bytes) "
                    f"pairs, got {item!r} at index {index}"
                )
            if len(item) != 2:
                raise SchedulerConfigError(
                    f"gpu.device_peak_bytes_by_tile[{index}] must have 2 entries"
                )
            tile = _joint_int(
                item[0],
                f"gpu.device_peak_bytes_by_tile[{index}].tile_px",
                minimum=MIN_TILE_PX,
            )
            peak = _joint_int(
                item[1],
                f"gpu.device_peak_bytes_by_tile[{index}].bytes",
                minimum=0,
            )
            normalized.append((tile, peak))
        normalized.sort()
        tiles = [tile for tile, _peak in normalized]
        if len(set(tiles)) != len(tiles):
            raise SchedulerConfigError(
                f"gpu.device_peak_bytes_by_tile has duplicate tile(s): {tiles}"
            )
        object.__setattr__(
            self, "device_peak_bytes_by_tile", tuple(normalized)
        )
        object.__setattr__(
            self,
            "prefetch_overlap_measured",
            _joint_bool(
                self.prefetch_overlap_measured,
                "gpu.prefetch_overlap_measured",
            ),
        )

    @property
    def device_peak_map(self) -> dict[int, int]:
        return {
            int(tile): int(peak)
            for tile, peak in self.device_peak_bytes_by_tile
        }

    @property
    def vram_reserved_estimate_bytes(self) -> int:
        return max(
            0, int(self.vram_total_bytes) - int(self.vram_free_bytes)
        )

    def to_dict(self) -> dict:
        return {
            "requested": bool(self.requested),
            "available": bool(self.available),
            "parity_qualified": bool(self.parity_qualified),
            "vram_total_bytes": int(self.vram_total_bytes),
            "vram_free_bytes": int(self.vram_free_bytes),
            "vram_budget_bytes": int(self.vram_budget_bytes),
            "vram_reserved_estimate_bytes": int(
                self.vram_reserved_estimate_bytes
            ),
            "device_name": self.device_name,
            "device_peak_bytes_by_tile": [
                [int(tile), int(peak)]
                for tile, peak in self.device_peak_bytes_by_tile
            ],
            "prefetch_overlap_measured": bool(self.prefetch_overlap_measured),
        }


@dataclass(frozen=True)
class JointRefusal:
    """One deterministic backend/phase-level refusal/eligibility entry."""

    backend: str
    reason_code: str
    detail: str
    phase: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise SchedulerConfigError(
                f"unknown refusal backend {self.backend!r}"
            )
        if self.reason_code not in REASON_CODES:
            raise SchedulerConfigError(
                f"reason_code {self.reason_code!r} outside the closed set"
            )
        if not isinstance(self.detail, str) or not self.detail:
            raise SchedulerConfigError(
                "refusal detail must be a non-empty string"
            )
        if self.phase is not None and self.phase not in PHASES:
            raise SchedulerConfigError(
                f"refusal phase {self.phase!r} is outside the closed phase set"
            )

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "phase": self.phase,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class JointPlanInput:
    """Explicit immutable input to the R55-D joint planner.

    ``contexts`` may contain at most one CPU ``cell_stack_inmem``, at most one
    CPU ``cell_stack_stream`` and at most one GPU ``cell_stack_stream``; at
    least one CPU context is required. Contexts are normalised canonically
    independent of caller order. All contexts share the host budget/factors and
    the frozen scientific identity; the stack phase may differ between inmem and
    stream. A GPU context requires explicit facts. CPU prefetch-overlap is an
    explicit serializable fact and defaults to unmeasured.
    """

    contexts: tuple[PlanContext, ...]
    gpu_facts: GpuFacts | None = None
    cpu_prefetch_overlap_measured: bool = False

    def __post_init__(self) -> None:
        raw = self.contexts
        if isinstance(raw, (str, bytes)) or not isinstance(raw, (tuple, list)):
            raise SchedulerConfigError(
                "joint contexts must be a sequence of PlanContext values"
            )
        contexts = tuple(raw)
        if not contexts:
            raise SchedulerConfigError("joint contexts must not be empty")
        for context in contexts:
            if not isinstance(context, PlanContext):
                raise SchedulerConfigError(
                    "joint contexts entries must be PlanContext, got "
                    f"{type(context).__name__}"
                )
        seen: set[tuple[str, str]] = set()
        cpu_contexts: list[PlanContext] = []
        gpu_contexts: list[PlanContext] = []
        for context in contexts:
            if context.backend not in BACKENDS:
                raise SchedulerConfigError(
                    f"joint context has unknown backend {context.backend!r}"
                )
            key = (context.backend, context.phase)
            if key in seen:
                raise SchedulerConfigError(
                    "duplicate joint context for "
                    f"(backend={context.backend!r}, phase={context.phase!r})"
                )
            seen.add(key)
            if context.backend == BACKEND_CPU:
                cpu_contexts.append(context)
            else:
                gpu_contexts.append(context)
        if not cpu_contexts:
            raise SchedulerConfigError(
                "joint planner requires at least one CPU context"
            )
        if len(gpu_contexts) > 1:
            raise SchedulerConfigError(
                "joint planner accepts at most one GPU context, got "
                f"{len(gpu_contexts)}"
            )
        for context in cpu_contexts:
            _validate_context(context)
        if gpu_contexts:
            _validate_gpu_context(gpu_contexts[0])
        if gpu_contexts and self.gpu_facts is None:
            raise SchedulerConfigError(
                "a GPU PlanContext requires explicit gpu_facts (never probed)"
            )
        if not gpu_contexts and self.gpu_facts is not None:
            raise SchedulerConfigError(
                "gpu_facts supplied without a GPU PlanContext"
            )
        if self.gpu_facts is not None and not isinstance(
            self.gpu_facts, GpuFacts
        ):
            raise SchedulerConfigError(
                f"gpu_facts must be GpuFacts, got "
                f"{type(self.gpu_facts).__name__}"
            )
        measured = _joint_bool(
            self.cpu_prefetch_overlap_measured,
            "cpu_prefetch_overlap_measured",
        )
        object.__setattr__(
            self, "cpu_prefetch_overlap_measured", measured
        )
        normalized = tuple(
            sorted(cpu_contexts + gpu_contexts, key=_context_rank)
        )
        object.__setattr__(self, "contexts", normalized)
        reference = next(
            context
            for context in normalized
            if context.backend == BACKEND_CPU
        )
        for context in normalized:
            if context is reference:
                continue
            _validate_joint_consistency(reference, context)
        if gpu_contexts:
            assert self.gpu_facts is not None
            _validate_device_peak_coverage(gpu_contexts[0], self.gpu_facts)

    @property
    def cpu_contexts(self) -> tuple[PlanContext, ...]:
        return tuple(
            context
            for context in self.contexts
            if context.backend == BACKEND_CPU
        )

    @property
    def cpu_context(self) -> PlanContext:
        """Canonical first CPU context (inmem before stream)."""
        return self.cpu_contexts[0]

    @property
    def cpu_context_count(self) -> int:
        return len(self.cpu_contexts)

    @property
    def gpu_context(self) -> PlanContext | None:
        for context in self.contexts:
            if context.backend == BACKEND_GPU:
                return context
        return None

    def cpu_context_for_phase(self, phase: str) -> PlanContext | None:
        for context in self.cpu_contexts:
            if context.phase == phase:
                return context
        return None

    def stack_context_for(self, candidate: Candidate) -> PlanContext | None:
        if not isinstance(candidate, Candidate):
            raise SchedulerConfigError(
                "stack_context_for expects a Candidate"
            )
        if candidate.backend == BACKEND_GPU:
            return self.gpu_context
        if candidate.tile_px is None:
            return self.cpu_context_for_phase(PHASE_CELL_STACK_INMEM)
        return self.cpu_context_for_phase(PHASE_CELL_STACK_STREAM)

    def host_context_for(self, phase: str | None = None) -> PlanContext:
        if phase is not None:
            context = self.cpu_context_for_phase(phase)
            if context is not None:
                return context
        return self.cpu_context


def _context_rank(context: PlanContext) -> tuple[int, int]:
    backend_rank = 0 if context.backend == BACKEND_CPU else 1
    return (backend_rank, PHASES.index(context.phase))


def _validate_gpu_context(context: PlanContext) -> None:
    _validate_context(context, allow_gpu=True)
    if context.backend != BACKEND_GPU:
        raise SchedulerConfigError("expected a GPU PlanContext")
    if context.phase != PHASE_CELL_STACK_STREAM:
        raise SchedulerConfigError(
            "R55-D models a GPU joint stack only for cell_stack_stream"
        )


def _validate_joint_consistency(
    reference: PlanContext, other: PlanContext
) -> None:
    problems: list[str] = []
    for name in (
        "raw_available_bytes",
        "factor_num",
        "factor_den",
        "budget_bytes",
        "cpu_logical",
        "units_total",
        "channels",
        "n_frames",
        "halo_px",
        "refinement_rank",
        "run_id",
    ):
        if getattr(reference, name) != getattr(other, name):
            problems.append(f"host/science fact {name} differs")
    if (
        reference.margin_num,
        reference.margin_den,
    ) != (other.margin_num, other.margin_den):
        problems.append("margin differs between joint contexts")
    if tuple(reference.layout) != tuple(other.layout):
        problems.append("layout differs")
    if reference.patch_hw != other.patch_hw:
        problems.append("patch_hw differs")
    if reference.identity != other.identity:
        problems.append(
            "frozen identity (layout/cells/frames/halo/rank) differs"
        )
    if bool(reference.pinned) != bool(other.pinned):
        problems.append("pinned differs")
    if problems:
        raise SchedulerConfigError(
            "joint contexts must describe the same host and science identity: "
            + "; ".join(problems)
        )


def _validate_device_peak_coverage(
    gpu: PlanContext, facts: GpuFacts
) -> None:
    declared = {int(tile) for tile in gpu.tile_ladder}
    provided = {
        int(tile) for tile, _peak in facts.device_peak_bytes_by_tile
    }
    missing = sorted(declared - provided)
    extra = sorted(provided - declared)
    if missing:
        raise SchedulerConfigError(
            "gpu.device_peak_bytes_by_tile is incomplete; missing declared "
            f"tile(s) {missing}"
        )
    if extra:
        raise SchedulerConfigError(
            "gpu.device_peak_bytes_by_tile declares tile(s) outside the GPU "
            f"ladder: {extra}"
        )


def _gpu_blocking_refusal(joint: JointPlanInput) -> JointRefusal | None:
    gpu = joint.gpu_context
    facts = joint.gpu_facts
    if gpu is None:
        return JointRefusal(
            BACKEND_GPU, REASON_DEFAULT_OFF, "no GPU PlanContext supplied"
        )
    if facts is None:
        raise SchedulerConfigError("GPU context without gpu_facts")
    if not facts.requested:
        return JointRefusal(
            BACKEND_GPU,
            REASON_DEFAULT_OFF,
            "GPU was not explicitly requested",
        )
    if not facts.available:
        return JointRefusal(
            BACKEND_GPU, REASON_GPU_UNAVAILABLE, "GPU is not available"
        )
    if not facts.parity_qualified:
        return JointRefusal(
            BACKEND_GPU, REASON_PARITY_FAIL, "GPU is not parity-qualified"
        )
    if int(facts.vram_budget_bytes) <= 0:
        return JointRefusal(
            BACKEND_GPU,
            REASON_REFUSE_VRAM,
            "GPU device budget must be a positive explicit integer",
        )
    return None


def _enumerate_gpu_candidates(
    joint: JointPlanInput,
) -> tuple[Candidate, ...]:
    gpu = joint.gpu_context
    facts = joint.gpu_facts
    assert gpu is not None and facts is not None
    peak_map = facts.device_peak_map
    cpu_slots = max(0, int(gpu.cpu_logical) - CPU_OS_RESERVE)
    disk_cap = (
        int(gpu.disk_free_bytes) * int(gpu.disk_fraction_num)
    ) // int(gpu.disk_fraction_den)
    tiles = tuple(int(tile) for tile in gpu.tile_ladder)
    in_range_tiles = set(tiles)
    if gpu.profile is not None:
        lo, hi = gpu.profile.range_map["tile_px"]
        in_range_tiles = {tile for tile in tiles if lo <= tile <= hi}

    out: list[Candidate] = []
    for tile in tiles:
        in_range = tile in in_range_tiles
        per_unit, method, cal_id = _candidate_stack_bytes(
            gpu, tile, calibrated=in_range
        )
        # Single owner: the GPU stack always runs K == 1.
        k = 1
        for prefetch in gpu.prefetch_ladder:
            if k * (1 + int(prefetch)) > cpu_slots:
                continue
            w_max = max(1, int(gpu.cpu_logical) // k)
            if gpu.workers_cap is not None:
                w_max = min(w_max, int(gpu.workers_cap))
            for workers in gpu.workers_ladder:
                if int(workers) > w_max:
                    continue
                host_peak = (
                    int(gpu.shared_bytes)
                    + k * per_unit
                    + int(prefetch) * int(gpu.per_unit_prefetch_bytes)
                )
                disk = disk_residency_bytes(
                    units_in_flight=k,
                    prefetch_units=int(prefetch),
                    per_unit_cache_bytes=int(gpu.per_unit_cache_bytes),
                    per_unit_prefetch_bytes=int(
                        gpu.per_unit_prefetch_bytes
                    ),
                )
                device_peak = int(peak_map[tile])

                if not in_range:
                    feasible = False
                    reason = REASON_UNCALIBRATED
                    detail = (
                        f"tile {tile} is outside the GPU profile's declared "
                        "valid_ranges[tile_px]="
                        f"{gpu.profile.range_map['tile_px']}; refusing to "
                        "extrapolate"
                    )
                elif disk > disk_cap:
                    feasible = False
                    reason = REASON_REFUSE_DISK
                    detail = (
                        f"disk residency {disk} B > budget {disk_cap} B "
                        f"(free {gpu.disk_free_bytes} x "
                        f"{gpu.disk_fraction_num}/{gpu.disk_fraction_den})"
                    )
                elif ceil_margin_div(
                    host_peak, gpu.margin_num, gpu.margin_den
                ) > int(gpu.budget_bytes):
                    feasible = False
                    reason = REASON_REFUSE_HOST_RAM
                    detail = (
                        f"host peak {host_peak} B with margin "
                        f"{gpu.margin_num}/{gpu.margin_den} exceeds budget "
                        f"{gpu.budget_bytes} B"
                    )
                elif device_peak > int(facts.vram_budget_bytes):
                    feasible = False
                    reason = REASON_REFUSE_VRAM
                    detail = (
                        f"device peak {device_peak} B > explicit device "
                        f"budget {facts.vram_budget_bytes} B"
                    )
                else:
                    feasible = True
                    reason = (
                        REASON_OK
                        if method == "calibrated"
                        else REASON_UNCALIBRATED
                    )
                    detail = None

                makespan = None
                if (
                    feasible
                    and method == "calibrated"
                    and gpu.time_model is not None
                ):
                    # No measured overlap => prefetch receives NO speed credit.
                    model_prefetch = (
                        int(prefetch)
                        if facts.prefetch_overlap_measured
                        else 0
                    )
                    wave = int(gpu.units_total)  # K == 1
                    per_unit_ns = int(
                        gpu.time_model(tile, k, model_prefetch)
                    )
                    if per_unit_ns < 0:
                        raise SchedulerConfigError(
                            "time_model returned a negative duration"
                        )
                    makespan = wave * per_unit_ns

                out.append(
                    Candidate(
                        backend=BACKEND_GPU,
                        tile_px=int(tile),
                        units_in_flight=k,
                        prefetch_units=int(prefetch),
                        workers_per_unit=int(workers),
                        host_peak_bytes=int(host_peak),
                        disk_bytes=int(disk),
                        device_peak_bytes=int(device_peak),
                        feasible=bool(feasible),
                        reason_code=reason,
                        infeasible_reason=detail,
                        predicted_makespan_ns=makespan,
                        estimate_method=method,
                        calibration_id=cal_id,
                    )
                )
    out.sort(key=lambda candidate: candidate.ordering_key())
    return tuple(out)


def _enumerate_cpu_joint(
    context: PlanContext, *, measured: bool
) -> tuple[Candidate, ...]:
    """R55-B CPU candidates with joint-only prefetch-credit normalization.

    The legacy :func:`enumerate_candidates` API is untouched. In the additive
    joint layer, an unmeasured CPU prefetch depth may remain feasible but must
    receive the SAME time-model result as prefetch=0. Only an explicit measured
    overlap fact may pass the real prefetch depth to the CPU time model.
    """
    candidates = enumerate_candidates(context)
    if measured or context.time_model is None:
        return tuple(candidates)
    out: list[Candidate] = []
    for candidate in candidates:
        if (
            candidate.prefetch_units <= 0
            or candidate.predicted_makespan_ns is None
        ):
            out.append(candidate)
            continue
        wave = (
            int(context.units_total) + int(candidate.units_in_flight) - 1
        ) // int(candidate.units_in_flight)
        per_unit_ns = int(
            context.time_model(
                candidate.tile_px,
                int(candidate.units_in_flight),
                0,
            )
        )
        if per_unit_ns < 0:
            raise SchedulerConfigError(
                "time_model returned a negative duration"
            )
        out.append(
            replace(
                candidate,
                predicted_makespan_ns=wave * per_unit_ns,
            )
        )
    out.sort(key=lambda candidate: candidate.ordering_key())
    return tuple(out)


def enumerate_joint_candidates(
    joint: JointPlanInput,
) -> tuple[Candidate, ...]:
    """Exhaustive canonical CPU-first joint enumeration.

    The union covers CPU in-memory (tile=None), CPU streaming (integer product
    tiles) and GPU streaming (K=1). GPU candidates are added only when the
    caller explicitly supplies a requested, available, parity-qualified context
    with valid complete device facts. The result is independent of
    backend-context order, mapping order and cell order.
    """
    if not isinstance(joint, JointPlanInput):
        raise SchedulerConfigError(
            f"expected JointPlanInput, got {type(joint).__name__}"
        )
    cpu_candidates: list[Candidate] = []
    for context in joint.cpu_contexts:
        cpu_candidates.extend(
            _enumerate_cpu_joint(
                context,
                measured=joint.cpu_prefetch_overlap_measured,
            )
        )
    gpu_candidates: tuple[Candidate, ...] = ()
    if joint.gpu_context is not None and _gpu_blocking_refusal(joint) is None:
        gpu_candidates = _enumerate_gpu_candidates(joint)
    out = tuple(
        sorted(
            tuple(cpu_candidates) + tuple(gpu_candidates),
            key=lambda candidate: candidate.ordering_key(),
        )
    )
    keys = [candidate.ordering_key() for candidate in out]
    if len(set(keys)) != len(keys):
        raise SchedulerConfigError(
            "joint enumeration produced duplicate ordering keys"
        )
    return out


def joint_refusal_trail(joint: JointPlanInput) -> tuple[JointRefusal, ...]:
    """Deterministic backend-level eligibility/refusal trail."""
    if not isinstance(joint, JointPlanInput):
        raise SchedulerConfigError(
            f"expected JointPlanInput, got {type(joint).__name__}"
        )
    entries = [
        JointRefusal(
            BACKEND_CPU,
            REASON_OK,
            f"cpu context accepted ({context.phase})",
            phase=context.phase,
        )
        for context in joint.cpu_contexts
    ]
    if joint.gpu_context is None:
        entries.append(
            JointRefusal(
                BACKEND_GPU,
                REASON_DEFAULT_OFF,
                "no GPU PlanContext supplied",
            )
        )
    else:
        blocked = _gpu_blocking_refusal(joint)
        if blocked is not None:
            entries.append(
                replace(blocked, phase=joint.gpu_context.phase)
            )
        else:
            entries.append(
                JointRefusal(
                    BACKEND_GPU,
                    REASON_OK,
                    "gpu context eligible",
                    phase=joint.gpu_context.phase,
                )
            )
    return tuple(entries)


def _joint_no_winner_reason(
    joint: JointPlanInput, candidates: Sequence[Candidate]
) -> tuple[str, str]:
    if not candidates:
        for entry in joint_refusal_trail(joint):
            if entry.reason_code != REASON_OK:
                return entry.reason_code, entry.detail
        return (
            REASON_REFUSE_HOST_RAM,
            "joint enumeration produced no candidate",
        )
    reasons = {candidate.reason_code for candidate in candidates}
    if reasons == {REASON_REFUSE_DISK}:
        return (
            REASON_REFUSE_DISK,
            "no joint candidate fits the hard constraints (disk shortfall: "
            "persistent aligned cache does not shrink with the stack tile)",
        )
    if reasons == {REASON_REFUSE_VRAM} or (
        REASON_REFUSE_VRAM in reasons
        and reasons <= {REASON_REFUSE_VRAM, REASON_UNCALIBRATED}
    ):
        return (
            REASON_REFUSE_VRAM,
            "no joint candidate fits the explicit device budget",
        )
    return (
        REASON_REFUSE_HOST_RAM,
        "no joint candidate fits the hard host/disk/device constraints",
    )


def select_joint_candidate(
    joint: JointPlanInput,
    *,
    candidates: Sequence[Candidate] | None = None,
) -> Candidate:
    """Select the objective winner among feasible joint candidates.

    This is the objective-only winner. :func:`plan_joint` additionally builds and
    validates the PlanSet, so an aggregate-level refusal can still move to the
    next candidate.
    """
    if candidates is None:
        candidates = enumerate_joint_candidates(joint)
    else:
        candidates = tuple(candidates)
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        reason, detail = _joint_no_winner_reason(joint, candidates)
        raise PlanRefused(reason, detail, candidates=candidates)
    return min(feasible, key=objective_key)


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
        # D1: the active stack phase is simultaneous with assembly (retained
        # cores + halo). Exactly one stack phase exists, so the pair is declared
        # symmetrically with the assembly PhaseRequest and never both stacks.
        overlaps_with=(PHASE_ASSEMBLY,),
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
    # D1: assembly must declare the ACTIVE stack phase as a symmetric overlap.
    # The stack envelope declares assembly in turn; validate_envelope_set then
    # proves the pair. Only one stack phase exists in a plan.
    assembly_req = next(r for r in requests if r.phase == PHASE_ASSEMBLY)
    if tuple(assembly_req.overlaps_with) != (context.phase,):
        raise SchedulerConfigError(
            "the assembly PhaseRequest must declare overlaps_with="
            f"({context.phase!r},) so simultaneous retained cores + active stack "
            f"are summed, got {tuple(assembly_req.overlaps_with)!r}"
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


def _joint_phase_requests(
    requests: Sequence[PhaseRequest], phase: str
) -> tuple[PhaseRequest, ...]:
    """Rewrite the assembly overlap to the CHOSEN stack phase.

    Multi-context joint plans cannot know the winner when the caller builds the
    requests. The joint builder derives the truthful assembly overlap from the
    chosen context. A single-context CPU call whose assembly request already
    names its stack phase is returned unchanged, preserving R55-B bytes.
    """
    out: list[PhaseRequest] = []
    for request in requests:
        if not isinstance(request, PhaseRequest):
            raise SchedulerConfigError(
                "phase_requests entries must be PhaseRequest"
            )
        if request.phase == PHASE_ASSEMBLY:
            for other in request.overlaps_with:
                if other not in CELL_STACK_PHASES:
                    raise SchedulerConfigError(
                        "joint assembly overlaps_with may only name "
                        "cell_stack_* phases or be empty, got "
                        f"{request.overlaps_with!r}"
                    )
            if tuple(request.overlaps_with) != (phase,):
                request = replace(request, overlaps_with=(phase,))
        out.append(request)
    return tuple(out)


def _gpu_stack_envelope(
    joint: JointPlanInput, candidate: Candidate
) -> ResourceEnvelope:
    """The GPU stack envelope: GPU backend, device block, one owner."""
    gpu = joint.gpu_context
    facts = joint.gpu_facts
    assert gpu is not None and facts is not None
    per_unit, _method, _cal_id = _candidate_stack_bytes(
        gpu,
        candidate.tile_px,
        calibrated=(candidate.estimate_method == "calibrated"),
    )
    components = {
        "prefetch_residency_bytes": (
            int(candidate.prefetch_units) * int(gpu.per_unit_prefetch_bytes)
        ),
        "shared_bytes": int(gpu.shared_bytes),
        "stack_in_flight_bytes": (
            int(candidate.units_in_flight) * int(per_unit)
        ),
    }
    estimate = make_estimate(
        components,
        method=candidate.estimate_method,
        calibration_id=candidate.calibration_id,
    )
    return ResourceEnvelope(
        envelope_id=f"{gpu.phase}.{BACKEND_GPU}",
        phase=gpu.phase,
        backend=BACKEND_GPU,
        host=_host_envelope(gpu),
        device=DeviceEnvelope(
            vram_total_bytes=int(facts.vram_total_bytes),
            vram_free_bytes=int(facts.vram_free_bytes),
            vram_reserved_estimate_bytes=int(
                facts.vram_reserved_estimate_bytes
            ),
            budget_bytes=int(facts.vram_budget_bytes),
            device_name=facts.device_name,
        ),
        geometry=_geometry(gpu, tile_px=candidate.tile_px),
        concurrency=ConcurrencyEnvelope(
            units=int(gpu.units_total),
            units_in_flight=int(candidate.units_in_flight),
            workers_per_unit=int(candidate.workers_per_unit),
            prefetch_units=int(candidate.prefetch_units),
        ),
        estimate=estimate,
        bound=_bound(gpu, candidate.estimate_method),
        accounting=AccountingEnvelope(
            aggregate_rule=AGGREGATE_RULE_MAX_OVER_PHASES,
            host_peak_bytes=int(estimate.peak_bytes),
            device_peak_bytes=int(candidate.device_peak_bytes),
            disk_bytes=int(candidate.disk_bytes),
            transfer_bytes=0,
        ),
        overlaps_with=(PHASE_ASSEMBLY,),
        identity=IdentityEnvelope(
            run_id=gpu.run_id, plan_hash="", pinned=bool(gpu.pinned)
        ),
        reason_code=candidate.reason_code,
    )


def build_joint_plan_set(
    joint: JointPlanInput,
    *,
    candidate: Candidate,
    phase_requests: Sequence[PhaseRequest],
    mode: str = MODE_OBSERVE,
) -> PlanSet:
    """Build a truthful mixed-backend PlanSet for one joint candidate.

    A CPU winner delegates to the accepted R55-B :func:`build_plan_set`, so its
    bytes are untouched. A GPU winner puts GPU only on the stack envelope;
    gauge/cache/assembly/export remain CPU and carry no device block. The GPU
    stack carries a valid DeviceEnvelope and device accounting. Overlap stays
    explicit and symmetric.
    """
    if not isinstance(joint, JointPlanInput):
        raise SchedulerConfigError(
            f"expected JointPlanInput, got {type(joint).__name__}"
        )
    if mode not in MODES:
        raise InvalidSchedulerMode(f"unknown plan mode {mode!r}")
    if not isinstance(candidate, Candidate):
        raise SchedulerConfigError("candidate must be a Candidate")
    if not candidate.feasible:
        raise PlanRefused(
            candidate.reason_code,
            "cannot build a joint plan from an infeasible candidate: "
            f"{candidate.infeasible_reason}",
            candidates=[candidate],
        )
    enumerated = enumerate_joint_candidates(joint)
    match = next(
        (
            item
            for item in enumerated
            if item.ordering_key() == candidate.ordering_key()
        ),
        None,
    )
    if match is None or match != candidate:
        raise SchedulerConfigError(
            "candidate does not match the joint enumeration "
            "(stale or tampered plan)"
        )
    stack_context = joint.stack_context_for(match)
    if stack_context is None:
        raise SchedulerConfigError(
            "no joint context corresponds to the chosen stack candidate"
        )
    if candidate.backend == BACKEND_CPU:
        plan_set = build_plan_set(
            stack_context,
            candidate=match,
            phase_requests=_joint_phase_requests(
                phase_requests, stack_context.phase
            ),
            mode=mode,
        )
        if len(joint.contexts) > 1:
            # Multi-context joint calls are schema-versioned additive behavior.
            # A single-context CPU call is returned byte-identically.
            decisions = tuple(
                {
                    **dict(decision),
                    "at": "plan_joint",
                    "joint_context_count": len(joint.contexts),
                }
                for decision in plan_set.decisions
            )
            plan_set = replace(
                plan_set,
                decisions=decisions,
                scheduler_version=JOINT_SCHEDULER_VERSION,
            )
        return plan_set

    gpu = joint.gpu_context
    facts = joint.gpu_facts
    if gpu is None or facts is None:
        raise SchedulerConfigError(
            "GPU candidate without a GPU context/facts"
        )
    if int(candidate.units_in_flight) != 1:
        raise SchedulerConfigError(
            "GPU candidate must be a single owner (K == 1)"
        )
    if candidate.tile_px is None:
        raise SchedulerConfigError(
            "GPU candidate must declare a tile_px"
        )
    expected_peak = facts.device_peak_map.get(int(candidate.tile_px))
    if expected_peak is None:
        raise SchedulerConfigError(
            f"no device peak for GPU tile {candidate.tile_px}"
        )
    if int(candidate.device_peak_bytes) != int(expected_peak):
        raise SchedulerConfigError(
            "candidate device peak does not match the explicit device model"
        )

    host_context = joint.host_context_for(gpu.phase)
    requests = _joint_phase_requests(phase_requests, gpu.phase)
    phases_present = {request.phase for request in requests}
    missing = [p for p in REQUIRED_PLAN_PHASES if p not in phases_present]
    if missing:
        raise SchedulerConfigError(
            f"phase_requests is missing required PLAN phase(s): {missing}"
        )
    if gpu.phase in phases_present:
        raise SchedulerConfigError(
            f"phase_requests must not include the stack phase {gpu.phase!r}: "
            "it is derived from the chosen candidate"
        )
    stacks = [p for p in phases_present if p in CELL_STACK_PHASES]
    if stacks:
        raise SchedulerConfigError(
            f"phase_requests must contain no cell_stack_* phase, got {stacks}"
        )
    assembly_req = next(
        request for request in requests if request.phase == PHASE_ASSEMBLY
    )
    if tuple(assembly_req.overlaps_with) != (gpu.phase,):
        raise SchedulerConfigError(
            "the assembly PhaseRequest must declare overlaps_with="
            f"({gpu.phase!r},) so simultaneous retained cores + active stack "
            f"are summed, got {tuple(assembly_req.overlaps_with)!r}"
        )

    envelopes = [
        _request_envelope(host_context, request)
        for request in requests
    ]
    envelopes.append(_gpu_stack_envelope(joint, match))
    validated = validate_envelope_set(envelopes, headless=True)
    raw_peak, peak = aggregate_host_peak_bytes(validated)
    if peak > int(host_context.budget_bytes):
        raise PlanRefused(
            REASON_REFUSE_HOST_RAM,
            f"aggregate host peak {peak} B (raw {raw_peak} B + margin "
            f"{host_context.margin_num}/"
            f"{host_context.margin_den}) > budget "
            f"{host_context.budget_bytes} B",
            candidates=[match],
        )
    device_raw, _device_margin_applied = aggregate_device_peak_bytes(
        validated
    )
    # The EXPLICIT device budget is authoritative and already carries its own
    # reserve; the host safety margin is never applied a second time to VRAM.
    if device_raw > int(facts.vram_budget_bytes):
        raise PlanRefused(
            REASON_REFUSE_VRAM,
            f"aggregate device peak {device_raw} B > explicit device budget "
            f"{facts.vram_budget_bytes} B",
            candidates=[match],
        )

    plan_hash = compute_plan_hash(validated)
    stamped = tuple(
        ResourceEnvelope(
            envelope_id=env.envelope_id,
            phase=env.phase,
            backend=env.backend,
            host=env.host,
            device=env.device,
            geometry=env.geometry,
            concurrency=env.concurrency,
            estimate=env.estimate,
            bound=env.bound,
            accounting=env.accounting,
            overlaps_with=env.overlaps_with,
            identity=IdentityEnvelope(
                run_id=env.identity.run_id,
                plan_hash=plan_hash,
                pinned=env.identity.pinned,
            ),
            reason_code=env.reason_code,
        )
        for env in validated
    )
    device_raw_stamped, _device_margin_stamped = (
        aggregate_device_peak_bytes(stamped)
    )
    chosen_env = next(env for env in stamped if env.phase == gpu.phase)
    decision = {
        "at": "plan_joint",
        "chosen": chosen_env.envelope_id,
        "stack_phase": gpu.phase,
        "host_phase": host_context.phase,
        "reason_code": match.reason_code,
        "candidate": {
            "backend": match.backend,
            "tile_px": match.tile_px,
            "units_in_flight": match.units_in_flight,
            "prefetch_units": match.prefetch_units,
            "workers_per_unit": match.workers_per_unit,
        },
        "host_peak_bytes": int(match.host_peak_bytes),
        "device_peak_bytes": int(match.device_peak_bytes),
        "disk_bytes": int(match.disk_bytes),
        "budget_bytes": int(host_context.budget_bytes),
        "device_budget_bytes": int(facts.vram_budget_bytes),
        "aggregate_host_peak_bytes": int(peak),
        "aggregate_device_peak_bytes": int(device_raw_stamped),
        "plan_hash": plan_hash,
    }
    if match.reason_code not in REASON_CODES:
        raise SchedulerConfigError(
            f"reason_code {match.reason_code!r} outside the closed set"
        )
    return PlanSet(
        mode=mode,
        budget_bytes=int(host_context.budget_bytes),
        raw_available_bytes=int(host_context.raw_available_bytes),
        aggregate_host_peak_bytes=int(peak),
        aggregate_host_peak_bytes_raw=int(raw_peak),
        aggregate_device_peak_bytes=int(device_raw_stamped),
        aggregate_disk_bytes=int(aggregate_disk_bytes(stamped)),
        chosen=match,
        phases=stamped,
        decisions=(decision,),
        plan_hash=plan_hash,
        identity=host_context.identity,
        run_id=host_context.run_id,
        scheduler_version=JOINT_SCHEDULER_VERSION,
    )


def plan_joint(
    joint: JointPlanInput,
    *,
    phase_requests: Sequence[PhaseRequest],
    mode: str = MODE_OBSERVE,
) -> PlanSet:
    """Choose and build the deterministic joint winner."""
    candidates = enumerate_joint_candidates(joint)
    feasible = sorted(
        (candidate for candidate in candidates if candidate.feasible),
        key=objective_key,
    )
    if not feasible:
        reason, detail = _joint_no_winner_reason(joint, candidates)
        raise PlanRefused(reason, detail, candidates=candidates)
    last: PlanRefused | None = None
    for candidate in feasible:
        try:
            return build_joint_plan_set(
                joint,
                candidate=candidate,
                phase_requests=phase_requests,
                mode=mode,
            )
        except PlanRefused as exc:
            last = exc
            continue
    assert last is not None
    raise last


def _strict_int(value, name: str, *, minimum: int = 0) -> int:
    """Strict adapter-boundary integer check (no bool/float/string coercion)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchedulerConfigError(
            f"{name} must be an int, got {type(value).__name__}={value!r}"
        )
    if value < minimum:
        raise SchedulerConfigError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _strict_pair(value, name: str, *, positive: bool = False) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise SchedulerConfigError(
            f"{name} must be a 2-sequence of ints, got {type(value).__name__}={value!r}"
        )
    if len(value) != 2:
        raise SchedulerConfigError(f"{name} must have exactly 2 entries, got {value!r}")
    pair = (
        _strict_int(value[0], f"{name}[0]", minimum=(1 if positive else 0)),
        _strict_int(value[1], f"{name}[1]", minimum=(1 if positive else 0)),
    )
    return pair


def _strict_seq(value, name: str, *, minimum: int = 0) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise SchedulerConfigError(
            f"{name} must be a sequence of ints, got {type(value).__name__}={value!r}"
        )
    return tuple(
        _strict_int(item, f"{name}[{i}]", minimum=minimum)
        for i, item in enumerate(value)
    )


def _strict_str_seq(value, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise SchedulerConfigError(
            f"{name} must be a sequence of strings, got {type(value).__name__}={value!r}"
        )
    out = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise SchedulerConfigError(
                f"{name}[{i}] must be a non-empty string, got {item!r}"
            )
        out.append(item)
    return tuple(out)


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
    # Strict integer boundary: every numeric public argument is type-proved
    # BEFORE any conversion. ``True``/``64.0``/``'64'`` must never silently
    # become an int budget or tile.
    strict_budget = _strict_int(budget_bytes, "budget_bytes")
    strict_raw = _strict_int(raw_available_bytes, "raw_available_bytes", minimum=1)
    strict_fnum = _strict_int(factor_num, "factor_num", minimum=1)
    strict_fden = _strict_int(factor_den, "factor_den", minimum=1)
    strict_cpu = _strict_int(cpu_logical, "cpu_logical", minimum=1)
    strict_shared = _strict_int(shared_bytes, "shared_bytes")
    strict_stack = _strict_int(per_unit_stack_bytes, "per_unit_stack_bytes", minimum=1)
    strict_cache = _strict_int(per_unit_cache_bytes, "per_unit_cache_bytes")
    strict_prefetch = _strict_int(per_unit_prefetch_bytes, "per_unit_prefetch_bytes")
    strict_disk = _strict_int(disk_free_bytes, "disk_free_bytes")
    strict_layout = _strict_pair(layout, "layout", positive=True)
    strict_rank = _strict_int(refinement_rank, "refinement_rank")
    strict_patch = (
        None if patch_hw is None else _strict_pair(patch_hw, "patch_hw", positive=True)
    )
    strict_halo = _strict_int(halo_px, "halo_px")
    strict_channels = _strict_int(channels, "channels", minimum=1)
    strict_frames = _strict_int(n_frames, "n_frames", minimum=1)
    strict_dfn = _strict_int(disk_fraction_num, "disk_fraction_num", minimum=1)
    strict_dfd = _strict_int(disk_fraction_den, "disk_fraction_den", minimum=1)
    strict_mnum = _strict_int(margin_num, "margin_num")
    strict_mden = _strict_int(margin_den, "margin_den", minimum=1)
    strict_tiles = _strict_seq(tile_ladder, "tile_ladder", minimum=1)
    strict_prefetches = _strict_seq(prefetch_ladder, "prefetch_ladder", minimum=0)
    strict_workers = _strict_seq(workers_ladder, "workers_ladder", minimum=1)
    strict_cap = (
        None if workers_cap is None else _strict_int(workers_cap, "workers_cap", minimum=1)
    )
    cells = _strict_str_seq(cell_ids, "cell_ids")
    frames = _strict_str_seq(frame_ids, "frame_ids")
    if not isinstance(observation_source, str) or not observation_source:
        raise SchedulerConfigError("observation_source must be a non-empty string")
    if not isinstance(run_id, str):
        raise SchedulerConfigError(f"run_id must be a string, got {run_id!r}")
    if not isinstance(pinned, bool):
        raise SchedulerConfigError(f"pinned must be a bool, got {pinned!r}")
    if units_total is None:
        if not cells:
            raise SchedulerConfigError(
                "units_total is required when cell_ids is not supplied"
            )
        strict_units = len(cells)
    else:
        strict_units = _strict_int(units_total, "units_total", minimum=1)
    context = PlanContext(
        phase=phase,
        backend=backend,
        budget_bytes=strict_budget,
        raw_available_bytes=strict_raw,
        factor_num=strict_fnum,
        factor_den=strict_fden,
        cpu_logical=strict_cpu,
        units_total=strict_units,
        shared_bytes=strict_shared,
        per_unit_stack_bytes=strict_stack,
        per_unit_cache_bytes=strict_cache,
        per_unit_prefetch_bytes=strict_prefetch,
        disk_free_bytes=strict_disk,
        layout=strict_layout,
        refinement_rank=strict_rank,
        patch_hw=strict_patch,
        halo_px=strict_halo,
        channels=strict_channels,
        n_frames=strict_frames,
        disk_fraction_num=strict_dfn,
        disk_fraction_den=strict_dfd,
        tile_ladder=strict_tiles,
        prefetch_ladder=strict_prefetches,
        workers_ladder=strict_workers,
        workers_cap=strict_cap,
        margin_num=strict_mnum,
        margin_den=strict_mden,
        profile=profile,
        time_model=time_model,
        cell_ids=cells,
        frame_ids=frames,
        observation_source=observation_source,
        run_id=run_id,
        pinned=pinned,
    )
    return plan(context, phase_requests=phase_requests, mode=mode)


# ---------------------------------------------------------------------------
# R55-D — deterministic joint serialization/hash/evidence
# ---------------------------------------------------------------------------

def _canonical_json_bytes(obj) -> bytes:
    """Deterministic evidence bytes: sorted keys, fixed indent, ASCII, LF."""
    return (
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")


def _joint_context_payload(context: PlanContext) -> dict:
    return {
        "phase": context.phase,
        "backend": context.backend,
        "budget_bytes": int(context.budget_bytes),
        "raw_available_bytes": int(context.raw_available_bytes),
        "factor_num": int(context.factor_num),
        "factor_den": int(context.factor_den),
        "cpu_logical": int(context.cpu_logical),
        "units_total": int(context.units_total),
        "shared_bytes": int(context.shared_bytes),
        "per_unit_stack_bytes": int(context.per_unit_stack_bytes),
        "per_unit_cache_bytes": int(context.per_unit_cache_bytes),
        "per_unit_prefetch_bytes": int(context.per_unit_prefetch_bytes),
        "disk_free_bytes": int(context.disk_free_bytes),
        "disk_fraction_num": int(context.disk_fraction_num),
        "disk_fraction_den": int(context.disk_fraction_den),
        "layout": [int(context.layout[0]), int(context.layout[1])],
        "refinement_rank": int(context.refinement_rank),
        "patch_hw": (
            None
            if context.patch_hw is None
            else [int(context.patch_hw[0]), int(context.patch_hw[1])]
        ),
        "halo_px": int(context.halo_px),
        "channels": int(context.channels),
        "n_frames": int(context.n_frames),
        "tile_ladder": [int(tile) for tile in context.tile_ladder],
        "prefetch_ladder": [
            int(value) for value in context.prefetch_ladder
        ],
        "workers_ladder": [int(value) for value in context.workers_ladder],
        "workers_cap": (
            None
            if context.workers_cap is None
            else int(context.workers_cap)
        ),
        "margin_num": int(context.margin_num),
        "margin_den": int(context.margin_den),
        "profile": (
            None
            if context.profile is None
            else {
                "calibration_id": context.profile.calibration_id,
                "backend": context.profile.backend,
                "phase": context.profile.phase,
                "margin_num": int(context.profile.margin_num),
                "margin_den": int(context.profile.margin_den),
                "path": context.profile.path,
                "product_admissible": bool(
                    context.profile.product_admissible
                ),
                "coefficients": {
                    name: int(value)
                    for name, value in context.profile.coefficients
                },
                "valid_ranges": {
                    name: [int(lo), int(hi)]
                    for name, (lo, hi) in context.profile.valid_ranges
                },
            }
        ),
        "time_model_present": bool(context.time_model is not None),
        "cell_ids": sorted(str(cell) for cell in context.cell_ids),
        "frame_ids": sorted(str(frame) for frame in context.frame_ids),
        "observation_source": context.observation_source,
        "run_id": context.run_id,
        "pinned": bool(context.pinned),
    }


def joint_input_payload(joint: JointPlanInput) -> dict:
    """Canonical, serializer-safe description of the joint input."""
    if not isinstance(joint, JointPlanInput):
        raise SchedulerConfigError(
            f"expected JointPlanInput, got {type(joint).__name__}"
        )
    return {
        "joint_schema": JOINT_SCHEMA,
        "scheduler_version": JOINT_SCHEDULER_VERSION,
        "contexts": [
            _joint_context_payload(context) for context in joint.contexts
        ],
        "cpu_context_count": len(joint.cpu_contexts),
        "cpu_prefetch_overlap_measured": bool(
            joint.cpu_prefetch_overlap_measured
        ),
        "gpu_facts": (
            None if joint.gpu_facts is None else joint.gpu_facts.to_dict()
        ),
    }


def joint_enumeration_payload(joint: JointPlanInput) -> dict:
    """Deterministic exhaustive enumeration/refusal payload."""
    candidates = enumerate_joint_candidates(joint)
    refusals = joint_refusal_trail(joint)
    return {
        "schema": JOINT_SCHEMA,
        "scheduler_version": JOINT_SCHEDULER_VERSION,
        "input": joint_input_payload(joint),
        "cpu_context_count": len(joint.cpu_contexts),
        "refusals": [entry.to_dict() for entry in refusals],
        "candidate_count": len(candidates),
        "feasible_count": sum(
            1 for candidate in candidates if candidate.feasible
        ),
        "gpu_candidate_count": sum(
            1
            for candidate in candidates
            if candidate.backend == BACKEND_GPU
        ),
        "candidates": [
            candidate.to_dict() for candidate in candidates
        ],
    }


def joint_enumeration_hash(joint: JointPlanInput) -> str:
    """SHA-256 over the canonical enumeration payload."""
    payload = joint_enumeration_payload(joint)
    return "sha256:" + hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()


def joint_evidence_payload(
    joint: JointPlanInput,
    *,
    phase_requests: Sequence[PhaseRequest] | None = None,
    mode: str = MODE_OBSERVE,
) -> dict:
    """Deterministic joint evidence block.

    With ``phase_requests`` it also records the chosen plan (or its closed
    refusal). Without them it records enumeration + refusal trail only. No
    timestamp, absolute path, PID or observed metric is embedded.
    """
    enumeration = joint_enumeration_payload(joint)
    payload = {
        "schema": JOINT_EVIDENCE_SCHEMA,
        "mission": "ZM-ZEGRID-R55-D-JOINT-PLANNER-20261023",
        "phase": "D1-rework-1",
        "joint_schema": JOINT_SCHEMA,
        "scheduler_version": JOINT_SCHEDULER_VERSION,
        "input": enumeration["input"],
        "cpu_context_count": enumeration["cpu_context_count"],
        "refusals": enumeration["refusals"],
        "candidate_count": enumeration["candidate_count"],
        "feasible_count": enumeration["feasible_count"],
        "gpu_candidate_count": enumeration["gpu_candidate_count"],
        "candidates": enumeration["candidates"],
        "enumeration_hash": joint_enumeration_hash(joint),
        "winner": None,
        "plan": None,
        "refusal": None,
    }
    if phase_requests is not None:
        try:
            plan_set = plan_joint(
                joint, phase_requests=phase_requests, mode=mode
            )
        except PlanRefused as exc:
            payload["refusal"] = exc.refusal_manifest_dict(
                mode=mode, run_id=joint.cpu_context.run_id
            )
        else:
            payload["winner"] = plan_set.chosen.to_dict()
            payload["plan"] = {
                "plan_hash": plan_set.plan_hash,
                "scheduler_version": plan_set.scheduler_version,
                "chosen": plan_set.chosen.to_dict(),
                "aggregate_host_peak_bytes": int(
                    plan_set.aggregate_host_peak_bytes
                ),
                "aggregate_host_peak_bytes_raw": int(
                    plan_set.aggregate_host_peak_bytes_raw
                ),
                "aggregate_device_peak_bytes": int(
                    plan_set.aggregate_device_peak_bytes
                ),
                "aggregate_disk_bytes": int(
                    plan_set.aggregate_disk_bytes
                ),
                "phases": [
                    env.to_dict() for env in plan_set.phases
                ],
                "decisions": [
                    dict(decision) for decision in plan_set.decisions
                ],
            }
    return payload


def joint_evidence_json(
    joint: JointPlanInput,
    *,
    phase_requests: Sequence[PhaseRequest] | None = None,
    mode: str = MODE_OBSERVE,
) -> str:
    """Canonical evidence JSON (stable across processes/input order)."""
    return _canonical_json_bytes(
        joint_evidence_payload(
            joint, phase_requests=phase_requests, mode=mode
        )
    ).decode("ascii")


def joint_evidence_hash(
    joint: JointPlanInput,
    *,
    phase_requests: Sequence[PhaseRequest] | None = None,
    mode: str = MODE_OBSERVE,
) -> str:
    """SHA-256 over the canonical evidence JSON bytes."""
    return "sha256:" + hashlib.sha256(
        joint_evidence_json(
            joint, phase_requests=phase_requests, mode=mode
        ).encode("ascii")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Shrink-only API (contract only; R55-B never mutates a RUN)
# ---------------------------------------------------------------------------

class ActivePlanRejected(SchedulerError):
    """An ``active`` plan cannot be executed by the product's CPU knobs.

    R55-D2 grants execution authority ONLY to a deterministic, fully mappable
    CPU winner. Every unsupported backend/phase/tile/concurrency/worker or
    prefetch-positive winner is rejected here (fail closed) instead of being
    silently coerced to the legacy plan.
    """

    def __init__(
        self,
        reason_code: str,
        detail: str,
        *,
        candidate: Candidate | None = None,
        mapping: Mapping | None = None,
    ):
        super().__init__(f"{reason_code}: {detail}")
        if reason_code not in REASON_CODES:
            raise SchedulerConfigError(
                f"reason_code {reason_code!r} outside the closed set"
            )
        self.reason_code = str(reason_code)
        self.detail = str(detail)
        self.candidate = candidate
        self.mapping = None if mapping is None else dict(mapping)

    def refusal_manifest_dict(
        self, *, mode: str = MODE_ACTIVE, run_id: str = ""
    ) -> dict:
        block = {
            "schema": ACTIVE_MANIFEST_SCHEMA,
            "scheduler_version": SCHEDULER_VERSION,
            "mode": "active",
            "requested_mode": mode,
            "run_id": str(run_id),
            "status": "refused",
            "refused": True,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "candidate": (
                None if self.candidate is None else self.candidate.to_dict()
            ),
            "mapping": self.mapping,
            "observed_metrics": None,
        }
        return _stamp_active_hash(block)


def active_cpu_phase(candidate: Candidate) -> str:
    """Return the executable CPU stack phase for ``candidate`` (or reject).

    ``tile_px is None`` is EXACTLY the in-memory stack; any tile is the
    streaming stack. A GPU or otherwise unsupported candidate is refused.
    """
    if not isinstance(candidate, Candidate):
        raise SchedulerConfigError(
            f"expected Candidate, got {type(candidate).__name__}"
        )
    if candidate.backend != BACKEND_CPU:
        raise ActivePlanRejected(
            REASON_UNKNOWN_BACKEND,
            "R55-D2 active grants execution authority to the CPU backend only; "
            f"winner backend={candidate.backend!r}",
            candidate=candidate,
        )
    return (
        PHASE_CELL_STACK_INMEM
        if candidate.tile_px is None
        else PHASE_CELL_STACK_STREAM
    )


def active_mapping_for_candidate(
    candidate: Candidate,
    *,
    tile_ladder: Sequence[int] = PRODUCT_TILE_LADDER,
    workers_ladder: Sequence[int] = WORKERS_LADDER,
) -> dict:
    """Map ONE candidate onto the existing CPU execution knobs (fail closed).

    This is the single mapping implementation shared by the initial active
    winner and every scheduler-produced shrink step, so a pressure correction
    can never drift from the accepted mapping contract.
    """
    if not isinstance(candidate, Candidate):
        raise SchedulerConfigError(
            f"expected Candidate, got {type(candidate).__name__}"
        )
    phase = active_cpu_phase(candidate)
    if int(candidate.prefetch_units) != 0:
        raise ActivePlanRejected(
            REASON_SHRINK_PREFETCH,
            "R55-D2 active refuses a prefetch-positive winner "
            f"(prefetch_units={candidate.prefetch_units}); R55-E owns prefetch "
            "execution",
            candidate=candidate,
        )
    if phase == PHASE_CELL_STACK_STREAM:
        allowed_tiles = tuple(int(t) for t in tile_ladder)
        if candidate.tile_px is None or int(candidate.tile_px) not in allowed_tiles:
            raise ActivePlanRejected(
                REASON_SHRINK_TILE,
                "stream winner tile "
                f"{candidate.tile_px!r} is outside the product ladder "
                f"{allowed_tiles}",
                candidate=candidate,
            )
        mapped_tile = int(candidate.tile_px)
    else:
        mapped_tile = None
    units = int(candidate.units_in_flight)
    if units < 1:
        raise ActivePlanRejected(
            REASON_REFUSE_HOST_RAM,
            f"units_in_flight must be >= 1, got {candidate.units_in_flight}",
            candidate=candidate,
        )
    allowed_workers = tuple(int(w) for w in workers_ladder)
    workers = int(candidate.workers_per_unit)
    if workers < 1 or (allowed_workers and workers not in allowed_workers):
        raise ActivePlanRejected(
            REASON_SHRINK_WORKERS,
            "workers_per_unit "
            f"{candidate.workers_per_unit!r} is not an existing executable "
            f"cache-worker value {allowed_workers}",
            candidate=candidate,
        )
    return {
        "schema": ACTIVE_MAPPING_SCHEMA,
        "backend": BACKEND_CPU,
        "phase": phase,
        "cell_task_mode": (
            "inmem" if phase == PHASE_CELL_STACK_INMEM else "stream"
        ),
        "cpu_streaming_tile": mapped_tile,
        "units_in_flight": units,
        "workers_per_unit": workers,
        "prefetch_units": 0,
    }


def map_active_cpu_candidate(
    plan_set: PlanSet,
    *,
    tile_ladder: Sequence[int] = PRODUCT_TILE_LADDER,
    workers_ladder: Sequence[int] = WORKERS_LADDER,
) -> dict:
    """Map an ``active`` CPU winner EXACTLY onto the product's existing knobs.

    Contract (all fail closed):

    * the PlanSet was produced in ``active`` mode and carries a stable hash;
    * backend is ``cpu``;
    * phase is ``cell_stack_inmem`` (``tile_px is None``) or
      ``cell_stack_stream`` (tile on the canonical PRODUCT ladder);
    * ``prefetch_units == 0`` (R55-E owns prefetch execution);
    * ``units_in_flight`` maps to the cell-stack pool and ``workers_per_unit``
      to the existing per-cell cache-worker knob. No new executor/pool exists.

    The returned mapping is the object recorded in the active manifest; the
    caller must apply exactly these values (no second unreported plan).
    """
    if not isinstance(plan_set, PlanSet):
        raise SchedulerConfigError(
            f"expected PlanSet, got {type(plan_set).__name__}"
        )
    if plan_set.mode != MODE_ACTIVE:
        raise ActivePlanRejected(
            REASON_UNKNOWN_BACKEND,
            f"active mapping requires mode={MODE_ACTIVE!r}, got {plan_set.mode!r}",
            candidate=plan_set.chosen,
        )
    if not isinstance(plan_set.plan_hash, str) or not plan_set.plan_hash:
        raise ActivePlanRejected(
            REASON_INTERNAL_ERROR,
            "active plan has no stable plan hash",
            candidate=plan_set.chosen,
        )
    return active_mapping_for_candidate(
        plan_set.chosen,
        tile_ladder=tile_ladder,
        workers_ladder=workers_ladder,
    )


def active_final_plan_hash(
    *,
    candidate: Candidate,
    mapping: Mapping,
    identity: FrozenIdentity,
    envelope,
) -> str:
    """Deterministic hash of the FINAL executable active decision.

    The payload is schema-labelled and covers the final candidate, its exact
    mapping, the frozen identity and the final validated product PlanEnvelope.
    It is NOT the D1 scheduler PlanSet hash; that hash is preserved separately as
    ``initial_plan_hash`` when a pressure correction changed the plan.
    """
    if not isinstance(candidate, Candidate):
        raise SchedulerConfigError(
            f"expected Candidate, got {type(candidate).__name__}"
        )
    if not isinstance(identity, FrozenIdentity):
        raise SchedulerConfigError(
            f"expected FrozenIdentity, got {type(identity).__name__}"
        )
    expected = active_mapping_for_candidate(candidate)
    if dict(mapping) != expected:
        raise ActivePlanRejected(
            REASON_INTERNAL_ERROR,
            "final plan hash mapping does not exactly match the final candidate "
            f"(mapping={mapping!r}, expected={expected!r})",
            candidate=candidate,
            mapping=mapping,
        )
    if hasattr(envelope, "to_dict"):
        env = envelope.to_dict()
    elif isinstance(envelope, Mapping):
        env = dict(envelope)
    else:
        raise SchedulerConfigError(
            "final plan hash requires a PlanEnvelope or its mapping"
        )
    payload = {
        "schema": ACTIVE_FINAL_PLAN_SCHEMA,
        "candidate": candidate.to_dict(),
        "mapping": dict(mapping),
        "identity": identity.to_dict(),
        "product_envelope": env,
    }
    return "sha256:" + hashlib.sha256(
        _canonical_active_json(payload)
    ).hexdigest()


def _canonical_active_json(block: Mapping) -> bytes:
    """Canonical ASCII JSON for the active manifest hash."""
    return json.dumps(
        block,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _stamp_active_hash(block: Mapping) -> dict:
    """Return ``dict(block)`` plus a deterministic ``evidence_hash``."""
    payload = {k: v for k, v in dict(block).items() if k != "evidence_hash"}
    stamped = dict(payload)
    stamped["evidence_hash"] = "sha256:" + hashlib.sha256(
        _canonical_active_json(payload)
    ).hexdigest()
    return stamped


def active_manifest_dict(
    plan_set: PlanSet,
    *,
    mapping: Mapping,
    status: str,
    execution: Mapping | None = None,
    plan_observation: Mapping | None = None,
    initial_plan_observation: Mapping | None = None,
    admission: Mapping | None = None,
    preflight_admission: Mapping | None = None,
    launch_revalidation: Mapping | None = None,
    shrink_trail: Sequence[Mapping] = (),
    refusal: Mapping | None = None,
    note: str = "",
    final_candidate: Candidate | None = None,
    initial_candidate: Candidate | None = None,
    initial_plan_hash: str | None = None,
    final_envelope=None,
    plan_hash: str | None = None,
    plan_hash_schema: str | None = None,
) -> dict:
    """Build the deterministic R55-D2 active manifest block.

    The block describes the plan that ACTUALLY governs execution: the chosen
    candidate, its exact knob mapping, the PLAN/ADMIT observations and the
    shrink-only correction trail. It never fabricates an observed metric.
    ``evidence_hash`` is SHA-256 over the canonical block bytes without the
    hash itself, so two identical plans produce identical evidence.
    """
    if not isinstance(plan_set, PlanSet):
        raise SchedulerConfigError(
            f"expected PlanSet, got {type(plan_set).__name__}"
        )
    if not isinstance(mapping, Mapping):
        raise SchedulerConfigError("active manifest mapping must be a mapping")
    if not isinstance(status, str) or not status:
        raise SchedulerConfigError("active manifest status must be non-empty")
    final_candidate = (
        plan_set.chosen if final_candidate is None else final_candidate
    )
    initial_candidate = (
        plan_set.chosen if initial_candidate is None else initial_candidate
    )
    if not isinstance(final_candidate, Candidate) or not isinstance(
        initial_candidate, Candidate
    ):
        raise SchedulerConfigError(
            "active manifest candidates must be Candidate values"
        )
    if initial_plan_hash is None:
        initial_plan_hash = plan_set.plan_hash
    if plan_hash is None:
        if final_envelope is not None:
            plan_hash = active_final_plan_hash(
                candidate=final_candidate,
                mapping=mapping,
                identity=plan_set.identity,
                envelope=final_envelope,
            )
        else:
            plan_hash = plan_set.plan_hash
    if plan_hash_schema is None:
        plan_hash_schema = (
            ACTIVE_FINAL_PLAN_SCHEMA
            if final_envelope is not None
            else "scheduler_plan_set:" + str(plan_set.scheduler_version)
        )
    block = {
        "schema": ACTIVE_MANIFEST_SCHEMA,
        "scheduler_version": str(plan_set.scheduler_version),
        "mode": "active",
        "requested_mode": MODE_ACTIVE,
        "status": str(status),
        "run_id": str(plan_set.run_id),
        "plan_hash": str(plan_hash),
        "plan_hash_schema": str(plan_hash_schema),
        "initial_plan_hash": str(initial_plan_hash),
        "initial_candidate": initial_candidate.to_dict(),
        "chosen_candidate": final_candidate.to_dict(),
        "mapping": dict(mapping),
        "execution": None if execution is None else dict(execution),
        "final_envelope": (
            final_envelope.to_dict()
            if hasattr(final_envelope, "to_dict")
            else (None if final_envelope is None else dict(final_envelope))
        ),
        "plan_observation": (
            None if plan_observation is None else dict(plan_observation)
        ),
        "initial_plan_observation": (
            None if initial_plan_observation is None
            else dict(initial_plan_observation)
        ),
        "admission": None if admission is None else dict(admission),
        "preflight_admission": (
            None if preflight_admission is None else dict(preflight_admission)
        ),
        "launch_revalidation": (
            None if launch_revalidation is None else dict(launch_revalidation)
        ),
        "shrink_trail": [dict(item) for item in shrink_trail],
        "refusal": None if refusal is None else dict(refusal),
        "identity": plan_set.identity.to_dict(),
        "observed_metrics": None,
        "note": str(note),
    }
    return _stamp_active_hash(block)


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


def assert_joint_shrink_only(previous: Candidate, candidate: Candidate) -> None:
    """Joint shrink guard: no CPU->GPU growth and no byte growth.

    The backend/tile/K/prefetch/worker contract is R55-B's
    :func:`assert_shrink_only`; this additive guard also refuses any host,
    device or disk byte growth. GPU->CPU is the only backend shrink and is
    already enforced there. Frozen layout/cells/frames/halo are protected by
    :func:`assert_same_identity`.
    """
    assert_shrink_only(previous, candidate)
    problems: list[str] = []
    if int(candidate.host_peak_bytes) > int(previous.host_peak_bytes):
        problems.append(
            "host_peak_bytes increased "
            f"{previous.host_peak_bytes} -> {candidate.host_peak_bytes}"
        )
    if int(candidate.device_peak_bytes) > int(previous.device_peak_bytes):
        problems.append(
            "device_peak_bytes increased "
            f"{previous.device_peak_bytes} -> {candidate.device_peak_bytes}"
        )
    if int(candidate.disk_bytes) > int(previous.disk_bytes):
        problems.append(
            "disk_bytes increased "
            f"{previous.disk_bytes} -> {candidate.disk_bytes}"
        )
    if problems:
        raise GrowthRejected("; ".join(problems))
