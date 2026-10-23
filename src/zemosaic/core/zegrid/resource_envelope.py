"""ZM-ZEGRID-R55-B — per-phase, per-backend integer resource envelopes.

This module is the PURE data-contract layer of the R55 adaptive resource
scheduler. It deliberately imports no product mode, no psutil, no GPU module and
no scientific code: it is unit-testable, deterministic and reusable.

Binding rules implemented here (R55-A acceptance + R55-B mission):

* one :class:`ResourceEnvelope` per (phase, backend); a plan is a SET of
  envelopes, never one global record;
* bytes and budgets are integers only — ``bool`` is refused because it is an
  ``int`` subclass that would silently mean 1 byte;
* ``estimate.components`` is mandatory and must sum EXACTLY to the subtotal and
  peak. The safety margin is NOT baked into a phase peak: it is applied exactly
  once at the aggregate boundary (see :func:`aggregate_host_peak_bytes`);
* ``overlaps_with`` is mandatory, symmetric and closed over the known phase
  vocabulary. Undeclared or asymmetric overlap fails closed;
* aggregation is ``max`` over mutually exclusive overlap groups and an explicit
  ``sum`` inside each declared overlap group;
* ``plan_hash`` is a canonical SHA-256 over immutable decision fields only, so it
  is stable across process restarts and dictionary insertion order.

Nothing here selects a plan or executes work.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

__all__ = [
    "SCHEMA",
    "PHASES",
    "CELL_STACK_PHASES",
    "REQUIRED_PLAN_PHASES",
    "BACKENDS",
    "BACKEND_CPU",
    "BACKEND_GPU",
    "AGGREGATE_RULES",
    "AGGREGATE_RULE_MAX_OVER_PHASES",
    "ESTIMATE_METHODS",
    "BOUND_KINDS",
    "DEFAULT_MARGIN_NUM",
    "DEFAULT_MARGIN_DEN",
    "REASON_CODES",
    "EnvelopeError",
    "InvalidEnvelope",
    "InvalidOverlap",
    "HostEnvelope",
    "DeviceEnvelope",
    "GeometryEnvelope",
    "ConcurrencyEnvelope",
    "EstimateEnvelope",
    "BoundEnvelope",
    "AccountingEnvelope",
    "IdentityEnvelope",
    "ResourceEnvelope",
    "require_int",
    "ceil_margin_div",
    "validate_envelope",
    "validate_envelope_set",
    "overlap_groups",
    "aggregate_host_peak_bytes",
    "aggregate_device_peak_bytes",
    "aggregate_disk_bytes",
    "compute_plan_hash",
    "make_estimate",
]

SCHEMA = "zemosaic-r55-resource-envelope-v1"

PHASE_GAUGE = "gauge"
PHASE_CACHE_BUILD = "cache_build"
PHASE_CELL_STACK_INMEM = "cell_stack_inmem"
PHASE_CELL_STACK_STREAM = "cell_stack_stream"
PHASE_ASSEMBLY = "assembly"
PHASE_EXPORT = "export"
PHASE_GUI_PREVIEW = "gui_preview"

# Closed phase vocabulary (schema, examples and validation all use this order).
PHASES = (
    PHASE_GAUGE,
    PHASE_CACHE_BUILD,
    PHASE_CELL_STACK_INMEM,
    PHASE_CELL_STACK_STREAM,
    PHASE_ASSEMBLY,
    PHASE_EXPORT,
    PHASE_GUI_PREVIEW,
)
CELL_STACK_PHASES = (PHASE_CELL_STACK_INMEM, PHASE_CELL_STACK_STREAM)
# Required at PLAN; exactly ONE cell_stack_* is additionally required.
REQUIRED_PLAN_PHASES = (
    PHASE_GAUGE,
    PHASE_CACHE_BUILD,
    PHASE_ASSEMBLY,
    PHASE_EXPORT,
)

BACKEND_CPU = "cpu"
BACKEND_GPU = "gpu"
BACKENDS = (BACKEND_CPU, BACKEND_GPU)

AGGREGATE_RULE_MAX_OVER_PHASES = "max_over_phases"
AGGREGATE_RULES = (AGGREGATE_RULE_MAX_OVER_PHASES,)

ESTIMATE_METHODS = ("calibrated", "static_bound")
BOUND_KINDS = ("upper",)

DEFAULT_MARGIN_NUM = 15
DEFAULT_MARGIN_DEN = 100

# Closed reason-code set (stable strings; no ad-hoc codes may be emitted).
REASON_OK = "R55_OK"
REASON_UNCALIBRATED = "R55_UNCALIBRATED"
REASON_SHRINK_PREFETCH = "R55_SHRINK_PREFETCH"
REASON_SHRINK_WORKERS = "R55_SHRINK_WORKERS"
REASON_SHRINK_CONCURRENCY = "R55_SHRINK_CONCURRENCY"
REASON_SHRINK_TILE = "R55_SHRINK_TILE"
REASON_GPU_FALLBACK_PRE = "R55_GPU_FALLBACK_PRE"
REASON_GPU_FALLBACK_RUNTIME = "R55_GPU_FALLBACK_RUNTIME"
REASON_REFUSE_HOST_RAM = "R55_REFUSE_HOST_RAM"
REASON_REFUSE_VRAM = "R55_REFUSE_VRAM"
REASON_REFUSE_DISK = "R55_REFUSE_DISK"
REASON_PARITY_FAIL = "R55_PARITY_FAIL"
REASON_PINNED_GEOMETRY = "R55_PINNED_GEOMETRY"
REASON_UNKNOWN_BACKEND = "R55_UNKNOWN_BACKEND"
REASON_DEFAULT_OFF = "R55_DEFAULT_OFF"
REASON_INTERNAL_ERROR = "R55_INTERNAL_ERROR"

REASON_CODES = frozenset(
    {
        REASON_OK,
        REASON_UNCALIBRATED,
        REASON_SHRINK_PREFETCH,
        REASON_SHRINK_WORKERS,
        REASON_SHRINK_CONCURRENCY,
        REASON_SHRINK_TILE,
        REASON_GPU_FALLBACK_PRE,
        REASON_GPU_FALLBACK_RUNTIME,
        REASON_REFUSE_HOST_RAM,
        REASON_REFUSE_VRAM,
        REASON_REFUSE_DISK,
        REASON_PARITY_FAIL,
        REASON_PINNED_GEOMETRY,
        REASON_UNKNOWN_BACKEND,
        REASON_DEFAULT_OFF,
        REASON_INTERNAL_ERROR,
    }
)


# ---------------------------------------------------------------------------
# Fail-closed validation helpers
# ---------------------------------------------------------------------------

class EnvelopeError(ValueError):
    """Base class for envelope-contract violations."""


class InvalidEnvelope(EnvelopeError):
    """An envelope is structurally invalid or lies about its aggregate."""


class InvalidOverlap(EnvelopeError):
    """The overlap declaration is missing, asymmetric or unknown."""


def require_int(value, name: str, *, minimum: int = 0) -> int:
    """Return ``value`` as an int, refusing bool / non-int / out-of-range.

    ``bool`` is refused even though it subclasses ``int``: ``True`` silently
    meaning one byte is exactly the optimistic input this contract forbids.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidEnvelope(
            f"{name} must be an int, got {type(value).__name__}={value!r}"
        )
    if value < minimum:
        raise InvalidEnvelope(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def ceil_margin_div(value: int, num: int, den: int) -> int:
    """``ceil(value * (den + num) / den)`` with integer arithmetic only.

    This is the ONE place the safety margin is applied. Callers must not apply
    it again to a phase peak (no double reserve).
    """
    v = require_int(value, "value")
    n = require_int(num, "margin_num")
    d = require_int(den, "margin_den", minimum=1)
    if n > d:
        raise InvalidEnvelope(f"margin_num {n} > margin_den {d}")
    return (v * (d + n) + d - 1) // d


def _require_nonempty_str(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidEnvelope(f"{name} must be a non-empty string, got {value!r}")
    if value != value.strip():
        raise InvalidEnvelope(f"{name} must not have leading/trailing whitespace: {value!r}")
    return value


def _require_pair(value, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise InvalidEnvelope(f"{name} must be exactly two ints, got {value!r}")
    return (
        require_int(value[0], f"{name}[0]"),
        require_int(value[1], f"{name}[1]"),
    )


# ---------------------------------------------------------------------------
# Envelope records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HostEnvelope:
    observation_source: str
    raw_available_bytes: int
    factor_num: int
    factor_den: int
    budget_bytes: int
    reserve_bytes: int = 0
    process_baseline_bytes: int = 0
    observed_at_monotonic_s: float | None = None

    def to_dict(self) -> dict:
        return {
            "observation_source": self.observation_source,
            "raw_available_bytes": int(self.raw_available_bytes),
            "factor_num": int(self.factor_num),
            "factor_den": int(self.factor_den),
            "budget_bytes": int(self.budget_bytes),
            "reserve_bytes": int(self.reserve_bytes),
            "process_baseline_bytes": int(self.process_baseline_bytes),
            "observed_at_monotonic_s": self.observed_at_monotonic_s,
        }

    def validate(self) -> None:
        _require_nonempty_str(self.observation_source, "host.observation_source")
        raw = require_int(self.raw_available_bytes, "host.raw_available_bytes", minimum=1)
        num = require_int(self.factor_num, "host.factor_num", minimum=1)
        den = require_int(self.factor_den, "host.factor_den", minimum=1)
        budget = require_int(self.budget_bytes, "host.budget_bytes")
        reserve = require_int(self.reserve_bytes, "host.reserve_bytes")
        baseline = require_int(self.process_baseline_bytes, "host.process_baseline_bytes")
        expected = (num * raw) // den
        if budget != expected:
            raise InvalidEnvelope(
                f"host.budget_bytes {budget} != (factor_num*raw)//factor_den "
                f"{expected} (factor {num}/{den}, raw {raw})"
            )
        if reserve > budget:
            raise InvalidEnvelope(
                f"host.reserve_bytes {reserve} > host.budget_bytes {budget}"
            )
        if baseline > raw:
            raise InvalidEnvelope(
                f"host.process_baseline_bytes {baseline} > raw available {raw}"
            )
        if self.observed_at_monotonic_s is not None:
            v = self.observed_at_monotonic_s
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                raise InvalidEnvelope(
                    f"host.observed_at_monotonic_s must be a non-negative number, got {v!r}"
                )


@dataclass(frozen=True)
class DeviceEnvelope:
    vram_total_bytes: int
    vram_free_bytes: int
    vram_reserved_estimate_bytes: int
    budget_bytes: int
    device_name: str

    def to_dict(self) -> dict:
        return {
            "vram_total_bytes": int(self.vram_total_bytes),
            "vram_free_bytes": int(self.vram_free_bytes),
            "vram_reserved_estimate_bytes": int(self.vram_reserved_estimate_bytes),
            "budget_bytes": int(self.budget_bytes),
            "device_name": self.device_name,
        }

    def validate(self) -> None:
        total = require_int(self.vram_total_bytes, "device.vram_total_bytes")
        free = require_int(self.vram_free_bytes, "device.vram_free_bytes")
        reserved = require_int(
            self.vram_reserved_estimate_bytes,
            "device.vram_reserved_estimate_bytes",
        )
        budget = require_int(self.budget_bytes, "device.budget_bytes")
        _require_nonempty_str(self.device_name, "device.device_name")
        if free > total:
            raise InvalidEnvelope(
                f"device.vram_free_bytes {free} > vram_total_bytes {total}"
            )
        if budget > free:
            raise InvalidEnvelope(
                f"device.budget_bytes {budget} > vram_free_bytes {free}"
            )
        if reserved > total:
            raise InvalidEnvelope(
                f"device.vram_reserved_estimate_bytes {reserved} > total {total}"
            )


@dataclass(frozen=True)
class GeometryEnvelope:
    layout: tuple[int, int]
    refinement_rank: int
    cell_id: str | None
    patch_hw: tuple[int, int] | None
    tile_px: int | None
    halo_px: int
    channels: int

    def to_dict(self) -> dict:
        return {
            "layout": [int(self.layout[0]), int(self.layout[1])],
            "refinement_rank": int(self.refinement_rank),
            "cell_id": self.cell_id,
            "patch_hw": (
                None if self.patch_hw is None
                else [int(self.patch_hw[0]), int(self.patch_hw[1])]
            ),
            "tile_px": (None if self.tile_px is None else int(self.tile_px)),
            "halo_px": int(self.halo_px),
            "channels": int(self.channels),
        }

    def validate(self) -> None:
        lx, ly = _require_pair(self.layout, "geometry.layout")
        if lx <= 0 or ly <= 0:
            raise InvalidEnvelope(
                f"geometry.layout must be two POSITIVE dims, got {self.layout!r}"
            )
        require_int(self.refinement_rank, "geometry.refinement_rank")
        if self.cell_id is not None:
            _require_nonempty_str(self.cell_id, "geometry.cell_id")
        if self.patch_hw is not None:
            ph, pw = _require_pair(self.patch_hw, "geometry.patch_hw")
            if ph <= 0 or pw <= 0:
                raise InvalidEnvelope(
                    f"geometry.patch_hw must be two POSITIVE dims, got {self.patch_hw!r}"
                )
        if self.tile_px is not None:
            require_int(self.tile_px, "geometry.tile_px", minimum=1)
        require_int(self.halo_px, "geometry.halo_px")
        require_int(self.channels, "geometry.channels", minimum=1)


@dataclass(frozen=True)
class ConcurrencyEnvelope:
    units: int
    units_in_flight: int
    workers_per_unit: int
    prefetch_units: int

    def to_dict(self) -> dict:
        return {
            "units": int(self.units),
            "units_in_flight": int(self.units_in_flight),
            "workers_per_unit": int(self.workers_per_unit),
            "prefetch_units": int(self.prefetch_units),
        }

    def validate(self) -> None:
        units = require_int(self.units, "concurrency.units")
        k = require_int(self.units_in_flight, "concurrency.units_in_flight")
        w = require_int(self.workers_per_unit, "concurrency.workers_per_unit")
        p = require_int(self.prefetch_units, "concurrency.prefetch_units")
        if units == 0:
            if k or w or p:
                raise InvalidEnvelope(
                    "empty phase must declare zero units_in_flight/workers/prefetch"
                )
            return
        if k < 1:
            raise InvalidEnvelope("concurrency.units_in_flight must be >= 1")
        if k > units:
            raise InvalidEnvelope(
                f"concurrency.units_in_flight {k} > units {units}"
            )
        if w < 1:
            raise InvalidEnvelope("concurrency.workers_per_unit must be >= 1")


@dataclass(frozen=True)
class EstimateEnvelope:
    method: str
    calibration_id: str | None
    components: tuple[tuple[str, int], ...]
    subtotal_bytes: int
    peak_bytes: int

    def components_dict(self) -> dict:
        return {name: int(value) for name, value in self.components}

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "calibration_id": self.calibration_id,
            "components": {name: int(value) for name, value in self.components},
            "subtotal_bytes": int(self.subtotal_bytes),
            "peak_bytes": int(self.peak_bytes),
        }

    def validate(self) -> None:
        if self.method not in ESTIMATE_METHODS:
            raise InvalidEnvelope(
                f"estimate.method {self.method!r} not in {ESTIMATE_METHODS}"
            )
        if self.method == "calibrated":
            _require_nonempty_str(self.calibration_id, "estimate.calibration_id")
        elif self.calibration_id is not None:
            _require_nonempty_str(self.calibration_id, "estimate.calibration_id")
        if not isinstance(self.components, tuple) or not self.components:
            raise InvalidEnvelope("estimate.components must be a non-empty tuple")
        names = []
        total = 0
        for name, value in self.components:
            _require_nonempty_str(name, "estimate.components name")
            names.append(name)
            total += require_int(value, f"estimate.components[{name}]")
        if names != sorted(names):
            raise InvalidEnvelope(
                f"estimate.components must be stored in lexical order, got {names}"
            )
        if len(set(names)) != len(names):
            raise InvalidEnvelope(f"duplicate estimate component names: {names}")
        subtotal = require_int(self.subtotal_bytes, "estimate.subtotal_bytes")
        peak = require_int(self.peak_bytes, "estimate.peak_bytes")
        if subtotal != total:
            raise InvalidEnvelope(
                f"estimate.subtotal_bytes {subtotal} != sum(components) {total}"
            )
        if peak != subtotal:
            raise InvalidEnvelope(
                "estimate.peak_bytes must equal subtotal_bytes: the safety margin "
                "is applied ONCE at the aggregate boundary, never inside a phase "
                f"(peak {peak} != subtotal {subtotal})"
            )


@dataclass(frozen=True)
class BoundEnvelope:
    kind: str = "upper"
    margin_num: int = DEFAULT_MARGIN_NUM
    margin_den: int = DEFAULT_MARGIN_DEN
    confidence: str = "static_bound"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "margin_num": int(self.margin_num),
            "margin_den": int(self.margin_den),
            "confidence": self.confidence,
        }

    def validate(self) -> None:
        if self.kind not in BOUND_KINDS:
            raise InvalidEnvelope(f"bound.kind {self.kind!r} not in {BOUND_KINDS}")
        num = require_int(self.margin_num, "bound.margin_num")
        den = require_int(self.margin_den, "bound.margin_den", minimum=1)
        if num > den:
            raise InvalidEnvelope(f"bound.margin_num {num} > margin_den {den}")
        _require_nonempty_str(self.confidence, "bound.confidence")


@dataclass(frozen=True)
class AccountingEnvelope:
    aggregate_rule: str
    host_peak_bytes: int
    device_peak_bytes: int
    disk_bytes: int
    transfer_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "aggregate_rule": self.aggregate_rule,
            "host_peak_bytes": int(self.host_peak_bytes),
            "device_peak_bytes": int(self.device_peak_bytes),
            "disk_bytes": int(self.disk_bytes),
            "transfer_bytes": int(self.transfer_bytes),
        }

    def validate(self) -> None:
        if self.aggregate_rule not in AGGREGATE_RULES:
            raise InvalidEnvelope(
                f"accounting.aggregate_rule {self.aggregate_rule!r} not in "
                f"{AGGREGATE_RULES}"
            )
        require_int(self.host_peak_bytes, "accounting.host_peak_bytes")
        require_int(self.device_peak_bytes, "accounting.device_peak_bytes")
        require_int(self.disk_bytes, "accounting.disk_bytes")
        require_int(self.transfer_bytes, "accounting.transfer_bytes")


@dataclass(frozen=True)
class IdentityEnvelope:
    run_id: str = ""
    plan_hash: str = ""
    pinned: bool = False

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "plan_hash": self.plan_hash,
            "pinned": bool(self.pinned),
        }

    def validate(self) -> None:
        if not isinstance(self.run_id, str):
            raise InvalidEnvelope("identity.run_id must be a str")
        if not isinstance(self.pinned, bool):
            raise InvalidEnvelope("identity.pinned must be a bool")
        if self.plan_hash:
            _require_plan_hash(self.plan_hash)


def _require_plan_hash(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise InvalidEnvelope(f"plan_hash must be 'sha256:<hex>', got {value!r}")
    hex_part = value[len("sha256:"):]
    if len(hex_part) != 64 or any(c not in "0123456789abcdef" for c in hex_part):
        raise InvalidEnvelope(f"plan_hash must carry 64 lowercase hex chars, got {value!r}")
    return value


@dataclass(frozen=True)
class ResourceEnvelope:
    envelope_id: str
    phase: str
    backend: str
    host: HostEnvelope
    device: DeviceEnvelope | None
    geometry: GeometryEnvelope
    concurrency: ConcurrencyEnvelope
    estimate: EstimateEnvelope
    bound: BoundEnvelope
    accounting: AccountingEnvelope
    overlaps_with: tuple[str, ...]
    identity: IdentityEnvelope
    reason_code: str
    schema: str = SCHEMA

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "envelope_id": self.envelope_id,
            "phase": self.phase,
            "backend": self.backend,
            "host": self.host.to_dict(),
            "device": (None if self.device is None else self.device.to_dict()),
            "geometry": self.geometry.to_dict(),
            "concurrency": self.concurrency.to_dict(),
            "estimate": self.estimate.to_dict(),
            "bound": self.bound.to_dict(),
            "accounting": self.accounting.to_dict(),
            "overlaps_with": list(self.overlaps_with),
            "identity": self.identity.to_dict(),
            "reason_code": self.reason_code,
        }


# ---------------------------------------------------------------------------
# Envelope validation
# ---------------------------------------------------------------------------

def validate_envelope(envelope: ResourceEnvelope) -> None:
    """Validate ONE envelope; raises :class:`InvalidEnvelope` / ``InvalidOverlap``."""
    if not isinstance(envelope, ResourceEnvelope):
        raise InvalidEnvelope(
            f"expected a ResourceEnvelope, got {type(envelope).__name__}"
        )
    if envelope.schema != SCHEMA:
        raise InvalidEnvelope(f"schema {envelope.schema!r} != {SCHEMA!r}")
    if envelope.phase not in PHASES:
        raise InvalidEnvelope(
            f"unknown phase {envelope.phase!r} (closed vocabulary {PHASES})"
        )
    if envelope.backend not in BACKENDS:
        raise InvalidEnvelope(
            f"unknown backend {envelope.backend!r} (closed vocabulary {BACKENDS})"
        )
    _require_nonempty_str(envelope.envelope_id, "envelope_id")
    envelope.host.validate()
    envelope.geometry.validate()
    envelope.concurrency.validate()
    envelope.estimate.validate()
    envelope.bound.validate()
    envelope.accounting.validate()
    envelope.identity.validate()

    # Device vs backend: CPU carries no device and no device peak; GPU carries a
    # device envelope and a positive (or explicitly zero-before-launch) peak.
    if envelope.backend == BACKEND_CPU:
        if envelope.device is not None:
            raise InvalidEnvelope("cpu envelope must not carry a device block")
        if int(envelope.accounting.device_peak_bytes) != 0:
            raise InvalidEnvelope("cpu envelope must declare device_peak_bytes = 0")
    else:
        if envelope.device is None:
            raise InvalidEnvelope("gpu envelope must carry a device block")
        envelope.device.validate()
        if int(envelope.accounting.device_peak_bytes) > int(envelope.device.budget_bytes):
            raise InvalidEnvelope(
                f"device_peak_bytes {envelope.accounting.device_peak_bytes} > "
                f"vram budget {envelope.device.budget_bytes}"
            )

    if int(envelope.accounting.host_peak_bytes) != int(envelope.estimate.peak_bytes):
        raise InvalidEnvelope(
            f"accounting.host_peak_bytes {envelope.accounting.host_peak_bytes} != "
            f"estimate.peak_bytes {envelope.estimate.peak_bytes}"
        )

    # overlaps_with is MANDATORY (even when empty) and closed.
    ov = envelope.overlaps_with
    if not isinstance(ov, tuple):
        raise InvalidOverlap(
            f"{envelope.phase}.overlaps_with must be a tuple (mandatory field)"
        )
    if len(set(ov)) != len(ov):
        raise InvalidOverlap(f"{envelope.phase}.overlaps_with has duplicates: {ov}")
    for other in ov:
        if other == envelope.phase:
            raise InvalidOverlap(f"{envelope.phase} must not overlap itself")
        if other not in PHASES:
            raise InvalidOverlap(
                f"{envelope.phase}.overlaps_with declares unknown phase {other!r}"
            )

    if envelope.reason_code not in REASON_CODES:
        raise InvalidEnvelope(
            f"reason_code {envelope.reason_code!r} is outside the closed R55 set"
        )

    # No double-counted envelope identity.
    expected_id = f"{envelope.phase}.{envelope.backend}"
    if not envelope.envelope_id.startswith(expected_id):
        raise InvalidEnvelope(
            f"envelope_id {envelope.envelope_id!r} must start with "
            f"{expected_id!r} (one envelope per phase/backend)"
        )


def validate_envelope_set(
    envelopes: Iterable[ResourceEnvelope],
    *,
    headless: bool = True,
) -> tuple[ResourceEnvelope, ...]:
    """Validate a PLAN set: uniqueness, required phases, overlap symmetry.

    ``headless=True`` (the product default) refuses a ``gui_preview`` envelope:
    a headless run must never count a preview. ``headless=False`` allows at most
    one preview envelope, and only alongside a CPU backend.

    Returns the envelopes in the canonical :data:`PHASES` order.
    """
    envs = tuple(envelopes)
    if not envs:
        raise InvalidEnvelope("an envelope set cannot be empty")
    seen: dict[str, ResourceEnvelope] = {}
    for env in envs:
        validate_envelope(env)
        if env.phase in seen:
            raise InvalidEnvelope(
                f"duplicate envelope for phase {env.phase!r}: one envelope per phase"
            )
        seen[env.phase] = env

    missing = [p for p in REQUIRED_PLAN_PHASES if p not in seen]
    if missing:
        raise InvalidEnvelope(f"missing required PLAN phase envelope(s): {missing}")
    stacks = [p for p in CELL_STACK_PHASES if p in seen]
    if len(stacks) != 1:
        raise InvalidEnvelope(
            "a PLAN set requires exactly ONE cell_stack_* envelope, got "
            f"{stacks or 'none'}"
        )
    if PHASE_GUI_PREVIEW in seen:
        if headless:
            raise InvalidEnvelope(
                "gui_preview must not be counted in a headless plan"
            )
        if seen[PHASE_GUI_PREVIEW].backend != BACKEND_CPU:
            raise InvalidEnvelope("gui_preview must be a CPU envelope")

    # Overlap symmetry: undeclared/asymmetric overlap fails closed.
    for env in envs:
        for other_phase in env.overlaps_with:
            other = seen.get(other_phase)
            if other is None:
                raise InvalidOverlap(
                    f"{env.phase}.overlaps_with declares {other_phase!r}, which is "
                    "not part of this plan set (undecidable overlap)"
                )
            if env.phase not in other.overlaps_with:
                raise InvalidOverlap(
                    f"asymmetric overlap: {env.phase} declares {other_phase} but "
                    f"{other_phase} does not declare {env.phase}"
                )

    # Uniform margin: the aggregate applies exactly ONE margin value.
    margins = {(e.bound.margin_num, e.bound.margin_den) for e in envs}
    if len(margins) != 1:
        raise InvalidEnvelope(
            f"all phase envelopes must share one margin, got {sorted(margins)}"
        )

    order = {p: i for i, p in enumerate(PHASES)}
    return tuple(sorted(envs, key=lambda e: order[e.phase]))


# ---------------------------------------------------------------------------
# Overlap aggregation
# ---------------------------------------------------------------------------

def overlap_groups(
    envelopes: Sequence[ResourceEnvelope],
) -> tuple[tuple[str, ...], ...]:
    """Return the canonical overlap groups (connected components) of a plan.

    Two phases in the same group are SIMULTANEOUS and their peaks are summed.
    Different groups are mutually exclusive and only the largest counts.
    """
    envs = tuple(envelopes)
    phases = [e.phase for e in envs]
    parent = {p: p for p in phases}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for env in envs:
        for other in env.overlaps_with:
            if other in parent:
                union(env.phase, other)

    groups: dict[str, list[str]] = {}
    for p in phases:
        groups.setdefault(find(p), []).append(p)
    order = {p: i for i, p in enumerate(PHASES)}
    return tuple(
        tuple(sorted(g, key=lambda p: order[p]))
        for _root, g in sorted(groups.items(), key=lambda kv: order[min(kv[1], key=lambda p: order[p])])
    )


def _aggregate(
    envelopes: Sequence[ResourceEnvelope],
    value,
) -> int:
    """Group-sum then cross-group max (margin NOT applied)."""
    envs = tuple(envelopes)
    per_phase = {e.phase: int(value(e)) for e in envs}
    best = 0
    for group in overlap_groups(envs):
        best = max(best, sum(per_phase[p] for p in group))
    return int(best)


def aggregate_host_peak_bytes(envelopes: Sequence[ResourceEnvelope]) -> tuple[int, int]:
    """Return ``(raw_aggregate, margin_applied_peak)``.

    ``raw`` is the group-sum/cross-group-max projection. ``peak`` applies the
    shared margin EXACTLY ONCE at this boundary. No phase peak and no caller may
    apply it again (no double reserve).
    """
    envs = validate_envelope_set(envelopes)
    raw = _aggregate(envs, lambda e: e.estimate.peak_bytes)
    margin_num, margin_den = envs[0].bound.margin_num, envs[0].bound.margin_den
    return raw, ceil_margin_div(raw, margin_num, margin_den)


def aggregate_device_peak_bytes(envelopes: Sequence[ResourceEnvelope]) -> tuple[int, int]:
    """Device analogue of :func:`aggregate_host_peak_bytes` (0 for CPU plans)."""
    envs = tuple(envelopes)
    raw = _aggregate(envs, lambda e: e.accounting.device_peak_bytes)
    margin_num, margin_den = envs[0].bound.margin_num, envs[0].bound.margin_den
    return raw, ceil_margin_div(raw, margin_num, margin_den)


def aggregate_disk_bytes(envelopes: Sequence[ResourceEnvelope]) -> int:
    """Disk residency projection (group-sum, cross-group max; no margin)."""
    envs = tuple(envelopes)
    return _aggregate(envs, lambda e: e.accounting.disk_bytes)


# ---------------------------------------------------------------------------
# Canonical plan hash
# ---------------------------------------------------------------------------

def _canonical_envelope_payload(env: ResourceEnvelope) -> dict:
    """Immutable decision fields only.

    Deliberately EXCLUDES ``identity`` (run id, plan hash, pinned is included
    separately as a decision), observation timestamps and human notes, so the
    hash is stable across process restarts and dictionary insertion order.
    """
    return {
        "envelope_id": env.envelope_id,
        "phase": env.phase,
        "backend": env.backend,
        "host": {
            "raw_available_bytes": int(env.host.raw_available_bytes),
            "factor_num": int(env.host.factor_num),
            "factor_den": int(env.host.factor_den),
            "budget_bytes": int(env.host.budget_bytes),
            "reserve_bytes": int(env.host.reserve_bytes),
            "process_baseline_bytes": int(env.host.process_baseline_bytes),
        },
        "device": (
            None if env.device is None else {
                "vram_total_bytes": int(env.device.vram_total_bytes),
                "vram_free_bytes": int(env.device.vram_free_bytes),
                "vram_reserved_estimate_bytes": int(env.device.vram_reserved_estimate_bytes),
                "budget_bytes": int(env.device.budget_bytes),
                "device_name": env.device.device_name,
            }
        ),
        "geometry": {
            "layout": [int(env.geometry.layout[0]), int(env.geometry.layout[1])],
            "refinement_rank": int(env.geometry.refinement_rank),
            "cell_id": env.geometry.cell_id,
            "patch_hw": (
                None if env.geometry.patch_hw is None
                else [int(env.geometry.patch_hw[0]), int(env.geometry.patch_hw[1])]
            ),
            "tile_px": (None if env.geometry.tile_px is None else int(env.geometry.tile_px)),
            "halo_px": int(env.geometry.halo_px),
            "channels": int(env.geometry.channels),
        },
        "concurrency": {
            "units": int(env.concurrency.units),
            "units_in_flight": int(env.concurrency.units_in_flight),
            "workers_per_unit": int(env.concurrency.workers_per_unit),
            "prefetch_units": int(env.concurrency.prefetch_units),
        },
        "estimate": {
            "method": env.estimate.method,
            "calibration_id": env.estimate.calibration_id,
            "components": {k: int(v) for k, v in env.estimate.components},
            "subtotal_bytes": int(env.estimate.subtotal_bytes),
            "peak_bytes": int(env.estimate.peak_bytes),
        },
        "bound": {
            "kind": env.bound.kind,
            "margin_num": int(env.bound.margin_num),
            "margin_den": int(env.bound.margin_den),
            "confidence": env.bound.confidence,
        },
        "accounting": {
            "aggregate_rule": env.accounting.aggregate_rule,
            "disk_bytes": int(env.accounting.disk_bytes),
            "transfer_bytes": int(env.accounting.transfer_bytes),
        },
        "overlaps_with": list(env.overlaps_with),
        "reason_code": env.reason_code,
        "pinned": bool(env.identity.pinned),
    }


def compute_plan_hash(envelopes: Sequence[ResourceEnvelope]) -> str:
    """Canonical ``sha256:`` hash over the immutable decision fields of a plan."""
    envs = validate_envelope_set(envelopes)
    payload = [_canonical_envelope_payload(e) for e in envs]
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def make_estimate(
    components: Mapping[str, int],
    *,
    method: str = "static_bound",
    calibration_id: str | None = None,
) -> EstimateEnvelope:
    """Build a validated estimate from an explicit component mapping.

    The components must be supplied by the caller (from a calibration profile or
    a declared static bound). No coefficient is invented here.
    """
    items = tuple(sorted((str(k), require_int(v, f"component[{k}]"))
                         for k, v in components.items()))
    total = sum(v for _k, v in items)
    return EstimateEnvelope(
        method=method,
        calibration_id=calibration_id,
        components=items,
        subtotal_bytes=int(total),
        peak_bytes=int(total),
    )
