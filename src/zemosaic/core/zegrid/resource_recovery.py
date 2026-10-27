"""ZM-ZEGRID-RESILIENT-RUN-RESUME M3 — resource-pressure recovery.

This module is the M3 runtime seam (plan §5.3, §5.4, §5.7). It owns the
*policy* of one recovery episode; it never touches science arrays, budgets or
publication. The product coordinator supplies the real release callbacks and
the real capacity probe.

Non-negotiable rules implemented here
-------------------------------------
* RELEASE BEFORE WAIT. A recovery episode stops admission, cancels/drains
  speculative (uncommitted) work and releases the credits/staging it owns
  BEFORE entering ``WAITING_RESOURCES``. A report of what was actually released
  is kept; waiting while still holding the buffers is refused.
* ONE GOVERNING OBSERVATION PER EPISODE. The first valid, fresh observation of
  an episode governs every reduction of that episode. Later samples may not be
  mixed in to buy a more favourable budget. A *new* episode after a wait may
  use new measurements, and that is journaled explicitly.
* ONE-WAY AUTO -> BOUNDED. ``auto`` may fall back once; an explicit ``dense``
  request is never switched silently; ``bounded`` stays bounded.
* BOUNDED RETRIES. A cell/plan gets a bounded number of reduced retries; after
  exhaustion the run waits for real capacity (or produces a resumable
  diagnostic) instead of relaunching forever.
* GENERATIONS. Every task/payload carries ``run_id``, ``cell_id``,
  ``attempt_id`` and ``plan_generation``. A late result from a cancelled
  generation is rejected, never published; previously committed payloads remain
  valid and their identities immutable.
* LIGHT, CANCELLABLE WAIT. Probes are spaced (5 s, backoff to 30 s) and the
  wait sleeps on an event; readmission needs TWO admissible observations spaced
  by at least 5 s. Nothing polls in a tight loop. A stop request returns
  ``STOPPED_RESUMABLE`` and never erases already-committed work.

The module is isolated (stdlib only, plus the sibling registry for the optional
fence delegation) so it can be tested without a GPU, a product run or Windows.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from . import run_registry as zrr
from . import storage_profile as zsp

__all__ = [
    "SCHEMA",
    "STATE_ACTIVE",
    "STATE_RECOVERING",
    "STATE_WAITING_RESOURCES",
    "STATE_READMITTING",
    "STATE_STOPPED_RESUMABLE",
    "STATE_CAPACITY_INFEASIBLE",
    "PROBE_INITIAL_INTERVAL_S",
    "PROBE_MAX_INTERVAL_S",
    "REQUIRED_ADMISSIBLE_OBSERVATIONS",
    "MIN_ADMISSIBLE_SPACING_S",
    "DEFAULT_RETRY_ATTEMPTS",
    "RecoveryError",
    "ObservationError",
    "CapacityWaitCancelled",
    "CapacityWaitDeadline",
    "CapacityWaitExhausted",
    "ReleaseBlockedError",
    "ReleaseIncompleteError",
    "StructuralCapacityRefused",
    "RetryExhausted",
    "GenerationError",
    "StaleGenerationError",
    "FutureGenerationError",
    "RunMismatchError",
    "CapacityObservation",
    "CapacityWaitResult",
    "parse_observation",
    "CapacityWaiter",
    "ProfileDecision",
    "StorageProfilePolicy",
    "ReleaseReport",
    "TaskGeneration",
    "FenceOutcome",
    "GenerationFence",
    "RetryBudget",
    "RecoveryOutcome",
    "RecoveryCoordinator",
    "PipelineReleaseAdapter",
    "default_capacity_probe",
]

SCHEMA = "zemosaic-zegrid-resource-recovery-v1"

#: Run-control states. ``WAITING_RESOURCES`` is deliberately NOT a worker
#: death nor a terminal run state: it is visible, light and cancellable.
STATE_ACTIVE = "ACTIVE"
STATE_RECOVERING = "RECOVERING"
STATE_WAITING_RESOURCES = "WAITING_RESOURCES"
STATE_READMITTING = "READMITTING"
STATE_STOPPED_RESUMABLE = "STOPPED_RESUMABLE"
STATE_CAPACITY_INFEASIBLE = "CAPACITY_INFEASIBLE"

#: Plan §5.3 policy numbers, frozen by the M3 tests.
PROBE_INITIAL_INTERVAL_S = 5.0
PROBE_MAX_INTERVAL_S = 30.0
REQUIRED_ADMISSIBLE_OBSERVATIONS = 2
MIN_ADMISSIBLE_SPACING_S = 5.0

#: Plan §5.4: a bounded number of reduced attempts per (cell, plan generation).
DEFAULT_RETRY_ATTEMPTS = 2

#: A latched/fresh sample older than this is not a valid governing fact.
DEFAULT_MAX_OBSERVATION_AGE_S = 30.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RecoveryError(RuntimeError):
    """Base class for M3 recovery errors."""


class ObservationError(RecoveryError):
    """A capacity observation is malformed, stale or from the future."""


class CapacityWaitCancelled(RecoveryError):
    """The wait was cancelled (stop requested). Work is resumable."""

    state = STATE_STOPPED_RESUMABLE


class CapacityWaitDeadline(RecoveryError):
    """The optional wait deadline elapsed. Work is resumable."""

    state = STATE_STOPPED_RESUMABLE

    def __init__(self, message: str, *, result: "CapacityWaitResult | None" = None) -> None:
        super().__init__(message)
        self.result = result


class CapacityWaitExhausted(RecoveryError):
    """A bounded test/probe budget was exhausted without readmission."""

    state = STATE_WAITING_RESOURCES


class ReleaseBlockedError(RecoveryError):
    """A COMMITTING transaction could not finish; never destroy it in flight."""


class ReleaseIncompleteError(RecoveryError):
    """Waiting was requested while credits/staging were still owned."""


class StructuralCapacityRefused(RecoveryError):
    """The structural minimum cannot fit even with total machine capacity."""

    state = STATE_CAPACITY_INFEASIBLE


class RetryExhausted(RecoveryError):
    """Bounded retries for this (cell, plan) are spent: wait for capacity."""

    state = STATE_WAITING_RESOURCES

    def __init__(self, message: str, *, cell_id: str, plan_generation: int) -> None:
        super().__init__(message)
        self.cell_id = str(cell_id)
        self.plan_generation = int(plan_generation)


class GenerationError(RecoveryError):
    """Base class for generation/fencing errors."""


class StaleGenerationError(GenerationError):
    """A result belongs to a cancelled/older plan generation."""


class FutureGenerationError(GenerationError):
    """A result claims a generation the coordinator has not reached."""


class RunMismatchError(GenerationError):
    """A result/task belongs to another run."""


# ---------------------------------------------------------------------------
# Capacity observations
# ---------------------------------------------------------------------------


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ObservationError(f"{name} must be a strict int, got {value!r}")
    if int(value) < int(minimum):
        raise ObservationError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


@dataclass(frozen=True)
class CapacityObservation:
    """One validated host-capacity fact, with its own observation time."""

    raw_available_bytes: int
    observed_monotonic: float
    source: str

    def admissible(self, minimum_bytes: int) -> bool:
        return int(self.raw_available_bytes) >= int(minimum_bytes)

    def to_dict(self) -> dict:
        return {
            "raw_available_bytes": int(self.raw_available_bytes),
            "observed_monotonic": float(self.observed_monotonic),
            "source": str(self.source),
        }


def parse_observation(
    value: Any,
    *,
    now: float | None = None,
    max_age_s: float | None = DEFAULT_MAX_OBSERVATION_AGE_S,
) -> CapacityObservation:
    """Validate one observation; fail closed on anything ambiguous.

    A missing observation time is accepted only when ``max_age_s`` is ``None``
    (tests/standalone callers must be explicit). In production the real sampler
    always stamps the fact.
    """
    if isinstance(value, CapacityObservation):
        if max_age_s is None:
            return value
        current = time.monotonic() if now is None else float(now)
        age = current - float(value.observed_monotonic)
        if age < 0.0:
            raise ObservationError("capacity observation is from the future")
        if age > float(max_age_s):
            raise ObservationError(
                f"capacity observation is stale ({age:.3f}s > {float(max_age_s):.3f}s)"
            )
        return value
    if not isinstance(value, Mapping):
        raise ObservationError(
            f"capacity observation must be a mapping, got {type(value).__name__}"
        )
    raw = _strict_int(value.get("raw_available_bytes"), "raw_available_bytes")
    source = str(value.get("source", "unspecified") or "unspecified")
    stamp = value.get("observed_monotonic")
    if stamp is None:
        if max_age_s is not None:
            raise ObservationError(
                "capacity observation has no observed_monotonic timestamp"
            )
        stamp = 0.0 if now is None else float(now)
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        raise ObservationError(f"observed_monotonic must be a number, got {stamp!r}")
    stamp = float(stamp)
    if max_age_s is not None:
        current = time.monotonic() if now is None else float(now)
        age = current - stamp
        if age < 0.0:
            raise ObservationError("capacity observation is from the future")
        if age > float(max_age_s):
            raise ObservationError(
                f"capacity observation is stale ({age:.3f}s > {float(max_age_s):.3f}s)"
            )
    return CapacityObservation(
        raw_available_bytes=int(raw),
        observed_monotonic=stamp,
        source=source,
    )


def default_capacity_probe() -> dict:
    """Read the LIVE host available memory (never a synthetic value).

    Linux uses ``/proc/meminfo``; other platforms use psutil when installed.
    The caller may inject any probe; the recovery policy itself is probe
    agnostic. An unavailable probe raises, and the wait fails closed.
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return {
                        "raw_available_bytes": int(line.split()[1]) * 1024,
                        "observed_monotonic": time.monotonic(),
                        "source": "meminfo:MemAvailable",
                    }
    except OSError:
        pass
    try:
        import psutil  # type: ignore

        return {
            "raw_available_bytes": int(psutil.virtual_memory().available),
            "observed_monotonic": time.monotonic(),
            "source": "psutil:available",
        }
    except Exception as exc:  # noqa: BLE001 - fail closed, never fabricate
        raise ObservationError(
            f"no host-capacity probe is available: {type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Light, cancellable capacity wait
# ---------------------------------------------------------------------------


@dataclass
class CapacityWaitResult:
    status: str
    admissible: bool
    probes: int
    admissible_observations: int
    waited_s: float
    sleeps: int
    last: CapacityObservation | None
    observations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": str(self.status),
            "admissible": bool(self.admissible),
            "probes": int(self.probes),
            "admissible_observations": int(self.admissible_observations),
            "waited_s": float(self.waited_s),
            "sleeps": int(self.sleeps),
            "last": None if self.last is None else self.last.to_dict(),
            "observations": [dict(item) for item in self.observations],
        }


class CapacityWaiter:
    """Spaced, event-based wait for durable capacity.

    The wait is intentionally boring: at most one probe per interval, and the
    thread sleeps on an event between probes (no busy loop). Cancellation and
    the optional deadline are checked at each wake-up, so a stop never waits for
    a long interval to elapse.
    """

    def __init__(
        self,
        probe: Callable[[], Any],
        *,
        minimum_bytes: int,
        stop_event: threading.Event | None = None,
        cancel_event: threading.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] | None = None,
        initial_interval_s: float = PROBE_INITIAL_INTERVAL_S,
        max_interval_s: float = PROBE_MAX_INTERVAL_S,
        required_observations: int = REQUIRED_ADMISSIBLE_OBSERVATIONS,
        min_spacing_s: float = MIN_ADMISSIBLE_SPACING_S,
        deadline_s: float | None = None,
        max_probes: int | None = None,
        max_age_s: float | None = DEFAULT_MAX_OBSERVATION_AGE_S,
        on_observation: Callable[[CapacityObservation], None] | None = None,
    ) -> None:
        if not callable(probe):
            raise RecoveryError("probe must be callable")
        self.minimum_bytes = _strict_int(minimum_bytes, "minimum_bytes")
        if (
            isinstance(initial_interval_s, bool)
            or not isinstance(initial_interval_s, (int, float))
            or float(initial_interval_s) <= 0.0
        ):
            raise RecoveryError("initial_interval_s must be > 0")
        if (
            isinstance(max_interval_s, bool)
            or not isinstance(max_interval_s, (int, float))
            or float(max_interval_s) < float(initial_interval_s)
        ):
            raise RecoveryError(
                "max_interval_s must be a number >= initial_interval_s"
            )
        if (
            isinstance(required_observations, bool)
            or not isinstance(required_observations, int)
            or int(required_observations) < 2
        ):
            raise RecoveryError("required_observations must be an int >= 2")
        if (
            isinstance(min_spacing_s, bool)
            or not isinstance(min_spacing_s, (int, float))
            or float(min_spacing_s) < 0.0
        ):
            raise RecoveryError("min_spacing_s must be >= 0")
        if deadline_s is not None and (
            isinstance(deadline_s, bool)
            or not isinstance(deadline_s, (int, float))
            or float(deadline_s) <= 0.0
        ):
            raise RecoveryError("deadline_s must be None or > 0")
        if max_probes is not None and (
            isinstance(max_probes, bool)
            or not isinstance(max_probes, int)
            or int(max_probes) < 1
        ):
            raise RecoveryError("max_probes must be None or an int >= 1")
        self._probe = probe
        self._stop_event = stop_event
        self._cancel_event = cancel_event
        self._clock = clock
        self._sleep = sleep
        self._initial_interval_s = float(initial_interval_s)
        self._max_interval_s = float(max_interval_s)
        self._required = int(required_observations)
        self._min_spacing_s = float(min_spacing_s)
        self._deadline_s = None if deadline_s is None else float(deadline_s)
        self._max_probes = max_probes
        self._max_age_s = max_age_s
        self._on_observation = on_observation
        self.probes = 0
        self.sleeps = 0
        self.observations: list[dict] = []
        self._last: CapacityObservation | None = None

    # -- cancellation ------------------------------------------------------

    def _cancelled(self) -> bool:
        for event in (self._stop_event, self._cancel_event):
            if event is not None and event.is_set():
                return True
        return False

    def _check_cancel(self) -> None:
        if self._cancelled():
            raise CapacityWaitCancelled(
                "resource wait cancelled (stop requested); work is resumable"
            )

    def _wait(self, delay: float) -> None:
        delay = max(0.0, float(delay))
        if delay <= 0.0:
            # Never spin: even a satisfied spacing constraint yields the CPU.
            delay = 0.001
        self.sleeps += 1
        if self._sleep is not None:
            self._sleep(delay)
            return
        event = self._cancel_event or self._stop_event
        if event is not None:
            event.wait(delay)
            return
        time.sleep(delay)

    # -- main loop ---------------------------------------------------------

    def wait(self) -> CapacityWaitResult:
        started = float(self._clock())
        interval = float(self._initial_interval_s)
        streak: list[CapacityObservation] = []
        while True:
            self._check_cancel()
            if self._max_probes is not None and self.probes >= self._max_probes:
                raise CapacityWaitExhausted(
                    "capacity wait exhausted its bounded probe budget; "
                    "work remains resumable"
                )
            try:
                observation = parse_observation(
                    self._probe(), now=float(self._clock()), max_age_s=self._max_age_s
                )
            except ObservationError:
                # A malformed/failed live probe is not evidence of capacity.
                # It fails closed immediately instead of being retried in a way
                # that could admit work from a stale fallback fact.
                raise
            self.probes += 1
            self._last = observation
            record = observation.to_dict()
            record["admissible"] = observation.admissible(self.minimum_bytes)
            self.observations.append(record)
            if self._on_observation is not None:
                self._on_observation(observation)
            if record["admissible"]:
                if streak:
                    gap = (
                        observation.observed_monotonic
                        - streak[-1].observed_monotonic
                    )
                    if gap < self._min_spacing_s:
                        # Same burst, not a second durable observation.
                        streak = [observation]
                    else:
                        streak.append(observation)
                else:
                    streak = [observation]
                if len(streak) >= self._required:
                    return CapacityWaitResult(
                        status=STATE_ACTIVE,
                        admissible=True,
                        probes=self.probes,
                        admissible_observations=len(streak),
                        waited_s=max(0.0, float(self._clock()) - started),
                        sleeps=self.sleeps,
                        last=observation,
                        observations=[dict(x) for x in self.observations],
                    )
                delay = max(
                    0.0,
                    self._min_spacing_s
                    - (float(self._clock()) - observation.observed_monotonic),
                )
            else:
                streak = []
                delay = interval
                interval = min(self._max_interval_s, float(interval) * 2.0)
            if self._deadline_s is not None:
                elapsed = max(0.0, float(self._clock()) - started)
                remaining = self._deadline_s - elapsed
                if remaining <= 0.0:
                    result = CapacityWaitResult(
                        status=STATE_STOPPED_RESUMABLE,
                        admissible=False,
                        probes=self.probes,
                        admissible_observations=len(streak),
                        waited_s=elapsed,
                        sleeps=self.sleeps,
                        last=self._last,
                        observations=[dict(x) for x in self.observations],
                    )
                    raise CapacityWaitDeadline(
                        "resource wait deadline reached; STOPPED_RESUMABLE, "
                        "committed work is untouched",
                        result=result,
                    )
                delay = min(delay, remaining)
            self._wait(delay)


# ---------------------------------------------------------------------------
# Storage profile policy (one-way AUTO -> BOUNDED)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileDecision:
    requested: str
    effective: str
    switched: bool
    reason: str
    transition: dict | None = None

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "effective": self.effective,
            "switched": bool(self.switched),
            "reason": str(self.reason),
            "transition": None if self.transition is None else dict(self.transition),
        }


class StorageProfilePolicy:
    """The single, one-way AUTO -> BOUNDED transition, enforced in one place.

    ``dense`` requested explicitly NEVER switches. ``bounded`` stays bounded.
    ``auto`` switches at most once, on persistent qualified pressure. The
    effective profile is published before any rebuild by the caller.
    """

    def __init__(self, *, requested: str, effective: str | None = None) -> None:
        req = zsp.normalize_storage_profile(
            requested, default=zsp.STORAGE_PROFILE_AUTO
        )
        if req not in zsp.STORAGE_REQUESTS:
            raise RecoveryError(f"invalid storage request {requested!r}")
        eff = (
            zsp.STORAGE_PROFILE_DENSE
            if effective is None and req != zsp.STORAGE_PROFILE_BOUNDED
            else zsp.STORAGE_PROFILE_BOUNDED
            if effective is None
            else zsp.normalize_storage_profile(effective)
        )
        if req == zsp.STORAGE_PROFILE_DENSE and eff != zsp.STORAGE_PROFILE_DENSE:
            raise RecoveryError(
                "an explicit dense request cannot start in the bounded profile"
            )
        if req == zsp.STORAGE_PROFILE_BOUNDED and eff != zsp.STORAGE_PROFILE_BOUNDED:
            raise RecoveryError(
                "an explicit bounded request cannot start in the dense profile"
            )
        self.requested = req
        self.effective = eff
        self.transitions: list[dict] = []
        self._switch_count = 1 if eff == zsp.STORAGE_PROFILE_BOUNDED and req == zsp.STORAGE_PROFILE_AUTO else 0

    def decide(
        self,
        *,
        pressure_persists: bool,
        phase: str,
        reason_code: str,
        detail: str,
    ) -> ProfileDecision:
        if not pressure_persists:
            return ProfileDecision(
                requested=self.requested,
                effective=self.effective,
                switched=False,
                reason="no_persistent_pressure",
            )
        if self.requested == zsp.STORAGE_PROFILE_DENSE:
            return ProfileDecision(
                requested=self.requested,
                effective=self.effective,
                switched=False,
                reason="explicit_dense_never_switched",
            )
        if self.requested == zsp.STORAGE_PROFILE_BOUNDED:
            return ProfileDecision(
                requested=self.requested,
                effective=self.effective,
                switched=False,
                reason="bounded_stays_bounded",
            )
        # AUTO
        if self.effective == zsp.STORAGE_PROFILE_BOUNDED:
            return ProfileDecision(
                requested=self.requested,
                effective=self.effective,
                switched=False,
                reason="auto_already_switched_once",
            )
        self.effective = zsp.STORAGE_PROFILE_BOUNDED
        self._switch_count += 1
        transition = {
            "from": zsp.STORAGE_PROFILE_DENSE,
            "to": zsp.STORAGE_PROFILE_BOUNDED,
            "phase": str(phase),
            "reason_code": str(reason_code),
            "detail": str(detail),
        }
        self.transitions.append(transition)
        return ProfileDecision(
            requested=self.requested,
            effective=self.effective,
            switched=True,
            reason="auto_to_bounded_once",
            transition=dict(transition),
        )

    def snapshot(self) -> dict:
        return {
            "requested": self.requested,
            "effective": self.effective,
            "switch_count": int(self._switch_count),
            "transitions": [dict(x) for x in self.transitions],
        }


# ---------------------------------------------------------------------------
# Real release of speculative work
# ---------------------------------------------------------------------------


@dataclass
class ReleaseReport:
    stopped_admission: bool = False
    cancelled_units: int = 0
    drained_units: int = 0
    released_bytes: int = 0
    host_released_bytes: int = 0
    device_released_bytes: int = 0
    disk_released_bytes: int = 0
    credits_balanced: bool = False
    consumer_state_before: str = "IDLE"
    consumer_state_after: str = "IDLE"
    notes: list[str] = field(default_factory=list)

    @property
    def units_released(self) -> int:
        return int(self.cancelled_units) + int(self.drained_units)

    def to_dict(self) -> dict:
        return {
            "stopped_admission": bool(self.stopped_admission),
            "cancelled_units": int(self.cancelled_units),
            "drained_units": int(self.drained_units),
            "units_released": int(self.units_released),
            "released_bytes": int(self.released_bytes),
            "host_released_bytes": int(self.host_released_bytes),
            "device_released_bytes": int(self.device_released_bytes),
            "disk_released_bytes": int(self.disk_released_bytes),
            "credits_balanced": bool(self.credits_balanced),
            "consumer_state_before": str(self.consumer_state_before),
            "consumer_state_after": str(self.consumer_state_after),
            "notes": [str(x) for x in self.notes],
        }


class PipelineReleaseAdapter:
    """Release speculative pipeline work BEFORE a resource wait.

    The adapter is deliberately conservative:

    * a consumer in ``COMMITTING`` is never killed mid-transaction; the adapter
      waits (event-based, bounded) for that commit to finish, because finishing
      a safe commit preserves work;
    * admission is stopped and the speculative latch is cancelled;
    * queued units are drained and every ledger lease is released;
    * optional external drains (e.g. a worker IPC return channel) run last.

    It never returns while it still owns a queued unit or a ledger lease. If the
    pipeline cannot reach a balanced state inside the bounded grace, the adapter
    fails closed with :class:`ReleaseBlockedError` instead of waiting with the
    pressure still resident.
    """

    def __init__(
        self,
        pipeline=None,
        *,
        consumer_state: Callable[[], str] | None = None,
        commit_done_event: threading.Event | None = None,
        external_drains: Sequence[Callable[[], Mapping[str, Any]]] = (),
        clock: Callable[[], float] = time.monotonic,
        wait_event: threading.Event | None = None,
        commit_grace_s: float = 30.0,
        quiesce_grace_s: float = 10.0,
        poll_s: float = 0.05,
    ) -> None:
        self._pipeline = pipeline
        self._consumer_state = consumer_state or (lambda: "IDLE")
        self._commit_done = commit_done_event
        self._external_drains = tuple(external_drains)
        self._clock = clock
        self._wait_event = wait_event
        self._commit_grace_s = float(commit_grace_s)
        self._quiesce_grace_s = float(quiesce_grace_s)
        self._poll_s = max(0.005, float(poll_s))

    def _sleep(self, delay: float) -> None:
        if self._wait_event is not None:
            self._wait_event.wait(delay)
        else:
            time.sleep(delay)

    def _wait_commit(self) -> str:
        """Wait for a COMMITTING consumer to finish; never interrupt it."""
        state = str(self._consumer_state() or "IDLE")
        if state != "COMMITTING":
            return state
        started = float(self._clock())
        while str(self._consumer_state() or "IDLE") == "COMMITTING":
            if float(self._clock()) - started >= self._commit_grace_s:
                raise ReleaseBlockedError(
                    "consumer is still COMMITTING after "
                    f"{self._commit_grace_s:.3f}s; refusing to cancel a "
                    "transaction in flight (release before wait)"
                )
            self._sleep(self._poll_s)
        return str(self._consumer_state() or "IDLE")

    def __call__(self) -> ReleaseReport:
        report = ReleaseReport()
        report.consumer_state_before = str(self._consumer_state() or "IDLE")
        report.consumer_state_after = self._wait_commit()
        pipeline = self._pipeline
        before_ledger: dict = {}
        before_units = 0
        if pipeline is not None:
            if hasattr(pipeline, "ledger_snapshot"):
                before_ledger = pipeline.ledger_snapshot()
            if hasattr(pipeline, "telemetry"):
                try:
                    before_units = int(
                        pipeline.telemetry().get("units_released", 0) or 0
                    )
                except Exception:  # pragma: no cover - observational only
                    before_units = 0
            if hasattr(pipeline, "stop_admission"):
                pipeline.stop_admission("resource_pressure")
                report.stopped_admission = True
            else:  # pragma: no cover - defensive for duck-typed callers
                report.notes.append("pipeline has no stop_admission()")
            if hasattr(pipeline, "cancel"):
                pipeline.cancel("resource_pressure")
            # Let the active consumer relinquish its lease. This is a bounded,
            # event-spaced wait on the REAL ledger, not a state-only claim.
            if hasattr(pipeline, "ledger_snapshot"):
                started = float(self._clock())
                while not bool(pipeline.ledger_snapshot().get("balanced")):
                    if float(self._clock()) - started >= self._quiesce_grace_s:
                        raise ReleaseIncompleteError(
                            "pipeline credits are still held after "
                            f"{self._quiesce_grace_s:.3f}s; refusing to wait "
                            "with resident staging"
                        )
                    self._sleep(self._poll_s)
            if hasattr(pipeline, "drain_speculative"):
                drained = pipeline.drain_speculative()
                if isinstance(drained, Mapping):
                    report.cancelled_units += int(drained.get("cancelled_units", 0) or 0)
                    report.drained_units += int(drained.get("drained_units", 0) or 0)
                    report.host_released_bytes += int(
                        drained.get("host_released_bytes", 0) or 0
                    )
                    report.device_released_bytes += int(
                        drained.get("device_released_bytes", 0) or 0
                    )
                    report.disk_released_bytes += int(
                        drained.get("disk_released_bytes", 0) or 0
                    )
            if hasattr(pipeline, "ledger_snapshot"):
                ledger = pipeline.ledger_snapshot()
                report.credits_balanced = bool(ledger.get("balanced"))
                for kind in ("host", "device", "disk"):
                    delta = max(
                        0,
                        int(before_ledger.get(f"{kind}_used_bytes", 0))
                        - int(ledger.get(f"{kind}_used_bytes", 0)),
                    )
                    setattr(report, f"{kind}_released_bytes", delta)
                if hasattr(pipeline, "telemetry"):
                    try:
                        after_units = int(
                            pipeline.telemetry().get("units_released", 0) or 0
                        )
                        report.cancelled_units += max(0, after_units - before_units)
                    except Exception:  # pragma: no cover - observational only
                        pass
                if not report.released_bytes:
                    report.released_bytes = int(
                        before_ledger.get("host_used_bytes", 0)
                    )
            else:  # pragma: no cover - duck-typed pipeline
                report.credits_balanced = True
        else:
            report.credits_balanced = True
        for drain in self._external_drains:
            extra = drain()
            if not isinstance(extra, Mapping):
                raise ReleaseIncompleteError(
                    "external release drain must return a mapping"
                )
            report.cancelled_units += int(extra.get("cancelled_units", 0) or 0)
            report.drained_units += int(extra.get("drained_units", 0) or 0)
            report.host_released_bytes += int(extra.get("host_released_bytes", 0) or 0)
            report.device_released_bytes += int(
                extra.get("device_released_bytes", 0) or 0
            )
            report.disk_released_bytes += int(extra.get("disk_released_bytes", 0) or 0)
        report.released_bytes = (
            int(report.host_released_bytes)
            + int(report.device_released_bytes)
            + int(report.disk_released_bytes)
        )
        if not report.credits_balanced:
            raise ReleaseIncompleteError(
                "pipeline credits are not balanced after the speculative drain; "
                "refusing to wait while still holding reservations"
            )
        return report


# ---------------------------------------------------------------------------
# Generations: task envelopes, fencing, late-result rejection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskGeneration:
    """Identity carried by one task/payload from dispatch to publication."""

    run_id: str
    cell_id: str
    attempt_id: str
    plan_generation: int

    def __post_init__(self) -> None:
        for name in ("run_id", "cell_id", "attempt_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise GenerationError(f"{name} must be a non-empty string")
        if isinstance(self.plan_generation, bool) or not isinstance(
            self.plan_generation, int
        ):
            raise GenerationError("plan_generation must be a strict int")
        if int(self.plan_generation) < 0:
            raise GenerationError("plan_generation must be >= 0")

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "attempt_id": self.attempt_id,
            "plan_generation": int(self.plan_generation),
        }


@dataclass
class FenceOutcome:
    accepted: bool
    reason: str
    cell_id: str
    plan_generation: int
    receipt: Any = None
    detail: str | None = None

    def to_dict(self) -> dict:
        return {
            "accepted": bool(self.accepted),
            "reason": str(self.reason),
            "cell_id": self.cell_id,
            "plan_generation": int(self.plan_generation),
            "receipt": (
                self.receipt.to_dict()
                if hasattr(self.receipt, "to_dict")
                else self.receipt
            ),
            "detail": None if self.detail is None else str(self.detail),
        }


class GenerationFence:
    """Fence results by run id and plan generation; never adopt a late result.

    The optional ``registry`` is the M1 owner of accepted payloads. The fence
    adds:

    * an explicit cancelled-generation set (a generation transition marks the
      previous generation cancelled);
    * a non-raising acceptance path for late results (a late callback must not
      abort an otherwise healthy run, but it must never be published).

    Already-committed results are never touched: advancing/cancelling a
    generation only changes what may be accepted NEXT.
    """

    def __init__(self, *, run_id: str, registry=None) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise GenerationError("run_id must be a non-empty string")
        self.run_id = str(run_id)
        self._registry = registry
        self._generation = int(registry.generation) if registry is not None else 0
        self._cancelled: set[int] = set()
        self._cancelled_reasons: dict[int, str] = {}
        self._journal: list[dict] = []

    # -- generation state --------------------------------------------------

    @property
    def generation(self) -> int:
        return int(self._generation)

    def envelope(
        self, cell_id: str, *, attempt_id: str | None = None
    ) -> TaskGeneration:
        if not isinstance(cell_id, str) or not cell_id:
            raise GenerationError("cell_id must be a non-empty string")
        attempt = (
            str(attempt_id)
            if attempt_id is not None
            else f"gen{self._generation}:{cell_id}"
        )
        return TaskGeneration(
            run_id=self.run_id,
            cell_id=cell_id,
            attempt_id=attempt,
            plan_generation=int(self._generation),
        )

    def advance(self, *, reason: str = "plan_transition") -> int:
        """Cancel the current generation and move to the next one."""
        old = int(self._generation)
        self._cancelled.add(old)
        self._cancelled_reasons[old] = str(reason)
        self._journal.append(
            {
                "event": "generation_cancelled",
                "generation": old,
                "reason": str(reason),
            }
        )
        self._generation = old + 1
        if self._registry is not None:
            self._registry.advance_generation(reason=str(reason))
        self._journal.append(
            {
                "event": "generation_advanced",
                "previous": old,
                "generation": int(self._generation),
                "reason": str(reason),
            }
        )
        return int(self._generation)

    def cancel(self, generation: int, *, reason: str) -> bool:
        gen = int(generation)
        if gen in self._cancelled:
            return False
        self._cancelled.add(gen)
        self._cancelled_reasons[gen] = str(reason)
        self._journal.append(
            {
                "event": "generation_cancelled",
                "generation": gen,
                "reason": str(reason),
            }
        )
        return True

    def is_cancelled(self, generation: int) -> bool:
        return int(generation) in self._cancelled

    # -- validation --------------------------------------------------------

    def check(self, envelope: TaskGeneration) -> None:
        if envelope.run_id != self.run_id:
            raise RunMismatchError(
                f"task run_id {envelope.run_id!r} != fence run {self.run_id!r}"
            )
        if envelope.plan_generation > self._generation:
            raise FutureGenerationError(
                f"task generation {envelope.plan_generation} is newer than the "
                f"current {self._generation}"
            )
        if envelope.plan_generation < self._generation or self.is_cancelled(
            envelope.plan_generation
        ):
            raise StaleGenerationError(
                f"task generation {envelope.plan_generation} is cancelled/stale; "
                f"current is {self._generation}"
            )

    # -- publication -------------------------------------------------------

    def accept(
        self,
        envelope: TaskGeneration,
        result: Mapping[str, Any],
        *,
        backend: str | None,
    ) -> FenceOutcome:
        """Accept through the registry, or refuse a late/stale result.

        A late result is NEVER published and never raises into the healthy run.
        A schema/acceptance error from the live generation still raises (it is
        an internal failure, not a timing artefact).
        """
        cel = str(result.get("cell_id", "")) if isinstance(result, Mapping) else ""
        try:
            self.check(envelope)
            if cel and cel != envelope.cell_id:
                raise GenerationError(
                    f"result cell_id {cel!r} != envelope {envelope.cell_id!r}"
                )
        except (StaleGenerationError, FutureGenerationError, RunMismatchError) as exc:
            self._journal.append(
                {
                    "event": "late_result_rejected",
                    "cell_id": envelope.cell_id,
                    "plan_generation": int(envelope.plan_generation),
                    "current_generation": int(self._generation),
                    "reason": type(exc).__name__,
                }
            )
            return FenceOutcome(
                accepted=False,
                reason=(
                    "stale_generation"
                    if isinstance(exc, StaleGenerationError)
                    else (
                        "future_generation"
                        if isinstance(exc, FutureGenerationError)
                        else "run_mismatch"
                    )
                ),
                cell_id=envelope.cell_id,
                plan_generation=int(envelope.plan_generation),
                detail=str(exc),
            )
        except GenerationError as exc:
            return FenceOutcome(
                accepted=False,
                reason="identity_mismatch",
                cell_id=envelope.cell_id,
                plan_generation=int(envelope.plan_generation),
                detail=str(exc),
            )
        if self._registry is None:
            return FenceOutcome(
                accepted=True,
                reason="no_registry",
                cell_id=envelope.cell_id,
                plan_generation=int(envelope.plan_generation),
            )
        try:
            receipt = self._registry.accept(
                result,
                backend=backend,
                plan_generation=int(envelope.plan_generation),
                attempt_id=envelope.attempt_id,
            )
        except zrr.StaleGenerationError as exc:
            self.cancel(envelope.plan_generation, reason="registry_stale")
            return FenceOutcome(
                accepted=False,
                reason="stale_generation",
                cell_id=envelope.cell_id,
                plan_generation=int(envelope.plan_generation),
                detail=str(exc),
            )
        except zrr.FutureGenerationError as exc:
            return FenceOutcome(
                accepted=False,
                reason="future_generation",
                cell_id=envelope.cell_id,
                plan_generation=int(envelope.plan_generation),
                detail=str(exc),
            )
        except zrr.DuplicateCellError as exc:
            return FenceOutcome(
                accepted=False,
                reason="duplicate_conflict",
                cell_id=envelope.cell_id,
                plan_generation=int(envelope.plan_generation),
                detail=str(exc),
            )
        return FenceOutcome(
            accepted=True,
            reason="accepted",
            cell_id=envelope.cell_id,
            plan_generation=int(envelope.plan_generation),
            receipt=receipt,
        )

    def journal(self) -> list[dict]:
        return [dict(item) for item in self._journal]

    def snapshot(self) -> dict:
        return {
            "run_id": self.run_id,
            "generation": int(self._generation),
            "cancelled_generations": sorted(int(x) for x in self._cancelled),
            "cancelled_reasons": {
                str(k): str(v) for k, v in sorted(self._cancelled_reasons.items())
            },
            "journal": [dict(item) for item in self._journal],
        }


# ---------------------------------------------------------------------------
# Bounded retries
# ---------------------------------------------------------------------------


class RetryBudget:
    """A bounded retry allowance per (cell_id, plan_generation).

    A new generation gets its own small allowance (the plan changed, so a new
    reduced attempt is legitimate). Repeating the SAME failure at the SAME plan
    never loops forever: once exhausted the caller must wait for capacity or
    return a resumable diagnostic.
    """

    def __init__(self, *, max_attempts: int = DEFAULT_RETRY_ATTEMPTS) -> None:
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or int(max_attempts) < 1
        ):
            raise RecoveryError("max_attempts must be an int >= 1")
        self.max_attempts = int(max_attempts)
        self._used: dict[tuple[str, int], int] = {}
        self._journal: list[dict] = []

    def attempt(self, cell_id: str, plan_generation: int) -> int:
        key = (str(cell_id), int(plan_generation))
        used = int(self._used.get(key, 0))
        if used >= self.max_attempts:
            raise RetryExhausted(
                f"cell {key[0]} generation {key[1]} exhausted "
                f"{self.max_attempts} reduced retries; wait for real capacity",
                cell_id=key[0],
                plan_generation=key[1],
            )
        self._used[key] = used + 1
        self._journal.append(
            {
                "cell_id": key[0],
                "plan_generation": key[1],
                "attempt": used + 1,
                "max_attempts": int(self.max_attempts),
            }
        )
        return used + 1

    def used(self, cell_id: str, plan_generation: int) -> int:
        return int(self._used.get((str(cell_id), int(plan_generation)), 0))

    def remaining(self, cell_id: str, plan_generation: int) -> int:
        return max(
            0,
            int(self.max_attempts)
            - self.used(cell_id, plan_generation),
        )

    def reset(self, cell_id: str, plan_generation: int) -> None:
        self._used.pop((str(cell_id), int(plan_generation)), None)

    def snapshot(self) -> dict:
        return {
            "max_attempts": int(self.max_attempts),
            "used": {
                f"{cid}@{gen}": int(count)
                for (cid, gen), count in sorted(self._used.items())
            },
            "journal": [dict(item) for item in self._journal],
        }


# ---------------------------------------------------------------------------
# Episode coordinator
# ---------------------------------------------------------------------------


@dataclass
class RecoveryOutcome:
    state: str
    episode: int
    governing: CapacityObservation | None
    released: ReleaseReport | None
    profile: ProfileDecision | None
    wait: CapacityWaitResult | None
    reason_code: str
    detail: str

    def to_dict(self) -> dict:
        return {
            "state": str(self.state),
            "episode": int(self.episode),
            "governing": None if self.governing is None else self.governing.to_dict(),
            "released": None if self.released is None else self.released.to_dict(),
            "profile": None if self.profile is None else self.profile.to_dict(),
            "wait": None if self.wait is None else self.wait.to_dict(),
            "reason_code": str(self.reason_code),
            "detail": str(self.detail),
        }


class RecoveryCoordinator:
    """One run's resource-recovery episodes (plan §5.3/§5.4/§5.7).

    The coordinator is *policy + journal*. The caller supplies:

    * ``release`` — a callable that really stops admission, cancels/drains
      speculative work and returns a :class:`ReleaseReport`;
    * ``probe`` — the live host-capacity probe;
    * optional stop signals — the wait is cancellable at every wake-up;
    * the profile policy — one-way AUTO -> BOUNDED.

    ``recover`` enforces the order release -> fresh observation -> reduction ->
    wait, and refuses to wait when the release report is incomplete.
    """

    def __init__(
        self,
        *,
        run_id: str,
        release: Callable[[], ReleaseReport],
        probe: Callable[[], Any] = default_capacity_probe,
        profile: StorageProfilePolicy | None = None,
        stop_event: threading.Event | None = None,
        cancel_event: threading.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] | None = None,
        initial_interval_s: float = PROBE_INITIAL_INTERVAL_S,
        max_interval_s: float = PROBE_MAX_INTERVAL_S,
        required_observations: int = REQUIRED_ADMISSIBLE_OBSERVATIONS,
        min_spacing_s: float = MIN_ADMISSIBLE_SPACING_S,
        max_age_s: float | None = DEFAULT_MAX_OBSERVATION_AGE_S,
        journal_max: int = 256,
    ) -> None:
        if not callable(release):
            raise RecoveryError("release must be callable")
        if not callable(probe):
            raise RecoveryError("probe must be callable")
        if isinstance(journal_max, bool) or not isinstance(journal_max, int) or journal_max < 8:
            raise RecoveryError("journal_max must be an int >= 8")
        self.run_id = str(run_id)
        self._release = release
        self._probe = probe
        self._policy = profile or StorageProfilePolicy(
            requested=zsp.STORAGE_PROFILE_AUTO
        )
        self._stop_event = stop_event
        self._cancel_event = cancel_event
        self._clock = clock
        self._sleep = sleep
        self._initial_interval_s = float(initial_interval_s)
        self._max_interval_s = float(max_interval_s)
        self._required = int(required_observations)
        self._min_spacing_s = float(min_spacing_s)
        self._max_age_s = max_age_s
        self._journal: list[dict] = []
        self._journal_max = int(journal_max)
        self._state = STATE_ACTIVE
        self._episode = 0
        self._governing: CapacityObservation | None = None
        self._release_done = False
        self._release_report: ReleaseReport | None = None
        self._last_outcome: RecoveryOutcome | None = None
        self._stopped = False

    # -- journal -----------------------------------------------------------

    def _log(self, event: str, **fields: Any) -> None:
        if len(self._journal) >= self._journal_max:
            self._journal.pop(0)
        record = {
            "event": str(event),
            "monotonic": float(self._clock()),
            "state": self._state,
            "episode": int(self._episode),
        }
        record.update(fields)
        self._journal.append(record)

    def journal(self) -> list[dict]:
        return [dict(item) for item in self._journal]

    @property
    def state(self) -> str:
        return self._state

    @property
    def episode(self) -> int:
        return int(self._episode)

    @property
    def profile(self) -> StorageProfilePolicy:
        return self._policy

    def stop(self, reason: str = "stop requested") -> str:
        self._state = STATE_STOPPED_RESUMABLE
        self._stopped = True
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._log("stopped_resumable", reason=str(reason))
        return self._state

    def _cancelled(self) -> bool:
        if self._stopped:
            return True
        for event in (self._stop_event, self._cancel_event):
            if event is not None and event.is_set():
                return True
        return False

    # -- release -----------------------------------------------------------

    def release_speculative(self, *, phase: str, reason_code: str) -> ReleaseReport:
        """Really release before any wait; never report a state-only release."""
        report = self._release()
        if not isinstance(report, ReleaseReport):
            raise ReleaseIncompleteError(
                "release callback must return a ReleaseReport, got "
                f"{type(report).__name__}"
            )
        if report.units_released < 0 or report.released_bytes < 0:
            raise ReleaseIncompleteError("release report has negative amounts")
        self._release_report = report
        self._release_done = True
        self._log(
            "speculative_released",
            phase=str(phase),
            reason_code=str(reason_code),
            report=report.to_dict(),
        )
        return report

    # -- episode -----------------------------------------------------------

    def recover(
        self,
        *,
        facts: Any,
        phase: str,
        reason_code: str,
        detail: str,
        minimum_bytes: int,
        structural_minimum_bytes: int | None = None,
        structural_capacity_bytes: int | None = None,
        wait: bool = True,
        deadline_s: float | None = None,
        max_probes: int | None = None,
    ) -> RecoveryOutcome:
        """Run ONE recovery episode: release, observe, reduce, optionally wait.

        The governing observation is the FIRST valid fresh fact of this episode;
        it is never replaced by a later, more favourable sample during the same
        episode.
        """
        if self._cancelled():
            self._state = STATE_STOPPED_RESUMABLE
            raise CapacityWaitCancelled(
                "recovery not started: stop already requested"
            )
        self._episode += 1
        episode = int(self._episode)
        self._state = STATE_RECOVERING
        self._release_done = False
        self._release_report = None
        self._log(
            "episode_started",
            phase=str(phase),
            reason_code=str(reason_code),
            detail=str(detail),
        )
        # 1) A valid, fresh observation governs the WHOLE episode.
        governing = parse_observation(
            facts, now=float(self._clock()), max_age_s=self._max_age_s
        )
        self._governing = governing
        self._log("governing_observation", observation=governing.to_dict())

        # 2) RELEASE BEFORE WAIT. This is not optional and not just a state.
        report = self.release_speculative(
            phase=phase, reason_code=reason_code
        )
        if not report.credits_balanced and wait:
            raise ReleaseIncompleteError(
                "cannot wait for capacity while credits/staging are still held "
                f"({report.to_dict()})"
            )

        # 3) ONE-WAY profile reduction on persistent pressure.
        profile_decision = self._policy.decide(
            pressure_persists=True,
            phase=phase,
            reason_code=reason_code,
            detail=detail,
        )
        if profile_decision.switched:
            self._log("profile_switched", decision=profile_decision.to_dict())

        # 4) Structural impossibility: never wait indefinitely for a minimum
        #    that cannot fit even the whole machine.
        if (
            structural_minimum_bytes is not None
            and structural_capacity_bytes is not None
            and int(structural_minimum_bytes) > int(structural_capacity_bytes)
        ):
            self._state = STATE_CAPACITY_INFEASIBLE
            self._log(
                "capacity_infeasible",
                structural_minimum_bytes=int(structural_minimum_bytes),
                structural_capacity_bytes=int(structural_capacity_bytes),
            )
            raise StructuralCapacityRefused(
                "the structural working set "
                f"{int(structural_minimum_bytes)} B exceeds the total machine "
                f"capacity {int(structural_capacity_bytes)} B; no wait can help"
            )

        # 5) If the released working set already fits, no wait is needed.
        if int(governing.raw_available_bytes) >= int(minimum_bytes):
            self._state = STATE_ACTIVE
            self._release_done = True
            outcome = RecoveryOutcome(
                state=self._state,
                episode=episode,
                governing=governing,
                released=report,
                profile=profile_decision,
                wait=None,
                reason_code=str(reason_code),
                detail="released working set fits the governing observation",
            )
            self._last_outcome = outcome
            self._log("episode_completed_without_wait", outcome=outcome.to_dict())
            return outcome
        if not wait:
            self._state = STATE_WAITING_RESOURCES
            outcome = RecoveryOutcome(
                state=self._state,
                episode=episode,
                governing=governing,
                released=report,
                profile=profile_decision,
                wait=None,
                reason_code=str(reason_code),
                detail="insufficient capacity; wait not requested",
            )
            self._last_outcome = outcome
            return outcome

        # 6) Light, cancellable wait; readmission needs TWO spaced observations.
        self._state = STATE_WAITING_RESOURCES
        self._log(
            "waiting_resources",
            minimum_bytes=int(minimum_bytes),
            governing=governing.to_dict(),
        )
        waiter = CapacityWaiter(
            self._probe,
            minimum_bytes=int(minimum_bytes),
            stop_event=self._stop_event,
            cancel_event=self._cancel_event,
            clock=self._clock,
            sleep=self._sleep,
            initial_interval_s=self._initial_interval_s,
            max_interval_s=self._max_interval_s,
            required_observations=self._required,
            min_spacing_s=self._min_spacing_s,
            deadline_s=deadline_s,
            max_probes=max_probes,
            max_age_s=self._max_age_s,
        )
        try:
            wait_result = waiter.wait()
        except CapacityWaitCancelled:
            self._state = STATE_STOPPED_RESUMABLE
            self._log("wait_cancelled")
            raise
        except CapacityWaitDeadline as exc:
            self._state = STATE_STOPPED_RESUMABLE
            self._log("wait_deadline", result=exc.result.to_dict() if exc.result else None)
            raise
        except CapacityWaitExhausted:
            self._state = STATE_WAITING_RESOURCES
            self._log("wait_probe_budget_exhausted")
            raise
        self._state = STATE_READMITTING
        self._log("readmitting", wait=wait_result.to_dict())
        # The NEW episode may use the fresh admissible observation; it never
        # reuses the old governing minimum.
        self._governing = wait_result.last
        self._state = STATE_ACTIVE
        outcome = RecoveryOutcome(
            state=self._state,
            episode=episode,
            governing=self._governing,
            released=report,
            profile=profile_decision,
            wait=wait_result,
            reason_code=str(reason_code),
            detail="durable capacity observed twice, readmitted",
        )
        self._last_outcome = outcome
        self._log("episode_completed", outcome=outcome.to_dict())
        return outcome

    def snapshot(self) -> dict:
        return {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "state": self._state,
            "episode": int(self._episode),
            "governing": (
                None if self._governing is None else self._governing.to_dict()
            ),
            "release_done": bool(self._release_done),
            "release": (
                None if self._release_report is None
                else self._release_report.to_dict()
            ),
            "profile": self._policy.snapshot(),
            "journal": self.journal(),
            "note": (
                "WAITING_RESOURCES is visible and cancellable; it is not a "
                "worker death and never erases committed results"
            ),
        }
