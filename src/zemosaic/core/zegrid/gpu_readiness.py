"""ZM-ZEGRID-R55-C — portable GPU readiness + provisional VRAM policy (observe-only).

This module is a *fact and contract* layer. It does not activate GPU scheduling,
does not change the active ZeGrid GPU budget (``gpu.vram_budget_bytes`` remains
``free x 50%`` with its 64 MiB floor), and never makes CuPy mandatory.

Design invariants
-----------------
* **Lazy**: importing this module (or any CPU/R55 module) never imports CuPy and
  never initialises CUDA. Only an explicit :func:`probe_gpu_readiness` call does.
* **Truthful**: the readiness snapshot uses one closed status/reason vocabulary
  and distinguishes "declared dependency", "importable dependency", "usable CUDA
  device", "product GPU requested/effective" (owned elsewhere) and "R55 policy
  merely observed" (here).
* **One canonical availability definition**: device usability is delegated to
  :func:`zemosaic.core.canonical_stacking.canonical_gpu_available`; no
  Windows-only or ad-hoc check is introduced.
* **Pure integers**: the VRAM policy uses integer arithmetic only, refuses
  ``bool``/``float`` coercion, never inflates a budget up to a floor, and reports
  the explicit residual reserve.
* **Provisional**: 70% of currently free VRAM is ``provisional/observe_only``.
  It is NOT product-qualified; qualification requires real product-path HWM and
  parity evidence in a later mission.
* **Single owner / whole batch**: the RUN revalidation contract is shrink-only
  (75 -> 70 -> 65 -> 50 -> CPU), never grows a resource after pressure, and a
  GPU abandonment is an exact **whole-batch** CPU fallback with no mixed partial
  result.

Nothing here selects a plan, allocates device memory or executes work.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Sequence

try:  # Python 3.11+; the project also supports 3.10
    import tomllib
except Exception:  # pragma: no cover - 3.10 fallback
    tomllib = None  # type: ignore[assignment]

__all__ = [
    "SCHEMA",
    "POLICY_SCHEMA",
    "GPU_OPTIONAL_EXTRA",
    "STATUS_NOT_PROBED",
    "STATUS_USABLE",
    "STATUS_DEPENDENCY_MISSING",
    "STATUS_DEPENDENCY_UNIMPORTABLE",
    "STATUS_NO_DEVICE",
    "STATUS_DEVICE_PROBE_ERROR",
    "STATUSES",
    "REASON_GPU_OK",
    "REASON_GPU_NOT_PROBED",
    "REASON_GPU_DEP_MISSING",
    "REASON_GPU_DEP_UNIMPORTABLE",
    "REASON_GPU_DRIVER_UNAVAILABLE",
    "REASON_GPU_NO_DEVICE",
    "REASON_GPU_PROBE_ERROR",
    "REASON_VRAM_CANDIDATE_OK",
    "REASON_VRAM_BELOW_MINIMUM",
    "REASON_VRAM_CONTINUE",
    "REASON_VRAM_SHRINK",
    "REASON_GPU_FALLBACK_PRE",
    "REASON_REFUSE_VRAM",
    "REASON_VRAM_GROWTH_REJECTED",
    "REASONS",
    "VRAM_LADDER",
    "INITIAL_TARGET_PERCENT",
    "QUALIFICATION_LADDER",
    "MIN_VRAM_BUDGET_BYTES",
    "POLICY_PROVENANCE",
    "GpuReadiness",
    "VramPolicy",
    "VramCandidate",
    "RunGpuDecision",
    "GpuReadinessError",
    "InvalidVramObservation",
    "GrowthRejected",
    "probe_gpu_readiness",
    "probe_gpu_readiness_safe",
    "gpu_extra_declared",
    "policy_budget_bytes",
    "policy_residual_reserve_bytes",
    "shrink_only",
    "revalidate_for_unit",
    "readiness_manifest_dict",
    "canonical_contract_dict",
]

SCHEMA = "zemosaic-r55-gpu-readiness-v1"
POLICY_SCHEMA = "zemosaic-r55-vram-policy-v1"

GPU_OPTIONAL_EXTRA = "gpu"
CUPY_DISTRIBUTION = "cupy"

# ---------------------------------------------------------------------------
# Closed status vocabulary
# ---------------------------------------------------------------------------

STATUS_NOT_PROBED = "not_probed"
STATUS_USABLE = "usable"
STATUS_DEPENDENCY_MISSING = "dependency_missing"
STATUS_DEPENDENCY_UNIMPORTABLE = "dependency_unimportable"
STATUS_NO_DEVICE = "no_device"
STATUS_DEVICE_PROBE_ERROR = "device_probe_error"

STATUSES = (
    STATUS_NOT_PROBED,
    STATUS_USABLE,
    STATUS_DEPENDENCY_MISSING,
    STATUS_DEPENDENCY_UNIMPORTABLE,
    STATUS_NO_DEVICE,
    STATUS_DEVICE_PROBE_ERROR,
)

# ---------------------------------------------------------------------------
# Closed reason-code vocabulary
# ---------------------------------------------------------------------------

REASON_GPU_OK = "R55_GPU_OK"
REASON_GPU_NOT_PROBED = "R55_GPU_NOT_PROBED"
REASON_GPU_DEP_MISSING = "R55_GPU_DEP_MISSING"
REASON_GPU_DEP_UNIMPORTABLE = "R55_GPU_DEP_UNIMPORTABLE"
REASON_GPU_DRIVER_UNAVAILABLE = "R55_GPU_DRIVER_UNAVAILABLE"
REASON_GPU_NO_DEVICE = "R55_GPU_NO_DEVICE"
REASON_GPU_PROBE_ERROR = "R55_GPU_PROBE_ERROR"
REASON_VRAM_CANDIDATE_OK = "R55_VRAM_CANDIDATE_OK"
REASON_VRAM_BELOW_MINIMUM = "R55_VRAM_BELOW_MINIMUM"
REASON_VRAM_CONTINUE = "R55_VRAM_CONTINUE"
REASON_VRAM_SHRINK = "R55_VRAM_SHRINK"
REASON_GPU_FALLBACK_PRE = "R55_GPU_FALLBACK_PRE"
REASON_REFUSE_VRAM = "R55_REFUSE_VRAM"
REASON_VRAM_GROWTH_REJECTED = "R55_VRAM_GROWTH_REJECTED"

REASONS = frozenset(
    {
        REASON_GPU_OK,
        REASON_GPU_NOT_PROBED,
        REASON_GPU_DEP_MISSING,
        REASON_GPU_DEP_UNIMPORTABLE,
        REASON_GPU_DRIVER_UNAVAILABLE,
        REASON_GPU_NO_DEVICE,
        REASON_GPU_PROBE_ERROR,
        REASON_VRAM_CANDIDATE_OK,
        REASON_VRAM_BELOW_MINIMUM,
        REASON_VRAM_CONTINUE,
        REASON_VRAM_SHRINK,
        REASON_GPU_FALLBACK_PRE,
        REASON_REFUSE_VRAM,
        REASON_VRAM_GROWTH_REJECTED,
    }
)

# ---------------------------------------------------------------------------
# Provisional VRAM policy constants
# ---------------------------------------------------------------------------

# Provisional candidate ladder, ascending, in integer percent of CURRENTLY FREE
# VRAM. 70 is the initial provisional target (product-owner decision R55-C).
VRAM_LADDER = (50, 65, 70, 75)
INITIAL_TARGET_PERCENT = 70
# Progressive qualification path for a later mission (65 -> 70 -> 75).
QUALIFICATION_LADDER = (65, 70, 75)

# Same absolute floor as the active legacy planner
# (``gpu.GPU_MIN_VRAM_BUDGET_BYTES``); duplicated here so this module stays
# dependency-light. A test pins the two constants equal.
MIN_VRAM_BUDGET_BYTES = 64 * 1024 * 1024

# 70% is provisional and observe-only; it is NOT product-qualified.
POLICY_PROVENANCE = "provisional_observe_only"

_ACTION_CONTINUE = "continue"
_ACTION_SHRINK = "shrink"
_ACTION_FALLBACK_CPU = "fallback_cpu"
_ACTIONS = (_ACTION_CONTINUE, _ACTION_SHRINK, _ACTION_FALLBACK_CPU)


# ---------------------------------------------------------------------------
# Errors / integer discipline
# ---------------------------------------------------------------------------

class GpuReadinessError(ValueError):
    """Base class for readiness/policy contract violations."""


class InvalidVramObservation(GpuReadinessError):
    """A VRAM observation is missing or not a plain non-negative integer."""


class GrowthRejected(GpuReadinessError):
    """A shrink-only transition was asked to grow a resource."""


def _require_plain_int(value, name: str, *, minimum: int = 0) -> int:
    """Return ``value`` as int; refuse bool/float/str (no silent coercion)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidVramObservation(
            f"{name} must be a plain int, got {type(value).__name__}={value!r}"
        )
    if value < minimum:
        raise InvalidVramObservation(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _require_percent(value, name: str = "percent") -> int:
    p = _require_plain_int(value, name, minimum=1)
    if p > 100:
        raise InvalidVramObservation(f"{name} must be <= 100, got {p}")
    return p


# ---------------------------------------------------------------------------
# Optional-dependency declaration (packaging fact, no import)
# ---------------------------------------------------------------------------

def _pyproject_path() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None


def gpu_extra_declared() -> bool | None:
    """Whether the project declares CuPy under an optional ``gpu`` extra.

    A packaging fact, never an import: tries installed distribution metadata
    first, then the repository ``pyproject.toml``. Returns ``None`` when neither
    is readable (unknown is reported honestly rather than guessed).
    """
    try:
        requires = importlib.metadata.requires("ZeMosaic") or []
        for raw in requires:
            text = str(raw).lower()
            if "cupy" in text and "extra ==" in text and GPU_OPTIONAL_EXTRA in text:
                return True
        # Metadata was readable but no cupy extra was found.
        if requires:
            return False
    except Exception:
        pass
    path = _pyproject_path()
    if path is None or tomllib is None:
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    optional = data.get("project", {}).get("optional-dependencies", {})
    deps = optional.get(GPU_OPTIONAL_EXTRA)
    if deps is None:
        return False
    return any("cupy" in str(d).lower() for d in deps)


# ---------------------------------------------------------------------------
# Readiness snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GpuReadiness:
    """Truthful, immutable readiness snapshot.

    ``available`` is the canonical device-usability answer and is True ONLY for
    ``status == "usable"``. ``errors`` holds short classified tokens
    (``field:ExceptionClass``), never raw volatile messages.
    """

    status: str
    reason_code: str
    available: bool
    dependency_declared: bool | None
    dependency_importable: bool
    cupy_version: str | None = None
    device_count: int = 0
    device_index: int | None = None
    device_name: str | None = None
    driver_version: int | None = None
    runtime_version: int | None = None
    vram_total_bytes: int | None = None
    vram_free_bytes: int | None = None
    pool_used_bytes: int | None = None
    pool_total_bytes: int | None = None
    pinned_pool_used_bytes: int | None = None
    pinned_pool_total_bytes: int | None = None
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise GpuReadinessError(f"unknown readiness status {self.status!r}")
        if self.reason_code not in REASONS:
            raise GpuReadinessError(f"unknown reason code {self.reason_code!r}")
        if not isinstance(self.available, bool):
            raise GpuReadinessError("available must be a bool")
        if self.available != (self.status == STATUS_USABLE):
            raise GpuReadinessError(
                "available must be True iff status is 'usable' "
                f"(status={self.status!r}, available={self.available!r})"
            )
        if self.dependency_declared is not None and not isinstance(
            self.dependency_declared, bool
        ):
            raise GpuReadinessError("dependency_declared must be bool or None")
        if not isinstance(self.dependency_importable, bool):
            raise GpuReadinessError("dependency_importable must be a bool")
        for name in ("device_count",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GpuReadinessError(f"{name} must be a non-negative int")
        for name in (
            "device_index",
            "driver_version",
            "runtime_version",
            "vram_total_bytes",
            "vram_free_bytes",
            "pool_used_bytes",
            "pool_total_bytes",
            "pinned_pool_used_bytes",
            "pinned_pool_total_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise GpuReadinessError(f"{name} must be a non-negative int or None")
        if self.vram_total_bytes is not None and self.vram_free_bytes is not None:
            if self.vram_free_bytes > self.vram_total_bytes:
                raise GpuReadinessError(
                    "vram_free_bytes cannot exceed vram_total_bytes"
                )
        if not isinstance(self.errors, tuple) or any(
            not isinstance(e, str) for e in self.errors
        ):
            raise GpuReadinessError("errors must be a tuple of strings")

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "status": self.status,
            "reason_code": self.reason_code,
            "available": self.available,
            "dependency_declared": self.dependency_declared,
            "dependency_importable": self.dependency_importable,
            "cupy_version": self.cupy_version,
            "device_count": int(self.device_count),
            "device_index": self.device_index,
            "device_name": self.device_name,
            "driver_version": self.driver_version,
            "runtime_version": self.runtime_version,
            "vram_total_bytes": self.vram_total_bytes,
            "vram_free_bytes": self.vram_free_bytes,
            "pool_used_bytes": self.pool_used_bytes,
            "pool_total_bytes": self.pool_total_bytes,
            "pinned_pool_used_bytes": self.pinned_pool_used_bytes,
            "pinned_pool_total_bytes": self.pinned_pool_total_bytes,
            "errors": list(self.errors),
        }


def _classify_error_token(field: str, exc: BaseException) -> str:
    return f"{field}:{type(exc).__name__}"


def _find_cupy_spec():
    """Distribution presence without importing anything (overridable in tests)."""
    return importlib.util.find_spec(CUPY_DISTRIBUTION)


def _import_cupy():
    """The ONLY CuPy import in this module; called from an explicit probe."""
    import cupy as _cupy  # noqa: WPS433 (deliberate lazy import)

    return _cupy


def _canonical_available() -> bool:
    """One canonical availability definition (kept as a seam for tests)."""
    from zemosaic.core.canonical_stacking import canonical_gpu_available

    return bool(canonical_gpu_available())


def _probe_device_fields(cupy, device_index: int) -> tuple[dict, list[str]]:
    """Collect per-field device facts; every failure is classified, never fatal."""
    facts: dict = {}
    errors: list[str] = []
    runtime = cupy.cuda.runtime

    try:
        facts["cupy_version"] = str(cupy.__version__)
    except Exception as exc:  # defensive
        errors.append(_classify_error_token("cupy_version", exc))

    try:
        facts["device_count"] = int(runtime.getDeviceCount())
    except Exception as exc:
        errors.append(_classify_error_token("getDeviceCount", exc))
        facts["device_count"] = 0

    try:
        facts["driver_version"] = int(runtime.driverGetVersion())
    except Exception as exc:
        errors.append(_classify_error_token("driverGetVersion", exc))

    try:
        facts["runtime_version"] = int(runtime.runtimeGetVersion())
    except Exception as exc:
        errors.append(_classify_error_token("runtimeGetVersion", exc))

    try:
        props = runtime.getDeviceProperties(int(device_index))
        name = props.get("name")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        facts["device_name"] = str(name) if name else None
    except Exception as exc:
        errors.append(_classify_error_token("getDeviceProperties", exc))

    # Device-specific facts MUST be read with the requested device current;
    # the context restores the previous device on exit (and we belt-and-braces
    # restore it in ``finally``). Properties/count above may remain global.
    previous_device = None
    try:
        previous_device = int(runtime.getDevice())
    except Exception as exc:  # defensive; restore becomes best-effort
        errors.append(_classify_error_token("getDevice", exc))
    try:
        with cupy.cuda.Device(int(device_index)):
            try:
                free_bytes, total_bytes = runtime.memGetInfo()
                facts["vram_free_bytes"] = int(free_bytes)
                facts["vram_total_bytes"] = int(total_bytes)
            except Exception as exc:
                errors.append(_classify_error_token("memGetInfo", exc))
            try:
                pool = cupy.get_default_memory_pool()
                facts["pool_used_bytes"] = int(pool.used_bytes())
                facts["pool_total_bytes"] = int(pool.total_bytes())
            except Exception as exc:
                errors.append(_classify_error_token("default_pool", exc))
    except Exception as exc:
        errors.append(_classify_error_token("device_context", exc))
    finally:
        if previous_device is not None:
            try:
                if int(runtime.getDevice()) != previous_device:
                    runtime.setDevice(previous_device)
            except Exception as exc:
                errors.append(_classify_error_token("restore_device", exc))

    try:
        pinned_factory = getattr(cupy, "get_default_pinned_memory_pool", None)
        if pinned_factory is not None:
            pinned = pinned_factory()
            # CuPy's PinnedMemoryPool exposes n_free_blocks but not used/total;
            # report only what is actually available (None is truthful).
            used_fn = getattr(pinned, "used_bytes", None)
            total_fn = getattr(pinned, "total_bytes", None)
            if callable(used_fn):
                facts["pinned_pool_used_bytes"] = int(used_fn())
            if callable(total_fn):
                facts["pinned_pool_total_bytes"] = int(total_fn())
    except Exception as exc:
        errors.append(_classify_error_token("pinned_pool", exc))

    return facts, errors


def probe_gpu_readiness(
    *,
    device_index: int = 0,
    declared: bool | None = None,
) -> GpuReadiness:
    """Explicitly probe CuPy/CUDA and return a truthful readiness snapshot.

    Importing this module never calls this function. On any failure the snapshot
    is returned with a closed status/reason code and ``available=False``; the
    caller's CPU path is unaffected (nothing here raises for an unavailable GPU).
    """
    idx = _require_plain_int(device_index, "device_index")
    dependency_declared = gpu_extra_declared() if declared is None else bool(declared)

    try:
        spec = _find_cupy_spec()
    except Exception:
        spec = None
    if spec is None:
        return GpuReadiness(
            status=STATUS_DEPENDENCY_MISSING,
            reason_code=REASON_GPU_DEP_MISSING,
            available=False,
            dependency_declared=dependency_declared,
            dependency_importable=False,
        )

    try:
        cupy = _import_cupy()
    except Exception as exc:
        return GpuReadiness(
            status=STATUS_DEPENDENCY_UNIMPORTABLE,
            reason_code=REASON_GPU_DEP_UNIMPORTABLE,
            available=False,
            dependency_declared=dependency_declared,
            dependency_importable=False,
            errors=(_classify_error_token("cupy_import", exc),),
        )

    facts, errors = _probe_device_fields(cupy, idx)
    count = int(facts.get("device_count", 0))

    try:
        canonical = _canonical_available()
    except Exception as exc:  # defensive: canonical helper is itself guarded
        canonical = False
        errors.append(_classify_error_token("canonical_gpu_available", exc))

    count_failed = any(e.startswith("getDeviceCount:") for e in errors)

    if count_failed:
        status = STATUS_DEVICE_PROBE_ERROR
        reason = (
            REASON_GPU_DRIVER_UNAVAILABLE
            if any(_looks_like_driver_error_token(e) for e in errors)
            else REASON_GPU_PROBE_ERROR
        )
        available = False
    elif canonical and count > 0 and idx < count:
        status = STATUS_USABLE
        reason = REASON_GPU_OK
        available = True
    elif count == 0:
        status = STATUS_NO_DEVICE
        reason = REASON_GPU_NO_DEVICE
        available = False
    elif not canonical and count > 0:
        # Device count succeeded but the canonical check still says no: classify
        # a driver/runtime failure narrowly, otherwise report a generic probe error.
        status = STATUS_DEVICE_PROBE_ERROR
        reason = REASON_GPU_PROBE_ERROR
        available = False
    else:
        status = STATUS_DEVICE_PROBE_ERROR
        reason = REASON_GPU_PROBE_ERROR
        available = False

    if status == STATUS_DEVICE_PROBE_ERROR and not errors:
        # No device-level field failed, yet canonical availability was negative:
        # record the classified cause without inventing a message.
        errors.append("canonical_gpu_available:False")
    if reason == REASON_GPU_PROBE_ERROR and any(
        _looks_like_driver_error_token(e) for e in errors
    ):
        reason = REASON_GPU_DRIVER_UNAVAILABLE

    return GpuReadiness(
        status=status,
        reason_code=reason,
        available=available,
        dependency_declared=dependency_declared,
        dependency_importable=True,
        cupy_version=facts.get("cupy_version"),
        device_count=count,
        device_index=(idx if count > 0 and idx < count else None),
        device_name=facts.get("device_name"),
        driver_version=facts.get("driver_version"),
        runtime_version=facts.get("runtime_version"),
        vram_total_bytes=facts.get("vram_total_bytes"),
        vram_free_bytes=facts.get("vram_free_bytes"),
        pool_used_bytes=facts.get("pool_used_bytes"),
        pool_total_bytes=facts.get("pool_total_bytes"),
        pinned_pool_used_bytes=facts.get("pinned_pool_used_bytes"),
        pinned_pool_total_bytes=facts.get("pinned_pool_total_bytes"),
        errors=tuple(errors),
    )


def probe_gpu_readiness_safe(*, device_index: int = 0, declared: bool | None = None) -> GpuReadiness:
    """Contained readiness probe: never raises, always returns a closed snapshot.

    Any unexpected failure (including an invalid ``device_index`` or a broken
    packaging lookup) is reported as ``device_probe_error``/``R55_GPU_PROBE_ERROR``
    so an observe path can record it without aborting CPU work.
    """
    try:
        return probe_gpu_readiness(device_index=device_index, declared=declared)
    except Exception as exc:  # defensive containment boundary
        declared_value: bool | None
        try:
            declared_value = gpu_extra_declared() if declared is None else bool(declared)
        except Exception:
            declared_value = None
        return GpuReadiness(
            status=STATUS_DEVICE_PROBE_ERROR,
            reason_code=REASON_GPU_PROBE_ERROR,
            available=False,
            dependency_declared=declared_value,
            dependency_importable=False,
            errors=(_classify_error_token("probe_gpu_readiness", exc),),
        )


def _looks_like_driver_error_token(token: str) -> bool:
    lowered = token.lower()
    return any(
        key in lowered
        for key in ("driver", "cudart", "cudaapi", "cudaruntime", "cudaerror")
    )


# ---------------------------------------------------------------------------
# Pure provisional VRAM policy
# ---------------------------------------------------------------------------

def policy_budget_bytes(free_bytes: int, percent: int) -> int:
    """``floor(free_bytes * percent / 100)`` using integer arithmetic only."""
    free = _require_plain_int(free_bytes, "free_bytes")
    p = _require_percent(percent)
    return (free * p) // 100


def policy_residual_reserve_bytes(free_bytes: int, percent: int) -> int:
    """Bytes of the observed free VRAM deliberately left unclaimed."""
    free = _require_plain_int(free_bytes, "free_bytes")
    return free - policy_budget_bytes(free, percent)


@dataclass(frozen=True)
class VramCandidate:
    """One rung of the provisional policy ladder for a single observation."""

    percent: int
    free_bytes: int
    budget_bytes: int
    residual_reserve_bytes: int
    eligible: bool
    reason_code: str
    provenance: str = POLICY_PROVENANCE
    observe_only: bool = True

    def __post_init__(self) -> None:
        p = _require_percent(self.percent)
        free = _require_plain_int(self.free_bytes, "free_bytes")
        budget = _require_plain_int(self.budget_bytes, "budget_bytes")
        residual = _require_plain_int(
            self.residual_reserve_bytes, "residual_reserve_bytes"
        )
        if budget != policy_budget_bytes(free, p):
            raise GpuReadinessError(
                f"candidate budget {budget} != floor({free}*{p}/100)"
            )
        if residual != free - budget:
            raise GpuReadinessError("residual reserve does not match free - budget")
        if not isinstance(self.eligible, bool):
            raise GpuReadinessError("eligible must be a bool")
        if self.reason_code not in REASONS:
            raise GpuReadinessError(f"unknown reason code {self.reason_code!r}")
        if self.provenance != POLICY_PROVENANCE:
            raise GpuReadinessError(
                f"R55-C only emits provisional policy, got {self.provenance!r}"
            )
        if self.observe_only is not True:
            raise GpuReadinessError("R55-C policy candidates are observe-only")

    def to_dict(self) -> dict:
        return {
            "percent": int(self.percent),
            "free_bytes": int(self.free_bytes),
            "budget_bytes": int(self.budget_bytes),
            "residual_reserve_bytes": int(self.residual_reserve_bytes),
            "eligible": self.eligible,
            "reason_code": self.reason_code,
            "provenance": self.provenance,
            "observe_only": self.observe_only,
        }


@dataclass(frozen=True)
class VramPolicy:
    """Provisional, pure, integer-only VRAM budget policy.

    The ladder is evaluated against ONE fresh ``free_bytes`` observation.
    Nothing is cached, so a policy cannot be reused with a stale observation.
    """

    ladder: tuple[int, ...] = VRAM_LADDER
    initial_percent: int = INITIAL_TARGET_PERCENT
    minimum_budget_bytes: int = MIN_VRAM_BUDGET_BYTES
    # Optional reserve gate: a rung is eligible only when the bytes left
    # unclaimed are at least this many. Defaults to 0 (report-only reserve).
    minimum_reserve_bytes: int = 0
    provenance: str = POLICY_PROVENANCE

    def __post_init__(self) -> None:
        if isinstance(self.ladder, (str, bytes)) or not isinstance(
            self.ladder, Sequence
        ):
            raise GpuReadinessError("ladder must be a sequence of ints")
        ladder = tuple(self.ladder)
        if not ladder:
            raise GpuReadinessError("ladder must not be empty")
        for p in ladder:
            _require_percent(p, "ladder entry")
        if tuple(sorted(set(ladder))) != ladder:
            raise GpuReadinessError("ladder must be strictly ascending and unique")
        if self.initial_percent not in ladder:
            raise GpuReadinessError("initial_percent must be on the ladder")
        _require_plain_int(self.minimum_budget_bytes, "minimum_budget_bytes")
        _require_plain_int(self.minimum_reserve_bytes, "minimum_reserve_bytes")
        if self.provenance != POLICY_PROVENANCE:
            raise GpuReadinessError(
                f"R55-C policy is provisional/observe-only, got {self.provenance!r}"
            )
        object.__setattr__(self, "ladder", ladder)

    # -- pure evaluation ---------------------------------------------------
    def candidate(self, free_bytes: int, percent: int) -> VramCandidate:
        p = _require_percent(percent)
        if p not in self.ladder:
            raise GpuReadinessError(f"percent {p} is not on the policy ladder")
        free = _require_plain_int(free_bytes, "free_bytes")
        budget = policy_budget_bytes(free, p)
        residual = free - budget
        eligible = (
            budget >= self.minimum_budget_bytes
            and budget > 0
            and residual >= self.minimum_reserve_bytes
        )
        return VramCandidate(
            percent=p,
            free_bytes=free,
            budget_bytes=budget,
            residual_reserve_bytes=residual,
            eligible=eligible,
            reason_code=(
                REASON_VRAM_CANDIDATE_OK if eligible else REASON_VRAM_BELOW_MINIMUM
            ),
        )

    def candidates(self, free_bytes: int) -> tuple[VramCandidate, ...]:
        free = _require_plain_int(free_bytes, "free_bytes")
        return tuple(self.candidate(free, p) for p in self.ladder)

    def initial(self, free_bytes: int) -> VramCandidate:
        """The initial provisional target (70%) for a fresh observation."""
        return self.candidate(free_bytes, self.initial_percent)

    def next_shrink(self, current_percent: int, free_bytes: int) -> VramCandidate | None:
        """Next rung strictly below ``current_percent`` (or None at the bottom)."""
        current = _require_percent(current_percent)
        if current not in self.ladder:
            raise GpuReadinessError(f"percent {current} is not on the policy ladder")
        lower = [p for p in self.ladder if p < current]
        if not lower:
            return None
        return self.candidate(free_bytes, max(lower))

    def qualification_ladder(self) -> tuple[int, ...]:
        return tuple(p for p in QUALIFICATION_LADDER if p in self.ladder)


def shrink_only(current_percent: int, requested_percent: int) -> int:
    """Return the shrink-only result; raise :class:`GrowthRejected` on growth."""
    current = _require_percent(current_percent)
    requested = _require_percent(requested_percent)
    if requested > current:
        raise GrowthRejected(
            f"shrink-only transition cannot grow {current}% -> {requested}%"
        )
    return requested


@dataclass(frozen=True)
class RunGpuDecision:
    """Pure RUN-time revalidation decision for one bounded GPU unit."""

    action: str
    percent: int | None
    budget_bytes: int | None
    free_bytes: int
    reason_code: str
    whole_batch: bool = True
    mixed_partial: bool = False
    provenance: str = POLICY_PROVENANCE

    def __post_init__(self) -> None:
        if self.action not in _ACTIONS:
            raise GpuReadinessError(f"unknown action {self.action!r}")
        if self.reason_code not in REASONS:
            raise GpuReadinessError(f"unknown reason code {self.reason_code!r}")
        _require_plain_int(self.free_bytes, "free_bytes")
        if self.action == _ACTION_FALLBACK_CPU:
            if self.percent is not None or self.budget_bytes is not None:
                raise GpuReadinessError("CPU fallback carries no GPU budget")
            if self.whole_batch is not True or self.mixed_partial is not False:
                raise GpuReadinessError(
                    "GPU abandonment is a whole-batch exact CPU fallback"
                )
        else:
            _require_percent(self.percent, "percent")
            _require_plain_int(self.budget_bytes, "budget_bytes")

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "percent": self.percent,
            "budget_bytes": self.budget_bytes,
            "free_bytes": int(self.free_bytes),
            "reason_code": self.reason_code,
            "whole_batch": self.whole_batch,
            "mixed_partial": self.mixed_partial,
            "provenance": self.provenance,
        }


def revalidate_for_unit(
    policy: VramPolicy,
    current_percent: int,
    free_bytes: int,
    *,
    device_usable: bool = True,
    required_bytes: int | None = None,
    required_bytes_by_percent: Mapping[int, int] | Callable[[int], int] | None = None,
) -> RunGpuDecision:
    """Pure RUN revalidation: fresh observation, shrink-only, whole-batch fallback.

    ``free_bytes`` MUST be a fresh integer observation (no cached/stale value,
    no ``None``, no float/bool). The decision never grows a budget.

    Requirement modes (mutually exclusive):

    * ``required_bytes_by_percent`` — the correct shrink model: a mapping (or
      callable) giving the SMALLER tile/workspace requirement at each rung. Each
      rung is evaluated against its own requirement, so a lower rung can succeed
      precisely because its workspace requirement shrank.
    * ``required_bytes`` — one fixed requirement. Descending percent only reduces
      the budget, so a miss CANNOT be fixed by a lower rung: the decision is an
      immediate whole-batch CPU fallback, never a misleading ``shrink``.
    * neither — reserve-only mode: descending percent can restore eligibility
      only through the reserve floor (and the minimum budget), never through a
      workspace requirement.

    When no rung works the whole batch falls back to exact CPU with no mixed
    partial result.
    """
    if not isinstance(policy, VramPolicy):
        raise GpuReadinessError("policy must be a VramPolicy")
    current = _require_percent(current_percent)
    if current not in policy.ladder:
        raise GpuReadinessError(f"percent {current} is not on the policy ladder")
    free = _require_plain_int(free_bytes, "free_bytes")  # fresh observation required
    if not isinstance(device_usable, bool):
        raise GpuReadinessError("device_usable must be a bool")
    if required_bytes is not None and required_bytes_by_percent is not None:
        raise GpuReadinessError(
            "supply either required_bytes or required_bytes_by_percent, not both"
        )

    fixed: int | None = None
    if required_bytes is not None:
        fixed = _require_plain_int(required_bytes, "required_bytes", minimum=1)

    def _req(percent: int) -> int | None:
        spec = required_bytes_by_percent
        if spec is None:
            return fixed
        if isinstance(spec, Mapping):
            if set(spec.keys()) != set(policy.ladder):
                raise GpuReadinessError(
                    "required_bytes_by_percent must map every ladder rung exactly"
                )
            return _require_plain_int(
                spec[percent], f"required_bytes_by_percent[{percent}]", minimum=1
            )
        if callable(spec):
            return _require_plain_int(
                spec(percent), f"required_bytes_by_percent({percent})", minimum=1
            )
        raise GpuReadinessError(
            "required_bytes_by_percent must be a mapping or a callable"
        )

    # Validate the full requirement spec up-front (fail closed, not mid-walk).
    for p in policy.ladder:
        _req(p)

    def _works(candidate, required: int | None) -> bool:
        if not candidate.eligible:
            return False
        if required is not None and candidate.budget_bytes < required:
            return False
        return True

    if not device_usable:
        return RunGpuDecision(
            action=_ACTION_FALLBACK_CPU,
            percent=None,
            budget_bytes=None,
            free_bytes=free,
            reason_code=REASON_GPU_FALLBACK_PRE,
        )

    candidate = policy.candidate(free, current)
    if _works(candidate, _req(current)):
        return RunGpuDecision(
            action=_ACTION_CONTINUE,
            percent=candidate.percent,
            budget_bytes=candidate.budget_bytes,
            free_bytes=free,
            reason_code=REASON_VRAM_CONTINUE,
        )

    # A fixed workspace requirement cannot be rescued by a smaller budget.
    if fixed is not None:
        return RunGpuDecision(
            action=_ACTION_FALLBACK_CPU,
            percent=None,
            budget_bytes=None,
            free_bytes=free,
            reason_code=REASON_REFUSE_VRAM,
        )

    for p in sorted((x for x in policy.ladder if x < current), reverse=True):
        lower = policy.candidate(free, p)
        if _works(lower, _req(p)):
            return RunGpuDecision(
                action=_ACTION_SHRINK,
                percent=lower.percent,
                budget_bytes=lower.budget_bytes,
                free_bytes=free,
                reason_code=REASON_VRAM_SHRINK,
            )

    return RunGpuDecision(
        action=_ACTION_FALLBACK_CPU,
        percent=None,
        budget_bytes=None,
        free_bytes=free,
        reason_code=REASON_REFUSE_VRAM,
    )


# ---------------------------------------------------------------------------
# Deterministic observe-only manifest / canonical contract evidence
# ---------------------------------------------------------------------------

def _policy_contract_dict(policy: VramPolicy) -> dict:
    return {
        "schema": POLICY_SCHEMA,
        "ladder": [int(p) for p in policy.ladder],
        "initial_percent": int(policy.initial_percent),
        "qualification_ladder": [int(p) for p in policy.qualification_ladder()],
        "minimum_budget_bytes": int(policy.minimum_budget_bytes),
        "minimum_reserve_bytes": int(policy.minimum_reserve_bytes),
        "provenance": policy.provenance,
        "observe_only": True,
        "product_qualified": False,
        "execution_authority": False,
        "formula": "budget_bytes = floor(free_bytes * percent / 100)",
        "observation": "current free VRAM, fresh per bounded unit",
        "shrink_order": "75 -> 70 -> 65 -> 50 -> cpu",
        "growth_after_pressure": False,
        "fallback_scope": "whole_batch",
        "mixed_partial_result": False,
        "single_owner": True,
    }


def readiness_manifest_dict(
    readiness: GpuReadiness | None = None,
    *,
    policy: VramPolicy | None = None,
    free_bytes: int | None = None,
    requested_backend: str | None = None,
    effective_backend: str | None = None,
) -> dict:
    """Deterministic, additive observe-only GPU block.

    Contains no timestamp and no volatile hardware reading unless the caller
    explicitly passes an observation. Structure (key set and ordering) is stable
    across processes, so it can be SHA-pinned by the contract evidence tool.
    """
    pol = VramPolicy() if policy is None else policy
    if not isinstance(pol, VramPolicy):
        raise GpuReadinessError("policy must be a VramPolicy")
    if readiness is not None and not isinstance(readiness, GpuReadiness):
        raise GpuReadinessError("readiness must be a GpuReadiness or None")
    for name, value in (
        ("requested_backend", requested_backend),
        ("effective_backend", effective_backend),
    ):
        if value is not None and value not in ("cpu", "gpu"):
            raise GpuReadinessError(f"{name} must be 'cpu', 'gpu' or None")
    available = bool(readiness.available) if readiness is not None else False

    if readiness is None:
        readiness_block: dict = {
            "schema": SCHEMA,
            "status": STATUS_NOT_PROBED,
            "reason_code": REASON_GPU_NOT_PROBED,
            "available": False,
            "dependency_declared": None,
            "dependency_importable": False,
            "observed": False,
        }
    else:
        readiness_block = dict(readiness.to_dict())
        readiness_block["observed"] = True

    if free_bytes is None:
        candidates: list[dict] = []
        initial = None
    else:
        free = _require_plain_int(free_bytes, "free_bytes")
        candidates = [c.to_dict() for c in pol.candidates(free)]
        initial = pol.initial(free).to_dict()

    return {
        "schema": "zemosaic-r55-gpu-observe-v1",
        "observe_only": True,
        "execution_authority": False,
        "readiness": readiness_block,
        "product": {
            "requested_backend": requested_backend,
            "available": available,
            "effective_backend": effective_backend,
            "policy_applied": False,
        },
        "policy": _policy_contract_dict(pol),
        "initial_candidate": initial,
        "candidates": candidates,
        "observed_metrics": None,
        "note": (
            "R55-C observe-only: GPU readiness facts and a provisional VRAM "
            "policy. No GPU scheduling is activated and no observed product "
            "metric is recorded."
        ),
    }


def canonical_contract_dict() -> dict:
    """Deterministic contract evidence (policy + closed vocabularies).

    Independent of hardware: safe for byte-identity checks across two builds.
    """
    return {
        "schema": SCHEMA,
        "statuses": list(STATUSES),
        "reasons": sorted(REASONS),
        "policy": _policy_contract_dict(VramPolicy()),
        "readiness_unprobed": readiness_manifest_dict(None)["readiness"],
        "example_observation": {
            "free_bytes": 1073741824,
            "candidates": [
                c.to_dict() for c in VramPolicy().candidates(1073741824)
            ],
        },
        "dependency": {
            "optional_extra": GPU_OPTIONAL_EXTRA,
            "distribution": CUPY_DISTRIBUTION,
            "mandatory": False,
            "lazy_probe_only": True,
        },
    }
