"""ZM-ZEGRID-R55-E1 — CPU-safe RUN checkpoints and telemetry.

This module is the narrow RUN-time seam added by E1. It contains **no**
scheduler authority of its own: the caller (``zemosaic_zegrid_mode``) owns the
decision, the shrink order and the science. Here we only provide:

* strict integer validation at the RUN boundary (bool/float/negative refused);
* a CPU-safe host / parent / cgroup v2 / disk / OS-I/O sampler;
* a pressure latch and a bounded (<= 5 s) observational sampler that can only
  latch facts for the NEXT safe wave boundary (it never mutates or kills a
  running scientific unit);
* deterministic wave partitioning helpers;
* honest telemetry aggregation and a normalized deterministic projection.

The module never imports CuPy, never initializes CUDA and never touches the
GUI telemetry CSV. It is safe to import on the CPU legacy/observe paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import ram_contract as rcontract

RUN_CONTROL_SCHEMA = "zemosaic-r55e1-run-control-v1"
RUN_TELEMETRY_SCHEMA = "zemosaic-r55e1-run-telemetry-v1"
RUN_DECISION_SCHEMA = "zemosaic-r55e1-run-decision-v1"
RUN_EVIDENCE_SCHEMA = "zemosaic-r55e1-run-evidence-v1"

#: A sampler may never wait longer than this between two observations.
RUN_MAX_SAMPLE_INTERVAL_S = 5.0

#: Observational keys are excluded from every deterministic projection/hash.
OBSERVATIONAL_KEYS = frozenset(
    {
        "sampled_monotonic_ns",
        "started_monotonic_ns",
        "finished_monotonic_ns",
        "wall_s",
        "total_wall_s",
        "parent_rss_bytes",
        "parent_peak_rss_bytes",
        "cgroup_current_bytes",
        "cgroup_peak_bytes",
        "disk_io_read_bytes",
        "disk_io_write_bytes",
        "worker_peak_rss_kib_values",
        "worker_peak_sum_kib",
        "worker_peak_max_kib",
        "latched_raw_available_bytes",
        "latched_budget_bytes",
        "latched_monotonic_ns",
        "interval_s",
        "samples",
        "errors",
    }
)

DEFAULT_MEMINFO_PATH = "/proc/meminfo"
DEFAULT_PROC_STATUS_PATH = "/proc/self/status"
DEFAULT_CGROUP_ROOT = "/sys/fs/cgroup"
DEFAULT_PROC_IO_PATH = "/proc/self/io"


class RuntimeControlError(Exception):
    """A RUN fact is missing or malformed; active control fails closed."""


# ---------------------------------------------------------------------------
# Strict integer trust boundary
# ---------------------------------------------------------------------------


def strict_int(
    value: Any,
    field_name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return ``value`` as an exact int or raise; never coerce bool/float/str."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeControlError(
            f"{field_name} must be a strict int, got "
            f"{type(value).__name__}={value!r}"
        )
    result = int(value)
    if minimum is not None and result < int(minimum):
        raise RuntimeControlError(
            f"{field_name} must be >= {minimum}, got {result}"
        )
    if maximum is not None and result > int(maximum):
        raise RuntimeControlError(
            f"{field_name} must be <= {maximum}, got {result}"
        )
    return result


def run_budget_bytes(raw_available_bytes: int) -> int:
    """Exact integer 90 % RUN budget (shared with the ADMIT contract)."""
    raw = strict_int(raw_available_bytes, "MemAvailable", minimum=1)
    budget = int(rcontract.admit_budget_bytes(raw))
    expected = (rcontract.ADMIT_FACTOR_NUM * raw) // rcontract.ADMIT_FACTOR_DEN
    if budget != expected:  # pragma: no cover - contract drift guard
        raise RuntimeControlError(
            "RUN budget disagrees with the ADMIT contract: "
            f"{budget} != {expected}"
        )
    return int(budget)


# ---------------------------------------------------------------------------
# Samplers (all CPU-safe; optional facts are null + reason, never invented 0)
# ---------------------------------------------------------------------------


def _unavailable(reason: str) -> dict:
    return {"value": None, "unavailable_reason": str(reason)}


def _read_single_int(path: Path, field_name: str) -> int:
    try:
        text = path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise RuntimeControlError(f"{field_name} unavailable: {exc}") from exc
    return strict_int(int(text, 10), field_name, minimum=0)


def read_meminfo(path: str | os.PathLike[str] = DEFAULT_MEMINFO_PATH) -> dict:
    """Read ``MemAvailable`` (and total, when present) as strict bytes."""
    p = Path(path)
    try:
        lines = p.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise RuntimeControlError(
            f"host MemAvailable unavailable ({p.name}): {exc}"
        ) from exc
    values: dict[str, int] = {}
    wanted = ("MemAvailable", "MemTotal")
    for line in lines:
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        if key not in wanted:
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            kib = int(parts[0], 10)
        except ValueError as exc:
            raise RuntimeControlError(
                f"host {key} is not an integer in {p.name}"
            ) from exc
        if kib < 0:
            raise RuntimeControlError(
                f"host {key} is negative in {p.name}"
            )
        if parts[-1].lower() == "kb":
            kib *= 1024
        values[key] = int(kib)
    if "MemAvailable" not in values:
        raise RuntimeControlError(
            f"host MemAvailable is missing from {p.name}"
        )
    result = {
        "raw_available_bytes": int(values["MemAvailable"]),
        "source": f"proc:{p.name}",
    }
    if "MemTotal" in values:
        result["total_bytes"] = int(values["MemTotal"])
    return result


def read_parent_memory(
    path: str | os.PathLike[str] = DEFAULT_PROC_STATUS_PATH,
) -> dict:
    """Current RSS and VmHWM peak of the PARENT process (diagnostic)."""
    p = Path(path)
    try:
        lines = p.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        return {
            "current_rss_bytes": None,
            "peak_rss_bytes": None,
            "unavailable_reason": f"proc_status_unavailable: {exc}",
        }
    raw: dict[str, int] = {}
    wanted = ("VmRSS", "VmHWM")
    for line in lines:
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        if key not in wanted:
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            value = int(parts[0], 10)
        except ValueError:
            return {
                "current_rss_bytes": None,
                "peak_rss_bytes": None,
                "unavailable_reason": f"malformed_proc_status: {key}",
            }
        if value < 0:
            return {
                "current_rss_bytes": None,
                "peak_rss_bytes": None,
                "unavailable_reason": f"negative_proc_status: {key}",
            }
        if parts[-1].lower() == "kb":
            value *= 1024
        raw[key] = int(value)
    current = raw.get("VmRSS")
    peak = raw.get("VmHWM")
    if current is None and peak is None:
        return {
            "current_rss_bytes": None,
            "peak_rss_bytes": None,
            "unavailable_reason": "proc_status_missing_VmRSS_and_VmHWM",
        }
    result = {
        "current_rss_bytes": None if current is None else int(current),
        "peak_rss_bytes": None if peak is None else int(peak),
        "unavailable_reason": (
            None
            if current is not None and peak is not None
            else "proc_status_missing_one_metric"
        ),
    }
    return result


def read_cgroup_memory(
    root: str | os.PathLike[str] = DEFAULT_CGROUP_ROOT,
) -> dict:
    """cgroup v2 ``memory.current`` / ``memory.peak`` when readable."""
    base = Path(root)
    result = {
        "current_bytes": None,
        "peak_bytes": None,
        "unavailable_reason": None,
    }
    reasons: list[str] = []
    for filename, key in (
        ("memory.current", "current_bytes"),
        ("memory.peak", "peak_bytes"),
    ):
        path = base / filename
        try:
            result[key] = _read_single_int(path, f"cgroup_v2_{filename}")
        except RuntimeControlError as exc:
            reasons.append(str(exc))
    if reasons:
        result["unavailable_reason"] = "; ".join(reasons)
    return result


def read_disk_io(
    path: str | os.PathLike[str] = DEFAULT_PROC_IO_PATH,
) -> dict:
    """OS I/O counters from ``/proc/self/io`` (diagnostic)."""
    p = Path(path)
    try:
        lines = p.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        return {
            "read_bytes": None,
            "write_bytes": None,
            "unavailable_reason": f"proc_io_unavailable: {exc}",
        }
    values: dict[str, int] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        parts = rest.split()
        if not parts:
            continue
        try:
            value = int(parts[0], 10)
        except ValueError:
            return {
                "read_bytes": None,
                "write_bytes": None,
                "unavailable_reason": f"malformed_proc_io: {key.strip()}",
            }
        if value < 0:
            return {
                "read_bytes": None,
                "write_bytes": None,
                "unavailable_reason": f"negative_proc_io: {key.strip()}",
            }
        values[key.strip()] = int(value)
    return {
        "read_bytes": values.get("read_bytes"),
        "write_bytes": values.get("write_bytes"),
        "unavailable_reason": (
            None
            if "read_bytes" in values and "write_bytes" in values
            else "proc_io_missing_counters"
        ),
    }


def read_disk_free(root: str | os.PathLike[str]) -> int:
    """Free bytes of the filesystem holding ``root`` (admission fact)."""
    try:
        usage = shutil.disk_usage(str(root))
    except OSError as exc:
        raise RuntimeControlError(
            f"disk free unavailable for the run root: {exc}"
        ) from exc
    return strict_int(int(usage.free), "disk_free_bytes", minimum=0)


def collect_run_facts(
    *,
    root: str | os.PathLike[str],
    available_fn: Callable[[], int] | None = None,
    meminfo_path: str | os.PathLike[str] = DEFAULT_MEMINFO_PATH,
    cgroup_root: str | os.PathLike[str] = DEFAULT_CGROUP_ROOT,
    proc_status_path: str | os.PathLike[str] = DEFAULT_PROC_STATUS_PATH,
    proc_io_path: str | os.PathLike[str] = DEFAULT_PROC_IO_PATH,
    include_diagnostics: bool = True,
) -> dict:
    """Collect one strict RUN checkpoint fact set.

    ``MemAvailable`` and disk free are admission facts: malformed/unavailable
    values raise :class:`RuntimeControlError` (the caller refuses). Parent RSS,
    cgroup v2 and OS I/O are diagnostic: they degrade to ``None`` + a reason and
    never alter a decision.
    """
    if available_fn is None:
        host = read_meminfo(meminfo_path)
    else:
        raw = strict_int(available_fn(), "MemAvailable", minimum=1)
        host = {"raw_available_bytes": int(raw), "source": "injected"}
    raw = strict_int(host["raw_available_bytes"], "MemAvailable", minimum=1)
    budget = run_budget_bytes(raw)
    disk_free = read_disk_free(root)
    result = {
        "schema": RUN_CONTROL_SCHEMA,
        "host": {
            "raw_available_bytes": int(raw),
            "budget_bytes": int(budget),
            "factor_num": int(rcontract.ADMIT_FACTOR_NUM),
            "factor_den": int(rcontract.ADMIT_FACTOR_DEN),
            "source": str(host.get("source", "unknown")),
        },
        "disk": {
            "free_bytes": int(disk_free),
            "unavailable_reason": None,
        },
        "sampled_monotonic_ns": int(time.monotonic_ns()),
    }
    if "total_bytes" in host:
        result["host"]["total_bytes"] = int(host["total_bytes"])
    if include_diagnostics:
        result["parent"] = read_parent_memory(proc_status_path)
        result["cgroup"] = read_cgroup_memory(cgroup_root)
        io = read_disk_io(proc_io_path)
        result["disk"]["io_read_bytes"] = io["read_bytes"]
        result["disk"]["io_write_bytes"] = io["write_bytes"]
        if io["unavailable_reason"] and result["disk"]["unavailable_reason"] is None:
            # The free-bytes fact is available; only the optional I/O counters
            # are unavailable. Keep them separate and explicit.
            result["disk"]["io_unavailable_reason"] = io["unavailable_reason"]
    else:
        result["parent"] = {
            "current_rss_bytes": None,
            "peak_rss_bytes": None,
            "unavailable_reason": "diagnostics_disabled",
        }
        result["cgroup"] = {
            "current_bytes": None,
            "peak_bytes": None,
            "unavailable_reason": "diagnostics_disabled",
        }
        result["disk"]["io_read_bytes"] = None
        result["disk"]["io_write_bytes"] = None
        result["disk"]["io_unavailable_reason"] = "diagnostics_disabled"
    return result


def validate_checkpoint_facts(facts: Mapping[str, Any]) -> dict:
    """Normalize/validate injected facts; malformed facts fail closed."""
    if not isinstance(facts, Mapping):
        raise RuntimeControlError(
            f"RUN checkpoint facts must be a mapping, got {type(facts).__name__}"
        )
    host = facts.get("host")
    if not isinstance(host, Mapping):
        raise RuntimeControlError("RUN checkpoint facts.host is missing")
    raw = strict_int(host.get("raw_available_bytes"), "MemAvailable", minimum=1)
    budget = strict_int(host.get("budget_bytes"), "RUN budget", minimum=0)
    expected = run_budget_bytes(raw)
    if budget != expected:
        raise RuntimeControlError(
            f"RUN budget {budget} != exact 90% budget {expected} for raw {raw}"
        )
    fnum = strict_int(host.get("factor_num"), "factor_num", minimum=1)
    fden = strict_int(host.get("factor_den"), "factor_den", minimum=1)
    if (fnum, fden) != (
        int(rcontract.ADMIT_FACTOR_NUM),
        int(rcontract.ADMIT_FACTOR_DEN),
    ):
        raise RuntimeControlError(
            f"RUN budget factor must be {rcontract.ADMIT_FACTOR_NUM}/"
            f"{rcontract.ADMIT_FACTOR_DEN}, got {fnum}/{fden}"
        )
    disk = facts.get("disk")
    if not isinstance(disk, Mapping):
        raise RuntimeControlError("RUN checkpoint facts.disk is missing")
    disk_free = strict_int(disk.get("free_bytes"), "disk_free_bytes", minimum=0)
    result = dict(facts)
    result["host"] = dict(host)
    result["host"]["raw_available_bytes"] = int(raw)
    result["host"]["budget_bytes"] = int(budget)
    result["host"]["factor_num"] = int(fnum)
    result["host"]["factor_den"] = int(fden)
    result["disk"] = dict(disk)
    result["disk"]["free_bytes"] = int(disk_free)
    return result


# ---------------------------------------------------------------------------
# Pressure latch + bounded observational sampler
# ---------------------------------------------------------------------------


def validate_latched_fact(sample: Any) -> dict:
    """Strictly validate one latched fact; malformed facts fail closed.

    The stored budget is checked against the exact 90 % contract rather than
    trusted. This is the single validation used by both the latch and the RUN
    checkpoint (an injected malformed latch must refuse, not slip through).
    """
    if not isinstance(sample, Mapping):
        raise RuntimeControlError(
            "latched RUN fact must be a mapping, got "
            f"{type(sample).__name__}"
        )
    raw = strict_int(
        sample.get("raw_available_bytes"),
        "latched MemAvailable",
        minimum=1,
    )
    expected = run_budget_bytes(raw)
    if "budget_bytes" not in sample or sample.get("budget_bytes") is None:
        raise RuntimeControlError(
            "latched RUN fact is missing the exact 90% budget"
        )
    stored_int = strict_int(sample.get("budget_bytes"), "latched RUN budget", minimum=0)
    if stored_int != expected:
        raise RuntimeControlError(
            "latched RUN budget "
            f"{stored_int} != exact 90% budget {expected} for raw {raw}"
        )
    return {
        "raw_available_bytes": int(raw),
        "budget_bytes": int(expected),
        "monotonic_ns": int(sample.get("monotonic_ns", 0) or 0),
    }


class PressureLatch:
    """Thread-safe one-slot latch of the LOWEST valid pressure observation.

    The sampler may write during a wave; the caller consumes only at the next
    wave boundary. A malformed admission fact (or a collector failure) is a
    PENDING FATAL latch error: a later valid sample does NOT erase it. The
    consumer atomically takes the lowest valid sample and any pending error,
    raising on the error and clearing that consumed generation.

    Nothing here can mutate a task, a result or a science array.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sample: dict | None = None
        self._count = 0
        self._fatal_errors: list[str] = []

    def record(self, sample: Mapping[str, Any]) -> None:
        """Record one in-wave sample; malformed facts become fatal latches."""
        try:
            if not isinstance(sample, Mapping):
                raise RuntimeControlError(
                    "latched sample must be a mapping, got "
                    f"{type(sample).__name__}"
                )
            host = sample.get("host", sample)
            if not isinstance(host, Mapping):
                raise RuntimeControlError(
                    "latched sample.host must be a mapping"
                )
            entry = validate_latched_fact(host)
            entry["monotonic_ns"] = int(
                sample.get("sampled_monotonic_ns", 0) or 0
            )
        except Exception as exc:  # noqa: BLE001 - preserved as a fatal latch
            self.record_error(exc)
            return
        with self._lock:
            self._count += 1
            if (
                self._sample is None
                or int(entry["raw_available_bytes"])
                < int(self._sample["raw_available_bytes"])
            ):
                self._sample = entry

    def record_error(self, exc: BaseException | str) -> None:
        """Preserve a fatal in-wave admission/collection error for the boundary."""
        message = (
            exc if isinstance(exc, str)
            else f"{type(exc).__name__}: {exc}"
        )
        with self._lock:
            self._count += 1
            if len(self._fatal_errors) < 8:
                self._fatal_errors.append(str(message))

    def peek(self) -> dict | None:
        with self._lock:
            return None if self._sample is None else dict(self._sample)

    def has_pending_error(self) -> bool:
        with self._lock:
            return bool(self._fatal_errors)

    def consume_validated(self) -> dict | None:
        """Atomically consume the lowest valid sample AND pending fatal errors.

        If a pending malformed/admission error exists, raise
        :class:`RuntimeControlError` and clear that consumed generation. The
        following consume returns ``None`` absent new samples/errors.
        """
        with self._lock:
            sample = self._sample
            errors = list(self._fatal_errors)
            self._sample = None
            self._fatal_errors = []
        if errors:
            raise RuntimeControlError(
                "pending latched RUN fact error: " + "; ".join(errors)
            )
        if sample is None:
            return None
        return validate_latched_fact(sample)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "samples": int(self._count),
                "lowest": None if self._sample is None else dict(self._sample),
                "errors": list(self._fatal_errors),
                "fatal_errors": list(self._fatal_errors),
                "pending_fatal": bool(self._fatal_errors),
            }


class RunSampler:
    """Bounded observational sampler (<= ``RUN_MAX_SAMPLE_INTERVAL_S``).

    The sampler runs in a daemon thread, records into a :class:`PressureLatch`,
    and is always joined before the wave returns. Exceptions are recorded and
    never propagate into the scientific unit.
    """

    def __init__(
        self,
        sample_fn: Callable[[], Mapping[str, Any]],
        *,
        latch: PressureLatch | None = None,
        interval_s: float = RUN_MAX_SAMPLE_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], bool] | None = None,
    ) -> None:
        if (
            isinstance(interval_s, bool)
            or not isinstance(interval_s, (int, float))
            or not (0 < float(interval_s) <= RUN_MAX_SAMPLE_INTERVAL_S)
        ):
            raise RuntimeControlError(
                "sampler interval must be a number in (0, "
                f"{RUN_MAX_SAMPLE_INTERVAL_S}], got {interval_s!r}"
            )
        self._sample_fn = sample_fn
        self._latch = latch if latch is not None else PressureLatch()
        self._interval_s = float(interval_s)
        self._clock = clock
        self._stop_event = threading.Event()
        self._wait_fn = wait
        self._thread: threading.Thread | None = None
        self.samples = 0
        self.errors = 0
        self.last_error: str | None = None
        self.started_monotonic: float | None = None
        self.finished_monotonic: float | None = None
        self.requested_waits: list[float] = []

    def start(self) -> None:
        if self._thread is not None:
            return
        self.started_monotonic = float(self._clock())
        self._thread = threading.Thread(
            target=self._run, name="r55e1-run-sampler", daemon=True
        )
        self._thread.start()

    def _wait(self, delay: float) -> bool:
        self.requested_waits.append(float(delay))
        if self._wait_fn is not None:
            return bool(self._wait_fn(float(delay)))
        return bool(self._stop_event.wait(float(delay)))

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                started = float(self._clock())
                try:
                    sample = self._sample_fn()
                    self._latch.record(sample)
                    self.samples += 1
                except Exception as exc:  # noqa: BLE001 - fatal at the boundary
                    self.errors += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self._latch.record_error(exc)
                elapsed = float(self._clock()) - started
                delay = max(0.0, self._interval_s - elapsed)
                if self._wait(delay):
                    break
        finally:
            self.finished_monotonic = float(self._clock())

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def stop_and_join(self, timeout: float | None = None) -> None:
        """Stop and PROVE termination; a live monitor is a hard error."""
        self.stop()
        if self._thread is None:
            return
        wait_s = (
            max(1.0, self._interval_s * 2.0)
            if timeout is None else float(timeout)
        )
        self._thread.join(timeout=wait_s)
        if self._thread.is_alive():
            raise RuntimeControlError(
                "R55-E1 RUN sampler did not terminate before return"
            )

    def to_dict(self) -> dict:
        return {
            "interval_s": float(self._interval_s),
            "samples": int(self.samples),
            "errors": int(self.errors),
            "last_error": self.last_error,
            "started_monotonic_ns": (
                None
                if self.started_monotonic is None
                else int(self.started_monotonic * 1_000_000_000)
            ),
            "finished_monotonic_ns": (
                None
                if self.finished_monotonic is None
                else int(self.finished_monotonic * 1_000_000_000)
            ),
            "latch": self._latch.snapshot(),
            "note": (
                "observational only; latched pressure may inform the NEXT wave "
                "boundary and never mutates or kills a running unit"
            ),
        }


# ---------------------------------------------------------------------------
# Deterministic wave partition + canonical order
# ---------------------------------------------------------------------------


def plan_wave_bounds(total_tasks: int, units_in_flight: int) -> tuple[tuple[int, int], ...]:
    """Contiguous canonical wave slices; every slice has <= ``units_in_flight``."""
    total = strict_int(total_tasks, "total_tasks", minimum=0)
    bound = strict_int(units_in_flight, "units_in_flight", minimum=1)
    return tuple(
        (start, min(start + bound, total)) for start in range(0, total, bound)
    )


def next_wave_bound(offset: int, total_tasks: int, units_in_flight: int) -> tuple[int, int]:
    """Return the next ``(start, end)`` wave slice for a RUN checkpoint."""
    total = strict_int(total_tasks, "total_tasks", minimum=0)
    start = strict_int(offset, "offset", minimum=0, maximum=total)
    bound = strict_int(units_in_flight, "units_in_flight", minimum=1)
    return int(start), int(min(start + bound, total))


# ---------------------------------------------------------------------------
# Telemetry aggregation (honest labels, no fabricated zeros)
# ---------------------------------------------------------------------------


def aggregate_worker_peaks(results: Sequence[Mapping[str, Any]]) -> dict:
    """Summarize returned per-task worker peaks WITHOUT a dedup claim."""
    values: list[int] = []
    for result in results:
        if not isinstance(result, Mapping):
            continue
        value = result.get("peak_rss_kib")
        if value is None:
            continue
        try:
            values.append(strict_int(value, "worker peak_rss_kib", minimum=0))
        except RuntimeControlError:
            continue
    if not values:
        return {
            "values_kib": None,
            "sum_kib": None,
            "max_kib": None,
            "count": 0,
            "label": "unavailable",
            "current_available": False,
            "current_unavailable_reason": "no worker returned peak_rss_kib",
        }
    return {
        "values_kib": [int(v) for v in values],
        "sum_kib": int(sum(values)),
        "max_kib": int(max(values)),
        "count": int(len(values)),
        "label": (
            "sum_of_returned_per_task_peak_rss_no_dedup_upper_bound"
        ),
        "current_available": False,
        "current_unavailable_reason": (
            "cell workers return peak RSS only, not current RSS"
        ),
    }


def build_wave_record(
    *,
    index: int,
    offset: int,
    wave_units: int,
    total_tasks: int,
    remaining_units_after: int,
    candidate: Mapping[str, Any],
    mapping: Mapping[str, Any],
    facts: Mapping[str, Any],
    shrink_steps: Sequence[Mapping[str, Any]] = (),
    latched: Mapping[str, Any] | None = None,
    decision_latched: Mapping[str, Any] | None = None,
    pending_latched: Mapping[str, Any] | None = None,
    results: Sequence[Mapping[str, Any]] = (),
    wall_s: float | None = None,
    parent: Mapping[str, Any] | None = None,
    cgroup: Mapping[str, Any] | None = None,
    sampler: Mapping[str, Any] | None = None,
    heartbeat: Mapping[str, Any] | None = None,
) -> dict:
    """Build one truthful wave telemetry record."""
    det_facts = {
        "raw_available_bytes": int(facts["host"]["raw_available_bytes"]),
        "budget_bytes": int(facts["host"]["budget_bytes"]),
        "factor_num": int(facts["host"]["factor_num"]),
        "factor_den": int(facts["host"]["factor_den"]),
        "disk_free_bytes": int(facts["disk"]["free_bytes"]),
        "remaining_units": int(
            facts.get("remaining_units", remaining_units_after)
        ),
    }
    return {
        "index": int(index),
        "offset": int(offset),
        "wave_units": int(wave_units),
        "total_tasks": int(total_tasks),
        "units_remaining_after": int(remaining_units_after),
        "candidate": dict(candidate),
        "mapping": dict(mapping),
        "facts": dict(facts),
        "deterministic_facts": det_facts,
        "shrink_steps": [dict(step) for step in shrink_steps],
        "latched": (
            None if latched is None else dict(latched)
        ),
        "decision_latched": (
            None if decision_latched is None else dict(decision_latched)
        ),
        "pending_latched": (
            None if pending_latched is None else dict(pending_latched)
        ),
        "heartbeat": dict(heartbeat or {}),
        "worker_peaks": aggregate_worker_peaks(results),
        "wall_s": None if wall_s is None else float(wall_s),
        "parent": dict(parent or {}),
        "cgroup": dict(cgroup or {}),
        "sampler": dict(sampler or {}),
    }


def build_run_telemetry(
    *,
    waves: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
    shrink_count: int,
    refusal: Mapping[str, Any] | None = None,
    parent_end: Mapping[str, Any] | None = None,
    deterministic_ref: str | None = None,
    terminal: Mapping[str, Any] | None = None,
) -> dict:
    """Observational RUN telemetry block (never used as a decision hash)."""
    total_wall = 0.0
    for wave in waves:
        if wave.get("wall_s") is not None:
            total_wall += float(wave["wall_s"])
    return {
        "schema": RUN_TELEMETRY_SCHEMA,
        "waves": [dict(wave) for wave in waves],
        "checkpoints": [dict(cp) for cp in checkpoints],
        "metrics": {
            "wave_count": int(len(waves)),
            "checkpoint_count": int(len(checkpoints)),
            "shrink_count": int(shrink_count),
            "examined_shrink_steps": int(
                sum(len(cp.get("shrink_steps", ())) for cp in checkpoints)
            ),
            "refusal_count": int(0 if refusal is None else 1),
            "total_wall_s": float(total_wall),
            "heartbeat_calls": int(
                sum(
                    int((wave.get("heartbeat") or {}).get(
                        "heartbeat_calls", 0
                    ))
                    for wave in waves
                )
            ),
            "heartbeat_errors": int(
                sum(
                    int((wave.get("heartbeat") or {}).get(
                        "heartbeat_errors", 0
                    ))
                    for wave in waves
                )
            ),
            "parent_end": dict(parent_end or {}),
        },
        "refusal": None if refusal is None else dict(refusal),
        "terminal": None if terminal is None else dict(terminal),
        "deterministic_ref": deterministic_ref,
        "note": (
            "R55-E1 RUN telemetry: observational only. Wall times, RSS, "
            "cgroup, OS I/O and disk free are excluded from deterministic "
            "plan/decision hashes; unavailable optional metrics are null with "
            "an explicit reason and are never fabricated as zero."
        ),
    }


# ---------------------------------------------------------------------------
# Deterministic serialization
# ---------------------------------------------------------------------------


def _strip_observational(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_observational(item)
            for key, item in value.items()
            if key not in OBSERVATIONAL_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_strip_observational(item) for item in value]
    return value


def normalized_projection(value: Mapping[str, Any]) -> dict:
    """Remove observational keys recursively (deterministic comparisons)."""
    return _strip_observational(value)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_payload(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def deterministic_hash(value: Mapping[str, Any]) -> str:
    return sha256_payload(normalized_projection(value))


__all__ = [
    "RuntimeControlError",
    "RUN_CONTROL_SCHEMA",
    "RUN_TELEMETRY_SCHEMA",
    "RUN_DECISION_SCHEMA",
    "RUN_EVIDENCE_SCHEMA",
    "RUN_MAX_SAMPLE_INTERVAL_S",
    "OBSERVATIONAL_KEYS",
    "strict_int",
    "run_budget_bytes",
    "read_meminfo",
    "read_parent_memory",
    "read_cgroup_memory",
    "read_disk_io",
    "read_disk_free",
    "collect_run_facts",
    "validate_checkpoint_facts",
    "validate_latched_fact",
    "PressureLatch",
    "RunSampler",
    "plan_wave_bounds",
    "next_wave_bound",
    "aggregate_worker_peaks",
    "build_wave_record",
    "build_run_telemetry",
    "normalized_projection",
    "canonical_json",
    "sha256_payload",
    "deterministic_hash",
]
