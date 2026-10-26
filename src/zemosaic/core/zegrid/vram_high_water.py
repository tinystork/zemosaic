"""ZM-WIN-VRAM-HW — sampled VRAM high-water during ``per_cell_stack``.

The ZeGrid "Resource monitor" used to sample GPU memory only at PHASE boundaries
(setup / layout / gauge / per-cell / assembly). During a 465 s per-cell stack it
therefore produced six point observations, and the per-cell snapshots were taken
at CELL boundaries — i.e. right AFTER :func:`gpu_cell_boundary_cleanup` purged
the allocator pool. The intra-cell peak was invisible by construction.

This module owns a small, observational sampler that runs ONLY while the
per-cell stacking phase executes:

* it samples the CUDA-free GPU probe (:func:`sample_gpu_via_nvml`: pynvml ->
  ctypes NVML -> ``nvidia-smi``) on a bounded cadence;
* it keeps the **MAX** ``gpu_used_mb`` observed, never the last value;
* it NEVER creates a CUDA context and NEVER touches CuPy — the probe it uses
  queries the driver only (see ``zemosaic_resource_telemetry``). This is a hard
  contract asserted by an AST guard in the tests;
* it is best-effort: a probe failure becomes a recorded token, never an
  exception in the run, and ``stop()`` can never block for long (bounded join).

Honesty about the measurement
-----------------------------
The published value is a **sampled** high-water mark, not the driver's own
high-water counter. ``VRAM_HIGH_WATER_LABEL`` says so in every record. The
sampling cadence is bounded below by the probe. ``nvidia-smi`` answers through
a 5 s TTL cache in the probe module, so a 0.5 s requested cadence CANNOT be
honoured on the ``nvidia-smi`` path; in that case the sampler degrades its
EFFECTIVE cadence to >= 5 s and records that explicitly instead of pretending
it sampled faster (which would silently duplicate stale values).

GPU OFF / telemetry OFF publish ``None`` fields plus an explicit reason; they
never fabricate a ``0``.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable

__all__ = [
    "VRAM_HIGH_WATER_SCHEMA",
    "VRAM_HIGH_WATER_LABEL",
    "DEFAULT_INTERVAL_S",
    "MIN_INTERVAL_S",
    "NVIDIA_SMI_TTL_S",
    "VramHighWaterSampler",
    "disabled_record",
    "start_vram_high_water_sampler",
    "stop_vram_high_water_sampler",
    "stop_all_active_vram_samplers",
    "active_sampler_count",
]

# Version of the persisted record shape. Bump when keys change meaning.
VRAM_HIGH_WATER_SCHEMA = "zemosaic-win-vram-hw-sampled-v1"

# Every published high-water says WHAT it is. Never a driver hardware counter.
VRAM_HIGH_WATER_LABEL = "sampled_high_water_not_driver_counter"

# Default cadence, mirroring ``resource_telemetry_interval_sec``. The floor
# keeps a pathological config from spinning a probe thread.
DEFAULT_INTERVAL_S = 1.5
MIN_INTERVAL_S = 0.5

# ``nvidia-smi`` probe cache TTL (see zemosaic_resource_telemetry). On that
# path a faster cadence would only repeat a stale value, so it is NOT claimed.
NVIDIA_SMI_TTL_S = 5.0

# Bounded join: a probe can take up to ~2 s (``nvidia-smi`` timeout). The stop
# path never waits longer than this, even on a wedged driver.
_JOIN_TIMEOUT_S = 3.0

_HIGH_FREQ_SOURCES = frozenset({"pynvml", "ctypes_nvml"})


def _clamp_interval(value: Any) -> float:
    """Return a finite cadence within ``[MIN_INTERVAL_S, 3600]``."""
    try:
        interval = float(value)
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_S
    if not math.isfinite(interval) or interval <= 0.0:
        return DEFAULT_INTERVAL_S
    return max(MIN_INTERVAL_S, min(interval, 3600.0))


def _default_probe() -> dict[str, Any]:
    """CUDA-free probe (imported lazily to keep this module import-light)."""
    from ... import zemosaic_resource_telemetry as rt

    return rt.sample_gpu_via_nvml()


def disabled_record(reason: str) -> dict[str, Any]:
    """Explicit "not sampled" record: every measurement field is null."""
    return {
        "schema": VRAM_HIGH_WATER_SCHEMA,
        "label": VRAM_HIGH_WATER_LABEL,
        "vram_high_water_available": False,
        "vram_high_water_sampled_mb": None,
        "vram_high_water_last_sample_mb": None,
        "vram_high_water_sample_count": 0,
        "vram_high_water_sample_attempts": 0,
        "vram_high_water_interval_s": None,
        "vram_high_water_requested_interval_s": None,
        "vram_high_water_source": None,
        "vram_high_water_effective_cadence_degraded": False,
        "vram_high_water_effective_cadence_note": None,
        "vram_high_water_window_s": None,
        "vram_high_water_unavailable_reason": str(reason) if reason else None,
        "vram_high_water_sample_errors": [],
        "vram_high_water_started": False,
        "vram_high_water_stopped": False,
    }


class VramHighWaterSampler:
    """Thread-safe MAX-of-``gpu_used_mb`` sampler with a bounded lifetime.

    Parameters (all injectable so CPU-only tests can drive it deterministically):

    ``interval_sec``  requested cadence (clamped, floor ``MIN_INTERVAL_S``).
    ``probe``         ``() -> dict`` GPU probe; defaults to the CUDA-free one.
    ``on_sample``     optional callback ``(record_dict) -> None`` invoked after
                      every completed sample (used to mirror the value into the
                      resource telemetry). Exceptions are swallowed.
    ``clock``         monotonic clock used for the sampling window.
    ``wait``          ``(delay_s) -> bool`` blocking wait returning True when
                      the sampler must stop. Defaults to the stop event.
    """

    def __init__(
        self,
        *,
        interval_sec: Any = DEFAULT_INTERVAL_S,
        probe: Callable[[], dict[str, Any]] | None = None,
        on_sample: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], float] | None = None,
        wait: Callable[[float], bool] | None = None,
        name: str = "zegrid-vram-highwater",
    ) -> None:
        self.requested_interval_s = _clamp_interval(interval_sec)
        self._probe = probe or _default_probe
        self._on_sample = on_sample
        self._clock = clock or time.monotonic
        self._thread_name = str(name)
        self._stop_event = threading.Event()
        self._wait = wait if wait is not None else self._stop_event.wait
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._stop_reason: str | None = None
        self._samples = 0
        self._attempts = 0
        self._max_mb: float | None = None
        self._last_mb: float | None = None
        self._source: str | None = None
        self._source_counts: dict[str, int] = {}
        self._errors: list[str] = []
        self._effective_interval_s = self.requested_interval_s
        self._degraded = False
        self._degraded_note: str | None = None
        self._unavailable_reason: str | None = None

    # -- lifetime ---------------------------------------------------------
    def start(self) -> "VramHighWaterSampler":
        with self._lock:
            if self._started:
                return self
            self._started = True
            self._started_at = float(self._clock())
        thread = threading.Thread(
            target=self._run, name=self._thread_name, daemon=True
        )
        self._thread = thread
        thread.start()
        return self

    def stop(self, reason: str = "stopped") -> dict[str, Any]:
        """Stop the thread (bounded join) and return the final record."""
        with self._lock:
            self._stop_event.set()
            if self._stop_reason is None:
                self._stop_reason = str(reason)
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            try:
                thread.join(timeout=_JOIN_TIMEOUT_S)
            except Exception:
                pass
        with self._lock:
            self._stopped = True
            if self._stopped_at is None:
                self._stopped_at = float(self._clock())
        return self.snapshot()

    @property
    def started(self) -> bool:
        with self._lock:
            return bool(self._started)

    # -- sampling ---------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._sample_once()
            except BaseException:
                # The sampler must never kill its thread with an exception; the
                # per-attempt guard already records tokens, this is the last net.
                with self._lock:
                    self._record_error("sampler_loop_exception")
            delay = self._current_interval()
            try:
                if self._wait(delay):
                    break
            except BaseException:
                break

    def _current_interval(self) -> float:
        with self._lock:
            return float(self._effective_interval_s)

    def _record_error(self, token: str) -> None:
        # Caller holds the lock.
        if len(self._errors) < 8 and token not in self._errors:
            self._errors.append(str(token))

    def _sample_once(self) -> None:
        info: Any = None
        failure: str | None = None
        try:
            info = self._probe()
        except BaseException as exc:  # noqa: BLE001 - best-effort by contract
            failure = f"probe_exception:{type(exc).__name__}"

        if not isinstance(info, dict):
            if failure is None:
                failure = "probe_returned_non_mapping"

        value_mb: float | None = None
        source: str | None = None
        unavailable: str | None = None
        if isinstance(info, dict):
            source = info.get("gpu_sample_source") or info.get("gpu_source")
            if source is not None:
                source = str(source)
            raw_used = info.get("gpu_used_mb")
            try:
                candidate = float(raw_used) if raw_used is not None else None
            except (TypeError, ValueError):
                candidate = None
            if candidate is not None and math.isfinite(candidate) and candidate >= 0.0:
                value_mb = candidate
            else:
                unavailable = str(
                    info.get("gpu_unavailable_reason")
                    or "gpu_used_mb_unavailable"
                )

        record: dict[str, Any] | None = None
        with self._lock:
            self._attempts += 1
            if failure is not None:
                self._record_error(failure)
            if source is not None:
                self._source = source
                self._source_counts[source] = (
                    self._source_counts.get(source, 0) + 1
                )
            if value_mb is not None:
                self._samples += 1
                self._last_mb = float(value_mb)
                if self._max_mb is None or value_mb > self._max_mb:
                    self._max_mb = float(value_mb)
                self._unavailable_reason = None
                self._sync_cadence_locked(source)
            else:
                if unavailable is not None:
                    self._record_error(unavailable)
                if self._samples == 0:
                    self._unavailable_reason = unavailable or "gpu_sample_unavailable"
            record = self._snapshot_locked()
        if record is not None and self._on_sample is not None:
            try:
                self._on_sample(record)
            except Exception:
                pass

    def _sync_cadence_locked(self, source: str | None) -> None:
        """Degrade the EFFECTIVE cadence when only ``nvidia-smi`` answers.

        The ``nvidia-smi`` probe is TTL-cached for 5 s. Sampling it every 0.5 s
        would return the SAME cached value; claiming a 0.5 s cadence would be a
        lie. We therefore wait >= 5 s on that path and say so in the record.
        """
        if source is not None and source in _HIGH_FREQ_SOURCES:
            self._effective_interval_s = float(self.requested_interval_s)
            self._degraded = False
            self._degraded_note = None
            return
        if source == "nvidia_smi":
            degraded_interval = max(
                float(self.requested_interval_s), NVIDIA_SMI_TTL_S
            )
            self._effective_interval_s = float(degraded_interval)
            self._degraded = degraded_interval > float(
                self.requested_interval_s
            )
            self._degraded_note = (
                "effective_cadence_limited_by_nvidia_smi_ttl_cache"
                if self._degraded
                else None
            )
            return
        # Unknown/failed source: keep the requested cadence; failures are cheap
        # and already recorded. No cadence claim is degraded without evidence.

    # -- publication ------------------------------------------------------
    def _snapshot_locked(self) -> dict[str, Any]:
        window = None
        if self._started_at is not None:
            end = self._stopped_at if self._stopped_at is not None else self._clock()
            try:
                window = max(0.0, float(end) - float(self._started_at))
            except Exception:
                window = None
        return {
            "schema": VRAM_HIGH_WATER_SCHEMA,
            "label": VRAM_HIGH_WATER_LABEL,
            "vram_high_water_available": self._samples > 0,
            "vram_high_water_sampled_mb": (
                round(float(self._max_mb), 3) if self._max_mb is not None else None
            ),
            "vram_high_water_last_sample_mb": (
                round(float(self._last_mb), 3)
                if self._last_mb is not None
                else None
            ),
            "vram_high_water_sample_count": int(self._samples),
            "vram_high_water_sample_attempts": int(self._attempts),
            "vram_high_water_interval_s": (
                round(float(self._effective_interval_s), 3)
                if self._samples > 0 or self._started
                else None
            ),
            "vram_high_water_requested_interval_s": round(
                float(self.requested_interval_s), 3
            ),
            "vram_high_water_source": self._source,
            "vram_high_water_source_counts": dict(self._source_counts),
            "vram_high_water_effective_cadence_degraded": bool(self._degraded),
            "vram_high_water_effective_cadence_note": self._degraded_note,
            "vram_high_water_window_s": (
                round(float(window), 3) if window is not None else None
            ),
            "vram_high_water_unavailable_reason": (
                None if self._samples > 0 else self._unavailable_reason
            ),
            "vram_high_water_sample_errors": list(self._errors),
            "vram_high_water_started": bool(self._started),
            "vram_high_water_stopped": bool(self._stopped),
            "vram_high_water_stop_reason": self._stop_reason,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()


# ---------------------------------------------------------------------------
# Process-local registry: the safety net for exception / cancellation paths.
# ---------------------------------------------------------------------------
_ACTIVE_SAMPLERS: list[VramHighWaterSampler] = []
_REGISTRY_LOCK = threading.Lock()


def start_vram_high_water_sampler(
    *,
    interval_sec: Any = DEFAULT_INTERVAL_S,
    probe: Callable[[], dict[str, Any]] | None = None,
    on_sample: Callable[[dict[str, Any]], None] | None = None,
    clock: Callable[[], float] | None = None,
    wait: Callable[[float], bool] | None = None,
) -> VramHighWaterSampler:
    """Start (and register) one sampler.

    Registration is what makes the caller's ``finally`` path provable: even an
    unexpected ``BaseException`` escaping the phase can be cleaned up with
    :func:`stop_all_active_vram_samplers` without threading a reference through
    every intermediate frame.
    """
    sampler = VramHighWaterSampler(
        interval_sec=interval_sec, probe=probe, on_sample=on_sample,
        clock=clock, wait=wait,
    )
    with _REGISTRY_LOCK:
        _ACTIVE_SAMPLERS.append(sampler)
    sampler.start()
    return sampler


def stop_vram_high_water_sampler(
    sampler: VramHighWaterSampler | None, reason: str = "stopped"
) -> dict[str, Any] | None:
    if sampler is None:
        return None
    with _REGISTRY_LOCK:
        try:
            _ACTIVE_SAMPLERS.remove(sampler)
        except ValueError:
            pass
    return sampler.stop(reason)


def stop_all_active_vram_samplers(reason: str = "run_exit") -> list[dict[str, Any]]:
    """Stop every registered sampler (bounded); used by run-level ``finally``."""
    with _REGISTRY_LOCK:
        samplers = list(_ACTIVE_SAMPLERS)
        _ACTIVE_SAMPLERS.clear()
    records: list[dict[str, Any]] = []
    for sampler in samplers:
        try:
            records.append(sampler.stop(reason))
        except Exception:
            continue
    return records


def active_sampler_count() -> int:
    with _REGISTRY_LOCK:
        return len(_ACTIVE_SAMPLERS)
