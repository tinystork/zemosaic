"""Default-OFF bounded stacking phase profiler (ZM-STACK-INSTRUMENTATION).

Diagnostic-only. The module imports NumPy only; it NEVER imports CuPy itself.
Device spans are event-timed by the streaming executor, which already owns the
backend module object, and are passed back here as pre-computed seconds.

Design invariants
-----------------
* OFF is the default: no profiler is instantiated, no perf_counter call is
  added to the hot loop, no CUDA event is created and no synchronization is
  performed for measurement. Importing this module is side-effect free.
* Bounded memory: only scalar accumulators (fixed phase set + counts + a few
  integers). There is deliberately NO per-tile / per-frame event list.
* Strict JSON: :meth:`snapshot` / :func:`aggregate` return only ``str``,
  ``int``, ``float`` (finite) or ``None`` values. Durations are rounded to
  microseconds and negative deltas clamp to 0.0.
* Timings never feed a scientific, coverage or durable identity decision; they
  are observational only.
"""

from __future__ import annotations

import math

STACK_PHASE_PROFILE_SCHEMA = "zegrid-stack-phase-profile-v1"

#: Host wall-clock phases accumulated per cell. Each value is a real operation
#: boundary in :func:`canonical_streaming._phase2`, not a guessed split.
HOST_PHASES = (
    "tile_read_s",          # provider.get_tile() reads for N frames
    "tile_prepare_s",       # coefficient re-application + finite/valid masks
    "taper_weights_s",      # taper map + q*m*a weight map
    "reject_combine_host_s",# whole reject+combine call (submission + host work)
    "host_reduction_s",     # _axis_sum_f64 / eligible.sum host reductions
    "host_readback_s",      # _as_host_numpy / host materialisation of results
    "support_accum_s",      # w1/w2 support accumulation
    "writes_s",             # interior extraction, plane writes, sink flush
    "boundary_sync_s",      # existing GPU sync+pool-purge boundaries
)


def _finite_or_none(value):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _round_s(value):
    return round(max(0.0, float(value)), 6)


class StackPhaseProfiler:
    """Bounded accumulator for ONE cell's real stacking phases."""

    __slots__ = (
        "_backend",
        "_phases",
        "_tiles",
        "_frames_read",
        "_device_span_s",
        "_device_pairs",
        "_device_reason",
        "_extra_syncs",
    )

    def __init__(self, *, backend: str = "cpu") -> None:
        self._backend = "gpu" if str(backend).strip().lower() == "gpu" else "cpu"
        self._phases = {name: [0.0, 0] for name in HOST_PHASES}
        self._tiles = 0
        self._frames_read = 0
        self._device_span_s = 0.0
        self._device_pairs = 0
        # Reason the device span is unavailable; None means it is measurable.
        self._device_reason: str | None = (
            None if self._backend == "gpu" else "cpu_backend"
        )
        self._extra_syncs = 0

    # -- accumulation ------------------------------------------------------

    def add(self, phase: str, seconds: float) -> None:
        bucket = self._phases.get(phase)
        if bucket is None:
            raise KeyError(f"unknown stack phase {phase!r}")
        bucket[0] += max(0.0, float(seconds))
        bucket[1] += 1

    def add_device_span(self, seconds: float) -> None:
        self._device_span_s += max(0.0, float(seconds))
        self._device_pairs += 1

    def note_device_unavailable(self, reason: str) -> None:
        self._device_reason = str(reason)

    def count_tile(self, frames_read: int) -> None:
        self._tiles += 1
        self._frames_read += max(0, int(frames_read))

    def note_extra_sync(self, count: int = 1) -> None:
        self._extra_syncs += max(0, int(count))

    # -- serialisation -----------------------------------------------------

    def snapshot(self) -> dict:
        phases = {
            name: {
                "total_s": _round_s(total),
                "count": int(count),
            }
            for name, (total, count) in self._phases.items()
        }
        return {
            "schema": STACK_PHASE_PROFILE_SCHEMA,
            "enabled": True,
            "clock": "perf_counter_host_wall",
            "backend": self._backend,
            "tiles": int(self._tiles),
            "frames_read": int(self._frames_read),
            "host_phases": phases,
            "device": {
                "combine_stream_span_s": (
                    None
                    if self._device_pairs == 0
                    else _round_s(self._device_span_s)
                ),
                "combine_event_pairs": int(self._device_pairs),
                "unavailable_reason": self._device_reason,
                "note": (
                    "CUDA-event span across the reject+combine submission. It "
                    "includes host-induced gaps between submissions and is NOT "
                    "pure kernel compute; never read it as GPU-only time."
                ),
            },
            "perturbation": {
                "extra_synchronizations": int(self._extra_syncs),
                "cuda_event_sync_per_tile": bool(
                    self._backend == "gpu" and self._device_pairs > 0
                ),
                "note": (
                    "Device spans use one synchronized elapsed_time() per tile "
                    "while profiling is ON. OFF performs no measurement sync."
                ),
            },
        }


def aggregate(profiles, *, attribution: str = "current") -> dict:
    """Merge per-cell snapshots into ONE bounded run-level record.

    ``profiles`` is an iterable of per-cell snapshot dicts. Malformed entries
    are counted, never trusted. Historical/current attribution is explicit: the
    caller passes ``attribution`` (``"current"`` or ``"historical_restored"``)
    and the aggregate never adds resume wall time to restored phase totals.
    """
    totals = {name: {"total_s": 0.0, "count": 0} for name in HOST_PHASES}
    cells = 0
    malformed = 0
    device_total = 0.0
    device_pairs = 0
    device_reasons: dict[str, int] = {}
    tiles = 0
    frames = 0
    backend_counts = {"cpu": 0, "gpu": 0}
    for snap in profiles or ():
        if not isinstance(snap, dict) or snap.get("schema") != STACK_PHASE_PROFILE_SCHEMA:
            malformed += 1
            continue
        cells += 1
        tiles += int(snap.get("tiles", 0) or 0)
        frames += int(snap.get("frames_read", 0) or 0)
        backend = str(snap.get("backend", "cpu"))
        if backend in backend_counts:
            backend_counts[backend] += 1
        phases = snap.get("host_phases")
        if isinstance(phases, dict):
            for name, bucket in phases.items():
                if name not in totals or not isinstance(bucket, dict):
                    continue
                totals[name]["total_s"] += max(
                    0.0, _finite_or_none(bucket.get("total_s")) or 0.0
                )
                totals[name]["count"] += int(bucket.get("count", 0) or 0)
        device = snap.get("device")
        if isinstance(device, dict):
            span = _finite_or_none(device.get("combine_stream_span_s"))
            if span is not None:
                device_total += max(0.0, span)
                device_pairs += int(device.get("combine_event_pairs", 0) or 0)
            reason = device.get("unavailable_reason")
            if reason:
                device_reasons[str(reason)] = (
                    device_reasons.get(str(reason), 0) + 1
                )
    return {
        "schema": STACK_PHASE_PROFILE_SCHEMA,
        "enabled": True,
        "attribution": str(attribution),
        "cells_profiled": int(cells),
        "cells_malformed": int(malformed),
        "backend_counts": dict(backend_counts),
        "tiles": int(tiles),
        "frames_read": int(frames),
        "host_phases": {
            name: {
                "total_s": _round_s(bucket["total_s"]),
                "count": int(bucket["count"]),
            }
            for name, bucket in totals.items()
        },
        "device": {
            "combine_stream_span_s": (
                None if device_pairs == 0 else _round_s(device_total)
            ),
            "combine_event_pairs": int(device_pairs),
            "unavailable_reasons": dict(device_reasons),
            "note": (
                "Sum of per-cell CUDA-event reject+combine stream spans. "
                "Includes host gaps; not pure GPU compute."
            ),
        },
        "resume_attribution": {
            "current": (int(cells) if attribution == "current" else 0),
            "historical_restored": (
                int(cells) if attribution == "historical_restored" else 0
            ),
            "note": (
                "Host phase totals above belong ONLY to this attribution. "
                "Restored historical work was not re-profiled and is never "
                "summed with current resume wall time."
            ),
        },
    }
