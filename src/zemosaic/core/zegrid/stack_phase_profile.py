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
* Nesting is explicit: ``reject_combine_host_s`` is a parent span that contains
  the transfer/reduction buckets. They are NOT additive with the parent.
"""

from __future__ import annotations

import math

STACK_PHASE_PROFILE_SCHEMA = "zegrid-stack-phase-profile-v1"

#: Host wall-clock phases accumulated per cell. Each value is a real operation
#: boundary in :func:`canonical_streaming._phase2`, not a guessed split.
#: ``reject_combine_host_s`` is the PARENT span; the transfer/reduction buckets
#: below are nested inside it and must not be added to it.
HOST_PHASES = (
    "tile_read_s",           # provider.get_tile() reads only
    "tile_prepare_s",        # allocation + coefficient re-application + f64 cast
    "taper_weights_s",       # taper map + q*m*a weight map
    "reject_combine_host_s", # parent span: reject+combine submission + host work
    "h2d_transfer_s",        # nested: xp.asarray(host_array) calls
    "d2h_transfer_s",        # nested: xp.asnumpy / _as_host_numpy calls
    "host_reduction_s",      # nested: np.* reductions on host arrays
    "support_accum_s",       # w1/w2 support accumulation
    "writes_s",              # interior extraction, plane writes, sink flush
    "boundary_sync_s",       # existing GPU sync+pool-purge boundaries
)

_DISJOINT_PHASES = (
    "tile_read_s",
    "tile_prepare_s",
    "taper_weights_s",
    "reject_combine_host_s",
    "support_accum_s",
    "writes_s",
    "boundary_sync_s",
)
_NESTED_PHASES = ("h2d_transfer_s", "d2h_transfer_s", "host_reduction_s")

_PHASE_TOPOLOGY = {
    "disjoint_phases": list(_DISJOINT_PHASES),
    "nested_inside_reject_combine_host_s": list(_NESTED_PHASES),
    "note": (
        "The disjoint phases are non-overlapping top-level spans. The nested "
        "transfer/reduction call durations occur INSIDE reject_combine_host_s; "
        "do not add them to it (that would double count) and do not treat the "
        "parent span as pure GPU compute (it includes host submission and gaps)."
    ),
}


def _finite_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    if not math.isfinite(f):
        return None
    return f


def _round_s(value):
    return round(max(0.0, float(value)), 6)


def _int_or_none(value):
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return int(value)


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
            "phase_topology": dict(_PHASE_TOPOLOGY),
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
                    "Device spans use one synchronized get_elapsed_time() per "
                    "tile while profiling is ON. OFF performs no measurement "
                    "sync and creates no CUDA event."
                ),
            },
        }


def _parse_snapshot(snap):
    """Strictly normalise ONE snapshot; return None if ANY field is malformed.

    Atomic: a malformed snapshot contributes NOTHING (no partial totals), so a
    corrupt tile counter can never inflate a phase total.
    """
    if not isinstance(snap, dict):
        return None
    if snap.get("schema") != STACK_PHASE_PROFILE_SCHEMA:
        return None
    backend = snap.get("backend")
    if backend not in ("cpu", "gpu"):
        return None
    tiles = _int_or_none(snap.get("tiles"))
    frames = _int_or_none(snap.get("frames_read"))
    if tiles is None or frames is None:
        return None
    phases_raw = snap.get("host_phases")
    if not isinstance(phases_raw, dict):
        return None
    phases = {}
    for name in HOST_PHASES:
        bucket = phases_raw.get(name)
        if not isinstance(bucket, dict):
            return None
        total = _finite_or_none(bucket.get("total_s"))
        count = _int_or_none(bucket.get("count"))
        if total is None or count is None:
            return None
        phases[name] = (max(0.0, total), count)
    device = snap.get("device")
    if not isinstance(device, dict):
        return None
    span = device.get("combine_stream_span_s")
    if span is not None:
        span = _finite_or_none(span)
        if span is None:
            return None
        span = max(0.0, span)
    pairs = _int_or_none(device.get("combine_event_pairs"))
    if pairs is None:
        return None
    reason = device.get("unavailable_reason")
    if reason is not None and not isinstance(reason, str):
        return None
    perturbation = snap.get("perturbation")
    if not isinstance(perturbation, dict):
        return None
    extra_syncs = _int_or_none(perturbation.get("extra_synchronizations"))
    if extra_syncs is None:
        return None
    per_tile = perturbation.get("cuda_event_sync_per_tile")
    if not isinstance(per_tile, bool):
        return None
    return {
        "backend": backend,
        "tiles": tiles,
        "frames_read": frames,
        "phases": phases,
        "device_span": span,
        "device_pairs": pairs,
        "device_reason": reason,
        "extra_syncs": extra_syncs,
        "cuda_event_sync_per_tile": per_tile,
    }


def aggregate_records(records, *, current_ids, all_current=False):
    """Split result records by ACTUAL cell identity and aggregate each side.

    ``records`` are per-cell result mappings carrying ``cell_id`` and an
    optional ``stack_phase_profile``. ``current_ids`` is the set of cells
    executed in THIS attempt (resume pending set); ``all_current=True`` marks
    every record current (fresh run). A record without a profile (empty cell or
    an old/historical store) is still counted in its side's cell count and
    surfaced as ``cells_without_profile``. The two aggregates are never summed.
    """
    current_ids = {
        str(cid) for cid in (current_ids or ())
    }
    current_profiles = []
    historical_profiles = []
    current_cells = 0
    historical_cells = 0
    for record in records or ():
        if not isinstance(record, dict):
            continue
        cid = str(record.get("cell_id", ""))
        profile = record.get("stack_phase_profile")
        is_current = bool(all_current or cid in current_ids)
        if is_current:
            current_cells += 1
            if isinstance(profile, dict):
                current_profiles.append(profile)
        else:
            historical_cells += 1
            if isinstance(profile, dict):
                historical_profiles.append(profile)
    current = aggregate(current_profiles, attribution="current")
    historical = aggregate(
        historical_profiles, attribution="historical_restored"
    )
    current["cells_in_attempt"] = int(current_cells)
    current["cells_without_profile"] = max(
        0, int(current_cells) - int(current["cells_profiled"])
    )
    historical["cells_restored"] = int(historical_cells)
    historical["cells_without_profile"] = max(
        0, int(historical_cells) - int(historical["cells_profiled"])
    )
    return current, historical


def aggregate(profiles, *, attribution: str = "current") -> dict:
    """Merge per-cell snapshots into ONE bounded run-level record.

    ``profiles`` is an iterable of per-cell snapshot dicts. Each entry is
    validated ATOMICALLY: a malformed entry is counted and contributes nothing.
    Historical/current attribution is explicit: the caller passes
    ``attribution`` (``"current"`` or ``"historical_restored"``) and the
    aggregate never adds resume wall time to restored phase totals.
    """
    totals = {name: {"total_s": 0.0, "count": 0} for name in HOST_PHASES}
    cells = 0
    malformed = 0
    device_total = 0.0
    device_pairs = 0
    device_reasons: dict[str, int] = {}
    extra_syncs = 0
    cuda_event_sync_cells = 0
    tiles = 0
    frames = 0
    backend_counts = {"cpu": 0, "gpu": 0}
    for snap in profiles or ():
        parsed = _parse_snapshot(snap)
        if parsed is None:
            malformed += 1
            continue
        cells += 1
        tiles += parsed["tiles"]
        frames += parsed["frames_read"]
        backend_counts[parsed["backend"]] += 1
        for name, (total, count) in parsed["phases"].items():
            totals[name]["total_s"] += total
            totals[name]["count"] += count
        if parsed["device_span"] is not None:
            device_total += parsed["device_span"]
            device_pairs += parsed["device_pairs"]
        if parsed["device_reason"]:
            reason = parsed["device_reason"]
            device_reasons[reason] = device_reasons.get(reason, 0) + 1
        extra_syncs += parsed["extra_syncs"]
        if parsed["cuda_event_sync_per_tile"]:
            cuda_event_sync_cells += 1
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
        "phase_topology": dict(_PHASE_TOPOLOGY),
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
        "perturbation": {
            "extra_synchronizations": int(extra_syncs),
            "cells_with_cuda_event_sync": int(cuda_event_sync_cells),
            "cuda_event_sync_per_tile": bool(cuda_event_sync_cells > 0),
            "note": (
                "Summed measurement perturbation for this attribution. ON/GPU "
                "uses one synchronized get_elapsed_time() per profiled tile; "
                "OFF performs none."
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
