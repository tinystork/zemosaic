"""SINGLE shared run-volatile manifest classifier for the R40-B2 suites.

Bounded task G (hygiene, Nono B2-F L1): the run-volatile classifier used to be
DUPLICATED verbatim in ``tests/test_zegrid_r40b2_panel.py`` and
``tests/test_zegrid_r40b2_offon_identity.py``, so the two copies could drift while
both kept passing. It now lives in exactly ONE place, imported by both suites.

What it classifies (frozen §8 clause 12 background): the manifest leaves that are
pure functions of the FREE RAM measured at plan time — wall-timing/RSS measurement
leaves and the environment-driven scheduler plan (mode, footprint, concurrency,
plan candidates, reprojection path, executor/worker counts, RAM-budget figures,
and the per-cell ``mode``/``bound_bytes`` the planner stamps into each cell record).

The classifier is NOT a blanket exclusion: it never excuses a product/science
field, and the ``.cells`` subtree is never wholesale excluded (the completeness
obligation is proven by measurement in
``tests/test_zegrid_r40b2_offon_identity.py::test_run_volatile_classifier_covers_plan_differences``
and re-stated, with a fail-loud negative control, in the panel suite).
"""

from __future__ import annotations

# --- (a)+(b) run-VOLATILE = measurement + environment-driven scheduler plan. ---
VOLATILE_PREFIXES = (
    ".timings",
    ".gauge_diagnostics.instrumentation",
    ".gauge_diagnostics.phases",
    ".gauge_diagnostics.reprojection",
    ".reprojection",
    ".cpu_streaming",
    ".gpu.cpu_fallback_plan",
    ".per_cell_diagnostics.plan_candidates",
    ".per_cell_diagnostics.cpu_streaming",
    ".per_cell_diagnostics.cpu_fallback_plan",
    ".per_cell_diagnostics.reprojection",
)
VOLATILE_LEAF_EXACT = {
    "mode", "bound_bytes", "per_cell_footprint_bytes", "available_bytes",
    "ram_budget_bytes", "plan_chosen_reason", "cells_in_flight", "workers",
    "cache_build_workers", "cache_workers_per_cell", "executor", "seconds",
    "phase_seconds", "fast_calls", "path", "cpu_streaming_tile",
    "cpu_streaming_tile_selected", "gpu_streaming_tile",
    "gpu_streaming_host_peak_bytes", "peak_rss_kib", "aggregate_peak_rss_kib",
}
VOLATILE_SUFFIXES = ("_s", "_seconds", "_sum_s", "_per_unit", "_unit")
VOLATILE_SUBSTRINGS = ("rss_kib", "seconds", "reproject_s", "trials")


def segments(path):
    """Split a dotted manifest leaf path into non-empty segments."""
    return [s for s in path.split(".") if s]


# --- (a) PRE-EXISTING measurement fields: wall seconds / per-unit, RSS. --------
MEASUREMENT_SUFFIXES = ("_s", "_seconds", "_sum_s", "_per_unit", "_unit")
MEASUREMENT_SUBSTRINGS = ("rss_kib", "seconds", "reproject_s")
MEASUREMENT_EXACT = {"seconds", "phase_seconds", "peak_rss_kib", "aggregate_peak_rss_kib"}


def is_measurement_volatile(path):
    """The measurement-only subset of the run-volatile classifier."""
    segs = segments(path)
    if segs and segs[0] == "timings":
        return True
    for seg in segs:
        if seg.endswith(MEASUREMENT_SUFFIXES) or seg in MEASUREMENT_EXACT:
            return True
        if any(h in seg for h in MEASUREMENT_SUBSTRINGS):
            return True
    return False


def prefix_hit(path):
    """True when the leaf path sits under one of the declared volatile prefixes."""
    return any(path == p or path.startswith(p + ".") or path.startswith(p + "[")
               for p in VOLATILE_PREFIXES)


def is_run_volatile(path):
    """THE single shared run-volatile predicate (measurement + scheduler plan)."""
    if prefix_hit(path):
        return True
    for seg in segments(path):
        if seg.endswith(VOLATILE_SUFFIXES) or seg in VOLATILE_LEAF_EXACT:
            return True
        if any(h in seg for h in VOLATILE_SUBSTRINGS):
            return True
    return False


__all__ = [
    "VOLATILE_PREFIXES",
    "VOLATILE_LEAF_EXACT",
    "VOLATILE_SUFFIXES",
    "VOLATILE_SUBSTRINGS",
    "MEASUREMENT_SUFFIXES",
    "MEASUREMENT_SUBSTRINGS",
    "MEASUREMENT_EXACT",
    "segments",
    "prefix_hit",
    "is_run_volatile",
    "is_measurement_volatile",
]
