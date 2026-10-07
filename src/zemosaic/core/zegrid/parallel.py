"""ZM-ZEGRID-R12 — order-preserving process-pool parallel map (deterministic).

Reprojection (``reproject_interp``) is the dominant cache-build cost and is
SERIAL and CPU-bound. This module provides a tiny, deterministic, memory-aware
parallel map so the frame-major reprojection can be spread across 2-4 processes
WITHOUT changing the produced bytes (the aligned cache content stays bit-equal
to the serial build — see the R12 tests that verify it by hash).

Determinism contract: the worker is a pure function of its task (no shared
mutable state, no randomness); ``pmap`` preserves input order; the caller writes
``.npy``/manifest in that same deterministic order. Result == serial order.

Memory policy: worker count is derived from ``os.cpu_count()`` and (optionally)
available memory, defaulting to 2-4. A single worker (or a trivial task list)
falls back to a plain serial loop (no pool spawn).

DAEMON SAFETY (ZM-ZEGRID-R13): the product runs the worker inside a DAEMONIC
process (``run_hierarchical_mosaic_process``). Python forbids a daemonic process
from starting child processes (``multiprocessing`` asserts in ``start()`` for
EVERY start method), so a ``ProcessPoolExecutor`` spawned from a daemonic parent
crashes the whole run. When the parent is daemonic we therefore select a
``ThreadPoolExecutor`` instead (threads are allowed inside a daemonic process and
share the parent's module state, so the copy-on-write globals used by the gauge
workers still work). The result is identical: workers remain pure functions of
their task and ``pool.map`` still preserves input order.
"""

from __future__ import annotations

import logging
import multiprocessing
import os

log = logging.getLogger(__name__)

# Fixed per-worker baseline (bytes): numpy/astropy/reproject/shapely import
# footprint (~330 MiB, matching the R6 STREAMING_BASELINE_KIB measurement).
_PER_WORKER_BASELINE_BYTES = 340 * 1024 * 1024

# ZM-ZEGRID-R16 adaptive worker rule. ``cpu - 2`` leaves ~2 cores for the OS (the
# user's suggestion); the memory side divides the available RAM by the per-worker
# footprint for the CURRENT phase. The result is clamped to [2, 14] so a 16-core
# box uses ~14 processes while small/low-RAM machines never under- or over-spawn.
DEFAULT_WORKERS = 4  # retained for backward-compat references only (R16 rule below)
_CPU_OS_RESERVE = 2
_WORKERS_MIN = 2
_WORKERS_MAX = 14

# Estimated per-worker footprint (bytes) for the GLOBAL-GAUGE phase. Each gauge
# worker reprojects a frame over a full-canvas / union bbox (the dominant cost),
# holding a float32 RGB plane + float64 temporaries; this is measurably heavier
# than the per-cell cache build, so it gets its own (larger) footprint.
_GAUGE_PER_WORKER_FOOTPRINT_BYTES = _PER_WORKER_BASELINE_BYTES + (1600 * 1200 * 3 * 8)

# Estimated per-worker footprint (bytes) for the CACHE-BUILD phase (per-cell
# patch reprojection — smaller working set than the full-canvas gauge).
_CACHE_PER_WORKER_FOOTPRINT_BYTES = _PER_WORKER_BASELINE_BYTES + (512 * 512 * 3 * 8)


def adaptive_worker_count(
    cpu: int,
    available_bytes: int | None,
    per_worker_footprint_bytes: int | None = None,
) -> int:
    """ZM-ZEGRID-R16 adaptive rule: ``clamp(min(cpu-2, RAM//footprint), 2, 14)``.

    Parameters
    ----------
    cpu:
        Logical CPU count (``os.cpu_count()``).
    available_bytes:
        Available RAM at the decision point (``None``/0 = no memory bound).
    per_worker_footprint_bytes:
        Estimated per-worker cost (bytes) for the current phase (gauge vs cache
        build). Defaults to the import baseline.

    Returns a worker count in ``[2, 14]`` (never < 2 on a healthy multi-core box;
    memory/CPU may force ``1`` only when they genuinely cannot support 2).
    """
    footprint = int(per_worker_footprint_bytes or _PER_WORKER_BASELINE_BYTES)
    footprint = max(1, footprint)

    cpu_budget = max(1, int(cpu) - _CPU_OS_RESERVE)
    ram_budget = 0
    if available_bytes is not None and available_bytes > 0:
        ram_budget = max(1, int(available_bytes // footprint))

    if ram_budget > 0:
        target = min(cpu_budget, ram_budget)
    else:
        target = cpu_budget

    # Clamp to [2, 14]; only CPU/RAM genuinely too small may force < 2 (floor 1).
    workers = target
    if workers > _WORKERS_MAX:
        workers = _WORKERS_MAX
    if workers < _WORKERS_MIN:
        # Force up to 2 unless the machine cannot actually support 2 workers.
        hard_floor = _WORKERS_MIN
        if int(cpu) < _WORKERS_MIN:
            hard_floor = max(1, int(cpu))
        if ram_budget > 0 and ram_budget < hard_floor:
            hard_floor = max(1, ram_budget)
        workers = max(workers, hard_floor)
    return max(1, workers)


def choose_workers(
    requested: int | None = None,
    available_bytes: int | None = None,
    per_worker_footprint_bytes: int | None = None,
) -> int:
    """Memory/CPU-aware worker count (R16 adaptive rule).

    When ``requested`` is ``None`` (the normal production case), returns the R16
    adaptive count ``clamp(min(cpu-2, RAM//footprint), 2, 14)``. An explicit
    ``requested`` value is still honoured (clamped to CPU and, when available,
    to the memory budget) — this keeps an operator's manual pin working.

    Note: the daemonic-parent case does NOT force ``1`` here — a daemonic parent
    simply uses threads (see :func:`pmap`) and threads are cheap, so the same
    memory/CPU clamp applies unchanged.
    """
    cpu = int(os.cpu_count() or 1)
    if requested is not None:
        workers = max(1, int(requested))
        workers = min(workers, cpu)
        footprint = int(per_worker_footprint_bytes or _PER_WORKER_BASELINE_BYTES)
        if available_bytes is not None and available_bytes > 0:
            budget = max(1, int(available_bytes // max(1, footprint)))
            workers = min(workers, budget)
        return workers
    return adaptive_worker_count(cpu, available_bytes, per_worker_footprint_bytes)


def _parent_is_daemonic() -> bool:
    """True when the current process is daemonic (cannot spawn children)."""
    try:
        return bool(multiprocessing.current_process().daemon)
    except Exception:  # pragma: no cover - defensive, never mis-classify as daemon
        return False


def pmap(worker, tasks, workers: int, progress_callback=None):
    """Run ``worker`` over ``tasks`` in order; returns a list of results.

    ``worker`` must be a module-level (picklable-by-reference) pure function of
    one task argument. Serial fallback when ``workers <= 1`` or ``len(tasks) < 2``.

    ``progress_callback`` (optional) is ``callable(done, total)`` invoked after
    each completed task (1-based ``done``, ``total`` = number of tasks), used by
    the R14 live gauge progress reporting. It is best-effort: a failure of the
    callback is swallowed and never affects the results.

    Executor selection (ZM-ZEGRID-R13):
      * non-daemonic parent -> ``ProcessPoolExecutor`` (fork on Linux: workers
        inherit the parent's imported modules, so no re-import cost);
      * daemonic parent     -> ``ThreadPoolExecutor`` (a daemonic process may not
        spawn child processes; threads preserve order and share module state).

    FAIL-SAFE: any failure of the parallel path (daemonic asserts, spawn/pickle
    errors, ``OSError``, ``BrokenProcessPool``, ...) degrades to the serial loop
    with a single WARN — a parallel failure can never crash the run.
    """
    tasks = list(tasks)
    total = len(tasks)

    def _report(done):
        if progress_callback is not None:
            try:
                progress_callback(int(done), int(total))
            except Exception:  # noqa: BLE001 - progress is never fatal
                pass

    if workers <= 1 or total < 2:
        out = []
        for i, t in enumerate(tasks):
            out.append(worker(t))
            _report(i + 1)
        return out

    if _parent_is_daemonic():
        from concurrent.futures import ThreadPoolExecutor

        Executor = ThreadPoolExecutor
        kind = "thread"
    else:
        from concurrent.futures import ProcessPoolExecutor

        Executor = ProcessPoolExecutor
        kind = "process"

    log.info(
        "[ZEGRID] pmap: %s pool, workers=%d, tasks=%d",
        kind,
        workers,
        total,
    )

    try:
        with Executor(max_workers=workers) as pool:
            out = []
            for i, res in enumerate(pool.map(worker, tasks)):
                out.append(res)
                _report(i + 1)
            return out
    except Exception as exc:  # noqa: BLE001 - fail-safe: never let parallelism crash
        log.warning(
            "[ZEGRID] parallel map unavailable (%s: %s); falling back to serial",
            type(exc).__name__,
            exc,
        )
        out = []
        for i, t in enumerate(tasks):
            out.append(worker(t))
            _report(i + 1)
        return out
