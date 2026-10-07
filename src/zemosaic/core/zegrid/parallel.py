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

DEFAULT_WORKERS = 4


def choose_workers(requested: int | None = None, available_bytes: int | None = None) -> int:
    """Memory-aware worker count: ``min(requested, cpu_count, mem_budget)``.

    Returns 1..4 (clamped to the CPU count and, when ``available_bytes`` is given,
    to the number of ~340 MiB baselines that fit). Never 0.

    Note: the daemonic-parent case does NOT force ``1`` here — a daemonic parent
    simply uses threads (see :func:`pmap`) and threads are cheap, so the same
    memory/CPU clamp applies unchanged.
    """
    cpu = int(os.cpu_count() or 1)
    workers = DEFAULT_WORKERS if requested is None else int(requested)
    workers = max(1, min(workers, cpu))
    if available_bytes is not None and available_bytes > 0:
        budget = max(1, int(available_bytes // _PER_WORKER_BASELINE_BYTES))
        workers = min(workers, budget)
    return workers


def _parent_is_daemonic() -> bool:
    """True when the current process is daemonic (cannot spawn children)."""
    try:
        return bool(multiprocessing.current_process().daemon)
    except Exception:  # pragma: no cover - defensive, never mis-classify as daemon
        return False


def pmap(worker, tasks, workers: int):
    """Run ``worker`` over ``tasks`` in order; returns a list of results.

    ``worker`` must be a module-level (picklable-by-reference) pure function of
    one task argument. Serial fallback when ``workers <= 1`` or ``len(tasks) < 2``.

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
    if workers <= 1 or len(tasks) < 2:
        return [worker(t) for t in tasks]

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
        len(tasks),
    )

    try:
        with Executor(max_workers=workers) as pool:
            return list(pool.map(worker, tasks))
    except Exception as exc:  # noqa: BLE001 - fail-safe: never let parallelism crash
        log.warning(
            "[ZEGRID] parallel map unavailable (%s: %s); falling back to serial",
            type(exc).__name__,
            exc,
        )
        return [worker(t) for t in tasks]
