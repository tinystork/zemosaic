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
"""

from __future__ import annotations

import os

# Fixed per-worker baseline (bytes): numpy/astropy/reproject/shapely import
# footprint (~330 MiB, matching the R6 STREAMING_BASELINE_KIB measurement).
_PER_WORKER_BASELINE_BYTES = 340 * 1024 * 1024

DEFAULT_WORKERS = 4


def choose_workers(requested: int | None = None, available_bytes: int | None = None) -> int:
    """Memory-aware worker count: ``min(requested, cpu_count, mem_budget)``.

    Returns 1..4 (clamped to the CPU count and, when ``available_bytes`` is given,
    to the number of ~340 MiB baselines that fit). Never 0.
    """
    cpu = int(os.cpu_count() or 1)
    workers = DEFAULT_WORKERS if requested is None else int(requested)
    workers = max(1, min(workers, cpu))
    if available_bytes is not None and available_bytes > 0:
        budget = max(1, int(available_bytes // _PER_WORKER_BASELINE_BYTES))
        workers = min(workers, budget)
    return workers


def pmap(worker, tasks, workers: int):
    """Run ``worker`` over ``tasks`` in order; returns a list of results.

    ``worker`` must be a module-level (picklable-by-reference) pure function of
    one task argument. Serial fallback when ``workers <= 1`` or ``len(tasks) < 2``.
    Uses ``ProcessPoolExecutor`` (fork on Linux: workers inherit the parent's
    imported modules, so there is no re-import cost and no spawn-time re-import).
    """
    tasks = list(tasks)
    if workers <= 1 or len(tasks) < 2:
        return [worker(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(worker, tasks))
