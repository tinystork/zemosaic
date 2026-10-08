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
import time

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

# ZM-ZEGRID-R20: per-cell STACKING concurrency cap (mirrors _WORKERS_MAX; the
# RAM//footprint term usually binds first on real mosaics, so the cap is a
# sanity ceiling, not the normal operating point).
_CELLS_IN_FLIGHT_MAX = 14

# Estimated per-worker footprint (bytes) for the GLOBAL-GAUGE phase. Each gauge
# worker reprojects a frame over a full-canvas / union bbox (the dominant cost),
# holding a float32 RGB plane + float64 temporaries; this is measurably heavier
# than the per-cell cache build, so it gets its own (larger) footprint.
_GAUGE_PER_WORKER_FOOTPRINT_BYTES = _PER_WORKER_BASELINE_BYTES + (1600 * 1200 * 3 * 8)

# Estimated per-worker footprint (bytes) for the CACHE-BUILD phase (per-cell
# patch reprojection — smaller working set than the full-canvas gauge).
_CACHE_PER_WORKER_FOOTPRINT_BYTES = _PER_WORKER_BASELINE_BYTES + (512 * 512 * 3 * 8)

# ZM-ZEGRID-R17 (R16-M1 fix): the static estimates above UNDER-count the real
# per-worker cost on a large mosaic. A gauge worker reprojects a frame over its
# footprint bbox, and the pairs pass over the UNION of the reference's and the
# frame's bboxes — which can approach the CANVAS area (e.g. ~18 Mpx on the real
# Caldwell canvas -> ~0.66-1.0 GiB/worker instead of ~0.39 GiB). The production
# call site therefore derives the footprint from the ACTUAL canvas / patch pixel
# area (helpers below); the constants remain as conservative fallbacks.
_CHANNELS_F64_TEMPS_BYTES = 3 * 8   # float64 RGB temporaries inside the worker
_CHANNELS_F32_PLANE_BYTES = 3 * 4   # the float32 RGB plane the task keeps

# ZM-ZEGRID-R17 (R16-M1 fix): the production decision budgets on a FRACTION of the
# available RAM so the workers' peak (which is on top of the main process and the
# product) cannot exhaust the machine. The rule itself is unchanged; the caller
# applies this headroom to the `available_bytes` it passes.
RAM_SAFETY_FRACTION = 0.8


def gauge_footprint_bytes(canvas_px: int) -> int:
    """Worst-case per-worker gauge footprint: the union bbox may approach the canvas.

    ``canvas_px`` is the canvas area in pixels (``width * height``). Using the whole
    canvas is a deliberate CONSERVATIVE upper bound (the pairs pass unions the
    reference's and the frame's bboxes); it prevents over-spawning on large mosaics
    where a static small-bbox estimate under-counts by ~2-2.6x.
    """
    px = max(0, int(canvas_px))
    return _PER_WORKER_BASELINE_BYTES + px * (_CHANNELS_F64_TEMPS_BYTES + _CHANNELS_F32_PLANE_BYTES)


def cache_footprint_bytes(patch_px: int) -> int:
    """Per-worker cache-build footprint (one per-cell patch reprojection)."""
    px = max(0, int(patch_px))
    return _PER_WORKER_BASELINE_BYTES + px * (_CHANNELS_F64_TEMPS_BYTES + _CHANNELS_F32_PLANE_BYTES)


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


def cells_in_flight(
    cpu: int,
    ram_budget_bytes: int | None,
    per_cell_footprint_bytes: int,
    max_cells: int | None = None,
) -> int:
    """ZM-ZEGRID-R20: how many cells to stack concurrently.

    ``clamp(min(cpu-2, RAM_budget // per_cell_footprint), 1, K)``.

    * ``cpu`` — logical CPU count (``os.cpu_count()``).
    * ``ram_budget_bytes`` — RAM budget for the whole in-flight batch (typically
      ``available * RAM_SAFETY_FRACTION``); ``None``/0 = no memory bound.
    * ``per_cell_footprint_bytes`` — peak bytes of ONE cell's stack (the in-memory
      bound for mode=inmem, or the streaming estimate incl. the ~340 MiB worker
      baseline for mode=stream).

    Never exceeds the RAM budget; degrades to 1 when RAM is tight. Deterministic
    (pure function) so the concurrency bound can be unit-checked.
    """
    cpu = int(cpu or 1)
    cpu_budget = max(1, cpu - _CPU_OS_RESERVE)
    footprint = max(1, int(per_cell_footprint_bytes))
    ram_budget = 0
    if ram_budget_bytes is not None and int(ram_budget_bytes) > 0:
        ram_budget = max(1, int(int(ram_budget_bytes) // footprint))
    target = min(cpu_budget, ram_budget) if ram_budget > 0 else cpu_budget
    cap = int(max_cells or _CELLS_IN_FLIGHT_MAX)
    return max(1, min(target, cap))


# ---------------------------------------------------------------------------
# ZM-ZEGRID-R22: joint mode + concurrency planner
# ---------------------------------------------------------------------------

# Relative per-cell wall-clock cost used to COMPARE modes in the planner. This is
# a PLANNING HEURISTIC (not a promised timing): streaming pays a small per-cell
# penalty vs in-memory (tile re-materialisation + memmap page-ins) but the
# canonical science (the dominant winsor/combine cost) is the SAME algorithm, so
# the penalty is modest and, when it unlocks >=2 concurrent cells, streaming wins
# the wall-clock on a CPU-bound stack.
_PLANNER_INMEM_COST = 1.0
_PLANNER_STREAM_COST = 1.2


def plan_cell_concurrency(
    cell_bounds,
    cpu: int,
    ram_budget_bytes: int | None,
    *,
    gpu_backend: bool = False,
    cache_build_workers: int = _WORKERS_MIN,
) -> dict:
    """Joint mode + concurrency planner (replaces the R20 two-step trap).

    ``cell_bounds`` is a sequence of ``(inmem_bound_bytes, stream_bound_bytes)``
    (one per non-empty cell). For EACH mode it derives the safe concurrency from
    the RAM budget (and CPU), compares the ESTIMATED makespan, and returns the
    winning plan plus the full candidate comparison (recorded in the manifest so
    the choice is auditable, never a silent ``serial`` surprise).

    * GPU backend: the GPU stack phase is SINGLE-OWNER (cells_in_flight == 1 for
      the stack) so no two cell processes contend for one device; the CPU cache
      build may still be parallel (bounded). This is stated explicitly in
      ``gpu_serialized``, not hidden.
    * When only one cell is active, the cache build is NOT forced to workers=1:
      ``cache_workers_per_cell`` rises to ``cache_build_workers``; for K>1 it is
      bounded to ``cpu // K`` so the total never explodes to ``K x cache_workers``.

    Returns a dict (JSON-serialisable) with ``mode``, ``cells_in_flight``,
    ``cache_workers_per_cell``, ``gpu_serialized`` and the ``candidates`` list.
    """
    bounds = [(int(a), int(b)) for a, b in cell_bounds]
    n_cells = len(bounds)
    if n_cells == 0:
        return {
            "mode": "stream", "cells_in_flight": 1, "cache_workers_per_cell": 1,
            "gpu_serialized": bool(gpu_backend), "candidates": [],
            "n_cells": 0,
        }
    cpu = int(cpu or 1)
    cpu_budget = max(1, cpu - _CPU_OS_RESERVE)
    ram_budget = int(ram_budget_bytes) if ram_budget_bytes else 0

    max_inmem = max(a for a, _b in bounds)
    max_stream = max(b for _a, b in bounds)

    def _in_flight(footprint: int) -> int:
        # Reuse the existing cells_in_flight seam (so tests can force a fixed
        # concurrency), then cap at n_cells (never more concurrent than cells).
        n = cells_in_flight(cpu, ram_budget if ram_budget > 0 else None, footprint)
        return max(1, min(n, n_cells))

    def _makespan(in_flight: int, per_cell_cost: float) -> float:
        waves = (n_cells + in_flight - 1) // in_flight
        return waves * per_cell_cost

    inmem_in_flight = _in_flight(max_inmem)
    stream_in_flight = _in_flight(max_stream)

    if gpu_backend:
        # Single GPU owner + bounded device memory: the STACK phase serialises
        # (cells_in_flight == 1) regardless of RAM, AND the mode is forced to
        # stream (the memmap provider + VRAM-bounded tiles never materialise
        # N x full-cell on the device, so the in-memory provider's RAM footprint
        # buys nothing). This is stated explicitly, not hidden.
        cache_workers_per_cell = max(1, int(cache_build_workers))
        candidates = [
            {
                "mode": "inmem", "cells_in_flight": 1,
                "max_bound_bytes": int(max_inmem),
                "per_cell_cost_rel": _PLANNER_INMEM_COST,
                "makespan_rel": _makespan(1, _PLANNER_INMEM_COST),
                "gpu_serialized": True,
            },
            {
                "mode": "stream", "cells_in_flight": 1,
                "max_bound_bytes": int(max_stream),
                "per_cell_cost_rel": _PLANNER_STREAM_COST,
                "makespan_rel": _makespan(1, _PLANNER_STREAM_COST),
                "gpu_serialized": True,
            },
        ]
        return {
            "mode": "stream",
            "cells_in_flight": 1,
            "cache_workers_per_cell": int(cache_workers_per_cell),
            "gpu_serialized": True,
            "cpu": cpu,
            "ram_budget_bytes": ram_budget,
            "n_cells": n_cells,
            "chosen_reason": (
                "GPU backend: single-device owner serialises the stack phase and "
                "forces stream mode (VRAM-bounded tiles; N x full-cell never "
                "materialised on the device)"
            ),
            "candidates": candidates,
        }

    def _makespan(in_flight: int, per_cell_cost: float) -> float:
        waves = (n_cells + in_flight - 1) // in_flight
        return waves * per_cell_cost

    inmem_makespan = _makespan(inmem_in_flight, _PLANNER_INMEM_COST)
    stream_makespan = _makespan(stream_in_flight, _PLANNER_STREAM_COST)

    # Choose the lower estimated makespan; break ties toward inmem (faster/cell).
    if stream_makespan < inmem_makespan - 1e-9:
        mode = "stream"
        chosen_in_flight = stream_in_flight
        chosen_reason = (
            f"streaming estimated makespan {stream_makespan:.3f} (cells_in_flight="
            f"{stream_in_flight}) < in-memory {inmem_makespan:.3f} "
            f"(cells_in_flight={inmem_in_flight})"
        )
    else:
        mode = "inmem"
        chosen_in_flight = inmem_in_flight
        chosen_reason = (
            f"in-memory estimated makespan {inmem_makespan:.3f} (cells_in_flight="
            f"{inmem_in_flight}) <= streaming {stream_makespan:.3f} "
            f"(cells_in_flight={stream_in_flight})"
        )

    # Bounded cache-build parallelism: full budget when one cell is active, else
    # cpu // K so the total workers never explode to K x cache_build_workers.
    cache_workers_per_cell = max(1, min(int(cache_build_workers), max(1, cpu // max(1, chosen_in_flight))))
    if chosen_in_flight == 1:
        cache_workers_per_cell = max(1, int(cache_build_workers))

    candidates = [
        {
            "mode": "inmem",
            "cells_in_flight": int(inmem_in_flight),
            "max_bound_bytes": int(max_inmem),
            "per_cell_cost_rel": _PLANNER_INMEM_COST,
            "makespan_rel": float(inmem_makespan),
            "gpu_serialized": bool(gpu_backend),
        },
        {
            "mode": "stream",
            "cells_in_flight": int(stream_in_flight),
            "max_bound_bytes": int(max_stream),
            "per_cell_cost_rel": _PLANNER_STREAM_COST,
            "makespan_rel": float(stream_makespan),
            "gpu_serialized": bool(gpu_backend),
        },
    ]

    return {
        "mode": mode,
        "cells_in_flight": int(chosen_in_flight),
        "cache_workers_per_cell": int(cache_workers_per_cell),
        "gpu_serialized": bool(gpu_backend),
        "cpu": cpu,
        "ram_budget_bytes": ram_budget,
        "n_cells": n_cells,
        "chosen_reason": chosen_reason,
        "candidates": candidates,
    }


def _parent_is_daemonic() -> bool:
    """True when the current process is daemonic (cannot spawn children)."""
    try:
        return bool(multiprocessing.current_process().daemon)
    except Exception:  # pragma: no cover - defensive, never mis-classify as daemon
        return False


def pmap(worker, tasks, workers: int, progress_callback=None, initializer=None, initargs=(), emit=None, meta=None, serial_fallback: bool = True):
    """Run ``worker`` over ``tasks`` in order; returns a list of results.

    ``worker`` must be a module-level (picklable-by-reference) pure function of
    one task argument. Serial fallback when ``workers <= 1`` or ``len(tasks) < 2``.

    ``serial_fallback`` (ZM-ZEGRID-R22): when True (default), a parallel-path
    failure degrades to the serial loop with the SAME worker (the historical
    fail-safe). When False, the exception is RE-RAISED instead of retrying serially
    — used by the GPU cell batch so a CuPy OOM/driver error is NOT retried with the
    SAME GPU config (which would repeat the error and crash); the caller then owns
    the one-shot exact-CPU whole-batch rerun.

    ``progress_callback`` (optional) is ``callable(done, total)`` invoked after
    each completed task (1-based ``done``, ``total`` = number of tasks), used by
    the R14 live gauge progress reporting. Best-effort: a failure of the callback
    is swallowed and never affects the results.

    ``initializer`` / ``initargs`` (optional, ZM-ZEGRID-R19): a child-startup
    callable forwarded to ``ProcessPoolExecutor`` (which runs it in every worker
    process, under fork AND spawn). ``ThreadPoolExecutor`` has no initializer, so
    for the thread path it is called DIRECTLY in the parent before the map (the
    parent's module state is shared with the threads). The parent also calls it
    up-front so the SERIAL fallback path has the same module state. This is what
    makes pooled workers start-method-agnostic (no fork-only copy-on-write
    globals).

    ``emit`` (optional) is ``callable(msg, lvl="INFO")`` used to surface the
    executor decision and any fail-safe fallback through the run's
    ``progress_callback`` (visible in the GUI log / run log / breadcrumbs) — the
    ZM-ZEGRID-R19 visibility requirement.

    ``meta`` (optional) is a mutable dict the caller may pass to receive the
    EFFECTIVE executor kind + timing (for the manifest diagnostic):
    ``executor`` ("serial"|"thread"|"process"), ``parent_daemon``, ``workers``,
    ``tasks``, ``seconds``, ``seconds_per_unit``, ``fallback``,
    ``fallback_reason``.

    Executor selection (ZM-ZEGRID-R13):
      * non-daemonic parent -> ``ProcessPoolExecutor`` (fork on Linux; workers
        inherit the parent's imported modules);
      * daemonic parent     -> ``ThreadPoolExecutor`` (a daemonic process may not
        spawn child processes; threads preserve order and share module state).

    FAIL-SAFE: any failure of the parallel path (daemonic asserts, spawn/pickle
    errors, ``OSError``, ``BrokenProcessPool``, ...) degrades to the serial loop
    with a WARN surfaced via BOTH ``log`` and ``emit`` — a parallel failure can
    never crash the run.
    """
    tasks = list(tasks)
    total = len(tasks)

    def _report(done):
        if progress_callback is not None:
            try:
                progress_callback(int(done), int(total))
            except Exception:  # noqa: BLE001 - progress is never fatal
                pass

    def _surf(msg, lvl="INFO"):
        if emit is not None:
            try:
                emit(msg, lvl)
            except Exception:  # noqa: BLE001 - surfacing is never fatal
                pass

    def _finish(meta_dict, start_ts, n_units):
        if meta_dict is None:
            return
        elapsed = float(time.perf_counter() - start_ts)
        meta_dict["seconds"] = elapsed
        meta_dict["seconds_per_unit"] = elapsed / max(1, n_units)

    # Call the initializer in the PARENT up-front so the serial/thread paths (and
    # the thread-shared module state) have it; the process path forwards it to the
    # executor so each child runs it too (spawn-safe).
    if initializer is not None:
        try:
            initializer(*tuple(initargs))
        except Exception as exc:  # noqa: BLE001 - never fatal
            log.warning("[ZEGRID] pmap initializer failed: %s", exc)

    parent_daemon = _parent_is_daemonic()
    if meta is not None:
        meta["parent_daemon"] = bool(parent_daemon)
        meta["workers"] = int(workers)
        meta["tasks"] = int(total)
        meta["fallback"] = False

    t0 = time.perf_counter()

    if workers <= 1 or total < 2:
        if meta is not None:
            meta["executor"] = "serial"
        _surf(f"[ZEGRID] pmap: serial (workers={workers}, tasks={total})")
        out = []
        for i, t in enumerate(tasks):
            out.append(worker(t))
            _report(i + 1)
        _finish(meta, t0, total)
        return out

    if parent_daemon:
        from concurrent.futures import ThreadPoolExecutor

        Executor = ThreadPoolExecutor
        kind = "thread"
    else:
        from concurrent.futures import ProcessPoolExecutor

        Executor = ProcessPoolExecutor
        kind = "process"

    if meta is not None:
        meta["executor"] = kind

    log.info(
        "[ZEGRID] pmap: %s pool, workers=%d, daemon=%s, tasks=%d",
        kind,
        workers,
        parent_daemon,
        total,
    )
    _surf(
        f"[ZEGRID] pmap: {kind} pool, workers={workers}, daemon={parent_daemon}, tasks={total}"
    )

    try:
        if kind == "thread":
            # ThreadPoolExecutor has no initializer; the parent already called it.
            with Executor(max_workers=workers) as pool:
                out = []
                for i, res in enumerate(pool.map(worker, tasks)):
                    out.append(res)
                    _report(i + 1)
        else:
            with Executor(
                max_workers=workers,
                initializer=initializer,
                initargs=tuple(initargs),
            ) as pool:
                out = []
                for i, res in enumerate(pool.map(worker, tasks)):
                    out.append(res)
                    _report(i + 1)
        _finish(meta, t0, total)
        return out
    except Exception as exc:  # noqa: BLE001 - fail-safe: never let parallelism crash
        if not serial_fallback:
            # ZM-ZEGRID-R22: re-raise instead of the SAME-config serial retry (the
            # caller classifies the error and owns the one-shot CPU rerun).
            raise
        log.warning(
            "[ZEGRID] parallel map unavailable (%s: %s); falling back to serial",
            type(exc).__name__,
            exc,
        )
        _surf(
            f"[ZEGRID] parallel map unavailable ({type(exc).__name__}: {exc}); "
            f"falling back to SERIAL — this phase will be slow",
            lvl="WARN",
        )
        if meta is not None:
            meta["executor"] = "serial"
            meta["fallback"] = True
            meta["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        out = []
        for i, t in enumerate(tasks):
            out.append(worker(t))
            _report(i + 1)
        _finish(meta, t0, total)
        return out
