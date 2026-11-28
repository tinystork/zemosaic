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

import concurrent.futures
import logging
import multiprocessing
import os
import queue
import threading
import time

# ZM-ZEGRID-R41: pure host-RAM contract + phase-accounting math. Imported as a
# submodule (no psutil, no product mode) so the opt-in planner branch below can
# account for SIMULTANEOUS phase totals without duplicating the arithmetic.
from . import ram_contract as rcontract

log = logging.getLogger(__name__)


class ParallelTaskError(RuntimeError):
    """A worker call failed at a known task index (NEVER a pool fallback).

    ZM-IO-REJECT-01: a deterministic worker failure (unreadable frame, science
    error, programming error) must propagate with its index and original cause.
    It must never be re-labelled ``parallel map unavailable`` nor replayed on the
    whole wave in serial.
    """

    def __init__(self, index: int, task, original: BaseException) -> None:
        self.index = int(index)
        self.task = task
        self.original = original
        super().__init__(
            f"parallel task failed at index {self.index} "
            f"({type(original).__name__}: {original})"
        )


class ParallelPoolError(RuntimeError):
    """The executor/pool itself failed after work had already been dispatched.

    No serial replay is performed: re-running already launched tasks could
    duplicate side effects and hides the real pool failure. A pool failure
    BEFORE any useful work may still take the historical bounded serial
    fallback (see :func:`pmap`).

    ``index``/``task`` name the task being collected when the pool died, when
    that is known. They are diagnostic only: a child-pool death does NOT prove
    OOM, a GPU-device failure or a science error.
    """

    def __init__(self, message: str, *, index: int | None = None, task=None) -> None:
        self.index = None if index is None else int(index)
        self.task = task
        super().__init__(message)


class ResultConsumeError(RuntimeError):
    """A parent-side RESULT CONSUMER failed while collecting a finished task.

    ZM-CELL-LIVE-COMMIT (lot 1): the consumer is the parent's per-cell
    transaction (accept -> durable commit -> verification -> progress -> safe
    cleanup). Such a failure is a PARENT transaction failure, not a worker, pool
    or science failure:

    * it is FATAL (never swallowed as telemetry);
    * it never triggers the historical whole-wave serial replay and never
      re-accepts an already committed cell;
    * the ORIGINAL exception is re-raised UNCHANGED by :func:`pmap` /
      :func:`pmap_reserved`, so a caller keeps its exact type (e.g. a
      ``StoreCommitNotDurable``) and can report the truth.

    This wrapper exists only INSIDE this module: it marks the failure as
    consumer-owned so the map's generic "executor failed after dispatch"
    reclassification can never mislabel it as a pool failure.
    """

    def __init__(self, original: BaseException, *, index: int | None = None) -> None:
        self.original = original
        self.index = None if index is None else int(index)
        super().__init__(
            "parent result consumer failed"
            + ("" if index is None else f" at index {self.index}")
            + f" ({type(original).__name__}: {original})"
        )


def _is_broken_pool_error(exc: BaseException) -> bool:
    try:
        from concurrent.futures.process import BrokenProcessPool

        return isinstance(exc, BrokenProcessPool)
    except Exception:  # pragma: no cover - defensive
        return False

# Fixed per-worker baseline (bytes): numpy/astropy/reproject/shapely import
# footprint (~330 MiB, matching the R6 STREAMING_BASELINE_KIB measurement).
# ZM-ZEGRID-R41 (L3): a SINGLE authoritative process-baseline source. The
# contract constant owns the value; this name is assigned FROM it so the two can
# never drift. The numerical value is unchanged (340 MiB), so the default
# product behaviour is identical.
_PER_WORKER_BASELINE_BYTES = rcontract.PROCESS_BASELINE_BYTES

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

    NOTE (ZM-ZEGRID-R36): this is a PURE clamping seam. A zero RAM quotient
    (budget < one cell footprint) legally degrades to ``1`` here, so this function
    alone does NOT prove a mode is runnable — a mode whose ``max_bound`` exceeds
    the budget is INFEASIBLE and is excluded by :func:`plan_cell_concurrency`
    (which never treats this clamp as a feasible serial plan).
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

# ZM-ZEGRID-CELL-MEMORY-BOUND: the concurrency DECISION keeps 10 % of the plan
# budget in reserve (allocator/OS headroom, and the Windows working-set
# overshoot). It is a decision headroom, never a smaller published budget: a
# one-cell reservation that only fits the FULL budget still runs, serially.
RESERVATION_BUDGET_HEADROOM_FRAC = 0.10


class ModeInfeasible(RuntimeError):
    """Raised when NO stacking mode fits the RAM budget (ZM-ZEGRID-R36).

    A mode whose worst-cell bound exceeds the budget is INFEASIBLE: it cannot be
    rescued by a concurrency of 1 (``cells_in_flight`` clamping the zero quotient
    to 1). The joint planner excludes such modes from the choice and raises this
    instead of silently reporting a plan the machine cannot hold.
    """


def _plan_cell_concurrency_phases(
    cell_bounds,
    cpu: int,
    ram_budget_bytes: int | None,
    *,
    gpu_backend: bool,
    cache_build_workers: int,
    shared_bytes: int | None,
    process_baseline_bytes: int | None,
    cache_worker_footprint_bytes: int | None,
    cache_worker_px: int | None,
    max_cells_in_flight: int | None = None,
    cell_reservations=None,
    reservation_source: str = "model",
    reservation_cell_id: str | None = None,
) -> dict:
    """ZM-ZEGRID-R41 opt-in phase accounting for :func:`plan_cell_concurrency`.

    Unlike the legacy branch (which bounds ONE cell's footprint by the budget and
    lets the concurrency clamp derive itself), this branch accounts for the
    SIMULTANEOUS host footprint of the whole in-flight batch, per phase::

        stack_total = shared + K * stack_per_cell_bound
        cache_per_cell = PROCESS_BASELINE + W * max(0, cache_worker_footprint - PROCESS_BASELINE)
        cache_total = shared + K * cache_per_cell
        aggregate   = max(stack_total, cache_total)

    ``shared`` is uncapped and counted ONCE. K is REDUCED from the CPU bound until
    the aggregate fits the budget; a mode whose ``K == 1`` aggregate still exceeds
    the budget is INFEASIBLE (one-cell refusal — never clamp-to-1). The GPU backend
    keeps its single-owner K == 1 and is checked with the SAME phase aggregate.

    The returned dict is the legacy schema PLUS explicit per-phase totals (only in
    this opt-in mode, so the default path stays byte-for-byte identical).
    """
    bounds = [
        (
            rcontract.require_nonneg_int(a, "cell inmem_bound_bytes"),
            rcontract.require_nonneg_int(b, "cell stream_bound_bytes"),
        )
        for a, b in cell_bounds
    ]
    n_cells = len(bounds)
    # ZM-ZEGRID-CELL-MEMORY-BOUND: per-CELL reservations (opt-in). Each entry is
    # the ONE cell's own bound for the mode it will actually execute. ``None``
    # keeps the historical plan-wide worst-cell constant.
    reservations = None
    if cell_reservations is not None:
        reservations = [
            rcontract.require_nonneg_int(r, "cell_reservations")
            for r in cell_reservations
        ]
        if len(reservations) != n_cells:
            raise ValueError(
                "cell_reservations must have one entry per cell "
                f"({len(reservations)} != {n_cells})"
            )
        if str(reservation_source) not in ("model", "measured"):
            raise ValueError(
                "reservation_source must be 'model' or 'measured', got "
                f"{reservation_source!r}"
            )
    # M-1: validate the RAW values BEFORE any int()/max() coercion. ``cpu=None`` is
    # the only tolerated sentinel (os.cpu_count robustness) and conservatively
    # becomes 1; otherwise CPU must be a non-negative int (bool refused), with an
    # effective floor of 1.
    if cpu is None:
        cpu = 1
    else:
        cpu = rcontract.require_nonneg_int(cpu, "cpu")
    cpu = max(1, cpu)
    # ``cache_build_workers`` must be a NON-NEGATIVE int (bool refused); a negative
    # value is REFUSED (never masked), and the effective floor of 1 is applied
    # only AFTER validation (0 -> 1 conservative).
    cache_build_workers = max(
        1, rcontract.require_nonneg_int(cache_build_workers, "cache_build_workers")
    )
    # Explicit UNBOUNDED is ONLY ``ram_budget_bytes=None`` (M1). A finite 0 is a
    # real, empty budget that refuses any positive aggregate; a negative finite
    # budget is refused outright.
    ram_budget = (
        None
        if ram_budget_bytes is None
        else rcontract.require_nonneg_int(ram_budget_bytes, "ram_budget_bytes")
    )
    # ``shared_bytes`` is optional (default 0); when SUPPLIED it is validated raw.
    shared = (
        0
        if shared_bytes is None
        else rcontract.require_nonneg_int(shared_bytes, "shared_bytes")
    )
    # ``process_baseline_bytes`` is optional (defaults to the single-source module
    # baseline); when SUPPLIED it is validated raw.
    pb = (
        int(_PER_WORKER_BASELINE_BYTES)
        if process_baseline_bytes is None
        else rcontract.require_nonneg_int(process_baseline_bytes, "process_baseline_bytes")
    )

    if cache_worker_footprint_bytes is not None:
        cwf = rcontract.require_nonneg_int(
            cache_worker_footprint_bytes, "cache_worker_footprint_bytes"
        )
        cwf_source = "explicit"
    elif cache_worker_px is not None:
        cwf = int(
            cache_footprint_bytes(
                rcontract.require_nonneg_int(cache_worker_px, "cache_worker_px")
            )
        )
        cwf_source = "derived(cache_footprint_bytes)"
    else:
        cwf = int(_CACHE_PER_WORKER_FOOTPRINT_BYTES)
        cwf_source = "default(_CACHE_PER_WORKER_FOOTPRINT_BYTES)"

    def _budget_txt() -> str:
        return "unbounded" if ram_budget is None else f"{ram_budget / 2**20:.1f} MiB"

    def _cache_workers_for_k(k: int) -> int:
        if k <= 1:
            return max(1, int(cache_build_workers))
        return max(1, min(int(cache_build_workers), max(1, cpu // int(k))))

    def _phase_keys(*, k, w, cpc, st, ct, agg) -> dict:
        return {
            "phase_accounting": True,
            "shared_bytes": shared,
            "process_baseline_bytes": pb,
            "cache_worker_footprint_bytes": cwf,
            "cache_worker_footprint_source": cwf_source,
            "cache_per_cell_bytes": int(cpc),
            "stack_total_bytes": int(st),
            "cache_total_bytes": int(ct),
            "aggregate_bytes": int(agg),
            "budget_bytes": (None if ram_budget is None else int(ram_budget)),
            "cells_in_flight": int(k),
        }

    if n_cells == 0:
        # Coherent empty schema: nothing runs, only ``shared`` is resident. A
        # finite budget smaller than ``shared`` must refuse (never silently pass).
        if ram_budget is not None and shared > ram_budget:
            raise ModeInfeasible(
                f"empty plan refused: shared {shared / 2**20:.1f} MiB > finite "
                f"budget {ram_budget / 2**20:.1f} MiB"
            )
        cpc = rcontract.cache_per_cell_bytes(pb, 0, cwf)
        return {
            "mode": "stream",
            "cells_in_flight": 0,
            "cache_workers_per_cell": 0,
            "gpu_serialized": bool(gpu_backend),
            "candidates": [],
            "n_cells": 0,
            "feasible": True,
            "cpu": cpu,
            "ram_budget_bytes": (None if ram_budget is None else ram_budget),
            "chosen_reason": "no cells to schedule (empty plan)",
            **_phase_keys(k=0, w=0, cpc=cpc, st=shared, ct=shared, agg=shared),
        }

    max_inmem = max(a for a, _b in bounds)
    max_stream = max(b for _a, b in bounds)
    # The STREAM candidate is bounded by the PER-CELL reservations when they are
    # supplied; without them the plan-wide worst streaming bound is used (the
    # in-memory candidate is never changed by reservations).
    stream_bound_bytes = (
        max(reservations) if reservations else int(max_stream)
    )
    # 10 % decision headroom: K is chosen against 90 % of the plan budget. The
    # FULL budget still governs the one-cell feasibility fallback below.
    decision_budget = (
        None
        if ram_budget is None
        else int(ram_budget * (1.0 - RESERVATION_BUDGET_HEADROOM_FRAC))
    )

    cpu_budget = max(1, cpu - _CPU_OS_RESERVE)
    max_k = max(1, min(cpu_budget, n_cells))
    if max_cells_in_flight is not None:
        # ZM-ZEGRID-R41-B: a bounded replan may NEVER grow the concurrency, even
        # when a later observation finds MORE RAM than the plan budget assumed.
        _cap = rcontract.require_nonneg_int(max_cells_in_flight, "max_cells_in_flight")
        max_k = max(1, min(max_k, max(1, int(_cap))))
    # ZM-ZEGRID-PER-CELL-SCHEDULER: the CONCURRENCY CAP of the per-cell dispatch
    # queue, computed ONCE here (it may only shrink on a bounded replan). It is
    # the CPU cap, further bounded by the CACHE phase: K cells x cache_per_cell
    # (built with the single-cell worker count, the conservative case) must fit
    # the same decision budget. Memory itself is gated per cell at dispatch.
    _cpc_for_cap = rcontract.cache_per_cell_bytes(
        pb, max(1, int(cache_build_workers)), cwf
    )
    _cache_cap = (
        int(max_k)
        if decision_budget is None
        else max(1, (max(0, int(decision_budget) - int(shared))) // max(1, int(_cpc_for_cap)))
    )
    dispatch_cap = int(max(1, min(int(max_k), int(_cache_cap), n_cells)))

    def _makespan(k: int, cost: float) -> float:
        waves = (n_cells + k - 1) // k
        return waves * cost

    def _search(bound: int, budget=None, max_k_override=None) -> dict:
        # ZM-ZEGRID-R41-B (rework-2): REUSE the established per-mode concurrency
        # seam as a conservative INITIAL cap (so an existing
        # ``zpar.cells_in_flight`` monkeypatch keeps governing K), then let the
        # simultaneous phase accounting REDUCE K further. K can only go DOWN.
        # ``budget`` defaults to the plan budget; the reservation path passes
        # the 90 % decision budget. ``max_k`` never GROWS (cap untouched).
        _budget = ram_budget if budget is None else budget
        _seam_budget = _budget if (_budget or 0) > 0 else None
        _seam_cap = cells_in_flight(cpu, _seam_budget, int(bound))
        _cap_k = int(max_k) if max_k_override is None else int(max_k_override)
        mode_max_k = max(1, min(_cap_k, int(_seam_cap), n_cells))
        k, st, ct, agg, cpc, w = rcontract.feasible_concurrency(
            budget_bytes=_budget,
            shared_bytes=shared,
            stack_per_cell_bound_bytes=int(bound),
            process_baseline_bytes=pb,
            cache_worker_footprint_bytes=cwf,
            cache_workers_for_k=_cache_workers_for_k,
            max_k=mode_max_k,
        )
        return {"k": int(k), "stack_total": st, "cache_total": ct, "aggregate": agg,
                "cache_per_cell": cpc, "cache_workers": w, "bound": int(bound),
                "decision_budget": _budget}

    def _candidate(mode: str, res: dict, cost: float) -> dict:
        feasible = res["k"] >= 1
        entry = {
            "mode": mode,
            "cells_in_flight": int(res["k"]),
            "max_bound_bytes": int(res["bound"]),
            "feasible": bool(feasible),
            "infeasible_reason": (
                None if feasible else (
                    f"one-cell {mode} aggregate {res['bound'] / 2**20:.1f} MiB + "
                    f"shared exceeds budget {_budget_txt()} (K==1 refused)"
                )
            ),
            "per_cell_cost_rel": cost,
            "makespan_rel": (_makespan(res["k"], cost) if feasible else None),
            "gpu_serialized": bool(gpu_backend),
        }
        if feasible:
            entry.update({
                "cache_workers_per_cell": int(res["cache_workers"]),
                "cache_per_cell_bytes": int(res["cache_per_cell"]),
                "stack_total_bytes": int(res["stack_total"]),
                "cache_total_bytes": int(res["cache_total"]),
                "aggregate_bytes": int(res["aggregate"]),
            })
        return entry

    if gpu_backend:
        # Single GPU owner: the stack phase is K == 1, but the host footprint of
        # that single wave is STILL accounted with the same phase math.
        w = max(1, int(cache_build_workers))
        cpc = rcontract.cache_per_cell_bytes(pb, w, cwf)
        st = rcontract.stack_phase_total_bytes(shared, 1, max_stream)
        ct = rcontract.cache_phase_total_bytes(shared, 1, cpc)
        agg = rcontract.aggregate_phase_bytes(st, ct)
        gpu_feasible = (ram_budget is None) or (agg <= ram_budget)
        candidates = [
            {
                "mode": "inmem", "cells_in_flight": 0,
                "max_bound_bytes": int(max_inmem), "feasible": False,
                "infeasible_reason": (
                    "not considered on the GPU backend (single-owner stream)"
                ),
                "per_cell_cost_rel": _PLANNER_INMEM_COST,
                "makespan_rel": _makespan(1, _PLANNER_INMEM_COST),
                "gpu_serialized": True,
            },
            {
                "mode": "stream", "cells_in_flight": 1,
                "max_bound_bytes": int(max_stream), "feasible": bool(gpu_feasible),
                "infeasible_reason": (
                    None if gpu_feasible else (
                        f"single-owner stream host aggregate {agg / 2**20:.1f} MiB "
                        f"> ram_budget {_budget_txt()}"
                    )
                ),
                "per_cell_cost_rel": _PLANNER_STREAM_COST,
                "makespan_rel": _makespan(1, _PLANNER_STREAM_COST),
                "gpu_serialized": True,
                "cache_workers_per_cell": int(w),
                "cache_per_cell_bytes": int(cpc),
                "stack_total_bytes": int(st),
                "cache_total_bytes": int(ct),
                "aggregate_bytes": int(agg),
            },
        ]
        if not gpu_feasible:
            exc = ModeInfeasible(
                "GPU backend requested but its host-RAM phase aggregate is "
                f"INFEASIBLE: stack_total {st / 2**20:.1f} MiB / cache_total "
                f"{ct / 2**20:.1f} MiB (aggregate {agg / 2**20:.1f} MiB) > RAM "
                f"budget {_budget_txt()} (n_cells={n_cells}). The "
                "product path must degrade to exact CPU before execution."
            )
            exc.candidates = candidates
            raise exc
        return {
            "mode": "stream",
            "cache_workers_per_cell": int(w),
            "gpu_serialized": True,
            "feasible": True,
            "cpu": cpu,
            "ram_budget_bytes": ram_budget,
            "n_cells": n_cells,
            "chosen_reason": (
                "GPU backend: single-device owner serialises the stack phase and "
                "forces stream mode; host phase aggregate (stack/cache) checked with "
                "shared counted once"
            ),
            "candidates": candidates,
            "reservation": {
                "budget_bytes": (None if ram_budget is None else int(ram_budget)),
                "decision_budget_bytes": (
                    None if decision_budget is None else int(decision_budget)
                ),
                "headroom_fraction": 0.0,
                "reservation_bytes": int(stream_bound_bytes),
                "source": str(reservation_source) if reservations else "model",
                "cell_id": (reservation_cell_id if reservations else None),
                "cells_in_flight": 1,
                "dispatch_cap": 1,
                "reason": "GPU backend is single-owner: K == 1 by contract",
            },
            **_phase_keys(k=1, w=w, cpc=cpc, st=st, ct=ct, agg=agg),
        }

    inmem = _search(max_inmem)
    reservation_note = None
    if reservations:
        # The per-cell reservation bounds the STREAM candidate, and K is chosen
        # against the 90 % decision budget. A single reservation that only fits
        # the FULL budget still runs: serial K == 1, with the reason recorded.
        stream = _search(
            int(stream_bound_bytes),
            budget=(decision_budget if decision_budget is not None else 0),
        )
        if stream["k"] < 1 and (
            ram_budget is None or int(stream_bound_bytes) <= ram_budget
        ):
            w1 = _cache_workers_for_k(1)
            cpc1 = rcontract.cache_per_cell_bytes(pb, int(w1), cwf)
            st1 = rcontract.stack_phase_total_bytes(shared, 1, int(stream_bound_bytes))
            ct1 = rcontract.cache_phase_total_bytes(shared, 1, cpc1)
            agg1 = rcontract.aggregate_phase_bytes(st1, ct1)
            if ram_budget is None or agg1 <= ram_budget:
                stream = {
                    "k": 1, "stack_total": st1, "cache_total": ct1,
                    "aggregate": agg1, "cache_per_cell": cpc1,
                    "cache_workers": int(w1), "bound": int(stream_bound_bytes),
                    "decision_budget": decision_budget,
                }
                reservation_note = (
                    f"one-cell reservation {int(stream_bound_bytes) / 2**20:.1f} MiB "
                    f"exceeds the "
                    f"{100.0 * (1.0 - RESERVATION_BUDGET_HEADROOM_FRAC):.0f}% decision "
                    f"budget; serial K=1 under the full budget {_budget_txt()}"
                )
    else:
        stream = _search(max_stream)
    inmem_feasible = inmem["k"] >= 1
    stream_feasible = stream["k"] >= 1

    if not inmem_feasible and not stream_feasible:
        _stream_num = (
            f"one-cell reservation {int(stream_bound_bytes) / 2**20:.1f} MiB "
            "(per-cell reservations)"
            if reservations
            else f"one-cell aggregate >= {(shared + max_stream) / 2**20:.1f} MiB"
        )
        raise ModeInfeasible(
            "no stacking mode fits the host phase aggregate budget "
            f"{_budget_txt()}: in-memory one-cell aggregate "
            f">= {(shared + max_inmem) / 2**20:.1f} MiB and streaming {_stream_num} "
            f"both exceed it (n_cells={n_cells})"
        )

    inmem_ms = _makespan(inmem["k"], _PLANNER_INMEM_COST) if inmem_feasible else None
    stream_ms = _makespan(stream["k"], _PLANNER_STREAM_COST) if stream_feasible else None

    if stream_feasible and (not inmem_feasible or stream_ms < inmem_ms - 1e-9):
        mode = "stream"
        res = stream
        if not inmem_feasible:
            chosen_reason = (
                f"in-memory one-cell aggregate INFEASIBLE (>= "
                f"{(shared + max_inmem) / 2**20:.1f} MiB > budget "
                f"{_budget_txt()}); streaming selected (K={stream['k']}, "
                f"aggregate {(stream['aggregate'] or 0) / 2**20:.1f} MiB, makespan "
                f"{stream_ms:.3f})"
            )
        else:
            chosen_reason = (
                f"streaming host aggregate makespan {stream_ms:.3f} (K={stream['k']}, "
                f"aggregate {(stream['aggregate'] or 0) / 2**20:.1f} MiB) < in-memory "
                f"{inmem_ms:.3f} (K={inmem['k']}, aggregate "
                f"{(inmem['aggregate'] or 0) / 2**20:.1f} MiB)"
            )
    else:
        mode = "inmem"
        res = inmem
        if not stream_feasible:
            chosen_reason = (
                f"in-memory selected (K={inmem['k']}, aggregate "
                f"{(inmem['aggregate'] or 0) / 2**20:.1f} MiB, makespan "
                f"{inmem_ms:.3f}); streaming INFEASIBLE (one-cell aggregate >= "
                f"{(shared + max_stream) / 2**20:.1f} MiB > budget "
                f"{_budget_txt()})"
            )
        else:
            chosen_reason = (
                f"in-memory host aggregate makespan {inmem_ms:.3f} (K={inmem['k']}, "
                f"aggregate {(inmem['aggregate'] or 0) / 2**20:.1f} MiB) <= streaming "
                f"{stream_ms:.3f} (K={stream['k']})"
            )

    return {
        "mode": mode,
        "cache_workers_per_cell": int(res["cache_workers"]),
        "gpu_serialized": bool(gpu_backend),
        "feasible": True,
        "cpu": cpu,
        "ram_budget_bytes": ram_budget,
        "n_cells": n_cells,
        "chosen_reason": chosen_reason,
        "candidates": [
            _candidate("inmem", inmem, _PLANNER_INMEM_COST),
            _candidate("stream", stream, _PLANNER_STREAM_COST),
        ],
        "reservation": {
            "budget_bytes": (None if ram_budget is None else int(ram_budget)),
            "decision_budget_bytes": (
                None
                if decision_budget is None
                else int(decision_budget)
            ),
            "headroom_fraction": (
                RESERVATION_BUDGET_HEADROOM_FRAC if reservations else 0.0
            ),
            # The number that bounded K: the per-cell reservation for the
            # STREAM plan, the plan-wide in-memory bound for an in-memory plan.
            "reservation_bytes": int(
                stream_bound_bytes if mode == "stream" else max_inmem
            ),
            "source": (
                str(reservation_source) if reservations else "model"
            ),
            "cell_id": (reservation_cell_id if reservations else None),
            "cells_in_flight": int(res["k"]),
            "dispatch_cap": int(dispatch_cap),
            "reason": reservation_note,
        },
        **_phase_keys(
            k=res["k"], w=res["cache_workers"], cpc=res["cache_per_cell"],
            st=res["stack_total"], ct=res["cache_total"], agg=res["aggregate"],
        ),
    }


def plan_cell_concurrency(
    cell_bounds,
    cpu: int,
    ram_budget_bytes: int | None,
    *,
    gpu_backend: bool = False,
    cache_build_workers: int = _WORKERS_MIN,
    shared_bytes: int | None = None,
    cache_worker_footprint_bytes: int | None = None,
    cache_worker_px: int | None = None,
    process_baseline_bytes: int | None = None,
    max_cells_in_flight: int | None = None,
    cell_reservations=None,
    reservation_source: str = "model",
    reservation_cell_id: str | None = None,
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

    ZM-ZEGRID-R41 OPT-IN phase accounting (added kwargs, ALL defaulting to the
    historical behaviour so every existing caller/result is byte-for-byte
    unchanged):

    * ``shared_bytes`` — host footprint counted ONCE (passed at all);
    * ``cache_worker_footprint_bytes`` — explicit cache-worker footprint;
    * ``cache_worker_px`` — derive it via :func:`cache_footprint_bytes`;
    * ``process_baseline_bytes`` — per-process floor (defaults to the module
      baseline).

    When ANY of ``shared_bytes`` / ``cache_worker_footprint_bytes`` /
    ``cache_worker_px`` / ``process_baseline_bytes`` is supplied, the planner
    accounts for the SIMULTANEOUS stack/cache phase totals (see
    :func:`_plan_cell_concurrency_phases`) and adds explicit per-phase keys to the
    result; otherwise it runs the legacy path verbatim (no new keys).

    ``cell_reservations`` / ``reservation_source`` / ``reservation_cell_id``
    (ZM-ZEGRID-CELL-MEMORY-BOUND, opt-in) replace the plan-wide worst-cell
    constant on the STREAM path by one reservation per ACTIVE cell (see
    :func:`zemosaic.core.zegrid.auto_layout.reservation_bytes`). K is then chosen
    against 90 % of the plan budget (10 % headroom) and the plan carries an
    auditable ``reservation`` block. ``None``/omitted makes the phase path
    compute the historical plan-wide stream bound, unchanged.

    ``max_cells_in_flight`` (ZM-ZEGRID-R41-B, opt-in, default ``None``) is a
    hard UPPER BOUND on the chosen concurrency, applied on BOTH branches. It is
    used by a bounded replan so a later (larger) observation can never buy a
    HIGHER concurrency than the plan already had (the accepted no-growth
    invariant). ``None`` keeps the exact historical behaviour.
    """
    if (
        shared_bytes is not None
        or cache_worker_footprint_bytes is not None
        or cache_worker_px is not None
        or process_baseline_bytes is not None  # L-1: supplied alone MUST activate
        or cell_reservations is not None
    ):
        # M-1: hand the RAW arguments to the strict helper (no int()/max()
        # pre-coercion in the outer hand-off).
        return _plan_cell_concurrency_phases(
            cell_bounds,
            cpu,
            ram_budget_bytes,
            gpu_backend=bool(gpu_backend),
            cache_build_workers=cache_build_workers,
            shared_bytes=shared_bytes,
            process_baseline_bytes=process_baseline_bytes,
            cache_worker_footprint_bytes=cache_worker_footprint_bytes,
            cache_worker_px=cache_worker_px,
            max_cells_in_flight=max_cells_in_flight,
            cell_reservations=cell_reservations,
            reservation_source=reservation_source,
            reservation_cell_id=reservation_cell_id,
        )
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
        if max_cells_in_flight is not None:
            # R41-B bounded replan: never grow past the previous concurrency.
            n = min(n, max(1, int(max_cells_in_flight)))
        return max(1, min(n, n_cells))

    def _makespan(in_flight: int, per_cell_cost: float) -> float:
        waves = (n_cells + in_flight - 1) // in_flight
        return waves * per_cell_cost

    if gpu_backend:
        # Single GPU owner + bounded device memory: the STACK phase serialises
        # (cells_in_flight == 1) regardless of RAM, AND the mode is forced to
        # stream (the memmap provider + VRAM-bounded tiles never materialise
        # N x full-cell on the device, so the in-memory provider's RAM footprint
        # buys nothing). This is stated explicitly, not hidden.
        #
        # ZM-ZEGRID-R36 (F2): the GPU path only considers its single stream mode,
        # BUT that mode is still RAM-bounded — a Cell whose streaming bound
        # exceeds the budget is NOT executable (the RAM budget must hold the
        # stream's host-side footprint). ``ram_budget <= 0`` (no budget) keeps the
        # historical unconditional behaviour. When the GPU stream is RAM-infeasible
        # the planner refuses to pretend the GPU is executable: it raises so the
        # PRODUCT path degrades loudly to exact CPU BEFORE execution and re-plans
        # with ``gpu_backend=False``.
        cache_workers_per_cell = max(1, int(cache_build_workers))
        gpu_stream_feasible = (ram_budget <= 0) or (int(max_stream) <= ram_budget)
        candidates = [
            {
                "mode": "inmem", "cells_in_flight": 0,
                "max_bound_bytes": int(max_inmem),
                "feasible": False,
                "infeasible_reason": (
                    "not considered on the GPU backend (single-owner stream)"
                ),
                "per_cell_cost_rel": _PLANNER_INMEM_COST,
                "makespan_rel": _makespan(1, _PLANNER_INMEM_COST),
                "gpu_serialized": True,
            },
            {
                "mode": "stream", "cells_in_flight": 1,
                "max_bound_bytes": int(max_stream),
                "feasible": bool(gpu_stream_feasible),
                "infeasible_reason": (
                    None if gpu_stream_feasible
                    else f"worst-cell stream bound {max_stream / 2**20:.1f} MiB > ram_budget"
                ),
                "per_cell_cost_rel": _PLANNER_STREAM_COST,
                "makespan_rel": _makespan(1, _PLANNER_STREAM_COST),
                "gpu_serialized": True,
            },
        ]
        if not gpu_stream_feasible:
            exc = ModeInfeasible(
                "GPU backend requested but its RAM-bounded streaming mode is "
                f"INFEASIBLE: worst-cell streaming bound {max_stream / 2**20:.1f} MiB "
                f"> RAM budget {ram_budget / 2**20:.1f} MiB (n_cells={n_cells}). The "
                "product path must degrade to exact CPU before execution."
            )
            exc.candidates = candidates  # audit trail on the raised signal
            raise exc
        return {
            "mode": "stream",
            "cells_in_flight": 1,
            "cache_workers_per_cell": int(cache_workers_per_cell),
            "gpu_serialized": True,
            "feasible": True,
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

    # ZM-ZEGRID-R36: a mode whose worst-cell bound exceeds the RAM budget is
    # INFEASIBLE — it must NOT be rescued by a concurrency of 1 (the
    # ``cells_in_flight`` seam clamps the zero quotient to 1, which previously
    # made an over-budget mode look like a valid serial plan). ``ram_budget <= 0``
    # (no budget) keeps the historical behaviour: every mode is feasible.
    def _mode_feasible(footprint: int) -> bool:
        return ram_budget <= 0 or int(footprint) <= ram_budget

    inmem_feasible = _mode_feasible(max_inmem)
    stream_feasible = _mode_feasible(max_stream)

    if not inmem_feasible and not stream_feasible:
        raise ModeInfeasible(
            "no stacking mode fits the RAM budget "
            f"{ram_budget / 2**20:.1f} MiB: in-memory worst-cell bound "
            f"{max_inmem / 2**20:.1f} MiB and streaming worst-cell bound "
            f"{max_stream / 2**20:.1f} MiB both exceed it (n_cells={n_cells})"
        )

    inmem_in_flight = _in_flight(max_inmem) if inmem_feasible else 0
    stream_in_flight = _in_flight(max_stream) if stream_feasible else 0
    inmem_makespan = _makespan(inmem_in_flight, _PLANNER_INMEM_COST) if inmem_feasible else None
    stream_makespan = _makespan(stream_in_flight, _PLANNER_STREAM_COST) if stream_feasible else None

    # Choose the lower estimated makespan among the FEASIBLE modes; break ties
    # toward inmem (faster/cell). An infeasible mode is never chosen (R36).
    if stream_feasible and (not inmem_feasible or stream_makespan < inmem_makespan - 1e-9):
        mode = "stream"
        chosen_in_flight = stream_in_flight
        if not inmem_feasible:
            chosen_reason = (
                f"in-memory mode INFEASIBLE (worst-cell bound "
                f"{max_inmem / 2**20:.1f} MiB > budget {ram_budget / 2**20:.1f} MiB); "
                f"streaming selected (cells_in_flight={stream_in_flight}, "
                f"makespan {stream_makespan:.3f})"
            )
        else:
            chosen_reason = (
                f"streaming estimated makespan {stream_makespan:.3f} (cells_in_flight="
                f"{stream_in_flight}) < in-memory {inmem_makespan:.3f} "
                f"(cells_in_flight={inmem_in_flight})"
            )
    else:
        mode = "inmem"
        chosen_in_flight = inmem_in_flight
        if not inmem_feasible:  # pragma: no cover - guarded by the raise above
            chosen_reason = "in-memory selected (only feasible mode)"
        elif not stream_feasible:
            # ZM-ZEGRID-R36 (F1): in-memory fits and streaming does NOT — a VALID
            # plan, never a formatting crash on the absent stream makespan.
            chosen_reason = (
                f"in-memory selected (cells_in_flight={inmem_in_flight}, "
                f"makespan {inmem_makespan:.3f}); streaming INFEASIBLE "
                f"(worst-cell bound {max_stream / 2**20:.1f} MiB > budget "
                f"{ram_budget / 2**20:.1f} MiB)"
            )
        else:
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
            "feasible": bool(inmem_feasible),
            "infeasible_reason": (
                None if inmem_feasible
                else f"worst-cell bound {max_inmem / 2**20:.1f} MiB > ram_budget"
            ),
            "per_cell_cost_rel": _PLANNER_INMEM_COST,
            "makespan_rel": (float(inmem_makespan) if inmem_makespan is not None else None),
            "gpu_serialized": bool(gpu_backend),
        },
        {
            "mode": "stream",
            "cells_in_flight": int(stream_in_flight),
            "max_bound_bytes": int(max_stream),
            "feasible": bool(stream_feasible),
            "infeasible_reason": (
                None if stream_feasible
                else f"worst-cell bound {max_stream / 2**20:.1f} MiB > ram_budget"
            ),
            "per_cell_cost_rel": _PLANNER_STREAM_COST,
            "makespan_rel": (float(stream_makespan) if stream_makespan is not None else None),
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


DISPATCH_REASON_NOTHING_FITS = "nothing_fits_serial_k1"

# ZM-ZEGRID-PER-CELL-SCHEDULER: the per-cell queue only replaces the plan-wide
# pool for the STREAMING execution mode. An in-memory plan sized K from the
# IN-MEMORY bounds, so dispatching it against stream-sized reservations would
# oversubscribe RAM; it keeps the historical whole-batch ``pmap`` untouched.
DISPATCH_EXECUTION_MODES = ("stream", "external")

THREAD_EXECUTOR_WARNING = (
    "ZeGrid: cell dispatch — thread executor (no process parallelism)"
)


def dispatch_is_applicable(chosen_mode, reservations, cap, n_tasks) -> bool:
    """True only when the per-cell queue may replace the historical ``pmap``.

    Structural guard for the OOM class found in review: an in-memory plan's K
    was derived from the in-memory bounds, so it may NEVER be dispatched against
    streaming reservations. Missing reservations, a single cell or a cap of 1
    keep the historical executor too.
    """
    return (
        str(chosen_mode) in DISPATCH_EXECUTION_MODES
        and bool(reservations)
        and int(cap) > 1
        and int(n_tasks) > 1
    )


def cell_lifecycle_line(kind, cell_id, *, frames=None, mode=None,
                        stack_s=None, status=None, attempt=None, generation=None,
                        payload_sha=None, phase=None, stage=None, error=None,
                        k=None, reservation_bytes=None) -> str:
    """ONE per-cell lifecycle line for the RUN LOG FILE (never the console only).

    ``kind`` is one of:

    * ``"QUEUED"``     — the cell belongs to the current wave but no worker was
      admitted for it yet (the batch is NOT "started");
    * ``"SUBMITTED"``  — the parent really submitted THIS cell to the executor
      (dispatch point, not a whole-batch announcement);
    * ``"RUNNING"``    — a worker really began the cell (reported by the worker
      telemetry, never inferred from a submission);
    * ``"START"``      — legacy alias kept for byte-compatible older logs;
    * ``"DONE"``       — stack_s + status (legacy batch: cell returned);
    * ``"COMPUTED"``  — stack_s + status, emitted at RESULT RECEPTION (the cell
      is computed and its payload exists in the parent, NOT yet durable);
    * ``"COMMITTING"`` — the parent began the durable transaction;
    * ``"COMMITTED"`` — attempt/generation/payload digest, emitted ONLY after the
      durable commit was verified (COMPUTED != COMMITTED);
    * ``"FAILED"``     — the cell failed, with the STAGE that failed;
    * ``"CANCELLED"`` — only a cell that really never started.

    Kept as a pure formatter so the exact text is testable.
    """
    token = str(kind).upper()
    if token == "START":
        return (
            f"ZeGrid: cell START id={cell_id} frames={int(frames or 0)} "
            f"mode={mode}"
        )
    if token == "QUEUED":
        return (
            f"ZeGrid: cell QUEUED id={cell_id} "
            f"frames={int(frames or 0)} mode={mode}"
        )
    if token == "SUBMITTED":
        return (
            f"ZeGrid: cell SUBMITTED id={cell_id} k={k} "
            f"reservation={int(reservation_bytes or 0)}B"
        )
    if token == "RUNNING":
        return (
            f"ZeGrid: cell RUNNING id={cell_id} phase={phase} "
            f"gen={int(generation or 0)}"
        )
    if token == "DONE":
        return (
            f"ZeGrid: cell DONE id={cell_id} stack_s={float(stack_s or 0.0):.1f} "
            f"status={status}"
        )
    if token == "COMPUTED":
        return (
            f"ZeGrid: cell COMPUTED id={cell_id} "
            f"stack_s={float(stack_s or 0.0):.1f} status={status}"
        )
    if token == "COMMITTING":
        return (
            f"ZeGrid: cell COMMITTING id={cell_id} gen={int(generation or 0)}"
        )
    if token == "COMMITTED":
        return (
            f"ZeGrid: cell COMMITTED id={cell_id} attempt={attempt} "
            f"gen={int(generation or 0)} "
            f"payload={str(payload_sha or '')[:16]}"
        )
    if token == "FAILED":
        return (
            f"ZeGrid: cell FAILED id={cell_id} stage={stage} "
            f"error={type(error).__name__ if isinstance(error, BaseException) else error}"
        )
    if token == "CANCELLED":
        return (
            f"ZeGrid: cell CANCELLED id={cell_id} gen={int(generation or 0)} "
            "(never started)"
        )
    raise ValueError(f"unknown lifecycle kind {kind!r}")


def fatal_lifecycle_line(error, *, cell_id=None, index=None, stage=None) -> str:
    """ONE fatal line, WRITTEN BEFORE the executor drain (never after it).

    The pre-L2 behaviour surfaced a fatal cell/consumer error only after
    ``shutdown(wait=True)`` returned, so a slow sibling could hide the failure
    for minutes. The parent now reports this line BEFORE the drain. It does not
    promise that the running worker stops: a started task is drained, and that
    is stated explicitly here.
    """
    token = type(error).__name__ if isinstance(error, BaseException) else "error"
    where = (
        f"cell={cell_id}" if cell_id is not None else f"index={index}"
    )
    return (
        f"ZeGrid: FATAL {stage or 'cell'} failure at {where} — {token}: "
        f"{error}. Already-finished cells stay committed; tasks the executor "
        "already started are drained before this error propagates (no forced "
        "kill, no replay)."
    )


def heartbeat_due(last_ts, now_ts, interval_s: float = 60.0) -> bool:
    """True when a rate-limited progress line may be written again."""
    return float(now_ts) - float(last_ts) >= float(interval_s)


def cells_heartbeat_line(committed, total, in_flight) -> str:
    """Rate-limited progress line: how many cells are durable, how many run."""
    return (
        f"ZeGrid: cells committed={int(committed)}/{int(total)} "
        f"in_flight={int(in_flight)}"
    )


def dispatch_wave_bound(chosen_mode, dispatch_cap, units_in_flight) -> int:
    """How many cells ONE wave may contain (ZM-ZEGRID-PER-CELL-SCHEDULER).

    The WAVE DRIVER must slice by the dispatch cap, never by the plan-wide K:
    for the external class the plan K is 1, and slicing by it made every wave a
    single cell, starving the per-cell queue (production defect #5). A
    non-dispatchable plan (in-memory, missing cap) keeps the historical K.
    """
    try:
        cap = int(dispatch_cap or 0)
    except (TypeError, ValueError):
        cap = 0
    k = max(1, int(units_in_flight or 1))
    if str(chosen_mode) in DISPATCH_EXECUTION_MODES and cap > 1:
        return int(cap)
    return int(k)


def external_dispatch_reservation(
    *,
    reservations_by_cell,
    budget_bytes,
    shared_bytes,
    cpu_count,
    cells_in_flight=1,
    source="model",
    n_cells=None,
):
    """Reservation block for the EXTERNAL pseudo-plan (same shape as a phase plan).

    The external class streams from disk: its per-cell reservation is the
    external host bound of the tile/batch the cell actually executes. The cap is
    derived ONCE here and is shrink-only: the CPU cap, further bounded by how
    many reservations fit the decision budget (90 % of the plan budget)::

        cap = min(cpu_cap, (decision - shared) // max_reservation, n_cells)

    Returns ``None`` when no sound reservation exists (empty map, a zero bound,
    or a non-positive budget): the caller then keeps the historical serial path
    rather than guessing.
    """
    rows = {
        str(cid): int(res)
        for cid, res in dict(reservations_by_cell or {}).items()
    }
    if not rows or any(v <= 0 for v in rows.values()):
        return None
    if budget_bytes is None:
        return None
    budget = int(budget_bytes)
    if budget <= 0:
        return None
    shared = max(0, int(shared_bytes))
    decision = int(budget * (1.0 - RESERVATION_BUDGET_HEADROOM_FRAC))
    max_res = max(rows.values())
    cell_id = max(rows, key=lambda cid: rows[cid])
    n = int(n_cells) if n_cells is not None else len(rows)
    cpu_cap = max(1, min(max(1, int(cpu_count or 1) - _CPU_OS_RESERVE), max(1, n)))
    # ZM-ZEGRID-PER-CELL-SCHEDULER (rework-3): ``dispatch_cap`` is ONLY the pool
    # size / worker upper bound — the CPU cap. The BUDGET is enforced per cell at
    # dispatch time by ``dispatch_fits(shared, running, reservation, decision)``,
    # so deriving the cap from the MAX reservation would silently reinstate the
    # plan-wide-worst-cell defect (one 14 GiB tail cell -> cap 1 -> no queue).
    cap = int(max(1, min(cpu_cap, max(1, n))))
    return {
        "budget_bytes": int(budget),
        "decision_budget_bytes": int(decision),
        "headroom_fraction": float(RESERVATION_BUDGET_HEADROOM_FRAC),
        "reservation_bytes": int(max_res),
        "source": str(source),
        "cell_id": str(cell_id),
        "cells_in_flight": int(cells_in_flight),
        "dispatch_cap": int(cap),
        "reason": (
            f"external dispatch cap = min(cpu {cpu_cap}, cells {n}) = {cap} "
            "(budget enforced per cell at dispatch, 90% headroom on "
            f"decision {decision} B)"
        ),
    }


def dispatch_reservation_bound(inmem_bound_bytes, stream_bound_bytes, chosen_mode):
    """The bound of the mode the cell ACTUALLY executes (never the other one)."""
    if str(chosen_mode) == "inmem":
        return int(inmem_bound_bytes)
    return int(stream_bound_bytes)


def dispatch_fits(running_bytes, reservation_bytes, shared_bytes, budget_bytes):
    """True when a NEW cell fits the remaining decision budget.

    ``budget_bytes is None`` is explicit UNBOUNDED. ``running_bytes`` is the sum
    of the reservations already in flight (shared is counted ONCE, outside).
    """
    if budget_bytes is None:
        return True
    return (
        int(shared_bytes) + int(running_bytes) + int(reservation_bytes)
        <= int(budget_bytes)
    )


def dispatch_log_line(cell_id, reservation_bytes, running_bytes, k, budget_bytes,
                      reason=None):
    """ONE deterministic dispatch line (budget/reservation/cell/k)."""

    def _mib(value):
        return "unbounded" if value is None else f"{int(value) / 2**20:.1f}MiB"

    return (
        f"ZeGrid: cell dispatch — cell={cell_id} "
        f"reservation={_mib(reservation_bytes)} "
        f"running={_mib(running_bytes)} k={int(k)} "
        f"budget={_mib(budget_bytes)}"
        + (f" — {reason}" if reason else "")
    )


def pmap_reserved(
    worker,
    tasks,
    reservations,
    *,
    shared_bytes=0,
    decision_budget_bytes=None,
    workers=1,
    cell_ids=None,
    progress_callback=None,
    initializer=None,
    initargs=(),
    emit=None,
    meta=None,
    serial_fallback=True,
    heartbeat=None,
    heartbeat_interval_s=5.0,
    consume=None,
    state_cb=None,
    on_fatal=None,
):
    """Per-CELL dispatch under a reservation budget (ZM-ZEGRID-PER-CELL-SCHEDULER).

    Unlike :func:`pmap` (a fixed pool of ``workers`` sized by the WORST cell),
    every cell is admitted only when ITS OWN reservation fits the remaining
    budget::

        shared + running_sum + reservation(cell) <= decision_budget

    A cell that does not fit is SKIPPED, never blocking the queue; the scan
    resumes at the next pending cell. When nothing is in flight and NOTHING
    fits, the first pending cell runs ALONE (the historical serial K=1) with an
    explicit reason. ``workers`` is the plan-time CPU/cache/dispatch cap and
    never grows here.

    Results are returned in TASK ORDER, exactly like :func:`pmap`, and every
    cell runs the SAME ``worker`` on the SAME task: the dispatched path is
    byte-identical to the sequential one.

    ``consume`` (ZM-CELL-LIVE-COMMIT, optional) is ``callable(index, result)``
    invoked by the PARENT, in the collecting thread, IMMEDIATELY when a result
    is received — before any other result of the same wave is awaited. Its
    return value (a light receipt) replaces the raw result in the returned list
    so no large payload reference is retained. A consumer exception is FATAL:
    it is never swallowed, never replays an already consumed task and is
    re-raised UNCHANGED (see :class:`ResultConsumeError`); results that are
    ALREADY finished are still handed to the consumer first, so a computed cell
    is never lost behind a still-running peer. ZM-M33-DRAIN-DURABLE: tasks the
    executor had already STARTED when a peer failed are also drained and
    consumed exactly once before the ORIGINAL error escapes; after a consumer
    (durable-commit) failure the policy closes and later results are reported
    as uncommitted (fail-closed store contract).
    """
    tasks = list(tasks)
    reservations = [int(r) for r in reservations]
    total = len(tasks)
    if len(reservations) != total:
        raise ValueError(
            "reservations must have one entry per task "
            f"({len(reservations)} != {total})"
        )
    if any(r < 0 for r in reservations):
        raise ValueError("reservations must be non-negative")
    if cell_ids is None:
        cell_ids = [str(i) for i in range(total)]
    else:
        cell_ids = [str(c) for c in cell_ids]
        if len(cell_ids) != total:
            raise ValueError("cell_ids must have one entry per task")
    shared = max(0, int(shared_bytes))
    cap = max(1, int(workers))
    budget = None if decision_budget_bytes is None else int(decision_budget_bytes)
    interval_s = (
        _heartbeat_interval(heartbeat_interval_s) if heartbeat is not None else None
    )
    t0 = time.perf_counter()
    _fatal_state = {"reported": False}

    if meta is not None:
        meta["scheduler"] = "per_cell_reservation"
        meta["workers"] = int(cap)
        meta["tasks"] = int(total)
        meta["shared_bytes"] = int(shared)
        meta["decision_budget_bytes"] = budget
        meta["events"] = []
        meta["dispatches"] = 0
        meta["max_concurrent"] = 0
        meta["heartbeat_calls"] = 0
        meta["heartbeat_errors"] = 0
        meta["heartbeat_mode"] = "none" if heartbeat is None else "pending"
        meta["heartbeat_interval_s"] = interval_s
        meta["consume_calls"] = 0
        meta["consume_failures"] = []
        meta["sibling_worker_failures"] = []
        meta["fatal_reported"] = False
        # ZM-M33-DRAIN-DURABLE: same truthful receipt bookkeeping as ``pmap``.
        meta["consumed_indices"] = []
        meta["uncommitted_outputs"] = []
        meta["pending_never_dispatched"] = []
        meta["consume_policy"] = "enabled" if consume is not None else "none"
        meta["drain"] = {
            "started": False,
            "seconds": 0.0,
            "primary": None,
            "cancelled": [],
            "consumed": [],
        }
        meta["submit_failures"] = []

    def _state(event):
        """Best-effort parent state event (SUBMITTED/COMPUTED/FAILED/...)."""
        if state_cb is None:
            return
        try:
            state_cb(dict(event))
        except Exception:  # noqa: BLE001 - telemetry is never fatal
            pass

    def _fatal(exc, *, index=None, stage="cell"):
        """Surface a fatal error BEFORE the executor drain (Nono LOW-1).

        Called at most once per wave. It never stops a worker that already
        started and never claims to: the caller's message states that finished
        cells stay committed and started tasks are drained.
        """
        if on_fatal is None:
            return
        if _fatal_state["reported"]:
            return
        _fatal_state["reported"] = True
        if meta is not None:
            meta["fatal_reported"] = True
        cid = (
            cell_ids[index]
            if index is not None and 0 <= int(index) < total
            else None
        )
        try:
            on_fatal(
                {
                    "event": "fatal",
                    "index": None if index is None else int(index),
                    "cell_id": cid,
                    "stage": str(stage),
                    "error": exc,
                }
            )
        except Exception:  # noqa: BLE001 - reporting is never fatal
            pass

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

    def _dispatch(idx, running_sum, k, reason=None):
        _surf(
            dispatch_log_line(
                cell_ids[idx], reservations[idx], running_sum + reservations[idx],
                k, budget, reason=reason,
            )
        )
        if meta is not None:
            meta["dispatches"] = int(meta.get("dispatches", 0)) + 1
            if k > int(meta.get("max_concurrent", 0)):
                meta["max_concurrent"] = int(k)
            meta["events"].append(
                {
                    "event": "dispatch",
                    "index": int(idx),
                    "cell_id": cell_ids[idx],
                    "reservation_bytes": int(reservations[idx]),
                    "running_bytes": int(running_sum + reservations[idx]),
                    "k": int(k),
                    "reason": reason,
                }
            )
        _state(
            {
                "event": "submitted",
                "index": int(idx),
                "cell_id": cell_ids[idx],
                "k": int(k),
            }
        )

    def _done(idx):
        if meta is not None:
            meta["events"].append(
                {"event": "done", "index": int(idx), "cell_id": cell_ids[idx]}
            )

    def _finish(meta_dict, n_units):
        if meta_dict is None:
            return
        elapsed = float(time.perf_counter() - t0)
        meta_dict["seconds"] = elapsed
        meta_dict["seconds_per_unit"] = elapsed / max(1, n_units)

    # The parent initializer runs up-front exactly like pmap (serial/thread
    # paths depend on it); children get the same initializer.
    if initializer is not None:
        try:
            initializer(*tuple(initargs))
        except Exception as exc:  # noqa: BLE001 - never fatal
            log.warning("[ZEGRID] pmap_reserved initializer failed: %s", exc)

    parent_daemon = _parent_is_daemonic()

    def _run_serial():
        monitor = None
        if heartbeat is not None:
            monitor = _HeartbeatMonitor(heartbeat, interval_s)
            monitor.start()
            if meta is not None:
                meta["heartbeat_mode"] = "serial_thread"
        out = []
        try:
            for i, task in enumerate(tasks):
                _dispatch(i, 0, 1, reason="serial_path")
                try:
                    _value = worker(task)
                except BaseException as _worker_exc:  # noqa: BLE001
                    _state(
                        {
                            "event": "failed",
                            "index": int(i),
                            "cell_id": cell_ids[i],
                            "stage": "worker",
                        }
                    )
                    raise
                _state(
                    {
                        "event": "computed",
                        "index": int(i),
                        "cell_id": cell_ids[i],
                    }
                )
                try:
                    # The consumer runs HERE: the result is committed before the
                    # next task is even dispatched, and nothing else is awaited.
                    if consume is not None:
                        out.append(consume(i, _value))
                        if meta is not None:
                            meta["consume_calls"] = int(
                                meta.get("consume_calls", 0)
                            ) + 1
                            meta["consumed_indices"].append(int(i))
                    else:
                        out.append(_value)
                finally:
                    _value = None
                _done(i)
                _report(i + 1)
        finally:
            if monitor is not None:
                monitor.stop_and_join()
                if meta is not None:
                    meta["heartbeat_calls"] = int(monitor.calls)
                    meta["heartbeat_errors"] = int(monitor.errors)
                    meta["heartbeat_last_error"] = monitor.last_error
        return out

    if cap <= 1 or total < 2:
        if meta is not None:
            meta["executor"] = "serial"
            meta["fallback"] = False
        _surf(f"[ZEGRID] per-cell dispatch: serial (cap={cap}, tasks={total})")
        out = _run_serial()
        _finish(meta, total)
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
        meta["fallback"] = False
    if kind == "thread":
        # A CPU-bound batch on threads does not parallelise (GIL): say it ONCE,
        # loudly, so a silent placebo can never be mistaken for a speed-up.
        _surf(THREAD_EXECUTOR_WARNING, "WARN")

    try:
        if kind == "thread":
            pool = Executor(max_workers=cap)
        else:
            pool = Executor(
                max_workers=cap, initializer=initializer, initargs=tuple(initargs)
            )
    except Exception as exc:  # noqa: BLE001 - executor creation only
        if not serial_fallback:
            raise
        log.warning(
            "[ZEGRID] per-cell dispatch pool unavailable (%s: %s); serial",
            type(exc).__name__, exc,
        )
        _surf(
            f"[ZEGRID] per-cell dispatch pool unavailable "
            f"({type(exc).__name__}: {exc}); falling back to SERIAL",
            lvl="WARN",
        )
        if meta is not None:
            meta["executor"] = "serial"
            meta["fallback"] = True
            meta["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        out = _run_serial()
        _finish(meta, total)
        return out

    out = [None] * total
    pending = list(range(total))
    running = {}
    running_sum = 0
    reported = 0
    progress = {"collected": 0}
    last_call = time.monotonic() if heartbeat is not None else None
    # Explicit, fail-closed consumption policy (same contract as ``pmap``).
    consume_state = {"enabled": consume is not None, "stop_reason": None}

    def _stop_consuming(reason):
        if consume_state["enabled"]:
            consume_state["enabled"] = False
            consume_state["stop_reason"] = str(reason)
            if meta is not None:
                meta["consume_policy"] = f"stopped: {reason}"

    def _record_uncommitted(idx, reason):
        if meta is not None:
            meta["uncommitted_outputs"].append(
                {"index": int(idx), "reason": str(reason)}
            )

    def _record_secondary(idx, kind, exc):
        detail = f"{type(exc).__name__}: {exc}"
        if meta is not None:
            if kind == "worker":
                meta["sibling_worker_failures"].append(
                    {"index": int(idx), "kind": "worker", "error": detail}
                )
            else:
                meta["consume_failures"].append(
                    {"index": int(idx), "kind": "consume", "error": detail}
                )
        if kind == "consume":
            _stop_consuming(f"consumer failure at cell {cell_ids[idx]} ({detail})")

    def _submit_failed(idx, exc):
        """Partial ``pool.submit`` failure: same drain/consume lifecycle (F1).

        No new dispatch after the failure, no replay, the ORIGINAL cause is
        preserved in the raised :class:`ParallelPoolError`, and every already
        started cell is drained and consumed exactly once (caller thread).
        """
        if meta is not None:
            meta["submit_failures"].append(
                {"index": int(idx), "error": f"{type(exc).__name__}: {exc}"}
            )
        _drain_cells_after_failure(idx, "submit", exc)
        raise ParallelPoolError(
            "per-cell dispatch submit failed for cell "
            f"{cell_ids[idx]} ({type(exc).__name__}: {exc})",
            index=idx, task=tasks[idx],
        ) from exc

    def _drain_cells_after_failure(_idx, _kind, _exc):
        """Drain the STARTED cells of a failing reserved wave, exactly once.

        ZM-M33-DRAIN-DURABLE: the fatal state is surfaced first, ONLY the
        never-started futures are cancelled (no new dispatch, no replay), then
        every already-started cell is received and consumed while the original
        error is pending. A consumer failure closes the policy (fail-closed
        store contract) and later results are named uncommitted.
        """
        nonlocal running_sum, last_call
        _fatal(
            _exc,
            index=_idx,
            stage={"consume": "consume", "submit": "pool"}.get(
                _kind, "worker"
            ),
        )
        if meta is not None:
            meta["drain"]["started"] = True
            meta["drain"]["primary"] = {
                "index": int(_idx),
                "kind": str(_kind),
                "error": f"{type(_exc).__name__}: {_exc}",
            }
            # Tasks that were never handed to the executor are named as such:
            # they never ran and can never have a durable output.
            meta["pending_never_dispatched"] = sorted(int(i) for i in pending)
        if _kind == "consume":
            _stop_consuming(
                f"primary consumer failure at cell {cell_ids[_idx]} "
                f"({type(_exc).__name__}: {_exc})"
            )
        started_at = time.perf_counter()
        try:
            # 1. Cancel ONLY futures that truly never started.
            for _fut in sorted(list(running), key=lambda f: int(running[f])):
                _cell = int(running[_fut])
                try:
                    if _fut.done() or not _fut.cancel():
                        continue
                except Exception:  # noqa: BLE001 - telemetry only
                    continue
                running.pop(_fut, None)
                running_sum -= reservations[_cell]
                if meta is not None:
                    meta["drain"]["cancelled"].append(_cell)
                _state(
                    {
                        "event": "cancelled",
                        "index": int(_cell),
                        "cell_id": cell_ids[_cell],
                    }
                )
            # 2. Receive + consume every ALREADY-STARTED cell.
            while running:
                timeout = None
                if heartbeat is not None:
                    timeout = max(
                        0.0, interval_s - (time.monotonic() - last_call)
                    )
                done, _ = concurrent.futures.wait(
                    list(running),
                    timeout=timeout,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for _fut in sorted(done, key=lambda f: int(running[f])):
                    _cell = int(running[_fut])
                    running.pop(_fut, None)
                    running_sum -= reservations[_cell]
                    _value = None
                    try:
                        _value = _fut.result()
                    except Exception as _sib_exc:  # noqa: BLE001
                        _state(
                            {
                                "event": "failed",
                                "index": int(_cell),
                                "cell_id": cell_ids[_cell],
                                "stage": "worker",
                            }
                        )
                        _record_secondary(_cell, "worker", _sib_exc)
                        continue
                    _state(
                        {
                            "event": "computed",
                            "index": int(_cell),
                            "cell_id": cell_ids[_cell],
                        }
                    )
                    try:
                        if consume is None:
                            out[_cell] = _value
                        elif not consume_state["enabled"]:
                            _record_uncommitted(
                                _cell,
                                consume_state["stop_reason"]
                                or "consumption policy closed",
                            )
                        else:
                            out[_cell] = consume(_cell, _value)
                            if meta is not None:
                                meta["consume_calls"] = int(
                                    meta.get("consume_calls", 0)
                                ) + 1
                                meta["consumed_indices"].append(_cell)
                                meta["drain"]["consumed"].append(_cell)
                            progress["collected"] += 1
                            _report(min(total, progress["collected"]))
                            _done(_cell)
                    except Exception as _cons_exc:  # noqa: BLE001 - fatal
                        _record_secondary(_cell, "consume", _cons_exc)
                    finally:
                        _value = None
                done = None
                if heartbeat is not None:
                    now = time.monotonic()
                    if now - last_call >= interval_s:
                        try:
                            heartbeat()
                            if meta is not None:
                                meta["heartbeat_calls"] = int(
                                    meta.get("heartbeat_calls", 0)
                                ) + 1
                        except Exception as _hb_exc:  # noqa: BLE001
                            if meta is not None:
                                meta["heartbeat_errors"] = int(
                                    meta.get("heartbeat_errors", 0)
                                ) + 1
                                meta["heartbeat_last_error"] = (
                                    f"{type(_hb_exc).__name__}: {_hb_exc}"
                                )
                        last_call = time.monotonic()
        finally:
            if meta is not None:
                meta["drain"]["seconds"] = float(
                    time.perf_counter() - started_at
                )

    try:
        while pending or running:
            progressed = True
            while pending and progressed:
                progressed = False
                if len(running) >= cap:
                    break
                for pos, idx in enumerate(pending):
                    reservation = reservations[idx]
                    if dispatch_fits(
                        running_sum, reservation, shared, budget
                    ):
                        pending.pop(pos)
                        # Submit FIRST: a failed submission is never reported
                        # as a dispatched cell (truthful dispatch bookkeeping).
                        try:
                            fut = pool.submit(worker, tasks[idx])
                        except Exception as exc:  # noqa: BLE001
                            _submit_failed(idx, exc)
                        _dispatch(idx, running_sum, len(running) + 1)
                        running_sum += reservation
                        running[fut] = idx
                        progressed = True
                        break
                if not running:
                    # Nothing is in flight and the head does not fit: run the
                    # FIRST pending cell ALONE (historical serial K=1).
                    idx = pending.pop(0)
                    try:
                        fut = pool.submit(worker, tasks[idx])
                    except Exception as exc:  # noqa: BLE001
                        _submit_failed(idx, exc)
                    _dispatch(
                        idx, 0, 1, reason=DISPATCH_REASON_NOTHING_FITS
                    )
                    running_sum = reservations[idx]
                    running[fut] = idx
                    progressed = True
                elif not progressed:
                    break
            if not running:
                # nothing submitted and nothing running cannot loop forever:
                # the block above always starts one cell.
                continue
            timeout = None
            if heartbeat is not None:
                now = time.monotonic()
                if now - last_call >= interval_s:
                    try:
                        heartbeat()
                        if meta is not None:
                            meta["heartbeat_calls"] = int(
                                meta.get("heartbeat_calls", 0)
                            ) + 1
                    except Exception as exc:  # noqa: BLE001
                        if meta is not None:
                            meta["heartbeat_errors"] = int(
                                meta.get("heartbeat_errors", 0)
                            ) + 1
                            meta["heartbeat_last_error"] = (
                                f"{type(exc).__name__}: {exc}"
                            )
                    last_call = time.monotonic()
                timeout = max(0.0, interval_s - (time.monotonic() - last_call))
            done, _ = concurrent.futures.wait(
                list(running),
                timeout=timeout,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            failure = None
            for fut in sorted(done, key=lambda f: int(running[f])):
                idx = running.pop(fut)
                running_sum -= reservations[idx]
                _value = None
                try:
                    _value = fut.result()
                except Exception as exc:  # noqa: BLE001
                    # Keep collecting the OTHER already-finished cells of this
                    # wait batch: a computed peer must not be lost because a
                    # sibling failed. Pending tasks are cancelled below.
                    _state(
                        {
                            "event": "failed",
                            "index": int(idx),
                            "cell_id": cell_ids[idx],
                            "stage": "worker",
                        }
                    )
                    if failure is None:
                        failure = (idx, "worker", exc)
                    continue
                # COMPUTED != COMMITTED: the result really exists in the parent
                # now, but nothing durable happened yet (the consumer owns it).
                _state(
                    {
                        "event": "computed",
                        "index": int(idx),
                        "cell_id": cell_ids[idx],
                    }
                )
                delivered = False
                try:
                    if consume is None:
                        out[idx] = _value
                        delivered = True
                    elif not consume_state["enabled"]:
                        # ZM-M33-DRAIN-DURABLE rework-1 (F2): a result of the
                        # SAME completed batch is never committed after a
                        # durable-commit failure; it is named uncommitted.
                        _record_uncommitted(
                            idx,
                            consume_state["stop_reason"]
                            or "consumption policy closed",
                        )
                    else:
                        # PARENT transaction, in this thread, at RECEIPT time —
                        # never a worker-side or future-done callback.
                        out[idx] = consume(idx, _value)
                        if meta is not None:
                            meta["consume_calls"] = int(
                                meta.get("consume_calls", 0)
                            ) + 1
                            meta["consumed_indices"].append(int(idx))
                        delivered = True
                except Exception as exc:  # noqa: BLE001 - FATAL, not telemetry
                    # Close the policy AT the collection point, so no sibling
                    # of this same batch can be committed after this failure.
                    _stop_consuming(
                        f"consumer failure at cell {cell_ids[idx]} "
                        f"({type(exc).__name__}: {exc})"
                    )
                    if failure is None:
                        failure = (idx, "consume", exc)
                    else:
                        _record_secondary(idx, "consume", exc)
                finally:
                    # Drop the large payload reference NOW (the future is also
                    # dropped below); only a light receipt may survive.
                    _value = None
                _done(idx)
                if consume is not None:
                    # Results may be committed OUT OF ORDER: count RECEPTIONS
                    # (only a real receipt), never a contiguous prefix.
                    if delivered:
                        progress["collected"] += 1
                        _report(min(total, progress["collected"]))
                else:
                    while reported < total and out[reported] is not None:
                        reported += 1
                        _report(reported)
            # ZM-CELL-LIVE-COMMIT R1 (payload lifetime): a Future OWNS its
            # result (``Future._result``) until the Future itself is dropped.
            # ``_value = None`` above is NOT enough: this ``done`` set and the
            # loop variable would keep every just-collected Future — and thus
            # its payload — alive across the NEXT (possibly very long) wait on a
            # slow peer. Drop those two references here, so only a light receipt
            # survives. This is plain reference ownership, not private mutation.
            done = None
            fut = None
            if failure is not None:
                _idx, _kind, _exc = failure
                _state(
                    {
                        "event": "failed",
                        "index": int(_idx),
                        "cell_id": cell_ids[_idx],
                        "stage": str(_kind),
                    }
                )
                # ZM-M33-DRAIN-DURABLE: fatal first, cancel ONLY pending,
                # then receive + durably consume every already-started cell.
                _drain_cells_after_failure(_idx, _kind, _exc)
                if _kind == "consume":
                    raise ResultConsumeError(_exc, index=_idx) from _exc
                if _is_broken_pool_error(_exc):
                    raise ParallelPoolError(
                        "executor failed while collecting cell "
                        f"{cell_ids[_idx]} ({type(_exc).__name__}: {_exc})",
                        index=_idx, task=tasks[_idx],
                    ) from _exc
                raise ParallelTaskError(_idx, tasks[_idx], _exc) from _exc
    except ResultConsumeError as _consume_exc:
        # A consumer failure keeps its ORIGINAL type for the caller (e.g. a
        # StoreCommitNotDurable). It was already fatal and never replays work.
        raise _consume_exc.original
    except BaseException as _fatal_exc:  # noqa: BLE001 - report before the drain
        _fatal(_fatal_exc, stage="worker")
        raise
    finally:
        # SINGLE shutdown owner: pending (never started) tasks are cancelled;
        # tasks the executor already STARTED were drained AND consumed above
        # (exactly once), so no committable cell is held behind the drain.
        try:
            pool.shutdown(wait=True, cancel_futures=True)
        except Exception:  # noqa: BLE001 - shutdown is best-effort
            pass
    _finish(meta, total)
    return out


def reservation_log_line(plan) -> str | None:
    """One auditable line per plan: budget, reservation, source, K, cell.

    ZM-ZEGRID-CELL-MEMORY-BOUND: the line names the CELL whose reservation bound
    K, so a later reader never has to guess whether a single plan-wide constant
    governed the concurrency (the defect this replaces).
    """
    if not isinstance(plan, dict):
        return None
    res = plan.get("reservation")
    if not isinstance(res, dict):
        return None

    def _mib(value):
        return "unbounded" if value is None else f"{int(value) / 2**20:.1f}MiB"

    return (
        "ZeGrid: cell memory reservation — "
        f"budget={_mib(res.get('budget_bytes'))} "
        f"decision_budget={_mib(res.get('decision_budget_bytes'))} "
        f"reservation={_mib(res.get('reservation_bytes'))} "
        f"source={res.get('source', 'model')} "
        f"K={int(plan.get('cells_in_flight', res.get('cells_in_flight', 0)) or 0)} "
        f"dispatch_cap={int(res.get('dispatch_cap') or 0)} "
        f"cell={res.get('cell_id')}"
        + (
            f" — {res['reason']}"
            if res.get("reason")
            else ""
        )
    )


def _parent_is_daemonic() -> bool:
    """True when the current process is daemonic (cannot spawn children)."""
    try:
        return bool(multiprocessing.current_process().daemon)
    except Exception:  # pragma: no cover - defensive, never mis-classify as daemon
        return False


class _HeartbeatMonitor:
    """Serial-path heartbeat thread (NO process is forked in this path).

    Runnable only when the caller is executing the plain serial loop (or the
    serial fallback after a pool has already failed). It is always stopped and
    joined before the wave returns; a live monitor after the join is a hard
    error (no daemon escape).
    """

    def __init__(self, heartbeat, interval_s: float):
        self._heartbeat = heartbeat
        self._interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="r55e1-heartbeat", daemon=True
        )
        self.calls = 0
        self.errors = 0
        self.last_error = None

    def _call(self):
        try:
            self._heartbeat()
            self.calls += 1
        except Exception as exc:  # noqa: BLE001 - fatal at the RUN boundary
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"

    def _run(self):
        while not self._stop.is_set():
            self._call()
            if self._stop.wait(self._interval_s):
                break

    def start(self):
        self._thread.start()

    def stop_and_join(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._interval_s * 2.0))
        if self._thread.is_alive():
            raise RuntimeError(
                "R55-E1 heartbeat monitor did not terminate before wave return"
            )


def _consume_finished_siblings(
    running, tasks, reservations, out, consume, meta, done_cb
) -> None:
    """Hand every ALREADY-FINISHED result of a failing wave to the consumer.

    ZM-CELL-LIVE-COMMIT (lot 1): when one task/consumer fails, the cells of the
    SAME wait batch that already completed are still valid results. They are
    consumed here (parent thread, at receipt) so a durable success is never held
    behind a still-running peer nor dropped by the failure path. This is NOT a
    replay: each future is consumed at most once, only when ``done()``, and any
    failure of this best-effort pass is recorded in ``meta`` without masking the
    original fatal error.

    ZM-M33-DRAIN-DURABLE rework-1 (N1): RETAINED FOR THE DIRECT WITNESSES ONLY.
    It is no longer on the production failure path: ``pmap`` uses
    ``_drain_started`` and ``pmap_reserved`` uses ``_drain_cells_after_failure``,
    which also drain the tasks already STARTED (this helper only touches
    ``done()`` futures) and enforce the fail-closed consumption policy.
    """
    if consume is None:
        return
    for _fut in sorted(list(running), key=lambda f: int(running[f])):
        try:
            if _fut.cancelled() or not _fut.done():
                continue
        except Exception:  # pragma: no cover - defensive
            continue
        _idx = running.pop(_fut, None)
        if _idx is None:  # pragma: no cover - defensive
            continue
        _sibling = None
        # The dict entry is gone; the loop local is the last owner besides the
        # (already popped) future itself. ``_drop`` below releases it so the
        # sibling payload can be freed instead of surviving to the wave end.
        try:
            _sibling = _fut.result()
        except Exception as _sib_exc:  # noqa: BLE001 - never mask the first error
            # ZM-CELL-LIVE-OBS (Nono LOW-2): a SIBLING WORKER failure is not a
            # consumer failure. It is recorded under its own label so an
            # operator never reads "consume failure" for a worker that raised.
            if meta is not None:
                meta.setdefault("sibling_worker_failures", []).append(
                    {
                        "index": int(_idx),
                        "kind": "worker",
                        "error": f"{type(_sib_exc).__name__}: {_sib_exc}",
                    }
                )
        else:
            try:
                out[_idx] = consume(_idx, _sibling)
                if meta is not None:
                    meta["consume_calls"] = int(meta.get("consume_calls", 0)) + 1
            except Exception as _sib_exc:  # noqa: BLE001 - never mask the first
                if meta is not None:
                    meta["consume_failures"].append(
                        {
                            "index": int(_idx),
                            "kind": "consume",
                            "error": f"{type(_sib_exc).__name__}: {_sib_exc}",
                        }
                    )
        finally:
            _sibling = None
        try:
            done_cb(_idx)
        except Exception:  # pragma: no cover - meta only
            pass


def _heartbeat_interval(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"heartbeat interval must be a number, got {value!r}")
    interval = float(value)
    if not (0.0 < interval <= 5.0):
        raise ValueError(
            f"heartbeat interval must be in (0, 5.0], got {value!r}"
        )
    return interval


def pmap(worker, tasks, workers: int, progress_callback=None, initializer=None, initargs=(), emit=None, meta=None, serial_fallback: bool = True, heartbeat=None, heartbeat_interval_s: float = 5.0, consume=None, state_cb=None, on_fatal=None):
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

    ``consume`` (ZM-CELL-LIVE-COMMIT, optional) is ``callable(index, result)``
    invoked by the PARENT as soon as a result is received, BEFORE any other
    result of the same batch is awaited. Its return value replaces the raw
    result in the returned list (a light receipt), so no large payload is
    retained. A consumer exception is FATAL and is re-raised UNCHANGED with its
    original type; already finished siblings are still consumed first, and no
    task is ever replayed or re-consumed.

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

    FAIL-SAFE: an executor CREATION failure (before any task is submitted)
    degrades to the serial loop with a WARN surfaced via BOTH ``log`` and
    ``emit``. A deterministic ``worker(task)`` / ``future.result()`` failure is
    NOT a pool-unavailability event: FIRST_COMPLETED detects it promptly, all
    still-PENDING futures are cancelled (no new dispatch, no replay), and tasks
    the executor already STARTED are drained AND durably consumed exactly once
    (reception consumer, heartbeat/progress cadence preserved) before the
    ORIGINAL exception propagates as :class:`ParallelTaskError` with its index
    and original cause. Secondary worker/consumer failures are explicit in
    ``meta`` and never mask the primary one. The wave is NEVER replayed in
    serial (see ZM-IO-REJECT-01). A mid-run pool failure propagates as
    :class:`ParallelPoolError` without replay.
    """
    tasks = list(tasks)
    total = len(tasks)
    interval_s = (
        _heartbeat_interval(heartbeat_interval_s)
        if heartbeat is not None else None
    )

    if meta is not None:
        meta["heartbeat_calls"] = 0
        meta["heartbeat_errors"] = 0
        meta["heartbeat_mode"] = "none" if heartbeat is None else "pending"
        meta["heartbeat_interval_s"] = interval_s
        meta["consume_calls"] = 0
        meta["consume_failures"] = []
        meta["sibling_worker_failures"] = []
        meta["fatal_reported"] = False
        # ZM-M33-DRAIN-DURABLE: truthful receipt bookkeeping. ``drain`` describes
        # what really happened AFTER the first fatal failure; a result that was
        # received but deliberately NOT handed to the consumer is named in
        # ``uncommitted_outputs`` (never silently counted as committed).
        meta["consumed_indices"] = []
        meta["uncommitted_outputs"] = []
        meta["consume_policy"] = "enabled" if consume is not None else "none"
        meta["drain"] = {
            "started": False,
            "seconds": 0.0,
            "primary": None,
            "cancelled": [],
            "consumed": [],
        }
        meta["submit_failed_index"] = None
        meta["never_submitted"] = []
        meta["submit_failure"] = None

    _fatal_state = {"reported": False}

    def _state(event):
        if state_cb is None:
            return
        try:
            state_cb(dict(event))
        except Exception:  # noqa: BLE001 - telemetry is never fatal
            pass

    def _cell_id(index):
        # ``pmap`` has no cell identity of its own: the caller maps the INDEX to
        # its own cell id (never a Stringified index, which could collide with a
        # real cell id).
        return None

    def _fatal(exc, *, index=None, stage="cell"):
        """Surface a fatal error BEFORE the drain (never after it)."""
        if on_fatal is None or _fatal_state["reported"]:
            return
        _fatal_state["reported"] = True
        if meta is not None:
            meta["fatal_reported"] = True
        try:
            on_fatal(
                {
                    "event": "fatal",
                    "index": None if index is None else int(index),
                    "cell_id": None if index is None else _cell_id(index),
                    "stage": str(stage),
                    "error": exc,
                }
            )
        except Exception:  # noqa: BLE001 - reporting is never fatal
            pass

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

    def _run_serial():
        monitor = None
        if heartbeat is not None:
            monitor = _HeartbeatMonitor(heartbeat, interval_s)
            try:
                monitor.start()
            except Exception as exc:  # noqa: BLE001 - active control must fail closed
                raise RuntimeError(
                    "R55-E1 active heartbeat monitor could not start; "
                    "refusing to run the wave unsampled"
                ) from exc
            if meta is not None:
                meta["heartbeat_mode"] = "serial_thread"
        out = []
        try:
            for i, t in enumerate(tasks):
                _state(
                    {
                        "event": "submitted",
                        "index": int(i),
                        "cell_id": _cell_id(i),
                    }
                )
                try:
                    _value = worker(t)
                except BaseException:  # noqa: BLE001
                    _state(
                        {
                            "event": "failed",
                            "index": int(i),
                            "cell_id": _cell_id(i),
                            "stage": "worker",
                        }
                    )
                    raise
                _state(
                    {
                        "event": "computed",
                        "index": int(i),
                        "cell_id": _cell_id(i),
                    }
                )
                try:
                    # Committed at RECEIPT time: nothing else is awaited between
                    # the worker returning and the parent transaction.
                    if consume is not None:
                        out.append(consume(i, _value))
                        if meta is not None:
                            meta["consume_calls"] = int(
                                meta.get("consume_calls", 0)
                            ) + 1
                            meta["consumed_indices"].append(int(i))
                    else:
                        out.append(_value)
                finally:
                    _value = None
                _report(i + 1)
        finally:
            if monitor is not None:
                monitor.stop_and_join()
                if meta is not None:
                    meta["heartbeat_calls"] = int(monitor.calls)
                    meta["heartbeat_errors"] = int(monitor.errors)
                    meta["heartbeat_last_error"] = monitor.last_error
        return out

    if workers <= 1 or total < 2:
        if meta is not None:
            meta["executor"] = "serial"
        _surf(f"[ZEGRID] pmap: serial (workers={workers}, tasks={total})")
        out = _run_serial()
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

    def _run_pool(pool):
        """Ordered results, with an optional parent-side heartbeat wait loop.

        ZM-IO-REJECT-01: a worker/future exception is ALWAYS surfaced as
        :class:`ParallelTaskError` with its task index and original cause. It is
        never reclassified as an executor failure, so it can never trigger the
        historical whole-wave serial replay. FIRST_COMPLETED detects the failure
        promptly; futures still PENDING are cancelled. Tasks the executor has
        already STARTED are not replayable and are drained by the single
        shutdown owner in :func:`pmap` before the exception propagates.
        """
        out = [None] * total
        recorded = set()
        reported = 0
        # Reception counter shared by the normal loop and the failure drain.
        progress = {"collected": 0}
        last_call = None
        futures = []
        # Explicit, fail-closed consumption policy. Enabled while the store is
        # healthy; switched OFF for good after a consumer (durable-commit)
        # failure, because the store contract refuses every later intake anyway
        # and blindly consuming would only manufacture silent non-commits.
        consume_state = {"enabled": consume is not None, "stop_reason": None}

        def _call_heartbeat(now):
            nonlocal last_call
            if heartbeat is None:
                return
            try:
                heartbeat()
                if meta is not None:
                    meta["heartbeat_calls"] = int(
                        meta.get("heartbeat_calls", 0)
                    ) + 1
            except Exception as exc:  # noqa: BLE001 - fatal at the RUN boundary
                if meta is not None:
                    meta["heartbeat_errors"] = int(
                        meta.get("heartbeat_errors", 0)
                    ) + 1
                    meta["heartbeat_last_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                _surf(
                    f"[ZEGRID] heartbeat sample failed ({type(exc).__name__}: "
                    f"{exc}); the next checkpoint still fails closed on its "
                    "fresh strict facts",
                    lvl="WARN",
                )
            last_call = now

        def _stop_consuming(reason):
            """Fail closed: never consume again after a durable-commit failure."""
            if consume_state["enabled"]:
                consume_state["enabled"] = False
                consume_state["stop_reason"] = str(reason)
                if meta is not None:
                    meta["consume_policy"] = f"stopped: {reason}"

        def _record_uncommitted(index, reason):
            """A result was RECEIVED but deliberately not committed."""
            if meta is not None:
                meta["uncommitted_outputs"].append(
                    {"index": int(index), "reason": str(reason)}
                )

        def _record_secondary(index, outcome):
            """Record a non-primary worker/consumer failure explicitly."""
            kind, exc = outcome
            detail = f"{type(exc).__name__}: {exc}"
            if meta is not None:
                if kind == "worker":
                    meta["sibling_worker_failures"].append(
                        {"index": int(index), "kind": "worker", "error": detail}
                    )
                elif kind == "consume":
                    meta["consume_failures"].append(
                        {"index": int(index), "kind": "consume", "error": detail}
                    )
            if kind == "consume":
                _stop_consuming(
                    f"consumer failure at index {int(index)} ({detail})"
                )

        def _collect(index, future, *, during_drain=False):
            """Deliver ONE finished future to the parent (consumer when given).

            Returns ``None`` on success, ``("worker"|"consume", exc)`` on a
            failure, or ``("unconsumed", None)`` when the consumption policy
            is closed (the result is then named in ``uncommitted_outputs``).
            The index is marked recorded FIRST, so neither the same wait batch
            nor the drain can ever deliver it twice. Never raises: the caller
            owns the single fatal-error decision and the sibling drain.
            """
            index = int(index)
            recorded.add(index)
            # ZM-CELL-LIVE-COMMIT R1 (payload lifetime): the Future OWNS its
            # result (``Future._result``). ``futures`` is the list this map keeps
            # for the whole batch, so the slot MUST stop pinning a collected
            # Future — otherwise every completed payload survives until the map
            # returns. Plain reference ownership; no private attribute is touched.
            try:
                futures[index] = None
            except (IndexError, TypeError):  # pragma: no cover - defensive
                pass
            value = None
            try:
                value = future.result()
            except Exception as exc:  # noqa: BLE001
                _state(
                    {
                        "event": "failed",
                        "index": int(index),
                        "cell_id": _cell_id(index),
                        "stage": "worker",
                    }
                )
                if during_drain:
                    # A sibling that fails while we drain is NOT the primary
                    # failure and is never allowed to mask it.
                    _record_secondary(index, ("worker", exc))
                return ("worker", exc)
            # The result really exists in the parent now (still NOT durable).
            _state(
                {
                    "event": "computed",
                    "index": int(index),
                    "cell_id": _cell_id(index),
                }
            )
            try:
                if consume is None:
                    out[index] = value
                    return None
                if not consume_state["enabled"]:
                    # Fail-closed drain: never retry a commit on a degraded
                    # store, never pretend the result was consumed.
                    _record_uncommitted(
                        index,
                        consume_state["stop_reason"]
                        or "consumption policy closed",
                    )
                    return ("unconsumed", None)
                out[index] = consume(index, value)
                if meta is not None:
                    meta["consume_calls"] = int(
                        meta.get("consume_calls", 0)
                    ) + 1
                    meta["consumed_indices"].append(index)
                return None
            except Exception as exc:  # noqa: BLE001 - consumer failure is FATAL
                # ZM-M33-DRAIN-DURABLE rework-1 (F2): close the policy HERE, at
                # the exact collection point, so siblings of the SAME completed
                # batch are never consumed after a durable-commit failure.
                _stop_consuming(
                    f"consumer failure at index {index} "
                    f"({type(exc).__name__}: {exc})"
                )
                return ("consume", exc)
            finally:
                # Only a light receipt may survive; the payload reference and
                # its completed future are dropped here.
                value = None

        def _cancel_pending():
            """Cancel ONLY futures that truly never started (never a replay)."""
            cancelled = []
            for _index, pending_future in enumerate(futures):
                if pending_future is None or _index in recorded:
                    continue
                try:
                    if pending_future.done():
                        continue
                    if pending_future.cancel():
                        recorded.add(int(_index))
                        futures[_index] = None
                        cancelled.append(int(_index))
                        _state(
                            {
                                "event": "cancelled",
                                "index": int(_index),
                                "cell_id": _cell_id(_index),
                            }
                        )
                except Exception:  # noqa: BLE001 - telemetry only
                    pass
            if meta is not None and cancelled:
                meta["drain"]["cancelled"].extend(cancelled)
            return cancelled

        def _drain_started():
            """Receive + consume the tasks the executor had ALREADY STARTED.

            ZM-M33-DRAIN-DURABLE: runs AFTER the fatal state was surfaced and
            the truly pending futures were cancelled. Every result is delivered
            by :func:`_collect` AT MOST ONCE; heartbeat/progress keep their
            cadence; a secondary worker/consumer failure is recorded without
            masking the primary one. A consumer failure closes the consumption
            policy (fail-closed store contract), and the still-received results
            are named as uncommitted instead of being silently committed.
            """
            if meta is not None:
                meta["drain"]["started"] = True
            started_at = time.perf_counter()
            try:
                while True:
                    pending_indices = [
                        i
                        for i, fut in enumerate(futures)
                        if fut is not None and i not in recorded
                    ]
                    if not pending_indices:
                        return
                    timeout = None
                    if heartbeat is not None:
                        timeout = max(
                            0.0, interval_s - (time.monotonic() - last_call)
                        )
                    done, _ = concurrent.futures.wait(
                        [futures[i] for i in pending_indices],
                        timeout=timeout,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for index in sorted(
                        i
                        for i in pending_indices
                        if futures[i] is not None and futures[i] in done
                    ):
                        outcome = _collect(
                            index, futures[index], during_drain=True
                        )
                        if outcome is None:
                            if consume is not None:
                                progress["collected"] += 1
                                _report(min(total, progress["collected"]))
                                if meta is not None:
                                    meta["drain"]["consumed"].append(int(index))
                            continue
                        _kind, _exc = outcome
                        if _kind == "consume":
                            # A secondary durable-commit failure is explicit and
                            # closes the policy for every later result.
                            _record_secondary(index, outcome)
                        # ("unconsumed", None) and a sibling worker failure were
                        # already recorded by ``_collect``/``_record_secondary``.
                    # R1: never keep a just-collected Future (and its payload)
                    # alive across the next wait on a slow peer.
                    done = None
                    if heartbeat is not None:
                        now = time.monotonic()
                        if now - last_call >= interval_s:
                            _call_heartbeat(now)
            finally:
                if meta is not None:
                    meta["drain"]["seconds"] = float(
                        time.perf_counter() - started_at
                    )

        def _raise_submit_failure(failure, task_list):
            """Partial ``pool.submit`` failure: same drain/consume lifecycle.

            ZM-M33-DRAIN-DURABLE rework-1 (F1): no replay, no new submission
            after the failure; the ORIGINAL cause is preserved; every task that
            WAS submitted is drained and durably consumed exactly once before
            the :class:`ParallelPoolError` escapes. The failed submission and
            the never-attempted tasks are bookkept separately and truthfully.
            """
            index, exc = failure
            _state(
                {
                    "event": "failed",
                    "index": int(index),
                    "cell_id": _cell_id(index),
                    "stage": "submit",
                }
            )
            _fatal(exc, index=index, stage="pool")
            if meta is not None:
                meta["drain"]["primary"] = {
                    "index": int(index),
                    "kind": "submit",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                meta["submit_failed_index"] = int(index)
                meta["never_submitted"] = list(range(int(index), total))
                meta["submit_failure"] = f"{type(exc).__name__}: {exc}"
            _cancel_pending()
            _drain_started()
            raise ParallelPoolError(
                f"executor submit failed after {int(index)} submitted task(s) "
                f"({type(exc).__name__}: {exc})",
                index=index,
                task=task_list[index],
            ) from exc

        def _raise_collect_failure(failure, task_list):
            """The ONE fatal-exit path of the collection loop.

            Order (ZM-M33-DRAIN-DURABLE): surface the fatal state BEFORE any
            wait, cancel ONLY the truly pending futures, then keep receiving
            and durably consuming the tasks the executor already STARTED before
            the ORIGINAL error is raised. Already-finished siblings are never
            lost behind a failure nor behind a still-running peer.
            """
            index, (kind, exc) = failure
            _state(
                {
                    "event": "failed",
                    "index": int(index),
                    "cell_id": _cell_id(index),
                    "stage": str(kind),
                }
            )
            # 1. Prompt, honest failure surfacing (before the drain wait).
            _fatal(exc, index=index, stage=str(kind))
            if meta is not None:
                meta["drain"]["primary"] = {
                    "index": int(index),
                    "kind": str(kind),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            # 2. A consumer failure means the durable store is degraded: stop
            #    committing (fail closed) instead of manufacturing non-commits.
            if kind == "consume":
                _stop_consuming(
                    f"primary consumer failure at index {int(index)} "
                    f"({type(exc).__name__}: {exc})"
                )
            # 3. Cancel ONLY tasks that never started; no new dispatch/replay.
            _cancel_pending()
            # 4. Drain the already-started tasks (consume successes once).
            _drain_started()
            # 5. The ORIGINAL failure is always what escapes.
            if kind == "consume":
                # Parent transaction failure: the ORIGINAL exception is
                # preserved (a StoreCommitNotDurable stays one).
                raise ResultConsumeError(exc, index=index) from exc
            if _is_broken_pool_error(exc):
                raise ParallelPoolError(
                    "executor failed while collecting task "
                    f"{index} ({type(exc).__name__}: {exc})",
                    index=index,
                    task=task_list[index],
                ) from exc
            raise ParallelTaskError(index, task_list[index], exc) from exc

        # Sample once BEFORE submitting (no worker is forked yet), then keep a
        # parent-side cadence while futures run.
        if heartbeat is not None:
            _call_heartbeat(time.monotonic())
        # ZM-M33-DRAIN-DURABLE rework-1 (F1): a PARTIAL submission failure is a
        # failure like any other: the already-submitted tasks must be drained
        # and durably consumed (or explicitly named uncommitted), never lost
        # behind the raised pool error, and no further task is submitted.
        futures = [None] * total
        submit_failure = None
        for _i, _task in enumerate(tasks):
            try:
                futures[_i] = pool.submit(worker, _task)
            except Exception as exc:  # noqa: BLE001 - partial submission
                submit_failure = (int(_i), exc)
                break
            _state(
                {
                    "event": "submitted",
                    "index": int(_i),
                    "cell_id": _cell_id(_i),
                }
            )
        if submit_failure is not None:
            _raise_submit_failure(submit_failure, tasks)
        pending = set(futures)
        while pending:
            if heartbeat is None:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
            else:
                now = time.monotonic()
                timeout = max(0.0, interval_s - (now - last_call))
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=timeout,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
            failure = None
            for index, future in enumerate(futures):
                if index in recorded or future is None or future not in done:
                    continue
                outcome = _collect(index, future)
                if outcome is not None:
                    if failure is None:
                        failure = (index, outcome)
                    else:
                        _record_secondary(index, outcome)
                    continue
                progress["collected"] += 1
            # R1: ``done`` (and the loop local) would keep the just-collected
            # Futures — and their payloads — alive across the next wait on a
            # slow peer. Release them before blocking again.
            done = None
            future = None
            if failure is not None:
                _raise_collect_failure(failure, tasks)
            if consume is not None:
                # R3: count RECEPTIONS. A contiguous-prefix counter would report
                # nothing while the FIRST task of the batch is still running,
                # even though later results are already received and committed.
                if progress["collected"]:
                    _report(min(total, progress["collected"]))
            else:
                while reported < total and reported in recorded:
                    reported += 1
                    _report(reported)
            if heartbeat is not None:
                now = time.monotonic()
                if now - last_call >= interval_s:
                    _call_heartbeat(now)
        for index, future in enumerate(futures):
            if index not in recorded and future is not None:
                outcome = _collect(index, future)
                if outcome is not None:
                    _raise_collect_failure((index, outcome), tasks)
                progress["collected"] += 1
                if consume is not None:
                    _report(min(total, progress["collected"]))
        if consume is None:
            while reported < total and reported in recorded:
                reported += 1
                _report(reported)
        if heartbeat is not None and meta is not None:
            meta["heartbeat_mode"] = "parent_wait"
        return out

    # Executor CREATION is the only failure that may still take the historical
    # bounded serial fallback (no useful work has started yet). A worker/future
    # failure below never falls back.
    try:
        if kind == "thread":
            pool = Executor(max_workers=workers)
        else:
            pool = Executor(
                max_workers=workers,
                initializer=initializer,
                initargs=tuple(initargs),
            )
    except Exception as exc:  # noqa: BLE001 - executor creation failed
        if not serial_fallback:
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
        out = _run_serial()
        _finish(meta, t0, total)
        return out

    # SINGLE shutdown owner for both success and failure. On failure the pool is
    # drained (wait=True) AFTER FIRST_COMPLETED detected the error and PENDING
    # futures were cancelled. This is a bounded drain of tasks the executor had
    # already started, never a replay and never the whole remaining wave.
    def _shutdown_pool(*, wait: bool) -> None:
        try:
            pool.shutdown(wait=wait, cancel_futures=True)
        except TypeError:  # pragma: no cover - older Python without cancel_futures
            try:
                pool.shutdown(wait=wait)
            except Exception:
                pass
        except Exception:
            pass

    try:
        out = _run_pool(pool)
    except ResultConsumeError as exc:
        # A parent result-consumer failure (durable commit) is re-raised with its
        # ORIGINAL type: it is never a pool failure and never a serial replay.
        # ZM-CELL-LIVE-OBS (Nono LOW-1): the fatal state is surfaced BEFORE the
        # drain, so a slow sibling can no longer hide it for minutes.
        _fatal(
            exc.original,
            index=getattr(exc, "index", None),
            stage="consume",
        )
        _shutdown_pool(wait=True)
        if meta is not None:
            meta["fallback"] = False
            meta["fallback_reason"] = f"{type(exc.original).__name__}: {exc.original}"
        raise exc.original
    except ParallelTaskError as exc:
        _fatal(
            exc.original if getattr(exc, "original", None) is not None else exc,
            index=getattr(exc, "index", None),
            stage="worker",
        )
        _shutdown_pool(wait=True)
        if meta is not None:
            meta["fallback"] = False
            meta["fallback_reason"] = None
        raise
    except ParallelPoolError as exc:
        _fatal(exc, index=getattr(exc, "index", None), stage="pool")
        _shutdown_pool(wait=True)
        if meta is not None:
            meta["fallback"] = False
            meta["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    except Exception as exc:  # noqa: BLE001 - pool/context failure after submission
        _fatal(exc, stage="pool")
        _shutdown_pool(wait=True)
        if isinstance(exc, RuntimeError) and "could not start" in str(exc):
            raise
        log.error(
            "[ZEGRID] parallel pool failed after dispatch (%s: %s); "
            "no serial replay is performed",
            type(exc).__name__,
            exc,
        )
        if meta is not None:
            meta["fallback"] = False
            meta["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        raise ParallelPoolError(
            f"executor failed after dispatch ({type(exc).__name__}: {exc})"
        ) from exc
    else:
        _shutdown_pool(wait=True)
        _finish(meta, t0, total)
        return out


# ---------------------------------------------------------------------------
# ZM-ZEGRID-R55-E2 — bounded CPU -> GPU producer/consumer pipeline
# ---------------------------------------------------------------------------
#
# Execution primitive only: no plan, no science, no scheduler authority.
#
#   * ``produce(task)``  — CPU-side staging of ONE canonical unit;
#   * ``consume(unit)``  — single-owner GPU work for that unit;
#   * ``release(unit)``  — deterministic release/cleanup of the staged unit.
#
# Contract (enforced here):
#
#   * ONE bounded asynchronous CPU producer thread and ONE consumer (GPU
#     owner), K == 1. The consumer is the CALLER thread (``run`` consumes
#     synchronously), so a long healthy GPU unit is never mistaken for a
#     deadlock by a fixed join budget. Only the producer is a thread, and it is
#     joined with a finite bound after a terminal state.
#   * an EXPLICIT bounded queue ``prefetch_units + 1`` plus explicit host and
#     device byte budgets. The producer blocks (backpressure) when either is
#     exhausted; it never drops, duplicates or grows a buffer after pressure.
#   * every staged unit takes exactly one host lease; the consumed unit takes
#     exactly one device lease. Both are released EXACTLY once on success,
#     cancellation, producer/consumer exception and fallback. A balanced ledger
#     is asserted before ``run`` returns.
#   * deterministic canonical result order (input task order).
#   * cancellation is a one-way latch: a cancelled pipeline publishes nothing,
#     drains the queue, releases every staged unit and joins the producer.
#     An in-flight uninterruptible ``consume`` call cannot be pre-empted; the
#     latch is honoured at the next unit/task boundary (documented, not hidden).
#
# No CuPy import, no CUDA initialisation: ``consume`` is the caller's callback.

PIPELINE_SCHEMA = "zemosaic-r55e2-bounded-pipeline-v1"
# ZM-ZEGRID-R56-BOUNDED-PREFETCH: the ledger gained an explicit DISK dimension so
# that on-disk aligned-cache bytes are never again booked as resident host RAM.
# The telemetry payload is a superset of v1; every v1 key is still emitted with
# its historical meaning, so existing consumers stay valid.
PIPELINE_SCHEMA_V2 = "zemosaic-r56-bounded-pipeline-v2"

#: Kinds understood by :class:`BufferLedger`. ``disk`` exists so a queued
#: payload's ON-DISK footprint is accounted exactly and separately from the
#: resident host reservation it also carries.
LEDGER_KINDS = ("host", "device", "disk")


def _intervals_union_intersection_s(a, b) -> float:
    """Length of the intersection of two unions of half-open intervals.

    Intervals are ``(start, end)`` tuples in the SAME monotonic clock. Used to
    prove real producer/consumer overlap from observed work intervals only: a
    nonzero result means work genuinely ran concurrently, never that two
    threads merely existed.
    """

    def _merge(items):
        merged = []
        for start, end in sorted(
            (float(s), float(e)) for s, e in items if float(e) > float(s)
        ):
            if merged and start <= merged[-1][1]:
                if end > merged[-1][1]:
                    merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    left = _merge(a)
    right = _merge(b)
    total = 0.0
    i = j = 0
    while i < len(left) and j < len(right):
        lo = max(left[i][0], right[j][0])
        hi = min(left[i][1], right[j][1])
        if hi > lo:
            total += hi - lo
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return float(total)


def _interval_hits_any(interval, others) -> bool:
    start, end = float(interval[0]), float(interval[1])
    for other_start, other_end in others:
        if min(end, float(other_end)) > max(start, float(other_start)):
            return True
    return False


class PipelineError(RuntimeError):
    """Base class for bounded pipeline contract/execution failures."""


class PipelineCancelled(PipelineError):
    """The pipeline was cancelled; no result may be published."""


class PipelineAccountingError(PipelineError):
    """The buffer ledger or a byte budget is inconsistent (fail closed)."""


def _pipeline_int(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineError(
            f"{name} must be a strict int, got {type(value).__name__}={value!r}"
        )
    if int(value) < int(minimum):
        raise PipelineError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


class BufferLedger:
    """Exact host/device/disk byte accounting with blocking reservation.

    ``reserve`` blocks until the requested bytes fit the explicit budget, so a
    caller can never exceed the envelope. A request larger than the whole budget
    fails closed immediately (it could never complete).

    Three independent dimensions are tracked:

    * ``host`` — RESIDENT process memory reserved by a queued/consumed payload;
    * ``device`` — GPU workspace reserved by the consuming unit;
    * ``disk`` — ON-DISK bytes retained for a queued/consumed payload.

    ``disk`` is deliberately a separate ledger: billing aligned-cache FILE bytes
    against the resident host budget conflated two different resources and is
    what made ``prefetch >= 1`` look unaffordable (ZM-ZEGRID-R56).
    """

    def __init__(
        self,
        *,
        host_budget_bytes: int,
        device_budget_bytes: int,
        disk_budget_bytes: int = 0,
        clock=time.monotonic,
    ) -> None:
        self.host_budget_bytes = _pipeline_int(
            host_budget_bytes, "host_budget_bytes", minimum=1
        )
        self.device_budget_bytes = _pipeline_int(
            device_budget_bytes, "device_budget_bytes", minimum=1
        )
        self.disk_budget_bytes = _pipeline_int(
            disk_budget_bytes, "disk_budget_bytes", minimum=0
        )
        self._clock = clock
        self._cond = threading.Condition()
        self._host_used = 0
        self._device_used = 0
        self._disk_used = 0
        self._host_peak = 0
        self._device_peak = 0
        self._disk_peak = 0
        self._host_leases: dict[int, int] = {}
        self._device_leases: dict[int, int] = {}
        self._disk_leases: dict[int, int] = {}
        self._next_lease = 1
        self._wait_events = 0
        self._wait_started_events = 0
        self._wait_s = 0.0
        self._reconciliations = 0
        self._surplus_released_bytes = 0
        self._budget_reductions: list[dict] = []

    def _capacity(self, kind: str) -> int:
        if kind == "host":
            return int(self.host_budget_bytes)
        if kind == "device":
            return int(self.device_budget_bytes)
        if kind == "disk":
            return int(self.disk_budget_bytes)
        raise PipelineAccountingError(f"unknown lease kind {kind!r}")

    def _used(self, kind: str) -> int:
        if kind == "host":
            return int(self._host_used)
        if kind == "device":
            return int(self._device_used)
        if kind == "disk":
            return int(self._disk_used)
        raise PipelineAccountingError(f"unknown lease kind {kind!r}")

    def _set_used(self, kind: str, value: int) -> None:
        if kind == "host":
            self._host_used = int(value)
            self._host_peak = max(self._host_peak, int(value))
        elif kind == "device":
            self._device_used = int(value)
            self._device_peak = max(self._device_peak, int(value))
        elif kind == "disk":
            self._disk_used = int(value)
            self._disk_peak = max(self._disk_peak, int(value))
        else:
            raise PipelineAccountingError(f"unknown lease kind {kind!r}")

    def _leases(self, kind: str) -> dict[int, int]:
        if kind == "host":
            return self._host_leases
        if kind == "device":
            return self._device_leases
        if kind == "disk":
            return self._disk_leases
        raise PipelineAccountingError(f"unknown lease kind {kind!r}")

    def _set_capacity(self, kind: str, value: int) -> None:
        if kind == "host":
            self.host_budget_bytes = int(value)
        elif kind == "device":
            self.device_budget_bytes = int(value)
        elif kind == "disk":
            self.disk_budget_bytes = int(value)
        else:
            raise PipelineAccountingError(f"unknown lease kind {kind!r}")

    def reconcile(self, kind: str, lease: int, actual_bytes: int) -> int:
        """Shrink a PRE-ACQUIRED lease to the measured/actual payload size.

        The producer reserves a conservative prediction BEFORE allocating. Once
        the real payload exists its actual bytes must be <= that prediction:
        an underestimate FAILS CLOSED here (the allocation already happened, so
        approving it would silently break the budget). Surplus is released in
        the same critical section, so there is never an uncredited gap between
        the conservative reservation and the smaller actual payload.

        Returns the surplus released.
        """
        actual = _pipeline_int(actual_bytes, f"{kind}_actual_bytes", minimum=0)
        leases = self._leases(kind)
        with self._cond:
            if lease not in leases:
                raise PipelineAccountingError(
                    f"unknown or already released {kind} lease {lease}"
                )
            reserved = int(leases[lease])
            if actual > reserved:
                raise PipelineAccountingError(
                    f"{kind} actual payload {actual} B exceeds the pre-acquired "
                    f"reservation {reserved} B: the prediction was too small "
                    "and the extra allocation cannot be approved"
                )
            surplus = int(reserved) - int(actual)
            if surplus:
                used = self._used(kind) - surplus
                if used < 0:
                    raise PipelineAccountingError(
                        f"{kind} ledger underflow while reconciling {surplus} B"
                    )
                leases[lease] = int(actual)
                self._set_used(kind, used)
                self._reconciliations += 1
                self._surplus_released_bytes += int(surplus)
                self._cond.notify_all()
            return int(surplus)

    def reduce_budget(self, kind: str, new_capacity: int) -> int:
        """SHRINK-ONLY budget reduction (fresh external observation).

        Never grows a budget. Used by the RUN-time fresh-disk revalidation: the
        caller re-observes the volume and may only lower the disk capacity.
        Raises if the caller tries to raise it or passes a negative value.
        """
        new = _pipeline_int(new_capacity, f"{kind}_new_capacity", minimum=0)
        with self._cond:
            current = self._capacity(kind)
            if new > current:
                raise PipelineAccountingError(
                    f"refusing to GROW the {kind} budget {current} -> {new}"
                )
            if new != current:
                self._set_capacity(kind, new)
                self._budget_reductions.append(
                    {"kind": str(kind), "from_bytes": int(current), "to_bytes": int(new)}
                )
            self._cond.notify_all()
            return int(new)

    def reserve(
        self,
        kind: str,
        nbytes: int,
        *,
        cancel_event: threading.Event | None = None,
        poll_s: float = 0.05,
    ) -> int:
        """Reserve ``nbytes`` and return a unique lease id (blocking)."""
        want = _pipeline_int(nbytes, f"{kind}_bytes", minimum=0)
        capacity = self._capacity(kind)
        if want > capacity:
            raise PipelineAccountingError(
                f"{kind} request {want} B exceeds the explicit budget "
                f"{capacity} B"
            )
        waited = False
        started = 0.0
        with self._cond:
            while self._capacity(kind) - self._used(kind) < want:
                # A concurrent SHRINK can make the pending request structurally
                # impossible. Fail closed instead of waiting forever for a
                # capacity that can no longer cover it.
                if self._capacity(kind) < want:
                    raise PipelineAccountingError(
                        f"{kind} capacity shrank to {self._capacity(kind)} B "
                        f"during the wait, below the pending request {want} B"
                    )
                if cancel_event is not None and cancel_event.is_set():
                    raise PipelineCancelled(
                        f"cancelled while waiting for {kind} credit"
                    )
                if not waited:
                    waited = True
                    started = float(self._clock())
                    self._wait_started_events += 1
                self._cond.wait(poll_s)
            if waited:
                self._wait_events += 1
                self._wait_s += max(0.0, float(self._clock()) - started)
            lease = self._next_lease
            self._next_lease += 1
            self._leases(kind)[lease] = want
            self._set_used(kind, self._used(kind) + want)
            return lease

    def release(self, kind: str, lease: int) -> int:
        """Release exactly one outstanding lease; double release fails closed."""
        leases = self._leases(kind)
        with self._cond:
            if lease not in leases:
                raise PipelineAccountingError(
                    f"unknown or already released {kind} lease {lease}"
                )
            amount = leases.pop(lease)
            used = self._used(kind) - amount
            if used < 0:
                raise PipelineAccountingError(
                    f"{kind} ledger underflow after releasing {amount} B"
                )
            self._set_used(kind, used)
            self._cond.notify_all()
            return int(amount)

    def snapshot(self) -> dict:
        with self._cond:
            return {
                "schema": PIPELINE_SCHEMA_V2,
                "host_budget_bytes": int(self.host_budget_bytes),
                "device_budget_bytes": int(self.device_budget_bytes),
                "disk_budget_bytes": int(self.disk_budget_bytes),
                "host_used_bytes": int(self._host_used),
                "host_peak_bytes": int(self._host_peak),
                "device_used_bytes": int(self._device_used),
                "device_peak_bytes": int(self._device_peak),
                "disk_used_bytes": int(self._disk_used),
                "disk_peak_bytes": int(self._disk_peak),
                "host_leases": len(self._host_leases),
                "device_leases": len(self._device_leases),
                "disk_leases": len(self._disk_leases),
                "credit_wait_events": int(self._wait_events),
                "credit_wait_started_events": int(self._wait_started_events),
                "credit_wait_s": float(self._wait_s),
                "reconciliations": int(self._reconciliations),
                "surplus_released_bytes": int(self._surplus_released_bytes),
                "budget_reductions": [dict(x) for x in self._budget_reductions],
                "balanced": (
                    not self._host_leases
                    and not self._device_leases
                    and not self._disk_leases
                ),
                "dimensions": {
                    "host": {
                        "budget_bytes": int(self.host_budget_bytes),
                        "role": "resident_process_reservation",
                    },
                    "device": {
                        "budget_bytes": int(self.device_budget_bytes),
                        "role": "gpu_workspace_reservation",
                    },
                    "disk": {
                        "budget_bytes": int(self.disk_budget_bytes),
                        "role": "on_disk_payload_bytes",
                    },
                },
            }


class BoundedPipeline:
    """Bounded, single-owner CPU -> GPU producer/consumer runner.

    Exactly one producer thread is started; the consumer (GPU owner) runs on
    the thread that calls :meth:`run`. The queue capacity is explicit
    (``prefetch_units + 1``): ``prefetch_units`` complete units may wait ahead
    of the one being consumed. Both byte budgets are hard caps enforced by
    :class:`BufferLedger`.

    Long healthy work is therefore the normal case: ``run`` waits for the next
    unit (checking cancellation every poll) instead of imposing a wall-clock
    budget on a unit. A finite join bound is used only once a terminal state is
    reached (sentinel, error or cancellation), where a surviving producer is a
    genuine stuck-thread failure. An optional ``stall_timeout_s`` (disabled by
    default) can additionally fail closed when the producer makes no progress
    at all; it is never a per-unit work budget.
    """

    def __init__(
        self,
        *,
        tasks,
        produce,
        consume,
        release,
        prefetch_units: int,
        host_budget_bytes: int,
        device_budget_bytes: int,
        disk_budget_bytes: int = 0,
        disk_enabled: bool | None = None,
        cleanup_task=None,
        host_bytes=lambda payload: 0,
        device_bytes=lambda payload: 0,
        disk_bytes=lambda payload: 0,
        # ZM-ZEGRID-R56 REWORK-1 (F1): PREDICTED per-TASK reservations made
        # BEFORE ``produce`` allocates or writes anything. When supplied the
        # producer reserves ``host_reserve_bytes(task)`` / ``disk_reserve_bytes(task)``
        # first, runs ``produce``, then reconciles down to the actual payload
        # (an underestimate fails closed). Generic callers that omit them keep
        # the legacy post-produce accounting.
        host_reserve_bytes=None,
        disk_reserve_bytes=None,
        # F4: fresh external revalidation hook, called BEFORE the pre-acquired
        # reservation for every task. It may only shrink a budget (the ledger
        # refuses growth) and may raise to refuse the unit before any staging.
        before_produce=None,
        clock=time.monotonic,
        thread_join_s: float = 30.0,
        stall_timeout_s=None,
    ) -> None:
        if not callable(produce) or not callable(consume) or not callable(release):
            raise PipelineError("produce/consume/release must be callables")
        if not callable(host_bytes) or not callable(device_bytes):
            raise PipelineError("host_bytes/device_bytes must be callables")
        if not callable(disk_bytes):
            raise PipelineError("disk_bytes must be callable")
        for name, fn in (
            ("host_reserve_bytes", host_reserve_bytes),
            ("disk_reserve_bytes", disk_reserve_bytes),
            ("before_produce", before_produce),
            ("cleanup_task", cleanup_task),
        ):
            if fn is not None and not callable(fn):
                raise PipelineError(f"{name} must be callable or None")
        if disk_enabled is not None and not isinstance(disk_enabled, bool):
            raise PipelineError("disk_enabled must be a bool or None")
        if (
            isinstance(thread_join_s, bool)
            or not isinstance(thread_join_s, (int, float))
            or not (0 < float(thread_join_s) <= 600.0)
        ):
            raise PipelineError("thread_join_s must be in (0, 600]")
        if stall_timeout_s is not None and (
            isinstance(stall_timeout_s, bool)
            or not isinstance(stall_timeout_s, (int, float))
            or not (0 < float(stall_timeout_s) <= 600.0)
        ):
            raise PipelineError("stall_timeout_s must be None or in (0, 600]")
        self._tasks = tuple(tasks)
        self._produce = produce
        self._consume = consume
        self._release_payload = release
        self._host_bytes_fn = host_bytes
        self._device_bytes_fn = device_bytes
        self._disk_bytes_fn = disk_bytes
        self._host_reserve_fn = host_reserve_bytes
        self._disk_reserve_fn = disk_reserve_bytes
        self._before_produce = before_produce
        self._cleanup_task = cleanup_task
        self._disk_enabled = (
            bool(disk_budget_bytes) > 0
            if disk_enabled is None
            else bool(disk_enabled)
        )
        self._pre_acquired_units = 0
        self._prefetch_units = _pipeline_int(
            prefetch_units, "prefetch_units", minimum=0
        )
        self._queue_max = int(self._prefetch_units) + 1
        self._clock = clock
        self._thread_join_s = float(thread_join_s)
        self._stall_timeout_s = (
            None if stall_timeout_s is None else float(stall_timeout_s)
        )
        self._ledger = BufferLedger(
            host_budget_bytes=host_budget_bytes,
            device_budget_bytes=device_budget_bytes,
            disk_budget_bytes=disk_budget_bytes,
            clock=clock,
        )
        self._queue: "queue.Queue" = queue.Queue(maxsize=self._queue_max)
        self._cancel = threading.Event()
        # ZM-ZEGRID-RESILIENT-RUN-RESUME M3: admission gate. When set, the
        # producer starts no new unit (before credit OR allocation). It is a
        # one-way latch for a recovery episode; cancellation remains separate.
        self._admission_stopped = threading.Event()
        self._admission_reason: str | None = None
        self._producer_error: BaseException | None = None
        self._consumer_error: BaseException | None = None
        self._cleanup_failures: list[str] = []
        self._cancel_reason: str | None = None
        self._results: list = [None] * len(self._tasks)
        self._produced = 0
        self._consumed = 0
        self._released = 0
        self._transfers = 0
        self._queue_depth_max = 0
        self._queue_depth_last = 0
        self._queue_depth_samples = 0
        self._queue_depth_sum = 0
        self._consumer_idle_s = 0.0
        # ZM-ZEGRID-R56: observed work intervals (monotonic seconds). Only the
        # produce() and consume() CALLS are timed, so an intersection is real
        # concurrent work, never queue capacity or two live threads.
        self._producer_intervals: list[tuple[float, float]] = []
        self._consumer_intervals: list[tuple[float, float]] = []
        self._producer_work_s = 0.0
        self._consumer_work_s = 0.0
        self._run_started_at: float | None = None
        self._run_ended_at: float | None = None
        self._producer_backpressure_events = 0
        self._producer_backpressure_s = 0.0
        self._producer_thread: threading.Thread | None = None
        self._owner_thread_name: str | None = None
        self._owner_thread_ident: int | None = None
        self._active_unit_index: int | None = None
        self._producer_task_index: int | None = None
        self._producer_task_started_at: float | None = None
        self._last_progress_at: float | None = None
        self._stalled = False
        self._interrupt: str | None = None
        self._finished = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def queue_max(self) -> int:
        return int(self._queue_max)

    @property
    def prefetch_units(self) -> int:
        return int(self._prefetch_units)

    def cancel(self, reason: str = "cancelled") -> None:
        """One-way cancellation latch (idempotent)."""
        if not self._cancel.is_set():
            self._cancel_reason = str(reason)
            self._cancel.set()
        with self._ledger._cond:  # noqa: SLF001 - same-module primitive
            self._ledger._cond.notify_all()  # noqa: SLF001

    def stop_admission(self, reason: str = "resource_pressure") -> None:
        """Stop admitting NEW units; idempotent and independent of cancel().

        M3 uses this at the first safe point of a pressure episode. The
        producer re-checks it before acquiring credit and before allocating, so
        no speculative unit starts after the decision. Work already produced is
        released by the ordinary cancellation/drain path, never leaked.
        """
        if not self._admission_stopped.is_set():
            self._admission_reason = str(reason)
            self._admission_stopped.set()

    def admission_stopped(self) -> bool:
        return bool(self._admission_stopped.is_set())

    def cancelled(self) -> bool:
        """True once the one-way cancellation latch is set."""
        return bool(self._cancel.is_set())

    def ledger_snapshot(self) -> dict:
        """Thread-safe snapshot of the credit ledger (M3 release proof)."""
        return self._ledger.snapshot()

    def _cancelled(self) -> bool:
        return bool(self._cancel.is_set())

    def _admission_closed(self) -> bool:
        return bool(self._cancelled() or self._admission_stopped.is_set())

    def _sample_queue(self) -> int:
        depth = int(self._queue.qsize())
        self._queue_depth_last = depth
        self._queue_depth_max = max(self._queue_depth_max, depth)
        self._queue_depth_samples += 1
        self._queue_depth_sum += depth
        return depth

    # -- unit cleanup ------------------------------------------------------

    def _release_unit(self, unit) -> None:
        """Release ledger leases and the caller payload exactly once."""
        if not isinstance(unit, dict) or unit.get("released"):
            return
        try:
            self._release_payload(unit["payload"])
        except BaseException as exc:  # noqa: BLE001 - best effort, never masks cause
            self._cleanup_failures.append(
                f"unit[{unit.get('index')}]:{type(exc).__name__}:{exc}"
            )
        try:
            if unit.get("device_lease") is not None:
                self._ledger.release("device", int(unit["device_lease"]))
            if unit.get("host_lease") is not None:
                self._ledger.release("host", int(unit["host_lease"]))
            if unit.get("disk_lease") is not None:
                self._ledger.release("disk", int(unit["disk_lease"]))
        finally:
            unit["released"] = True
            self._released += 1

    # -- producer ----------------------------------------------------------

    def _put(self, unit) -> None:
        """Bounded put that honours cancellation (no unbounded blocking)."""
        while True:
            if self._admission_closed():
                self._release_unit(unit)
                raise PipelineCancelled(
                    "producer admission closed before enqueue"
                )
            started = float(self._clock())
            try:
                self._queue.put(unit, timeout=0.05)
            except queue.Full:
                self._producer_backpressure_events += 1
                self._producer_backpressure_s += max(
                    0.0, float(self._clock()) - started
                )
                continue
            # A bounded put that had to wait for QUEUE SPACE is backpressure
            # even when it completed before the 50 ms poll timeout.
            waited = max(0.0, float(self._clock()) - started)
            if waited >= 0.0005:
                self._producer_backpressure_events += 1
                self._producer_backpressure_s += waited
            self._sample_queue()
            return

    def _put_sentinel(self) -> None:
        while not self._cancelled():
            try:
                self._queue.put({"sentinel": True}, timeout=0.05)
                return
            except queue.Full:
                continue

    def _release_pre(self, pre) -> None:
        """Release pre-acquired leases exactly once (partial-produce path)."""
        if not isinstance(pre, dict):
            return
        for kind in ("host", "disk"):
            lease = pre.get(f"{kind}_lease")
            if lease is None:
                continue
            try:
                self._ledger.release(kind, int(lease))
            except BaseException as exc:  # noqa: BLE001 - best effort
                self._cleanup_failures.append(
                    f"pre_release[{kind}]:{type(exc).__name__}:{exc}"
                )
            pre[f"{kind}_lease"] = None

    def _pre_acquire(self, task):
        """Reserve predicted host/disk credit BEFORE any produce allocation.

        ZM-ZEGRID-R56 REWORK-1 (F1). The prediction comes from the PLAN/task
        bounds (caller callbacks), never from a completed payload, so the ledger
        observes the peak at ALLOCATION time. Returns a dict of leases (or None
        when the callers did not opt in to pre-acquisition).
        """
        if self._host_reserve_fn is None and self._disk_reserve_fn is None:
            return None
        pre = {
            "host_lease": None,
            "disk_lease": None,
            "host_reserved": None,
            "disk_reserved": None,
        }
        try:
            # REWORK-2 ordering: host credit first (its wait may be long), then a
            # FRESH disk observation, then disk credit. Never a stale pre-wait
            # observation admitted as fresh.
            if self._host_reserve_fn is not None:
                host_pred = _pipeline_int(
                    self._host_reserve_fn(task), "host_reserve_bytes", minimum=0
                )
                pre["host_reserved"] = int(host_pred)
                pre["host_lease"] = self._ledger.reserve(
                    "host", host_pred, cancel_event=self._cancel
                )
            if self._before_produce is not None:
                self._before_produce(task, self._ledger, self._cancel)
            if self._disk_reserve_fn is not None and self._disk_enabled:
                disk_pred = _pipeline_int(
                    self._disk_reserve_fn(task), "disk_reserve_bytes", minimum=0
                )
                pre["disk_reserved"] = int(disk_pred)
                pre["disk_lease"] = self._ledger.reserve(
                    "disk", disk_pred, cancel_event=self._cancel
                )
        except BaseException:
            self._release_pre(pre)
            raise
        self._pre_acquired_units += 1
        return pre

    def _producer_run(self) -> None:
        try:
            for index, task in enumerate(self._tasks):
                if self._admission_closed():
                    return
                self._producer_task_index = int(index)
                self._producer_task_started_at = float(self._clock())
                # F1: credit FIRST, allocation second. A refusal/cancel here
                # happens before any cache write or buffer materialisation.
                pre = self._pre_acquire(task)
                if self._admission_closed():
                    self._release_pre(pre)
                    return
                _produce_started = float(self._clock())
                try:
                    payload = self._produce(task)
                except BaseException:
                    # Cleanup BEFORE credits are released: a partial cache write
                    # must not leave files behind while the ledger frees space.
                    if self._cleanup_task is not None:
                        try:
                            self._cleanup_task(task, None)
                        except BaseException as cleanup_exc:  # noqa: BLE001
                            self._cleanup_failures.append(
                                "cleanup_task:"
                                f"{type(cleanup_exc).__name__}:{cleanup_exc}"
                            )
                    self._release_pre(pre)
                    raise
                finally:
                    _produce_ended = float(self._clock())
                    self._producer_intervals.append(
                        (_produce_started, _produce_ended)
                    )
                    self._producer_work_s += max(
                        0.0, _produce_ended - _produce_started
                    )
                self._producer_task_index = None
                self._producer_task_started_at = None
                self._last_progress_at = float(self._clock())
                # OWNERSHIP BEGINS HERE. Once ``produce`` returns, the payload
                # belongs to the pipeline: every path that does NOT hand it to
                # the consumer MUST release it exactly once. ``_release_unit``
                # is idempotent, so no path can double-release.
                unit = {
                    "index": int(index),
                    "payload": payload,
                    "host_bytes": None,
                    "device_bytes": None,
                    "disk_bytes": None,
                    "host_lease": None if pre is None else pre["host_lease"],
                    "device_lease": None,
                    "disk_lease": None if pre is None else pre["disk_lease"],
                    "host_reserved": None if pre is None else pre["host_reserved"],
                    "disk_reserved": None if pre is None else pre["disk_reserved"],
                    "reconciled": False,
                    "released": False,
                }
                if pre is not None:
                    # The unit now owns the leases; a failure below must not
                    # release them twice through the pre path.
                    pre["host_lease"] = None
                    pre["disk_lease"] = None
                # ``units_produced`` counts OWNED payloads (produce returned),
                # not successful enqueues: every owned payload is released, so
                # produced == released on any terminal path that finishes.
                self._produced += 1
                if self._admission_closed():
                    self._release_unit(unit)
                    return
                try:
                    host_need = _pipeline_int(
                        self._host_bytes_fn(payload), "host_bytes", minimum=0
                    )
                    device_need = _pipeline_int(
                        self._device_bytes_fn(payload), "device_bytes", minimum=0
                    )
                    disk_need = _pipeline_int(
                        self._disk_bytes_fn(payload), "disk_bytes", minimum=0
                    )
                    unit["host_bytes"] = int(host_need)
                    unit["device_bytes"] = int(device_need)
                    unit["disk_bytes"] = int(disk_need)
                    if unit["host_lease"] is not None:
                        self._ledger.reconcile("host", int(unit["host_lease"]), host_need)
                        unit["reconciled"] = True
                    else:
                        unit["host_lease"] = self._ledger.reserve(
                            "host", host_need, cancel_event=self._cancel
                        )
                    if self._disk_enabled:
                        if unit["disk_lease"] is not None:
                            self._ledger.reconcile(
                                "disk", int(unit["disk_lease"]), disk_need
                            )
                            unit["reconciled"] = True
                        else:
                            unit["disk_lease"] = self._ledger.reserve(
                                "disk", disk_need, cancel_event=self._cancel
                            )
                    self._put(unit)
                except BaseException:
                    # Covers reconciliation refusal (prediction too small),
                    # host-credit cancellation, accounting refusal and
                    # byte-callback errors: release and re-raise so the PRIMARY
                    # classification is preserved.
                    self._release_unit(unit)
                    raise
            self._put_sentinel()
        except PipelineCancelled:
            return
        except BaseException as exc:  # noqa: BLE001 - re-raised by run()
            self._producer_error = exc
            self.cancel(f"producer_error:{type(exc).__name__}")
            self._put_sentinel()
        finally:
            self._producer_task_index = None
            self._producer_task_started_at = None
            self._last_progress_at = float(self._clock())

    # -- consumer (caller thread owns the GPU) ------------------------------

    def _age_s(self, started) -> float | None:
        if started is None:
            return None
        return max(0.0, float(self._clock()) - float(started))

    def _producer_stalled(self) -> bool:
        """Optional no-progress watchdog (disabled by default).

        This is deliberately NOT a per-unit work budget: it only fires when the
        producer has made no observable progress at all for the configured
        window. A healthy long cell keeps updating ``_last_progress_at``.
        """
        if self._stall_timeout_s is None:
            return False
        if self._last_progress_at is None:
            return False
        if (
            float(self._clock()) - float(self._last_progress_at)
            <= float(self._stall_timeout_s)
        ):
            return False
        self._stalled = True
        self.cancel("stall_timeout")
        return True

    def _consume_loop(self) -> None:
        """Consume units on the CALLING thread until a terminal state.

        A healthy GPU unit may take arbitrarily long; waiting for the next unit
        is bounded only by cancellation/producer liveness, never by a wall-clock
        budget on the unit itself.
        """
        while True:
            if self._cancelled():
                return
            started = float(self._clock())
            try:
                unit = self._queue.get(timeout=0.05)
            except queue.Empty:
                # Count THIS timed-out poll before any terminal return. Every
                # queue.get() call's elapsed host wait is therefore accumulated
                # exactly once, including the empty-poll / producer-death /
                # stall / cancellation paths; the earlier version lost all
                # queue.Empty timeout durations by ``continue``-ing first.
                self._consumer_idle_s += max(
                    0.0, float(self._clock()) - started
                )
                if self._producer_stalled():
                    return
                if (
                    self._producer_thread is not None
                    and not self._producer_thread.is_alive()
                ):
                    # The producer is gone and no sentinel/unit is left: this is
                    # a genuine terminal anomaly, never a slow healthy unit.
                    if self._producer_error is None and not self._cancelled():
                        self._producer_error = PipelineError(
                            "producer thread exited without a terminal sentinel"
                        )
                    if not self._cancelled():
                        self.cancel("producer_exited")
                    return
                continue
            self._consumer_idle_s += max(0.0, float(self._clock()) - started)
            self._sample_queue()
            if isinstance(unit, dict) and unit.get("sentinel"):
                return
            self._active_unit_index = int(unit["index"])
            self._last_progress_at = float(self._clock())
            try:
                unit["device_lease"] = self._ledger.reserve(
                    "device",
                    int(unit["device_bytes"]),
                    cancel_event=self._cancel,
                )
                _consume_started = float(self._clock())
                try:
                    result = self._consume(unit["payload"])
                finally:
                    _consume_ended = float(self._clock())
                    self._consumer_intervals.append(
                        (_consume_started, _consume_ended)
                    )
                    self._consumer_work_s += max(
                        0.0, _consume_ended - _consume_started
                    )
                self._transfers += 1
            except PipelineCancelled:
                return
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:  # noqa: BLE001 - re-raised by run()
                self._consumer_error = exc
                self.cancel(f"consumer_error:{type(exc).__name__}")
                return
            else:
                self._results[int(unit["index"])] = result
                self._consumed += 1
            finally:
                self._active_unit_index = None
                self._last_progress_at = float(self._clock())
                self._release_unit(unit)

    # -- run ---------------------------------------------------------------

    def _drain(self) -> dict:
        units = 0
        while True:
            try:
                unit = self._queue.get_nowait()
            except queue.Empty:
                return {"units": int(units)}
            if isinstance(unit, dict) and unit.get("sentinel"):
                continue
            try:
                self._release_unit(unit)
                units += 1
            except BaseException as exc:  # noqa: BLE001
                self._cleanup_failures.append(
                    f"drain_release:{type(exc).__name__}:{exc}"
                )

    def drain_speculative(self) -> dict:
        """Drain queued units and report the credits ACTUALLY released.

        Safe to call from the recovery thread while the pipeline is still
        ending: it only touches the queue and the ledger, both thread-safe. A
        unit currently held by the consumer is NOT drained here; the caller
        must wait for the consumer to relinquish it (the ledger snapshot says
        whether the pipeline is balanced).
        """
        before = self._ledger.snapshot()
        drained = self._drain()
        after = self._ledger.snapshot()
        return {
            "drained_units": int(drained["units"]),
            "cancelled_units": 0,
            "host_released_bytes": max(
                0,
                int(before["host_used_bytes"]) - int(after["host_used_bytes"]),
            ),
            "device_released_bytes": max(
                0,
                int(before["device_used_bytes"])
                - int(after["device_used_bytes"]),
            ),
            "disk_released_bytes": max(
                0,
                int(before["disk_used_bytes"]) - int(after["disk_used_bytes"]),
            ),
            "credits_balanced": bool(after["balanced"]),
        }

    def _join_producer(self) -> bool:
        """Finite-bound join used ONLY after a terminal state."""
        thread = self._producer_thread
        if thread is None:
            return False
        thread.join(self._thread_join_s)
        if thread.is_alive():
            self.cancel("producer_join_timeout")
            thread.join(max(1.0, self._thread_join_s / 6))
        alive = bool(thread.is_alive())
        if alive:
            self._cleanup_failures.append(
                "producer_thread_alive_after_join:"
                f"{self._thread_join_s:.3f}s"
            )
        return alive

    def run(self):
        """Run the pipeline; return canonical-order results or raise.

        The consumer runs on the calling thread, so a long healthy GPU unit
        cannot be mistaken for a deadlock. On any failure the pipeline is
        cancelled, every staged unit is released, the producer is joined with a
        finite bound, and the PRIMARY error is re-raised (type preserved so a
        CuPy OOM stays classified). Partial GPU results are never returned.
        """
        if self._finished:
            raise PipelineError("a BoundedPipeline instance runs at most once")
        self._finished = True
        self._owner_thread_name = threading.current_thread().name
        self._owner_thread_ident = threading.get_ident()
        self._last_progress_at = float(self._clock())
        self._run_started_at = float(self._clock())
        self._producer_thread = threading.Thread(
            target=self._producer_run,
            name="r55e2-pipeline-producer",
            daemon=True,
        )
        self._producer_thread.start()

        interrupt: BaseException | None = None
        try:
            self._consume_loop()
        except BaseException as exc:  # noqa: BLE001 - caller interrupt wins
            interrupt = exc
            self._interrupt = f"{type(exc).__name__}: {exc}"
            self.cancel(f"caller_interrupt:{type(exc).__name__}")

        try:
            producer_alive = self._join_producer()
        except BaseException as exc:  # noqa: BLE001 - interrupt during join
            if interrupt is None:
                interrupt = exc
                self._interrupt = f"{type(exc).__name__}: {exc}"
                self.cancel(f"caller_interrupt:{type(exc).__name__}")
            producer_alive = bool(
                self._producer_thread is not None
                and self._producer_thread.is_alive()
            )
        self._drain()
        self._run_ended_at = float(self._clock())
        ledger = self._ledger.snapshot()

        if interrupt is not None:
            raise interrupt
        if self._consumer_error is not None:
            # Primary consumer classification wins (e.g. a CuPy OOM must stay a
            # CuPy OOM); a surviving producer is recorded as cleanup telemetry.
            raise self._consumer_error
        if self._producer_error is not None:
            if isinstance(self._producer_error, PipelineAccountingError):
                raise self._producer_error
            raise PipelineError(
                "producer failed: "
                f"{type(self._producer_error).__name__}: {self._producer_error}"
            ) from self._producer_error
        if self._stalled:
            raise PipelineError(
                "bounded pipeline made no producer progress for "
                f"{self._stall_timeout_s}s (stall_timeout)"
            )
        if producer_alive:
            raise PipelineError(
                "bounded pipeline producer thread did not terminate "
                f"(producer_alive=True, owner_thread={self._owner_thread_name})"
            )
        if self._cancelled():
            raise PipelineCancelled(
                f"pipeline cancelled ({self._cancel_reason or 'unknown'})"
            )
        if not ledger["balanced"]:
            raise PipelineAccountingError(
                "bounded pipeline buffer ledger is not balanced: "
                f"{ledger!r}"
            )
        if self._consumed != len(self._tasks):
            raise PipelineError(
                f"pipeline consumed {self._consumed}/{len(self._tasks)} units"
            )
        return list(self._results)

    # -- telemetry ---------------------------------------------------------

    def telemetry(self) -> dict:
        """Truthful observational telemetry (never part of a plan hash).

        ``units_produced`` counts payloads the pipeline TOOK OWNERSHIP of (i.e.
        ``produce`` returned): a produced payload that is cancelled before
        enqueue or refused by the ledger is included. ``units_released`` counts
        payloads whose release callback and any leases were reconciled exactly
        once. On every terminal path that finishes cleanup the two are equal;
        neither count claims a GPU transfer (``transfers`` does that).
        """
        ledger = self._ledger.snapshot()
        overlap_s = _intervals_union_intersection_s(
            self._producer_intervals, self._consumer_intervals
        )
        overlap_units = sum(
            1
            for interval in self._consumer_intervals
            if _interval_hits_any(interval, self._producer_intervals)
        )
        wall_s = (
            None
            if self._run_started_at is None or self._run_ended_at is None
            else max(0.0, float(self._run_ended_at) - float(self._run_started_at))
        )
        return {
            "schema": PIPELINE_SCHEMA,
            "schema_v2": PIPELINE_SCHEMA_V2,
            "units_total": int(len(self._tasks)),
            "units_produced": int(self._produced),
            "units_consumed": int(self._consumed),
            "units_released": int(self._released),
            "lifecycle": "caller_owned_consumer",
            "consumer_owner": "caller_thread",
            "owner_thread_name": self._owner_thread_name,
            "owner_thread_ident": self._owner_thread_ident,
            "active_unit_index": self._active_unit_index,
            "producer_thread_name": (
                None
                if self._producer_thread is None
                else str(self._producer_thread.name)
            ),
            "producer_alive": (
                self._producer_thread is not None
                and self._producer_thread.is_alive()
            ),
            "producer_task_index": self._producer_task_index,
            "producer_task_age_s": self._age_s(self._producer_task_started_at),
            "last_progress_age_s": self._age_s(self._last_progress_at),
            "stall_timeout_s": self._stall_timeout_s,
            "stalled": bool(self._stalled),
            "interrupt": self._interrupt,
            "queue_max": int(self._queue_max),
            "prefetch_units": int(self._prefetch_units),
            "queue_depth_max": int(self._queue_depth_max),
            "queue_depth_last": int(self._queue_depth_last),
            "queue_depth_final": int(self._queue.qsize()),
            "queue_depth_samples": int(self._queue_depth_samples),
            "queue_depth_mean": (
                None
                if not self._queue_depth_samples
                else float(self._queue_depth_sum)
                / float(self._queue_depth_samples)
            ),
            "ledger": ledger,
            "transfers": int(self._transfers),
            "consumer_idle_s": float(self._consumer_idle_s),
            # R55-E2 telemetry semantics: this is HOST queue.get() waiting on the
            # consumer (owner) thread, including every queue.Empty timeout poll.
            # It is not a device/GPU idle measurement; asynchronous GPU work is
            # not observed by this counter.
            "consumer_idle_semantics": (
                "host_consumer_queue_wait_s (all queue.get calls, including "
                "queue.Empty timeout polls); not device/GPU idle time"
            ),
            "credit_accounting": {
                "mode": (
                    "pre_acquire"
                    if self._host_reserve_fn is not None
                    or self._disk_reserve_fn is not None
                    else "post_produce_legacy"
                ),
                "pre_acquired_units": int(self._pre_acquired_units),
                "reconciliations": int(ledger["reconciliations"]),
                "surplus_released_bytes": int(ledger["surplus_released_bytes"]),
                "note": (
                    "pre_acquire reserves the PLAN prediction before produce() "
                    "allocates; the lease is then reconciled DOWN to the actual "
                    "payload and an underestimate fails closed."
                ),
            },
            "observed_overlap": {
                "clock": "time.monotonic",
                "producer_work_s": float(self._producer_work_s),
                "consumer_work_s": float(self._consumer_work_s),
                "wall_s": wall_s,
                "intersection_s": float(overlap_s),
                "overlap_units": int(overlap_units),
                "consumer_units_timed": int(len(self._consumer_intervals)),
                "producer_units_timed": int(len(self._producer_intervals)),
                "overlap_proved": bool(overlap_s > 0.0 and overlap_units > 0),
                "note": (
                    "Intersection of the observed produce()/consume() CALL "
                    "intervals only. Nonzero means both sides performed work "
                    "concurrently; it is not a queue-capacity or thread-count "
                    "claim."
                ),
            },
            "producer_backpressure_events": int(
                self._producer_backpressure_events
            ),
            "producer_backpressure_s": float(
                self._producer_backpressure_s
            ),
            "cancelled": bool(self._cancelled()),
            "cancel_reason": self._cancel_reason,
            "producer_error": (
                None
                if self._producer_error is None
                else f"{type(self._producer_error).__name__}: {self._producer_error}"
            ),
            "consumer_error": (
                None
                if self._consumer_error is None
                else f"{type(self._consumer_error).__name__}: {self._consumer_error}"
            ),
            "cleanup_failures": list(self._cleanup_failures),
            "threads_joined": (
                self._producer_thread is not None
                and not self._producer_thread.is_alive()
            ),
        }
