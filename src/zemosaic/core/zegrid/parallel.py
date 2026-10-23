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
import threading
import time

# ZM-ZEGRID-R41: pure host-RAM contract + phase-accounting math. Imported as a
# submodule (no psutil, no product mode) so the opt-in planner branch below can
# account for SIMULTANEOUS phase totals without duplicating the arithmetic.
from . import ram_contract as rcontract

log = logging.getLogger(__name__)

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

    cpu_budget = max(1, cpu - _CPU_OS_RESERVE)
    max_k = max(1, min(cpu_budget, n_cells))
    if max_cells_in_flight is not None:
        # ZM-ZEGRID-R41-B: a bounded replan may NEVER grow the concurrency, even
        # when a later observation finds MORE RAM than the plan budget assumed.
        _cap = rcontract.require_nonneg_int(max_cells_in_flight, "max_cells_in_flight")
        max_k = max(1, min(max_k, max(1, int(_cap))))

    def _makespan(k: int, cost: float) -> float:
        waves = (n_cells + k - 1) // k
        return waves * cost

    def _search(bound: int) -> dict:
        # ZM-ZEGRID-R41-B (rework-2): REUSE the established per-mode concurrency
        # seam as a conservative INITIAL cap (so an existing
        # ``zpar.cells_in_flight`` monkeypatch keeps governing K), then let the
        # simultaneous phase accounting REDUCE K further. K can only go DOWN.
        _seam_budget = ram_budget if (ram_budget or 0) > 0 else None
        _seam_cap = cells_in_flight(cpu, _seam_budget, int(bound))
        mode_max_k = max(1, min(int(max_k), int(_seam_cap), n_cells))
        k, st, ct, agg, cpc, w = rcontract.feasible_concurrency(
            budget_bytes=ram_budget,
            shared_bytes=shared,
            stack_per_cell_bound_bytes=int(bound),
            process_baseline_bytes=pb,
            cache_worker_footprint_bytes=cwf,
            cache_workers_for_k=_cache_workers_for_k,
            max_k=mode_max_k,
        )
        return {"k": int(k), "stack_total": st, "cache_total": ct, "aggregate": agg,
                "cache_per_cell": cpc, "cache_workers": w, "bound": int(bound)}

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
            **_phase_keys(k=1, w=w, cpc=cpc, st=st, ct=ct, agg=agg),
        }

    inmem = _search(max_inmem)
    stream = _search(max_stream)
    inmem_feasible = inmem["k"] >= 1
    stream_feasible = stream["k"] >= 1

    if not inmem_feasible and not stream_feasible:
        raise ModeInfeasible(
            "no stacking mode fits the host phase aggregate budget "
            f"{_budget_txt()}: in-memory one-cell aggregate "
            f">= {(shared + max_inmem) / 2**20:.1f} MiB and streaming one-cell "
            f"aggregate >= {(shared + max_stream) / 2**20:.1f} MiB both exceed it "
            f"(n_cells={n_cells})"
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


def _heartbeat_interval(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"heartbeat interval must be a number, got {value!r}")
    interval = float(value)
    if not (0.0 < interval <= 5.0):
        raise ValueError(
            f"heartbeat interval must be in (0, 5.0], got {value!r}"
        )
    return interval


def pmap(worker, tasks, workers: int, progress_callback=None, initializer=None, initargs=(), emit=None, meta=None, serial_fallback: bool = True, heartbeat=None, heartbeat_interval_s: float = 5.0):
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
    interval_s = (
        _heartbeat_interval(heartbeat_interval_s)
        if heartbeat is not None else None
    )

    if meta is not None:
        meta["heartbeat_calls"] = 0
        meta["heartbeat_errors"] = 0
        meta["heartbeat_mode"] = "none" if heartbeat is None else "pending"
        meta["heartbeat_interval_s"] = interval_s

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
                out.append(worker(t))
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
        """Ordered results with a parent-side heartbeat wait loop (no thread).

        The heartbeat is called by the PARENT while futures run. No sampler
        thread exists when ProcessPoolExecutor forks/spawns its workers.
        """
        out = [None] * total
        recorded = set()
        reported = 0
        last_call = None

        def _call_heartbeat(now):
            nonlocal last_call
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

        # Sample once BEFORE submitting (no worker is forked yet, no E1
        # thread exists), then keep a parent-side cadence while futures run.
        now = time.monotonic()
        _call_heartbeat(now)
        futures = [pool.submit(worker, task) for task in tasks]
        pending = set(futures)
        while pending:
            now = time.monotonic()
            timeout = max(0.0, interval_s - (now - last_call))
            done, pending = concurrent.futures.wait(
                pending, timeout=timeout
            )
            for index, future in enumerate(futures):
                if index in recorded or future not in done:
                    continue
                out[index] = future.result()
                recorded.add(index)
            while reported < total and reported in recorded:
                reported += 1
                _report(reported)
            now = time.monotonic()
            if now - last_call >= interval_s:
                _call_heartbeat(now)
        for index, future in enumerate(futures):
            if index not in recorded:
                out[index] = future.result()
                recorded.add(index)
        while reported < total and reported in recorded:
            reported += 1
            _report(reported)
        if meta is not None:
            meta["heartbeat_mode"] = "parent_wait"
        return out

    try:
        if kind == "thread":
            # ThreadPoolExecutor has no initializer; the parent already called it.
            with Executor(max_workers=workers) as pool:
                if heartbeat is None:
                    out = []
                    for i, res in enumerate(pool.map(worker, tasks)):
                        out.append(res)
                        _report(i + 1)
                else:
                    out = _run_pool(pool)
        else:
            with Executor(
                max_workers=workers,
                initializer=initializer,
                initargs=tuple(initargs),
            ) as pool:
                if heartbeat is None:
                    out = []
                    for i, res in enumerate(pool.map(worker, tasks)):
                        out.append(res)
                        _report(i + 1)
                else:
                    out = _run_pool(pool)
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
        out = _run_serial()
        _finish(meta, t0, total)
        return out
