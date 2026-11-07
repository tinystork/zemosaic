"""ZM-ZEGRID-AUTO-EXTERNAL-STREAMING M0 — honest, side-effect-free dry-plan.

This module is the smallest product-quality planning surface that can answer,
BEFORE any gauge/stack/cache work:

* which layout/geometry is scientifically admissible (the caller supplies the
  floor verdict; this module never derives a science floor itself);
* whether a complete in-memory plan fits the PLAN budget;
* for the CPU streaming ladder, EVERY rung's worst-cell bound, the selected
  (largest feasible) rung, and the exact reason a coarser rung did not fit;
* the GPU ladder SEPARATELY (it is never mixed into the CPU selection, and it
  is never enumerated below its own >= 32 floor);
* what is known about the disk/scratch envelope and what is explicitly NOT yet
  priced (the future external spool belongs to M2 and is never claimed to
  exist);
* whether the next required class is an external working set (future M2).

Design constraints (bounded M0):

* PURE: no filesystem access, no gauge, no cache build, no stack, no global
  mutation. Every observed quantity (RAM, disk free, cells, floors) is an
  explicit input.
* ONE model: candidates are enumerated by
  :func:`zemosaic.core.zegrid.scheduler.enumerate_candidates` and every bound
  comes from :func:`zemosaic.core.zegrid.streaming.estimate_streaming_peak_kib`
  (the SAME estimator the executor/admission path uses). There is no duplicate
  estimator here that could drift from execution.
* NO external-spool claim: ``external_working_set_required`` is a refusal
  class, not an implemented capability. The report always carries
  ``external.implemented = False``.

The report is strict JSON (string keys, ints, bools, lists, dicts, strings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from . import scheduler as zsched
from . import streaming as zstream
from .ram_contract import PROCESS_BASELINE_BYTES
from .resource_envelope import (
    BACKEND_CPU,
    DEFAULT_MARGIN_DEN,
    DEFAULT_MARGIN_NUM,
    PHASE_CELL_STACK_INMEM,
    PHASE_CELL_STACK_STREAM,
    ceil_margin_div,
)
from .storage_profile import (
    STORAGE_PROFILE_DENSE,
    normalize_storage_profile,
)

__all__ = [
    "DRY_PLAN_SCHEMA",
    "REFUSAL_SCIENTIFIC_FLOOR",
    "REFUSAL_NO_IN_MEMORY_PLAN",
    "REFUSAL_NO_BOUNDED_RAM_TILE",
    "REFUSAL_DISK_ENVELOPE_UNAVAILABLE",
    "REFUSAL_EXTERNAL_WORKING_SET_REQUIRED",
    "REFUSAL_INVALID_OBSERVATION",
    "REFUSAL_CLASSES",
    "DISK_PRICED",
    "DISK_PARTIAL",
    "DISK_UNPRICED",
    "DryPlanSpec",
    "build_dry_plan",
]

DRY_PLAN_SCHEMA = "zemosaic-zegrid-dry-plan-v1"

REFUSAL_SCIENTIFIC_FLOOR = "scientific_floor"
REFUSAL_NO_IN_MEMORY_PLAN = "no_in_memory_plan"
REFUSAL_NO_BOUNDED_RAM_TILE = "no_bounded_ram_tile"
REFUSAL_DISK_ENVELOPE_UNAVAILABLE = "disk_envelope_unavailable"
REFUSAL_EXTERNAL_WORKING_SET_REQUIRED = "external_working_set_required"
REFUSAL_INVALID_OBSERVATION = "invalid_observation"
REFUSAL_CLASSES = (
    REFUSAL_SCIENTIFIC_FLOOR,
    REFUSAL_NO_IN_MEMORY_PLAN,
    REFUSAL_NO_BOUNDED_RAM_TILE,
    REFUSAL_DISK_ENVELOPE_UNAVAILABLE,
    REFUSAL_EXTERNAL_WORKING_SET_REQUIRED,
    REFUSAL_INVALID_OBSERVATION,
)

DISK_PRICED = "priced"
DISK_PARTIAL = "partial"
DISK_UNPRICED = "unpriced"

# Terms the future external working set (M2) will have to price. They are
# NAMED here so a report can never let them be zero-by-omission.
_EXTERNAL_UNPRICED_TERMS = (
    "external_spool_peak_bytes",
    "external_column_batch_bytes",
    "external_spool_write_bytes_per_cell",
    "external_spool_cleanup_margin_bytes",
)


def _require_int(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}={value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _require_hw(value, name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a (height, width) pair, got {value!r}")
    return (_require_int(value[0], f"{name}[0]", minimum=1),
            _require_int(value[1], f"{name}[1]", minimum=1))


@dataclass(frozen=True)
class DryPlanSpec:
    """Every input the dry-plan needs. No hidden probes, no hidden defaults.

    ``cells`` is the list of NON-EMPTY cells as ``(n_contributors, patch_hw)``
    pairs. ``layout``/``refinement_rank`` identify the candidate geometry.
    ``in_memory_bound_by_cell`` is the caller's already-computed in-memory
    bound (the same ``FITTED_MEMORY_MODEL`` value the product planner uses);
    when absent, no in-memory plan is claimed.

    ``per_unit_cache_bytes`` is the EXACT aligned-cache byte contract of ONE
    worst cell (the product's single cache-bytes contract). When absent the
    disk envelope is reported as UNPRICED — never as zero.
    """

    raw_available_bytes: int
    plan_budget_bytes: int | None
    cells: Sequence[tuple[int, tuple[int, int]]]
    layout: tuple[int, int]
    refinement_rank: int
    halo_px: int
    channels: int = 3
    storage_profile: str = STORAGE_PROFILE_DENSE
    cpu_logical: int = 1
    disk_free_bytes: int | None = None
    per_unit_cache_bytes: int | None = None
    in_memory_bound_by_cell: Sequence[int] | None = None
    known_disk_terms: Mapping[str, int] = field(default_factory=dict)
    geometry_admissible: bool = True
    scientific_floor_reason: str | None = None
    floors: Mapping[str, object] = field(default_factory=dict)
    gpu_ladder: Sequence[int] | None = None
    gpu_device_peak_by_tile: Mapping[int, int] | None = None


def _json_terms(terms: Mapping[str, int]) -> dict:
    out = {}
    for key in sorted(terms):
        out[str(key)] = _require_int(terms[key], f"known_disk_terms[{key!r}]")
    return out


def _empty_report(spec: DryPlanSpec | None, refusal_class: str, detail: str) -> dict:
    raw = None if spec is None else getattr(spec, "raw_available_bytes", None)
    budget = None if spec is None else getattr(spec, "plan_budget_bytes", None)
    return {
        "schema": DRY_PLAN_SCHEMA,
        "mode": "dry_plan",
        "side_effects": "none",
        "ok": False,
        "refusal_class": refusal_class,
        "refusal_detail": str(detail),
        "refusal_classes_considered": list(REFUSAL_CLASSES),
        "budget": {
            "raw_available_bytes": (None if raw is None else int(raw)),
            "plan_budget_bytes": (None if budget is None else int(budget)),
            "factor_num": 4,
            "factor_den": 5,
            "margin_num": int(DEFAULT_MARGIN_NUM),
            "margin_den": int(DEFAULT_MARGIN_DEN),
        },
        "geometry": {
            "layout": (None if spec is None else [int(v) for v in spec.layout]),
            "refinement_rank": (None if spec is None else int(spec.refinement_rank)),
            "halo_px": (None if spec is None else int(spec.halo_px)),
            "floors": ({} if spec is None else dict(spec.floors)),
        },
        "cpu": {
            "ladder": [int(t) for t in zsched.PRODUCT_TILE_LADDER],
            "preferred_tile_size": int(zsched.PRODUCT_TILE_LADDER[0]),
            "minimum_tile_size": int(zsched.MIN_CPU_TILE_PX),
            "trials": [],
            "selected_tile": None,
            "selected_working_set": None,
            "predicted_bound_bytes": None,
        },
        "in_memory": {"considered": False, "feasible": None,
                      "worst_bound_bytes": None},
        "gpu": {"ladder": None, "trials": [], "selected_tile": None,
                "minimum_tile_size": int(zsched.MIN_TILE_PX),
                "reason": "not_enumerated"},
        "disk": {
            "free_bytes": (None if spec is None else spec.disk_free_bytes),
            "pricing": DISK_UNPRICED,
            "known_terms": {},
            "unknown_terms": list(_EXTERNAL_UNPRICED_TERMS),
        },
        "external": {
            "required": refusal_class == REFUSAL_EXTERNAL_WORKING_SET_REQUIRED,
            "implemented": False,
            "future_mission": "M2",
            "unpriced_terms": list(_EXTERNAL_UNPRICED_TERMS),
        },
    }


def _candidate_trials(bounds_by_tile: Mapping[int, int]) -> list[dict]:
    """One honest trial per CPU rung (descending ladder order)."""
    return [
        {
            "tile": int(tile),
            "predicted_bound_bytes": int(bounds_by_tile[int(tile)]),
        }
        for tile in zsched.PRODUCT_TILE_LADDER
    ]


def build_dry_plan(spec: DryPlanSpec) -> dict:
    """Build the strict-JSON dry-plan report (PURE: no I/O, no execution)."""
    if not isinstance(spec, DryPlanSpec):
        raise ValueError(f"expected DryPlanSpec, got {type(spec).__name__}")

    raw = _require_int(spec.raw_available_bytes, "raw_available_bytes")
    if raw <= 0:
        return _empty_report(spec, REFUSAL_INVALID_OBSERVATION,
                             f"non-positive raw RAM observation {raw}")

    budget = (
        (4 * raw) // 5
        if spec.plan_budget_bytes is None
        else _require_int(spec.plan_budget_bytes, "plan_budget_bytes")
    )
    channels = _require_int(spec.channels, "channels", minimum=1)
    halo = _require_int(spec.halo_px, "halo_px")
    layout = _require_hw(spec.layout, "layout")
    rank = _require_int(spec.refinement_rank, "refinement_rank")
    profile = normalize_storage_profile(spec.storage_profile)

    cells: list[tuple[int, tuple[int, int]]] = []
    for index, cell in enumerate(spec.cells or ()):
        if isinstance(cell, (str, bytes)) or not isinstance(cell, (tuple, list)) or len(cell) != 2:
            raise ValueError(f"cells[{index}] must be (n_contributors, patch_hw), got {cell!r}")
        n = _require_int(cell[0], f"cells[{index}].n_contributors", minimum=1)
        hw = _require_hw(cell[1], f"cells[{index}].patch_hw")
        cells.append((n, hw))
    if not cells:
        raise ValueError("cells must contain at least one non-empty cell")

    if not spec.geometry_admissible:
        report = _empty_report(
            spec,
            REFUSAL_SCIENTIFIC_FLOOR,
            spec.scientific_floor_reason or "geometry is not scientifically admissible",
        )
        report["budget"]["plan_budget_bytes"] = int(budget)
        return report

    worst_hw = max((hw for _n, hw in cells), key=lambda hw: hw[0] * hw[1])
    max_n = max(n for n, _hw in cells)

    # ONE estimator (the execution/admission estimator), per CPU rung and cell.
    bounds_by_tile: dict[int, int] = {}
    for tile in zsched.PRODUCT_TILE_LADDER:
        bounds_by_tile[int(tile)] = max(
            int(
                zstream.estimate_streaming_peak_kib(
                    n, hw, int(tile), channels, storage_profile=profile
                )
            )
            * 1024
            for n, hw in cells
        )

    # The scheduler consumes the ADDED bound (baseline counted once as
    # ``shared_bytes``), exactly like the product ACTIVE path.
    added_by_tile = {
        int(tile): max(1, int(bound) - int(PROCESS_BASELINE_BYTES))
        for tile, bound in bounds_by_tile.items()
    }

    # Disk envelope: honest pricing only from explicit inputs.
    known_terms = _json_terms(spec.known_disk_terms or {})
    if spec.per_unit_cache_bytes is None:
        per_unit_cache = 0
    else:
        per_unit_cache = _require_int(
            spec.per_unit_cache_bytes, "per_unit_cache_bytes"
        )
        known_terms = {
            "per_cell_aligned_cache_bytes": int(per_unit_cache),
            **known_terms,
        }
    unknown_terms = list(_EXTERNAL_UNPRICED_TERMS)

    disk_free = (
        None
        if spec.disk_free_bytes is None
        else _require_int(spec.disk_free_bytes, "disk_free_bytes")
    )

    common = dict(
        backend=BACKEND_CPU,
        budget_bytes=int(budget),
        raw_available_bytes=int(raw),
        factor_num=4,
        factor_den=5,
        cpu_logical=_require_int(spec.cpu_logical, "cpu_logical", minimum=1),
        units_total=len(cells),
        shared_bytes=int(PROCESS_BASELINE_BYTES),
        per_unit_cache_bytes=int(per_unit_cache),
        per_unit_prefetch_bytes=0,
        disk_free_bytes=int(disk_free or 0),
        layout=(int(layout[0]), int(layout[1])),
        refinement_rank=int(rank),
        patch_hw=worst_hw,
        halo_px=int(halo),
        channels=int(channels),
        n_frames=int(max_n),
        prefetch_ladder=(0,),
        margin_num=int(DEFAULT_MARGIN_NUM),
        margin_den=int(DEFAULT_MARGIN_DEN),
        storage_profile=profile,
    )

    # --- in-memory plan (only claimed when the caller supplied its bound) ---
    inmem_report = {"considered": False, "feasible": None, "worst_bound_bytes": None}
    inmem_candidates: tuple = ()
    if spec.in_memory_bound_by_cell is not None:
        inmem_worst = max(
            _require_int(v, "in_memory_bound_by_cell entry", minimum=1)
            for v in spec.in_memory_bound_by_cell
        )
        inmem_ctx = zsched.PlanContext(
            phase=PHASE_CELL_STACK_INMEM,
            per_unit_stack_bytes=int(inmem_worst),
            **common,
        )
        inmem_candidates = zsched.enumerate_candidates(inmem_ctx)
        inmem_report = {
            "considered": True,
            "feasible": any(c.feasible for c in inmem_candidates),
            "worst_bound_bytes": int(inmem_worst),
        }

    # --- CPU streaming ladder: enumerate the SAME candidate model ---
    stream_ctx = zsched.PlanContext(
        phase=PHASE_CELL_STACK_STREAM,
        per_unit_stack_bytes=int(added_by_tile[int(zsched.PRODUCT_TILE_LADDER[0])]),
        per_tile_stack_bytes=dict(added_by_tile),
        tile_ladder=tuple(int(t) for t in zsched.PRODUCT_TILE_LADDER),
        **common,
    )
    stream_candidates = zsched.enumerate_candidates(stream_ctx)
    feasible_tiles = sorted({c.tile_px for c in stream_candidates if c.feasible}, reverse=True)

    trials = _candidate_trials(bounds_by_tile)
    for trial in trials:
        tile = trial["tile"]
        tile_candidates = [c for c in stream_candidates if c.tile_px == tile]
        trial["feasible"] = any(c.feasible for c in tile_candidates)
        trial["added_bytes"] = int(added_by_tile[tile])
        trial["reason_code"] = (
            tile_candidates[0].reason_code if tile_candidates else None
        )
        trial["margined_bound_bytes"] = int(
            ceil_margin_div(
                int(trial["added_bytes"]) + int(PROCESS_BASELINE_BYTES),
                int(DEFAULT_MARGIN_NUM),
                int(DEFAULT_MARGIN_DEN),
            )
        )

    selected_tile = int(feasible_tiles[0]) if feasible_tiles else None
    predicted_bound = None if selected_tile is None else int(bounds_by_tile[selected_tile])

    # Product preference: a complete in-memory plan is used when it fits.
    if inmem_report["feasible"]:
        selected_working_set = "inmem"
    elif selected_tile is not None:
        selected_working_set = "ram_tile"
    else:
        selected_working_set = None

    # --- GPU ladder is reported SEPARATELY (never part of CPU selection) ---
    gpu_report = {
        "ladder": None,
        "trials": [],
        "selected_tile": None,
        "minimum_tile_size": int(zsched.MIN_TILE_PX),
        "reason": "gpu_not_requested",
    }
    if spec.gpu_ladder is not None:
        gpu_tiles = tuple(int(t) for t in spec.gpu_ladder)
        if not gpu_tiles:
            raise ValueError("gpu_ladder must not be empty")
        if tuple(sorted(set(gpu_tiles), reverse=True)) != gpu_tiles:
            raise ValueError("gpu_ladder must be strictly descending and unique")
        if min(gpu_tiles) < int(zsched.MIN_TILE_PX):
            raise ValueError(
                f"gpu_ladder contains a tile below the GPU floor "
                f"{zsched.MIN_TILE_PX}: {min(gpu_tiles)}"
            )
        gpu_trials = []
        peaks = spec.gpu_device_peak_by_tile or {}
        for tile in gpu_tiles:
            entry = {"tile": int(tile), "device_peak_bytes": None}
            if peaks:
                key = int(tile)
                if key not in peaks and key not in {int(k) for k in peaks}:
                    raise ValueError(f"gpu_device_peak_by_tile is missing tile {tile}")
                value = peaks.get(key, peaks.get(str(key)))
                entry["device_peak_bytes"] = _require_int(
                    value, f"gpu_device_peak_by_tile[{tile}]"
                )
            gpu_trials.append(entry)
        gpu_report = {
            "ladder": [int(t) for t in gpu_tiles],
            "trials": gpu_trials,
            "selected_tile": None,
            "minimum_tile_size": int(zsched.MIN_TILE_PX),
            "reason": (
                "gpu_ladder_reported_separately; GPU selection is not part of "
                "this CPU dry-plan"
            ),
        }

    # --- refusal classification (honest, ordered) ---
    refusal_class = None
    refusal_detail = None
    refusal_causes: list[str] = []
    if selected_working_set is None:
        if inmem_report["feasible"] is False:
            refusal_causes.append(REFUSAL_NO_IN_MEMORY_PLAN)
        refusal_causes.append(REFUSAL_NO_BOUNDED_RAM_TILE)
        # The next required class is an external working set (future M2). It is
        # NEVER claimed to exist here.
        external_required = True
        refusal_class = REFUSAL_EXTERNAL_WORKING_SET_REQUIRED
        refusal_detail = (
            "the in-memory bound does not fit the PLAN budget and no bounded "
            "RAM tile fits it either; the next required class is an external "
            "working set (M2), which is NOT implemented in this build"
        )
    else:
        external_required = False

    disk_evaluable = (
        disk_free is not None
        and spec.per_unit_cache_bytes is not None
    )
    disk_pricing = DISK_PRICED if disk_evaluable else DISK_UNPRICED

    disk_block = {
        "free_bytes": disk_free,
        "pricing": disk_pricing,
        "known_terms": known_terms,
        "unknown_terms": unknown_terms,
        "unknown_terms_scope": (
            "future external working set (M2) only; they are NOT priced and "
            "MUST NOT be read as zero"
        ),
        "priced_envelope_bytes": (
            None if not disk_evaluable else int(per_unit_cache)
        ),
        "envelope_fits_free_bytes": (
            None
            if not disk_evaluable
            else bool(int(per_unit_cache) <= int(disk_free))
        ),
    }
    if (
        selected_working_set is not None
        and disk_evaluable
        and not disk_block["envelope_fits_free_bytes"]
    ):
        # A RAM plan whose known disk term cannot fit is still not executable.
        refusal_causes.append(REFUSAL_DISK_ENVELOPE_UNAVAILABLE)
        refusal_class = REFUSAL_DISK_ENVELOPE_UNAVAILABLE
        refusal_detail = (
            f"known disk envelope {per_unit_cache} B exceeds free {disk_free} B"
        )
        selected_working_set = None
        selected_tile = None
        predicted_bound = None

    report = {
        "schema": DRY_PLAN_SCHEMA,
        "mode": "dry_plan",
        "side_effects": "none",
        "ok": refusal_class is None,
        "admissible_for_launch": bool(
            refusal_class is None and disk_evaluable
        ),
        "refusal_class": refusal_class,
        "refusal_causes": refusal_causes,
        "refusal_detail": refusal_detail,
        "refusal_classes_considered": list(REFUSAL_CLASSES),
        "budget": {
            "raw_available_bytes": int(raw),
            "plan_budget_bytes": int(budget),
            "factor_num": 4,
            "factor_den": 5,
            "margin_num": int(DEFAULT_MARGIN_NUM),
            "margin_den": int(DEFAULT_MARGIN_DEN),
        },
        "geometry": {
            "layout": [int(layout[0]), int(layout[1])],
            "refinement_rank": int(rank),
            "halo_px": int(halo),
            "floors": dict(spec.floors or {}),
        },
        "cells": {
            "count": len(cells),
            "worst_n_contributors": int(max_n),
            "worst_patch_hw": [int(worst_hw[0]), int(worst_hw[1])],
        },
        "storage_profile": profile,
        "cpu": {
            "ladder": [int(t) for t in zsched.PRODUCT_TILE_LADDER],
            "preferred_tile_size": int(zsched.PRODUCT_TILE_LADDER[0]),
            "minimum_tile_size": int(zsched.MIN_CPU_TILE_PX),
            "trials": trials,
            "selected_tile": selected_tile,
            "selected_working_set": selected_working_set,
            "predicted_bound_bytes": predicted_bound,
        },
        "in_memory": inmem_report,
        "gpu": gpu_report,
        "disk": disk_block,
        "external": {
            "required": bool(external_required),
            "implemented": False,
            "future_mission": "M2",
            "unpriced_terms": list(_EXTERNAL_UNPRICED_TERMS),
        },
    }
    return report