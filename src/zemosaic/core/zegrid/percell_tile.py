"""Experimental per-Cell GPU streaming-tile planning (ZM-GPU-PERCELL-PRODUCT).

This module is the SINGLE focused helper for the experimental
``zegrid_gpu_percell_tiles`` product option. It is deliberately pure: no GPU
probe, no CuPy import, no scheduler mutation and no science/geometry input.

Contract
--------
* Every ACTIVE Cell gets its OWN exact ACTIVE GPU ladder derived from that
  Cell's real ``(n_i, patch_hw_i)`` pair -- never from an independent
  ``max(n)`` x ``max(area)`` product.
* Each rung keeps its EXACT device workspace
  (:func:`gpu.gpu_tile_workspace_bytes`) and its EXACT added host streaming
  bound (:func:`streaming.estimate_streaming_peak_kib` minus the process
  baseline already counted once by ``shared_bytes``).
* The per-Cell admitted tile is the LARGEST rung whose exact device workspace
  fits the R55-E10 selection budget AND whose exact projected host peak (shared
  once + this Cell's added stack + the real producer owner at prefetch >= 1 +
  the real resident prefetch credit) fits the plan budget with the SAME margin
  the envelope validator applies.
* The per-Cell reservation (``auto_layout.reservation_bytes``) is an explicit
  FLOOR: a Cell whose own mode bound is below its measured-envelope floor is
  admitted against that floor, never against a smaller optimistic number.
* OFF is represented by ``None`` everywhere: callers that do not opt in keep
  their historical behaviour byte for byte.

The serialized extension is OPTIONAL, strict and versioned. It lives inside the
existing admission object under ``percell_tiles`` (see
:func:`zegrid.run_resume.validate_admission`). A v1 store (no field) keeps its
exact old reader behaviour; a v2 store is rejected by an older reader because
the version is bumped only when the field is present.
"""

from __future__ import annotations

import hashlib

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from . import auto_layout as zal
from . import gpu as zgpu
from . import ram_contract as rcontract
from . import resource_envelope as renv
from . import scheduler as zsched
from . import streaming as zstream

#: Policy identifier persisted with every per-Cell plan. An unknown policy is
#: refused (fail closed); it is never guessed or coerced.
PERCELL_POLICY = "percell_v1"
#: Serialized schema of the optional admission extension.
PERCELL_SCHEMA = "zegrid-percell-tile-v1"
#: Durable shrink-cap document (separate from the immutable admission).
SHRINK_CAPS_SCHEMA = "zegrid-percell-shrink-caps-v1"
#: Schema label of the heterogeneous per-Cell FINAL executable plan hash.
#: It EXTENDS ``zsched.ACTIVE_FINAL_PLAN_SCHEMA`` (v1); it never replaces it.
#: A homogeneous run keeps the v1 label and the exact strict hash byte for
#: byte. v2 is emitted ONLY when a frozen per-Cell plan governs the run,
#: because such a run has TWO distinct final facts v1 cannot express: the
#: scheduler REFERENCE candidate (one real Cell + its exact mapping) and the
#: per-Cell EFFECTIVE tiles (heterogeneous, shrink-only).
FINAL_PLAN_SCHEMA = "zemosaic-r55d2-percell-final-executable-plan-v2"

#: The GPU ladder stays at/above the qualified 32 px floor. This is a POLICY
#: floor (not a physical one) and is deliberately unchanged by this feature.
GPU_STREAMING_TILE_MINIMUM = int(zsched.MIN_TILE_PX)
GPU_STREAMING_TILE_LADDER = tuple(int(t) for t in zsched.GPU_PRODUCT_TILE_LADDER)


def active_gpu_tile_ladder(max_tile: Any) -> tuple[int, ...]:
    """Deterministic descending GPU tile ladder (identical to R55-E9).

    ``max_tile`` is the exact interior side already derived from the fresh VRAM
    budget and clamped to the patch geometry. The top rung is preserved, then
    coarse 64-px-aligned rungs down to 128, then the canonical GPU tail. The
    result is strictly descending and deduplicated.
    """
    if isinstance(max_tile, bool) or not isinstance(max_tile, int):
        raise ValueError(f"max_tile must be a plain int, got {max_tile!r}")
    if int(max_tile) < GPU_STREAMING_TILE_MINIMUM:
        return ()
    top = int(max_tile)
    rungs = [top]
    coarse = (top // 64) * 64
    while coarse >= 128:
        if coarse < rungs[-1]:
            rungs.append(int(coarse))
        coarse -= 64
    for tail_tile in GPU_STREAMING_TILE_LADDER:
        if int(tail_tile) < rungs[-1]:
            rungs.append(int(tail_tile))
    return tuple(rungs)


def _stream_total_bytes(
    n_contributors: int, patch_hw: tuple[int, int], tile_px: int, storage_profile: str
) -> int:
    """TOTAL streaming peak (bytes) for one Cell/rung, profile-aware."""
    return int(
        zstream.estimate_streaming_peak_kib(
            int(n_contributors),
            (int(patch_hw[0]), int(patch_hw[1])),
            int(tile_px),
            3,
            storage_profile=str(storage_profile),
        )
        * 1024
    )


def _added_host_bound(total_bytes: int) -> int:
    """ADDED host bound above the process baseline counted once in ``shared``."""
    return max(1, int(total_bytes) - int(rcontract.PROCESS_BASELINE_BYTES))


def _device_peak(n_contributors: int, tile_px: int) -> int:
    side = int(tile_px) + 2 * int(zgpu.GPU_TAPER_HALO_PX)
    return int(
        zgpu.gpu_tile_workspace_bytes(int(n_contributors), side, side, 3)
    )


def _host_aggregate(
    *,
    shared_bytes: int,
    added_stack_bytes: int,
    producer_owner_bytes: int,
    prefetch_units: int,
    prefetch_resident_per_slot_bytes: int,
) -> int:
    return int(
        rcontract.projected_host_peak_bytes(
            int(shared_bytes),
            int(added_stack_bytes),
            int(producer_owner_bytes),
            int(prefetch_units),
            int(prefetch_resident_per_slot_bytes),
        )
    )


@dataclass(frozen=True)
class CellTileFacts:
    """The real per-Cell inputs. No cross-Cell maximum is ever formed."""

    cell_id: str
    n_contributors: int
    patch_hw: tuple[int, int]
    patch_area_px: int

    @classmethod
    def build(cls, cell_id: Any, n_contributors: Any, patch_hw: Any, patch_area_px: Any = None):
        cid = str(cell_id)
        if not cid:
            raise ValueError("cell_id must be a non-empty string")
        n = int(n_contributors)
        if n < 1:
            raise ValueError(f"n_contributors must be >= 1, got {n}")
        if (
            not isinstance(patch_hw, (tuple, list))
            or len(patch_hw) != 2
        ):
            raise ValueError(f"patch_hw must be (h, w), got {patch_hw!r}")
        hw = (int(patch_hw[0]), int(patch_hw[1]))
        if hw[0] < 1 or hw[1] < 1:
            raise ValueError(f"patch_hw must be positive, got {hw!r}")
        area = hw[0] * hw[1] if patch_area_px is None else int(patch_area_px)
        if area < 1:
            raise ValueError(f"patch_area_px must be >= 1, got {area}")
        return cls(cell_id=cid, n_contributors=n, patch_hw=hw, patch_area_px=area)


@dataclass(frozen=True)
class PerCellEntry:
    """One Cell's frozen per-Cell tile decision."""

    cell_id: str
    n_contributors: int
    patch_hw: tuple[int, int]
    patch_area_px: int
    tile_px: int
    declared_ladder: tuple[int, ...]
    device_peak_bytes_by_tile: tuple[tuple[int, int], ...]
    host_bound_by_tile: tuple[tuple[int, int], ...]
    reservation_bytes: int
    host_aggregate_bytes: int
    device_peak_bytes: int
    basis: str = "plan"

    def host_bound_for(self, tile_px: int) -> int:
        return dict(self.host_bound_by_tile)[int(tile_px)]

    def device_peak_for(self, tile_px: int) -> int:
        return dict(self.device_peak_bytes_by_tile)[int(tile_px)]

    def admitted_host_bound(self, tile_px: int) -> int:
        """THE one floor-adjusted ADDED host bound for this Cell/rung.

        It is used IDENTICALLY by PLAN, APPLY and RUN. The measured reservation
        is a FLOOR; the process baseline is already counted once in
        ``shared_bytes``, so only ``reservation - baseline`` can be the added
        floor. Returning the raw model bound here was the D1 runtime hole.
        """
        raw = int(self.host_bound_for(tile_px))
        floor = max(
            0,
            int(self.reservation_bytes) - int(rcontract.PROCESS_BASELINE_BYTES),
        )
        return int(max(raw, floor))

    def admitted_host_bound_by_tile(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(t), int(self.admitted_host_bound(t))) for t in self.declared_ladder
        )

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "n_contributors": int(self.n_contributors),
            "patch_hw": [int(self.patch_hw[0]), int(self.patch_hw[1])],
            "patch_area_px": int(self.patch_area_px),
            "tile_px": int(self.tile_px),
            "device_workspace_bytes": int(self.device_peak_bytes),
            "host_aggregate_bytes": int(self.host_aggregate_bytes),
            "reservation_bytes": int(self.reservation_bytes),
            "declared_ladder": [int(t) for t in self.declared_ladder],
            "device_peak_bytes_by_tile": {
                str(int(t)): int(b) for t, b in self.device_peak_bytes_by_tile
            },
            "host_bound_bytes_by_tile": {
                str(int(t)): int(b) for t, b in self.host_bound_by_tile
            },
            "basis": str(self.basis),
        }


@dataclass(frozen=True)
class PerCellPlan:
    """The complete per-Cell plan persisted in the durable admission."""

    entries: tuple[PerCellEntry, ...]
    device_budget_bytes: int
    selection_budget_bytes: int
    host_budget_bytes: int
    max_tile_px: int
    storage_profile: str
    policy: str = PERCELL_POLICY
    schema: str = PERCELL_SCHEMA
    entries_by_cell: Mapping[str, PerCellEntry] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self):
        object.__setattr__(
            self,
            "entries_by_cell",
            {entry.cell_id: entry for entry in self.entries},
        )
        if len(self.entries_by_cell) != len(self.entries):
            raise ValueError("per-Cell plan has duplicate cell ids")

    def entry_for(self, cell_id: Any) -> PerCellEntry:
        cid = str(cell_id)
        if cid not in self.entries_by_cell:
            raise KeyError(f"per-Cell plan has no entry for cell {cid!r}")
        return self.entries_by_cell[cid]

    def cell_ids(self) -> tuple[str, ...]:
        return tuple(entry.cell_id for entry in self.entries)

    def with_caps(self, caps: Mapping[str, Any]) -> "PerCellPlan":
        """Return a plan whose committed Cells can never grow past their cap.

        Used on resume: a Cell that actually SHRANK in a previous attempt has
        its durable effective tile recorded; the resumed plan starts at that
        tile (or lower), never back at the larger planned rung.
        """
        capped = []
        for entry in self.entries:
            raw = caps.get(entry.cell_id)
            if raw is None:
                capped.append(entry)
                continue
            cap = _as_plain_int(raw, f"cap[{entry.cell_id}]", minimum=1)
            eligible = [t for t in entry.declared_ladder if int(t) <= cap]
            if not eligible:
                raise ValueError(
                    f"Cell {entry.cell_id!r} committed tile cap {cap} is below "
                    "its declared GPU ladder"
                )
            tile = int(eligible[0])
            if tile == int(entry.tile_px):
                capped.append(entry)
                continue
            capped.append(
                replace(
                    entry,
                    tile_px=tile,
                    device_peak_bytes=int(
                        entry.device_peak_for(tile)
                    ),
                    basis="resume_capped_by_committed_tile",
                )
            )
        return PerCellPlan(
            entries=tuple(capped),
            device_budget_bytes=int(self.device_budget_bytes),
            selection_budget_bytes=int(self.selection_budget_bytes),
            host_budget_bytes=int(self.host_budget_bytes),
            max_tile_px=max(int(e.tile_px) for e in capped),
            storage_profile=str(self.storage_profile),
            policy=str(self.policy),
            schema=str(self.schema),
        )

    def to_admission(self) -> dict:
        return {
            "schema": str(self.schema),
            "policy": str(self.policy),
            "device_budget_bytes": int(self.device_budget_bytes),
            "selection_budget_bytes": int(self.selection_budget_bytes),
            "host_budget_bytes": int(self.host_budget_bytes),
            "max_tile_px": int(self.max_tile_px),
            "storage_profile": str(self.storage_profile),
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _as_plain_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a plain int, got {value!r}")
    if minimum is not None and int(value) < int(minimum):
        raise ValueError(f"{label} must be >= {minimum}, got {value!r}")
    return int(value)


def _strict_keys(raw: Mapping, label: str, required: Iterable[str], optional: Iterable[str] = ()):
    allowed = set(required) | set(optional)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown key(s): {unknown}")
    missing = sorted(set(required) - set(raw))
    if missing:
        raise ValueError(f"{label} is missing required key(s): {missing}")


def _pairs_map(raw: Any, label: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    out: list[tuple[int, int]] = []
    for key, value in raw.items():
        try:
            tile = int(str(key))
        except (TypeError, ValueError):
            raise ValueError(f"{label} has a non-integer tile key: {key!r}")
        out.append((tile, _as_plain_int(value, f"{label}[{tile}]", minimum=1)))
    out.sort()
    if len({tile for tile, _ in out}) != len(out):
        raise ValueError(f"{label} has duplicate tiles")
    return tuple(out)


def plan_from_admission(raw: Any, *, expected_cell_ids=None) -> PerCellPlan:
    """Strictly parse the optional admission extension (fail closed).

    When ``expected_cell_ids`` is supplied, EXACT coverage is enforced: every
    expected Cell must have an entry and no extra entry is accepted. The
    resume path always passes the run's memberships; the raw validator cannot
    know them.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("percell_tiles must be an object")
    _strict_keys(
        raw,
        "percell_tiles",
        (
            "schema",
            "policy",
            "device_budget_bytes",
            "selection_budget_bytes",
            "host_budget_bytes",
            "max_tile_px",
            "storage_profile",
            "entries",
        ),
    )
    if str(raw["schema"]) != PERCELL_SCHEMA:
        raise ValueError(
            f"percell_tiles.schema {raw['schema']!r} != {PERCELL_SCHEMA!r}"
        )
    if str(raw["policy"]) != PERCELL_POLICY:
        raise ValueError(
            f"percell_tiles.policy {raw['policy']!r} is not supported "
            f"({PERCELL_POLICY!r})"
        )
    device_budget = _as_plain_int(
        raw["device_budget_bytes"], "device_budget_bytes", minimum=1
    )
    selection_budget = _as_plain_int(
        raw["selection_budget_bytes"], "selection_budget_bytes", minimum=1
    )
    host_budget = _as_plain_int(
        raw["host_budget_bytes"], "host_budget_bytes", minimum=1
    )
    # D2(c): the E10 selection budget can never exceed the policy device budget.
    if int(selection_budget) > int(device_budget):
        raise ValueError(
            "percell_tiles.selection_budget_bytes "
            f"{selection_budget} exceeds device_budget_bytes {device_budget}"
        )
    entries_raw = raw["entries"]
    if not isinstance(entries_raw, list):
        raise ValueError("percell_tiles.entries must be a list")
    entries: list[PerCellEntry] = []
    seen: set[str] = set()
    for index, item in enumerate(entries_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"percell_tiles.entries[{index}] must be an object")
        _strict_keys(
            item,
            f"percell_tiles.entries[{index}]",
            (
                "cell_id",
                "n_contributors",
                "patch_hw",
                "patch_area_px",
                "tile_px",
                "device_workspace_bytes",
                "host_aggregate_bytes",
                "reservation_bytes",
                "declared_ladder",
                "device_peak_bytes_by_tile",
                "host_bound_bytes_by_tile",
                "basis",
            ),
        )
        cid = str(item["cell_id"])
        if not cid or cid in seen:
            raise ValueError(
                f"percell_tiles.entries[{index}] has empty/duplicate cell_id {cid!r}"
            )
        seen.add(cid)
        hw_raw = item["patch_hw"]
        if not isinstance(hw_raw, (list, tuple)) or len(hw_raw) != 2:
            raise ValueError(
                f"percell_tiles.entries[{index}].patch_hw must be [h, w]"
            )
        hw = (
            _as_plain_int(hw_raw[0], "patch_hw[0]", minimum=1),
            _as_plain_int(hw_raw[1], "patch_hw[1]", minimum=1),
        )
        ladder_raw = item["declared_ladder"]
        if not isinstance(ladder_raw, list) or not ladder_raw:
            raise ValueError(
                f"percell_tiles.entries[{index}].declared_ladder must be a "
                "non-empty list"
            )
        ladder = tuple(
            _as_plain_int(t, "declared_ladder[]", minimum=1) for t in ladder_raw
        )
        if any(b >= a for a, b in zip(ladder, ladder[1:])):
            raise ValueError(
                f"percell_tiles.entries[{index}].declared_ladder must be "
                "strictly descending"
            )
        device_by_tile = dict(
            _pairs_map(
                item["device_peak_bytes_by_tile"],
                "device_peak_bytes_by_tile",
            )
        )
        host_by_tile = dict(
            _pairs_map(
                item["host_bound_bytes_by_tile"],
                "host_bound_bytes_by_tile",
            )
        )
        if set(device_by_tile) != set(ladder):
            raise ValueError(
                f"percell_tiles.entries[{index}] device map does not match "
                "its declared ladder"
            )
        if set(host_by_tile) != set(ladder):
            raise ValueError(
                f"percell_tiles.entries[{index}] host map does not match "
                "its declared ladder"
            )
        device_map = tuple((t, device_by_tile[t]) for t in ladder)
        host_map = tuple((t, host_by_tile[t]) for t in ladder)
        tile = _as_plain_int(item["tile_px"], "tile_px", minimum=1)
        if tile not in ladder:
            raise ValueError(
                f"percell_tiles.entries[{index}] tile_px {tile} is outside "
                "its declared ladder"
            )
        selected_workspace = _as_plain_int(
            item["device_workspace_bytes"],
            "device_workspace_bytes",
            minimum=1,
        )
        # D2(a): the SELECTED workspace must be the exact map value for the
        # selected tile (a hand-edited scalar cannot diverge from its map).
        if int(selected_workspace) != int(device_by_tile[int(tile)]):
            raise ValueError(
                f"percell_tiles.entries[{index}] device_workspace_bytes "
                f"{selected_workspace} != device_peak_bytes_by_tile[{tile}] "
                f"{device_by_tile[int(tile)]}"
            )
        # Selected rung must fit the E10 selection budget. Nonselected rungs are
        # deliberately NOT rejected here: a legitimate frozen ladder may carry a
        # taller rung that the current contract never selects.
        if int(selected_workspace) > int(selection_budget):
            raise ValueError(
                f"percell_tiles.entries[{index}] selected device peak "
                f"{selected_workspace} exceeds selection_budget_bytes "
                f"{selection_budget}"
            )
        entries.append(
            PerCellEntry(
                cell_id=cid,
                n_contributors=_as_plain_int(
                    item["n_contributors"], "n_contributors", minimum=1
                ),
                patch_hw=hw,
                patch_area_px=_as_plain_int(
                    item["patch_area_px"], "patch_area_px", minimum=1
                ),
                tile_px=tile,
                declared_ladder=ladder,
                device_peak_bytes_by_tile=device_map,
                host_bound_by_tile=host_map,
                reservation_bytes=_as_plain_int(
                    item["reservation_bytes"], "reservation_bytes", minimum=0
                ),
                host_aggregate_bytes=_as_plain_int(
                    item["host_aggregate_bytes"], "host_aggregate_bytes", minimum=1
                ),
                device_peak_bytes=int(selected_workspace),
                basis=str(item["basis"]),
            )
        )
    if not entries:
        raise ValueError("percell_tiles.entries must not be empty")
    max_tile = _as_plain_int(raw["max_tile_px"], "max_tile_px", minimum=1)
    for entry in entries:
        if int(entry.tile_px) > int(max_tile):
            raise ValueError(
                f"percell_tiles entry {entry.cell_id} tile {entry.tile_px} "
                f"exceeds max_tile_px {max_tile}"
            )
        if int(entry.device_peak_bytes) > int(raw["device_budget_bytes"]):
            raise ValueError(
                f"percell_tiles entry {entry.cell_id} device peak "
                f"{entry.device_peak_bytes} exceeds device_budget_bytes "
                f"{raw['device_budget_bytes']}"
            )
        if int(entry.reservation_bytes) > int(host_budget):
            raise ValueError(
                f"percell_tiles entry {entry.cell_id} reservation "
                f"{entry.reservation_bytes} exceeds host_budget_bytes "
                f"{host_budget}"
            )
        # D2(b): the admitted aggregate cannot exceed its own host budget.
        if int(entry.host_aggregate_bytes) > int(host_budget):
            raise ValueError(
                f"percell_tiles entry {entry.cell_id} host_aggregate_bytes "
                f"{entry.host_aggregate_bytes} exceeds host_budget_bytes "
                f"{host_budget}"
            )
    if expected_cell_ids is not None:
        expected = {str(cid) for cid in expected_cell_ids}
        got = {entry.cell_id for entry in entries}
        if got != expected:
            raise ValueError(
                "percell_tiles coverage mismatch (missing="
                f"{sorted(expected - got)}, extra={sorted(got - expected)})"
            )
    plan = PerCellPlan(
        entries=tuple(entries),
        device_budget_bytes=int(device_budget),
        selection_budget_bytes=int(selection_budget),
        host_budget_bytes=int(host_budget),
        max_tile_px=max_tile,
        storage_profile=str(raw["storage_profile"]),
        policy=str(raw["policy"]),
        schema=str(raw["schema"]),
    )
    return plan


def build_percell_plan(
    *,
    cells: Sequence[CellTileFacts],
    shared_bytes: int,
    producer_owner_bytes: int,
    prefetch_units: int,
    prefetch_resident_per_slot_bytes: int,
    plan_budget_bytes: int,
    producer_owner_by_cell: Mapping[str, Any] | None = None,
    prefetch_resident_by_cell: Mapping[str, Any] | None = None,
    device_budget_bytes: int,
    selection_budget_bytes: int,
    storage_profile: str,
    observations: Sequence[Any] = (),
    max_gpu_tile: int | None = None,
    margin_num: int = int(renv.DEFAULT_MARGIN_NUM),
    margin_den: int = int(renv.DEFAULT_MARGIN_DEN),
) -> PerCellPlan:
    """Build the exact per-Cell plan (pure arithmetic, deterministic order).

    Raises :class:`ValueError` when a Cell has no admissible GPU rung (the run
    can then decline the GPU exactly as the historical whole-run path does).
    Cells are evaluated in ``cell_id`` sorted order so the plan is stable.
    """
    if not cells:
        raise ValueError("per-Cell GPU planning requires at least one Cell")
    shared = _as_plain_int(shared_bytes, "shared_bytes", minimum=0)
    producer = _as_plain_int(producer_owner_bytes, "producer_owner_bytes", minimum=0)
    prefetch = _as_plain_int(prefetch_units, "prefetch_units", minimum=0)
    resident = _as_plain_int(
        prefetch_resident_per_slot_bytes,
        "prefetch_resident_per_slot_bytes",
        minimum=0,
    )
    plan_budget = _as_plain_int(plan_budget_bytes, "plan_budget_bytes", minimum=1)
    device_budget = _as_plain_int(device_budget_bytes, "device_budget_bytes", minimum=1)
    selection_budget = _as_plain_int(
        selection_budget_bytes, "selection_budget_bytes", minimum=1
    )
    margin_n = _as_plain_int(margin_num, "margin_num", minimum=0)
    margin_d = _as_plain_int(margin_den, "margin_den", minimum=1)
    if margin_n > margin_d:
        raise ValueError(f"invalid margin {margin_n}/{margin_d}")

    entries: list[PerCellEntry] = []
    for facts in sorted(cells, key=lambda item: item.cell_id):
        n = int(facts.n_contributors)
        hw = (int(facts.patch_hw[0]), int(facts.patch_hw[1]))
        area = int(facts.patch_area_px)
        max_tile = zgpu.choose_gpu_tile_size(n, 3, hw, int(device_budget))
        if max_tile is None or int(max_tile) < GPU_STREAMING_TILE_MINIMUM:
            raise ValueError(
                f"Cell {facts.cell_id!r} has no GPU streaming tile >= "
                f"{GPU_STREAMING_TILE_MINIMUM} within the device budget "
                f"({device_budget} B); the GPU cannot cover this Cell"
            )
        full_ladder = active_gpu_tile_ladder(int(max_tile))
        if max_gpu_tile is not None:
            full_ladder = tuple(
                t for t in full_ladder if int(t) <= int(max_gpu_tile)
            )
        if not full_ladder:
            raise ValueError(
                f"Cell {facts.cell_id!r} has no GPU rung at or below the frozen "
                f"tile ceiling {max_gpu_tile}"
            )
        device_by_tile = {
            int(t): _device_peak(n, int(t)) for t in full_ladder
        }
        # Q1: the declared ladder is the EXACT product E10-pruned ladder
        # (declared rungs whose exact device workspace fits the selection
        # budget), exactly like the historical ACTIVE `gpu_tiles` tuple.
        ladder = tuple(
            int(t)
            for t in full_ladder
            if int(device_by_tile[int(t)]) <= int(selection_budget)
        )
        if not ladder:
            raise ValueError(
                f"Cell {facts.cell_id!r} has no GPU rung whose exact device "
                f"workspace fits the E10 selection budget {selection_budget} B"
            )
        device_map = tuple((int(t), int(device_by_tile[int(t)])) for t in ladder)
        host_map = tuple(
            (
                int(t),
                _added_host_bound(
                    _stream_total_bytes(n, hw, int(t), str(storage_profile))
                ),
            )
            for t in ladder
        )
        host_by_tile = dict(host_map)
        # Q2: the producer/cache owner and the per-slot residency are REAL
        # per-Cell facts when supplied. K=1 bounds the consumer, not the queued
        # prefetched owners, so they must be priced per Cell.
        owner_i = int(
            producer
            if producer_owner_by_cell is None
            else producer_owner_by_cell.get(str(facts.cell_id), producer)
        )
        resident_i = int(
            resident
            if prefetch_resident_by_cell is None
            else prefetch_resident_by_cell.get(str(facts.cell_id), resident)
        )
        admitted_tile = None
        admitted_aggregate = None
        reservation = None
        for tile in ladder:
            peak = int(device_by_tile[int(tile)])
            if peak > int(selection_budget):
                continue
            reservation_i = int(
                zal.reservation_bytes(
                    n,
                    area,
                    _stream_total_bytes(
                        n, hw, int(tile), str(storage_profile)
                    ),
                    observations=observations,
                )
            )
            added_floor = max(
                int(host_by_tile[int(tile)]),
                max(
                    0,
                    int(reservation_i) - int(rcontract.PROCESS_BASELINE_BYTES),
                ),
            )
            aggregate = _host_aggregate(
                shared_bytes=shared,
                added_stack_bytes=int(added_floor),
                producer_owner_bytes=int(owner_i),
                prefetch_units=prefetch,
                prefetch_resident_per_slot_bytes=int(resident_i),
            )
            if (
                renv.ceil_margin_div(aggregate, margin_n, margin_d)
                > int(plan_budget)
            ):
                continue
            admitted_tile = int(tile)
            admitted_aggregate = int(aggregate)
            reservation = int(reservation_i)
            break
        if admitted_tile is None:
            raise ValueError(
                f"Cell {facts.cell_id!r} has no individually admissible GPU rung "
                f"(device selection budget {selection_budget} B, plan budget "
                f"{plan_budget} B, margin {margin_n}/{margin_d})"
            )
        entries.append(
            PerCellEntry(
                cell_id=facts.cell_id,
                n_contributors=n,
                patch_hw=hw,
                patch_area_px=area,
                tile_px=int(admitted_tile),
                declared_ladder=tuple(int(t) for t in ladder),
                device_peak_bytes_by_tile=tuple(device_map),
                host_bound_by_tile=tuple(host_map),
                reservation_bytes=int(reservation),
                host_aggregate_bytes=int(admitted_aggregate),
                device_peak_bytes=int(device_by_tile[int(admitted_tile)]),
                basis="plan",
            )
        )
    return PerCellPlan(
        entries=tuple(entries),
        device_budget_bytes=int(device_budget),
        selection_budget_bytes=int(selection_budget),
        host_budget_bytes=int(plan_budget),
        max_tile_px=max(int(entry.tile_px) for entry in entries),
        storage_profile=str(storage_profile),
    )


def reference_entry(plan: PerCellPlan) -> PerCellEntry:
    """The REAL Cell whose own admission is the most constraining.

    The choice is a deterministic function of one actual Cell pair, never an
    independent ``max(n)`` x ``max(area)`` product. Ties are broken by the
    lexicographic ``cell_id`` so the plan is reproducible.
    """
    return sorted(
        plan.entries,
        key=lambda entry: (
            -int(entry.host_aggregate_bytes),
            -int(entry.patch_area_px),
            -int(entry.n_contributors),
            entry.cell_id,
        ),
    )[0]


def attach_percell(decision, plan: PerCellPlan):
    """Return ``decision`` with the per-Cell plan attached (frozen dataclass)."""
    return replace(decision, percell_plan=plan, percell_entries=tuple(plan.entries))


def plan_frozen_from_admission(raw: Any) -> PerCellPlan:
    return plan_from_admission(raw)


# ---------------------------------------------------------------------------
# Heterogeneous final executable plan hash (V3)
# ---------------------------------------------------------------------------


def final_plan_hash(
    *,
    reference_candidate,
    reference_mapping,
    mapping,
    plan: "PerCellPlan",
    effective_tiles,
    identity,
    envelope,
    tile_ladder=zsched.PRODUCT_TILE_LADDER,
) -> str:
    """Deterministic hash of the heterogeneous per-Cell FINAL decision.

    WHY THIS EXISTS
    ---------------
    :func:`scheduler.active_final_plan_hash` is the exact, strict identity of
    ONE homogeneous executable decision: one candidate plus the ONE mapping
    that candidate produces. It is deliberately NOT relaxed here -- a per-Cell
    run genuinely executes DIFFERENT tiles per Cell, so its final execable
    decision is a PAIR of facts:

    * the scheduler REFERENCE decision -- the admitted candidate of one REAL
      reference Cell and the exact mapping ``active_gpu_mapping_for_candidate``
      derives from it (never a forged candidate at another Cell's tile);
    * the frozen per-Cell plan and the per-Cell EFFECTIVE tiles that executed.

    VALIDATION SCOPE (REWORK-1, L1) -- stated exactly, never overstated:
    * the REFERENCE half is validated by CALLING the strict scheduler function
      itself: candidate fields, ``reference_mapping`` derived from that
      candidate, the frozen identity and the envelope TYPE are checked there,
      and its hash is part of the payload;
    * the RUN-WIDE envelope is checked HERE and separately (GPU backend, K=1,
      ``gpu_tile`` == largest effective per-Cell tile, prefetch/workers equal to
      the run-level mapping). The scheduler function does not cross-check
      ``envelope.gpu_tile`` against a mapping, and this function does not claim
      it does.
    * the per-Cell half (coverage, declared rung, ``tile <= admitted``) is
      checked here.

    ``mapping`` is the run-level executable mapping published in the manifest:
    identical to ``reference_mapping`` except ``gpu_streaming_tile``, which is
    the LARGEST effective per-Cell tile (the documented descriptive rule of
    ``_r55_active_gpu_apply``). Any other difference is refused.
    """
    if not isinstance(plan, PerCellPlan):
        raise zsched.SchedulerConfigError(
            f"final plan hash requires a PerCellPlan, got {type(plan).__name__}"
        )
    if not isinstance(reference_mapping, Mapping) or not isinstance(mapping, Mapping):
        raise zsched.SchedulerConfigError(
            "final plan hash requires both the reference and the run-level "
            "mapping"
        )
    if not isinstance(effective_tiles, Mapping) or not effective_tiles:
        raise zsched.SchedulerConfigError(
            "final plan hash requires a non-empty effective tile mapping"
        )
    ladder = tuple(int(t) for t in tile_ladder)
    # 1) REFERENCE half: the strict scheduler validator/hash is used AS IS.
    reference_hash = zsched.active_final_plan_hash(
        candidate=reference_candidate,
        mapping=reference_mapping,
        identity=identity,
        envelope=envelope,
        tile_ladder=ladder,
    )
    # 2) Per-Cell half: EXACT coverage, declared rung, never above the admitted
    #    (or resume-capped) per-Cell tile. No growth is representable.
    expected = set(plan.cell_ids())
    got = {str(cid) for cid in effective_tiles}
    if got != expected:
        raise zsched.ActivePlanRejected(
            renv.REASON_INTERNAL_ERROR,
            "final per-Cell plan hash coverage mismatch (missing="
            f"{sorted(expected - got)}, extra={sorted(got - expected)})",
            mapping=mapping,
        )
    normalised: dict[str, int] = {}
    for cid in plan.cell_ids():
        raw = effective_tiles[cid]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise zsched.SchedulerConfigError(
                f"effective tile for Cell {cid!r} must be a plain int, "
                f"got {type(raw).__name__}"
            )
        tile = int(raw)
        entry = plan.entry_for(cid)
        if tile not in entry.declared_ladder:
            raise zsched.ActivePlanRejected(
                renv.REASON_SHRINK_TILE,
                f"effective tile {tile} for Cell {cid!r} is outside its "
                f"declared ladder {list(entry.declared_ladder)}",
                mapping=mapping,
            )
        if tile > int(entry.tile_px):
            raise zsched.GrowthRejected(
                f"effective tile {tile} for Cell {cid!r} grows above its "
                f"admitted tile {int(entry.tile_px)}"
            )
        normalised[str(cid)] = tile
    max_tile = max(normalised.values())
    # 3) The run-level mapping is the reference mapping with EXACTLY ONE
    #    descriptive field replaced by the largest effective per-Cell tile.
    executable = dict(reference_mapping)
    executable["gpu_streaming_tile"] = int(max_tile)
    if dict(mapping) != executable:
        raise zsched.ActivePlanRejected(
            renv.REASON_INTERNAL_ERROR,
            "final per-Cell plan hash mapping is neither the reference mapping "
            f"nor its documented run-level max (mapping={dict(mapping)!r}, "
            f"expected={executable!r})",
            mapping=dict(mapping),
        )
    # 4) The product envelope must describe the SAME execution: GPU, K=1, the
    #    run-level max tile and the mapping's own knobs.
    env = envelope.to_dict() if hasattr(envelope, "to_dict") else dict(envelope)
    if str(env.get("backend")) != str(zsched.BACKEND_GPU):
        raise zsched.ActivePlanRejected(
            renv.REASON_UNKNOWN_BACKEND,
            f"final per-Cell plan hash requires a GPU envelope, got "
            f"backend={env.get('backend')!r}",
            mapping=dict(mapping),
        )
    if env.get("gpu_tile") is None or int(env["gpu_tile"]) != int(max_tile):
        raise zsched.ActivePlanRejected(
            renv.REASON_INTERNAL_ERROR,
            "final per-Cell plan hash envelope gpu_tile "
            f"{env.get('gpu_tile')!r} is not the largest effective per-Cell "
            f"tile {int(max_tile)}",
            mapping=dict(mapping),
        )
    if int(env.get("cells_in_flight", 0)) != 1:
        raise zsched.ActivePlanRejected(
            renv.REASON_REFUSE_HOST_RAM,
            "final per-Cell plan hash requires the single-owner K=1 contract",
            mapping=dict(mapping),
        )
    if int(env.get("prefetch_units", -1)) != int(mapping["prefetch_units"]):
        raise zsched.ActivePlanRejected(
            renv.REASON_SHRINK_PREFETCH,
            "final per-Cell plan hash prefetch mismatch (envelope "
            f"{env.get('prefetch_units')!r} != mapping "
            f"{mapping['prefetch_units']!r})",
            mapping=dict(mapping),
        )
    if int(env.get("cache_workers_per_cell", -1)) != int(
        mapping["workers_per_unit"]
    ):
        raise zsched.ActivePlanRejected(
            renv.REASON_SHRINK_WORKERS,
            "final per-Cell plan hash cache-worker mismatch (envelope "
            f"{env.get('cache_workers_per_cell')!r} != mapping "
            f"{mapping['workers_per_unit']!r})",
            mapping=dict(mapping),
        )
    payload = {
        "schema": FINAL_PLAN_SCHEMA,
        "reference_plan_hash": str(reference_hash),
        "reference_candidate": reference_candidate.to_dict(),
        "reference_mapping": dict(reference_mapping),
        "mapping": dict(mapping),
        "percell_plan": plan.to_admission(),
        "effective_tile_px_by_cell": {
            str(cid): int(tile) for cid, tile in sorted(normalised.items())
        },
        "max_effective_tile_px": int(max_tile),
        "identity": identity.to_dict(),
        "product_envelope": env,
    }
    # The canonical encoder is the scheduler's (same bytes as the v1 hash).
    return "sha256:" + hashlib.sha256(
        zsched._canonical_active_json(payload)  # noqa: SLF001 - same package
    ).hexdigest()


# ---------------------------------------------------------------------------
# Durable per-Cell shrink caps (D3)
# ---------------------------------------------------------------------------


def validate_shrink_cap(
    plan: PerCellPlan, cell_id: Any, tile_px: Any
) -> int:
    """Strictly validate ONE shrink cap against the FROZEN per-Cell plan.

    The cap is a tile the Cell ACTUALLY executed. It must belong to the Cell's
    declared ladder and can never exceed its planned tile (a cap is a shrink,
    never a growth).
    """
    cid = str(cell_id)
    entry = plan.entry_for(cid)
    tile = _as_plain_int(tile_px, f"shrink_cap[{cid}]", minimum=1)
    if tile not in entry.declared_ladder:
        raise ValueError(
            f"shrink cap {tile} for Cell {cid!r} is outside its declared ladder"
        )
    if tile > int(entry.tile_px):
        raise ValueError(
            f"shrink cap {tile} for Cell {cid!r} exceeds its planned tile "
            f"{entry.tile_px}"
        )
    return int(tile)


def merge_shrink_cap(
    plan: PerCellPlan,
    caps: Mapping[str, Any] | None,
    cell_id: Any,
    tile_px: Any,
) -> dict[str, int]:
    """Return caps with ``cell_id`` lowered to ``tile_px`` (monotone minimum)."""
    out = {str(k): validate_shrink_cap(plan, k, v) for k, v in (caps or {}).items()}
    cid = str(cell_id)
    tile = validate_shrink_cap(plan, cid, tile_px)
    if cid in out:
        out[cid] = min(int(out[cid]), int(tile))
    else:
        out[cid] = int(tile)
    return out


def build_shrink_caps_document(
    *, run_id: Any, plan: PerCellPlan, caps: Mapping[str, Any]
) -> dict:
    normalized = {
        str(k): validate_shrink_cap(plan, k, v) for k, v in (caps or {}).items()
    }
    return {
        "schema": SHRINK_CAPS_SCHEMA,
        "run_id": str(run_id),
        "policy": str(plan.policy),
        "entries": [
            {"cell_id": cid, "tile_px": int(normalized[cid])}
            for cid in sorted(normalized)
        ],
    }


def parse_shrink_caps_document(
    raw: Any,
    *,
    plan: PerCellPlan,
    expected_run_id: Any = None,
    expected_cell_ids=None,
) -> dict[str, int]:
    """Strictly read the durable shrink-cap document (fail closed).

    ``expected_cell_ids`` is the pending/expected Cell set of THIS resume. A cap
    for an unknown Cell, a wrong run id, an unknown policy/schema, a non-integer
    or a cap outside the frozen ladder/planned tile is REFUSED; it is never
    ignored or clamped.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("shrink caps document must be an object")
    _strict_keys(
        raw, "shrink_caps", ("schema", "run_id", "policy", "entries")
    )
    if str(raw["schema"]) != SHRINK_CAPS_SCHEMA:
        raise ValueError(
            f"shrink caps schema {raw['schema']!r} != {SHRINK_CAPS_SCHEMA!r}"
        )
    if str(raw["policy"]) != str(plan.policy):
        raise ValueError(
            f"shrink caps policy {raw['policy']!r} != {plan.policy!r}"
        )
    if expected_run_id is not None and str(raw["run_id"]) != str(expected_run_id):
        raise ValueError(
            f"shrink caps run_id {raw['run_id']!r} != expected "
            f"{expected_run_id!r}"
        )
    entries_raw = raw["entries"]
    if not isinstance(entries_raw, list):
        raise ValueError("shrink_caps.entries must be a list")
    caps: dict[str, int] = {}
    for index, item in enumerate(entries_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"shrink_caps.entries[{index}] must be an object")
        _strict_keys(item, f"shrink_caps.entries[{index}]", ("cell_id", "tile_px"))
        cid = str(item["cell_id"])
        if not cid or cid in caps:
            raise ValueError(
                f"shrink_caps.entries[{index}] has empty/duplicate cell_id {cid!r}"
            )
        caps[cid] = validate_shrink_cap(plan, cid, item["tile_px"])
    if expected_cell_ids is not None:
        expected = {str(cid) for cid in expected_cell_ids}
        extra = sorted(set(caps) - expected)
        if extra:
            raise ValueError(
                f"shrink caps name Cell(s) outside this run: {extra}"
            )
    return caps
