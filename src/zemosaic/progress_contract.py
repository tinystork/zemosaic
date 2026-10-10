"""ZM-PROGRESS-CONTRACT-R29 — pure, deterministic progress contract (no Qt).

This module is the single authority for turning raw ``STAGE_PROGRESS`` stage
events into honest, monotonic GLOBAL progress and a stable ordered phase
presentation. It has NO heavy imports (no Qt, no astropy, no numpy) so it can be
imported by the worker, the GUI and the tests without pulling in the decode/
reproject stack.

Design rules (see the R29 mission):

* **Stable ids, never counters.** A stage id such as ``zegrid:per_cell_stack``
  is the machine identity. The legacy counter-embedded form
  ``zegrid:<phase>:<done>/<total>`` is NORMALIZED back to ``zegrid:<phase>`` for
  backward compatibility, but producers emit the stable form directly and the
  ``current``/``total`` counters travel separately.
* **Weights are UI-progress shares, NOT time/ETA.** They are normalized to 100
  and are intentionally conservative; they make no claim about duration. The
  calibrated ETA model is an R30 concern and must not be conflated with these
  shares.
* **Only explicit success reaches 100.** Stage events can drive global progress
  at most to :data:`PRE_TERMINAL_CAP_PCT` (strictly below 100). ``mark_success``
  is the ONLY way to reach 100. Failure/cancel never reaches 100.
* **Unknown stages never jump progress.** An unknown stage id keeps the prior
  global progress and is surfaced safely (``known=False``); its local
  ``current/total`` is never mapped to a global percent.
* **Roman ordinals / localized labels are presentation only.** Machine ids are
  never translated, and translated strings are never used as protocol keys.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stage-driven global progress is capped strictly below 100.0; only the explicit
# terminal success signal may set 100.0. This prevents a phase reaching its
# local 100% (or the whole plan finishing its writes) from being presented as a
# completed run before the worker actually reports success.
PRE_TERMINAL_CAP_PCT = 99.0

# Roman ordinal labels — PRESENTATION ONLY (never protocol keys).
ROMAN_ORDINALS = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")

# Localization keys (presentation; never machine ids).
ZEGRID_PHASE_SETUP = "zegrid_phase_setup"
ZEGRID_PHASE_LAYOUT = "zegrid_phase_layout"
ZEGRID_PHASE_GAUGE = "zegrid_phase_gauge"
ZEGRID_PHASE_PER_CELL_STACK = "zegrid_phase_per_cell_stack"
ZEGRID_PHASE_ASSEMBLY = "zegrid_phase_assembly"
ZEGRID_PHASE_FINALIZE = "zegrid_phase_finalize"
ZEGRID_PHASE_LABEL_FORMAT = "zegrid_phase_label_format"
ETA_ESTIMATION_IN_PROGRESS = "eta_estimation_in_progress"
# ZM-ETA-ALLMODES-R31: shared Roman phase label formatter for all modes.
PHASE_LABEL_FORMAT = "phase_label_format"


def roman_ordinal(position: int) -> str:
    """Return the Roman ordinal (1-based) for a presentation position.

    Positions beyond the known list fall back to the integer (still presentation
    only, never a machine id).
    """
    idx = int(position) - 1
    if 0 <= idx < len(ROMAN_ORDINALS):
        return ROMAN_ORDINALS[idx]
    return str(int(position))


# ---------------------------------------------------------------------------
# Stage descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageDescriptor:
    """An immutable stage in a mode plan.

    Attributes:
        id:          Stable machine id (never translated, never embeds counters).
        mode:        Owning mode plan name (``"legacy"`` | ``"sds"`` | ``"zegrid"``).
        ordinal:     1-based ordered position within the plan (presentation).
        display_key: Localization key for the human-readable operation label.
        position:    0-based UI order used for transition ordering.
        weight:      UI-progress share (sums to 100 across the plan; NOT ETA).
        aliases:     Legacy/alias ids that normalize to this canonical id.
    """

    id: str
    mode: str
    ordinal: int
    display_key: str
    position: int
    weight: float
    aliases: FrozenSet[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ModePlan:
    """An immutable, ordered plan of stages for one processing mode."""

    mode: str
    stages: Tuple[StageDescriptor, ...]
    aliases: Mapping[str, str] = field(default_factory=dict)

    # -- lookup -------------------------------------------------------------

    def stage(self, stage_id: str) -> Optional[StageDescriptor]:
        """Return the descriptor for a canonical id, or ``None`` if unknown."""
        for s in self.stages:
            if s.id == stage_id:
                return s
        return None

    def position(self, stage_id: str) -> Optional[int]:
        desc = self.stage(stage_id)
        return None if desc is None else desc.position

    def normalize_id(self, stage_id: str) -> str:
        """Normalize a producer stage id to a canonical id (never raises).

        Backward-compatible handling of the legacy counter-embedded ZeGrid form
        ``zegrid:<phase>:<done>/<total>`` (and the bare ``zegrid:<phase>`` form)
        collapses it to the stable ``zegrid:<phase>``. Legacy alias ids map to
        their canonical ids. Anything still unknown is returned unchanged so the
        caller can safely mark it ``known=False``.
        """
        sid = str(stage_id or "").strip()
        if not sid:
            return sid
        if sid.startswith("zegrid:"):
            parts = sid.split(":")
            # ``zegrid:<phase>:<done>/<total>`` -> drop the trailing counter token.
            if len(parts) >= 3 and _is_counter_token(parts[-1]):
                sid = ":".join(parts[:-1])
        return self.aliases.get(sid, sid)

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` if the plan is structurally invalid.

        Checks: unique stable ids, contiguous 0-based positions matching
        1-based ordinals, positive weights normalized to ~100, and aliases that
        never collide with a canonical id.
        """
        seen: set[str] = set()
        for i, s in enumerate(self.stages):
            if s.id in seen:
                raise ValueError(f"{self.mode}: duplicate stage id {s.id!r}")
            seen.add(s.id)
            if s.mode != self.mode:
                raise ValueError(f"{self.mode}: stage {s.id!r} mode mismatch {s.mode!r}")
            if s.position != i:
                raise ValueError(f"{self.mode}: stage {s.id!r} position {s.position} != {i}")
            if s.ordinal != i + 1:
                raise ValueError(f"{self.mode}: stage {s.id!r} ordinal {s.ordinal} != {i + 1}")
            if s.weight <= 0.0:
                raise ValueError(f"{self.mode}: stage {s.id!r} non-positive weight {s.weight}")
        total = sum(s.weight for s in self.stages)
        if abs(total - 100.0) > 1e-6:
            raise ValueError(f"{self.mode}: weights sum to {total}, expected 100.0")
        for alias, canonical in self.aliases.items():
            if canonical not in seen:
                raise ValueError(f"{self.mode}: alias {alias!r} -> unknown {canonical!r}")
            if alias in seen:
                raise ValueError(f"{self.mode}: alias {alias!r} collides with canonical id")


def _is_counter_token(token: str) -> bool:
    """True if ``token`` is a ``<int>/<int>`` counter suffix (never a phase)."""
    if "/" not in token:
        return False
    lhs, _, rhs = token.partition("/")
    try:
        int(lhs.strip())
        int(rhs.strip())
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

# Legacy (Classic) plan — mirrors the historical GUI ``_stage_order``/
# ``_stage_weights``/``_stage_aliases`` exactly so existing weighted-progress
# behavior is preserved. Ordinal/weights are UI shares only.
_LEGACY_STAGES = (
    StageDescriptor("phase1", "legacy", 1, "qt_stage_phase1", 0, 30.0,
                    frozenset({"phase1_scan"})),
    StageDescriptor("phase2", "legacy", 2, "qt_stage_phase2", 1, 5.0,
                    frozenset({"phase2_cluster"})),
    StageDescriptor("phase3", "legacy", 3, "qt_stage_phase3", 2, 35.0,
                    frozenset({"phase3_master_tiles"})),
    StageDescriptor("phase4", "legacy", 4, "qt_stage_phase4", 3, 5.0,
                    frozenset({"phase4_grid"})),
    StageDescriptor("phase4_5", "legacy", 5, "qt_stage_phase4_5", 4, 6.0),
    StageDescriptor("phase5", "legacy", 6, "qt_stage_phase5", 5, 9.0,
                    frozenset({"phase5_intertile", "phase5_incremental", "phase5_reproject"})),
    StageDescriptor("phase6", "legacy", 7, "qt_stage_phase6", 6, 8.0),
    StageDescriptor("phase7", "legacy", 8, "qt_stage_phase7", 7, 2.0),
)
_LEGACY_ALIASES = {
    "phase1_scan": "phase1",
    "phase2_cluster": "phase2",
    "phase3_master_tiles": "phase3",
    "phase4_grid": "phase4",
    "phase5_intertile": "phase5",
    "phase5_incremental": "phase5",
    "phase5_reproject": "phase5",
}
LEGACY_PLAN = ModePlan("legacy", _LEGACY_STAGES, _LEGACY_ALIASES)

# SDS plan (Mosaic-First / SupaDup) — 7 equal phases matching ``_sds_total_phases``.
# The GUI keeps its SDS specialization via an adapter; this plan exists so the
# contract describes the real SDS ordering for validation/completeness.
_SDS_NAMES = (
    ("sds_phase_1", "Preprocess"),
    ("sds_phase_2", "Cluster"),
    ("sds_phase_3", "MasterTiles"),
    ("sds_phase_4", "GlobalCoadd"),
    ("sds_phase_5", "Polish"),
    ("sds_phase_6", "Save"),
    ("sds_phase_7", "Cleanup"),
)
_SDS_WEIGHT = 100.0 / 7.0
SDS_PLAN = ModePlan(
    "sds",
    tuple(
        StageDescriptor(
            f"sds_phase_{n}", "sds", n, f"qt_stage_sds_phase{n}", n - 1, _SDS_WEIGHT
        )
        for n, (_key, _name) in enumerate(_SDS_NAMES, start=1)
    ),
)

# ZeGrid plan — the six honest phases of the ZeGrid run. ``cache_build`` is an
# alias of ``per_cell_stack`` (merged reality since R20): the cache build is
# fused into the per-cell batch and must NOT be presented as separate future
# work. Weights are conservative UI shares only (NOT duration claims).
_ZEGRID_STAGES = (
    StageDescriptor("zegrid:setup", "zegrid", 1, ZEGRID_PHASE_SETUP, 0, 5.0),
    StageDescriptor("zegrid:layout", "zegrid", 2, ZEGRID_PHASE_LAYOUT, 1, 15.0),
    StageDescriptor("zegrid:gauge", "zegrid", 3, ZEGRID_PHASE_GAUGE, 2, 25.0),
    StageDescriptor("zegrid:per_cell_stack", "zegrid", 4, ZEGRID_PHASE_PER_CELL_STACK, 3, 30.0,
                    frozenset({"zegrid:cache_build"})),
    StageDescriptor("zegrid:assembly", "zegrid", 5, ZEGRID_PHASE_ASSEMBLY, 4, 15.0),
    StageDescriptor("zegrid:finalize", "zegrid", 6, ZEGRID_PHASE_FINALIZE, 5, 10.0),
)
_ZEGRID_ALIASES = {"zegrid:cache_build": "zegrid:per_cell_stack"}
ZEGRID_PLAN = ModePlan("zegrid", _ZEGRID_STAGES, _ZEGRID_ALIASES)

_PLANS = (LEGACY_PLAN, SDS_PLAN, ZEGRID_PLAN)


def plan_for_mode(mode: str) -> ModePlan:
    """Return the plan for a mode name (``"legacy"`` | ``"sds"`` | ``"zegrid"``)."""
    for p in _PLANS:
        if p.mode == mode:
            return p
    raise ValueError(f"unknown mode {mode!r}")


def plan_for_stage_id(stage_id: str) -> ModePlan:
    """Choose a plan by stage-id prefix (ZeGrid ids are ``zegrid:*``)."""
    if str(stage_id or "").startswith("zegrid:"):
        return ZEGRID_PLAN
    return LEGACY_PLAN


def legacy_phase_id_to_stage(phase_id: str) -> Optional[str]:
    """Map a legacy ``PHASE_UPDATE:<id>`` id to a canonical stage id.

    ``"1".."7"`` -> ``phase1..phase7``; ``"4.5"`` / ``"4_5"`` -> ``phase4_5``.
    Returns ``None`` for unknown ids (caller keeps the safe raw fallback).
    """
    pid = str(phase_id or "").strip()
    mapping = {
        "1": "phase1", "2": "phase2", "3": "phase3", "4": "phase4",
        "4.5": "phase4_5", "4_5": "phase4_5",
        "5": "phase5", "6": "phase6", "7": "phase7",
    }
    return mapping.get(pid)


def sds_phase_id_to_stage(phase_id: object) -> Optional[str]:
    """Map an SDS numeric phase (``1..7``) to a canonical SDS stage id.

    ``1..7`` -> ``sds_phase_1..sds_phase_7``. Returns ``None`` otherwise (the
    caller keeps the safe raw fallback).
    """
    try:
        n = int(str(phase_id or "").strip())
    except (TypeError, ValueError):
        return None
    if 1 <= n <= 7:
        return f"sds_phase_{n}"
    return None


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageResult:
    """Deterministic result of feeding one stage event to the aggregator."""

    normalized_id: str
    known: bool
    local_fraction: float
    global_percent: float
    phase_id: Optional[str]
    ordinal: Optional[int]
    display_key: Optional[str]
    position: Optional[int]
    terminal_state: Optional[str]


class ProgressAggregator:
    """Deterministic, monotonic global-progress aggregator for one mode plan.

    Input is a stable stage id + ``current``/``total`` (plus an explicit terminal
    signal via :meth:`mark_success` / :meth:`mark_fail` / :meth:`mark_cancel`).
    Output is a :class:`StageResult` with the normalized id, the local fraction,
    the monotonic global percent and the phase metadata.

    Guarantees:

    * global percent is monotonic and in ``[0, PRE_TERMINAL_CAP_PCT]`` until an
      explicit terminal signal; only ``mark_success`` sets exactly 100;
    * a later stage marks prior plan stages complete (the transition is proof
      they finished) — but never inferred from translated text;
    * per-stage fractions are monotonic (duplicate/out-of-order events never
      regress a completed stage);
    * unknown stages keep the prior global progress (never a local->global jump);
    * :meth:`reset` fully clears run state (fractions, mode, terminal, timings).
    """

    def __init__(
        self,
        plan: ModePlan,
        *,
        clock: Optional[Callable[[], float]] = None,
        pre_terminal_cap_pct: float = PRE_TERMINAL_CAP_PCT,
    ) -> None:
        self._plan = plan
        self._clock = clock if clock is not None else time.monotonic
        self._cap = float(pre_terminal_cap_pct)
        self.reset()

    # -- state --------------------------------------------------------------

    def reset(self) -> None:
        self._fractions: Dict[str, float] = {}
        self._global_pct = 0.0
        self._terminal: Optional[str] = None
        self._current_id: Optional[str] = None
        self._current_position: int = -1
        self._last_event_at: Optional[float] = None

    @property
    def plan(self) -> ModePlan:
        return self._plan

    @property
    def global_percent(self) -> float:
        return self._global_pct

    @property
    def terminal(self) -> Optional[str]:
        return self._terminal

    @property
    def current_stage_id(self) -> Optional[str]:
        return self._current_id

    # -- events -------------------------------------------------------------

    def on_stage(self, stage_id: str, current: int, total: int) -> StageResult:
        normalized = self._plan.normalize_id(stage_id)
        desc = self._plan.stage(normalized)
        if desc is None:
            # Unknown stage: never map local current/total to global progress.
            return StageResult(
                normalized_id=normalized,
                known=False,
                local_fraction=0.0,
                global_percent=self._global_pct,
                phase_id=normalized,
                ordinal=None,
                display_key=None,
                position=None,
                terminal_state=self._terminal,
            )
        if self._terminal is not None:
            return self._result_for(desc, normalized)
        # A later stage is proof that every earlier plan stage finished.
        for prior in self._plan.stages:
            if prior.position < desc.position:
                self._fractions[prior.id] = 1.0
        cur = _to_int(current)
        tot = _to_int(total)
        if tot > 0:
            fraction = max(0.0, min(1.0, cur / float(tot)))
        else:
            # No measurable total: keep the stage floor (never fabricate).
            fraction = self._fractions.get(desc.id, 0.0)
        # Per-stage monotonic: duplicates / out-of-order events never regress.
        self._fractions[desc.id] = max(self._fractions.get(desc.id, 0.0), fraction)
        self._current_id = desc.id
        self._current_position = desc.position
        self._last_event_at = self._clock()
        self._global_pct = max(self._global_pct, self._compute_global())
        return self._result_for(desc, normalized)

    def mark_success(self) -> StageResult:
        # ZM-PROGRESS-CONTRACT-R29 F2: a terminal state is immutable. Once
        # fail/cancel is set, success must NOT override it (never reach 100 from
        # a nonterminal state). Only success from a nonterminal state reaches 100.
        if self._terminal is None:
            self._terminal = "success"
            self._global_pct = 100.0
        return self._terminal_result()

    def mark_fail(self) -> StageResult:
        if self._terminal is None:
            self._terminal = "fail"
        return self._terminal_result()

    def mark_cancel(self) -> StageResult:
        if self._terminal is None:
            self._terminal = "cancel"
        return self._terminal_result()

    # -- internals ----------------------------------------------------------

    def _compute_global(self) -> float:
        raw = 0.0
        for s in self._plan.stages:
            raw += s.weight * self._fractions.get(s.id, 0.0)
        return min(raw, self._cap)

    def _result_for(self, desc: StageDescriptor, normalized: str) -> StageResult:
        return StageResult(
            normalized_id=normalized,
            known=True,
            local_fraction=self._fractions.get(desc.id, 0.0),
            global_percent=self._global_pct,
            phase_id=desc.id,
            ordinal=desc.ordinal,
            display_key=desc.display_key,
            position=desc.position,
            terminal_state=self._terminal,
        )

    def _terminal_result(self) -> StageResult:
        desc = self._plan.stage(self._current_id) if self._current_id else None
        if desc is None:
            return StageResult(
                normalized_id=self._current_id or "",
                known=False,
                local_fraction=0.0,
                global_percent=self._global_pct,
                phase_id=self._current_id,
                ordinal=None,
                display_key=None,
                position=None,
                terminal_state=self._terminal,
            )
        return self._result_for(desc, self._current_id or desc.id)


def _to_int(value: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
