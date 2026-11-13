"""ZM-ZEGRID-RESILIENT-RUN-RESUME M4 — resume bootstrap (plan §6.4 / §6.6).

This module owns the IMPERATIVE ORDER of a restart:

    choose/open the run -> acquire the single-writer lock -> read the minimal
    identity/state -> validate inputs by CONTENT -> restore geometry/gauge/
    memberships and the storage policy -> admit ONLY the remaining cells.

It never plans from a ``0..K`` prefix: resume always operates on the sparse set
of validated ``cell_id`` values, and completeness is the exact predicate
``expected == COMPLETE + EMPTY_VALIDATED``.

Policy (V1):
* ``zegrid_resume_mode=never`` (default) — a NEW run is created and the old
  generations are never overwritten; the launch-time REUSE is disabled, not the
  safety saving of the new run.
* ``zegrid_resume_mode=auto`` — resume a compatible existing run. An
  incompatible run STOPS with an explicit reason (never a silent 10 h recompute).
  A complete run is a verified no-op. Multiple matching runs require an explicit
  selection (``zegrid_resume_run_id``), never "the last directory found".

Corruption policy: one corrupt cell checkpoint invalidates THAT cell only (its
derived outputs are invalidated with it); a corrupt global gauge/identity is a
refusal, never a best-effort recovery.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import run_store as zrs

RESUME_MODE_AUTO = "auto"
RESUME_MODE_NEVER = "never"
RESUME_MODES = (RESUME_MODE_AUTO, RESUME_MODE_NEVER)
RESUME_STATE_SCHEMA = "zegrid-run-resume-state-v1"
RESUME_STATE_VERSION = "1"
RESUME_STATE_FILENAME = "state.json"
# ZM-ZEGRID-ACTIVE-RESUME-PARITY: the admission inputs belong to the SCIENCE
# identity, so they are FROZEN with the run at creation and REUSED on resume.
# They are never re-derived from the current machine (free RAM changes the
# chosen tile between two launches of the SAME run).
ADMISSION_SCHEMA = "zegrid-run-admission-contract-v1"
ADMISSION_VERSION = "1"
ADMISSION_BACKENDS = ("cpu", "gpu")
ADMISSION_REMEDY = (
    "free enough memory so the frozen tile fits on this host again, or start "
    "a NEW run in a NEW output folder (a new run freezes its own admission)"
)
# ZM-ZEGRID-CLASS-DOWNGRADE-RESUME: the ONLY working-set class transition a
# resume may perform. It is a strict DOWNGRADE (strictly less RAM), recorded
# explicitly from/to with the RAM observation that forced it. Every other
# direction (ram_tile -> inmem, anything -> external_spool, external_spool ->
# anything) stays refused: those would change the admitted residency upward or
# move to a code path whose bit-parity has not been proven here.
WORKING_SET_TRANSITION_SCHEMA = "zegrid-working-set-transition-v1"
WORKING_SET_DOWNGRADES = (("inmem", "ram_tile"),)
# ZM-ZEGRID-CLASS-UPGRADE-RESUME: EXACTLY ONE additional direction is allowed,
# and only at the resume boundary: a run frozen on the single-owner external
# spool may leave it for the RAM streaming (ram_tile) class when a FRESH
# admission on the current host proves the SAME frozen geometry/halo/
# memberships/science/tile contract fits. It is not a downgrade and it is NOT
# part of ``WORKING_SET_DOWNGRADES`` (that predicate stays the strict
# shrink-direction source the engine and validator share).
WORKING_SET_UPGRADES = (("external_spool", "ram_tile"),)

def is_working_set_downgrade(from_class, to_class) -> bool:
    """True only for the ONE permitted class downgrade (single source).

    ``WORKING_SET_DOWNGRADES`` is the sole encoding of the permitted shrink
    set; the engine and the state validator both consult it through this
    predicate. The external_spool -> ram_tile upgrade is deliberately NOT in
    this set: it has its own predicate and its own fresh-admission gate.
    """
    return (str(from_class), str(to_class)) in WORKING_SET_DOWNGRADES


def is_working_set_upgrade(from_class, to_class) -> bool:
    """True only for the ONE permitted class upgrade (single source).

    ``WORKING_SET_UPGRADES`` is the sole encoding of the permitted upgrade
    set. The engine only reaches it at the resume boundary and only after a
    FRESH admission proved the target class fits on the frozen contract.
    """
    return (str(from_class), str(to_class)) in WORKING_SET_UPGRADES


def is_working_set_transition(from_class, to_class) -> bool:
    """True for any recorded class direction (downgrade OR upgrade)."""
    return is_working_set_downgrade(from_class, to_class) or is_working_set_upgrade(
        from_class, to_class
    )


WORKING_SET_TRANSITION_REQUIRED = (
    "schema",
    "from_class",
    "to_class",
    "tile_px",
    "column_batch",
    "reason",
    "ram_observation",
    "plan_generation",
    "identity_sha256",
    "identity_unchanged",
    "recorded_unix_ns",
)

# ZM-ZEGRID-FROZEN-BATCH-SHRINK: the ONLY column-batch transition a resume may
# perform. It is strictly SHRINK-ONLY (``to_batch < from_batch``), never a
# growth and never a class/tile change, and it is recorded from/to with the
# EXACT host peak and phase budget that forced it. The batch is a staging
# parameter (bit-identical payloads across batch sizes proven on real data);
# the scientific identity is never touched.
COLUMN_BATCH_TRANSITION_SCHEMA = "zegrid-column-batch-transition-v1"
COLUMN_BATCH_TRANSITION_REQUIRED = (
    "schema",
    "from_batch",
    "to_batch",
    "frozen_batch",
    "required_peak_bytes",
    "admitted_peak_bytes",
    "phase_budget_bytes",
    "reason",
    "plan_generation",
    "identity_sha256",
    "identity_unchanged",
    "recorded_unix_ns",
)


class ResumeError(RuntimeError):
    """Base class for every explicit resume refusal (never silent)."""


class ResumeModeError(ResumeError):
    pass


class ResumeSelectionRequired(ResumeError):
    def __init__(self, message: str, *, candidates: Sequence[Mapping[str, Any]]):
        super().__init__(message)
        self.candidates = [dict(item) for item in candidates]


class ResumeNotFound(ResumeError):
    pass


class ResumeIncompatible(ResumeError):
    def __init__(self, message: str, *, reasons: Sequence[Mapping[str, Any]] = ()):
        super().__init__(message)
        self.reasons = [dict(item) for item in reasons]


class ResumeStateError(ResumeError):
    pass


class ResumeCorruptGlobal(ResumeError):
    """run.json / global identity / gauge is corrupt: refuse, do not guess."""


class ResumeLocked(ResumeError):
    pass


class ResumeSchedulerUnsupported(ResumeError):
    """Durable resume requested while the effective scheduler is not active.

    The durable store, the result registry and the cooperative stop channel are
    created ONLY by the ``active`` scheduler. A ``legacy``/``observe`` run can
    therefore never consume a durable resume request: it is refused before any
    store directory or lock is opened or created (fail-closed, no fallback).
    """


class ResumeFrozenAdmissionUnavailable(ResumeError):
    """The run's FROZEN admission contract cannot be honoured on this host.

    Raised instead of shrinking the tile, switching the backend, re-deriving
    the admission from current RAM or silently starting a fresh run. The
    attributes are structured so a caller/log can name the exact deficit.
    """

    def __init__(
        self,
        message: str,
        *,
        frozen_backend: str,
        frozen_tile_px: int,
        required_bytes: int | None = None,
        available_bytes: int | None = None,
        deficit_bytes: int | None = None,
        detail: str | None = None,
    ):
        super().__init__(message)
        self.frozen_backend = str(frozen_backend)
        self.frozen_tile_px = int(frozen_tile_px)
        self.required_bytes = (
            None if required_bytes is None else int(required_bytes)
        )
        self.available_bytes = (
            None if available_bytes is None else int(available_bytes)
        )
        self.deficit_bytes = (
            None if deficit_bytes is None else int(deficit_bytes)
        )
        self.detail = detail


class ResumeLegacyBackendIdentity(ResumeError):
    """A store produced by the known CPU-frozen-identity/GPU-commit bug.

    The durable science identity says a CPU execution contract while immutable
    commit records uniformly declare ``backend=gpu``. The immutable identity is
    never rewritten and the commits are never modified. Recovery would require
    the COMPLETE effective GPU contract (backend AND execution tile) to be
    provable from durable evidence; it is not, so the resume refuses with an
    explicit ``legacy_inconsistent_backend_identity`` diagnostic.
    """

    reason_code = "legacy_inconsistent_backend_identity"

    def __init__(
        self,
        message: str,
        *,
        frozen_backend: str,
        frozen_tile_px: int,
        commit_backends: Sequence[str] = (),
        commit_count: int = 0,
        detail: str | None = None,
        reason_code: str | None = None,
    ):
        super().__init__(message)
        self.frozen_backend = str(frozen_backend)
        self.frozen_tile_px = int(frozen_tile_px)
        self.commit_backends = [str(v) for v in commit_backends]
        self.commit_count = int(commit_count)
        self.detail = detail
        if reason_code is not None:
            self.reason_code = str(reason_code)


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------


def normalize_resume_mode(value: Any) -> str:
    """Strictly normalise ``zegrid_resume_mode`` (default ``never``)."""
    if value is None:
        return RESUME_MODE_NEVER
    if isinstance(value, bool):
        raise ResumeModeError(
            "zegrid_resume_mode must be 'auto' or 'never', not a boolean"
        )
    token = str(value).strip().lower()
    if token in ("", RESUME_MODE_NEVER):
        return RESUME_MODE_NEVER
    if token == RESUME_MODE_AUTO:
        return RESUME_MODE_AUTO
    raise ResumeModeError(
        "zegrid_resume_mode must be 'auto' or 'never', "
        f"got {value!r} (no force, no coercion)"
    )


# ---------------------------------------------------------------------------
# Discovery and explicit selection
# ---------------------------------------------------------------------------


def _read_candidate_identity(root: Path) -> dict:
    path = root / "run.json"
    try:
        raw = json.loads(path.read_bytes().decode("utf-8"))
    except Exception as exc:
        raise ResumeCorruptGlobal(
            f"run identity at {path} is not readable JSON: {exc}"
        ) from exc
    try:
        identity = zrs.validate_run_identity(raw)
    except Exception as exc:
        raise ResumeCorruptGlobal(
            f"run identity at {path} is invalid: {type(exc).__name__}: {exc}"
        ) from exc
    return identity


def _candidate(root: Path, *, root_kind: str) -> dict:
    entry: dict[str, Any] = {
        "path": str(root),
        "kind": root_kind,
        "run_id": None,
        "identity_sha256": None,
        "created_unix_ns": None,
        "identity_error": None,
    }
    try:
        identity = _read_candidate_identity(root)
    except ResumeError as exc:
        entry["identity_error"] = f"{type(exc).__name__}: {exc}"
        return entry
    entry["run_id"] = str(identity["run_id"])
    entry["identity_sha256"] = zrs.identity_sha256(identity)
    entry["created_unix_ns"] = int(identity["created_unix_ns"])
    return entry


def discover_runs(store_root: str | os.PathLike[str]) -> list[dict]:
    """List run generations WITHOUT choosing one.

    A store directory may be either a legacy single run (``run.json`` directly
    inside) or a container of ``run_id`` generations (child directories). Both
    are reported; the caller must select explicitly when more than one exists.
    """
    root = Path(store_root)
    if not root.exists():
        return []
    if root.is_symlink():
        raise ResumeStateError(f"store root must not be a symlink: {root}")
    if not root.is_dir():
        raise ResumeStateError(f"store root is not a directory: {root}")
    out: list[dict] = []
    if (root / "run.json").is_file():
        out.append(_candidate(root, root_kind="root"))
    for child in sorted(root.iterdir()):
        if child.name in ("tmp", "commits", "objects"):
            continue
        if child.is_symlink():
            continue
        if not child.is_dir():
            continue
        if (child / "run.json").is_file():
            out.append(_candidate(child, root_kind="generation"))
    out.sort(key=lambda item: (
        int(item.get("created_unix_ns") or 0),
        str(item.get("run_id") or ""),
        str(item.get("path") or ""),
    ))
    return out


def select_run(
    store_root: str | os.PathLike[str],
    run_id: str | None = None,
) -> dict | None:
    """Select ONE run explicitly. Never "the last directory found"."""
    candidates = discover_runs(store_root)
    if run_id is not None:
        token = str(run_id).strip()
        if not token:
            raise ResumeSelectionRequired(
                "zegrid_resume_run_id is empty; select the run explicitly",
                candidates=candidates,
            )
        matches = [c for c in candidates if str(c.get("run_id")) == token]
        if not matches:
            raise ResumeNotFound(
                f"no run with run_id {token!r} exists under {store_root} "
                f"({len(candidates)} candidate(s))"
            )
        if len(matches) > 1:
            raise ResumeSelectionRequired(
                f"run_id {token!r} is ambiguous ({len(matches)} directories); "
                "remove the duplicate or pass an explicit generation path",
                candidates=matches,
            )
        return matches[0]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    raise ResumeSelectionRequired(
        f"{len(candidates)} runs exist under {store_root}; "
        "set zegrid_resume_run_id to choose one explicitly",
        candidates=candidates,
    )


# ---------------------------------------------------------------------------
# Frozen run state (expected cell set + effective policy)
# ---------------------------------------------------------------------------


def _req(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise ResumeStateError(f"{label} is missing required key {key!r}")
    return mapping[key]


def _strict_keys(
    mapping: Mapping[str, Any], label: str, required: Sequence[str],
    optional: Sequence[str] = (),
) -> None:
    if not isinstance(mapping, Mapping):
        raise ResumeStateError(f"{label} must be an object")
    allowed = set(required) | set(optional)
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ResumeStateError(f"{label} has unknown key(s): {unknown}")
    missing = sorted(set(required) - set(mapping))
    if missing:
        raise ResumeStateError(f"{label} is missing required key(s): {missing}")


def _as_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResumeStateError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ResumeStateError(f"{label} must be >= {minimum}")
    return int(value)


def validate_policy(policy: Any) -> dict:
    _strict_keys(
        policy, "policy",
        ("requested", "effective", "fallback_count", "transitions"),
        ("dense_refusal",),
    )
    requested = str(policy["requested"])
    effective = str(policy["effective"])
    if requested not in ("auto", "dense", "bounded"):
        raise ResumeStateError(
            f"policy.requested must be auto|dense|bounded, got {requested!r}"
        )
    if effective not in ("dense", "bounded"):
        raise ResumeStateError(
            f"policy.effective must be dense|bounded, got {effective!r}"
        )
    if requested == "bounded" and effective != "bounded":
        raise ResumeStateError("policy requested bounded but effective is not")
    if requested == "dense" and effective != "dense":
        raise ResumeStateError("policy requested dense but effective is not")
    fallback = _as_int(policy["fallback_count"], "policy.fallback_count", minimum=0)
    if fallback not in (0, 1):
        raise ResumeStateError("policy.fallback_count must be 0 or 1 (no re-dense)")
    transitions = policy["transitions"]
    if not isinstance(transitions, list):
        raise ResumeStateError("policy.transitions must be a list")
    norm_transitions = []
    for index, item in enumerate(transitions):
        if not isinstance(item, Mapping):
            raise ResumeStateError(f"policy.transitions[{index}] must be an object")
        norm_transitions.append({str(k): v for k, v in item.items()})
    dense_refusal = policy.get("dense_refusal")
    if dense_refusal is not None and not isinstance(dense_refusal, Mapping):
        raise ResumeStateError("policy.dense_refusal must be null or an object")
    return {
        "requested": requested,
        "effective": effective,
        "fallback_count": int(fallback),
        "transitions": norm_transitions,
        "dense_refusal": (
            None if dense_refusal is None else dict(dense_refusal)
        ),
    }


def validate_admission(admission: Any) -> dict:
    """Strictly validate the FROZEN admission contract of one run.

    The contract is written ONCE at run creation and never re-derived. It is
    deliberately small and schema-versioned: effective ``backend``, the
    ``tile_px`` that backend executes, and the ``parity_qualified`` flag that
    qualifies a backend/tile change. Unknown/extra keys are refused like every
    other durable section.
    """
    _strict_keys(
        admission, "admission",
        ("schema", "schema_version", "backend", "tile_px", "parity_qualified"),
        ("working_set",),
    )
    if admission["schema"] != ADMISSION_SCHEMA:
        raise ResumeStateError(
            f"admission schema {admission['schema']!r} != {ADMISSION_SCHEMA!r}"
        )
    if str(admission["schema_version"]) != ADMISSION_VERSION:
        raise ResumeStateError(
            f"admission version {admission['schema_version']!r} is not "
            f"supported ({ADMISSION_VERSION})"
        )
    backend = admission["backend"]
    if not isinstance(backend, str) or backend not in ADMISSION_BACKENDS:
        raise ResumeStateError(
            "admission.backend must be one of "
            f"{list(ADMISSION_BACKENDS)}, got {backend!r}"
        )
    tile_px = _as_int(admission["tile_px"], "admission.tile_px", minimum=1)
    parity = admission["parity_qualified"]
    if not isinstance(parity, bool):
        raise ResumeStateError("admission.parity_qualified must be a boolean")
    working_set = admission.get("working_set")
    norm_ws = None
    if working_set is not None:
        if not isinstance(working_set, Mapping):
            raise ResumeStateError("admission.working_set must be null or an object")
        _strict_keys(
            working_set, "admission.working_set",
            ("class", "tile_px", "column_batch"),
        )
        ws_class = str(working_set["class"])
        if ws_class not in ("inmem", "ram_tile", "external_spool"):
            raise ResumeStateError(
                "admission.working_set.class must be inmem|ram_tile|external_spool"
            )
        ws_tile = _as_int(
            working_set["tile_px"], "admission.working_set.tile_px", minimum=0
        )
        ws_batch = working_set["column_batch"]
        if ws_batch is not None:
            ws_batch = _as_int(
                ws_batch, "admission.working_set.column_batch", minimum=1
            )
        if ws_class != "inmem" and int(ws_tile) < 1:
            raise ResumeStateError(
                "admission.working_set.tile_px must be >= 1 for a tiled class"
            )
        if ws_class == "external_spool" and (ws_tile is None or ws_batch is None):
            raise ResumeStateError(
                "admission.working_set external_spool requires tile_px and column_batch"
            )
        norm_ws = {
            "class": ws_class,
            "tile_px": int(ws_tile),
            "column_batch": (None if ws_batch is None else int(ws_batch)),
        }
    return {
        "schema": ADMISSION_SCHEMA,
        "schema_version": ADMISSION_VERSION,
        "backend": str(backend),
        "tile_px": int(tile_px),
        "parity_qualified": bool(parity),
        **({} if norm_ws is None else {"working_set": norm_ws}),
    }


def validate_working_set_transition(raw: Any) -> dict:
    """Validate ONE recorded working-set class transition.

    The record is additive evidence, never a second source of scientific
    identity. It names the exact from/to classes, the derived streaming tile,
    the RAM observation that forced the change, the plan generation at which it
    happened and the (unchanged) identity digest.
    """
    _strict_keys(
        raw, "working_set_transitions[]", WORKING_SET_TRANSITION_REQUIRED
    )
    if raw["schema"] != WORKING_SET_TRANSITION_SCHEMA:
        raise ResumeStateError(
            f"working-set transition schema {raw['schema']!r} != "
            f"{WORKING_SET_TRANSITION_SCHEMA!r}"
        )
    from_class = str(raw["from_class"])
    to_class = str(raw["to_class"])
    if not is_working_set_transition(from_class, to_class):
        raise ResumeStateError(
            f"working-set transition {from_class!r} -> {to_class!r} is not an "
            "allowed class transition; only "
            + ", ".join(
                f"{a!r} -> {b!r}"
                for a, b in WORKING_SET_DOWNGRADES + WORKING_SET_UPGRADES
            )
            + " may be recorded"
        )
    tile_px = _as_int(
        raw["tile_px"], "working_set_transitions[].tile_px", minimum=1
    )
    batch = raw["column_batch"]
    if batch is not None:
        batch = _as_int(
            batch, "working_set_transitions[].column_batch", minimum=1
        )
    reason = str(raw["reason"])
    if not reason.strip():
        raise ResumeStateError(
            "working_set_transitions[].reason must not be empty"
        )
    observation = raw["ram_observation"]
    if not isinstance(observation, Mapping):
        raise ResumeStateError(
            "working_set_transitions[].ram_observation must be an object"
        )
    # ZM-ZEGRID-HARDENING-PASS (H4): a WORKING-SET UPGRADE carries the plan
    # aggregate it was admitted with and the budget that admitted it. A record
    # whose own numbers say the aggregate exceeded the budget describes an
    # admission that cannot have happened (a hand-edited state, or a future
    # caller that forgot the guard), so it is REFUSED here — this validator is
    # called by BOTH the recorder (before any write) and validate_state (on
    # load), so a contradictory pair can never enter or leave the store.
    _obs_aggregate = observation.get("plan_aggregate_bytes")
    _obs_budget = observation.get("plan_budget_bytes")
    if _obs_aggregate is not None and _obs_budget is not None:
        _aggregate = _as_int(
            _obs_aggregate,
            "working_set_transitions[].ram_observation.plan_aggregate_bytes",
            minimum=0,
        )
        _budget = _as_int(
            _obs_budget,
            "working_set_transitions[].ram_observation.plan_budget_bytes",
            minimum=0,
        )
        if int(_aggregate) > int(_budget):
            raise ResumeStateError(
                "working_set_transitions[].ram_observation is incoherent: "
                f"plan_aggregate_bytes {_aggregate} > plan_budget_bytes "
                f"{_budget}; an upgrade cannot have been admitted with an "
                "aggregate above its own budget"
            )
    identity_sha = str(raw["identity_sha256"])
    if not identity_sha:
        raise ResumeStateError(
            "working_set_transitions[].identity_sha256 must not be empty"
        )
    if raw["identity_unchanged"] is not True:
        raise ResumeStateError(
            "working_set_transitions[].identity_unchanged must be True: a "
            "class downgrade never changes the scientific identity"
        )
    return {
        "schema": WORKING_SET_TRANSITION_SCHEMA,
        "from_class": from_class,
        "to_class": to_class,
        "tile_px": int(tile_px),
        "column_batch": (None if batch is None else int(batch)),
        "reason": reason,
        "ram_observation": dict(observation),
        "plan_generation": _as_int(
            raw["plan_generation"],
            "working_set_transitions[].plan_generation",
            minimum=0,
        ),
        "identity_sha256": identity_sha,
        "identity_unchanged": True,
        "recorded_unix_ns": _as_int(
            raw["recorded_unix_ns"],
            "working_set_transitions[].recorded_unix_ns",
            minimum=0,
        ),
    }


def validate_column_batch_transition(raw: Any) -> dict:
    """Validate ONE recorded SHRINK-ONLY frozen column-batch transition.

    The record is additive evidence, never a second source of scientific
    identity. It names the exact from/to/frozen batches, the EXACT host peak
    the frozen value required, the phase-local budget that refused it, the
    peak the admitted batch actually needs, the reason and the (unchanged)
    identity digest. It enforces the same discipline as the class downgrade:
    strictly smaller, never a growth, never a class/tile change.
    """
    _strict_keys(
        raw, "column_batch_transitions[]", COLUMN_BATCH_TRANSITION_REQUIRED
    )
    if raw["schema"] != COLUMN_BATCH_TRANSITION_SCHEMA:
        raise ResumeStateError(
            f"column-batch transition schema {raw['schema']!r} != "
            f"{COLUMN_BATCH_TRANSITION_SCHEMA!r}"
        )
    from_batch = _as_int(
        raw["from_batch"], "column_batch_transitions[].from_batch", minimum=1
    )
    to_batch = _as_int(
        raw["to_batch"], "column_batch_transitions[].to_batch", minimum=1
    )
    frozen_batch = _as_int(
        raw["frozen_batch"],
        "column_batch_transitions[].frozen_batch",
        minimum=1,
    )
    if int(frozen_batch) != int(from_batch):
        raise ResumeStateError(
            "column_batch_transitions[].frozen_batch must equal from_batch: "
            "the transition describes the value it shrinks away from"
        )
    if int(to_batch) >= int(from_batch):
        raise ResumeStateError(
            "column_batch_transitions[].to_batch must be STRICTLY smaller "
            f"than from_batch ({to_batch} >= {from_batch}); the frozen batch "
            "is never grown or re-sized upward"
        )
    required_peak = _as_int(
        raw["required_peak_bytes"],
        "column_batch_transitions[].required_peak_bytes",
        minimum=1,
    )
    admitted_peak = _as_int(
        raw["admitted_peak_bytes"],
        "column_batch_transitions[].admitted_peak_bytes",
        minimum=1,
    )
    phase_budget = _as_int(
        raw["phase_budget_bytes"],
        "column_batch_transitions[].phase_budget_bytes",
        minimum=1,
    )
    if int(required_peak) <= int(phase_budget):
        raise ResumeStateError(
            "column_batch_transitions[].required_peak_bytes must EXCEED the "
            "phase budget; the frozen batch was shrunk although it fitted"
        )
    if int(admitted_peak) > int(phase_budget):
        raise ResumeStateError(
            "column_batch_transitions[].admitted_peak_bytes must fit the "
            "phase budget; a transition that does not fit admits nothing"
        )
    reason = str(raw["reason"])
    if not reason.strip():
        raise ResumeStateError(
            "column_batch_transitions[].reason must not be empty"
        )
    identity_sha = str(raw["identity_sha256"])
    if not identity_sha:
        raise ResumeStateError(
            "column_batch_transitions[].identity_sha256 must not be empty"
        )
    if raw["identity_unchanged"] is not True:
        raise ResumeStateError(
            "column_batch_transitions[].identity_unchanged must be True: the "
            "column batch is a staging parameter, never a science parameter"
        )
    return {
        "schema": COLUMN_BATCH_TRANSITION_SCHEMA,
        "from_batch": int(from_batch),
        "to_batch": int(to_batch),
        "frozen_batch": int(frozen_batch),
        "required_peak_bytes": int(required_peak),
        "admitted_peak_bytes": int(admitted_peak),
        "phase_budget_bytes": int(phase_budget),
        "reason": reason,
        "plan_generation": _as_int(
            raw["plan_generation"],
            "column_batch_transitions[].plan_generation",
            minimum=0,
        ),
        "identity_sha256": identity_sha,
        "identity_unchanged": True,
        "recorded_unix_ns": _as_int(
            raw["recorded_unix_ns"],
            "column_batch_transitions[].recorded_unix_ns",
            minimum=0,
        ),
    }


def build_admission(
    *,
    backend: Any,
    tile_px: Any,
    parity_qualified: Any,
    working_set: Any = None,
) -> dict:
    """Build and validate the frozen admission contract (never fabricated)."""
    raw = {
        "schema": ADMISSION_SCHEMA,
        "schema_version": ADMISSION_VERSION,
        "backend": backend,
        "tile_px": tile_px,
        "parity_qualified": parity_qualified,
    }
    if working_set is not None:
        raw["working_set"] = working_set
    return validate_admission(raw)


def validate_state(raw: Any) -> dict:
    _strict_keys(
        raw, "resume state",
        (
            "schema", "schema_version", "run_id", "created_unix_ns",
            "expected_cell_ids", "policy", "plan_generation", "finalized",
        ),
        ("gauge_sha256", "outputs", "note", "admission",
         "working_set_transitions", "column_batch_transitions"),
    )
    if raw["schema"] != RESUME_STATE_SCHEMA:
        raise ResumeStateError(
            f"resume state schema {raw['schema']!r} != {RESUME_STATE_SCHEMA!r}"
        )
    if str(raw["schema_version"]) != RESUME_STATE_VERSION:
        raise ResumeStateError(
            f"resume state version {raw['schema_version']!r} is not supported "
            f"({RESUME_STATE_VERSION})"
        )
    run_id = str(raw["run_id"])
    if not run_id:
        raise ResumeStateError("resume state.run_id must not be empty")
    created = _as_int(raw["created_unix_ns"], "resume state.created_unix_ns")
    ids = raw["expected_cell_ids"]
    if not isinstance(ids, list) or not ids:
        raise ResumeStateError("resume state.expected_cell_ids must be a non-empty list")
    cells: list[str] = []
    for index, value in enumerate(ids):
        if not isinstance(value, str) or not value:
            raise ResumeStateError(
                f"resume state.expected_cell_ids[{index}] must be a non-empty string"
            )
        cells.append(value)
    if len(set(cells)) != len(cells):
        raise ResumeStateError("resume state.expected_cell_ids has duplicates")
    finalized = raw["finalized"]
    if not isinstance(finalized, bool):
        raise ResumeStateError("resume state.finalized must be a boolean")
    outputs = raw.get("outputs")
    if outputs is not None and not isinstance(outputs, Mapping):
        raise ResumeStateError("resume state.outputs must be null or an object")
    normalised = {
        "schema": RESUME_STATE_SCHEMA,
        "schema_version": RESUME_STATE_VERSION,
        "run_id": run_id,
        "created_unix_ns": int(created),
        "expected_cell_ids": cells,
        "policy": validate_policy(_req(raw, "policy", "resume state")),
        "plan_generation": _as_int(
            raw["plan_generation"], "resume state.plan_generation", minimum=0
        ),
        "finalized": bool(finalized),
        "gauge_sha256": (
            None if raw.get("gauge_sha256") is None
            else str(raw["gauge_sha256"])
        ),
        "outputs": {} if outputs is None else dict(outputs),
        "note": None if raw.get("note") is None else str(raw["note"]),
    }
    # BACKWARD COMPATIBILITY: a store written before this section exists has
    # no ``admission`` key. It stays ABSENT (conservative: re-derive and refuse
    # on mismatch); nothing is fabricated on load and no ``null`` placeholder
    # is ever written for a store that genuinely has no frozen contract.
    if raw.get("admission") is not None:
        normalised["admission"] = validate_admission(raw["admission"])
    # ZM-ZEGRID-CLASS-DOWNGRADE-RESUME: the optional, append-only audit trail of
    # explicit working-set class transitions. Absent for every pre-existing
    # store (byte-compatible); never fabricated on load.
    if raw.get("working_set_transitions") is not None:
        transitions = raw["working_set_transitions"]
        if not isinstance(transitions, list):
            raise ResumeStateError(
                "resume state.working_set_transitions must be a list"
            )
        if len(transitions) > 1:
            raise ResumeStateError(
                "resume state.working_set_transitions has "
                f"{len(transitions)} entries; exactly one class transition per "
                "run is permitted (a second entry would silently describe a "
                "class the run never admitted, and a run may never record both "
                "a class upgrade and a class downgrade)"
            )
        normalised["working_set_transitions"] = [
            validate_working_set_transition(item) for item in transitions
        ]
        if normalised["working_set_transitions"]:
            record = normalised["working_set_transitions"][0]
            admission = normalised.get("admission")
            working_set = (
                None if admission is None else admission.get("working_set")
            )
            if working_set is None:
                raise ResumeStateError(
                    "resume state records a working-set downgrade but its "
                    "admission has no working-set section; the state is "
                    "internally inconsistent"
                )
            if str(working_set.get("class")) != str(record["to_class"]):
                raise ResumeStateError(
                    "resume state working-set transition moves to "
                    f"{record['to_class']!r} but admission.working_set.class is "
                    f"{working_set.get('class')!r}; the state would describe a "
                    "class the admission does not claim"
                )
            if int(working_set.get("tile_px") or 0) != int(record["tile_px"]):
                raise ResumeStateError(
                    "resume state working-set transition names tile_px="
                    f"{record['tile_px']} but admission.working_set.tile_px is "
                    f"{working_set.get('tile_px')!r}; the state would describe "
                    "a tile the admission does not claim"
                )
            if record["column_batch"] != working_set.get("column_batch"):
                raise ResumeStateError(
                    "resume state working-set transition names column_batch="
                    f"{record['column_batch']!r} but admission.working_set"
                    f".column_batch is {working_set.get('column_batch')!r}"
                )
    # ZM-ZEGRID-FROZEN-BATCH-SHRINK: the optional, append-only audit trail of
    # SHRINK-ONLY column-batch transitions. Absent for every pre-existing store
    # (byte-compatible); never fabricated on load. The chain must be strictly
    # decreasing and its LAST ``to_batch`` must be the batch the admission now
    # claims, otherwise the state would describe a batch the run never used.
    if raw.get("column_batch_transitions") is not None:
        transitions = raw["column_batch_transitions"]
        if not isinstance(transitions, list):
            raise ResumeStateError(
                "resume state.column_batch_transitions must be a list"
            )
        records = [
            validate_column_batch_transition(item) for item in transitions
        ]
        previous_to = None
        for index, record in enumerate(records):
            if previous_to is not None:
                if int(record["from_batch"]) != int(previous_to):
                    raise ResumeStateError(
                        "resume state.column_batch_transitions is not a "
                        f"contiguous shrink chain at index {index}: from_batch "
                        f"{record['from_batch']} != previous to_batch "
                        f"{previous_to}"
                    )
            previous_to = int(record["to_batch"])
        normalised["column_batch_transitions"] = records
        if records:
            admission = normalised.get("admission")
            working_set = (
                None if admission is None else admission.get("working_set")
            )
            if working_set is None:
                raise ResumeStateError(
                    "resume state records a column-batch shrink but its "
                    "admission has no working-set section; the state is "
                    "internally inconsistent"
                )
            # ZM-ZEGRID-CLASS-UPGRADE-RESUME: a run that was shrunk while
            # external and LATER upgraded to ram_tile keeps the batch history
            # as immutable evidence of its external era. The live admission now
            # carries no batch at all, so the historical chain is not compared
            # against it (and no further shrink is possible on the RAM class).
            upgraded = any(
                is_working_set_upgrade(
                    str(item["from_class"]), str(item["to_class"])
                )
                for item in normalised.get("working_set_transitions") or ()
            )
            if upgraded:
                if working_set.get("column_batch") is not None:
                    raise ResumeStateError(
                        "resume state records a class upgrade to the tiled RAM "
                        "class but admission.working_set.column_batch is not "
                        "null; the upgraded class carries no column batch"
                    )
            else:
                if str(working_set.get("class")) != "external_spool":
                    raise ResumeStateError(
                        "resume state records a column-batch shrink but "
                        "admission.working_set.class is "
                        f"{working_set.get('class')!r}; only the external_spool "
                        "class carries a column batch"
                    )
                if int(working_set.get("column_batch") or 0) != int(
                    records[-1]["to_batch"]
                ):
                    raise ResumeStateError(
                        "resume state column-batch transition ends at "
                        f"{records[-1]['to_batch']} but admission.working_set"
                        f".column_batch is {working_set.get('column_batch')!r}; "
                        "the state would describe a batch the run never admitted"
                    )
    return normalised


def state_path(run_root: str | os.PathLike[str]) -> Path:
    return Path(run_root) / RESUME_STATE_FILENAME


def read_state(run_root: str | os.PathLike[str]) -> dict:
    path = state_path(run_root)
    if not path.exists():
        raise ResumeStateError(
            f"run {run_root} has no frozen {RESUME_STATE_FILENAME}: this "
            "generation predates M4 resume state and cannot be resumed "
            "(never guessed from names/mtimes)"
        )
    try:
        raw = json.loads(path.read_bytes().decode("utf-8"))
    except Exception as exc:
        raise ResumeCorruptGlobal(
            f"resume state at {path} is not readable JSON: {exc}"
        ) from exc
    return validate_state(raw)


def write_state(
    run_root: str | os.PathLike[str],
    *,
    run_id: str,
    expected_cell_ids: Iterable[str],
    policy: Mapping[str, Any],
    plan_generation: int = 0,
    finalized: bool = False,
    gauge_sha256: str | None = None,
    outputs: Mapping[str, Any] | None = None,
    created_unix_ns: int | None = None,
    note: str | None = None,
    admission: Mapping[str, Any] | None = None,
    working_set_transitions: Iterable[Mapping[str, Any]] | None = None,
    column_batch_transitions: Iterable[Mapping[str, Any]] | None = None,
) -> dict:
    """Atomically persist the frozen expected set, policy and admission.

    ``admission`` is the schema-versioned FROZEN admission contract of the run
    (effective backend + ``tile_px`` + ``parity_qualified``). It is written at
    CREATION and preserved verbatim afterwards; ``None`` keeps a store that
    predates the section byte-compatible (the key is then omitted entirely, so
    such a store stays on the conservative re-derive path).
    """
    root = Path(run_root)
    cells = sorted({str(cid) for cid in expected_cell_ids})
    payload = {
        "schema": RESUME_STATE_SCHEMA,
        "schema_version": RESUME_STATE_VERSION,
        "run_id": str(run_id),
        "created_unix_ns": int(
            time.time_ns() if created_unix_ns is None else created_unix_ns
        ),
        "expected_cell_ids": cells,
        "policy": validate_policy(policy),
        "plan_generation": int(plan_generation),
        "finalized": bool(finalized),
        "gauge_sha256": gauge_sha256,
        "outputs": {} if outputs is None else dict(outputs),
        "note": note,
    }
    if admission is not None:
        payload["admission"] = validate_admission(admission)
    if working_set_transitions is not None:
        payload["working_set_transitions"] = [
            validate_working_set_transition(item)
            for item in working_set_transitions
        ]
    if column_batch_transitions is not None:
        payload["column_batch_transitions"] = [
            validate_column_batch_transition(item)
            for item in column_batch_transitions
        ]
    state = validate_state(payload)
    tmp_dir = root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    zrs.atomic_write_bytes(
        state_path(root), zrs.canonical_json(state), tmp_dir=tmp_dir
    )
    return state


def _durable_identity_sha256(run_root: str | os.PathLike[str]) -> str:
    """The store's OWN durable identity digest (never a caller claim).

    Reading ``run.json`` directly avoids re-opening (and re-locking) the store
    while the resume process already owns the single-writer lock.
    """
    path = Path(run_root) / "run.json"
    try:
        raw = json.loads(path.read_bytes().decode("utf-8"))
    except Exception as exc:
        raise ResumeStateError(
            f"cannot verify the durable run identity at {path}: {exc}"
        ) from exc
    try:
        identity = zrs.validate_run_identity(raw)
    except Exception as exc:
        raise ResumeStateError(
            f"the durable run identity at {path} is invalid: {exc}"
        ) from exc
    return zrs.identity_sha256(identity)


def _record_working_set_transition(
    run_root: str | os.PathLike[str],
    *,
    transition: Mapping[str, Any],
    identity_sha256: str,
    allowed: Sequence[tuple[str, str]],
    kind: str,
) -> dict:
    """Shared persistence for the ONE recorded class transition of a run.

    ``allowed`` is one of the two single-source direction sets
    (:data:`WORKING_SET_DOWNGRADES` / :data:`WORKING_SET_UPGRADES`) and
    ``kind`` is the human wording ("downgrade"/"upgrade") used in refusals.
    Both wrappers below therefore share the SAME invariants:

    * refuse any direction outside ``allowed``;
    * refuse a second transition (exactly one per run, so a run can never
      record both an upgrade and a downgrade);
    * preserve ``backend``/``tile_px``/``parity_qualified`` byte-for-byte;
    * rewrite only the durable working-set class/tile of the admission;
    * append the explicit record (from, to, reason, RAM observation, plan
      generation, identity digest, identity_unchanged) to the state.
    """
    state = read_state(run_root)
    admission = state.get("admission")
    if admission is None:
        raise ResumeStateError(
            f"cannot record a working-set {kind}: the run has no frozen "
            "admission"
        )
    existing = admission.get("working_set")
    if existing is None:
        raise ResumeStateError(
            f"cannot record a working-set {kind}: the frozen admission has "
            "no working-set section"
        )
    proposal = dict(transition)
    from_class = str(proposal.get("from_class"))
    to_class = str(proposal.get("to_class"))
    if (from_class, to_class) not in allowed:
        raise ResumeStateError(
            f"working-set transition {from_class!r} -> {to_class!r} is not "
            "permitted on resume"
        )
    if str(existing.get("class")) != from_class:
        raise ResumeStateError(
            f"cannot record a working-set {kind}: the frozen class "
            f"{existing.get('class')!r} is not the transition's from_class "
            f"{from_class!r}"
        )
    prior = state.get("working_set_transitions") or []
    if prior:
        raise ResumeStateError(
            "a working-set class transition is already recorded for this run; "
            "exactly one transition is permitted (a run may never record both "
            "a class upgrade and a class downgrade)"
        )
    durable_sha = _durable_identity_sha256(run_root)
    supplied_sha = str(identity_sha256)
    if supplied_sha != durable_sha:
        raise ResumeStateError(
            f"cannot record a working-set {kind}: the supplied "
            f"identity_sha256 {supplied_sha!r} does not match the store's "
            f"durable identity {durable_sha!r}; the record would carry false "
            "provenance"
        )
    record = {
        "schema": WORKING_SET_TRANSITION_SCHEMA,
        "from_class": from_class,
        "to_class": to_class,
        "tile_px": int(proposal["tile_px"]),
        "column_batch": (
            None
            if proposal.get("column_batch") is None
            else int(proposal["column_batch"])
        ),
        "reason": str(proposal["reason"]),
        "ram_observation": dict(proposal["ram_observation"]),
        "plan_generation": int(state["plan_generation"]),
        "identity_sha256": durable_sha,
        "identity_unchanged": True,
        "recorded_unix_ns": int(time.time_ns()),
    }
    if to_class == "ram_tile" and record["column_batch"] is not None:
        raise ResumeStateError(
            f"cannot record a working-set {kind}: the ram_tile class carries "
            "no column batch"
        )
    validate_working_set_transition(record)
    new_admission = build_admission(
        backend=admission["backend"],
        tile_px=admission["tile_px"],
        parity_qualified=admission["parity_qualified"],
        working_set={
            "class": to_class,
            "tile_px": int(record["tile_px"]),
            "column_batch": (
                None
                if record["column_batch"] is None
                else int(record["column_batch"])
            ),
        },
    )
    write_state(
        run_root,
        run_id=state["run_id"],
        expected_cell_ids=state["expected_cell_ids"],
        policy=state["policy"],
        plan_generation=int(state["plan_generation"]),
        finalized=bool(state["finalized"]),
        gauge_sha256=state.get("gauge_sha256"),
        outputs=state.get("outputs"),
        created_unix_ns=int(state["created_unix_ns"]),
        note=state.get("note"),
        admission=new_admission,
        working_set_transitions=list(prior) + [record],
        column_batch_transitions=state.get("column_batch_transitions"),
    )
    return record


def record_working_set_downgrade(
    run_root: str | os.PathLike[str],
    *,
    transition: Mapping[str, Any],
    identity_sha256: str,
) -> dict:
    """Persist EXACTLY ONE approved working-set class downgrade on resume.

    The caller has already proven (in the planner) that the frozen ``inmem``
    class no longer fits the observed budget while ``ram_tile`` does. Only
    ``inmem`` -> ``ram_tile`` is accepted; the scientific identity is never
    touched (the working-set class is not part of ``identity_fingerprints``).
    """
    return _record_working_set_transition(
        run_root,
        transition=transition,
        identity_sha256=identity_sha256,
        allowed=WORKING_SET_DOWNGRADES,
        kind="downgrade",
    )


def record_working_set_upgrade(
    run_root: str | os.PathLike[str],
    *,
    transition: Mapping[str, Any],
    identity_sha256: str,
) -> dict:
    """Persist EXACTLY ONE approved working-set class UPGRADE on resume.

    The caller has already proven (in the planner, under a FRESH admission on
    the current host) that the frozen ``external_spool`` class can leave the
    single-owner spool for ``ram_tile`` on the SAME frozen geometry, halo,
    memberships, science and tile contract. Only ``external_spool`` ->
    ``ram_tile`` is accepted. The record is additive evidence; ``backend``,
    ``tile_px`` and ``parity_qualified`` are preserved verbatim and the
    scientific identity is unchanged (the class is not part of
    ``identity_fingerprints``).
    """
    return _record_working_set_transition(
        run_root,
        transition=transition,
        identity_sha256=identity_sha256,
        allowed=WORKING_SET_UPGRADES,
        kind="upgrade",
    )


def record_column_batch_shrink(
    run_root: str | os.PathLike[str],
    *,
    transition: Mapping[str, Any],
    identity_sha256: str,
) -> dict:
    """Persist ONE explicit SHRINK-ONLY frozen column-batch transition.

    The caller has already proven (in the planner, with the SAME exact host
    accounting the ``frozen_batch_ram`` refusal uses) that the frozen batch no
    longer fits the phase-local budget while a strictly SMALLER one does. This
    function:

    * refuses anything except ``external_spool`` with a frozen column batch;
    * refuses a non-shrink (``to_batch >= from_batch``) or a from_batch that
      is not the batch the admission currently claims;
    * preserves ``backend``/``tile_px``/``parity_qualified`` byte-for-byte and
      keeps the class/tile untouched;
    * rewrites ONLY ``admission.working_set.column_batch`` to the admitted
      smaller value, so a later resume starts from the batch it actually uses;
    * appends the explicit record (from/to/frozen, required/admitted peak,
      phase budget, reason, plan generation, identity digest, unchanged flag)
      to the state.

    The scientific identity is never touched: the column batch is a staging
    parameter and is not part of ``identity_fingerprints``.
    """
    state = read_state(run_root)
    admission = state.get("admission")
    if admission is None:
        raise ResumeStateError(
            "cannot record a column-batch shrink: the run has no frozen "
            "admission"
        )
    existing = admission.get("working_set")
    if existing is None:
        raise ResumeStateError(
            "cannot record a column-batch shrink: the frozen admission has no "
            "working-set section"
        )
    if str(existing.get("class")) != "external_spool":
        raise ResumeStateError(
            "cannot record a column-batch shrink: the frozen class "
            f"{existing.get('class')!r} carries no column batch"
        )
    current_batch = existing.get("column_batch")
    if current_batch is None:
        raise ResumeStateError(
            "cannot record a column-batch shrink: the frozen external "
            "working set has no column_batch"
        )
    proposal = dict(transition)
    from_batch = int(proposal.get("from_batch") or 0)
    to_batch = int(proposal.get("to_batch") or 0)
    frozen_batch = int(proposal.get("frozen_batch") or 0)
    if int(current_batch) != from_batch:
        raise ResumeStateError(
            "cannot record a column-batch shrink: the frozen batch "
            f"{current_batch!r} is not the transition's from_batch "
            f"{from_batch!r}"
        )
    if from_batch < 1 or to_batch < 1 or frozen_batch != from_batch:
        raise ResumeStateError(
            "cannot record a column-batch shrink: malformed from/to/frozen "
            f"batch ({from_batch!r}, {to_batch!r}, {frozen_batch!r})"
        )
    if to_batch >= from_batch:
        raise ResumeStateError(
            "cannot record a column-batch shrink: the frozen batch is never "
            f"grown or kept ({to_batch} >= {from_batch})"
        )
    required_peak = int(proposal.get("required_peak_bytes") or 0)
    admitted_peak = int(proposal.get("admitted_peak_bytes") or 0)
    phase_budget = int(proposal.get("phase_budget_bytes") or 0)
    if required_peak < 1 or admitted_peak < 1 or phase_budget < 1:
        raise ResumeStateError(
            "cannot record a column-batch shrink: the peak/budget accounting "
            "is missing or non-positive"
        )
    if admitted_peak > phase_budget or required_peak <= phase_budget:
        raise ResumeStateError(
            "cannot record a column-batch shrink: required_peak must exceed "
            "the phase budget and the admitted peak must fit it "
            f"(required={required_peak}, admitted={admitted_peak}, "
            f"budget={phase_budget})"
        )
    reason = str(proposal.get("reason") or "")
    if not reason.strip():
        raise ResumeStateError(
            "cannot record a column-batch shrink: reason must not be empty"
        )
    durable_sha = _durable_identity_sha256(run_root)
    supplied_sha = str(identity_sha256)
    if supplied_sha != durable_sha:
        raise ResumeStateError(
            "cannot record a column-batch shrink: the supplied identity_sha256 "
            f"{supplied_sha!r} does not match the store's durable identity "
            f"{durable_sha!r}; the record would carry false provenance"
        )
    prior = list(state.get("column_batch_transitions") or [])
    if prior:
        if int(prior[-1]["to_batch"]) != from_batch:
            raise ResumeStateError(
                "cannot record a column-batch shrink: the last recorded "
                f"to_batch {prior[-1]['to_batch']!r} is not the frozen batch "
                f"{from_batch!r}; the shrink chain is not contiguous"
            )
    record = {
        "schema": COLUMN_BATCH_TRANSITION_SCHEMA,
        "from_batch": int(from_batch),
        "to_batch": int(to_batch),
        "frozen_batch": int(frozen_batch),
        "required_peak_bytes": int(required_peak),
        "admitted_peak_bytes": int(admitted_peak),
        "phase_budget_bytes": int(phase_budget),
        "reason": reason,
        "plan_generation": int(state["plan_generation"]),
        "identity_sha256": durable_sha,
        "identity_unchanged": True,
        "recorded_unix_ns": int(time.time_ns()),
    }
    validate_column_batch_transition(record)
    new_admission = build_admission(
        backend=admission["backend"],
        tile_px=admission["tile_px"],
        parity_qualified=admission["parity_qualified"],
        working_set={
            "class": str(existing["class"]),
            "tile_px": int(existing["tile_px"]),
            "column_batch": int(to_batch),
        },
    )
    write_state(
        run_root,
        run_id=state["run_id"],
        expected_cell_ids=state["expected_cell_ids"],
        policy=state["policy"],
        plan_generation=int(state["plan_generation"]),
        finalized=bool(state["finalized"]),
        gauge_sha256=state.get("gauge_sha256"),
        outputs=state.get("outputs"),
        created_unix_ns=int(state["created_unix_ns"]),
        note=state.get("note"),
        admission=new_admission,
        working_set_transitions=state.get("working_set_transitions"),
        column_batch_transitions=prior + [record],
    )
    return record


def attach_working_set(
    run_root: str | os.PathLike[str],
    working_set: Mapping[str, Any],
) -> dict:
    """Append the selected working-set class to the FROZEN admission (M4).

    Called at the FIRST state write AFTER selection. It never rewrites the
    frozen backend/tile/parity values: it only adds the ``working_set`` section
    (or verifies the exact same section on a resume). A conflicting section is
    refused; the caller must never silently re-derive the class.
    """
    state = read_state(run_root)
    admission = state.get("admission")
    if admission is None:
        raise ResumeStateError(
            "cannot attach a working set: the run has no frozen admission"
        )
    candidate = build_admission(
        backend=admission["backend"],
        tile_px=admission["tile_px"],
        parity_qualified=admission["parity_qualified"],
        working_set=working_set,
    )["working_set"]
    existing = admission.get("working_set")
    if existing is not None:
        if existing != candidate:
            raise ResumeStateError(
                "frozen working_set "
                f"{existing!r} cannot be replaced by {candidate!r}; the run "
                "identity is not silently changed"
            )
        return state
    new_admission = build_admission(
        backend=admission["backend"],
        tile_px=admission["tile_px"],
        parity_qualified=admission["parity_qualified"],
        working_set=candidate,
    )
    return write_state(
        run_root,
        run_id=state["run_id"],
        expected_cell_ids=state["expected_cell_ids"],
        policy=state["policy"],
        plan_generation=int(state["plan_generation"]),
        finalized=bool(state["finalized"]),
        gauge_sha256=state.get("gauge_sha256"),
        outputs=state.get("outputs"),
        created_unix_ns=int(state["created_unix_ns"]),
        note=state.get("note"),
        admission=new_admission,
        working_set_transitions=state.get("working_set_transitions"),
        column_batch_transitions=state.get("column_batch_transitions"),
    )


def update_policy(
    run_root: str | os.PathLike[str],
    policy: Mapping[str, Any],
    *,
    plan_generation: int | None = None,
    finalized: bool | None = None,
) -> dict:
    """Rewrite only the mutable policy/generation fields of an existing state."""
    state = read_state(run_root)
    state["policy"] = validate_policy(policy)
    if plan_generation is not None:
        state["plan_generation"] = int(plan_generation)
    if finalized is not None:
        state["finalized"] = bool(finalized)
    return write_state(
        run_root,
        run_id=state["run_id"],
        expected_cell_ids=state["expected_cell_ids"],
        policy=state["policy"],
        plan_generation=state["plan_generation"],
        finalized=state["finalized"],
        gauge_sha256=state["gauge_sha256"],
        outputs=state["outputs"],
        created_unix_ns=state["created_unix_ns"],
        note=state["note"],
        # The admission contract is FROZEN: a policy rewrite never touches it.
        admission=state.get("admission"),
        # ... and the explicit class-transition audit trail is never dropped.
        working_set_transitions=state.get("working_set_transitions"),
        # ... neither is the explicit column-batch shrink audit trail.
        column_batch_transitions=state.get("column_batch_transitions"),
    )


def policy_from_snapshot(snapshot: Mapping[str, Any]) -> dict:
    """Convert the mode module's storage-policy snapshot to a resume policy."""
    return validate_policy(
        {
            "requested": snapshot.get("requested", "auto"),
            "effective": snapshot.get("effective", "dense"),
            "fallback_count": int(snapshot.get("fallback_count", 0) or 0),
            "transitions": list(snapshot.get("transitions") or []),
            "dense_refusal": snapshot.get("dense_refusal"),
        }
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _stat_token(path: Path) -> tuple[int, int]:
    st = path.stat()
    return int(st.st_size), int(getattr(st, "st_mtime_ns", 0))


def detect_legacy_backend_identity(store: zrs.RunCellStore) -> dict | None:
    """Detect pre-fix durable backend inconsistencies (fail-closed diagnosis).

    ZM-ACTIVE-RESUME-FROZEN-IDENTITY (REWORK-1): a durable commit is now
    fail-closed, so every newly written commit matches the frozen identity. A
    store written by an older build can still hold commits whose backend is
    ``unknown``/invalid or disagrees with the frozen identity. Such a store is
    INTERNALLY INCONSISTENT and must be refused explicitly; the immutable
    identity and every commit are never rewritten.

    Returns ``None`` for a consistent store, or a diagnostic dict describing the
    first class of inconsistency. Never mutates anything.

    * ``legacy_unqualified_backend_identity`` — a commit backend is not a valid
      execution token (``unknown``, empty, malformed, another token).
    * ``legacy_inconsistent_backend_identity`` — every token is valid but at
      least one disagrees with the frozen backend (the known real CPU-identity /
      GPU-commit shape is this class).
    """
    contract = store.identity["science_contract"]["backend"]
    frozen_backend = str(contract["backend"])
    backends: list[str] = []
    for record in store.iter_commit_records():
        backends.append(str(record.backend))
    if not backends:
        return None

    invalid = sorted({b for b in backends if b not in ("cpu", "gpu")})
    mismatched = sorted(
        {b for b in backends if b in ("cpu", "gpu") and b != frozen_backend}
    )
    if not invalid and not mismatched:
        return None

    if invalid:
        reason_code = "legacy_unqualified_backend_identity"
        detail = (
            "one or more immutable commit records do not declare a valid "
            f"execution backend {sorted(set(backends))}; the frozen identity "
            f"declares backend={frozen_backend!r}. The complete effective "
            "execution contract is not provable from durable evidence, so "
            "this run stays forensic-only and is never silently migrated"
        )
    else:
        reason_code = "legacy_inconsistent_backend_identity"
        detail = (
            "the frozen science identity declares "
            f"backend={frozen_backend!r} but immutable commit records declare "
            f"backend(s) {sorted(set(backends))}; the complete effective "
            "execution contract (backend AND execution tile) is not provable "
            "from durable evidence, so this run stays forensic-only and is "
            "never silently migrated"
        )
    return {
        "reason_code": reason_code,
        "frozen_backend": frozen_backend,
        "frozen_tile_px": int(contract["tile_px"]),
        "parity_qualified": bool(contract["parity_qualified"]),
        "commit_count": int(len(backends)),
        "commit_backends": sorted(set(backends)),
        "invalid_backends": invalid,
        "mismatched_backends": mismatched,
        "recoverable": False,
        "detail": detail,
    }


class ResumeBootstrap:
    """An opened, locked, validated view of ONE resumable run."""

    def __init__(self, store: zrs.RunCellStore, state: dict, run_root: Path):
        self.store = store
        self.state = state
        self.run_root = Path(run_root)
        self._closed = False

    # -- construction ----------------------------------------------------
    @classmethod
    def open(
        cls,
        store_root: str | os.PathLike[str],
        *,
        run_id: str | None = None,
        budget: zrs.DiskBudget | None = None,
        session_id: str | None = None,
        fault_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
        free_bytes_fn: Callable[[Path], int] | None = None,
    ) -> "ResumeBootstrap | None":
        """Select + lock + read identity/state. Returns None when no run exists."""
        candidate = select_run(store_root, run_id)
        if candidate is None:
            return None
        if candidate.get("identity_error"):
            raise ResumeCorruptGlobal(
                f"run identity is corrupt at {candidate['path']}: "
                f"{candidate['identity_error']}"
            )
        root = Path(candidate["path"])
        try:
            store = zrs.RunCellStore.open(
                root,
                identity=None,
                budget=budget,
                session_id=session_id,
                fault_hook=fault_hook,
                free_bytes_fn=free_bytes_fn,
            )
        except zrs.StoreLockError as exc:
            raise ResumeLocked(str(exc)) from exc
        except (zrs.StoreFormatError, zrs.StoreIntegrityError, zrs.StorePathError) as exc:
            raise ResumeCorruptGlobal(
                f"global run state is corrupt at {root}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            state = read_state(root)
            if state["run_id"] != store.run_id:
                raise ResumeCorruptGlobal(
                    f"resume state run_id {state['run_id']!r} != store run_id "
                    f"{store.run_id!r}"
                )
            # The frozen admission must be the run's OWN contract. A section
            # that disagrees with the stored science identity is corruption:
            # refuse instead of executing a fabricated/mixed admission.
            frozen = state.get("admission")
            if frozen is not None:
                stored_backend = store.identity["science_contract"]["backend"]
                observed = (
                    str(frozen["backend"]),
                    int(frozen["tile_px"]),
                    bool(frozen["parity_qualified"]),
                )
                expected = (
                    str(stored_backend["backend"]),
                    int(stored_backend["tile_px"]),
                    bool(stored_backend["parity_qualified"]),
                )
                if observed != expected:
                    raise ResumeCorruptGlobal(
                        "resume state admission disagrees with the stored run "
                        f"identity: admission backend={observed[0]!r} "
                        f"tile_px={observed[1]} parity_qualified={observed[2]} "
                        f"!= identity backend={expected[0]!r} "
                        f"tile_px={expected[1]} parity_qualified={expected[2]} "
                        "(never a silent re-derivation)"
                    )
            # ZM-ACTIVE-RESUME-FROZEN-IDENTITY: a store produced by the known
            # CPU-identity/GPU-commit bug is refused with an explicit diagnostic.
            # Immutable identity and every commit are preserved untouched.
            legacy = detect_legacy_backend_identity(store)
            if legacy is not None:
                raise ResumeLegacyBackendIdentity(
                    "resume refused: "
                    f"{legacy['reason_code']} — the frozen identity declares "
                    f"backend={legacy['frozen_backend']!r} tile_px="
                    f"{legacy['frozen_tile_px']} while {legacy['commit_count']} "
                    f"immutable commit(s) declare {legacy['commit_backends']} "
                    f"(invalid={legacy['invalid_backends']}, "
                    f"mismatched={legacy['mismatched_backends']}); the complete "
                    "effective execution contract is not provable from "
                    "durable evidence, so this run is never silently migrated. "
                    "The immutable identity and every commit are preserved as "
                    "forensic evidence; start a NEW run in a NEW output folder "
                    "to resume the science.",
                    frozen_backend=legacy["frozen_backend"],
                    frozen_tile_px=legacy["frozen_tile_px"],
                    commit_backends=legacy["commit_backends"],
                    commit_count=legacy["commit_count"],
                    detail=legacy["detail"],
                    reason_code=legacy["reason_code"],
                )
        except Exception:
            store.close()
            raise
        return cls(store, state, root)

    # -- lifecycle -------------------------------------------------------
    def close(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        return bool(self.store.close())

    @property
    def identity(self) -> dict:
        return self.store.identity

    @property
    def policy(self) -> dict:
        return dict(self.state["policy"])

    @property
    def expected_cell_ids(self) -> list[str]:
        return list(self.state["expected_cell_ids"])

    def retained_counts(self) -> dict:
        """Diagnostic counts from the durable commit index (never a re-plan).

        Used by a frozen-capacity refusal so the operator sees what would have
        been retained: how many expected Cells already have a durable commit and
        how many would still be pending. Read-only and tolerant of an isolated
        corrupt attempt (it simply does not count as complete).
        """
        complete = 0
        for cell_id in self.state["expected_cell_ids"]:
            try:
                attempts = self.store.attempts(cell_id)
            except zrs.RunStoreError:
                attempts = []
            for attempt in attempts:
                try:
                    record = self.store.commit_record(cell_id, attempt)
                except zrs.RunStoreError:
                    continue
                if str(record.status) in (
                    zrs.STATUS_COMPLETE,
                    zrs.STATUS_EMPTY_VALIDATED,
                ):
                    complete += 1
                    break
        total = int(len(self.state["expected_cell_ids"]))
        return {
            "expected": total,
            "complete": int(complete),
            "pending": int(total - complete),
        }

    @property
    def admission(self) -> dict | None:
        """The FROZEN admission contract, or ``None`` for a pre-parity store."""
        value = self.state.get("admission")
        return None if value is None else dict(value)

    @property
    def finalized(self) -> bool:
        return bool(self.state["finalized"])

    # -- validation ------------------------------------------------------
    def verify_identity(self, candidate_identity: Mapping[str, Any]) -> dict:
        """Validate the NEW process' identity against the stored run.

        Content is identity; name/mtime never is. RAM/workers/prefetch changes
        are runtime-only and never invalidate the run.
        """
        verdict = zrs.check_identity_compatibility(
            self.store.identity, candidate_identity
        )
        if not verdict["compatible"]:
            raise ResumeIncompatible(
                "resume refused: the new process does not match the stored "
                "run identity — "
                f"{zrs.describe_identity_mismatch(verdict['reasons'])}",
                reasons=verdict["reasons"],
            )
        return verdict

    def verify_inputs(
        self,
        source_paths: Mapping[str, str | os.PathLike[str]],
        *,
        hasher: Callable[[str], tuple[str, int]] | None = None,
    ) -> dict:
        """Re-hash every input by CONTENT and compare the frozen rows.

        A dataset relocation is NOT a change (paths are provenance only). A
        changed byte is a refusal. A file that changes while it is hashed
        invalidates the attempt rather than being accepted on a stale digest.
        """
        frame_rows = self.store.identity["science_contract"]["inputs"]["frames"]
        hash_fn = hasher or (lambda path: zrs.content_sha256_file(path))
        checked: list[dict] = []
        reasons: list[dict] = []
        seen: set[str] = set()
        for row in frame_rows:
            frame_id = str(row["frame_id"])
            seen.add(frame_id)
            source = source_paths.get(frame_id)
            if source is None:
                reasons.append(
                    {
                        "frame_id": frame_id,
                        "reason": "input is absent from the new selection "
                                  "(the frozen inventory may never be re-discovered)",
                    }
                )
                continue
            path = Path(source)
            try:
                before = _stat_token(path)
                digest, nbytes = hash_fn(str(path))
                after = _stat_token(path)
            except OSError as exc:
                reasons.append(
                    {"frame_id": frame_id, "reason": f"unreadable: {exc}"}
                )
                continue
            if before != after:
                reasons.append(
                    {
                        "frame_id": frame_id,
                        "reason": "input changed while it was hashed; the "
                                  "attempt is invalidated (never a stale digest)",
                    }
                )
                continue
            if digest != str(row["content_sha256"]):
                reasons.append(
                    {
                        "frame_id": frame_id,
                        "reason": f"content sha256 {digest} != stored "
                                  f"{row['content_sha256']}",
                    }
                )
                continue
            if int(nbytes) != int(row["nbytes"]):
                reasons.append(
                    {
                        "frame_id": frame_id,
                        "reason": f"size {nbytes} != stored {row['nbytes']}",
                    }
                )
                continue
            checked.append(
                {"frame_id": frame_id, "content_sha256": digest, "nbytes": int(nbytes)}
            )
        extra = sorted(set(str(k) for k in source_paths) - seen)
        if extra:
            reasons.append(
                {
                    "frame_id": None,
                    "reason": f"new frames appeared in the selection: {extra} "
                              "(the selection inventory is frozen)",
                }
            )
        if reasons:
            raise ResumeIncompatible(
                "resume refused: input content does not match the frozen "
                f"inventory ({len(reasons)} reason(s))",
                reasons=reasons,
            )
        return {"checked": checked, "count": len(checked)}

    def restore_policy(self, apply: Callable[[dict], Any] | None = None) -> dict:
        """Return the stored effective policy and optionally apply it.

        This is what prevents a restart from silently re-initialising a bounded
        run as dense (and from resetting the one-shot fallback counter).
        """
        policy = self.policy
        if apply is not None:
            apply(dict(policy))
        return policy

    def plan(self, expected_cell_ids: Sequence[str] | None = None) -> dict:
        """Exact sparse completeness; invalid cells stay isolated.

        Returns ``pending`` (missing + isolated-invalid, never a prefix) and the
        exact completion predicate. Global corruption is a refusal.
        """
        frozen = self.expected_cell_ids
        if expected_cell_ids is not None:
            observed = sorted({str(cid) for cid in expected_cell_ids})
            if observed != sorted(frozen):
                missing = sorted(set(frozen) - set(observed))
                extra = sorted(set(observed) - set(frozen))
                raise ResumeIncompatible(
                    "resume refused: the planned cell set changed since the run "
                    f"was frozen (missing={missing[:5]}, extra={extra[:5]}, "
                    f"frozen={len(frozen)}, planned={len(observed)})",
                    reasons=[
                        {
                            "section": "inventory",
                            "missing": missing,
                            "extra": extra,
                            "detail": "the selection inventory is frozen; new "
                                      "images are never discovered on resume",
                        }
                    ],
                )
        for entry in self.store.report().get("invalid_entries", []):
            reason = str(entry.get("reason", ""))
            if (
                "another run identity" in reason
                or "another run_id" in reason
                or "belongs to run_id" in reason
            ):
                raise ResumeCorruptGlobal(
                    "global identity is corrupt: a commit belongs to another "
                    f"run/identity ({entry.get('file')}: {reason}); refusing any "
                    "best-effort recovery"
                )
        completeness = self.store.completeness(frozen)
        # Multiple valid attempts are ambiguous: never "last found".
        ambiguous: list[str] = []
        for cell_id in completeness["complete_cell_ids"] + completeness["empty_cell_ids"]:
            attempts = self.store.attempts(cell_id)
            if len(attempts) > 1:
                ambiguous.append(cell_id)
        if ambiguous:
            raise ResumeIncompatible(
                "resume refused: several durable attempts exist for "
                f"{len(ambiguous)} cell(s) ({ambiguous[:5]}); an explicit "
                "attempt selection is required (never 'last found')",
                reasons=[
                    {"section": "attempts", "cells": ambiguous}
                ],
            )
        invalid = list(completeness["invalid"])
        pending = [
            cid for cid in frozen
            if cid in set(completeness["missing"]) or cid in set(invalid)
        ]
        return {
            "run_id": self.store.run_id,
            "expected": len(frozen),
            "complete": len(completeness["complete_cell_ids"]),
            "empty_validated": len(completeness["empty_cell_ids"]),
            "pending": pending,
            "pending_count": len(pending),
            "invalid_isolated": invalid,
            "invalidated_outputs": [
                {"cell_id": cid, "derived": "patch_and_offsets"} for cid in invalid
            ],
            "is_complete": not pending,
            "completed_cell_ids": (
                completeness["complete_cell_ids"]
                + completeness["empty_cell_ids"]
            ),
            "policy": "sparse cell_id set; never a 0..K prefix",
        }

    def completed_descriptors(self) -> list[dict]:
        """Light metadata of finished cells (no arrays are loaded here)."""
        rows = []
        for cell_id in self.state["expected_cell_ids"]:
            attempts = self.store.attempts(cell_id)
            if not attempts:
                continue
            for attempt_id in attempts:
                receipt = self.store.verify_cell(cell_id, attempt_id)
                if receipt["valid"]:
                    record = self.store.commit_record(cell_id, attempt_id)
                    result_record = (record.manifest.get("payload_meta") or {}).get(
                        "record"
                    ) or {}
                    rows.append(
                        {
                            "cell_id": cell_id,
                            "attempt_id": attempt_id,
                            "payload_sha256": receipt["payload_sha256"],
                            "stored_bytes": receipt["stored_bytes"],
                            # Plan §6.1: the per-Cell photometric reference is
                            # stored WITH the cell commit and never re-derived
                            # from the cells that happen to remain.
                            "reference_frame_id": result_record.get(
                                "reference_frame_id"
                            ),
                            "reference_frame_role": result_record.get(
                                "reference_frame_role"
                            ),
                            "bookkeeping_reference_frame_id": result_record.get(
                                "bookkeeping_reference_frame_id"
                            ),
                        }
                    )
        return rows

    # -- frozen artifacts (REWORK-1) -------------------------------------
    def restore_gauge(self):
        """Load the COMPLETE stored gauge (never re-derive it)."""
        from . import run_artifacts as zart

        state_sha = self.state.get("gauge_sha256")
        identity_sha = str(
            self.store.identity["science_contract"]["gauge"]["sha256"]
        )
        if state_sha is not None and str(state_sha) != identity_sha:
            raise zart.ResumeIncompatible(
                "resume refused: frozen state gauge hash disagrees with the "
                "run identity; refusing any gauge reconstruction"
            )
        return zart.restore_global_gauge(self.run_root, identity=self.identity)

    def restore_geometry(self):
        """Load the frozen canvas/layout/cells (never run the layout search)."""
        from . import run_artifacts as zart

        return zart.restore_geometry(self.run_root, identity=self.identity)

    def mark_finalized(self, outputs: Mapping[str, Any] | None = None) -> dict:
        self.state = write_state(
            self.run_root,
            run_id=self.state["run_id"],
            expected_cell_ids=self.state["expected_cell_ids"],
            policy=self.state["policy"],
            plan_generation=self.state["plan_generation"],
            finalized=True,
            gauge_sha256=self.state["gauge_sha256"],
            outputs=outputs or self.state["outputs"],
            created_unix_ns=self.state["created_unix_ns"],
            note=self.state["note"],
            admission=self.state.get("admission"),
            working_set_transitions=self.state.get(
                "working_set_transitions"
            ),
            column_batch_transitions=self.state.get(
                "column_batch_transitions"
            ),
        )
        return self.state

    def __enter__(self) -> "ResumeBootstrap":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
