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
    return {
        "schema": ADMISSION_SCHEMA,
        "schema_version": ADMISSION_VERSION,
        "backend": str(backend),
        "tile_px": int(tile_px),
        "parity_qualified": bool(parity),
    }


def build_admission(
    *,
    backend: Any,
    tile_px: Any,
    parity_qualified: Any,
) -> dict:
    """Build and validate the frozen admission contract (never fabricated)."""
    return validate_admission(
        {
            "schema": ADMISSION_SCHEMA,
            "schema_version": ADMISSION_VERSION,
            "backend": backend,
            "tile_px": tile_px,
            "parity_qualified": parity_qualified,
        }
    )


def validate_state(raw: Any) -> dict:
    _strict_keys(
        raw, "resume state",
        (
            "schema", "schema_version", "run_id", "created_unix_ns",
            "expected_cell_ids", "policy", "plan_generation", "finalized",
        ),
        ("gauge_sha256", "outputs", "note", "admission"),
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
    state = validate_state(payload)
    tmp_dir = root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    zrs.atomic_write_bytes(
        state_path(root), zrs.canonical_json(state), tmp_dir=tmp_dir
    )
    return state


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
        )
        return self.state

    def __enter__(self) -> "ResumeBootstrap":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
