"""G1b — PURE per-group admission evaluator for the bounded gauge swap.

This module is the decision layer of the G0 frozen contract
(``docs/refactor/ZEGRID_GAUGE_SWAP_G0_CONTRACT_20261115.md`` §4.7–§4.8, §11
G1b, §12 ``G0-ORACLE-3``). It is deliberately free of product wiring: it
imports only the two pure contract modules (:mod:`resource_envelope` for the
frozen overlap topology and hash, :mod:`gauge_swap_capacity` for the G1a
credit decision) and never touches ``/proc``, the scheduler or the engine.

Topology preserved bit-for-bit
-----------------------------
The existing model aggregates **overlap groups**: a sum inside each group, a
max across groups, and exactly ONE 15 % margin at the boundary
(:func:`resource_envelope.aggregate_host_peak_bytes`). G1b keeps that topology
and adds a **per-group ceiling**:

``raw_g      = sum(peak(p) for p in g)``
``margined_g = ceil(raw_g * 115 / 100)``      # once per group
``ceiling_g  = B_ram``                        # every group by default
``           = B_ram + S_credit``             # the eligible singleton gauge group
``admitted   <=> every group has margined_g <= ceiling_g``

Because the margin function is monotone, ``max_g(ceil(raw_g))`` equals
``ceil(max_g(raw_g))``. The OFF path is therefore literally the same number as
:func:`resource_envelope.aggregate_host_peak_bytes` — never a re-derivation with
a new tolerance.

Credit eligibility (all mandatory, frozen order)
-----------------------------------------------
1. policy is ``bounded_swap`` (absent/``ram_only`` ⇒ credit disabled);
2. the gauge cell is fresh compute (not restored, not imported);
3. ``gauge_workers == 1``;
4. qualified platform (Linux only in v1);
5. the group is exactly ``("gauge",)`` — no overlap with
   stack/cache/assembly/export;
6. the gauge envelope is a CPU (compute) envelope — GPU/external are not
   applicable;
7. capacity is proven: no required fact unknown, ``S_effectif`` at or above
   ``S_MIN_USABLE``, commit gate and non-pageable RAM guard both satisfied;
8. the resulting credit covers ``credit_needed_bytes``, is actually NEEDED
   (``need > 0``) and was evaluated against this same ``B_ram`` (a mismatched
   budget is a typed input error). A credit that is not needed is never
   granted: ``credit_bytes = 0`` and the legacy ceiling stays ``B_ram``.

Consequences enforced by construction, not by the caller:

* the credit can **never** save a stack/cache/assembly/export group: their
  ceiling is always ``B_ram``;
* a gauge group that overlaps a non-eligible phase is refused with
  ``GAUGE_CREDIT_GROUP_NOT_SINGLETON`` instead of being subsidised;
* the true aggregate peak (raw sums) is always published in ``diagnostics`` and
  is never reduced by the credit.

Hashing
-------
``legacy_plan_hash`` is :func:`resource_envelope.compute_plan_hash` over the same
envelopes, whatever the policy: it is the durable scientific identity and must
not move when a swap policy is enabled. For an absent/``ram_only`` policy the
returned ``decision_hash`` **is** that legacy hash (byte-identical payloads). For
``bounded_swap`` a separate, versioned ``execution_digest`` is produced
(``zegrid-gauge-admission-exec-v1``) **only when a credit is actually applied**
(``credit_bytes > 0``); with comfortable RAM (``need == 0``) no credit exists, so
``decision_hash`` stays the legacy hash and G2a-3 ("policy ON + comfortable RAM
⇒ identical hash") is satisfiable. The scientific hash is identical either way.

Refusals are a closed, typed table (§10.7 codes); no generic ``except`` and no
free-form code is emitted. Nothing here selects, launches or sizes a run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import gauge_swap_capacity as cap
from . import resource_envelope as renv

__all__ = [
    "SCHEMA",
    "EXECUTION_SCHEMA",
    "POLICY_ABSENT",
    "REFUSAL_CODES",
    "REASON_OK",
    "REASON_REFUSE_HOST_RAM",
    "REASON_CREDIT_DISABLED",
    "REASON_CREDIT_NOT_APPLICABLE",
    "REASON_CREDIT_REQUIRES_ONE_WORKER",
    "REASON_CREDIT_GROUP_NOT_SINGLETON",
    "REASON_CREDIT_UNKNOWN_CAPACITY",
    "REASON_CREDIT_SWAP_BELOW_MINIMUM",
    "REASON_CREDIT_INSUFFICIENT",
    "REASON_CREDIT_COMMIT_INSUFFICIENT",
    "REASON_CREDIT_NONPAGEABLE_RAM_INSUFFICIENT",
    "AdmissionError",
    "InvalidGroups",
    "InvalidBudget",
    "InvalidCredit",
    "GaugeCredit",
    "GroupVerdict",
    "Refusal",
    "GroupDecision",
    "groups_from_envelopes",
    "admit_validated",
    "evaluate_groups",
]

#: Stable schema identifiers (versioned, separate from the legacy plan hash).
SCHEMA = "zegrid-gauge-admission-v1"
EXECUTION_SCHEMA = "zegrid-gauge-admission-exec-v1"

#: ``None`` means the policy key was absent from the run config.
POLICY_ABSENT = None

# ---------------------------------------------------------------------------
# Closed refusal table (§10.7). Stable strings; no ad-hoc code may be emitted.
# ---------------------------------------------------------------------------

REASON_OK = "R55_OK"
REASON_REFUSE_HOST_RAM = "R55_REFUSE_HOST_RAM"
REASON_CREDIT_DISABLED = "GAUGE_CREDIT_DISABLED"
REASON_CREDIT_NOT_APPLICABLE = "GAUGE_CREDIT_NOT_APPLICABLE"
REASON_CREDIT_REQUIRES_ONE_WORKER = "GAUGE_CREDIT_REQUIRES_ONE_WORKER"
REASON_CREDIT_GROUP_NOT_SINGLETON = "GAUGE_CREDIT_GROUP_NOT_SINGLETON"
REASON_CREDIT_UNKNOWN_CAPACITY = "GAUGE_CREDIT_UNKNOWN_CAPACITY"
REASON_CREDIT_SWAP_BELOW_MINIMUM = "GAUGE_CREDIT_SWAP_BELOW_MINIMUM"
REASON_CREDIT_INSUFFICIENT = "GAUGE_CREDIT_INSUFFICIENT"
REASON_CREDIT_COMMIT_INSUFFICIENT = "GAUGE_CREDIT_COMMIT_INSUFFICIENT"
REASON_CREDIT_NONPAGEABLE_RAM_INSUFFICIENT = (
    "GAUGE_CREDIT_NONPAGEABLE_RAM_INSUFFICIENT"
)

REFUSAL_CODES = frozenset(
    {
        REASON_REFUSE_HOST_RAM,
        REASON_CREDIT_DISABLED,
        REASON_CREDIT_NOT_APPLICABLE,
        REASON_CREDIT_REQUIRES_ONE_WORKER,
        REASON_CREDIT_GROUP_NOT_SINGLETON,
        REASON_CREDIT_UNKNOWN_CAPACITY,
        REASON_CREDIT_SWAP_BELOW_MINIMUM,
        REASON_CREDIT_INSUFFICIENT,
        REASON_CREDIT_COMMIT_INSUFFICIENT,
        REASON_CREDIT_NONPAGEABLE_RAM_INSUFFICIENT,
    }
)

_KNOWN_POLICIES = (cap.POLICY_RAM_ONLY, cap.POLICY_BOUNDED_SWAP)

#: Credit-refusal detail strings (stable, diagnostic only).
_DETAIL_RESTORED = "gauge_restored"
_DETAIL_IMPORTED = "gauge_imported"
_DETAIL_NOT_LINUX = "platform_not_linux"
_DETAIL_NOT_SINGLETON = "gauge_group_not_singleton"
_DETAIL_NOT_COMPUTE = "gauge_not_compute_backend"
_DETAIL_NO_DECISION = "credit_decision_missing"
_DETAIL_MEM_AVAILABLE_UNKNOWN = "mem_available_unknown"
_DETAIL_SWAP_UNKNOWN = "effective_swap_unknown"
_DETAIL_NO_DISK_SWAP = "no_eligible_disk_swap"
_DETAIL_BELOW_MIN = "effective_swap_below_minimum"
_DETAIL_NONPAGEABLE = "b_ram_below_non_pageable_new"
_DETAIL_COMMIT = "commit_headroom_insufficient"
_DETAIL_INSUFFICIENT = "credit_below_credit_needed"
_DETAIL_CREDIT_NOT_NEEDED = "credit_not_needed"


# ---------------------------------------------------------------------------
# Typed input errors (fail closed; never a generic exception handler)
# ---------------------------------------------------------------------------

class AdmissionError(ValueError):
    """Base class for malformed admission inputs."""


class InvalidGroups(AdmissionError):
    """The provided grouping does not match the frozen overlap topology."""


class InvalidBudget(AdmissionError):
    """``b_ram`` is not a valid non-negative integer budget."""


class InvalidCredit(AdmissionError):
    """The credit record is structurally invalid or self-contradictory."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GaugeCredit:
    """The credit adjudication offered to the admission evaluator.

    ``decision`` is the G1a :class:`gauge_swap_capacity.CreditDecision` (or
    ``None`` when capacity was never read). The remaining fields are run-level
    eligibility facts the capacity provider does not own: whether the gauge is
    restored/imported, how many workers it runs and on which platform.

    ``policy=None`` means the policy key was absent, and is treated exactly like
    ``ram_only`` for decisions and hashes.
    """

    decision: cap.CreditDecision | None = None
    policy: str | None = POLICY_ABSENT
    restored: bool = False
    imported: bool = False
    workers: int = 1
    platform: str = "linux"

    def normalized_policy(self) -> str:
        """Return ``ram_only``/``bounded_swap``; refuse anything else."""
        policy = self.policy
        if policy is None:
            return cap.POLICY_RAM_ONLY
        if not isinstance(policy, str) or not policy.strip():
            raise InvalidCredit(f"policy must be a non-empty string or None, got {policy!r}")
        value = policy.strip()
        if value not in _KNOWN_POLICIES:
            raise InvalidCredit(
                f"unknown memory policy {value!r}; closed enum is {_KNOWN_POLICIES}"
            )
        return value


@dataclass(frozen=True)
class Refusal:
    """One typed refusal, carrying its reason, actor and raw numbers."""

    code: str
    actor: str
    phases: tuple[str, ...]
    raw_bytes: int
    margined_bytes: int
    ceiling_bytes: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "actor": self.actor,
            "phases": list(self.phases),
            "raw_bytes": int(self.raw_bytes),
            "margined_bytes": int(self.margined_bytes),
            "ceiling_bytes": int(self.ceiling_bytes),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class GroupVerdict:
    """Per-group arithmetic and verdict (margin applied once, in the group)."""

    phases: tuple[str, ...]
    raw_bytes: int
    margined_bytes: int
    ceiling_bytes: int
    admitted: bool
    is_gauge: bool
    contains_gauge: bool
    credit_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "phases": list(self.phases),
            "raw_bytes": int(self.raw_bytes),
            "margined_bytes": int(self.margined_bytes),
            "ceiling_bytes": int(self.ceiling_bytes),
            "admitted": bool(self.admitted),
            "is_gauge": bool(self.is_gauge),
            "contains_gauge": bool(self.contains_gauge),
            "credit_applied": bool(self.credit_applied),
        }


@dataclass(frozen=True)
class GroupDecision:
    """Immutable, JSON-safe admission decision for one plan candidate."""

    schema: str
    policy: str
    b_ram_bytes: int
    admitted: bool
    groups: tuple[GroupVerdict, ...]
    aggregate_raw_bytes: int
    aggregate_margined_bytes: int
    peak_group: tuple[str, ...]
    margin_num: int
    margin_den: int
    credit_status: str
    credit_granted: bool
    credit_bytes: int
    credit_needed_bytes: int | None
    credit_refusal_code: str | None
    credit_refusal_detail: str | None
    reason_code: str
    refusals: tuple[Refusal, ...]
    legacy_plan_hash: str
    scientific_plan_hash: str
    execution_digest: str | None
    decision_hash: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy": self.policy,
            "admitted": bool(self.admitted),
            "reason_code": self.reason_code,
            "b_ram_bytes": int(self.b_ram_bytes),
            "margin": {
                "num": int(self.margin_num),
                "den": int(self.margin_den),
            },
            "groups": [group.to_dict() for group in self.groups],
            "aggregate_raw_bytes": int(self.aggregate_raw_bytes),
            "aggregate_margined_bytes": int(self.aggregate_margined_bytes),
            "peak_group": list(self.peak_group),
            "credit": {
                "status": self.credit_status,
                "granted": bool(self.credit_granted),
                "bytes": int(self.credit_bytes),
                "needed_bytes": self.credit_needed_bytes,
                "refusal_code": self.credit_refusal_code,
                "refusal_detail": self.credit_refusal_detail,
            },
            "refusals": [refusal.to_dict() for refusal in self.refusals],
            "legacy_plan_hash": self.legacy_plan_hash,
            "scientific_plan_hash": self.scientific_plan_hash,
            "execution_digest": self.execution_digest,
            "decision_hash": self.decision_hash,
            "diagnostics": dict(self.diagnostics),
        }

    def to_json(self) -> str:
        """Canonical JSON (sorted keys, compact separators, ASCII only)."""
        return _canonical_json(self.to_dict())


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ---------------------------------------------------------------------------
# Group input helpers
# ---------------------------------------------------------------------------

_PHASE_ORDER = {phase: index for index, phase in enumerate(renv.PHASES)}


def _canonical_phases(phases: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(phases, key=lambda p: _PHASE_ORDER[p]))


def groups_from_envelopes(
    envelopes: Sequence[renv.ResourceEnvelope],
) -> tuple[tuple[renv.ResourceEnvelope, ...], ...]:
    """Return the validated envelope set grouped by the frozen overlap topology.

    This is a thin adapter over :func:`resource_envelope.overlap_groups` so that
    no caller ever re-implements the grouping rule.
    """
    envs = renv.validate_envelope_set(envelopes)
    by_phase = {env.phase: env for env in envs}
    return tuple(
        tuple(by_phase[phase] for phase in group)
        for group in renv.overlap_groups(envs)
    )


def admit_validated(
    envelopes: Sequence[renv.ResourceEnvelope],
    b_ram: int,
    credit: GaugeCredit | cap.CreditDecision | None = None,
) -> GroupDecision:
    """Product bridge: admit a VALIDATED envelope set through the one evaluator.

    This is the single admission authority the scheduler's final validator and
    the gauge plan path call. It adds no arithmetic of its own: the grouping is
    the frozen overlap topology and the ceiling rules are G1b's.
    """
    return evaluate_groups(groups_from_envelopes(envelopes), b_ram, credit)


def _validate_b_ram(b_ram: int) -> int:
    if isinstance(b_ram, bool) or not isinstance(b_ram, int):
        raise InvalidBudget(f"b_ram must be an int, got {type(b_ram).__name__}={b_ram!r}")
    if b_ram < 0:
        raise InvalidBudget(f"b_ram must be >= 0, got {b_ram}")
    return int(b_ram)


def _normalize_groups(
    groups: Sequence[Sequence[renv.ResourceEnvelope]],
) -> tuple[tuple[tuple[str, ...], tuple[renv.ResourceEnvelope, ...]], ...]:
    """Validate the grouping and return ``(canonical_phases, envelopes)`` tuples.

    Fail-closed integrity rules:

    * every group is non-empty and holds ``ResourceEnvelope`` instances;
    * no phase appears twice, in one group or across groups;
    * the flattened set passes :func:`resource_envelope.validate_envelope_set`;
    * the grouping is EXACTLY the connected components of the declared overlap
      graph. A caller cannot split a group (for example isolate ``gauge``) to
      dodge the intra-group sum or to fake a singleton gauge.
    """
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise InvalidGroups(f"groups must be a sequence of groups, got {type(groups).__name__}")
    raw_groups = tuple(groups)
    if not raw_groups:
        raise InvalidGroups("groups cannot be empty")

    flat: list[renv.ResourceEnvelope] = []
    seen: set[str] = set()
    staged: list[tuple[tuple[str, ...], tuple[renv.ResourceEnvelope, ...]]] = []
    for index, group in enumerate(raw_groups):
        if isinstance(group, (str, bytes)) or not isinstance(group, Sequence):
            raise InvalidGroups(
                f"group[{index}] must be a sequence of ResourceEnvelope, "
                f"got {type(group).__name__}"
            )
        members = tuple(group)
        if not members:
            raise InvalidGroups(f"group[{index}] is empty")
        for env in members:
            if not isinstance(env, renv.ResourceEnvelope):
                raise InvalidGroups(
                    f"group[{index}] member must be a ResourceEnvelope, "
                    f"got {type(env).__name__}"
                )
            if env.phase not in _PHASE_ORDER:
                raise InvalidGroups(
                    f"group[{index}] carries unknown phase {env.phase!r}"
                )
            if env.phase in seen:
                raise InvalidGroups(f"phase {env.phase!r} appears more than once")
            seen.add(env.phase)
            flat.append(env)
        staged.append((_canonical_phases(env.phase for env in members), members))

    envs = renv.validate_envelope_set(flat)
    expected = renv.overlap_groups(envs)
    provided = frozenset(group for group, _members in staged)
    if provided != frozenset(expected):
        raise InvalidGroups(
            "provided grouping is not the frozen overlap topology: "
            f"provided={sorted(provided)}, expected={sorted(expected)}"
        )
    order = {group: index for index, group in enumerate(expected)}
    staged.sort(key=lambda item: order[item[0]])
    return tuple(staged)


# ---------------------------------------------------------------------------
# Credit eligibility (frozen order — one typed code, no guessing)
# ---------------------------------------------------------------------------

def _validate_credit_fields(credit: GaugeCredit) -> None:
    """Validate every INPUT field before any business refusal is emitted.

    Input errors must never be masked by an earlier business refusal (for
    example a restored gauge would otherwise hide ``workers=True``). The order
    of the business refusals themselves is unchanged and stays per contract.
    """
    if not isinstance(credit.restored, bool):
        raise InvalidCredit(
            f"restored must be a bool, got {type(credit.restored).__name__}"
        )
    if not isinstance(credit.imported, bool):
        raise InvalidCredit(
            f"imported must be a bool, got {type(credit.imported).__name__}"
        )
    if isinstance(credit.workers, bool) or not isinstance(credit.workers, int):
        raise InvalidCredit(
            f"workers must be an int, got {type(credit.workers).__name__}"
        )
    if credit.workers < 1:
        raise InvalidCredit(f"workers must be >= 1, got {credit.workers}")
    if not isinstance(credit.platform, str) or not credit.platform.strip():
        raise InvalidCredit(
            f"platform must be a non-empty str, got {credit.platform!r}"
        )
    if credit.decision is not None and not isinstance(credit.decision, cap.CreditDecision):
        raise InvalidCredit(
            "credit.decision must be a gauge_swap_capacity.CreditDecision or None, "
            f"got {type(credit.decision).__name__}"
        )


def _credit_refusal(
    credit: GaugeCredit,
    policy: str,
    group_phases: tuple[str, ...],
    gauge_backend: str | None,
    b_ram: int,
) -> tuple[str | None, str | None]:
    """Return ``(refusal_code, detail)`` or ``(None, None)`` when eligible.

    The order is frozen and documented in the module docstring. The first
    failing condition wins, so the emitted code is deterministic. Input fields
    are validated first: a malformed field is a typed error, not a refusal.
    """
    _validate_credit_fields(credit)
    if policy != cap.POLICY_BOUNDED_SWAP:
        return REASON_CREDIT_DISABLED, "policy_ram_only"
    if credit.restored:
        return REASON_CREDIT_NOT_APPLICABLE, _DETAIL_RESTORED
    if credit.imported:
        return REASON_CREDIT_NOT_APPLICABLE, _DETAIL_IMPORTED
    if credit.workers != 1:
        return REASON_CREDIT_REQUIRES_ONE_WORKER, f"workers_{credit.workers}"
    if credit.platform.strip().lower() != "linux":
        return REASON_CREDIT_NOT_APPLICABLE, _DETAIL_NOT_LINUX
    if group_phases != (renv.PHASE_GAUGE,):
        return REASON_CREDIT_GROUP_NOT_SINGLETON, _DETAIL_NOT_SINGLETON
    if gauge_backend != renv.BACKEND_CPU:
        return REASON_CREDIT_NOT_APPLICABLE, _DETAIL_NOT_COMPUTE

    decision = credit.decision
    if decision is None:
        return REASON_CREDIT_UNKNOWN_CAPACITY, _DETAIL_NO_DECISION
    if decision.policy != cap.POLICY_BOUNDED_SWAP:
        raise InvalidCredit(
            "credit.decision was evaluated under policy "
            f"{decision.policy!r} but admission policy is bounded_swap"
        )
    if isinstance(decision.s_credit, bool) or not isinstance(decision.s_credit, int):
        raise InvalidCredit("credit.decision.s_credit must be an int")
    if decision.s_credit < 0:
        raise InvalidCredit(f"credit.decision.s_credit must be >= 0, got {decision.s_credit}")

    # L3: the decision budget is part of the invariant, not a bypass. If
    # ``a_raw`` is unknown the budget is legitimately unknown and admission
    # fails closed with UNKNOWN_CAPACITY; a known A with a missing budget is a
    # contradictory record and must raise.
    if decision.a_raw is None:
        return REASON_CREDIT_UNKNOWN_CAPACITY, _DETAIL_MEM_AVAILABLE_UNKNOWN
    if decision.b_ram is None:
        raise InvalidCredit(
            "credit.decision.b_ram is None while a_raw is known (contradictory record)"
        )
    if int(decision.b_ram) != int(b_ram):
        # A credit computed against a different plan budget must never be spent
        # here: the ceiling would be built from two different observations.
        raise InvalidCredit(
            f"credit.decision.b_ram {decision.b_ram} != admission b_ram {b_ram}"
        )

    if decision.cgroup_swap_status == cap.UNKNOWN or decision.s_effectif is None:
        return REASON_CREDIT_UNKNOWN_CAPACITY, _DETAIL_SWAP_UNKNOWN
    if decision.s_disk_free is not None and decision.s_disk_free <= 0:
        return REASON_CREDIT_SWAP_BELOW_MINIMUM, _DETAIL_NO_DISK_SWAP
    if int(decision.s_effectif) < cap.S_MIN_USABLE:
        return REASON_CREDIT_SWAP_BELOW_MINIMUM, _DETAIL_BELOW_MIN
    if not decision.ram_guard_ok:
        return REASON_CREDIT_NONPAGEABLE_RAM_INSUFFICIENT, _DETAIL_NONPAGEABLE
    if not decision.commit.allowed:
        return REASON_CREDIT_COMMIT_INSUFFICIENT, _DETAIL_COMMIT
    if decision.credit_needed_bytes is None:
        raise InvalidCredit(
            "credit.decision.credit_needed_bytes is None while b_ram is known"
        )
    if int(decision.credit_needed_bytes) > 0 and not decision.covered:
        return REASON_CREDIT_INSUFFICIENT, _DETAIL_INSUFFICIENT
    return None, None


def _coerce_credit(credit: GaugeCredit | cap.CreditDecision | None) -> GaugeCredit:
    if credit is None:
        return GaugeCredit()
    if isinstance(credit, cap.CreditDecision):
        return GaugeCredit(decision=credit, policy=credit.policy)
    if isinstance(credit, GaugeCredit):
        return credit
    raise InvalidCredit(
        "credit must be a GaugeCredit, a CreditDecision or None, "
        f"got {type(credit).__name__}"
    )


# ---------------------------------------------------------------------------
# Public evaluator
# ---------------------------------------------------------------------------

def evaluate_groups(
    groups: Sequence[Sequence[renv.ResourceEnvelope]],
    b_ram: int,
    credit: GaugeCredit | cap.CreditDecision | None = None,
) -> GroupDecision:
    """Admit or refuse a plan candidate under a per-group ceiling.

    ``groups`` are the overlap groups of the plan (typically produced by
    :func:`groups_from_envelopes`); ``b_ram`` is the integer plan budget
    ``floor(4*A/5)``; ``credit`` carries the G1a decision and the run-level
    eligibility facts. No fact is read from the host here.

    With an absent/``ram_only`` policy the returned numbers, verdict and hashes
    are byte-identical to the current path
    (:func:`resource_envelope.aggregate_host_peak_bytes`); with
    ``bounded_swap`` only the singleton eligible compute gauge group may use
    ``B_ram + S_credit``.
    """
    b_ram_int = _validate_b_ram(b_ram)
    credit_obj = _coerce_credit(credit)
    policy = credit_obj.normalized_policy()
    normalized = _normalize_groups(groups)

    flat_envs = tuple(env for _phases, members in normalized for env in members)
    # ``validate_envelope_set`` guarantees one uniform margin across the plan.
    margin_num = int(flat_envs[0].bound.margin_num)
    margin_den = int(flat_envs[0].bound.margin_den)

    gauge_backend: str | None = None
    gauge_raw_peak: int | None = None
    for phases, members in normalized:
        if phases == (renv.PHASE_GAUGE,):
            gauge_backend = members[0].backend
            gauge_raw_peak = sum(int(env.estimate.peak_bytes) for env in members)
            break

    gauge_phases = next(
        (phases for phases, _members in normalized if renv.PHASE_GAUGE in phases),
        None,
    )
    credit_code, credit_detail = _credit_refusal(
        credit_obj,
        policy,
        gauge_phases if gauge_phases is not None else (),
        gauge_backend,
        b_ram_int,
    )

    credit_granted = credit_code is None
    credit_bytes = 0
    credit_needed = None if credit_obj.decision is None else credit_obj.decision.credit_needed_bytes
    credit_status = "not_evaluated" if credit_obj.decision is None else credit_obj.decision.status
    if credit_granted:
        # M1: a credit that is not NEEDED is never granted. With need == 0 the
        # gauge already fits B_ram; publishing S_credit here would inflate the
        # ceiling, the telemetry and (M2) the execution digest for a run that
        # uses no swap at all.
        if credit_needed is None:
            raise InvalidCredit(
                "credit.decision.credit_needed_bytes is None while eligible"
            )
        if int(credit_needed) == 0:
            credit_granted = False
            credit_bytes = 0
            credit_status = "not_needed"
            credit_detail = _DETAIL_CREDIT_NOT_NEEDED
        else:
            credit_bytes = int(credit_obj.decision.s_credit)
            # L6 (G3 rework-2, defence in depth): a decision computed for a
            # DIFFERENT gauge raw peak must never widen this admission. The
            # credited singleton gauge executes at exactly one worker, so the
            # decision's own `p_gauge_raw` is the raw peak of the group being
            # admitted; any mismatch is a stale/incoherent record and fails
            # closed instead of granting a ceiling that was not priced here.
            if gauge_raw_peak is None:
                raise InvalidCredit(
                    "a credit was granted but no singleton gauge group exists"
                )
            if int(credit_obj.decision.p_gauge_raw) != int(gauge_raw_peak):
                raise InvalidCredit(
                    "stale gauge credit decision: p_gauge_raw="
                    f"{int(credit_obj.decision.p_gauge_raw)} but the admitted "
                    f"gauge group raw peak is {int(gauge_raw_peak)}"
                )

    verdicts: list[GroupVerdict] = []
    refusals: list[Refusal] = []
    for phases, members in normalized:
        raw = sum(int(env.estimate.peak_bytes) for env in members)
        margined = renv.ceil_margin_div(raw, margin_num, margin_den)
        is_gauge = phases == (renv.PHASE_GAUGE,)
        contains_gauge = renv.PHASE_GAUGE in phases
        applied = bool(is_gauge and credit_granted)
        ceiling = b_ram_int + credit_bytes if applied else b_ram_int
        admitted = margined <= ceiling
        verdicts.append(
            GroupVerdict(
                phases=phases,
                raw_bytes=int(raw),
                margined_bytes=int(margined),
                ceiling_bytes=int(ceiling),
                admitted=bool(admitted),
                is_gauge=bool(is_gauge),
                contains_gauge=bool(contains_gauge),
                credit_applied=applied,
            )
        )
        if admitted:
            continue
        # OFF/ram_only keeps the legacy RAM-only refusal verbatim; the credit
        # adjudication status is diagnostic only and never rewrites it.
        if contains_gauge and credit_code is not None and policy == cap.POLICY_BOUNDED_SWAP:
            code = credit_code
            detail = credit_detail or ""
        else:
            code = REASON_REFUSE_HOST_RAM
            detail = "margined_peak_exceeds_ram_ceiling"
        refusals.append(
            Refusal(
                code=code,
                actor=_dominant_phase(members),
                phases=phases,
                raw_bytes=int(raw),
                margined_bytes=int(margined),
                ceiling_bytes=int(ceiling),
                detail=detail,
            )
        )

    admitted_all = all(verdict.admitted for verdict in verdicts)
    if admitted_all:
        reason_code = REASON_OK
    else:
        reason_code = refusals[0].code
        for refusal in refusals:
            if renv.PHASE_GAUGE in refusal.phases:
                reason_code = refusal.code
                break

    aggregate_raw = max(verdict.raw_bytes for verdict in verdicts)
    aggregate_margined = max(verdict.margined_bytes for verdict in verdicts)
    peak_group = min(
        (verdict for verdict in verdicts if verdict.raw_bytes == aggregate_raw),
        key=lambda verdict: _group_order(verdict.phases, normalized),
    ).phases

    legacy_hash = renv.compute_plan_hash(flat_envs)
    execution_digest: str | None = None
    # M2: the versioned execution digest exists ONLY when a credit is actually
    # applied. Policy ON with comfortable RAM (need == 0) therefore keeps the
    # legacy hash, which makes the frozen G2a-3 criterion satisfiable.
    if policy == cap.POLICY_BOUNDED_SWAP and credit_bytes > 0:
        execution_digest = _execution_digest(
            policy=policy,
            b_ram=b_ram_int,
            margin_num=margin_num,
            margin_den=margin_den,
            verdicts=tuple(verdicts),
            aggregate_raw=aggregate_raw,
            aggregate_margined=aggregate_margined,
            peak_group=peak_group,
            admitted=admitted_all,
            reason_code=reason_code,
            legacy_plan_hash=legacy_hash,
            credit_status=credit_status,
            credit_granted=credit_granted,
            credit_bytes=credit_bytes,
            credit_needed=credit_needed,
            credit_refusal_code=credit_code,
            refusals=tuple(refusals),
        )
    decision_hash = legacy_hash if execution_digest is None else execution_digest

    diagnostics = {
        # The TRUE aggregate peak stays the group-sum / cross-group max of the
        # raw peaks. The credit is never subtracted from it (§4.7).
        "aggregate_raw_bytes": int(aggregate_raw),
        "aggregate_margined_bytes": int(aggregate_margined),
        "credit_never_subtracted_from_peak": True,
        "b_ram_bytes": int(b_ram_int),
        "credit_bytes": int(credit_bytes),
        "credit_granted": bool(credit_granted),
        "credit_refusal_code": credit_code,
        "policy": policy,
        "group_count": len(verdicts),
    }

    return GroupDecision(
        schema=SCHEMA,
        policy=policy,
        b_ram_bytes=b_ram_int,
        admitted=bool(admitted_all),
        groups=tuple(verdicts),
        aggregate_raw_bytes=int(aggregate_raw),
        aggregate_margined_bytes=int(aggregate_margined),
        peak_group=peak_group,
        margin_num=margin_num,
        margin_den=margin_den,
        credit_status=credit_status,
        credit_granted=bool(credit_granted),
        credit_bytes=int(credit_bytes),
        credit_needed_bytes=credit_needed,
        credit_refusal_code=credit_code,
        credit_refusal_detail=credit_detail,
        reason_code=reason_code,
        refusals=tuple(refusals),
        legacy_plan_hash=legacy_hash,
        scientific_plan_hash=legacy_hash,
        execution_digest=execution_digest,
        decision_hash=decision_hash,
        diagnostics=diagnostics,
    )


def _dominant_phase(members: Sequence[renv.ResourceEnvelope]) -> str:
    """Actor of a refusal: the largest peak, canonical order breaking ties."""
    best = max(
        members,
        key=lambda env: (int(env.estimate.peak_bytes), -_PHASE_ORDER[env.phase]),
    )
    return best.phase


def _group_order(
    phases: tuple[str, ...],
    normalized: Sequence[tuple[tuple[str, ...], tuple[renv.ResourceEnvelope, ...]]],
) -> int:
    for index, (candidate, _members) in enumerate(normalized):
        if candidate == phases:
            return index
    return len(normalized)


def _execution_digest(
    *,
    policy: str,
    b_ram: int,
    margin_num: int,
    margin_den: int,
    verdicts: Sequence[GroupVerdict],
    aggregate_raw: int,
    aggregate_margined: int,
    peak_group: tuple[str, ...],
    admitted: bool,
    reason_code: str,
    legacy_plan_hash: str,
    credit_status: str,
    credit_granted: bool,
    credit_bytes: int,
    credit_needed: int | None,
    credit_refusal_code: str | None,
    refusals: Sequence[Refusal],
) -> str:
    """Versioned execution digest, separate from the scientific plan hash.

    Volatile observations never enter it: only the decision inputs and outputs
    that an auditor must reproduce.
    """
    payload = {
        "schema": EXECUTION_SCHEMA,
        "policy": policy,
        "b_ram_bytes": int(b_ram),
        "margin": {"num": int(margin_num), "den": int(margin_den)},
        "groups": [
            {
                "phases": list(verdict.phases),
                "raw_bytes": int(verdict.raw_bytes),
                "margined_bytes": int(verdict.margined_bytes),
                "ceiling_bytes": int(verdict.ceiling_bytes),
                "admitted": bool(verdict.admitted),
                "credit_applied": bool(verdict.credit_applied),
            }
            for verdict in verdicts
        ],
        "aggregate_raw_bytes": int(aggregate_raw),
        "aggregate_margined_bytes": int(aggregate_margined),
        "peak_group": list(peak_group),
        "admitted": bool(admitted),
        "reason_code": reason_code,
        "legacy_plan_hash": legacy_plan_hash,
        "credit": {
            "status": credit_status,
            "granted": bool(credit_granted),
            "bytes": int(credit_bytes),
            "needed_bytes": credit_needed,
            "refusal_code": credit_refusal_code,
        },
        "refusals": [refusal.to_dict() for refusal in refusals],
    }
    blob = _canonical_json(payload).encode("ascii")
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"
