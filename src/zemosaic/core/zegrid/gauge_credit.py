"""G2a — product edge of the bounded gauge-swap credit (Linux, gauge only).

This module is the ONLY product seam that reads host capacity for the gauge
credit. It stays deliberately small and side-effect free apart from reading
``/proc``/``/sys`` text (read-only, never written):

* :func:`resolve_gauge_memory_policy` reads the execution policy from the
  existing worker config namespace (absent ⇒ ``ram_only``, unknown ⇒ typed
  refusal). G2b owns the GUI/config key; G2a only consumes it.
* :func:`collect_live_capacity` reads the G1a facts ONCE. Every failure is
  mapped to an explicit ``unknown`` (never to "unlimited"): an unparsable
  ``/proc/self/cgroup`` is capacity unknown, and ``/`` is NEVER substituted as
  the caller's cgroup (that would silently drop every systemd-scope cap).
* :class:`GaugeCreditSession` turns those facts into per-bundle
  :class:`gauge_swap_capacity.CreditDecision` values and produces the frozen
  :class:`gauge_admission.GaugeCredit` the per-group evaluator consumes. It is
  a PERMIT: ``open_for_compute``/``close`` only mark the interval during which
  the gauge may use swap, and ``close`` always reports ``swap_used_bytes=None``
  (this module never pretends to observe swap usage).

Contract obligations honoured here:

1. the A/B_ram pair is the bundle's own observation, so ``evaluate_credit`` and
   ``evaluate_groups`` always share the same ``B_ram`` (§4.4, G2a obligation 2);
2. a credit is only ever offered for a FRESH compute gauge, exactly one worker,
   Linux and a non-``per_cell`` gauge mode (§4.7 eligibility);
3. once :meth:`GaugeCreditSession.close` ran, :meth:`credit_for_compute` is
   ``None`` forever: the credit can never reach the cache/stack.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from . import gauge_admission as adm
from . import gauge_swap_capacity as cap

__all__ = [
    "POLICY_KEY",
    "EVENT_OPEN",
    "EVENT_CLOSED",
    "GaugeCreditError",
    "GaugePolicyError",
    "LiveCapacity",
    "GaugeCreditSession",
    "current_platform",
    "resolve_gauge_memory_policy",
    "collect_live_capacity",
]

#: Execution-config key consumed by G2a (defaulted/validated by G2b).
POLICY_KEY = "zegrid_gauge_memory_policy"

EVENT_OPEN = "gauge_credit_open"
EVENT_CLOSED = "gauge_credit_closed"

CGROUP_MOUNT = "/sys/fs/cgroup"
_PROC_MEMINFO = "/proc/meminfo"
_PROC_SWAPS = "/proc/swaps"
_PROC_SELF_CGROUP = "/proc/self/cgroup"
_SYS_VM = "/proc/sys/vm"

_POLICIES = (cap.POLICY_RAM_ONLY, cap.POLICY_BOUNDED_SWAP)


class GaugeCreditError(RuntimeError):
    """Base class for gauge-credit wiring errors (fail closed).

    Subclasses :class:`RuntimeError` so the existing pre-stack refusal
    convention is preserved, and carries an optional closed ``reason_code``
    (§10.7) so G2b can classify the refusal instead of parsing prose.
    """

    def __init__(self, message: str, *, reason_code: str | None = None) -> None:
        super().__init__(message)
        if reason_code is None:
            prefix = str(message).split(":", 1)[0].strip()
            reason_code = prefix if prefix.startswith("GAUGE_CREDIT_") else None
        self.reason_code = reason_code


class GaugePolicyError(GaugeCreditError, ValueError):
    """The configured memory policy is unknown (never silently coerced)."""


def current_platform() -> str:
    """Return the platform token used by the frozen eligibility rule."""
    return "linux" if sys.platform.startswith("linux") else str(sys.platform)


def resolve_gauge_memory_policy(config: Any) -> str:
    """Return ``ram_only``/``bounded_swap`` from an execution config namespace.

    ``config`` is the existing worker namespace (``zconfig``); an absent
    attribute, ``None`` or the empty string is the historical ``ram_only``
    behaviour, byte for byte. Any other value is a typed error: the engine must
    reject an unknown enum instead of guessing.
    """
    raw = None if config is None else getattr(config, POLICY_KEY, None)
    if raw is None:
        return cap.POLICY_RAM_ONLY
    if not isinstance(raw, str):
        raise GaugePolicyError(
            f"{POLICY_KEY} must be a string, got {type(raw).__name__}={raw!r}"
        )
    value = raw.strip().lower()
    if not value:
        return cap.POLICY_RAM_ONLY
    if value not in _POLICIES:
        raise GaugePolicyError(
            f"unknown {POLICY_KEY}={raw!r}; closed enum is {_POLICIES}"
        )
    return value


# ---------------------------------------------------------------------------
# Live capacity (read once, read-only, every failure = unknown)
# ---------------------------------------------------------------------------

def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _read_int(path: str) -> int | None:
    text = _read_text(path)
    if text is None:
        return None
    stripped = text.strip()
    try:
        return int(stripped)
    except ValueError:
        return None


@dataclass(frozen=True)
class LiveCapacity:
    """The parsed facts, with ``None`` meaning UNKNOWN (never unlimited)."""

    devices: tuple[cap.SwapDevice, ...] | None
    cgroup_levels: tuple[cap.CgroupLevel, ...] | None
    overcommit: cap.CommitFacts | None
    facts: Mapping[str, Any] = field(default_factory=dict)

    def for_a_raw(self, a_raw: int) -> cap.GaugeSwapCapacity:
        """Bind the facts to the caller's OWN observation ``A``.

        The A value is never re-read here: the admission budget and the credit
        budget must come from the same observation (G2a obligation 2).
        """
        return cap.GaugeSwapCapacity(
            a_raw=int(a_raw),
            devices=self.devices,
            cgroup_levels=self.cgroup_levels,
            overcommit=self.overcommit,
            facts=dict(self.facts),
        )


def collect_live_capacity() -> LiveCapacity:
    """Read the G1a facts from this Linux host, without ever guessing.

    * ``/proc/meminfo`` unreadable ⇒ the A-dependent fields stay unknown at
      evaluation time (the evaluator receives the caller's A explicitly).
    * ``/proc/swaps`` unreadable ⇒ ``devices=None`` (credit 0).
    * ``/proc/self/cgroup`` unparsable (v1-only, malformed) ⇒
      ``cgroup_levels=None`` (credit 0). ``/`` is NEVER substituted.
    * ``/proc/sys/vm`` unreadable ⇒ ``overcommit=None`` (credit 0 in strict
      mode; the evaluator fails closed on a missing mode).
    """
    meminfo_text = _read_text(_PROC_MEMINFO)
    meminfo = cap.read_meminfo_text(meminfo_text) if meminfo_text else {}

    swaps_text = _read_text(_PROC_SWAPS)
    devices = cap.read_swap_devices(swaps_text) if swaps_text is not None else None

    cgroup_text = _read_text(_PROC_SELF_CGROUP)
    levels: tuple[cap.CgroupLevel, ...] | None
    own_cgroup = None
    cgroup_parse_error = None
    if cgroup_text is None:
        cgroup_parse_error = "self_cgroup_unreadable"
    else:
        try:
            own_cgroup = cap.parse_self_cgroup(cgroup_text)
        except ValueError as exc:
            cgroup_parse_error = f"self_cgroup_invalid: {exc}"
    if own_cgroup is None:
        levels = None
    else:
        try:
            levels = tuple(cap.read_cgroup_levels(own_cgroup, CGROUP_MOUNT))
        except (OSError, ValueError):
            levels = None

    mode = _read_int(os.path.join(_SYS_VM, "overcommit_memory"))
    ratio = _read_int(os.path.join(_SYS_VM, "overcommit_ratio"))
    kbytes = _read_int(os.path.join(_SYS_VM, "overcommit_kbytes"))
    try:
        overcommit = cap.read_overcommit(
            mode,
            ratio,
            kbytes,
            commit_limit=meminfo.get("CommitLimit"),
            committed_as=meminfo.get("Committed_AS"),
        )
    except (TypeError, ValueError):
        overcommit = None

    facts = {
        "schema": "gauge-credit-live-sources-v1",
        "cgroup_path": own_cgroup,
        "cgroup_error": cgroup_parse_error,
        "meminfo_available": bool(meminfo_text),
        "swaps_available": swaps_text is not None,
        "platform": current_platform(),
    }
    return LiveCapacity(
        devices=devices,
        cgroup_levels=levels,
        overcommit=overcommit,
        facts=facts,
    )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class GaugeCreditSession:
    """Eligibility facts + live capacity, scoped to one gauge compute.

    The session is created BEFORE the pre-gauge plan admission (so the planner
    may price the singleton gauge group against ``B_ram + S_credit``) and is
    only ``open`` around ``compute_global_gauge``. It is closed in a ``finally``
    block; from that point the credit is unreachable.
    """

    policy: str
    capacity: LiveCapacity
    platform: str = field(default_factory=current_platform)
    restored: bool = False
    imported: bool = False
    workers: int = 1
    cell_gauge_mode: str = "global"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    _closed: bool = False
    _active: bool = False
    _active_decision: cap.CreditDecision | None = None

    def __post_init__(self) -> None:
        if self.policy not in _POLICIES:
            raise GaugePolicyError(f"unknown policy {self.policy!r}")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int):
            raise GaugeCreditError("workers must be an int")
        if self.workers < 1:
            raise GaugeCreditError("workers must be >= 1")

    # -- eligibility -----------------------------------------------------
    @property
    def eligible(self) -> bool:
        """Frozen §4.7 eligibility, minus the per-candidate capacity facts."""
        return bool(
            self.policy == cap.POLICY_BOUNDED_SWAP
            and str(self.platform).strip().lower() == "linux"
            and not self.restored
            and not self.imported
            and not self._closed
            and str(self.cell_gauge_mode or "").strip().lower()
            != "per_cell"
        )

    @property
    def closed(self) -> bool:
        return self._closed

    # -- decisions -------------------------------------------------------
    def decision_for(self, *, a_raw: int, p_gauge_raw: int) -> cap.CreditDecision | None:
        """Evaluate the credit for ONE gauge phase against the caller's ``A``."""
        if not self.eligible:
            return None
        decision = cap.evaluate_credit(
            self.capacity.for_a_raw(int(a_raw)),
            int(p_gauge_raw),
            self.policy,
        )
        return decision

    def decision_for_bundle(self, bundle: Any, *, p_gauge_raw: int):
        """Evaluate the credit on the bundle's OWN raw observation.

        Returns ``(decision, credit_or_None)``. The credit is only built when
        the evaluator's ``B_ram`` equals the bundle's own plan budget; a
        mismatch would let two different observations build one ceiling and is
        refused (fail closed) instead of being spent.
        """
        envelope = getattr(bundle, "envelope", None)
        if envelope is None or not self.eligible:
            return None, None
        raw = int(getattr(envelope, "raw_available_bytes", 0))
        budget = int(getattr(envelope, "budget_bytes", -1))
        if raw <= 0:
            return None, None
        decision = self.decision_for(a_raw=raw, p_gauge_raw=int(p_gauge_raw))
        if decision is None or decision.b_ram is None:
            return decision, None
        if int(decision.b_ram) != budget:
            self.diagnostics["budget_mismatch"] = {
                "credit_b_ram_bytes": int(decision.b_ram),
                "bundle_budget_bytes": int(budget),
            }
            return decision, None
        return decision, self.plan_credit(decision, workers=self.workers)

    def plan_credit(
        self, decision: cap.CreditDecision | None, *, workers: int
    ) -> adm.GaugeCredit | None:
        """Build the evaluator input (not a grant: eligibility is re-checked)."""
        if decision is None or self._closed:
            return None
        return adm.GaugeCredit(
            decision=decision,
            policy=self.policy,
            restored=bool(self.restored),
            imported=bool(self.imported),
            workers=int(workers),
            platform=str(self.platform),
        )

    @staticmethod
    def grants_credit(decision: cap.CreditDecision | None) -> bool:
        """True iff a credit is actually NEEDED and fully covered."""
        return bool(
            decision is not None
            and decision.policy == cap.POLICY_BOUNDED_SWAP
            and decision.credit_needed_bytes is not None
            and int(decision.credit_needed_bytes) > 0
            and decision.status == "credit_sufficient"
            and decision.covered
        )

    # -- compute interval ------------------------------------------------
    def refresh_capacity(self) -> None:
        """Re-read the live capacity facts BEFORE the compute interval.

        Q3: the session snapshot is taken before the pre-gauge admission; swap
        may be disabled while the plan is being fitted. Re-reading (at least)
        ``/proc/swaps`` and the cgroup chain closes that hole. ``A`` is NEVER
        touched here: it stays the bundle's own observation, so ``B_ram`` is
        unchanged and both evaluators still share the same pair.
        """
        self.capacity = collect_live_capacity()
        self.diagnostics["capacity_refresh_count"] = (
            int(self.diagnostics.get("capacity_refresh_count", 0)) + 1
        )

    def open_for_compute(
        self, bundle: Any, *, p_gauge_raw: int
    ) -> dict[str, Any] | None:
        """Arm the credit for the gauge compute; return the open event or None.

        A credit that is no longer sufficient under the CURRENT observation is
        a hard refusal (typed code): the run must not start a swap-financed
        gauge on a stale permit.
        """
        if self._closed:
            raise GaugeCreditError("credit session is already closed")
        if not self.eligible:
            return None
        # Q3: refresh the swap/cgroup/commit facts at the LAST moment; the
        # plan-time snapshot is never trusted as a permit for the compute.
        self.refresh_capacity()
        decision, credit = self.decision_for_bundle(
            bundle, p_gauge_raw=int(p_gauge_raw)
        )
        if decision is None or credit is None:
            return None
        if not self.grants_credit(decision):
            self.events.append(
                {
                    "event": EVENT_OPEN,
                    "policy": self.policy,
                    "opened": False,
                    "reason": str(decision.status),
                    "credit_needed_bytes": decision.credit_needed_bytes,
                }
            )
            return None
        if self.workers != 1:
            raise GaugeCreditError(
                "the bounded gauge credit requires exactly ONE gauge worker"
            )
        self._active = True
        self._active_decision = decision
        event = {
            "event": EVENT_OPEN,
            "policy": self.policy,
            "opened": True,
            "credit_bytes": int(decision.s_credit),
            "credit_needed_bytes": int(decision.credit_needed_bytes or 0),
            "a_raw_bytes": int(decision.a_raw or 0),
            "b_ram_bytes": int(decision.b_ram or 0),
            "p_gauge_raw_bytes": int(decision.p_gauge_raw),
            "s_effectif_bytes": (
                None if decision.s_effectif is None else int(decision.s_effectif)
            ),
            "workers": 1,
            "platform": self.platform,
        }
        self.events.append(dict(event))
        return event

    def credit_for_compute(self) -> adm.GaugeCredit | None:
        """The credit while (and only while) the gauge compute is running."""
        if self._closed or not self._active or self._active_decision is None:
            return None
        return self.plan_credit(self._active_decision, workers=1)

    def close(self) -> dict[str, Any]:
        """Close the credit and return the ``gauge_credit_closed`` event.

        Idempotent. The event never reports the credit as observed swap use.
        """
        was_active = bool(self._active)
        decision = self._active_decision
        self._closed = True
        self._active = False
        self._active_decision = None
        event = {
            "event": EVENT_CLOSED,
            "closed": True,
            "policy": self.policy,
            "was_open": was_active,
            "credit_granted_during_gauge": was_active,
            "credit_bytes_released": (
                int(decision.s_credit) if was_active and decision is not None else 0
            ),
            "swap_used_bytes": None,
            "note": (
                "the credit is a PERMIT, never an observed swap usage; "
                "it is closed before any cache/stack/admission step"
            ),
        }
        self.events.append(dict(event))
        return event

    def require_closed_for_stack(self) -> None:
        """Fail loudly if a stack step is reached with the credit still open."""
        if not self._closed:
            raise GaugeCreditError(
                "the gauge credit must be closed before the cache/stack admission"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "zegrid-gauge-credit-session-v1",
            "policy": self.policy,
            "platform": self.platform,
            "restored": bool(self.restored),
            "imported": bool(self.imported),
            "workers": int(self.workers),
            "cell_gauge_mode": str(self.cell_gauge_mode),
            "eligible": bool(self.eligible),
            "closed": bool(self._closed),
            "diagnostics": dict(self.diagnostics),
            "events": [dict(event) for event in self.events],
        }
