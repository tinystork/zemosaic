"""G1a — PURE Linux gauge swap-capacity provider (schema ``gauge-swap-capacity-v1``).

Scope (mission ``ZM-GAUGE-SWAP-G1A-20261115``, contract
``docs/refactor/ZEGRID_GAUGE_SWAP_G0_CONTRACT_20261115.md`` §4, §5, §6, §11 G1a):

* parse the facts that decide the bounded gauge-swap credit — ``/proc/meminfo``
  text, ``/proc/swaps`` text, swapfile backing, cgroup v2 ancestor caps and
  ``vm.overcommit_*`` — into immutable records with explicit trinary status
  (``not_applicable`` / ``unlimited`` / ``limited`` / ``unknown``);
* evaluate the frozen integer credit formula for ONE gauge compute phase;
* never write anything, never launch anything, and never import product code
  (``zemosaic_zegrid_mode``, the scheduler or any engine module).

Design rules frozen by G0:

* **integer arithmetic only** — no float factor, no ``math.inf``, no rounding
  drift. ``floor``/``ceil`` are exact ``//`` expressions.
* The global ``SwapFree`` is **never** used as capacity: disk swap is summed
  per eligible device, zram is excluded (it is RAM-backed), a swap file whose
  allocated blocks do not cover its declared size is excluded ("backing not
  proven").
* ``SwapCached`` is neither added to ``A`` (``MemAvailable`` is already net) nor
  to swap free.
* ``memory.max`` never finances the credit; it only bounds the non-pageable
  guard through ``B_ram_eff = min(B_ram, cgroup RAM headroom)``.
* every **required** unknown fact yields credit 0 with an explicit reason;
  ``unknown`` is never converted into ``unlimited``.
* credit is ``floor(9 * S_effectif / 10)`` with ``R_MIN = 0`` and the
  eligibility floor ``S_MIN_USABLE = 1 GiB`` (a floor, not a cap).

The six contract entry points are :func:`read_meminfo_text`,
:func:`read_swap_devices`, :func:`read_overcommit`, :func:`read_cgroup_levels`,
:func:`combine_cgroup` and :func:`evaluate_credit`. Support helpers
(:func:`plan_budget`, :func:`admit_budget`, :func:`ceil_margin`,
:func:`swap_credit`, :func:`credit_needed`, :func:`s_needed_for`,
:func:`evaluate_commit`, :func:`canonical_json`) are pure as well.

This module is not wired into the engine in G1a: no product call site imports it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple

#: Stable schema identifier carried by every serialized decision.
SCHEMA = "gauge-swap-capacity-v1"

MIB = 1024 * 1024
GIB = 1024 ** 3

#: Budget factors (unchanged by this mission).
PLAN_NUM, PLAN_DEN = 4, 5
CREDIT_NUM, CREDIT_DEN = 9, 10
MARGIN_NUM, MARGIN_DEN = 15, 100

#: Eligibility floor for the bounded-swap path (NOT a cap, NOT a reservation).
S_MIN_USABLE = 1 * GIB
#: V1 reserve. The 10 % retained by floor(9*S/10) is the only retained margin.
R_MIN = 0

#: Conservative upper bound of the new gauge worker's non-pageable resident
#: bytes. Already counted once in ``P_gauge_raw``; never re-subtracted from the
#: credit.
PROCESS_BASELINE_BYTES = 340 * MIB

NA = "not_applicable"
UNLIMITED = "unlimited"
LIMITED = "limited"
UNKNOWN = "unknown"

POLICY_RAM_ONLY = "ram_only"
POLICY_BOUNDED_SWAP = "bounded_swap"

_CGROUP_KINDS = ("ram", "swap")

_MEMINFO_RE = re.compile(r"^([A-Za-z0-9_()]+):\s+(\d+)(?:\s+kB)?\s*$")
_ZRAM_RE = re.compile(r"zram\d+")
_INT_RE = re.compile(r"-?\d+")

__all__ = [
    "SCHEMA",
    "MIB",
    "GIB",
    "PLAN_NUM",
    "PLAN_DEN",
    "CREDIT_NUM",
    "CREDIT_DEN",
    "MARGIN_NUM",
    "MARGIN_DEN",
    "S_MIN_USABLE",
    "R_MIN",
    "PROCESS_BASELINE_BYTES",
    "NA",
    "UNLIMITED",
    "LIMITED",
    "UNKNOWN",
    "POLICY_RAM_ONLY",
    "POLICY_BOUNDED_SWAP",
    "SwapDevice",
    "CgroupLevel",
    "CombinedStatus",
    "CommitFacts",
    "CommitGate",
    "GaugeSwapCapacity",
    "CreditDecision",
    "plan_budget",
    "admit_budget",
    "ceil_margin",
    "swap_credit",
    "credit_needed",
    "s_needed_for",
    "read_meminfo_text",
    "is_zram_name",
    "read_swap_devices",
    "disk_swap_free",
    "cgroup_chain_paths",
    "parse_self_cgroup",
    "read_cgroup_levels",
    "combine_cgroup",
    "read_overcommit",
    "evaluate_commit",
    "evaluate_credit",
    "canonical_json",
]


# ---------------------------------------------------------------------------
# Exact integer formulas (frozen)
# ---------------------------------------------------------------------------

def _as_int(name: str, value: int) -> int:
    """Validate a non-negative integer input (bools are rejected)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def plan_budget(available_bytes: int) -> int:
    """``B_plan = floor(4*A/5)`` — unchanged by the swap policy."""
    return (PLAN_NUM * _as_int("available_bytes", available_bytes)) // PLAN_DEN


def admit_budget(available_bytes: int) -> int:
    """``B_admit = floor(9*A/10)`` — admission ceiling, never used for credit."""
    return (CREDIT_NUM * _as_int("available_bytes", available_bytes)) // CREDIT_DEN


def ceil_margin(raw_bytes: int, num: int = MARGIN_NUM, den: int = MARGIN_DEN) -> int:
    """The ONE 15 % margin, ``ceil(raw * (den + num) / den)``, applied once per group."""
    raw = _as_int("raw_bytes", raw_bytes)
    return (raw * (int(den) + int(num)) + int(den) - 1) // int(den)


def swap_credit(s_effectif: int | None) -> int:
    """``S_credit = floor(9*S_effectif/10)`` under the frozen eligibility floor.

    ``None`` (unknown effective swap) always yields 0: unknown is never
    promoted to unlimited.
    """
    if s_effectif is None:
        return 0
    s = max(0, int(s_effectif) - R_MIN)
    if s < S_MIN_USABLE:
        return 0
    return (CREDIT_NUM * s) // CREDIT_DEN


def credit_needed(p_gauge_raw: int, b_ram: int) -> int:
    """Credit bytes required so the margined gauge fits ``B_ram + credit``."""
    return max(0, ceil_margin(p_gauge_raw) - int(b_ram))


def s_needed_for(credit_bytes: int) -> int:
    """Smallest ``S_effectif`` whose 90 % credit covers ``credit_bytes``."""
    c = max(0, int(credit_bytes))
    return (c * CREDIT_DEN + CREDIT_NUM - 1) // CREDIT_NUM


# ---------------------------------------------------------------------------
# /proc/meminfo
# ---------------------------------------------------------------------------

def read_meminfo_text(text: str) -> dict[str, int]:
    """Parse ``/proc/meminfo`` text into **bytes**, last value winning.

    Lines carrying ``kB`` are multiplied by 1024; other numeric fields are kept
    as-is. Malformed lines are skipped (they carry no decision fact). The parse
    is pure: no file is opened here.
    """
    if not isinstance(text, str):
        raise TypeError(f"meminfo text must be a str, got {type(text).__name__}")
    values: dict[str, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        match = _MEMINFO_RE.match(stripped)
        if not match:
            continue
        key, raw = match.group(1), int(match.group(2))
        values[key] = raw * 1024 if "kB" in stripped else raw
    return values


# ---------------------------------------------------------------------------
# /proc/swaps + swapfile backing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SwapDevice:
    """One ``/proc/swaps`` row with its eligibility verdict."""

    name: str
    kind: str            # "file" | "partition" | "other"
    size_bytes: int
    used_bytes: int
    zram: bool
    eligible: bool
    reason: str

    @property
    def free_bytes(self) -> int:
        return max(0, int(self.size_bytes) - int(self.used_bytes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.kind,
            "size_bytes": int(self.size_bytes),
            "used_bytes": int(self.used_bytes),
            "free_bytes": int(self.free_bytes),
            "zram": bool(self.zram),
            "eligible_for_disk_credit": bool(self.eligible),
            "reason": self.reason,
        }


def _basename(name: str) -> str:
    return os.path.basename(os.path.realpath(name) if name.startswith("/dev/") else name)


def is_zram_name(name: str, sys_block: str | os.PathLike[str] = "/sys/block") -> bool:
    """Detect zram by basename and, defensively, by the sysfs block entry.

    ``/proc/swaps`` reports zram as ``partition`` on the reference host, so the
    TYPE column is not sufficient. The device basename is authoritative.
    """
    base = _basename(name)
    if _ZRAM_RE.fullmatch(base):
        return True
    try:
        entry = Path(sys_block) / base
        if entry.exists() and Path(os.path.realpath(entry)).name.startswith("zram"):
            return True
    except OSError:
        pass
    return False


def read_swap_devices(
    text: str,
    stat_fn: Callable[[str], Any] | None = None,
    *,
    sys_block: str | os.PathLike[str] = "/sys/block",
) -> list[SwapDevice]:
    """Parse ``/proc/swaps`` text and classify every device.

    Excluded from the credit: zram (RAM-backed), swap files whose allocated
    blocks do not cover the declared size (backing not proven), unqualified
    types, and files whose ``stat`` fails. ``stat_fn`` is injectable so tests
    never need a real swap file.

    An unreadable ``/proc/swaps`` is **not** modelled here: pass ``None`` as
    :attr:`GaugeSwapCapacity.devices` to signal "device facts unknown" and the
    evaluator will fail closed with ``swap_devices_unknown``.
    """
    if not isinstance(text, str):
        raise TypeError(f"swaps text must be a str, got {type(text).__name__}")
    stat = stat_fn or os.stat
    devices: list[SwapDevice] = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        name, kind, size_kb, used_kb = parts[0], parts[1], parts[2], parts[3]
        try:
            size_b = int(size_kb) * 1024
            used_b = int(used_kb) * 1024
        except ValueError:
            continue
        zram = is_zram_name(name, sys_block)
        eligible, reason = True, "disk_backed"
        if zram:
            eligible, reason = False, "zram_is_ram_backed"
        elif kind == "file":
            try:
                st = stat(name)
                # ``st_blocks`` is in 512-byte units, never ``st_blksize``
                # (the preferred I/O block size).  Using st_blksize made the
                # "backing proven" guard up to 8x too lenient.
                allocated = int(st.st_blocks) * 512
                if allocated < size_b:
                    eligible, reason = False, "swapfile_not_fully_allocated"
            except OSError as exc:
                eligible, reason = False, f"swapfile_stat_failed: {exc.__class__.__name__}"
        elif kind != "partition":
            eligible, reason = False, f"unsupported_swap_type:{kind}"
        devices.append(
            SwapDevice(
                name=name,
                kind=kind,
                size_bytes=size_b,
                used_bytes=used_b,
                zram=zram,
                eligible=eligible,
                reason=reason,
            )
        )
    return devices


def disk_swap_free(devices: list[SwapDevice] | tuple[SwapDevice, ...]) -> int:
    """Sum of free bytes over eligible (disk-backed) devices only."""
    return sum(device.free_bytes for device in devices if device.eligible)


# ---------------------------------------------------------------------------
# cgroup v2 chain (trinary semantics)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CgroupLevel:
    """One cgroup v2 ancestor with explicit RAM/swap trinary statuses."""

    path: str
    is_root: bool
    memory_controller: bool
    ram_status: str
    ram_headroom: int | None
    swap_status: str
    swap_headroom: int | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "is_root": bool(self.is_root),
            "memory_controller": bool(self.memory_controller),
            "ram_status": self.ram_status,
            "ram_headroom_bytes": self.ram_headroom,
            "swap_status": self.swap_status,
            "swap_headroom_bytes": self.swap_headroom,
            "detail": self.detail,
        }


class CombinedStatus(NamedTuple):
    """Trinary combination of ancestor caps (tuple-compatible with G0)."""

    status: str
    headroom_bytes: int | None
    unknown_paths: list[str]


def _read_text(path: Path) -> str | None:
    """Read a small sysfs/procfs file; ANY failure means the fact is unknown.

    ``UnicodeDecodeError`` is not an ``OSError`` but is exactly as unreadable
    for a decision: it must become ``None`` (⇒ ``unknown``), never escape.
    """
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _read_int(path: Path) -> int | None:
    text = _read_text(path)
    if text is None or text == "max" or not _INT_RE.fullmatch(text):
        return None
    return int(text)


def _cap_status(text: str | None) -> tuple[str, int | None]:
    """Classify a cgroup cap file: unlimited / limited(value) / unknown.

    Values are validated with :data:`_INT_RE` before ``int()`` so a malformed
    file is ``unknown``, never a ``ValueError`` escaping the provider.
    """
    if text is None:
        return UNKNOWN, None
    if text == "max":
        return UNLIMITED, None
    if _INT_RE.fullmatch(text):
        return LIMITED, int(text)
    return UNKNOWN, None


def parse_self_cgroup(text: str) -> str:
    """Return the caller's cgroup v2 path from ``/proc/self/cgroup`` text.

    Pure parser: no file is opened here. Only the v2 line
    (``<hierarchy-id>:<controllers>:<path>`` with id ``0`` and an EMPTY
    controller list) is accepted; a cgroup v1-only file is refused because v1
    is not qualified by the contract. Callers must treat a refusal as
    "capacity unknown" (no credit), never fall back to ``/`` silently: passing
    ``/`` loses every systemd-scope cap.
    """
    if not isinstance(text, str):
        raise TypeError(f"cgroup text must be a str, got {type(text).__name__}")
    for line in text.splitlines():
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy != "0" or controllers != "":
            continue
        if not path.startswith("/"):
            raise ValueError(f"cgroup v2 path must be absolute, got {path!r}")
        return path
    raise ValueError("no cgroup v2 '0::/path' entry in /proc/self/cgroup text")


def cgroup_chain_paths(
    own_cgroup: str,
    root: str | os.PathLike[str] = "/sys/fs/cgroup",
) -> list[str]:
    """Absolute cgroup directories from the ROOT down to the caller's cgroup."""
    root_s = str(root).rstrip("/")
    rel = (own_cgroup or "/").strip()
    if not rel.startswith("/"):
        raise ValueError(f"cgroup path must be absolute, got {own_cgroup!r}")
    parts = [part for part in rel.split("/") if part]
    paths = [root_s]
    for index in range(len(parts)):
        paths.append(root_s + "/" + "/".join(parts[: index + 1]))
    return paths


def read_cgroup_levels(
    own_cgroup: str,
    root: str | os.PathLike[str] = "/sys/fs/cgroup",
) -> list[CgroupLevel]:
    """Read every ancestor's caps with explicit NA/unlimited/unknown.

    Fail-closed detection (review M3):

    * the PRESENCE of ``memory.max`` or ``memory.swap.max`` is authoritative —
      the memory controller is active even when ``cgroup.controllers`` is
      empty or unreadable;
    * a directory that exists without any cap file and with an
      absent/unreadable ``cgroup.controllers`` is ``unknown``, never ``NA``;
    * ``not_applicable`` is reserved for the root without any cap file, or for
      a level whose readable ``cgroup.controllers`` really omits ``memory``;
    * a malformed/unreadable cap value is ``unknown`` (never an exception);
    * ``max`` is ``unlimited``; an integer is ``limited`` with
      ``max(0, max - current)`` headroom.

    ``parse_self_cgroup`` must be used by callers to obtain a real
    ``own_cgroup``; passing ``"/"`` would skip every systemd-scope cap.
    """
    levels: list[CgroupLevel] = []
    root_s = str(root).rstrip("/")
    for path_s in cgroup_chain_paths(own_cgroup, root):
        path = Path(path_s)
        is_root = Path(path_s) == Path(root_s)
        if not path.is_dir():
            levels.append(
                CgroupLevel(
                    path_s, is_root, False, UNKNOWN, None, UNKNOWN, None,
                    "cgroup_directory_missing",
                )
            )
            continue

        ram_cap_present = (path / "memory.max").exists()
        swap_cap_present = (path / "memory.swap.max").exists()
        has_cap_files = ram_cap_present or swap_cap_present
        controllers_text = _read_text(path / "cgroup.controllers")
        controllers = None if controllers_text is None else controllers_text.split()
        has_mem = controllers is not None and "memory" in controllers
        active = has_cap_files or has_mem

        if is_root and not has_cap_files:
            # The root cgroup is not limit-able: no cap file means NA.
            ram_status, ram_head = NA, None
            swap_status, swap_head = NA, None
            detail = "root_cgroup_is_unbounded"
        elif not active:
            if controllers is None:
                # Directory exists, no cap file, controllers unreadable: this
                # must fail closed instead of pretending "not applicable".
                ram_status, ram_head = UNKNOWN, None
                swap_status, swap_head = UNKNOWN, None
                detail = "memory_controller_undetectable"
            else:
                ram_status, ram_head = NA, None
                swap_status, swap_head = NA, None
                detail = "memory_controller_not_active_here"
        else:
            ram_status, ram_head = _cap_status(_read_text(path / "memory.max"))
            if ram_status == LIMITED:
                ram_cur = _read_int(path / "memory.current")
                if ram_cur is None:
                    ram_status, ram_head = UNKNOWN, None
                else:
                    ram_head = max(0, int(ram_head) - ram_cur)

            swap_status, swap_head = _cap_status(_read_text(path / "memory.swap.max"))
            if swap_status in (LIMITED, UNLIMITED):
                # max/current are read TOGETHER: an unusable current makes the
                # swap cap unknown, even when the cap itself is "max".
                swap_cur = _read_int(path / "memory.swap.current")
                if swap_cur is None:
                    swap_status, swap_head = UNKNOWN, None
                elif swap_status == LIMITED:
                    swap_head = max(0, int(swap_head) - swap_cur)
            detail = "memory_controller_active"

        levels.append(
            CgroupLevel(path_s, is_root, active, ram_status, ram_head,
                        swap_status, swap_head, detail)
        )
    return levels


def combine_cgroup(
    levels: list[CgroupLevel] | tuple[CgroupLevel, ...],
    kind: str,
) -> CombinedStatus:
    """Combine ancestor caps: ``min`` over ``limited``; ``unknown`` fails closed.

    ``not_applicable`` levels never constrain and never poison the chain.
    Returns ``(status, headroom_bytes, unknown_paths)``; the status is one of
    :data:`NA`, :data:`UNLIMITED`, :data:`LIMITED`, :data:`UNKNOWN`.
    """
    if kind not in _CGROUP_KINDS:
        raise ValueError(f"kind must be one of {_CGROUP_KINDS}, got {kind!r}")
    statuses = [getattr(level, f"{kind}_status") for level in levels]
    if UNKNOWN in statuses:
        return CombinedStatus(
            UNKNOWN,
            None,
            [level.path for level in levels if getattr(level, f"{kind}_status") == UNKNOWN],
        )
    limited = [level for level in levels if getattr(level, f"{kind}_status") == LIMITED]
    if not limited:
        return CombinedStatus(UNLIMITED if UNLIMITED in statuses else NA, None, [])
    headroom = min(int(getattr(level, f"{kind}_headroom")) for level in limited)
    return CombinedStatus(LIMITED, headroom, [])


# ---------------------------------------------------------------------------
# vm.overcommit_* (commit gate)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommitFacts:
    """``vm.overcommit_*`` facts plus, when known, the meminfo commit fields."""

    mode: int | None
    ratio: int | None
    kbytes: int | None
    commit_limit_bytes: int | None
    committed_as_bytes: int | None
    status: str
    headroom_bytes: int | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "overcommit_mode": self.mode,
            "overcommit_ratio": self.ratio,
            "overcommit_kbytes": self.kbytes,
            "commit_limit_bytes": self.commit_limit_bytes,
            "committed_as_bytes": self.committed_as_bytes,
            "status": self.status,
            "headroom_bytes": self.headroom_bytes,
            "reason": self.reason,
        }


def read_overcommit(
    mode: int | None,
    ratio: int | None,
    kbytes: int | None,
    *,
    commit_limit: int | None = None,
    committed_as: int | None = None,
) -> CommitFacts:
    """Classify the overcommit mode without inventing a headroom.

    ``mode``/``ratio``/``kbytes`` are the three ``/proc/sys/vm`` values. The
    two ``CommitLimit``/``Committed_AS`` values come from ``/proc/meminfo`` and
    are attached here so the strict-mode gate can be evaluated later by
    :func:`evaluate_commit`. Mode 2 with a missing commit field is ``unknown``,
    never inferred from ``overcommit_ratio``.
    """
    if mode is not None and (isinstance(mode, bool) or not isinstance(mode, int)):
        raise TypeError(f"mode must be int|None, got {type(mode).__name__}")
    limit = None if commit_limit is None else _as_int("commit_limit", commit_limit)
    committed = None if committed_as is None else _as_int("committed_as", committed_as)
    if mode is None:
        status, headroom, reason = UNKNOWN, None, "overcommit_mode_unknown"
    elif mode in (0, 1):
        status, headroom, reason = NA, None, f"overcommit_mode_{mode}_no_hard_gate"
    elif mode == 2:
        if limit is None or committed is None:
            status, headroom, reason = UNKNOWN, None, "commit_fields_unknown"
        else:
            status = LIMITED
            headroom = max(0, limit - committed)
            reason = "commit_headroom_declared"
    else:
        status, headroom, reason = UNKNOWN, None, f"unknown_overcommit_mode_{mode}"
    return CommitFacts(
        mode=mode,
        ratio=ratio,
        kbytes=kbytes,
        commit_limit_bytes=limit,
        committed_as_bytes=committed,
        status=status,
        headroom_bytes=headroom,
        reason=reason,
    )


@dataclass(frozen=True)
class CommitGate:
    """Commit verdict for the gauge's NEW private anonymous bytes."""

    mode: int | None
    status: str
    headroom_bytes: int | None
    new_commit_bytes: int
    allowed: bool
    reason: str
    sources: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overcommit_mode": self.mode,
            "status": self.status,
            "headroom_bytes": self.headroom_bytes,
            "new_commit_bytes": int(self.new_commit_bytes),
            "allowed": bool(self.allowed),
            "reason": self.reason,
            "sources": dict(self.sources),
        }


def evaluate_commit(
    facts: CommitFacts | None,
    new_commit_bytes: int,
) -> CommitGate:
    """Commit feasibility for NEW private anonymous bytes.

    ``new_commit_bytes`` is the gauge's modelled peak (``P_gauge_raw``). It is
    never reduced by ``B_ram``: the commit headroom counts virtual bytes, the
    RAM budget counts physical bytes, and subtracting one from the other would
    double-count.
    """
    new = _as_int("new_commit_bytes", new_commit_bytes)
    mode = None if facts is None else facts.mode
    sources = {
        "commit_limit": None if facts is None else facts.commit_limit_bytes,
        "committed_as": None if facts is None else facts.committed_as_bytes,
    }
    if mode is None:
        return CommitGate(None, UNKNOWN, None, new, False, "overcommit_mode_unknown", sources)
    if mode in (0, 1):
        return CommitGate(
            mode, NA, None, new, True, f"overcommit_mode_{mode}_no_hard_gate", sources
        )
    if mode == 2:
        if facts is None or facts.commit_limit_bytes is None or facts.committed_as_bytes is None:
            return CommitGate(mode, UNKNOWN, None, new, False, "commit_fields_unknown", sources)
        headroom = max(0, facts.commit_limit_bytes - facts.committed_as_bytes)
        allowed = new <= headroom
        return CommitGate(
            mode, LIMITED, headroom, new, allowed,
            "commit_headroom_ok" if allowed else "commit_headroom_insufficient",
            sources,
        )
    return CommitGate(
        mode, UNKNOWN, None, new, False, f"unknown_overcommit_mode_{mode}", sources
    )


# ---------------------------------------------------------------------------
# Inputs of a decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GaugeSwapCapacity:
    """Every fact needed to decide the credit, already parsed.

    ``None`` in ``a_raw``/``devices``/``cgroup_levels``/``overcommit`` means the
    fact is **unknown** (unreadable source). The evaluator fails closed on any
    unknown that is required for the credit; nothing is defaulted to infinity.
    """

    a_raw: int | None
    devices: list[SwapDevice] | tuple[SwapDevice, ...] | None
    cgroup_levels: list[CgroupLevel] | tuple[CgroupLevel, ...] | None
    overcommit: CommitFacts | None
    non_pageable_new_bytes: int = PROCESS_BASELINE_BYTES
    facts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.a_raw is not None:
            _as_int("a_raw", self.a_raw)
        _as_int("non_pageable_new_bytes", self.non_pageable_new_bytes)


# ---------------------------------------------------------------------------
# Full credit evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CreditDecision:
    """Immutable, JSON-safe outcome of :func:`evaluate_credit`."""

    policy: str
    a_raw: int | None
    b_ram: int | None
    b_admit: int | None
    b_ram_eff: int | None
    p_gauge_raw: int
    p_gauge_margined: int
    s_disk_free: int | None
    s_effectif: int | None
    s_credit: int
    credit_needed_bytes: int | None
    covered: bool
    status: str
    reasons: tuple[str, ...]
    devices: tuple[SwapDevice, ...]
    cgroup: tuple[CgroupLevel, ...]
    commit: CommitGate
    cgroup_swap_status: str
    cgroup_swap_headroom: int | None
    cgroup_ram_status: str
    cgroup_ram_headroom: int | None
    ram_guard_np_bytes: int
    ram_guard_ok: bool
    facts: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "policy": self.policy,
            "a_raw_bytes": None if self.a_raw is None else int(self.a_raw),
            "b_ram_bytes": None if self.b_ram is None else int(self.b_ram),
            "b_admit_bytes": None if self.b_admit is None else int(self.b_admit),
            "b_ram_eff_bytes": None if self.b_ram_eff is None else int(self.b_ram_eff),
            "p_gauge_raw_bytes": int(self.p_gauge_raw),
            "p_gauge_margined_bytes": int(self.p_gauge_margined),
            "s_disk_free_bytes": None if self.s_disk_free is None else int(self.s_disk_free),
            "s_effectif_bytes": None if self.s_effectif is None else int(self.s_effectif),
            "s_credit_bytes": int(self.s_credit),
            "credit_needed_bytes": (
                None if self.credit_needed_bytes is None else int(self.credit_needed_bytes)
            ),
            "covered": bool(self.covered),
            "status": self.status,
            "reasons": list(self.reasons),
            "swap_devices": [device.to_dict() for device in self.devices],
            "cgroup_levels": [level.to_dict() for level in self.cgroup],
            "cgroup_swap": {
                "status": self.cgroup_swap_status,
                "headroom_bytes": self.cgroup_swap_headroom,
            },
            "cgroup_ram": {
                "status": self.cgroup_ram_status,
                "headroom_bytes": self.cgroup_ram_headroom,
            },
            "commit": self.commit.to_dict(),
            "ram_guard": {
                "non_pageable_new_bytes": int(self.ram_guard_np_bytes),
                "b_ram_covers_non_pageable": bool(self.ram_guard_ok),
            },
            "facts": dict(self.facts),
            "limits": [
                "credit is a PERMIT, never a reservation: the OS may keep pages in RAM, "
                "evict other processes or refuse an allocation",
                "a synthetic fact or a fake MemAvailable proves the logic, never "
                "physical swap",
                "zram free is RAM-backed and is excluded from S_disk_free",
                "SwapCached is neither added to A nor to swap free",
                "no OOM guarantee is made by this contract",
                "integer arithmetic only: no float and no infinity enters a decision",
            ],
        }

    def to_json(self) -> str:
        """Canonical JSON: sorted keys, compact separators, ASCII-only."""
        return canonical_json(self.to_dict())


def canonical_json(payload: Any) -> str:
    """Stable serialization used for golden vectors (sorted keys, no spaces)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def evaluate_credit(
    capacity: GaugeSwapCapacity,
    p_gauge_raw: int,
    policy: str = POLICY_RAM_ONLY,
) -> CreditDecision:
    """Evaluate the frozen G0 contract for ONE gauge compute phase.

    Mirrors ``tools/zegrid_gauge_swap/g0_capacity.py::evaluate_contract``
    exactly (same integer formulas, same trinary combination, same reason
    ordering). Any policy other than ``bounded_swap`` yields
    ``status=not_applicable`` and **zero** credit, so a caller can never obtain
    credit without opting in.
    """
    p = _as_int("p_gauge_raw", p_gauge_raw)
    reasons: list[str] = []

    if capacity.a_raw is None:
        b_ram: int | None = None
        b_admit: int | None = None
        reasons.append("mem_available_unknown")
    else:
        b_ram = plan_budget(capacity.a_raw)
        b_admit = admit_budget(capacity.a_raw)
    p_margined = ceil_margin(p)

    devices = () if capacity.devices is None else tuple(capacity.devices)
    if capacity.devices is None:
        s_disk: int | None = None
        reasons.append("swap_devices_unknown")
    else:
        s_disk = disk_swap_free(devices)

    if capacity.cgroup_levels is None:
        levels: tuple[CgroupLevel, ...] = ()
        reasons.append("cgroup_levels_unknown")
        swap_status, swap_head, _ = UNKNOWN, None, []
        ram_status, ram_head, _ = UNKNOWN, None, []
    else:
        levels = tuple(capacity.cgroup_levels)
        swap_status, swap_head, _ = combine_cgroup(levels, "swap")
        ram_status, ram_head, _ = combine_cgroup(levels, "ram")

    if swap_status == UNKNOWN:
        s_eff: int | None = None
        reasons.append("cgroup_swap_unknown")
    elif swap_status == LIMITED:
        if s_disk is None:
            s_eff = None
        else:
            s_eff = min(s_disk, int(swap_head))
            if int(swap_head) < s_disk:
                reasons.append("cgroup_swap_caps_effective_swap")
    else:
        s_eff = s_disk

    if s_eff is not None and s_eff < S_MIN_USABLE:
        reasons.append("effective_swap_below_minimum")
    if s_disk is not None and s_disk <= 0:
        reasons.append("no_eligible_disk_swap")

    credit = swap_credit(s_eff)
    need = None if b_ram is None else credit_needed(p, b_ram)

    commit = evaluate_commit(capacity.overcommit, p)
    if not commit.allowed:
        reasons.append(commit.reason)

    if b_ram is None:
        b_ram_eff: int | None = None
    else:
        b_ram_eff = b_ram
        if ram_status == LIMITED and int(ram_head) < b_ram:
            # Two independent upper bounds: take the smaller, never subtract.
            b_ram_eff = int(ram_head)
            reasons.append("cgroup_ram_tighter_than_host")
    ram_guard_ok = b_ram_eff is not None and int(capacity.non_pageable_new_bytes) <= b_ram_eff
    if not ram_guard_ok:
        reasons.append("b_ram_below_non_pageable_new")

    if policy != POLICY_BOUNDED_SWAP:
        return CreditDecision(
            policy=policy,
            a_raw=capacity.a_raw,
            b_ram=b_ram,
            b_admit=b_admit,
            b_ram_eff=b_ram_eff,
            p_gauge_raw=p,
            p_gauge_margined=p_margined,
            s_disk_free=s_disk,
            s_effectif=None,
            s_credit=0,
            credit_needed_bytes=need,
            covered=False,
            status=NA,
            reasons=("policy_ram_only" if policy == POLICY_RAM_ONLY else "policy_not_bounded_swap",),
            devices=devices,
            cgroup=levels,
            commit=commit,
            cgroup_swap_status=swap_status,
            cgroup_swap_headroom=swap_head,
            cgroup_ram_status=ram_status,
            cgroup_ram_headroom=ram_head,
            ram_guard_np_bytes=int(capacity.non_pageable_new_bytes),
            ram_guard_ok=ram_guard_ok,
            facts=dict(capacity.facts),
        )

    covered = (
        need is not None
        and need > 0
        and credit >= need
        and commit.allowed
        and ram_guard_ok
        and s_eff is not None
        and swap_status != UNKNOWN
    )
    if need is None:
        status = "credit_insufficient"
    elif need == 0:
        status = "not_needed"
    elif covered:
        status = "credit_sufficient"
    else:
        status = "credit_insufficient"
    return CreditDecision(
        policy=policy,
        a_raw=capacity.a_raw,
        b_ram=b_ram,
        b_admit=b_admit,
        b_ram_eff=b_ram_eff,
        p_gauge_raw=p,
        p_gauge_margined=p_margined,
        s_disk_free=s_disk,
        s_effectif=s_eff,
        s_credit=credit,
        credit_needed_bytes=need,
        covered=covered,
        status=status,
        reasons=tuple(reasons),
        devices=devices,
        cgroup=levels,
        commit=commit,
        cgroup_swap_status=swap_status,
        cgroup_swap_headroom=swap_head,
        cgroup_ram_status=ram_status,
        cgroup_ram_headroom=ram_head,
        ram_guard_np_bytes=int(capacity.non_pageable_new_bytes),
        ram_guard_ok=ram_guard_ok,
        facts=dict(capacity.facts),
    )
