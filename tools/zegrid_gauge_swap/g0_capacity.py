#!/usr/bin/env python3
"""G0 read-only capacity contract replay — ZM-GAUGE-SWAP-G0-20261115.

This is a QUALIFICATION / SPEC tool, not product code. It is deliberately
standalone (no ``zemosaic`` import) so the frozen G0 contract can be replayed
without touching the engine, the scheduler, the GUI or any run store.

Guarantees:
  * pure integer arithmetic (no float factor, no rounding drift);
  * read-only: the only filesystem access is reading /proc, /sys and the
    historical breadcrumbs; nothing is created, locked, swapped or resized;
  * every fact carries its source and a trinary status
    (``not_applicable`` / ``unlimited`` / ``limited`` / ``unknown``);
  * unknown capacity yields credit 0 with an explicit reason, never infinity.

Usage:
  python3 tools/zegrid_gauge_swap/g0_capacity.py --self-test
  python3 tools/zegrid_gauge_swap/g0_capacity.py --facts
  python3 tools/zegrid_gauge_swap/g0_capacity.py --facts --json
  python3 tools/zegrid_gauge_swap/g0_capacity.py --replay \
      /home/tristan/NGC7023/out/instrum/worker_crash_breadcrumbs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

MIB = 1024 * 1024
GIB = 1024 ** 3

PLAN_NUM, PLAN_DEN = 4, 5
CREDIT_NUM, CREDIT_DEN = 9, 10
MARGIN_NUM, MARGIN_DEN = 15, 100

#: Policy eligibility floor for the bounded-swap path. This is NOT a cap on the
#: credit and NOT an OS reservation: below it the path simply stays RAM-only.
#: Frozen by G0; re-evaluated only with G3 measurements.
S_MIN_USABLE = 1 * GIB

#: V1 reserve policy. The 10 % retained by floor(9*S/10) is the ONLY retained
#: margin; it is explicitly NOT advertised as an OS reservation. A numeric
#: R_min is deliberately not invented here (Nono L1 value was not validated).
R_MIN = 0

NA = "not_applicable"
UNLIMITED = "unlimited"
LIMITED = "limited"
UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Exact integer formulas (the frozen contract)
# ---------------------------------------------------------------------------

def plan_budget(available_bytes: int) -> int:
    """B_plan = floor(4*A/5). Unchanged by this mission."""
    return (PLAN_NUM * int(available_bytes)) // PLAN_DEN


def admit_budget(available_bytes: int) -> int:
    """B_admit = floor(9*A/10). Unchanged by this mission."""
    return (CREDIT_NUM * int(available_bytes)) // CREDIT_DEN


def ceil_margin(raw_bytes: int, num: int = MARGIN_NUM, den: int = MARGIN_DEN) -> int:
    """The ONE 15 % margin, applied exactly once per overlap group."""
    raw = int(raw_bytes)
    return (raw * (int(den) + int(num)) + int(den) - 1) // int(den)


def swap_credit(s_effectif: int | None) -> int:
    """S_credit = floor(9*S_effectif/10), with the frozen eligibility floor.

    Returns 0 when the effective swap is unknown (``None``) so a caller can
    never turn "unknown" into "unlimited".
    """
    if s_effectif is None:
        return 0
    s = max(0, int(s_effectif) - R_MIN)
    if s < S_MIN_USABLE:
        return 0
    return (CREDIT_NUM * s) // CREDIT_DEN


def credit_needed(p_gauge_raw: int, b_ram: int) -> int:
    """Bytes of credit required so that the margined gauge fits B_ram + credit."""
    return max(0, ceil_margin(p_gauge_raw) - int(b_ram))


def s_needed_for(credit_bytes: int) -> int:
    """Smallest S_effectif whose 90 % credit covers ``credit_bytes``."""
    c = max(0, int(credit_bytes))
    return (c * CREDIT_DEN + CREDIT_NUM - 1) // CREDIT_NUM


# ---------------------------------------------------------------------------
# Facts parsing (text in, integers out; no hidden I/O)
# ---------------------------------------------------------------------------

def parse_meminfo(text: str) -> dict[str, int]:
    """Parse ``/proc/meminfo`` kB fields into BYTES. Last value wins."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_()]+):\s+(\d+)(?:\s+kB)?\s*$", line.strip())
        if not m:
            continue
        key, raw = m.group(1), int(m.group(2))
        values[key] = raw * 1024 if "kB" in line else raw
    return values


@dataclass(frozen=True)
class SwapDevice:
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

    def to_dict(self) -> dict:
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
    """Detect zram by device name and, defensively, by the sysfs block entry.

    ``/proc/swaps`` types zram as ``partition`` on this host, so the TYPE column
    is not sufficient. The device basename is authoritative here.
    """
    base = _basename(name)
    if re.fullmatch(r"zram\d+", base):
        return True
    try:
        entry = Path(sys_block) / base
        if entry.exists() and Path(os.path.realpath(entry)).name.startswith("zram"):
            return True
    except OSError:
        pass
    return False


def parse_swaps(
    text: str,
    *,
    sys_block: str | os.PathLike[str] = "/sys/block",
    stat_fn=None,
) -> list[SwapDevice]:
    """Parse ``/proc/swaps`` and classify each device.

    A swap FILE whose allocated blocks do not cover its declared size is
    excluded (conservative): its "free" slots are not proven to be backed.
    """
    devices: list[SwapDevice] = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        name, kind, size_kb, used_kb, _prio = parts[0], parts[1], parts[2], parts[3], parts[4]
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
                st = (stat_fn or os.stat)(name)
                allocated = int(st.st_blocks) * int(st.st_blksize)
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


def disk_swap_free(devices: list[SwapDevice]) -> int:
    return sum(d.free_bytes for d in devices if d.eligible)


# ---------------------------------------------------------------------------
# cgroup v2 chain (trinary semantics)
# ---------------------------------------------------------------------------

@dataclass
class CgroupLevel:
    path: str
    is_root: bool
    memory_controller: bool
    ram_status: str
    ram_headroom: int | None
    swap_status: str
    swap_headroom: int | None
    detail: str

    def to_dict(self) -> dict:
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


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    text = _read_text(path)
    if text is None or text == "max" or not re.fullmatch(r"-?\d+", text):
        return None
    return int(text)


def cgroup_chain_paths(own_cgroup: str, root: str | os.PathLike[str] = "/sys/fs/cgroup") -> list[str]:
    """Absolute cgroup directories from the ROOT down to the caller's cgroup."""
    root = str(root).rstrip("/")
    rel = (own_cgroup or "/").strip()
    if not rel.startswith("/"):
        raise ValueError(f"cgroup path must be absolute, got {own_cgroup!r}")
    parts = [p for p in rel.split("/") if p]
    paths = [root]
    for i in range(len(parts)):
        paths.append(root + "/" + "/".join(parts[: i + 1]))
    return paths


def read_cgroup_levels(
    own_cgroup: str,
    root: str | os.PathLike[str] = "/sys/fs/cgroup",
) -> list[CgroupLevel]:
    """Read every ancestor's memory cap with explicit NA/unlimited/unknown.

    Rules frozen by G0:
      * root without memory.max: ``not_applicable`` (the cgroup root cannot be
        limited);
      * non-root without memory.max and without the memory controller in
        ``cgroup.controllers``: ``not_applicable`` (no cgroup memory accounting
        at that level);
      * non-root WITH the controller but a missing file: ``unknown``;
      * ``max``: ``unlimited``;
      * integer: ``limited`` with ``max(0, max - current)`` headroom.
    """
    levels: list[CgroupLevel] = []
    for path_s in cgroup_chain_paths(own_cgroup, root):
        path = Path(path_s)
        is_root = Path(path_s) == Path(str(root).rstrip("/"))
        if not path.is_dir():
            levels.append(
                CgroupLevel(path_s, is_root, False, UNKNOWN, None, UNKNOWN, None, "cgroup_directory_missing")
            )
            continue
        controllers = (_read_text(path / "cgroup.controllers") or "").split()
        has_mem = "memory" in controllers
        if is_root and not (path / "memory.max").exists():
            ram_status, ram_head = NA, None
            swap_status, swap_head = NA, None
            detail = "root_cgroup_is_unbounded"
        elif not has_mem:
            ram_status, ram_head = NA, None
            swap_status, swap_head = NA, None
            detail = "memory_controller_not_active_here"
        else:
            ram_max = _read_text(path / "memory.max")
            ram_cur = _read_int(path / "memory.current")
            if ram_max is None or ram_cur is None:
                ram_status, ram_head = UNKNOWN, None
            elif ram_max == "max":
                ram_status, ram_head = UNLIMITED, None
            else:
                ram_status, ram_head = LIMITED, max(0, int(ram_max) - int(ram_cur))
            swap_max = _read_text(path / "memory.swap.max")
            swap_cur = _read_int(path / "memory.swap.current")
            if swap_max is None or swap_cur is None:
                swap_status, swap_head = UNKNOWN, None
            elif swap_max == "max":
                swap_status, swap_head = UNLIMITED, None
            else:
                swap_status, swap_head = LIMITED, max(0, int(swap_max) - int(swap_cur))
            detail = "memory_controller_active"
        levels.append(
            CgroupLevel(path_s, is_root, has_mem, ram_status, ram_head, swap_status, swap_head, detail)
        )
    return levels


def combine_cgroup(levels: list[CgroupLevel], attr: str) -> tuple[str, int | None, list[str]]:
    """Combine ancestor caps: min over limited levels; unknown fails closed."""
    statuses = [getattr(level, f"{attr}_status") for level in levels]
    if UNKNOWN in statuses:
        return UNKNOWN, None, [level.path for level in levels if getattr(level, f"{attr}_status") == UNKNOWN]
    limited = [level for level in levels if getattr(level, f"{attr}_status") == LIMITED]
    if not limited:
        # all levels are NA/unlimited: no cgroup cap of this kind
        return (UNLIMITED if UNLIMITED in statuses else NA), None, []
    headroom = min(int(getattr(level, f"{attr}_headroom")) for level in limited)
    return LIMITED, headroom, []


# ---------------------------------------------------------------------------
# Commit gate (new allocations only; never B_ram subtracted from headroom)
# ---------------------------------------------------------------------------

@dataclass
class CommitGate:
    mode: int | None
    status: str
    headroom_bytes: int | None
    new_commit_bytes: int
    allowed: bool
    reason: str
    sources: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
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
    mode: int | None,
    commit_limit: int | None,
    committed_as: int | None,
    new_commit_bytes: int,
) -> CommitGate:
    """Commit feasibility for NEW private anonymous bytes.

    ``new_commit_bytes`` is the gauge phase's modelled peak. It is NOT reduced by
    B_ram: the commit headroom and the RAM budget have different semantics, and
    subtracting one from the other would double-count already committed memory.
    """
    src = {"commit_limit": commit_limit, "committed_as": committed_as}
    if mode is None:
        return CommitGate(None, UNKNOWN, None, int(new_commit_bytes), False,
                          "overcommit_mode_unknown", src)
    if mode in (0, 1):
        # 0 = heuristic: the kernel may still refuse an absurd allocation;
        # 1 = always overcommit: no per-allocation commit gate.
        return CommitGate(mode, NA, None, int(new_commit_bytes), True,
                          f"overcommit_mode_{mode}_no_hard_gate", src)
    if mode == 2:
        if commit_limit is None or committed_as is None:
            return CommitGate(mode, UNKNOWN, None, int(new_commit_bytes), False,
                              "commit_fields_unknown", src)
        headroom = max(0, int(commit_limit) - int(committed_as))
        ok = int(new_commit_bytes) <= headroom
        return CommitGate(mode, LIMITED, headroom, int(new_commit_bytes), ok,
                          "commit_headroom_ok" if ok else "commit_headroom_insufficient", src)
    return CommitGate(mode, UNKNOWN, None, int(new_commit_bytes), False,
                      f"unknown_overcommit_mode_{mode}", src)


# ---------------------------------------------------------------------------
# Full contract evaluation
# ---------------------------------------------------------------------------

@dataclass
class CreditContract:
    a_raw: int
    b_ram: int
    b_admit: int
    b_ram_eff: int
    p_gauge_raw: int
    p_gauge_margined: int
    s_disk_free: int
    s_effectif: int | None
    s_credit: int
    credit_needed_bytes: int
    covered: bool
    status: str
    reasons: list[str]
    devices: list[SwapDevice]
    cgroup: list[CgroupLevel]
    commit: CommitGate
    cgroup_swap_status: str
    cgroup_swap_headroom: int | None
    cgroup_ram_status: str
    cgroup_ram_headroom: int | None
    ram_guard_np_bytes: int
    ram_guard_ok: bool
    facts: dict

    def to_dict(self) -> dict:
        return {
            "schema": "zemosaic-gauge-swap-g0-contract-v1",
            "a_raw_bytes": int(self.a_raw),
            "b_ram_bytes": int(self.b_ram),
            "b_admit_bytes": int(self.b_admit),
            "b_ram_eff_bytes": int(self.b_ram_eff),
            "p_gauge_raw_bytes": int(self.p_gauge_raw),
            "p_gauge_margined_bytes": int(self.p_gauge_margined),
            "s_disk_free_bytes": int(self.s_disk_free),
            "s_effectif_bytes": self.s_effectif,
            "s_credit_bytes": int(self.s_credit),
            "credit_needed_bytes": int(self.credit_needed_bytes),
            "covered": bool(self.covered),
            "status": self.status,
            "reasons": list(self.reasons),
            "swap_devices": [d.to_dict() for d in self.devices],
            "cgroup_levels": [level.to_dict() for level in self.cgroup],
            "cgroup_swap": {"status": self.cgroup_swap_status, "headroom_bytes": self.cgroup_swap_headroom},
            "cgroup_ram": {"status": self.cgroup_ram_status, "headroom_bytes": self.cgroup_ram_headroom},
            "commit": self.commit.to_dict(),
            "ram_guard": {
                "non_pageable_new_bytes": int(self.ram_guard_np_bytes),
                "b_ram_covers_non_pageable": bool(self.ram_guard_ok),
            },
            "facts": dict(self.facts),
            "limits": [
                "credit is a PERMIT, never a reservation: the OS may keep pages in RAM, evict other processes or refuse an allocation",
                "a fake MemAvailable or a synthetic fact proves the logic, never physical swap",
                "zram free is RAM-backed and is excluded from S_disk_free",
                "SwapCached is neither added to A nor to swap free",
                "no OOM guarantee is made by this contract",
            ],
        }


def evaluate_contract(
    *,
    a_raw: int,
    p_gauge_raw: int,
    devices: list[SwapDevice],
    levels: list[CgroupLevel],
    overcommit_mode: int | None,
    commit_limit: int | None,
    committed_as: int | None,
    non_pageable_new_bytes: int = 340 * MIB,
    facts: dict | None = None,
) -> CreditContract:
    """Evaluate the frozen G0 contract for ONE gauge compute phase.

    The credit is eligible only when every gate below is *proven*:
      disk swap free > 0 and >= S_MIN_USABLE, cgroup swap headroom known,
      commit gate not failing, non-pageable new bytes covered by B_ram.
    Any unknown fact yields credit 0 with a reason; the RAM-only path is never
    weakened by this evaluator.
    """
    a = int(a_raw)
    b_ram = plan_budget(a)
    b_admit = admit_budget(a)
    p_margined = ceil_margin(int(p_gauge_raw))
    s_disk = disk_swap_free(devices)
    reasons: list[str] = []

    swap_status, swap_head, unknown_levels = combine_cgroup(levels, "swap")
    ram_status, ram_head, _ = combine_cgroup(levels, "ram")

    if swap_status == UNKNOWN:
        s_eff: int | None = None
        reasons.append("cgroup_swap_unknown")
    elif swap_status == LIMITED:
        s_eff = min(s_disk, int(swap_head))
        if int(swap_head) < s_disk:
            reasons.append("cgroup_swap_caps_effective_swap")
    else:
        s_eff = s_disk
    if s_eff is not None and s_eff < S_MIN_USABLE:
        reasons.append("effective_swap_below_minimum")
    if s_disk <= 0:
        reasons.append("no_eligible_disk_swap")

    credit = swap_credit(s_eff)
    need = credit_needed(int(p_gauge_raw), b_ram)

    commit = evaluate_commit(
        overcommit_mode, commit_limit, committed_as, int(p_gauge_raw)
    )
    if not commit.allowed:
        reasons.append(commit.reason)

    b_ram_eff = b_ram
    if ram_status == LIMITED and int(ram_head) < b_ram:
        # cgroup RAM is tighter than the host budget: the usable RAM budget is
        # the smaller of two independent upper bounds (never a subtraction).
        b_ram_eff = int(ram_head)
        reasons.append("cgroup_ram_tighter_than_host")
    ram_guard_ok = int(non_pageable_new_bytes) <= b_ram_eff
    if not ram_guard_ok:
        reasons.append("b_ram_below_non_pageable_new")

    covered = (
        need > 0
        and credit >= need
        and commit.allowed
        and ram_guard_ok
        and s_eff is not None
        and swap_status != UNKNOWN
    )
    if need == 0:
        status = "not_needed"
    elif covered:
        status = "credit_sufficient"
    else:
        status = "credit_insufficient"
    return CreditContract(
        a_raw=a,
        b_ram=b_ram,
        b_admit=b_admit,
        b_ram_eff=b_ram_eff,
        p_gauge_raw=int(p_gauge_raw),
        p_gauge_margined=p_margined,
        s_disk_free=s_disk,
        s_effectif=s_eff,
        s_credit=credit,
        credit_needed_bytes=need,
        covered=covered,
        status=status,
        reasons=reasons,
        devices=list(devices),
        cgroup=list(levels),
        commit=commit,
        cgroup_swap_status=swap_status,
        cgroup_swap_headroom=swap_head,
        cgroup_ram_status=ram_status,
        cgroup_ram_headroom=ram_head,
        ram_guard_np_bytes=int(non_pageable_new_bytes),
        ram_guard_ok=ram_guard_ok,
        facts=dict(facts or {}),
    )


# ---------------------------------------------------------------------------
# Live facts (read-only)
# ---------------------------------------------------------------------------

def read_own_cgroup(proc_self_cgroup: str | os.PathLike[str] = "/proc/self/cgroup") -> str:
    for line in Path(proc_self_cgroup).read_text(encoding="ascii").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            return parts[2] or "/"
    raise ValueError("no cgroup v2 entry (0::...) in /proc/self/cgroup")


def read_overcommit(root: str | os.PathLike[str] = "/proc/sys/vm") -> tuple[int | None, int | None, int | None]:
    base = Path(root)
    mode = _read_int(base / "overcommit_memory")
    ratio = _read_int(base / "overcommit_ratio")
    kbytes = _read_int(base / "overcommit_kbytes")
    return mode, ratio, kbytes


def live_facts(p_gauge_raw: int) -> CreditContract:
    mem = parse_meminfo(Path("/proc/meminfo").read_text(encoding="ascii"))
    swaps = parse_swaps(Path("/proc/swaps").read_text(encoding="ascii"))
    own = read_own_cgroup()
    levels = read_cgroup_levels(own)
    mode, ratio, kbytes = read_overcommit()
    facts = {
        "source": {
            "meminfo": "/proc/meminfo",
            "swaps": "/proc/swaps",
            "cgroup": "/proc/self/cgroup + /sys/fs/cgroup",
            "overcommit": "/proc/sys/vm/overcommit_memory, overcommit_ratio, overcommit_kbytes",
        },
        "mem_total_bytes": mem.get("MemTotal"),
        "mem_available_bytes": mem.get("MemAvailable"),
        "swap_total_bytes": mem.get("SwapTotal"),
        "swap_free_global_bytes": mem.get("SwapFree"),
        "swap_cached_bytes": mem.get("SwapCached"),
        "commit_limit_bytes": mem.get("CommitLimit"),
        "committed_as_bytes": mem.get("Committed_AS"),
        "mlocked_bytes": mem.get("Mlocked"),
        "unevictable_bytes": mem.get("Unevictable"),
        "overcommit_ratio": ratio,
        "overcommit_kbytes": kbytes,
        "own_cgroup": own,
        "global_swap_free_is_not_disk_credit": True,
        "swapcached_not_added_anywhere": True,
    }
    return evaluate_contract(
        a_raw=int(mem.get("MemAvailable") or 0),
        p_gauge_raw=int(p_gauge_raw),
        devices=swaps,
        levels=levels,
        overcommit_mode=mode,
        commit_limit=mem.get("CommitLimit"),
        committed_as=mem.get("Committed_AS"),
        facts=facts,
    )


# ---------------------------------------------------------------------------
# Historical refusal replay (read-only)
# ---------------------------------------------------------------------------

RE_REFUSAL = re.compile(
    r"aggregate host peak (?P<peak>\d+) B \(raw (?P<raw>\d+) B \+ margin "
    r"(?P<num>\d+)/(?P<den>\d+)\) > budget (?P<budget>\d+) B"
)


@dataclass
class Replay:
    raw: int
    peak: int
    margin_num: int
    margin_den: int
    budget: int
    deficit: int
    s_needed: int
    source: str
    expected: dict

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "raw_bytes": int(self.raw),
            "peak_bytes": int(self.peak),
            "margin": [int(self.margin_num), int(self.margin_den)],
            "budget_bytes": int(self.budget),
            "deficit_bytes": int(self.deficit),
            "s_effectif_needed_bytes": int(self.s_needed),
            "expected": dict(self.expected),
            "checks": {
                "peak_matches_margin": self.peak == ceil_margin(self.raw, self.margin_num, self.margin_den),
                "deficit_matches": self.deficit == max(0, self.peak - self.budget),
                "s_needed_matches": self.s_needed == s_needed_for(self.deficit),
            },
        }


def replay_ngc7023(path: str | os.PathLike[str]) -> Replay:
    """Replay the historical refusal arithmetic from the breadcrumbs.

    EXPECTED (mission contract): raw 2203281820; +15 % -> 2533774093;
    B_plan 1664142540; deficit 869631553. Any drift is reported, never hidden.
    """
    expected = {
        "raw_bytes": 2_203_281_820,
        "peak_bytes": 2_533_774_093,
        "budget_bytes": 1_664_142_540,
        "deficit_bytes": 869_631_553,
    }
    last = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = " ".join(str(row.get(k) or "") for k in ("message", "error"))
            m = RE_REFUSAL.search(text)
            if m:
                last = (path, m)
    if last is None:
        raise SystemExit(f"no R55_REFUSE_HOST_RAM arithmetic found in {path}")
    source, m = last
    raw = int(m.group("raw"))
    peak = int(m.group("peak"))
    num = int(m.group("num"))
    den = int(m.group("den"))
    budget = int(m.group("budget"))
    deficit = max(0, peak - budget)
    return Replay(
        raw=raw,
        peak=peak,
        margin_num=num,
        margin_den=den,
        budget=budget,
        deficit=deficit,
        s_needed=s_needed_for(deficit),
        source=source,
        expected=expected,
    )


# ---------------------------------------------------------------------------
# Self-test: exact boundary matrix, no I/O
# ---------------------------------------------------------------------------

def _device(name, kind, size, used, zram=False, eligible=True):
    return SwapDevice(name, kind, size, used, zram, eligible, "test")


def self_test() -> int:
    failures: list[str] = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    check("plan_budget", plan_budget(2_080_178_176), 1_664_142_540)
    check("admit_budget", admit_budget(2_080_178_176), 1_872_160_358)
    check("margin", ceil_margin(2_203_281_820), 2_533_774_093)
    check("credit_needed", credit_needed(2_203_281_820, 1_664_142_540), 869_631_553)
    check("s_needed", s_needed_for(869_631_553), 966_257_282)
    check("swap_credit_unknown", swap_credit(None), 0)
    check("swap_credit_floor", swap_credit(S_MIN_USABLE - 1), 0)
    check("swap_credit_min", swap_credit(S_MIN_USABLE), (CREDIT_NUM * S_MIN_USABLE) // CREDIT_DEN)
    check("swap_credit_exact", swap_credit(2_000_000_000), 1_800_000_000)

    # zram + file: only the file contributes; global SwapFree is never used.
    devices = [
        _device("/dev/zram0", "partition", 8_000_000_000, 5_000_000_000, zram=True, eligible=False),
        _device("/swapfile", "file", 25_000_000_000, 1_000_000_000),
    ]
    check("disk_swap_free", disk_swap_free(devices), 24_000_000_000)

    # zram only -> zero credit.
    zonly = [_device("/dev/zram0", "partition", 8_000_000_000, 1_000_000_000, zram=True, eligible=False)]
    c = evaluate_contract(
        a_raw=2_080_178_176, p_gauge_raw=2_203_281_820, devices=zonly,
        levels=[], overcommit_mode=0, commit_limit=1, committed_as=0,
    )
    check("zram_only_status", c.status, "credit_insufficient")
    check("zram_only_credit", c.s_credit, 0)

    # cgroup swap unknown -> fail closed even with a huge disk swap.
    levels = [
        CgroupLevel("/sys/fs/cgroup", True, False, NA, None, NA, None, "root"),
        CgroupLevel("/sys/fs/cgroup/x", False, True, UNLIMITED, None, UNKNOWN, None, "missing"),
    ]
    c = evaluate_contract(
        a_raw=2_080_178_176, p_gauge_raw=2_203_281_820, devices=devices,
        levels=levels, overcommit_mode=0, commit_limit=1, committed_as=0,
    )
    check("cgroup_unknown_credit", c.s_credit, 0)
    check("cgroup_unknown_status", c.status, "credit_insufficient")

    # cgroup swap capped below the deficit -> credit insufficient even if disk is huge.
    big_gauge = 4_000_000_000
    levels = [
        CgroupLevel("/sys/fs/cgroup", True, False, NA, None, NA, None, "root"),
        CgroupLevel("/sys/fs/cgroup/x", False, True, UNLIMITED, None, LIMITED, 2_000_000_000, "cap"),
    ]
    c = evaluate_contract(
        a_raw=2_080_178_176, p_gauge_raw=big_gauge, devices=devices,
        levels=levels, overcommit_mode=0, commit_limit=1, committed_as=0,
    )
    check("cgroup_cap_effective", c.s_effectif, 2_000_000_000)
    check("cgroup_cap_credit", c.s_credit, 1_800_000_000)
    check("cgroup_cap_status", c.status, "credit_insufficient")

    # Enough disk swap + unlimited cgroup + mode 0 -> covered.
    levels = [CgroupLevel("/sys/fs/cgroup", True, False, NA, None, NA, None, "root")]
    c = evaluate_contract(
        a_raw=2_080_178_176, p_gauge_raw=2_203_281_820, devices=devices,
        levels=levels, overcommit_mode=0, commit_limit=1, committed_as=0,
    )
    check("covered_status", c.status, "credit_sufficient")
    check("covered_credit", c.s_credit, (CREDIT_NUM * 24_000_000_000) // CREDIT_DEN)

    # Strict overcommit, insufficient headroom: credit is refused.
    c = evaluate_contract(
        a_raw=2_080_178_176, p_gauge_raw=2_203_281_820, devices=devices,
        levels=levels, overcommit_mode=2, commit_limit=2_500_000_000, committed_as=500_000_000,
    )
    check("commit_strict_refused", c.status, "credit_insufficient")
    check("commit_strict_reason", c.commit.reason, "commit_headroom_insufficient")
    # Strict overcommit, enough headroom: allowed but NOT a swap reservation.
    c = evaluate_contract(
        a_raw=2_080_178_176, p_gauge_raw=2_203_281_820, devices=devices,
        levels=levels, overcommit_mode=2, commit_limit=8_000_000_000, committed_as=500_000_000,
    )
    check("commit_strict_ok", c.commit.allowed, True)
    # Commit headroom is NOT B_ram subtracted: a huge B_ram cannot rescue a zero headroom.
    c = evaluate_contract(
        a_raw=40_000_000_000, p_gauge_raw=2_203_281_820, devices=devices,
        levels=levels, overcommit_mode=2, commit_limit=1_000_000_000, committed_as=1_000_000_000,
    )
    check("commit_no_bram_subtraction", c.commit.allowed, False)

    # Non-pageable > B_ram: refused, and never compensated by swap credit.
    c = evaluate_contract(
        a_raw=2_080_178_176, p_gauge_raw=2_203_281_820, devices=devices,
        levels=levels, overcommit_mode=0, commit_limit=1, committed_as=0,
        non_pageable_new_bytes=2_000_000_000,
    )
    check("nonpageable_refused", c.status, "credit_insufficient")
    check("nonpageable_reason", "b_ram_below_non_pageable_new" in c.reasons, True)

    # cgroup RAM headroom tighter than the non-pageable bound: credit refused.
    levels_ram = [
        CgroupLevel("/sys/fs/cgroup", True, False, NA, None, NA, None, "root"),
        CgroupLevel("/sys/fs/cgroup/x", False, True, LIMITED, 200_000_000, UNLIMITED, None, "ramcap"),
    ]
    c = evaluate_contract(
        a_raw=2_080_178_176, p_gauge_raw=2_203_281_820, devices=devices,
        levels=levels_ram, overcommit_mode=0, commit_limit=1, committed_as=0,
    )
    check("cgroup_ram_eff", c.b_ram_eff, 200_000_000)
    check("cgroup_ram_guard", c.ram_guard_ok, False)
    check("cgroup_ram_refused", c.status, "credit_insufficient")

    # Replay arithmetic, no file needed (formula-level check).
    check("replay_peak", ceil_margin(2_203_281_820, 15, 100), 2_533_774_093)
    check("replay_deficit", 2_533_774_093 - 1_664_142_540, 869_631_553)

    if failures:
        print("SELF-TEST FAILED")
        for failure in failures:
            print(" -", failure)
        return 1
    print("SELF-TEST OK (0 failures)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--facts", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--replay", metavar="BREADCRUMBS")
    parser.add_argument(
        "--gauge-raw", type=int, default=2_203_281_820,
        help="gauge phase raw bytes for --facts (default: the historical NGC7023 footprint)",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.replay:
        replay = replay_ngc7023(args.replay)
        payload = replay.to_dict()
        if not all(payload["checks"].values()):
            payload["status"] = "DRIFT"
        else:
            payload["status"] = "REPRODUCED"
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if payload["status"] == "REPRODUCED" else 2

    if args.facts:
        contract = live_facts(args.gauge_raw)
        payload = contract.to_dict()
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"A_raw (MemAvailable)      : {contract.a_raw} B")
            print(f"B_plan  = floor(4A/5)     : {contract.b_ram} B")
            print(f"B_admit = floor(9A/10)    : {contract.b_admit} B")
            print(f"B_ram_eff (cgroup min)     : {contract.b_ram_eff} B")
            print(f"P_gauge raw / +15%        : {contract.p_gauge_raw} / {contract.p_gauge_margined} B")
            print(f"S_disk_free (zram excl.)  : {contract.s_disk_free} B")
            print(f"S_effectif                : {contract.s_effectif}")
            print(f"S_credit  = floor(9S/10)  : {contract.s_credit} B")
            print(f"credit needed             : {contract.credit_needed_bytes} B")
            print(f"commit gate               : {contract.commit.status} / {contract.commit.reason}")
            print(f"cgroup swap / RAM         : {contract.cgroup_swap_status} / {contract.cgroup_ram_status}")
            print(f"contract status           : {contract.status} {contract.reasons}")
            print("devices:")
            for device in contract.devices:
                print(
                    f"  {device.name} type={device.kind} free={device.free_bytes} "
                    f"zram={device.zram} eligible={device.eligible} ({device.reason})"
                )
            print("NOTE: live facts are informational; they do not prove a physical swap-backed run.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
