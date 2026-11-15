"""G1a — pure Linux gauge swap-capacity provider tests (schema ``gauge-swap-capacity-v1``).

Scope: :mod:`zemosaic.core.zegrid.gauge_swap_capacity` ONLY. No product call
site is imported or executed. Every fact is injected through real temporary
files (meminfo text, swaps text, cgroup tree) or through an injected ``stat_fn``,
so no test depends on this host's ``/proc``.

The behavioral oracle is ``tools/zegrid_gauge_swap/g0_capacity.py`` (G0 frozen
contract): the module must return the same decision values and the same reason
ordering on the same facts.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
G0_TOOL = REPO_ROOT / "tools" / "zegrid_gauge_swap" / "g0_capacity.py"
MODULE_PATH = SRC / "zemosaic" / "core" / "zegrid" / "gauge_swap_capacity.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import gauge_swap_capacity as cap  # noqa: E402

MIB = 1024 ** 2
GIB = 1024 ** 3

# The historical NGC7023 facts (arithmetic only; not a live observation).
HISTORICAL_A = 2_080_178_176
HISTORICAL_P = 2_203_281_820
HISTORICAL_B_RAM = 1_664_142_540
HISTORICAL_NEED = 869_631_553
HISTORICAL_S_NEEDED = 966_257_282

BASELINE_MEMINFO = {
    "MemTotal": 8_000_000,
    "MemAvailable": 2_031_424,  # 2 080 178 176 B -> floor(4*A/5) == historical B_plan
    "SwapTotal": 30_000_000,
    "SwapFree": 26_000_000,
    "SwapCached": 992_344,
    "CommitLimit": 36_988_548,
    "Committed_AS": 19_448_896,
}


# ---------------------------------------------------------------------------
# Helpers: text files, cgroup trees, G0 oracle
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="ascii")
    return path.read_text(encoding="ascii")


def _meminfo_text(fields: dict[str, int]) -> str:
    return "".join(f"{key}: {value} kB\n" for key, value in fields.items())


def _swaps_text(rows: list[tuple[str, str, int, int]]) -> str:
    lines = ["Filename\t\t\t\tType\t\tSize\t\tUsed\t\tPriority"]
    for name, kind, size_kb, used_kb in rows:
        lines.append(f"{name}\t{kind}\t{size_kb}\t{used_kb}\t10")
    return "\n".join(lines) + "\n"


def _stat_ok(name: str) -> SimpleNamespace:
    # st_blocks is in 512-byte units: 50M * 512 == 25.6 GB, so the 25 GB
    # swapfile below is fully backed (L1: st_blksize is NOT the unit).
    return SimpleNamespace(st_blocks=50_000_000, st_blksize=4096)


def _cgroup_level(
    root: Path,
    rel: str,
    *,
    controllers: str | None = "memory",
    memory_max: str | None = None,
    memory_current: int | None = None,
    swap_max: str | None = None,
    swap_current: int | None = None,
) -> Path:
    directory = root / rel if rel else root
    directory.mkdir(parents=True, exist_ok=True)
    if controllers is not None:
        (directory / "cgroup.controllers").write_text(controllers + "\n", encoding="ascii")
    for name, value in (
        ("memory.max", memory_max),
        ("memory.current", None if memory_current is None else str(memory_current)),
        ("memory.swap.max", swap_max),
        ("memory.swap.current", None if swap_current is None else str(swap_current)),
    ):
        if value is not None:
            (directory / name).write_text(value, encoding="ascii")
    return directory


def _baseline_devices() -> list[cap.SwapDevice]:
    text = _swaps_text(
        [
            ("/dev/zram0", "partition", 8_000_000, 5_000_000),
            ("/swapfile", "file", 25_000_000, 1_000_000),
        ]
    )
    return cap.read_swap_devices(text, _stat_ok)


def _root_na() -> cap.CgroupLevel:
    return cap.CgroupLevel(
        "/sys/fs/cgroup", True, False, cap.NA, None, cap.NA, None,
        "root_cgroup_is_unbounded",
    )


def _decision(
    *,
    a_raw: int | None = HISTORICAL_A,
    p_gauge_raw: int = HISTORICAL_P,
    devices=None,
    levels=None,
    overcommit: cap.CommitFacts | None = None,
    non_pageable_new_bytes: int = cap.PROCESS_BASELINE_BYTES,
    policy: str = cap.POLICY_BOUNDED_SWAP,
) -> cap.CreditDecision:
    if devices is None:
        devices = _baseline_devices()
    if levels is None:
        levels = [_root_na()]
    if overcommit is None:
        overcommit = cap.read_overcommit(0, 50, 0)
    return cap.evaluate_credit(
        cap.GaugeSwapCapacity(
            a_raw=a_raw,
            devices=devices,
            cgroup_levels=levels,
            overcommit=overcommit,
            non_pageable_new_bytes=non_pageable_new_bytes,
        ),
        p_gauge_raw,
        policy,
    )


def _s_for_credit(credit: int) -> int:
    """Smallest effective swap whose floor(9*S/10) equals ``credit``."""
    s = (credit * 10 + 8) // 9
    while cap.swap_credit(s) > credit:
        s -= 1
    while cap.swap_credit(s) < credit:
        s += 1
    assert cap.swap_credit(s) == credit
    return s


def _load_g0():
    spec = importlib.util.spec_from_file_location("zegrid_g0_capacity", G0_TOOL)
    module = importlib.util.module_from_spec(spec)
    # dataclasses with ``from __future__ import annotations`` need the module
    # registered before execution.
    sys.modules["zegrid_g0_capacity"] = module
    spec.loader.exec_module(module)
    return module


G0 = _load_g0()


# ---------------------------------------------------------------------------
# meminfo
# ---------------------------------------------------------------------------

def test_meminfo_parsing_units_and_last_wins(tmp_path):
    text = _write(
        tmp_path,
        "meminfo.txt",
        "MemTotal:        100 kB\n"
        "MemAvailable:    200 kB\n"
        "HugePages_Total: 7\n"
        "garbage line without colon\n"
        "MemAvailable:    300 kB\n",
    )
    values = cap.read_meminfo_text(text)
    assert values["MemTotal"] == 100 * 1024
    assert values["MemAvailable"] == 300 * 1024  # last value wins
    assert values["HugePages_Total"] == 7        # no kB -> raw bytes
    assert cap.read_meminfo_text(text) == G0.parse_meminfo(text)


def test_meminfo_rejects_non_text():
    with pytest.raises(TypeError):
        cap.read_meminfo_text(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# /proc/swaps: zram, files, backing proof
# ---------------------------------------------------------------------------

def test_zram_only_yields_zero_credit(tmp_path):
    text = _write(
        tmp_path,
        "swaps.txt",
        _swaps_text([("/dev/zram0", "partition", 8_000_000, 5_000_000)]),
    )
    devices = cap.read_swap_devices(text, _stat_ok)
    assert len(devices) == 1
    assert devices[0].zram is True
    assert devices[0].eligible is False
    assert devices[0].reason == "zram_is_ram_backed"
    assert cap.disk_swap_free(devices) == 0

    decision = _decision(devices=devices)
    assert decision.s_disk_free == 0
    assert decision.s_credit == 0
    assert decision.status == "credit_insufficient"
    assert "no_eligible_disk_swap" in decision.reasons
    assert decision.covered is False


def test_zram_plus_file_counts_only_the_file(tmp_path):
    text = _write(
        tmp_path,
        "swaps.txt",
        _swaps_text(
            [
                ("/dev/zram0", "partition", 8_000_000, 1_000_000),
                ("/swapfile", "file", 25_000_000, 1_000_000),
            ]
        ),
    )
    devices = cap.read_swap_devices(text, _stat_ok)
    assert [device.eligible for device in devices] == [False, True]
    expected = (25_000_000 - 1_000_000) * 1024
    assert cap.disk_swap_free(devices) == expected
    decision = _decision(devices=devices)
    assert decision.s_disk_free == expected
    assert decision.s_credit == (9 * expected) // 10
    assert decision.status == "credit_sufficient"


def test_global_swap_free_is_never_the_capacity(tmp_path):
    text = _write(
        tmp_path,
        "swaps.txt",
        _swaps_text([("/swapfile", "file", 25_000_000, 1_000_000)]),
    )
    devices = cap.read_swap_devices(text, _stat_ok)
    lots = dict(BASELINE_MEMINFO, SwapFree=300_000_000)
    little = dict(BASELINE_MEMINFO, SwapFree=0)
    mem_lots = cap.read_meminfo_text(_write(tmp_path, "a.txt", _meminfo_text(lots)))
    mem_little = cap.read_meminfo_text(_write(tmp_path, "b.txt", _meminfo_text(little)))
    # Same A, same devices: global SwapFree changes nothing in the decision.
    assert mem_lots["SwapFree"] != mem_little["SwapFree"]
    expected = (25_000_000 - 1_000_000) * 1024
    assert _decision(devices=devices).s_disk_free == expected
    assert _decision(devices=devices).s_disk_free != mem_lots["SwapFree"]
    assert _decision(devices=devices).s_disk_free != mem_little["SwapFree"]


def test_sparse_swapfile_is_excluded_but_allocated_file_is_kept(tmp_path):
    sparse = tmp_path / "sparse.swap"
    with open(sparse, "wb") as handle:
        handle.truncate(MIB)
    dense = tmp_path / "dense.swap"
    with open(dense, "wb") as handle:
        handle.write(b"\0" * (4096 * 64))

    sparse_text = _swaps_text([(str(sparse), "file", MIB // 1024, 0)])
    dense_text = _swaps_text([(str(dense), "file", (4096 * 64) // 1024, 0)])

    sparse_devices = cap.read_swap_devices(sparse_text)
    assert sparse_devices[0].eligible is False
    assert sparse_devices[0].reason == "swapfile_not_fully_allocated"
    assert cap.disk_swap_free(sparse_devices) == 0

    dense_devices = cap.read_swap_devices(dense_text)
    assert dense_devices[0].eligible is True
    assert cap.disk_swap_free(dense_devices) == 4096 * 64


def test_swapfile_stat_failure_is_excluded(tmp_path):
    text = _write(
        tmp_path,
        "swaps.txt",
        _swaps_text([("/missing/swapfile", "file", 1_000_000, 0)]),
    )

    def broken(_name):
        raise FileNotFoundError("gone")

    devices = cap.read_swap_devices(text, broken)
    assert devices[0].eligible is False
    assert devices[0].reason == "swapfile_stat_failed: FileNotFoundError"


def test_unsupported_swap_type_is_excluded(tmp_path):
    text = _write(tmp_path, "swaps.txt", _swaps_text([("/dev/loop9", "weird", 1000, 0)]))
    devices = cap.read_swap_devices(text, _stat_ok)
    assert devices[0].eligible is False
    assert devices[0].reason == "unsupported_swap_type:weird"


def test_zram_detected_through_sysfs_symlink(tmp_path):
    sys_block = tmp_path / "sys_block"
    (sys_block / "zram1").mkdir(parents=True)
    (sys_block / "zd0").symlink_to(sys_block / "zram1")
    assert cap.is_zram_name("/dev/zd0", sys_block) is True
    assert cap.is_zram_name("/dev/sda2", sys_block) is False
    assert cap.is_zram_name("/dev/zram7", sys_block) is True


# ---------------------------------------------------------------------------
# SwapCached is parsed but never used
# ---------------------------------------------------------------------------

def test_swapcached_is_ignored_everywhere(tmp_path):
    with_cached = cap.read_meminfo_text(
        _write(tmp_path, "a.txt", _meminfo_text(BASELINE_MEMINFO))
    )
    without = dict(BASELINE_MEMINFO, SwapCached=0)
    without_cached = cap.read_meminfo_text(_write(tmp_path, "b.txt", _meminfo_text(without)))
    assert with_cached["SwapCached"] == 992_344 * 1024
    assert without_cached["SwapCached"] == 0
    assert _decision().to_json() == _decision().to_json()
    decision = _decision()
    assert decision.s_credit == (9 * decision.s_disk_free) // 10
    assert 992_344 * 1024 not in (
        decision.s_disk_free, decision.s_credit, decision.a_raw,
    )
    assert "SwapCached" not in json.dumps(decision.to_dict()["swap_devices"])


# ---------------------------------------------------------------------------
# cgroup v2 chain
# ---------------------------------------------------------------------------

def _read_levels(tmp_path, own: str = "/work/run"):
    root = tmp_path / "cgroup"
    return cap.read_cgroup_levels(own, root), root


def test_cgroup_unknown_swap_fails_closed(tmp_path):
    root = tmp_path / "cgroup"
    _cgroup_level(root, "", controllers="memory")
    _cgroup_level(
        root, "run", controllers="memory",
        memory_max="max", memory_current=0,
        swap_max=None, swap_current=None,
    )
    levels = cap.read_cgroup_levels("/run", root)
    assert [level.swap_status for level in levels] == [cap.NA, cap.UNKNOWN]
    combined = cap.combine_cgroup(levels, "swap")
    assert combined.status == cap.UNKNOWN
    assert combined.unknown_paths == [str(root / "run")]

    decision = _decision(levels=levels)
    assert decision.s_credit == 0
    assert decision.covered is False
    assert decision.status == "credit_insufficient"
    assert "cgroup_swap_unknown" in decision.reasons


def test_cgroup_swap_cap_reduces_effective_swap(tmp_path):
    root = tmp_path / "cgroup"
    _cgroup_level(root, "", controllers="memory")
    _cgroup_level(
        root, "run", controllers="memory",
        memory_max="max", memory_current=0,
        swap_max="2000000000", swap_current="0",
    )
    levels = cap.read_cgroup_levels("/run", root)
    assert levels[-1].swap_status == cap.LIMITED
    assert levels[-1].swap_headroom == 2_000_000_000

    decision = _decision(levels=levels)
    assert decision.s_effectif == 2_000_000_000
    assert decision.s_credit == 1_800_000_000
    assert decision.s_disk_free == 24_576_000_000
    assert "cgroup_swap_caps_effective_swap" in decision.reasons

    at_cap = _decision(levels=levels, p_gauge_raw=2_000_000_000)
    assert at_cap.credit_needed_bytes is not None
    assert at_cap.credit_needed_bytes <= 1_800_000_000
    assert at_cap.status == "credit_sufficient"
    over_cap = _decision(levels=levels, p_gauge_raw=5_000_000_000)
    assert over_cap.credit_needed_bytes is not None
    assert over_cap.credit_needed_bytes > 1_800_000_000
    assert over_cap.status == "credit_insufficient"


def test_cgroup_swap_max_without_current_is_unknown(tmp_path):
    root = tmp_path / "cgroup"
    _cgroup_level(root, "", controllers="memory")
    _cgroup_level(
        root, "run", controllers="memory",
        memory_max="max", memory_current=0,
        swap_max="2000000000", swap_current=None,
    )
    levels = cap.read_cgroup_levels("/run", root)
    assert levels[-1].swap_status == cap.UNKNOWN
    assert _decision(levels=levels).s_credit == 0


def test_root_without_memory_max_is_not_applicable(tmp_path):
    root = tmp_path / "cgroup"
    _cgroup_level(root, "", controllers="memory")  # no memory.max -> NA
    _cgroup_level(
        root, "run", controllers="memory",
        memory_max="max", memory_current=0,
        swap_max="max", swap_current=0,
    )
    levels = cap.read_cgroup_levels("/run", root)
    assert levels[0].is_root is True
    assert levels[0].ram_status == cap.NA
    assert levels[0].swap_status == cap.NA
    assert levels[0].detail == "root_cgroup_is_unbounded"
    combined = cap.combine_cgroup(levels, "swap")
    assert combined.status == cap.UNLIMITED
    decision = _decision(levels=levels)
    assert decision.cgroup_swap_status == cap.UNLIMITED
    assert decision.s_effectif == decision.s_disk_free
    assert decision.status == "credit_sufficient"


def test_missing_root_directory_is_unknown(tmp_path):
    levels = cap.read_cgroup_levels("/work/run", tmp_path / "absent")
    assert levels[0].detail == "cgroup_directory_missing"
    assert levels[0].swap_status == cap.UNKNOWN
    assert _decision(levels=levels).s_credit == 0
    assert "cgroup_swap_unknown" in _decision(levels=levels).reasons


def test_memory_controller_disabled_is_not_applicable(tmp_path):
    root = tmp_path / "cgroup"
    _cgroup_level(root, "", controllers="cpu io")
    # Truly absent controller: cgroup.controllers is readable and omits memory,
    # and NO cap file exists at that level (M3: a cap file would be authoritative).
    _cgroup_level(root, "run", controllers="cpu")
    levels = cap.read_cgroup_levels("/run", root)
    assert [level.detail for level in levels] == [
        "root_cgroup_is_unbounded",
        "memory_controller_not_active_here",
    ]
    assert cap.combine_cgroup(levels, "swap").status == cap.NA
    decision = _decision(levels=levels)
    assert decision.s_effectif == decision.s_disk_free
    assert decision.status == "credit_sufficient"


# ---------------------------------------------------------------------------
# Rework-1: M3 fail-closed detection, L1 backing units, L2 malformed facts,
# L5 self-cgroup parsing, L7 margin parity
# ---------------------------------------------------------------------------

def test_m3_cap_file_presence_is_authoritative_over_empty_controllers(tmp_path):
    root = tmp_path / "cgroup"
    _cgroup_level(root, "", controllers="memory")
    _cgroup_level(
        root, "run", controllers="",
        memory_max="1000", memory_current="900",
    )
    levels = cap.read_cgroup_levels("/run", root)
    level = levels[-1]
    assert level.memory_controller is True
    assert level.ram_status == cap.LIMITED
    assert level.ram_headroom == 100
    assert level.ram_status != cap.NA
    # And the cap really reaches the credit path through combine_cgroup.
    assert cap.combine_cgroup(levels, "ram").status == cap.LIMITED


def test_m3_unreadable_controllers_without_cap_files_is_unknown(tmp_path):
    root = tmp_path / "cgroup"
    _cgroup_level(root, "", controllers="memory")
    _cgroup_level(root, "run", controllers=None)  # dir exists, no controllers, no caps
    levels = cap.read_cgroup_levels("/run", root)
    level = levels[-1]
    assert level.detail == "memory_controller_undetectable"
    assert level.ram_status == cap.UNKNOWN
    assert level.swap_status == cap.UNKNOWN
    assert cap.combine_cgroup(levels, "swap").status == cap.UNKNOWN
    assert _decision(levels=levels).s_credit == 0


def test_m3_missing_swap_cap_file_is_unknown_on_an_active_controller(tmp_path):
    root = tmp_path / "cgroup"
    _cgroup_level(root, "", controllers="memory")
    _cgroup_level(
        root, "run", controllers="",
        memory_max="max", memory_current=0,
    )
    levels = cap.read_cgroup_levels("/run", root)
    level = levels[-1]
    assert level.ram_status == cap.UNLIMITED
    assert level.swap_status == cap.UNKNOWN   # expected file absent


def test_l2_malformed_cap_value_is_unknown_not_an_exception(tmp_path):
    root = tmp_path / "cgroup"
    _cgroup_level(root, "", controllers="memory")
    _cgroup_level(
        root, "run", controllers="",
        memory_max="garbage", memory_current=0,
        swap_max="garbage", swap_current=0,
    )
    levels = cap.read_cgroup_levels("/run", root)  # must not raise
    level = levels[-1]
    assert level.ram_status == cap.UNKNOWN
    assert level.swap_status == cap.UNKNOWN


def test_l2_undecodable_cap_value_is_unknown(tmp_path):
    root = tmp_path / "cgroup"
    _cgroup_level(root, "", controllers="memory")
    directory = _cgroup_level(root, "run", controllers="")
    (directory / "memory.max").write_bytes(b"\xff\xfe\n")
    (directory / "memory.current").write_text("0", encoding="ascii")
    levels = cap.read_cgroup_levels("/run", root)  # must not raise
    assert levels[-1].ram_status == cap.UNKNOWN


def test_l1_swapfile_st_blocks_are_512_byte_units():
    size_kb = 100_000
    size_b = size_kb * 1024
    text = _swaps_text([("/swapfile", "file", size_kb, 0)])

    def half_backed(_name):
        # st_blocks == size/1024 in 512-byte units => ~50 % allocated.
        return SimpleNamespace(st_blocks=size_b // 1024, st_blksize=4096)

    half = cap.read_swap_devices(text, half_backed)
    assert half[0].eligible is False
    assert half[0].reason == "swapfile_not_fully_allocated"

    def fully_backed(_name):
        return SimpleNamespace(st_blocks=size_b // 512, st_blksize=4096)

    full = cap.read_swap_devices(text, fully_backed)
    assert full[0].eligible is True
    assert full[0].reason == "disk_backed"


def test_l5_parse_self_cgroup_v2_line():
    text = "0::/user.slice/user-1000.slice/session-2.scope\n"
    assert cap.parse_self_cgroup(text) == "/user.slice/user-1000.slice/session-2.scope"


def test_l5_parse_self_cgroup_ignores_v1_lines():
    text = (
        "12:cpuset:/legacy\n"
        "4:memory:/user.slice/legacy\n"
        "0::/user.slice/real.scope\n"
    )
    assert cap.parse_self_cgroup(text) == "/user.slice/real.scope"


@pytest.mark.parametrize(
    "text",
    ["", "\n", "0:/user.slice/no_empty_controllers\n", "12:memory:/legacy\n"],
)
def test_l5_parse_self_cgroup_refuses_v1_or_empty(text):
    with pytest.raises(ValueError):
        cap.parse_self_cgroup(text)


def test_l7_margin_parity_with_resource_envelope():
    from zemosaic.core.zegrid import resource_envelope as renv

    for value in range(0, 200_001, 7):
        assert cap.ceil_margin(value) == renv.ceil_margin_div(value, 15, 100)
    for value in (0, 1, 2**40, 2**63, 2**64 - 1):
        assert cap.ceil_margin(value) == renv.ceil_margin_div(value, 15, 100)


def test_cgroup_ram_headroom_only_bounds_non_pageable_guard(tmp_path):
    root = tmp_path / "cgroup"
    _cgroup_level(root, "", controllers="memory")
    _cgroup_level(
        root, "run", controllers="memory",
        memory_max="500000000", memory_current="100000000",
        swap_max="max", swap_current=0,
    )
    levels = cap.read_cgroup_levels("/run", root)
    decision = _decision(levels=levels)
    assert decision.cgroup_ram_status == cap.LIMITED
    assert decision.cgroup_ram_headroom == 400_000_000
    assert decision.b_ram_eff == 400_000_000
    assert "cgroup_ram_tighter_than_host" in decision.reasons
    # memory.max is not swap: the effective swap is still the disk sum.
    assert decision.s_effectif == decision.s_disk_free
    # PROCESS_BASELINE_BYTES (340 MiB) > 400 MB? 340 MiB = 356 515 840 B: covered.
    assert decision.ram_guard_ok is True
    assert decision.status == "credit_sufficient"

    tight = _decision(levels=levels, non_pageable_new_bytes=400_000_001)
    assert tight.ram_guard_ok is False
    assert "b_ram_below_non_pageable_new" in tight.reasons
    assert tight.status == "credit_insufficient"


def test_cgroup_chain_paths_and_validation(tmp_path):
    root = tmp_path / "cg"
    assert cap.cgroup_chain_paths("/a/b", root) == [
        str(root),
        str(root / "a"),
        str(root / "a" / "b"),
    ]
    assert cap.cgroup_chain_paths("/", root) == [str(root)]
    with pytest.raises(ValueError):
        cap.cgroup_chain_paths("relative/path", root)
    with pytest.raises(ValueError):
        cap.combine_cgroup([], "memory")


# ---------------------------------------------------------------------------
# Trinary semantics, exhaustively
# ---------------------------------------------------------------------------

def _level(status: str, headroom: int | None = None, path: str = "p") -> cap.CgroupLevel:
    return cap.CgroupLevel(path, False, True, status, headroom, status, headroom, "test")


@pytest.mark.parametrize("kind", ["ram", "swap"])
def test_combine_cgroup_trinary_matrix(kind):
    assert cap.combine_cgroup([], kind).status == cap.NA
    assert cap.combine_cgroup([_level(cap.NA)], kind).status == cap.NA

    unlimited = cap.combine_cgroup([_level(cap.NA), _level(cap.UNLIMITED)], kind)
    assert (unlimited.status, unlimited.headroom_bytes) == (cap.UNLIMITED, None)

    limited = cap.combine_cgroup(
        [_level(cap.NA), _level(cap.LIMITED, 10), _level(cap.UNLIMITED)], kind
    )
    assert (limited.status, limited.headroom_bytes) == (cap.LIMITED, 10)

    minimum = cap.combine_cgroup(
        [_level(cap.LIMITED, 10), _level(cap.LIMITED, 3), _level(cap.LIMITED, 7)], kind
    )
    assert (minimum.status, minimum.headroom_bytes) == (cap.LIMITED, 3)

    unknown = cap.combine_cgroup(
        [_level(cap.UNLIMITED), _level(cap.UNKNOWN, path="u"), _level(cap.LIMITED, 5)], kind
    )
    assert unknown.status == cap.UNKNOWN
    assert unknown.headroom_bytes is None
    assert unknown.unknown_paths == ["u"]

    # Tuple compatibility with the G0 oracle.
    assert cap.combine_cgroup([_level(cap.UNKNOWN, path="u")], kind) == (
        cap.UNKNOWN, None, ["u"],
    )


# ---------------------------------------------------------------------------
# Commit gate (vm.overcommit_memory)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [0, 1])
def test_overcommit_modes_0_and_1_have_no_hard_gate(mode):
    facts = cap.read_overcommit(mode, 50, 0)
    gate = cap.evaluate_commit(facts, 10 ** 12)
    assert gate.allowed is True
    assert gate.status == cap.NA
    assert gate.reason == f"overcommit_mode_{mode}_no_hard_gate"
    assert gate.to_dict() == G0.evaluate_commit(mode, None, None, 10 ** 12).to_dict()


def test_overcommit_mode_2_strict_gate():
    facts = cap.read_overcommit(
        2, 50, 0, commit_limit=2_500_000_000, committed_as=500_000_000
    )
    assert facts.status == cap.LIMITED
    assert facts.headroom_bytes == 2_000_000_000

    allowed = cap.evaluate_commit(facts, 2_000_000_000)
    assert allowed.allowed is True
    assert allowed.reason == "commit_headroom_ok"
    refused = cap.evaluate_commit(facts, 2_000_000_001)
    assert refused.allowed is False
    assert refused.reason == "commit_headroom_insufficient"
    assert allowed.to_dict() == G0.evaluate_commit(
        2, 2_500_000_000, 500_000_000, 2_000_000_000
    ).to_dict()

    # Committed above the limit clamps the headroom to zero (never negative).
    exhausted = cap.read_overcommit(2, 50, 0, commit_limit=100, committed_as=200)
    assert exhausted.headroom_bytes == 0
    assert cap.evaluate_commit(exhausted, 1).allowed is False


def test_overcommit_mode_2_missing_fields_is_unknown():
    facts = cap.read_overcommit(2, 50, 0, commit_limit=None, committed_as=0)
    assert facts.status == cap.UNKNOWN
    assert facts.reason == "commit_fields_unknown"
    gate = cap.evaluate_commit(facts, 1)
    assert gate.allowed is False
    assert gate.status == cap.UNKNOWN
    assert gate.to_dict() == G0.evaluate_commit(2, None, 0, 1).to_dict()


def test_overcommit_unknown_mode_fails_closed():
    for mode in (None, 3):
        facts = cap.read_overcommit(mode, None, None)
        gate = cap.evaluate_commit(facts, 1)
        assert gate.allowed is False
        assert gate.status == cap.UNKNOWN
        expected = "overcommit_mode_unknown" if mode is None else "unknown_overcommit_mode_3"
        assert gate.reason == expected
        assert gate.to_dict() == G0.evaluate_commit(mode, None, None, 1).to_dict()


def test_commit_gate_never_subtracts_b_ram():
    facts = cap.read_overcommit(
        2, 50, 0, commit_limit=1_000_000_000, committed_as=1_000_000_000
    )
    decision = _decision(a_raw=40_000_000_000, p_gauge_raw=50_000_000_000, overcommit=facts)
    assert decision.b_ram is not None and decision.b_ram > 20_000_000_000
    assert decision.commit.headroom_bytes == 0
    assert decision.commit.allowed is False
    assert decision.status == "credit_insufficient"
    assert "commit_headroom_insufficient" in decision.reasons


# ---------------------------------------------------------------------------
# Non-pageable guard
# ---------------------------------------------------------------------------

def test_non_pageable_over_b_ram_cannot_be_rescued_by_swap():
    good = _decision()
    assert good.status == "credit_sufficient"
    b_ram = good.b_ram
    assert b_ram is not None

    at_limit = _decision(non_pageable_new_bytes=b_ram)
    assert at_limit.ram_guard_ok is True
    assert at_limit.status == "credit_sufficient"

    over = _decision(non_pageable_new_bytes=b_ram + 1)
    assert over.ram_guard_ok is False
    assert "b_ram_below_non_pageable_new" in over.reasons
    assert over.status == "credit_insufficient"
    assert over.s_credit > 0  # the credit exists, it is simply not usable


# ---------------------------------------------------------------------------
# Exact formulas and odd rounding
# ---------------------------------------------------------------------------

def test_historical_arithmetic_oracle():
    assert cap.plan_budget(HISTORICAL_A) == HISTORICAL_B_RAM
    assert cap.ceil_margin(HISTORICAL_P) == 2_533_774_093
    assert cap.credit_needed(HISTORICAL_P, HISTORICAL_B_RAM) == HISTORICAL_NEED
    assert cap.s_needed_for(HISTORICAL_NEED) == HISTORICAL_S_NEEDED
    assert cap.admit_budget(HISTORICAL_A) == 1_872_160_358

    decision = _decision()
    assert decision.b_ram == HISTORICAL_B_RAM
    assert decision.credit_needed_bytes == HISTORICAL_NEED
    assert decision.s_credit >= decision.credit_needed_bytes  # arithmetic only


@pytest.mark.parametrize(
    "raw, expected",
    [
        (1, 2),
        (3, 4),
        (7, 9),
        (20, 23),
        (21, 25),
        (2_203_281_821, ((2_203_281_821 * 115) + 99) // 100),
        (999_999_999, ((999_999_999 * 115) + 99) // 100),
    ],
)
def test_ceil_margin_odd_rounding(raw, expected):
    assert cap.ceil_margin(raw) == expected
    assert isinstance(cap.ceil_margin(raw), int)


@pytest.mark.parametrize("s_effectif, expected", [
    (GIB - 1, 0),
    (GIB, (9 * GIB) // 10),
    (GIB + 1, (9 * (GIB + 1)) // 10),
    (1_073_741_825, 966_367_642),
    (1_073_741_826, 966_367_643),
])
def test_swap_credit_floor_and_odd_values(s_effectif, expected):
    assert cap.swap_credit(s_effectif) == expected
    assert cap.swap_credit(None) == 0


def _device(free_bytes: int) -> cap.SwapDevice:
    return cap.SwapDevice("/swapfile", "file", free_bytes, 0, False, True, "disk_backed")


def test_one_gib_boundary_is_a_floor_not_a_cap():
    below = _decision(devices=[_device(GIB - 1)])
    assert below.s_effectif == GIB - 1
    assert below.s_credit == 0
    assert "effective_swap_below_minimum" in below.reasons
    assert below.status == "credit_insufficient"

    exact = _decision(devices=[_device(GIB)])
    assert exact.s_effectif == GIB
    assert exact.s_credit == (9 * GIB) // 10

    above = _decision(devices=[_device(GIB + 1)])
    assert above.s_effectif == GIB + 1
    assert above.s_credit == (9 * (GIB + 1)) // 10


def test_credit_boundary_is_exact_at_one_byte():
    good = _decision()
    b_ram = good.b_ram
    assert b_ram is not None

    p = 3_000_000_000
    need = cap.credit_needed(p, b_ram)
    assert need > 0

    exact = _decision(p_gauge_raw=p, devices=[_device(_s_for_credit(need))])
    assert exact.credit_needed_bytes == need
    assert exact.s_credit == need
    assert exact.covered is True
    assert exact.status == "credit_sufficient"

    just_below = _decision(
        p_gauge_raw=p, devices=[_device(_s_for_credit(need - 1))]
    )
    assert just_below.s_credit == need - 1
    assert just_below.credit_needed_bytes == need
    assert just_below.covered is False
    assert just_below.status == "credit_insufficient"


def test_credit_needed_round_trip_with_odd_bytes():
    for need in (966_367_641, 966_367_642, 1_000_000_001, 2_118_400_000):
        s = _s_for_credit(need)
        assert cap.swap_credit(s) == need
        assert cap.swap_credit(s - 1) < need
        assert cap.s_needed_for(need) <= s


def test_zero_need_is_not_needed_even_without_credit():
    tiny = _decision(devices=[], p_gauge_raw=1)
    assert tiny.credit_needed_bytes == 0
    assert tiny.status == "not_needed"
    assert tiny.s_credit == 0
    assert tiny.covered is False


# ---------------------------------------------------------------------------
# Unknown facts must fail closed
# ---------------------------------------------------------------------------

def test_unknown_capacity_inputs_fail_closed():
    decision = cap.evaluate_credit(
        cap.GaugeSwapCapacity(
            a_raw=None,
            devices=None,
            cgroup_levels=None,
            overcommit=None,
        ),
        HISTORICAL_P,
        cap.POLICY_BOUNDED_SWAP,
    )
    assert decision.s_credit == 0
    assert decision.s_effectif is None
    assert decision.b_ram is None
    assert decision.b_ram_eff is None
    assert decision.credit_needed_bytes is None
    assert decision.status == "credit_insufficient"
    assert decision.covered is False
    for reason in (
        "mem_available_unknown",
        "swap_devices_unknown",
        "cgroup_levels_unknown",
        "cgroup_swap_unknown",
        "overcommit_mode_unknown",
        "b_ram_below_non_pageable_new",
    ):
        assert reason in decision.reasons

    # No infinity leaks into the serialized payload.
    text = decision.to_json()
    json.loads(text, parse_constant=_reject_constant)


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def test_policy_gate_never_grants_credit():
    ram_only = _decision(policy=cap.POLICY_RAM_ONLY)
    assert ram_only.status == cap.NA
    assert ram_only.s_credit == 0
    assert ram_only.s_effectif is None
    assert ram_only.covered is False
    assert ram_only.reasons == ("policy_ram_only",)

    weird = _decision(policy="bounded_swap_v2")
    assert weird.status == cap.NA
    assert weird.s_credit == 0
    assert weird.reasons == ("policy_not_bounded_swap",)


# ---------------------------------------------------------------------------
# JSON canonical stability
# ---------------------------------------------------------------------------

def test_decision_byte_fields_are_python_ints():
    payload = _decision().to_dict()
    for key in (
        "a_raw_bytes", "b_ram_bytes", "b_admit_bytes", "b_ram_eff_bytes",
        "p_gauge_raw_bytes", "p_gauge_margined_bytes", "s_disk_free_bytes",
        "s_effectif_bytes", "s_credit_bytes", "credit_needed_bytes",
    ):
        assert type(payload[key]) is int, key
    for device in payload["swap_devices"]:
        assert type(device["free_bytes"]) is int


def test_canonical_json_is_stable_and_strict():
    first = _decision()
    second = _decision()
    assert first.to_json() == second.to_json()
    assert first.to_json() == cap.canonical_json(first.to_dict())
    assert first.to_json() == json.dumps(
        first.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    payload = json.loads(first.to_json(), parse_constant=_reject_constant)
    assert payload["schema"] == "gauge-swap-capacity-v1"
    assert payload["policy"] == cap.POLICY_BOUNDED_SWAP
    assert payload["s_credit_bytes"] == first.s_credit
    assert list(payload) == sorted(payload)

    keys = [
        "a_raw_bytes", "b_admit_bytes", "b_ram_bytes", "b_ram_eff_bytes",
        "cgroup_levels", "cgroup_ram", "cgroup_swap", "commit", "covered",
        "credit_needed_bytes", "facts", "limits", "p_gauge_margined_bytes",
        "p_gauge_raw_bytes", "policy", "ram_guard", "reasons", "s_credit_bytes",
        "s_disk_free_bytes", "s_effectif_bytes", "schema", "status", "swap_devices",
    ]
    assert sorted(payload) == keys


def test_decision_is_immutable():
    decision = _decision()
    with pytest.raises(Exception):
        decision.s_credit = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Purity: no float, no write, no product import
# ---------------------------------------------------------------------------

def test_module_has_no_float_and_no_writes_and_no_product_imports():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    float_constants = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
    ]
    assert float_constants == []

    float_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "float"
    ]
    assert float_calls == []

    math_inf = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "inf"
    ]
    assert math_inf == []

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert all("zemosaic" not in name for name in imported)
    assert all("scheduler" not in name for name in imported)

    write_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open"
    ]
    assert write_calls == []
    write_attrs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in {"write_text", "write_bytes"}
    ]
    assert write_attrs == []


def test_importing_module_does_not_import_mode_or_scheduler():
    code = (
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        "import zemosaic.core.zegrid.gauge_swap_capacity as m\n"
        "bad = sorted(name for name in sys.modules if 'zemosaic' in name and "
        "('zemosaic_zegrid_mode' in name or name.endswith('zegrid.scheduler')))\n"
        "print(','.join(bad))\n"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_importing_module_does_not_import_product_engine_at_all():
    code = (
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        "before = {name for name in sys.modules if name.startswith('zemosaic')}\n"
        "import zemosaic.core.zegrid.gauge_swap_capacity as m\n"
        "after = {name for name in sys.modules if name.startswith('zemosaic')}\n"
        "print(sorted(after - before))\n"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    # Importing the package + this module may add package bookkeeping only.
    added = result.stdout.strip()
    assert "zemosaic_zegrid_mode" not in added
    assert "scheduler" not in added


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def test_integer_validation_is_strict():
    with pytest.raises(TypeError):
        cap.plan_budget(True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        cap.plan_budget(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cap.plan_budget(-1)
    with pytest.raises(ValueError):
        cap.ceil_margin(-5)
    with pytest.raises(TypeError):
        cap.evaluate_credit(
            cap.GaugeSwapCapacity(HISTORICAL_A, [], [], None), 1.5  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        cap.GaugeSwapCapacity(HISTORICAL_A, [], [], None, non_pageable_new_bytes=-1)
    with pytest.raises(TypeError):
        cap.read_swap_devices(None)  # type: ignore[arg-type]


def test_capacity_is_immutable():
    capacity = cap.GaugeSwapCapacity(HISTORICAL_A, [], [], None)
    with pytest.raises(Exception):
        capacity.a_raw = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Coherence with the G0 oracle on the same facts
# ---------------------------------------------------------------------------

_PARITY_FIELDS = (
    "a_raw",
    "b_ram",
    "b_admit",
    "b_ram_eff",
    "p_gauge_raw",
    "p_gauge_margined",
    "s_disk_free",
    "s_effectif",
    "s_credit",
    "credit_needed_bytes",
    "covered",
    "status",
    "cgroup_swap_status",
    "cgroup_swap_headroom",
    "cgroup_ram_status",
    "cgroup_ram_headroom",
    "ram_guard_np_bytes",
    "ram_guard_ok",
)


def _assert_parity(
    *,
    a_raw: int,
    p_gauge_raw: int,
    devices: list,
    levels: list,
    overcommit_mode: int | None,
    commit_limit: int | None,
    committed_as: int | None,
    non_pageable_new_bytes: int = cap.PROCESS_BASELINE_BYTES,
    expect_status: str,
):
    facts = cap.read_overcommit(
        overcommit_mode, 50, 0, commit_limit=commit_limit, committed_as=committed_as
    )
    mine = cap.evaluate_credit(
        cap.GaugeSwapCapacity(
            a_raw=a_raw,
            devices=devices,
            cgroup_levels=levels,
            overcommit=facts,
            non_pageable_new_bytes=non_pageable_new_bytes,
        ),
        p_gauge_raw,
        cap.POLICY_BOUNDED_SWAP,
    )
    oracle = G0.evaluate_contract(
        a_raw=a_raw,
        p_gauge_raw=p_gauge_raw,
        devices=devices,
        levels=levels,
        overcommit_mode=overcommit_mode,
        commit_limit=commit_limit,
        committed_as=committed_as,
        non_pageable_new_bytes=non_pageable_new_bytes,
    )
    assert mine.status == expect_status
    for field in _PARITY_FIELDS:
        left = getattr(mine, field)
        right = getattr(oracle, field)
        if isinstance(left, tuple):
            right = tuple(right)
        assert left == right, (field, left, right)
    assert list(mine.reasons) == list(oracle.reasons)
    assert mine.commit.to_dict() == oracle.commit.to_dict()
    assert [device.to_dict() for device in mine.devices] == [
        device.to_dict() for device in oracle.devices
    ]
    assert [level.to_dict() for level in mine.cgroup] == [
        level.to_dict() for level in oracle.cgroup
    ]


def _parity_facts(tmp_path):
    """Shared fact set built from real temporary files."""
    mem_text = _write(tmp_path, "meminfo.txt", _meminfo_text(BASELINE_MEMINFO))
    swaps_text = _write(
        tmp_path,
        "swaps.txt",
        _swaps_text(
            [
                ("/dev/zram0", "partition", 8_000_000, 5_000_000),
                ("/swapfile", "file", 25_000_000, 1_000_000),
            ]
        ),
    )
    mem_mine = cap.read_meminfo_text(mem_text)
    mem_g0 = G0.parse_meminfo(mem_text)
    assert mem_mine == mem_g0

    devices_mine = cap.read_swap_devices(swaps_text, _stat_ok)
    devices_g0 = G0.parse_swaps(swaps_text, stat_fn=_stat_ok)
    assert [device.to_dict() for device in devices_mine] == [
        device.to_dict() for device in devices_g0
    ]

    root = tmp_path / "cgroup"
    _cgroup_level(root, "", controllers="memory")
    _cgroup_level(
        root, "run", controllers="memory",
        memory_max="max", memory_current=0,
        swap_max="max", swap_current=0,
    )
    levels_mine = cap.read_cgroup_levels("/run", root)
    levels_g0 = G0.read_cgroup_levels("/run", root)
    assert [level.to_dict() for level in levels_mine] == [
        level.to_dict() for level in levels_g0
    ]
    assert cap.combine_cgroup(levels_mine, "swap") == G0.combine_cgroup(levels_g0, "swap")
    assert cap.combine_cgroup(levels_mine, "ram") == G0.combine_cgroup(levels_g0, "ram")
    return mem_mine, devices_mine, levels_mine


def test_parity_with_g0_formulas():
    for value in (0, 1, 7, HISTORICAL_A, 999_999_999, 40_000_000_000):
        assert cap.plan_budget(value) == G0.plan_budget(value)
        assert cap.admit_budget(value) == G0.admit_budget(value)
        assert cap.ceil_margin(value) == G0.ceil_margin(value)
        assert cap.credit_needed(value, HISTORICAL_B_RAM) == G0.credit_needed(
            value, HISTORICAL_B_RAM
        )
        assert cap.s_needed_for(value) == G0.s_needed_for(value)
        assert cap.swap_credit(value) == G0.swap_credit(value)
    assert cap.swap_credit(None) == G0.swap_credit(None)


def test_parity_with_g0_on_shared_facts(tmp_path):
    mem, devices, levels = _parity_facts(tmp_path)

    def go(**kwargs):
        params = dict(
            a_raw=mem["MemAvailable"],
            p_gauge_raw=HISTORICAL_P,
            devices=devices,
            levels=levels,
            overcommit_mode=0,
            commit_limit=mem["CommitLimit"],
            committed_as=mem["Committed_AS"],
        )
        params.update(kwargs)
        return _assert_parity(**params)

    go(expect_status="credit_sufficient")

    # No disk swap at all.
    go(devices=[], expect_status="credit_insufficient")

    # cgroup swap unknown.
    unknown_levels = [
        cap.CgroupLevel("/sys/fs/cgroup", True, False, cap.NA, None, cap.NA, None, "root"),
        cap.CgroupLevel(
            "/sys/fs/cgroup/work/run", False, True, cap.UNLIMITED, None,
            cap.UNKNOWN, None, "missing",
        ),
    ]
    go(levels=unknown_levels, expect_status="credit_insufficient")

    # cgroup swap cap.
    capped_levels = [
        cap.CgroupLevel("/sys/fs/cgroup", True, False, cap.NA, None, cap.NA, None, "root"),
        cap.CgroupLevel(
            "/sys/fs/cgroup/work/run", False, True, cap.UNLIMITED, None,
            cap.LIMITED, 2_000_000_000, "cap",
        ),
    ]
    go(levels=capped_levels, p_gauge_raw=5_000_000_000,
       expect_status="credit_insufficient")

    # cgroup RAM cap narrower than B_ram.
    ram_levels = [
        cap.CgroupLevel("/sys/fs/cgroup", True, False, cap.NA, None, cap.NA, None, "root"),
        cap.CgroupLevel(
            "/sys/fs/cgroup/work/run", False, True, cap.LIMITED, 400_000_000,
            cap.UNLIMITED, None, "ramcap",
        ),
    ]
    go(levels=ram_levels, expect_status="credit_sufficient")

    # Non-pageable above B_ram.
    go(non_pageable_new_bytes=2_000_000_000, expect_status="credit_insufficient")

    # Strict commit headroom insufficient.
    go(
        overcommit_mode=2,
        commit_limit=2_500_000_000,
        committed_as=500_000_000,
        expect_status="credit_insufficient",
    )
    # Strict commit headroom sufficient.
    go(
        overcommit_mode=2,
        commit_limit=8_000_000_000,
        committed_as=500_000_000,
        expect_status="credit_sufficient",
    )
    # Mode 2 with a missing field.
    go(
        overcommit_mode=2,
        commit_limit=None,
        committed_as=500_000_000,
        expect_status="credit_insufficient",
    )
    # Unknown overcommit mode.
    go(overcommit_mode=None, commit_limit=None, committed_as=None,
       expect_status="credit_insufficient")

    # Zero need: both say "not_needed".
    go(p_gauge_raw=1, expect_status="not_needed")

    # Odd huge gauge footprint and a huge A.
    go(p_gauge_raw=2_203_281_821, expect_status="credit_sufficient")
    go(a_raw=40_000_000_000, p_gauge_raw=99_999_999_999,
       expect_status="credit_insufficient")


def test_g0_tool_still_self_tests_green(capsys):
    assert G0.self_test() == 0
    assert "SELF-TEST OK" in capsys.readouterr().out
