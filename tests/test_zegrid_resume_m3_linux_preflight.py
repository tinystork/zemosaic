"""Focused tests for the READ-ONLY M3 Linux preflight tool.

These tests never allocate pressure, start a product run or modify the host.
They exercise the pure decision function, the strict telemetry readers and the
cgroup ancestor walk.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools/zegrid_resume_m3/m3_linux_preflight.py"

GIB = 1024**3
MIB = 1024**2


def _load_tool():
    spec = importlib.util.spec_from_file_location("m3_linux_preflight", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pf = _load_tool()


def _mem(available, total=8 * GIB, swap_total=2 * GIB, swap_free=None):
    if swap_free is None:
        swap_free = swap_total
    return {
        "available": True,
        "path": "/proc/meminfo",
        "missing": [],
        "errors": [],
        "values": {
            "MemTotal": int(total),
            "MemAvailable": int(available),
            "SwapTotal": int(swap_total),
            "SwapFree": int(swap_free),
        },
    }


def _mount(fstype="ext4", free=10 * GIB):
    return {
        "available": True,
        "path": "/tmp/workspace",
        "mount_point": "/",
        "fstype": fstype,
        "source": "/dev/nvme0n1p7",
        "free_bytes": int(free),
        "total_bytes": int(free) * 2,
        "errors": [],
    }


def _cgroup(state="unlimited", limit_bytes=None, current=1 * GIB):
    return {
        "available": state != "unknown",
        "cgroup_path": "/user.slice/test",
        "limit_bytes": limit_bytes,
        "limit_is_unlimited": state == "unlimited",
        "limit_state": state,
        "effective_limit_source": "/" if state == "finite" else None,
        "ancestors": [],
        "current_bytes": int(current),
        "peak_bytes": None,
        "swap_limit_bytes": None,
        "errors": [] if state != "unknown" else ["unknown limit"],
    }


def _runtime(**missing):
    modules = {"numpy": True, "astropy": True, "psutil": True, "pytest": True}
    for name in missing:
        modules[name] = False
    return {
        "python_version": "3.13.5",
        "python_implementation": "CPython",
        "executable": sys.executable,
        "platform": "test",
        "machine": "x86_64",
        "modules": modules,
    }


def _source(**overrides):
    base = {
        "files": {},
        "missing_files": [],
        "missing_tokens": [],
        "m3_seam_present": True,
        "declared_recovery_config_key": True,
        "declared_resume_config_key": True,
        "declared_durable_store_config_key": True,
        "headless_product_cli": True,
        "activation_note": "",
    }
    base.update(overrides)
    return base


def _caps(
    reserve=1 * GIB,
    pressure=512 * MIB,
    pressure_max=512 * MIB,
    disk=2 * GIB,
):
    return {
        "reserve_bytes": int(reserve),
        "pressure_bytes": int(pressure),
        "pressure_max_bytes": int(pressure_max),
        "min_disk_free_bytes": int(disk),
    }


def _evaluate(available, **kwargs):
    kwargs.setdefault("meminfo", _mem(available))
    kwargs.setdefault("mount", _mount())
    kwargs.setdefault("cgroup", _cgroup())
    kwargs.setdefault("runtime", _runtime())
    kwargs.setdefault("source", _source())
    kwargs.setdefault("caps", _caps())
    return pf.evaluate_preflight(**kwargs)


def _codes(report, key="blockers"):
    return {item["code"] for item in report[key]}


def test_missing_meminfo_fails_closed(tmp_path):
    mem = pf.read_meminfo(tmp_path / "absent")
    assert mem["available"] is False
    report = pf.evaluate_preflight(
        meminfo=mem,
        mount=_mount(),
        cgroup=_cgroup(),
        runtime=_runtime(),
        source=_source(),
        caps=_caps(),
    )
    assert report["status"] == "NOT_READY"
    assert "MISSING_MEMORY_TELEMETRY" in _codes(report)
    assert report["m3_qualified"] is False


def test_meminfo_without_swap_fails_closed(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text("MemTotal: 1024 kB\nMemAvailable: 512 kB\n", encoding="ascii")
    mem = pf.read_meminfo(path)
    assert mem["available"] is False
    assert "SwapTotal" in mem["missing"] and "SwapFree" in mem["missing"]


def test_meminfo_rejects_unknown_units_and_inconsistent_ranges(tmp_path):
    bad_unit = tmp_path / "bad_unit"
    bad_unit.write_text(
        "MemTotal: 1 GB\nMemAvailable: 512 MB\n"
        "SwapTotal: 0 kB\nSwapFree: 0 kB\n",
        encoding="ascii",
    )
    assert pf.read_meminfo(bad_unit)["available"] is False

    inconsistent = tmp_path / "inconsistent"
    inconsistent.write_text(
        "MemTotal: 1024 kB\nMemAvailable: 2048 kB\n"
        "SwapTotal: 0 kB\nSwapFree: 0 kB\n",
        encoding="ascii",
    )
    assert pf.read_meminfo(inconsistent)["available"] is False

    bad_swap = tmp_path / "bad_swap"
    bad_swap.write_text(
        "MemTotal: 4096 kB\nMemAvailable: 2048 kB\n"
        "SwapTotal: 1024 kB\nSwapFree: 2048 kB\n",
        encoding="ascii",
    )
    assert pf.read_meminfo(bad_swap)["available"] is False


def test_available_exceeds_total_is_not_ready():
    report = _evaluate(2 * GIB, meminfo=_mem(2 * GIB, total=1 * GIB))
    assert report["status"] == "NOT_READY"
    assert "INVALID_MEMORY_TELEMETRY" in _codes(report)


def test_below_reserve_is_not_ready_with_quantified_shortfall():
    available = 1400 * MIB
    report = _evaluate(available)
    assert report["status"] == "NOT_READY"
    blocker = next(
        item for item in report["blockers"] if item["code"] == "RESERVE_SHORTFALL"
    )
    raw_after = available - 512 * MIB
    assert blocker["shortfall_bytes"] == 1 * GIB - raw_after
    assert blocker["required_available_for_reserve_bytes"] == 1 * GIB + 512 * MIB


def test_contract_rejects_weak_reserve_and_large_pressure():
    weak = _evaluate(
        4 * GIB,
        caps=_caps(reserve=1, pressure=512 * MIB, pressure_max=512 * MIB),
    )
    assert weak["status"] == "NOT_READY"
    assert "RESERVE_BELOW_CONTRACT" in _codes(weak)

    large = _evaluate(
        4 * GIB,
        caps=_caps(reserve=1 * GIB, pressure=768 * MIB, pressure_max=1 * GIB),
    )
    assert large["status"] == "NOT_READY"
    assert "PRESSURE_CAP_ABOVE_CONTRACT" in _codes(large)
    assert "PRESSURE_ABOVE_CONTRACT" in _codes(large)


def test_sane_feasible_bounded_case_is_ready():
    # A=4GiB, P=512MiB, R=1GiB; required(X)=3.778GiB lies in (3.5GiB, 4GiB].
    report = _evaluate(
        4 * GIB,
        bounded_aggregate_bytes=int(3.4 * GIB),
    )
    assert report["status"] == "READY_TO_PLAN"
    assert report["blockers"] == []
    bounded = report["aggregate_math"]["bounded"]
    assert bounded["required_raw_bytes"] == pf.required_raw_bytes(int(3.4 * GIB))
    assert bounded["refused_under_pressure"] is True
    assert bounded["above_reserve_floor"] is True
    assert report["aggregate_math"]["dose_window_bytes"]["non_empty"] is True


def test_bounded_min_below_reserve_floor_is_infeasible():
    report = _evaluate(4 * GIB, bounded_aggregate_bytes=800 * MIB)
    assert report["status"] == "NOT_READY"
    assert "BOUNDED_MIN_FITS_AT_RESERVE" in _codes(report)


def test_pressure_cap_insufficient_is_quantified():
    report = _evaluate(4 * GIB, bounded_aggregate_bytes=1 * GIB)
    assert report["status"] == "NOT_READY"
    blocker = next(
        item
        for item in report["blockers"]
        if item["code"] == "PRESSURE_CAP_INSUFFICIENT"
    )
    assert blocker["required_pressure_bytes"] > 512 * MIB


def test_smaller_safe_dose_is_reported_without_moving_host():
    # A=1.4GiB: 512MiB breaches the reserve, but a smaller dose in
    # [~0.2GiB .. 0.4GiB] can still refuse the 1.2GiB bounded requirement.
    report = _evaluate(1400 * MIB, bounded_aggregate_bytes=int(1.2 * GIB))
    assert report["status"] == "NOT_READY"
    assert "RESERVE_SHORTFALL" in _codes(report)
    warning = next(
        item for item in report["warnings"] if item["code"] == "SAFE_DOSE_AVAILABLE"
    )
    assert warning["low_bytes"] <= warning["recommended_bytes"] <= warning["high_bytes"]
    assert warning["recommended_bytes"] <= 512 * MIB


def test_unmeasured_bounded_aggregate_fails_closed():
    report = _evaluate(4 * GIB)
    assert report["status"] == "NOT_READY"
    assert "BOUNDED_AGGREGATE_NOT_MEASURED" in _codes(report)


def test_tmpfs_and_finite_cgroup_fail_closed():
    report = _evaluate(
        4 * GIB,
        mount=_mount(fstype="tmpfs"),
        cgroup=_cgroup(state="finite", limit_bytes=2 * GIB),
        bounded_aggregate_bytes=int(3.4 * GIB),
    )
    assert report["status"] == "NOT_READY"
    assert "UNSUPPORTED_FILESYSTEM" in _codes(report)
    assert "CGROUP_LIMIT_NOT_MODELLED" in _codes(report)


def test_unknown_cgroup_is_not_unlimited():
    report = _evaluate(
        4 * GIB,
        cgroup=_cgroup(state="unknown"),
        bounded_aggregate_bytes=int(3.4 * GIB),
    )
    assert report["status"] == "NOT_READY"
    assert "CGROUP_PROBE_UNAVAILABLE" in _codes(report)


def test_cgroup_finite_parent_is_effective(tmp_path):
    root = tmp_path / "cgroup"
    leaf = root / "parent" / "leaf"
    leaf.mkdir(parents=True)
    (leaf / "memory.max").write_text("max", encoding="ascii")
    (leaf / "memory.current").write_text(str(100 * MIB), encoding="ascii")
    (root / "parent" / "memory.max").write_text(str(1 * GIB), encoding="ascii")
    proc = tmp_path / "self_cgroup"
    proc.write_text("0::/parent/leaf\n", encoding="ascii")

    result = pf.read_self_cgroup(proc_self_cgroup=proc, cgroup_root=root)
    assert result["available"] is True
    assert result["limit_state"] == "finite"
    assert result["limit_bytes"] == 1 * GIB
    assert result["effective_limit_source"] == "/parent"
    assert [a["path"] for a in result["ancestors"]] == ["/", "/parent", "/parent/leaf"]

    # The inherited limit must reach the public decision, not only the reader.
    report = _evaluate(
        4 * GIB,
        cgroup=result,
        bounded_aggregate_bytes=int(3.4 * GIB),
    )
    assert report["status"] == "NOT_READY"
    assert "CGROUP_LIMIT_NOT_MODELLED" in _codes(report)


def test_cgroup_unlimited_chain_and_root_absent(tmp_path):
    root = tmp_path / "cgroup"
    leaf = root / "parent" / "leaf"
    leaf.mkdir(parents=True)
    (root / "parent" / "memory.max").write_text("max", encoding="ascii")
    (leaf / "memory.max").write_text("max", encoding="ascii")
    (leaf / "memory.current").write_text(str(100 * MIB), encoding="ascii")
    proc = tmp_path / "self_cgroup"
    proc.write_text("0::/parent/leaf\n", encoding="ascii")

    result = pf.read_self_cgroup(proc_self_cgroup=proc, cgroup_root=root)
    assert result["available"] is True
    assert result["limit_state"] == "unlimited"
    assert result["limit_is_unlimited"] is True


def test_cgroup_malformed_or_missing_ancestor_is_unknown(tmp_path):
    for name, write in (
        ("malformed", lambda path: path.write_text("bogus", encoding="ascii")),
        ("missing", lambda path: None),
    ):
        root = tmp_path / name / "cgroup"
        leaf = root / "parent" / "leaf"
        leaf.mkdir(parents=True)
        (leaf / "memory.max").write_text("max", encoding="ascii")
        (leaf / "memory.current").write_text(str(100 * MIB), encoding="ascii")
        write(root / "parent" / "memory.max")
        proc = tmp_path / name / "self_cgroup"
        proc.write_text("0::/parent/leaf\n", encoding="ascii")

        result = pf.read_self_cgroup(proc_self_cgroup=proc, cgroup_root=root)
        assert result["available"] is False
        assert result["limit_state"] == "unknown"
        assert result["limit_is_unlimited"] is None


def test_missing_runtime_module_blocks():
    report = _evaluate(4 * GIB, runtime=_runtime(astropy=False))
    assert report["status"] == "NOT_READY"
    assert "MISSING_RUNTIME_MODULE" in _codes(report)


def test_real_repo_source_probe_reports_missing_declared_key():
    """The real-tree probe must report the CURRENT truthful seam state.

    Historically the declared key and the headless CLI were missing and this
    test asserted their gaps. Later accepted work shipped both, so the probe
    must now report them present and those gaps absent. Gap-detection coverage
    is preserved separately by the synthetic-fixture test below (the name of
    this node is kept for stable identity).
    """
    source = pf.probe_source(REPO_ROOT)
    assert source["m3_seam_present"] is True
    assert source["declared_recovery_config_key"] is True
    assert source["headless_product_cli"] is True
    report = _evaluate(
        4 * GIB,
        source=source,
        bounded_aggregate_bytes=int(3.4 * GIB),
    )
    gap_codes = _codes(report, "gaps")
    assert "GAP_NO_DECLARED_RECOVERY_CONFIG_KEY" not in gap_codes
    assert "GAP_NO_HEADLESS_ZEGRID_CLI" not in gap_codes
    # Gaps are disclosed but are not host blockers.
    assert report["status"] == "READY_TO_PLAN"


def _synthetic_source_tree(root: Path, *, declared_key: bool, headless_cli: bool) -> Path:
    """Minimal tree carrying only the M3 seam tokens the probe checks."""
    files = {
        "src/zemosaic/zemosaic_zegrid_mode.py": (
            "def run_zegrid_mode():\n"
            "def _r55_m3_required_raw():\n"
            "def _r55_m3_make_coordinator():\n"
            "zegrid_resource_recovery_enabled\n"
            "resource_readmitted\n"
        ),
        "src/zemosaic/core/zegrid/resource_recovery.py": (
            "class RecoveryCoordinator:\n"
            "class CapacityWaiter:\n"
            "STATE_WAITING_RESOURCES\n"
        ),
        "src/zemosaic/core/zegrid/run_store.py": (
            "def commit_cell():\n" "def commit_and_release():\n"
        ),
        "src/zemosaic/core/zegrid/runtime_control.py": (
            "def collect_run_facts():\n" "MemAvailable\n"
        ),
        "src/zemosaic/zemosaic_config.py": (
            'DEFAULT_CONFIG = {"zegrid_resource_recovery_enabled": False}\n'
            if declared_key
            else "DEFAULT_CONFIG = {}\n"
        ),
    }
    if headless_cli:
        files["src/zemosaic/zegrid_cli.py"] = "def build_parser():\ndef main():\n"
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def test_synthetic_source_still_reports_closed_gaps_when_missing(tmp_path):
    """Gap detection must survive: a genuinely missing key/module still gaps."""
    root = _synthetic_source_tree(
        tmp_path / "repo", declared_key=False, headless_cli=False
    )
    source = pf.probe_source(root)
    assert source["m3_seam_present"] is True
    assert source["declared_recovery_config_key"] is False
    assert source["headless_product_cli"] is False
    report = _evaluate(
        4 * GIB,
        source=source,
        bounded_aggregate_bytes=int(3.4 * GIB),
    )
    gap_codes = _codes(report, "gaps")
    assert "GAP_NO_DECLARED_RECOVERY_CONFIG_KEY" in gap_codes
    assert "GAP_NO_HEADLESS_ZEGRID_CLI" in gap_codes
    assert report["status"] == "READY_TO_PLAN"


def test_root_mount_is_matched_for_absolute_paths(tmp_path):
    mounts = tmp_path / "mounts"
    mounts.write_text("/dev/root / ext4 rw 0 0\n", encoding="utf-8")
    target = tmp_path / "workspace"
    target.mkdir()
    mount = pf.read_mount_for(target, mounts)
    assert mount["available"] is True
    assert mount["mount_point"] == "/"
    assert mount["fstype"] == "ext4"


def test_cli_rejects_unsafe_parameters_before_output(tmp_path):
    out = tmp_path / "preflight.json"
    command = [
        sys.executable,
        str(TOOL_PATH),
        "--repo",
        str(REPO_ROOT),
        "--workspace",
        str(REPO_ROOT),
        "--reserve-bytes",
        "1",
        "--no-psutil-crosscheck",
        "--json-out",
        str(out),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 3
    assert "reserve-bytes" in result.stderr
    assert not out.exists()


def test_cli_writes_owned_json_and_refuses_overwrite(tmp_path):
    out = tmp_path / "preflight.json"
    command = [
        sys.executable,
        str(TOOL_PATH),
        "--repo",
        str(REPO_ROOT),
        "--workspace",
        str(REPO_ROOT),
        "--min-disk-free-bytes",
        "1",
        "--no-psutil-crosscheck",
        "--json-out",
        str(out),
    ]
    first = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert first.returncode in (0, 2), first.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == pf.SCHEMA
    assert payload["m3_qualified"] is False
    assert payload["caps"]["contract_enforced"] is True
    second = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert second.returncode == 3
    assert "refusing to overwrite" in second.stderr
