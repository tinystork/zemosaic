#!/usr/bin/env python3
"""ZM-ZEGRID M3 — READ-ONLY Linux physical-pressure preflight.

This tool answers ONE bounded question:

    is the CURRENT host safe/feasible for the proposed M3 pressure experiment,
    or is it NOT_READY (with a quantified reason)?

It does **not** allocate pressure, drop caches, write swap/sysctl settings,
change cgroups, start a product run, or qualify M3.  Its only optional write is
an explicitly-owned JSON report passed with ``--json-out``.

Observation contract (important)
--------------------------------
The product's M3 probe is built as::

    _r55_m3_observation_from_facts(_r55_run_checkpoint_sample(disk_root))

``_r55_run_checkpoint_sample`` calls ``available_memory_bytes()`` which uses
``psutil.virtual_memory().available``.  On Linux psutil derives that value from
``/proc/meminfo`` ``MemAvailable``.  This tool reads ``/proc/meminfo`` directly
and (when psutil is importable) cross-checks the value.  It never treats a
cgroup limit as physical pressure and never assumes the product sees one.

The decision is evidence for PLANNING only:

    READY_TO_PLAN != M3 PASS
    observed MemAvailable at this instant != memory still available later

Exit codes:
    0  READY_TO_PLAN
    2  NOT_READY (including missing/malformed telemetry; fail closed)
    3  tool/usage/output failure
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MIB = 1 << 20
GIB = 1 << 30

SCHEMA = "zemosaic-zegrid-m3-linux-preflight-v1"

DEFAULT_RESERVE_BYTES = 1 * GIB
DEFAULT_PRESSURE_MAX_BYTES = 512 * MIB
DEFAULT_MIN_DISK_FREE_BYTES = 2 * GIB

#: Public contract: the tool may allow STRICTER values, never weaker ones.
CONTRACT_MIN_RESERVE_BYTES = 1 * GIB
CONTRACT_MAX_PRESSURE_BYTES = 512 * MIB

REQUIRED_MEMINFO = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")

REAL_DISK_FSTYPES = {
    "ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "f2fs", "jfs", "reiserfs",
    "vfat", "exfat", "ntfs", "ntfs3",
}
UNSAFE_FSTYPES = {
    "tmpfs", "ramfs", "devtmpfs", "overlay", "squashfs", "aufs", "proc",
    "sysfs", "cgroup", "cgroup2", "debugfs", "securityfs", "pstore",
    "tracefs", "configfs", "fusectl", "mqueue", "hugetlbfs", "autofs",
}

#: Source tokens that prove the inspected tree really contains the M3 seam.
SOURCE_TOKENS = {
    "src/zemosaic/zemosaic_zegrid_mode.py": (
        "def run_zegrid_mode(",
        "def _r55_m3_required_raw(",
        "def _r55_m3_make_coordinator(",
        "zegrid_resource_recovery_enabled",
        "resource_readmitted",
    ),
    "src/zemosaic/core/zegrid/resource_recovery.py": (
        "class RecoveryCoordinator",
        "class CapacityWaiter",
        "STATE_WAITING_RESOURCES",
    ),
    "src/zemosaic/core/zegrid/run_store.py": (
        "def commit_cell(",
        "def commit_and_release(",
    ),
    "src/zemosaic/core/zegrid/runtime_control.py": (
        "def collect_run_facts(",
        "MemAvailable",
    ),
}


def _mib(n):
    return None if n is None else round(float(n) / MIB, 1)


def _parse_meminfo_text(text: str) -> tuple[dict, dict, list[str]]:
    """Parse fields with explicit units; never treat an unknown unit as bytes."""
    values = {}
    units = {}
    malformed = []
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        parts = rest.split()
        if not parts:
            continue
        try:
            amount = int(parts[0], 10)
        except ValueError:
            malformed.append(f"{key}: non-integer value")
            continue
        if amount < 0:
            malformed.append(f"{key}: negative value")
            continue
        unit = parts[1].lower() if len(parts) > 1 else "b"
        if unit == "kb":
            amount *= 1024
        elif unit not in ("b", "bytes"):
            malformed.append(f"{key}: unsupported unit {parts[1]!r}")
            continue
        values[key] = int(amount)
        units[key] = unit
    return values, units, malformed


def read_meminfo(path: str | os.PathLike[str] = "/proc/meminfo") -> dict:
    result = {
        "available": False,
        "path": str(path),
        "values": {},
        "units": {},
        "missing": list(REQUIRED_MEMINFO),
        "errors": [],
    }
    try:
        text = Path(path).read_text(encoding="ascii")
    except OSError as exc:
        result["errors"].append(f"MemAvailable unavailable ({path}): {exc}")
        return result
    values, units, malformed = _parse_meminfo_text(text)
    result["values"] = values
    result["units"] = units
    result["missing"] = [key for key in REQUIRED_MEMINFO if key not in values]
    if result["missing"]:
        result["errors"].append(
            "missing required /proc/meminfo fields: "
            + ", ".join(result["missing"])
        )
        return result
    required_malformed = [
        item for item in malformed if item.split(":", 1)[0] in REQUIRED_MEMINFO
    ]
    if required_malformed:
        result["errors"].append(
            "malformed required /proc/meminfo fields: "
            + "; ".join(required_malformed)
        )
        return result
    total = int(values["MemTotal"])
    available = int(values["MemAvailable"])
    swap_total = int(values["SwapTotal"])
    swap_free = int(values["SwapFree"])
    if total <= 0:
        result["errors"].append("MemTotal is not positive")
    elif available <= 0 or available > total:
        result["errors"].append(
            f"MemAvailable {available} B is not in (0, MemTotal {total} B]"
        )
    elif swap_free > swap_total:
        result["errors"].append(
            f"SwapFree {swap_free} B exceeds SwapTotal {swap_total} B"
        )
    if result["errors"]:
        return result
    result["available"] = True
    return result


def _unescape_mount(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def read_mount_for(
    path: str | os.PathLike[str],
    mounts_path: str | os.PathLike[str] = "/proc/mounts",
) -> dict:
    """Longest-prefix mount lookup + statvfs of the real path."""
    target = str(Path(path).resolve())
    result = {
        "available": False,
        "path": target,
        "mount_point": None,
        "fstype": None,
        "source": None,
        "free_bytes": None,
        "total_bytes": None,
        "errors": [],
    }
    best = None
    try:
        lines = Path(mounts_path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        result["errors"].append(f"mount table unavailable ({mounts_path}): {exc}")
        lines = []
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        source, mount_point, fstype = parts[0], _unescape_mount(parts[1]), parts[2]
        mp = mount_point.rstrip("/") or "/"
        # "/" must match every absolute path; the naive ``mp + "/"`` form
        # becomes "//" and silently misses the root filesystem.
        if mp == "/" or target == mp or target.startswith(mp + "/"):
            if best is None or len(mp) > len(best[1]):
                best = (source, mp, fstype)
    if best is None:
        result["errors"].append(f"no mount point found for {target}")
        return result
    result["source"], result["mount_point"], result["fstype"] = best
    try:
        stat = os.statvfs(target)
    except OSError as exc:
        result["errors"].append(f"statvfs failed for {target}: {exc}")
        return result
    result["free_bytes"] = int(stat.f_bavail) * int(stat.f_frsize)
    result["total_bytes"] = int(stat.f_blocks) * int(stat.f_frsize)
    result["available"] = True
    return result


def _read_int_file(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        value = int(text, 10)
    except ValueError:
        return None
    return value if value >= 0 else None


def _read_memory_max(path: Path) -> tuple[str, int | None, str | None]:
    """Return (state, limit_bytes, error).

    state is one of ``finite``/``max``/``absent``/``unreadable``/``malformed``.
    """
    try:
        text = path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return "absent", None, None
    except OSError as exc:
        return "unreadable", None, str(exc)
    if text == "max":
        return "max", None, None
    try:
        value = int(text, 10)
    except ValueError:
        return "malformed", None, f"malformed memory.max: {text!r}"
    if value < 0:
        return "malformed", None, f"negative memory.max: {text!r}"
    return "finite", value, None


def read_self_cgroup(
    *,
    proc_self_cgroup: str | os.PathLike[str] = "/proc/self/cgroup",
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
) -> dict:
    """Read the CALLER's effective cgroup v2 memory limit (read-only).

    The leaf alone is not enough: an ancestor may impose a finite limit.  The
    walk covers every ancestor up to the cgroup mount root.  A missing
    ``memory.max`` at the mount root is the known normal cgroup v2 shape when
    the controller files are not present there; any other missing/unreadable or
    malformed ancestor is UNKNOWN, never "unlimited".

    A finite effective limit is unsupported for this host-wide experiment: the
    product observes host ``MemAvailable`` while the container could OOM first.
    """
    result = {
        "available": False,
        "cgroup_path": None,
        "limit_bytes": None,
        "limit_is_unlimited": None,
        "limit_state": "unknown",
        "effective_limit_source": None,
        "ancestors": [],
        "current_bytes": None,
        "peak_bytes": None,
        "swap_limit_bytes": None,
        "errors": [],
    }
    try:
        text = Path(proc_self_cgroup).read_text(encoding="ascii")
    except OSError as exc:
        result["errors"].append(f"cgroup membership unavailable: {exc}")
        return result
    membership = None
    for line in text.splitlines():
        if line.startswith("0::"):
            membership = line.partition("::")[2].strip()
            break
    if not membership:
        result["errors"].append("no cgroup v2 (0::) membership line")
        return result
    if not membership.startswith("/"):
        result["errors"].append(f"malformed cgroup path: {membership!r}")
        return result
    parts = [part for part in membership.strip("/").split("/") if part]
    if any(part in (".", "..") for part in parts):
        result["errors"].append(f"unsafe cgroup path: {membership!r}")
        return result
    result["cgroup_path"] = "/" + "/".join(parts) if parts else "/"
    root = Path(cgroup_root)
    paths = [root]
    current = root
    for part in parts:
        current = current / part
        paths.append(current)

    limits = []
    unknown = []
    for path in paths:
        state, value, error = _read_memory_max(path / "memory.max")
        try:
            rel = "/" + str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        if rel == "/.":
            rel = "/"
        entry = {
            "path": rel,
            "state": state,
            "memory_max": (
                "max" if state == "max"
                else (str(value) if state == "finite" else None)
            ),
            "limit_bytes": value,
        }
        if error:
            entry["error"] = error
        result["ancestors"].append(entry)
        if state == "finite":
            limits.append((int(value), rel))
        elif state == "max":
            continue
        elif state == "absent" and path == root:
            # Known normal shape: the mount root has no memory controller files.
            continue
        elif state == "absent":
            unknown.append(f"memory.max absent at ancestor {rel}")
        else:
            unknown.append(
                f"memory.max {state} at ancestor {rel}"
                + (f": {error}" if error else "")
            )

    leaf = paths[-1]
    current_bytes = _read_int_file(leaf / "memory.current")
    if current_bytes is None:
        unknown.append(f"memory.current unavailable or invalid at {leaf}")
    if unknown:
        result["errors"].extend(unknown)
        result["limit_state"] = "unknown"
        result["limit_is_unlimited"] = None
        return result

    result["available"] = True
    result["current_bytes"] = int(current_bytes)
    result["peak_bytes"] = _read_int_file(leaf / "memory.peak")
    result["swap_limit_bytes"] = _read_int_file(leaf / "memory.swap.max")
    if limits:
        effective, source = min(limits, key=lambda item: item[0])
        result["limit_bytes"] = int(effective)
        result["limit_is_unlimited"] = False
        result["limit_state"] = "finite"
        result["effective_limit_source"] = source
    else:
        result["limit_bytes"] = None
        result["limit_is_unlimited"] = True
        result["limit_state"] = "unlimited"
        result["effective_limit_source"] = None
    return result


def read_runtime() -> dict:
    def has_module(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "modules": {
            name: has_module(name)
            for name in ("numpy", "astropy", "psutil", "pytest")
        },
    }


def cross_check_psutil(meminfo: dict) -> dict:
    """Optional read-only equivalence check of the product's probe."""
    result = {
        "available": False,
        "value_bytes": None,
        "delta_bytes": None,
        "error": None,
    }
    try:
        import psutil  # local: the tool remains stdlib-only when absent
    except Exception as exc:  # noqa: BLE001 - report, never fabricate
        result["error"] = f"psutil unavailable: {type(exc).__name__}: {exc}"
        return result
    try:
        value = int(psutil.virtual_memory().available)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"psutil probe failed: {type(exc).__name__}: {exc}"
        return result
    result["available"] = True
    result["value_bytes"] = value
    if meminfo.get("available"):
        result["delta_bytes"] = int(value) - int(
            meminfo["values"]["MemAvailable"]
        )
    return result


def probe_source(repo_root: str | os.PathLike[str]) -> dict:
    """Cheap textual probe of the real product seam (no heavy imports)."""
    repo = Path(repo_root)
    checks = {}
    missing_files = []
    missing_tokens = []
    for rel, tokens in SOURCE_TOKENS.items():
        path = repo / rel
        if not path.is_file():
            missing_files.append(rel)
            checks[rel] = {"present": False, "missing_tokens": list(tokens)}
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            missing_files.append(rel)
            checks[rel] = {"present": False, "missing_tokens": list(tokens)}
            continue
        absent = [token for token in tokens if token not in text]
        checks[rel] = {"present": True, "missing_tokens": absent}
        missing_tokens.extend(f"{rel}: {token}" for token in absent)
    config_path = repo / "src/zemosaic/zemosaic_config.py"
    config_text = ""
    if config_path.is_file():
        try:
            config_text = config_path.read_text(encoding="utf-8")
        except OSError:
            config_text = ""
    declared_recovery_key = '"zegrid_resource_recovery_enabled"' in config_text
    declared_resume_key = '"zegrid_resume_mode"' in config_text
    declared_store_key = '"zegrid_durable_store_dir"' in config_text
    # The headless CLI is a shipped product entry: detect the real module and
    # its parser/main seam instead of hard-coding absence.
    cli_path = repo / "src/zemosaic/zegrid_cli.py"
    cli_text = ""
    if cli_path.is_file():
        try:
            cli_text = cli_path.read_text(encoding="utf-8")
        except OSError:
            cli_text = ""
    headless_cli = bool(
        cli_text and "def build_parser(" in cli_text and "def main(" in cli_text
    )
    if declared_recovery_key and headless_cli:
        activation_note = (
            "run_zegrid_mode() is a real Python product entry; "
            "zegrid_resource_recovery_enabled is a declared config key and a "
            "headless ZeGrid CLI ships, so the protocol can use the shipped "
            "entry points directly."
        )
    else:
        activation_note = (
            "run_zegrid_mode() is a real Python product entry, but this tree "
            "lacks a declared zegrid_resource_recovery_enabled config key "
            "and/or a headless ZeGrid CLI; enabling M3 then requires a zconfig "
            "attribute (harness seam) until the product key exists."
        )
    return {
        "files": checks,
        "missing_files": missing_files,
        "missing_tokens": missing_tokens,
        "m3_seam_present": not missing_files and not missing_tokens,
        "declared_recovery_config_key": declared_recovery_key,
        "declared_resume_config_key": declared_resume_key,
        "declared_durable_store_config_key": declared_store_key,
        "headless_product_cli": headless_cli,
        "activation_note": activation_note,
    }


def required_raw_bytes(aggregate_bytes: int) -> int:
    """Exact inverse of the RUN 90 % budget: ceil(10 * aggregate / 9)."""
    peak = max(0, int(aggregate_bytes))
    if peak <= 0:
        return 1
    return (peak * 10 + 8) // 9


def _blocker(code: str, detail: str, **extra) -> dict:
    return {"code": str(code), "detail": str(detail), **extra}


def evaluate_preflight(
    *,
    meminfo: dict,
    mount: dict,
    cgroup: dict,
    runtime: dict,
    source: dict,
    psutil_check: dict | None = None,
    caps: dict | None = None,
    dense_aggregate_bytes: int | None = None,
    bounded_aggregate_bytes: int | None = None,
) -> dict:
    """Pure decision function: no probing, no allocation, no writes."""
    caps = dict(caps or {})
    reserve = int(caps.get("reserve_bytes", DEFAULT_RESERVE_BYTES))
    pressure = int(caps.get("pressure_bytes", DEFAULT_PRESSURE_MAX_BYTES))
    pressure_cap = int(
        caps.get("pressure_max_bytes", DEFAULT_PRESSURE_MAX_BYTES)
    )
    min_disk = int(
        caps.get("min_disk_free_bytes", DEFAULT_MIN_DISK_FREE_BYTES)
    )

    blockers: list[dict] = []
    warnings: list[dict] = []
    gaps: list[dict] = []

    if not meminfo.get("available"):
        blockers.append(
            _blocker(
                "MISSING_MEMORY_TELEMETRY",
                "; ".join(meminfo.get("errors") or ["MemAvailable unavailable"]),
            )
        )
        values = {}
    else:
        values = dict(meminfo["values"])

    available = values.get("MemAvailable")
    total = values.get("MemTotal")
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    if available is not None:
        if total is None or total <= 0:
            blockers.append(
                _blocker("INVALID_MEMORY_TELEMETRY", "MemTotal is missing/invalid")
            )
        elif available <= 0 or available > total:
            blockers.append(
                _blocker(
                    "INVALID_MEMORY_TELEMETRY",
                    f"MemAvailable {available} B is not in (0, MemTotal {total} B]",
                    available_bytes=int(available),
                    total_bytes=int(total),
                )
            )
        if swap_total is None or swap_free is None:
            blockers.append(
                _blocker(
                    "MISSING_SWAP_TELEMETRY",
                    "SwapTotal/SwapFree are required to report pressure risk",
                )
            )
        elif swap_total < 0 or swap_free < 0 or swap_free > swap_total:
            blockers.append(
                _blocker(
                    "INVALID_SWAP_TELEMETRY",
                    f"swap values are inconsistent: SwapFree {swap_free} B, "
                    f"SwapTotal {swap_total} B",
                )
            )

    if pressure <= 0:
        blockers.append(
            _blocker("INVALID_PRESSURE_CAP", "pressure_bytes must be > 0")
        )
    if reserve < CONTRACT_MIN_RESERVE_BYTES:
        blockers.append(
            _blocker(
                "RESERVE_BELOW_CONTRACT",
                f"reserve {reserve} B is below the mandatory "
                f"{CONTRACT_MIN_RESERVE_BYTES} B (1 GiB) floor; stricter "
                "values only",
                reserve_bytes=int(reserve),
                contract_floor_bytes=int(CONTRACT_MIN_RESERVE_BYTES),
            )
        )
    if pressure_cap > CONTRACT_MAX_PRESSURE_BYTES:
        blockers.append(
            _blocker(
                "PRESSURE_CAP_ABOVE_CONTRACT",
                f"pressure_max {pressure_cap} B is above the mandatory "
                f"{CONTRACT_MAX_PRESSURE_BYTES} B (512 MiB) ceiling; the cap "
                "may only be stricter",
                pressure_max_bytes=int(pressure_cap),
                contract_ceiling_bytes=int(CONTRACT_MAX_PRESSURE_BYTES),
            )
        )
    if pressure > CONTRACT_MAX_PRESSURE_BYTES:
        blockers.append(
            _blocker(
                "PRESSURE_ABOVE_CONTRACT",
                f"selected pressure {pressure} B is above the mandatory "
                f"{CONTRACT_MAX_PRESSURE_BYTES} B (512 MiB) ceiling",
                pressure_bytes=int(pressure),
                contract_ceiling_bytes=int(CONTRACT_MAX_PRESSURE_BYTES),
            )
        )
    if pressure > pressure_cap:
        blockers.append(
            _blocker(
                "PRESSURE_CAP_EXCEEDED",
                f"proposed pressure {pressure} B > cap {pressure_cap} B",
                proposed_bytes=pressure,
                cap_bytes=pressure_cap,
            )
        )
    if reserve <= 0:
        blockers.append(
            _blocker("INVALID_RESERVE", "reserve_bytes must be > 0")
        )

    raw_after = None
    reserve_shortfall = None
    window = {"low_bytes": None, "high_bytes": None, "non_empty": False}
    if available is not None and pressure > 0:
        raw_after = int(available) - int(pressure)
        if raw_after < reserve:
            reserve_shortfall = int(reserve) - int(raw_after)
            blockers.append(
                _blocker(
                    "RESERVE_SHORTFALL",
                    "MemAvailable - proposed pressure is below the live reserve: "
                    f"{raw_after} B < {reserve} B (shortfall "
                    f"{reserve_shortfall} B / {_mib(reserve_shortfall)} MiB)",
                    raw_after_pressure_bytes=int(raw_after),
                    reserve_bytes=int(reserve),
                    shortfall_bytes=int(reserve_shortfall),
                    required_available_for_reserve_bytes=int(reserve + pressure),
                )
            )
        if raw_after > 0:
            window = {
                "low_bytes": (9 * raw_after) // 10 + 1,
                "high_bytes": (9 * int(available)) // 10,
                "non_empty": False,
            }
            window["non_empty"] = bool(
                window["low_bytes"] <= window["high_bytes"]
            )
    if available is not None and pressure > 0 and not window["non_empty"]:
        blockers.append(
            _blocker(
                "EMPTY_PRESSURE_WINDOW",
                "no aggregate byte size would be admitted before pressure and "
                "refused after it under the proposed caps",
            )
        )

    # -- filesystem -------------------------------------------------------
    if not mount.get("available"):
        blockers.append(
            _blocker(
                "MISSING_FILESYSTEM_TELEMETRY",
                "; ".join(mount.get("errors") or ["mount unavailable"]),
            )
        )
    else:
        fstype = str(mount.get("fstype") or "").lower()
        if fstype in UNSAFE_FSTYPES:
            blockers.append(
                _blocker(
                    "UNSUPPORTED_FILESYSTEM",
                    f"{mount.get('path')} is on {fstype!r}; the durable store "
                    "and corpus must live on a real local disk",
                    fstype=fstype,
                )
            )
        elif fstype not in REAL_DISK_FSTYPES:
            blockers.append(
                _blocker(
                    "UNKNOWN_FILESYSTEM",
                    f"filesystem type {fstype!r} is not in the qualified local "
                    "disk set; refusing to assume durability",
                    fstype=fstype,
                )
            )
        free = mount.get("free_bytes")
        if free is None:
            blockers.append(
                _blocker("MISSING_DISK_FREE", "statvfs free bytes unavailable")
            )
        elif int(free) < min_disk:
            blockers.append(
                _blocker(
                    "DISK_HEADROOM",
                    f"free {int(free)} B < required {min_disk} B",
                    free_bytes=int(free),
                    required_bytes=int(min_disk),
                )
            )

    # -- cgroup -----------------------------------------------------------
    if not cgroup.get("available"):
        blockers.append(
            _blocker(
                "CGROUP_PROBE_UNAVAILABLE",
                "; ".join(cgroup.get("errors") or ["cgroup memory limit unknown"]),
                limit_state=cgroup.get("limit_state", "unknown"),
            )
        )
    elif cgroup.get("limit_state") == "finite":
        blockers.append(
            _blocker(
                "CGROUP_LIMIT_NOT_MODELLED",
                "a finite effective cgroup v2 memory.max is set (leaf or "
                "ancestor); the product observes host MemAvailable only, so "
                "host pressure and cgroup OOM are not the same threshold",
                limit_bytes=cgroup.get("limit_bytes"),
                limit_source=cgroup.get("effective_limit_source"),
                current_bytes=cgroup.get("current_bytes"),
            )
        )
    elif cgroup.get("limit_state") != "unlimited":
        blockers.append(
            _blocker(
                "CGROUP_LIMIT_UNKNOWN",
                "the effective cgroup memory limit could not be established; "
                "unknown is never treated as unlimited",
                limit_state=cgroup.get("limit_state"),
                errors=cgroup.get("errors") or [],
            )
        )

    # -- runtime ----------------------------------------------------------
    modules = runtime.get("modules") or {}
    for name in ("numpy", "astropy", "psutil"):
        if not modules.get(name):
            blockers.append(
                _blocker(
                    "MISSING_RUNTIME_MODULE",
                    f"Python module {name!r} is not importable from "
                    f"{runtime.get('executable')!r}",
                    module=name,
                )
            )

    # -- product seam -----------------------------------------------------
    if not source.get("m3_seam_present"):
        blockers.append(
            _blocker(
                "PRODUCT_SEAM_MISSING",
                "M3 seam/tokens not found: "
                + "; ".join(source.get("missing_tokens") or ["(unknown)"]),
            )
        )
    if source.get("declared_recovery_config_key") is not True:
        gaps.append(
            {
                "code": "GAP_NO_DECLARED_RECOVERY_CONFIG_KEY",
                "detail": (
                    "zegrid_resource_recovery_enabled is read with getattr but "
                    "is absent from DEFAULT_CONFIG/GUI; the experiment needs a "
                    "zconfig attribute (harness seam), not a shipped user path"
                ),
            }
        )
    if source.get("headless_product_cli") is not True:
        gaps.append(
            {
                "code": "GAP_NO_HEADLESS_ZEGRID_CLI",
                "detail": (
                    "the product entry is run_zegrid_mode(...); no CLI command "
                    "runs a ZeGrid folder end-to-end, so the protocol must call "
                    "the real Python entry from a harness"
                ),
            }
        )
    gaps.append(
        {
            "code": "GAP_LIVE_WAIT_STATE_NOT_PUBLISHED",
            "detail": (
                "the product has no live WAITING_RESOURCES export; the protocol "
                "plans a test-only transparent observer of the real "
                "RecoveryCoordinator journal. Without that live evidence (or "
                "equivalent real-object capture), a post-run checkpoint alone "
                "is not enough for PASS."
            ),
        }
    )

    if psutil_check is not None:
        if not psutil_check.get("available"):
            warnings.append(
                {
                    "code": "PSUTIL_CROSSCHECK_UNAVAILABLE",
                    "detail": str(psutil_check.get("error")),
                }
            )
        elif psutil_check.get("delta_bytes") is not None:
            delta = abs(int(psutil_check["delta_bytes"]))
            if delta > 64 * MIB:
                warnings.append(
                    {
                        "code": "PROBE_DIVERGENCE",
                        "detail": (
                            "psutil.available and /proc/meminfo MemAvailable "
                            f"differ by {delta} B; re-observe before planning"
                        ),
                        "delta_bytes": int(psutil_check["delta_bytes"]),
                    }
                )

    if swap_total is not None and swap_free is not None:
        swap_used = int(swap_total) - int(swap_free)
        if swap_used > 0:
            warnings.append(
                {
                    "code": "SWAP_IN_USE",
                    "detail": (
                        f"{swap_used} B of swap are in use; a pressure process "
                        "may be partly swapped and MemAvailable may move less "
                        "than its RSS"
                    ),
                }
            )
        if int(swap_free) < pressure:
            warnings.append(
                {
                    "code": "SWAP_CAPACITY_BELOW_PRESSURE",
                    "detail": (
                        f"free swap {int(swap_free)} B < proposed pressure "
                        f"{pressure} B; pressure cannot be silently absorbed, "
                        "but the live MemAvailable trace governs"
                    ),
                }
            )

    # -- aggregate math ---------------------------------------------------
    aggregate = {
        "required_raw_formula": "ceil(10 * aggregate_bytes / 9)",
        "reserve_bytes": int(reserve),
        "pressure_bytes": int(pressure),
        "pressure_max_bytes": int(pressure_cap),
        "raw_after_pressure_bytes": raw_after,
        "window_bytes": window,
        "dose_window_bytes": {
            "low_bytes": None,
            "high_bytes": None,
            "non_empty": False,
            "recommended_bytes": None,
        },
        "dense": None,
        "bounded": None,
    }
    if dense_aggregate_bytes is not None:
        required = required_raw_bytes(int(dense_aggregate_bytes))
        dense = {
            "aggregate_bytes": int(dense_aggregate_bytes),
            "required_raw_bytes": int(required),
            "admissible_before": bool(available is not None and required <= available),
            "refused_under_pressure": bool(
                raw_after is not None and required > raw_after
            ),
        }
        aggregate["dense"] = dense
        if not dense["admissible_before"]:
            blockers.append(
                _blocker(
                    "DENSE_AGGREGATE_NOT_ADMISSIBLE_BEFORE",
                    f"required raw {required} B > MemAvailable {available} B",
                )
            )
        if not dense["refused_under_pressure"]:
            blockers.append(
                _blocker(
                    "DENSE_AGGREGATE_NOT_REFUSED_UNDER_PRESSURE",
                    f"required raw {required} B <= raw after pressure {raw_after} B",
                )
            )
    if bounded_aggregate_bytes is not None:
        required = required_raw_bytes(int(bounded_aggregate_bytes))
        bounded = {
            "aggregate_bytes": int(bounded_aggregate_bytes),
            "required_raw_bytes": int(required),
            "admissible_before": bool(available is not None and required <= available),
            "refused_under_pressure": bool(
                raw_after is not None and required > raw_after
            ),
            "above_reserve_floor": bool(required > reserve),
            "required_pressure_to_refuse_bytes": (
                None
                if available is None
                else max(1, int(available) - int(required) + 1)
            ),
        }
        aggregate["bounded"] = bounded
        if available is not None:
            # A safe dose must (a) preserve the reserve and (b) be large enough
            # that the real bounded minimum is still refused.  This prevents
            # treating the 512 MiB MAX as a mandatory dose or mandating a host
            # move when a smaller admissible dose exists.
            low = max(1, int(available) - int(required) + 1)
            high = min(int(pressure_cap), int(available) - int(reserve))
            dose = {
                "low_bytes": int(low),
                "high_bytes": int(high),
                "non_empty": bool(low <= high),
                "recommended_bytes": int(low) if low <= high else None,
            }
            aggregate["dose_window_bytes"] = dose
            if not dose["non_empty"]:
                blockers.append(
                    _blocker(
                        "NO_SAFE_PRESSURE_DOSE",
                        "no pressure dose <= the cap can both preserve the "
                        "reserve and force the bounded minimum to refuse",
                        low_bytes=int(low),
                        high_bytes=int(high),
                    )
                )
            elif not (low <= int(pressure) <= high):
                warnings.append(
                    {
                        "code": "SAFE_DOSE_AVAILABLE",
                        "detail": (
                            f"selected pressure {pressure} B is outside the safe "
                            f"dose window [{low} .. {high}] B; choose "
                            f"--pressure-bytes within it (recommended {low} B) "
                            "or keep NOT_READY; do not lower the reserve"
                        ),
                        "recommended_bytes": int(low),
                        "low_bytes": int(low),
                        "high_bytes": int(high),
                    }
                )
        if not bounded["admissible_before"]:
            blockers.append(
                _blocker(
                    "BOUNDED_MIN_NOT_ADMISSIBLE",
                    f"bounded minimum requires {required} B > MemAvailable "
                    f"{available} B; the run cannot even reach the wait",
                )
            )
        if not bounded["above_reserve_floor"]:
            blockers.append(
                _blocker(
                    "BOUNDED_MIN_FITS_AT_RESERVE",
                    f"bounded minimum requires {required} B <= reserve {reserve} B; "
                    "at the safe reserve floor the bounded plan would fit, so M3 "
                    "would never enter WAITING_RESOURCES without lowering the guard",
                    bounded_required_bytes=int(required),
                    reserve_bytes=int(reserve),
                )
            )
        if not bounded["refused_under_pressure"]:
            needed = bounded["required_pressure_to_refuse_bytes"]
            blockers.append(
                _blocker(
                    "PRESSURE_CAP_INSUFFICIENT",
                    f"bounded minimum requires {required} B but raw after "
                    f"pressure is {raw_after} B; need pressure >= {needed} B, "
                    f"cap is {pressure_cap} B",
                    required_pressure_bytes=needed,
                    pressure_cap_bytes=int(pressure_cap),
                )
            )
    else:
        blockers.append(
            _blocker(
                "BOUNDED_AGGREGATE_NOT_MEASURED",
                "bounded_aggregate_bytes was not supplied; without the real "
                "bounded minimum the preflight cannot prove that M3 would "
                "enter WAITING_RESOURCES at the reserve floor instead of "
                "silently re-admitting a smaller bounded plan. Measure it with "
                "a no-pressure bounded reference, then re-run this preflight.",
            )
        )

    status = "READY_TO_PLAN" if not blockers else "NOT_READY"
    contract_ok = bool(
        reserve >= CONTRACT_MIN_RESERVE_BYTES
        and 0 < pressure <= pressure_cap <= CONTRACT_MAX_PRESSURE_BYTES
        and pressure <= CONTRACT_MAX_PRESSURE_BYTES
    )
    dose = aggregate["dose_window_bytes"]
    selected_in_dose = bool(
        dose["non_empty"]
        and dose["low_bytes"] is not None
        and dose["low_bytes"] <= int(pressure) <= dose["high_bytes"]
    )
    if blockers:
        blocker_codes = {item["code"] for item in blockers}
        if blocker_codes == {"BOUNDED_AGGREGATE_NOT_MEASURED"}:
            next_step = (
                "Measure the real bounded minimum at the pressure-governing raw "
                "with a no-pressure bounded reference (or scheduler minimum "
                "metadata), then re-run this preflight with "
                "--bounded-aggregate-bytes. Do not move host or lower the "
                "reserve solely because the bounded minimum is unknown."
            )
        elif "BOUNDED_MIN_FITS_AT_RESERVE" in blocker_codes:
            next_step = (
                "NOT_READY for this corpus/host: the bounded minimum fits at "
                "the 1 GiB reserve floor, so M3 would never wait. Do not lower "
                "the reserve or change the probe/identity to force a wait."
            )
        elif dose["non_empty"] and not selected_in_dose:
            next_step = (
                f"NOT_READY for the selected {int(pressure)} B dose, but a "
                f"safe dose exists in [{dose['low_bytes']} .. "
                f"{dose['high_bytes']}] B. Re-run with --pressure-bytes in "
                f"that window (recommended {dose['recommended_bytes']} B) "
                "only if the measured bounded minimum is real; do not lower "
                "the reserve, change probes/thresholds/identity, or move host "
                "solely because the 512 MiB maximum was selected."
            )
        else:
            next_step = (
                "NO-GO: fix the quantified blockers; do not change probes, "
                "thresholds, reserve, identity or science to obtain green."
            )
    else:
        next_step = (
            "Measure the real bounded minimum aggregate with a no-pressure "
            "bounded reference, re-run this preflight with "
            "--bounded-aggregate-bytes, then freeze the pressure runner/guards "
            "for review before any physical pressure."
        )
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "status": status,
        "m3_qualified": False,
        "claim": (
            "preflight evidence only; READY_TO_PLAN is not an M3 PASS and does "
            "not promise that memory remains available"
        ),
        "observation_contract": {
            "product_probe": "psutil.virtual_memory().available",
            "linux_mapping": "/proc/meminfo MemAvailable",
            "scope": "host-wide",
            "cgroup_aware_for_admission": False,
            "cgroup_role": (
                "diagnostic only; a cgroup limit is neither physical pressure "
                "nor proof that the product sees the same threshold"
            ),
            "meminfo": meminfo,
            "psutil_crosscheck": psutil_check,
        },
        "host": {
            "mem_total_bytes": total,
            "mem_available_bytes": available,
            "swap_total_bytes": swap_total,
            "swap_free_bytes": swap_free,
            "raw_after_pressure_bytes": raw_after,
            "max_safe_pressure_bytes": (
                None
                if available is None
                else max(0, int(available) - int(reserve))
            ),
        },
        "filesystem": mount,
        "cgroup": cgroup,
        "runtime": runtime,
        "source": source,
        "caps": {
            "reserve_bytes": int(reserve),
            "pressure_bytes": int(pressure),
            "pressure_max_bytes": int(pressure_cap),
            "min_disk_free_bytes": int(min_disk),
            "validated": False,
            "contract_enforced": True,
            "contract_min_reserve_bytes": int(CONTRACT_MIN_RESERVE_BYTES),
            "contract_max_pressure_bytes": int(CONTRACT_MAX_PRESSURE_BYTES),
            "note": (
                "physical ceilings are proposals, not validated; the reserve "
                "floor and pressure ceiling are enforced, and only stricter "
                "values are accepted"
            ),
        },
        "checks": {
            "reserve_ok": bool(
                reserve_shortfall is None
                and available is not None
                and reserve >= CONTRACT_MIN_RESERVE_BYTES
            ),
            "reserve_shortfall_bytes": reserve_shortfall,
            "pressure_cap_ok": bool(
                0 < pressure <= pressure_cap <= CONTRACT_MAX_PRESSURE_BYTES
            ),
            "contract_ok": contract_ok,
            "pressure_window_non_empty": bool(window["non_empty"]),
            "dose_window_non_empty": bool(dose["non_empty"]),
            "selected_dose_within_window": bool(selected_in_dose),
            "filesystem_ok": bool(mount.get("available"))
            and str(mount.get("fstype") or "").lower() in REAL_DISK_FSTYPES,
            "cgroup_supported": bool(cgroup.get("available"))
            and cgroup.get("limit_state") == "unlimited",
            "product_seam_ok": bool(source.get("m3_seam_present")),
            "declared_recovery_config_key": bool(
                source.get("declared_recovery_config_key")
            ),
            "headless_product_cli": bool(source.get("headless_product_cli")),
        },
        "aggregate_math": aggregate,
        "blockers": blockers,
        "warnings": warnings,
        "gaps": gaps,
        "next_step": next_step,
    }


def collect_and_evaluate(
    *,
    repo: str | os.PathLike[str],
    workspace: str | os.PathLike[str],
    meminfo_path: str | os.PathLike[str] = "/proc/meminfo",
    mounts_path: str | os.PathLike[str] = "/proc/mounts",
    proc_self_cgroup: str | os.PathLike[str] = "/proc/self/cgroup",
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
    caps: dict | None = None,
    dense_aggregate_bytes: int | None = None,
    bounded_aggregate_bytes: int | None = None,
    crosscheck: bool = True,
) -> dict:
    meminfo = read_meminfo(meminfo_path)
    mount = read_mount_for(workspace, mounts_path)
    cgroup = read_self_cgroup(
        proc_self_cgroup=proc_self_cgroup, cgroup_root=cgroup_root
    )
    runtime = read_runtime()
    source = probe_source(repo)
    psutil_check = cross_check_psutil(meminfo) if crosscheck else None
    return evaluate_preflight(
        meminfo=meminfo,
        mount=mount,
        cgroup=cgroup,
        runtime=runtime,
        source=source,
        psutil_check=psutil_check,
        caps=caps,
        dense_aggregate_bytes=dense_aggregate_bytes,
        bounded_aggregate_bytes=bounded_aggregate_bytes,
    )


def _human_summary(report: dict) -> str:
    host = report["host"]
    caps = report["caps"]
    lines = [
        f"M3 Linux preflight: {report['status']} (not an M3 qualification)",
        f"  MemAvailable: {_mib(host.get('mem_available_bytes'))} MiB"
        f" / MemTotal {_mib(host.get('mem_total_bytes'))} MiB",
        f"  swap free: {_mib(host.get('swap_free_bytes'))} MiB"
        f" / total {_mib(host.get('swap_total_bytes'))} MiB",
        f"  reserve: {_mib(caps['reserve_bytes'])} MiB;"
        f" proposed pressure <= {_mib(caps['pressure_max_bytes'])} MiB;"
        f" raw after pressure: {_mib(host.get('raw_after_pressure_bytes'))} MiB",
        f"  filesystem: {report['filesystem'].get('fstype')} at "
        f"{report['filesystem'].get('mount_point')} "
        f"({_mib(report['filesystem'].get('free_bytes'))} MiB free)",
        f"  cgroup: {report['cgroup'].get('cgroup_path')} "
        f"limit={report['cgroup'].get('limit_bytes') or 'max'}",
        "  product seam: declared_recovery_config_key="
        f"{report['source'].get('declared_recovery_config_key')} "
        f"headless_product_cli={report['source'].get('headless_product_cli')}",
    ]
    if report["blockers"]:
        lines.append("  blockers:")
        for item in report["blockers"]:
            lines.append(f"    - {item['code']}: {item['detail']}")
    if report["warnings"]:
        lines.append("  warnings:")
        for item in report["warnings"]:
            lines.append(f"    - {item['code']}: {item['detail']}")
    if report["gaps"]:
        lines.append("  gaps:")
        for item in report["gaps"]:
            lines.append(f"    - {item['code']}: {item['detail']}")
    lines.append(f"  next: {report['next_step']}")
    return "\n".join(lines)


def _exclusive_write(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Linux feasibility preflight for the ZeGrid M3 "
            "physical-pressure protocol. Never allocates pressure."
        )
    )
    default_repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo", default=str(default_repo))
    parser.add_argument("--workspace", default=str(default_repo))
    parser.add_argument("--meminfo", default="/proc/meminfo")
    parser.add_argument("--mounts", default="/proc/mounts")
    parser.add_argument("--proc-self-cgroup", default="/proc/self/cgroup")
    parser.add_argument("--cgroup-root", default="/sys/fs/cgroup")
    parser.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES)
    parser.add_argument(
        "--pressure-bytes", type=int, default=DEFAULT_PRESSURE_MAX_BYTES
    )
    parser.add_argument(
        "--pressure-max-bytes", type=int, default=DEFAULT_PRESSURE_MAX_BYTES
    )
    parser.add_argument(
        "--min-disk-free-bytes", type=int, default=DEFAULT_MIN_DISK_FREE_BYTES
    )
    parser.add_argument("--dense-aggregate-bytes", type=int, default=None)
    parser.add_argument("--bounded-aggregate-bytes", type=int, default=None)
    parser.add_argument(
        "--no-psutil-crosscheck",
        action="store_true",
        help="skip the optional psutil equivalence read",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="optional owned JSON path; refuses to overwrite an existing file",
    )
    args = parser.parse_args(argv)

    # Public CLI contract: reject unsafe/weak parameters before any probe or
    # output.  The pure evaluator also fails closed, so a direct caller cannot
    # obtain READY outside this contract.
    parameter_errors = []
    if args.reserve_bytes < CONTRACT_MIN_RESERVE_BYTES:
        parameter_errors.append(
            f"--reserve-bytes must be >= {CONTRACT_MIN_RESERVE_BYTES} B (1 GiB); "
            "stricter values only"
        )
    if args.pressure_bytes <= 0:
        parameter_errors.append("--pressure-bytes must be > 0")
    if args.pressure_bytes > CONTRACT_MAX_PRESSURE_BYTES:
        parameter_errors.append(
            f"--pressure-bytes must be <= {CONTRACT_MAX_PRESSURE_BYTES} B "
            "(512 MiB)"
        )
    if args.pressure_max_bytes <= 0:
        parameter_errors.append("--pressure-max-bytes must be > 0")
    if args.pressure_max_bytes > CONTRACT_MAX_PRESSURE_BYTES:
        parameter_errors.append(
            f"--pressure-max-bytes must be <= {CONTRACT_MAX_PRESSURE_BYTES} B "
            "(512 MiB)"
        )
    if args.pressure_bytes > args.pressure_max_bytes:
        parameter_errors.append(
            "--pressure-bytes must be <= --pressure-max-bytes"
        )
    if args.min_disk_free_bytes <= 0:
        parameter_errors.append("--min-disk-free-bytes must be > 0")
    if parameter_errors:
        for message in parameter_errors:
            print(f"m3_linux_preflight: {message}", file=sys.stderr)
        return 3

    caps = {
        "reserve_bytes": args.reserve_bytes,
        "pressure_bytes": args.pressure_bytes,
        "pressure_max_bytes": args.pressure_max_bytes,
        "min_disk_free_bytes": args.min_disk_free_bytes,
    }
    try:
        report = collect_and_evaluate(
            repo=args.repo,
            workspace=args.workspace,
            meminfo_path=args.meminfo,
            mounts_path=args.mounts,
            proc_self_cgroup=args.proc_self_cgroup,
            cgroup_root=args.cgroup_root,
            caps=caps,
            dense_aggregate_bytes=args.dense_aggregate_bytes,
            bounded_aggregate_bytes=args.bounded_aggregate_bytes,
            crosscheck=not args.no_psutil_crosscheck,
        )
    except Exception as exc:  # noqa: BLE001 - tool failure is code 3
        print(
            f"m3_linux_preflight: internal failure: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3

    if args.json_out:
        out = Path(args.json_out).expanduser()
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            _exclusive_write(out, report)
        except FileExistsError:
            print(
                f"m3_linux_preflight: refusing to overwrite existing {out}",
                file=sys.stderr,
            )
            return 3
        except OSError as exc:
            print(
                f"m3_linux_preflight: cannot write {out}: {exc}",
                file=sys.stderr,
            )
            return 3

    print(_human_summary(report))
    return 0 if report["status"] == "READY_TO_PLAN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
