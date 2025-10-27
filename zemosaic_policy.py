"""Shared policy helpers reused between the worker and GUI."""
from __future__ import annotations

from typing import Any, Mapping
import math
import os
import shutil

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - environment without psutil
    psutil = None  # type: ignore

__all__ = [
    "probe_system_resources_for_gui",
    "compute_auto_tile_caps",
    "compute_auto_max_raw_per_master_tile",
]


def probe_system_resources_for_gui(cache_dir: str | None = None) -> dict[str, float | None]:
    """Lightweight resource probe usable from the GUI preview."""

    info: dict[str, float | None] = {
        "ram_total_mb": None,
        "ram_available_mb": None,
        "usable_ram_mb": None,
        "disk_total_mb": None,
        "disk_free_mb": None,
        "usable_disk_mb": None,
        "gpu_total_mb": None,
        "gpu_free_mb": None,
        "usable_vram_mb": None,
    }

    try:
        if psutil is not None:
            vm = psutil.virtual_memory()
            info["ram_total_mb"] = vm.total / (1024 * 1024)
            info["ram_available_mb"] = vm.available / (1024 * 1024)
            available = info["ram_available_mb"]
            total = info["ram_total_mb"]
            if available is not None:
                usable = available * 0.6
                if total is not None:
                    usable = min(usable, total)
                info["usable_ram_mb"] = usable
    except Exception:
        pass

    try:
        target_dir = cache_dir if cache_dir and os.path.isdir(cache_dir) else os.getcwd()
        du = shutil.disk_usage(target_dir)
        disk_total_mb = du.total / (1024 * 1024)
        disk_free_mb = du.free / (1024 * 1024)
        info["disk_total_mb"] = disk_total_mb
        info["disk_free_mb"] = disk_free_mb
        info["usable_disk_mb"] = disk_free_mb * 0.7
    except Exception:
        pass

    return info


def compute_auto_tile_caps(
    resource_info: Mapping[str, Any] | None,
    per_frame_info: Mapping[str, Any] | None,
    *,
    policy_max: int = 50,
    policy_min: int = 8,
    disk_threshold_mb: float = 8192.0,
    user_max_override: int | None = None,
) -> dict[str, Any]:
    """Combine resource probes and frame heuristics into adaptive caps."""

    resource_info = dict(resource_info or {})
    per_frame_info = dict(per_frame_info or {})

    per_frame_mb = float(per_frame_info.get("per_frame_mb", 0.0) or 0.0)
    usable_ram_mb = float(resource_info.get("usable_ram_mb") or 0.0)
    ram_available_mb = float(resource_info.get("ram_available_mb") or 0.0)

    if user_max_override and user_max_override > 0:
        policy_max = min(policy_max, int(user_max_override))

    frames_by_ram = 0
    if per_frame_mb > 0 and usable_ram_mb > 0:
        frames_by_ram = max(0, int(math.floor(usable_ram_mb / per_frame_mb)))

    cap_candidate = policy_max if policy_max > 0 else frames_by_ram or policy_min
    if frames_by_ram > 0:
        cap_candidate = min(cap_candidate, frames_by_ram)
    cap_candidate = max(policy_min, cap_candidate)

    disk_free_mb = float(resource_info.get("disk_free_mb") or 0.0)
    usable_disk_mb = float(resource_info.get("usable_disk_mb") or 0.0)

    memmap_enabled = False
    memmap_budget_mb = None
    if frames_by_ram < policy_min and disk_free_mb > disk_threshold_mb:
        memmap_enabled = True
        if usable_disk_mb:
            memmap_budget_mb = usable_disk_mb * 0.2
        else:
            memmap_budget_mb = disk_free_mb * 0.2
        memmap_budget_mb = max(policy_min * per_frame_mb, memmap_budget_mb)

    gpu_hint = None
    usable_vram_mb = float(resource_info.get("usable_vram_mb") or 0.0)
    if per_frame_mb > 0 and usable_vram_mb > 0:
        gpu_hint = max(1, min(cap_candidate, int(math.floor(usable_vram_mb / per_frame_mb))))

    parallel_cap = 1
    if frames_by_ram and cap_candidate > 0:
        parallel_cap = max(1, frames_by_ram // max(1, cap_candidate))
    if memmap_enabled:
        parallel_cap = 1

    return {
        "per_frame_mb": per_frame_mb,
        "frames_by_ram": frames_by_ram,
        "cap": int(cap_candidate),
        "min_cap": int(policy_min),
        "memmap": bool(memmap_enabled),
        "memmap_budget_mb": memmap_budget_mb,
        "gpu_batch_hint": gpu_hint,
        "ram_available_mb": ram_available_mb,
        "parallel_groups": int(parallel_cap),
    }


def compute_auto_max_raw_per_master_tile(
    total_raws: int,
    resource_info: Mapping[str, Any] | None,
    per_frame_info: Mapping[str, Any] | None,
    *,
    user_value: int | None,
    min_tiles_floor: int = 14,
    return_details: bool = False,
) -> int | tuple[int, dict[str, int | str]]:
    """Determine the auto/manual cap for master tiles."""

    resource_info = dict(resource_info or {})
    per_frame_info = dict(per_frame_info or {})

    caps = compute_auto_tile_caps(resource_info, per_frame_info, policy_max=50, policy_min=8)
    cap_ram = int(max(8, min(50, caps.get("cap", 50))))

    N = int(max(0, total_raws))
    if N <= 150:
        cap_rule = 10
    elif N <= 400:
        cap_rule = 8
    elif N <= 3000:
        cap_rule = 8
    elif N <= 12000:
        cap_rule = 6
    else:
        cap_rule = 6

    manual_override = bool(user_value and user_value > 0)
    if manual_override:
        C = int(user_value)
    else:
        C = int(min(cap_ram, cap_rule))
        floor_tiles = max(1, int(min_tiles_floor))
        if N > 0 and C > 0:
            tiles_est = max(1, N // C)
            if tiles_est < floor_tiles:
                C = max(4, N // floor_tiles)
        C = int(max(4, min(200, C)))

    details = {
        "total_raws": N,
        "cap_ram": cap_ram,
        "cap_rule": cap_rule,
        "cap_final": int(C),
        "mode": "manual" if manual_override else "auto",
    }

    if return_details:
        return int(C), details
    return int(C)
