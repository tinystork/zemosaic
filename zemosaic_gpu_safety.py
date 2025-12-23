"""
Lightweight GPU safety heuristics for ZeMosaic.

This module centralizes best-effort probes about the GPU/OS environment and
applies conservative clamps to parallel plans on risky systems (e.g. Windows
laptops with hybrid graphics). All optional dependencies are guarded so the
module can be imported on machines without psutil/WMI/CuPy.
"""

from __future__ import annotations

import logging
import os
import platform
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil may be missing
    psutil = None

try:  # pragma: no cover - optional dependency, Windows-only
    import wmi  # type: ignore
except Exception:  # pragma: no cover - wmi absent on most platforms
    wmi = None

try:
    from parallel_utils import ParallelCapabilities, ParallelPlan, detect_parallel_capabilities  # type: ignore
except Exception:  # pragma: no cover - keep imports optional
    ParallelCapabilities = Any  # type: ignore
    ParallelPlan = Any  # type: ignore

    def detect_parallel_capabilities() -> Any:  # type: ignore
        return None

LOGGER = logging.getLogger(__name__)


@dataclass
class GpuRuntimeContext:
    os_name: str
    platform_system: str
    gpu_available: bool
    gpu_name: str | None
    gpu_vendor: str
    vram_total_bytes: int | None
    vram_free_bytes: int | None
    has_battery: bool | None
    power_plugged: bool | None
    on_battery: bool
    is_windows: bool
    is_hybrid_graphics: bool | None
    safe_mode: bool
    reasons: list[str] = field(default_factory=list)


def _detect_vendor(name: str | bytes | None) -> str:
    if not name:
        return "unknown"
    
    if isinstance(name, bytes):
        try:
            name = name.decode("utf-8")
        except UnicodeDecodeError:
            return "unknown"

    lowered = name.lower()
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered or "rtx" in lowered:
        return "nvidia"
    if "intel" in lowered:
        return "intel"
    if "amd" in lowered or "radeon" in lowered or "advanced micro devices" in lowered:
        return "amd"
    if "apple" in lowered:
        return "apple"
    return "unknown"


def _probe_battery_status() -> Tuple[bool | None, bool | None, bool]:
    """Return ``(has_battery, power_plugged, on_battery)`` best effort."""

    has_battery: bool | None = None
    power_plugged: bool | None = None
    on_battery = False

    def _probe_windows_power_status_ctypes() -> Tuple[bool | None, bool | None, bool] | None:
        if not platform.system().lower().startswith("windows"):
            return None

        if importlib.util.find_spec("ctypes") is None:
            return None

        import ctypes
        from ctypes import wintypes

        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", wintypes.BYTE),
                ("BatteryFlag", wintypes.BYTE),
                ("BatteryLifePercent", wintypes.BYTE),
                ("SystemStatusFlag", wintypes.BYTE),
                ("BatteryLifeTime", wintypes.DWORD),
                ("BatteryFullLifeTime", wintypes.DWORD),
            ]

        status = SYSTEM_POWER_STATUS()
        try:
            ok = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
        except Exception:
            return None
        if not ok:
            return None

        ac_line_status = status.ACLineStatus
        if ac_line_status == 255:
            return None

        power_plugged = ac_line_status == 1
        on_battery = ac_line_status == 0

        battery_present: bool | None = None
        battery_flag = status.BatteryFlag
        if battery_flag != 255:
            battery_present = battery_flag != 128

        return battery_present, power_plugged, on_battery

    windows_probe = _probe_windows_power_status_ctypes()
    if windows_probe is not None:
        has_battery, power_plugged, on_battery = windows_probe

    if psutil is not None:
        try:
            info = psutil.sensors_battery()
            if info is not None:
                if has_battery is None:
                    has_battery = True
                if power_plugged is None:
                    power_plugged = getattr(info, "power_plugged", None)
                    if power_plugged is not None:
                        on_battery = power_plugged is False
        except Exception:
            pass

    if has_battery is None and wmi is not None and platform.system().lower().startswith("windows"):
        try:
            conn = wmi.WMI()  # type: ignore
            batteries = conn.Win32_Battery()  # type: ignore[attr-defined]
            if batteries is not None:
                has_battery = len(batteries) > 0
        except Exception:
            pass

    return has_battery, power_plugged, on_battery


def _probe_hybrid_graphics() -> bool | None:
    """Best-effort Windows hybrid graphics detection via WMI."""

    if wmi is None or not platform.system().lower().startswith("windows"):
        return None
    try:
        conn = wmi.WMI()  # type: ignore
        controllers = conn.Win32_VideoController()  # type: ignore[attr-defined]
        names = [getattr(ctrl, "Name", None) for ctrl in controllers or []]
        lowered = [str(n).lower() for n in names if n]
        has_intel = any("intel" in n for n in lowered)
        has_nvidia = any(("nvidia" in n) or ("geforce" in n) or ("quadro" in n) for n in lowered)
        has_amd = any(("amd" in n) or ("radeon" in n) for n in lowered)
        if has_intel and (has_nvidia or has_amd):
            return True
        if len(lowered) >= 2:
            return True
        if lowered:
            return False
    except Exception:
        return None
    return None


def probe_gpu_runtime_context(
    *,
    preferred_gpu_id: int | None = None,
    caps: ParallelCapabilities | None = None,
) -> GpuRuntimeContext:
    """
    Gather a runtime GPU safety context with minimal dependencies.

    ``preferred_gpu_id`` is informational only; it can be used in the future to
    bias per-GPU queries. Today it just flows through the API for completeness.
    """

    del preferred_gpu_id  # reserved for future selection hints
    platform_system = platform.system() or ""
    os_name = platform_system.lower()
    is_windows = os_name.startswith("windows")

    capabilities = caps
    if capabilities is None:
        try:
            capabilities = detect_parallel_capabilities()
        except Exception:
            capabilities = None

    gpu_name = getattr(capabilities, "gpu_name", None) if capabilities is not None else None
    vram_total = getattr(capabilities, "gpu_vram_total_bytes", None) if capabilities is not None else None
    vram_free = getattr(capabilities, "gpu_vram_free_bytes", None) if capabilities is not None else None
    gpu_available = bool(getattr(capabilities, "gpu_available", False)) if capabilities is not None else False

    vendor = _detect_vendor(gpu_name)
    has_battery, power_plugged, on_battery = _probe_battery_status()
    is_hybrid = _probe_hybrid_graphics()

    reasons: list[str] = []
    safe_mode = False
    if is_windows and on_battery is True:
        safe_mode = True
        reasons.append("on_battery")
    if is_windows and is_hybrid is True:
        safe_mode = True
        reasons.append("hybrid_graphics")
    if vendor == "intel" and (is_hybrid is False or is_hybrid is None):
        reasons.append("intel_only_gpu")
    if vendor == "unknown" and gpu_available:
        reasons.append("unknown_gpu_vendor")

    ctx = GpuRuntimeContext(
        os_name=os_name,
        platform_system=platform_system,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vendor=vendor,
        vram_total_bytes=vram_total,
        vram_free_bytes=vram_free,
        has_battery=has_battery,
        power_plugged=power_plugged,
        on_battery=on_battery,
        is_windows=is_windows,
        is_hybrid_graphics=is_hybrid,
        safe_mode=safe_mode,
        reasons=reasons,
    )
    return ctx


def _clamp_gpu_chunks(plan: ParallelPlan, ctx: GpuRuntimeContext) -> bool:
    """Clamp GPU chunk sizing when in safe mode."""

    try:
        use_gpu = bool(getattr(plan, "use_gpu", False))
    except Exception:
        use_gpu = False
    if not use_gpu or not ctx.safe_mode:
        return False

    budget_bytes = 256 * 1024 * 1024
    if ctx.safe_mode and (ctx.on_battery or ctx.is_hybrid_graphics):
        budget_bytes = max(64 * 1024 * 1024, min(budget_bytes, 128 * 1024 * 1024))

    if ctx.vram_free_bytes:
        try:
            budget_bytes = min(budget_bytes, int(ctx.vram_free_bytes * 0.33))
        except Exception:
            pass
    budget_bytes = max(budget_bytes, 32 * 1024 * 1024)

    def _estimate_bytes_per_row() -> int | None:
        candidates = []
        try:
            chunk = getattr(plan, "gpu_max_chunk_bytes", None)
            rows_hint = getattr(plan, "gpu_rows_per_chunk", None)
            if chunk and rows_hint:
                candidates.append(chunk / float(rows_hint))
        except Exception:
            pass

        try:
            chunk_cpu = getattr(plan, "max_chunk_bytes", None)
            rows_cpu = getattr(plan, "rows_per_chunk", None)
            if chunk_cpu and rows_cpu:
                candidates.append(chunk_cpu / float(rows_cpu))
        except Exception:
            pass

        for candidate in candidates:
            try:
                if candidate and candidate > 0:
                    return int(candidate)
            except Exception:
                continue
        return None

    bytes_per_row = _estimate_bytes_per_row()

    try:
        current_bytes = getattr(plan, "gpu_max_chunk_bytes", None)
    except Exception:
        current_bytes = None
    if current_bytes is None or current_bytes <= 0:
        current_bytes = getattr(plan, "max_chunk_bytes", None)

    new_chunk_bytes = budget_bytes
    if current_bytes:
        try:
            new_chunk_bytes = min(new_chunk_bytes, int(current_bytes))
        except Exception:
            pass
    new_chunk_bytes = max(new_chunk_bytes, 32 * 1024 * 1024)

    try:
        setattr(plan, "gpu_max_chunk_bytes", int(new_chunk_bytes))
    except Exception:
        pass

    if bytes_per_row is None or bytes_per_row <= 0:
        rows = 256
    else:
        try:
            rows = max(1, int(new_chunk_bytes // max(1, bytes_per_row)))
        except Exception:
            rows = 256
    rows = max(32, min(2048, rows))
    try:
        setattr(plan, "gpu_rows_per_chunk", int(rows))
    except Exception:
        pass

    budget_mib = float(new_chunk_bytes) / (1024.0 ** 2)
    vram_free_mib = (float(ctx.vram_free_bytes) / (1024.0 ** 2)) if ctx.vram_free_bytes else None
    try:
        LOGGER.info(
            "GPU_SAFETY: chosen gpu_rows_per_chunk=%s (budget=%.1f MiB, bytes_per_row=%s, vram_free=%s MiB, safe_mode=%s, on_battery=%s)",
            rows,
            budget_mib,
            bytes_per_row if bytes_per_row is not None else "unknown",
            f"{vram_free_mib:.1f}" if vram_free_mib is not None else "unknown",
            ctx.safe_mode,
            ctx.on_battery,
        )
    except Exception:
        pass

    return True

def apply_gpu_safety_to_parallel_plan(
    plan: ParallelPlan | None,
    caps: ParallelCapabilities | None,
    config: Mapping[str, Any] | None,
    *,
    operation: str,
    logger: logging.Logger | None = None,
) -> tuple[ParallelPlan | None, GpuRuntimeContext]:
    """Return a (possibly) clamped plan plus the runtime context."""

    _ = config  # Reserved for future per-config overrides
    log = logger or LOGGER
    ctx = probe_gpu_runtime_context(caps=caps)
    reasons = list(ctx.reasons)

    try:
        log.info(
            "[GPU_SAFETY] power_plugged=%s on_battery=%s (has_battery=%s)",
            ctx.power_plugged,
            ctx.on_battery,
            ctx.has_battery,
        )
    except Exception:
        pass

    if plan is None:
        summary = (
            "[GPU_SAFETY] operation=%s safe_mode=%d vendor=%s hybrid=%s battery=%s power_plugged=%s on_battery=%s plan=None reasons=%s"
            % (
                operation,
                1 if ctx.safe_mode else 0,
                ctx.gpu_vendor,
                ctx.is_hybrid_graphics,
                ctx.has_battery,
                ctx.power_plugged,
                ctx.on_battery,
                ",".join(reasons),
            )
        )
        try:
            log.info(summary)
        except Exception:
            pass
        return None, ctx

    disable_gpu = False
    if ctx.gpu_vendor == "intel" and not ctx.is_hybrid_graphics:
        disable_gpu = True
        reasons.append("disable_intel_only_gpu")
    if not ctx.gpu_available:
        disable_gpu = True
        reasons.append("gpu_unavailable")

    if disable_gpu:
        try:
            setattr(plan, "use_gpu", False)
        except Exception:
            pass

    clamped = _clamp_gpu_chunks(plan, ctx)
    if clamped:
        reasons.append("safe_mode_clamp")

    ctx.reasons = list(dict.fromkeys(reasons))
    summary = (
        "[GPU_SAFETY] operation=%s safe_mode=%d vendor=%s hybrid=%s battery=%s power_plugged=%s on_battery=%s "
        "plan_gpu=%d gpu_rows=%s gpu_chunk_mb=%.1f reasons=%s"
    ) % (
        operation,
        1 if ctx.safe_mode else 0,
        ctx.gpu_vendor,
        ctx.is_hybrid_graphics,
        ctx.has_battery,
        ctx.power_plugged,
        ctx.on_battery,
        1 if getattr(plan, "use_gpu", False) else 0,
        getattr(plan, "gpu_rows_per_chunk", None),
        float(getattr(plan, "gpu_max_chunk_bytes", 0) or 0) / (1024.0 ** 2),
        ",".join(ctx.reasons),
    )
    try:
        log.info(summary)
    except Exception:
        pass
    return plan, ctx


def apply_gpu_safety_to_phase5_flag(
    use_gpu_phase5_flag: bool,
    ctx: GpuRuntimeContext,
    *,
    logger: logging.Logger | None = None,
) -> bool:
    """Determine whether Phase 5 GPU should remain enabled under safety rules."""

    log = logger or LOGGER
    effective = bool(use_gpu_phase5_flag)
    reason = None

    intel_only = ctx.gpu_vendor == "intel" and not ctx.is_hybrid_graphics
    vram_unknown = ctx.vram_total_bytes is None and ctx.vram_free_bytes is None

    if intel_only:
        effective = False
        reason = "intel_only"
    elif ctx.safe_mode and (ctx.gpu_vendor == "unknown" or vram_unknown):
        effective = False
        reason = "safe_mode_unknown_gpu"

    if reason:
        try:
            log.info("[GPU_SAFETY] phase5_gpu=%d reason=%s", 1 if effective else 0, reason)
        except Exception:
            pass
    return effective


def get_env_safe_mode_flag(ctx: GpuRuntimeContext) -> bool:
    """Set an opt-in environment flag when safe mode is active."""

    if ctx.safe_mode:
        try:
            os.environ["ZEMOSAIC_GPU_SAFE_MODE"] = "1"
            return True
        except Exception:
            return False
    return False


__all__ = [
    "GpuRuntimeContext",
    "apply_gpu_safety_to_parallel_plan",
    "apply_gpu_safety_to_phase5_flag",
    "get_env_safe_mode_flag",
    "probe_gpu_runtime_context",
]
