from __future__ import annotations

import ctypes
import importlib
import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger("ZeMosaicWorker").getChild("telemetry")
logger.propagate = True

SYSTEM_NAME = platform.system().lower()
IS_WINDOWS = SYSTEM_NAME == "windows"
CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None
PYNVML_AVAILABLE = importlib.util.find_spec("pynvml") is not None

# ZM-WIN-GPUPROBE: the crash-breadcrumb probe is CUDA-free by contract. It
# tries, in order: ``pynvml`` (if importable), NVML directly through ctypes
# (``nvml.dll`` on Windows / ``libnvidia-ml.so.1`` elsewhere) and finally the
# ``nvidia-smi`` CLI (short timeout + TTL cache so a 2 s heartbeat cannot
# spawn a process on every tick). CuPy / ``cuda.runtime.memGetInfo()`` is
# DELIBERATELY absent from this path: it would create a CUDA context on the
# heartbeat thread. The CuPy reader stays reserved for the telemetry owner.
_CTYPES = ctypes  # indirection so tests can fake the driver binding
_MONOTONIC = time.monotonic  # indirection so tests can drive the TTL clock
_NVIDIA_SMI_TIMEOUT_S = 2.0
_NVIDIA_SMI_TTL_S = 5.0
_NVIDIA_SMI_QUERY = (
    "--query-gpu=memory.used,memory.total,memory.free",
    "--format=csv,noheader,nounits",
)
_NVIDIA_SMI_CACHE: dict[str, Any] = {"at": None, "result": None}

_NVML_STATE: dict[str, Any] = {
    "checked": False,
    "ok": False,
    "handle": None,
    "name": None,
    "error": None,
    "detail": None,
}


def _gpu_info_defaults() -> dict[str, Any]:
    return {
        "gpu_used_mb": None,
        "gpu_total_mb": None,
        "gpu_free_mb": None,
        "gpu_util_percent": None,
        "gpu_mem_util_percent": None,
        "gpu_name": None,
        "gpu_backend": None,
        "gpu_sample_source": None,
    }


def _get_nvml_handle() -> Any | None:
    if _NVML_STATE["checked"]:
        return _NVML_STATE["handle"] if _NVML_STATE["ok"] else None
    _NVML_STATE["checked"] = True
    if not PYNVML_AVAILABLE:
        _NVML_STATE["error"] = "pynvml_unavailable"
        return None
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name_raw = pynvml.nvmlDeviceGetName(handle)
        name = name_raw.decode("utf-8", errors="replace") if isinstance(name_raw, bytes) else str(name_raw)
        _NVML_STATE["handle"] = handle
        _NVML_STATE["name"] = name
        _NVML_STATE["ok"] = True
        return handle
    except Exception as exc:
        _NVML_STATE["error"] = f"pynvml_failed:{type(exc).__name__}"
        _NVML_STATE["detail"] = str(exc)
        _NVML_STATE["ok"] = False
        _NVML_STATE["handle"] = None
        return None


def _pynvml_error_token() -> str:
    """Explicit, aggregated-reason token for the pynvml source."""
    return str(_NVML_STATE.get("error") or "pynvml_unavailable")


def _sample_gpu_via_pynvml() -> dict[str, Any]:
    info = _gpu_info_defaults()
    handle = _get_nvml_handle()
    if handle is None:
        return info
    try:
        import pynvml  # type: ignore

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        total_mb = float(mem.total) / (1024.0 * 1024.0)
        used_mb = float(mem.used) / (1024.0 * 1024.0)
        free_raw = getattr(mem, "free", None)
        free_mb = (
            float(free_raw) / (1024.0 * 1024.0)
            if free_raw is not None
            else max(0.0, total_mb - used_mb)
        )
        info.update(
            gpu_used_mb=used_mb,
            gpu_total_mb=total_mb,
            gpu_free_mb=free_mb,
            gpu_util_percent=float(getattr(util, "gpu", 0.0)),
            gpu_mem_util_percent=float(getattr(util, "memory", 0.0)),
            gpu_name=_NVML_STATE.get("name"),
            gpu_backend="nvml",
            gpu_sample_source="pynvml",
        )
    except Exception as exc:
        # Keep the failure explicit: a query failure is NOT "unavailable".
        _NVML_STATE["error"] = f"pynvml_failed:{type(exc).__name__}"
        _NVML_STATE["detail"] = str(exc)
    return info


def _sample_gpu_via_cupy() -> dict[str, Any]:
    info = _gpu_info_defaults()
    if not CUPY_AVAILABLE:
        return info
    try:
        import cupy  # type: ignore

        cupy.cuda.Device().use()
        free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
        total_mb = total_bytes / (1024 * 1024)
        free_mb = free_bytes / (1024 * 1024)
        used_mb = total_mb - free_mb
        info.update(
            gpu_total_mb=total_mb,
            gpu_free_mb=free_mb,
            gpu_used_mb=used_mb,
            gpu_backend="cupy",
            gpu_sample_source="cupy_meminfo",
        )
        try:
            props = cupy.cuda.runtime.getDeviceProperties(0)
            name_raw = props.get("name")
            if isinstance(name_raw, bytes):
                info["gpu_name"] = name_raw.decode("utf-8", errors="replace")
            elif name_raw:
                info["gpu_name"] = str(name_raw)
        except Exception:
            pass
    except Exception:
        pass
    return info


def _nvml_library_names() -> tuple[str, ...]:
    """Driver libraries tried by the ctypes NVML fallback (never CUDA)."""
    if IS_WINDOWS:
        return ("nvml.dll",)
    return ("libnvidia-ml.so.1", "libnvidia-ml.so")


def _load_nvml_library() -> tuple[Any | None, str | None]:
    """Load the NVML driver library, returning ``(library, error_token)``."""
    loader = getattr(_CTYPES, "WinDLL" if IS_WINDOWS else "CDLL", None)
    if loader is None:
        return None, "ctypes_nvml_failed:DLL"
    for name in _nvml_library_names():
        try:
            return loader(name), None
        except Exception:
            continue
    return None, "ctypes_nvml_failed:DLL"


def _nvml_bind(lib: Any, names: tuple[str, ...], restype: Any, argtypes: Any) -> Any | None:
    """Bind the first available NVML symbol and type it (best-effort)."""
    for name in names:
        fn = getattr(lib, name, None)
        if fn is None:
            continue
        try:
            fn.restype = restype
            fn.argtypes = argtypes
        except Exception:
            pass
        return fn
    return None


def _sample_gpu_via_ctypes_nvml() -> tuple[dict[str, Any] | None, str | None]:
    """Query the NVML driver directly through ctypes (no Python binding).

    Initialises NVML, reads device 0's memory info and name, then shuts NVML
    down again. It never loads CuPy and never creates a CUDA context.
    """
    lib, load_error = _load_nvml_library()
    if lib is None:
        return None, load_error

    init = _nvml_bind(lib, ("nvmlInit_v2", "nvmlInit"), _CTYPES.c_int, ())
    if init is None:
        return None, "ctypes_nvml_failed:init_symbol"
    get_handle = _nvml_bind(
        lib,
        ("nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex"),
        _CTYPES.c_int,
        (_CTYPES.c_uint, _CTYPES.POINTER(_CTYPES.c_void_p)),
    )
    if get_handle is None:
        return None, "ctypes_nvml_failed:handle_symbol"

    class _NvmlMemory(_CTYPES.Structure):
        _fields_ = (
            ("total", _CTYPES.c_ulonglong),
            ("used", _CTYPES.c_ulonglong),
            ("free", _CTYPES.c_ulonglong),
        )

    mem_fn = _nvml_bind(
        lib,
        ("nvmlDeviceGetMemoryInfo",),
        _CTYPES.c_int,
        (_CTYPES.c_void_p, _CTYPES.POINTER(_NvmlMemory)),
    )
    if mem_fn is None:
        return None, "ctypes_nvml_failed:memory_symbol"

    inited = False
    try:
        rc = int(init())
        if rc != 0:
            return None, f"ctypes_nvml_failed:init_rc{rc}"
        inited = True
        handle = _CTYPES.c_void_p()
        rc = int(get_handle(_CTYPES.c_uint(0), _CTYPES.byref(handle)))
        if rc != 0:
            return None, "ctypes_nvml_failed:no_device"
        mem = _NvmlMemory()
        rc = int(mem_fn(handle, _CTYPES.byref(mem)))
        if rc != 0:
            return None, f"ctypes_nvml_failed:meminfo_rc{rc}"
        total_mb = float(mem.total) / (1024.0 * 1024.0)
        used_mb = float(mem.used) / (1024.0 * 1024.0)
        if total_mb <= 0.0:
            return None, "ctypes_nvml_failed:zero_total"
        free_raw = getattr(mem, "free", None)
        free_mb = (
            float(free_raw) / (1024.0 * 1024.0)
            if free_raw is not None
            else max(0.0, total_mb - used_mb)
        )
        info = _gpu_info_defaults()
        info.update(
            gpu_used_mb=used_mb,
            gpu_total_mb=total_mb,
            gpu_free_mb=free_mb,
            gpu_name=_nvml_device_name(lib, handle),
            gpu_backend="nvml",
            gpu_sample_source="ctypes_nvml",
        )
        return info, None
    except Exception as exc:
        return None, f"ctypes_nvml_failed:{type(exc).__name__}"
    finally:
        if inited:
            shutdown = getattr(lib, "nvmlShutdown", None)
            if shutdown is not None:
                try:
                    shutdown()
                except Exception:
                    pass


def _nvml_device_name(lib: Any, handle: Any) -> str | None:
    """Best-effort device name through NVML (harmless when unsupported)."""
    try:
        name_fn = _nvml_bind(
            lib,
            ("nvmlDeviceGetName",),
            _CTYPES.c_int,
            (_CTYPES.c_void_p, _CTYPES.c_void_p, _CTYPES.c_uint),
        )
        if name_fn is None:
            return None
        buf = _CTYPES.create_string_buffer(96)
        if int(name_fn(handle, buf, _CTYPES.c_uint(96))) != 0:
            return None
        raw = getattr(buf, "value", None)
        if raw is None:
            raw = bytes(buf)
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace").rstrip("\x00")
        return str(raw).rstrip("\x00")
    except Exception:
        return None


def _nvidia_smi_executable() -> str | None:
    """Resolve ``nvidia-smi`` (PATH, then the Windows System32 fallback)."""
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if IS_WINDOWS:
        root = os.environ.get("SystemRoot") or r"C:\Windows"
        candidate = os.path.join(root, "System32", "nvidia-smi.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


def _query_nvidia_smi() -> tuple[dict[str, Any] | None, str | None]:
    """One short ``nvidia-smi`` memory query, parsed into used/total/free MiB."""
    exe = _nvidia_smi_executable()
    if not exe:
        return None, "nvidia_smi_not_found"
    try:
        proc = subprocess.run(
            [exe, *_NVIDIA_SMI_QUERY],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "nvidia_smi_timeout"
    except Exception as exc:
        return None, f"nvidia_smi_failed:{type(exc).__name__}"
    if int(getattr(proc, "returncode", 1)) != 0:
        return None, f"nvidia_smi_failed:rc{int(proc.returncode)}"
    lines = [ln.strip() for ln in str(proc.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None, "nvidia_smi_empty"
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) < 3:
        return None, "nvidia_smi_parse_failed"
    try:
        used_mb, total_mb, free_mb = (float(part) for part in parts[:3])
    except (TypeError, ValueError):
        return None, "nvidia_smi_parse_failed"
    if not (total_mb > 0.0):
        return None, "nvidia_smi_parse_failed"
    info = _gpu_info_defaults()
    info.update(
        gpu_used_mb=used_mb,
        gpu_total_mb=total_mb,
        gpu_free_mb=free_mb,
        gpu_backend="nvidia_smi",
        gpu_sample_source="nvidia_smi",
    )
    return info, None


def _sample_gpu_via_nvidia_smi() -> tuple[dict[str, Any] | None, str | None]:
    """TTL-cached ``nvidia-smi`` query (a heartbeat is every 2 s)."""
    now = _MONOTONIC()
    cached_at = _NVIDIA_SMI_CACHE.get("at")
    cached = _NVIDIA_SMI_CACHE.get("result")
    if (
        isinstance(cached_at, (int, float))
        and isinstance(cached, tuple)
        and (now - float(cached_at)) < _NVIDIA_SMI_TTL_S
    ):
        return cached
    result = _query_nvidia_smi()
    _NVIDIA_SMI_CACHE["at"] = now
    _NVIDIA_SMI_CACHE["result"] = result
    return result


def reset_gpu_probe_caches() -> None:
    """Clear the optional probe caches (tests / driver hot-plug)."""
    _NVIDIA_SMI_CACHE["at"] = None
    _NVIDIA_SMI_CACHE["result"] = None
    _NVML_STATE.update(
        {
            "checked": False,
            "ok": False,
            "handle": None,
            "name": None,
            "error": None,
            "detail": None,
        }
    )


def sample_gpu_via_nvml() -> dict[str, Any]:
    """Public CUDA-free GPU observation for crash breadcrumbs.

    Sources are tried in order and the FIRST that answers wins:

    1. ``pynvml`` (optional Python binding);
    2. NVML through ctypes (``nvml.dll`` / ``libnvidia-ml.so.1``);
    3. ``nvidia-smi --query-gpu=memory.used,memory.total,memory.free`` with a
       short timeout and a TTL cache.

    Every source queries the driver only; none initialises CUDA, so this is safe
    from the breadcrumb heartbeat thread and on a GPU-OFF run. When all sources
    fail, the fields stay null and ``gpu_unavailable_reason`` aggregates the
    explicit per-source tokens (never a fabricated 0).
    """
    reasons: list[str] = []
    info: dict[str, Any] | None = None
    source: str | None = None
    backend: str | None = None

    info = _sample_gpu_via_pynvml()
    if info.get("gpu_total_mb") is not None:
        source, backend = "pynvml", "nvml"
    else:
        reasons.append(_pynvml_error_token())
        info, ctypes_error = _sample_gpu_via_ctypes_nvml()
        if info is not None:
            source, backend = "ctypes_nvml", "nvml"
        else:
            reasons.append(ctypes_error or "ctypes_nvml_failed")
            info, smi_error = _sample_gpu_via_nvidia_smi()
            if info is not None:
                source, backend = "nvidia_smi", "nvidia_smi"
            else:
                reasons.append(smi_error or "nvidia_smi_failed")
                info = None

    if info is None:
        info = _gpu_info_defaults()
    info["gpu_source"] = source
    info["gpu_sample_source"] = source
    if source is not None:
        info["gpu_backend"] = backend
    info["gpu_unavailable_reason"] = (
        None if source is not None else ";".join(r for r in reasons if r)
    )
    return info


def _default_log_and_callback(
    message_key_or_raw,
    progress_value=None,
    level: str | None = "INFO",
    callback: Callable | None = None,
    **kwargs,
) -> None:
    """Lightweight logger + callback helper used by telemetry."""

    level_str = str(level).upper() if isinstance(level, str) else "INFO"
    try:
        logger.log(getattr(logging, level_str, logging.INFO), message_key_or_raw)
    except Exception:
        pass

    if callback:
        try:
            callback(message_key_or_raw, progress_value, level_str, **kwargs)
        except Exception:
            try:
                logger.debug("Telemetry callback failed", exc_info=True)
            except Exception:
                pass


def _sample_runtime_resources_for_telemetry() -> dict:
    """Return a best-effort snapshot of current CPU/RAM/GPU usage."""

    info = {
        "cpu_percent": None,
        "ram_used_mb": None,
        "ram_total_mb": None,
        "ram_available_mb": None,
    }
    info.update(_gpu_info_defaults())

    try:
        import psutil as _ps

        vm = _ps.virtual_memory()
        info["ram_total_mb"] = vm.total / (1024 * 1024)
        info["ram_available_mb"] = vm.available / (1024 * 1024)
        info["ram_used_mb"] = (vm.total - vm.available) / (1024 * 1024)
        info["cpu_percent"] = _ps.cpu_percent(interval=None)
    except Exception:
        pass

    try:
        gpu_info = _sample_gpu_via_pynvml()
        if gpu_info.get("gpu_total_mb") is None:
            gpu_info = _sample_gpu_via_cupy()
        info.update(gpu_info)
    except Exception:
        pass

    return info


class ResourceTelemetryController:
    _DEFAULT_FIELDS = [
        "timestamp_iso",
        "phase_index",
        "phase_name",
        "cpu_percent",
        "ram_used_mb",
        "ram_total_mb",
        "ram_available_mb",
        "gpu_used_mb",
        "gpu_total_mb",
        "gpu_free_mb",
        "gpu_util_percent",
        "gpu_mem_util_percent",
        "gpu_name",
        "gpu_backend",
        "gpu_sample_source",
        "files_done",
        "files_total",
        "tiles_done",
        "tiles_total",
        "eta_seconds",
        "cpu_workers",
        "rows_per_chunk",
        "gpu_rows_per_chunk",
        "max_chunk_bytes",
        "gpu_max_chunk_bytes",
        "use_gpu",
        "use_gpu_phase5",
    ]

    def __init__(
        self,
        enabled: bool,
        interval_sec: float,
        callback: Callable | None,
        csv_path: str | None = None,
        *,
        log_and_callback: Callable | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.interval_sec = float(interval_sec) if interval_sec and interval_sec > 0 else 1.5
        if self.interval_sec < 0.5:
            self.interval_sec = 0.5
        self._callback = callback
        self._csv_path = csv_path
        self._last_sample = 0.0
        self._csv_file = None
        self._csv_writer = None
        self._csv_header_written = False
        self._log_and_callback = log_and_callback or _default_log_and_callback

    def _open_csv_if_needed(self, fieldnames: list[str]) -> None:
        if not self._csv_path or self._csv_writer is not None:
            return
        try:
            import csv

            dir_path = os.path.dirname(self._csv_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            f = open(self._csv_path, "w", newline="", encoding="utf-8")
            self._csv_file = f
            self._csv_writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            self._csv_writer.writeheader()
            self._csv_header_written = True
        except Exception:
            self._csv_path = None
            self._csv_file = None
            self._csv_writer = None

    def emit_stats(self, context: dict | None = None, *, force: bool = False) -> None:
        if not self.enabled or self._callback is None:
            return
        import time as _time

        now = _time.monotonic()
        if (not force) and self._last_sample and (now - self._last_sample) < self.interval_sec:
            return
        self._last_sample = now

        base_context = context.copy() if isinstance(context, dict) else {}
        try:
            resources = _sample_runtime_resources_for_telemetry()
        except Exception:
            resources = {}
        payload = {}
        payload.update(base_context)
        payload.update(resources)
        payload["timestamp_iso"] = datetime.utcnow().isoformat() + "Z"

        try:
            self._log_and_callback(
                "STATS_UPDATE",
                progress_value=None,
                level="INFO",
                callback=self._callback,
                **payload,
            )
        except Exception:
            pass

        if self._csv_path:
            try:
                import csv

                if self._csv_writer is None:
                    fieldnames = list(dict.fromkeys(self._DEFAULT_FIELDS + sorted(payload.keys())))
                    self._open_csv_if_needed(fieldnames)
                if self._csv_writer is not None:
                    self._csv_writer.writerow(payload)
                    if self._csv_file is not None:
                        self._csv_file.flush()
            except Exception:
                self._csv_path = None

    def maybe_emit_stats(self, context: dict | None = None) -> None:
        self.emit_stats(context, force=False)

    def close(self) -> None:
        try:
            if self._csv_file is not None:
                self._csv_file.close()
        except Exception:
            pass
        self._csv_file = None
        self._csv_writer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
