from __future__ import annotations

import importlib
import logging
import os
import platform
from datetime import datetime
from typing import Callable

logger = logging.getLogger("ZeMosaicWorker").getChild("telemetry")
logger.propagate = True

SYSTEM_NAME = platform.system().lower()
IS_WINDOWS = SYSTEM_NAME == "windows"
CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None and IS_WINDOWS


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
        "gpu_used_mb": None,
        "gpu_total_mb": None,
        "gpu_free_mb": None,
    }

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
        if CUPY_AVAILABLE:
            import cupy  # type: ignore

            try:
                cupy.cuda.Device().use()
                free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
                total_mb = total_bytes / (1024 * 1024)
                free_mb = free_bytes / (1024 * 1024)
                used_mb = total_mb - free_mb
                info["gpu_total_mb"] = total_mb
                info["gpu_free_mb"] = free_mb
                info["gpu_used_mb"] = used_mb
            except Exception:
                pass
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
