"""Stateless crash-breadcrumb engine (neutral core).

Mission: `ZM-ARCH-R2-LOT2B-CRASH-BREADCRUMB-EXTRACTION-20261004`.

This module is the *mechanical* extraction of the crash-autopsy side-channel
(append-only JSONL breadcrumbs + last-state JSON + runtime RAM/VRAM snapshot)
from ``zemosaic.zemosaic_worker`` into a neutral, stateless core.

It holds **no mutable breadcrumb runtime state** (no breadcrumb paths/mode/lock
singleton) and imports **no** ``zemosaic_worker``, Qt/Tk, GPU/CuPy, GUI, config,
solver, or science modules, and adds **no** new dependency.  Every observable
mutable value (breadcrumb paths, mode, lock) remains owned by the worker adapter,
which reads its own globals at call time and passes them in here.

Failure boundaries mirror the original worker functions exactly: the worker-facing
adapters preserve the original *selective* fail-open behavior, and this module adds
**no** additional exception handling beyond what those original functions already
did.  In particular, mode normalization may raise if an object's ``__str__`` raises,
and a snapshot callback (or the ``os``/``time`` calls in the snapshot) may raise
before the JSONL/state write ``try``/``except`` — exactly as in the pre-extraction
code.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

_VALID_CRASH_BREADCRUMB_MODES: frozenset[str] = frozenset(
    {"always", "errors_only", "off"}
)

__all__ = (
    "normalize_crash_breadcrumb_mode",
    "compute_crash_breadcrumb_paths",
    "build_runtime_snapshot",
    "emit_crash_breadcrumb",
)


def normalize_crash_breadcrumb_mode(mode: str | None) -> str:
    """Normalize a configured crash-breadcrumb mode string.

    Mirrors the original ``_configure_crash_breadcrumbs`` normalization exactly:
    ``str(mode or "always").strip().lower()`` with an invalid value falling back
    to ``"always"``.
    """

    mode_norm = str(mode or "always").strip().lower()
    if mode_norm not in _VALID_CRASH_BREADCRUMB_MODES:
        mode_norm = "always"
    return mode_norm


def compute_crash_breadcrumb_paths(
    output_folder: str | None, mode_norm: str
) -> tuple[Path | None, Path | None]:
    """Derive ``(jsonl_path, state_path)`` for an output folder and mode.

    Best-effort: returns ``(None, None)`` for an empty/off configuration and
    swallows any ``str``/``Path``/filesystem failure.  The output directory is
    created with ``mkdir(parents=True, exist_ok=True)``.
    """

    try:
        if not output_folder or mode_norm == "off":
            return (None, None)
        out_dir = Path(str(output_folder)).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        return (
            out_dir / "worker_crash_breadcrumbs.jsonl",
            out_dir / "worker_last_state.json",
        )
    except Exception:
        return (None, None)


def _process_memory_snapshot(psutil_module: Any) -> dict[str, Any]:
    """Portable process/process-tree memory snapshot (best-effort, labelled).

    The historical breadcrumb RAM fields were SYSTEM RAM, which made a worker
    crash look like the whole machine. The keys below separate the system view
    from the consuming process and its children. The tree value is an explicit
    SAMPLED SUM (children are read one after another, never simultaneously), so
    it is never presented as a true peak or a simultaneous footprint.
    """
    out: dict[str, Any] = {
        "process_rss_mb": None,
        "process_peak_rss_mb": None,
        "process_peak_rss_source": None,
        "process_tree_rss_sampled_sum_mb": None,
        "process_tree_rss_sampled_sum_label": "sampled_sum_not_simultaneous",
        "process_tree_sample_count": None,
        "process_memory_unavailable_reason": None,
    }
    try:
        proc = psutil_module.Process()
        info = proc.memory_info()
        out["process_rss_mb"] = float(getattr(info, "rss", 0.0)) / (1024.0 * 1024.0)
        peak = getattr(info, "peak_wset", None)
        if peak is not None:
            out["process_peak_rss_mb"] = float(peak) / (1024.0 * 1024.0)
            out["process_peak_rss_source"] = "psutil_peak_wset"
        else:
            # Linux exposes VmHWM; read it without a dependency or CUDA touch.
            try:
                with open("/proc/self/status", "r", encoding="ascii") as fh:
                    for line in fh:
                        if line.startswith("VmHWM:"):
                            kib = float(line.split()[1])
                            out["process_peak_rss_mb"] = kib / 1024.0
                            out["process_peak_rss_source"] = "proc_status_VmHWM"
                            break
            except Exception:
                pass
        tree = float(getattr(info, "rss", 0.0))
        count = 1
        try:
            for child in proc.children(recursive=True):
                try:
                    tree += float(child.memory_info().rss)
                    count += 1
                except Exception:
                    continue
        except Exception:
            pass
        out["process_tree_rss_sampled_sum_mb"] = tree / (1024.0 * 1024.0)
        out["process_tree_sample_count"] = int(count)
    except Exception as exc:
        out["process_memory_unavailable_reason"] = (
            f"psutil_process_unavailable: {type(exc).__name__}: {exc}"
        )
    return out


def _gpu_snapshot(gpu_probe: Callable[[], dict[str, Any]] | None) -> dict[str, Any]:
    """Best-effort GPU observation from an injected SAFE probe.

    The probe must be a driver-level (NVML) or already-initialised backend
    reader: it must never initialise CUDA. When absent, the fields stay null
    with an explicit reason (GPU OFF / no safe probe), never a fabricated 0.
    """
    out: dict[str, Any] = {
        "gpu_used_mb": None,
        "gpu_total_mb": None,
        "gpu_free_mb": None,
        "gpu_source": None,
        "gpu_unavailable_reason": "gpu_probe_not_enabled",
    }
    if gpu_probe is None:
        return out
    try:
        sample = gpu_probe()
    except Exception as exc:
        out["gpu_unavailable_reason"] = f"gpu_probe_failed: {type(exc).__name__}"
        return out
    if not isinstance(sample, dict):
        out["gpu_unavailable_reason"] = "gpu_probe_returned_non_dict"
        return out
    for key in ("gpu_used_mb", "gpu_total_mb", "gpu_free_mb", "gpu_source"):
        if key in sample:
            out[key] = sample.get(key)
    if out["gpu_total_mb"] is None:
        out["gpu_unavailable_reason"] = str(
            sample.get("gpu_unavailable_reason")
            or sample.get("gpu_sample_source")
            or "gpu_probe_no_total"
        )
    else:
        out["gpu_unavailable_reason"] = None
        if out["gpu_source"] is None:
            out["gpu_source"] = sample.get("gpu_sample_source") or sample.get(
                "gpu_backend"
            )
    return out


def build_runtime_snapshot(
    psutil_module: Any,
    zemosaic_utils_module: Any,
    zemosaic_utils_available: bool,
    *,
    gpu_probe: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Best-effort RAM/VRAM/pid snapshot for crash forensics.

    ``psutil_module`` and the legacy ``zemosaic_utils`` arguments are injected by
    the worker adapter so monkeypatching still takes effect. The legacy GPU
    helper ``get_gpu_vram_info`` never existed in ``zemosaic_utils`` (the probe
    always failed), so GPU data now comes only from the injected, CUDA-free
    ``gpu_probe``. Absent/unavailable GPU data is null + explicit reason.
    """
    snap: dict[str, Any] = {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "ts_unix": time.time(),
    }
    try:
        vm = psutil_module.virtual_memory()
        used_mb = float(getattr(vm, "used", 0.0)) / (1024.0 * 1024.0)
        total_mb = float(getattr(vm, "total", 0.0)) / (1024.0 * 1024.0)
        pct = float(getattr(vm, "percent", 0.0))
        # Legacy keys kept for store compatibility (they are SYSTEM RAM).
        snap.update(
            {
                "ram_used_mb": used_mb,
                "ram_total_mb": total_mb,
                "ram_pct": pct,
                # Explicit, unambiguous names for new consumers.
                "system_ram_used_mb": used_mb,
                "system_ram_total_mb": total_mb,
                "system_ram_percent": pct,
            }
        )
    except Exception:
        pass

    snap.update(_process_memory_snapshot(psutil_module))
    snap.update(_gpu_snapshot(gpu_probe))
    return snap


def emit_crash_breadcrumb(
    event: str,
    *,
    path_jsonl: Path | None,
    path_state: Path | None,
    mode: str,
    lock: Any,
    snapshot_fn: Callable[[], dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    """Append one JSON line breadcrumb and refresh the last-state JSON.

    Side-channel with the original selective failure boundaries: write and lock
    failures are swallowed, while failures before that section (for example in
    ``snapshot_fn``) still propagate exactly as they did before extraction.
    All runtime state (paths/mode/lock/snapshot callback) is injected so
    ``zemosaic_worker`` remains the single source of truth and
    monkeypatch/direct-assignment seams keep their exact effect.
    """

    if mode == "off":
        return
    if mode == "errors_only":
        e = str(event or "").upper()
        if not ("ERROR" in e or "EXCEPTION" in e or "CRASH" in e):
            return
    if path_jsonl is None and path_state is None:
        return

    record: dict[str, Any] = {
        "event": str(event),
        "iso": datetime.utcnow().isoformat() + "Z",
    }
    record.update(snapshot_fn())
    if payload:
        record.update(payload)

    try:
        with lock:
            if path_jsonl is not None:
                try:
                    with path_jsonl.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(record, ensure_ascii=False, default=str) + "\n"
                        )
                except Exception:
                    pass
            if path_state is not None:
                try:
                    with path_state.open("w", encoding="utf-8") as f:
                        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
                except Exception:
                    pass
    except Exception:
        pass
