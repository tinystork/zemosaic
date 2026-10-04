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


def build_runtime_snapshot(
    psutil_module: Any,
    zemosaic_utils_module: Any,
    zemosaic_utils_available: bool,
) -> dict[str, Any]:
    """Best-effort RAM/VRAM/pid snapshot for crash forensics.

    ``psutil_module``, ``zemosaic_utils_module`` and ``zemosaic_utils_available``
    are injected by the worker adapter so that monkeypatching the worker globals
    still takes effect; pid/ppid/time use the standard library directly.
    """

    snap: dict[str, Any] = {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "ts_unix": time.time(),
    }
    try:
        vm = psutil_module.virtual_memory()
        snap.update(
            {
                "ram_used_mb": float(getattr(vm, "used", 0.0)) / (1024.0 * 1024.0),
                "ram_total_mb": float(getattr(vm, "total", 0.0)) / (1024.0 * 1024.0),
                "ram_pct": float(getattr(vm, "percent", 0.0)),
            }
        )
    except Exception:
        pass

    try:
        if (
            zemosaic_utils_available
            and zemosaic_utils_module
            and hasattr(zemosaic_utils_module, "get_gpu_vram_info")
        ):
            vram_used, vram_total, vram_free = zemosaic_utils_module.get_gpu_vram_info()
            snap.update(
                {
                    "gpu_used_mb": float(vram_used) if vram_used is not None else None,
                    "gpu_total_mb": float(vram_total) if vram_total is not None else None,
                    "gpu_free_mb": float(vram_free) if vram_free is not None else None,
                }
            )
    except Exception:
        pass
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
