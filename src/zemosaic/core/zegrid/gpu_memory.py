"""ZM-ZEGRID-R55-E11 — bounded GPU device-memory boundary helpers.

This module owns the *lifetime* half of the GPU streaming contract that
:mod:`zemosaic.core.zegrid.gpu` deliberately does not model: synchronising the
device before deciding what is unused, dropping unreachable temporaries, and
purging the CuPy allocator pool at a cell boundary so the NEXT fresh VRAM
observation is not polluted by blocks this process already finished with.

Design rules (all enforced here):

* **No CuPy import at module import time.** ``xp`` may be a CuPy module handed
  in by the caller (the canonical streaming backend) or ``None``, in which case
  CuPy is imported lazily *only* when a GPU explicit path asks for it. GPU OFF
  never reaches this code, so it never initialises CUDA.
* **Synchronise BEFORE purging.** ``free_all_blocks()`` must not run while
  kernels that still reference pool blocks are in flight. The order here is
  ``sync -> free_all_blocks -> snapshot``.
* **Never free live arrays.** ``free_all_blocks`` only returns blocks the pool
  already owns and that are not checked out; this helper never calls
  ``memory_free`` on a live array and never walks a live graph.
* **Never raise, never mask.** Every failure is captured as a short classified
  token in the returned record. The caller (a consumer that may be unwinding a
  science error) keeps its original exception.
* **Snapshots are labelled.** ``memGetInfo`` and pool counters are
  instantaneous observations, explicitly **not** a measured high-water mark.

The returned dictionaries are plain JSON-safe values so the caller can persist
them in the RUN telemetry/manifest without a second conversion layer.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "GPU_MEMORY_SCHEMA",
    "GPU_MEMORY_SNAPSHOT_LABEL",
    "device_memory_snapshot",
    "gpu_cell_boundary_cleanup",
]

# Version of the persisted record shape. Bump when keys change meaning.
GPU_MEMORY_SCHEMA = "zemosaic-r55e11-gpu-memory-boundary-v1"

# Every snapshot says this out loud: these are point observations, not peaks.
GPU_MEMORY_SNAPSHOT_LABEL = "instantaneous_snapshot_not_high_water"


def _lazy_cupy():
    """Import CuPy on an explicit GPU path only (never at module import)."""
    import cupy as _cp  # noqa: PLC0415 - lazy by contract

    return _cp


def _resolve_xp(xp):
    """Return ``(xp_or_None, errors)``; ``None`` means "no device path"."""
    errors: list[str] = []
    if xp is not None:
        return xp, errors
    try:
        return _lazy_cupy(), errors
    except Exception as exc:  # pragma: no cover - environment dependent
        errors.append(f"cupy_import:{type(exc).__name__}")
        return None, errors


def _is_numpy(xp) -> bool:
    mod = getattr(xp, "__name__", "")
    return mod == "numpy"


def _read_pool(pool) -> tuple[dict, list[str]]:
    """Best-effort pool counters; absent attributes are reported, not guessed."""
    out: dict = {
        "pool_used_bytes": None,
        "pool_total_bytes": None,
        "pool_free_bytes": None,
        "pinned_pool_used_bytes": None,
        "pinned_pool_total_bytes": None,
    }
    errors: list[str] = []
    if pool is None:
        return out, errors
    for attr, key in (
        ("used_bytes", "pool_used_bytes"),
        ("total_bytes", "pool_total_bytes"),
        ("free_bytes", "pool_free_bytes"),
    ):
        try:
            member = getattr(pool, attr)
        except Exception as exc:  # defensive: property may raise
            errors.append(f"{key}:{type(exc).__name__}")
            continue
        try:
            # CuPy exposes these counters as BOUND METHODS
            # (``pool.used_bytes()``); doubles/future backends may expose plain
            # attributes. Call when callable; never cast a bound method to int
            # (that was the historical TypeError).
            value = member() if callable(member) else member
            out[key] = int(value) if value is not None else None
        except Exception as exc:  # defensive
            errors.append(f"{key}:{type(exc).__name__}")
    return out, errors


def device_memory_snapshot(xp=None, *, device_index: int = 0) -> dict:
    """One instantaneous device/pool observation (never raises, never a peak).

    ``memGetInfo`` free/total are the driver's view; the pool counters describe
    THIS process's CuPy default allocator. The two must not be added together:
    pool blocks are already counted as used in ``memGetInfo``.
    """
    resolved, errors = _resolve_xp(xp)
    record: dict[str, Any] = {
        "schema": GPU_MEMORY_SCHEMA,
        "label": GPU_MEMORY_SNAPSHOT_LABEL,
        "available": False,
        "mem_get_info_free_bytes": None,
        "mem_get_info_total_bytes": None,
        "pool_used_bytes": None,
        "pool_total_bytes": None,
        "pool_free_bytes": None,
        "pinned_pool_used_bytes": None,
        "pinned_pool_total_bytes": None,
        "errors": errors,
    }
    if resolved is None or _is_numpy(resolved):
        return record
    try:
        with resolved.cuda.Device(int(device_index)):
            free_bytes, total_bytes = resolved.cuda.runtime.memGetInfo()
        record["mem_get_info_free_bytes"] = int(free_bytes)
        record["mem_get_info_total_bytes"] = int(total_bytes)
        record["available"] = True
    except Exception as exc:
        errors.append(f"mem_get_info:{type(exc).__name__}")
    try:
        pool = resolved.get_default_memory_pool()
    except Exception as exc:
        errors.append(f"default_pool:{type(exc).__name__}")
        pool = None
    pool_facts, pool_errors = _read_pool(pool)
    record.update(pool_facts)
    errors.extend(pool_errors)
    try:
        pinned = resolved.get_default_pinned_memory_pool()
    except Exception as exc:
        errors.append(f"pinned_pool:{type(exc).__name__}")
        pinned = None
    if pinned is not None:
        for attr, key in (
            ("used_bytes", "pinned_pool_used_bytes"),
            ("total_bytes", "pinned_pool_total_bytes"),
        ):
            try:
                member = getattr(pinned, attr)
                value = member() if callable(member) else member
                record[key] = int(value) if value is not None else None
            except Exception as exc:
                errors.append(f"{key}:{type(exc).__name__}")
    return record


def gpu_cell_boundary_cleanup(
    xp=None, *, device_index: int = 0, reason: str = "cell_boundary"
) -> dict:
    """Synchronise, purge only unused pool blocks, then re-observe.

    Returns a JSON-safe record::

        {
          reason, available, sync_ok, purge_ok,
          before: <snapshot>, after: <snapshot>,
          cleanup_failures: [...],
        }

    The returned ``before``/``after`` snapshots are labelled instantaneous
    observations. ``before`` is taken before the explicit sync, so a caller can
    see the pending-work state; ``after`` is the post-purge device view that the
    next RUN checkpoint may safely compare against.
    """
    resolved, errors = _resolve_xp(xp)
    record: dict[str, Any] = {
        "schema": GPU_MEMORY_SCHEMA,
        "label": GPU_MEMORY_SNAPSHOT_LABEL,
        "reason": str(reason),
        "available": False,
        "sync_ok": False,
        "purge_ok": False,
        "before": None,
        "after": None,
        "cleanup_failures": list(errors),
    }
    if resolved is None or _is_numpy(resolved):
        return record
    record["available"] = True
    before = device_memory_snapshot(resolved, device_index=device_index)
    record["before"] = before
    # 1) SYNCHRONISE FIRST: no pool mutation while work may still reference it.
    try:
        with resolved.cuda.Device(int(device_index)):
            resolved.cuda.Stream.null.synchronize()
        record["sync_ok"] = True
    except Exception as exc:
        record["cleanup_failures"].append(f"sync:{type(exc).__name__}")
    # 2) Purge only blocks the allocator already owns and no array references.
    try:
        pool = resolved.get_default_memory_pool()
        pool.free_all_blocks()
        # Pinned (host) blocks are independent of VRAM; clean them too so the
        # next cell does not inherit stale pinned buffers.
        try:
            resolved.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as exc:
            record["cleanup_failures"].append(
                f"pinned_purge:{type(exc).__name__}"
            )
        record["purge_ok"] = True
    except Exception as exc:
        record["cleanup_failures"].append(f"purge:{type(exc).__name__}")
    # 3) Observe AFTER, so the caller's next decision uses the post-boundary view.
    record["after"] = device_memory_snapshot(resolved, device_index=device_index)
    return record
