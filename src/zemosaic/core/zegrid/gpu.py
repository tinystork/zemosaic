"""ZM-ZEGRID-R22 — GPU backend probe + VRAM-bounded tiled planner for ZeGrid.

The ZeGrid engine is CPU-only by default, but honours an explicit GPU preference
resolved from the product's GPU flags (see ``zemosaic_zegrid_mode.
resolve_gpu_preference``). This module owns the two runtime facts the engine needs
to use the GPU SAFELY and HONESTLY:

* ``probe_gpu_backend`` — a lazy, dependency-guarded probe returning whether CuPy
  + a CUDA device are actually usable, plus the device name and VRAM total/free
  (never initialises CuPy at import time; never claims GPU from CuPy alone).
* ``choose_gpu_tile_size`` / ``gpu_tile_workspace_bytes`` — a DETERMINISTIC
  VRAM-bounded tile planner so the streaming phase-2 GPU path materialises at most
  ``O(N x tile_area)`` on the device (never ``N x full-cell``). One tile at a
  time; the tile size is a pure function of (N, channels, patch, VRAM budget).

No science lives here; everything is import-safe (stdlib + numpy only at import).

Single-GPU-owner contract (documented, enforced by the caller): the GPU stack
phase runs ONE cell at a time (``cells_in_flight == 1`` for the stack phase), so
no two cell processes ever contend for the same device. The CPU cache build may
still run with bounded parallelism (it never touches the GPU).
"""

from __future__ import annotations

import math

import numpy as np

from zemosaic.core.canonical_stacking import canonical_gpu_available

__all__ = [
    "probe_gpu_backend",
    "vram_budget_bytes",
    "gpu_tile_workspace_bytes",
    "choose_gpu_tile_size",
    "GPU_VRAM_SAFETY_FRACTION",
    "GPU_MIN_VRAM_BUDGET_BYTES",
    "GPU_WORKSPACE_BYTES_PER_CELL",
    "GPU_TILE_FIXED_OVERHEAD_BYTES",
]

# Fraction of FREE VRAM budgeted for the ZeGrid tiled workspace (leave headroom for
# the CUDA context / allocator / any other process on a shared laptop GPU).
GPU_VRAM_SAFETY_FRACTION = 0.5

# Absolute floor: below this the tiled GPU path cannot hold even a small tile and
# the engine degrades loudly to exact CPU.
GPU_MIN_VRAM_BUDGET_BYTES = 64 * 1024 * 1024

# Conservative per-(pixel, channel, frame) bytes for the GPU phase-2 tile
# workspace: images64 (f64) + wmap (f64) + sort transient (f64) + masked/winsor
# (f64) + survivor bool + elementwise temporaries. Rounding UP (8 bytes x 6 = 48
# -> 56) leaves a safety margin for the allocator.
GPU_WORKSPACE_BYTES_PER_CELL = 56

# Fixed per-tile CUDA launch/allocator overhead (bytes), conservative.
GPU_TILE_FIXED_OVERHEAD_BYTES = 2 * 1024 * 1024


def probe_gpu_backend() -> dict:
    """Return a truthful, bounded GPU probe dict (never raises, never claims).

    ``available`` is True ONLY when ``canonical_gpu_available()`` (CuPy importable
    AND >=1 CUDA device) reports True; it is never inferred from CuPy init alone.
    On success it also reports ``device`` (name), ``vram_total_bytes`` and
    ``vram_free_bytes`` via ``cupy.cuda.runtime.memGetInfo``. On any failure it
    returns ``available=False`` with a stable ``reason``.
    """
    out = {
        "available": False,
        "device": None,
        "cupy_version": None,
        "vram_total_bytes": None,
        "vram_free_bytes": None,
        "reason": None,
    }
    if not canonical_gpu_available():
        out["reason"] = "cupy_or_cuda_unavailable"
        return out
    try:
        import cupy as _cp
    except Exception as exc:  # pragma: no cover - defensive
        out["reason"] = f"cupy_import_failed: {type(exc).__name__}"
        return out
    try:
        out["cupy_version"] = str(_cp.__version__)
        free_bytes, total_bytes = _cp.cuda.runtime.memGetInfo()
        out["vram_total_bytes"] = int(total_bytes)
        out["vram_free_bytes"] = int(free_bytes)
        props = _cp.cuda.runtime.getDeviceProperties(0)
        name = props.get("name")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        out["device"] = str(name) if name else None
        out["available"] = True
    except Exception as exc:  # pragma: no cover - defensive
        out["available"] = False
        out["reason"] = f"cupy_probe_failed: {type(exc).__name__}"
    return out


def vram_budget_bytes(probe: dict) -> int | None:
    """Safe VRAM budget (bytes) for the tiled GPU workspace, or None if unusable.

    ``probe`` is a :func:`probe_gpu_backend` dict. Returns ``None`` when the GPU
    is unavailable or the free VRAM is unknown; otherwise
    ``free_vram * GPU_VRAM_SAFETY_FRACTION``, floored to
    :const:`GPU_MIN_VRAM_BUDGET_BYTES` when positive.
    """
    if not probe.get("available"):
        return None
    free = probe.get("vram_free_bytes")
    if free is None:
        total = probe.get("vram_total_bytes")
        if total is None:
            return None
        free = total
    free = int(free)
    budget = int(free * GPU_VRAM_SAFETY_FRACTION)
    if budget <= 0:
        return None
    return max(budget, GPU_MIN_VRAM_BUDGET_BYTES)


def gpu_tile_workspace_bytes(n_contributors: int, th: int, tw: int, channels: int) -> int:
    """Worst-case device bytes for ONE tile's phase-2 workspace (rejection+combine)."""
    n = max(1, int(n_contributors))
    th = max(1, int(th))
    tw = max(1, int(tw))
    c = max(1, int(channels))
    cells = n * th * tw * c
    return cells * GPU_WORKSPACE_BYTES_PER_CELL + GPU_TILE_FIXED_OVERHEAD_BYTES


def choose_gpu_tile_size(
    n_contributors: int,
    channels: int,
    patch_hw: tuple[int, int],
    vram_budget: int | None,
) -> int | None:
    """Largest square tile whose GPU workspace fits the VRAM budget (or None).

    Deterministic, pure. ``None`` means the tiled GPU path cannot safely hold
    even the minimum tile within the budget (the caller then degrades to CPU).
    The tile is clamped to the patch dims (a tile >= the patch == single tile).
    """
    if vram_budget is None or int(vram_budget) <= 0:
        return None
    n = max(1, int(n_contributors))
    c = max(1, int(channels))
    h, w = int(patch_hw[0]), int(patch_hw[1])
    budget = int(vram_budget)

    # Max tile AREA (th*tw) whose workspace fits: budget >= n*th*tw*c*W + F.
    per_area = n * c * GPU_WORKSPACE_BYTES_PER_CELL
    max_area = (budget - GPU_TILE_FIXED_OVERHEAD_BYTES) // per_area
    if max_area < 1:
        return None
    side = int(math.isqrt(max_area))
    side = max(1, side)
    # Clamp to the patch (never larger than the patch, so it stays a single
    # bounded tile when the patch itself fits).
    side = min(side, max(h, w))
    return side
