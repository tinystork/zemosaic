"""ZM-ZEGRID-R22 — GPU backend probe + VRAM-bounded tiled planner + error classifier.

The ZeGrid engine is CPU-only by default, but honours an explicit GPU preference
resolved from the product's GPU flags (see ``zemosaic_zegrid_mode.
resolve_gpu_preference``). This module owns the runtime facts the engine needs to
use the GPU SAFELY and HONESTLY:

* ``probe_gpu_backend`` — a lazy, dependency-guarded probe returning whether CuPy
  + a CUDA device are actually usable, plus the device name and VRAM total/free
  (never initialises CuPy at import time; never claims GPU from CuPy alone).
* ``vram_budget_bytes`` — the SAFE budget (free × fraction); returns ``None``
  (degrade) when the computed budget is below the minimum floor — NEVER inflates
  (H3 fix).
* ``choose_gpu_tile_size`` / ``gpu_tile_workspace_bytes`` — a DETERMINISTIC
  VRAM-bounded tile planner so the streaming phase-2 GPU path materialises at most
  ``O(N x tile_area)`` on the device (never ``N x full-cell``). The workspace bound
  now models the halo expansion (footprint taper reach) AND the simultaneous
  float64 copies/temporaries + a conservative allocator-retention factor.
* ``is_gpu_runtime_error`` — a narrow classifier for CuPy OOM / CUDA runtime /
  driver / kernel / compile failures (traverses ``__cause__``/``__context__``),
  so the caller can degrade a GPU batch to exact CPU WITHOUT swallowing arbitrary
  science/programming errors.

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
    "is_gpu_runtime_error",
    "GPU_VRAM_SAFETY_FRACTION",
    "GPU_MIN_VRAM_BUDGET_BYTES",
    "GPU_WORKSPACE_BYTES_PER_CELL",
    "GPU_TILE_FIXED_OVERHEAD_BYTES",
    "GPU_TAPER_HALO_PX",
]

# Fraction of FREE VRAM budgeted for the ZeGrid tiled workspace (leave headroom for
# the CUDA context / allocator retention / any other process on a shared laptop GPU).
GPU_VRAM_SAFETY_FRACTION = 0.5

# Absolute floor: below this the tiled GPU path cannot hold even a small tile and
# the engine degrades loudly to exact CPU. The budget is NEVER raised up to this
# floor (H3): a computed budget below the floor returns None.
GPU_MIN_VRAM_BUDGET_BYTES = 64 * 1024 * 1024

# Footprint-taper halo reach (must match canonical_streaming._TAPER_HALO_PX =
# ceil(taper_px=8) + 1 = 9). The streaming phase-2 extends each interior tile by
# this many pixels per side, so the DEVICE workspace is over (side + 2*halo)^2,
# not side^2.
GPU_TAPER_HALO_PX = 9

# Conservative per-(pixel, channel, frame) bytes for the GPU phase-2 tile
# workspace. Modelled arrays (all float64 unless noted): images64 + wmap(1ch) +
# masked_images + w_contrib + product + orig2 + masked + winsor + sort transient
# + a couple of elementwise temporaries, plus the bool masks. ~9 f64 arrays x 8B
# = 72 B, rounded UP to 96 B to cover the allocator's block rounding / pool
# retention between simultaneous arrays (H3 rework-1). The device high-water can
# still exceed the live set (CuPy pool does not return freed blocks to the OS), so
# the caller ALSO frees pool blocks between tiles and the 50% headroom covers the
# residual + the CUDA context.
GPU_WORKSPACE_BYTES_PER_CELL = 96

# Fixed per-tile CUDA launch/allocator overhead (bytes), conservative.
GPU_TILE_FIXED_OVERHEAD_BYTES = 4 * 1024 * 1024

# Exception class names (must ALSO be from the cupy module tree — a plain Python
# ``MemoryError`` is NOT a GPU error) that classify a runtime GPU failure worth a
# one-shot exact-CPU whole-batch rerun.
_GPU_ERROR_CLASS_NAMES = frozenset({
    "OutOfMemoryError",
    "CUDARuntimeError",
    "CUDADriverError",
    "CUDAMemoryError",
    "CudaAPIError",
    "CompileException",
    "NVRTCError",
})


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
    is unavailable, the free VRAM is unknown, OR the computed budget
    (``free * GPU_VRAM_SAFETY_FRACTION``) is below :const:`GPU_MIN_VRAM_BUDGET_BYTES`.
    The budget is NEVER inflated up to the floor (H3 fix): a too-small budget
    degrades loudly to exact CPU.
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
    if budget < GPU_MIN_VRAM_BUDGET_BYTES:
        return None
    return budget


def gpu_tile_workspace_bytes(n_contributors: int, th: int, tw: int, channels: int) -> int:
    """Worst-case device bytes for ONE (extended) tile's phase-2 workspace.

    ``th``/``tw`` are the EXTENDED tile dims (interior + 2×halo). Includes the
    fixed per-tile overhead.
    """
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
    """Largest INTERIOR square tile whose extended GPU workspace fits the budget.

    Deterministic, pure. Accounts for the footprint-taper halo
    (:const:`GPU_TAPER_HALO_PX`) — the device workspace is over
    ``(side + 2*halo)^2``, not ``side^2``. ``None`` means the tiled GPU path cannot
    safely hold even the minimum tile within the budget (the caller degrades to
    CPU). The tile is clamped to the patch dims.
    """
    if vram_budget is None or int(vram_budget) <= 0:
        return None
    n = max(1, int(n_contributors))
    c = max(1, int(channels))
    h, w = int(patch_hw[0]), int(patch_hw[1])
    budget = int(vram_budget)

    per_area = n * c * GPU_WORKSPACE_BYTES_PER_CELL
    # Extended tile AREA (e^2) whose workspace fits: budget >= n*e^2*c*W + F.
    max_ext_area = (budget - GPU_TILE_FIXED_OVERHEAD_BYTES) // per_area
    if max_ext_area < 1:
        return None
    max_ext_side = int(math.isqrt(max_ext_area))
    interior_side = max(1, max_ext_side - 2 * GPU_TAPER_HALO_PX)
    # Clamp to the patch (never larger than the patch).
    interior_side = min(interior_side, max(h, w))
    return interior_side


def is_gpu_runtime_error(exc: BaseException) -> bool:
    """Narrow classifier: is ``exc`` a CuPy OOM / CUDA runtime / driver / kernel /
    compile failure (traversing wrapped ``__cause__``/``__context__`` chains)?

    Returns True ONLY for exceptions whose class name is a known CUDA/CuPy error
    AND whose defining module is inside the ``cupy`` package tree — so an arbitrary
    science/programming ``ValueError``/``MemoryError``/``RuntimeError`` is NEVER
    reclassified as a GPU failure. Import-safe (never imports CuPy).
    """
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        mod = (type(cur).__module__ or "").lower()
        if "cupy" in mod and type(cur).__name__ in _GPU_ERROR_CLASS_NAMES:
            return True
        cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
    return False
