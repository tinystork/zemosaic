"""SCI-05 — bounded-memory (streaming) canonical execution, EXACT equivalent of
:func:`zemosaic.core.canonical_engine.run_canonical_stack` (Levier 2, step 1).

``run_canonical_stack`` materialises the canonical pipeline over the full
``(N, H, W, C)`` patch, so peak memory scales as ``O(N x patch_area)``. On a
7.5 GiB host a cell with N=66 at ~410k px already peaks at ~4.1 GiB. This module
provides a **two-phase bounded-memory** executor whose peak scales as
``O(N x tile_area)`` (plus the ``O(patch_area)`` output planes), while producing
**bit-identical** results to ``run_canonical_stack`` for the same inputs.

Design (two phases, matching the frozen SCI-05 formulas exactly)
----------------------------------------------------------------
The canonical pipeline splits cleanly into operations that are *per-frame global*
and operations that are *per-pixel across the N axis*:

* **Phase 1 (per-frame statistics, <= 2 frames resident, memory O(patch_area)):**
  the per-frame normalization coefficients ``(a, b)``, the scalar quality weights
  ``q`` (sigma / FWHM), the valid counts, the reference selection and the
  exclusions. These only ever need **one or two frames** resident at a time, so
  the existing public stage functions are reused verbatim by feeding them 1-frame
  (reference) and 2-frame (reference + frame i) batches:

  * ``prepare_canonical_inputs([ref, i], ...)``
  * ``normalize_canonical_images(batch2, method, reference_index=0)``
  * ``compute_canonical_quality_weights(norm2, method)``

  The 2-frame batch puts the reference at index 0 and frame ``i`` at index 1, so
  ``reference_index=0`` reproduces the full-pipeline reference identity exactly
  and the per-frame fit is *ref <-> frame i* (which is the only cross-frame
  dependency a frame's normalization/weighting has).

* **Phase 2 (per tile, all N frames but only a tile + halo):** for each spatial
  tile, gather the N aligned tile slices (+ supports), re-apply the precomputed
  per-frame coefficients pointwise (exactly reproducing the normalized float32
  values and the post-normalization valid mask), then run rejection / combine /
  support accumulation **per pixel across N**. Every per-pixel statistic is
  tiling-invariant (``nanmedian`` / ``nanstd`` / ``nanquantile`` / ``sum`` along
  the N axis), so streaming a tile of width ``tile_cells`` reproduces the
  full-patch reductions bit-for-bit. The footprint taper is a per-frame EDT whose
  reach is ``feather_px``; a halo of ``ceil(taper_px) + 1`` makes the interior
  tile taper exact (see ``_halo_px``).

The only genuinely global scalars are aggregated across tiles exactly:
``iterations_used`` is ``min(iters, max_c t_c)`` (the max over cells of the
sigma-clip convergence step), and every count (initial/surviving/rejected/
low_n/degenerate/valid-output/…) is a disjoint sum over interior tile cells.

Provider contract (generic; no hard-coded arrays or files)
----------------------------------------------------------
``CanonicalFrameProvider`` is a small abstraction: ``get_raw_frame(i)`` returns
the raw (H,W)/(H,W,C) input + 2-D bool support for frame ``i`` (phase 1), and
``get_tile(i, y0, y1, x0, x1)`` returns the *prepared* float32 HWC tile + valid
bool tile for frame ``i`` over the requested bounds (phase 2). ``InMemoryCanonicalProvider``
wraps existing arrays (the parity tests use it). A memmap/file-backed provider is
a FUTURE step (not wired here).

Scope / non-goals
-----------------
* New module only; **no** existing canonical module is modified (behaviour or
  signature). No production dispatch wiring. No new dependency.
* Backend ``"cpu"`` only (streaming GPU is a future step; ``"gpu"`` raises a
  clear validation error, never a silent fallback).
* Explicit per-frame taper maps are not yet streamed (``none``/``footprint`` are
  supported, matching the required method matrix); an explicit taper raises a
  clear validation error rather than silently changing semantics.
* ``equalize_rgb`` is applied out-of-place on the final ``O(patch_area)`` science
  exactly as the engine does (it is a post-combine spatial-only step).
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from zemosaic.core.canonical_stacking import (
    CanonicalInputBatch,
    CanonicalStackFailure,
    CanonicalStackValidationError,
    FrameExclusion,
    # Public stage functions reused verbatim in phase 1 (fed 1/2-frame batches).
    compute_canonical_quality_weights,
    normalize_canonical_images,
    prepare_canonical_inputs,
    select_canonical_reference,
    # Private but deterministic, stable step primitives reused so the streamed
    # rejection / combine reductions are bit-identical to the full-patch ones.
    _as_bool_mask,
    _as_frame_array,
    _as_host_numpy,
    _astype_float32,
    _axis_sum_f64,
    _get_xp,
    _kappa_sigma_step,
    _nan_axis_median,
    _require_frame_sequence,
    _safe_divide,
    _validate_max_iters,
    _validate_reject_sigma,
    _validate_winsor_limit,
    _winsorized_sigma_clip_step,
    canonical_gpu_available,
)
from zemosaic.core.canonical_support import make_footprint_taper
from zemosaic.core.zegrid import gpu_memory
from zemosaic.core.canonical_engine import (
    CanonicalStackRequest,
    CanonicalStackResult,
)
from zemosaic.core.canonical_equalize import equalize_rgb_medians_copy

__all__ = [
    "CanonicalFrameProvider",
    "InMemoryCanonicalProvider",
    "FixedNormalization",
    "compute_fixed_normalization",
    "subset_fixed_normalization",
    "run_canonical_stack_streaming",
]

# ---------------------------------------------------------------------------
# Method tokens (mirror the frozen SCI-05 tokens; validation raises instead of
# falling back, so requested == effective always).
# ---------------------------------------------------------------------------

_NORM_METHODS = ("none", "linear_fit", "sky_mean")
_WEIGHT_METHODS = ("none", "noise_variance", "noise_fwhm")
_REJECT_METHODS = ("none", "kappa_sigma", "winsorized_sigma_clip")
_REJECT_TOKEN_REMOVED = "unsupported_removed_sci05"
_COMBINE_METHODS = ("mean", "median")


def _release_gpu_workspace(xp, *, reason: str = "tile_boundary") -> None:
    """Best-effort device boundary cleanup between tiles/cells (never raises).

    ZM-ZEGRID-R55-E11: the historical order was wrong. ``free_all_blocks()`` was
    called *before* the null-stream synchronisation and while the tile's device
    temporaries were still bound as locals, so (a) blocks still referenced by an
    in-flight kernel could be considered for release, and (b) the arrays that had
    just finished could not be returned to the pool until the NEXT tile, leaving
    the last tile's buffers retained across the inter-cell boundary.

    :func:`zemosaic.core.zegrid.gpu_memory.gpu_cell_boundary_cleanup` enforces the
    correct order (snapshot -> synchronise -> free only unused blocks -> snapshot)
    and never raises. Callers must invoke it only AFTER the device temporaries
    have left their bounded scope. A no-op for NumPy.
    """
    if xp is np:
        return
    gpu_memory.gpu_cell_boundary_cleanup(xp, reason=str(reason))


# ---------------------------------------------------------------------------
# Provider contract
# ---------------------------------------------------------------------------

@runtime_checkable
class CanonicalFrameProvider(Protocol):
    """Minimal provider of aligned frames (raw) and prepared tile slices.

    A conforming provider exposes the aligned ``(H, W)`` / ``(H, W, C)`` frame
    domain once, then serves raw frames for phase 1 and prepared tiles for
    phase 2. Tile bounds are half-open ``[y0:y1, x0:x1]`` in patch pixels.
    """

    n_frames: int
    height: int
    width: int
    channels: int
    original_mono: bool
    original_ndim: int
    original_shape: tuple

    def get_raw_frame(self, i: int) -> "tuple[np.ndarray, np.ndarray]":
        """Return ``(raw_image, raw_support)`` for frame ``i``.

        ``raw_image`` is the original ``(H, W)`` (mono) or ``(H, W, C)`` input;
        ``raw_support`` is the matching 2-D bool geometric support.
        """
        ...

    def get_tile(self, i: int, y0: int, y1: int, x0: int, x1: int) -> "tuple[np.ndarray, np.ndarray]":
        """Return the *prepared* float32 HWC tile + valid bool tile over bounds.

        Equivalent to ``prepare_canonical_inputs([frame[i] slice], [support[i] slice])``:
        float32 image, channel-invariant validity = support & finite-all-channels.
        """
        ...


class InMemoryCanonicalProvider:
    """Provider wrapping in-memory aligned arrays (exactly the engine's inputs).

    Validates the same batch-level invariants as ``prepare_canonical_inputs``
    (homogeneous HW mono or HWC C∈{1,3}, consistent H/W/C, matching support
    count) but never materialises the full ``(N, H, W, C)`` batch: frames are
    converted lazily, one slice at a time.
    """

    def __init__(self, images, geometric_support):
        n = _require_frame_sequence(images)
        if geometric_support is None or isinstance(geometric_support, (str, bytes, np.ndarray)):
            raise CanonicalStackValidationError(
                "geometric_support must be a sequence of 2-D boolean masks"
            )
        try:
            n_sup = len(geometric_support)
        except (TypeError, AttributeError):
            raise CanonicalStackValidationError(
                "geometric_support must be a sequence of 2-D boolean masks"
            )
        if n_sup != n:
            raise CanonicalStackValidationError(
                f"geometric_support has {n_sup} masks but there are {n} images"
            )

        shapes = []
        mono_flags = []
        for i in range(n):
            a, (h, w, c) = _as_frame_array(images[i], i)
            shapes.append((h, w, c))
            mono_flags.append(a.ndim == 2)
            _as_bool_mask(geometric_support[i], i, (h, w))

        if not (all(mono_flags) or not any(mono_flags)):
            raise CanonicalStackValidationError("mixed HW (mono) and HWC (color) frames")
        hw_set = {(s[0], s[1]) for s in shapes}
        if len(hw_set) != 1:
            raise CanonicalStackValidationError("frames have inconsistent spatial shapes (H, W)")
        c_set = {s[2] for s in shapes}
        if len(c_set) != 1:
            raise CanonicalStackValidationError("frames have inconsistent channel counts")

        height, width = next(iter(hw_set))
        channels = next(iter(c_set))
        original_mono = all(mono_flags)

        self._images = list(images)
        self._support = list(geometric_support)
        self.n_frames = n
        self.height = height
        self.width = width
        self.channels = channels
        self.original_mono = original_mono
        self.original_ndim = 2 if original_mono else 3
        self.original_shape = (height, width) if original_mono else (height, width, channels)

    # -- metadata -----------------------------------------------------------
    @property
    def shape(self):
        return (self.height, self.width)

    # -- phase 1 raw access -------------------------------------------------
    def get_raw_frame(self, i):
        return self._images[i], self._support[i]

    # -- phase 2 prepared tile access ---------------------------------------
    def get_tile(self, i, y0, y1, x0, x1):
        raw = self._images[i]
        sup = self._support[i]
        if self.original_mono:
            tile = raw[y0:y1, x0:x1]
        else:
            tile = raw[y0:y1, x0:x1, :]
        sup_tile = sup[y0:y1, x0:x1]
        batch = prepare_canonical_inputs([tile], [sup_tile])
        return batch.images[0], batch.valid_mask[0]


# ---------------------------------------------------------------------------
# Internal phase-1 result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Phase1:
    reference_index: int
    coefficients: np.ndarray      # (N, C, 2) float64 (a, b); NaN for excluded
    norm_active: np.ndarray       # (N,) bool  (normalization-level active)
    weights: np.ndarray           # (N,) float64 quality weights
    weight_active: np.ndarray     # (N,) bool  (weighting-level active)
    exclusions: tuple             # FrameExclusion (normalization then weighting)
    n_frames: int
    height: int
    width: int
    channels: int
    original_mono: bool
    original_ndim: int
    original_shape: tuple


@dataclass(frozen=True)
class FixedNormalization:
    """ZM-ZEGRID-R11 — a precomputed, cell-independent photometric gauge.

    Carries the per-frame photometric normalization for a set of frames, all
    expressed against ONE reference frame (``reference_index``). This is the
    GLOBAL gauge: the coefficients/weights/active flags are computed ONCE over
    the frames' full valid footprint (see
    :func:`zemosaic.core.zegrid.photometric.compute_global_gauge`) and then
    injected into every Cell, so all Cells share a single photometric anchor and
    the inter-cell level steps disappear.

    Fields
    ------
    reference_index:
        Index (within the frame order this gauge covers) of the reference frame,
        whose coefficient is the identity ``(1, 0)``.
    coefficients:
        Owned ``(N, C, 2)`` float64 ``(a, b)`` per frame/channel; NaN for frames
        excluded at the normalization stage.
    norm_active:
        ``(N,)`` bool — frames surviving the normalization stage.
    weights:
        ``(N,)`` float64 canonical quality weights (normalized, max == 1).
    weight_active:
        ``(N,)`` bool — frames surviving the weighting stage.
    exclusions:
        Tuple of :class:`FrameExclusion` (normalization then weighting), indexed
        by the frame order this gauge covers.
    frame_ids:
        Optional tuple of frame ids in the order this gauge covers. When set
        (``subset_fixed_normalization`` sets it to the Cell's frame order), the
        streaming executor verifies it against the provider's ``frame_ids`` so a
        wrong-but-same-N gauge cannot apply silently (R11 I1).
    """

    reference_index: int
    coefficients: np.ndarray      # (N, C, 2) float64
    norm_active: np.ndarray       # (N,) bool
    weights: np.ndarray           # (N,) float64
    weight_active: np.ndarray     # (N,) bool
    exclusions: tuple             # FrameExclusion
    frame_ids: tuple | None = None


# --------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_method_token(method, supported, what: str) -> str:
    if not isinstance(method, str):
        raise CanonicalStackValidationError(
            f"{what} method must be a string, got {type(method).__name__}"
        )
    token = method.strip().lower()
    if token not in supported:
        raise CanonicalStackValidationError(
            f"unsupported {what} method {method!r}; expected one of "
            f"{'/'.join(supported)} (aliases are not accepted)"
        )
    return token


def _resolve_reject_token(method) -> str:
    if not isinstance(method, str):
        raise CanonicalStackValidationError(
            f"rejection method must be a string, got {type(method).__name__}"
        )
    token = method.strip().lower()
    if token == "linear_fit_clip":
        raise CanonicalStackValidationError(
            "rejection method linear_fit_clip is removed; use none/kappa_sigma/"
            f"winsorized_sigma_clip ({_REJECT_TOKEN_REMOVED})"
        )
    if token not in _REJECT_METHODS:
        raise CanonicalStackValidationError(
            f"unsupported rejection method {method!r}; expected one of "
            "none/kappa_sigma/winsorized_sigma_clip (aliases are not accepted)"
        )
    return token


def _resolve_taper(request: CanonicalStackRequest):
    """Return ``(taper_kind, taper_px, taper_floor, halo)`` (single source).

    Kept in ONE place so the prefetch admission and the per-tile pipeline can
    never disagree about the halo / extended-tile size.
    """
    taper_arg = request.taper
    if not isinstance(taper_arg, str):
        raise CanonicalStackValidationError(
            "explicit per-frame taper maps are not yet supported by the streaming "
            "executor (supported: 'none' / 'footprint')"
        )
    taper_kind = taper_arg.strip().lower()
    if taper_kind not in ("footprint", "none"):
        raise CanonicalStackValidationError(
            f"taper string must be 'footprint' or 'none', got {request.taper!r}"
        )
    taper_px = float(request.taper_px)
    taper_floor = float(request.taper_floor)
    return taper_kind, taper_px, taper_floor, _halo_px(taper_kind, taper_px)


def _halo_px(taper_kind: str, taper_px: float) -> int:
    """Halo (pixels) needed to make the interior footprint taper exact.

    The footprint taper is a per-frame EDT whose reach is ``feather_px``; only
    pixels within Euclidean distance ``feather_px`` of a boundary depend on
    context beyond ``ceil(feather_px)`` (the +1 covers the single-pixel EDT pad
    and boundary rounding). ``"none"`` needs no halo.
    """
    if taper_kind == "none":
        return 0
    return int(np.ceil(float(taper_px))) + 1


def _iter_tiles(height, width, tile_size):
    """Yield ``(y0, y1, x0, x1)`` interior tile bounds, row-major, deterministic."""
    if tile_size is None:
        yield (0, height, 0, width)
        return
    if isinstance(tile_size, (int, np.integer)):
        th = tw = int(tile_size)
    else:
        th, tw = int(tile_size[0]), int(tile_size[1])
    th = max(1, th)
    tw = max(1, tw)
    for y0 in range(0, height, th):
        y1 = min(height, y0 + th)
        for x0 in range(0, width, tw):
            x1 = min(width, x0 + tw)
            yield (y0, y1, x0, x1)


# ---------------------------------------------------------------------------
# ZM-PHASE3-TILE-PREFETCH: bounded intra-cell prefetch of exactly ONE prepared
# spatial tile (GPU ZeGrid path only, opt-in, admission-gated).
#
# The producer thread calls ``provider.get_tile`` for the NEXT spatial tile while
# the consumer thread processes the CURRENT tile (rejection/combine/support).
# Exactly one spatial tile is ever in flight beyond the tile being consumed
# (queue capacity 1), every (tile, frame) pair is read exactly once, and the
# consumer never calls the provider while prefetch is active (single-reader
# contract). Phase 1 is complete before the producer starts.
# ---------------------------------------------------------------------------

#: Honest fixed per-slot overhead (queue/object/array-header/thread bookkeeping).
_TILE_PREFETCH_OVERHEAD_BYTES = 1 << 20
#: Admission margin on the extra working set (never an assumption about saving).
_TILE_PREFETCH_MARGIN_NUM = 115
_TILE_PREFETCH_MARGIN_DEN = 100
#: Finite bound for the terminal join (after the stop latch + drain).
_TILE_PREFETCH_JOIN_TIMEOUT_S = 30.0
_TILE_PREFETCH_SCHEMA = "zegrid-tile-prefetch-v1"

_TILE_PREFETCH_SENTINEL = object()


def _tile_dims(tile_size):
    """Return ``(th, tw)`` or ``None`` for the single-full-patch tile."""
    if tile_size is None:
        return None
    if isinstance(tile_size, (int, np.integer)) and not isinstance(tile_size, bool):
        th = tw = max(1, int(tile_size))
    else:
        try:
            th, tw = int(tile_size[0]), int(tile_size[1])
        except Exception:
            raise CanonicalStackValidationError(
                f"tile_size must be None, an int or a (th, tw) pair, got {tile_size!r}"
            )
        th, tw = max(1, th), max(1, tw)
    return th, tw


def estimate_tile_prefetch_slot_bytes(
    n_frames: int,
    height: int,
    width: int,
    channels: int,
    tile_size=None,
    halo_px: int = 0,
) -> int:
    """Exact host bytes retained by ONE prepared prefetch slot (upper bound).

    A slot holds, for every frame, the two arrays ``provider.get_tile`` returns:
    the prepared float32 HWC image ``(ext_th, ext_tw, C)`` and the valid bool mask
    ``(ext_th, ext_tw)``. The extended tile includes the footprint-taper halo on
    both sides and is clipped to the patch, so the maximum over all tiles is
    ``min(H, th + 2*halo) x min(W, tw + 2*halo)``. A fixed, documented overhead
    covers the queue entry, ndarray headers and thread bookkeeping.

    ``images64`` / taper / coefficient / rejection / combine arrays are NOT
    charged here: they are produced by the consumer from the slot and were
    already part of the historical streaming working set.
    """
    if int(n_frames) < 0 or int(height) < 0 or int(width) < 0 or int(channels) < 1:
        raise CanonicalStackValidationError("invalid dimensions for tile prefetch")
    dims = _tile_dims(tile_size)
    if dims is None:
        ext_h, ext_w = int(height), int(width)
    else:
        th, tw = dims
        halo = max(0, int(halo_px))
        ext_h = min(int(height), th + 2 * halo)
        ext_w = min(int(width), tw + 2 * halo)
    data = int(n_frames) * int(ext_h) * int(ext_w) * (4 * int(channels) + 1)
    return int(data) + int(_TILE_PREFETCH_OVERHEAD_BYTES)


def _normalize_tile_prefetch_request(value) -> str:
    """Normalize the public ``tile_prefetch`` token (no silent aliases)."""
    if value is None or value is False:
        return "off"
    if value is True:
        return "auto"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        v = int(value)
        if v == 0:
            return "off"
        if v == 1:
            return "1"
        raise CanonicalStackValidationError(
            f"tile_prefetch must be off/auto/1, got {value!r}"
        )
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("off", "none", "0", "disabled"):
            return "off"
        if token == "auto":
            return "auto"
        if token == "1":
            return "1"
    raise CanonicalStackValidationError(
        f"tile_prefetch must be off/auto/1 (or None/False), got {value!r}"
    )


def admit_tile_prefetch(
    *,
    requested,
    backend: str,
    n_frames: int,
    height: int,
    width: int,
    channels: int,
    tile_size=None,
    halo_px: int = 0,
    available_bytes: int | None = None,
    budget_bytes: int | None = None,
    base_aggregate_bytes: int | None = None,
    host_budget_bytes: int | None = None,
    margin_num: int = _TILE_PREFETCH_MARGIN_NUM,
    margin_den: int = _TILE_PREFETCH_MARGIN_DEN,
) -> dict:
    """Decide whether ONE prefetch slot may be activated (fail closed).

    Admission is mandatory and explicit: without a fresh ``available_bytes`` or
    an explicit ``budget_bytes`` the request is refused. The request is only
    honoured on the GPU backend, with a bounded tiled phase (a single-tile run has
    no next tile). The returned record always states requested/admitted/
    effective, the exact slot model, the margin and the human-readable reason.
    """
    token = _normalize_tile_prefetch_request(requested)
    slot_bytes = int(
        estimate_tile_prefetch_slot_bytes(
            n_frames, height, width, channels, tile_size, halo_px
        )
    )
    required = int((slot_bytes * int(margin_num) + int(margin_den) - 1) // int(margin_den))
    # F2 (rework R1): the PLANNER/RUN budget headroom. ``base_aggregate_bytes``
    # is the admitted host aggregate of the unit (the ``check`` projection), so
    # ``base + required <= host_budget`` is the authoritative 90 % cap. A raw
    # MemAvailable check alone is NOT sufficient.
    run_headroom = None
    if base_aggregate_bytes is not None and host_budget_bytes is not None:
        run_headroom = max(
            0, int(host_budget_bytes) - int(base_aggregate_bytes)
        )
    if run_headroom is None:
        effective_budget = None if budget_bytes is None else int(budget_bytes)
        budget_kind = None if budget_bytes is None else "headroom"
    else:
        effective_budget = (
            run_headroom if budget_bytes is None
            else min(int(budget_bytes), run_headroom)
        )
        budget_kind = (
            "run_budget" if budget_bytes is None else "run_budget+headroom"
        )
    record = {
        "schema": _TILE_PREFETCH_SCHEMA,
        "requested": token,
        "admitted": False,
        "effective": 0,
        "slot_bytes": int(slot_bytes),
        "required_bytes": int(required),
        "margin_num": int(margin_num),
        "margin_den": int(margin_den),
        "available_bytes": None if available_bytes is None else int(available_bytes),
        "budget_bytes": None if budget_bytes is None else int(budget_bytes),
        "base_aggregate_bytes": (
            None if base_aggregate_bytes is None else int(base_aggregate_bytes)
        ),
        "host_budget_bytes": (
            None if host_budget_bytes is None else int(host_budget_bytes)
        ),
        "budget_headroom_bytes": (
            None if effective_budget is None else int(effective_budget)
        ),
        "budget_kind": budget_kind,
        "projected_aggregate_bytes": (
            None if base_aggregate_bytes is None else int(base_aggregate_bytes)
        ),
        "reason": None,
    }
    if token == "off":
        record["reason"] = "requested_off"
        return record
    if str(backend).strip().lower() != "gpu":
        record["reason"] = "cpu_backend"
        return record
    if tile_size is None:
        record["reason"] = "single_tile_no_next"
        return record
    if available_bytes is None and effective_budget is None:
        record["reason"] = "admission_unavailable"
        return record
    if effective_budget is not None and required > int(effective_budget):
        if run_headroom is not None and required > int(run_headroom):
            record["reason"] = "insufficient_run_budget"
        else:
            record["reason"] = "insufficient_budget"
        return record
    if available_bytes is not None and required > int(available_bytes):
        record["reason"] = "insufficient_available_ram"
        return record
    record["admitted"] = True
    record["effective"] = 1
    record["reason"] = "admitted"
    if base_aggregate_bytes is not None:
        record["projected_aggregate_bytes"] = int(base_aggregate_bytes) + int(required)
    return record


def _intervals_intersection_s(producer_intervals, consumer_intervals) -> float:
    """Observed overlap (seconds) between real produce/consume work intervals."""
    if not producer_intervals or not consumer_intervals:
        return 0.0
    total = 0.0
    for c0, c1 in consumer_intervals:
        start, end = float(c0), float(c1)
        if end <= start:
            continue
        for p0, p1 in producer_intervals:
            lo = max(start, float(p0))
            hi = min(end, float(p1))
            if hi > lo:
                total += hi - lo
    return float(total)


class _TilePrefetcher:
    """Bounded (capacity 1) producer of exactly ONE prepared spatial tile.

    The producer is a single thread that calls ``provider.get_tile`` in the exact
    historical order (tile-major, then frame index) and publishes the per-frame
    ``(prepared_tile, pre_valid)`` pairs. The consumer owns the calling thread and
    never touches the provider. Every terminal path (success, provider error,
    consumer error, cancellation, early exit) latches ``stop``, drains the queue
    and joins the producer with a finite bound; the root exception is preserved
    by the caller because ``shutdown`` never raises.
    """

    def __init__(
        self,
        provider,
        tile_bounds,
        n_frames: int,
        height: int,
        width: int,
        halo: int,
        *,
        timed: bool = False,
    ) -> None:
        self._provider = provider
        self._tiles = list(tile_bounds)
        self._n = int(n_frames)
        self._h = int(height)
        self._w = int(width)
        self._halo = max(0, int(halo))
        self._timed = bool(timed)
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        # ONE in-flight slot beyond the tile being consumed. The producer must
        # ACQUIRE it before it allocates a tile and the consumer releases it when
        # it takes that tile, so the producer can never build a second prepared
        # tile while one is already queued (the byte admission prices exactly one
        # extra tile, never two).
        self._slot = threading.Semaphore(1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        # F4 (rework R2): deferred-close handshake for an uninterruptible read.
        # The caller arms a closer ONLY while the producer is still running; the
        # producer's own final block runs it after it actually exits. Both sides
        # share the lock, and ``_thread_finished`` closes the finish/arm race.
        self._close_lock = threading.Lock()
        self._thread_finished = False
        self._deferred_closer = None
        self._producer_reads = 0
        self._tiles_produced = 0
        self._tiles_consumed = 0
        self._queue_max = 0
        self._producer_work_s = 0.0
        self._producer_intervals: list[tuple[float, float]] = []
        self._consumer_intervals: list[tuple[float, float]] = []
        self._consumer_wait_s = 0.0
        self._consumer_wait_count = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="cogrid-tile-prefetch", daemon=True
        )
        self._thread.start()

    def _ext(self, bounds):
        y0, y1, x0, x1 = bounds
        halo = self._halo
        return (
            max(0, y0 - halo),
            min(self._h, y1 + halo),
            max(0, x0 - halo),
            min(self._w, x1 + halo),
        )

    def _sample_queue(self) -> None:
        self._queue_max = max(self._queue_max, int(self._queue.qsize()))

    def _acquire_slot(self) -> bool:
        while not self._stop.is_set():
            if self._slot.acquire(timeout=0.05):
                return True
        return False

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
            except queue.Full:
                continue
            self._sample_queue()
            return True
        return False

    def _run(self) -> None:
        try:
            for bounds in self._tiles:
                # Capacity contract: only ONE prepared tile may exist ahead of
                # the consumer, so wait for the single slot BEFORE allocating.
                if not self._acquire_slot():
                    return
                if self._stop.is_set():
                    return
                ey0, ey1, ex0, ex1 = self._ext(bounds)
                frames = []
                t0 = time.perf_counter() if self._timed else 0.0
                try:
                    for i in range(self._n):
                        if self._stop.is_set():
                            return
                        raw_tile, pre_valid = self._provider.get_tile(
                            i, ey0, ey1, ex0, ex1
                        )
                        self._producer_reads += 1
                        frames.append((raw_tile, pre_valid))
                finally:
                    if self._timed:
                        t1 = time.perf_counter()
                        self._producer_work_s += max(0.0, t1 - t0)
                        self._producer_intervals.append((t0, t1))
                if self._stop.is_set():
                    return
                if not self._put({"bounds": bounds, "frames": frames}):
                    return
                self._tiles_produced += 1
                # Drop the producer's own references: the queued payload (or the
                # consumer) is the ONLY owner of this prepared set. Keeping the
                # last ``raw_tile``/``frames`` alive in this frame was the extra
                # residency F1 measured.
                frames = None
                try:
                    raw_tile = None
                    pre_valid = None
                except UnboundLocalError:  # pragma: no cover - n >= 1
                    pass
            self._put(_TILE_PREFETCH_SENTINEL)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer
            self._error = exc
            # Wake a consumer blocked on an empty queue (bounded, stop-aware).
            if not self._stop.is_set():
                self._put(_TILE_PREFETCH_SENTINEL)
        finally:
            # F4: exact-once deferred close. If the caller armed a closer while
            # this thread was still alive, it runs HERE, after the provider call
            # has really returned. If the caller found the thread already
            # finished, it closed the provider itself (arm returns False).
            with self._close_lock:
                _closer = self._deferred_closer
                self._deferred_closer = None
                self._thread_finished = True
            if _closer is not None:
                try:
                    _closer()
                except BaseException:  # noqa: BLE001 - close must not mask root
                    pass

    def next(self):
        """Return the frame list of the next spatial tile (consumer thread)."""
        t0 = time.perf_counter() if self._timed else 0.0
        _wait_counted = False
        while True:
            if self._error is not None:
                raise self._error
            if not _wait_counted and self._queue.empty():
                # Structural counter (no timer): this tile fetch had to wait.
                self._consumer_wait_count += 1
                _wait_counted = True
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                if not _wait_counted:
                    self._consumer_wait_count += 1
                    _wait_counted = True
                if self._thread is not None and not self._thread.is_alive():
                    if self._error is not None:
                        raise self._error
                    raise CanonicalStackFailure(
                        "tile prefetch producer exited without a payload"
                    )
                continue
            if self._timed:
                self._consumer_wait_s += max(0.0, time.perf_counter() - t0)
            self._tiles_consumed += 1
            if item is _TILE_PREFETCH_SENTINEL:
                if self._error is not None:
                    raise self._error
                if self._tiles_consumed <= len(self._tiles):
                    raise CanonicalStackFailure(
                        "tile prefetch ended before all tiles were consumed"
                    )
                raise CanonicalStackFailure(
                    "tile prefetch sentinel consumed past the last tile"
                )
            return item["frames"]

    def arm_deferred_close(self, closer) -> bool:
        """Arm ``closer`` to run when this producer REALLY exits (F4 rework R2).

        Returns ``False`` when the producer has already finished, in which case
        the caller must close immediately. Exact-once and race-free with the
        producer's own final block; the closer never runs while a provider call
        is in flight.
        """
        if not callable(closer):
            raise TypeError("deferred closer must be callable")
        with self._close_lock:
            if self._thread_finished:
                return False
            self._deferred_closer = closer
            return True

    def release_slot(self) -> None:
        """Release the single in-flight slot AFTER the consumer dropped the raw set.

        ZM-PHASE3-TILE-PREFETCH rework R1 (F1): releasing on dequeue let the
        producer build tile N+1 while the consumer still held the complete raw
        set of tile N and was allocating its historical ``images_t``/``valid_t``
        (two extra prepared sets). The caller releases only once every raw
        reference is gone, so the admitted one-slot model is the real peak.
        """
        self._slot.release()

    def note_consumer_interval(self, started: float, ended: float) -> None:
        if self._timed and ended > started:
            self._consumer_intervals.append((float(started), float(ended)))

    def shutdown(self) -> dict:
        """Stop, drain and join the producer (never raises)."""
        self._stop.set()
        # Unblock a producer waiting on a full queue.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        thread = self._thread
        if thread is not None:
            thread.join(_TILE_PREFETCH_JOIN_TIMEOUT_S)
            if thread.is_alive():
                # Second finite bound; a provider that blocks forever cannot be
                # killed in-process, so report it truthfully instead of lying.
                thread.join(_TILE_PREFETCH_JOIN_TIMEOUT_S)
        # Final drain: a payload enqueued between the first drain and the stop
        # observation must not stay referenced after the pipeline returns.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        alive = bool(thread is not None and thread.is_alive())
        overlap_s = (
            _intervals_intersection_s(
                self._producer_intervals, self._consumer_intervals
            )
            if self._timed
            else None
        )
        return {
            "queue_capacity": 1,
            "queue_max": int(self._queue_max),
            "tiles_total": int(len(self._tiles)),
            "tiles_prefetched": int(self._tiles_produced),
            "tiles_consumed": int(self._tiles_consumed),
            "provider_reads": int(self._producer_reads),
            "consumer_wait_count": int(self._consumer_wait_count),
            "consumer_wait_s": (round(self._consumer_wait_s, 6) if self._timed else None),
            "producer_work_s": (round(self._producer_work_s, 6) if self._timed else None),
            "overlap_s": (None if overlap_s is None else round(overlap_s, 6)),
            "producer_alive_after_join": bool(alive),
        }


def _restore_shape(arr_c, original_mono):
    """Restore the canonical (H, W, C) combine output to the original shape."""
    if original_mono:
        return arr_c[..., 0].copy()
    return np.ascontiguousarray(arr_c)


def _validate_rejection_sink(sink, n: int, h: int, w: int, c: int) -> None:
    """Validate a caller-supplied rejection sink (ZM-ZEGRID-R52-OOC Lot A).

    The sink replaces the dense ``(N, H, W, C)`` bool diagnostic plane for a
    caller that does not need the full mask. It must expose the same shape/dtype
    contract and a ``flush_batch()`` boundary hook. Validation is explicit and
    raises; there is no silent fallback to a dense mask (that would hide an
    accidental O(N*H*W*C) allocation).
    """
    shape = getattr(sink, "shape", None)
    try:
        sink_shape = tuple(int(v) for v in shape)
    except Exception:
        raise CanonicalStackValidationError(
            "rejection_sink must expose an integer shape tuple (N, H, W, C)"
        )
    if sink_shape != (int(n), int(h), int(w), int(c)):
        raise CanonicalStackValidationError(
            f"rejection_sink shape {sink_shape} != expected {(int(n), int(h), int(w), int(c))}"
        )
    try:
        sink_dtype = np.dtype(getattr(sink, "dtype", None))
    except Exception:
        raise CanonicalStackValidationError(
            "rejection_sink must expose a numpy dtype (bool)"
        )
    if sink_dtype != np.dtype(bool):
        raise CanonicalStackValidationError(
            f"rejection_sink dtype {sink_dtype} != bool"
        )
    if not callable(getattr(sink, "flush_batch", None)):
        raise CanonicalStackValidationError(
            "rejection_sink must provide a callable flush_batch()"
        )


# ---------------------------------------------------------------------------
# Phase 1 — per-frame statistics via reused stage functions (<= 2 frames resident)
# ---------------------------------------------------------------------------

def _phase1(provider: CanonicalFrameProvider, request: CanonicalStackRequest) -> _Phase1:
    n = provider.n_frames
    h = provider.height
    w = provider.width
    c = provider.channels

    norm_token = _resolve_method_token(request.normalization, _NORM_METHODS, "normalization")
    weight_token = _resolve_method_token(request.weighting, _WEIGHT_METHODS, "weighting")

    # --- 1a: per-frame valid counts (reuse prepare_canonical_inputs on 1 frame) ---
    counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        raw, sup = provider.get_raw_frame(i)
        b = prepare_canonical_inputs([raw], [sup])
        counts[i] = int(b.frame_valid_counts[0])

    # --- 1b: reference selection (reuse select_canonical_reference) ---
    probe = CanonicalInputBatch(
        images=np.empty((0,), dtype=np.float32),
        valid_mask=np.empty((0,), dtype=bool),
        original_mono=provider.original_mono,
        frame_valid_counts=counts,
        n_frames=n,
        height=h,
        width=w,
        channels=c,
        original_ndim=provider.original_ndim,
        original_shape=provider.original_shape,
    )
    ref_idx = select_canonical_reference(probe, request.reference_index)

    # --- 1c/1d: normalization + weighting, interleaved per frame (2 frames resident) ---
    coefficients = np.full((n, c, 2), np.nan, dtype=np.float64)
    norm_active = np.zeros(n, dtype=bool)
    weights = np.zeros(n, dtype=np.float64)
    raw_weights = np.full(n, np.nan, dtype=np.float64)
    noise_sigma = np.full(n, np.nan, dtype=np.float64)
    fwhm = np.full(n, np.nan, dtype=np.float64)
    norm_exclusions = []
    weight_exclusions = []

    ref_raw, ref_sup = provider.get_raw_frame(ref_idx)
    coefficients[ref_idx, :, :] = (1.0, 0.0)
    norm_active[ref_idx] = counts[ref_idx] > 0

    # Reference frame weighting (1-frame batch; reference at index 0). A reference
    # quality failure raises CanonicalStackFailure (matches the full pipeline).
    if weight_token != "none":
        ref_norm = normalize_canonical_images(
            prepare_canonical_inputs([ref_raw], [ref_sup]), norm_token, reference_index=0
        )
        ref_weight = compute_canonical_quality_weights(ref_norm, weight_token)
        raw_weights[ref_idx] = ref_weight.raw_weights[0]
        noise_sigma[ref_idx] = ref_weight.noise_sigma[0]
        fwhm[ref_idx] = ref_weight.fwhm[0]

    for i in range(n):
        if i == ref_idx:
            continue
        raw, sup = provider.get_raw_frame(i)
        batch2 = prepare_canonical_inputs([ref_raw, raw], [ref_sup, sup])
        norm2 = normalize_canonical_images(batch2, norm_token, reference_index=0)

        coefficients[i] = norm2.coefficients[1]
        norm_active[i] = bool(norm2.active_frames[1])
        for e in norm2.exclusions:
            if e.index == 1:
                norm_exclusions.append(
                    FrameExclusion(index=i, stage=e.stage, reason=e.reason, detail=e.detail)
                )

        if weight_token != "none" and norm_active[i]:
            weight2 = compute_canonical_quality_weights(norm2, weight_token)
            raw_weights[i] = weight2.raw_weights[1]
            noise_sigma[i] = weight2.noise_sigma[1]
            fwhm[i] = weight2.fwhm[1]
            if not bool(weight2.active_frames[1]):
                for e in weight2.exclusions:
                    if e.index == 1 and e.stage == "weighting":
                        weight_exclusions.append(
                            FrameExclusion(index=i, stage=e.stage, reason=e.reason, detail=e.detail)
                        )

    if weight_token == "none":
        weights[norm_active] = 1.0
        raw_weights[norm_active] = 1.0
        weight_active = norm_active.copy()
    else:
        # Weighting-level active = norm-active survivors of the quality metric.
        weight_active = norm_active.copy()
        for e in weight_exclusions:
            weight_active[e.index] = False
        if not weight_active.any():
            raise CanonicalStackFailure("no frame remains after quality weighting")
        max_raw = float(np.max(raw_weights[weight_active]))
        if not (np.isfinite(max_raw) and max_raw > 0.0):
            raise CanonicalStackFailure("no positive quality weight remains after weighting")
        weights[weight_active] = raw_weights[weight_active] / max_raw

    exclusions = tuple(norm_exclusions) + tuple(weight_exclusions)
    return _Phase1(
        reference_index=ref_idx,
        coefficients=coefficients,
        norm_active=norm_active,
        weights=weights,
        weight_active=weight_active,
        exclusions=exclusions,
        n_frames=n,
        height=h,
        width=w,
        channels=c,
        original_mono=provider.original_mono,
        original_ndim=provider.original_ndim,
        original_shape=provider.original_shape,
    )


def compute_fixed_normalization(
    provider: CanonicalFrameProvider, request: CanonicalStackRequest
) -> "FixedNormalization":
    """Compute the per-frame photometric gauge of ``provider`` once.

    This is the GLOBAL gauge for the set of frames the provider serves: it runs
    the EXACT same phase-1 pipeline (reference selection, per-frame normalization
    coefficients, quality weights, active flags, exclusions) that
    :func:`run_canonical_stack_streaming` would run per-Cell, but computed over
    the provider's full domain (e.g. the full-canvas alignment), so every Cell can
    then reuse the SAME coefficients instead of recomputing them over its own
    patch. The coefficients are computed with the frozen canonical functions
    verbatim — no re-implementation.
    """
    p1 = _phase1(provider, request)
    return FixedNormalization(
        reference_index=int(p1.reference_index),
        coefficients=np.array(p1.coefficients, dtype=np.float64, copy=True),
        norm_active=np.array(p1.norm_active, dtype=bool, copy=True),
        weights=np.array(p1.weights, dtype=np.float64, copy=True),
        weight_active=np.array(p1.weight_active, dtype=bool, copy=True),
        exclusions=tuple(p1.exclusions),
    )


def subset_fixed_normalization(
    fixed: "FixedNormalization",
    global_frame_ids,
    cell_frame_ids,
) -> "FixedNormalization":
    """Subset a global gauge to a Cell's frame list (in the Cell's order).

    ``global_frame_ids`` is the frame order ``fixed`` covers (length ``N``);
    ``cell_frame_ids`` is the Cell's frame order (a subset, possibly reordered).
    Returns a :class:`FixedNormalization` over ``cell_frame_ids`` with
    ``reference_index`` remapped to the Cell-local index of the global reference.

    LEVEL-CORRECTNESS (ZM-ZEGRID-R11 rework-1, F1): the returned coefficients are
    ALWAYS the GLOBAL coefficients, i.e. expressed relative to the global
    reference ``R``. For ``sky_mean`` (``a=1``) the offset is
    ``b_i = mean(R) - mean(frame_i)``, so ``frame_i + b_i`` lands on R's sky
    level. When ``R`` is NOT a contributor to the Cell, the Cell is NOT re-anchored
    to a local frame ``L`` — re-anchoring (``b'_i = b_i - b_L``) MOVES the level to
    L's sky level (a residual step of exactly ``b_L``), which broke the global
    anchor on real mosaics (most cells lack R). Instead the coefficients stay on
    R's level and ``reference_index`` becomes a bookkeeping placeholder (the Cell's
    highest-weight active frame); phase 2 applies the coefficients uniformly (it
    never special-cases the reference), so the Cell lands on R's GLOBAL level
    exactly like a Cell that contains R. The photometric SCALE and LEVEL are both
    unchanged by this operation.
    """
    if len(global_frame_ids) != fixed.coefficients.shape[0]:
        raise CanonicalStackValidationError(
            f"global_frame_ids length {len(global_frame_ids)} != gauge N "
            f"{fixed.coefficients.shape[0]}"
        )
    gidx = {fid: i for i, fid in enumerate(global_frame_ids)}
    try:
        cell_global_idx = [gidx[fid] for fid in cell_frame_ids]
    except KeyError as exc:
        raise CanonicalStackValidationError(
            f"cell frame {exc.args[0]!r} not present in the global gauge frame order"
        ) from exc

    idx = np.asarray(cell_global_idx, dtype=np.int64)
    n_cell = len(idx)
    c = fixed.coefficients.shape[1]

    coefficients = np.array(fixed.coefficients[idx], dtype=np.float64, copy=True)
    norm_active = np.array(fixed.norm_active[idx], dtype=bool, copy=True)
    weights = np.array(fixed.weights[idx], dtype=np.float64, copy=True)
    weight_active = np.array(fixed.weight_active[idx], dtype=bool, copy=True)

    present = set(idx.tolist())
    ref = int(fixed.reference_index)
    if ref in present:
        cell_ref = int(np.where(idx == ref)[0][0])
        # The global reference frame keeps its identity coefficient (a=1, b=0) by
        # construction; verify it (defensive invariant).
        a_ref = float(coefficients[cell_ref, 0, 0])
        if np.isfinite(a_ref):
            b_ref = float(coefficients[cell_ref, 0, 1])
            if not (np.isclose(a_ref, 1.0, atol=0.0) and np.isclose(b_ref, 0.0, atol=0.0)):
                raise CanonicalStackValidationError(
                    "global reference frame does not carry the identity coefficient (1, 0)"
                )
    else:
        # R is NOT a contributor to this Cell (the real-mosaic case: on M106 5x4 the
        # max-canvas-support reference covers ~26.4% of the canvas, so ~8/20 cells
        # lack R). F1 fix: DO NOT re-anchor. The coefficients are already R-relative
        # (global level); re-anchoring to a local frame L would shift the level by
        # -b_L. The reference_index is a deterministic bookkeeping placeholder (the
        # highest-weight active frame); phase 2 does not special-case the reference,
        # so the science still lands on R's global level.
        active_cell = np.where(weight_active)[0]
        if active_cell.size == 0:
            raise CanonicalStackValidationError(
                "cell gauge has no active frame (no bookkeeping reference available)"
            )
        # Highest weight; ties resolved to the lowest index (deterministic).
        local_weights = np.where(weight_active, weights, -np.inf)
        cell_ref = int(np.argmax(local_weights))
        # coefficients stay R-relative; no re-anchoring.

    # Remap exclusions to the Cell-local frame order (drop frames not in the Cell).
    exclusions = tuple(
        FrameExclusion(
            index=int(np.where(idx == int(e.index))[0][0]),
            stage=e.stage,
            reason=e.reason,
            detail=e.detail,
        )
        for e in fixed.exclusions
        if int(e.index) in present
    )

    return FixedNormalization(
        reference_index=cell_ref,
        coefficients=np.ascontiguousarray(coefficients),
        norm_active=norm_active,
        weights=weights,
        weight_active=weight_active,
        exclusions=exclusions,
        frame_ids=tuple(cell_frame_ids),
    )


def _fixed_phase1(
    provider: CanonicalFrameProvider,
    request: CanonicalStackRequest,
    fixed: "FixedNormalization | None",
) -> _Phase1:
    """Return the phase-1 result: the fixed gauge when provided, else computed."""
    if fixed is None:
        return _phase1(provider, request)
    if not isinstance(fixed, FixedNormalization):
        raise CanonicalStackValidationError(
            f"fixed must be a FixedNormalization, got {type(fixed).__name__}"
        )
    n = provider.n_frames
    if fixed.coefficients.shape[0] != n:
        raise CanonicalStackValidationError(
            f"fixed gauge N {fixed.coefficients.shape[0]} != provider.n_frames {n}"
        )
    if fixed.norm_active.shape != (n,):
        raise CanonicalStackValidationError(
            f"fixed norm_active shape {fixed.norm_active.shape} != ({n},)"
        )
    if fixed.weights.shape != (n,) or fixed.weight_active.shape != (n,):
        raise CanonicalStackValidationError(
            "fixed weights/weight_active shapes do not match provider N"
        )
    if not (0 <= int(fixed.reference_index) < n):
        raise CanonicalStackValidationError(
            f"fixed reference_index {fixed.reference_index} out of range [0, {n})"
        )
    # R11 I1: optional frame-id consistency check. A wrong-but-same-N gauge would
    # otherwise apply silently (coefficients are keyed by POSITION, so a permuted
    # or misaligned frame list would silently mis-map coefficients). When the fixed
    # gauge carries frame ids AND the provider exposes its own, they must match.
    if fixed.frame_ids is not None and hasattr(provider, "frame_ids"):
        prov_ids = tuple(provider.frame_ids)
        if tuple(fixed.frame_ids) != prov_ids:
            raise CanonicalStackValidationError(
                "fixed gauge frame_ids do not match the provider frame order "
                f"({len(fixed.frame_ids)} vs {len(prov_ids)} frames)"
            )
    return _Phase1(
        reference_index=int(fixed.reference_index),
        coefficients=np.array(fixed.coefficients, dtype=np.float64, copy=True),
        norm_active=np.array(fixed.norm_active, dtype=bool, copy=True),
        weights=np.array(fixed.weights, dtype=np.float64, copy=True),
        weight_active=np.array(fixed.weight_active, dtype=bool, copy=True),
        exclusions=tuple(fixed.exclusions),
        n_frames=n,
        height=provider.height,
        width=provider.width,
        channels=provider.channels,
        original_mono=provider.original_mono,
        original_ndim=provider.original_ndim,
        original_shape=provider.original_shape,
    )


# ---------------------------------------------------------------------------
# Phase 2 — per-tile rejection / combine / support
# ---------------------------------------------------------------------------

def _phase2(provider, p1: _Phase1, request: CanonicalStackRequest, tile_size, backend="cpu", rejection_sink=None, phase_profiler=None, tile_prefetch_plan=None) -> CanonicalStackResult:
    n = p1.n_frames
    h = p1.height
    w = p1.width
    c = p1.channels

    # ZM-ZEGRID-R22: resolve the backend module up front. The rejection + combine
    # stages run xp-generic on ``xp`` (NumPy or CuPy); the reductions use the
    # bit-exact host-side helpers so CPU vs GPU outputs are bit-identical.
    xp = _get_xp(backend)
    # ZM-STACK-INSTRUMENTATION: optional, default-OFF bounded profiler. When it
    # is None this adds one branch per phase point and NO timer/synchronization.
    _pf = phase_profiler
    _use_dev_events = bool(_pf is not None and xp is not np)
    _use_transfers = bool(_pf is not None and xp is not np)

    def _prof_add(phase, started):
        if _pf is not None:
            _pf.add(phase, time.perf_counter() - started)


    # Method tokens + validated rejection parameters (reused validators).
    reject_token = _resolve_reject_token(request.rejection)
    combine_token = _resolve_method_token(request.combine, _COMBINE_METHODS, "combine")
    s_low = _validate_reject_sigma(request.sigma_low, "sigma_low")
    s_high = _validate_reject_sigma(request.sigma_high, "sigma_high")
    iters = _validate_max_iters(request.max_iters)
    w_low = _validate_winsor_limit(request.winsor_limit_low, "winsor_limit_low")
    w_high = _validate_winsor_limit(request.winsor_limit_high, "winsor_limit_high")
    if w_low + w_high >= 1.0:
        raise CanonicalStackValidationError(
            f"winsor_limit_low + winsor_limit_high must be < 1, got {w_low} + {w_high}"
        )

    # Taper kind (none / footprint; explicit raises). ONE resolver so the
    # prefetch admission and this loop share the same halo.
    taper_kind, taper_px, taper_floor, halo = _resolve_taper(request)

    active = p1.weight_active
    q = p1.weights
    coeff = p1.coefficients
    ref_idx = p1.reference_index

    # Output planes (O(patch_area), inherent to the result — not O(N x patch_area)).
    science_c = np.empty((h, w, c), dtype=np.float32)
    weight_sum_c = np.zeros((h, w, c), dtype=np.float64)
    valid_mask_c = np.zeros((h, w, c), dtype=bool)
    surviving_c = np.zeros((h, w, c), dtype=np.int64)
    support_w1 = np.zeros((h, w), dtype=np.float64)
    support_w2 = np.zeros((h, w), dtype=np.float64)

    # Global rejection diagnostics (aggregated over disjoint interior cells).
    iterations_used = 0
    initial_count = 0
    surviving_count = 0
    rejected_count = 0
    low_n_count = 0
    degenerate_count = 0
    # ZM-ZEGRID-R52-OOC Lot A: the dense (N, H, W, C) bool diagnostic plane stays
    # the DEFAULT, byte-for-byte. A caller that explicitly supplies a sink gets
    # the same interior-block writes and flush boundaries without that residency;
    # the returned result then carries rejection_mask=None (a count-only sink
    # cannot honestly reproduce the full mask, and a fake ndarray is forbidden).
    if rejection_sink is None:
        rejection_mask = np.zeros((n, h, w, c), dtype=bool)
        rejection_mask_out = rejection_mask
    else:
        _validate_rejection_sink(rejection_sink, n, h, w, c)
        rejection_mask = rejection_sink
        rejection_mask_out = None

    _prefetcher = None
    if tile_prefetch_plan is not None and int(tile_prefetch_plan.get("effective", 0)) == 1:
        # Capacity-1 pipeline: ONE prepared spatial tile ahead of the consumer.
        _prefetcher = _TilePrefetcher(
            provider,
            list(_iter_tiles(h, w, tile_size)),
            n, h, w, halo,
            timed=bool(_pf is not None),
        )
    _prefetch_report = None
    try:
        if _prefetcher is not None:
            _prefetcher.start()
        for (y0, y1, x0, x1) in _iter_tiles(h, w, tile_size):
            # Extended tile bounds (with halo, clipped to the patch).
            ey0 = max(0, y0 - halo)
            ey1 = min(h, y1 + halo)
            ex0 = max(0, x0 - halo)
            ex1 = min(w, x1 + halo)
            th = ey1 - ey0
            tw = ex1 - ex0

            # Gather N frames: prepared raw tile + valid; then re-apply coefficients.
            # Profiling boundary: only provider.get_tile() calls are tile READ time;
            # allocation, coefficient re-application and the float64 conversion are
            # tile PREP. With intra-cell prefetch ON, tile_read_s is the consumer's
            # WAIT for the prefetched tile: it happens BEFORE this tile's compute
            # and is therefore DISJOINT (not overlapping). The overlapping read is
            # reported separately as producer_work_s. tile_read_s is NOT
            # comparable between OFF (provider duration) and ON (consumer wait).
            _prefetched = None
            if _prefetcher is not None:
                _t_wait = time.perf_counter() if _pf is not None else 0.0
                _prefetched = _prefetcher.next()
                if _pf is not None:
                    _pf.add("tile_read_s", time.perf_counter() - _t_wait)
            # Real consumer work interval starts AFTER the wait.
            _tile_start = (
                time.perf_counter()
                if (_pf is not None and _prefetcher is not None)
                else 0.0
            )
            _t_prep_start = time.perf_counter() if _pf is not None else 0.0
            _read_s = 0.0
            images_t = np.empty((n, th, tw, c), dtype=np.float32)
            valid_t = np.zeros((n, th, tw), dtype=bool)
            for i in range(n):
                if _prefetched is not None:
                    raw_tile, pre_valid = _prefetched[i]
                else:
                    if _pf is not None:
                        _t_read = time.perf_counter()
                    raw_tile, pre_valid = provider.get_tile(i, ey0, ey1, ex0, ex1)
                    if _pf is not None:
                        _read_s += time.perf_counter() - _t_read
                if not p1.norm_active[i]:
                    # normalization-excluded frame: all-NaN image, all-false valid.
                    images_t[i] = np.nan
                    valid_t[i] = False
                    continue
                a = coeff[i]  # (C, 2)
                src64 = raw_tile.astype(np.float64)
                transformed = np.empty((th, tw, c), dtype=np.float64)
                for ch in range(c):
                    transformed[:, :, ch] = a[ch, 0] * src64[:, :, ch] + a[ch, 1]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    transformed32 = transformed.astype(np.float32)
                finite_all = np.all(np.isfinite(transformed32), axis=-1)
                post_valid = pre_valid & finite_all
                frame = images_t[i]
                frame[:] = np.nan
                frame[post_valid] = transformed32[post_valid]
                valid_t[i] = post_valid

            if _prefetcher is not None:
                # F1: the raw prefetched set must be COMPLETELY dropped before
                # the producer may build the next tile. ``images_t``/``valid_t``
                # are the consumer's own historical buffers; nothing below reads
                # the raw set again. Deleting the container and the last loop
                # bindings releases it deterministically (CPython refcounts).
                _prefetched.clear()
                del _prefetched
                _prefetched = None
                try:
                    del raw_tile
                    del pre_valid
                except UnboundLocalError:  # pragma: no cover - n >= 1
                    pass
                _prefetcher.release_slot()
            images64 = images_t.astype(np.float64)
            # Historical provider-read bucket ONLY when prefetch is absent. In
            # ON mode the single tile_read_s add already happened after next();
            # branching on ``_prefetched`` here double-counted the bucket because
            # it is deliberately set to None after the raw set is dropped.
            if _pf is not None and _prefetcher is None:
                _pf.add("tile_read_s", _read_s)
                _pf.add(
                    "tile_prepare_s",
                    max(0.0, (time.perf_counter() - _t_prep_start) - _read_s),
                )
            _t_taper = time.perf_counter() if _pf is not None else 0.0

            # --- taper (footprint EDT with halo) -> estimator-weight map ---
            if taper_kind == "footprint":
                a_map = np.empty((n, th, tw), dtype=np.float64)
                for i in range(n):
                    if active[i]:
                        a_map[i] = make_footprint_taper(valid_t[i], feather_px=taper_px, floor=taper_floor).astype(np.float64)
                    else:
                        a_map[i] = 0.0
            else:
                a_map = None

            # wmap = q * m * a (zero outside valid, zero for inactive).
            wmap = np.zeros((n, th, tw), dtype=np.float64)
            for i in range(n):
                if active[i]:
                    m = valid_t[i].astype(np.float64)
                    if a_map is None:
                        wmap[i] = q[i] * m
                    else:
                        wmap[i] = q[i] * m * a_map[i]
                    wmap[i] = np.where(valid_t[i], wmap[i], 0.0)
            _prof_add("taper_weights_s", _t_taper)
            _t_combine = time.perf_counter() if _pf is not None else 0.0
            _dev_events = None
            if _use_dev_events:
                try:
                    # CuPy API: Event(block=False, disable_timing=False, ...).
                    # There is NO enable_timing keyword (that is torch's API).
                    _dev_events = (
                        xp.cuda.Event(disable_timing=False),
                        xp.cuda.Event(disable_timing=False),
                    )
                    _dev_events[0].record()
                except Exception as _ev_exc:  # noqa: BLE001 - telemetry only
                    _dev_events = None
                    _pf.note_device_unavailable(
                        f"cuda_event_unavailable: {type(_ev_exc).__name__}"
                    )

            # --- rejection + combine in a BOUNDED scope --------------------------
            # ZM-ZEGRID-R55-E11: every device temporary is created and dropped inside
            # this helper. When it returns, nothing but host arrays is reachable, so
            # the caller can synchronise and purge the pool WITHOUT holding live
            # blocks. Purging while these arrays were still bound was the retention
            # defect behind the observed inter-cell VRAM drop.
            def _reject_and_combine():
                # --- rejection (reused step primitives on (N, tile_cells)) ---
                n_cells_t = th * tw * c
                initial = active[:, None, None, None] & valid_t[..., None] & np.isfinite(images64)
                orig2_host = images64.reshape(n, n_cells_t)
                survivor2_host = initial.reshape(n, n_cells_t)
                degenerate_seen_host = np.zeros(n_cells_t, dtype=bool)
                tile_iters = 0
                if reject_token == "none":
                    survivor = initial.copy()
                else:
                    # xp-generic rejection; the step primitives use the bit-exact
                    # host-side mean/std reductions (see canonical_stacking), so the
                    # survivor mask is bit-identical CPU vs GPU.
                    _t_h2d = time.perf_counter() if _use_transfers else 0.0
                    orig2 = xp.asarray(orig2_host)
                    survivor2 = xp.asarray(survivor2_host)
                    if _use_transfers:
                        _pf.add("h2d_transfer_s", time.perf_counter() - _t_h2d)
                    degenerate_seen = xp.zeros(n_cells_t, dtype=bool)
                    for _ in range(iters):
                        count = survivor2.sum(axis=0)
                        if reject_token == "kappa_sigma":
                            new_survivor2, degenerate = _kappa_sigma_step(
                                orig2, survivor2, count, s_low, s_high, xp,
                                profile=_pf,
                            )
                        else:
                            new_survivor2, degenerate = _winsorized_sigma_clip_step(
                                orig2, survivor2, count, s_low, s_high, w_low, w_high,
                                xp, profile=_pf,
                            )
                        tile_iters += 1
                        degenerate_seen |= degenerate
                        if bool(xp.array_equal(new_survivor2, survivor2)):
                            survivor2 = new_survivor2
                            break
                        survivor2 = new_survivor2
                    _t_rb = time.perf_counter() if _use_transfers else 0.0
                    survivor = _as_host_numpy(survivor2, xp).reshape(n, th, tw, c)
                    degenerate_seen_host = _as_host_numpy(degenerate_seen, xp)
                    if _use_transfers:
                        _pf.add("d2h_transfer_s", time.perf_counter() - _t_rb)

                rejection_t = initial & ~survivor

                # --- combine (reused reductions on the N axis, bit-exact across backends) ---
                w_positive = wmap[:, :, :, None] > 0.0
                eligible = survivor & w_positive & np.isfinite(images64)
                _t_h2d = time.perf_counter() if _use_transfers else 0.0
                images64_xp = xp.asarray(images64)
                wmap_xp = xp.asarray(wmap[:, :, :, None])
                eligible_xp = xp.asarray(eligible)
                if _use_transfers:
                    _pf.add("h2d_transfer_s", time.perf_counter() - _t_h2d)
                if combine_token == "mean":
                    masked_images = xp.where(eligible_xp, images64_xp, 0.0)
                    w_contrib = xp.where(eligible_xp, wmap_xp, 0.0)
                    # bit-exact float64 reductions: _axis_sum_f64 reduces on the host for
                    # the GPU backend (matching np.sum), so science is bit-identical.
                    # Its D2H transfer and host reduction are attributed internally.
                    numerator = _axis_sum_f64(
                        masked_images * w_contrib, xp, profile=_pf
                    )
                    denominator = _axis_sum_f64(w_contrib, xp, profile=_pf)
                    den_valid = denominator > 0.0
                    estimate64 = _safe_divide(numerator, denominator, np)
                    science32 = _astype_float32(estimate64, np)
                    final_finite = np.isfinite(estimate64) & np.isfinite(science32)
                    valid_out = den_valid & final_finite
                    science_t = np.where(valid_out, science32, np.nan)
                    weight_sum_t = np.where(valid_out, denominator, 0.0)
                else:  # median
                    masked = xp.where(eligible_xp, images64_xp, xp.nan)
                    median64_xp = _nan_axis_median(masked, xp)
                    _t_rb = time.perf_counter() if _use_transfers else 0.0
                    median64 = _as_host_numpy(median64_xp, xp)
                    if _use_transfers:
                        _pf.add("d2h_transfer_s", time.perf_counter() - _t_rb)
                    science32 = _astype_float32(median64, np)
                    _t_red = time.perf_counter() if _pf is not None else 0.0
                    count = eligible.sum(axis=0).astype(np.float64)
                    _prof_add("host_reduction_s", _t_red)
                    den_valid = count > 0.0
                    final_finite = np.isfinite(median64) & np.isfinite(science32)
                    valid_out = den_valid & final_finite
                    science_t = np.where(valid_out, science32, np.nan)
                    weight_sum_t = np.where(valid_out, count, 0.0)

                _t_red = time.perf_counter() if _pf is not None else 0.0
                surviving_t = eligible.sum(axis=0).astype(np.int64)
                _prof_add("host_reduction_s", _t_red)
                valid_mask_t = weight_sum_t > 0.0

                return (
                    science_t, weight_sum_t, surviving_t, valid_mask_t,
                    rejection_t, initial, degenerate_seen_host, tile_iters,
                )

            (
                science_t, weight_sum_t, surviving_t, valid_mask_t,
                rejection_t, initial, degenerate_seen_host, tile_iters,
            ) = _reject_and_combine()
            _prof_add("reject_combine_host_s", _t_combine)
            if _dev_events is not None:
                try:
                    _dev_events[1].record()
                    _dev_events[1].synchronize()
                except Exception as _ev_exc:  # noqa: BLE001 - telemetry only
                    _pf.note_device_unavailable(
                        f"cuda_event_sync_failed: {type(_ev_exc).__name__}"
                    )
                else:
                    # The synchronization HAPPENED (and perturbs the run) even if
                    # the elapsed query itself fails; count it truthfully.
                    _pf.note_extra_sync(1)
                    try:
                        # CuPy API: module-level get_elapsed_time returns ms.
                        _pf.add_device_span(
                            xp.cuda.get_elapsed_time(
                                _dev_events[0], _dev_events[1]
                            )
                            / 1000.0
                        )
                    except Exception as _ev_exc:  # noqa: BLE001 - telemetry only
                        _pf.note_device_unavailable(
                            "cuda_event_elapsed_failed: "
                            f"{type(_ev_exc).__name__}"
                        )
            iterations_used = max(iterations_used, tile_iters)
            _t_support = time.perf_counter() if _pf is not None else 0.0

            # --- support accumulation (per frame in order, interior written below) ---
            w1_tile = np.zeros((th, tw), dtype=np.float64)
            w2_tile = np.zeros((th, tw), dtype=np.float64)
            for i in range(n):
                if active[i]:
                    w1_tile += wmap[i]
                    w2_tile += wmap[i] * wmap[i]
            _prof_add("support_accum_s", _t_support)
            _t_writes = time.perf_counter() if _pf is not None else 0.0

            # --- interior extraction + writes ---
            iy0 = y0 - ey0
            iy1 = y1 - ey0
            ix0 = x0 - ex0
            ix1 = x1 - ex0
            sci_int = science_t[iy0:iy1, ix0:ix1]
            science_c[y0:y1, x0:x1] = sci_int
            weight_sum_c[y0:y1, x0:x1] = weight_sum_t[iy0:iy1, ix0:ix1]
            valid_mask_c[y0:y1, x0:x1] = valid_mask_t[iy0:iy1, ix0:ix1]
            surviving_c[y0:y1, x0:x1] = surviving_t[iy0:iy1, ix0:ix1]
            support_w1[y0:y1, x0:x1] = w1_tile[iy0:iy1, ix0:ix1]
            support_w2[y0:y1, x0:x1] = w2_tile[iy0:iy1, ix0:ix1]

            rejection_mask[:, y0:y1, x0:x1, :] = rejection_t[:, iy0:iy1, ix0:ix1, :]
            if rejection_sink is not None:
                rejection_sink.flush_batch()

            init_int = initial[:, iy0:iy1, ix0:ix1, :]
            rej_int = rejection_t[:, iy0:iy1, ix0:ix1, :]
            initial_count += int(init_int.sum())
            rejected_count += int(rej_int.sum())
            low_n_count += int((init_int.sum(axis=0) < 3).sum())
            degenerate_count += int(degenerate_seen_host.reshape(th, tw, c)[iy0:iy1, ix0:ix1, :].sum())
            _prof_add("writes_s", _t_writes)
            if _pf is not None and _prefetcher is not None:
                # Observed consumer work interval for real overlap telemetry.
                _prefetcher.note_consumer_interval(
                    _tile_start, time.perf_counter()
                )
            if _pf is not None:
                _pf.count_tile(n)

            # ZM-ZEGRID-R22 (H3 rework-1) / R55-E11: release the per-tile device
            # workspace between tiles. The temporaries are unreachable now (bounded
            # helper above), so the boundary helper can synchronise and return only
            # genuinely unused blocks to the allocator. Never a global gc, never a
            # live-array free.
            if xp is not np:
                _t_sync = time.perf_counter() if _pf is not None else 0.0
                _release_gpu_workspace(xp, reason="tile_boundary")
                if _pf is not None:
                    _pf.add("boundary_sync_s", time.perf_counter() - _t_sync)

    finally:
        # Every terminal path (success, provider/consumer error, cancellation,
        # early exit) stops, drains and joins the producer BEFORE propagating.
        if _prefetcher is not None:
            _prefetch_report = dict(tile_prefetch_plan or {})
            _prefetch_report.update(_prefetcher.shutdown())
            _prefetch_report["effective"] = 1
            _prefetch_report["admitted"] = True
            if _pf is not None:
                _pf.note_tile_prefetch(_prefetch_report)
            if _prefetch_report.get("producer_alive_after_join"):
                # F4 (rework R1): a surviving producer means the provider call
                # is not interruptible in-process. NEVER present that as a normal
                # success and never let a caller assume quiescence. On an error
                # path the ROOT exception wins (annotated, never replaced).
                _alive_msg = (
                    "tile prefetch producer survived the bounded join "
                    f"({_TILE_PREFETCH_JOIN_TIMEOUT_S:.1f}s x2); the provider "
                    "call is not interruptible in-process and may still be "
                    "executing. Provider is NOT quiescent."
                )
                _active = sys.exc_info()[1]
                if _active is not None:
                    try:
                        _active.add_note(_alive_msg)
                        _active.prefetch_report = dict(_prefetch_report)
                        # F4 (rework R2): the live producer handle lets the
                        # product DEFER provider close until the thread exits.
                        _active.prefetch_handle = _prefetcher
                    except Exception:
                        pass
                else:
                    _alive_exc = CanonicalStackFailure(_alive_msg)
                    try:
                        _alive_exc.prefetch_report = dict(_prefetch_report)
                        _alive_exc.prefetch_handle = _prefetcher
                    except Exception:
                        pass
                    raise _alive_exc
    # ZM-ZEGRID-R55-E11: CELL boundary. Every device temporary from the tile
    # loop is out of scope here; purge the pool before the orchestrator reads
    # fresh VRAM for the NEXT cell. Best-effort, never raises.
    if xp is not np:
        _release_gpu_workspace(xp, reason="cell_boundary")

    return _finalize_phase2_outputs(
        p1,
        science_c=science_c,
        weight_sum_c=weight_sum_c,
        valid_mask_c=valid_mask_c,
        surviving_c=surviving_c,
        support_w1=support_w1,
        support_w2=support_w2,
        iterations_used=iterations_used,
        initial_count=initial_count,
        rejected_count=rejected_count,
        low_n_count=low_n_count,
        degenerate_count=degenerate_count,
        rejection_mask_out=rejection_mask_out,
        prefetch_report=_prefetch_report,
    )


def _compute_n_eff_bounded(support_w1, support_w2):
    """Exact ``n_eff = W1**2 / W2`` with a genuinely bounded live set.

    Reference semantics (preserved bit-for-bit)::

        w1_sq = w1 * w1
        n_eff = w1_sq / w2
        ratio = w1 / sqrt(w2)
        safe  = ratio * ratio
        n_eff = where((~isfinite(w1_sq)) & (w2 > 0), safe, n_eff)
        n_eff = where((w2 > 0) & isfinite(n_eff) & (n_eff >= 0), n_eff, 0)

    The historical expression simultaneously held ``w1_sq``, ``n_eff``,
    ``sqrt(w2)``, ``ratio``, ``safe`` plus boolean/``where`` outputs (~62 B per
    patch pixel). This version reuses the ``n_eff`` buffer for ``w1_sq`` and
    computes the overflow-safe branch only when it is actually needed, using at
    most two additional patch-sized float64 arrays. The arithmetic operations
    (multiply, divide, sqrt, multiply) and their order are unchanged, so every
    output bit — including NaN/zero/overflow handling — is identical.
    """
    w1 = np.asarray(support_w1)
    w2 = np.asarray(support_w2)
    if w1.shape != w2.shape or w1.ndim != 2:
        raise CanonicalStackValidationError(
            "n_eff requires equal 2-D support arrays, got "
            f"{w1.shape} and {w2.shape}"
        )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        # n_eff is reused as the w1_sq buffer (one patch-sized float64 array).
        n_eff = np.multiply(w1, w1, dtype=np.float64)
        overflowed = ~np.isfinite(n_eff)
        np.divide(n_eff, w2, out=n_eff)
        valid_sup = w2 > 0.0
        overflow_valid = overflowed & valid_sup
        if np.any(overflow_valid):
            # Bounded overflow-safe branch: sqrt + ratio/safe (two transient
            # patch-sized arrays), only for a real overflow.
            sqrt_w2 = np.sqrt(w2)
            safe = np.divide(w1, sqrt_w2, dtype=np.float64)
            np.multiply(safe, safe, out=safe)
            np.copyto(n_eff, safe, where=overflow_valid)
            del sqrt_w2, safe
        # Exact final zeroing for invalid/non-finite/negative cells.
        bad = ~valid_sup
        bad |= ~np.isfinite(n_eff)
        bad |= n_eff < 0.0
        np.copyto(n_eff, 0.0, where=bad)
    return n_eff


def _finalize_phase2_outputs(
    p1: _Phase1,
    *,
    science_c,
    weight_sum_c,
    valid_mask_c,
    surviving_c,
    support_w1,
    support_w2,
    iterations_used: int,
    initial_count: int,
    rejected_count: int,
    low_n_count: int,
    degenerate_count: int,
    rejection_mask_out,
    prefetch_report,
):
    """Restore shapes, compute n_eff and assemble diagnostics (single source).

    Extracted verbatim from :func:`_phase2` so the external column-batched
    executor produces BYTE-IDENTICAL planes/diagnostics without duplicating
    science code. No behaviour change for existing callers.
    """
    # --- restore shapes + n_eff + diagnostics ---
    rejected_fraction = (rejected_count / initial_count) if initial_count > 0 else 0.0

    # n_eff_support = W1**2 / W2 (exact, bounded live set; see helper).
    n_eff = _compute_n_eff_bounded(support_w1, support_w2)

    science = _restore_shape(science_c, p1.original_mono)
    weight_sum = _restore_shape(weight_sum_c, p1.original_mono)
    valid_mask = _restore_shape(valid_mask_c, p1.original_mono)
    surviving = _restore_shape(surviving_c, p1.original_mono)

    return science, weight_sum, valid_mask, surviving, support_w1, support_w2, n_eff, {
        "iterations_used": iterations_used,
        "initial_sample_count": initial_count,
        "rejected_sample_count": rejected_count,
        "rejected_fraction": rejected_fraction,
        "low_n_cell_count": low_n_count,
        "degenerate_cell_count": degenerate_count,
        "rejection_mask": rejection_mask_out,
        "tile_prefetch": prefetch_report,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_canonical_stack_streaming(
    provider: CanonicalFrameProvider,
    request: CanonicalStackRequest,
    *,
    tile_size=None,
    fixed: "FixedNormalization | None" = None,
    rejection_sink=None,
    phase_profiler=None,
    tile_prefetch=None,
    tile_prefetch_available_bytes=None,
    tile_prefetch_budget_bytes=None,
    tile_prefetch_base_aggregate_bytes=None,
    tile_prefetch_host_budget_bytes=None,
) -> CanonicalStackResult:
    """Bounded-memory canonical stack, bit-equivalent to ``run_canonical_stack``.

    Runs the canonical pipeline in two phases through ``provider`` (see module
    docstring). Returns a :class:`CanonicalStackResult` whose science/support/
    masks/diagnostics/provenance match ``run_canonical_stack(request)`` for the
    same inputs (the in-memory provider wraps ``request.images`` /
    ``request.geometric_support``).

    ``tile_size`` is ``None`` (single tile == full patch), an int (square tile),
    or a ``(th, tw)`` tuple. Results are identical across tile sizes (within the
    documented bit-exact contract). Backend ``"gpu"`` and explicit taper maps are
    not yet supported and raise clearly (never silently substituted).

    ``rejection_sink`` (ZM-ZEGRID-R52-OOC Lot A): optional bounded sink with
    ``shape == (N, H, W, C)``, bool ``dtype`` and a ``flush_batch()`` method.
    When supplied, the dense rejection plane is NOT allocated; interior
    rejection blocks are written to the sink exactly where the dense plane would
    have been written, ``flush_batch()`` is called after each interior block, and
    every scalar diagnostic (initial/surviving/rejected/low_n/degenerate/
    iterations) plus science/support/valid-mask planes are bit-identical to the
    dense run. The result then exposes ``rejection_mask=None`` because a
    count-only sink cannot reproduce the full mask. Default ``None`` preserves
    the existing dense behaviour exactly for every caller.

    ``fixed`` (ZM-ZEGRID-R11): an optional precomputed
    :class:`FixedNormalization` whose per-frame coefficients/weights/active flags
    replace the per-Cell phase-1 computation. When provided, the phase-1
    coefficient/weight computation is SKIPPED and the fixed gauge is applied
    verbatim (phase 2 applies the fixed coefficients pointwise exactly as it does
    the freshly-computed ones), so every Cell given the same gauge shares one
    photometric anchor. Results are bit-equal to computing the gauge over a Cell
    whose footprint equals the gauge's full footprint (see the R11 tests).

    With ``fixed``, the GLOBAL exclusions / active flags apply to EVERY Cell
    (R11 I2): a frame excluded or de-weighted at the gauge level (insufficient
    overlap with the global reference, failed quality metric, …) is excluded or
    de-weighted in every Cell, even a Cell whose per-Cell footprint would have
    kept it. This is a deliberate behaviour change vs the per-Cell exclusions of
    the non-fixed path and is what makes the science layout-independent.
    """
    if not isinstance(request, CanonicalStackRequest):
        raise CanonicalStackValidationError(
            f"request must be a CanonicalStackRequest, got {type(request).__name__}"
        )

    # Backend: cpu (default) or gpu (ZM-ZEGRID-R22, opt-in). Mirrors the engine's
    # validation: 'gpu' raises when CuPy/GPU is unavailable (never a silent CPU
    # fallback INSIDE the executor — the ZeGrid orchestrator owns the loud degrade).
    backend = request.backend
    if not isinstance(backend, str):
        raise CanonicalStackValidationError(
            f"backend must be a string 'cpu' or 'gpu', got {type(backend).__name__}"
        )
    token = backend.strip().lower()
    if token not in ("cpu", "gpu"):
        raise CanonicalStackValidationError(
            f"unknown backend {backend!r}; expected 'cpu' or 'gpu' (aliases not accepted)"
        )
    if token == "gpu" and not canonical_gpu_available():
        raise CanonicalStackValidationError(
            "backend 'gpu' requested but CuPy/GPU is unavailable"
        )

    # Basic provider sanity checks.
    for attr in ("n_frames", "height", "width", "channels", "original_mono"):
        if not hasattr(provider, attr):
            raise CanonicalStackValidationError(f"provider is missing attribute {attr!r}")
    if provider.n_frames != len(request.images):
        raise CanonicalStackValidationError(
            f"provider.n_frames {provider.n_frames} != request.images length {len(request.images)}"
        )

    if rejection_sink is not None:
        _validate_rejection_sink(
            rejection_sink,
            provider.n_frames,
            provider.height,
            provider.width,
            provider.channels,
        )

    p1 = _fixed_phase1(provider, request, fixed)

    # ZM-PHASE3-TILE-PREFETCH: admission BEFORE any activation. The slot is the
    # extra working set of ONE prepared spatial tile; it is priced separately
    # from the planned streaming bound and never bypasses the product RAM caps.
    # Phase 1 is already complete here, so the producer starts only now.
    _taper_kind, _taper_px, _taper_floor, _prefetch_halo = _resolve_taper(request)
    tile_prefetch_plan = admit_tile_prefetch(
        requested=tile_prefetch,
        backend=token,
        n_frames=provider.n_frames,
        height=provider.height,
        width=provider.width,
        channels=provider.channels,
        tile_size=tile_size,
        halo_px=_prefetch_halo,
        available_bytes=tile_prefetch_available_bytes,
        budget_bytes=tile_prefetch_budget_bytes,
        base_aggregate_bytes=tile_prefetch_base_aggregate_bytes,
        host_budget_bytes=tile_prefetch_host_budget_bytes,
    )

    (science, weight_sum, valid_mask, surviving, support_w1, support_w2, n_eff, agg) = _phase2(
        provider, p1, request, tile_size, backend=token, rejection_sink=rejection_sink,
        phase_profiler=phase_profiler, tile_prefetch_plan=tile_prefetch_plan,
    )

    return finalize_streaming_result(
        provider, request, p1,
        science=science, weight_sum=weight_sum, valid_mask=valid_mask,
        surviving=surviving, support_w1=support_w1, support_w2=support_w2,
        n_eff=n_eff, agg=agg, backend_token=token,
        tile_prefetch_plan=tile_prefetch_plan,
    )


def finalize_streaming_result(
    provider,
    request: CanonicalStackRequest,
    p1: _Phase1,
    *,
    science,
    weight_sum,
    valid_mask,
    surviving,
    support_w1,
    support_w2,
    n_eff,
    agg,
    backend_token: str,
    tile_prefetch_plan,
    backend_requested_token: str | None = None,
) -> CanonicalStackResult:
    """Assemble the canonical result from phase-2 arrays (single source).

    Extracted verbatim from :func:`run_canonical_stack_streaming` so the
    external (column-batched) executor builds BYTE-IDENTICAL results/provenance
    without duplicating the post-combine/equalization/provenance logic. No
    behaviour change for existing callers.

    ``backend_requested_token`` (additive, default ``None``) lets an executor
    that fell back report the REQUESTED backend separately from the EFFECTIVE
    ``backend_token`` (e.g. GPU requested, CPU actually reduced). ``None`` keeps
    the historical single-token provenance byte-for-byte.
    """
    # --- post-combine RGB equalization (out-of-place, spatial-only) ---
    equalize_info = None
    if request.equalize_rgb:
        if p1.channels != 3:
            raise CanonicalStackValidationError(
                "equalize_rgb requires an RGB (3-channel) stack; got "
                f"channels={p1.channels} (mono/HWC1 not supported)"
            )
        science, equalize_info = equalize_rgb_medians_copy(science)

    if equalize_info is None:
        equalize_prov = {"requested": False, "applied": False}
        eq_applied = False
        eq_decision = None
        eq_gains = [1.0, 1.0, 1.0]
    else:
        tm = equalize_info["target_median"]
        tm = None if not np.isfinite(tm) else float(tm)
        equalize_prov = {
            "requested": True,
            "decision": equalize_info["decision"],
            "applied": bool(equalize_info["applied"]),
            "raw_gains": equalize_info["raw_gains"],
            "clipped_gains": equalize_info["clipped_gains"],
            "target_median": tm,
            "samples": int(equalize_info["samples"]),
            "mask_coverage": float(equalize_info["mask_coverage"]),
        }
        eq_applied = bool(equalize_info["applied"])
        eq_decision = equalize_info["decision"]
        eq_gains = equalize_info["clipped_gains"]

    # --- provenance (mirrors run_canonical_stack exactly) ---
    requested_backend = (
        str(backend_token)
        if backend_requested_token is None
        else str(backend_requested_token)
    )
    reference_mode = "explicit" if request.reference_index is not None else "auto"
    excluded_frames = [(int(e.index), e.stage, e.reason) for e in p1.exclusions]

    # Taper kind for provenance (string form; explicit already rejected above).
    taper_arg = request.taper
    taper_kind = taper_arg.strip().lower() if isinstance(taper_arg, str) else "explicit"

    provenance = {
        "normalization": {"requested": request.normalization, "effective": _resolve_method_token(request.normalization, _NORM_METHODS, "normalization")},
        "weighting": {"requested": request.weighting, "effective": _resolve_method_token(request.weighting, _WEIGHT_METHODS, "weighting")},
        "rejection": {"requested": request.rejection, "effective": _resolve_reject_token(request.rejection)},
        "combine": {"requested": request.combine, "effective": _resolve_method_token(request.combine, _COMBINE_METHODS, "combine")},
        "backend": {
            "requested": requested_backend,
            "normalization": "cpu",
            "weighting": "cpu",
            "support": "cpu",
            "rejection": backend_token,
            "combine": backend_token,
        },
        "reference": {"mode": reference_mode, "index": int(p1.reference_index)},
        "taper": {"kind": taper_kind, "px": float(request.taper_px), "floor": float(request.taper_floor)},
        "equalize_rgb": equalize_prov,
        "excluded_frames": excluded_frames,
        "effective_event": {
            "normalization": _resolve_method_token(request.normalization, _NORM_METHODS, "normalization"),
            "weighting": _resolve_method_token(request.weighting, _WEIGHT_METHODS, "weighting"),
            "rejection": _resolve_reject_token(request.rejection),
            "combine": _resolve_method_token(request.combine, _COMBINE_METHODS, "combine"),
            "backend_requested": requested_backend,
            "backend_normalization": "cpu",
            "backend_weighting": "cpu",
            "backend_support": "cpu",
            "backend_rejection": backend_token,
            "backend_combine": backend_token,
            "reference_index": int(p1.reference_index),
            "reference_mode": reference_mode,
            "taper": taper_kind,
            "taper_px": float(request.taper_px),
            "taper_floor": float(request.taper_floor),
            "equalize_rgb_applied": eq_applied,
            "equalize_rgb_decision": eq_decision,
            "equalize_rgb_gains": eq_gains,
        },
    }
    # Additive, request-only key: the historical (off) provenance stays
    # byte-identical for every generic caller and existing parity test.
    if str(tile_prefetch_plan.get("requested", "off")) != "off":
        # F3 (rework R1): publish the RUNTIME report (queue/reads/tiles/join)
        # when the mechanism was effective, not just the admission plan.
        _runtime_prefetch = agg.get("tile_prefetch")
        provenance["tile_prefetch"] = dict(
            _runtime_prefetch if _runtime_prefetch is not None else tile_prefetch_plan
        )

    return CanonicalStackResult(
        science=science,
        estimator_weight_sum=weight_sum,
        valid_mask=valid_mask,
        surviving_sample_count=surviving,
        support_w1=support_w1,
        support_w2=support_w2,
        n_eff_support=n_eff,
        rejection_mask=agg["rejection_mask"],
        rejection_method=_resolve_reject_token(request.rejection),
        iterations_used=agg["iterations_used"],
        initial_sample_count=agg["initial_sample_count"],
        rejected_sample_count=agg["rejected_sample_count"],
        rejected_fraction=agg["rejected_fraction"],
        low_n_cell_count=agg["low_n_cell_count"],
        degenerate_cell_count=agg["degenerate_cell_count"],
        provenance=provenance,
        original_mono=p1.original_mono,
        n_frames=p1.n_frames,
        height=p1.height,
        width=p1.width,
        channels=p1.channels,
        original_ndim=p1.original_ndim,
        original_shape=p1.original_shape,
    )
