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
    _astype_float32,
    _kappa_sigma_step,
    _nan_axis_median,
    _require_frame_sequence,
    _safe_divide,
    _validate_max_iters,
    _validate_reject_sigma,
    _validate_winsor_limit,
    _winsorized_sigma_clip_step,
)
from zemosaic.core.canonical_support import make_footprint_taper
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


def _restore_shape(arr_c, original_mono):
    """Restore the canonical (H, W, C) combine output to the original shape."""
    if original_mono:
        return arr_c[..., 0].copy()
    return np.ascontiguousarray(arr_c)


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

def _phase2(provider, p1: _Phase1, request: CanonicalStackRequest, tile_size) -> CanonicalStackResult:
    n = p1.n_frames
    h = p1.height
    w = p1.width
    c = p1.channels

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

    # Taper kind (none / footprint; explicit raises).
    taper_arg = request.taper
    taper_kind = "none"
    if isinstance(taper_arg, str):
        taper_kind = taper_arg.strip().lower()
        if taper_kind == "footprint":
            pass
        elif taper_kind == "none":
            taper_kind = "none"
        else:
            raise CanonicalStackValidationError(
                f"taper string must be 'footprint' or 'none', got {request.taper!r}"
            )
    else:
        raise CanonicalStackValidationError(
            "explicit per-frame taper maps are not yet supported by the streaming "
            "executor (supported: 'none' / 'footprint')"
        )
    taper_px = float(request.taper_px)
    taper_floor = float(request.taper_floor)
    halo = _halo_px(taper_kind, taper_px)

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
    rejection_mask = np.zeros((n, h, w, c), dtype=bool)

    for (y0, y1, x0, x1) in _iter_tiles(h, w, tile_size):
        # Extended tile bounds (with halo, clipped to the patch).
        ey0 = max(0, y0 - halo)
        ey1 = min(h, y1 + halo)
        ex0 = max(0, x0 - halo)
        ex1 = min(w, x1 + halo)
        th = ey1 - ey0
        tw = ex1 - ex0

        # Gather N frames: prepared raw tile + valid; then re-apply coefficients.
        images_t = np.empty((n, th, tw, c), dtype=np.float32)
        valid_t = np.zeros((n, th, tw), dtype=bool)
        for i in range(n):
            raw_tile, pre_valid = provider.get_tile(i, ey0, ey1, ex0, ex1)
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

        images64 = images_t.astype(np.float64)

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

        # --- rejection (reused step primitives on (N, tile_cells)) ---
        n_cells_t = th * tw * c
        initial = active[:, None, None, None] & valid_t[..., None] & np.isfinite(images64)
        orig2 = images64.reshape(n, n_cells_t)
        survivor2 = initial.reshape(n, n_cells_t)
        degenerate_seen = np.zeros(n_cells_t, dtype=bool)
        tile_iters = 0
        if reject_token == "none":
            survivor = initial.copy()
        else:
            for _ in range(iters):
                count = survivor2.sum(axis=0)
                if reject_token == "kappa_sigma":
                    new_survivor2, degenerate = _kappa_sigma_step(
                        orig2, survivor2, count, s_low, s_high, np
                    )
                else:
                    new_survivor2, degenerate = _winsorized_sigma_clip_step(
                        orig2, survivor2, count, s_low, s_high, w_low, w_high, np
                    )
                tile_iters += 1
                degenerate_seen |= degenerate
                if bool(np.array_equal(new_survivor2, survivor2)):
                    survivor2 = new_survivor2
                    break
                survivor2 = new_survivor2
            survivor = survivor2.reshape(n, th, tw, c)
            iterations_used = max(iterations_used, tile_iters)

        rejection_t = initial & ~survivor

        # --- combine (reused reductions on the N axis) ---
        w_positive = wmap[:, :, :, None] > 0.0
        eligible = survivor & w_positive & np.isfinite(images64)
        if combine_token == "mean":
            masked_images = np.where(eligible, images64, 0.0)
            w_contrib = np.where(eligible, wmap[:, :, :, None], 0.0)
            numerator = np.sum(masked_images * w_contrib, axis=0, dtype=np.float64)
            denominator = np.sum(w_contrib, axis=0, dtype=np.float64)
            den_valid = denominator > 0.0
            estimate64 = _safe_divide(numerator, denominator, np)
            science32 = _astype_float32(estimate64, np)
            final_finite = np.isfinite(estimate64) & np.isfinite(science32)
            valid_out = den_valid & final_finite
            science_t = np.where(valid_out, science32, np.nan)
            weight_sum_t = np.where(valid_out, denominator, 0.0)
        else:  # median
            masked = np.where(eligible, images64, np.nan)
            median64 = _nan_axis_median(masked, np)
            science32 = _astype_float32(median64, np)
            count = eligible.sum(axis=0).astype(np.float64)
            den_valid = count > 0.0
            final_finite = np.isfinite(median64) & np.isfinite(science32)
            valid_out = den_valid & final_finite
            science_t = np.where(valid_out, science32, np.nan)
            weight_sum_t = np.where(valid_out, count, 0.0)

        surviving_t = eligible.sum(axis=0).astype(np.int64)
        valid_mask_t = weight_sum_t > 0.0

        # --- support accumulation (per frame in order, interior written below) ---
        w1_tile = np.zeros((th, tw), dtype=np.float64)
        w2_tile = np.zeros((th, tw), dtype=np.float64)
        for i in range(n):
            if active[i]:
                w1_tile += wmap[i]
                w2_tile += wmap[i] * wmap[i]

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

        init_int = initial[:, iy0:iy1, ix0:ix1, :]
        rej_int = rejection_t[:, iy0:iy1, ix0:ix1, :]
        initial_count += int(init_int.sum())
        rejected_count += int(rej_int.sum())
        low_n_count += int((init_int.sum(axis=0) < 3).sum())
        degenerate_count += int(degenerate_seen.reshape(th, tw, c)[iy0:iy1, ix0:ix1, :].sum())

    # --- restore shapes + n_eff + diagnostics ---
    rejected_fraction = (rejected_count / initial_count) if initial_count > 0 else 0.0

    # n_eff_support = W1**2 / W2 (exact-first + overflow-resistant fallback).
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        w1_sq = support_w1 * support_w1
        n_eff = w1_sq / support_w2
        ratio = support_w1 / np.sqrt(support_w2)
        safe = ratio * ratio
    valid_sup = support_w2 > 0.0
    overflowed = ~np.isfinite(w1_sq)
    n_eff = np.where(overflowed & valid_sup, safe, n_eff)
    n_eff = np.where(valid_sup & np.isfinite(n_eff) & (n_eff >= 0.0), n_eff, 0.0)

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
        "rejection_mask": rejection_mask,
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

    # Backend: CPU only for streaming (validated up front, mirroring the engine).
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
    if token == "gpu":
        raise CanonicalStackValidationError(
            "backend 'gpu' is not yet supported by the streaming executor (CPU only)"
        )

    # Basic provider sanity checks.
    for attr in ("n_frames", "height", "width", "channels", "original_mono"):
        if not hasattr(provider, attr):
            raise CanonicalStackValidationError(f"provider is missing attribute {attr!r}")
    if provider.n_frames != len(request.images):
        raise CanonicalStackValidationError(
            f"provider.n_frames {provider.n_frames} != request.images length {len(request.images)}"
        )

    p1 = _fixed_phase1(provider, request, fixed)

    (science, weight_sum, valid_mask, surviving, support_w1, support_w2, n_eff, agg) = _phase2(
        provider, p1, request, tile_size
    )

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
            "requested": token,
            "normalization": "cpu",
            "weighting": "cpu",
            "support": "cpu",
            "rejection": token,
            "combine": token,
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
            "backend_requested": token,
            "backend_normalization": "cpu",
            "backend_weighting": "cpu",
            "backend_support": "cpu",
            "backend_rejection": token,
            "backend_combine": token,
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
