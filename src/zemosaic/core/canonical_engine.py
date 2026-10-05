"""SCI-05 Gate E2 — canonical engine orchestration (request → result).

Assembles the already-accepted canonical stages into ONE backend-neutral engine
entry point ``run_canonical_stack``:

    ALIGNED INPUTS
      → B1  prepare_canonical_inputs / select_canonical_reference /
            normalize_canonical_images
      → B2  compute_canonical_quality_weights
      → E1  build_canonical_estimator_weights (w = q*m*a) + positive-support
            accumulation (SUP_W1 / SUP_W2 / N_eff)
      → C1  reject_canonical_samples
      → C2  combine_canonical_samples
      → CanonicalStackResult(science, estimator_weight_sum, valid_mask,
            surviving_sample_count, support_w1/w2/n_eff_support, rejection_mask +
            diagnostics, provenance, shape metadata)

The support maps are accumulated from the **pre-rejection** ``s_i = w_i = q*m*a``
map and are therefore **rejection-independent** (a ``none`` rejection and a
``kappa_sigma``/WSC rejection over the same corpus produce identical support).

Gate E2 only: no Coverage render, no RGB equalizer wiring (``equalize_rgb=True``
raises an explicit deferred-token error), no GUI/config/migration/locales, no
production caller wiring, no ZSSS import.

Design invariants
-----------------
* Deterministic and side-effect free: never mutates the request or any input.
* Validation failures raise :class:`zemosaic.core.canonical_stacking.
  CanonicalStackValidationError`; a reference quality failure surfaces as
  ``CanonicalStackFailure`` (never swallowed); a backend ``"gpu"`` request
  propagates the explicit Gate-D behavior (validation error if unavailable, no
  silent CPU fallback).
* Per-stage backend: B1 (normalization), B2 (weighting), and E1 (support) execute
  on CPU; only C1 (rejection) and C2 (combine) honour ``request.backend``.
  ``provenance["backend"]`` reports the requested token plus the per-stage
  effective backend, and ``effective_event`` mirrors it — never claiming GPU for
  stages that ran on CPU.
* Provenance is a bounded dict of scalars/strings/lists (no array dumps).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zemosaic.core.canonical_stacking import (
    CanonicalStackValidationError,
    canonical_gpu_available,
    combine_canonical_samples,
    compute_canonical_quality_weights,
    normalize_canonical_images,
    prepare_canonical_inputs,
    reject_canonical_samples,
    select_canonical_reference,
)
from zemosaic.core.canonical_support import (
    PositiveSupportAccumulator,
    build_canonical_estimator_weights,
)
from zemosaic.core.canonical_equalize import equalize_rgb_medians_copy

__all__ = [
    "CanonicalStackRequest",
    "CanonicalStackResult",
    "run_canonical_stack",
]

_BACKEND_CPU = "cpu"
_BACKEND_GPU = "gpu"
_SUPPORTED_BACKENDS = (_BACKEND_CPU, _BACKEND_GPU)


def _resolve_backend_token(backend) -> str:
    """Normalize/validate the backend token (``"cpu"``/``"gpu"``), like the stages.

    Mirrors ``canonical_stacking._resolve_backend``: strip/lower then exact match;
    a non-string/unknown token, or ``"gpu"`` with CuPy/GPU unavailable, raises
    :class:`CanonicalStackValidationError` (never a silent CPU fallback). Only C1
    (rejection) and C2 (combine) honour this token; B1/B2/support stay CPU.
    """
    if not isinstance(backend, str):
        raise CanonicalStackValidationError(
            f"backend must be a string 'cpu' or 'gpu', got {type(backend).__name__}"
        )
    token = backend.strip().lower()
    if token not in _SUPPORTED_BACKENDS:
        raise CanonicalStackValidationError(
            f"unknown backend {backend!r}; expected 'cpu' or 'gpu' (aliases not accepted)"
        )
    if token == _BACKEND_GPU and not canonical_gpu_available():
        raise CanonicalStackValidationError(
            "backend 'gpu' requested but CuPy/GPU is unavailable"
        )
    return token


@dataclass(frozen=True)
class CanonicalStackRequest:
    """Frozen canonical stack request (raw inputs + stage config).

    Attributes
    ----------
    images / geometric_support:
        Raw inputs exactly as ``prepare_canonical_inputs`` accepts (a sequence of
        HW mono or HWC frames and a matching sequence of 2-D bool masks).
    normalization / weighting / rejection / combine:
        Stage method tokens (validated by the respective stage).
    reference_index:
        Optional explicit reference frame index (``None`` = auto-select).
    taper / taper_px / taper_floor:
        Footprint-taper config: ``"footprint"`` (generate per-frame from
        ``valid_mask`` via ``make_footprint_taper``), ``"none"`` (``a_i = 1``), or
        an EXPLICIT per-frame taper map — an ``(N, H, W)`` float array in ``[0, 1]``
        or a length-N sequence of ``(H, W)`` float arrays — forwarded verbatim to
        ``build_canonical_estimator_weights`` (validated there).
    sigma_low / sigma_high / max_iters / winsor_limit_low / winsor_limit_high:
        Rejection parameters (forwarded to ``reject_canonical_samples``).
    backend:
        ``"cpu"`` (default) or ``"gpu"``; forwarded to C1/C2 only (B1/B2/support
        stay CPU).
    equalize_rgb:
        Optional post-combine RGB equalization (decision J). ``True`` applies the
        canonical equalizer out-of-place to the combined science (RGB only;
        mono/HWC1 raises ``CanonicalStackValidationError``).
    """

    images: object
    geometric_support: object
    normalization: str
    weighting: str
    rejection: str
    combine: str
    reference_index: int | None = None
    taper: object = "footprint"
    taper_px: float = 8.0
    taper_floor: float = 0.0
    sigma_low: float = 3.0
    sigma_high: float = 3.0
    max_iters: int = 5
    winsor_limit_low: float = 0.05
    winsor_limit_high: float = 0.05
    backend: str = "cpu"
    equalize_rgb: bool = False


@dataclass(frozen=True)
class CanonicalStackResult:
    """Deterministic output of :func:`run_canonical_stack`.

    Attributes
    ----------
    science / estimator_weight_sum / valid_mask / surviving_sample_count:
        C2 combine outputs (restored original shape; float32 / float64 / bool /
        int64 respectively).
    support_w1 / support_w2 / n_eff_support:
        Owned float64 ``(H, W)`` positive-support maps from the E1 accumulator
        over the pre-rejection ``s_i = w_i = q*m*a`` (rejection-independent).
    rejection_mask:
        Owned bool ``(N, H, W, C)`` C1 rejection mask.
    rejection_method / iterations_used / initial_sample_count /
    rejected_sample_count / rejected_fraction / low_n_cell_count /
    degenerate_cell_count:
        C1 rejection diagnostics (bounded scalars).
    provenance:
        Bounded dict (requested/effective per stage, reference mode+index, taper,
        equalize_rgb, excluded_frames, effective_event) — no array dumps.
    original_mono / n_frames / height / width / channels / original_ndim /
    original_shape:
        Shape metadata for downstream restoration/assembly.
    """

    science: np.ndarray
    estimator_weight_sum: np.ndarray
    valid_mask: np.ndarray
    surviving_sample_count: np.ndarray
    support_w1: np.ndarray
    support_w2: np.ndarray
    n_eff_support: np.ndarray
    rejection_mask: np.ndarray
    rejection_method: str
    iterations_used: int
    initial_sample_count: int
    rejected_sample_count: int
    rejected_fraction: float
    low_n_cell_count: int
    degenerate_cell_count: int
    provenance: dict
    original_mono: bool
    n_frames: int
    height: int
    width: int
    channels: int
    original_ndim: int
    original_shape: tuple


def run_canonical_stack(request: CanonicalStackRequest) -> CanonicalStackResult:
    """Run the canonical stack pipeline (B1 → B2 → E1 → C1 → C2).

    Deterministic, backend-neutral, side-effect free. Never mutates the request
    or any input array. ``equalize_rgb=True`` applies the canonical equalizer
    out-of-place to the combined science (RGB only); a reference quality failure
    propagates ``CanonicalStackFailure``; a ``"gpu"`` backend propagates the
    explicit Gate-D validation (no silent CPU fallback).
    """
    if not isinstance(request, CanonicalStackRequest):
        raise CanonicalStackValidationError(
            f"request must be a CanonicalStackRequest, got {type(request).__name__}"
        )

    # --- engine-level config validation (fail before any stage work) ---
    taper_arg = request.taper
    taper_kind = "none"
    builder_taper = None
    if isinstance(taper_arg, str):
        taper_kind = taper_arg.strip().lower()
        if taper_kind == "footprint":
            builder_taper = "footprint"
        elif taper_kind == "none":
            builder_taper = None
        else:
            raise CanonicalStackValidationError(
                f"taper string must be 'footprint' or 'none', got {request.taper!r}"
            )
    else:
        # A1 (additive): explicit per-frame taper map (N,H,W) array or length-N
        # sequence of (H,W) arrays in [0,1] — forwarded verbatim (validated by
        # build_canonical_estimator_weights via _resolve_taper).
        taper_kind = "explicit"
        builder_taper = taper_arg

    # Resolve/validate the backend token up front (only C1/C2 honour it; B1/B2/
    # support stay CPU). Fails before any stage work on an invalid/unavailable gpu.
    backend_token = _resolve_backend_token(request.backend)

    # --- B1: input preparation + reference selection + normalization ---
    batch = prepare_canonical_inputs(request.images, request.geometric_support)
    normalization = normalize_canonical_images(
        batch, request.normalization, reference_index=request.reference_index
    )

    # --- B2: scalar quality weights (reference failure raises CanonicalStackFailure) ---
    weighting = compute_canonical_quality_weights(normalization, request.weighting)

    # --- E1: estimator-weight map + positive-support accumulation ---
    wmap = build_canonical_estimator_weights(
        weighting.weights,
        normalization.valid_mask,
        taper=builder_taper,
        taper_px=request.taper_px,
        taper_floor=request.taper_floor,
        active_frames=weighting.active_frames,
    )

    # Accumulate pre-rejection support s_i = w_i in original-exposure order; only
    # active frames contribute (inactive frames contribute 0 and are skipped).
    accumulator = PositiveSupportAccumulator(
        (normalization.height, normalization.width), dtype=np.float64
    )
    for i in range(normalization.n_frames):
        if weighting.active_frames[i]:
            accumulator.add(wmap[i])

    # --- C1: rejection ---
    rejection = reject_canonical_samples(
        normalization,
        weighting,
        request.rejection,
        sigma_low=request.sigma_low,
        sigma_high=request.sigma_high,
        max_iters=request.max_iters,
        winsor_limit_low=request.winsor_limit_low,
        winsor_limit_high=request.winsor_limit_high,
        backend=backend_token,
    )

    # --- C2: combine ---
    combine = combine_canonical_samples(
        normalization, weighting, rejection, wmap, request.combine,
        backend=backend_token,
    )

    # --- E4: optional post-combine RGB equalization (out-of-place) ---
    science = combine.science
    equalize_info = None
    if request.equalize_rgb:
        if combine.channels != 3:
            raise CanonicalStackValidationError(
                "equalize_rgb requires an RGB (3-channel) stack; got "
                f"channels={combine.channels} (mono/HWC1 not supported)"
            )
        science, equalize_info = equalize_rgb_medians_copy(combine.science)

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

    # --- provenance (bounded: scalars/strings/lists, no arrays) ---
    reference_mode = "explicit" if request.reference_index is not None else "auto"
    excluded_frames = [
        (int(e.index), e.stage, e.reason) for e in rejection.exclusions
    ]
    provenance = {
        "normalization": {
            "requested": request.normalization,
            "effective": normalization.effective_method,
        },
        "weighting": {
            "requested": request.weighting,
            "effective": weighting.effective_method,
        },
        "rejection": {
            "requested": request.rejection,
            "effective": rejection.effective_method,
        },
        "combine": {
            "requested": request.combine,
            "effective": combine.effective_method,
        },
        "backend": {
            "requested": backend_token,
            "normalization": _BACKEND_CPU,
            "weighting": _BACKEND_CPU,
            "support": _BACKEND_CPU,
            "rejection": backend_token,
            "combine": backend_token,
        },
        "reference": {
            "mode": reference_mode,
            "index": int(normalization.reference_index),
        },
        "taper": {
            "kind": taper_kind,
            "px": float(request.taper_px),
            "floor": float(request.taper_floor),
        },
        "equalize_rgb": equalize_prov,
        "excluded_frames": excluded_frames,
        "effective_event": {
            "normalization": normalization.effective_method,
            "weighting": weighting.effective_method,
            "rejection": rejection.effective_method,
            "combine": combine.effective_method,
            "backend_requested": backend_token,
            "backend_normalization": _BACKEND_CPU,
            "backend_weighting": _BACKEND_CPU,
            "backend_support": _BACKEND_CPU,
            "backend_rejection": backend_token,
            "backend_combine": backend_token,
            "reference_index": int(normalization.reference_index),
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
        estimator_weight_sum=combine.estimator_weight_sum,
        valid_mask=combine.valid_mask,
        surviving_sample_count=combine.surviving_sample_count,
        support_w1=accumulator.support_w1,
        support_w2=accumulator.support_w2,
        n_eff_support=accumulator.n_eff_support,
        rejection_mask=rejection.rejection_mask,
        rejection_method=rejection.effective_method,
        iterations_used=rejection.iterations_used,
        initial_sample_count=rejection.initial_sample_count,
        rejected_sample_count=rejection.rejected_sample_count,
        rejected_fraction=rejection.rejected_fraction,
        low_n_cell_count=rejection.low_n_cell_count,
        degenerate_cell_count=rejection.degenerate_cell_count,
        provenance=provenance,
        original_mono=combine.original_mono,
        n_frames=combine.n_frames,
        height=combine.height,
        width=combine.width,
        channels=combine.channels,
        original_ndim=combine.original_ndim,
        original_shape=combine.original_shape,
    )
