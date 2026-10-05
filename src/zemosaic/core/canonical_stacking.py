"""SCI-05 canonical stacking — Gate B1: input, reference selection, normalization.

Pure CPU, deterministic, backend-neutral primitives implementing the frozen
SCI-05 contract (``docs/science/SCI05_CANONICAL_STACKING_CONTRACT.md``) for the
**normalization and scalar quality weighting stages**, the **Gate C1**
canonical rejection primitives (``none``/``kappa_sigma``/
``winsorized_sigma_clip``), and the **Gate C2** pure canonical combine
primitive (``mean``/``median``). This is **Gate B1** (canonical input validation,
reference selection, per-frame normalization), **Gate B2** (scalar quality
weighting ``none``/``noise_variance``/``noise_fwhm``), **Gate C1** (rejection)
and **Gate C2** (combine only). Coverage support/taper construction (Gate E) and
the final request/result orchestration are explicitly out of scope here and are
assembled by later gates.

Design invariants
-----------------
* Side-effect free and deterministic. Never mutates caller arrays or masks.
* Canonical internal form is **NHWC float32** (owned, contiguous) with a
  channel-invariant 2-D ``bool`` valid mask ``(N, H, W)``.
* Statistics and affine coefficients are computed in **float64**; the canonical
  output images remain **float32**.
* The ``N`` axis and original frame indices are always preserved: zero-valid and
  excluded frames stay on the axis, marked ``active=False`` with a stable reason,
  never dropped or reindexed.
* Validation failures raise :class:`CanonicalStackValidationError`; a
  no-viable-frame / stage-level failure raises :class:`CanonicalStackFailure`.
  A per-frame normalization failure is never a hidden no-op or a method
  substitution: the frame is excluded with a structured reason.

Public API
----------
``prepare_canonical_inputs``, ``select_canonical_reference``,
``normalize_canonical_images``, ``compute_canonical_quality_weights``,
``reject_canonical_samples``, ``combine_canonical_samples``,
``canonical_noise_fwhm_available`` plus the frozen dataclasses
``CanonicalInputBatch`` / ``CanonicalNormalizationResult`` /
``CanonicalWeightingResult`` / ``CanonicalRejectionResult`` /
``CanonicalCombineResult`` and ``FrameExclusion``.
"""

from __future__ import annotations

import inspect
import operator
import warnings
from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clipped_stats

try:  # pragma: no cover - import availability is environment-dependent
    from photutils.segmentation import SegmentationImage as _SegmentationImage
    from photutils.segmentation import SourceCatalog as _SourceCatalog
    from photutils.segmentation import detect_sources as _detect_sources
    from photutils.utils.exceptions import NoDetectionsWarning as _NoDetectionsWarning
except Exception:  # pragma: no cover - defensive; photutils is a declared dependency
    _SegmentationImage = None
    _SourceCatalog = None
    _detect_sources = None
    _NoDetectionsWarning = None

__all__ = [
    "CanonicalStackValidationError",
    "CanonicalStackFailure",
    "FrameExclusion",
    "CanonicalInputBatch",
    "CanonicalNormalizationResult",
    "CanonicalWeightingResult",
    "CanonicalRejectionResult",
    "prepare_canonical_inputs",
    "select_canonical_reference",
    "normalize_canonical_images",
    "compute_canonical_quality_weights",
    "reject_canonical_samples",
    "combine_canonical_samples",
    "CanonicalCombineResult",
    "canonical_noise_fwhm_available",
]

# ---------------------------------------------------------------------------
# Stable, machine-testable reason codes and constants
# ---------------------------------------------------------------------------

REASON_ZERO_VALID_SUPPORT = "zero_valid_support"
REASON_INSUFFICIENT_COMMON = "insufficient_common"
REASON_DEGENERATE_OLS = "degenerate_ols"
REASON_NONFINITE_FIT = "nonfinite_fit"
REASON_SLOPE_OUT_OF_RANGE = "slope_out_of_range"
REASON_SKY_MEAN_FAILED = "sky_mean_failed"
REASON_NONFINITE_NORMALIZED_OUTPUT = "nonfinite_normalized_output"

STAGE_NORMALIZATION = "normalization"
STAGE_WEIGHTING = "weighting"

REASON_INSUFFICIENT_QUALITY_SAMPLES = "insufficient_quality_samples"
REASON_NOISE_SIGMA_FAILED = "noise_sigma_failed"
REASON_FWHM_INSUFFICIENT_SOURCES = "fwhm_insufficient_sources"
REASON_FWHM_MEASUREMENT_FAILED = "fwhm_measurement_failed"
REASON_RAW_WEIGHT_FAILED = "raw_weight_failed"

_METHOD_WEIGHT_NONE = "none"
_METHOD_WEIGHT_NOISE_VARIANCE = "noise_variance"
_METHOD_WEIGHT_NOISE_FWHM = "noise_fwhm"
_SUPPORTED_WEIGHT_METHODS = (
    _METHOD_WEIGHT_NONE,
    _METHOD_WEIGHT_NOISE_VARIANCE,
    _METHOD_WEIGHT_NOISE_FWHM,
)

_MIN_QUALITY_SAMPLES = 256
_REC709_R = 0.2126
_REC709_G = 0.7152
_REC709_B = 0.0722
_QUALITY_SIGMA_LOW = 3.0
_QUALITY_SIGMA_HIGH = 3.0
_QUALITY_SIGMA_MAXITERS = 5
_FWHM_MIN = 0.8
_FWHM_MAX = 20.0
_FWHM_ECC_MAX = 0.8
_FWHM_MIN_SOURCES = 3
_FWHM_THRESHOLD_SIGMA = 3.0
_FWHM_N_PIXELS = 5
_FWHM_CONNECTIVITY = 8

_METHOD_NONE = "none"
_METHOD_LINEAR_FIT = "linear_fit"
_METHOD_SKY_MEAN = "sky_mean"
_SUPPORTED_METHODS = (_METHOD_NONE, _METHOD_LINEAR_FIT, _METHOD_SKY_MEAN)

_MIN_COMMON_FLOOR = 256
_MIN_COMMON_FRACTION = 0.01
_MAD_SCALE = 1.4826
_KEEP_SIGMA = 3.0
_MAX_REFINEMENTS = 5
_SLOPE_LOW = 0.25
_SLOPE_HIGH = 4.0
_SKY_SIGMA_LOW = 3.0
_SKY_SIGMA_HIGH = 3.0
_SKY_MAXITERS = 5


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CanonicalStackValidationError(ValueError):
    """Raised for request/input/token/reference validation failures."""


class CanonicalStackFailure(RuntimeError):
    """Raised when no viable frame remains or a canonical stage fails."""


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameExclusion:
    """Structured, bounded diagnostic for a single excluded frame.

    Attributes
    ----------
    index:
        Original (preserved) frame index on the ``N`` axis.
    stage:
        Pipeline stage that produced the exclusion (``"normalization"`` here).
    reason:
        Stable machine-testable reason code (see module-level ``REASON_*``).
    detail:
        Optional bounded human-readable detail; never an array dump.
    """

    index: int
    stage: str
    reason: str
    detail: str | None = None


@dataclass(frozen=True)
class CanonicalInputBatch:
    """Owned, validated canonical input batch.

    Attributes
    ----------
    images:
        Owned contiguous ``(N, H, W, C)`` float32; never aliases the input.
    valid_mask:
        Owned contiguous ``(N, H, W)`` bool channel-invariant valid mask.
    original_mono:
        True only when every input was a 2-D ``HW`` image.
    frame_valid_counts:
        ``(N,)`` int64 count of valid pixels per frame (pre-normalization ``m_i``).
    n_frames / height / width / channels:
        Canonical shape metadata.
    original_ndim / original_shape:
        Enough metadata to restore the original mono/HWC shape downstream.
    """

    images: np.ndarray
    valid_mask: np.ndarray
    original_mono: bool
    frame_valid_counts: np.ndarray
    n_frames: int
    height: int
    width: int
    channels: int
    original_ndim: int
    original_shape: tuple


@dataclass(frozen=True)
class CanonicalNormalizationResult:
    """Deterministic output of :func:`normalize_canonical_images`.

    Attributes
    ----------
    images:
        Canonical NHWC float32; excluded frames are all-NaN; valid samples hold
        the normalized values, invalid samples stay NaN.
    valid_mask:
        NHW bool; all-false for excluded frames, otherwise unchanged.
    active_frames:
        ``(N,)`` bool preserving ``N`` and original indices.
    reference_index:
        Chosen reference frame index (identity).
    requested_method / effective_method:
        Normalized method token; always equal (validation errors raise instead
        of falling back).
    coefficients:
        ``(N, C, 2)`` float64 ``(a, b)`` for ``y = a*x + b``. Reference and
        ``none`` frames are identity ``(1, 0)``; excluded frames are NaN.
    exclusions:
        Tuple of :class:`FrameExclusion` records.
    original_mono / n_frames / height / width / channels / original_ndim /
    original_shape:
        Output shape metadata for downstream restoration.
    """

    images: np.ndarray
    valid_mask: np.ndarray
    active_frames: np.ndarray
    reference_index: int
    requested_method: str
    effective_method: str
    coefficients: np.ndarray
    exclusions: tuple
    original_mono: bool
    n_frames: int
    height: int
    width: int
    channels: int
    original_ndim: int
    original_shape: tuple


@dataclass(frozen=True)
class CanonicalWeightingResult:
    """Deterministic output of :func:`compute_canonical_quality_weights`.

    Attributes
    ----------
    weights:
        Owned contiguous ``(N,)`` float64 canonical normalized quality weights.
        Active survivors are positive with max exactly 1 for weighted methods;
        inactive/excluded frames are exactly 0.
    raw_weights:
        Owned ``(N,)`` float64. 1 for active ``none`` frames, the raw formula for
        successful weighted frames, NaN for metric-failed/prior-inactive frames.
    noise_sigma:
        Owned ``(N,)`` float64 robust sigma; NaN for ``none``/failed/inactive.
    fwhm:
        Owned ``(N,)`` float64 median source FWHM; NaN except successful
        ``noise_fwhm`` frames.
    active_frames:
        Owned ``(N,)`` bool preserving N and original frame indices.
    reference_index:
        Chosen normalization reference frame index (identity).
    requested_method / effective_method:
        Weighting method token; always equal (validation errors raise instead of
        falling back).
    exclusions:
        Tuple of :class:`FrameExclusion` records (B1 inherited + B2 weighting).
    original_mono / n_frames / height / width / channels / original_ndim /
    original_shape:
        Output shape metadata for downstream restoration (Gate C).
    """

    weights: np.ndarray
    raw_weights: np.ndarray
    noise_sigma: np.ndarray
    fwhm: np.ndarray
    active_frames: np.ndarray
    reference_index: int
    requested_method: str
    effective_method: str
    exclusions: tuple
    original_mono: bool
    n_frames: int
    height: int
    width: int
    channels: int
    original_ndim: int
    original_shape: tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_frame_sequence(images) -> int:
    if images is None or isinstance(images, (str, bytes, np.ndarray)):
        raise CanonicalStackValidationError(
            "images must be a non-empty sequence of frames"
        )
    try:
        n = len(images)
    except (TypeError, AttributeError):
        raise CanonicalStackValidationError(
            "images must be a non-empty sequence of frames"
        )
    if n == 0:
        raise CanonicalStackValidationError("images must be a non-empty sequence")
    return n


def _coerce_index(value) -> int:
    if isinstance(value, bool):
        raise CanonicalStackValidationError("reference_index must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError:
        raise CanonicalStackValidationError(
            f"reference_index must be an integer, got {type(value).__name__}"
        )


def _as_frame_array(arr, index: int):
    """Validate one input frame; return ``(ndarray, (h, w, c))``."""
    try:
        a = np.asarray(arr)
    except Exception as exc:  # pragma: no cover - defensive
        raise CanonicalStackValidationError(f"frame {index}: not array-convertible: {exc}")
    if a.dtype.kind not in "iuf":
        raise CanonicalStackValidationError(
            f"frame {index}: unsupported dtype {a.dtype}; expected real numeric "
            f"(integer or float), not object/complex/non-numeric"
        )
    if a.ndim == 2:
        h, w = a.shape
        c = 1
    elif a.ndim == 3:
        h, w, c = a.shape
        if c not in (1, 3):
            raise CanonicalStackValidationError(
                f"frame {index}: channel count {c} not in {{1, 3}}"
            )
    else:
        raise CanonicalStackValidationError(
            f"frame {index}: expected 2-D (HW) or 3-D (HWC), got shape {a.shape}"
        )
    if h == 0 or w == 0:
        raise CanonicalStackValidationError(
            f"frame {index}: empty spatial dimensions {a.shape}"
        )
    return a, (h, w, c)


def _as_bool_mask(mask, index: int, expected_shape: tuple) -> np.ndarray:
    try:
        m = np.asarray(mask)
    except Exception as exc:  # pragma: no cover - defensive
        raise CanonicalStackValidationError(
            f"geometric_support[{index}]: not array-convertible: {exc}"
        )
    if m.dtype != np.bool_:
        raise CanonicalStackValidationError(
            f"geometric_support[{index}]: expected bool dtype, got {m.dtype}"
        )
    if m.ndim != 2:
        raise CanonicalStackValidationError(
            f"geometric_support[{index}]: expected 2-D, got shape {m.shape}"
        )
    if m.shape != expected_shape:
        raise CanonicalStackValidationError(
            f"geometric_support[{index}]: shape {m.shape} != image shape {expected_shape}"
        )
    return m


def _to_float32_copy(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.array(a, dtype=np.float32, copy=True)


def _compute_min_common(ref_count: int, src_count: int) -> int:
    frac = int(np.ceil(_MIN_COMMON_FRACTION * min(int(ref_count), int(src_count))))
    return max(_MIN_COMMON_FLOOR, frac)


def _ols(x, y):
    """Unweighted float64 OLS for ``y = a*x + b``; returns ``(a, b)`` or ``(None, None)``.

    Degeneracy is deterministic: a zero or non-finite centered sum of squares
    denominator fails (no epsilon fallback).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return None, None
    mx = float(x.mean())
    my = float(y.mean())
    xc = x - mx
    yc = y - my
    sxx = float(np.dot(xc, xc))
    if not np.isfinite(sxx) or sxx <= 0.0:
        return None, None
    sxy = float(np.dot(xc, yc))
    a = sxy / sxx
    b = my - a * mx
    return a, b


def _sigma_clipped_mean(data) -> float:
    """Sigma-clipped MEAN (Astropy, median centering, 3/3 sigma, 5 iters).

    Any Astropy/statistics exception from the sigma-clip computation is converted
    to ``NaN`` here (narrow scope: only the statistics call is wrapped) so the
    caller turns it into a structured ``sky_mean_failed`` frame exclusion instead
    of leaking a raw exception or silently falling back.
    """
    data = np.asarray(data, dtype=np.float64)
    try:
        stats = sigma_clipped_stats(
            data,
            sigma_lower=_SKY_SIGMA_LOW,
            sigma_upper=_SKY_SIGMA_HIGH,
            maxiters=_SKY_MAXITERS,
        )
        mean = stats[0]
        if np.ma.is_masked(mean):
            return float("nan")
        return float(mean)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Public: input preparation
# ---------------------------------------------------------------------------

def prepare_canonical_inputs(images, geometric_support) -> CanonicalInputBatch:
    """Validate ``images`` + ``geometric_support`` and build an owned canonical batch.

    Returns a frozen :class:`CanonicalInputBatch`. See module docstring for the
    accepted input contract (homogeneous HW mono or HWC C∈{1,3}, explicit 2-D
    bool mask per image, owned NHWC float32 copy, channel-invariant validity
    from geometric support AND finite-all-channels, NaN outside the valid mask).
    """
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

    frames = []
    mono_flags = []
    shapes = []
    for i in range(n):
        a, (h, w, c) = _as_frame_array(images[i], i)
        frames.append(a)
        mono_flags.append(a.ndim == 2)
        shapes.append((h, w, c))

    if not (all(mono_flags) or not any(mono_flags)):
        raise CanonicalStackValidationError("mixed HW (mono) and HWC (color) frames")

    hw_set = {(s[0], s[1]) for s in shapes}
    if len(hw_set) != 1:
        raise CanonicalStackValidationError(
            "frames have inconsistent spatial shapes (H, W)"
        )
    c_set = {s[2] for s in shapes}
    if len(c_set) != 1:
        raise CanonicalStackValidationError("frames have inconsistent channel counts")

    height, width = next(iter(hw_set))
    channels = next(iter(c_set))
    original_mono = all(mono_flags)

    out_images = np.empty((n, height, width, channels), dtype=np.float32)
    out_mask = np.empty((n, height, width), dtype=bool)
    for i in range(n):
        a = _to_float32_copy(frames[i])
        if original_mono:
            a = a[..., np.newaxis]  # (H, W) -> (H, W, 1)
        finite_all_channels = np.all(np.isfinite(a), axis=-1)
        m = _as_bool_mask(geometric_support[i], i, (height, width))
        out_mask[i] = m & finite_all_channels
        out_images[i] = a

    # Invalidate every channel outside the valid mask (NaN), never mutating input.
    out_images[~out_mask] = np.nan

    counts = out_mask.sum(axis=(1, 2)).astype(np.int64)
    original_shape = (height, width) if original_mono else (height, width, channels)
    original_ndim = 2 if original_mono else 3

    return CanonicalInputBatch(
        images=np.ascontiguousarray(out_images),
        valid_mask=np.ascontiguousarray(out_mask),
        original_mono=original_mono,
        frame_valid_counts=counts,
        n_frames=n,
        height=height,
        width=width,
        channels=channels,
        original_ndim=original_ndim,
        original_shape=original_shape,
    )


# ---------------------------------------------------------------------------
# Public: reference selection
# ---------------------------------------------------------------------------

def select_canonical_reference(batch: CanonicalInputBatch, reference_index=None) -> int:
    """Choose the normalization reference index (identity frame).

    Explicit index must be an integer (not bool), in range, and have a positive
    pre-normalization valid count. Auto selects the frame with the greatest
    ``frame_valid_counts`` (no weights/rejection); a tie resolves to the lowest
    original index. All-zero counts raise :class:`CanonicalStackFailure`.
    """
    counts = np.asarray(batch.frame_valid_counts, dtype=np.int64)
    n = batch.n_frames

    if reference_index is None:
        if not counts.any():
            raise CanonicalStackFailure(
                "all frames have zero valid support; no reference available"
            )
        return int(np.argmax(counts))  # stable: lowest index on ties

    idx = _coerce_index(reference_index)
    if idx < 0 or idx >= n:
        raise CanonicalStackValidationError(f"reference_index {idx} out of range [0, {n})")
    if counts[idx] <= 0:
        raise CanonicalStackValidationError(
            f"reference_index {idx} has zero valid support"
        )
    return idx


# ---------------------------------------------------------------------------
# Public: normalization
# ---------------------------------------------------------------------------

def _fit_linear_channel(ref, src, common, min_common: int):
    """Robust per-channel affine fit ``y_ref = a*x_src + b``.

    Returns ``(a, b, reason, final_mask)``; ``reason`` is ``None`` on success and
    ``a``/``b`` are ``None`` on failure. ``final_mask`` is the accepted common
    subset (monotonic non-increasing refinement, ≤5 iterations).
    """
    x = src[common]
    y = ref[common]
    a, b = _ols(x, y)
    if a is None:
        return None, None, REASON_DEGENERATE_OLS, common
    if not (np.isfinite(a) and np.isfinite(b)):
        return None, None, REASON_NONFINITE_FIT, common

    mask = common.copy()
    for _ in range(_MAX_REFINEMENTS):
        xm = src[mask]
        ym = ref[mask]
        resid = ym - (a * xm + b)
        center = float(np.median(resid))
        mad = float(np.median(np.abs(resid - center)))
        scale = _MAD_SCALE * mad
        if scale > 0.0:
            keep = np.abs(resid - center) <= _KEEP_SIGMA * scale
        else:
            keep = resid == center
        new_mask = mask.copy()
        new_mask[mask] = keep  # monotonic intersection with existing mask
        changed = not np.array_equal(new_mask, mask)
        mask = new_mask
        if int(mask.sum()) < min_common:
            return None, None, REASON_INSUFFICIENT_COMMON, mask
        a, b = _ols(src[mask], ref[mask])
        if a is None:
            return None, None, REASON_DEGENERATE_OLS, mask
        if not (np.isfinite(a) and np.isfinite(b)):
            return None, None, REASON_NONFINITE_FIT, mask
        if not changed:
            break

    if not (_SLOPE_LOW <= a <= _SLOPE_HIGH):
        return None, None, REASON_SLOPE_OUT_OF_RANGE, mask
    return a, b, None, mask


def _sky_mean_channel(ref, src, common):
    """Return ``(offset, reason)`` for the additive sky-mean shift (a=1)."""
    mr = _sigma_clipped_mean(ref[common])
    ms = _sigma_clipped_mean(src[common])
    offset = mr - ms
    if not (np.isfinite(mr) and np.isfinite(ms) and np.isfinite(offset)):
        return None, REASON_SKY_MEAN_FAILED
    return offset, None


def normalize_canonical_images(
    batch: CanonicalInputBatch, method, reference_index=None
) -> CanonicalNormalizationResult:
    """Apply canonical normalization to a prepared batch.

    ``method`` must be one of ``none``, ``linear_fit``, ``sky_mean`` (after
    strip/lower). ``reference_index`` is optional; when omitted the reference is
    auto-selected. See module docstring for the full per-method contract.
    """
    if not isinstance(method, str):
        raise CanonicalStackValidationError(
            f"method must be a string, got {type(method).__name__}"
        )
    norm = method.strip().lower()
    if norm not in _SUPPORTED_METHODS:
        raise CanonicalStackValidationError(
            f"unsupported normalization method {method!r}; expected one of "
            "none/linear_fit/sky_mean (aliases are not accepted)"
        )

    ref_idx = select_canonical_reference(batch, reference_index)

    n = batch.n_frames
    h = batch.height
    w = batch.width
    c = batch.channels
    counts = np.asarray(batch.frame_valid_counts, dtype=np.int64)

    images = np.array(batch.images, dtype=np.float32, copy=True)
    valid = np.array(batch.valid_mask, dtype=bool, copy=True)
    coefficients = np.full((n, c, 2), np.nan, dtype=np.float64)
    active = counts > 0
    exclusions = []

    ref64 = batch.images[ref_idx].astype(np.float64)

    for i in range(n):
        if counts[i] <= 0:
            active[i] = False
            exclusions.append(
                FrameExclusion(index=i, stage=STAGE_NORMALIZATION, reason=REASON_ZERO_VALID_SUPPORT)
            )
            images[i] = np.nan
            valid[i] = False
            continue

        if i == ref_idx:
            coefficients[i, :, :] = (1.0, 0.0)
            continue

        if norm == _METHOD_NONE:
            coefficients[i, :, :] = (1.0, 0.0)
            continue

        common = valid[ref_idx] & valid[i]
        common_count = int(common.sum())
        min_common = _compute_min_common(int(counts[ref_idx]), int(counts[i]))
        if common_count < min_common:
            active[i] = False
            exclusions.append(
                FrameExclusion(
                    index=i,
                    stage=STAGE_NORMALIZATION,
                    reason=REASON_INSUFFICIENT_COMMON,
                    detail=f"common={common_count} < min_common={min_common}",
                )
            )
            images[i] = np.nan
            valid[i] = False
            coefficients[i, :, :] = np.nan
            continue

        src64 = batch.images[i].astype(np.float64)
        per_channel = []
        failed_reason = None
        for ch in range(c):
            if norm == _METHOD_LINEAR_FIT:
                a, b, reason, _ = _fit_linear_channel(
                    ref64[:, :, ch], src64[:, :, ch], common, min_common
                )
                if reason is not None:
                    failed_reason = reason
                    break
                per_channel.append((a, b))
            else:  # sky_mean
                offset, reason = _sky_mean_channel(
                    ref64[:, :, ch], src64[:, :, ch], common
                )
                if reason is not None:
                    failed_reason = reason
                    break
                per_channel.append((1.0, offset))

        if failed_reason is not None:
            active[i] = False
            exclusions.append(
                FrameExclusion(index=i, stage=STAGE_NORMALIZATION, reason=failed_reason)
            )
            images[i] = np.nan
            valid[i] = False
            coefficients[i, :, :] = np.nan
            continue

        # Apply the accepted affine/additive transform to ALL source-valid pixels
        # per channel (not only the overlap-fit pixels), computed in float64 and
        # cast to float32 under controlled warning handling. No clipping/saturation.
        transformed = np.empty((h, w, c), dtype=np.float64)
        for ch in range(c):
            a, b = per_channel[ch]
            coefficients[i, ch, :] = (a, b)
            transformed[:, :, ch] = a * src64[:, :, ch] + b

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            transformed32 = transformed.astype(np.float32)

        # Re-enforce the finite/all-channel invariant: a float32 overflow
        # (finite float64 -> Inf float32) invalidates the pixel for ALL channels
        # (channel-invariant), so NaN/Inf is never left marked valid.
        prior_valid = valid[i]
        finite_all_channels = np.all(np.isfinite(transformed32), axis=-1)
        post_valid = prior_valid & finite_all_channels

        frame = images[i]
        frame[:] = np.nan
        frame[post_valid] = transformed32[post_valid]
        valid[i] = post_valid

        if not post_valid.any():
            # Whole non-reference frame became nonfinite after normalization.
            active[i] = False
            exclusions.append(
                FrameExclusion(
                    index=i,
                    stage=STAGE_NORMALIZATION,
                    reason=REASON_NONFINITE_NORMALIZED_OUTPUT,
                )
            )
            images[i] = np.nan
            valid[i] = False
            coefficients[i, :, :] = np.nan
            continue

    if not active.any():  # pragma: no cover - reference is always active
        raise CanonicalStackFailure("no active frame remains after normalization")

    return CanonicalNormalizationResult(
        images=np.ascontiguousarray(images),
        valid_mask=np.ascontiguousarray(valid),
        active_frames=active,
        reference_index=ref_idx,
        requested_method=norm,
        effective_method=norm,
        coefficients=coefficients,
        exclusions=tuple(exclusions),
        original_mono=batch.original_mono,
        n_frames=n,
        height=h,
        width=w,
        channels=c,
        original_ndim=batch.original_ndim,
        original_shape=batch.original_shape,
    )


# ---------------------------------------------------------------------------
# Gate B2: scalar quality weighting
# ---------------------------------------------------------------------------

def _frame_luminance(frame: np.ndarray) -> np.ndarray:
    """Return the float64 luminance plane of one NHWC frame.

    RGB uses exact Rec.709 ``0.2126R + 0.7152G + 0.0722B``; HWC1 uses channel 0.
    """
    if frame.shape[-1] == 1:
        return frame[:, :, 0].astype(np.float64)
    r = frame[:, :, 0].astype(np.float64)
    g = frame[:, :, 1].astype(np.float64)
    b = frame[:, :, 2].astype(np.float64)
    return _REC709_R * r + _REC709_G * g + _REC709_B * b


def _noise_stats(lum: np.ndarray, valid: np.ndarray) -> tuple:
    """Return ``(median, sigma, reason)`` for the valid luminance samples.

    ``reason`` is ``None`` on success. Requires at least
    ``_MIN_QUALITY_SAMPLES`` valid samples; sigma-clipped stats use default
    median centering / std clipping (3/3, maxiters 5). Requires finite sigma
    > 0 and finite median. Astropy/statistics exceptions are contained into
    ``noise_sigma_failed`` (no raw leak, no fallback).
    """
    n_valid = int(np.count_nonzero(valid))
    if n_valid < _MIN_QUALITY_SAMPLES:
        return float("nan"), float("nan"), REASON_INSUFFICIENT_QUALITY_SAMPLES
    valid_lum = lum[valid]
    try:
        stats = sigma_clipped_stats(
            valid_lum,
            sigma_lower=_QUALITY_SIGMA_LOW,
            sigma_upper=_QUALITY_SIGMA_HIGH,
            maxiters=_QUALITY_SIGMA_MAXITERS,
        )
        median = float(stats[1])
        sigma = float(stats[2])
    except Exception:
        return float("nan"), float("nan"), REASON_NOISE_SIGMA_FAILED
    if not (np.isfinite(sigma) and sigma > 0.0):
        return float("nan"), float("nan"), REASON_NOISE_SIGMA_FAILED
    if not np.isfinite(median):
        return float("nan"), float("nan"), REASON_NOISE_SIGMA_FAILED
    return median, sigma, None


def _source_value_to_float(value) -> float:
    """Deterministically convert a Photutils source property to float.

    Astropy ``Quantity`` values expose ``.value``; ``float(Quantity)`` raises for
    dimensioned quantities (e.g. FWHM in pixels), so ``.value`` is read explicitly.
    """
    if hasattr(value, "value"):
        return float(value.value)
    return float(value)


def _frame_fwhm(
    lum: np.ndarray, valid: np.ndarray, median: float, sigma: float
) -> tuple:
    """Return ``(fwhm_median, reason)`` from Photutils source detection.

    Exactly one detect pass on ``luminance - median`` (invalid pixels zeroed and
    supplied as mask), threshold ``3*sigma``, ``n_pixels=5``, connectivity 8. No
    deblend, no lowered-threshold/DAO/moment/custom fallback, no property
    fallback. Accepts sources with finite ``0.8 < fwhm < 20`` px and
    ``eccentricity <= 0.8``; requires >= 3 accepted sources; frame FWHM is the
    float64 median of accepted FWHMs.
    """
    if _detect_sources is None or _SourceCatalog is None:
        return float("nan"), REASON_FWHM_MEASUREMENT_FAILED

    plane = (lum - median).astype(np.float64)
    plane[~valid] = 0.0
    mask = ~valid

    try:
        with warnings.catch_warnings():
            if _NoDetectionsWarning is not None:
                # "no sources found" is an expected, handled branch (mapped to
                # fwhm_insufficient_sources), not an actionable warning.
                warnings.simplefilter("ignore", _NoDetectionsWarning)
            segm = _detect_sources(
                plane,
                threshold=_FWHM_THRESHOLD_SIGMA * sigma,
                n_pixels=_FWHM_N_PIXELS,
                connectivity=_FWHM_CONNECTIVITY,
                mask=mask,
            )
    except Exception:
        return float("nan"), REASON_FWHM_MEASUREMENT_FAILED

    if segm is None:
        return float("nan"), REASON_FWHM_INSUFFICIENT_SOURCES
    try:
        # Current Photutils name is ``n_labels`` (``nlabels`` deprecated).
        n_labels = int(segm.n_labels)
    except AttributeError:  # pragma: no cover - older Photutils
        n_labels = int(getattr(segm, "nlabels", 0))
    except Exception:
        n_labels = 0
    if n_labels <= 0:
        return float("nan"), REASON_FWHM_INSUFFICIENT_SOURCES

    try:
        catalog = _SourceCatalog(plane, segm, mask=mask, progress_bar=False)
        accepted = []
        for source in catalog:
            try:
                fwhm = _source_value_to_float(source.fwhm)
                ecc = _source_value_to_float(source.eccentricity)
            except Exception:
                continue  # reject missing/non-convertible values
            if not (np.isfinite(fwhm) and np.isfinite(ecc)):
                continue
            if not (_FWHM_MIN < fwhm < _FWHM_MAX):
                continue
            if not (ecc <= _FWHM_ECC_MAX):
                continue
            accepted.append(fwhm)
    except Exception:
        return float("nan"), REASON_FWHM_MEASUREMENT_FAILED

    if len(accepted) < _FWHM_MIN_SOURCES:
        return float("nan"), REASON_FWHM_INSUFFICIENT_SOURCES
    fwhm_median = float(np.median(np.asarray(accepted, dtype=np.float64)))
    if not (np.isfinite(fwhm_median) and fwhm_median > 0.0):
        return float("nan"), REASON_FWHM_MEASUREMENT_FAILED
    return fwhm_median, None


def canonical_noise_fwhm_available() -> bool:
    """Read-only availability signal for the ``noise_fwhm`` weighting method.

    True only when Photutils detection/catalog support is importable AND every
    exact capability the estimator calls is present, so an importable-but-API-
    incompatible Photutils reports unavailable **before** any per-frame work:

    * ``detect_sources`` accepts named ``n_pixels``, ``connectivity``, ``mask``;
    * ``SourceCatalog`` accepts named ``mask``, ``progress_bar``;
    * ``SourceCatalog`` exposes the exact ``fwhm`` property (no substitute);
    * ``SegmentationImage`` exposes ``n_labels`` (or legacy ``nlabels``).

    Signature introspection is wrapped in a bounded try/except; any missing
    capability or introspection failure returns False. Suitable for later GUI
    option validation.
    """
    if _detect_sources is None or _SourceCatalog is None or _SegmentationImage is None:
        return False
    try:
        detect_params = inspect.signature(_detect_sources).parameters
        if not {"n_pixels", "connectivity", "mask"}.issubset(detect_params):
            return False
        catalog_params = inspect.signature(_SourceCatalog).parameters
        if not {"mask", "progress_bar"}.issubset(catalog_params):
            return False
    except (TypeError, ValueError):
        return False
    if not hasattr(_SourceCatalog, "fwhm"):
        return False
    if not (hasattr(_SegmentationImage, "n_labels") or hasattr(_SegmentationImage, "nlabels")):
        return False
    return True


def compute_canonical_quality_weights(
    normalization: CanonicalNormalizationResult, method: str
) -> CanonicalWeightingResult:
    """Compute canonical scalar quality weights on a normalized batch.

    ``method`` must be exactly ``none``, ``noise_variance``, or ``noise_fwhm``
    (after strip/lower; aliases/unknown rejected with
    :class:`CanonicalStackValidationError`). Never mutates/aliases the
    normalization inputs; inherits B1 ``active_frames``/exclusions;
    prior-inactive frames stay inactive (q=0, metrics NaN). One scalar per frame
    shared by all channels. See module docstring for the full per-method
    contract.
    """
    if not isinstance(method, str):
        raise CanonicalStackValidationError(
            f"weighting method must be a string, got {type(method).__name__}"
        )
    token = method.strip().lower()
    if token not in _SUPPORTED_WEIGHT_METHODS:
        raise CanonicalStackValidationError(
            f"unsupported weighting method {method!r}; expected one of "
            "none/noise_variance/noise_fwhm (aliases are not accepted)"
        )
    if token == _METHOD_WEIGHT_NOISE_FWHM and not canonical_noise_fwhm_available():
        raise CanonicalStackValidationError(
            "weighting method noise_fwhm unavailable: Photutils source "
            "detection/catalog support is missing or lacks SourceCatalog.fwhm"
        )

    n = normalization.n_frames
    active = np.array(normalization.active_frames, dtype=bool, copy=True)
    ref_idx = int(normalization.reference_index)

    weights = np.zeros(n, dtype=np.float64)
    raw_weights = np.full(n, np.nan, dtype=np.float64)
    noise_sigma = np.full(n, np.nan, dtype=np.float64)
    fwhm = np.full(n, np.nan, dtype=np.float64)
    new_exclusions = []

    if token == _METHOD_WEIGHT_NONE:
        weights[active] = 1.0
        raw_weights[active] = 1.0
    else:
        for i in range(n):
            if not active[i]:
                continue  # prior inactive: q=0, raw/metrics NaN (already set)

            lum = _frame_luminance(normalization.images[i])
            valid = normalization.valid_mask[i]

            median, sigma, reason = _noise_stats(lum, valid)
            fwhm_val = np.nan
            if reason is None and token == _METHOD_WEIGHT_NOISE_FWHM:
                fwhm_val, reason = _frame_fwhm(lum, valid, median, sigma)

            if reason is not None:
                if i == ref_idx:
                    raise CanonicalStackFailure(
                        f"reference frame {i} quality metric failed "
                        f"(stage=weighting, reason={reason})"
                    )
                active[i] = False
                new_exclusions.append(
                    FrameExclusion(index=i, stage=STAGE_WEIGHTING, reason=reason)
                )
                continue

            if token == _METHOD_WEIGHT_NOISE_VARIANCE:
                with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                    raw = float(1.0 / (np.float64(sigma) * np.float64(sigma)))
            else:
                with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                    raw = float(
                        1.0
                        / (
                            np.float64(sigma)
                            * np.float64(sigma)
                            * np.float64(fwhm_val)
                            * np.float64(fwhm_val)
                        )
                    )
            if not (np.isfinite(raw) and raw > 0.0):
                if i == ref_idx:
                    raise CanonicalStackFailure(
                        f"reference frame {i} quality metric failed "
                        f"(stage=weighting, reason={REASON_RAW_WEIGHT_FAILED})"
                    )
                active[i] = False
                new_exclusions.append(
                    FrameExclusion(
                        index=i, stage=STAGE_WEIGHTING, reason=REASON_RAW_WEIGHT_FAILED
                    )
                )
                continue

            noise_sigma[i] = sigma
            fwhm[i] = fwhm_val
            raw_weights[i] = raw

        if not active.any():
            raise CanonicalStackFailure("no frame remains after quality weighting")
        max_raw = float(np.max(raw_weights[active]))
        if not (np.isfinite(max_raw) and max_raw > 0.0):
            raise CanonicalStackFailure("no positive quality weight remains after weighting")
        weights[active] = raw_weights[active] / max_raw

    return CanonicalWeightingResult(
        weights=np.ascontiguousarray(weights),
        raw_weights=np.ascontiguousarray(raw_weights),
        noise_sigma=np.ascontiguousarray(noise_sigma),
        fwhm=np.ascontiguousarray(fwhm),
        active_frames=np.ascontiguousarray(active),
        reference_index=ref_idx,
        requested_method=token,
        effective_method=token,
        exclusions=tuple(normalization.exclusions) + tuple(new_exclusions),
        original_mono=normalization.original_mono,
        n_frames=n,
        height=normalization.height,
        width=normalization.width,
        channels=normalization.channels,
        original_ndim=normalization.original_ndim,
        original_shape=normalization.original_shape,
    )


# ---------------------------------------------------------------------------
# Gate C1: canonical rejection primitives
# ---------------------------------------------------------------------------

_METHOD_REJECT_NONE = "none"
_METHOD_REJECT_KAPPA_SIGMA = "kappa_sigma"
_METHOD_REJECT_WSC = "winsorized_sigma_clip"
_SUPPORTED_REJECT_METHODS = (
    _METHOD_REJECT_NONE,
    _METHOD_REJECT_KAPPA_SIGMA,
    _METHOD_REJECT_WSC,
)
_REJECT_TOKEN_REMOVED = "unsupported_removed_sci05"

_REJECT_DEFAULT_SIGMA_LOW = 3.0
_REJECT_DEFAULT_SIGMA_HIGH = 3.0
_REJECT_DEFAULT_MAX_ITERS = 5
_REJECT_DEFAULT_WINSOR_LOW = 0.05
_REJECT_DEFAULT_WINSOR_HIGH = 0.05


@dataclass(frozen=True)
class CanonicalRejectionResult:
    """Deterministic output of :func:`reject_canonical_samples`.

    Attributes
    ----------
    survivor_mask:
        Owned contiguous ``(N, H, W, C)`` bool; True only for original
        normalized valid samples from B2-active frames that survive rejection.
    rejection_mask:
        Owned contiguous ``(N, H, W, C)`` bool; True exactly for initially-valid
        active samples rejected by the canonical algorithm; never marks prior
        invalid/inactive samples.
    active_frames:
        Owned ``(N,)`` bool inherited from weighting (rejection never globally
        excludes frames).
    reference_index:
        Chosen normalization reference frame index (identity).
    requested_method / effective_method:
        Rejection method token; always equal (validation errors raise instead of
        falling back).
    sigma_low / sigma_high / max_iters / winsor_limit_low / winsor_limit_high:
        Exact validated parameters (winsor limits retained but unused for the
        ``none``/``kappa_sigma`` methods).
    iterations_used:
        ``0`` for ``none``; otherwise the number of loop passes executed.
    initial_sample_count / surviving_sample_count / rejected_sample_count:
        Exact integer sample counts over ``(N, H, W, C)``.
    rejected_fraction:
        ``rejected / initial``, or ``0.0`` when ``initial == 0``.
    low_n_cell_count:
        Number of distinct cells whose INITIAL active-valid count is ``< 3``
        (each cell counted once; includes 0/1/2).
    degenerate_cell_count:
        Number of distinct cells encountering finite ``std <= 0`` in at least one
        executed iteration (counted once, not per iteration; ``0`` for ``none``).
    exclusions:
        Inherited ``FrameExclusion`` records (B1 + B2).
    original_mono / n_frames / height / width / channels / original_ndim /
    original_shape:
        Output shape metadata for downstream restoration (Gate C2).
    """

    survivor_mask: np.ndarray
    rejection_mask: np.ndarray
    active_frames: np.ndarray
    reference_index: int
    requested_method: str
    effective_method: str
    sigma_low: float
    sigma_high: float
    max_iters: int
    winsor_limit_low: float
    winsor_limit_high: float
    iterations_used: int
    initial_sample_count: int
    surviving_sample_count: int
    rejected_sample_count: int
    rejected_fraction: float
    low_n_cell_count: int
    degenerate_cell_count: int
    exclusions: tuple
    original_mono: bool
    n_frames: int
    height: int
    width: int
    channels: int
    original_ndim: int
    original_shape: tuple


def _validate_reject_sigma(value, name: str) -> float:
    if isinstance(value, bool):
        raise CanonicalStackValidationError(f"{name} must be a real number, not bool")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise CanonicalStackValidationError(
            f"{name} must be a real number, got {type(value).__name__}"
        )
    f = float(value)
    if not np.isfinite(f) or not (f > 0.0):
        raise CanonicalStackValidationError(f"{name} must be finite and > 0, got {value!r}")
    return f


def _validate_winsor_limit(value, name: str) -> float:
    if isinstance(value, bool):
        raise CanonicalStackValidationError(f"{name} must be a real number, not bool")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise CanonicalStackValidationError(
            f"{name} must be a real number, got {type(value).__name__}"
        )
    f = float(value)
    if not np.isfinite(f) or not (0.0 <= f < 0.5):
        raise CanonicalStackValidationError(f"{name} must be finite in [0, 0.5), got {value!r}")
    return f


def _validate_max_iters(value) -> int:
    if isinstance(value, bool):
        raise CanonicalStackValidationError("max_iters must be an integer, not bool")
    try:
        i = operator.index(value)
    except TypeError:
        raise CanonicalStackValidationError(
            f"max_iters must be an integer, got {type(value).__name__}"
        )
    if not (1 <= i <= 5):
        raise CanonicalStackValidationError(f"max_iters must be in 1..5, got {i}")
    return i


def _nan_axis_median(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=0)


def _nan_axis_mean(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=0)


def _nan_axis_popstd(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=0)


def _nan_axis_quantile(a, q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanquantile(a, q, axis=0, method="linear")


def _apply_sigma_interval(orig2, center, std, active, sigma_low, sigma_high):
    """Return ``(keep, degenerate)`` for one iteration over the ``N`` axis.

    ``orig2`` is ``(N, M)`` float64 original normalized samples; ``center`` and
    ``std`` are ``(M,)``; ``active`` is ``(M,)`` bool (current N >= 3). For
    active cells ``keep`` is the inclusive asymmetric sigma interval test applied
    to the ORIGINAL samples (degenerate ``std <= 0`` uses the zero-width equality
    test at ``center``); inactive cells are frozen (``keep`` all-True).
    """
    lower = center - sigma_low * std
    upper = center + sigma_high * std
    interval = (orig2 >= lower[np.newaxis, :]) & (orig2 <= upper[np.newaxis, :])
    degenerate = (std <= 0.0) & active
    equality = orig2 == center[np.newaxis, :]
    keep = np.where(degenerate[np.newaxis, :], equality, interval)
    keep = np.where(active[np.newaxis, :], keep, True)
    return keep, degenerate


def _kappa_sigma_step(orig2, survivor2, count, sigma_low, sigma_high):
    active = count >= 3
    masked = np.where(survivor2, orig2, np.nan)
    center = _nan_axis_median(masked)
    std = _nan_axis_popstd(masked)
    keep, degenerate = _apply_sigma_interval(orig2, center, std, active, sigma_low, sigma_high)
    return survivor2 & keep, degenerate


def _winsorized_sigma_clip_step(
    orig2, survivor2, count, sigma_low, sigma_high, winsor_low, winsor_high
):
    active = count >= 3
    masked = np.where(survivor2, orig2, np.nan)
    q_low = _nan_axis_quantile(masked, winsor_low)
    q_high = _nan_axis_quantile(masked, 1.0 - winsor_high)
    winsor = np.clip(masked, q_low[np.newaxis, :], q_high[np.newaxis, :])
    center = _nan_axis_mean(winsor)
    std = _nan_axis_popstd(winsor)
    keep, degenerate = _apply_sigma_interval(orig2, center, std, active, sigma_low, sigma_high)
    return survivor2 & keep, degenerate


def reject_canonical_samples(
    normalization,
    weighting,
    method,
    *,
    sigma_low: float = _REJECT_DEFAULT_SIGMA_LOW,
    sigma_high: float = _REJECT_DEFAULT_SIGMA_HIGH,
    max_iters: int = _REJECT_DEFAULT_MAX_ITERS,
    winsor_limit_low: float = _REJECT_DEFAULT_WINSOR_LOW,
    winsor_limit_high: float = _REJECT_DEFAULT_WINSOR_HIGH,
) -> CanonicalRejectionResult:
    """Compute canonical outlier-rejection masks on a normalized/weighted batch.

    ``method`` must be exactly ``none``, ``kappa_sigma``, or
    ``winsorized_sigma_clip`` (after strip/lower; aliases and unknown tokens are
    rejected with :class:`CanonicalStackValidationError`; the removed token
    ``linear_fit_clip`` fails with the stable token
    ``unsupported_removed_sci05`` and is never migrated/substituted).

    Never mutates/aliases the normalization/weighting inputs. Weight values,
    magnitudes, and raw metrics do **not** enter any rejection calculation: only
    B2 ``active_frames`` gates absent frames. Rejection is per pixel/channel
    along the ``N`` axis; statistics are float64; survivor updates are monotonic
    intersections over the original normalized samples; a cell with fewer than 3
    current survivors is frozen (explicit low-N no-rejection success).
    """
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
    if token not in _SUPPORTED_REJECT_METHODS:
        raise CanonicalStackValidationError(
            f"unsupported rejection method {method!r}; expected one of "
            "none/kappa_sigma/winsorized_sigma_clip (aliases are not accepted)"
        )

    s_low = _validate_reject_sigma(sigma_low, "sigma_low")
    s_high = _validate_reject_sigma(sigma_high, "sigma_high")
    iters = _validate_max_iters(max_iters)
    w_low = _validate_winsor_limit(winsor_limit_low, "winsor_limit_low")
    w_high = _validate_winsor_limit(winsor_limit_high, "winsor_limit_high")
    if w_low + w_high >= 1.0:
        raise CanonicalStackValidationError(
            f"winsor_limit_low + winsor_limit_high must be < 1, got {w_low} + {w_high}"
        )

    if not isinstance(normalization, CanonicalNormalizationResult):
        raise CanonicalStackValidationError(
            f"normalization must be a CanonicalNormalizationResult, got "
            f"{type(normalization).__name__}"
        )
    if not isinstance(weighting, CanonicalWeightingResult):
        raise CanonicalStackValidationError(
            f"weighting must be a CanonicalWeightingResult, got {type(weighting).__name__}"
        )

    n = normalization.n_frames
    h = normalization.height
    w = normalization.width
    c = normalization.channels

    if weighting.n_frames != n:
        raise CanonicalStackValidationError(
            f"normalization/weighting N mismatch ({n} vs {weighting.n_frames})"
        )
    if (weighting.height, weighting.width, weighting.channels) != (h, w, c):
        raise CanonicalStackValidationError(
            "normalization/weighting H/W/C mismatch"
        )
    if weighting.original_mono != normalization.original_mono:
        raise CanonicalStackValidationError(
            "normalization/weighting original_mono mismatch"
        )
    if weighting.original_shape != normalization.original_shape:
        raise CanonicalStackValidationError(
            "normalization/weighting original_shape mismatch"
        )
    if weighting.reference_index != normalization.reference_index:
        raise CanonicalStackValidationError(
            "normalization/weighting reference_index mismatch"
        )
    if np.asarray(weighting.active_frames).shape != (n,):
        raise CanonicalStackValidationError(
            f"weighting.active_frames must have length {n}"
        )
    if np.asarray(normalization.images).shape != (n, h, w, c):
        raise CanonicalStackValidationError("normalization.images shape mismatch")
    if np.asarray(normalization.valid_mask).shape != (n, h, w):
        raise CanonicalStackValidationError("normalization.valid_mask shape mismatch")

    active_frames = np.array(weighting.active_frames, dtype=bool, copy=True)
    images64 = np.array(normalization.images, dtype=np.float64, copy=True)
    valid = np.asarray(normalization.valid_mask, dtype=bool)

    finite = np.isfinite(images64)
    initial = active_frames[:, None, None, None] & valid[..., None] & finite

    n_cells = h * w * c

    if token == _METHOD_REJECT_NONE:
        survivor = initial.copy()
        iterations_used = 0
        degenerate_seen = np.zeros(n_cells, dtype=bool)
    else:
        orig2 = images64.reshape(n, n_cells)
        survivor2 = initial.reshape(n, n_cells)
        degenerate_seen = np.zeros(n_cells, dtype=bool)
        iterations_used = 0
        for _ in range(iters):
            count = survivor2.sum(axis=0)
            if token == _METHOD_REJECT_KAPPA_SIGMA:
                new_survivor2, degenerate = _kappa_sigma_step(
                    orig2, survivor2, count, s_low, s_high
                )
            else:
                new_survivor2, degenerate = _winsorized_sigma_clip_step(
                    orig2, survivor2, count, s_low, s_high, w_low, w_high
                )
            iterations_used += 1
            degenerate_seen |= degenerate
            if np.array_equal(new_survivor2, survivor2):
                survivor2 = new_survivor2
                break
            survivor2 = new_survivor2
        survivor = survivor2.reshape(n, h, w, c)

    rejection = initial & ~survivor

    initial_count = int(initial.sum())
    surviving_count = int(survivor.sum())
    rejected_count = int(rejection.sum())
    rejected_fraction = (rejected_count / initial_count) if initial_count > 0 else 0.0
    low_n_cell_count = int((initial.sum(axis=0) < 3).sum())
    degenerate_cell_count = int(degenerate_seen.sum())

    return CanonicalRejectionResult(
        survivor_mask=np.array(survivor, dtype=bool, copy=True),
        rejection_mask=np.array(rejection, dtype=bool, copy=True),
        active_frames=np.array(active_frames, dtype=bool, copy=True),
        reference_index=int(normalization.reference_index),
        requested_method=token,
        effective_method=token,
        sigma_low=s_low,
        sigma_high=s_high,
        max_iters=iters,
        winsor_limit_low=w_low,
        winsor_limit_high=w_high,
        iterations_used=iterations_used,
        initial_sample_count=initial_count,
        surviving_sample_count=surviving_count,
        rejected_sample_count=rejected_count,
        rejected_fraction=rejected_fraction,
        low_n_cell_count=low_n_cell_count,
        degenerate_cell_count=degenerate_cell_count,
        exclusions=tuple(weighting.exclusions),
        original_mono=normalization.original_mono,
        n_frames=n,
        height=h,
        width=w,
        channels=c,
        original_ndim=normalization.original_ndim,
        original_shape=normalization.original_shape,
    )


# ---------------------------------------------------------------------------
# Gate C2: canonical combine primitive (mean / median)
# ---------------------------------------------------------------------------

_METHOD_COMBINE_MEAN = "mean"
_METHOD_COMBINE_MEDIAN = "median"
_SUPPORTED_COMBINE_METHODS = (_METHOD_COMBINE_MEAN, _METHOD_COMBINE_MEDIAN)


@dataclass(frozen=True)
class CanonicalCombineResult:
    """Deterministic output of :func:`combine_canonical_samples`.

    Attributes
    ----------
    science:
        Owned contiguous float32; restored original shape (HW mono, HWC1, RGB
        HWC). No-estimate cells are NaN; valid cells are never Inf/NaN.
    estimator_weight_sum:
        Owned contiguous float64 with the **same shape as ``science``**. For
        ``mean`` it is the per-pixel/channel sum of the canonical estimator
        weights ``w_i`` over original surviving samples; for ``median`` it is
        the per-pixel/channel **count** of eligible originals (unit effective
        estimator weights, not ``Σ q·m·a``); ``0`` where no survivors.
    valid_mask:
        Owned contiguous bool with the same shape as ``science``; exactly
        ``estimator_weight_sum > 0``.
    surviving_sample_count:
        Owned contiguous int64 with the same shape as ``science``; count of
        original survivors with ``w_i > 0`` (for mean and median diagnostics;
        for mean the estimator-weight sum can differ).
    requested_method / effective_method:
        Combine method token (``mean``/``median``); always equal (validation
        errors raise instead of falling back).
    reference_index:
        Chosen normalization reference frame index (identity), inherited from
        normalization.
    normalization_method / weighting_method / rejection_method:
        Inherited stage method tokens (bounded metadata for the final result).
    sigma_low / sigma_high / max_iters / winsor_limit_low / winsor_limit_high:
        Inherited validated rejection parameters.
    iterations_used / initial_sample_count / rejected_sample_count /
    rejected_fraction / low_n_cell_count / degenerate_cell_count:
        Inherited C1 rejection scalar diagnostics.
    input_surviving_sample_count:
        Python int: original survivors **before** the positive-weight gate
        (equals ``rejection.surviving_sample_count``).
    contributing_sample_count:
        Python int: eligible samples **after** the positive-weight gate.
    valid_output_count / invalid_output_count:
        Python ints: number of output channel cells with/without an estimate
        (invalid includes every no-estimate cell).
    nonfinite_output_count:
        Python int: number of output cells invalidated because a mathematically
        valid estimate became nonfinite (float64 or float32 post-cast). Defensive
        seam: with finite float32 inputs and weights in ``[0, 1]`` the mean and
        median are convex combinations of finite values and stay finite, so this
        is ``0`` in practice (see the combine function docstring).
    exclusions:
        Inherited ``FrameExclusion`` records (B1 + B2).
    original_mono / n_frames / height / width / channels / original_ndim /
    original_shape:
        Original shape metadata for downstream final-result assembly.
    """

    science: np.ndarray
    estimator_weight_sum: np.ndarray
    valid_mask: np.ndarray
    surviving_sample_count: np.ndarray
    requested_method: str
    effective_method: str
    reference_index: int
    normalization_method: str
    weighting_method: str
    rejection_method: str
    sigma_low: float
    sigma_high: float
    max_iters: int
    winsor_limit_low: float
    winsor_limit_high: float
    iterations_used: int
    initial_sample_count: int
    rejected_sample_count: int
    rejected_fraction: float
    low_n_cell_count: int
    degenerate_cell_count: int
    input_surviving_sample_count: int
    contributing_sample_count: int
    valid_output_count: int
    invalid_output_count: int
    nonfinite_output_count: int
    exclusions: tuple
    original_mono: bool
    n_frames: int
    height: int
    width: int
    channels: int
    original_ndim: int
    original_shape: tuple


def _validate_estimator_weights(estimator_weights, n: int, h: int, w: int) -> np.ndarray:
    """Validate the explicit pre-rejection ``(N, H, W)`` estimator-weight map.

    Returns an owned contiguous float64 copy. Rejects non-array, non-real-numeric
    (bool/object/complex), wrong ndim/shape, nonfinite, and out-of-``[0, 1]``
    values **before** any copy/mutation of the caller's array.
    """
    try:
        arr = np.asarray(estimator_weights)
    except Exception as exc:  # pragma: no cover - defensive
        raise CanonicalStackValidationError(
            f"estimator_weights is not array-convertible: {exc}"
        )
    if arr.dtype == np.bool_ or arr.dtype.kind not in "iuf":
        raise CanonicalStackValidationError(
            f"estimator_weights must be real numeric (int/uint/float), not "
            f"bool/object/complex; got dtype {arr.dtype}"
        )
    if arr.ndim != 3:
        raise CanonicalStackValidationError(
            f"estimator_weights must be 3-D (N, H, W), got shape {arr.shape}"
        )
    if arr.shape != (n, h, w):
        raise CanonicalStackValidationError(
            f"estimator_weights shape {arr.shape} != (N, H, W) {(n, h, w)}"
        )
    out = np.array(arr, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(out)):
        raise CanonicalStackValidationError("estimator_weights must be finite")
    if np.any(out < 0.0) or np.any(out > 1.0):
        raise CanonicalStackValidationError("estimator_weights must be in [0, 1]")
    return out


def combine_canonical_samples(normalization, weighting, rejection, estimator_weights, method):
    """Combine canonical samples into the final science/estimate arrays.

    Pure, deterministic, backend-neutral combine primitive (Gate C2). It consumes
    the B1 normalized originals, the B2 scalar quality weights, the C1 survivor/
    rejection masks, and an **explicit** pre-rejection 2-D canonical
    estimator-weight map ``w_i = q_i * m_i * a_i`` per exposure (``(N, H, W)``
    float64). There is **no** optional/default weight map and **no** invented
    ``a_i = 1``: the frozen default footprint taper is ON, and Gate E owns taper/
    support construction, so this stage requires the explicit map.

    ``method`` must be exactly ``mean`` or ``median`` (after strip/lower;
    aliases/unknown/non-string rejected).

    Semantics
    ---------
    * Eligible per-channel sample = ``rejection.survivor_mask`` AND
      ``estimator_weights > 0`` AND finite original. Original normalized values
      only (never winsorized replacements); weight magnitude never changes median
      values (rejection is already weight-independent).
    * ``mean``: float64 ``Σ(original * w_i) / Σ(w_i)`` per pixel/channel; the
      denominator condition is exactly ``> 0`` (no epsilon); ``denom <= 0`` →
      NaN science, sum ``0``, valid False.
    * ``median``: float64 unweighted ``median`` of eligible originals (even-N
      NumPy convention averages the middle two); no survivors → NaN science,
      sum ``0``, valid False. ``estimator_weight_sum`` is the **count** of
      eligible originals (unit effective estimator weights), not ``Σ q·m·a``.
    * Output ``science`` is float32; ``estimator_weight_sum`` float64;
      ``valid_mask`` bool (exactly ``estimator_weight_sum > 0``);
      ``surviving_sample_count`` int64 (count of ``w_i > 0`` survivors).
    * Post-cast invariant: no NaN/Inf is ever marked valid. Defensive seam —
      with finite float32 inputs and weights in ``[0, 1]``, both the weighted
      mean and the median are convex combinations of finite values and therefore
      stay finite in float64 and in float32, so the ``nonfinite_output_count``
      branch is unreachable for well-formed inputs. It is kept (no clip/saturate)
      so any genuinely nonfinite estimate invalidates its cell deterministically.

    Validation (all **before** any mutation):
    * exact result types; B1↔B2↔C1 agreement of N/H/W/C, original_mono/ndim/
      shape, reference index, active_frames (C1 exactly equals B2; B2 is a
      subset of B1), and exclusions (C1 exactly equals B2; B1 is a prefix of B2);
    * C1 masks are exact ``(N, H, W, C)`` bool, survivor/rejection subsets of
      active+valid+finite, disjoint, and ``survivor | rejection == initial``;
    * ``weighting.weights`` exact ``(N,)``, finite in ``[0, 1]``, active strictly
      > 0, inactive exactly 0;
    * ``estimator_weights`` exact ``(N, H, W)`` real numeric, finite in
      ``[0, 1]``; zero on inactive frames, zero outside ``valid_mask``, and
      ``<= weighting.weights[i]`` (canonical ``w = q*m*a``); zero on a valid
      sample (absent) is allowed.
    """
    if not isinstance(method, str):
        raise CanonicalStackValidationError(
            f"combine method must be a string, got {type(method).__name__}"
        )
    token = method.strip().lower()
    if token not in _SUPPORTED_COMBINE_METHODS:
        raise CanonicalStackValidationError(
            f"unsupported combine method {method!r}; expected one of mean/median "
            "(aliases are not accepted)"
        )

    if not isinstance(normalization, CanonicalNormalizationResult):
        raise CanonicalStackValidationError(
            f"normalization must be a CanonicalNormalizationResult, got "
            f"{type(normalization).__name__}"
        )
    if not isinstance(weighting, CanonicalWeightingResult):
        raise CanonicalStackValidationError(
            f"weighting must be a CanonicalWeightingResult, got "
            f"{type(weighting).__name__}"
        )
    if not isinstance(rejection, CanonicalRejectionResult):
        raise CanonicalStackValidationError(
            f"rejection must be a CanonicalRejectionResult, got "
            f"{type(rejection).__name__}"
        )

    n = normalization.n_frames
    h = normalization.height
    w = normalization.width
    c = normalization.channels

    # --- B1 <-> B2 <-> C1 agreement ---
    if weighting.n_frames != n or rejection.n_frames != n:
        raise CanonicalStackValidationError(
            "normalization/weighting/rejection N mismatch"
        )
    if (weighting.height, weighting.width, weighting.channels) != (h, w, c):
        raise CanonicalStackValidationError("normalization/weighting H/W/C mismatch")
    if (rejection.height, rejection.width, rejection.channels) != (h, w, c):
        raise CanonicalStackValidationError("normalization/rejection H/W/C mismatch")
    if weighting.original_mono != normalization.original_mono:
        raise CanonicalStackValidationError("normalization/weighting original_mono mismatch")
    if rejection.original_mono != normalization.original_mono:
        raise CanonicalStackValidationError("normalization/rejection original_mono mismatch")
    if (
        weighting.original_ndim != normalization.original_ndim
        or rejection.original_ndim != normalization.original_ndim
    ):
        raise CanonicalStackValidationError("original_ndim mismatch")
    if (
        weighting.original_shape != normalization.original_shape
        or rejection.original_shape != normalization.original_shape
    ):
        raise CanonicalStackValidationError("original_shape mismatch")
    if (
        weighting.reference_index != normalization.reference_index
        or rejection.reference_index != normalization.reference_index
    ):
        raise CanonicalStackValidationError("reference_index mismatch")

    active = np.asarray(weighting.active_frames, dtype=bool)
    if active.shape != (n,):
        raise CanonicalStackValidationError(
            f"weighting.active_frames must have length {n}"
        )
    if not np.array_equal(active, np.asarray(rejection.active_frames, dtype=bool)):
        raise CanonicalStackValidationError("weighting/rejection active_frames mismatch")
    norm_active = np.asarray(normalization.active_frames, dtype=bool)
    if norm_active.shape != (n,):
        raise CanonicalStackValidationError(
            f"normalization.active_frames must have length {n}"
        )
    if not np.all(active <= norm_active):
        raise CanonicalStackValidationError(
            "weighting reactivated a frame that normalization excluded"
        )

    if tuple(rejection.exclusions) != tuple(weighting.exclusions):
        raise CanonicalStackValidationError("weighting/rejection exclusions mismatch")
    b1_excl = tuple(normalization.exclusions)
    b2_excl = tuple(weighting.exclusions)
    if b2_excl[: len(b1_excl)] != b1_excl:
        raise CanonicalStackValidationError(
            "normalization exclusions are not a prefix of weighting exclusions"
        )

    # --- array shape / dtype guards ---
    images = np.asarray(normalization.images)
    valid = np.asarray(normalization.valid_mask)
    survivor = np.asarray(rejection.survivor_mask)
    rejection_mask = np.asarray(rejection.rejection_mask)
    if images.shape != (n, h, w, c):
        raise CanonicalStackValidationError("normalization.images shape mismatch")
    if valid.shape != (n, h, w):
        raise CanonicalStackValidationError("normalization.valid_mask shape mismatch")
    if survivor.shape != (n, h, w, c):
        raise CanonicalStackValidationError("rejection.survivor_mask shape mismatch")
    if rejection_mask.shape != (n, h, w, c):
        raise CanonicalStackValidationError("rejection.rejection_mask shape mismatch")
    if survivor.dtype != np.bool_ or rejection_mask.dtype != np.bool_:
        raise CanonicalStackValidationError("rejection masks must be bool dtype")

    # --- weighting.weights validation ---
    weights = np.asarray(weighting.weights)
    if weights.shape != (n,):
        raise CanonicalStackValidationError(f"weighting.weights must have length {n}")
    if weights.dtype == np.bool_ or weights.dtype.kind not in "iuf":
        raise CanonicalStackValidationError(
            f"weighting.weights must be real numeric, got dtype {weights.dtype}"
        )
    if not np.all(np.isfinite(weights)):
        raise CanonicalStackValidationError("weighting.weights must be finite")
    if np.any(weights < 0.0) or np.any(weights > 1.0):
        raise CanonicalStackValidationError("weighting.weights must be in [0, 1]")
    if np.any(weights[active] <= 0.0):
        raise CanonicalStackValidationError(
            "active frames must have strictly positive quality weight"
        )
    if np.any(weights[~active] != 0.0):
        raise CanonicalStackValidationError(
            "inactive frames must have exactly zero quality weight"
        )

    # --- estimator weights + canonical w = q*m*a ---
    wmap = _validate_estimator_weights(estimator_weights, n, h, w)
    if np.any(wmap[~active] != 0.0):
        raise CanonicalStackValidationError(
            "estimator_weights must be zero for inactive frames"
        )
    if np.any(wmap[~valid] != 0.0):
        raise CanonicalStackValidationError(
            "estimator_weights must be zero outside the valid mask"
        )
    if np.any(wmap > weights[:, None, None]):
        raise CanonicalStackValidationError(
            "estimator_weights exceed the quality weight q_i (canonical w=q*m*a)"
        )

    # --- C1 mask invariants ---
    images64 = np.array(images, dtype=np.float64, copy=True)
    finite = np.isfinite(images64)
    initial = active[:, None, None, None] & valid[..., None] & finite
    if np.any(survivor & ~initial):
        raise CanonicalStackValidationError(
            "survivor_mask is not a subset of active/valid/finite"
        )
    if np.any(rejection_mask & ~initial):
        raise CanonicalStackValidationError(
            "rejection_mask is not a subset of active/valid/finite"
        )
    if np.any(survivor & rejection_mask):
        raise CanonicalStackValidationError("survivor_mask and rejection_mask overlap")
    if not np.array_equal(survivor | rejection_mask, initial):
        raise CanonicalStackValidationError(
            "survivor | rejection != initial active-valid samples"
        )

    # --- combine ---
    w_positive = wmap[:, :, :, None] > 0.0  # (N, H, W, 1) broadcasts to channels
    eligible = survivor & w_positive & finite  # (N, H, W, C)

    input_surviving = int(survivor.sum())
    contributing = int(eligible.sum())

    if token == _METHOD_COMBINE_MEAN:
        masked_images = np.where(eligible, images64, 0.0)
        w_contrib = np.where(eligible, wmap[:, :, :, None], 0.0)
        numerator = np.sum(masked_images * w_contrib, axis=0, dtype=np.float64)
        denominator = np.sum(w_contrib, axis=0, dtype=np.float64)
        den_valid = denominator > 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
                estimate64 = numerator / denominator
                science32 = estimate64.astype(np.float32)
        final_finite = np.isfinite(estimate64) & np.isfinite(science32)
        valid_out = den_valid & final_finite
        science_c = np.where(valid_out, science32, np.nan)
        weight_sum = np.where(valid_out, denominator, 0.0)
        nonfinite_count = int((den_valid & ~valid_out).sum())
    else:  # median
        masked = np.where(eligible, images64, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median64 = np.nanmedian(masked, axis=0)
            with np.errstate(over="ignore", invalid="ignore"):
                science32 = median64.astype(np.float32)
        count = eligible.sum(axis=0).astype(np.float64)
        den_valid = count > 0.0
        final_finite = np.isfinite(median64) & np.isfinite(science32)
        valid_out = den_valid & final_finite
        science_c = np.where(valid_out, science32, np.nan)
        weight_sum = np.where(valid_out, count, 0.0)
        nonfinite_count = int((den_valid & ~valid_out).sum())

    surviving_c = eligible.sum(axis=0).astype(np.int64)
    valid_mask_c = weight_sum > 0.0

    # --- restore original shape (strip canonical C=1 for original mono) ---
    if normalization.original_mono:
        # ``arr[..., 0]`` is a contiguous view; ``.copy()`` guarantees an owned
        # C-contiguous array (``np.ascontiguousarray`` would return the view).
        science = science_c[..., 0].copy()
        weight_sum = weight_sum[..., 0].copy()
        valid_mask = valid_mask_c[..., 0].copy()
        surviving = surviving_c[..., 0].copy()
    else:
        science = np.ascontiguousarray(science_c)
        weight_sum = np.ascontiguousarray(weight_sum)
        valid_mask = np.ascontiguousarray(valid_mask_c)
        surviving = np.ascontiguousarray(surviving_c)

    valid_output_count = int(valid_mask_c.sum())
    invalid_output_count = (h * w * c) - valid_output_count

    return CanonicalCombineResult(
        science=science,
        estimator_weight_sum=weight_sum,
        valid_mask=valid_mask,
        surviving_sample_count=surviving,
        requested_method=token,
        effective_method=token,
        reference_index=int(normalization.reference_index),
        normalization_method=normalization.requested_method,
        weighting_method=weighting.requested_method,
        rejection_method=rejection.requested_method,
        sigma_low=rejection.sigma_low,
        sigma_high=rejection.sigma_high,
        max_iters=rejection.max_iters,
        winsor_limit_low=rejection.winsor_limit_low,
        winsor_limit_high=rejection.winsor_limit_high,
        iterations_used=rejection.iterations_used,
        initial_sample_count=rejection.initial_sample_count,
        rejected_sample_count=rejection.rejected_sample_count,
        rejected_fraction=rejection.rejected_fraction,
        low_n_cell_count=rejection.low_n_cell_count,
        degenerate_cell_count=rejection.degenerate_cell_count,
        input_surviving_sample_count=input_surviving,
        contributing_sample_count=contributing,
        valid_output_count=valid_output_count,
        invalid_output_count=invalid_output_count,
        nonfinite_output_count=nonfinite_count,
        exclusions=tuple(rejection.exclusions),
        original_mono=normalization.original_mono,
        n_frames=n,
        height=h,
        width=w,
        channels=c,
        original_ndim=normalization.original_ndim,
        original_shape=normalization.original_shape,
    )
