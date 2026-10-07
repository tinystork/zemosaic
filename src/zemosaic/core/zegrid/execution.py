"""ZM-ZEGRID-R1 — execution: local section reads, local WCS reprojection.

Isolated R1 module. No production dispatch wiring.

Responsibilities:
* Read prepared RGB source sections locally (``hdu.section`` / true memmap
  slice), axis-aware, never a full-frame ``.data`` read.
* Reproject the cropped source to the EXACT patch shape with explicit
  interpolation order and validity convention.
* Carry an explicit geometric validity/support map (for prepared RGB with no
  ALPHA, support = the reprojected geometric source domain, not brightness).

Prepared RGB FITS contract (produced by ``tools/zegrid_r1/prepare_rgb_fixture.py``):
* Primary HDU, float32, ``NAXIS=3``, ``NAXIS1=W``, ``NAXIS2=H``, ``NAXIS3=3``,
  i.e. numpy shape ``(C, H, W)`` = ``(3, 1920, 1080)`` (channels-first on disk).
* The raw 2-D celestial WCS is preserved on axes 1-2; ``WCS(header).celestial``
  yields the exact 2-D WCS.
* Logical RGB is ``H x W x 3 = 1920 x 1080 x 3``; the reader transposes CHW -> HWC.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel
from reproject import reproject_interp
from scipy.ndimage import map_coordinates as _scipy_map_coordinates

from .geometry import (
    FrameDescriptor,
    GlobalCanvas,
    ProcessingPatch,
    SourceBounds,
    SourceCropPlan,
)

# Interpolation order used for local reprojection (bilinear). Explicit.
REPROJECT_ORDER = "bilinear"

# Sentinel used outside the geometric support domain.
_INVALID = np.nan

# ---------------------------------------------------------------------------
# ZM-ZEGRID-R21 — fast reprojection path (bit-exact to ``reproject_interp``)
# ---------------------------------------------------------------------------
#
# ``reproject_interp`` is the dominant CPU cost of the engine. For a plain
# (undistorted) 2-D celestial TAN WCS its work decomposes into
#   1. output-grid -> world   (``all_pix2world``)
#   2. world -> input-pixel   (``all_world2pix``)
#   3. bilinear interpolation (``scipy.ndimage.map_coordinates``, order=1)
# and the mapping (1+2) is channel-independent, so it can be computed ONCE and
# reused across the 3 RGB channels.  The result is BIT-IDENTICAL to
# ``reproject_interp`` on undistorted TAN (verified by the R21 test matrix)
# when the input is fed to ``map_coordinates`` as float64 and the border/NaN
# policy below is reproduced exactly.  For SIP/distorted/non-TAN WCS we FALL
# BACK to ``reproject_interp`` (never silently change the science).

_LOGGER = logging.getLogger(__name__)

# A/B test hook: set ``ZEGRID_REPROJECT_FORCE_FALLBACK=1`` to force the fast path
# OFF (every reprojection uses ``reproject_interp``).  Read at import so it is
# inherited by forked/spawned workers.  Default (unset) = fast path enabled.
_REPROJECT_FORCE_FALLBACK = os.environ.get("ZEGRID_REPROJECT_FORCE_FALLBACK", "") in (
    "1", "true", "yes", "on",
)

# Reasons a reprojection took the fallback path (recorded, surfaced once).
_FALLBACK_REASON_NON_TAN = (
    "non-TAN/distorted WCS (fast path is restricted to undistorted RA---TAN/DEC--TAN)"
)
_FALLBACK_REASON_FORCED = "fast path disabled (ZEGRID_REPROJECT_FORCE_FALLBACK)"

# Process-local "warned once" flag so the fallback WARN is not spammed per call.
_FALLBACK_WARNED = False


class ReprojectPathStats:
    """Per-process accumulator for the reprojection path actually used.

    ``fast`` = the R21 fast path (map_coordinates); ``fallback`` = the
    ``reproject_interp`` path.  One instance lives per process (module
    singleton); the production orchestrator aggregates it across the main
    process and the per-cell/gauge worker processes.
    """

    def __init__(self) -> None:
        self.fast_calls = 0
        self.fallback_calls = 0
        self.fast_seconds = 0.0
        self.fallback_seconds = 0.0
        self.last_fallback_reason: str | None = None

    def record_fast(self, seconds: float) -> None:
        self.fast_calls += 1
        self.fast_seconds += float(seconds)

    def record_fallback(self, seconds: float, reason: str | None = None) -> None:
        self.fallback_calls += 1
        self.fallback_seconds += float(seconds)
        if reason is not None:
            self.last_fallback_reason = reason

    def to_dict(self) -> dict:
        if self.fast_calls == 0 and self.fallback_calls == 0:
            path = "unused"
        elif self.fallback_calls == 0:
            path = "fast"
        elif self.fast_calls == 0:
            path = "fallback"
        else:
            path = "mixed"
        return {
            "path": path,
            "fast_calls": int(self.fast_calls),
            "fallback_calls": int(self.fallback_calls),
            "fast_seconds": round(self.fast_seconds, 6),
            "fallback_seconds": round(self.fallback_seconds, 6),
            "fallback_reason": self.last_fallback_reason,
        }


_REPROJECT_STATS = ReprojectPathStats()


def reset_reproject_path_stats() -> None:
    """Zero the process-local reprojection-path accumulator."""
    global _REPROJECT_STATS, _FALLBACK_WARNED
    _REPROJECT_STATS = ReprojectPathStats()
    _FALLBACK_WARNED = False


def get_reproject_path_stats() -> ReprojectPathStats:
    """Return the process-local reprojection-path accumulator."""
    return _REPROJECT_STATS


def merge_reproject_stats(*dicts) -> dict:
    """Sum reproject-path stat dicts (from several processes) into one."""
    fast_calls = 0
    fallback_calls = 0
    fast_seconds = 0.0
    fallback_seconds = 0.0
    fallback_reason = None
    for d in dicts:
        if not d:
            continue
        fast_calls += int(d.get("fast_calls", 0))
        fallback_calls += int(d.get("fallback_calls", 0))
        fast_seconds += float(d.get("fast_seconds", 0.0))
        fallback_seconds += float(d.get("fallback_seconds", 0.0))
        if fallback_reason is None and d.get("fallback_reason"):
            fallback_reason = d.get("fallback_reason")
    if fast_calls == 0 and fallback_calls == 0:
        path = "unused"
    elif fallback_calls == 0:
        path = "fast"
    elif fast_calls == 0:
        path = "fallback"
    else:
        path = "mixed"
    return {
        "path": path,
        "fast_calls": int(fast_calls),
        "fallback_calls": int(fallback_calls),
        "fast_seconds": round(fast_seconds, 6),
        "fallback_seconds": round(fallback_seconds, 6),
        "fallback_reason": fallback_reason,
    }


def _warn_fallback(emit, reason: str) -> None:
    """Surface a fallback loudly (log always; emit WARN once per process)."""
    global _FALLBACK_WARNED
    _LOGGER.warning("[ZEGRID] reprojection fallback: %s", reason)
    if not _FALLBACK_WARNED and emit is not None:
        try:
            emit(f"reprojection using FALLBACK path (reproject_interp): {reason}", "WARN")
        except Exception:  # noqa: BLE001 - surfacing is never fatal
            pass
        _FALLBACK_WARNED = True


def _plain_tan_wcs(w: WCS) -> bool:
    """True when ``w`` is a plain undistorted 2-D celestial TAN WCS.

    The fast path is bit-exact to ``reproject_interp`` only for such WCS; any
    SIP / PV / lookup (CPDIS/DET2IM) distortion or non-TAN projection falls
    back.  Defensive: any exception -> not plain TAN (fallback).
    """
    try:
        if w.pixel_n_dim != 2:
            return False
        if not w.has_celestial:
            return False
        if tuple(w.wcs.ctype) != ("RA---TAN", "DEC--TAN"):
            return False
        if w.sip is not None:
            return False
        if w.has_distortion:
            return False
        if w.wcs.get_pv():
            return False
        for attr in ("cpdis1", "cpdis2", "det2im1", "det2im2"):
            if getattr(w, attr, None) is not None:
                return False
        return True
    except Exception:  # noqa: BLE001 - defensive; fall back
        return False


def _clip_coords(shape, coords: np.ndarray) -> np.ndarray:
    """Reproduce ``reproject``'s border-pixel clip (in-place on a copy).

    Maps coords in ``[-0.5, 0)`` to ``0`` and in ``[shape-1, shape-0.5)`` to
    ``shape-1`` so the outer half of a border pixel samples the pixel centre.
    NaN coordinates are left untouched (they are handled as NaN downstream).
    """
    coords = coords.copy()
    for i in range(coords.shape[0]):
        coords[i][(coords[i] < 0) & (coords[i] >= -0.5)] = 0
        coords[i][
            (coords[i] < shape[i] - 0.5) & (coords[i] >= shape[i] - 1)
        ] = shape[i] - 1
    return coords


def _compute_pixel_in(
    cropped_wcs: WCS,
    patch_wcs: WCS,
    patch_shape_hw: tuple[int, int],
) -> np.ndarray:
    """Map the output pixel grid to input-pixel coordinates.

    Returns a float64 array of shape ``(2, H*W)`` where row 0 = input y (rows)
    and row 1 = input x (cols), matching ``scipy.ndimage.map_coordinates``'
    coordinate convention.

    Uses ``astropy.wcs.utils.pixel_to_pixel`` (``pixel_to_world`` +
    ``world_to_pixel``) — the EXACT transform ``reproject_interp`` uses — rather
    than the low-level ``all_pix2world``/``all_world2pix`` pair.  The two agree
    bit-for-bit on a plain CD-matrix TAN WCS, but for a PC+CDELT TAN WCS
    ``world_to_pixel`` (SkyCoord-based) differs from ``all_world2pix`` by ~0.01
    px, so using ``pixel_to_pixel`` is required for bit-equality with
    ``reproject_interp`` on the M16 corpus.  (The ``reproject`` roundtrip NaN
    check is a no-op for undistorted TAN, so it is intentionally omitted.)
    """
    h, w = patch_shape_hw
    x, y = np.meshgrid(
        np.arange(w, dtype=float), np.arange(h, dtype=float), indexing="xy"
    )
    xi, yi = pixel_to_pixel(patch_wcs, cropped_wcs, x.ravel(), y.ravel())
    return np.stack([yi, xi])


def reproject_cropped_fast(
    cropped_chw: np.ndarray,
    cropped_wcs: WCS,
    patch_wcs: WCS,
    patch_shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Fast bilinear reprojection (bit-exact to ``reproject_interp`` on plain TAN).

    Returns ``(h_w_c_rgb_float32, geometric_support_bool)`` with the EXACT same
    semantics as :func:`reproject_cropped`.  Raises on any unexpected error so
    the caller can fall back (this function never swallows an exception).
    """
    c = cropped_chw.shape[0]
    h, w = patch_shape_hw
    # Contiguous float64 planes.  Bit-exactness requires float64: ``reproject``
    # interpolates into a float64 output array, and the float32->float64 upcast
    # is lossless, so feeding float64 to ``map_coordinates`` reproduces it.
    src = np.ascontiguousarray(cropped_chw, dtype=np.float64)  # (C, H_in, W_in)
    with warnings.catch_warnings():
        # Match the pre-R21 path, which silenced ``reproject_interp`` warnings
        # (e.g. TAN-singularity coordinate warnings).
        warnings.simplefilter("ignore")
        coords = _compute_pixel_in(cropped_wcs, patch_wcs, patch_shape_hw)
    coords = _clip_coords(src.shape[1:], coords)
    reset = np.zeros(coords.shape[1], dtype=bool)
    for i in range(coords.shape[0]):
        reset |= coords[i] < -0.5
        reset |= coords[i] > src.shape[1 + i] - 0.5

    out = np.empty((h, w, c), dtype=np.float32)
    geom: np.ndarray | None = None
    for ch in range(c):
        values = _scipy_map_coordinates(
            src[ch], coords, order=1, mode="constant", cval=np.nan
        )
        values[reset] = np.nan
        values = values.reshape(h, w)
        if ch == 0:
            footprint = (~np.isnan(values)).astype(float)
            geom = footprint > 0.0
        out[..., ch] = np.asarray(values, dtype=np.float32)
    out[~geom] = _INVALID
    return out, geom


@dataclass
class SectionReadRecord:
    """Instrumentation record proving locality of a prepared-source read."""

    path: str
    source_bounds: SourceBounds
    axis_layout: str
    channel_slice: tuple[int, int, int]
    n_pixels_read: int
    full_frame_pixels: int


class SectionReadTracker:
    """Records every section read so tests can prove no full-frame ``.data`` read."""

    def __init__(self) -> None:
        self.records: list[SectionReadRecord] = []

    def add(self, record: SectionReadRecord) -> None:
        self.records.append(record)


def _read_section_chw(
    path: str,
    bounds: SourceBounds,
    *,
    tracker: SectionReadTracker | None = None,
) -> np.ndarray:
    """Read a ``(C, H, W)`` section from a prepared RGB FITS (no full ``.data``).

    Uses ``hdu.section`` (lazy partial read, memmap-backed) with the channel
    axis explicit. Never calls ``hdu.data``.
    """
    with fits.open(path, memmap=True, do_not_scale_image_data=False) as hdul:
        hdu = hdul[0]
        # section indexing follows the in-memory (C, H, W) order.
        section = hdu.section[:, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
        arr = np.ascontiguousarray(np.asarray(section, dtype=np.float32))
    if tracker is not None:
        full = int(np.prod(hdu.shape)) if hdu.shape else 0
        tracker.add(
            SectionReadRecord(
                path=path,
                source_bounds=bounds,
                axis_layout="CHW",
                channel_slice=(0, 3, 1),
                n_pixels_read=int(arr.size),
                full_frame_pixels=full,
            )
        )
    return arr


def slice_wcs(wcs_2d: WCS, bounds: SourceBounds) -> WCS:
    """Exact SIP-aware crop via ``WCS.slice``.

    ``WCS.slice`` produces the exact cropped WCS for BOTH plain TAN and
    SIP-distorted WCS (it propagates the SIP coefficients correctly). For a
    plain TAN WCS the result is IDENTICAL to the previous manual CRPIX shift
    (verified by the R10 regression test); for a SIP WCS the manual shift was
    INVALID (it ignored the distortion polynomial's reference frame), so this
    is the required correction. The cropped array's pixel ``(x,y)`` corresponds
    to source pixel ``(x + bounds.x0, y + bounds.y0)``.
    """
    sliced = wcs_2d.slice(
        (slice(bounds.x0, bounds.x1), slice(bounds.y0, bounds.y1)),
        numpy_order=False,
    )
    sliced.array_shape = (bounds.height, bounds.width)
    return sliced


def reproject_cropped(
    cropped_chw: np.ndarray,
    cropped_wcs: WCS,
    patch_wcs: WCS,
    patch_shape_hw: tuple[int, int],
    *,
    emit=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject a cropped ``(C,H,W)`` source to the exact patch shape.

    Returns ``(h_w_c_rgb, geometric_support)`` where ``geometric_support`` is a
    2-D bool map of the explicit reprojected geometric source domain
    (``footprint > 0``), independent of brightness. The RGB is HWC float32 with
    NaN outside the geometric support domain.

    ZM-ZEGRID-R21: uses the bit-exact fast path (:func:`reproject_cropped_fast`)
    for plain undistorted TAN WCS, and falls back to ``reproject_interp`` (the
    pre-R21 path) for SIP/distorted/non-TAN WCS or on any fast-path error. The
    path actually used is recorded in the process-local
    :func:`get_reproject_path_stats` accumulator and any fallback is surfaced
    loudly (log + ``emit`` WARN, once per process).
    """
    c = cropped_chw.shape[0]
    eligible = (
        not _REPROJECT_FORCE_FALLBACK
        and _plain_tan_wcs(cropped_wcs)
        and _plain_tan_wcs(patch_wcs)
    )
    fallback_reason: str | None = None

    if eligible:
        t0 = time.perf_counter()
        try:
            out, geom = reproject_cropped_fast(
                cropped_chw, cropped_wcs, patch_wcs, patch_shape_hw
            )
            _REPROJECT_STATS.record_fast(time.perf_counter() - t0)
            return out, geom
        except Exception as exc:  # noqa: BLE001 - fast path failure must fall back
            fallback_reason = f"{type(exc).__name__}: {exc}"
            _warn_fallback(emit, f"fast reprojection failed ({fallback_reason})")
    elif _REPROJECT_FORCE_FALLBACK:
        fallback_reason = _FALLBACK_REASON_FORCED
    else:
        fallback_reason = _FALLBACK_REASON_NON_TAN

    # Fallback path — the pre-R21 implementation, unchanged science.
    out = np.empty((patch_shape_hw[0], patch_shape_hw[1], c), dtype=np.float32)
    support: np.ndarray | None = None
    t0 = time.perf_counter()
    for ch in range(c):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr, footprint = reproject_interp(
                (cropped_chw[ch], cropped_wcs),
                output_projection=patch_wcs,
                shape_out=patch_shape_hw,
                order=REPROJECT_ORDER,
                return_footprint=True,
            )
        out[..., ch] = np.asarray(arr, dtype=np.float32)
        if support is None:
            support = np.asarray(footprint, dtype=np.float32)
    support = np.asarray(support, dtype=np.float32)
    geom = support > 0.0
    out[~geom] = _INVALID
    _REPROJECT_STATS.record_fallback(time.perf_counter() - t0, fallback_reason)
    return out, geom


@dataclass
class AlignedContributor:
    """One patch contributor: aligned RGB + explicit geometric support."""

    frame_id: str
    rgb: np.ndarray  # HWC float32, patch shape
    geometric_support: np.ndarray  # bool (H, W)
    crop: SourceCropPlan | None


def build_patch_contributors(
    frames: Sequence[FrameDescriptor],
    prepared_paths: dict[str, str],
    canvas: GlobalCanvas,
    patch: ProcessingPatch,
    crop_plans: dict[str, SourceCropPlan],
    *,
    tracker: SectionReadTracker | None = None,
) -> list[AlignedContributor]:
    """Read sections and reproject every patch contributor in stable order.

    ``frames`` must already be sorted by FrameId. ``prepared_paths`` maps a
    frame's logical path to its prepared RGB FITS path. ``crop_plans`` maps a
    frame's logical path to its precomputed :class:`SourceCropPlan`.
    """
    patch_wcs = patch.patch_wcs()
    contributors: list[AlignedContributor] = []
    # Stable sorted FrameId order — determinism regardless of input list order.
    for f in sorted(frames, key=lambda f: f.frame_id):
        key = f.frame_id.logical_path
        plan = crop_plans.get(key)
        if plan is None:
            continue
        prep = prepared_paths.get(key)
        if prep is None:
            raise ValueError(f"no prepared RGB for {key}")
        cropped = _read_section_chw(prep, plan.source_bounds, tracker=tracker)
        cropped_wcs = slice_wcs(f.wcs(), plan.source_bounds)
        rgb, geom = reproject_cropped(
            cropped, cropped_wcs, patch_wcs, patch.patch_shape_hw
        )
        contributors.append(
            AlignedContributor(frame_id=key, rgb=rgb, geometric_support=geom, crop=plan)
        )
    return contributors


def reproject_full_source(
    frame: FrameDescriptor,
    prepared_path: str,
    patch_wcs: WCS,
    patch_shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Full-source reprojection onto the same patch (oracle reference).

    Reads the FULL prepared RGB (CHW) and reprojects. Used only by the oracle
    test to prove the local crop covers every requested valid target pixel.
    """
    with fits.open(prepared_path, memmap=True) as hdul:
        hdu = hdul[0]
        full = np.ascontiguousarray(np.asarray(hdu.section[:], dtype=np.float32))
    wcs_2d = frame.wcs()
    c = full.shape[0]
    out = np.empty((patch_shape_hw[0], patch_shape_hw[1], c), dtype=np.float32)
    support = None
    for ch in range(c):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr, footprint = reproject_interp(
                (full[ch], wcs_2d),
                output_projection=patch_wcs,
                shape_out=patch_shape_hw,
                order=REPROJECT_ORDER,
                return_footprint=True,
            )
        out[..., ch] = np.asarray(arr, dtype=np.float32)
        if support is None:
            support = np.asarray(footprint, dtype=np.float32)
    support = np.asarray(support, dtype=np.float32)
    geom = support > 0.0
    out[~geom] = _INVALID
    return out, geom
