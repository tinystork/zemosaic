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

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

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
    """Exact CRPIX shift for an undistorted 2-D celestial WCS crop.

    ``WCS.slice`` / equivalent exact CRPIX shift for undistorted 2-D TAN. The
    cropped array's pixel ``(x,y)`` corresponds to source pixel
    ``(x + bounds.x0, y + bounds.y0)``.
    """
    w = wcs_2d.deepcopy()
    w.wcs.crpix = np.array(
        [wcs_2d.wcs.crpix[0] - bounds.x0, wcs_2d.wcs.crpix[1] - bounds.y0]
    )
    w.array_shape = (bounds.height, bounds.width)
    return w


def reproject_cropped(
    cropped_chw: np.ndarray,
    cropped_wcs: WCS,
    patch_wcs: WCS,
    patch_shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject a cropped ``(C,H,W)`` source to the exact patch shape.

    Returns ``(h_w_c_rgb, geometric_support)`` where ``geometric_support`` is a
    2-D bool map of the explicit reprojected geometric source domain
    (``footprint > 0``), independent of brightness. The RGB is HWC float32 with
    NaN outside the geometric support domain.
    """
    c = cropped_chw.shape[0]
    out = np.empty((patch_shape_hw[0], patch_shape_hw[1], c), dtype=np.float32)
    support: np.ndarray | None = None
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
