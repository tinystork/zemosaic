"""ZM-ZEGRID-R1 — manifest geometry, canvas, layout, cell, patch, source ROI.

Isolated R1 module. No production dispatch wiring. Reproduces the frozen
M106 R0 geometry (canvas 2403 x 3278, layout 5x4, Cell ``r0000c0000``) within
tight numeric tolerance and mirrors the R0 header-only simulator's algorithm
(``tools/zegrid_r0/geometry.py``) so the two can be cross-checked.

Conventions (frozen, must not drift):
* All bounds are integer half-open pixel indices named ``x0,y0,x1,y1``.
* ``x`` is the first / column axis (width), ``y`` the second / row axis (height),
  matching the R0 simulator's ``core=(x0,y0,x1,y1)`` order.
* NumPy arrays are ``(H, W) = (y, x)``; slicing is ``arr[y0:y1, x0:x1]``.
* Polygon pixel edges are ``x0-0.5`` etc. (zero-origin pixel *centres* are
  ``x+0.5``), i.e. the same pixel-edge convention as the R0 simulator.
* Qualified WCS only: 2-D celestial RA/DEC TAN — either plain (undistorted) or
  SIP-distorted (``RA---TAN-SIP`` / ``DEC--TAN-SIP``). PV / CPDIS / DET2IM and
  non-TAN projections are still rejected. No HWC/CHW channel-axis ambiguity.
"""

from __future__ import annotations

import copy
import hashlib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject.mosaicking import find_optimal_celestial_wcs
from shapely.geometry import Polygon, box

# Geometry policy version — bumped only on a breaking coordinate/partition change.
GEOMETRY_POLICY_VERSION = "zegrid-geom-v1"

# Source interpolation margin (source pixels) — explicit hypothesis, tied to the
# bilinear (order=1) reproject kernel + a conservative extra pixel. Verified by
# the full-source vs cropped-source same-patch oracle in tests.
DEFAULT_SOURCE_MARGIN_PX = 2

# Nonzero-area intersection threshold for membership (mirrors R0 ``1e-8``).
INTERSECTION_AREA_EPS = 1e-8

# Hemisphere roundtrip tolerance (deg) used to reject opposite-TAN projections.
ROUNDTRIP_TOL_DEG = 1e-7

# SIP footprint sampling policy (ZM-ZEGRID-R10). With SIP distortion the source
# edges project to CURVED edges, so a 4-corner polygon under-bounds the true
# footprint. We sample each source edge at a fixed deterministic step, project
# every sample, take the convex hull, and add a small conservative pixel margin.
# The margin is expressed in target (canvas) pixels and is far above the
# measured SIP magnitude on the Caldwell 11 corpus (max ~0.5 px; see the R10
# cross-consistency measurement).
SIP_FOOTPRINT_EDGE_STEP_PX = 128   # source-pixel step along each edge
SIP_FOOTPRINT_MARGIN_PX = 1.0      # conservative target-pixel margin

# ZM-ZEGRID-R11 L1 advisory: the margin above is CORPUS-VALIDATED, not a formal
# bound. It is justified only because the measured SIP distortion on the Caldwell
# 11 corpus (max ~0.5 px) is well below ``SIP_FOOTPRINT_MARGIN_PX`` (= 1.0 px). It
# is NOT a guaranteed bound for arbitrarily strong SIP distortion (large/high-
# order ``A``/``B`` coefficients can bend edges by more than the margin). For such
# inputs the correct fix is a curvature-adaptive sampling step (subdivide edges
# whose projected chord-vs-arc deviation exceeds a tolerance), not a larger fixed
# margin. The convex-hull construction already guarantees a VALID polygon; this
# advisory only bounds its geometric tightness, not its validity.
SIP_MARGIN_IS_CORPUS_VALIDATED = True  # NOT a formal bound for arbitrary SIP


# ---------------------------------------------------------------------------
# Coordinate types (distinct for global / patch / source)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GlobalBounds:
    """Integer half-open bounds in canvas (global) pixel indices."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class PatchBounds:
    """Integer half-open bounds in patch-local pixel indices."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class SourceBounds:
    """Integer half-open bounds in source pixel indices (read rectangle)."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def empty(self) -> bool:
        return self.x1 <= self.x0 or self.y1 <= self.y0


# ---------------------------------------------------------------------------
# FrameId — stable, Unicode-normalized, POSIX relative logical path
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameId:
    """Stable manifest identity: Unicode-normalized POSIX relative logical path.

    Root relocation must not change the ID; ordering uses this ID (never CSV
    ``order``, Python hash, or inode enumeration). Case preserved.
    """

    logical_path: str

    def __post_init__(self) -> None:
        norm = unicodedata.normalize("NFC", str(self.logical_path))
        norm = norm.replace("\\", "/")
        if norm.startswith("./"):
            norm = norm[2:]
        object.__setattr__(self, "logical_path", norm)

    def __lt__(self, other: "FrameId") -> bool:
        return self.logical_path < other.logical_path


# ---------------------------------------------------------------------------
# WCS qualification
# ---------------------------------------------------------------------------

def _has_sip(w: WCS) -> bool:
    """Return True when the WCS carries SIP distortion coefficients."""
    return w.sip is not None


def _sip_base_ctype(ct: str) -> str | None:
    """Return the base projection for a SIP ctype, or None if not a '-SIP' ctype."""
    return ct[:-4] if ct.endswith("-SIP") else None


def strip_sip(w: WCS) -> WCS:
    """Return a distortion-free copy of a SIP WCS (legacy-style strip).

    Removes the SIP coefficients and the ``-SIP`` CTYPE suffix, yielding the
    underlying plain TAN (linear) WCS. Used by the CONFIGURABLE fallback for
    SIP-vs-TAN consistency. Non-SIP WCS are returned unchanged.
    """
    out = w.deepcopy()
    if not _has_sip(out):
        return out
    out.sip = None
    ctype = list(out.wcs.ctype)
    out.wcs.ctype = [c[:-4] if c.endswith("-SIP") else c for c in ctype]
    return out


def qualify_wcs(w: WCS) -> str | None:
    """Return ``None`` if ``w`` is a qualified 2-D celestial TAN WCS.

    Accepts BOTH plain (undistorted) TAN and SIP-distorted TAN
    (``RA---TAN-SIP`` / ``DEC--TAN-SIP``) — and only those. Everything else is
    rejected with an explicit human-readable reason: PV / lookup (CPDIS/DET2IM)
    distortion, non-2-D, non-celestial, and non-TAN projections. This is a
    *gate*, not a silent strip.
    """
    try:
        if w.pixel_n_dim != 2:
            return f"pixel_n_dim={w.pixel_n_dim} (expected 2-D)"
        if not w.has_celestial:
            return "WCS has no celestial component"
        ctype = tuple(w.wcs.ctype)
        if _has_sip(w):
            # SIP-distorted celestial TAN — accepted only on a pure TAN base,
            # with no additional PV / lookup distortion on top of the SIP terms.
            base = tuple(_sip_base_ctype(c) for c in ctype)
            if any(b is None for b in base):
                return f"ctype={ctype!r} (SIP coefficients present without '-SIP' suffix; ambiguous)"
            if base != ("RA---TAN", "DEC--TAN"):
                return f"ctype={ctype!r} (SIP requires a TAN base; expected RA---TAN-SIP/DEC--TAN-SIP)"
            if w.wcs.get_pv():
                return "WCS has PV terms in addition to SIP; unsupported"
            for attr in ("cpdis1", "cpdis2", "det2im1", "det2im2"):
                if getattr(w, attr, None) is not None:
                    return f"WCS has lookup distortion ({attr}) in addition to SIP; unsupported"
            return None
        # Non-SIP path (unchanged frozen semantics): undistorted TAN only.
        if w.has_distortion:
            return "WCS has distortion terms (PV/CPDIS/DET2IM); undistorted TAN required"
        if ctype != ("RA---TAN", "DEC--TAN"):
            return f"ctype={ctype!r} (expected ('RA---TAN', 'DEC--TAN'))"
        if w.wcs.get_pv():
            return "WCS has PV terms; undistorted TAN required"
        # Explicit lookup-distortion guard (belt-and-braces on top of has_distortion).
        for attr in ("cpdis1", "cpdis2", "det2im1", "det2im2"):
            if getattr(w, attr, None) is not None:
                return f"WCS has lookup distortion ({attr}); undistorted TAN required"
        return None
    except Exception as exc:  # pragma: no cover - defensive
        return f"WCS qualification raised {type(exc).__name__}: {exc}"


def qualify_axis_layout(shape: Sequence[int], declared: str) -> str | None:
    """Reject ambiguous HWC/CHW channel-axis layouts.

    For R1 prepared RGB the axis layout is declared explicitly (``"CHW"`` FITS
    storage), so no ambiguity exists. This gate makes the contract explicit and
    fails on undeclared or contradictory 3-D shapes.
    """
    shape = tuple(int(s) for s in shape)
    if len(shape) == 2:
        return None  # mono source, no channel axis
    if len(shape) != 3:
        return f"expected 2-D or 3-D source, got shape {shape}"
    declared = (declared or "").upper()
    if declared not in ("CHW", "HWC"):
        return f"3-D source with undeclared/ambiguous channel layout {declared!r} (shape {shape})"
    return None


# ---------------------------------------------------------------------------
# FrameDescriptor / manifest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameDescriptor:
    """Immutable manifest frame: identity + geometry + WCS snapshot (no pixels)."""

    frame_id: FrameId
    source_path: str
    shape_hw: tuple[int, int]  # (H, W)
    wcs_header: str  # immutable serialized WCS snapshot
    header_sha256: str = ""
    instrument: str = ""

    @property
    def height(self) -> int:
        return self.shape_hw[0]

    @property
    def width(self) -> int:
        return self.shape_hw[1]

    def wcs(self) -> WCS:
        """Decode a private WCS copy from the serialized snapshot."""
        return WCS(self.wcs_header)


def _serialize_wcs(w: WCS) -> str:
    hdr = w.to_header(relax=True)
    return hdr.tostring()


def read_manifest(
    root: str | Path,
    *,
    reject_invalid: bool = False,
) -> tuple[list[FrameDescriptor], list[dict]]:
    """Read a header-only manifest from a FITS directory (no pixel access).

    Mirrors the R0 simulator's ``read_frames``: ``fits.getheader`` never touches
    ``HDU.data``. Returns ``(frames, rejected)`` where ``frames`` are the
    qualified frames sorted by :class:`FrameId`.
    """
    root = Path(root).resolve()
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".fit", ".fits", ".fts"))
    if not paths:
        raise ValueError("no FITS files")
    frames: list[FrameDescriptor] = []
    rejected: list[dict] = []
    seen: set[Path] = set()
    for p in paths:
        rel = p.relative_to(root).as_posix()
        reason = None
        try:
            resolved = p.resolve()
            if resolved in seen:
                reason = "duplicate resolved path"
            else:
                seen.add(resolved)
                h = fits.getheader(p, 0)
                if int(h.get("NAXIS", 0)) != 2:
                    reason = "R1 manifest accepts 2-D primary images only (CHW/HWC must be declared)"
                else:
                    w = WCS(h)
                    reason = qualify_wcs(w)
                    if reason is None:
                        shape = (int(h["NAXIS2"]), int(h["NAXIS1"]))
                        if min(shape) <= 0:
                            reason = "nonpositive shape"
                        else:
                            frames.append(
                                FrameDescriptor(
                                    frame_id=FrameId(rel),
                                    source_path=str(p),
                                    shape_hw=shape,
                                    wcs_header=_serialize_wcs(w),
                                    header_sha256=hashlib.sha256(
                                        h.tostring().encode()
                                    ).hexdigest(),
                                    instrument=str(h.get("INSTRUME", "")),
                                )
                            )
        except Exception as exc:
            reason = str(exc)
        if reason is not None:
            rejected.append({"path": rel, "reason": reason})
    if not frames:
        raise ValueError("no supported WCS; exclusions: " + str(rejected[:5]))
    frames.sort(key=lambda f: f.frame_id)
    return frames, rejected


# ---------------------------------------------------------------------------
# Canvas builder (separated from layout)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GlobalCanvas:
    """Frozen canvas: WCS snapshot + shape + resolution. No layout policy."""

    canvas_id: str
    wcs_header: str
    width: int
    height: int
    resolution_deg: float

    def wcs(self) -> WCS:
        return WCS(self.wcs_header)

    def bounds(self) -> GlobalBounds:
        return GlobalBounds(0, 0, self.width, self.height)


def _boundary(shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])


def _boundary_samples(shape_hw: tuple[int, int], step: int) -> np.ndarray:
    """Return corner + per-edge sample points (source pixel edges) at fixed step.

    Walks the four pixel-edge segments (bottom, right, top, left) in a closed
    loop, sampling at a deterministic ``step`` so every curved SIP edge is
    resolved. The corners are included exactly once each.
    """
    h, w = shape_hw
    step = max(1, int(step))
    pts: list[tuple[float, float]] = []
    # bottom edge y = -0.5, x from -0.5 .. w-0.5
    for x in np.arange(-0.5, w - 0.5 + 1e-9, step):
        pts.append((float(x), -0.5))
    # right edge x = w-0.5, y from -0.5 .. h-0.5
    for y in np.arange(-0.5, h - 0.5 + 1e-9, step):
        pts.append((w - 0.5, float(y)))
    # top edge y = h-0.5, x from w-0.5 .. -0.5
    for x in np.arange(w - 0.5, -0.5 - 1e-9, -step):
        pts.append((float(x), h - 0.5))
    # left edge x = -0.5, y from h-0.5 .. -0.5
    for y in np.arange(h - 0.5, -0.5 - 1e-9, -step):
        pts.append((-0.5, float(y)))
    # Ensure the loop closes on the bottom-left corner.
    if not pts or (abs(pts[-1][0] - (-0.5)) > 1e-9 or abs(pts[-1][1] - (-0.5)) > 1e-9):
        pts.append((-0.5, -0.5))
    return np.array(pts)


def _project_points(points: np.ndarray, source: WCS, target: WCS) -> np.ndarray:
    sky = source.pixel_to_world(points[:, 0], points[:, 1])
    x, y = target.world_to_pixel(sky)
    xy = np.column_stack((x, y))
    if not np.isfinite(xy).all():
        raise ValueError("nonfinite projection; no geometry fallback")
    back = target.pixel_to_world(x, y)
    if np.any(back.separation(sky).deg > ROUNDTRIP_TOL_DEG):
        raise ValueError("projection roundtrip/hemisphere failure")
    return xy


def _source_polygon(shape_hw: tuple[int, int], src_wcs: WCS, tgt_wcs: WCS) -> Polygon:
    if _has_sip(src_wcs):
        return _sip_source_polygon(shape_hw, src_wcs, tgt_wcs)
    # Plain TAN path (unchanged frozen semantics): exact 4-corner projection.
    p = Polygon(_project_points(_boundary(shape_hw), src_wcs, tgt_wcs))
    if not p.is_valid or p.area <= 0:
        raise ValueError("invalid projected footprint")
    return p


def _sip_source_polygon(
    shape_hw: tuple[int, int],
    src_wcs: WCS,
    tgt_wcs: WCS,
    *,
    step: int = SIP_FOOTPRINT_EDGE_STEP_PX,
    margin: float = SIP_FOOTPRINT_MARGIN_PX,
) -> Polygon:
    """Conservative SIP footprint: sampled edges -> convex hull -> margin.

    With SIP distortion the projected edges are curved, so the 4-corner polygon
    under-bounds the true footprint. We sample every edge at a fixed
    deterministic step, project all samples, take the convex hull (which bounds
    every sampled point), and dilate by a small conservative margin in target
    pixels. The result bounds the true SIP footprint with a small margin and is
    guaranteed valid.
    """
    pts = _boundary_samples(shape_hw, step)
    xy = _project_points(pts, src_wcs, tgt_wcs)
    hull = Polygon(xy).convex_hull
    if margin and margin > 0:
        hull = hull.buffer(margin)
    p = Polygon(hull.exterior.coords) if hull.geom_type != "Polygon" else hull
    if not p.is_valid or p.area <= 0:
        raise ValueError("invalid projected SIP footprint")
    return p


def build_canvas(frames: Sequence[FrameDescriptor]) -> GlobalCanvas:
    """Reproduce the frozen M106 canvas (2403 x 3278) exactly.

    Same algorithm as the R0 simulator ``make_canvas``: sorted stable inputs,
    numeric pixel-scale median, ``find_optimal_celestial_wcs`` (TAN,
    auto_rotate), projected-footprint bounds, single baked offset. The offset is
    baked into CRPIX once; canvas pixels have no hidden offset.
    """
    frames = sorted(frames, key=lambda f: f.frame_id)
    if len({f.frame_id for f in frames}) != len(frames):
        raise ValueError("duplicate frame identity")
    wcs_list: list[WCS] = []
    for f in frames:
        w = f.wcs()
        reason = qualify_wcs(w)
        if reason is not None:
            raise ValueError(f"frame {f.frame_id.logical_path}: {reason}")
        wcs_list.append(w)

    resolution = float(
        np.median([np.mean(np.abs(proj_plane_pixel_scales(w))) for w in wcs_list])
    )
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("invalid scale")

    optimal, _ = find_optimal_celestial_wcs(
        [(f.shape_hw, w) for f, w in zip(frames, wcs_list)],
        resolution=resolution * u.deg,
        projection="TAN",
        auto_rotate=True,
    )

    bounds = np.array(
        [_source_polygon(f.shape_hw, w, optimal).bounds for f, w in zip(frames, wcs_list)]
    )
    x0 = int(np.floor(bounds[:, :2].min(axis=0)[0] + 0.5))
    y0 = int(np.floor(bounds[:, :2].min(axis=0)[1] + 0.5))
    x1 = int(np.ceil(bounds[:, 2:].max(axis=0)[0] + 0.5))
    y1 = int(np.ceil(bounds[:, 2:].max(axis=0)[1] + 0.5))

    optimal = copy.deepcopy(optimal)
    optimal.wcs.crpix -= np.array([x0, y0])
    optimal.array_shape = (int(y1 - y0), int(x1 - x0))

    width = int(x1 - x0)
    height = int(y1 - y0)
    canvas_id = hashlib.sha256(
        (_serialize_wcs(optimal) + f"{width}x{height}" + GEOMETRY_POLICY_VERSION).encode()
    ).hexdigest()[:16]
    return GlobalCanvas(
        canvas_id=canvas_id,
        wcs_header=_serialize_wcs(optimal),
        width=width,
        height=height,
        resolution_deg=resolution,
    )


# ---------------------------------------------------------------------------
# Layout / cell / patch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZeGridLayout:
    """Disjoint integer partition policy (no pixels, no source file order)."""

    layout_id: str
    canvas_id: str
    nx: int
    ny: int

    def cell_bounds(self, row: int, col: int, canvas: GlobalCanvas) -> GlobalBounds:
        if not (0 <= col < self.nx and 0 <= row < self.ny):
            raise ValueError(f"cell ({row},{col}) out of layout {self.nx}x{self.ny}")
        x0 = col * canvas.width // self.nx
        x1 = (col + 1) * canvas.width // self.nx
        y0 = row * canvas.height // self.ny
        y1 = (row + 1) * canvas.height // self.ny
        return GlobalBounds(x0, y0, x1, y1)

    def iter_cells(self, canvas: GlobalCanvas) -> Iterable[tuple[int, int, GlobalBounds]]:
        for row in range(self.ny):
            for col in range(self.nx):
                yield row, col, self.cell_bounds(row, col, canvas)


def build_layout(canvas: GlobalCanvas, nx: int, ny: int) -> ZeGridLayout:
    if not (1 <= nx <= canvas.width) or not (1 <= ny <= canvas.height):
        raise ValueError("invalid layout dimensions")
    layout_id = hashlib.sha256(
        f"{canvas.canvas_id}|{nx}x{ny}|{GEOMETRY_POLICY_VERSION}".encode()
    ).hexdigest()[:16]
    return ZeGridLayout(layout_id=layout_id, canvas_id=canvas.canvas_id, nx=nx, ny=ny)


def cell_id(row: int, col: int) -> str:
    return f"r{row:04d}c{col:04d}"


@dataclass(frozen=True)
class ZeGridCell:
    """Disjoint Cell core (global bounds). Halo/read buffers are separate."""

    cell_id: str
    canvas_id: str
    layout_id: str
    row: int
    col: int
    core: GlobalBounds


@dataclass(frozen=True)
class ProcessingPatch:
    """Cell core + target-pixel halo clipped to canvas, plus core slice in patch.

    ``core_slice_in_patch`` is expressed as ``(y0, y1, x0, x1)`` (NumPy order)
    so ``arr[ys:ye, xs:xe]`` extracts the core from any patch-shaped plane.
    """

    cell_id: str
    patch: GlobalBounds
    core_slice: PatchBounds  # patch-local (y0,y1,x0,x1)
    halo_px: int
    patch_wcs_header: str

    @property
    def patch_shape_hw(self) -> tuple[int, int]:
        return (self.patch.height, self.patch.width)

    def patch_wcs(self) -> WCS:
        return WCS(self.patch_wcs_header)


def build_patch(canvas: GlobalCanvas, cell: ZeGridCell, halo_px: int) -> ProcessingPatch:
    """Build a ProcessingPatch = cell + halo clipped to the canvas.

    The patch WCS is the canvas WCS with CRPIX shifted by the patch origin (the
    canvas offset is already baked), so a patch pixel centre ``(x,y)`` maps
    exactly to canvas pixel ``(x + patch.x0, y + patch.y0)``.
    """
    if halo_px < 0:
        raise ValueError("negative halo")
    core = cell.core
    px0 = max(0, core.x0 - halo_px)
    py0 = max(0, core.y0 - halo_px)
    px1 = min(canvas.width, core.x1 + halo_px)
    py1 = min(canvas.height, core.y1 + halo_px)
    patch = GlobalBounds(px0, py0, px1, py1)
    # Core slice within the patch (patch-local, NumPy (y,x) order).
    cs_x0 = core.x0 - px0
    cs_y0 = core.y0 - py0
    cs_x1 = core.x1 - px0
    cs_y1 = core.y1 - py0
    core_slice = PatchBounds(cs_x0, cs_y0, cs_x1, cs_y1)

    w = canvas.wcs()
    w = copy.deepcopy(w)
    w.wcs.crpix -= np.array([px0, py0])
    w.array_shape = (patch.height, patch.width)
    return ProcessingPatch(
        cell_id=cell.cell_id,
        patch=patch,
        core_slice=core_slice,
        halo_px=halo_px,
        patch_wcs_header=_serialize_wcs(w),
    )


# ---------------------------------------------------------------------------
# Membership + source ROI planning
# ---------------------------------------------------------------------------

def _rect(bounds: GlobalBounds) -> box:
    return box(bounds.x0 - 0.5, bounds.y0 - 0.5, bounds.x1 - 0.5, bounds.y1 - 0.5)


@dataclass(frozen=True)
class CellMembership:
    """Sorted geometric candidate IDs for a cell core and its patch."""

    cell_id: str
    core_ids: tuple[str, ...]  # sorted by FrameId
    patch_ids: tuple[str, ...]  # sorted by FrameId


def compute_membership(
    frames: Sequence[FrameDescriptor],
    canvas: GlobalCanvas,
    cell: ZeGridCell,
    patch: ProcessingPatch,
) -> CellMembership:
    """Narrow-phase polygon intersect rectangle membership (core vs patch).

    Sorted by FrameId for determinism. ``core_ids`` may be a strict subset of
    ``patch_ids`` (halo-only contributors); nonempty halo with empty core is not
    used to invent core data.
    """
    canvas_wcs = canvas.wcs()
    core_rect = _rect(cell.core)
    patch_rect = _rect(patch.patch)
    core_ids: list[str] = []
    patch_ids: list[str] = []
    for f in sorted(frames, key=lambda f: f.frame_id):
        poly = _source_polygon(f.shape_hw, f.wcs(), canvas_wcs)
        if poly.intersection(core_rect).area > INTERSECTION_AREA_EPS:
            core_ids.append(f.frame_id.logical_path)
        if poly.intersection(patch_rect).area > INTERSECTION_AREA_EPS:
            patch_ids.append(f.frame_id.logical_path)
    return CellMembership(
        cell_id=cell.cell_id,
        core_ids=tuple(core_ids),
        patch_ids=tuple(patch_ids),
    )


@dataclass(frozen=True)
class SourceCropPlan:
    """Conservative source read rectangle for one frame -> one patch."""

    frame_id: str
    cell_id: str
    source_bounds: SourceBounds
    margin_px: int


def plan_source_roi(
    frame: FrameDescriptor,
    canvas: GlobalCanvas,
    patch: ProcessingPatch,
    margin_px: int = DEFAULT_SOURCE_MARGIN_PX,
) -> SourceCropPlan | None:
    """Narrow phase: source polygon intersect patch pixel-edge rectangle.

    Inverse-maps the intersection vertices to source pixels, takes conservative
    outward integer bounds, adds the explicit source-pixel interpolation margin,
    and clips to the source. Returns ``None`` for zero-area overlap (no plan).
    """
    if margin_px < 0:
        raise ValueError("negative source margin")
    canvas_wcs = canvas.wcs()
    src_wcs = frame.wcs()
    poly = _source_polygon(frame.shape_hw, src_wcs, canvas_wcs)
    intersection = poly.intersection(_rect(patch.patch))
    if intersection.is_empty or intersection.area <= 0:
        return None
    if intersection.geom_type != "Polygon":
        # A degenerate/multi-part intersection is not qualified for a simple
        # rectangular read; a robust planner must sample adaptively. R1 inputs
        # are convex undistorted TAN so this is not expected.
        raise ValueError(
            f"nonconvex/disconnected intersection for {frame.frame_id.logical_path} "
            f"not qualified ({intersection.geom_type})"
        )
    xy = _project_points(np.array(intersection.exterior.coords), canvas_wcs, src_wcs)
    h, w = frame.shape_hw
    lo = np.floor(xy.min(axis=0) - margin_px).astype(int)
    hi = np.ceil(xy.max(axis=0) + margin_px).astype(int) + 1
    sx0 = int(max(0, lo[0]))
    sy0 = int(max(0, lo[1]))
    sx1 = int(min(w, hi[0]))
    sy1 = int(min(h, hi[1]))
    return SourceCropPlan(
        frame_id=frame.frame_id.logical_path,
        cell_id=patch.cell_id,
        source_bounds=SourceBounds(sx0, sy0, sx1, sy1),
        margin_px=margin_px,
    )
