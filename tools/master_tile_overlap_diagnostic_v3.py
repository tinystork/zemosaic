#!/usr/bin/env python3
"""Standalone GUI + CLI diagnostic for ZeMosaic master-tile overlaps (V3).

V3 additions
------------
- Emulates ZeMosaic-style top-K pair pruning + bridge reconnection.
- Compares raw overlap graph vs retained/pruned graph.
- Adds acquisition-date visualization (or fallback scalar map when dates are absent).
- Exports richer CSV/log diagnostics to help understand whether seams come from
  geometry, pruning, peripheral anchoring, or heterogeneous acquisition sessions.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    import astropy.units as u
except Exception as exc:  # pragma: no cover
    print(f"ERROR: astropy is required: {exc}", file=sys.stderr)
    raise

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import Polygon as MplPolygon
except Exception as exc:  # pragma: no cover
    print(f"ERROR: matplotlib is required: {exc}", file=sys.stderr)
    raise

try:
    from PySide6.QtCore import QThread, Qt, QUrl, Signal
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
    HAVE_QT = True
except Exception:
    HAVE_QT = False


FITS_EXTS = {".fits", ".fit", ".fts", ".fz"}
DEFAULT_PRUNE_K = 8


@dataclass
class TileInfo:
    index: int
    path: Path
    name: str
    shape_hw: Tuple[int, int]
    center_radec_deg: Tuple[float, float]
    center_xy_deg: Tuple[float, float]
    pa_deg: Optional[float]
    pixel_scale_arcsec: Optional[float]
    footprint_radec_deg: np.ndarray
    footprint_xy_deg: np.ndarray
    area_sqdeg: float
    width_deg: float
    height_deg: float
    filter_name: str
    exposure_s: Optional[float]
    date_obs: str
    date_ord: Optional[float]
    stack_count: Optional[int]
    raw_header_keys: dict


@dataclass
class OverlapInfo:
    i: int
    j: int
    tile_i: str
    tile_j: str
    area_sqdeg: float
    frac_of_i: float
    frac_of_j: float
    pair_strength_min: float
    pair_strength_avg: float
    center_dist_deg: float
    bbox_touch: bool
    weight_area: float = 0.0
    weight_strength: float = 0.0
    weight_hybrid: float = 0.0


@dataclass
class TileMetric:
    index: int
    name: str
    raw_neighbors: int
    kept_neighbors: int
    bridge_neighbors: int
    raw_best_local_overlap_frac: float
    raw_mean_local_overlap_frac: float
    kept_best_local_overlap_frac: float
    kept_mean_local_overlap_frac: float
    raw_best_pair_strength: float
    raw_mean_pair_strength: float
    kept_best_pair_strength: float
    kept_mean_pair_strength: float
    raw_connectivity: float
    kept_connectivity: float
    anchor_score_raw: float
    anchor_score_kept: float
    center_x_deg: float
    center_y_deg: float
    radial_distance_deg: float
    area_sqdeg: float
    width_deg: float
    height_deg: float
    date_ord: Optional[float]
    days_from_median: Optional[float]
    retention_ratio: float
    stack_weight_proxy: float
    is_isolated_raw: bool = False
    is_isolated_kept: bool = False
    is_low_neighbor_raw: bool = False
    is_low_neighbor_kept: bool = False
    is_low_best_overlap_raw: bool = False
    is_low_best_overlap_kept: bool = False
    is_peripheral: bool = False
    is_date_outlier: bool = False


@dataclass
class GraphSummary:
    applied: bool
    fallback_used: bool
    raw_pairs: int
    kept_pairs: int
    topk_pairs: int
    raw_components: int
    topk_components: int
    final_components: int
    bridges_added: int
    active_tiles: int
    kept_keys: set[Tuple[int, int]]
    bridge_keys: set[Tuple[int, int]]
    dropped_keys: set[Tuple[int, int]]


@dataclass
class DiagnosticConfig:
    inputs: List[Path]
    recursive: bool = True
    outdir: Path = Path("tile_overlap_diag")
    prefix: str = "master_tiles"
    min_overlap_frac: float = 0.0
    ref_ra: Optional[float] = None
    ref_dec: Optional[float] = None
    annotate: bool = False
    draw_links: bool = True
    top_n_summary: int = 20
    label_top_n: int = 16
    emulate_pruning: bool = True
    prune_k: int = DEFAULT_PRUNE_K
    force_prune: bool = False
    prune_weight_mode: str = "area"


@dataclass
class DiagnosticResult:
    tile_count: int
    overlap_count: int
    kept_overlap_count: int
    out_overview_png: Path
    out_log: Path
    out_overlaps_csv: Path
    out_tiles_csv: Path


class CancelledError(RuntimeError):
    pass


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _edge_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


def polygon_area(poly: np.ndarray) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    signed = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return abs(signed)


def signed_polygon_area(poly: np.ndarray) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly.copy() if signed_polygon_area(poly) >= 0 else poly[::-1].copy()


def _inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    return _cross(b - a, p - a) >= -1e-12


def _segment_intersection(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = p2 - p1
    s = b - a
    denom = _cross(r, s)
    if abs(denom) < 1e-15:
        return (p1 + p2) / 2.0
    t = _cross(a - p1, s) / denom
    return p1 + t * r


def convex_polygon_intersection(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    if subject is None or clipper is None or len(subject) < 3 or len(clipper) < 3:
        return np.empty((0, 2), dtype=float)
    output = _ensure_ccw(np.asarray(subject, dtype=float))
    clip = _ensure_ccw(np.asarray(clipper, dtype=float))
    for idx in range(len(clip)):
        a = clip[idx]
        b = clip[(idx + 1) % len(clip)]
        input_list = output
        if len(input_list) == 0:
            break
        output_pts: List[np.ndarray] = []
        s = input_list[-1]
        for e in input_list:
            e_in = _inside(e, a, b)
            s_in = _inside(s, a, b)
            if e_in:
                if not s_in:
                    output_pts.append(_segment_intersection(s, e, a, b))
                output_pts.append(e)
            elif s_in:
                output_pts.append(_segment_intersection(s, e, a, b))
            s = e
        output = np.array(output_pts, dtype=float) if output_pts else np.empty((0, 2), dtype=float)
    return output


def normalize_ra_deg(ra_deg: np.ndarray, ref_ra_deg: float) -> np.ndarray:
    return ((ra_deg - ref_ra_deg + 180.0) % 360.0) - 180.0 + ref_ra_deg


def project_radec_to_plane_deg(points_radec: np.ndarray, ref_ra_deg: float, ref_dec_deg: float) -> np.ndarray:
    ra = normalize_ra_deg(np.asarray(points_radec[:, 0], dtype=float), ref_ra_deg)
    dec = np.asarray(points_radec[:, 1], dtype=float)
    cos_dec0 = math.cos(math.radians(ref_dec_deg))
    if abs(cos_dec0) < 1e-8:
        cos_dec0 = 1e-8 if cos_dec0 >= 0 else -1e-8
    x = (ra - ref_ra_deg) * cos_dec0
    y = dec - ref_dec_deg
    return np.column_stack([x, y])


def compute_position_angle_deg(wcs_obj: WCS, width: int, height: int) -> Optional[float]:
    try:
        cx = width / 2.0
        cy = height / 2.0
        c0 = wcs_obj.pixel_to_world(cx, cy)
        c1 = wcs_obj.pixel_to_world(cx + 1.0, cy)
        return float(c0.position_angle(c1).to(u.deg).value) % 360.0
    except Exception:
        return None


def compute_footprint_radec_deg(wcs_obj: WCS, width: int, height: int) -> np.ndarray:
    corners_px = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=float,
    )
    sky = wcs_obj.pixel_to_world(corners_px[:, 0], corners_px[:, 1])
    ra_deg = np.asarray(sky.ra.to(u.deg).value, dtype=float)
    dec_deg = np.asarray(sky.dec.to(u.deg).value, dtype=float)
    return np.column_stack([ra_deg, dec_deg])


def estimate_width_height_deg(poly_xy_deg: np.ndarray) -> Tuple[float, float]:
    return (
        float(np.nanmax(poly_xy_deg[:, 0]) - np.nanmin(poly_xy_deg[:, 0])),
        float(np.nanmax(poly_xy_deg[:, 1]) - np.nanmin(poly_xy_deg[:, 1])),
    )


def extract_stack_count(header) -> Optional[int]:
    for key in ("STACKCNT", "STACKN", "NSTACK", "NCOMBINE", "IMCNT"):
        val = header.get(key)
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, float) and np.isfinite(val):
            return int(round(val))
    return None


def parse_date_obs_to_ordinal(date_obs: str) -> Optional[float]:
    txt = str(date_obs or "").strip()
    if not txt:
        return None
    txt = txt.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(txt)
    except Exception:
        dt = None
    if dt is None:
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y",
        ):
            try:
                dt = datetime.strptime(txt, fmt)
                break
            except Exception:
                continue
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp() / 86400.0)


def iter_fits_files(paths: Sequence[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        if path.is_file():
            if path.suffix.lower() in FITS_EXTS:
                yield path
            continue
        if not path.is_dir():
            continue
        iterator = path.rglob("*") if recursive else path.glob("*")
        for child in iterator:
            if child.is_file() and child.suffix.lower() in FITS_EXTS:
                yield child


def open_header_and_wcs(path: Path) -> Tuple[Optional[object], Optional[WCS]]:
    try:
        header = fits.getheader(path, 0)
    except Exception:
        return None, None
    try:
        wcs_obj = WCS(header, naxis=2, relax=True)
    except Exception:
        return header, None
    if not getattr(wcs_obj, "is_celestial", False):
        return header, None
    return header, wcs_obj


def build_tile_infos(
    files: Sequence[Path],
    ref_ra_deg: Optional[float] = None,
    ref_dec_deg: Optional[float] = None,
    progress: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> List[TileInfo]:
    pre_tiles = []
    centers = []
    total = len(files)
    for idx, path in enumerate(files, start=1):
        if cancel_check and cancel_check():
            raise CancelledError("Cancelled by user")
        if progress and (idx == 1 or idx == total or idx % 25 == 0):
            progress(f"Reading headers/WCS: {idx}/{total}")
        header, wcs_obj = open_header_and_wcs(path)
        if header is None or wcs_obj is None:
            continue
        width = header.get("NAXIS1")
        height = header.get("NAXIS2")
        if not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer)):
            continue
        if width <= 1 or height <= 1:
            continue
        try:
            cx = width / 2.0
            cy = height / 2.0
            center = wcs_obj.pixel_to_world(cx, cy)
            ra0 = float(center.ra.deg)
            dec0 = float(center.dec.deg)
        except Exception:
            ra_val = header.get("CRVAL1")
            dec_val = header.get("CRVAL2")
            if ra_val is None or dec_val is None:
                continue
            ra0 = float(ra_val)
            dec0 = float(dec_val)
        centers.append((ra0, dec0))
        pre_tiles.append((path, header, wcs_obj, int(width), int(height), ra0, dec0))

    if not pre_tiles:
        return []

    if ref_ra_deg is None or ref_dec_deg is None:
        ra_vals = np.array([c[0] for c in centers], dtype=float)
        dec_vals = np.array([c[1] for c in centers], dtype=float)
        ref_ra_deg = float(np.median(ra_vals)) if ref_ra_deg is None else float(ref_ra_deg)
        ref_dec_deg = float(np.median(dec_vals)) if ref_dec_deg is None else float(ref_dec_deg)

    tiles: List[TileInfo] = []
    for idx, (path, header, wcs_obj, width, height, ra0, dec0) in enumerate(pre_tiles):
        if cancel_check and cancel_check():
            raise CancelledError("Cancelled by user")
        fp_radec = compute_footprint_radec_deg(wcs_obj, width, height)
        fp_xy = project_radec_to_plane_deg(fp_radec, ref_ra_deg, ref_dec_deg)
        center_xy = project_radec_to_plane_deg(np.array([[ra0, dec0]], dtype=float), ref_ra_deg, ref_dec_deg)[0]
        area_sqdeg = polygon_area(fp_xy)
        width_deg, height_deg = estimate_width_height_deg(fp_xy)
        scale_arcsec = None
        try:
            scales = proj_plane_pixel_scales(wcs_obj.celestial) * 3600.0
            if scales is not None and len(scales) >= 2:
                scale_arcsec = float(np.nanmean(np.abs(scales[:2])))
        except Exception:
            scale_arcsec = None
        exposure_val = header.get("EXPTIME")
        exposure_s = float(exposure_val) if isinstance(exposure_val, (int, float, np.integer, np.floating)) else None
        date_obs = str(header.get("DATE-OBS", "") or "")
        tiles.append(
            TileInfo(
                index=idx,
                path=path,
                name=path.name,
                shape_hw=(int(height), int(width)),
                center_radec_deg=(float(ra0), float(dec0)),
                center_xy_deg=(float(center_xy[0]), float(center_xy[1])),
                pa_deg=compute_position_angle_deg(wcs_obj, width, height),
                pixel_scale_arcsec=scale_arcsec,
                footprint_radec_deg=fp_radec,
                footprint_xy_deg=fp_xy,
                area_sqdeg=float(area_sqdeg),
                width_deg=float(width_deg),
                height_deg=float(height_deg),
                filter_name=str(header.get("FILTER", header.get("FILTNAME", "")) or ""),
                exposure_s=exposure_s,
                date_obs=date_obs,
                date_ord=parse_date_obs_to_ordinal(date_obs),
                stack_count=extract_stack_count(header),
                raw_header_keys={
                    key: header.get(key)
                    for key in (
                        "OBJECT", "TELESCOP", "INSTRUME", "FILTER", "EXPTIME",
                        "DATE-OBS", "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2",
                        "CDELT1", "CDELT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2",
                        "PC1_1", "PC1_2", "PC2_1", "PC2_2", "STACKCNT", "NCOMBINE",
                    )
                },
            )
        )
    return tiles


def bbox_intersects(a: np.ndarray, b: np.ndarray) -> bool:
    ax0, ay0 = np.nanmin(a[:, 0]), np.nanmin(a[:, 1])
    ax1, ay1 = np.nanmax(a[:, 0]), np.nanmax(a[:, 1])
    bx0, by0 = np.nanmin(b[:, 0]), np.nanmin(b[:, 1])
    bx1, by1 = np.nanmax(b[:, 0]), np.nanmax(b[:, 1])
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def analyze_overlaps(
    tiles: Sequence[TileInfo],
    min_fraction: float = 0.0,
    progress: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> List[OverlapInfo]:
    overlaps: List[OverlapInfo] = []
    n = len(tiles)
    total_pairs = n * (n - 1) // 2
    pair_index = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_index += 1
            if cancel_check and cancel_check():
                raise CancelledError("Cancelled by user")
            if progress and (pair_index == 1 or pair_index == total_pairs or pair_index % max(1, total_pairs // 20 or 1) == 0):
                progress(f"Analyzing overlaps: {pair_index}/{total_pairs}")
            ti = tiles[i]
            tj = tiles[j]
            bbox_touch = bbox_intersects(ti.footprint_xy_deg, tj.footprint_xy_deg)
            if not bbox_touch:
                continue
            inter = convex_polygon_intersection(ti.footprint_xy_deg, tj.footprint_xy_deg)
            inter_area = polygon_area(inter)
            if inter_area <= 0.0:
                continue
            frac_i = inter_area / max(ti.area_sqdeg, 1e-15)
            frac_j = inter_area / max(tj.area_sqdeg, 1e-15)
            if max(frac_i, frac_j) < min_fraction:
                continue
            pair_strength_min = min(frac_i, frac_j)
            pair_strength_avg = 0.5 * (frac_i + frac_j)
            dx = ti.center_radec_deg[0] - tj.center_radec_deg[0]
            dy = ti.center_radec_deg[1] - tj.center_radec_deg[1]
            center_dist = math.hypot(dx * math.cos(math.radians((ti.center_radec_deg[1] + tj.center_radec_deg[1]) * 0.5)), dy)
            overlaps.append(
                OverlapInfo(
                    i=i,
                    j=j,
                    tile_i=ti.name,
                    tile_j=tj.name,
                    area_sqdeg=float(inter_area),
                    frac_of_i=float(frac_i),
                    frac_of_j=float(frac_j),
                    pair_strength_min=float(pair_strength_min),
                    pair_strength_avg=float(pair_strength_avg),
                    center_dist_deg=float(center_dist),
                    bbox_touch=bbox_touch,
                    weight_area=float(inter_area),
                    weight_strength=float(pair_strength_min),
                    weight_hybrid=float(inter_area * pair_strength_min),
                )
            )
    overlaps.sort(key=lambda o: (o.pair_strength_min, o.area_sqdeg), reverse=True)
    return overlaps


def count_active_components(num_tiles: int, overlaps: Sequence[OverlapInfo], keys: Optional[set[Tuple[int, int]]] = None) -> Tuple[int, int]:
    uf = _UnionFind(num_tiles)
    active = np.zeros(num_tiles, dtype=bool)
    for ov in overlaps:
        key = _edge_key(ov.i, ov.j)
        if keys is not None and key not in keys:
            continue
        active[ov.i] = True
        active[ov.j] = True
        uf.union(ov.i, ov.j)
    active_count = int(np.sum(active))
    if active_count <= 1:
        return active_count, active_count
    roots = set()
    for idx, flag in enumerate(active):
        if flag:
            roots.add(uf.find(idx))
    return len(roots), active_count


def edge_weight_for_mode(ov: OverlapInfo, mode: str) -> float:
    if mode == "strength":
        return float(ov.weight_strength)
    if mode == "hybrid":
        return float(ov.weight_hybrid)
    return float(ov.weight_area)


def emulate_zemosaic_pruning(
    overlaps: Sequence[OverlapInfo],
    num_tiles: int,
    prune_k: int,
    force_prune: bool,
    weight_mode: str,
) -> GraphSummary:
    raw_pairs = len(overlaps)
    raw_components, active_tiles = count_active_components(num_tiles, overlaps)
    prune_k = max(0, int(prune_k))
    condition = force_prune or (raw_pairs > num_tiles * prune_k * 2 and num_tiles >= 10)

    if not condition:
        kept_keys = {_edge_key(ov.i, ov.j) for ov in overlaps}
        return GraphSummary(
            applied=False,
            fallback_used=False,
            raw_pairs=raw_pairs,
            kept_pairs=raw_pairs,
            topk_pairs=raw_pairs,
            raw_components=raw_components,
            topk_components=raw_components,
            final_components=raw_components,
            bridges_added=0,
            active_tiles=active_tiles,
            kept_keys=kept_keys,
            bridge_keys=set(),
            dropped_keys=set(),
        )

    neighbors: Dict[int, List[Tuple[float, Tuple[int, int]]]] = {i: [] for i in range(num_tiles)}
    edge_map: Dict[Tuple[int, int], OverlapInfo] = {}
    active = np.zeros(num_tiles, dtype=bool)
    for ov in overlaps:
        key = _edge_key(ov.i, ov.j)
        edge_map[key] = ov
        w = edge_weight_for_mode(ov, weight_mode)
        neighbors[ov.i].append((w, key))
        neighbors[ov.j].append((w, key))
        active[ov.i] = True
        active[ov.j] = True

    kept_keys: set[Tuple[int, int]] = set()
    for tile_id in range(num_tiles):
        if not active[tile_id]:
            continue
        neighbors[tile_id].sort(key=lambda x: (-x[0], x[1][0], x[1][1]))
        if neighbors[tile_id]:
            if prune_k == 0:
                kept_keys.add(neighbors[tile_id][0][1])
            else:
                for k_idx in range(min(prune_k, len(neighbors[tile_id]))):
                    kept_keys.add(neighbors[tile_id][k_idx][1])

    topk_pairs = len(kept_keys)
    topk_components, _ = count_active_components(num_tiles, overlaps, kept_keys)

    uf = _UnionFind(num_tiles)
    for key in kept_keys:
        uf.union(key[0], key[1])

    bridge_keys: set[Tuple[int, int]] = set()
    bridges_added = 0
    if active_tiles > 1 and topk_components > 1:
        overlaps_sorted = sorted(
            overlaps,
            key=lambda ov: (-edge_weight_for_mode(ov, weight_mode), _edge_key(ov.i, ov.j)[0], _edge_key(ov.i, ov.j)[1]),
        )
        for ov in overlaps_sorted:
            key = _edge_key(ov.i, ov.j)
            if key in kept_keys:
                continue
            if uf.union(ov.i, ov.j):
                kept_keys.add(key)
                bridge_keys.add(key)
                bridges_added += 1
                components_now, _ = count_active_components(num_tiles, overlaps, kept_keys)
                if components_now <= 1:
                    break

    final_components, _ = count_active_components(num_tiles, overlaps, kept_keys)
    fallback = active_tiles > 1 and final_components > 1
    if fallback:
        kept_keys = {_edge_key(ov.i, ov.j) for ov in overlaps}
        bridge_keys = set()
        final_components = raw_components
        kept_pairs = raw_pairs
    else:
        kept_pairs = len(kept_keys)

    all_keys = {_edge_key(ov.i, ov.j) for ov in overlaps}
    dropped_keys = all_keys - kept_keys
    return GraphSummary(
        applied=True,
        fallback_used=fallback,
        raw_pairs=raw_pairs,
        kept_pairs=kept_pairs,
        topk_pairs=topk_pairs,
        raw_components=raw_components,
        topk_components=topk_components,
        final_components=final_components,
        bridges_added=bridges_added,
        active_tiles=active_tiles,
        kept_keys=kept_keys,
        bridge_keys=bridge_keys,
        dropped_keys=dropped_keys,
    )


def build_stack_weight_proxy(tiles: Sequence[TileInfo]) -> np.ndarray:
    weights = np.ones(len(tiles), dtype=float)
    counts = np.array([
        float(tile.stack_count) if tile.stack_count is not None and tile.stack_count > 0 else np.nan
        for tile in tiles
    ], dtype=float)
    if np.any(np.isfinite(counts)):
        valid = counts[np.isfinite(counts) & (counts > 0)]
        med = float(np.median(valid)) if valid.size else 1.0
        if not math.isfinite(med) or med <= 0.0:
            med = 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            normed = counts / med
        normed = np.where(np.isfinite(normed) & (normed > 0.0), normed, 1.0)
        weights = np.sqrt(normed)
    return weights


def compute_tile_metrics(
    tiles: Sequence[TileInfo],
    overlaps: Sequence[OverlapInfo],
    graph: GraphSummary,
    prune_weight_mode: str,
) -> Dict[int, TileMetric]:
    metrics: Dict[int, TileMetric] = {
        t.index: TileMetric(
            index=t.index,
            name=t.name,
            raw_neighbors=0,
            kept_neighbors=0,
            bridge_neighbors=0,
            raw_best_local_overlap_frac=0.0,
            raw_mean_local_overlap_frac=0.0,
            kept_best_local_overlap_frac=0.0,
            kept_mean_local_overlap_frac=0.0,
            raw_best_pair_strength=0.0,
            raw_mean_pair_strength=0.0,
            kept_best_pair_strength=0.0,
            kept_mean_pair_strength=0.0,
            raw_connectivity=0.0,
            kept_connectivity=0.0,
            anchor_score_raw=0.0,
            anchor_score_kept=0.0,
            center_x_deg=t.center_xy_deg[0],
            center_y_deg=t.center_xy_deg[1],
            radial_distance_deg=0.0,
            area_sqdeg=t.area_sqdeg,
            width_deg=t.width_deg,
            height_deg=t.height_deg,
            date_ord=t.date_ord,
            days_from_median=None,
            retention_ratio=0.0,
            stack_weight_proxy=1.0,
        )
        for t in tiles
    }
    raw_local_vals: Dict[int, List[float]] = {t.index: [] for t in tiles}
    raw_pair_vals: Dict[int, List[float]] = {t.index: [] for t in tiles}
    kept_local_vals: Dict[int, List[float]] = {t.index: [] for t in tiles}
    kept_pair_vals: Dict[int, List[float]] = {t.index: [] for t in tiles}

    for ov in overlaps:
        key = _edge_key(ov.i, ov.j)
        w = edge_weight_for_mode(ov, prune_weight_mode)
        metrics[ov.i].raw_neighbors += 1
        metrics[ov.j].raw_neighbors += 1
        raw_local_vals[ov.i].append(ov.frac_of_i)
        raw_local_vals[ov.j].append(ov.frac_of_j)
        raw_pair_vals[ov.i].append(ov.pair_strength_min)
        raw_pair_vals[ov.j].append(ov.pair_strength_min)
        metrics[ov.i].raw_connectivity += w
        metrics[ov.j].raw_connectivity += w
        if key in graph.kept_keys:
            metrics[ov.i].kept_neighbors += 1
            metrics[ov.j].kept_neighbors += 1
            kept_local_vals[ov.i].append(ov.frac_of_i)
            kept_local_vals[ov.j].append(ov.frac_of_j)
            kept_pair_vals[ov.i].append(ov.pair_strength_min)
            kept_pair_vals[ov.j].append(ov.pair_strength_min)
            metrics[ov.i].kept_connectivity += w
            metrics[ov.j].kept_connectivity += w
        if key in graph.bridge_keys:
            metrics[ov.i].bridge_neighbors += 1
            metrics[ov.j].bridge_neighbors += 1

    cx = np.array([t.center_xy_deg[0] for t in tiles], dtype=float)
    cy = np.array([t.center_xy_deg[1] for t in tiles], dtype=float)
    x0 = float(np.median(cx)) if len(cx) else 0.0
    y0 = float(np.median(cy)) if len(cy) else 0.0

    date_vals = np.array([t.date_ord for t in tiles if t.date_ord is not None], dtype=float)
    date_ref = float(np.median(date_vals)) if date_vals.size else None
    weight_proxy = build_stack_weight_proxy(tiles)

    for idx, metric in metrics.items():
        rl = np.array(raw_local_vals[idx], dtype=float)
        rp = np.array(raw_pair_vals[idx], dtype=float)
        kl = np.array(kept_local_vals[idx], dtype=float)
        kp = np.array(kept_pair_vals[idx], dtype=float)
        if rl.size:
            metric.raw_best_local_overlap_frac = float(np.nanmax(rl))
            metric.raw_mean_local_overlap_frac = float(np.nanmean(rl))
        if rp.size:
            metric.raw_best_pair_strength = float(np.nanmax(rp))
            metric.raw_mean_pair_strength = float(np.nanmean(rp))
        if kl.size:
            metric.kept_best_local_overlap_frac = float(np.nanmax(kl))
            metric.kept_mean_local_overlap_frac = float(np.nanmean(kl))
        if kp.size:
            metric.kept_best_pair_strength = float(np.nanmax(kp))
            metric.kept_mean_pair_strength = float(np.nanmean(kp))
        metric.radial_distance_deg = float(math.hypot(metric.center_x_deg - x0, metric.center_y_deg - y0))
        metric.is_isolated_raw = metric.raw_neighbors == 0
        metric.is_isolated_kept = metric.kept_neighbors == 0
        metric.retention_ratio = float(metric.kept_neighbors / metric.raw_neighbors) if metric.raw_neighbors > 0 else 0.0
        metric.stack_weight_proxy = float(weight_proxy[idx]) if idx < len(weight_proxy) else 1.0
        metric.anchor_score_raw = float(metric.raw_connectivity * metric.stack_weight_proxy)
        metric.anchor_score_kept = float(metric.kept_connectivity * metric.stack_weight_proxy)
        if date_ref is not None and metric.date_ord is not None:
            metric.days_from_median = float(metric.date_ord - date_ref)

    if metrics:
        raw_neigh = np.array([m.raw_neighbors for m in metrics.values()], dtype=float)
        kept_neigh = np.array([m.kept_neighbors for m in metrics.values()], dtype=float)
        raw_best = np.array([m.raw_best_local_overlap_frac for m in metrics.values()], dtype=float)
        kept_best = np.array([m.kept_best_local_overlap_frac for m in metrics.values()], dtype=float)
        radial = np.array([m.radial_distance_deg for m in metrics.values()], dtype=float)
        raw_low_thr = float(np.nanpercentile(raw_neigh, 10)) if raw_neigh.size else 0.0
        kept_low_thr = float(np.nanpercentile(kept_neigh, 10)) if kept_neigh.size else 0.0
        raw_best_thr = float(np.nanpercentile(raw_best, 10)) if raw_best.size else 0.0
        kept_best_thr = float(np.nanpercentile(kept_best, 10)) if kept_best.size else 0.0
        peripheral_thr = float(np.nanpercentile(radial, 90)) if radial.size else 0.0
        if date_ref is not None:
            deltas = np.array([abs(m.days_from_median) if m.days_from_median is not None else 0.0 for m in metrics.values()], dtype=float)
            date_outlier_thr = float(np.nanpercentile(deltas, 90)) if deltas.size else 0.0
        else:
            date_outlier_thr = float("inf")
        for metric in metrics.values():
            metric.is_low_neighbor_raw = metric.raw_neighbors <= raw_low_thr
            metric.is_low_neighbor_kept = metric.kept_neighbors <= kept_low_thr
            metric.is_low_best_overlap_raw = metric.raw_best_local_overlap_frac <= raw_best_thr
            metric.is_low_best_overlap_kept = metric.kept_best_local_overlap_frac <= kept_best_thr
            metric.is_peripheral = metric.radial_distance_deg >= peripheral_thr
            if metric.days_from_median is not None:
                metric.is_date_outlier = abs(metric.days_from_median) >= date_outlier_thr

    return metrics


def choose_anchor(metrics: Dict[int, TileMetric], mode: str = "kept") -> int:
    best_idx = 0
    best_score = -float("inf")
    for idx, metric in metrics.items():
        score = metric.anchor_score_kept if mode == "kept" else metric.anchor_score_raw
        if score > best_score:
            best_score = score
            best_idx = idx
    return int(best_idx)


def write_overlap_csv(overlaps: Sequence[OverlapInfo], metrics: Dict[int, TileMetric], graph: GraphSummary, out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "index_i", "index_j", "tile_i", "tile_j", "area_sqdeg",
            "frac_of_i", "frac_of_j", "pair_strength_min", "pair_strength_avg",
            "center_dist_deg", "weight_area", "weight_strength", "weight_hybrid",
            "neighbors_raw_i", "neighbors_raw_j", "neighbors_kept_i", "neighbors_kept_j",
            "kept_final", "is_bridge", "bbox_touch",
        ])
        for ov in overlaps:
            key = _edge_key(ov.i, ov.j)
            writer.writerow([
                ov.i, ov.j, ov.tile_i, ov.tile_j,
                f"{ov.area_sqdeg:.8f}", f"{ov.frac_of_i:.6f}", f"{ov.frac_of_j:.6f}",
                f"{ov.pair_strength_min:.6f}", f"{ov.pair_strength_avg:.6f}",
                f"{ov.center_dist_deg:.6f}", f"{ov.weight_area:.8f}", f"{ov.weight_strength:.6f}", f"{ov.weight_hybrid:.8f}",
                metrics[ov.i].raw_neighbors, metrics[ov.j].raw_neighbors,
                metrics[ov.i].kept_neighbors, metrics[ov.j].kept_neighbors,
                int(key in graph.kept_keys), int(key in graph.bridge_keys), int(ov.bbox_touch),
            ])


def write_tile_metrics_csv(tiles: Sequence[TileInfo], metrics: Dict[int, TileMetric], out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "index", "tile_name", "path", "raw_neighbors", "kept_neighbors", "bridge_neighbors",
            "raw_best_local_overlap_frac", "raw_mean_local_overlap_frac",
            "kept_best_local_overlap_frac", "kept_mean_local_overlap_frac",
            "raw_best_pair_strength", "raw_mean_pair_strength",
            "kept_best_pair_strength", "kept_mean_pair_strength",
            "raw_connectivity", "kept_connectivity", "retention_ratio",
            "radial_distance_deg", "center_x_deg", "center_y_deg",
            "area_sqdeg", "width_deg", "height_deg", "pixel_scale_arcsec", "pa_deg",
            "filter", "exposure_s", "date_obs", "date_ord", "days_from_median",
            "stack_count", "stack_weight_proxy", "anchor_score_raw", "anchor_score_kept",
            "is_isolated_raw", "is_isolated_kept", "is_low_neighbor_raw", "is_low_neighbor_kept",
            "is_low_best_overlap_raw", "is_low_best_overlap_kept", "is_peripheral", "is_date_outlier",
        ])
        for tile in tiles:
            m = metrics[tile.index]
            writer.writerow([
                tile.index, tile.name, str(tile.path), m.raw_neighbors, m.kept_neighbors, m.bridge_neighbors,
                f"{m.raw_best_local_overlap_frac:.6f}", f"{m.raw_mean_local_overlap_frac:.6f}",
                f"{m.kept_best_local_overlap_frac:.6f}", f"{m.kept_mean_local_overlap_frac:.6f}",
                f"{m.raw_best_pair_strength:.6f}", f"{m.raw_mean_pair_strength:.6f}",
                f"{m.kept_best_pair_strength:.6f}", f"{m.kept_mean_pair_strength:.6f}",
                f"{m.raw_connectivity:.8f}", f"{m.kept_connectivity:.8f}", f"{m.retention_ratio:.6f}",
                f"{m.radial_distance_deg:.6f}", f"{m.center_x_deg:.6f}", f"{m.center_y_deg:.6f}",
                f"{m.area_sqdeg:.8f}", f"{m.width_deg:.6f}", f"{m.height_deg:.6f}",
                "" if tile.pixel_scale_arcsec is None else f"{tile.pixel_scale_arcsec:.6f}",
                "" if tile.pa_deg is None else f"{tile.pa_deg:.4f}",
                tile.filter_name, "" if tile.exposure_s is None else f"{tile.exposure_s:.6f}", tile.date_obs,
                "" if tile.date_ord is None else f"{tile.date_ord:.6f}",
                "" if m.days_from_median is None else f"{m.days_from_median:.3f}",
                "" if tile.stack_count is None else tile.stack_count,
                f"{m.stack_weight_proxy:.6f}", f"{m.anchor_score_raw:.8f}", f"{m.anchor_score_kept:.8f}",
                int(m.is_isolated_raw), int(m.is_isolated_kept), int(m.is_low_neighbor_raw), int(m.is_low_neighbor_kept),
                int(m.is_low_best_overlap_raw), int(m.is_low_best_overlap_kept), int(m.is_peripheral), int(m.is_date_outlier),
            ])


def _format_metric_line(tile: TileInfo, metric: TileMetric) -> str:
    date_txt = "n/a" if metric.days_from_median is None else f"{metric.days_from_median:+.1f} d"
    return (
        f"[{tile.index:03d}] {tile.name} | rawN={metric.raw_neighbors} keptN={metric.kept_neighbors} bridgeN={metric.bridge_neighbors} | "
        f"rawBest={metric.raw_best_local_overlap_frac:.4f} keptBest={metric.kept_best_local_overlap_frac:.4f} | "
        f"ret={metric.retention_ratio:.3f} | radial={metric.radial_distance_deg:.4f} deg | dateΔ={date_txt}"
    )


def write_tile_log(
    tiles: Sequence[TileInfo],
    overlaps: Sequence[OverlapInfo],
    metrics: Dict[int, TileMetric],
    graph: GraphSummary,
    out_log: Path,
    ref_ra_deg: float,
    ref_dec_deg: float,
    top_n_summary: int,
    prune_weight_mode: str,
) -> None:
    raw_isolated = [tile.name for tile in tiles if metrics[tile.index].is_isolated_raw]
    kept_isolated = [tile.name for tile in tiles if metrics[tile.index].is_isolated_kept]
    weak_kept = sorted(tiles, key=lambda t: (metrics[t.index].kept_neighbors, metrics[t.index].kept_best_local_overlap_frac, -metrics[t.index].raw_neighbors, metrics[t.index].radial_distance_deg))
    weak_raw = sorted(tiles, key=lambda t: (metrics[t.index].raw_neighbors, metrics[t.index].raw_best_local_overlap_frac, metrics[t.index].radial_distance_deg))
    peripheral = sorted(tiles, key=lambda t: metrics[t.index].radial_distance_deg, reverse=True)
    low_retention = sorted(tiles, key=lambda t: (metrics[t.index].retention_ratio, metrics[t.index].kept_neighbors, -metrics[t.index].raw_neighbors))
    date_outliers = sorted(tiles, key=lambda t: abs(metrics[t.index].days_from_median or 0.0), reverse=True)
    weak_pairs = sorted(overlaps, key=lambda ov: (ov.pair_strength_min, ov.pair_strength_avg, ov.center_dist_deg))
    strong_pairs = sorted(overlaps, key=lambda ov: (ov.pair_strength_min, ov.pair_strength_avg), reverse=True)
    dropped_pairs = sorted(
        [ov for ov in overlaps if _edge_key(ov.i, ov.j) in graph.dropped_keys],
        key=lambda ov: (ov.pair_strength_min, ov.area_sqdeg),
        reverse=True,
    )
    bridge_pairs = sorted(
        [ov for ov in overlaps if _edge_key(ov.i, ov.j) in graph.bridge_keys],
        key=lambda ov: edge_weight_for_mode(ov, prune_weight_mode),
        reverse=True,
    )

    raw_anchor = choose_anchor(metrics, mode="raw")
    kept_anchor = choose_anchor(metrics, mode="kept")
    raw_anchor_score = metrics[raw_anchor].anchor_score_raw if raw_anchor in metrics else 0.0
    kept_anchor_score = metrics[kept_anchor].anchor_score_kept if kept_anchor in metrics else 0.0

    raw_neigh = np.array([metrics[t.index].raw_neighbors for t in tiles], dtype=float) if tiles else np.array([], dtype=float)
    kept_neigh = np.array([metrics[t.index].kept_neighbors for t in tiles], dtype=float) if tiles else np.array([], dtype=float)
    strength_vals = np.array([ov.pair_strength_min for ov in overlaps], dtype=float) if overlaps else np.array([], dtype=float)
    dates = np.array([t.date_ord for t in tiles if t.date_ord is not None], dtype=float)

    with out_log.open("w", encoding="utf-8") as fh:
        fh.write("ZeMosaic master-tile overlap diagnostic (V3)\n")
        fh.write("=" * 92 + "\n")
        fh.write(f"Tiles scanned                      : {len(tiles)}\n")
        fh.write(f"Pairs with overlap (raw)           : {len(overlaps)}\n")
        fh.write(f"Pairs kept for retained graph      : {graph.kept_pairs}\n")
        fh.write(f"Projection reference               : RA={ref_ra_deg:.6f} deg  Dec={ref_dec_deg:.6f} deg\n")
        fh.write(f"Pruning emulation                  : {'on' if graph.applied else 'off'}\n")
        fh.write(f"Pruning weight mode                : {prune_weight_mode}\n")
        fh.write(f"Top-K pairs before bridges         : {graph.topk_pairs}\n")
        fh.write(f"Bridges added                      : {graph.bridges_added}\n")
        fh.write(f"Graph components raw/topK/final    : {graph.raw_components} / {graph.topk_components} / {graph.final_components}\n")
        fh.write(f"Pruning fallback to raw            : {'yes' if graph.fallback_used else 'no'}\n")
        fh.write(f"Active tiles in overlap graph      : {graph.active_tiles}\n")
        fh.write(f"Isolated tiles raw                 : {len(raw_isolated)}\n")
        fh.write(f"Isolated tiles kept                : {len(kept_isolated)}\n")
        if raw_neigh.size:
            fh.write(f"Raw neighbors per tile avg/med     : {float(np.nanmean(raw_neigh)):.2f} / {float(np.nanmedian(raw_neigh)):.2f}\n")
            fh.write(f"Raw neighbors per tile min/max     : {int(np.nanmin(raw_neigh))} / {int(np.nanmax(raw_neigh))}\n")
        if kept_neigh.size:
            fh.write(f"Kept neighbors per tile avg/med    : {float(np.nanmean(kept_neigh)):.2f} / {float(np.nanmedian(kept_neigh)):.2f}\n")
            fh.write(f"Kept neighbors per tile min/max    : {int(np.nanmin(kept_neigh))} / {int(np.nanmax(kept_neigh))}\n")
        if strength_vals.size:
            fh.write(f"Pair strength min/med/max          : {float(np.nanmin(strength_vals)):.4f} / {float(np.nanmedian(strength_vals)):.4f} / {float(np.nanmax(strength_vals)):.4f}\n")
        if dates.size:
            fh.write(f"Date span (days)                   : {float(np.nanmax(dates) - np.nanmin(dates)):.1f}\n")
            fh.write(f"Unique acquisition days            : {len(set(int(math.floor(v)) for v in dates))}\n")
        fh.write(f"Raw anchor candidate               : [{raw_anchor:03d}] {tiles[raw_anchor].name if tiles else '-'} score={raw_anchor_score:.6f}\n")
        fh.write(f"Kept anchor candidate              : [{kept_anchor:03d}] {tiles[kept_anchor].name if tiles else '-'} score={kept_anchor_score:.6f}\n")
        fh.write("\n")

        sections = [
            ("Lowest-raw-neighbor tiles", weak_raw),
            ("Lowest-kept-neighbor tiles", weak_kept),
            ("Lowest retention-ratio tiles", low_retention),
            ("Most peripheral tiles", peripheral),
            ("Largest date outliers", date_outliers),
        ]
        for title, seq in sections:
            fh.write(title + "\n")
            fh.write("-" * 92 + "\n")
            for tile in seq[:top_n_summary]:
                fh.write(_format_metric_line(tile, metrics[tile.index]) + "\n")
            fh.write("\n")

        fh.write("Strongest dropped pairs (potentially informative edges removed by top-K)\n")
        fh.write("-" * 92 + "\n")
        for ov in dropped_pairs[:top_n_summary]:
            fh.write(
                f"[{ov.i:03d}] {ov.tile_i} <-> [{ov.j:03d}] {ov.tile_j} | "
                f"pair_strength={ov.pair_strength_min:.4f} | area={ov.area_sqdeg:.6f} | dist={ov.center_dist_deg:.4f} deg\n"
            )
        fh.write("\n")

        fh.write("Bridge pairs added to restore connectivity\n")
        fh.write("-" * 92 + "\n")
        for ov in bridge_pairs[:top_n_summary]:
            fh.write(
                f"[{ov.i:03d}] {ov.tile_i} <-> [{ov.j:03d}] {ov.tile_j} | "
                f"pair_strength={ov.pair_strength_min:.4f} | area={ov.area_sqdeg:.6f} | dist={ov.center_dist_deg:.4f} deg\n"
            )
        fh.write("\n")

        fh.write("Weakest overlap pairs (raw graph)\n")
        fh.write("-" * 92 + "\n")
        for ov in weak_pairs[:top_n_summary]:
            fh.write(
                f"[{ov.i:03d}] {ov.tile_i} <-> [{ov.j:03d}] {ov.tile_j} | "
                f"pair_strength={ov.pair_strength_min:.4f} | avg_pair={ov.pair_strength_avg:.4f} | "
                f"frac_i={ov.frac_of_i:.4f} | frac_j={ov.frac_of_j:.4f} | dist={ov.center_dist_deg:.4f} deg\n"
            )
        fh.write("\n")

        fh.write("Strongest overlap pairs (raw graph)\n")
        fh.write("-" * 92 + "\n")
        for ov in strong_pairs[:top_n_summary]:
            fh.write(
                f"[{ov.i:03d}] {ov.tile_i} <-> [{ov.j:03d}] {ov.tile_j} | "
                f"pair_strength={ov.pair_strength_min:.4f} | avg_pair={ov.pair_strength_avg:.4f} | "
                f"frac_i={ov.frac_of_i:.4f} | frac_j={ov.frac_of_j:.4f} | dist={ov.center_dist_deg:.4f} deg\n"
            )
        fh.write("\n")

        fh.write("Per-tile characteristics\n")
        fh.write("-" * 92 + "\n")
        for tile in tiles:
            metric = metrics[tile.index]
            flags = []
            if metric.is_isolated_raw:
                flags.append("isolated-raw")
            if metric.is_isolated_kept:
                flags.append("isolated-kept")
            if metric.is_low_neighbor_raw:
                flags.append("low-neighbor-raw")
            if metric.is_low_neighbor_kept:
                flags.append("low-neighbor-kept")
            if metric.is_low_best_overlap_raw:
                flags.append("low-best-overlap-raw")
            if metric.is_low_best_overlap_kept:
                flags.append("low-best-overlap-kept")
            if metric.is_peripheral:
                flags.append("peripheral")
            if metric.is_date_outlier:
                flags.append("date-outlier")
            fh.write(f"[{tile.index:03d}] {tile.name}\n")
            fh.write(f"  path                       : {tile.path}\n")
            fh.write(f"  shape (H,W)                : {tile.shape_hw[0]} x {tile.shape_hw[1]}\n")
            fh.write(f"  center (RA,Dec)            : {tile.center_radec_deg[0]:.6f}, {tile.center_radec_deg[1]:.6f} deg\n")
            fh.write(f"  center (x,y)               : {tile.center_xy_deg[0]:.6f}, {tile.center_xy_deg[1]:.6f} deg\n")
            fh.write(f"  footprint WxH              : {tile.width_deg:.5f} x {tile.height_deg:.5f} deg\n")
            fh.write(f"  footprint area             : {tile.area_sqdeg:.8f} sq.deg\n")
            fh.write(f"  pixel scale                : {tile.pixel_scale_arcsec:.4f} arcsec/px\n" if tile.pixel_scale_arcsec is not None else "  pixel scale                : n/a\n")
            fh.write(f"  PA                         : {tile.pa_deg:.2f} deg\n" if tile.pa_deg is not None else "  PA                         : n/a\n")
            fh.write(f"  filter / exposure          : {tile.filter_name or '-'} / {tile.exposure_s if tile.exposure_s is not None else 'n/a'} s\n")
            fh.write(f"  date-obs                   : {tile.date_obs or '-'}\n")
            fh.write(f"  date delta                 : {metric.days_from_median:+.2f} days\n" if metric.days_from_median is not None else "  date delta                 : n/a\n")
            fh.write(f"  stack count                : {tile.stack_count if tile.stack_count is not None else 'n/a'}\n")
            fh.write(f"  raw neighbors              : {metric.raw_neighbors}\n")
            fh.write(f"  kept neighbors             : {metric.kept_neighbors}\n")
            fh.write(f"  bridge neighbors           : {metric.bridge_neighbors}\n")
            fh.write(f"  raw best local overlap     : {metric.raw_best_local_overlap_frac:.4f}\n")
            fh.write(f"  kept best local overlap    : {metric.kept_best_local_overlap_frac:.4f}\n")
            fh.write(f"  raw pair strength mean     : {metric.raw_mean_pair_strength:.4f}\n")
            fh.write(f"  kept pair strength mean    : {metric.kept_mean_pair_strength:.4f}\n")
            fh.write(f"  raw connectivity           : {metric.raw_connectivity:.6f}\n")
            fh.write(f"  kept connectivity          : {metric.kept_connectivity:.6f}\n")
            fh.write(f"  retention ratio            : {metric.retention_ratio:.4f}\n")
            fh.write(f"  stack weight proxy         : {metric.stack_weight_proxy:.4f}\n")
            fh.write(f"  anchor score raw/kept      : {metric.anchor_score_raw:.6f} / {metric.anchor_score_kept:.6f}\n")
            fh.write(f"  radial distance            : {metric.radial_distance_deg:.4f} deg\n")
            fh.write(f"  flags                      : {', '.join(flags) if flags else '-'}\n")
            hdr_bits = []
            for key, value in tile.raw_header_keys.items():
                if value is None or value == "":
                    continue
                hdr_bits.append(f"{key}={value}")
            fh.write("  header keys                : " + "; ".join(hdr_bits) + "\n\n")

        fh.write("Full pairwise overlap table is available in the CSV output.\n")


def _select_label_indices(tiles: Sequence[TileInfo], metrics: Dict[int, TileMetric], top_n: int) -> List[int]:
    weak_kept = sorted(tiles, key=lambda t: (metrics[t.index].kept_neighbors, metrics[t.index].kept_best_local_overlap_frac, metrics[t.index].retention_ratio, metrics[t.index].radial_distance_deg))
    low_retention = sorted(tiles, key=lambda t: (metrics[t.index].retention_ratio, metrics[t.index].kept_neighbors, -metrics[t.index].raw_neighbors))
    peripheral = sorted(tiles, key=lambda t: metrics[t.index].radial_distance_deg, reverse=True)
    date_outliers = sorted(tiles, key=lambda t: abs(metrics[t.index].days_from_median or 0.0), reverse=True)
    picked: List[int] = []
    for seq in (weak_kept, low_retention, peripheral, date_outliers):
        for tile in seq[:top_n]:
            if tile.index not in picked:
                picked.append(tile.index)
    return picked[: max(top_n, 1)]


def _scalar_for_dates_or_fallback(tiles: Sequence[TileInfo], metrics: Dict[int, TileMetric]) -> Tuple[np.ndarray, str]:
    date_vals = np.array([t.date_ord if t.date_ord is not None else np.nan for t in tiles], dtype=float)
    finite = np.isfinite(date_vals)
    if np.any(finite):
        return date_vals, "Acquisition date (ordinal days)"
    fallback = np.array([metrics[t.index].retention_ratio for t in tiles], dtype=float)
    return fallback, "Retention ratio"


def plot_overview_v3(
    tiles: Sequence[TileInfo],
    overlaps: Sequence[OverlapInfo],
    metrics: Dict[int, TileMetric],
    graph: GraphSummary,
    out_png: Path,
    annotate: bool,
    draw_overlap_links: bool,
    label_top_n: int,
) -> None:
    if not tiles:
        raise RuntimeError("No tiles to plot")

    raw_neighbor_values = np.array([metrics[t.index].raw_neighbors for t in tiles], dtype=float)
    kept_neighbor_values = np.array([metrics[t.index].kept_neighbors for t in tiles], dtype=float)
    edge_values = np.array([ov.pair_strength_min for ov in overlaps], dtype=float) if overlaps else np.array([0.0], dtype=float)
    date_values, date_label = _scalar_for_dates_or_fallback(tiles, metrics)

    raw_norm = Normalize(vmin=float(np.nanmin(raw_neighbor_values)), vmax=float(np.nanmax(raw_neighbor_values) if np.nanmax(raw_neighbor_values) > np.nanmin(raw_neighbor_values) else np.nanmin(raw_neighbor_values) + 1.0))
    kept_norm = Normalize(vmin=float(np.nanmin(kept_neighbor_values)), vmax=float(np.nanmax(kept_neighbor_values) if np.nanmax(kept_neighbor_values) > np.nanmin(kept_neighbor_values) else np.nanmin(kept_neighbor_values) + 1.0))
    edge_norm = Normalize(vmin=float(np.nanmin(edge_values)), vmax=float(np.nanmax(edge_values) if np.nanmax(edge_values) > np.nanmin(edge_values) else np.nanmin(edge_values) + 1.0))
    finite_date = date_values[np.isfinite(date_values)]
    if finite_date.size:
        date_norm = Normalize(vmin=float(np.nanmin(finite_date)), vmax=float(np.nanmax(finite_date) if np.nanmax(finite_date) > np.nanmin(finite_date) else np.nanmin(finite_date) + 1.0))
    else:
        date_norm = Normalize(vmin=0.0, vmax=1.0)

    tile_cmap = plt.get_cmap("viridis")
    edge_cmap = plt.get_cmap("plasma")
    date_cmap = plt.get_cmap("cividis")

    label_indices = set(_select_label_indices(tiles, metrics, label_top_n)) if annotate else set()
    kept_anchor = choose_anchor(metrics, mode="kept")

    fig, axes = plt.subplots(2, 2, figsize=(17, 14), dpi=170)
    ax0, ax1, ax2, ax3 = axes.ravel()

    def _add_tile_panel(ax, scalar_values: np.ndarray, norm: Normalize, cmap, title: str, cbar_label: str, marker_mode: str = "raw") -> None:
        patches = [MplPolygon(tile.footprint_xy_deg, closed=True) for tile in tiles]
        pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="black", linewidth=0.65, alpha=0.42)
        pc.set_array(scalar_values)
        ax.add_collection(pc)
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04).set_label(cbar_label)
        for tile, val in zip(tiles, scalar_values):
            cx, cy = tile.center_xy_deg
            if marker_mode == "raw":
                size = max(10.0, 6.0 + 0.8 * metrics[tile.index].raw_neighbors)
            else:
                size = max(10.0, 6.0 + 1.0 * metrics[tile.index].kept_neighbors)
            ax.scatter([cx], [cy], s=size, c=[cmap(norm(val if np.isfinite(val) else norm.vmin))], edgecolors="black", linewidths=0.25, alpha=0.9)
            if tile.index in label_indices:
                ax.text(cx, cy, str(tile.index), fontsize=8, ha="center", va="center", color="black")
        ax.set_title(title)

    _add_tile_panel(ax0, raw_neighbor_values, raw_norm, tile_cmap, "Raw graph: footprints colored by raw neighbor count", "Raw neighbors per tile", marker_mode="raw")

    patches1 = [MplPolygon(tile.footprint_xy_deg, closed=True) for tile in tiles]
    pc1 = PatchCollection(patches1, cmap=tile_cmap, norm=raw_norm, edgecolor="black", linewidth=0.5, alpha=0.18)
    pc1.set_array(raw_neighbor_values)
    ax1.add_collection(pc1)
    if draw_overlap_links and overlaps:
        lines = []
        strengths = []
        for ov in overlaps:
            ti = tiles[ov.i]
            tj = tiles[ov.j]
            lines.append([(ti.center_xy_deg[0], ti.center_xy_deg[1]), (tj.center_xy_deg[0], tj.center_xy_deg[1])])
            strengths.append(ov.pair_strength_min)
        lc = LineCollection(lines, cmap=edge_cmap, norm=edge_norm, linewidths=0.30, alpha=0.50)
        lc.set_array(np.array(strengths, dtype=float))
        ax1.add_collection(lc)
    fig.colorbar(ScalarMappable(norm=edge_norm, cmap=edge_cmap), ax=ax1, fraction=0.046, pad=0.04).set_label("Raw overlap strength = min(frac_i, frac_j)")
    for tile in tiles:
        cx, cy = tile.center_xy_deg
        ax1.plot(cx, cy, marker="+", markersize=4, color="black", alpha=0.65)
        if tile.index in label_indices:
            ax1.text(cx, cy, str(tile.index), fontsize=8, ha="center", va="center", color="black")
    ax1.set_title("Raw overlap network colored by conservative overlap strength")

    _add_tile_panel(ax2, kept_neighbor_values, kept_norm, tile_cmap, "Retained graph after top-K prune + bridges", "Kept neighbors per tile", marker_mode="kept")
    if draw_overlap_links and overlaps:
        kept_lines = []
        kept_strengths = []
        bridge_lines = []
        for ov in overlaps:
            key = _edge_key(ov.i, ov.j)
            ti = tiles[ov.i]
            tj = tiles[ov.j]
            line = [(ti.center_xy_deg[0], ti.center_xy_deg[1]), (tj.center_xy_deg[0], tj.center_xy_deg[1])]
            if key in graph.bridge_keys:
                bridge_lines.append(line)
            elif key in graph.kept_keys:
                kept_lines.append(line)
                kept_strengths.append(ov.pair_strength_min)
        if kept_lines:
            lc_kept = LineCollection(kept_lines, cmap=edge_cmap, norm=edge_norm, linewidths=0.55, alpha=0.80)
            lc_kept.set_array(np.array(kept_strengths, dtype=float))
            ax2.add_collection(lc_kept)
        if bridge_lines:
            lc_bridge = LineCollection(bridge_lines, colors=["#ff8800"], linewidths=1.10, alpha=0.95)
            ax2.add_collection(lc_bridge)
    anchor_tile = tiles[kept_anchor]
    ax2.scatter([anchor_tile.center_xy_deg[0]], [anchor_tile.center_xy_deg[1]], s=220, marker="*", c=["red"], edgecolors="black", linewidths=0.8, zorder=6)
    ax2.text(anchor_tile.center_xy_deg[0], anchor_tile.center_xy_deg[1], f"A{kept_anchor}", fontsize=9, ha="left", va="bottom", color="red")

    _add_tile_panel(ax3, date_values, date_norm, date_cmap, "Acquisition chronology / heterogeneity map", date_label, marker_mode="kept")
    for tile in tiles:
        if metrics[tile.index].is_date_outlier:
            ax3.scatter([tile.center_xy_deg[0]], [tile.center_xy_deg[1]], s=90, marker="o", facecolors="none", edgecolors="red", linewidths=1.0, alpha=0.9)

    for ax in (ax0, ax1, ax2, ax3):
        ax.set_xlabel("Delta RA * cos(Dec₀) [deg]")
        ax.set_ylabel("Delta Dec [deg]")
        ax.grid(True, alpha=0.22)
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_xaxis()

    prune_status = "off" if not graph.applied else ("fallback raw" if graph.fallback_used else f"topK+bridges kept={graph.kept_pairs}")
    fig.suptitle(f"Master-tile overlap diagnostic V3 — raw={graph.raw_pairs} kept={graph.kept_pairs} prune={prune_status}", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def run_diagnostic(
    config: DiagnosticConfig,
    progress: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> DiagnosticResult:
    def emit(msg: str) -> None:
        if progress:
            progress(msg)

    if not config.inputs:
        raise ValueError("No input paths were provided.")

    emit("Scanning input paths...")
    files = sorted(set(iter_fits_files(config.inputs, recursive=config.recursive)))
    if cancel_check and cancel_check():
        raise CancelledError("Cancelled by user")
    if not files:
        raise RuntimeError("No FITS files found.")

    emit(f"FITS files found: {len(files)}")
    tiles = build_tile_infos(
        files,
        ref_ra_deg=config.ref_ra,
        ref_dec_deg=config.ref_dec,
        progress=progress,
        cancel_check=cancel_check,
    )
    if not tiles:
        raise RuntimeError("No valid celestial WCS could be extracted from the provided FITS headers.")

    ref_ra = float(np.median([t.center_radec_deg[0] for t in tiles])) if config.ref_ra is None else float(config.ref_ra)
    ref_dec = float(np.median([t.center_radec_deg[1] for t in tiles])) if config.ref_dec is None else float(config.ref_dec)

    if config.ref_ra is not None or config.ref_dec is not None:
        emit("Reprojecting footprints around custom reference...")
        updated_tiles: List[TileInfo] = []
        for t in tiles:
            if cancel_check and cancel_check():
                raise CancelledError("Cancelled by user")
            fp_xy = project_radec_to_plane_deg(t.footprint_radec_deg, ref_ra, ref_dec)
            center_xy = project_radec_to_plane_deg(np.array([[t.center_radec_deg[0], t.center_radec_deg[1]]], dtype=float), ref_ra, ref_dec)[0]
            area_sqdeg = polygon_area(fp_xy)
            width_deg, height_deg = estimate_width_height_deg(fp_xy)
            updated_tiles.append(
                TileInfo(
                    index=t.index,
                    path=t.path,
                    name=t.name,
                    shape_hw=t.shape_hw,
                    center_radec_deg=t.center_radec_deg,
                    center_xy_deg=(float(center_xy[0]), float(center_xy[1])),
                    pa_deg=t.pa_deg,
                    pixel_scale_arcsec=t.pixel_scale_arcsec,
                    footprint_radec_deg=t.footprint_radec_deg,
                    footprint_xy_deg=fp_xy,
                    area_sqdeg=area_sqdeg,
                    width_deg=width_deg,
                    height_deg=height_deg,
                    filter_name=t.filter_name,
                    exposure_s=t.exposure_s,
                    date_obs=t.date_obs,
                    date_ord=t.date_ord,
                    stack_count=t.stack_count,
                    raw_header_keys=t.raw_header_keys,
                )
            )
        tiles = updated_tiles

    emit("Computing pairwise overlaps...")
    overlaps = analyze_overlaps(
        tiles,
        min_fraction=max(0.0, float(config.min_overlap_frac)),
        progress=progress,
        cancel_check=cancel_check,
    )

    if config.emulate_pruning:
        emit(f"Emulating ZeMosaic top-K pruning (K={max(0, int(config.prune_k))}, weight={config.prune_weight_mode})...")
        graph = emulate_zemosaic_pruning(
            overlaps,
            num_tiles=len(tiles),
            prune_k=max(0, int(config.prune_k)),
            force_prune=bool(config.force_prune),
            weight_mode=str(config.prune_weight_mode or 'area').strip().lower(),
        )
    else:
        graph = GraphSummary(
            applied=False,
            fallback_used=False,
            raw_pairs=len(overlaps),
            kept_pairs=len(overlaps),
            topk_pairs=len(overlaps),
            raw_components=count_active_components(len(tiles), overlaps)[0],
            topk_components=count_active_components(len(tiles), overlaps)[0],
            final_components=count_active_components(len(tiles), overlaps)[0],
            bridges_added=0,
            active_tiles=count_active_components(len(tiles), overlaps)[1],
            kept_keys={_edge_key(ov.i, ov.j) for ov in overlaps},
            bridge_keys=set(),
            dropped_keys=set(),
        )

    metrics = compute_tile_metrics(tiles, overlaps, graph, str(config.prune_weight_mode or 'area').strip().lower())

    outdir = config.outdir.expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = config.prefix.strip() or "master_tiles"
    out_overview_png = outdir / f"{prefix}_v3_overview.png"
    out_log = outdir / f"{prefix}_diagnostic_v3.log"
    out_overlaps_csv = outdir / f"{prefix}_overlaps_v3.csv"
    out_tiles_csv = outdir / f"{prefix}_tiles_v3.csv"

    emit("Writing outputs...")
    plot_overview_v3(
        tiles,
        overlaps,
        metrics,
        graph,
        out_overview_png,
        annotate=config.annotate,
        draw_overlap_links=config.draw_links,
        label_top_n=max(1, int(config.label_top_n)),
    )
    write_tile_log(tiles, overlaps, metrics, graph, out_log, ref_ra, ref_dec, max(1, int(config.top_n_summary)), str(config.prune_weight_mode or 'area').strip().lower())
    write_overlap_csv(overlaps, metrics, graph, out_overlaps_csv)
    write_tile_metrics_csv(tiles, metrics, out_tiles_csv)

    emit(f"Tiles analyzed : {len(tiles)}")
    emit(f"Raw overlaps   : {len(overlaps)}")
    emit(f"Kept overlaps  : {graph.kept_pairs}")
    emit(f"Overview PNG   : {out_overview_png}")
    emit(f"LOG            : {out_log}")
    emit(f"Tiles CSV      : {out_tiles_csv}")
    emit(f"Overlaps CSV   : {out_overlaps_csv}")

    return DiagnosticResult(
        tile_count=len(tiles),
        overlap_count=len(overlaps),
        kept_overlap_count=graph.kept_pairs,
        out_overview_png=out_overview_png,
        out_log=out_log,
        out_overlaps_csv=out_overlaps_csv,
        out_tiles_csv=out_tiles_csv,
    )


class DiagnosticWorker(QThread):
    log_line = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, config: DiagnosticConfig, parent: Optional[QWidget] = None) -> None:  # type: ignore[name-defined]
        super().__init__(parent)
        self.config = config
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    def run(self) -> None:
        try:
            result = run_diagnostic(self.config, progress=self.log_line.emit, cancel_check=self._is_cancelled)
            if self._cancel_requested:
                self.cancelled.emit()
            else:
                self.succeeded.emit(result)
        except CancelledError:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())


if HAVE_QT:
    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.worker: Optional[DiagnosticWorker] = None
            self.last_result: Optional[DiagnosticResult] = None
            self.setWindowTitle("ZeMosaic Master Tile Overlap Diagnostic V3")
            self.resize(1120, 820)

            central = QWidget(self)
            self.setCentralWidget(central)
            root = QVBoxLayout(central)

            splitter = QSplitter(Qt.Orientation.Vertical)
            root.addWidget(splitter)

            top = QWidget()
            top_layout = QVBoxLayout(top)
            splitter.addWidget(top)

            input_group = QGroupBox("Inputs")
            ig = QVBoxLayout(input_group)
            self.input_list = QListWidget()
            self.input_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
            ig.addWidget(self.input_list)
            btn_row = QHBoxLayout()
            self.add_files_btn = QPushButton("Add FITS files...")
            self.add_dir_btn = QPushButton("Add folder...")
            self.remove_btn = QPushButton("Remove selected")
            self.clear_btn = QPushButton("Clear")
            btn_row.addWidget(self.add_files_btn)
            btn_row.addWidget(self.add_dir_btn)
            btn_row.addWidget(self.remove_btn)
            btn_row.addWidget(self.clear_btn)
            ig.addLayout(btn_row)
            self.recursive_check = QCheckBox("Recursive search inside folders")
            self.recursive_check.setChecked(True)
            ig.addWidget(self.recursive_check)
            top_layout.addWidget(input_group)

            options_group = QGroupBox("Options")
            og = QGridLayout(options_group)
            self.outdir_edit = QLineEdit(str(Path("tile_overlap_diag")))
            self.outdir_btn = QPushButton("Browse...")
            self.prefix_edit = QLineEdit("master_tiles")
            self.min_overlap_spin = QDoubleSpinBox()
            self.min_overlap_spin.setRange(0.0, 1.0)
            self.min_overlap_spin.setDecimals(3)
            self.min_overlap_spin.setSingleStep(0.01)
            self.min_overlap_spin.setValue(0.0)
            self.ref_ra_edit = QLineEdit("")
            self.ref_dec_edit = QLineEdit("")
            self.annotate_check = QCheckBox("Annotate suspicious/peripheral indices")
            self.annotate_check.setChecked(False)
            self.links_check = QCheckBox("Draw overlap links")
            self.links_check.setChecked(True)
            self.top_n_spin = QDoubleSpinBox()
            self.top_n_spin.setRange(1, 200)
            self.top_n_spin.setDecimals(0)
            self.top_n_spin.setValue(20)
            self.label_top_n_spin = QDoubleSpinBox()
            self.label_top_n_spin.setRange(1, 100)
            self.label_top_n_spin.setDecimals(0)
            self.label_top_n_spin.setValue(16)
            self.prune_enable_check = QCheckBox("Emulate ZeMosaic top-K pruning")
            self.prune_enable_check.setChecked(True)
            self.force_prune_check = QCheckBox("Force pruning even if heuristic would skip it")
            self.force_prune_check.setChecked(False)
            self.prune_k_spin = QDoubleSpinBox()
            self.prune_k_spin.setRange(0, 128)
            self.prune_k_spin.setDecimals(0)
            self.prune_k_spin.setValue(DEFAULT_PRUNE_K)
            self.prune_weight_combo = QComboBox()
            self.prune_weight_combo.addItems(["area", "strength", "hybrid"])
            self.prune_weight_combo.setCurrentText("area")

            og.addWidget(QLabel("Output folder:"), 0, 0)
            og.addWidget(self.outdir_edit, 0, 1)
            og.addWidget(self.outdir_btn, 0, 2)
            og.addWidget(QLabel("Filename prefix:"), 1, 0)
            og.addWidget(self.prefix_edit, 1, 1, 1, 2)
            og.addWidget(QLabel("Min overlap fraction:"), 2, 0)
            og.addWidget(self.min_overlap_spin, 2, 1)
            og.addWidget(QLabel("Summary top-N:"), 2, 2)
            og.addWidget(self.top_n_spin, 2, 3)
            og.addWidget(QLabel("Label top-N:"), 3, 2)
            og.addWidget(self.label_top_n_spin, 3, 3)
            og.addWidget(QLabel("Ref. RA (deg, optional):"), 3, 0)
            og.addWidget(self.ref_ra_edit, 3, 1)
            og.addWidget(QLabel("Ref. Dec (deg, optional):"), 4, 0)
            og.addWidget(self.ref_dec_edit, 4, 1)
            og.addWidget(self.prune_enable_check, 4, 2, 1, 2)
            og.addWidget(QLabel("Top-K per tile:"), 5, 0)
            og.addWidget(self.prune_k_spin, 5, 1)
            og.addWidget(QLabel("Prune weight mode:"), 5, 2)
            og.addWidget(self.prune_weight_combo, 5, 3)
            og.addWidget(self.force_prune_check, 6, 0, 1, 2)
            og.addWidget(self.annotate_check, 6, 2, 1, 1)
            og.addWidget(self.links_check, 6, 3, 1, 1)
            top_layout.addWidget(options_group)

            actions_group = QGroupBox("Run")
            ag = QHBoxLayout(actions_group)
            self.run_btn = QPushButton("Run diagnostic")
            self.cancel_btn = QPushButton("Cancel")
            self.cancel_btn.setEnabled(False)
            self.open_out_btn = QPushButton("Open output folder")
            self.open_out_btn.setEnabled(False)
            ag.addWidget(self.run_btn)
            ag.addWidget(self.cancel_btn)
            ag.addWidget(self.open_out_btn)
            top_layout.addWidget(actions_group)

            bottom = QWidget()
            bottom_layout = QVBoxLayout(bottom)
            splitter.addWidget(bottom)
            self.status_label = QLabel("Ready.")
            bottom_layout.addWidget(self.status_label)
            self.log_edit = QPlainTextEdit()
            self.log_edit.setReadOnly(True)
            bottom_layout.addWidget(self.log_edit)

            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)

            self.add_files_btn.clicked.connect(self.add_files)
            self.add_dir_btn.clicked.connect(self.add_directory)
            self.remove_btn.clicked.connect(self.remove_selected)
            self.clear_btn.clicked.connect(self.input_list.clear)
            self.outdir_btn.clicked.connect(self.pick_outdir)
            self.run_btn.clicked.connect(self.run_diagnostic)
            self.cancel_btn.clicked.connect(self.cancel_diagnostic)
            self.open_out_btn.clicked.connect(self.open_output_folder)

        def append_log(self, text: str) -> None:
            if not text:
                return
            self.log_edit.appendPlainText(text.rstrip())
            self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())

        def add_files(self) -> None:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select FITS files",
                "",
                "FITS files (*.fits *.fit *.fts *.fz);;All files (*)",
            )
            for path in files:
                self._add_unique_item(path)

        def add_directory(self) -> None:
            folder = QFileDialog.getExistingDirectory(self, "Select folder")
            if folder:
                self._add_unique_item(folder)

        def _add_unique_item(self, path: str) -> None:
            norm = str(Path(path).expanduser())
            existing = {self.input_list.item(i).text() for i in range(self.input_list.count())}
            if norm not in existing:
                self.input_list.addItem(QListWidgetItem(norm))

        def remove_selected(self) -> None:
            for item in self.input_list.selectedItems():
                row = self.input_list.row(item)
                self.input_list.takeItem(row)

        def pick_outdir(self) -> None:
            folder = QFileDialog.getExistingDirectory(self, "Select output folder", self.outdir_edit.text().strip() or "")
            if folder:
                self.outdir_edit.setText(folder)

        def _parse_optional_float(self, edit: QLineEdit, label: str) -> Optional[float]:
            text = edit.text().strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                raise ValueError(f"{label} must be a valid number.")

        def _collect_inputs(self) -> List[Path]:
            return [Path(self.input_list.item(i).text()).expanduser() for i in range(self.input_list.count())]

        def build_config(self) -> DiagnosticConfig:
            inputs = self._collect_inputs()
            if not inputs:
                raise ValueError("Please add at least one FITS file or folder.")
            return DiagnosticConfig(
                inputs=inputs,
                recursive=self.recursive_check.isChecked(),
                outdir=Path(self.outdir_edit.text().strip() or "tile_overlap_diag").expanduser(),
                prefix=self.prefix_edit.text().strip() or "master_tiles",
                min_overlap_frac=float(self.min_overlap_spin.value()),
                ref_ra=self._parse_optional_float(self.ref_ra_edit, "Reference RA"),
                ref_dec=self._parse_optional_float(self.ref_dec_edit, "Reference Dec"),
                annotate=self.annotate_check.isChecked(),
                draw_links=self.links_check.isChecked(),
                top_n_summary=int(self.top_n_spin.value()),
                label_top_n=int(self.label_top_n_spin.value()),
                emulate_pruning=self.prune_enable_check.isChecked(),
                prune_k=int(self.prune_k_spin.value()),
                force_prune=self.force_prune_check.isChecked(),
                prune_weight_mode=self.prune_weight_combo.currentText().strip().lower() or "area",
            )

        def set_running_state(self, running: bool) -> None:
            self.run_btn.setEnabled(not running)
            self.cancel_btn.setEnabled(running)
            self.add_files_btn.setEnabled(not running)
            self.add_dir_btn.setEnabled(not running)
            self.remove_btn.setEnabled(not running)
            self.clear_btn.setEnabled(not running)

        def run_diagnostic(self) -> None:
            try:
                config = self.build_config()
            except Exception as exc:
                QMessageBox.warning(self, "Invalid configuration", str(exc))
                return

            self.log_edit.clear()
            self.last_result = None
            self.open_out_btn.setEnabled(False)
            self.append_log("Starting diagnostic V3...")
            self.status_label.setText("Running...")
            self.set_running_state(True)

            self.worker = DiagnosticWorker(config, self)
            self.worker.log_line.connect(self.append_log)
            self.worker.succeeded.connect(self.on_success)
            self.worker.failed.connect(self.on_failure)
            self.worker.cancelled.connect(self.on_cancelled)
            self.worker.start()

        def cancel_diagnostic(self) -> None:
            if self.worker is not None:
                self.append_log("Cancellation requested...")
                self.status_label.setText("Cancelling...")
                self.worker.request_cancel()

        def on_success(self, result: DiagnosticResult) -> None:
            self.last_result = result
            self.status_label.setText(f"Done. {result.tile_count} tiles, {result.overlap_count} raw overlaps, {result.kept_overlap_count} kept.")
            self.append_log("Diagnostic completed successfully.")
            self.set_running_state(False)
            self.open_out_btn.setEnabled(True)
            self.worker = None
            QMessageBox.information(
                self,
                "Diagnostic finished",
                f"Tiles analyzed: {result.tile_count}\n"
                f"Raw overlaps: {result.overlap_count}\n"
                f"Kept overlaps: {result.kept_overlap_count}\n\n"
                f"Overview PNG: {result.out_overview_png}\n"
                f"LOG: {result.out_log}\n"
                f"Tiles CSV: {result.out_tiles_csv}\n"
                f"Overlaps CSV: {result.out_overlaps_csv}",
            )

        def on_failure(self, details: str) -> None:
            self.status_label.setText("Failed.")
            self.append_log(details)
            self.set_running_state(False)
            self.open_out_btn.setEnabled(False)
            self.worker = None
            QMessageBox.critical(self, "Diagnostic failed", "The diagnostic failed. See the log panel for details.")

        def on_cancelled(self) -> None:
            self.status_label.setText("Cancelled.")
            self.append_log("Diagnostic cancelled.")
            self.set_running_state(False)
            self.worker = None

        def open_output_folder(self) -> None:
            target = None
            if self.last_result is not None:
                target = self.last_result.out_overview_png.parent
            else:
                text = self.outdir_edit.text().strip()
                if text:
                    target = Path(text).expanduser()
            if target is not None:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Master tile overlap diagnostic V3 (GUI + CLI)")
    parser.add_argument("inputs", nargs="*", help="Input FITS files and/or directories")
    parser.add_argument("--recursive", action="store_true", help="Recurse into input directories")
    parser.add_argument("--outdir", default="tile_overlap_diag", help="Output directory")
    parser.add_argument("--prefix", default="master_tiles", help="Output filename prefix")
    parser.add_argument("--min-overlap-frac", type=float, default=0.0, help="Minimum pairwise overlap fraction to keep")
    parser.add_argument("--ref-ra", type=float, default=None, help="Override projection reference RA in deg")
    parser.add_argument("--ref-dec", type=float, default=None, help="Override projection reference Dec in deg")
    parser.add_argument("--annotate", action="store_true", help="Annotate suspicious/peripheral tile indices on the PNG")
    parser.add_argument("--no-links", action="store_true", help="Disable overlap link lines in the PNG")
    parser.add_argument("--top-n-summary", type=int, default=20, help="Top-N rows for summary sections in the log")
    parser.add_argument("--label-top-n", type=int, default=16, help="Maximum number of labeled suspicious/peripheral tiles")
    parser.add_argument("--no-prune", action="store_true", help="Disable pruning emulation")
    parser.add_argument("--force-prune", action="store_true", help="Force pruning even if the heuristic would skip it")
    parser.add_argument("--prune-k", type=int, default=DEFAULT_PRUNE_K, help="Top-K neighbors per tile for pruning emulation")
    parser.add_argument("--prune-weight", choices=["area", "strength", "hybrid"], default="area", help="Weight used for pruning emulation")
    parser.add_argument("--gui", action="store_true", help="Force GUI mode")
    return parser


def run_cli_from_args(args: argparse.Namespace) -> int:
    config = DiagnosticConfig(
        inputs=[Path(p).expanduser() for p in args.inputs],
        recursive=args.recursive,
        outdir=Path(args.outdir).expanduser(),
        prefix=args.prefix,
        min_overlap_frac=max(0.0, float(args.min_overlap_frac)),
        ref_ra=args.ref_ra,
        ref_dec=args.ref_dec,
        annotate=bool(args.annotate),
        draw_links=not args.no_links,
        top_n_summary=max(1, int(args.top_n_summary)),
        label_top_n=max(1, int(args.label_top_n)),
        emulate_pruning=not args.no_prune,
        prune_k=max(0, int(args.prune_k)),
        force_prune=bool(args.force_prune),
        prune_weight_mode=str(args.prune_weight).strip().lower(),
    )
    try:
        run_diagnostic(config, progress=print)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def run_gui() -> int:
    if not HAVE_QT:
        print("ERROR: PySide6 is required for GUI mode.", file=sys.stderr)
        return 2
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.gui or not args.inputs:
        return run_gui()
    return run_cli_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
