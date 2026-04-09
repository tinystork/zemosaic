#!/usr/bin/env python3
"""Standalone GUI + CLI diagnostic for ZeMosaic master-tile overlaps.

Features
--------
- Scans FITS files / directories
- Extracts celestial WCS from headers
- Computes tile footprints and pairwise overlaps
- Writes PNG / LOG / CSV outputs
- PySide6 GUI with file/folder selection, live log, graceful cancel request
- CLI mode still available when inputs are provided on the command line
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

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
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
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


@dataclass
class TileInfo:
    index: int
    path: Path
    name: str
    shape_hw: Tuple[int, int]
    center_radec_deg: Tuple[float, float]
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
    center_dist_deg: float
    bbox_touch: bool


@dataclass
class DiagnosticConfig:
    inputs: List[Path]
    recursive: bool = True
    outdir: Path = Path("tile_overlap_diag")
    prefix: str = "master_tiles"
    min_overlap_frac: float = 0.0
    ref_ra: Optional[float] = None
    ref_dec: Optional[float] = None
    annotate: bool = True
    draw_links: bool = True


@dataclass
class DiagnosticResult:
    tile_count: int
    overlap_count: int
    out_png: Path
    out_log: Path
    out_csv: Path


class CancelledError(RuntimeError):
    pass


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
        tiles.append(
            TileInfo(
                index=idx,
                path=path,
                name=path.name,
                shape_hw=(int(height), int(width)),
                center_radec_deg=(float(ra0), float(dec0)),
                pa_deg=compute_position_angle_deg(wcs_obj, width, height),
                pixel_scale_arcsec=scale_arcsec,
                footprint_radec_deg=fp_radec,
                footprint_xy_deg=fp_xy,
                area_sqdeg=float(area_sqdeg),
                width_deg=float(width_deg),
                height_deg=float(height_deg),
                filter_name=str(header.get("FILTER", header.get("FILTNAME", "")) or ""),
                exposure_s=exposure_s,
                date_obs=str(header.get("DATE-OBS", "") or ""),
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
                    center_dist_deg=float(center_dist),
                    bbox_touch=bbox_touch,
                )
            )
    overlaps.sort(key=lambda o: max(o.frac_of_i, o.frac_of_j), reverse=True)
    return overlaps


def write_overlap_csv(overlaps: Sequence[OverlapInfo], out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "index_i", "index_j", "tile_i", "tile_j", "area_sqdeg",
            "frac_of_i", "frac_of_j", "center_dist_deg", "bbox_touch",
        ])
        for ov in overlaps:
            writer.writerow([
                ov.i, ov.j, ov.tile_i, ov.tile_j,
                f"{ov.area_sqdeg:.8f}", f"{ov.frac_of_i:.6f}", f"{ov.frac_of_j:.6f}",
                f"{ov.center_dist_deg:.6f}", int(ov.bbox_touch),
            ])


def write_tile_log(tiles: Sequence[TileInfo], overlaps: Sequence[OverlapInfo], out_log: Path, ref_ra_deg: float, ref_dec_deg: float) -> None:
    overlap_count = {tile.index: 0 for tile in tiles}
    best_overlap = {tile.index: 0.0 for tile in tiles}
    for ov in overlaps:
        overlap_count[ov.i] += 1
        overlap_count[ov.j] += 1
        best_overlap[ov.i] = max(best_overlap[ov.i], ov.frac_of_i)
        best_overlap[ov.j] = max(best_overlap[ov.j], ov.frac_of_j)

    isolated = [tile.name for tile in tiles if overlap_count[tile.index] == 0]

    with out_log.open("w", encoding="utf-8") as fh:
        fh.write("ZeMosaic master-tile overlap diagnostic\n")
        fh.write("=" * 72 + "\n")
        fh.write(f"Tiles scanned          : {len(tiles)}\n")
        fh.write(f"Pairs with overlap     : {len(overlaps)}\n")
        fh.write(f"Projection reference   : RA={ref_ra_deg:.6f} deg  Dec={ref_dec_deg:.6f} deg\n")
        fh.write(f"Isolated tiles         : {len(isolated)}\n")
        if isolated:
            fh.write("  " + ", ".join(isolated) + "\n")
        fh.write("\n")
        fh.write("Per-tile characteristics\n")
        fh.write("-" * 72 + "\n")
        for tile in tiles:
            fh.write(f"[{tile.index:03d}] {tile.name}\n")
            fh.write(f"  path               : {tile.path}\n")
            fh.write(f"  shape (H,W)        : {tile.shape_hw[0]} x {tile.shape_hw[1]}\n")
            fh.write(f"  center (RA,Dec)    : {tile.center_radec_deg[0]:.6f}, {tile.center_radec_deg[1]:.6f} deg\n")
            fh.write(f"  footprint WxH      : {tile.width_deg:.5f} x {tile.height_deg:.5f} deg\n")
            fh.write(f"  footprint area     : {tile.area_sqdeg:.8f} sq.deg\n")
            fh.write(f"  pixel scale        : {tile.pixel_scale_arcsec:.4f} arcsec/px\n" if tile.pixel_scale_arcsec is not None else "  pixel scale        : n/a\n")
            fh.write(f"  PA                 : {tile.pa_deg:.2f} deg\n" if tile.pa_deg is not None else "  PA                 : n/a\n")
            fh.write(f"  filter / exposure  : {tile.filter_name or '-'} / {tile.exposure_s if tile.exposure_s is not None else 'n/a'} s\n")
            fh.write(f"  date-obs           : {tile.date_obs or '-'}\n")
            fh.write(f"  stack count        : {tile.stack_count if tile.stack_count is not None else 'n/a'}\n")
            fh.write(f"  neighbors          : {overlap_count[tile.index]}\n")
            fh.write(f"  best overlap frac  : {best_overlap[tile.index]:.4f}\n")
            hdr_bits = []
            for key, value in tile.raw_header_keys.items():
                if value is None or value == "":
                    continue
                hdr_bits.append(f"{key}={value}")
            fh.write("  header keys        : " + "; ".join(hdr_bits) + "\n\n")

        fh.write("Pairwise overlaps\n")
        fh.write("-" * 72 + "\n")
        if not overlaps:
            fh.write("No overlap detected.\n")
        else:
            for ov in overlaps:
                fh.write(
                    f"[{ov.i:03d}] {ov.tile_i}  <->  [{ov.j:03d}] {ov.tile_j} | "
                    f"area={ov.area_sqdeg:.8f} sq.deg | "
                    f"frac_i={ov.frac_of_i:.4f} | frac_j={ov.frac_of_j:.4f} | "
                    f"center_dist={ov.center_dist_deg:.4f} deg\n"
                )


def plot_tiles(
    tiles: Sequence[TileInfo],
    overlaps: Sequence[OverlapInfo],
    out_png: Path,
    annotate: bool = True,
    draw_overlap_links: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10), dpi=160)
    for tile in tiles:
        poly = tile.footprint_xy_deg
        patch = MplPolygon(poly, closed=True, fill=False, linewidth=1.4, alpha=0.85)
        ax.add_patch(patch)
        cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
        ax.plot(cx, cy, marker="+", markersize=5)
        if annotate:
            ax.text(cx, cy, f"{tile.index}\n{tile.name}", fontsize=7, ha="center", va="center")

    if draw_overlap_links:
        for ov in overlaps:
            ti = tiles[ov.i]
            tj = tiles[ov.j]
            cxi = float(np.mean(ti.footprint_xy_deg[:, 0]))
            cyi = float(np.mean(ti.footprint_xy_deg[:, 1]))
            cxj = float(np.mean(tj.footprint_xy_deg[:, 0]))
            cyj = float(np.mean(tj.footprint_xy_deg[:, 1]))
            ax.plot([cxi, cxj], [cyi, cyj], linestyle="--", linewidth=0.6, alpha=0.35)

    ax.set_title("Master-tile footprints and overlaps (local tangent-plane projection)")
    ax.set_xlabel("Delta RA * cos(Dec₀) [deg]")
    ax.set_ylabel("Delta Dec [deg]")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_xaxis()
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
            area_sqdeg = polygon_area(fp_xy)
            width_deg, height_deg = estimate_width_height_deg(fp_xy)
            updated_tiles.append(
                TileInfo(
                    index=t.index,
                    path=t.path,
                    name=t.name,
                    shape_hw=t.shape_hw,
                    center_radec_deg=t.center_radec_deg,
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

    outdir = config.outdir.expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = config.prefix.strip() or "master_tiles"
    out_png = outdir / f"{prefix}_footprints.png"
    out_log = outdir / f"{prefix}_diagnostic.log"
    out_csv = outdir / f"{prefix}_overlaps.csv"

    emit("Writing outputs...")
    plot_tiles(tiles, overlaps, out_png, annotate=config.annotate, draw_overlap_links=config.draw_links)
    write_tile_log(tiles, overlaps, out_log, ref_ra, ref_dec)
    write_overlap_csv(overlaps, out_csv)

    emit(f"Tiles analyzed : {len(tiles)}")
    emit(f"Overlaps kept  : {len(overlaps)}")
    emit(f"PNG            : {out_png}")
    emit(f"LOG            : {out_log}")
    emit(f"CSV            : {out_csv}")

    return DiagnosticResult(
        tile_count=len(tiles),
        overlap_count=len(overlaps),
        out_png=out_png,
        out_log=out_log,
        out_csv=out_csv,
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
            self.setWindowTitle("ZeMosaic Master Tile Overlap Diagnostic")
            self.resize(980, 720)

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
            self.annotate_check = QCheckBox("Annotate tile labels in PNG")
            self.annotate_check.setChecked(True)
            self.links_check = QCheckBox("Draw overlap links")
            self.links_check.setChecked(True)

            og.addWidget(QLabel("Output folder:"), 0, 0)
            og.addWidget(self.outdir_edit, 0, 1)
            og.addWidget(self.outdir_btn, 0, 2)
            og.addWidget(QLabel("Filename prefix:"), 1, 0)
            og.addWidget(self.prefix_edit, 1, 1, 1, 2)
            og.addWidget(QLabel("Min overlap fraction:"), 2, 0)
            og.addWidget(self.min_overlap_spin, 2, 1)
            og.addWidget(QLabel("Ref. RA (deg, optional):"), 3, 0)
            og.addWidget(self.ref_ra_edit, 3, 1)
            og.addWidget(QLabel("Ref. Dec (deg, optional):"), 4, 0)
            og.addWidget(self.ref_dec_edit, 4, 1)
            og.addWidget(self.annotate_check, 5, 0, 1, 2)
            og.addWidget(self.links_check, 5, 2)
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
            self.append_log("Starting diagnostic...")
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
            self.status_label.setText(f"Done. {result.tile_count} tiles, {result.overlap_count} overlaps.")
            self.append_log("Diagnostic completed successfully.")
            self.set_running_state(False)
            self.open_out_btn.setEnabled(True)
            self.worker = None
            QMessageBox.information(
                self,
                "Diagnostic finished",
                f"Tiles analyzed: {result.tile_count}\n"
                f"Overlaps kept: {result.overlap_count}\n\n"
                f"PNG: {result.out_png}\n"
                f"LOG: {result.out_log}\n"
                f"CSV: {result.out_csv}",
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
                target = self.last_result.out_png.parent
            else:
                text = self.outdir_edit.text().strip()
                if text:
                    target = Path(text).expanduser()
            if target is not None:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Master tile overlap diagnostic (GUI + CLI)")
    parser.add_argument("inputs", nargs="*", help="Input FITS files and/or directories")
    parser.add_argument("--recursive", action="store_true", help="Recurse into input directories")
    parser.add_argument("--outdir", default="tile_overlap_diag", help="Output directory")
    parser.add_argument("--prefix", default="master_tiles", help="Output filename prefix")
    parser.add_argument("--min-overlap-frac", type=float, default=0.0, help="Minimum pairwise overlap fraction to keep")
    parser.add_argument("--ref-ra", type=float, default=None, help="Override projection reference RA in deg")
    parser.add_argument("--ref-dec", type=float, default=None, help="Override projection reference Dec in deg")
    parser.add_argument("--no-annotate", action="store_true", help="Disable tile labels in the PNG")
    parser.add_argument("--no-links", action="store_true", help="Disable overlap link lines in the PNG")
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
        annotate=not args.no_annotate,
        draw_links=not args.no_links,
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

    # No inputs => default to GUI. Explicit --gui also forces GUI.
    if args.gui or not args.inputs:
        return run_gui()
    return run_cli_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
