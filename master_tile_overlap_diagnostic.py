#!/usr/bin/env python3
"""Visualize ZeMosaic master-tile overlaps from FITS headers/WCS.

This standalone helper scans FITS files, extracts celestial WCS from headers,
computes each tile footprint, projects them to a local tangent plane, and
produces:

- a PNG overview of the master-tile footprints
- a text log with per-tile characteristics and pairwise overlap diagnostics
- a CSV file with overlap metrics

Designed as a lightweight diagnostic companion for ZeMosaic seam analysis.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    from astropy.coordinates import SkyCoord
    import astropy.units as u
except Exception as exc:  # pragma: no cover
    print(f"ERROR: astropy is required: {exc}", file=sys.stderr)
    sys.exit(2)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
except Exception as exc:  # pragma: no cover
    print(f"ERROR: matplotlib is required: {exc}", file=sys.stderr)
    sys.exit(2)

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
    footprint_radec_deg: np.ndarray  # (N,2) RA,Dec in deg
    footprint_xy_deg: np.ndarray  # (N,2) projected x,y in deg
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


# ---------- small geometry helpers ----------

def polygon_area(poly: np.ndarray) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    if polygon_area(poly) <= 0:
        return poly[::-1].copy()
    return poly.copy()


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
    """Return intersection polygon of two convex polygons using Sutherland-Hodgman."""
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


# ---------- WCS / projection helpers ----------

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


# ---------- IO / scan ----------

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


def build_tile_infos(files: Sequence[Path], ref_ra_deg: Optional[float] = None, ref_dec_deg: Optional[float] = None) -> List[TileInfo]:
    pre_tiles = []
    centers = []
    for path in files:
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
            ra0 = float(header.get("CRVAL1")) if header.get("CRVAL1") is not None else None
            dec0 = float(header.get("CRVAL2")) if header.get("CRVAL2") is not None else None
            if ra0 is None or dec0 is None:
                continue
        centers.append((ra0, dec0))
        pre_tiles.append((path, header, wcs_obj, width, height, ra0, dec0))

    if not pre_tiles:
        return []

    if ref_ra_deg is None or ref_dec_deg is None:
        ra_vals = np.array([c[0] for c in centers], dtype=float)
        dec_vals = np.array([c[1] for c in centers], dtype=float)
        ref_ra_deg = float(np.median(ra_vals))
        ref_dec_deg = float(np.median(dec_vals))

    tiles: List[TileInfo] = []
    for idx, (path, header, wcs_obj, width, height, ra0, dec0) in enumerate(pre_tiles):
        fp_radec = compute_footprint_radec_deg(wcs_obj, width, height)
        fp_xy = project_radec_to_plane_deg(fp_radec, ref_ra_deg, ref_dec_deg)
        area_sqdeg = polygon_area(fp_xy)
        wh_deg = estimate_width_height_deg(fp_xy)
        scale_arcsec = None
        try:
            scales = proj_plane_pixel_scales(wcs_obj.celestial) * 3600.0
            if scales is not None and len(scales) >= 2:
                scale_arcsec = float(np.nanmean(np.abs(scales[:2])))
        except Exception:
            scale_arcsec = None
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
                width_deg=float(wh_deg[0]),
                height_deg=float(wh_deg[1]),
                filter_name=str(header.get("FILTER", header.get("FILTNAME", "")) or ""),
                exposure_s=float(header.get("EXPTIME")) if isinstance(header.get("EXPTIME"), (int, float, np.integer, np.floating)) else None,
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


# ---------- overlap analysis ----------

def bbox_intersects(a: np.ndarray, b: np.ndarray) -> bool:
    ax0, ay0 = np.nanmin(a[:, 0]), np.nanmin(a[:, 1])
    ax1, ay1 = np.nanmax(a[:, 0]), np.nanmax(a[:, 1])
    bx0, by0 = np.nanmin(b[:, 0]), np.nanmin(b[:, 1])
    bx1, by1 = np.nanmax(b[:, 0]), np.nanmax(b[:, 1])
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def analyze_overlaps(tiles: Sequence[TileInfo], min_fraction: float = 0.0) -> List[OverlapInfo]:
    overlaps: List[OverlapInfo] = []
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
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


# ---------- outputs ----------

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
            fh.write("  header keys        : ")
            hdr_bits = []
            for key, value in tile.raw_header_keys.items():
                if value is None or value == "":
                    continue
                hdr_bits.append(f"{key}={value}")
            fh.write("; ".join(hdr_bits) + "\n")
            fh.write("\n")

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
            label = f"{tile.index}\n{tile.name}"
            ax.text(cx, cy, label, fontsize=7, ha="center", va="center")

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
    ax.invert_xaxis()  # sky-like orientation: east to the left
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ---------- CLI ----------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize master-tile overlaps from FITS headers/WCS and write a diagnostic log.",
    )
    parser.add_argument("inputs", nargs="+", help="Input FITS files and/or directories")
    parser.add_argument("--recursive", action="store_true", help="Recurse into input directories")
    parser.add_argument("--outdir", default="tile_overlap_diag", help="Output directory")
    parser.add_argument("--prefix", default="master_tiles", help="Output filename prefix")
    parser.add_argument("--min-overlap-frac", type=float, default=0.0, help="Minimum pairwise overlap fraction to keep in CSV/log")
    parser.add_argument("--ref-ra", type=float, default=None, help="Override projection reference RA in deg")
    parser.add_argument("--ref-dec", type=float, default=None, help="Override projection reference Dec in deg")
    parser.add_argument("--no-annotate", action="store_true", help="Disable tile labels in the PNG")
    parser.add_argument("--no-links", action="store_true", help="Disable overlap link lines between tile centers")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_paths = [Path(p).expanduser() for p in args.inputs]
    files = sorted(set(iter_fits_files(input_paths, recursive=args.recursive)))
    if not files:
        print("No FITS files found.", file=sys.stderr)
        return 1

    tiles = build_tile_infos(files, ref_ra_deg=args.ref_ra, ref_dec_deg=args.ref_dec)
    if not tiles:
        print("No valid celestial WCS could be extracted from the provided FITS headers.", file=sys.stderr)
        return 1

    ref_ra = float(np.median([t.center_radec_deg[0] for t in tiles])) if args.ref_ra is None else float(args.ref_ra)
    ref_dec = float(np.median([t.center_radec_deg[1] for t in tiles])) if args.ref_dec is None else float(args.ref_dec)

    # rebuild local projected polygons if reference overridden after first pass
    if args.ref_ra is not None or args.ref_dec is not None:
        updated_tiles: List[TileInfo] = []
        for t in tiles:
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

    overlaps = analyze_overlaps(tiles, min_fraction=max(0.0, float(args.min_overlap_frac)))

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    out_png = outdir / f"{prefix}_footprints.png"
    out_log = outdir / f"{prefix}_diagnostic.log"
    out_csv = outdir / f"{prefix}_overlaps.csv"

    plot_tiles(
        tiles,
        overlaps,
        out_png,
        annotate=not args.no_annotate,
        draw_overlap_links=not args.no_links,
    )
    write_tile_log(tiles, overlaps, out_log, ref_ra, ref_dec)
    write_overlap_csv(overlaps, out_csv)

    print(f"Tiles analyzed : {len(tiles)}")
    print(f"Overlaps kept  : {len(overlaps)}")
    print(f"PNG            : {out_png}")
    print(f"LOG            : {out_log}")
    print(f"CSV            : {out_csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
