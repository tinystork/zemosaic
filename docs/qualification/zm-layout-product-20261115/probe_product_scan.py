"""ZM-LAYOUT-PRODUCT-20261115 — bounded local probe of the REAL layout chooser.

Measures, on the production entry point ``_choose_layout_mode_aware``:

* wall time (median of ``--repeats`` runs after a warm-up) for a few budgets;
* process RSS before/after each run (psutil) — the resident cost, including the
  native STRtree allocation which ``tracemalloc`` cannot see;
* a SHA-256 of the FULL returned layout dict (decision signature), so the same
  probe run in the baseline tree and in the candidate tree proves that the
  decisions are byte-identical;
* the once-per-run index build time and its native RSS cost, at the real M106
  size (66 footprints) and at the historical M33 scale (21 818 polygons).

Read-only: the M106 corpus is opened header-only (``read_manifest``); nothing is
written outside ``--out``. No Windows, no M33, no pixels, no multiprocessing.

Usage (from the repository root)::

    PYTHONPATH=src python docs/qualification/zm-layout-product-20261115/probe_product_scan.py \
        --out docs/qualification/zm-layout-product-20261115/evidence/<tree>.json

Run it once in the baseline worktree (``a59e72c``) and once in the candidate
worktree, then compare ``signature_sha256`` per case: they must be identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from astropy.wcs import WCS

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg

M106_LIGHTS = Path("/home/tristan/M106/lights")
GiB = 2 ** 30


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

def load_m106():
    if not M106_LIGHTS.is_dir():
        return None
    frames, _rejected = zg.read_manifest(M106_LIGHTS)
    canvas = zg.build_canvas(frames)
    return frames, canvas


def _frame(i, h, w, scale, ra, dec):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
    wcs.array_shape = (h, w)
    fid = f"p{i:05d}.fits"
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid), source_path=f"/probe/{fid}", shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(),
        header_sha256="", instrument="",
    )


def synthetic(n, grid, side, step, tag):
    descs = []
    for i in range(n):
        gx = i % grid
        gy = (i // grid) % grid
        descs.append(_frame(i, side, side, 0.001,
                            10.0 + gx * step, 30.0 + gy * step))
    return descs, zg.build_canvas(descs)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _rss():
    return int(psutil.Process().memory_info().rss)


def signature(layout) -> str:
    return hashlib.sha256(
        json.dumps(layout, sort_keys=True, default=str).encode()
    ).hexdigest()


def measure(canvas, descs, *, budget, floors=None, repeats=3, **kw):
    logs = []
    warm = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=budget, floors=floors, log_line=logs.append, **kw
    )
    times = []
    sigs = set()
    rss_before = _rss()
    rss_after_max = rss_before
    for _ in range(repeats):
        logs.clear()
        t0 = time.perf_counter()
        lay = zz._choose_layout_mode_aware(
            canvas, descs, ram_budget=budget, floors=floors,
            log_line=logs.append, footprints_in=None, **kw,
        )
        times.append(time.perf_counter() - t0)
        # SAMPLED AFTER each complete chooser call, never during the scan: this is
        # the POST-CALL RSS maximum, not a transient peak (rework-1 L2).
        rss_after_max = max(rss_after_max, _rss())
        sigs.add(signature(lay))
    assert sigs == {signature(warm)}, "non-deterministic layout decision"
    return {
        "budget_bytes": budget,
        "chosen": [warm["nx"], warm["ny"]],
        "layout_origin": warm["layout_origin"],
        "median_seconds": statistics.median(times),
        "min_seconds": min(times),
        "max_seconds": max(times),
        "repeats": repeats,
        "signature_sha256": signature(warm),
        "rss_before_bytes": rss_before,
        # Max of RSS sampled immediately AFTER each complete choice. The actual
        # transient peak inside the scan is NOT measured by this probe.
        "rss_after_bytes_max": rss_after_max,
        "log_lines": len(logs),
    }


def index_cost(polygons, label):
    """Build-time and NATIVE resident cost of the footprint index."""
    if not hasattr(zal, "FootprintIndex"):
        return {"label": label, "polygons": len(polygons),
                "backends": {}, "absent": "baseline has no footprint index"}
    out = {"label": label, "polygons": len(polygons), "backends": {}}
    for backend in ("strtree", "sorted"):
        try:
            base = _rss()
            t0 = time.perf_counter()
            idx = zal.FootprintIndex(polygons, backend=backend)
            elapsed = time.perf_counter() - t0
            after = _rss()
            out["backends"][backend] = {
                "build_seconds": elapsed,
                "rss_delta_bytes": int(after - base),
                "python_structures_bytes": idx.memory_bytes(),
                "indexed": idx.indexed,
                "uncertain": idx.uncertain,
                "build_fallback": idx.build_fallback,
            }
            del idx
        except Exception as exc:  # pragma: no cover - reported, never hidden
            out["backends"][backend] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _synthetic_polygons(n, side=60.0, cols=150):
    from shapely.geometry import Polygon
    polys = []
    for i in range(n):
        x = (i % cols) * side * 1.2
        y = (i // cols) * side * 1.2
        polys.append(Polygon([(x, y), (x + side, y), (x + side, y + side),
                              (x, y + side)]))
    return polys


def _deep_size(obj, seen=None):
    """Deep ``sys.getsizeof`` of a plain Python object graph (cycles guarded).

    This is the RETAINED PYTHON OBJECT cost of the value, not RSS: it excludes
    the interpreter, the polygons, interned/constant strings shared elsewhere and
    the allocator overhead.
    """
    import sys as _sys
    seen = set() if seen is None else seen
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    total = _sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            total += _deep_size(k, seen) + _deep_size(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            total += _deep_size(v, seen)
    return total


def retained_cells(canvas, footprints, nx, ny):
    """Retained cost of the per-Cell dicts the chooser keeps for reuse (L4).

    Scope: exactly the ``cells_out`` list that ``_scan_layout_mode_aware`` hands
    to the chosen-grid reuse (one dict per Cell: cell_id/row/col/n/patch_area/
    patch_hw/inmem+stream bounds/cheaper_mode) plus the ``dict(c)`` worst-cell
    copy. ``sys.getsizeof`` deep, i.e. Python objects only - NOT RSS, and NOT the
    transient peak.
    """
    import inspect
    params = inspect.signature(zz._iter_cell_bounds_mode_aware).parameters
    index = None
    if hasattr(zal, "FootprintIndex") and "index" in params:
        index = zal.FootprintIndex(footprints.polygons)
    kw = {"index": index} if index is not None else {}
    cells = list(zz._iter_cell_bounds_mode_aware(
        canvas, footprints, nx, ny, zz.zsw.HALO_PX, zz.STREAM_TILE_SIZE, **kw
    ))
    worst_copy = dict(cells[0]) if cells else {}
    total = _deep_size(cells) + _deep_size(worst_copy)
    return {
        "grid": [int(nx), int(ny)],
        "cells": len(cells),
        "retained_python_bytes": int(total),
        "bytes_per_cell": (float(total) / len(cells)) if cells else 0.0,
        "scope": (
            "per-Cell dicts retained for chosen-scan reuse + dict(c) worst copy; "
            "sys.getsizeof deep, Python objects only (no RSS, no transient peak)"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats-slow", type=int, default=3)
    ap.add_argument("--repeats-fast", type=int, default=5)
    ap.add_argument(
        "--only-retained", action="store_true",
        help="bounded micro-probe: retained per-Cell cost of the largest "
             "qualified grid (no timing benchmark)",
    )
    args = ap.parse_args(argv)

    if args.only_retained:
        out = {"retained_cells": []}
        m106 = load_m106()
        if m106 is not None:
            frames, canvas = m106
            fp = zal.project_footprints(frames, canvas)
            out["retained_cells"].append(retained_cells(canvas, fp, 67, 39))
            out["environment"] = {"m106_frames": len(frames),
                                  "m106_canvas": [canvas.width, canvas.height]}
        descs, canvas = synthetic(2000, 45, 160, 0.02, "dense")
        fp = zal.project_footprints(descs, canvas)
        out["retained_cells"].append(retained_cells(canvas, fp, 67, 39))
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1, sort_keys=True))
        print(json.dumps(out, indent=1, sort_keys=True))
        return 0

    result = {"cases": [], "index": [], "environment": {}}
    m106 = load_m106()
    if m106 is not None:
        frames, canvas = m106
        result["environment"].update({
            "m106_frames": len(frames),
            "m106_canvas": [canvas.width, canvas.height],
        })
        for budget in (None, GiB, 2 * GiB):
            r = measure(canvas, frames, budget=budget,
                        repeats=args.repeats_fast)
            r["case"] = f"m106 budget={budget}"
            result["cases"].append(r)
            print(f"{r['case']:34s} {r['median_seconds']:.4f}s "
                  f"{r['chosen']} {r['signature_sha256'][:12]}", flush=True)
        # index cost at the real M106 size (already projected, once)
        fp = zal.project_footprints(frames, canvas)
        result["index"].append(index_cost(fp.polygons, "real M106"))
    else:
        result["environment"]["m106_frames"] = None

    # A layout scan that actually runs several candidate grids.
    descs, canvas = synthetic(2000, 45, 160, 0.02, "dense")
    rel = zal.ScientificFloors(min_patch_area_px=1, min_contributors=1,
                               max_halo_overhead=1e9)
    result["environment"]["synthetic_canvas"] = [canvas.width, canvas.height]
    for budget in (GiB, 512 * 2 ** 20):
        r = measure(canvas, descs, budget=budget, floors=rel,
                    repeats=args.repeats_slow)
        r["case"] = f"dense2000 budget={budget}"
        result["cases"].append(r)
        print(f"{r['case']:34s} {r['median_seconds']:.4f}s "
              f"{r['chosen']} {r['signature_sha256'][:12]}", flush=True)

    # Native RSS of the index at the historical M33 scale (21 818 footprints).
    result["index"].append(
        index_cost(_synthetic_polygons(21818), "synthetic 21818")
    )
    result["index"].append(
        index_cost(_synthetic_polygons(1024), "synthetic 1024")
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
