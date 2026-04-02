#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

import numpy as np


KEYS_OF_INTEREST = [
    "intertile_gain_offset_v2",
    "intertile_offset_only_v1",
    "intertile_prune_k",
    "intertile_prune_weight_mode",
    "intertile_affine_blend",
    "tile_weight_v4_enabled",
    "tile_weight_v4_curve",
    "tile_weight_v4_strength",
    "tile_weight_v4_min",
    "tile_weight_v4_max",
    "enable_tile_weighting",
    "radial_feather_fraction",
]

TWOPASS_RE = re.compile(r"\[TwoPassWorst\].*abs_delta_med=([0-9eE+\-.]+)")
M2_RE = re.compile(r"M2 gain\+offset solve.*")


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_weights(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out = {}
    for col in ("tile_weight_raw", "tile_weight_effective"):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col]))
            except Exception:
                pass
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[col] = {
                "min": float(np.min(arr)),
                "median": float(np.median(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
            }
    out["rows"] = len(rows)
    return out


def summarize_log(path: Path):
    if not path.exists():
        return None
    vals = []
    m2_lines = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = TWOPASS_RE.search(line)
            if m:
                try:
                    vals.append(float(m.group(1)))
                except Exception:
                    pass
            if "M2 gain+offset solve" in line:
                m2_lines.append(line.strip())
    summary = {}
    if vals:
        arr = np.asarray(vals, dtype=np.float64)
        summary["twopass_abs_delta_med"] = {
            "count": int(arr.size),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
        }
    if m2_lines:
        summary["m2_markers"] = m2_lines[-3:]
    return summary


def summarize_mosaic_dat(path: Path):
    if not path.exists():
        return None
    size = path.stat().st_size
    if size % 4 != 0:
        return {"error": "unexpected size (not float32)"}
    arr = np.memmap(path, dtype=np.float32, mode="r")
    finite = np.isfinite(arr)
    if not np.any(finite):
        return {"elements": int(arr.size), "finite": 0}
    finite_arr = arr[finite]
    return {
        "elements": int(arr.size),
        "finite": int(finite_arr.size),
        "min": float(np.min(finite_arr)),
        "median": float(np.median(finite_arr)),
        "max": float(np.max(finite_arr)),
    }


def compare_mosaic_dat(a: Path, b: Path):
    if not a.exists() or not b.exists():
        return None
    aa = np.memmap(a, dtype=np.float32, mode="r")
    bb = np.memmap(b, dtype=np.float32, mode="r")
    if aa.shape != bb.shape:
        return {"shape_mismatch": [aa.shape, bb.shape]}
    d = np.abs(aa - bb)
    nz = int(np.count_nonzero(d))
    return {
        "elements": int(aa.size),
        "nonzero": nz,
        "fraction_nonzero": float(nz) / float(aa.size) if aa.size else 0.0,
        "max_diff": float(np.max(d)) if d.size else 0.0,
        "mean_diff": float(np.mean(d)) if d.size else 0.0,
        "p99_diff": float(np.percentile(d, 99)) if d.size else 0.0,
    }


def find_file(run_dir: Path, name: str) -> Path:
    return run_dir / name


def main():
    ap = argparse.ArgumentParser(description="Compare two ZeMosaic runs for M2/seams diagnostics")
    ap.add_argument("run_a", help="Path to run A directory")
    ap.add_argument("run_b", help="Path to run B directory")
    args = ap.parse_args()

    A = Path(args.run_a)
    B = Path(args.run_b)

    print(f"# M2 seam report\n")
    print(f"Run A: {A}")
    print(f"Run B: {B}\n")

    cfg_a = load_json(find_file(A, "run_config_snapshot.json")) or {}
    cfg_b = load_json(find_file(B, "run_config_snapshot.json")) or {}

    print("## Config diffs (keys of interest)")
    any_diff = False
    for k in KEYS_OF_INTEREST:
        va = cfg_a.get(k)
        vb = cfg_b.get(k)
        if va != vb:
            any_diff = True
            print(f"- {k}: {va} -> {vb}")
    if not any_diff:
        print("- none")
    print()

    print("## tile_weights_final.csv summary")
    wa = summarize_weights(find_file(A, "tile_weights_final.csv"))
    wb = summarize_weights(find_file(B, "tile_weights_final.csv"))
    print("A:", json.dumps(wa, ensure_ascii=False))
    print("B:", json.dumps(wb, ensure_ascii=False))
    print()

    print("## log summary")
    la = summarize_log(find_file(A, "zemosaic_worker.log"))
    lb = summarize_log(find_file(B, "zemosaic_worker.log"))
    print("A:", json.dumps(la, ensure_ascii=False))
    print("B:", json.dumps(lb, ensure_ascii=False))
    print()

    print("## mosaic dat comparison")
    da = find_file(A, "mosaic_5873x4667x3.dat")
    db = find_file(B, "mosaic_5873x4667x3.dat")
    if da.exists() and db.exists():
        cmpd = compare_mosaic_dat(da, db)
        print(json.dumps(cmpd, ensure_ascii=False))
    else:
        print("- mosaic dat files not found with default name")


if __name__ == "__main__":
    main()
