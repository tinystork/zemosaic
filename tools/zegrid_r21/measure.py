"""ZM-ZEGRID-R21 measurement — reprojection phase BEFORE vs AFTER (end-to-end).

Runs the production ``run_zegrid_mode`` on a pinned M16 subset twice: once with
the R21 fast path (AFTER) and once with ``ZEGRID_REPROJECT_FORCE_FALLBACK=1``
(BEFORE, the pre-R21 ``reproject_interp`` path).  Reports the per-phase timings
(gauge / cache_build / per_cell_stack), the end-to-end wall time, the manifest
``reprojection`` diagnostic (path + fast/fallback seconds), and the science
SHA-256 (must match -> bit-equal output).

Usage (run from repo root)::

    python tools/zegrid_r21/measure.py [corpus_dir] [output_dir] [--frames N] [--layout LX x LY]

Defaults: corpus = /home/tristan/M16/quick, output = /home/tristan/zegrid_r21_measure,
frames = 8, layout = 2x2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

M16 = Path("/home/tristan/M16/quick")
OUT = Path("/home/tristan/zegrid_r21_measure")
LAYOUT = "2x2"


def _make_subset(corpus: Path, n: int, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    fits_files = sorted(p for p in corpus.iterdir() if p.suffix.lower() == ".fit")
    for p in fits_files[:n]:
        shutil.copy2(p, dst / p.name)
    return dst


def _run_one(corpus_dir: Path, out_dir: Path, force_fallback: bool, layout: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.iterdir():
        if f.is_file():
            f.unlink()
    env = dict(os.environ)
    if force_fallback:
        env["ZEGRID_REPROJECT_FORCE_FALLBACK"] = "1"
    script = (
        "import json, sys, time\n"
        "sys.path.insert(0, r'%s')\n"
        "from types import SimpleNamespace\n"
        "from zemosaic import zemosaic_zegrid_mode as zegrid\n"
        "t0 = time.perf_counter()\n"
        "zegrid.run_zegrid_mode(r'%s', r'%s', zconfig=SimpleNamespace(zegrid_layout='%s'))\n"
        "print('WALL', time.perf_counter() - t0)\n"
    ) % (SRC, corpus_dir, out_dir, layout)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print("RUN FAILED:", proc.stderr[-3000:])
        return {}
    wall = None
    for line in proc.stdout.splitlines():
        if line.startswith("WALL"):
            wall = float(line.split()[1])
    manifest = json.loads((out_dir / "zegrid_manifest.json").read_text(encoding="utf-8"))
    timings = manifest.get("timings", {})
    reproj = manifest.get("reprojection", {})
    sci = out_dir / "mosaic_grid.fits"
    sha = None
    if sci.exists():
        import numpy as np
        from astropy.io import fits
        with fits.open(sci) as hdul:
            data = np.ascontiguousarray(np.asarray(hdul[0].data, dtype=np.float32))
        sha = hashlib.sha256(data.tobytes()).hexdigest()[:16]
    return {
        "wall_s": round(wall, 3) if wall else None,
        "timings": timings,
        "reprojection": reproj,
        "science_sha": sha,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir", nargs="?", default=str(M16))
    ap.add_argument("output_dir", nargs="?", default=str(OUT))
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--layout", default=LAYOUT)
    args = ap.parse_args()
    layout = args.layout

    corpus = Path(args.corpus_dir)
    out_root = Path(args.output_dir)

    print(f"corpus={corpus} layout={layout}\n")

    print("== BEFORE (reproject_interp, fast path forced off) ==")
    before = _run_one(corpus, out_root / "before", force_fallback=True, layout=layout)
    print(json.dumps(before, indent=2))

    print("\n== AFTER (R21 fast path) ==")
    after = _run_one(corpus, out_root / "after", force_fallback=False, layout=layout)
    print(json.dumps(after, indent=2))

    if before and after and before["timings"] and after["timings"]:
        print("\n== phase comparison ==")
        for name in ("gauge", "cache_build", "per_cell_stack", "setup", "layout", "assembly"):
            b = before["timings"].get(name)
            a = after["timings"].get(name)
            if b is None or a is None:
                continue
            sp = b / a if a > 0 else float("inf")
            print(f"  {name:16s} before={b:9.3f}s after={a:9.3f}s speedup={sp:.2f}x")
        bt = before["timings"].get("total")
        at = after["timings"].get("total")
        if bt and at:
            print(f"  {'total':16s} before={bt:9.3f}s after={at:9.3f}s speedup={bt/at:.2f}x")
        if before["wall_s"] and after["wall_s"]:
            print(f"  {'wall':16s} before={before['wall_s']:9.3f}s after={after['wall_s']:9.3f}s speedup={before['wall_s']/after['wall_s']:.2f}x")
        if before["science_sha"] and after["science_sha"]:
            print(f"\n  science SHA before={before['science_sha']} after={after['science_sha']} equal={before['science_sha'] == after['science_sha']}")


if __name__ == "__main__":
    main()
