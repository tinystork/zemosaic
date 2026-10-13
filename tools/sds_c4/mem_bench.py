"""ZM-SDS-R35C-C4 — peak-RSS benchmark: in-memory vs streaming SDS final stack.

Runs the SDS megatiles -> final canonical stack on a synthetic representative
corpus and prints the executor's *added* peak RSS (KiB) — the high-water mark
after the input mega-tiles are generated. Used for the C4 durable report and by
the (gated) heavy memory test.

The in-memory mode is the historical path: ``run_canonical_stack`` materialises
``O(N x patch_area)`` float64/float32/bool intermediates. The streaming mode is
the new SDS provider + ``run_canonical_stack_streaming`` path (``O(N x tile_area)``
workspace, served from a memmap disk cache).

Usage:
    python mem_bench.py --mode inmem  --n 7 --h 1280 --w 1280
    python mem_bench.py --mode stream --n 7 --h 1280 --w 1280 --tile 256
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack  # noqa: E402
from zemosaic.core.sds_streaming import (  # noqa: E402
    SdsMegatileFileProvider,
    run_sds_final_stack_streaming,
    write_sds_megatile_cache,
)


def _peak_rss_kib() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


def _corpus(n, h, w, c=3, seed=0):
    rng = np.random.default_rng(seed)
    mosaics, covs = [], []
    for i in range(n):
        m = (600.0 + rng.normal(0.0, 4.0, (h, w, c))).astype(np.float32)
        cov = np.ones((h, w), np.float32)
        cov[:, : i * (w // (n + 2))] = 0.0
        m[cov <= 0] = np.nan
        mosaics.append(m)
        covs.append(cov)
    return mosaics, covs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["inmem", "stream"], required=True)
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--h", type=int, default=1280)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--c", type=int, default=3)
    args = ap.parse_args()

    mosaics, covs = _corpus(args.n, args.h, args.w, args.c)
    supports = [(np.asarray(cc, np.float32) > 0) for cc in covs]
    ref_idx = 0 if np.any(supports[0]) else None

    baseline_kib = _peak_rss_kib()

    sp = {"stack_reject_algo": "winsorized_sigma_clip", "stack_final_combine": "mean"}
    if args.mode == "inmem":
        req = CanonicalStackRequest(
            images=mosaics,
            geometric_support=supports,
            normalization="none",
            weighting="none",
            rejection="winsorized_sigma_clip",
            combine="mean",
            reference_index=ref_idx,
            taper="none",
            backend="cpu",
        )
        run_canonical_stack(req)
    else:
        with tempfile.TemporaryDirectory() as td:
            write_sds_megatile_cache(td, [f"m{i}" for i in range(args.n)], mosaics, supports)
            with SdsMegatileFileProvider(td) as provider:
                run_sds_final_stack_streaming(provider, sp, tile_size=args.tile)

    peak_kib = _peak_rss_kib()
    print(f"added_peak_kib={peak_kib - baseline_kib}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
