"""Peak-RSS benchmark: in-memory vs streaming canonical execution (Levier 2).

Runs the canonical executor on a synthetic ~1 Mpx mono patch with N frames and
prints the executor's *added* peak RSS (KiB), i.e. the high-water mark after the
input arrays are generated. Used by the gated heavy memory test and for the
durable report numbers.

Usage:
    python mem_bench.py --mode inmem --n 30 --h 1024 --w 1024
    python mem_bench.py --mode stream --n 30 --h 1024 --w 1024 --tile 256
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack  # noqa: E402
from zemosaic.core.canonical_streaming import (  # noqa: E402
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)


def _peak_rss_kib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["inmem", "stream"], required=True)
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--h", type=int, default=1024)
    ap.add_argument("--w", type=int, default=1024)
    args = ap.parse_args()

    # Deterministic synthetic input (constant background + light noise), mono.
    rng = np.random.default_rng(0)
    frames = [
        (50.0 + rng.normal(0.0, 3.0, (args.h, args.w))).astype(np.float32)
        for _ in range(args.n)
    ]
    masks = [np.ones((args.h, args.w), dtype=bool)] * args.n

    baseline_kib = _peak_rss_kib()

    req = CanonicalStackRequest(
        images=frames, geometric_support=masks,
        normalization="none", weighting="none", rejection="none",
        combine="mean", taper="none",
    )
    if args.mode == "inmem":
        run_canonical_stack(req)
    else:
        run_canonical_stack_streaming(
            InMemoryCanonicalProvider(frames, masks), req, tile_size=args.tile
        )

    peak_kib = _peak_rss_kib()
    print(f"added_peak_kib={peak_kib - baseline_kib}")


if __name__ == "__main__":
    main()
