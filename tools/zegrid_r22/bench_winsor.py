"""ZM-ZEGRID-R22 — local real proof benchmark (CPU baseline vs optimized CPU vs GPU).

Bounded, representative M106 probe (37 frames x 128x128x3, 10% NaN — the exact
shape the R20 diagnostic used). Measures the quantile primitive, one winsor step,
and the GPU quantile on the MX150. NO 3-hour full run.

Usage:
    python tools/zegrid_r22/bench_winsor.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.canonical_stacking import _nan_axis_quantile, _nan_axis_quantile_gpu


def _probe():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(37, 128 * 128 * 3)).astype(np.float64)
    a[rng.random(a.shape) < 0.10] = np.nan
    return a


def bench_quantile_primitive():
    a = _probe()
    for q in (0.05, 0.95):
        t0 = time.perf_counter()
        np.nanquantile(a, q, axis=0, method="linear")
        t_ref = time.perf_counter() - t0
        t0 = time.perf_counter()
        _nan_axis_quantile(a, q, np)
        t_new = time.perf_counter() - t0
        print(f"quantile q={q}: np.nanquantile={t_ref:.4f}s sort_helper={t_new:.4f}s "
              f"speedup={t_ref / t_new:.1f}x")


def bench_one_winsor_step():
    """One full winsorized step: 2 quantiles + clip + mean + std (optimized path)."""
    a = _probe()
    # The optimized winsor step = 2x _nan_axis_quantile + clip + nanmean + nanstd.
    t0 = time.perf_counter()
    q_low = _nan_axis_quantile(a, 0.05, np)
    q_high = _nan_axis_quantile(a, 0.95, np)
    win = np.clip(a, q_low[None, :], q_high[None, :])
    _c = np.nanmean(win, axis=0)
    _s = np.nanstd(win, axis=0)
    t_opt = time.perf_counter() - t0
    # Reference (old path): np.nanquantile x2 + same.
    t0 = time.perf_counter()
    q_low_r = np.nanquantile(a, 0.05, axis=0, method="linear")
    q_high_r = np.nanquantile(a, 0.95, axis=0, method="linear")
    win_r = np.clip(a, q_low_r[None, :], q_high_r[None, :])
    _c2 = np.nanmean(win_r, axis=0)
    _s2 = np.nanstd(win_r, axis=0)
    t_ref = time.perf_counter() - t0
    print(f"one winsor step: np.nanquantile path={t_ref:.4f}s optimized path={t_opt:.4f}s "
          f"speedup={t_ref / t_opt:.1f}x")


def bench_gpu_quantile():
    try:
        import cupy as cp
    except ImportError:
        print("GPU: CuPy not available in this interpreter — skip")
        return
    if cp.cuda.runtime.getDeviceCount() < 1:
        print("GPU: no CUDA device — skip")
        return
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    a = _probe()
    ag = cp.asarray(a)
    # warm up
    _nan_axis_quantile_gpu(ag, 0.05, cp)
    t0 = time.perf_counter()
    _nan_axis_quantile_gpu(ag, 0.05, cp)
    _nan_axis_quantile_gpu(ag, 0.95, cp)
    t_gpu = time.perf_counter() - t0
    # verify bit-exact vs CPU
    ref = np.nanquantile(a, 0.05, axis=0, method="linear")
    got = cp.asnumpy(_nan_axis_quantile_gpu(ag, 0.05, cp))
    bit_exact = bool(np.array_equal(ref, got, equal_nan=True))
    print(f"GPU ({name}, cupy {cp.__version__}): 2 quantiles (N=37, 49152 cols) = "
          f"{t_gpu:.4f}s  bit_exact_vs_numpy={bit_exact}")


if __name__ == "__main__":
    print(f"probe shape: (37, 128*128*3=49152) float64, 10% NaN")
    bench_quantile_primitive()
    bench_one_winsor_step()
    bench_gpu_quantile()
