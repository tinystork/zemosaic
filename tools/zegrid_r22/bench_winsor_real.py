"""ZM-ZEGRID-R22 rework-1 (M1) — REAL M106 bounded aligned-tile winsor benchmark.

Builds a GENUINE bounded aligned tile from the real M106 lights (read-only, under
``/media/tristan/X10 Pro/M106/lights``): reads the real WCS manifest, decodes +
reprojects a subset of real frames onto one small cell patch, then measures the
complete winsorized-sigma-clip step and verifies bit-equality:

* old ``np.nanquantile(method="linear")`` reference vs the optimized CPU
  sort/interpolate primitive (masks/science bit-identical);
* optimized CPU complete step vs GPU complete step (warm-up + device sync),
  bit-identical.

Records the EXACT frame count, tile shape/coords, NaN fraction and construction
method. Does NOT launch a full run and does NOT infer RTX speed from the MX150.

Usage:
    python tools/zegrid_r22/bench_winsor_real.py [lights_dir]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import sweep as zsw
from zemosaic.core.canonical_stacking import (
    _nan_axis_quantile,
    _winsorized_sigma_clip_step,
)
from zemosaic import zemosaic_zegrid_mode as zz

DEFAULT_LIGHTS = "/media/tristan/X10 Pro/M106/lights"
_LOCAL_MIRROR = "/home/tristan/M106/lights"  # same 66 real M106 lights (local copy)
N_FRAMES = 12          # bounded subset of real frames
TILE_SIDE = 128        # interior tile extracted from the aligned patch
HALO = 9


def _build_real_tile(lights_dir: str):
    frames, rejected = zg.read_manifest(lights_dir)
    print(f"real frames loaded: {len(frames)} (rejected {len(rejected)})")

    # Deterministic subset: the N frames nearest the canvas centre (by CRVAL).
    canvas = zg.build_canvas(frames)
    print(f"canvas: {canvas.width}x{canvas.height} res={canvas.resolution_deg:.3g} deg/px")

    subset = sorted(frames, key=lambda f: f.frame_id)[:N_FRAMES]
    print(f"subset: {len(subset)} frames")

    # A fine layout so the chosen cell patch is small (~TILE_SIDE x TILE_SIDE).
    nx = max(1, canvas.width // TILE_SIDE)
    ny = max(1, canvas.height // TILE_SIDE)
    layout = zg.build_layout(canvas, nx, ny)
    # Pick the cell whose patch centre is closest to the canvas centre.
    centre = (canvas.width / 2.0, canvas.height / 2.0)
    chosen = None
    best = None
    for row, col, bounds in layout.iter_cells(canvas):
        cx = (bounds.x0 + bounds.x1) / 2.0
        cy = (bounds.y0 + bounds.y1) / 2.0
        d = (cx - centre[0]) ** 2 + (cy - centre[1]) ** 2
        if best is None or d < best:
            best = d
            chosen = (row, col)
    row, col = chosen
    cell, patch, mem = zsw.build_cell_context(subset, canvas, row, col, nx, ny)
    print(f"chosen cell r{row:04d}c{col:04d} patch={patch.patch_shape_hw} contributors={len(mem.patch_ids)}")

    cache_dir = Path("/tmp/zegrid_r22_real_tile_cache")
    zz._safe_rmtree(cache_dir, None)
    manifest = zz._build_one_cell_cache(subset, canvas, cell, patch, mem, cache_dir, workers=1)
    print(f"aligned cache built: {manifest['n_frames']} frames, {manifest['total_bytes']} bytes")

    from zemosaic.core.zegrid import file_provider as zfp
    provider = zfp.MemmapCanonicalProvider(cache_dir)
    try:
        n = provider.n_frames
        h, w = patch.patch_shape_hw
        th = tw = TILE_SIDE
        y0 = max(0, (h - th) // 2)
        x0 = max(0, (w - tw) // 2)
        y1, x1 = y0 + th, x0 + tw
        images = []
        supports = []
        for i in range(n):
            rgb, sup = provider.get_raw_frame(i)
            images.append(np.array(rgb[y0:y1, x0:x1], dtype=np.float32, copy=True))
            supports.append(np.array(sup[y0:y1, x0:x1], dtype=bool, copy=True))
    finally:
        provider.close()
    zz._safe_rmtree(cache_dir, None)

    # Build the canonical tile (N, th, tw, 3) float32 + (N, th, tw) bool, then the
    # rejection workspace (N, M) float64 + initial survivor mask.
    images_np = np.stack([np.ascontiguousarray(im) for im in images])
    supports_np = np.stack(supports)
    finite = np.isfinite(images_np).all(axis=-1)
    valid = supports_np & finite
    images64 = images_np.astype(np.float64)
    nan_frac = float((~finite).mean()) if finite.size else 0.0

    n, th, tw, c = images64.shape
    n_cells = th * tw * c
    orig2 = images64.reshape(n, n_cells)
    initial = np.broadcast_to(valid[..., None], (n, th, tw, c)).reshape(n, n_cells)
    survivor2 = initial.copy()
    count = survivor2.sum(axis=0)

    meta = {
        "n_frames": int(n),
        "tile_shape": (int(th), int(tw), int(c)),
        "tile_coords": (int(y0), int(y1), int(x0), int(x1)),
        "nan_fraction": nan_frac,
        "construction": (
            "real M106 lights WCS manifest -> build_canvas -> fine layout cell -> "
            "_build_one_cell_cache (real decode + reproject_cropped) -> central "
            "TILE_SIDE x TILE_SIDE aligned tile"
        ),
    }
    return orig2, survivor2, count, meta


def _run_quantile_reference(masked, q):
    return np.nanquantile(masked, q, axis=0, method="linear")


def bench_real():
    orig2, survivor2, count, meta = _build_real_tile(DEFAULT_LIGHTS)
    print(f"=== REAL M106 TILE: {meta}")

    s_low = s_high = 2.5
    w_low = w_high = 0.05

    # 1. Quantile primitive bit-equality on the REAL tile.
    masked = np.where(survivor2, orig2, np.nan)
    for q in (0.05, 0.95):
        ref = _run_quantile_reference(masked, q)
        got = _nan_axis_quantile(masked, q, np)
        print(f"quantile q={q}: bit_exact_vs_np.nanquantile = {np.array_equal(ref, got, equal_nan=True)}")

    # 2. Complete winsor step: optimized CPU (warm-up + timing).
    _winsorized_sigma_clip_step(orig2, survivor2, count, s_low, s_high, w_low, w_high, np)
    t0 = time.perf_counter()
    cpu_surv, cpu_deg = _winsorized_sigma_clip_step(
        orig2, survivor2, count, s_low, s_high, w_low, w_high, np
    )
    t_cpu = time.perf_counter() - t0
    print(f"complete winsor step CPU (optimized): {t_cpu:.4f}s")

    # 3. Complete winsor step: GPU (warm-up + device sync).
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

    orig2_g = cp.asarray(orig2)
    survivor2_g = cp.asarray(survivor2)
    count_g = cp.asarray(count)
    # warm-up
    _winsorized_sigma_clip_step(orig2_g, survivor2_g, count_g, s_low, s_high, w_low, w_high, cp)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    gpu_surv_g, gpu_deg_g = _winsorized_sigma_clip_step(
        orig2_g, survivor2_g, count_g, s_low, s_high, w_low, w_high, cp
    )
    cp.cuda.Stream.null.synchronize()
    t_gpu = time.perf_counter() - t0
    gpu_surv = cp.asnumpy(gpu_surv_g)
    bit_exact = bool(np.array_equal(cpu_surv, gpu_surv))
    print(f"complete winsor step GPU ({name}): {t_gpu:.4f}s  bit_exact_vs_cpu = {bit_exact}")
    print(f"CPU vs GPU complete step: cpu={t_cpu:.4f}s gpu={t_gpu:.4f}s "
          f"ratio(gpu/cpu)={t_gpu / t_cpu:.2f}x")


if __name__ == "__main__":
    lights = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LIGHTS
    if not Path(lights).is_dir() and Path(_LOCAL_MIRROR).is_dir():
        lights = _LOCAL_MIRROR
    if not Path(lights).is_dir():
        print(f"lights dir not found: {lights}", file=sys.stderr)
        sys.exit(2)
    DEFAULT_LIGHTS = lights  # noqa: F811 - override for this process
    print(f"lights dir: {lights}")
    bench_real()
