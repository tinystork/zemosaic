#!/usr/bin/env python3
import argparse
import sys
from types import SimpleNamespace

import numpy as np

try:
    import zemosaic_align_stack as zas
except Exception as exc:
    print(f"Failed to import zemosaic_align_stack: {exc}")
    sys.exit(1)


def _gpu_available() -> bool:
    try:
        import zemosaic_align_stack_gpu as gpu_mod

        return bool(getattr(gpu_mod, "_gpu_is_usable", lambda: False)())
    except Exception:
        return False


def _make_frames(seed: int, frames: int, height: int, width: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    stack = []
    for idx in range(frames):
        frame = rng.normal(loc=1000.0, scale=5.0, size=(height, width, 3)).astype(np.float32)
        frame += idx * 0.25
        stack.append(frame)
    stack[0][0, 0, 0] = np.nan
    stack[1][1, 1, 2] = np.inf
    return stack


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare CPU/GPU PixInsight WSC parity on synthetic data.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--kappa", type=float, default=3.0)
    args = parser.parse_args()

    frames = _make_frames(args.seed, args.frames, args.height, args.width)
    zconfig_cpu = SimpleNamespace(
        stack_use_gpu=False,
        use_gpu_stack=False,
        use_gpu=False,
        poststack_equalize_rgb=False,
    )
    zconfig_gpu = SimpleNamespace(
        stack_use_gpu=True,
        use_gpu_stack=True,
        use_gpu=True,
        poststack_equalize_rgb=False,
    )

    cpu_out, _ = zas.stack_winsorized_sigma_clip(
        frames,
        zconfig=zconfig_cpu,
        stack_metadata={},
        winsor_limits=(0.05, 0.05),
        kappa=float(args.kappa),
        apply_rewinsor=True,
        progress_callback=None,
    )

    if not _gpu_available():
        print("GPU unavailable; CPU output computed. Skipping GPU comparison.")
        return 0

    gpu_out, _ = zas.stack_winsorized_sigma_clip(
        frames,
        zconfig=zconfig_gpu,
        stack_metadata={},
        winsor_limits=(0.05, 0.05),
        kappa=float(args.kappa),
        apply_rewinsor=True,
        progress_callback=None,
    )

    diff = np.abs(cpu_out.astype(np.float32) - gpu_out.astype(np.float32))
    max_abs = float(np.nanmax(diff)) if diff.size else 0.0
    print(f"max_abs_diff={max_abs:.9g}")
    if max_abs == 0.0:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
