#!/usr/bin/env python3
"""Debug helper to compare CPU and GPU stacking outputs on synthetic frames.

This script is intentionally out of the main execution path. Run it manually
when you want a quick CPU vs GPU parity check without invoking the full GUI.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_modules():
    try:
        cpu_mod = importlib.import_module("zemosaic_align_stack")
    except ImportError as exc:  # pragma: no cover - debug helper
        print(f"CPU stacker unavailable: {exc}")
        return None, None

    try:
        gpu_mod = importlib.import_module("zemosaic_align_stack_gpu")
    except ImportError as exc:  # pragma: no cover - debug helper
        print(f"GPU stacker unavailable: {exc}")
        return cpu_mod, None

    return cpu_mod, gpu_mod


def _build_frames(args: argparse.Namespace) -> list[np.ndarray]:
    rng = np.random.default_rng(args.seed)
    base = rng.normal(
        loc=args.base_mean,
        scale=args.base_sigma,
        size=(args.height, args.width, args.channels),
    ).astype(np.float32)

    frames: list[np.ndarray] = []
    for idx in range(args.frames):
        noise = rng.normal(
            scale=args.noise + idx * args.noise_growth,
            size=base.shape,
        ).astype(np.float32)
        frames.append(np.clip(base + noise + idx * args.bias_step, 0, None))
    return frames


def run_parity_check(args: argparse.Namespace) -> int:
    cpu_mod, gpu_mod = _load_modules()
    if cpu_mod is None or gpu_mod is None:
        return 2

    if not getattr(gpu_mod, "_gpu_is_usable", lambda: False)():
        print("GPU not usable; skipping comparison.")
        return 2

    frames = _build_frames(args)
    zconfig = SimpleNamespace(
        stack_use_gpu=True,
        use_gpu_stack=True,
        use_gpu=True,
        poststack_equalize_rgb=False,
    )

    cpu_stack = cpu_mod.stack_aligned_images(
        frames,
        normalize_method="none",
        weighting_method="noise_variance",
        rejection_algorithm="none",
        final_combine_method="mean",
        sigma_clip_low=3.0,
        sigma_clip_high=3.0,
        winsor_limits=(0.05, 0.05),
        minimum_signal_adu_target=0.0,
        apply_radial_weight=False,
        radial_feather_fraction=0.8,
        radial_shape_power=2.0,
        winsor_max_workers=1,
        progress_callback=None,
        zconfig=zconfig,
        stack_metadata={},
        parallel_plan=None,
    )
    if cpu_stack is None:
        print("CPU stack failed.")
        return 1

    stacking_params = {
        "stack_norm_method": "none",
        "stack_weight_method": "noise_variance",
        "stack_reject_algo": "none",
        "stack_kappa_low": 3.0,
        "stack_kappa_high": 3.0,
        "parsed_winsor_limits": (0.05, 0.05),
        "stack_final_combine": "mean",
        "apply_radial_weight": False,
        "radial_feather_fraction": 0.8,
        "radial_shape_power": 2.0,
        "poststack_equalize_rgb": False,
    }

    try:
        gpu_stack, _ = gpu_mod.gpu_stack_from_arrays(
            frames,
            stacking_params,
            parallel_plan=None,
            logger=None,
            pcb_tile=None,
            tile_id=None,
            zconfig=zconfig,
        )
    except gpu_mod.GPUStackingError as exc:
        print(f"GPU stack failed: {exc}")
        return 1

    if gpu_stack is None:
        print("GPU stack returned no data.")
        return 1

    diff = np.abs(cpu_stack.astype(np.float32) - gpu_stack.astype(np.float32))
    max_diff = float(np.nanmax(diff))
    med_cpu = np.nanmedian(cpu_stack, axis=(0, 1))
    med_gpu = np.nanmedian(gpu_stack, axis=(0, 1))
    med_diff = np.abs(med_cpu - med_gpu)

    print(f"CPU median per channel: {med_cpu}")
    print(f"GPU median per channel: {med_gpu}")
    print(f"Max abs diff: {max_diff:.6f}")
    print(f"Median abs diff per channel: {med_diff}")

    ok = max_diff <= args.tolerance and np.all(med_diff <= args.tolerance)
    if not ok:
        print(f"Parity check FAILED (tolerance {args.tolerance}).")
        return 1

    print(f"Parity check OK (tolerance {args.tolerance}).")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU vs GPU stacking parity check (debug).")
    parser.add_argument("--frames", type=int, default=4, help="Number of synthetic frames.")
    parser.add_argument("--height", type=int, default=12, help="Frame height.")
    parser.add_argument("--width", type=int, default=8, help="Frame width.")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels.")
    parser.add_argument("--base-mean", type=float, default=1000.0, help="Mean of the base signal.")
    parser.add_argument(
        "--base-sigma",
        type=float,
        default=35.0,
        help="Standard deviation of the base signal.",
    )
    parser.add_argument("--bias-step", type=float, default=2.0, help="Bias increment per frame.")
    parser.add_argument("--noise", type=float, default=6.0, help="Baseline noise level.")
    parser.add_argument("--noise-growth", type=float, default=3.0, help="Noise increment per frame.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Allowed max/median absolute difference.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return run_parity_check(args)


if __name__ == "__main__":  # pragma: no cover - manual debug entry point
    raise SystemExit(main())
