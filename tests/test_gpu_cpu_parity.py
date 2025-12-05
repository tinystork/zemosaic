import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Ensure repository root is importable when running tests directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

zas = pytest.importorskip("zemosaic_align_stack", reason="CPU stacker module unavailable on sys.path")

try:
    gpu_mod = importlib.import_module("zemosaic_align_stack_gpu")
except ImportError as exc:
    pytest.skip(f"GPU stacker module unavailable: {exc}", allow_module_level=True)

GPUStackingError = gpu_mod.GPUStackingError
_gpu_is_usable = gpu_mod._gpu_is_usable
gpu_stack_from_arrays = gpu_mod.gpu_stack_from_arrays


@pytest.mark.skipif(not _gpu_is_usable(), reason="CuPy/CUDA unavailable for GPU parity check")
def test_gpu_stack_matches_cpu_noise_variance():
    rng = np.random.default_rng(42)
    base = rng.normal(loc=1000.0, scale=35.0, size=(12, 8, 3)).astype(np.float32)
    frames = []
    for idx in range(4):
        noise = rng.normal(scale=6.0 + idx * 3.0, size=base.shape).astype(np.float32)
        frames.append(np.clip(base + noise + idx * 2.0, 0, None))

    zconfig = SimpleNamespace(
        stack_use_gpu=True,
        use_gpu_stack=True,
        use_gpu=True,
        poststack_equalize_rgb=False,
    )

    cpu_stack = zas.stack_aligned_images(
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
    assert cpu_stack is not None

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
        gpu_stack, _ = gpu_stack_from_arrays(
            frames,
            stacking_params,
            parallel_plan=None,
            logger=None,
            pcb_tile=None,
            tile_id=None,
            zconfig=zconfig,
        )
    except GPUStackingError as exc:
        pytest.skip(f"GPU stack unavailable: {exc}")

    assert gpu_stack.shape == cpu_stack.shape

    diff = np.abs(cpu_stack.astype(np.float32) - gpu_stack.astype(np.float32))
    max_diff = float(np.nanmax(diff))
    med_diff = np.abs(np.nanmedian(cpu_stack, axis=(0, 1)) - np.nanmedian(gpu_stack, axis=(0, 1)))
    # Temporary diagnostics to understand parity gap on remote GPU runs.
    print("cpu median", np.nanmedian(cpu_stack, axis=(0, 1)))
    print("gpu median", np.nanmedian(gpu_stack, axis=(0, 1)))
    print("max diff", max_diff)
    max_idx = np.unravel_index(np.nanargmax(diff), diff.shape)
    print("max diff location/value", max_idx, diff[max_idx])

    assert max_diff < 1e-3
    assert np.all(med_diff < 1e-3)
