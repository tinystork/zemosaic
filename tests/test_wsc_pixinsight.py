import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

zas = pytest.importorskip("zemosaic_align_stack", reason="CPU stacker module unavailable on sys.path")

try:
    gpu_mod = pytest.importorskip("zemosaic_align_stack_gpu")
    _gpu_is_usable = getattr(gpu_mod, "_gpu_is_usable", lambda: False)
except Exception:
    _gpu_is_usable = lambda: False  # type: ignore[assignment]


def _cpu_config():
    return SimpleNamespace(
        stack_use_gpu=False,
        use_gpu_stack=False,
        use_gpu=False,
        poststack_equalize_rgb=False,
    )


def _gpu_config():
    return SimpleNamespace(
        stack_use_gpu=True,
        use_gpu_stack=True,
        use_gpu=True,
        poststack_equalize_rgb=False,
    )


def test_wsc_cosmic_ray_suppression():
    frames = [np.full((8, 8), 1000.0, dtype=np.float32) for _ in range(6)]
    frames[0][2, 3] = 10000.0

    stacked, _ = zas.stack_winsorized_sigma_clip(
        frames,
        zconfig=_cpu_config(),
        stack_metadata={},
        winsor_limits=(0.05, 0.05),
        kappa=3.0,
        apply_rewinsor=True,
        progress_callback=None,
    )

    assert stacked.shape == frames[0].shape
    assert abs(float(stacked[2, 3]) - 1000.0) < 1e-3


def test_wsc_dead_pixel_suppression():
    frames = [np.full((8, 8), 1000.0, dtype=np.float32) for _ in range(6)]
    frames[1][4, 5] = 0.0

    stacked, _ = zas.stack_winsorized_sigma_clip(
        frames,
        zconfig=_cpu_config(),
        stack_metadata={},
        winsor_limits=(0.05, 0.05),
        kappa=3.0,
        apply_rewinsor=True,
        progress_callback=None,
    )

    assert stacked.shape == frames[0].shape
    assert abs(float(stacked[4, 5]) - 1000.0) < 1e-3


def test_wsc_faint_diffuse_preserved_vs_kappa():
    rng = np.random.default_rng(123)
    n_frames = 8
    diffuse = 0.8
    frames = []
    for idx in range(n_frames):
        frame = rng.normal(loc=1000.0, scale=2.0, size=(16, 16)).astype(np.float32)
        frame += diffuse
        frames.append(frame)

    frames[0][3, 3] += 50.0
    frames[4][10, 5] -= 40.0

    wsc_out, _ = zas.stack_winsorized_sigma_clip(
        frames,
        zconfig=_cpu_config(),
        stack_metadata={},
        winsor_limits=(0.05, 0.05),
        kappa=3.0,
        apply_rewinsor=True,
        progress_callback=None,
    )

    kappa_out = zas.stack_aligned_images(
        frames,
        normalize_method="none",
        weighting_method="none",
        rejection_algorithm="kappa_sigma",
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
        zconfig=_cpu_config(),
        stack_metadata={},
        parallel_plan=None,
    )

    expected = 1000.0 + diffuse
    wsc_mean = float(np.mean(wsc_out))
    kappa_mean = float(np.mean(kappa_out))
    diff_wsc = abs(wsc_mean - expected)
    diff_kappa = abs(kappa_mean - expected)

    assert diff_wsc <= diff_kappa + 0.05
    assert diff_wsc <= 0.3


@pytest.mark.skipif(not _gpu_is_usable(), reason="CuPy/CUDA unavailable for WSC parity check")
def test_wsc_gpu_cpu_parity_strict():
    rng = np.random.default_rng(42)
    frames = []
    for idx in range(6):
        frame = rng.normal(loc=1000.0, scale=5.0, size=(12, 12, 3)).astype(np.float32)
        frame += idx * 0.25
        frames.append(frame)

    cpu_out, _ = zas.stack_winsorized_sigma_clip(
        frames,
        zconfig=_cpu_config(),
        stack_metadata={},
        winsor_limits=(0.05, 0.05),
        kappa=3.0,
        apply_rewinsor=True,
        progress_callback=None,
    )

    gpu_out, _ = zas.stack_winsorized_sigma_clip(
        frames,
        zconfig=_gpu_config(),
        stack_metadata={},
        winsor_limits=(0.05, 0.05),
        kappa=3.0,
        apply_rewinsor=True,
        progress_callback=None,
    )

    diff = np.abs(cpu_out.astype(np.float32) - gpu_out.astype(np.float32))
    assert float(np.nanmax(diff)) == 0.0
