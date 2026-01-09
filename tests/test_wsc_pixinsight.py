import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.robust_rejection import wsc_pixinsight_core

zas = pytest.importorskip("zemosaic_align_stack", reason="CPU stacker module unavailable on sys.path")

try:
    gpu_mod = pytest.importorskip("zemosaic_align_stack_gpu")
    _gpu_is_usable = getattr(gpu_mod, "_gpu_is_usable", lambda: False)
except Exception:
    gpu_mod = None
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


def _make_star_frame(shape, sigma, rng, noise_sigma=2.0, border=0, grid=5, amplitude=800.0):
    yy, xx = np.indices(shape)
    frame = rng.normal(loc=1000.0, scale=noise_sigma, size=shape).astype(np.float32)
    margin = max(border + 10, int(min(shape) * 0.1))
    max_y = max(margin, shape[0] - margin - 1)
    max_x = max(margin, shape[1] - margin - 1)
    if grid < 2:
        positions = [(shape[0] // 2, shape[1] // 2)]
    else:
        ys = np.linspace(margin, max_y, grid).astype(int)
        xs = np.linspace(margin, max_x, grid).astype(int)
        positions = [(int(y), int(x)) for y in ys for x in xs]
    denom = 2.0 * float(sigma) ** 2
    for y0, x0 in positions:
        frame += amplitude * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / denom)
    if border:
        frame[:border, :] = np.nan
        frame[-border:, :] = np.nan
        frame[:, :border] = np.nan
        frame[:, -border:] = np.nan
    return frame


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


def test_noise_fwhm_nan_borders_produces_weights():
    if not getattr(zas, "PHOTOUTILS_AVAILABLE", False) or not getattr(zas, "SIGMA_CLIP_AVAILABLE", False):
        pytest.skip("Photutils/Astropy not available for noise_fwhm weighting")

    rng = np.random.default_rng(123)
    shape = (128, 128)
    sigmas = [1.5, 2.5, 3.5]
    frames = [
        _make_star_frame(shape, sigma, rng, noise_sigma=0.8, border=16, grid=5, amplitude=800.0)
        for sigma in sigmas
    ]

    weights, method_used, _stats = zas._compute_quality_weights(
        frames, "noise_fwhm", progress_callback=None
    )

    assert method_used == "noise_fwhm"
    assert weights is not None
    values = [float(np.nanmean(w)) for w in weights if w is not None]
    assert values
    assert all(np.isfinite(v) for v in values)
    assert len(set(np.round(values, 3))) > 1


def test_prepare_frames_and_weights_keeps_quality_weights_without_radial():
    if gpu_mod is None:
        pytest.skip("GPU stacker module unavailable on sys.path")

    rng = np.random.default_rng(321)
    shape = (128, 128)
    sigmas = [1.5, 2.5, 3.5]
    noise_scales = [0.8, 1.2, 1.6]
    frames = [
        _make_star_frame(shape, sigma, rng, noise_sigma=noise, border=16, grid=5, amplitude=800.0)
        for sigma, noise in zip(sigmas, noise_scales)
    ]
    stacking_params = {
        "stack_norm_method": "none",
        "stack_reject_algo": "winsorized_sigma_clip",
        "stack_weight_method": "noise_fwhm",
        "apply_radial_weight": False,
    }

    (
        _prepared_frames,
        _weights_stack,
        weight_method_used,
        _weight_stats,
        _expanded_channels,
        wsc_weights_block,
        _wsc_weight_method,
        _wsc_weight_stats,
        _sky_offsets,
    ) = gpu_mod._prepare_frames_and_weights(
        frames,
        stacking_params,
        zconfig=_cpu_config(),
        pcb_tile=None,
        logger=None,
    )

    assert weight_method_used != "none"
    assert wsc_weights_block is not None


def test_p3_wsc_prep_applies_sky_mean_normalization():
    if gpu_mod is None:
        pytest.skip("GPU stacker module unavailable on sys.path")

    frames = [
        np.full((6, 6, 3), 100.0, dtype=np.float32),
        np.full((6, 6, 3), 120.0, dtype=np.float32),
        np.full((6, 6, 3), 80.0, dtype=np.float32),
    ]
    stacking_params = {
        "stack_norm_method": "sky_mean",
        "stack_reject_algo": "winsorized_sigma_clip",
        "stack_weight_method": "none",
    }

    (
        prepared_frames,
        _weights_stack,
        _weight_method_used,
        _weight_stats,
        _expanded_channels,
        _wsc_weights_block,
        _wsc_weight_method,
        _wsc_weight_stats,
        sky_offsets,
    ) = gpu_mod._prepare_frames_and_weights(
        frames,
        stacking_params,
        zconfig=_cpu_config(),
        pcb_tile=None,
        logger=None,
    )

    assert sky_offsets is None
    ref_mean = float(np.mean(prepared_frames[0]))
    assert np.isclose(ref_mean, 100.0, atol=1e-5, rtol=0.0)
    assert np.isclose(float(np.mean(prepared_frames[1])), ref_mean, atol=1e-5, rtol=0.0)
    assert np.isclose(float(np.mean(prepared_frames[2])), ref_mean, atol=1e-5, rtol=0.0)
    assert not np.allclose(prepared_frames[1], frames[1])


def test_p3_wsc_prep_sky_mean_low_mem_returns_offsets(monkeypatch):
    if gpu_mod is None:
        pytest.skip("GPU stacker module unavailable on sys.path")

    # Force low-memory normalization mode without allocating massive frames.
    monkeypatch.setattr(gpu_mod, "_available_ram_bytes", lambda: 1)

    frames = [
        np.full((6, 6, 3), 100.0, dtype=np.float32),
        np.full((6, 6, 3), 120.0, dtype=np.float32),
        np.full((6, 6, 3), 80.0, dtype=np.float32),
    ]
    stacking_params = {
        "stack_norm_method": "sky_mean",
        "stack_reject_algo": "winsorized_sigma_clip",
        "stack_weight_method": "none",
    }

    (
        prepared_frames,
        _weights_stack,
        _weight_method_used,
        _weight_stats,
        _expanded_channels,
        _wsc_weights_block,
        _wsc_weight_method,
        _wsc_weight_stats,
        sky_offsets,
    ) = gpu_mod._prepare_frames_and_weights(
        frames,
        stacking_params,
        zconfig=_cpu_config(),
        pcb_tile=None,
        logger=None,
    )

    assert sky_offsets is not None
    assert np.allclose(np.asarray(sky_offsets, dtype=np.float32), np.asarray([0.0, -20.0, 20.0], dtype=np.float32))
    # In low-mem mode, frames are left untouched and offsets are applied per chunk during stacking.
    assert np.isclose(float(np.mean(prepared_frames[1])), 120.0, atol=1e-5, rtol=0.0)


def test_wsc_pixinsight_core_weights_block_changes_output():
    stack = np.stack(
        [
            np.full((5, 5), 10.0, dtype=np.float32),
            np.full((5, 5), 12.0, dtype=np.float32),
            np.full((5, 5), 14.0, dtype=np.float32),
        ],
        axis=0,
    )
    weights_a = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    weights_b = np.asarray([1.0, 1.0, 10.0], dtype=np.float32)

    out_a = wsc_pixinsight_core(
        np,
        stack,
        sigma_low=5.0,
        sigma_high=5.0,
        max_iters=5,
        weights_block=weights_a,
    )
    out_b = wsc_pixinsight_core(
        np,
        stack,
        sigma_low=5.0,
        sigma_high=5.0,
        max_iters=5,
        weights_block=weights_b,
    )

    diff = abs(float(np.mean(out_a)) - float(np.mean(out_b)))
    assert diff > 0.5


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
