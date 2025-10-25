import numpy as np
from types import SimpleNamespace

import zemosaic_align_stack as zas


def test_winsorized_sigma_clip_memmap_streaming(tmp_path):
    # Create a memmap file with multiple frames to exercise the streaming CPU path
    frame_shape = (4, 6, 6)
    memmap_path = tmp_path / "winsor_memmap.dat"
    data = np.random.rand(*frame_shape).astype(np.float32)
    mm = np.memmap(memmap_path, mode="w+", dtype=np.float32, shape=frame_shape)
    mm[:] = data
    del mm  # flush to disk

    mm_read = np.memmap(memmap_path, mode="r", dtype=np.float32, shape=frame_shape)
    zconfig = SimpleNamespace(use_gpu=False, winsor_max_frames_per_pass=2)

    stacked, rejected = zas.stack_winsorized_sigma_clip(
        mm_read,
        zconfig=zconfig,
        winsor_max_frames_per_pass=2,
    )

    assert stacked.shape == frame_shape[1:]
    assert stacked.dtype == np.float32
    assert np.isfinite(stacked).all()
    assert isinstance(rejected, float)


def test_winsorized_sigma_clip_handles_none_limits():
    frames = np.random.rand(4, 5, 5).astype(np.float32)
    stacked, rejected = zas.stack_winsorized_sigma_clip(frames, winsor_limits=None)

    assert stacked.shape == (5, 5)
    assert stacked.dtype == np.float32
    assert isinstance(rejected, float)


def test_stack_aligned_images_defaults_winsor_limits():
    imgs = [np.random.rand(6, 6, 3).astype(np.float32) for _ in range(4)]
    result = zas.stack_aligned_images(
        imgs,
        normalize_method="none",
        weighting_method="none",
        rejection_algorithm="winsorized_sigma_clip",
        final_combine_method="mean",
        winsor_limits=None,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (6, 6, 3)
    assert result.dtype == np.float32
