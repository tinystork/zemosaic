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


def test_winsorized_sigma_clip_nan_safe():
    frames = []
    base_vals = [10.0, 12.0, 14.0]
    for idx, val in enumerate(base_vals):
        frame = np.full((6, 6, 3), val, dtype=np.float32)
        frame[:, :1, :] = np.nan
        if idx == 1:
            frame[:, -1:, :] = np.nan
        frames.append(frame)

    zconfig = SimpleNamespace(
        stack_use_gpu=False,
        use_gpu_stack=False,
        use_gpu=False,
        poststack_equalize_rgb=False,
    )

    stacked, _ = zas.stack_winsorized_sigma_clip(
        frames,
        zconfig=zconfig,
        stack_metadata={},
        winsor_limits=(0.05, 0.05),
        kappa=3.0,
        apply_rewinsor=True,
        progress_callback=None,
    )

    assert stacked.shape == frames[0].shape

    inner = stacked[:, 1:-1, :]
    assert np.all(np.isfinite(inner))
    assert np.allclose(inner, 12.0, atol=1e-3)

    left = stacked[:, :1, :]
    assert np.all(~np.isfinite(left) | (left == 0.0))

    right = stacked[:, -1:, :]
    assert np.all(np.isfinite(right))
    assert np.allclose(right, 12.0, atol=1e-3)
