import numpy as np
import pytest

import grid_mode


@pytest.mark.skipif(
    grid_mode.equalize_rgb_medians_inplace is None,
    reason="classic equalization helper unavailable",
)
def test_grid_equalization_matches_classic_medians():
    base = np.array(
        [
            [[0.2, 0.5, 1.0], [0.1, 0.4, 0.9]],
            [[0.3, 0.45, 1.2], [0.05, 0.55, 1.1]],
            [[0.6, 0.35, 0.8], [0.4, 0.25, 0.7]],
        ],
        dtype=np.float32,
    )
    arr_classic = base.copy()
    arr_grid = base.copy()
    gains_classic = grid_mode.equalize_rgb_medians_inplace(arr_classic)
    weight_map = np.ones((*base.shape[:2], 3), dtype=np.float32)
    arr_grid = grid_mode.grid_post_equalize_rgb(arr_grid, weight_map)

    med_classic = np.nanmedian(arr_classic, axis=(0, 1))
    med_grid = np.nanmedian(arr_grid, axis=(0, 1))
    assert np.allclose(med_classic, med_grid, atol=1e-5)

    original_medians = np.nanmedian(base, axis=(0, 1))
    target = np.nanmedian(original_medians[np.isfinite(original_medians) & (original_medians > 0)])
    expected_gains = target / original_medians
    assert np.all(np.sign(expected_gains) == np.sign(gains_classic[:3]))
