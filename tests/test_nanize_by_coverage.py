import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zemosaic_worker import _nanize_by_coverage


def test_nanize_by_coverage_marks_zeros_and_alpha():
    mosaic = np.ones((2, 2, 3), dtype=np.float32)
    coverage = np.array([[1, 0], [0, 2]], dtype=np.float32)
    alpha = np.array([[255, 0], [128, 0]], dtype=np.uint8)

    result = _nanize_by_coverage(mosaic, coverage, alpha_u8=alpha)

    expected_nan_mask = np.array([[False, True], [True, True]])
    assert result.shape == (2, 2, 3)
    assert result.dtype == np.float32
    assert np.isnan(result[expected_nan_mask]).all()
    assert np.isfinite(result[~expected_nan_mask]).all()

