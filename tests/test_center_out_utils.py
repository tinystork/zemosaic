import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zemosaic_utils import compute_sky_statistics, estimate_sky_affine_to_ref


def test_estimate_sky_affine_to_ref_recovers_linear_params():
    rng = np.random.default_rng(42)
    src = rng.uniform(10, 100, size=5000).astype(np.float32)
    gain = 1.2
    offset = -5.0
    ref = gain * src + offset
    # Add a few outliers to ensure clipping engages
    ref[:50] += rng.normal(0, 25, size=50)

    fit = estimate_sky_affine_to_ref(src, ref, sky_low=25.0, sky_high=60.0, clip_sigma=2.5)
    assert fit is not None
    est_gain, est_offset, sample_count = fit
    assert sample_count > 100
    assert abs(est_gain - gain) < 0.02
    assert abs(est_offset - offset) < 0.5


def test_compute_sky_statistics_matches_percentiles():
    data = np.linspace(0, 100, 101, dtype=np.float32)
    stats = compute_sky_statistics(data, 25.0, 75.0)
    assert stats is not None
    assert stats["median"] == np.median(data)
    assert stats["low"] == np.percentile(data, 25.0)
    assert stats["high"] == np.percentile(data, 75.0)
