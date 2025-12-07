import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import zemosaic_worker as zw


@pytest.fixture(scope="module")
def _fake_pipeline_mask():
    mask = np.ones((4, 4), dtype=np.float32)
    mask[0, :] = 0.0
    mask[-1, :] = 0.0
    return mask


def test_sds_post_pipeline_matches_non_sds(monkeypatch, _fake_pipeline_mask):
    """Ensure quality options behave identically with or without SDS."""

    base_data = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    base_cov = np.ones((4, 4), dtype=np.float32)
    base_alpha = np.full((4, 4), 255, dtype=np.uint8)

    crop_mask = np.ones_like(_fake_pipeline_mask)
    crop_mask[:, 0] = 0.0
    crop_mask[:, -1] = 0.0
    expected_mask = _fake_pipeline_mask * crop_mask

    expected_data = np.where(expected_mask[..., None] > 0, base_data, np.nan)
    expected_cov = base_cov * expected_mask
    expected_alpha = (expected_mask * 255.0).astype(np.uint8)

    def fake_pipeline(arr, cfg):
        masked = np.where(_fake_pipeline_mask[..., None] > 0, arr, np.nan)
        return masked, _fake_pipeline_mask.copy()

    monkeypatch.setattr(zw, "_apply_lecropper_pipeline", fake_pipeline)

    pipeline_cfg = {
        "quality_crop_enabled": True,
        "quality_crop_band_px": 32,
        "quality_crop_k_sigma": 2.0,
        "quality_crop_margin_px": 8,
        "quality_crop_min_run": 2,
        "altaz_cleanup_enabled": True,
        "altaz_margin_percent": 5.0,
        "altaz_decay": 0.15,
        "altaz_nanize": True,
    }

    sds_data, sds_cov, sds_alpha = zw._apply_phase5_post_stack_pipeline(
        base_data.copy(),
        base_cov.copy(),
        base_alpha.copy(),
        enable_lecropper_pipeline=True,
        pipeline_cfg=pipeline_cfg,
        enable_master_tile_crop=True,
        master_tile_crop_percent=25.0,
        two_pass_enabled=False,
        two_pass_sigma_px=50,
        two_pass_gain_clip=(0.85, 1.18),
        final_output_wcs=object(),
        final_output_shape_hw=(4, 4),
        use_gpu_two_pass=False,
        logger=None,
        collected_tiles=None,
        fallback_two_pass_loader=None,
    )

    np.testing.assert_allclose(sds_cov, expected_cov, equal_nan=True)
    np.testing.assert_allclose(sds_data, expected_data, equal_nan=True)
    assert np.array_equal(sds_alpha, expected_alpha)

    non_sds_data, non_sds_cov, non_sds_alpha = zw._apply_phase5_post_stack_pipeline(
        base_data.copy(),
        base_cov.copy(),
        base_alpha.copy(),
        enable_lecropper_pipeline=True,
        pipeline_cfg=pipeline_cfg,
        enable_master_tile_crop=True,
        master_tile_crop_percent=25.0,
        two_pass_enabled=False,
        two_pass_sigma_px=50,
        two_pass_gain_clip=(0.85, 1.18),
        final_output_wcs=object(),
        final_output_shape_hw=(4, 4),
        use_gpu_two_pass=False,
        logger=None,
        collected_tiles=None,
        fallback_two_pass_loader=None,
    )

    np.testing.assert_allclose(non_sds_cov, expected_cov, equal_nan=True)
    np.testing.assert_allclose(non_sds_data, expected_data, equal_nan=True)
    assert np.array_equal(non_sds_alpha, expected_alpha)

    np.testing.assert_allclose(sds_cov, non_sds_cov, equal_nan=True)
    np.testing.assert_allclose(sds_data, non_sds_data, equal_nan=True)
    assert np.array_equal(sds_alpha, non_sds_alpha)
