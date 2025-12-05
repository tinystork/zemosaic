import sys
from pathlib import Path

import logging
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import zemosaic_utils as zu
import zemosaic_worker as zw


def _make_wcs(size: int = 2):
    pytest.importorskip("astropy")
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.crpix = [size / 2, size / 2]
    w.wcs.cdelt = np.array([-1.0, 1.0])
    w.wcs.crval = [0.0, 0.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (size, size)
    return w


def test_reproject_wrapper_respects_weight_maps():
    """Per-tile weight maps must drive the blended mosaic and coverage."""

    pytest.importorskip("astropy")
    pytest.importorskip("reproject")
    wcs = _make_wcs(size=2)

    arr1 = np.full((2, 2), 1.0, dtype=np.float32)
    arr2 = np.full((2, 2), 3.0, dtype=np.float32)
    weight1 = np.array([[1.0, 0.5], [1.0, 0.5]], dtype=np.float32)
    weight2 = np.array([[0.2, 1.0], [0.2, 1.0]], dtype=np.float32)

    mosaic, coverage = zu.reproject_and_coadd_wrapper(
        data_list=[arr1, arr2],
        wcs_list=[wcs, wcs],
        shape_out=(2, 2),
        output_projection=wcs,
        use_gpu=False,
        tile_weight_maps=[weight1, weight2],
        combine_function="mean",
        reproject_function=zu.reproject_interp,
    )

    expected = np.array(
        [
            [(1 * 1.0 + 3 * 0.2) / (1.0 + 0.2), (1 * 0.5 + 3 * 1.0) / (0.5 + 1.0)],
            [(1 * 1.0 + 3 * 0.2) / (1.0 + 0.2), (1 * 0.5 + 3 * 1.0) / (0.5 + 1.0)],
        ],
        dtype=np.float32,
    )
    expected_cov = np.array([[1.2, 1.5], [1.2, 1.5]], dtype=np.float32)

    np.testing.assert_allclose(mosaic, expected)
    np.testing.assert_allclose(coverage, expected_cov)


def test_radial_weighting_propagates_to_coverage(tmp_path):
    """Radial feathering should taper the coverage map in the final coadd."""

    pytest.importorskip("astropy")
    pytest.importorskip("reproject")
    from astropy.io import fits

    wcs = _make_wcs(size=4)
    data = np.ones((4, 4), dtype=np.float32)
    tile_path = tmp_path / "tile.fits"
    fits.writeto(tile_path, data, header=wcs.to_header(), overwrite=True)

    mosaic, coverage, _ = zw.assemble_final_mosaic_reproject_coadd(
        master_tile_fits_with_wcs_list=[(str(tile_path), wcs)],
        final_output_wcs=wcs,
        final_output_shape_hw=(4, 4),
        progress_callback=None,
        n_channels=1,
        match_bg=False,
        apply_crop=False,
        use_gpu=False,
        apply_radial_weight=True,
        radial_feather_fraction=0.8,
        radial_shape_power=2.0,
        min_radial_weight_floor=0.0,
    )

    radial_map = zw.zemosaic_utils.make_radial_weight_map(
        4, 4, feather_fraction=0.8, shape_power=2.0, min_weight_floor=0.0
    )

    assert mosaic is not None and coverage is not None
    np.testing.assert_allclose(mosaic[..., 0], data)
    np.testing.assert_allclose(coverage, radial_map, rtol=1e-5, atol=1e-6)


def test_phase5_forwards_radial_config(monkeypatch, tmp_path):
    """Phase 5 must forward radial config to reproject+coadd without crashing."""

    captured = {}

    def fake_assemble(*args, **kwargs):
        captured.update(
            {
                "apply": kwargs.get("apply_radial_weight"),
                "feather": kwargs.get("radial_feather_fraction"),
                "power": kwargs.get("radial_shape_power"),
                "floor": kwargs.get("min_radial_weight_floor"),
            }
        )
        h, w = kwargs.get("final_output_shape_hw") or (2, 2)
        data = np.zeros((h, w, 3), dtype=np.float32)
        cov = np.ones((h, w), dtype=np.float32)
        alpha = np.ones((h, w), dtype=np.uint8)
        return data, cov, alpha

    monkeypatch.setattr(zw, "assemble_final_mosaic_reproject_coadd", fake_assemble)
    monkeypatch.setattr(zw, "_apply_phase5_post_stack_pipeline", lambda data, cov, alpha, **_: (data, cov, alpha))
    monkeypatch.setattr(zw, "_derive_final_alpha_mask", lambda alpha_map, data, cov, logger: alpha_map)
    monkeypatch.setattr(zw, "_log_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(zw, "reset_phase5_gpu_runtime_state", lambda: None)
    monkeypatch.setattr(zw, "phase5_gpu_runtime_state", lambda: (False, None, None))
    monkeypatch.setattr(zw, "PARALLEL_HELPERS_AVAILABLE", False)

    tile_path = tmp_path / "tile.fits"
    tile_path.write_text("x")

    dummy_wcs = SimpleNamespace(is_celestial=True, pixel_shape=(2, 2))
    phase45_options = {"base_progress": 0.0, "progress_weight": 0.0, "enable": False, "worker_config": {}}
    phase5_options = {
        "base_progress": 0.0,
        "progress_weight": 0.0,
        "final_assembly_method": "reproject_coadd",
        "apply_master_tile_crop": False,
        "quality_crop_enabled": False,
        "master_tile_crop_percent": 0.0,
        "intertile_match_flag": False,
        "match_background_flag": False,
        "feather_parity_flag": False,
        "two_pass_enabled": False,
        "two_pass_sigma_px": 0,
        "two_pass_gain_clip": (0.9, 1.1),
        "two_pass_coverage_renorm": False,
        "use_gpu_phase5": False,
        "assembly_process_workers": 0,
        "intertile_preview_size": 64,
        "intertile_overlap_min": 0.05,
        "intertile_sky_percentile": (30.0, 70.0),
        "intertile_robust_clip_sigma": 2.5,
        "intertile_global_recenter": False,
        "intertile_recenter_clip": (0.85, 1.18),
        "use_auto_intertile": False,
        "coadd_use_memmap": False,
        "coadd_memmap_dir": None,
        "global_anchor_shift": None,
        "parallel_plan": None,
        "parallel_capabilities": None,
        "telemetry": None,
        "sds_mode": False,
        "tile_weighting_enabled": False,
        "tile_weight_mode": "n_frames",
        "apply_radial_weight": True,
        "radial_feather_fraction": 0.6,
        "radial_shape_power": 2.0,
        "min_radial_weight_floor": 0.05,
    }

    results = zw._run_shared_phase45_phase5_pipeline(
        [(str(tile_path), dummy_wcs)],
        final_output_wcs=SimpleNamespace(),
        final_output_shape_hw=(2, 2),
        temp_master_tile_storage_dir=None,
        output_folder=str(tmp_path),
        cache_retention_mode="",
        phase45_options=phase45_options,
        phase5_options=phase5_options,
        final_quality_pipeline_cfg={},
        start_time_total_run=None,
        progress_callback=None,
        pcb=lambda *args, **kwargs: None,
        logger=logging.getLogger("test_phase5_forward"),
    )

    assert results[1] is not None and results[2] is not None
    assert captured == {
        "apply": True,
        "feather": 0.6,
        "power": 2.0,
        "floor": 0.05,
    }
