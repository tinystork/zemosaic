import os
import sys
from types import SimpleNamespace

# Ensure repository root is on sys.path so tests can import module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from zemosaic_worker import create_master_tile


def test_extreme_group_cleanup_after_tile(tmp_path):
    """
    Test that the EXTREME_GROUP mode works for a large group and that
    the implementation correctly handles cleanup after execution.
    Since parameters are local to create_master_tile, we test the behavior
    by verifying that a second call reverts to normal mode.
    """
    # Build a large synthetic group to trigger EXTREME_GROUP
    group_large = []
    for i in range(1500):
        group_large.append({
            "path_preprocessed_cache": str(tmp_path / f"cache_large_{i}.npy"),
            "path_raw": f"raw_large_{i}.fit",
            "header": {"SIMPLE": True},
            "wcs": SimpleNamespace(is_celestial=True),
        })

    # Build a small synthetic group to verify normal mode after
    group_small = []
    for i in range(5):
        group_small.append({
            "path_preprocessed_cache": str(tmp_path / f"cache_small_{i}.npy"),
            "path_raw": f"raw_small_{i}.fit",
            "header": {"SIMPLE": True},
            "wcs": SimpleNamespace(is_celestial=True),
        })

    messages = []

    def progress_callback(key, prog=None, lvl=None, **kwargs):
        messages.append((key, kwargs))

    # First call: process large group (should activate EXTREME_GROUP)
    try:
        (out, wcs), failed = create_master_tile(
            seestar_stack_group_info=group_large,
            tile_id=1,
            output_temp_dir=str(tmp_path),
            stack_norm_method="median",
            stack_weight_method="none",
            stack_reject_algo="kappa",
            stack_kappa_low=2.0,
            stack_kappa_high=2.0,
            parsed_winsor_limits=(0.05, 0.05),
            stack_final_combine="mean",
            poststack_equalize_rgb=False,
            apply_radial_weight=False,
            radial_feather_fraction=0.0,
            radial_shape_power=2.0,
            min_radial_weight_floor=0.0,
            quality_crop_enabled=False,
            quality_crop_band_px=32,
            quality_crop_k_sigma=2.0,
            quality_crop_margin_px=8,
            quality_crop_min_run=2,
            altaz_cleanup_enabled=False,
            altaz_margin_percent=5.0,
            altaz_decay=0.15,
            altaz_nanize=True,
            quality_gate_enabled=False,
            quality_gate_threshold=0.85,
            quality_gate_edge_band_px=8,
            quality_gate_k_sigma=2.0,
            quality_gate_erode_px=1,
            quality_gate_move_rejects=False,
            astap_exe_path_global="",
            astap_data_dir_global="",
            astap_search_radius_global="",
            astap_downsample_global=1,
            astap_sensitivity_global=1,
            astap_timeout_seconds_global=30,
            winsor_pool_workers=1,
            winsor_max_frames_per_pass=1000,
            progress_callback=progress_callback,
            resource_strategy=None,
            center_out_context=None,
            center_out_settings=None,
            center_out_rank=None,
            parallel_plan=None,
        )
    except (FileNotFoundError, OSError, KeyError):
        pass

    # Verify EXTREME_GROUP was activated for large group
    extreme_logs_first = [entry for entry in messages if entry[0] == "mastertile_extreme_group"]
    assert len(extreme_logs_first) > 0, "EXTREME_GROUP should have been activated for large group"

    # Second call: process small group (should NOT activate EXTREME_GROUP)
    messages.clear()
    try:
        (out, wcs), failed = create_master_tile(
            seestar_stack_group_info=group_small,
            tile_id=2,
            output_temp_dir=str(tmp_path),
            stack_norm_method="median",
            stack_weight_method="none",
            stack_reject_algo="kappa",
            stack_kappa_low=2.0,
            stack_kappa_high=2.0,
            parsed_winsor_limits=(0.05, 0.05),
            stack_final_combine="mean",
            poststack_equalize_rgb=False,
            apply_radial_weight=False,
            radial_feather_fraction=0.0,
            radial_shape_power=2.0,
            min_radial_weight_floor=0.0,
            quality_crop_enabled=False,
            quality_crop_band_px=32,
            quality_crop_k_sigma=2.0,
            quality_crop_margin_px=8,
            quality_crop_min_run=2,
            altaz_cleanup_enabled=False,
            altaz_margin_percent=5.0,
            altaz_decay=0.15,
            altaz_nanize=True,
            quality_gate_enabled=False,
            quality_gate_threshold=0.85,
            quality_gate_edge_band_px=8,
            quality_gate_k_sigma=2.0,
            quality_gate_erode_px=1,
            quality_gate_move_rejects=False,
            astap_exe_path_global="",
            astap_data_dir_global="",
            astap_search_radius_global="",
            astap_downsample_global=1,
            astap_sensitivity_global=1,
            astap_timeout_seconds_global=30,
            winsor_pool_workers=1,
            winsor_max_frames_per_pass=1000,
            progress_callback=progress_callback,
            resource_strategy=None,
            center_out_context=None,
            center_out_settings=None,
            center_out_rank=None,
            parallel_plan=None,
        )
    except (FileNotFoundError, OSError, KeyError):
        pass

    # Verify EXTREME_GROUP was NOT activated for small group
    extreme_logs_second = [entry for entry in messages if entry[0] == "mastertile_extreme_group"]
    assert len(extreme_logs_second) == 0, "EXTREME_GROUP should NOT have been activated for small group"
