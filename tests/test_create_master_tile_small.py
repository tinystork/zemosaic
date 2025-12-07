import os
import sys
import numpy as np

from types import SimpleNamespace

# Ensure repository root is on sys.path so tests can import module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from zemosaic_worker import create_master_tile


def test_create_master_tile_small_group(tmp_path):
    # Build a small synthetic group (5 entries). We don't need real .npy caches
    # because create_master_tile will early-return when dependencies are missing,
    # but it will still emit logs for EXTREME_GROUP if wrongly activated.
    group = []
    for i in range(5):
        group.append({
            "path_preprocessed_cache": str(tmp_path / f"cache_{i}.npy"),
            "path_raw": f"raw_{i}.fit",
            "header": {"SIMPLE": True},
            "wcs": SimpleNamespace(is_celestial=True),
        })

    messages = []

    def progress_callback(key, prog=None, lvl=None, **kwargs):
        messages.append((key, kwargs))

    # Call with small group to ensure EXTREME_GROUP is not activated
    (out, wcs), failed = create_master_tile(
        seestar_stack_group_info=group,
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
        astap_search_radius_global=0.0,
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

    # Ensure EXTREME_GROUP log was not emitted for this small group
    assert not any(key == "mastertile_extreme_group" for key, _ in messages)
