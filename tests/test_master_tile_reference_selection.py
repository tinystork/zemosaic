import os
import sys
from types import SimpleNamespace

import numpy as np

# Ensure repository root is on sys.path so tests can import module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import zemosaic_worker as zw


def _make_info(idx: int, ra_deg: float, dec_deg: float, cache_path: str) -> dict:
    return {
        "path_preprocessed_cache": cache_path,
        "path_raw": f"raw_{idx}.fit",
        "header": {"CRVAL1": ra_deg, "CRVAL2": dec_deg},
        "wcs": SimpleNamespace(is_celestial=True),
    }


def test_pick_central_reference_index_basic():
    infos = []
    for idx, ra in enumerate([10, 11, 12, 13, 14]):
        infos.append(_make_info(idx, ra, 0.0, f"cache_{idx}.npy"))

    ref_idx = zw._pick_central_reference_index(infos, require_cache_exists=False)
    assert ref_idx == 2


def test_pick_central_reference_index_missing_middle():
    infos = []
    for idx, ra in enumerate([10, 11, 13, 14]):
        infos.append(_make_info(idx, ra, 0.0, f"cache_{idx}.npy"))

    ref_idx = zw._pick_central_reference_index(infos, require_cache_exists=False)
    assert ref_idx == 1


def test_failed_alignment_indices_map_to_group(monkeypatch, tmp_path):
    group = []
    for idx, ra in enumerate([10, 11, 12, 13]):
        cache_path = str(tmp_path / f"cache_{idx}.npy")
        group.append(_make_info(idx, ra, 0.0, cache_path))

    class DummyHeader(dict):
        pass

    class DummyAligner:
        @staticmethod
        def align_images_in_group(
            image_data_list,
            reference_image_index=0,
            propagate_mask=False,
            progress_callback=None,
        ):
            return [None for _ in image_data_list], [1]

    def fake_path_exists(path):
        return "cache_1.npy" not in str(path)

    def fake_load_cache(path, pcb=None, tile_id=None):
        return np.zeros((2, 2, 3), dtype=np.float32)

    monkeypatch.setattr(zw, "ZEMOSAIC_UTILS_AVAILABLE", True)
    monkeypatch.setattr(zw, "ZEMOSAIC_ALIGN_STACK_AVAILABLE", True)
    monkeypatch.setattr(zw, "ASTROPY_AVAILABLE", True)
    monkeypatch.setattr(zw, "zemosaic_utils", object())
    monkeypatch.setattr(zw, "zemosaic_align_stack", DummyAligner)
    monkeypatch.setattr(zw, "fits", SimpleNamespace(Header=DummyHeader))
    monkeypatch.setattr(zw, "_path_exists", fake_path_exists)
    monkeypatch.setattr(zw, "_safe_load_cache", fake_load_cache)

    (out, wcs), failed_groups = zw.create_master_tile(
        seestar_stack_group_info=group,
        tile_id=7,
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
        progress_callback=lambda *args, **kwargs: None,
        resource_strategy=None,
        center_out_context=None,
        center_out_settings=None,
        center_out_rank=None,
        parallel_plan=None,
        allow_batch_duplication=False,
        target_stack_size=4,
        min_safe_stack_size=3,
    )

    assert failed_groups, "Expected failed groups to retry from alignment failures"
    assert failed_groups[0][0].get("path_raw") == "raw_2.fit"
