import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

zas = pytest.importorskip("zemosaic_align_stack", reason="CPU stacker module unavailable on sys.path")


def _cpu_only_zconfig() -> SimpleNamespace:
    return SimpleNamespace(
        stack_use_gpu=False,
        use_gpu_stack=False,
        use_gpu=False,
        poststack_equalize_rgb=False,
    )


def _worker_source() -> str:
    return (REPO_ROOT / "zemosaic_worker.py").read_text(encoding="utf-8", errors="ignore")


def _adaptation_block() -> str:
    src = _worker_source()
    start = src.find("# Per-tile adaptive working-set control (S3): adjust pass/chunk sizing only.")
    end = src.find("master_tile_stacked_HWC, stack_metadata, used_gpu = _stack_master_tile_auto(", start)
    assert start >= 0 and end > start
    return src[start:end]


def test_winsor_pass_split_matches_full_membership_output():
    rng = np.random.default_rng(123)
    frames = []
    for i in range(9):
        base = np.full((32, 24, 3), 1000.0 + i * 2.0, dtype=np.float32)
        noise = rng.normal(0.0, 0.8, size=base.shape).astype(np.float32)
        frame = base + noise
        if i == 8:
            frame[5:10, 5:10, :] += 120.0  # outlier patch
        frames.append(frame)

    zconfig = _cpu_only_zconfig()

    full, _ = zas.stack_winsorized_sigma_clip(
        frames,
        zconfig=zconfig,
        stack_metadata={},
        winsor_limits=(0.05, 0.05),
        kappa=3.0,
        apply_rewinsor=True,
        winsor_max_frames_per_pass=0,
        winsor_max_workers=1,
        progress_callback=None,
    )

    split, _ = zas.stack_winsorized_sigma_clip(
        frames,
        zconfig=zconfig,
        stack_metadata={},
        winsor_limits=(0.05, 0.05),
        kappa=3.0,
        apply_rewinsor=True,
        winsor_max_frames_per_pass=3,
        winsor_max_workers=1,
        progress_callback=None,
    )

    assert full.shape == split.shape
    assert np.allclose(full, split, equal_nan=True, atol=1e-4, rtol=1e-4)


def test_kappa_chunk_hint_matches_default_output():
    rng = np.random.default_rng(321)
    frames = []
    for i in range(7):
        frame = rng.normal(loc=500.0 + i * 0.5, scale=1.2, size=(28, 20, 3)).astype(np.float32)
        frames.append(frame)

    zconfig = _cpu_only_zconfig()

    ref, _ = zas.stack_kappa_sigma_clip(
        frames,
        zconfig=zconfig,
        stack_metadata={},
        sigma_low=3.0,
        sigma_high=3.0,
        progress_callback=None,
        parallel_plan=None,
    )

    hinted_plan = SimpleNamespace(rows_per_chunk=8, max_chunk_bytes=64 * 1024 * 1024)
    hinted, _ = zas.stack_kappa_sigma_clip(
        frames,
        zconfig=zconfig,
        stack_metadata={},
        sigma_low=3.0,
        sigma_high=3.0,
        progress_callback=None,
        parallel_plan=hinted_plan,
    )

    assert ref.shape == hinted.shape
    assert np.allclose(ref, hinted, equal_nan=True, atol=1e-4, rtol=1e-4)


def test_adaptation_block_keeps_logical_membership_contract():
    block = _adaptation_block()

    assert "Scientific membership is preserved" in block
    assert "n_frames=int(len(valid_aligned_images))" in block

    forbidden_patterns = [
        "valid_aligned_images = valid_aligned_images[",
        "valid_aligned_images = [img for img in valid_aligned_images",
        "valid_aligned_images.pop(",
        "valid_aligned_images.remove(",
        "del valid_aligned_images[",
        "valid_aligned_images = []",
    ]
    for pattern in forbidden_patterns:
        assert pattern not in block


def test_phase3_runtime_controller_backoff_hysteresis_and_recovery_markers_present():
    src = _worker_source()

    required_markers = [
        "if ram_used_pct >= ram_critical_pct:",
        "elif ram_used_pct >= ram_high_pct:",
        "elif ram_level == 1:",
        "else:  # ram_level == 2",
        "elif ram_used_pct <= ram_low_pct:",
        "new_ph3_limit = min(new_ph3_limit, 1)",
        "ram_cap = max(1, int(math.ceil(int(actual_num_workers_ph3) * 0.5)))",
        "_phase3_budget_change_block_reason(",
        "budget_change_blocked_reason",
        "new_ph3_limit = max(new_ph3_limit, min(int(actual_num_workers_ph3)",
    ]

    for marker in required_markers:
        assert marker in src


def test_phase3_pass_and_chunk_shrink_markers_present():
    src = _worker_source()

    assert "target_fraction = 0.40 if pressure_level == 2 else 0.50" in src
    assert "winsor_after = max(min_pass, min(total_frames, int(max(1, winsor_after // 2))))" in src
    assert "scale = 0.50 if pressure_level >= 2 else 0.70" in src
    assert "adaptive_parallel_plan = replace(" in src


def test_phase3_runtime_controller_and_retry_contract_markers_present():
    src = _worker_source()

    assert 'runtime_launch_limit: dict[str, Any] = {"value": int(actual_num_workers_ph3), "pause_until": 0.0, "pause_reason": ""}' in src
    assert "if pause_until > time.monotonic():" in src
    assert "return 0" in src

    assert "RAM_ADAPT_RT: admission_pause" in src
    assert "RAM_ADAPT_RT: launch_budget_hold" in src

    # Retry path must re-enter lazy queue and not bypass launch control.
    assert "pending_launch_queue.append((filtered_retry_group, new_tile_id, retry_rank))" in src
    assert "_submit_master_tile_group(filtered_retry_group, new_tile_id, retry_rank)" not in src


def test_phase3_adaptation_telemetry_fields_markers_present():
    src = _worker_source()

    assert "P3_MEM_ADAPT_TILE" in src
    for marker in [
        "tile_id=tile_id",
        "level=int(pressure_level)",
        "ram_used_pct=None if ram_used_pct is None else float(ram_used_pct)",
        "ram_available_mb=None if ram_avail_mb is None else float(ram_avail_mb)",
        "reject_algo=reject_algo_norm",
        "combine_mode=combine_mode_norm",
        "pass_adapt_enabled=bool(pass_adapt_enabled)",
        "chunk_adapt_enabled=bool(chunk_adapt_enabled)",
        "winsor_before=int(winsor_before)",
        "winsor_after=int(winsor_after)",
        "rows_before=rows_before",
        "rows_after=rows_after",
        "chunk_before=chunk_before",
        "chunk_after=chunk_after",
        "n_frames=int(len(valid_aligned_images))",
    ]:
        assert marker in src

    assert "P3_MEM_ADAPT_MODE_CONSERVATIVE" in src



def test_create_master_tile_adaptation_preserves_membership_and_modes(monkeypatch, tmp_path):
    astropy_wcs = pytest.importorskip("astropy.wcs")
    astropy_fits = pytest.importorskip("astropy.io.fits")
    zw = pytest.importorskip("zemosaic_worker")
    pu = pytest.importorskip("parallel_utils")

    # Build synthetic cached frames used by Phase 3 tile processing.
    group = []
    for i in range(4):
        arr = np.full((18, 14, 3), 100.0 + i, dtype=np.float32)
        cache_path = tmp_path / f"frame_{i}.npy"
        np.save(cache_path, arr)

        w = astropy_wcs.WCS(naxis=2)
        w.wcs.crpix = [7.0, 9.0]
        w.wcs.cdelt = np.array([-0.00028, 0.00028])
        w.wcs.crval = [180.0, 45.0]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        hdr = astropy_fits.Header()
        hdr["EXPTIME"] = 10.0

        group.append(
            {
                "path_preprocessed_cache": str(cache_path),
                "path_raw": f"raw_{i}.fit",
                "wcs": w,
                "header": hdr,
            }
        )

    # Force pressure path so adaptive knobs are exercised.
    monkeypatch.setattr(zw.psutil, "virtual_memory", lambda: SimpleNamespace(percent=95.0, available=64 * 1024 * 1024))

    # Keep alignment deterministic and simple in unit test.
    monkeypatch.setattr(zw.zemosaic_align_stack, "align_images_in_group", lambda image_data_list, reference_image_index, propagate_mask, progress_callback: (image_data_list, []))

    captured: dict = {}

    def _fake_stack_master_tile_auto(images, **kwargs):
        captured["n_images"] = len(images)
        captured["stack_reject_algo"] = kwargs.get("stack_reject_algo")
        captured["stack_final_combine"] = kwargs.get("stack_final_combine")
        captured["winsor_max_frames_per_pass"] = kwargs.get("winsor_max_frames_per_pass")
        captured["parallel_plan"] = kwargs.get("parallel_plan")
        return np.ones((18, 14, 3), dtype=np.float32), {"rgb_equalization": {}}, False

    monkeypatch.setattr(zw, "_stack_master_tile_auto", _fake_stack_master_tile_auto)

    def _fake_save_fits_image(image_data, output_path, **kwargs):
        Path(output_path).write_bytes(b"TEST")

    monkeypatch.setattr(zw.zemosaic_utils, "save_fits_image", _fake_save_fits_image)
    monkeypatch.setattr(zw, "_PH3_CONCURRENCY_SEMAPHORE", __import__("threading").Semaphore(8))
    monkeypatch.setattr(zw, "_CACHE_IO_SEMAPHORE", __import__("threading").Semaphore(8))

    plan = pu.ParallelPlan(
        cpu_workers=4,
        use_memmap=True,
        max_chunk_bytes=512 * 1024 * 1024,
        rows_per_chunk=256,
        use_gpu=False,
        gpu_rows_per_chunk=128,
        gpu_max_chunk_bytes=256 * 1024 * 1024,
    )

    (tile_path, _tile_wcs), retry_groups = zw.create_master_tile(
        seestar_stack_group_info=group,
        tile_id=1,
        output_temp_dir=str(tmp_path),
        stack_norm_method="none",
        stack_weight_method="none",
        stack_reject_algo="winsorized_sigma_clip",
        stack_kappa_low=3.0,
        stack_kappa_high=3.0,
        parsed_winsor_limits=(0.05, 0.05),
        stack_final_combine="mean",
        poststack_equalize_rgb=False,
        apply_radial_weight=False,
        radial_feather_fraction=0.0,
        radial_shape_power=1.0,
        min_radial_weight_floor=0.0,
        quality_crop_enabled=False,
        quality_crop_band_px=16,
        quality_crop_k_sigma=2.0,
        quality_crop_margin_px=8,
        quality_crop_min_run=4,
        altaz_cleanup_enabled=False,
        altaz_margin_percent=0.0,
        altaz_decay=0.0,
        altaz_nanize=False,
        quality_gate_enabled=False,
        quality_gate_threshold=0.0,
        quality_gate_edge_band_px=0,
        quality_gate_k_sigma=0.0,
        quality_gate_erode_px=0,
        quality_gate_move_rejects=False,
        astap_exe_path_global="",
        astap_data_dir_global="",
        astap_search_radius_global=0.0,
        astap_downsample_global=0,
        astap_sensitivity_global=0,
        astap_timeout_seconds_global=1,
        winsor_pool_workers=1,
        winsor_max_frames_per_pass=0,
        progress_callback=None,
        resource_strategy=None,
        center_out_context=None,
        center_out_settings=None,
        center_out_rank=None,
        parallel_plan=plan,
        allow_batch_duplication=False,
        target_stack_size=5,
        min_safe_stack_size=3,
        dbg_tile_ids=None,
    )

    assert retry_groups == []
    assert tile_path is not None
    assert captured["n_images"] == len(group)

    # No silent mode drift in adaptation path.
    assert captured["stack_reject_algo"] == "winsorized_sigma_clip"
    assert captured["stack_final_combine"] == "mean"

    # Under high pressure, pass/chunk knobs should tighten but still keep all frame membership.
    assert 1 <= int(captured["winsor_max_frames_per_pass"]) <= len(group)
    adapted_plan = captured["parallel_plan"]
    assert adapted_plan is not None
    assert adapted_plan.rows_per_chunk <= plan.rows_per_chunk
    assert adapted_plan.max_chunk_bytes <= plan.max_chunk_bytes



def test_create_master_tile_output_count_parity_baseline_vs_pressure(monkeypatch, tmp_path):
    astropy_wcs = pytest.importorskip("astropy.wcs")
    astropy_fits = pytest.importorskip("astropy.io.fits")
    zw = pytest.importorskip("zemosaic_worker")
    pu = pytest.importorskip("parallel_utils")

    group = []
    for i in range(3):
        arr = np.full((16, 12, 3), 50.0 + i, dtype=np.float32)
        cache_path = tmp_path / f"parity_{i}.npy"
        np.save(cache_path, arr)

        w = astropy_wcs.WCS(naxis=2)
        w.wcs.crpix = [6.0, 8.0]
        w.wcs.cdelt = np.array([-0.00028, 0.00028])
        w.wcs.crval = [180.0, 45.0]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        hdr = astropy_fits.Header()
        hdr["EXPTIME"] = 8.0

        group.append(
            {
                "path_preprocessed_cache": str(cache_path),
                "path_raw": f"raw_parity_{i}.fit",
                "wcs": w,
                "header": hdr,
            }
        )

    vm_state = {"percent": 30.0, "available": 1024 * 1024 * 1024}
    monkeypatch.setattr(zw.psutil, "virtual_memory", lambda: SimpleNamespace(percent=vm_state["percent"], available=vm_state["available"]))

    monkeypatch.setattr(zw.zemosaic_align_stack, "align_images_in_group", lambda image_data_list, reference_image_index, propagate_mask, progress_callback: (image_data_list, []))
    monkeypatch.setattr(zw, "_PH3_CONCURRENCY_SEMAPHORE", __import__("threading").Semaphore(8))
    monkeypatch.setattr(zw, "_CACHE_IO_SEMAPHORE", __import__("threading").Semaphore(8))

    def _fake_stack_master_tile_auto(images, **kwargs):
        return np.ones((16, 12, 3), dtype=np.float32), {"rgb_equalization": {}}, False

    monkeypatch.setattr(zw, "_stack_master_tile_auto", _fake_stack_master_tile_auto)
    monkeypatch.setattr(zw.zemosaic_utils, "save_fits_image", lambda image_data, output_path, **kwargs: Path(output_path).write_bytes(b"PARITY"))

    plan = pu.ParallelPlan(
        cpu_workers=2,
        use_memmap=True,
        max_chunk_bytes=256 * 1024 * 1024,
        rows_per_chunk=128,
        use_gpu=False,
        gpu_rows_per_chunk=64,
        gpu_max_chunk_bytes=128 * 1024 * 1024,
    )

    def _run_once(tile_id: int):
        return zw.create_master_tile(
            seestar_stack_group_info=group,
            tile_id=tile_id,
            output_temp_dir=str(tmp_path),
            stack_norm_method="none",
            stack_weight_method="none",
            stack_reject_algo="winsorized_sigma_clip",
            stack_kappa_low=3.0,
            stack_kappa_high=3.0,
            parsed_winsor_limits=(0.05, 0.05),
            stack_final_combine="mean",
            poststack_equalize_rgb=False,
            apply_radial_weight=False,
            radial_feather_fraction=0.0,
            radial_shape_power=1.0,
            min_radial_weight_floor=0.0,
            quality_crop_enabled=False,
            quality_crop_band_px=16,
            quality_crop_k_sigma=2.0,
            quality_crop_margin_px=8,
            quality_crop_min_run=4,
            altaz_cleanup_enabled=False,
            altaz_margin_percent=0.0,
            altaz_decay=0.0,
            altaz_nanize=False,
            quality_gate_enabled=False,
            quality_gate_threshold=0.0,
            quality_gate_edge_band_px=0,
            quality_gate_k_sigma=0.0,
            quality_gate_erode_px=0,
            quality_gate_move_rejects=False,
            astap_exe_path_global="",
            astap_data_dir_global="",
            astap_search_radius_global=0.0,
            astap_downsample_global=0,
            astap_sensitivity_global=0,
            astap_timeout_seconds_global=1,
            winsor_pool_workers=1,
            winsor_max_frames_per_pass=0,
            progress_callback=None,
            resource_strategy=None,
            center_out_context=None,
            center_out_settings=None,
            center_out_rank=None,
            parallel_plan=plan,
            allow_batch_duplication=False,
            target_stack_size=5,
            min_safe_stack_size=3,
            dbg_tile_ids=None,
        )

    # Baseline (no pressure)
    vm_state["percent"] = 30.0
    vm_state["available"] = 1024 * 1024 * 1024
    (baseline_path, _baseline_wcs), baseline_retries = _run_once(tile_id=11)

    # Adaptive pressure path
    vm_state["percent"] = 95.0
    vm_state["available"] = 64 * 1024 * 1024
    (pressure_path, _pressure_wcs), pressure_retries = _run_once(tile_id=12)

    baseline_outputs = 1 if baseline_path else 0
    pressure_outputs = 1 if pressure_path else 0

    assert baseline_retries == pressure_retries == []
    assert baseline_outputs == pressure_outputs == 1



def test_phase3_budget_change_guard_allows_change_when_limits_clear():
    zw = pytest.importorskip("zemosaic_worker")
    times, reason = zw._phase3_budget_change_block_reason(
        now_loop=100.0,
        last_change_t=0.0,
        change_times=[],
        adapt_cooldown_s=10.0,
        max_level_changes_per_min=6,
    )
    assert reason is None
    assert times == []


def test_phase3_budget_change_guard_blocks_on_cooldown():
    zw = pytest.importorskip("zemosaic_worker")
    times, reason = zw._phase3_budget_change_block_reason(
        now_loop=105.0,
        last_change_t=100.0,
        change_times=[90.0],
        adapt_cooldown_s=10.0,
        max_level_changes_per_min=6,
    )
    assert reason == "cooldown"
    assert times == [90.0]


def test_phase3_budget_change_guard_blocks_on_rate_limit_and_prunes_old_entries():
    zw = pytest.importorskip("zemosaic_worker")
    # Two entries are stale (>60s) and must be pruned before rate-limit check.
    raw_times = [20.0, 25.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]
    times, reason = zw._phase3_budget_change_block_reason(
        now_loop=90.0,
        last_change_t=0.0,
        change_times=raw_times,
        adapt_cooldown_s=0.0,
        max_level_changes_per_min=6,
    )
    assert times == [55.0, 60.0, 65.0, 70.0, 75.0, 80.0]
    assert reason == "rate_limit"



def test_affine_photometric_summary_flags_degenerate_solution():
    zw = pytest.importorskip("zemosaic_worker")
    summary = zw._log_affine_photometric_summary(
        {"tile:0001": (0.0, 996.0), "tile:0002": (0.0, 996.0)},
        logger_obj=zw.logger,
        anchor_shift_applied=False,
        global_anchor_shift=None,
    )
    assert summary["degenerate"] is True
    assert summary["unique_count"] == 1


def test_affine_photometric_summary_flags_non_degenerate_solution():
    zw = pytest.importorskip("zemosaic_worker")
    summary = zw._log_affine_photometric_summary(
        {"tile:0001": (0.95, 10.0), "tile:0002": (1.03, -7.0)},
        logger_obj=zw.logger,
        anchor_shift_applied=False,
        global_anchor_shift=None,
    )
    assert summary["degenerate"] is False
    assert summary["unique_count"] >= 2


def test_source_contract_skips_non_user_degenerate_affine_application():
    src = _worker_source()
    assert "apply_photometric: skipping degenerate intertile solution" in src
    assert 'if (not affine_from_user) and bool(affine_summary.get("degenerate"))' in src



def test_source_contract_phase6_writes_display_fits_companion():
    src = _worker_source()
    assert 'display_fits_path = output_folder_path / f"{output_base_name}_display.fits"' in src
    assert 'run_info_phase6_display_fits_saved' in src
    assert 'ZMDISPF' in src



def test_eta_seconds_from_progress_monotonic_behavior():
    zw = pytest.importorskip("zemosaic_worker")
    now = zw.time.monotonic()
    start = now - 100.0
    eta_10 = zw._eta_seconds_from_progress(start, 10.0)
    eta_50 = zw._eta_seconds_from_progress(start, 50.0)
    eta_90 = zw._eta_seconds_from_progress(start, 90.0)
    assert eta_10 > eta_50 > eta_90 >= 0.0


def test_eta_smoothing_phase_shift_allows_reset():
    zw = pytest.importorskip("zemosaic_worker")
    prev = 120.0
    huge = 20000.0
    smoothed = zw._eta_smooth_seconds(prev, huge, alpha=0.22)
    # Large regime shifts (phase transitions) should reset ETA immediately.
    assert smoothed == huge


def test_eta_smoothing_regular_changes_stay_smoothed():
    zw = pytest.importorskip("zemosaic_worker")
    prev = 120.0
    new_eta = 180.0
    smoothed = zw._eta_smooth_seconds(prev, new_eta, alpha=0.22)
    assert prev < smoothed < new_eta


def test_source_contract_eta_uses_max_not_double_count_sum():
    src = _worker_source()
    assert 'total_eta_sec = max(eta_pre_sec, eta_global_sec, eta_phase_model_sec)' in src
    assert 'total_eta_sec = max(eta_ch_sec, eta_global_sec, eta_phase_model_sec)' in src



def test_eta_phase_model_conservative_vs_progress_in_phase3_midrun():
    zw = pytest.importorskip("zemosaic_worker")
    start = zw.time.monotonic() - 180.0
    # Typical mid-phase3 progress where global-progress-only ETA can underpredict.
    eta_prog = zw._eta_seconds_from_progress(start, 48.0)
    eta_model = zw._eta_seconds_from_phase_model(start, phase_index=3, phase_fraction=0.35)
    assert eta_model > eta_prog


def test_source_contract_eta_uses_phase_model():
    src = _worker_source()
    assert '_eta_seconds_from_phase_model(' in src
    assert 'phase_index=3' in src
    assert 'phase_index=5' in src



def test_source_contract_intertile_pairs_emit_eta_updates():
    src = _worker_source()
    assert 'if message_or_stage == "phase5_intertile_pairs":' in src
    assert 'def _emit_intertile_eta(done_pairs: int, total_pairs: int) -> None:' in src
    assert 'f"ETA_UPDATE:{h:02d}:{m:02d}:{s:02d}"' in src
