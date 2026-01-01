import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import zemosaic_worker as zw


def test_should_use_gpu_helper_respects_plan(monkeypatch):
    """should_use_gpu_for_reproject honours config, plan, and GPU availability."""

    monkeypatch.setattr(zw, "gpu_is_available", lambda: True)
    zw.reset_phase5_gpu_runtime_state()
    config = {"use_gpu_phase5": True}
    plan = type("Plan", (), {"use_gpu": True})()
    assert zw.should_use_gpu_for_reproject("phase5_reproject_coadd", config, plan)

    plan.use_gpu = False
    assert not zw.should_use_gpu_for_reproject("phase5_reproject_coadd", config, plan)

    plan.use_gpu = True
    config["use_gpu_phase5"] = False
    assert not zw.should_use_gpu_for_reproject("phase5_reproject_coadd", config, plan)
    zw.reset_phase5_gpu_runtime_state()


def test_phase5_rows_per_chunk_bumps_when_plugged(caplog):
    plan = SimpleNamespace(
        gpu_rows_per_chunk=69,
        gpu_max_chunk_bytes=128 * 1024 * 1024,
        use_gpu=True,
    )
    ctx = SimpleNamespace(safe_mode=1, on_battery=False, power_plugged=True)

    with caplog.at_level(logging.INFO):
        zw._maybe_bump_phase5_gpu_rows_per_chunk(plan, ctx, (100, 2282), 30, logging.getLogger(__name__))

    assert plan.gpu_rows_per_chunk > 69
    assert plan.gpu_rows_per_chunk <= 256
    assert any("Phase5 GPU: bump rows_per_chunk" in msg for msg in caplog.messages)


def test_phase5_rows_per_chunk_skips_on_battery(caplog):
    plan = SimpleNamespace(
        gpu_rows_per_chunk=69,
        gpu_max_chunk_bytes=128 * 1024 * 1024,
        use_gpu=True,
    )
    ctx = SimpleNamespace(safe_mode=1, on_battery=True, power_plugged=False)

    with caplog.at_level(logging.INFO):
        zw._maybe_bump_phase5_gpu_rows_per_chunk(plan, ctx, (100, 2282), 30, logging.getLogger(__name__))

    assert plan.gpu_rows_per_chunk == 69
    assert not any("Phase5 GPU: bump rows_per_chunk" in msg for msg in caplog.messages)


def test_two_pass_gpu_error_falls_back_to_cpu(monkeypatch):
    """Simulate a GPU failure and ensure CPU fallback completes."""

    monkeypatch.setattr(zw, "REPROJECT_AVAILABLE", True)
    monkeypatch.setattr(zw, "ASTROPY_AVAILABLE", True)
    monkeypatch.setattr(zw, "reproject_and_coadd", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        zw,
        "reproject_interp",
        lambda *args, **kwargs: (np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)),
    )
    zw.reset_phase5_gpu_runtime_state()

    call_log: list[bool] = []

    def fake_reproject_wrapper(*args, **kwargs):
        use_gpu = kwargs.get("use_gpu", False)
        call_log.append(bool(use_gpu))
        if use_gpu:
            raise RuntimeError("gpu boom")
        shape_out = kwargs.get("shape_out") or (2, 2)
        mosaic = np.full(shape_out, 5.0, dtype=np.float32)
        coverage = np.ones(shape_out, dtype=np.float32)
        return mosaic, coverage

    monkeypatch.setattr(zw.zemosaic_utils, "reproject_and_coadd_wrapper", fake_reproject_wrapper)

    tiles = [np.ones((2, 2, 1), dtype=np.float32)]
    tiles_wcs = [object()]
    output = zw.run_second_pass_coverage_renorm(
        tiles,
        tiles_wcs,
        final_wcs_p1=object(),
        coverage_p1=np.ones((2, 2), dtype=np.float32),
        shape_out=(2, 2),
        sigma_px=1,
        gain_clip=(0.9, 1.1),
        use_gpu_two_pass=True,
        logger=None,
    )

    assert output is not None
    assert any(call_log), "GPU path was never attempted"
    assert call_log[-1] is False, "CPU fallback was not triggered"
    disabled, _, _ = zw.phase5_gpu_runtime_state()
    assert disabled, "GPU runtime flag was not set after failure"
    zw.reset_phase5_gpu_runtime_state()


def test_gpu_collects_normalized_tiles_like_cpu(tmp_path, monkeypatch):
    """GPU assembly must cache photometrically normalized tiles (parity with CPU)."""

    pytest.importorskip("astropy")
    from astropy.io import fits
    from astropy.wcs import WCS

    # Ensure core dependencies appear available so assemble_final_mosaic_reproject_coadd proceeds.
    monkeypatch.setattr(zw, "REPROJECT_AVAILABLE", True)
    monkeypatch.setattr(zw, "ASTROPY_AVAILABLE", True)
    monkeypatch.setattr(zw, "reproject_and_coadd", object(), raising=False)
    monkeypatch.setattr(zw, "reproject_interp", object(), raising=False)

    # Simplify the heavy reproject step with a deterministic stub and bypass alpha-union building.
    call_records: list[tuple[bool, list[np.ndarray]]] = []

    def fake_reproject_and_coadd_wrapper(data_list, *_, use_gpu=False, **__):
        mosa = np.mean(np.stack(data_list, axis=0), axis=0).astype(np.float32)
        cov = np.ones_like(mosa, dtype=np.float32)
        call_records.append((bool(use_gpu), [np.array(d, copy=True) for d in data_list]))
        return mosa, cov

    monkeypatch.setattr(zw.zemosaic_utils, "reproject_and_coadd_wrapper", fake_reproject_and_coadd_wrapper)
    monkeypatch.setattr(zw, "_build_alpha_union_map", lambda *args, **kwargs: None)

    def _make_wcs():
        w = WCS(naxis=2)
        w.wcs.crpix = [2.0, 2.0]
        w.wcs.cdelt = np.array([-0.1, 0.1])
        w.wcs.crval = [0.0, 0.0]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.pixel_shape = (4, 4)
        return w

    def _write_tile(value: float, name: str):
        arr_hwc = np.full((4, 4, 3), value, dtype=np.float32)
        fits_data = np.moveaxis(arr_hwc, -1, 0)
        path = tmp_path / f"{name}.fits"
        fits.writeto(path, fits_data, header=_make_wcs().to_header(), overwrite=True)
        return str(path), _make_wcs()

    tile1, wcs1 = _write_tile(1.0, "tile1")
    tile2, wcs2 = _write_tile(10.0, "tile2")
    final_wcs = _make_wcs()

    affine = [(1.0, 0.0), (0.1, 0.0)]  # Tile 2 must be scaled down to match tile 1.
    cpu_cache: list[tuple[np.ndarray, object, np.ndarray | None, float]] = []
    gpu_cache: list[tuple[np.ndarray, object, np.ndarray | None, float]] = []

    cpu_result = zw.assemble_final_mosaic_reproject_coadd(
        master_tile_fits_with_wcs_list=[(tile1, wcs1), (tile2, wcs2)],
        final_output_wcs=final_wcs,
        final_output_shape_hw=(4, 4),
        progress_callback=None,
        n_channels=3,
        match_bg=True,
        use_gpu=False,
        collect_tile_data=cpu_cache,
        tile_affine_corrections=affine,
        intertile_photometric_match=False,
    )

    gpu_result = zw.assemble_final_mosaic_reproject_coadd(
        master_tile_fits_with_wcs_list=[(tile1, wcs1), (tile2, wcs2)],
        final_output_wcs=final_wcs,
        final_output_shape_hw=(4, 4),
        progress_callback=None,
        n_channels=3,
        match_bg=True,
        use_gpu=True,
        collect_tile_data=gpu_cache,
        tile_affine_corrections=affine,
        intertile_photometric_match=False,
    )

    cpu_mosaic, _, _ = cpu_result
    gpu_mosaic, _, _ = gpu_result

    assert np.allclose(cpu_mosaic, gpu_mosaic)
    assert call_records[0][0] is False and call_records[-1][0] is True
    assert len(cpu_cache) == len(gpu_cache) == 2
    # Tile 2 should be normalized to the same scale as tile 1 in both caches.
    for (arr_cpu, _, _, _), (arr_gpu, _, _, _) in zip(cpu_cache, gpu_cache):
        assert np.allclose(arr_cpu, arr_gpu)
    assert np.allclose(cpu_cache[1][0], np.full((4, 4, 3), 1.0, dtype=np.float32))


def test_two_pass_gpu_rms_matches_cpu(monkeypatch):
    """GPU two-pass coverage renorm should match CPU output within 1% RMS."""

    pytest.importorskip("astropy")
    pytest.importorskip("reproject")
    cupy = pytest.importorskip("cupy")
    if not cupy.is_available():
        pytest.skip("CuPy GPU unavailable")

    # Allow GPU path on non-Windows platforms during the test run.
    monkeypatch.setattr(zw, "CUPY_AVAILABLE", True, raising=False)

    from astropy.wcs import WCS

    def _make_wcs():
        w = WCS(naxis=2)
        w.wcs.crpix = [2.0, 2.0]
        w.wcs.cdelt = np.array([-0.5, 0.5])
        w.wcs.crval = [0.0, 0.0]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.pixel_shape = (4, 4)
        return w

    final_wcs = _make_wcs()
    tiles_wcs = [_make_wcs() for _ in range(3)]
    base = np.linspace(90.0, 110.0, 16, dtype=np.float32).reshape(4, 4)
    tiles = [
        np.stack([base, base * 1.05, base * 0.95], axis=-1),
        np.stack([base * 1.01, base * 0.99, base * 1.02], axis=-1),
        np.stack([base * 0.97, base * 1.03, base], axis=-1),
    ]
    coverage_p1 = np.ones((4, 4), dtype=np.float32)
    plan = SimpleNamespace(
        cpu_workers=1,
        rows_per_chunk=4,
        gpu_rows_per_chunk=4,
        max_chunk_bytes=0,
        gpu_max_chunk_bytes=0,
        use_memmap=False,
    )

    zw.reset_phase5_gpu_runtime_state()
    cpu_run = zw.run_second_pass_coverage_renorm(
        tiles,
        tiles_wcs,
        final_wcs_p1=final_wcs,
        coverage_p1=coverage_p1,
        shape_out=(4, 4),
        sigma_px=1,
        gain_clip=(0.85, 1.18),
        use_gpu_two_pass=False,
        parallel_plan=plan,
    )
    zw.reset_phase5_gpu_runtime_state()
    gpu_run = zw.run_second_pass_coverage_renorm(
        tiles,
        tiles_wcs,
        final_wcs_p1=final_wcs,
        coverage_p1=coverage_p1,
        shape_out=(4, 4),
        sigma_px=1,
        gain_clip=(0.85, 1.18),
        use_gpu_two_pass=True,
        parallel_plan=plan,
    )

    assert cpu_run is not None and gpu_run is not None
    cpu_mosaic, cpu_cov = cpu_run
    gpu_mosaic, gpu_cov = gpu_run
    assert cpu_mosaic.shape == gpu_mosaic.shape == (4, 4, 3)
    assert cpu_cov.shape == gpu_cov.shape == (4, 4)

    def _rms(arr):
        return float(np.sqrt(np.nanmean(np.square(arr.astype(np.float64)))))

    diff = cpu_mosaic.astype(np.float64) - gpu_mosaic.astype(np.float64)
    rms = _rms(diff)
    span = float(np.nanmax(cpu_mosaic) - np.nanmin(cpu_mosaic))
    assert rms < 0.01 * max(span, 1.0)

    cov_rms = _rms(cpu_cov - gpu_cov)
    cov_span = float(np.nanmax(cpu_cov) - np.nanmin(cpu_cov))
    assert cov_rms < 0.01 * max(cov_span, 1.0)
