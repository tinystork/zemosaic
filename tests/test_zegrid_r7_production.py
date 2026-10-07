"""ZM-ZEGRID-R7 targeted tests — production wiring (stack_plan.csv -> ZeGrid).

Covers (i) the end-to-end run on a REAL small M106 subset, (ii) the engine
dispatch (``grid_engine`` = "zegrid" default / "legacy" unchanged / no silent
fallback), (iii) the portable (psutil) memory probe, and (iv) the normalization
policy (sky_mean default; linear_fit honoured + warned).

The heavy end-to-end test is gated on the presence of the real M106 lights
directory (skipped with an explicit reason when absent), mirroring the R6 tests.
"""

from __future__ import annotations

import importlib.util
import inspect
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from zemosaic import zemosaic_stack_plan
from zemosaic import zemosaic_zegrid_mode as zegrid
from zemosaic import zemosaic_worker as zw
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import sweep as zsw
from zemosaic.zemosaic_worker import (
    _resolve_zegrid_normalization,
)
from zemosaic.zemosaic_utils import load_image_with_optional_alpha

LIGHTS = Path("/home/tristan/M106/lights")
N_END_TO_END_FRAMES = 6


# ---------------------------------------------------------------------------
# (iii) Portable memory probe (psutil; no POSIX-only call in production path)
# ---------------------------------------------------------------------------

def test_portable_memory_probe_positive():
    assert zegrid.available_memory_bytes() > 0


def test_portable_memory_probe_uses_psutil(monkeypatch):
    class _VM:
        available = 123456789

    monkeypatch.setattr(zegrid.psutil, "virtual_memory", lambda: _VM())
    assert zegrid.available_memory_bytes() == 123456789


def test_sweep_read_available_memory_psutil_first(monkeypatch):
    class _VM:
        available = 424242

    monkeypatch.setattr(zsw, "psutil", SimpleNamespace(virtual_memory=lambda: _VM()))
    assert zsw.read_available_memory() == 424242


def test_sweep_peak_rss_portable_without_resource(monkeypatch):
    # With the POSIX ``resource`` module absent (Windows), peak_rss_kib must not
    # crash and must fall back to psutil's current RSS.
    monkeypatch.setattr(zsw, "resource", None)

    class _MI:
        rss = 7 * 1024 * 1024  # 7 MiB

    monkeypatch.setattr(
        zsw,
        "psutil",
        SimpleNamespace(Process=lambda *a, **k: SimpleNamespace(memory_info=lambda: _MI())),
    )
    assert zsw.peak_rss_kib() == 7 * 1024


# ---------------------------------------------------------------------------
# (ii) Engine dispatch + normalization resolution
# ---------------------------------------------------------------------------

def test_resolve_zegrid_normalization_default_sky_mean():
    # No explicit choice (empty cache) -> sky_mean (ZeGrid default), NOT legacy linear_fit.
    assert _resolve_zegrid_normalization({}, "linear_fit") == "sky_mean"
    assert _resolve_zegrid_normalization({}, "linear_fit") == "sky_mean"


def test_resolve_zegrid_normalization_explicit_honoured():
    assert _resolve_zegrid_normalization({"stacking_normalize_method": "sky_mean"}, "x") == "sky_mean"
    assert _resolve_zegrid_normalization({"stacking_normalize_method": "linear_fit"}, "x") == "linear_fit"
    assert _resolve_zegrid_normalization({"stacking_normalize_method": "none"}, "x") == "none"
    assert _resolve_zegrid_normalization({"stack_norm_method": "none"}, "x") == "none"


def test_legacy_grid_module_removed():
    # The legacy Grid engine module is GONE (archived at
    # origin/archive/zegrid-legacy-grid-5.0.0); it must no longer be importable.
    assert importlib.util.find_spec("zemosaic.grid_mode") is None


# ---------------------------------------------------------------------------
# Normalization policy (sky_mean default; linear_fit honoured + warned)
# ---------------------------------------------------------------------------

def test_normalization_default_sky_mean():
    assert zegrid.resolve_normalization(None) == "sky_mean"
    assert zegrid.resolve_normalization("") == "sky_mean"
    assert zegrid.resolve_normalization("sky_mean") == "sky_mean"
    assert zegrid.resolve_normalization("none") == "none"


def test_normalization_linear_fit_honoured(caplog):
    assert zegrid.resolve_normalization("linear_fit") == "linear_fit"


# ---------------------------------------------------------------------------
# No silent fallback
# ---------------------------------------------------------------------------

def test_zegrid_mode_raises_on_failure_no_fallback(tmp_path):
    # run_zegrid_mode must RAISE on failure; there is no legacy fallback anymore.
    (tmp_path / "stack_plan.csv").write_text("file_path\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        zegrid.run_zegrid_mode(str(tmp_path), str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# ZM-ZEGRID-R8 — direct stack_plan.csv -> ZeGrid dispatch + relocated decoder
# ---------------------------------------------------------------------------

def test_worker_dispatch_direct_to_zegrid_no_engine_setting():
    # The legacy branch and _resolve_grid_engine are removed; the dispatcher
    # routes stack_plan.csv DIRECTLY to run_zegrid_mode (no grid_engine setting).
    assert not hasattr(zw, "_resolve_grid_engine")
    dispatcher_src = inspect.getsource(zw.run_hierarchical_mosaic)
    assert "run_zegrid_mode" in dispatcher_src
    assert "run_grid_mode" not in dispatcher_src
    assert "grid_engine" not in dispatcher_src


def test_worker_detect_grid_mode_delegates_to_relocated_module(tmp_path):
    # Worker's detect_grid_mode delegates to zemosaic_stack_plan.detect_grid_mode.
    assert zw.detect_grid_mode(str(tmp_path)) is False
    (tmp_path / "stack_plan.csv").write_text("file_path\n", encoding="utf-8")
    assert zw.detect_grid_mode(str(tmp_path)) is True
    assert zemosaic_stack_plan.detect_grid_mode(str(tmp_path)) is True


def test_load_image_with_optional_alpha_mono_hwc(tmp_path):
    # Relocated decoder (formerly grid_mode._load_image_with_optional_alpha) must
    # decode a mono (non-Bayer) 2-D FITS into HWC float32 with no alpha weights.
    h, w = 8, 10
    data = np.arange(h * w, dtype=np.float32).reshape(h, w)
    path = tmp_path / "mono.fits"
    fits.writeto(path, data, overwrite=True)

    arr, weights = load_image_with_optional_alpha(path)

    assert arr.shape == (h, w, 1)
    assert arr.dtype == np.float32
    assert weights is None
    np.testing.assert_allclose(arr[..., 0], data, rtol=1e-6)


# ---------------------------------------------------------------------------
# (i) End-to-end on a REAL small M106 subset
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not LIGHTS.is_dir(), reason="M106 lights directory not present")
def test_end_to_end_real_m106_subset(tmp_path):
    frames = sorted(LIGHTS.glob("*.fit"))[:N_END_TO_END_FRAMES]
    assert len(frames) == N_END_TO_END_FRAMES

    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for p in frames:
        shutil.copy2(p, input_dir / p.name)

    lines = ["file_path,exposure"]
    for p in frames:
        lines.append(f"{p.name},10.0")
    (input_dir / "stack_plan.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = tmp_path / "out"
    zegrid.run_zegrid_mode(str(input_dir), str(out))

    sci_path = out / "mosaic_grid.fits"
    cov_path = out / "mosaic_grid_coverage.fits"
    manifest_path = out / "zegrid_manifest.json"
    assert sci_path.exists(), "mosaic_grid.fits not produced"
    assert cov_path.exists(), "coverage FITS not produced"
    assert manifest_path.exists(), "manifest not produced"

    # Cache must live under the output folder (never /tmp).
    assert (out / zegrid.CACHE_DIR_NAME).is_dir()

    with fits.open(sci_path) as hdul:
        sci = hdul[0].data
    with fits.open(cov_path) as hdul:
        cov = hdul[0].data

    # science is (3, H, W) on disk -> HWC in memory.
    assert sci.ndim == 3 and sci.shape[0] == 3
    science_hwc = np.moveaxis(np.asarray(sci, dtype=np.float32), 0, -1)  # (H, W, 3)
    covered = np.asarray(cov) > 0
    assert int(np.count_nonzero(covered)) > 0, "no coverage produced"

    # Finite science over the covered area (holes are NaN, covered is finite).
    finite_covered = np.isfinite(science_hwc[covered])
    assert float(np.mean(finite_covered)) > 0.99, "science not finite over covered area"

    import json

    manifest = json.loads(manifest_path.read_text())
    assert manifest["engine"] == "zegrid"
    assert manifest["normalization"] == "sky_mean"
    assert manifest["n_frames"] == N_END_TO_END_FRAMES
    assert manifest["outputs"]["science"] == "mosaic_grid.fits"
    assert manifest["outputs"]["coverage"] == "mosaic_grid_coverage.fits"
    assert manifest["coverage_pixels"] == int(np.count_nonzero(covered))
    # Ownership invariant: exactly one owner per covered pixel (no overlap).
    assert manifest["complete_cells"], "no complete cells in the assembled canvas"


# ---------------------------------------------------------------------------
# Mode-aware layout (rework-1): coarser, floor-honouring, bounded, deterministic
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not LIGHTS.is_dir(), reason="M106 lights directory not present")
def test_mode_aware_layout_coarser_floor_honouring_bounded_deterministic(m106_corpus):
    frames, _rejected, canvas = m106_corpus

    for gib in (1.0, 2.0, 4.0):
        budget = int(gib * 2**30)
        before = zal.choose_layout(canvas, frames, ram_budget=budget)
        after = zegrid._choose_layout_mode_aware(canvas, frames, ram_budget=budget)

        # Mode-aware is never finer than the in-memory-only policy.
        assert after["cell_count"] <= before.nx * before.ny
        # Scientific floors must hold.
        assert after["floors"]["min_patch_area_px"]["ok"]
        assert after["floors"]["max_halo_overhead"]["ok"]
        # No cell's cheaper-mode bound exceeds the budget.
        for c in after["cells"]:
            cheaper = min(c["inmem_bound_bytes"], c["stream_bound_bytes"])
            assert cheaper <= budget

    # Determinism: identical inputs -> identical layout.
    d1 = zegrid._choose_layout_mode_aware(canvas, frames, ram_budget=int(2.0 * 2**30))
    d2 = zegrid._choose_layout_mode_aware(canvas, frames, ram_budget=int(2.0 * 2**30))
    assert (d1["nx"], d1["ny"]) == (d2["nx"], d2["ny"])


# ---------------------------------------------------------------------------
# rework-2: pinned layout, cache reuse, selected-mode bound (L2)
# ---------------------------------------------------------------------------

def test_parse_pinned_layout():
    assert zegrid._parse_pinned_layout("6x5") == (6, 5)
    assert zegrid._parse_pinned_layout("6X5") == (6, 5)
    assert zegrid._parse_pinned_layout((3, 2)) == (3, 2)
    assert zegrid._parse_pinned_layout(None) is None
    assert zegrid._parse_pinned_layout("") is None
    assert zegrid._parse_pinned_layout("6*5") == (6, 5)
    with pytest.raises(ValueError):
        zegrid._parse_pinned_layout("garbage")
    with pytest.raises(ValueError):
        zegrid._parse_pinned_layout("6")


@pytest.mark.slow
@pytest.mark.skipif(not LIGHTS.is_dir(), reason="M106 lights directory not present")
def test_pinned_layout_honoured_floors_and_infeasible(m106_corpus):
    frames, _rejected, canvas = m106_corpus

    # Honoured: pinned (6,5) at 4 GiB bypasses the RAM-adaptive choice.
    d = zegrid._choose_layout_mode_aware(canvas, frames, ram_budget=int(4 * 2**30), pinned_layout=(6, 5))
    assert d["layout_source"] == "pinned"
    assert (d["nx"], d["ny"]) == (6, 5)

    # Infeasible budget: pinned (3,2) at 1 GiB (cheaper bound > budget).
    with pytest.raises(zal.LayoutInfeasible):
        zegrid._choose_layout_mode_aware(canvas, frames, ram_budget=int(1 * 2**30), pinned_layout=(3, 2))

    # Floor violation: pinned (200,200) violates min_patch_area / max_halo_overhead.
    with pytest.raises(zal.LayoutInfeasible):
        zegrid._choose_layout_mode_aware(canvas, frames, ram_budget=int(4 * 2**30), pinned_layout=(200, 200))


def test_pick_mode_reports_selected_mode_bound():
    n = 6
    area = 500_000
    patch_hw = (500, 1000)

    # Tiny available -> stream; the reported bound must be the STREAMING bound.
    mode, bound = zegrid._pick_mode(n, area, patch_hw, 1)
    assert mode == "stream"
    assert bound == zegrid._estimate_streaming_bytes(n, patch_hw)

    # Huge available -> inmem; the reported bound must be the R4 in-memory bound.
    mode2, bound2 = zegrid._pick_mode(n, area, patch_hw, 10**15)
    assert mode2 == "inmem"
    assert bound2 == int(zal.FITTED_MEMORY_MODEL.predict_bound_bytes(n, area))


@pytest.mark.slow
@pytest.mark.skipif(not LIGHTS.is_dir(), reason="M106 lights directory not present")
def test_cache_reuse_second_build(tmp_path):
    frames = sorted(LIGHTS.glob("*.fit"))[:4]
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for p in frames:
        shutil.copy2(p, input_dir / p.name)

    descs, _ = zg.read_manifest(input_dir)
    canvas = zg.build_canvas(descs)
    nx, ny = 2, 2
    cell_ctxs = []
    for row, col, _b in zg.build_layout(canvas, nx, ny).iter_cells(canvas):
        cell, patch, mem = zsw.build_cell_context(descs, canvas, row, col, nx, ny)
        cell_ctxs.append((row, col, cell, patch, mem))
    cache_root = tmp_path / "cache"

    _, manifests1, report1 = zegrid._build_aligned_cache_frame_major(
        descs, canvas, cell_ctxs, cache_root, None)
    _, manifests2, report2 = zegrid._build_aligned_cache_frame_major(
        descs, canvas, cell_ctxs, cache_root, None)

    assert report1["reused"] == []
    assert report1["rebuilt"] != []
    # Second build: everything reused, nothing rebuilt (no re-decode).
    assert report2["rebuilt"] == []
    assert set(report2["reused"]) == set(report1["rebuilt"])
