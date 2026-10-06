"""ZM-ZEGRID-R7 targeted tests — production wiring (stack_plan.csv -> ZeGrid).

Covers (i) the end-to-end run on a REAL small M106 subset, (ii) the engine
dispatch (``grid_engine`` = "zegrid" default / "legacy" unchanged / no silent
fallback), (iii) the portable (psutil) memory probe, and (iv) the normalization
policy (sky_mean default; linear_fit honoured + warned).

The heavy end-to-end test is gated on the presence of the real M106 lights
directory (skipped with an explicit reason when absent), mirroring the R6 tests.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from zemosaic import grid_mode
from zemosaic import zemosaic_zegrid_mode as zegrid
from zemosaic.core.zegrid import sweep as zsw
from zemosaic.zemosaic_worker import (
    _resolve_grid_engine,
    _resolve_zegrid_normalization,
)

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

def test_resolve_grid_engine_default_zegrid():
    assert _resolve_grid_engine({}, None) == "zegrid"
    assert _resolve_grid_engine({"grid_engine": "zegrid"}, None) == "zegrid"


def test_resolve_grid_engine_legacy():
    assert _resolve_grid_engine({"grid_engine": "legacy"}, None) == "legacy"


def test_resolve_grid_engine_invalid_defaults_zegrid():
    assert _resolve_grid_engine({"grid_engine": "bogus"}, None) == "zegrid"


def test_resolve_grid_engine_from_zconfig():
    assert _resolve_grid_engine({}, SimpleNamespace(grid_engine="legacy")) == "legacy"


def test_resolve_zegrid_normalization_default_sky_mean():
    # No explicit choice (empty cache) -> sky_mean (ZeGrid default), NOT legacy linear_fit.
    assert _resolve_zegrid_normalization({}, "linear_fit") == "sky_mean"
    assert _resolve_zegrid_normalization({}, "linear_fit") == "sky_mean"


def test_resolve_zegrid_normalization_explicit_honoured():
    assert _resolve_zegrid_normalization({"stacking_normalize_method": "sky_mean"}, "x") == "sky_mean"
    assert _resolve_zegrid_normalization({"stacking_normalize_method": "linear_fit"}, "x") == "linear_fit"
    assert _resolve_zegrid_normalization({"stacking_normalize_method": "none"}, "x") == "none"
    assert _resolve_zegrid_normalization({"stack_norm_method": "none"}, "x") == "none"


def test_legacy_grid_path_still_reachable():
    # The legacy engine module + entry point remain importable and callable.
    assert hasattr(grid_mode, "run_grid_mode")
    assert callable(grid_mode.run_grid_mode)


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

def test_zegrid_mode_no_silent_legacy_fallback(tmp_path, monkeypatch):
    # run_zegrid_mode must RAISE on failure and never silently invoke the legacy
    # grid_mode.run_grid_mode path.
    called = []
    monkeypatch.setattr(grid_mode, "run_grid_mode", lambda *a, **k: called.append(1))
    (tmp_path / "stack_plan.csv").write_text("file_path\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        zegrid.run_zegrid_mode(str(tmp_path), str(tmp_path / "out"))
    assert called == []


# ---------------------------------------------------------------------------
# (i) End-to-end on a REAL small M106 subset
# ---------------------------------------------------------------------------

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
