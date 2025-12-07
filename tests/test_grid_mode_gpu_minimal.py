import os
import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import grid_mode


@pytest.mark.skipif(
    not getattr(grid_mode, "_CUPY_AVAILABLE", False),
    reason="CuPy not available",
)
@pytest.mark.skipif(
    not getattr(grid_mode, "_ASTROPY_AVAILABLE", False)
    or not getattr(grid_mode, "_REPROJECT_AVAILABLE", False),
    reason="Astropy/Reproject not available",
)
def test_grid_mode_gpu_minimal(caplog, monkeypatch):
    """Minimal test for Grid mode GPU path with synthetic data."""
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create synthetic FITS files
        def create_synthetic_fits(path: Path, ra_deg: float, dec_deg: float, data_shape: tuple[int, int]):
            # Create simple WCS
            w = WCS(naxis=2)
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            w.wcs.crval = [ra_deg, dec_deg]
            w.wcs.crpix = [data_shape[1] // 2, data_shape[0] // 2]
            w.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]  # 1 arcsec/pixel

            # Create synthetic data (random noise)
            data = np.random.normal(1000, 100, data_shape).astype(np.float32)

            # Write FITS
            header = w.to_header()
            fits.writeto(path, data, header, overwrite=True)

        # Create 3 synthetic frames
        frames = []
        for i, (ra, dec) in enumerate([(0.0, 0.0), (0.01, 0.0), (0.02, 0.0)]):
            fits_path = tmp_path / f"frame_{i}.fits"
            create_synthetic_fits(fits_path, ra, dec, (100, 100))
            frames.append(fits_path)

        # Create stack_plan.csv
        csv_path = tmp_path / "stack_plan.csv"
        with open(csv_path, 'w') as f:
            f.write("file_path,exposure\n")
            for path in frames:
                f.write(f"{path.name},1.0\n")

        # Run Grid mode with GPU enabled
        input_folder = str(tmp_path)
        output_folder = str(tmp_path / "output")
        os.makedirs(output_folder, exist_ok=True)

        # Monkeypatch config to set grid_size_factor and overlap
        monkeypatch.setattr("grid_mode._load_config_from_disk", lambda: {
            "grid_size_factor": 10.0,
            "batch_overlap_pct": 0.0,
        })

        caplog.clear()
        with caplog.at_level("DEBUG"):
            grid_mode.run_grid_mode(
                input_folder=input_folder,
                output_folder=output_folder,
                grid_use_gpu=True,
            )

        # Check that GPU logs are present
        logs = caplog.text
        assert "GPU stacking started" in logs, "GPU stacking start log not found"
        assert ("GPU stacking completed successfully" in logs or
                "GPU stack failed, falling back to CPU" in logs), "GPU completion or fallback log not found"

        # Check that output mosaic exists
        mosaic_path = Path(output_folder) / "mosaic_grid.fits"
        assert mosaic_path.exists(), "Output mosaic not created"

        # Optionally, run with CPU and compare shape/dtype, but for minimal test, just check existence
        # Since GPU may fallback, we mainly check logs indicate attempt
