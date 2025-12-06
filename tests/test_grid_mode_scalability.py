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
    not getattr(grid_mode, "_ASTROPY_AVAILABLE", False)
    or not getattr(grid_mode, "_REPROJECT_AVAILABLE", False),
    reason="Astropy/Reproject not available",
)
def test_grid_mode_scalability_many_frames(caplog, monkeypatch):
    """Test Grid mode scalability with many frames per tile."""
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create synthetic FITS files (many frames to stress chunking)
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

        # Create 100 synthetic frames in a small area to force one tile
        num_frames = 100
        frames = []
        for i in range(num_frames):
            ra = 0.0 + (i % 10) * 0.001  # Small spread
            dec = 0.0 + (i // 10) * 0.001
            fits_path = tmp_path / f"frame_{i:03d}.fits"
            create_synthetic_fits(fits_path, ra, dec, (100, 100))
            frames.append(fits_path)

        # Create stack_plan.csv
        csv_path = tmp_path / "stack_plan.csv"
        with open(csv_path, 'w') as f:
            f.write("file_path,exposure\n")
            for path in frames:
                f.write(f"{path.name},1.0\n")

        # Run Grid mode with large grid_size_factor to force one tile, small chunk budget to trigger chunking
        input_folder = str(tmp_path)
        output_folder = str(tmp_path / "output")
        os.makedirs(output_folder, exist_ok=True)

        # Monkeypatch config to set grid_size_factor, overlap, and chunk budget
        monkeypatch.setattr("grid_mode._load_config_from_disk", lambda: {
            "grid_size_factor": 100.0,
            "batch_overlap_pct": 0.0,
            "grid_chunk_ram_mb": 1.0,
        })

        caplog.clear()
        with caplog.at_level("INFO"):
            grid_mode.run_grid_mode(
                input_folder=input_folder,
                output_folder=output_folder,
            )

        # Check that it completes without error
        mosaic_path = Path(output_folder) / "mosaic_grid.fits"
        assert mosaic_path.exists(), "Output mosaic not created"

        # Check logs for chunking and memory telemetry
        logs = caplog.text
        assert "chunked stacking enabled" in logs, "Chunking not triggered"
        assert "Memory telemetry" in logs, "Memory telemetry not logged"

        # Check that no errors occurred
        assert "failed" not in logs.lower() or "completed" in logs.lower(), "Processing failed"
