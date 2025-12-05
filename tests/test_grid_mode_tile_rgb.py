import numpy as np
import pytest

import grid_mode
import zemosaic_utils


@pytest.mark.skipif(not grid_mode._ASTROPY_AVAILABLE, reason="Astropy unavailable")
def test_grid_tile_written_with_rgb_channels(monkeypatch, tmp_path):
    tile_size = 64
    x_grad = np.tile(np.linspace(0, 1, tile_size, dtype=np.float32), (tile_size, 1))
    y_grad = np.tile(np.linspace(0, 0.6, tile_size, dtype=np.float32)[:, None], (1, tile_size))
    const_b = np.full((tile_size, tile_size), 0.2, dtype=np.float32)
    synthetic_patch = np.stack([x_grad, y_grad, const_b], axis=-1)
    footprint = np.ones((tile_size, tile_size), dtype=np.float32)

    def fake_reproject_frame_to_tile(frame, tile, tile_shape, progress_callback=None):
        return synthetic_patch, footprint

    def fake_frame_weight(*_, **__):
        return 1.0

    monkeypatch.setattr(grid_mode, "_reproject_frame_to_tile", fake_reproject_frame_to_tile)
    monkeypatch.setattr(grid_mode, "_compute_frame_weight", fake_frame_weight)

    class DummyWCS:
        def to_header(self, relax=True):
            return grid_mode.fits.Header() if grid_mode._ASTROPY_AVAILABLE and grid_mode.fits else None

    tile = grid_mode.GridTile(tile_id=1, bbox=(0, tile_size, 0, tile_size), wcs=DummyWCS())
    tile.frames.append(grid_mode.FrameInfo(path=tmp_path / "frame_0001.fits"))

    config = grid_mode.GridModeConfig(stack_final_combine="mean")
    output = grid_mode.process_tile(tile, tmp_path, config)
    assert output is not None and output.exists()

    data, header, info = zemosaic_utils.load_and_validate_fits(output, normalize_to_float32=False)
    assert data.shape == (tile_size, tile_size, 3)

    medians = np.nanmedian(data, axis=(0, 1))
    assert not np.allclose(medians[0], medians[1])
    assert not np.allclose(medians[1], medians[2])

    # Channels should preserve their gradients after round-trip
    assert medians[0] > medians[2]
    assert medians[1] > medians[2]
