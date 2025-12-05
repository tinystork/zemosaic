import logging
from pathlib import Path

import numpy as np

import grid_mode


class DummyWCS:
    def to_header(self):
        return None


def _make_tile(tmp_path: Path, tile_id: int, data: np.ndarray, bbox: tuple[int, int, int, int]):
    path = tmp_path / f"tile_{tile_id}.fits"
    grid_mode.fits.writeto(path, data.astype(np.float32), overwrite=True)
    return grid_mode.GridTile(tile_id=tile_id, bbox=bbox, wcs=None, output_path=path)


def _base_grid(tmp_path: Path) -> grid_mode.GridDefinition:
    return grid_mode.GridDefinition(
        global_wcs=DummyWCS(),
        global_shape_hw=(4, 4),
        tile_size_px=2,
        overlap_fraction=0.0,
        tiles=[],
    )


def test_assemble_tiles_skips_invalid_and_succeeds(tmp_path, caplog):
    grid = _base_grid(tmp_path)
    valid_tile = _make_tile(tmp_path, 1, np.ones((2, 2, 1), dtype=np.float32), (0, 2, 0, 2))
    invalid_tile = _make_tile(tmp_path, 2, np.zeros((2, 2, 1), dtype=np.float32), (2, 4, 0, 2))
    grid.tiles = [valid_tile, invalid_tile]

    with caplog.at_level(logging.INFO, logger="ZeMosaicWorker"):
        result = grid_mode.assemble_tiles(
            grid,
            grid.tiles,
            tmp_path / "mosaic.fits",
            grid_rgb_equalize=False,
        )

    assert result is not None and result.exists()
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Assembly: tile 2 has empty valid-mask, skipping" in messages
    assert "Assembly summary: attempted=2, io_fail=0, channel_mismatch=0, empty_mask=1, kept=1" in messages


def test_assemble_tiles_all_invalid_returns_none(tmp_path, caplog):
    grid = _base_grid(tmp_path)
    tile_a = _make_tile(tmp_path, 10, np.zeros((2, 2, 1), dtype=np.float32), (0, 2, 0, 2))
    tile_b = _make_tile(tmp_path, 11, np.zeros((2, 2, 1), dtype=np.float32), (2, 4, 0, 2))
    grid.tiles = [tile_a, tile_b]

    with caplog.at_level(logging.INFO, logger="ZeMosaicWorker"):
        result = grid_mode.assemble_tiles(
            grid,
            grid.tiles,
            tmp_path / "mosaic_invalid.fits",
            grid_rgb_equalize=False,
        )

    assert result is None
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "kept=0" in messages
    assert "Assembly summary: attempted=2, io_fail=0, channel_mismatch=0, empty_mask=2, kept=0" in messages
