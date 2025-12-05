import os
import sys
from pathlib import Path

import pytest
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
def test_fallback_grid_uses_offset_and_positive_bboxes(monkeypatch):
    pytest.importorskip("reproject")

    def make_wcs(ra_deg: float) -> WCS:
        scale = 1.0 / 3600.0  # 1 arcsec/pixel
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [ra_deg, 0.0]
        w.wcs.crpix = [5.0, 5.0]
        w.wcs.cdelt = [-scale, scale]
        return w

    monkeypatch.setattr(
        grid_mode,
        "find_optimal_celestial_wcs",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("forced failure")),
    )

    frames = [
        grid_mode.FrameInfo(path=Path("f1.fits"), wcs=make_wcs(0.0), shape_hw=(20, 20)),
        grid_mode.FrameInfo(
            path=Path("f2.fits"), wcs=make_wcs(5.0 / 3600.0), shape_hw=(20, 20)
        ),
    ]

    grid = grid_mode.build_global_grid(frames, grid_size_factor=1.0, overlap_fraction=0.0)
    assert grid is not None
    assert grid.offset_xy[0] < 0  # offset should rebase negative footprint into positive space
    assert grid.global_shape_hw[0] > 0 and grid.global_shape_hw[1] > 0

    for frame in frames:
        assert frame.footprint is not None
        fx0, fx1, fy0, fy1 = frame.footprint
        assert fx0 >= 0 and fy0 >= 0
        assert fx1 <= grid.global_shape_hw[1]
        assert fy1 <= grid.global_shape_hw[0]

    assert grid.tiles, "expected at least one tile to be generated"
    for tile in grid.tiles:
        tx0, tx1, ty0, ty1 = tile.bbox
        assert tx0 >= 0 and ty0 >= 0
        assert tx1 <= grid.global_shape_hw[1]
        assert ty1 <= grid.global_shape_hw[0]
