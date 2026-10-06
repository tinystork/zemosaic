from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from zemosaic import zemosaic_stack_plan


def test_load_stack_plan_resolves_windows_paths_against_input_folder(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    file_a = input_dir / "Light_A.fit"
    file_b = input_dir / "Light_B.fit"
    file_a.write_bytes(b"dummy")
    file_b.write_bytes(b"dummy")

    csv_path = input_dir / "stack_plan.csv"
    csv_path.write_text(
        "file_path,exposure\n"
        "D:\\ASTRO\\project\\Light_A.fit,10\n"
        "D:\\ASTRO\\project\\Light_B.fit,20\n",
        encoding="utf-8",
    )

    frames = zemosaic_stack_plan.load_stack_plan(csv_path)

    assert len(frames) == 2
    assert frames[0].path == file_a
    assert frames[1].path == file_b
    assert all(frame.path.is_file() for frame in frames)


def test_detect_grid_mode_presence(tmp_path: Path) -> None:
    # False without the file, True once stack_plan.csv exists; None-tolerant.
    assert zemosaic_stack_plan.detect_grid_mode(str(tmp_path)) is False
    (tmp_path / "stack_plan.csv").write_text("file_path\n", encoding="utf-8")
    assert zemosaic_stack_plan.detect_grid_mode(str(tmp_path)) is True
    assert zemosaic_stack_plan.detect_grid_mode(None) is False


def test_load_frame_wcs_populates_wcs_and_shape(tmp_path: Path) -> None:
    h, w = 32, 48
    data = np.zeros((h, w), dtype=np.float32)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cdelt = [-0.00028, 0.00028]
    path = tmp_path / "frame.fits"
    fits.writeto(path, data, header=wcs.to_header(), overwrite=True)

    fi = zemosaic_stack_plan.FrameInfo(path=path)
    ok = zemosaic_stack_plan.load_frame_wcs(fi)

    assert ok is True
    assert fi.wcs is not None and getattr(fi.wcs, "is_celestial", False)
    assert fi.shape_hw == (h, w)
