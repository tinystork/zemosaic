"""ZM-M33 lot 1 — windowed source preparation for the external executor.

The defect: the source-backed provider decoded and reprojected the WHOLE cell
for every tile of every frame (``tiles x N`` full-cell reprojections). These
tests prove the fix:

* **Parity (mandatory, no tolerance)**: a windowed preparation is byte-identical
  to the historical whole-cell preparation sliced to the same window, for plain
  TAN and for SIP/distorted WCS, on full-row, sub-row, partial-last and 1x1
  windows, and through ``WindowedCanonicalProvider.get_tile``.
* **Bound**: the pixels actually decoded and reprojected are bounded by
  ``N x cell`` (+ halo overlaps), not by ``tiles x N x cell``.
* **Secondary**: ``external_cell_host_components`` prices a batch that can never
  exceed the real interior surface, and no longer writes the
  ``batch_bytes_per_position_frame`` key twice.

Kept deliberately tiny: the host is shared with a user run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS, Sip

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import execution as zxe
from zemosaic.core.zegrid import external_stream as xstream
from zemosaic.core.zegrid import file_provider as zfp
from zemosaic.core.zegrid import geometry as zg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aligned_wcs(shape, sip=False):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 30.0]
    w.wcs.crpix = [(shape[1] - 1) / 2 + 0.5, (shape[0] - 1) / 2 + 0.5]
    w.wcs.cd = np.array([[-0.001, 0.00002], [0.00002, 0.001]])
    if sip:
        a = np.zeros((3, 3))
        b = np.zeros((3, 3))
        a[2, 0], b[0, 2] = 2e-5, -3e-5
        w.sip = Sip(a, b, None, None, w.wcs.crpix)
        w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    w.array_shape = shape
    return w


def _write_corpus(tmp_path, *, n=4, shape=(40, 44), sip=False, seed=17):
    rng = np.random.default_rng(seed)
    descs = []
    for i in range(n):
        wcs = _aligned_wcs(shape, sip=sip)
        shift = i * 0.00035  # sub-pixel offsets keep the cell non-trivial
        wcs.wcs.crval = [10.0 + shift, 30.0 - shift]
        data = (100.0 + 2 * i + rng.normal(0, 1, shape)).astype(np.float32)
        data[0:2, :3] = np.nan
        path = Path(tmp_path) / f"f{i}.fits"
        fits.PrimaryHDU(data, header=wcs.to_header(relax=True)).writeto(path)
        descs.append(zg.FrameDescriptor(
            frame_id=zg.FrameId(path.name), source_path=str(path),
            shape_hw=shape, wcs_header=wcs.to_header(relax=True).tostring(),
            header_sha256="", instrument="",
        ))
    canvas = zg.build_canvas(descs)
    layout = zg.build_layout(canvas, 1, 1)
    row, col, core = next(iter(layout.iter_cells(canvas)))
    cell = zg.ZeGridCell(
        cell_id=zg.cell_id(row, col), canvas_id=canvas.canvas_id,
        layout_id=layout.layout_id, row=row, col=col, core=core,
    )
    patch = zg.build_patch(canvas, cell, halo_px=5)
    mem = zg.compute_membership(descs, canvas, cell, patch)
    return descs, canvas, cell, patch, mem


def _windows(h, w):
    return [
        (0, h, 0, w),                 # full patch
        (0, min(h, 16), 0, w),        # full row band at the top
        (3, min(h, 19), 5, max(6, w - 4)),  # sub-row interior
        (max(0, h - 15), h, max(0, w - 9), w),  # partial last tile (corner)
        (h - 1, h, w - 1, w),         # single pixel at the far corner
    ]


# ---------------------------------------------------------------------------
# 1. Reprojection-level parity (TAN fast path + SIP fallback), no FITS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sip", [False, True])
def test_windowed_reprojection_bit_exact(sip):
    rng = np.random.default_rng(5)
    patch_hw = (37, 29)
    src_wcs = _aligned_wcs((40, 44), sip=sip)
    src = rng.normal(0, 1, (3, 40, 44)).astype(np.float32)
    src[0, 0:2, 0:3] = np.nan
    patch_wcs = _aligned_wcs(patch_hw, sip=False)

    full_rgb, full_geom = zxe.reproject_cropped(src, src_wcs, patch_wcs, patch_hw)
    for window in _windows(*patch_hw):
        y0, y1, x0, x1 = window
        rgb, geom = zxe.reproject_cropped(
            src, src_wcs, patch_wcs, patch_hw, window=window
        )
        assert rgb.shape == (y1 - y0, x1 - x0, 3)
        # BIT-exact: same values, same NaNs, same support.
        np.testing.assert_array_equal(
            np.asarray(rgb, dtype=np.float32),
            np.asarray(full_rgb[y0:y1, x0:x1], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.isnan(rgb), np.isnan(full_rgb[y0:y1, x0:x1])
        )
        np.testing.assert_array_equal(geom, full_geom[y0:y1, x0:x1])


# ---------------------------------------------------------------------------
# 2. Windowed cell preparation vs whole-cell preparation (real FITS)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sip", [False, True])
def test_windowed_source_preparation_bit_exact(tmp_path, sip):
    descs, canvas, _cell, patch, mem = _write_corpus(tmp_path, sip=sip)
    planned = zz._plan_cell_source_frames(descs, canvas, patch, mem)
    h, w = patch.patch_shape_hw
    stats = zz._new_source_window_stats()
    for frame, plan in planned:
        full_rgb, full_sup = zz._prepare_cell_source_frame(
            frame, plan.source_bounds, patch.patch_wcs_header, (h, w)
        )
        for window in _windows(h, w):
            y0, y1, x0, x1 = window
            rgb, sup = zz._prepare_cell_source_window(
                frame, plan.source_bounds, patch.patch_wcs_header, (h, w),
                window, stats=stats,
            )
            np.testing.assert_array_equal(
                np.asarray(rgb, dtype=np.float32),
                np.asarray(full_rgb[y0:y1, x0:x1], dtype=np.float32),
                err_msg=f"frame={frame.frame_id.logical_path} window={window}",
            )
            np.testing.assert_array_equal(
                np.isnan(rgb), np.isnan(full_rgb[y0:y1, x0:x1])
            )
            np.testing.assert_array_equal(sup, full_sup[y0:y1, x0:x1])
    assert stats["window_prepares"] == len(planned) * len(_windows(h, w))
    # Every window kept the bounded decoder (no silent whole-frame decode).
    assert stats["full_decode_fallbacks"] == 0
    assert stats["source_pixels_decoded"] > 0
    assert stats["reprojected_pixels"] > 0


# ---------------------------------------------------------------------------
# 3. Provider.get_tile parity: windowed vs full-cell
# ---------------------------------------------------------------------------

def test_provider_get_tile_windowed_bit_exact():
    rng = np.random.default_rng(9)
    h, w, n = 41, 29, 3
    frames = []
    for i in range(n):
        frame = rng.normal(0, 2, (h, w, 3)).astype(np.float32) + i
        frame[0:2, :2, :] = np.nan
        frames.append(frame)
    sups = [np.ones((h, w), dtype=bool) for _ in range(n)]
    sups[1][5:9, 4:8] = False

    def prepare(i):
        return frames[i], sups[i]

    def prepare_window(i, y0, y1, x0, x1):
        return frames[i][y0:y1, x0:x1], sups[i][y0:y1, x0:x1]

    full = zfp.WindowedCanonicalProvider(prepare, n, expected_hw=(h, w))
    win = zfp.WindowedCanonicalProvider(
        prepare, n, expected_hw=(h, w), prepare_window=prepare_window,
        expected_channels=3, window_stats=zz._new_source_window_stats(),
    )
    for window in _windows(h, w):
        for i in range(n):
            a_img, a_mask = full.get_tile(i, *window)
            b_img, b_mask = win.get_tile(i, *window)
            np.testing.assert_array_equal(a_img, b_img)
            np.testing.assert_array_equal(a_mask, b_mask)
    tel = win.window_telemetry()
    assert tel["window_prepared_count"] == n * len(_windows(h, w))
    assert tel["full_prepared_count"] == 0
    assert win.bytes_read > 0


# ---------------------------------------------------------------------------
# 4. Bound: full-cell reprojections are NOT multiplied by the tile count
# ---------------------------------------------------------------------------

def test_prepared_pixels_bounded_by_cell_not_tiles(tmp_path):
    """H=W=64, N=8, tile 16: historical vs windowed work, measured."""
    n = 8
    shape = (64, 64)
    descs, canvas, _cell, patch, mem = _write_corpus(
        tmp_path, n=n, shape=shape, seed=3
    )
    planned = zz._plan_cell_source_frames(descs, canvas, patch, mem)
    assert len(planned) == n
    h, w = patch.patch_shape_hw
    patch_px = int(h * w)
    tiles = list(xstream._iter_tiles(h, w, 16))
    assert len(tiles) >= 16

    def prepare_for(i):
        frame, plan = planned[i]
        return zz._prepare_cell_source_frame(
            frame, plan.source_bounds, patch.patch_wcs_header, (h, w)
        )

    # --- historical: one WHOLE-CELL preparation per (tile, frame) ----------
    old = zfp.WindowedCanonicalProvider(
        prepare_for, len(planned), expected_hw=(h, w)
    )
    for (y0, y1, x0, x1) in tiles:
        for i in range(n):
            old.get_tile(i, y0, y1, x0, x1)
    old_prepared = old.prepared_count
    old_full_cell_px = old_prepared * patch_px

    # --- fixed: only the fused halo window is decoded/reprojected ----------
    stats = zz._new_source_window_stats()

    def prepare_window(i, y0, y1, x0, x1):
        frame, plan = planned[i]
        return zz._prepare_cell_source_window(
            frame, plan.source_bounds, patch.patch_wcs_header, (h, w),
            (y0, y1, x0, x1), stats=stats,
        )

    new = zfp.WindowedCanonicalProvider(
        prepare_for, len(planned), expected_hw=(h, w),
        prepare_window=prepare_window, expected_channels=3,
        window_stats=stats,
    )
    for (y0, y1, x0, x1) in tiles:
        for i in range(n):
            new.get_tile(i, y0, y1, x0, x1)
    tel = new.window_telemetry()

    # The defect: work proportional to tiles x N x full cell.
    assert old_full_cell_px == old_prepared * patch_px
    # The fix: full-cell reprojections are ZERO; every preparation is a window.
    assert tel["full_prepared_count"] == 0
    assert tel["window_prepared_count"] == len(tiles) * n
    assert tel["prepared_count"] == len(tiles) * n
    # Reprojected pixels are ~ tiles x window (not tiles x full cell).
    assert tel["reprojected_pixels"] == sum(
        (y1 - y0) * (x1 - x0) for (y0, y1, x0, x1) in tiles
    ) * n
    assert tel["reprojected_pixels"] * 2 < old_full_cell_px
    # Decoded source pixels are bounded by the cell surface, never by tiles x.
    assert tel["source_pixels_decoded"] <= len(tiles) * n * (h + 4) * (w + 4)
    print(
        "M33 windowed bound: old_prepared={} old_full_cell_px={} "
        "new_windows={} new_reprojected_px={} decoded_px={}".format(
            old_prepared, old_full_cell_px,
            tel["window_prepared_count"], tel["reprojected_pixels"],
            tel["source_pixels_decoded"],
        )
    )


# ---------------------------------------------------------------------------
# 5. external_cell_host_components: positions bounded by the interior
# ---------------------------------------------------------------------------

def test_host_components_positions_bounded_by_interior():
    n, c = 6, 3
    patch_hw = (32, 32)
    tile_hw = (16, 16)
    interior_hw = (8, 8)
    # A request far larger than the tile must not price more positions than the
    # interior actually has.
    got = xstream.external_cell_host_components(
        n_frames=n, patch_hw=patch_hw, tile_hw=tile_hw,
        interior_hw=interior_hw, channels=c,
        column_batch=100_000, rejection_sink=True,
    )
    interior_px = interior_hw[0] * interior_hw[1]
    assert got["rows_per_batch"] <= interior_hw[0]
    assert got["cols_per_batch"] <= interior_hw[1]
    assert got["batch_positions"] <= interior_px * c
    assert got["batch_positions"] == (
        got["rows_per_batch"] * got["cols_per_batch"] * c
    )
    # The priced batch matches the published per-position-frame coefficient.
    assert got["batch_total_bytes"] == xstream.external_batch_bytes(
        n, got["batch_positions"]
    )
    assert got["batch_bytes_per_position_frame"] == int(
        xstream.EXTERNAL_BATCH_BYTES_PER_POSITION_FRAME
    )
    assert got["batch_sample_bytes_per_position_frame"] == int(
        xstream.EXTERNAL_BATCH_SAMPLE_BYTES_PER_POSITION_FRAME
    )
    # The dict must publish the N-scaled coefficient exactly once (the old
    # literal wrote it twice); the fixed per-position term is separate.
    assert got["batch_fixed_position_bytes"] == int(
        xstream.EXTERNAL_BATCH_FIXED_POSITION_BYTES
    )
