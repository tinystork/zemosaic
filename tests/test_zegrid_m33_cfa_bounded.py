"""ZM-M33 lot 2 — bounded CFA decode + adaptive decoded-frame cache.

Two properties are proven here, with no tolerance:

* **CFA sections are bit-exact**: a phase-aligned section decode + local
  debayer (with the loader's exact global-statistics pre-scan) reproduces the
  whole-frame debayer sliced to the same window, byte-for-byte. Covered:
  real Bayer layout, unaligned/odd origins, partial/corner windows, tiny
  windows, and mono non-CFA sources.
* **The decoded-frame cache is honest**: it activates only when ALL N frames
  fit the explicit budget, it never exceeds that budget, and it changes no
  pixel. A budget that cannot fit the whole set leaves it inactive (tile-major
  order would thrash a partial cache), with all counters at zero.

Kept tiny: the host is shared with a user run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import geometry as zg


def _aligned_wcs(shape):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 30.0]
    w.wcs.crpix = [(shape[1] - 1) / 2 + 0.5, (shape[0] - 1) / 2 + 0.5]
    w.wcs.cd = np.array([[-0.001, 0.00002], [0.00002, 0.001]])
    w.array_shape = shape
    return w


def _corpus(tmp_path, *, shape=(64, 72), n=3, bayer="GRBG", seed=31):
    """Real FITS frames: uint16 CFA (BZERO path) or float32 mono."""
    rng = np.random.default_rng(seed)
    descs = []
    for i in range(n):
        wcs = _aligned_wcs(shape)
        wcs.wcs.crval = [10.0 + i * 0.00035, 30.0]
        header = wcs.to_header(relax=True)
        path = Path(tmp_path) / f"c{i}.fit"
        if bayer:
            # uint16 is written by astropy as BITPIX=16 + BZERO=32768, i.e. the
            # exact real-LP integer scaling the loader applies manually.
            raw = rng.integers(100, 4000, shape).astype(np.uint16)
            header["BAYERPAT"] = bayer
            fits.PrimaryHDU(raw, header=header).writeto(path)
            wcs_header = header.tostring()
        else:
            raw = (rng.normal(100, 5, shape)).astype(np.float32)
            raw[0:2, :3] = np.nan
            fits.PrimaryHDU(raw, header=header).writeto(path)
            wcs_header = header.tostring()
        descs.append(zg.FrameDescriptor(
            frame_id=zg.FrameId(path.name), source_path=str(path),
            shape_hw=shape, wcs_header=wcs_header,
            header_sha256="", instrument="",
        ))
    canvas = zg.build_canvas(descs)
    layout = zg.build_layout(canvas, 1, 1)
    row, col, core = next(iter(layout.iter_cells(canvas)))
    cell = zg.ZeGridCell(
        cell_id=zg.cell_id(row, col), canvas_id=canvas.canvas_id,
        layout_id=layout.layout_id, row=row, col=col, core=core,
    )
    patch = zg.build_patch(canvas, cell, halo_px=4)
    mem = zg.compute_membership(descs, canvas, cell, patch)
    planned = zz._plan_cell_source_frames(descs, canvas, patch, mem)
    return descs, patch, planned


def _windows(h, w):
    return [
        (0, h, 0, w),
        (0, min(h, 17), 0, w),
        (3, min(h, 20), 5, max(6, w - 4)),
        (max(0, h - 13), h, max(0, w - 11), w),
        (h - 1, h, w - 1, w),
        (1, 2, 1, 2),
    ]


@pytest.mark.parametrize("bayer", ["GRBG", "RGGB", "GBRG", "BGGR"])
def test_cfa_section_decode_bit_exact(tmp_path, bayer):
    descs, patch, planned = _corpus(tmp_path, bayer=bayer)
    h, w = patch.patch_shape_hw
    stats = zz._new_source_window_stats()
    stats_cache = {}
    for frame, plan in planned:
        full_rgb, full_sup = zz._prepare_cell_source_frame(
            frame, plan.source_bounds, patch.patch_wcs_header, (h, w)
        )
        for window in _windows(h, w):
            y0, y1, x0, x1 = window
            rgb, sup = zz._prepare_cell_source_window(
                frame, plan.source_bounds, patch.patch_wcs_header, (h, w),
                window, stats=stats, stats_cache=stats_cache,
            )
            np.testing.assert_array_equal(
                np.asarray(rgb, dtype=np.float32),
                np.asarray(full_rgb[y0:y1, x0:x1], dtype=np.float32),
                err_msg=f"{bayer} {frame.frame_id.logical_path} {window}",
            )
            np.testing.assert_array_equal(
                np.isnan(rgb), np.isnan(full_rgb[y0:y1, x0:x1])
            )
            np.testing.assert_array_equal(sup, full_sup[y0:y1, x0:x1])
    # Known CFA layouts never fall back to a whole-frame decode.
    assert stats["full_decode_fallbacks"] == 0
    assert stats["bounds_expand_fallbacks"] == 0
    # One exact pre-scan per frame (never per tile), and billed.
    assert stats["prescans"] == len(planned)
    assert stats["source_bytes_decoded"] > 0


def test_cfa_section_direct_decode_matches_full_debayer(tmp_path):
    """Direct check of the uint16/debayer/statistics chain on odd origins."""
    descs, patch, planned = _corpus(tmp_path, bayer="GRBG")
    stats_cache = {}
    for frame, plan in planned:
        ref = zz._decode_frame_hwc(frame, None)
        for bounds in (
            zg.SourceBounds(0, 0, 1, 1),
            zg.SourceBounds(7, 5, 24, 21),
            zg.SourceBounds(0, 0, 32, 32),
            zg.SourceBounds(frame.shape_hw[1] - 9, frame.shape_hw[0] - 7,
                            frame.shape_hw[1], frame.shape_hw[0]),
        ):
            got = zz._decode_frame_section_hwc(
                frame, bounds, stats_cache=stats_cache,
                stats=zz._new_source_window_stats(),
            )
            assert got is not None
            arr, _nbytes, _npx = got
            want = ref[bounds.y0:bounds.y1, bounds.x0:bounds.x1]
            np.testing.assert_array_equal(
                np.asarray(arr, dtype=np.float32),
                np.asarray(want, dtype=np.float32),
            )


def test_mono_non_cfa_section_bit_exact(tmp_path):
    descs, patch, planned = _corpus(tmp_path, bayer=None)
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
                window, stats=stats, stats_cache={},
            )
            np.testing.assert_array_equal(
                np.asarray(rgb, dtype=np.float32),
                np.asarray(full_rgb[y0:y1, x0:x1], dtype=np.float32),
            )
            np.testing.assert_array_equal(sup, full_sup[y0:y1, x0:x1])
    assert stats["full_decode_fallbacks"] == 0


def _provider(tmp_path, budget):
    _descs, patch, planned = _corpus(tmp_path, bayer="GRBG")
    ctx = zz._CellSourceContext(
        planned=tuple((f, p.source_bounds) for f, p in planned),
        patch_wcs_header=patch.patch_wcs_header,
        patch_hw=tuple(patch.patch_shape_hw),
    )
    h, w = patch.patch_shape_hw
    frame_bytes = max(
        int(f.shape_hw[0] * f.shape_hw[1] * 3 * 4) for f, _p in planned
    )
    return ctx, patch, planned, frame_bytes, h, w


def _compare_providers(a, b, tiles, n, h, w):
    for (y0, y1, x0, x1) in tiles:
        for i in range(n):
            ta, ma = a.get_tile(i, y0, y1, x0, x1)
            tb, mb = b.get_tile(i, y0, y1, x0, x1)
            np.testing.assert_array_equal(ta, tb)
            np.testing.assert_array_equal(ma, mb)


def test_adaptive_cache_full_fit_is_active_and_bit_exact(tmp_path):
    ctx, patch, planned, frame_bytes, h, w = _provider(tmp_path, 0)
    n = len(planned)
    tiles = [(0, min(h, 40), 0, min(w, 40)), (5, min(h, 45), 7, min(w, 47))]
    off = ctx.open_provider(budget_bytes=0)
    on = ctx.open_provider(budget_bytes=frame_bytes * n)
    try:
        _compare_providers(off, on, tiles, n, h, w)
        tel = on.window_telemetry()
        assert tel["cache_active"] is True
        assert tel["decodes"] == n                 # one decode per frame
        assert tel["cache_hits"] > 0
        assert tel["cache_misses"] == n
        assert tel["cache_bytes"] <= frame_bytes * n
        assert tel["cache_bytes"] == frame_bytes * n
        assert tel["source_bytes_decoded"] > 0     # the decodes are billed
        # Cache OFF: bounded CFA sections, no whole-frame decode at all.
        tel_off = off.window_telemetry()
        assert tel_off["cache_active"] is False
        assert tel_off["cache_bytes"] == 0
        assert tel_off["decodes"] == 0
        assert tel_off["full_decode_fallbacks"] == 0
        assert tel_off["prescans"] == n
    finally:
        off.close()
        on.close()


def test_adaptive_cache_is_honestly_inactive_when_budget_is_small(tmp_path):
    ctx, patch, planned, frame_bytes, h, w = _provider(tmp_path, 0)
    n = len(planned)
    tile = [(0, min(h, 40), 0, min(w, 40))]
    off = ctx.open_provider(budget_bytes=0)
    # One frame short: the whole set cannot be cached -> cache stays OFF.
    short = ctx.open_provider(budget_bytes=frame_bytes * (n - 1))
    try:
        tel = short.window_telemetry()
        assert tel["cache_active"] is False
        assert tel["cache_bytes"] == 0
        assert tel["cache_hits"] == 0
        assert tel["cache_misses"] == 0
        assert tel["decodes"] == 0
        assert tel["budget_bytes"] == frame_bytes * (n - 1)
        # ... and it still serves exact pixels through the section decoder.
        _compare_providers(off, short, tile, n, h, w)
        assert short.window_telemetry()["full_decode_fallbacks"] == 0
    finally:
        off.close()
        short.close()


def test_source_frame_cache_never_exceeds_budget():
    class _F:  # minimal nbytes-bearing frame stand-in
        def __init__(self, nbytes):
            self.nbytes = nbytes

    stats = {}
    cache = zz._SourceFrameCache(300, 100, 3, stats)
    assert cache.enabled is True
    for i in range(3):
        cache.put(i, _F(100))
    assert cache.bytes == 300
    assert stats["cache_bytes"] == 300
    assert cache.bytes <= cache.budget_bytes
    # 2.9 frames cannot hold 3: never advertised as active.
    small = zz._SourceFrameCache(290, 100, 3, {})
    assert small.enabled is False
    small.put(0, _F(100))
    assert small.bytes == 0
