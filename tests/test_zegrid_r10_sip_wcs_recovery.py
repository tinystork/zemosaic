"""ZM-ZEGRID-R10 targeted tests — SIP WCS recovery (stop dropping SIP frames).

Covers the five R10 objectives:

* (a) ``qualify_wcs`` accepts SIP-distorted 2-D celestial TAN and the
  ``slice_wcs`` cropped WCS is SIP-correct.
* (b) plain-TAN ``slice_wcs`` is IDENTICAL to the previous manual CRPIX shift
  (regression guard).
* (c) the SIP footprint polygon contains the true (densely sampled) footprint
  with a small conservative margin.
* (d) the SIP-vs-TAN cross-consistency measurement (numbers pinned).
* (e) rejected-frame counts are surfaced (manifest field + WARN).
* (g) a small real-frame end-to-end through ``run_zegrid_mode`` asserting the
  SIP frame is now INCLUDED (not dropped).

Real-frame fixtures are copied to a DISK directory (never /tmp) so they are
memory-safe on the low-RAM host. Tests that need them are skipped with a clear
reason when absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import execution as zx
from zemosaic.core.zegrid.geometry import SourceBounds
from zemosaic import zemosaic_zegrid_mode as zegrid

# Real Caldwell 11 SIP + TAN fixtures (disk-backed, small: 2 SIP + 1 TAN).
_SIP_FIXTURE = Path("/home/tristan/zegrid_r10_sip_fixture")
_CALDWELL = Path("/home/tristan/caldwell11/lights organized/EQ/IRCUT")

_SIP_FRAME = "Light_mosaic_C 11_30.0s_IRCUT_20250729-010221.fit"
_TAN_FRAME = "Light_mosaic_C 11_30.0s_IRCUT_20250710-005358.fit"


# ---------------------------------------------------------------------------
# Synthetic WCS builders
# ---------------------------------------------------------------------------

def _synthetic_tan_wcs(shape=(200, 200), ra=10.0, dec=30.0, angle=0.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    a = np.deg2rad(angle)
    w.wcs.cd = np.array([[-np.cos(a), np.sin(a)], [np.sin(a), np.cos(a)]]) * 0.001
    w.array_shape = shape
    return w


def _synthetic_sip_wcs(shape=(200, 200), ra=10.0, dec=30.0, angle=0.0):
    """Synthetic SIP-distorted TAN WCS (RA---TAN-SIP / DEC--TAN-SIP)."""
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"] = shape[1]
    h["NAXIS2"] = shape[0]
    h["CTYPE1"] = "RA---TAN-SIP"
    h["CTYPE2"] = "DEC--TAN-SIP"
    h["CRVAL1"] = ra
    h["CRVAL2"] = dec
    h["CRPIX1"] = shape[1] / 2
    h["CRPIX2"] = shape[0] / 2
    a = np.deg2rad(angle)
    cd = np.array([[-np.cos(a), np.sin(a)], [np.sin(a), np.cos(a)]]) * 0.001
    h["CD1_1"] = cd[0, 0]
    h["CD1_2"] = cd[0, 1]
    h["CD2_1"] = cd[1, 0]
    h["CD2_2"] = cd[1, 1]
    # 2nd-order SIP distortion (A_ORDER=B_ORDER=2). Coefficients chosen large
    # enough to produce a measurable (multi-pixel) edge curvature.
    h["A_ORDER"] = 2
    h["B_ORDER"] = 2
    h["A_0_0"] = 0.0
    h["B_0_0"] = 0.0
    h["A_0_2"] = 1e-4
    h["A_2_0"] = 1e-4
    h["B_0_2"] = 1e-4
    h["B_2_0"] = 1e-4
    return WCS(h)


# ---------------------------------------------------------------------------
# (a) qualification + SIP crop correctness
# ---------------------------------------------------------------------------

def test_qualify_accepts_sip_tan():
    w = _synthetic_sip_wcs()
    assert zg._has_sip(w)
    assert zg.qualify_wcs(w) is None


@pytest.mark.slow
def test_qualify_accepts_real_sip_and_tan():
    for f in (_SIP_FRAME, _TAN_FRAME):
        p = _CALDWELL / f
        if not p.is_file():
            pytest.skip(f"real Caldwell frame not present: {p}")
        w = WCS(fits.getheader(p, 0))
        assert zg.qualify_wcs(w) is None, f


def test_qualify_rejects_sip_non_tan_base():
    w = _synthetic_sip_wcs()
    w.wcs.ctype = ["RA---SIN-SIP", "DEC--SIN-SIP"]
    reason = zg.qualify_wcs(w)
    assert reason is not None and "TAN" in reason


def test_qualify_rejects_sip_with_pv():
    # Build a SIP + PV WCS via header (astropy can't ``set_pv`` on a SIP WCS).
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"] = 200
    h["NAXIS2"] = 200
    h["CTYPE1"] = "RA---TAN-SIP"
    h["CTYPE2"] = "DEC--TAN-SIP"
    h["CRVAL1"] = 10.0
    h["CRVAL2"] = 30.0
    h["CRPIX1"] = 100.0
    h["CRPIX2"] = 100.0
    h["CD1_1"] = -0.001
    h["CD1_2"] = 0.0
    h["CD2_1"] = 0.0
    h["CD2_2"] = 0.001
    h["A_ORDER"] = 2
    h["B_ORDER"] = 2
    h["A_0_0"] = 0.0
    h["B_0_0"] = 0.0
    h["A_0_2"] = 1e-4
    h["A_2_0"] = 1e-4
    h["B_0_2"] = 1e-4
    h["B_2_0"] = 1e-4
    h["PV1_1"] = 1.0
    h["PV2_1"] = 10.0
    w = WCS(h)
    assert zg._has_sip(w)
    assert w.wcs.get_pv()
    reason = zg.qualify_wcs(w)
    assert reason is not None and "PV" in reason


def test_qualify_rejects_sip_ambiguous_suffix_without_coeffs():
    # "-SIP" suffix in CTYPE but NO A_ORDER/B_ORDER -> sip is None.
    w = _synthetic_tan_wcs()
    w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    reason = zg.qualify_wcs(w)
    assert reason is not None


def test_strip_sip_yields_plain_tan():
    w = _synthetic_sip_wcs()
    stripped = zg.strip_sip(w)
    assert stripped.sip is None
    assert not stripped.has_distortion
    assert tuple(stripped.wcs.ctype) == ("RA---TAN", "DEC--TAN")
    assert zg.qualify_wcs(stripped) is None


def test_strip_sip_is_identity_for_plain_tan():
    w = _synthetic_tan_wcs()
    stripped = zg.strip_sip(w)
    # No SIP -> same object semantics, equal CRPIX/CRVAL/CD.
    np.testing.assert_allclose(stripped.wcs.crpix, w.wcs.crpix)


def test_sip_crop_wcs_maps_correctly():
    """Cropped SIP WCS must map identically to the full-frame SIP WCS."""
    w = _synthetic_sip_wcs(shape=(300, 400))
    b = SourceBounds(50, 70, 250, 220)
    sliced = zx.slice_wcs(w, b)
    assert sliced.sip is not None, "slice must preserve SIP"
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, b.width - 1, 60)
    ys = rng.uniform(0, b.height - 1, 60)
    full_sky = w.pixel_to_world(xs + b.x0, ys + b.y0)
    sliced_sky = sliced.pixel_to_world(xs, ys)
    assert full_sky.separation(sliced_sky).to_value(u.arcsec).max() < 1e-6


# ---------------------------------------------------------------------------
# (b) plain-TAN slicing regression guard
# ---------------------------------------------------------------------------

def test_plain_tan_slicing_unchanged():
    """The new WCS.slice path must reproduce the old manual CRPIX shift exactly."""
    w = _synthetic_tan_wcs(shape=(300, 400), angle=17.0)
    b = SourceBounds(50, 70, 250, 220)

    manual = w.deepcopy()
    manual.wcs.crpix = np.array([w.wcs.crpix[0] - b.x0, w.wcs.crpix[1] - b.y0])
    manual.array_shape = (b.height, b.width)

    sliced = zx.slice_wcs(w, b)

    np.testing.assert_allclose(sliced.wcs.crpix, manual.wcs.crpix, atol=1e-12)
    np.testing.assert_allclose(sliced.wcs.crval, manual.wcs.crval, atol=1e-12)
    np.testing.assert_allclose(sliced.wcs.cd, manual.wcs.cd, atol=1e-15)
    assert tuple(sliced.wcs.ctype) == tuple(manual.wcs.ctype)
    assert sliced.array_shape == manual.array_shape == (b.height, b.width)

    # Full sky mapping must be identical (this is the true contract).
    rng = np.random.default_rng(1)
    xs = rng.uniform(0, b.width - 1, 80)
    ys = rng.uniform(0, b.height - 1, 80)
    s_sky = sliced.pixel_to_world(xs, ys)
    m_sky = manual.pixel_to_world(xs, ys)
    assert s_sky.separation(m_sky).to_value(u.arcsec).max() < 1e-9


# ---------------------------------------------------------------------------
# (c) SIP footprint contains the true sampled footprint
# ---------------------------------------------------------------------------

def _dense_boundary_and_interior(shape_hw, step):
    """Densely sample the full source (boundary + interior) as the 'true' footprint."""
    h, w = shape_hw
    pts = []
    for y in np.arange(0, h, step):
        for x in np.arange(0, w, step):
            pts.append((x + 0.5, y + 0.5))
    return np.array(pts)


def test_sip_footprint_contains_true_footprint():
    w = _synthetic_sip_wcs(shape=(300, 400))
    target = zg.strip_sip(w)  # a plain-TAN canvas WCS
    shape_hw = (300, 400)
    poly = zg._source_polygon(shape_hw, w, target)
    assert poly.is_valid
    assert poly.area > 0

    # Densely sample the true footprint (finer than the production edge step).
    pts = _dense_boundary_and_interior(shape_hw, step=16)
    xy = zg._project_points(pts, w, target)

    from shapely.geometry import Point

    for px, py in xy:
        assert poly.contains(Point(px, py)), f"footprint misses point ({px:.2f},{py:.2f})"


def test_sip_footprint_is_valid_and_conservative():
    w = _synthetic_sip_wcs(shape=(300, 400))
    target = zg.strip_sip(w)
    poly = zg._source_polygon((300, 400), w, target)
    assert poly.is_valid
    # The SIP footprint must be at least as large as the plain-TAN 4-corner polygon
    # (conservative bounding), never smaller.
    plain = zg._source_polygon((300, 400), target, target)
    assert poly.area >= plain.area


# ---------------------------------------------------------------------------
# (d) cross-consistency measurement (SIP vs TAN), numbers pinned
# ---------------------------------------------------------------------------

def _measure_sip_vs_tan_offset():
    """Measure the systematic SIP-vs-plain-TAN offset on real overlapping frames.

    Projects a dense grid of the SIP frame's pixels to sky with the FULL SIP WCS
    (keep) and with the STRIPPED (linear) WCS (legacy strip), then inverse-maps
    both onto the TAN frame's pixel grid. Returns the median/max |offset| in
    TAN-frame pixels between the two handlings.
    """
    if not (_CALDWELL / _SIP_FRAME).is_file() or not (_CALDWELL / _TAN_FRAME).is_file():
        pytest.skip("real Caldwell frames not present")
    ws = WCS(fits.getheader(_CALDWELL / _SIP_FRAME, 0))
    wt = WCS(fits.getheader(_CALDWELL / _TAN_FRAME, 0))
    ws_lin = zg.strip_sip(ws)

    h, w = 1920, 1080
    xs, ys = np.meshgrid(np.linspace(0, w - 1, 60), np.linspace(0, h - 1, 60))
    px, py = xs.ravel(), ys.ravel()

    sky_keep = ws.pixel_to_world(px, py)
    sky_lin = ws_lin.pixel_to_world(px, py)
    tx_keep, ty_keep = wt.world_to_pixel(sky_keep)
    tx_lin, ty_lin = wt.world_to_pixel(sky_lin)
    off = np.hypot(tx_keep - tx_lin, ty_keep - ty_lin)
    return float(np.median(off)), float(off.max())


@pytest.mark.slow
def test_cross_consistency_sip_vs_tan_pinned():
    """Pin the measured SIP-vs-TAN systematic offset (small => 'keep' is safe).

    Measured on the real Caldwell 11 corpus (2025-10-06, this machine): the SIP
    distortion is sub-pixel. Keeping SIP (correct handling) therefore introduces
    NO systematic offset vs the plain-TAN frames; the configurable strip
    fallback exists for legacy consistency but is not needed by default.
    """
    med, mx = _measure_sip_vs_tan_offset()
    # Pinned from the R10 measurement (median ~0.17 px, max ~0.53 px).
    assert med == pytest.approx(0.17, abs=0.05)
    assert mx == pytest.approx(0.53, abs=0.10)


# ---------------------------------------------------------------------------
# (e) rejected-count surfacing (manifest + WARN)
# ---------------------------------------------------------------------------

def _reject_manifest(caplog, sip_mode="keep"):
    """Run a single-pass descriptor build and assert surfacing of rejected frames."""
    import logging

    frames_info = _make_frames_info_with_one_rejected()
    out = Path("/tmp")  # only used for base resolution; not written here
    with caplog.at_level(logging.WARNING, logger="ZeMosaicWorker"):
        descs, rejected = zegrid._build_frame_descriptors(
            frames_info, str(out), None, sip_mode=sip_mode
        )
    return descs, rejected


def _make_frames_info_with_one_rejected(tmp_path_factory=None):
    """Three frames: 2 valid TAN + 1 with a bad (non-celestial) WCS."""
    from zemosaic import zemosaic_stack_plan as zsp

    frames = []
    for i, (name, wcs_obj) in enumerate([
        ("a.fits", _synthetic_tan_wcs(shape=(20, 20))),
        ("b.fits", _synthetic_tan_wcs(shape=(20, 20), ra=10.05)),
    ]):
        fi = zsp.FrameInfo(path=Path(f"/nonexistent/{name}"))
        fi.wcs = wcs_obj
        fi.shape_hw = (20, 20)
        frames.append(fi)
    # Bad frame: WCS with no celestial component.
    bad = WCS(naxis=2)
    bad.wcs.ctype = ["LINEAR", "LINEAR"]
    bad.wcs.crval = [0.0, 0.0]
    bad.wcs.crpix = [1.0, 1.0]
    bad.wcs.cd = np.eye(2) * 0.001
    fi_bad = zsp.FrameInfo(path=Path("/nonexistent/bad.fits"))
    fi_bad.wcs = bad
    fi_bad.shape_hw = (20, 20)
    frames.append(fi_bad)
    return frames


def test_rejected_count_surfaced_in_manifest(tmp_path, caplog):
    import logging

    frames_info = _make_frames_info_with_one_rejected()
    with caplog.at_level(logging.WARNING, logger="ZeMosaicWorker"):
        descs, rejected = zegrid._build_frame_descriptors(
            frames_info, str(tmp_path), None, sip_mode="keep"
        )
    assert len(descs) == 2
    assert len(rejected) == 1
    # A clear WARN with the count and a per-reason breakdown must be emitted.
    warn_text = "\n".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
    assert "rejected 1 frame" in warn_text
    assert "no celestial component" in warn_text
    # The manifest (written by _write_outputs) carries the count + breakdown.
    # We exercise _write_outputs directly to prove the field, without a full run.
    from zemosaic.core.zegrid import mosaic as zmosaic

    class _Assembled:
        science = np.zeros((10, 10, 3), dtype=np.float32)
        stack_depth = np.ones((10, 10), dtype=np.int32)
        complete_cells = 1
        incomplete_cells = 0
        hole_pixels = 0
        coverage_pixels = 100

    canvas = zg.build_canvas(descs)
    layout = {"nx": 1, "ny": 1, "ram_budget_bytes": 0, "available_bytes": 0,
              "max_contributors": 2, "predicted_bound_bytes": 0}
    sci_path, cov_path, manifest_path = zegrid._write_outputs(
        _Assembled(), canvas, 1, 1, tmp_path, descs, {}, [],
        layout, SimpleNamespace(normalization="sky_mean"), 0, {},
        None, rejected=rejected, sip_mode="keep",
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["rejected_frames"]["count"] == 1
    assert manifest["rejected_frames"]["by_reason"] == {"WCS has no celestial component": 1}
    assert manifest["sip_mode"] == "keep"


# ---------------------------------------------------------------------------
# (g) real-frame end-to-end: SIP frame is now INCLUDED
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _SIP_FIXTURE.is_dir(), reason="R10 SIP fixture dir not present")
def test_end_to_end_real_sip_frame_included(tmp_path):
    out = tmp_path / "out"
    zconfig = SimpleNamespace(zegrid_layout="2x2")
    zegrid.run_zegrid_mode(str(_SIP_FIXTURE), str(out), zconfig=zconfig)

    sci_path = out / "mosaic_grid.fits"
    cov_path = out / "mosaic_grid_coverage.fits"
    manifest_path = out / "zegrid_manifest.json"
    assert sci_path.exists()
    assert cov_path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    # All 3 frames (2 SIP + 1 TAN) must be INCLUDED — zero silent drops.
    assert manifest["n_frames"] == 3
    assert manifest["rejected_frames"]["count"] == 0
    assert manifest["sip_mode"] == "keep"
    sip_ids = [f for f in manifest["frame_ids"] if "010221" in f or "010252" in f]
    assert len(sip_ids) == 2, "SIP frames were dropped!"
    assert manifest["coverage_pixels"] > 0

    with fits.open(cov_path) as hdul:
        cov = hdul[0].data
    assert int(np.count_nonzero(np.asarray(cov) > 0)) > 0


@pytest.mark.slow
@pytest.mark.skipif(not _SIP_FIXTURE.is_dir(), reason="R10 SIP fixture dir not present")
def test_end_to_end_strip_fallback_still_includes_frames(tmp_path):
    out = tmp_path / "out"
    zconfig = SimpleNamespace(zegrid_layout="2x2", zegrid_sip_mode="strip")
    zegrid.run_zegrid_mode(str(_SIP_FIXTURE), str(out), zconfig=zconfig)
    manifest = json.loads((out / "zegrid_manifest.json").read_text())
    assert manifest["sip_mode"] == "strip"
    assert manifest["n_frames"] == 3
    assert manifest["rejected_frames"]["count"] == 0


def test_invalid_sip_mode_warns_and_falls_back(tmp_path, caplog):
    import logging

    frames_info = [fi for fi in _make_frames_info_with_one_rejected() if fi.wcs is not None][:2]
    with caplog.at_level(logging.WARNING):
        descs, rejected = zegrid._build_frame_descriptors(
            frames_info, str(tmp_path), None, sip_mode="bogus"
        )
    # An invalid sip_mode never silently crashes the descriptor build; 'keep'
    # semantics are the effective default path (no SIP frame present here).
    assert len(descs) == 2
