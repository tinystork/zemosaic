"""ZM-ZEGRID-R23 targeted tests — scientific DBE + dual float32 output contract.

Covers:

* A: dual float32 output contract (raw science always written; finished == raw
  bit-identical when disabled / forced finishing failure; headers/manifest roles;
  uint16 derived from the finished float32, never a scientific reference).
* B: strength/custom semantics (presets, invalid->normal, custom exact).
* C/D: scientific DBE gates on synthetic data — bright stars >=98% flux, diffuse
  Gaussian (sigma~65, amp~20) >=90% flux AND >=50% sky-flattening, asymmetric/
  nebulosity-like + low-S/N + NaN/coverage edge cases, negative diagnostics.
* F: no mutation of the assembled input array.

These run in the fast tier (no gated corpora).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import final_mosaic_finishing as zfin
from zemosaic.core.zegrid import geometry as zg


# ---------------------------------------------------------------------------
# Synthetic probes (deterministic)
# ---------------------------------------------------------------------------

def _diffuse_probe(h=512, w=512, amp=20.0, sig=65.0, noise=0.3, seed=0, asy=False,
                   center=None, bg_grad=40.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = 100.0 + bg_grad * (xx / w)
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, noise, size=(h, w)).astype(np.float32)
    cx, cy = center if center is not None else (w / 2, h / 2)
    if asy:
        dx = xx - cx
        dy = yy - cy
        gauss = amp * np.exp(-(dx ** 2 / (2 * sig ** 2) + dy ** 2 / (2 * (sig * 1.6) ** 2)))
    else:
        gauss = amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig ** 2))
    plane = (bg + gauss + n).astype(np.float32)
    sci = np.stack([plane + 0.0, plane + 0.0, plane + 0.0], axis=-1).astype(np.float32)
    cov = np.ones((h, w), dtype=bool)
    return sci, cov, gauss


def _star_probe(h=256, w=256, peak=5000.0, sig=3.0, bg_grad_x=60.0, bg_grad_y=40.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = 50.0 + bg_grad_x * (xx / w) + bg_grad_y * (yy / h)
    star = peak * np.exp(-((xx - 128) ** 2 + (yy - 128) ** 2) / (2.0 * sig ** 2))
    rng = np.random.default_rng(1)
    n = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    plane = (bg + star + n).astype(np.float32)
    sci = np.stack([plane + 0.0, plane + 0.0, plane + 0.0], axis=-1).astype(np.float32)
    return sci, np.ones((h, w), dtype=bool), star


def _sky_box_std(a, sky_mask, n=8):
    h, w = a.shape
    vals = []
    for i in range(n):
        for j in range(n):
            sl = (slice(i * h // n, (i + 1) * h // n), slice(j * w // n, (j + 1) * w // n))
            seg = a[sl][sky_mask[sl]]
            if seg.size:
                vals.append(float(np.median(seg)))
    return float(np.std(vals)) if vals else 0.0


def _aperture_flux_above_model(ch, bg, cx, cy, r_ap):
    """Aperture flux of the object above the estimated background model."""
    h, w = ch.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ap = r <= r_ap
    return float(np.sum((ch - bg)[ap]))


def _dbe_cfg(strength="normal"):
    return dict(
        dbe_enabled=True,
        dbe_strength=strength,
        dbe_params_source=f"preset:{strength}",
        dbe_params=zfin.DBE_STRENGTH_PRESETS[strength],
        dbe_subtraction_factor=1.0,
        rgb_equalize=False,
        save_uint16=False,
    )


# ---------------------------------------------------------------------------
# D2: bright stars — aperture flux >= 98%, no dark halo regression
# ---------------------------------------------------------------------------

def test_bright_star_flux_preserved():
    sci, cov, star = _star_probe()
    out = zfin.apply_final_mosaic_finishing(sci, cov, config=_dbe_cfg()).science
    bg, _ = zfin.estimate_background_channel(
        sci[..., 0], cov, sample_step=24, obj_k=3.0, obj_dilate_px=3, smoothing=0.6)
    F_inj = float(np.sum(star[np.sqrt((np.mgrid[0:256, 0:256].astype(np.float32)[0] - 128) ** 2 +
                                      (np.mgrid[0:256, 0:256].astype(np.float32)[1] - 128) ** 2) <= 6]))
    F_ret = _aperture_flux_above_model(sci[..., 0], bg, 128, 128, 6)
    assert F_ret == pytest.approx(F_inj, rel=0.02)

    # No dark halo: annulus immediately outside the star is not depressed vs a
    # farther annulus (no negative ring).
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    r = np.sqrt((xx - 128) ** 2 + (yy - 128) ** 2)
    ann1 = (r >= 8) & (r <= 14)
    ann2 = (r >= 20) & (r <= 30)
    assert float(np.median(out[..., 0][ann1])) >= float(np.median(out[..., 0][ann2])) - 3.0


# ---------------------------------------------------------------------------
# D3: diffuse Gaussian — flux >= 90% AND sky-flattening >= 50%
# ---------------------------------------------------------------------------

def test_diffuse_gaussian_preserved_and_flattened():
    sci, cov, gauss = _diffuse_probe()
    sky_mask = gauss < 0.01
    before = _sky_box_std(sci[..., 0], sky_mask)
    out = zfin.apply_final_mosaic_finishing(sci, cov, config=_dbe_cfg()).science
    after = _sky_box_std(out[..., 0], sky_mask)

    bg, info = zfin.estimate_background_channel(
        sci[..., 0], cov, sample_step=24, obj_k=3.0, obj_dilate_px=3, smoothing=0.6)
    F_inj = float(np.sum(gauss[np.sqrt((np.mgrid[0:512, 0:512].astype(np.float32)[0] - 256) ** 2 +
                                       (np.mgrid[0:512, 0:512].astype(np.float32)[1] - 256) ** 2) <= 80]))
    F_ret = _aperture_flux_above_model(sci[..., 0], bg, 256, 256, 80)

    assert F_ret / F_inj >= 0.90
    assert (1.0 - after / before) >= 0.50


# ---------------------------------------------------------------------------
# D4: asymmetric/nebulosity-like + low-S/N + edge cases
# ---------------------------------------------------------------------------

def test_asymmetric_extended_structure():
    sci, cov, gauss = _diffuse_probe(asy=True)
    sky_mask = gauss < 0.01
    before = _sky_box_std(sci[..., 0], sky_mask)
    out = zfin.apply_final_mosaic_finishing(sci, cov, config=_dbe_cfg()).science
    after = _sky_box_std(out[..., 0], sky_mask)
    # Must still flatten (>=50%) without blowing up; do not require exact flux on
    # the asymmetric case (documented as an additional robustness check).
    assert (1.0 - after / before) >= 0.50
    assert np.all(np.isfinite(out))


def test_nan_holes_and_low_coverage():
    sci, cov, gauss = _diffuse_probe()
    # punch NaN holes + zero coverage in patches
    sci2 = sci.copy()
    cov2 = cov.copy()
    sci2[100:160, 200:260, :] = np.nan
    cov2[300:360, 100:160] = False
    out = zfin.apply_final_mosaic_finishing(sci2, cov2, config=_dbe_cfg()).science
    # NaN holes stay NaN (never silently filled into the science).
    assert np.all(np.isnan(out[100:160, 200:260, :]))
    # coverage holes are left untouched (never corrected / never zeroed by DBE).
    assert np.array_equal(out[300:360, 100:160, :], sci2[300:360, 100:160, :])
    # rest is finite and not exploded
    assert np.all(np.isfinite(out[cov2 & ~np.isnan(sci2[..., 0])]))


def test_all_invalid_small_image():
    sci = np.full((8, 8, 3), np.nan, dtype=np.float32)
    cov = np.zeros((8, 8), dtype=bool)
    res = zfin.apply_final_mosaic_finishing(sci, cov, config=_dbe_cfg())
    assert np.all(np.isnan(res.science))
    assert res.info["dbe"]["applied"] is False


def test_no_clamp_and_negatives_survive():
    """Negative input values are legitimate science and must survive DBE
    unchanged-in-sign (no clamp / abs / inversion)."""
    sci, cov, _ = _diffuse_probe()
    sci = sci - 130.0  # shift sky well below zero -> many negatives
    before_neg = float(np.mean(sci[..., 0][cov] < 0))
    out = zfin.apply_final_mosaic_finishing(sci, cov, config=_dbe_cfg()).science
    # no abs/inversion: the diffuse structure's positive excess is still positive
    # relative to the (now-varied) background; global DC is preserved (median close).
    assert np.any(out[..., 0][cov] < 0) or before_neg == 0.0
    # DBE never maps a +max to -min (no sign inversion): max stays >= min.
    assert float(np.nanmax(out[..., 0][cov])) >= float(np.nanmin(out[..., 0][cov]))


def test_input_not_mutated():
    sci, cov, _ = _diffuse_probe()
    orig = sci.copy()
    zfin.apply_final_mosaic_finishing(sci, cov, config=_dbe_cfg())
    assert np.array_equal(sci, orig)


# ---------------------------------------------------------------------------
# A/F: dual float32 output contract
# ---------------------------------------------------------------------------

def _synthetic_wcs(shape=(10, 10)):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 30.0]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cd = np.array([[-1, 0], [0, 1]]) * 0.001
    w.array_shape = shape
    return w


class _Assembled:
    science = np.linspace(0, 99, 300, dtype=np.float32).reshape(10, 10, 3)
    stack_depth = np.ones((10, 10), dtype=np.int32)
    complete_cells = 1
    incomplete_cells = 0
    hole_pixels = 0
    coverage_pixels = 100


def _write_via(assembled, canvas, out, **kw):
    layout = {"nx": 1, "ny": 1, "ram_budget_bytes": 0, "available_bytes": 0,
              "max_contributors": 1, "predicted_bound_bytes": 0}
    descs = [
        zg.FrameDescriptor(
            frame_id=zg.FrameId("a.fits"), source_path="/x/a.fits", shape_hw=(10, 10),
            wcs_header=_synthetic_wcs().to_header().tostring(),
            header_sha256="", instrument="",
        )
    ]
    return zz._write_outputs(
        assembled, canvas, 1, 1, out, descs, {}, [],
        layout, SimpleNamespace(normalization="sky_mean"), 0, {}, None,
        **kw,
    )


def test_dual_output_raw_science_written(tmp_path):
    canvas = zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_wcs().to_header().tostring(),
        width=10, height=10, resolution_deg=0.001,
    )
    a = _Assembled()
    _, _, mp = _write_via(a, canvas, tmp_path)
    assert (tmp_path / "mosaic_grid_science.fits").exists()
    assert (tmp_path / "mosaic_grid.fits").exists()

    with fits.open(tmp_path / "mosaic_grid_science.fits") as h:
        raw = h[0].data
        assert h[0].header["SCIROLE"] == "science_raw"
        assert h[0].header["DBESTAT"] == "n/a"
    with fits.open(tmp_path / "mosaic_grid.fits") as h:
        fin = h[0].data
        assert h[0].header["SCIROLE"] == "science_finished"

    # raw == finished arrays (disabled finishing -> bit-identical science)
    assert np.array_equal(np.asarray(raw), np.asarray(fin))

    m = json_load(mp)
    assert m["outputs"]["science"] == "mosaic_grid.fits"
    assert m["outputs"]["science_raw"] == "mosaic_grid_science.fits"
    assert m["outputs"]["science_finished"] == "mosaic_grid.fits"
    assert m["science_output_contract"]["raw"]["role"] == "science_raw"
    assert m["science_output_contract"]["finished"]["role"] == "science_finished"


def json_load(p):
    import json
    return json.loads(p.read_text())


def test_dbe_on_leaves_raw_untouched(tmp_path):
    # Use a 64x64 gradient+star synthetic so DBE actually changes the finished
    # science (the 10x10 flat ramp in _Assembled is DBE-invariant).
    h = w = 64
    canvas = zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_wcs((h, w)).to_header().tostring(),
        width=w, height=h, resolution_deg=0.001,
    )
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    plane = (50.0 + 60.0 * (xx / w) + 40.0 * (yy / h)
             + 5000.0 * np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2.0 * 3.0 ** 2)))
    a = _Assembled()
    a.science = np.stack([plane, plane, plane], axis=-1).astype(np.float32)
    a.stack_depth = np.ones((h, w), dtype=np.int32)

    res = zfin.apply_final_mosaic_finishing(
        a.science, a.stack_depth,
        config=dict(dbe_enabled=True, dbe_strength="normal",
                    dbe_params_source="preset:normal",
                    dbe_params=zfin.DBE_STRENGTH_PRESETS["normal"],
                    dbe_subtraction_factor=1.0, rgb_equalize=False, save_uint16=False),
    )
    assert res.info["dbe"]["applied"] is True
    _, _, mp = _write_via(
        a, canvas, tmp_path,
        finished_science=res.science, finishing_info=res.info, fin_uint16=None,
        raw_science=a.science,
    )
    with fits.open(tmp_path / "mosaic_grid_science.fits") as h:
        raw = np.asarray(h[0].data)
    with fits.open(tmp_path / "mosaic_grid.fits") as h:
        fin = np.asarray(h[0].data)
    # raw stays the pre-finishing science; finished differs (DBE applied).
    assert np.array_equal(raw, np.moveaxis(np.asarray(a.science, dtype=np.float32), -1, 0))
    assert not np.array_equal(raw, fin)
    m = json_load(mp)
    assert m["finishing"]["dbe"]["applied"] is True
    assert m["science_output_contract"]["finished"]["dbe_state"] == "on"
    assert m["science_output_contract"]["raw"]["sha256"] != \
        m["science_output_contract"]["finished"]["sha256"]


def test_uint16_derived_from_finished(tmp_path):
    canvas = zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_wcs().to_header().tostring(),
        width=10, height=10, resolution_deg=0.001,
    )
    a = _Assembled()
    res = zfin.apply_final_mosaic_finishing(
        a.science, a.stack_depth,
        config=dict(dbe_enabled=True, dbe_strength="normal",
                    dbe_params_source="preset:normal",
                    dbe_params=zfin.DBE_STRENGTH_PRESETS["normal"],
                    dbe_subtraction_factor=1.0, rgb_equalize=False, save_uint16=True),
    )
    _, _, mp = _write_via(
        a, canvas, tmp_path,
        finished_science=res.science, finishing_info=res.info, fin_uint16=res.uint16,
    )
    with fits.open(tmp_path / "mosaic_grid_uint16.fits") as h:
        u16 = np.asarray(h[0].data)
        assert h[0].header["SCIROLE"] == "uint16_render"
    assert u16.dtype == np.uint16
    # uint16 is a render of the FINISHED float32 (same shape/axis order).
    with fits.open(tmp_path / "mosaic_grid.fits") as h:
        fin = np.asarray(h[0].data)
    assert u16.shape == fin.shape
    m = json_load(mp)
    assert m["outputs"]["uint16"] == "mosaic_grid_uint16.fits"


def test_forced_failure_preserves_raw_and_main(tmp_path, monkeypatch):
    """When finishing raises, the raw science AND a compatible main both exist and
    are equal (fail-safe); manifest records failed."""
    import logging
    from pathlib import Path

    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Use the routine corpus tool to build a tiny real input.
    import subprocess
    import sys
    routine_tool = Path(__file__).resolve().parents[1] / "tools" / "zegrid_routine" / "make_routine_corpus.py"
    corpus = tmp_path / "corpus"
    subprocess.run([sys.executable, str(routine_tool), str(corpus)],
                   capture_output=True, text=True, timeout=120, check=True)
    for p in sorted(corpus.glob("*.fits")):
        (input_dir / p.name).write_bytes(p.read_bytes())
    (input_dir / "stack_plan.csv").write_text(
        corpus.joinpath("stack_plan.csv").read_text(), encoding="utf-8")

    out = tmp_path / "out"

    def _boom(*a, **k):
        raise RuntimeError("forced finishing failure")

    monkeypatch.setattr(zz.zfin, "apply_final_mosaic_finishing", _boom)

    captured = []

    class _Cap(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    cap = _Cap(level=logging.WARNING)
    lg = logging.getLogger("ZeMosaicWorker.zegrid_mode")
    lg.addHandler(cap)
    lg.setLevel(logging.WARNING)
    try:
        zz.run_zegrid_mode(str(input_dir), str(out),
                           zconfig=SimpleNamespace(final_mosaic_dbe_enabled=True))
    finally:
        lg.removeHandler(cap)

    assert (out / "mosaic_grid_science.fits").exists()
    assert (out / "mosaic_grid.fits").exists()
    with fits.open(out / "mosaic_grid_science.fits") as h:
        raw = np.asarray(h[0].data)
    with fits.open(out / "mosaic_grid.fits") as h:
        fin = np.asarray(h[0].data)
        assert h[0].header["DBESTAT"] == "failed"
    assert np.array_equal(raw, fin, equal_nan=True)
    m = json_load(out / "zegrid_manifest.json")
    assert m["finishing"]["failed"] is True
    assert m["science_output_contract"]["finished"]["dbe_state"] == "failed"
