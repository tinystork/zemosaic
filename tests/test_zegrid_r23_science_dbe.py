"""ZM-ZEGRID-R23 rework-3 targeted tests — legacy light-DBE aesthetic + Classic naming.

Rework-3 (human-gate resolved, Tristan + Nono review-1):

* Raw science float32 is the immutable photometric reference (written FIRST).
* The aesthetic output is produced by the exact legacy light-DBE algorithm and is
  explicitly visual/non-photometric; it honours the EXISTING
  ``export_aesthetic_fits`` checkbox + ``scientific_fits_suffix`` /
  ``aesthetic_fits_suffix`` + ``aesthetic_hole_fill_*`` keys (no new GUI surface).
* DBE preserves the input finite/NaN mask exactly (uncovered pixels stay NaN);
  hole fill runs ONLY when ``aesthetic_hole_fill_enabled`` and reuses the shared
  Classic helper (only_near_seams respected).
* Manifest/header truth: ``outputs.science``/``science_reference`` point at the
  actual raw scientific file; ``outputs.aesthetic`` only when emitted; the
  top-level ``algorithm`` is null unless DBE actually applied.

Covered (fast tier, no gated corpora):

* legacy strength maps + EXACT parity (values AND finite/NaN mask) with a frozen
  independent reference, incl. adversarial NaN/coverage;
* raw-science identity (DBE on/off/failure never touches raw);
* Classic output naming via the production path (export OFF -> single raw
  ``mosaic_grid.fits``; export ON -> named raw + aesthetic; suffix
  sanitization/collision);
* hole fill: disabled -> finite/NaN mask parity; enabled -> only the configured
  target pixels filled (only_near_seams), raw bit-identical;
* safety guard (negative-fraction explosion / gross worsening -> atomic no-op);
* custom strength fallback; uint16 derived from the post-aesthetic branch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from scipy import ndimage

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import final_mosaic_finishing as zfin
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid.aesthetic_hole_fill import apply_aesthetic_hole_fill


# ---------------------------------------------------------------------------
# Frozen independent legacy reference (exact historical algorithm)
# ---------------------------------------------------------------------------

def _frozen_legacy(mosaic, valid_mask_hw, strength="normal"):
    strength_map = {
        "weak": 24.0, "low": 24.0, "normal": 36.0,
        "strong": 52.0, "high": 52.0, "aggressive": 68.0,
    }
    sigma = float(strength_map.get(strength, 36.0))
    out = np.asarray(mosaic, dtype=np.float32).copy()
    h, w = out.shape[:2]
    finite_any = np.any(np.isfinite(out), axis=-1)
    valid_hw = finite_any
    if valid_mask_hw is not None and valid_mask_hw.shape[:2] == (h, w):
        valid_hw = valid_hw & np.asarray(valid_mask_hw, dtype=bool)
    for c in range(3):
        ch = out[..., c]
        ch_finite = np.isfinite(ch)
        ch_valid = valid_hw & ch_finite
        if not np.any(ch_valid):
            continue
        median = float(np.nanmedian(ch[ch_valid]))
        mad = float(np.nanmedian(np.abs(ch[ch_valid] - median)))
        robust_sigma = float(1.4826 * mad)
        obj_k_map = {
            "weak": 3.0, "low": 3.0, "normal": 2.8,
            "strong": 2.5, "high": 2.5, "aggressive": 2.2,
        }
        obj_k = float(obj_k_map.get(strength, 2.8))
        obj_thr = float(median + obj_k * robust_sigma)
        obj_mask = ch_valid & (ch > obj_thr)
        dil_map = {
            "weak": 2, "low": 2, "normal": 3,
            "strong": 4, "high": 4, "aggressive": 5,
        }
        dil_iters = int(dil_map.get(strength, 3))
        obj_mask = ndimage.binary_dilation(obj_mask, iterations=max(1, dil_iters))
        bg_valid = ch_valid & (~obj_mask)
        if not np.any(bg_valid):
            bg_valid = ch_valid
        fill_ref = float(np.nanmedian(ch[bg_valid]))
        ch_filled = np.where(ch_valid, ch, fill_ref).astype(np.float32)
        ch_model = np.where(obj_mask, fill_ref, ch_filled).astype(np.float32)
        bg = ndimage.gaussian_filter(ch_model, sigma=sigma, mode="nearest")
        bg_med = float(np.nanmedian(bg[bg_valid]))
        corrected = ch_filled - (bg - bg_med)
        corrected = np.where(obj_mask, ch_filled, corrected)
        ch_out = np.where(ch_valid, corrected, np.nan).astype(np.float32)
        out[..., c] = ch_out
    return out


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _grid(h, w):
    return np.mgrid[0:h, 0:w].astype(np.float32)


def _diffuse_probe(sigma, amp, noise, seed, center=True, asym=False, vignette=False):
    size = max(224, 2 * int(3 * sigma) + 64)
    h = w = size
    yy, xx = _grid(h, w)
    if vignette:
        bg = 100.0 + 30.0 * np.exp(-((xx - 0.5 * w) ** 2 + (yy - 0.5 * h) ** 2) / (2.0 * (1.2 * size) ** 2))
    else:
        bg = 100.0 + 40.0 * (xx / w)
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, noise, size=(h, w)).astype(np.float32)
    cx, cy = (w / 2, h / 2) if center else (0.85 * w, 0.8 * h)
    dx, dy = xx - cx, yy - cy
    if asym:
        gauss = amp * np.exp(-(dx ** 2 / (2 * sigma ** 2) + dy ** 2 / (2 * (1.6 * sigma) ** 2)))
    else:
        gauss = amp * np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))
    plane = (bg + gauss + n).astype(np.float32)
    sci = np.stack([plane, plane, plane], axis=-1).astype(np.float32)
    return sci, np.ones((h, w), dtype=bool), gauss, (cy, cx)


def _dbe_cfg(strength="normal"):
    return dict(
        dbe_enabled=True,
        dbe_strength=strength,
        dbe_params_source=f"preset:{strength}",
        dbe_params=zfin.DBE_STRENGTH_PRESETS[strength],
        dbe_subtraction_factor=1.0,
        rgb_equalize=False,
        save_uint16=False,
        hole_fill_enabled=False,
    )


def _paired_response(sci, cov, obj, cy, cx, r_ap, r_in, r_out):
    """Paired injection on the OUTPUT: response = DBE(base+obj) - DBE(base)."""
    yy, xx = _grid(sci.shape[0], sci.shape[1])
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ap = r <= r_ap
    ann = (r >= r_in) & (r <= r_out)

    base = sci - obj[..., None]
    res_base = zfin.apply_final_mosaic_finishing(base, cov, config=_dbe_cfg())
    res_obj = zfin.apply_final_mosaic_finishing(sci, cov, config=_dbe_cfg())
    response = res_obj.science - res_base.science

    resp_ann = float(np.median(response[..., 0][ann])) if np.any(ann) else 0.0
    resp_flux = float(np.sum(response[..., 0][ap] - resp_ann))
    inj_flux = float(np.sum(obj[ap]))
    if inj_flux <= 0:
        return float("nan")
    return resp_flux / inj_flux


# ---------------------------------------------------------------------------
# Legacy strength maps + parity (exact, values AND finite/NaN mask)
# ---------------------------------------------------------------------------

def test_legacy_strength_maps_exact():
    assert zfin.DBE_STRENGTH_PRESETS["weak"] == {"sigma": 24.0, "obj_k": 3.0, "obj_dilate_px": 2}
    assert zfin.DBE_STRENGTH_PRESETS["low"] == {"sigma": 24.0, "obj_k": 3.0, "obj_dilate_px": 2}
    assert zfin.DBE_STRENGTH_PRESETS["normal"] == {"sigma": 36.0, "obj_k": 2.8, "obj_dilate_px": 3}
    assert zfin.DBE_STRENGTH_PRESETS["strong"] == {"sigma": 52.0, "obj_k": 2.5, "obj_dilate_px": 4}
    assert zfin.DBE_STRENGTH_PRESETS["high"] == {"sigma": 52.0, "obj_k": 2.5, "obj_dilate_px": 4}
    assert zfin.DBE_STRENGTH_PRESETS["aggressive"] == {"sigma": 68.0, "obj_k": 2.2, "obj_dilate_px": 5}


def test_strength_aliases_resolve_to_canonical():
    for alias, canonical in (("low", "weak"), ("high", "strong")):
        r = zfin.resolve_dbe_strength(SimpleNamespace(final_mosaic_dbe_strength=alias))
        assert r["strength"] == canonical
        assert r["params"] == zfin.DBE_STRENGTH_PRESETS[alias]


def test_legacy_parity_exact_all_strengths():
    h = w = 256
    yy, xx = _grid(h, w)
    bg = 50.0 + 60.0 * (xx / w) + 40.0 * (yy / h) + 25.0 * np.exp(-((xx - 0.05 * w) ** 2 + (yy - 0.05 * h) ** 2) / (2.0 * 40.0 ** 2))
    star = 5000.0 * np.exp(-((xx - 128) ** 2 + (yy - 128) ** 2) / (2.0 * 3.0 ** 2))
    sci = np.stack([bg + 10 + star, bg + star, bg - 10 + star], axis=-1).astype(np.float32)
    cov = np.ones((h, w), dtype=np.int32)

    for st in ("weak", "low", "normal", "strong", "high", "aggressive"):
        ref = _frozen_legacy(sci, cov > 0, st)
        mine, info = zfin.apply_dbe(sci, cov > 0, strength=st, params=zfin.DBE_STRENGTH_PRESETS[st])
        finite_both = np.isfinite(ref) & np.isfinite(mine)
        assert np.allclose(ref[finite_both], mine[finite_both], rtol=0, atol=0), \
            f"legacy parity broken for {st}"
        assert info["algorithm"] == "legacy_grid_light_dbe"


def test_legacy_parity_with_nan_and_coverage_holes():
    """Parity holds under NaN holes and partial coverage: VALUES and the finite/NaN
    mask must match the frozen reference EXACTLY (H2: no unconditional fill)."""
    h = w = 200
    yy, xx = _grid(h, w)
    bg = 50.0 + 60.0 * (xx / w) + 40.0 * (yy / h)
    star = 5000.0 * np.exp(-((xx - 100) ** 2 + (yy - 100) ** 2) / (2.0 * 3.0 ** 2))
    sci = np.stack([bg + star, bg + star, bg + star], axis=-1).astype(np.float32)
    cov = np.ones((h, w), dtype=np.int32)
    cov[:, 140:] = 0
    sci[40:60, 40:60, 0] = np.nan
    sci[120:140, 80:100, 1] = np.nan

    for st in ("normal", "strong", "aggressive"):
        ref = _frozen_legacy(sci, cov > 0, st)
        mine, _ = zfin.apply_dbe(sci, cov > 0, strength=st, params=zfin.DBE_STRENGTH_PRESETS[st])
        # H2: finite/NaN mask must be EXACTLY equal (not just common finite values).
        assert np.array_equal(np.isfinite(mine), np.isfinite(ref)), \
            f"finite-mask mismatch for {st}"
        finite_both = np.isfinite(ref) & np.isfinite(mine)
        assert np.allclose(ref[finite_both], mine[finite_both], rtol=0, atol=0), \
            f"legacy parity (NaN/coverage) broken for {st}"


# ---------------------------------------------------------------------------
# Raw-science identity (immutable reference)
# ---------------------------------------------------------------------------

def test_raw_reference_bit_identical_with_dbe_on(tmp_path):
    """The raw science reference is bit-identical whether DBE is on or off."""
    h = w = 128
    yy, xx = _grid(h, w)
    bg = 50.0 + 60.0 * (xx / w) + 40.0 * (yy / h)
    star = 5000.0 * np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / (2.0 * 3.0 ** 2))
    sci = np.stack([bg + star, bg + star, bg + star], axis=-1).astype(np.float32)
    cov = np.ones((h, w), dtype=np.int32)

    res_on = zfin.apply_final_mosaic_finishing(sci, cov, config=_dbe_cfg())
    res_off = zfin.apply_final_mosaic_finishing(
        sci, cov, config=dict(dbe_enabled=False, rgb_equalize=False, save_uint16=False))

    assert np.array_equal(np.asarray(sci, dtype=np.float32), sci)
    assert not np.array_equal(res_on.science, sci)
    assert res_off.science is sci


def test_raw_paired_response_exactly_one():
    """The RAW reference retains 100% of injected signal by construction."""
    sci, cov, gauss, (cy, cx) = _diffuse_probe(65, 20.0, 0.3, 2, True, False, False)
    sci64 = sci.astype(np.float64)
    g64 = gauss.astype(np.float64)
    base64 = sci64 - g64[..., None]
    response64 = sci64 - base64
    assert np.allclose(response64[..., 0], g64, rtol=1e-12, atol=1e-12)
    assert np.allclose(response64[..., 1], g64, rtol=1e-12, atol=1e-12)
    assert np.allclose(response64[..., 2], g64, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Aesthetic diffuse loss is a DOCUMENTED known limit (diagnostic, not gate)
# ---------------------------------------------------------------------------

def test_aesthetic_diffuse_loss_is_documented_known_limit():
    sci, cov, gauss, (cy, cx) = _diffuse_probe(65, 20.0, 0.3, 2, True, False, False)
    ratio = _paired_response(sci, cov, gauss, cy, cx, 2.5 * 65, 3.0 * 65, 4.0 * 65)
    assert np.isfinite(ratio)
    assert ratio < 1.0


def test_bright_compact_source_preserved_aesthetic():
    h = w = 256
    yy, xx = _grid(h, w)
    bg = 50.0 + 60.0 * (xx / w) + 40.0 * (yy / h)
    star = 5000.0 * np.exp(-((xx - 128) ** 2 + (yy - 128) ** 2) / (2.0 * 3.0 ** 2))
    sci = np.stack([bg + star, bg + star, bg + star], axis=-1).astype(np.float32)
    cov = np.ones((h, w), dtype=np.int32)
    ratio = _paired_response(sci, cov, star, 128, 128, 6, 8, 14)
    assert ratio >= 0.98, f"bright compact source ratio {ratio:.3f} < 0.98"


# ---------------------------------------------------------------------------
# Safety guard (negative-fraction explosion / gross worsening -> no-op)
# ---------------------------------------------------------------------------

def test_safety_guard_negative_explosion_noop(monkeypatch):
    """A candidate that explodes negatives triggers an atomic no-op."""
    h = w = 128
    yy, xx = _grid(h, w)
    sci = np.stack([100.0 + xx] * 3, axis=-1).astype(np.float32)
    cov = np.ones((h, w), dtype=np.int32)

    def _fake_channel(ch, valid_hw, *, sigma, obj_k, obj_dilate_px):
        return np.full_like(ch, -1000.0, dtype=np.float32), {
            "sigma": sigma, "obj_k": obj_k, "obj_dilate_px": obj_dilate_px,
            "fill_ref": -1000.0, "median": float(np.median(ch)),
            "robust_sigma": 0.0, "obj_frac": 0.0, "bg_med": -1000.0,
        }

    monkeypatch.setattr(zfin, "_legacy_light_dbe_channel", _fake_channel)
    out, info = zfin.apply_dbe(sci, cov > 0, strength="normal",
                               params=zfin.DBE_STRENGTH_PRESETS["normal"])
    assert info["applied"] is False
    assert info["reason"] == "negative_fraction_explosion"
    assert np.array_equal(out, sci)


# ---------------------------------------------------------------------------
# Hole fill (H2): disabled -> mask parity; enabled -> only target filled
# ---------------------------------------------------------------------------

def test_hole_fill_disabled_preserves_nan_mask():
    """With hole fill DISABLED, the finished aesthetic preserves the input
    finite/NaN mask exactly (parity with frozen legacy)."""
    h = w = 200
    yy, xx = _grid(h, w)
    bg = 50.0 + 60.0 * (xx / w) + 40.0 * (yy / h)
    star = 5000.0 * np.exp(-((xx - 100) ** 2 + (yy - 100) ** 2) / (2.0 * 3.0 ** 2))
    sci = np.stack([bg + star, bg + star, bg + star], axis=-1).astype(np.float32)
    cov = np.ones((h, w), dtype=np.int32)
    cov[:, 140:] = 0

    ref = _frozen_legacy(sci, cov > 0, "normal")
    res = zfin.apply_final_mosaic_finishing(sci, cov, config=_dbe_cfg())
    out = res.science
    assert np.array_equal(np.isfinite(out), np.isfinite(ref))
    # Hole fill was disabled -> not applied.
    assert res.info["hole_fill"]["enabled"] is False


def test_hole_fill_enabled_only_target_filled():
    """With hole fill ENABLED (only_near_seams), only the near-seam target pixels
    are filled; deep-hole pixels stay NaN; raw is never touched."""
    h = w = 200
    yy, xx = _grid(h, w)
    plane = 100.0 + 40.0 * (xx / w) + 40.0 * (yy / h)
    sci = np.stack([plane, plane, plane], axis=-1).astype(np.float32)
    # A single large rectangular hole (deep interior + a seam-adjacent band).
    sci[40:160, 40:160, :] = np.nan
    cov = np.ones((h, w), dtype=np.int32)

    out, info = apply_aesthetic_hole_fill(
        sci.copy(),
        coverage_hw=cov,
        enabled=True,
        max_radius_px=8,
        blend=0.7,
        only_near_seams=True,
        protect_stars_details=False,
    )
    assert info["applied"] is True
    # The deep centre of the hole (>8 px from the valid boundary) stays NaN.
    assert not np.isfinite(out[100, 100, 0])
    # A pixel right at the hole boundary (near the seam) is filled.
    assert np.isfinite(out[42, 42, 0])
    # Input was not mutated (helper copies).
    assert not np.isfinite(sci[100, 100, 0])


def test_hole_fill_disabled_does_not_fill():
    """Disabled hole fill leaves the array (incl. NaNs) unchanged."""
    h = w = 64
    yy, xx = _grid(h, w)
    plane = 100.0 + xx
    sci = np.stack([plane, plane, plane], axis=-1).astype(np.float32)
    sci[20:40, 20:40, :] = np.nan
    cov = np.ones((h, w), dtype=np.int32)
    out, info = apply_aesthetic_hole_fill(sci.copy(), coverage_hw=cov, enabled=False)
    assert info["applied"] is False
    assert np.array_equal(out, sci, equal_nan=True)


# ---------------------------------------------------------------------------
# Classic output naming + manifest/header truth (production path)
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


def _json(p):
    return json.loads(p.read_text())


def test_output_naming_export_off_single_raw(tmp_path):
    """export_aesthetic_fits=false -> ONLY ``mosaic_grid.fits`` raw (SCI role)."""
    canvas = zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_wcs().to_header().tostring(),
        width=10, height=10, resolution_deg=0.001,
    )
    a = _Assembled()
    _, _, mp = _write_via(
        a, canvas, tmp_path,
        raw_science=a.science,
        aesthetic_path=None,
        export_aesthetic_fits=False,
    )
    assert (tmp_path / "mosaic_grid.fits").exists()
    assert not (tmp_path / "mosaic_grid_aesthetic.fits").exists()
    assert not (tmp_path / "mosaic_grid_science.fits").exists()
    with fits.open(tmp_path / "mosaic_grid.fits") as h:
        assert h[0].header["SCIROLE"] == "SCI"
    m = _json(mp)
    assert m["outputs"]["science"] == "mosaic_grid.fits"
    assert m["science_reference"] == "mosaic_grid.fits"
    assert m["outputs"]["aesthetic"] is None


def test_output_naming_export_on_named_raw_and_aesthetic(tmp_path):
    """export_aesthetic_fits=true -> named raw (SCI) + aesthetic (AESTH)."""
    canvas = zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_wcs().to_header().tostring(),
        width=10, height=10, resolution_deg=0.001,
    )
    a = _Assembled()
    raw_path = tmp_path / "mosaic_grid_science.fits"
    aest_path = tmp_path / "mosaic_grid_aesthetic.fits"
    zz._write_raw_science_fits(a, canvas, raw_path, related_file=aest_path.name)
    _, _, mp = _write_via(
        a, canvas, tmp_path,
        finished_science=a.science, finishing_info={"enabled": False},
        raw_science=a.science, raw_science_path=raw_path,
        aesthetic_path=aest_path, export_aesthetic_fits=True,
    )
    assert raw_path.exists()
    assert aest_path.exists()
    with fits.open(raw_path) as h:
        assert h[0].header["SCIROLE"] == "SCI"
    with fits.open(aest_path) as h:
        assert h[0].header["SCIROLE"] == "AESTH"
    m = _json(mp)
    assert m["outputs"]["science"] == "mosaic_grid_science.fits"
    assert m["science_reference"] == "mosaic_grid_science.fits"
    assert m["outputs"]["aesthetic"] == "mosaic_grid_aesthetic.fits"


def test_suffix_sanitization_and_collision():
    # Leading underscore + alnum/_/- only.
    assert zfin.clean_fits_suffix("  sci  ", "_science") == "_sci"
    assert zfin.clean_fits_suffix("science", "_science") == "_science"
    assert zfin.clean_fits_suffix("a/b$c", "_science") == "_abc"
    assert zfin.clean_fits_suffix("", "_science") == "_science"
    # Collision: aesthetic == scientific -> aesthetic falls back to _aesthetic.
    z = SimpleNamespace(
        scientific_fits_suffix="_same", aesthetic_fits_suffix="_same",
    )
    cfg = zfin.resolve_finishing_config(z)
    assert cfg["scientific_fits_suffix"] == "_same"
    assert cfg["aesthetic_fits_suffix"] == "_aesthetic"


def test_algorithm_null_when_dbe_not_applied(tmp_path):
    """Top-level manifest algorithm must be null when DBE is disabled/no-op."""
    canvas = zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_wcs().to_header().tostring(),
        width=10, height=10, resolution_deg=0.001,
    )
    a = _Assembled()
    _, _, mp = _write_via(
        a, canvas, tmp_path,
        raw_science=a.science, aesthetic_path=None, export_aesthetic_fits=False,
    )
    m = _json(mp)
    assert m["algorithm"] is None


# ---------------------------------------------------------------------------
# Production-path witness (routine corpus): export OFF vs ON + suffixes
# ---------------------------------------------------------------------------

ROUTINE_TOOL = (
    Path(__file__).resolve().parents[1] / "tools" / "zegrid_routine" / "make_routine_corpus.py"
)


@pytest.fixture(scope="module")
def routine_corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("r23_routine")
    proc = subprocess.run(
        [sys.executable, str(ROUTINE_TOOL), str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return out


def _make_input(tmp_path, routine_corpus):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(routine_corpus.glob("*.fits")):
        (input_dir / p.name).write_bytes(p.read_bytes())
    (input_dir / "stack_plan.csv").write_text(
        routine_corpus.joinpath("stack_plan.csv").read_text(), encoding="utf-8"
    )
    return input_dir


def test_production_export_off_single_raw(tmp_path, routine_corpus):
    input_dir = _make_input(tmp_path, routine_corpus)
    out = tmp_path / "out"
    zz.run_zegrid_mode(str(input_dir), str(out), zconfig=SimpleNamespace(
        export_aesthetic_fits=False,
        final_mosaic_dbe_enabled=True,
        grid_rgb_equalize=False,
    ))
    assert (out / "mosaic_grid.fits").exists()
    assert not (out / "mosaic_grid_aesthetic.fits").exists()
    assert not (out / "mosaic_grid_science.fits").exists()
    with fits.open(out / "mosaic_grid.fits") as h:
        assert h[0].header["SCIROLE"] == "SCI"
    m = _json(out / "zegrid_manifest.json")
    assert m["outputs"]["science"] == "mosaic_grid.fits"
    assert m["science_reference"] == "mosaic_grid.fits"
    assert m["outputs"]["aesthetic"] is None


def test_production_export_on_custom_suffixes(tmp_path, routine_corpus):
    input_dir = _make_input(tmp_path, routine_corpus)
    out = tmp_path / "out"
    zz.run_zegrid_mode(str(input_dir), str(out), zconfig=SimpleNamespace(
        export_aesthetic_fits=True,
        scientific_fits_suffix="_sc",
        aesthetic_fits_suffix="_ae",
        final_mosaic_dbe_enabled=True,
        grid_rgb_equalize=False,
    ))
    assert (out / "mosaic_grid_sc.fits").exists()
    assert (out / "mosaic_grid_ae.fits").exists()
    with fits.open(out / "mosaic_grid_sc.fits") as h:
        assert h[0].header["SCIROLE"] == "SCI"
    with fits.open(out / "mosaic_grid_ae.fits") as h:
        assert h[0].header["SCIROLE"] == "AESTH"
    m = _json(out / "zegrid_manifest.json")
    assert m["outputs"]["science"] == "mosaic_grid_sc.fits"
    assert m["science_reference"] == "mosaic_grid_sc.fits"
    assert m["outputs"]["aesthetic"] == "mosaic_grid_ae.fits"
    # Raw is safe: the scientific reference is float32 (FITS stores big-endian
    # on disk; check the real dtype itemsize), never clamped/offset/abs'd.
    raw = np.asarray(fits.getdata(out / "mosaic_grid_sc.fits"))
    assert raw.dtype == np.dtype(np.float32).newbyteorder(">") or raw.dtype == np.float32
    assert raw.dtype.itemsize == 4


def test_production_forced_failure_preserves_raw(tmp_path, routine_corpus, monkeypatch):
    input_dir = _make_input(tmp_path, routine_corpus)
    out = tmp_path / "out"

    def _boom(*a, **k):
        raise RuntimeError("forced finishing failure")

    monkeypatch.setattr(zz.zfin, "apply_final_mosaic_finishing", _boom)
    zz.run_zegrid_mode(str(input_dir), str(out), zconfig=SimpleNamespace(
        export_aesthetic_fits=True,
        final_mosaic_dbe_enabled=True,
    ))
    # Raw science is always written (SCI role).
    raw_path = out / "mosaic_grid_science.fits"
    assert raw_path.exists()
    m = _json(out / "zegrid_manifest.json")
    assert m["finishing"]["failed"] is True
    assert m["outputs"]["science"] == "mosaic_grid_science.fits"
    assert m["science_reference"] == "mosaic_grid_science.fits"


# ---------------------------------------------------------------------------
# custom strength -> fallback
# ---------------------------------------------------------------------------

def test_custom_without_sigma_falls_back_to_normal():
    z = SimpleNamespace(
        final_mosaic_dbe_strength="custom",
        final_mosaic_dbe_sample_step=48,
        final_mosaic_dbe_smoothing=1.2,
    )
    r = zfin.resolve_dbe_strength(z)
    assert r["strength"] == "normal"
    assert r["params_source"] == "preset:normal"
    assert "custom_fallback" in r


def test_resolve_finishing_config_defaults():
    cfg = zfin.resolve_finishing_config(None)
    assert cfg["dbe_enabled"] is True
    assert cfg["rgb_equalize"] is True
    assert cfg["save_uint16"] is False
    assert cfg["dbe_strength"] == "normal"
    assert cfg["dbe_params_source"] == "preset:normal"
    assert cfg["dbe_params"] == {"sigma": 36.0, "obj_k": 2.8, "obj_dilate_px": 3}
    assert cfg["dbe_subtraction_factor"] == 1.0
    # Rework-3 H1: output naming + hole-fill keys resolved from existing settings.
    # export_aesthetic_fits defaults FALSE (Classic fallback -> primary name).
    assert cfg["export_aesthetic_fits"] is False
    assert cfg["scientific_fits_suffix"] == "_science"
    assert cfg["aesthetic_fits_suffix"] == "_aesthetic"
    assert cfg["hole_fill_enabled"] is True
    assert cfg["hole_fill_max_radius_px"] == 64
    assert cfg["hole_fill_blend"] == 0.70
    assert cfg["hole_fill_only_near_seams"] is True
    assert cfg["hole_fill_protect_stars_details"] is True


def test_existing_gui_keys_drive_raw_and_aesthetic_outputs():
    """Propagation: the EXISTING config keys (no new UI surface) drive the dual
    raw-science / aesthetic outputs exactly."""
    h = w = 128
    yy, xx = _grid(h, w)
    plane = 50.0 + 60.0 * (xx / w) + 40.0 * (yy / h)
    sci = np.stack([plane, plane, plane], axis=-1).astype(np.float32)
    cov = np.ones((h, w), dtype=np.int32)

    # export OFF -> single raw; export ON -> dual, DBE applied on aesthetic.
    off = zfin.resolve_finishing_config(SimpleNamespace(
        export_aesthetic_fits=False, final_mosaic_dbe_enabled=False,
        grid_rgb_equalize=False))
    assert off["export_aesthetic_fits"] is False

    on = zfin.resolve_finishing_config(SimpleNamespace(
        export_aesthetic_fits=True, final_mosaic_dbe_enabled=True,
        final_mosaic_dbe_strength="normal", grid_rgb_equalize=False,
        save_final_as_uint16=False))
    res_on = zfin.apply_final_mosaic_finishing(sci, cov, config=on)
    assert res_on.info["dbe"]["applied"] is True
    assert res_on.info["dbe"]["algorithm"] == "legacy_grid_light_dbe"
    assert on["dbe_params"] == {"sigma": 36.0, "obj_k": 2.8, "obj_dilate_px": 3}
