"""ZM-ZEGRID-R18 targeted tests — final-mosaic finishing (DBE + RGB equalize + uint16).

Covers (fast tier, no gated corpora):

* disabled -> bit-equal (identity + byte-identical science FITS);
* the documented strength mapping;
* the measured background flattening on a synthetic M16-like case (smooth
  gradient + corner vignette), with HONEST pinned numbers;
* object protection (a bright source's flux above background is preserved);
* the strength factor actually scales the subtraction;
* ``smoothing`` is honoured (changes the background model);
* ``grid_rgb_equalize`` is honoured (per-channel sky medians aligned);
* ``save_final_as_uint16`` is honoured (uint16 array + manifest/header scaling);
* a forced finishing failure WARNS and lets the run complete (fail-safe);
* the ignored-lists were updated (now-honoured keys removed).

The routine-corpus end-to-end checks reuse ``tools/zegrid_routine`` so the full
production ``run_zegrid_mode`` path is exercised in the fast tier.
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

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import final_mosaic_finishing as zfin
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import instrumentation as zin

ROUTINE_TOOL = (
    Path(__file__).resolve().parents[1]
    / "tools" / "zegrid_routine" / "make_routine_corpus.py"
)


def _synthetic_wcs(shape=(10, 10)):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 30.0]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cd = np.array([[-1, 0], [0, 1]]) * 0.001
    w.array_shape = shape
    return w


# ---------------------------------------------------------------------------
# Synthetic M16-like mosaic with a known background structure
# ---------------------------------------------------------------------------

def synthetic_mosaic(h=256, w=256):
    """Smooth linear gradient + corner vignette + one bright source.

    Background = 50 + 60*(x/W) + 40*(y/H) + 25*exp(-((x-0.05W)^2+(y-0.05H)^2)/(2*40^2)).
    Channel offsets +10 / 0 / -10 (for the RGB equalization check). A bright
    Gaussian source (peak 5000) sits at the centre so object protection can be
    measured. Deterministic (no RNG).
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = (
        50.0
        + 60.0 * (xx / w)
        + 40.0 * (yy / h)
        + 25.0 * np.exp(-((xx - 0.05 * w) ** 2 + (yy - 0.05 * h) ** 2) / (2.0 * 40.0**2))
    )
    star = 5000.0 * np.exp(-((xx - 128) ** 2 + (yy - 128) ** 2) / (2.0 * 3.0**2))
    sci = np.stack(
        [bg + 10.0 + star, bg + 0.0 + star, bg - 10.0 + star], axis=-1
    ).astype(np.float32)
    cov = np.ones((h, w), dtype=np.int32)
    return sci, cov


def _sky_box_std(a, n=8):
    """std of the per-sky-box medians (background-uniformity metric)."""
    h, w = a.shape[:2]
    vals = []
    for i in range(n):
        for j in range(n):
            b = a[i * h // n : (i + 1) * h // n, j * w // n : (j + 1) * w // n, 0]
            vals.append(float(np.median(b[np.isfinite(b)])))
    return float(np.std(vals))


def _star_flux(a, cy=128, cx=128, r_ap=6, r_in=8, r_out=14):
    """Aperture flux minus the local annulus background (object-preservation)."""
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ap = r <= r_ap
    ann = (r >= r_in) & (r <= r_out)
    bg = float(np.median(a[..., 0][ann]))
    return float(np.sum(a[..., 0][ap] - bg))


def _dbe_config(**over):
    cfg = dict(
        dbe_enabled=True,
        dbe_strength="normal",
        dbe_params_source="preset:normal",
        dbe_subtraction_factor=1.0,
        dbe_params=dict(obj_k=3.0, obj_dilate_px=3, sample_step=24, smoothing=0.6),
        rgb_equalize=False,
        save_uint16=False,
    )
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------------------
# Strength semantics (parameter presets, NOT subtraction multipliers)
# ---------------------------------------------------------------------------

def test_strength_presets_and_custom():
    # weak / normal / strong map to the documented parameter presets exactly.
    assert zfin.DBE_STRENGTH_PRESETS["weak"] == {
        "obj_k": 4.0, "obj_dilate_px": 2, "sample_step": 32, "smoothing": 1.0}
    assert zfin.DBE_STRENGTH_PRESETS["normal"] == {
        "obj_k": 3.0, "obj_dilate_px": 3, "sample_step": 24, "smoothing": 0.6}
    assert zfin.DBE_STRENGTH_PRESETS["strong"] == {
        "obj_k": 2.2, "obj_dilate_px": 4, "sample_step": 16, "smoothing": 0.25}

    r = zfin.resolve_dbe_strength(SimpleNamespace(final_mosaic_dbe_strength="strong"))
    assert r["strength"] == "strong"
    assert r["params_source"] == "preset:strong"
    assert r["params"]["sample_step"] == 16


def test_strength_invalid_falls_back_to_normal():
    for bad in ("bogus", "off", "low", "high", "", None):
        z = SimpleNamespace(final_mosaic_dbe_strength=bad)
        r = zfin.resolve_dbe_strength(z)
        assert r["strength"] == "normal"
        assert r["params_source"] == "preset:normal"
        assert r["params"] == zfin.DBE_STRENGTH_PRESETS["normal"]


def test_strength_custom_reads_explicit_config():
    z = SimpleNamespace(
        final_mosaic_dbe_strength="custom",
        final_mosaic_dbe_obj_k=4.5,
        final_mosaic_dbe_obj_dilate_px=5,
        final_mosaic_dbe_sample_step=48,
        final_mosaic_dbe_smoothing=1.2,
    )
    r = zfin.resolve_dbe_strength(z)
    assert r["strength"] == "custom"
    assert r["params_source"] == "custom_cfg"
    assert r["params"]["obj_k"] == 4.5
    assert r["params"]["obj_dilate_px"] == 5
    assert r["params"]["sample_step"] == 48
    assert r["params"]["smoothing"] == 1.2


def test_strength_presets_produce_monotonic_correction():
    """weak/normal/strong must change the correction via parameter presets, not a
    scalar subtraction amplitude. On a smooth gradient the finer/smoother preset
    (strong) removes more background variation than the coarser (weak)."""
    sci, cov = synthetic_mosaic()

    def removed(strength):
        out = zfin.apply_final_mosaic_finishing(
            sci, cov, config=_dbe_config(dbe_strength=strength,
                                         dbe_params_source=f"preset:{strength}",
                                         dbe_params=zfin.DBE_STRENGTH_PRESETS[strength]),
        ).science
        return float(np.mean(np.abs(sci[..., 0] - out[..., 0])))

    r_weak = removed("weak")
    r_normal = removed("normal")
    r_strong = removed("strong")
    assert r_weak < r_normal < r_strong


def test_smoothing_honoured():
    sci, cov = synthetic_mosaic()
    ch = sci[..., 0]
    valid = cov > 0
    _, i0 = zfin.estimate_background_channel(
        ch, valid, sample_step=24, obj_k=3.0, obj_dilate_px=3, smoothing=0.0
    )
    _, i5 = zfin.estimate_background_channel(
        ch, valid, sample_step=24, obj_k=3.0, obj_dilate_px=3, smoothing=5.0
    )
    # Stronger smoothing -> a flatter (lower-std) background model.
    assert i5["grid_std"] < i0["grid_std"]
    assert i5["grid_std"] < 0.5 * i0["grid_std"]


# ---------------------------------------------------------------------------
# DBE: measured flattening + object protection (honest pinned numbers)
# ---------------------------------------------------------------------------

def test_dbe_flattens_background_pinned():
    sci, cov = synthetic_mosaic()
    before = _sky_box_std(sci)
    out = zfin.apply_final_mosaic_finishing(sci, cov, config=_dbe_config()).science
    after = _sky_box_std(out)

    # Honest numbers for the synthetic M16-like case (256x256, gradient+vignette),
    # R23 variation-only correction (preserves global sky/DC; no full subtraction).
    assert before == pytest.approx(18.41, abs=0.5)
    assert after == pytest.approx(7.03, abs=0.6)
    assert after < 0.45 * before


def test_dbe_preserves_bright_source():
    sci, cov = synthetic_mosaic()
    out = zfin.apply_final_mosaic_finishing(sci, cov, config=_dbe_config()).science
    flux_before = _star_flux(sci)
    flux_after = _star_flux(out)
    # Object protection: the source's flux above background is preserved (the
    # star is masked from the background estimate, so it is not subtracted out).
    assert flux_after == pytest.approx(flux_before, rel=0.02)
    # The source remains clearly above the (now-flattened) background.
    assert float(np.max(out[..., 0])) > 4000.0


def test_dbe_applied_flag():
    sci, cov = synthetic_mosaic()
    res = zfin.apply_final_mosaic_finishing(sci, cov, config=_dbe_config())
    assert res.info["dbe"]["applied"] is True
    assert res.info["dbe"]["strength"] == "normal"
    assert res.info["dbe"]["params_source"] == "preset:normal"
    assert res.info["dbe"]["subtraction_factor"] == 1.0
    assert res.info["dbe"]["params"]["sample_step"] == 24


# ---------------------------------------------------------------------------
# RGB equalization
# ---------------------------------------------------------------------------

def test_grid_rgb_equalize_honoured():
    sci, cov = synthetic_mosaic()
    res = zfin.apply_final_mosaic_finishing(
        sci, cov,
        config=dict(dbe_enabled=False, rgb_equalize=True, save_uint16=False),
    )
    info = res.info["rgb_equalize"]
    before = info["skies_before"]
    after = info["skies_after"]
    # Channel offsets (+10 / 0 / -10) are visible before, aligned after.
    assert before[0] - before[2] == pytest.approx(20.0, abs=0.5)
    assert max(after) - min(after) < 0.5
    assert info["applied"] is True


def test_grid_rgb_equalize_disabled_is_noop():
    sci, cov = synthetic_mosaic()
    res = zfin.apply_final_mosaic_finishing(
        sci, cov,
        config=dict(dbe_enabled=False, rgb_equalize=False, save_uint16=False),
    )
    assert res.science is sci  # identity (bit-equal)


# ---------------------------------------------------------------------------
# uint16
# ---------------------------------------------------------------------------

def test_uint16_honoured():
    sci, cov = synthetic_mosaic()
    res = zfin.apply_final_mosaic_finishing(
        sci, cov,
        config=dict(dbe_enabled=False, rgb_equalize=False, save_uint16=True),
    )
    u16 = res.uint16
    assert u16 is not None
    assert u16.dtype == np.uint16
    assert u16.shape == sci.shape
    assert int(u16.min()) >= 0 and int(u16.max()) <= 65535
    uinfo = res.info["uint16"]
    assert uinfo["written"] is True
    assert uinfo["vmin"] < uinfo["vmax"]
    # scaling is documented/recorded
    assert "formula" in uinfo


# ---------------------------------------------------------------------------
# Bit-equal when disabled
# ---------------------------------------------------------------------------

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


def test_disabled_output_bit_equal(tmp_path):
    canvas = zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_wcs().to_header().tostring(),
        width=10, height=10, resolution_deg=0.001,
    )

    class _Assembled:
        science = np.linspace(0, 99, 300, dtype=np.float32).reshape(10, 10, 3)
        stack_depth = np.ones((10, 10), dtype=np.int32)
        complete_cells = 1
        incomplete_cells = 0
        hole_pixels = 0
        coverage_pixels = 100

    a = _Assembled()
    (tmp_path / "raw").mkdir()
    (tmp_path / "finished_off").mkdir()
    # Pre-change path (no finishing keyword at all).
    _, _, _ = _write_via(a, canvas, tmp_path / "raw")
    raw_bytes = (tmp_path / "raw" / "mosaic_grid.fits").read_bytes()

    # Disabled finishing path: finished_science is the SAME array.
    sci, cov = a.science, a.stack_depth
    res = zfin.apply_final_mosaic_finishing(
        sci, cov,
        config=dict(dbe_enabled=False, rgb_equalize=False, save_uint16=False),
    )
    assert res.science is sci
    _, _, _ = _write_via(
        a, canvas, tmp_path / "finished_off",
        finished_science=res.science, finishing_info=res.info, fin_uint16=None,
    )
    off_bytes = (tmp_path / "finished_off" / "mosaic_grid.fits").read_bytes()
    assert raw_bytes == off_bytes, "disabled finishing must be bit-equal"
    # No uint16 file when disabled.
    assert not (tmp_path / "finished_off" / "mosaic_grid_uint16.fits").exists()


def test_uint16_output_file_and_manifest(tmp_path):
    canvas = zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_wcs().to_header().tostring(),
        width=10, height=10, resolution_deg=0.001,
    )

    class _Assembled:
        science = np.linspace(0, 99, 300, dtype=np.float32).reshape(10, 10, 3)
        stack_depth = np.ones((10, 10), dtype=np.int32)
        complete_cells = 1
        incomplete_cells = 0
        hole_pixels = 0
        coverage_pixels = 100

    a = _Assembled()
    res = zfin.apply_final_mosaic_finishing(
        a.science, a.stack_depth,
        config=dict(dbe_enabled=False, rgb_equalize=False, save_uint16=True),
    )
    _, _, mp = _write_via(
        a, canvas, tmp_path,
        finished_science=res.science, finishing_info=res.info, fin_uint16=res.uint16,
    )
    assert (tmp_path / "mosaic_grid_uint16.fits").exists()
    with fits.open(tmp_path / "mosaic_grid_uint16.fits") as hdul:
        data = hdul[0].data
    assert data.dtype == np.uint16
    assert data.shape[0] == 3
    m = json.loads(mp.read_text())
    assert m["finishing"]["uint16"]["written"] is True
    assert m["outputs"]["uint16"] == "mosaic_grid_uint16.fits"


# ---------------------------------------------------------------------------
# Ignored-list updates
# ---------------------------------------------------------------------------

def test_ignored_lists_updated():
    # final_mosaic_dbe_* is no longer an ignored setting prefix.
    assert "final_mosaic_dbe_" not in zin.IGNORED_SETTING_PREFIXES
    assert zin.IGNORED_SETTING_PREFIXES == ()
    # save_final_as_uint16 / grid_rgb_equalize are no longer ignored run args.
    assert "save_final_as_uint16" not in zin.IGNORED_RUN_ARGS
    assert "grid_rgb_equalize" not in zin.IGNORED_RUN_ARGS
    # ZM-ZEGRID-R22: use_gpu is now honoured (GPU backend resolution).
    assert "use_gpu" not in zin.IGNORED_RUN_ARGS
    # Still-ignored args remain surfaced.
    assert "stack_weight_method" in zin.IGNORED_RUN_ARGS
    assert "legacy_rgb_cube" in zin.IGNORED_RUN_ARGS


def test_resolve_finishing_config_defaults_and_flags():
    # Defaults (no config): DBE on (product default), equalize on, uint16 off.
    cfg = zfin.resolve_finishing_config(None)
    assert cfg["dbe_enabled"] is True
    assert cfg["rgb_equalize"] is True
    assert cfg["save_uint16"] is False
    assert cfg["dbe_strength"] == "normal"
    assert cfg["dbe_params_source"] == "preset:normal"
    assert cfg["dbe_subtraction_factor"] == 1.0

    # Explicit off flags.
    z = SimpleNamespace(
        final_mosaic_dbe_enabled=False,
        grid_rgb_equalize=False,
        save_final_as_uint16=True,
        final_mosaic_dbe_strength="strong",
    )
    cfg = zfin.resolve_finishing_config(z)
    assert cfg["dbe_enabled"] is False
    assert cfg["rgb_equalize"] is False
    assert cfg["save_uint16"] is True
    assert cfg["dbe_strength"] == "strong"
    assert cfg["dbe_params_source"] == "preset:strong"
    assert cfg["dbe_subtraction_factor"] == 1.0  # no hidden scalar multiplier

    # ``custom`` strength reads the explicit numeric config fields.
    z2 = SimpleNamespace(
        final_mosaic_dbe_strength="custom",
        final_mosaic_dbe_sample_step=48,
        final_mosaic_dbe_smoothing=1.2,
        final_mosaic_dbe_obj_k=4.0,
        final_mosaic_dbe_obj_dilate_px=5,
    )
    p = zfin.resolve_finishing_config(z2)["dbe_params"]
    assert p["sample_step"] == 48
    assert p["smoothing"] == 1.2
    assert p["obj_k"] == 4.0
    assert p["obj_dilate_px"] == 5
    assert zfin.resolve_finishing_config(z2)["dbe_params_source"] == "custom_cfg"


# ---------------------------------------------------------------------------
# End-to-end (routine corpus): finishing applied + fail-safe
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def routine_corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("r18_routine")
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


def test_end_to_end_finishing_applied(tmp_path, routine_corpus):
    input_dir = _make_input(tmp_path, routine_corpus)
    out = tmp_path / "out"
    zconfig = SimpleNamespace(
        final_mosaic_dbe_enabled=True,
        final_mosaic_dbe_strength="normal",
        grid_rgb_equalize=True,
        save_final_as_uint16=True,
    )
    zz.run_zegrid_mode(str(input_dir), str(out), zconfig=zconfig)

    manifest = json.loads((out / "zegrid_manifest.json").read_text())
    fin = manifest["finishing"]
    assert fin["enabled"] is True
    assert fin["failed"] is False
    assert fin["dbe"]["applied"] is True
    assert fin["rgb_equalize"]["applied"] is True
    assert fin["uint16"]["written"] is True
    assert (out / "mosaic_grid_uint16.fits").exists()


def test_forced_finishing_failure_warns_and_completes(tmp_path, routine_corpus, monkeypatch):
    import logging

    input_dir = _make_input(tmp_path, routine_corpus)
    out = tmp_path / "out"

    def _boom(*a, **k):
        raise RuntimeError("forced finishing failure")

    monkeypatch.setattr(zz.zfin, "apply_final_mosaic_finishing", _boom)

    # Capture the WARN via a dedicated handler on the exact emitting logger
    # (robust against the suite's shared-root-logger state, unlike ``caplog``).
    captured = []

    class _Cap(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    cap = _Cap(level=logging.WARNING)
    lg = logging.getLogger("ZeMosaicWorker.zegrid_mode")
    old_level = lg.level
    lg.addHandler(cap)
    lg.setLevel(logging.WARNING)
    try:
        # Must NOT raise: the finishing failure WARNS and the run completes.
        zz.run_zegrid_mode(
            str(input_dir), str(out),
            zconfig=SimpleNamespace(final_mosaic_dbe_enabled=True),
        )
    finally:
        lg.removeHandler(cap)
        lg.setLevel(old_level)

    # Raw mosaic still produced (fail-safe).
    assert (out / "mosaic_grid.fits").exists()
    manifest = json.loads((out / "zegrid_manifest.json").read_text())
    assert manifest["finishing"]["failed"] is True
    assert "forced finishing failure" in manifest["finishing"]["failure_reason"]
    # A WARN was emitted (dedicated handler) and the durable run log records it.
    assert any("finishing FAILED" in m for m in captured)
    run_log = (out / zz.RUN_LOG_NAME).read_text()
    assert "failed: True" in run_log
