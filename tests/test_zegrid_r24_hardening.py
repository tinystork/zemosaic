"""ZM-ZEGRID-R24 post-acceptance LOW hardening tests (A1-A5).

Targeted, fast-tier witnesses for the six LOW hardenings recorded after R23
acceptance. Each item below maps to the mission's A1-A5 (A6 is documentation /
maintainability, covered by the source docstrings/comments rather than a test).

A1  Manifest absence truth: when no aesthetic FITS is emitted, the
    ``science_output_contract.aesthetic`` sub-record is OMITTED entirely (no
    ``file=None`` + in-memory SHA); ``outputs.aesthetic`` stays an explicit
    ``null``. When emitted, the exact file/role/dtype/DBE/hash truth is retained.

A2  Atomic temp cleanup + conservative stale aesthetic cleanup: ``_atomic_writeto``
    leaves no ``<target>.tmp`` on success or on any write/replace failure, and a
    pre-existing valid target survives a failed temp write; a prior
    manifest-declared stale aesthetic is removed (never glob-deleted, never
    science/coverage/uint16/user files, never outside ``output_dir``).

A3  Classic regression witness: ``zemosaic_worker._apply_aesthetic_hole_fill``
    with ``only_near_seams=True`` fills target-edge holes, leaves deep/non-target
    NaNs unchanged, and never mutates its input.

A4  Robust bool coercion: Classic-compatible coercion (real bool / 0/1 / typed
    strings / empty / None); unknown strings fall back (never silently True).

A5  No cosmetic empty-hole warning: fully-empty RGB holes no longer emit
    ``RuntimeWarning: Mean of empty slice``; finite-pixel luminance semantics are
    preserved.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import final_mosaic_finishing as zfin
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid.aesthetic_hole_fill import apply_aesthetic_hole_fill


# ---------------------------------------------------------------------------
# Shared production-path helpers (mirrors the R23 fast-tier setup)
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


def _canvas():
    return zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_wcs().to_header().tostring(),
        width=10, height=10, resolution_deg=0.001,
    )


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
    return json.loads(Path(p).read_text())


# ---------------------------------------------------------------------------
# A1 — manifest absence truth
# ---------------------------------------------------------------------------

def test_a1_contract_omits_aesthetic_when_absent(tmp_path):
    canvas = _canvas()
    a = _Assembled()
    _, _, mp = _write_via(
        a, canvas, tmp_path,
        raw_science=a.science,
        aesthetic_path=None,
        export_aesthetic_fits=False,
    )
    m = _json(mp)
    # Stable schema: top-level outputs.aesthetic stays explicit null.
    assert m["outputs"]["aesthetic"] is None
    # Contract truth: the aesthetic sub-record is OMITTED entirely.
    assert "aesthetic" not in m["science_output_contract"]
    assert "raw" in m["science_output_contract"]
    assert m["science_output_contract"]["raw"]["file"] == "mosaic_grid.fits"
    assert m["science_output_contract"]["raw"]["role"] == "SCI"


def test_a1_contract_retains_exact_aesthetic_truth_when_present(tmp_path):
    canvas = _canvas()
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
    m = _json(mp)
    assert m["outputs"]["aesthetic"] == "mosaic_grid_aesthetic.fits"
    c = m["science_output_contract"]["aesthetic"]
    assert c["file"] == "mosaic_grid_aesthetic.fits"
    assert c["role"] == "AESTH"
    assert c["dtype"] == "float32"
    assert c["dbe_state"] == "off"
    assert isinstance(c["sha256"], str) and len(c["sha256"]) == 64


# ---------------------------------------------------------------------------
# A2 — atomic temp cleanup + conservative stale aesthetic cleanup
# ---------------------------------------------------------------------------

class _BoomHDU:
    """FITS-HDU stand-in whose ``writeto`` writes a partial temp then raises."""

    def __init__(self, fail_on_replace=False):
        self.fail_on_replace = fail_on_replace

    def writeto(self, p, overwrite=False):
        Path(p).write_bytes(b"partial")
        if not self.fail_on_replace:
            raise RuntimeError("boom")


def test_a2_atomic_writeto_failure_cleans_tmp_and_preserves_target(tmp_path):
    target = tmp_path / "mosaic_grid.fits"
    target.write_bytes(b"VALID")
    with pytest.raises(RuntimeError):
        zz._atomic_writeto(_BoomHDU(), target)
    assert not (tmp_path / "mosaic_grid.fits.tmp").exists()
    assert target.read_bytes() == b"VALID"


def test_a2_atomic_writeto_replace_failure_cleans_tmp_and_preserves_target(tmp_path, monkeypatch):
    target = tmp_path / "mosaic_grid.fits"
    target.write_bytes(b"VALID")

    def _boom_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(zz.os, "replace", _boom_replace)
    hdu = fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(OSError):
        zz._atomic_writeto(hdu, target)
    assert not (tmp_path / "mosaic_grid.fits.tmp").exists()
    assert target.read_bytes() == b"VALID"


def test_a2_atomic_writeto_success_leaves_no_tmp(tmp_path):
    target = tmp_path / "mosaic_grid.fits"
    zz._atomic_writeto(fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32)), target)
    assert target.exists()
    assert not (tmp_path / "mosaic_grid.fits.tmp").exists()


def test_a2_stale_cleanup_on_to_off(tmp_path):
    (tmp_path / "mosaic_grid_ae.fits").write_bytes(b"old aesthetic")
    (tmp_path / "mosaic_grid.fits").write_bytes(b"science")
    (tmp_path / "zegrid_manifest.json").write_text(
        json.dumps({"outputs": {"aesthetic": "mosaic_grid_ae.fits"}})
    )
    rec = zz._cleanup_stale_aesthetic(
        tmp_path, None,
        protected_names={"mosaic_grid.fits", "mosaic_grid_coverage.fits",
                         "zegrid_run.log"},
    )
    assert rec["removed"] == ["mosaic_grid_ae.fits"]
    assert not (tmp_path / "mosaic_grid_ae.fits").exists()
    assert (tmp_path / "mosaic_grid.fits").exists()


def test_a2_stale_cleanup_suffix_changed(tmp_path):
    (tmp_path / "mosaic_grid_ae.fits").write_bytes(b"old aesthetic")
    (tmp_path / "mosaic_grid_ae2.fits").write_bytes(b"current aesthetic")
    (tmp_path / "zegrid_manifest.json").write_text(
        json.dumps({"outputs": {"aesthetic": "mosaic_grid_ae.fits"}})
    )
    rec = zz._cleanup_stale_aesthetic(
        tmp_path, tmp_path / "mosaic_grid_ae2.fits",
        protected_names={"mosaic_grid.fits", "mosaic_grid_coverage.fits",
                         "mosaic_grid_ae2.fits", "zegrid_run.log"},
    )
    assert rec["removed"] == ["mosaic_grid_ae.fits"]
    assert not (tmp_path / "mosaic_grid_ae.fits").exists()
    assert (tmp_path / "mosaic_grid_ae2.fits").exists()


def test_a2_stale_cleanup_malicious_outside_path_refused(tmp_path):
    (tmp_path / "zegrid_manifest.json").write_text(
        json.dumps({"outputs": {"aesthetic": "../evil.fits"}})
    )
    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=set())
    assert rec["removed"] == []
    assert rec["reason"] == "invalid_prior_declaration"
    assert "../evil.fits" in rec["kept"]


def test_a2_stale_cleanup_reserved_science_name_refused(tmp_path):
    (tmp_path / "mosaic_grid.fits").write_bytes(b"science")
    (tmp_path / "zegrid_manifest.json").write_text(
        json.dumps({"outputs": {"aesthetic": "mosaic_grid.fits"}})
    )
    rec = zz._cleanup_stale_aesthetic(
        tmp_path, None,
        protected_names={"mosaic_grid_coverage.fits", "zegrid_run.log"},
    )
    assert rec["removed"] == []
    assert rec["reason"] == "invalid_prior_declaration"
    assert (tmp_path / "mosaic_grid.fits").exists()


def test_a2_stale_cleanup_no_prior_manifest_noop(tmp_path):
    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=set())
    assert rec["removed"] == []
    assert rec["reason"] == "no_prior_manifest"


def _run_outputs(tmp_path, aesthetic_name=None):
    """One production ``_write_outputs`` pass; returns the manifest dict."""
    canvas = _canvas()
    a = _Assembled()
    kw = dict(raw_science=a.science, finished_science=a.science,
              finishing_info={"enabled": False})
    if aesthetic_name is None:
        # export OFF -> single raw ``mosaic_grid.fits``, no aesthetic.
        zz._write_raw_science_fits(a, canvas, tmp_path / "mosaic_grid.fits")
        _, _, mp = _write_via(a, canvas, tmp_path,
                              raw_science_path=tmp_path / "mosaic_grid.fits",
                              aesthetic_path=None, export_aesthetic_fits=False, **kw)
    else:
        raw_path = tmp_path / "mosaic_grid_science.fits"
        aest_path = tmp_path / aesthetic_name
        zz._write_raw_science_fits(a, canvas, raw_path, related_file=aest_path.name)
        _, _, mp = _write_via(a, canvas, tmp_path,
                              raw_science_path=raw_path,
                              aesthetic_path=aest_path, export_aesthetic_fits=True, **kw)
    return _json(mp)


def test_a2_production_rerun_on_to_off_cleans_stale_aesthetic(tmp_path):
    # Run 1: export ON -> named raw + aesthetic ``mosaic_grid_ae.fits``.
    m1 = _run_outputs(tmp_path, "mosaic_grid_ae.fits")
    assert (tmp_path / "mosaic_grid_ae.fits").exists()
    assert m1["outputs"]["aesthetic"] == "mosaic_grid_ae.fits"
    assert (tmp_path / "mosaic_grid_science.fits").exists()

    # Run 2: export OFF -> single raw ``mosaic_grid.fits``; prior aesthetic
    # declared by the prior manifest is removed; prior SCIENCE preserved.
    m2 = _run_outputs(tmp_path, None)
    assert not (tmp_path / "mosaic_grid_ae.fits").exists()
    assert (tmp_path / "mosaic_grid_science.fits").exists()  # prior science kept
    assert (tmp_path / "mosaic_grid.fits").exists()
    assert m2["outputs"]["aesthetic"] is None
    assert m2["stale_cleanup"]["removed"] == ["mosaic_grid_ae.fits"]
    assert "aesthetic" not in m2["science_output_contract"]


def test_a2_production_rerun_suffix_changed_cleans_stale_aesthetic(tmp_path):
    # Run 1: aesthetic suffix ``_ae``.
    _run_outputs(tmp_path, "mosaic_grid_ae.fits")
    assert (tmp_path / "mosaic_grid_ae.fits").exists()

    # Run 2: aesthetic suffix ``_ae2`` -> prior ``_ae`` removed, ``_ae2`` kept.
    m2 = _run_outputs(tmp_path, "mosaic_grid_ae2.fits")
    assert not (tmp_path / "mosaic_grid_ae.fits").exists()
    assert (tmp_path / "mosaic_grid_ae2.fits").exists()
    assert m2["stale_cleanup"]["removed"] == ["mosaic_grid_ae.fits"]


# ---------------------------------------------------------------------------
# A3 — Classic regression witness (only_near_seams target semantics)
# ---------------------------------------------------------------------------

def test_a3_classic_worker_only_near_seams_fills_edge_not_deep(tmp_path):
    from zemosaic import zemosaic_worker as zw

    h = w = 64
    mosaic = np.full((h, w, 3), 100.0, dtype=np.float32)
    # Deep central hole (36x36): edge pixels near seams, centre far from any
    # valid pixel (distance > max_radius_px).
    cy, cx = h // 2, w // 2
    r = 18
    mosaic[cy - r:cy + r, cx - r:cx + r] = np.nan
    frozen = mosaic.copy()

    out, info = zw._apply_aesthetic_hole_fill(
        mosaic.copy(),
        enabled=True,
        only_near_seams=True,
        max_radius_px=8,
        protect_stars_details=True,
    )

    assert info["applied"] is True
    assert info["filled_px"] > 0
    # Deep, non-target holes are left untouched (NaN stays NaN).
    assert info["hole_px"] > info["filled_px"]

    finite = np.isfinite(out).all(axis=-1)
    # Hole edge (within max_radius of a valid pixel) is filled.
    assert bool(finite[cy - r, cx])
    assert bool(finite[cy, cx - r])
    # Deep centre is NOT filled.
    assert not bool(finite[cy, cx])

    # Input is never mutated (helper works on a copy).
    assert np.array_equal(mosaic, frozen, equal_nan=True)


# ---------------------------------------------------------------------------
# A4 — robust bool coercion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (False, False), (True, True),
    (0, False), (1, True), (0.0, False), (2, True),
    ("false", False), ("FALSE", False), ("False", False),
    ("true", True), ("TRUE", True),
    ("no", False), ("off", False), ("disable", False), ("disabled", False),
    ("yes", True), ("on", True), ("enable", True), ("enabled", True),
    ("0", False), ("1", True),
    ("", True), ("   ", True), (None, True),
    # unknown strings fall back to the given fallback (never silently True)
    ("bogus", False),
])
def test_a4_coerce_bool_table(value, expected):
    # fallback=True for the truthy rows; for the 'bogus'/'' /None rows the
    # expected value encodes the fallback (True for ''/None, False for 'bogus').
    fallback = True
    if value == "bogus":
        fallback = False
    assert zfin._coerce_bool(value, fallback) is expected


def test_a4_unknown_string_uses_fallback_and_is_visible():
    notes = []
    r = zfin._coerce_bool("definitely-not-a-bool", False, notes=notes)
    assert r is False
    assert notes == [("definitely-not-a-bool", False)]


def test_a4_resolve_finishing_config_typed_false_strings():
    z = SimpleNamespace(
        final_mosaic_dbe_enabled="false",
        grid_rgb_equalize="off",
        save_final_as_uint16="no",
        export_aesthetic_fits="disable",
        aesthetic_hole_fill_enabled="false",
        aesthetic_hole_fill_only_near_seams="off",
        aesthetic_hole_fill_protect_stars_details="no",
    )
    cfg = zfin.resolve_finishing_config(z)
    assert cfg["dbe_enabled"] is False
    assert cfg["rgb_equalize"] is False
    assert cfg["save_uint16"] is False
    assert cfg["export_aesthetic_fits"] is False
    assert cfg["hole_fill_enabled"] is False
    assert cfg["hole_fill_only_near_seams"] is False
    assert cfg["hole_fill_protect_stars_details"] is False


def test_a4_resolve_finishing_config_run_arg_coercion():
    cfg = zfin.resolve_finishing_config(
        None, grid_rgb_equalize="false", save_final_as_uint16="true"
    )
    assert cfg["rgb_equalize"] is False
    assert cfg["save_uint16"] is True


def test_a4_resolve_finishing_config_unknown_string_visible():
    cfg = zfin.resolve_finishing_config(
        SimpleNamespace(export_aesthetic_fits="what")
    )
    assert cfg["export_aesthetic_fits"] is False  # field fallback, not True
    fields = [f["field"] for f in cfg["bool_coercion_fallbacks"]]
    assert "export_aesthetic_fits" in fields


def test_a4_resolve_finishing_config_empty_and_none_use_defaults():
    cfg = zfin.resolve_finishing_config(
        SimpleNamespace(export_aesthetic_fits="", save_final_as_uint16=" ")
    )
    # empty/whitespace -> field fallback (False for both).
    assert cfg["export_aesthetic_fits"] is False
    assert cfg["save_uint16"] is False
    # Normal default path (not flagged as a coercion fallback).
    assert cfg["bool_coercion_fallbacks"] == []


# ---------------------------------------------------------------------------
# A5 — no cosmetic empty-hole warning
# ---------------------------------------------------------------------------

def test_a5_empty_hole_does_not_emit_mean_of_empty_slice():
    mosaic = np.full((32, 32, 3), 100.0, dtype=np.float32)
    mosaic[10, 10] = np.nan  # a fully-empty RGB hole (all 3 channels NaN)
    frozen = mosaic.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out, info = apply_aesthetic_hole_fill(
            mosaic.copy(),
            enabled=True,
            only_near_seams=True,
            max_radius_px=8,
            protect_stars_details=True,
        )

    assert info["applied"] is True
    # The hole is a target (adjacent to valid pixels) and gets filled.
    assert np.isfinite(out[10, 10]).all()
    # Finite (valid) pixels preserve their luminance semantics (unchanged).
    assert np.allclose(out[0, 0], frozen[0, 0])
    # Input unmutated.
    assert np.array_equal(mosaic, frozen, equal_nan=True)
