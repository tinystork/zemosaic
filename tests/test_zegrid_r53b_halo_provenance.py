"""ZM-ZEGRID-R53b — honest halo provenance, FITS cards and run-log witness.

The tests exercise the REAL product seams (``run_zegrid_mode`` -> ``_run_single``
-> assembly -> output writing) on small deterministic corpora. Host RAM planning
is pinned to an ample value so the assertions do not depend on the test machine's
free memory; the science/geometry/assembly code is the production code.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from zemosaic import zegrid_halo as zhalo
from zemosaic import zemosaic_zegrid_mode as zz

GIB = 2 ** 30
SHAPE = (512, 512)


@pytest.fixture(autouse=True)
def _ample_memory(monkeypatch):
    # Planning/admission observe this value; the arrays here are tiny.
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 8 * GIB)


# ---------------------------------------------------------------------------
# A. Source precedence + gauge/feather matrix (no science run needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "explicit,config,req,eff,source",
    [
        (8, None, 8, 8, "explicit_argument"),
        ("16", None, 16, 16, "explicit_argument"),
        (64, 8, 64, 64, "explicit_argument"),
        (None, "32", 32, 32, "config"),
        (None, 256, 256, 256, "config"),
        (None, None, None, 64, "product_default"),
    ],
)
def test_run_halo_provenance_precedence(explicit, config, req, eff, source):
    prov = zhalo.resolve_run_halo_provenance(explicit, config)
    assert prov == {
        "requested_px": req,
        "effective_px": eff,
        "source": source,
        "units": "output_canvas_px",
    }


def test_run_halo_provenance_invalid_values_fail_loudly():
    with pytest.raises(zhalo.ZeGridHaloError):
        zhalo.resolve_run_halo_provenance(64.0, None)
    with pytest.raises(zhalo.ZeGridHaloError):
        zhalo.resolve_run_halo_provenance(None, 512)


@pytest.mark.parametrize("eff", [8, 64, 256])
@pytest.mark.parametrize(
    "requested_mode,effective_mode,feather",
    [
        ("global", "global", None),
        ("per_cell", "per_cell", {"halo_px": "eff"}),
        ("auto", "global", None),
        ("auto", "per_cell", {"halo_px": "eff"}),
    ],
)
def test_pass_provenance_matrix_has_no_contradiction(
    eff, requested_mode, effective_mode, feather
):
    base = zhalo.build_halo_provenance(eff, eff, "explicit_argument")
    ctxs = [(0, 0, None, SimpleNamespace(halo_px=eff), None)]
    finfo = None if feather is None else {"halo_px": eff}
    prov = zz._halo_provenance_for_pass(
        base,
        effective_px=eff,
        cell_ctxs=ctxs,
        requested_gauge_mode=requested_mode,
        effective_gauge_mode=effective_mode,
        feather_info=finfo,
    )
    assert prov["requested_px"] == prov["effective_px"] == eff
    assert prov["effective_gauge_mode"] == effective_mode
    assert prov["requested_gauge_mode"] == requested_mode
    assert prov["controls_patch_geometry"] is True
    assert prov["controls_contributor_membership"] is True
    assert prov["controls_gauge_patch_domain"] is (effective_mode == "per_cell")
    assert prov["feather_applied"] is (effective_mode == "per_cell")
    assert prov["feather_px"] == (eff if effective_mode == "per_cell" else None)
    assert prov["measurement_px"] == (2 * eff if effective_mode == "per_cell" else None)
    # No contradictory pair can be published.
    assert (prov["feather_px"] is None) != prov["feather_applied"]
    assert (prov["measurement_px"] is None) != (effective_mode == "per_cell")


def test_pass_provenance_rejects_contradictions():
    base = zhalo.build_halo_provenance(8, 8, "explicit_argument")
    ctxs = [(0, 0, None, SimpleNamespace(halo_px=8), None)]
    with pytest.raises(RuntimeError):
        zz._halo_provenance_for_pass(
            base, effective_px=8, cell_ctxs=ctxs,
            requested_gauge_mode="per_cell", effective_gauge_mode="per_cell",
            feather_info=None,
        )
    with pytest.raises(RuntimeError):
        zz._halo_provenance_for_pass(
            base, effective_px=8, cell_ctxs=ctxs,
            requested_gauge_mode="global", effective_gauge_mode="global",
            feather_info={"halo_px": 8},
        )
    with pytest.raises(RuntimeError):
        zz._halo_provenance_for_pass(
            base, effective_px=8, cell_ctxs=ctxs,
            requested_gauge_mode="per_cell", effective_gauge_mode="per_cell",
            feather_info={"halo_px": 16},
        )
    with pytest.raises(RuntimeError):
        zz._halo_provenance_for_pass(
            zhalo.build_halo_provenance(16, 16, "explicit_argument"),
            effective_px=8, cell_ctxs=ctxs,
            requested_gauge_mode="global", effective_gauge_mode="global",
            feather_info=None,
        )


# ---------------------------------------------------------------------------
# Real product runs
# ---------------------------------------------------------------------------

def _write_frame(path, *, ra, dec, seed=0, eqmode=None, shape=SHAPE):
    h, w = shape
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [float(ra), float(dec)]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * 0.001
    wcs.array_shape = (h, w)
    rng = np.random.default_rng(seed)
    data = (100.0 + rng.normal(0.0, 0.5, (h, w))).astype(np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.header.update(wcs.to_header(relax=True))
    if eqmode is not None:
        hdu.header["EQMODE"] = int(eqmode)
    hdu.writeto(path, overwrite=True)


def _corpus(root: Path, specs):
    root.mkdir(parents=True, exist_ok=True)
    for i, (ra, dec, eqmode) in enumerate(specs):
        _write_frame(root / f"f{i:02d}.fits", ra=ra, dec=dec, seed=i, eqmode=eqmode)
    return root


def _run(tmp_path, specs, *, name="run", layout="2x2", export_aesthetic=False,
         save_uint16=False, **kwargs):
    inp = _corpus(tmp_path / name / "in", specs)
    out = tmp_path / name / "out"
    zconfig = SimpleNamespace(
        zegrid_layout=layout,
        export_aesthetic_fits=export_aesthetic,
        final_mosaic_dbe_enabled=False,
    )
    zz.run_zegrid_mode(
        str(inp), str(out), zconfig=zconfig,
        save_final_as_uint16=save_uint16,
        grid_rgb_equalize=False,
        **kwargs,
    )
    return out


def _manifest(out):
    return json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))


def _assert_cards(path, *, halo, feather, mode):
    header = fits.getheader(path)
    assert header["ZGHALO"] == int(halo)
    assert bool(header["ZGFEATH"]) is bool(feather)
    assert header["ZGGMODE"] == mode


def _assert_cell_audit(manifest):
    assert manifest["cells"], "expected at least one Cell record"
    for rec in manifest["cells"]:
        assert rec["n_patch_contributors"] == rec["n_contributors"]
        assert len(rec["patch_bounds"]) == 4
        assert all(isinstance(v, int) for v in rec["patch_bounds"])
        assert len(rec["patch_shape_hw"]) == 2
        assert all(isinstance(v, int) and v > 0 for v in rec["patch_shape_hw"])


SAME_SKY = [(10.0, 30.0, None), (10.005, 30.0, None)]


def test_real_global_run_outputs_cards_manifest_and_log(tmp_path):
    out = _run(
        tmp_path, SAME_SKY, name="global", export_aesthetic=True,
        save_uint16=True, zegrid_halo_px=8, cell_gauge_mode="global",
    )
    m = _manifest(out)
    halo = m["zegrid_halo"]
    assert halo["requested_px"] == 8
    assert halo["effective_px"] == 8
    assert halo["source"] == "explicit_argument"
    assert halo["units"] == "output_canvas_px"
    assert halo["requested_gauge_mode"] == "global"
    assert halo["effective_gauge_mode"] == "global"
    assert halo["controls_patch_geometry"] is True
    assert halo["controls_contributor_membership"] is True
    assert halo["controls_gauge_patch_domain"] is False
    assert halo["feather_applied"] is False
    assert halo["feather_px"] is None
    assert halo["measurement_px"] is None
    assert "does NOT remove gradients" in halo["note"]
    assert m["cell_feather"] == {"enabled": False}
    _assert_cell_audit(m)
    assert m["cache"]["reuse_identity"]["bound_to"] == [
        "patch_shape_hw", "patch_bounds", "patch_wcs",
    ]
    assert "geometry-bound" in m["cache"]["reuse_identity"]["note"].lower()

    raw = out / "mosaic_grid_science.fits"
    aesthetic = out / "mosaic_grid_aesthetic.fits"
    coverage = out / "mosaic_grid_coverage.fits"
    uint16 = out / "mosaic_grid_uint16.fits"
    for path in (raw, aesthetic, coverage, uint16):
        assert path.exists(), path
        _assert_cards(path, halo=8, feather=False, mode="global")

    log = (out / "zegrid_run.log").read_text(encoding="utf-8")
    lines = [ln for ln in log.splitlines() if ln.startswith("ZeGrid halo provenance:")]
    assert len(lines) == 1
    assert "requested=8 px (source=explicit_argument)" in lines[0]
    assert "effective=8 px (output_canvas_px)" in lines[0]
    assert "feather_applied=no" in lines[0]


def test_real_per_cell_run_outputs_cards_manifest_and_log(tmp_path):
    out = _run(
        tmp_path, SAME_SKY, name="percell", export_aesthetic=False,
        save_uint16=False, zegrid_halo_px=8, cell_gauge_mode="per_cell",
    )
    m = _manifest(out)
    halo = m["zegrid_halo"]
    assert halo["requested_gauge_mode"] == "per_cell"
    assert halo["effective_gauge_mode"] == "per_cell"
    assert halo["controls_gauge_patch_domain"] is True
    assert halo["feather_applied"] is True
    assert halo["feather_px"] == 8
    assert halo["measurement_px"] == 16
    assert m["cell_feather"]["enabled"] is True
    assert m["cell_feather"]["halo_px"] == 8
    _assert_cell_audit(m)

    raw = out / "mosaic_grid.fits"
    coverage = out / "mosaic_grid_coverage.fits"
    assert raw.exists() and coverage.exists()
    assert not (out / "mosaic_grid_aesthetic.fits").exists()
    assert not (out / "mosaic_grid_uint16.fits").exists()
    for path in (raw, coverage):
        _assert_cards(path, halo=8, feather=True, mode="per_cell")

    log = (out / "zegrid_run.log").read_text(encoding="utf-8")
    line = [ln for ln in log.splitlines() if ln.startswith("ZeGrid halo provenance:")][0]
    assert "gauge requested=per_cell -> effective=per_cell" in line
    assert "feather_applied=yes feather=8 px measurement=16 px" in line


def test_auto_resolves_global_and_per_cell_with_same_schema(tmp_path):
    global_out = _run(
        tmp_path, SAME_SKY, name="auto_global", zegrid_halo_px=8,
        cell_gauge_mode="auto", layout="2x2",
    )
    gm = _manifest(global_out)
    gh = gm["zegrid_halo"]
    assert gm["photometric_gauge"]["auto_selection"]["resolved"] == "global"
    assert gh["requested_gauge_mode"] == "auto"
    assert gh["effective_gauge_mode"] == "global"
    assert gh["feather_applied"] is False
    assert gh["measurement_px"] is None

    # Two disjoint groups: the global reference cannot serve every Cell.
    disjoint = [
        (10.0, 30.0, None), (10.005, 30.0, None),
        (12.0, 30.0, None), (12.005, 30.0, None),
    ]
    per_out = _run(
        tmp_path, disjoint, name="auto_percell", zegrid_halo_px=8,
        cell_gauge_mode="auto", layout="2x1",
    )
    pm = _manifest(per_out)
    ph = pm["zegrid_halo"]
    assert pm["photometric_gauge"]["auto_selection"]["resolved"] == "per_cell"
    assert pm["photometric_gauge"]["auto_selection"]["fallback_fired"] is True
    assert ph["requested_gauge_mode"] == "auto"
    assert ph["effective_gauge_mode"] == "per_cell"
    assert ph["feather_applied"] is True
    assert ph["measurement_px"] == 16
    assert pm["cell_feather"]["enabled"] is True
    assert pm["cell_feather"]["halo_px"] == 8
    # Same top-level schema shape in both outcomes.
    assert set(gm["zegrid_halo"]) == set(pm["zegrid_halo"])


def test_mount_split_both_passes_carry_identical_provenance(tmp_path):
    from zemosaic.zemosaic_stack_plan import FrameInfo, load_frame_wcs

    specs = [
        (10.0, 30.0, 1), (10.005, 30.0, 1),
        (12.0, 30.0, 0), (12.005, 30.0, 0),
    ]
    inp = _corpus(tmp_path / "mount" / "in", specs)
    out = tmp_path / "mount" / "out"
    # Mount metadata is populated from each FITS header by the production WCS
    # loader. Passing the pre-loaded frames_info mirrors the worker contract and
    # makes the EQ/ALTZ split observable before the passes start.
    frames_info = [FrameInfo(path=p, order=i) for i, p in enumerate(sorted(inp.glob("*.fits")))]
    for fi in frames_info:
        assert load_frame_wcs(fi) is True
    assert {fi.mount for fi in frames_info} == {"EQ", "ALTZ"}
    zconfig = SimpleNamespace(
        zegrid_layout="2x2", export_aesthetic_fits=False,
        final_mosaic_dbe_enabled=False,
    )
    zz.run_zegrid_mode(
        str(inp), str(out), zconfig=zconfig, zegrid_halo_px=8,
        cell_gauge_mode="global", save_final_as_uint16=False,
        grid_rgb_equalize=False, frames_info=frames_info,
    )
    manifests = {}
    for label in ("EQ", "ALTZ"):
        d = out / f"grid_{label}"
        m = _manifest(d)
        h = m["zegrid_halo"]
        assert h["requested_px"] == 8
        assert h["effective_px"] == 8
        assert h["source"] == "explicit_argument"
        assert h["effective_gauge_mode"] == "global"
        assert h["feather_applied"] is False
        _assert_cards(d / "mosaic_grid.fits", halo=8, feather=False, mode="global")
        _assert_cards(d / "mosaic_grid_coverage.fits", halo=8, feather=False, mode="global")
        manifests[label] = h
    assert manifests["EQ"] == manifests["ALTZ"]
