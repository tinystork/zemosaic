"""ZM-ZEGRID-R53b — honest halo provenance, FITS cards and run-log witness.

The tests exercise the REAL product seams (``run_zegrid_mode`` -> ``_run_single``
-> assembly -> output writing) on small deterministic corpora. Host RAM planning
is pinned to an ample value so the assertions do not depend on the test machine's
free memory; the science/geometry/assembly code is the production code.
"""

from __future__ import annotations

import hashlib
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
    note = m["reproducibility_note"]
    assert "GLOBAL-gauge normalized" in note
    assert "layout-INDEPENDENT" in note
    assert "per-Cell gauge" not in note
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
    note = m["reproducibility_note"]
    assert "GLOBAL-gauge normalized" not in note
    assert "OPT-IN per-Cell gauge" in note
    assert "additive offsets" in note
    assert "does NOT apply" in note

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


# ---------------------------------------------------------------------------
# F-1 — mode-conditional reproducibility note
# ---------------------------------------------------------------------------

def _crafted_provenance(mode, *, feather):
    return {
        "requested_px": 8,
        "effective_px": 8,
        "source": "explicit_argument",
        "units": "output_canvas_px",
        "controls_patch_geometry": True,
        "controls_contributor_membership": True,
        "controls_gauge_patch_domain": mode == "per_cell",
        "requested_gauge_mode": mode,
        "effective_gauge_mode": mode,
        "feather_applied": bool(feather),
        "feather_px": 8 if feather else None,
        "measurement_px": 16 if feather else None,
        "note": zz.HALO_PROVENANCE_NOTE,
    }


@pytest.mark.parametrize(
    "prov_overrides,config_mode,feather_info",
    [
        # Gauge mode disagrees with the resolved science config.
        ({"effective_gauge_mode": "per_cell", "feather_applied": False,
          "feather_px": None, "measurement_px": None}, "global", None),
        # Feather claimed but no feather assembly ran.
        ({"feather_applied": True, "feather_px": 8, "measurement_px": 16},
         "global", None),
        # Feather assembly ran but provenance says it did not.
        ({"feather_applied": False, "feather_px": None, "measurement_px": None},
         "per_cell", {"halo_px": 8}),
        # Feather halo width disagrees with the effective H.
        ({"feather_applied": True, "feather_px": 8, "measurement_px": 16},
         "per_cell", {"halo_px": 16}),
        # Feather assembly present in a non-per_cell mode.
        ({"feather_applied": True, "feather_px": 8, "measurement_px": 16},
         "global", {"halo_px": 8}),
    ],
)
def test_halo_consistency_helper_rejects_each_material_contradiction(
    prov_overrides, config_mode, feather_info
):
    prov = _crafted_provenance(config_mode, feather=config_mode == "per_cell")
    prov.update(prov_overrides)
    science_config = SimpleNamespace(cell_gauge_mode=config_mode)
    with pytest.raises(RuntimeError):
        zz._validate_halo_provenance_consistency(
            prov, science_config, feather_info
        )


def _sentinel_outputs(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    names = [
        "mosaic_grid_science.fits",
        "mosaic_grid_aesthetic.fits",
        "mosaic_grid_coverage.fits",
        "mosaic_grid_uint16.fits",
        "zegrid_manifest.json",
    ]
    payload = {}
    for name in names:
        data = f"SENTINEL-{name}".encode("ascii")
        (out / name).write_bytes(data)
        payload[name] = data
    return payload


@pytest.mark.parametrize(
    "case,actual_mode,inject_info",
    [
        ("gauge_mismatch", "global", False),
        ("feather_true_no_info", "global", False),
        ("feather_false_with_info", "per_cell", False),
        ("halo_mismatch", "per_cell", True),
        ("feather_in_global", "global", True),
    ],
)
def test_contradictions_fail_before_any_current_run_output(
    tmp_path, monkeypatch, case, actual_mode, inject_info
):
    """F-2: the pre-publication guard must leave all prior outputs untouched."""
    inp = _corpus(tmp_path / case / "in", SAME_SKY)
    out = tmp_path / case / "out"
    sentinels = _sentinel_outputs(out)

    if case == "gauge_mismatch":
        crafted = _crafted_provenance("global", feather=False)
        crafted["effective_gauge_mode"] = "per_cell"
    elif case == "feather_false_with_info":
        crafted = _crafted_provenance("per_cell", feather=False)
    else:
        crafted = _crafted_provenance(actual_mode, feather=True)

    monkeypatch.setattr(zz, "_halo_provenance_for_pass", lambda *a, **k: dict(crafted))
    if inject_info:
        real_assemble = zz._assemble_cells

        def _assemble_with_bad_info(*a, **k):
            assembled, info = real_assemble(*a, **k)
            if case == "halo_mismatch":
                info = {"halo_px": 16}
            else:  # feather_in_global: global normally has no feather info
                info = {"halo_px": 8}
            return assembled, info

        monkeypatch.setattr(zz, "_assemble_cells", _assemble_with_bad_info)

    zconfig = SimpleNamespace(
        zegrid_layout="2x2", export_aesthetic_fits=True,
        final_mosaic_dbe_enabled=False,
    )
    with pytest.raises(RuntimeError):
        zz.run_zegrid_mode(
            str(inp), str(out), zconfig=zconfig, zegrid_halo_px=8,
            cell_gauge_mode=actual_mode, save_final_as_uint16=True,
            grid_rgb_equalize=False,
        )

    # No current-run FITS/manifest was created or replaced.
    for name, data in sentinels.items():
        assert (out / name).read_bytes() == data, name
    assert sorted(p.name for p in out.glob("*.fits")) == sorted(
        n for n in sentinels if n.endswith(".fits")
    )
    assert not (out / "mosaic_grid.fits").exists()


# ---------------------------------------------------------------------------
# F-3 — committed explicit-H=8 array parity with accepted R53a
# ---------------------------------------------------------------------------

# These constants were produced by running the ACCEPTED base a8fe8df on this
# exact deterministic fixture (2 x 512x512 TAN frames, seeds 0/1, pinned 2x2,
# H=8, finishing disabled). Do NOT regenerate them from current code.
_R53A_PARITY = {
    "global": (
        "410eced6618d06283d7f842079fe6c2cfd33b72a5c7e1be644d20d224b66c768",
        "ebfda7867fd779a5500a057596ca74890fbd01ca5f17804833c35d26cfe86580",
    ),
    "per_cell": (
        "7c2626e548c63ae47bc00273f2f4e6672b8d80a9459b3364fc5bab32a413dd63",
        "ebfda7867fd779a5500a057596ca74890fbd01ca5f17804833c35d26cfe86580",
    ),
}


@pytest.mark.parametrize("mode", ["global", "per_cell"])
def test_explicit_h8_array_parity_with_accepted_r53a(tmp_path, mode):
    out = _run(
        tmp_path, SAME_SKY, name=f"parity_{mode}", layout="2x2",
        export_aesthetic=False, save_uint16=False,
        zegrid_halo_px=8, cell_gauge_mode=mode,
    )
    raw = np.ascontiguousarray(fits.getdata(out / "mosaic_grid.fits"))
    cov = np.ascontiguousarray(fits.getdata(out / "mosaic_grid_coverage.fits"))
    expected_raw, expected_cov = _R53A_PARITY[mode]
    assert hashlib.sha256(raw.tobytes()).hexdigest() == expected_raw
    assert hashlib.sha256(cov.tobytes()).hexdigest() == expected_cov
    m = _manifest(out)
    assert m["zegrid_halo"]["effective_px"] == 8
    assert m["zegrid_halo"]["effective_gauge_mode"] == mode


# ---------------------------------------------------------------------------
# F-4 — real empty cell + gauge-zero-support direct seam
# ---------------------------------------------------------------------------

def test_real_sparse_layout_records_empty_cell(tmp_path):
    """A real product run with a genuinely empty Cell keeps the audit keys."""
    specs = [
        (10.0, 30.0, None), (10.5, 30.0, None), (10.0, 29.5, None),
    ]
    out = _run(
        tmp_path, specs, name="sparse", layout="3x3", export_aesthetic=False,
        save_uint16=False, zegrid_halo_px=8, cell_gauge_mode="per_cell",
    )
    m = _manifest(out)
    empties = [rec for rec in m["cells"] if rec["status"] == "empty"]
    assert empties, "expected the L-shaped fixture to leave at least one empty Cell"
    for rec in empties:
        assert rec["n_contributors"] == 0
        assert rec["n_patch_contributors"] == 0
        assert len(rec["patch_bounds"]) == 4
        assert all(isinstance(v, int) for v in rec["patch_bounds"])
        assert len(rec["patch_shape_hw"]) == 2
        assert all(isinstance(v, int) and v > 0 for v in rec["patch_shape_hw"])
    halo = m["zegrid_halo"]
    assert halo["effective_gauge_mode"] == "per_cell"
    assert halo["controls_patch_geometry"] is True
    assert halo["feather_applied"] is True
    assert halo["feather_px"] == 8
    assert halo["measurement_px"] == 16


def test_stack_cell_zero_support_record_keeps_patch_fields(tmp_path, monkeypatch):
    """Focused seam: a populated Cell whose gauge has zero support."""
    from zemosaic.core.zegrid import auto_layout as zal
    from zemosaic.core.zegrid import geometry as zg

    inp = _corpus(tmp_path / "zero" / "in", SAME_SKY)
    descs, rejected = zg.read_manifest(inp)
    assert not rejected
    canvas = zg.build_canvas(descs)
    fps = zal.project_footprints(descs, canvas)
    ctxs = zz._build_chosen_cell_contexts(fps, canvas, 2, 2, halo_px=8)
    cell, patch, mem = next(
        (cell, patch, mem)
        for (_r, _c, cell, patch, mem) in ctxs
        if mem.patch_ids
    )
    science_config = zz.ExecutorConfig().science_config()
    zz._init_stack_worker(
        descs, canvas, science_config, None, [], None, cache_workers=1
    )
    monkeypatch.setattr(
        zz, "_resolve_cell_normalization",
        lambda *a, **k: (None, "empty", "synthetic zero support"),
    )
    cache_dir = tmp_path / "zero" / "cache" / cell.cell_id
    result = zz._stack_cell((cell, patch, mem, cache_dir, "inmem", 0))
    assert result["status"] == "empty"
    rec = result["record"]
    assert rec["status"] == "empty"
    assert rec["n_contributors"] == len(mem.patch_ids) > 0
    assert rec["n_patch_contributors"] == len(mem.patch_ids)
    assert rec["patch_bounds"] == [
        patch.patch.x0, patch.patch.y0, patch.patch.x1, patch.patch.y1,
    ]
    assert rec["patch_shape_hw"] == [
        patch.patch.height, patch.patch.width,
    ]
    assert rec["gauge_status"] == "empty"
