"""ZM-ZEGRID-R38-A focused tests — bounded causal diagnostic harness.

Covers, WITHOUT any real Caldwell run:

* module-scope import audit (the tool imports no ``zemosaic`` at import time and
  importing it leaves ``zemosaic`` out of ``sys.modules``);
* strict JSON helpers (NaN/Infinity literals rejected; canonical dump);
* the source-set digest is byte-compatible with the accepted algorithm;
* capture/manifest identity validation: PASS on a linked synthetic capture,
  tamper failures (sha256 / bytes / capture id / source digest / capped holdout
  / dropped relations), and the ``any`` (generic) mode;
* source resolution refusals (missing / ambiguous basename / traversal /
  absolute / duplicate);
* panel freeze determinism + slot composition (both orientations, corner,
  declared worst edge) and the NOT_QUALIFICATION label;
* bounded execution planning (orientation coverage + budget);
* gauge cross-comparison on synthetic gauges (constant step recovered, non
  additive abstention);
* ROI metrics against a hand-computed difference;
* the read-only ROI provider over a real R6 aligned cache;
* factor-probe NOT_RUN refusals (no fabricated result);
* the pre-registered conclusion rule (PARTIAL / FALSIFIED / INCONCLUSIVE);
* end-to-end ``inventory`` on a synthetic capture: strict JSON report, no writes
  under the capture tree, deterministic repeat;
* output-safety refusals.

The synthetic FITS writes trigger the PRE-EXISTING astropy ``Card is too long``
VerifyWarning (identical to the accepted R37-D/R37-E e2e tests); filter ONLY that
warning so ``-W error`` still guards everything R38-A cares about.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

_TOOL_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "r38_causal_probe.py"

pytestmark = pytest.mark.filterwarnings(
    "ignore:Card is too long:astropy.io.fits.verify.VerifyWarning",
)


def _load_tool():
    name = "r38_causal_probe"
    spec = importlib.util.spec_from_file_location(name, _TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


probe = _load_tool()


# ---------------------------------------------------------------------------
# 1. Import audit
# ---------------------------------------------------------------------------

def test_import_audit_no_zemosaic_at_module_scope():
    audit = probe.import_audit()
    assert audit["engine_access_is_lazy"] is True
    assert audit["zemosaic_module_scope_imports"] == []
    assert "zemosaic" not in audit["module_scope_imports"]


def test_importing_tool_does_not_import_zemosaic():
    # Order-independent: load the tool by path in a FRESH interpreter and assert
    # no ``zemosaic`` module is pulled in (engine access is fully lazy).
    import subprocess
    code = (
        "import importlib.util, sys; "
        f"s=importlib.util.spec_from_file_location('r38t', r'{_TOOL_PATH}'); "
        "m=importlib.util.module_from_spec(s); sys.modules['r38t']=m; s.loader.exec_module(m); "
        "bad=[x for x in sys.modules if x=='zemosaic' or x.startswith('zemosaic.')]; "
        "print('IMPORTED' if bad else 'CLEAN')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "CLEAN"


# ---------------------------------------------------------------------------
# 2. Strict JSON + digest
# ---------------------------------------------------------------------------

def test_load_json_strict_rejects_non_finite(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"a": NaN}')
    with pytest.raises(ValueError):
        probe.load_json_strict(p)
    p.write_text('{"a": Infinity}')
    with pytest.raises(ValueError):
        probe.load_json_strict(p)
    p.write_text('{"a": 1.5, "b": [true, null]}')
    assert probe.load_json_strict(p) == {"a": 1.5, "b": [True, None]}


def test_dumps_strict_is_canonical_and_rejects_nan():
    assert probe.dumps_strict({"b": 1, "a": 2}).splitlines()[1].strip().startswith('"a"')
    with pytest.raises(ValueError):
        probe.dumps_strict({"x": float("nan")})


def test_fin_maps_non_finite_to_none():
    assert probe._fin(1.25) == 1.25
    assert probe._fin(float("nan")) is None
    assert probe._fin(float("inf")) is None
    assert probe._fin("nope") is None


def test_source_set_digest_matches_accepted_algorithm():
    ids = ["a.fit", "b.fit", "a.fit"]
    h = hashlib.sha256()
    h.update((probe.SOURCE_SET_DIGEST_ALGO + "\n").encode())
    for name in sorted({"a.fit", "b.fit"}):
        h.update(name.encode())
        h.update(b"\n")
    assert probe.source_set_digest(ids) == h.hexdigest()
    # order invariance
    assert probe.source_set_digest(["b.fit", "a.fit"]) == probe.source_set_digest(["a.fit", "b.fit"])


# ---------------------------------------------------------------------------
# Synthetic capture fixture
# ---------------------------------------------------------------------------

def _grid_relations(nx=6, ny=4, offsets=None):
    rels = []
    for r in range(ny):
        for c in range(nx):
            u = f"r{r:04d}c{c:04d}"
            if c + 1 < nx:
                rels.append((u, f"r{r:04d}c{c+1:04d}", "h"))
            if r + 1 < ny:
                rels.append((u, f"r{r+1:04d}c{c:04d}", "v"))
    out = []
    for i, (u, v, o) in enumerate(rels):
        for ch in range(3):
            val = (offsets or {}).get((u, v), 10.0 * (i + 1)) + ch
            out.append({
                "u": u, "v": v, "channel": ch, "orientation": o,
                "local_offset_train": val, "robust_sigma": 1.5,
                "n_train": 100, "rejection_fraction": 0.1, "level_mean": 1000.0,
            })
    return rels, out


def _sha256(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _make_capture(tmp_path, *, nx=6, ny=4, n_frames=6, tamper=None, keep_grid=True):
    """Build a synthetic capture dir (manifest + artifact) + source FITS dir."""
    from astropy.io import fits
    from astropy.wcs import WCS

    capture = tmp_path / "capture"
    src = tmp_path / "src"
    capture.mkdir()
    src.mkdir()

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [32.0, 32.0]
    w.wcs.cdelt = [-0.001, 0.001]
    w.wcs.crval = [10.0, 20.0]
    shape = (64, 64)
    frame_ids = []
    for i in range(n_frames):
        name = f"Light_{i:03d}.fit"
        hdr = w.to_header(relax=True)
        hdr["NAXIS"] = 2
        fits.PrimaryHDU(data=np.full(shape, 100.0 + i, dtype=np.float32), header=hdr).writeto(src / name)
        frame_ids.append(name)

    digest = probe.source_set_digest(frame_ids)
    cap_id = "c" * 64
    rels, relations = _grid_relations(nx=nx, ny=ny)
    artifact = {
        "schema": "zemosaic.zegrid.background_diagnostics",
        "schema_version": "r37c-1",
        "provenance": {
            "capture": {
                "capture_id": cap_id, "capture_stage": probe.CAPTURE_STAGE,
                "n_source_frames": n_frames, "source_set_digest": digest,
                "source_set_digest_algo": probe.SOURCE_SET_DIGEST_ALGO,
                "effective_cell_gauge_mode": "per_cell",
                "halo_px": 8, "measurement_px": 16, "feather_px": 8,
            },
            "correction_applied": False,
            "production_wired": True,
        },
        "meta": {"n_relations": len(relations), "n_dropped": 0, "nx": nx, "ny": ny,
                 "canvas_hw": [0, 0]},
        "relations": relations,
    }
    art_path = capture / probe.CAPTURE_ARTIFACT_BASENAME
    art_path.write_text(json.dumps(artifact))

    man = {
        "engine": "zegrid",
        "n_frames": n_frames,
        "frame_ids": frame_ids,
        "photometric_gauge": {"mode": "per_cell"},
        "cell_feather": {"enabled": True, "r37_diagnostic_capture": {}},
        "canvas": {"width": 0, "height": 0, "id": "x"},
        "layout": {"nx": nx, "ny": ny},
    }
    # A real geometry pass is needed for the canvas facts; do it lazily below.
    return capture, src, man, art_path, artifact


def _finalise_capture(capture, src, man, art, nx=6, ny=4):
    """Run the production geometry once to fill truthful canvas facts."""
    res = probe.resolve_sources(src, man["frame_ids"])
    geo = probe.reconstruct_geometry(res["paths"], src)
    canvas = geo["canvas"]
    man["canvas"] = {"width": canvas["width"], "height": canvas["height"], "id": canvas["id"]}
    art["meta"]["canvas_hw"] = [canvas["height"], canvas["width"]]
    art_path = capture / probe.CAPTURE_ARTIFACT_BASENAME
    art_path.write_text(json.dumps(art))
    sha = _sha256(art_path)
    man["cell_feather"]["r37_diagnostic_capture"] = {
        "status": "written",
        "filename": probe.CAPTURE_ARTIFACT_BASENAME,
        "sha256": sha,
        "bytes": art_path.stat().st_size,
        "capture_id": art["provenance"]["capture"]["capture_id"],
        "n_source_frames": man["n_frames"],
        "source_set_digest": art["provenance"]["capture"]["source_set_digest"],
        "source_set_digest_algo": probe.SOURCE_SET_DIGEST_ALGO,
        "n_relations": art["meta"]["n_relations"],
        "n_dropped": art["meta"]["n_dropped"],
        "holdout_capped": False,
        "holdout_exact": True,
        "qualification": "EXACT",
        "halo_px": 8, "measurement_px": 16, "feather_px": 8,
    }
    (capture / probe.MANIFEST_BASENAME).write_text(json.dumps(man))
    return man


def _tamper_capture(capture, man, art, key):
    art_path = capture / probe.CAPTURE_ARTIFACT_BASENAME
    if key == "artifact_sha":
        man["cell_feather"]["r37_diagnostic_capture"]["sha256"] = "deadbeef"
    elif key == "artifact_bytes":
        man["cell_feather"]["r37_diagnostic_capture"]["bytes"] += 1
    elif key == "capture_id":
        man["cell_feather"]["r37_diagnostic_capture"]["capture_id"] = "other"
    elif key == "source_digest":
        man["cell_feather"]["r37_diagnostic_capture"]["source_set_digest"] = "bad"
    elif key == "capped":
        man["cell_feather"]["r37_diagnostic_capture"]["holdout_capped"] = True
    elif key == "dropped":
        art["meta"]["n_dropped"] = 3
    elif key == "correction":
        art["provenance"]["correction_applied"] = True
    elif key == "gauge_mode":
        man["photometric_gauge"]["mode"] = "global"
    elif key == "engine":
        man["engine"] = "classic"
    (capture / probe.MANIFEST_BASENAME).write_text(json.dumps(man))
    art_path.write_text(json.dumps(art))


# ---------------------------------------------------------------------------
# 3. Identity validation
# ---------------------------------------------------------------------------

def test_identity_pass_and_generic_mode(tmp_path):
    capture, src, man, art_path, art = _make_capture(tmp_path)
    _finalise_capture(capture, src, man, art)
    got = probe.validate_identity(capture, expect="any")
    assert got["checks"]["checked_identity_matched"] is False
    assert got["checks"]["linkage"]["manifest_qualification"] == "EXACT"
    assert got["artifact"]["sha256"] == _sha256(art_path)
    assert got["computed"]["source_set_digest"] == art["provenance"]["capture"]["source_set_digest"]
    # ``checked`` mode must refuse a capture that is NOT the recorded one
    with pytest.raises(probe.IdentityError):
        probe.validate_identity(capture, expect="checked")


@pytest.mark.parametrize("key", [
    "artifact_sha", "artifact_bytes", "capture_id", "source_digest", "capped",
    "dropped", "correction", "gauge_mode", "engine",
])
def test_identity_tamper_fails(tmp_path, key):
    capture, src, man, art_path, art = _make_capture(tmp_path)
    _finalise_capture(capture, src, man, art)
    _tamper_capture(capture, man, art, key)
    with pytest.raises(probe.IdentityError):
        probe.validate_identity(capture, expect="any")


# ---------------------------------------------------------------------------
# 4. Source resolution
# ---------------------------------------------------------------------------

def _touch(dirpath, names):
    for n in names:
        p = dirpath / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def test_resolve_sources_unique_and_refusals(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    _touch(root, ["a.fit", "sub/b.fit"])
    got = probe.resolve_sources(root, ["a.fit", "sub/b.fit"])
    assert got["n_resolved"] == 2
    with pytest.raises(probe.ResolutionError):
        probe.resolve_sources(root, ["missing.fit"])
    with pytest.raises(probe.ResolutionError):
        probe.resolve_sources(root, ["../a.fit"])
    with pytest.raises(probe.ResolutionError):
        probe.resolve_sources(root, ["/abs/a.fit"])
    with pytest.raises(probe.ResolutionError):
        probe.resolve_sources(root, ["a.fit", "a.fit"])
    # ambiguous basename
    _touch(root, ["dup/c.fit", "other/c.fit"])
    with pytest.raises(probe.ResolutionError):
        probe.resolve_sources(root, ["c.fit"])


# ---------------------------------------------------------------------------
# 5. Panel freeze + planning
# ---------------------------------------------------------------------------

def _cells_for(nx=6, ny=4, n=10):
    return {f"r{r:04d}c{c:04d}": {"n_patch": n, "row": r, "col": c}
            for r in range(ny) for c in range(nx)}


def test_freeze_panel_deterministic_and_composed():
    _, relations = _grid_relations()
    artifact = {"relations": relations}
    cells = _cells_for()
    a = probe.freeze_panel(artifact, cells, 6, 4)
    b = probe.freeze_panel({"relations": list(reversed(relations))}, cells, 6, 4)
    assert a == b, "panel must be deterministic and order-invariant"
    assert a["label"] == "NOT_QUALIFICATION"
    assert len(a["relations"]) == 6
    assert set(a["orientations_covered"]) == {"h", "v"}
    assert a["corner_cells_covered"], "a canvas corner must be covered"
    assert a["frozen_before_factor_results"] is True


def test_panel_includes_declared_worst_edge_when_present():
    _, relations = _grid_relations()
    art = {"relations": relations}
    panel = probe.freeze_panel(art, _cells_for(), 6, 4)
    picked = {(x["u"], x["v"]) for x in panel["relations"]}
    assert probe.DECLARED_WORST_EDGE in picked


def test_estimate_costs_orientation_and_budget():
    _, relations = _grid_relations()
    panel = probe.freeze_panel({"relations": relations}, _cells_for(), 6, 4)
    costs = probe.estimate_costs(panel, _cells_for(), max_cell_contributors=100,
                                 max_frames_total=100)
    orients = {s["orientation"] for s in costs["selected_relations"]}
    assert orients == {"h", "v"}
    assert costs["estimated_frames"] <= 100


# ---------------------------------------------------------------------------
# 6. Gauge comparison + metrics
# ---------------------------------------------------------------------------

def _fake_gauge(cell_id, frame_ids, b_values, a=1.0):
    n = len(frame_ids)
    coef = np.zeros((n, 3, 2), dtype=np.float64)
    coef[:, :, 0] = a
    for ch in range(3):
        coef[:, ch, 1] = b_values
    return {
        "cell_id": cell_id, "status": "ok", "n_contributors": n,
        "frame_ids": list(frame_ids),
        "reference_frame_id": frame_ids[0],
        "coefficients": coef.tolist(),
        "weights": [1.0] * n,
        "norm_active": [True] * n,
        "weight_active": [True] * n,
        "excluded": [],
        "coefficients_a_values": [1.0] if a == 1.0 else [a],
    }


def test_compare_gauges_recovers_constant_step():
    ids = [f"f{i}.fit" for i in range(5)]
    gu = _fake_gauge("r0000c0000", ids, b_values=100.0)
    gv = _fake_gauge("r0000c0001", ids, b_values=40.0)
    observed = {"0": {"local_offset_train": 60.0}}
    out = probe.compare_gauges(gu, gv, observed)
    assert out["comparable"] is True
    ch0 = out["per_channel"]["0"]
    assert ch0["gauge_level_step_median_adu"] == pytest.approx(60.0)
    assert ch0["gauge_level_step_mad_adu"] == pytest.approx(0.0)
    assert ch0["gauge_step_over_observed"] == pytest.approx(1.0)


def test_compare_gauges_abstains_on_non_additive():
    ids = ["f0.fit", "f1.fit"]
    gu = _fake_gauge("u", ids, 1.0, a=1.2)
    gv = _fake_gauge("v", ids, 1.0)
    out = probe.compare_gauges(gu, gv, {})
    assert out["comparable"] is False
    assert out["reason"] == "non_additive_coefficients"


def test_roi_metrics_hand_computed():
    a = np.zeros((4, 8, 1), dtype=np.float64)
    b = np.zeros((4, 8, 1), dtype=np.float64)
    a[:, :, 0] = 10.0
    b[:, :, 0] = 4.0
    du = {"science": a, "valid": np.ones((4, 8), dtype=bool), "rejected_fraction": 0.0,
          "rejected_sample_count": 0, "n_frames": 3}
    dv = {"science": b, "valid": np.ones((4, 8), dtype=bool), "rejected_fraction": 0.0,
          "rejected_sample_count": 0, "n_frames": 3}
    m = probe._roi_metrics(du, dv)
    assert m["n"] == 32
    assert m["per_channel"]["0"]["median_adu"] == pytest.approx(6.0)
    assert m["per_channel"]["0"]["rms_adu"] == pytest.approx(6.0)
    assert m["per_channel"]["0"]["tangent_slope_adu_per_px"] == pytest.approx(0.0)


def test_roi_metrics_handles_per_channel_valid_mask():
    # The canonical result's valid_mask is (H, W, C) for RGB; metrics must reduce
    # it to the shared 2-D support instead of broadcasting a 3-D/2-D mismatch.
    a = np.full((4, 8, 3), 10.0)
    b = np.full((4, 8, 3), 4.0)
    du = {"science": a, "valid": np.ones((4, 8, 3), dtype=bool), "n_frames": 3}
    dv = {"science": b, "valid": np.ones((4, 8, 3), dtype=bool), "n_frames": 3}
    m = probe._roi_metrics(du, dv)
    assert m["n"] == 32
    assert m["per_channel"]["0"]["median_adu"] == pytest.approx(6.0)
    # a channel dropped anywhere removes the pixel from the shared support
    du2 = {"science": a, "valid": np.ones((4, 8, 3), dtype=bool), "n_frames": 3}
    du2["valid"][0, 0, 1] = False
    assert probe._roi_metrics(du2, dv)["n"] == 31


def test_roi_metrics_abstains_without_shared_pixels():
    a = np.zeros((4, 4, 1))
    b = np.zeros((4, 4, 1))
    du = {"science": a, "valid": np.zeros((4, 4), dtype=bool), "n_frames": 1}
    dv = {"science": b, "valid": np.zeros((4, 4), dtype=bool), "n_frames": 1}
    assert probe._roi_metrics(du, dv)["n"] == 0


# ---------------------------------------------------------------------------
# 7. ROI provider over a real R6 aligned cache
# ---------------------------------------------------------------------------

def test_roi_provider_slices_real_cache(tmp_path):
    from zemosaic.core.zegrid import file_provider as zfp

    ids = ["a.fit", "b.fit"]
    rgb0 = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)
    sup0 = np.ones((3, 4), dtype=bool)
    rgb1 = rgb0 + 100.0
    sup1 = np.ones((3, 4), dtype=bool)
    cache = tmp_path / "cache"
    zfp.write_aligned_cache_from_arrays(cache, ids, [rgb0, rgb1], [sup0, sup1])

    prov = probe._RoiProvider(cache, ["b.fit"], (1, 3, 0, 2))
    try:
        assert prov.n_frames == 1 and prov.height == 2 and prov.width == 2
        raw, sup = prov.get_raw_frame(0)
        np.testing.assert_array_equal(raw, rgb1[1:3, 0:2])
        tile, valid = prov.get_tile(0, 0, 2, 0, 2)
        np.testing.assert_array_equal(tile, rgb1[1:3, 0:2])
        assert valid.all()
    finally:
        prov.close()


# ---------------------------------------------------------------------------
# 8. Factor probe refusals (never a fabricated result)
# ---------------------------------------------------------------------------

def test_factor_probe_not_run_without_gauges():
    rel = {"u": "r0000c0000", "v": "r0000c0001", "orientation": "h", "membership_cost": 10}
    out = probe.run_factor_probe(rel, {}, {}, {"max_cell_contributors": 10,
                                               "band_px": 8, "max_tangent_px": 32}, None)
    assert out["status"] == "NOT_RUN"
    assert out["reason"] == "gauge_facts_unavailable"


def test_factor_probe_not_run_over_cap():
    ids = ["f0.fit", "f1.fit"]
    g = _fake_gauge("r0000c0000", ids, 1.0)
    out = probe.run_factor_probe(
        {"u": "r0000c0000", "v": "r0000c0001", "orientation": "h", "membership_cost": 10},
        {"r0000c0000": g, "r0000c0001": g},
        {"r0000c0000": {"cache_dir": "/nope"}, "r0000c0001": {"cache_dir": "/nope"}},
        {"max_cell_contributors": 1, "band_px": 8, "max_tangent_px": 32}, None,
    )
    assert out["status"] == "NOT_RUN"
    assert "contributor_cap_exceeded" in out["reason"]


# ---------------------------------------------------------------------------
# 9. Conclusion rule
# ---------------------------------------------------------------------------

def _cmp(u, v, ratios, consistency=0.05):
    return {"comparable": True, "u": u, "v": v,
            "per_channel": {str(i): {"gauge_step_over_observed": r,
                                     "consistency_mad_over_abs_median": consistency,
                                     "observed_pair_diff_adu": 100.0}
                            for i, r in enumerate(ratios)}}


def test_conclude_partial_cause():
    res = probe.conclude([_cmp("a", "b", [0.9, 0.8]), _cmp("c", "d", [0.7, 0.6])], [])
    assert res["state"] == "PARTIAL_CAUSE"
    assert res["dominant_cause_confirmed"] is False


def test_conclude_falsified():
    res = probe.conclude([_cmp("a", "b", [0.01, 0.02]), _cmp("c", "d", [0.0, 0.03])], [])
    assert res["state"] == "CAUSE_FALSIFIED"


def test_conclude_inconclusive_when_too_few():
    res = probe.conclude([_cmp("a", "b", [0.9])], [])
    assert res["state"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# 10. Output safety
# ---------------------------------------------------------------------------

def test_write_safety_refuses_read_only_tree(tmp_path):
    forbidden = pathlib.Path("/home/tristan/caldwell11")
    with pytest.raises(probe.BudgetError):
        probe._assert_write_allowed(forbidden / "x" / "y.json", label="report file")
    with pytest.raises(probe.BudgetError):
        probe._assert_write_allowed(tmp_path / "ok.json", label="report file",
                                    extra_forbidden=(tmp_path,))


# ---------------------------------------------------------------------------
# 11. End-to-end inventory on a synthetic capture
# ---------------------------------------------------------------------------

def test_inventory_end_to_end_synthetic(tmp_path):
    capture, src, man, art_path, art = _make_capture(tmp_path, n_frames=6)
    _finalise_capture(capture, src, man, art)
    before = {p.name: p.stat().st_mtime_ns for p in capture.iterdir()}
    out = tmp_path / "out" / "inv.json"
    rc = probe.main([
        "--capture-dir", str(capture), "--source-dir", str(src),
        "--mode", "inventory", "--expect-identity", "any",
        "--scratch-dir", str(tmp_path / "scratch"),
        "--report-dir", str(tmp_path / "out"),
        "--out", str(out), "--determinism-check",
    ])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["schema"] == probe.SCHEMA_NAME
    assert report["mission"] == probe.MISSION
    assert report["panel"]["label"] == "NOT_QUALIFICATION"
    assert len(report["panel"]["relations"]) == 6
    assert report["import_audit"]["engine_access_is_lazy"] is True
    assert report["determinism"]["stable"] is True
    # strict JSON: no NaN tokens
    assert "NaN" not in out.read_text()
    # read-only: the capture tree is byte-identical (only the synthetic source
    # was never touched either)
    after = {p.name: p.stat().st_mtime_ns for p in capture.iterdir()}
    assert before == after
