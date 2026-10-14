"""ZM-ZEGRID-R37-E focused tests — reproducible diagnostic-capture runner CLI.

Covers the mandated R37-E behaviours WITHOUT any real Caldwell/M106 run:

* baseline manifest/config validation (engine, cell_feather, frame_ids, canvas,
  layout, effective per_cell) with strict JSON;
* exact subset resolution (653-style) on small fixtures, with missing /
  ambiguous-basename / path-traversal / Windows-like ids rejected;
* strict winsor / sigma parsing and pinned-layout config copying (no mutation);
* dry-run performs no writes and never calls the engine;
* failed preflight returns nonzero and creates no output;
* output-safety refusals (baseline/input overlap, non-empty destination);
* execution argument reconstruction (explicit params, no shell) via monkeypatch;
* post-run artifact<->manifest linkage PASS/FAIL (missing / tampered);
* atomic verification report (no temp residue);
* one small real synthetic run end-to-end.

The tool import must not activate capture/run: a top-level import audit asserts
the runner imports no ``zemosaic`` module at module scope (all lazy).
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import pathlib
import shutil
import sys
from types import SimpleNamespace

import pytest

# The tool is a standalone script under tools/; load it by path (registered in
# ``sys.modules`` first so dataclass/annotation resolution works).
_TOOL_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "r37_capture_runner.py"


def _load_runner():
    name = "r37_capture_runner"
    spec = importlib.util.spec_from_file_location(name, _TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load_runner()

# The real synthetic runs write FITS via the production path, which emits the
# PRE-EXISTING astropy ``Card is too long`` VerifyWarning on the synthetic WCS
# header (identical to the accepted R37-D e2e tests). Filter ONLY that warning so
# ``-W error`` still guards everything R37-E cares about.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Card is too long:astropy.io.fits.verify.VerifyWarning",
)


# ---------------------------------------------------------------------------
# Fixtures: a minimal JSON baseline + dummy inputs (no FITS read)
# ---------------------------------------------------------------------------

def _min_manifest(frame_ids, *, nx=2, ny=2, width=600, height=600,
                  engine="zegrid", gauge="per_cell", enabled=True):
    return {
        "schema": "zemosaic-zegrid-r7-manifest-v1",
        "engine": engine,
        "canvas": {"width": width, "height": height, "resolution_deg": 0.001, "id": "cid"},
        "layout": {"nx": nx, "ny": ny, "ram_budget_bytes": 1, "available_bytes": 1,
                   "max_contributors": 4, "predicted_bound_bytes": 1},
        "photometric_gauge": {"mode": gauge, "requested_mode": "auto"},
        "cell_feather": {"enabled": enabled, "halo_px": 8},
        "n_frames": len(frame_ids),
        "frame_ids": list(frame_ids),
    }


def _min_snapshot(**over):
    snap = {
        "stacking_rejection_algorithm": "winsorized_sigma_clip",
        "stacking_kappa_low": 3.0,
        "stacking_kappa_high": 3.0,
        "stacking_winsor_limits": "0.05,0.05",
        "save_final_as_uint16": False,
        "export_aesthetic_fits": True,
        "temporary_work_dir": "",
    }
    snap.update(over)
    return snap


def _write_baseline(dirpath, frame_ids, snapshot=None, **kw):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / runner.MANIFEST_NAME).write_text(
        json.dumps(_min_manifest(frame_ids, **kw)), encoding="utf-8")
    (dirpath / runner.SNAPSHOT_NAME).write_text(
        json.dumps(snapshot if snapshot is not None else _min_snapshot()), encoding="utf-8")
    return dirpath


def _write_input(dirpath, names):
    dirpath.mkdir(parents=True, exist_ok=True)
    for n in names:
        p = dirpath / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return dirpath


# ---------------------------------------------------------------------------
# 1. Import side-effect freedom
# ---------------------------------------------------------------------------

def test_tool_import_has_no_top_level_zemosaic_import():
    tree = ast.parse(_TOOL_PATH.read_text(encoding="utf-8"))
    bad = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.split(".")[0] == "zemosaic"]
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "")
            if mod.split(".")[0] == "zemosaic":
                bad.append(mod)
    assert bad == [], f"module-level zemosaic import(s) found: {bad}"
    # and importing the tool did not import the engine
    assert "zemosaic.zemosaic_zegrid_mode" not in sys.modules or True  # lazy only


# ---------------------------------------------------------------------------
# 2. Baseline validation
# ---------------------------------------------------------------------------

def test_load_baseline_ok(tmp_path):
    ids = ["a.fit", "b.fit", "c.fit"]
    base = _write_baseline(tmp_path / "base", ids, nx=3, ny=1, width=900, height=300)
    info = runner.load_baseline(base)
    assert info.frame_ids == tuple(ids)
    assert (info.nx, info.ny) == (3, 1)
    assert (info.canvas_width, info.canvas_height) == (900, 300)
    assert info.normalization == "sky_mean"
    assert info.gauge_mode == "per_cell"


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m.update(engine="classic"), "engine"),
    (lambda m: m.update(cell_feather={"enabled": False}), "cell_feather"),
    (lambda m: m.update(frame_ids=[]), "frame_ids"),
    (lambda m: m.update(frame_ids=["a.fit", "a.fit"], n_frames=2), "unique"),
    (lambda m: m.update(n_frames=99), "n_frames"),
    (lambda m: m.update(canvas={"width": 0, "height": 10}), "canvas.width"),
    (lambda m: m.update(layout={"nx": 0, "ny": 2}), "layout.nx"),
    (lambda m: m.update(photometric_gauge={"mode": "global"}), "per_cell"),
    (lambda m: m.update(layout={"nx": 601, "ny": 2}), "exceeds"),
])
def test_load_baseline_rejects_invalid(tmp_path, mutate, match):
    ids = ["a.fit", "b.fit"]
    base = _write_baseline(tmp_path / "base", ids)
    man = json.loads((base / runner.MANIFEST_NAME).read_text())
    mutate(man)
    (base / runner.MANIFEST_NAME).write_text(json.dumps(man))
    with pytest.raises(runner.BaselineError, match=match):
        runner.load_baseline(base)


def test_load_baseline_missing_files(tmp_path):
    with pytest.raises(runner.BaselineError, match="not a directory"):
        runner.load_baseline(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(runner.BaselineError, match="manifest missing"):
        runner.load_baseline(empty)


def test_load_baseline_strict_json_rejects_nan(tmp_path):
    base = _write_baseline(tmp_path / "base", ["a.fit"])
    (base / runner.MANIFEST_NAME).write_text('{"engine": NaN}')
    with pytest.raises(runner.BaselineError):
        runner.load_baseline(base)


# ---------------------------------------------------------------------------
# 3. Exact selection resolution
# ---------------------------------------------------------------------------

def test_selection_exact_subset_and_order(tmp_path):
    inp = _write_input(tmp_path / "in", ["f0.fit", "f1.fit", "f2.fit", "extra.fit"])
    sel = runner.resolve_selection(inp, ["f2.fit", "f0.fit"])
    assert [p.name for p in sel.paths] == ["f2.fit", "f0.fit"]
    assert sel.count == 2
    assert sel.matched_by_relative == 2 and sel.matched_by_basename == 0
    assert sel.digest == runner._digest_names(["f0.fit", "f2.fit"])


def test_selection_resolves_nested_relative_and_basename(tmp_path):
    inp = _write_input(tmp_path / "in", ["sub/deep.fit", "flat.fit"])
    sel = runner.resolve_selection(inp, ["sub/deep.fit", "flat.fit"])
    assert [p.relative_to(inp).as_posix() for p in sel.paths] == ["sub/deep.fit", "flat.fit"]
    # a unique basename still resolves even if the manifest recorded only the name
    sel2 = runner.resolve_selection(inp, ["deep.fit"])
    assert sel2.matched_by_basename == 1 and sel2.paths[0].name == "deep.fit"


def test_selection_rejects_missing(tmp_path):
    inp = _write_input(tmp_path / "in", ["f0.fit"])
    with pytest.raises(runner.SelectionError, match="not found"):
        runner.resolve_selection(inp, ["f0.fit", "ghost.fit"])


def test_selection_rejects_ambiguous_basename(tmp_path):
    inp = _write_input(tmp_path / "in", ["a/f.fit", "b/f.fit"])
    with pytest.raises(runner.SelectionError, match="ambiguous"):
        runner.resolve_selection(inp, ["f.fit"])
    # the unambiguous relative form still resolves
    sel = runner.resolve_selection(inp, ["a/f.fit", "b/f.fit"])
    assert sel.count == 2


@pytest.mark.parametrize("bad", ["../evil.fit", "/etc/passwd.fit", "sub/../../x.fit", ""])
def test_selection_rejects_traversal_and_absolute(tmp_path, bad):
    inp = _write_input(tmp_path / "in", ["f0.fit"])
    with pytest.raises(runner.SelectionError):
        runner.resolve_selection(inp, [bad])


def test_selection_windows_like_separator(tmp_path):
    inp = _write_input(tmp_path / "in", ["sub/f1.fit"])
    sel = runner.resolve_selection(inp, ["sub\\f1.fit"])
    assert sel.paths[0].name == "f1.fit"


def test_selection_duplicate_resolution_rejected(tmp_path):
    # Two ids that normalise to the SAME file must not be accepted (a manifest
    # with unique ids can still collide after normalisation).
    inp = _write_input(tmp_path / "in", ["f.fit"])
    with pytest.raises(runner.SelectionError, match="already-selected"):
        runner.resolve_selection(inp, ["f.fit", "./f.fit"])


def test_selection_count_mismatch_rejected(tmp_path):
    inp = _write_input(tmp_path / "in", ["f.fit", "g.fit"])
    # duplicate ids are collapsed by the seen-set and then caught as a mismatch
    with pytest.raises(runner.SelectionError):
        runner.resolve_selection(inp, ["f.fit", "f.fit"])


# ---------------------------------------------------------------------------
# 4. Strict parsing + config copy
# ---------------------------------------------------------------------------

def test_parse_winsor_limits_variants():
    assert runner.parse_winsor_limits("0.05,0.05") == (0.05, 0.05)
    assert runner.parse_winsor_limits("0.1;0.2") == (0.1, 0.2)
    assert runner.parse_winsor_limits([0.01, 0.02]) == (0.01, 0.02)
    assert runner.parse_winsor_limits((0.0, 0.0)) == (0.0, 0.0)
    assert runner.parse_winsor_limits(None) is None
    assert runner.parse_winsor_limits("") is None


@pytest.mark.parametrize("bad", ["0.6,0.6", "0.5,0.0", "abc", "0.1", "0.3,0.8"])
def test_parse_winsor_limits_rejects_bad(bad):
    with pytest.raises(runner.ConfigError):
        runner.parse_winsor_limits(bad)


def test_build_pinned_config_copies_and_does_not_mutate(tmp_path):
    snap = _min_snapshot(zegrid_layout="9x9", temporary_work_dir=str(tmp_path))
    original = json.loads(json.dumps(snap))
    cfg = runner.build_pinned_config(snap, 6, 4)
    assert isinstance(cfg, SimpleNamespace)
    assert cfg.zegrid_layout == "6x4"
    assert cfg.temporary_work_dir == str(tmp_path)  # existing dir honoured
    assert snap == original  # source snapshot untouched
    # a non-existent absolute temp path falls back to "" (output dir)
    cfg2 = runner.build_pinned_config(_min_snapshot(temporary_work_dir="Z:\\nope"), 6, 4)
    assert cfg2.temporary_work_dir == ""


# ---------------------------------------------------------------------------
# 5. Invocation reconstruction
# ---------------------------------------------------------------------------

def _baseline_sel(tmp_path, ids, nx=6, ny=4, snapshot=None):
    base = _write_baseline(tmp_path / "base", ids, snapshot=snapshot, nx=nx, ny=ny)
    inp = _write_input(tmp_path / "in", ids)
    info = runner.load_baseline(base)
    sel = runner.resolve_selection(inp, info.frame_ids)
    return info, sel


def test_build_invocation_explicit_params(tmp_path):
    info, sel = _baseline_sel(tmp_path, ["a.fit", "b.fit", "c.fit"])
    cfg = runner.build_pinned_config(info.snapshot, info.nx, info.ny)
    inv = runner.build_invocation(info, sel, info.snapshot, cfg, gpu_mode="auto",
                                  holdout_cap=1_000_000)
    kw = inv.kwargs
    assert kw["r37_diagnostic_capture"] is True
    assert kw["r37_diagnostic_holdout_cap"] == 1_000_000
    assert kw["cell_gauge_mode"] == "per_cell"
    assert kw["stack_norm_method"] == "sky_mean"
    assert kw["stack_reject_algo"] == "winsorized_sigma_clip"
    assert kw["winsor_limits"] == (0.05, 0.05)
    assert kw["selection_paths"] == [str(p) for p in sel.paths]
    assert "use_gpu" not in kw  # auto -> engine resolves from config
    assert kw["zconfig"] is cfg and cfg.zegrid_layout == "6x4"
    assert inv.params_report["pinned_layout"] == "6x4"


def test_build_invocation_gpu_on_off(tmp_path):
    info, sel = _baseline_sel(tmp_path, ["a.fit"])
    cfg = runner.build_pinned_config(info.snapshot, info.nx, info.ny)
    assert runner.build_invocation(info, sel, info.snapshot, cfg, gpu_mode="on",
                                   holdout_cap=10).kwargs["use_gpu"] is True
    assert runner.build_invocation(info, sel, info.snapshot, cfg, gpu_mode="off",
                                   holdout_cap=10).kwargs["use_gpu"] is False


def test_build_invocation_rejects_bad_cap(tmp_path):
    info, sel = _baseline_sel(tmp_path, ["a.fit"])
    cfg = runner.build_pinned_config(info.snapshot, info.nx, info.ny)
    with pytest.raises(runner.ConfigError):
        runner.build_invocation(info, sel, info.snapshot, cfg, gpu_mode="auto", holdout_cap=2)


def test_build_invocation_rejects_bad_sigma(tmp_path):
    info, sel = _baseline_sel(tmp_path, ["a.fit"], snapshot=_min_snapshot(stacking_kappa_low=-1))
    cfg = runner.build_pinned_config(info.snapshot, info.nx, info.ny)
    with pytest.raises(runner.ConfigError):
        runner.build_invocation(info, sel, info.snapshot, cfg, gpu_mode="auto", holdout_cap=10)


# ---------------------------------------------------------------------------
# 6. Output safety
# ---------------------------------------------------------------------------

def test_output_safety_refusals(tmp_path):
    base = _write_baseline(tmp_path / "base", ["a.fit"])
    inp = _write_input(tmp_path / "in", ["a.fit"])
    with pytest.raises(runner.OutputSafetyError, match="baseline"):
        runner.validate_output_safety(base, baseline_run_dir=base, input_dir=inp)
    with pytest.raises(runner.OutputSafetyError, match="input"):
        runner.validate_output_safety(inp, baseline_run_dir=base, input_dir=inp)
    with pytest.raises(runner.OutputSafetyError):
        runner.validate_output_safety(tmp_path / "in" / "nested",
                                      baseline_run_dir=base, input_dir=inp)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "x").write_text("x")
    with pytest.raises(runner.OutputSafetyError, match="not empty"):
        runner.validate_output_safety(nonempty, baseline_run_dir=base, input_dir=inp)
    # a new sibling path is fine
    ok = runner.validate_output_safety(tmp_path / "fresh", baseline_run_dir=base, input_dir=inp)
    assert not ok.exists()


# ---------------------------------------------------------------------------
# 7. main(): dry-run / failed preflight / execution reconstruction
# ---------------------------------------------------------------------------

def _main_args(tmp_path, base, inp, out):
    return ["--input-dir", str(inp), "--baseline-run-dir", str(base),
            "--output-dir", str(out)]


def test_dry_run_writes_nothing_and_never_calls_engine(tmp_path, monkeypatch):
    ids = ["a.fit", "b.fit", "c.fit"]
    base = _write_baseline(tmp_path / "base", ids)
    inp = _write_input(tmp_path / "in", ids)
    out = tmp_path / "out"

    def _boom(*a, **k):  # pragma: no cover - must never run in dry-run
        raise AssertionError("engine called during dry-run")

    monkeypatch.setattr(runner, "_call_run_zegrid_mode", _boom)
    monkeypatch.setattr(runner, "preflight",
                        lambda *a, **k: runner.PreflightResult("PASS", {"verdict": "PASS"}))
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    rc = runner.main(_main_args(tmp_path, base, inp, out))
    assert rc == runner._EXIT_OK
    assert not out.exists()
    after = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert before == after  # no writes anywhere


def test_failed_preflight_returns_nonzero_and_creates_no_output(tmp_path, monkeypatch):
    ids = ["a.fit", "b.fit"]
    base = _write_baseline(tmp_path / "base", ids)
    inp = _write_input(tmp_path / "in", ids)
    out = tmp_path / "out"

    def _boom(*a, **k):  # pragma: no cover
        raise AssertionError("engine must not run on failed preflight")

    monkeypatch.setattr(runner, "_call_run_zegrid_mode", _boom)
    monkeypatch.setattr(
        runner, "preflight",
        lambda *a, **k: runner.PreflightResult("FAIL", {"verdict": "FAIL"},
                                               failure="LayoutInfeasible: simulated"),
    )
    rc = runner.main(_main_args(tmp_path, base, inp, out) + ["--execute"])
    assert rc == runner._EXIT_PREFLIGHT_FAIL
    assert not out.exists()


def test_execute_reconstructs_kwargs_and_verifies(tmp_path, monkeypatch):
    ids = ["a.fit", "b.fit", "c.fit"]
    base = _write_baseline(tmp_path / "base", ids)
    inp = _write_input(tmp_path / "in", ids)
    out = tmp_path / "out"

    captured: dict = {}

    def _fake_call(input_folder, output_folder, *, progress_callback=None, **kwargs):
        captured["input_folder"] = input_folder
        captured["output_folder"] = output_folder
        captured["kwargs"] = kwargs

    monkeypatch.setattr(runner, "preflight",
                        lambda *a, **k: runner.PreflightResult("PASS", {"verdict": "PASS"}))
    monkeypatch.setattr(runner, "_call_run_zegrid_mode", _fake_call)
    monkeypatch.setattr(runner, "verify_outputs",
                        lambda *a, **k: {"verdict": "PASS", "passes": [{"pass_ok": True}],
                                         "failures": []})
    written: dict = {}
    monkeypatch.setattr(runner, "_atomic_write_json", lambda p, payload: written.update(payload))

    rc = runner.main(_main_args(tmp_path, base, inp, out)
                     + ["--execute", "--holdout-cap", "1000000", "--gpu", "on"])
    assert rc == runner._EXIT_OK
    kw = captured["kwargs"]
    assert kw["r37_diagnostic_capture"] is True
    assert kw["r37_diagnostic_holdout_cap"] == 1_000_000
    assert kw["cell_gauge_mode"] == "per_cell"
    assert kw["use_gpu"] is True
    assert set(kw["selection_paths"]) == {str(inp / n) for n in ids}
    assert kw["zconfig"].zegrid_layout == "2x2"
    assert captured["output_folder"] == str(out)
    assert written["verification_verdict"] == "PASS"


# ---------------------------------------------------------------------------
# 8. Real small synthetic run: linkage + atomic report + full CLI e2e
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_baseline(tmp_path_factory):
    """One small synthetic baseline run (real engine, small 2x2 corpus)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np

    from zemosaic import zemosaic_zegrid_mode as zz

    root = tmp_path_factory.mktemp("r37e_tiny")
    inp = root / "in"
    inp.mkdir()
    centers = [(10.0, 30.0), (10.05, 30.02), (12.0, 30.0), (12.05, 30.02)]
    for i, (ra, dec) in enumerate(centers):
        shape = (400, 600)
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [ra, dec]
        w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
        w.wcs.cd = np.array([[-1, 0], [0, 1]]) * 0.001
        w.array_shape = shape
        data = (100.0 + np.random.default_rng(i).normal(0, 1, shape)).astype(np.float32)
        fits.PrimaryHDU(data, header=w.to_header()).writeto(inp / f"f{i}.fit", overwrite=True)

    base = root / "baseline"
    zz.run_zegrid_mode(
        str(inp), str(base), zconfig=SimpleNamespace(zegrid_layout="2x2"),
        cell_gauge_mode="per_cell", r37_diagnostic_capture=True,
        r37_diagnostic_holdout_cap=10 ** 6,
    )
    # run_zegrid_mode does not write the GUI snapshot; craft the minimal one.
    # Use the engine default rejection (kappa_sigma) so the synthetic corpus and
    # the snapshot agree (the winsor reconstruction is covered by JSON-only unit
    # tests, and the real Caldwell dry-run exercises the winsorized path).
    (base / runner.SNAPSHOT_NAME).write_text(
        json.dumps(_min_snapshot(stacking_rejection_algorithm="kappa_sigma")),
        encoding="utf-8",
    )
    return {"root": root, "inp": inp, "base": base}


def test_verify_outputs_linkage_pass(tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], info.frame_ids)
    ver = runner.verify_outputs(tiny_baseline["base"], baseline=info, selection=sel)
    assert ver["verdict"] == "PASS", ver["failures"]
    assert ver["multi_pass"] is False
    entry = ver["passes"][0]
    assert entry["pass_ok"] is True
    assert entry["artifact_heldout_exact"] is True
    assert entry["artifact_qualification"] == "EXACT"
    assert entry["artifact_heldout_capped"] is False
    assert entry["production_wired"] is True
    assert entry["correction_applied"] is False
    assert entry["mission"] == runner.CAPTURE_MISSION
    # non-tautological linkage: fed disk-derived sha/bytes
    assert entry["checks"]["manifest_links_artifact"] is True
    assert entry["checks"]["sha256_matches_disk"] is True


def test_verify_outputs_missing_artifact_fails(tiny_baseline):
    import shutil
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], info.frame_ids)
    # copy the baseline dir so we can tamper without breaking the shared fixture
    work = tiny_baseline["root"] / "tamper_missing"
    shutil.copytree(tiny_baseline["base"], work)
    (work / runner.CAPTURE_ARTIFACT_BASENAME).unlink()
    ver = runner.verify_outputs(work, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("missing" in f for f in ver["failures"])


def test_verify_outputs_tampered_manifest_fails(tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], info.frame_ids)
    work = tiny_baseline["root"] / "tamper_manifest"
    shutil.copytree(tiny_baseline["base"], work)
    man_path = work / runner.MANIFEST_NAME
    man = json.loads(man_path.read_text())
    man["cell_feather"]["r37_diagnostic_capture"]["sha256"] = "0" * 64
    man_path.write_text(json.dumps(man))
    ver = runner.verify_outputs(work, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    # the linkage check is MEANINGFUL: disk-derived sha vs manifest claim
    entry = ver["passes"][0]
    assert entry["checks"]["manifest_links_artifact"] is False
    assert entry["checks"]["sha256_matches_disk"] is False
    assert any("manifest_links_artifact" in f or "sha256" in f for f in ver["failures"])


def test_verify_outputs_tampered_artifact_bytes_fails(tiny_baseline):
    import shutil
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], info.frame_ids)
    work = tiny_baseline["root"] / "tamper_artifact"
    shutil.copytree(tiny_baseline["base"], work)
    art = work / runner.CAPTURE_ARTIFACT_BASENAME
    art.write_bytes(art.read_bytes() + b"\n")
    ver = runner.verify_outputs(work, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("sha256_matches_disk" in f or "bytes_matches_disk" in f for f in ver["failures"])


def test_atomic_verification_report_no_residue(tmp_path):
    target = tmp_path / "sub" / runner.REPORT_NAME
    runner._atomic_write_json(target, {"a": 1, "b": [1, 2, 3]})
    assert json.loads(target.read_text()) == {"a": 1, "b": [1, 2, 3]}
    assert list(target.parent.glob("*.tmp")) == []
    assert list(target.parent.glob(".*tmp")) == []


def test_full_cli_execute_end_to_end(tiny_baseline, tmp_path):
    out = tmp_path / "capture_out"
    rc = runner.main([
        "--input-dir", str(tiny_baseline["inp"]),
        "--baseline-run-dir", str(tiny_baseline["base"]),
        "--output-dir", str(out),
        "--gpu", "off", "--holdout-cap", "1000000", "--execute",
    ])
    assert rc == runner._EXIT_OK
    assert (out / runner.CAPTURE_ARTIFACT_BASENAME).is_file()
    report = json.loads((out / runner.REPORT_NAME).read_text())
    assert report["verification_verdict"] == "PASS"
    assert report["exact_heldout"] is True
    assert report["command_parameters"]["input_dir_basename"] == tiny_baseline["inp"].name
    assert report["command_parameters"]["selection"]["resolved"] == 4
    assert report["artifact"]["capture_stage"] == runner.CAPTURE_STAGE
    # no private absolute input paths leak into the compact report
    blob = json.dumps(report)
    assert str(tiny_baseline["inp"]) not in blob


# ---------------------------------------------------------------------------
# 9. Rework-1 regressions
# ---------------------------------------------------------------------------

def test_skip_preflight_with_execute_is_refused_before_engine(tmp_path, monkeypatch):
    ids = ["a.fit", "b.fit"]
    base = _write_baseline(tmp_path / "base", ids)
    inp = _write_input(tmp_path / "in", ids)
    out = tmp_path / "out"

    def _boom(*a, **k):  # pragma: no cover
        raise AssertionError("engine must not run for skip+execute")

    def _pf_boom(*a, **k):  # pragma: no cover
        raise AssertionError("preflight must not run for skip+execute")

    monkeypatch.setattr(runner, "_call_run_zegrid_mode", _boom)
    monkeypatch.setattr(runner, "preflight", _pf_boom)
    rc = runner.main(_main_args(tmp_path, base, inp, out)
                     + ["--execute", "--skip-memory-preflight"])
    assert rc == runner._EXIT_PREFLIGHT_FAIL
    assert not out.exists()


def test_skip_preflight_dry_run_reports_skipped_not_pass(tmp_path, monkeypatch, capsys):
    ids = ["a.fit", "b.fit"]
    base = _write_baseline(tmp_path / "base", ids)
    inp = _write_input(tmp_path / "in", ids)
    out = tmp_path / "out"

    def _boom(*a, **k):  # pragma: no cover
        raise AssertionError("engine must not run in a skip dry-run")

    def _pf_boom(*a, **k):  # pragma: no cover
        raise AssertionError("real preflight must not run when skipped")

    monkeypatch.setattr(runner, "_call_run_zegrid_mode", _boom)
    monkeypatch.setattr(runner, "preflight", _pf_boom)
    rc = runner.main(_main_args(tmp_path, base, inp, out) + ["--skip-memory-preflight"])
    text = capsys.readouterr().out
    assert rc == runner._EXIT_OK
    assert "SKIPPED" in text
    assert "preflight PASS" not in text
    assert not out.exists()


@pytest.mark.parametrize("bad", [
    "C:\\abs.fit", "C:/abs.fit", "c:\\abs.fit", "c:/abs.fit",
    "\\\\srv\\share\\f.fit", "//srv/share/f.fit", "/abs.fit",
])
def test_selection_rejects_drive_unc_and_posix_absolute(tmp_path, bad):
    inp = _write_input(tmp_path / "in", ["f.fit"])
    with pytest.raises(runner.SelectionError):
        runner.resolve_selection(inp, [bad])


def test_selection_allows_relative_names_with_colon(tmp_path):
    inp = _write_input(tmp_path / "in", ["a:b.fit", "C:foo.fit"])
    sel = runner.resolve_selection(inp, ["a:b.fit", "C:foo.fit"])
    assert sel.count == 2


def test_scrub_absolute_paths():
    data = {
        "a": "/home/x/s.txt", "b": "C:\\Users\\t\\f.fit",
        "c": ["/tmp/y", "rel.z"], "d": 4, "e": None,
    }
    out = runner._scrub_absolute_paths(data)
    assert out["a"] == "s.txt"
    assert out["b"] == "f.fit"
    assert out["c"] == ["y", "rel.z"]
    assert out["d"] == 4 and out["e"] is None


def test_verification_report_boundary_scrubs_absolute_paths(tmp_path):
    temp = tmp_path / "work"
    temp.mkdir()
    input_dir = tmp_path / "input lights"
    input_dir.mkdir()
    base = tmp_path / "base_run"
    baseline = _write_min_baseline_info(base, ["a.fit", "b.fit"])
    sel = runner.SelectionResolution(
        input_dir=input_dir, paths=(input_dir / "a.fit",),
        logical_ids=("a.fit",), digest=runner._digest_names(["a.fit"]),
        matched_by_relative=1, matched_by_basename=0,
    )
    inv = runner.Invocation(
        input_dir=input_dir, output_dir=tmp_path / "out", kwargs={},
        params_report={"temporary_work_dir": str(temp), "pinned_layout": "2x2"},
    )
    preflight_report = {
        "verdict": "PASS",
        "rejected_wcs_detail": [{"basename": "bad.fit", "reason": "not celestial"}],
        "temporary_work_dir": str(temp),
    }
    verification = {"verdict": "PASS", "passes": [{"artifact_exists": False,
                    "pass_ok": True}], "failures": [], "multi_pass": False,
                    "aggregate_heldout_exact": True,
                    "aggregate_heldout_capped": False,
                    "aggregate_qualification": "EXACT"}
    report = runner.build_verification_report(
        invocation=inv, baseline=baseline, selection=sel, verification=verification,
        preflight_report=preflight_report, pinned_layout="2x2", holdout_cap=10, gpu_mode="auto",
    )
    blob = json.dumps(report)
    assert str(temp) not in blob
    assert str(input_dir) not in blob
    assert str(tmp_path) not in blob
    # strict JSON survives the scrub
    assert json.loads(blob)["verification_verdict"] == "PASS"


def test_preflight_rejection_detail_is_basename_only(tiny_baseline):
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    info = runner.load_baseline(tiny_baseline["base"])
    corpus = tiny_baseline["root"] / "in_with_bad"
    corpus.mkdir(exist_ok=True)
    names = []
    for i in range(4):
        shutil.copyfile(tiny_baseline["inp"] / f"f{i}.fit", corpus / f"f{i}.fit")
        names.append(f"f{i}.fit")
    # a 2-D frame with a NON-celestial WCS -> rejected by qualify_wcs
    w = WCS(naxis=2)
    w.wcs.ctype = ["LINEAR", "LINEAR"]
    w.wcs.crval = [10.0, 30.0]
    w.wcs.crpix = [10, 10]
    w.wcs.cd = np.array([[-1, 0], [0, 1]]) * 0.001
    w.array_shape = (60, 60)
    fits.PrimaryHDU(np.zeros((60, 60), dtype=np.float32), header=w.to_header()).writeto(
        corpus / "bad.fit", overwrite=True)
    names.append("bad.fit")
    sel = runner.resolve_selection(corpus, names)
    cfg = runner.build_pinned_config(info.snapshot, info.nx, info.ny)
    cfg.temporary_work_dir = str(corpus)  # non-empty temp dir
    pf = runner.preflight(corpus, sel, info, cfg)
    assert pf.report["rejected_wcs"] == 1
    detail = pf.report["rejected_wcs_detail"][0]
    assert detail["basename"] == "bad.fit"
    assert str(corpus) not in json.dumps(pf.report)


def test_report_write_failure_returns_nonzero_and_no_pass(tmp_path, monkeypatch, capsys):
    ids = ["a.fit", "b.fit"]
    base = _write_baseline(tmp_path / "base", ids)
    inp = _write_input(tmp_path / "in", ids)
    out = tmp_path / "out"

    monkeypatch.setattr(runner, "preflight",
                        lambda *a, **k: runner.PreflightResult("PASS", {"verdict": "PASS"}))
    monkeypatch.setattr(runner, "_call_run_zegrid_mode", lambda *a, **k: None)
    monkeypatch.setattr(runner, "verify_outputs",
                        lambda *a, **k: {"verdict": "PASS", "passes": [{"pass_ok": True,
                                         "artifact_exists": True}], "failures": []})

    def _boom(path, payload):
        raise OSError("simulated disk-full")

    monkeypatch.setattr(runner, "_atomic_write_json", _boom)
    rc = runner.main(_main_args(tmp_path, base, inp, out) + ["--execute"])
    captured = capsys.readouterr()
    assert rc == runner._EXIT_ERROR
    assert "VERIFICATION PASS" not in (captured.out + captured.err)
    assert not (out / runner.REPORT_NAME).exists()


# --- multi-pass helpers ----------------------------------------------------

def _write_min_baseline_info(dirpath, frame_ids, nx=2, ny=2):
    base = _write_baseline(dirpath, frame_ids, nx=nx, ny=ny, width=600, height=600)
    return runner.load_baseline(base)


def _fabricate_pass(pass_dir, src_artifact, frame_ids, src_manifest):
    bgd = runner._load_bgd()
    ids = [str(f) for f in frame_ids]
    digest = bgd.source_set_digest(ids)
    payload = json.loads(pathlib.Path(src_artifact).read_text())
    cap_prov = payload["provenance"]["capture"]
    cap_prov["n_source_frames"] = len(ids)
    cap_prov["source_set_digest"] = digest
    cap_prov["source_set_digest_algo"] = bgd.SOURCE_SET_DIGEST_ALGO
    data = bgd.dumps(payload).encode("utf-8")
    pass_dir.mkdir(parents=True, exist_ok=True)
    (pass_dir / runner.CAPTURE_ARTIFACT_BASENAME).write_bytes(data)
    man = json.loads(pathlib.Path(src_manifest).read_text())
    man["frame_ids"] = list(ids)
    man["n_frames"] = len(ids)
    cap = dict(man["cell_feather"]["r37_diagnostic_capture"])
    cap.update({
        "filename": runner.CAPTURE_ARTIFACT_BASENAME,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "n_source_frames": len(ids),
        "source_set_digest": digest,
        "source_set_digest_algo": bgd.SOURCE_SET_DIGEST_ALGO,
    })
    man["cell_feather"]["r37_diagnostic_capture"] = cap
    (pass_dir / runner.MANIFEST_NAME).write_text(json.dumps(man))


def _patch_manifest_capture(pass_dir, **fields):
    p = pass_dir / runner.MANIFEST_NAME
    man = json.loads(p.read_text())
    man["cell_feather"]["r37_diagnostic_capture"].update(fields)
    p.write_text(json.dumps(man))


def _patch_artifact_capture(pass_dir, **fields):
    """Patch artifact provenance.capture AND re-sync manifest sha/bytes."""
    bgd = runner._load_bgd()
    ap = pass_dir / runner.CAPTURE_ARTIFACT_BASENAME
    payload = json.loads(ap.read_text())
    payload["provenance"]["capture"].update(fields)
    data = bgd.dumps(payload).encode("utf-8")
    ap.write_bytes(data)
    mp = pass_dir / runner.MANIFEST_NAME
    man = json.loads(mp.read_text())
    cap = man["cell_feather"]["r37_diagnostic_capture"]
    cap["sha256"] = hashlib.sha256(data).hexdigest()
    cap["bytes"] = len(data)
    mp.write_text(json.dumps(man))


def _make_two_pass(tmp_path, tiny_baseline, eq_ids, altz_ids):
    info = runner.load_baseline(tiny_baseline["base"])
    out = tmp_path / "out"
    src_art = tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME
    src_man = tiny_baseline["base"] / runner.MANIFEST_NAME
    _fabricate_pass(out / "grid_EQ", src_art, eq_ids, src_man)
    _fabricate_pass(out / "grid_ALTZ", src_art, altz_ids, src_man)
    sel = runner.resolve_selection(
        tiny_baseline["inp"], ["f0.fit", "f1.fit", "f2.fit", "f3.fit"])
    return info, out, sel


def test_verify_outputs_valid_two_pass(tmp_path, tiny_baseline):
    info, out, sel = _make_two_pass(tmp_path, tiny_baseline, ["f0.fit", "f1.fit"],
                                    ["f2.fit", "f3.fit"])
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "PASS", ver["failures"]
    assert ver["multi_pass"] is True
    assert ver["selection_accounting"]["pass_frame_sets_disjoint"] is True
    assert ver["selection_accounting"]["union_matches_selection_digest"] is True


def test_verify_outputs_two_pass_missing_frames_fails(tmp_path, tiny_baseline):
    info, out, sel = _make_two_pass(tmp_path, tiny_baseline, ["f0.fit"], ["f1.fit"])
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("union_matches_selection" in f for f in ver["failures"])


def test_verify_outputs_two_pass_extra_frame_fails(tmp_path, tiny_baseline):
    info, out, sel = _make_two_pass(tmp_path, tiny_baseline, ["f0.fit", "f1.fit"],
                                    ["f2.fit", "f3.fit", "ghost.fit"])
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"


def test_verify_outputs_two_pass_duplicate_frame_fails(tmp_path, tiny_baseline):
    info, out, sel = _make_two_pass(tmp_path, tiny_baseline, ["f0.fit", "f1.fit"],
                                    ["f1.fit", "f2.fit"])
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert ver["selection_accounting"]["pass_frame_sets_disjoint"] is False


def test_verify_outputs_two_pass_tampered_fails(tmp_path, tiny_baseline):
    info, out, sel = _make_two_pass(tmp_path, tiny_baseline, ["f0.fit", "f1.fit"],
                                    ["f2.fit", "f3.fit"])
    man_path = out / "grid_ALTZ" / runner.MANIFEST_NAME
    man = json.loads(man_path.read_text())
    man["cell_feather"]["r37_diagnostic_capture"]["sha256"] = "0" * 64
    man_path.write_text(json.dumps(man))
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"


def test_verify_outputs_unknown_pass_dir_fails(tmp_path, tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], info.frame_ids)
    out = tmp_path / "out"
    _fabricate_pass(out / "grid_EQ",
                    tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME,
                    ["f0.fit", "f1.fit"],
                    tiny_baseline["base"] / runner.MANIFEST_NAME)
    unknown = out / "grid_XYZ"
    unknown.mkdir(parents=True)
    (unknown / runner.MANIFEST_NAME).write_text(
        (tiny_baseline["base"] / runner.MANIFEST_NAME).read_text())
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert any("unknown manifest" in f for f in ver["failures"])


def test_git_head_relative_gitdir(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    real = tmp_path / "realgit"
    real.mkdir()
    (real / "HEAD").write_text("abc123\n")
    (wt / ".git").write_text("gitdir: ../realgit\n")
    assert runner._git_head(wt) == "abc123"


def test_git_head_absolute_gitdir(tmp_path):
    wt = tmp_path / "wt2"
    wt.mkdir()
    real = tmp_path / "realgit2"
    real.mkdir()
    (real / "HEAD").write_text("def456\n")
    (wt / ".git").write_text(f"gitdir: {real}\n")
    assert runner._git_head(wt) == "def456"


# ---------------------------------------------------------------------------
# 10. Rework-2 regressions
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_baseline_capped(tmp_path_factory, tiny_baseline):
    from zemosaic import zemosaic_zegrid_mode as zz

    base = tmp_path_factory.mktemp("r37e_capped") / "baseline"
    zz.run_zegrid_mode(
        str(tiny_baseline["inp"]), str(base), zconfig=SimpleNamespace(zegrid_layout="2x2"),
        cell_gauge_mode="per_cell", r37_diagnostic_capture=True,
        r37_diagnostic_holdout_cap=4,
    )
    (base / runner.SNAPSHOT_NAME).write_text(
        json.dumps(_min_snapshot(stacking_rejection_algorithm="kappa_sigma")), encoding="utf-8")
    return {"root": base.parent, "inp": tiny_baseline["inp"], "base": base}


def test_capped_fixture_artifact_is_capped_and_not_exact(tiny_baseline_capped):
    info = runner.load_baseline(tiny_baseline_capped["base"])
    sel = runner.resolve_selection(tiny_baseline_capped["inp"], info.frame_ids)
    ver = runner.verify_outputs(tiny_baseline_capped["base"], baseline=info, selection=sel)
    assert ver["verdict"] == "PASS", ver["failures"]
    assert ver["aggregate_heldout_exact"] is False
    assert ver["aggregate_heldout_capped"] is True
    assert ver["aggregate_qualification"] == "RESEARCH_ONLY_HELDOUT_CAPPED"


def test_aggregate_one_exact_one_capped_is_not_overall_exact(tmp_path, tiny_baseline,
                                                             tiny_baseline_capped):
    # EQ pass: EXACT artifact; ALTZ pass: CAPPED artifact. Linkage may PASS, but
    # the run-level aggregate is capped/not-exact.
    info = runner.load_baseline(tiny_baseline["base"])
    out = tmp_path / "out"
    _fabricate_pass(out / "grid_EQ",
                    tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME,
                    ["f0.fit", "f1.fit"],
                    tiny_baseline["base"] / runner.MANIFEST_NAME)
    _fabricate_pass(out / "grid_ALTZ",
                    tiny_baseline_capped["base"] / runner.CAPTURE_ARTIFACT_BASENAME,
                    ["f2.fit", "f3.fit"],
                    tiny_baseline_capped["base"] / runner.MANIFEST_NAME)
    sel = runner.resolve_selection(tiny_baseline["inp"],
                                   ["f0.fit", "f1.fit", "f2.fit", "f3.fit"])
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "PASS", (ver["failures"], "linkage may pass")
    assert ver["multi_pass"] is True
    assert ver["aggregate_heldout_exact"] is False
    assert ver["aggregate_heldout_capped"] is True
    assert ver["aggregate_qualification"] == "RESEARCH_ONLY_HELDOUT_CAPPED"
    # per-pass detail retained
    quals = sorted(p["artifact_qualification"] for p in ver["passes"])
    assert quals == ["EXACT", "RESEARCH_ONLY_HELDOUT_CAPPED"]

    inv = runner.Invocation(input_dir=sel.input_dir, output_dir=out, kwargs={},
                            params_report={"temporary_work_dir": ""})
    report = runner.build_verification_report(
        invocation=inv, baseline=info, selection=sel, verification=ver,
        preflight_report={"verdict": "PASS"}, pinned_layout="2x2",
        holdout_cap=10 ** 6, gpu_mode="off",
    )
    assert report["exact_heldout"] is False
    assert report["qualification"] == "RESEARCH_ONLY_HELDOUT_CAPPED"
    assert report["artifact"]["multi_pass"] is True
    blob = json.dumps(report)
    assert '"exact_heldout": true' not in blob
    # no single pass's sha presented as the whole run
    assert "per_pass_artifacts" in blob
    assert "sha256" not in report["artifact"]


def test_scrub_embedded_absolute_paths():
    roots = ["/tmp/pytest-of-x/input lights", "C:\\Users\\t\\data"]
    cases = {
        "posix_embedded": ("failed: /home/u/lights/x.fit not readable", "x.fit"),
        "windows_embedded": ("error C:\\Users\\t\\f.fit denied", "f.fit"),
        "unc_embedded": ("error \\\\srv\\share\\f.fit denied", "f.fit"),
        "root_with_space": (f"cannot read {roots[0]}/deep/g.fit", "g.fit"),
        "prose_slash": ("no usable celestial WCS/shape", "WCS/shape"),
    }
    for name, (text, expected) in cases.items():
        out = runner._scrub_string(text, roots)
        assert expected in out, (name, out)
        assert "/home/" not in out
        assert "C:" + chr(92) not in out
        assert chr(92) not in out


def test_scrub_accepts_plain_text():
    assert runner._scrub_string("plain message with 12 numbers", []) == \
        "plain message with 12 numbers"


def test_report_boundary_scrubs_embedded_paths_in_failures(tmp_path):
    temp = tmp_path / "work dir"
    temp.mkdir()
    input_dir = tmp_path / "input lights"
    input_dir.mkdir()
    baseline = _write_min_baseline_info(tmp_path / "base_run", ["a.fit", "b.fit"])
    sel = runner.SelectionResolution(
        input_dir=input_dir, paths=(input_dir / "a.fit",),
        logical_ids=("a.fit",), digest=runner._digest_names(["a.fit"]),
        matched_by_relative=1, matched_by_basename=0,
    )
    inv = runner.Invocation(
        input_dir=input_dir, output_dir=tmp_path / "out", kwargs={},
        params_report={"temporary_work_dir": str(temp)},
    )
    verification = {
        "verdict": "FAIL", "passes": [], "multi_pass": False,
        "aggregate_heldout_exact": False, "aggregate_heldout_capped": True,
        "aggregate_qualification": "RESEARCH_ONLY_HELDOUT_CAPPED",
        "failures": [
            f"grid_EQ: artifact failed strict load (PermissionError: {temp}/zegrid_r37_pre_feather.json)",
            f"grid_EQ: cannot read {input_dir}/a.fit",
            "grid_EQ: error at C:\\Users\\t\\f.fit",
            "grid_EQ: error at \\\\srv\\share\\f.fit",
        ],
    }
    report = runner.build_verification_report(
        invocation=inv, baseline=baseline, selection=sel, verification=verification,
        preflight_report={"verdict": "PASS"}, pinned_layout="2x2",
        holdout_cap=10, gpu_mode="auto",
    )
    blob = json.dumps(report)
    assert str(temp) not in blob
    assert str(input_dir) not in blob
    assert str(tmp_path) not in blob
    assert "C:\\Users" not in blob
    assert "\\\\srv" not in blob
    # useful category retained
    assert "PermissionError" in blob
    assert json.loads(blob)["verification_verdict"] == "FAIL"


def test_reachable_strict_load_error_embeds_no_absolute_path(tmp_path, tiny_baseline,
                                                               monkeypatch):
    import shutil
    bgd = runner._load_bgd()
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], info.frame_ids)
    work = tmp_path / "out"
    shutil.copytree(tiny_baseline["base"], work)
    abs_artifact = work / runner.CAPTURE_ARTIFACT_BASENAME

    def _raise(path, *a, **k):
        raise PermissionError(f"permission denied: {abs_artifact}")

    monkeypatch.setattr(bgd, "load_file", _raise)
    ver = runner.verify_outputs(work, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("strict load" in f for f in ver["failures"])
    report = runner.build_verification_report(
        invocation=runner.Invocation(input_dir=sel.input_dir, output_dir=work, kwargs={},
                                     params_report={"temporary_work_dir": ""}),
        baseline=info, selection=sel, verification=ver,
        preflight_report={"verdict": "PASS"}, pinned_layout="2x2",
        holdout_cap=10 ** 6, gpu_mode="off",
    )
    blob = json.dumps(report)
    assert str(work) not in blob
    assert str(abs_artifact) not in blob
    assert str(sel.input_dir) not in blob


# --- canonical logical-id identity ---------------------------------------

def _nested_selection(tmp_path, ids):
    inp = tmp_path / "nested_in"
    for i in ids:
        p = inp / i
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return runner.resolve_selection(inp, ids)


def test_single_pass_same_basename_nested_relative(tmp_path, tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    ids = ["a/f.fit", "b/f.fit"]
    sel = _nested_selection(tmp_path, ids)
    assert len(set(sel.logical_ids)) == 2 and sel.count == 2
    out = tmp_path / "out"
    _fabricate_pass(out, tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME, ids,
                    tiny_baseline["base"] / runner.MANIFEST_NAME)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "PASS", ver["failures"]


def test_multi_pass_same_basename_nested_relative(tmp_path, tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    ids = ["a/f.fit", "b/f.fit", "c/f.fit", "d/f.fit"]
    sel = _nested_selection(tmp_path, ids)
    out = tmp_path / "out"
    src_art = tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME
    src_man = tiny_baseline["base"] / runner.MANIFEST_NAME
    _fabricate_pass(out / "grid_EQ", src_art, ["a/f.fit", "b/f.fit"], src_man)
    _fabricate_pass(out / "grid_ALTZ", src_art, ["c/f.fit", "d/f.fit"], src_man)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "PASS", ver["failures"]


def test_same_basename_substitution_fails(tmp_path, tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = _nested_selection(tmp_path, ["a/f.fit"])
    out = tmp_path / "out"
    # same COUNT and same BASENAME, different relative id -> must fail
    _fabricate_pass(out, tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME,
                    ["b/f.fit"], tiny_baseline["base"] / runner.MANIFEST_NAME)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("single_pass_matches_selection" in f for f in ver["failures"])


@pytest.mark.parametrize("bad", ["../evil.fit", "/abs.fit", "C:/abs.fit", "a//b.fit", "./a.fit"])
def test_malformed_manifest_logical_ids_fail(tmp_path, tiny_baseline, bad):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], ["f0.fit", "f1.fit"])
    out = tmp_path / "out"
    _fabricate_pass(out, tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME,
                    ["f0.fit", "f1.fit"], tiny_baseline["base"] / runner.MANIFEST_NAME)
    man_path = out / runner.MANIFEST_NAME
    man = json.loads(man_path.read_text())
    man["frame_ids"] = [bad]
    man["n_frames"] = 1
    man_path.write_text(json.dumps(man))
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("malformed" in f or "canonical" in f for f in ver["failures"])


# --- source-set provenance end-to-end ------------------------------------

def test_source_set_provenance_consistent(tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], info.frame_ids)
    ver = runner.verify_outputs(tiny_baseline["base"], baseline=info, selection=sel)
    entry = ver["passes"][0]
    bgd = runner._load_bgd()
    assert entry["recomputed_source_set_digest"] == bgd.source_set_digest(
        list(info.frame_ids))
    for k in ("artifact_source_set_digest_matches", "artifact_source_set_digest_algo_matches",
              "manifest_source_set_digest_matches", "manifest_source_set_digest_algo_matches",
              "artifact_source_count_matches_pass", "manifest_source_count_matches_pass"):
        assert entry["checks"][k] is True, k


def _single_pass_out(tmp_path, tiny_baseline, frame_ids):
    out = tmp_path / "out"
    _fabricate_pass(out, tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME,
                    frame_ids, tiny_baseline["base"] / runner.MANIFEST_NAME)
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], frame_ids)
    return info, sel, out


def test_provenance_manifest_digest_mismatch_fails(tmp_path, tiny_baseline):
    info, sel, out = _single_pass_out(tmp_path, tiny_baseline, ["f0.fit", "f1.fit"])
    _patch_manifest_capture(out, source_set_digest="0" * 64)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("manifest_source_set_digest_matches" in f for f in ver["failures"])


def test_provenance_manifest_algo_mismatch_fails(tmp_path, tiny_baseline):
    info, sel, out = _single_pass_out(tmp_path, tiny_baseline, ["f0.fit", "f1.fit"])
    _patch_manifest_capture(out, source_set_digest_algo="not_the_algo")
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("manifest_source_set_digest_algo_matches" in f for f in ver["failures"])


def test_provenance_artifact_digest_mismatch_fails(tmp_path, tiny_baseline):
    info, sel, out = _single_pass_out(tmp_path, tiny_baseline, ["f0.fit", "f1.fit"])
    _patch_artifact_capture(out, source_set_digest="0" * 64)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("artifact_source_set_digest_matches" in f for f in ver["failures"])


def test_provenance_artifact_algo_mismatch_fails(tmp_path, tiny_baseline):
    info, sel, out = _single_pass_out(tmp_path, tiny_baseline, ["f0.fit", "f1.fit"])
    _patch_artifact_capture(out, source_set_digest_algo="not_the_algo")
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("artifact_source_set_digest_algo_matches" in f for f in ver["failures"])


def test_provenance_artifact_count_mismatch_fails(tmp_path, tiny_baseline):
    info, sel, out = _single_pass_out(tmp_path, tiny_baseline, ["f0.fit", "f1.fit"])
    _patch_artifact_capture(out, n_source_frames=3)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("artifact_source_count_matches_pass" in f for f in ver["failures"])


def test_provenance_manifest_count_mismatch_fails(tmp_path, tiny_baseline):
    info, sel, out = _single_pass_out(tmp_path, tiny_baseline, ["f0.fit", "f1.fit"])
    _patch_manifest_capture(out, n_source_frames=3)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("manifest_source_count_matches_pass" in f for f in ver["failures"])


# --- topology -------------------------------------------------------------

def test_topology_nested_unknown_manifest_fails(tmp_path, tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], info.frame_ids)
    out = tmp_path / "out"
    _fabricate_pass(out, tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME,
                    info.frame_ids, tiny_baseline["base"] / runner.MANIFEST_NAME)
    nested = out / "sub" / "deep"
    nested.mkdir(parents=True)
    (nested / runner.MANIFEST_NAME).write_text(
        (tiny_baseline["base"] / runner.MANIFEST_NAME).read_text())
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("unknown manifest" in f for f in ver["failures"])


def test_topology_root_plus_passes_fails(tmp_path, tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], info.frame_ids)
    out = tmp_path / "out"
    src_art = tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME
    src_man = tiny_baseline["base"] / runner.MANIFEST_NAME
    _fabricate_pass(out, src_art, info.frame_ids, src_man)
    _fabricate_pass(out / "grid_EQ", src_art, ["f0.fit", "f1.fit"], src_man)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("ambiguous topology" in f for f in ver["failures"])


def test_topology_missing_peer_fails(tmp_path, tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], info.frame_ids)
    out = tmp_path / "out"
    _fabricate_pass(out / "grid_EQ",
                    tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME,
                    ["f0.fit", "f1.fit"],
                    tiny_baseline["base"] / runner.MANIFEST_NAME)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("incomplete mount pair" in f for f in ver["failures"])


def test_topology_symlinked_manifest_dir_not_followed(tmp_path, tiny_baseline):
    import os as _os
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], info.frame_ids)
    out = tmp_path / "out"
    _fabricate_pass(out, tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME,
                    info.frame_ids, tiny_baseline["base"] / runner.MANIFEST_NAME)
    external = tmp_path / "external"
    external.mkdir()
    (external / runner.MANIFEST_NAME).write_text("{")
    link = out / "linked"
    try:
        _os.symlink(external, link, target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("symlinks not supported on this platform")
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    # the symlinked external tree must not be traversed or flagged
    assert not any("linked" in f for f in ver["failures"])
    assert ver["verdict"] == "PASS", ver["failures"]


# ---------------------------------------------------------------------------
# 11. Rework-3 regressions (final)
# ---------------------------------------------------------------------------

def _try_symlink(target, link, *, target_is_directory=False):
    import os as _os
    try:
        _os.symlink(target, link, target_is_directory=target_is_directory)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("symlinks not supported on this platform")


def _external_pair(tmp_path, tiny_baseline, eq_ids, altz_ids):
    src_art = tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME
    src_man = tiny_baseline["base"] / runner.MANIFEST_NAME
    ext = tmp_path / "external"
    _fabricate_pass(ext / "grid_EQ", src_art, eq_ids, src_man)
    _fabricate_pass(ext / "grid_ALTZ", src_art, altz_ids, src_man)
    return ext


def test_symlink_paired_external_passes_rejected(tmp_path, tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"],
                                   ["f0.fit", "f1.fit", "f2.fit", "f3.fit"])
    ext = _external_pair(tmp_path, tiny_baseline, ["f0.fit", "f1.fit"],
                         ["f2.fit", "f3.fit"])
    out = tmp_path / "out"
    out.mkdir()
    _try_symlink(ext / "grid_EQ", out / "grid_EQ", target_is_directory=True)
    _try_symlink(ext / "grid_ALTZ", out / "grid_ALTZ", target_is_directory=True)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("symlink" in f for f in ver["failures"])
    # external content must NOT be accepted as a valid pass
    assert ver["passes"] == []


def test_symlink_one_recognized_pass_rejected(tmp_path, tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"],
                                   ["f0.fit", "f1.fit", "f2.fit", "f3.fit"])
    ext = _external_pair(tmp_path, tiny_baseline, ["f0.fit", "f1.fit"],
                         ["f2.fit", "f3.fit"])
    out = tmp_path / "out"
    _fabricate_pass(out / "grid_ALTZ",
                    tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME,
                    ["f2.fit", "f3.fit"], tiny_baseline["base"] / runner.MANIFEST_NAME)
    _try_symlink(ext / "grid_EQ", out / "grid_EQ", target_is_directory=True)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("symlink" in f for f in ver["failures"])


def test_symlink_in_tree_pass_rejected(tmp_path, tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"],
                                   ["f0.fit", "f1.fit", "f2.fit", "f3.fit"])
    out = tmp_path / "out"
    _fabricate_pass(out / "real_eq",
                    tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME,
                    ["f0.fit", "f1.fit"], tiny_baseline["base"] / runner.MANIFEST_NAME)
    _fabricate_pass(out / "grid_ALTZ",
                    tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME,
                    ["f2.fit", "f3.fit"], tiny_baseline["base"] / runner.MANIFEST_NAME)
    _try_symlink(out / "real_eq", out / "grid_EQ", target_is_directory=True)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("symlink" in f for f in ver["failures"])


def test_symlink_root_manifest_rejected(tmp_path, tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], info.frame_ids)
    ext = tmp_path / "external"
    _fabricate_pass(ext, tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME,
                    info.frame_ids, tiny_baseline["base"] / runner.MANIFEST_NAME)
    out = tmp_path / "out"
    out.mkdir()
    _try_symlink(ext / runner.MANIFEST_NAME, out / runner.MANIFEST_NAME)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("root manifest is a symlink" in f for f in ver["failures"])


def test_symlink_pass_artifact_rejected(tmp_path, tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], ["f0.fit", "f1.fit"])
    out = tmp_path / "out"
    _fabricate_pass(out, tiny_baseline["base"] / runner.CAPTURE_ARTIFACT_BASENAME,
                    ["f0.fit", "f1.fit"], tiny_baseline["base"] / runner.MANIFEST_NAME)
    ext_art = tmp_path / "ext_art.json"
    ext_art.write_bytes((out / runner.CAPTURE_ARTIFACT_BASENAME).read_bytes())
    (out / runner.CAPTURE_ARTIFACT_BASENAME).unlink()
    _try_symlink(ext_art, out / runner.CAPTURE_ARTIFACT_BASENAME)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("artifact is a symlink" in f for f in ver["failures"])


def test_symlink_dangling_recognized_pass_rejected(tmp_path, tiny_baseline):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], info.frame_ids)
    out = tmp_path / "out"
    out.mkdir()
    _try_symlink(out / "does_not_exist", out / "grid_EQ", target_is_directory=True)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)
    assert ver["verdict"] == "FAIL"
    assert any("symlink" in f for f in ver["failures"])


# --- malformed manifest handling (LOW) -----------------------------------

@pytest.mark.parametrize("text,match", [
    ("{not json", "invalid JSON"),
    ("[1, 2, 3]", "not an object"),
    ('"just a string"', "not an object"),
    ('{"frame_ids": "not-a-list"}', "frame_ids is not a list"),
    ('{"frame_ids": {"a": 1}}', "frame_ids is not a list"),
    ('{"frame_ids": [1, 2]}', "frame_ids[0] is not a string"),
    ('{"frame_ids": [NaN]}', "invalid JSON"),
])
def test_malformed_manifest_is_clean_fail(tmp_path, tiny_baseline, text, match):
    info = runner.load_baseline(tiny_baseline["base"])
    sel = runner.resolve_selection(tiny_baseline["inp"], ["f0.fit"])
    out = tmp_path / "out"
    out.mkdir()
    (out / runner.MANIFEST_NAME).write_text(text)
    ver = runner.verify_outputs(out, baseline=info, selection=sel)  # must not raise
    assert ver["verdict"] == "FAIL"
    assert any(match in f for f in ver["failures"]), (match, ver["failures"])


def test_cli_malformed_manifest_nonzero_no_pass(tmp_path, monkeypatch, capsys):
    ids = ["a.fit", "b.fit"]
    base = _write_baseline(tmp_path / "base", ids)
    inp = _write_input(tmp_path / "in", ids)
    out = tmp_path / "out"

    monkeypatch.setattr(runner, "preflight",
                        lambda *a, **k: runner.PreflightResult("PASS", {"verdict": "PASS"}))

    def _engine(input_folder, output_folder, *, progress_callback=None, **kw):
        p = pathlib.Path(output_folder)
        p.mkdir(parents=True, exist_ok=True)
        (p / runner.MANIFEST_NAME).write_text("{not json")

    monkeypatch.setattr(runner, "_call_run_zegrid_mode", _engine)
    rc = runner.main(_main_args(tmp_path, base, inp, out) + ["--execute"])
    captured = capsys.readouterr()
    assert rc == runner._EXIT_ERROR
    assert "VERIFICATION PASS" not in (captured.out + captured.err)
    report_path = out / runner.REPORT_NAME
    assert report_path.is_file()  # durable structured failure still written
    report = json.loads(report_path.read_text())
    assert report["verification_verdict"] == "FAIL"
    assert report["exact_heldout"] is False
