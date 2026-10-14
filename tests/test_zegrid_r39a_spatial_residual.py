"""ZM-ZEGRID-R39-A focused tests — read-only post-offset spatial residual tool.

Covers, WITHOUT any real capture / Caldwell path / FITS read:

* module-scope import safety (no ``zemosaic`` at import time) + CLI help;
* strict JSON (NaN/Infinity refused) and deterministic ordering;
* exact capture/manifest identity happy path (generic mode) and each material
  mismatch/refusal (schema, stage, wiring, sha256/bytes, capture id, digest,
  capped holdout, dropped relations, widths, offsets, counts) — and NO report is
  written on a prepare failure;
* output safety (protected capture/dataset root, traversal, symlink, Windows
  drive both slashes/case, UNC, file URI, embedded absolute-path privacy, atomic
  replace + temp cleanup);
* ``d_post`` sign/order vs the accepted production ``apply_additive_offsets``;
  endpoint swap; analytic fixture <= 1e-10;
* coordinate forward/inverse, pixel/block boundaries, INV_TOL refusal, odd/odd
  parity, out-of-band refusal;
* no FITS / image-pixel reads (``open`` on ``.fits`` trapped);
* C0 identity; known tangent/normal/both signals; rank/condition/support gates;
  fold leakage guard; axis sign conflict; C4 relative rule;
* bootstrap determinism, whole-block resampling, effective-B reporting;
* metric statistics (RMS0 1.25 as RMS-around-zero, broad zero-median ->
  UNEXPLAINED, constant -> CONSTANT_ONLY, spatial -> SPATIAL_STABLE, non-centred
  p90);
* one-channel unexplained hard relation blocker;
* panel exact mixes / blocker precedence / >=2 h and >=2 v;
* universal SUPPORT_NOT_AVAILABLE attribution separation;
* strict artifact has no absolute private path recursively;
* unexpected exceptions => FAILED/nonzero; no writes under input roots.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import pathlib
import sys
import types

import numpy as np
import pytest

_TOOL_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "r39a_spatial_residual.py"
_ALGO = "sha256_sorted_unique_relative_logical_paths_v1"


def _load_tool():
    name = "r39a_spatial_residual"
    spec = importlib.util.spec_from_file_location(name, _TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


r39 = _load_tool()


# ---------------------------------------------------------------------------
# 1. Import safety / CLI
# ---------------------------------------------------------------------------

def test_import_audit_no_zemosaic_at_module_scope():
    audit = r39.import_audit()
    assert audit["engine_access_is_lazy"] is True
    assert audit["zemosaic_module_scope_imports"] == []
    assert "zemosaic" not in audit["module_scope_imports"]
    # never astropy / fits at module scope
    assert not any(m.startswith("astropy") for m in audit["module_scope_imports"])


def test_importing_tool_does_not_import_zemosaic():
    code = (
        "import importlib.util, sys; "
        f"s=importlib.util.spec_from_file_location('r39t', r'{_TOOL_PATH}'); "
        "m=importlib.util.module_from_spec(s); sys.modules['r39t']=m; s.loader.exec_module(m); "
        "bad=[x for x in sys.modules if x=='zemosaic' or x.startswith('zemosaic.')]; "
        "print('IMPORTED' if bad else 'CLEAN')"
    )
    import subprocess
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "CLEAN"


def test_cli_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        r39.main(["--help"])
    assert exc.value.code == 0


def test_cli_requires_capture_dir():
    with pytest.raises(SystemExit) as exc:
        r39.main([])
    assert exc.value.code == 2


def test_bootstrap_default_is_frozen_2000():
    assert r39.BOOTSTRAP_B_DEFAULT == 2000
    assert r39.BOOTSTRAP_SEED == 0
    assert r39.N_FOLDS == 4
    parser = r39.build_arg_parser()
    # no CLI override for the bootstrap replicate count
    opts = {a.dest for a in parser._actions}
    assert not any("boot" in o for o in opts)


# ---------------------------------------------------------------------------
# 2. Strict JSON
# ---------------------------------------------------------------------------

def test_dumps_strict_refuses_nan():
    with pytest.raises(ValueError):
        r39.dumps_strict({"x": float("nan")})
    with pytest.raises(ValueError):
        r39.dumps_strict({"x": float("inf")})


def test_dumps_strict_is_sorted_deterministic():
    a = r39.dumps_strict({"b": 1, "a": [2, 1]})
    b = r39.dumps_strict({"a": [2, 1], "b": 1})
    assert a == b
    assert a.index('"a"') < a.index('"b"')


def test_load_json_strict_rejects_constants(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"x": NaN}', encoding="utf-8")
    with pytest.raises(r39.R39Error):
        r39.load_json_strict(p)


def test_source_set_digest_matches_accepted_algorithm():
    ids = ["b.fit", "a.fit", "a.fit"]
    import hashlib as _h
    uniq = sorted(set(ids))
    expect = _h.sha256()
    expect.update((_ALGO + "\n").encode())
    for i in uniq:
        expect.update((i + "\n").encode())
    assert r39.source_set_digest(ids) == expect.hexdigest()


# ---------------------------------------------------------------------------
# Synthetic compact capture builders (self-contained; NO Caldwell paths)
# ---------------------------------------------------------------------------

def _grid_adjacency(nx, ny):
    edges = []
    for r in range(ny):
        for c in range(nx - 1):
            edges.append((f"r{r:04d}c{c:04d}", f"r{r:04d}c{c+1:04d}", "h"))
    for r in range(ny - 1):
        for c in range(nx):
            edges.append((f"r{r:04d}c{c:04d}", f"r{r+1:04d}c{c:04d}", "v"))
    return edges


def _band_grid(u, v, orientation, W, H, nx, ny):
    b = r39.relation_boundary(u, v, orientation, W, H, nx, ny)
    if orientation == "h":
        cols = np.array([c for c in range(W) if abs((c + 0.5) - b) <= 8.0], dtype=np.int64)
        rows = np.arange(H, dtype=np.int64)
    else:
        rows = np.array([r for r in range(H) if abs((r + 0.5) - b) <= 8.0], dtype=np.int64)
        cols = np.arange(W, dtype=np.int64)
    return cols, rows


def _relation_record(u, v, orientation, channel, W, H, nx, ny, rng, signal, noise_sigma,
                     n_target=1800):
    cols_all, rows_all = _band_grid(u, v, orientation, W, H, nx, ny)
    cc, rr = np.meshgrid(cols_all, rows_all, indexing="ij")
    cc = cc.ravel()
    rr = rr.ravel()
    # artifact support scalars are computed over ALL band pixels (production rule)
    tangent_blocks = int(np.unique(r39.tangent_block_index(cc, rr, orientation)).size)
    normal_blocks = int(np.unique(r39.normal_block_index(cc, rr, orientation)).size)
    ho_mask = ((cc // 4) % 2 == 1) & ((rr // 4) % 2 == 1)
    hc = cc[ho_mask]
    hr = rr[ho_mask]
    if hc.size > n_target:
        idx = np.linspace(0, hc.size - 1, n_target).astype(np.int64)
        hc = hc[idx]
        hr = hr[idx]
    n, t, s = r39.relation_normal_tangent(hc, hr, orientation, W, H, nx, ny, u, v)
    diff = signal(n, t, channel) + rng.normal(0.0, noise_sigma, size=n.size)
    xy = np.column_stack([r39.forward_coord(hc, W), r39.forward_coord(hr, H)])
    rec = {
        "u": u, "v": v, "channel": int(channel), "orientation": orientation,
        "n_train": int(cc.size - hc.size),
        "n_holdout": int(hc.size), "n_holdout_kept": int(hc.size),
        "holdout_capped": False,
        "effective_train": float(cc.size - hc.size),
        "tangent_blocks": tangent_blocks, "normal_blocks": normal_blocks,
        "robust_sigma": float(noise_sigma),
        "theta_global": [0.0, 0.0, 0.0],
        "information": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "rhs": [0.0, 0.0, 0.0],
        "measurement_px": 16,
        "holdout_xy": xy.tolist(),
        "holdout_diff": diff.tolist(),
    }
    return rec, hc, hr, n, t, s


def build_capture(root: pathlib.Path, *, nx=2, ny=2, W=400, H=400, channels=3,
                  signal=None, noise_sigma=1.0, seed=7, n_target=1800,
                  canvas_id="deadbeefdeadbeef", anchor=None, mutate=None):
    """Write a compact synthetic artifact + linked manifest; return its dir."""
    root.mkdir(parents=True, exist_ok=True)
    if signal is None:
        signal = lambda n, t, c: np.zeros_like(n)  # noqa: E731
    if anchor is None:
        anchor = f"r{ny // 2:04d}c{nx // 2:04d}"
    rng = np.random.default_rng(seed)
    edges = _grid_adjacency(nx, ny)
    relations = []
    for (u, v, orient) in edges:
        for ch in range(channels):
            rec, *_ = _relation_record(u, v, orient, ch, W, H, nx, ny, rng, signal,
                                       noise_sigma, n_target=n_target)
            relations.append(rec)
    cells = [f"r{r:04d}c{c:04d}" for r in range(ny) for c in range(nx)]
    offsets = {cid: [0.0] * channels for cid in cells}
    n_records = len(relations)
    n_geom = len(edges)

    artifact = {
        "schema": "zemosaic.zegrid.background_diagnostics",
        "schema_version": "r37c-1",
        "provenance": {
            "production_wired": True,
            "correction_applied": False,
            "capture_stage": "pre_offset_pre_feather",
            "capture": {
                "capture_id": "c" * 64,
                "n_source_frames": 4,
                "source_set_digest": "0",
                "source_set_digest_algo": _ALGO,
                "effective_cell_gauge_mode": "per_cell",
                "halo_px": 8, "measurement_px": 16, "feather_px": 8,
            },
            "references": ["https://example.invalid/ref"],
        },
        "meta": {
            "nx": nx, "ny": ny, "canvas_hw": [H, W], "channels": channels,
            "anchor": anchor, "measurement_px": 16, "block_px": 4, "feather_px": 8,
            "n_relations": n_records, "n_dropped": 0,
            "normalization": "global_canvas_normalized_[-1,1]",
            "coordinate_convention": "pixel_centre_+0.5_global_x_from_col_y_from_row",
        },
        "dropped": [],
        "relations": relations,
    }
    if mutate is not None:
        mutate(artifact, "artifact")

    frame_ids = [f"logical/frame_{i:04d}.fit" for i in range(4)]
    digest = r39.source_set_digest(frame_ids)
    artifact["provenance"]["capture"]["source_set_digest"] = digest

    cap_dir = root
    cap_path = cap_dir / r39.CAPTURE_ARTIFACT_BASENAME
    cap_path.write_text(json.dumps(artifact), encoding="utf-8")
    art_sha = hashlib.sha256(cap_path.read_bytes()).hexdigest()
    art_bytes = cap_path.stat().st_size
    manifest = {
        "engine": "zegrid",
        "n_frames": len(frame_ids),
        "frame_ids": frame_ids,
        "canvas": {"width": W, "height": H, "id": canvas_id},
        "layout": {"nx": nx, "ny": ny},
        "photometric_gauge": {"mode": "per_cell"},
        "cell_feather": {
            "enabled": True,
            "anchor_cell": anchor,
            "channels": channels,
            "halo_px": 8,
            "edges_used": n_geom, "edges_built": n_geom,
            "offsets": offsets,
            "r37_diagnostic_capture": {
                "capture_id": "c" * 64,
                "filename": r39.CAPTURE_ARTIFACT_BASENAME,
                "sha256": art_sha,
                "bytes": art_bytes,
                "source_set_digest": digest,
                "source_set_digest_algo": _ALGO,
                "n_source_frames": len(frame_ids),
                "qualification": "EXACT",
                "holdout_exact": True,
                "holdout_capped": False,
                "capture_stage": "pre_offset_pre_feather",
                "effective_cell_gauge_mode": "per_cell",
                "halo_px": 8, "measurement_px": 16, "feather_px": 8,
                "n_cells": nx * ny, "n_relations": n_records, "n_dropped": 0,
            },
        },
    }
    if mutate is not None:
        mutate(manifest, "manifest")
    (cap_dir / r39.MANIFEST_BASENAME).write_text(json.dumps(manifest), encoding="utf-8")
    return cap_dir


@pytest.fixture(scope="module")
def capture_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("cap0")
    return build_capture(root)


def _policy(cap):
    return r39.build_write_policy(capture_dir=cap)


# ---------------------------------------------------------------------------
# 3. Identity: happy path + each material mismatch
# ---------------------------------------------------------------------------

def test_identity_happy_path_generic(capture_dir):
    ident = r39.validate_identity(capture_dir, expect="any")
    comp = ident["computed"]
    assert comp["n_records"] == 12
    assert comp["n_relations_geometric"] == 4
    assert comp["orientation_h"] == 2 and comp["orientation_v"] == 2
    assert comp["block_px"] == 4 and comp["measurement_px"] == 16
    assert comp["n_dropped"] == 0


def test_identity_checked_mode_refuses_synthetic(capture_dir):
    # default (checked) frozen Windows816 identity must NOT accept synthetic data
    with pytest.raises(r39.IdentityError):
        r39.validate_identity(capture_dir, expect="checked")


def test_identity_checked_mode_accepts_matched_constants(capture_dir, monkeypatch):
    ident = r39.validate_identity(capture_dir, expect="any")
    monkeypatch.setattr(r39, "CHECKED_IDENTITY", dict(ident["computed"]))
    ident2 = r39.validate_identity(capture_dir, expect="checked")
    assert ident2["computed"]["n_records"] == 12


def _run_validated(cap, tmp_path):
    return r39.run(cap, expect="any", policy=_policy(cap), n_boot=8)


@pytest.mark.parametrize("mutate_desc", [
    "schema", "schema_version", "stage", "correction_applied", "production_wired",
    "halo_px", "measurement_px", "feather_px", "block_px", "n_dropped_meta",
    "dropped_list", "capped_relation", "bad_offsets", "manifest_sha", "manifest_bytes",
    "manifest_capture_id", "manifest_digest", "manifest_width", "edges_used",
    "gauge_mode", "capture_stage_manifest", "n_frames",
])
def test_identity_mismatches_refuse(tmp_path, mutate_desc):
    def mutate(obj, kind):
        if kind == "artifact":
            if mutate_desc == "schema":
                obj["schema"] = "wrong"
            elif mutate_desc == "schema_version":
                obj["schema_version"] = "r0"
            elif mutate_desc == "stage":
                obj["provenance"]["capture_stage"] = "post"
            elif mutate_desc == "correction_applied":
                obj["provenance"]["correction_applied"] = True
            elif mutate_desc == "production_wired":
                obj["provenance"]["production_wired"] = False
            elif mutate_desc == "halo_px":
                obj["provenance"]["capture"]["halo_px"] = 4
            elif mutate_desc == "measurement_px":
                obj["provenance"]["capture"]["measurement_px"] = 8
            elif mutate_desc == "feather_px":
                obj["provenance"]["capture"]["feather_px"] = 4
            elif mutate_desc == "block_px":
                obj["meta"]["block_px"] = 8
            elif mutate_desc == "n_dropped_meta":
                obj["meta"]["n_dropped"] = 2
            elif mutate_desc == "dropped_list":
                obj["dropped"] = [{"u": "a", "v": "b", "channel": 0, "reason": "x"}]
            elif mutate_desc == "capped_relation":
                obj["relations"][0]["holdout_capped"] = True
        else:  # manifest
            if mutate_desc == "manifest_sha":
                obj["cell_feather"]["r37_diagnostic_capture"]["sha256"] = "0" * 64
            elif mutate_desc == "manifest_bytes":
                obj["cell_feather"]["r37_diagnostic_capture"]["bytes"] = 1
            elif mutate_desc == "manifest_capture_id":
                obj["cell_feather"]["r37_diagnostic_capture"]["capture_id"] = "d" * 64
            elif mutate_desc == "manifest_digest":
                obj["cell_feather"]["r37_diagnostic_capture"]["source_set_digest"] = "9" * 64
            elif mutate_desc == "manifest_width":
                obj["canvas"]["width"] = obj["canvas"]["width"] + 1
            elif mutate_desc == "edges_used":
                obj["cell_feather"]["edges_used"] = 99
            elif mutate_desc == "gauge_mode":
                obj["photometric_gauge"]["mode"] = "global"
            elif mutate_desc == "capture_stage_manifest":
                obj["cell_feather"]["r37_diagnostic_capture"]["capture_stage"] = "x"
            elif mutate_desc == "n_frames":
                obj["n_frames"] = 99

    cap = build_capture(tmp_path / "cap", mutate=mutate)
    if mutate_desc == "bad_offsets":
        man = json.loads((cap / r39.MANIFEST_BASENAME).read_text())
        man["cell_feather"]["offsets"]["r0000c0000"] = [float("nan")] * 3
        cap.joinpath(r39.MANIFEST_BASENAME).write_text(json.dumps(man))
    with pytest.raises(r39.R39Error):
        r39.validate_identity(cap, expect="any")


def test_no_report_written_on_prepare_failure(tmp_path, capsys):
    cap = build_capture(tmp_path / "cap",
                        mutate=lambda o, k: o.update({"schema": "bad"}) if k == "artifact" else None)
    out = tmp_path / "out" / "r.json"
    rc = r39.main(["--capture-dir", str(cap), "--out", str(out),
                   "--expect-identity", "any"])
    assert rc == 1
    assert not out.exists()
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["status"] == "FAILED"
    assert printed["wrote_report"] is False


def test_structural_validation_refuses_bad_parity(tmp_path):
    def mutate(obj, kind):
        if kind != "artifact":
            return
        # move one held-out sample onto an even block in both axes
        xy = obj["relations"][0]["holdout_xy"]
        xy[0] = [r39.forward_coord(0, 400), r39.forward_coord(0, 400)]
    cap = build_capture(tmp_path / "cap", mutate=mutate)
    with pytest.raises(r39.IdentityError):
        _run_validated(cap, tmp_path)


def test_structural_validation_refuses_out_of_band(tmp_path):
    def mutate(obj, kind):
        if kind != "artifact":
            return
        xy = obj["relations"][0]["holdout_xy"]
        # corrupt X to a non-reproducible value (fails the INV_TOL reproduction)
        xy[0] = [xy[0][0] + 1e-6, xy[0][1]]
    cap = build_capture(tmp_path / "cap", mutate=mutate)
    with pytest.raises(r39.IdentityError):
        _run_validated(cap, tmp_path)


# ---------------------------------------------------------------------------
# 4. Output safety
# ---------------------------------------------------------------------------

def test_out_inside_capture_root_refused(tmp_path, capsys):
    cap = build_capture(tmp_path / "cap")
    out = cap / "report.json"
    rc = r39.main(["--capture-dir", str(cap), "--out", str(out), "--expect-identity", "any"])
    assert rc == 1
    assert not out.exists()


def test_out_inside_dataset_root_refused(tmp_path, capsys):
    cap = build_capture(tmp_path / "data" / "cap")
    dataset = tmp_path / "data"
    out = dataset / "nested" / "report.json"
    rc = r39.main(["--capture-dir", str(cap), "--dataset-root", str(dataset),
                   "--out", str(out), "--expect-identity", "any"])
    assert rc == 1
    assert not out.exists()


@pytest.mark.parametrize("raw", [
    "C:/x/report.json", "C:\\x\\report.json", "c:/x/report.json",
    "\\\\server\\share\\r.json", "file:///x/r.json", "../escape/r.json",
])
def test_ambiguous_out_paths_refused(raw, tmp_path, capsys):
    cat = r39.raw_path_is_ambiguous(raw)
    assert cat is not None
    cap = build_capture(tmp_path / "cap")
    rc = r39.main(["--capture-dir", str(cap), "--out", raw, "--expect-identity", "any"])
    assert rc == 1


def test_symlinked_parent_refused(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(r39, "BOOTSTRAP_B_DEFAULT", 8)
    cap = build_capture(tmp_path / "cap")
    real = tmp_path / "real_out"
    real.mkdir()
    link = tmp_path / "link_out"
    link.symlink_to(real, target_is_directory=True)
    out = link / "r.json"
    rc = r39.main(["--capture-dir", str(cap), "--out", str(out), "--expect-identity", "any"])
    assert rc == 1
    assert not (real / "r.json").exists()


def test_atomic_write_no_temp_residue(tmp_path):
    out = tmp_path / "out" / "r.json"
    policy = {"protected_roots": ()}
    out.parent.mkdir(parents=True)
    report = {"schema": r39.SCHEMA_NAME, "ok": True}
    n = r39._write_report(out, report, policy)
    assert n > 0
    assert json.loads(out.read_text())["ok"] is True
    assert not any(p.name.endswith(".tmp") for p in out.parent.iterdir())


def test_privacy_scan_catches_absolute_and_drive_paths():
    payload = {"a": "see /usr/local/x", "b": "C:\\Users\\x", "c": "\\\\srv\\share",
               "d": "file:///x"}
    found = r39.scan_path_leaks(payload)
    cats = {f["category"] for f in found}
    assert "posix_absolute_embedded" in cats
    assert "windows_drive_absolute" in cats
    assert "unc" in cats
    assert "file_uri" in cats


def test_privacy_boundary_redacts_and_rescans(tmp_path):
    payload = {"path": "/etc/passwd"}
    safe, leaks = r39.enforce_privacy_boundary(payload, [])
    assert leaks == []
    assert safe["path"] == r39.REDACTED_PATH


# ---------------------------------------------------------------------------
# 5. d_post sign/order
# ---------------------------------------------------------------------------

def test_sign_order_fixture_passes():
    out = r39.sign_order_fixtures()
    assert out["production_identity"] is True
    assert out["swapped_endpoint_flip"] is True
    assert out["analytic_d_post_max_abs_err"] <= 1e-10


def test_d_post_matches_production_apply_additive_offsets():
    zcf = r39._lazy_cell_feather()
    su = np.full((4, 4, 3), 10.0)
    sv = np.full((4, 4, 3), 3.0)
    su[0, 0] = 100.0
    off_u = np.array([1.0, 2.0, 3.0])
    off_v = np.array([-1.0, 0.5, 0.0])
    lhs = zcf.apply_additive_offsets(su, off_u) - zcf.apply_additive_offsets(sv, off_v)
    diff = su - sv
    for ch in range(3):
        rhs = r39.post_offset_residual(diff[:, :, ch], off_u[ch], off_v[ch])
        assert np.allclose(lhs[:, :, ch], rhs, atol=1e-12)


def test_d_post_endpoint_swap_flips_sign():
    diff = np.array([1.0, -2.0, 3.0])
    a = r39.post_offset_residual(diff, 0.5, -0.25)
    b = r39.post_offset_residual(-diff, -0.25, 0.5)
    assert np.allclose(b, -a, atol=1e-12)


# ---------------------------------------------------------------------------
# 6. Coordinates
# ---------------------------------------------------------------------------

def test_coordinate_fixtures_roundtrip():
    out = r39.coordinate_fixtures(400, 300)
    assert out["passed"] is True
    assert out["checks"] > 0


def test_inverse_tol_refusal():
    X = r39.forward_coord(10, 400)
    with pytest.raises(r39.IdentityError):
        r39.inverse_index(X + 1e-6, 400)
    assert r39.inverse_index(X, 400) == 10


def test_inverse_range_refusal():
    with pytest.raises(r39.IdentityError):
        r39.inverse_index(1.5, 400)


def test_tangent_block_along_boundary():
    # h -> tangent is rows; v -> tangent is cols (documented resolution)
    assert r39.tangent_block_index(11, 5, "h") == 1
    assert r39.tangent_block_index(11, 5, "v") == 2
    assert r39.normal_block_index(11, 5, "h") == 2
    assert r39.normal_block_index(11, 5, "v") == 1


# ---------------------------------------------------------------------------
# 7. No FITS / pixel reads
# ---------------------------------------------------------------------------

def test_no_fits_or_pixel_reads(capture_dir, monkeypatch):
    real_open = builtins.open
    opened = []

    def _guard(file, *a, **k):
        name = str(file)
        opened.append(name)
        if name.lower().endswith((".fits", ".fit", ".fts")):
            raise AssertionError(f"FITS/pixel read attempted: {name}")
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", _guard)
    monkeypatch.setitem(sys.modules, "astropy", types.ModuleType("astropy"))
    result = _run_validated(capture_dir, pathlib.Path("."))
    assert result["analysis"]["panel"]["panel_state"]
    assert not any(n.lower().endswith((".fits", ".fit", ".fts")) for n in opened)


# ---------------------------------------------------------------------------
# 8. Model family
# ---------------------------------------------------------------------------

def _synthetic_samples(kind, *, n=2500, seed=3, noise=1.0, amp=8.0):
    rng = np.random.default_rng(seed)
    block = np.arange(n) // 60              # ~42 blocks
    n_coord = rng.uniform(-0.0038, 0.0038, n)
    t_coord = rng.uniform(-0.5, 0.5, n)
    if kind == "flat":
        y = noise * rng.normal(size=n)
    elif kind == "const":
        y = amp + noise * rng.normal(size=n)
    elif kind == "tangent":
        y = amp * t_coord / 0.5 + noise * rng.normal(size=n)
    elif kind == "normal":
        y = amp * n_coord / 0.0038 + noise * rng.normal(size=n)
    elif kind == "both":
        y = amp * t_coord / 0.5 + amp * n_coord / 0.0038 + noise * rng.normal(size=n)
    elif kind == "broad":
        y = rng.laplace(0.0, amp, size=n)  # median ~0, broad spread (> 1.25 sigma_ref)
    elif kind == "signconflict":
        fold = block % 4
        sgn = np.where(fold % 2 == 0, 1.0, -1.0)
        y = amp * t_coord / 0.5 * sgn + noise * rng.normal(size=n)
    else:
        raise ValueError(kind)
    return {"n": n_coord, "t": t_coord, "s": n_coord * 200.0, "y": y,
            "tangent_block": block, "fold": block % r39.N_FOLDS, "sigma_ref": noise}


def _meta():
    return {"nx": 2, "ny": 2, "canvas_w": 400, "canvas_h": 400, "channels": 3,
            "block_px": 4, "measurement_px": 16, "halo_px": 8}


def _rel_dict(s, orient="h", u="r0000c0000", v="r0000c0001"):
    return {
        "u": u, "v": v, "orientation": orient, "channel": 0,
        "n": s["n"], "t": s["t"], "s": s["s"], "e_post": s["y"], "e_pre": s["y"],
        "tangent_block": s["tangent_block"], "normal_block": s["tangent_block"] // 2,
        "sigma_ref": s["sigma_ref"], "n_train": 5000, "effective_train": 5000.0,
        "artifact_tangent_blocks": 40, "artifact_normal_blocks": 4,
    }


def test_c0_identity_zero_model_constant_only():
    s = _synthetic_samples("const", amp=10.0)
    rec = r39.analyse_one_relation(_rel_dict(s), _meta(), n_boot=20)
    assert rec["models"]["C0"]["admitted"] is True
    assert rec["state"] == "CONSTANT_ONLY"
    assert rec["sig_const"] is True


def test_tangent_signal_spatial_stable():
    s = _synthetic_samples("tangent", amp=8.0)
    rec = r39.analyse_one_relation(_rel_dict(s), _meta(), n_boot=30)
    assert rec["models"]["C1"]["admitted"] is True
    assert rec["state"] == "SPATIAL_STABLE"
    assert rec["best_model"] in ("C1", "C2", "C3")


def test_normal_signal_spatial_stable():
    s = _synthetic_samples("normal", amp=6.0)
    rec = r39.analyse_one_relation(_rel_dict(s, "v", "r0000c0000", "r0001c0000"),
                                   _meta(), n_boot=30)
    assert rec["models"]["C2"]["admitted"] is True
    assert rec["state"] == "SPATIAL_STABLE"


def test_broad_zero_median_is_unexplained_not_negligible():
    s = _synthetic_samples("broad", amp=9.0)
    rec = r39.analyse_one_relation(_rel_dict(s), _meta(), n_boot=20)
    # median ~ 0 (a median-only gate would mis-classify) but RMS0 > 1.25 sigma_ref
    assert abs(rec["metrics"]["median"]) < 1.25 * s["sigma_ref"]
    assert rec["state"] == "UNEXPLAINED_STRUCTURE"
    assert r39.FLAG_SPATIAL_STRUCTURE_PARTIAL in rec["flags"] or rec["best_model"] == "C0"


def test_flat_noise_is_negligible():
    s = _synthetic_samples("flat", amp=0.0)
    rec = r39.analyse_one_relation(_rel_dict(s), _meta(), n_boot=20)
    assert rec["state"] in ("NEGLIGIBLE", "CONSTANT_ONLY")
    assert rec["sig_const"] is False or rec["state"] == "CONSTANT_ONLY"


def test_axis_sign_conflict_not_admitted():
    s = _synthetic_samples("signconflict", amp=8.0)
    ff = r39._fold_fits("C1", s["n"], s["t"], s["y"], s["fold"],
                        s["tangent_block"], s["tangent_block"] // 2)
    signs = {float(np.sign(c[1])) for c in ff["coefs_gate_ok"]}
    assert len(signs) > 1  # genuinely conflicting fold signs
    rec = r39.analyse_one_relation(_rel_dict(s), _meta(), n_boot=20)
    assert rec["models"]["C1"]["admitted"] is False


def test_fold_leakage_guard():
    # fold 0 contains wild outliers; a leaked fit would produce small OOF residuals
    n = np.zeros(400)
    t = np.zeros(400)
    y = np.zeros(400)
    block = np.arange(400) // 40  # 10 blocks -> folds 0..3
    fold = block % 4
    y[fold == 0] = 1000.0
    y[fold != 0] = 1.0
    ff = r39._fold_fits("C0", n, t, y, fold, block, block)
    oof = ff["oof"]
    expected = 1000.0 - np.median(y[fold != 0])
    m = fold == 0
    assert np.allclose(oof[m], expected, atol=1e-6)
    assert ff["oof_finite"] is True


def test_bootstrap_replicate_no_train_test_overlap():
    block = np.repeat(np.arange(12), 50)
    fold = r39._fold_ids(block)
    rng = np.random.default_rng(0)
    for _ in range(25):
        for d in r39.bootstrap_replicate_draws(block, fold, rng):
            tr_blocks = set(d["train_blocks"])
            te_blocks = set(d["test_blocks"])
            assert tr_blocks.isdisjoint(te_blocks)          # whole-block disjointness
            tr_folds = {int(fold[np.nonzero(block == b)[0][0]]) for b in tr_blocks}
            assert d["k"] not in tr_folds                    # no own test-fold block
            assert {int(fold[np.nonzero(block == b)[0][0]]) for b in te_blocks} == {d["k"]}
            # every resampled index belongs to a block on its own side
            assert set(np.unique(block[d["train_idx"]])) <= tr_blocks
            assert set(np.unique(block[d["test_idx"]])) <= te_blocks


def test_bootstrap_statistical_leakage_repro():
    # block 0 sits in fold 0 and carries a huge mean shift; every other block is ~0.
    block = np.repeat(np.arange(8), 40)
    fold = r39._fold_ids(block)
    n = np.zeros(block.size)
    t = np.zeros(block.size)
    y = np.where(fold == 0, 1000.0, 0.0)
    rng = np.random.default_rng(1)
    # for the fold-0 draw, the fit must not have seen fold-0 blocks -> OOF ~ 1000
    saw_shift_resid = False
    for _ in range(10):
        for d in r39.bootstrap_replicate_draws(block, fold, rng):
            if d["k"] != 0 or d["test_idx"].size == 0:
                continue
            A_tr = r39.model_design("C0", n[d["train_idx"]], t[d["train_idx"]])
            coef, *_ = r39.robust_fit(A_tr, y[d["train_idx"]])
            resid = y[d["test_idx"]] - coef[0]
            assert np.all(resid > 500.0)   # leaked fit would drive this to ~0
            saw_shift_resid = True
    assert saw_shift_resid


def test_rank_and_condition_gates():
    collinear = np.column_stack([np.ones(500), np.arange(500.0), np.arange(500.0)])
    assert r39.design_rank_full(collinear) is False
    ok, reason = r39._fold_gate(collinear, np.ones(500))
    assert ok is False
    good = r39.model_design("C3", np.linspace(-0.004, 0.004, 500),
                            np.random.default_rng(0).uniform(-0.5, 0.5, 500))
    assert r39.design_rank_full(good) is True
    ok2, _ = r39._fold_gate(good, np.ones(500))
    assert ok2 is True


def test_fold_status_three_pass_one_fail_admitted():
    # t has variation ONLY in fold 1 -> fold 1's C1 training design is rank deficient,
    # the other three folds gate-pass (M2: >=3/4 admitted).
    rng = np.random.default_rng(0)
    n = rng.uniform(-0.003, 0.003, 1200)
    block = np.arange(1200) // 30
    fold = r39._fold_ids(block)
    t = np.where(fold == 1, rng.uniform(-0.5, 0.5, 1200), 0.0)
    y = 5.0 * t + rng.normal(0.0, 0.5, 1200)
    ff = r39._fold_fits("C1", n, t, y, fold, block, block // 2)
    assert ff["n_gate_ok"] == 3
    assert ff["oof_finite"] is True
    assert sum(1 for f in ff["folds"] if not f["gate_ok"]) == 1


def test_fold_status_two_pass_refused():
    # t varies only in fold 1 and n only in fold 2 -> C3 training design is rank
    # deficient for BOTH fold 1 and fold 2 -> only 2 gate-passing folds (refused).
    rng = np.random.default_rng(0)
    block = np.arange(1200) // 30
    fold = r39._fold_ids(block)
    t = np.where(fold == 1, rng.uniform(-0.5, 0.5, 1200), 0.0)
    n = np.where(fold == 2, rng.uniform(-0.003, 0.003, 1200), 0.0)
    y = 5.0 * t + 5.0 * n + rng.normal(0.0, 0.5, 1200)
    ff = r39._fold_fits("C3", n, t, y, fold, block, block // 2)
    assert ff["n_gate_ok"] == 2
    assert ff["n_gate_ok"] < r39.MIN_ADMITTED_FOLDS


def test_c4_relative_rule():
    models = {
        "C0": {"oof": {"red_sigma": None, "sigma_oof": 2.0}, "admitted": True, "axes": {}},
        "C3": {"oof": {"red_sigma": 0.5, "sigma_oof": 1.0}, "admitted": True, "axes": {}},
        "C4": {"oof": {"red_sigma": 0.5, "sigma_oof": 0.5}, "admitted": True,
               "axes": {"q_n": {"ok": True, "sign": 1.0}}},
    }
    r39._apply_c4_rule(models)
    assert models["C4"]["admitted"] is True
    assert models["C4"]["q_n_state"] == "ADMITTED"
    # C3 not admitted/valid -> NOT_IDENTIFIABLE
    models["C3"]["admitted"] = False
    models["C4"]["admitted"] = True
    r39._apply_c4_rule(models)
    assert models["C4"]["q_n_state"] == "NOT_IDENTIFIABLE"
    assert models["C4"]["c4_reason"] == "C3_not_admitted"
    # weak OVERALL reduction vs C0 -> NOT_IDENTIFIABLE
    models["C3"]["admitted"] = True
    models["C4"]["oof"]["red_sigma"] = 0.1
    models["C4"]["admitted"] = True
    r39._apply_c4_rule(models)
    assert models["C4"]["q_n_state"] == "NOT_IDENTIFIABLE"
    assert models["C4"]["c4_reason"] == "overall_reduction_below_0.20"
    # weak relative reduction BEYOND C3 -> NOT_IDENTIFIABLE
    models["C4"]["oof"]["red_sigma"] = 0.5
    models["C4"]["oof"]["sigma_oof"] = 0.95
    models["C4"]["admitted"] = True
    r39._apply_c4_rule(models)
    assert models["C4"]["q_n_state"] == "NOT_IDENTIFIABLE"
    assert models["C4"]["c4_reason"] == "relative_beyond_c3_below_0.20"
    # zero C3 denominator -> NOT_IDENTIFIABLE
    models["C3"]["oof"]["sigma_oof"] = 0.0
    models["C4"]["oof"]["sigma_oof"] = 0.5
    models["C4"]["admitted"] = True
    r39._apply_c4_rule(models)
    assert models["C4"]["q_n_state"] == "NOT_IDENTIFIABLE"
    assert models["C4"]["c4_reason"] == "c3_denominator"
    # C4 base gates fail -> NOT_ADMITTED
    models["C4"]["admitted"] = False
    r39._apply_c4_rule(models)
    assert models["C4"]["q_n_state"] == "NOT_ADMITTED"


# ---------------------------------------------------------------------------
# 9. Bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_deterministic_and_block_resampling():
    n = np.zeros(600)
    t = np.zeros(600)
    block = np.arange(600) // 100  # 6 blocks
    y = np.where(block < 3, 0.0, 10.0) + 0.01 * np.arange(600)
    fold = block % 4
    a = r39.bootstrap_relation(n, t, y, block, fold, n_boot=40, seed=0)
    b = r39.bootstrap_relation(n, t, y, block, fold, n_boot=40, seed=0)
    assert a["c0_se"] == b["c0_se"]
    assert r39.dumps_strict(a["coef_se"]) == r39.dumps_strict(b["coef_se"])
    assert a["resampling"] == "whole_tangent_block_leak_free_per_fold"
    assert a["fold_aggregate"] == "mean_of_finite_fold_coefficients"
    assert len(a["_coef_draws"]["C0"]) == 40


def test_bootstrap_reports_effective_b_and_seed():
    s = _synthetic_samples("tangent")
    rec = r39.analyse_one_relation(_rel_dict(s), _meta(), n_boot=11)
    assert rec["models"]["C1"]["coef_se"]["b_t"] is not None


# ---------------------------------------------------------------------------
# 10. Metrics
# ---------------------------------------------------------------------------

def test_p90_is_non_centred():
    x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    assert r39._p90_abs(x) == pytest.approx(np.percentile(np.abs(x), 90.0))
    centred = np.percentile(np.abs(x - np.median(x)), 90.0)
    # a shifted array shows the two differ
    y = x + 100.0
    assert r39._p90_abs(y) == pytest.approx(np.percentile(np.abs(y), 90.0))
    assert r39._p90_abs(y) != pytest.approx(centred)


def test_rms0_statistic():
    x = np.array([3.0, 4.0])
    assert r39._rms0(x) == pytest.approx(np.sqrt((9.0 + 16.0) / 2.0))


# ---------------------------------------------------------------------------
# 11. Relation aggregate / panel
# ---------------------------------------------------------------------------

def _chan(state, sig=None, model=None):
    return {"state": state}


def _model(sig_q=None, b_t=None, b_n=None):
    axes = {}
    if b_t is not None:
        axes["b_t"] = {"ok": True, "sign": b_t}
    if b_n is not None:
        axes["b_n"] = {"ok": True, "sign": b_n}
    if sig_q is not None:
        axes["q_n"] = {"ok": True, "sign": sig_q}
    return {"axes": axes}


def test_one_channel_unexplained_hard_blocker():
    states = [{"state": "UNEXPLAINED_STRUCTURE"}, {"state": "SPATIAL_STABLE"},
              {"state": "SPATIAL_STABLE"}]
    models = [{"axes": {}}, _model(b_t=1.0), _model(b_t=1.0)]
    agg = r39.aggregate_relation(states, models)
    assert agg["state"] == "UNEXPLAINED_STRUCTURE"
    assert r39.FLAG_UNEXPLAINED_CHANNEL in agg["flags"]


def test_relation_2_of_3_vote():
    states = [{"state": "SPATIAL_STABLE"}, {"state": "SPATIAL_STABLE"},
              {"state": "NEGLIGIBLE"}]
    models = [_model(b_t=1.0), _model(b_t=1.0), {"axes": {}}]
    agg = r39.aggregate_relation(states, models)
    assert agg["state"] == "SPATIAL_STABLE"
    assert agg["spatial_signature"] == (("t",), (1,))


def _rel(u, v, orient, state, sig=None):
    return {"u": u, "v": v, "orientation": orient, "state": state,
            "spatial_signature": sig, "flags": []}


def test_panel_spatial_confirmed_requires_both_orientations():
    sig = ([["t"], [1]])
    rels = [
        _rel("a", "b", "h", "SPATIAL_STABLE", sig),
        _rel("c", "d", "h", "SPATIAL_STABLE", sig),
        _rel("e", "f", "v", "SPATIAL_STABLE", sig),
        _rel("g", "h", "v", "SPATIAL_STABLE", sig),
    ]
    panel = r39.panel_state(rels)
    assert panel["panel_state"] == "SPATIAL_CONFIRMED"
    assert panel["r39b_authorized"] is True
    assert panel["r39b_track"] == "B1"
    # only one orientation -> opens neither
    rels2 = [r for r in rels if r["orientation"] == "h"] + \
        [_rel("e", "f", "v", "CONSTANT_ONLY"), _rel("g", "h", "v", "CONSTANT_ONLY")]
    assert r39.panel_state(rels2)["panel_state"] == "INCONCLUSIVE"
    assert r39.panel_state(rels2)["r39b_authorized"] is False


def test_panel_constant_only_mix_opens_b0():
    rels = [_rel("a", "b", "h", "CONSTANT_ONLY"), _rel("c", "d", "h", "CONSTANT_ONLY"),
            _rel("e", "f", "v", "CONSTANT_ONLY"), _rel("g", "h", "v", "CONSTANT_ONLY"),
            _rel("i", "j", "v", "NEGLIGIBLE")]
    panel = r39.panel_state(rels)
    assert panel["panel_state"] == "CONSTANT_ONLY"
    assert panel["r39b_authorized"] is True and panel["r39b_track"] == "B0"


def test_panel_negligible_and_unexplained_precedence():
    allneg = [_rel("a", "b", "h", "NEGLIGIBLE"), _rel("c", "d", "v", "NEGLIGIBLE")]
    assert r39.panel_state(allneg)["panel_state"] == "NEGLIGIBLE"
    assert r39.panel_state(allneg)["r39b_authorized"] is False
    unmixed = [_rel("a", "b", "h", "SPATIAL_STABLE", ([["t"], [1]])),
               _rel("c", "d", "h", "SPATIAL_STABLE", ([["t"], [1]])),
               _rel("e", "f", "v", "SPATIAL_STABLE", ([["t"], [1]])),
               _rel("g", "h", "v", "SPATIAL_STABLE", ([["t"], [1]])),
               _rel("i", "j", "v", "UNEXPLAINED_STRUCTURE")]
    panel = r39.panel_state(unmixed)
    assert panel["panel_state"] == "UNEXPLAINED_STRUCTURE"
    assert panel["r39b_authorized"] is False and panel["r39b_track"] == "NONE"


def test_panel_inconclusive_on_not_run_and_mixed():
    mixed = [_rel("a", "b", "h", "SPATIAL_STABLE", ([["t"], [1]])),
             _rel("c", "d", "v", "CONSTANT_ONLY")]
    assert r39.panel_state(mixed)["panel_state"] == "INCONCLUSIVE"
    assert r39.panel_state(mixed)["r39b_track"] == "NONE"
    nr = [_rel("a", "b", "h", "CONSTANT_ONLY")]
    assert r39.panel_state(nr, not_run=["x"])["panel_state"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# 12. Universal attribution separation
# ---------------------------------------------------------------------------

def test_universal_support_not_available(capture_dir):
    result = _run_validated(capture_dir, pathlib.Path("."))
    for ch in result["analysis"]["channels"]:
        assert ch["nuisance_covariates"] == "NOT_AVAILABLE"
        assert ch["SUPPORT_NOT_AVAILABLE"] is True
        assert ch["causal_attribution_state"] == "NOT_AVAILABLE"
    for rel in result["analysis"]["relations"]:
        assert rel["nuisance_covariates"] == "NOT_AVAILABLE"
        assert rel["SUPPORT_NOT_AVAILABLE"] is True
        assert rel["causal_attribution_state"] == "NOT_AVAILABLE"


# ---------------------------------------------------------------------------
# 13. End-to-end report + privacy + determinism
# ---------------------------------------------------------------------------

def test_end_to_end_report_strict_privacy_and_panel(capture_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(r39, "BOOTSTRAP_B_DEFAULT", 8)
    out = tmp_path / "reports" / "r.json"
    rc = r39.main(["--capture-dir", str(capture_dir), "--out", str(out),
                   "--expect-identity", "any", "--determinism-check"])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["schema"] == r39.SCHEMA_NAME
    assert report["qualification"] in ("DEV_NOT_QUALIFICATION", "DEV_GENERIC")
    assert report["expect_identity"] == "any"
    assert report["scientific_acceptance"] is False
    assert report["correction_applied"] is False
    assert set(report["r39b_track"] in ("NONE", "B0", "B1") for _ in [0]) == {True}
    assert report["contextual_final_fits_controls"]["state"] == "NOT_RUN"
    assert report["runtime"]["bootstrap"]["configured_B"] == r39.BOOTSTRAP_B_DEFAULT
    assert r39.scan_path_leaks(report) == []
    # deterministic ordering of relations/channels
    keys = [(r["orientation"], r["u"], r["v"]) for r in report["relations"]]
    assert keys == sorted(keys)


def test_e2e_spatial_signal_opens_b1(tmp_path, monkeypatch):
    monkeypatch.setattr(r39, "BOOTSTRAP_B_DEFAULT", 24)

    def signal(n, t, c):
        return 20.0 * t / 0.5
    cap = build_capture(tmp_path / "cap", nx=2, ny=2, W=400, H=400,
                        signal=signal, noise_sigma=1.0, n_target=1500)
    result = r39.run(cap, expect="any", policy=_policy(cap), n_boot=24)
    panel = result["analysis"]["panel"]
    # a strong shared tangent slope in BOTH orientations must confirm spatially
    assert panel["panel_state"] == "SPATIAL_CONFIRMED"
    assert panel["r39b_track"] == "B1"


def test_e2e_report_has_no_absolute_path(tmp_path, monkeypatch, capture_dir):
    monkeypatch.setattr(r39, "BOOTSTRAP_B_DEFAULT", 8)
    out = tmp_path / "r.json"
    rc = r39.main(["--capture-dir", str(capture_dir), "--out", str(out),
                   "--expect-identity", "any"])
    assert rc == 0
    text = out.read_text()
    assert str(capture_dir) not in text
    assert "/home/" not in text
    assert r39.scan_path_leaks(json.loads(text)) == []


def test_embedded_leak_is_redacted_before_write(capture_dir, tmp_path, monkeypatch):
    # inject an absolute path into the report via a monkeypatched field
    monkeypatch.setattr(r39, "BOOTSTRAP_B_DEFAULT", 8)
    orig = r39.build_report

    def _bad(identity, analysis, **kw):
        rep = orig(identity, analysis, **kw)
        rep["leaked"] = "/home/private/secret"
        return rep

    monkeypatch.setattr(r39, "build_report", _bad)
    out = tmp_path / "r.json"
    rc = r39.main(["--capture-dir", str(capture_dir), "--out", str(out),
                   "--expect-identity", "any"])
    assert rc == 0
    report = json.loads(out.read_text())
    # the absolute path never survives into the durable artifact (fail-closed)
    assert "/home/private/secret" not in out.read_text()
    assert report["leaked"] == r39.REDACTED_PATH
    assert r39.scan_path_leaks(report) == []


def test_minimal_failure_record_is_path_free():
    rec = r39.minimal_failure_record(stage="privacy", leak_locations=[
        {"json_path": "$.x", "category": "posix_absolute"}])
    assert rec["status"] == "FAILED"
    assert r39.scan_path_leaks(rec) == []


# ---------------------------------------------------------------------------
# 14. Failure semantics / write safety
# ---------------------------------------------------------------------------

def test_unexpected_exception_is_failed_nonzero(capture_dir, tmp_path, monkeypatch, capsys):
    def _boom(*a, **k):
        raise RuntimeError("programming error")

    monkeypatch.setattr(r39, "analyse_records", _boom)
    out = tmp_path / "r.json"
    rc = r39.main(["--capture-dir", str(capture_dir), "--out", str(out),
                   "--expect-identity", "any"])
    assert rc == 1
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["status"] == "FAILED"
    assert printed["wrote_report"] is False


def test_no_writes_under_input_root(capture_dir):
    before = sorted(p.name for p in capture_dir.iterdir())
    r39.run(capture_dir, expect="any", policy=_policy(capture_dir), n_boot=8)
    after = sorted(p.name for p in capture_dir.iterdir())
    assert before == after


def test_predicted_imprint_is_present_and_non_driving(tmp_path, monkeypatch):
    monkeypatch.setattr(r39, "BOOTSTRAP_B_DEFAULT", 12)
    cap = build_capture(tmp_path / "cap", nx=2, ny=2, W=400, H=400, n_target=800)
    result = r39.run(cap, expect="any", policy=_policy(cap), n_boot=12)
    imp = result["analysis"]["predicted_feather_imprint"]
    assert imp["state"] in ("PREDICTED_OK", "NOT_RUN", "NOT_APPLICABLE")
    if imp["state"] == "PREDICTED_OK":
        assert imp["label"] == "PREDICTED"


# ---------------------------------------------------------------------------
# 15. H1: weak-but-significant must never promote (RED_SIGMA_MIN admission)
# ---------------------------------------------------------------------------

def _weak_sample(n=2500, slope=0.30, seed=5):
    rng = np.random.default_rng(seed)
    block = np.arange(n) // 60
    ncoord = rng.uniform(-0.0038, 0.0038, n)
    tcoord = rng.uniform(-0.5, 0.5, n)
    y = slope * tcoord + rng.normal(0.0, 1.0, n)
    return {"n": ncoord, "t": tcoord, "s": ncoord * 200.0, "y": y, "tangent_block": block}


def _weak_rel():
    s = _weak_sample()
    return {"u": "r0000c0000", "v": "r0000c0001", "orientation": "h", "channel": 0,
            "n": s["n"], "t": s["t"], "s": s["s"], "e_post": s["y"], "e_pre": s["y"],
            "tangent_block": s["tangent_block"], "normal_block": s["tangent_block"] // 2,
            "sigma_ref": 1.0, "n_train": 5000, "effective_train": 5000.0,
            "artifact_tangent_blocks": 40, "artifact_normal_blocks": 4}


def test_h1_weak_but_significant_never_spatial():
    rec = r39.analyse_one_relation(_weak_rel(), _meta(), n_boot=30)
    assert rec["state"] not in ("SPATIAL_STABLE", "CURVATURE")
    c1 = rec["models"]["C1"]
    assert c1["admitted"] is False
    red = c1["oof"]["red_sigma"]
    assert red is None or red < r39.RED_SIGMA_MIN


def test_h1_red_sigma_bound_is_fail_closed_in_classify():
    # a channel whose best model is "admitted" but with red_sigma < 0.20 must NOT
    # classify as SPATIAL_STABLE (independent fail-closed recheck).
    values = {
        "sigma_ref": 1.0, "e_post": np.zeros(300), "e_pre": np.zeros(300),
        "n_holdout": 300, "effective_train": 5000.0,
        "tangent_blocks": 40, "normal_blocks": 40,
        "tangent_blocks_artifact": 40, "normal_blocks_artifact": 4,
        "c0_se": 0.001,
        "models": {
            "C0": {"admitted": True, "oof": {"sigma_oof": 1.0, "rms0_oof": 1.0, "red_sigma": 0.0}},
            "C1": {"admitted": True, "oof": {"sigma_oof": 0.9, "rms0_oof": 0.9, "red_sigma": 0.02}},
            "C2": {"admitted": False, "oof": {"sigma_oof": None, "rms0_oof": None, "red_sigma": None}},
            "C3": {"admitted": False, "oof": {"sigma_oof": None, "rms0_oof": None, "red_sigma": None}},
            "C4": {"admitted": False, "oof": {"sigma_oof": None, "rms0_oof": None, "red_sigma": None}},
        },
    }
    cls = r39.classify_channel(values)
    assert cls["state"] not in ("SPATIAL_STABLE", "CURVATURE")


def test_h1_weak_panel_never_confirms(tmp_path, monkeypatch):
    monkeypatch.setattr(r39, "BOOTSTRAP_B_DEFAULT", 16)
    sig = lambda n, t, c: 0.35 * t  # noqa: E731  (weak, near-significant, both orientations)
    cap = build_capture(tmp_path / "cap", nx=2, ny=2, W=400, H=400, signal=sig,
                        noise_sigma=1.0, n_target=1500)
    res = r39.run(cap, expect="any", policy=_policy(cap), n_boot=16)
    panel = res["analysis"]["panel"]
    assert panel["panel_state"] != "SPATIAL_CONFIRMED"
    assert panel["r39b_authorized"] is False
    assert panel["r39b_track"] != "B1"


# ---------------------------------------------------------------------------
# 16. expect_identity / DEV_GENERIC authorization separation
# ---------------------------------------------------------------------------

def test_any_mode_is_dev_generic_and_never_authorizes(capture_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(r39, "BOOTSTRAP_B_DEFAULT", 8)
    out = tmp_path / "r.json"
    rc = r39.main(["--capture-dir", str(capture_dir), "--out", str(out),
                   "--expect-identity", "any"])
    assert rc == 0
    rep = json.loads(out.read_text())
    assert rep["expect_identity"] == "any"
    assert rep["qualification"] == "DEV_GENERIC"
    assert rep["scientific_acceptance"] is False
    assert rep["r39b_authorized"] is False
    assert rep["r39b_track"] == "NONE"
    assert rep["identity"]["expect_identity"] == "any"


def test_checked_mode_refuses_synthetic_capture(capture_dir, tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = r39.main(["--capture-dir", str(capture_dir), "--out", str(out)])
    assert rc == 1
    assert not out.exists()


def test_checked_mode_may_authorize_but_any_cannot(capture_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(r39, "BOOTSTRAP_B_DEFAULT", 8)
    # force a SPATIAL_CONFIRMED panel analytically and check mode gating
    ident = r39.validate_identity(capture_dir, expect="any")
    sig = ([["t"], [1]])
    analysis = {
        "panel": {"panel_state": "SPATIAL_CONFIRMED", "r39b_authorized": True, "r39b_track": "B1"},
        "channels": [], "relations": [], "orientation": {}, "structural_validation": {},
        "predicted_feather_imprint": {"state": "NOT_RUN"}, "n_resumed": 0, "n_new": 0,
    }
    args_any = types.SimpleNamespace(expect_identity="any", _effective_b=8,
                                     _scratch_enabled=False, _scratch_identity=None)
    rep_any = r39.build_report(ident, analysis, args=args_any,
                               policy=_policy(capture_dir), component_checks={}, t_start=0.0)
    assert rep_any["r39b_authorized"] is False and rep_any["r39b_track"] == "NONE"
    args_checked = types.SimpleNamespace(expect_identity="checked", _effective_b=8,
                                         _scratch_enabled=False, _scratch_identity=None)
    rep_checked = r39.build_report(ident, analysis, args=args_checked,
                                   policy=_policy(capture_dir), component_checks={}, t_start=0.0)
    assert rep_checked["r39b_authorized"] is True and rep_checked["r39b_track"] == "B1"


# ---------------------------------------------------------------------------
# 17. Checkpoint / resume (deterministic; binding-bound; fail-closed)
# ---------------------------------------------------------------------------

def test_checkpoint_partial_then_resume_equals_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(r39, "BOOTSTRAP_B_DEFAULT", 12)
    cap = build_capture(tmp_path / "cap", nx=2, ny=2, W=400, H=400, n_target=500)
    scratch = tmp_path / "scratch"
    out = tmp_path / "out" / "r.json"
    rc = r39.main(["--capture-dir", str(cap), "--out", str(out), "--expect-identity", "any",
                   "--scratch-dir", str(scratch), "--max-seconds", "0.000001"])
    assert rc == 3                     # distinct non-success checkpoint status
    assert not out.exists()            # NO final report while partial
    assert list(scratch.glob("r39a_ckpt_*.json"))
    rc2 = r39.main(["--capture-dir", str(cap), "--out", str(out), "--expect-identity", "any",
                    "--scratch-dir", str(scratch)])
    assert rc2 == 0
    resumed = json.loads(out.read_text())
    assert resumed["runtime"]["n_records_resumed"] > 0
    assert resumed["runtime"]["checkpoint"]["enabled"] is True
    out2 = tmp_path / "fresh" / "r.json"
    rc3 = r39.main(["--capture-dir", str(cap), "--out", str(out2), "--expect-identity", "any",
                    "--scratch-dir", ""])
    assert rc3 == 0
    fresh = json.loads(out2.read_text())
    assert r39._canonical_metrics(resumed) == r39._canonical_metrics(fresh)
    # the scratch path itself is never persisted
    assert str(scratch) not in out.read_text()


def test_checkpoint_binding_mismatch_refuses(tmp_path, monkeypatch):
    cap = build_capture(tmp_path / "cap", nx=2, ny=2, W=400, H=400, n_target=300)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    ident = r39.validate_identity(cap, expect="any")
    binding = r39._checkpoint_binding(ident, effective_b=8)
    path = r39._ckpt_path(scratch, binding)
    path.write_text(json.dumps({"schema": r39.CKPT_SCHEMA, "version": r39.CKPT_VERSION,
                                "binding": dict(binding, artifact_sha256="0" * 64),
                                "records": {}}))
    with pytest.raises(r39.ResolutionError):
        r39._load_checkpoint(scratch, binding)


def test_checkpoint_malformed_refuses(tmp_path):
    cap = build_capture(tmp_path / "cap", nx=2, ny=2, W=400, H=400, n_target=300)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    ident = r39.validate_identity(cap, expect="any")
    binding = r39._checkpoint_binding(ident, effective_b=8)
    r39._ckpt_path(scratch, binding).write_text("{ not json")
    with pytest.raises(r39.ResolutionError):
        r39._load_checkpoint(scratch, binding)


def test_checkpoint_symlink_refuses(tmp_path):
    cap = build_capture(tmp_path / "cap", nx=2, ny=2, W=400, H=400, n_target=300)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    ident = r39.validate_identity(cap, expect="any")
    binding = r39._checkpoint_binding(ident, effective_b=8)
    target = tmp_path / "elsewhere.json"
    target.write_text("{}")
    r39._ckpt_path(scratch, binding).symlink_to(target)
    with pytest.raises(r39.ResolutionError):
        r39._load_checkpoint(scratch, binding)


def test_scratch_inside_protected_root_refused(tmp_path, capsys):
    cap = build_capture(tmp_path / "cap")
    rc = r39.main(["--capture-dir", str(cap), "--out", str(tmp_path / "out" / "r.json"),
                   "--expect-identity", "any", "--scratch-dir", str(cap)])
    assert rc == 1


def test_checkpoint_rebuild_flag_discards_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(r39, "BOOTSTRAP_B_DEFAULT", 4)
    cap = build_capture(tmp_path / "cap", nx=2, ny=2, W=400, H=400, n_target=300)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "r39a_ckpt_deadbeef.json").write_text("{}")   # unrelated stale file
    rc = r39.main(["--capture-dir", str(cap), "--out", str(tmp_path / "out" / "r.json"),
                   "--expect-identity", "any", "--scratch-dir", str(scratch),
                   "--rebuild-checkpoint"])
    assert rc == 0
