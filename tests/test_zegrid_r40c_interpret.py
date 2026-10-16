"""ZM-ZEGRID-R40-C focused tests — offline interpretation of the capture artifact.

Covers, WITHOUT any real corpus / real capture / FITS pixel read:

* module-scope import safety (no ``zemosaic`` at import time) + CLI help;
* the identity-first, fail-closed path on a **synthetic committed generation**
  (index sha == manifest cross-link == verification report, capture_id, panel
  38/114, ``n_dropped == 0``, ``holdout_capped == false``, ``qualification EXACT``,
  ``production_wired``) and each material mismatch/refusal;
* relation-first aggregation (2-of-3 channel rule, ≥2 relations/orientation);
* the whole-tangent-block leak-free bootstrap contrast (determinism, floors);
* the T1..T5 mechanics (T4 is NOT_AVAILABLE and never proxied; T3/T5 context-only);
* strict JSON, atomic write, no absolute private path in a durable value;
* the NO-FINAL-FITS / NO-PIXEL guarantee (``.fits`` opens are trapped).

Explicit NON-CLAIMS: this suite does NOT re-run the capture, does NOT read the real
R40-C artifact, does NOT open the final product FITS, claims no Windows816 mechanism,
changes no tolerance/threshold/gate, and leaves the capture default-OFF.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

from zemosaic.core.zegrid import r40_capture as rc

_TOOL_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "r40c_interpret.py"


def _load_tool():
    name = "r40c_interpret"
    spec = importlib.util.spec_from_file_location(name, _TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


r40c = _load_tool()

# --- synthetic capture constants --------------------------------------------
REPO_HEAD = "a" * 40
CAP_ID = "deadbeef" * 8  # 64 hex
NX, NY = 2, 2
CANVAS_W, CANVAS_H = 800, 800
OFFSETS = {
    "r0000c0000": [3.5, -2.25, 1.0],
    "r0000c0001": [-1.5, 4.0, -0.75],
    "r0001c0000": [0.5, -3.0, 2.25],
    "r0001c0001": [7.0, 1.25, -4.5],
}


def _cells():
    return [[f"r{r:04d}c{c:04d}" for c in range(NX)] for r in range(NY)]


def _relations():
    cells = _cells()
    rels = []
    for r in range(NY):
        for c in range(NX - 1):
            rels.append((cells[r][c], cells[r][c + 1], "h"))
    for r in range(NY - 1):
        for c in range(NX):
            rels.append((cells[r][c], cells[r + 1][c], "v"))
    return rels


def _pixels(orientation, sign):
    """64 odd/odd pixels split +1 (neff_u>neff_v) / -1 (neff_u<neff_v)."""
    tang = [4, 5, 6, 7, 12, 13, 14, 15]   # rows/cols in odd blocks (1 and 3)
    near_cols = [396, 397, 398, 399]      # odd block 99
    far_cols = [404, 405, 406, 407]       # odd block 101
    out = []
    if orientation == "h":
        half = len(tang) * len(near_cols)  # 32
        for i, gx in enumerate(near_cols):
            for gy in tang:
                out.append((gx, gy, 1 if sign != 0 else 0))
        for gx in far_cols:
            for gy in tang:
                out.append((gx, gy, -1 if sign != 0 else 0))
        # ensure the sign partition is balanced for the "associated" scenario
        if sign != 0:
            out = [(gx, gy, (1 if (gx < 400) else -1)) for gx, gy, _ in out]
    else:
        for gy in near_cols:
            for gx in tang:
                out.append((gx, gy, 1 if sign != 0 else 0))
        for gy in far_cols:
            for gx in tang:
                out.append((gx, gy, -1 if sign != 0 else 0))
        if sign != 0:
            out = [(gx, gy, (1 if (gy < 400) else -1)) for gx, gy, _ in out]
    return out


def _make_header(u, v, orientation):
    return {
        "gauge_mode": "per_cell",
        "stack_tokens": {
            "normalization": "sky_mean", "weighting": "noise_variance",
            "rejection": "kappa_sigma", "combine": "mean", "taper": "footprint",
            "taper_px": 8.0, "taper_floor": 0.0, "equalize_rgb": False,
            "sigma_low": 3.0, "sigma_high": 3.0, "winsor": None,
        },
        "reference_frame_id": {u: "frame_a", v: "frame_b"},
        "frame_order_digest": {u: "0" * 64, v: "1" * 64},
        "excluded": {u: [], v: []},
        "measurement_px": 16, "feather_px": 8, "halo_px": 8, "block_px": 4,
    }


def _records(u, v, orientation, sign):
    records = []
    for gx, gy, s in _pixels(orientation, sign):
        for ch in range(3):
            off_u = OFFSETS[u][ch]
            off_v = OFFSETS[v][ch]
            d_target = 3.0 * s
            pre_v = np.float32(100.0)
            pre_u = np.float32(100.0 + d_target - (off_u - off_v))
            records.append({
                "gy": gy, "gx": gx, "channel": ch,
                "sci_u_pre": float(pre_u), "sci_v_pre": float(pre_v),
                "sci_u_post": float(np.float64(pre_u) + off_u),
                "sci_v_post": float(np.float64(pre_v) + off_v),
                "valid_u": True, "valid_v": True,
                "estimator_weight_sum_u": 10.0 + s, "estimator_weight_sum_v": 10.0 - s,
                "surviving_sample_count_u": 5, "surviving_sample_count_v": 5,
                "support_w1_u": 0.5, "support_w1_v": 0.5,
                "support_w2_u": 0.5, "support_w2_v": 0.5,
                "n_eff_support_u": 2.0 + s, "n_eff_support_v": 2.0 - s,
                "ramp_u": 1.0 if s > 0 else 0.5, "ramp_v": 0.5 if s > 0 else 1.0,
            })
    return records


def _pixel_coords(orientation):
    return [(gx, gy) for gx, gy, _ in _pixels(orientation, 1)]


def _build_capture(root: pathlib.Path, *, sign: int = 1, production_wired: bool = True,
                   tamper: str | None = None) -> dict:
    root = pathlib.Path(root)
    output_root = root
    shards = []
    for u, v, orientation in _relations():
        shards.append(rc.RelationShard(
            u=u, v=v, orientation=orientation,
            header=_make_header(u, v, orientation),
            records=tuple(_records(u, v, orientation, sign)),
            candidate_coords=tuple(_pixel_coords(orientation)),
        ))
    identity = rc.make_capture_identity(
        capture_id=CAP_ID, source_set_digest="ab" * 32, n_frames=10,
        layout={"nx": NX, "ny": NY},
        canvas={"width": CANVAS_W, "height": CANVAS_H, "id": "c0ffee01"},
        gauge_mode="per_cell", repo_head=REPO_HEAD, production_wired=production_wired,
    )
    config = rc.R40CaptureConfig(enabled=True, output_root=output_root,
                                 disk_budget_bytes=64 * 1024 * 1024)
    written = rc.write_capture_generation(output_root, shards=shards, identity=identity,
                                          config=config, expected_production_wired=True)
    gen_dir = output_root / written["generation_dir"]
    index_path = gen_dir / "index.json"
    index_sha = rc.sha256_bytes(index_path.read_bytes())
    index_bytes = int(index_path.stat().st_size)
    n_records = 0
    for entry in json.loads(index_path.read_text(encoding="utf-8"))["shards"]:
        n_records += int(entry["count"])
    block = {
        "capture_id": CAP_ID, "filename": written["generation_dir"],
        "sha256": index_sha, "bytes": index_bytes,
    }
    manifest = {
        "engine": "zegrid",
        "cell_feather": {"enabled": True, "channels": 3, "halo_px": 8,
                         "offsets": {k: list(v) for k, v in OFFSETS.items()}},
        "r40_diagnostic_capture": block,
    }
    verification = {
        "schema": "zemosaic.zegrid.r40c.capture_verification",
        "status": "VERIFIED",
        "capture": {
            "capture_id": CAP_ID, "generation_sha256": index_sha,
            "index_sha256": index_sha, "generation_bytes": index_bytes,
            "n_shards": 4, "n_records": n_records,
        },
        "verification": {
            "ok": True, "problems": [], "expected_production_wired": True,
            "checked_records": n_records,
        },
    }
    if tamper == "index":
        text = index_path.read_text(encoding="utf-8")
        index_path.write_text(text.replace('"capture_id": "%s"' % CAP_ID,
                                           '"capture_id": "ffffffffffffffff"', 1),
                              encoding="utf-8")
    if tamper == "capture_id":
        manifest["r40_diagnostic_capture"]["capture_id"] = "0" * 64
    if tamper == "sha":
        manifest["r40_diagnostic_capture"]["sha256"] = "0" * 64
    if tamper == "block":
        del manifest["r40_diagnostic_capture"]
    if tamper == "wire":
        index_doc = json.loads(index_path.read_text(encoding="utf-8"))
        index_doc["identity"]["production_wired"] = False
        index_path.write_text(json.dumps(index_doc), encoding="utf-8")
    (root / "zegrid_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "r40_capture_verification.json").write_text(json.dumps(verification), encoding="utf-8")
    return {"root": root, "index_sha256": index_sha, "index_bytes": index_bytes,
            "n_records": n_records, "gen_dir": gen_dir}


def _patch_constants(monkeypatch, built):
    monkeypatch.setattr(r40c, "EXPECTED_CAPTURE_ID", CAP_ID)
    monkeypatch.setattr(r40c, "EXPECTED_INDEX_SHA256", built["index_sha256"])
    monkeypatch.setattr(r40c, "EXPECTED_INDEX_BYTES", built["index_bytes"])
    monkeypatch.setattr(r40c, "EXPECTED_N_SHARDS", 4)
    monkeypatch.setattr(r40c, "EXPECTED_N_RECORDS", built["n_records"])
    monkeypatch.setattr(r40c, "EXPECTED_ORIENTATIONS", {"h": 2, "v": 2})
    monkeypatch.setattr(r40c, "EXPECTED_RELATION_CHANNELS", 12)
    monkeypatch.setattr(r40c, "CANVAS_LAYOUT", (NX, NY))
    monkeypatch.setattr(r40c, "EXPECTED_N_FRAMES", 10)


@pytest.fixture()
def capture(tmp_path, monkeypatch):
    built = _build_capture(tmp_path)
    _patch_constants(monkeypatch, built)
    return built


# ---------------------------------------------------------------------------
# 1. Import safety / CLI
# ---------------------------------------------------------------------------

def test_import_audit_no_zemosaic_at_module_scope():
    audit = r40c.import_audit()
    assert audit["engine_access_is_lazy"] is True
    assert audit["zemosaic_module_scope_imports"] == []
    assert not any(m.startswith("astropy") for m in audit["module_scope_imports"])


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        r40c.main(["--help"])
    assert exc.value.code == 0
    assert "capture-root" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 2. Identity-first, fail-closed
# ---------------------------------------------------------------------------

def test_identity_happy_path(capture):
    ident = r40c.verify_capture_identity(capture["root"])
    assert ident["capture_id"] == CAP_ID
    assert ident["index_sha256"] == capture["index_sha256"]
    assert ident["n_shards"] == 4
    assert ident["orientation_counts"] == {"h": 2, "v": 2}
    assert ident["production_wired"] is True
    assert ident["verification_ok"] is True
    assert ident["n_dropped"] == 0
    assert ident["holdout_capped"] is False
    assert ident["qualification"] == "EXACT"


@pytest.mark.parametrize("tamper,stop", [
    ("index", r40c.STOP_IDENTITY_MISMATCH),
    ("capture_id", r40c.STOP_IDENTITY_MISMATCH),
    ("sha", r40c.STOP_IDENTITY_MISMATCH),
    ("block", r40c.STOP_IDENTITY_MISMATCH),
    ("wire", r40c.STOP_IDENTITY_MISMATCH),
])
def test_identity_mismatches_refuse(tmp_path, monkeypatch, tamper, stop):
    built = _build_capture(tmp_path, tamper=tamper)
    _patch_constants(monkeypatch, built)
    with pytest.raises(r40c.Refusal) as exc:
        r40c.verify_capture_identity(built["root"])
    assert exc.value.stop == stop


def test_missing_manifest_refuses(tmp_path, monkeypatch):
    built = _build_capture(tmp_path)
    _patch_constants(monkeypatch, built)
    (built["root"] / "zegrid_manifest.json").unlink()
    with pytest.raises(r40c.Refusal) as exc:
        r40c.verify_capture_identity(built["root"])
    assert exc.value.stop == r40c.STOP_IDENTITY_MISMATCH


# ---------------------------------------------------------------------------
# 3. Relation-first aggregation
# ---------------------------------------------------------------------------

def test_two_of_three_channel_rule_promotes():
    agg = r40c.aggregate_relation_first([
        (r40c.STATE_DETECTED_ASSOCIATION, 1),
        (r40c.STATE_DETECTED_ASSOCIATION, 1),
        ("NO_EFFECT", 0),
    ])
    assert agg["state"] == r40c.STATE_DETECTED_ASSOCIATION
    assert agg["sign"] == 1


def test_two_of_three_channel_rule_conflict_no_promotion():
    agg = r40c.aggregate_relation_first([
        (r40c.STATE_DETECTED_ASSOCIATION, 1),
        (r40c.STATE_DETECTED_ASSOCIATION, -1),
        ("NO_EFFECT", 0),
    ])
    assert agg["state"] == "NO_EFFECT"


def test_orientation_requires_two_distinct_relations():
    rels = [
        {"orientation": "h", "relation_id": "a", "state": r40c.STATE_DETECTED_ASSOCIATION, "sign": 1},
        {"orientation": "h", "relation_id": "b", "state": r40c.STATE_DETECTED_ASSOCIATION, "sign": 1},
        {"orientation": "v", "relation_id": "c", "state": r40c.STATE_DETECTED_ASSOCIATION, "sign": 1},
    ]
    out = r40c.aggregate_orientation(rels)
    assert out["h"]["qualifies"] is True
    assert out["v"]["qualifies"] is False


# ---------------------------------------------------------------------------
# 4. Leak-free whole-tangent-block bootstrap contrast
# ---------------------------------------------------------------------------

def test_block_contrast_detects_and_is_deterministic():
    rng = np.random.default_rng(0)
    n = 400
    tb = np.repeat(np.arange(8), 50)
    ind = np.where(np.arange(n) % 2 == 0, 1, -1)
    y = np.where(ind > 0, 5.0, -5.0) + rng.normal(0, 0.01, n)
    a = r40c.block_contrast(y, ind, tb, n_boot=200, seed=0)
    b = r40c.block_contrast(y, ind, tb, n_boot=200, seed=0)
    assert a == b
    assert a["excludes_zero"] is True
    assert a["sign"] == 1


def test_block_contrast_null_does_not_exclude_zero():
    tb = np.repeat(np.arange(8), 50)
    ind = np.where(np.arange(400) % 2 == 0, 1, -1)
    y = np.zeros(400, dtype=np.float64)
    res = r40c.block_contrast(y, ind, tb, n_boot=100, seed=0)
    assert res["excludes_zero"] is False


def test_block_contrast_floors_return_none():
    tb = np.array([1, 1, 1, 1], dtype=np.int64)
    ind = np.array([1, 1, -1, -1], dtype=np.int64)
    y = np.array([1.0, 1.0, -1.0, -1.0])
    assert r40c.block_contrast(y, ind, tb, n_boot=10, seed=0) is None


# ---------------------------------------------------------------------------
# 5. Full analysis on the synthetic generation
# ---------------------------------------------------------------------------

def test_analysis_association_path(capture):
    analysis = r40c.analyse_capture(capture["root"], n_boot=100, seed=0)
    assert analysis["panel"]["panel_complete"] is True
    assert analysis["per_relation_channel_n"] == 12
    assert analysis["tests"]["T1"]["state"] == r40c.STATE_DETECTED_ASSOCIATION
    assert analysis["tests"]["T1"]["orientations_qualify"] == {"h": True, "v": True}
    assert analysis["stop"]["stop"] == r40c.STOP_NUISANCE_LINKED
    assert analysis["stop"]["observational_only"] is True
    assert analysis["stop"]["causal_attribution"] is False


def test_analysis_null_path(tmp_path, monkeypatch):
    built = _build_capture(tmp_path, sign=0)
    _patch_constants(monkeypatch, built)
    analysis = r40c.analyse_capture(built["root"], n_boot=100, seed=0)
    assert analysis["tests"]["T1"]["state"] == "NO_EFFECT"
    assert analysis["stop"]["stop"] == r40c.STOP_NO_EFFECT


def test_t4_is_not_available_and_never_proxied(capture):
    analysis = r40c.analyse_capture(capture["root"], n_boot=50, seed=0)
    t4 = analysis["tests"]["T4"]
    assert t4["state"] == r40c.STATE_NOT_IDENTIFIABLE
    assert t4["available"] is False
    assert t4["reachable"] is False
    assert t4["intervention_families"] == []


def test_t3_and_t5_are_context_only(capture):
    analysis = r40c.analyse_capture(capture["root"], n_boot=50, seed=0)
    assert analysis["tests"]["T3"]["state"] == r40c.STATE_CONTEXT
    assert analysis["tests"]["T5"]["state"] == r40c.STATE_CONTEXT


# ---------------------------------------------------------------------------
# 6. CLI / report writing
# ---------------------------------------------------------------------------

def test_cli_writes_strict_json_report(capture, tmp_path, capsys):
    out = tmp_path / "report" / "r40c.json"
    rc_code = r40c.main(["--capture-root", str(capture["root"]), "--out", str(out),
                         "--bootstrap-b", "50"])
    assert rc_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["mission"] == "ZM-ZEGRID-R40-C-INTERPRET"
    assert payload["qualification"] == "DEV_NOT_QUALIFICATION"
    assert payload["scientific_acceptance"] is False
    assert payload["correction_applied"] is False
    assert payload["stop_outcome"] == r40c.STOP_NUISANCE_LINKED
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    stdout = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(stdout)["wrote_report"] is True


def test_cli_refusal_writes_nothing(tmp_path, monkeypatch, capsys):
    built = _build_capture(tmp_path, tamper="capture_id")
    _patch_constants(monkeypatch, built)
    out = tmp_path / "report" / "r40c.json"
    code = r40c.main(["--capture-root", str(built["root"]), "--out", str(out)])
    assert code == 2
    assert not out.exists()
    assert json.loads(capsys.readouterr().out.strip())["stop"] == r40c.STOP_IDENTITY_MISMATCH


def test_report_has_no_absolute_private_path(capture, tmp_path):
    out = tmp_path / "report" / "r40c.json"
    r40c.main(["--capture-root", str(capture["root"]), "--out", str(out),
               "--bootstrap-b", "50"])
    text = out.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    assert str(capture["root"]) not in text
    leaks = r40c._load_r39a().scan_path_leaks(json.loads(text))
    assert leaks == []


# ---------------------------------------------------------------------------
# 7. No FITS / pixel read
# ---------------------------------------------------------------------------

def test_no_fits_or_pixel_read(capture, monkeypatch):
    real_open = builtins.open

    def guard(file, *args, **kwargs):
        if isinstance(file, (str, pathlib.Path)) and str(file).lower().endswith(".fits"):
            raise AssertionError("the interpretation tool must never open a FITS file")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guard)
    analysis = r40c.analyse_capture(capture["root"], n_boot=20, seed=0)
    assert analysis["panel"]["panel_complete"] is True


def test_expected_constants_are_the_real_artifact():
    assert r40c.EXPECTED_CAPTURE_ID == (
        "5dcef7b90a2eb109685e7f81fcc714c8ceb07e1ab562ab6747fd31b295848ec5")
    assert r40c.EXPECTED_INDEX_SHA256 == (
        "d4681ae937b3c8b0e1785010af669024496c7758106d15fe6f11a9fcb4799f89")
    assert r40c.EXPECTED_N_SHARDS == 38
    assert r40c.EXPECTED_N_RECORDS == 388887
    assert r40c.EXPECTED_ORIENTATIONS == {"h": 20, "v": 18}
    assert r40c.BOOTSTRAP_B == 2000
    assert r40c.BOOTSTRAP_SEED == 0
