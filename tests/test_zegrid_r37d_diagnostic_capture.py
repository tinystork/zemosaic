"""ZM-ZEGRID-R37-D focused tests — explicit pre-offset/pre-feather capture seam.

Covers the mandated R37-D behaviours:

* default OFF: no artifact, the diagnostic module is never called, existing
  outputs/manifests are unchanged;
* explicit ``per_cell`` ON: the accepted R37-C strict JSON artifact is published
  at the exact pre-offset/pre-feather point (atomic, no temp residue, digest and
  byte metadata match the file, relations match a fresh build from the same
  patches);
* hold-out cap semantics: a large cap gives ``EXACT``; a small cap yields
  ``RESEARCH_ONLY_HELDOUT_CAPPED`` and can never qualify ``ROBUST_PLAN``;
* ``auto`` resolves to ``per_cell`` and captures; explicit ``global`` and
  ``auto -> global`` fail LOUDLY before per-Cell stacking;
* mount segregation writes ONE artifact per actual output directory, no collision;
* publication failures are loud (no partial file, no residue);
* capture ON vs OFF is science/aesthetic/coverage BYTE-IDENTICAL and the R34c
  offsets are IDENTICAL; the pure assembler is array-equal (excluding the capture
  metadata);
* strict argument validation, no config/GUI activation, no correction applied.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import pathlib

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from types import SimpleNamespace

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import background_diagnostics as bgd
from zemosaic.core.zegrid import background_model as bm
from zemosaic.core.zegrid import cell_feather as zcf
from zemosaic.core.zegrid import geometry as zg
from zemosaic.zemosaic_stack_plan import FrameInfo

# The end-to-end runs write FITS via the production path, which emits astropy's
# ``Card is too long, comment will be truncated`` VerifyWarning on the synthetic
# WCS header. This is a PRE-EXISTING, non-scientific warning of the shared FITS
# write path (the accepted R34f e2e test trips it identically under ``-W error``),
# unrelated to the R37-D capture seam. Filter ONLY that exact warning so ``-W
# error`` still guards everything the mission cares about.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Card is too long:astropy.io.fits.verify.VerifyWarning",
)


# ---------------------------------------------------------------------------
# Synthetic corpus builders (self-contained; mirrors the R34f cluster corpus)
# ---------------------------------------------------------------------------

DISJOINT_CENTERS = [(10.0, 30.0), (10.05, 30.02), (12.0, 30.0), (12.05, 30.02)]


def _write_frame(path, ra, dec, shape=(400, 600), scale=0.001, seed=0, eqmode=None):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cd = np.array([[-1, 0], [0, 1]]) * scale
    w.array_shape = shape
    rng = np.random.default_rng(seed)
    data = (100.0 + rng.normal(0, 1, shape)).astype(np.float32)
    header = w.to_header()
    if eqmode is not None:
        header["EQMODE"] = eqmode
    fits.PrimaryHDU(data, header=header).writeto(path, overwrite=True)
    return str(path)


def _write_corpus(root, centers, **kw):
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (ra, dec) in enumerate(centers):
        paths.append(_write_frame(root / f"f{i}.fit", ra, dec, seed=i, **kw))
    return paths


def _frame_infos(paths, mount=None):
    if mount is None:
        return [FrameInfo(path=pathlib.Path(p), order=i) for i, p in enumerate(paths)]
    return [FrameInfo(path=pathlib.Path(p), order=i, mount=mount)
            for i, p in enumerate(paths)]


def _read_fits_bytes(path):
    with fits.open(path, memmap=False) as hdul:
        return np.ascontiguousarray(np.asarray(hdul[0].data))


def _sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# --- pure-assembly fixtures (CellPatch level, no FITS) ---------------------

CORE = 32
HALO = 8


def _canvas(nx, ny, core=CORE):
    return zg.GlobalCanvas(
        canvas_id=f"c{nx}x{ny}", wcs_header="", width=nx * core, height=ny * core,
        resolution_deg=0.001,
    )


def _make_patches(nx, ny, levels, *, core=CORE, halo=HALO, seed=0, sigma=0.3):
    rng = np.random.default_rng(seed)
    canvas = _canvas(nx, ny, core)
    patches = {}
    for row in range(ny):
        for col in range(nx):
            cid = f"r{row:04d}c{col:04d}"
            cy0, cy1 = row * core, (row + 1) * core
            cx0, cx1 = col * core, (col + 1) * core
            py0, py1 = max(0, cy0 - halo), min(canvas.height, cy1 + halo)
            px0, px1 = max(0, cx0 - halo), min(canvas.width, cx1 + halo)
            h, w = py1 - py0, px1 - px0
            lv = np.full(3, float(levels[cid]))
            sci = lv[None, None, :] + rng.normal(0.0, sigma, (h, w, 3))
            valid = np.ones((h, w, 3), dtype=bool)
            patches[cid] = zcf.CellPatch(
                cell_id=cid, bounds=(py0, py1, px0, px1),
                core_slice=(cy0 - py0, cy1 - py0, cx0 - px0, cx1 - px0),
                science=sci.astype(np.float32), valid_mask=valid,
            )
    return canvas, patches


def _results_from_patches(patches):
    out = {}
    for cid, cp in patches.items():
        out[cid] = {
            "cell_id": cid, "status": "complete",
            "patch": {
                "bounds": cp.bounds, "core_slice": cp.core_slice,
                "science": cp.science, "valid_mask": cp.valid_mask,
                "surviving": None,
            },
        }
    return out


def _levels_2x2():
    return {"r0000c0000": 100.0, "r0000c0001": 104.0,
            "r0001c0000": 96.0, "r0001c0001": 110.0}


# ---------------------------------------------------------------------------
# 1. Strict argument validation / defaults (no activation)
# ---------------------------------------------------------------------------

def test_capture_defaults_off():
    sig = inspect.signature(zz.run_zegrid_mode)
    assert sig.parameters["r37_diagnostic_capture"].default is False
    assert sig.parameters["r37_diagnostic_holdout_cap"].default == zz.R37_CAPTURE_HOLDOUT_CAP_DEFAULT
    # the seam's default cap must track the ACCEPTED artifact cap
    assert zz.R37_CAPTURE_HOLDOUT_CAP_DEFAULT == bgd.HOLDOUT_CAP
    assert zz.R37_CAPTURE_FILENAME == bgd.CAPTURE_ARTIFACT_BASENAME
    assert zz.R37_CAPTURE_STAGE == bgd.CAPTURE_STAGE == "pre_offset_pre_feather"


def test_strict_validation_types_and_cap():
    assert zz._validate_r37_diagnostic_capture(False, 64) == (False, 64)
    assert zz._validate_r37_diagnostic_capture(True, 16) == (True, 16)
    with pytest.raises(ValueError):
        zz._validate_r37_diagnostic_capture(1, 64)          # int is not a bool
    with pytest.raises(ValueError):
        zz._validate_r37_diagnostic_capture("true", 64)
    with pytest.raises(ValueError):
        zz._validate_r37_diagnostic_capture(True, True)     # bool is not an int cap
    with pytest.raises(ValueError):
        zz._validate_r37_diagnostic_capture(True, 2)        # cap < 4 with capture on
    # a disabled request may carry any int cap (never used)
    assert zz._validate_r37_diagnostic_capture(False, 2) == (False, 2)


def test_no_config_or_gui_activation_and_no_correction():
    src_root = pathlib.Path(zz.__file__).resolve().parents[1]
    offenders = []
    for p in src_root.rglob("*.py"):
        if p.name in ("zemosaic_zegrid_mode.py", "background_diagnostics.py"):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "r37_diagnostic" in text:
            offenders.append(str(p))
    assert offenders == []
    # the capture must never wire the R37 correction
    zz_src = pathlib.Path(zz.__file__).read_text(encoding="utf-8")
    assert "apply_plane_correction" not in zz_src


def test_zz_module_has_no_module_level_diagnostic_import():
    tree = ast.parse(pathlib.Path(zz.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not any("background_diagnostics" in a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module and "background_diagnostics" in node.module)


# ---------------------------------------------------------------------------
# 2. Publication: atomic, validated, digest-accurate (unit level)
# ---------------------------------------------------------------------------

def _valid_payload():
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    return bgd.build_payload(patches, 2, 2, (canvas.height, canvas.width),
                             measurement_px=2 * HALO, holdout_cap=16)


def test_publish_artifact_atomic_digest_and_no_residue(tmp_path):
    payload = _valid_payload()
    meta = bgd.publish_artifact(tmp_path, payload)
    target = tmp_path / bgd.CAPTURE_ARTIFACT_BASENAME
    assert target.is_file()
    assert meta["status"] == "written"
    assert meta["filename"] == target.name
    assert meta["capture_stage"] == "pre_offset_pre_feather"
    assert meta["bytes"] == target.stat().st_size
    assert meta["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert meta["schema"] == bgd.SCHEMA_NAME and meta["schema_version"] == bgd.SCHEMA_VERSION
    # strict-loadable + provenance stage
    art = bgd.load_file(target)
    assert art.provenance["production_wired"] is False
    # no temp residue anywhere in the directory
    leftovers = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob(".*tmp"))
    assert leftovers == []


def test_publish_artifact_rejects_before_write(tmp_path):
    payload = _valid_payload()
    payload["relations"][0]["information"] = None  # corrupt
    with pytest.raises(bgd.DiagnosticSchemaError):
        bgd.publish_artifact(tmp_path, payload)
    assert list(tmp_path.iterdir()) == []  # nothing published, no residue


def test_publish_artifact_write_failure_is_loud_and_no_partial(tmp_path, monkeypatch):
    payload = _valid_payload()

    def _boom(*a, **k):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(bgd.os, "replace", _boom)
    with pytest.raises(OSError):
        bgd.publish_artifact(tmp_path, payload)
    assert not (tmp_path / bgd.CAPTURE_ARTIFACT_BASENAME).exists()
    assert list(tmp_path.glob("*.tmp")) == [] and list(tmp_path.glob(".*tmp")) == []


def test_publish_artifact_has_no_basename_override(tmp_path):
    # L2: the fixed production filename is not caller-controllable.
    import inspect
    params = inspect.signature(bgd.publish_artifact).parameters
    assert "basename" not in params
    assert list(params) == ["output_dir", "payload"]
    bgd.publish_artifact(tmp_path, _valid_payload())
    assert (tmp_path / bgd.CAPTURE_ARTIFACT_BASENAME).is_file()


def test_publish_invalidates_existing_target_on_validation_failure(tmp_path):
    # M2.3: a pre-existing valid target must NOT survive a failed (re)publication.
    bgd.publish_artifact(tmp_path, _valid_payload())
    target = tmp_path / bgd.CAPTURE_ARTIFACT_BASENAME
    assert target.is_file()
    bad = _valid_payload()
    bad["relations"][0]["information"] = None
    with pytest.raises(bgd.DiagnosticSchemaError):
        bgd.publish_artifact(tmp_path, bad)
    assert not target.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_publish_invalidates_existing_target_on_replace_failure(tmp_path, monkeypatch):
    # M2.3: os.replace failure must leave NO fixed target (old invalidated).
    bgd.publish_artifact(tmp_path, _valid_payload())
    target = tmp_path / bgd.CAPTURE_ARTIFACT_BASENAME
    assert target.is_file()

    def _boom(*a, **k):
        raise OSError("simulated disk full")

    monkeypatch.setattr(bgd.os, "replace", _boom)
    with pytest.raises(OSError):
        bgd.publish_artifact(tmp_path, _valid_payload())
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == [] and list(tmp_path.glob(".*tmp")) == []


# ---------------------------------------------------------------------------
# 3. Pure assembler: capture ON vs OFF array-equality + payload relations
# ---------------------------------------------------------------------------

def test_pure_assemble_capture_on_off_arrays_equal(tmp_path):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results = _results_from_patches(patches)
    off_a, info_off = zz._assemble_per_cell_feathered(canvas, 2, 2, results)
    on_a, info_on = zz._assemble_per_cell_feathered(
        canvas, 2, 2, results, capture=True, output_dir=tmp_path,
    )
    np.testing.assert_array_equal(off_a.science, on_a.science)
    np.testing.assert_array_equal(off_a.stack_depth, on_a.stack_depth)
    np.testing.assert_array_equal(off_a.valid_fraction, on_a.valid_fraction)
    np.testing.assert_array_equal(off_a.owner, on_a.owner)
    # info equal once the capture metadata is set aside
    assert info_on.pop("r37_diagnostic_capture")
    assert info_on == info_off
    prov = json.loads((tmp_path / zz.R37_CAPTURE_FILENAME).read_text())["provenance"]
    assert prov["capture_stage"] == "pre_offset_pre_feather"
    assert prov["correction_applied"] is False
    assert prov["capture"]["capture_stage"] == "pre_offset_pre_feather"
    assert prov["capture"]["effective_cell_gauge_mode"] == "per_cell"
    assert prov["capture"]["halo_px"] == HALO
    assert prov["capture"]["measurement_px"] == 2 * HALO
    assert prov["capture"]["feather_px"] == HALO


def test_pure_assemble_capture_disabled_writes_nothing(tmp_path):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results = _results_from_patches(patches)
    _, info = zz._assemble_per_cell_feathered(canvas, 2, 2, results)
    assert "r37_diagnostic_capture" not in info
    assert list(tmp_path.iterdir()) == []


def test_pure_assemble_payload_relations_match_fresh_build(tmp_path):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results = _results_from_patches(patches)
    zz._assemble_per_cell_feathered(canvas, 2, 2, results, capture=True, output_dir=tmp_path)
    art = bgd.load_file(tmp_path / zz.R37_CAPTURE_FILENAME)
    fresh = bgd.load_artifact(
        bgd.build_payload(patches, 2, 2, (canvas.height, canvas.width),
                          measurement_px=2 * HALO,
                          holdout_cap=zz.R37_CAPTURE_HOLDOUT_CAP_DEFAULT)
    )
    assert {o.key() for o in art.observations} == {o.key() for o in fresh.observations}
    assert len(art.dropped) == len(fresh.dropped)
    assert tuple(art.cell_ids()) == tuple(fresh.cell_ids())


# ---------------------------------------------------------------------------
# 3b. M1 — published provenance is truthful + single-valued (read alone)
# ---------------------------------------------------------------------------

def test_published_provenance_truthful_and_single_valued(tmp_path):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results = _results_from_patches(patches)
    zz._assemble_per_cell_feathered(canvas, 2, 2, results, capture=True, output_dir=tmp_path)
    prov = bgd.load_file(tmp_path / zz.R37_CAPTURE_FILENAME).provenance
    # single-valued wiring facts
    assert prov["production_wired"] is True
    assert prov["mission"] == bgd.CAPTURE_MISSION == "ZM-ZEGRID-R37-D"
    assert prov["status"] == bgd.CAPTURE_PROVENANCE_STATUS
    assert prov["capture_stage"] == "pre_offset_pre_feather"
    assert prov["correction_applied"] is False
    assert prov["artifact_kind"] == "production_capture"
    # NO contradictory duplicate inside the nested block
    assert "production_wired" not in prov["capture"]
    assert "correction_applied" not in prov["capture"]
    assert prov["capture"]["capture_stage"] == prov["capture_stage"]
    # the pure default still says research-only for standalone build_payload use
    standalone = bgd.build_payload(patches, 2, 2, (canvas.height, canvas.width),
                                   measurement_px=2 * HALO, holdout_cap=16)
    assert standalone["provenance"]["production_wired"] is False
    assert standalone["provenance"]["mission"] == "ZM-ZEGRID-R37-C"


# ---------------------------------------------------------------------------
# 3c. M2/L1 identity, digest, linkage, widths
# ---------------------------------------------------------------------------

def test_source_set_digest_stable_and_sensitive():
    a = ["a.fit", "b.fit", "c.fit"]
    assert bgd.source_set_digest(a) == bgd.source_set_digest(list(reversed(a)))
    assert bgd.source_set_digest(a) == bgd.source_set_digest(a + ["a.fit"])  # dedup
    # path-object types do not matter
    assert bgd.source_set_digest(["a.fit", "b.fit"]) == bgd.source_set_digest(
        [pathlib.Path("a.fit"), pathlib.Path("b.fit")]
    )
    assert bgd.source_set_digest(a) != bgd.source_set_digest(["a.fit", "b.fit", "d.fit"])
    assert bgd.source_set_digest(a) != bgd.source_set_digest(["a.fit", "b.fit"])


def test_capture_run_id_distinct_per_pass():
    d = bgd.source_set_digest(["a.fit"])
    base = bgd.capture_run_id(start_ts="t", pass_label="single", n_frames=1,
                              source_set_digest_value=d)
    assert base == bgd.capture_run_id(start_ts="t", pass_label="single", n_frames=1,
                                      source_set_digest_value=d)
    assert base != bgd.capture_run_id(start_ts="t", pass_label="EQ", n_frames=1,
                                      source_set_digest_value=d)
    assert base != bgd.capture_run_id(start_ts="t2", pass_label="single", n_frames=1,
                                      source_set_digest_value=d)
    assert base != bgd.capture_run_id(start_ts="t", pass_label="single", n_frames=2,
                                      source_set_digest_value=d)


def test_manifest_links_artifact_rule():
    art = {"capture_id": "id1", "filename": "f.json", "sha256": "aa", "bytes": 10}
    assert bgd.manifest_links_artifact(art, dict(art)) is True
    assert bgd.manifest_links_artifact(art, None) is False
    assert bgd.manifest_links_artifact(art, {"capture_id": "id1"}) is False
    assert bgd.manifest_links_artifact(art, {**art, "sha256": "bb"}) is False
    assert bgd.manifest_links_artifact(art, {**art, "capture_id": "id2"}) is False
    assert bgd.manifest_links_artifact(art, {**art, "bytes": 11}) is False


def test_nondefault_halo_widths_consistent(tmp_path):
    # L1: halo 4 -> measurement 8, feather 4, in both provenance capture and meta.
    canvas, patches = _make_patches(2, 2, _levels_2x2(), halo=4)
    results = _results_from_patches(patches)
    zz._assemble_per_cell_feathered(
        canvas, 2, 2, results, halo_px=4, capture=True, output_dir=tmp_path,
    )
    art = bgd.load_file(tmp_path / zz.R37_CAPTURE_FILENAME)
    cap = art.provenance["capture"]
    assert cap["halo_px"] == 4 and cap["measurement_px"] == 8 and cap["feather_px"] == 4
    assert art.meta["measurement_px"] == 8
    assert art.meta["feather_px"] == 4


# ---------------------------------------------------------------------------
# 4. End-to-end: default OFF (no artifact, diagnostic never touched)
# ---------------------------------------------------------------------------

def test_e2e_default_off_no_artifact_and_no_diagnostic_call(tmp_path, monkeypatch):
    inp = tmp_path / "in"
    _write_corpus(inp, DISJOINT_CENTERS)
    out = tmp_path / "out"

    def _boom(*a, **k):
        raise AssertionError("diagnostic module touched on the disabled path")

    monkeypatch.setattr(bgd, "build_payload", _boom)
    monkeypatch.setattr(bgd, "publish_artifact", _boom)
    zz.run_zegrid_mode(str(inp), str(out), zconfig=SimpleNamespace(zegrid_layout="2x2"))

    man = json.loads((out / "zegrid_manifest.json").read_text())
    assert man["cell_feather"]["enabled"] is True          # auto self-healed to per_cell
    assert "r37_diagnostic_capture" not in man["cell_feather"]
    assert not (out / zz.R37_CAPTURE_FILENAME).exists()
    assert (out / "mosaic_grid.fits").exists()


# ---------------------------------------------------------------------------
# 5. End-to-end: explicit per_cell ON publishes a validated artifact
# ---------------------------------------------------------------------------

def test_e2e_per_cell_capture_publishes_validated_artifact(tmp_path):
    inp = tmp_path / "in"
    _write_corpus(inp, DISJOINT_CENTERS)
    out = tmp_path / "out"
    zz.run_zegrid_mode(
        str(inp), str(out), zconfig=SimpleNamespace(zegrid_layout="2x2"),
        cell_gauge_mode="per_cell", r37_diagnostic_capture=True,
    )
    target = out / zz.R37_CAPTURE_FILENAME
    assert target.is_file()
    art = bgd.load_file(target)
    prov = art.provenance
    assert prov["capture_stage"] == "pre_offset_pre_feather"
    assert prov["production_wired"] is True and prov["correction_applied"] is False
    assert prov["capture"]["effective_cell_gauge_mode"] == "per_cell"
    # M2 identity generated in _run_single, present in the artifact alone
    ident = prov["capture"]
    assert len(ident["capture_id"]) == 64
    assert ident["run_label"] == "single"
    assert ident["n_source_frames"] == 4
    assert len(ident["source_set_digest"]) == 64
    assert ident["source_set_digest_algo"] == bgd.SOURCE_SET_DIGEST_ALGO

    man = json.loads((out / "zegrid_manifest.json").read_text())
    cap = man["cell_feather"]["r37_diagnostic_capture"]
    assert cap["status"] == "written"
    assert cap["filename"] == target.name
    assert cap["capture_stage"] == "pre_offset_pre_feather"
    assert cap["schema"] == bgd.SCHEMA_NAME
    assert cap["schema_version"] == bgd.SCHEMA_VERSION
    assert cap["bytes"] == target.stat().st_size
    assert cap["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert cap["n_relations"] == len(art.observations) > 0
    assert cap["n_dropped"] == len(art.dropped)
    assert cap["holdout_cap"] == zz.R37_CAPTURE_HOLDOUT_CAP_DEFAULT
    assert cap["measurement_px"] == 2 * zz.zsw.HALO_PX
    assert cap["halo_px"] == zz.zsw.HALO_PX and cap["feather_px"] == zz.zsw.HALO_PX
    assert cap["effective_cell_gauge_mode"] == "per_cell"
    # M2.4: the manifest cross-links the SAME artifact (all four keys match)
    assert bgd.manifest_links_artifact(
        {"capture_id": prov["capture"]["capture_id"], "filename": target.name,
         "sha256": cap["sha256"], "bytes": cap["bytes"]}, cap) is True
    assert cap["capture_id"] == ident["capture_id"]
    assert cap["source_set_digest"] == ident["source_set_digest"]
    # the capture identity is recorded in the run log BEFORE stacking outcomes
    run_log = (out / "zegrid_run.log").read_text()
    assert "R37-D capture identity" in run_log
    assert ident["capture_id"] in run_log
    # no temp residue in the output directory
    assert list(out.glob("*.tmp")) == [] and list(out.glob(".*tmp")) == []


def test_e2e_capture_survives_later_failure_as_uncommitted(tmp_path, monkeypatch):
    # M2.4: a later run failure after a successful pre-feather capture leaves the
    # artifact as (uncommitted) evidence; with no/old manifest it does not qualify.
    inp = tmp_path / "in"
    _write_corpus(inp, DISJOINT_CENTERS)
    out = tmp_path / "out"

    def _boom(*a, **k):
        raise RuntimeError("simulated later failure (post-capture)")

    monkeypatch.setattr(zz, "_write_outputs", _boom)
    with pytest.raises(RuntimeError):
        zz.run_zegrid_mode(
            str(inp), str(out), zconfig=SimpleNamespace(zegrid_layout="2x2"),
            cell_gauge_mode="per_cell", r37_diagnostic_capture=True,
        )
    target = out / zz.R37_CAPTURE_FILENAME
    assert target.is_file()  # capture evidence survives
    assert not (out / "zegrid_manifest.json").exists()  # no manifest
    art = bgd.load_file(target)
    capture_meta = {
        "capture_id": art.provenance["capture"]["capture_id"],
        "filename": target.name,
    }
    # no manifest -> uncommitted / non-qualifying
    assert bgd.manifest_links_artifact(capture_meta, None) is False


def test_e2e_cap_large_exact_and_small_capped(tmp_path):
    inp = tmp_path / "in"
    _write_corpus(inp, DISJOINT_CENTERS)

    out_exact = tmp_path / "out_exact"
    zz.run_zegrid_mode(
        str(inp), str(out_exact), zconfig=SimpleNamespace(zegrid_layout="2x2"),
        cell_gauge_mode="per_cell", r37_diagnostic_capture=True,
        r37_diagnostic_holdout_cap=10 ** 6,
    )
    art_exact = bgd.load_file(out_exact / zz.R37_CAPTURE_FILENAME)
    assert art_exact.heldout_exact() is True
    man_exact = json.loads((out_exact / "zegrid_manifest.json").read_text())
    assert man_exact["cell_feather"]["r37_diagnostic_capture"]["qualification"] == bgd.QUALIFICATION_EXACT
    assert man_exact["cell_feather"]["r37_diagnostic_capture"]["holdout_capped"] is False

    out_capped = tmp_path / "out_capped"
    zz.run_zegrid_mode(
        str(inp), str(out_capped), zconfig=SimpleNamespace(zegrid_layout="2x2"),
        cell_gauge_mode="per_cell", r37_diagnostic_capture=True,
        r37_diagnostic_holdout_cap=4,
    )
    art_capped = bgd.load_file(out_capped / zz.R37_CAPTURE_FILENAME)
    assert art_capped.heldout_capped is True
    man_capped = json.loads((out_capped / "zegrid_manifest.json").read_text())
    cap_meta = man_capped["cell_feather"]["r37_diagnostic_capture"]
    assert cap_meta["qualification"] == bgd.QUALIFICATION_RESEARCH_ONLY_CAPPED
    assert cap_meta["holdout_capped"] is True
    # a capped artifact can never qualify ROBUST_PLAN
    res = bgd.recompute_decision(art_capped)
    assert res.decision != bm.STATE_ROBUST_PLAN
    assert res.provenance["diagnostic"]["qualified"] is False
    assert not any(k.startswith("diagnostic") for k in res.gates)


# ---------------------------------------------------------------------------
# 6. auto -> per_cell captures; global / auto->global fail loudly
# ---------------------------------------------------------------------------

def test_e2e_auto_resolves_per_cell_and_captures(tmp_path):
    inp = tmp_path / "in"
    _write_corpus(inp, DISJOINT_CENTERS)
    out = tmp_path / "out"
    zz.run_zegrid_mode(
        str(inp), str(out), zconfig=SimpleNamespace(zegrid_layout="2x2"),
        r37_diagnostic_capture=True,  # auto (default)
    )
    man = json.loads((out / "zegrid_manifest.json").read_text())
    assert man["photometric_gauge"]["mode"] == "per_cell"
    assert (out / zz.R37_CAPTURE_FILENAME).is_file()


def test_e2e_explicit_global_capture_fails_loud_and_early(tmp_path):
    inp = tmp_path / "in"
    _write_corpus(inp, DISJOINT_CENTERS)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        zz.run_zegrid_mode(
            str(inp), str(out), zconfig=SimpleNamespace(zegrid_layout="2x2"),
            cell_gauge_mode="global", r37_diagnostic_capture=True,
        )
    assert not (out / zz.R37_CAPTURE_FILENAME).exists()


def test_e2e_auto_to_global_capture_fails_loud(tmp_path):
    # all frames identical footprint -> every Cell is globally served -> auto->global
    inp = tmp_path / "in"
    _write_corpus(inp, [(10.0, 30.0)] * 4)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        zz.run_zegrid_mode(
            str(inp), str(out), zconfig=SimpleNamespace(zegrid_layout="2x2"),
            r37_diagnostic_capture=True,  # auto resolves to global
        )
    assert not (out / zz.R37_CAPTURE_FILENAME).exists()


# ---------------------------------------------------------------------------
# 7. Mount segregation: one artifact per actual output directory
# ---------------------------------------------------------------------------

def test_e2e_mount_segregation_one_artifact_per_output_dir(tmp_path):
    inp = tmp_path / "in"
    eq_paths = _write_corpus(inp / "eq", DISJOINT_CENTERS, eqmode=1)
    altz_paths = _write_corpus(inp / "altz", DISJOINT_CENTERS, eqmode=0)
    frames = _frame_infos(eq_paths, "EQ") + _frame_infos(altz_paths, "ALTZ")
    out = tmp_path / "out"
    zz.run_zegrid_mode(
        str(inp), str(out), frames_info=frames,
        zconfig=SimpleNamespace(zegrid_layout="2x2"),
        cell_gauge_mode="per_cell", r37_diagnostic_capture=True,
    )
    eq_art = out / "grid_EQ" / zz.R37_CAPTURE_FILENAME
    altz_art = out / "grid_ALTZ" / zz.R37_CAPTURE_FILENAME
    assert eq_art.is_file() and altz_art.is_file()
    # distinct directories, distinct files, no collision
    assert eq_art.parent != altz_art.parent
    bgd.load_file(eq_art)
    bgd.load_file(altz_art)
    for sub in ("grid_EQ", "grid_ALTZ"):
        man = json.loads((out / sub / "zegrid_manifest.json").read_text())
        assert man["cell_feather"]["r37_diagnostic_capture"]["status"] == "written"


# ---------------------------------------------------------------------------
# 8. Capture ON vs OFF: byte-identical outputs + identical R34c offsets
# ---------------------------------------------------------------------------

def test_e2e_capture_on_off_outputs_and_offsets_identical(tmp_path):
    inp = tmp_path / "in"
    _write_corpus(inp, DISJOINT_CENTERS)
    out_off = tmp_path / "off"
    out_on = tmp_path / "on"
    # export the aesthetic float too, so the invariant covers science + aesthetic +
    # coverage (the aesthetic FITS is written only when requested).
    cfg_off = SimpleNamespace(zegrid_layout="2x2", export_aesthetic_fits=True)
    cfg_on = SimpleNamespace(zegrid_layout="2x2", export_aesthetic_fits=True)
    zz.run_zegrid_mode(str(inp), str(out_off), zconfig=cfg_off, cell_gauge_mode="per_cell")
    zz.run_zegrid_mode(str(inp), str(out_on), zconfig=cfg_on, cell_gauge_mode="per_cell",
                       r37_diagnostic_capture=True)

    sci_off = _read_fits_bytes(out_off / "mosaic_grid_science.fits")
    sci_on = _read_fits_bytes(out_on / "mosaic_grid_science.fits")
    np.testing.assert_array_equal(sci_off, sci_on)
    assert _sha(sci_off) == _sha(sci_on)

    aes_off = _read_fits_bytes(out_off / "mosaic_grid_aesthetic.fits")
    aes_on = _read_fits_bytes(out_on / "mosaic_grid_aesthetic.fits")
    np.testing.assert_array_equal(aes_off, aes_on)
    assert _sha(aes_off) == _sha(aes_on)

    cov_off = _read_fits_bytes(out_off / "mosaic_grid_coverage.fits")
    cov_on = _read_fits_bytes(out_on / "mosaic_grid_coverage.fits")
    np.testing.assert_array_equal(cov_off, cov_on)

    m_off = json.loads((out_off / "zegrid_manifest.json").read_text())
    m_on = json.loads((out_on / "zegrid_manifest.json").read_text())
    # R34c offsets + coverage/geometry metadata identical
    assert m_on["cell_feather"]["offsets"] == m_off["cell_feather"]["offsets"]
    assert m_on["complete_cells"] == m_off["complete_cells"]
    assert m_on["incomplete_cells"] == m_off["incomplete_cells"]
    assert m_on["coverage_pixels"] == m_off["coverage_pixels"]
    assert m_on["hole_pixels"] == m_off["hole_pixels"]
    # ...and the science_output_contract hashes agree (aesthetic is really exported)
    assert m_on["science_output_contract"] == m_off["science_output_contract"]
    # only the diagnostic metadata differs
    assert "r37_diagnostic_capture" not in m_off["cell_feather"]
    assert "r37_diagnostic_capture" in m_on["cell_feather"]
    m_on_clean = copy.deepcopy(m_on)
    del m_on_clean["cell_feather"]["r37_diagnostic_capture"]
    assert m_on_clean["cell_feather"] == m_off["cell_feather"]
