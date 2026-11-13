"""ZM-ZEGRID-GAUGE-IMPORT — a fresh run imports a VERIFIED global gauge.

The gauge is a function of the frames, the canvas and the photometric
parameters; the tiling does not enter it. The M33 gauge over 21 818 frames takes
hours and is bound to the run identity (which includes the layout), so a new,
better layout used to force a full recomputation. This feature lets a FRESH run
import the artifact from another run of the same frames/canvas/science, with the
SAME verification the resume path already trusts.

These tests prove:
* the import skips the gauge phase and the identity's ``gauge`` sha256 equals
  the COMPUTED one (the acceptance proof), across a DIFFERENT layout;
* the imported gauge yields identical per-Cell payloads for the same layout;
* every tampered section (frames/digests, canvas, science, gauge, artifact) is
  refused, naming the section, with no product written;
* a run without the option is unchanged (computed gauge, historical manifest
  shape) and the artifact is never written to.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import run_artifacts as zart  # noqa: E402
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

TINY_CORPUS_TOOL = REPO_ROOT / "tools" / "zegrid_m6" / "make_tiny_corpus.py"
MIB = 2 ** 20
MANIFEST = "zegrid_manifest.json"


def _config(store, layout, resume_mode="never"):
    return SimpleNamespace(
        zegrid_scheduler_mode="active",
        zegrid_durable_store_dir=str(store),
        zegrid_resume_run_id=None,
        zegrid_durable_store_budget_bytes=64 * MIB,
        zegrid_layout=layout,
        zegrid_cell_gauge_mode="global",
        zegrid_resource_recovery_enabled=False,
        zegrid_resource_recovery_deadline_s=None,
        zegrid_sip_mode="keep",
        temporary_work_dir="",
        zegrid_resume_mode=resume_mode,
    )


def _run(corpus, out, store, layout, *, gauge_import=None, overrides=None):
    logs = []
    zz.run_zegrid_mode(
        str(corpus), str(out), lambda *args, **_kw: logs.append(str(args[0])),
        zconfig=_config(store, layout),
        zegrid_halo_px=8,
        use_gpu=False,
        zegrid_gauge_import=gauge_import,
        **(overrides or {}),
    )
    manifest = json.loads((Path(out) / MANIFEST).read_text(encoding="utf-8"))
    identity = json.loads((Path(store) / "run.json").read_text(encoding="utf-8"))
    return {"logs": logs, "manifest": manifest, "identity": identity}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("gauge-import-corpus")
    corpus = root / "in"
    subprocess.run(
        [sys.executable, str(TINY_CORPUS_TOOL), str(corpus)],
        check=True, capture_output=True,
    )
    return corpus


@pytest.fixture(scope="module")
def computed(corpus, tmp_path_factory):
    root = tmp_path_factory.mktemp("gauge-import-computed")
    out, store = root / "out", root / "store"
    result = _run(corpus, out, store, "2x2")
    result.update(root=root, out=out, store=store)
    return result


@pytest.fixture(scope="module")
def imported_same_layout(corpus, computed, tmp_path_factory):
    root = tmp_path_factory.mktemp("gauge-import-same-layout")
    out, store = root / "out", root / "store"
    option = {
        "gauge_path": str(computed["store"] / "gauge.json"),
        "origin_run_json": str(computed["store"] / "run.json"),
    }
    result = _run(corpus, out, store, "2x2", gauge_import=option)
    result.update(root=root, out=out, store=store, option=option)
    return result


@pytest.fixture(scope="module")
def imported_other_layout(corpus, computed, tmp_path_factory):
    root = tmp_path_factory.mktemp("gauge-import-other-layout")
    out, store = root / "out", root / "store"
    option = {
        "gauge_path": str(computed["store"] / "gauge.json"),
        "origin_run_json": str(computed["store"] / "run.json"),
    }
    result = _run(corpus, out, store, "1x2", gauge_import=option)
    result.update(root=root, out=out, store=store, option=option)
    return result


# ---------------------------------------------------------------------------
# 1. Import skips the gauge phase and keeps the IDENTITY gauge sha256
# ---------------------------------------------------------------------------

def test_import_skips_gauge_and_preserves_identity_gauge_sha(imported_other_layout):
    manifest = imported_other_layout["manifest"]
    identity = imported_other_layout["identity"]
    logs = imported_other_layout["logs"]

    # The layout really is different from the origin's 2x2.
    assert manifest["layout"]["nx"] == 1 and manifest["layout"]["ny"] == 2

    # The gauge phase never computed anything.
    diagnostics = manifest["gauge_diagnostics"]
    assert diagnostics["source"] == "imported_gauge_artifact"
    assert diagnostics["compute_global_gauge_executed"] is False
    assert diagnostics["import"]["source"] == "imported_artifact"
    assert diagnostics["import"]["origin_run_id"]
    assert diagnostics["import"]["gauge_sha256"] == identity[
        "science_contract"]["gauge"]["sha256"]

    # The run log says IMPORTED, and never claims it computed the gauge.
    joined = "\n".join(logs)
    assert "IMPORTED from a verified artifact" in joined
    assert "gauge phase will NOT execute" in joined
    assert "computing global photometric gauge" not in joined

    # Provenance is additive and explicit in the manifest.
    provenance = manifest["photometric_gauge"]["source"]
    assert provenance == "imported_artifact"
    block = manifest["photometric_gauge"]["import"]
    assert block["gauge_path"] == imported_other_layout["option"]["gauge_path"]
    assert block["origin_run_json_path"] == (
        imported_other_layout["option"]["origin_run_json"])
    assert block["gauge_sha256"] == identity["science_contract"]["gauge"]["sha256"]
    assert len(block["artifact_sha256"]) == 64


def test_import_identity_gauge_sha_equals_the_computed_one(
    computed, imported_other_layout
):
    computed_sha = computed["identity"]["science_contract"]["gauge"]["sha256"]
    imported_sha = imported_other_layout["identity"]["science_contract"]["gauge"][
        "sha256"]
    assert imported_sha == computed_sha
    # The whole scientific contract matches too (a gauge import is not a
    # science change): same frames/canvas/science/gauge; only grid differs.
    a = computed["identity"]["science_contract"]
    c = imported_other_layout["identity"]["science_contract"]
    for section in ("inputs", "canvas", "science", "gauge", "dtypes", "backend"):
        assert a[section] == c[section], section
    assert a["grid"] != c["grid"]  # the layout DID change


# ---------------------------------------------------------------------------
# 2. Same layout -> identical per-Cell payloads (not just acceptance)
# ---------------------------------------------------------------------------

def test_imported_gauge_yields_identical_per_cell_payloads(
    computed, imported_same_layout
):
    a, c = computed["manifest"], imported_same_layout["manifest"]
    # Compare the SCIENTIFIC per-Cell payload exactly. The admission-plan fields
    # (``mode``/``bound_bytes``/``backend_used``) are host-RAM decisions that a
    # fresh observation may legitimately choose differently between two runs;
    # they are not gauge-derived payload.
    scientific = (
        "cell_id", "row", "col", "patch_bounds", "patch_shape_hw",
        "n_contributors", "n_patch_contributors", "reference_frame_id",
        "reference_frame_role", "bookkeeping_reference_frame_id",
        "excluded", "status", "cell_gauge_mode",
    )
    assert [{k: cell[k] for k in scientific} for cell in a["cells"]] == [
        {k: cell[k] for k in scientific} for cell in c["cells"]
    ]
    assert a["complete_cells"] == c["complete_cells"]
    assert a["incomplete_cells"] == c["incomplete_cells"]
    assert a["hole_pixels"] == c["hole_pixels"]
    assert a["coverage_pixels"] == c["coverage_pixels"]
    for name in ("mosaic_grid.fits", "mosaic_grid_coverage.fits"):
        da = fits.getdata(computed["out"] / name)
        dc = fits.getdata(imported_same_layout["out"] / name)
        assert np.array_equal(da, dc, equal_nan=True), name
    assert computed["identity"]["science_contract"] == imported_same_layout[
        "identity"]["science_contract"]


# ---------------------------------------------------------------------------
# 3. Tampered frame set -> refused, naming the inputs section
# ---------------------------------------------------------------------------

def _tamper_one_frame(corpus, target):
    """Copy the corpus and change ONE frame's CONTENT (same id, new digest)."""
    shutil.copytree(corpus, target)
    path = sorted(target.glob("*.fits"))[0]
    with fits.open(path) as hdul:
        data = np.array(hdul[0].data, dtype=np.float64)
        data[0, 0] = float(data[0, 0]) + 1.0
        header = hdul[0].header.copy()
    fits.PrimaryHDU(data=data.astype("f4"), header=header).writeto(path, overwrite=True)
    return path


def test_tampered_frame_digest_is_refused(tmp_path, corpus, computed):
    tampered = tmp_path / "tampered"
    changed = _tamper_one_frame(corpus, tampered)
    artifact_before = _sha(computed["store"] / "gauge.json")
    out, store = tmp_path / "out", tmp_path / "store"
    with pytest.raises(zres.ResumeIncompatible) as exc:
        _run(
            tampered, out, store, "2x2",
            gauge_import={
                "gauge_path": str(computed["store"] / "gauge.json"),
                "origin_run_json": str(computed["store"] / "run.json"),
            },
        )
    assert exc.value.reasons[0]["section"] == "inputs"
    assert "inputs" in str(exc.value)
    # Nothing was published and the artifact is intact.
    assert not (out / MANIFEST).is_file()
    assert not (out / "mosaic_grid.fits").is_file()
    assert _sha(computed["store"] / "gauge.json") == artifact_before
    assert changed.is_file()


# ---------------------------------------------------------------------------
# 4. Canvas / science / gauge mismatches -> refused, naming the section
# ---------------------------------------------------------------------------

def _origin(computed):
    identity = computed["identity"]
    contract = identity["science_contract"]
    return {
        "origin_identity": identity,
        "origin_run_json_path": str(computed["store"] / "run.json"),
        "expected_frame_rows": contract["inputs"]["frames"],
        "expected_frame_ids": [row["frame_id"] for row in contract["inputs"]["frames"]],
        "expected_canvas": contract["canvas"],
        "expected_science_params": contract["science"]["params"],
        "expected_mode": contract["gauge"]["mode"],
    }


def _import(computed, **overrides):
    kwargs = _origin(computed)
    kwargs.update(overrides)
    return zart.import_gauge_artifact(
        str(computed["store"] / "gauge.json"), **kwargs
    )


def test_canvas_mismatch_is_refused(computed):
    canvas = dict(_origin(computed)["expected_canvas"])
    canvas["shape"] = [canvas["shape"][0] + 1, canvas["shape"][1]]
    with pytest.raises(zres.ResumeIncompatible) as exc:
        _import(computed, expected_canvas=canvas)
    assert exc.value.reasons[0]["section"] == "canvas"


def test_science_mismatch_is_refused(computed):
    params = dict(_origin(computed)["expected_science_params"])
    params["normalization"] = "linear_fit"
    with pytest.raises(zres.ResumeIncompatible) as exc:
        _import(computed, expected_science_params=params)
    assert exc.value.reasons[0]["section"] == "science"
    assert "normalization" in str(exc.value)


def test_gauge_mode_mismatch_is_refused(computed):
    with pytest.raises(zres.ResumeIncompatible) as exc:
        _import(computed, expected_mode="per_cell")
    assert exc.value.reasons[0]["section"] == "gauge"


def test_end_to_end_science_change_is_refused(tmp_path, corpus, computed):
    out, store = tmp_path / "out", tmp_path / "store"
    with pytest.raises(zres.ResumeIncompatible) as exc:
        _run(
            corpus, out, store, "2x2",
            gauge_import={
                "gauge_path": str(computed["store"] / "gauge.json"),
                "origin_run_json": str(computed["store"] / "run.json"),
            },
            overrides={"stack_norm_method": "linear_fit"},
        )
    assert exc.value.reasons[0]["section"] == "science"
    assert not (out / MANIFEST).is_file()


# ---------------------------------------------------------------------------
# 5. Corrupt artifact / bad self-digest -> refused
# ---------------------------------------------------------------------------

def test_corrupt_artifact_is_refused(tmp_path, computed):
    raw = json.loads((computed["store"] / "gauge.json").read_text())
    raw["gauge_sha256"] = "0" * 64  # invalidates artifact_sha256
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text(json.dumps(raw), encoding="utf-8")
    kwargs = _origin(computed)
    with pytest.raises(zres.ResumeIncompatible) as exc:
        zart.import_gauge_artifact(str(corrupt), **kwargs)
    assert exc.value.reasons[0]["section"] == "artifact"
    assert "self-digest" in str(exc.value)


def test_unreadable_artifact_is_refused(tmp_path, computed):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    kwargs = _origin(computed)
    with pytest.raises(zres.ResumeIncompatible) as exc:
        zart.import_gauge_artifact(str(broken), **kwargs)
    assert exc.value.reasons[0]["section"] == "artifact"


# ---------------------------------------------------------------------------
# 6. Without the option: unchanged; artifact never written
# ---------------------------------------------------------------------------

def test_without_the_option_the_gauge_is_computed_as_before(computed):
    manifest = computed["manifest"]
    assert "source" not in manifest["photometric_gauge"]
    assert "import" not in manifest["photometric_gauge"]
    assert manifest["gauge_diagnostics"].get("source") != "imported_gauge_artifact"
    joined = "\n".join(computed["logs"])
    assert "computing global photometric gauge" in joined
    assert "IMPORTED from a verified artifact" not in joined
    # Same gauge as the imported run (the corpus is identical).
    assert computed["identity"]["science_contract"]["gauge"]["sha256"]


def test_artifact_is_read_only(computed, imported_same_layout):
    gauge_path = computed["store"] / "gauge.json"
    origin_path = computed["store"] / "run.json"
    before = (_sha(gauge_path), _sha(origin_path),
              gauge_path.stat().st_mtime_ns, origin_path.stat().st_mtime_ns)
    assert imported_same_layout["identity"]["science_contract"]["gauge"]["sha256"]
    after = (_sha(gauge_path), _sha(origin_path),
             gauge_path.stat().st_mtime_ns, origin_path.stat().st_mtime_ns)
    assert after == before


# ---------------------------------------------------------------------------
# 7. The option is explicit and validated before anything is written
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {"gauge_path": "/nonexistent/gauge.json", "origin_run_json": "/nonexistent/run.json"},
    {"gauge_path": "/tmp", "origin_run_json": "/tmp/run.json"},
    {"gauge_path": "/tmp/x.json"},
    {"gauge_path": "/tmp/x.json", "origin_run_json": "/tmp/y.json", "extra": 1},
    "not-a-mapping",
])
def test_invalid_option_is_refused_before_any_write(tmp_path, corpus, bad):
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        zz.run_zegrid_mode(
            str(corpus), str(out), None,
            zconfig=_config(tmp_path / "store", "2x2"),
            zegrid_halo_px=8, use_gpu=False, zegrid_gauge_import=bad,
        )
    assert not out.exists()


# ---------------------------------------------------------------------------
# REWORK-1 — M1: the corpus is read ONCE on an import run
# ---------------------------------------------------------------------------

def test_rework1_import_hashes_each_frame_exactly_once(
    tmp_path, corpus, computed, monkeypatch
):
    """M1: no second full-corpus read (import verification + identity build).

    A normal run hashes each frame once (the identity build). An import run must
    ALSO hash each frame once: the rows computed for the verification are reused
    verbatim by ``_r55_build_run_store_identity``.
    """
    calls = []
    real = zrs.content_sha256_file

    def spy(path):
        calls.append(str(path))
        return real(path)

    monkeypatch.setattr(zrs, "content_sha256_file", spy)
    n_frames = len(list(corpus.glob("*.fits")))

    _run(corpus, tmp_path / "out-normal", tmp_path / "store-normal", "2x2")
    assert len(calls) == n_frames, "a normal run must hash each frame once"
    assert len(set(calls)) == n_frames

    calls.clear()
    _run(
        corpus, tmp_path / "out-import", tmp_path / "store-import", "2x2",
        gauge_import={
            "gauge_path": str(computed["store"] / "gauge.json"),
            "origin_run_json": str(computed["store"] / "run.json"),
        },
    )
    assert len(calls) == n_frames, (
        "an import run hashed the corpus more than once "
        f"({len(calls)} reads for {n_frames} frames)"
    )
    assert len(set(calls)) == n_frames


# ---------------------------------------------------------------------------
# REWORK-1 — L1: the backend exclusion is locked by a test
# ---------------------------------------------------------------------------

def _forge_origin_backend(computed, target, backend):
    """Re-sign the computed artifact for an origin identity declaring ``backend``."""
    identity = json.loads((computed["store"] / "run.json").read_text())
    identity["science_contract"]["science"]["params"]["backend"] = backend
    identity["science_contract"]["backend"]["backend"] = backend
    zrs.validate_run_identity(identity)  # the forged origin is a VALID identity
    raw = json.loads((computed["store"] / "gauge.json").read_text())
    raw["identity_sha256"] = zrs.identity_sha256(identity)
    body = {key: value for key, value in raw.items() if key != "artifact_sha256"}
    raw["artifact_sha256"] = zart._self_digest(body)
    gauge_path = Path(target) / f"{backend}_gauge.json"
    run_path = Path(target) / f"{backend}_run.json"
    gauge_path.write_text(json.dumps(raw), encoding="utf-8")
    run_path.write_text(json.dumps(identity), encoding="utf-8")
    return gauge_path, run_path, identity


def test_l1_gpu_origin_artifact_is_accepted_by_a_cpu_run(tmp_path, corpus, computed):
    """L1: an artifact whose ORIGIN identity says ``backend=gpu`` imports into a
    CPU-effective run, and the identity's gauge sha256 is unchanged.

    What this proves: the photometric comparison deliberately ignores ``backend``
    and the import does not reject a backend difference; the gauge fingerprint
    is preserved verbatim.

    What it CANNOT prove: that a gauge actually computed on a GPU is numerically
    identical to the CPU one. There is no GPU here and the gauge path
    (``photometric.py``) has no GPU implementation at all, so the artifact's
    numbers are the CPU-computed ones by construction. CPU↔GPU parity remains a
    separate contract; if the gauge ever gains a GPU path, this exclusion must be
    re-examined.
    """
    gauge_path, run_path, forged = _forge_origin_backend(computed, tmp_path, "gpu")
    result = _run(
        corpus, tmp_path / "out", tmp_path / "store", "1x2",
        gauge_import={
            "gauge_path": str(gauge_path),
            "origin_run_json": str(run_path),
        },
    )
    computed_sha = computed["identity"]["science_contract"]["gauge"]["sha256"]
    assert result["identity"]["science_contract"]["gauge"]["sha256"] == computed_sha
    provenance = result["manifest"]["photometric_gauge"]["import"]
    assert provenance["origin_identity_sha256"] == zrs.identity_sha256(forged)
    assert provenance["gauge_sha256"] == computed_sha
    assert "computing global photometric gauge" not in "\n".join(result["logs"])


def test_l1_reverse_direction_is_accepted_too(computed):
    """L1 (reverse): origin ``backend=cpu`` imported by a run expecting ``gpu``."""
    kwargs = _origin(computed)
    params = dict(kwargs["expected_science_params"])
    params["backend"] = "gpu"
    kwargs["expected_science_params"] = params
    _gauge, frame_ids, reference, provenance = zart.import_gauge_artifact(
        str(computed["store"] / "gauge.json"), **kwargs
    )
    assert provenance["gauge_sha256"] == computed["identity"][
        "science_contract"]["gauge"]["sha256"]
    assert frame_ids == tuple(
        row["frame_id"] for row in computed["identity"]["science_contract"][
            "inputs"]["frames"])
    assert reference


# ---------------------------------------------------------------------------
# REWORK-1 — L2: a refused import leaves nothing behind
# ---------------------------------------------------------------------------

def test_l2_refused_import_leaves_nothing_on_disk(tmp_path, corpus, computed):
    tampered = tmp_path / "tampered"
    _tamper_one_frame(corpus, tampered)
    out = tmp_path / "out"
    assert not out.exists()
    with pytest.raises(zres.ResumeIncompatible):
        _run(
            tampered, out, tmp_path / "store", "2x2",
            gauge_import={
                "gauge_path": str(computed["store"] / "gauge.json"),
                "origin_run_json": str(computed["store"] / "run.json"),
            },
        )
    # Verification now runs before the cache root and the layout, and this
    # attempt created the output folder: it is gone, with its run log.
    assert not out.exists()


def test_l2_refusal_never_deletes_a_preexisting_output_folder(
    tmp_path, corpus, computed
):
    tampered = tmp_path / "tampered"
    _tamper_one_frame(corpus, tampered)
    out = tmp_path / "out"
    out.mkdir()
    marker = out / "user-file.txt"
    marker.write_text("keep me", encoding="utf-8")
    with pytest.raises(zres.ResumeIncompatible):
        _run(
            tampered, out, tmp_path / "store", "2x2",
            gauge_import={
                "gauge_path": str(computed["store"] / "gauge.json"),
                "origin_run_json": str(computed["store"] / "run.json"),
            },
        )
    assert marker.read_text(encoding="utf-8") == "keep me"
    # No layout/cache artifact was created before the refusal.
    assert not (out / "zegrid-cache").exists()
    assert not (out / MANIFEST).exists()


# ---------------------------------------------------------------------------
# ZM-ZEGRID-GAUGE-IMPORT-FAILFAST — cheap checks refuse with ZERO hashing
# ---------------------------------------------------------------------------

def _precheck_kwargs(computed, **overrides):
    kwargs = _origin(computed)
    out = {
        "origin_identity": kwargs["origin_identity"],
        "origin_run_json_path": kwargs["origin_run_json_path"],
        "expected_frame_ids": kwargs["expected_frame_ids"],
        "expected_canvas": kwargs["expected_canvas"],
        "expected_science_params": kwargs["expected_science_params"],
        "expected_mode": kwargs["expected_mode"],
    }
    out.update(overrides)
    return out


def _no_hash(monkeypatch):
    calls = []
    real = zrs.content_sha256_file

    def spy(path):
        calls.append(str(path))
        return real(path)

    monkeypatch.setattr(zrs, "content_sha256_file", spy)
    return calls


def test_precheck_science_mismatch_refuses_with_zero_hashes(computed, monkeypatch):
    calls = _no_hash(monkeypatch)
    params = dict(_precheck_kwargs(computed)["expected_science_params"])
    params["cell_gauge_mode"] = None  # origin carries "global"
    with pytest.raises(zres.ResumeIncompatible) as exc:
        zart.precheck_gauge_import(
            str(computed["store"] / "gauge.json"),
            **_precheck_kwargs(computed, expected_science_params=params),
        )
    assert exc.value.reasons[0]["section"] == "science"
    assert "'cell_gauge_mode'" in str(exc.value)
    assert calls == [], "a science mismatch must not read/hash any frame"


def test_precheck_canvas_mismatch_refuses_with_zero_hashes(computed, monkeypatch):
    calls = _no_hash(monkeypatch)
    canvas = dict(_precheck_kwargs(computed)["expected_canvas"])
    canvas["shape"] = [canvas["shape"][0] + 1, canvas["shape"][1]]
    with pytest.raises(zres.ResumeIncompatible) as exc:
        zart.precheck_gauge_import(
            str(computed["store"] / "gauge.json"),
            **_precheck_kwargs(computed, expected_canvas=canvas),
        )
    assert exc.value.reasons[0]["section"] == "canvas"
    assert calls == []


def test_precheck_frame_order_mismatch_refuses_with_zero_hashes(computed, monkeypatch):
    calls = _no_hash(monkeypatch)
    ids = list(_precheck_kwargs(computed)["expected_frame_ids"])
    with pytest.raises(zres.ResumeIncompatible) as exc:
        zart.precheck_gauge_import(
            str(computed["store"] / "gauge.json"),
            **_precheck_kwargs(computed, expected_frame_ids=list(reversed(ids))),
        )
    assert exc.value.reasons[0]["section"] == "inputs"
    assert calls == []


def test_precheck_gauge_mode_mismatch_refuses_with_zero_hashes(computed, monkeypatch):
    calls = _no_hash(monkeypatch)
    with pytest.raises(zres.ResumeIncompatible) as exc:
        zart.precheck_gauge_import(
            str(computed["store"] / "gauge.json"),
            **_precheck_kwargs(computed, expected_mode="per_cell"),
        )
    assert exc.value.reasons[0]["section"] == "gauge"
    assert calls == []


def test_import_still_requires_the_content_rows_no_hole(computed):
    kwargs = _origin(computed)
    with pytest.raises(zres.ResumeIncompatible) as exc:
        zart.import_gauge_artifact(
            str(computed["store"] / "gauge.json"),
            **{**kwargs, "expected_frame_rows": None},
        )
    assert exc.value.reasons[0]["section"] == "inputs"
    truncated = list(kwargs["expected_frame_rows"])[:-1]
    with pytest.raises(zres.ResumeIncompatible) as exc:
        zart.import_gauge_artifact(
            str(computed["store"] / "gauge.json"),
            **{**kwargs, "expected_frame_rows": truncated},
        )
    assert exc.value.reasons[0]["section"] == "inputs"
    assert "frame count differs" in str(exc.value)


def test_end_to_end_science_mismatch_hashes_no_frame(
    tmp_path, corpus, computed, monkeypatch
):
    """The real defect: the whole corpus was read before a one-token refusal."""
    calls = _no_hash(monkeypatch)
    out, store = tmp_path / "out", tmp_path / "store"
    with pytest.raises(zres.ResumeIncompatible) as exc:
        _run(
            corpus, out, store, "2x2",
            gauge_import={
                "gauge_path": str(computed["store"] / "gauge.json"),
                "origin_run_json": str(computed["store"] / "run.json"),
            },
            overrides={"cell_gauge_mode": "auto"},
        )
    assert exc.value.reasons[0]["section"] == "science"
    assert "cell_gauge_mode" in str(exc.value)
    assert calls == [], "the import refusal must happen before any frame read"
    assert not (out / MANIFEST).is_file()
