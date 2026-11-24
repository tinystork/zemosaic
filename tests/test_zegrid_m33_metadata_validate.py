"""ZM-M33 P0P1 (P0) — the read-only M33 metadata validator.

The validator is meant to run on the Windows evidence copy BEFORE any gauge
import/resume witness. These tests exercise it on a synthetic-but-CANONICAL
bundle produced by the product writers themselves (``build_run_identity``,
``write_gauge_artifact``, ``write_geometry_artifact``), so the happy path and
the fail-closed paths are both covered without any FITS file.

No FITS is created or read anywhere: the frame rows are synthetic content
digests, which is exactly what the fail-fast precheck is forbidden to require.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "zegrid_m33_metadata_validate.py"


def _load_tool(path: Path, name: str):
    """Load the validator by path (never mutate sys.path for other tests)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_tool(TOOL_PATH, "zegrid_m33_metadata_validate_under_test")

from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import run_artifacts as zart  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402
from zemosaic.core.canonical_streaming import FixedNormalization  # noqa: E402

FRAME_IDS = ("frame00000001.fits", "frame00000002.fits")


def _canvas():
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [23.4621, 30.6599]
    wcs.wcs.crpix = [20.0, 15.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * 0.0007
    wcs.array_shape = (30, 40)
    return zg.GlobalCanvas(
        canvas_id="m33-validator-fixture",
        wcs_header=wcs.to_header(relax=True).tostring(),
        width=40,
        height=30,
        resolution_deg=0.0007,
    )


def _gauge():
    coefficients = np.zeros((len(FRAME_IDS), 1, 2), dtype=np.float64)
    coefficients[:, 0, 0] = 1.0
    coefficients[:, 0, 1] = 0.5
    return FixedNormalization(
        reference_index=0,
        coefficients=coefficients,
        norm_active=np.ones(len(FRAME_IDS), dtype=bool),
        weights=np.ones(len(FRAME_IDS), dtype=np.float64),
        weight_active=np.ones(len(FRAME_IDS), dtype=bool),
        exclusions=(),
    )


def _build_bundle(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    canvas = _canvas()
    layout = zg.build_layout(canvas, 1, 1)
    cell_ctxs = []
    for row, col, bounds in layout.iter_cells(canvas):
        cid = zg.cell_id(row, col)
        cell = zg.ZeGridCell(
            cid, canvas.canvas_id, layout.layout_id, row, col, bounds
        )
        patch = zg.build_patch(canvas, cell, 16)
        mem = zg.CellMembership(
            cell_id=cid, core_ids=FRAME_IDS, patch_ids=FRAME_IDS
        )
        cell_ctxs.append((row, col, cell, patch, mem))
    fingerprints = zart.geometry_fingerprints(
        canvas=canvas, layout={"nx": 1, "ny": 1}, cell_ctxs=cell_ctxs,
        halo_px=16,
    )
    gauge = _gauge()
    gauge_sha = zart.gauge_fingerprint(
        gauge=gauge,
        mode="global",
        frame_ids=FRAME_IDS,
        reference_frame_id=FRAME_IDS[0],
    )
    rows = [
        {
            "frame_id": fid,
            "content_sha256": "%064x" % (index + 1),
            "nbytes": 1024 * (index + 1),
            "order": index,
            "role": "light",
        }
        for index, fid in enumerate(FRAME_IDS)
    ]
    identity = zrs.build_run_identity(
        run_id="m33-validator-fixture-run",
        inputs={"frames": rows},
        canvas={
            "shape": [canvas.height, canvas.width],
            "wcs_sha256": fingerprints["canvas_wcs_sha256"],
            "geometry_sha256": fingerprints["canvas_geometry_sha256"],
        },
        grid={
            "nx": 1,
            "ny": 1,
            "tile_px": 128,
            "geometry_sha256": fingerprints["grid_geometry_sha256"],
            "cell_ids_sha256": fingerprints["grid_cell_ids_sha256"],
        },
        halo={"halo_px": 16},
        memberships={
            "sha256": fingerprints["memberships_sha256"],
            "n_cells": fingerprints["memberships_n_cells"],
        },
        gauge={
            "mode": "global",
            "sha256": gauge_sha,
            "reference_frame_id": FRAME_IDS[0],
        },
        science={
            "sha256": "b" * 64,
            "contract_version": "1",
            "params": {
                "normalization": "sky_mean",
                "weighting": "noise_variance",
                "cell_gauge_mode": "global",
            },
        },
        dtypes={"planes": {"science_core": ["<f4"]}},
        backend={"backend": "cpu", "tile_px": 128, "parity_qualified": False},
        execution_context={
            "available_ram_bytes": 18941517824,
            "workers": 1,
            "prefetch_units": 0,
            "host": "fixture",
            "note": "metadata validator fixture (no pixels)",
        },
    )
    (root / "run.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    zart.write_gauge_artifact(
        root,
        run_id=identity["run_id"],
        identity=identity,
        gauge=gauge,
        mode="global",
        frame_ids=FRAME_IDS,
        reference_frame_id=FRAME_IDS[0],
    )
    zart.write_geometry_artifact(
        root,
        run_id=identity["run_id"],
        identity=identity,
        canvas=canvas,
        layout={"nx": 1, "ny": 1},
        cell_ctxs=cell_ctxs,
        halo_px=16,
    )
    (root / "state.json").write_text(
        json.dumps(
            {
                "run_id": identity["run_id"],
                "gauge_sha256": gauge_sha,
                "finalized": False,
                "outputs": {},
                "working_set": {"class": "external_spool"},
                "backend": "cpu",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_validator_accepts_a_canonical_bundle_without_hashing_fits(tmp_path):
    root = _build_bundle(tmp_path / "bundle")
    report = validator.validate(bundle_dir=root)
    assert report["ok"] is True, json.dumps(report, indent=2)
    ids = {item["id"] for item in report["checks"] if item["ok"]}
    assert ids == {
        "files.present",
        "run.identity",
        "gauge.artifact",
        "gauge.import_precheck_no_fits",
        "geometry.artifact",
        "state.cross_reference",
    }
    precheck = report["gauge_import_precheck"]
    assert precheck["frame_count"] == len(FRAME_IDS)
    assert precheck["mode"] == "global"
    assert precheck["source"] == "imported_artifact"
    # The validator must be strictly read-only.
    before = sorted(p.name for p in root.iterdir())
    validator.validate(bundle_dir=root)
    assert sorted(p.name for p in root.iterdir()) == before


def test_validator_reports_expected_digest_drift(tmp_path):
    root = _build_bundle(tmp_path / "bundle")
    good = validator.validate(bundle_dir=root)["file_sha256"]
    report = validator.validate(
        bundle_dir=root,
        expected_hashes={
            "gauge.json": good["gauge.json"],
            "run.json": "0" * 64,
        },
    )
    assert report["ok"] is False
    assert "files.expected_sha256" in report["failed_checks"]
    assert report["checks"][1]["detail"]["run.json"]["observed"] == good["run.json"]


# ---------------------------------------------------------------------------
# Fail-closed paths
# ---------------------------------------------------------------------------

def test_validator_refuses_a_missing_artifact(tmp_path):
    root = _build_bundle(tmp_path / "bundle")
    (root / "gauge.json").unlink()
    report = validator.validate(bundle_dir=root)
    assert report["ok"] is False
    assert report["failed_checks"] == ["files.present"]


def test_validator_refuses_a_tampered_state_gauge_hash(tmp_path):
    root = _build_bundle(tmp_path / "bundle")
    state = json.loads((root / "state.json").read_text())
    state["gauge_sha256"] = "c" * 64
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")
    report = validator.validate(bundle_dir=root)
    assert report["ok"] is False
    assert report["failed_checks"] == ["state.cross_reference"]


def test_validator_refuses_a_corrupted_gauge_document(tmp_path):
    root = _build_bundle(tmp_path / "bundle")
    payload = json.loads((root / "gauge.json").read_text())
    payload["weights"] = [0.0] * len(FRAME_IDS)   # breaks the 90 %-margin? no: 0 weight
    (root / "gauge.json").write_text(json.dumps(payload), encoding="utf-8")
    report = validator.validate(bundle_dir=root)
    assert report["ok"] is False
    assert {"gauge.artifact", "gauge.import_precheck_no_fits"} & set(
        report["failed_checks"]
    )


def test_validator_refuses_an_identity_binding_change(tmp_path):
    """Editing the identity invalidates the artifact binding first.

    The photometric comparison itself is the SAME product check that a real
    cross-run import performs (``precheck_gauge_import`` against the TARGET
    run's science parameters); a self-consistent bundle can only ever be
    checked for binding here.
    """
    root = _build_bundle(tmp_path / "bundle")
    identity = json.loads((root / "run.json").read_text())
    identity["science_contract"]["science"]["params"]["normalization"] = (
        "median"
    )
    (root / "run.json").write_text(json.dumps(identity), encoding="utf-8")
    report = validator.validate(bundle_dir=root)
    assert report["ok"] is False
    assert "gauge.import_precheck_no_fits" in report["failed_checks"]
    detail = next(
        item for item in report["checks"]
        if item["id"] == "gauge.import_precheck_no_fits"
    )
    assert "another identity" in detail["detail"]


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------

def test_cli_exit_codes(tmp_path, capsys):
    root = _build_bundle(tmp_path / "bundle")
    assert validator.main(["--bundle-dir", str(root)]) == 0
    capsys.readouterr()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert validator.main(["--bundle-dir", str(empty)]) == 2
    out = tmp_path / "report.json"
    assert validator.main(["--bundle-dir", str(root), "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["ok"] is True
    assert payload["mode"] == "read_only_metadata_no_fits_hashing"


# ---------------------------------------------------------------------------
# Subprocess provenance: the documented CLI must use THIS checkout
# ---------------------------------------------------------------------------

def _clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def test_cli_pins_the_checkout_without_pythonpath(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            "tools/zegrid_m33_metadata_validate.py",
            "--bundle-dir",
            str(empty),
        ],
        cwd=str(REPO_ROOT),
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 2, proc.stderr
    report = json.loads(proc.stdout)
    src = (REPO_ROOT / "src").resolve()
    code_source = report["code_source"]
    assert code_source["locked"] is True
    assert Path(code_source["zemosaic_file"]).resolve() == (
        src / "zemosaic" / "__init__.py"
    )
    assert any(
        "zemosaic/src" in entry
        for entry in code_source["foreign_sys_path_entries_removed"]
    ), code_source
    assert report["failed_checks"] == ["files.present"]


def test_cli_refuses_a_tree_without_the_local_package(tmp_path):
    tree = tmp_path / "tree"
    (tree / "tools").mkdir(parents=True)
    shutil.copy2(TOOL_PATH, tree / "tools" / "zegrid_m33_metadata_validate.py")
    bundle = tree / "bundle"
    bundle.mkdir()
    out = tree / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "tools/zegrid_m33_metadata_validate.py",
            "--bundle-dir",
            str(bundle),
            "--out",
            str(out),
        ],
        cwd=str(tree),
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 4, (proc.returncode, proc.stdout, proc.stderr)
    assert "REFUSED" in proc.stderr
    assert not out.exists()
