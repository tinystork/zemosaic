"""ZM-ZEGRID-ROOT-DELIVERABLES-20261031 — finished FITS at the OUTPUT ROOT.

Application consistency with Classic/SDS: a finished ZeGrid run DELIVERS its
verified products at the output root under the Classic-compatible names, while
the immutable generation directory remains the durable archive (resume,
``CURRENT.json``, verification). The delivery is an atomic copy: a temp file,
an fsync, then ONE ``os.replace``; a prior root file is never torn.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for _path in (SRC, REPO_ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from zemosaic.core.zegrid import run_publication as zpub  # noqa: E402

CLASSIC_NAMES = {
    zpub.ROLE_SCIENCE: "mosaic_grid_science.fits",
    zpub.ROLE_AESTHETIC: "mosaic_grid_aesthetic.fits",
    zpub.ROLE_COVERAGE: "mosaic_grid_coverage.fits",
    zpub.ROLE_UINT16: "mosaic_grid_uint16.fits",
    zpub.ROLE_MANIFEST: "zegrid_manifest.json",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_generation(name: str, directory: Path, contents: dict[str, bytes]):
    """A PublishedGeneration whose members are real files in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    members = {}
    for role, data in contents.items():
        filename = CLASSIC_NAMES[role]
        (directory / filename).write_bytes(data)
        members[role] = {
            "file": filename,
            "sha256": hashlib.sha256(data).hexdigest(),
            "nbytes": len(data),
        }
    return zpub.PublishedGeneration(
        run_id="root-run",
        generation=name,
        generation_dir=directory,
        pointer_path=directory.parent / zpub.POINTER_FILENAME,
        publication_path=directory / zpub.PUBLICATION_FILENAME,
        publication_sha256="a" * 64,
        members=members,
        required_roles=(zpub.ROLE_SCIENCE, zpub.ROLE_COVERAGE, zpub.ROLE_MANIFEST),
        optional_roles={},
        created_unix_ns=1,
        identity_sha256=None,
        verified=True,
    )


def _staged_temp_files(root: Path) -> list[Path]:
    tmp = root / "tmp"
    return sorted(p for p in tmp.iterdir()) if tmp.is_dir() else []


def _run(tmp_path, **kwargs):
    from test_zegrid_visible_artifacts import _run as real_run

    return real_run(tmp_path, **kwargs)


def test_real_run_delivers_verified_products_at_the_output_root(tmp_path):
    out, _store, _corpus = _run(tmp_path)
    payload = json.loads(
        (out / zpub.PRODUCTS_JSON_FILENAME).read_text(encoding="utf-8")
    )
    gen_dir = Path(payload["generation_dir"])
    root_files = payload["root_files"]
    assert root_files, "no root deliverables were recorded"
    # A default tiny run has no aesthetic export -> Classic science name.
    assert (out / "mosaic_grid.fits").is_file()
    assert set(root_files) >= {"science", "coverage", "manifest"}
    assert "aesthetic" not in root_files
    for role, row in root_files.items():
        root_path = Path(row["path"])
        assert root_path.parent == out.resolve(), root_path
        assert root_path.is_file()
        assert _sha(root_path) == row["sha256"]
        assert root_path.stat().st_size == int(row["nbytes"])
        member = gen_dir / payload["products"][role]["file"]
        assert _sha(member) == row["sha256"]
    # The immutable archive is intact and still lists every member digest.
    publication = json.loads(
        (gen_dir / zpub.PUBLICATION_FILENAME).read_text(encoding="utf-8")
    )
    for row in publication["members"].values():
        assert _sha(gen_dir / row["file"]) == row["sha256"]
    # PRODUCTS.txt names the ROOT paths and no longer denies a root copy.
    text = (out / zpub.PRODUCTS_TEXT_FILENAME).read_text(encoding="utf-8")
    assert str((out / "mosaic_grid.fits").resolve()) in text
    assert str(gen_dir) in text
    assert "no copy is made" not in text
    assert "OUTPUT ROOT" in text


def test_second_generation_replaces_root_files_and_leaves_gen1_intact(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    gen1 = tmp_path / "archive" / "gen-1"
    gen2 = tmp_path / "archive" / "gen-2"
    first = _fake_generation(
        "gen-1",
        gen1,
        {
            zpub.ROLE_SCIENCE: b"science-one",
            zpub.ROLE_COVERAGE: b"coverage-one",
            zpub.ROLE_MANIFEST: b"manifest-one",
        },
    )
    second = _fake_generation(
        "gen-2",
        gen2,
        {
            zpub.ROLE_SCIENCE: b"science-two",
            zpub.ROLE_COVERAGE: b"coverage-two",
            zpub.ROLE_MANIFEST: b"manifest-two",
        },
    )
    gen1_before = {row["file"]: _sha(gen1 / row["file"]) for row in first.members.values()}

    delivery1 = zpub.publish_root_deliverables(out, first)
    assert delivery1["failures"] == []
    assert (out / "mosaic_grid_science.fits").read_bytes() == b"science-one"
    zpub.write_visible_products(out, first, root_files=delivery1["root_files"])

    delivery2 = zpub.publish_root_deliverables(out, second)
    assert delivery2["failures"] == []
    zpub.write_visible_products(out, second, root_files=delivery2["root_files"])
    assert (out / "mosaic_grid_science.fits").read_bytes() == b"science-two"
    assert (out / "mosaic_grid_coverage.fits").read_bytes() == b"coverage-two"
    assert (out / "zegrid_manifest.json").read_bytes() == b"manifest-two"
    # The previous generation is never touched.
    gen1_after = {row["file"]: _sha(gen1 / row["file"]) for row in first.members.values()}
    assert gen1_after == gen1_before
    assert _staged_temp_files(out) == []
    # Markers name the ROOT paths of gen-2 with the gen-2 digests.
    payload = json.loads(
        (out / zpub.PRODUCTS_JSON_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["generation"] == "gen-2"
    assert payload["generation_dir"] == str(gen2.resolve())
    assert payload["root_files"]["science"]["sha256"] == delivery2["root_files"]["science"]["sha256"]
    text = (out / zpub.PRODUCTS_TEXT_FILENAME).read_text(encoding="utf-8")
    assert str((out / "mosaic_grid_science.fits").resolve()) in text


def test_delivery_failure_keeps_prior_root_bytes_and_reports(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    gen1 = tmp_path / "archive" / "gen-1"
    gen2 = tmp_path / "archive" / "gen-2"
    first = _fake_generation(
        "gen-1",
        gen1,
        {
            zpub.ROLE_SCIENCE: b"science-one",
            zpub.ROLE_COVERAGE: b"coverage-one",
            zpub.ROLE_MANIFEST: b"manifest-one",
        },
    )
    second = _fake_generation(
        "gen-2",
        gen2,
        {
            zpub.ROLE_SCIENCE: b"science-two",
            zpub.ROLE_COVERAGE: b"coverage-two",
            zpub.ROLE_MANIFEST: b"manifest-two",
        },
    )
    zpub.publish_root_deliverables(out, first)
    coverage_before = (out / "mosaic_grid_coverage.fits").read_bytes()

    real_replace = zpub.os.replace

    def flaky(src, dst):
        if Path(dst).name == "mosaic_grid_coverage.fits":
            raise OSError("simulated delivery failure")
        return real_replace(src, dst)

    monkeypatch.setattr(zpub.os, "replace", flaky)
    delivery = zpub.publish_root_deliverables(out, second)
    assert delivery["failures"] and "coverage" in delivery["failures"][0]
    # A convenience copy never raises; the prior root file stays byte-identical.
    assert (out / "mosaic_grid_coverage.fits").read_bytes() == coverage_before
    # No temp file is left behind, and the new generation is intact.
    assert _staged_temp_files(out) == []
    assert (gen2 / "mosaic_grid_coverage.fits").read_bytes() == b"coverage-two"
    # The markers are truthful: the failed role is NOT claimed as delivered.
    zpub.write_visible_products(out, second, root_files=delivery["root_files"])
    payload = json.loads(
        (out / zpub.PRODUCTS_JSON_FILENAME).read_text(encoding="utf-8")
    )
    assert "coverage" not in payload["root_files"]
    text = (out / zpub.PRODUCTS_TEXT_FILENAME).read_text(encoding="utf-8")
    assert "root delivery failed" in text


def test_already_complete_reconcile_redelivers_without_recomputation(tmp_path):
    out, store, corpus = _run(tmp_path)
    gen_dir = Path(
        json.loads(
            (out / zpub.PRODUCTS_JSON_FILENAME).read_text(encoding="utf-8")
        )["generation_dir"]
    )
    root_names = (
        "mosaic_grid.fits",
        "mosaic_grid_coverage.fits",
        "zegrid_manifest.json",
    )
    expected = {name: _sha(gen_dir / name) for name in root_names}
    # Simulate a run finished BEFORE this change: root FITS absent.
    for name in root_names:
        (out / name).unlink(missing_ok=True)
    lines: list[str] = []
    _run(
        tmp_path,
        resume_mode="auto",
        store_dir=store,
        out=out,
        corpus=corpus,
        lines=lines,
    )
    assert any("ALREADY COMPLETE" in line for line in lines), lines[-5:]
    for name in root_names:
        assert (out / name).is_file(), name
        assert _sha(out / name) == expected[name]
    # Idempotent: a second reconcile leaves the root bytes unchanged.
    again = {name: (out / name).read_bytes() for name in expected}
    _run(
        tmp_path,
        resume_mode="auto",
        store_dir=store,
        out=out,
        corpus=corpus,
        lines=lines,
    )
    assert {name: (out / name).read_bytes() for name in expected} == again
