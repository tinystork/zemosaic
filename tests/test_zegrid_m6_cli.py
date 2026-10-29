"""ZM-ZEGRID-M6 Part B — headless CLI (no Qt) and one REAL tiny run.

Cheap checks:
* importing ``zemosaic.zegrid_cli`` never imports PySide6/PyQt;
* argument validation refuses bad/contradictory options with exit 2;
* one real 4-frame 256x256 / 2x2 / CPU / 1-worker run produces FITS + manifest
  with exit 0 and a JSON summary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from zemosaic import zegrid_cli as cli

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
CORPUS_TOOL = REPO_ROOT / "tools" / "zegrid_m6" / "make_tiny_corpus.py"


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def _run(args, **kwargs):
    kwargs.setdefault("env", _env())
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("cwd", str(REPO_ROOT))
    return subprocess.run([sys.executable, "-m", "zemosaic.zegrid_cli", *args], **kwargs)


def _make_corpus(path: Path) -> Path:
    proc = subprocess.run(
        [sys.executable, str(CORPUS_TOOL), str(path)],
        env=_env(), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return path


def _publish(tmp_path: Path, corpus: Path, name: str = "out", extra=()) -> dict:
    out = tmp_path / name
    summary = tmp_path / f"{name}.json"
    proc = _run(
        [
            "--input", str(corpus), "--output", str(out),
            "--workers", "1", "--layout", "2x2", "--halo-px", "8",
            "--gauge-mode", "global", "--json-summary", str(summary),
            *extra,
        ],
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return {"out": out, "summary": json.loads(summary.read_text())}


def _publication_digests(out: Path) -> dict[str, str]:
    import hashlib

    digests = {}
    for path in sorted(out.glob("mosaic_grid*.fits")) + sorted(
        out.glob("zegrid_manifest.json")
    ) + sorted(out.glob("zegrid_run.log")):
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def test_cli_import_does_not_import_qt():
    code = (
        "import sys\n"
        "import zemosaic.zegrid_cli\n"
        "bad = sorted(m for m in sys.modules "
        "if m.startswith('PySide6') or m.startswith('PyQt'))\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=_env(), capture_output=True, text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "ok"


def test_cli_help_exits_zero():
    proc = _run(["--help"])
    assert proc.returncode == 0
    assert "exit codes" in proc.stdout.lower()


def test_cli_refuses_missing_input(tmp_path):
    proc = _run(["--input", str(tmp_path / "nope"), "--output", str(tmp_path / "out")])
    assert proc.returncode == 2
    assert "REFUSED" in proc.stderr


def test_cli_refuses_empty_input(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = _run(["--input", str(empty), "--output", str(tmp_path / "out")])
    assert proc.returncode == 2
    assert "no FITS" in proc.stderr


def test_cli_refuses_budget_without_store(tmp_path):
    corpus = _make_corpus(tmp_path / "in")
    proc = _run([
        "--input", str(corpus), "--output", str(tmp_path / "out"),
        "--durable-store-budget-bytes", "1048576",
    ])
    assert proc.returncode == 2
    assert "requires --durable-store-dir" in proc.stderr


def test_cli_refuses_deadline_without_recovery(tmp_path):
    corpus = _make_corpus(tmp_path / "in")
    proc = _run([
        "--input", str(corpus), "--output", str(tmp_path / "out"),
        "--resource-recovery-deadline-s", "5",
    ])
    assert proc.returncode == 2
    assert "requires --resource-recovery" in proc.stderr


def test_cli_refuses_resume_auto_without_store(tmp_path):
    corpus = _make_corpus(tmp_path / "in")
    proc = _run([
        "--input", str(corpus), "--output", str(tmp_path / "out"),
        "--resume-mode", "auto",
    ])
    assert proc.returncode == 2
    assert "requires --durable-store-dir" in proc.stderr


def test_cli_refuses_store_inside_output(tmp_path):
    corpus = _make_corpus(tmp_path / "in")
    out = tmp_path / "out"
    proc = _run([
        "--input", str(corpus), "--output", str(out),
        "--durable-store-dir", str(out / "store"),
    ])
    assert proc.returncode == 2
    assert "must not be inside the output" in proc.stderr


def test_cli_refuses_invalid_halo(tmp_path):
    corpus = _make_corpus(tmp_path / "in")
    proc = _run([
        "--input", str(corpus), "--output", str(tmp_path / "out"),
        "--halo-px", "7",
    ])
    assert proc.returncode == 2
    assert "zegrid_halo_px" in proc.stderr


def test_publication_freshness_rejects_a_stale_manifest(tmp_path):
    manifest = tmp_path / "zegrid_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    old_ns = time.time_ns() - 60 * 10**9
    os.utime(manifest, ns=(old_ns, old_ns))
    # A reused output folder: the manifest predates this run -> stale.
    assert cli._publication_is_fresh(manifest, time.time_ns()) is False
    # The same file is fresh for a run that started before it was written.
    assert cli._publication_is_fresh(manifest, old_ns - 1) is True
    # A missing manifest is never "fresh".
    assert cli._publication_is_fresh(tmp_path / "absent.json", 0) is False


def test_cli_real_tiny_run_produces_fits_and_manifest(tmp_path):
    corpus = _make_corpus(tmp_path / "in")
    out = tmp_path / "out"
    summary_path = tmp_path / "summary.json"
    proc = _run(
        [
            "--input", str(corpus),
            "--output", str(out),
            "--workers", "1",
            "--layout", "2x2",
            "--halo-px", "8",
            "--gauge-mode", "global",
            "--json-summary", str(summary_path),
        ],
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]

    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "COMPLETE"
    assert summary["exit_code"] == 0
    assert summary["resource_recovery_enabled"] is False
    assert summary["overwrite"] is False
    assert summary["previous_publication"]["found"] is False
    assert summary["gpu"] is False
    assert summary["publication"]["science"] is not None

    science = Path(summary["publication"]["science"])
    manifest_path = Path(summary["publication"]["manifest"])
    assert science.exists() and science.suffix == ".fits"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["engine"] == "zegrid"
    assert manifest["n_frames"] == 4
    assert manifest["layout"]["nx"] * manifest["layout"]["ny"] == 4
    assert manifest["outputs"]["science"] == science.name
    assert len(manifest["complete_cells"]) == 4

    # A stop was never requested, no durable store, no publication pointer.
    assert summary["stop"]["requested"] is False
    assert not (out / ".zegrid_publications").exists()


# ---------------------------------------------------------------------------
# REWORK-1 C1 — a reused output folder is never silently overwritten
# ---------------------------------------------------------------------------


def test_second_run_into_same_output_is_refused_and_files_untouched(tmp_path):
    corpus = _make_corpus(tmp_path / "in")
    published = _publish(tmp_path, corpus)
    before = _publication_digests(published["out"])
    assert before, "the first run must have published files"

    proc = _run(
        [
            "--input", str(corpus), "--output", str(published["out"]),
            "--workers", "1", "--layout", "2x2", "--halo-px", "8",
            "--gauge-mode", "global",
        ],
        timeout=120,
    )
    assert proc.returncode == 2
    assert "previous ZeGrid publication" in proc.stderr
    assert "--overwrite" in proc.stderr
    # The refusal happened BEFORE any product work or file write.
    combined = proc.stdout + proc.stderr
    assert "ZeGrid:" not in combined
    assert _publication_digests(published["out"]) == before


def test_overwrite_is_explicit_and_recorded(tmp_path):
    corpus = _make_corpus(tmp_path / "in")
    published = _publish(tmp_path, corpus)
    before = _publication_digests(published["out"])

    proc = _run(
        [
            "--input", str(corpus), "--output", str(published["out"]),
            "--workers", "1", "--layout", "2x2", "--halo-px", "8",
            "--gauge-mode", "global", "--overwrite",
            "--json-summary", str(tmp_path / "ow.json"),
        ],
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    summary = json.loads((tmp_path / "ow.json").read_text())
    assert summary["status"] == "COMPLETE"
    assert summary["overwrite"] is True
    assert summary["previous_publication"]["found"] is True
    assert summary["previous_publication"]["zegrid"] is True
    assert "zegrid_manifest.json" in summary["previous_publication"]["evidence"]
    assert summary["previous_publication"]["foreign"] == []
    # The replacement really happened (the manifest contains fresh timing data).
    assert _publication_digests(published["out"]) != before


def test_overwrite_with_resume_auto_is_refused(tmp_path):
    corpus = _make_corpus(tmp_path / "in")
    proc = _run(
        [
            "--input", str(corpus), "--output", str(tmp_path / "out"),
            "--durable-store-dir", str(tmp_path / "store"),
            "--resume-mode", "auto", "--overwrite",
        ],
        timeout=120,
    )
    assert proc.returncode == 2
    assert "--overwrite refuses --resume-mode=auto" in proc.stderr


def test_overwrite_refuses_foreign_content_and_preserves_it(tmp_path):
    corpus = _make_corpus(tmp_path / "in")
    published = _publish(tmp_path, corpus)
    before = _publication_digests(published["out"])
    foreign = published["out"] / "notes.txt"
    foreign.write_text("not ours\n", encoding="utf-8")

    proc = _run(
        [
            "--input", str(corpus), "--output", str(published["out"]),
            "--workers", "1", "--layout", "2x2", "--halo-px", "8",
            "--gauge-mode", "global", "--overwrite",
        ],
        timeout=120,
    )
    assert proc.returncode == 2
    assert "not ours to destroy" in proc.stderr
    assert foreign.read_text(encoding="utf-8") == "not ours\n"
    assert _publication_digests(published["out"]) == before


def test_nested_mount_publication_is_detected_and_refused(tmp_path):
    """Mount-segregated runs publish into grid_EQ/grid_ALTZ; a reused output
    folder must detect those nested publications too (REWORK-2)."""
    corpus = _make_corpus(tmp_path / "in")
    out = tmp_path / "out"
    child = out / "grid_EQ"
    child.mkdir(parents=True)
    fits = child / "mosaic_grid.fits"
    fits.write_bytes(b"previous mosaic")
    manifest = child / "zegrid_manifest.json"
    manifest.write_text(json.dumps({"engine": "zegrid", "n_frames": 4}))

    proc = _run(
        [
            "--input", str(corpus), "--output", str(out),
            "--workers", "1", "--layout", "2x2", "--halo-px", "8",
            "--gauge-mode", "global",
        ],
        timeout=120,
    )
    assert proc.returncode == 2
    assert "grid_EQ/mosaic_grid.fits" in proc.stderr
    assert "ZeGrid:" not in (proc.stdout + proc.stderr)
    assert fits.read_bytes() == b"previous mosaic"


def test_store_overwrite_publishes_a_distinct_generation(tmp_path):
    corpus = _make_corpus(tmp_path / "in")
    first = _publish(tmp_path, corpus, name="out", extra=(
        "--durable-store-dir", str(tmp_path / "storeA"),
    ))
    first_pointer = first["summary"]["publication"]["pointer"]
    assert first_pointer is not None

    second = _publish(tmp_path, corpus, name="out", extra=(
        "--durable-store-dir", str(tmp_path / "storeB"),
        "--overwrite",
    ))
    second_pointer = second["summary"]["publication"]["pointer"]
    assert second_pointer is not None
    assert second_pointer["path"] != first_pointer["path"]
    assert second_pointer["run_id"] != first_pointer["run_id"]
    assert second["summary"]["previous_publication"]["pointer"] == first_pointer

