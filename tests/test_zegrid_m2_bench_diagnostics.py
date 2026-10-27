"""M2 REWORK-3 — bench diagnostic selection regression.

`collect_store_diagnostics` must surface the REAL durable-store failure (the
`open_stage` / `open_error` produced by the engine) and never a source comment
that merely mentions "durable store". The previous implementation substring-
matched the whole log, so a docstring/comment could occupy the 15-line cap and
hide a `FileNotFoundError` from the identity build (Windows probe 03).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = REPO_ROOT / "tools" / "zegrid_resume_m2" / "m2_bench.py"


def _load_bench():
    spec = importlib.util.spec_from_file_location("_m2_bench_for_tests", BENCH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


bench = _load_bench()


def _result(log_path: Path, suite: str = "m2_real_store_run") -> dict:
    return {
        "suite": suite,
        "log": str(log_path),
        "failed": 1,
        "errors": 0,
        "returncode": 1,
    }


def test_m2_bench_surfaces_the_real_error_over_source_comments(tmp_path):
    log = tmp_path / "suite.log"
    noise = "\n".join(
        f"        # about the durable store's own sites ({index})"
        for index in range(40)
    )
    log.write_text(
        "src/zemosaic/zemosaic_zegrid_mode.py:9065: in _r55_digest_tree\n"
        f"{noise}\n"
        "E   Failed: durable store did not become active: "
        "stage='identity_build' "
        "open_error=\"FileNotFoundError: [Errno 2] frame_0000.fits\"\n"
        "E   _R55RunStoreIdentityError: durable identity: source unreadable\n",
        encoding="utf-8",
    )
    hits = bench.collect_store_diagnostics([_result(log)], limit=15)
    assert hits, "no diagnostic selected"
    assert len(hits) <= 15
    assert "identity_build" in hits[0] or "FileNotFoundError" in hits[0], hits
    assert not any("about the durable store" in hit for hit in hits), hits
    assert any("FileNotFoundError" in hit for hit in hits)


def test_m2_bench_ignores_comments_that_only_mention_the_store(tmp_path):
    log = tmp_path / "suite.log"
    log.write_text(
        "\n".join(
            f"        # the durable store comment line {index}" for index in range(30)
        )
        + "\nE   assert 1 == 2\n",
        encoding="utf-8",
    )
    assert bench.collect_store_diagnostics([_result(log)]) == []


def test_m2_bench_surfaces_engine_emit_lines(tmp_path):
    log = tmp_path / "suite.log"
    log.write_text(
        "ZeGrid: durable run store UNAVAILABLE — stage=identity_build "
        "FileNotFoundError: [Errno 2] No such file or directory: "
        "'frame_0000.fits'. Continuing with in-memory conservation only.\n"
        "E   RuntimeError: store not active\n",
        encoding="utf-8",
    )
    hits = bench.collect_store_diagnostics([_result(log)])
    assert hits, hits
    assert "stage=identity_build" in hits[0]
    assert "FileNotFoundError" in hits[0]


def test_m2_bench_cap_cannot_hide_a_strong_diagnostic(tmp_path):
    log = tmp_path / "suite.log"
    lines = [
        f"E   open_stage=store_open open_error=StoreLockError: attempt {i}"
        for i in range(20)
    ]
    lines.append(
        "E   durable store did not become active: stage='identity_build' "
        "open_error='FileNotFoundError: frame_0009.fits'"
    )
    log.write_text("\n".join(lines), encoding="utf-8")
    hits = bench.collect_store_diagnostics([_result(log)], limit=15)
    assert len(hits) == 15
    assert any("identity_build" in hit for hit in hits), hits


def test_m2_bench_skips_passing_suites(tmp_path):
    log = tmp_path / "suite.log"
    log.write_text("E   open_error=should not appear\n", encoding="utf-8")
    result = _result(log)
    result["failed"] = 0
    result["returncode"] = 0
    assert bench.collect_store_diagnostics([result]) == []
