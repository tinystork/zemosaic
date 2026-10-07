"""ZM-ZEGRID-R20 targeted tests — concurrent per-cell STACKING (fast tier).

Covers (no gated corpora; the routine deterministic corpus is generated on disk):

* the concurrency decision function ``cells_in_flight`` is deterministic and never
  exceeds the RAM/cpu cap (never > cpu-2, never > RAM//footprint, clamped [1, 14]);
* the full production pipeline is BIT-EQUAL between the serial path
  (``cells_in_flight=1``) and the concurrent path (``cells_in_flight>1``) —
  science FITS sha256 identical;
* a forced process-pool failure WARNs loudly and the serial fallback completes
  with the SAME result (never crash, never silent);
* the R20 diagnostics (effective executor, parent daemon flag, workers /
  cells-in-flight, seconds/unit) appear in the manifest and the run log.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import parallel as zpar  # noqa: E402

TOOL = REPO_ROOT / "tools" / "zegrid_routine" / "make_routine_corpus.py"


# ---------------------------------------------------------------------------
# Decision function (deterministic unit check)
# ---------------------------------------------------------------------------

def test_cells_in_flight_decision_function():
    gb = 2**30
    # CPU-bound: min(cpu-2, RAM//footprint) -> cpu-2.
    assert zpar.cells_in_flight(16, 13 * gb, gb) == 13
    assert zpar.cells_in_flight(8, 100 * gb, gb) == 6
    # RAM-bound.
    assert zpar.cells_in_flight(16, 3 * gb, gb) == 3
    assert zpar.cells_in_flight(16, 2 * gb, gb) == 2
    # RAM tighter than one cell -> degrade to 1 (never exceed the budget).
    assert zpar.cells_in_flight(16, gb // 2, gb) == 1
    # Cap at 14 (mirrors the worker max).
    assert zpar.cells_in_flight(64, 1000 * gb, gb) == 14
    # Small CPU floors to 1 (cpu-2 <= 0 -> max(1, ...)).
    assert zpar.cells_in_flight(2, 100 * gb, gb) == 1
    assert zpar.cells_in_flight(1, 100 * gb, gb) == 1
    # No memory bound -> cpu-2 only.
    assert zpar.cells_in_flight(16, None, gb) == 14


def test_cells_in_flight_never_exceeds_bounds():
    """Property: result is always <= cpu-2 and <= RAM//footprint, and >= 1."""
    gb = 2**30
    for cpu in (1, 2, 4, 8, 16, 64):
        for avail in (0, gb // 2, gb, 3 * gb, 10 * gb, 100 * gb):
            for fp in (gb // 4, gb, 2 * gb):
                n = zpar.cells_in_flight(cpu, avail, fp)
                assert n >= 1
                assert n <= max(1, cpu - 2)
                if avail > 0:
                    assert n <= max(1, avail // fp)
                assert n <= 14


# ---------------------------------------------------------------------------
# Full-pipeline bit-equality (serial vs concurrent)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def routine_corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("r20_routine_corpus")
    proc = subprocess.run(
        [sys.executable, str(TOOL), str(out)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return out


def _prepare_input(input_dir, routine_corpus):
    input_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(routine_corpus.glob("*.fits")):
        (input_dir / p.name).write_bytes(p.read_bytes())
    (input_dir / "stack_plan.csv").write_text(
        (routine_corpus / "stack_plan.csv").read_text(), encoding="utf-8"
    )


def _run_pipeline(input_dir, out_dir):
    zz.run_zegrid_mode(str(input_dir), str(out_dir))
    return out_dir


def _science_sha256(out_dir):
    with fits.open(out_dir / "mosaic_grid.fits") as hdul:
        data = np.ascontiguousarray(np.asarray(hdul[0].data, dtype=np.float32))
    return hashlib.sha256(data.tobytes()).hexdigest()


def test_full_pipeline_concurrent_bit_equal_serial(tmp_path, routine_corpus, monkeypatch):
    # Serial path (cells_in_flight=1).
    s_in, s_out = tmp_path / "in_s", tmp_path / "out_s"
    _prepare_input(s_in, routine_corpus)
    monkeypatch.setattr(zpar, "cells_in_flight", lambda *a, **k: 1)
    _run_pipeline(s_in, s_out)
    hash_serial = _science_sha256(s_out)

    # Concurrent path (cells_in_flight=3, forces a real process pool when >=2 cells).
    c_in, c_out = tmp_path / "in_c", tmp_path / "out_c"
    _prepare_input(c_in, routine_corpus)
    monkeypatch.setattr(zpar, "cells_in_flight", lambda *a, **k: 3)
    _run_pipeline(c_in, c_out)
    hash_concurrent = _science_sha256(c_out)

    assert hash_serial == hash_concurrent


# ---------------------------------------------------------------------------
# Forced failure -> loud WARN + serial fallback (same result)
# ---------------------------------------------------------------------------

class _BoomExecutor:
    def __init__(self, *a, **k):
        raise OSError("simulated process-pool spawn failure")


def test_per_cell_forced_failure_falls_back_serial(tmp_path, routine_corpus, monkeypatch, caplog):
    # Serial reference.
    s_in, s_out = tmp_path / "in_s", tmp_path / "out_s"
    _prepare_input(s_in, routine_corpus)
    monkeypatch.setattr(zpar, "cells_in_flight", lambda *a, **k: 1)
    _run_pipeline(s_in, s_out)
    hash_serial = _science_sha256(s_out)

    # Concurrent path with a forced process-pool failure -> serial fallback.
    fb_in, fb_out = tmp_path / "in_fb", tmp_path / "out_fb"
    _prepare_input(fb_in, routine_corpus)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _BoomExecutor)
    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: False)
    monkeypatch.setattr(zpar, "cells_in_flight", lambda *a, **k: 3)
    with caplog.at_level(logging.WARNING):
        _run_pipeline(fb_in, fb_out)
    hash_fallback = _science_sha256(fb_out)

    assert hash_serial == hash_fallback
    # The failure was surfaced loudly (WARN via the logger / progress path).
    assert ("falling back to SERIAL" in caplog.text) or ("parallel map unavailable" in caplog.text)


# ---------------------------------------------------------------------------
# Diagnostics in the manifest + run log
# ---------------------------------------------------------------------------

def test_per_cell_diagnostics_in_manifest_and_log(tmp_path, routine_corpus, monkeypatch):
    c_in, c_out = tmp_path / "in", tmp_path / "out"
    _prepare_input(c_in, routine_corpus)
    monkeypatch.setattr(zpar, "cells_in_flight", lambda *a, **k: 3)
    _run_pipeline(c_in, c_out)

    manifest = json.loads((c_out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    assert "per_cell_diagnostics" in manifest
    d = manifest["per_cell_diagnostics"]
    assert d["cells_in_flight"] == 3
    assert "parent_daemon" in d
    assert d["executor"] in ("process", "thread", "serial")
    assert d["workers"] >= 1
    assert d["tasks"] >= 1
    assert d["seconds"] >= 0.0
    assert d["seconds_per_unit"] >= 0.0
    assert "per_cell_footprint_bytes" in d
    assert "ram_budget_bytes" in d

    run_log = (c_out / "zegrid_run.log").read_text(encoding="utf-8")
    assert "Per-cell stacking concurrency" in run_log
