"""ZM-ZEGRID-REDUCE-VISIBILITY — per-cell lifecycle lines in the RUN LOG FILE.

The real M33 stack ran ~3 h with ZERO commits and no way to tell whether a cell
was 1 % or 99 % done: the phase lines land in the file, the per-cell chatter did
not exist. These tests pin the exact text and the fact that it reaches the run
log FILE (not only the console), plus the 60 s heartbeat rate limit.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import parallel as zpar  # noqa: E402
from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402


def test_cell_start_line_names_id_frames_and_mode():
    line = zpar.cell_lifecycle_line(
        "START", "c-17", frames=1113, mode="external"
    )
    assert line == "ZeGrid: cell START id=c-17 frames=1113 mode=external"


def test_cell_done_line_names_stack_seconds_and_status():
    line = zpar.cell_lifecycle_line(
        "DONE", "c-17", stack_s=2364.42, status="complete"
    )
    assert line == "ZeGrid: cell DONE id=c-17 stack_s=2364.4 status=complete"


def test_unknown_lifecycle_kind_is_refused():
    import pytest

    with pytest.raises(ValueError):
        zpar.cell_lifecycle_line("PROGRESS", "c")


def test_lifecycle_lines_reach_the_run_log_file(tmp_path):
    # The EXACT writer the run uses; a reader must see the lines during the run.
    zpar_lines = [
        zpar.cell_lifecycle_line("START", "c-1", frames=40, mode="stream"),
        zpar.cell_lifecycle_line("DONE", "c-1", stack_s=85.0, status="complete"),
    ]
    for line in zpar_lines:
        zz._append_run_log_line(str(tmp_path), line)
    log = (tmp_path / zz.RUN_LOG_NAME).read_text(encoding="utf-8")
    assert "ZeGrid: cell START id=c-1 frames=40 mode=stream" in log
    assert "ZeGrid: cell DONE id=c-1 stack_s=85.0 status=complete" in log


def test_heartbeat_line_and_rate_limit():
    assert zpar.cells_heartbeat_line(3, 104, 6) == (
        "ZeGrid: cells committed=3/104 in_flight=6"
    )
    assert zpar.heartbeat_due(100.0, 159.9) is False
    assert zpar.heartbeat_due(100.0, 160.0) is True
    assert zpar.heartbeat_due(0.0, 1e9) is True
