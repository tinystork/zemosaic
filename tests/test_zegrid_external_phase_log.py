"""ZM-ZEGRID-EXTERNAL-REDUCE-HANG — per-cell phase breadcrumbs inside the worker.

A deep external cell that stops committing leaves NO evidence today. The worker
now writes write/ reduce/ finalize START/END lines into its own spool attempt
directory (`phases.log`), which survives a hang, and the parent echoes the
measured phase elapsed_s into the run log BEFORE the durable commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.canonical_engine import CanonicalStackRequest  # noqa: E402
from zemosaic.core.zegrid import external_stream as xs  # noqa: E402
from zemosaic.core.zegrid import file_provider as fp  # noqa: E402

N, H, W, C = 3, 16, 16, 3
_BASE = np.random.default_rng(5).normal(100.0, 2.0, (N, H, W, C)).astype(np.float32)


def _prepare(i):
    return np.ascontiguousarray(_BASE[int(i)]), np.ones((H, W), dtype=bool)


def _prepare_window(i, y0, y1, x0, x1):
    h, w = int(y1 - y0), int(x1 - x0)
    src = _BASE[int(i)]
    return (
        np.ascontiguousarray(src[np.arange(h) % H][:, np.arange(w) % W]),
        np.ones((h, w), dtype=bool),
    )


def _provider():
    return fp.WindowedCanonicalProvider(
        _prepare, N, expected_hw=(H, W),
        prepare_window=_prepare_window, expected_channels=C,
    )


def _request():
    return CanonicalStackRequest(
        images=[None] * N, geometric_support=[None] * N,
        normalization="sky_mean", weighting="noise_variance",
        rejection="winsorized_sigma_clip", combine="mean",
        taper="footprint", taper_px=4.0,
    )


def _run(tmp_path, **kw):
    return xs.run_cell_streaming_external(
        _provider(), _request(),
        tile_size=8, column_batch=48,
        spool_root=str(tmp_path), cell_id="c1", attempt_id="a1",
        keep_spool=True, **kw,
    )


def _phase_lines(tmp_path):
    files = list(Path(tmp_path).rglob(xs.PHASE_LOG_NAME))
    assert files, "no phases.log written"
    return files[0].read_text(encoding="utf-8").splitlines()


def test_external_cell_writes_all_phase_boundaries(tmp_path):
    _run(tmp_path)
    lines = _phase_lines(tmp_path)

    assert "ZeGrid: cell c1 phase=write START" in lines
    assert "ZeGrid: cell c1 phase=reduce START" in lines
    assert "ZeGrid: cell c1 phase=finalize START" in lines
    for label in ("write", "reduce", "finalize"):
        ends = [ln for ln in lines if f"phase={label} END (" in ln]
        assert len(ends) == 1, (label, lines)
        # the duration is a float with three decimals
        assert ends[0].endswith("s)") and float(
            ends[0].split("(")[1].rstrip("s)")
        ) >= 0.0
    # the write phase starts BEFORE the first reduce, finalize closes last
    assert lines.index("ZeGrid: cell c1 phase=write START") < lines.index(
        "ZeGrid: cell c1 phase=reduce START"
    )
    assert lines[-1].startswith("ZeGrid: cell c1 phase=finalize END")


def test_a_failing_phase_leaves_its_start_without_an_end(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("reduce died")

    monkeypatch.setattr(xs, "reduce_spooled_tile", _boom)
    with pytest.raises(RuntimeError):
        _run(tmp_path, retain_on_failure=True)

    lines = _phase_lines(tmp_path)
    # The last START with no END localises WHERE the cell died.
    assert "ZeGrid: cell c1 phase=write START" in lines
    assert "ZeGrid: cell c1 phase=reduce START" in lines
    assert not [ln for ln in lines if "phase=reduce END (" in ln]
    # the tile loop never completed, so write has no END either
    assert not [ln for ln in lines if "phase=write END (" in ln]
    assert "ZeGrid: cell c1 phase=finalize START" not in lines
