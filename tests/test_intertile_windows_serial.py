"""Windows Phase 5 inter-tile serialization: policy + execution-path tests.

Covers the conservative beta fix that forces single-worker sequential execution
for intertile overlap pairs on Windows (reason token ``windows_reproject_wcs_serial``),
while leaving Linux/macOS parallelism unchanged.

These tests exercise ``zemosaic.zemosaic_utils.compute_intertile_workers_limit``
(pure policy) and ``compute_intertile_affine_calibration`` (execution path) through
hermetic seams: the native ``reproject_interp`` and overlap detection are replaced
with fakes, so no real reprojection or WCS footprint math runs.
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

zu = pytest.importorskip("zemosaic.zemosaic_utils", reason="zemosaic_utils unavailable")

SERIAL_TOKEN = "windows_reproject_wcs_serial"


# ---------------------------------------------------------------------------
# A. Pure policy matrix: compute_intertile_workers_limit
# ---------------------------------------------------------------------------

_WINDOWS_MATRIX = list(
    itertools.product(
        [None, 2, 6, 14, 64],  # requested_workers (None == auto)
        [4, 279, 5000],        # pairs_count (small / 279 / large)
        [128, 512, 1024],      # preview_size
        [2000, 64000],         # available_mb (low / high)
    )
)


@pytest.mark.parametrize("requested,pairs,preview,ram", _WINDOWS_MATRIX)
def test_windows_workers_limit_always_serial(requested, pairs, preview, ram):
    effective, reasons = zu.compute_intertile_workers_limit(
        requested,
        pairs,
        preview,
        cpu_total=16,
        platform_system="Windows",
        available_mb=ram,
    )
    assert effective == 1
    assert SERIAL_TOKEN in reasons


def test_linux_workers_limit_auto_preserved():
    effective, reasons = zu.compute_intertile_workers_limit(
        None, 279, 512, cpu_total=16, platform_system="Linux", available_mb=32000
    )
    assert effective == 8
    assert "auto_base" in reasons
    assert SERIAL_TOKEN not in reasons


def test_linux_workers_limit_requested_preserved():
    effective, reasons = zu.compute_intertile_workers_limit(
        14, 279, 512, cpu_total=16, platform_system="Linux", available_mb=32000
    )
    assert effective == 8
    assert SERIAL_TOKEN not in reasons


def test_darwin_workers_limit_preserved():
    effective, reasons = zu.compute_intertile_workers_limit(
        6, 279, 512, cpu_total=16, platform_system="Darwin", available_mb=32000
    )
    assert effective == 6
    assert SERIAL_TOKEN not in reasons


# ---------------------------------------------------------------------------
# B + C hermetic execution-path harness
# ---------------------------------------------------------------------------

_EXPECTED_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


class _RecordingLogger:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def info(self, m):
        self.messages.append(("INFO", m))

    def warning(self, m):
        self.messages.append(("WARN", m))

    def error(self, m):
        self.messages.append(("ERROR", m))

    def debug(self, m):
        self.messages.append(("DEBUG", m))


def _make_fake_reproject(concurrency, seen):
    def fake_reproject(input_data, output_projection, shape_out=None, **kwargs):
        concurrency["active"] += 1
        concurrency["max"] = max(concurrency["max"], concurrency["active"])
        try:
            time.sleep(0.02)
            data = input_data[0]
            tile_id = int(round(float(np.asarray(data).reshape(-1)[0] - 1000.0)))
            seen.append(tile_id)
            h, w = shape_out
            arr = (np.arange(h * w, dtype=np.float32).reshape(h, w) / float(h * w)) + tile_id * 100.0
            return arr, arr.astype(np.float32)
        finally:
            concurrency["active"] -= 1

    return fake_reproject


def _fake_estimate_overlap_pairs(
    wcs_list, shapes_hw, final_output_wcs, final_output_shape_hw, min_overlap_fraction=0.05
):
    return [{"i": i, "j": j, "bbox": (0, 16, 0, 16), "weight": 1.0} for (i, j) in _EXPECTED_PAIRS]


class _ForbiddenPool:
    """ThreadPoolExecutor stand-in that fails the test if constructed."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("ThreadPoolExecutor must not be constructed on the Windows intertile path")


def _make_tiles():
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    return [(np.full((32, 32), 1000.0 + i, dtype=np.float32), wcs) for i in range(4)], wcs


def _run_calibration(monkeypatch, simulate_windows, forbidden_pool):
    from astropy.wcs import WCS

    concurrency = {"active": 0, "max": 0}
    seen = []
    monkeypatch.setattr(zu, "reproject_interp", _make_fake_reproject(concurrency, seen))
    monkeypatch.setattr(zu, "estimate_overlap_pairs", _fake_estimate_overlap_pairs)
    if forbidden_pool:
        monkeypatch.setattr(zu, "ThreadPoolExecutor", _ForbiddenPool)
    if simulate_windows:
        monkeypatch.setattr(zu.sys, "platform", "win32")
        monkeypatch.setattr(zu.platform, "system", lambda: "Windows")

    tiles, wcs = _make_tiles()
    logger = _RecordingLogger()
    progress = []

    result = zu.compute_intertile_affine_calibration(
        tiles,
        wcs,
        (32, 32),
        preview_size=512,
        cpu_workers=4,
        logger=logger,
        progress_callback=lambda *a: progress.append(a),
    )

    seen_pairs = sorted(
        frozenset((seen[2 * k], seen[2 * k + 1])) for k in range(len(seen) // 2)
    )
    return {
        "result": result,
        "concurrency": concurrency,
        "seen": seen,
        "seen_pairs": seen_pairs,
        "logger": logger,
        "progress": progress,
    }


def test_windows_execution_path_never_builds_pool_and_is_serial(monkeypatch):
    out = _run_calibration(monkeypatch, simulate_windows=True, forbidden_pool=True)

    # No exception means _ForbiddenPool.__init__ was never reached.
    assert set(out["result"].keys()) == {0, 1, 2, 3}

    # Concurrency guard: reproject never overlaps under Windows.
    assert out["concurrency"]["max"] == 1

    # All expected scientific pairs processed exactly once, no drop/dup beyond
    # the legacy duplicate pair-entry semantics (12 reproject calls = 6 pairs x2).
    expected = sorted(frozenset(p) for p in _EXPECTED_PAIRS)
    assert out["seen_pairs"] == expected
    assert len(out["seen"]) == 12
    assert zu.LAST_INTERTILE_DIAGNOSTICS["raw_pairs_count"] == 6
    assert zu.LAST_INTERTILE_DIAGNOSTICS["pruned_pairs_count"] == 6
    assert zu.LAST_INTERTILE_DIAGNOSTICS["pair_entries_count"] == 12

    # Progress reached total.
    assert any(
        a[0] == "phase5_intertile_pairs" and a[1] == 6 and a[2] == 6 for a in out["progress"]
    )

    # Logs contain the serialization reason + sequential execution mode.
    log_text = "\n".join(m for _, m in out["logger"].messages)
    assert SERIAL_TOKEN in log_text
    assert "sequential" in log_text.lower()


def test_windows_defense_in_depth_seam_mismatch_forces_serial(monkeypatch):
    """If platform.system() disagrees with sys.platform, the execution-function
    guard (sys.platform == 'win32') must still force serialization."""
    from astropy.wcs import WCS

    concurrency = {"active": 0, "max": 0}
    seen = []
    monkeypatch.setattr(zu, "reproject_interp", _make_fake_reproject(concurrency, seen))
    monkeypatch.setattr(zu, "estimate_overlap_pairs", _fake_estimate_overlap_pairs)
    monkeypatch.setattr(zu, "ThreadPoolExecutor", _ForbiddenPool)
    # Helper will NOT see Windows (platform.system -> Linux), but sys.platform
    # says win32: the guard must still close the seam.
    monkeypatch.setattr(zu.sys, "platform", "win32")
    monkeypatch.setattr(zu.platform, "system", lambda: "Linux")

    tiles, wcs = _make_tiles()
    logger = _RecordingLogger()
    result = zu.compute_intertile_affine_calibration(
        tiles, wcs, (32, 32), preview_size=512, cpu_workers=4, logger=logger
    )

    assert set(result.keys()) == {0, 1, 2, 3}
    assert concurrency["max"] == 1
    assert len(seen) == 12
    assert SERIAL_TOKEN in "\n".join(m for _, m in logger.messages)


def test_linux_execution_path_still_constructs_pool(monkeypatch):
    """Non-Windows regression: the parallel ThreadPool path must remain intact."""
    from concurrent.futures import ThreadPoolExecutor as _RealTPE

    constructed: list[int | None] = []

    class _RecordingPool(_RealTPE):
        def __init__(self, max_workers=None, *args, **kwargs):
            constructed.append(max_workers)
            super().__init__(max_workers, *args, **kwargs)

    concurrency = {"active": 0, "max": 0}
    seen = []
    monkeypatch.setattr(zu, "reproject_interp", _make_fake_reproject(concurrency, seen))
    monkeypatch.setattr(zu, "estimate_overlap_pairs", _fake_estimate_overlap_pairs)
    monkeypatch.setattr(zu, "ThreadPoolExecutor", _RecordingPool)

    tiles, wcs = _make_tiles()
    result = zu.compute_intertile_affine_calibration(
        tiles, wcs, (32, 32), preview_size=512, cpu_workers=4
    )

    assert set(result.keys()) == {0, 1, 2, 3}
    # A real ThreadPoolExecutor with >1 workers was constructed and used.
    assert len(constructed) == 1
    assert constructed[0] is not None and constructed[0] > 1
    # All pairs still processed (parallelism changed nothing about pair set).
    assert len(seen) == 12
    assert zu.LAST_INTERTILE_DIAGNOSTICS["pair_entries_count"] == 12
