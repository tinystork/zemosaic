"""ZM-RESUME-TELEMETRY-TRUTH-20261110 — Resume telemetry attribution.

Observability-only regressions proving that ZeGrid Resume telemetry and logs
distinguish work executed in the CURRENT attempt from historical work restored
from durable commits:

* a COMPLETE non-finalized replay (GPU or CPU frozen) executes zero stack/cache
  work, so it must not open a ``cache_build`` phase, must not claim newly
  completed per-cell stacking, and must report zero current cache/stack/
  reprojection totals while the restored durable evidence stays visible under
  explicit additive ``*_historical_restored`` fields;
* a PARTIAL resume attributes only the sparse pending Cells to the current
  attempt and keeps the reused durable Cells separate;
* a FRESH run keeps its established current-attempt telemetry and progress.

The runs drive the REAL product entry (``run_zegrid_mode``) over the shared
synthetic tiny corpus and a REAL ``RunCellStore``; the GPU kernel is simulated
with the real CPU math under the GPU config (same technique as
``test_zegrid_resume_complete_gpu_rerun``). No science value is asserted here:
these are telemetry/attribution contracts only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import test_zegrid_resume_complete_gpu_rerun as RERUN  # noqa: E402


_CACHE_START = "phase START: cache_build"
_CACHE_PROGRESS = "phase PROGRESS: cache_build"
_PROGRESS_RE = re.compile(
    r"phase PROGRESS: (cache_build|per_cell_stack)\s+(\d+)/(\d+)"
)


def _progress_samples(log_text: str, phase: str) -> list[tuple[int, int]]:
    return [
        (int(done), int(total))
        for name, done, total in _PROGRESS_RE.findall(log_text)
        if name == phase
    ]


class _TelemetryRecorder:
    """Record every resource-telemetry sample the product publishes."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def maybe_emit_stats(self, ctx) -> None:
        self.calls.append(dict(ctx))


def _run_with_telemetry(tmp_path, monkeypatch, *, use_gpu, resume_mode="never",
                        telemetry=None, progress=None):
    """Same product run as ``RERUN._run``, with an optional telemetry sink."""
    if use_gpu:
        RERUN.R55._fake_gpu_available(monkeypatch)
    corpus = RERUN._corpus(tmp_path)
    out = tmp_path / "out"
    store = out / "zegrid-durable"
    RERUN.zz.run_zegrid_mode(
        str(corpus), str(out), progress,
        zconfig=RERUN._zconfig(store, resume_mode),
        zegrid_halo_px=8,
        use_gpu=bool(use_gpu),
        telemetry=telemetry,
    )
    return out, store


def _phase3(recorder: _TelemetryRecorder) -> list[dict]:
    return [c for c in recorder.calls if int(c.get("phase_index", -1)) == 3]


# ---------------------------------------------------------------------------
# 1 — COMPLETE GPU-frozen replay: zero current telemetry, no cache rebuild
# ---------------------------------------------------------------------------


def test_complete_gpu_replay_reports_zero_current_work(tmp_path, monkeypatch):
    lines: list[str] = []
    out, store = RERUN._run(
        tmp_path, monkeypatch, use_gpu=True,
        progress=lambda msg, *a, **k: lines.append(str(msg)),
    )
    total = len(RERUN._commits(store))
    assert total == 4
    fresh_log = RERUN._log(out)
    fresh_frames = _progress_samples(fresh_log, "cache_build")
    assert fresh_frames, fresh_log[-2000:]
    fresh_total_frames = max(total_frames for _done, total_frames in fresh_frames)
    assert fresh_total_frames > 0

    RERUN._break_finalization(out, store)
    resumed_lines: list[str] = []
    recorder = _TelemetryRecorder()
    _out, _store = _run_with_telemetry(
        tmp_path, monkeypatch, use_gpu=True, resume_mode="auto",
        telemetry=recorder,
        progress=lambda msg, *a, **k: resumed_lines.append(str(msg)),
    )
    text = "\n".join(resumed_lines)
    assert "executed_this_attempt=0" in text
    assert "durable replay assembly" in text
    # The run-level resume coverage stays explicit in the plan line.
    assert "resume plan" in text

    log = RERUN._log(out)
    # NO cache rebuild progress/start may be claimed by a replay.
    assert _CACHE_START not in log, log[-3000:]
    assert _CACHE_PROGRESS not in log, log[-3000:]
    assert _progress_samples(log, "cache_build") == []
    # ... and no canonical stacking phase either (REWORK-1 F4).
    assert _progress_samples(log, "per_cell_stack") == []
    assert "phase START: per_cell_stack" not in log
    assert "phase END: per_cell_stack" not in log
    assert "per-cell stacking complete" not in log

    manifest = RERUN._published_manifest(out)
    diag = manifest["per_cell_diagnostics"]
    # Current-attempt totals are zero: nothing was stacked/cached/reprojected.
    assert diag["executed_this_attempt"] == 0
    assert diag["cache_build_cpu_sum_s"] == 0.0
    assert diag["stack_cpu_sum_s"] == 0.0
    assert diag["reprojection"]["path"] == "unused"
    assert diag["reprojection"]["fast_calls"] == 0
    assert diag["reprojection"]["fallback_calls"] == 0
    # Attribution is explicit and by actual cell identity.
    attribution = diag["work_attribution"]
    assert attribution["current_cells"] == 0
    assert attribution["historical_restored_cells"] == total
    assert attribution["historical_source"] == "durable_commits"
    # Restored durable evidence stays visible, additively and historically.
    assert diag["stack_historical_restored_sum_s"] > 0
    assert diag["cache_build_historical_restored_sum_s"] >= 0
    assert "historical_restored" in diag["reprojection_historical_restored"][
        "attribution"
    ]
    # The GPU truth of THIS attempt is untouched and the historical backend
    # evidence of the restored commits is additive, not redefined.
    gpu = manifest["gpu"]
    assert gpu["used"] is False
    assert gpu["actually_used"] is False
    assert gpu["gpu_executed_cells"] == 0
    assert gpu["executed_this_attempt"] == 0
    historical = gpu["durable_historical_execution"]
    assert historical["attribution"] == "historical_restored"
    assert historical["source"] == "durable_commits"
    assert historical["cells"] == total
    assert historical["backend"] == "gpu"
    assert historical["backend_counts"]["gpu"] == total
    # Manifest cache figures are the CURRENT attempt's (no rebuilt cache).
    cache = manifest["cache"]
    assert cache["total_bytes"] == 0
    assert cache["peak_cell_bytes"] == 0
    assert cache["n_frame_entries"] == 0
    assert cache["rebuilt_cells"] == []
    cache_attr = cache["attribution"]
    assert cache_attr["current_attempt"]["cells_count"] == 0
    assert cache_attr["historical_restored"]["cells_count"] == total
    assert cache_attr["historical_restored"]["source"] == "durable_commits"

    # REWORK-1 F2: no cell-stacking time or per-unit stack timing is claimed for
    # a zero-stack replay; the real replay wall time is explicit instead.
    assert diag["seconds"] == 0.0
    assert diag["seconds_per_unit"] is None
    assert diag["durable_replay_seconds"] >= 0.0
    assert manifest["timings"]["per_cell_stack"] == 0.0
    assert manifest["timings"]["durable_replay"] >= 0.0
    # REWORK-2 L3: ONE explicit assertion locks the complete-replay triplet
    # together: zero current cache work, zero current stack work, and a real
    # (strictly positive) durable-replay wall time.
    assert (
        diag["cache_build_cpu_sum_s"],
        diag["stack_cpu_sum_s"],
        diag["durable_replay_seconds"] > 0.0,
    ) == (0.0, 0.0, True)

    # REWORK-1 F1: the current aggregate RSS excludes every restored worker
    # peak; the historical worker evidence is additive and separate.
    restored_aggregate = diag[
        "aggregate_peak_rss_kib_historical_restored"
    ]
    assert restored_aggregate is not None and restored_aggregate > 0
    assert manifest[
        "aggregate_peak_rss_kib_historical_restored"
    ] == restored_aggregate
    assert manifest["aggregate_peak_rss_kib"] < restored_aggregate

    # REWORK-1 F3: the resource sample describes zero current stack units and
    # labels itself as durable replay, carrying the restored count separately.
    samples3 = _phase3(recorder)
    assert samples3, recorder.calls
    final3 = samples3[-1]
    assert final3["phase_name"] == "Durable replay"
    assert final3["tiles_done"] == 0
    assert final3["tiles_total"] == 0
    assert final3["executed_this_attempt"] == 0
    assert final3["restored_durable_cells"] == total


# ---------------------------------------------------------------------------
# 2 — COMPLETE CPU-frozen replay: same truth, CPU backend evidence
# ---------------------------------------------------------------------------


def test_complete_cpu_replay_reports_zero_current_work(tmp_path, monkeypatch):
    out, store = RERUN._run(tmp_path, monkeypatch, use_gpu=False)
    total = len(RERUN._commits(store))
    assert total == 4
    RERUN._break_finalization(out, store)
    resumed_lines: list[str] = []
    recorder = _TelemetryRecorder()
    _out, _store = _run_with_telemetry(
        tmp_path, monkeypatch, use_gpu=False, resume_mode="auto",
        telemetry=recorder,
        progress=lambda msg, *a, **k: resumed_lines.append(str(msg)),
    )
    text = "\n".join(resumed_lines)
    assert "durable replay assembly" in text
    assert "per-cell stacking complete" not in text

    log = RERUN._log(out)
    assert _CACHE_START not in log
    assert _CACHE_PROGRESS not in log
    assert "phase START: per_cell_stack" not in log
    assert "phase END: per_cell_stack" not in log

    manifest = RERUN._published_manifest(out)
    diag = manifest["per_cell_diagnostics"]
    assert diag["executed_this_attempt"] == 0
    assert diag["cache_build_cpu_sum_s"] == 0.0
    assert diag["stack_cpu_sum_s"] == 0.0
    assert diag["stack_historical_restored_sum_s"] > 0
    gpu = manifest["gpu"]
    assert gpu["actually_used"] is False
    assert gpu["executed_this_attempt"] == 0
    historical = gpu["durable_historical_execution"]
    assert historical["backend"] == "cpu"
    assert historical["backend_counts"]["cpu"] == total
    assert historical["backend_counts"]["gpu"] == 0
    assert manifest["cache"]["rebuilt_cells"] == []
    assert manifest["cache"]["attribution"]["historical_restored"][
        "cells_count"
    ] == total
    # REWORK-1 F1/F2/F3 on the CPU replay.
    assert diag["seconds"] == 0.0
    assert diag["seconds_per_unit"] is None
    assert diag["durable_replay_seconds"] >= 0.0
    assert manifest["timings"]["per_cell_stack"] == 0.0
    assert manifest["timings"]["durable_replay"] >= 0.0
    # REWORK-2 L3: the same zero-cache / zero-stack / positive-replay triplet.
    assert (
        diag["cache_build_cpu_sum_s"],
        diag["stack_cpu_sum_s"],
        diag["durable_replay_seconds"] > 0.0,
    ) == (0.0, 0.0, True)
    assert diag["aggregate_peak_rss_kib_historical_restored"] > 0
    assert (
        manifest["aggregate_peak_rss_kib"]
        < manifest["aggregate_peak_rss_kib_historical_restored"]
    )
    final3 = _phase3(recorder)[-1]
    assert final3["phase_name"] == "Durable replay"
    assert (final3["tiles_done"], final3["tiles_total"]) == (0, 0)
    assert final3["restored_durable_cells"] == total


# ---------------------------------------------------------------------------
# 3 — PARTIAL resume: only pending Cells are current
# ---------------------------------------------------------------------------


def test_partial_resume_splits_pending_and_restored(tmp_path, monkeypatch):
    out, store = RERUN._run(tmp_path, monkeypatch, use_gpu=False)
    commits = RERUN._commits(store)
    assert len(commits) == 4
    removed_ids = sorted(RERUN._commit_cid(path) for path in commits[2:])
    expected_pending_frames = 0
    for path in commits[2:]:
        payload_meta = json.loads(path.read_text(encoding="utf-8"))[
            "payload_meta"
        ]
        expected_pending_frames += int(payload_meta.get("n_frames") or 0)
    assert expected_pending_frames > 0
    for path in commits[2:]:
        path.unlink()

    RERUN._run(
        tmp_path, monkeypatch, use_gpu=False, resume_mode="auto",
    )
    log = RERUN._log(out)
    samples = _progress_samples(log, "cache_build")
    assert samples, log[-3000:]
    partial_total_frames = samples[0][1]
    # Only the pending Cells' frames are counted: never the whole run.
    assert partial_total_frames == expected_pending_frames
    assert samples[-1][0] == partial_total_frames
    # REWORK-1 F4: a PARTIAL resume still opens the canonical stacking phase
    # with the RUN-level coverage (monotonic, no overrun).
    stack_samples = _progress_samples(log, "per_cell_stack")
    assert stack_samples and stack_samples[-1] == (4, 4)
    assert 5 not in [done for done, _total in stack_samples]
    assert "phase START: per_cell_stack" in log
    assert "phase END: per_cell_stack" in log

    manifest = RERUN._published_manifest(out)
    diag = manifest["per_cell_diagnostics"]
    assert diag["executed_this_attempt"] == 2
    assert diag["reused_durable_cells"] == 2
    attribution = diag["work_attribution"]
    assert attribution["current_cells"] == 2
    assert attribution["historical_restored_cells"] == 2
    # Both sides are measured; neither is folded into the other.
    assert diag["stack_cpu_sum_s"] > 0
    assert diag["stack_historical_restored_sum_s"] > 0
    assert "historical_restored" in diag["reprojection_historical_restored"][
        "attribution"
    ]
    cache = manifest["cache"]
    assert sorted(cache["rebuilt_cells"]) == removed_ids
    restored = cache["attribution"]["historical_restored"]
    assert restored["cells_count"] == 2
    assert restored["source"] == "durable_commits"
    # No complete-replay claim on a partial resume.
    assert "durable_replay" not in diag
    assert diag["seconds"] > 0.0
    assert diag["seconds_per_unit"] is not None
    assert "durable_replay_seconds" not in diag
    assert manifest["timings"]["per_cell_stack"] > 0.0
    assert "durable_replay" not in manifest["timings"]
    # Current aggregate RSS covers the parent + pending workers only, while the
    # restored workers' peaks are additive historical evidence.
    assert diag["aggregate_peak_rss_kib_historical_restored"] > 0
    assert manifest["aggregate_peak_rss_kib_historical_restored"] == diag[
        "aggregate_peak_rss_kib_historical_restored"
    ]
    assert manifest["gpu"]["executed_this_attempt"] == 2
    historical = manifest["gpu"]["durable_historical_execution"]
    assert historical["cells"] == 2
    assert historical["backend"] == "cpu"


# ---------------------------------------------------------------------------
# 4 — FRESH run: established current-attempt telemetry stays compatible
# ---------------------------------------------------------------------------


def test_fresh_run_keeps_current_attribution_and_cache_progress(
    tmp_path, monkeypatch
):
    lines: list[str] = []
    recorder = _TelemetryRecorder()
    out, store = _run_with_telemetry(
        tmp_path, monkeypatch, use_gpu=True, telemetry=recorder,
        progress=lambda msg, *a, **k: lines.append(str(msg)),
    )
    text = "\n".join(lines)
    total = len(RERUN._commits(store))
    log = RERUN._log(out)
    # The cache diagnostic phase is still opened and still counts frames.
    assert _CACHE_START in log
    samples = _progress_samples(log, "cache_build")
    assert samples and samples[-1][0] == samples[-1][1]
    assert "per-cell stacking complete" in text
    # The canonical stacking phase keeps its normal START/PROGRESS/END.
    stack_samples = _progress_samples(log, "per_cell_stack")
    assert stack_samples and stack_samples[-1] == (total, total)
    assert "phase START: per_cell_stack" in log
    assert "phase END: per_cell_stack" in log

    manifest = RERUN._published_manifest(out)
    diag = manifest["per_cell_diagnostics"]
    assert diag["executed_this_attempt"] == total
    assert diag["reused_durable_cells"] == 0
    attribution = diag["work_attribution"]
    assert attribution["current_cells"] == total
    assert attribution["historical_restored_cells"] == 0
    assert attribution["historical_source"] is None
    assert diag["stack_cpu_sum_s"] > 0
    assert diag["cache_build_historical_restored_sum_s"] == 0.0
    assert diag["stack_historical_restored_sum_s"] == 0.0
    assert "reprojection_historical_restored" not in diag
    cache = manifest["cache"]
    assert len(cache["rebuilt_cells"]) == total
    assert cache["total_bytes"] > 0
    assert cache["attribution"]["historical_restored"]["cells"] == []
    gpu = manifest["gpu"]
    assert gpu["actually_used"] is True
    assert gpu["executed_this_attempt"] == total
    assert gpu["durable_historical_execution"] is None
    # REWORK-1: fresh current timing/telemetry/RSS reporting is unchanged.
    assert diag["seconds"] > 0.0
    assert diag["seconds_per_unit"] is not None
    assert "durable_replay_seconds" not in diag
    assert manifest["timings"]["per_cell_stack"] > 0.0
    assert "durable_replay" not in manifest["timings"]
    assert diag["aggregate_peak_rss_kib_historical_restored"] is None
    assert manifest["aggregate_peak_rss_kib_historical_restored"] is None
    final3 = _phase3(recorder)[-1]
    assert final3["phase_name"] == "Per-cell stacking"
    assert (final3["tiles_done"], final3["tiles_total"]) == (total, total)
    assert final3["historical_restored_cells"] == 0
