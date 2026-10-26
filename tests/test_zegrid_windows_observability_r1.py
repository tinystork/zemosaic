"""ZM-WIN-OBSERVABILITY-20261026 — Windows-ZeGrid progress/ETA + resource tests.

Focused, CPU-only regressions for the fixes in this mission:

* a FRAME-counted ``zegrid:cache_build`` can never contaminate the canonical
  CELL-counted ``zegrid:per_cell_stack`` totals/ETA;
* the real Qt feed path (not the estimator alone) learns 9 cells and produces a
  sensible ETA from the authentic cache 0/1751 -> stack 0/9 -> 1..9 sequence;
* breadcrumb RAM separates system RAM from process/process-tree RSS and a
  CUDA-free GPU probe is the only GPU source;
* ``read_parent_memory`` has a portable psutil fallback;
* CuPy pool counters are read by CALLING bound methods;
* the ResourceTelemetryController honours ON/OFF and writes the CSV.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from zemosaic import progress_contract as zprogress
from zemosaic import eta_service as zeta
from zemosaic.core import crash_breadcrumbs
from zemosaic.core.zegrid import gpu_memory as zgm
from zemosaic.core.zegrid import runtime_control as zrt
from zemosaic import zemosaic_resource_telemetry as ztelemetry


# ---------------------------------------------------------------------------
# Progress contract: units are never mixed
# ---------------------------------------------------------------------------

def test_cache_build_is_a_diagnostic_not_a_cell_stage():
    plan = zprogress.ZEGRID_PLAN
    assert "zegrid:cache_build" in zprogress.ZEGRID_DIAGNOSTIC_STAGE_IDS
    assert plan.stage("zegrid:cache_build") is None
    assert plan.normalize_id("zegrid:cache_build") == "zegrid:cache_build"

    agg = zprogress.ProgressAggregator(plan)
    before = agg.global_percent
    result = agg.on_stage("zegrid:cache_build", 0, 1751)
    assert result.known is False
    assert agg.global_percent == before  # never mapped to global progress

    cells = agg.on_stage("zegrid:per_cell_stack", 9, 9)
    assert cells.known is True
    assert cells.local_fraction == 1.0
    assert agg.global_percent < 100.0  # no false 100 before finalize/success
    assert agg.mark_success().global_percent == 100.0


def test_failure_and_cancel_never_reach_success():
    for method in ("mark_fail", "mark_cancel"):
        agg = zprogress.ProgressAggregator(zprogress.ZEGRID_PLAN)
        agg.on_stage("zegrid:per_cell_stack", 9, 9)
        getattr(agg, method)()
        assert agg.global_percent < 100.0
        assert agg.mark_success().global_percent < 100.0


# ---------------------------------------------------------------------------
# Real Qt feed path (offscreen, real methods, no full window construction)
# ---------------------------------------------------------------------------

class _Label:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, value: str) -> None:  # noqa: N802 - Qt API
        self.text = str(value)


class _Bar:
    def __init__(self) -> None:
        self.value = 0

    def setValue(self, value: int) -> None:  # noqa: N802 - Qt API
        self.value = int(value)


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


def _qt_feed_harness(clock: _Clock):
    """A real ZeMosaicQtMainWindow object with presentation stubbed only.

    All progress/ETA methods exercised here are the production methods; only
    widgets and disk history are replaced. Qt runs offscreen, and the window
    constructor is skipped (it would build the whole product UI).
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from zemosaic import zemosaic_gui_qt as gui

    _app = QApplication.instance() or QApplication([])

    class _Harness(gui.ZeMosaicQtMainWindow):
        def __init__(self) -> None:  # bypass the full product UI construction
            pass

    obj = _Harness()
    obj._sds_progress_active = False
    obj._weighted_progress_active = False
    obj._zegrid_active = False
    obj._progress_agg = None
    obj._progress_agg_mode = None
    obj._last_global_progress = 0.0
    obj._zegrid_phase_floor = 0
    obj._zegrid_n_frames = None
    obj._zegrid_cell_count = None
    obj._zegrid_stage_totals = {}
    obj._zegrid_eta_records = []
    obj._zegrid_eta = zeta.HybridEtaEstimator(zprogress.ZEGRID_PLAN, clock=clock)
    obj._zegrid_eta.set_priors(
        {
            "zegrid:gauge": 20.0,
            "zegrid:per_cell_stack": 60.0,
            "zegrid:assembly": 5.0,
            "zegrid:finalize": 2.0,
        }
    )
    obj.progress_bar = _Bar()
    obj.tiles_value_label = _Label()
    obj.phase_value_label = _Label()
    obj._eta_display = ""

    obj._tr = lambda key, default=None: default if default is not None else key
    obj._set_eta_display = lambda text, force=False: setattr(
        obj, "_eta_display", str(text)
    )

    def feed(stage: str, current: int, total: int):
        obj._update_stage_progress(stage, current, total)
        return obj._zegrid_eta.diagnostics().get("last_estimate")

    return obj, feed


def test_qt_feed_replay_keeps_cells_and_sensible_eta(monkeypatch):
    # Keep the injected priors stable: this test is about UNIT handling, not
    # history-prior selection (no history is available in the test HOME).
    monkeypatch.setattr(
        zeta,
        "select_zegrid_priors",
        lambda *a, **k: (
            {
                "zegrid:gauge": 20.0,
                "zegrid:per_cell_stack": 60.0,
                "zegrid:assembly": 5.0,
                "zegrid:finalize": 2.0,
            },
            0,
        ),
    )
    clock = _Clock()
    obj, feed = _qt_feed_harness(clock)

    # Authentic observed sequence: FRAME cache counter first, then CELL stack.
    feed("zegrid:cache_build", 0, 1751)
    assert obj.tiles_value_label.text == ""
    assert obj._zegrid_cell_count is None

    feed("zegrid:per_cell_stack", 0, 9)
    assert obj.tiles_value_label.text == "0 / 9"

    last = None
    for i in range(1, 10):
        clock.advance(6.0)
        last = feed("zegrid:per_cell_stack", i, 9)

    assert obj._zegrid_cell_count == 9
    st = obj._zegrid_eta._stages["zegrid:per_cell_stack"]
    assert st.total == 9  # never inflated to 1751
    assert st.last_current == 9

    # A late cache diagnostic must not touch the learned cell total/ETA.
    feed("zegrid:cache_build", 900, 1751)
    assert obj._zegrid_cell_count == 9
    assert obj._zegrid_eta._stages["zegrid:per_cell_stack"].total == 9

    assert last is not None and last["ready"] is True
    assert last["remaining_seconds"] is not None
    # The regression produced multi-hour estimates from 1751 "cells"; the
    # correct plan leaves only assembly+finalize (seconds), never hours.
    assert last["remaining_seconds"] < 120.0
    assert obj._progress_agg.global_percent < 100.0


# ---------------------------------------------------------------------------
# Breadcrumb RAM/GPU portability
# ---------------------------------------------------------------------------

def test_breadcrumb_separates_system_and_process_memory():
    snap = crash_breadcrumbs.build_runtime_snapshot(
        __import__("psutil"), None, False, gpu_probe=None
    )
    # Legacy system-RAM keys remain, plus explicit names.
    for key in ("ram_used_mb", "ram_total_mb", "ram_pct",
                "system_ram_used_mb", "system_ram_total_mb", "system_ram_percent"):
        assert key in snap
    assert snap["process_rss_mb"] is not None
    assert snap["process_tree_rss_sampled_sum_mb"] is not None
    assert snap["process_tree_rss_sampled_sum_label"] == "sampled_sum_not_simultaneous"
    assert snap["process_tree_sample_count"] >= 1
    # No probe -> explicit reason, never a fabricated value or CUDA init.
    assert snap["gpu_total_mb"] is None
    assert snap["gpu_unavailable_reason"] == "gpu_probe_not_enabled"


def test_breadcrumb_gpu_probe_is_best_effort():
    snap = crash_breadcrumbs.build_runtime_snapshot(
        __import__("psutil"),
        None,
        False,
        gpu_probe=lambda: {
            "gpu_used_mb": 12.0,
            "gpu_total_mb": 100.0,
            "gpu_free_mb": 88.0,
            "gpu_source": "nvml",
        },
    )
    assert snap["gpu_used_mb"] == 12.0
    assert snap["gpu_source"] == "nvml"
    assert snap["gpu_unavailable_reason"] is None

    failed = crash_breadcrumbs.build_runtime_snapshot(
        __import__("psutil"),
        None,
        False,
        gpu_probe=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert failed["gpu_total_mb"] is None
    assert "gpu_probe_failed" in failed["gpu_unavailable_reason"]


# ---------------------------------------------------------------------------
# Portable parent memory
# ---------------------------------------------------------------------------

def _fake_process(*, peak=True):
    info = types.SimpleNamespace(rss=123 * 1024 * 1024)
    if peak:
        info.peak_wset = 456 * 1024 * 1024
    return types.SimpleNamespace(memory_info=lambda: info)


def test_read_parent_memory_psutil_fallback_for_windows(tmp_path: Path):
    result = zrt.read_parent_memory(
        None,
        process_factory=lambda: _fake_process(peak=True),
    )
    assert result["source"] == "psutil_parent"
    assert result["current_rss_bytes"] == 123 * 1024 * 1024
    assert result["peak_rss_bytes"] == 456 * 1024 * 1024
    assert result["unavailable_reason"] is None


def test_read_parent_memory_peak_unknown_is_explicit(tmp_path: Path):
    result = zrt.read_parent_memory(
        None,
        process_factory=lambda: _fake_process(peak=False),
    )
    assert result["current_rss_bytes"] == 123 * 1024 * 1024
    assert result["peak_rss_bytes"] is None
    assert result["unavailable_reason"] == "parent_peak_not_exposed_by_platform"


# ---------------------------------------------------------------------------
# CuPy pool counters: call bound methods
# ---------------------------------------------------------------------------

class _CallablePool:
    def used_bytes(self):
        return 11

    def total_bytes(self):
        return 22

    def free_bytes(self):
        return 11


class _PlainPool:
    used_bytes = 1
    total_bytes = 2
    free_bytes = 1


class _RaisingPool:
    def used_bytes(self):
        raise TypeError("real CuPy method must be called")

    def total_bytes(self):
        return 2

    def free_bytes(self):
        return 1


def test_pool_counters_call_bound_methods_and_keep_plain_attributes():
    called, errors = zgm._read_pool(_CallablePool())
    assert errors == []
    assert called["pool_used_bytes"] == 11
    assert called["pool_total_bytes"] == 22
    assert called["pool_free_bytes"] == 11

    plain, errors = zgm._read_pool(_PlainPool())
    assert errors == []
    assert plain["pool_used_bytes"] == 1

    bad, errors = zgm._read_pool(_RaisingPool())
    assert bad["pool_used_bytes"] is None
    assert bad["pool_total_bytes"] == 2
    assert any("pool_used_bytes" in e for e in errors)


# ---------------------------------------------------------------------------
# Resource telemetry ON/OFF lifecycle
# ---------------------------------------------------------------------------

def test_resource_telemetry_off_is_a_noop(tmp_path: Path):
    events = []
    ctrl = ztelemetry.ResourceTelemetryController(
        enabled=False,
        interval_sec=0.5,
        callback=lambda *a, **k: events.append((a, k)),
        csv_path=str(tmp_path / "off.csv"),
    )
    ctrl.maybe_emit_stats({"phase_name": "Init"})
    ctrl.close()
    assert events == []
    assert not (tmp_path / "off.csv").exists()


def test_resource_telemetry_on_emits_stats_and_csv(tmp_path: Path):
    events = []
    ctrl = ztelemetry.ResourceTelemetryController(
        enabled=True,
        interval_sec=0.5,
        callback=lambda *a, **k: events.append((a, k)),
        csv_path=str(tmp_path / "on.csv"),
    )
    ctrl.maybe_emit_stats(
        {
            "phase_name": "Per-cell stacking",
            "phase_index": 3,
            "tiles_done": 4,
            "tiles_total": 9,
        }
    )
    ctrl.close()
    assert len(events) == 1
    assert events[0][0][0] == "STATS_UPDATE"
    assert events[0][1]["tiles_done"] == 4
    csv_path = tmp_path / "on.csv"
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    assert "phase_name" in text and "Per-cell stacking" in text


def test_zegrid_entrypoint_accepts_telemetry_controller():
    import inspect

    from zemosaic import zemosaic_zegrid_mode as zzm

    params = inspect.signature(zzm.run_zegrid_mode).parameters
    assert "telemetry" in params
