"""ZM-ZEGRID-DISK-TELEMETRY — additive, non-intrusive disk/I-O measurement.

These tests pin the measurement contract only: no execution behaviour changes,
existing CSV columns are preserved in order, and every failure becomes a reason
instead of an exception.
"""

from __future__ import annotations

import csv
import sys
import time
from types import SimpleNamespace

import pytest

from zemosaic import zemosaic_resource_telemetry as rt

# The EXACT header before this mission (pinned): it must remain a prefix.
OLD_FIELDS = [
    "timestamp_iso", "phase_index", "phase_name", "cpu_percent",
    "ram_used_mb", "ram_total_mb", "ram_available_mb", "gpu_used_mb",
    "gpu_total_mb", "gpu_free_mb", "gpu_util_percent", "gpu_mem_util_percent",
    "gpu_name", "gpu_backend", "gpu_sample_source", "files_done",
    "files_total", "tiles_done", "tiles_total", "eta_seconds", "cpu_workers",
    "rows_per_chunk", "gpu_rows_per_chunk", "max_chunk_bytes",
    "gpu_max_chunk_bytes", "use_gpu", "use_gpu_phase5",
]
NEW_FIELDS = [
    "cache_bytes", "cache_files", "cache_delta_bytes", "io_read_bytes",
    "io_write_bytes", "io_read_delta_bytes", "io_write_delta_bytes",
    "phase_elapsed_s", "disk_unavailable_reason", "io_unavailable_reason",
]


def _controller(*, cache_root=None, disk_min_interval_sec=0.0, csv_path=None):
    return rt.ResourceTelemetryController(
        True, 0.0, lambda *a, **k: None, csv_path,
        cache_root=cache_root, disk_min_interval_sec=disk_min_interval_sec,
    )


def _ctx(phase="Layout", index=1):
    return {"phase_name": phase, "phase_index": index}


def test_csv_header_is_append_only_and_old_readers_parse(tmp_path):
    csv_path = tmp_path / "resource_telemetry.csv"
    ctrl = _controller(csv_path=str(csv_path))
    ctrl.emit_stats(_ctx(), force=True)
    ctrl.close()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, data = rows[0], rows[1:]
    assert header[: len(OLD_FIELDS)] == OLD_FIELDS, "existing columns moved/renamed"
    assert header[len(OLD_FIELDS):] == NEW_FIELDS, "new columns are not appended"
    assert data and len(data[0]) == len(header)
    # An OLD reader (positional old field set) still finds every value.
    old_view = dict(zip(OLD_FIELDS, data[0]))
    assert old_view["phase_name"] == "Layout"
    assert old_view["phase_index"] == "1"


def test_cache_bytes_grow_and_delta_matches_the_new_file(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.bin").write_bytes(b"x" * 1000)
    ctrl = _controller(cache_root=str(cache))
    first = ctrl.emit_stats(_ctx(), force=True)
    assert first["cache_bytes"] == 1000
    assert first["cache_files"] == 1
    assert first["cache_delta_bytes"] is None

    (cache / "b.bin").write_bytes(b"y" * 500)
    second = ctrl.emit_stats(_ctx(), force=True)
    assert second["cache_bytes"] == 1500
    assert second["cache_files"] == 2
    assert second["cache_delta_bytes"] == 500
    line = ctrl.format_disk_line(second)
    assert line and line.startswith("ZeGrid: disk — ")
    assert "cache=" in line and "files" in line and "phase delta=+500.0 B" in line


def test_phase_elapsed_is_positive_and_monotonic_per_phase():
    ctrl = _controller()
    first = ctrl.emit_stats(_ctx("Gauge", 2), force=True)
    time.sleep(0.02)
    second = ctrl.emit_stats(_ctx("Gauge", 2), force=True)
    assert first["phase_elapsed_s"] >= 0
    assert second["phase_elapsed_s"] > first["phase_elapsed_s"]
    # A new phase restarts the clock.
    third = ctrl.emit_stats(_ctx("Assembly", 4), force=True)
    assert third["phase_elapsed_s"] <= second["phase_elapsed_s"]


def test_a_raising_cache_walk_cannot_fail_the_sample(monkeypatch):
    ctrl = _controller(cache_root="/whatever")

    def _boom():
        raise OSError("permission denied")

    monkeypatch.setattr(ctrl, "_walk_cache", _boom)
    payload = ctrl.emit_stats(_ctx(), force=True)  # must not raise
    assert "cache_walk_failed" in payload["disk_unavailable_reason"]
    assert "cache_bytes" not in payload or payload.get("cache_bytes") is None


def test_unavailable_io_counters_record_a_reason_not_zero(monkeypatch):
    fake = SimpleNamespace()
    def _no_process(_pid):
        raise RuntimeError("io counters unavailable")
    fake.Process = _no_process
    monkeypatch.setitem(sys.modules, "psutil", fake)
    ctrl = _controller()
    payload = ctrl.emit_stats(_ctx(), force=True)
    assert payload["io_read_bytes"] is None
    assert payload["io_write_bytes"] is None
    assert "io counters unavailable" in payload["io_unavailable_reason"]
    # Never a fabricated zero.
    assert payload["io_read_bytes"] != 0 and payload["io_write_bytes"] != 0


def test_cache_walk_is_rate_limited(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.bin").write_bytes(b"x" * 10)
    ctrl = _controller(cache_root=str(cache), disk_min_interval_sec=3600.0)
    first = ctrl.emit_stats(_ctx(), force=True)
    assert first["cache_bytes"] == 10
    second = ctrl.emit_stats(_ctx(), force=True)
    assert ctrl._disk_walk_calls == 1, "two immediate samples both walked"
    assert second["disk_unavailable_reason"] == "rate_limited"
    assert second["cache_delta_bytes"] == 0
