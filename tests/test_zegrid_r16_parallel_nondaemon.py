"""ZM-ZEGRID-R16 targeted tests — non-daemon worker + parent watchdog + adaptive
worker count + rejection choice honouring.

Covers (non-gated, fast):

* both spawn sites use ``daemon=False`` (static assertion);
* the parent watchdog exits a worker when its parent disappears (spawn a dummy
  parent + child, kill the parent, assert the child exits within a bounded time);
* the adaptive worker rule returns the expected values for bench cases and never
  < 2 unless CPU/RAM genuinely force it;
* the rejection mapping produces the expected science config for ``kappa_sigma``
  and ``winsorized_sigma_clip`` and warns/surfaces unsupported values.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_process_guard as zpg  # noqa: E402
from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import parallel as zpar  # noqa: E402


# ---------------------------------------------------------------------------
# (a) Both spawn sites use daemon=False
# ---------------------------------------------------------------------------

def test_both_spawn_sites_use_daemon_false():
    qt_path = SRC / "zemosaic" / "zemosaic_gui_qt.py"
    tk_path = SRC / "zemosaic" / "zemosaic_gui.py"

    qt = qt_path.read_text(encoding="utf-8")
    tk = tk_path.read_text(encoding="utf-8")

    # Qt spawn site (~line 595): the Process(...) named ZeMosaicWorkerProcessQt
    # must carry daemon=False.
    qt_worker = qt[qt.index('name="ZeMosaicWorkerProcessQt"') - 400:qt.index('name="ZeMosaicWorkerProcessQt"') + 5]
    assert "daemon=False" in qt_worker, "Qt worker spawn site must be daemon=False"
    assert "daemon=True" not in qt_worker

    # Tk spawn site (~line 4845): the Process(...) named ZeMosaicWorkerProcess
    # must carry daemon=False.
    tk_worker = tk[tk.index('name="ZeMosaicWorkerProcess"') - 400:tk.index('name="ZeMosaicWorkerProcess"') + 5]
    assert "daemon=False" in tk_worker, "Tk worker spawn site must be daemon=False"
    assert "daemon=True" not in tk_worker


# ---------------------------------------------------------------------------
# (b) Parent watchdog exits the worker when the parent disappears
# ---------------------------------------------------------------------------

def _child_entry(ready_q, parent_pid):
    """Child process: install the watchdog watching ``parent_pid``, then block in
    a Python loop. The watchdog thread raises KeyboardInterrupt in the main
    thread when the parent dies, which this loop turns into a clean exit."""
    zpg.install_parent_watchdog(parent_pid=parent_pid, poll_interval=0.2)
    ready_q.put("ready")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        ready_q.put("stopped")


def test_parent_watchdog_exits_worker_when_parent_disappears():
    """Kill the parent; the watched child must exit (cleanly) within a bound."""
    # A dummy "parent" that just lives until we kill it.
    parent = multiprocessing.Process(target=time.sleep, args=(60,), daemon=True)
    parent.start()
    try:
        q = multiprocessing.Queue()
        child = multiprocessing.Process(
            target=_child_entry, args=(q, parent.pid), daemon=False
        )
        child.start()
        assert q.get(timeout=15) == "ready", "child did not report ready"

        # Kill the parent; the child watchdog must notice and exit.
        parent.terminate()
        parent.join(timeout=10)

        child.join(timeout=15)
        assert not child.is_alive(), "watched child survived its dead parent (orphan)"
        assert child.exitcode == 0, f"child should exit cleanly (0), got {child.exitcode}"
        # The clean KeyboardInterrupt path must have been exercised.
        assert q.get(timeout=5) == "stopped", "child did not stop via the watchdog"
    finally:
        for p in (child, parent):
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


def test_watchdog_fail_open_when_parent_alive():
    """A live parent must NOT trigger the watchdog (no spurious kill)."""
    q = multiprocessing.Queue()
    parent = multiprocessing.Process(target=time.sleep, args=(60,), daemon=True)
    parent.start()
    child = multiprocessing.Process(
        target=_child_entry, args=(q, parent.pid), daemon=False
    )
    child.start()
    try:
        assert q.get(timeout=15) == "ready"
        time.sleep(1.0)  # longer than several poll intervals
        assert child.is_alive(), "watchdog killed a worker whose parent is alive"
    finally:
        child.terminate()
        parent.terminate()
        child.join(timeout=5)
        parent.join(timeout=5)


# ---------------------------------------------------------------------------
# (c) Adaptive worker count
# ---------------------------------------------------------------------------

_GAUGE_FP = zpar._GAUGE_PER_WORKER_FOOTPRINT_BYTES
_CACHE_FP = zpar._CACHE_PER_WORKER_FOOTPRINT_BYTES


def test_adaptive_worker_count_bench_cases():
    # 16 cpu / 13 GiB -> ~14 (the user's box: memory never binds, CPU is 16).
    assert zpar.adaptive_worker_count(16, 13 * 2**30, _GAUGE_FP) == 14
    # 4 cpu -> 2 (cpu - 2).
    assert zpar.adaptive_worker_count(4, 100 * 2**30, _GAUGE_FP) == 2
    # low RAM -> memory-bound (well below cpu - 2).
    assert zpar.adaptive_worker_count(16, 3 * _GAUGE_FP, _GAUGE_FP) == 3
    # plenty of RAM/CPU -> capped at 14.
    assert zpar.adaptive_worker_count(64, 1000 * 2**30, _GAUGE_FP) == 14


def test_adaptive_worker_count_never_below_two_unless_forced():
    # Healthy multi-core, plenty RAM: never < 2.
    assert zpar.adaptive_worker_count(8, 100 * 2**30, _GAUGE_FP) >= 2
    assert zpar.adaptive_worker_count(2, 100 * 2**30, _GAUGE_FP) == 2
    # Only a genuinely tiny machine forces below 2.
    assert zpar.adaptive_worker_count(1, 100 * 2**30, _GAUGE_FP) == 1
    assert zpar.adaptive_worker_count(16, int(0.5 * _GAUGE_FP), _GAUGE_FP) == 1


def test_choose_workers_auto_uses_adaptive_rule():
    # choose_workers(None, ...) must now be the R16 adaptive rule, not the old
    # DEFAULT_WORKERS=4 clamp.
    cpu = int(os.cpu_count() or 1)
    n = zpar.choose_workers(None, 100 * 2**30, _GAUGE_FP)
    assert n == zpar.adaptive_worker_count(cpu, 100 * 2**30, _GAUGE_FP)
    # On a healthy box it is >= 2 (and never stuck at the old fixed 4 when CPU is
    # larger than 6).
    if cpu >= 8:
        assert n >= 6


def test_choose_workers_explicit_request_still_honoured():
    # A manual pin still works (clamped), independent of the adaptive rule.
    assert zpar.choose_workers(3, 100 * 2**30, _GAUGE_FP) == 3
    assert zpar.choose_workers(3, int(0.1 * 2**30), _GAUGE_FP) == 1  # memory-bound


# ---------------------------------------------------------------------------
# (d) Rejection mapping honours the user's choice (or surfaces it)
# ---------------------------------------------------------------------------

def test_rejection_mapping_kappa_sigma():
    overrides, unhonoured = zz.resolve_rejection_science(
        "kappa_sigma", 3.0, 3.0, (0.05, 0.05)
    )
    assert overrides["rejection"] == "kappa_sigma"
    assert overrides["sigma_low"] == 3.0
    assert overrides["sigma_high"] == 3.0
    assert overrides["winsor_limit_low"] == 0.05
    assert overrides["winsor_limit_high"] == 0.05
    assert unhonoured == {}


def test_rejection_mapping_winsorized_sigma_clip_applied():
    overrides, unhonoured = zz.resolve_rejection_science(
        "winsorized_sigma_clip", 2.5, 3.5, (0.10, 0.10)
    )
    assert overrides["rejection"] == "winsorized_sigma_clip"
    assert overrides["sigma_low"] == 2.5
    assert overrides["sigma_high"] == 3.5
    assert overrides["winsor_limit_low"] == 0.10
    assert overrides["winsor_limit_high"] == 0.10
    assert unhonoured == {}


def test_rejection_mapping_none():
    overrides, unhonoured = zz.resolve_rejection_science(
        "none", 3.0, 3.0, (0.05, 0.05)
    )
    assert overrides["rejection"] == "none"
    assert unhonoured == {}


def test_rejection_mapping_warns_for_unsupported_algo():
    overrides, unhonoured = zz.resolve_rejection_science(
        "linear_fit_clip", 3.0, 3.0, (0.05, 0.05)
    )
    # Unsupported algo is NOT silently substituted: it is surfaced and the frozen
    # default rejection is left in place (no `rejection` override).
    assert "rejection" not in overrides
    assert unhonoured["stack_reject_algo"] == "linear_fit_clip"


def test_rejection_mapping_warns_for_invalid_params():
    # Invalid sigma (<= 0) and out-of-range winsor limits are surfaced, not dropped.
    overrides, unhonoured = zz.resolve_rejection_science(
        "winsorized_sigma_clip", -1.0, 3.0, (0.6, 0.6)
    )
    assert "rejection" in overrides and overrides["rejection"] == "winsorized_sigma_clip"
    assert "sigma_low" not in overrides  # invalid -> not forwarded
    assert "winsor_limit_low" not in overrides  # invalid -> not forwarded
    assert unhonoured["stack_kappa_low"] == -1.0
    assert unhonoured["stack_kappa_high"] == 3.0
    assert unhonoured["winsor_limits"] == (0.6, 0.6)


def test_rejection_mapping_builds_science_config():
    """The mapping flows into the real ExecutorConfig -> MiniTileScienceConfig."""
    from zemosaic.core.zegrid.executor import ExecutorConfig

    overrides, unhonoured = zz.resolve_rejection_science(
        "winsorized_sigma_clip", 2.0, 4.0, (0.15, 0.05)
    )
    cfg = ExecutorConfig().science_config()
    from dataclasses import replace

    cfg = replace(cfg, **overrides)
    assert cfg.rejection == "winsorized_sigma_clip"
    assert cfg.sigma_low == 2.0
    assert cfg.sigma_high == 4.0
    assert cfg.winsor_limit_low == 0.15
    assert cfg.winsor_limit_high == 0.05
    assert unhonoured == {}
