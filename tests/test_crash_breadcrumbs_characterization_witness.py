"""Characterization witness: crash-breadcrumb / last-state / heartbeat contracts.

Mission `ZM-ARCH-R2-LOT2A-CRASH-BREADCRUMB-WITNESS-20261004`.

This is a TEST/DOCS-ONLY witness.  It pins the *current* behavior of the crash
autopsy side-channel in ``zemosaic.zemosaic_worker`` before any stateful R2
extraction, so a future move/refactor can prove "behavior unchanged".  It is a
characterization, not an endorsement: everything asserted here is the baseline
to preserve, not the desired ideal.

Scope (no production edits; no `src/` changes; CPU-only; no GPU required):
    A. ``_configure_crash_breadcrumbs``  (valid/invalid/off modes, filenames,
       dir creation, no-output/off clearing, best-effort failure, globals).
    B. ``_safe_runtime_snapshot``        (pid/ppid/time, psutil RAM fields,
       fail-open RAM probe, optional GPU fields, fail-open GPU probe).
    C. ``_emit_crash_breadcrumb``        (append-only JSONL + replace last-state,
       event/iso/snapshot/payload, ``errors_only`` substring filter, off/no-path
       no-op, ``default=str``, swallowed write/lock failures).
    D. Concurrency/lock                  (bounded multithread: JSONL lines stay
       valid/non-interleaved, last-state is a complete record).
    E. ``run_hierarchical_mosaic_process`` lifecycle in-process with a
       monkeypatched scientific runner + queue double (off/always/errors_only,
       WORKER_START/DONE, controlled exception → WORKER_EXCEPTION + PROCESS_ERROR
       + PROCESS_DONE, signal-handler restore, heartbeat thread lifecycle, and a
       bounded-event heartbeat cadence witness).
    F. Every test restores worker globals and isolates HOME/XDG/CWD/tmp.

Companion witnesses (not duplicated here): ``test_spawn_worker_process_witness.py``
covers real ``spawn`` with crash mode off; ``test_dispatch_propagation_witness.py``
already manipulates/restores the three crash globals and suppresses files/threads.
"""

from __future__ import annotations

import inspect
import json
import os
import signal
import sys
import threading
import time
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import zemosaic.zemosaic_worker as zw  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal queue double (same shape as the dispatch witness; wrapper needs .put)
# ---------------------------------------------------------------------------


class _RecordingQueue:
    """Minimal stand-in for the worker's multiprocessing queue (needs only .put())."""

    def __init__(self) -> None:
        self.items: list[object] = []

    def put(self, item: object) -> None:
        self.items.append(item)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Read the append-only breadcrumb file as a list of parsed records."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _read_state(path: Path) -> dict | None:
    """Read the last-state JSON file (or None if absent)."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _real_signature() -> inspect.Signature:
    return inspect.signature(zw.run_hierarchical_mosaic)


class _BrokenLock:
    """Lock whose acquire always fails, to witness outer-lock-failure swallowing."""

    def __enter__(self) -> None:
        raise RuntimeError("lock broken")

    def __exit__(self, *exc: object) -> bool:
        return False


class _FakeUtils:
    """Fake ``zemosaic_utils`` exposing only ``get_gpu_vram_info``."""

    def __init__(self, vram: tuple | None = None, raise_on_gpu: bool = False) -> None:
        self._vram = vram
        self._raise_on_gpu = raise_on_gpu

    def get_gpu_vram_info(self) -> tuple:
        if self._raise_on_gpu:
            raise RuntimeError("gpu probe failed")
        assert self._vram is not None
        return self._vram


class _ExplodingStr:
    """Object whose ``str()`` raises, to force best-effort config failure."""

    def __str__(self) -> str:  # noqa: D401 - not a docstring
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Isolation + global restore fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate HOME/XDG/cwd and restore worker crash globals + signal handlers."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / ".local" / "share"))

    saved = (
        zw._CRASH_BREADCRUMB_PATH,
        zw._CRASH_STATE_PATH,
        zw._CRASH_BREADCRUMB_MODE,
    )
    sigterm = signal.getsignal(signal.SIGTERM)
    sigint = signal.getsignal(signal.SIGINT)

    yield

    zw._CRASH_BREADCRUMB_PATH, zw._CRASH_STATE_PATH, zw._CRASH_BREADCRUMB_MODE = saved
    try:
        signal.signal(signal.SIGTERM, sigterm)
        signal.signal(signal.SIGINT, sigint)
    except Exception:
        pass


# ===========================================================================
# A. _configure_crash_breadcrumbs
# ===========================================================================


@pytest.mark.parametrize("mode", ["always", "errors_only"])
def test_configure_valid_modes_set_exact_paths_and_dir(
    _isolated_env: None, tmp_path: Path, mode: str
) -> None:
    out = tmp_path / "out"
    zw._configure_crash_breadcrumbs(str(out), mode=mode)

    assert zw._CRASH_BREADCRUMB_MODE == mode
    assert zw._CRASH_BREADCRUMB_PATH == out / "worker_crash_breadcrumbs.jsonl"
    assert zw._CRASH_STATE_PATH == out / "worker_last_state.json"
    assert out.is_dir(), "output directory must be created"


def test_configure_off_clears_paths_and_skips_dir(
    _isolated_env: None, tmp_path: Path
) -> None:
    out = tmp_path / "off_out"
    zw._configure_crash_breadcrumbs(str(out), mode="off")

    assert zw._CRASH_BREADCRUMB_MODE == "off"
    assert zw._CRASH_BREADCRUMB_PATH is None
    assert zw._CRASH_STATE_PATH is None
    assert not out.exists(), "off mode must not create the output directory"


def test_configure_invalid_mode_falls_back_to_always(
    _isolated_env: None, tmp_path: Path
) -> None:
    out = tmp_path / "invalid_out"
    zw._configure_crash_breadcrumbs(str(out), mode="bogus_mode")

    assert zw._CRASH_BREADCRUMB_MODE == "always"
    assert zw._CRASH_BREADCRUMB_PATH == out / "worker_crash_breadcrumbs.jsonl"
    assert zw._CRASH_STATE_PATH == out / "worker_last_state.json"


@pytest.mark.parametrize("output", [None, ""])
def test_configure_no_output_clears_paths(
    _isolated_env: None, output: str | None
) -> None:
    zw._configure_crash_breadcrumbs(output, mode="always")

    assert zw._CRASH_BREADCRUMB_PATH is None
    assert zw._CRASH_STATE_PATH is None


def test_configure_best_effort_failure_clears_paths(_isolated_env: None) -> None:
    zw._configure_crash_breadcrumbs(_ExplodingStr(), mode="always")

    assert zw._CRASH_BREADCRUMB_PATH is None
    assert zw._CRASH_STATE_PATH is None


def test_configure_worker_globals_observable(_isolated_env: None, tmp_path: Path) -> None:
    out = tmp_path / "globals_out"
    zw._configure_crash_breadcrumbs(str(out), mode="errors_only")

    # The three module-level globals remain directly observable after config.
    assert isinstance(zw._CRASH_BREADCRUMB_PATH, Path)
    assert isinstance(zw._CRASH_STATE_PATH, Path)
    assert zw._CRASH_BREADCRUMB_MODE == "errors_only"


# ===========================================================================
# B. _safe_runtime_snapshot
# ===========================================================================


def test_snapshot_has_pid_ppid_ts(_isolated_env: None) -> None:
    snap = zw._safe_runtime_snapshot()

    assert snap["pid"] == os.getpid()
    assert snap["ppid"] == os.getppid()
    assert isinstance(snap["ts_unix"], float)
    assert snap["ts_unix"] > 0.0
    assert abs(snap["ts_unix"] - time.time()) < 5.0


def test_snapshot_ram_fields_when_available(monkeypatch: pytest.MonkeyPatch, _isolated_env: None) -> None:
    fake_vm = types.SimpleNamespace(used=2 * 1024 * 1024, total=16 * 1024 * 1024, percent=12.5)
    monkeypatch.setattr(zw.psutil, "virtual_memory", lambda: fake_vm)

    snap = zw._safe_runtime_snapshot()

    assert snap["ram_used_mb"] == pytest.approx(2.0)
    assert snap["ram_total_mb"] == pytest.approx(16.0)
    assert snap["ram_pct"] == pytest.approx(12.5)


def test_snapshot_ram_probe_failure_fail_open(monkeypatch: pytest.MonkeyPatch, _isolated_env: None) -> None:
    def _boom() -> None:
        raise OSError("no /proc")

    monkeypatch.setattr(zw.psutil, "virtual_memory", _boom)

    snap = zw._safe_runtime_snapshot()

    # Fail-open: pid/ppid/ts still present, RAM fields absent.
    assert snap["pid"] == os.getpid()
    assert snap["ppid"] == os.getppid()
    assert "ts_unix" in snap
    for key in ("ram_used_mb", "ram_total_mb", "ram_pct"):
        assert key not in snap


def test_snapshot_gpu_fields_absent_by_default(_isolated_env: None) -> None:
    # At BASE, zemosaic_utils has no get_gpu_vram_info helper, so the GPU branch
    # is skipped and no gpu_* keys appear.
    snap = zw._safe_runtime_snapshot()
    for key in ("gpu_used_mb", "gpu_total_mb", "gpu_free_mb"):
        assert key not in snap


def test_snapshot_gpu_fields_when_helper_present(monkeypatch: pytest.MonkeyPatch, _isolated_env: None) -> None:
    monkeypatch.setattr(zw, "ZEMOSAIC_UTILS_AVAILABLE", True)
    monkeypatch.setattr(zw, "zemosaic_utils", _FakeUtils(vram=(100.0, 2048.0, 900.0)))

    snap = zw._safe_runtime_snapshot()

    assert snap["gpu_used_mb"] == pytest.approx(100.0)
    assert snap["gpu_total_mb"] == pytest.approx(2048.0)
    assert snap["gpu_free_mb"] == pytest.approx(900.0)


def test_snapshot_gpu_probe_failure_fail_open(monkeypatch: pytest.MonkeyPatch, _isolated_env: None) -> None:
    monkeypatch.setattr(zw, "ZEMOSAIC_UTILS_AVAILABLE", True)
    monkeypatch.setattr(zw, "zemosaic_utils", _FakeUtils(vram=(0.0, 0.0, 0.0), raise_on_gpu=True))

    snap = zw._safe_runtime_snapshot()

    for key in ("gpu_used_mb", "gpu_total_mb", "gpu_free_mb"):
        assert key not in snap


# ===========================================================================
# C. _emit_crash_breadcrumb
# ===========================================================================


def test_emit_writes_jsonl_append_and_state_replace(_isolated_env: None, tmp_path: Path) -> None:
    out = tmp_path / "out"
    zw._configure_crash_breadcrumbs(str(out), mode="always")

    zw._emit_crash_breadcrumb("FIRST", n=1)
    zw._emit_crash_breadcrumb("SECOND", n=2)

    records = _read_jsonl(out / "worker_crash_breadcrumbs.jsonl")
    assert [r["event"] for r in records] == ["FIRST", "SECOND"]

    state = _read_state(out / "worker_last_state.json")
    assert state is not None
    assert state["event"] == "SECOND"  # last-state is replaced, not appended


def test_emit_record_structure(_isolated_env: None, tmp_path: Path) -> None:
    out = tmp_path / "out"
    zw._configure_crash_breadcrumbs(str(out), mode="always")

    zw._emit_crash_breadcrumb("TEST_EVENT", a=1, b="x")

    records = _read_jsonl(out / "worker_crash_breadcrumbs.jsonl")
    assert len(records) == 1
    rec = records[0]

    assert rec["event"] == "TEST_EVENT"
    assert isinstance(rec["iso"], str) and rec["iso"].endswith("Z")
    assert rec["pid"] == os.getpid()
    assert rec["ppid"] == os.getppid()
    assert isinstance(rec["ts_unix"], float)
    assert rec["a"] == 1
    assert rec["b"] == "x"


def test_emit_payload_overrides_snapshot_keys(_isolated_env: None, tmp_path: Path) -> None:
    out = tmp_path / "out"
    zw._configure_crash_breadcrumbs(str(out), mode="always")

    # Payload is merged last, so it overrides the snapshot-derived pid/ppid/ts_unix.
    zw._emit_crash_breadcrumb("OVERRIDE", pid=999999, ppid=888888, ts_unix=1.25)

    records = _read_jsonl(out / "worker_crash_breadcrumbs.jsonl")
    assert len(records) == 1
    rec = records[0]
    assert rec["pid"] == 999999
    assert rec["ppid"] == 888888
    assert rec["ts_unix"] == 1.25
    # event/iso are not part of payload here, so they retain their emit-time values.
    assert rec["event"] == "OVERRIDE"
    assert rec["iso"].endswith("Z")


@pytest.mark.parametrize(
    "event",
    [
        "ERROR", "error", "EXCEPTION", "CRASH", "Crash", "Exception",
        # Substring (not exact-token) semantics: the filter matches when the
        # uppercased event string CONTAINS ERROR/EXCEPTION/CRASH anywhere.
        "WORKER_ERROR_X", "foo_exception_bar", "preCRASHpost",
    ],
)
def test_emit_errors_only_passes_error_substrings(
    _isolated_env: None, tmp_path: Path, event: str
) -> None:
    out = tmp_path / "out"
    zw._configure_crash_breadcrumbs(str(out), mode="errors_only")

    zw._emit_crash_breadcrumb(event)

    records = _read_jsonl(out / "worker_crash_breadcrumbs.jsonl")
    assert [r["event"] for r in records] == [event]


@pytest.mark.parametrize(
    "event",
    [
        "INFO", "WORKER_START", "heartbeat", "STAGE_PROGRESS",
        # Near-miss tokens that do NOT contain ERROR/EXCEPTION/CRASH as substrings.
        "ERR", "EROR", "EXCEPTON", "CRSH",
    ],
)
def test_emit_errors_only_suppresses_non_error_events(
    _isolated_env: None, tmp_path: Path, event: str
) -> None:
    out = tmp_path / "out"
    zw._configure_crash_breadcrumbs(str(out), mode="errors_only")

    zw._emit_crash_breadcrumb(event)

    assert not (out / "worker_crash_breadcrumbs.jsonl").exists()
    assert not (out / "worker_last_state.json").exists()


def test_emit_off_and_no_paths_noop(_isolated_env: None, tmp_path: Path) -> None:
    # mode off (paths None) -> no-op even for ERROR events
    out_off = tmp_path / "off"
    zw._configure_crash_breadcrumbs(str(out_off), mode="off")
    zw._emit_crash_breadcrumb("ERROR")
    assert not out_off.exists()

    # mode always but no paths configured -> no-op
    zw._configure_crash_breadcrumbs(None, mode="always")
    zw._emit_crash_breadcrumb("ERROR")
    assert not list(tmp_path.rglob("worker_crash_breadcrumbs.jsonl"))


def test_emit_default_str_serialization(_isolated_env: None, tmp_path: Path) -> None:
    out = tmp_path / "out"
    zw._configure_crash_breadcrumbs(str(out), mode="always")

    marker = Path("/not/json/serializable/path")
    zw._emit_crash_breadcrumb("OBJ_EVENT", obj=marker)

    records = _read_jsonl(out / "worker_crash_breadcrumbs.jsonl")
    assert records[0]["obj"] == str(marker)


def test_emit_swallows_simultaneous_write_failures(_isolated_env: None, tmp_path: Path) -> None:
    # Point BOTH paths at directories so open("a")/open("w") raise IsADirectoryError.
    jsonl_dir = tmp_path / "jsonl_dir"
    state_dir = tmp_path / "state_dir"
    jsonl_dir.mkdir()
    state_dir.mkdir()
    zw._CRASH_BREADCRUMB_PATH = jsonl_dir
    zw._CRASH_STATE_PATH = state_dir
    zw._CRASH_BREADCRUMB_MODE = "always"

    # Must not raise.
    zw._emit_crash_breadcrumb("SWALLOWED_WRITE")


def test_emit_jsonl_write_failure_state_succeeds(_isolated_env: None, tmp_path: Path) -> None:
    # JSONL path fails (directory) while the state path is a valid writable file.
    jsonl_dir = tmp_path / "jsonl_dir"
    jsonl_dir.mkdir()
    state_file = tmp_path / "worker_last_state.json"
    zw._CRASH_BREADCRUMB_PATH = jsonl_dir
    zw._CRASH_STATE_PATH = state_file
    zw._CRASH_BREADCRUMB_MODE = "always"

    zw._emit_crash_breadcrumb("JSONL_FAIL_STATE_OK")  # must not raise

    # The surviving artifact (last-state) is written.
    state = _read_state(state_file)
    assert state is not None and state["event"] == "JSONL_FAIL_STATE_OK"
    assert not (jsonl_dir / "worker_crash_breadcrumbs.jsonl").exists()


def test_emit_state_write_failure_jsonl_succeeds(_isolated_env: None, tmp_path: Path) -> None:
    # State path fails (directory) while the JSONL path is a valid writable file.
    state_dir = tmp_path / "state_dir"
    state_dir.mkdir()
    jsonl_file = tmp_path / "worker_crash_breadcrumbs.jsonl"
    zw._CRASH_BREADCRUMB_PATH = jsonl_file
    zw._CRASH_STATE_PATH = state_dir
    zw._CRASH_BREADCRUMB_MODE = "always"

    zw._emit_crash_breadcrumb("STATE_FAIL_JSONL_OK")  # must not raise

    # The surviving artifact (JSONL) is written.
    records = _read_jsonl(jsonl_file)
    assert [r["event"] for r in records] == ["STATE_FAIL_JSONL_OK"]
    assert not (state_dir / "worker_last_state.json").exists()


def test_emit_swallows_outer_lock_failure(monkeypatch: pytest.MonkeyPatch, _isolated_env: None, tmp_path: Path) -> None:
    out = tmp_path / "out"
    zw._configure_crash_breadcrumbs(str(out), mode="always")
    monkeypatch.setattr(zw, "_CRASH_BREADCRUMB_LOCK", _BrokenLock())

    # Must not raise despite the broken lock.
    zw._emit_crash_breadcrumb("SWALLOWED_LOCK")


# ===========================================================================
# D. Concurrency / lock
# ===========================================================================


def test_concurrent_emits_produce_valid_non_interleaved_jsonl(
    _isolated_env: None, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    zw._configure_crash_breadcrumbs(str(out), mode="always")

    n_threads = 8
    n_events = 25

    def _worker(thread_idx: int) -> None:
        for i in range(n_events):
            zw._emit_crash_breadcrumb(f"T{thread_idx}_E{i}", thread=thread_idx, i=i)

    threads = [threading.Thread(target=_worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
        assert not t.is_alive(), f"thread {t.name!r} did not finish within the bound"

    jsonl_path = out / "worker_crash_breadcrumbs.jsonl"
    records = _read_jsonl(jsonl_path)
    assert len(records) == n_threads * n_events, "every emitted line must be present exactly once"

    # Every line is a complete, valid JSON record with the expected event shape.
    for rec in records:
        assert rec["event"].startswith("T")
        assert rec["event"].count("_E") == 1
        assert "thread" in rec
        assert "i" in rec
        assert "pid" in rec
        assert "iso" in rec

    # last-state is a complete valid record too.
    state = _read_state(out / "worker_last_state.json")
    assert state is not None
    assert "event" in state and "iso" in state and "pid" in state


# ===========================================================================
# E. run_hierarchical_mosaic_process lifecycle (in-process)
# ===========================================================================


def _capture_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
    heartbeat_sec: float,
    raise_error: bool,
    tmp_path: Path,
    state: dict,
) -> tuple[list[dict], list[object]]:
    """Run the real wrapper in-process with ``run_hierarchical_mosaic`` replaced.

    Returns ``(breadcrumb_records, queue_items)``.  ``state`` is mutated with
    in-run observations (heartbeat thread liveness).
    """
    out = tmp_path / "out"
    real_sig = _real_signature()

    def stub(**final_kwargs: object) -> None:
        state["hb_threads_during_run"] = [
            t.name for t in threading.enumerate() if t.name == "ZeMosaicCrashHB"
        ]
        state["paths_during_run"] = (zw._CRASH_BREADCRUMB_PATH, zw._CRASH_STATE_PATH)
        if raise_error:
            raise RuntimeError("controlled boom")

    stub.__signature__ = real_sig  # type: ignore[attr-defined]
    monkeypatch.setattr(zw, "run_hierarchical_mosaic", stub)

    queue = _RecordingQueue()
    zw.run_hierarchical_mosaic_process(
        queue,
        solver_settings_dict=None,
        output_dir=str(out),
        crash_breadcrumbs_mode=mode,
        crash_breadcrumbs_heartbeat_sec=heartbeat_sec,
    )

    records = _read_jsonl(out / "worker_crash_breadcrumbs.jsonl")
    return records, list(queue.items)


def test_lifecycle_off_produces_no_files_and_no_thread(
    monkeypatch: pytest.MonkeyPatch, _isolated_env: None, tmp_path: Path
) -> None:
    state: dict = {}
    sigterm_before = signal.getsignal(signal.SIGTERM)
    sigint_before = signal.getsignal(signal.SIGINT)

    records, queue_items = _capture_lifecycle(
        monkeypatch, mode="off", heartbeat_sec=2.0, raise_error=False, tmp_path=tmp_path, state=state
    )

    assert records == []
    assert state["hb_threads_during_run"] == [], "mode off must not start a heartbeat thread"
    assert state["paths_during_run"] == (None, None)
    keys = [m[0] if isinstance(m, tuple) and m else None for m in queue_items]
    assert keys == ["PROCESS_DONE"]

    assert signal.getsignal(signal.SIGTERM) == sigterm_before
    assert signal.getsignal(signal.SIGINT) == sigint_before


def test_lifecycle_always_emits_start_and_done(
    monkeypatch: pytest.MonkeyPatch, _isolated_env: None, tmp_path: Path
) -> None:
    state: dict = {}
    sigterm_before = signal.getsignal(signal.SIGTERM)
    sigint_before = signal.getsignal(signal.SIGINT)

    # Large heartbeat interval: thread is started but never fires within the run,
    # so the JSONL is deterministically [WORKER_START, WORKER_DONE].
    records, queue_items = _capture_lifecycle(
        monkeypatch, mode="always", heartbeat_sec=3600.0, raise_error=False, tmp_path=tmp_path, state=state
    )

    assert [r["event"] for r in records] == ["WORKER_START", "WORKER_DONE"]
    assert records[0]["context"]["phase"] == "run_hierarchical_mosaic"
    assert records[0]["context"]["operation"] == "before_run"
    assert records[1]["graceful_stop"] is False
    assert isinstance(records[1]["context"], dict)

    # Heartbeat thread was started during the run (mode always) ...
    assert state["hb_threads_during_run"] == ["ZeMosaicCrashHB"]
    # ... and terminated afterward.
    assert not any(t.name == "ZeMosaicCrashHB" for t in threading.enumerate())

    keys = [m[0] if isinstance(m, tuple) and m else None for m in queue_items]
    assert keys == ["PROCESS_DONE"]

    assert signal.getsignal(signal.SIGTERM) == sigterm_before
    assert signal.getsignal(signal.SIGINT) == sigint_before


def test_lifecycle_controlled_exception(
    monkeypatch: pytest.MonkeyPatch, _isolated_env: None, tmp_path: Path
) -> None:
    state: dict = {}
    records, queue_items = _capture_lifecycle(
        monkeypatch, mode="always", heartbeat_sec=3600.0, raise_error=True, tmp_path=tmp_path, state=state
    )

    assert [r["event"] for r in records] == ["WORKER_START", "WORKER_EXCEPTION", "WORKER_DONE"]

    exc = records[1]
    assert exc["error"] == "controlled boom"
    assert "controlled boom" in exc["traceback"]

    # Breadcrumb/state paths are populated during the error path.
    keys = [m[0] if isinstance(m, tuple) and m else None for m in queue_items]
    assert keys == ["PROCESS_ERROR", "PROCESS_DONE"]

    error_msg = queue_items[0]
    assert isinstance(error_msg, tuple) and len(error_msg) == 4
    assert error_msg[2] == "ERROR"
    assert error_msg[3]["error"] == "controlled boom"
    assert error_msg[3]["breadcrumb_path"]
    assert error_msg[3]["last_state_path"]


def test_lifecycle_errors_only_suppresses_non_error_lifecycle(
    monkeypatch: pytest.MonkeyPatch, _isolated_env: None, tmp_path: Path
) -> None:
    state: dict = {}
    records, queue_items = _capture_lifecycle(
        monkeypatch, mode="errors_only", heartbeat_sec=2.0, raise_error=True, tmp_path=tmp_path, state=state
    )

    # Only the exception event survives the errors_only filter.
    assert [r["event"] for r in records] == ["WORKER_EXCEPTION"]
    # errors_only does not start the heartbeat thread.
    assert state["hb_threads_during_run"] == []

    keys = [m[0] if isinstance(m, tuple) and m else None for m in queue_items]
    assert keys == ["PROCESS_ERROR", "PROCESS_DONE"]


def test_lifecycle_errors_only_success_writes_no_files(
    monkeypatch: pytest.MonkeyPatch, _isolated_env: None, tmp_path: Path
) -> None:
    state: dict = {}
    records, queue_items = _capture_lifecycle(
        monkeypatch, mode="errors_only", heartbeat_sec=2.0, raise_error=False, tmp_path=tmp_path, state=state
    )

    assert records == []
    assert state["hb_threads_during_run"] == []
    # The output dir is created by _configure_crash_breadcrumbs, but no
    # breadcrumb/state files are written on a clean run in errors_only mode.
    out = tmp_path / "out"
    assert out.is_dir()
    assert not (out / "worker_crash_breadcrumbs.jsonl").exists()
    assert not (out / "worker_last_state.json").exists()

    keys = [m[0] if isinstance(m, tuple) and m else None for m in queue_items]
    assert keys == ["PROCESS_DONE"]


def test_heartbeat_cadence_bounded_event(
    monkeypatch: pytest.MonkeyPatch, _isolated_env: None, tmp_path: Path
) -> None:
    """Witness a real WORKER_HEARTBEAT emission using a bounded event, no arbitrary sleep."""
    out = tmp_path / "out"
    real_sig = _real_signature()
    real_emit = zw._emit_crash_breadcrumb

    heartbeat_seen = threading.Event()
    seen: list[str] = []

    def tracking_emit(event: str, **payload: object) -> None:
        real_emit(event, **payload)
        seen.append(event)
        if event == "WORKER_HEARTBEAT":
            heartbeat_seen.set()

    monkeypatch.setattr(zw, "_emit_crash_breadcrumb", tracking_emit)

    def stub(**final_kwargs: object) -> None:
        # Block only until an actual heartbeat fires (bounded, event-driven).
        assert heartbeat_seen.wait(timeout=10.0), "heartbeat did not fire within bound"

    stub.__signature__ = real_sig  # type: ignore[attr-defined]
    monkeypatch.setattr(zw, "run_hierarchical_mosaic", stub)

    queue = _RecordingQueue()
    zw.run_hierarchical_mosaic_process(
        queue,
        solver_settings_dict=None,
        output_dir=str(out),
        crash_breadcrumbs_mode="always",
        crash_breadcrumbs_heartbeat_sec=0.5,
    )

    assert "WORKER_START" in seen
    assert "WORKER_DONE" in seen
    hb_count = sum(1 for e in seen if e == "WORKER_HEARTBEAT")
    assert hb_count >= 1, "expected at least one WORKER_HEARTBEAT"

    start_idx = seen.index("WORKER_START")
    done_idx = seen.index("WORKER_DONE")
    hb_idx = seen.index("WORKER_HEARTBEAT")
    assert start_idx < hb_idx < done_idx, "heartbeat must be emitted between START and DONE"

    # Heartbeat thread terminated; no leaked thread.
    assert not any(t.name == "ZeMosaicCrashHB" for t in threading.enumerate())

    # The heartbeat line was actually persisted to the JSONL via the real emit.
    records = _read_jsonl(out / "worker_crash_breadcrumbs.jsonl")
    assert any(r["event"] == "WORKER_HEARTBEAT" for r in records)
