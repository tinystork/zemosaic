"""ZM-P0-ZEGRID-RESUME-INTEGRITY — incident regressions + real recovery proof.

This suite pins the product contract introduced for the real Windows incident
(59/72 accepted cells, then a producer child-pool death, then a user-visible
``resume=auto`` relaunch that silently restarted setup/layout/gauge from zero):

1. NO SILENT CONTRADICTION. A non-off generic resume intent on an effective
   ZeGrid launch is never transported as ``zegrid_resume_mode=never`` with no
   durable store.
2. The snapshot and the real worker transport are the SAME object for the
   store, the resume mode/run id, the stop capability and the effective generic
   resume intent.
3. A resume promise the launch cannot keep (scheduler != active) is REFUSED
   before any spawn.
4. A terminal producer/pool failure after accepted cells reports the ACTUAL
   durability: durable commits are resumable, in-memory-only results are
   explicitly not.
5. BrokenProcessPool is classified as a producer pool death, never as OOM or a
   GPU-device failure.
6. REAL two-worker proof: a real GUI-built invocation -> real spawned worker
   commits N cells -> controlled terminal ParallelPoolError/BrokenProcessPool ->
   a second real GUI-built launch selects the run and executes ONLY pending
   cells, with committed payloads and gauge/layout artifacts untouched.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import queue as queue_mod
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _path in (SRC, TESTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import zemosaic.zemosaic_config as zc  # noqa: E402
import zemosaic.zemosaic_gui_qt as gui  # noqa: E402
from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import parallel as zpar  # noqa: E402
from zemosaic.core.zegrid import run_publication as zpub  # noqa: E402

import zegrid_p0_worker_harness as harness  # noqa: E402
from test_zegrid_m6_qt_stop_resume import (  # noqa: E402
    _FakeMessageBox,
    _window,
)

CORPUS_TOOL = REPO_ROOT / "tools" / "zegrid_m6" / "make_tiny_corpus.py"

TRANSPORT_KEYS = (
    "zegrid_stop_capability",
    "zegrid_durable_store_dir",
    "zegrid_resume_mode",
    "zegrid_resume_run_id",
)


def _keys(mapping: dict) -> dict:
    return {key: mapping[key] for key in TRANSPORT_KEYS}


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:  # pragma: no cover - Qt unavailable
        pytest.skip("PySide6 unavailable")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _launch_snapshot_and_transport(win) -> tuple[dict, dict]:
    """Run the production launch policy without spawning a worker."""
    assert win._validate_zegrid_resume_intent_for_start() is True
    capability = win._resolve_stop_capability()
    win._pending_stop_capability = capability
    win._pending_run_is_zegrid = (
        capability == win.STOP_CAPABILITY_ZEGRID_DURABLE
    )
    win._log_stop_capability(capability)
    if capability == win.STOP_CAPABILITY_ZEGRID_DURABLE:
        assert win._prepare_zegrid_resume_for_start() is True
    else:
        win._pending_zegrid_resume = {
            "mode": "never",
            "run_id": None,
            "store_dir": None,
        }
    _args, transport = win._build_worker_invocation(skip_filter_ui=True)
    return win._snapshot_for_run(), transport


def _make_corpus(path: Path) -> Path:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(
        [sys.executable, str(CORPUS_TOOL), str(path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return path


def _tree_digest(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return out


def _cell_waves(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    return [
        json.loads(line)["cell_id"]
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _commits(store: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted((store / "commits").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[str(payload["cell_id"])] = {
            "attempt_id": payload.get("attempt_id"),
            "created_unix_ns": payload.get("created_unix_ns"),
        }
    return out


def _run_store_root(output: Path) -> Path:
    return output / gui.ZeMosaicQtMainWindow.ZEGRID_DEFAULT_STORE_DIRNAME


def _store_run_root(store: Path) -> Path:
    if (store / "run.json").is_file():
        return store
    children = [
        child
        for child in store.iterdir()
        if child.is_dir() and (child / "run.json").is_file()
    ]
    assert len(children) == 1, children
    return children[0]


def _spawn_harness(ctx, kwargs: dict, *, executed_log: Path | None = None,
                   wave_size=None, fail_at_wave=None,
                   synthetic_available_bytes=None, gpu_control=False,
                   nested_cache_fail_at_unit=None, record_path=None):
    queue_obj = ctx.Queue()
    kw = dict(kwargs)
    kw["crash_breadcrumbs_mode"] = "off"
    kw["filter_overrides"] = {"processing_mode": "zegrid"}
    process = ctx.Process(
        target=harness.run_worker,
        args=(queue_obj,),
        kwargs={
            **kw,
            "executed_log": None if executed_log is None else str(executed_log),
            "wave_size": wave_size,
            "fail_at_wave": fail_at_wave,
            "synthetic_available_bytes": synthetic_available_bytes,
            "gpu_control": bool(gpu_control),
            "nested_cache_fail_at_unit": nested_cache_fail_at_unit,
            "record_path": (
                None if record_path is None else str(record_path)
            ),
        },
        daemon=False,
    )
    process.start()
    return process, queue_obj


def _drain(process, queue_obj, timeout):
    messages = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            item = queue_obj.get(timeout=0.2)
        except queue_mod.Empty:
            if not process.is_alive():
                break
            continue
        messages.append(item)
        if item[0] == "PROCESS_DONE":
            break
    process.join(timeout=20.0)
    return messages


# ---------------------------------------------------------------------------
# 1. Exact incident witness + generic combination guard
# ---------------------------------------------------------------------------


def test_incident_generic_auto_is_transported_as_durable_auto(tmp_path, monkeypatch):
    """The real incident literal: resume=auto, checkpoints OFF, no store.

    The old GUI transported ``zegrid_resume_mode=never`` with no store. The
    fixed contract transports a deterministic store and mode=auto, and the run
    snapshot says exactly the same thing.
    """
    win = _window(
        tmp_path,
        monkeypatch,
        {
            "resume": "auto",
            "zegrid_qt_save_checkpoints_enabled": False,
            "zegrid_durable_store_dir": None,
            "zegrid_scheduler_mode": "active",
        },
    )
    logs: list[str] = []
    monkeypatch.setattr(
        win,
        "_append_log",
        lambda message, level=None, **kw: logs.append(str(message)),
    )
    snapshot, transport = _launch_snapshot_and_transport(win)

    store_root = str(_run_store_root(Path(win.config["output_dir"])))
    assert _keys(transport) == {
        "zegrid_stop_capability": "zegrid_durable",
        "zegrid_durable_store_dir": store_root,
        "zegrid_resume_mode": "auto",
        "zegrid_resume_run_id": None,
    }
    assert _keys(snapshot) == _keys(transport)
    assert snapshot["resume"] == "auto"
    # Never the incident contradiction:
    assert not (
        transport["zegrid_resume_mode"] == "never"
        and transport["zegrid_durable_store_dir"] is None
    )
    # The per-launch decision is NOT persisted into the saved config.
    assert win.config["zegrid_resume_mode"] == zc.DEFAULT_CONFIG[
        "zegrid_resume_mode"
    ]
    # User-visible honesty: the persisted Save checkbox is still false, but the
    # log must NOT claim this effective launch is non-durable.
    log_blob = "\n".join(logs)
    assert "checkpoint saving OFF, resume intent auto" in log_blob
    assert (
        "crash recovery after a worker exit is available from the durable store"
        in log_blob
    )


def test_resume_intent_with_non_active_scheduler_is_refused_before_spawn(
    tmp_path, monkeypatch
):
    win = _window(
        tmp_path,
        monkeypatch,
        {
            "resume": "auto",
            "zegrid_qt_save_checkpoints_enabled": False,
            "zegrid_scheduler_mode": "legacy",
        },
    )
    warnings = []
    monkeypatch.setattr(
        gui.QMessageBox,
        "warning",
        lambda *a, **k: warnings.append((a, k)),
    )
    assert win._validate_zegrid_resume_intent_for_start() is False
    assert warnings, "the refusal must be user-visible"
    text = str(warnings[0][0][2])
    assert "active" in text and "resume" in text.lower()
    # No plan was frozen and no spawn plan exists.
    assert win._pending_zegrid_resume is None


def test_checkpoints_off_and_resume_off_is_honest_non_durable(
    tmp_path, monkeypatch
):
    win = _window(
        tmp_path,
        monkeypatch,
        {
            "resume": "off",
            "zegrid_qt_save_checkpoints_enabled": False,
            "zegrid_scheduler_mode": "active",
        },
    )
    logs: list[str] = []
    monkeypatch.setattr(
        win, "_append_log", lambda message, level=None, **kw: logs.append(str(message))
    )
    snapshot, transport = _launch_snapshot_and_transport(win)
    assert _keys(transport) == {
        "zegrid_stop_capability": "legacy_graceful",
        "zegrid_durable_store_dir": None,
        "zegrid_resume_mode": "never",
        "zegrid_resume_run_id": None,
    }
    assert _keys(snapshot) == _keys(transport)
    assert snapshot["resume"] == "off"
    assert any(
        "CRASH RECOVERY AFTER A WORKER EXIT IS UNAVAILABLE" in line
        for line in logs
    ), logs


@pytest.mark.parametrize("mode", ["zegrid", "classic"])
@pytest.mark.parametrize("scheduler", ["active", "legacy"])
@pytest.mark.parametrize("checkpoints", [True, False])
@pytest.mark.parametrize("resume", ["auto", "off"])
def test_snapshot_equals_transport_for_meaningful_combinations(
    tmp_path, monkeypatch, mode, scheduler, checkpoints, resume
):
    win = _window(tmp_path, monkeypatch, {})
    if mode == "classic":
        for name, box in win._mode_checkboxes.items():
            box.setChecked(name == "classic")
    else:
        win._mode_checkboxes["zegrid"].setChecked(True)
    win.config["zegrid_scheduler_mode"] = scheduler
    win.config["zegrid_qt_save_checkpoints_enabled"] = bool(checkpoints)
    win.config["resume"] = resume
    checkbox = getattr(win, "_zegrid_save_checkpoints_checkbox", None)
    if checkbox is not None:
        checkbox.setChecked(bool(checkpoints))

    warnings = []
    monkeypatch.setattr(
        gui.QMessageBox,
        "warning",
        lambda *a, **k: warnings.append((a, k)),
    )
    allowed = win._validate_zegrid_resume_intent_for_start()
    if mode == "zegrid" and resume != "off" and scheduler != "active":
        assert allowed is False
        assert warnings
        return
    assert allowed is True

    snapshot, transport = _launch_snapshot_and_transport(win)
    for key in TRANSPORT_KEYS:
        assert snapshot[key] == transport[key], (key, snapshot[key], transport[key])

    if mode != "zegrid":
        assert transport["zegrid_stop_capability"] == "legacy_graceful"
        assert transport["zegrid_durable_store_dir"] is None
        assert transport["zegrid_resume_mode"] == "never"
        return

    durable_expected = scheduler == "active" and (
        checkpoints or resume != "off"
    )
    if durable_expected:
        assert transport["zegrid_stop_capability"] == "zegrid_durable"
        assert transport["zegrid_durable_store_dir"] is not None
        if resume != "off":
            assert transport["zegrid_resume_mode"] == "auto"
        else:
            assert transport["zegrid_resume_mode"] == "never"
        assert snapshot["resume"] == ("auto" if resume != "off" else "off")
    else:
        assert transport["zegrid_stop_capability"] == "legacy_graceful"
        assert transport["zegrid_durable_store_dir"] is None
        assert transport["zegrid_resume_mode"] == "never"


# ---------------------------------------------------------------------------
# 2. Truthful failure classification
# ---------------------------------------------------------------------------


def test_broken_pool_is_not_classified_as_oom_or_gpu_failure():
    from concurrent.futures.process import BrokenProcessPool
    from types import SimpleNamespace

    task = (SimpleNamespace(cell_id="r0006c0005"),)
    try:
        raise zpar.ParallelPoolError(
            "executor failed while collecting task 254 "
            "(BrokenProcessPool: child died)",
            index=254,
            task=task,
        ) from BrokenProcessPool("native child death")
    except zpar.ParallelPoolError as exc:
        details = zz._r55_pool_failure_details(exc)

    assert details["is_pool_failure"] is True
    assert "ParallelPoolError" in details["matched_types"]
    assert "BrokenProcessPool" in details["matched_types"]
    assert details["root_cause_type"] == "BrokenProcessPool"
    assert details["native_cause_proven"] is False
    assert details["failing_task_index"] == 254
    assert details["failing_cell_id"] == "r0006c0005"
    # No unproven causal labels.
    text = json.dumps(details).lower()
    assert "outofmemory" not in text
    assert "gpu-device failure" in text
    assert "failing_task_index" in details
    chain_types = [item["type"] for item in details["exception_chain"]]
    assert chain_types == ["ParallelPoolError", "BrokenProcessPool"]


# ---------------------------------------------------------------------------
# 3. REAL two-worker failure -> resume witness
# ---------------------------------------------------------------------------


def test_real_two_worker_pool_failure_then_gui_resume(tmp_path, monkeypatch):
    corpus = _make_corpus(tmp_path / "corpus")
    output = tmp_path / "out"
    output.mkdir(parents=True, exist_ok=True)
    executed_first = tmp_path / "executed-first.jsonl"
    executed_second = tmp_path / "executed-second.jsonl"

    base_config = {
        "resume": "auto",
        "zegrid_qt_save_checkpoints_enabled": False,
        "zegrid_durable_store_dir": None,
        "zegrid_durable_store_budget_bytes": 2 * 2 ** 30,
        "zegrid_scheduler_mode": "active",
        "zegrid_layout": "2x2",
        "zegrid_halo_px": 8,
        "zegrid_cell_gauge_mode": "per_cell",
        "use_gpu_phase5": False,
        "use_gpu_stack": False,
        "use_gpu_grid": False,
    }

    # --- FIRST real GUI-built launch (incident shape) -----------------------
    win = _window(tmp_path, monkeypatch, dict(base_config))
    win.config["input_dir"] = str(corpus)
    win.config["output_dir"] = str(output)
    snapshot1, transport1 = _launch_snapshot_and_transport(win)
    assert transport1["zegrid_stop_capability"] == "zegrid_durable"
    assert transport1["zegrid_resume_mode"] == "auto"
    store_root = Path(transport1["zegrid_durable_store_dir"])
    assert store_root == _run_store_root(output)
    assert _keys(snapshot1) == _keys(transport1)

    ctx = multiprocessing.get_context("spawn")
    first, queue1 = _spawn_harness(
        ctx,
        dict(win._build_worker_invocation(skip_filter_ui=True)[1]),
        executed_log=executed_first,
        wave_size=1,
        fail_at_wave=3,
        synthetic_available_bytes=16 * 2 ** 30,
    )
    messages1 = _drain(first, queue1, timeout=300)
    errors1 = [m for m in messages1 if m[0] == "PROCESS_ERROR"]
    assert errors1, messages1[-5:]
    error_text = str(errors1[0][3].get("error"))
    assert "executor failed while collecting task" in error_text
    assert "BrokenProcessPool" in error_text
    assert not (output / "zegrid_publications" / "CURRENT.json").exists()

    # Two waves committed before the injected third-wave pool death.
    assert _commits(store_root) and len(_commits(store_root)) == 2
    run_root = _store_run_root(store_root)
    run_id = json.loads((run_root / "run.json").read_text(encoding="utf-8"))[
        "run_id"
    ]
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    assert state["finalized"] is False
    committed = _commits(store_root)
    objects_before = _tree_digest(run_root / "objects")
    gauge_digest = hashlib.sha256(
        (run_root / "gauge.json").read_bytes()
    ).hexdigest()
    geometry_digest = hashlib.sha256(
        (run_root / "geometry.json").read_bytes()
    ).hexdigest()

    # The terminal-failure log must claim DURABLE resumability, never that the
    # accepted work was lost.
    log_blob = "\n".join(str(m[0]) for m in messages1)
    assert "Durable store=" in log_blob, log_blob[-2000:]
    assert "reopenable/resumable" in log_blob
    assert "NOT resumable after this worker exits" not in log_blob

    # --- SECOND real GUI-built launch: discovery + explicit resume ----------
    win2 = _window(tmp_path, monkeypatch, dict(base_config))
    win2.config["input_dir"] = str(corpus)
    win2.config["output_dir"] = str(output)

    info = win2._classify_zegrid_durable_candidates(str(store_root))
    resumable = [c for c in info["candidates"] if c["status"] == "resumable"]
    assert [c["run_id"] for c in resumable] == [run_id], info

    monkeypatch.setattr(gui, "QMessageBox", _FakeMessageBox)
    _FakeMessageBox.click_token = "resume"
    monkeypatch.setattr(
        gui.QInputDialog,
        "getItem",
        lambda parent, title, label, labels, current, editable: (labels[0], True),
    )
    assert win2._validate_zegrid_resume_intent_for_start() is True
    win2._pending_stop_capability = win2._resolve_stop_capability()
    assert win2._prepare_zegrid_resume_for_start() is True
    assert win2._pending_zegrid_resume == {
        "mode": "auto",
        "run_id": run_id,
        "store_dir": str(store_root),
    }
    snapshot2, resume_kwargs = _launch_snapshot_and_transport_no_prepare(win2)
    assert _keys(snapshot2) == _keys(resume_kwargs)
    assert resume_kwargs["zegrid_resume_mode"] == "auto"
    assert resume_kwargs["zegrid_resume_run_id"] == run_id
    assert resume_kwargs["zegrid_durable_store_dir"] == str(store_root)

    second, queue2 = _spawn_harness(
        ctx,
        resume_kwargs,
        executed_log=executed_second,
        synthetic_available_bytes=16 * 2 ** 30,
    )
    messages2 = _drain(second, queue2, timeout=300)
    errors2 = [m for m in messages2 if m[0] == "PROCESS_ERROR"]
    assert not errors2, errors2[:1]

    plan_lines = [str(m[0]) for m in messages2 if "resume plan" in str(m[0])]
    assert plan_lines, messages2[-5:]
    assert "complete=2" in plan_lines[0], plan_lines[0]
    assert "pending=2" in plan_lines[0], plan_lines[0]

    expected_ids = set(state["expected_cell_ids"])
    pending_ids = expected_ids - set(committed)
    executed = _cell_waves(executed_second)
    assert executed, "the second worker never reached a cell wave"
    assert set(executed) == pending_ids
    assert set(executed).isdisjoint(committed)

    # Pre-existing commits and payloads are byte-identical; the resumed run
    # added ONLY the pending commits.
    commits_after = _commits(store_root)
    for cell_id, record in committed.items():
        assert commits_after.get(cell_id) == record, cell_id
    assert len(commits_after) == len(committed) + len(pending_ids)
    objects_after = _tree_digest(run_root / "objects")
    for rel, digest in objects_before.items():
        assert objects_after.get(rel) == digest, rel
    # Gauge/layout artifacts were RESTORED, never recomputed/rewritten.
    assert hashlib.sha256((run_root / "gauge.json").read_bytes()).hexdigest() == (
        gauge_digest
    )
    assert hashlib.sha256(
        (run_root / "geometry.json").read_bytes()
    ).hexdigest() == geometry_digest
    assert "gauge and layout phases will NOT run" in "\n".join(
        str(m[0]) for m in messages2
    )

    # The resumed run completes and publishes a verified generation.
    final_state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    assert final_state["finalized"] is True
    generation = zpub.resolve_current(str(output), run_id)
    assert generation.path(zpub.ROLE_SCIENCE).is_file()


def _launch_snapshot_and_transport_no_prepare(win) -> tuple[dict, dict]:
    """Snapshot + transport for an ALREADY-prepared launch (no second prompt)."""
    _args, transport = win._build_worker_invocation(skip_filter_ui=True)
    return win._snapshot_for_run(), transport


# ---------------------------------------------------------------------------
# 4. Rework-1: exact chained-cause evidence (M2)
# ---------------------------------------------------------------------------


def test_exact_gpu_wrapper_chain_preserves_inner_pool_index_only():
    """Observed shape: wrapper -> ParallelPoolError(254) -> BrokenProcessPool.

    The ParallelPoolError carries the INNER frame-task index/task. It must not
    be mistaken for (or overwritten by) the OUTER producer unit/cell.
    """
    from concurrent.futures.process import BrokenProcessPool
    from types import SimpleNamespace

    # Frame-cache-like tuple: (frame_desc, bounds, header, hw, rgb, sup,
    # index, frame_id). It does NOT expose a cell_id.
    frame_task = (
        SimpleNamespace(frame_id="frame_0001.fits"),
        "bounds",
        "patch-header",
        (16, 16),
        "rgb.npy",
        "support.npy",
        254,
        "frame_0001.fits",
    )
    try:
        try:
            raise zpar.ParallelPoolError(
                "executor failed while collecting task 254 "
                "(BrokenProcessPool: child died)",
                index=254,
                task=frame_task,
            ) from BrokenProcessPool("native child death")
        except zpar.ParallelPoolError as inner:
            raise zz._R55GpuFallback(
                "R55_GPU_PIPELINE_FAILED", "pipeline wrapper"
            ) from inner
    except zz._R55GpuFallback as outer:
        details = zz._r55_pool_failure_details(outer)

    assert details["is_pool_failure"] is True
    assert details["failing_task_index"] == 254
    # A frame task does not expose a cell id: never invent one.
    assert details["failing_cell_id"] is None
    assert details["root_cause_type"] == "BrokenProcessPool"
    assert details["native_cause_proven"] is False
    chain_types = [item["type"] for item in details["exception_chain"]]
    assert chain_types == [
        "_R55GpuFallback",
        "ParallelPoolError",
        "BrokenProcessPool",
    ]

    # SEPARATE outer producer evidence (unit 2 -> cell r0006c0005) composes
    # without conflating the two identities.
    cell = SimpleNamespace(cell_id="r0006c0005")
    identity = zz._r55_terminal_failure_evidence(
        details, (2, (cell, "patch", "cache-dir"))
    )
    assert identity["pool_task_index"] == 254
    assert identity["producer_unit_index"] == 2
    assert identity["failing_cell_id"] == "r0006c0005"
    assert identity["failing_task_index"] == 254  # inner alias only


def test_direct_cell_object_evidence_never_invents_an_index():
    from types import SimpleNamespace

    cell = SimpleNamespace(cell_id="r0006c0005")
    # The observed bug shape: pipeline telemetry holds a cell OBJECT.
    index, cell_id = zz._r55_normalize_task_cell_evidence(cell)
    assert index is None
    assert cell_id == "r0006c0005"
    # Pipeline pair (index, canonical task).
    assert zz._r55_normalize_task_cell_evidence((7, (cell, "p"))) == (
        7,
        "r0006c0005",
    )
    # Mapping form.
    assert zz._r55_normalize_task_cell_evidence(
        {"producer_task_index": cell}
    ) == (None, "r0006c0005")
    # Malformed evidence never raises.
    assert zz._r55_normalize_task_cell_evidence(object()) == (None, None)


# ---------------------------------------------------------------------------
# 5. Rework-1: REAL nested GPU-producer failure -> GUI resume (M1)
# ---------------------------------------------------------------------------


def test_real_gpu_nested_producer_failure_then_gui_resume(tmp_path, monkeypatch):
    corpus = _make_corpus(tmp_path / "corpus")
    output = tmp_path / "out"
    output.mkdir(parents=True, exist_ok=True)
    executed_second = tmp_path / "executed-second-gpu.jsonl"
    record_path = tmp_path / "conserved-record.json"

    base_config = {
        "resume": "auto",
        "zegrid_qt_save_checkpoints_enabled": False,
        "zegrid_durable_store_dir": None,
        "zegrid_durable_store_budget_bytes": 2 * 2 ** 30,
        "zegrid_scheduler_mode": "active",
        "zegrid_layout": "2x2",
        "zegrid_halo_px": 8,
        "zegrid_cell_gauge_mode": "per_cell",
        "use_gpu_phase5": True,
        "use_gpu_stack": True,
        "use_gpu_grid": True,
    }

    # --- FIRST real GUI-built launch: ACTIVE GPU control flow -----------------
    win = _window(tmp_path, monkeypatch, dict(base_config))
    win.config["input_dir"] = str(corpus)
    win.config["output_dir"] = str(output)
    snapshot1, transport1 = _launch_snapshot_and_transport(win)
    assert transport1["zegrid_stop_capability"] == "zegrid_durable"
    assert transport1["zegrid_resume_mode"] == "auto"
    store_root = Path(transport1["zegrid_durable_store_dir"])
    assert _keys(snapshot1) == _keys(transport1)

    ctx = multiprocessing.get_context("spawn")
    first, queue1 = _spawn_harness(
        ctx,
        dict(win._build_worker_invocation(skip_filter_ui=True)[1]),
        synthetic_available_bytes=16 * 2 ** 30,
        gpu_control=True,
        nested_cache_fail_at_unit=3,
        record_path=record_path,
    )
    messages1 = _drain(first, queue1, timeout=420)
    errors1 = [m for m in messages1 if m[0] == "PROCESS_ERROR"]
    assert errors1, messages1[-8:]
    error_text = str(errors1[0][3].get("error"))
    assert "R55_RUN_RESULTS_CONSERVED" in error_text, error_text
    assert record_path.is_file(), error_text

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    record = payload["record"]
    assert payload["type"] == "_R55RunConservedResults"
    assert record["reason_code"] == "R55_RUN_RESULTS_CONSERVED"
    assert record["category"] == "producer_pool_failure"
    assert record["failure_classification"]["is_pool_failure"] is True
    assert record["failure_classification"]["native_cause_proven"] is False
    assert "BrokenProcessPool" in record["failure_classification"][
        "matched_types"
    ]
    # No unproven causal labels.
    assert record["category"] not in ("resource_shortage", "vram_refusal")
    assert "OutOfMemory" not in json.dumps(record)
    run_root = _store_run_root(store_root)
    run_id = json.loads((run_root / "run.json").read_text(encoding="utf-8"))[
        "run_id"
    ]
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    expected_ids = list(state["expected_cell_ids"])

    # TWO distinct identities, both preserved without conflation.
    inner_pool_index = record["pool_task_index"]
    # The harness raises with the ACTUAL first frame-task index of the nested
    # _build_one_cell_cache list, which is enumerate(planned)[0] == 0.
    assert inner_pool_index == 0
    assert record["producer_unit_index"] == 2
    assert inner_pool_index != record["producer_unit_index"]
    assert record["failing_task_index"] == inner_pool_index  # inner alias only
    assert record["failing_cell_id"] == expected_ids[2]
    assert (
        record["failure_classification"]["failing_task_index"]
        == inner_pool_index
    )
    assert record["failure_classification"]["failing_cell_id"] is None
    assert record["durable_resume"] is True
    assert record["accepted_in_memory_only"] is False
    assert record["durable_complete_cells"] == 2
    assert record["durable_pending_cells"] == 2
    assert record["cpu_reuse"] == "refused_no_numeric_parity_qualification"

    committed = _commits(store_root)
    assert len(committed) == 2
    objects_before = _tree_digest(run_root / "objects")
    gauge_digest = hashlib.sha256(
        (run_root / "gauge.json").read_bytes()
    ).hexdigest()
    geometry_digest = hashlib.sha256(
        (run_root / "geometry.json").read_bytes()
    ).hexdigest()

    log_blob = "\n".join(str(m[0]) for m in messages1)
    assert "producer/executor child pool died" in log_blob
    assert "DURABLE and resumable from the store" in log_blob
    assert f"run_id={run_id}" in log_blob
    assert f"pool task={inner_pool_index}" in log_blob
    assert "producer unit=2" in log_blob
    assert f"cell={expected_ids[2]}" in log_blob
    assert "NOT resumable after this worker exits" not in log_blob
    assert "whole-batch CPU rerun would double the work" in log_blob
    # No CPU whole-batch rerun executed cells: exactly the two durable commits.
    assert len(_commits(store_root)) == 2

    # --- SECOND real GUI-built launch: resume ONLY pending cells -------------
    win2 = _window(tmp_path, monkeypatch, dict(base_config))
    win2.config["input_dir"] = str(corpus)
    win2.config["output_dir"] = str(output)
    info = win2._classify_zegrid_durable_candidates(str(store_root))
    resumable = [c for c in info["candidates"] if c["status"] == "resumable"]
    assert [c["run_id"] for c in resumable] == [run_id], info

    monkeypatch.setattr(gui, "QMessageBox", _FakeMessageBox)
    _FakeMessageBox.click_token = "resume"
    monkeypatch.setattr(
        gui.QInputDialog,
        "getItem",
        lambda parent, title, label, labels, current, editable: (labels[0], True),
    )
    assert win2._validate_zegrid_resume_intent_for_start() is True
    win2._pending_stop_capability = win2._resolve_stop_capability()
    assert win2._prepare_zegrid_resume_for_start() is True
    snapshot2, resume_kwargs = _launch_snapshot_and_transport_no_prepare(win2)
    assert _keys(snapshot2) == _keys(resume_kwargs)
    assert resume_kwargs["zegrid_resume_mode"] == "auto"
    assert resume_kwargs["zegrid_resume_run_id"] == run_id
    assert resume_kwargs["zegrid_durable_store_dir"] == str(store_root)

    second, queue2 = _spawn_harness(
        ctx,
        resume_kwargs,
        executed_log=executed_second,
        synthetic_available_bytes=16 * 2 ** 30,
        gpu_control=True,
    )
    messages2 = _drain(second, queue2, timeout=420)
    errors2 = [m for m in messages2 if m[0] == "PROCESS_ERROR"]
    assert not errors2, errors2[:1]

    plan_lines = [
        str(m[0]) for m in messages2 if "resume plan" in str(m[0])
    ]
    assert plan_lines, messages2[-8:]
    assert "complete=2" in plan_lines[0], plan_lines[0]
    assert "pending=2" in plan_lines[0], plan_lines[0]

    pending_ids = set(expected_ids) - set(committed)
    executed = _cell_waves(executed_second)
    assert executed, "the resumed GPU producer never produced a cell"
    assert set(executed) == pending_ids
    assert set(executed).isdisjoint(committed)

    commits_after = _commits(store_root)
    for cell_id, attempt in committed.items():
        assert commits_after.get(cell_id) == attempt, cell_id
    assert len(commits_after) == len(committed) + len(pending_ids)

    objects_after = _tree_digest(run_root / "objects")
    for rel, digest in objects_before.items():
        assert objects_after.get(rel) == digest, rel
    assert hashlib.sha256((run_root / "gauge.json").read_bytes()).hexdigest() == (
        gauge_digest
    )
    assert hashlib.sha256(
        (run_root / "geometry.json").read_bytes()
    ).hexdigest() == geometry_digest
    assert "gauge and layout phases will NOT run" in "\n".join(
        str(m[0]) for m in messages2
    )

    final_state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    assert final_state["finalized"] is True
    generation = zpub.resolve_current(str(output), run_id)
    assert generation.path(zpub.ROLE_SCIENCE).is_file()


def test_terminal_evidence_malformed_never_masks_primary_exception():
    """Partial/malformed evidence yields nulls and never masks the primary."""
    malformed_pool = zz._r55_pool_failure_details(
        zpar.ParallelPoolError("broken", index=None, task=object())
    )
    identity = zz._r55_terminal_failure_evidence(
        malformed_pool, object(), cell_tasks=None
    )
    assert identity == {
        "pool_task_index": None,
        "producer_unit_index": None,
        "failing_cell_id": None,
        "failing_task_index": None,
    }
    partial = zz._r55_terminal_failure_evidence(
        malformed_pool, {"cell_id": None, "task": object()}, cell_tasks=None
    )
    assert all(value is None for value in partial.values())

    primary = RuntimeError("primary pipeline failure")
    try:
        raise primary
    except RuntimeError as exc:
        # The diagnostic composition must not raise, so the ORIGINAL exception
        # is the one that propagates.
        zz._r55_terminal_failure_evidence(malformed_pool, object())
        assert exc is primary


def test_product_failure_diagnostics_do_not_block_resume_but_foreign_does(
    tmp_path, monkeypatch
):
    import test_zegrid_resume_m4 as m4

    win = _window(tmp_path, monkeypatch, {})
    out = Path(win.config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    store = tmp_path / "resume-store"
    m4.seeded_store(
        store,
        identity=m4.make_identity(run_id="run-allow"),
        cell_indexes=(0,),
    )
    win.config["zegrid_durable_store_dir"] = str(store)
    win._zegrid_checkpoint_store_edit.setText(str(store))
    win._on_zegrid_checkpoint_store_edited(user_edit=True)

    # Product-owned terminal diagnostics from a crashed GPU run.
    for name in (
        "zegrid_gpu_failure.json",
        "zegrid_gpu_failure.jsonl",
        "zegrid_run.log",
    ):
        (out / name).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        win,
        "_ask_zegrid_resume_choice",
        lambda *a, **k: ("resume", "run-allow"),
    )
    win._pending_zegrid_resume = None
    before = _tree_digest(tmp_path)
    assert win._prepare_zegrid_resume_for_start() is True
    assert win._pending_zegrid_resume == {
        "mode": "auto",
        "run_id": "run-allow",
        "store_dir": str(store),
    }
    assert _tree_digest(tmp_path) == before, "resume preflight mutated bytes"

    # A genuinely unrelated file still refuses, without mutation.
    (out / "unrelated.txt").write_text("foreign", encoding="utf-8")
    before_foreign = _tree_digest(tmp_path)
    win._pending_zegrid_resume = None
    monkeypatch.setattr(
        gui.QMessageBox, "warning", lambda *a, **k: None
    )
    assert win._prepare_zegrid_resume_for_start() is False
    assert _tree_digest(tmp_path) == before_foreign
