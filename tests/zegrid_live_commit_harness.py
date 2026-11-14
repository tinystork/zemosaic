"""Test-only spawned-run harness for the ZM-CELL-LIVE-COMMIT lot 1 witnesses.

It runs the REAL production entry point (``run_zegrid_mode``) in its own process,
so the durable store, the active CPU wave driver, the per-cell reservation
dispatcher (``pmap_reserved``), the parent result consumer and the transaction
store/fence are all the production code paths.

The only test controls are bounded and explicit (all FILESYSTEM based, so they
work across the process pool without any sleep-based race):

* ``gate_cell`` / ``release_path``: that cell's worker does not run until the
  test creates ``release_path``;
* ``done_marker_cell`` / ``done_marker_path``: that cell's worker writes a marker
  right after the real worker returned;
* ``fail_cell`` / ``fail_release_path``: that cell's worker raises a plain
  ``ValueError`` (a real task failure), optionally only once
  ``fail_release_path`` exists;
* ``executed_log``: appends the ``cell_id`` of every cell the engine asked a cell
  worker to execute, so a resume can prove it ran ONLY the pending cells;
* ``commit_fail_cell`` / ``commit_fail_on_call``: makes
  ``RunCellStore.commit_and_release`` raise a real ``RunStoreError`` for a named
  cell or for the Nth call (deterministic commit failure);
* ``status_path``: persists ``{"status": "done"|"error", ...}`` when the run
  returns or raises (a SIGKILLed run simply never writes it);
* ``record_path``: persists the terminal exception record when one exists.

The gate wrapper is a MODULE-LEVEL function because ``pmap``/``pmap_reserved``
submit the worker to a process pool (pickled by reference): after the fork the
child resolves ``zegrid_live_commit_harness._gated_stack_cell`` and inherits the
populated ``_GATE`` dict. This mirrors ``zegrid_p0_worker_harness``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

MIB = 2 ** 20
SYNTHETIC_AVAILABLE_BYTES = 8 * 2 ** 30

#: Filled by :func:`run_gated` BEFORE the production run starts.
_GATE: dict = {}


def _wait_for(path, timeout_s, what):
    if not path:
        return
    deadline = time.time() + float(timeout_s)
    while not os.path.exists(str(path)):
        if time.time() > deadline:
            raise RuntimeError(f"live-commit harness timed out waiting for {what}")
        time.sleep(0.02)


def _gated_stack_cell(task):
    """Production ``_stack_cell`` plus the bounded test controls above."""
    real = _GATE["real"]
    cid = str(task[0].cell_id)
    executed_log = _GATE.get("executed_log")
    if executed_log:
        with open(executed_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"cell_id": cid}) + "\n")
    timeout_s = float(_GATE.get("gate_timeout_s", 900.0))
    gate_cell = _GATE.get("gate_cell")
    if gate_cell is not None and cid == str(gate_cell):
        _wait_for(_GATE.get("release_path"), timeout_s, f"release of {cid}")
    fail_cell = _GATE.get("fail_cell")
    if fail_cell is not None and cid == str(fail_cell):
        _wait_for(_GATE.get("fail_release_path"), timeout_s, f"failure trigger for {cid}")
        raise ValueError(f"injected cell worker failure for {cid}")
    result = real(task)
    marker_cell = _GATE.get("done_marker_cell")
    if marker_cell is not None and cid == str(marker_cell):
        Path(str(_GATE["done_marker_path"])).write_text("returned", encoding="utf-8")
    return result


def spawn_probe_worker(task):
    """Module-level worker for the PORTABLE SPAWN executor witness.

    A spawned child re-imports this module, so NOTHING may be carried in module
    state: the whole control (delay, marker path, cell_id) travels in the task
    payload, which is exactly what a real spawn deployment does.
    """
    delay = float(task.get("delay_s", 0.0) or 0.0)
    if delay:
        time.sleep(delay)
    marker = task.get("marker")
    if marker:
        Path(str(marker)).write_text("done", encoding="utf-8")
    return {"cell_id": str(task.get("cell_id"))}


def _gated_consume_stack_cell(prepared, *args, **kwargs):
    """Production ``_consume_stack_cell`` with an optional POST-prepare gate.

    ZM-CELL-LIVE-OBS witness: the gate sits AFTER the real ``_prepare_stack_cell``
    (cache build + live RUNNING/progress telemetry) and BEFORE the stacking
    reduction, so a test can prove the LIVE status truthfully says RUNNING while a
    long cell is still working. It changes no production code.
    """
    cid = None
    try:
        cid = str(prepared["cell"].cell_id)
    except Exception:
        cid = None
    gate = _GATE.get("consume_gate_cell")
    if gate is not None and cid == str(gate):
        _wait_for(
            _GATE.get("consume_release_path"),
            float(_GATE.get("gate_timeout_s", 900.0)),
            f"consume release of {cid}",
        )
    return _CONSUME["real"](prepared, *args, **kwargs)


#: The production consume implementation, captured before any gate wrapper.
_CONSUME: dict = {}


def _config(store_dir):
    """The minimal durable/active configuration used by the commit-integrity tests."""
    return SimpleNamespace(
        zegrid_scheduler_mode="active",
        zegrid_durable_store_dir=str(store_dir),
        zegrid_resume_mode=str(_GATE.get("resume_mode", "never")),
        zegrid_resume_run_id=None,
        zegrid_durable_store_budget_bytes=64 * MIB,
        zegrid_layout="2x2",
        zegrid_cell_gauge_mode="global",
        zegrid_resource_recovery_enabled=False,
        zegrid_resource_recovery_deadline_s=None,
        zegrid_sip_mode="keep",
        temporary_work_dir="",
    )


def run_gated(
    *,
    corpus,
    out,
    store_dir,
    status_path,
    executed_log=None,
    gate_cell=None,
    release_path=None,
    done_marker_cell=None,
    done_marker_path=None,
    fail_cell=None,
    fail_release_path=None,
    commit_fail_cell=None,
    commit_fail_on_call=None,
    resume_mode="never",
    record_path=None,
    gate_timeout_s=900.0,
    consume_gate_cell=None,
    consume_release_path=None,
):
    """Run the production ZeGrid entry once with the bounded test controls."""
    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid import run_store as zrs

    # Own SESSION/process group: a test that stops this run must be able to stop
    # the WHOLE tree (the runner, its pool workers and their inherited file
    # descriptors) exactly like a user closing the application, instead of
    # leaving orphan workers that still hold the store lock.
    try:
        os.setsid()
    except OSError:  # pragma: no cover - already a group leader
        pass

    _GATE.clear()
    _GATE.update(
        {
            "real": zz._stack_cell,
            "executed_log": executed_log,
            "gate_cell": gate_cell,
            "release_path": release_path,
            "done_marker_cell": done_marker_cell,
            "done_marker_path": done_marker_path,
            "fail_cell": fail_cell,
            "fail_release_path": fail_release_path,
            "gate_timeout_s": float(gate_timeout_s),
            "resume_mode": str(resume_mode),
            "consume_gate_cell": consume_gate_cell,
            "consume_release_path": consume_release_path,
        }
    )
    zz._stack_cell = _gated_stack_cell
    # The POST-prepare gate is wired on ``_consume_stack_cell`` (resolved by the
    # module-level ``_stack_cell``), so the real prepare — and its live RUNNING /
    # progress telemetry — always runs first.
    if consume_gate_cell is not None:
        _CONSUME["real"] = zz._consume_stack_cell
        zz._consume_stack_cell = _gated_consume_stack_cell
    # A synthetic PLAN observation only (the tiny corpus is ~3 MB per cell): it
    # makes the admitted CPU plan hold several cells per wave, which is exactly
    # the production dispatch shape this lot must prove. No science is affected.
    zz.available_memory_bytes = lambda: int(SYNTHETIC_AVAILABLE_BYTES)
    if commit_fail_cell is not None or commit_fail_on_call is not None:
        real_commit = zrs.RunCellStore.commit_and_release
        state = {"calls": 0}

        def _failing_commit(self, registry, cell_id, **kwargs):
            state["calls"] += 1
            if commit_fail_cell is not None and str(cell_id) == str(
                commit_fail_cell
            ):
                raise zrs.RunStoreError(
                    f"injected durable commit failure for cell {cell_id}"
                )
            if commit_fail_on_call is not None and state["calls"] == int(
                commit_fail_on_call
            ):
                raise zrs.RunStoreError(
                    "injected durable commit failure on commit call "
                    f"{state['calls']} (cell {cell_id})"
                )
            return real_commit(self, registry, cell_id, **kwargs)

        zrs.RunCellStore.commit_and_release = _failing_commit

    payload = {"status": "done"}
    try:
        zz.run_zegrid_mode(
            str(corpus),
            str(out),
            None,
            zconfig=_config(store_dir),
            zegrid_halo_px=8,
            use_gpu=False,
        )
    except BaseException as exc:  # noqa: BLE001 - the harness reports, never hides
        payload = {
            "status": "error",
            "type": type(exc).__name__,
            "message": str(exc)[:4000],
            "chain": _chain(exc),
        }
        record = getattr(exc, "record", None)
        if record_path and record is not None:
            try:
                Path(record_path).write_text(
                    json.dumps(payload, default=str), encoding="utf-8"
                )
            except Exception:
                pass
    finally:
        try:
            Path(status_path).write_text(
                json.dumps(payload, default=str), encoding="utf-8"
            )
        except Exception:
            pass


def _chain(exc) -> list[str]:
    out: list[str] = []
    current = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        out.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return out
