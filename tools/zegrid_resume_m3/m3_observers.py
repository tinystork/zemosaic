#!/usr/bin/env python3
"""ZM-ZEGRID M3 — test-only transparent observers (harness-injected).

This module adds **no production behaviour**. Every observer is installed by the
harness *around* an existing module-level callable and forwards the ORIGINAL
callable's arguments/results/exceptions unchanged. Nothing here writes a manifest
field, rewrites a sampled fact, clamps a value or replaces a policy.

Observers
---------
* :class:`ExecutionObserver` — wraps ``_stack_cell`` (the real CPU worker used by
  ``zpar.pmap``) and records ``pid``/``cell_id``/start/end monotonic per
  invocation.
* :class:`CommitObserver` — wraps ``RunCellStore.commit_cell`` and records the
  returned ``CommitReceipt`` (cell_id, attempt_id, payload_sha256, stored_bytes).
* :class:`WaitObserver` — wraps ``_r55_m3_make_coordinator`` so the REAL
  ``RecoveryCoordinator`` instance exports its own ``_log`` events live. Release,
  probe, policy, intervals and minimum bytes are untouched.
* :class:`BarrierObserver` — wraps ``_r55_run_checkpoint_sample`` and arms the
  safe-wave hook ONLY when the caller frame is the exact safe-wave sampler call
  site (``mode.py:11681`` resolved from source). Heartbeat (``:11576``), the M3
  probe lambda (``:11405``), the M3 post-wait re-sampler (``:11919``) and the GPU
  samplers (``:12705``/``:13036``) never arm it.

All trace writers are append-only JSONL. A trace failure is swallowed by the
product path (observability must never change the run); the harness compares the
trace against the commit inventory and reports ``NOT_RUN`` when evidence is
missing.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import threading
import time
from pathlib import Path

SCHEMA = "zemosaic-zegrid-m3-observer-v1"
EXECUTION_SCHEMA = "zemosaic-zegrid-m3-execution-trace-v1"
COMMIT_SCHEMA = "zemosaic-zegrid-m3-commit-trace-v1"
WAIT_SCHEMA = "zemosaic-zegrid-m3-wait-trace-v1"
BARRIER_SCHEMA = "zemosaic-zegrid-m3-barrier-trace-v1"

CPU_WAVE_ANCHOR = "facts = _r55_run_checkpoint_sample(disk_root)"
CPU_WAVE_LOOP_ANCHOR = "while offset < total_tasks:"


def _now_ns() -> int:
    return int(time.monotonic_ns())


class TraceSink:
    """Append-only JSONL sink.

    Each event is one ``os.write`` on an ``O_APPEND`` descriptor followed by
    ``fsync`` so a forked/spawned child never interleaves a half line and the
    harness can count durable commits from the file itself.
    """

    def __init__(self, path, *, role="observer"):
        self.path = Path(path)
        self.role = str(role)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.errors = 0

    def emit(self, kind: str, **fields) -> None:
        record = {
            "schema": SCHEMA,
            "kind": str(kind),
            "role": self.role,
            "pid": os.getpid(),
            "monotonic_ns": _now_ns(),
            "wall_unix_ns": int(time.time_ns()),
        }
        record.update(fields)
        line = (json.dumps(record, sort_keys=True, default=str) + "\n").encode(
            "utf-8"
        )
        try:
            with self._lock:
                fd = os.open(
                    str(self.path),
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                    0o644,
                )
                try:
                    os.write(fd, line)
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except Exception:  # noqa: BLE001 - observation is never fatal
            self.errors += 1

    def count(self, kind: str | None = None) -> int:
        try:
            total = 0
            with self.path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    if kind is None:
                        if raw.strip():
                            total += 1
                        continue
                    try:
                        if json.loads(raw).get("kind") == kind:
                            total += 1
                    except Exception:  # noqa: BLE001 - tolerate a torn tail
                        continue
            return total
        except FileNotFoundError:
            return 0


def read_trace(path):
    """Read a JSONL trace, tolerating a torn last line. Never mutates it."""
    out = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    out.append(json.loads(raw))
                except Exception:  # noqa: BLE001
                    continue
    except FileNotFoundError:
        return []
    return out


def resolve_safe_sampler_line(source_text: str) -> int:
    """Resolve the exact safe-wave sampler call site from the product source.

    The anchor is the ``facts = _r55_run_checkpoint_sample(disk_root)`` line that
    appears after the CPU wave loop ``while offset < total_tasks:``. It is the
    sampler that governs the CURRENT wave (``mode.py:11681`` at the frozen
    revision). Returns the 1-based line number; raises when the anchor is
    ambiguous/missing so a caller can never silently arm the wrong site.
    """
    lines = source_text.splitlines()
    loop_line = None
    for index, line in enumerate(lines, start=1):
        if line.strip() == CPU_WAVE_LOOP_ANCHOR:
            loop_line = index
            break
    if loop_line is None:
        raise RuntimeError("CPU wave loop anchor not found")
    candidates = []
    for index, line in enumerate(lines, start=1):
        if index <= loop_line:
            continue
        if line.strip() == CPU_WAVE_ANCHOR:
            # The governing sampler is the FIRST such call after the loop head;
            # the later M3 post-wait re-sampler is nested inside the recovery.
            candidates.append(index)
    if not candidates:
        raise RuntimeError("safe-wave sampler anchor not found")
    return candidates[0]


class ExecutionObserver:
    def __init__(self, sink: TraceSink):
        self.sink = sink
        self.installed = False
        self.original = None

    def install(self, module, *, attr="_stack_cell"):
        original = getattr(module, attr)
        if getattr(original, "_m3_observer", False):
            raise RuntimeError(f"{attr} already carries an M3 observer")
        observer = self

        @functools.wraps(original)
        def wrapper(task):
            pid = os.getpid()
            cell_id = _task_cell_id(task)
            started = _now_ns()
            observer.sink.emit(
                "execution_start", cell_id=cell_id, worker_pid=pid,
                start_monotonic_ns=started,
            )
            status = "ok"
            try:
                result = original(task)
            except BaseException:
                status = "error"
                raise
            finally:
                ended = _now_ns()
                observer.sink.emit(
                    "execution_end", cell_id=cell_id, worker_pid=pid,
                    start_monotonic_ns=started, end_monotonic_ns=ended,
                    duration_ns=int(ended - started), status=status,
                )
            return result

        wrapper._m3_observer = True
        wrapper._m3_original = original
        setattr(module, attr, wrapper)
        self.original = original
        self.installed = True
        self.sink.emit(
            "observer_installed",
            observer="execution",
            attr=str(attr),
            original_qualname=str(getattr(original, "__qualname__", "")),
        )
        return self


def _task_cell_id(task):
    try:
        return str(task[0].cell_id)
    except Exception:  # noqa: BLE001 - never change product behaviour
        return None


class CommitObserver:
    def __init__(self, sink: TraceSink):
        self.sink = sink
        self.installed = False
        self._original = None

    def install(self, store_cls, *, attr="commit_cell"):
        original = getattr(store_cls, attr)
        if getattr(original, "_m3_observer", False):
            raise RuntimeError(f"{attr} already carries an M3 observer")
        observer = self

        @functools.wraps(original)
        def wrapper(self, result, **kwargs):
            started = _now_ns()
            receipt = original(self, result, **kwargs)
            ended = _now_ns()
            observer.sink.emit(
                "commit",
                cell_id=str(getattr(receipt, "cell_id", "")),
                attempt_id=str(getattr(receipt, "attempt_id", "")),
                payload_sha256=str(getattr(receipt, "payload_sha256", "")),
                stored_bytes=int(getattr(receipt, "stored_bytes", 0)),
                logical_bytes=int(getattr(receipt, "logical_bytes", 0)),
                idempotent=bool(getattr(receipt, "idempotent", False)),
                commit_file=str(getattr(receipt, "commit_file", "")),
                start_monotonic_ns=started,
                end_monotonic_ns=ended,
            )
            return receipt

        wrapper._m3_observer = True
        wrapper._m3_original = original
        setattr(store_cls, attr, wrapper)
        self._original = original
        self.installed = True
        self.sink.emit(
            "observer_installed",
            observer="commit",
            target=f"{store_cls.__module__}.{store_cls.__name__}.{attr}",
        )
        return self

    def durable_commit_count(self) -> int:
        """Distinct non-idempotent commits recorded by THIS run's trace."""
        seen = set()
        for item in read_trace(self.sink.path):
            if item.get("kind") != "commit":
                continue
            if item.get("idempotent"):
                continue
            key = (item.get("cell_id"), item.get("attempt_id"))
            seen.add(key)
        return len(seen)


class WaitObserver:
    def __init__(self, sink: TraceSink):
        self.sink = sink
        self.installed = False
        self.coordinators = []

    def install(self, module, *, attr="_r55_m3_make_coordinator"):
        original = getattr(module, attr)
        if getattr(original, "_m3_observer", False):
            raise RuntimeError(f"{attr} already carries an M3 observer")
        observer = self

        @functools.wraps(original)
        def wrapper(**kwargs):
            coordinator = original(**kwargs)
            observer._capture(coordinator)
            return coordinator

        wrapper._m3_observer = True
        wrapper._m3_original = original
        setattr(module, attr, wrapper)
        self.installed = True
        self.sink.emit("observer_installed", observer="wait", attr=str(attr))
        return self

    def _capture(self, coordinator):
        original_log = coordinator._log

        def log(event, **fields):
            original_log(event, **fields)
            self.sink.emit(
                "wait_event",
                event=str(event),
                state=str(getattr(coordinator, "state", "")),
                episode=int(getattr(coordinator, "episode", 0)),
                coordinator_id=id(coordinator),
                **fields,
            )

        coordinator._log = log
        self.coordinators.append(coordinator)
        self.sink.emit(
            "coordinator_captured",
            coordinator_id=id(coordinator),
            run_id=str(getattr(coordinator, "run_id", "")),
        )


class BarrierObserver:
    """One-shot safe-wave barrier at the ORIGINAL sampler call site.

    ``mode="observe"`` records every sampler call with its attribution and never
    arms. ``mode="experiment"`` calls the injected ``hook`` BEFORE the original
    sampler exactly once, only when the caller frame is the resolved safe-wave
    call site and ``minimum_commits`` durable commits already exist.
    """

    def __init__(
        self,
        sink: TraceSink,
        *,
        mode="observe",
        hook=None,
        durable_commit_count=None,
        minimum_commits=0,
        ack_deadline_s=2.0,
        source_path=None,
    ):
        self.sink = sink
        self.mode = str(mode)
        self.hook = hook
        self._commit_count = durable_commit_count or (lambda: 0)
        self.minimum_commits = int(minimum_commits)
        self.ack_deadline_s = float(ack_deadline_s)
        self.source_path = Path(source_path) if source_path else None
        self.safe_line = None
        self.installed = False
        self.armed = False
        self.one_shot_consumed = False
        self.armed_count = 0
        self.hook_result = None

    def install(self, module, *, attr="_r55_run_checkpoint_sample"):
        source_path = self.source_path or Path(module.__file__)
        self.source_path = Path(source_path)
        self.module_file = Path(module.__file__).resolve()
        self.safe_line = resolve_safe_sampler_line(
            self.source_path.read_text(encoding="utf-8")
        )
        original = getattr(module, attr)
        if getattr(original, "_m3_observer", False):
            raise RuntimeError(f"{attr} already carries an M3 observer")
        observer = self

        @functools.wraps(original)
        def wrapper(disk_root):
            caller = sys._getframe(1)
            lineno = int(caller.f_lineno)
            # Attribution is the EXACT resolved call site inside the product
            # module (the statement at that line is the only sampler call), not
            # a guessed function name: the CPU wave loop is a nested closure.
            attributed = (
                lineno == observer.safe_line
                and Path(caller.f_code.co_filename).resolve()
                == Path(observer.source_path).resolve()
            )
            armed_here = False
            hook_fields = {}
            if attributed and observer._should_arm():
                armed_here = True
                observer.one_shot_consumed = True
                observer.armed_count += 1
                observer.sink.emit(
                    "barrier_hook",
                    wave_caller=str(caller.f_code.co_name),
                    cpu_sampler_line=lineno,
                    safe_site_line=int(observer.safe_line),
                    hook_monotonic_ns=_now_ns(),
                    ack_deadline_s=observer.ack_deadline_s,
                )
                try:
                    hook_fields = dict(observer.hook() or {})
                except Exception as exc:  # noqa: BLE001 - classify, never crash
                    hook_fields = {
                        "pressure_applied": False,
                        "status": "NOT_RUN",
                        "reason": f"hook_error:{type(exc).__name__}:{exc}",
                    }
                observer.hook_result = hook_fields
                observer.sink.emit(
                    "barrier_ack",
                    **{k: v for k, v in hook_fields.items()},
                )
            sampler_started = _now_ns()
            facts = original(disk_root)
            sampler_ended = _now_ns()
            try:
                raw = int(facts["host"]["raw_available_bytes"])
                source = str(facts["host"].get("source", "unknown"))
                sampled_ns = int(facts.get("sampled_monotonic_ns", 0))
            except Exception:  # noqa: BLE001
                raw, source, sampled_ns = None, "unknown", 0
            observer.sink.emit(
                "sampler",
                attribution="safe_wave_site" if attributed else "other_site",
                caller=str(caller.f_code.co_name),
                caller_line=lineno,
                safe_site_line=int(observer.safe_line),
                armed_here=bool(armed_here),
                sampler_start_monotonic_ns=sampler_started,
                sampler_end_monotonic_ns=sampler_ended,
                sampled_monotonic_ns=sampled_ns,
                raw_available_bytes=raw,
                host_source=source,
            )
            return facts

        wrapper._m3_observer = True
        wrapper._m3_original = original
        setattr(module, attr, wrapper)
        self.installed = True
        self.sink.emit(
            "observer_installed",
            observer="barrier",
            mode=self.mode,
            safe_site_line=int(self.safe_line),
            source_path=str(self.source_path),
        )
        return self

    def arm(self):
        self.armed = True
        self.sink.emit("barrier_armed", mode=self.mode)

    def _should_arm(self):
        if self.mode != "experiment":
            return False
        if not self.armed or self.one_shot_consumed:
            return False
        try:
            return int(self._commit_count()) >= self.minimum_commits
        except Exception:  # noqa: BLE001 - no evidence means no arm
            return False


class ObserverBundle:
    def __init__(self):
        self.execution = None
        self.commit = None
        self.wait = None
        self.barrier = None
        self.sinks = {}

    def as_dict(self):
        return {
            "schema": SCHEMA,
            "execution_installed": bool(self.execution and self.execution.installed),
            "commit_installed": bool(self.commit and self.commit.installed),
            "wait_installed": bool(self.wait and self.wait.installed),
            "barrier_installed": bool(self.barrier and self.barrier.installed),
            "barrier_mode": None if self.barrier is None else self.barrier.mode,
            "barrier_safe_site_line": (
                None if self.barrier is None else self.barrier.safe_line
            ),
            "traces": {k: str(v.path) for k, v in self.sinks.items()},
        }
