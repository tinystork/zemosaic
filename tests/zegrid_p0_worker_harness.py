"""Test-only spawned-worker harness for the P0 resume-integrity witness.

It runs the REAL production worker entry point (``run_hierarchical_mosaic_process``)
and therefore the REAL ZeGrid engine, GUI-built transport and durable store.
The only modifications are test controls:

* ``wave_size`` fixes the active CPU wave boundary so a bounded corpus produces
  several waves (production keeps its RAM-derived bound);
* ``fail_at_wave`` raises the SAME ``ParallelPoolError`` from the SAME result
  collection point as a real broken producer child pool, chained to a real
  ``BrokenProcessPool``. No science parameter, task, result or output is
  changed by the harness;
* ``executed_log`` records the ``cell_id`` of every cell the engine asked the
  cell-cache pool to execute, so a resume can prove it ran ONLY pending cells.

This module is imported by the spawned child; it must stay importable from
``tests/`` (pytest inserts that directory on ``sys.path``).
"""

from __future__ import annotations

import json


def run_worker(
    queue,
    *,
    wave_size=None,
    fail_at_wave=None,
    executed_log=None,
    synthetic_available_bytes=None,
    **kwargs,
):
    from concurrent.futures.process import BrokenProcessPool

    from zemosaic import zemosaic_worker as zw
    from zemosaic.core.zegrid import parallel as zpar
    from zemosaic.core.zegrid import runtime_control as zrt

    if synthetic_available_bytes is not None:
        import psutil

        real_virtual_memory = psutil.virtual_memory
        synthetic = int(synthetic_available_bytes)

        def _virtual_memory():
            base = real_virtual_memory()
            try:
                total = int(base.total)
            except Exception:
                total = max(synthetic, 1)
            percent = 100.0 * (1.0 - (synthetic / max(total, 1)))
            try:
                return base._replace(
                    available=synthetic,
                    percent=max(0.0, min(100.0, percent)),
                )
            except Exception:
                from types import SimpleNamespace

                return SimpleNamespace(
                    total=total,
                    available=synthetic,
                    percent=percent,
                    used=max(0, total - synthetic),
                    free=synthetic,
                )

        psutil.virtual_memory = _virtual_memory

    if wave_size is not None:
        fixed = int(wave_size)
        if fixed < 1:
            raise ValueError("wave_size must be >= 1")

        def _fixed_next_wave_bound(offset, total_tasks, units_in_flight):
            start = int(offset)
            return start, int(min(start + fixed, int(total_tasks)))

        zrt.next_wave_bound = _fixed_next_wave_bound

    real_pmap = zpar.pmap
    state = {"cell_waves": 0}

    def _recording_pmap(worker, tasks, *args, **pmap_kwargs):
        is_cell_batch = getattr(worker, "__name__", "") == "_stack_cell"
        if not is_cell_batch:
            return real_pmap(worker, tasks, *args, **pmap_kwargs)
        state["cell_waves"] += 1
        task_list = list(tasks)
        if executed_log:
            with open(executed_log, "a", encoding="utf-8") as handle:
                for task in task_list:
                    handle.write(
                        json.dumps({"cell_id": str(task[0].cell_id)}) + "\n"
                    )
        if fail_at_wave is not None and state["cell_waves"] == int(fail_at_wave):
            first_task = task_list[0] if task_list else None
            raise zpar.ParallelPoolError(
                "executor failed while collecting task 0 "
                f"(BrokenProcessPool: simulated native child pool death)",
                index=0,
                task=first_task,
            ) from BrokenProcessPool(
                "simulated native child pool death (P0 harness)"
            )
        return real_pmap(worker, task_list, *args, **pmap_kwargs)

    zpar.pmap = _recording_pmap
    zw.run_hierarchical_mosaic_process(queue, **kwargs)
