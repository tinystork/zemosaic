"""Test-only spawned-worker harness for the P0 resume-integrity witnesses.

It runs the REAL production worker entry point
(``run_hierarchical_mosaic_process``) and therefore the REAL ZeGrid engine,
GUI-built transport, durable store, active CPU waves and active GPU
producer/consumer pipeline. The only modifications are bounded test controls:

* ``wave_size`` / ``fail_at_wave``: fix the active CPU wave boundary and inject
  the SAME ``ParallelPoolError`` from the SAME result collection point as a real
  broken producer pool (CPU witness).
* ``gpu_control``: install the M0-style CONTROL-ONLY GPU seams (no CuPy, no
  CUDA device). The GPU readiness/plan/admission/pipeline/fallback/store code is
  the real production code; only the device execution is replaced by the CPU
  canonical executor, with the request's backend forced to ``cpu``.
* ``nested_cache_fail_at_unit``: inject a controlled ``ParallelPoolError`` from
  the NESTED ``zpar.pmap(_cell_cache_frame_worker, ...)`` inside
  ``_build_one_cell_cache`` for the Nth produced unit, chained to a real
  ``BrokenProcessPool``. This matches the observed incident shape
  (``_prepare_stack_cell -> _build_one_cell_cache -> nested pmap`` -> producer
  failure -> ``_R55GpuFallback`` -> conserved-results path).
* ``executed_log``: record the ``cell_id`` of every cell the engine asked a
  cell-cache pool to execute, so a resume can prove it ran ONLY pending cells.
* ``record_path``: when the engine raises ``_R55RunConservedResults``, persist
  its terminal ``record`` for assertions (observation only).

This module is imported by the spawned child; it must stay importable from
``tests/`` (pytest inserts that directory on ``sys.path``).
"""

from __future__ import annotations

import json


def _install_gpu_control(zz) -> None:
    """CONTROL-ONLY GPU seams: no CuPy import, no CUDA device, deterministic.

    Mirrors ``tests/test_zegrid_resume_m0_baseline_witness.py``. The GPU
    orchestration is real; only the device executor is the CPU canonical path.
    """
    from zemosaic.core.zegrid import gpu_readiness as zgr

    gib = 2 ** 30
    readiness = zgr.GpuReadiness(
        status=zgr.STATUS_USABLE,
        reason_code=zgr.REASON_GPU_OK,
        available=True,
        dependency_declared=True,
        dependency_importable=True,
        cupy_version="14.1.1",
        device_count=1,
        device_index=0,
        device_name="control-only-no-device",
        vram_total_bytes=8 * gib,
        vram_free_bytes=4 * gib,
    )
    probe = {
        "available": True,
        "device": "control-only-no-device",
        "cupy_version": "14.1.1",
        "vram_total_bytes": 8 * gib,
        "vram_free_bytes": 4 * gib,
        "reason": None,
    }
    zz.zgpu.probe_gpu_backend = lambda: dict(probe)
    zz.zgr.probe_gpu_readiness_safe = lambda **_kw: readiness
    zz.zgr.read_vram_free_bytes = lambda device_index=0: 4 * gib
    zz.available_memory_bytes = lambda: 16 * gib

    import dataclasses

    real_stream = zz.run_canonical_stack_streaming

    def _control_stream(provider, request, *args, **kwargs):
        return real_stream(
            provider,
            dataclasses.replace(request, backend="cpu"),
            *args,
            **kwargs,
        )

    zz.run_canonical_stack_streaming = _control_stream
    # SHRINK-ONLY measurement seam: one unit in flight, matching the recorded
    # incident's single-owner producer/consumer shape.
    zz._R55E2_MEASURE_PREFETCH_CAP = 0


def run_worker(
    queue,
    *,
    wave_size=None,
    fail_at_wave=None,
    executed_log=None,
    synthetic_available_bytes=None,
    gpu_control=False,
    nested_cache_fail_at_unit=None,
    record_path=None,
    **kwargs,
):
    from concurrent.futures.process import BrokenProcessPool

    from zemosaic import zemosaic_worker as zw
    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid import parallel as zpar
    from zemosaic.core.zegrid import runtime_control as zrt

    if gpu_control:
        _install_gpu_control(zz)

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

    prepare_state = {"calls": 0, "task": None, "index": None}
    if nested_cache_fail_at_unit is not None or (gpu_control and executed_log):
        real_prepare = zz._prepare_stack_cell

        def _observed_prepare_stack_cell(task):
            prepare_state["calls"] += 1
            prepare_state["task"] = task
            prepare_state["index"] = int(prepare_state["calls"]) - 1
            if gpu_control and executed_log:
                try:
                    with open(executed_log, "a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {"cell_id": str(task[0].cell_id)}
                            )
                            + "\n"
                        )
                except Exception:
                    pass
            return real_prepare(task)

        zz._prepare_stack_cell = _observed_prepare_stack_cell

    if record_path:
        real_run_zegrid = zz.run_zegrid_mode

        def _observed_run_zegrid(*args, **run_kwargs):
            try:
                return real_run_zegrid(*args, **run_kwargs)
            except BaseException as exc:
                record = getattr(exc, "record", None)
                if record is not None:
                    try:
                        with open(record_path, "w", encoding="utf-8") as handle:
                            json.dump(
                                {
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                    "record": record,
                                },
                                handle,
                                sort_keys=True,
                                default=str,
                            )
                    except Exception:
                        pass
                raise

        zz.run_zegrid_mode = _observed_run_zegrid

    real_pmap = zpar.pmap
    state = {"cell_waves": 0, "cache_builds": 0}

    def _recording_pmap(worker, tasks, *args, **pmap_kwargs):
        task_list = list(tasks)
        worker_name = getattr(worker, "__name__", "")
        if worker_name == "_cell_cache_frame_worker":
            state["cache_builds"] += 1
            if (
                nested_cache_fail_at_unit is not None
                and state["cache_builds"] == int(nested_cache_fail_at_unit)
            ):
                raise zpar.ParallelPoolError(
                    "executor failed while collecting task 0 "
                    "(BrokenProcessPool: simulated native child pool death)",
                    index=prepare_state["index"],
                    task=prepare_state["task"],
                ) from BrokenProcessPool(
                    "simulated native child pool death (P0 GPU harness)"
                )
            return real_pmap(worker, task_list, *args, **pmap_kwargs)
        if worker_name != "_stack_cell":
            return real_pmap(worker, task_list, *args, **pmap_kwargs)
        state["cell_waves"] += 1
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
                "(BrokenProcessPool: simulated native child pool death)",
                index=0,
                task=first_task,
            ) from BrokenProcessPool(
                "simulated native child pool death (P0 harness)"
            )
        return real_pmap(worker, task_list, *args, **pmap_kwargs)

    zpar.pmap = _recording_pmap
    zw.run_hierarchical_mosaic_process(queue, **kwargs)
