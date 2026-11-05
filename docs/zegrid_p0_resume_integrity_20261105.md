# ZM-P0-ZEGRID-RESUME-INTEGRITY (2026-11-05)

## Incident (observed)

A real Windows ZeGrid run (`WCSfix_2`) ran for ~7 h 16 min, completed
setup/layout/gauge and accepted 59/72 cells, then failed while
`_build_one_cell_cache` prepared cell `r0006c0005` / task 254. The terminal
error was a `ParallelPoolError` wrapping `BrokenProcessPool`.

The user-visible snapshot said generic `resume="auto"`, but the same launch
transported:

- `zegrid_resume_mode="never"`
- `zegrid_qt_save_checkpoints_enabled=false`
- `zegrid_durable_store_dir=null`

No `<output>/zegrid-durable` existed. The relaunch therefore started a fresh
run and recomputed setup/layout/gauge from zero. The error text implied that
accepted payloads were retained, but they existed only in the M1 in-memory
registry and died with the worker.

**The native cause of the Windows child-process death is NOT proven.** The
evidence proves only that a producer/executor child pool died after dispatch;
it does not prove OOM or a GPU-device failure.

## Fixed contract

1. **No silent contradiction.** On an effective ZeGrid launch, a non-off generic
   `resume` intent (`auto`, or `force` treated as `auto`) is never transported
   as `zegrid_resume_mode=never` with no store.
2. **Honour the promise.** ZeGrid + scheduler `active` + a non-off generic resume
   intent makes the launch durable for *that launch*: deterministic store
   (explicit folder or `<output>/zegrid-durable`), effective `auto` when a
   compatible run is selected, and explicit candidate selection when several
   exist. A per-launch decision is never persisted into the config file.
3. **Refuse an unkeepable promise.** If resume is requested but the effective
   scheduler is not `active`, the GUI refuses before any worker spawn with an
   actionable message (set scheduler `active`, or set `resume=off`).
4. **Honest non-durable runs.** With `resume=off` and checkpoint saving OFF, a
   non-resumable run is still supported, but the log states before spawn that
   crash recovery after a worker exit is unavailable.
5. **Snapshot == transport.** The run snapshot and the spawned worker consume the
   same per-launch transport for `zegrid_durable_store_dir`,
   `zegrid_resume_mode`, `zegrid_resume_run_id`, the stop capability and the
   effective generic resume intent. A generic invariant test covers the
   meaningful combinations (mode × scheduler × checkpoints × resume intent).
6. **Failure durability/honesty.** On a terminal wave/producer failure after
   accepted cells: with a healthy durable store the log names store, run id,
   complete and pending counts and states that committed cells are
   reopenable/resumable; without a store it states that accepted in-memory
   results are NOT resumable after worker exit. The original exception remains
   the root cause.
7. **Truthful classification.** `ParallelPoolError`/`BrokenProcessPool` is
   classified as `producer_pool_failure`, records the failing task/cell and the
   exception chain, and is never relabelled OOM, GPU-device failure or science
   error.

Science bytes, normalization, weights, rejection, order, N/tile/K/WCS, cache
format and CPU/GPU semantics are unchanged.

## Tests

- `tests/test_zegrid_p0_resume_integrity.py`
  - exact incident witness (`resume=auto`, checkpoints OFF, no store);
  - refusal for resume + non-active scheduler before spawn;
  - honest non-durable `resume=off` + checkpoints OFF;
  - snapshot/transport invariant over the meaningful combinations;
  - truthful `BrokenProcessPool` classification;
  - REAL two-worker proof: GUI-built launch → spawned worker → two durable
    commits → controlled terminal `ParallelPoolError`/`BrokenProcessPool` on the
    third wave → second GUI-built launch discovers/selects the run and executes
    ONLY the two pending cells, with committed payloads and gauge/geometry
    artifacts byte-identical.
- `tests/zegrid_p0_worker_harness.py` is the test-only spawn harness (fixed wave
  size + injected producer pool failure; it forwards the real worker/engine and
  changes no science).

Baseline gate (unchanged): `tests/test_zegrid_m6_qt_stop_resume.py`,
`tests/test_zegrid_active_resume_parity.py`,
`tests/test_zegrid_gpu_durable_integration.py`.

## Residual risk / next step

The native Windows child-death root cause is still unproven. This change makes
the recovery promise honest and proves recovery after a producer-pool death; it
does not diagnose the Windows runtime. The next step (separate mission) is a
native Windows breadcrumb/crash-dump capture around the producer child boundary.
