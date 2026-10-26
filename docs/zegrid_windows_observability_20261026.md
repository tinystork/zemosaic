# ZeGrid Windows observability — 2026-10-26

Mission: `ZM-WIN-OBSERVABILITY-20261026` (branch `fix/zegrid-windows-observability`).

## Findings (from the supplied evidence)

- ACTIVE GPU ran 9/9 cells and finished; the UI stayed at `tiles 0/9` for the whole
  61-minute per-cell phase because the GPU pipeline never reported per-cell
  completion. CPU/legacy paths did.
- `zegrid:cache_build` counts **frames** (0/1751) but was aliased to the
  **cells** stage `zegrid:per_cell_stack` in `progress_contract`. The Qt feed then
  learned `cell_count=1751`; `HybridEtaEstimator` keeps `max(total)` and produced
  multi-hour ETA. Cache and cell units must never be mixed.
- Crash-breadcrumb RAM keys were **system RAM**, and the GPU probe called
  `zemosaic_utils.get_gpu_vram_info`, which does not exist. GPU fields were absent.
- `read_parent_memory` only read `/proc/self/status`; Windows therefore returned
  null with an explicit reason.
- `gpu_memory._read_pool` did `int(bound_method)`, raising `TypeError` for the real
  CuPy pool API (`used_bytes()` etc.).
- ZeGrid returned before the classic/SDS `ResourceTelemetryController` setup, so
  the Resource-monitor checkbox could not affect ZeGrid at all.
- Structured `STAGE_PROGRESS` events were not copied into the breadcrumb
  `tile_progress`; `cpu_workers` published the requested value, not an effective
  count.

## Changes

- `src/zemosaic/progress_contract.py`: `zegrid:cache_build` is a diagnostic id,
  no longer an alias of `zegrid:per_cell_stack`. Unknown/diagnostic events keep
  prior global progress and are ignored by consumers.
- `src/zemosaic/zemosaic_zegrid_mode.py`:
  - cache reporter is diagnostics-only (never emits `STAGE_PROGRESS`);
  - the ACTIVE GPU consumer reports each **successful** completed cell with its
    real `cid` (`_stack_rep.progress`), before the batch ends; failed/discarded
    units are never counted;
  - on a whole-batch GPU fallback the ETA pass sampling restarts;
  - optional telemetry is threaded through `run_zegrid_mode` -> `_run_single`
    and sampled at phase boundaries and per cell; the controller is a pure
    observer and cannot alter admission.
- `src/zemosaic/zemosaic_worker.py`: creates the ZeGrid telemetry controller from
  `enable_resource_telemetry` (+ interval/CSV) and closes it in `finally`;
  breadcrumb `tile_progress` now comes from real stage events; requested workers
  are labelled as requested, not effective.
- `src/zemosaic/zemosaic_gui_qt.py`: the tiles counter only follows canonical
  (known) stages; diagnostic cache events can no longer overwrite it.
- `src/zemosaic/core/crash_breadcrumbs.py`: separates system RAM, process RSS,
  process peak (when the platform exposes one) and the explicitly labelled
  process-tree **sampled sum**; GPU data comes only from an injected CUDA-free
  probe (NVML), otherwise null + reason.
- `src/zemosaic/zemosaic_resource_telemetry.py`: public `sample_gpu_via_nvml()`
  (driver-only, no CUDA context) for breadcrumbs.
- `src/zemosaic/core/zegrid/runtime_control.py`: `read_parent_memory(path=None)`
  uses a portable psutil fallback (Windows); explicit non-default paths keep the
  strict null+reason behaviour.
- `src/zemosaic/core/zegrid/gpu_memory.py`: pool counters call bound methods when
  callable and keep supporting plain attributes; unavailable metrics are null +
  error token, never fabricated.

## Validation

- `tests/test_zegrid_windows_observability_r1.py` (new, 11 tests): unit
  separation, real Qt feed replay (cache 0/1751 -> stack 0/9 -> 1..9), process /
  tree / GPU breadcrumb fields, portable parent memory, method-style CuPy pool
  counters, telemetry ON/OFF + CSV.
- Targeted existing modules: progress contract R29, ETA R30/R31, resource
  telemetry, crash-breadcrumb witness, R55-E11 GPU adaptation, GPU admission
  fallback, R55-E2 prefetch, R55-D2 CPU active, ACTIVE OOC parity, layout
  observability, R55-E1 checkpoints.
- Two `test_zegrid_r55d2_cpu_active.py` failures
  (`test_plan_refusal_...`, `test_admission_exhaustion_...`) reproduce at
  baseline `689e9c8` and are **pre-existing**, not regressions.

## Limits / next step

- Real Windows sampling was **NOT_RUN** (no Windows host here). The portable
  paths are unit-tested with fakes only.
- The end-to-end GUI -> worker flag -> ZeGrid lifecycle was validated at the
  controller/signature level and by phase/per-cell wiring, not by a full GUI run.
- Next: run one short Windows ZeGrid ACTIVE GPU job with the Resource monitor ON
  and OFF, and confirm `resource_telemetry.csv`, `tiles 1..9/9`, ETA and
  `worker_crash_breadcrumbs.jsonl` process/system/tree/GPU fields.
