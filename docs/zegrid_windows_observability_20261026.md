# ZeGrid Windows observability — 2026-10-26

Mission: `ZM-WIN-OBSERVABILITY-20261026` (branch `fix/zegrid-windows-observability`).
Follow-up: `ZM-WIN-GPUPROBE-20261026` (branch `fix/zegrid-gpu-probe-fallback`).

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
  (driver-only, no CUDA context) for breadcrumbs. It now tries, in order,
  `pynvml` -> direct NVML through `ctypes` (`nvml.dll` /
  `libnvidia-ml.so.1`) -> `nvidia-smi --query-gpu=...` (2 s timeout, 5 s TTL
  cache). The first source that answers publishes `gpu_source`
  (`pynvml` | `ctypes_nvml` | `nvidia_smi`) and `gpu_backend`; if all fail the
  fields stay null with an AGGREGATED reason
  (e.g. `pynvml_unavailable;ctypes_nvml_failed:DLL;nvidia_smi_timeout`). The
  probe never loads CuPy or calls `cuda.runtime.memGetInfo()`: that path stays
  reserved for the telemetry owner thread.
- `src/zemosaic/zemosaic_zegrid_mode.py`: an ACTIVE GPU run is single-owner, so
  the crash breadcrumb now publishes `in_flight = 1`
  (`in_flight_source = "active_gpu_single_owner"`) during `per_cell_stack`, and
  clears it with an explicit reason on fallback/phase end. This is a pure
  observability tuple; no plan, admission, ETA or execution path reads it.
- `src/zemosaic/zemosaic_worker.py`: the breadcrumb context refreshes that
  published value (without ever importing the ZeGrid engine from the heartbeat)
  and explains `winsor_workers = null` with
  `winsor_workers_unavailable_reason = "not_used_by_zegrid"`.
- `src/zemosaic/core/zegrid/runtime_control.py`: `read_parent_memory(path=None)`
  uses a portable psutil fallback (Windows); explicit non-default paths keep the
  strict null+reason behaviour.
- `src/zemosaic/core/zegrid/gpu_memory.py`: pool counters call bound methods when
  callable and keep supporting plain attributes; unavailable metrics are null +
  error token, never fabricated.

## Deprecated breadcrumb keys

`ram_used_mb` (and the legacy `ram_total_mb` / `ram_pct`) are **deprecated
aliases for SYSTEM RAM**, kept only for store compatibility. A worker crash that
reads `ram_used_mb` therefore describes the whole machine, not the process.
New consumers must use:

- `system_ram_used_mb`, `system_ram_total_mb`, `system_ram_percent` — the
  system-wide view (same values as the legacy keys);
- `process_rss_mb` — the worker process RSS;
- `process_peak_rss_mb` + `process_peak_rss_source` — the process peak when the
  platform exposes one (`psutil_peak_wset`, `proc_status_VmHWM`);
- `process_tree_rss_sampled_sum_mb` (+ `..._label` and
  `process_tree_sample_count`) — an explicit SAMPLED SUM over the process tree,
  never a simultaneous footprint and never a peak.

Do not add new logic on top of `ram_used_mb`; treat it as a read-only legacy
field.

## Validation

- `tests/test_zegrid_windows_observability_r1.py` (new, 11 tests): unit
  separation, real Qt feed replay (cache 0/1751 -> stack 0/9 -> 1..9), process /
  tree / GPU breadcrumb fields, portable parent memory, method-style CuPy pool
  counters, telemetry ON/OFF + CSV.
- Targeted existing modules: progress contract R29, ETA R30/R31, resource
  telemetry, crash-breadcrumb witness, R55-E11 GPU adaptation, GPU admission
  fallback, R55-E2 prefetch, R55-D2 CPU active, ACTIVE OOC parity, layout
  observability, R55-E1 checkpoints.
- `tests/test_zegrid_windows_gpu_probe_fallback.py` (new, 9 tests): ordered
  `pynvml` -> `ctypes` NVML -> `nvidia-smi` cascade with fakes (no GPU), the
  5 s TTL cache, the aggregated all-sources-unavailable reason, a static guard
  that the probe never references CuPy/`memGetInfo`, the ZeGrid
  `in_flight = 1` publication, and a before/after proof that telemetry ON
  changes neither the plan JSON nor the resource envelope.
- Two `test_zegrid_r55d2_cpu_active.py` failures
  (`test_plan_refusal_...`, `test_admission_exhaustion_...`) reproduce at
  baseline `689e9c8` and are **pre-existing**, not regressions.

## Limits / next step

- Real Windows sampling was **NOT_RUN** (no Windows host here). The portable
  paths are unit-tested with fakes only; in particular the real `nvml.dll`
  struct layout and the real `nvidia-smi` output are exercised by fakes, not by
  a Windows GPU.
- The end-to-end GUI -> worker flag -> ZeGrid lifecycle was validated at the
  controller/signature level and by phase/per-cell wiring, not by a full GUI run.
- Next: run one short Windows ZeGrid ACTIVE GPU job with the Resource monitor ON
  and OFF, and confirm `resource_telemetry.csv`, `tiles 1..9/9`, ETA and
  `worker_crash_breadcrumbs.jsonl` process/system/tree/GPU fields.
