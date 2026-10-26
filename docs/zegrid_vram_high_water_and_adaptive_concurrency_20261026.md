# ZeGrid — measured VRAM peak and adaptive device concurrency (design note)

Mission: `ZM-WIN-VRAM-HW-20261026` (branch `feat/zegrid-vram-highwater`).
Base commit: `49975540d6d3d3e77486e7ed3a47f90db278d211` (published `origin/beta`).

## 1. Problem observed on a real Windows run (M106)

Three numbers disagreed about the same per-cell stacking phase:

| Source | Value | What it really was |
|---|---|---|
| Scheduler admission | ~4.13 GiB | the plan's device reservation for tile 448 (an upper bound) |
| CuPy sample (`gpu_used_mb`) | 1.08 GiB max | **6 point samples** over a 465 s stack |
| User's Windows Resource Monitor | ~2.2 GiB | a coarse, human-observed peak |

Two distinct defects, both fixed here:

1. **Telemetry sampled only at phase boundaries.** `_telemetry_stats` ran at
   setup / layout / gauge / per-cell completion / assembly. The Resource monitor
   therefore never monitored *during* the stack.
2. **GPU snapshots were taken at cell boundaries**, i.e. right after
   `gpu_memory.gpu_cell_boundary_cleanup()` purged the allocator pool. The
   intra-cell peak was invisible **by construction**. The existing label
   (`instantaneous_snapshot_not_high_water`) was honest, but unusable to decide
   any concurrency.

## 2. What changed (implementation)

### 2.1 Periodic sampling during `per_cell_stack`

New module `src/zemosaic/core/zegrid/vram_high_water.py`:

- `VramHighWaterSampler` runs a daemon thread that samples on a cadence derived
  from `resource_telemetry_interval_sec` (default 1.5 s, hard floor 0.5 s).
- Active **only** when a telemetry consumer is supplied (the Resource-monitor
  option is ON) and the effective backend is GPU. Otherwise no thread is
  created and the published record is null + explicit reason
  (`resource_telemetry_disabled`, `gpu_disabled`, `gpu_not_effective`).
- Every sample is mirrored into the existing telemetry context, so the
  Resource monitor / CSV now receives `STATS_UPDATE` rows **during** the phase,
  not only at its boundaries.
- The sampler is registered process-wide. The success path stops it right after
  the per-cell wall clock; `run_zegrid_mode` wraps every `_run_single` call in a
  `finally: stop_all_active_vram_samplers()` safety net, so a science failure,
  a cancellation (`BaseException`) or an unexpected escape cannot leak the
  thread. A GPU→CPU runtime fallback deliberately **keeps** sampling across the
  CPU rerun: the peak of the whole per-cell phase is what was being measured.

### 2.2 Sampled high-water (MAX, never last)

The sampler keeps the **maximum** `gpu_used_mb` observed, the number of
successful samples, the source that answered, and the effective cadence. It
uses the CUDA-free probe `zemosaic_resource_telemetry.sample_gpu_via_nvml()`
(`pynvml` → ctypes NVML → `nvidia-smi`): no CUDA context, no CuPy, no
`memGetInfo()`. This is asserted by an AST guard in the tests, extended to the
new module and to the new engine helpers.

Published in the manifest (flat keys) and in every telemetry row:

| Field | Meaning |
|---|---|
| `vram_high_water_sampled_mb` | MAX `gpu_used_mb` observed during the stack (MiB), or `null` |
| `vram_high_water_label` | always `sampled_high_water_not_driver_counter` |
| `vram_high_water_sample_count` | number of successful numeric samples |
| `vram_high_water_interval_s` | **effective** cadence actually honoured |
| `vram_reservation_mb` | the admitted plan's device reservation for the executing tile (MiB) |
| `vram_high_water_source` | `pynvml` / `ctypes_nvml` / `nvidia_smi` / `null` |
| `vram_high_water_effective_cadence_degraded` | `true` when the probe could not honour the requested cadence |
| `vram_high_water_unavailable_reason` | explicit token when nothing was sampled |

The full record is also stored under `per_cell_diagnostics.vram_high_water`
(schema `zemosaic-win-vram-hw-sampled-v1`) with the window, attempt count,
per-source counts and bounded error tokens.

This is a **sampled** peak, not the driver's hardware high-water counter. The
label says so everywhere.

### 2.3 Cadence honesty on the `nvidia-smi` path

`nvidia-smi` answers through a **5 s TTL cache** in the probe module. Looping it
at 0.5 s would repeat the same cached value while claiming a 0.5 s cadence.
The sampler therefore:

- uses the requested cadence when `pynvml` or `ctypes_nvml` answers;
- degrades its **effective** cadence to `max(requested, 5.0 s)` when only
  `nvidia_smi` answers, records
  `vram_high_water_effective_cadence_note =
  "effective_cadence_limited_by_nvidia_smi_ttl_cache"` and sets the degraded
  flag — instead of pretending.

The sampler is best-effort: a probe exception becomes a short error token,
never an exception in the run, and `stop()` uses a bounded join (3 s) so a
wedged driver can never block the run for long.

## 3. Design note — adaptive device concurrency (NOT implemented)

This section is a design only. **No concurrency policy was changed** by this
mission; thresholds, budgets, admission, science and I/O behaviour are exactly
as before.

### 3.1 Idea

Today an ACTIVE GPU run is single-owner (`K = 1`). With a measured per-unit
device peak, a plausible rule is:

```
K = floor(device_budget / measured_peak_per_unit)
K = max(1, min(K, host_ram_limit, disk_limit, configured_cap))
```

where:

- `device_budget` is the already-admitted VRAM budget (the ACTIVE 70 % policy
  value, `gpu_qualification.budget_bytes`) — never the raw device size;
- `measured_peak_per_unit` is the **sampled** high-water of ONE unit
  (`vram_high_water_sampled_mb`), ideally as a conservative percentile over
  several units, not a single cell and not the last sample;
- `host_ram_limit` and `disk_limit` come from the **existing per-unit ledgers**
  (`per_unit_prefetch_host`, `per_unit_prefetch_disk`, `per_cell_stack_total_bytes`,
  the bounded-prefetch credit model). They already count **per unit**, so the
  host/disk side of `K` must reuse those ledgers instead of inventing a second
  accounting.

`vram_reservation_mb` (the plan's reservation) is published next to the sample
precisely so the ratio `sampled / reservation` is readable. It is an **input to
a future decision**, not a decision.

### 3.2 Data required before any such decision

1. A per-unit sampled peak distribution (at least several cells, on the target
   OS/driver), with the sampler source and cadence recorded — a 5 s-degraded
   `nvidia-smi` cadence cannot support a 0.5 s-scale claim.
2. The effective `device_budget` and its provenance (ACTIVE policy vs legacy
   probe), already published.
3. The per-unit host and disk ledgers, unchanged.
4. The existing queue/prefetch model's per-slot device footprint (the `K`
   change must not silently multiply the prefetch credit).

### 3.3 Guardrails required with any future `K > 1`

- **Scientific A/B parity**: bit-identical (or explicitly bounded) output
  versus `K = 1` on a fixed dataset; no combine/weight/rejection/halo/offset
  change may be smuggled in with a scheduling change.
- **Determinism**: canonical task order and the deterministic plan hash must
  stay stable; runtime adaptation may only shrink, never re-slice geometry.
- **Fallback**: any VRAM/OOM/pressure signal must still cancel, release staged
  buffers and fall back to the ONE bounded CPU path — never a partial mosaic.
- **Failure semantics**: a refusal must be loud and quantitative, exactly as
  the current admission/refusal tokens are.
- **No performance promise**: the plan's `K` is a decision variable, not a
  claim that more concurrency is faster. Benchmarks must demonstrate the
  benefit on the target hardware before any default changes.

### 3.4 Explicit non-goals

No concurrency implementation, no threshold/budget/policy change, no science,
normalization, rejection, weight, combine, halo, offset, layout, prefetch or
admission change, no new dependency. The `K` formula above is documentation,
not code.

## 4. What remains NOT_RUN

- A real Windows measurement with the new sampler (Tristan must run it): the
  actual sampled peak, the probe source and the effective cadence on the target
  driver are **unknown** until that run. On a machine where only `nvidia-smi`
  answers, expect a ≥ 5 s effective cadence and a coarser peak.
- No real ZeGrid run was executed in this mission; validation is compile +
  targeted CPU-only suites.
