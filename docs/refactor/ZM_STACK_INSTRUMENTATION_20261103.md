# ZeGrid stack phase instrumentation (ZM-STACK-INSTRUMENTATION-20261103)

Diagnostic-only, default-OFF instrumentation of the real per-cell stacking
path, corrected in REWORK-1. It exists to measure where a Cell's wall time
actually goes before any scheduler/concurrency experiment is designed. It never
changes science, coverage, rejection, frame participation, reduction order,
masks or output identity, and it never feeds a plan/hash/durable identity
decision.

## Activation

Product config key (strict boolean, default `False`):

```json
{ "zegrid_stack_phase_profiling": true }
```

Headless CLI:

```bash
python -m zemosaic.zegrid_cli \
  --input /data/frames --output /data/out \
  --layout 2x2 --halo-px 64 --gauge-mode global \
  --stack-phase-profiling
```

The flag flows through `zemosaic_config.DEFAULT_CONFIG` and
`validate_zegrid_run_policy` (a non-boolean value is refused, never coerced),
through `run_zegrid_mode`, into the CPU worker initializer (`pmap`) and the GPU
consumer process. When OFF, the worker initargs keep their historical shape and
no profiler object, CUDA event, extra synchronization or per-tile record is
created.

## Where the metrics come from

Per cell, `canonical_streaming._phase2` accumulates host `perf_counter` wall
time around REAL operation boundaries.

### Disjoint top-level phases (never overlapping each other)

| Phase | What it covers |
| --- | --- |
| `tile_read_s` | ONLY `provider.get_tile()` reads for the N frames of one spatial tile |
| `tile_prepare_s` | tile array allocation, coefficient re-application (`a*raw+b`), float32 conversion, finite/valid masks and the float64 cast |
| `taper_weights_s` | footprint taper map and the `q*m*a` estimator-weight map |
| `reject_combine_host_s` | the whole reject+combine call wall time (host submission + nested host work + device wait) |
| `support_accum_s` | per-frame `w1`/`w2` support accumulation |
| `writes_s` | interior extraction, output-plane writes and rejection-sink flush |
| `boundary_sync_s` | the existing `_release_gpu_workspace` tile-boundary sync+purge (GPU only) |

### Nested call buckets (inside `reject_combine_host_s`, NOT additive with it)

| Phase | What it covers |
| --- | --- |
| `h2d_transfer_s` | each `xp.asarray(host_array)` upload call |
| `d2h_transfer_s` | each `xp.asnumpy` / `_as_host_numpy` download call (survivors, degenerate mask, median, and the reductions' own downloads) |
| `host_reduction_s` | each `np.nanmean` / `np.nanstd` / `np.sum` host reduction call |

The artifact repeats this topology in `phase_topology`. Do not sum the nested
buckets into `reject_combine_host_s`: the parent already contains them plus the
submission/device wait. Rejection statistics (`_nan_axis_mean`,
`_nan_axis_popstd`) and the mean-combine reductions (`_axis_sum_f64`) attribute
their internal D2H + host reduction + H2D calls truthfully instead of hiding the
transfer inside a single "host reduction" number.

### Device span

`device.combine_stream_span_s` is a CuPy event pair around the reject+combine
submission, read with `cupy.cuda.get_elapsed_time(start, end) / 1000.0` (CuPy's
`Event` has no `elapsed_time` method and no `enable_timing` keyword). It is a
**device-stream span**, not pure kernel time: it includes host-induced gaps
between submissions. Unavailable device timing is `null` with an explicit
`unavailable_reason` (`cpu_backend`, `cuda_event_unavailable:*`,
`cuda_event_sync_failed:*`, `cuda_event_elapsed_failed:*`); it is never faked as
`0.0`. The `perturbation` block records how many measurement synchronizations
actually happened, including the case where the synchronization succeeded but
the elapsed query failed.

`consumer_idle_s` from `BoundedPipeline.telemetry()` is a separate counter: it
is the consumer thread's HOST `queue.get()` wait (every timed-out poll, terminal
path and cancellation included), not device idle time.

## Artifact

When ON, each executed Cell carries a bounded `stack_phase_profile` snapshot.
The run aggregate is persisted additively at
`manifest["per_cell_diagnostics"]["stack_phase_profile"]`:

```json
{
  "schema": "zegrid-stack-phase-profile-v1",
  "basis": "cell_identity_current_attempt",
  "current": { ... current-attempt aggregate ... },
  "historical_restored": { ... durable restored aggregate ... }
}
```

Each side carries per-phase `total_s` + `count`, `tiles`, `frames_read`,
backend counts, device span + reasons, `phase_topology`, the summed
`perturbation` block, and `resume_attribution`. Attribution uses ACTUAL cell
identities:

* fresh run: every executed cell is `current`;
* resume: only the pending cells launched in THIS attempt are `current`;
  restored durable descriptors are `historical_restored` (their persisted
  profiles are real evidence, not "unavailable");
* cells without a profile (empty cells, or an old store that predates
  profiling) are counted in `cells_without_profile` on their side;
* the two aggregates are never summed, so restored historical work is never
  added to current resume wall time.

Only scalars/counts are retained (no per-tile or per-frame event list), so the
record is bounded by the number of phases, not by image/tile count. Duration
values are rounded to microseconds, clamped non-negative and rejected if
non-finite, so the block is strict JSON.

## Overhead and limitations

* OFF: no profiling work, no CUDA event, no measurement sync.
* ON, CPU: extra `perf_counter` calls only.
* ON, GPU: one synchronized `get_elapsed_time()` per profiled tile at the
  reject+combine boundary, i.e. an explicit measurement perturbation. Compare
  ON runs against ON runs; do not compare ON wall time to OFF as a speed claim.
* Instrumentation measures; it does not optimize. No speedup is claimed.
* `reject_combine_host_s` includes asynchronous submission and device wait; do
  not relabel it as GPU compute.
* Windows runtime/corpus experiments are out of scope for this work.

## Short profiling procedure

1. Pick ONE representative cell/tile batch and the validated geometry.
2. Run once with `--stack-phase-profiling` and a modest layout.
3. Read `manifest.per_cell_diagnostics.stack_phase_profile.current`; inspect
   `reject_combine_host_s`, its nested `h2d_transfer_s` / `d2h_transfer_s` /
   `host_reduction_s`, and the disjoint `tile_prepare_s` / `tile_read_s`.
4. If GPU is available, read `device.combine_stream_span_s` only as a stream
   span and cross-check the same run's host phases.
5. Keep the A/B gate: identical science/coverage/support/rejection/frames and
   bounded peak RAM/VRAM/disk before considering any scheduler change.
