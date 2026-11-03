# ZeGrid stack phase instrumentation (ZM-STACK-INSTRUMENTATION-20261103)

Diagnostic-only, default-OFF instrumentation of the real per-cell stacking
path. It exists to measure where a Cell's wall time actually goes before any
scheduler/concurrency experiment is designed. It never changes science,
coverage, rejection, frame participation, reduction order, masks or output
identity, and it never feeds a plan/hash/durable identity decision.

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
time around REAL operation boundaries:

| Phase | What it covers |
| --- | --- |
| `tile_read_s` | `provider.get_tile()` reads for all N frames of one spatial tile (plus the tile array allocation) |
| `tile_prepare_s` | per-frame coefficient re-application (`a*raw+b`), float32 conversion and finite/valid masks |
| `taper_weights_s` | footprint taper map and the `q*m*a` estimator-weight map |
| `reject_combine_host_s` | the whole reject+combine call wall time (host submission + inline host work) |
| `host_reduction_s` | the exact `_axis_sum_f64` / `eligible.sum` host reduction calls |
| `host_readback_s` | the exact `_as_host_numpy` result materialisations |
| `support_accum_s` | per-frame `w1`/`w2` support accumulation |
| `writes_s` | interior extraction, output-plane writes and rejection-sink flush |
| `boundary_sync_s` | the existing `_release_gpu_workspace` tile-boundary sync+purge (GPU only) |

GPU device span (when CuPy is active and measurable):
`device.combine_stream_span_s` is a pair of CUDA events around the reject+combine
submission. It is a **device-stream span**, not pure kernel time: it includes
host-induced gaps between submissions. Reading it as "GPU compute" is wrong.
When the backend is CPU or CuPy/events are unavailable, the value is `null`
with an explicit `unavailable_reason`; it is never faked as `0.0`.

`consumer_idle_s` from `BoundedPipeline.telemetry()` is a separate counter: it
is the consumer thread's HOST `queue.get()` wait (all timed-out polls included),
not device idle time.

## Artifact

When ON, each executed Cell carries a bounded `stack_phase_profile` snapshot in
its result. The run aggregate is persisted additively at
`manifest["per_cell_diagnostics"]["stack_phase_profile"]` with:

* `schema = "zegrid-stack-phase-profile-v1"` (strict JSON: finite numbers only);
* per-phase `total_s` and call `count`;
* `cells_profiled`, `tiles`, `frames_read`, backend counts;
* `device` span + unavailable reasons + the non-pure-GPU note;
* `resume_attribution`: `current` vs `historical_restored`. Restored historical
  Cells are not re-profiled and are never summed with current resume wall time.

Only scalars/counts are retained (no per-tile or per-frame event list), so the
record is bounded by the number of phases, not by image/tile count.

## Overhead and limitations

* OFF: no profiling work, no CUDA profiling API call, no measurement sync.
* ON, CPU: extra `perf_counter` calls only.
* ON, GPU: one synchronized `elapsed_time()` per tile at the reject+combine
  boundary, i.e. an explicit measurement perturbation (`perturbation` block).
  Compare ON runs against ON runs; do not compare ON wall time to OFF as a
  speed claim.
* Instrumentation measures; it does not optimize. No speedup is claimed.
* Windows runtime/corpus experiments are out of scope for this work.

## Short profiling procedure

1. Pick ONE representative cell/tile batch and the validated geometry.
2. Run once with `--stack-phase-profiling` and a modest layout.
3. Read `manifest.per_cell_diagnostics.stack_phase_profile`; compare
   `reject_combine_host_s`, `host_reduction_s`, `host_readback_s`,
   `tile_prepare_s` and `tile_read_s` against the phase totals.
4. If GPU is available, read `device.combine_stream_span_s` only as a stream
   span and cross-check the same run's host phases.
5. Keep the A/B gate: identical science/coverage/support/rejection/frames and
   bounded peak RAM/VRAM/disk before considering any scheduler change.
