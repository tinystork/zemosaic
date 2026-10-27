# ZeGrid resilient run/resume — M1 runtime conservation

Date: 2026-10-27
Mission: `ZM-ZEGRID-RESILIENT-RUN-RESUME-M1-20261027`
Phase: implementation (M1)
Baseline: `231b597867b4d251afac6b767000c8a37e2cf3b0` (`origin/beta`)
Worktree: `worktrees/zemosaic-resilient-run-resume`, branch
`feat/zegrid-resilient-run-resume`
Status: **IMPLEMENTED + REWORK-1 — awaiting Junior verification and Nono review.**
Coco does not accept M1. Section 8 records the rework-1 corrections and
supersedes the r0 test numbers below; the r0 narrative is kept as history.

M0 is CLOSED/ACCEPTED. This document does not claim durable resume, process
restart, GUI stop wiring, GPU/CPU parity or Windows qualification. It claims
only in-memory conservation of accepted cell results for the lifetime of one
run, plus the ownership/transaction/generation groundwork listed below.

---

## 1. What M1 adds

### 1.1 `src/zemosaic/core/zegrid/run_registry.py` (new)

`RunResultRegistry` is owned by the run, not by the pipeline:

- expected identity is the **sparse** `cell_id` set (never `0..K`);
- `accept()` stages the complete payload fingerprint and byte accounting
  **before** taking the lock, then inserts the record and updates counters in
  one transaction. A fingerprinting/accounting failure leaves no record;
- the fingerprint covers every core plane, the per-cell pre-offset/pre-feather
  patch (`bounds`, `core_slice`, `science`, `valid_mask`, `surviving` with
  halo), the photometric reference pin (`reference_frame_id`,
  `reference_frame_role`, `bookkeeping_reference_frame_id`, gauge mode,
  exclusions), geometry and the real backend token;
- duplicate acceptance of the same identity + same payload + same attempt is
  idempotent and counted once; a different payload/attempt is rejected and
  never overwrites the accepted record;
- `advance_generation()` fences plan transitions; older-generation late
  results are rejected without contaminating accepted state;
- the event journal is bounded (default 512 events) and holds no payload
  reference;
- `dispose()` is the single explicit ownership release and is idempotent; a
  second call does not double-count;
- `CooperativeStopChannel` provides request/ack over injectable events
  (`threading` by default, `multiprocessing` context events for spawn-safe
  workers) and never parses logs.

### 1.2 Product coordinator (`zemosaic_zegrid_mode.py`)

- `_run_single` creates ONE registry for the ACTIVE path from the sparse
  expected `cell_id` set before any cell executes.
- GPU pipeline consumer (`_consume_guarded`) accepts each successful unit and
  returns a **lightweight receipt**. `BoundedPipeline._results` therefore keeps
  receipts, not payloads; the registry is the single owner of the arrays.
- On successful GPU completion, assembly receives the registry payloads in
  canonical task order.
- On GPU refusal after `K > 0` accepted cells, the run raises
  `_R55RunConservedResults` (reason `R55_RUN_RESULTS_CONSERVED`) instead of
  re-running the whole batch on CPU. Recomputing accepted cells would double
  the work and silently mix backends before numeric parity is qualified.
  Accepted payloads remain owned/accessible via `exc.result_registry` until
  explicit disposal. `K = 0` keeps the historical exact whole-batch CPU
  fallback.
- Completed units are recorded as `retained=true, discarded=false` in the GPU
  failure artifact; only unaccepted work is discarded.
- CPU bounded waves accept each result as its own transaction at the safe wave
  boundary; the registry—not `results_local`—owns the payloads. The existing
  dense→bounded CPU recovery semantics are unchanged; its plan transition
  advances the registry generation.
- Cooperative stop is checked before each GPU unit and each CPU wave. A stop
  acknowledges `STOPPED_IN_MEMORY`, retains accepted work, publishes nothing
  and explicitly records `durable_stopped_resumable=false`.
- After successful publication the registry is disposed and a lightweight
  snapshot is recorded in `per_cell_diagnostics["r55_result_registry"]`
  (no payloads). On typed failures the snapshot is attached to the exception
  record.

### 1.3 Explicit non-claims

- No durable checkpoint, no restart after process exit, no `STOPPED_RESUMABLE`
  disk marker.
- No GUI wiring: the 4-second `terminate()` path is untouched (M6/T2).
- No CPU/GPU payload mixing. A conserved GPU run waits for M3/M4 policy; it is
  not assembled on CPU.
- No performance tuning. Full-payload fingerprinting costs real CPU time; it is
  bounded by corpus size and is not measured as a production budget here.

---

## 2. Acceptance mapping (evidence)

All commands run in the worktree with `PYTHONPATH` unset (tests insert `src`).
`HEAD` remains `231b597867b4d251afac6b767000c8a37e2cf3b0`; no commit.

| # | Requirement | Evidence |
|---|---|---|
| 1 | Real orchestration after ≥2 real cells preserves exact IDs and complete payload hashes through refusal/cleanup; sparse pending enumerable; no double count | `tests/test_zegrid_resume_m1_conservation.py::test_m1_refusal_preserves_accepted_payloads_and_sparse_pending_set` (real `run_zegrid_mode`, real FITS corpus, real canonical computations, CONTROL-ONLY device substitution). Asserts registry fingerprint == independently computed full fingerprint (core + patch + reference pin), accepted IDs, `expected = accepted + pending`, bytes accounted once, no FITS/coverage/manifest, balanced pipeline ledger, no cleanup failures. |
| 2 | Registry not retained by pipeline queue; ownership/release/disposal balanced; retained bytes honest | GPU consumer returns receipts only; `RunResultRegistry` weakref test proves `dispose()` releases the payload arrays; integration disposes the real registry and asserts one-time release; `units_released == units_produced`, zero leases, empty staging. |
| 3 | Out-of-order CPU success, empty holes, duplicate accept, stale generations/late callbacks, acceptance failure, cancellation/retry boundary; no partial record | `tests/test_zegrid_run_registry_m1.py` (10 tests) covers out-of-order acceptance, `EMPTY_VALIDATED` hole, idempotent + conflicting duplicate, stale generation, staging failure, disposition, bounded journal, stop channel (thread + spawn-context events). `test_m1_cooperative_stop_ack_preserves_accepted_cells` covers a requested stop at the real unit boundary. |
| 4 | No-pressure control unchanged science/coverage | `tools/zegrid_resume_m1/m1_no_pressure_control.py` run against baseline `git archive HEAD` and the M1 source: identical science SHA-256 `3772b6cd…6e00a79` and coverage SHA-256 `49ed57f7…80d9728`, same shapes `[3,801,763]` / `[801,763]`, manifest present in both. Real FITS + real assembly, per_cell gauge, no GPU. |
| 5 | Preserve M0 immutable evidence; do not weaken the old discard assertion; document changed expectation; bounded Windows command | M0 docs and `.a2a-reports` evidence untouched. The local M0 replay test now explicitly documents and asserts the M1 changed expectation (retention instead of discard, conserved refusal instead of CPU rerun); historical M0 discard facts remain in the M0 report/Windows witness. M1 Windows probe supplied under `tools/zegrid_resume_m1/`, labelled `WINDOWS_M1_CONTROL_ONLY_CPU` / `NOT_VERIFIED_INSTALLED_RUNTIME`, **NOT_RUN** here. |
| 6 | Focused tests incl. R55 GPU fallback / CPU recovery / assembly consumers; baseline failures attributed | New tests: 12 passed. M0 witness + harness: 32 passed (combined run before stop test addition; witness re-run separately 1 passed). Focused R55/CPU/assembly suite: 216 passed, 7 skipped, 2 failed — both failures were the old post-acceptance whole-batch fallback assertions; the K=0 intent of each test was restored by moving the injected refusal/OOM before the first accepted cell (documented in the tests), and both pass individually. No blanket suite. |
| 7 | Project state + M1 doc updated; do not self-accept | This document + `ZEGRID_RESILIENT_RUN_STATE.md`; status remains pending Junior/Nono. |

---

## 3. Changed expectations (explicit, not silent)

1. **M0 local live replay**
   `tests/test_zegrid_resume_m0_baseline_witness.py` previously asserted
   `discarded=true` for every completed GPU unit and terminal
   `AdmissionExhausted` after the whole-batch CPU rerun. Under M1 the same real
   path retains both accepted payloads and raises
   `_R55RunConservedResults`; the test was renamed and its docstring records the
   M0 historical facts separately. The immutable M0 evidence is unchanged.

2. **R55-E11 runtime VRAM fallback**
   `test_e11_no_admissible_tile_falls_back_once_and_preserves_decision` now
   refuses the first GPU unit (`50 MiB` free at the first consume) so it keeps
   testing the K=0 whole-batch CPU fallback + preserved GPU decision. A
   post-acceptance VRAM refusal is covered by the M1 conservation test (host-RAM
   variant) and shares the same fallback handler.

3. **R55-E2 CuPy OOM**
   `test_gpu_oom_falls_back_to_exact_whole_batch_cpu` now raises the simulated
   OOM on the first GPU unit, preserving the K=0 exact CPU-rerun contract. A
   post-acceptance OOM follows the conserved-refusal path (not re-run on CPU).

No assertion was relaxed: the replaced assertions are stronger for M1
(retained payload hashes, no CPU reuse) or preserve the same K=0 contract.

---

## 4. Exact commands and results

Registry + conservation + stop (12 passed):

```
python3 -m pytest -q -p no:cacheprovider --tb=short --no-header \
  tests/test_zegrid_run_registry_m1.py \
  tests/test_zegrid_resume_m1_conservation.py
→ 12 passed, 30 warnings in 21.31s
```

M1 + M0 witness/harness (32 passed; stop test was added afterwards):

```
python3 -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_zegrid_run_registry_m1.py \
  tests/test_zegrid_resume_m1_conservation.py \
  tests/test_zegrid_resume_m0_baseline_witness.py \
  tests/test_zegrid_resume_m0_windows_harness.py
→ 32 passed, 54 warnings in 32.71s
```

Focused R55/CPU/assembly consumers (before the two K=0 test adjustments):

```
python3 -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_zegrid_gpu_admission_fallback.py \
  tests/test_zegrid_r55e11_runtime_vram_adaptation.py \
  tests/test_zegrid_r55e2_active_gpu_prefetch.py \
  tests/test_zegrid_r55e1_run_checkpoints.py \
  tests/test_zegrid_r55e9_dynamic_gpu_tile_ladder.py \
  tests/test_zegrid_r52_active_ooc_fallback.py \
  tests/test_zegrid_r3_executor_assembly.py \
  tests/test_zegrid_r34d_cell_feather.py \
  tests/test_zegrid_r34c_cell_offset.py
→ 216 passed, 7 skipped, 2 failed in 408.80s
```

The two failures were the changed post-acceptance fallback expectations; after
the documented K=0 injection adjustment (refusal/OOM before the first accepted
cell), both pass together:

```
python3 -m pytest -q -p no:cacheprovider --tb=short --no-header \
  tests/test_zegrid_r55e11_runtime_vram_adaptation.py::test_e11_no_admissible_tile_falls_back_once_and_preserves_decision \
  tests/test_zegrid_r55e2_active_gpu_prefetch.py::test_gpu_oom_falls_back_to_exact_whole_batch_cpu
→ 2 passed, 99 warnings in 42.60s
```

No-pressure identity control (real FITS/assembly, per_cell, CPU):

```
python3 tools/zegrid_resume_m1/m1_no_pressure_control.py \
  --src-root /tmp/zemosaic_m1_baseline/src --output-root /tmp/m1_np_baseline \
  --json-out /tmp/m1_np_baseline.json
→ ok=True, science 3772b6cd…6e00a79, coverage 49ed57f7…80d9728

python3 tools/zegrid_resume_m1/m1_no_pressure_control.py \
  --src-root src --output-root /tmp/m1_np_candidate \
  --json-out /tmp/m1_np_candidate.json
→ ok=True, identical science/coverage SHA-256 and shapes
```

Baseline tree: `git archive HEAD | tar -x -C /tmp/zemosaic_m1_baseline`
(no network, no installed runtime).

---

## 5. Windows check (NOT_RUN here)

Machine-readable command:

```
powershell -ExecutionPolicy Bypass -File `
  <worktree>\tools\zegrid_resume_m1\run_m1_windows_probe.ps1 `
  -SourceRoot <worktree> `
  -OutputRoot C:\Users\JarvisJunior\Desktop\zgprobe\m1_probe
```

Requirements: Python 3.11+ with `pytest`, `numpy`, `astropy`, `psutil`; no
CuPy required. Outputs `probe_summary.json` + `pytest.log`, max 900 s, exit
codes 0 PASS / 2 NOT_RUN / 3 FAIL / 4 TIMEOUT. It runs only the explicit source
tree, verifies the product/registry module hashes are unchanged during the
probe, and never edits the installed runtime. If compared to an installed
baseline, label that `NOT_VERIFIED_INSTALLED_RUNTIME`.

Status on this Linux host: **NOT_RUN** (no Windows). No GPU qualification is
implied: the probe is `WINDOWS_M1_CONTROL_ONLY_CPU`.

---

## 6. Limitations / unresolved

- In-memory only. `dispose()`/process exit releases the payloads; no durable
  resume exists.
- The stop channel is implemented and honoured at GPU-unit and CPU-wave
  boundaries, but the GUI/worker wiring (no 4 s terminate for a clean stop) and
  the durable `STOPPED_RESUMABLE` marker remain M2–M6 work.
- CPU wave results are accepted as independent transactions at the wave
  boundary (pmap returns the wave), not streamed as each worker finishes. The
  registry itself accepts arbitrary order (unit-tested); true per-cell CPU
  streaming is not claimed.
- Assembly still receives a transient ordered `results` list from the registry.
  This is a consumer view, not a second owner; it is dropped after publication.
- Full-payload hashing cost is unmeasured. It is required for conservation
  evidence and is not optimised in M1.
- `AdmissionExhausted` structured attributes remain absent (M0 known limit).
- Windows probe is NOT_RUN; no Windows file-lock/spawn claim for M1.
- CPU/GPU mixing remains forbidden without a qualified numeric-parity contract.

---

## 7. Next step

Junior independently verifies the diff/tests; Nono reviews the registry
transaction, ownership and fallback semantics. If accepted, M2 adds the durable
directory-per-run/cell store on top of this registry (strict JSON + separate
array files + atomic cell commit marker), then M3 adds the wait/replan loop and
M4 process restart. No commit/push is authorised by M1.

---

## 8. Rework-1 corrections (supersedes the r0 evidence numbers)

Nono F1–F5 were confirmed and corrected; the Junior disposition constrained F4
to the stop-channel transport only (the registry stays coordinator/process-local
and is not serialized).

### 8.1 Accepted-state contracts

- `accept()` now **validates the real producer schema before accounting**:
  COMPLETE requires all core planes, consistent spatial shapes (2-D or the real
  3-D colour planes), boolean masks, integer survivor counts and a non-empty
  reference pin; `per_cell` additionally requires the pre-offset/pre-feather
  patch with halo and geometry consistent with `record.patch_bounds`
  (x0,y0,x1,y1) and `patch.bounds` (y0,y1,x0,x1). Global patch stays optional.
- `EMPTY_VALIDATED` has its own schema: record with zero contributors and patch
  geometry, no core/patch planes. Legitimate empty holes are preserved.
- `status` is in the fingerprint and duplicate identity. Identical
  payload/status/attempt is idempotent and counts once; any contradictory
  duplicate raises `DuplicateCellError` and leaves the first record untouched.
- Generation is **required and captured at dispatch**: `gen != current` is
  rejected (`StaleGenerationError` / `FutureGenerationError`) with no mutation.
  `advance_generation()` preserves previously accepted payloads.
- Validation/fingerprint/accounting failures happen before the insertion lock
  and leave counts/bytes/pending unchanged.

### 8.2 Honest retained-buffer accounting

- Two explicit counters: `logical_bytes` (sum of plane `nbytes`) and
  `retained_bytes` (unique backing allocations).
- `_root_buffer` walks the `.base` chain and counts the ROOT allocation once:
  slices/non-contiguous views, core↔halo aliases and cross-cell aliases are
  deduplicated; the temporary contiguous copy used for hashing is never
  counted. `bytes`/`bytearray`/`memoryview` bases (real producer arrays) are
  supported and counted by length; `np.memmap` is counted conservatively and
  labelled `memmap_bytes`; unsupported owners fail closed.
- The manifest snapshot publishes both counters with an explicit note that
  `retained_bytes` is not added to already-measured parent RSS. No budget or
  floor changed.

### 8.3 Spawn-safe stop

- `CooperativeStopChannel` remains the same-process thread channel and now
  fails pickling with a clear `TypeError` (it is not a cross-process object).
- New `SpawnStopChannel.create(ctx)` holds ONLY multiprocessing primitives
  (events, shared fixed-size metadata, a multiprocessing lock). A REAL
  `ctx.Process` spawn test covers parent request → child observes → child
  acknowledges → parent observes, plus the no-ack timeout and idempotence.
- The registry itself is explicitly process-local (`__reduce__` refuses with a
  clear message); typed exceptions carry the local handle only. GUI transport
  and durable STOPPED_RESUMABLE remain deferred.

### 8.4 Failure ownership / classification / progress

- CPU acceptance failure after K accepted cells now attaches the registry
  handle (`exc.result_registry`) in addition to the snapshot, and clears the
  transient wave list so a traceback cannot retain unaccepted payloads.
- GPU `K>0` conservation distinguishes `resource_shortage`
  (host/VRAM/disk/OOM), `acceptance_failure` (internal schema/registry) and
  `measurement_failure`; requested stop remains its own typed path. `K=0`
  whole-batch CPU fallback is unchanged and still covered.
- GPU progress/completion increments only after a successful **non-idempotent**
  acceptance; stale/future/duplicate callbacks do not add completion. The GPU
  attempt/generation is captured at dispatch.
- New real-orchestration tests cover GPU `K>0` OOM, GPU `K>0` internal
  acceptance failure, and CPU acceptance failure after K with explicit disposal.

### 8.5 Bench boundaries

- `tools/zegrid_resume_m1/m1_no_pressure_control.py` no longer pins 16 GiB: it
  uses the LIVE `available_memory_bytes()` and records
  `budget_source=live_observed`, `synthetic_inflation=false`.
- The Windows probe is now `m1_windows_probe.py` (called by the PowerShell
  wrapper): NEW absolute output strictly under the authorised `zgprobe` root,
  existing/symlink paths refused, `pytest --basetemp` inside the output,
  `PYTHONDONTWRITEBYTECODE=1`, explicit `--python-exe`, bounded timeout and
  live artifact-cap watcher, source-module hashes before/after and an import
  provenance check that the modules come from the source tree. Missing deps,
  nonzero child, timeout or cap can never yield PASS. It labels itself
  `WINDOWS_M1_CONTROL_ONLY_CPU`; Windows real GPU remains NOT_RUN.

### 8.6 Final rework-1 evidence

```
# M1 registry + real conservation + probe guards + M0 preservation
→ 52 passed, 108 warnings in 64.18s

# R55-E1 CPU checkpoint/refusal file (previously 8 failures)
→ 42 passed, 82 warnings in 144.95s

# Focused GPU fallback / CPU recovery / assembly consumers (final)
→ 218 passed, 7 skipped, 596 warnings in 364.16s
```

No-pressure live-memory A/B:

```
observed_available_bytes = 559,968,256
budget_source = live_observed, synthetic_inflation = false
status = NOT_RUN (LayoutInfeasible: 2x2 needs 781.1 MiB worst-cell bound;
budget 441.6 MiB)
```

This is an honest NOT_RUN on a small host, not a science comparison. Required
bench step: run `m1_no_pressure_control.py` against the baseline `git archive`
tree and this source on a host whose live available RAM admits the 2x2 corpus
(or after qualifying a smaller corpus), and require identical non-null
science/coverage SHA-256.

Windows probe remains NOT_RUN locally; the command is
`run_m1_windows_probe.ps1` with an explicit venv Python and a NEW output under
`C:\Users\JarvisJunior\Desktop\zgprobe`.

### 8.7 Residual limitations

- No durable store/restart; no GUI stop wiring; no CPU/GPU mixing.
- CPU wave acceptance is transactional at the wave boundary (pmap returns the
  wave), not per-worker streaming.
- Full-payload hashing cost is unmeasured (performance is not an M1 goal).
- The no-pressure A/B and Windows probe are NOT_RUN on this host; Junior must
  replay them on a suitable host/runtime.

---

## 9. Rework-2 corrections (2026-10-27)

1. **Zero-support EMPTY_VALIDATED preserved.** The real producer returns an
   empty result for a POPULATED cell with `cell_fixed=None` and
   `gauge_status="empty"` (`n_contributors > 0`), distinct from the geometric
   `n_contributors == 0` hole. The registry now accepts the former ONLY with
   the `gauge_status="empty"` evidence, preserves the real contributor count in
   metadata, and still rejects arbitrary populated empties and never rewrites
   counts. Producer evidence: `test_zegrid_r53b...zero_support...` and
   `test_zegrid_r34b_cell_photometry.py` (20 passed / 1 skipped).
2. **True buffer-owner accounting.** `_root_buffer` now follows memmap slices to
   the underlying mapping (counted once) and follows `memoryview.obj` to the
   real owner; a 4 MB mapping/bytearray viewed as 60/240 bytes is reported as
   the full mapping once, not the view. `memmap_bytes` is a separate mapping
   extent label; `retained_bytes` is a unique-allocation bound, never claimed as
   resident RSS. Independent views over the same owner/mapping deduplicate; the
   hash copy is never counted. Unsupported owners still fail closed.
3. **Rejected probe output is never written.** `probe_summary.json` is written
   only after the output path was validated AND newly created by this invocation.
   Existing/outside/symlink paths are stdout-only; guard tests assert the
   pre-existing filesystem content is byte-identical.
4. **Artifact cap is enforced live.** `run_bounded` polls a size watcher and the
   deadline, kills the owned process tree immediately on either, closes Popen
   streams deterministically and records the terminal outcome. Preflight phases
   share the same deadline. Short real-subprocess tests prove cap and timeout
   termination (not post-hoc labels).
5. **Dispatch generation captured before the wave.** `wave_generation` is read
   before `_run_cell_batch`; a generation transition while the wave is in flight
   cannot stamp old results with the new generation. A deterministic pmap seam
   test advances the real registry during the second wave: the old-wave result is
   rejected as stale, the first wave's payload is preserved, and the generation
   is not rewritten.
6. **New M1 execution clamped to live RAM.** The M1 conservation tests clamp the
   synthetic trajectory to `min(live, env live)`; they never fabricate capacity.
   If the live environment cannot exercise the injected refusal, the case skips
   explicitly (`conservation NOT_RUN`) instead of passing. Registry unit tests
   still run on a low-memory host but are NOT conservation qualification. The
   Windows probe sets `ZEMOSAIC_M1_LIVE_AVAILABLE_BYTES`, requires the five
   conservation node IDs to be PASSED (not skipped) and at least 20 registry
   tests passed; otherwise it returns NOT_RUN/FAIL. The historical M0 fixture and
   evidence are untouched.

### 9.1 Rework-2 evidence

```
# Registry units (23), including zero-support, memmap-slice and memoryview-slice
→ 23 passed in 2.63s

# Probe guards incl. rejected-path preservation and live cap/timeout kills
→ 10 passed in 3.04s

# Zero-support producer evidence
→ 20 passed, 1 skipped in 2.36s

# Combined M1/M0 preservation (conservation cases skipped honestly on this host)
→ 54 passed, 6 skipped in 21.24s

# Focused GPU/CPU/assembly regression after rework-2
→ 218 passed, 7 skipped in 378.31s
```

Live no-pressure control on this host:

```
candidate: status=PASS, observed_available_bytes=1,306,472,448,
           science=3772b6cd…6e00a79, coverage=49ed57f7…80d9728
baseline : status=NOT_RUN (845,000,704 B LayoutInfeasible; retry 825,798,656 B
           ModeInfeasible) → A/B NOT_RUN: the two live observations are not
           comparable; no inflation was used
```

The candidate standalone run is real FITS/assembly evidence; it is not an A/B
parity claim. Windows M1 remains NOT_RUN.

### 9.2 Residual limits after rework-2

- The K>0 conservation cases are NOT_RUN on this low-memory host; Junior must
  exercise them on a host that admits the 2x2 corpus (Windows probe or larger
  Linux). The test code paths are present but not locally executed.
- No durable store/restart, GUI stop wiring, CPU/GPU mixing or per-cell CPU
  streaming.
- Full-payload hashing cost remains unmeasured.

---

## 10. Rework-3 final proof-gate corrections (2026-10-27)

1. **No regression-to-skip.** `_require_expected(run, expected, context)` now:
   - skips ONLY when no injection was verifiably armed AND the terminal is a
     known pre-case resource/setup failure (or no terminal);
   - FAILS for any unexpected exception (`RuntimeError`, `TypeError`, …) even
     when not armed;
   - FAILS when a success follows a verifiably armed injection.
   The run records `hits` (`memory_seam`, `stop`, `gpu_oom`, `accept_fail`,
   `generation`), accepted-cell count and observation samples as evidence.
   Low-memory-independent negative tests cover unexpected exception, armed
   success and pre-case NOT_RUN.
2. **Strict terminal precedence.** The probe now uses a pure
   `classify_probe_outcome`:
   `CAP_EXCEEDED/TIMEOUT → nonzero return → printed failed/errors → skipped →
   missing exact required case/unit floor → PASS`. A zero exit with a printed
   FAILED is contradictory and FAILS; skips are NOT_RUN only after no failure.
   Required identities are the exact six node IDs (a substring/superset does
   not satisfy them). Guard matrices cover failed+skip, error+skip, nonzero all
   PASSED, zero+failed, required absent, mandatory skipped, cap and timeout.
3. **Six deterministic required cases.** The mid-wave generation case is in
   `REQUIRED_CONSERVATION` (six total). It uses a TEST-ONLY bounded wave size
   (two cells per wave) so a high-RAM host cannot collapse it to one wave; it
   asserts the first wave is accepted, the old in-flight result is rejected as
   stale, and no output is published. Product workers/budget/scheduling are
   unchanged.
4. **Fresh no-inflation RAM observation.** Every actual
   `available_memory_bytes` seam re-reads live RAM through
   `_ClampedObservation` and returns
   `min(fresh, start/env cap, requested trajectory)`. Pressure can only lower
   it. Invalid/zero/negative/bool readings raise (fail closed). Deterministic
   observation-sequence tests cover a falling-RAM sequence and invalid reads.
   The M0 historical fixture/evidence is untouched.

### 10.1 Rework-3 evidence

```
# Probe guards + registry units + conservation-file unit helpers
→ 38 passed, 6 skipped in 3.36s

# Conservation file skip detail:
# 3 unit helpers passed; 6 real-orchestration cases SKIPPED because
# live RAM 1,218,064,384 < 1,258,291,200 (1200 MiB)
→ 3 passed, 6 skipped in 1.76s

# M0 preservation + zero-support producer evidence
→ 41 passed, 1 skipped in 20.51s

# Focused GPU/CPU/assembly regression (product unchanged in rework-3)
→ 218 passed, 7 skipped in 378.31s (rework-2 run)
```

The M1 conservation six-case gate is **NOT_RUN on this host** for capacity, not
because of an implementation failure. The guard/unit layer is green; the
Windows/probe gate requires the six exact cases to be PASSED, not skipped.

No-pressure A/B remains **NOT_RUN**: the candidate standalone live run passed in
rework-2 (science `3772b6cd…`, coverage `49ed57f7…` at 1,306,472,448 B), while
the baseline live runs returned NOT_RUN at 845,000,704 B / 825,798,656 B (not
comparable, no inflation). A standalone candidate PASS is not an A/B claim.

Windows M1 remains NOT_RUN. The known adequate-host Python is
`C:\Users\JarvisJunior\projects\zemosaic\.venv\Scripts\python.exe`.

### 10.2 Remaining limitations

- The six real-orchestration conservation cases need an adequate-memory host
  (Windows bench or larger Linux); they are NOT_RUN here.
- No durable store/restart, GUI stop wiring, CPU/GPU mixing or per-worker CPU
  streaming.
- Full-payload hashing cost remains unmeasured.
- A/B science/coverage parity is not measured; only the candidate standalone
  no-pressure run is recorded.



