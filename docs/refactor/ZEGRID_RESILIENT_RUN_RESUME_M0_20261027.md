# ZeGrid resilient run/resume — M0 contracts and baseline witness

Date: 2026-10-27
Mission: `ZM-ZEGRID-RESILIENT-RUN-RESUME-M0-20261027`
Phase: implementation / M0
Baseline: `231b597867b4d251afac6b767000c8a37e2cf3b0` (`origin/beta`)
Worktree: `worktrees/zemosaic-resilient-run-resume`, branch
`feat/zegrid-resilient-run-resume`
Status: **PARTIAL / DONE-for-M0** — baseline defect reproduced in a labelled
CONTROL-ONLY witness and on the real product path up to GPU readiness; the
real-GPU Windows witness is prepared but **NOT_RUN** on this Linux host (no
CuPy). Contracts are pinned to source symbols; two decisions remain for Junior.

This document is the M0 contract deliverable. It does **not** claim that
resume exists, that the baseline failure is fixed, or that Windows/GPU science
is qualified.

---

## 1. What M0 actually proved

### 1.1 CONTROL-ONLY witness (local Linux, real orchestration + real CPU science)

`tests/test_zegrid_resume_m0_baseline_witness.py` drives the REAL product entry
`run_zegrid_mode` → `_run_single` with:

* a real deterministic 8 × 512 px FITS corpus (2×2 pinned layout, halo 8,
  **real** `auto_layout.ScientificFloors`, no floor lowered);
* the REAL dense/bounded GPU plan, admission, bounded producer/consumer,
  leases, `_R55GpuFallback`, cache cleanup and whole-batch CPU re-admission;
* REAL canonical cell computations, but the **GPU device executor is replaced
  by the CPU canonical executor**. This is labelled `CONTROL_ONLY_FAKE_GPU`.
  It is not a GPU qualification and cannot certify GPU science.

Observed on the first run and reproduced identically on a second run:

| Fact | Value |
|---|---|
| Completed cells before pressure | 2 (`r0000c0000`, `r0000c0001`) |
| Cell result identity | core SHA-256 stable across runs |
| Pressure injection | `available_memory_bytes` 16 GiB → 700 MiB after cell 2 |
| GPU pipeline result | `_R55GpuFallback`, `R55_REFUSE_HOST_RAM` |
| Tile / prefetch | tile 409 px (whole declared ladder walked), prefetch 0 |
| Completed units | `status=completed`, **`discarded=true`** |
| Released units | `units_released == units_produced`, ledger balanced, zero leases |
| Publication | no `mosaic_grid.fits`, no coverage, no `zegrid_manifest.json` |
| CPU re-admission | `AdmissionExhausted`: `aggregate 843255808 > B_admit 660602880` |
| OOM/MemoryError | none |
| Stage reached | `cpu_readmission_failure` |

The artifact is
`.a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M0-20261027.artifacts/baseline_witness_control_only.json`.
`baseline_failure_reproduced=true`, `product_success=false`.

### 1.2 Real product path / Windows

The same harness was validated locally only up to
`gpu_readiness` and correctly returns **NOT_RUN** because CuPy is absent:

`tools/zegrid_resume_m0/m0_windows_witness.py` + PowerShell launcher
`run_m0_windows_witness.ps1`.

No actual GPU pixel was computed in M0 on this host. There is **no GPU proof**
and no claim of one. Junior must run the Windows harness after inspecting it;
its output is labelled `WINDOWS_REAL_GPU` / `WINDOWS_NO_GPU` and separates
`EXPECTED_BASELINE_FAILURE` from `UNEXPECTED_FAILURE`.

---

## 2. Producer / consumer inventory (baseline symbols)

### 2.1 Run preparation

| Producer | Artifact / field | Consumers |
|---|---|---|
| `_scan` / inventory (product entry) | ordered `frames_info`, logical IDs | geometry, gauge, manifest |
| `geometry.read_manifest` / `build_canvas` | `FrameDescriptor`, `GlobalCanvas` (`canvas_id`, WCS, shape) | layout, cache build, gauge, assembly |
| `auto_layout` + `scheduler` / `resource_envelope` | `layout`, `PlanCandidate`, `PlanContext`, envelopes | `_build_execution_bundle`, GPU/CPU admission, manifest |
| `_resolve_normalization` / science config | effective science tokens (`normalization`, weighting, rejection, combine, taper, sigmas, backend) | every cell stack, gauge, manifest |
| `photometric.compute_global_gauge` | `FixedNormalization` + `frame_ids` | every cell in `global`; auto→per_cell decision |
| `_r55_active_decide` / `_r55_active_admit` | `decision`, `candidate`, `mapping` | GPU pipeline, CPU waves, telemetry |
| `_r55_active_gpu_decide` / `_r55_active_gpu_admit` | GPU `decision` (`gpu_tile_ladder`, `gpu_host_bound_by_tile`, `device_peak_bytes_by_tile`, `qualification`, `facts`) | GPU pipeline, manifest provenance |

### 2.2 Cell task and payload

`_build_execution_bundle` emits `cell_tasks` as
`(cell, patch, mem, cache_dir, mode, bound[, storage_profile])`. Identity is
`cell.cell_id = f"r{row:04d}c{col:04d}"`.

| Producer | Payload | Consumers |
|---|---|---|
| `_prepare_stack_cell` | `cell`, `patch`, `mem`, `cache_dir`, `mode`, `bound`, `patch_bounds`, `patch_shape_hw`, `manifest`, `cell_frame_ids`, `cell_fixed`, `cell_gauge_mode`, `gauge_status`, `gauge_detail`, optional `storage_profile` | `_consume_stack_cell`, pipeline producer/consumer, lease ledger |
| `_build_one_cell_cache` | cache manifest `n_frames`, `total_bytes` | lease accounting, result `total_bytes` |
| `_resolve_cell_normalization` / `photometric.compute_cell_gauge` | `FixedNormalization` (per cell in per_cell), `CellGaugeResult` pin (`reference_frame_id`, `reference_support`, `total_valid_support`, `active_frames`) | `_run_cell_stream`, result record, offset graph |

### 2.3 Cell result

`_consume_stack_cell` returns:

```
cell_id, status, core{science_core, estimator_weight_sum_core,
support_w1_core, support_w2_core, n_eff_support_core, valid_mask_core,
surviving_sample_count_core}, record{...}, backend_used, n_frames,
total_bytes, cache_build_s, stack_s, peak_rss_kib, cleanup_failed,
reproject, cache_dir[, storage_profile][, patch]
```

`record` carries `cell_id`, `row`, `col`, `status`, `mode`, contributor counts,
`patch_bounds`, `patch_shape_hw`, `cell_gauge_mode`, `backend_used`,
`reference_frame_id`, `reference_frame_role`,
`bookkeeping_reference_frame_id`, `excluded`, `bound_bytes`, and storage info.

`patch` (per_cell only) is the pre-offset/pre-feather
`{bounds, core_slice, science, valid_mask, surviving}` payload. **This is the
checkpoint payload for M2**; `core` alone is insufficient for per_cell.

Consumers of `core`: `mosaic.assemble_canvas` (global) and, for per_cell, the
patch path. Consumer of `patch`: `_assemble_per_cell_feathered`, which builds
`cell_feather.CellPatch` and then `build_cell_offset_graph` +
`assemble_canvas_feathered`. Offsets are solved from the **complete validated
patch set**; a partial set is never an offset authority.

### 2.4 Pipeline ownership / leases

`BoundedPipeline` (`core/zegrid/parallel.py`) owns one producer thread and the
caller-owned consumer. It records `units_total`, `units_produced`,
`units_consumed`, `units_released`, queue depth, `ledger`
(`host/device/disk` budgets, used/peak, leases, reconciliations,
`balanced`), `cancelled`, `cancel_reason`, `cleanup_failures`,
`threads_joined`. A payload is released exactly once; credits are returned
after `_release` (cache removal), not merely after a state change.

### 2.5 Science/assembly/finishing

| Producer | Artifact | Consumer |
|---|---|---|
| `cell_feather.build_cell_offset_graph` | `CellFeatherResult` (`anchor`, `offsets`, `cell_ids`, `channels`, `edges_*`, `disconnected`, `policy`, `per_cell_reference`) | `assemble_canvas_feathered`, manifest `cell_feather` |
| `cell_feather.assemble_canvas_feathered` | `AssembledCanvas` science/coverage/support planes | FITS writer, finishing, manifest |
| `final_mosaic_finishing` | final science/coverage/uint16 outputs and info | publication |
| manifest writer (`zegrid_manifest.json`, `zemosaic-zegrid-r7-manifest-v1`) | layout, gauge provenance, storage, cell feather, per-cell records, GPU block | GUI/CLI, audit, M2 resume provenance |

Assembly allocates at least `num` + `wsum` as float64 `(H, W, 3)` plus
`stack_depth`/`owner` int32 `(H, W)`, `support_w1`, `support_w2`, `n_eff`
float64 `(H, W)`, and `science` float64 `(H, W, 3)`; finishing converts to
float64/uint16. These are **not** covered by the per-cell stack envelope and
must be preflighted in M5.

---

## 3. Checkpoint and resume contract (M1–M4 target)

### 3.1 Run-level fields (all required)

| Field | Why / consumer |
|---|---|
| `schema_version`, `science_contract_version` | compatibility gate before reuse |
| `run_id` (stable, not output-path identity), `run_generation` | run selection, append-only journal |
| `inventory` (ordered logical IDs, content SHA-256, sizes, mtimes as evidence only) | refuse silent input changes; freeze set |
| `canvas` (WCS, shape, `canvas_id`, resolution) | geometry restore |
| `layout` (`nx`, `ny`, origin, halo, halo provenance) | exact cell membership/offset graph |
| `effective_science` (normalization/weighting/rejection/combine/taper/sigmas/winsor/backend/dtypes) | re-run only identical science |
| `gauge` (full atomic `FixedNormalization`: reference index + `reference_frame_id`, coefficients, `norm_active`, weights, `weight_active`, exclusions, covered `frame_ids`) | no re-gauge, no re-anchor from remaining cells |
| `gauge_mode` decision (`auto` result, global/per_cell) | assembly dispatch |
| `plan_generations` (accepted mapping per wave/phase, tile/prefetch/K/workers/backend, shrink history) | restart continues the effective plan, no silent dense reset |
| `cell_status_index` (sparse `cell_id → COMPLETE/EMPTY_VALIDATED/...`) | remaining work = expected − complete − empty |
| `publication_state` and `final_generation` | never present partial science as complete |
| `event_journal` (bounded transitions, pauses, retries, reasons) | audit and provenance |
| `cache_policy` (disposable projections vs durable results) | never delete durable results as "cache" |

### 3.2 Cell-level fields

| Field | Why / consumer |
|---|---|
| `cell_id`, `row`, `col`, `status` | identity; sparse completion |
| `bounds` (global patch) and `core_slice` | placement, offset graph, assembly |
| `contributors` (logical IDs + order) | reproducibility; frame-order checks |
| `core_slice` planes: `science_core`, `estimator_weight_sum_core`, `support_w1_core`, `support_w2_core`, `n_eff_support_core`, `valid_mask_core`, `surviving_sample_count_core` | global assembly |
| `patch` (per_cell): `bounds`, `core_slice`, `science`, `valid_mask`, `surviving` **pre-offset/pre-feather, with halo** | per_cell offset/feather assembly |
| `cell_reference_pin`: `reference_frame_id`, `reference_support`, `total_valid_support`, `active_frames` | per_cell jauge stability; never re-derive from remaining cells |
| `exclusions` and provenance (`backend_used`, gauge status/detail) | audit; no silent backend mixing |
| `dtype`, `shape`, `artifact_sha256`, `bytes` | integrity, corruption invalidation |
| `attempt_id`, `plan_generation` | idempotent commit, late-result rejection |
| `storage_profile` and storage counters | provider/sink compatibility |
| `commit_state` (temporary/data/commit/index) | crash recovery |

### 3.3 Generations, sparse sets, ownership

* Every task/payload carries `run_id`, `cell_id`, `attempt_id`,
  `plan_generation`. A result from an older generation is **never published**
  after a plan transition.
* Resume plans over the **set** of valid `cell_id`, not `0..K`; a CPU wave can
  finish out of order, and empty cells create holes. Completion is
  `expected == COMPLETE + EMPTY_VALIDATED`.
* One owner of a cell result buffer and one idempotent `release`; the pipeline
  keeps only a lightweight receipt after commit. No hidden duplicate retained
  in futures, queues, tracebacks, closures or NumPy views.
* Commit-before-release: `_consume_stack_cell` result → durable cell commit →
  lightweight receipt → release cache/credits. A failed commit must not be
  counted as complete.
* Index is reconstructible from cell commits; a lost index receipt must be
  resolved by scanning the cell commit (same `cell_id` + `attempt_id`) before
  any retry.

### 3.4 Run / cell state machines

Run: `NEW → PREPARING → RUNNING → ASSEMBLING → FINALIZING → COMPLETE`;
side states `WAITING_RESOURCES`, `STOPPED_RESUMABLE`, `FAILED_DIAGNOSTIC`.
Interrupted `PREPARING/STACKING/COMMITTING` becomes `PENDING` at restart;
`COMPLETE` only after commit verification.

Cell: `PENDING → PREPARING → STACKING → COMMITTING → COMPLETE`;
`EMPTY_VALIDATED`, `RETRY_PENDING`, `INVALID_CHECKPOINT`. Progress counters do
not increment before commit; display computed/saved/reused/empty separately.

### 3.5 Cancellation / IPC ack

Baseline `WorkerController.stop(graceful=True)` sends SIGTERM and then
`terminate()` after 4 s (GUI `zemosaic_gui_qt.py:678`). M1 must add a
cooperative command/ack channel independent of logs and spawn-safe on Windows:
GUI sends STOP → worker stops new admissions → finishes/aborts the current
transaction safely → writes `STOPPED_RESUMABLE` best-effort → acks
`STOPPED`/`COMPLETE`; GUI does **not** auto-terminate at 4 s for a clean stop.
Classic path keeps its behaviour. Forced stop is explicit and may lose only the
uncommitted attempt. A missing final marker never invalidates prior commits.

---

## 4. Resource envelope by phase (M0 preflight contract)

| Phase | Baseline owner | Current bound | M0 requirement |
|---|---|---|---|
| discovery / hashing | product entry | streaming decode once | hash incrementally; record I/O cost; no giant concatenation |
| geometry / layout | `auto_layout`, `scheduler`, `resource_envelope` | plan budget = 4/5 available | preflight; no post-hoc growth |
| global gauge | `photometric.compute_global_gauge` | bounded streaming phase-1, no disk cache | resumable atomic artifact; no per-cell-only control; cost displayed |
| auto→per_cell decision | `_run_single` | gauge decision | record exact decision and refs |
| cache build | `_prepare_stack_cell` / `_build_one_cell_cache` | per-cell cache bound | bounded/streamed; disposable |
| GPU stack | `_run_r55_active_gpu_units` | admitted host/VRAM/disk envelope + lease ledger | commit result before release; no discard on pressure |
| CPU waves | `_run_r55_active_cpu_waves` / `_bounded_run_recovery` | per-wave checkpoint, 9/10 admit | keep completed waves; sparse commits |
| checkpoint write | M2 store | new | reserve + temp + fsync + verify + atomic publish |
| assembly | `assemble_canvas_feathered` | `num` + `wsum` float64 + plans | bounded re-read or explicit wait/stop; no double offsets |
| finishing/publication | `final_mosaic_finishing` | float64/uint16 planes | generation + last marker; no mixed files |

`ram_contract` constants pinned: `PLAN_FACTOR=4/5`, `ADMIT_FACTOR=9/10`,
`PROCESS_BASELINE_BYTES=340 MiB`.

---

## 5. Test / scenario matrix

| ID | Scenario | Purpose | Cost target |
|---|---|---|---|
| T0 | transition model: high→low→high budget, invalid observation, real refusal, repeated cell, stale generation, sparse commits, STOP/WAIT, commit without ack | pure logic, no sleeps | seconds |
| T1 | small real end-to-end: 2×2 / valid halo / real FITS / real gauge / real assembly; global then auto→per_cell; one empty cell; overlaps; prefetch 0 then >0; pressure **after ≥2 commits** | product path, real science | few minutes |
| T2 | real process kill + Windows GUI: Stop and Save, close, kill during commit, relaunch, resume; prove cooperative ack is not pre-empted by 4 s terminate; CPU spawn and daemon-thread paths | cancel/durability | short |
| T3 | dimension stress: many cells/manifests, late results; large patch/canvas assembly allocations; deep contributor working set; RSS slope vs commits; long/spaced paths, Windows file contention | scaling, not small-N parity | bounded |
| T4 | Windows canary then M31 | only after T0–T3 green | long, gated |

T1 corpus: the M0 witness corpus (8 × 512 px, 2×2, halo 8, real floors) is a
valid seed but must be extended with an empty cell and a per_cell variant before
M2 acceptance. M31 replay validates the decision, not the real memory peaks.

### 5.1 First vertical slice (M1–M4 contract)

1. M1: real engine computes ≥2 cells; each success is registered/committed to
   an in-memory journal; a simulated pressure after K does **not** erase the K
   results; no double counting.
2. M2: same run with a durable cell store; kill at every write stage; restart
   recovers only complete commits; sparse set (including an empty cell) is
   respected; index reconstructed.
3. M3: pressure after K then RAM recovery; no re-computation of K; wait is
   light (no giant buffers/credits held); late results from cancelled
   generations rejected; AUTO→bounded is one-way and honors explicit profiles.
4. M4: new process after kill; restore gauge/identity; recompute only missing
   cells; assemble identical science; manifest publishes `resumed_from`,
   reused vs recomputed counts, per-cell backend and pause reasons.

No milestone is declared complete by this document. M0 acceptance remains
Junior verification + Nono review.

---

## 6. Decisions needed / blockers

1. **[DECISION — Junior]** Checkpoint storage format and layout: a strict
   JSON+`.npy`/raw-shard schema versus a single archive. M0 recommends
   directory-per-run + one directory per cell, strict JSON metadata, separate
   array files, atomic rename of the cell directory marker, no pickle. Need
   approval before M2.
2. **[DECISION — Junior]** CPU/GPU mixed resume: V1 contract says no mixing
   without a qualified numeric-parity test. If M2/M4 cannot qualify parity,
   resume must preserve GPU commits and wait for an admissible GPU rather than
   silently re-run on CPU. Confirm this conservative default.
3. **[BLOCKER — environment]** Real-GPU Windows witness is NOT_RUN here (no
   CuPy). It is prepared, not executed. No GPU or Windows claim is made.
4. **[UNCERTAINTY]** `AdmissionExhausted` does not expose `aggregate_bytes` /
   `budget_bytes` as typed attributes; M0 records the parsed message. M2 should
   attach structured refusal fields (already partly in `exc.record`).
5. **[UNCERTAINTY]** Assembly/finishing peak RAM is not currently budgeted as
   a first-class envelope row; M5 must measure it on the corpus before a large
   run. No number is invented here.
6. **[UNCERTAINTY]** The installed Windows runtime provenance must be supplied
   to the harness; the script refuses to run without it unless explicitly
   labelled unverified. Junior must confirm the module path/SHA.

---

## 7. Exact M0 evidence

Focused baseline tests (no full suite, no long benchmark):

```
python3 -m pytest -q -p no:cacheprovider \
  tests/test_zegrid_gpu_admission_fallback.py \
  tests/test_zegrid_r55e11_runtime_vram_adaptation.py \
  tests/test_zegrid_r55e2_active_gpu_prefetch.py \
  tests/test_zegrid_r55e1_run_checkpoints.py \
  tests/test_zegrid_r55e9_dynamic_gpu_tile_ladder.py
→ 145 passed, 488 warnings in 293.80s
```

M0 witness:

```
M0_ARTIFACT_DIR=... python3 -m pytest -q -p no:cacheprovider \
  tests/test_zegrid_resume_m0_baseline_witness.py
→ 1 passed (two deterministic runs)
```

Windows harness local NOT_RUN smoke:

```
python3 tools/zegrid_resume_m0/m0_windows_witness.py \
  --src-root src --output-root /tmp/m0win \
  --allow-outside-zgprobe --allow-unverified-provenance --max-seconds 60
→ status=NOT_RUN, stage=gpu_readiness, exit=2, new_artifacts=8,434,423 B
```

Git: branch `feat/zegrid-resilient-run-resume`, HEAD
`231b597867b4d251afac6b767000c8a37e2cf3b0`, no commit, no push.

## 8. Non-claims

* Resume is **not implemented**; M1–M4 are contracts only.
* The baseline defect is **reproduced**, not fixed.
* The local witness is control-only; it is not GPU science.
* The Windows witness is runnable but NOT_RUN; no actual GPU result exists.
* Scientific floors, RAM/VRAM budgets and the original project plan are
  unchanged. The original plan document outside this worktree is preserved.

---

## 9. Rework-1 (Nono review-0 findings F1–F7)

Status: implementation reworked; local tests run; Windows real GPU remains
**NOT_RUN** on this Linux host. No product source changed.

### 9.1 Fail-closed changes

| Finding | Change |
|---|---|
| F1 provenance | Parent **and** child enforce `installed_module_sha256 == actual imported module file sha256`; source/actual equality and key dependency hashes are recorded. Mismatch refuses unless `--allow-unverified-provenance`; the witness then carries `provenance_verified=false` and `qualification=NON_QUALIFYING_CONTROL`. |
| F2 parent verdict | Parent succeeds only with `returncode==0` **and** `witness.status=="PASS"` **and** `classification=="EXPECTED_BASELINE_FAILURE"` **and** no cap breach. Nonzero child exit is authoritative. |
| F3 PASS predicate | Requires ≥ `pressure_after` completed cells with `backend_used`, completed units `discarded=true`, refused unit `fallback_cpu`, zero publication, `cleanup_failures==[]`, balanced ledger with zero leases, no OOM, and verified provenance. `gpu_qualified` is true only for `WINDOWS_REAL_GPU` + real GPU readiness + `backend_used=="gpu"` + expected failure + verified provenance. |
| F4 memory/mode | `effective_high = min(16 GiB, live_available)`; both values and `synthetic_observation_capped_by_live_ram` are recorded. `mode` is derived (`LOCAL_CONTROL_CPU`, `WINDOWS_REAL_GPU`, `WINDOWS_NO_GPU`), never hardcoded. |
| F5 cap/stdout | 4 B/px corpus estimate plus support/cache/output/log margin; child stdout/stderr streamed to `child.log`; a live watcher kills the owned process tree on cap breach; `CAP_EXCEEDED` has exit 4 and can never be `PASS`. |
| F6 tests | New `tests/test_zegrid_resume_m0_windows_harness.py` covers provenance refusal/override/match, parent fail-closed, cap, timeout, NOT_RUN coherence, full witness predicate and GPU qualification. The local witness now asserts the **second run's** terminal outcome too. A no-pressure real-memory success control is **out of scope on this host**: live available RAM (~0.6–0.8 GiB) is below the product PLAN budget, so the run refuses before any cell; no `gpu_qualified` claim is made. |
| F7 docs | Added `tools/zegrid_resume_m0/README.md`; labels control-only CPU vs Windows real GPU; aligns the 1 GiB cap with the conservative estimate; explains tile 409 vs M31 tile 32 (small-corpus ladder/geometry, not M31 replay). |

### 9.2 Additional evidence

```
python3 -m pytest -q -p no:cacheprovider \
  tests/test_zegrid_resume_m0_windows_harness.py
→ 20 passed in 2.28s

python3 -m pytest -q -p no:cacheprovider \
  tests/test_zegrid_resume_m0_baseline_witness.py \
  tests/test_zegrid_resume_m0_windows_harness.py
→ 21 passed, 36 warnings in 19.18s
```

Local NOT_RUN smoke (no CuPy, live RAM below envelope):

```
python3 tools/zegrid_resume_m0/m0_windows_witness.py \
  --src-root src --output-root /tmp/m0win4 \
  --allow-outside-zgprobe --allow-unverified-provenance --max-seconds 60
→ exit 2, status=NOT_RUN, mode=LOCAL_CONTROL_CPU,
  gpu_qualified=False, provenance_verified=False,
  stage_reached=gpu_readiness, new_artifacts_bytes=3039,
  reason="live available RAM below the minimum witness envelope:
  1336205312 < 2147483648 bytes"
```

The Windows real-GPU run is still **NOT_RUN**. The harness is now fail-closed;
no Windows or GPU claim is made by this rework.

### 9.3 Unresolved / still pending

* Windows real-GPU execution and its provenance path must be run by Junior on
  the target host; the installed module path/SHA must be supplied.
* A real no-pressure success reference remains unmeasured on this host (insufficient
  live RAM). It must be established before any `gpu_qualified` claim on Windows.
* Architecture decisions from M0 r0 remain: strict JSON + separate numeric
  arrays + atomic marker (no pickle); no CPU/GPU mixing without qualified
  numeric parity.

---

## 10. M1 supersession note (local live replay only)

Added by `ZM-ZEGRID-RESILIENT-RUN-RESUME-M1-20261027`. The M0 facts and
artifacts above are historical evidence and are unchanged.

When the local M0 witness test is replayed against M1 working code, the old
post-refusal behaviour no longer holds by design: the two accepted GPU payloads
are retained by the run registry and the whole-batch CPU rerun is refused
(`R55_RUN_RESULTS_CONSERVED`) rather than reaching `AdmissionExhausted`. The
test file now documents that changed expectation and asserts retention; the
historical discard facts remain authoritative only for the M0 baseline and the
Windows witness artifact. M1 acceptance evidence is separate in
`ZEGRID_RESILIENT_RUN_RESUME_M1_20261027.md`.

The Windows real-GPU witness remains NOT_RUN on the Linux host and is not
replaced by any M1 result.
