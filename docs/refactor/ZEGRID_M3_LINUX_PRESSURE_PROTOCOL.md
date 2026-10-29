# ZEGRID M3 — Linux physical-pressure qualification protocol

Status: **PREPARED, NOT EXECUTED (pressure)**. This document is a reviewable
protocol for the M3 CPU wait→resume proof. It is **not** an M3 qualification and
contains no PASS claim. Physical pressure is explicitly out of scope for the
missions that wrote it (`ZM-ZEGRID-M3-LINUX-PREFLIGHT-20261028`,
`ZM-ZEGRID-M3-LINUX-RUNNER-20261029`).

Rework-2 (`ZM-ZEGRID-M3-LINUX-RUNNER-20261029`) build the two bricks §4 declared
missing (transparent observers, owned pressure runner), measured the bounded
minimum `X_b` from real no-pressure references, and re-ran the preflight. The
measured result is **NOT_READY** (`BOUNDED_MIN_FITS_AT_RESERVE`); see §9. The
pressure experiment itself remains **NOT_RUN**.

- Repository: `/home/tristan/.openclaw/workspace/worktrees/zemosaic-resilient-run-resume`
- Branch: `feat/zegrid-resilient-run-resume`
- HEAD at preparation: `a18dd12e4cc2775fc397d5bcfe9c334bbaffb205`
- Scope: Linux CPU path only. GPU, Windows/NTFS, cooperative GUI stop, full
  restart/new process, and M31 remain separate later gates.
- Evidence split:
  - **Code inspection** of the current tree (citations below).
  - **Executed** read-only preflight on the current host
    (`tools/zegrid_resume_m3/m3_linux_preflight.py`); result **NOT_READY**,
    quantified in §3.
  - No physical pressure, no product run, no store write, no cgroup/sysctl/swap/
    mount change was performed.

---

## 1. Real observation path on Linux

### 1.1 What the product actually observes

The product's M3 probe is built from the RUN checkpoint sampler:

- `src/zemosaic/zemosaic_zegrid_mode.py:330-332` — `available_memory_bytes()`
  returns `psutil.virtual_memory().available`.
- `src/zemosaic/zemosaic_zegrid_mode.py:8560-8565` — `_r55_run_checkpoint_sample()`
  calls `zrt.collect_run_facts(root=..., available_fn=available_memory_bytes)`.
- `src/zemosaic/zemosaic_zegrid_mode.py:9579-9591` — `_r55_m3_observation_from_facts()`
  projects `host.raw_available_bytes`, `sampled_monotonic_ns`, and `source` into
  the M3 observation contract.
- `src/zemosaic/zemosaic_zegrid_mode.py:11402-11408` — the M3 coordinator's
  `probe` lambda uses exactly that projection.
- `src/zemosaic/core/zegrid/runtime_control.py:400-470` — `collect_run_facts()`
  validates the raw value strictly (`minimum=1`), recomputes the exact 90 %
  budget, and refuses malformed facts.
- `src/zemosaic/core/zegrid/ram_contract.py:91-92,220-230` — integer factors:
  `B_plan = 4·A/5`, `B_admit = 9·A/10`.
- `src/zemosaic/core/zegrid/resource_recovery.py:200-280` — `parse_observation()`
  refuses missing/negative/from-the-future/stale observations (default max age
  30 s); a malformed live probe fails closed.
- `src/zemosaic/core/zegrid/resource_recovery.py:338-520` — `CapacityWaiter`
  spaces probes (5 s initial, 30 s max backoff) and requires **two admissible
  observations spaced by at least 5 s**; it sleeps on an event, not a busy loop.

On Linux, psutil's `available` is derived from `/proc/meminfo` `MemAvailable`.
This tool reads `/proc/meminfo` directly and optionally cross-checks psutil; it
does not substitute one for the other silently.

**Consequence:** pressure means *host-wide* `MemAvailable` reduction. It is not a
cgroup-local threshold and it is not a VRAM threshold.

### 1.2 Host vs cgroup

- `src/zemosaic/core/zegrid/runtime_control.py:318-340` — `read_cgroup_memory()`
  reads `/sys/fs/cgroup/memory.current` and `memory.peak` at the **root** only.
  It is diagnostic; it never changes the RUN budget.
- The preflight does **not** repeat that mistake: it walks the caller's cgroup
  chain from the leaf to the cgroup mount root and records every ancestor's
  `memory.max`. A finite limit at any ancestor is the effective limit. A missing
  `memory.max` at the mount root is accepted as the known normal cgroup v2
  shape; a missing/unreadable/malformed ancestor is **unknown**, never
  "unlimited". Evidence keeps each ancestor path and limit.
- On the current host the caller is in
  `/user.slice/.../openclaw-gateway.service`; every ancestor with a memory file
  says `max`, and the mount root has no memory-controller files. The caller's
  effective limit is therefore known-unlimited. The product's own root-only
  cgroup diagnostic still reports unavailable; that remains an observation gap.
- A **finite** effective limit makes host-level pressure and cgroup OOM
  different thresholds. The preflight fails closed with
  `CGROUP_LIMIT_NOT_MODELLED` (finite) or `CGROUP_LIMIT_UNKNOWN` (cannot
  establish the chain). It never assumes a cgroup limit is physical pressure or
  that the product sees it.
- No mount, cgroup, sysctl, swap or cache configuration changes are permitted.
  The preflight is read-only apart from an explicitly owned JSON output.

---

## 2. Deterministic bounded corpus and product path

### 2.1 Product entry

The real entry is the Python callable
`src/zemosaic/zemosaic_zegrid_mode.py:16394` `run_zegrid_mode(input_folder,
output_folder, ..., zconfig=..., workers=...)`. There is **no headless ZeGrid
CLI**: `run_zemosaic.py` / `python -m zemosaic` launch the Qt application
(`src/zemosaic/_app.py:282`). A protocol harness must call `run_zegrid_mode`
directly, exactly as accepted M1/M5-B harnesses do
(`tools/zegrid_resume_m1/m1_no_pressure_control.py:132-141`,
`tools/zegrid_resume_m4/m4_resume_harness.py`).

### 2.2 Activation gap (FINDING, not a product change)

`zegrid_resource_recovery_enabled` is read by
`getattr(zconfig, ..., False)` at
`src/zemosaic/zemosaic_zegrid_mode.py:11397`, but it is **absent from
`DEFAULT_CONFIG` / the GUI** (`src/zemosaic/zemosaic_config.py:98-150`; the M4
keys `zegrid_resume_mode` / `zegrid_durable_store_dir` are declared, the recovery
key is not). `zegrid_resource_recovery_deadline_s` and
`zegrid_durable_store_budget_bytes` are also undeclared
(`...:11803`, `...:2240`, `...:14021`).

`load_config()` preserves unknown JSON keys
(`src/zemosaic/zemosaic_config.py:970-980`) and the worker builds
`SimpleNamespace(**worker_config_cache)` (`src/zemosaic/zemosaic_worker.py:31106`),
so a hidden JSON key can reach the engine. That is still not a shipped,
validated user path. For this protocol the harness supplies a `zconfig`
namespace; Junior accepted this **ENGINE-only** scope explicitly. It is not a
GUI/CLI qualification. The missing product key remains a later task; no product
key is added here.

### 2.3 Corpus proposal (deterministic, bounded, real FITS)

Use a real FITS corpus with stable content and WCS:

- 8 frames, 512×512, float32, deterministic Gaussian + seeded noise, WCS TAN
  shifts as in `tools/zegrid_resume_m1/m1_no_pressure_control.py:33-72`.
- Grid `4×4` = 16 cells, `cell_gauge_mode="per_cell"`, product halo
  `zegrid_halo_px=64` (declared default in `src/zemosaic/zegrid_halo.py:50`;
  do not lower it to make the corpus cheaper).
- Fixed science settings: `sky_mean` normalization, `noise_variance` weighting,
  `kappa_sigma` rejection (3.0/3.0), winsor `0.05,0.05`, `mean` combine,
  no radial weight, no uint16 conversion, CPU only.
- `workers=1` is proposed to obtain several safe wave boundaries. This is a
  **proposal**: the first no-pressure reference must report
  `r55_run_waves` / `wave_count`; if fewer than 3 waves or fewer than 2 commits
  exist before the pressure point, enlarge the grid (e.g. `6×6`) rather than
  fabricate K or start pressure after the last wave.
- Product configuration for the experiment:
  - `zegrid_scheduler_mode="active"` (product default;
    `src/zemosaic/zemosaic_config.py:136-140`)
  - `storage_profile="auto"` — **required**: the M3 wait is only reached after
    the ONE `dense→bounded` retry
    (`src/zemosaic/zemosaic_zegrid_mode.py:2157-2163,11790-11810`).
  - `zegrid_durable_store_dir=<fresh>/store` (real ext4 path, not tmpfs)
  - `zegrid_resume_mode="never"` (fresh run; no accidental reuse)
  - `zegrid_resource_recovery_enabled=True` via `zconfig` (harness seam)
  - `zegrid_resource_recovery_deadline_s=900` (bounded; explicit)
  - `workers=1`, `use_gpu=False`, `cell_gauge_mode="per_cell"`,
    `zegrid_halo_px=64`.

### 2.4 Durable commit boundary

A cell becomes durable at the atomic rename of its commit manifest, after its
object files and manifest have been written:

- `src/zemosaic/core/zegrid/run_store.py:2156-2260` — `commit_cell()` validates
  the real producer schema, stages objects, checks idempotence by
  `(cell_id, attempt_id)`, and writes the commit last.
- `...:2337-2345` — `atomic_write_bytes(target=commits/<cell>.<attempt>.json)`;
  prior commits are untouched on failure.
- `...:2763-2800` — `commit_and_release()` commits **then** releases the
  registry's large arrays; release is never before the commit.
- `src/zemosaic/zemosaic_zegrid_mode.py:11980-12100` — the CPU wave path accepts
  each result, then calls the store hand-off; a store failure degrades to M1
  conservation, keeps the payload in RAM, and never creates a partial commit.

The store also has `run.json`, `objects/`, `index.json`, `LOCK`, and
`commits/*.json` (`run_store.py:1602-1690,1969-1990`). The index is
reconstructible and is not, by itself, proof of execution.

### 2.5 Reference processes

Two fresh, no-pressure references are required **before** pressure:

1. **Dense reference** — default `storage_profile` / no pressure. Provides the
   baseline science/coverage/offset digests and the real run duration.
2. **Bounded reference** — `storage_profile="bounded"`, same corpus, settings,
   binary and Python. Provides the **real bounded minimum aggregate** needed by
   the feasibility gate. This is a real product run, not a mock; it does not
   prove M3 and does not use pressure.

`X_b` must describe the plan the engine would actually build under the
**pressure-governing raw** `A-P`, not a comfortable high-`A` plan. A bounded
reference run on a large-memory host gives an upper bound and can falsely pass
or falsely fail condition 4. The protocol therefore requires either (a) a
bounded reference executed at a comparable low-`A` observation, or (b) the
scheduler's minimum-plan metadata for the frozen corpus/settings, produced
without synthetic inflation. If neither is available, the preflight remains
`NOT_READY` (`BOUNDED_AGGREGATE_NOT_MEASURED`) rather than guessing.

The experiment process must be a fresh process with its own output/store roots.
Do not reuse a reference output directory or a store from another run.

### 2.6 Separate product integration gap: already-bounded / repeated pressure

The M3 wait is nested only inside the ONE `dense→bounded` retry
(`src/zemosaic/zemosaic_zegrid_mode.py:2157-2163,11790-11810`). If the run is
already `bounded`, or the one fallback was already consumed, a host-RAM refusal
at a later wave terminalizes instead of entering `WAITING_RESOURCES`. This is a
**separate unqualified product integration gap**, not a reason to weaken or
reinterpret this protocol. Fixing it (e.g. a safe-boundary wait independent of
the storage fallback) is a later product task; no code change is made here and
full profile qualification is not claimed.

---

## 3. Feasibility and guard contract

### 3.1 Exact requirements

- `A` = live host `MemAvailable` observed by the preflight/product probe.
- `R` = **1 GiB live reserve** that must remain available at all times. The
  public preflight rejects `R < 1 GiB`; only a stricter (larger) reserve is
  allowed.
- `P` = the **selected** pressure dose, not automatically the maximum. The hard
  ceiling is `P_max = 512 MiB`; `P` must satisfy `0 < P <= P_max`. The public
  preflight rejects a selected dose or a cap above 512 MiB; only stricter
  (smaller) values are allowed.
- `X_b` = **bounded minimum aggregate** measured by the bounded reference
  (`bundle.envelope.aggregate_bytes` for the smallest admitted bounded plan)
  at the pressure-governing raw `A-P` (see §2.5; a high-`A` reference is not
  sufficient). A supplied number is not proof of provenance or of being the
  minimum; the measurement must be recorded and reviewable.
- `required_raw(X) = ceil(10·X/9)` — exact inverse of the RUN 90 % budget
  (`src/zemosaic/zemosaic_zegrid_mode.py:9570-9577`).
- The preflight also rejects malformed telemetry: `MemTotal` must be positive,
  `0 < MemAvailable <= MemTotal`, `0 <= SwapFree <= SwapTotal`, and required
  fields with an unknown unit are invalid rather than silently bytes.

M3 wait is reachable only if, after pressure:

1. `A - P >= R` (live reserve preserved), and
2. `required_raw(X_b) > A - P` (the bounded retry still does not fit, so the
   coordinator enters `WAITING_RESOURCES`), and
3. `required_raw(X_b) <= A` (after pressure release the same plan can be
   admitted), and
4. `required_raw(X_b) > R` (at the reserve floor the bounded minimum is still
   not satisfiable; otherwise it would fit and M3 would never wait).

If condition 4 fails, **no amount of pressure within the safe reserve can force
the wait**: the bounded plan fits at 0.9·R. Report `NOT_READY`
(`BOUNDED_MIN_FITS_AT_RESERVE`) rather than lowering `R`, changing the probe, or
shrinking the corpus science.

The aggregate admission window for any plan is:

```
low  = floor(9·(A-P)/10) + 1
high = floor(9·A/10)
```

A bounded `X_b` outside `[low, high]` cannot be both admitted before and refused
after pressure.

The safe **dose window** for a measured `X_b` is:

```
dose_low  = max(1, A - required_raw(X_b) + 1)
dose_high = min(P_max, A - R)
```

A non-empty `[dose_low, dose_high]` means a smaller-than-maximum pressure can be
selected safely. If the selected 512 MiB dose breaches the reserve but this
window is non-empty, report `NOT_READY` for that dose and recommend a dose in
the window; do **not** mandate moving host merely because the maximum was
selected. If the window is empty, no dose within the cap can force the bounded
minimum to refuse while preserving `R`: `NOT_READY` (`NO_SAFE_PRESSURE_DOSE`).
Never lower `R` to enlarge the window.

**Scenario arithmetic is not physical evidence.** The preflight's `A` is taken
before product imports/allocations; the product's governing checkpoint `A` at
the pressure boundary will differ. `A-P` is a planning inequality, not proof
that a pressure process of RSS `P` lowers host `MemAvailable` by exactly `P`:
swap, page cache, reclaim and concurrent host activity can absorb or amplify the
change. The experiment must use the **live boundary observation** and a
calibrated process-footprint measurement; if the actual boundary values do not
satisfy the inequalities, the result is `NOT_RUN`, not a retrofitted PASS.

### 3.2 Observed host result (2026-10-28, read-only)

Executed (rework-1, unique artifact):

```
/home/tristan/.openclaw/workspace/projects/zemosaic/.venv/bin/python \
  tools/zegrid_resume_m3/m3_linux_preflight.py \
  --json-out /home/tristan/.openclaw/workspace/.m3-linux-preflight-20261028/preflight.rework1.221808Z.json
```

Observed at the rework snapshot (`2026-10-28T22:18:08Z`):

| Fact | Value |
|---|---:|
| `MemTotal` | 7,697.3 MiB |
| `MemAvailable` | 1,694.9 MiB |
| `SwapTotal` / `SwapFree` | 32,273.0 / 28,469.9 MiB |
| `A - 512 MiB` | 1,182.9 MiB |
| Reserve shortfall at 512 MiB | **none at this snapshot** |
| Filesystem | `ext4` at `/`, 22,234.5 MiB free |
| cgroup | caller chain known-unlimited; each ancestor evidence recorded |
| Product probe | `psutil.available` → Linux `MemAvailable` (host-wide) |
| Bounded aggregate | **not measured** |
| Decision | **NOT_READY**, exit 2, sole host blocker `BOUNDED_AGGREGATE_NOT_MEASURED` |

This snapshot does **not** establish a reserve shortfall and does **not** mandate
a host move. Other snapshots in this session and Junior's independent run were
different (`MemAvailable` ~0.8 GiB, ~1.47 GiB, and 1,829.3 MiB respectively);
memory is volatile and each snapshot is preserved as an observation, never as a
reservation or a universal host fact. The bounded minimum remains unmeasured, so
the M3 wait reachability is still unproven.

The first preflight run (r0) exposed and fixed a tool defect: the longest
mount-prefix matcher rejected `/` (`target.startswith("//")`), producing
`MISSING_FILESYSTEM_TELEMETRY`. r1/r2 include that fix; r0 is retained as
pre-fix evidence. Rework-1 additionally fixed the cgroup ancestor walk,
contract enforcement and telemetry validation.

### 3.3 Proposed pressure process and guards

This is a **design**, not implemented in this mission. The preflight tool has no
allocator and never spawns pressure.

- Allocator: one owned process, one PID, anonymous `mmap` + page touch in
  bounded steps (e.g. 64 MiB), no children, no `madvise`/cgroup/sysctl changes.
- Hard ceiling: the **whole pressure process RSS** (payload + interpreter +
  allocator overhead) must stay `<= 512 MiB`. The payload target is therefore
  lower (e.g. `<= 480 MiB`), and the guard kills at `> 512 MiB` without a
  separate `+32 MiB` allowance. Any overshoot margin is inside the 512 MiB hard
  cap.
- Product tree ceiling: a calibrated ceiling derived from the no-pressure
  reference, e.g. `reference_peak_tree_rss + 64 MiB`, recorded and frozen before
  the experiment. If the product tree exceeds it, the guard stops pressure and
  classifies FAIL; it is not a silent allowance.
- Lifetime: `<=900 s` total, `<=300 s` in pressure, then mandatory release.
- Guard sampling: `<=1 s` interval (proposal 0.25 s), with a documented
  detection-latency/overshoot window. A sample-based guard is **best effort**:
  concurrent host changes or allocation between samples can defeat it. The
  kernel is the only hard boundary; the protocol does not claim a sampled RSS
  ceiling is a guarantee.
- Guard order:
  1. If pressure process RSS `> 512 MiB`, or `MemAvailable < R`, kill
     **only that owned pressure PID** immediately.
  2. Wait for it to exit and for its RSS to reach 0.
  3. If the product is still not resuming and the deadline expires, request the
     **cooperative** M1 stop (`RunResultRegistry.request_stop`; product path at
     `src/zemosaic/core/zegrid/run_registry.py:1216`), then wait.
  4. Only after cooperative stop times out, kill the owned product tree. This is
     a failure (FAIL), never a PASS.
- Cleanup: SIGTERM then SIGKILL only owned PIDs; remove no store files; release
  all pressure before exit; record final RSS/MemAvailable.
- Never kill unrelated processes, never use global OOM behavior as a signal,
  never drop caches to fake recovery.

These ceilings are **proposals**; the preflight does not validate them. They
must be reviewed before execution. The 512 MiB value is a **maximum**, not a
mandatory dose: select a smaller dose when the measured requirement allows it.

---

## 4. Instrumentation and proof obligations

### 4.1 What exists in the product

- Per-wave checkpoint records: `src/zemosaic/zemosaic_zegrid_mode.py:11680-11945`
  (refusals, bounded retry, M3 wait/readmit), published in
  `r55_run_telemetry.checkpoints` (`...:12296-12320,16350-16360`) and in the
  active scheduler block `r55_scheduler.run` (`...:15161`).
- Durable store diagnostics: `per_cell_diagnostics.r55_durable_store`
  (`...:15337-15370`) with commits, released cells/bytes, store report.
- Result registry snapshot: `per_cell_diagnostics.r55_result_registry`
  (`...:15329-15333`).
- Commit attempt identity: `cpu-wave:<generation>:<wave>:<cell>`
  (`...:11988-11993`), stored in each commit manifest.
- Store commits: `commits/*.json` with `cell_id`, `attempt_id`,
  `payload_sha256`, `commit_sha256`, `created_unix_ns`, planes and metadata
  (`run_store.py:2280-2380`).

### 4.2 What exists vs what is missing

The coordinator already keeps an internal journal: `RecoveryCoordinator._log()`
and `journal()` (`src/zemosaic/core/zegrid/resource_recovery.py:1316-1329`)
record `episode_started`, `governing_observation`, `waiting_resources`,
`readmitting`, `readmission`, `episode_completed`, etc. The product does **not**
export that journal live and the M3 block (`...:11790-11930`) has no `_emit`/file
write. A post-run `status="resource_readmitted"` checkpoint is therefore the
only direct product record today. No production journal is added in this
mission.

### 4.3 Proposed test-only transparent observers (no production change)

Zero execution of the K committed cells is a **mandatory requirement**, not an
optional caveat. Commit files alone cannot prove it: a recomputation that never
commits would be invisible. The protocol requires an execution trace from the
real callables, a live wait trace, and a durable-commit trace. If any trace is
missing or cannot be attributed to the real process, the result is `NOT_RUN`,
never PASS.

**1. Actual cell-execution trace**

- CPU active path: `_run_cell_batch()` calls `zpar.pmap(_stack_cell, ...)`
  (`src/zemosaic/zemosaic_zegrid_mode.py:11291-11340`); `_stack_cell()`
  (`...:3782`) is the real worker and calls `_prepare_stack_cell()`
  (`...:3521`) / `_consume_stack_cell()` (`...:3645`). If the selected plan uses
  `_run_cell_stream` / `_run_cell_inmem`, instrument that actual selected path
  instead; do not assume the pmap path.
- A test-only wrapper calls the **original** callable and returns its result
  unchanged. It logs, per invocation, at least: `pid`, `cell_id`, start/end
  monotonic, and the attempt/generation if available. Child processes append to
  a per-PID JSONL (or append+fsync to one file with short writes); the parent
  aggregates. The wrapper must be installed before the pool is created and
  verified by a trace line from every PID that actually computes. A spawn-based
  child that does not inherit the wrapper is a `NOT_RUN`/instrumentation gap, not
a reason to fall back to a fake producer.
- Proof: every one of the K committed `cell_id`s appears **exactly once** in the
execution trace before pressure, and **zero times** after pressure/release.
  Both the trace and the commit inventory are required. No PASS without this.

**2. Safe-boundary synchronization (not polling), before the sampler**

The product CPU loop samples FIRST, then decides:
`facts = _r55_run_checkpoint_sample(disk_root)` at
`src/zemosaic/zemosaic_zegrid_mode.py:11681`, then
`_r55_run_checkpoint(..., facts=facts, ...)` at `:11712` (the bounded retry
reuses the same `facts` at `:11779`). Wrapping only `_r55_run_checkpoint` is too
late: the current wave would be governed by pre-pressure facts.

Correct order for the designated safe wave:

1. test-only `safe_wave_boundary_hook()` (one-shot; only at the chosen CPU wave
   boundary, with K commits already durable);
2. original, unmodified `facts = _r55_run_checkpoint_sample(disk_root)`;
3. original, unmodified `_r55_run_checkpoint(..., facts=facts, ...)`.

The runner designates exactly one candidate safe wave, only after K durable
commits exist. If K is not durable when the hook runs, or the candidate is
skipped, the result is `NOT_RUN`; there is no later re-selection and no re-arm.

The hook must be attributed to the exact safe-wave sampler call site
(`:11681` / its caller frame), not merely to a function name. Do not arm at:

- the in-wave heartbeat `_r55_sample_pressure()` (`:11576`);
- the M3 coordinator probe `_r55_run_checkpoint_sample()` inside the lambda
  (`:11405`), which runs during wait/recovery;
- the M3 post-wait re-evaluation sampler (`:11919`);
- the GPU inter-cell/unit samplers (`:12705`, `:13036`; GPU is out of scope).

If attribution is ambiguous or the designated call is missed, do not arm; the
wave is `NOT_RUN`. Arm exactly once; never re-arm during wait/recovery/heartbeat.

**One-shot bounded IPC.** The parent observer owns pressure. The product-side
hook sends one bounded request and waits at most a short deadline (proposal
`<=2 s`) for an acknowledgement that (a) the owned pressure process is applied,
(b) its whole-process RSS is within the hard 512 MiB cap, and (c) the measured
effect reached the intended target within caps. On timeout or refusal: release
pressure, do not proceed as if pressure governed, and classify `NOT_RUN`. Never
fabricate, rewrite, clamp, normalize or post-process the original sampled facts
or the checkpoint result. The hook is transparent and one-shot.

**Residual race, stated honestly.** Even with this order, an allocation between
the sampler and the decision, or concurrent host changes, can invalidate the
assumption. The governing observation is the original sampler's value at that
boundary; `A-P` is scenario arithmetic, not proof that pressure RSS equals the
physical `MemAvailable` drop. The checkpoint may also consume a **latched**
in-wave sample; if that latched value is lower and was taken before the arm, it
would govern a pre-arm fact. PASS therefore requires that any governing latched
fact has `latched_host.monotonic_ns >= pressure_applied`, or that no pre-arm
latched fact governs. Record `host_source`, `latched_host` and timestamps in the
barrier trace; otherwise `NOT_RUN`. PASS also requires the barrier trace to show
`hook < pressure_applied <= sampler` for the same wave and the governing fact to
be the post-arm value. Otherwise the result is `NOT_RUN`.

A completed cell is not rerun: the hook runs before the next wave's sampler, the
accepted prefix stays immutable, and only pending tasks continue. If K is not
reached before the last boundary/deadline, stop with `NOT_RUN`; do not start
pressure after completion or fake a wave.

`workers=1` is only a request, not proof of serial execution. The execution
trace records real PIDs/concurrency; report multiple PIDs rather than claiming
serial execution. The pressure process is spawned by the **parent observer**,
not by a pmap worker; it is outside the product process tree and is not
inherited by fork or spawn children. Pressure ownership stays in the parent; a
worker must never own it. If the observer cannot be kept in the parent,
`NOT_RUN`.

**3. Live wait trace**

- Test-only wrapper/subclass of `RecoveryCoordinator` in the engine module,
  forwarding the real `release`/`probe` callables and recording `_log` events to
  a live JSONL. It must not replace the release, probe, policy, intervals or
  minimum bytes. If subclassing is impossible, capture the real instance and
  poll its `journal()`; still a real object, not a mock. Missing live
  `waiting_resources`/`readmitting` events means `NOT_RUN`; the post-run
  checkpoint alone is not enough for PASS.
- The journal is the direct state evidence. The post-run checkpoint
  (`status="resource_readmitted"`, `m3_recovery.wait.observations`) must agree
  with it.

**4. Durable commit trace**

- Test-only wrapper around `RunCellStore.commit_cell()` calls the original and
  logs the returned `CommitReceipt` (`cell_id`, `attempt_id`, `payload_sha256`,
  `stored_bytes`, monotonic). The commit files remain the authority; the trace
  proves completion order and lets the handshake choose K without races.

**5. External pressure/host samples**

- `events.jsonl` at <=1 s (proposal 0.25 s): timestamp, `MemAvailable`,
  `SwapFree`, product parent/tree RSS, CPU ticks (`utime+stime`), commit count,
  pressure PID/RSS, guard state.
- Process identity allowlist only: `pid`, `ppid`, role, start monotonic, RSS,
  CPU ticks, exit code. **Do not collect** raw `/proc/<pid>/environ`, command
  line, open files, credentials or full process inventories.
- `pressure.json`: owned PID, start/end monotonic, hard cap, observed peak RSS,
  allocation steps, release reason.
- `barrier-trace.jsonl`: per candidate wave, `wave_index`, boundary id/source
  attribution, `hook_monotonic`, request/ack status, `pressure_pid`,
  `pressure_applied_monotonic`, `pressure_rss_at_ack`, `sampler_monotonic`,
  `host_source`, `latched_monotonic_ns`, governing raw after arm, and
  `ack_deadline_s`. A PASS needs `hook < pressure_applied <= sampler` for the
  same wave and the post-arm governing fact; a missing/ambiguous attribution is
  `NOT_RUN`.

**Wait evidence**

- Live journal: `waiting_resources` entry with `minimum_bytes`, governing
  observation, `probes`, `sleeps`; later `readmitting` and
  `episode_completed`. Post-run checkpoint with >=2 admissible observations
  spaced >=5 s, `release.credits_balanced=true`, one profile switch, episode
  number.
- Corroborating plateau: product alive, commits unchanged, `MemAvailable`
  below `required_raw(X_b)`, CPU-tick delta small over >=10 s. Proposed
  threshold: <=0.5 s CPU over 10 s (about 5 % of one core); this is a proposal,
  not validated. Record raw ticks and the threshold; do not hide either.
- No hot loop: journal probe count/interval plus external CPU trace; product
  constants are 5 s initial / 30 s max backoff, two observations >=5 s apart.

**After release**

- Pressure stopped, RSS 0, `MemAvailable` recovered; two spaced admissible
  observations; live `readmitting`; new commits appear.
- Execution trace: zero post-release invocations of the K committed cells;
  commit inventory: all pre-pressure commits still present with identical
  `payload_sha256`; no new attempt for a committed cell; final
  `r55_durable_store.released_cells` and registry accepted set agree.
- Final science/coverage/offsets compared to both references. Use exact array
  digests where the same binary/settings make the contract deterministic; if not
  deterministic, declare `NOT_MEASURED` rather than widen tolerances post hoc.

---

## 5. Failure classification

### FAIL

- Malformed or missing evidence: preflight JSON absent/invalid, negative/zero
  `MemAvailable`, commit file unreadable or mutated, timestamps inconsistent.
- Product crash, unhandled exception, nonzero product exit, timeout.
- Guard breach: `MemAvailable < R`, pressure RSS > cap, unowned/multiple
  pressure processes, product killed before cooperative stop, pressure not
  released.
- Reference mismatch: science/coverage/offset digests differ beyond a tolerance
  frozen before measurement.
- Hot loop, wait plateau without progress after release, deadline exhausted
  without a classified reason.
- Recompute of a committed cell: new attempt for a cell already committed, or a
  changed payload under the same identity.
- Missing/inconsistent final manifest or publication.

### NOT_RUN

- Preflight `NOT_READY` (reserve shortfall, unsupported filesystem/cgroup,
  missing runtime, missing bounded aggregate).
- Pressure cap cannot cross the threshold without violating `R`.
- Bounded minimum fits at the reserve floor (`BOUNDED_MIN_FITS_AT_RESERVE`).
- Fewer than K valid commits before the pressure point, or no remaining wave.
- Structural capacity refusal (bounded minimum > `MemTotal`) — no wait can help.
- Host unsupported by the declared contract; Windows/GPU excluded by scope.

No automatic retry to obtain green. A rerun requires a new, classified cause and
new evidence; the first failure stays recorded. Do not change probes,
thresholds, reserve, identity or science to convert NOT_RUN into PASS.

Separate gates, never merged into this protocol:

- reference (no pressure) vs experiment;
- wait→resume;
- cooperative cancel while waiting (separate scenario; stop must return
  `STOPPED_RESUMABLE` in <2 s per the product seam test);
- full restart/new process (M4/M7);
- Windows/NTFS and GUI (M6/M7).

---

## 6. Proposed commands, artifacts and criteria

### 6.1 Runnable now (read-only)

```
PY=/home/tristan/.openclaw/workspace/projects/zemosaic/.venv/bin/python
$PY tools/zegrid_resume_m3/m3_linux_preflight.py \
  --repo <repo> \
  --workspace <real-disk-workspace> \
  --bounded-aggregate-bytes <measured-from-bounded-reference> \
  --json-out <owned>/preflight.json
```

Exit codes: `0` READY_TO_PLAN, `2` NOT_READY, `3` tool/output failure or a
parameter outside the public contract. `--reserve-bytes` below 1 GiB,
`--pressure-bytes`/`--pressure-max-bytes` above 512 MiB, or a selected dose above
its cap are rejected before any probe; only stricter values are accepted.
Existing `--json-out` is never overwritten.

### 6.2 Proposed only (runner **not built**)

The pressure runner does not exist in this mission. These commands are a
contract for the next review, not runnable today:

```
# 1) Dense reference (no pressure)
$PY tools/zegrid_resume_m3/m3_linux_pressure_runner.py reference \
  --corpus <fresh-corpus> --out <fresh>/reference-dense \
  --store <fresh>/store-dense --json-out <fresh>/reference-dense.json

# 2) Bounded reference (no pressure; measures X_b)
$PY tools/zegrid_resume_m3/m3_linux_pressure_runner.py reference \
  --storage-profile bounded \
  --corpus <fresh-corpus> --out <fresh>/reference-bounded \
  --store <fresh>/store-bounded --json-out <fresh>/reference-bounded.json

# 3) Pressure experiment (only after preflight READY with X_b)
$PY tools/zegrid_resume_m3/m3_linux_pressure_runner.py experiment \
  --corpus <fresh-corpus> --reference <fresh>/reference-dense.json \
  --out <fresh>/experiment --store <fresh>/store-experiment \
  --workers 1 --k-commits 2 --reserve-bytes 1073741824 \
  --pressure-bytes 536870912 --pressure-cap-bytes 536870912 \
  --max-lifetime-s 900 --json-out <fresh>/experiment.json

# 4) Separate cancellation scenario (optional, later)
$PY tools/zegrid_resume_m3/m3_linux_pressure_runner.py cancel-while-waiting \
  --corpus <fresh-corpus> --out <fresh>/cancel --store <fresh>/store-cancel \
  --json-out <fresh>/cancel.json
```

Proposed runner exit codes: `0` PASS, `2` NOT_RUN, `3` FAIL, `4` tool/usage.

### 6.3 Artifacts

```
<artifact>/
  preflight.json                 # read-only host decision
  reference-dense/
    manifest.json                # real product manifest
    science.sha256
    coverage.sha256
    stdout.log / stderr.log
    exit.json
  reference-bounded/
    manifest.json                # aggregate_bytes / final_execution
    science.sha256 / coverage.sha256
    stdout.log / stderr.log
    exit.json
  experiment/
    events.jsonl                 # <=1 s external samples
    barrier-trace.jsonl          # hook/pressure/sampler timestamps per wave
    execution-trace.jsonl        # real cell invocations (pid/cell/start/end)
    wait-trace.jsonl             # live RecoveryCoordinator journal events
    pressure.json                # owned PID, RSS, steps, release
    commits-before.json          # K commits + payload digests
    commits-after.json           # final inventory; no new attempt for K
    manifest.json                # r55_run_telemetry / r55_durable_store
    science.sha256 / coverage.sha256
    stdout.log / stderr.log
    exit.json                    # PASS/NOT_RUN/FAIL + classification
```

### 6.4 Criteria → proof matrix

| Criterion | Required proof | Where |
|---|---|---|
| Safe host headroom | preflight READY with `A-P≥R`; observed timestamp | `preflight.json` |
| Bounded minimum known and refusable | `X_b`, `required_raw(X_b)`, `>R`, `>A-P` | `reference-bounded/manifest.json` + preflight |
| K real commits before pressure | K distinct commit files + payload SHA | `commits-before.json` |
| Barrier before the governing sampler | `barrier-trace.jsonl` shows `hook_monotonic < pressure_applied_monotonic <= sampler_monotonic` for the same wave; dedicated safe-wave attribution; one-shot | `barrier-trace.jsonl` |
| Real execution, not a counter | execution-trace event from the real `_stack_cell`/`_prepare`/`_consume` callable (pid, cell_id, start/end); commit `attempt_id`/`payload_sha256`; registry accepted set | `execution-trace.jsonl`, commit files, manifest `r55_durable_store` |
| WAITING_RESOURCES entry | live coordinator journal `waiting_resources` event + post-run `status=resource_readmitted` / `m3_recovery` agreeing | `wait-trace.jsonl`, `r55_run_telemetry.checkpoints` |
| Wait is light/no hotloop | live journal probe count/interval; >=2 spaced observations; external CPU trace | `wait-trace.jsonl`, `events.jsonl` |
| Release and re-admission | pressure RSS→0; `MemAvailable` recovery; live `readmitting`; 2 admissible obs >=5 s | `pressure.json`, `events.jsonl`, `wait-trace.jsonl` |
| No recompute of K | execution trace shows each K cell exactly once before pressure and **zero** after; same commit identity+payload; no new attempt for K cells | `execution-trace.jsonl`, `commits-before/after` |
| Governing fact post-arm | the checkpoint's governing `raw_available_bytes` equals the post-arm sampler value for the same wave; no rewrite. If `host_source` includes a latched fact, its `monotonic_ns` is `>= pressure_applied`, else NOT_RUN | `barrier-trace.jsonl`, manifest checkpoint |
| Final parity | science/coverage/offset digests vs dense and bounded references | hashes + manifests |
| Cleanup | no owned PID alive; no store deletion; product exit 0 | `exit.json`, `pressure.json` |
| No false green | FAIL/NOT_RUN classification preserved; no retry loop | `exit.json` |

### 6.5 Next single executable step

The only step runnable today is the read-only preflight. The rework-1 snapshot
(`2026-10-28T22:18:08Z`, `MemAvailable` 1,694.9 MiB) **passes** the 1 GiB
reserve at 512 MiB, and the current sole blocker is the unmeasured bounded
minimum (`BOUNDED_AGGREGATE_NOT_MEASURED`). Do not move host solely because an
earlier snapshot was low. The next executable step after review is:

1. run the no-pressure bounded reference with the frozen corpus/settings to
   obtain a provenance-recorded `X_b` at the pressure-governing raw, then
2. re-run the preflight with `--bounded-aggregate-bytes X_b` and the selected
   dose. If 512 MiB breaches the reserve but the dose window is non-empty,
   select a smaller dose in the window; do not lower the reserve.

If `X_b` still fits at the reserve floor, stop with `NOT_READY` for this corpus;
do not lower the reserve or shrink the science to force the wait.

---

## 7. Source map (code inspection)

| Topic | Source |
|---|---|
| Product probe / RUN facts | `src/zemosaic/zemosaic_zegrid_mode.py:330-332,8560-8565`; `src/zemosaic/core/zegrid/runtime_control.py:400-470` |
| RAM factors / budgets | `src/zemosaic/core/zegrid/ram_contract.py:91-92,220-230` |
| M3 observation contract | `src/zemosaic/zemosaic_zegrid_mode.py:9579-9591` |
| M3 required raw inverse | `src/zemosaic/zemosaic_zegrid_mode.py:9570-9577` |
| M3 coordinator / release | `src/zemosaic/zemosaic_zegrid_mode.py:9593-9632` |
| M3 arming gate | `src/zemosaic/zemosaic_zegrid_mode.py:11389-11417` |
| M3 wait path | `src/zemosaic/zemosaic_zegrid_mode.py:11790-11930` |
| CPU wave loop / commits | `src/zemosaic/zemosaic_zegrid_mode.py:11344-12180,11980-12100` |
| Product entry | `src/zemosaic/zemosaic_zegrid_mode.py:16394` |
| Waiter / readmission | `src/zemosaic/core/zegrid/resource_recovery.py:200-280,338-520,1380-1570` |
| cgroup diagnostics (root-only) | `src/zemosaic/core/zegrid/runtime_control.py:318-340` |
| Durable commit boundary | `src/zemosaic/core/zegrid/run_store.py:2156-2260,2337-2345,2763-2800` |
| Store layout/index | `src/zemosaic/core/zegrid/run_store.py:1602-1690,1969-1990` |
| M3 deterministic tests | `tests/test_zegrid_resume_m3_product_seam.py`, `...pipeline.py`, `...recovery.py` |
| Prior harness patterns | `tools/zegrid_resume_m1/m1_no_pressure_control.py`, `tools/zegrid_resume_m4/m4_resume_harness.py`, `tools/zegrid_resume_m5b/*` |

## 8. Explicit limitations

- This protocol does **not** execute physical pressure, does not run the product,
  and does not write a store.
- The preflight decision is not an M3 PASS; `READY_TO_PLAN` only means the host
  math and probes are acceptable for planning.
- The preflight enforces the public contract (`R >= 1 GiB`, `0 < P <= 512 MiB`,
  cap `<= 512 MiB`); a direct pure-function caller outside it cannot obtain
  READY. The physical ceilings and the guard are proposals, not validated.
- The cgroup walk treats the mount root without memory files as normal, but any
  missing/malformed/unreadable ancestor is unknown and NOT_READY. A finite
  effective limit is unsupported for this host-wide experiment.
- The current host snapshot passes the reserve at 512 MiB but the bounded
  minimum is unmeasured; other snapshots differed. No snapshot is a reservation,
  and no universal host fact is inferred.
- Mandatory zero-compute proof depends on test-only execution/live-wait/commit
  observers that do not exist yet. Without them there is no PASS; commit IDs
  alone are not proof. No production journal is added here.
- The pressure handshake is protocol-level only and not implemented. It must be
  attributed to the exact safe-wave sampler call site (`:11681`) and run before
  that original sampler; heartbeat, M3 recovery/probe and GPU sampler sites must
  not arm it. If the designated wave or attribution is missed, `NOT_RUN`.
  Sampling is best-effort: an allocation between sampler and decision, or host
  changes, can still invalidate the assumption; the original post-arm fact
  governs.
- Full FITS parity, GPU/M31, Windows/NTFS, cooperative GUI stop, full restart,
  already-bounded/repeated-pressure profile qualification, and M5
  finishing/publication remain NOT_RUN for M3.
- The preflight fixed tool defects found during this mission (root mount
  matching; inherited cgroup limit; weak parameter overrides; inconsistent
  telemetry; unmeasured bounded aggregate). r0 is retained as pre-fix evidence;
  rework-1 artifacts are the corrected evidence.

## 9. Rework-2 result — bricks built, `X_b` measured, still NOT_READY

Mission `ZM-ZEGRID-M3-LINUX-RUNNER-20261029` (branch
`feat/zegrid-resilient-run-resume`, parent HEAD `7d0b759`). No production,
science, schema, API or default change. Physical pressure was **not** executed.

### 9.1 The two bricks now exist

- `tools/zegrid_resume_m3/m3_observers.py` — harness-injected wrappers that
  forward the ORIGINAL callables unchanged:
  - execution trace on the real `_stack_cell` (pid, cell_id, start/end);
  - durable-commit trace on `RunCellStore.commit_cell` (receipt forwarded);
  - live wait trace by capturing the REAL `RecoveryCoordinator._log` through the
    `_r55_m3_make_coordinator` factory (release/probe/policy untouched);
  - barrier trace on `_r55_run_checkpoint_sample`, armed one-shot ONLY when the
    caller frame is the resolved safe-wave call site (line 11681, resolved from
    source after the `while offset < total_tasks:` anchor).
  - **Correction to §4.2/§4.3 attribution:** the CPU wave loop is the nested
    closure `_run_r55_active_cpu_waves`; attribution is by EXACT resolved line in
    the product file, because the function name alone also matches the M3
    post-wait re-sampler at `:11919`. A name-based check was observed to classify
    real safe-site calls as `other_site`; it is no longer used.
- `tools/zegrid_resume_m3/m3_linux_pressure_runner.py` — owned pressure runner:
  one child, anonymous mmap + page touch in 64 MiB steps, whole-process RSS hard
  cap 512 MiB, parent-owned sampling at 0.25 s, guard order
  (pressure RSS/reserve → wait exit → cooperative stop → last-resort owned kill),
  one-shot unix-socket IPC with a <=2 s ack deadline, `pressure.json`/events
  artifacts, SIGTERM then SIGKILL only owned PIDs, no cgroups/swap/sysctl/cache
  drops. Experiment mode is gated: it refuses before spawning anything unless
  BOTH `--i-am-reviewed` and `M3_PRESSURE_EXPERIMENT_REVIEWED=1` are present, and
  even then this mission's build reports
  `REVIEWED_FLOW_NOT_EXECUTED_IN_THIS_MISSION` / NOT_RUN.

### 9.2 Corpus gap found by the real product

The §2.3 corpus proposal (8×512×512, grid 4×4, halo 64) is **scientifically
inadmissible** and the product refuses it before any gauge work:

```
LayoutInfeasible: pinned layout 4x4 violates scientific floors:
min_patch_area=104304 (limit 50000), halo_overhead=1.7337 (limit 0.5)
```

Halo 64 with a ~800 px canvas (512 frame + dither) cannot satisfy
`max_halo_overhead=0.5` for any multi-cell grid: it needs a cell of ~570 px
(≈2300 px canvas). The floors were NOT lowered and the halo was NOT reduced.
The reference corpus was therefore scaled (frames 1024², dither ×2, grid 2×2,
halo 64) — same science settings, same halo, same per_cell gauge, more pixels —
which is admissible (`halo_overhead≈0.35`). A 1600²/4×4 attempt was also started
and exceeded the 900 s mission budget; it is retained as an aborted-capacity
observation, not as evidence.

### 9.3 Measured `X_b` (real bounded reference, no pressure)

- Dense reference: PASS, 59–60 s, 4/4 cells, 4 commits, 4 execution events,
  peak tree RSS 783.6 MiB, aggregate 1,034,015,744 B.
- Bounded reference: PASS, 62 s, 4/4 cells, 4 commits, 4 execution events,
  peak tree RSS 767.8 MiB, `X_b = 960,876,544 B` from
  `r55_scheduler.execution.aggregate_bytes` (identical value in
  `per_cell_diagnostics.ram_contract.aggregate_bytes`), plan raw
  `2,850,992,128 B`, plan budget `2,280,793,702 B`, layout 2×2,
  `units_in_flight=1`, `prefetch_units=0`, 4 waves, all commits idempotent=0.
- Dense and bounded science/coverage arrays are SHA-256 identical for this
  corpus (`8580fe50…`/`899aba8f…`): the profile switch changed the RAM plan, not
  the science.

### 9.4 Preflight with `X_b` — NOT_READY (unchanged verdict, now measured)

```
required_raw(X_b) = ceil(10*960876544/9) = 1,067,640,605 B
reserve R                              = 1,073,741,824 B
```

`required_raw(X_b) <= R`, so condition 4 of §3.1 fails: at the safe reserve floor
the bounded minimum would still be admitted and M3 would never enter
`WAITING_RESOURCES`. The tool reports three blockers —
`BOUNDED_MIN_FITS_AT_RESERVE`, `NO_SAFE_PRESSURE_DOSE`,
`PRESSURE_CAP_INSUFFICIENT` (a 512 MiB cap can never move `A` from ~2.7 GiB to
1.07 GiB) — exit 2. No threshold, reserve, probe, identity or science value was
changed to avoid this.

**Consequence:** with this corpus/host, no dose within the 512 MiB cap and the
1 GiB reserve can force the bounded minimum to refuse. The M3 wait is
unreachable at this scale; a future experiment needs a corpus/frame budget whose
bounded minimum exceeds `required_raw > R` (≈ >= 0.97 GiB aggregate) while
`required_raw <= A - P` remains true for the selected dose — i.e. a larger
science payload, not a weaker guard.

### 9.5 Residual limits (unchanged or newly explicit)

- Pressure experiment: **NOT_RUN** (no physical pressure was applied; guards were
  tested only with simulated children).
- Already-bounded / repeated-pressure integration gap (§2.6) remains open.
- Engine-only namespace: the recovery key is still not a shipped config key.
- Sampling is best-effort; `A-P` is planning arithmetic, never measured proof.
- The measured `X_b` belongs to the admissible 1024²/2×2 corpus; it does not
  describe the (inadmissible) 512²/4×4 proposal.

### 9.6 Continuation r1 — 1600² measurement, still NOT_READY (host-fit limit)

Artifacts were relocated OUT of the product checkout to
`/home/tristan/.openclaw/workspace/.m3-linux-runner-20261029/`; the runner now
defaults to a workspace artifact base outside any git checkout and refuses a
root inside one unless `--allow-inside-git` is passed explicitly (the relocated
workspace base is itself a notes checkout, hence the explicit override for this
mission).

One no-pressure bounded reference at frames 1600², dither ×2, grid 2×2,
halo 64 (fresh roots, real ext4, reserve guard 1 GiB, frozen tree ceiling):

```
PASS  138.8 s  exit 0   4/4 cells, 4 commits, 4 execution events, 4 waves
peak product tree RSS   1,139,167,232 B (1.061 GiB)   ceiling 2,073,052,864 B (not tripped)
min MemAvailable        2,013,675,520 B (1.876 GiB)   (reserve 1 GiB never breached)
X_b = 1,133,114,368 B  @ r55_scheduler.execution.aggregate_bytes
      = per_cell_diagnostics.ram_contract.aggregate_bytes
plan raw 2,516,406,272 B   plan budget 2,013,125,017 B   layout [2,2]
```

`required_raw(X_b) = ceil(10·1133114368/9) = 1,259,015,965 B`.
Condition 4 now holds (`> R = 1,073,741,824 B`), but condition 2 fails:
`A - P = 2,668,462,080 B` (live preflight `A` 3,056.8 MiB, `P` 512 MiB) is still
above `required_raw(X_b)`, so the bounded retry would fit *after* pressure and
M3 would never enter `WAITING_RESOURCES`. Preflight verdict: **NOT_READY**
(`NO_SAFE_PRESSURE_DOSE`, `PRESSURE_CAP_INSUFFICIENT`, exit 2; the tool reports a
required dose of 1,946,317,028 B against the 512 MiB cap).

**Host-fit conclusion.** The measured scaling is sublinear (1024² → 960.9 MB,
1600² → 1,133.1 MB, ~2.44× pixels for 1.18× aggregate). To put `required_raw`
inside `(A-P, A]` on this host, `X_b` would need ≈ 2.24–2.5 GB aggregate, i.e.
roughly a 2260–2400 px canvas. Extrapolating the same working-set growth, the
product tree there is ≈ 1.5–1.8 GB; plus the mandatory 1 GiB reserve that is
≈ 2.5–2.8 GB against a live `A` of 2.7–3.0 GiB — no safe margin, and the plan
itself must still satisfy `X_b <= 0.8·A`. **The corpus that makes the M3 wait
reachable most likely does not fit this 7.7 GiB host at all** without lowering
the reserve or the science, both of which are forbidden. This is a host-capacity
conclusion, not a product defect.

The pressure experiment remains **NOT_RUN**; no physical pressure was applied.
