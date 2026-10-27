# ZeGrid resilient run/resume — M2 durable store

Date: 2026-10-27
Mission: `ZM-ZEGRID-RESILIENT-RUN-RESUME-M2-20261027`
Phase: implementation (M2)
Baseline: `f09c9cf0ac05afc97c6efe164364c8e5269f9240` (M1 accepted and published)
Worktree: `worktrees/zemosaic-resilient-run-resume`, branch
`feat/zegrid-resilient-run-resume`
Status: **IMPLEMENTED — awaiting Junior verification and Nono review.**
Coco does not accept M2. Windows execution of the bench is **NOT_RUN** here
(section 6); nothing in this document is a Windows or GPU qualification claim.

---

## 1. What M2 adds

### 1.1 `src/zemosaic/core/zegrid/run_store.py` (new, isolated)

`RunCellStore` is the durable counterpart of the M1 in-memory registry. It
owns one run directory and exposes:

- `build_run_identity(...)` / `validate_run_identity(...)`: the frozen
  scientific identity (plan §6.1). The scientific contract holds the input
  inventory with **content** hashes, canvas/WCS, grid/geometry, halo,
  memberships, the complete atomic gauge result, the science contract and the
  dtype contract. RAM/workers/prefetch live in a separate `execution_context`
  that **never** participates in compatibility.
- `commit_cell(result, attempt_id=..., backend=..., plan_generation=...)`:
  validates the REAL producer schema (the same `run_registry` validators as
  M1), then publishes atomically. Idempotent by `(cell_id, attempt_id)`; a
  conflicting payload for the same validated identity is refused, never
  overwritten.
- `verify_cell` / `verify_all`-style `scan()`: per-cell sha256 verification of
  every declared array object plus the manifest's own digest.
- `load_payload`: verified, **read-only** arrays (no writable mmap, no pickle,
  no content execution).
- `index()` / `rebuild_index()`: the global index is a pure function of the
  on-disk commits; it is rewritten atomically and rebuilt on open whenever it
  is missing, stale or tampered.
- `completeness(expected_cell_ids)`: exact sparse completeness
  `expected == COMPLETE + EMPTY_VALIDATED` over the real `cell_id` set.
- `commit_and_release(registry, cell_id, ...)`: the write-path hand-off
  (commit → verify → release the M1 payload) and
  `ordered_payloads(registry, cell_ids)`: canonical-order payloads that reload
  released cells from the verified commit.
- `DiskBudget` + `preflight_bytes` + `_reserve_space`: separate disk lines,
  free-space margin and a real reservation where the OS offers one.

### 1.2 `src/zemosaic/core/zegrid/run_registry.py` (minimal additive change)

- `release_payload(cell_id, expected_payload_sha256=...)`: releases ONE cell's
  payload after a durable commit. Idempotent, digest-checked, refcounted per
  unique backing allocation (a shared buffer is counted once and only freed
  with its last owner), and it records `released`/`released_bytes` in the
  metadata.
- `PayloadReleasedError`: released payloads are never silently re-served from
  RAM; callers must re-read them from the store.
- `counts()` gains `released_cells`/`released_bytes`. No M1 semantic was
  weakened: acceptance, idempotence, generations, disposal and byte accounting
  are unchanged (23 M1 registry tests still pass).

### 1.3 `src/zemosaic/zemosaic_zegrid_mode.py` (write-path seam only)

- The store is created **after** the gauge and the admitted bundle, from the
  real run facts (`descs`, canvas, bundle layout/memberships, gauge result,
  science config), and only when the caller sets the explicit
  `zegrid_durable_store_dir` config attribute. No new GUI, no new default disk
  consumption: production enablement policy is M4/M6.
- Each accepted CPU cell is handed to the store at the M1 acceptance boundary;
  the large arrays are released only after the commit is durable.
- A store failure (disk full, ENOSPC, integrity) is reported explicitly,
  preserves prior commits and **degrades to M1 in-memory conservation**: the
  payload is not released, no durable claim is made, the run continues.
- At the end of the CPU waves the canonical results are read back from the
  store for the cells that were released, so assembly receives the exact
  payloads. The final publication path, the science, the budgets and the
  floors are untouched.
- `per_cell_diagnostics["r55_durable_store"]` is recorded only when a store was
  requested (a store-less run keeps its exact manifest shape).

### 1.4 `tools/zegrid_resume_m2/` (bench)

`m2_bench.py` + `run_m2_bench.ps1` + `README.md`: fail-closed, bounded,
source-tree-only bench. Any skip is `NOT_RUN`, never `PASS`.
Exit codes `0 PASS / 2 NOT_RUN / 3 FAIL / 4 TIMEOUT / 5 CAP_EXCEEDED / 6 GUARD`.

---

## 2. Conformance table (plan §6.1–6.7)

| Plan | Requirement | Implementation | Test evidence |
|---|---|---|---|
| §6.1 | Run identity: schema/version, inventory with content identity + order, canvas/WCS, grid, halo, memberships, global/per_cell decision, **complete atomic gauge** (references, weights, exclusions, coefficients), calculation/dependency version | `build_run_identity`, `validate_run_identity`, `identity_sections`, `IDENTITY_SECTIONS`; gauge fingerprint hashes the whole gauge object (arrays are hashed, not inlined) plus reference ids/mode | `test_m2_runtime_context_change_never_invalidates_science`, `test_m2_scientific_change_is_refused_with_a_reason[gauge]`, `test_m2_real_cpu_run_commits_then_releases_and_reloads_identical_payloads` (NOT_RUN on this host) |
| §6.1 | Per cell: identity, status, bounds, core_slice, contributors, every really-required plane (science, estimator, support_w1/w2, n_eff, validity, surviving), per_cell PATCH with pre-offset/pre-feather halo, per-cell photometric reference pin, backend, effective parameters, sizes/dtypes/shapes/digests | `commit_cell` plane extraction (`_plane_items`) + `zrr` schema validation + `payload_meta`; manifest `planes[]` declares name/path/dtype/shape/nbytes/sha256/array_sha256 | `test_m2_commit_publishes_verified_planes_and_roundtrips_payload`, `test_m2_empty_validated_cell_publishes_without_arrays` |
| §6.2 | Order: reserve → attempt-private temp → flush+fsync → verify sizes/shapes/digests → atomic cell commit → atomic index; index reconstructible between 5 and 6 | `_write_object` (`_reserve_space` → temp `.part` → `os.fsync` → sha256/size verification → `os.replace`), `atomic_write_bytes` for the manifest, `_write_index` last; `_load_or_rebuild_index` rebuilds from commits | `tests/test_zegrid_run_store_m2_kill.py` (20 cases, 4 steps × 5 iterations), `test_m2_index_is_rebuilt_when_interrupted_between_commit_and_index`, `test_m2_index_is_reconstructible_after_a_lost_index` |
| §6.2 | No pickle/content execution; strict schema; paths confined to the run; single-writer lock; PID alone insufficient; orphan recoverable, live lock never stolen | JSON-only manifests, dtype allowlist, no object dtype; `confined_path`/`safe_name`; `StoreLock` = `flock`/`msvcrt.locking` + owner `(instance, pid, process start token, boot id, host, session)`; `owner_is_live`, `orphan_recovered`, clean-release record | `test_m2_no_pickle_and_no_object_dtype`, `test_m2_unknown_schema_and_unknown_fields_are_refused`, `test_m2_manifest_path_traversal_is_refused`, `test_m2_cell_id_and_attempt_are_path_confined`, lock family (second writer, orphan, live owner, concurrent process) |
| §6.3 | Content hash not name/mtime; RAM/workers/prefetch never invalidate science; inputs/WCS/halo/memberships/gauge/science refuse explicitly; tile/backend only under a qualified parity contract; relocation ≠ new identity; new software version = explicit compatibility (no force) | `content_sha256_file`, content-only inventory rows (`_inputs_identity_rows`, self-consistent aggregate), `check_identity_compatibility` with per-section reasons, `StoreIdentityMismatch`, schema/version refusal, tile/backend refusal unless `parity_qualified` | identity parametrised refusal matrix (8 sections), `test_m2_relocated_dataset_keeps_identity`, `test_m2_tile_change_without_parity_is_refused`, `test_m2_runtime_context_change_never_invalidates_science` |
| §6.4 | Validate store, restore identity/gauge, load descriptors, plan only missing cells; isolated corruption invalidates only that cell; corrupted gauge/identity never "guessed" | `open` = lock → identity → scan commits → index/rebuild; `verify_cell`/`scan` isolate invalid entries; `load_payload` refuses a corrupted cell; identity/index digest mismatch refuses the open | `test_m2_corrupted_cell_is_invalidated_alone`, `test_m2_corrupted_commit_metadata_is_invalid_on_reopen`, `test_m2_index_is_rebuilt_when_interrupted_between_commit_and_index` |
| §6.5 | Separate budget lines (disposable caches / durable cells / assembly / transactional margin); free-space preflight with margin; disk full preserves prior commits, no fake COMPLETE, no automatic deletion | `DiskBudget` (4 lines, `to_dict`), `preflight_bytes` (budget + free − margin), `_reserve_space` (`posix_fallocate` / Windows `FileAllocationInfo` / ftruncate fallback), ENOSPC → `StoreDiskFullError` before publication; nothing is ever deleted | `test_m2_disk_full_preserves_prior_commits_and_publishes_nothing` (3 stages), `test_m2_budget_exhaustion_refuses_before_writing`, `test_m2_free_space_margin_is_enforced`, `test_m2_identity_write_failure_fails_closed` |
| §6.6 | Order: open run → lock → read identity → validate inputs/artifacts → restore geometry/gauge → admit only remaining work; sparse cell set, never `0..K`, never "last folder found" | `open` enforces lock-before-identity and identity-before-scan; `completeness` works on an arbitrary sparse `cell_id` set; `commit_record` refuses an ambiguous multi-attempt lookup (no "last found") | `test_m2_sparse_completeness_never_uses_a_prefix`, `test_m2_concurrent_writer_process_is_refused`, kill matrix |
| §6.7 | Idempotent retry by `(cell_id, attempt_id)`; lost receipt found by lookup; two different contents refused; pre-offset state immutable/read-only; compatibility matrix; distinguish garbage / index orphans / old runs; local FS only; OS-ownership lock; index not O(pixels) | `commit_cell` idempotence + `StoreConflictError`; `load_payload` returns `writeable=False` arrays; `report()` distinguishes invalid entries, index state and lock; index is O(cells) | `test_m2_publication_is_idempotent_by_cell_and_attempt`, `test_m2_conflicting_payload_for_same_identity_is_refused`, `test_m2_commit_publishes_verified_planes_and_roundtrips_payload` (read-only assertion) |

Not implemented by design (later milestones): run restore/resume engine and
automatic run selection (§6.6 policy, M4), resource wait/backoff and
generations (M3), bounded assembly/offsets (§6.4 reread, M5), GUI (M6),
Windows GPU science qualification (M7).

---

## 3. Quantified disk budget

Measured with the real store on a 1024 px core cell with a 32 px halo
(1088×1088 patch, RGB float32 science):

| Plane | dtype | Bytes |
|---|---|---|
| `core.science_core` | `<f4` (1024²×3) | 12,582,912 |
| `core.estimator_weight_sum_core` | `<f4` | 4,194,304 |
| `core.support_w1_core` | `<f4` | 4,194,304 |
| `core.support_w2_core` | `<f4` | 4,194,304 |
| `core.n_eff_support_core` | `<f4` | 4,194,304 |
| `core.valid_mask_core` | `\|b1` | 1,048,576 |
| `core.surviving_sample_count_core` | `<i4` | 4,194,304 |
| `patch.science` | `<f4` (1088²×3) | 14,204,928 |
| `patch.valid_mask` | `\|b1` | 1,183,744 |
| `patch.surviving` | `<i4` | 4,735,744 |
| **objects total** | | **54,726,656** (52.19 MiB) |
| commit manifest (JSON) | | 4,121 |
| **stored per cell** | | **54,730,943** (52.19 MiB) |

Budget lines (formula and one worked example):

| Line | Formula | Worked example |
|---|---|---|
| Durable cells | `n_cells × stored_per_cell` | 187 × 54,730,943 = **10,234,686,341 B ≈ 10.23 GB (9.53 GiB)** |
| Assembly | `W×H×3×4` (science) + `2×W×H×8` (num/wsum f64) | 12000×10000: 1.44 GB + 1.92 GB = **3.36 GB** |
| Disposable caches | measured per-cell cache peak × in-flight cells (never durable) | not part of the durable line |
| Transactional margin | `max(16 MiB, durable // 20)` (`default_transactional_margin`) | **511,734,317 B ≈ 488 MiB** |
| **Total run disk** | sum of the above | **≈ 14.1 GB** |

Notes:

- The store is a **disk line of its own**, not a free residue: it can approach
  or exceed the final mosaic size (plan §6.5). At 64 px halo the patch grows to
  1152² and the per-cell figure rises by ~14 %.
- The budget is a **hard ceiling** (`durable_cell_bytes`): a commit that would
  cross it is refused before any byte is written (`StoreDiskFullError`), prior
  commits stay valid and nothing is deleted.
- The free-space preflight keeps `transactional_margin_bytes` untouched, so a
  failed transaction can always be rolled back without touching prior commits.
- Disk-full policy: preserve prior commits, no `COMPLETE`, no automatic
  checkpoint or user-data deletion. After space is freed the same attempt is
  simply retried (idempotent by `(cell_id, attempt_id)`).

---

## 4. Outcomes on this host (exact)

Bench (fail-closed):

```
python3 tools/zegrid_resume_m2/m2_bench.py \
  --source-root "$PWD" --output-root /tmp/zgprobe/m2_bench \
  --zgprobe-root /tmp/zgprobe --max-seconds 1200
→ {"status": "NOT_RUN",
   "totals": {"passed": 107, "failed": 0, "errors": 0, "skipped": 8},
   "reasons": ["m2_real_store_run skipped 2 test(s)",
               "m1_conservation skipped 6 test(s)"]}
```

| Suite | Passed | Skipped | Failed | Errors | Wall |
|---|---|---|---|---|---|
| `m2_store_contract` | 39 | 0 | 0 | 0 | 5.4 s |
| `m2_kill_matrix` | 20 | 0 | 0 | 0 | 8.5 s |
| `m2_real_store_run` | 0 | 2 | 0 | 0 | 3.2 s |
| `m1_registry` | 23 | 0 | 0 | 0 | 3.8 s |
| `m1_conservation` | 4 | 6 | 0 | 0 | 3.1 s |
| `m0_witness` | 1 | 0 | 0 | 0 | 19.2 s |
| `m0_harness` | 20 | 0 | 0 | 0 | 3.7 s |

Bench artifacts 1,627 B; source provenance hashes identical before/after;
no source file was edited by the bench. `py_compile` OK for
`run_store.py`, `run_registry.py`, `zemosaic_zegrid_mode.py`,
`tools/zegrid_resume_m2/m2_bench.py`.

Kill matrix detail: 4 write steps × 5 iterations = 20 cases, all passed. For
each: the two pre-kill cells validate, the target cell is either fully
published or absent, `scan()["invalid"]` is empty, the index equals the on-disk
commit set, and deleting `index.json` and reopening rebuilds the identical set.

---

## 5. What is proven, and what is NOT_RUN

Proven on this host (Linux 6.12, real OS primitives):

- atomic per-cell publication: a SIGKILL at temp write, fsync, commit rename or
  index update never exposes a partial cell (20/20);
- index reconstruction after a killed index update;
- isolated corruption: one corrupted object/commit invalidates that cell only;
- single-writer lock: second in-process writer, second live process and a
  fabricated live-owner record are all refused; a dead owner is recovered;
- disk full (ENOSPC injection at three stages + budget + free-space margin)
  preserves prior commits, publishes nothing and retries cleanly;
- identity compatibility: RAM/workers/prefetch changes are accepted; 8
  scientific sections + tile/backend change are refused with explicit reasons;
  dataset relocation keeps the identity;
- format safety: no pickle, strict schema, unknown field/version refused,
  path traversal refused, dtype outside the contract refused;
- RAM release: committed cells drop `retained_bytes`, shared buffers are
  counted once, release is idempotent, a failed commit never releases, and the
  store re-supplies byte-identical read-only payloads;
- M1 (23 + 4 passed) and M0 (1 + 20 passed) non-regression.

NOT_RUN here (declared, not qualified):

- `tests/test_zegrid_resume_m2_store_run.py` (2 cases): the host has ~670 MiB
  available RAM against the 1,200 MiB floor of the 2×2 corpus, so the
  store-enabled REAL orchestration path could not execute. It skips with an
  explicit reason; the bench reports `NOT_RUN`;
- the 6 M1 real-orchestration conservation cases (same RAM floor);
- **all Windows execution** of the bench (lock `msvcrt`, TerminateProcess kill
  matrix, NTFS rename semantics) — shipped but not run here.

---

## 6. Design decisions and limits

1. **Store is opt-in** (`zegrid_durable_store_dir`). Enabling checkpoint
   writing by default would consume gigabytes on every run without a UI/budget
   decision; that policy belongs to M4/M6. The seam is real and exercised; the
   default remains the M1 behaviour.
2. **Store failure degrades, never aborts science.** A disk-full or integrity
   failure stops durable publishing for the rest of the run, keeps the payload
   in RAM and records the reason. This preserves "no partial commit" and "no
   fake COMPLETE" without turning a checkpoint disk problem into a lost 10-hour
   run. M3/M4 refine the retry policy.
3. **`ordered_payloads` reload is not bounded assembly.** Released payloads are
   re-read per cell and returned to the existing assembly path; this preserves
   the current memory profile at assembly time. Real bounded reread is M5.
4. **fsync semantics.** Files are fsynced before rename and the parent directory
   is fsynced on POSIX. On Windows/NTFS the directory fsync is a documented
   no-op; `os.fsync` still flushes the file. Rename+fsync is not a power-loss
   guarantee and is not claimed as one.
5. **Reservation strength** is recorded per object: `posix_fallocate` /
   Windows `FileAllocationInfo` where available, `ftruncate` otherwise. The
   commit manifest is always published last, so a weak reservation cannot
   expose a partial cell.
6. **dtype contract uses per-plane allowlists** (`<f4`/`<f8` for float planes,
   `|b1` masks, `<i4`/`<i8` counts) because the bounded and legacy producers
   legitimately differ; object/complex/non-native dtypes are always refused.
7. **Frame identity is content-only.** The inventory keeps the logical path as
   provenance; the aggregate and compatibility rows use content hash, size,
   order and role only, so a moved dataset is not a new run (§6.3).
8. **Ambiguous state is refused, not resolved.** Multiple attempts for one
   cell require an explicit `attempt_id`; an escaping/tampered manifest is
   reported invalid and never repaired; a live owner without an OS lock is a
   manual-diagnosis refusal.
9. **Retention.** Nothing is auto-deleted. Orphan objects from a killed
   transaction remain (content-addressed, unreferenced) and are reported as
   such; explicit cleanup belongs to M4 and is confined to a `run_id`.

---

## Rework-3 (2026-10-27) — identity input contract and test fidelity

**Product fix — what the identity hashes.** The real frames carry a *portable
logical id* (`FrameDescriptor.frame_id.logical_path`, relative to the input
root) and a separate *location* (`source_path`, the path the loader reads).
The identity builder hashed the logical id as if it were a filesystem path:
inside a run whose cwd is not the input folder this raised `FileNotFoundError`,
and the store degraded to M1 without a `run.json`. It could also have digested a
same-named decoy in the cwd. The builder now:

- hashes **`source_path`** (the real bytes), never the logical id;
- keeps the **relative logical id** in the scientific inventory, so a relocated
  dataset with preserved ids/content keeps identical fingerprints;
- fails explicitly (`_R55RunStoreIdentityError`) when `source_path` is absent or
  unreadable — no cwd fallback, no basename guessing, no hashing of a
  same-named cwd file.

The aggregate stays content-only and is derived by the store, so no absolute
path ever enters the scientific identity.

**Test corrections (no product lock semantics changed).**

- `LOCK` is never read through another handle while it is held (Windows denies
  that read even in-process); the owner record is asserted while held and the
  byte-exact placeholder/record after release.
- The live-owner test uses an explicit readiness handshake carrying the
  **actual interpreter pid**: a Windows venv launcher's `Popen.pid` is not the
  interpreter that holds the lock. Cleanup kills the process tree, and the
  orphan assertion checks the recorded pid equals the handshake pid.
- The kill matrix kills the actual writer (process group on POSIX, tree on
  Windows) and asserts the surviving owner record names that pid.
- The binary-mode test asserts the real object temp-write site
  (`tmp/<session>/obj-*.part`), the manifest temp site and the lock site
  separately; the directory fsync site is POSIX-only. The sentinel spy restores
  the native `O_BINARY` before calling the real `os.open`, so Windows I/O stays
  binary under the spy.

**Bench diagnostics.** `collect_store_diagnostics` ranks real failure text
(root-cause error lines first) and excludes source comments/code echoes; the
15-line cap can no longer hide a `identity_build` / `FileNotFoundError` behind
noise.

**Correction of the r2 report.** The r2 claim "tests no longer read `LOCK` at
all" was materially inaccurate: two direct reads remained
(`tests/test_zegrid_run_store_m2_io.py` while held and
`tests/test_zegrid_resume_m2_identity.py` while held). Both are removed.
