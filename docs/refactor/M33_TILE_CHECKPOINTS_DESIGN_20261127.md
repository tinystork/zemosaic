# M33 lot 2 — durable completed-tile checkpoints (CPU external path)

Mission `ZM-M33-TILE-CHECKPOINTS-20261127` — branch
`fix/m33-drain-durable-results`, baseline HEAD
`9cd04c93f9952d62b8dddd00f74b0c7f7d372109`.

Lot 1 (drain lifecycle) is CLOSED/ACCEPT and its uncommitted work is preserved
exactly. This lot adds **one** bounded capability: after a clean stop/crash in
the middle of an external (CPU) Cell, a NEW attempt reopens the Cell and reuses
the already reduced tiles instead of recomputing them.

No GPU integration, no publication, no store/dataset access, no commit, no push,
no new dependency, no scientific-math change. The old M33 run is protected and
nothing is claimed about its non-checkpoint tiles.

## 1. Observed defect (baseline)

`run_cell_streaming_external` writes each tile to `ExternalSpool`, reduces it in
column micro-batches, copies the interior into the Cell planes, then
`spool.discard_tile(handle)` (external_stream.py ~2701). The reduced tile arrays
are deleted with `del tile`. A stop/crash therefore loses every completed tile:
the next attempt restarts the whole Cell from tile 1. The raw `ExternalSpool` is
an *input* staging area (N × tile frames) and is deliberately discarded per
tile; it is not a resume artifact.

## 2. Acceptance contract (fixed BEFORE the edits)

1. **Reduced payload only.** A checkpoint stores the tile's reduced scientific
   planes + support + counters + executed block geometry + the exact per-tile
   rejection-sink deltas. It never stores the N-axis tile pool (no second
   archive of the raw spool; disk stays bounded by reduced cell outputs plus
   metadata).
2. **Atomic typed store.** Payload bytes are written to a temp file inside the
   checkpoint root, flushed, `fsync`ed, closed, renamed atomically; the JSON
   marker is then published atomically (temp -> fsync -> rename -> dir fsync).
   The marker carries per-array dtype/shape/offset/sha256 + the payload sha256 +
   its own sha256. A payload without a valid marker is never a PASS.
3. **Exact restore once.** Reused tiles are restored in canonical tile order and
   their counters are aggregated exactly once. `BoundedCountingRejectionSink`
   is restored with an explicit additive delta (no double count, no lost count);
   the per-tile `flush_batch()` is still called once. A dense `(N,H,W,C)`
   rejection mask is an explicitly REFUSED contract (it cannot be reconstructed
   without archiving the N-axis mask); the product path uses the counting sink.
4. **Durable before discard.** A computed tile is checkpointed after its reduce
   and BEFORE `spool.discard_tile`. A checkpoint write failure is logged and
   never fails the science (it only loses that tile's future reuse). Callback/log
   errors can never forge a commit: the final Cell commit remains the only
   authority and is owned by the parent commit path.
5. **Identity.** Reuse requires identical source order, request/science tokens,
   gauge (actual `_Phase1` arrays), backend, layout and tile geometry. The
   manifest stores per-section digests and refuses loudly on an incompatible
   contract (no silent reuse, no restart-erase). A stored/current `run_id` is
   provenance: if both are present and differ, the contract is refused.
6. **Corruption policy.** One corrupt/truncated payload, marker or hash makes
   ONLY that tile recomputable (logged); every other checkpoint is preserved and
   reused. A corrupt/invalid Cell manifest is a loud refusal (identity cannot be
   established safely). No full purge.
7. **Failure lifecycle.** Completed tile checkpoints survive a failed attempt.
   An aborted/uncommitted tile has no valid marker and is recomputed. Stale
   owner (dead process) is recoverable; a live owner is refused. The root is
   explicit, parent-owned (run output directory), outside the temp/cache
   deletion scope, and never points at corpus data.
8. **Cleanup.** Checkpoints are removed only AFTER the final durable Cell commit
   (`_cpu_durable_commit` / `_gpu_durable_commit`). If that coordination is not
   possible, the bounded reduced checkpoints are retained with documented manual
   cleanup; they are never deleted prematurely and never cause a whole-Cell
   restart.
9. **Backward compatibility.** `run_cell_streaming_external` with no checkpoint
   argument is byte-identical to the current behaviour. The product caller
   opts in through an internal run-scoped seam (env kill switch
   `ZEMOSAIC_ZEGRID_TILE_CHECKPOINTS`, default on); no GUI knob, no resume
   schema change.
10. **No recovery claim.** Tile checkpoints do not promise that the stopped old
    M33 run (tiles 74/81) is recoverable: that run wrote no checkpoints.

## 3. Design

New module `src/zemosaic/core/zegrid/tile_checkpoints.py`:

- `build_tile_identity(...)`: canonical JSON + sha256 of source identity
  (`provider.frame_ids` order and shapes), request tokens, gauge arrays, tile
  geometry, backend and `cell_id`; per-section digests are kept so a refusal can
  name what changed.
- `TileCheckpointStore`: one Cell directory, single-writer `StoreLock`
  (reused from `run_store`, Windows-safe owner record), manifest
  `checkpoint.json` (atomic), tile files
  `tiles/<index>/payload.bin` + `commit.json`.
- `save_tile(index, ...)`: exact dtype/shape validation, bounded payload,
  fsync, atomic rename, marker last.
- `load_tile(index, ...)`: containment + symlink refusal, marker sha, payload
  sha, per-array sha/dtype/shape/offset; returns `None` when absent, raises
  `TileCheckpointCorrupt` when the tile is unusable.
- `apply_sink_delta(sink, delta)`: explicit additive restore on
  `BoundedCountingRejectionSink`; any other sink is refused.
- `cleanup_cell(root, cell_id)`: post-commit best-effort removal, skipping a
  live owner.

Integration:

- `external_stream.run_cell_streaming_external(..., checkpoint_root=None,
  checkpoint_run_id=None, checkpoint_policy="recompute_corrupt",
  checkpoint_out=None)`. Defaults `None` keep the historical path.
- `zemosaic_zegrid_mode._run_cell_external` forwards the new keywords from the
  external task payload.
- `_build_execution_bundle` adds explicit `checkpoint_root` / `checkpoint_run_id`
  to the external task info (root = `<output_dir>/.zegrid-tile-checkpoints`,
  run id = frozen store run id when a resume is in progress, else `None`).
- `_cpu_durable_commit` / `_gpu_durable_commit` call the checkpoint cleanup only
  after `require_commit` proved the final commit durable.

## 4. Tests plan (bounded, real external pipeline)

- uninterrupted reference vs checkpointed run: all planes/counters/sink stats
  bit-exact; checkpoint payload contains reduced tiles only.
- deterministic crash AFTER tile K's marker but BEFORE tile K+1: new attempt
  reopens, `reduce_spooled_tile` is called only for the remaining tiles, result
  bit-exact, sink counted exactly once.
- corrupt payload / truncated payload / removed or torn marker / hash mismatch:
  only that tile is recomputed, the rest kept.
- incompatible gauge / source order / request / tile geometry: loud refusal,
  files preserved.
- live owner refused, stale owner recovered, same-process double lock refused.
- dense mask + checkpoints refused loudly.
- product caller `_run_cell_external` integration (not only the low-level
  helper), with a stable checkpoint binding.
- public default unchanged (no checkpoint files when no argument is passed).

## 5. Out of scope

GPU/VRAM integration, publication/beta, old M33 tile salvage, global resume
schema changes, GUI knobs, scientific math, dependencies, commits/pushes.

## 6. Implementation status (2026-11-27)

Implemented:

- `src/zemosaic/core/zegrid/tile_checkpoints.py` (new): typed identity,
  atomic reduced-tile store, lock/owner handling, corruption policy, sink
  delta restore, post-commit cleanup.
- `src/zemosaic/core/zegrid/external_stream.py`: opt-in `checkpoint_root` /
  `checkpoint_run_id` / `checkpoint_policy` / `checkpoint_out`; reuse before
  write/reduce; checkpoint after reduce and before spool discard; no cleanup
  inside the executor.
- `src/zemosaic/core/zegrid/rejection_sink.py`: additive
  `restore_tile_delta` (exact once, bounded, validated).
- `src/zemosaic/zemosaic_zegrid_mode.py`: product seam
  (`_run_cell_external` kwargs, external task info, internal env switch,
  post-commit cleanup hooks).
- `tests/test_zegrid_m33_tile_checkpoints.py`: 12 bounded tests (bit-exact,
  crash/resume skip, corrupt/torn/hash, incompatible identity, live/stale
  owner, dense-mask refusal, spawn close/reopen, post-commit cleanup, product
  `_run_cell_external` integration).

Validation summary is in the mission report; no commit/push was performed.

## 7. Rework-1 (Nono r0 findings)
- **H1 (source binding):** `build_tile_identity` now requires an explicit
  `source_binding_digest` and stores it in a `source_binding` section. The
  product caller builds it from the existing frozen-identity frame rows
  (`content_sha256`/`nbytes`/order), each frame's serialized WCS/transform
  sha256 and the canvas WCS/shape; `_run_cell_external` adds the per-Cell patch
  WCS/shape/halo/bounds. Unknown binding is fail-closed: the product DISABLES
  checkpointing with a loud reason, the public executor REFUSES activation.
  The file hashing is the same bounded streaming hash already used by the
  durable run identity (one pass, cached per run; disclosed disk I/O).
- **H2 (batch):** the identity now includes the requested `column_batch` and
  the resolved rows/cols/positions; a changed batch is an explicit
  `TileCheckpointIncompatible` refusal.
- **H3 (lock lifecycle):** result allocations and the spool constructor run
  BEFORE the checkpoint lock is opened; the main `try/except/finally` owns the
  lock through tile work, spool manifest and success bookkeeping. The finalizer
  runs after the lock is released. `MemoryError`/constructor/finalizer failures
  cannot leak the lock into the next attempt.
- **H4 (backend):** this lot is CPU-only; a requested non-CPU backend is refused
  for checkpoint activation, and the product disables checkpointing for any
  GPU-shaped task. No false GPU provenance is persisted for CPU reductions.
- **H5 (multiprocess witness):** a spawn product witness (process A aborts
  after 2 tiles; process B, new PID, resumes only the remaining tiles) checks
  the 7 planes/counters bit-exactly against an uninterrupted product reference.
- **N1 (`complete` semantics):** the in-memory flag is `runtime_complete` and
  is explicitly NOT durable (`durable_complete=False`). Telemetry now reports
  `saved_tiles`, `write_failures`, `checkpointed_tiles` and
  `not_checkpointed_tiles`; a non-fatal checkpoint write failure leaves the
  science exact but never claims every tile is resumable.
- **Run-id binding:** a stored run id and the current run id must match exactly;
  a durable run id and a store-less `None` are no longer interchangeable. The
  product binds the opened durable store's run id into the already-built
  external task payloads before dispatch.

## 8. Rework-2 (Nono r1/Junior final hardening)

- **Unconditional lock ownership:** the checkpoint `Store.open()` and the
  immediately following info-dict/update/logger are now INSIDE the one existing
  `try/except/finally`; the `finally` closes the store on every exit. A
  `checkpoint_out` update failure right after open is covered by the same
  owner and cannot leak the lock.
- **Strict WCS proof:** the source-binding helper no longer hashes an empty
  `""` WCS fallback. A frame without a serialized WCS/transform (or without a
  valid `shape_hw`) disables the binding with an explicit reason, and a Cell
  patch without `patch_wcs_header` disables checkpointing for that Cell with a
  loud warning instead of hashing an empty proof.
- **Cache lifetime:** `_checkpoint_source_binding`'s cache key is the object
  identity of the frozen descriptor list/canvas; those objects must remain
  immutable for the run (a mutated list/canvas would make the cached proof
  stale). This is documented at the helper.
- **Contract epoch:** `SOURCE_BINDING_SCHEMA` is the explicit source-binding
  contract epoch. The current identity hashes source content + WCS/transform +
  canvas/patch + frozen gauge/request; it does NOT claim that a decoder/policy
  version is hashed. Any future GPU/decoder policy change must bump the epoch
  explicitly rather than rely on an implicit hash.
- **Store-less vs durable:** a stored run id and the current run id must match
  exactly; `None` (store-less) and a durable run id are deliberately
  incompatible. This closes the nullable-run-id loophole.
- **Boundary proof:** a real `_build_execution_bundle` -> external task-info
  test asserts the source binding is built from actual file bytes/WCS (no
  digest injected into `_run_cell_external`) and that the frozen run id is
  attached to the already-built task payload. A real forced-external product
  run (no injected digest) asserts the persisted checkpoint manifest carries
  the source-binding digest reported at execution time, recomputed frame
  hashes match the actual corpus bytes/WCS, and the execution section is
  persisted.
