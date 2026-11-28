# M33 lot 3 — external GPU product integration (design)

Mission `ZM-M33-EXTERNAL-GPU-INTEGRATION-20261127` — branch
`fix/m33-drain-durable-results`, baseline HEAD
`9cd04c93f9952d62b8dddd00f74b0c7f7d372109` (lots 1 and 2 uncommitted/accepted
are preserved). This lot integrates the EXISTING external GPU reducer into the
product external working set. It does not add a dependency, a GUI knob, a
science change, a reprojection/CFA change, a prefetch optimisation or a
whole-tile resident GPU formula.

## 1. Objective

Execute the SAME two-producer CPU preparation + disk spool with the EXISTING
`reduce_spooled_tile(..., backend="gpu")` reduction, where a block is
`N x positions` and ALWAYS contains ALL N contributors of every pixel. The
product dispatch must honour the existing GPU request, preserve the CPU path,
and keep fresh/frozen identity truthful. A GPU that is unavailable, unknown,
too small, non-ACTIVE, K>1 or CPU-frozen is declined BEFORE execution and the
CPU external path continues byte-for-byte.

## 2. Known model and its limits

The resident whole-tile planner (`gpu.choose_gpu_tile_size`,
`gpu.gpu_tile_workspace_bytes`) prices `N x extended_tile_area` frames. It is
deliberately NOT applied to the external class: the external reducer keeps the
spool on disk and uploads ONE column block at a time.

New pure module `core/zegrid/gpu_external.py`:

* `GPU_EXTERNAL_DEVICE_BYTES_PER_POSITION_FRAME = 96` B — conservative upper
  bound audited from the real block lifetimes in `reduce_spooled_tile`
  (`orig2` f64, survivor/degenerate/new bool planes, masked/w_contrib/product
  f64, median sort copy). The observed simultaneous live set is ~60 B/cell;
  96 B covers a median sort duplicate and allocator rounding.
* `GPU_EXTERNAL_FIXED_POSITION_BYTES = 512` B — the same `(positions,)`
  quantile/count/index temporary bound as the host model.
* `GPU_EXTERNAL_FIXED_RESIDUAL_BYTES = 16 MiB` — an EXPLICIT declared reserve
  for the CUDA context/libraries, allocator block rounding and CuPy pool
  retention. It is NOT a measured or proved device peak.
* Budget = `90 % of CURRENT free VRAM` (the single product policy,
  `gpu.GPU_VRAM_SAFETY_PERCENT`) minus the fixed residual. A budget below the
  accepted floor, an unavailable device or an unknown free figure declines.
  A known TOTAL is NEVER substituted for an unknown FREE figure (another process
  may own the card). Malformed probe fields (bool/float/str/negative,
  free>total, non-bool `available`) decline as `probe_invalid` without raising;
  `numpy.int64` is accepted and normalized to a plain JSON int.

The batch is chosen by integer inversion over
`positions * (N * 96 + 512)`, clamped to the frozen interior surface and NEVER
grown above the host-admitted batch. One pixel (`channels` positions) is the
floor; if it does not fit, the GPU is declined. The model is an upper bound, not
a promised 8 GiB fit. The runtime executor keeps the accepted classified
CUDA/OOM sticky CPU replay; residual VRAM pressure therefore degrades truthfully
to CPU rather than failing the run.

## 3. Product seams

Dispatch sites inspected before the product edits (there is ONE external
selection site, so no global scheduler rewrite was needed):

1. `_build_execution_bundle`: `_force_external` early branch (resident GPU rung
   suppressed), the single `_select_external_fallback` call, the external
   pseudo-plan, `_working_set_block`/`external_contract`, `cell_tasks`.
2. ACTIVE run scope: the `_r55_gpu_active_requested` resident attempt, the
   bounded retry's `_r55_gpu_attempt`, `_r55_active_external_decide` /
   `_r55_active_apply` / `_r55_active_admit_external`, `_effective_run_backend`,
   `_r55_execution_contract_from_bundle`, `_r55_run_store_working_set`.
3. Per-Cell: `_cell_task` slot 7 contract → `_run_cell_worker` external branch →
   `_run_cell_external` → `run_cell_streaming_external`; per-Cell record
   `backend_used` and the `external` storage block.
4. Executor/checkpoints: duplicate CPU-only checkpoint gates, tile reuse/save,
   `TileCheckpointStore.save_tile/load_tile`, `gpu_fallback_out`,
   `checkpoint_out`.

* `core/zegrid/gpu_external.py` — pure typed admission record
  (`requested/selected/effective`, reason, batch, device model, probe, owner).
* `zemosaic_zegrid_mode._external_gpu_admission` — product helper over the REAL
  external selection. It may only SHRINK the staging batch; it re-prices the
  host peak with the same `estimate_external_working_set` model and refuses if
  the re-priced peak would exceed the admitted phase budget.
* `_build_execution_bundle` — only on the ACTIVE scheduler, after the external
  pseudo-plan exists, evaluates the admission once with the fresh probe. The
  external contract carries `backend` (`gpu`/`cpu`) and the full `gpu` record.
  The resident GPU tile rung is never entered for the external class.
* `_r55_active_apply` — preserves `gpu_used=True` and `gpu_serialized=True` for
  an admitted external GPU bundle; still K=1 and prefetch-free.
* `_r55_execution_contract_from_bundle` — an external GPU bundle reports
  `backend="gpu"` with `tile_px` = frozen spool tile and
  `parity_qualified=False` (no legacy whole-tile parity claim).
* `_r55_run_store_working_set` — an external GPU run stays
  `external_spool` (tile + batch). Describing it as `ram_tile` would be false.
* Point-of-use revalidation — AFTER the CPU spool write / producer join and
  IMMEDIATELY BEFORE the first actual GPU reduce branch of each Cell (never for
  a checkpoint-only Cell), the executor takes a FRESH probe and validates the
  SAME frozen block batch + declared residual. The already-written staging tile
  is reduced on CPU if the check declines; no GPU allocation is attempted.
  Unknown/insufficient/current-failure => sticky CPU execution for the remaining
  uncomputed tiles, frozen requested identity/batch unchanged. The contract
  batch is never adapted here.
* `_run_cell_external` — takes the backend from the frozen task contract
  (request → bundle → executor), never from a stale process global. It forwards
  `backend` and the GPU admission record to the executor and publishes
  requested/selected/effective/sticky per Cell.
* Per-Cell record `backend_used` is the executor's EFFECTIVE backend; a GPU Cell
  that fell back is recorded as CPU.

## 4. Checkpoints and identity

* The checkpoint identity still binds the REQUESTED backend (`cpu`/`gpu`) and
  the source/request/gauge/geometry/batch. CPU identity/markers are unchanged
  (no `backend` key on CPU markers), so the accepted CPU epoch is not
  invalidated.
* A GPU tile marker now carries additive
  `backend: {requested, effective}`. A marker without the field is a legacy CPU
  checkpoint (identity proves CPU).
* A reused tile with effective `cpu` restores the sticky CPU fallback for the
  rest of the Cell. A VALID historical GPU tile may still be restored AFTER a
  CPU-recomputed hole (e.g. tile 1 corrupt, tiles 2..N previous GPU
  checkpoints): historical per-tile provenance is distinct from the CURRENT
  sticky CPU state, so the verified tile is restored and NEW work stays CPU.
  Requested-backend mismatches and malformed markers are still refused.
  Counters/sink deltas/geometry remain exact-once; mixed counts and final
  effective CPU are published truthfully.
* `checkpoint_out` and `gpu_fallback_out` are updated progress-style, so a
  crash mid-Cell still reports truthful saved/reused/effective/sticky counts.
* No checkpoints are silently disabled for GPU when the combined integration is
  advertised. Write failures remain non-fatal to science and truthfully reduce
  resumability.

## 5. Frozen/requested/effective contract

* Fresh GPU run: identity backend `gpu`, working set `external_spool`.
* CPU-frozen run: the external GPU admission declines with
  `frozen_backend_cpu`; the CPU external path continues and the CPU checkpoint
  epoch is unchanged. A CPU-frozen run is never resumed as GPU.
* GPU-frozen run: the external GPU admission must succeed; if it cannot, the
  single effective-backend resolver raises the typed frozen-capacity refusal.
* A GPU Cell that fell back stores requested `gpu`/effective `cpu`; a resume
  restores that sticky state instead of marking reused tiles GPU.
* Legacy GPU stream path and its parity/qualification gates are unchanged
  (regressions executed).

## 6. Tests

Bounded witnesses (`tests/test_zegrid_m33_external_gpu_integration.py`, 11):

* pure model: all-N pricing, shrink under pressure, one-pixel floor;
  declines on unknown free (even with a known total), malformed probe
  (`probe_invalid`, never raises, `numpy.int64` normalized), below floor, K>1,
  non-ACTIVE, CPU-frozen, no request; point-of-use fit/decline;
* REAL `_select_external_fallback` → `_external_gpu_admission` provenance and
  host re-pricing; REAL `_build_execution_bundle` admits/declines and the
  per-task contract carries the backend;
* REAL external executor dispatched to `backend='gpu'` through an EXPLICIT CPU
  mock (`_get_xp -> numpy`, never a real-GPU claim): checkpointed run, injected
  classified OOM on tile 2, crash, NEW attempt restores sticky CPU with no GPU
  call, seven planes/counters bit-exact vs CPU; point-of-use probe is observed
  AFTER the write and GPU reduce count is ZERO on unknown/low VRAM; an all-reuse
  Cell performs ZERO probes/reductions; adversarial mixed historical GPU after a
  CPU-recomputed hole restores honestly; programming error propagates; CPU
  marker shape unchanged;
* REAL `_build_execution_bundle` passes the REAL plan K (captured=1) and refuses
  a simulated K=2; REAL builder -> `_r55_build_run_store_identity` -> RunCellStore
  close/reopen proves the frozen external GPU identity is compatible (CPU-only,
  no GPU execution).

Real hardware (tiny, optional/authorised): MX150 2 GiB + CuPy 14.1.1, synthetic
11x17x19, tile 8, column_batch 64, finite bounded run. Attempt A: real GPU,
9/9 tiles GPU. Attempt B on the reused checkpoint path: 9/9 reused, 0 GPU
reductions. REWORK-1 comparison is EXACT: same dtype/shape, identical NaN
positions, identical raw bytes for every non-NaN element (catches zero signs)
and exact counters, CPU vs GPU and GPU vs reused. Label: direct executor parity,
not a full-product run. Sampled VRAM only; no performance/memory qualification.

Regression batches: lot1 drain, lot2 checkpoints, external wiring, external
GPU fallback/parity, external lifetimes/phase log/batch truth/phase budget/
model measure, M4 durability, source plumbing, M33 P1 wave dispatch.

## 7. Out of scope / limits

No GPU reprojection/CFA, no block prefetch, no whole-tile stream formula for
external blocks, no global scheduler rewrite, no new GUI knob, no speed claim,
no proved device peak, no 8 GiB promise. Known baseline failures
(M6/resume envelope and legacy GPU host-RAM tests) are tracked, not hidden.

## 8. Rework-3 — shared verified rows and checkpoint-root ownership

Two confirmed integration regressions were fixed without touching science,
admission, GPU backend semantics or checkpoint identity:

1. **One frame-content hash pass per product run.** `_run_single` computes the
   frozen `_r55_frame_input_rows(descs)` ONCE (import verification, run
   identity and `_checkpoint_source_binding` all consume that SAME list).
   `_checkpoint_source_binding(..., frame_rows=...)` validates the row count and
   required fields before use; a short/malformed list fails closed with a loud
   reason. `None` keeps direct helper callers working (they hash once locally).
   The cache lifetime is ONE `_run_single` call, so a source change between runs
   cannot reuse a stale proof; content/WCS/shape binding is unchanged.
2. **Owned checkpoint residue vs publication.** `cleanup_cell` removes the
   owned checkpoint root when it becomes EMPTY after the final commit, so a
   COMPLETE run leaves no `.zegrid-tile-checkpoints` residue. If residue must
   remain (incomplete/failed run), `run_publication.detect_previous_publication`
   recognises it only through `tile_checkpoints.inspect_owned_checkpoint_root`:
   plain directory (never a symlink), only plain Cell dirs with a valid typed
   manifest (schema/version/identity/marker hash), only product tile/tmp names,
   no foreign entries, no live owner. A forged/foreign/malformed entry, unknown
   file or link stays foreign and the existing refusal still protects it. This
   is an ownership proof, never a name-only ignore list. Explicit
   `--overwrite` may replace valid owned residue; active/foreign content is
   still refused.

Adversarial tests cover valid residue, unknown cell entry, malformed manifest,
symlink root and live owner; the existing CLI overwrite test and gauge-import
hash-count test pass unchanged.

## 9. Rework-4 — ownership inspector hardening

- A parsed tile `commit.json` must be a JSON OBJECT; any scalar/list/null (or
  malformed JSON) is FOREIGN, never an exception.
- `tiles` and `tmp` are checked for `is_symlink()` BEFORE `exists()`, so a
  dangling or real-target link is refused without traversal/mutation.
- When present, `LOCK`/`OWNER.json` must be plain regular files (no symlink,
  including dangling, no directory). A present malformed `OWNER.json` makes
  ownership unprovable (foreign); a MISSING owner is legitimate crash residue
  and remains owned if the rest is valid. Live owners remain protected.
- Scope note: this inspector proves STRUCTURAL ownership only. It does not prove
  that checkpoint payloads are reusable science, and it is not cryptographic
  provenance or defense against a fully forged but self-consistent manifest.
  Tile/bounds/counters typing is validated by the checkpoint store at load.
