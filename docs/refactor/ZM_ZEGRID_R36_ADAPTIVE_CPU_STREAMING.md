# ZM-ZEGRID-R36 — RAM-adaptive CPU streaming tile + joint-planner feasibility

Mission: `ZM-ZEGRID-R36` (implementation). Repo `zemosaic`, branch `beta`.

## BASE

- Branch: `beta`
- Base HEAD: `38d9c5f84b02b849118fe3ec7b719041550fb866` (`== origin/beta` at start)
- This base already contains the accepted/published SDS chain C1–C5; SDS and
  `core/canonical_streaming.py` are NOT modified by R36.
- Frozen policy inputs: `min_patch_area_px=50_000`, `max_halo_overhead=0.50`,
  `RAM_SAFETY_FRACTION=0.8`.

## OLD CONTRACT (pre-R36)

1. **Layout admissibility** (mode-aware planner, `_choose_layout_mode_aware`):
   a candidate layout was admissible iff its worst-cell *cheaper* bound
   `min(inmem, stream@128) <= ram_budget`. The CPU streaming tile was a single
   constant, `STREAM_TILE_SIZE = 128`, everywhere.
2. **Joint mode/concurrency planner** (`parallel.plan_cell_concurrency`): a mode
   whose worst-cell bound exceeded the RAM budget could still be reported as a
   valid plan because `cells_in_flight()` clamps a zero RAM quotient to `1`
   (the "clamp-to-1 rescue"). The plan could then return `inmem` even after the
   in-memory bound was known to exceed the budget.
3. **GPU→CPU fallback**: both the pre-run VRAM-infeasible path and the classified
   runtime-error rerun re-coded the CPU tile to `STREAM_TILE_SIZE` (128),
   regardless of what the CPU could actually hold.

## NEW CONTRACT (R36)

1. **Deterministic CPU streaming-tile ladder** (performance→economy):
   `CPU_STREAMING_TILE_LADDER = (128, 112, 96, 80, 64, 48, 32)`.
   `128` = preferred; `32` = documented **performance floor** (NEVER a science
   floor). R36 never goes below 32.
2. **Adaptive tile selection.** For each floor-feasible layout, the planner
   computes the in-memory worst-cell bound and, for EACH ladder tile, the
   worst-cell streaming bound; it then selects the **largest** tile whose worst
   streaming bound fits `ram_budget`.
3. **Admissibility.** A layout is admissible iff the in-memory bound fits OR
   some (>= minimum) ladder tile fits. Auto picks the first/coarsest admissible
   candidate (unchanged scan order + early stop).
4. **Pinned.** `(nx, ny)` is preserved EXACTLY; only the CPU streaming tile
   adapts. If neither in-memory nor the minimum tile (32) fits, a loud
   `LayoutInfeasible` states the layout, the minimum tile, the predicted bound,
   the budget, and that the next step would be a deeper **disk-backed streaming**
   executor (not implemented in R36 / no R37 promise).
5. **Joint planner feasibility.** A mode with `max_bound > ram_budget` is
   **INFEASIBLE**: it is excluded from the choice, is NEVER rescued by
   `cells_in_flight == 1`, is never chosen, and when no mode fits the planner
   raises an explicit `parallel.ModeInfeasible`. No-budget (`None`/0) behaviour
   is preserved (every mode feasible).
6. **Propagation.** The selected CPU tile is propagated verbatim to the per-cell
   stream estimates, the joint planner, the workers (`_init_stack_worker` →
   `_STACK_TILE_SIZE` → `_run_cell_stream`), the executor call
   (`run_canonical_stack_streaming(tile_size=...)`) and every GPU→CPU fallback.
7. **GPU.** `choose_gpu_tile_size()` and the VRAM policy are untouched. A GPU
   success uses its GPU tile; a GPU→CPU degradation always uses the selected CPU
   tile. The manifest distinguishes `cpu_streaming_tile` and `gpu_streaming_tile`.

## MEMORY MODEL

R36 reuses the FROZEN estimator exclusively: `core/zegrid/streaming.py::
estimate_streaming_peak_kib()` via `zemosaic_zegrid_mode._estimate_streaming_bytes()`.
No second estimator, no arbitrary multiplier. The in-memory bound remains
`auto_layout.FITTED_MEMORY_MODEL.predict_bound_bytes()`.

Note: the streaming estimator is calibrated at tiles 128/256, so its per-tile
fixed-overhead term (`TILE_FIXED_OVERHEAD_KIB=2560`/tile) makes small tiles
*more* expensive for small patches and can make them *cheaper* for large-N large
patches — the ladder therefore only helps in the regime the estimator models,
and the selected tile can legitimately be `128` on a comfortable machine.

## PROPAGATION (layout → execution)

```
layer    _choose_layout_mode_aware()
           └─ _scan_layout_mode_aware()   # 1 membership pass:
                cheapest-bound (unchanged) + inmem_worst + stream_worst_by_tile
           └─ _select_cpu_streaming_tile() -> (selected, trials, reason)
           └─ _cpu_streaming_block() -> layout["cpu_streaming"]
                {adaptive, preferred_tile_size, selected_tile_size,
                 minimum_tile_size, selection_reason, predicted_peak_bytes,
                 ram_budget_bytes, inmem_bound_bytes, stream_bound_bytes,
                 estimate_tile_size, ladder, trials}
per-cell   _estimate_streaming_bytes(n, patch_hw, cpu_streaming_tile)
joint      parallel.plan_cell_concurrency(cell_bound_pairs, cpu, budget, ...)
worker     _init_stack_worker(..., tile_size=cpu_streaming_tile, ...)  -> _STACK_TILE_SIZE
stream     _run_cell_stream(..., tile_size=_STACK_TILE_SIZE, ...)
executor   run_canonical_stack_streaming(provider, request, tile_size=...)
GPU->CPU   _run_cell_batch(science_config, cpu_streaming_tile, gpu_attempt=False)
```

`layout["predicted_bound_bytes"]` is kept with its pre-R36 meaning (the
cheaper-mode bound at `STREAM_TILE_SIZE`) for backward/schema compatibility;
the R36 budgeted bound is `layout["cpu_streaming"]["predicted_peak_bytes"]`
(the worst-cell streaming bound at the selected tile).

## CONCURRENCY

`plan_cell_concurrency` now gates each mode on `worst_bound <= ram_budget`:

- `max_bound > budget` ⇒ candidate `feasible=False`, `infeasible_reason` set,
  `cells_in_flight=0`, `makespan_rel=None`; the mode is excluded from the choice.
- in-memory infeasible + streaming feasible ⇒ `mode="stream"` (never inmem).
- neither feasible ⇒ `ModeInfeasible` (explicit; no silent serial plan).
- `ram_budget` falsy ⇒ every mode feasible (historical behaviour preserved).

The `cells_in_flight()` seam itself is unchanged (its zero-quotient→1 clamp
remains unit-tested as a pure function); R36 forbids the *planner* from treating
that clamp as feasibility.

## GPU

- `core/zegrid/gpu.py::choose_gpu_tile_size()` NOT modified; VRAM policy NOT modified.
- GPU success: `_run_cell_batch(science_config, gpu_tile_size, gpu_attempt=True)`.
- Pre-run VRAM-infeasible (`_gpu_ts is None`): `gpu_used=False`, then the CPU
  batch runs with `cpu_streaming_tile`.
- Classified runtime GPU error: one-shot whole-batch CPU rerun with
  `cpu_streaming_tile` (never a re-coded 128).
- Manifest: `gpu.cpu_streaming_tile` and `gpu.gpu_streaming_tile`.

## SCIENCE

- No science/geometry/weights/rejection/combine/support/feather change.
- `tests/test_zegrid_r36_adaptive_cpu_tile.py` adds an explicit ladder
  comparison 128/96/64/32 of the canonical streaming executor against the
  in-memory engine: bit-exact science, `estimator_weight_sum`, `support_w1/w2`,
  `n_eff_support`, `valid_mask`, `surviving_sample_count`, `rejection_mask`,
  rejection counters/method and `provenance`. The existing SCI-05 gates are
  preserved.

## TINYDEBIAN WITNESS (planning only — no Caldwell long run)

Read-only inputs: `/home/tristan/caldwell11/lights organized/EQ/IRCUT`,
`/home/tristan/caldwell11/zegrid_out/R34F/{zegrid_run.log,run_config_snapshot.json}`.
Scratch: `.a2a-reports/ZM-ZEGRID-R36-scratch/` (nothing written under
`/home/tristan/caldwell11`).

- **OLD (frozen R34F, authoritative):** 653 plan entries, layout `24x14`,
  tile `128`, cheaper bound `2836.2 MiB`, budget `2514.4 MiB` → **FAIL**
  (`zegrid_run.log` lines 16–35).
- **NEW population (current inventory, R35A):** `discovered=816`,
  `selection_applied=false`, `included=816`, `rejected_wcs=0`; canvas `4117x4377`.
  The population differs (653 → 816) because the input inventory grew; the NEW
  witness is NOT a strictly identical comparison to the OLD frozen 653 run.
- **NEW @ the SAME budget (2514.4 MiB):** **FIT** — layout `15x9` (135 cells,
  maxN 571), CPU streaming tile **48**, stream worst bound `2351.1 MiB` ≤ budget
  (in-memory worst `13383.1 MiB`). Tiles `128/112/96/80/64` exceed the budget.
- **NEW @ current machine budget (`avail*0.8 = 1329.2 MiB`):** **FIT** — layout
  `24x15` (360 cells, maxN 539), CPU tile **32**, stream bound `1260.2 MiB` ≤ budget
  (in-memory worst `5260.9 MiB`).

Observation to be explicit about: with the R36 admissibility rule, the coarsest
admissible layout can be *coarser* than pre-R36 (Caldwell: `15x9` vs the pre-R36
finest `24x14`), because the auto path keeps its first/coarsest rule while the
admissibility predicate widened. This is the frozen policy as specified; it is
flagged under OPEN ISSUES because it changes the production `(nx, ny)`.

## PERFORMANCE (micro-witness)

Synthetic aligned cell, `N=8`, patch `220x200`, canonical streaming executor,
best/median of 3 runs (no multiprocessing/cache/prefetch/reprojection changes):

| tile | min (s) | median (s) |
|------|---------|------------|
| 128  | 0.1479  | 0.149 |
| 64   | 0.1792  | 0.181 |

`tile64` ≈ **1.21x** slower than `tile128`. A slowdown is accepted (RAM safety
buys depth); no new parallelism was added.

## TESTS

Commands actually executed (repo root, `python -m pytest`):

- `tests/test_zegrid_layout_performance_r28.py tests/test_zegrid_layout_observability_r27.py tests/test_zegrid_r20_percell_parallel.py tests/test_zegrid_r22_gpu_contract.py` → **49 passed**.
- `tests/test_zegrid_r36_adaptive_cpu_tile.py -k "not plan_equals and not manifest_cpu and not gpu_"` → **16 passed** (planner + SCI-05 ladder).
- `tests/test_zegrid_r36_adaptive_cpu_tile.py -k "manifest_cpu"` → **1 passed**.
- `tests/test_zegrid_r36_adaptive_cpu_tile.py -k "plan_equals or manifest_cpu or gpu_"` → **3 passed, 1 failed** (a buggy assertion in the manifest test itself, since fixed and re-run green).
- `tests/test_zegrid_r6_file_provider.py tests/test_sci05_canonical_streaming.py tests/test_zegrid_r4_auto_layout.py -m "not slow"` → **66 passed, 1 skipped** (pre-existing env skip).
- Full `tests/test_zegrid_r36_adaptive_cpu_tile.py` → **20 passed**.

Pre-existing environmental failure noted by the mission
(`test_sci05_canonical_streaming::TestValidation::test_gpu_backend_raises`,
because CuPy+GPU are present) is unrelated to R36 and was not forced green.

## OPEN ISSUES

1. **Layout selection can coarsen** (see WITNESS). A layout whose pre-R36
   cheaper bound exceeded the budget can now be admissible via a smaller tile,
   so the auto path can choose a *coarser* `(nx, ny)`. This is per the frozen
   policy but is an observable production-geometry change. If the intent was to
   keep the pre-R36 layout and only shrink the tile, the admissibility widening
   must be scoped so the coarse→fine scan keeps the pre-R36 predicate.
2. **`layout.predicted_bound_bytes` vs budget.** For a layout admitted only via
   a small streaming tile, the legacy `predicted_bound_bytes` (cheaper@128) can
   exceed the budget while the run succeeds. The R36 budgeted bound lives in
   `cpu_streaming.predicted_peak_bytes`; the legacy key is retained for
   R28/schema compatibility.
3. **Estimator extrapolation.** The ladder reuses the frozen estimator calibrated
   at 128/256; tiles below 128 are extrapolated (its per-tile fixed overhead
   dominates small patches). No new calibration was added (out of scope).
4. **`32` is a performance floor**, not a science floor; a machine needing a
   smaller tile is not served by R36.

## NEXT STEP

If tile `32` still cannot fit the finest floor-feasible layout, the next step is
a deeper **disk-backed streaming** executor (R37) — not implemented, not
promised here. R36 stops at the documented tile-32 performance floor with a
loud `LayoutInfeasible`.
