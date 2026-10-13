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
   in-memory bound was known to exceed the budget. Two further bugs (fixed in
   rework-1, see below): the inmem-only branch formatted an absent stream
   makespan (`TypeError`), and the GPU branch ignored the RAM gate entirely.
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

### Rework-1 corrections

8. **F1 — in-memory-only is a VALID plan.** When in-memory fits and streaming
   does not, the planner returns `mode=inmem`, a stream candidate with
   `feasible=False`/`cells_in_flight=0`/`infeasible_reason`, and an explicit
   reason — never a `TypeError` on the absent stream makespan.
9. **F2 — the GPU branch is RAM-bounded.** The GPU path still considers only its
   single stream mode and keeps the existing single-owner policy, but that stream
   is now `feasible = (no budget) or (max_stream <= ram_budget)`; GPU candidates
   carry coherent `feasible`/`infeasible_reason`. When the GPU stream is
   RAM-infeasible the planner refuses to pretend it is executable (raises
   `ModeInfeasible`, candidates attached for audit) and the PRODUCT path degrades
   LOUDLY to exact CPU BEFORE execution, then re-plans with `gpu_backend=False`
   (inmem if only inmem fits, stream if only stream fits, `ModeInfeasible` if
   neither). `science_config`, `gpu_used`, `gpu_ctx` and the manifest/log record
   the real pre-execution fallback; the GPU tile/VRAM computation is untouched.
10. **F3 — the predicted bound is the SELECTED bound.**
    `layout.predicted_bound_bytes` is now `min` over the ADMITTED modes' worst-cell
    bounds (in-memory if it fits and/or the selected streaming tile if it fits), so
    it is `<= ram_budget` for an ACCEPTED layout. The legacy preferred-tile (128)
    streaming bound is kept SEPARATELY (and NOT called the plan's predicted bound)
    in `cpu_streaming.preferred_stream_bound_bytes`; the CHOSEN log prints
    `predicted bound ... ; CPU streaming tile <t> (stream@<t>=...)`.
11. **F-M1 (rework-2) — the GPU gate uses the GPU's OWN tile.** The VRAM-aware
    GPU tile is determined BEFORE the joint plan and the GPU RAM-feasibility gate
    uses the published estimator at that EFFECTIVE tile. A CPU@48 fit does NOT
    imply a GPU@128 fit (the canonical GPU materialises host-side buffers that
    scale with the real tile). When the GPU's effective-tile host bound exceeds the
    budget the GPU is refused BEFORE execution (zero GPU attempt) and the CPU is
    re-planned and executed at the adaptive `cpu_streaming_tile`. `per_cell_footprint`
    and the GPU candidate reflect the GPU-tile bound when the GPU actually runs.

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

`layout["predicted_bound_bytes"]` is the bound the layout was ACTUALLY admitted
with: `min` over the admitted modes' worst-cell bounds (in-memory if it fits
and/or the selected streaming tile if it fits), so it is `<= ram_budget` for an
ACCEPTED layout (rework-1 F3). The legacy preferred-tile (128) streaming bound is
kept separately as `layout["cpu_streaming"]["preferred_stream_bound_bytes"]` and
is NOT the plan's predicted bound. The R36 budgeted streaming bound is still
`layout["cpu_streaming"]["predicted_peak_bytes"]`.

## CONCURRENCY

`plan_cell_concurrency` now gates each mode on `worst_bound <= ram_budget`:

- `max_bound > budget` ⇒ candidate `feasible=False`, `infeasible_reason` set,
  `cells_in_flight=0`, `makespan_rel=None`; the mode is excluded from the choice.
- in-memory infeasible + streaming feasible ⇒ `mode="stream"` (never inmem).
- in-memory feasible + streaming infeasible ⇒ `mode="inmem"` (VALID; F1).
- neither feasible ⇒ `ModeInfeasible` (explicit; no silent serial plan).
- `ram_budget` falsy ⇒ every mode feasible (historical behaviour preserved).

GPU (F2): the GPU branch considers only its stream mode, marked
`feasible = (no budget) or (max_stream <= ram_budget)` with coherent
`feasible`/`infeasible_reason`; a RAM-infeasible GPU stream raises `ModeInfeasible`
(candidates attached) so the product path degrades to exact CPU BEFORE execution.

The `cells_in_flight()` seam itself is unchanged (its zero-quotient→1 clamp
remains unit-tested as a pure function); R36 forbids the *planner* from treating
that clamp as feasibility, and its docstring now says so explicitly.

## GPU

- `core/zegrid/gpu.py::choose_gpu_tile_size()` NOT modified; VRAM policy NOT modified.
- **F-M1 (rework-2):** the VRAM-aware GPU tile is determined BEFORE the joint
  plan, and the GPU RAM-feasibility gate is evaluated with the published estimator
  at the GPU's OWN effective tile (`_estimate_streaming_bytes(n, patch_hw,
  gpu_tile_size)`), not at the CPU tile. A GPU whose effective-tile host bound
  exceeds `per_cell_budget` is refused BEFORE execution (zero GPU attempt).
- GPU success: `_run_cell_batch(science_config, gpu_tile_size, gpu_attempt=True)`;
  single-owner (`cells_in_flight == 1`).
- Pre-run VRAM-infeasible (`choose_gpu_tile_size() is None`): `gpu_used=False`
  BEFORE the plan, `pre_execution_fallback.reason="vram_tile_infeasible"`, then the
  CPU plan/batch runs with `cpu_streaming_tile`.
- GPU effective-tile host-RAM-infeasible: `gpu_used=False` BEFORE the plan,
  `pre_execution_fallback.reason="gpu_stream_host_ram_infeasible"` (with
  `gpu_streaming_tile`/`gpu_streaming_host_peak_bytes`/`ram_budget_bytes`), then a
  CPU re-plan (inmem/stream/`ModeInfeasible`).
- Classified runtime GPU error after a VALID GPU plan: one-shot whole-batch CPU
  rerun with `cpu_streaming_tile` (never a re-coded 128, never the GPU tile).
- Manifest/log: `gpu.cpu_streaming_tile`, `gpu.gpu_streaming_tile`,
  `gpu.gpu_streaming_host_peak_bytes`, `gpu.pre_execution_fallback`; when the GPU
  is not attempted for host RAM, `attempted=false`, `final_backend=cpu`, explicit
  reason. `per_cell_diagnostics.gpu_streaming_host_peak_bytes` reflects the
  EXECUTED GPU bound (not the CPU bound).

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
  maxN 571), CPU streaming tile **48**, `predicted_bound_bytes` **2351.1 MiB**
  (<= budget; this is the accepted bound, F3), in-memory worst `13383.1 MiB`,
  legacy preferred stream@128 `4266.6 MiB` (kept as
  `cpu_streaming.preferred_stream_bound_bytes`, NOT the predicted bound).
  Tiles `128/112/96/80/64` exceed the budget; `48` and `32` fit.
- **Dynamic (informational, single capture `2026-10-13T19:09:42Z`):** the host's
  `avail*0.8` was `987.4 MiB` → **NO FIT** (the finest floor-feasible layout
  `26x16` needs `1191.8 MiB` even at the minimum tile 32; in-memory worst
  `4597.0 MiB`). This block is INFORMATIONAL only — the host's free memory varies
  between captures, which is exactly why only the FIXED 2514.4 MiB witness is
  authoritative. Both blocks come from the SAME capture
  (`.a2a-reports/ZM-ZEGRID-R36-scratch/witness_planning_result.json`,
  `_captured_at`).

Observation to be explicit about: with the R36 admissibility rule, the coarsest
admissible layout can be *coarser* than pre-R36 (Caldwell: `15x9` vs the pre-R36
finest `24x14`), because the auto path keeps its first/coarsest rule while the
admissibility predicate widened. This is the frozen policy as specified; it is
flagged under OPEN ISSUES because it changes the production `(nx, ny)`.
The accepted layout's `predicted_bound_bytes` is now the SELECTED bound
(`2351.1 MiB <= 2514.4 MiB`), no longer the legacy `4266.6` cheaper@128 bound
(F3).

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

- `tests/test_zegrid_layout_performance_r28.py tests/test_zegrid_layout_observability_r27.py tests/test_zegrid_r20_percell_parallel.py tests/test_zegrid_r22_gpu_contract.py` → **49 passed** (r0) / **44 passed** for R27+R28+R22 after the rework (R20 re-run green in the rework set).
- **`tests/test_zegrid_r36_adaptive_cpu_tile.py`** → **28 passed** (r0: 20; rework-1 added F1/F2/F3 direct + GPU-RAM-infeasibility integration tests; rework-2 added the two F-M1 GPU-effective-tile tests).
- Targeted regression set `test_zegrid_r20_percell_parallel.py test_zegrid_r22_runtime_fallback.py test_zegrid_r22_gpu_contract.py test_zegrid_r6_file_provider.py test_zegrid_r4_auto_layout.py test_zegrid_layout_performance_r28.py test_zegrid_layout_observability_r27.py test_sci05_canonical_streaming.py -m "not slow"` → **117 passed, 1 skipped** (rework-2).
- Pre-existing environmental failure (CuPy+GPU present):
  `test_sci05_canonical_streaming::TestValidation::test_gpu_backend_raises` —
  unrelated to R36, not forced green.

Rework-1 direct tests: F1 inmem-only plan (no TypeError, stream candidate
infeasible); F2 GPU over-budget raises (candidates attached) and GPU feasible
candidates are coherent; F3 predicted bound == selected bound (<= budget) with the
legacy bound kept separately; F2 product-path GPU-RAM-infeasible → CPU inmem with
NO GPU executor attempt and an honest manifest; both-infeasible fails loud with no
executor attempt.

Rework-2 (F-M1) tests: CPU tile 64 fits + GPU chooser 128 with `estimate@128` >
budget → zero GPU executor call, WARN + `pre_execution_fallback`
(`gpu_stream_host_ram_infeasible`, `gpu_streaming_tile=128`,
`gpu_streaming_host_peak_bytes`), CPU re-plan + EXECUTION at tile 64; GPU effective
tile 48 (host bound fits) → GPU IS attempted at 48 (single-owner) with the 48
bound gating, then a runtime OOM reruns CPU at tile 64; GPU VRAM tile `None` →
CPU re-plan before execution; runtime GPU failure after a valid plan → CPU rerun at
the selected CPU tile.

## OPEN ISSUES

1. **Layout selection can coarsen** (see WITNESS). A layout whose pre-R36
   cheaper bound exceeded the budget can now be admissible via a smaller tile,
   so the auto path can choose a *coarser* `(nx, ny)`. This is per the frozen
   policy but is an observable production-geometry change. If the intent was to
   keep the pre-R36 layout and only shrink the tile, the admissibility widening
   must be scoped so the coarse→fine scan keeps the pre-R36 predicate.
2. **Estimator extrapolation.** The ladder reuses the frozen estimator calibrated
   at 128/256; tiles below 128 are extrapolated (its per-tile fixed overhead
   dominates small patches). No new calibration was added (out of scope).
3. **`32` is a performance floor**, not a science floor; a machine needing a
   smaller tile is not served by R36. Deeper disk-backed streaming = R37.

## NEXT STEP

If tile `32` still cannot fit the finest floor-feasible layout, the next step is
a deeper **disk-backed streaming** executor (R37) — not implemented, not
promised here. R36 stops at the documented tile-32 performance floor with a
loud `LayoutInfeasible`.
