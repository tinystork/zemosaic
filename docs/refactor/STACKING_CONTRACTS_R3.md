# ZeMosaic — Stacking Contracts (R3) — PRE-R2 BASELINE FREEZE (post-R2 audited/accepted)

Mission: `ZM-ARCH-R3-BASELINE-FREEZE-20261004`
Phase: pre-R2 baseline freeze (accepted) → post-R2 audited/accepted
Date: 2026-10-04
Base SHA: `c03d0bb965d073b12ad9978094327829f0d0c366`
Branch: `refactor/zm-architecture-cleanup-r0-r3`

Chronology (explicit):
- **Pre-R2 source/audit HEAD** `e975c936c84b8c6622664b1f273b71c5b8c311d3` — the HEAD at which
  this map's rows were authored and independently re-anchored (Nono review-1).
- **Accepted pre-R2 freeze commit** `7c469562cb40521bb4f4d8a4ab511a5099ce17a4`
  (`docs: freeze pre-R2 stacking contracts`).
- **Post-R2 audited implementation HEAD** `1282cfe8901fac86ef21751991dcd98ae63679d2`
  (`refactor: extract crash breadcrumb engine`).

The map rows below were authored from the pre-R2 source at `e975c93` (== canonical base science;
no stacking code changed after it). They were **not** regenerated from a different science base;
the post-R2 audit only re-verified that the two bounded extractions do not touch stacking.

> **Status: POST-R2 AUDITED — ACCEPTED** after Nono review-0 ACCEPT + Junior acceptance.
> The pre-R2 freeze was accepted by Junior after Nono review-1 ACCEPT. Post-R2, the two
> bounded extractions (R2 lot 1 `core/grouping_helpers.py`, R2 lot 2B `core/crash_breadcrumbs.py`)
> were re-audited against this map: **neither extraction touches stacking math/order/weights/
> rejection/WCS/FITS/science**, so every row below remains valid unchanged. This is a technical
> acceptance of the map, **not** M106 scientific acceptance.
> This document consolidates the actual reachable stacking paths
> from source at the canonical base, plus the accepted runtime witnesses. It replaces the
> earlier `PRELIMINARY R0 map (rework-1)` / `corrected rework-2` labels. It is **not** a
> numerical-parity validation. Cells marked UNKNOWN / NOT_RUN are open; they are not invented,
> and no CPU↔GPU parity is claimed. The post-R2 final audit is recorded in `FINAL_REPORT.md`
> and tracked in `todo.md`.

Legend:
- **requested** = config/flag asked for; **effective** = post-guard/fallback decision;
  **executed** = what actually ran. Static evidence only unless a runtime witness is cited.
- **Evidence status** per row: `WITNESSED` (runtime test, path + result cited) |
  `STATIC ACTIVE` (source-reachable, inspected, no runtime witness) | `NOT_RUN`
  (reachable but never executed in any accepted witness) | `DORMANT` (gated off / only
  reachable on a missing-import path at base).

## GPU / JIT qualification fact (unchanged, kept from R0)

CuPy 14.1.1 ships its own NVRTC (`cupy_backends.cuda.libs.nvrtc`, version 12.9).
`RawKernel`/`RawModule`/JIT work **without `nvcc`** — a `RawKernel` witness succeeded on the
MX150 (`[1,2,3,4]` → `[3,4,5,6]`). Qualification limits are ~2 GB VRAM and that GPU stacking
and CPU↔GPU parity were **NOT_RUN** in R0, **not** JIT availability. `nvcc` absence is
irrelevant to CuPy JIT.

## Mode/backend matrix (actually reachable)

| Row | Entry / caller | Backend (requested → effective → executed) | Evidence |
| --- | --- | --- | --- |
| Classic CPU | `run_hierarchical_mosaic_classic_legacy` (`worker:23352`) → `_stack_master_tile_auto` (`:15763`, call `:17475`) → `_stack_master_tile_cpu` (`:15559`) → `stack_winsorized_sigma_clip` / `stack_kappa_sigma_clip` / `stack_linear_fit_clip` / `stack_aligned_images` (`:15645/15658/15668/15677`) | CPU (numpy); GPU flags zeroed by `_stack_master_tile_cpu` (`:15590-15604`) | STATIC ACTIVE; low-N/all-invalid WITNESSED (`tests/test_stacking_low_n_all_invalid_witness.py`) |
| Classic GPU | `_stack_master_tile_auto` → Phase-3 `_p3_gpu_stack_from_paths` (`:13389`, `zemosaic_align_stack_gpu.gpu_stack_from_paths`); plus wrapper-internal `gpu_stack_winsorized`/`gpu_stack_kappa`/`gpu_stack_linear` (`align_stack:1539/1628/1695`) gated by `_plan_gpu_stack_execution` (`:1139`) | GPU (cupy) | STATIC ACTIVE; NOT_RUN (no GPU stacking executed in R0) |
| SDS | `assemble_global_mosaic_sds` (`worker:37965`) → `_stack_mosaics` (`:38683`) → `stack_winsorized_sigma_clip` / `stack_kappa_sigma_clip` with `zconfig=None` (`:38704-38708`, `:38719-38723`) | **CPU (numpy)** — wrappers called with `zconfig=None` → `use_gpu=False` | STATIC ACTIVE; NOT_RUN (no SDS runtime witness) |
| Grid CPU | `_stack_weighted_patches` (`grid_mode:1913`) | CPU (numpy) | STATIC ACTIVE; low-N/all-invalid WITNESSED (`test_stacking_low_n_all_invalid_witness.py`) |
| Grid GPU (core) | `_stack_weighted_patches_gpu` (`grid_mode:1984`) → `stack_core(backend='gpu')` (`:2010-2028`) | GPU if `_CUPY_AVAILABLE` + `config.use_gpu` | STATIC ACTIVE; NOT_RUN (no GPU `stack_core` execution) |
| Grid GPU (legacy fallback) | same fn, `stack_core` is None → legacy cp rejection (`grid_mode:2034+`) | GPU, numpy rejection via `cp.asnumpy` | DORMANT (reachable only if `stack_core` import fails); NOT_RUN |
| Phase 4.5 (alpha-weighted direct) | `_run_phase4_5_inter_master_merge` `weights_ready` branch (`worker:8315-8337`) | CPU (numpy direct weighted mean) | STATIC ACTIVE; NOT_RUN (no Phase 4.5 execution) |
| Phase 4.5 (configured rejection) | same fn rejection branch (`worker:8342-8372`) | **CPU (numpy)** — wrappers called with `zconfig=None` | STATIC ACTIVE; NOT_RUN |

## Row detail — Classic CPU

1. **Entry/caller** — `run_hierarchical_mosaic` (`worker:30675`) detects non-Grid, non-SDS
   and returns `run_hierarchical_mosaic_classic_legacy(...)` (`:30983`). Per master tile,
   `_stack_master_tile_auto` (`:15763`) is invoked (`:17475`); its CPU fallback is
   `_stack_master_tile_cpu` (`:15559`).
2. **Config names / aliases / precedence** — canonical GUI keys `stacking_normalize_method`,
   `stacking_weighting_method`, `stacking_rejection_algorithm`, `stacking_kappa_low/high`,
   `stacking_winsor_limits`, `stacking_final_combine_method` (`zemosaic_config.py:114-121`).
   Worker rename map (`worker:36263+`) → `stack_norm_method`/`stack_weight_method`/
   `stack_reject_algo`/`stack_kappa_low/high`/`stack_final_combine`; `stacking_winsor_limits`
   parsed to `parsed_winsor_limits` tuple (`:36285-36289`, fallback `(0.05,0.05)`). The
   wrapper resolves `use_gpu` via `stack_use_gpu` → `use_gpu_stack` → `use_gpu`
   (`align_stack:1792-1795` / `:2413-2416` / `:2540-2543`).
3. **requested → effective → executed** — `stack_reject_algo` selects the wrapper
   (`:15645/15658/15668`); any other value → `stack_aligned_images` (`:15677`).
   `_stack_master_tile_cpu` saves and force-zeroes `stack_use_gpu`/`use_gpu_stack`/`use_gpu`
   and `parallel_plan.use_gpu` before the call and restores them in `finally`
   (`:15590-15604`, `:15697-15703`), so the executed backend is **CPU (numpy)**.
4. **Normalization** — `normalize_method` ∈ `linear_fit` (`_normalize_images_linear_fit`,
   `:3296`) | `sky_mean` (`_normalize_images_sky_mean`, `:3389`) | else none (`:4991-4996`).
   This is **normalization**, distinct from linear-fit rejection (below).
5. **Weighting** — `_compute_quality_weights` (`:4241`): `noise_variance`
   (`:3519`) | `noise_fwhm` (`:3772`, falls back to `noise_variance` if photutils absent or
   no usable weights) | `none`/unknown → no weights. Plus optional radial weight map.
6. **Rejection algorithm / identity** — `stack_winsorized_sigma_clip` (`:1756`),
   `stack_kappa_sigma_clip` (`:2399`), `stack_linear_fit_clip` (`:2526`). See "Rejection
   algorithms" section for the exact implementation identity of each.
7. **Final combine** — mean (weighted) or median; median ignores weight magnitude
   (`stack_aligned_images` mean `:5170` / median `:5210`; `stack_core` mean `:345` / median
   `:350`). See "Final combine".
8. **dtype / axes / NaN-Inf / masks / low-N** — float32 HWC; `_filter_statistically_dead_frames`
   drops empty/degenerate frames; NaN→0 before combine; `where(sum_weights>1e-9)` division.
   Low-N witnessed: kappa N=1 returns input, N=2 returns per-pixel mean (`rejected=0.0`);
   winsorized N=1 logs `"Winsorized clip needs >=3 images; forcing CPU."` and returns input
   with `rejected=0.0` (`test_stacking_low_n_all_invalid_witness.py`, CPU).
9. **Error/import/OOM fallback** — wrapper GPU attempt falls back to CPU on any GPU error;
   winsorized CPU uses external Seestar `cpu_stack_winsorized` else internal
   `_cpu_stack_winsorized_fallback` (`:5563`); kappa uses `cpu_stack_kappa` else
   `_cpu_stack_kappa_fallback` (`:5396`); linear uses `cpu_stack_linear` else
   `_cpu_stack_linear_fallback` (`:5511`).
10. **Evidence** — STATIC ACTIVE; low-N/all-invalid WITNESSED (CPU). Full N≥3 NOT_RUN.

## Row detail — Classic GPU

1. **Entry/caller** — `_stack_master_tile_auto` (`worker:15763`) → Phase-3 GPU auto via
   `_p3_gpu_stack_from_paths` (imported `:13389` from `zemosaic_align_stack_gpu`); the same
   wrappers as Classic CPU can also dispatch GPU internally when `use_gpu` is honored.
2. **Config / precedence** — `_phase3_gpu_candidate` (`:15712`) gates on `_P3_GPU_STATE`
   (`allowed`/`hard_disabled`/`healthy`), `_P3_GPU_HELPERS_AVAILABLE`, and
   `parallel_plan.use_gpu`. Wrapper GPU path honors `stack_use_gpu` → `use_gpu_stack` →
   `use_gpu` (see Classic CPU).
3. **requested → effective → executed** — two distinct sub-paths:
   - **Phase-3 auto** `_p3_gpu_stack_from_paths` = `gpu_stack_from_paths`
     (`zemosaic_align_stack_gpu:1908`) → `gpu_stack_from_arrays` (`:1386`), gated by
     `_phase3_gpu_candidate` (`worker:15712`) on `_P3_GPU_STATE` (`allowed`/`hard_disabled`/
     `healthy`) + `_P3_GPU_HELPERS_AVAILABLE` + `parallel_plan.use_gpu`, and by
     `_gpu_is_usable()` inside.
   - **Wrapper-internal** GPU `gpu_stack_winsorized`/`gpu_stack_kappa`/`gpu_stack_linear`
     (`align_stack:1539/1628/1695`) gated by `_plan_gpu_stack_execution` (`align_stack:1139`)
     on `GPU_AVAILABLE` + `gpu_is_available()` + VRAM budget. Effective backend is GPU only
     when all guards pass.
4. **Normalization** — **Phase-3 auto**: `gpu_stack_from_arrays` normalizes via
   `_normalize_frame` (`zemosaic_align_stack_gpu:329`) + `_compute_sky_mean_offsets`
   (`:154`, sky offsets passed to the core); `stack_norm_method` read at `:1464`.
   **Wrapper-internal**: same `linear_fit`/`sky_mean`/none as Classic CPU (GPU percentiles via
   `use_gpu_norm` when enabled, `:4991-4996`).
5. **Weighting** — **Phase-3 auto**: `_prepare_frames_and_weights` builds quality weights
   (`_compute_quality_weights`, `:1141`) + optional radial map (`_compute_radial_weight_map`,
   `:917`) + WSC `weights_block` (`_build_wsc_weights_block`, `:957`); weights only apply when
   `combine_method == 'mean'` (logged warning otherwise, `:1473`). **Wrapper-internal**:
   `gpu_stack_*` have **no `weights` parameter** → weights ignored (logged); WSC PixInsight
   GPU path (`_wsc_pixinsight_stack_gpu`, `:1474`) accepts a `weights_block`.
6. **Rejection / identity** — **Phase-3 auto**: rejection selected by `stack_reject_algo`
   (`:1456`); `winsorized_sigma_clip` + `wsc_impl=pixinsight` → PixInsight WSC core (with
   `_wsc_gpu_parity_check`, `:1525`); else kappa/linear/winsorized via core helpers.
   **Wrapper-internal**: `gpu_stack_winsorized` (`:1539`, winsorize→sigma→mean),
   `gpu_stack_kappa` (`:1628`, median/σ clip), `gpu_stack_linear` (`:1695`, median-residual
   clip). See "Rejection algorithms".
7. **Final combine** — **Phase-3 auto**: `cp.nanmean` / `cp.nanmedian` (`zemosaic_align_stack_gpu:826-831`)
   selected by `stack_final_combine` (mean/median). **Wrapper-internal**: `cp.nanmean` only
   in each `gpu_stack_*` helper (mean-only on this sub-path).
8. **dtype / axes / NaN-Inf / masks / low-N** — float32; row-chunked when VRAM budget exceeded
   (`_resolve_chunk_rows_for_gpu_helper` / `_resolve_rows_per_chunk`).
9. **Error/OOM fallback** — `_stack_master_tile_auto` retries once on GPU error, shrinks plan
   on OOM (`_shrink_parallel_plan_for_gpu`), then hard-disables Phase-3 GPU for the run and
   falls back to `_stack_master_tile_cpu` (`:15846-15872`). Wrapper GPU errors → CPU fallback.
10. **Evidence** — STATIC ACTIVE; NOT_RUN (no GPU stacking executed; CuPy JIT only).

## Row detail — SDS

1. **Entry/caller** — `assemble_global_mosaic_sds` (`worker:37965`), called from
   `run_hierarchical_mosaic` (`:27168`, `:33344`). Final per-batch mosaic stack is
   `_stack_mosaics` (`:38683`).
2. **Config / precedence** — `stack_params["stack_reject_algo"]` (default
   `winsorized_sigma_clip`, `:38692`), `stack_weight_method`, `stack_kappa_low/high`,
   `parsed_winsor_limits`, `winsor_worker_limit`, `winsor_max_frames_per_pass`.
3. **requested → effective → executed** — `_stack_mosaics` calls `stack_winsorized_sigma_clip`
   / `stack_kappa_sigma_clip` with **`zconfig=None`** (`:38704-38708`, `:38719-38723`).
   Each wrapper resolves `use_gpu` **from `zconfig` only** → `zconfig is None` →
   `use_gpu=False`. **Executed backend is CPU (numpy).** There is **no SDS GPU stacking** —
   no SDS wrapper call passes a `zconfig` that could enable GPU, and `parallel_plan` does not
   toggle GPU on these wrappers.
4. **Normalization** — none at the final SDS stack (coverage-weighted manual weights only).
   SDS-specific normalization lives in `_normalize_sds_megatiles_photometry` (`:5178`) and
   `_finalize_sds_global_mosaic` (`:5251`), separate from the stacking combine.
5. **Weighting** — `manual_weights` from per-batch coverage (`_coverage_weight`, `:38653-38678`),
   a scalar `(N,)` float32 array passed as `weights=` to the wrappers.
6. **Rejection / identity** — `winsorized_sigma_clip` → `stack_winsorized_sigma_clip`;
   `kappa_sigma` → `stack_kappa_sigma_clip`; else plain `nanmedian`/`nanmean` (`:38688-38691`,
   `:38733-38736`). See "Rejection algorithms" for implementation identity.
7. **Final combine** — wrappers (mean weighted / median); plain `nanmedian`/`nanmean` fallback.
8. **dtype / axes / NaN-Inf / masks / low-N** — float32 mosaics; `nanmedian`/`nanmean` on the
   stack; N=1 short-circuits to `stack_cube[0]` (`:38688`).
9. **Error/OOM fallback** — `_stack_mosaics` wrapped in try/except (`:38737+`); if
   `zemosaic_align_stack` unavailable, plain `nanmedian`/`nanmean` fallback (`:38685-38691`).
10. **Evidence** — STATIC ACTIVE; NOT_RUN (no SDS runtime witness). Backend independently
    confirmed CPU from `zconfig=None` call kwargs, not assumed.

## Row detail — Grid CPU

1. **Entry/caller** — `run_grid_mode` (`grid_mode:4152`) → `process_tile` (`:2115`) →
   `_stack_weighted_patches` (`:1913`). Dispatched from `run_hierarchical_mosaic` when
   `detect_grid_mode(input_folder)` (`worker:30898`).
2. **Config / precedence** — `GridModeConfig` (`:481`): `stack_norm_method` (default
   `linear_fit`), `stack_weight_method`, `stack_reject_algo` (default `kappa_sigma`),
   `stack_kappa_low/high=3.0`, `winsor_limits=(0.05,0.05)`, `stack_final_combine="mean"`,
   `use_gpu` (`:498`). Grid `use_gpu` precedence: `run_grid_mode` `use_gpu` param else config
   `use_gpu_grid` (`:4213-4217`).
3. **requested → effective → executed** — CPU branch executes `_stack_weighted_patches`
   (numpy) directly when `config.use_gpu` is false (or `_CUPY_AVAILABLE` false).
4. **Normalization** — `_normalize_patches` (`:1798`) with `method=config.stack_norm_method`:
   `none`/`unit` → passthrough; `linear_fit` → `_fit_linear_scale` per-channel gain/offset
   (`:1718`); default median scaling.
5. **Weighting** — per-patch weight maps (coverage/alpha); `data_stack = where(weight>0, data,
   nan)` (`:1938`).
6. **Rejection / identity** — kappa → established `_reject_outliers_kappa_sigma` (`:1943`);
   winsorized → established `_reject_outliers_winsorized_sigma_clip` (`:1950`), imported from
   `zemosaic_align_stack` (`grid_mode:133-139`). See "Rejection algorithms" for the dispatcher
   semantics of the winsorized helper.
7. **Final combine** — median (`:1969`) or weighted mean with `weight_sum` clipped at `1e-6`
   (`:1960`).
8. **dtype / axes / NaN-Inf / masks / low-N** — float32 HWC; empty patches → `None`; all-zero
   weights / all-NaN → zero tiles (division clipped at `1e-6`, not NaN); median ignores weight
   **magnitude** but treats `weight<=0` as invalid. WITNESSED
   (`test_stacking_low_n_all_invalid_witness.py`, 18 pass, CPU).
9. **Error/OOM fallback** — `_stack_weighted_patches_gpu` falls back to this CPU path on any
   GPU error/OOM (`grid_mode:2098+`); if `_CUPY_AVAILABLE` false → CPU directly.
10. **Evidence** — STATIC ACTIVE; low-N/all-invalid WITNESSED (CPU).

## Row detail — Grid GPU (via `stack_core`)

1. **Entry/caller** — `_stack_weighted_patches_gpu` (`grid_mode:1984`) when `_CUPY_AVAILABLE`
   and `config.use_gpu`.
2. **Config / precedence** — same `GridModeConfig`; `stack_core` receives
   `normalize_method='none'`, `rejection_algorithm=config.stack_reject_algo`,
   `final_combine_method=config.stack_final_combine`, `sigma_clip_low/high=config.stack_kappa_low/high`
   (`:2012-2017`).
3. **requested → effective → executed** — GPU normalization upstream (`_normalize_patches_gpu`,
   `:1860`), then `stack_core(..., backend='gpu')` (`:2018-2028`) when `stack_core` is not
   None. `stack_core` returns numpy, re-wrapped as `cp.asarray`.
4. **Normalization** — upstream GPU `linear_fit`/`none`/median (`_normalize_patches_gpu`,
   `:1860`); `stack_core` itself gets `normalize_method='none'`. Do not attribute the
   `stack_core` `linear_fit` placeholder (median subtraction, `stack_core:294-296`) to Grid —
   Grid normalizes upstream and passes `none`.
5. **Weighting** — `cp_weights` passed as `weights=` to `stack_core`.
6. **Rejection / identity** — `stack_core` kappa: when the imported
   `_reject_outliers_kappa_sigma` is available, GPU does `cp.asnumpy(stacked)` then calls the
   **same established helper** (`stack_core:302-317`); fallback median/σ clip only if import
   failed. `stack_core` winsorized_sigma_clip = **simplified median/σ clip, NOT winsorization
   and NOT PixInsight WSC** (`stack_core:325-333`) — structural divergence from Grid CPU
   (SCI-01). See "Rejection algorithms".
7. **Final combine** — `stack_core` mean (weighted, `where(weight_sum>0, ..., nan)`) or
   median (ignores weights).
8. **dtype / axes / NaN-Inf / masks / low-N** — `_ensure_hwc_tile` (`stack_core:71`); all-zero
   weights / all-NaN → NaN (not 0.0) at zero `weight_sum` (contrast Grid CPU zeros); median
   ignores weights entirely. WITNESSED on CPU backend only
   (`test_stacking_low_n_all_invalid_witness.py`); GPU backend NOT_RUN.
9. **Error/OOM fallback** — `_stack_weighted_patches_gpu` catches GPU OOM/error →
   `_stack_weighted_patches` (CPU) (`grid_mode:2098+`); if `stack_core` is None → legacy GPU
   branch (next row).
10. **Evidence** — STATIC ACTIVE; NOT_RUN (no GPU `stack_core` execution).

## Row detail — Grid GPU (legacy fallback)

1. **Entry/caller** — `_stack_weighted_patches_gpu` (`grid_mode:1984`) with `stack_core is
   None` (import failed) → legacy GPU rejection block (`:2034+`).
2. **Config / precedence** — same `GridModeConfig` keys as Grid CPU/GPU.
3. **requested → effective → executed** — GPU (cupy) stack; rejection via numpy
   `cp.asnumpy` round-trips.
4. **Normalization** — `_normalize_patches_gpu` (`:1860`).
5. **Weighting** — `cp_weights`; `data_stack = where(weight>0, data, nan)`.
6. **Rejection / identity** — kappa/winsorized via `cp.asnumpy` → established
   `_reject_outliers_kappa_sigma` / `_reject_outliers_winsorized_sigma_clip`
   (kappa `:2042` / winsorized `:2051`), then back to `cp.asarray`. See "Rejection algorithms"
   for the winsorized dispatcher semantics.
7. **Final combine** — median (numpy round-trip, `:2072`) or weighted mean (cupy, `weight_sum`
   clipped `1e-6`).
8. **dtype / axes / NaN-Inf / masks / low-N** — float32; empty valid positions → zero tile.
9. **Error/OOM fallback** — same outer try/except → CPU `_stack_weighted_patches` (`:2098+`).
10. **Evidence** — DORMANT (reachable only if `stack_core` import fails); NOT_RUN.

## Row detail — Phase 4.5 alpha-weighted direct branch

1. **Entry/caller** — `_run_phase4_5_inter_master_merge` (`worker:7335`), invoked from
   `_run_shared_phase45_phase5_pipeline` (`:11009`) only when `phase45_options["enable"]`.
2. **Config / precedence** — `stack_cfg["stacking_normalize_method"]`/`normalize_method`/
   `stack_norm_method` (`:7676-7700`); `stack_cfg["reject_algo"]`/`stack_reject_algo`
   (`:7682`); `inter_cfg["overlap_threshold"/"min_group_size"/"max_group"/"memmap_policy"/
   "local_scale"]`; `photometry_intragroup`/`photometry_intersuper`/`photometry_clip_sigma`
   (`:7364-7366`).
3. **requested → effective → executed** — `weights_ready = any(isinstance(w, np.ndarray) for w
   in frame_weights)` (`:8315`). When true, direct numpy weighted mean is executed; it
   **bypasses configured reject/combine**. On exception → falls through to configured
   rejection branch. Backend is **CPU (numpy)** throughout Phase 4.5.
4. **Normalization** — pre-stack normalization (B step below) may already have mutated
   `frames` in place; this branch does no further normalization.
5. **Weighting** — weight maps `np.clip(np.nan_to_num(w, nan=0.0), 0.0, 1.0)`; non-array/
   mismatched entries → `ones` (`:8321-8327`).
6. **Rejection / identity** — **none**; direct weighted mean, no rejection.
7. **Final combine** — `super_arr = np.where(den>0, nansum(frames*w)/nansum(w), np.nan)`
   (`:8334`); `alpha_out = (nanmax(weight_stack, axis=0)*255).astype(uint8)` (`:8336`).
8. **dtype / axes / NaN-Inf / masks / low-N** — float32 HWC; NaN weights → 0; NaN output where
   `den<=0`; alpha = max weight ×255 (uint8).
9. **Error fallback** — exception → `super_arr=None` → configured rejection branch.
10. **Evidence** — STATIC ACTIVE; NOT_RUN (no Phase 4.5 execution).

## Row detail — Phase 4.5 configured-rejection branch

1. **Entry/caller** — same `_run_phase4_5_inter_master_merge`; reached only when
   `super_arr is None` (`worker:8342`).
2. **Config / precedence** — `reject_algo = stack_cfg.get("reject_algo",
   stack_cfg.get("stack_reject_algo", "winsorized_sigma_clip"))` (`:7682`).
   **`inter_cfg["stack_method"]` / `inter_master_stack_method` is passed but NOT consumed**
   (no read inside the merge); real selection is `stack_cfg["reject_algo"]`.
3. **requested → effective → executed** — `winsor`/`winsorized_sigma_clip` →
   `stack_winsorized_sigma_clip`; `kappa_sigma` → `stack_kappa_sigma` (if present) else
   `stack_kappa_sigma_clip`; `linear_fit_clip` → `stack_linear_fit_clip` (`:8343-8372`). All
   with **`zconfig=None`** → `use_gpu=False` → **CPU (numpy)**. `parallel_plan` does not
   enable GPU; `stack_cfg_phase45` has no GPU key.
4. **Normalization** — pre-stack B-step may have mutated `frames`; no further normalization
   here.
5. **Weighting** — `weight_method` passed through to wrappers; no alpha maps in this branch.
6. **Rejection / identity** — the three wrappers (see "Rejection algorithms").
7. **Final combine** — wrapper combine; last resort `nanmedian`/`nanmean` (`final_combine`,
   `:8385-8388`).
8. **dtype / axes / NaN-Inf / masks / low-N** — float32 HWC; `_ensure_hwc_master_tile` applied
   to `super_arr` after stacking (`:8398`).
9. **Error fallback** — stack exception → cleanup + skip chunk (`:8389-8393`).
10. **Evidence** — STATIC ACTIVE; NOT_RUN.

## Phase 4.5 A…E temporal map (exact, unchanged from accepted R0)

Inside `_run_phase4_5_inter_master_merge` (`worker:7335`), backend CPU/numpy throughout.
The normalization/photometry data-flow is five labeled steps in exact order:

- **A. Helper-gated pre-stack affine / micro-align branches — DORMANT (missing helpers).**
  Gates `micro_align_available` / `photometry_estimator_available` /
  `photometry_apply_available` = `hasattr(zemosaic_align_stack, ...)` (`:7412-7414`). At BASE
  these are `False False False` (no `micro_align_stack` / `estimate_affine_photometry` /
  `apply_affine_photometry` defined in src). Micro-align (`:8034`), intra-group affine
  (`do_chunk_photometry` `:7701-7705`, calls `:7766/:8084/:8095`), legacy affine
  (`:8151-8154`) are INERT. Config flags `photometry_intragroup`/`photometry_intersuper`/
  `photometry_clip_sigma` read at `:7364-7366`. NOT PROVEN DEAD.
- **B. ACTIVE pre-stack explicit linear_fit / sky_mean normalization** (`:8202-8290`), mutates
  `frames` **in place** before stacking. Gate `norm_method in ("linear_fit","sky_mean") and
  len(frames) >= 2`; `norm_method` from `stack_cfg["stacking_normalize_method"]` →
  `normalize_method` → `stack_norm_method` → `none` (`:7676-7700`).
  - `linear_fit`: per channel, common-finite mask, min overlap `max(5000, ceil(1%))`, sigma
    clip via `clip_sigma_norm` (=`photometry_clip_sigma`, clipped `[0.1,10.0]` `:8168-8174`),
    OLS `slope=dot(xv,yv)/dot(xv,xv)` **clipped `[0.25,4.0]`**, `intercept=ym-slope*xm`,
    applied in place to valid source pixels.
  - `sky_mean`: per channel, percentile band `[sky_low,sky_high]` (`intertile_sky_percentile`,
    default `30,70`, `:8175-8200`), per-channel medians, additive `delta=bg_ref-bg_src`.
  - Exceptions caught/logged; `frames`/channels mutated **in place** with no copy/rollback →
    possible **partial in-place normalization** after an exception.
- **C. Stacking — alpha-weighted early branch OR configured rejection branch** (two rows above).
- **D. ACTIVE post-stack inter-super gain-only normalization** (`:8622-8832`), independent of
  absent helpers. Gate `photometry_intersuper and len(candidate_super_tiles) >= 2`
  (`:8624-8629`). Reads saved super-tile FITS (`fits.open(memmap=True,
  do_not_scale_image_data=True)`), per-channel medians weighted by valid-pixel counts (IQR
  clip when `photometry_clip_sigma>0`), weighted-median reference (fallback dominant tile),
  gains `ref/med` **clipped `two_pass_cov_gain_clip` default `[0.85,1.18]`** (`:8700-8712`),
  reopens each FITS `mode="update"`, multiplies channels in place, writes `ZM45NORM=True` +
  HISTORY (`:8758-8795`). **Reachable at BASE.**
- **E. Global-affine inter-super branch** (`:8837+`, `estimate_affine_photometry` at `:8865`)
  — helper-gated (`photometry_estimator_available and photometry_apply_available`), **INERT
  at BASE**.

- **Backend: CPU (numpy).** Wrappers resolve `use_gpu` from `zconfig` only; `zconfig=None` →
  `use_gpu=False`. `parallel_plan` does not enable GPU; `stack_cfg_phase45` has no GPU key.
- **`inter_master_stack_method` / `inter_cfg["stack_method"]` is passed but not consumed**;
  actual rejection selection is `stack_cfg["reject_algo"]`.
- **Nothing applies affine photometry to `super_arr` in memory.** B acts on `frames`
  pre-stack; D rewrites saved super-tile FITS post-stack.
- Evidence: A/E DORMANT (missing helper); B/D ACTIVE STATIC; C ACTIVE STATIC; no Phase 4.5
  path executed at runtime (NOT_RUN).

## Config aliases / precedence (stacking keys)

- GUI/config canonical names: `stacking_normalize_method`, `stacking_weighting_method`,
  `stacking_rejection_algorithm`, `stacking_kappa_low/high`, `stacking_winsor_limits`,
  `stacking_final_combine_method` (`zemosaic_config.py:114-121`). `wsc_impl` default
  `"pixinsight"` (`:120`).
- Worker process wrapper rename map (`worker:36263+`): `stacking_normalize_method →
  stack_norm_method`, `stacking_weighting_method → stack_weight_method`,
  `stacking_rejection_algorithm → stack_reject_algo`, `stacking_final_combine_method →
  stack_final_combine`, `stacking_kappa_low/high → stack_kappa_low/high`; `stacking_winsor_limits`
  parsed to `parsed_winsor_limits` tuple (`:36285-36289`). `_config` suffix promotion
  (`:36295+`) and silent drop of unknown kwargs (`:36302-36303`) are characterized by TEST-03
  (`tests/test_dispatch_propagation_witness.py`, 14 pass).
- Internal aliases consumed by `_compute_quality_weights`: `noise_variance` | `noise_fwhm`
  (fallback→`noise_variance`) | `none`/unknown → no weights.
- `use_gpu` precedence: `stack_use_gpu` → `use_gpu_stack` → `use_gpu` (align_stack wrappers);
  Grid: `use_gpu` param → `use_gpu_grid` config (`grid_mode:4213-4217`).

## Normalization (linear_fit vs sky_mean vs none; distinct from rejection)

- Classic: `_normalize_images_linear_fit` (`:3296`) / `_normalize_images_sky_mean` (`:3389`),
  invoked when `normalize_method='linear_fit'`/`'sky_mean'` (`:4991-4996`).
- Grid CPU/GPU: `_normalize_patches` / `_normalize_patches_gpu` (`grid_mode:1798/1860`)
  `method=config.stack_norm_method`; `linear_fit` = `_fit_linear_scale(_gpu)` per-channel
  gain/offset; `none` = passthrough; default median scaling.
- `stack_core`: `linear_fit` is a **placeholder** (median subtraction, `stack_core:294-296`);
  Grid GPU normalizes upstream and passes `'none'` to the core, so the placeholder is not
  reached on the Grid path. **SCI-02 CHARACTERIZED / CLOSED-NO-FIX**
  (`ZM-SCI-02-STACKCORE-LINEARFIT-CHAR-20261004`): placeholder proven bit-exact == `median`
  and non-affine (residual 240.33 vs 7.6e-06 for real Grid linear fit); AST inventory confirms
  a single production caller (`grid_mode._stack_weighted_patches_gpu`) that passes `none`;
  real linear-fit paths (Grid covariance/variance regression; classic percentile-based) are
  genuine affine mappings distinct from the placeholder; normalization `linear_fit` ≠
  rejection `linear_fit_clip`. Witness: `tests/test_stack_core_linear_fit_characterization.py`
  (16 pass) + `docs/refactor/SCI_02_STACK_CORE_LINEAR_FIT_CHARACTERIZATION.md`.

## Weighting

- Scalar per-frame `(N,)` broadcast, or per-pixel planes `(N,H,W[,C])` / alpha maps.
- Classic/SDS/Phase4.5 wrappers: `_compute_quality_weights` (noise_variance / noise_fwhm /
  none) + optional radial map + optional manual `weights=`.
- Grid: per-patch weight maps (coverage/alpha); `data = where(weight>0, data, nan)`.
- Phase 4.5 alpha branch: per-pixel alpha maps clipped `[0,1]`.

## Rejection algorithms and implementation identity

- **Kappa-Sigma** (`kappa_sigma`): two distinct implementations.
  - Established helper `_reject_outliers_kappa_sigma` (`align_stack:4371`, astropy
    `sigma_clipped_stats`, per channel) — used by `stack_aligned_images` (`:5132`), Grid CPU
    (`grid_mode:1943`), Grid GPU legacy (`grid_mode:2042`), and `stack_core` (CPU/GPU via
    `cp.asnumpy`, `stack_core:302-317`).
  - Wrapper `stack_kappa_sigma_clip` (`:2399`) → GPU `gpu_stack_kappa` (median/σ clip) or CPU
    `cpu_stack_kappa` (external Seestar) / `_cpu_stack_kappa_fallback` (`:5396`, median/σ clip).
- **Winsorized Sigma Clip** (`winsorized_sigma_clip`): the algorithm is selected by `wsc_impl`
  (`resolve_wsc_impl`, `robust_rejection:14`; precedence env `ZEMOSAIC_WSC_IMPL` → config
  `wsc_impl`/`winsor_impl`/`stack_winsor_impl` → default **`pixinsight`**).
  - `pixinsight` (DEFAULT) → `wsc_pixinsight_core` (`robust_rejection:88`) /
    `wsc_pixinsight_core_streaming_numpy` (`:328`) — PixInsight WSC. The wrapper
    `stack_winsorized_sigma_clip` (`:1756`) routes CPU→`_wsc_pixinsight_stack_numpy`
    (`:1337`) and GPU→`_wsc_pixinsight_stack_gpu` (`:1474`) under pixinsight.
  - `legacy_quantile` → winsorize-then-clip (scipy `winsorize` + astropy
    `sigma_clipped_stats`) inside `_reject_outliers_winsorized_sigma_clip` (`:4486`, legacy
    branch) and `_cpu_stack_winsorized_fallback` (`:5563`).
  - **`_reject_outliers_winsorized_sigma_clip` (`:4486`) is a dispatcher, not a single
    implementation**: `effective_impl = wsc_impl or resolve_wsc_impl()` and, when pixinsight,
    it calls `wsc_pixinsight_core` (`:4528-4530`) and returns its broadcast output. It is
    imported and called by Grid CPU (`grid_mode:1950`) and Grid GPU legacy
    (`grid_mode:2051`) **without** an explicit `wsc_impl`, so those paths execute PixInsight
    WSC by default (refining the coarse R0 note "Grid is not a WSC consumer": `grid_mode`
    does not import `core/robust_rejection` directly, but the established helper it calls
    delegates to WSC by default).
  - **`stack_core` winsorized_sigma_clip is NOT WSC and NOT winsorization** — it is a
    simplified median/σ clip (`stack_core:325-333`). Structural divergence from Grid CPU
    (SCI-01); numeric impact CHARACTERIZED/MEASURED (no fix) — see
    `SCI_01_GRID_WSC_CHARACTERIZATION.md` + `tests/test_grid_wsc_characterization.py`
    (mission ZM-SCI-01-GRID-WSC-CHAR-20261004). Grid is not a *stack_core*-level WSC consumer.
- **Linear-fit clip** (`linear_fit_clip`): wrapper `stack_linear_fit_clip` (`:2526`) →
  GPU `gpu_stack_linear` (median-residual clip, `:1695`) or CPU `cpu_stack_linear` (external
  Seestar) / `_cpu_stack_linear_fallback` (`:5511`, median-residual clip). **Note:**
  `_reject_outliers_linear_fit_clip` (`:4726`) is a **placeholder** that returns data
  unchanged with an all-True mask and has **no callers** — it is not the executed linear-fit
  clip path. Unsupported/`none` rejection = passthrough (no rejection).

## Final combine

- Mean: weighted `sum(data*w)/sum(w)`; `weight_sum` clipped at `1e-6` (Grid CPU) or
  `where(weight_sum>0, ..., nan)` (`stack_core`); Classic uses `where(sum_weights>1e-9)`.
- Median: `nanmedian` — weight **magnitude** ignored everywhere. Grid CPU additionally treats
  `weight<=0` as a validity mask (masks the frame to NaN before the median, `grid_mode:1938`);
  Classic (`stack_aligned_images`, median branch logs "median with weights not supported") and
  `stack_core` (`:350-352`) ignore weights entirely in the median.

## dtype / axes / NaN-Inf / masks / low-N

- HWC (channels-last) canonical; `_ensure_hwc_tile` (`stack_core:71`) and `_ensure_hwc_array`
  (`grid_mode:387`) normalize; CHW→HWC when `C` small. float32 output throughout.
- NaN masking: Classic/Grid mask non-positive weights / non-finite data before combine;
  `stack_core` uses `xp.isfinite` gating and `nan` for `weight_sum==0`.
- low-N / all-invalid (CPU, WITNESSED via `tests/test_stacking_low_n_all_invalid_witness.py`,
  18 pass, deterministic float32 HWC 2×2×1, CPU only):
  - Grid CPU empty patches → bare `None`; N=1 valid → input unchanged; all-zero weights /
    all-NaN → zero tiles (all-invalid pixel in mixed mean → `0.0`, division clipped `1e-6`).
  - Grid CPU median: weight magnitude ignored, `weight<=0` treated invalid.
  - `stack_core` (CPU) N=1 valid → input unchanged; empty `images` → `ValueError`; 2D input →
    2D output; all-zero weights/all-NaN → NaN everywhere (all-invalid pixel in mixed mean →
    NaN); median ignores weights entirely.
  - **PINNED DIVERGENCE (SCI-03):** same all-invalid input yields zeros (Grid CPU) vs NaN
    (`stack_core`). No fix applied.
  - Classic low-N (WITNESSED, N=1/N=2 only): `stack_kappa_sigma_clip` N=1 returns input
    unchanged, N=2 per-pixel mean, both `rejected=0.0`; `stack_winsorized_sigma_clip` N=1 logs
    `"Winsorized clip needs >=3 images; forcing CPU."`, returns input `rejected=0.0`.

## Fallback / error / OOM semantics

- Classic GPU (Phase 3 auto): OOM shrink + one retry → hard-disable Phase-3 GPU for the run →
  `_stack_master_tile_cpu` (`worker:15846-15872`).
- Wrapper GPU error → CPU fallback (winsorized WSC parity failure → CPU, `:2133+`).
- Grid GPU OOM/error → CPU `_stack_weighted_patches` (`grid_mode:2098+`); `stack_core`
  unavailable → legacy GPU logic (`:2034+`).
- Phase 4.5 alpha branch failure → `super_arr=None` → configured rejection path.
- SDS `_stack_mosaics` exception → caught; `zemosaic_align_stack` unavailable → plain
  `nanmedian`/`nanmean`.

## Evidence status per row

| Row | Status |
| --- | --- |
| Classic CPU | STATIC ACTIVE; low-N/all-invalid WITNESSED (CPU) |
| Classic GPU | STATIC ACTIVE; NOT_RUN |
| SDS | STATIC ACTIVE (CPU); NOT_RUN |
| Grid CPU | STATIC ACTIVE; low-N/all-invalid WITNESSED (CPU) |
| Grid GPU (core) | STATIC ACTIVE; NOT_RUN |
| Grid GPU (legacy fallback) | DORMANT; NOT_RUN |
| Phase 4.5 (alpha) | STATIC ACTIVE; NOT_RUN |
| Phase 4.5 (configured rejection) | STATIC ACTIVE; NOT_RUN |

## Tests / evidence limits

- WITNESSED: `test_dispatch_propagation_witness.py` (14 pass, dispatch/config rename + winsor
  parsing + silent kwarg drop, TEST-03); `test_stacking_low_n_all_invalid_witness.py`
  (18 pass, low-N/all-invalid/zero-weight CPU characterization, Grid CPU vs `stack_core` vs
  classic N<3, TEST-04); `test_spawn_worker_process_witness.py` (1 pass, TEST-05);
  `test_cache_resume_characterization_witness.py` (18 pass, TEST-06);
  `test_phase3_adaptive_invariants.py` (33 pass/0 skip after TEST-01 import fix);
  `test_packaging.py` (17 pass).
- Historical R0 22 pass/11 skip for `test_phase3_adaptive_invariants.py` is **historical**
  (pre-TEST-01); current witness state is 33 pass/0 skip.
- **NOT_RUN / no witness**: no CPU↔GPU parity, no WSC numerical equivalence, no Phase 4.5
  execution, no Grid GPU `stack_core` backend execution, no classic N≥3 / SDS stacking
  execution, no `linear_fit` normalization/rejection numerical witness. All such cells remain
  UNKNOWN.
- low-N/all-invalid witnessed **CPU backend only**; Grid CPU zeros vs `stack_core` NaN pinned
  (SCI-03), not resolved.
- GPU qualification: CuPy runtime OK on MX150 (compute 6.1, ~2 GB); RawKernel/JIT OK without
  nvcc (NVRTC 12.9 bundled); no GPU stacking execution run in R0.

## SDS backend conclusion

SDS stacking is **CPU-only** in code. The only final-SDS stack call sites
(`worker:38704-38708`, `:38719-38723`) pass `zconfig=None` to `stack_winsorized_sigma_clip` /
`stack_kappa_sigma_clip`, and each wrapper resolves `use_gpu` exclusively from `zconfig`
(`stack_use_gpu` → `use_gpu_stack` → `use_gpu`, then `False` when `zconfig is None`). No SDS
path constructs a `zconfig` that would enable GPU, and `parallel_plan` does not toggle GPU on
these wrappers. There is no SDS GPU branch — do not invent one.

## Remaining UNKNOWN / NOT_RUN

- SCI-01 (Grid CPU/legacy = PixInsight WSC by default vs `stack_core` = simplified median/σ),
  SCI-02 (`stack_core` linear_fit placeholder reachability), SCI-03 (Grid CPU zeros vs
  `stack_core` NaN): kept as STOP/UNKNOWN, not "to fix" here.
- Classic N≥3, SDS runtime, Grid GPU `stack_core`, Phase 4.5 execution, CPU↔GPU parity, WSC
  numerical equivalence, `linear_fit` normalization/rejection numerics: NOT_RUN.

## TODO / R2 boundary

See `todo.md`: pre-R2 characterization witnesses complete; R1 closed with no deletion
preserved; R3 baseline map complete **before** R2 while the final post-R2 R3 audit/report
remains unchecked; first R2 lot bounded to the three pure shared filter helpers
(`_merge_small_groups`, `_split_group_by_orientation`, `_circular_dispersion_deg`) with their
acceptance criteria. Not implemented here.
