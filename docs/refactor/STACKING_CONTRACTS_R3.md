# ZeMosaic — Stacking Contracts (R3) — PRELIMINARY R0 map (rework-1)

Mission: `ZM-ARCH-CLEANUP-R0-20261003`
Phase: rework-1 (R0 documentation/archaeology corrections only)
Date: 2026-10-03

> **PRELIMINARY R0 map.** This is a static, evidence-linked inventory assembled during R0
> archaeology. It is **not** a validation of numerical parity. Entries marked UNKNOWN /
> NOT_RUN / SUSPECTED must be treated as open, not invented. It must be completed (and any
> CPU/GPU parity claims backed by witnesses) **before** any R2 extraction touches stacking.

Legend: **requested** = config/flag asked for; **effective** = post-guard/fallback decision;
**executed** = what actually ran. Static evidence only unless a runtime witness is cited.

## GPU / JIT qualification fact (corrected)

CuPy 14.1.1 ships its own NVRTC (`cupy_backends.cuda.libs.nvrtc`, version 12.9).
`RawKernel`/`RawModule`/JIT work **without `nvcc`** — a `RawKernel` witness succeeded on the
MX150 (`[1,2,3,4]` → `[3,4,5,6]`). Qualification limits are ~2 GB VRAM and that GPU stacking
and CPU↔GPU parity were **NOT_RUN** in R0, **not** JIT availability. `nvcc` absence is
irrelevant to CuPy JIT.

## Mode/backend matrix (actually reachable)

| Row | Entry / caller | Backend (requested → effective) | Evidence |
| --- | --- | --- | --- |
| Classic CPU | `stack_aligned_images` ← `run_hierarchical_mosaic_classic_legacy` (`worker:15644-15677`) | CPU (numpy); GPU flags zeroed by `_stack_master_tile_cpu` (`worker:15586-15604`) | static |
| Classic GPU | `gpu_stack_winsorized` / `gpu_stack_kappa` / `gpu_stack_linear` (`align_stack:1539/1628/1695`) | GPU (cupy), gated by `_plan_gpu_stack_execution` (`:1139`) | static |
| Grid CPU | `_stack_weighted_patches` (`grid_mode:1913`) | CPU | static |
| Grid GPU (core) | `_stack_weighted_patches_gpu` → `stack_core(backend='gpu')` (`grid_mode:1984-2028`) | GPU if `_CUPY_AVAILABLE` + `config.use_gpu` | static; runtime CuPy OK (MX150) |
| Grid GPU (legacy fallback) | same fn, `stack_core` None → legacy cp rejection (`grid_mode:2034+`) | GPU, numpy rejection via `cp.asnumpy` | static, reachable only if `stack_core` import fails |
| SDS | inline SDS final stack → `stack_winsorized_sigma_clip` / `stack_kappa_sigma_clip` (`worker:38702-38730`) | CPU (numpy) via `zemosaic_align_stack` | static |
| Phase 4.5 (configured rejection) | `_run_phase4_5_inter_master_merge` rejection branch (`worker:8342-8372`) | **CPU (numpy)** — wrappers called with `zconfig=None` | static; see below |
| Phase 4.5 (alpha-weighted early branch) | same fn `weights_ready` branch (`worker:8315-8337`) | CPU (numpy direct weighted mean) | static; bypasses reject/combine |

## Columns per method

### Median combine
- Classic: `final_combine_method='median'` → `np.nanmedian` (median **ignores weights**).
  Same in `stack_core` and Grid CPU/GPU median branches.
- Weights: ignored for the median itself; masks/NaN handled via `np.isfinite` gating.

### Mean combine
- Weighted mean: `sum(data*w)/sum(w)` with `weight_sum` clipped at `1e-6` (Grid CPU
  `:1958-1961`); `stack_core` uses `where(weight_sum>0, .../..., nan)`; classic uses
  `_xp_errstate` guarded division.
- Weights: scalar per-frame (`(N,)` broadcast) or per-pixel planes (`(N,H,W[,C])`).

### Kappa-Sigma rejection (corrected — NOT divergent on GPU)
- Classic CPU/GPU: `_reject_outliers_kappa_sigma` (`align_stack:4371`) — established
  implementation; `sigma_clip_low/high` from `stack_kappa_low/high`.
- Grid CPU: same established helper (`grid_mode:1943`).
- Grid GPU (core): `stack_core` (`zemosaic_stack_core.py:291-317`) — when the imported
  helper is available, GPU does `cp.asnumpy(stacked)` then calls the **same established
  `_reject_outliers_kappa_sigma`**; CPU calls it directly. The simplified median/σ clip
  fallback runs **only when the helper import failed**, for **both** backends.
- `core/robust_rejection` users: `wsc_pixinsight_core` / `wsc_pixinsight_core_streaming_numpy`
  / `wsc_parity_check` are used by `zemosaic_align_stack` and `zemosaic_align_stack_gpu`
  (WSC path), **not** by `grid_mode`.

### Winsorized Sigma Clip rejection (the genuinely divergent core path)
- Classic CPU/GPU: `_reject_outliers_winsorized_sigma_clip` (`align_stack:4486`) — full
  winsorize-then-clip with `winsor_limits`, `winsor_max_workers`, `winsor_max_frames_per_pass`,
  memory planning (`WinsorMemoryPlan` `:732`, `WinsorStreamingState` `:849`).
- Grid CPU: established helper (`grid_mode:1950`).
- Grid GPU (core): `stack_core` **simplified** winsorized = plain median/σ clip, no
  winsorize pass (`zemosaic_stack_core.py:320-327`), for both backends. **Structural
  divergence from Grid CPU (SCI-01); numeric impact unmeasured.** This is *not* PixInsight
  WSC — Grid is not a WSC consumer.

### WSC (PixInsight-style, distinct from winsorized-sigma clip)
- Classic/SDS: `resolve_wsc_impl` → `wsc_pixinsight_core` / `wsc_pixinsight_core_streaming_numpy`
  (`align_stack:1394-1520`), streaming chosen by `_should_use_wsc_streaming` (`:1294`);
  `wsc_parity_check` (`:1050`) for CPU/GPU parity.
- Grid: **not** a WSC consumer (uses kappa/winsor rejection helpers). Do not call the
  `stack_core` winsorized-sigma path "WSC".

### linear-fit NORMALIZATION (distinct from rejection)
- Classic: `_normalize_images_linear_fit` (`align_stack:3296`), invoked when
  `normalize_method='linear_fit'` (`:4991`).
- Grid CPU/GPU: `_normalize_patches` / `_normalize_patches_gpu` `method='linear_fit'`
  (`grid_mode:1798/1853`).
- `stack_core`: `linear_fit` is a **placeholder** (median subtraction, `:294`) but Grid GPU
  normalizes upstream and passes `'none'` to the core.

### linear-fit REJECTION (distinct from normalization)
- Classic: `stack_linear_fit_clip` (`align_stack:2526`) → `_reject_outliers_linear_fit_clip`
  (`:4726`) / `gpu_stack_linear` (`:1695`).

## Phase 4.5 (corrected rework-2 — A…E temporal map)

Inside `_run_phase4_5_inter_master_merge` (`worker:7335`), backend CPU/numpy throughout.
The normalization/photometry data-flow is five distinct steps in exact order:

- **A. Helper-gated pre-stack affine / micro-align branches — DORMANT (missing helper).**
  Availability gates `micro_align_available` / `photometry_estimator_available` /
  `photometry_apply_available` = `hasattr(zemosaic_align_stack, ...)` (`worker:7412-7414`).
  At BASE these are `False False False` (no `micro_align_stack` / `estimate_affine_photometry` /
  `apply_affine_photometry` defined anywhere in src). Micro-align (`:8034`), intra-group
  affine (`do_chunk_photometry` `:7701-7705`, calls `:7766/:8084/:8095`), legacy affine
  (`:8151-8154`) are INERT. Config flags `photometry_intragroup`/`photometry_intersuper`/
  `photometry_clip_sigma` read at `:7364-7366`. NOT PROVEN DEAD.
- **B. ACTIVE pre-stack explicit linear_fit / sky_mean normalization** (`worker:8202-8290`),
  mutates `frames` **in place** before stacking. Gate `norm_method in ("linear_fit",
  "sky_mean") and len(frames) >= 2`; `norm_method` from `stack_cfg["stacking_normalize_method"]`
  → `normalize_method` → `stack_norm_method` → `none` (`:7676-7700`).
  - `linear_fit`: per channel, common-finite mask, min overlap `max(5000, ceil(1%))`,
    sigma clip via `clip_sigma_norm` (=`photometry_clip_sigma`, clipped `[0.1,10.0]`,
    `:8168-8174`), OLS `slope=dot(xv,yv)/dot(xv,xv)` **clipped `[0.25,4.0]`**,
    `intercept=ym-slope*xm`, applied in place to valid source pixels.
  - `sky_mean`: per channel, percentile band `[sky_low,sky_high]` (`intertile_sky_percentile`,
    default `30,70`, `:8175-8200`), per-channel medians, additive `delta=bg_ref-bg_src`.
  - Exceptions are caught and logged, and processing continues; `frames`/channels are mutated
    **in place** inside the nested loops with **no copy/rollback**, so an exception after
    prior assignments can leave **partial in-place normalization** for the already-processed
    frames/channels.
- **C. Stacking — alpha-weighted early branch OR configured rejection branch.**
  1. **Alpha-weighted early branch** (`worker:8315-8337`): `weights_ready = any(isinstance(w,
     np.ndarray) for w in frame_weights)`. When true: weight maps `np.clip(np.nan_to_num(w,
     nan=0.0), 0.0, 1.0)`, non-array/mismatched entries → `ones`; `super_arr =
     np.where(den>0, nansum(frames*w)/nansum(w), np.nan)`; `alpha_out =
     (np.nanmax(weight_stack, axis=0)*255).astype(uint8)`. **Bypasses configured
     reject/combine.** Mask/NaN: NaN weights→0; NaN output where `den<=0`; alpha = max
     weight ×255 (uint8). On exception → configured rejection path.
  2. **Configured rejection branch** (only when `super_arr is None`, `worker:8342-8372`):
     `reject_algo = stack_cfg.get("reject_algo", stack_cfg.get("stack_reject_algo",
     "winsorized_sigma_clip"))` (`:7682`). `winsor`/`winsorized_sigma_clip` →
     `stack_winsorized_sigma_clip`; `kappa_sigma` → `stack_kappa_sigma` (if present) else
     `stack_kappa_sigma_clip`; `linear_fit_clip` → `stack_linear_fit_clip`. All `zconfig=None`
     (`use_gpu=False`) → **CPU/numpy**, `weight_method`, `stack_kwargs`. Last resort
     `nanmedian`/`nanmean` (`final_combine`).
- **D. ACTIVE post-stack inter-super gain-only normalization** (`worker:8622-8832`),
  independent of the absent helpers. Gate `photometry_intersuper and
  len(candidate_super_tiles) >= 2` (`:8624-8629`). Reads saved super-tile FITS
  (`fits.open(memmap=True, do_not_scale_image_data=True)`), per-channel medians with
  valid-pixel counts as weights (IQR clip when `photometry_clip_sigma>0`), weighted-median
  reference (fallback dominant tile), gains `ref/med` **clipped `two_pass_cov_gain_clip`
  default `[0.85,1.18]`** (`:8700-8712`), reopens each FITS `mode="update"`, multiplies
  channels in place, writes `ZM45NORM=True` + HISTORY (`:8758-8795`). **Reachable at BASE.**
- **E. Global-affine inter-super branch** (`worker:8837+`, `estimate_affine_photometry` at
  `:8865`) — helper-gated (`photometry_estimator_available and photometry_apply_available`),
  **INERT at BASE**.

- **Backend: CPU (numpy)**. Wrappers resolve `use_gpu` from `zconfig` only
  (`stack_winsorized_sigma_clip` `align_stack:1792-1795`; `stack_kappa_sigma_clip`
  `:2413-2416`; `stack_linear_fit_clip` `:2540-2543`); `zconfig=None` → `use_gpu=False`.
  `parallel_plan` does not enable GPU; `stack_cfg_phase45` has no GPU key.
- **`inter_master_stack_method` / `inter_cfg["stack_method"]` is passed but not consumed**;
  actual rejection selection is `stack_cfg["reject_algo"]`.
- **Nothing applies affine photometry to `super_arr` in memory.** B acts on `frames`
  pre-stack; D rewrites saved super-tile FITS post-stack.
- Evidence status: A/E DORMANT (missing helper); B/D ACTIVE STATIC; C ACTIVE STATIC;
  no Phase 4.5 path executed at runtime (NOT_RUN).

## Config aliases / precedence (stacking keys)

- GUI/config canonical names: `stacking_normalize_method`, `stacking_weighting_method`,
  `stacking_rejection_algorithm`, `stacking_kappa_low/high`, `stacking_winsor_limits`,
  `stacking_final_combine_method` (`zemosaic_config.py:114-121`).
- Worker process wrapper rename map (`worker:36251-36265`):
  `stacking_normalize_method → stack_norm_method`, `stacking_weighting_method → stack_weight_method`,
  `stacking_rejection_algorithm → stack_reject_algo`, `stacking_final_combine_method → stack_final_combine`,
  `stacking_kappa_low/high → stack_kappa_low/high`; `stacking_winsor_limits` parsed to `parsed_winsor_limits` tuple.
- Internal aliases consumed by `_compute_quality_weights`: `noise_variance` | `noise_fwhm`
  (fallback→`noise_variance`) | `none`/unknown → no weights.
- `use_gpu` precedence: `stack_use_gpu` → `use_gpu_stack` → `use_gpu` (align_stack wrappers);
  Grid: `use_gpu` param → `use_gpu_grid` config.

## dtype / axes / NaN-Inf / masks / low-N

- HWC (channels-last) is the canonical internal form; `_ensure_hwc_tile` (`stack_core:71`)
  and `_ensure_hwc_array` (`grid_mode:387`) normalize; CHW→HWC when `C` small.
- float32 output throughout stacking (`stack_core`, Grid `_stack_weighted_patches`, classic).
- NaN masking: classic and Grid mask non-positive weights / non-finite data before combine;
  `stack_core` uses `xp.isfinite` gating and `nan` for `weight_sum==0`.
- low-N / all-invalid: Grid CPU returns zero tiles when `not any(valid_positions)`
  (`grid_mode:1960`); `stack_core` returns `nan` where `weight_sum<=0`; classic filters
  statistically-dead frames (`_filter_statistically_dead_frames`) post-normalization.
- `winsor_limits` aliases: `(low,high)` tuple; default `(0.05,0.05)`; string parsed in worker.

## Fallback / error / OOM

- Grid GPU OOM → caught, logs, returns CPU result (`grid_mode:2098+`). Grid `stack_core`
  unavailable → legacy GPU logic (`:2034+`).
- Phase 3 GPU: `_stack_master_tile_auto` retries once after OOM shrink, then hard-disables
  Phase-3 GPU for the run and falls back to `_stack_master_tile_cpu` (`worker:15763-15872`).
- Phase 4.5 alpha branch failure → `super_arr=None` → configured rejection path.
- Backend error distinction: requested vs effective vs executed must be recorded in any
  future parity witness; R0 makes no parity claim.

## Tests / evidence limits

- Witnessed: `test_grid_mode_dbe.py` (DBE + star protection, not full stacking parity),
  `test_grid_mode_stack_plan_paths.py` (CSV path resolution only),
  `test_phase3_adaptive_invariants.py` (22 passed, 11 silently skipped — worker-source
  text invariants NOT exercised due to flat `importorskip("zemosaic_worker")`).
- **NOT_RUN / no witness**: no CPU↔GPU parity measurement, no WSC numerical equivalence
  measurement, no low-N/all-invalid characterization, no Phase 4.5 execution. All such
  cells remain UNKNOWN.
- GPU qualification limits: CuPy runtime OK on MX150 (compute 6.1, ~2 GB); RawKernel/JIT OK
  without nvcc (NVRTC 12.9 bundled); no GPU stacking execution was run in R0.
