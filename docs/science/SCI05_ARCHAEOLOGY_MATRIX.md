# SCI-05 — Gate A Archaeology Matrix: GUI → runtime stacking routes

- **Mission:** `ZM-SCI-05-GATE-A-ARCHAEOLOGY-CONTRACT-20261005`
- **Phase:** implementation (Gate A only — archaeology + draft contract + red witnesses)
- **Repository:** `/home/tristan/.openclaw/workspace/projects/zemosaic`
- **Branch:** `science/zm-sci-05-canonical-stacking-coverage`
- **Base / HEAD:** `ea4b189f2017a03797261c5484702102ad5d3828` (`origin/beta`)
- **Donor (read-only):** `zsss-sci05-donor` worktree @ `9b891de6e7ba71967d03db75d11c3fc279854b26`
- **Status:** DRAFT — archaeology fact-finding only; no science/runtime/config/GUI/dependency/version change.

This matrix records the **current** GUI → persisted-key/token → dispatch/rename →
actually-executed symbol per stacking route. It is an evidence base for the
canonical contract (`SCI05_CANONICAL_STACKING_CONTRACT.md`), **not** a fix and **not**
a parity claim. Line numbers refer to the HEAD above; symbol names are the stable
identity. Cells marked UNKNOWN / NOT_RUN are open — not invented.

---

## 0. Route vocabulary (dispatch map)

GUI (Qt `zemosaic_gui_qt.py`) → config (`zemosaic_config.py`) → worker-process
rename map (`zemosaic_worker.run_hierarchical_mosaic_process`) → per-route engine.

| Route | Entry symbol | Executed backend |
| --- | --- | --- |
| **Classic CPU** | `run_hierarchical_mosaic_classic_legacy` (`worker:23352`) → `_stack_master_tile_auto` → `_stack_master_tile_cpu` | CPU (numpy); GPU flags force-zeroed |
| **Classic GPU (Phase-3 auto)** | `_stack_master_tile_auto` → `_p3_gpu_stack_from_paths` → `gpu_stack_from_arrays` (`zemosaic_align_stack_gpu:1386`) | GPU (cupy) |
| **Classic GPU (wrapper-internal)** | `stack_winsorized_sigma_clip` / `stack_kappa_sigma_clip` / `stack_linear_fit_clip` honoring `use_gpu` (`align_stack`) | GPU (cupy) via `gpu_stack_*` |
| **SDS** | `assemble_global_mosaic_sds` (`worker:37965`) → `_stack_mosaics` (`worker:38683`) | CPU (numpy); wrappers called with `zconfig=None` |
| **Grid CPU** | `run_grid_mode` → `process_tile` → `_stack_weighted_patches` (`grid_mode:1913`) | CPU (numpy) |
| **Grid GPU core** | `_stack_weighted_patches_gpu` (`grid_mode:1984`) → `stack_core(backend='gpu')` | GPU (cupy) |
| **Grid GPU legacy fallback** | `_stack_weighted_patches_gpu` with `stack_core is None` (`grid_mode:2034+`) | GPU (cupy); numpy rejection round-trips |
| **Phase 4.5 alpha-weighted** | `_run_phase4_5_inter_master_merge` `weights_ready` branch (`worker:8315-8337`) | CPU (numpy inline weighted mean) |
| **Phase 4.5 configured rejection** | same fn rejection branch (`worker:8342-8372`) | CPU (numpy); wrappers `zconfig=None` |
| **Global coadd** | `_assemble_global_mosaic_first_impl` (`worker:36511`) | CPU (numpy) + optional GPU reproject helper |

### Rename map (GUI key → worker argument)

`zemosaic_worker.run_hierarchical_mosaic_process` (`worker:36210-36219`):

| GUI/config key | Worker arg |
| --- | --- |
| `stacking_normalize_method` | `stack_norm_method` |
| `stacking_weighting_method` | `stack_weight_method` |
| `stacking_rejection_algorithm` | `stack_reject_algo` |
| `stacking_final_combine_method` | `stack_final_combine` |
| `stacking_kappa_low` / `stacking_kappa_high` | `stack_kappa_low` / `stack_kappa_high` |
| `stacking_winsor_limits` (string) | `parsed_winsor_limits` (tuple; fallback `(0.05,0.05)`) |
| `global_coadd_method` | `global_coadd_method` (consumed as `coadd_method` in plan) |
| `apply_radial_weight` / `radial_feather_fraction` / `min_radial_weight_floor` | `apply_radial_weight_config` / `radial_feather_fraction_config` / `min_radial_weight_floor_config` |

`_config` suffix promotion and silent drop of unknown kwargs are characterized by
TEST-03 (`tests/test_dispatch_propagation_witness.py`).

### Defaults (verified, not trusted)

`zemosaic_config.py` `DEFAULT_CONFIG` and the Qt GUI default block agree:
`stacking_normalize_method=linear_fit`, `stacking_weighting_method=noise_variance`,
`stacking_rejection_algorithm=winsorized_sigma_clip`, `stacking_kappa_low/high=3.0`,
`stacking_winsor_limits="0.05,0.05"`, `wsc_impl=pixinsight`,
`stacking_final_combine_method=mean`, `global_coadd_method=kappa_sigma`,
`poststack_equalize_rgb=False`, `apply_radial_weight=False`,
`radial_feather_fraction=0.8`, `min_radial_weight_floor=0.0`,
`radial_shape_power=2.0`.

> **Qt exposure nuance (verified):** `radial_shape_power` is a persisted default but
> has **no Qt widget** in the stacking group — it is config-only. The three radial
> controls exposed in Qt are `apply_radial_weight`, `radial_feather_fraction`,
> `min_radial_weight_floor`. (Proven by `tests/test_sci05_archaeology_characterization.py`
> `test_legacy_radial_controls_registered`.)

---

## 1. Normalization (`stacking_normalize_method` → `stack_norm_method`)

GUI tokens: `none` ("None") | `linear_fit` ("Linear Fit (Sky)") | `sky_mean` ("Sky Mean Subtraction").

| Route | `none` | `linear_fit` | `sky_mean` | Evidence |
| --- | --- | --- | --- | --- |
| Classic CPU | passthrough | `_normalize_images_linear_fit` (percentile points `_calculate_robust_stats_for_linear_fit`, `a=Δref/Δsrc`, `b=ref_low−a·src_low`, gain clamp ±20, delta gates) | `_normalize_images_sky_mean` (luminance `0.299/0.587/0.114`, `nanpercentile(sky_percentile)`, additive `offset=ref_sky−src_sky`) | STATIC (SCI-02/SCI-04); `none` = default fallthrough (`align_stack:4991-4996`) |
| Classic GPU (Phase-3 auto) | passthrough (`_normalize_frame` only) | `_zas._normalize_images_linear_fit` (called from `_prepare_frames_and_weights`, `align_stack_gpu:1085-1090`) | `_zas._normalize_images_sky_mean`; the low-memory sub-path only computes per-frame offsets via `_compute_sky_mean_offsets` (`:1101-1105`) | STATIC; NOT_RUN |
| Classic GPU (wrapper-internal) | passthrough | same classic `linear_fit`/`sky_mean` (GPU percentiles via `use_gpu_norm`) | same | STATIC; NOT_RUN |
| SDS | n/a (no final-stack normalization; SDS photometry lives in `_normalize_sds_megatiles_photometry`) | n/a | n/a | STATIC (SCI-04) |
| Grid CPU | passthrough | `_fit_linear_scale` per-channel covariance/variance regression (`slope=cov/var`, `intercept=ȳ−slope·x̄`) | **UNSUPPORTED** (Grid normalizes `none`/`linear_fit`/median only) | STATIC + CPU RUNTIME (SCI-02) |
| Grid GPU core | passthrough (upstream `_normalize_patches_gpu`) | `_fit_linear_scale_gpu` (CuPy mirror) | **UNSUPPORTED** | STATIC; ROUTING SEAM (SCI-02); physical GPU NOT_RUN |
| Grid GPU legacy | passthrough | `_fit_linear_scale_gpu` | **UNSUPPORTED** | STATIC |
| Phase 4.5 alpha | n/a (pre-stack B-step may have mutated frames) | inline `linear_fit` (slope clip `[0.25,4.0]`, intercept on unclipped slope, gate `max(5000,1%)` px) | inline `sky_mean` (percentile band `[30,70]`, median delta, no px gate) | RECONSTRUCTION (SCI-04) |
| Phase 4.5 configured | n/a | same inline B-step | same inline B-step | RECONSTRUCTION (SCI-04) |
| Global coadd | n/a (coadd has no normalization) | n/a | n/a | STATIC |

**Headline:** normalization `linear_fit` and rejection `linear_fit_clip` are
**distinct keys and distinct code**. The `stack_core` `linear_fit` is a **median
placeholder** (see §5), bypassed in production because Grid normalizes upstream then
passes `normalize_method='none'`.

---

## 2. Weighting (`stacking_weighting_method` → `stack_weight_method`)

GUI tokens: `none` ("None") | `noise_variance` ("Noise Variance (1/σ²)") | `noise_fwhm` ("Noise + FWHM").

| Route | `none` | `noise_variance` | `noise_fwhm` | Evidence |
| --- | --- | --- | --- | --- |
| Classic CPU | no quality weights (unweighted) | `_compute_quality_weights` → `_calculate_image_weights_noise_variance`: per-channel sigma-clipped (`sigma=3.0/3.0,maxiters=5`) variance `σ²`; global `min_overall_variance` (min finite positive variance, floored `1e-9`); weight `min_overall_variance/σ²`; invalid/≤0 variance → `1e-6`; unprocessed valid frame → `1.0` (mono) / `[1,1,1]` (color) | `_calculate_image_weights_noise_fwhm`; **Photutils unavailable → explicit `noise_variance` substitution** (effective `noise_variance`); **available + all-unusable/star-free → unit weights → sanitizer drops → `(None,"none",None)` (no weighting)**; **partial → `min_overall_valid_fwhm/fwhm` for usable, constant `1e-6` for failed, `1.0` for skipped**; all-usable → genuine `min_fwhm/fwhm` | CPU RUNTIME (SCI-03) + real-path/partial-seam proofs (§7) |
| Classic GPU (Phase-3 auto) | none | `_prepare_frames_and_weights` → `_zas._compute_quality_weights` (same Classic formulas) + radial map + WSC `weights_block` (weights apply only when `combine==mean`) | same Classic `noise_fwhm` behavior | STATIC; NOT_RUN |
| Classic GPU (wrapper-internal) | none | `gpu_stack_*` have **no `weights` parameter** → weights ignored (logged) | n/a | STATIC |
| SDS | manual coverage weights only | n/a (manual weights from `_coverage_weight`) | n/a | STATIC |
| Grid CPU | `_compute_frame_weight` `none|unit|unity` → exposure only (`max(exposure,1e-3)`) | `exposure_w / max(nanstd²,1e-8)` (NOT the Classic `min_var/σ²` formula; exposure-folded) | **variance-only fallback** (`exposure_w/variance`, no FWHM estimate — a different fallback than Classic) | CPU RUNTIME (SCI-03) + dynamic proof (§7) |
| Grid GPU core/legacy | same scalar weight path | same | same | STATIC |
| Phase 4.5 alpha | n/a (alpha maps `np.clip(np.nan_to_num(w),0,1)`) | n/a | n/a | RECONSTRUCTION |
| Phase 4.5 configured | passthrough `weight_method` to wrappers | via wrappers | via wrappers | STATIC |

**Headline:** weighting semantics differ **per route and per method**: Classic `none` is
unweighted while Grid `none` is exposure-folded (DIVERGENT for heterogeneous exposure);
Classic `noise_variance` is `min_overall_variance/σ²` (relative, with `1e-6`/`1.0`
fallbacks) while Grid is `exposure_w/variance` (DIVERGENT); and `noise_fwhm` has
**three distinct reachable Classic behaviors** (variance substitution only when
Photutils is unavailable; no weighting when star-free; genuine/partial FWHM otherwise)
plus a separate Grid variance-only fallback — it is **not** "every route → variance".

---

## 3. Rejection (`stacking_rejection_algorithm` → `stack_reject_algo`)

GUI tokens: `none` ("None") | `kappa_sigma` ("Kappa-Sigma Clip") | `winsorized_sigma_clip` ("Winsorized Sigma Clip") | `linear_fit_clip` ("Linear Fit Clip").

| Route | `none` | `kappa_sigma` | `winsorized_sigma_clip` | `linear_fit_clip` | Evidence |
| --- | --- | --- | --- | --- | --- |
| Classic CPU | `stack_aligned_images` passthrough | `stack_kappa_sigma_clip` → CPU `cpu_stack_kappa` / `_cpu_stack_kappa_fallback` (median/σ clip) | `stack_winsorized_sigma_clip` → `_wsc_pixinsight_stack_numpy` (PixInsight WSC core) or legacy-quantile | `stack_linear_fit_clip` → CPU `cpu_stack_linear` / `_cpu_stack_linear_fallback` (median-residual clip) | STATIC; low-N WITNESSED (TEST-04) |
| Classic GPU (Phase-3 auto) | passthrough | core kappa (established helper via `cp.asnumpy`) | `_wsc_pixinsight_stack_gpu` (WSC PixInsight) | linear core helper | STATIC; NOT_RUN |
| Classic GPU (wrapper-internal) | n/a | `gpu_stack_kappa` (median/σ) | `gpu_stack_winsorized` (winsorize→σ→mean) | `gpu_stack_linear` (median-residual) | STATIC; NOT_RUN |
| SDS | plain `nanmedian`/`nanmean` | `stack_kappa_sigma_clip(zconfig=None)` → CPU | `stack_winsorized_sigma_clip(zconfig=None)` → CPU | n/a (no SDS linear-fit-clip branch) | STATIC |
| Grid CPU | passthrough | `_reject_outliers_kappa_sigma` (astropy `sigma_clipped_stats`) | `_reject_outliers_winsorized_sigma_clip` (PixInsight WSC by default — no `wsc_impl` passed) | **UNSUPPORTED** (no branch → passthrough) | CPU RUNTIME (SCI-01/SCI-03) |
| Grid GPU core | passthrough (verbatim to `stack_core`) | `stack_core` kappa → established helper (CPU round-trip) | `stack_core` **simplified median/σ** (NOT WSC, NOT winsorization) | falls through to no-op | ROUTING SEAM (SCI-01/SCI-03) |
| Grid GPU legacy | passthrough | established helper via `cp.asnumpy` | established helper via `cp.asnumpy` (PixInsight WSC) | **UNSUPPORTED** | STATIC |
| Phase 4.5 alpha | none (no rejection) | n/a | n/a | n/a | STATIC |
| Phase 4.5 configured | passthrough | `stack_kappa_sigma` absent → `stack_kappa_sigma_clip` | `stack_winsorized_sigma_clip` | `stack_linear_fit_clip` | DYNAMIC (SCI-04) |
| Global coadd | n/a | `_finalize_kappa_sigma` (mean + second-moment σ, `kappa_sigma_k=2.0` clip, per-patch re-accumulate) | `_finalize_chunked("winsorized")` (nanpercentile clip + weighted mean — **not** WSC) | n/a | AST seam (§6) |

**Headline:** `winsorized_sigma_clip` has **three different executed meanings**:
PixInsight WSC (Classic CPU/GPU, SDS, Grid CPU/legacy, Phase 4.5 configured),
simplified median/σ (Grid GPU core via `stack_core`), and nanpercentile clip
(global coadd). `linear_fit_clip` is a **visible Qt choice whose actual executed
helpers are median-residual clips** (CPU/GPU wrappers), while the direct helper
`_reject_outliers_linear_fit_clip` is a no-op placeholder with **no callers**
(SCI-02). Grid CPU/legacy have **no `linear_fit_clip` branch** (silent passthrough).

---

## 4. Final combine (`stacking_final_combine_method` → `stack_final_combine`)

GUI tokens: `mean` ("Mean") | `median` ("Median").

| Route | `mean` | `median` | Evidence |
| --- | --- | --- | --- |
| Classic CPU | weighted `Σ(data·w)/Σ(w)`, `where(Σw>1e-9)` else 0; `nan_to_num` then divide | `np.nanmedian` (weights ignored, logged "median with weights not supported") | CPU RUNTIME + STATIC |
| Classic GPU (Phase-3 auto) | `cp.nanmean` | `cp.nanmedian` | STATIC; NOT_RUN |
| Classic GPU (wrapper-internal) | `cp.nanmean` (mean-only on this sub-path) | n/a | STATIC |
| SDS | wrapper weighted mean | wrapper `nanmedian` | STATIC |
| Grid CPU | weighted mean `Σ(data·w)/clip(Σw,1e-6)` | `np.nanmedian` (weight>0 validity gate applied pre-rejection) | CPU RUNTIME (SCI-03/TEST-04) |
| Grid GPU core | `stack_core` mean `where(weight_sum>0,…,nan)` | `stack_core` median (ignores weights entirely) | ROUTING SEAM (SCI-03) |
| Grid GPU legacy | cupy weighted mean `clip(Σw,1e-6)` | numpy round-trip `nanmedian` | STATIC |
| Phase 4.5 alpha | inline `nansum(frames·w)/nansum(w)` (NaN where den≤0) | n/a | RECONSTRUCTION |
| Phase 4.5 configured | wrapper mean | wrapper median / `nanmedian` fallback | STATIC |
| Global coadd | `_finalize_mean` (`sum_grid/weight_grid`, NaN where weight≤0) | `_finalize_chunked("median")` (`np.nanmedian` per chunk) | AST seam (§6) |

**Headline:** mean is **weighted** everywhere but the weight semantics diverge
(magnitude vs sign vs mask); median is **unweighted** everywhere but whether a
`weight<=0` frame is excluded differs (Grid CPU/legacy exclude; `stack_core` does
not). All-invalid: Grid CPU → zero tile; `stack_core` mean → NaN; pinned SCI-03.

---

## 5. `stack_core` shared core (Grid GPU core route)

`zemosaic_stack_core.stack_core(images, weights, stack_config, backend)`:

- **Normalization:** `median` and `linear_fit` run the **identical** per-pixel
  median-subtraction (`linear_fit` is an explicit placeholder); `none` passthrough.
- **Rejection:** `kappa_sigma` → established `_reject_outliers_kappa_sigma` (if
  importable) else median/σ clip; `winsorized_sigma_clip` → **simplified median/σ
  clip** (explicit placeholder, not winsorization, not PixInsight WSC); `none`/
  unknown → no rejection.
- **Combine:** mean `where(weight_sum>0, Σ(w·d)/Σw, nan)` (raw weights, no pre-mask);
  median `nanmedian` (weights ignored); unknown → `ValueError`.
- **Weights:** broadcastable `(N,)` → `(N,1,…)`; no nonpositive pre-mask; no
  `winsor_limits` accepted (dropped by the Grid GPU caller).

Verified by SCI-01 (`tests/test_grid_wsc_characterization.py`), SCI-02
(`tests/test_stack_core_linear_fit_characterization.py`), SCI-03
(`tests/test_grid_mask_weight_characterization.py`) and re-probed lightly in
`tests/test_sci05_archaeology_characterization.py`.

---

## 6. Global coadd (`global_coadd_method` → `coadd_method`)

GUI tokens: `kappa_sigma` | `winsorized` | `mean` | `median`. Default `kappa_sigma`.
Allowed set at `worker:36709`: `{mean, median, kappa_sigma, winsorized}` (invalid → `kappa_sigma`).

Executed finalizers (AST seam, `tests/test_sci05_archaeology_characterization.py`):

| `coadd_method` | Executed symbol | Formula |
| --- | --- | --- |
| `mean` | `_finalize_mean` | `sum_grid / weight_grid` (NaN where `weight_grid≤0`) |
| `kappa_sigma` | `_finalize_kappa_sigma` | mean + second-moment `std=√(E[x²]−E[x]²)`; per-patch `|data−mean| ≤ k·max(std, min_sigma)` clip; `min_sigma=max(5th pct std, 1e-4)`; `count≤1.5` pixels always accepted |
| `median` | `_finalize_chunked("median")` | chunked `np.nanmedian` |
| `winsorized` | `_finalize_chunked("winsorized")` | chunked `np.nanpercentile(low/high)` clip + weighted mean (NOT WSC) |

> **Naming hazard (to freeze):** the GUI label "Global coadd: Winsorized" maps to a
> nanpercentile clip, **not** the `winsorized_sigma_clip` WSC used at master-tile
> level, and **not** the `stack_core` simplified clip. Same-label-different-symbol
> across master-tile vs global-coadd must be resolved in the contract.

---

## 7. `noise_fwhm` reachable behavior (dynamic + real-path + partial-seam proof)

`tests/test_sci05_archaeology_characterization.py` proves, with honest evidence labels:

1. **Photutils unavailable** (reachable dependency seam: photutils is optional,
   `PHOTOUTILS_AVAILABLE` starts `False`) → `_compute_quality_weights` explicitly
   substitutes `_calculate_image_weights_noise_variance`, effective `noise_variance`.
2. **Photutils available + all-unusable/star-free** (REAL PATH, project venv, the
   actual estimator on deterministic star-free 64×64 noise frames) → the estimator
   returns unit weights, the sanitizer sees no effect and returns
   `(None, "none", None)` — requested FWHM becomes **no weighting**, NOT variance.
3. **Partial success** (controlled SEAM: estimator result monkeypatched to the
   realistic mixed `[min_fwhm/fwhm, 1e-6, 1.0]` finalization) → effective stays
   `noise_fwhm`; the `1e-6` (failed) weight is kept, the `1.0` (skipped) frame is
   dropped by the sanitizer.
4. **Grid** `_compute_frame_weight("noise_fwhm") == _compute_frame_weight("noise_variance")`
   (both `exposure_w/variance`) — Grid never computes FWHM; a **separate** fallback
   from Classic, not the same one.

Conclusion: Classic `noise_fwhm` is **not** "→ variance on every route". It is
variance **only** when Photutils is unavailable; otherwise it is no-weighting /
partial-FWHM / genuine-FWHM. Grid's `noise_fwhm` is variance-only with exposure fold.
Under Tristan's absolute fallback policy, **every** branch of this behavior that
silently degrades the requested `noise_fwhm` — variance substitution, effective
no-weighting (`none`), and the hidden `1e-6`/`1.0` constant substitutions — is a
prohibited `SILENT_SCIENCE_FALLBACK` / `SILENT_SCIENCE_DEGRADATION`, even though
the exact numeric outcomes differ (see §9 taxonomy).

---

## 8. Radial weighting (center-radial map) and the footprint-taper gap

ZeMosaic's only "spatial" weighting is the **center-radial cosine map**
(`zemosaic_utils.make_radial_weight_map`): `w = cos(π/2 · clip(r/feather,0,1))^shape_power`,
floor-clamped. It depends on **absolute distance from the image centre**.

`tests/test_sci05_archaeology_characterization.py` proves (RECONSTRUCTION, no donor
import):

- the center-radial map **fails translation invariance** (two identical sub-regions
  at different positions get different weights);
- the donor's footprint-taper *concept* (EDT distance to the real footprint
  boundary → taper) **is** translation/rotation invariant and is a different
  semantic (interior unity, boundary ramp to floor, 0 outside).

This is the target semantic for `s_i` spatial support in the canonical contract
(§see contract), and the deprecation/migration of the center-radial keys
(`apply_radial_weight`, `radial_feather_fraction`, `min_radial_weight_floor`,
`radial_shape_power`) is a **Junior decision**, not chosen here.

### Other stacking controls

| GUI control | Persisted key | Current executed semantics |
| --- | --- | --- |
| Equalize RGB (per sub-stack) | `poststack_equalize_rgb` | `_poststack_rgb_equalization` post-combine gain (per-channel medians, gain clip), invoked by wrappers; classic GPU Phase-3 auto has no post-stack RGB eq | STATIC |
| Apply Radial Weighting | `apply_radial_weight` | multiplies `make_radial_weight_map` into per-frame weights (classic `stack_aligned_images`; GPU Phase-3 auto `_compute_radial_weight_map`) | STATIC |
| Radial Feather Fraction | `radial_feather_fraction` (0.1–1.0) | cosine feather parameter | STATIC |
| Min Radial Weight Floor | `min_radial_weight_floor` (0.0–0.5) | cosine floor | STATIC |
| Radial Shape Power | `radial_shape_power` (config-only, no widget) | cosine exponent | STATIC |

---

## 9. Verdict legend

- **CANONICAL** — one coherent behavior at a given *stage* across all reachable
  routes (or the single route where the control is reachable). **A stage-local
  CANONICAL verdict does not imply identical whole-pipeline outcomes** across
  routes: pipelines still differ via weight pre-masking, quality/radial weight
  combination, and dead-frame filtering. Verdicts above state "stage-local"
  explicitly where that distinction matters.
- **DIVERGENT** — multiple different executed behaviors across routes (pinned, not fixed).
- **PLACEHOLDER** — explicitly incomplete/no-op implementation.
- **SILENT_SCIENCE_FALLBACK** — a requested method silently executes a **different**
  method (e.g. `noise_fwhm` → `noise_variance` when Photutils is unavailable), no
  error, no structured provenance. Prohibited by Tristan's absolute fallback policy.
- **SILENT_SCIENCE_DEGRADATION** — a requested method silently degrades without a
  different-method label: effective no-weighting (`noise_fwhm` → `none` when the
  estimator is star-free) or hidden constant substitutions (`1e-6` for failed
  estimates, `1.0` for skipped frames). Also prohibited by policy. Both terms
  describe the same *kind* of prohibited silent degradation; they differ only in
  whether the effective label changes. Grid's `noise_fwhm` → variance-only is also
  a silent science fallback/degradation.
- **UNSUPPORTED** — a reachable route has no branch for the requested value
  (silent passthrough/ignored).
- **NOT_REACHABLE** — no production route reaches the value at this stage.
- **NOT_RUN** — reachable but never executed in an accepted witness (esp. physical GPU).

Evidence class: **STATIC** (source/AST) | **DYNAMIC** (executed) | **RECONSTRUCTION**
(test-only mirror of an inline block) | **ROUTING_SEAM** (hermetic fake-CuPy /
recording adapter) | **PHYSICAL_GPU_NOT_RUN** (no GPU arithmetic executed).

### Summary verdict (high-level, stage-local semantics separated from pipeline outcome)

A "stage-local" verdict describes what a *single stage* does with a given value
(e.g. `none` = no transformation/no rejection). It does **not** claim identical
whole-pipeline *outcomes* across routes, which still differ due to pre/post
weight-masking, quality/radial weight combination, and dead-frame filtering.

| Control | Verdict |
| --- | --- |
| Normalization `none` | CANONICAL stage-local (passthrough); pipeline outcome still differs by route (weights/masks downstream) |
| Normalization `linear_fit` | DIVERGENT (Grid regression vs classic percentile vs Phase-4.5 inline vs core placeholder) |
| Normalization `sky_mean` | DIVERGENT (classic additive percentile vs Phase-4.5 percentile band vs GPU low-mem offsets); UNSUPPORTED on Grid |
| Weighting `none` | DIVERGENT (Classic unweighted vs Grid exposure-fold `max(exposure,1e-3)`) |
| Weighting `noise_variance` | DIVERGENT (Classic `min_overall_variance/σ²` + `1e-6`/`1.0` fallbacks vs Grid `exposure_w/σ²`) |
| Weighting `noise_fwhm` | DIVERGENT + reachability-dependent (Classic: variance-substitution iff Photutils unavailable, else no-weighting `none` if star-free, else partial `1e-6`/`1.0` / genuine FWHM; Grid: variance-only) — **every degraded branch is a prohibited SILENT_SCIENCE_FALLBACK/DEGRADATION** |
| Rejection `none` | CANONICAL stage-local (passthrough); Grid still pre-masks `weight<=0` → pipeline outcome differs |
| Rejection `kappa_sigma` | DIVERGENT estimator/implementation: Grid/`stack_core` = astropy iterative `sigma_clipped_stats(..., maxiters=5)` helper; Classic CPU wrapper = optional external `cpu_stack_kappa` else internal `_cpu_stack_kappa_fallback` (raw `nanmedian`/`nanstd` single-pass thresholds); wrapper GPU `gpu_stack_kappa` = median/raw-std style; core helper may CPU-roundtrip to the established helper |
| Rejection `winsorized_sigma_clip` | DIVERGENT (WSC vs simplified median/σ vs nanpercentile coadd) |
| Rejection `linear_fit_clip` | DIVERGENT (median-residual wrapper) + PLACEHOLDER (no-op `_reject_outliers_linear_fit_clip`); UNSUPPORTED on Grid |
| Combine `mean` | CANONICAL (weighted) with DIVERGENT invalid/zero semantics |
| Combine `median` | CANONICAL (unweighted valid sample) with DIVERGENT `weight<=0` exclusion |
| Global coadd `kappa_sigma`/`winsorized`/`mean`/`median` | CANONICAL per method (CPU finalizers); naming DIVERGENT for `winsorized` |
| Equalize RGB | CANONICAL (post-stack gain) with GPU-Phase-3-auto gap |
| Radial weighting | CANONICAL (center-radial) — target semantic under review (footprint taper) |

---

## 10. Verification

- `tests/test_sci05_archaeology_characterization.py` — **24 passed** (Gate A red witnesses).
- Adjacent SCI witnesses (re-run, unchanged): see report.
- No source/config/dependency/version change; no commit; no push.
