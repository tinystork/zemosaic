# SCI-04 — Classic / SDS / Phase 4.5 variants inventory characterization (NO FIX)

- **Mission:** `ZM-SCI-04-CLASSIC-SDS-PHASE45-VARIANTS-20261004`
- **Phase:** implementation (characterization-only witness)
- **Branch:** `science/sci-04-classic-sds-phase45-variants`
- **Base / HEAD:** `5ff5be553550ab2ba8bb0acd8455c616d267ed35` (`origin/beta`)
- **Status:** CHARACTERIZED / CLOSED-NO-FIX — route/difference inventory plus SDS pure-helper
  contracts and Phase 4.5 vs Classic normalization/stacking differences quantified; no
  consolidation, unification, refactor, fix, or default change.
- **Witness:** `tests/test_classic_sds_phase45_variants_characterization.py` (34 tests, all passing)

This document records **current behavior and its differences** so a future consolidation
decision has evidence. It deliberately does **not** correct, harmonize, tune, or choose an
algorithm.

---

## 1. Objective

SCI-04 is a **characterization-only** witness. It maps and, where cheaply feasible,
dynamically proves the current DIFFERENCES among:

1. The **Classic legacy pipeline** (`run_hierarchical_mosaic_classic_legacy`), non-grid
   non-SDS;
2. **SDS mode** (dispatcher continuation with SDS helpers);
3. The **Phase 4.5 pre-stack path** (`_run_phase4_5_inter_master_merge`);
4. plus **Grid** and the shared **core** (`stack_core`) as reference rows;
5. and **low-N** behavior and **chunking** mechanisms.

It is an inventory, not a unification. Any future consolidation is gated on the
"consolidation prerequisites" list in Section 6.

---

## 2. Route / difference inventory (evidence class per row)

**STATIC** = AST/import/source inventory; **DYNAMIC** = executed against real code on a
deterministic corpus; **RECONSTRUCTION** = executed against a faithful test-only mirror of an
inline production block (the block is not importable as a standalone callable); **NOT_RUN** =
not executed (heavy compute / requires full pipeline / GPU).

| Route | Entry point (owner) | Normalization | Weighting | Rejection | Combine | low-N | Chunking | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Classic legacy** | `zemosaic_worker.run_hierarchical_mosaic_classic_legacy` (dispatcher falls through when not grid and not SDS) | `_normalize_images_linear_fit` / `_normalize_images_sky_mean` (`zemosaic_align_stack`) when `stack_norm_method` set | `stack_weight_method` → `_compute_quality_weights` | `stack_winsorized_sigma_clip` / `stack_kappa_sigma_clip` / `stack_linear_fit_clip` / `stack_aligned_images` (align_stack wrappers) | wrapper combine / `stack_aligned_images` combine | kappa/winsor N<3 → force CPU, valid stack `rejected=0.0` (TEST-04) | own `_apply_safe_dynamic_chunk_profile` call at entry | STATIC (route) + cross-ref TEST-04/SCI-02 |
| **SDS** | dispatcher `run_hierarchical_mosaic` continuation when `sds_mode_flag` (own SDS resolution block duplicated from legacy — ARCH-05) | SDS helpers `_sds_compute_tile_payload` / `_normalize_sds_megatiles_photometry` (median-based photometry) | coverage maps via `_sanitize_sds_megatile_payload` | `_mask_sds_low_coverage_pixels` masks low-coverage pixels (not sigma-clip rejection) | `_finalize_sds_global_mosaic` (nanize by coverage/alpha) | not separately characterized (shared CPU stack wrappers with `zconfig=None`) | `_apply_safe_dynamic_chunk_profile` at entry | DYNAMIC (helpers) + STATIC (route) |
| **Phase 4.5 pre-stack** | `zemosaic_worker._run_phase4_5_inter_master_merge` | active pre-stack `linear_fit` (slope clip `[0.25,4.0]`) / `sky_mean` (percentile median delta) — inline, mutates `frames` in place | alpha-weighted direct branch: `np.clip(np.nan_to_num(w,nan=0),0,1)` | alpha branch: **none**; configured-rejection branch: `stack_winsorized_sigma_clip` / `stack_kappa_sigma_clip` / `stack_linear_fit_clip` | alpha branch: `nansum(frames*w)/nansum(w)` (NaN where den<=0); rejection branch: wrapper combine | not separately characterized (shared wrappers) | `max_group` bounded chunk loop over tile members | RECONSTRUCTION (normalization + alpha branch) + STATIC (chunk loop) + DYNAMIC (wrapper existence) |
| **Grid CPU** | `grid_mode._stack_weighted_patches` | `_normalize_patches` (`_fit_linear_scale` regression) | masks `weight<=0` before rejection | `_reject_outliers_winsorized_sigma_clip` (PixInsight default) | mean uses positive weight magnitude; median honors `weight>0` gate | all-invalid → zeros + `weight_sum=0` (TEST-04/SCI-03) | n/a (grid tile batches) | cross-ref SCI-01/02/03 |
| **Grid GPU core** | `grid_mode._stack_weighted_patches_gpu` → `stack_core` | upstream-normalized then `normalize_method='none'` | raw/unmasked weights | `stack_core` simplified median/σ clip | `stack_core` mean/median | zero `weight_sum` → NaN (mean); negative weights unclamped (SCI-03) | n/a | cross-ref SCI-01/02/03 |
| **core** | `zemosaic_stack_core.stack_core` | `linear_fit` = placeholder == `median` (SCI-02) | broadcast weights | kappa/winsorized simplified | mean/median | — | n/a | cross-ref SCI-02 |

**Distinct entry points proven:** `run_hierarchical_mosaic` (dispatcher) and
`run_hierarchical_mosaic_classic_legacy` are two separate module-level functions in
`zemosaic_worker`; the dispatcher calls the legacy wrapper (AST call-site inventory) and the
Grid runner `grid_mode.run_grid_mode`. Each stacking route is owned by its module/function as
tabulated above (AST function-definition inventory).

---

## 3. SDS pure-helper contracts (DYNAMIC, deterministic tiny arrays)

All five helpers are module-level, importable, and were executed directly on the real code
with explicit NaN masks (no `equal_nan`).

### `_mask_sds_low_coverage_pixels(mosaic_hwc, coverage_hw, min_keep_fraction, target_hw)`

- Coverage normalized by its max (`max_cov`); pixels with `coverage_norm < min_keep_fraction`
  are masked: coverage zeroed and mosaic set to NaN (broadcast over channels for HWC).
- No-op when `coverage_hw` is None, empty, `max_cov <= 0`, or `frac <= 0`; returns the
  original mosaic identity for the None case.
- `target_hw` reshapes 1D coverage (size == H*W) to 2D.
- Summary fields: `{"max_cov": <float>, "masked_pixels": <int>}`.
- Pinned example: coverage `[[1,2],[3,4]]`, `frac=0.5` → `max_cov=4.0`, `masked_pixels=1`
  (only normalized `0.25` masked), coverage `[0,0]=0.0`, mosaic `[0,0]=NaN`, non-masked
  pixels retain original coverage.

### `_sanitize_sds_megatile_payload(mosaic_hwc, coverage_hw, alpha_hw)`

- Coverage from `coverage_hw`, or from `alpha_hw` normalized by its max when coverage is None.
- `valid_mask = (coverage > 1e-6) & isfinite(coverage)`; mosaic NaN outside `valid_mask`.
- Alpha sanitized to `uint8` via `clip(0,255)`; when no alpha input, alpha derived from
  `coverage/max*255` (uint8).
- Coverage fallback `ones` when both coverage and alpha are None (or 1D dropped / shape
  coercion fails); mismatched 2D shape cropped/padded to target.
- 1D coverage dropped (cannot be reshaped meaningfully) → fallback ones.
- Pinned example: coverage `[[1,2],[3,4]]` → alpha `[[63,127],[191,255]]` uint8;
  alpha-only `[[255,128],[64,0]]` → coverage `[0,1]≈0.50196`, zero-alpha pixel NaN in mosaic.

### `_sds_compute_tile_payload(tile_arr, coverage_arr)`

- Median, or coverage-weighted median with a **1% of peak** threshold (pixels below
  `0.01*coverage_max` excluded, falling back to `cov>0` when the threshold excludes all).
- All-invalid fallback → median `1.0`; abs/positive normalization (`abs(median)`, fallback
  `abs` of full array).
- Stats fields: `{"coverage_weight": sum(cov), "coverage_pixels": count(cov>0),
  "coverage_max": max(cov)}` (zeroed when coverage is None).
- Pinned example: `[[1,2],[3,4]]` + unit coverage → median `2.5`, `coverage_weight=4.0`,
  `coverage_pixels=4`, `coverage_max=1.0`.

### `_sds_choose_reference_index(payloads, requested_index)`

- Requested valid index wins; otherwise max `coverage_weight` (strictly positive); central
  fallback (`count // 2`) when no positive weight; empty list → `0`.
- Pinned example: weights `[0.1,0.9,0.5]` → `1` (both for `None` and out-of-range `99`).

### `_normalize_sds_megatiles_photometry(mega_tiles, coverages, ref_index)`

- Per-tile gain `ref_median / tile_median`; reference tile returned untouched; non-finite or
  `<=0` gain → `1.0`; empty → `[]`.
- Pinned example: `[2.0, 4.0]` tiles, `ref_index=0` → bright tile scaled to `≈2.0`
  (gain `0.5`); zero-median tile → gain `1.0` (untouched).

---

## 4. Phase 4.5 vs Classic stacking differences (bounded)

### 4.1 Stacking entry (DYNAMIC existence + RECONSTRUCTION)

- **Phase 4.5 alpha-weighted direct branch** is an **inline** numpy weighted-mean — it is
  **not** a call to any `stack_*` wrapper. When per-frame weight maps are present
  (`weights_ready`), it runs `num/den` with `np.clip(np.nan_to_num(w, nan=0.0), 0.0, 1.0)`
  and `alpha_out = (nanmax(weight_stack, axis=0)*255).astype(uint8)`. It **bypasses**
  configured reject/combine. Characterized via a faithful test-only reconstruction
  (RECONSTRUCTION): equal unit weights → plain mean; NaN weight → 0; weight `>1` → clipped to
  `1.0`; `den<=0` → NaN output.
- **Phase 4.5 configured-rejection branch** (`super_arr is None`) calls the **real**
  `zemosaic_align_stack` wrappers: `stack_winsorized_sigma_clip` (winsor/winsorized_sigma_clip),
  kappa via `stack_kappa_sigma` **if present else** `stack_kappa_sigma_clip`, and
  `stack_linear_fit_clip`. **DYNAMIC proof:** `stack_winsorized_sigma_clip`,
  `stack_kappa_sigma_clip`, `stack_linear_fit_clip`, `stack_aligned_images` are all callable,
  while `stack_kappa_sigma` **does not exist** (`hasattr` is `False`) — so the Phase 4.5 kappa
  branch in fact resolves to `stack_kappa_sigma_clip`.
- **Classic legacy route** calls the **same** real `zemosaic_align_stack` wrappers
  (`stack_winsorized_sigma_clip` / `stack_kappa_sigma_clip` / `stack_linear_fit_clip`) plus
  the generic `stack_aligned_images` fallback (STATIC AST + cross-ref STACKING_CONTRACTS_R3).
  The **discriminator** is that Classic has **no** alpha-weighted direct branch — it always
  goes through a rejection wrapper, whereas Phase 4.5 prefers the inline weighted-mean when
  weights are present.

### 4.2 Active pre-stack normalization `linear_fit` vs `sky_mean` (RECONSTRUCTION)

The inline block in `_run_phase4_5_inter_master_merge` is not importable as a standalone
callable, so it is characterized via a faithful test-only reconstruction (no formula copied
into production; labeled RECONSTRUCTION, not a direct production-call proof).

- **`linear_fit`** (per channel): common-finite mask; `min_overlap_required =
  max(5000, ceil(1%))` common pixels **hard gate**; optional sigma clip
  (`clip_sigma_norm`, default 3.0); OLS `slope = dot(xv,yv)/dot(xv,xv)`; skip when
  `denom <= 0` or non-finite; `intercept = ym - slope*xm` computed from the **unclipped**
  slope; then `slope` **clipped `[0.25, 4.0]`**; applied in place to valid source pixels.
  - Known-affine corpus `src = 2*ref + 10` (6400 px) → maps back to `ref` (slope `0.5`
    within range).
  - `src = 0.1*ref` → raw slope `10` **clipped to `4.0`**, result `0.4*ref` (not `ref`),
    proving the clip changes the transform and that the intercept uses the unclipped slope.
  - Tiny corpus (16 px) below the 5000-px gate → **no-op**; constant source → `denom<=0`
    → **no-op**.
- **`sky_mean`** (per channel): percentile band `[sky_low, sky_high]` (default `30,70`);
  per-channel medians within the band; additive `delta = bg_ref - bg_src` applied in place.
  **No pixel-count gate** (unlike `linear_fit`).
  - Corpus `src = ref + 50` → offset removed (`src → ref`) even on a tiny 4×4 corpus.
- **Discriminating difference:** on a tiny corpus `linear_fit` is a no-op while `sky_mean`
  removes the offset — a concrete behavioral divergence between the two active methods
  (RECONSTRUCTION, quantified in Section 5).

**No claim about dormant absent Phase 4.5 helpers** beyond SCI-07 status (the affine /
photometry helpers `estimate_affine_photometry` / `apply_affine_photometry` / `micro_align_stack`
are absent at BASE and gated off; documented in STACKING_CONTRACTS_R3 "Phase 4.5 A…E temporal
map"). No claim of correctness/superiority/parity.

---

## 5. Difference matrix (concrete values where executed)

| Item | Classic legacy | SDS | Phase 4.5 | Evidence |
| --- | --- | --- | --- | --- |
| Stacking entry | `stack_winsorized_sigma_clip` / `stack_kappa_sigma_clip` / `stack_linear_fit_clip` / `stack_aligned_images` | SDS helpers (`_sds_compute_tile_payload` → `_normalize_sds_megatiles_photometry` → `_finalize_sds_global_mosaic`) | alpha-weighted inline (weights present) OR rejection wrappers (weights absent) | STATIC + DYNAMIC (wrapper existence) + RECONSTRUCTION |
| `stack_kappa_sigma` exists? | n/a (calls `_clip`) | n/a | **No** (`hasattr` False → falls to `stack_kappa_sigma_clip`) | DYNAMIC |
| Pre-stack normalize | `_normalize_images_linear_fit` / `_normalize_images_sky_mean` (align_stack) | n/a (median photometry) | inline `linear_fit` / `sky_mean` (mutates in place) | STATIC + RECONSTRUCTION |
| linear_fit slope clip | n/a (percentile-based classic) | n/a | `[0.25, 4.0]` (intercept from unclipped slope) | RECONSTRUCTION |
| linear_fit min-overlap gate | n/a | n/a | `max(5000, ceil(1%))` common px → tiny corpora no-op | RECONSTRUCTION |
| sky_mean pixel-count gate | n/a | n/a | none (offset removed even on 4×4) | RECONSTRUCTION |
| low-coverage masking | n/a | `_mask_sds_low_coverage_pixels` (normalized `< min_keep_fraction`) | n/a | DYNAMIC |
| low-N (N<3) | kappa/winsor force CPU, valid stack `rejected=0.0` (TEST-04) | shared CPU wrappers (`zconfig=None`) | shared wrappers | cross-ref TEST-04 |
| all-invalid | n/a | n/a (per-helper fallback median 1.0) | n/a | DYNAMIC (helper) / cross-ref SCI-03 |
| Chunking | `_apply_safe_dynamic_chunk_profile` (config profile) | `_apply_safe_dynamic_chunk_profile` (config profile) | `max_group` chunk loop (tile members) | DYNAMIC (profile) + STATIC (loop) |

---

## 6. Consolidation prerequisites (inventory only — no recommendation executed)

These are the facts that must be **proven** before any future unification; this mission does
**not** perform any of them.

1. **Phase 4.5 execution** — the Phase 4.5 pre-stack path has never been executed at runtime
   (NOT_RUN); its inline normalization/alpha branch are characterized only via
   reconstruction. Any unification requires a real Phase 4.5 run (or a promoted seam) to
   confirm the reconstructed semantics against production.
2. **SDS real-data behavior** — SDS helpers are proven pure/small on tiny arrays, but no SDS
   run on real data (physical GPU / Seestar data) is claimed. Consolidating SDS masking/photometry
   requires a real SDS run.
3. **Classic N≥3** — Classic kappa/winsor with N≥3 is **not** covered (TEST-04 only covers
   N<3). Full low-N parity must include N≥3 before unifying low-N handling.
4. **Grid GPU `stack_core` physical execution** — Grid GPU core arithmetic is NOT_RUN
   (hermetic seams only); no CPU↔GPU parity claim.
5. **Dormant affine/photometry helpers** — SCI-07 status (absent at BASE, gated off) must be
   resolved before touching the Phase 4.5 photometry flow.
6. **SDS flag resolution precedence** — pinned STATIC-only (inline in both pipeline
   functions); a hermetically reachable seam (or a promoted helper) is needed before any
   consolidation of the duplicated resolution blocks (ARCH-05).
7. **DBE chunked RBF evaluation** — inventoried STATIC only (heavy compute); its chunk
   mechanism must be executed before any chunking unification.

---

## 7. Chunking mechanisms distinguished (three different mechanisms)

1. **`_apply_safe_dynamic_chunk_profile`** (DYNAMIC) — a **config/zconfig profile applier**.
   Modes `safe_dynamic` (default), `safe_dynamic_plus`, `aggressive`, `baseline` (raw values,
   no keys applied), invalid/None/empty → `safe_dynamic`. Sets 9 keys (e.g.
   `parallel_target_cpu_load`, `parallel_target_ram_fraction`, `parallel_gpu_vram_fraction`,
   `phase5_chunk_auto`, `phase3_ram_high/critical_pct`, `phase3_chunk_scale_high/critical`).
   Safe when cfg is None/non-dict or zconfig is None. Pinned per-mode values in the witness.
2. **Phase 4.5 `max_group` chunk loop** (STATIC; heavy execution NOT_RUN) — a **tile-member
   iteration loop** inside `_run_phase4_5_inter_master_merge`:
   `group_chunks = max(1, ceil(len(members)/max_group))`, iterating
   `range(0, len(members), max_group)`. Bounds the number of tiles processed per chunk
   (default `max_group=64`).
3. **Phase 5 VRAM chunk-budget helper** `_compute_phase5_vram_budget_bytes` (DYNAMIC) — a
   **byte-budget calculator** keyed on power state / VRAM probe (fraction AC `0.80`, battery
   `0.25`; unknown probe → fallback `128` MiB). Distinct from both (1) and (2).
4. **DBE chunked RBF evaluation** (STATIC; heavy compute NOT_RUN) — chunks the **evaluation
   points** of a scipy RBF model (`chunk_points = max(1024, max_eval_pairs_chunk // n_samples)`)
   to bound per-chunk memory. Distinct from all of the above.

---

## 8. Evidence status / NOT_RUN

- **DYNAMIC:** route entry existence; all five SDS pure helpers; `_apply_safe_dynamic_chunk_profile`
  (all modes + invalid fallback + safe-when-missing); `_compute_phase5_vram_budget_bytes`
  (power-state fractions + unknown-probe fallback); Phase 4.5 rejection wrapper existence
  (including `stack_kappa_sigma` absence).
- **RECONSTRUCTION:** Phase 4.5 inline `linear_fit`/`sky_mean` normalization; inline
  alpha-weighted branch; SDS flag resolution precedence chain (uses real `_coerce_bool_flag`).
- **STATIC:** route ownership (AST call-site + function-definition inventory); Phase 4.5
  `max_group` chunk loop; DBE chunked RBF evaluation; Classic legacy stacking call sites.
- **NOT_RUN:** Phase 4.5 full execution; SDS real-data run; physical GPU / Grid GPU
  `stack_core` arithmetic; DBE heavy compute; classic N≥3; any CPU↔GPU parity.

No correctness/superiority/parity claim is made anywhere; no source, config, version, or
dependency change was made.

---

## 9. Test list (34 tests)

Route inventory (7): dispatcher/legacy distinct entry points; dispatcher→legacy/Grid static
calls; module ownership static; SDS helpers importable; Phase 4.5 rejection wrapper existence;
SDS flag resolution precedence (reconstruction). SDS helper contracts (13): mask (normalize,
no-op cases, reshape), sanitize (coverage/alpha/valid-mask/fallback/1D-drop/coercion),
compute-payload (median/weighted/all-invalid/abs/stats), choose-reference, normalize-megatiles.
Phase 4.5 vs Classic (6): alpha branch vs rejection; alpha nan_to_num+clip; linear_fit affine
normalize; linear_fit slope clip; linear_fit skip tiny/constant; sky_mean percentile delta;
linear_fit vs sky_mean distinct. Chunking (8): profile per-mode ×3, baseline, invalid fallback,
safe-when-missing, phase5 budget distinct, phase45 chunk loop static, DBE chunked static.
