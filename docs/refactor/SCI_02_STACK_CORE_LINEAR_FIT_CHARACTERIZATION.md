# SCI-02 — `stack_core` `linear_fit` reachability & semantics (NO FIX)

> **HISTORICAL RECORD** (marked 2026-10-06, ZM-ZEGRID-R9). The subject module
> `src/zemosaic/zemosaic_stack_core.py` **no longer exists**: it was dead code
> after the legacy Grid removal (R8) and was deleted in R9. This document is kept
> unmodified as a historical characterization record of that now-removed module;
> it is **not** an active reference and does not describe shipped behaviour.

- **Mission:** `ZM-SCI-02-STACKCORE-LINEARFIT-CHAR-20261004`
- **Phase:** implementation (characterization-only witness)
- **Branch:** `science/sci-02-linear-fit-characterization`
- **Base / HEAD:** `1b964fdf2862a96f84db6c0039a47854822e9d4e` (`origin/beta`)
- **Status:** CHARACTERIZED / CLOSED-NO-FIX — placeholder semantics and reachability proven; no correction, unification, tuning, or algorithm choice.
- **Witness:** `tests/test_stack_core_linear_fit_characterization.py` (16 tests, all passing)

This document records **current behavior and its reachability** so a future product
decision has evidence. It deliberately does **not** correct, harmonize, tune, or choose
an algorithm.

---

## 1. Objective

SCI-02 is a **characterization-only** witness. It answers three questions with evidence:

1. **What does `stack_core(normalize_method='linear_fit')` actually do?** — It is an
   explicit placeholder that executes the *exact same* code as `normalize_method='median'`:
   the per-pixel median across the frame axis is subtracted from every frame. It is **not**
   a genuine per-channel affine mapping.
2. **Is any production route actually reaching that placeholder?** — No. The only in-repo
   production call to `stack_core` (Grid GPU) normalizes *upstream* with a real linear-fit
   normalization and then passes `normalize_method='none'` into the core.
3. **How does it differ from the real upstream/user-facing linear-fit normalization?** —
   The real linear-fit paths (Grid `_normalize_patches` / `_normalize_patches_gpu`, classic
   `_normalize_images_linear_fit`, align-stack GPU) are genuine affine mappings that align
   each patch/image to a reference; the placeholder is median subtraction. The normalization
   key `linear_fit` is also distinct from the **rejection** key/function `linear_fit_clip`.

---

## 2. Symbol / caller map (anchored to `1b964fdf`)

| Symbol | File / function | Role |
| --- | --- | --- |
| `stack_core(images, weights, stack_config, backend, …)` | `src/zemosaic/zemosaic_stack_core.py` | Shared CPU/GPU core. `normalize_method='linear_fit'` branch is an explicit placeholder ("Placeholder: for now, use median. Full linear fit would require per-pixel regression.") that runs the identical median-subtraction code as `'median'`. |
| `_normalize_patches(patches, reference_median, *, method)` | `src/zemosaic/grid_mode.py` | **Grid CPU upstream.** `linear_fit` → `_fit_linear_scale` (per-channel covariance/variance slope + intercept), a genuine affine mapping of each patch onto the reference. |
| `_normalize_patches_gpu(patches, reference_median, *, method)` | `src/zemosaic/grid_mode.py` | **Grid GPU upstream** (CuPy mirror). `linear_fit` → `_fit_linear_scale_gpu` (same affine regression). |
| `_fit_linear_scale(ref_patch, patch)` / `_fit_linear_scale_gpu(...)` | `src/zemosaic/grid_mode.py` | Per-channel slope `cov(x,y)/var(x)` and intercept; the actual Grid affine regression. |
| `_stack_weighted_patches(patches, weights, config, …)` | `src/zemosaic/grid_mode.py` | **Grid CPU.** Calls `_normalize_patches(method=config.stack_norm_method)`; never calls `stack_core`. |
| `_stack_weighted_patches_gpu(patches, weights, config, …)` | `src/zemosaic/grid_mode.py` | **Grid GPU.** Calls `_normalize_patches_gpu(method=config.stack_norm_method)`, then (if `stack_core` available) calls `stack_core(images=normalized, weights=cp_weights, stack_config={'normalize_method':'none', …}, backend='gpu')`. **The only production `stack_core` call site.** |
| `_normalize_images_linear_fit(image_list, reference_index, low_percentile, high_percentile, …)` | `src/zemosaic/zemosaic_align_stack.py` | **Classic/user-facing** linear-fit normalization. Percentile-based (`_calculate_robust_stats_for_linear_fit`: low/high percentile points per channel) → `a = Δref/Δsrc`, `b = ref_low − a·src_low`; a genuine affine mapping. Distinct algorithm from Grid regression. |
| `_calculate_robust_stats_for_linear_fit(image_2d, low_pct, high_pct, …)` | `src/zemosaic/zemosaic_align_stack.py` | Percentile (robust) statistic helper used by classic linear fit. |
| `_reject_outliers_linear_fit_clip(stacked, progress_callback)` | `src/zemosaic/zemosaic_align_stack.py` | **Rejection** placeholder (explicitly "PLACEHOLDER"): returns input unchanged + all-True keep mask. Unrelated to normalization. |
| `stack_linear_fit_clip(frames, …)` | `src/zemosaic/zemosaic_align_stack.py` | Rejection wrapper dispatching to external GPU/CPU linear-fit-clip implementations. Rejection family, not normalization. |

`stack_core` is **internal**: it is not re-exported in `zemosaic.__all__` (which only
exposes `__version__`) and is imported only by `grid_mode` via a guarded relative import.

---

## 3. Exact caller / reachability table

| Caller (module / function) | Backend | `stack_core` receives | `normalize_method` into core | Evidence |
| --- | --- | --- | --- | --- |
| `grid_mode._stack_weighted_patches_gpu` (the **only** production call) | GPU (CuPy) | upstream-normalized `images` + `weights` | **`none`** (linear-fit already applied upstream) | AST caller inventory + ROUTING SEAM (Section 5) |
| `grid_mode._stack_weighted_patches` (Grid CPU) | CPU | — (never calls `stack_core`) | — | ROUTING SEAM forbidden-monkeypatch (Section 5) |
| classic / SDS / Phase 4.5 | CPU/GPU | — (no `stack_core` call) | — | AST inventory (Section 5) |

**Bounded conclusion:** the direct callable placeholder is executable through direct /
internal API use (`stack_core(normalize_method='linear_fit')`), but the current in-repo
production Grid GPU caller bypasses it by passing `none`. No external third-party caller is
claimed to exist or not exist beyond the in-repo AST inventory (external callers are outside
this witness's API contract and are **NOT_RUN**).

---

## 4. Numeric matrix (CPU runtime, deterministic float32 `4×5×3` HWC affine corpus)

Corpus: reference `ref` (per-channel ramps), targets `t1 = ref·[1.3,0.8,1.1]+[20,−10,30]`,
`t2 = ref·[0.6,1.7,0.9]+[150,5,−40]`. Equal unit weights, rejection `none`.

| Route | pixel `[0,0,:]` | mean | median | max\|residual vs ref\| |
| --- | --- | --- | --- | --- |
| `stack_core` `none` / mean | `[153.33, 56.67, 196.67]` | `153.88890` | `179.91666` | `53.33333` |
| `stack_core` `none` / median | `[150, 50, 200]` | `154.25000` | `185.75000` | `66.50000` |
| `stack_core` `median` / mean | `[3.33, 6.67, −3.33]` | `−0.36111` | `−3.33333` | `240.33333` |
| `stack_core` `linear_fit` / mean | `[3.33, 6.67, −3.33]` | `−0.36111` | `−3.33333` | `240.33333` |
| `stack_core` `median` / median | `[0, 0, 0]` | `0.0` | `0.0` | `237.00000` |
| `stack_core` `linear_fit` / median | `[0, 0, 0]` | `0.0` | `0.0` | `237.00000` |
| Grid `_normalize_patches` `linear_fit` (normalized patches) | — | — | — | `3.1e-05` / `7.6e-06` (maps onto ref) |
| Grid `_stack_weighted_patches` `linear_fit` (final stack) | — | — | — | `3.8e-06` (== ref) |

Observations:

- **Placeholder == median, bit-exact.** `linear_fit` and `median` produce identical output
  for both `mean` and `median` combine (`np.array_equal` holds).
- **Placeholder is not affine.** On the affine corpus the real Grid linear-fit maps every
  target onto the reference (residual ≤ `7.6e-06`), while the placeholder leaves residuals up
  to `240.33` — the median-subtraction recentres the stack but does **not** align the affine
  family to the reference.
- **Classic `_normalize_images_linear_fit`** also maps the affine targets onto the reference
  (residual ≤ `3.1e-05`), but via a *different* algorithm (percentile points, not
  covariance/variance regression). The two genuine affine normalizers are distinct code paths
  and neither is the placeholder.

---

## 5. What the witness proves (dynamically)

- **STATIC (A + D).** `stack_core` `linear_fit` == `median` (bit-exact) and differs from
  `none`; shapes/dtypes/weight_sum/rejected_pct pinned. The AST caller inventory finds
  exactly one production call site — `grid_mode._stack_weighted_patches_gpu` — and confirms
  `stack_core` is not exported at the package surface.
- **CPU RUNTIME (A + B + E).** Real execution of `stack_core(backend='cpu')`, Grid
  `_normalize_patches`/`_stack_weighted_patches`, and classic `_normalize_images_linear_fit`
  on the affine corpus. The placeholder is proven non-affine; the real normalizers are proven
  affine (align targets → reference).
- **ROUTING SEAM (B + C).** Grid CPU spies record `_normalize_patches(method='linear_fit')`
  and a forbidden-monkeypatch proves `stack_core` is never called on Grid CPU. The Grid GPU
  routing seam (NumPy-backed fake CuPy + recording `stack_core`) proves `_normalize_patches_gpu`
  receives `linear_fit`, the core is called exactly once with `backend='gpu'` and
  `normalize_method='none'`, and the already-normalized patches (aligned to reference) are what
  reach the core. A guard asserts the placeholder `linear_fit` is **never** passed into
  `stack_core` on this route.
- **NOT_RUN.** Physical GPU numerical execution (no CPU↔GPU parity claim); NaN/masked/finite
  semantics of the placeholder; Classic/SDS/Phase 4.5 parity; external third-party callers.
- **INFERENCE.** That Grid GPU core and Grid GPU upstream would follow the same normalization
  dispatch as their CPU counterparts is inferred from the identical call contracts captured at
  the seam, not from GPU arithmetic.

---

## 6. Normalization vs rejection distinction (explicit)

- Normalization key `linear_fit` selects a **normalization** (median-subtraction placeholder in
  `stack_core`; genuine affine mapping in Grid/classic). It is a value of
  `stacking_normalize_method` / `stack_norm_method` / `stack_core.stack_config['normalize_method']`.
- Rejection key/function `linear_fit_clip` selects a **rejection** algorithm
  (`_reject_outliers_linear_fit_clip` — a no-op placeholder; `stack_linear_fit_clip` — its
  wrapper). It is a value of `stacking_rejection_algorithm` / `stack_reject_algo`.
- `stack_core` implements rejection only for `kappa_sigma` and `winsorized_sigma_clip`; a
  `rejection_algorithm='linear_fit_clip'` there falls through to no-op (identical to `none`,
  `rejected_pct == 0.0`). The witness asserts selecting normalization `linear_fit` never
  invokes the `linear_fit_clip` rejection functions.

---

## 7. Conclusion

- **Placeholder semantics proven:** `linear_fit` == `median` (bit-exact), non-affine.
- **Reachability proven and bounded:** the only in-repo production caller (Grid GPU) normalizes
  upstream and passes `none`; the placeholder is bypassed in current production. The placeholder
  remains directly executable through internal API use. No external-caller claim is made beyond
  the in-repo AST inventory.
- **Real linear-fit paths are distinct and genuine:** Grid (covariance/variance regression) and
  classic (percentile-based) are affine mappings, not the placeholder, and are distinct from each
  other.
- **Normalization vs rejection is explicit** and dynamically asserted.
- **No algorithm is declared superior; no production behavior changed** (no `src/`, config,
  version, dependency, packaging, or workflow edits).

---

## 8. Recommended decision options (decision only — not implementation)

1. **Keep documented** (accept as-is: the core `linear_fit` option is a documented median
   placeholder, unreachable in the current in-repo Grid path).
2. **Remove the unsupported advertised core option** — only after a full API audit (external /
   programmatic callers) proves nothing relies on `stack_core(normalize_method='linear_fit')`.
3. **Implement genuine core affine normalization** — only under a separate science/API mission.

Any behavior/API change is a **human gate** and is explicitly out of scope here.
