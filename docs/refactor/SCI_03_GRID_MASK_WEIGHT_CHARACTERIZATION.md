# SCI-03 — Grid mask/weight/all-invalid/alias/winsor_limits characterization (NO FIX)

- **Mission:** `ZM-SCI-03-GRID-MASK-WEIGHT-CHAR-20261004`
- **Phase:** implementation (characterization-only witness)
- **Branch:** `science/sci-03-grid-mask-weight-characterization`
- **Base / HEAD:** `472fd1eaf1ed31fafdea9556aa32ddcf9e30b965` (`origin/beta`)
- **Status:** CHARACTERIZED / CLOSED-NO-FIX — mask ordering, weight semantics, all-invalid behavior, aliases, and `winsor_limits` propagation proven; no correction, no parity claim.
- **Witness:** `tests/test_grid_mask_weight_characterization.py` (31 tests, all passing)

This document records **current behavior and its route-level contracts** so a future product
decision has evidence. It deliberately does **not** correct, harmonize, or claim parity.

---

## 1. Objective

SCI-03 is a **characterization-only** witness. It maps and dynamically proves the current
contracts of the three Grid stacking routes for masks/weights, zero/negative weights,
finite/non-finite samples, per-pixel/all-invalid behavior, aliases, and `winsor_limits`
propagation:

1. **Grid CPU** — `grid_mode._stack_weighted_patches`.
2. **Grid GPU-legacy** — `grid_mode._stack_weighted_patches_gpu` with `stack_core is None`
   (reachable only if the `stack_core` import fails).
3. **Grid GPU-core** — `grid_mode._stack_weighted_patches_gpu` with `stack_core` available,
   which forwards to `zemosaic_stack_core.stack_core`.

It quantifies **three robust divergences** plus the `winsor_limits` drop, without declaring any
algorithm superior and without recommending a correction as fact.

---

## 2. Symbol / route map (anchored to `472fd1e`)

| Route | Entry | Weighting / masking | Rejection | Final combine |
| --- | --- | --- | --- | --- |
| Grid CPU | `_stack_weighted_patches` | `data_stack = where(weight>0, data, nan)` **before** rejection; after rejection `weight_effective = where(finite, weight, 0)` | `kappa_sigma`/`kappa` → `_reject_outliers_kappa_sigma`; `winsorized_sigma_clip`/`winsor` → `_reject_outliers_winsorized_sigma_clip(…, config.winsor_limits, …)` | median (`where(valid_positions, …)`, magnitude ignored) or weighted mean `sum(data*weight)/clip(weight_sum,1e-6)` |
| Grid GPU-legacy | `_stack_weighted_patches_gpu` (`stack_core is None`) | same `where(weight>0, data, nan)` ordering through CuPy; rejection via `cp.asnumpy` round-trip | same helpers, same `config.winsor_limits` pass-through | median (NumPy round-trip) or weighted mean (CuPy, `clip(weight_sum,1e-6)`) |
| Grid GPU-core | `_stack_weighted_patches_gpu` → `stack_core(backend='gpu')` | **no** pre-mask of nonpositive weight; raw `cp_weights` forwarded | `rejection_algorithm` verbatim; `stack_core` dispatches only exact `kappa_sigma` / `winsorized_sigma_clip` | `stack_core` mean `where(weight_sum>0, …, nan)` or median (ignores weights) |

`stack_core` finite-mask is based only on data/rejection result; mean applies the supplied
weights verbatim (including zero/negative if directly supplied) and returns **NaN** at zero
`weight_sum`; median ignores weights entirely. See SCI-02 (`stack_core` normalize/reachability)
and SCI-01 (WSC implementation selection) for adjacent contracts.

---

## 3. Evidence labels

- **STATIC** — read from source/annotations only (no runtime).
- **CPU RUNTIME** — real `_stack_weighted_patches` / `stack_core(backend='cpu')` execution.
- **NUMPY-BACKED GPU ROUTING SEAM** — `_stack_weighted_patches_gpu` exercised with a hermetic
  NumPy-backed fake CuPy; proves *which* code path and *which* call contract, not GPU arithmetic.
- **CORE-CPU ADAPTER** — a recording `stack_core` stand-in hands the captured GPU-core call to
  the *real* `stack_core(backend='cpu')`; CPU core semantics reached *through* the routing seam,
  **not** GPU numerical execution.
- **NOT_RUN** — not executed here.
- **INFERENCE** — reasoned from identical captured call contracts, not from GPU arithmetic.

---

## 4. Numeric matrix (CPU runtime / core-CPU adapter, deterministic float32 `2×2×1` HWC)

Corpora use explicit per-pixel values. Equal-unit weights unless stated. Rejection `none`
unless stated. All results below are reproduced by `tests/test_grid_mask_weight_characterization.py`.

| Case | Grid CPU | Grid GPU-legacy (seam) | Grid GPU-core (core-CPU adapter) | Divergence |
| --- | --- | --- | --- | --- |
| all-zero weights, mean | `0.0` tile, `weight_sum=0` | `0.0` tile, `weight_sum=0` | **`NaN` tile**, `weight_sum=0` | **D1** |
| median, frames `[10,100]`, weights `[0,1]` | `100.0` (zero-weight frame excluded) | `100.0` | **`55.0`** (zero-weight frame included) | **D2** |
| mean, frames `[10,100]`, weights `[2,-1]` | `10.0`, `weight_sum=2` (weight<0 dropped) | `10.0` | **`-80.0`**, `weight_sum=1` (negative weight applied) | **D3** |
| mean, frames `[10,20]`, weights `[1,4]` | `18.0`, `weight_sum=5` (magnitude used) | `18.0` | `18.0` (same weighted mean) | — |
| median, frames `[10,100]`, weights `[-1,1]` | `100.0`, `weight_sum=1` (negative excluded) | `100.0` | `55.0` (weights ignored) | D2 family |
| mixed per-pixel all-invalid | invalid pixel → `0.0`, `weight_sum=0` | `0.0` | `NaN` (per-pixel) | D1 family |
| whole-tile all-invalid | `0.0` tile | `0.0` tile | `NaN` tile | D1 |

D1/D2/D3 are the three robust divergences required; the `winsor_limits` drop is a fourth,
separate propagation divergence (Section 7).

- **Mean uses positive weight magnitude** in Grid CPU/legacy (the `[10,20]`/`[1,4]` case
  yields `18.0`, not the unweighted `15.0`), while `stack_core` applies the raw weight verbatim
  including its sign.
- **Median ignores positive magnitude but honors the `weight>0` gate** in Grid CPU/legacy
  (a `weight<0` or `weight==0` frame is excluded); `stack_core` median ignores weights entirely.
- **The matrix is assembled from focused runtime assertions plus the directly inspected
  route formulas; its key discriminating route cells are exercised explicitly.** The GPU-legacy median row
  (`[10,100]`/`[0,1]` → `100.0`) is proven dynamically through the NumPy-backed routing seam
  (`test_gpu_legacy_seam_median_excludes_zero_weight_frame`), not merely by CPU code similarity;
  the GPU-core mixed per-pixel all-invalid row is proven through the route + core-CPU adapter
  (`test_gpu_core_mixed_per_pixel_all_invalid_pixel_is_nan`).

---

## 5. Mask/weight ordering (dynamic proof)

Both Grid CPU and Grid GPU-legacy apply the `weight>0` gate **before** rejection. The witness
spies on the exact stack handed to `_reject_outliers_kappa_sigma` and proves:

- a finite sample with `weight == 0` arrives at rejection as **NaN** (weight-invalid);
- a finite sample with `weight < 0` arrives at rejection as **NaN** (weight-invalid);
- a finite sample with `weight > 0` arrives at rejection **unchanged** (survives);
- an originally-NaN sample with `weight > 0` stays NaN (data-invalid) — so the two invalidation
  causes (weight-invalid vs data-invalid) are distinguished by origin, not by the shared NaN.

After rejection, `weight_sum` counts only finite surviving samples: a `kappa_sigma` run
(low/high `2.0`) over four `10.0` frames plus one `40.0` outlier rejects the outlier and yields
`weight_sum=4.0` (not `5.0`), with the combined result the mean of the four survivors.

Grid GPU-core, in contrast, does **not** pre-mask: a spy on `zemosaic_stack_core`'s rejection
input proves a zero-weight finite sample (value `1000.0`) remains **finite** at core rejection,
and the recording adapter proves the raw `cp_weights` (including `0` and `-1` entries) are
forwarded unmasked.

---

## 6. Aliases

- **Grid CPU and GPU-legacy** dispatch `kappa` → `_reject_outliers_kappa_sigma` and
  `winsor` → `_reject_outliers_winsorized_sigma_clip` (proven by spy call counts: the alias
  triggers exactly one call of the correct helper and zero of the other).
- **Grid GPU-core** forwards `stack_reject_algo` verbatim into `stack_core`; `stack_core`
  dispatches rejection only for the exact strings `kappa_sigma` / `winsorized_sigma_clip`, so
  the aliases `kappa` / `winsor` fall through to **no rejection**. Proven for **both** aliases
  through the real route + core-CPU adapter: `kappa` → 0 core rejection dispatches
  (`kappa_sigma` → 1), and `winsor` → `rejected_pct == 0.0` with plain-mean output
  (`winsorized_sigma_clip` → `rejected_pct == 20.0` on the four-10.0-plus-40.0 corpus at
  sigma 2.0).
- **`_compute_frame_weight`** method aliases: `none` / `unit` / `unity` all return the exposure
  factor alone (`exposure_w = max(frame.exposure, 1e-3)`); `noise_fwhm` falls back to
  variance-only (a DEBUG emit, then `exposure_w / variance`); an unknown value also falls
  through to variance-only. An all-NaN patch returns `exposure_w` (no usable variance).
  **Exposure factor is included** (longer integrations are rewarded). This characterizes the
  *internal accepted* values (`none|unit|unity`, `noise_variance`, `noise_fwhm`); it does **not**
  assert which values the GUI advertises.
- **Out of scope / route-specific (stated, not silently broadened):** `stack_final_combine`
  aliases. Grid CPU and GPU-legacy branches treat any value other than `median` as `mean`
  (no error). `stack_core` is stricter: it accepts only `mean`/`median` and raises `ValueError`
  for any other value; the outer GPU wrapper then re-raises (when `raise_on_gpu_failure=True`)
  or falls back to the CPU branch otherwise. The core raise and CPU fallthrough are proven
  dynamically (`test_stack_core_raises_on_unsupported_combine` /
  `test_grid_cpu_unknown_combine_falls_through_to_mean`); the wrapper re-raise/fallback is a
  route-specific source fact. No final-combine alias tests are broadened beyond that.
  Normalization aliases (`none`/`unit`/`unity`, `linear_fit`/`linear`,
  default median) are already covered conceptually by SCI-02 and are not re-characterized here.

---

## 7. `winsor_limits` propagation and behavior

- **Grid CPU and GPU-legacy** pass `config.winsor_limits` **exactly** (sentinel `(0.2, 0.1)`)
  to `_reject_outliers_winsorized_sigma_clip` for both the canonical `winsorized_sigma_clip`
  and the alias `winsor` (proven by `*args` spy capturing the second positional argument).
- **Grid GPU-core** builds the core config **without** `winsor_limits` (recording adapter proves
  the key is absent), and `stack_core`'s simplified winsorized path is **invariant** to changing
  Grid `winsor_limits` at fixed images/sigmas (two runs with `(0.2,0.1)` vs `(0.05,0.05)` produce
  bit-identical output).
- In the **default** `resolve_wsc_impl() == pixinsight` path, `winsor_limits` is received but not
  consumed by the WSC helper (the PixInsight core takes only `sigma_low/high`); its behavioral
  effect is visible only on the legacy-quantile path. That implementation-selection contract is
  **SCI-01** territory and is not re-litigated here.

---

## 8. What the witness proves (dynamically)

- **CPU RUNTIME (A).** Mask ordering (`weight<=0` → NaN before rejection), weighted-mean
  magnitude use, median magnitude-ignore + `weight>0` gate, per-pixel/whole-tile all-invalid →
  zero, data-invalid vs weight-invalid distinction.
- **NUMPY-BACKED GPU ROUTING SEAM (B).** GPU-legacy mirrors the CPU mask/rejection ordering
  (nonpositive-weight sample arrives at rejection as NaN) and matches CPU numerically for seam
  cases; returned arrays are host NumPy `float32`; all-zero weights → zero tile; median
  zero-weight finite-frame exclusion is dynamically exercised through the seam (not CPU
  similarity).
- **CORE-CPU ADAPTER (C).** The recording adapter captures raw/unmasked weights + normalized
  finite images + `backend='gpu'` + exact config (no `winsor_limits`), then hands to real
  `stack_core(backend='cpu')`; three divergences (all-zero mean NaN, zero-weight median
  inclusion, negative-weight not clamped) plus rejection-input visibility are quantified, and
  mixed per-pixel all-invalid (valid pixels combine, only the all-invalid pixel is NaN with
  zero per-pixel `weight_sum`) is exercised through the route.
- **ALIAS DISPATCH (D).** CPU/legacy `kappa`/`winsor` → helpers (spy call counts); GPU-core
  forwards both aliases verbatim — `kappa` → no core dispatch and `winsor` → `rejected_pct == 0`
  — while canonical `kappa_sigma` / `winsorized_sigma_clip` dispatch (the latter with
  `rejected_pct == 20.0` on the outlier corpus).
- **STATIC / INFERENCE.** Alias dispatch and `winsor_limits` propagation are proven via spies and
  call counts, not source strings. INFERENCE (from identical captured call contracts) that the
  physical GPU paths follow the same dispatch is labeled, not claimed as GPU arithmetic.
- **NOT_RUN.** Physical GPU numerical execution (no CPU↔GPU parity claim); the default PixInsight
  WSC *behavioral* consumption of `winsor_limits`; classic/SDS/Phase 4.5 parity.

---

## 9. Bounded production note

`grid_mode.process_tile` builds each frame's weight map as
`np.clip(footprint, 0.0, 1.0) * weight_scalar` (footprint clipped to `[0,1]`, times a scalar
`_compute_frame_weight` value). Consequently the **normal** production weight maps are
nonnegative. The negative-weight case (D3) is therefore **contract probing** of what `stack_core`
would do with a directly-supplied negative weight, **not** a claim that production generates
negative weights.

---

## 10. Conclusion

- Grid CPU and GPU-legacy pre-mask `weight<=0` before rejection, use positive weight magnitude in
  mean, gate median on `weight>0`, and return zero (not NaN) for all-invalid pixels.
- Grid GPU-core forwards raw weights and omits `winsor_limits`; `stack_core` returns NaN at zero
  `weight_sum`, includes finite zero-weight frames in median, and does not clamp negative weights.
- Aliases `kappa`/`winsor` are honored by CPU/legacy but forwarded verbatim (→ no rejection) by
  the core; `none|unit|unity` and `noise_fwhm` frame-weight behavior is characterized with the
  exposure factor.
- `winsor_limits` propagates exactly on CPU/legacy and is dropped by the core (invariant there).
- **No algorithm is declared superior; no production behavior changed** (no `src/`, config,
  version, dependency, packaging, or workflow edits).

---

## 11. Recommended decision options (decision only — not implementation)

1. **Document/retain** — accept the current divergent contracts as documented.
2. **Align core pre-mask / invalid semantics** — only under a separate science/API mission (e.g.,
   have `stack_core` treat nonpositive weights as invalid and/or return zero at zero `weight_sum`
   to match Grid CPU), after a full API audit of `stack_core` callers.
3. **Canonicalize aliases / propagate `winsor_limits`** — only under a separate compatibility
   mission (map `kappa`→`kappa_sigma`, `winsor`→`winsorized_sigma_clip` at the core boundary, and
   forward `winsor_limits` into `stack_core`).

Any behavior/API change is a **human gate** and is explicitly out of scope here.
