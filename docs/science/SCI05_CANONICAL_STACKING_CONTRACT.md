# SCI-05 — Canonical Stacking Contract (JUNIOR SCIENTIFIC ACCEPT — IMPLEMENTATION CONTRACT)

- **Mission:** `ZM-SCI-05-GATE-A2-CONTRACT-FREEZE-20261005`
- **Phase:** Gate A2 decision freeze — Junior scientific decisions recorded; contract frozen as the implementation target.
- **Repository:** `/home/tristan/.openclaw/workspace/projects/zemosaic`
- **Branch:** `science/zm-sci-05-canonical-stacking-coverage`
- **Base:** `ea4b189f2017a03797261c5484702102ad5d3828` (`origin/beta`)
- **Donor (read-only):** `zsss-sci05-donor` @ `9b891de6e7ba71967d03db75d11c3fc279854b26`

> ## ⚠️ STATUS: JUNIOR SCIENTIFIC ACCEPT — IMPLEMENTATION CONTRACT
>
> This document records the **frozen Junior scientific decisions** that Gate B must
> implement. It is no longer a candidate/decision register: every previously-open
> decision point (N1–N4, W1–W3, S1–S3, R1–R3, C1–C2, G1) is resolved in §16.
>
> **Evidence:** Nono independent review of Gate A archaeology (`SCI05_ARCHAEOLOGY_MATRIX.md`
> + the archaeology witnesses) reached `review-2: ACCEPT`. Junior scientific acceptance
> of these decisions is recorded below.
>
> **Review status of THIS freeze: ACCEPT.** Nono review-0 findings F1–F5 were bounded
> contract-fidelity/clarity defects only (no science reopened); REWORK-2 resolved all five,
> and independent Nono review-1 returned **ACCEPT**. Junior independently verified the
> resulting contract. Gate B (implementation/port) may open. No runtime source/test/config/
> GUI/locales/deps/version/packaging/workflow change accompanies this freeze.

---

## 1. Purpose and non-goals

Freeze a single canonical target pipeline and its per-stage semantics so Gate B can
implement the coverage-aware support domain without silently changing the existing
stacking science. This document is the **target contract**; it does not claim any of
this is implemented at HEAD.

**Canonical pipeline order (frozen):**

```
ALIGNED INPUTS
  → NORMALIZATION
  → SCALAR QUALITY WEIGHTS
  → GEOMETRIC/SCIENCE VALIDITY + SUPPORT TAPER
  → OUTLIER REJECTION
  → COMBINATION
  → [OPTIONAL explicit post-combine RGB equalization]
  → CanonicalStackResult(science, estimator_weight_sum, support_w1, support_w2,
    n_eff_support, rejection_mask/diagnostics, provenance)
```

Coverage **render** is later/downstream and never mutates `CanonicalStackResult`
science or support (§12).

Non-goals: no canonical-engine implementation here, no algorithm replacement, no GUI
change, no coverage port yet, no commit/push. This is the contract only.

---

## 2. Canonical engine (request/result) — decision A

- One **backend-neutral engine**:
  `CanonicalStackRequest(images, geometric_support, normalization, weighting, rejection, combine, optional_reference_index, parameters)`
  → `CanonicalStackResult(science, estimator_weight_sum, support_w1, support_w2, n_eff_support, rejection_mask/diagnostics, provenance)`.
- HWC/HW inputs accepted; internal canonical form is HWC **float32 inputs** with
  **float64** statistics/accumulation; output `science` is **float32** restoring the mono
  shape; support maps are **exactly 2-D float64** internally.
- `estimator_weight_sum` has the **same spatial/channel shape as `science`** (HW mono,
  HWC RGB — rejection may be channel-specific), float64 internally and in the result, and
  is **separate from the 2-D positive support maps**: for `mean` it is the per-pixel/channel
  **sum of `w_i`** over original surviving valid samples; for `median` it is the
  per-pixel/channel **count** of original surviving valid samples with `w_i > 0` (unit
  effective estimator weights — positive weight magnitude intentionally ignored); `0` where
  no survivors (see §9).

---

## 3. Reference selection — decision B (N1)

- The caller may provide an **explicit valid `reference_index`**.
- Otherwise: choose the frame with the greatest `count_nonzero(m_i)` — the count of
  channel-invariant valid geometric-support pixels `m_i` on the aligned **PRE-normalization**
  input validity (finite/all-channel), with **no weights and no rejection involved**; on a
  stable tie, choose the **lowest original input index**.
- The reference frame is **identity** (no normalization applied to it).
- The **same chosen reference** is used across channels and methods.
- Provenance records `explicit`/`auto` and the index.

---

## 4. Valid sample and failure rules — decision C

- Geometric support **must be explicit 2-D**; never infer it from brightness/nonzero.
- A science sample is **valid iff** geometric support is true **AND** (mono finite OR all
  RGB channels finite). This is channel-invariant 2-D validity.
- NaN/Inf → invalid.
- `quality weight == 0` means **absent**; `quality weight < 0` or nonfinite is **invalid
  input** and fails **before mutation**.
- **All-invalid output** = NaN/invalid science, `estimator_weight_sum = 0`, `support = 0`.
  **No historical zero sentinel.**
- A method failure is **never** a different method or a no-op disguised as success.
  A frame whose requested normalization/quality metric cannot be computed is explicitly
  excluded **with reason**; if the reference is invalid or no frame remains, the stack
  request **fails cleanly**.
- N=1 normalization is a **documented identity success**, not a fallback.

---

## 5. Normalization — decision D (N2/N3/N4)

**`none`:** exact passthrough on valid samples.

**`linear_fit`:** per-channel affine `source → reference`, convention `y_ref = a*x_src + b`,
fit on common 2-D validity. Exact estimator:
1. require `min_common = max(256, ceil(0.01 * min(ref_valid_count, src_valid_count)))`
   common pixels;
2. initial unweighted float64 OLS;
3. residual center = median, scale = `1.4826*MAD`; if `scale > 0` retain residuals within
   the inclusive interval `±3*scale`; if `scale == 0` retain only residuals exactly equal
   to the residual median; the residual-mask update is **monotonic**; refit; max 5
   iterations or stable (mask unchanged);
4. after every update require at least `min_common` retained samples AND a finite
   nondegenerate OLS denominator — otherwise the frame **fails explicitly**; require
   finite `a,b` and `0.25 <= a <= 4.0`; **out-of-range FAILS the frame (do not clip
   slope)**; final intercept from final accepted set;
5. apply only to valid pixels; invalid remain invalid.

**`sky_mean`:** true name/formula, **not median**: same reference/common-valid/min_common
rule; per channel Astropy-equivalent iterative sigma-clipped **MEAN**
(`sigma_lower=sigma_upper=3`, `maxiters=5`); offset `mean_ref - mean_src`; apply additive
offset to valid pixels. N=1 identity. Failure excludes the frame explicitly.

**`stack_core` placeholder (`linear_fit == median`):** must **disappear from supported
paths**. Old callers route to the canonical implementation or error — **never** median
substitution.

---

## 6. Quality weights — decision E (W1/W2/W3)

- **Exactly one scalar per frame**, shared by all channels (preserve color); **no**
  per-channel/per-pixel quality magnitude. Geometric/taper remain separate 2-D factors.
- **`none`:** scalar `1.0`; **NO exposure fold**.
- **robust noise sigma:** computed on the **normalized valid samples** (pipeline order),
  retaining channel-invariant 2-D validity and a scalar per-frame weight: luminance for
  RGB (`0.2126R + 0.7152G + 0.0722B`), mono direct; geometric-valid finite samples;
  sigma-clipped std (`3/3`, `maxiters=5`), float64. Nonfinite/nonpositive sigma → metric
  failure.
- **`noise_variance`:** raw `1/sigma²`, computed float64, then normalize all surviving
  positive frame weights by the maximum raw weight so max = 1; no epsilon/constant/unity
  fallback; failed frame excluded explicitly.
- **`noise_fwhm`:** raw `1/(sigma² * FWHM²)` (background-limited point-source justification:
  PSF area proportional FWHM²), then same max=1 normalization. FWHM = median
  `equivalent_fwhm` from Photutils sources satisfying finite `0.8 < FWHM < 20 px`,
  eccentricity ≤ 0.8, minimum 3 accepted sources. Missing Photutils disables the GUI
  option; persisted/programmatic request fails validation **before run**. Per-frame
  insufficient sources excludes the frame explicitly. **No** variance/no-weight/unit/1e-6
  fallback.
- Quality raw weights are computed **after** normalization exclusions; max-normalization is
  applied **only over surviving positive weights** (frames with a failed/absent metric are
  excluded first).
- Quality weight magnitudes affect **only** weighted-mean combination and the quality factor
  `q_i` in positive support (§7 — always present, `q_i = 1` for `none`); rejection
  thresholds/masks **never** consume weight magnitudes.

---

## 7. Support / coverage — decision F (S1/S2/S3 + donor)

- `q_i` = normalized canonical scalar quality weight, or `1` for `none`.
- `m_i` = explicit channel-invariant 2-D valid geometric/science sample mask.
- `a_i` = donor-exact footprint taper if enabled, else `1` within mask and `0` outside.
- Per original exposure (before rejection): the quality contribution to support is **not
  optional** once the request is resolved — `s_i == w_i == q_i * m_i * a_i`, with `q_i = 1`
  for `none`. The **SAME factor** multiplies numerator and denominator (constant-field
  invariant).
- "Independent" support means the support pair is **accumulated independently** and is
  **never reconstructed** from `estimator_weight_sum`, the final estimator WHT, or rejection
  survivors/masks; it does **not** mean `q` is omitted. Rejection **does not** change
  support.
- **Donor support-accumulator invariants (restored):**
  - Each `s_i` must be **exact-shape 2-D**, **finite**, and **non-negative**.
  - `s_i²` and candidate cumulative W1/W2 are preflighted: negative / NaN / Inf / shape
    mismatch / square overflow / cumulative overflow **fail before mutation**.
  - `SUP_W1` and `SUP_W2` are an **atomic pair** — either both update or neither; a failed
    add/restore leaves both unchanged.
  - Target internal accumulators remain **float64** (float32 is **not** reintroduced as the
    target default or alternative).
  - `N_eff` is a **pure derived view** that never mutates state: exact-first `W1²/W2` where
    finite; overflow-resistant `(W1/√W2)²` fallback where `W1²` would overflow; undefined /
    nonfinite / negative → documented neutral `0`.
  - Support accumulates in **original-exposure ordered-add** semantics; never derived
    post-hoc from estimator maps or rejection masks.
  - Required donor algorithms (`make_footprint_taper` and the support accumulator) are
    **source-ported / reimplemented locally into ZeMosaic** with **no runtime import or
    dependency on ZSSS**.
- median value ignores positive weight magnitude but requires `w_i > 0` validity; support
  still records `q/m/a`.
- donor exact `make_footprint_taper`: boolean 2-D footprint, padded false border, scipy EDT
  primary + donor chamfer fallback, `feather_px=8.0`, `floor=0.0`; `1` interior / ramp /
  `floor` / `0` outside. Taper preserves **translation/rotation invariance** and
  **padded-boundary symmetry (up to rasterization)**.

---

## 8. Rejection — decision G (R1/R2/R3)

**General:** per pixel/channel, operate on **original normalized valid samples**; weight
magnitudes do **NOT** affect rejection. Rejection mask may be channel-specific diagnostics;
2-D support remains independent.

**`none`:** keep all valid.

**`kappa_sigma`** — UNIQUE canonical algorithm CPU/GPU: float64; initial survivors = valid;
each iteration center = median(survivors), dispersion = population std (`ddof=0`) of
survivors; keep original samples inside `[center - sigma_low*std, center + sigma_high*std]`;
monotonic mask; max 5 or stable. Defaults `low/high = 3.0`. If valid N<3, explicit low-N
no-rejection success (not fallback). Degenerate `std<=0` keeps equal finite samples, rejects
values unequal to center only if outside the zero-width interval; diagnostics explicit.

**`winsorized_sigma_clip`** — UNIQUE canonical true WSC using **BOTH** winsor limits and
sigma limits:
1. survivors = valid; require N≥3 else explicit no-rejection success;
2. on current survivors compute per-pixel/channel lower/upper quantiles at
   `winsor_limit_low` and `1 - winsor_limit_high` (defaults `0.05/0.05`; each finite in
   `[0, 0.5)`);
3. winsorize survivors to those quantile bounds;
4. center = mean and dispersion = population std of winsorized survivors;
5. update **MONOTONIC** survivor mask by applying asymmetric
   `[center - sigma_low*std, center + sigma_high*std]` to **ORIGINAL** normalized samples;
6. max 5 or stable; final combine uses **original surviving samples**, not winsorized
   replacements.

Defaults sigma low/high = 3.0. Degenerate policy (explicit): if the winsorized population
std <= 0, use a **zero-width inclusive interval** at the winsorized mean applied to the
**ORIGINAL** normalized samples — original samples exactly equal to the center survive,
unequal values are rejected; the survivor-mask update remains a **monotonic intersection**;
diagnostics record the degenerate case. Stable means the survivor mask is unchanged. WSC
quantile convention is pinned to NumPy/CuPy `method='linear'` semantics (or a mathematically
exact equivalent if the backend API differs), so CPU/GPU/chunked implementations cannot
choose different interpolation variants. Remove implementation selector/env ambiguity
(`wsc_impl` **cannot** choose another science on supported paths). CPU/GPU same `xp`
algorithm; chunking may change resource use only, not operation semantics beyond documented
numeric tolerance.

**`linear_fit_clip`:** DISABLE/REMOVE from Qt supported choices and the canonical enum.
Persisted/programmatic token fails validation with explicit `unsupported_removed_sci05`;
**do NOT** migrate to `none`/`kappa`/WSC. Legacy internal helpers may remain temporarily
unreachable, clearly legacy, with no supported caller.

---

## 9. Combine — decision H (C1/C2)

**`mean`:** float64 `SUM(original_surviving_normalized_sample * w_i) / SUM(w_i)`; only valid
survivors; denominator condition is **exactly `> 0`** (no route-specific epsilon);
`denom <= 0` → NaN invalid + `estimator_weight_sum = 0`. `estimator_weight_sum` is the
per-pixel/channel **sum of `w_i`** over original surviving valid samples (float64).

**`median`:** float64 unweighted median of original surviving samples for which `w_i > 0`;
positive weight magnitude ignored; **no** invented weighted median; no survivors → NaN.
`estimator_weight_sum` is the per-pixel/channel **count** of original surviving valid
samples with `w_i > 0` (unit effective estimator weights, **not** `Σ q·m·a`); `0` where no
survivors. Output float32.

---

## 10. Global coadd — decision I (G1)

- GUI labels `Mean`, `Median`, `Kappa-Sigma`, `Winsorized` route to the **SAME canonical
  engine/method contracts**, processed in spatial chunks if needed.
- The current percentile-winsorized global implementation is **removed** from the supported
  `Winsorized` route (no same-label approximation).
- No rename workaround selected: **unify science**.

---

## 11. Equalize RGB — decision J

- Keep as an explicit **OPTIONAL POST-COMBINE SCIENCE TRANSFORM**, never hidden in
  normalization/rejection and never in coverage render.
- Same existing robust implementation for all supported callers:
  `equalize_rgb_medians_inplace`, background percentile `5/85`, min samples `5000`,
  min coverage `0.01`, gain clip `[0.95, 1.05]`; requested/effective/applied + gains logged.
- If inapplicable, explicit no-op status, not falsely "applied". Scientific result reflects
  it when enabled.

---

## 12. Coverage UX / migration / render — decision K

- Replace normal modern radial controls with **Coverage support taper** (default ON) and
  **Coverage-aware final reconstruction** (default OFF).
- Internal taper params remain `8.0/0.0`; render tuning is internal, not GUI.
- Deprecated old keys: `apply_radial_weight`, `radial_feather_fraction`,
  `min_radial_weight_floor`, `radial_shape_power`. **Never** map values mathematically.
  Migration sets the legacy radial path disabled/inert; both old `true` and old `false` yield
  the independent fresh canonical support-taper default ON; neither old feather/floor/power
  value maps to new px/floor. Record migration/provenance.
- Donor exact render formula: `alpha = clip(1 - N_eff/n_ref, 0, 1)`, `n_ref=32`,
  `sigma_denoise=2`, `sigma_low=32`; B+D detail blend. No support → no-op/fail-open
  diagnostic; high support no-op; no gain/inpainting/low-frequency coverage correction.
- Render acts ONLY on a separate preview/display render array. Scientific FITS
  pixels/header/WCS and `CanonicalStackResult`
  science/estimator_weight_sum/SUP_W1/SUP_W2/N_eff/rejection diagnostics are **exactly
  unchanged** ON/OFF. No rendered FITS overwrite.

---

## 13. Fallback / provenance — decision L

- Backend fallback GPU→CPU allowed **only** under the same exact method/params/masks
  contract, logged.
- Any method unavailable/failed is an explicit validation/frame-exclusion/request failure as
  above; **no substitution**.
- Bounded `STACK_EFFECTIVE` and coverage events record requested/effective normalization,
  weighting, rejection, combine, backend, reference, excluded frames/reasons, taper, render,
  parameters, and backend fallback reason. FITS headers describe the **actual executed**
  method.

---

## 14. Implementation gates — decision M (sequence only; no code now)

- **B** — normalization + weighting + deterministic witnesses.
- **C** — rejection + combine.
- **D** — CPU/GPU convergence / physical qualification.
- **E** — donor Coverage transplant + settings/GUI/migration/render A/B.
- **F** — caller convergence: Classic / SDS / Grid / Phase 4.5 / global coadd.
- **G** — final independent audit + one full suite + optional real-data/human science gate.

Full suite runs **once at final**. New ZeGrid is out of scope.

---

## 15. Archaeology separation and no-implementation claim

The current-behavior archaeology is **kept separate** in
`docs/science/SCI05_ARCHAEOLOGY_MATRIX.md` (9 routes; per-column executed symbols; verdicts
CANONICAL / DIVERGENT / PLACEHOLDER / SILENT_SCIENCE_FALLBACK / SILENT_SCIENCE_DEGRADATION /
UNSUPPORTED / NOT_REACHABLE / NOT_RUN; evidence class STATIC/DYNAMIC/RECONSTRUCTION/
ROUTING_SEAM/PHYSICAL_GPU_NOT_RUN). **This contract does not claim any target semantics are
implemented at HEAD** — it is the specification Gate B will build toward.

**Donor discrepancy (preserved, do not hide):** the donor's Qt settings dataclass
(`settings_state.py:269`) defaults `apply_coverage_render=True`, while the engine instance
(`queue_manager.py:5131`) defaults `apply_coverage_render=False`. These conflict;
**Tristan's fresh target is OFF**, and ZeMosaic must follow Tristan, not silently infer from
the incidental Qt dataclass default.

## 16. Resolved decision table (frozen)

| ID | Resolved decision |
| --- | --- |
| N1 | Reference selection: explicit valid `reference_index`; else greatest `count_nonzero(m_i)` (channel-invariant valid mask) on aligned PRE-normalization validity, no weights/rejection; stable tie → lowest index; identity reference; same reference across channels/methods; provenance `explicit`/`auto` + index. |
| N2 | Single `linear_fit` estimator: float64 OLS + robust MAD refinement (median center, `1.4826*MAD`, ±3·scale, monotonic, ≤5 iters), `0.25 ≤ a ≤ 4.0`, out-of-range fails frame (no clip). |
| N3 | Low-N/min-pixel gate + failure: `min_common = max(256, ceil(0.01 * min(counts)))`; failure excludes frame with reason; N=1 normalization is identity success; no silent substitution. |
| N4 | `stack_core` `linear_fit==median` placeholder removed from supported paths; old callers route canonical or error, never median substitution. |
| W1 | `noise_fwhm`: real estimator (Photutils `equivalent_fwhm` median, `0.8<FWHM<20`, ecc≤0.8, ≥3 sources); missing Photutils disables GUI / persisted fails validation; per-frame insufficient sources excluded; no fallback. |
| W2 | No exposure fold (`none` = scalar `1.0`); `noise_variance` = raw `1/σ²` then max=1 normalization; no floor/epsilon/constant/unity fallback. |
| W3 | Exactly one scalar weight per frame, shared across channels; geometric/taper are separate 2-D factors. |
| S1 | All-invalid output = NaN/invalid science + `estimator_weight_sum=0` + `support=0`; no historical zero sentinel. |
| S2 | `weight==0` = absent; `weight<0`/nonfinite = invalid input failing before mutation. |
| S3 | Channel-invariant 2-D support: `m_i` explicit 2-D; `s_i == w_i == q_i*m_i*a_i` per exposure (`q=1` for `none`); `SUP_W1/W2` float64 atomic pair, fail-before-mutation (negative/NaN/Inf/shape/overflow), original-exposure ordered-add, never derived from estimator WHT/rejection; `N_eff` pure derived view `W1²/W2` (overflow-resistant `(W1/√W2)²` fallback, neutral `0` if undefined); donor algorithms source-ported, no runtime ZSSS dependency. |
| R1 | Unique WSC target: true WSC using both winsor+sigma limits (PixInsight-style defaults); unify `stack_core` simplified and global-coadd percentile clip onto it. |
| R2 | kappa: median center + population std `ddof=0`, sigma `low/high=3.0`, ≤5 iters; WSC winsor `0.05/0.05`, sigma `3.0`; N<3 → explicit no-rejection success. |
| R3 | Linear Fit Clip: disable/remove (fork B); persisted token fails `unsupported_removed_sci05`; no migration to none/kappa/WSC. |
| C1 | Single zero-sum epsilon: denominator is **exactly `>0`** (no epsilon). |
| C2 | Median: unweighted median of `w_i>0` original surviving samples; magnitude ignored; no survivors → NaN. |
| G1 | Global-coadd labels route to the same canonical engine; remove percentile-winsorized; no rename (unify science). |

---

## 17. Honesty and limits

- No physical GPU run was performed at Gate A; all GPU cells remain STATIC / ROUTING_SEAM /
  NOT_RUN.
- No CPU↔GPU parity, no WSC numerical equivalence beyond the existing SCI-01/02/03/04
  witnesses, no Phase 4.5 / SDS / classic N≥3 runtime execution.
- This contract is the **accepted decision freeze**; it is not an implementation and does
  not alter runtime behavior. Independent Nono review-1 returned **ACCEPT**; Gate B may
  start under this contract.
