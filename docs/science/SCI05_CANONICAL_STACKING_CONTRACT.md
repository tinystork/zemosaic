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

**E2 implementation-resolution clarifications (engine entry point):**
* Assembled as one backend-neutral entry point
  `run_canonical_stack(CanonicalStackRequest) → CanonicalStackResult` in
  `src/zemosaic/core/canonical_engine.py`, running B1 (input/reference/normalization) →
  B2 (scalar quality weights) → E1 (estimator-weight map + support accumulation) → C1
  (rejection) → C2 (combine) in order.
* Support is accumulated from the **pre-rejection** `s_i = w_i = q*m*a` map
  (original-exposure ordered-add, only active frames contribute), so `SUP_W1/W2/N_eff` are
  **rejection-independent** (a `none` rejection and a `kappa_sigma`/WSC rejection over the
  same corpus give identical support).
* The result carries **bounded provenance** (requested/effective per stage, reference
  mode+index, taper kind/px/floor, equalize_rgb requested/applied, excluded frames, and a
  bounded `STACK_EFFECTIVE`-style `effective_event`) — scalars/strings/lists only, no array
  dumps.
* **Per-stage backend**: B1 (normalization), B2 (weighting), and E1 (support) execute on
  CPU; only C1 (rejection) and C2 (combine) honour `request.backend`. `provenance["backend"]`
  reports the requested token plus the per-stage effective backend, and `effective_event`
  mirrors it — never claiming GPU for stages that ran on CPU. `equalize_rgb=True` raises an
  explicit deferred-token validation error (the RGB equalizer is **not** wired at this gate);
  `backend="gpu"` propagates the explicit Gate-D validation (no silent CPU fallback); a
  reference quality failure surfaces as `CanonicalStackFailure`.
* **No claim**: Coverage render, the RGB equalizer, GUI/config/migration/locales, and
  production caller wiring remain **not** implemented (later gates E3+/F).

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

**F1 implementation-resolution clarification (N4):**
* `zemosaic_stack_core.stack_core` now raises an explicit `ValueError` containing
  `unsupported_removed_sci05` for `normalize_method='linear_fit'` (the median-substitution
  placeholder is gone); `none`/`median` remain supported. The sole production caller
  (`grid_mode._stack_weighted_patches_gpu`) already passes `normalize_method='none'` and is
  unaffected.

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
  `SourceCatalog.fwhm` from Photutils 3.0 sources satisfying finite `0.8 < FWHM < 20 px`,
  eccentricity ≤ 0.8, minimum 3 accepted sources. **Property correction (B2
  implementation-resolution):** the previously frozen name `equivalent_fwhm` is stale —
  it does not exist in installed Photutils 3.0.0; the intended measurement is
  `SourceCatalog.fwhm`, the **circularized FWHM of the 2-D Gaussian with the same
  second-order central moments** (equal-second-moment / circularized-Gaussian FWHM).
  **Deterministic detection mechanics (frozen by Junior, B2):** on the normalized
  luminance plane with the inherited 2-D valid mask, require at least **256 valid
  luminance samples** for any noise metric (fewer → explicit
  `insufficient_quality_samples`); robust sigma via Astropy sigma-clipped std (3/3,
  maxiters=5); detection background = the sigma-clipped **median**; build float64
  `luminance - median` (invalid pixels zeroed and supplied as mask); exactly one Photutils
  `detect_sources` pass with `threshold = 3*sigma`, `n_pixels=5`, `connectivity=8` (current
  keyword `n_pixels`, never deprecated `npixels`); **no** deblend, **no** second pass,
  **no** lowered-threshold / DAO / moment / custom fallback, **no** unrelated-property
  probing. Accept a source only if its `fwhm` is finite and strictly `0.8 < fwhm < 20` px
  AND its `eccentricity` is finite and `≤ 0.8` (missing/non-finite values rejected);
  require ≥ 3 accepted sources (frame failure `fwhm_insufficient_sources`); frame FWHM =
  float64 median of accepted FWHMs (finite > 0 else `fwhm_measurement_failed`). No
  image-size gate, no cap/top-N, no brightness/exposure fold, no fallback constants.
  **Availability preflight (B2):** a missing **or** importable-but-API-incompatible
  Photutils (signature/property mismatch) fails method availability **before** any
  per-frame work — the persisted/programmatic request fails validation (no per-frame
  substitution). Current API capabilities checked: `detect_sources` named `n_pixels`,
  `connectivity`, `mask`; `SourceCatalog` named `mask`, `progress_bar`;
  `SourceCatalog.fwhm` property; `SegmentationImage.n_labels` (or legacy `nlabels`).
  Missing Photutils disables the GUI option; persisted/programmatic request fails
  validation **before run**. Per-frame insufficient sources excludes the frame explicitly.
  **No** variance/no-weight/unit/1e-6 fallback.
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
* **Gate D scope note**: the backend-neutrality work covers **rejection (§8) and combine
  (§9)** only. The support/taper accumulator, `make_footprint_taper`, and positive-support
  maps (Gate E) and the normalization/weighting stages remain **CPU-only** in Gate D; their
  GPU parity (if any) is a **later bounded step**.
* **E1 implementation-resolution clarifications (support + taper + builder):**
  * Implemented as a **new** pure CPU module `src/zemosaic/core/canonical_support.py`
    (donor source-port, **no** runtime ZSSS import): `make_footprint_taper` (scipy EDT
    primary + donor chamfer fallback, `feather_px=8.0`/`floor=0.0`, footprint-following —
    never radial), `PositiveSupportAccumulator` + `accumulate_support_pair` (atomic
    `SUP_W1 += s_i` / `SUP_W2 += s_i**2`, float64 default, fail-before-mutation), and
    `build_canonical_estimator_weights` (the explicit `(N,H,W)` float64 `w_i=q_i*m_i*a_i`
    map directly consumable by the C2 `combine_canonical_samples`).
  * The builder's `taper` argument is `None` (`a_i=1`), `"footprint"` (generate per-frame
    from `valid_mask` via `make_footprint_taper`), an explicit `(N,H,W)` array, or a
    length-N sequence of `(H,W)` tapers — never a radial map.
  * **No claim**: Coverage render, the final request/result engine, and production caller
    wiring remain **not** implemented at this sub-lot (Gate E2+).
* **F2 implementation-resolution clarification (Classic CPU route convergence + support source):**
  * The Classic CPU stacking route (`stack_aligned_images`) now executes the canonical engine
    (`run_canonical_stack`, `backend="cpu"`) when the geometric footprints are threaded; the
    worker threads them unconditionally (`align_images_in_group(..., propagate_mask=True,
    return_footprints=True)` → `_stack_master_tile_auto` → `_stack_master_tile_cpu` →
    `stack_aligned_images(geometric_support=<footprints>)`).
  * The Classic geometric support `m_i` is the **transform-derived alignment footprint** —
    astroalign `propagate_mask` on the astroalign path, or the axis-aligned overlap rectangle
    from `_overlap_slices_from_shift` on the FFT-only/fallback path (reference identity frame
    = all-True). It is **never** inferred from brightness/NaN; frames with no honest footprint
    are explicitly excluded.
  * `coverage_support_taper` (default ON) selects the footprint taper for the estimator-weight
    map (`taper="footprint"`/`"none"`); legacy radial weighting stays inert; the Coverage render
    stays preview-only (never mutates science). No silent science fallback (`noise_fwhm` +
    missing Photutils fails explicitly; `linear_fit_clip` fails `unsupported_removed_sci05`).

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

**F1 implementation-resolution clarification (R3):**
* `linear_fit_clip` removed from the Qt/Tk rejection choices; the two worker rejection sites
  now raise explicitly with `unsupported_removed_sci05` (never silently migrate to
  `none`/`kappa`/WSC); the GPU error token is aligned; `stack_linear_fit_clip`/
  `_reject_outliers_linear_fit_clip` remain only as clearly-legacy, unreachable code with no
  supported caller.
* **R1 (central guard):** `_validate_rejection_token` (in `zemosaic_align_stack.py`) normalizes
  (strip/lower) + validates the rejection token at `stack_aligned_images` (the legacy entry
  point); `linear_fit_clip` (any case/whitespace) raises `unsupported_removed_sci05`, unknown/
  alias tokens raise an explicit validation error, and `none`/`kappa_sigma`/
  `winsorized_sigma_clip` pass through unchanged (bit-identical).

**C1 implementation-resolution clarifications (rejection stage):**
* kappa `std` is the ordinary population std (`ddof=0`) **around the sample mean**, while the
  interval bounds remain centered on the **median** (asymmetric `sigma_low`/`sigma_high`
  applied to the ORIGINAL samples). WSC `std` is the population std (`ddof=0`) of the
  winsorized current survivors.
* A cell whose **current** survivor count is `< 3` is frozen (explicit low-N no-rejection
  success from the current state) for that and all later iterations; it never falls back and
  rejected samples never re-enter (monotonic intersection only).
* Interval bounds are inclusive on both sides; `stable` means the complete survivor mask is
  exactly unchanged; rejection stops when stable or after `max_iters`.
* Parameter validation (before any work): `sigma_low/high` finite real non-bool `> 0`;
  `max_iters` integer `1..5`; winsor limits finite real non-bool each `0 <= limit < 0.5` with
  `low + high < 1`; the removed token `linear_fit_clip` fails with
  `unsupported_removed_sci05`; aliases/unknown rejected.
* Diagnostics (deterministic scalar definitions): `cell` = one `(H,W,C)` position across
  frames; `initial_sample_count`/`surviving_sample_count`/`rejected_sample_count` are integer
  sums over `(N,H,W,C)`; `rejected_fraction = rejected/initial` (or `0` when `initial == 0`);
  `low_n_cell_count` = number of distinct cells whose INITIAL active-valid count `< 3`
  (includes 0/1/2, counted once); `degenerate_cell_count` = number of distinct cells that
  encounter finite `std <= 0` in at least one executed iteration (counted once; `0` for
  `none`); `iterations_used = 0` for `none`, otherwise the number of loop passes executed.

**Gate D implementation-resolution clarification (rejection — backend neutrality):**
* One **shared `xp`-generic algorithm** runs on NumPy (CPU, **default**) or CuPy (GPU,
  **explicit opt-in** via a `backend` keyword). There is **no** silent CPU↔GPU fallback,
  no `os.environ`/`wsc_impl`/config selection inside the canonical layer, and no different
  science per backend.
* `backend` validation (before work): a non-string/unknown token, or `"gpu"` with
  CuPy/GPU unavailable, raises a validation error (never a substituted CPU run).
* GPU results are **host-converted** (device→host) into the **unchanged** owned NumPy
  result contract (masks bool, diagnostics Python scalars); CPU results stay byte-identical
  to the accepted C1 behavior.
* Parity: masks and diagnostic integers exactly equal; statistics use identical semantics
  (population std `ddof=0`; WSC quantile `method="linear"` — CuPy lacks
  `cupy.nanquantile`, so a **bit-exact** NaN-aware equivalent using NumPy's
  two-sided `_lerp` form is used and documented in the code).

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

**C2 implementation-resolution clarifications (combine stage):**
* The combine primitive consumes an **explicit pre-rejection 2-D canonical
  estimator-weight map** `w_i = q_i * m_i * a_i` per exposure (`(N,H,W)` float64)
  produced by the **future Gate E** support/taper builder. **No** default or optional
  weight map and **no** invented `a_i = 1`: the frozen default footprint taper is ON, so
  combine requires the explicit map.
* Estimator-weight validation (before any mutation): exact `(N,H,W)`, real numeric (not
  bool/object/complex), finite in `[0,1]`; **zero** on prior-inactive frames, **zero**
  outside `normalization.valid_mask`, and `w_i <= weighting.weights[i]` (the scalar
  `q_i`); zero on a valid sample (absent) is allowed. Weights are **shared across
  channels** for a given exposure/pixel, while the survivor mask is channel-specific.
* Eligible per-channel sample = `survivor_mask` AND `w_i > 0` AND finite original;
  original normalized values only (never winsorized replacements); weight magnitude
  never changes median values.
* Output shape/dtype: `science` float32, `estimator_weight_sum` float64, `valid_mask`
  bool (exactly `estimator_weight_sum > 0`), `surviving_sample_count` int64; all
  restored to the original shape (HW mono strips canonical C=1; HWC1 stays `(H,W,1)`;
  RGB stays HWC). Mean `estimator_weight_sum = Σ w_i`; median = **count** of eligible
  originals (unit effective estimator weights).
* Post-cast invariant: no NaN/Inf is ever marked valid; a mathematically valid estimate
  that becomes nonfinite (float64 or float32 post-cast) invalidates its cell (`science`
  NaN, `estimator_weight_sum` 0) and increments `nonfinite_output_count`. With finite
  float32 inputs and weights in `[0,1]`, mean/median are convex combinations of finite
  values and stay finite — the branch is a defensive seam, no clip/saturate.
* **No claim**: the support/taper accumulator and the final request/result engine are
  **not** implemented/accepted at this gate.

**Gate D implementation-resolution clarification (combine — backend neutrality):**
* One **shared `xp`-generic algorithm** runs on NumPy (CPU, **default**) or CuPy (GPU,
  **explicit opt-in**); no silent fallback and no config/env selection.
* GPU results are **host-converted** into the **unchanged** owned NumPy result contract
  (`science` float32, `estimator_weight_sum` float64, `valid_mask` bool,
  `surviving_sample_count` int64).
* Parity: `valid_mask`/`surviving_sample_count`/diagnostic integers exactly equal;
  `science`/`estimator_weight_sum` equal within a documented `~1e-12` relative tolerance
  (GPU float64 tree reductions may sum in a different order — a ~1e-15 effect); the
  denominator-`>0` and count-vs-sum semantics are identical.

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

**E4 implementation-resolution clarification (equalize RGB):**
* Source-ported into `src/zemosaic/core/canonical_equalize.py` as
  `equalize_rgb_medians_canonical` (out-of-place, never mutates the caller's array) +
  `equalize_rgb_medians_copy` (returns `(new_float32_array, info)`), reproducing the existing
  robust implementation exactly (background percentile `5/85`, min samples `5000`, min
  coverage `0.01`, gain clip `[0.95, 1.05]`) with the stable decision strings. No heavy-module
  import (numpy + stdlib only).
* Wired into the engine as an explicit **out-of-place post-combine** transform: `science`
  reflects equalization when applied; `estimator_weight_sum`/`valid_mask`/
  `surviving_sample_count`/support maps/rejection diagnostics are never modified.
  `equalize_rgb=True` on mono/HWC1 raises a validation error (RGB-only); a non-applied
  decision (e.g. insufficient samples) is an explicit no-op (`applied=False`), never a fake
  "applied".
* **No claim**: GUI/config/migration and production caller wiring remain **not** implemented
  (later gates).

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

**E3 implementation-resolution clarification (preview-only render):**
* Source-ported into `src/zemosaic/core/canonical_render.py` as
  `coverage_aware_render(sci, n_eff_support, *, n_ref=32.0, sigma_denoise=2.0,
  sigma_low=32.0)` (donor-exact `B + D` detail blend), plus `render_preview` and a bounded
  `coverage_render_event` dict. scipy `gaussian_filter` primary; unavailable → `sci`
  unchanged.
* Strictly **preview-only** and pure: never mutates `sci`/support, never applied inside
  `run_canonical_stack`; `result.science`/`support_w1`/`support_w2`/`n_eff_support` stay
  bit-identical. No gain for low coverage (flat fields stay exactly flat), no generative
  inpainting, no low-frequency signal modification.
* **No claim**: settings/GUI/migration, the RGB equalizer, and production caller wiring
  remain **not** implemented (later gates).

**E5a implementation-resolution clarification (settings + migration + locales):**
* Two public config booleans added to `DEFAULT_CONFIG`: `coverage_support_taper=True`
  (support taper ON) and `coverage_aware_reconstruction=False` (reconstruction OFF —
  Tristan's fresh target). The internal taper params (`8.0` px / `0.0` floor) stay internal
  canonical defaults, **never** exposed as new config/GUI knobs.
* `migrate_coverage_settings(config)` (pure, idempotent) migrates legacy radial settings:
  both legacy `apply_radial_weight=True` and `=False` yield `coverage_support_taper=True` +
  `coverage_aware_reconstruction=False`; the legacy radial path is marked **inert** (forced
  `apply_radial_weight=False`); `radial_feather_fraction`/`min_radial_weight_floor`/
  `radial_shape_power` values are **never** mapped to taper px/floor; bounded notes, no array
  dumps. Integrated non-breakingly into `load_config`.
* Locale keys added to all seven locale files for the two controls
  (`stacking_coverage_support_taper_label/_note`, `stacking_coverage_reconstruction_label/_note`).
* **No claim**: GUI widgets and runtime/worker wiring remain **not** implemented (E5b/Gate F).

**E5b implementation-resolution clarification (Qt/Tk GUI controls + runtime inert):**
* Qt GUI (`zemosaic_gui_qt.py`) stacking group: the legacy radial widgets/fields
  (`apply_radial_weight`, `radial_feather_fraction`, `min_radial_weight_floor`) are removed
  from the UI and `_config_fields`; two checkboxes bound to config are added —
  `coverage_support_taper` (default `True`) and `coverage_aware_reconstruction` (default
  `False`) — using the locale label/note keys.
* Legacy Tk GUI (`zemosaic_gui.py`): the same two canonical booleans are exposed and bound to
  config; the legacy radial controls remain constructable but are forced inert
  (`apply_radial_weight_var=False`).
* Legacy radial weighting runtime-INERT: `zemosaic_align_stack_gpu.py::_compute_radial_weight_map`
  is a documented no-op returning `None` regardless of `apply_radial_weight` (bounded one-time
  deprecation log).
* **No claim**: caller convergence — the taper/render actually consumed by Classic/SDS/Grid/
  Phase4.5/coadd — remains **not** implemented (Gate F). This gate only disables legacy radial
  weighting; it does not enable the canonical taper in any caller.

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
  **F2 (Classic CPU route, implemented):** `stack_aligned_images` routes through
  `run_canonical_stack` (backend `cpu`) for the supported method set; geometric support
  `m_i` = transform-derived alignment footprint (astroalign `propagate_mask` or FFT overlap
  rectangle), propagation unconditional, no-footprint frames excluded; `coverage_support_taper`
  consumed; legacy radial inert; render preview-only. SDS / Grid / Phase 4.5 / global coadd
  remain later F lots.
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
| N4 | `stack_core` `linear_fit==median` placeholder removed from supported paths; old callers route canonical or error, never median substitution. **F1:** implemented (raises `unsupported_removed_sci05`). |
| W1 | `noise_fwhm`: real estimator (Photutils 3 `SourceCatalog.fwhm` — **circularized FWHM from equal second-order central moments**; the previously frozen `equivalent_fwhm` is stale/nonexistent in Photutils 3.0.0 and is corrected to `fwhm`); deterministic detection: ≥256 valid luminance samples, sigma-clipped std (3/3, 5 iters) + median background, one `detect_sources` pass `threshold=3σ, n_pixels=5, connectivity=8`, no deblend/second-pass/threshold/property fallback; accept finite `0.8<FWHM<20`, ecc≤0.8, ≥3 accepted sources (else `fwhm_insufficient_sources`), frame FWHM = float64 median (finite >0 else `fwhm_measurement_failed`); **missing OR signature/property-incompatible Photutils fails availability preflight before per-frame work** (disables GUI / persisted fails validation; capabilities: `detect_sources(n_pixels, connectivity, mask)`, `SourceCatalog(mask, progress_bar, .fwhm)`, `SegmentationImage.n_labels`); per-frame insufficient sources excluded; no fallback. |
| W2 | No exposure fold (`none` = scalar `1.0`); `noise_variance` = raw `1/σ²` then max=1 normalization; no floor/epsilon/constant/unity fallback. |
| W3 | Exactly one scalar weight per frame, shared across channels; geometric/taper are separate 2-D factors. |
| S1 | All-invalid output = NaN/invalid science + `estimator_weight_sum=0` + `support=0`; no historical zero sentinel. |
| S2 | `weight==0` = absent; `weight<0`/nonfinite = invalid input failing before mutation. |
| S3 | Channel-invariant 2-D support: `m_i` explicit 2-D; `s_i == w_i == q_i*m_i*a_i` per exposure (`q=1` for `none`); `SUP_W1/W2` float64 atomic pair, fail-before-mutation (negative/NaN/Inf/shape/overflow), original-exposure ordered-add, never derived from estimator WHT/rejection; `N_eff` pure derived view `W1²/W2` (overflow-resistant `(W1/√W2)²` fallback, neutral `0` if undefined); donor algorithms source-ported, no runtime ZSSS dependency. **E1:** source-ported into `src/zemosaic/core/canonical_support.py` — `make_footprint_taper` (EDT primary + chamfer fallback), `PositiveSupportAccumulator`/`accumulate_support_pair`, and `build_canonical_estimator_weights` (explicit `(N,H,W)` `w=q*m*a` map for C2). **E2:** assembled by `run_canonical_stack` (`canonical_engine.py`): support accumulated **pre-rejection** (rejection-independent), bounded provenance (per-stage backend: B1/B2/support CPU, C1/C2 requested); Coverage render / RGB equalizer / GUI / caller wiring **not** implemented. **E3:** `coverage_aware_render` source-ported into `canonical_render.py` (preview-only, never mutates science/support). **E4:** `equalize_rgb_medians_canonical` source-ported into `canonical_equalize.py` (out-of-place post-combine, RGB-only). **E5a:** `coverage_support_taper=True`/`coverage_aware_reconstruction=False` config defaults + no-value-mapping `migrate_coverage_settings` (legacy radial inert). **E5b:** Qt/Tk coverage controls + legacy radial runtime-inert (`_compute_radial_weight_map` no-op). |
| R1 | Unique WSC target: true WSC using both winsor+sigma limits (PixInsight-style defaults); unify `stack_core` simplified and global-coadd percentile clip onto it. |
| R2 | kappa: median center + population std `ddof=0` (ordinary std **around the mean**, bounds centered on median), sigma `low/high=3.0`, ≤5 iters, inclusive bounds, `stable` = exact mask unchanged; WSC winsor `0.05/0.05`, sigma `3.0`; current-survivor-count `< 3` freezes the cell for that and later iterations; N<3 → explicit no-rejection success. |
| R3 | Linear Fit Clip: disable/remove (fork B); persisted token fails `unsupported_removed_sci05`; no migration to none/kappa/WSC. **F1:** implemented (Qt/Tk choice removed; worker/GPU raise `unsupported_removed_sci05`). **R1:** centralized `_validate_rejection_token` guard at `stack_aligned_images` (normalize + validate). |
| C1 | Single zero-sum epsilon: denominator is **exactly `>0`** (no epsilon); combine consumes an explicit pre-rejection `w_i=q*m*a` estimator-weight map (no default `a=1`); mean `estimator_weight_sum = Σ w_i` over original survivors. |
| C2 | Median: unweighted median of `w_i>0` original surviving samples; magnitude ignored; no survivors → NaN; `estimator_weight_sum = count` of `w_i>0` originals (unit effective estimator weights, not `Σ q·m·a`). |
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
