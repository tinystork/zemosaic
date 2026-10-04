# SCI-05 — Canonical Stacking Contract (DRAFT — requires Junior scientific acceptance before Gate B)

- **Mission:** `ZM-SCI-05-GATE-A-ARCHAEOLOGY-CONTRACT-20261005`
- **Phase:** Gate A draft only — facts separated from Junior decisions; nothing frozen as approved.
- **Repository:** `/home/tristan/.openclaw/workspace/projects/zemosaic`
- **Branch:** `science/zm-sci-05-canonical-stacking-coverage`
- **Base / HEAD:** `ea4b189f2017a03797261c5484702102ad5d3828` (`origin/beta`)
- **Donor (read-only):** `zsss-sci05-donor` @ `9b891de6e7ba71967d03db75d11c3fc279854b26`

> ## ⚠️ STATUS: DRAFT
> This document is a **candidate** scientific contract authored from current
> behavior + the coverage donor inventory. **Coco does not choose the final
> science.** Every disputed method is listed as an explicit **Junior decision
> point**. Nothing here is frozen until Junior issues scientific acceptance,
> which gates Gate B (port/implementation). No source change accompanies this
> draft.

---

## 1. Purpose and non-goals

Freeze a single canonical target pipeline and its per-stage semantics so Gate B
can implement the coverage-aware support domain without silently changing the
existing stacking science.

**Target pipeline order (frozen for discussion):**

```
ALIGNED INPUTS
  → NORMALIZATION
  → QUALITY WEIGHTS
  → SUPPORT / VALIDITY (channel-invariant 2-D positive support)
  → OUTLIER REJECTION
  → COMBINATION
  → CanonicalStackResult(science, support, diagnostics)
```

Non-goals for Gate A/B: no canonical-engine implementation here, no algorithm
replacement, no GUI change, no coverage port yet, no commit/push.

---

## 2. Coverage donor inventory (pinned SHA, read-only)

Donor worktree `zsss-sci05-donor` @ `9b891de6e7ba71967d03db75d11c3fc279854b26`.
ZeMosaic must **not** import or runtime-depend on the donor; these are target
semantics to reproduce, not code to link.

### 2.1 Portable symbols / algorithms (target semantics)

| Donor symbol | File | Target semantics |
| --- | --- | --- |
| `PositiveSupportAccumulator` / `accumulate_support_pair` | `seestar/core/coverage_support.py` | `SUP_W1 += s_i`, `SUP_W2 += s_i²`; positive-only; fail-before-mutation (negative/NaN/Inf/shape/overflow rejected); atomic pair; float64 default (float32 opt-in); channel-invariant 2-D `(H,W)` |
| `N_eff_support = SUP_W1²/SUP_W2` (else 0.0) | same | derived view; overflow-resistant `(W1/√W2)²` fallback; never mutates |
| `SUPPORT_STATE_VERSION=1`, `SUPPORT_DTYPES=(f32,f64)` | same | snapshot/restore schema |
| `make_footprint_taper(mask, feather_px=8.0, floor=0.0)` | `seestar/enhancement/weight_utils.py` | EDT distance-to-footprint-boundary → `1.0` interior, ramp to `floor` over `feather_px` px near boundary, `0.0` outside; translation/rotation invariant; chamfer fallback; padded so array boundary feathers symmetrically |
| `_footprint_distance_fallback` | same | chamfer 3-4 two-pass fallback (no scipy) |
| `make_radial_weight_map(h,w,feather,floor)` | same | **legacy** center-radial falloff (the thing COV-02 replaces) |
| `coverage_aware_render(sci, neff_support, n_ref=32, …)` | `seestar/enhancement/coverage_render.py` | render-only: `RENDER = B + (1−α)·D + α·D_denoised`, `α=clip(1−N_eff/n_ref,0,1)`; never brightness-gains low coverage; never inpaints |
| `start_processing` render default **False** | `seestar/queuep/queue_manager.py:5131` | `apply_coverage_render=False`, `support_taper_px=8.0`, `support_taper_floor=0.0`, `coverage_render_n_ref=32.0` |

### 2.2 Support formula (donor canonical)

```
s_i = valid_geometric_support
      * optional_quality_significance
      * optional_footprint_taper          (spatial, per original exposure)

SUP_W1 += s_i ;  SUP_W2 += s_i²
N_eff_support = SUP_W1²/SUP_W2  (SUP_W2>0) else 0.0
```

- `SUP_W1` is a raw exposure count **only** in the unit-weight case (`s_i ∈ {0,1}`).
- Support is independent of estimator WHT and of rejection survivors; no rejection
  mask is consumed.
- Support is channel-invariant and exactly 2-D `(H,W)` — one map per original
  exposure, no per-channel copies. No `science != 0` test.

### 2.3 Donor wiring (what Gate B would need to reproduce)

- `queue_manager.py`: `apply_batch_feathering=True` (COV-02), `apply_coverage_render=False`
  (COV-04), `support_taper_px=8.0`, `support_taper_floor=0.0`, `_ibn_reliable_fraction=0.02`
  (COV-03), `_reproject_support_tracking_enabled` (COV-01D).
- `gui_qt/settings_state.py`: `apply_batch_feathering=True`, `apply_coverage_render=True`
  (Qt dataclass default), `apply_feathering=False` (legacy inverse-WHT feather OFF).
- `settings_migration.py`: `SETTINGS_SCHEMA_VERSION_KEY="settings_schema_version"`.

### 2.4 Donor tests (portable witnesses, not ported here)

`test_coverage_support.py` (23), `test_coverage_support_classic.py`,
`test_coverage_taper.py` (translation invariance, boundary symmetry, bad-param
rejection, mean flat-field invariance), `test_coverage_render.py` (flat-field no-gain,
high-support no-op), `test_cov06b_render_ab.py` (render OFF/ON → equal scientific FITS,
differ only in preview).

### 2.5 ZeMosaic true-footprint source candidates (per route)

ZeMosaic has **no** dedicated positive-support accumulator. Candidate geometric
footprint sources per route (to be mapped at Gate B):

| Route | Candidate geometric source | Status |
| --- | --- | --- |
| Classic CPU | footprint masks from `align_images_in_group` (`_coerce_footprint_to_hw_bool`), alpha/coverage from `_stack_master_tile_cpu` | CANDIDATE (needs Gate-B mapping) |
| Classic GPU (Phase-3 auto) | `_prepare_frames_and_weights` footprint/alpha | CANDIDATE |
| SDS | `_sanitize_sds_megatile_payload` coverage/alpha (normalized by max) | CANDIDATE |
| Grid CPU/GPU | `process_tile` `footprint` → `np.clip(footprint,0,1)*weight_scalar` | CANDIDATE |
| Global coadd | `coverage_map` (weight_grid) / alpha map | CANDIDATE |
| Reproject (intertile) | reprojection footprint / validity mask | UNKNOWN (needs Gate-B archaeology) |

Only genuinely-geometric footprints (reprojection footprint, alpha, valid mask,
coverage) qualify; **NaN-only is not a geometric footprint** and must be labeled
UNKNOWN where the source is not yet established.

---

## 3. Frozen candidate contracts and Junior decision points

Each subsection: **(a) exact current formulas**, **(b) candidate canonical math**,
**(c) unresolved Junior decision**.

### 3.1 Normalization — `none` / `linear_fit` / `sky_mean`

**Current (verified):**
- `none` = passthrough everywhere.
- `linear_fit`:
  - Grid: per-channel `slope=cov(x,y)/var(x)`, `intercept=ȳ−slope·x̄` (`_fit_linear_scale`).
  - Classic: percentile points `a=Δref/Δsrc`, `b=ref_low−a·src_low`, gain clamp ±20, delta gates.
  - Phase 4.5 inline: OLS `slope=Σxy/Σx²` clipped `[0.25,4.0]`, intercept on **unclipped** slope, `max(5000,1%)` px gate.
  - `stack_core`: **median placeholder** (non-affine).
- `sky_mean`:
  - Classic: luminance percentile, additive `offset=ref_sky−src_sky`.
  - Phase 4.5 inline: percentile band `[30,70]`, per-channel median, additive delta, no px gate.
  - Grid: unsupported.

**Candidate canonical math (for Junior):**
- Reference selection: explicit reference index / median-of-stack / first-frame (pick one).
- Finite mask: common-finite-only overlap for every method.
- Mono/RGB: per-channel affine for `linear_fit`; luminance-additive for `sky_mean`.
- Robust estimator: robust percentile (classic) vs covariance/variance (Grid) vs OLS (4.5) — **must converge to one**.
- Min pixels / low-N: a single hard gate (e.g. `max(5000, 1%)`) or per-method.
- Failure policy: no-op (keep unnormalized) vs drop-frame vs fallback.

**Junior decisions (N1–N4):**
- **N1** — reference selection rule (which image is the reference?).
- **N2** — single `linear_fit` estimator (percentile vs covariance vs OLS) or
  accept documented per-route divergence.
- **N3** — low-N / min-pixel gate and failure policy (no-op vs fallback; **never**
  substitute a different science method).
- **N4** — dispose of the `stack_core` `linear_fit` median placeholder: keep
  documented, remove the option, or implement genuine affine (separate mission).

### 3.2 Weighting — `uniform` / `noise_variance` / `noise_fwhm`

**Current (verified):**
- `none`/`unit`/`unity`: Classic → no quality weights (unweighted); Grid →
  exposure-only `exposure_w=max(exposure,1e-3)` (exposure-folded). DIVERGENT for
  heterogeneous exposure.
- `noise_variance` (Classic): per-channel sigma-clipped (`sigma=3.0/3.0,maxiters=5`)
  variance `σ²`; global `min_overall_variance` (min finite positive variance, floored
  `1e-9`); weight `min_overall_variance/σ²`; invalid/≤0 variance → `1e-6`;
  unprocessed valid frame → `1.0`/`[1,1,1]`. Grid: `exposure_w / max(nanstd²,1e-8)`.
  DIVERGENT (formula, exposure fold, fallback constants).
- `noise_fwhm` (Classic) — reachable behaviors, NOT "every route → variance":
  1. Photutils **unavailable** → explicit `noise_variance` substitution (effective
     `noise_variance`).
  2. Photutils available + **all-unusable/star-free** → the estimator returns unit
     weights, the sanitizer sees no effect → `(None, "none", None)` = **no
     weighting**, not variance.
  3. Photutils available + **partial** → usable frames `min_overall_valid_fwhm/fwhm`,
     failed estimates constant `1e-6`, skipped/unprocessed `1.0`; effective label
     may remain `noise_fwhm`.
  4. All usable → genuine `min_fwhm/fwhm` weighting.
  Grid `noise_fwhm` → variance-only `exposure_w/variance` (a **separate** fallback).
  Under Tristan's absolute fallback policy, branches 1, 2, and 3 (and Grid's
  variance-only) are all prohibited `SILENT_SCIENCE_FALLBACK` /
  `SILENT_SCIENCE_DEGRADATION` — the exact numeric outcome differs but the
  classification does not.

**Candidate canonical math (for Junior):**
- Estimator: robust sigma-clipped stddev (Astropy) — pin sigma_lower/upper/maxiters.
- Floor: variance floor `1e-8`/`1e-9` (pick one) vs `∞` sentinel.
- FWHM: either a real FWHM estimator (photutils `equivalent_fwhm`) **or** an
  explicit unsupported-status (no silent fallback). Dependence policy on photutils.

**Junior decisions (W1–W3):**
- **W1** — `noise_fwhm`: implement a real estimator, or make it an honest
  `UNSUPPORTED` (GUI disabled / structured status) instead of the current silent
  fallback/degradation mix (variance-if-unavailable / no-weighting-if-star-free /
  partial constants) — all currently prohibited `SILENT_SCIENCE_FALLBACK` /
  `SILENT_SCIENCE_DEGRADATION`.
- **W2** — variance floor and exposure-fold policy (fold exposure into
  `noise_variance` like Grid, or keep pure `min_var/σ²` like classic).
- **W3** — weight application shape (scalar vs per-channel `(1,1,C)` vs per-pixel).

### 3.3 NaN / Inf / footprint / weight zero-negative / all-invalid / support

**Current (verified):**
- Grid CPU/legacy mask `weight<=0` → NaN **before** rejection; mean uses positive
  magnitude; median gates on `weight>0`; all-invalid → zero tile.
- `stack_core`: raw weights (zero/negative included); mean → NaN at `weight_sum=0`;
  median ignores weights entirely; no pre-mask.
- Classic: `_filter_statistically_dead_frames` drops empty/degenerate; `nan_to_num`
  before combine; `where(Σw>1e-9)` division.

**Candidate canonical (for Junior):**
- NaN/Inf data = invalid (masked) everywhere; support mask is channel-invariant 2-D.
- `weight == 0` = zero support (frame excluded from mean); `weight < 0` = invalid
  (rejected/zeroed) — never applied as a magnitude.
- Support **independent of estimator WHT**; no `science != 0` predicate.
- All-invalid output is **an open decision**, not a presumption of zero.

**Junior decisions (S1–S3):**
- **S1** — all-invalid output: decide neutrally among NaN/invalid, zero, or another
  **deliberately chosen documented sentinel**. **Tristan's explicit constraint:**
  the result must be NaN/invalid or a documented sentinel chosen on purpose; an
  arbitrary historical zero is **not** acceptable. (Recorded, not decided here.)
- **S2** — unify `weight<=0` handling (pre-mask vs raw-forward) at the core boundary.
- **S3** — freeze the 2-D channel-invariant support contract (no per-channel copies).

### 3.4 Rejection — kappa / WSC / linear-fit-clip

**Current (verified):** see matrix §3; the three meanings of `winsorized_sigma_clip`
and the `linear_fit_clip` no-op placeholder.

**Candidate canonical (for Junior):**
- `kappa_sigma`: Grid/`stack_core` uses the astropy iterative
  `sigma_clipped_stats(..., maxiters=5)` helper; Classic CPU wrapper may use the
  optional external `cpu_stack_kappa`, else the internal `_cpu_stack_kappa_fallback`
  (raw `nanmedian`/`nanstd` single-pass thresholds); wrapper GPU `gpu_stack_kappa`
  is median/raw-std style. **These are divergent estimators/implementations** — the
  canonical contract must pick one; current code has no canonical parity here.
- `winsorized_sigma_clip`: **one** implementation (PixInsight WSC is the default);
  `winsor_limits` honored; low-N policy.
- `linear_fit_clip`: **fork A** = define a real linear-fit clip; **fork B** = disable/
  remove the GUI option (current helper is a no-op placeholder).

**Junior decisions (R1–R3):**
- **R1** — WSC unique target: freeze PixInsight WSC as the single winsorized
  implementation (and migrate `stack_core` simplified + global-coadd percentile clip).
- **R2** — kappa/WSC exact parameters (sigma defaults, winsor limits defaults,
  max iters) and low-N (<3) policy.
- **R3** — Linear Fit Clip: real definition (A) vs GUI disable/remove (B).

### 3.5 Combine — mean / median

**Current (verified):** mean weighted everywhere (divergent invalid semantics);
median unweighted everywhere (divergent `weight<=0` exclusion).

**Candidate canonical (for Junior):**
- mean: `Σ(w·d)/Σ(w)` with `Σw≤ε → 0` (pick ε: `1e-6` Grid vs `1e-9` classic vs `>0` core).
- median: explicit **unweighted-valid-sample** contract (weights only gate validity,
  never scale).

**Junior decisions (C1–C2):**
- **C1** — a single zero-sum epsilon.
- **C2** — median weight contract: unweighted-valid-sample (weights = validity mask
  only) unless Junior decides otherwise.

### 3.6 Global coadd

**Current (verified):** `mean`/`median`/`kappa_sigma`/`winsorized` (nanpercentile clip).

**Candidate canonical (for Junior):**
- Same-label consistency rule: a given label must mean the **same executed symbol**
  at master-tile and global-coadd level, or be renamed.

**Junior decision (G1):**
- **G1** — rename global-coadd `winsorized` (or unify its semantics with the
  master-tile `winsorized_sigma_clip`).

### 3.7 CPU/GPU same-algorithm requirement

- Every route's CPU and GPU path **must execute the same algorithm** (same estimator,
  same params, same support). Where they currently differ (WSC vs simplified), the
  canonical contract requires convergence.
- Honest provenance: report `PHYSICAL_GPU_PASS` only when GPU arithmetic actually ran;
  otherwise `PHYSICAL_GPU_NOT_RUN`. No CPU↔GPU parity claim without a physical run.

### 3.8 Fallback policy

- **Backend fallback (allowed):** GPU→CPU fallback that executes the **same science**.
- **Scientific fallback / degradation (forbidden):** any silent substitution of a
  different method (`noise_fwhm` → `noise_variance`), any silent degradation to
  no-weighting (effective `none`), and any hidden constant substitution (`1e-6` /
  `1.0`) must each be surfaced as a structured status, not a silent WARN/DEBUG.

### 3.9 Logging / provenance

- Per stage, record **requested** and **effective** method (and reason), bounded
  (no array dumps). The donor's `COVERAGE_CONFIG` / `COVERAGE_RENDER_RESULT` /
  `RUN_EFFECTIVE` structured-event pattern is the target; ZeMosaic currently lacks
  this (see matrix §8 gap).

---

## 4. Fresh target defaults (required by Tristan — not inferred)

- **Support taper ON** (`support_taper_px=8.0`, `support_taper_floor=0.0`).
- **Final reconstruction OFF** (`apply_coverage_render=False`).

**Donor discrepancy to report (do not hide):** the donor's Qt settings dataclass
`settings_state.py:269` defaults `apply_coverage_render=True`, while the engine
instance `queue_manager.py:5131` defaults `apply_coverage_render=False`. These
conflict; **Tristan's fresh target is OFF**, and ZeMosaic must follow Tristan, not
silently infer from the incidental Qt dataclass default.

---

## 5. Junior decision register (consolidated)

| ID | Decision |
| --- | --- |
| N1 | Normalization reference selection rule |
| N2 | Single `linear_fit` estimator vs documented divergence |
| N3 | Low-N / min-pixel gate + failure policy (no silent substitution) |
| N4 | Dispose of `stack_core` `linear_fit` median placeholder |
| W1 | `noise_fwhm`: real estimator vs honest UNSUPPORTED (no silent fallback) |
| W2 | Variance floor + exposure-fold policy |
| W3 | Weight application shape (scalar/per-channel/per-pixel) |
| S1 | All-invalid output: NaN/invalid vs another **deliberately documented sentinel**; arbitrary historical zero **forbidden** (Tristan constraint; §3.3) |
| S2 | `weight<=0` pre-mask vs raw-forward at core boundary |
| S3 | 2-D channel-invariant support contract |
| R1 | WSC unique target + migrate `stack_core`/global-coadd |
| R2 | kappa/WSC params + low-N policy |
| R3 | Linear Fit Clip: real (A) vs disable/remove (B) |
| C1 | Single zero-sum epsilon |
| C2 | Median unweighted-valid-sample contract |
| G1 | Global-coadd label consistency / rename |

---

## 6. Honesty and limits

- No physical GPU run was performed at Gate A; all GPU cells are STATIC /
  ROUTING_SEAM / NOT_RUN.
- No CPU↔GPU parity, no WSC numerical equivalence beyond the existing SCI-01/02/03
  witnesses, no Phase 4.5 / SDS / classic N≥3 runtime execution.
- This contract **freezes no disputed method as approved**. Gate B may not start
  until Junior accepts the contract and resolves (or defers) the decision register.
