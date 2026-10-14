# R39 — Post-offset spatial residual: pre-registration (R39-A design gate)

**Mission:** ZM-ZEGRID-R39-A-SPATIAL-PREREG · **Phase:** rework-2 (design gate)
**Baseline:** `beta` @ `a7d3d377e5ffd6ee71fb17ee29e6d74af3089264`
**Status:** PRE-REGISTRATION / DOCUMENTATION ONLY. No R39-A instrument, no R39-B
prototype, no R39-C qualification and no R39-D integration is implemented or
authorized by this document.
**Decision authority:** Junior (architect/reviewer) after independent Nono review;
Tristan is the product owner and final human authority.
**Relation to prior gates:** R38 is CLOSED/STOPPED at Gate A (see
`R38_A_CAUSAL_PROBE.md` §7 and `R38_LAYOUT_INDEPENDENT_CALIBRATION_PLAN.md` §8). This
document does **not** reopen, rename or retroactively authorize R38-B/C/D/E. R39 is a
new, separately pre-registered sequence.

This document must be committed/reviewed **before** any real R39 outcome metric is
computed on the R37E Windows816 capture or on any other corpus.

**Rework-1 (after Nono review-0: 0 HIGH, 5 MEDIUM, 3 LOW).** Resolves M1 (noise/
shape semantics and the mis-inherited 1.25 σ statistic + `UNEXPLAINED_STRUCTURE`),
M3 (unavailable per-sample nuisance covariates → `NOT_AVAILABLE`/abstention), M4
(exact unambiguous A→B panel rule), M5 (B0/B1 authorization tracks), M6 (frozen
normalized→integer/block inverse), L1 (non-centred p90), L2 (relative C4 reduction),
L3/CV (frozen train/test mechanics). No fact changes; documentation-only.

**Rework-2 (after Nono review-1: 0 HIGH, 3 MEDIUM, 1 LOW).** Resolves R1
(`UNEXPLAINED_STRUCTURE` is now a hard blocker at relation and panel level), R2 (exact
permitted residual mix per panel state), R3 (decidable nuisance-availability rule +
B1a evidence subgate; detection separated from causal attribution), R4 (per-model/
per-axis sign admission; explicit axis family + sign tuple). No fact changes;
documentation-only.

---

## 0. Purpose and non-authorization

R38-A demonstrated a near-constant **per-Cell additive level** mechanism applied
*before* the R34c offset graph, on 5/6 frozen DEV relations in both orientations
(ratio span 0.994589–1.002271). That level term is already estimated and removed by
the accepted R34c offsets, so it does **not** explain the **unresolved post-offset
spatial residual** — the broader Cell-correlated disagreement that survives the
offset solve and the feather blend.

R39-A is a bounded, read-only, offline diagnostic that must be able to falsify or
localize the remaining post-offset spatial residual using the **exact
already-captured** R37E Windows816 evidence. It is a *localization / model-identification*
gate, not a correction, not a qualification, and not a production change.

Nothing in this document authorizes: any source/tool/test/config/GUI/manifest/FITS
change; any correction application; any new photometric solver; any production run;
any write under a dataset root.

Label discipline used throughout (as in `R37_ARCHAEOLOGY_AND_RISKS.md`):

- **STATIC** — source/contract/artifact inspection;
- **DYNAMIC** — a read-only measurement actually executed;
- **PREDICTED** — a derived quantity that is *not* an observation;
- **NOT RUN / NOT AVAILABLE** — not executed and not inferred as a pass.

---

## 1. Frozen inputs, identity and the DEV/diagnostic label

### 1.1 Immutable lineage of the R37E Windows816 capture (STATIC + DYNAMIC)

The single real input for R39-A is the accepted R37E pre-offset/pre-feather capture.
It is identified **logically** (no private absolute path is stored in this document or
in any R39 artifact):

| Fact | Frozen value |
|---|---|
| capture artifact basename | `zegrid_r37_pre_feather.json` |
| artifact schema / version | `zemosaic.zegrid.background_diagnostics` / `r37c-1` |
| artifact sha256 | `a260b3309c39fbfc5264cacc60ba3048c678f83cf8bf6bf9f39c94a134a84032` |
| artifact bytes | `23 252 804` |
| `capture_stage` | `pre_offset_pre_feather` |
| `correction_applied` | `false` |
| `production_wired` | `true` |
| capture id | `6c311af729425e69c030f87969fe6369dd9e9fc4ab9c133379662d70b65a8e00` |
| source-set digest / algo | `e3f4c16af30fca1a4c660ba5f76ea6f3d62868dd825bc09a9d83802bfccee3ae` / `sha256_sorted_unique_relative_logical_paths_v1` |
| source frames | `816` |
| canvas (W×H) | `4117 × 4377` (id `e8f1877f16f8cca4`) |
| layout | `nx = 6`, `ny = 4`, `24` cells |
| channels | `3` |
| geometric relations / records | `38` (20 × `h`, 18 × `v`) / `114` = 38 × 3 |
| dropped relations | `0` |
| normalization / convention | `global_canvas_normalized_[-1,1]` / `pixel_centre_+0.5_global_x_from_col_y_from_row` |
| `measurement_px` / `feather_px` / `halo_px` / `block_px` | `16` / `8` / `8` / `4` |
| `holdout_capped` / qualification | `false` / `EXACT` (uncapped held-out) |
| anchor | `r0001c0002` |
| manifest `cell_feather.offsets` | `24` cells × `3` channels (scalar additive) |
| manifest `cell_feather.edges_used/built` | `38 / 38` |
| manifest `weighted_residual_rms` (R34c) | `[33.896973, 30.434222, 34.183499]` ADU |
| per-relation `n_holdout` | min `1792`, median `2857`, max `4448`, all kept exactly |

The manifest cross-links the artifact on the committed identity keys
(`capture_id`, `filename`, `sha256`, `bytes`) inside
`cell_feather.r37_diagnostic_capture`, plus `n_cells`, `n_relations`, `n_dropped`,
`holdout_cap`, `holdout_capped`, `holdout_exact`, `qualification`, `capture_stage`,
`effective_cell_gauge_mode`, `halo_px`, `measurement_px`, `feather_px`,
`n_source_frames` and `source_set_digest` — the same commitment rule enforced by the
R37D/R37E tooling (`background_diagnostics.manifest_links_artifact`).

### 1.2 Consumed DEV/diagnostic data (no independent qualification here)

- The R37/R38 Windows816-IRCUT evidence is **DEV/diagnostic**. Its held-out support was
  already consumed for the R37 M0/M1 decision (`MODEL_MISMATCH`; see §1.3). A new
  split of the *same* pixels is not independent qualification, and R37-A already
  records that a tuned model needs a new, independent validation corpus.
- R39-A may use this capture only to **localize and model-compare**, never to qualify
  a correction. Qualification is reserved to R39-C (§11).
- **LP corpus (reserved for R39-C).** A header-only preflight of the LP corpus
  (recorded, read scope = FITS primary headers and WCS footprints only, **no pixels**)
  found `341/341` valid-WCS frames, `OBJECT = NGC 7635 (C11)`, filter `LP`,
  `1080×1920`, `30.0 s`, five nights (`2025-07-08/09/15/16/26`), `ctypes`
  `RA---TAN` 245 + `RA---TAN-SIP` 96, and **one** connected footprint component of
  size 341 (geometric header-level connectivity only, which qualifies neither science
  nor seam residuals). This corpus is **reserved**: no LP pixels may be read and no LP
  seam metric may be computed in R39-A. It is the intended real validation set for
  R39-C **after** the model family and thresholds are frozen here.

### 1.3 Accepted prior facts that constrain the design (STATIC)

- R34c is an **additive-only scalar** offset per Cell/channel, anchored on the central
  cell, applied as `science_cell += offset_cell` (`cell_feather` docstring; production
  assembly). It does not model any spatial term.
- R37 M1 (`ROBUST_PLAN`, per-Cell additive plane) was **rejected**: M0 held-out RMS
  `46.6248` ADU vs M1 `18.9666` ADU (ratio `0.40679`), but absolute held-out ratio
  `7.918 σ` against the frozen `1.25 σ` gate, worst-channel condition `14 952.17`
  against the frozen `10 000` ceiling, `2/114` relations worsened by >5% at p90;
  decision `MODEL_MISMATCH`, effective model `OFFSET_ONLY`. Cycle residuals were
  complete `45/45` but large (median `2.66 σ`, p90 `10.63 σ`, max `68.66 σ`).
- The accepted 1.25 σ gate is applied in production to the **RMS around zero**
  (`background_model`: `abs_ok = r_plan <= cfg.heldout_abs_sigma_max * sigma_delta`;
  the metric section is documented as *"Held-out metrics (RMS around zero; symmetric
  p90)"*). It is **not** a gate on the median. §9 re-registers it against that exact
  statistic.
- R38-A concluded `PARTIAL_CAUSE` (constant per-Cell additive level *before* offset);
  `scientific_acceptance = false`, `r38b_authorized = false`; its determinism witness
  was a same-process JSON round-trip, explicitly **not** an independent real-data
  rerun.

**Consequence for R39-A.** Because the R34c offset is a *single scalar per
Cell/channel*, adding it changes only the **level** of the paired residual; it cannot
change the residual's **shape** in the measurement band. Therefore the spatial
(tangent/normal/curvature) structure of the post-offset paired residual equals the
spatial structure of the pre-offset paired residual, while the constant component is
partly absorbed by the solve. This is the analytical basis of §6 and of the
"constant vs spatial" separation in §7.

---

## 2. Falsifiable questions

R39-A must answer exactly one of the following for the frozen DEV panel, at
relation level and orientation level, with explicit abstention:

- **(a) Constant / graph compromise** — after exact R34c offsets the paired residual
  is predominantly a *level* term (leftover constant consistent with offset-estimation
  bias, robust/clipping asymmetry, or graph/cycle compromise), with no stable spatial
  trend.
- **(b) Stable linear spatial trend** — a repeatable linear trend in the
  **normal** and/or **tangent** coordinate survives offset removal, across held-out
  blocks and across ≥2 distinct geometric relations in both orientations.
- **(c) Curvature / nonlinear structure** — a single preregistered quadratic-normal
  term is identifiable and stable beyond the linear model.
- **(d) Support / reprojection / PSF / mask-linked** — the residual is better explained
  by support/reprojection/PSF/mask structure than by geometry. **In the compact R39-A
  artifact the per-sample PSF/mask/reprojection/support covariates are NOT_AVAILABLE**
  (§5.3), so (d) can only be reported as a *remaining localized hypothesis*, never
  confirmed; every relation therefore carries the universal, non-promoting
  `SUPPORT_NOT_AVAILABLE` caveat, while spatial **detection** (b)/(c) is still recorded
  and causal **attribution** is deferred to the B1a subgate (§10).
- **(e) Negligible / inconclusive** — the residual is consistent with noise at the
  frozen reference noise scale **in the RMS-around-zero sense** (§9.1), or the evidence
  is insufficient/ill-conditioned.

Mapping to the pre-registered conclusion states (§9): (a) → `CONSTANT_ONLY`
(noise-consistent after constant removal); (b) → `SPATIAL_STABLE`; (c) → `CURVATURE`;
(d) → the universal `SUPPORT_NOT_AVAILABLE` caveat /
`causal_attribution_state = NOT_AVAILABLE` (never confirmable here);
(e) → `NEGLIGIBLE` or `INCONCLUSIVE`. A structured residual that the bounded family
cannot reduce to the frozen noise scale is `UNEXPLAINED_STRUCTURE` (§9.1), which
confirms none of (a)–(e) and **STOPS with no R39-B**. Falsification is the default: a
question is answered affirmatively only when its frozen gate passes.

---

## 3. Coordinate reconstruction and offset sign (frozen, exact)

### 3.1 Normalized coordinates (no guessing from plots)

For global column `c` (0-based, from the left) and global row `r` (0-based, from the
top), with canvas `W = 4117`, `H = 4377`:

```
X = 2 * (c + 0.5) / W - 1        # pixel-centre convention, +0.5
Y = 2 * (r + 0.5) / H - 1
```

This is exactly `background_model.normalized_coords` and the artifact's declared
`coordinate_convention`. `holdout_xy[m] = (X, Y)` are these normalized coordinates.

### 3.2 Relation geometry derived from the frozen layout (never from images)

With `cell_w = W / nx`, `cell_h = H / ny` and canonical cell ids `rRRRRcCCCC`
(`background_model._parse_rc`), for a canonical relation `(u, v)`:

- **`h`** — `u` is the left cell, `v` the right: boundary global column
  `b_c = (col(u) + 1) * cell_w`; normalized boundary `X_b = 2*b_c/W - 1`.
  Normal coordinate `n = X - X_b` (u-side `n < 0`, v-side `n > 0`,
  in normalized `x` units); tangent coordinate `t = Y`.
- **`v`** — `u` is the upper cell, `v` the lower: boundary global row
  `b_r = (row(u) + 1) * cell_h`; normalized boundary `Y_b = 2*b_r/H - 1`.
  Normal coordinate `n = Y - Y_b`; tangent coordinate `t = X`.

The physical normal distance in pixels is `s = n * (W/2)` for `h` and `s = n * (H/2)`
for `v`. The measurement band is `|s| ≤ measurement_px/2 = 8 px`.

### 3.3 Post-offset paired residual (frozen sign/order)

```
d_post(m) = holdout_diff[m] + offset[u][channel] - offset[v][channel]
```

where `holdout_diff[m] = science_u(m) - science_v(m)` at the same global pixel,
pre-offset/pre-feather, and `offset[cell][channel]` is the manifest R34c scalar.
This follows directly from the production rule `science_cell += offset_cell`
(per channel). The expression must be applied **per channel** with `offset[u][channel]`
and `offset[v][channel]`, and **never** with a cross-channel or aggregated offset.

**Mandatory implementation evidence (R39-A).** The sign/order may not be assumed:

1. a unit test must exercise the accepted production application path (the
   `science + offsets` function in `cell_feather`) and confirm the identity
   `(science_u + off_u) - (science_v + off_v) == holdout_diff + off_u - off_v`
   on a synthetic fixture with a known injected step;
2. a swapped-endpoint fixture must show that reversing `(u, v)` flips the sign of
   `d_post` exactly;
3. a synthetic artifact with analytically known pre/post residuals must reproduce the
   analytically expected `d_post` to `≤ 1e-10` in float64.

Any failure of these three fixtures is a **hard refuse** (no real-data result emitted).

### 3.4 Exact integer/block inverse (frozen; required before any metric)

The artifact stores only normalized `holdout_xy`, so every global integer pixel and
block id must be recovered by this frozen inverse — **never** by a raw floor of
normalized floats:

```
p_float = ((coord + 1.0) * dim / 2.0) - 0.5      # dim = W for X, H for Y
assert isfinite(p_float)
nearest = floor(p_float + 0.5)                   # round-half-up, frozen
assert abs(p_float - nearest) <= INV_TOL         # INV_TOL = 1e-9 (frozen)
p = int(nearest)
assert 0 <= p < dim
```

Recover `col` from `X` (`dim = W`) and `row` from `Y` (`dim = H`). Block ids are
`col // 4` and `row // 4` (`block_px = 4`, frozen). The **tangent block index** used by
§5.1/§8 is `col // 4` for `h` relations and `row // 4` for `v` relations.

Mandatory fixtures (fail-closed, before any metric):

1. a synthetic forward/inverse boundary fixture over `dim ∈ {W, H}` that round-trips
   every integer index and every index adjacent to a pixel-centre/half-pixel and to a
   block boundary (`0, 1, 3, 4, 5, …, dim-1`) within `INV_TOL`;
2. real-artifact structural validation that **every** held-out sample recovers integers
   satisfying `0 <= p < dim`, reproduces its stored `X`/`Y` within `INV_TOL`, and
   satisfies the captured block-parity predicate `bx % 2 == 1 AND by % 2 == 1`.

Any failure hard-refuses the whole run (no report, no metrics).

---

## 4. Aggregation discipline (anti-double-counting)

- The 114 records are **38 geometric relations × 3 channels**. The three channels of
  one relation are **correlated measurements of one geometric boundary**; they are
  **never** independent evidence. Aggregation is therefore **relation first, channel
  second**, exactly as in R38-A §4.7.
- A relation "qualifies" a hypothesis when the frozen rule in §9 is met on its channel
  aggregate; then orientation-level conclusions count **distinct geometric relations**
  (not records and not channels).
- Panel-level reported quantities are always accompanied by the executed/total relation
  fraction and the orientation coverage. A single edge, a single channel, or a
  small-band subset is never presented as causal.
- Interactions (frame-set / gauge / weight / rejection proxies) are reported
  separately, never summed into a single "explained fraction".

---

## 5. Metrics on held-out points only (deterministic, no leakage)

All primary metrics use **only** `holdout_xy` / `holdout_diff` (the artifact persists
no independent pre-offset train points; train-side information exists only as the
sufficient statistics `theta_global` / `information` / `rhs` and `local_offset_train`,
which must **not** be used as if they were held-out observations).

**Reference noise scale `σ_ref` (immutable).** `σ_ref` is the artifact relation/channel
`robust_sigma` (per channel), frozen and **never** re-derived. A re-estimated
`MAD_sigma(e) = MAD_SCALE * MAD(e)` (`MAD_SCALE = 1.4826`) is reported **separately**
and is never substituted for `σ_ref` in any gate (M1).

Per relation × channel, on the held-out set, with `e_pre = holdout_diff` and
`e_post = d_post`:

| Metric | Definition |
|---|---|
| robust median | `median(e)` (the constant component) |
| MAD / sigma | `MAD_SCALE * MAD(e)` — reported as a **re-estimate**, separate from `σ_ref` |
| RMS around zero | `RMS0(e) = sqrt(mean(e^2))` — **the statistic the inherited 1.25 σ gate was frozen for** |
| p90 absolute residual | `percentile(|e|, 90)` **non-centred**, exactly as production (`background_model`, "symmetric p90"); a centred variant, if reported, is labelled `p90_centred [NEW, contextual]` and is never a gate |
| robust normal slope | Huber-weighted slope of `e_post` on `n` (named `b_n`) |
| robust tangent slope | Huber-weighted slope of `e_post` on `t` (named `b_t`) |
| bounded normal curvature | single quadratic `n²` coefficient (named `q_n`; only if identifiable, §8) |
| distance-bin profiles | frozen bins `|s| ∈ [0,1), [1,2), [2,4), [4,8) px`, split by side (`n<0`, `n>0`); per-bin robust median and count — a **distance trend**, never labelled as support/PSF evidence (§5.3) |
| support counts | `n_train`, `n_holdout`, `effective_train`, `tangent_blocks`, `normal_blocks` |
| block uncertainty | tangent-block bootstrap (§5.1) |
| orientation summaries | per-orientation relation counts and pooled robust statistics |
| reduction vs pre-offset | `1 - metric_post / metric_pre` for sigma, RMS, p90 |

Block-selection reuse: the held-out points are the artifact's deterministic
block-parity subsample (`bx % 2 == 1 & by % 2 == 1`, `block_px = 4`) recovered by §3.4,
so the split is already deterministic and reproducible.

### 5.1 Block bootstrap (frozen)

Group held-out samples by **tangent block index** recovered via §3.4 (`col // 4` for `h`,
`row // 4` for `v`). Resample whole tangent blocks with replacement using a fixed seed
(`seed = 0`) and `B = 2000` replicates. The bootstrap **estimates the uncertainty of the
out-of-fold (OOF) coefficient and metric estimates of §8** (it re-fits on the resampled
training blocks and scores the resampled test blocks within the frozen fold structure);
it is an uncertainty estimate only and **never selects the model**. Block resampling
(not per-sample) is mandatory because samples within a block are spatially correlated.

### 5.2 Honest limitations of same-global-pixel pairs (must be stated)

- Same-global-pixel differences **cancel the common sky** (the same celestial signal
  seen by both Cells) — that is why they are admissible — but they do **not** cancel
  PSF differences, reprojection/interpolation error, support/coverage asymmetry, mask
  edges, or cosmic-ray/outlier residuals. These are the nuisance channels of question
  (d); they are **not decidable per-sample** in this compact artifact (§5.3) and remain
  a localized hypothesis.
- Because R34c estimated the offsets on the full overlap (which includes the R37
  held-out pixels), the **level** of `d_post` at held-out points is not fully
  independent of the offset estimate; §1.3 shows this affects the *constant* term, not
  the *shape* terms. R39-A therefore treats spatial shape as the primary discriminating
  signal and reports the constant separately. This is a stated limitation, not a claim
  of independence.
- No pixel read is required for the primary paired diagnostic; the final science FITS
  is used only for non-causal *contextual* control metrics (§9.5) and its inability to
  retain separate Cell values is respected.

### 5.3 Nuisance covariates `NOT_AVAILABLE` in the compact artifact (M3)

The frozen artifact carries, per relation, only **scalars** (`rejection_fraction`,
`level_mean`, `tangent_blocks`, `normal_blocks`, `effective_train`, `n_train`,
`n_holdout`, `band_used_px`, `overlap_px`, …) plus the per-sample arrays
`holdout_xy` / `holdout_diff` (`information`, `rhs`, `theta_global` are sufficient
statistics, not covariates). Consequences, frozen:

- **Per-sample** PSF, mask, reprojection and support covariates are
  `NOT_AVAILABLE`; **PSF data is absent entirely**. No per-sample (per-fold)
  correlation between `|e_post|` and these covariates may be defined, and the previous
  `|ρ| ≥ 0.30` / "majority of folds" proxy test is **removed as a conclusion gate (M3)**.
- Question (d) can therefore only be **localized as a remaining hypothesis**; it cannot
  be confirmed from this artifact.
- The available per-sample observable is the **band-edge distance `|s|`**, reported as a
  distance trend (§5 table) and **never mislabelled** as direct support/PSF evidence.
- Relation-level scalars (`rejection_fraction`, `level_mean`, occupancy/counts) may be
  reported as **contextual/exploratory** across relations only, with a separate minimum
  cohort of **≥8 distinct geometric relations** and **no causal/promotion role**.
- **Decidable rule (R3):** the compact capture **always** records, for **every** relation,
  `nuisance_covariates = NOT_AVAILABLE` and `SUPPORT_NOT_AVAILABLE = true` (the artifact
  carries no per-sample nuisance covariates and no PSF data). This is a **universal
  causal-attribution caveat**, not a subjective per-relation classifier — there is no
  "would require" judgement. It **does not** make a relation `INCONCLUSIVE` and **does
  not** erase spatial detection (§9.1): same-pixel shape evidence still classifies
  `SPATIAL_STABLE`/`CURVATURE`. It **does** block any causal correction claim and any
  direct path to C/D; causal attribution is deferred to the B1a subgate (§10).

---

## 6. Separating the constant (graph-compromise) residual from the spatial residual

R39-A must decompose `d_post` per relation × channel as:

```
d_post ≈ c0 + (constant / graph-compromise term)
        + b_t * t + b_n * n      (linear spatial: tangent + normal)
        + q_n * n^2              (one quadratic-normal term, if identifiable)
        + noise
```

(Frozen coefficient names: `c0`, `b_t`, `b_n`, `q_n`; `q_n` replaces the earlier
overloaded `c3`.)

- The **constant term** `c0` is attributed to eligible graph-compromise / offset-estimation
  leftovers: robust-vs-mean mismatch, clipping asymmetry, and graph/cycle residuals. Its
  magnitude is compared to the R34c context (`weighted_residual_rms` per channel) and to
  the R37 cycle residuals (median `2.66 σ`, p90 `10.63 σ`, max `68.66 σ`).
- The **spatial terms** are the cross-validated shape component. A spatial claim is only
  made if it is stable under the OOF cross-validation of §8 and survives the relation/
  orientation rules of §9.
- The R37 M1 (`ROBUST_PLAN`) plane solution must **not** be reused as a correction.
  Cycle/edge residual context may be *reported*; it is not subtracted from `d_post` and
  is not a fix.

---

## 7. Feathering — PREDICTED imprint only (never observed evidence)

The final blended value in a two-cell band obeys, for partition-of-unity weights
`w_u(x) + w_v(x) = 1` (production `feather_weight_field` / `assemble_canvas_feathered`):

```
blend(x) = (s_v + off_v) + w_u(x) * d_post(x)
```

so the *predicted* deviation attributable to the paired residual is
`w_u(x) * d_post(x)`, i.e. a **PREDICTED feather imprint** (PREDICTED label mandatory).

Rules:

- The predicted imprint is computed **only** if the exact deterministic weights are
  obtained from the accepted production function (read-only reuse); no ramp is
  re-derived from plots or re-implemented.
- The predicted imprint is reported as **PREDICTED**, never as observed post-feather
  evidence. Confirming it against pixels would require separate per-Cell rasters that
  the final FITS does not retain; the final FITS may be used only for
  contextual/non-causal control metrics (§9.5).
- Multi-Cell corners (more than two contributing patches) are out of scope for this
  formula and must be flagged as `NOT_APPLICABLE`, not approximated.
- The feather width is a fixed assembly parameter; R39 must not widen it to mask any
  residual score.

---

## 8. Bounded model family, cross-validation and anti-overfit rules (frozen before any outcome)

Frozen candidate family (nothing larger):

- **C0** constant only (`c0`);
- **C1** constant + linear tangent (`c0`, `b_t`);
- **C2** constant + linear normal (`c0`, `b_n`);
- **C3** constant + linear tangent + linear normal (`c0`, `b_t`, `b_n`);
- **C4** C3 + **one** quadratic-normal term (`q_n`).

Explicitly forbidden: free splines, high-order polynomials, per-pixel models, learned
kernels, and any model whose parameter count grows with the band size.

Robust fitting: Huber with transition `1.345 × MAD_sigma` (same convention as
`background_model.HUBER_DELTA_FACTOR`), deterministic, no hidden RNG.

**Cross-validation mechanics (frozen; L3).** Train/test discipline is enforced
**inside** the held-out set, with no leakage:

- `K = 4` deterministic tangent-block folds: `fold = (tangent_block_index) % K`
  (§3.4).
- For each fold `k`, fit the model on the **other `K-1` folds only**; the training
  subset's design must pass every identifiability gate below; then score the **untouched
  fold** (out-of-fold, OOF). No test-fold point is ever used for fitting.
- **Per-fold admission:** a model is admitted in fold `k` iff the fold's training design
  passes all gates below **and** every fitted coefficient is finite.
- **Model/axis admission (frozen per model and per axis, R4):**
  - `C1` admits on the **tangent** axis only: `b_t` must be finite, pass the gates, and
    have the **same nonzero sign** in ≥ 3 of 4 folds.
  - `C2` admits on the **normal** axis only: `b_n` likewise.
  - `C3` is scored as one model, but its two axes are evaluated **separately**: `b_t`
    counts only if its own coefficient passes uncertainty and has the same nonzero sign
    in ≥ 3 of 4 folds, and `b_n` likewise. A relation reports its **qualifying axis set**
    and, per qualifying axis, the sign (the **sign tuple**). **At least one axis** must
    qualify for C3 to count as a spatial model; a coefficient that does not qualify
    **cannot inherit** the other axis's sign or admission.
  - `C4` additionally requires the `q_n` term's own uncertainty/sign test **and** the
    relative improvement beyond C3 (below); the curvature family is reported
    **distinctly** from the linear axes.
- **Pooled OOF reduction** `red_sigma`: pool the OOF residuals of all folds; with the
  robust sigma (`MAD_SCALE*MAD`), `red_sigma = (sigma_oof(C0) - sigma_oof(model)) /
  sigma_oof(C0)`; require `red_sigma ≥ 0.20` **[NEW]** and a finite, strictly positive
  `sigma_oof(C0)` (otherwise refusal).
- The **full-data** fit (all held-out points) is **reporting only** and can never
  substitute for the OOF evidence **[NEW]**.

Identifiability / refusal gates (mandatory; evaluated on each fold's training design):

- full-rank design after centring; no rank deficiency;
- whitened condition number `cond(A) ≤ 1e4` **[FROZEN-INHERITED (INH), R37 contract §6.1]**;
- finite coefficient uncertainties; a term counts only if `|coef| ≥ 3 × bootstrap SE`
  **[NEW]**;
- leverage reported; predicted term uncertainty at the band edge must not exceed
  `0.5 σ_ref` **[INH, R37 contract §6.1]** or the term is refused;
- minimum support `effective_train ≥ 256`, `tangent_blocks ≥ 8`, `normal_blocks ≥ 3`
  **[INH, R37 contract §6.1]**;
- **C4 relative rule (L2):** `q_n` is admitted only if it passes the gates above **and**
  the OOF relative robust-sigma reduction **beyond C3** satisfies
  `(sigma_oof(C3) - sigma_oof(C4)) / sigma_oof(C3) ≥ 0.20` with a finite strictly
  positive `sigma_oof(C3)` (zero/nonfinite denominator ⇒ refusal) **[NEW]**; otherwise it
  is reported `NOT_IDENTIFIABLE`.

**No threshold may be changed after observing the real outcome.** Any post-hoc change
requires a new pre-registration and a new (unconsumed) corpus.

---

## 9. Conclusion states, thresholds and abstention

Thresholds are frozen here. Provenance is marked **[INH]** (inherited from an
already-accepted frozen requirement, against the exact statistic it was frozen for) or
**[NEW]** (pre-registered now, justified from synthetic calibration; frozen before
outcome).

### 9.1 Per-relation × channel states (held-out points; mutually exclusive, by precedence)

Inputs: `σ_ref` (artifact `robust_sigma`, immutable, §5); `c0` = robust median of
`e_post`; `model_oof` = best model admitted by §8 (C0 if no spatial model is admitted);
`rms0_resid_model` = `RMS0` of pooled OOF residuals of `model_oof`;
`red_sigma` = pooled OOF relative robust-sigma reduction of `model_oof` vs C0 (§8);
`sig_const` = `|c0| ≥ 3 × bootstrap SE(c0)` **[NEW]**.

Evaluated in this strict order; the **first** matching state wins:

1. **`INCONCLUSIVE`** — a mandatory §8 support/leverage gate fails (cannot classify), or
   no state below can be reached. (The universal `SUPPORT_NOT_AVAILABLE` caveat of §5.3
   does **not** by itself make a relation `INCONCLUSIVE`; it is recorded on every relation
   and blocks only causal attribution / direct C-D, R3.)
2. **`UNEXPLAINED_STRUCTURE`** (model mismatch) — `rms0_resid_model > 1.25 × σ_ref`
   **[INH: R37 absolute gate, applied to the RMS-around-zero statistic it was frozen
   for]**: the best admitted bounded model leaves a noise-inconsistent residual the
   family does not represent. **STOP — opens no R39-B.** A partial spatial fit is
   reported alongside as the non-promoting diagnostic flag `SPATIAL_STRUCTURE_PARTIAL`
   (diagnostic explanation only; never a correction qualification).
3. **`CURVATURE`** — C4 admitted per §8 (≥3/4 folds, consistent sign, `|q_n| ≥ 3 ×
   bootstrap SE`, relative reduction beyond C3 ≥ 0.20) **and** `rms0_resid_model ≤
   1.25 × σ_ref`.
4. **`SPATIAL_STABLE`** — C1–C3 admitted per §8 **and** `rms0_resid_model ≤ 1.25 × σ_ref`.
5. **`CONSTANT_ONLY`** — no spatial model admitted (`red_sigma < 0.20`), `sig_const`
   true, **and** the residual is noise-consistent after constant removal:
   `RMS0(e_post − c0) ≤ 1.25 × σ_ref` **[INH statistic]**.
6. **`NEGLIGIBLE`** — no spatial model admitted (`red_sigma < 0.20`), `sig_const` false,
   **and** `RMS0(e_post − c0) ≤ 1.25 × σ_ref`.

Rule (stated explicitly, M1): the promotion-relevant states `SPATIAL_STABLE` /
`CURVATURE` **require** the OOF best-model residual to satisfy the inherited absolute
`1.25 × σ_ref` RMS-around-zero gate. This deliberately separates *diagnostic
explanation* (spatial structure detected; reported via `SPATIAL_STRUCTURE_PARTIAL`)
from *correction qualification* (residual reduced to noise-consistency within the
bounded family).

Detection vs attribution (R3). `SPATIAL_STABLE`/`CURVATURE` are a **detection**
classification (`spatial_detection_state`) from same-pixel shape evidence only; they
make **no** causal claim about whether the cause is support/reprojection/PSF/mask. The
causal state is reported separately as `causal_attribution_state = NOT_AVAILABLE` for
every relation (§5.3). `SUPPORT_NOT_AVAILABLE` does not erase detection and does not by
itself block the B1 evidence-isolation track; it blocks causal correction claims and any
direct path to C/D.

### 9.2 Relation-level aggregate (channels never independent)

Rank the per-channel states by precedence:

`INCONCLUSIVE` > `UNEXPLAINED_STRUCTURE` > `CURVATURE` > `SPATIAL_STABLE` >
`CONSTANT_ONLY` > `NEGLIGIBLE`.

A relation maps to the **highest-precedence** state assigned to **≥ 2 of 3 channels**;
if no state reaches 2/3 the relation is `INCONCLUSIVE`.

**Hard override (R1).** If **any** channel of a relation is `UNEXPLAINED_STRUCTURE`, the
relation is `UNEXPLAINED_STRUCTURE` and carries the non-promoting flag
**`UNEXPLAINED_CHANNEL`** — a severe single-channel mismatch can **never** be masked by a
2/3 channel vote.
A relation's three correlated channels can never by themselves satisfy any "≥2 relations"
rule — only **distinct geometric relations** count. `SUPPORT_NOT_AVAILABLE` is recorded on
every relation (§5.3) but does **not** change its state.

### 9.3 Orientation-level state

An **orientation qualifies spatially** when **≥ 2 distinct geometric relations** in that
orientation are `SPATIAL_STABLE` or `CURVATURE` with the **same explicit axis family and
sign tuple** (e.g. both on the normal axis with `b_n > 0`, or both on the tangent axis
with `b_t < 0`) **[NEW, R4]**. Relations whose qualifying axis sets/signs differ do not
pool. Curvature relations qualify through their distinct curvature family (`q_n`).

### 9.4 Panel state and the single A→B gate (M4/M5/R1/R2 — one source of truth)

The panel state is assigned by the following **exclusive precedence** (first match wins),
using relation states only from §9.2 (`U` = `UNEXPLAINED_STRUCTURE`,`X` = `INCONCLUSIVE`,
`S` = `SPATIAL_STABLE` or `CURVATURE`, `K` = `CONSTANT_ONLY`, `N` = `NEGLIGIBLE`):

1. **`INCONCLUSIVE`** — the complete panel did not run (any planned relation `NOT_RUN`), or
   any relation is `X`. → opens **neither** track.
2. **`UNEXPLAINED_STRUCTURE`** — any relation is `U`/`UNEXPLAINED_CHANNEL` (hard STOP,
   evaluated **before** `SPATIAL_CONFIRMED`/`CONSTANT_ONLY`, R1). → opens **neither**
   (`r39b_authorized = false`, `r39b_track = NONE`).
3. **`SPATIAL_CONFIRMED`** — permitted mix **M_spatial** (R2): every relation is in
   `{S, K, N}`; **no** relation is `U`/`UNEXPLAINED_CHANNEL` or `X`; and
   **≥ 2 distinct `h` relations and ≥ 2 distinct `v` relations** qualify spatially (§9.3,
   same axis family + sign tuple per orientation; minimum **4** distinct relations).
   → opens **R39-B1 (B1a evidence-isolation gate only, §10)**.
4. **`CONSTANT_ONLY`** — permitted mix **M_constant** (R2): **no** relation is `S`, `U`
   or `X`; **≥ 2 distinct `h` relations and ≥ 2 distinct `v` relations** are `K`; all
   remaining relations are `N`. → opens **R39-B0**.
5. **`NEGLIGIBLE`** — permitted mix **M_negligible**: **every** relation is `N`. → opens
   **neither** track.
6. **`INCONCLUSIVE`** (mixed) — the relation mix matches none of M_spatial / M_constant /
   M_negligible. → opens **neither** track. This replaces the earlier vague "any mixed
   outcome" wording with the exact permitted mixes (R2).

`SUPPORT_NOT_AVAILABLE` (universal, §5.3), `SPATIAL_STRUCTURE_PARTIAL` and
`UNEXPLAINED_CHANNEL` are **non-promoting flags**, not panel states; none opens any track
and none can upgrade or downgrade the permitted-mix rules above.

### 9.5 Contextual-only controls (non-causal, no leakage)

The final `mosaic_grid_science.fits` (and coverage) may be read **only** for
non-causal contextual controls (e.g., boundary-vs-control profiles as already reported
in R37), clearly labelled `CONTEXTUAL_NON_CAUSAL`. It must **not** be used to define
Cell pairs, to reconstruct pair residuals, or to qualify a model, because it does not
retain separate Cell values post-feather.

---

## 10. R39-B — conditional, pure/offline tracks (not implemented here)

R39-B has exactly **two conditional, mutually exclusive tracks**, each opened by a single
frozen §9.4 panel state. There is **no** other path into R39-B, and **no** direct path
from R39-A to C or D. R39-B is pure/offline, minimal, imports nothing from the product
path. `r39b_track = B1` authorizes **only** the B1a evidence-isolation gate below — never
correction acceptance.

- **R39-B0 — OFFSET-ESTIMATOR track (opened only by panel `CONSTANT_ONLY`, §9.4 item 4).**
  Evaluate the smallest **R34c estimator repair** on compact evidence (robust-vs-mean,
  clipping asymmetry, weights, cycle consistency). **Never** a spatial or shared
  photometric solver; no new global/additive/gain/plane model.
- **R39-B1 — SPATIAL track (opened only by panel `SPATIAL_CONFIRMED`, §9.4 item 3).**
  R39-A's `SPATIAL_STABLE`/`CURVATURE` result is a **detection** only
  (`causal_attribution_state = NOT_AVAILABLE`). B1 must therefore **begin** with a
  separately specified **B1a nuisance-evidence subgate** before any correction prototype:
  - **B1a required evidence (no fabricated proxy, R3):** per-sample valid-mask/support
    and weights; exact reprojection/interpolation provenance **or** a controlled
    same-input reprojection comparison; and PSF/star diagnostics where measurable. This
    evidence is acquired on **consumed DEV data** via a bounded, read-only ROI /
    instrumented diagnostic — **not** LP341 and **not** a full TinyDebian product run.
  - **B1a outcomes:** if that evidence is unavailable, not identifiable, or attributes the
    trend to nuisance (support/reprojection/PSF/mask), return
    **`NEEDS_EVIDENCE` / `SUPPORT_LINKED`** and **STOP** — no B1 correction, no C, no D.
  - Only if the spatial mismatch **survives** B1a may bounded correction prototyping
    continue: at most the C1–C4 family of §8 per relation, with leverage/rank/condition/
    uncertainty/refusal gates and cycle consistency. No spline, no high-order term, no
    per-pixel freedom.
- **Opens neither:** panel `INCONCLUSIVE`/mixed, `UNEXPLAINED_STRUCTURE` (any
  `UNEXPLAINED_CHANNEL` block), `NEGLIGIBLE`. The recorded recommendation is `NONE`.
- **Explicit prohibition.** Repeating R33's already-invalidated global additive,
  affine-global, or gain+plane variants under new names is forbidden. A new observable
  must carry information genuinely distinct from R33/R37 fits, else **STOP**.
- **B0 / B1a failure blocks C and D.** No direct C/D path exists.

---

## 11. R39-C — qualification (criteria frozen before any real outcome)

Only after the opened R39-B track (B0 or B1) passes. Corpus:

- **Synthetic truth corpus** with known ground truth: flat identity; known offsets; known
  planes; common celestial gradient with no instrumental mismatch; diffuse/IFN nebulosity;
  stars/bright clusters in-band; independent instrumental gradients; nonlinear background;
  outliers/contradictory relations; low/high SNR; weak/fragmented/disconnected support;
  heterogeneous nights/gain/spectral response; layouts `1×N`, `N×1`, `2×2`, `6×4` and
  larger, plus **layout variants** (alternative `nx×ny` and shifted grid origin) over the
  *same physical synthetic truth*.
- **Reserved LP341 real corpus** (§1.2): separate acquisitions and filter but the **same
  astronomical target**. It is a real, non-synthetic witness; it is **not** automatically
  a universal-generalization proof and must be reported as target-specific.

Qualification rules:

- Freeze all model families, thresholds and acceptance criteria **before** opening the
  real LP outcome; the R39-A/B artifacts stay DEV-identified.
- Fidelity requirements (carried from `R37_BACKGROUND_MODEL_CONTRACT.md` §6.3):
  common celestial-gradient amplitude ratio in `[0.99, 1.01]`; diffuse/IFN protected-region
  robust bias `≤ 0.10 σ_pixel`; background-subtracted stellar aperture flux median relative
  bias `≤ 0.1%`, p95 `≤ 0.5%`; masks/support/coverage and sample weights preserved exactly.
- Residual target **[INH: R38 plan §5]**: RMS of differences `≤ 1.25 σ`, improvement
  `≥ 20%` vs baseline, no relation worsened by more than 5% at p90, both orientations
  improved.
- **Hold-out independence**: the calibration/qualification split must be genuinely
  independent; if the available real data are not independent, the status is
  `NEEDS_INDEPENDENT_DATA`, never a fabricated PASS.
- Layout-independence: per-pixel corrections/weights independent of the partition;
  numerical stack differences reported separately from scientific differences (no
  bit-exact CPU/GPU promise).
- A consulted TEST becomes consumed; a model revision requires a new TEST.
- **C failure blocks D.**

---

## 12. R39-D — integration (only after C passes)

- ZeGrid **experimental option OFF by default**; the previous path unchanged and tested
  with the option OFF.
- **No** `canonical_*` / Legacy / SDS / WCS/reprojection policy changes.
- Adapter (ZeGrid ↔ accepted fixed coefficients/weights) separated from the provider; a
  spatial correction, if any, lives in a **separate ZeGrid experimental provider**, never
  smuggled through the scalar `FixedNormalization`.
- Manifest records model/version, coefficients, sources, uncertainties, components,
  actual backend, refusals/fallback and metrics.
- Tests: GUI → backend **parameter propagation**; **Windows/spawn** execution;
  **GPU/CPU fallback**; bounded memory (no `N_frames × canvas` cube, sparse graphs);
  **no silent frame loss** (a valid frame is never dropped for a pixel because it lies
  across a cell boundary); FITS interop unchanged.
- Aesthetic/DBE finishing is out of science qualification and must not be used to mask a
  seam score.

---

## 13. Windows human gate

- No Zealfie update or run until the R39-D candidate has passed local tests and
  independent review on the capable host.
- Only then may Tristan supply the **exact beta commit** and a **fresh output path**;
  the full product run is performed on Windows, never on TinyDebian.
- No full product run, no release, no tag, no push is implied by any R39 gate.

---

## 14. STOP outcomes at every gate

- **R39-A** closes with `r39b_authorized = false`, `r39b_track = NONE` for panel states
  `INCONCLUSIVE` (incl. mixed), `UNEXPLAINED_STRUCTURE` (any `UNEXPLAINED_CHANNEL` block),
  and `NEGLIGIBLE`.
- **R39-B opens** only via §9.4: `CONSTANT_ONLY` → track **B0**; `SPATIAL_CONFIRMED` →
  track **B1** (**B1a evidence-isolation gate only**, never correction acceptance). Any
  other state opens neither.
- **R39-B0 / R39-B1a failure** (incl. B1a `NEEDS_EVIDENCE`/`SUPPORT_LINKED`) blocks C
  and D.
- **R39-C failure** blocks D.
- **R39-D** integration does **not** imply release; release remains an explicit Tristan
  decision.
- Every gate is fail-closed: insufficient support, ill-conditioning, an unexplained
  relation/channel, a B1a nuisance attribution, or privacy/identity failure yields an
  explicit refusal state, never a fabricated pass.

---

## 15. R39-A outputs (specification only — not implemented by this mission)

A standalone, offline, **read-only** tool under `tools/` (working name
`tools/r39a_spatial_residual.py`; final name in the R39-A mission) plus focused tests
(working name `tests/test_zegrid_r39a_spatial_residual.py`):

- **Inputs:** `--capture-dir` (artifact + manifest, read-only) and optional `--out`. No
  product imports at module scope (lazy at call time).
- **Primary paired diagnostic requires no image-pixel reads** — `d_post` is computed
  from the artifact (`holdout_xy`, `holdout_diff`) and the manifest offsets.
- **Identity + inverse, fail-closed:** refuse (no output, no metrics) unless the
  artifact/manifest match §1.1 on schema/version, `capture_stage`,
  `correction_applied is False`, `production_wired is True`, artifact sha256/bytes ==
  manifest == on-disk, `capture_id`, filename, source-set digest/algo/count, `n_frames`,
  layout, canvas, effective gauge mode, `measurement_px`/`feather_px`/`halo_px`,
  `holdout_capped is False`, `holdout_exact`, `qualification == "EXACT"`, `n_dropped == 0`
  — **and** the §3.4 integer/block inverse fixtures and real-artifact structural
  validation pass.
- **Strict JSON:** `json.dumps(..., allow_nan=False)` must always succeed; non-finite
  values are refused, never degraded to `0`. Deterministic ordering everywhere.
- **Atomic write:** temp file in the destination directory + `flush` + `fsync` +
  `os.replace`; temp removed on failure.
- **Privacy fail-closed:** no absolute/private paths in any durable artifact or report
  (whole value and embedded; POSIX/Windows-drive/UNC/`file://`); a surviving leak sets
  `status = FAILED`.
- **No writes under dataset roots:** refuse any `--out` inside the capture/source trees
  or a protected dataset root (same fail-closed policy as R38-A).
- **Memory bounded:** `O(relations × holdout)` compact arrays only; no
  `N_frames × canvas` accumulation; ideally zero image-pixel reads for the primary
  diagnostic.
- **Report block:** `status` (`PASS`/`PARTIAL`/`FAILED`), `qualification =
  DEV_NOT_QUALIFICATION`, `scientific_acceptance = false`, `panel_state` (the §9.4
  precedence value), `spatial_detection_state` (per relation/channel; §9.1),
  `causal_attribution_state = NOT_AVAILABLE` for every relation (R3), `r39b_authorized`
  (bool), `r39b_track ∈ {NONE, B0, B1}` (`B1` = B1a gate only; the single authorization
  source of truth, exactly as §10/§14), the non-promoting flags (`SUPPORT_NOT_AVAILABLE`,
  `SPATIAL_STRUCTURE_PARTIAL`, `UNEXPLAINED_CHANNEL`), the evidence scope,
  panel/executed fractions, orientation coverage, the per-relation/channel metric table,
  the per-channel/relation states and the permitted-mix classification, and the exact
  commands of the identity and §3.4 checks.

---

## 16. Non-goals and invariants

- No `src/`, `tools/`, `tests/`, config, dependency, GUI, default, manifest, FITS
  behavior or production-code change in this mission (documentation only).
- No real R39 metric computation and no product stack.
- No rewriting of R37/R38 history; no reopening of R38-B/C/D/E.
- No user-data modification; no commit/push/merge/tag/release/reset/clean/rebase.
- No private absolute paths in durable docs/artifacts.
- R39 is **not** evidence retroactively authorizing R38-B, and does not imply release.

---

## 17. Validation of this document (R39-A design gate)

- `git diff --check`, `git status --porcelain=v1`, `git diff --name-only` — only this
  new document is added.
- Scan this document for private absolute paths and for contradictory authorization
  claims (no claim that R38-B/C/D/E ran or is authorized; no claim that R39-A ran; and
  a single, consistent `r39b_authorized` / `r39b_track` authorization statement).
- Re-check that conclusion states are mutually exclusive by explicit precedence, that
  no p90/RMS statistic is mislabelled, and that no LP pixel is read.
- Re-verify every cited immutable fact of §1.1/§1.3/§1.2 against the existing
  artifact/manifest/schema/code **without** computing any new R39 outcome.

## 18. Acceptance gate for this document

One documentation-only pre-registration that is implementable, falsifiable, bounded,
scientifically conservative, keeps R38 closed, and gives clear A → B0/B1 → C → D stop/go
gates — with **no real outcome peek**.
