# R40 — UNEXPLAINED_STRUCTURE observability: pre-registration (R40-A design gate)

**Mission:** ZM-ZEGRID-R40-A-OBSERVABILITY-PREREG · **Phase:** implementation (docs-only)
**Repository / branch:** `projects/zemosaic` · `beta`
**Baseline HEAD:** `597c6dbbaca6193e6f6e75b7dc9d982ab4f4de76` (1 local commit ahead of `origin/beta`,
not pushed)
**Status:** PRE-REGISTRATION / DOCUMENTATION ONLY. No R40 capture, instrument, prototype,
qualification or product change is implemented or authorized by this document.
**Decision authority:** Jarvis Junior (architect/reviewer) after independent review;
Tristan is the product owner and final human authority.

**Relation to prior gates (stated once, unambiguously).**

- R38 is **CLOSED / STOPPED** at Gate A. R38-B/C/D/E remain **NOT_RUN / NOT_AUTHORIZED**.
- R39-A remains **terminal** at panel `UNEXPLAINED_STRUCTURE` with
  `r39b_authorized = false`, `r39b_track = NONE`, `qualification = DEV_NOT_QUALIFICATION`,
  `scientific_acceptance = false`. This document **does not** reopen, rename, relax or
  relabel that hard STOP. R39-B0/B1, R39-C and R39-D remain **NOT_RUN** and are **not**
  retroactively opened by anything below.
- R40 is a **new, separately pre-registered sequence**. It investigates the *same*
  `UNEXPLAINED_STRUCTURE` outcome as an honest, unexplained fact of R39 — using
  **genuinely new observables** acquired at the in-memory pre-feather/feather boundary —
  with its own gates. R40 does **not** claim R39 was wrong, does not tune C0–C4 post hoc,
  and does not convert any R39 relation scalar into a causal claim.
- **LP341 remains reserved and unconsumed.** No LP pixel may be read in any R40-A/B/C/D
  step defined here.

This document must be committed/reviewed **before** any new R40 observable is captured
and **before** any new R40 outcome metric is computed.

Label discipline (as in `R37_ARCHAEOLOGY_AND_RISKS.md` / the R39 plan): **STATIC**
(source/contract/artifact inspection) · **DYNAMIC** (a read-only measurement actually
executed) · **PREDICTED** (a derived quantity that is not an observation) ·
**NOT RUN / NOT AVAILABLE** (not executed and not inferred as a pass).

---

## 0. Purpose and non-authorization

R39-A executed the complete frozen 114/114 relation-channel panel on the exact Windows816
DEV capture. Its best admitted bounded models (C0–C4) left **75/114 channels** and
**31/38 relations** above the inherited `RMS0 ≤ 1.25 × σ_ref` gate, so the panel is
`UNEXPLAINED_STRUCTURE` and the sequence is a hard STOP. The compact artifact that R39-A
consumed persists, per relation/channel, only *sufficient statistics and paired
held-out differences* — it carries **no per-sample nuisance covariates and no PSF data**.
The accepted R39-A record is therefore honest but *causally blind*: it can detect and
localize a residual, and it cannot attribute it.

R40-A exists to close exactly that gap **on paper first**: it freezes, before any new
outcome, (i) the minimal *new* observables that must be captured at the in-memory
pre-feather/feather boundary, (ii) the controlled tests that may separate mechanism
hypotheses without fitting any correction, (iii) a causal-attribution ladder whose top
rung is reachable only by an exact same-input **intervention**, (iv) the sampling/split/
hash rules, (v) the staged gates and exact STOP outcomes, and (vi) the safety contract for
a future default-OFF, zero-influence diagnostic capture.

Nothing in this document authorizes: any source/tool/test/config/GUI/manifest/FITS
change; any correction application; any new photometric or spatial solver; any product
run; any write under a dataset root; any Windows or Zealfie update/run; any commit,
push, tag, release or deploy.

**Rework-1 (after Junior review-0).** Corrects the causal-ladder defect (observational
conditioning must never reach intervention-level `CONTROLLED_EFFECT`), the record-schema/
dimension inconsistency (one coherent `relation × channel × global-pixel` unit), the
reprojection-oracle defect (exact existing production/oracle symbols only, no
higher-order kernel), the zero-influence/manifest contradiction (one additive
`r40_diagnostic_capture` block; pre-existing manifest fields canonical-identical **for
plan-independent product/metadata** — later scoped by rework-16, §8 clause 3), the
unsupported `≤1e-12` threshold (bit-identical recomputation or a synthetic-calibrated
tolerance frozen in a pre-real revision), and factual/wording fixes (uncapped R37E
illustration, read-only source roots vs writable fresh output, split R40-B stage, H3/H5
scoping). No fact about R39 changes; documentation-only.

**Rework-2 (after Nono review-0: 0 HIGH, 4 MEDIUM, 3 LOW).** Closes the four MEDIUM
defects and folds in the three LOW advisories: (R1) per-intervention identity — I1's
local-crop vs full-source arrays are **not** identical by construction, so I1 is split into
fixed-common-mask numeric parity and an intended support/coverage delta, and the blanket
"identical arrays/masks" rule is removed; (R2) M3 dtype — pre-offset `f32`, post-offset
`f64` (the production helper casts to float64 and returns float64), identity verified on the
f64 result with no f32 downcast, and M6 corrected to the canonical `int64`; (R3) H4 now has
a frozen **bounded end-to-end shadow propagation** contract (per-frame delta → frozen stack
→ same production R34c offsets → `delta_d_post`), not a raw per-frame delta claim; (R4) the
false "read by reference at the parent seam" claim is corrected (`_stack_cell` runs in
workers) and a frozen **sparse worker→parent IPC** design replaces full-plane forwarding.
LOW: R40-B1 gains disjoint synthetic calibration/validation sets; the shard contract is
specified (schema/version, dtype/endianness/order, per-shard hash, atomic multi-file
publication, cleanup/quarantine, fail-closed paths); M1–M11 completeness applies to the core
association capture, with the §5 H4 intervention evidence a separate conditional block. No
fact about R39 changes; documentation-only.

---

## 1. Problem statement and immutable lineage

### 1.1 The R39-A terminal fact this document must respect (STATIC + DYNAMIC)

| Fact | Frozen value |
|---|---|
| R39-A implementation HEAD | `597c6dbbaca6193e6f6e75b7dc9d982ab4f4de76` |
| R39-A durable artifact (rel. name) | `r39a_windows816.json` |
| R39-A artifact sha256 | `a4f29611f270532f8da0a67689f6284750f740cd7324e7189d5ef59d114e6423` |
| R39-A canonical metrics sha256 | `8a2c26e65d14a6bc44977a609a45bfccec917bc391197888b0bfbf94823a9636` |
| tool execution status | `PASS` (operational only — every planned compact relation executed) |
| execution completeness | `114/114`, bootstrap `B = 2000`, seed `0`, leak-free per fold |
| `qualification` | `DEV_NOT_QUALIFICATION` |
| `scientific_acceptance` | `false` |
| `panel_state` | `UNEXPLAINED_STRUCTURE` |
| `r39b_authorized` / `r39b_track` | `false` / `NONE` |
| `causal_attribution_state` | `NOT_AVAILABLE` (every relation) |
| `SUPPORT_NOT_AVAILABLE` | universal non-promoting caveat |

Descriptive counts (reported as facts of the R39-A outcome, **never** as targets):

- channels: `75/114` `UNEXPLAINED_STRUCTURE`, `39/114` `SPATIAL_STABLE`, no
  `CURVATURE`/`CONSTANT_ONLY`/`NEGLIGIBLE`/`INCONCLUSIVE`;
- relations (relation-first, hard single-channel override): `31/38`
  `UNEXPLAINED_STRUCTURE` (`16 h`, `15 v`), `7/38` `SPATIAL_STABLE` (`4 h`, `3 v`);
- among unexplained channels, best-model OOF RMS-to-`σ_ref` ratio median `2.180`,
  range `1.253–12.945`;
- among unexplained channels, `37/75` select `C0`; `38/75` select an **admitted**
  `C1`/`C3` and still remain above `RMS0 ≤ 1.25 × σ_ref`; `C2`/`C4` never win or admit.

**Mandatory honesty rule.** These numbers are *descriptive evidence to be explained*, not
inputs to be tuned against. No threshold, sampling rule, cohort minimum or admission
rule in this document may be derived from, or calibrated to, the values above.

### 1.2 Immutable input lineage (the DEV corpus)

The only real R39-A input, and the only real input contemplated for a future R40-C step,
is the accepted R37E pre-offset/pre-feather capture and its manifest cross-link:

| Fact | Frozen value |
|---|---|
| capture artifact basename | `zegrid_r37_pre_feather.json` |
| artifact schema / version | `zemosaic.zegrid.background_diagnostics` / `r37c-1` |
| artifact sha256 | `a260b3309c39fbfc5264cacc60ba3048c678f83cf8bf6bf9f39c94a134a84032` |
| artifact bytes | `23 252 804` |
| `capture_stage` | `pre_offset_pre_feather` |
| `correction_applied` / `production_wired` | `false` / `true` |
| source frames | `816` |
| canvas (W×H) | `4117 × 4377` (id `e8f1877f16f8cca4`) |
| layout | `nx = 6`, `ny = 4`, `24` cells |
| channels / geometric relations / records | `3` / `38` (`20 h`, `18 v`) / `114` |
| `measurement_px` / `feather_px` / `halo_px` / `block_px` | `16` / `8` / `8` / `4` |
| `holdout_capped` | `false` (**uncapped**, `EXACT` held-out; the schema *supports* a documented per-relation cap, but **this artifact is uncapped** — its held-out evidence must never be called "capped") |
| anchor | `r0001c0002` |
| per-relation `n_holdout` | min `1792`, median `2857`, max `4448` |

The Windows816-IRCUT evidence is **DEV / diagnostic** and its held-out support was
**already consumed** by the R37 M0/M1 decision and again by R37/R38/R39 diagnostics. A new
split of the *same* pixels is **not** independent qualification. R40-C can therefore only
**diagnose** on this corpus; independent real qualification would require a new,
independent corpus and a separate pre-registration (a reserved real corpus, LP341, remains
untouched).

### 1.3 The frozen evaluation lineage this document inherits

Reused verbatim from the accepted R39 plan (no re-derivation, no re-tuning):

- coordinate convention `global_canvas_normalized_[-1,1]` /
  `pixel_centre_+0.5_global_x_from_col_y_from_row`, `X = 2(c+0.5)/W − 1`,
  `Y = 2(r+0.5)/H − 1`;
- relation geometry from the frozen layout (never from images): `h` boundary column
  `b_c = (col(u)+1)·cell_w`, tangent coordinate `t = Y`, normal coordinate `n = X − X_b`;
  `v` transposed; physical normal distance `s = n·(W/2)` for `h`, `s = n·(H/2)` for `v`;
  measurement band `|s| ≤ 8 px`;
- the exact integer/block inverse (round-half-up, `INV_TOL = 1e-9`, **coordinate inversion
  only**) and the block-parity held-out predicate `bx % 2 == 1 AND by % 2 == 1`,
  `block_px = 4`, tangent block `row // 4` for `h` / `col // 4` for `v`;
- the post-offset paired residual sign rule
  `d_post = holdout_diff + off_u − off_v` (per channel, from the production
  `cell_feather.apply_additive_offsets` `science_cell += offset_cell`);
- the inherited statistics: `σ_ref = robust_sigma` (immutable), `RMS0(e) = sqrt(mean(e²))`,
  the `1.25 × σ_ref` absolute gate **[INH]**, `K = 4` tangent-block folds, leak-free OOF
  bootstrap `B = 2000` seed `0`, `red_sigma ≥ 0.20` **[NEW, frozen in R39]**, support gates
  `effective_train ≥ 256`, `tangent_blocks ≥ 8`, `normal_blocks ≥ 3` **[INH R37]**.

---

## 2. Evidence-availability matrix

Causality here means: *is there, in the currently persisted evidence, a quantity that can
distinguish two mechanism hypotheses at the same global pixel under a controlled
comparison?* "→ causal?" answers only that question.

| # | Observable / covariate | Currently exists? | Exact location (symbol) | → causal? |
|---|---|---|---|---|
| E1 | Same-global-pixel paired Cell science difference `holdout_diff` | YES (held-out subsample only) | capture `relations[*].holdout_diff` | no |
| E2 | Paired global-pixel coordinates `holdout_xy` (normalized) | YES | capture `relations[*].holdout_xy` | no |
| E3 | Relation scalars (`robust_sigma`, `level_mean`, `rejection_fraction`, `effective_train`, `tangent_blocks`, `normal_blocks`, `band_used_px`, `overlap_px`, `n_train`, `n_holdout`) | YES (relation level) | capture `relations[*]` | no |
| E4 | Sufficient statistics `theta_global` / `information` / `rhs`, `local_offset_train` | YES | capture `relations[*]` | no (train side, not per-sample) |
| E5 | Per-Cell R34c additive offsets (scalar, per cell × channel) | YES | manifest `cell_feather.offsets` | no |
| E6 | Per-Cell `valid_mask` (bool, patch-shaped, per channel) | **NO** (dropped at capture) | in-memory only: `MiniTile.valid_mask`, `CanonicalStackResult.valid_mask` | no (absent) |
| E7 | Per-Cell `estimator_weight_sum` (E1 pre-rejection weight sum, science-shaped) | **NO** | in-memory only: `MiniTile.estimator_weight_sum` | no (absent) |
| E8 | Per-Cell `surviving_sample_count` (per-channel stack depth) | **NO** | in-memory only: `MiniTile.surviving_sample_count` | no (absent) |
| E9 | Per-Cell `support_w1`, `support_w2`, `n_eff_support` (`(H,W)`, channel-invariant) | **NO** | in-memory only: `MiniTile.support_w1/2`, `n_eff_support` | no (absent) |
| E10 | Exact deterministic per-Cell feather ramp | RECONSTRUCTIBLE (no pixels needed) | `cell_feather.feather_weight_field` (from layout/bounds/core_slice/halo) | no (predicted only) |
| E11 | Per-Cell aggregate reprojection-path stats | YES (cell aggregate) | worker `"reproject"` ← `execution.get_reproject_path_stats()` | no |
| E12 | Per-(frame,cell) reprojection path identity (crop vs full-source sufficiency; fast vs `reproject_interp` parity) | **measurement realizable**, not persisted | `execution.reproject_cropped` / `reproject_cropped_fast` / `reproject_full_source` / `reproject_interp` | **yes, via the §5 same-input interventions** |
| E13 | Per-sample PSF / star-contamination diagnostics | **NO** (no PSF data at all) | absent | no (absent) |
| E14 | Per-Cell post-feather contribution to the final pixel | **NO** (unrecoverable) | `assemble_canvas_feathered` divides out `wsum` per pixel | no (absent) |
| E15 | Final science/coverage FITS | present in the exact run dir | `mosaic_grid_science.fits` etc. | no (feather-collapsed) |

### 2.1 Why the persisted artifacts are insufficient

- **Final science/coverage FITS cannot identify causality.** The feathered canvas stores
  `Σ_c w_c · (s_c + off_c) / Σ_c w_c` per pixel; the per-Cell numerators and the exact
  per-cell weights are **divided out and discarded**. Two different mechanisms (a support
  edge vs a weight edge vs a true scene structure) map to the same blended pixel, and the
  final FITS cannot recover separate Cell values — so it cannot form a same-pixel paired
  contrast with covariate conditioning. Reading it can only ever be
  `CONTEXTUAL_NON_CAUSAL` (as frozen in R39 §9.5).
- **The R37/R39 compact artifacts are insufficient.** They persist relation-level scalars
  plus a held-out paired-difference subsample (E1/E2/E3/E4). They carry **no per-sample**
  `valid_mask`, weight sum, stack depth, support maps, or PSF (E6–E9, E13). A per-sample
  `|e|` vs covariate contrast — the only thing that could separate "the residual tracks
  support/weights" from "the residual is independent of support/weights" — is therefore
  **undefined** on the existing evidence. R39 §5.3 already froze this as the universal
  `SUPPORT_NOT_AVAILABLE` caveat; R40-A must not pretend it away.
- **The `__zegrid_cache__` is empty** at the exact run directory, and the aligned per-frame
  per-cell arrays were deleted after stacking (bounded-disk cleanup). So the missing
  observables cannot be *recovered*; they must be *captured* at the moment they exist.

### 2.2 What actually exists in memory at the capture point (STATIC archaeology)

At the pre-offset/pre-feather assembly boundary the accepted production path already holds
every mandatory covariate, in these exact symbols:

- `zemosaic.core.canonical_engine.CanonicalStackResult`
  (`science`, `estimator_weight_sum`, `valid_mask`, `surviving_sample_count`,
  `support_w1`, `support_w2`, `n_eff_support`, `rejection_mask`, `provenance`) — each with
  the documented shapes (science `(H,W,C)`; `estimator_weight_sum` science-shaped;
  `support_w1/2`, `n_eff_support` `(H,W)` float64; `surviving_sample_count` `(H,W,C)`
  int64);
- `zemosaic.core.zegrid.assembly.MiniTile` — the same plane set, per Cell, with
  `core_slice`, `reference_frame_id`, `excluded`, `frame_order`, `provenance`;
- `zemosaic.core.zegrid.science_adapter.MiniTileScienceConfig` /
  `MiniTileScienceResult` — the frozen stack tokens (normalization/weighting/rejection/
  combine/taper/taper_px/taper_floor/equalize_rgb/sigma_low/sigma_high/winsor/cell_gauge_mode)
  and the FrameId-resolved reference/exclusions;
- `zemosaic.core.zegrid.cell_feather.feather_weight_field` — the **exact, deterministic**
  per-Cell ramp (a pure function of `bounds`, `core_slice`, `halo_px`, `canvas_hw`; no
  pixels);
- `zemosaic.core.zegrid.execution` reprojection primitives — `reproject_cropped` (the
  production entry point: fast path when eligible, `reproject_interp` fallback otherwise),
  `reproject_cropped_fast` (bit-exact fast bilinear path), `reproject_full_source` (the
  existing full-source **oracle reference**, used only by the oracle test), `REPROJECT_ORDER
  = "bilinear"`, and `ReprojectPathStats`
  (+ `reset_/get_/merge_reproject_stats`).

**Cross-checked gap (must be stated, not glossed):** the feather-stage
`cell_feather.CellPatch` currently propagates only `science`, `valid_mask` and
`surviving` from the per-cell worker result (`zemosaic_zegrid_mode._assemble_per_cell_feathered`
re-builds `CellPatch` from `result_by_cid[...]["patch"]`, which itself exposes only
`science`/`valid_mask`/`surviving`). The remaining planes (E7–E9) exist **upstream** on the
`MiniTile`/`CanonicalStackResult` but are **not** currently forwarded to the feather stage.
A future R40-B capture therefore must **not** claim these planes are parent-visible or read
"by reference at the seam": `_stack_cell` runs in worker processes (`zparallel.pmap` →
`ProcessPoolExecutor`) and the parent receives only the pickled `patch` dict
(`science`/`valid_mask`/`surviving`). The E7–E9 covariates must be **extracted sparsely
inside the worker before IPC** (§3.1.1); forwarding the full f64 E7–E9 patch planes would
be `O(full patch area)` of IPC and parent retention and is forbidden.

---

## 3. Frozen NEW observable / capture contract

The capture is defined **only** at the in-memory pre-feather/feather boundary — the same
seam as the accepted R37-D capture (inside `_assemble_per_cell_feathered`, after the
complete `CellPatch` objects coexist and before `build_cell_offset_graph` /
`assemble_canvas_feathered`). No capture may occur after blending (E14 is unrecoverable).

**FITS / pixel scope (frozen, explicit).** R40-A and the **core association capture path**
consume **in-memory arrays only** and perform **no** extra FITS/image read; they **never**
read the final product FITS (`mosaic_grid_science.fits` / aesthetic / coverage). The single
future exception is the **separately gated H4 shadow intervention** (§5.1), which
necessarily reads only the **prepared source FITS** (`fits.open(..., memmap=True)`,
read-only) through the existing `execution.reproject_full_source` — and **never** the final
science/aesthetic/coverage FITS. That H4 path is `NOT_RUN`/`NOT_AUTHORIZED` now and requires
R40-B1 + R40-B2 + the human gate (§11). **No R40-A read of any FITS/pixel has occurred.**

**Record unit (frozen, one coherent unit).** The primary record is
**`(relation, channel, global-pixel)`** — i.e. `38 relations × 3 channels × n_holdout`
records ≈ `114 × n_holdout` pairs. Channel-dependent values are stored **per channel**;
channel-invariant values (`support_w1/2`, `n_eff_support`, ramps) are stored **repeated per
channel by default**, and may be factored into a `(relation, pixel)` sidecar **only if** the
capture proves, fail-closed, that the three channels of a relation share the identical
sample pixel set *and* identical channel-invariant plane values. Per-channel coordinates
**can differ** across channels (each channel has its own validity/rejection and hence its
own held-out pixels); the capture must **not** assume channel supports are identical
without that proof. Identity/calibration items (M9–M11) are **headers** (once per relation),
never repeated per sample.

**Sample set (frozen).** For each relation × channel, the **exact** sample set already used
by the accepted paired evidence — the deterministic block-parity held-out subsample
(`bx % 2 == 1 AND by % 2 == 1`, `block_px = 4`) recovered by the frozen §1.3 inverse. No new
split, no random draw, no per-pixel raster. This guarantees "same global pixels" and makes
the capture directly comparable to `holdout_diff`.

**Sample key (frozen).** A record is keyed by
`key = sha256("zemosaic.zegrid.r40.pair.v1\n" + relation_id + "\n" + channel + "\n" + gx + "\n" + gy)`,
where `(gx, gy)` are the recovered **global integer** pixel coordinates. Ordering is
`(orientation, u, v, channel, gy, gx)`, strictly deterministic.

### 3.1 Mandatory observables (M)

Each mandatory item is required for at least one controlled test of §4; a capture missing
any mandatory item is refused (no partial "best effort" capture is accepted as evidence).

| ID | Observable | Unit / shape | Why mandatory |
|---|---|---|---|
| M1 | `relation_id (u,v)`, `orientation`, `channel` | header / record id | relation-first discipline; the unit of evidence is the relation, not the channel |
| M2 | global integers `gx`, `gy` | record, int32 ×2 (per channel) | same-global-pixel pairing is the only non-tautological pairing |
| M3 | per-Cell science at the pixel, **pre-offset** and **post-frozen-R34c-offset**, both Cells | record, pre = **f32** (original), post = **f64** (×2 stages ×2 Cells, per channel) | post-offset is the frozen residual basis; pre-offset lets the identity `d_post = diff + off_u − off_v` be re-verified fail-closed, not assumed |
| M4 | per-Cell `valid_mask` at the pixel | record, bool ×2 Cells (per channel) | support/validity-discontinuity hypothesis (**H1**) |
| M5 | per-Cell `estimator_weight_sum` | record, f64 ×2 Cells (per channel) | estimator-weight/survivor-discontinuity hypothesis (**H2**) |
| M6 | per-Cell `surviving_sample_count` | record, **int64** ×2 Cells (per channel) — the canonical result dtype (narrowing is not assumed safe) | stack-depth discontinuity (**H2**) |
| M7 | per-Cell `support_w1`, `support_w2`, `n_eff_support` | record (channel-invariant planes; repeated per channel by default, or factored to a `(relation,pixel)` sidecar once channel-identity is proven) | effective-support discontinuities (**H1/H2**); the E1 support maps the R39 artifact lacks |
| M8 | exact feather ramp value for both Cells at the pixel (`w_u`, `w_v`) | sidecar / record, f64 ×2 | feather interaction (**H3**); from the accepted `feather_weight_field`, never re-derived |
| M9 | reference / frame identity: `reference_frame_id` per Cell, a digest of `frame_order`, the `excluded` list | **header** (per relation) | identity + cross-link; guarantees inputs are the accepted ones |
| M10 | frozen stack + calibration identity: `cell_gauge_mode`, normalization/weighting/rejection/combine/taper tokens, `sigma_low/high`, winsor limits, `measurement_px`/`feather_px`/`halo_px`/`block_px` | **header** (per relation / capture) | identity of coefficients/weights/exclusions; a mismatch is a hard refusal |
| M11 | provenance binding: capture-contract version, artifact sha256/bytes, `capture_id`, source-set digest/algo, repo HEAD | **header** (capture) | fail-closed cross-link to §1.1/§1.2 |

Inside the worker the covariate planes live on the in-memory `MiniTile`
(`estimator_weight_sum`, `valid_mask`, `surviving_sample_count`, `support_w1/w2`,
`n_eff_support`); they are never recomputed. They are **not** parent-visible (§2.2), so the
future capture must read them **in the worker** and emit only the sparse samples of §3.1.1.
`observe_edge` copies to float64 for its own maths, so capturing must not mutate patches,
graph, offsets or any science array.

#### 3.1.1 Sparse worker→parent IPC (frozen)

- **Static geometry first.** The relation geometry and the **candidate** global coordinates
  (per relation: the odd-block/relation-band superset that can contain the exact R37
  held-out keys) are computed deterministically from the frozen layout **before** the
  workers start, and passed to each worker.
- **Per-worker extraction.** Before IPC, each worker extracts **only** the sparse candidate
  samples of its adjacent relations from E7–E9 (and the other new planes of M3–M8) — never
  a full f64 patch plane — and returns a **versioned sparse payload** tagged with the
  worker's cell id and candidate-set digest.
- **Parent join.** The parent joins the `u`/`v` payloads, applies the exact R37 held-out
  validity/sample selection, the frozen R34c offsets and the `feather_weight_field` ramps.
  **The exact held-out selection is recovered only through the §1.3 integer/block
  inverse/parity fixtures, fail-closed** (never a raw floor of normalized floats); the
  candidate-set and final pair-set digests are tied to that same frozen rule. If exact
  held-out keys require **both** Cells, workers may return the deterministic geometry
  **superset**; the parent fail-closed intersects/joins to the exact **uncapped** R37 keys
  (no cap assumption).
- **Digests.** The candidate-set digest and the final pair-set digest (§6.2) are both stored
  and verified; a mismatch is refuse/`IDENTITY_MISMATCH`.
- **Bounds (honest).** The existing baseline full-`science`/`valid`/`surviving` patch IPC
  is acknowledged **product behavior** and is unchanged by R40. The **R40 incremental**
  overhead is `O(candidate records)` (sparse samples), **not** `O(full patch E7–E9)`. The
  plan makes no claim that the whole product RAM is `O(records)`.
- **Synthetic tests (mandatory, R40-B2).** Windows spawn/pickling parity and the
  deterministic parent join are exercised on the synthetic corpus: the joined sparse
  output must equal, at the selected coordinates, a reference extraction computed
  in-process.

### 3.2 Contextual observables (C) — no promotion role

| ID | Observable | Why contextual |
|---|---|---|
| C1 | `owner` / multi-Cell corner flag at the pixel (`assemble_canvas_feathered` `owner`) | multi-Cell corners need ≥3 contributors and are out of scope for the two-Cell tests; recorded so they can be excluded (`NOT_APPLICABLE`), never approximated |
| C2 | per-Cell **aggregate** `ReprojectPathStats` (E11) | aggregate per cell, not per sample; usable only as a coarse covariate, never as a per-sample path identity |
| C3 | distance `|s|` and side (`n<0`/`n>0`), tangent block id | geometric context already implied by M2; never mislabelled as support/PSF evidence |
| C4 | `rejection_fraction`, `effective_train`, block counts | relation-level context, cohort-gated (§6), never causal |

### 3.3 Explicitly excluded from the contract (not captured)

- **Per-sample PSF / star diagnostics (E13)** — no PSF data exists; the scene/PSF
  hypothesis (**H5**) is `NOT_AVAILABLE` unless a separately pre-registered, bounded
  instrument is authorized. The capture must **not** fabricate a PSF or a star mask, and
  distance/tangent proxies **do not** test PSF/scene causality (they are context only).
- **Any pixel raster / cutout (E14, E15)** — refused by default. A raster would only be
  justified to test H3 against observed post-feather pixels, which the current prohibition
  on pixel reads and the feather-collapse (E14) both block; if ever needed it requires its
  own pre-registration, an explicit justification, a bounded region and a bounded size.

### 3.4 Deterministic offsets applied at capture (frozen)

M3's pre-offset stage stores the **original float32** Cell science. M3's post-offset stage
applies the **frozen** manifest `cell_feather.offsets` (additive-only, per channel) via the
accepted production helper `zemosaic.core.zegrid.cell_feather.apply_additive_offsets`.
That helper **casts its input to float64 and returns float64**
(`a = np.asarray(science, dtype=np.float64); return a + off…`), so **post-offset science is
f64** per Cell/channel. The capture re-verifies the identity
`(sci_u + off_u) − (sci_v + off_v) == holdout_diff + off_u − off_v` **fail-closed**, using
the **exact f64 helper result and operation order** — **no f32 downcast** of the post-offset
value: it must either (a) be **bit-identical** in f64, or (b) use an explicit **ULP /
absolute tolerance calibrated on synthetic fixtures and frozen in a pre-real revision
(R40-B1, §7)** before any real capture. `INV_TOL = 1e-9` is a coordinate-inversion tolerance
and **must not** be reused for science values without justification. No offset is
re-estimated, smoothed, refined or replaced, and **no unsupported numeric tolerance may be
smuggled in**.

**M3 tolerance — FROZEN by R40-B1 (pre-real; see §15).** On the disjoint synthetic
calibration/validation groups the accepted production helper produced the post-offset value
**bit-identically** for every calibrated case (f32 science across sign / magnitude /
subnormal / large-finite cases), so the frozen M3 pre/post rule is **BIT IDENTITY,
tolerance exactly `0`**, on the **f64** helper result with **no f32 downcast**. The
*separate* algebraically reordered cross-expression is calibrated and frozen as a
**scale-aware ULP bound**
`|(sci_u+off_u) − (sci_v+off_v) − ((sci_u−sci_v) + (off_u−off_v))| ≤ ulp_k · ε_f64 · scale`
with `scale = max(|sci_u|,|sci_v|,|post_u|,|post_v|,|off_u|,|off_v|)` and **`ulp_k = 5`**
(calibration max scale-ratio `0.6338271846408026`; `5` is an explicit IEEE rounding count
`(2+3)` — **not** a fabricated floor-derived bound; the raw absolute error is scale-dependent
— `1.11e-16` at unit scale, `1.41e14` at scale `1e30` — which is exactly why the bound is
expressed scale-aware). **Both rules are scoped to the M3
identity/cross-expression ONLY** and are **never** applied to the OFF/ON arrays/FITS
identity, which stays exact byte identity at tolerance `0`.

---

## 4. Mechanism hypotheses, controlled tests, causal-attribution ladder

All tests are **same-global-pixel paired**, **relation-first**, **both orientations**, and
**do not fit any correction**. No test may add, average, normalize, smooth or model away a
mechanism; each test only *conditions* (observational), *stratifies* (observational), or
*intervenes* (exact same-input manipulation) on the recorded mandatories.

### 4.1 Hypotheses

- **H1 — support/validity discontinuity.** The residual is associated with a systematic
  difference in `valid_mask` / `n_eff_support` across the boundary.
- **H2 — estimator-weight / survivor discontinuity.** The residual is associated with a
  systematic difference in `estimator_weight_sum` / `surviving_sample_count` (a
  depth/weight edge, not a sky edge).
- **H3 — feather-ramp interaction.** The residual is an artifact of the blend. **This
  contract holds no observed separate post-feather Cell contribution and no final seam
  measurement** (E14/E15), so the only available H3 quantity is the **PREDICTED**
  arithmetic `w_u(x)·d_post(x)`; it is **association/context only** and cannot reach
  `CONTROLLED_EFFECT`.
- **H4 — reprojection / interpolation.** The residual is generated by the
  crop→`slice_wcs`→`reproject_cropped` path (crop/slice coverage or fast-path numerics)
  differing between the two Cells at the same global pixel. **This is the one hypothesis
  with a realizable same-input intervention (§5).**
- **H5 — scene / PSF / star contamination.** The residual is real scene structure. With no
  PSF/star data this is `NOT_AVAILABLE`; distance/tangent proxies cannot establish it.

### 4.2 Tests and their exact reach

| Test | Kind | Inputs | Operation | Max reach |
|---|---|---|---|---|
| **T1 (H1)** | **observational conditioning** | M3, M4, M7 | paired indicator contrast, then restrict to both-valid pixels (`valid_u ∧ valid_v`) | `DETECTED_ASSOCIATION` **only** — conditioning is not an intervention |
| **T2 (H2)** | **observational stratification** | M3, M5, M6 | stratify by `estimator_weight_sum` / `surviving_sample_count` strata fixed **before** outcome; contrast within the both-Cells-same-class stratum | `DETECTED_ASSOCIATION` **only** |
| **T3 (H3)** | **predicted arithmetic** | M3, M8 | compare the paired residual against the **PREDICTED** imprint `w_u·d_post`; **no** post-feather pixel read | association/context **only** |
| **T4 (H4)** | **exact same-input intervention** | §5 + §5.1 | production-vs-oracle same-input reprojection comparisons (I1a parity / I1b coverage delta; I2 fast-path parity) **propagated end-to-end through the frozen stack and the same R34c offsets** (bounded shadow, §5.1) | may reach `CONTROLLED_EFFECT` **if** the intervention is identity-proven, the propagation holds, and `delta_d_post` removes/reduces the residual under the R40-B1-frozen criteria |
| **T5 (H5)** | context proxy | M3, C3 | residual vs distance-bin / tangent-block profile | context only; cannot exclude a mechanism |

**Observational vs interventional rule (frozen).** T1/T2/T3/T5 are observational or
predicted: however strong an association survives conditioning, it reaches **at most
`DETECTED_ASSOCIATION`**. Survival or disappearance of an association under conditioning
may **prioritize** hypotheses (an ordered shortlist) and must **never** be reported as an
established cause. Only an **exact same-input intervention** (T4, §5) that changes exactly
one target mechanism while holding every other input array / order / WCS / mask / shape
fixed — with synthetic validation and effect criteria **frozen before** the real outcome —
may reach `CONTROLLED_EFFECT`. A **per-frame or per-ROI reprojection difference alone is
not** comparable to the paired residual: T4 must propagate the intervention end-to-end
through the frozen stack and the **same** R34c offsets (§5.1) before any comparison with
`d_post`.

### 4.3 Causal-attribution ladder (frozen names, exclusive by precedence)

Per relation × channel, per hypothesis, the causal state is exactly one of:

1. **`NOT_IDENTIFIABLE`** — the hypothesis cannot be separated from its confounders with
   the acquired evidence (required observable absent, or two hypotheses predict the same
   pattern, or no same-input intervention is technically possible/bounded). *Default when in
   doubt; also the state when a mandatory observable is missing or `NOT_AVAILABLE`.*
2. **`DETECTED_ASSOCIATION`** — a repeatable association exists between the residual and
   the candidate observable, **relation-first** (≥2 distinct geometric relations per
   orientation, same explicit axis family + sign tuple), with the deterministic
   block-bootstrap band excluding zero. **Correlation — and every observational
   conditioning/stratification result — stops here.**
3. **`CONTROLLED_EFFECT`** — the association is produced/removed by an **exact same-input
   intervention/ablation** that changes **exactly one target mechanism** while every other
   input array, order, WCS, mask and shape is held fixed, with the same relation-first /
   ≥2-per-orientation / sign-tuple rule, and with **frozen effect criteria** (calibrated on
   synthetic truth **before** the real outcome). On real Windows816 data this is reachable
   **only** through T4/§5-style interventions.

**Precedence / reachability rules (frozen).**

- A correlation, an observational conditioning/stratification result, a predicted feather
  imprint, a relation scalar, or a final-FITS pattern **cannot** reach `CONTROLLED_EFFECT`.
- **A synthetic injected effect validates the instrument; it is not causal localization on
  Windows816.** Synthetic recovery may only qualify the capture/tool (R40-B), never a
  Windows816 mechanism.
- A **shared-frame / shared-gauge** control is a **tautological** control (both Cells share
  the same reference/weight frame by construction); it can never be independent causal
  proof, and its agreement may not be counted as an intervention.
- The three correlated channels of one relation are **never** independent relations;
  orientation/panel aggregation counts **distinct geometric relations** only.
- `CONTROLLED_EFFECT` localizes **one** mechanism; it is **not** a correction, not a
  qualification, and not an authorization for any product change (§7).

---

## 5. Exact same-input reprojection interventions (H4)

Only the **existing production/oracle symbols** are used; **no new or higher-order
interpolation kernel** may be introduced as an oracle (a different kernel is a different
estimator and cannot prove production causality). Identity is stated **per intervention** —
there is **no** blanket "identical arrays/masks" rule (such a rule would make I1 impossible,
since I1's whole point is that the local crop and the full source are different arrays).

### 5.1 Bounded end-to-end shadow propagation (frozen)

A per-(frame,cell) target-grid delta is **not** a Cell-stack residual, so it can never by
itself reach `CONTROLLED_EFFECT`. The intervention must be propagated end-to-end under this
exact contract:

1. **Fixed full panel for any panel-level claim.** Use the fixed full **38-relation DEV
   panel**. A partial subset may report **per-relation** results only and forces
   `INCOMPLETE_PANEL` / **no panel-level `CONTROLLED_EFFECT`**.
2. **Bounded ROIs, streaming.** Operate only on preregistered **relation-band ROIs / sample
   tiles**, streaming the frames; **never** materialize full oracle Cells/canvas nor an
   `N_frames × canvas` array.
3. **One changed operation.** For every affected Cell/frame/ROI, compute the **production**
   reprojection and the I1/I2 **oracle variant** from identity-proven the same source (as
   applicable to the chosen intervention).
4. **Same frozen stack.** Pass **both** variants through the **same** frozen canonical-stack
   path/config: same frame order/reference, fixed production normalization coefficients,
   `weight_active`/`norm_active`, frame scalar weights and exclusions. Any downstream
   per-pixel rejection/support consequence caused by the one changed reprojection operation
   is a **treatment mediator** and must be **reported**, not silently held equal.
5. **Synthetic ROI-shadow proof (mandatory).** Prove on synthetic fixtures that the ROI
   shadow output equals the corresponding **full-Cell production** output at the selected
   coordinates **before** any real use, under an **explicit B1-frozen numerical parity
   tolerance** (or exact bit-identity if achievable — no implicit tolerance, no silent
   default); a parity failure is `NOT_IDENTIFIABLE`.

   **FROZEN by R40-B1 (see §15).** When the shadow evaluates only the **pointwise/local**
   stages over **frozen inputs** (layout-derived `feather_weight_field` ramp, the **same**
   frozen production R34c offsets, the same frozen stack weights/supports), the ROI-shadow
   output is **BIT-IDENTICAL** (tolerance exactly `0`) to the full-Cell production value at
   the selected coordinates — verified across patch/core/halo boundaries, both orientations,
   NaN/invalid support, finite flat/gradient/star/diffuse fields and heterogeneous
   weights/survivors. Any shadow that **re-derives a Cell-global quantity from the ROI**
   alone (ROI-only normalisation / robust statistic / re-derived ramp context) is
   **`NOT_IDENTIFIABLE`** — **no tolerance is invented** for it. Reprojection parity is
   reported per path: the plain-TAN production fast path is **bit-identical** to the
   full-source `reproject_interp` reference on the common support; SIP/distorted/non-TAN
   assemblies take the **fallback** and are reported **separately**, never merged into the
   fast-path parity statistic.
6. **Same offsets, never recomputed.** Apply the **same frozen production R34c offsets** to
   both production and shadow Cell science (offsets are **never** recomputed for the
   oracle), then form `d_post_prod` and `d_post_shadow` on the **same global pixels** and
   evaluate `delta_d_post` and whether the shadow **removes/reduces** the residual under the
   **R40-B1-frozen criteria**.
7. **No shortcut to causality.** A per-frame difference alone **cannot** reach
   `CONTROLLED_EFFECT`; a mere matching sign/axis is **insufficient**.
8. **Fail-closed.** Any input/ROI/stack parity failure or incomplete propagation ⇒ H4
   `NOT_IDENTIFIABLE`/`NOT_AVAILABLE`.
9. **Bounds measured, not asserted.** Practical runtime/disk bounds must be **measured** and
   accepted **synthetically** (R40-B2) **before** the human gate. **No actual intervention
   is run now.**

### 5.2 I1 — crop/slice sufficiency (split into two sub-comparisons)

I1 compares the **local production path** `execution.reproject_cropped` (fast path when
eligible, `reproject_interp` fallback otherwise) against the existing full-source **oracle
reference** `execution.reproject_full_source`.

- **Per-intervention identity (I1).** The **cropped** array must be **byte-identical to the
  corresponding section of the same prepared full source** (same source file/section, exact
  source digest, the exact `slice_wcs` mapping the crop bounds, same **target WCS/shape**,
  frozen `REPROJECT_ORDER = "bilinear"`). The cropped and full arrays are **not** identical
  **by construction** — that difference is the intervention, not a violation.
- **I1a — common-valid-support numeric parity.** With a **fixed common valid mask** (the
  intersection of the two paths' valid supports, computed once and frozen), require the
  bilinear outputs to agree on that mask within the R40-B1-frozen tolerance. The mask is
  **held fixed** here; this is a pure numeric-parity test.
- **I1b — support/coverage delta (the intended intervention).** Report the **mask
  difference** itself: where the crop path marks a target pixel invalid (or valid) while the
  full-source path differs, and whether that coverage delta is the mechanism. Here the mask
  difference is the **treatment**, not a violation. Pixels outside the common support are
  `NOT_APPLICABLE`.

### 5.3 I2 — fast-path numerical parity (where applicable)

Compare `execution.reproject_cropped_fast` against `reproject_interp` on the **exact same
cropped array**, the **same WCS**, `order = REPROJECT_ORDER`, the **same target shape**, and
the **same mask** (identity is fully required here — only the code path varies). This
isolates the R21 fast-path numerics (bit-exact by design on plain undistorted TAN).
SIP/distorted/non-TAN cases for which the fast path is ineligible are reported **separately
as `NOT_APPLICABLE`/fallback**, never merged into the parity statistic.

### 5.4 What each intervention isolates / constraints

- I1a isolates **numeric parity** on the common support; I1b isolates the **crop/slice
  coverage** delta (the intervention).
- I2 isolates the **fast-path vs `reproject_interp`** numerical difference only.
- Neither isolates a **true scene** difference (H5) or a **blend** difference (H3); those
  remain context/`NOT_AVAILABLE`.
- **Input identity that *can* be required** (same prepared source bytes/section, exact
  source digest, same `slice_wcs`) must be proven **fail-closed**. Only the quantity the
  intervention intentionally varies may differ (crop extent for I1; code path for I2). If
  identity cannot be proven, return **`NOT_IDENTIFIABLE`/`NOT_AVAILABLE`** — never a soft
  pass.
- The comparison is **read-only** and offline; nothing in the production path is modified,
  the oracle result is never blended, never written into product output, and never becomes a
  correction. The I1 oracle (`reproject_full_source`) reads only the **prepared source FITS**
  (`memmap`, read-only) — **never** the final product FITS — and this H4 path is
  `NOT_RUN`/`NOT_AUTHORIZED` now (§3, §11).
- `CONTROLLED_EFFECT` for H4 requires §5.1 to hold (identity-proven intervention, bounded
  end-to-end propagation, same offsets, `delta_d_post` removing/reducing the residual under
  the R40-B1-frozen criteria, relation-first ≥2 distinct relations per orientation, sign
  consistent). A mere amplitude difference does not qualify.
- **A sensitivity experiment with a different interpolation kernel, if ever mentioned, is
  contextual only and is never an oracle and never `CONTROLLED_EFFECT`.**

---

## 6. Sampling, splits, thresholds, hash binding, size

### 6.1 Split freeze (before any new outcome)

- **Relation-first.** The panel is the `38` geometric relations (not `114` records).
- **Whole spatial blocks.** Any resampling is by whole tangent blocks (`block_px = 4`,
  `row // 4` for `h`, `col // 4` for `v`), reusing the frozen §1.3 mechanics — no
  pixel-level mixing.
- **Both orientations.** `h` and `v` are reported separately; a claim needs the same
  sign tuple in ≥2 distinct relations in **each** orientation.
- **No RGB pseudo-replication.** The three channels are correlated measurements of one
  boundary and are aggregated relation-first.
- **Same global pixels.** Only the frozen §3 sample set.
- **DEV consumed status.** The corpus is exhausted for qualification; R40-C is
  `DEV_NOT_QUALIFICATION` diagnosis only.

### 6.2 Deterministic binding

A durable R40 capture must bind and fail-closed on: capture-contract version; artifact
sha256/bytes; `capture_id`; source-set digest/algo and frame count; layout/canvas;
`measurement_px`/`feather_px`/`halo_px`/`block_px`; `holdout_capped == false` and
`qualification == EXACT`; the §1.3 integer/block inverse fixtures; and the repo HEAD
`597c6dbb…` (or a later **explicitly re-blessed** HEAD recorded in the capture). The
canonical sample-set digest is
`sha256("zemosaic.zegrid.r40.pairset.v1\n" + sorted(sample keys))`; it is stored and
compared on every resume.

### 6.3 Stale invalidation and resumability

A checkpoint is valid only if every bound identity above matches; otherwise it is
**stale-invalidated** (discarded, recomputed) — never partially reused. No final report is
emitted while any relation is incomplete.

### 6.4 Threshold policy (INHERITED vs NEW vs REFUSED)

| Threshold | Value | Provenance |
|---|---|---|
| `RMS0 ≤ 1.25 × σ_ref` | `1.25` | **INHERITED** (R37 absolute RMS-around-zero gate, applied to the statistic it was frozen for) |
| support gates `effective_train ≥ 256`, `tangent_blocks ≥ 8`, `normal_blocks ≥ 3` | as stated | **INHERITED** (R37 contract) |
| `K = 4` folds, bootstrap `B = 2000`, seed `0` | as stated | **INHERITED** (R39 plan) |
| coordinate inversion `INV_TOL = 1e-9` | `1e-9` | **INHERITED** — **coordinate inversion only**; never reused for science values |
| relation-first cohort `≥ 8` distinct relations for any cross-relation report | `8` | **INHERITED** (R39 §5.3 contextual cohort minimum) |
| same axis family + sign tuple in `≥ 2` distinct relations per orientation | `2` | **INHERITED** (R39 §9.3 mechanics) |
| M3 identity tolerance | **`0` (bit identity)** | **FROZEN by R40-B1** (§15): bit-identical helper-output recomputation was achievable on every synthetic calibration/validation case, so no tolerance is used; the *separate* algebraic cross-expression uses `ulp_k = 5` scale-aware (`|Δ| ≤ 5·ε_f64·scale`, an explicit IEEE rounding count `(2+3)`). **Both are scoped to the M3 identity ONLY** — they **never** apply to the OFF/ON array/FITS identity |
| **OFF/ON science/coverage/aesthetic array + FITS identity** | **exact byte-identity — tolerance ZERO** | **FROZEN NOW (§8.2)** — a mandatory safety gate; **never** calibrated, widened or relaxed in R40-B1/B2 or anywhere else |
| injected-effect recovery criterion (synthetic) | `reduction_rms ≥ 0.20` (inherited), `p90` worsening `≤ 5%` (inherited), `RMS0 ≤ 1.25·σ_ref` (inherited), `≥ 2` relations/orientation, `≥ 1` isolated family | **FROZEN by R40-B1** (§15.5) — calibrated on disjoint synthetic calibration/validation groups, validated **once** on the independent validation group; inherited gates reused verbatim, **no** threshold derived from R39 observed values. The synthetic injection amplitude is **generator metadata only** — there is **no** amplitude-promotion gate. The recovered/common-support delta is **`ORCHESTRATION_ONLY_NON_GATING`** (a structurally expected orchestration diagnostic) and gates **nothing**; the earlier `1e-12` recovered-Δ tolerance framing is **withdrawn** and appears nowhere in the frozen v2 criteria |
| controlled-effect admission criteria for H4 (identity-proven intervention + residual-structure reproduction, relation-first, ≥2/orientation, sign-consistent) | see §15 (`h4_controlled_effect_admission`) | **FROZEN by R40-B1** (§15): identity + parity gates, complete 38-relation panel, same global pixels / same fixed stack, exactly **one** isolated intervention family, both orientations with a consistent sign/axis signature, and the inherited `RMS0`/`20%`/`p90 5%` gates on `delta_d_post` after the same frozen offsets; RGB pseudo-replication, per-frame-delta-only, sign-match-only and amplitude-only are hard refusals; there is **no** amplitude-promotion gate |

**REFUSED (must never appear).** Any threshold derived from the observed R39 values
(`2.180`, `1.253–12.945`, `1.100`, `0.539`); any post-hoc relaxation/narrowing of the
`1.25 × σ_ref` gate; any threshold tuned so a predicted imprint or a conditioning result
becomes significant; any unsupported numeric tolerance presented as inherited; **any non-zero
"zero-influence tolerance" — the OFF/ON array/FITS identity is exact byte-identity at
tolerance `0` (§8.2) and may never be relaxed**; any "explained fraction" summing unrelated
proxies. A threshold that cannot be justified from synthetic calibration or an already-accepted
frozen requirement is **refused**, and the corresponding test returns `NOT_IDENTIFIABLE`.
The M3 ULP/absolute tolerance, if needed, applies **only** to the M3 pre/post algebraic
identity — never to the OFF/ON arrays/FITS.

### 6.5 Size / runtime estimate (ESTIMATE — from dimensions/counts, no pixels read)

Record unit is `relation × channel × global-pixel` (§3). Sample count ≈
`114 relation-channels × n_holdout` with `n_holdout ∈ [1792, 4448]`, median `2857` →
**≈ 3.3 × 10⁵ records** (ESTIMATE).

Per-record bytes (per channel, one Cell pair), conservative **repeated-per-channel** form;
M3 is stored as pre-`f32` + post-`f64`, M6 as the canonical `int64` (§3.1):

| Field group | Bytes |
|---|---|
| `gx`, `gy` int32 ×2 | 8 |
| `sci_u/sci_v` **pre f32** | 8 |
| `sci_u/sci_v` **post f64** | 16 |
| `valid_u/valid_v` bool | 2 (pad 4) |
| `estimator_weight_sum_u/v` f64 | 16 |
| `surviving_sample_count_u/v` **int64** | 16 |
| `support_w1/w2/n_eff` u/v f64 (repeated, channel-invariant) | 48 |
| `ramp_u/ramp_v` f64 | 16 |
| corner flag / padding | ~2 |
| **≈ total** | **≈ 136 B** |

If (and only if) channel-identical pixel sets and channel-invariant planes are **proven**
per relation, `support_w1/w2/n_eff` and `ramp` factor into a `(relation, pixel)` sidecar →
per-record ≈ **≈ 70 B** plus sidecar ≈ **≈ 74 B/pixel**.

- **Packed binary shard:** ≈ `3.3e5 × 136 B` ≈ **45 MB** (conservative) / ≈ `3.3e5 × 70 B`
  + `≈1.1e5 × 74 B` ≈ **31 MB** (factored) → order **30–75 MB** with index/overhead
  (ESTIMATE).
- **Strict JSON per-sample:** ≈ 5–8× → order **0.2–0.6 GB** (ESTIMATE) — **not**
  recommended for the per-sample block.

**Recommendation (frozen):** a **bounded shard** format — one fixed-width binary/`npy`
shard per relation plus a strict-JSON index/header (§8) — with a **hard disk budget** and
**fail-closed refusal** if exceeded. Only the run's own fresh output directory is written;
no absolute private paths in any durable artifact.

**Runtime (ESTIMATE):** the capture is write-only instrumentation over an already-computed
stack; cost is `O(candidate records)` serialization on top of the normal assembly. Expected
**minutes**, not the multi-hour stack; measured on the synthetic corpus in **R40-B1/B2** and
on the single R40-C run. **Honest bound:** the existing baseline full-patch
`science`/`valid`/`surviving` IPC is product behavior and is **not** changed by R40; the
**R40 incremental** overhead is `O(candidate records)` (sparse samples, §3.1.1), **not**
`O(full patch E7–E9)` and **not** an `N_frames × canvas` accumulation. This plan does **not**
claim the whole product RAM is `O(records)`.

---

## 7. Staged gates

| Gate | Content | Authorized now? |
|---|---|---|
| **R40-A** | This pre-registration + read-only code/artifact archaeology (docs-only). | YES (this task only) |
| **R40-B1** | **Synthetic calibration stage:** synthetic truth corpus with **disjoint calibration and validation groups** (criteria derived on the calibration group, validated once on an independent validation group at family/parameter/event level). Calibrates **only**: the M3 pre/post algebraic identity rule (bit-identity achieved), the §5.1 ROI-shadow numerical parity rule, the injected-effect recovery criterion, and the H4 controlled-effect criteria; **frozen in a pre-real document revision** (§15). Does **not** calibrate, and may **not** touch, the OFF/ON byte-identity gate (frozen at zero in §8.2). No real data. **STATUS: v1 `B1_PASS` WITHDRAWN as `B1_FAIL_PROTOCOL_INVALID`; the v2 reserved cohort was evaluated EXACTLY ONCE by the separate `ZM-ZEGRID-R40-B1V-ONE-SHOT` mission → terminal **ACCEPT** (`B1_PASS`, 18/18, claim `COMPLETED`, `r40b2_authorized=true`) at HEAD `61c4aa5`** (§15.7). | **DONE / terminal ACCEPT** (v1 invalidation history preserved byte-for-byte, §15.1) |
| **R40-B2-A** | **Capture-core stage (bounded task A):** the reusable default-OFF capture data contract + the bounded atomic shard-generation core (frozen dtype/schema, sample identity, hard disk budget, path safety, privacy, fail-closed identity, stale/resume, manifest link, sparse payload/parent join) implemented and tested on **synthetic / in-memory fixtures only**, with **no** production seam wired. | **DONE** (`ZM-ZEGRID-R40-B2-A-SHARD-CORE`, §15.8) — the shard core alone does **NOT** accept B2 |
| **R40-B2** | **Implementation + acceptance stage:** default-OFF diagnostic-capture implementation + synthetic qualification, **using the R40-B1-frozen criteria**, and **proving the exact OFF/ON byte-identity (zero tolerance, §8.2)**; prove bounded RAM/disk, fail-closed identity, stale invalidation and the **real product ROI parity**. | **Implementation AUTHORIZED** by the terminal B1 v2 ACCEPT; seam wiring is **DONE** (`ZM-ZEGRID-R40-B2-B`, §15.9) and the **real pooled worker→parent sparse transport is DONE** (`ZM-ZEGRID-R40-B2-C`, §15.10); the **zero-tolerance OFF/ON product byte-identity is now MEASURED on real synthetic end-to-end product runs** (`ZM-ZEGRID-R40-B2-D`, §15.11); **acceptance NOT achieved** — the 38-relation panel OFF/ON qualification, the injected-effect recovery re-proof, the real product ROI parity, the exact §1.3 held-out intersection and a real Windows run remain `NOT_RUN` |
| **HUMAN_GATE** | Explicit **Tristan** authorization of the exact commit, exact commands and a **fresh output path** on a **capable (Windows)** host. | NO — not authorized now (§11) |
| **R40-C** | **One** exact Windows816 DEV diagnostic capture (`DEV_NOT_QUALIFICATION`), then offline interpretation of the controlled tests under the R40-B1-frozen criteria. | NO — after R40-B2 + HUMAN_GATE |
| **R40-D** | Interpretation/decision **only if** an identity-proven same-input intervention yields `CONTROLLED_EFFECT` for exactly one mechanism. | NO — conditional |

**Staging rule (unambiguous).** R40-B is **split**: **R40-B1 must produce frozen criteria
and a document revision BEFORE R40-B2 acceptance, and both must complete BEFORE any real
R40-C outcome.** No criterion may remain "to be frozen" at the human gate. R40-B1's
calibration and validation sets/groups must be **disjoint** (criteria derived on the
calibration group, validated **once** independently on the validation group) and must use
**no R39 observed values**.

**Hard rule.** No diagnostic detection at any gate authorizes a correction, a prototype, a
qualification, a product default, a GUI change, or product integration. A correction would
require a **separate future plan** and an **independent real corpus**; R40 by itself
produces evidence, not a fix.

**Interface boundaries (so the plan stays bounded and implementable):**

- **DRY (implementation):** the capture lives in a new default-OFF seam plus a new
  standalone offline tool under `tools/`, reusing `MiniTile`/`CanonicalStackResult`,
  `cell_feather.feather_weight_field`, `apply_additive_offsets`, the `execution.reproject_*`
  primitives and the R39-A identity/inverse/strict-JSON/privacy machinery — **no
  duplicated pixel or difference maths**.
- **DRY (offline analysis):** R40-C interpretation reuses the R39-A relation-first,
  leak-free bootstrap and state machinery (as frozen) wherever a statistic is shared; new
  logic is limited to the covariate conditioning/stratification of §4 and the §5
  interventions.
- **YAGNI:** **no** new model family, **no** correction, **no** GUI, **no** production
  default; the capture is a bounded, removable, default-OFF instrument.
- **Deterministic:** all sampling/keys/hashes are deterministic; no hidden RNG.

---

## 8. Future capture safety contract

A future R40-B2 capture must satisfy, and prove, all of:

1. **Default OFF.** Not enabled unless explicitly requested; normal runs unchanged.
2. **Zero influence proven (tolerance exactly ZERO, frozen now).** With the capture ON but
   its output discarded, every **science, coverage and aesthetic array and the
   science/coverage/aesthetic FITS bytes** must be **exact byte-identical** to capture-OFF
   for the same inputs and seeds — **tolerance exactly `0`, frozen now in this section, and
   never calibrated, widened or relaxed** (this safety gate is **not** a synthetic-calibrated
   threshold).
3. **Manifest scheme (one truthful design, used consistently; wording scoped by rework-16,
   no obligation or tolerance changed).** The diagnostic is recorded as **exactly one
   additive block** `r40_diagnostic_capture` (mirroring the accepted R37-D
   `r37_diagnostic_capture` cross-link) **ON and NONE OFF**; **the manifest is never claimed
   byte-identical** — not as a whole, not as a claim of any kind. The **load-bearing
   obligation is UNCHANGED and EXACT** (identical to clause 2): the science, coverage and
   aesthetic **arrays**, the science/coverage/aesthetic **FITS bytes**, and **any other
   product array artifact emitted** must be **exactly byte-identical** ON vs OFF at
   **tolerance exactly `0`** — **never** calibrated, widened, relaxed or reinterpreted, here
   or anywhere else. The capture artifact is cross-linked to the manifest on the frozen
   identity keys (`capture_id`, `filename`, `sha256`, `bytes`) exactly as the accepted
   R38/R39 `manifest_links_artifact` rule requires. **Canonical projection.** Every
   pre-existing manifest field that is **plan-independent product/metadata** must be
   **canonical-identical** (same keys, same values, same serialisation) ON vs OFF. The
   manifest's inherently **run-volatile observability** fields are **exactly enumerated and
   excluded** from that canonical projection: **(a)** wall timing and **RSS** measurement
   leaves; and **(b)** the **environment-driven scheduler plan** — per-cell `mode`, per-cell
   footprint bytes (`per_cell_footprint_bytes`), concurrency / plan candidates
   (`cells_in_flight`, `plan_candidates`), executor and worker counts (`executor`, `workers`,
   `cache_build_workers`, `cache_workers_per_cell`), the reprojection path/counters, and the
   RAM-budget / available-memory figures (`available_bytes`, `ram_budget_bytes`) — **plus the
   per-cell `mode` / `bound_bytes` the planner stamps into each cell record**. These are pure
   functions of the free RAM sampled at plan time, so they differ between **any** two
   sequential runs (even OFF vs OFF); that is a **measured** fact, not an exemption. Nothing
   else is run-volatile, and **nothing in this list may be widened by assertion**.
   **Completeness is evidence-backed, never assumed.** The exclusion set is complete **only**
   as proven by measurement: **every** leaf that differs between **two capture-OFF runs** AND
   between runs with **forced different plans** must be **classified** volatile, and a focused
   test **MUST fail loudly** on any **unclassified** differing leaf (it may **never** silently
   normalise). Measured ON vs OFF, the **only** manifest difference beyond the single additive
   block must lie in the enumeration above.
   **No product field may ever be excluded.** The classifier may **never** cover a product
   field — science / coverage / aesthetic / uint8-uint16 render, contract, layout, cell
   product fields, or offsets. The `.cells` subtree is **never** blanket-excluded: under a
   plan flip **only** the two plan-metadata leaf names `mode` / `bound_bytes` differ, and every
   science/index/status/provenance per-cell field stays identical. (Alternative, if ever
   preferred: an external sidecar cross-link with **no** manifest mutation — but exactly one
   of the two designs is used, and it is stated in the capture provenance.)
4. **Atomic, strict output (frozen shard contract).** Fixed basename; temp file in the
   destination directory + `flush` + `fsync` + `os.replace`; temp removed on failure;
   `json.dumps(..., allow_nan=False)` must succeed; non-finite values refused, never degraded
   to `0`; deterministic ordering everywhere. The bounded shard format is frozen as:
   explicit **schema name/version** and dtype field list; **little-endian** (or the exact
   chosen endianness, declared); **C-order**; a **record ordering/sample key** (§3); a
   per-shard **count + byte length + SHA-256**; a **strict-JSON index** describing the
   shards; **atomic multi-file generation** written into a fresh generation directory with
   the **index/COMMIT marker written last**; `fsync` on every shard; **partial-shard cleanup
   / quarantine** and **stale-generation refusal** on any failure; **symlink / path-traversal
   / protected-root** failures closed. **No pickle, no object dtype.**
5. **No absolute private paths** in any durable artifact/report (whole value and embedded;
   POSIX/Windows-drive/UNC/`file://`); a surviving leak sets `status = FAILED` (same
   fail-closed privacy boundary as R38-A/R39-A).
6. **Source/dataset read-only, fresh output writable.** The 816 source frames **and the
   source/dataset roots** are opened strictly **read-only** and are never written. The
   **explicitly provided, fresh output directory** is the **only** writable target; the
   capture refuses any target inside the source/dataset trees or a previously-existing
   protected root (same policy as R39-A `derive_dataset_root` / protected-roots refusal).
   Calling the authorized fresh output a "dataset root" is forbidden — it is the run's own
   writable output.
7. **Fail-closed identity / cross-link.** Refuse (no metrics, no partial artifact) unless
   every §6.2 binding matches on schema/version, `capture_stage`, `correction_applied is
   False`, `production_wired is True`, artifact sha256/bytes == manifest == on-disk,
   `capture_id`, source-set digest/algo/count, `n_frames`, layout, canvas, gauge mode,
   `measurement_px`/`feather_px`/`halo_px`, `holdout_capped is False`, `qualification ==
   EXACT`, `n_dropped == 0` — and the §1.3 inverse/parity fixtures pass.
8. **Bounded RAM / disk.** `O(relations × samples)` scalars only; a hard disk budget with
   fail-closed refusal; no `N_frames × canvas` accumulation.
9. **Resumable / stale-invalidated.** Atomic checkpoint between fully completed records;
   any identity mismatch → stale invalidation; no final report while partial.
10. **No retained pixel cutouts** unless explicitly justified in a separate
    pre-registration with a bounded region and size (default: refused).
11. **Completeness scope (M1–M11 vs H4).** The "a capture missing any mandatory item is
    refused" rule of §3.1 applies to the **core association capture** (M1–M11 → T1/T2/T3/T5
    evidence). The §5 H4 intervention evidence is a **separate, optional/conditional
    block**: its absence yields H4 `NOT_IDENTIFIABLE`/`NOT_AVAILABLE` **only** and does
    **not** invalidate the T1/T2/T3/T5 association evidence; conversely, a present H4 block
    can never silently become "complete H4"; it must independently satisfy §5/§5.1.

---

## 9. Exact STOP outcomes

Each is terminal for the affected step and **must never silently become a correction**:

| STOP | Trigger |
|---|---|
| `EVIDENCE_UNAVAILABLE` | a mandatory observable (M1–M11) is absent, unavailable, or a required input identity cannot be proven |
| `INCOMPLETE_PANEL` | any planned relation is `NOT_RUN` or the relation panel is not `114/114` for the captured set |
| `IDENTITY_MISMATCH` | any §6.2 binding fails (schema, sha, bytes, capture_id, digest, HEAD, layout, gauge, widths, `holdout_capped`/`qualification`) |
| `UNSUPPORTED_CONFOUNDER` | a hypothesis cannot be tested without a confounder the evidence cannot hold fixed |
| `ASSOCIATION_ONLY` | only `DETECTED_ASSOCIATION` is reached (correlation or observational conditioning only, no identity-proven intervention) — **no** promotion, never `CONTROLLED_EFFECT` |
| `NUISANCE_LINKED` | the residual is attributed to a nuisance (support/weight/depth/reprojection/PSF/mask) rather than an unexplained mechanism |
| `NO_EFFECT` | the controlled test/intervention finds no repeatable association in either orientation |

`ASSOCIATION_ONLY` / `NUISANCE_LINKED` / `NO_EFFECT` all block any correction/qualification
path; `EVIDENCE_UNAVAILABLE` / `UNSUPPORTED_CONFOUNDER` block the affected hypothesis and
return `NOT_IDENTIFIABLE`. **None may silently become a correction.**

---

## 10. Acceptance criteria (this documentation task) and next single step

**Acceptance criteria for ZM-ZEGRID-R40-A-OBSERVABILITY-PREREG (rework-3):**

1. Exactly one new document is added:
   `docs/refactor/R40_UNEXPLAINED_STRUCTURE_OBSERVABILITY_PLAN.md`; no other file changes.
2. It states the immutable lineage (§1) with the exact R39 artifact/metrics SHAs, the
   implementation HEAD, status and counts, and the uncapped R37E illustration.
3. It contains the evidence-availability matrix and the explicit insufficiency argument
   (§2), cross-checked against real production symbols (no invented arrays).
4. It freezes a mandatory-vs-contextual NEW observable contract at the pre-feather/feather
   boundary (§3) with **one coherent `relation × channel × global-pixel` unit**, correct
   dtypes (M3 pre `f32`/post `f64`, M6 `int64`), and a frozen **sparse worker→parent IPC**
   design (§3.1.1); it states the seam/propagation gap honestly and makes no false
   parent-seam or whole-product-RAM claim.
5. It separates hypotheses and tests (§4) and defines the ladder so that **observational
   conditioning/stratification reaches at most `DETECTED_ASSOCIATION`**, and
   `CONTROLLED_EFFECT` requires an exact same-input intervention with frozen criteria;
   synthetic validation is not Windows816 causal localization.
6. It specifies **only existing production/oracle reprojection symbols** (§5), states
   identity **per intervention** (I1a fixed-common-mask parity vs I1b intended coverage
   delta; I2 full identity), freezes the **bounded end-to-end shadow propagation** (§5.1),
   and fails `NOT_IDENTIFIABLE`/`NOT_AVAILABLE` when identity cannot be proven; no
   higher-order kernel as oracle, no raw per-frame delta claim.
7. It freezes sampling/splits/binding/size and the INHERITED/NEW/REFUSED threshold policy
   (§6), with no unsupported `≤1e-12`, refusing thresholds derived from observed R39 values,
   and states that the **OFF/ON byte-identity gate is zero-tolerance, frozen now**, and that
   the M3 ULP/absolute tolerance applies only to the M3 identity, never to OFF/ON arrays/FITS.
   It scopes FITS access explicitly (core capture = in-memory only; the future H4 oracle reads
   only prepared source FITS; never final product FITS — §3).
8. It defines the **split** R40-B1/B2 stages (§7, disjoint synthetic calibration/validation),
   the capture-safety contract (§8, with the additive `r40_diagnostic_capture` manifest block,
   the frozen shard contract §8.4 and the M1–M11 vs H4 completeness scope §8.11), exact STOP
   outcomes (§9), the Windows-prohibition reconciliation (§11) and the un-pushed/local status
   (§12).
9. `git diff --check` is clean; `git status` shows only the new untracked/added document;
   no commit and no push are made.

**Next single step.** R40-B1 is now **terminal ACCEPT** (§15.7): the v2 reserved cohort was
executed **exactly once** by `ZM-ZEGRID-R40-B1V-ONE-SHOT` (`B1_PASS`, 18/18, claim
`COMPLETED`, `r40b2_authorized=true`) at HEAD `61c4aa5`, and the v1 invalidation history is
preserved byte-for-byte (§15.1). **R40-B2 implementation is authorized**; its bounded **task A**
(the default-OFF capture data contract + atomic shard-generation core, §15.8) is **DONE** on
synthetic / in-memory fixtures. **B2 is NOT accepted by that shard core**: the production seam
wiring (sparse worker→parent extraction into `_assemble_per_cell_feathered`), the **exact OFF/ON
science/coverage/aesthetic array + FITS byte identity (tolerance 0)**, the synthetic
injected-effect recovery re-proof and the **real product ROI parity** are all `NOT_RUN` and
remain `B2_MANDATORY`. **R40-C / R40-D / HUMAN_GATE / LP341 remain `NOT_RUN` /
`NOT_AUTHORIZED`.** No real capture, no model fit, no FITS/pixel read, and no product change is
authorized. Acceptance authority remains Jarvis Junior's.

**Vocabulary (frozen, to prevent conflation).**

- *DEV diagnosis*: R40-C on the consumed Windows816 corpus → `DEV_NOT_QUALIFICATION`.
- *Causal localization*: reaching `CONTROLLED_EFFECT` for exactly one mechanism via an
  identity-proven same-input intervention on DEV.
- *Synthetic qualification*: R40-B1/B2 proving OFF/ON zero influence and injected-effect
  recovery on a synthetic truth corpus (validates the instrument, not a Windows816 cause).
- *Independent real qualification*: a future, separately pre-registered corpus (LP341
  reserved) — **not** part of R40-A/B/C/D as defined here.

---

## 11. Windows / Zealfie prohibition reconciliation

The standing prohibition is absolute and remains so: **no Zealfie update, no Windows run,
no TinyDebian full stack** is performed or authorized by this task, and R40-A does not
run any product code.

This document may **describe** the future HUMAN_GATE, but it does **not** authorize it:
any single R40-C diagnostic capture is an explicit HUMAN_GATE requiring **Tristan's**
authorization, with the **exact commit**, the **exact commands**, a **fresh output path**,
and a **capable host**. Until that authorization exists, R40-C is `NOT_RUN /
NOT_AUTHORIZED` — the same status as R39-B/C/D. No full product run, release, tag or push
is implied by any R40 gate.

---

## 12. Local commit / publication status

The local `beta` line now carries **four** local commits ahead of `origin/beta` and is **not
pushed**: `597c6db` (R39 spatial residual diagnostic), `50b30eb` (R40 pre-registration),
`c8f2fd3` (R40 synthetic calibration) and `61c4aa5` (R40 bundle verification against the current
HEAD = the terminal B1 v2 `repo_head`). The bounded `ZM-ZEGRID-R40-B2-A-SHARD-CORE` mission adds
**two new untracked files** (`src/zemosaic/core/zegrid/r40_capture.py` and
`tests/test_zegrid_r40_capture_core.py`) and **no** commit and **no** push. No publication,
release, tag, deploy or remote resource is created or implied by R40.

---

## 13. Validation of this document (R40-A design gate)

- `git diff --check` clean; `git status --porcelain=v1` shows only the new document.
- Scan for private absolute paths (none) and for contradictory authorization claims
  (no claim that R38-B/C/D/E or R39-B/C/D ran or is authorized; a single, consistent
  statement that R39-A remains terminal and R40-A authorizes nothing beyond this doc).
- Re-verify every cited immutable fact of §1 against the existing capture/manifest/schema
  and source **without** computing any new outcome; cross-check every production symbol in
  §2.2/§3/§5 against the real modules.
- Confirm the conclusion ladder is mutually exclusive by precedence, that observational
  conditioning cannot reach `CONTROLLED_EFFECT`, that no p90/RMS statistic is mislabelled,
  and that no LP pixel and no FITS pixel is read.

---

## 14. Revision-history pointer

This section is a pointer, not a source of authority.

- **rework-1 (after Junior review-0), before any real R40 outcome:** causal-ladder
  correction (conditioning ≠ intervention); record-unit/dimension coherence; reprojection
  intervention restricted to existing production/oracle symbols; zero-influence/manifest
  scheme fixed to one additive `r40_diagnostic_capture` block with canonical-identical
  pre-existing fields (**scoped by rework-16**, §8 clause 3, to plan-independent
  product/metadata fields — the run-volatile observability fields are excluded);
  `≤1e-12` removed in favour of bit-identical recomputation or a
  synthetic-calibrated tolerance frozen in R40-B1; factual/wording fixes; R40-B split into
  B1 (calibration/freeze) and B2 (implementation/acceptance).
- **rework-2 (after Nono review-0: 0 HIGH / 4 MEDIUM / 3 LOW), before any real R40 outcome:**
  R1 per-intervention identity (I1 split into I1a fixed-common-mask parity and I1b intended
  coverage delta; blanket identical-arrays/masks rule removed); R2 M3 dtype (pre `f32`, post
  `f64`, identity on the f64 helper with no f32 downcast; M6 corrected to `int64`; size table
  corrected); R3 H4 bounded end-to-end shadow propagation contract (§5.1); R4 corrected seam
  claim + frozen sparse worker→parent IPC (§3.1.1); LOW: R40-B1 disjoint synthetic
  calibration/validation sets, frozen shard contract (§8.4), M1–M11 vs H4 completeness scope
  (§8.11).

- **rework-3 (after Nono review-1: 0 HIGH / 2 MEDIUM / 3 LOW), before any real R40 outcome:**
  M1 split the §6.4 zero-influence row — OFF/ON science/coverage/aesthetic array + FITS
  identity is **exact byte-identity, tolerance ZERO, frozen now** (§8.2), never calibrated or
  relaxed, while B1 calibrates only the injected-effect recovery criterion (and, if needed,
  the M3 identity tolerance and the ROI-shadow parity tolerance); M2 added the explicit
  FITS-scope carve-out (§3/§5.4: core capture = in-memory only, never final product FITS;
  the future H4 oracle reads only prepared source FITS, `NOT_RUN`/`NOT_AUTHORIZED` now).
  LOW: M3 tolerance scoped to M3 only; §5.1 ROI-shadow parity tolerance named as B1-frozen;
  §3.1.1 parent exact selection tied fail-closed to the §1.3 inverse/parity fixtures.

- **rework-4 (R40-B1 synthetic calibration + criteria freeze), before any real R40 outcome:**
  this revision is produced by the separately bounded `ZM-ZEGRID-R40-B1-SYNTH-CALIB`
  mission. It **calibrates on synthetic truth only** and **freezes** the previously-open
  numerical criteria (§3.4 M3 identity + algebraic cross-expression, §5.1 ROI-shadow parity,
  §6.4 injected-effect recovery, §6.4 H4 `CONTROLLED_EFFECT` admission) in a pre-real
  revision. The **OFF/ON science/coverage/aesthetic array + FITS identity remains exact byte
  identity at tolerance `0`** and was **not** calibrated, widened or reinterpreted (Nono
  LOW-A preserved verbatim).
- **rework-5 (R40-B1 rework-1: v1 invalidation + v2 prevalidation), before any real R40
  outcome:** after Nono review-0 returned 3 HIGH / 5 MEDIUM / 2 LOW, the v1 `B1_PASS` is
  **withdrawn** as **`B1_FAIL_PROTOCOL_INVALID`**, the v1 validation cohort is recorded as
  **consumed/compromised** (no rerun), and rework-1 repairs the methodology (fail-closed
  non-redacting privacy with recomputable hashes; deep-freeze + per-decision hash
  verification; durable content-addressed one-shot claim; relation-first correlated RGB with
  a 2-of-3 rule and hard override; real I1a/I1b/I2 arms on the production primitives with the
  actual canonical stack; analytic (non-floor) bounds; family-level split disjointness) and
  publishes a **v2 prevalidation bundle** `READY_FOR_NEW_VALIDATION`. **No v2 validation
  outcome is computed, no PASS is claimed, and R40-B2 is NOT authorized** — *historical record of the
  rework-5 state; superseded by the terminal v2 ACCEPT (§15.7)*.
- **rework-6 (R40-B1 rework-2: prevalidation-package repair), before any real R40 outcome:**
  after Nono review-1 (2 HIGH / 4 MEDIUM / 2 LOW, all code-level — the v2 reserved cohort was
  never evaluated and stays fresh), rework-2 closes: the deep-freeze **binding gap** (the
  persisted payload is now the authority; canonical equivalence among payload, canonical JSON,
  deep-frozen object and hash is required, and `h4_admit` consumes the bound snapshot only);
  the reserved-validation path now **applies the frozen criteria** (a frozen expected decision
  per group, panel groups routed through `h4_admit`, no default pass); a **whole-bundle
  preflight** (content hash, status, no-outcome state, cohort, expected decisions, code and
  repository-HEAD identity) runs **before** the durable claim; I1b is applicable on TAN **and**
  non-TAN with **independent** relations and fails when they cannot be formed; the
  recovered-delta rule is **measured** on the real I1b arms (not a hardcoded zero); and the
  durable claim validates its state directory/markers against symlink/path substitution. The
  v1 invalidation and the v1 artifact are preserved byte-for-byte, and the v2 prevalidation
  bundle is re-issued `READY_FOR_NEW_VALIDATION` with **no PASS and no B2 authorization** (§15).
- **rework-7 (R40-B1 rework-3: prevalidation-package repair), before any real R40 outcome:**
  after Nono review-2 (0 HIGH / 3 MEDIUM / 2 LOW, code-level — the v2 reserved cohort was never
  evaluated), rework-3 closes **RM1** (the code identity now binds an explicit, sorted,
  audited **product source manifest** of every in-repo module the validation arms invoke, plus
  the numeric **runtime dependency versions**, fail-closed and recomputed at preflight, so a
  dirty product edit with an unchanged HEAD refuses), **RM2** (the durable claim key now carries
  the canonical **`code_identity_sha256`**, so different frozen code identities cannot share a
  claim id) and **RM3** (the recovered/common-support delta is relabelled
  **`ORCHESTRATION_ONLY_NON_GATING`**, cannot gate H4 admission, and never authorizes B2). The
  prevalidation bundle is re-issued **non-final across a future commit** because the bindings
  include the repository HEAD; the v1 artifact and invalidation remain byte-for-byte. **No PASS,
  no B2 authorization, no v2 validation outcome** (§15).
- **rework-8 (R40-B1 freeze-closure: manifest completeness + truthfulness), before any real R40
  outcome:** after Nono review-3 (0 HIGH / 1 MEDIUM / 2 LOW, code-level — the v2 reserved cohort
  was never evaluated), the **freeze-closure** package closes **RM1** (the 11-module allowlist was
  proven incomplete against the real 24-module import closure, so the manifest now binds the
  **complete `src/zemosaic/**` product source tree** by **raw-byte** SHA-256 with a deterministic
  sorted path set, a manifest count/policy, a fail-closed preflight re-probe of the exact path set,
  and a closure audit that proves **zero** loaded local product module is unbound),
  **L1** (`PHASE = "freeze-closure"`, a stable truthful package label — the immutable v1
  invalidation record keeps its historical `rework-1` phase and stays byte-reproducible) and
  **L2** (`tool_source_sha256` and every product-source hash are exact **raw-byte** hashes, with
  exact-byte recomputation tests). RM2 (claim key carries `code_identity_sha256`) and RM3
  (non-gating `ORCHESTRATION_ONLY_NON_GATING` diagnostic) remain closed and are re-verified. The
  prevalidation bundle is **re-issued deterministically, non-final across a future commit**; the
  v1 artifact and invalidation remain byte-for-byte; **no PASS, no B2 authorization, no v2
  validation outcome** (§15).
- **rework-9 (R40-B1V one-shot: the v2 reserved cohort is evaluated exactly once), before any real
  R40 outcome:** the separate `ZM-ZEGRID-R40-B1V-ONE-SHOT` mission executed the reserved v2 cohort
  **exactly once** against the accepted freeze-closure bundle and returned a **terminal ACCEPT**
  (`B1_PASS`, 18/18, claim `COMPLETED`, `r40b2_authorized=true`) at HEAD `61c4aa5`; the v2 cohort is
  now **CONSUMED** and must **never** be re-invoked, and the v1 invalidation/artifact stay
  byte-for-byte (§15.7). **B2 implementation is thereby authorized; B2 acceptance is not.**
- **rework-10 (R40-B2-A capture-core, bounded task A), before any real R40-C outcome:** the
  default-OFF capture data contract + the bounded atomic shard-generation core are implemented and
  tested on **synthetic / in-memory fixtures only** (frozen schema/dtype, sample identity, hard disk
  budget, path safety, privacy, fail-closed identity, stale/resume, additive manifest link, sparse
  payload + deterministic parent join) with **no** production seam wired (§15.8). The shard core
  **does not accept B2**; the OFF/ON byte-identity gate and the 38-relation panel remain untested
  here and unchanged at tolerance zero. No criterion was changed by this revision — only stale
  threshold prose (`ulp_k = 4`, the recovered-Δ `1e-12` framing and an amplitude-promotion gate) was
  reconciled to the already-accepted frozen **v2** criteria (§15.5).

- **rework-11 (R40-B2-A capture-core rework-1, bounded task A), before any real R40-C outcome:**
  after Junior reproduced five defects against the r0 core (R1 HIGH fail-open disk budget; R2 MEDIUM
  un-verified COMMIT; R3 MEDIUM finite-input silent `inf`; R4 LOW `production_wired` semantics; R5 LOW
  generation-integrity gaps), the core is repaired: the exact final COMMIT/total is computed before any
  publish decision, the verifier rebuilds and checks the COMMIT + index identity + shard set, float/int
  representability is enforced on the encoded array, and `production_wired` is an explicit default-strict
  parameter (§15.8). Re-run: focused **128 passed**, combined regressions **417 passed**. The shard core
  still **does not accept B2**; no criterion was changed and the OFF/ON zero-tolerance gate was untouched.

- **rework-12 (R40-B2-B seam wiring, bounded task B), before any real R40-C outcome:** the accepted
  default-OFF capture core is wired into the PRODUCTION per-cell assembly seam
  (`_assemble_per_cell_feathered`, at the pinned pre-offset/pre-feather point, before
  `build_cell_offset_graph` / `assemble_canvas_feathered`) behind a strictly default-OFF opt-in
  (`r40_seam_capture`, strict `bool`), with the lazy-import guarantee kept (§15.9). The seam reuses the
  core (frozen dtype/identity/budget/privacy/path-safety/atomic COMMIT-last writer/additive manifest
  block) and duplicates no pixel/difference maths. Observed: focused **145 passed**
  (`r40_capture_core` + seam), combined regressions **417 passed**. **B2 acceptance is NOT claimed**: the
  exact OFF/ON FITS byte-identity gate, the 38-relation panel qualification, the injected-effect recovery
  re-proof, the real product ROI parity, the exact §1.3 held-out-key intersection, the real worker→parent
  transport on a pooled run and R40-C/D remain **`NOT_RUN`**. No criterion was changed and the OFF/ON
  zero-tolerance gate was untouched.

- **rework-13 (R40-B2-C real pooled transport, bounded task C), before any real R40-C outcome:** the REAL
  pooled worker→parent sparse transport is implemented behind the SAME default-OFF opt-in (§15.10). With the
  capture ON the parent computes the deterministic per-cell candidate geometry (reusing the sealed
  `_r40_candidate_coords`) and passes it to every pooling worker via the pool initargs; each worker
  (`_stack_cell`) returns ONLY the sparse candidate samples of its own cell(s) from the worker-local MiniTile
  planes (delegating every single-pixel read to the B2-B extractor — no duplicated pixel/difference maths,
  no full f64 patch plane, no raster/cutout, no FITS read). With the capture OFF the transport is completely
  absent: the worker initargs tuple stays the pre-R40 8-tuple, no `r40_sparse` key is added, the core is not
  imported and there is no retention or measurable behaviour change. Observed: focused **156 passed**
  (`r40b2_pooled` + seam + core), combined regressions **316 passed**; the payload bound is
  `O(candidate records)`. **B2 acceptance is NOT claimed**: the exact OFF/ON FITS byte identity, the
  38-relation panel qualification, the injected-effect recovery re-proof, the real product ROI parity, the
  exact §1.3 held-out-key intersection, a real Windows run and R40-C/D remain **`NOT_RUN`**. No criterion was
  changed and the OFF/ON zero-tolerance gate was untouched.

- **rework-14 (R40-B2-D OFF/ON product byte-identity measurement, bounded task D), before any
  real R40-C outcome:** the frozen §8.2 zero-tolerance gate is **measured** (never re-calibrated) on
  real synthetic end-to-end product runs (§15.11). The R40 capture ON (output discarded) produces
  **byte-identical** science/coverage/aesthetic/uint16 product FITS (full file bytes, tolerance
  exactly `0`) versus capture-OFF, with OFF shown deterministic and a controlled, environment-pinned
  run removing the scheduler confound; the manifest gains **exactly one** additive
  `r40_diagnostic_capture` block ON and none OFF, with every pre-existing run-stable field
  canonical-identical. **Zero production code was changed** (one new test file only). Observed:
  focused **172 passed** (new `r40b2_offon_identity` + pooled + seam + core), combined regressions
  **417 passed**. **B2 acceptance is NOT claimed**: the 38-relation panel qualification, the
  injected-effect recovery re-proof, the real product ROI parity, the exact §1.3 held-out-key
  intersection, a real Windows run and R40-C/D remain **`NOT_RUN`**. No criterion was changed and the
  OFF/ON zero-tolerance gate was left at exactly zero.

- **rework-15 (R40-B2-D rework-1: corrective iteration after an independent reproduction), before any real
  R40-C outcome:** Junior independently reproduced a **contradiction** of the r0 manifest narrative — in
  their environment the ON run chose `mode=inmem` while both OFF runs chose `mode=stream`, so the r0
  classifier was incomplete and the r0 claim over-stated (the §8.2 BYTE gate held regardless). Repairs:
  (R1) the plan/environment causality is now MEASURED and stated — the joint per-cell plan is a pure
  function of the free RAM sampled at plan time and is computed strictly BEFORE any capture work; with the
  memory seam pinned the plan is **identical** OFF vs ON at both a STREAM and an INMEM budget, and the
  memory input **alone** flips the plan with the capture ABSENT (so the wild flips are environment RAM
  fluctuation, **not** a capture effect). (R2) the run-volatile classifier is broadened to the full
  environment-driven scheduler plan and its completeness is **evidence-backed** against two capture-OFF
  runs with DIFFERENT plans; the strong canonical-identity claim is now scoped to the same-plan pinned
  pairs, and the uncontrolled-fixture claim is restated honestly (product bytes identical; every ON-vs-OFF
  difference confined to the classifier + the single additive block). (R3) a sensitivity negative control
  proves the byte gate detects a single-byte / single-ULP change (not vacuous). (R4) `.cells` is stated
  precisely (per-cell records; under a plan flip ONLY the plan-metadata leaves `mode`/`bound_bytes` differ,
  and only those two leaf NAMES are excluded — the subtree is never blanket-excluded). Observed, run twice
  for reproducibility: focused **20 passed** (twice), combined **176 passed** (twice), regressions **417
  passed**. **No production code was changed** and no criterion was altered. **B2 acceptance is NOT
  claimed**; the OFF/ON gate stays at exactly zero.

- **rework-16 (R40-B2-E contract-clarity amendment, bounded task E), before any real R40
  outcome (and before any real R40-C outcome):** section 8 clause 3 (the manifest scheme) is
  amended so the frozen text is **truthful and mechanically checkable**. This is a
  **wording/contract-only** change: **no code, no test expectation and NO tolerance** is
  touched. The literal "all pre-existing manifest fields canonical-identical ON vs OFF" claim
  was **unsatisfiable by construction**: the pre-existing manifest embeds inherently
  run-unstable observability fields (wall timing, RSS and the environment-driven scheduler
  plan) that differ between **any** two sequential runs — even OFF vs OFF — as measured in
  §15.11. It is replaced by a **scoped** statement: **(i)** the load-bearing product obligation
  is restated **unchanged and exact** (science/coverage/aesthetic arrays + FITS bytes, plus any
  other product array artifact emitted, exactly byte-identical ON vs OFF at **tolerance exactly
  `0`**, never calibrated/widened/relaxed/reinterpreted); **(ii)** exactly one additive
  `r40_diagnostic_capture` block ON and none OFF, with **no** whole-manifest byte-identity
  claim; **(iii)** every **plan-independent product/metadata** pre-existing field
  canonical-identical ON vs OFF; and **(iv)** the explicitly enumerated **run-volatile
  observability** fields (wall timing; RSS; environment-driven scheduler plan: per-cell `mode`,
  per-cell footprint bytes, concurrency/plan candidates, executor/worker counts, reprojection
  path/counters, RAM-budget and available-memory figures, and the per-cell `mode`/`bound_bytes`
  the planner stamps) excluded from the canonical projection, with completeness
  **evidence-backed by measurement** (two OFF runs and a forced plan flip) and a focused test
  that **fails loudly** on any unclassified differing leaf; **no product field may ever be
  excluded**. Cross-references (§0 rework-1; §14 rework-1; §15.8; §15.9; §15.11) are
  reconciled to the scoped wording. **No tolerance was changed** anywhere; the OFF/ON gate
  stays at exactly `0`. **B2 acceptance is NOT claimed**; `R40-C`/`R40-D` remain `NOT_RUN`.

Any further normative change before the first real R40-C outcome must be recorded directly
in the frozen sections above **and** logged here, with the explicit note that it happened
**before any real R40 outcome**. No real R40 capture, metric or outcome was computed for
this document.

---

## 15. R40-B1 outcome — v1 INVALIDATED, v2 prevalidation frozen (freeze-closure; pre-real)

**Mission** `ZM-ZEGRID-R40-B1-SYNTH-CALIB` · **phase** `freeze-closure` · **status**
**`B1_FAIL_PROTOCOL_INVALID`** / **`READY_FOR_NEW_VALIDATION`** · **branch** `beta` ·
**baseline HEAD** `50b30ebd7909a246160b59ff1111794b903c796c` (2 local commits ahead of
`origin/beta`, **not pushed**).

The first R40-B1 delivery (`B1_PASS`, `r40b2_authorized = true`) is **WITHDRAWN**. Nono
review-0 found 3 HIGH / 5 MEDIUM / 2 LOW; rework-1 published the v2 package; Nono review-1
found 2 HIGH / 4 MEDIUM / 2 LOW; rework-2 re-issued the package; Nono review-2 found
0 HIGH / 3 MEDIUM / 2 LOW (**RM1–RM3**), which **rework-3 closes**; Nono review-3 returned
0 HIGH / 1 MEDIUM / 2 LOW (**RM1 leak + L1/L2 metadata**), which the **freeze-closure**
package closes. Every review round was
code-level: the v2 reserved cohort has **never** been evaluated and is still fresh.
**The freeze-closure package does not evaluate the v2 cohort and does not authorize R40-B2.**

> **R40-B2 implementation is authorized by the terminal v2 accept (§15.7); nothing else here is.**
> The freeze-closure revision below authorized no B2 (it evaluated no v2 outcome). The separate
> `ZM-ZEGRID-R40-B1V-ONE-SHOT` mission has since executed the reserved v2 cohort exactly once and
> returned **ACCEPT** (`r40b2_implementation_authorized=true`). Nothing here (or there) authorizes
> C, D, Windows/Zealfie, LP341, any correction, or any product/GUI/config/default change. The
> OFF/ON byte-identity gate is preserved **verbatim** and was never calibrated, widened or
> reinterpreted.

> **Non-final artifact notice.** Prevalidation artifact bindings include the repository HEAD and
> the code identity. **Committing the current working tree moves HEAD**, so this artifact is
> **non-final across a future commit**: it must be **re-issued deterministically against the new
> HEAD** (and re-verified) after any local commit and before the one-shot validation mission is
> opened. It is published now for review only.

### 15.1 v1 invalidation (durable, unchanged)

| Fact | Value |
|---|---|
| invalidation record | `.a2a-reports/ZM-ZEGRID-R40-B1-SYNTH-CALIB/r40b1_v1_invalidation.json` |
| bytes / SHA-256 | `3 828` / `590b2072a1f9b365531650ca8ee40543facec773cdab7c9d9337eee2afe6457c` (**byte-identical, unchanged**) |
| v1 artifact (preserved, immutable, never overwritten) | `r40b1_synthetic_calibration.json`, `53 978` B, sha256 `7ec179a45a85a022f62c7f5aef3d4c8ff84cb496cbe239c9c11bab7d156e8290` (**byte-identical, unchanged**) |
| v1 claimed (withdrawn) | `final_status = B1_PASS`, `r40b2_authorized = true` |
| **v1 status now** | **`B1_FAIL_PROTOCOL_INVALID`**, `r40b2_authorized = false` |
| v1 validation cohort | **`CONSUMED_COMPROMISED`**, `rerun_forbidden = true` |

### 15.2 v2 prevalidation freeze (re-issued by freeze-closure)

| Frozen identity | Value |
|---|---|
| protocol SHA-256 (`r40b1-protocol-2`) | `d689b2d78cebebad0222f503f804a8d6c9a2bf60316c88067b1851e1a447ffa7` |
| criteria SHA-256 (`r40b1-criteria-2`) | `67be63e4423f782646f71bc350be5ff5a33a2007b67f0999a76a841154da92b9` |
| reserved validation cohort SHA-256 (`r40b1-cohort-2`) | `8e2ffe7026a5468546c23cc77b3ab3acb03fe62bb2383ab0316a60127b4a3743` |
| prevalidation bundle | `.a2a-reports/ZM-ZEGRID-R40-B1-SYNTH-CALIB/r40b1_v2_prevalidation.json` |
| bundle bytes / SHA-256 | `95 055` / `1ccc443e4821f8d6e31c100abf6d27ec24e1514c2f872c3c99578b6fa445a484` |
| bundle `content_sha256` | `cc1df83bb759ffeb687a2fac045a3a5c6821c77222997867e2405175dabe25f6` |
| bundle `calibration_sha256` | `79f22f34d6ee8e6ec107989aee7a830a4ffcd980ac8160f04e68a688f2aa60d2` |
| `code_identity_sha256` (RM2 claim-key digest) | `480f258729f08229a74550eedaa88d8c5fed41eb25f77a57c93f136dd993f0b7` |
| product source manifest (RM1) | **72** product modules (the complete `src/zemosaic/**.py` tree, raw-byte), manifest SHA-256 `83a4cca7ace88275b75ef149a9a6a103d3dd9c3a562c48fa3c4201d331e31107` |
| runtime versions (RM1) | python `3.13.5`, numpy `2.2.4`, scipy `1.15.3`, astropy `7.0.1`, reproject `0.21.0` |
| tool source SHA-256 / repo HEAD | `7218cc49f5552443331d93c577cffa607866f0ecc38e2948c61613b3ec0e4660` / `50b30ebd7909a246160b59ff1111794b903c796c` |
| validation status / runs / claim | **`NOT_RUN`** / **`0`** / `null` (reserved cohort never invoked) |

All declared hashes (protocol, criteria, cohort, calibration, content, code identity) are
**recomputable from the persisted artifact**, and the bundle is byte-deterministic across
regenerations.

### 15.3 Freeze-closure repairs (Nono review-3: RM1 + L1/L2; RM2/RM3 re-verified)

- **RM1 (MEDIUM) — the code identity now binds what validation actually invokes.**
  `code_identity` carries a `product_source_manifest` that is the **complete product source
  tree**: every `*.py` under `src/zemosaic/**`, repository-relative, deterministic sorted, with
  **raw-byte SHA-256** (`read_bytes`, no text decode) and a per-entry rationale. It also carries
  the **numeric runtime dependency versions** (Python/numpy/scipy/astropy/reproject), the manifest
  **count** and the manifest **policy** string. Whole-tree binding is deliberate **over-binding**
  of the import closure: the earlier hand-maintained **11-module allowlist was proven incomplete**
  (the arms' real closure is larger and eagerly imports the zegrid package `__init__`, which pulls
  in `executor.py`, `photometric.py`, `science_adapter.py`, `canonical_streaming.py`, …), and an
  unbound eagerly imported module could silently change validated behaviour. Over-binding can only
  *refuse* (fail closed) and force a re-issue; under-binding would falsely accept a stale package.
  Binding is **fail-closed**: an absent product root/empty tree/missing bound file, an unreadable
  version or a failed HEAD probe refuses (no `unavailable` identity is accepted for a
  validation-ready package), and `preflight_bundle` **recomputes the whole identity and refuses on
  any mismatch before claim** — so a dirty edit, an addition or a removal anywhere in the bound
  product source set with an unchanged git HEAD refuses. A closure audit
  (`audit_loaded_product_closure`) forces the I1a/I1b/I2/canonical-stack/calibration entrypoints and
  proves **zero** loaded local product module is unbound (currently **24** loaded ⊆ **72** bound).
- **RM2 (MEDIUM) — the durable claim key now includes the code identity.**
  `claim_key = {protocol_sha256, criteria_sha256, cohort_sha256, code_identity_sha256}` with a
  dedicated domain tag; the result records the identical digest and the claim key. Two frozen
  packages with different code identities (tool, manifest, versions or HEAD) cannot share a
  `claim_id`. The expected-decision contract remains transitively bound via the frozen/`hashed`
  cohort specs and the deterministic recomputation in `preflight_bundle` (no duplicate authority).
- **RM3 (MEDIUM) — the recovered/common-support delta is explicitly non-gating.**
  The production crop path and the full-source geometry carry the **identical support loss on
  their common support**, so their agreement is **structurally expected** and cannot establish
  recovery, causality or product parity. It is now labelled
  **`ORCHESTRATION_ONLY_NON_GATING`** with `scientific_gate = false`,
  `authorizes_b2 = false`, `consumed_by_h4_admission = false` and explanatory prose; the
  misleading `BIT_IDENTICAL`/tolerance/rule fields are removed from the criteria, and
  **H4 admission does not consume it**. The meaningful I1b gate remains
  support-subset + reduction (`≥ 0.20`) + **independent** relation construction, bounded to
  synthetic instrument calibration with product ROI parity `B2_MANDATORY`.

### 15.4 v2 calibration matrix (calibration groups only; reserved cohort never evaluated)

| Arm | Calibration outcome |
|---|---|
| M3 identity, adversarial (unit, signed zero, subnormal underflow, cancelling opposite offsets, large opposite-sign, non-finite) | bit-identical helper output on every evaluated case; f64; **2 non-finite refusals** |
| M3 algebraic cross-expression | max scale-ratio `0.6338271846408026`; `ulp_k = 5` from an explicit IEEE rounding count `(2+3)` |
| ROI pointwise parity (`ORCHESTRATION_ONLY`) / ROI global recompute | bit-identical, tolerance `0` / `NOT_IDENTIFIABLE` |
| I1a crop vs full-source common support | `PARITY_BIT_IDENTICAL` |
| I2 same-array fast vs reference (TAN) / non-TAN | `PARITY_BIT_IDENTICAL` / `NOT_APPLICABLE` |
| **I1b** coverage delta through the real canonical stack, **TAN and non-TAN**, 2 independent patches | `RECOVERED` (both), `2 h + 2 v` each, min reduction `0.616555166821928` |
| Recovered/common-support delta | **diagnostic only** (`ORCHESTRATION_ONLY_NON_GATING`, non-gating, 8 relations) |
| Relation-first panel (correlated RGB, 2-of-3, hard override) | isolated: reduction `0.506`, 4 promoted (2 h + 2 v); pseudo-single-relation: 1 relation; contradictory channel: 0 promoted (hard block); channel duplication: same count as isolated |

### 15.5 Frozen v2 criteria (normative)

1. **M3 pre/post identity** — `BIT_IDENTICAL`, tolerance **exactly `0`**, **f64** helper result,
   **no f32 downcast**; non-finite ⇒ **refuse**. Scope: M3 only.
2. **M3 algebraic cross-expression** — `SCALE_AWARE_ULP`, **`ulp_k = 5`** from an explicit IEEE
   rounding count — **not a fabricated floor**. Scope: cross-expression only. **Neither rule
   applies to OFF/ON arrays/FITS.**
3. **ROI-shadow parity** — pointwise/local frozen inputs = **BIT_IDENTICAL, tolerance `0`**; any
   ROI-recomputed Cell-global quantity = **`NOT_IDENTIFIABLE`**. `ORCHESTRATION_ONLY`: **product
   ROI parity remains `B2_MANDATORY`**.
4. **Reprojection intervention** — I1a parity `tolerance 0`; I2 same-array parity `tolerance 0`;
   **I1b coverage delta through the real canonical stack with `min reduction ≥ 0.20` on TAN and
   non-TAN, requiring independent relations**; only I2's fast path may be `NOT_APPLICABLE`.
5. **Injected-effect recovery** — relation-first, per-channel state then explicit **2-of-3**
   relation rule; inherited `reduction_rms ≥ 0.20`, `p90` worsening `≤ 5%`,
   `RMS0 ≤ 1.25·σ_ref`, `≥ 2` distinct relations per orientation. The recovered/common-support
   delta is **`ORCHESTRATION_ONLY_NON_GATING`** and gates nothing; synthetic injection amplitude
   remains generator metadata only.
6. **H4 `CONTROLLED_EFFECT` admission** — identity + parity gates, complete 38-relation panel,
   same global pixels / same fixed stack, exactly one isolated family, `≥ 2` relations per
   orientation with a consistent derived signature, `d_post_prod` vs `d_post_shadow` after the
   **same frozen production offsets**. Hard refusals: confounded ⇒ `NOT_IDENTIFIABLE`;
   incomplete ⇒ `INCOMPLETE_PANEL`; parity/identity failure, RGB pseudo-replication,
   per-frame-only, sign-only ⇒ `NOT_IDENTIFIABLE`. Precedence: `NOT_IDENTIFIABLE` >
   `INCOMPLETE_PANEL` > `NO_EFFECT` > `DETECTED_ASSOCIATION` > `CONTROLLED_EFFECT`.
   `authorizes_b2 = false`, `recovered_delta_consumed = false`.
7. **Expected-outcome contract** — every reserved group carries a **frozen expected decision**;
   panel groups are decided by the H4 ladder with the bound snapshots; **no default pass**.
8. **OFF/ON identity — UNCHANGED (verbatim).** Science/coverage/aesthetic arrays and the
   science/coverage/aesthetic FITS bytes remain **exact byte identity, tolerance exactly `0`**
   (§8.2). R40-B1 did **not** calibrate, widen, relax or reinterpret it.

### 15.6 Limitations, status and next gate

- Synthetic qualification validates the instrument, the frozen rules and the real primitive arms.
  It is **not** a Windows816 causal claim and no R39 observed value was used as a target.
- The I1b arm uses bounded synthetic patches as a two-cell proxy for the relation geometry; the
  product ROI path remains **`B2_MANDATORY`** and unproven until B2.
- The v2 reserved cohort was **fresh and unconsumed** (`generator_invoked=false`,
  `outcomes_present=false`, `validation_runs=0`, no claim/completed marker anywhere) **at the
  freeze-closure revision**. It has since been evaluated **exactly once** by `ZM-ZEGRID-R40-B1V-ONE-SHOT`
  → terminal **ACCEPT** (§15.7). The cohort is now **CONSUMED / `COMPLETED`** and **must never be
  re-invoked**; its bundle, result and markers are immutable.
- **R40-C/R40-D remain `NOT_RUN` / `NOT_AUTHORIZED`** and still require the HUMAN_GATE.
- **Next safe sequence:** the freeze-closure tool/tests/doc were accepted and committed; the
  prevalidation bundle was re-issued against the new HEAD; the separate one-shot validation mission
  executed the reserved v2 cohort exactly once → **terminal ACCEPT** (§15.7). **R40-B2
  implementation is authorized** and its bounded task A is **DONE** (§15.8); **B2 acceptance still
  requires** the production seam wiring + the exact OFF/ON byte identity + the injected-effect
  recovery re-proof + the real product ROI parity.

### 15.7 R40-B1 v2 one-shot validation — terminal ACCEPT (cohort CONSUMED)

**Mission** `ZM-ZEGRID-R40-B1V-ONE-SHOT` · **phase** `review-0` · **status**
**`B1_VALIDATION_ACCEPTED`** · **`r40b2_implementation_authorized = true`** · **branch** `beta` ·
**HEAD** `61c4aa50f18881ad624b0ecf55df9b049eb56ecb` (4 local commits ahead of `origin/beta`, **not pushed**).

The v2 reserved cohort was executed **exactly once** against the accepted freeze-closure bundle
(`r40b1_v2_prevalidation.json`, unchanged) and returned **18/18 `observed == expected`**, 0 failures.
The independent review (`ZM-ZEGRID-R40-B1V-ONE-SHOT.nono.r0.md`) returned **ACCEPT — 0 HIGH, 0 MEDIUM,
0 blocking LOW**. The cohort is **CONSUMED / `COMPLETED`** and is **never** re-invoked; its bundle,
result, claim/completed markers and state directory are **immutable**.

| Frozen terminal identity | Value |
|---|---|
| result artifact | `.a2a-reports/ZM-ZEGRID-R40-B1-SYNTH-CALIB/r40b1_v2_validation_result.json` |
| result bytes / SHA-256 | `40 965` / `d597cd1d67b0cd91593e1da777ed2dde4ff0105864cecd63fbfe259d327ab474` |
| result `content_sha256` | `4d206188788d42605fca7bd235680992ff794e1378099de6425ef6728c4cf3df` |
| `b1_status` / `status` / `runs` | `B1_PASS` / `VALIDATED` / `1` |
| claim id / state | `c965187cd826ade68c556a59e1f9373d5a4dc500` / `COMPLETED` (result sha = file sha) |
| protocol SHA-256 | `d689b2d78cebebad0222f503f804a8d6c9a2bf60316c88067b1851e1a447ffa7` |
| criteria SHA-256 | `67be63e4423f782646f71bc350be5ff5a33a2007b67f0999a76a841154da92b9` |
| cohort SHA-256 | `8e2ffe7026a5468546c23cc77b3ab3acb03fe62bb2383ab0316a60127b4a3743` |
| `code_identity_sha256` | `10416290fd4d699d92a94c9be518283d5f97b26e33a14aaa36673d682063ec64` |
| `baseline_head` / `repo_head` | `50b30ebd7909a246160b59ff1111794b903c796c` / `61c4aa50f18881ad624b0ecf55df9b049eb56ecb` |
| `admission_router` | `h4_admit with the bound frozen snapshots` |

**Limits carried forward (unchanged).** This is a **synthetic-only** validation of the instrument and
the frozen rules — it is **not** evidence of any Windows816 mechanism, product or product causality.
**Product ROI parity** and the **exact OFF/ON science/coverage/aesthetic array + FITS byte-identity
gate** remain **`B2_MANDATORY`** and are **not** proven by it.

### 15.8 R40-B2-A capture-core — implementation status (bounded task A)

**Mission** `ZM-ZEGRID-R40-B2-A-SHARD-CORE` · **phase** `implementation` · **branch** `beta` · **HEAD**
`61c4aa50f18881ad624b0ecf55df9b049eb56ecb` (unchanged) · **commit created** `NO` · **push** `NO`.

Bounded task A implements the reusable **default-OFF** capture data contract and the **bounded atomic
shard-generation core** on **synthetic / in-memory fixtures only**, without wiring the production seam.

| Added (untracked, no commit) | Purpose |
|---|---|
| `src/zemosaic/core/zegrid/r40_capture.py` | capture core (§3/§6/§8 mechanics; not imported by the package init) |
| `tests/test_zegrid_r40_capture_core.py` | focused synthetic tests |

**Implemented contract coverage (task-A items 1–11).** (1) explicit default-OFF config with strict-bool
validation and zero-I/O construction/import; (2) versioned little-endian fixed-width C-order record
dtype + relation headers (M3 pre `f32`/post `f64`, M6 `int64`, coordinates `int32`, bool validity, `f64`
support/weights/ramps), no pickle/object dtype; (3) the frozen §3 sample key and domain-tagged pair-set /
candidate-set digests; (4) atomic bounded generation writer (fresh generation dir, one shard per
relation, strict-JSON index, per-shard count/bytes/SHA-256, per-shard fsync, **COMMIT last**, dir fsync);
(5) conservative hard disk budget checked before **and** during writes, refusal leaves no COMMIT, partial
cleaned or quarantined, committed generations never overwritten; (6) fail-closed path safety (traversal,
symlinked components/targets, protected roots, non-fresh destination) and a non-mutating, prose-aware
privacy boundary that fires **before** COMMIT; (7) strict finite JSON, NaN/Inf refused (never coerced);
(8) fail-closed identity checker over every §6.2/§8.7 field incl. the terminal B1 v2 protocol/criteria/
result/claim bindings; (9) stale-invalidation / exact-resume primitive (checkpoint between completed
relation shards only; mismatch ⇒ `STALE_INVALIDATED`; no final COMMIT while partial); (10) additive
`r40_diagnostic_capture` manifest-link builder proving pre-existing fields canonical-identical
ON/OFF **for plan-independent product/metadata** (run-volatile observability excluded — scoped by
rework-16, §8 clause 3; never a whole-manifest byte-identity claim); (11) validated sparse worker-payload structures and
pure deterministic parent-join helpers (no full E7–E9 plane retention) — **`_assemble_per_cell_feathered`
is NOT wired**.

**rework-1 (corrective iteration 1, `ZM-ZEGRID-R40-B2-A-SHARD-CORE`).** Junior reproduced five
defects against the r0 core (synthetic fixtures only); all are fixed and covered by focused tests:

- **R1 (HIGH) — disk budget was fail-open.** The final `check_disk_budget` used the *pre-final* COMMIT
  bytes while the COMMIT payload was then mutated (adding `index_sha256`), so a published generation
  could exceed `disk_budget_bytes` yet return `COMMITTED`. **Fixed:** the FINAL COMMIT bytes (with the
  bound index digest) and the exact final published total are computed **before any publish/layout
  decision**; the exact total is refused before anything is written, the mid-write checks are exact, and
  the writer asserts the published total ≤ budget. Observed before: budget `6320` → published `6406`,
  `COMMITTED` (exceeded). After: exact-fit publishes, one byte under refuses with **no COMMIT**.
- **R2 (MEDIUM) — `verify_generation` did not verify COMMIT contents.** A tampered COMMIT verified
  `ok=True`. **Fixed:** the verifier rebuilds the expected COMMIT from the index and refuses any
  mismatch of `schema`/`schema_version`/`mission`/`capture_id`/`capture_stage`/`identity_digest`/
  `n_shards`/`n_records`/`shard_bytes`/`index_sha256`, and rejects unknown/extra COMMIT fields.
- **R3 (MEDIUM) — a finite input could be stored non-finite.** `<f4`/`<f8` inputs were validated as
  finite Python floats, so a finite out-of-range double (e.g. `sci_u_pre=1e300`) was silently downcast to
  `inf`, and an int out of int64 range raised a raw `OverflowError`. **Fixed:** every float field is
  refused (typed) when it cannot be represented finitely in the frozen dtype; int fields are range-checked
  (typed refusal, no silent wrap/narrow); the ENCODED structured array is re-checked with
  `assert_finite_array` (extended to structured dtypes) before returning.
- **R4 (LOW) — `production_wired` semantics.** `check_capture_identity` hard-required `False`. **Fixed
  without weakening:** the expected value is now an explicit, documented, default-strict parameter
  `expected_production_wired=EXPECTED_PRODUCTION_WIRED (False)`, threaded through the digest, writer,
  checkpoint/resume and verifier. The default-OFF core stays strict; the later wired seam must pass
  `True` explicitly (nothing is wired here).
- **R5 (LOW) — generation-level integrity gaps.** **Fixed:** the verifier re-checks the index's own
  identity digest (`sha256_payload(identity, IDENTITY_TAG)`), the capture-id/production-wired consistency,
  the shard set (`n_shards`/`n_records`/`shard_bytes` vs the listed entries), and flags any file present
  in the committed generation dir that the index does not list.

**Validation (executed, observed; rework-1 re-run).**

| Command | Result |
|---|---|
| `python3 -m pytest tests/test_zegrid_r40_capture_core.py -W error -q -p no:cacheprovider` | **128 passed** |
| `python3 -m pytest tests/test_zegrid_r37d_diagnostic_capture.py tests/test_zegrid_r37e_capture_runner.py tests/test_zegrid_r38a_causal_probe.py tests/test_zegrid_r39a_spatial_residual.py tests/test_zegrid_r40b1_synthetic_calibration.py -W error -q -p no:cacheprovider` | **417 passed** |
| `python3 -m py_compile src/zemosaic/core/zegrid/r40_capture.py tests/test_zegrid_r40_capture_core.py` | OK |
| `git diff --check` | clean |

*(The r0 run reported the same five regression files as two groups; a first combined re-run showed
4 failed / 47 errors, which did **not** involve the unreferenced capture core and did **not** recur on a
clean re-run — the final observed combined result is **417 passed**.)*

> **The shard core alone does NOT accept B2.** Production seam wiring (sparse worker→parent extraction),
the **exact OFF/ON science/coverage/aesthetic array + FITS byte identity (tolerance exactly `0`)** and the
38-relation panel OFF/ON qualification, the injected-effect recovery re-proof and the **real product ROI
parity** are all **`NOT_RUN`** in task A and remain **`B2_MANDATORY`**. The OFF/ON gate is untouched at
tolerance zero. `R40-C` / `R40-D` / `HUMAN_GATE` / `LP341` remain **`NOT_RUN` / `NOT_AUTHORIZED`**.

### 15.9 R40-B2-B production seam wiring — implementation status (bounded task B)

**Mission** `ZM-ZEGRID-R40-B2-B-SEAM-WIRING` · **phase** `implementation` · **branch** `beta` · **baseline
HEAD** `b23fbdc3a8ef6709f1891d355a4e083e62dbd5f6` · **commit created** `NO` · **push** `NO`.

Bounded task B wires the accepted default-OFF capture core into the PRODUCTION per-cell assembly seam
at the pinned pre-offset/pre-feather point (inside `_assemble_per_cell_feathered`, after the complete
`CellPatch` objects coexist and BEFORE `build_cell_offset_graph` / `assemble_canvas_feathered`) behind a
strictly default-OFF opt-in.

| Added / changed (no commit) | Purpose |
|---|---|
| `src/zemosaic/zemosaic_zegrid_mode.py` (modified) | seam opt-in (`r40_seam_capture`, strict `bool`, default `False`), threaded `run_zegrid_mode` → `_run_single` → `_assemble_cells` → `_assemble_per_cell_feathered`; the pinned seam capture (reuses `core/zegrid/r40_capture`); the additive `r40_diagnostic_capture` manifest block; picklable worker→parent sparse extractor |
| `tests/test_zegrid_r40b2_seam.py` (new) | focused seam tests incl. the four mandatory proofs (a)–(d) |

**Implemented (bounded task B).** (1) strictly default-OFF opt-in with strict-`bool` refusal and a required
non-empty output root when enabled; importing the module and running with the capture OFF does **not**
import the core and changes nothing; (2) at the seam the capture reads ONLY the sparse per-relation
candidate pixels from the in-memory planes (no full f64 patch plane, no raster/cutout, no
final-product FITS read) and reuses the production `shared_valid_mask` / `apply_additive_offsets` /
`feather_weight_field` primitives — no duplicated pixel/difference maths; (3) the R37-D capture path is
semantically unchanged and the two captures are independent; (4) fail-closed identity / mandatory-observable
/ budget / privacy / path / write refusals publish NO artifact and never alter the assembly result;
(5) exactly ONE additive `r40_diagnostic_capture` manifest block cross-linked on the frozen identity keys
(`capture_id`, `filename`, `sha256`, `bytes`), all plan-independent product/metadata pre-existing manifest
fields canonical-identical ON/OFF (run-volatile observability excluded — scoped by rework-16, §8 clause 3),
and **no** new field when OFF (never a whole-manifest byte-identity claim); (6) bounded RAM/disk through the
accepted core.

**Mandatory proofs — measured on synthetic in-memory fixtures (never asserted).**

| Proof | Result |
|---|---|
| (a) capture OFF = an INDEPENDENT baseline rebuilt from the frozen primitives (`build_cell_offset_graph` + `assemble_canvas_feathered`) | **MEASURED — byte-identical** |
| (b) capture ON, output discarded → science/coverage/aesthetic-equivalent arrays byte-identical to OFF (tolerance exactly `0`); only the additive manifest block differs | **MEASURED — byte-identical** |
| (c) worker→parent sparse payload + deterministic parent join == in-process reference at the selected coordinates, transported through the `spawn` context | **MEASURED — Linux-side spawn parity ONLY (NOT a Windows run)** |
| (d) injected identity / budget / privacy / path / write failure → assembly result byte-identical to OFF and **no** artifact published | **MEASURED** |

**Validation (executed, observed).**

| Command | Result |
|---|---|
| `python3 -m pytest tests/test_zegrid_r40_capture_core.py tests/test_zegrid_r40b2_seam.py -W error -q -p no:cacheprovider` | **145 passed** |
| `python3 -m pytest tests/test_zegrid_r37d_diagnostic_capture.py tests/test_zegrid_r37e_capture_runner.py tests/test_zegrid_r38a_causal_probe.py tests/test_zegrid_r39a_spatial_residual.py tests/test_zegrid_r40b1_synthetic_calibration.py -W error -q -p no:cacheprovider` | **417 passed** |
| `python3 -m py_compile src/zemosaic/zemosaic_zegrid_mode.py tests/test_zegrid_r40b2_seam.py` | OK |
| `git diff --check` | clean |

**Known limits carried forward (architect-visible, NOT_RUN).**

- **M5/M7 sourcing.** At this seam the parent cannot see the worker-local MiniTile covariate planes
  (`estimator_weight_sum`, `support_w1`/`support_w2`/`n_eff_support`; plan §2.2). The seam therefore takes
  them from the picklable worker→parent transport and **refuses fail-closed** when it is absent. The
  **real pooled-run** transport was **`NOT_RUN`** at task B (**superseded: implemented by
  `ZM-ZEGRID-R40-B2-C`, §15.10**).
- **Exact §1.3 held-out intersection.** The captured sample set is the deterministic block-parity
  **candidate geometry** in the shared halo band, filtered by the frozen `shared_valid_mask`; the exact
  R37 held-out-key inverse/parity intersection is **`NOT_RUN`**.
- **Frozen R34c offsets.** The pinned capture point precedes the assembly's own graph solve, so the seam
  obtains the frozen offsets by calling the SAME production solver (pure/read-only/deterministic) before
  the assembly's unchanged solve. This is a declared bounded decision (an extra deterministic solve when
  the capture is ON); it does not alter the assembly result.

> **Task B does NOT accept B2.** The exact OFF/ON science/coverage/aesthetic array + FITS byte identity
(tolerance exactly `0`), the 38-relation panel OFF/ON qualification, the injected-effect recovery re-proof,
the **real product ROI parity**, the real pooled worker→parent transport and the exact §1.3 held-out
intersection remain **`NOT_RUN`** / **`B2_MANDATORY`**. The OFF/ON gate is untouched at tolerance zero.
`R40-C` / `R40-D` / `HUMAN_GATE` / `LP341` remain **`NOT_RUN` / `NOT_AUTHORIZED`**.

### 15.10 R40-B2-C real pooled worker→parent sparse transport — implementation status (bounded task C)

**Mission** `ZM-ZEGRID-R40-B2-C-POOLED-TRANSPORT` · **phase** `implementation` · **branch** `beta` ·
**baseline HEAD** `2f3ee778177195b7a55f0115b4f4c4620b772dab` · **commit created** `NO` · **push** `NO`.

Bounded task C closes the honest gap flagged in B2-B: the REAL pooled worker→parent sparse transport.
Before this task the seam consumed `result_by_cid[cid]['r40_sparse']`, a key only synthetic fixtures
provided; a real pooled run carried nothing. Now the production pooling worker (`_stack_cell`, run by
`zpar.pmap`) computes and returns ONLY the sparse per-relation candidate samples of its OWN cell(s) from
the worker-local MiniTile planes.

| Added / changed (no commit) | Purpose |
|---|---|
| `src/zemosaic/zemosaic_zegrid_mode.py` (modified) | parent-side per-cell candidate geometry (`_r40_worker_candidate_coords`, reusing the sealed `_r40_candidate_coords`); picklable worker entry point `_r40_worker_transport_sparse` (delegates every single-pixel read to the B2-B extractor); `_r40_maybe_worker_sparse` (armed only when the parent supplied coords); transport armed through `_init_stack_worker(..., r40_worker_coords=None)` and the `_run_cell_batch` initargs (appended ONLY when the capture is ON) |
| `tests/test_zegrid_r40b2_pooled.py` (new) | focused transport tests incl. proofs (e)–(g) |

**Implemented (bounded task C).** (1) With the capture ON the parent computes the deterministic per-cell
candidate geometry **before** any worker starts (plan §3.1.1) and hands it to every worker through the
pool initargs; each worker returns only the sparse candidates of its own cell(s) for M5
(`estimator_weight_sum`), M7 (`support_w1`/`support_w2`/`n_eff_support`), M6
(`surviving_sample_count`), M4 (`valid_mask`) and M3-pre (`sci_pre`) — never a full f64 patch plane, never
a raster/cutout, never a FITS read; `_r40_worker_extract_covariates` and the sealed candidate geometry are
reused (no duplicated pixel/difference maths). (2) With the capture OFF the transport is COMPLETELY absent:
the worker initargs tuple stays the pre-R40 8-tuple (no extra IPC payload), no `r40_sparse` key is added to
any worker result, the capture core is never imported and there is no retention or measurable behaviour
change. (3) Fail-closed: capture ON but the transport missing/incomplete/wrongly shaped → the seam refuses
and publishes NOTHING, leaving the input planes byte-unchanged. (4) Bounded: the transported payload is
`O(candidate records)` (`len(coords) × channels`), never `O(full patch area)` and never `N_frames × canvas`.
(5) The R37-D path and the assembly result semantics are unchanged; the capture is default-OFF.

**Mandatory proofs — measured on synthetic fixtures (never asserted).**

| Proof | Result |
|---|---|
| (e1) REAL production pooled run (`zpar.pmap` real process pool, `executor=process`, `workers=3`) on a synthetic 2×2 corpus with the capture ON → every complete cell returns its own `r40_sparse` payload; the published generation verifies (`verify_generation(..., expected_production_wired=True)` → `ok=True`, `problems=[]`) and is cross-linked in the manifest | **MEASURED** |
| (e2) worker→parent payload == the in-process reference extraction (`_r40_worker_extract_covariates`) at the same coordinates, transported through a genuine `multiprocessing.get_context('spawn')` pool, then driving a verifying seam capture | **MEASURED — Linux-side spawn parity ONLY (NOT a Windows run)** |
| (f) the same REAL pooled run with the capture OFF → NO `r40_sparse` key on any worker result, the worker result key set exactly the pre-R40 set, the worker initargs tuple unchanged (8), NO artifact and NO manifest field, OFF science byte-identical across two identical runs and the OFF assembly byte-identical to the INDEPENDENT frozen-primitive baseline | **MEASURED** |
| (g) fault injection (transport dropped / truncated / an endpoint's samples lost, plus a worker that emits no transport at all) → refuse, NO artifact and input planes byte-unchanged | **MEASURED** |

**Validation (executed, observed).**

| Command | Result |
|---|---|
| `python3 -m pytest tests/test_zegrid_r40b2_pooled.py tests/test_zegrid_r40b2_seam.py tests/test_zegrid_r40_capture_core.py -W error -q -p no:cacheprovider` | **156 passed** |
| `python3 -m pytest tests/test_zegrid_r37d_diagnostic_capture.py tests/test_zegrid_r37e_capture_runner.py tests/test_zegrid_r38a_causal_probe.py tests/test_zegrid_r39a_spatial_residual.py tests/test_zegrid_r40b1_synthetic_calibration.py -W error -q -p no:cacheprovider` | **417 passed** (the earlier 316 count was the 4-file subset; corrected after independent re-run) |
| `python3 -m py_compile src/zemosaic/zemosaic_zegrid_mode.py tests/test_zegrid_r40b2_pooled.py` | OK |
| `git diff --check` | clean |

**Known limits carried forward (architect-visible, NOT_RUN).**

- **Windows spawn.** The pooled production run uses the production `zpar.pmap` executor (fork on Linux); the
  explicit `spawn` pool parity is proven on the transport entry point only. A real **Windows** run is
  **`NOT_RUN`** (Windows is not exercised).
- **Exact §1.3 held-out intersection.** The captured sample set is the deterministic block-parity
  **candidate geometry** in the shared halo band, filtered by the frozen `shared_valid_mask`; the exact R37
  held-out-key inverse/parity intersection is **`NOT_RUN`**.
- **Payload bound is per-run measured, not a global claim.** No claim is made that the whole product RAM is
  `O(records)`; the pre-existing full `science`/`valid`/`surviving` patch IPC is acknowledged product
  behaviour and is unchanged.

> **Task C does NOT accept B2.** The exact OFF/ON science/coverage/aesthetic array + FITS byte identity
(tolerance exactly `0`), the 38-relation panel OFF/ON qualification, the injected-effect recovery re-proof,
the **real product ROI parity** and the exact §1.3 held-out intersection remain **`NOT_RUN`** /
**`B2_MANDATORY`**. The OFF/ON gate is untouched at tolerance zero. `R40-C` / `R40-D` / `HUMAN_GATE` /
`LP341` remain **`NOT_RUN` / `NOT_AUTHORIZED`**.

---

### 15.11 R40-B2-D OFF/ON product byte-identity measurement — implementation status (bounded task D)

**Mission** `ZM-ZEGRID-R40-B2-D-OFFON-IDENTITY` · **phase** `implementation` · **branch** `beta` ·
**baseline HEAD** `5edb9d19aacded082f7b53129e9fc2df6d78f32a` · **commit created** `NO` · **push** `NO`.

Bounded task D MEASURES — it does not calibrate or relax — the frozen §8.2 zero-tolerance gate on
**real synthetic end-to-end product runs**. **Zero production code was changed**; one new focused test
file (`tests/test_zegrid_r40b2_offon_identity.py`) was added. The pipeline is fed by a synthetic 2×2
disjoint corpus synthesized in `tmp_path` only (**never** the real corpus), with the *full* finishing set
enabled so all four product FITS are emitted; the accepted B2-C concurrency seam is pinned so the real
production **process pool** runs with the capture ON.

**Measured — product FITS byte identity (tolerance exactly `0`; full file bytes, not headers only).**
Representative durable evidence run (identical under the test harness): OFF run 1, OFF run 2 and ON run
all produced the **same size and SHA-256** per artifact.

| Product FITS | bytes | SHA-256 (identical OFF1 = OFF2 = ON) | pass |
|---|---|---|---|
| `mosaic_grid_science.fits` (SCI) | 12170880 | `f66b3a2738ff5489e401f7520dbd305481e52f04d09ffbfe9f88298b5840e4f5` | **PASS** |
| `mosaic_grid_aesthetic.fits` (AESTH) | 12170880 | `423f71f3803a2a8ed48f7303e093fed99165b80b5896458455bc4c90c669efe0` | **PASS** |
| `mosaic_grid_coverage.fits` | 4060800 | `69f0356eea1a0941b12a37b3e0c2208a9d02d82015a348bd41d376544ad60cbb` | **PASS** |
| `mosaic_grid_uint16.fits` | 6088320 | `47744bbd1a22eb125756571fe386c945b2f5b03dc17be5c88098e5e0c3dead21` | **PASS** |
| in-array equality (`fits[0].data`) | — | identical per artifact | **PASS** |
| embedded `SCIHASH`/`SCIROLE` header | — | identical OFF vs ON | **PASS** |

The two non-product durable outputs are explicitly **not** byte-compared: `zegrid_run.log` embeds
wall-clock timestamps by construction, and `__zegrid_cache__` is a transient retention-cleaned
intermediate.

**Measured — manifest (NO whole-manifest byte-identity claim).** The ON manifest carries **exactly one**
additive top-level block and the OFF manifest none; the block's four frozen identity keys cross-link the
published generation to the manifest exactly (`sha256`/`bytes` == the on-disk `index.json`):
`capture_id`/`filename` `63f5472962ac9d3a2ea65f6cf7ee7f47bd62632d507242eec64d33094d7682e8`,
`sha256` `c19990ceeaf0e902aafedbc9b31a5d7e525d1a0acf2c96592fb07cc4491789e6`, `bytes` `7293`.
`verify_generation(..., expected_production_wired=True)` → `ok=True`, `problems=[]`, `n_shards=2`,
`n_records=15432`, `checked_bytes=3009240`, with `index_sha256` self-consistent.

**Measured — CAUSALITY of the plan: environment-driven, NOT capture-driven (R40-B2-D rework-1).**
The joint per-cell plan (`mode`, `per_cell_footprint_bytes`, `plan_candidates`, executor, worker counts)
is a pure function of the free RAM sampled at plan time (`budget = 0.8 × available`), CPU count and cell
bounds. Three independent measurements establish the causality:

1. **Structural.** Within a run, the plan is computed strictly BEFORE any R40 capture work exists — the
   capture candidate geometry is computed only after the plan, and every capture write happens at
   assembly (after the per-cell stack). The capture therefore cannot influence the plan of its own run.
2. **Held-fixed environment → identical plan.** With the memory seam pinned to a constant, capture-ON and
   capture-OFF produce the **byte-identical plan** at BOTH a pinned STREAM budget and a pinned INMEM
   budget (asserted directly).
3. **Memory input ALONE flips the plan, capture ABSENT.** Two capture-OFF runs whose pinned memory
differs produce DIFFERENT plans (`stream` vs `inmem`); an interleaved OFF/ON/OFF/ON/OFF/ON sequence
showed the plan-time `available_bytes` drifting ~±10% run-to-run with the capture-ON runs bracketed by
capture-OFF runs and **no systematic ON-vs-OFF offset**. The plan flips observed in the wild are
environment RAM fluctuation, **not** a capture effect.

**Measured — manifest (NO whole-manifest byte-identity claim).** The ON manifest carries **exactly one**
additive top-level block and the OFF manifest none; the block's four frozen identity keys cross-link the
published generation to the manifest exactly (`sha256`/`bytes` == the on-disk `index.json`):
`capture_id`/`filename` `63f5472962ac9d3a2ea65f6cf7ee7f47bd62632d507242eec64d33094d7682e8`,
`sha256` `c19990ceeaf0e902aafedbc9b31a5d7e525d1a0acf2c96592fb07cc4491789e6`, `bytes` `7293`.
`verify_generation(..., expected_production_wired=True)` → `ok=True`, `problems=[]`, `n_shards=2`,
`n_records=15432`, `checked_bytes=3009240`, with `index_sha256` self-consistent.

**Measured — the manifest confound is ISOLATED by an evidence-backed classifier, never widened.** The
manifest embeds PRE-EXISTING *run-observability* fields that are not product science, in two families:
(a) wall-timing / RSS measurements and (b) the environment-driven scheduler plan (mode, footprint,
concurrency, plan candidates, choson reason, reprojection path/call counters, executor/worker counts,
RAM-budget/available-memory figures, and the per-cell `mode` / `bound_bytes` the planner stamps into each
cell record). A declared classifier excludes exactly these; its COMPLETENESS is proven by measurement
(capture absent) and asserted in the suite:

1. **Plan differences covered.** Two capture-OFF runs with different plans (pinned STREAM vs pinned INMEM)
have differing leaves that are **entirely** in the classifier (same for the capture-ON variant of the same
plan flip).
2. **`.cells` is NOT blanket-excluded.** `.cells` holds the PER-CELL records (cell_id, row, col, status,
chosen `mode`, `n_contributors`, `cell_gauge_mode`, `backend_used`, the reference-frame provenance fields,
`excluded`, `bound_bytes`). Under a plan flip the ONLY differing per-cell leaves are the two PLAN-METADATA
leaves `mode` / `bound_bytes` (asserted); every science/index per-cell field is identical, so the
classifier excludes only those two leaf NAMES, never the subtree.
3. **Uncontrolled, faithful run.** The three runs' product FITS are byte-identical *even though the plans
may differ*, and every ON-vs-OFF manifest difference beyond the single block lies in the classifier; the
classifier also covers every OFF-vs-OFF difference.
4. **Same-plan capture pair (strong claim).** With the plan held FIXED (memory pinned; at BOTH a pinned
STREAM and a pinned INMEM budget) the **only** pre-existing manifest fields that differ ON vs OFF are
genuine wall-timing / RSS measurement leaves; no science, coverage, aesthetic, contract, cell, layout or
offset field changes.

**Corrected, narrower claim (rework-1; now formally amended into the frozen clause by rework-16, §8
clause 3).** The §8 clause-3 phrase "all pre-existing manifest fields
canonical-identical" is NOT literally satisfiable on a shared host, because the manifest's pre-existing
*observability* fields are functions of the environment and change between ANY two sequential runs. The
honest, proven claim is: **(i)** the load-bearing science/coverage/aesthetic/uint16 ARRAY + FITS BYTES are
exactly byte-identical ON vs OFF (tolerance `0`), both uncontrolled and with the plan pinned; **(ii)** the
manifest gains exactly one additive `r40_diagnostic_capture` block ON and none OFF; **(iii)** every
pre-existing manifest difference ON vs OFF is confined to the declared, evidence-backed run-volatile
classifier (measurement + environment-driven plan), never a product field.

**Explicit NON-CLAIMS on this platform.** This measurement does **not** claim B2 acceptance, the 38-relation
panel OFF/ON qualification, the injected-effect recovery re-proof, the real product ROI parity, the exact
§1.3 held-out-key intersection, or a real **Windows** run — all remain **`NOT_RUN`**. No real data and no
FITS/pixel/image/dataset traversal of the real corpus; no Windows816/Zealfie/LP341; no product
default/GUI/config change; no correction/model fitting. The zero tolerance applies to the
science/coverage/aesthetic array + FITS bytes (where it is met exactly) and was **not** widened anywhere.

**Validation (executed, observed).** Run twice for reproducibility on this host (the plan can legitimately
vary when free RAM drifts).

| Command | Result |
|---|---|
| `python3 -m pytest tests/test_zegrid_r40b2_offon_identity.py -W error -q -p no:cacheprovider` | **20 passed** (twice) |
| `python3 -m pytest tests/test_zegrid_r40b2_offon_identity.py tests/test_zegrid_r40b2_pooled.py tests/test_zegrid_r40b2_seam.py tests/test_zegrid_r40_capture_core.py -W error -q -p no:cacheprovider` | **176 passed** (twice: 176/176) |
| `python3 -m pytest tests/test_zegrid_r37d_diagnostic_capture.py tests/test_zegrid_r37e_capture_runner.py tests/test_zegrid_r38a_causal_probe.py tests/test_zegrid_r39a_spatial_residual.py tests/test_zegrid_r40b1_synthetic_calibration.py -W error -q -p no:cacheprovider` | **417 passed** |
| `python3 -m py_compile tests/test_zegrid_r40b2_offon_identity.py` | OK |
| `git diff --check` | clean (`git status` shows only the intended 2 files) |

> **Task D does NOT accept B2.** It proves the frozen §8.2 product byte-identity (tolerance exactly `0`)
on real synthetic end-to-end product runs — uncontrolled and with the plan pinned — together with the
single additive manifest block, the measured environment-vs-capture causality of the plan, and the
evidence-backed isolation of the pre-existing run-observable fields. The 38-relation panel qualification,
the injected-effect recovery re-proof, the real product ROI parity, the exact §1.3 held-out intersection
and a real Windows run remain **`NOT_RUN`**. `R40-C` / `R40-D` / `HUMAN_GATE` / `LP341` remain
**`NOT_RUN` / `NOT_AUTHORIZED`**.
