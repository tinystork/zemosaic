# R37 — Offline pre-feather diagnostic artifact (R37-C design)

**Mission:** ZM-ZEGRID-R37-C-DIAGNOSTIC · **Phase:** implementation · **Iteration:** r0
**Status:** PURE / OFFLINE / UNWIRED. No production import, no correction applied,
no default/config/GUI change, no R34/R36 semantic change.
**Inputs:** `docs/refactor/R37_ARCHAEOLOGY_AND_RISKS.md`,
`docs/refactor/R37_BACKGROUND_MODEL_CONTRACT.md`,
`docs/refactor/R37_MODEL_COMPARISON.md`.
**Decision authority:** Junior after independent review. This document PROPOSES.

Module: `src/zemosaic/core/zegrid/background_diagnostics.py`
Tests: `tests/test_zegrid_r37c_diagnostics.py` (37 tests)

---

## 1. Why this exists

The accepted R37-B prototype (`background_model.py`) can only be evaluated on
synthetic corpora because **the canonical final run `/home/tristan/caldwell11/
zegrid_out/R34F` (653 frames) persists no separately recoverable pre-feather
Cell-pair evidence**. Without the exact relations used at assembly time, no
source FITS can be re-measured for the real relation set, so Gate B's honest
conclusion stays `NEEDS RESEARCH`.

R37-C adds the smallest PURE, DETERMINISTIC, OFFLINE layer that captures that
evidence from in-memory pre-feather `PlanCell` / `CellPatch` objects and
round-trips it back into the accepted M0/M1 decision machinery. It is the
missing "diagnostic artifact proposal" named by the contract §10 — it does **not**
authorise production correction.

---

## 2. What is captured

For every 4-neighbour Cell pair × channel the canonical pipeline measures a
same-global-pixel difference. R37-C persists, per relation:

| Group | Fields | Purpose |
|---|---|---|
| identity | `u`, `v`, `channel`, `orientation` | relation identity |
| support | `n_train`, `n_holdout`, `n_holdout_kept`, `effective_train`, `tangent_blocks`, `normal_blocks` | evidence/occupancy gates |
| noise | `robust_sigma`, `rejection_fraction` | σΔ, mismatch, weighting |
| **sufficient statistics** | `theta_global[3]`, `information[3x3]`, `rhs[3]` | exact offline M1 solve |
| robust stats | `level_z`, `curvature_z`, `level_mean`, `local_offset_train` | mismatch + M0 level |
| band | `measurement_px`, `overlap_px`, `band_used_px`, `band_clamped` | measurement-band provenance |
| held-out evidence | `holdout_xy[m,2]`, `holdout_diff[m]`, `holdout_capped` | offline held-out metrics |

Plus layout metadata (`nx`, `ny`, `canvas_hw`, `channels`, `anchor`, `cell_ids`,
normalization, coordinate convention, band/block/feather parameters) and the
**explicit `dropped` relation inventory** (Cell absent, no shared valid pixel,
band without support, insufficient train support).

**No duplicate science.** Pair differences and the coordinate convention are
produced by the UNCHANGED `background_model.observe_edge` via
`build_observations`. There is no second implementation of the pixel/band math.

---

## 3. Deterministic capped held-out evidence

`n_holdout` stores the **full** held-out support count (so the M0 support weights
reproduce exactly). The persisted `holdout_xy`/`holdout_diff` are a deterministic
subsample, capped at a **documented fixed cap** `HOLDOUT_CAP = 64`, so storage is
`O(relations × cap)` — never `O(canvas)` or `O(frames)`.

Selection (`select_capped_holdout`), always the same for the same input:

1. **Forced boundary/extremes** — min/max of the tangent coordinate, min/max of
   the normal coordinate and the four diagonal extremes. This makes the capped
   set span the exact tangent/normal range of the full held-out set.
2. **Quantile stratification** — the remaining points are binned into a fixed
   `4 × 4` (tangent × normal) quantile grid.
3. **Systematic within-stratum picks** — the leftover budget is allocated across
   strata (≥1 pick per occupied stratum when the budget allows) and filled at
   evenly-spaced positions inside each stratum.

It is **never an order-biased prefix**. `holdout_capped` records when the subset
is a subsample. If capping ever mattered to a decision, the artifact says so
rather than hiding it.

---

## 4. Round-trip and offline decision reuse

The loader reconstructs `EdgeObservation` objects and `recompute_decision`
delegates to `background_model.compare_models_from_observations` — a
**behaviour-preserving extraction** of the tail of `compare_models` (see §7). The
accepted gates, thresholds and solver are exactly the accepted ones; no gate is
re-implemented here.

Declared tolerances:

| Quantity | Condition | Tolerance |
|---|---|---|
| M1 coefficients, M0 offsets | always | `COEF_ROUNDTRIP_ATOL = 1e-10` (observed: bit-exact) |
| held-out `r_plan` | no capping | exact (`1e-12` rel) |
| held-out `r_plan` | capping active | `HELDOUT_RMS_REL_TOL = 0.10` relative |

Measured on the synthetic 2×2 / 2×3 cases (seed 7, measurement band 16 px):

| cap | 2×2 rel Δ r_plan | 2×3 rel Δ r_plan | decision preserved |
|---:|---:|---:|:--:|
| 8 | 0.1030 | 0.1066 | yes |
| 16 | 0.0488 | 0.0298 | yes |
| ≥ full | 0.0 | 0.0 | yes |

M1 coefficients are unaffected by capping (they depend only on the sufficient
statistics); the held-out metric is the only quantity that moves, and it is
quantified by the tests, never claimed exact.

---

## 5. Strict JSON discipline

`json.dumps(payload, allow_nan=False)` always succeeds. Scientific arrays
(`theta_global`, `information`, `rhs`, `holdout_xy`, `holdout_diff`) and scalars
(`robust_sigma`, `effective_train`, `level_*`, `curvature_z`,
`local_offset_train`) are finite by construction and are **rejected** by the
loader if they are `null`, non-finite, wrongly shaped or wrongly typed; a
non-finite value is never degraded to a fabricated `0`. The only nullable
scientific field is `measurement_px` (absent band), which is semantically
inapplicable. `loads` also rejects the JSON `NaN`/`Infinity` literals through a
strict `parse_constant`.

The loader validates: schema/version, exact key sets, integer/boolean/float types
(booleans are not integers), `information` shape `(3,3)`, vector lengths, held-out
length consistency, `channel ∈ [0, channels)`, `anchor ∈ cell_ids`, unique
`cell_ids`, and the relation/dropped counts against the metadata.

---

## 6. Size and scaling

`estimate_footprint` computes analytic `O(relations × cap)` counts and never
builds a dense canvas, dense design matrix, FITS or large object.

| Witness | Cells | Relations | Held-out points | Est. bytes |
|---|---:|---:|---:|---:|
| canonical 6×4×3 (core 64 px) | 24 | 114 | 7 296 | ≈ 0.50 MB |
| sparse witness 100×100 | 10 000 | 19 800 | 1 267 200 | ≈ 86.7 MB |

The measured canonical 6×4×3 payload (band 16 px, 3 channels) is **464 090
bytes**. The 10 000-Cell witness is estimated only (no object is built); at that
scale a smaller `HOLDOUT_CAP` (or training-only sufficient statistics) would be
the operational choice, which the `O(relations × cap)` scaling makes explicit.

---

## 7. The only production-adjacent change

`background_model.py` gains `compare_models_from_observations(...)`: a
**behaviour-preserving extraction** of the body of `compare_models` after
`build_observations`, and `compare_models` now delegates to it. No gate,
threshold, solver, provenance or output value changes — the R37-B suite (72
tests) passes unchanged. This is what lets the offline artifact rerun the
accepted decisions instead of duplicating the science.

No other production module is touched. `background_diagnostics` is never imported
by `zemosaic_zegrid_mode.py`, `cell_feather.py`, config, GUI, dispatcher or
manifests (enforced by a test).

---

## 8. Prior art (research; no third-party code copied)

Primary/authoritative sources consulted (Gate-A read-only snapshots archived at
`.a2a-reports/ZM-ZEGRID-R37-gateA/references/`):

- **Montage — Background Modeling and Rectification** (Caltech/IPAC):
  <https://montage.ipac.caltech.edu/docs/algorithms.html>. Overlap-difference
  fitting of a low-spatial-frequency plane (offset + slopes) to each image
  against its neighbours, applied by half to avoid ringing and iterated to global
  consistency; the plane fit is a simplification of the full least-squares.
  *Reused conceptually:* the overlap-difference relation set, the planar
  additive model, the accepted-set-is-source-of-truth global solve. *Not copied:*
  any code; Montage assumes absolute brightness calibration, which R37-B does not
  establish.
- **Montage — `mBgModel`** <https://montage.ipac.caltech.edu/docs/mBgModel.html>:
  iterative global background model from an image-to-image difference parameter
  table. *Reused conceptually:* sufficient-statistics-per-relation + bounded
  iteration. Montage's "adjust by half each iteration" is deliberately **not**
  adopted here (the R37 solver solves once on the accepted set).
- **Montage paper** — Jacob et al. 2010, arXiv:1005.4454.
- **PhotometricMosaic** (John Murphy) <https://www.photometricmosaic.com/>:
  photometric **scale** matching + a **surface spline** for relative gradients.
  *Reused conceptually:* gradient modelling across overlaps is a real,
  publication-grade concern; *rejected* here as an executable model because a
  per-Cell spline is not identifiable without independent calibration
  (background_model already records this as `NONLINEAR_CANDIDATE`, research
  only). No code copied.
- **SDSS ubercalibration** — Padmanabhan et al. 2008, arXiv:astro-ph/0703454:
  IRLS-based relative recalibration over an overlap graph with a gauge anchor.
  *Reused conceptually:* robust (IRLS) overlap-graph estimation and explicit
  gauge fixing; R37-B's bounded graph IRLS and anchor are the same family of
  ideas, implemented independently.
- **SWarp** — Bertin 2010, arXiv:1010.0688: overlap background/scale
  equalisation. *Reused conceptually:* per-relation compact statistics.

**No third-party code, data or licence-encumbered material is copied.**
Methodological acknowledgements if a future mission adopts these methods: John
Murphy / PhotometricMosaic and Caltech/IPAC Montage. Live network fetches of
montage.ipac.caltech.edu, photometricmosaic.com and export.arxiv.org were
partially blocked (connection refused / HTTP 429) at run time; the archived
Gate-A snapshots plus the arXiv identifiers above are the recorded basis.

---

## 9. Limits and non-goals

- OFFLINE diagnostic capture only. No production import or wiring. No runtime
  default change. No corrected mosaic output. No correction application.
- No new dependency, no copied third-party code, no spline/nonlinear solver, no
  reprojection, no cache/prefetch/multiprocessing, no hidden RNG.
- No write under `/home/tristan/caldwell11` or M106. No Caldwell run.
- float64 scientific math; deterministic output.
- The artifact is sufficient **given** a future harness that can emit pre-feather
  `PlanCell`/`CellPatch` objects. Producing that capture in a real run (and
  therefore a real R37 evaluation) is out of scope here and remains the next
  step.
