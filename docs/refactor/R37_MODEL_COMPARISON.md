# R37 — Model comparison (Gate B, isolated pure prototype)

**Mission:** ZM-ZEGRID-R37-B
**Phase:** implementation (Gate B)
**Status:** pure prototype, UNWIRED. No production import, no default/GUI/config
change, no real run.
**Inputs (frozen):** `docs/refactor/R37_ARCHAEOLOGY_AND_RISKS.md`,
`docs/refactor/R37_BACKGROUND_MODEL_CONTRACT.md` (Gate A, ratified).
**Decision authority:** Junior (this document PROPOSES; it does not decide).

Prototype module: `src/zemosaic/core/zegrid/background_model.py`
Focused tests: `tests/test_zegrid_r37_background_model.py` (39 tests).
Benchmark + JSON: `.a2a-reports/ZM-ZEGRID-R37-B-scratch/` (`bench.py`,
`R37_bench.json`, `R37_bench.stdout.txt`; outside the repo).

---

## 1. What was built

Two additive inter-Cell models evaluated on the **same** synthetic corpora:

| | Model | Form | Solver |
|---|---|---|---|
| **M0** | `OFFSET_ONLY` (R34 baseline) | `science += o_i,c` | reused **UNCHANGED** `cell_offset.solve_additive_offsets` (central anchor) |
| **M1** | `ROBUST_PLAN` (candidate) | `science += a_i,c + b_i,c·x + d_i,c·y` | whitened **sparse LSQR** on `scipy.sparse`, slopes-only ridge |

`x,y ∈ [-1,1]` are **global canvas** normalized coordinates. Both are strictly
**additive-only**: no gain, colour matrix, rescaling or hidden multiply anywhere.
The central deterministic anchor Cell has `(a,b,d) = 0` (gauge only — it does not
itself prove identifiability).

Observation is a **same-global-pixel pair difference** on shared-valid / finite
pixels. Measurement band width is a parameter **separate** from the assembly
feather width. Split is a deterministic **75/25** train/holdout by fixed global
block parity. The edge fit is robust Huber (`1.345·MAD·1.4826`), float64, no RNG.

Because additive plane errors cancel in differences, the fitted correction is the
**negative** of the Cell's own plane error (`P_i = -(plane_i - plane_anchor)`).

### 1.1 Decision states

`OFFSET_ONLY | ROBUST_PLAN | INSUFFICIENT_EVIDENCE | ILL_CONDITIONED |
MODEL_MISMATCH | DISCONNECTED`. Precedence:

```
DISCONNECTED > INSUFFICIENT_EVIDENCE (support/leverage) > MODEL_MISMATCH
           > OFFSET_ONLY (plan offers no held-out gain) > ILL_CONDITIONED
           > ROBUST_PLAN (all §6 gates PASS)
```

`NONLINEAR_CANDIDATE` exists only as a **research note** — there is no spline
engine (see §5).

---

## 2. Gate results (contract §6)

The §6 thresholds were applied verbatim, data-only rank/condition being computed
separately from the regularized solve so regularization never hides absent
evidence.

| Contract gate | Prototype implementation | Result |
|---|---|---|
| §6.1.1 data-only full rank after anchoring | exact rank of whitened design | PASS on connected full-rank corpora (e.g. 6×4: 69/69) |
| §6.1.2 whitened `cond(A) ≤ 1e4` | `cond(A)` of whitened design (never `cond(AᵀA)`) | PASS wide band (cond 78–300); reported honestly |
| §6.1.3 ≥256 effective train samples/edge | Huber-weighted effective count | PASS for band ≥ 16 px with §2-sized Cells |
| §6.1.4 ≥8 tangent / 3 normal blocks | occupancy from block indices | PASS band ≥ 16 px |
| §6.1.5 finite coefficient uncertainties | covariance from regularized normal matrix | PASS where computable |
| §6.1.6 corner uncertainty ≤ `0.5σΔ` or ≤50% correction | per-corner information form | **FAIL at 16 px**; PASS at 32–64 px |
| §6.1.7 connected graph | BFS from anchor | PASS / DISCONNECTED otherwise |
| §6.1.8 order invariance ≤ `1e-10` | grid-order rebuild | PASS (exact) |
| §6.2.1 global held-out `R_plan ≤ 0.80 R_offset` | pooled robust sigma | PASS wide band (ratio ≈ 0.41) |
| §6.2.2 `R_plan ≤ 1.25 σΔ` | vs robust difference scale | PASS wide band |
| §6.2.3 per-edge p90 not worsening >5% | per-edge holdout p90 | PASS on accepted corpora |
| §6.2.4 improvement in both orientations | per-orientation ratio | PASS |
| §6.2.5 no train-only improvement | holdout governs | enforced |
| §6.3.1 recovery ≤ `1e-8` (noise-free) | gauge-fixed truth error | PASS (≤1e-8 on noise-free corpora) |
| §6.3.2 noisy correction `RMSE ≤ 0.25σ_pixel` | held-out residuals | PASS wide band |
| §6.3.3 celestial-gradient amplitude ratio ∈ [0.99,1.01] | per-pixel regression slope | PASS |
| §6.3.4 IFN bias ≤ `0.10σ`, ratio ∈ [0.99,1.01] | median bias + slope | PASS |
| §6.3.5 stellar flux median ≤0.1%, p95 ≤0.5% | annulus background-subtracted aperture | PASS |
| §6.3.6 identity exact ≤1e-10; noisy identity keeps M0 | coefficients / decision | PASS |
| §6.3.7 NaN/holes/dimensions preserved | application contract | PASS |
| §6.3.8 coverage/weights unchanged | science-only mutation | PASS |

---

## 3. Synthetic matrix results (contract §7)

Benchmark: `.a2a-reports/ZM-ZEGRID-R37-B-scratch/R37_bench.json`. Measured
single-shot on the dev host (8 vCPU, ~7 GiB RAM, SciPy 1.15.3 / NumPy 2.2.4).

| Corpus | Decision | held-out ratio | cond(A) | Notes |
|---|---|---:|---:|---|
| identity, noise-free | `OFFSET_ONLY` | — | 300 | correction ≤1e-10 |
| identity, noisy | `OFFSET_ONLY` | 1.00 | 296 | no gratuitous plan |
| known offsets | `OFFSET_ONLY` | 0.995 | 306 | offsets already suffice |
| known planes, noise-free | `OFFSET_ONLY` | ~0 | 78 | recovery ≤1e-8 |
| **known planes, wide band (64 px)** | **`ROBUST_PLAN`** | 0.414 | 78 | all gates PASS |
| known planes, 16 px band | `ILL_CONDITIONED` | 0.438 | 297 | corner-unc certain gate FAIL |
| common celestial gradient | `ROBUST_PLAN` | 0.417 | 79 | amplitude preserved |
| diffuse IFN / nebulosity | `ROBUST_PLAN` | 0.416 | 78 | bias ≤0.10σ, ratio∈[0.99,1.01] |
| bright stars (common) | `OFFSET_ONLY` | ~0 | 78 | Huber uninterfered, flux preserved |
| independent instrumental gradients | `OFFSET_ONLY` | ~0 | 78 | recovery ≤1e-8 |
| nonlinear, non-representable | **`MODEL_MISMATCH`** | 0.419 | 297 | not repaired by a plane |
| multiplicative gain/spectral mismatch | **`MODEL_MISMATCH`** | 0.713 | 1256 | detected |
| additive source-field control | `ILL_CONDITIONED` | 0.447 | 302 | **no** false mismatch |
| disconnected (6×1 gap) | `DISCONNECTED` | — | — | no fabricated bridge |
| weak/fragmented support | refusal | — | — | `INSUFFICIENT_EVIDENCE`/`DISCONNECTED` |
| layouts 1×4 / 4×1 / 2×2 / 6×4 | `ROBUST_PLAN` | 0.21 / 0.21 / 0.33 / 0.41 | 20 / 20 / 7 / 79 | wide band |
| RGB (3 channels) | all states reachable | — | — | per-channel solve |

Additional coverage: measurement bands 8/16/32/64 px; permutation determinism
(`≤1e-10`); pixel outliers; contradictory edge in a 2×2 cycle; low/high SNR;
NaN/hole/coverage preservation; JSON-safe provenance.

### 3.1 Key scientific findings

1. **A 16 px band cannot robustly identify the normal slope.** With a 16 px
   measurement band the corner-uncertainty gate fails (`ILL_CONDITIONED`) even
   for a clean synthetic plane truth, exactly as Gate A predicted
   (`cond(A)≈2430` ideal). Widening the measurement band to 32–64 px restores
   `ROBUST_PLAN`. The normal baseline — not the sample count — is the limiter.
2. **Common sky (gradient, IFN) is not a problem.** It cancels in same-pixel
   differences, so M1 neither absorbs nor biases it: amplitude ratios stay in
   `[0.99,1.01]` and IFN bias stays ≤`0.10σ`.
3. **Multiplicative mismatch is only conditionally detectable.** A per-Cell gain
   mismatch on a *locally planar* sky is mathematically indistinguishable from an
   additive plane shift (Montage's documented limit). It becomes detectable when
   the band carries **non-planar** structure (compact sources, high-frequency
   sky). The prototype flags it as `MODEL_MISMATCH`; when unobservable it stays
   silent rather than fabricating a verdict. This limitation is explicit.
4. **The correction is additive-only and background-neutral.** Additive
   application preserves NaN/support/holes/dimensions and leaves
   background-subtracted stellar flux unchanged (≤0.1%).

---

## 4. Scalability (contract §9)

Pure sparse witness on synthetic **compact** observations (no images, no FITS, no
`N_frames×canvas`) — `sparse_scale_witness`:

| Grid | Cells | Edges | Unknowns | Rows | Rank | cond(A) | solve | RSS Δ | RSS peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20×20 | 400 | 760 | 1,197 | 3,078 | 1,197 | 82 | 6.9 s* | 32 MiB | 195 MiB |
| 50×50 | 2,500 | 4,900 | 7,497 | 19,698 | NOT_RUN | NOT_RUN | 1.1 s* | <1 MiB | 195 MiB |
| 100×100 | **10,000** | 19,800 | 29,997 | 79,398 | NOT_RUN | NOT_RUN | 5.7 s* | 49 MiB | 244 MiB |

\* wall-clock includes host contention; values are indicative, not calibrated.

A dense design for 10,000 Cells would be ≈19 GB; the sparse path stays at
**~0.24 GiB peak** and never densifies. Exact rank/condition is reported only
when the column count is small enough for a dense SVD (`≤1500`); larger systems
report `NOT_RUN` honestly (structural O(cells+edges) construction is what is
claimed, not an O(1) condition estimate).

---

## 5. B3 — nonlinear / spline study (documentation only)

No engine was written. References (methodological only, **no code copied**):

- **Montage** (Caltech/IPAC) — assumes calibrated brightness; models
  non-sky low-frequency discrepancies with offsets/slopes; explicitly warns that
  generalized matching cannot separate sky from background when assumptions fail.
- **PhotometricMosaic** (John Murphy) — photometric scale matching plus a surface
  spline for relative gradients.

Position: a per-Cell spline could absorb real sky/IFN and is only identifiable
with independent calibration. It is therefore **not executable here**: any
nonlinear residual observed by the additive prototype is reported as
`MODEL_MISMATCH` needing research/upstream calibration, never repaired by a
plane. Proposed credits if R37 ever ships: PhotometricMosaic — John Murphy;
Montage — Caltech/IPAC.

---

## 6. Limits (explicit, not hidden)

- **No real-data qualification.** The exact 816-frame pre-feather product is
  `NOT AVAILABLE` locally (Gate A). Every number above is synthetic. A
  synthetic-perfect M1 still leaves real validation missing.
- **Narrow-band normal slope** is weakly constrained; the honest decision is
  `ILL_CONDITIONED` unless a wider measurement band is justified.
- **Multiplicative detection is conditional** (§3.1.3); a locally planar sky
  hides a pure gain mismatch.
- **Wide-band corpora use 64 px Cells–to–band geometry**; real Cells (~680–1000 px
  tangent) would give far better normal leverage, but that is an expectation, not
  a measurement.
- Regularized coefficient uncertainties are exact for the small corpora here; at
  scale only O(E) compact statistics are retained (no global posterior).
- Wall-clock/RSS are single-run and host-loaded.

---

## 7. Future diagnostic artifact (proposal only)

Produced **while patches coexist** during assembly, O(edges×channels×const),
no extra source pass: per edge/channel — endpoint ids + orientation, normalized
coordinate definition, raw & effective support, tangent/normal occupancy,
train/holdout counts, robust noise, fitted coefficients, compact information/rhs,
covariance/condition/leverage, held-out residual metrics, rejection fraction,
cycle contribution, refusal reason. This is what would allow a future authorized
real-data qualification.

---

## 8. Proposed scientific decision (Junior arbitrates)

- Prototype fidelity/evidence gates: **PASS on synthetic corpora when the
  measurement band is wide enough**; **explicit refusals** where it is not.
- Architecture: sparse, O(Cells+edges), no dense design, no images — suitable for
  a future compact diagnostic.
- **Proposed outcome: `NEEDS RESEARCH`.** Rationale: (a) the real 816-frame
  pre-feather observations are unavailable, so no production change can be
  qualified; (b) the normal slope is weakly identifiable in a narrow band and
  multiplicative mismatch is only conditionally detectable. A successful
  synthetic `ROBUST_PLAN` is **not** authorization to change production.

**No production wiring is proposed or performed.**
