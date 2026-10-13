# R37 — Model comparison (Gate B, isolated pure prototype) — rework-1

**Mission:** ZM-ZEGRID-R37-B · **Phase:** implementation (Gate B) · **Rework:** r1
**Status:** pure prototype, UNWIRED. No production import, no default/GUI/config
change, no real run.
**Inputs (frozen, unchanged):** `docs/refactor/R37_ARCHAEOLOGY_AND_RISKS.md`,
`docs/refactor/R37_BACKGROUND_MODEL_CONTRACT.md`.
**Independent review:** `…R37-B.nono.r0.md` (FINDINGS → REWORK). Junior ratified.
**Decision authority:** Junior (this document PROPOSES).

Prototype: `src/zemosaic/core/zegrid/background_model.py`
Tests: `tests/test_zegrid_r37_background_model.py` (57 tests).
Benchmark: `.a2a-reports/ZM-ZEGRID-R37-B-scratch/` (`bench.py`, `R37_bench.json`,
`R37_bench.stdout.txt`, `nono_repro.py`, `nono_repro.out.txt`).

---

## 1. Models

| | Model | Form | Solver |
|---|---|---|---|
| **M0** | `OFFSET_ONLY` | `science += o_i,c` | reused **UNCHANGED** `cell_offset.solve_additive_offsets` (per channel) |
| **M1** | `ROBUST_PLAN` | `science += a + b·x + d·y` | whitened **sparse LSQR**, slopes-only ridge, information cap + bounded graph IRLS + rejection |

`x,y ∈ [-1,1]` global normalized. Strictly additive-only. Central anchor
`(a,b,d)=0`. The recovered correction is `P_i = -(plane_i - plane_anchor)`.

---

## 2. Rework-1 corrections (finding → fix → test)

### HIGH

**H1 — measurement band was not separable from the overlap (crashed on band 8).**
The band filter now applies **consistently** to planes, validity masks and
coordinate arrays (single 2-D global-coordinate mask). A requested band wider
than the overlap is reported as `band_clamped` (+ `band_used_px`, `overlap_px`,
`measurement_px`) — never silently claimed.
*Tests:* `test_h1_band_independent_of_overlap_no_crash[4/8/16/32]`,
`test_h1_band_narrower_than_overlap_band_used_le_buget`,
`test_h1_band_wider_than_overlap_is_reported_clamped`. *Closure probe:* prod-like
`halo=8, band=8` → decision returned (no `IndexError`), `band_used_px ≤ 8`.

**H2 — fragmented per-channel support crashed the M0 baseline.**
M0 is now solved **one channel at a time** with only that channel's observed
relations, `raise_on_disconnected=False`; the R34c singular-cut exception can no
longer escape `compare_models`. An explicit refusal state is returned.
*Tests:* `test_h2_channel_hole_returns_refusal_not_exception`.

**H3 — per-channel rank collapsed to a MAX.**
Connectivity, rank, condition and uncertainty are computed **per channel**; the
global gate takes the **worst** channel (min rank / max cond); an
under-determined or disconnected channel gets **NaN** coefficients and an
explicit refusal — never a silent zero. `per_channel` diagnostics are in the
payload.
*Tests:* `test_h3_per_channel_diagnostics_and_no_silent_zero`,
`test_h3_worst_channel_governs_rank_and_condition`.

### MEDIUM

**M1 — §6.2.3 p90 gate relaxed + asymmetric.** Gate metrics are now **RMS around
zero** (bias-preserving) with a single symmetric operator for M0 and M1; p90 is
`percentile(abs(residual), 90)`; the undeclared `0.05·σΔ` floor is removed; the
frozen `cfg.per_edge_p90_worsen_max` is used. A non-finite orientation ratio is a
**FAIL**.
*Tests:* `test_m1_const_bias_is_visible_to_gate`,
`test_m1_p90_gate_is_strict_five_percent`, `test_l2_nonfinite_orientation_ratio_fails`.

**M2 — §6.1.6 corner uncertainty from the regularized system.** The gate now uses
the **data-only** design covariance (dense SVD of the design, explicitly
densified, ≤1500 cols; `NOT_RUN` above). The regularized covariance is reported
separately. The gate is invariant to λ.
*Tests:* `test_m2_corner_uncertainty_independent_of_lambda`; probe shows
`λ=0 vs λ=1` corner-σ difference `0.0`.

**M3 — no relational robustification / mislabelled test.** Added a deterministic
per-edge **information cap** (`EDGE_INFO_CAP_MULT=100`) and a bounded
**graph-level IRLS** plus a connectivity-safe, leave-one-out **relational
rejection** (`EDGE_REJECT_Z=8`, ≤25% of relations, must materially halve the
robust loss). A genuine contradictory relation is now tested at observation level.
*Tests:* `test_m3_huge_edge_cannot_dominate`,
`test_m3_true_contradictory_relation_in_cycle`,
`test_m3_graph_consistent_cell_shift_is_not_a_contradiction`,
`test_m3_corrupted_overlap_does_not_yield_confident_plan`,
`test_m3_chain_has_no_cycle_evidence`. *Closure probe:* σ=1e-6 contradictory edge
→ `max|coef| = 0.0` (was 50), capped 1, rejected 1.

**M4 — no cycle residuals / refusal reasons.** Per-edge `refusal_reason` is now
computed (band clamp etc.); an explicit **dropped-relation inventory** is
returned; **cycle residuals** are computed per elementary grid cycle and channel
(offset residual, σ-scale, z). `refusal_reason=None` is gone.
*Tests:* `test_m4_payload_has_cycles_dropped_and_reasons`.

**M5 — M0 comparison not truly held out.** M0's offset observation now uses the
**TRAIN-only** local difference (`local_offset_train`); M0 and M1 use the same
split. *Tests:* `test_m5_m0_uses_train_only_local_offset`.

**M6 — incoherent decision at σΔ→0.** A degenerate (non-estimable) noise scale now
returns **`INSUFFICIENT_EVIDENCE`** with an explicit reason, while still verifying
truth recovery — never a fake `OFFSET_ONLY` from 0/0, never an automatic
`ROBUST_PLAN`. *Tests:* `test_m6_noisefree_is_insufficient_evidence_with_recovery`.

**M7 — mismatch thresholds not in the frozen contract.** The detector thresholds
are labelled **PROVISIONAL** in provenance and payload
(`mismatch_thresholds_status = PROVISIONAL_NOT_IN_FROZEN_CONTRACT`); the state is
experimental and documented as such. *Tests:*
`test_m7_mismatch_thresholds_provisional`.

### LOW

**L1 — `densified` was a hard-coded literal.** Replaced by measured flags:
`global_solve_densified` (always false — LSQR) and `diagnostics_densified` (true
when a dense design SVD/covariance was actually used, ≤1500 cols). *Tests:*
`test_l1_densification_flags_are_measured`.

**L2 — orientation inf skipped.** See M1. *Test:*
`test_l2_nonfinite_orientation_ratio_fails`.

**L3 — weak/tautological assertions.** `or True`, the always-true enum assertion
and broad `in ALLOWED` were replaced by expected-outcome assertions where the
contract determines the result.

**L4 — matrix incomplete.** Added heterogeneous
nights/cameras/noise/exposure/spectral tests, alternate-layout-same-physical-truth
tests, a genuine contradictory-relation/cycle test, and an
application-invariant (NaN/dims/inputs-unmutated) test.

---

## 3. Contract §6 conformance (post-rework)

| Gate | Verdict |
|---|---|
| §6.1.1 data-only full rank (per channel) | PASS (worst channel; NaN + refusal otherwise) |
| §6.1.2 whitened cond ≤ 1e4 | PASS (reported; never `cond(AᵀA)`) |
| §6.1.3 ≥256 effective train | PASS band ≥16 px |
| §6.1.4 ≥8 tangent / ≥3 normal blocks | PASS band ≥16 px |
| §6.1.5 finite uncertainties | PASS where computable (data-only); `NOT_RUN` at scale |
| §6.1.6 corner unc ≤0.5σΔ or ≤50% corr | PASS wide band; FAIL 16 px — on **data-only** covariance |
| §6.1.7 connectivity (per channel) | PASS / DISCONNECTED |
| §6.1.8 order invariance ≤1e-10 | PASS (exact) |
| §6.2.1 R_plan ≤ 0.80 R_offset | PASS wide band (≈0.43) |
| §6.2.2 R_plan ≤ 1.25 σΔ | PASS wide band |
| §6.2.3 per-edge p90 ≤ +5% | frozen 5%, symmetric, no floor |
| §6.2.4 both orientations | present non-finite ratio → FAIL |
| §6.2.5 no train-only improvement | enforced |
| §6.3.1..8 fidelity/astronomy/NaN | PASS (noise-free exactness ≤1e-8; IFN/star preservation; application invariants) |

---

## 4. Synthetic matrix (post-rework, `R37_bench.json`)

| Corpus | Decision | ratio | cond(A) |
|---|---|---:|---:|
| identity, noise-free | `INSUFFICIENT_EVIDENCE` (degenerate σΔ) | — | 300 |
| identity, noisy | `OFFSET_ONLY` | 1.001 | 296 |
| known offsets | `OFFSET_ONLY` | 1.001 | 306 |
| known planes, noise-free | `INSUFFICIENT_EVIDENCE` (degenerate) | ~0 | 78 |
| **known planes, wide band** | **`ROBUST_PLAN`** | 0.430 | 78 |
| known planes, 16 px | `ILL_CONDITIONED` | 0.466 | 297 |
| common celestial gradient | `ROBUST_PLAN` | 0.429 | 79 |
| diffuse IFN | `ROBUST_PLAN` | 0.429 | 78 |
| bright stars (common) | `INSUFFICIENT_EVIDENCE` (degenerate) | ~0 | 78 |
| independent instr. gradients | `INSUFFICIENT_EVIDENCE` (degenerate) | ~0 | 78 |
| nonlinear non-representable | `MODEL_MISMATCH` | 0.398 | 297 |
| multiplicative gain mismatch | `MODEL_MISMATCH` | 0.793 | 1256 |
| additive source control | `ILL_CONDITIONED` | 0.477 | 302 |
| disconnected 6×1 | `DISCONNECTED` | — | — |
| layouts 1×4 / 4×1 / 2×2 / 6×4 | `ROBUST_PLAN` | .22/.22/.35/.43 | 20/20/7/79 |

Noise-free corpora now honestly report `INSUFFICIENT_EVIDENCE` (M6) instead of the
previous misleading `OFFSET_ONLY`; their recovery is still verified ≤1e-8.

---

## 5. Scalability (contract §9)

| Grid | Cells | Edges | Unknowns | Rank | cond | solve (s)* | RSS peak | global densified | diag densified |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 20×20 | 400 | 760 | 1,197 | 1,197 | 82 | 5.3 | 196 MiB | false | **true** (≤1500) |
| 50×50 | 2,500 | 4,900 | 7,497 | NOT_RUN | NOT_RUN | 3.9 | 196 MiB | false | false |
| 100×100 | **10,000** | 19,800 | 29,997 | NOT_RUN | NOT_RUN | 26.0 | 267 MiB | false | false |

\* host-loaded single run. Dense design for 10,000 Cells ≈ 4.8 GB; the sparse path
never densifies and does **no** dense diagnostic/covariance at scale.

---

## 6. Limits (explicit)

- **No real-data qualification** (the 816-frame pre-feather product is absent).
- **Narrow band**: 16 px normal slope weakly identifiable → `ILL_CONDITIONED`.
- **Multiplicative mismatch conditionally detectable** (locally-planar sky hides a
  pure gain mismatch) — documented, never fabricated.
- **Contradictory relations**: with low graph redundancy a bounded IRLS downweights
  but cannot fully eliminate a bad relation (a chain cannot identify a
  contradiction at all — reported as absence of cycle/evidence).
- Mismatch thresholds are **provisional**, not frozen contract.
- Wall-clock/RSS single-run, host-loaded.

---

## 7. Proposed scientific decision (Junior arbitrates)

All synthetic fidelity/evidence gates pass when the measurement band is wide
enough, with explicit refusals otherwise; the architecture is sparse and suitable
for a future compact diagnostic. Because the real pre-feather observations are
unavailable and narrow-band/multiplicative identifiability is limited, the
proposed outcome remains **NEEDS RESEARCH**. **No production wiring is proposed
or performed.**
