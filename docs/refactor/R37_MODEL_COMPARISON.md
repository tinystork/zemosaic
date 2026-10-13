# R37 — Model comparison (Gate B, isolated pure prototype) — rework-2

**Mission:** ZM-ZEGRID-R37-B · **Phase:** implementation (Gate B) · **Rework:** r2
**Status:** pure prototype, UNWIRED. No production import, no default/GUI/config
change, no real run.
**Inputs (frozen, unchanged):** `docs/refactor/R37_ARCHAEOLOGY_AND_RISKS.md`,
`docs/refactor/R37_BACKGROUND_MODEL_CONTRACT.md`.
**Reviews:** `…R37-B.nono.r0.md`, `…R37-B.nono.r1.md` (FINDINGS → REWORK r2).
**Decision authority:** Junior (this document PROPOSES).

Prototype: `src/zemosaic/core/zegrid/background_model.py`
Tests: `tests/test_zegrid_r37_background_model.py` (67 tests).
Benchmark: `.a2a-reports/ZM-ZEGRID-R37-B-scratch/` (`bench.py`, `R37_bench.json`,
`R37_bench.stdout.txt`, `nono_repro*.py`).

---

## 1. Models

| | Model | Form | Solver |
|---|---|---|---|
| **M0** | `OFFSET_ONLY` | `science += o_i,c` | unchanged `cell_offset.solve_additive_offsets` (per channel; NaN + `applicable=False` for a refused channel) |
| **M1** | `ROBUST_PLAN` | `science += a + b·x + d·y` | sparse whitened **LSQR** + slopes-only ridge + info cap + bounded graph IRLS + bounded connectivity-safe relational rejection |

Global normalized `x,y ∈ [-1,1]`; strictly additive-only; central anchor
`(a,b,d)=0`; correction `P_i = -(plane_i - plane_anchor)`.

**The ACCEPTED relation set is the source of truth.** After rejection the design is
rebuilt and rank/cond/covariance/rows/uncertainty/connectivity/heldout/evidence/
occupancy/mismatch/cycles are all evaluated on the accepted set.

---

## 2. Rework-2 corrections (N1–N11)

| # | Finding | Fix | Test |
|---|---|---|---|
| **N1** | rows counter omitted data rows; "4.8 GB" wrong | `rows` = exact final assembled design rows (data+ridge) per channel; witness rows/bytes derived from the real shape | `test_n1_rows_equal_final_design_shape_6x4`, `test_n1_scale_witness_counts` |
| **N2** | chain (no cycle) promoted to `ROBUST_PLAN` | accepted cyclomatic number `E−V+C` must be ≥1; acyclic → explicit `INSUFFICIENT_EVIDENCE` reason `no_cycle_redundancy` | `test_n2_chain_no_redundancy_is_refused`, `test_n2_chain_contradiction_is_not_robust_plan`, updated `test_layouts_and_truth` |
| **N3/N11** | gates on the raw set; no rejected identities | accepted-set gates; `per_channel[c]` persists `accepted_keys`, `rejected_edges`, `capped_edges` (u,v,channel,reason,status,initial_scale,final_weight); rank/cond/cov/rows recomputed post-rejection | `test_n3_rejected_identities_persisted_and_post_reject_diagnostics`, `test_n3_compare_models_reports_accepted_subset` |
| **N4** | M0 fabricated zeros for a refused channel | refused channel → `offset_offsets[:,c]=NaN` and `per_channel[c].m0_applicable=False`; JSON NaN→null | `test_n4_refused_channel_m0_is_nan_not_zero` |
| **N5** | null covariance directions → variance 0 | `_data_only_covariance` maps null directions to NaN; refused/rank-deficient channels emit NaN uncertainty | `test_n5_rank_deficient_channel_uncertainty_is_nan` |
| **N6** | unguarded empty residual median (2 warnings) | `t.size == 0` guard in the rejection loop; `-W error` clean | `pytest -W error` (0 warnings) |
| **N7** | vacuous L2 closure test | direct, non-vacuous unit test of `_orientation_ok` (present non-finite ratio → False) | `test_l2_nonfinite_orientation_ratio_is_fail` |
| **N8** | robust knobs under-declared | provenance `robust_knobs` with cap/IRLS/reject_z/**max_reject_fraction**/**acceptance rule**, status PROVISIONAL_RESEARCH_ONLY | `test_n8_provenance_declares_robust_knobs` |
| **N9** | p90 test had a trivial absolute clause | relative-only gate `p90_plan ≤ (1+cfg.per_edge_p90_worsen_max)·p90_m0`; non-finite p90 on an included edge → FAIL | `test_m1_p90_gate_is_strict_five_percent_relative_only` |
| **N10** | claim "`refusal_reason=None` is gone" inaccurate | `None` = accepted; explicit reason for clamped/dropped/rejected (identities persisted) | `test_n10_refusal_reason_semantics` |
| **N11** | diagnostics computed before rejection | diagnostics now computed on the **post-rejection accepted** design (see N3) | `test_n3_*` |

Carried over from r0/r1 (still in force): H1 band consistency, H2 no escaping
solver exception, H3 per-channel diagnostics, M1 held-out RMS-around-zero +
symmetric p90, M2 data-only corner uncertainty, M5 M0 train-only, M6 σΔ→0 refusal,
M7 provisional mismatch thresholds, L4 matrix.

---

## 3. Synthetic matrix (rework-2, `R37_bench.json`)

| Corpus | Decision | ratio | cond(A) |
|---|---|---:|---:|
| identity, noise-free | `INSUFFICIENT_EVIDENCE` (degenerate σΔ) | — | 300 |
| identity, noisy | `OFFSET_ONLY` | 1.001 | 340 |
| known offsets | `OFFSET_ONLY` | 1.001 | 336 |
| known planes, noise-free | `INSUFFICIENT_EVIDENCE` (degenerate) | ~0 | 84 |
| **known planes, wide band** | **`ROBUST_PLAN`** | 0.430 | 82 |
| known planes, 16 px | `ILL_CONDITIONED` | 0.466 | 318 |
| common celestial gradient | `ROBUST_PLAN` | 0.429 | 88 |
| diffuse IFN | `ROBUST_PLAN` | 0.429 | 82 |
| bright stars (common) | `INSUFFICIENT_EVIDENCE` (degenerate) | ~0 | 77 |
| independent instr. gradients | `INSUFFICIENT_EVIDENCE` (degenerate) | ~0 | 85 |
| nonlinear non-representable | `MODEL_MISMATCH` | 0.398 | 294 |
| multiplicative gain mismatch | `MODEL_MISMATCH` | 0.793 | 1396 |
| additive source control | `ILL_CONDITIONED` | 0.477 | 313 |
| disconnected 6×1 | `DISCONNECTED` | — | — |
| **layout 1×4 / 4×1** | **`INSUFFICIENT_EVIDENCE`** (no cycle) | .22/.22 | 20/20 |
| layout 2×2 | `ROBUST_PLAN` | 0.352 | 9 |
| layout 6×4 | `ROBUST_PLAN` | 0.429 | 78 |

Fidelity (unchanged): noise-free recovery ≤1e-8; celestial/IFN amplitude in
[0.99,1.01]; IFN bias ≤0.10σ; stellar flux bias ≤0.1%; application invariants
(NaN/dims/inputs) preserved.

---

## 4. Contract §6 conformance (post rework-2)

Data-only full rank (per channel, worst) ✓ · cond(A) ≤ 1e4 ✓ · ≥256 effective
train ✓ · ≥8 tangent/≥3 normal ✓ · finite uncertainties ✓ (NaN for refused) ·
§6.1.6 corner unc on the **data-only** system ✓ · connectivity per channel ✓ ·
order invariance exact ✓ · §6.2.1/2 ✓ (σΔ=0 → refusal) · §6.2.3 frozen 5%,
relative-only ✓ · §6.2.4 present non-finite → FAIL ✓ · §6.2.5 train-only forbidden ✓
· **cycle-redundancy gate** (new) ✓ · §6.3.1–8 fidelity/astronomy/NaN ✓.

---

## 5. Scalability (contract §9)

| Grid | Cells | Edges | Unknowns | rows (data+ridge) | Rank/cond | solve (s)* | RSS peak | global dense | diag dense |
|---|---:|---:|---:|---:|---|---:|---:|---|---|
| 20×20 | 400 | 760 | 1,197 | 3,078 | 1,197 / 86 | 5.4 | 196 MiB | false | **true** (≤1500) |
| 50×50 | 2,500 | 4,900 | 7,497 | 19,698 | NOT_RUN | 3.8 | 196 MiB | false | false |
| 100×100 | **10,000** | 19,800 | 29,997 | **79,398** | NOT_RUN | 24.8 | 260 MiB | false | false |

`rows` = Σ_channels (data rows + ridge rows) of the FINAL assembled design.
Dense-design estimate = `rows · unknowns · 8` (`channels` separate; witness
`channels=1`): **10,000 Cells → 79,398 rows → 19.05 GB**. The global solve is
sparse LSQR and never densifies; dense diagnostics (rank/cond/covariance) are used
only ≤1500 columns and are flagged.

\* host-loaded single run.

---

## 6. Limits (explicit)

- **No real-data qualification** (816-frame pre-feather product absent).
- **Narrow band**: 16 px normal slope weakly identifiable → `ILL_CONDITIONED`.
- **Chains / acyclic graphs** are refused (`no_cycle_redundancy`): relational
  contradictions cannot be qualified without cycle redundancy; no false confidence.
- **Contradictory relations** with low redundancy are downweighted/rejected by the
  bounded mechanism; a chain cannot identify one at all (reported).
- **Multiplicative mismatch conditionally detectable** (locally-planar sky).
- Robust/rejection knobs are PROVISIONAL, research-only (not frozen contract).
- Wall-clock/RSS single-run, host-loaded.

---

## 7. Proposed scientific decision (Junior arbitrates)

All synthetic fidelity/evidence gates pass **only where the measurement band is
wide enough and the accepted graph carries cycle redundancy**; otherwise the
system returns an explicit refusal. The architecture is sparse and suitable for a
future compact diagnostic. The real pre-feather observations remain unavailable,
so the proposed outcome is **NEEDS RESEARCH**. **No production wiring is proposed
or performed.**
