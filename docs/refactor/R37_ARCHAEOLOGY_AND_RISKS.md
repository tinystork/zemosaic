# R37 — Archaeology and risks: inter-Cell gradient harmonization

**Mission:** ZM-ZEGRID-R37  
**Gate:** A — Junior archaeology, before any delegated prototype  
**Base inspected:** `beta` / `739f1e8d9f30b265dc36f62eaa21e377dc461473`  
**Date:** 2026-10-14  
**Recommendation:** **PROCEED TO AN ISOLATED PURE PROTOTYPE ONLY**. No production wiring is scientifically justified by the available real artefacts.

## 1. Baseline and evidence status

At Gate A start:

- branch `beta`;
- `HEAD == origin/beta == 739f1e8d9f30b265dc36f62eaa21e377dc461473`;
- clean worktree;
- R36 targeted gate: **31 passed**;
- no production/config/GUI/default change performed for R37.

Evidence labels used below:

- **STATIC:** source/contract inspection;
- **DYNAMIC:** command or read-only measurement actually run;
- **NOT RUN:** not executed and not inferred as a pass;
- **NOT AVAILABLE:** required source artefact absent.

## 2. The supplied run description does not match the local artefact

The only complete local Caldwell R34F result is:

`/home/tristan/caldwell11/zegrid_out/R34F_WIN/`

Its manifest/log state:

| Field | Mission description | Local R34F_WIN | Status |
|---|---:|---:|---|
| included frames | 816 | **653** | mismatch |
| canvas | 4117×4377 | **4117×4075** | mismatch |
| layout | 6×4 | **6×4** | match |
| mode | auto→per_cell | **auto→per_cell**, 8 trigger Cells | match |
| relations | 38 | **38/38** | match |
| offset residual RMS RGB | ~33.90 / 30.43 / 34.18 | **15.756 / 15.172 / 6.876** | mismatch |
| duration | ~10945 s | **8499.144 s** + finalize 30.356 s | mismatch |

A recursive search found no local `zegrid_manifest.json` for an 816-frame complete run. The mission's exact 816-frame product is therefore **NOT AVAILABLE**. No value from that claimed run is converted into a measured fact. Dynamic Gate A measurements below apply only to the local 653-frame `R34F_WIN` product.

## 3. Current R34 path

### 3.1 Per-Cell data lifecycle — STATIC

Production flow:

1. each worker builds an aligned per-Cell cache;
2. R34b computes a local gauge when `cell_gauge_mode=per_cell`;
3. canonical stacking returns PATCH-level science, valid mask and surviving-count planes;
4. the worker deletes the aligned cache;
5. the parent temporarily retains each complete Cell's PATCH planes in `result_by_cid`;
6. `_assemble_per_cell_feathered()` constructs `CellPatch` objects;
7. R34c measures adjacent overlaps and solves scalar RGB offsets;
8. R34d applies offsets and feather-blends;
9. only the assembled science/aesthetic/coverage and a compact manifest are written.

The useful pairwise data exists **during assembly only**. It is not persisted after the run.

### 3.2 Observation domain — STATIC

- `HALO_PX=8`; adjacent patches overlap over a band about **16 px wide in the normal direction**.
- The tangent direction spans most of a Cell edge (roughly 680–1000 px on the local 6×4 canvas).
- R34 observes shared-valid, finite pixels in both Cells.
- Current graph weight is raw shared-support count. It does not include residual variance, depth balance, tangent/normal leverage, conditioning or contamination confidence.
- The final feather width and the measurement domain happen to share the same halo today, but they are conceptually different and must be separated in R37.

Consequences:

- tangent variation can be observed over a long baseline;
- normal slope is mathematically present through the 16 px width but has weak leverage;
- a one-dimensional boundary line cannot identify all three plane parameters;
- corners can involve four Cell patches and contradictory local constraints;
- fragmented support, bright stars, IFN/nebulosity, depth changes and PSF differences can dominate a narrow band.

### 3.3 What the final files preserve — STATIC + DYNAMIC

Local files:

- science FITS `(3,4075,4117)`, float32, raw assembled R34 reference;
- aesthetic FITS, after DBE/RGB equalization/hole filling;
- coverage FITS, float32 stack-depth proxy;
- manifest with scalar offsets and aggregate RMS.

The science exporter replaces non-finite values by zero in the serialized buffer (`ZNFILL=0`, 6,097,458 replacements). Coverage can identify globally uncovered pixels, but the final product no longer contains:

- the two separate Cell values in an overlap;
- per-edge observed deltas;
- per-edge weights or robust noise;
- training/holdout samples;
- edge residuals or covariance;
- cycle residuals;
- owner/feather contributions per Cell;
- the original pairwise valid masks.

Therefore the final FITS can quantify a residual boundary signature, but **cannot support a scientifically justified real-data fit of a new inter-Cell model**. Reconstructing pre-feather observations from it would be fictitious.

## 4. Residual-boundary witness on the local R34F product — DYNAMIC

Read-only script/artifacts:

- `.a2a-reports/ZM-ZEGRID-R37-gateA/analyze_r34f_boundaries.py`
- `.a2a-reports/ZM-ZEGRID-R37-gateA/r34f_boundary_profiles.json`
- `.a2a-reports/ZM-ZEGRID-R37-gateA/boundary_vs_control.json`

Method:

- exact 6×4 core boundaries from the integer ZeGrid layout;
- 38 Cell relations × 3 channels = 114 profiles;
- coverage>0 mask;
- robust paired differences at symmetric distances 1/4/8/16/32 px;
- tangent trends, pure-side normal slopes and robust local-noise proxies;
- comparison against internal, non-boundary control cuts.

Science FITS results:

| Metric | Cell boundaries | Internal controls | Observation |
|---|---:|---:|---|
| median `|Δ|` at ±1 px | **0.946 ADU** | 0.576 ADU | boundary/control 1.64× |
| p90 `|Δ|` at ±1 px | 1.852 ADU | 1.860 ADU | feather suppresses the immediate step |
| median `|Δ|` at ±16 px | **7.652 ADU** | 1.546 ADU | boundary/control **4.95×** |
| p90 `|Δ|` at ±16 px | **21.575 ADU** | 5.188 ADU | broad residual structure remains |
| Mann–Whitney, boundary > control, ±16 px | `p≈8.94e-23` | — | strong association with Cell boundaries |
| median local noise proxy | 4.877 ADU | — | residual scale is not negligible |

Worst robust ±16 px residuals reach ~31.2 ADU. Vertical edges are somewhat worse than horizontal edges (median 9.22 vs 7.02 ADU). The aesthetic branch reduces the median ±16 px residual to 5.84 ADU, but is explicitly non-photometrically-neutral and cannot qualify science.

Interpretation:

- R34 feathering works locally: median `|Δ1/Δ16|≈0.093`.
- A broader Cell-correlated mismatch remains outside the feather transition.
- Weak correlation with coverage imbalance (`corr≈0.10`) does not support a simple depth-only explanation.
- Tangent trends and normal-slope jumps exist, but the final blended image cannot distinguish instrumental planes from real sky structure.
- This is evidence for studying a plan model, **not evidence that a plan correction is scientifically valid**.

No exact pre-R34 product for the same 653-frame run exists locally; before/after real-data profiling is **NOT RUN**.

## 5. Current scalar solver robustness — STATIC + DYNAMIC

Current R34c:

- scalar additive offset per Cell/channel;
- central Cell pinned to zero;
- dense weighted normal equations `AᵀWA` + `np.linalg.solve`;
- edge pixel outliers are clipped before a robust median;
- no graph-level robust loss or edge rejection;
- no persisted per-edge diagnostics;
- no rank/condition/uncertainty report;
- no cycle-consistency output;
- connectedness is checked, but a connected graph can still be weakly constrained;
- production's disconnected policy leaves unreachable Cells at offset zero and records them.

For the ideal unweighted 6×4 scalar grid:

- 24 nodes, 38 edges, 23 free offsets;
- 15 independent cycle redundancies;
- rank 23;
- `cond(A)≈7.80`, `cond(AᵀA)≈60.8`.

Actual weighted conditioning cannot be reconstructed because R34 did not persist weights/deltas. The manifest's large offsets span roughly 2252/2892/2761 ADU across RGB, but a plane fit explains only R²≈0.02/0.05/0.08. Large amplitudes alone therefore neither prove an error nor support a smooth-gradient interpretation.

## 6. Plan identifiability

Candidate:

`P_i(x,y,c) = a_i,c + b_i,c*x + d_i,c*y`, with global normalized coordinates.

### 6.1 Boundary-line-only observations — DYNAMIC synthetic matrix

With only two tangent samples on the centre line of each edge, the ideal 6×4 design has:

- 69 free parameters/channel after anchoring;
- 76 observations;
- rank 61;
- nullity 8.

A line alone is insufficient.

### 6.2 Full 16 px overlap — DYNAMIC synthetic matrix

Using two normal positions and two tangent positions per edge makes the ideal 6×4 system full rank, but highly anisotropic:

- rank 69/69;
- `cond(A)≈2430`;
- `cond(AᵀA)≈5.9×10⁶`.

For an ideal 10×10 grid, `cond(A)≈4506`; 1×N/N×1 layouts are also full rank only through the tiny normal baseline and remain poorly conditioned (`cond(A)≈2100`). Real masks and fragmented support can be worse.

Conclusion: the normal slope is technically observable in a clean 16 px band, but is not automatically trustworthy. R37 must use direct sparse least squares (not normal equations), uncertainty/leverage diagnostics, regularization and an explicit `ILL_CONDITIONED` refusal.

Artifact: `.a2a-reports/ZM-ZEGRID-R37-gateA/graph_conditioning.json`.

## 7. Scalability

Approximate grid with 10,000 Cells (~19,800 relations):

| Model | Unknowns/channel | Dense design | Dense normal | Estimated dense temporaries |
|---|---:|---:|---:|---:|
| scalar offset | 9,999 | 1.48 GiB | 0.74 GiB | >3.7 GiB |
| 3-parameter plan | 29,997 | 4.43 GiB | 6.70 GiB | **>15.5 GiB** |

Dense cubic solves are unacceptable. The project already depends on SciPy. A prototype may use `scipy.sparse` + `lsqr/lsmr`; production-scale cost must be driven by Cells/relations/compact statistics, never `N_frames×canvas` or all-pairs Cells.

## 8. External methodological references — STATIC

Read-only snapshots were archived under `.a2a-reports/ZM-ZEGRID-R37-gateA/references/`.

- **Montage** assumes calibrated brightness and models non-sky low-frequency discrepancies with offsets/slopes. It uses overlap differences, planar approximations and global/iterative consistency. Montage explicitly warns that generalized matching cannot distinguish sky from background when the assumptions fail.
- **PhotometricMosaic** describes photometric scale matching plus a surface spline for relative gradients. It supports B3 as a research reference, not as evidence that a spline is identifiable here.

No third-party code was copied. README currently credits Tinystork, J.A.R.V.I.S. and Juan Conejero/PixInsight integration methods. If a future R37 implementation adopts these methods, propose explicit methodological acknowledgements for John Murphy/PhotometricMosaic and Caltech/IPAC Montage, with URLs; no licence conclusion is needed for citation-only inspiration.

## 9. Risk matrix

| Risk | Severity | Evidence | Gate / mitigation |
|---|---|---|---|
| Absorb real common sky/IFN | HIGH | scientific identifiability limit | same-pixel pair differences, protected-scene tests, held-out metrics, refuse weak model |
| 16 px normal slope weakly constrained | HIGH | cond(A)≈2430 ideal | covariance/leverage gate, slope regularization, `ILL_CONDITIONED` |
| Exact 816-frame artefact absent | HIGH | local search | no real-data qualification claim; future compact diagnostic run required |
| Final FITS cannot recover Cell pairs | HIGH | lifecycle inspection | collect compact observations while patches coexist |
| Graph connected but weak/contradictory | HIGH | current diagnostics absent | rank, condition, uncertainty, cycles, robust edge IRLS |
| Stars/PSF/depth contaminate differences | HIGH | narrow overlap + heterogeneous stacks | robust masks/weights, block holdout, noise/depth diagnostics |
| Additive model used on gain/spectral mismatch | HIGH | model mismatch | detect level-dependent residual; return `INSUFFICIENT_EVIDENCE`/upstream issue |
| Four-Cell corners conflict | MEDIUM | feather geometry | cycle/corner holdout diagnostics; no local silent override |
| Layout sensitivity | MEDIUM | local references differ by layout | physical synthetic truth, order/layout matrix; no false bit-equality demand |
| Dense solver at scale | HIGH | >15.5 GiB plan estimate | sparse design only; O(E) compact observations |
| Aesthetic DBE hides/creates evidence | MEDIUM | finishing applied | science FITS only for qualification |
| Large offsets automatically rejected | MEDIUM | amplitudes up to ~2.9k ADU | judge by residuals/uncertainty/cycles, not amplitude alone |

## 10. Recommended Gate-B architecture

Proceed only with a pure, un-wired prototype:

1. keep existing M0 scalar-offset baseline;
2. represent each pair using same-global-pixel differences over a 2-D band;
3. use global normalized `(x,y)` coordinates;
4. fit robust per-edge plane statistics with deterministic spatial train/holdout blocks;
5. retain compact sufficient statistics (normal matrix/rhs, robust scale, covariance/leverage, counts, masks/bins), not pixels;
6. solve globally with a sparse robust/regularized system;
7. anchor deterministically and report rank/condition/uncertainty/cycles;
8. select `OFFSET_ONLY`, `ROBUST_PLAN`, or explicit refusal from predeclared held-out and fidelity gates;
9. study nonlinear/spline models in documentation only;
10. import the prototype nowhere from production modules.

A future-run diagnostic artefact should contain, per edge/channel: endpoint IDs/orientation, normalized coordinate definition, support/effective samples, tangent/normal occupancy, train/holdout counts, robust noise, fitted coefficients, covariance/condition/leverage, held-out residual metrics, rejection fraction, cycle contribution and refusal reason. It can be produced while patches coexist with O(edges×channels×constant) memory and no extra source-image pass.

## 11. Gate-A decision

**Gate A: PASS for an isolated synthetic prototype, FAIL for real-data qualification or production integration.**

The residual is real enough to investigate, but the available final files cannot prove which correction preserves astronomy. Gate B may proceed under `R37_BACKGROUND_MODEL_CONTRACT.md`. Even a successful synthetic result will normally conclude **NEEDS RESEARCH** until a future authorized run emits compact pre-feather edge diagnostics and supplies an untouched real validation set.
