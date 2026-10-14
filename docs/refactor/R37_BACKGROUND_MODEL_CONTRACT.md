# R37 — Background model contract (Gate B, pure prototype only)

**Status:** frozen input contract for the isolated prototype.
**Production wiring:** forbidden.
**Default/config/GUI change:** forbidden.
**Decision authority:** Junior after independent Nono review.

## 1. Scope

Compare:

- `OFFSET_ONLY` — M0, current R34 scalar additive baseline;
- `ROBUST_PLAN` — candidate M1, per-Cell/channel additive plane;
- `NONLINEAR_CANDIDATE` — documentation/research only; no spline engine.

The prototype operates only on synthetic `CellPatch`-like arrays or compact boundary observations. It must not import itself into `zemosaic_zegrid_mode.py`, `cell_feather.py`, GUI/config/dispatcher or canonical SCI-05 modules.

## 2. Model

For Cell `i`, channel `c`:

`P_i(x,y,c) = a_i,c + b_i,c*x + d_i,c*y`

where `x,y ∈ [-1,1]` are global canvas coordinates. Corrections are additive only. No gain, colour matrix, multiplicative normalization or hidden rescaling is permitted.

The central deterministic anchor Cell has `(a,b,d)=(0,0,0)`. This fixes gauge freedom; it does not prove that slopes are identifiable.

## 3. Observation contract

Only same-global-pixel pair differences are admissible. A sample is usable iff both Cell values are valid and finite and any explicit protected-object mask admits it.

Feather width and measurement width are separate parameters. Synthetic tests must cover measurement bands of 8/16/32/64 px while the assembly feather may remain 8 px.

Each edge/channel observation records:

- Cell IDs and orientation;
- global normalized sample coordinates;
- train/holdout assignment;
- raw and effective sample counts;
- tangent and normal occupancy;
- robust residual scale and rejection fraction;
- information matrix/rhs or equivalent compact statistics;
- rank, leverage and condition diagnostics;
- held-out residual metrics;
- explicit refusal reason.

Raw support count alone is not an adequate confidence weight. Confidence must include residual noise and geometric leverage, with a deterministic cap so one huge edge cannot dominate the graph.

## 4. Robust fitting and regularization

- Use deterministic spatial blocks, not a random pixel split: **75% train / 25% holdout**, assigned by fixed block parity.
- Robust loss: Huber with fixed transition `1.345 × robust_sigma` (Gaussian-efficiency convention), or an equivalently documented fixed robust loss.
- Use direct sparse least squares (`scipy.sparse` + LSQR/LSMR or equivalent), never dense normal equations for the global solve.
- Penalize slopes, not offsets. The fixed candidate regularization ladder is `{0, 1e-4, 1e-3, 1e-2, 1e-1, 1}` in normalized-coordinate units.
- Select regularization using training/holdout results only; never tune after inspecting protected-scene truth labels. Tie-break toward the stronger/simple correction.
- Report data-only rank/condition separately from regularized condition. Regularization must not disguise absent evidence.

## 5. Decision states

The estimator must return one of:

- `OFFSET_ONLY` — M0 retained;
- `ROBUST_PLAN` — all M1 gates pass;
- `INSUFFICIENT_EVIDENCE` — support/holdout/leverage inadequate;
- `ILL_CONDITIONED` — rank/condition/uncertainty inadequate;
- `MODEL_MISMATCH` — additive plane fails (e.g. level-dependent/multiplicative/nonlinear residual);
- `DISCONNECTED` — unsupported graph component, no fabricated bridge.

`NONLINEAR_CANDIDATE` may appear only as a research recommendation, never an executable winner.

## 6. Predeclared acceptance gates

Let `σΔ` be the known or robustly estimated standard deviation of a same-pixel Cell-difference sample. Metrics are per channel unless stated.

### 6.1 Evidence and numerics

All are mandatory for `ROBUST_PLAN`:

1. data-only design full rank after anchoring;
2. whitened `cond(A) ≤ 1e4`; do not judge only `cond(AᵀA)`;
3. at least 256 effective train samples per used edge/channel;
4. at least 8 occupied tangent blocks and 3 occupied normal blocks;
5. finite coefficient uncertainties;
6. predicted correction uncertainty at every Cell corner ≤ `0.5 σΔ` **or** ≤50% of the proposed correction magnitude there (the absolute criterion handles near-zero corrections);
7. graph connected for every Cell to be corrected; unsupported components are refused;
8. deterministic output and invariance to Cell/edge input order (coefficients equal within `1e-10` in float64 on the same compact statistics).

### 6.2 Held-out improvement

Compared with `OFFSET_ONLY`, all are mandatory:

1. global held-out robust RMS: `R_plan ≤ 0.80 × R_offset`;
2. absolute held-out RMS: `R_plan ≤ 1.25 × σΔ`;
3. per-edge held-out p90 must not worsen by more than 5%;
4. improvement must appear in both edge orientations when both exist;
5. train improvement without held-out improvement is a refusal.

These thresholds demand a material gain beyond noise while allowing realistic estimation error. They are frozen before prototype results.

### 6.3 Synthetic truth and protected astronomy

On known-truth corpora:

1. noise-free offset/plane recovery max error ≤ `1e-8` in float64;
2. noisy artificial-background correction RMSE ≤ `0.25 σ_pixel`, p99 ≤ `0.75 σ_pixel`;
3. common celestial-gradient amplitude ratio after correction in `[0.99, 1.01]`;
4. diffuse/IFN protected-region robust bias ≤ `0.10 σ_pixel` and amplitude ratio in `[0.99,1.01]`;
5. background-subtracted stellar aperture flux median relative bias ≤0.1%, p95 ≤0.5%;
6. uniform identity corpus: noise-free correction exactly zero within `1e-10`; noisy identity must retain M0 unless every evidence/holdout gate independently passes;
7. NaN/support/hole locations and array dimensions exactly preserved;
8. coverage and supplied sample weights unchanged by correction application.

Any violation is `NO GO` for M1 regardless of visual seam reduction.

## 7. Required synthetic matrix

At minimum:

- flat identity;
- known offsets;
- known planes;
- common celestial gradient with no instrumental mismatch;
- common diffuse nebulosity/IFN crossing boundaries;
- stars/bright clusters in observation bands;
- independent instrumental gradients;
- nonlinear background not representable by a plane;
- outlier pixels and contradictory relations;
- low/high SNR;
- weak, fragmented and disconnected support;
- heterogeneous nights/cameras/noise/exposure/gain/transparency/spectral responses;
- `1×N`, `N×1`, `2×2`, `6×4`, and larger sparse grids;
- alternative layouts over the same physical synthetic truth.

Multiplicative or spectral mismatch must be detected as `MODEL_MISMATCH`/upstream calibration, not repaired by additive planes.

## 8. Model-comparison metrics

Report separately:

- residual level step;
- normal-slope mismatch;
- tangent residual trend;
- truth-background RMSE/bias;
- protected diffuse-signal bias;
- stellar aperture-flux bias;
- noise and coverage statistics;
- coefficient amplitude and uncertainty;
- rank, condition, leverage and cycle residuals;
- CPU time and peak memory;
- asymptotic cells/edges/statistics counts.

Do not combine them into one visual score.

## 9. Scalability contract

- Complexity target: sparse solve based on Cells, neighbor relations and compact statistics.
- No all-pairs Cell graph.
- No additional general reprojection.
- No `N_frames×canvas` accumulation.
- No generated 10,000-FITS benchmark.
- Include a pure sparse graph witness up to 10,000 Cells using synthetic compact observations; it may test construction/solve scaling without images.
- If the prototype densifies a 10,000-Cell design, it fails the architecture gate.

## 10. Gate-B outcome policy

- `GO PROPOSAL`: all synthetic fidelity/evidence/scaling gates pass **and** architecture is suitable for future compact real diagnostics. This does not authorize production.
- `NEEDS RESEARCH`: promising synthetic result but real pre-feather diagnostics or identifiability evidence missing.
- `NO GO`: protected signal is biased, model is systematically ill-conditioned, or cost is unjustified.

The canonical final R34F product is available at
`/home/tristan/caldwell11/zegrid_out/R34F/` and its manifest contains 653 frames.
The separate pre-feather Cell observations required by this contract are not
persisted there. Gate B therefore cannot by itself qualify a production change;
its strongest scientifically honest conclusion remains `NEEDS RESEARCH` plus a
future diagnostic-artifact proposal. This factual correction does not change any
frozen threshold.
