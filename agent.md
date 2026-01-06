# Mission: Implement TRUE PixInsight-like Winsorized Sigma Clipping (WSC) with CPU/GPU parity (no regressions)
(READ FIRST — NON-NEGOTIABLE)

### 0) Hard constraints (do not argue, do not “improve”)
- **NO GUI CHANGES**: do not touch PySide/Tk code, do not touch locales, do not touch settings widgets.
  Forbidden files (non-exhaustive): `zemosaic_gui_qt.py`, `zemosaic_filter_gui_qt.py`, any `*_gui*.py`, `locales/*.json`.
- **NO BEHAVIOR CHANGE OUTSIDE WSC**: do not modify other rejection modes, weighting logic, crop logic, master-tile quality gate, SDS/grid mode behavior.
- **DO NOT CHANGE batch-size semantics** (CRITICAL): the current behavior for “batch size = 0” and “batch size > 1” must remain EXACTLY as-is.
- **NO “close enough” GPU/CPU**: for Winsorized Sigma Clip, CPU and GPU outputs must be **identical**, not “similar”.
- **NO APPROXIMATION**: PixInsight-like WSC must NOT use frame-splitting (`frames_per_pass`) or any per-frame streaming approximation.
  Spatial chunking is allowed, but **each pixel must see all N frames**.

### 1) Scope of allowed code changes (keep it surgical)
Allowed files (expected):
- `zemosaic_align_stack.py` (CPU wiring + shared core)
- `zemosaic_align_stack_gpu.py` (GPU wiring, but must call the same shared core)
- `zemosaic_config.py` / settings plumbing ONLY if needed for hidden defaults / env switches
- `tests/` (new tests only; do not rewrite unrelated tests)

Forbidden changes:
- No renaming public APIs used by worker/modes.
- No refactor sweeping “cleanup”, “style changes”, or “performance refactors” outside WSC.
- No new external dependencies.

### 2) Single-source-of-truth algorithm (MANDATORY)
- Implement **ONE** backend-agnostic core: `wsc_pixinsight_core(xp, X_block, ...)` where xp = numpy or cupy.
- CPU path and GPU path MUST call this same function.
- DO NOT implement separate CPU vs GPU versions that can drift.

### 3) Algorithm definition is locked (do not reinterpret)
- Init: median + MAD, sigma0 = 1.4826 * MAD (sigma floor = 1e-10).
- Iterate: compute bounds → clip (winsorize) → update winsorized mean → update winsorized sigma (Huber/winsorized RMS).
- Output: winsorized mean (not masked sigma clip).
- NaN/inf: treated as missing samples; do not poison medians/means.

### 4) Parity enforcement (STRICT)
- Reference = CPU output after float32 cast.
- GPU output after float32 cast must satisfy:
  - `max_abs_diff == 0.0` on deterministic small tests (seeded RNG).
- If GPU cannot guarantee parity for WSC (any reason: precision, nondeterminism, edge cases):
  - **fallback to CPU for WSC only**, with a clear log line.
  - Do NOT disable GPU globally; keep GPU for other operations intact.
### Parity definition (do not reinterpret)
We distinguish:
- STRICT_PARITY (default): GPU may be used only if it matches CPU within 0 ULP on the parity test set.
  Otherwise WSC falls back to CPU (WSC only) with a clear log line.
- NUMERIC_PARITY (opt-in): GPU must match CPU within <= 1 ULP (float32) OR max_abs_diff <= 2e-7.
  Enabled only via env var: ZEMOSAIC_WSC_PARITY=NUMERIC
“Any GPU exception (OOM, kernel failure, unsupported op) must trigger CPU fallback for WSC only, with a single log line.”

### 5) Keep legacy behavior available (recommended safety hatch)
- Add an env/config switch (no GUI):
  - `ZEMOSAIC_WSC_IMPL=pixinsight|legacy_quantile`
- Default: `pixinsight`
- `legacy_quantile` must preserve the old behavior unchanged.

### 6) Required verification before marking done
You must provide evidence in code/tests that:
- Existing test suite still passes.
- New tests added:
  1) cosmic ray suppression
  2) dead pixel suppression
  3) faint diffuse preservation (IFN-like offset not crushed)
  4) CPU vs GPU strict parity (`max_abs_diff == 0.0`) when GPU is available

### 7) Logging (minimal, no UI)
- One log line per stack for WSC:
  - impl used (pixinsight vs legacy)
  - sigma_low/high, max_iters, iters_used
  - huber enabled/disabled
  - fraction clipped low/high
- No noisy per-iteration spam unless debug flag enabled.

### 8) Don’t break current GPU success (regression guard)
- Do not change GPU detection/selection logic that was recently fixed.
- Do not alter the “GPU enabled” codepath outside WSC unless strictly necessary.
- If you must touch it, isolate the change and explain why in the patch notes.

### 9) Output expectations
- Produce a small patch with minimal file changes.
- Include a short summary of what changed + how to run the parity test locally.

## Goal (non-negotiable)
Replace ZeMosaic’s current “winsorized_sigma_clip” (quantile-ish winsor + std-ish sigma) with a TRUE PixInsight-like
Winsorized Sigma Clipping that preserves faint diffuse signal (IFN) and behaves robustly against outliers.

**Critical constraints**
1) **NO GUI changes** (PySide + Tk must remain untouched).
2) **NO regressions** in any other rejection/stack mode (kappa-sigma, linear fit, none, grid mode, SDS, “I’m using master tiles”).
3) **NO approximation differences between CPU and GPU**:
   - Same math, same parameters, same defaults, same edge-case behavior.
   - CPU is the reference; GPU must match CPU for defined tests (see Acceptance).
4) **Batch size semantics must NOT change** (IMPORTANT: “batch size = 0” and “batch size > 1” behavior must remain EXACTLY as today).
5) **PixInsight WSC must NOT be implemented as quantile/percentile winsorization.**
   - No cp/np percentile, quantile, partition-based tail clipping as the core behavior (except if explicitly kept for legacy mode only).

## Scope: where WSC is used (must be consistent everywhere)
WSC can be executed through multiple codepaths. All must route to the same core algorithm:
- Phase 4.5 / group stacking: `zemosaic_worker.py` → `zemosaic_align_stack.stack_winsorized_sigma_clip()`
- Phase 3 GPU stacker: `zemosaic_align_stack_gpu.py` (reject algo = winsorized_sigma_clip)
- CPU stack core fallback path: `zemosaic_align_stack.stack_aligned_images()` when rejection_algorithm is winsorized_sigma_clip

**Requirement:** No matter which phase/mode triggers WSC, the output must be consistent.

## PixInsight-like WSC: exact algorithm specification (lock this down)
We implement a per-pixel iterative winsorization procedure:

Let X be a stack block shaped (N, H, W, C) in float32 (or float64 internally), N>=1.

### A) Validity / NaN policy
- Treat non-finite values (NaN/inf) as missing samples.
- Missing samples do NOT contribute to median/MAD/mean/sigma.
- If a pixel has <2 valid samples: output is that sample (or NaN if none).

### B) Initialization (robust)
Compute per-pixel, per-channel:
- `m0 = median(X)` over axis=0 (NumPy/CuPy median definition: for even N, average of the two middle values).
- `mad0 = median(|X - m0|)`
- `sigma0 = 1.4826 * mad0`
- If sigma0 == 0 → set sigma0 = 1e-10 (avoid div/zero; do NOT explode thresholds)

### C) Iterative winsorization (Huber/Winsorized estimates)
For i in 1..max_iters:
1) bounds:
   - `lo = m - sigma_low  * sigma`
   - `hi = m + sigma_high * sigma`
2) winsorize (clamp outliers):
   - `Xw = clip(X, lo, hi)` (only for valid samples)
3) update location (winsorized mean):
   - unweighted: `m_new = mean(Xw)`
   - weighted (if weights provided): `m_new = sum(w*Xw)/sum(w)`
4) update scale (winsorized sigma = Huber scale estimate):
   - residual: `r = Xw - m_new`
   - unweighted: `sigma_new = sqrt(mean(r^2))`
   - weighted: `sigma_new = sqrt(sum(w*r^2)/sum(w))`
   - sigma_new floor at 1e-10
   Huber scale update (exact):
    u = r / sigma
    w = 1                      if |u| <= c
    w = c / |u|                if |u| >  c
    sigma_new = sqrt( sum(w * r^2) / sum(w) )
    (Weights w above are Huber IRLS weights; do not use alternative Huber formulas.)

Weights scope (locked):
- WSC supports only per-frame weights shape (N,) or (N,1,1[,1]) broadcasting.
- Median/MAD are always unweighted.
- Weights apply only to the winsorized mean and winsorized sigma updates, and the final mean.
- If weights are missing or invalid, treat as uniform weights (do not error).

5) convergence:
   - stop if `max(|m_new-m|) <= eps_m` AND `max(|sigma_new-sigma|) <= eps_s`
   - default eps: `eps_m = 5e-4 * max(1, |m|)` and `eps_s = 5e-4 * max(1, |sigma|)` (relative-ish)
   - also stop if bounds no longer change (stable lo/hi), to avoid useless iterations.

### D) Final integration output
Return the final **winsorized mean** (NOT masked sigma clip):
- If weights: weighted mean of the last Xw
- Else: mean of the last Xw
Output dtype: float32.

### E) No frame-splitting approximation
PixInsight WSC is per-pixel across the full N samples. Therefore:
- **Do NOT implement WSC by “frames_per_pass” splitting** (streaming-by-frames is an approximation and changes results).
- WSC may use **spatial chunking only** (rows/tiles), but each pixel must see all N frames.

## Implementation plan (idiot-proof)
### 1) Single shared core (to prevent CPU/GPU drift)
Create ONE backend-agnostic implementation:
- `wsc_pixinsight_core(xp, X_block, sigma_low, sigma_high, max_iters, eps, weights_block=None, ...)`
Where `xp` is `numpy` or `cupy`.

**Hard rule:** CPU WSC and GPU WSC must call this same function.
No duplicated “similar” implementations.

### 2) Wiring: keep existing APIs stable
- Keep signatures of:
  - `zemosaic_align_stack.stack_winsorized_sigma_clip(...)`
  - `zemosaic_align_stack_gpu.gpu_stack_from_arrays / gpu_stack_from_paths(...)`
- Do not remove existing rejection algorithms or change their defaults.
- Only change behavior when `reject_algo == winsorized_sigma_clip`.

### 3) Legacy compatibility (optional but recommended)
To avoid surprising users who relied on old quantile behavior:
- Add hidden switch (config or env var):
  - `ZEMOSAIC_WSC_IMPL=pixinsight|legacy_quantile`
- Default: `pixinsight`
- If `legacy_quantile`, keep old behavior intact.

### 4) Determinism / parity requirements
- Use the same dtype strategy in both CPU and GPU:
  - Compute m/sigma in float64 (recommended) OR float32, but must match across backends.
  - Final output must be float32.
- Any GPU nondeterminism is unacceptable for this mode:
  - If parity cannot be guaranteed, GPU must **auto-fallback to CPU** for WSC only (with a clear log line),
    while leaving GPU enabled for other algorithms.

## Tests (must prevent regressions)
Add tests that codify “no approximation”:
1) **Outlier suppression**: huge cosmic ray pixel → output near baseline.
2) **Faint diffuse preservation**: add weak constant IFN-like signal to all frames → WSC must NOT suppress it vs kappa-sigma (SNR check).
3) **CPU vs GPU parity (strict)**:
   - On a small deterministic stack (seeded RNG), compare CPU and GPU WSC output AFTER float32 cast:
     - require bitwise equality OR max_abs_diff == 0.0
   - If GPU not available, test is skipped (but still runs in GPU-enabled CI/dev machines).
4) **Callsite consistency**:
   - Ensure Phase 4.5 WSC and Phase 3 GPU WSC route to the same core (smoke test / import check).

## Acceptance criteria (strict)
- WSC no longer crushes faint IFN compared to kappa-sigma on representative user data (visual + basic stats).
- CPU and GPU outputs match exactly on defined parity tests (no “close enough”).
- No change in outputs for other rejection modes.
- No GUI changes, no batch-size behavior changes.

NaN/Inf handling (exact):
- Build valid mask V = isfinite(X)
- For median/MAD: operate on compacted values per pixel if feasible; otherwise use masked large-value sentinel only if it cannot affect selection (documented).
- For mean/sigma updates: compute sums over V only (Nvalid-aware).

Core location (locked):
Place wsc_pixinsight_core in `core/robust_rejection.py` (new file allowed), imported by both CPU and GPU stackers.
Do not import GPU module from CPU module or vice versa.