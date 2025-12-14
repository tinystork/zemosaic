# Mission: TwoPass definitive diagnostics (logs only, DEBUG-only)

Objective
Provide a complete, objective diagnostic of the TwoPass normalization issue
(seams between tiles) using DEBUG-only logs, without modifying any algorithm,
parameters, or outputs.

This mission must allow answering definitively:
- Is the dataset too weak / overlap too small?
- Is gain-only normalization insufficient because offset dominates?
- Is the TwoPass math correct but ineffective by design?
- Or is there a pipeline bug (gain not applied, bad masks, reprojection mismatch)?

Scope
- File: zemosaic_worker.py ONLY
- Functions involved:
  - _apply_two_pass_coverage_renorm_if_requested(...)
  - run_second_pass_coverage_renorm(...)

Hard constraints
- DO NOT change any math, normalization logic, gains, sigma, clip, reprojection.
- DO NOT change outputs.
- Logs and diagnostic computations ONLY.
- All diagnostics MUST be wrapped in:
    if logger.isEnabledFor(logging.DEBUG):

--------------------------------------------------
A) [x] GLOBAL CONTEXT LOGS (ONCE)
--------------------------------------------------
Emit one block at TwoPass entry:

[TwoPassCfg]
- sigma
- clip_min / clip_max
- number of tiles
- output shape (H,W)
- dtype
- fallback_used (bool)
- downsample factor used for diagnostics (DS)

[TwoPassCoverage]
- coverage min / mean / median / max
- fraction of pixels with coverage > 0
- bounding box of non-zero coverage

--------------------------------------------------
B) [x] PER-TILE BASE STATS (BEFORE CORRECTION)
--------------------------------------------------
For each tile idx, BEFORE applying gain:

[TwoPassTileStats]
- idx
- valid_frac (finite RGB & alpha>0 & coverage>0 if available)
- median RGB
- MAD RGB (or IQR if already available)
- mean RGB (optional)

Purpose:
Detect weak tiles, noisy background, or insufficient valid pixels.

--------------------------------------------------
C) [x] OVERLAP-BASED DIAGNOSTICS (KEY PART)
--------------------------------------------------
Compute diagnostics ONLY on overlap region with FIRST-PASS mosaic.

Method:
- Use luminance only (L = 0.25R + 0.5G + 0.25B)
- Use LOW-RES reprojection (DS = 8 or 16) for speed
- overlap_mask = finite(tile_proj) & finite(ref_proj) & (coverage>0)

For each tile idx:

[TwoPassOverlap]
- idx
- overlap_frac
- delta_med      = median(tile - ref)
- abs_delta_med  = median(|tile - ref|)
- delta_mad
- slope (a) and intercept (b) from simple regression ref -> tile
- correlation r

Purpose:
- delta_med != 0 → offset dominates
- slope != 1 → gain mismatch
- low overlap_frac → dataset limitation
- low r → structure mismatch / bad overlap

--------------------------------------------------
D) [x] APPLY CHECK (GAIN EFFECTIVENESS)
--------------------------------------------------
After gain application, recompute overlap median delta:

[TwoPassApply]
- idx
- delta_med_pre
- delta_med_post

Purpose:
- If delta does not improve → gain-only insufficient or bug
- If unchanged everywhere → gain not applied or masked out

--------------------------------------------------
E) [x] GLOBAL SUMMARY (VERDICT HELPERS)
--------------------------------------------------
After all tiles processed:

[TwoPassWorst]
- top 5 tiles sorted by abs_delta_med
- for each: idx, overlap_frac, abs_delta_med, delta_med, slope, intercept

[TwoPassScore]
- global_overlap_mean
- global_abs_delta_med (weighted by overlap_frac)

--------------------------------------------------
F) [x] SANITY / MASK CHECKS
--------------------------------------------------
Log ONCE if any mismatch detected:
- RGB / alpha / coverage shape mismatch
- coverage threshold used (>0 or >eps)
- fraction of pixels rejected by coverage mask
- fraction of NaNs after reprojection

--------------------------------------------------
Output rules
- DEBUG-only logs.
- INFO-level behavior must remain unchanged.
- No additional files saved.
