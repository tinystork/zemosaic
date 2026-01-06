# Mission: Fix noise_fwhm weighting robustness (NaN-heavy frames) + ensure GPU applies quality weights

## Context
We observe repeated warnings during MasterTile stacking when using `stack_weight_method = "noise_fwhm"`:

- `weight_fwhm_bkg2d_error`: "All boxes contain <= N good pixels..."
- `weight_fwhm_warn_all_fwhm_are_infinite`
- Stack falls back to uniform weights → effectively no weighting.

Root cause:
Aligned/warped frames often contain large NaN/Inf regions (outside overlap). `photutils.Background2D` rejects all boxes when too many pixels are masked/non-finite, even with high exclude_percentile.
Also: GPU stack path currently may discard quality weights when radial weighting is disabled, causing weight methods to be ignored on GPU even if computed.

## Goals (must-have)
1. Make `noise_fwhm` weighting robust on NaN/Inf-heavy aligned frames:
   - Avoid Background2D fatal failures whenever reasonable.
   - Compute finite, meaningful FWHM weights when there is usable data.
   - If not possible, gracefully degrade to current behavior (infinite FWHM → uniform weights) WITHOUT crashing.

2. Ensure GPU stacking path actually applies quality weights (noise_variance / noise_fwhm) when selected:
   - CPU and GPU must behave consistently for the same settings.
   - No “silent ignore” of weights on GPU when radial weighting is off.

3. Preserve existing behavior for all other modes:
   - No GUI changes.
   - No changes to stacking algorithms (WSC / kappa-sigma / mean), except how weights are computed/propagated.
   - Keep existing log keys where possible (do not break localization).

## Non-goals (do NOT do)
- Do not modify GUI (Qt/Tk).
- Do not change default settings values.
- Do not change WSC implementation, sigma clip math, or normalization algorithms (except weight calculation plumbing).
- Do not add new dependencies.

## Files to inspect / likely edit
- `zemosaic_align_stack.py`:
  - `_compute_quality_weights(...)` and the `noise_fwhm` / FWHM helper(s).
  - The `photutils.Background2D` + `DAOStarFinder` part.
- `zemosaic_align_stack_gpu.py`:
  - `_prepare_frames_and_weights(...)` currently calls CPU helper `_compute_quality_weights`.
  - Check for logic that drops `quality_weights` when `radial_map is None` and fix it (see below).
- Optional: tests under `tests/`.

IMPORTANT: There might be multiple copies of modules in the repo (root vs core). Ensure you patch the one actually imported by runtime:
- Run: `python -c "import zemosaic_align_stack; print(zemosaic_align_stack.__file__)"`

## Required behavior changes (CPU side) — robust FWHM weighting
In `noise_fwhm` weighting, before calling `Background2D`:
1. Compute a finite mask on `target_data_for_fwhm`:
   - `finite = np.isfinite(target_data_for_fwhm)`
   - If no finite pixels: log existing key `weight_fwhm_no_finite_data` and set fwhm=inf for this image.

2. Define a “usable ROI” to avoid NaN borders:
   - Compute bounding box of finite pixels (min/max rows/cols where finite==True).
   - Optionally grow bbox by a small margin (e.g., 8 px) but clamp to image bounds.
   - If bbox is too small (e.g., < 64x64 or finite fraction < 0.10), skip Background2D and go directly to fallback stats (sigma_clipped_stats on finite pixels only).

3. Run Background2D on the ROI only:
   - Feed `roi = target_data_for_fwhm[y0:y1, x0:x1]`
   - Provide `mask=~np.isfinite(roi)` if supported by photutils Background2D.
   - Keep `exclude_percentile` high (e.g., 90) but do not rely on it alone.

4. Compute `data_subtracted` correctly:
   - `data_subtracted_roi = roi - bkg.background`
   - Then sanitize: `np.nan_to_num(..., nan=0, posinf=0, neginf=0)` and float32.

5. Threshold scalarization:
   - `threshold_daofind_val = 5.0 * background_rms`
   - If `background_rms` is an array, use a scalar robust reducer (`nanmedian` preferred, else `nanmean`).
   - Ensure threshold is finite; otherwise fallback stats.

6. Fallback stats must ignore NaNs:
   - Use sigma_clipped_stats on finite pixels only.
   - If the function requires full array, pass `roi_clean` where invalids are replaced by median of finite pixels.
   - Create `data_subtracted_roi = roi_clean - median_glob`
   - threshold = 5 * stddev_glob (finite guard).

7. DAOStarFinder compatibility with photutils:
   - Keep the existing fix: try with `sky=0.0`, if TypeError mentions sky, retry without `sky`.
   - Do NOT swallow unrelated TypeErrors silently.
   - If DAOStarFinder fails: keep current behavior (log `weight_fwhm_daofind_error`, fwhm=inf).

8. FWHM estimate must be stable:
   - If no detected sources, treat as invalid (fwhm=inf) and log existing key if present (or reuse existing warn).
   - If computed fwhm is non-finite or <=0, set inf.

9. Convert FWHM list → weights:
   - Keep EXACT current mapping formula (do not change weight law).
   - Only improve the upstream robustness so FWHM is computable more often.

## Required behavior changes (GPU side) — apply quality weights consistently
In `zemosaic_align_stack_gpu.py`, inside `_prepare_frames_and_weights`:

There is currently logic that effectively disables `quality_weights` when `radial_map is None`.
This makes GPU ignore `noise_fwhm`/`noise_variance` unless radial weighting is enabled.

Fix:
- Remove the unconditional drop.
- Instead, only drop/skip weights that are not broadcastable to frame shape.
- Keep parity with CPU:
  - If CPU applies per-frame scalar/channel weights for mean/WSC, GPU must too.
  - If CPU intentionally ignores weights for some combine methods, GPU should match that.

Implementation suggestion:
- Keep `quality_weights` as returned by `_compute_quality_weights`.
- During `combined_weights` building, use `_broadcast_weight_template(q_weight, frame.shape)` to validate.
- If broadcast fails for a frame, set that frame's q_weight to None (or drop weighting entirely only if too many invalid).
- Do not set `weight_method_used="none"` unless weights truly cannot be applied.

Also ensure WSC weights block (`wsc_weights_block`) remains consistent and is used when reject algo is winsorized sigma clip.

## Logging requirements
- Keep existing GUI keys used by worker logs:
  - `weight_fwhm_bkg2d_error`
  - `weight_fwhm_no_finite_data`
  - `weight_fwhm_global_stats_invalid`
  - `weight_fwhm_warn_all_fwhm_are_infinite`
  - `weight_fwhm_daofind_error`
- Optionally add DEBUG-only logs (not GUI-keyed) for:
  - finite fraction
  - bbox size
But do not spam warnings.

## Acceptance criteria
- On datasets with NaN borders (common aligned frames):
  - `noise_fwhm` no longer collapses to all-infinite FWHM in typical cases with usable overlap.
  - Weighting produces non-uniform weights (unless truly no stars / no finite pixels).
- GPU + CPU runs with same settings:
  - Both apply quality weights when selected (unless combine method doesn’t support weights).
  - No regression in other modes (noise_variance, none, other reject algos).

## Tests (must add at least 2)
Add unit tests that do NOT require a GPU:

1) `test_noise_fwhm_nan_borders_produces_weights`
- Build a small stack of synthetic frames (float32) with:
  - Central region containing a few gaussian “stars”
  - Borders set to NaN (simulate warp/outside overlap)
- Call `_compute_quality_weights(frames, "noise_fwhm")`
- Assert:
  - returned weights are finite
  - not all equal (non-uniform) when stars exist

2) `test_prepare_frames_and_weights_keeps_quality_weights_without_radial`
- Call GPU helper `_prepare_frames_and_weights` with:
  - `stack_weight_method="noise_fwhm"`
  - radial weighting disabled
- Assert:
  - `weight_method_used` remains `"noise_fwhm"` (or at least not forced to "none")
  - and/or `wsc_weights_block` non-None when weights computed

Keep tests small and fast.

## Deliverables
- PR-ready code changes.
- Tests added/updated and passing.
- Brief summary in followup.md including what was changed and why.
