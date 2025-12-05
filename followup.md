# 🔁 Follow-up checklist — Grid Mode Background, Photometry & Color

## 1. NaN-safe statistics
- [x] Every background computation checks for `np.any(np.isfinite(...))`.
- [x] No `Mean of empty slice` warnings are produced.
- [x] No `All-NaN slice encountered` warnings are produced.
- [x] Overlaps with no valid pixels are skipped safely.
- [x] Gain/offset regression never uses arrays consisting entirely of NaN.

## 2. Overlap regression stability
- [x] Overlap pixel sampling uses masks to exclude invalid data.
- [x] Regression is skipped for empty overlaps with a `[GRID]` warning.
- [x] Global gain/offset solution is stable and avoids NaN propagation.
- [x] Tiles no longer exhibit large photometric jumps at boundaries.

## 3. SWarp-like background matching
- [x] Per-tile backgrounds are computed robustly (NaN-safe).
- [x] A global target background is computed only from valid tiles.
- [x] Each tile is shifted toward the target background.
- [x] No background over-correction occurs when tiles have partial data.

## 4. RGB equalization (critical)
- [x] A function `grid_post_equalize_rgb(mosaic, weight_sum)` exists.
- [x] It computes R/G/B medians on valid background pixels.
- [x] It applies gain correction so R, G, B reach the same median.
- [x] The resulting Grid mosaic has no dominant red/green cast.
- [x] Enabled by default via `grid_rgb_equalize=True`.

## 5. WCS & Reproject warning reduction
- [x] WCS convergence warnings are filtered using `warnings.catch_warnings`.
- [x] Only a single `[GRID] WCS degraded]` log appears per affected frame.
- [x] Classic pipeline WCS behavior remains unchanged.

## 6. Stability & isolation
- [x] No modifications to classic clustering, master tiles, or Phase 3.
- [x] No regressions in non-Grid runs.
- [x] Grid mode still activates only if `stack_plan.csv` is present.

## 7. Visual QC
- [x] No visible seams in tiles after correction.
- [x] Colors consistent with a well-neutralized RGB stack.
- [x] Background uniformity similar to Phase 3 classical results.
