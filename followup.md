# followup.md

## What to run
1) Re-run the same scenario that produced:
- 3 master tiles, with group sizes like 502×1, 5×1, 3×1
- a final mosaic where the deep tile appears “écrasé” by noisy tiles

2) Capture `zemosaic_worker` log.

## What to look for in the log
- A new line from intertile calibration similar to:
  `[Intertile] Anchor selection biased: anchor=2 ... weight≈502 ...`
- Then in `assemble_final_mosaic_reproject_coadd`:
  - `apply_photometric_summary` should show corrections where the deepest tile is close to identity (gain≈1, offset≈0) compared to before.
  - Shallow tiles may be adjusted more strongly (that’s expected).

## Quick sanity checks
- If connectivity is nonzero, anchor should NOT be a 5-frame/3-frame tile anymore when a 502-frame tile exists.
- If tile_weights are missing/malformed, behavior must fall back to the previous anchor logic without crashing.

## If it still looks noisy
This patch addresses the “wrong anchor” failure mode. If the mosaic is noisy *outside overlaps*, that’s dataset reality (regions covered only by 3–5 frames will remain noisy). In that case the next step would be a UI warning:
- “Some master tiles have extremely low frame count; expect noisy regions”
and/or a filter option to exclude tiles below a minimum `tile_weight`.

(But do not implement that in this patch.)
