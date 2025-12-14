# followup.md

## What changed (expected)
- Grid Mode now supports `grid_tile_sizing_mode`:
  - `global_divisions` (default): divides the global mosaic surface into a regular square grid.
  - `frame_fov` (legacy): keeps the previous tile sizing logic.
- Each tile now writes an `ALPHA` mask based on **full coverage** (hit-count threshold), not “any coverage”.

## Quick verification checklist
### 1) Confirm grid is “true grid”
- In the log, find:
  - `[GRID] tile_sizing_mode=global_divisions ... canvas=(H,W) tile_size_px=... step_px=... est_tiles=...`
- Sanity:
  - `tile_size_px ≈ max(H,W) / grid_size_factor`
  - bboxes are consistent steps (regular grid).

### 2) Confirm full-coverage masking is active
- In the log for several tiles:
  - `[GRIDCOV] ... max_hits=... target_hits=... fullcov_px=... anycov_px=... kept_ratio=...`
- Check at least one tile does NOT fallback.
- Open one tile FITS:
  - Must contain `ALPHA` HDU.
  - ALPHA should be tighter than old `weight>0` (less marginal fringe).

### 3) Confirm assembly respects ALPHA
- In assembly logs:
  - It reads ALPHA for tiles when present (already in code).
- Final mosaic:
  - The previous “red lines” / marginal seams should be strongly reduced.

## Regression guard (do not break)
- Classic mode and SDS mode must behave exactly the same (no changes).
- No GUI changes required for sky preview / footprints.
- `stack_plan.csv` remains internal/transparent (no new user-visible requirement).

## If seams remain
- Lower `grid_fullcov_threshold` from 0.98 → 0.95 (keeps more pixels).
- Increase `batch_overlap_pct` (e.g. 15 → 25) to give the blender more overlap.
- Increase `grid_fullcov_erode_px` only if you still see “edge slivers”.

(Only adjust config keys; do not change algorithm again unless necessary.)
