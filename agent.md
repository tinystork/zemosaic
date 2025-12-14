# agent.md

## Mission (surgical / grid only)
Fix Grid Mode so it matches the original concept:
1) **Build a true regular grid** over the **global mosaic canvas** (global surface divided into squares).
2) **Eliminate marginal-coverage artifacts** (the “red lines”) by generating a **full-coverage ALPHA mask** per tile (after reprojection), and using it during assembly.

Constraints:
- **Scope only:** `grid_mode.py` (+ `zemosaic_config.py` for defaults if needed).
- **Do NOT modify** classic mode, SDS mode, worker pipeline logic, or GUI plotting/footprints in `zemosaic_filter_gui_qt.py`.
- The use of `stack_plan.csv` must remain **transparent** (still the internal input contract; no UI requirement added).

## Context (why it’s currently disappointing)
- Current Grid Mode *already* reprojects frames into tiles, but:
  - Tile sizing is effectively “frame-FOV based” with a default `grid_size_factor=1.0`, which often yields **too-large tiles / odd overlap behavior** compared to “divide the global surface into squares”.
  - The tile ALPHA mask is currently `running_weight > 0` (any coverage), which keeps **thin marginal areas** that later become **visible seams / red lines** during mosaic assembly.

## Files to edit
- `grid_mode.py`
- (optional) `zemosaic_config.py` only to add safe defaults for new grid-mode keys (no GUI).

---

## Task A — True regular grid over the global canvas
### Goal
Change the tile sizing strategy so Grid Mode can produce a **regular square grid** across the global mosaic extent.

### Implementation plan
1) In `grid_mode.py`, extend config parsing in `run_grid_mode()`:
   - Read a new key: `grid_tile_sizing_mode` (string).
     - Allowed: `"global_divisions"` (new default), `"frame_fov"` (legacy).
   - Keep `grid_size_factor` but interpret it depending on the mode:
     - If `grid_tile_sizing_mode == "global_divisions"`:
       - Treat `grid_size_factor` as **number of divisions across the largest mosaic dimension**.
       - Example: `div = max(1, int(round(grid_size_factor)))`
     - If `"frame_fov"`: keep the current behavior (do not delete legacy code).

2) In `build_global_grid(...)`, after `global_shape_hw` is known:
   - If sizing mode is `"global_divisions"`:
     - Let `H, W = global_shape_hw`
     - `div = max(1, int(round(grid_size_factor)))`
     - `tile_size_px = int(max(64, math.ceil(max(H, W) / div)))`
     - `step_px = max(1, int(round(tile_size_px * (1.0 - overlap_fraction))))`
   - Else (`"frame_fov"`): keep the existing `tile_size_deg / pixel_scale` logic.

3) Add a single clear log line:
   - `[GRID] tile_sizing_mode=<...> grid_size_factor=<...> tile_size_px=<...> step_px=<...> est_tiles=<...> canvas=<H,W>`

Acceptance for Task A:
- Grid tiles form a regular square grid over the global extent (consistent bbox steps).
- No GUI changes required.

---

## Task B — Full-coverage ALPHA to remove marginal artifacts (“red lines”)
### Goal
Within each processed tile, compute a stricter coverage mask that keeps only the **high-confidence interior** (full coverage by the dense overlap), and save it as `ALPHA` so assembly naturally ignores marginal pixels.

### Implementation plan (process_tile)
1) In `process_tile(...)`, create a per-tile hit counter:
   - `hit_count: np.ndarray` shape `(tile_h, tile_w)` dtype `np.uint16` or `np.int32`, initialized to zeros.
2) In the per-frame loop (right after `_reproject_frame_to_tile` returns `patch, footprint`):
   - Compute `hit_inc = (footprint > 0)` (2D boolean).
   - `hit_count += hit_inc.astype(hit_count.dtype)`
   - This must happen regardless of chunking (chunking only affects stacking, not hit accumulation).

3) After stacking is finalized and before saving:
   - Add config keys (read in `run_grid_mode`, stored in `GridModeConfig`):
     - `grid_fullcov_enabled` (bool, default True)
     - `grid_fullcov_threshold` (float in [0.0..1.0], default 0.98)
     - `grid_fullcov_erode_px` (int, default 1)
     - `grid_fullcov_min_keep` (float, default 0.10)  # safety so we don’t wipe a tile
   - If enabled and `hit_count.max() > 0`:
     - `max_hits = int(hit_count.max())`
     - `target = max(1, int(math.floor(max_hits * grid_fullcov_threshold)))`
     - `fullcov = (hit_count >= target)`
     - Optional: if `_NDIMAGE_AVAILABLE` and `grid_fullcov_erode_px > 0`:
       - `fullcov = ndimage.binary_erosion(fullcov, iterations=grid_fullcov_erode_px, border_value=0)`
     - Safety:
       - `coverage_any = (hit_count > 0)`
       - if `fullcov.sum() < grid_fullcov_min_keep * coverage_any.sum()` then fallback to `coverage_any`.
   - Define `coverage_mask_alpha = fullcov.astype(np.uint8) * 255`
   - Also **zero-out** data outside mask to avoid any residual:
     - if `stacked.ndim == 3`: `stacked[~fullcov, :] = 0.0`
     - else: `stacked[~fullcov] = 0.0`

4) Add logs (INFO/DEBUG):
   - `[GRIDCOV] tile_id=... max_hits=... threshold=... target_hits=... fullcov_px=... anycov_px=... kept_ratio=...`
   - If fallback triggered, log it explicitly.

Acceptance for Task B:
- Tile FITS contain `ALPHA` that reflects **full coverage**, not “any coverage”.
- Assembly uses ALPHA already; seams from marginal areas should be strongly reduced.

---

## Task C — Safe defaults (optional, config only)
In `zemosaic_config.py`, add non-breaking defaults so Grid Mode works “as intended” without user intervention:
- `grid_tile_sizing_mode`: `"global_divisions"`
- `grid_size_factor`: `4`   # 4 divisions across max dimension (tweakable)
- `batch_overlap_pct`: `15`  # modest overlap to blend seams
- `grid_fullcov_enabled`: `True`
- `grid_fullcov_threshold`: `0.98`
- `grid_fullcov_erode_px`: `1`
- `grid_fullcov_min_keep`: `0.10`
- Keep `use_gpu_grid` unchanged (don’t force GPU).

If you add keys, ensure `load_config()` merges them safely (existing user config must still override).

---

## Tests (manual, minimal)
1) Run Grid Mode on a known dataset with visible “red lines”.
2) Confirm logs show:
   - tile_sizing_mode=global_divisions
   - reasonable tile_size_px / step_px
   - fullcov stats per tile
3) Confirm output tiles include an `ALPHA` HDU and it’s “tighter” than before.
4) Confirm final mosaic shows fewer/no red seam lines at tile borders.

---

## Deliverable
- Commit changes with message:
  - `grid: true global grid + full-coverage alpha mask to kill marginal seams`
- Only touched files in scope.
