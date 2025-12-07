
### ✅ followup.md

```markdown
# Grid coverage diagnostics – follow-up checklist

## Instrumentation

- [x] In **both** `grid_mode_last_know_geomtric_tiles_ok.py` and `grid_mode.py`:
  - [x] Locate `_reproject_frame_to_tile(...)` (or equivalent reprojection helper).
  - [x] After computing `patch` and `weight_map`, log:
        - [x] `patch_shape`
        - [x] `finite_frac` (`isfinite(patch).mean()`)
        - [x] `nan_frac`
        - [x] `nonzero_weight_frac` (`(weight_map > 0).mean()` or `-1` if no map)
        - [ ] Optional: local `weight_bbox` where weight_map > 0
        - [x] Use prefix `[GRIDCOV]` and `frame={frame.path.name}`, `tile_id={tile.tile_id}`.
  - [x] In `process_tile(...)`, before the frame loop, log per tile:
        - [x] `tile_id`, `bbox`, `tile_shape`, `frames_in_tile`
        - [x] Prefix `[GRIDCOV]`, level INFO.
  - [x] In `process_tile(...)`, after computing `stacked` (just before saving), log:
        - [x] `stacked_shape`
        - [x] `finite_frac` on a gray version
        - [x] `nan_frac`
        - [x] `nonzero_frac` (`(stacked_gray != 0).mean()`)
        - [ ] Optional: `stacked_min`, `stacked_max`
        - [x] Prefix `[GRIDCOV]`, level INFO/DEBUG.
  - [ ] (Optional) In `assemble_tiles(...)`, after reading each tile FITS:
        - [ ] Log `shape`, `finite_frac`, `nonzero_frac` with prefix `[GRIDCOV]`.

## Runs

- [ ] Run **last-good** Grid mode (OK geometry) on the reference dataset with verbose logging:
      - [ ] Save worker log as e.g. `zemosaic_worker_grid_ok_coverage.log`.
- [ ] Run **current** Grid mode (faulty) on the same dataset:
      - [ ] Save worker log as e.g. `zemosaic_worker_grid_faulty_coverage.log`.

## Comparison

- [ ] Using a diff tool (WinMerge / Meld / VSCode), compare only `[GRIDCOV]` lines.
- [ ] For each `tile_id`, compare:
      - [ ] `tile_shape`, `frames_in_tile` (should match).
      - [ ] `stacked nonzero_frac` (expected to be much lower in faulty).
- [ ] For each (tile, frame) pair, compare:
      - [ ] `patch_shape` (should usually match).
      - [ ] `finite_frac` and `nonzero_weight_frac`.
      - [ ] `weight_bbox_local` (look for shrinkage/offset in faulty version).
- [ ] Note any patterns:
      - [ ] Specific tiles more affected?
      - [ ] Specific frames (e.g. those with WCS warnings) more affected?
      - [ ] Systematic offset of `weight_bbox` in faulty version?

## Fix preparation

- [ ] Summarise where coverage diverges:
      - [ ] e.g. “Reprojected weight map is much smaller / shifted in faulty version for tiles X/Y”.
- [ ] Propose a **minimal** geometric fix in `grid_mode.py` at the reprojection/weight-map level, keeping:
      - [ ] multithreading behaviour unchanged,
      - [ ] GPU flags untouched,
      - [ ] stacking formulas unchanged.
- [ ] After fixing, remove or downgrade `[GRIDCOV]` logs to DEBUG for normal usage.
````
