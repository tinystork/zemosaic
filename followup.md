
## 📝 `followup.md`

```markdown
# Follow-up – Grid Mode RGB + stacking parity

## Checklist

### 1. Instrumentation & diagnostics

- [ ] Add detailed `[GRID]` logs in `_load_image_with_optional_alpha` showing:
  - [ ] Raw FITS data shape/dtype.
  - [ ] Shape after `_ensure_hwc_array` (H,W,C) and channel count.
- [ ] Add logs in `_reproject_frame_to_tile` for each frame:
  - [ ] Frame basename, tile id, `reproj_stack.shape`.
- [ ] Add logs in `process_tile`:
  - [ ] Tile id, number of frames, `tile_shape`, stacked patch shape before writing.
- [ ] Add logs in assembly phase:
  - [ ] Mosaic canvas shape/dtype at allocation.
  - [ ] Mosaic shape just before RGB equalization, with explicit reason when equalization is skipped.
- [ ] Log one-line stacking config summary in `run_grid_mode` (norm, weight, reject, winsor, combine, radial params, rgb_equalize).

### 2. FITS loading & RGB handling

- [ ] Integrate `zemosaic_utils.load_and_validate_fits` into `_load_image_with_optional_alpha` (or equivalent helper).
- [ ] Detect Bayer / mono Seestar frames (using header `BAYERPAT` or other metadata).
- [ ] For 2D Bayer images, call `debayer_image(...)` to produce RGB HWC float32.
- [ ] Ensure non-Bayer RGB FITS are preserved as 3-channel HWC.
- [ ] Ensure `_ensure_hwc_array` always returns HWC with C=3 for RGB cases, C=1 for genuine mono.
- [ ] Ensure alpha maps (when present) are correctly shaped/broadcast for RGB data.
- [ ] Confirm via logs that grid mode sees RGB frames as `(..., 3)` for the user’s dataset.

### 3. Stacking semantics parity

- [ ] Review classic stacking behaviour in `zemosaic_align_stack.py` and `create_master_tile`.
- [ ] Document any unavoidable differences between grid mode stacking and classic stacking in comments.
- [ ] Verify `_stack_weighted_patches` correctly honors:
  - [ ] `stack_norm_method` semantics.
  - [ ] `stack_weight_method` semantics.
  - [ ] `stack_reject_algo` (`winsorized_sigma_clip`, `kappa_sigma`, `none`).
  - [ ] `winsor_limits`.
  - [ ] `stack_final_combine` (`mean` vs `median`).
- [ ] Ensure stacking operates channel-wise (patches `(H,W,C)` with C=1 or 3).
- [ ] Ensure return types remain consistent (`stacked`, `weight_sum`, `ref_median_used`).

### 4. Mosaic assembly & RGB equalization

- [ ] Ensure mosaic canvas is created with C derived from the first valid tile (`(H_global, W_global, C)`).
- [ ] Handle mixed mono/RGB tiles gracefully (convert mono to RGB or issue clear warning).
- [ ] Ensure `grid_post_equalize_rgb` is only called when mosaic is truly RGB (`ndim==3` and `C==3`).
- [ ] Keep clear logs when `grid_rgb_equalize=True` but equalization is skipped (shape + reason).
- [ ] Confirm via logs that on the user’s dataset, final mosaic shape is `(H,W,3)` and RGB equalization runs.

### 5. FITS output layout & compatibility

- [ ] Ensure tile FITS are written as HWC `(H,W,3)` for RGB data by default.
- [ ] Respect `legacy_rgb_cube` option by writing CHW `(3,H,W)` when enabled.
- [ ] Ensure final mosaic is written as HWC by default; support uint16 output when `save_final_as_uint16=True`.
- [ ] Add logs just before writing each FITS (tile and mosaic) showing final shape/dtype.

### 6. Tests & sanity checks

- [ ] Add a small test module (e.g. `tests/test_grid_mode_rgb.py`) with synthetic RGB data and a simple WCS:
  - [ ] Build a minimal `GridDefinition` with 1–2 tiles and 2–3 frames.
  - [ ] Run `process_tile` and assembly logic.
  - [ ] Assert stacked tiles and mosaic have 3 channels.
  - [ ] Assert `grid_post_equalize_rgb` preserves shape.
- [ ] Document in this file a manual QA procedure using the user’s real dataset.

### 7. Manual QA (real data)

- [ ] Run Grid Mode on the user’s mosaic dataset (the one producing the current mono mosaic).
- [ ] Inspect a few `tile_XXXX.fits` and `mosaic_grid.fits` with `astropy.io.fits.getdata`:
  - [ ] Confirm they are RGB `(H,W,3)` with reasonable ranges.
- [ ] Visually compare:
  - [ ] Colors vs classic pipeline mosaic.
  - [ ] Absence of odd color tints due to mis-applied equalization.

---

## Notes & decisions

- [ ] Confirmed which function(s) in the classic pipeline perform debayer + RGB normalization, and whether they can be reused directly in Grid Mode.
- [ ] Documented any non-trivial differences between Grid Mode stacking and classic stacking here.
- [ ] If some behaviour differences are intentional (e.g. different weighting to favor survey uniformity), explain them briefly.

---

## Known issues / open questions

Use this section to record anything unresolved:

- [ ] Do we need an explicit config flag to force mono-only grid mode for some users?
- [ ] Are there performance concerns when debayering inside Grid Mode for very large surveys? If yes, note potential mitigations (caching, pre-debayering, etc.).
````
