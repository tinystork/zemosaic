# Agent – Grid Mode RGB + stacking parity

## Context

Project: **ZeMosaic / Grid & Survey mode**

Files primarily involved:

- `grid_mode.py`
- `zemosaic_worker.py` (only for wiring & config forwarding, *no heavy refactor here*)
- `zemosaic_utils.py` (for reusing existing FITS loading / debayer logic, if needed)

Current symptoms (from user logs and visual output):

- Grid mode runs to completion, builds tiles, and assembles a final mosaic without throwing.
- All tiles and the final `mosaic_grid.fits` are **monochrome** with a single channel (shape `(H, W, 1)`).
- When converting some tiles to PNG, dimensions appear as `1 x 1920` instead of `1080 x 1920 x 3` as expected for RGB.
- Log excerpt:

  - Final mosaic is prepared as `shape=(3272, 2406, 1)` and RGB equalization is skipped because the mosaic is not RGB. :contentReference[oaicite:0]{index=0}  
  - Tiles are defined with `tile_size_px=1920` and bbox coordinates are correct; the **geometry/WCS is OK**, the problem is the **data layout / channels**.

Goal for this agent:

1. Make **Grid/Survey mode** handle **RGB data** correctly (3 channels), in a way that is consistent with the classic hierarchical pipeline.
2. Ensure stacking behaviour in grid mode (**norm, weights, reject algo, combine**) matches the main stacking pipeline semantics.
3. Add explicit logging around shapes and channel counts so future regressions are easy to spot.
4. Preserve backward compatibility for mono / legacy datasets where possible.

---

## High-level plan

1. **Instrumentation & diagnostics** – log shapes / channels at key stages so we can verify behaviour.
2. **Fix FITS loading / channel handling in Grid Mode** – ensure `_load_image_with_optional_alpha` and `_reproject_frame_to_tile` produce the correct H×W×C arrays.
3. **Verify & align stacking semantics** – ensure `_stack_weighted_patches` reflects `stack_norm_method`, `stack_weight_method`, `stack_reject_algo`, `stack_final_combine` in the same spirit as `create_master_tile` / classic stacking.
4. **Fix mosaic assembly & RGB equalization** – ensure final mosaic is RGB when input is RGB, and that `grid_post_equalize_rgb` is actually used.
5. **Add small regression tests / sanity checks** – at least smoke tests and logging.

Work incrementally and keep `followup.md` updated (checked tasks, notes, regressions).

---

## 1. Instrumentation & diagnostics

**File:** `grid_mode.py`

### 1.1. Log channel / shape info at frame load & reprojection

Add detailed `[GRID]` logs in these functions:

- `_load_image_with_optional_alpha`
- `_reproject_frame_to_tile`
- `process_tile`
- Assembly phase functions (where the final mosaic array is created and RGB equalization is called)

Concretely:

- After `_ensure_hwc_array(data)` in `_load_image_with_optional_alpha`, log:

  - Original FITS data shape and dtype (`hdul[0].data.shape`, `dtype`)
  - Shape after `_ensure_hwc_array` (H, W, C)
  - Whether `C == 1` or `C == 3`

- In `_reproject_frame_to_tile`, after computing `reproj_stack`:

  - Log the tile id, frame path (basename), and `reproj_stack.shape`.

- In `process_tile`, just before calling `_stack_weighted_patches` and just before writing the tile FITS:

  - Log the number of frames, tile bbox, tile_shape, and the shape of the stacked result.

- In the assembly phase:

  - When allocating the mosaic canvas, log its shape and dtype.
  - Just before calling `grid_post_equalize_rgb`, log the current mosaic shape and the fact that equalization will or will not run (include the reason when skipping).

Make logs explicit, e.g.:

- `[GRID] DEBUG_SHAPE: loaded frame 'Light_xxx.fit' raw=(H,W,?) hwc=(H,W,C)`
- `[GRID] DEBUG_SHAPE: tile 3 stacked patch shape=(H,W,C)`
- `[GRID] DEBUG_SHAPE: mosaic before RGB equalization shape=(H,W,C)`

This will confirm whether the monochrome behaviour comes from FITS loading or from the stacking/assembly steps.

### 1.2. Log stacking configuration

In `run_grid_mode`, after assembling `GridModeConfig`, log a single summary line:

- `stack_norm_method`
- `stack_weight_method`
- `stack_reject_algo`
- `winsor_limits`
- `stack_final_combine`
- `apply_radial_weight`, `radial_feather_fraction`, `radial_shape_power`
- `grid_rgb_equalize`

Example:

```python
_emit(
    "[GRID] Stacking config: norm={config.stack_norm_method}, "
    f"weight={config.stack_weight_method}, reject={config.stack_reject_algo}, "
    f"winsor={config.winsor_limits}, combine={config.stack_final_combine}, "
    f"radial={config.apply_radial_weight} "
    f"(feather={config.radial_feather_fraction}, power={config.radial_shape_power}), "
    f"rgb_equalize={grid_rgb_equalize}",
    lvl="INFO",
    callback=progress_callback,
)
````

---

## 2. Fix FITS loading & RGB handling in Grid Mode

**File:** `grid_mode.py` (primary), optionally `zemosaic_utils.py` for reuse.

Currently, `_load_image_with_optional_alpha` reads:

* `data = hdul[0].data`
* Then passes it to `_ensure_hwc_array`, which converts:

  * 2D arrays → `(H, W, 1)`
  * 3D arrays with C/H/W in first axis or last axis → `(H, W, C)`

This means:

* Seestar raw Bayer FITS (2D) become `(H, W, 1)` and are never debayered to RGB.
* The whole Grid Mode pipeline then works on a single channel → final mosaic `(H, W, 1)` and RGB equalization is skipped.

### 2.1. Decide on the source of RGB data

Goal: Grid Mode should operate on **the same type of images as the classic pipeline**:

* For Seestar / Bayer FITS: pipeline converts to RGB using `zemosaic_utils.debayer_image(...)`.
* For already RGB FITS (e.g. external pre-stacked masters): we should keep them as 3-channel images (H, W, 3).

Implementation guidelines:

1. **Try to reuse `zemosaic_utils.load_and_validate_fits`**:

   * It already handles BZERO/BSCALE, axis ordering (CHW vs HWC), NaN sanitization, etc.
   * It can return an array that is either 2D (mono) or 3D (HWC).

2. If you integrate `load_and_validate_fits`:

   * Import it at top of `grid_mode.py`:

     ```python
     from zemosaic_utils import load_and_validate_fits, debayer_image
     ```

   * In `_load_image_with_optional_alpha`:

     * Replace the raw `hdul[0].data` logic by:

       ```python
       img, header, info = load_and_validate_fits(path, normalize_to_float32=True, attempt_fix_nonfinite=True, progress_callback=None)
       ```

     * If `img is None`, raise or propagate an error so the frame is skipped with a `[GRID]` warning.

   * Use the header and/or `info` to detect whether the image is mono Bayer that needs debayering:

     * Check `BAYERPAT` in the header.
     * Or, rely on project conventions (e.g. Seestar FITS are 2D; dedicated config flag if needed).

3. **Debayer when necessary**:

   * If `img.ndim == 2` and you detect a Bayer pattern:

     ```python
     bayer_pattern = header.get("BAYERPAT", "GRBG")
     img_rgb = debayer_image(img, bayer_pattern=bayer_pattern)
     ```

   * Ensure `img_rgb` is normalized to [0, 1] float32 in HWC format.

4. **Ensure non-Bayer images stay as they are**:

   * If `img.ndim == 3` and appears to be RGB (C in last axis or first axis) rely on `load_and_validate_fits` axis handling.
   * After that, pass the image to `_ensure_hwc_array` to guarantee `(H, W, C)`.

5. **Return HWC in `_load_image_with_optional_alpha`**:

   * After the above, `_ensure_hwc_array` should give you `(H, W, C)`; `C` will be 3 for RGB data.
   * Only if the input is truly mono (e.g. a genuine grayscale survey image) should `C` remain 1.

6. **Alpha channel**:

   * Keep the alpha logic as is, but ensure that if the alpha HDU is 2D and the data is RGB, alpha is broadcast/expanded to `(H, W, C)`.

End goal:

* For Seestar lights in Grid Mode: **H×W×3 RGB arrays**, same photometric scale as the rest of the pipeline.

### 2.2. Check `_reproject_frame_to_tile` uses all channels

Function currently:

* Calls `_load_image_with_optional_alpha` → `data` (H, W, C) and `alpha_weights`.
* Sets `channels = data.shape[-1] if data.ndim == 3 else 1`.
* Loops over `c in range(channels)` and reprojects `data[..., c]`.

Verify / ensure:

* For RGB images, `channels == 3` and `reproj_stack.shape == (tile_h, tile_w, 3)`.
* For genuine mono images, `channels == 1` and you end with `(tile_h, tile_w, 1)`.

Keep the existing `alpha_weights` handling but ensure:

* If `alpha_weights` is 2D and `data` has 3 channels, broadcast alpha to 3 channels when applying it.

---

## 3. Verify & align stacking semantics

**File:** `grid_mode.py`

Function: `_stack_weighted_patches` + surrounding chunked stacking logic in `process_tile`.

Goal: Make Grid Mode stacking behave like classic stacking with respect to:

* `stack_norm_method`: e.g. `"none"`, `"linear_fit"`, etc.
* `stack_weight_method`: e.g. `"noise_variance"`, `"none"`, etc.
* `stack_reject_algo`: `"winsorized_sigma_clip"`, `"kappa_sigma"`, or `"none"`.
* `winsor_limits`: e.g. `(0.05, 0.05)`.
* `stack_final_combine`: `"mean"` or `"median"`.

Tasks:

1. **Review how classic stacking interprets these parameters**:

   * Look into `zemosaic_align_stack.py` and `zemosaic_worker.create_master_tile` for reference.
   * Identify how `linear_fit` normalization and `noise_variance` weighting are implemented in the standard pipeline.

2. **Document any differences in comments**:

   * If grid mode cannot exactly mirror the classic behaviour (e.g. due to different data shapes or constraints), document the deviation in comments at top of `_stack_weighted_patches`.

3. **Ensure `_stack_weighted_patches` uses config correctly**:

   * For rejection:

     * If `config.stack_reject_algo` is `winsorized_sigma_clip` or `winsor`, use `_reject_outliers_winsorized_sigma_clip`.
     * If `kappa_sigma` or `kappa`, use `_reject_outliers_kappa_sigma`.
     * If `"none"` or unrecognized, skip rejection.

   * For final combine:

     * `"mean"` → weighted sum / weight_sum.
     * `"median"` → median over valid positions.

   * Make sure all these branches are already correct; if not, fix & add comments.

4. **Channel-wise stacking**

   * Stacking should work identically per channel:

     * `patches` should be `(H, W, C)`.
     * The code should allow C=1 or C=3 seamlessly (no hardcoded assumption that C=1).
     * All normalization and rejection should operate over the frame axis, not across channels.

5. **Return types**

   * `_stack_weighted_patches` should always return:

     * `(stacked, weight_sum, ref_median_used)` when `return_weight_sum=True` and `return_ref_median=True`.
     * Ensure `stacked.shape` matches `(H, W, C)` and `weight_sum.shape` matches `(H, W, C)`.

---

## 4. Fix mosaic assembly & RGB equalization

**File:** `grid_mode.py`

1. **Creation of the mosaic canvas**

   * In the assembly phase (where tiles are read from disk and blended):

     * Ensure the canvas is created with shape `(H_global, W_global, C)` where `C` comes from the first valid tile.
     * If some tiles are mono and others RGB, log a warning and either:

       * Convert mono tiles to RGB by duplicating the channel; or
       * Skip RGB equalization and state why.

2. **RGB equalization**

   * Function `grid_post_equalize_rgb` already expects an RGB mosaic and logs:

     * `"RGB equalization: skipped (non-RGB mosaic with shape=...)"` in the current logs.

   * Make sure that:

     * We call `grid_post_equalize_rgb` **only when** `mosaic.ndim == 3` and `mosaic.shape[-1] == 3`.
     * When `grid_rgb_equalize=True` but `mosaic` is not RGB, we keep the log but ensure it’s clear (include shape, why it’s mono).

3. **Photometry and blending**

   * Verify that photometry and pyramidal blending continue to work with multi-channel arrays:

     * Either they already support HWC arrays, or they operate per channel.
     * If there are any assumptions of 2D arrays, adapt them to handle `(H, W, C)` generically.

---

## 5. FITS output layout & backward compatibility

**File:** `grid_mode.py`

1. **Tile FITS**

   * For RGB data, write the tile as HWC (H, W, 3) in the main HDU, consistent with what other parts of ZeMosaic expect.
   * If there is a `legacy_rgb_cube` option (`C, H, W`), respect it:

     * If `legacy_rgb_cube=True`, move axis from HWC to CHW before writing.

2. **Final mosaic**

   * Same approach: write final mosaic as HWC by default (H, W, 3).
   * If `save_final_as_uint16=True`, ensure proper scaling from float32 [0, 1] (or ADU range) to uint16.

3. **Channel count sanity check**

   * Immediately before writing each FITS, log:

     * Tile id (or `mosaic_grid`)
     * Final data shape and dtype.

---

## 6. Tests & sanity checks

### 6.1. Small regression / smoke tests

If there is an existing tests folder, add at least a small test module, e.g. `tests/test_grid_mode_rgb.py`:

* Use tiny synthetic images (e.g. 16×16 RGB with simple patterns) and a dummy WCS.
* Build a minimal `GridDefinition` with 1–2 tiles and 2–3 frames.
* Run `process_tile` and the assembly logic.

Assertions:

* Stacked tile has 3 channels.
* Mosaic has 3 channels.
* `grid_post_equalize_rgb` does not raise and does not change the shape.
* When `grid_rgb_equalize=False`, mosaic data is unchanged except for rounding differences.

### 6.2. Manual QA scenario

Document in `followup.md`:

* Steps to reproduce on the example dataset used by the user:

  1. Run ZeMosaic with Grid Mode on the existing `stack_plan.csv`.
  2. Inspect one tile FITS and the final `mosaic_grid.fits` using `astropy.io.fits.getdata`.
  3. Confirm shapes `(H, W, 3)` and visually check that:

     * Colors match those from the classic pipeline.
     * RGB equalization doesn’t introduce strange tints.

---

## 7. Constraints & style

* Keep changes **localized to Grid Mode** as much as possible.
* Avoid breaking existing mono-only workflows (if any); log clearly when mono mode is used.
* Use existing logging helper `_emit` for all messages.
* Follow the project’s style and type hints (PEP-8, annotations, etc.).
* Update `followup.md`:

  * Mark tasks as `[x]` when completed.
  * Add notes for deviations or blockers.

If you need to decide between **perfect parity** and **reasonable similarity** with the classic pipeline, choose **reasonable similarity**, document the difference, and move on.
