# Patch checklist (what to change exactly)

## 1) Fix preview NaN masking (definite bug)
In Phase6 preview section, currently:
    mask_3d = mask_zero[:, :, None]
    preview_view[mask_3d] = np.nan
This fails because boolean indexing does not broadcast (H,W,1) → (H,W,3).

Change to:
    preview_view = np.where(mask_zero[..., None], np.nan, preview_view)

Also keep the try/except, but this change should eliminate:
"phase6: preview NaN masking failed: boolean index did not match ..."

## 2) Preserve NaNs in coadd output (root cause of black rectangles)
In assemble_final_mosaic_reproject_coadd / global coadd finalize:
- Remove or avoid these conversions:
    chunk_result = np.nan_to_num(chunk_result, nan=0.0)
    final_image = np.nan_to_num(final_image, nan=0.0, ...)
Those lines turn transparent/invalid into real zeros → black.

Instead:
- After computing chunk_weight, enforce:
    invalid = (chunk_weight <= 0)
    chunk_result[invalid, :] = np.nan
- For mean/kappa finalize results:
    invalid = (coverage_map <= 0)
    final_image[invalid, :] = np.nan

Optionally sanitize inf:
    final_image[~np.isfinite(final_image)] = np.nan
but do NOT turn into 0.

## 3) Phase6 saving: do not "nan_to_num" the science mosaic
Before calling zemosaic_utils.save_fits_image for the main float FITS:
- ensure save_array keeps NaNs (float32 ok).
- if there is any pre-save nan_to_num in this path, remove it.

Keep writing ALPHA extension as now.

Note: if write_final_fits_uint16_color_aware exists and is used for viewer FITS,
it's okay if it clamps NaN to 0 in the viewer product. The "science" float FITS must preserve NaN.

## 4) Verify weight handling (avoid NaN polluting sums)
If anywhere you accumulate:
    sum += patch_data * patch_weight
Ensure patch_weight is 0 where patch_data is non-finite.
If not already guaranteed, add:
    finite = np.all(np.isfinite(patch_data), axis=-1)
    patch_weight = np.where(finite, patch_weight, 0.0)
    patch_data = np.where(finite[..., None], patch_data, 0.0)
This is only needed if NaNs currently propagate into sum_grid.

## 5) Quick verification steps (local)
- Run a small mosaic where lecropper creates masked pixels.
- Confirm in log:
  - no "preview NaN masking failed" warning
  - alpha extension written
- Open final float FITS and check:
  - np.isnan(data).sum() > 0
  - masked areas are NaN (not 0)
- Visually: black rectangles should no longer be “hard” blocks in the mosaic FITS stretch.
  (Some viewers show NaN as black; that’s fine, but they must not contribute to coadd and the PNG preview should be transparent.)

## Notes / gotchas
- FITS viewers: many ignore ALPHA extension → NaN is the reliable “transparent” signal in the main image.
- Keep behavior unchanged for batch size mode logic.
- Keep current logs; add one INFO like:
  "global_coadd: nanized %d pixels where coverage==0"
  (optional, but useful for debugging).
