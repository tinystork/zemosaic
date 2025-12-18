# Patch checklist (what to change exactly)

## 1) Fix preview NaN masking (definite bug)
- [x] In Phase6 preview section, replace the boolean mask assignment with:
      preview_view = np.where(mask_zero[..., None], np.nan, preview_view)
  to avoid the broadcast mismatch warning while keeping the try/except wrapper.

## 2) Preserve NaNs in coadd output (root cause of black rectangles)
- [x] Remove/avoid conversions like:
      chunk_result = np.nan_to_num(chunk_result, nan=0.0)
      final_image = np.nan_to_num(final_image, nan=0.0, ...)
  because they turn transparent/invalid regions into zeros.
- [x] After computing chunk_weight, enforce:
      invalid = (chunk_weight <= 0)
      chunk_result[invalid, :] = np.nan
- [x] For mean/kappa finalize results, ensure:
      invalid = (coverage_map <= 0)
      final_image[invalid, :] = np.nan
- [x] Optionally sanitize inf with:
      final_image[~np.isfinite(final_image)] = np.nan
  without converting NaNs to zero.

## 3) Phase6 saving: do not "nan_to_num" the science mosaic
- [x] Before calling zemosaic_utils.save_fits_image for the main float FITS, keep NaNs (float32 ok) and remove any pre-save nan_to_num in this path.
- [x] Keep writing the ALPHA extension as now. Viewer FITS may clamp NaN to 0; the science FITS must preserve NaN.

## 4) Verify weight handling (avoid NaN polluting sums)
- [x] Where accumulating sums, ensure patch_weight is 0 where patch_data is non-finite. If needed, gate both weight and data with a finite mask so NaNs do not reach sum_grid.

## 5) Quick verification steps (local)
- [ ] Run a small mosaic where lecropper creates masked pixels.
- [ ] Confirm in log: no "preview NaN masking failed" warning and alpha extension written.
- [ ] Open final float FITS and check np.isnan(data).sum() > 0 with masked areas as NaN (not 0).
- [ ] Visually ensure black rectangles are gone; NaNs should not contribute to coadd and PNG preview should be transparent where alpha==0.

## Notes / gotchas
- FITS viewers: many ignore ALPHA extension → NaN is the reliable “transparent” signal in the main image.
- Keep behavior unchanged for batch size mode logic.
- Keep current logs; add one INFO like:
  "global_coadd: nanized %d pixels where coverage==0"
  (optional, but useful for debugging).
