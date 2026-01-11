# Follow-up: Implementation plan and checks

## [x] 1) Implement WCS pad-shift in Phase 3 master tile creation (zemosaic_worker.py)

### Location
In the Phase 3 per-tile stack code path, right after AUTO_PAD detection:

- auto_pad_used is computed by comparing aligned image shapes vs original reference shape.
- orig_hw and aligned_hw are already available and logged in:
  "MT_AUTO_PAD: detected -> will skip MT_EDGE_TRIM"

### Add this immediately after that detection/log
Compute pad offsets consistent with _pad_center_nan():
- pad_top  = max(0, (aligned_h - orig_h) // 2)
- pad_left = max(0, (aligned_w - orig_w) // 2)

If either pad offset is non-zero:
- Deep-copy the WCS (do NOT mutate shared WCS objects):
  - prefer wcs_for_master_tile.deepcopy() if available, else copy.deepcopy()
- Shift:
  - wcs_copy.wcs.crpix[0] += pad_left
  - wcs_copy.wcs.crpix[1] += pad_top
- Update shapes:
  - wcs_copy.pixel_shape = (aligned_w, aligned_h)
  - wcs_copy.array_shape = (aligned_h, aligned_w) if present
- Replace:
  - wcs_for_master_tile = wcs_copy

### Add an INFO (not DEBUG) log that will be visible to users
Key: "MT_WCS_PAD"
Include:
- tile_id
- orig_hw, padded_hw
- pad_left, pad_top
- crpix_old -> crpix_new (if accessible)

This is crucial to debug user logs without needing DEBUG_DETAIL.

### Persist pad metadata in the saved master tile header
When building header_mt_save:
- Add:
  - ZMT_PAD  = T
  - ZMT_PADX = pad_left
  - ZMT_PADY = pad_top
  - Optional: ZMT_PADHW = "Horig,Worig->Hp,Wp" (string)

This is used later to sanity-check crops.

## [x] 2) Ensure quality crop remains correct and composes with pad shift
Quality crop already does:
- wcs_cropped.wcs.crpix -= x0/y0
- updates pixel_shape/array_shape

After step (1), that crop will now operate on the padded WCS, so the final WCS is correct.

No additional changes needed to quality crop logic, except:
- Make sure the pad shift is applied BEFORE the quality crop block.

## [x] 3) Apply master tile crop (Phase 5) compatibility
Phase 5 "apply_crop" uses zemosaic_utils.crop_image_and_wcs(), which slices data and subtracts dh/dw from CRPIX.

With step (1) fixed, CRPIX should no longer be "mysteriously negative" due to missing padding, which avoids the function’s safety clamp.

Optional improvement (safe):
- In zemosaic_utils.crop_image_and_wcs():
  - if CRPIX clamp to 1.0 triggers, emit a WARN log via callback:
    "CropUtil: CRPIX clamped (old=..., raw_new=..., clamped=...)"
This does not change behavior, only makes future regressions obvious.

## [ ] 4) Quick validation checklist (manual)
A) Re-run the failing dataset.
B) Confirm logs contain:
   - MT_AUTO_PAD: detected...
   - MT_WCS_PAD: applied dx/dy...
   - MT_CROP: quality-based rect=...
C) Pick one master tile FITS:
   - Verify header has ZMT_PADX/ZMT_PADY and ZMT_QBOX.
   - CRPIX should be consistent (typically more plausible; not shifted by exactly -pad offsets).

D) Confirm final mosaic has no tile ghosts.

## [x] 5) Regression safety
- Do nothing when AUTO_PAD not triggered (pad offsets 0).
- Do not change aligner function signatures.
- Do not touch grid_mode.py or SDS pipeline code paths.
- Keep performance impact negligible: one WCS deepcopy per padded tile only.

## [ ] 6) Extra sanity (optional, DEBUG only)
If debug_tile:
- log pixel_shape mismatch before/after pad shift:
  expected shape vs wcs.pixel_shape
This helps catch future WCS shape drift.
