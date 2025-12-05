## 🟧 `followup.md`

```markdown
# 🔁 Follow-up checklist: Grid/Survey FITS & assembly stabilization

Use this checklist to verify that your changes are correct and complete.

## 1. FITS loading (Grid mode)

- [ ] All FITS in the test dataset (e.g. Seestar M106 mosaic) load **without**:
      `Cannot load a memory-mapped image: BZERO/BSCALE/BLANK...`
- [ ] Grid mode uses `memmap=False` and `do_not_scale_image_data=True` (or a shared robust helper),
      consistent with the rest of the project.
- [ ] When a FITS truly cannot be read (corrupt file), it logs a `[GRID]` error and continues
      with the remaining frames.

## 2. Tile assembly (`assemble_tiles`)

- [ ] `assemble_tiles` no longer raises any `ValueError` or broadcasting error for shape mismatch.
- [ ] Tile bounding boxes are clamped to the global mosaic dimensions before slicing.
- [ ] Offsets into the tile data (`off_x`, `off_y`) are correctly computed for out-of-bounds bboxes.
- [ ] `used_h` and `used_w` are always positive for tiles that are actually written.
- [ ] `data` is always in shape `(H, W, C)` before cropping (2D → 3D handled, >3D → squeezed).
- [ ] `data_crop` and `mosaic_sum[slice_y, slice_x, :]` always share the same H and W.
- [ ] Optional: When a tile is skipped, a `[GRID]` debug/warn log explains why.

## 3. Behavior on the example dataset

- [ ] Running Grid mode on the M106 example (with stack_plan.csv) successfully:
      - builds the grid,
      - processes tiles,
      - assembles a final mosaic.
- [ ] No silent failure: if Grid mode aborts early, a clear `[GRID]` log explains why.

## 4. Fallback behavior

- [ ] If **no frames** could be loaded or no valid WCS are available, Grid mode logs
      `[GRID] No frames with valid WCS found` (or similar) and aborts cleanly.
- [ ] When Grid mode fails for any reason, `zemosaic_worker` logs
      `[GRID] Grid/Survey mode failed, continuing with classic pipeline`.
- [ ] Classic pipeline still runs and produces output in such failure cases.

## 5. Regression safety

- [ ] Classic pipeline behavior (without stack_plan.csv) is unchanged.
- [ ] No changes were made to clustering, master tiles, or Phase 5 logic.
- [ ] No new dependencies were introduced unnecessarily.
- [ ] All new logs are properly tagged `[GRID]`.
