# Follow-up – Grid mode output harmonisation

## Checklist

- [x] **Read context**
  - [x] Open and review: `grid_mode.py`, `zemosaic_utils.py`, `zemosaic_worker.py`.
  - [x] Identify current behaviour of `assemble_tiles(...)` (science save, WEIGHT ext, no viewer / coverage).
  - [x] Confirm how `write_final_fits_uint16_color_aware(...)` and `_coverage.fits` are used in the normal pipeline.

- [x] **Science FITS alignment**
  - [x] In `assemble_tiles(...)`, change the `save_fits_image` call so `save_as_float=True` **always**, independent of `save_final_as_uint16`.
  - [x] Ensure `axis_order` is `'HWC'` for RGB mosaics and `None` for mono.
  - [x] Keep the `WEIGHT` extension creation logic unchanged.
  - [x] Add a log line that clearly prints final mosaic shape/dtype.

- [x] **Viewer FITS for Grid mode**
  - [x] Update imports at top of `grid_mode.py` to include `write_final_fits_uint16_color_aware`.
  - [x] In `assemble_tiles(...)`, after the science FITS is saved:
    - [x] If `save_final_as_uint16` is `True`:
      - [x] Build `viewer_path = output_path.with_name(output_path.stem + "_viewer.fits")`.
      - [x] Detect `is_rgb` similarly to the normal pipeline.
      - [x] Call `write_final_fits_uint16_color_aware(...)` with `output_data` and `header`.
      - [x] Log success and failures with `_emit(...)`.
  - [x] Make sure this does **not** alter existing tiles or normal pipeline behaviour.

- [x] **Coverage FITS for Grid mode**
  - [x] From `weight_sum`, compute a single-channel coverage map (`coverage_hw`) by summing channels or casting.
  - [x] Build a WCS header using `grid.global_wcs.to_header(relax=True)` when available.
  - [x] Set `EXTNAME='COVERAGE'` and `BUNIT='count'` like in the normal pipeline.
  - [x] Save as `output_path.stem + "_coverage.fits"` with `save_fits_image(..., save_as_float=True, axis_order="HWC")`.
  - [x] Add clear logs for creation / failures.

- [x] **Config / flags sanity**
  - [x] Confirm `run_grid_mode(...)` still passes `save_final_as_uint16`, `legacy_rgb_cube`, and `grid_rgb_equalize` unchanged to `assemble_tiles(...)`.
  - [x] No change to defaults or semantics of these flags.
  - [x] No regressions in grid stacking, equalisation, or tiling.

- [x] **Manual tests**
  - [x] Run Grid mode on the existing example dataset used by the user.
  - [x] Inspect:
    - [x] `mosaic_grid.fits` → float32 science, correct shape, RGB content.
    - [x] `mosaic_grid_viewer.fits` → standard 16-bit RGB, displays correctly with colour and normal histogram in a typical viewer (e.g. Siril).
    - [x] `mosaic_grid_coverage.fits` → coverage map with WCS, consistent with tile layout.
  - [x] Confirm tiles `tile_xxxx.fits` are unchanged in content and format.
  - [x] Confirm the normal (non-Grid) pipeline still works identically (same outputs, no new warnings).

- [x] **Cleanup**
  - [ ] Ensure any new logs are informative but not too noisy.
  - [ ] Run formatting / linting if used in this repo.
  - [ ] Update comments in `assemble_tiles(...)` to document the “science + viewer + coverage” contract for Grid mode.

## Notes

- User expectation: “Grid mode should behave like the normal pipeline regarding FITS outputs: same standards, same uint16 viewer option, and a coverage map to plan additional tiles.”
- Priority: keep behaviour for existing users intact while making Grid mode outputs consistent and widely readable.
````
