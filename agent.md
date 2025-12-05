## `agent.md`

# ZeMosaic – Grid mode output harmonisation

## 0. Context

Project: `zemosaic` (ZeMosaic).

Goal: **Grid/Survey mode must produce the same kind of FITS outputs as the classic pipeline**, i.e.:

- 1 **science FITS** in float32 with full WCS (RGB cube if colour).
- 1 optional **viewer FITS** in 16-bit “standard” RGB format, controlled by the same `save_final_as_uint16` flag and using the same helper as the normal pipeline.
- 1 optional **coverage FITS** (`*_coverage.fits`) describing pixel coverage / weight, also with WCS.

Right now, Grid mode:
- writes `mosaic_grid.fits` via `save_fits_image(..., save_as_float = not save_final_as_uint16, ...)`,
- appends a `WEIGHT` extension directly on that file,
- does **not** call `write_final_fits_uint16_color_aware`,
- does **not** write a `_coverage.fits`.

This is a behavioural divergence from the “normal” pipeline and produces FITS that some viewers interpret as grayscale with strange histograms, even though the data are RGB.

Keep all existing behaviour of **tiles** (`tile_xxxx.fits`) and **normal pipeline** completely unchanged.

---

## 1. Files to inspect first

- `grid_mode.py`
  - `assemble_tiles(...)`
  - `run_grid_mode(...)`
  - any helper that computes/uses `weight_sum`, `overlap_*`, etc.
- `zemosaic_utils.py`
  - `save_fits_image(...)`
  - `write_final_fits_uint16_color_aware(...)`
- `zemosaic_worker.py`
  - Phase 6: final mosaic save logic
  - how `_coverage.fits` is produced
- (read-only) `zemosaic_config.py` if needed for flags.

Do **not** change the overall grid logic, clustering, stacking or tile generation; we only harmonise the **final outputs**.

---

## 2. Target behaviour

### 2.1 Science FITS (Grid mode)

For the main Grid mosaic `mosaic_grid.fits`:

- Always save **as float32** with WCS, same as the normal pipeline.
- This file is the **science** product, equivalent to `final_fits_path` in `zemosaic_worker.py`.

Concretely in `assemble_tiles(...)`:

- After RGB equalisation and before saving, we already build:

  ```python
  output_data = np.asarray(mosaic, dtype=np.float32, copy=False)
  axis_order = "HWC" if output_data.ndim == 3 else None
````

* Replace the current call to `save_fits_image`:

  ```python
  save_fits_image(
      image_data=output_data,
      output_path=str(output_path),
      header=header,
      overwrite=True,
      save_as_float=not save_final_as_uint16,  # OLD
      legacy_rgb_cube=legacy_rgb_cube,
      progress_callback=progress_callback,
      axis_order=axis_order,
  )
  ```

  by:

  ```python
  save_fits_image(
      image_data=output_data,
      output_path=str(output_path),
      header=header,
      overwrite=True,
      save_as_float=True,          # always float32, like normal pipeline
      legacy_rgb_cube=legacy_rgb_cube,
      progress_callback=progress_callback,
      axis_order=axis_order,
  )
  ```

* Keep the existing `WEIGHT` extension append logic as-is (this is useful science metadata).

### 2.2 Viewer FITS for Grid mode

Grid mode must also produce a **viewer FITS** in 16-bit RGB when `save_final_as_uint16` is `True`, using the **same helper** as the normal pipeline.

Implementation details:

1. At the top of `grid_mode.py`, extend the utils import:

   ```python
   from zemosaic_utils import debayer_image, load_and_validate_fits, save_fits_image
   ```

   → change to:

   ```python
   from zemosaic_utils import (
       debayer_image,
       load_and_validate_fits,
       save_fits_image,
       write_final_fits_uint16_color_aware,
   )
   ```

2. In `assemble_tiles(...)`, **after** the successful save of `mosaic_grid.fits` and the optional `WEIGHT` extension, add a new block to build a viewer FITS:

   * Only if:

     * `save_final_as_uint16` is `True`,
     * and `output_data.ndim == 3` and `output_data.shape[-1] in (3, 4)` (RGB(A)), or if you prefer, follow the same “is_rgb” logic as in `zemosaic_worker`.

   * Compute a viewer path:

     ```python
     viewer_path = output_path.with_name(output_path.stem + "_viewer.fits")
     ```

   * Call `write_final_fits_uint16_color_aware(...)`:

     ```python
     try:
         is_rgb = output_data.ndim == 3 and output_data.shape[-1] >= 3
         write_final_fits_uint16_color_aware(
             str(viewer_path),
             output_data,      # HWC float32
             header=header,
             force_rgb_planes=is_rgb,
             legacy_rgb_cube=legacy_rgb_cube,
             overwrite=True,
         )
         _emit(
             f"Grid viewer FITS saved to {viewer_path}",
             lvl="INFO",
             callback=progress_callback,
         )
     except Exception as exc_viewer:
         _emit(
             f"Grid viewer FITS save failed ({exc_viewer})",
             lvl="WARN",
             callback=progress_callback,
         )
     ```

   * Do **not** attach an ALPHA extension here unless it’s trivial. Alpha is optional for the user’s use-case; the key requirement is a clean RGB cube readable by mainstream software (Siril, etc.).

### 2.3 Coverage FITS for Grid mode

Grid mode already keeps a `weight_sum` array during assembly. We want a separate **coverage map** saved as `mosaic_grid_coverage.fits` (or `{output_stem}_coverage.fits`), with a similar header to the normal pipeline.

Implementation:

1. In `assemble_tiles(...)`, after `weight_sum` is fully accumulated and after the main mosaic is saved, compute a coverage HW array:

   ```python
   coverage_hw: np.ndarray | None = None
   try:
       if weight_sum.ndim == 3:
           coverage_hw = np.sum(weight_sum, axis=-1).astype(np.float32, copy=False)
       else:
           coverage_hw = weight_sum.astype(np.float32, copy=False)
   except Exception as exc_cov:
       _emit(
           f"Coverage: failed to derive coverage map from weight_sum ({exc_cov})",
           lvl="WARN",
           callback=progress_callback,
       )
       coverage_hw = None
   ```

2. If `coverage_hw` is not `None` and has at least some non-zero pixels, build a coverage header:

   ```python
   if coverage_hw is not None and np.any(coverage_hw > 0):
       cov_header = fits.Header()
       try:
           if getattr(grid, "global_wcs", None) is not None and hasattr(grid.global_wcs, "to_header"):
               cov_header.update(grid.global_wcs.to_header(relax=True))  # type: ignore[attr-defined]
       except Exception:
           pass
       cov_header["EXTNAME"] = ("COVERAGE", "Coverage Map")
       cov_header["BUNIT"] = ("count", "Pixel contributions or sum of weights")
   ```

3. Save coverage as a separate FITS:

   ```python
   cov_path = output_path.with_name(output_path.stem + "_coverage.fits")
   try:
       save_fits_image(
           image_data=coverage_hw,
           output_path=str(cov_path),
           header=cov_header,
           overwrite=True,
           save_as_float=True,
           axis_order="HWC",
       )
       _emit(
           f"Grid coverage map saved to {cov_path}",
           lvl="INFO",
           callback=progress_callback,
       )
   except Exception as exc_cov_save:
       _emit(
           f"Coverage: failed to save {cov_path} ({exc_cov_save})",
           lvl="WARN",
           callback=progress_callback,
       )
   ```

This mimics the `_coverage.fits` creation from `zemosaic_worker.py`, but using `grid.global_wcs` and `weight_sum`.

---

## 3. Respect GUI / config flags

* `run_grid_mode(...)` already receives:

  * `save_final_as_uint16`,
  * `legacy_rgb_cube`,
  * `grid_rgb_equalize`,
    and passes them to `assemble_tiles(...)`.
    Keep this path intact; just change the **internal handling** in `assemble_tiles(...)` as described.

* Do **not** change the semantics or default values of these flags.

---

## 4. Logging

* Reuse the existing `_emit(...)` pattern.
* Add a few clear log lines with prefix `[GRID]` or “Grid viewer / Grid coverage” so that `zemosaic_worker.log` clearly shows:

  * final mosaic shape & dtype,
  * viewer FITS path when created,
  * coverage FITS path when created,
  * any failure with a short but explicit message.

---

## 5. Tests / sanity checks

Add or update tests if present, but at minimum:

1. **Manual sanity script** (you can add it as a docstring snippet in comments):

   ```python
   from astropy.io import fits
   import numpy as np

   # Science FITS (float32)
   h = fits.open("mosaic_grid.fits", do_not_scale_image_data=True)
   sci = h[0].data
   print(sci.shape, sci.dtype)      # expect (H, W, 3) or (3, H, W) depending on axis_order (check)
   h.close()

   # Viewer FITS (uint16/int16)
   v = fits.open("mosaic_grid_viewer.fits", do_not_scale_image_data=True)
   arr = v[0].data
   print(arr.shape, arr.dtype)      # expect cube RGB with int16 (BITPIX=16 + BZERO)
   v.close()

   # Coverage
   c = fits.open("mosaic_grid_coverage.fits", do_not_scale_image_data=True)
   cov = c[0].data
   print(cov.shape, cov.dtype)      # expect (H, W) float32
   c.close()
   ```

2. Confirm that:

   * tiles (`tile_0001.fits`, etc.) are unchanged,
   * normal pipeline outputs are unchanged,
   * Grid mode now behaves just like normal pipeline from user perspective:

     * science float32,
     * viewer 16-bit RGB readable in mainstream astro tools,
     * coverage map available for planning.

---

## 6. Constraints

* Do not change the clustering, stacking or alignment logic.
* Do not touch the semantics of `grid_rgb_equalize`.
* Avoid heavy refactors; keep edits local to `grid_mode.py` (and imports) unless strictly needed.
* Keep code style consistent with existing file (logging style, typing, etc.).

