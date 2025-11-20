---

## 🧾 `followup.md`

```markdown
# Follow-up — Implement SDS “Global Mosaic = Final” Pipeline

## 0. Preparation

1. Locate the main orchestration logic:
   - `zemosaic_worker.py` (or similar worker module).
   - Search for:
     - `assemble_global_mosaic_sds`
     - `assemble_global_mosaic_first`
     - `assemble_final_mosaic_reproject_coadd`
     - Log lines:
       - `"P4 - Mosaic-First global coadd finished"`
       - `"Phase 5: Final assembly (Reproject & Coadd)..."`
       - `"run_info_phase5_finished_reproject_coadd"`

2. Identify where SDS configuration is read:
   - Flag(s) like `use_sds`, `sds_enabled`, `mosaic_first_mode`, etc.
   - Where they are passed into the worker/orchestration.

3. Inspect coverage logic:
   - Where the coverage map is currently computed and used.
   - Where auto-crop and alpha mask are applied (likely near the saving phase / Phase 5).

---

## 1. Make SDS global mosaic outputs explicit

Goal: **After Mosaic-First global coadd in SDS**, always obtain:

- `final_mosaic_data_HWC` (float32, shape `(H, W, C)`),
- `final_coverage_HW` (float32, shape `(H, W)`).

Steps:

1. In the SDS pipeline implementation (probably in `zemosaic_align_stack.py` or similar), find the function that performs:
   - `"P4 - Mosaic-First global coadd finished (kappa_sigma) ..."`.

2. Modify this function so that it:
   - returns both:
     - `mosaic_data_HWC` (the global SDS image),
     - `coverage_HW` (the corresponding coverage map).

3. In `zemosaic_worker.py` (orchestrator), in the **SDS ON** path:
   - Capture these outputs:
     ```python
     sds_mosaic_data_HWC, sds_coverage_HW = assemble_global_mosaic_sds(...)
     ```
   - Store them in variables that will be reused in Phase 5:
     ```python
     final_mosaic_data_HWC = sds_mosaic_data_HWC
     final_coverage_HW = sds_coverage_HW
     ```

4. Ensure the dtype / shape contract:
   - `final_mosaic_data_HWC`:
     - float32,
     - shape `(H, W, 3)` for RGB or `(H, W)` for mono depending on existing code.
   - `final_coverage_HW`:
     - float32,
     - shape `(H, W)`,
     - currently may be “number of contributions” or normalized. Clarify in code comments.

---

## 2. Coverage normalization & min_coverage_keep

Goal: **Mask low-coverage pixels** in SDS global mosaic.

1. Add a new configuration parameter:
   - In `zemosaic_config.py` or `solver_settings.py`:
     ```python
     # SDS-specific settings
     SDS_MIN_COVERAGE_KEEP = 0.4  # or 0.5, i.e. >=40–50% of max contributions
     ```
   - Provide a short docstring / comment explaining:
     - value in [0,1],
     - interpreted as “fraction of maximum possible contributions / frames”.

2. After `assemble_global_mosaic_sds` returns `(mosaic_data_HWC, coverage_HW)`:

   - Normalize coverage if needed:
     - If coverage is in raw “contribution count”:
       ```python
       max_cov = np.nanmax(coverage_HW)
       if max_cov > 0:
           coverage_norm = coverage_HW / max_cov
       else:
           coverage_norm = coverage_HW.copy()
       ```
     - If coverage is already in [0,1], simply set:
       ```python
       coverage_norm = coverage_HW
       ```

   - Apply threshold:
     ```python
     min_cov = config.SDS_MIN_COVERAGE_KEEP  # or from settings
     lowcov_mask = coverage_norm < min_cov

     mosaic_data_HWC[lowcov_mask] = np.nan
     coverage_HW[lowcov_mask] = 0.0
     ```

3. Store the final results:
   ```python
   final_mosaic_data_HWC = mosaic_data_HWC
   final_coverage_HW = coverage_HW
````

4. Logging:

   * Add log lines such as:

     * `"SDS: coverage normalized, min_coverage_keep=%.3f"`.
     * `"SDS: masked %d pixels below coverage threshold"`.

---

## 3. Phase 5 branching: SDS = polish, non-SDS = standard

Goal: **In Phase 5, skip full `reproject_and_coadd` when SDS is ON**.

1. Locate Phase 5 orchestration in `zemosaic_worker.py`:

   * Search for:

     * `"Phase 5: Final assembly (Reproject & Coadd)..."`
     * Call to `assemble_final_mosaic_reproject_coadd(...)`.

2. Introduce a clear branch:

   ```python
   if sds_enabled:
       # SDS branch: reuse global SDS mosaic
       log.info("Phase 5 (SDS): polish & save on global SDS mosaic (no extra reproject+coadd).")
       # Do NOT call assemble_final_mosaic_reproject_coadd here.
       mosaic_HWC = final_mosaic_data_HWC
       coverage_HW = final_coverage_HW
   else:
       # Non-SDS branch: legacy behavior
       log.info("Phase 5: Final assembly (Reproject & Coadd)...")
       mosaic_HWC, coverage_HW = assemble_final_mosaic_reproject_coadd(...)
   ```

3. Make sure that:

   * The **rest of Phase 5** (renorm / IBN / crop / alpha / save) operates on:

     * `mosaic_HWC`,
     * `coverage_HW`,
   * regardless of SDS ON/OFF.

4. Preserve existing log messages for the **non-SDS** branch to avoid confusion and help regression comparison.

---

## 4. Phase 5 polish operations on SDS mosaic

Goal: Reuse existing “polish” logic on the SDS global mosaic.

1. Identify existing functions that:

   * Apply global IBN / renorm,
   * Perform auto-crop based on coverage,
   * Build alpha masks,
   * Prepare data for `SaveFITS` and `SaveFITS coverage`.

2. Refactor so these functions operate on generic inputs:

   * `mosaic_HWC`,
   * `coverage_HW`,
   * plus any needed metadata (WCS, headers, etc.).

3. In SDS Phase 5 branch:

   * Call the same polish functions with:

     ```python
     mosaic_HWC = final_mosaic_data_HWC
     coverage_HW = final_coverage_HW
     mosaic_HWC, coverage_HW = apply_global_renorm_if_enabled(mosaic_HWC, coverage_HW, config, ...)
     mosaic_HWC, coverage_HW, crop_slices = apply_coverage_based_auto_crop(mosaic_HWC, coverage_HW, ...)
     alpha_HW = build_alpha_mask_from_coverage(coverage_HW, ...)
     ```
   * Then call save functions:

     ```python
     save_final_mosaic_fits(output_path, mosaic_HWC, wcs, ...)
     save_final_coverage_fits(output_cov_path, coverage_HW, wcs, ...)
     save_preview_png(output_preview_path, mosaic_HWC, ...)
     ```

4. Ensure that **NaN pixels** are handled correctly in:

   * renorm,
   * auto-crop,
   * saving routines (e.g. convert NaNs to 0 or masked pixels as needed).

---

## 5. Non-SDS pathway: ensure no change in behavior

1. Carefully compare:

   * Before vs after code for the non-SDS branch in Phase 5.
   * Ensure the same sequence of calls is preserved:

     * `assemble_final_mosaic_reproject_coadd`,
     * coverage / alpha logic,
     * save.

2. Confirm that:

   * No extra normalization, cropping, or masking is introduced **only for SDS** (or if introduced, is behind SDS flag).

3. Where needed, duplicate code paths to separate SDS polish branch from non-SDS path **without altering** the non-SDS logic.

---

## 6. Optional: GUI binding for SDS_MIN_COVERAGE_KEEP

*(Nice-to-have, not mandatory.)*

1. In `zemosaic_filter_gui_qt.py` (or relevant GUI module):

   * Add an advanced SDS field:

     * Slider or spinbox labeled:

       * `"SDS minimum coverage (0–1, default 0.4)"`.
   * Bind its value to `SDS_MIN_COVERAGE_KEEP` in config/settings.

2. Ensure:

   * Default value matches the config constant.
   * Input is clamped to [0.0, 1.0].

---

## 7. Logging and debug

1. In SDS branch, add clear log markers:

   * After SDS global coadd:

     * `"SDS: Mosaic-First global coadd completed — shape=(H,W,C), frames=%d"`.

   * After coverage normalization & masking:

     * `"SDS: coverage normalized (max=%.3f), min_coverage_keep=%.3f"`.
     * `"SDS: %d pixels masked as low coverage"`.

   * Before Phase 5 SDS polish:

     * `"Phase 5 (SDS): polish & save, skipping reproject_and_coadd on mega-tiles."`.

2. Keep the existing `"Phase 5: Final assembly (Reproject & Coadd)..."` log line **only** in non-SDS branch.

---

## 8. Testing

### 8.1 SDS ON

1. Run a known SDS dataset (e.g., one that previously produced strong noisy skirts).

2. Verify logs:

   * That SDS branch is taken.
   * That `assemble_final_mosaic_reproject_coadd` is **not** called in Phase 5.
   * That coverage thresholding and masking are reported.

3. Inspect outputs:

   * `final_mosaic.fits`:

     * Borders with low coverage should be NaN / cropped.
     * Visual noise on edges should be reduced.
   * `final_coverage.fits`:

     * Pixel values below `SDS_MIN_COVERAGE_KEEP` should be 0.

4. Check that the `preview.png` looks consistent with the cropped core region.

### 8.2 SDS OFF

1. Run the same dataset with SDS disabled.

2. Verify logs:

   * Legacy Phase 5 `Reproject & Coadd` is invoked.
   * No SDS-specific log entries.

3. Compare:

   * Final FITS & PNG with pre-refactor version.
   * They should be visually identical (or extremely close).

4. Run any existing non-SDS test datasets (regression suite, if available).

---

## 9. Final cleanup

1. Ensure:

   * No unused variables / flags remain.
   * Comments clearly document the SDS vs non-SDS paths.
   * Configuration parameters are documented (e.g., in comments or wiki docs).

2. Optionally:

   * Add a short developer note in code for future maintainers:

     * Explaining that:

       * **SDS ON uses the global SDS mosaic as the final image**, Phase 5 = polish only.
       * **SDS OFF** keeps the classic multi-phase reproject+coadd assembly.

```
