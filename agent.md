# Mission: Restore stable "classic" (non-grid) Phase 5 behaviour in `zemosaic_worker.py`

## High-level goal

Bring the **classic / non-grid pipeline** in `zemosaic_worker.py` back to a **stable, production-ready state**, matching the behaviour of the known-good reference file:

- `zemosaic_worker_non grid_ok.py`

The target is:

- **No more green-tinted mosaics** in classic mode.
- **No change** to Grid Mode or SDS phases.
- Keep the newer Phase 5 infrastructure (two-pass renorm, GPU, telemetry) but with the **same functional behaviour** as the reference file for the **classic, non-grid path**.

We do **not** need to re-invent the algorithm.  
The mission is to **align the call site(s)** of the Phase 5 pipeline in `zemosaic_worker.py` with the logic that exists in `zemosaic_worker_non grid_ok.py`.

---

## Context

- This project has **two workers**:
  - `zemosaic_worker_non grid_ok.py` → older worker, classic mode, **chromatically correct** (no green cast). This is our **behavioural reference**.
  - `zemosaic_worker.py` → current worker, supports **Grid Mode, SDS, GPU**, etc. In classic mode it still produces mosaics with a strong **green tint**.

- We already **disabled** the final RGB equalisation call in `zemosaic_worker.py`:
  - The block around `_apply_final_mosaic_rgb_equalization(...)` (lines ~6778–6796) is **commented out** and must remain disabled in this mission.
  - Despite that, the classic mosaic is still green → the problem is **not only** the final RGB eq.

- The remaining functional difference between the “OK” worker and the “green” worker in Phase 5 is:
  - In the reference file, the Phase 5 post-stack pipeline is called with:
    - `enable_lecropper_pipeline=False`
    - `enable_master_tile_crop=False`
  - In the new worker, we added logic that re-derives:
    - `enable_final_lecropper` and `enable_final_master_crop`
    - and passes them to `_apply_phase5_post_stack_pipeline(...)` on the **final mosaic**.

- This leads to **replaying the quality pipeline** (lecropper, alt-az cleanup, master crop, etc.) on the **assembled mosaic**, possibly **several times** (Phase 5 + `_derive_final_alpha_mask`), on data that already had master-tile level processing applied.

Result: the classic / non-grid mosaic shows a **strong green cast and ugly background**.

The fix is **not** to add more processing, but to **restore the same call parameters** as the known-good worker for the **classic, non-grid Phase 5 path**.

---

## Files to edit

1. `zemosaic_worker.py`  
   - Main worker with Grid/SDS logic.
2. (Read-only reference) `zemosaic_worker_non grid_ok.py`  
   - Do **not** modify this file.  
   - Use it only as a **behavioural reference**.

---

## Constraints

- **Do not modify**:
  - `grid_mode.py`
  - Any Grid-specific or SDS-specific code paths
  - FITS reading / writing logic
  - Existing RGB **final** equalisation helper `_apply_final_mosaic_rgb_equalization` (its implementation can stay as is, but the call must remain disabled).

- Keep the following intact:
  - Two-pass renormalisation infrastructure in Phase 5.
  - GPU/CPU parallel plan logic.
  - Telemetry controls.

- Focus only on:
  - The **classic / non-grid** Phase 5 path (after the final mosaic has been assembled).

---

## Task 1 – Restore Phase 5 call parameters for classic mode

### Goal

Make sure that, in **classic / non-grid mode** (non-SDS), the call to `_apply_phase5_post_stack_pipeline(...)` in `zemosaic_worker.py` uses the **same effective parameters** as in `zemosaic_worker_non grid_ok.py`, i.e.:

- `enable_lecropper_pipeline=False`
- `enable_master_tile_crop=False`

…while still passing the current `final_quality_pipeline_cfg`, two-pass parameters, WCS, coverage, etc.

### Steps

1. In `zemosaic_worker.py`, locate the **Phase 5 post-stack pipeline call** for the final mosaic, **outside** Grid Mode and **outside** SDS Phase 5:
   - This is the call to `_apply_phase5_post_stack_pipeline(...)` that happens:
     - After `assemble_final_mosaic_reproject_coadd(...)` or `assemble_final_mosaic_incremental(...)`,
     - On variables like `final_mosaic_data_HWC`, `final_mosaic_coverage_HW`, `final_alpha_map`.

2. You will find some logic similar to:

   ```python
   enable_final_lecropper = False
   if final_quality_pipeline_cfg:
       enable_final_lecropper = bool(
           final_quality_pipeline_cfg.get("quality_crop_enabled")
           or final_quality_pipeline_cfg.get("altaz_cleanup_enabled")
       )

   enable_final_master_crop = False
   if final_quality_pipeline_cfg:
       enable_final_master_crop = bool(
           final_quality_pipeline_cfg.get("master_tile_crop_enabled")
       )

   final_mosaic_data_HWC, final_mosaic_coverage_HW, final_alpha_map = _apply_phase5_post_stack_pipeline(
       final_mosaic_data_HWC,
       final_mosaic_coverage_HW,
       final_alpha_map,
       enable_lecropper_pipeline=enable_final_lecropper,
       pipeline_cfg=final_quality_pipeline_cfg,
       enable_master_tile_crop=enable_final_master_crop,
       ...
   )
````

3. For the **classic / non-grid path**, change this call so that it matches the reference worker’s behaviour:

   * Replace the dynamic `enable_final_lecropper` and `enable_final_master_crop` by **hard-coded `False`** for this specific call, just like in `zemosaic_worker_non grid_ok.py`.

   * The final call in the classic / non-grid path must look like:

   ```python
   final_mosaic_data_HWC, final_mosaic_coverage_HW, final_alpha_map = _apply_phase5_post_stack_pipeline(
       final_mosaic_data_HWC,
       final_mosaic_coverage_HW,
       final_alpha_map,
       enable_lecropper_pipeline=False,
       pipeline_cfg=final_quality_pipeline_cfg,
       enable_master_tile_crop=False,
       master_tile_crop_percent=master_tile_crop_percent_config,
       two_pass_enabled=bool(two_pass_enabled),
       two_pass_sigma_px=two_pass_sigma_px,
       two_pass_gain_clip=gain_clip_tuple,
       final_output_wcs=final_output_wcs,
       final_output_shape_hw=final_output_shape_hw,
       use_gpu_two_pass=use_gpu_phase5_flag,
       logger=logger,
       collected_tiles=collected_tiles_for_second_pass,
       fallback_two_pass_loader=fallback_two_pass_loader,
       parallel_plan=parallel_plan,
       telemetry_ctrl=None if sds_mode_phase5 else telemetry_ctrl,
   )
   ```

   Notes:

   * Keep all other arguments exactly as they currently are (two-pass, WCS, coverage, telemetry, etc.).
   * You may keep the `enable_final_lecropper` / `enable_final_master_crop` variables if they are used elsewhere (e.g. for SDS or other modes), but **do not** use them for this specific call in the classic / non-grid path.

4. Do **not** change the call site(s) of `_apply_phase5_post_stack_pipeline` that are specific to **SDS** or **Grid Mode**, unless they already share this common code path and a change would clearly break their behaviour.

5. Ensure that after this call, any `collected_tiles_for_second_pass` cleanup (like `.clear()`) still happens as in the current code.

---

## Task 2 – Keep final RGB equalisation disabled

### Goal

Ensure that the final RGB equalisation step on the mosaic remains **disabled**.

### Steps

1. In `zemosaic_worker.py`, locate the block that calls `_apply_final_mosaic_rgb_equalization(...)` (around lines 6778–6796 in the user’s version).

2. This block has already been commented out by the user.
   Leave it **commented** / disabled.

3. Do **not** add any new calls to `_apply_final_mosaic_rgb_equalization` in this mission.

4. The helper function `_apply_final_mosaic_rgb_equalization` can stay in the file, untouched, for potential future use.

---

## Task 3 – Sanity checks

### Goal

Make sure the project still runs and that the classic mosaic no longer has a strong green cast.

### Steps

1. Verify that `zemosaic_worker.py` imports still succeed (no unused-variable errors that break flake/mypy if such checks exist).

2. Run a classic (non-grid) mosaic generation using the same project that previously produced a green-tinted image.

3. Confirm:

   * No `[RGB-EQ] final mosaic` logs are emitted (because the final RGB eq is disabled).
   * Phase 5 logs still show the two-pass / coverage steps executing.
   * The resulting mosaic has:

     * Normal colour balance (no global green cast),
     * Similar appearance to the output produced by `zemosaic_worker_non grid_ok.py`.

4. If Grid Mode or SDS modes are exercised, they **must still work** as before.

---

## Acceptance criteria

* [ ] In `zemosaic_worker.py`, the Phase 5 post-stack pipeline call for the classic / non-grid path uses:

  * `enable_lecropper_pipeline=False`
  * `enable_master_tile_crop=False`
  * with all other arguments unchanged.
* [ ] The final RGB equalisation call remains disabled.
* [ ] The classic (non-grid) mosaic output no longer shows a strong green cast and visually matches (or is very close to) the reference behaviour of `zemosaic_worker_non grid_ok.py`.
* [ ] Grid Mode and SDS modes continue to work without regression.

