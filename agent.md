# Mission: Fix green cast in classic mode by restoring master-tile lecropper & disabling extra colour tweaks

## High-level goal

The current unified worker `zemosaic_worker.py` still produces a **green-tinted mosaic** in classic / non-grid mode, even after disabling the final RGB equalisation and simplifying Phase 5 options.

We have a **known-good reference**:

- `zemosaic_worker_non grid_ok.py` → classic mode, no grid, **clean colour** and proper master tiles.

The mission is to:

1. **Guarantee that master tiles in classic mode (non-SDS / non-Grid) go through the full lecropper pipeline**, exactly like in `zemosaic_worker_non grid_ok.py`.
2. Keep the final mosaic free from aggressive, redundant colour operations (no extra RGB-EQ, no extra lecropper on the assembled mosaic).
3. Reach a **production-ready classic mode** with clean background and no green cast, even if the root cause is not fully analysed.

We focus only on the **classic pipeline**; Grid and SDS must not be broken.

---

## Context

We already tried:

- Disabling final mosaic RGB equalisation by commenting out the call to `_apply_final_mosaic_rgb_equalization(...)`.
- Aligning the Phase 5 call to `_apply_phase5_post_stack_pipeline(...)` with the reference worker, using:
  - `enable_lecropper_pipeline=False`
  - `enable_master_tile_crop=False`
  for the classic / non-grid path.

Despite that, the mosaic remains green.  
Logs show that:

- `poststack_equalize_rgb` is still executed on master tiles (good).
- `MT_CROP: quality-based rect=...` is logged during Phase 3 (so quality crop is running).
- The final mosaic still looks heavily green.

This indicates that the **problem is rooted at the master-tile level** (background / gradients / alt-az artefacts) and/or that we are still doing **too many corrections** on top of each other.

We will therefore:

- **Strictly mirror** the master-tile lecropper/alt-az pipeline from `zemosaic_worker_non grid_ok.py`.
- Ensure no extra colour manipulation occurs after Phase 3 for classic mode.

---

## Files to edit

1. `zemosaic_worker.py`  *(main target)*
2. (read-only reference) `zemosaic_worker_non grid_ok.py`  *(do NOT modify; use for comparison only)*

---

## Constraints

- Do **not** touch:
  - `grid_mode.py`
  - Any Grid-specific or SDS-specific code paths
  - `lecropper.py` implementation itself (unless a blatant bug is found, which is not expected for this mission)
  - GUI files (`zemosaic_gui.py`, `zemosaic_gui_qt.py`, filter GUIs, etc.)

- The only worker to be edited is `zemosaic_worker.py`, and only:
  - in the **Phase 3 master-tile creation pipeline**, and
  - the already-known **Phase 5 / RGB-EQ final hooks**.

- Keep:
  - Two-pass renorm infrastructure, GPU/CPU parallel plan, and telemetry intact.
  - Existing log messages and `[CLÉ_POUR_GUI: ...]` keys.

---

## Task 1 – Mirror master-tile lecropper pipeline from reference worker

### Goal

Ensure that, in classic / non-grid mode, **every master tile** is processed by `lecropper` exactly as in the reference worker.

### Steps

1. In `zemosaic_worker_non grid_ok.py`, locate the function `create_master_tile(...)` (or equivalent) where:
   - A master tile `master_tile_stacked_HWC` is produced,
   - Quality-based crop is computed (`MT_CROP: quality-based rect=...`),
   - Then the lecropper pipeline is applied:

   ```python
   pipeline_cfg = {
       "quality_crop_enabled": quality_crop_enabled,
       "quality_crop_band_px": quality_crop_band_px,
       "quality_crop_k_sigma": quality_crop_k_sigma,
       "quality_crop_margin_px": quality_crop_margin_px,
       "quality_crop_min_run": quality_crop_min_run,
       "altaz_cleanup_enabled": altaz_cleanup_enabled,
       "altaz_margin_percent": altaz_margin_percent,
       "altaz_decay": altaz_decay,
       "altaz_nanize": altaz_nanize,
   }
   master_tile_stacked_HWC, pipeline_alpha_mask = _apply_lecropper_pipeline(master_tile_stacked_HWC, pipeline_cfg)
   ...
   pipeline_alpha_u8 = _normalize_alpha_mask(...)
   ...
   alpha_mask_for_quality = pipeline_alpha_mask or alpha_mask_out
   quality_gate_eval = _evaluate_quality_gate_metrics(..., alpha_mask=alpha_mask_for_quality, ...)
````

2. In `zemosaic_worker.py`, find the **corresponding block** (it should already look very similar, but with possible differences like `quality_crop_enabled_tile`, `quality_gate_enabled_tile`, etc.).

3. For the **classic / non-grid path**:

   * Ensure the `pipeline_cfg` keys and types match the reference worker:

     * `quality_crop_enabled` must reflect the global config for classic mode (not accidentally forced to `False` by per-tile logic, except when it is intentionally disabled by the user).
     * `altaz_cleanup_enabled` and related parameters must be passed through exactly as in the reference worker.

   * Confirm that `_apply_lecropper_pipeline(...)` is **always called** for each master tile in classic mode, i.e.:

     * There is no `if grid_mode` or `if sds_mode` or any early return that bypasses lecropper.
     * No branch that resets `quality_crop_enabled_tile` to `False` for classic mode.

   * Ensure that:

     * `pipeline_alpha_mask` is normalised via `_normalize_alpha_mask(...)`,
     * `alpha_mask_out` is computed and passed to the FITS save function,
     * `quality_gate_eval` uses `alpha_mask_for_quality` like in the reference worker.

4. Make sure that logs related to quality crop and lecropper remain intact:

   * `MT_CROP: quality-based rect=...`
   * warnings such as `MT_CROP: quality-based crop failed (...)` or `quality crop skipped (...)`.

5. If there are **per-tile overrides** (e.g. `quality_crop_enabled_tile`, `quality_gate_enabled_tile`), verify that:

   * They do not inadvertently **disable** the pipeline in classic mode unless explicitly requested.
   * Their default behaviour for classic mode mimics `quality_crop_enabled` / `quality_gate_enabled` from the reference worker.

---

## Task 2 – Keep post-master-tile colour stages minimal

### Goal

Avoid any extra colour transformations on the mosaic that might interact badly with master-tile processing.

### Steps

1. The final mosaic RGB equalisation (`_apply_final_mosaic_rgb_equalization(...)`) must remain **disabled by default**:

   * Either the call site is commented out,
   * Or it is guarded by a config flag `final_mosaic_rgb_equalize_enabled=False`.

   For this mission, we assume the call is already disabled and **should remain so**.

2. The Phase 5 call to `_apply_phase5_post_stack_pipeline(...)` for the **classic / non-grid** final mosaic should keep:

   ```python
   enable_lecropper_pipeline=False
   enable_master_tile_crop=False
   ```

   with all other arguments (two-pass, WCS, coverage, telemetry, etc.) preserved.

3. Do not add any **new** per-channel scaling or normalisation at the mosaic level.

---

## Task 3 – Sanity checks + diagnosis hooks

### Goal

Help confirm that the green cast originates from master-tile background, not from Phase 5.

### Steps

1. Add (or keep) an info log at the end of `create_master_tile(...)` that clearly indicates:

   * Whether the lecropper pipeline ran,
   * Whether alt-az cleanup was enabled for that tile,
   * The `quality_crop_enabled` state.

   Example:

   ```python
   pcb_tile(
       f"MT_PIPELINE: lecropper_applied=True, quality_crop={bool(quality_crop_enabled_tile)}, altaz_cleanup={bool(altaz_cleanup_enabled)}",
       prog=None,
       lvl="INFO_DETAIL",
   )
   ```

2. Ensure these logs are present in `zemosaic_worker.log` for all tiles in a classic run.

3. After running a classic mosaic on a known test project:

   * Confirm that `MT_CROP: quality-based rect=...` and `MT_PIPELINE: lecropper_applied=True...` appear for each tile.
   * Visually inspect the mosaic to verify that:

     * The global green cast has disappeared or is significantly reduced,
     * Background looks similar to the result produced by `zemosaic_worker_non grid_ok.py`.

---

## Acceptance criteria

* [x] In classic / non-grid mode, **every master tile** goes through `_apply_lecropper_pipeline(...)` with a `pipeline_cfg` equivalent to the reference worker.
* [x] Alt-az cleanup + quality crop are correctly applied and logged for master tiles in classic mode.
* [ ] Final mosaic RGB equalisation remains disabled (no `[RGB-EQ] final mosaic` logs).
* [ ] Phase 5 for classic mode does **not** reapply lecropper or master-tile crop on the final mosaic.
* [ ] The final classic mosaic has a **normal colour balance**, with no strong green cast, and visually matches the reference behaviour.
* [ ] Grid Mode and SDS remain functional without regression.


