# AGENT MISSION FILE — SDS / ZeSupaDupStack FIX

You are an autonomous coding agent working on the **ZeMosaic / ZeSeestarStacker** project.

The repository already contains (non-exhaustive):

- `run_zemosaic.py`
- `zemosaic_gui.py`
- `zemosaic_gui_qt.py`
- `zemosaic_filter_gui.py`
- `zemosaic_filter_gui_qt.py`
- `zemosaic_worker.py`
- `zemosaic_utils.py`
- `zemosaic_config.py`
- `zequalityMT.py`
- `lecropper.py`
- `zemosaic_align_stack.py`
- `locales/en.json`, `locales/fr.json`
- helper modules in `core/`, `locales/`, etc.

Your mission is to **fix and finalize the SDS (“ZeSupaDupStack”) mode**, both in the **Qt Filter GUI** and in the **worker**, so it matches the intended behaviour:

> **Build “mega-tiles” by stacking batches of multiple Seestar frames in a single global WCS, then stack those mega-tiles together.**  
> The goal is to **reduce the number of raw frames** (e.g. thousands → a few mega-tiles) while keeping all existing quality filters and mosaic logic intact.


---

## 1. Global Requirements

1. **Do NOT break existing workflows:**
   - Classic (non-Seestar) workflow must behave exactly as before.
   - Seestar Mosaic-First workflow (when SDS is OFF) must behave exactly as before.
   - GPU / CPU logic, memory chunking, Phase 4.5, two-pass coverage renorm, lecropper, ZeQualityMT, etc. must remain untouched except where explicitly stated.

2. **SDS = optional overlay:**
   - SDS is activated only when the existing SDS checkbox is ON in the Filter GUI (Qt), plus the internal `sds_mode` flag.
   - When SDS is OFF, behaviour must be strictly unchanged.

3. **Seestar only:**
   - SDS must *only* be used for Seestar sequences, as currently detected (`auto_detect_seestar`, `force_seestar_mode` and `_entry_is_seestar` in the worker).

4. **Keep logs and localization style:**
   - Use the existing worker logging helpers (`pcb_fn`, `_log_and_callback` style) and re-use the same logging keys style (`sds_info_*`, `sds_debug_*`, `sds_error_*`).
   - All new user-visible messages must be wired through localization keys in `en.json` / `fr.json` and accessed via the localization system, **not hard-coded strings** in the GUI.

5. **No regression in progress display / ETA / Phases:**
   - The progress, ETA and phase names shown in `zemosaic_gui_qt.py` must remain coherent.
   - The SDS work should still report via the existing Phase 5/6 (final assembly) messages and coverage summary logs.


---

## 2. SDS Concept and Target Behaviour

### 2.1. Current problem

Today, SDS builds coverage-based batches using only:

> “close batch as soon as coverage_threshold is reached”

This causes:
- On Seestar sequences where each frame already covers nearly the entire global WCS:
  - **each batch contains only 1 image**,
  - resulting in **N brutes → N reprojections**, which defeats the purpose of SDS (no real reduction).

Sometimes, the worker also logs `sds_error_no_valid_batches` and falls back to Mosaic-First without actually running SDS.

### 2.2. Target behaviour (what you must implement)

The intended SDS behaviour is:

1. **Global WCS**:
   - A single global WCS plan is built (already implemented).
   - SDS reuses that WCS for all batches.

2. **Build SDS batches = “mega-batch coverage passes”**:
   - Iterate over the list of valid Seestar frames (after Phase 2 filters).
   - Build batches of frames (indices / entries) such that:

     - Each batch is composed of **multiple frames** (never all degenerate to size 1 unless there truly is only one frame).
     - Each batch tries to reach at least a **coverage threshold** of the global WCS (e.g. 0.9–0.95).
     - There is a **minimum batch size** and a **target batch size** to avoid the “all batches = 1 image” degeneration.

3. **Per-batch mosaic (Mosaic-First restricted to that batch)**:
   - For each SDS batch, call the existing global mosaic assembly pipeline **Mosaic-First** but restricted to the frames of that batch (already implemented in `assemble_global_mosaic_sds` via `_assemble_global_mosaic_first_impl`).
   - Output a **batch mosaic** + **batch coverage** + **batch alpha** in the global WCS.

4. **Final SDS stack = stack of batch mosaics**:
   - Stack all batch mosaics together using the existing stacking engine `zemosaic_align_stack` with coverage as weights, as currently done in SDS.
   - Produce a final combined mosaic in the global WCS and pass it to Phase 5/6.
   - All downstream features (autocrop, lecropper, two-pass coverage renorm, alpha map, 16-bit save, etc.) remain unchanged.

5. **Fallback**:
   - If SDS cannot build **any valid batch**, or if any fatal error occurs in SDS, you must:
     - Log a clear SDS error message.
     - Cleanly **fallback** to the existing Mosaic-First workflow, as it does today.
   - SDS must never crash the entire worker.


---

## 3. SDS Batch Policy — Detailed Specification

You must implement a **batch policy** with **coverage + min size + target size**, applied **both**:

- In the GUI preview (`zemosaic_filter_gui_qt.py`) for display / preplan; and
- In the worker (`zemosaic_worker.py`) to actually define SDS batches when running.

### 3.1. Configuration (internal)

Add two new configuration keys (in `zemosaic_config.DEFAULT_CONFIG`):

- `sds_min_batch_size`
  - Type: integer
  - Default: `5`
  - Meaning: *Minimum* number of frames before a batch is allowed to be closed by coverage.
- `sds_target_batch_size`
  - Type: integer
  - Default: `10`
  - Meaning: *Soft* target; if the batch reaches this size but coverage is still below the threshold, the batch is closed anyway to avoid huge batches.

These keys:
- Are **not necessarily exposed in the GUI** for now (can stay config-only).
- Must be read in both the Filter GUI and the worker via the same configuration path used by `sds_coverage_threshold`.

### 3.2. Batch building algorithm (to be mirrored in GUI + worker)

Given:
- `images` = ordered list of entries (order already used by SDS; if an explicit sort exists, keep it),
- `coverage_threshold` in [0, 1] (e.g. 0.92),
- `min_batch_size` (e.g. 5),
- `target_batch_size` (e.g. 10),
- a global coverage grid (discretization already implemented in SDS).

Algorithm sketch:

1. Initialize:
   - `current_batch = []`
   - `current_coverage_grid = all zeros`
   - `batches = []`

2. For each image in `images`:
   - Add image to `current_batch`.
   - Update `current_coverage_grid` based on this image footprint in the global WCS (same logic as existing SDS).
   - Compute `coverage_fraction = covered_cells / total_cells`.

   - If **both**:
     - `len(current_batch) >= min_batch_size`
     - AND `coverage_fraction >= coverage_threshold`

     → **close the batch**:
     - append `current_batch` to `batches`
     - reset `current_batch`, `current_coverage_grid`.

   - Else, if:
     - `len(current_batch) >= target_batch_size`

     → **force close**:
     - append `current_batch` to `batches`
     - reset `current_batch`, `current_coverage_grid`.

3. After the loop:
   - If `current_batch` is non-empty:
     - If `len(current_batch) < min_batch_size` and there is already at least one batch:
       - Try to **merge** the leftover images into the last batch, as long as the total size remains reasonable (e.g. `<= 2 * target_batch_size`, or simply always merge leftovers into the last batch).
     - Else:
       - Append `current_batch` as a final batch.

4. Edge cases:
   - If there is **only one valid image in total**:
     - Allow a single batch of size 1.
   - If coverage grid or WCS information is missing or invalid:
     - Abort SDS batch building with a clean error and fallback to Mosaic-First.

You must implement this policy in a **single helper function** in each layer (GUI + worker) and reuse it, rather than duplicating ad-hoc logic.


---

## 4. Files to Modify

### 4.1. `zemosaic_config.py`

- Add new keys in `DEFAULT_CONFIG`:
  - `sds_min_batch_size` (default `5`).
  - `sds_target_batch_size` (default `10`).
- Ensure these defaults are used even if the user’s config file predates these keys.

### 4.2. `zemosaic_filter_gui_qt.py`

Goal: **SDS preview & preplan** must use the same SDS batch policy as the worker.

- Locate the SDS helper that builds batches for preview:
  - `_build_sds_batches_for_indices(...)` or equivalent.
- Modify it to:
  - Read `sds_min_batch_size` and `sds_target_batch_size` from the configuration/overrides (same mechanism as `sds_coverage_threshold`).
  - Use the **coverage + min + target** policy described in section 3.2.
  - Ensure the function still returns groups in the same shape as before (list of lists of indices or entries) so that:
    - The preview grouping UI works (group tree, counts).
    - The preplan groups are written into overrides (e.g. `overrides["preplan_master_groups"]`) exactly as before.

- Add minimal logging (to the Qt Filter log widget) when SDS preview is computed, e.g.:
  - number of SDS batches,
  - sizes of each batch.

- Do **not** change other grouping modes or the Tk filter GUI in this mission.

### 4.3. `zemosaic_worker.py`

Goal: **Runtime SDS assembly** must faithfully implement the same policy and avoid `sds_error_no_valid_batches` in normal cases.

- Locate:
  - `assemble_global_mosaic_sds(...)`,
  - any internal helper building SDS batches there.

- Tasks:

1. **Configuration:**
   - Read `sds_min_batch_size` / `sds_target_batch_size` from the config in the same way as `sds_coverage_threshold`.
   - Use safe coercion and defaults if they are missing or invalid (use `max(1, int(...))`).

2. **SDS batch building:**
   - Factor the SDS batch building logic into a clear helper inside the worker, applying the policy from section 3.2.
   - Use the same grid / WCS discretization already present in SDS.
   - Ensure this helper can be called either:
     - from `assemble_global_mosaic_sds`, or
     - from anywhere else SDS needs to compute coverage-based groups.

3. **Preplan integration:**
   - If `preplan_master_groups` are provided via overrides *and* SDS is active:
     - Option 1: use them as SDS batches directly *if* they cover the whole set of Seestar entries (recommended).
     - Option 2: or use them as seeds for SDS coverage grouping if appropriate.
   - The important part: **SDS must not silently ignore valid preplan groups**; it should exploit them when possible, or log clearly why they cannot be used.

4. **Runtime logging & diagnostics:**
   - Improve SDS logging:
     - Log when SDS starts, how many entries it sees, and configuration values:
       - `sds_coverage_threshold`
       - `sds_min_batch_size`
       - `sds_target_batch_size`
     - After batches are built, log:
       - Number of SDS batches.
       - Size and coverage fraction for each batch (`sds_debug_batch_coverage_summary` style).
   - When SDS cannot build any valid batch:
     - Emit `sds_error_no_valid_batches` with a clear reason (e.g. “no entries with valid WCS”, “coverage grid invalid”, etc.).
     - Immediately fallback to Mosaic-First, exactly as today.

5. **Final combination unchanged:**
   - Keep the current SDS pipeline:
     - For each batch: call `_assemble_global_mosaic_first_impl`.
     - Stack resulting batch mosaics with `zemosaic_align_stack` using coverage as weights.
     - Pass final mosaic, coverage and alpha to Phase 5/6.
   - Do not alter subsequent phases or non-SDS paths.


---

## 5. Localization (`en.json`, `fr.json`)

You may add **a minimal set** of new localization keys for SDS logs or labels **only if needed**.

Examples (if required):

- `sds_log_batch_policy`:  
  EN: `"SDS: coverage={coverage_threshold:.2f}, min_size={min_batch_size}, target_size={target_batch_size}."`  
  FR: `"SDS : coverage={coverage_threshold:.2f}, taille_min={min_batch_size}, taille_cible={target_batch_size}."`

- `sds_log_batch_summary`:  
  EN: `"SDS: built {batch_count} batch(es) with sizes: {sizes}."`  
  FR: `"SDS : {batch_count} lot(s) construit(s) avec les tailles : {sizes}."`

Make sure:
- They are added to both `en.json` and `fr.json`.
- The worker and/or GUI use these keys through the localization layer (no hard coded text in the UI).


---

## 6. Testing Scenarios

You must test (at least conceptually; via logs if no images are provided):

1. **Seestar mosaic with many frames where each frame covers almost the full field:**
   - SDS enabled.
   - Confirm that:
     - SDS creates **multiple batches** with `len(batch) >= sds_min_batch_size` in general.
     - You do **not** get “N inputs → N batches of 1” anymore.
     - Final SDS mosaic runs to completion without `sds_error_no_valid_batches`.

2. **Seestar with genuine multi-panel mosaic (bound-box arbitrary):**
   - SDS enabled.
   - Ensure:
     - Several SDS batches, each reasonably sized.
     - The final mosaic is computed in the global WCS and passed to Phase 5/6.
     - The Filter GUI preview shows the same batch counts (modulo preplan vs runtime differences) as the worker logs.

3. **Non-Seestar data or SDS disabled:**
   - Ensure behaviour is strictly unchanged:
     - No SDS batch building.
     - No SDS logs (except maybe a message stating SDS is disabled).

4. **Edge case: very few images:**
   - 1 or 2 frames:
     - SDS may legitimately create 1 batch of size 1 or 2.
     - No crash, no invalid coverage.

If any of these tests fail, you must refine the SDS batch policy but **without regressing** other workflows.


---

## 7. Non-Goals (Do NOT do this)

- Do not modify Tkinter GUIs (`zemosaic_gui.py`, `zemosaic_filter_gui.py`) for this mission.
- Do not change GPU vs CPU logic, CUDA helpers, or memory chunking logic.
- Do not expose new SDS options in the Qt main window UI unless necessary; batch sizes can remain config-only for now.
- Do not remove or rename existing config keys or CLI flags.
- Do not change how Phase 4.5 or Two-pass coverage renorm works.
