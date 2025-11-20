## `followup.md`

```markdown
# FOLLOW-UP — SDS vs Classic Pipeline Flow (Option A)

This follow-up provides a concrete step-by-step plan to implement the mission described in `agent.md`.

---

## 1. Identify the orchestration code in `zemosaic_worker.py`

1. Open `zemosaic_worker.py`.
2. Locate the main function that orchestrates all phases (commonly named something like `run_hierarchical_mosaic(...)` or similar).
3. Inside that function, find the block where:

   - `global_wcs_plan` is built (for Seestar mosaic),
   - and where the worker currently decides whether to call:
     - `assemble_global_mosaic_sds(...)`
     - and/or `assemble_global_mosaic_first(...)`
   - and assigns `final_mosaic_data_HWC`, `final_mosaic_coverage_HW`, `final_alpha_map`.

You will typically see something like:

```python
if global_wcs_plan and global_wcs_plan["enabled"]:
    mosaic_result = (None, None, None)

    if sds_mode_flag:
        mosaic_result = assemble_global_mosaic_sds(...)
    if mosaic_result[0] is None:
        mosaic_result = assemble_global_mosaic_first(...)

    final_mosaic_data_HWC, final_mosaic_coverage_HW, final_alpha_map = mosaic_result
````

followed later by:

```python
if final_mosaic_data_HWC is None:
    # Phase 3: classic master-tile pipeline
    ...
else:
    # Phase 5/6: postprocessing on final mosaic
    ...
```

The exact variable names may vary; adjust accordingly.

---

## 2. Enforce behaviour when SDS is OFF

Goal: when SDS is OFF → **classic master-tile pipeline MUST run**.

1. Ensure you have a boolean flag (e.g. `sds_mode_flag`) indicating whether SDS is active. It may come from:

   * configuration,
   * Filter GUI overrides,
   * Seestar detection.

2. In the `global_wcs_plan` block, change the logic so that:

   * If `not sds_mode_flag`:

     * **do NOT call** `assemble_global_mosaic_sds`.
     * **do NOT call** `assemble_global_mosaic_first`.
     * Explicitly set:

       ```python
       final_mosaic_data_HWC = None
       final_mosaic_coverage_HW = None
       final_alpha_map = None
       ```
     * Optionally log:

       ```python
       _log_and_callback(pcb_fn, "info", "sds_off_classic_mastertile_pipeline")
       ```
   * This guarantees that the later `if final_mosaic_data_HWC is None:` block will execute the classic master-tile Phase 3 pipeline.

3. Make sure that no other branch accidentally sets `final_mosaic_data_HWC` in the SDS-OFF case.

---

## 3. Enforce behaviour when SDS is ON

Goal: when SDS is ON → **SDS mega-tiles pipeline is primary**.

1. In the same orchestration block, handle the SDS ON case:

   ```python
   if sds_mode_flag and global_wcs_plan and global_wcs_plan["enabled"]:
       _log_and_callback(pcb_fn, "info", "sds_on_mega_tile_pipeline")
       mosaic_result = assemble_global_mosaic_sds(...)

       if mosaic_result[0] is None:
           _log_and_callback(pcb_fn, "warning", "sds_failed_fallback_mosaic_first")
           mosaic_result = assemble_global_mosaic_first(...)

       final_mosaic_data_HWC, final_mosaic_coverage_HW, final_alpha_map = mosaic_result
   ```

2. **Do not** call `assemble_global_mosaic_first` unless SDS fails.

3. After this block:

   * If `final_mosaic_data_HWC` is still `None`:

     * fall back to the classic Phase 3 pipeline:

       ```python
       _log_and_callback(pcb_fn, "warning", "sds_and_mosaic_first_failed_fallback_mastertiles")
       # ...run Phase 3 master-tile stacking
       ```
   * Else:

     * run the existing Phase 5/6 postprocessing on `final_mosaic_data_HWC` / coverage / alpha.

4. Do not modify the internals of `assemble_global_mosaic_sds` or `assemble_global_mosaic_first` beyond what’s necessary to align with this flow.

---

## 4. Preserve SDS batch policy and scalability

You must **not** touch:

* The SDS batch-building logic:

  * `sds_min_batch_size`, `sds_target_batch_size`,
  * coverage-based grouping.
* The chunking / streaming logic in SDS or Mosaic-First backends.
* Any memory-mapping options or GPU/CPU switching logic.

Sanity checks for scalability (conceptual):

* With 60 frames:

  * SDS ON → a handful of SDS batches (e.g. 10–15 mega-tiles), then one final stack.
* With 10 000 frames:

  * SDS ON → still builds batches using the same policy without blowing up memory.
  * The worker never tries to allocate full H×W×10 000 arrays.

If you see any code path that attempts to materialize all frames at once into memory, **do not introduce new ones**. Reuse the existing streaming/tiling approach.

---

## 5. Test matrix (conceptual)

### Test 1 — SDS OFF, Seestar data

* SDS checkbox off in Filter GUI Qt (and/or config).
* Run a Seestar dataset.
* Expectation:

  * Logs should **not** show SDS or Mosaic-First running.
  * `final_mosaic_data_HWC` should initially be `None`, triggering:

    * Phase 3 master-tile building,
    * then normal grid / final assembly.
  * Final image should be non-black and consistent with classic Tk workflow.

### Test 2 — SDS ON, normal case

* SDS checkbox ON, valid Seestar series with good WCS.
* Expectation:

  * Logs show `"sds_on_mega_tile_pipeline"`.
  * `assemble_global_mosaic_sds` builds batches and mega-tiles.
  * No fallback to classic Phase 3.
  * Final image non-black, produced via SDS.

### Test 3 — SDS ON, SDS failure

* Simulate an SDS failure (e.g. by forcing `assemble_global_mosaic_sds` to return `(None, None, None)` in a controlled test).
* Expectation:

  * Log `"sds_failed_fallback_mosaic_first"`.
  * Mosaic-First attempts to build final mosaic.
  * If Mosaic-First also fails, Phase 3 fallback is used with warning log.

### Test 4 — Non-Seestar data

* Ensure that:

  * Non-Seestar entries do NOT enter SDS or Mosaic-First Seestar-specific global WCS code.
  * Classic pipeline remains unchanged.

---

If any of these conceptual tests would fail with your changes, refine the flow control while keeping all constraints above.

