## `followup.md`

```markdown
# Follow-up: Steps to validate the coverage/RGB mask fix

## What was requested

You were asked to fix the construction of `common_mask` in the GRID photometric inter-tile matching code so that:

- 2D coverage masks `(H, W)` are correctly combined with 3D RGB masks `(H, W, C)`.
- No `ValueError` is raised due to broadcasting problems.
- The GRID pipeline no longer falls back to the classic pipeline because of this specific issue.

## What you should have changed

In `grid_mode.py`:

1. Locate the block in the tile photometric matching section where:

   ```python
   coverage_mask = None
   if cov_ref is not None and cov_tgt is not None:
       coverage_mask = _overlap_mask_from_coverage(cov_ref, cov_tgt)

   if coverage_mask is None or not np.any(coverage_mask):
       common_mask = mask_ref & mask_tgt
   else:
       _emit(
           f"[GRID] Coverage overlap for tile {info.tile_id} vs ref {reference_info.tile_id}: "
           f"pixels={int(np.sum(coverage_mask))}",
           lvl="DEBUG",
           callback=progress_callback,
       )
       common_mask = coverage_mask & mask_ref & mask_tgt
````

2. Replace the `common_mask = coverage_mask & mask_ref & mask_tgt` with a robust block that:

   * Converts `coverage_mask` to boolean,
   * Expands it to shape `(H, W, C)` if necessary (when `mask_ref` is 3D),
   * Verifies shape compatibility,
   * Either uses it (`cov_mask & mask_ref & mask_tgt`) or falls back to `mask_ref & mask_tgt` with a WARN log.

3. Ensure you use `_emit(..., lvl="WARN", callback=progress_callback)` for any fallback message related to coverage mask shape mismatch.

---

## Checklist for the human

When you review the changes, please verify:

* [ ] The only modified file for this mission is `grid_mode.py`.

* [x] The photometric matching section still logs the DEBUG message with coverage overlap (`pixels=...`).

* [x] The new code:

  * [x] Converts `coverage_mask` to a boolean numpy array.
  * [x] Handles the case where `coverage_mask` is 2D and `mask_ref` is 3D by expanding/broadcasting to match `mask_ref.shape`.
  * [x] Checks shape compatibility before applying `&`.
  * [x] Falls back to `mask_ref & mask_tgt` with a WARN log if shapes are incompatible.

* [x] No new exceptions are raised in this section when running GRID mode.

* [ ] In `zemosaic_worker.log`, after a GRID run using a `stack_plan.csv`:

  * [ ] You do **not** see a `[GRID] Fallback to classic pipeline: reason=... ValueError: operands could not be broadcast...` for this case.
  * [ ] GRID mode either completes fully, or any fallback is for another reason (which should be explicit in the logs).

(Optional but recommended):

* [ ] Run a small GRID dataset that previously triggered the fallback.
* [ ] Confirm that:

  * [ ] The pipeline stays in GRID mode (no forced fallback due to this error),
  * [ ] The result mosaic is produced without a broadcast error.

---

## Notes

* If you see a WARN log such as:

  > `[GRID] Coverage mask shape mismatch in photometric match; falling back to finite-pixel mask.`

  this is expected only in truly inconsistent shape situations, and the pipeline should still complete.

* If the fallback to the classic pipeline persists, inspect the worker log:

  * [ ] Confirm that the reason is **not** the broadcast error on `common_mask`.
  * [ ] If it is another error, that will need a separate mission.

