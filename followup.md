# Follow-up: Classic mode still green after disabling final RGB equalisation

## Current status

- The user had a classic mode that worked correctly using `zemosaic_worker_non grid_ok.py`.
- The new unified worker `zemosaic_worker.py` (with Grid/SDS integration) produces a **strongly green-tinted mosaic** in classic mode.
- We already **commented out** the block that calls `_apply_final_mosaic_rgb_equalization(...)` (around lines 6778–6796), so the final mosaic-level RGB eq is currently **disabled**.
- Despite this, the resulting image is still green, which means the bug is not just that final RGB eq.

The user’s goal now is **simple**:  
> “Just restore a production ready program for classic mode, even if we don’t fully understand all the internals.”

The reference implementation for “good” behaviour is the older file:

- `zemosaic_worker_non grid_ok.py`

---

## Observations / suspected root cause

By comparing the two workers around Phase 5:

- The implementation of `_apply_phase5_post_stack_pipeline` itself is effectively the same.
- The main difference is **how it is called** on the **final mosaic** in the classic / non-grid path.

In the **reference** `zemosaic_worker_non grid_ok.py`:

- The Phase 5 call on the final mosaic explicitly sets:

  ```python
  enable_lecropper_pipeline=False
  enable_master_tile_crop=False
````

In the **current** `zemosaic_worker.py`:

* Before calling `_apply_phase5_post_stack_pipeline` on the final mosaic, we derive:

  ```python
  enable_final_lecropper = [...]
  enable_final_master_crop = [...]
  ```

  based on `final_quality_pipeline_cfg`, and then pass these as:

  ```python
  enable_lecropper_pipeline=enable_final_lecropper
  enable_master_tile_crop=enable_final_master_crop
  ```

Consequences:

* The **quality pipeline** (lecropper, alt-az cleanup, master tile crop, etc.) is now potentially **reapplied on the final mosaic**, on top of what was already applied at master-tile level.
* `_derive_final_alpha_mask` can also trigger an additional `_apply_phase5_post_stack_pipeline` call, which means the mosaic may get processed **twice or more** by the same pipeline.

Given that these stages include **background cleanup and multi-channel operations**, repeatedly applying them on an already-processed mosaic can easily distort the colour balance and background, explaining the persistent **green cast** even after disabling final RGB equalisation.

---

## What was already done

* Final RGB equalisation on the mosaic has been disabled by commenting out the call to:

  ```python
  _apply_final_mosaic_rgb_equalization(...)
  ```

* This must remain disabled for now; the mission is to fix classic mode by aligning Phase 5 behaviour with the reference worker.

---

## What remains to be done (for this mission)

- [x] In `zemosaic_worker.py`, restrict the **Phase 5 post-stack pipeline call** on the final mosaic (classic / non-grid path) to use the **same effective options** as the reference worker:

  * `enable_lecropper_pipeline=False`
  * `enable_master_tile_crop=False`

  while keeping the rest of the arguments intact (two-pass, WCS, coverage, etc.).

- [x] Ensure that any new variables like `enable_final_lecropper` / `enable_final_master_crop` are **not used** for that classic call site.
  They can remain in the file if they are used for SDS / Grid Mode or other paths.

- [ ] Confirm that:

  * Classic mode output recovers a sane colour balance (no green cast).
  * Grid Mode and SDS outputs remain unchanged and functional.

---

## Notes and guardrails

* **Do not** re-enable `_apply_final_mosaic_rgb_equalization` in this task.
* **Do not** modify `grid_mode.py` or any Grid/SDS-specific sections.
* The reference worker `zemosaic_worker_non grid_ok.py` is the **source of truth** for how the Phase 5 call should look in classic / non-grid mode.
* It is acceptable that the internal reasoning for why the green cast happens remains partially opaque; the priority is to restore a **stable, production-grade classic mode** that behaves like the proven reference.

