# Mission: Fix GRID coverage/mask broadcast bug in photometric inter-tile match

## High-level goal

The GRID pipeline currently falls back to the classic pipeline because of a `ValueError` raised during the inter-tile photometric matching step.

The root cause is a shape mismatch between:

- `coverage_mask`: 2D boolean array `(H, W)` produced from coverage maps, and
- `mask_ref`, `mask_tgt`: 3D boolean arrays `(H, W, C)` produced from RGB tile data.

Numpy cannot broadcast `(H, W)` with `(H, W, 3)`, so the expression

```python
common_mask = coverage_mask & mask_ref & mask_tgt
````

throws a `ValueError: operands could not be broadcast together with shapes (H,W) (H,W,3)` and the worker code catches this and triggers a fallback to the classic pipeline.

The mission is to **fix the construction of `common_mask` so that 2D coverage masks work correctly with 3D RGB masks**, and to do so in a way that is robust and consistent with the existing helper logic used elsewhere in `grid_mode.py`.

---

## Context

* Project: ZeMosaic
* Main file to modify: `grid_mode.py`
* Related but **not to be changed** in this mission: `zemosaic_worker.py`

You will find in `grid_mode.py`:

1. A helper function that already handles coverage vs combined masks:

   ```python
   def _combine_mask_with_coverage(combined_mask: np.ndarray,
                                   coverage_mask: np.ndarray | None,
                                   ...) -> np.ndarray:
       ...
       if cov_mask.ndim == 2 and combined_mask.ndim == 3:
           cov_mask = np.repeat(cov_mask[..., np.newaxis], combined_mask.shape[-1], axis=2)
       ...
       return combined_mask & cov_mask
   ```

   This shows the intended way to make a 2D coverage mask compatible with a 3D RGB mask.

2. Later, in the **grid photometric inter-tile matching loop**, there is a block similar to:

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
       common_mask = coverage_mask & mask_ref & mask_tgt   # <-- BUG HERE
   ```

   At this point:

   * `coverage_mask` is typically 2D `(H, W)`.
   * `mask_ref` and `mask_tgt` are likely 3D `(H, W, C)`.

This is the **only part to fix** for this mission.

---

## Requirements

### 1. Make coverage/mask combination RGB-safe

Replace the naïve line:

```python
common_mask = coverage_mask & mask_ref & mask_tgt
```

by logic that:

1. Converts `coverage_mask` to a boolean numpy array (`np.asarray(..., dtype=bool)`).

2. If `coverage_mask` is 2D and `mask_ref` is 3D (RGB), **broadcast/expand** the coverage mask to 3D:

   * You may either:

     * Use `np.repeat(cov_mask[..., np.newaxis], mask_ref.shape[-1], axis=2)`, or
     * Use `np.broadcast_to`, as long as the resulting shape is exactly `mask_ref.shape`.

3. Verifies that the final coverage mask shape matches `mask_ref.shape` (and implicitly `mask_tgt.shape`).

4. If shapes are compatible, computes:

   ```python
   common_mask = cov_mask & mask_ref & mask_tgt
   ```

5. If shapes are **not** compatible (for any reason), logs a WARN (using `_emit` with `lvl="WARN"`) and falls back to:

   ```python
   common_mask = mask_ref & mask_tgt
   ```

This behavior must be robust: **no exception is allowed** to propagate out of this section due to shape mismatch.

### 2. Use existing logging style

* Use the existing `_emit(..., lvl="WARN", callback=progress_callback)` helper if available in this scope.
* The message should clearly indicate that there was a coverage mask shape mismatch and that the code is falling back to a finite-pixel mask (`mask_ref & mask_tgt`).

Example style:

```python
_emit(
    "[GRID] Coverage mask shape mismatch in photometric match; "
    "falling back to finite-pixel mask.",
    lvl="WARN",
    callback=progress_callback,
)
```

### 3. Do not change other behavior

* Do **not** change how `coverage_mask` is computed (`_overlap_mask_from_coverage` stays as-is).
* Do **not** change any GPU / multithreading logic.
* Do **not** touch any other parts of `grid_mode.py` not directly related to `common_mask` in the photometric tile-matching section.
* Do **not** modify `zemosaic_worker.py` in this mission.

---

## Acceptance criteria

* The previous `ValueError: operands could not be broadcast together with shapes (1920,768) (1920,768,3)` no longer occurs.
* In runs where `cov_ref` and `cov_tgt` are both available, the GRID pipeline:

  * Proceeds through the photometric inter-tile matching phase without raising,
  * Does **not** trigger the “[GRID] Fallback to classic pipeline: reason=...” warning solely because of this mask mismatch.
* Logs:

  * The existing DEBUG message about coverage overlap is preserved.
  * In case of incompatible coverage mask shape, a single WARN is emitted and the code falls back to `mask_ref & mask_tgt`.

