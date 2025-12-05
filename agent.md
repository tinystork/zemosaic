## 🟦 `agent.md`

````markdown
# 🛠 Mission: Stabilize Grid/Survey mode FITS loading and tile assembly

You are Codex-Max. This is a **small, focused bugfix mission** on the newly added `grid_mode.py` Grid/Survey pipeline.

The current behavior:
- Grid mode is correctly detected and started (stack_plan.csv present).
- FITS files are listed from stack_plan.csv.
- But:
  - FITS loading fails when BZERO/BSCALE/BLANK keywords are present (memmap error).
  - When this is fixed manually, `assemble_tiles` can crash with a **broadcast shape mismatch**:
    `ValueError: non-broadcastable output operand with shape (1,1,1) doesn't match the broadcast shape (3,3,1)`.
  - When Grid mode crashes, the worker falls back to the classic pipeline, but the user gets no mosaic from Grid.

Your mission is to:

1. **Fix FITS opening in Grid mode to be robust and consistent with the rest of ZeMosaic.**
2. **Fix `assemble_tiles` so tile placement into the global mosaic is safe and correctly clamped.**
3. **Ensure Grid mode either:**
   - runs to completion and produces a valid mosaic, **or**
   - fails cleanly and falls back to the classic pipeline without crashing.
4. **Do NOT modify the classic pipeline logic.**

---

## 0. Scope & constraints

- Scope: `grid_mode.py` (and at most minimal changes in `zemosaic_worker.py` for error handling / logging).
- Do NOT touch:
  - existing clustering / master-tile pipeline,
  - existing reproject & coadd logic,
  - GUI code.
- All Grid-specific logs must remain tagged with `[GRID]`.

---

## 1. FITS loading robustness (BZERO/BSCALE/BLANK)

Problem observed in logs:

- `Failed to open FITS ...: Cannot load a memory-mapped image: BZERO/BSCALE/BLANK header keywords present. Set memmap=False.`

This happens because the current code in `grid_mode.py` uses something like:

```python
with fits.open(frame.path, memmap=True) as hdul:
    header = hdul[0].header
    data = hdul[0].data
````

### Requirements:

* Change FITS loading in Grid mode to be **consistent with the rest of ZeMosaic**:

  * Use `memmap=False`.
  * Use `do_not_scale_image_data=True` to avoid surprises with BZERO/BSCALE.
* If there is already a utility function in the project (e.g. in `zemosaic_utils.py`) that wraps `astropy.io.fits.open` robustly, prefer **reusing that**.
* Grid mode must **no longer fail** on Seestar FITS or any FITS with BZERO/BSCALE/BLANK.

### Deliverables:

* A single, central helper in `grid_mode.py` (or use an existing shared utility) for opening FITS safely.
* All WCS/footprint computations and tile processing in Grid mode must use this helper instead of raw `fits.open(..., memmap=True)`.

---

## 2. Fix `assemble_tiles` broadcast / boundary bugs

Current crash:

```text
ValueError: non-broadcastable output operand with shape (1,1,1) doesn't match the broadcast shape (3,3,1)
```

This indicates a mismatch between:

* the slice on the global mosaic (`mosaic_sum[slice_y, slice_x, :]`), and
* the cropped tile data (`data_crop`).

Typical cause:

* `tile.bbox` may extend beyond the global mosaic dimensions,
* or the tile data is larger than the available mosaic region,
* but the code blindly assumes same H,W on both sides.

### Requirements:

In `assemble_tiles` (in `grid_mode.py`):

1. **Always clamp tile bounding boxes to the global mosaic size**:

   * Let `H_m, W_m, _ = mosaic_sum.shape`.
   * For each tile with bbox `(tx0, tx1, ty0, ty1)`:

     * clamp to mosaic bounds:

       ```python
       x0 = max(0, min(tx0, W_m))
       y0 = max(0, min(ty0, H_m))
       x1 = max(0, min(tx1, W_m))
       y1 = max(0, min(ty1, H_m))
       ```
     * If `x1 <= x0` or `y1 <= y0`: skip this tile.

2. **Account for possible negative or out-of-bounds bbox**:

   * If the tile bbox starts before 0, we need offsets into the tile data:

     ```python
     off_x = max(0, -tx0)
     off_y = max(0, -ty0)
     ```
   * Compute the final used height/width:

     ```python
     used_h = min(h_src - off_y, y1 - y0)
     used_w = min(w_src - off_x, x1 - x0)
     ```
   * If `used_h <= 0` or `used_w <= 0`: skip.

3. **Ensure dimensionality is consistent (H x W x C)**:

   * If a tile is 2D (H x W), convert to H x W x 1:

     ```python
     if data.ndim == 2:
         data = data[..., np.newaxis]
     elif data.ndim > 3:
         data = np.squeeze(data)
         if data.ndim == 2:
             data = data[..., np.newaxis]
     ```
   * `data_crop` and `mosaic_sum[slice_y, slice_x, :]` **must have identical H and W**.

4. **Only update mosaic with matching shapes**:

   * After cropping:

     ```python
     data_crop = data[off_y:off_y + used_h, off_x:off_x + used_w, :]
     weight_crop = np.ones_like(data_crop, dtype=np.float32)
     ```
   * Then:

     ```python
     mosaic_sum[slice_y, slice_x, :] += data_crop * weight_crop
     weight_sum[slice_y, slice_x, :] += weight_crop
     ```

5. Add optional debug logging when tiles are skipped or shapes look suspicious (tag `[GRID]`).

### Deliverables:

* A corrected `assemble_tiles` implementation that:

  * never raises a broadcasting error,
  * safely handles edge tiles,
  * places each tile in the correct region of the global mosaic.

---

## 3. Error handling & fallback behavior

When Grid mode fails (e.g., no valid frames, or assembly totally impossible), `run_grid_mode` should:

* Log a **clear `[GRID]` error** (already partially in place).
* Raise a controlled exception or return a clear failure indicator to `zemosaic_worker.run_hierarchical_mosaic`.
* `zemosaic_worker` should already catch this and log:

  * `[GRID] Grid/Survey mode failed, continuing with classic pipeline`.

### Requirements:

* Verify that:

  * If **no frames** could be loaded / have valid WCS, Grid mode aborts early and cleanly.
  * If `assemble_tiles` still encounters a fatal issue, it logs and fails cleanly.
* Do NOT change the classic pipeline logic or structure; only ensure that Grid mode failure does not crash the process.

---

## 4. Testing & validation

Implement basic internal tests / checks:

* Grid mode on the example dataset (the one with 53 frames in stack_plan.csv) must:

  * load all FITS without memmap errors,
  * build a grid,
  * assemble tiles,
  * produce a mosaic file in the configured output folder.
* If you need to add temporary extra logging to verify shapes (H,W,C) during development, keep it tagged `[GRID]` and keep it reasonably concise.

---

## 5. Deliverables summary

* [ ] Updated FITS loading logic in Grid mode (no more memmap/BZERO/BSCALE errors).
* [ ] Updated `assemble_tiles` with robust clipping and shape handling.
* [ ] Confirmed clean fallback behavior to classic pipeline when Grid mode cannot proceed.
* [ ] No changes to classic pipeline code paths.
* [ ] Clear `[GRID]` logs for:

  * frames loaded,
  * tiles processed,
  * any skipped tiles / errors.



