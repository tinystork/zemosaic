# 🌌 Mission: Photometric harmonization & blending for Grid/Survey mode

You are Codex-Max.  
This is a **focused enhancement** of the existing **Grid/Survey mode** implemented in `grid_mode.py`.

The current Grid mode:
- Builds a global WCS.
- Splits the field into geometric tiles.
- Reprojects frames into tiles and stacks them.
- Assembles tiles into a global mosaic.

Result: geometrically correct, but **photometrically broken**:
- visible rectangular seams,
- different backgrounds per tile,
- no proper overlap blending,
- black/invalid areas visible.

Your mission is to:
1. Implement **SWarp-like background matching between tiles**.
2. Implement **multi-resolution (pyramidal) blending** in overlaps.
3. Implement **linear regression correction** in overlaps (tile-to-tile photometric calibration).
4. Implement **automatic masking of invalid/empty zones** (avoid using NaNs/zeros as signal).
5. Do this **ONLY for Grid/Survey mode**, without touching the classic pipeline.

---

## 0. Scope & constraints

- Scope: `grid_mode.py` (Grid/Survey code path).
- Optional: tiny adjustments in `zemosaic_worker.py` *only for logging or Grid-specific options*.
- DO NOT modify:
  - classic clustering / master-tile / Phase 5 pipeline,
  - non-Grid mosaic assembly,
  - GUI, except possibly to add a simple toggle for Grid photometry (optional).

- All logs related to this work must be tagged `[GRID]`.

---

## 1. Masking invalid / empty regions

Before doing any photometry or blending, we must define **valid pixels**.

Requirements:

- In tile processing and global assembly:
  - A pixel is **valid** if:
    - it is finite (`np.isfinite`),
    - AND not equal to a known "empty" value (e.g. 0, or a configurable threshold).
  - Otherwise it's invalid and must not contribute to the mosaic.

- Implement helper functions in `grid_mode.py`:

  ```python
  def compute_valid_mask(data: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
      """Return a boolean mask where True = valid pixel."""
````

* Use this mask to:

  * build `weight_crop` (0 for invalid pixels),
  * avoid counting invalid pixels in statistics,
  * avoid using invalid pixels in overlap regression.

* When `weight_sum == 0` in the global mosaic, mark those pixels as invalid (e.g. NaN) and **exclude them from any mean/variance computation** to avoid `nanmean` warnings.

---

## 2. Tile-level background & gain estimation (linear regression in overlaps)

Goal: each tile should be photometrically consistent with its neighbors.

### 2.1. Overlap graph

* After all tiles are created (and before final assembly), build an **overlap graph**:

  ```python
  class TilePhotometryInfo:
      tile_id: int
      bbox: Tuple[int, int, int, int]
      data: np.ndarray  # or a lightweight representation / stats
      mask: np.ndarray  # valid pixels

  def build_tile_overlap_graph(tiles: List[TilePhotometryInfo]) -> OverlapGraph:
      """Determine which tiles overlap in the global mosaic."""
  ```

* For each pair of tiles that overlap in the global mosaic, compute:

  * intersection region indices `(x0, x1, y0, y1)`,
  * valid pixels mask in both tiles.

### 2.2. Linear regression in overlaps

For each tile pair `(A, B)` with a non-empty overlap:

* Extract the overlapping valid pixels:

  ```python
  A_vals, B_vals  # 1D arrays of overlapping pixel values
  ```

* Fit a simple linear relation:

  ```python
  B ≈ a * A + b
  ```

  using robust linear regression (at minimum least-squares with optional clipping of outliers).

* Store for each tile a set of constraints of the form:

  * tile_j ≈ a_ij * tile_i + b_ij.

Then:

* Choose a **reference tile** (e.g. the one with most overlaps or a central tile).
* Solve for each tile a global `(gain, offset)` pair that best satisfies all pairwise relations.

  * You can do this iteratively (Gauss-Seidel style) or via a small least-squares system.
  * Aim for simplicity but robustness.

Apply these `(gain, offset)` to tiles before final blending:

```python
tile_data_corr = gain * tile_data + offset
```

If regression fails (too few pixels, all invalid, etc.):

* log a `[GRID]` warning,
* fall back to neutral `(gain=1, offset=0)` for that relation.

---

## 3. SWarp-like background matching

In addition to pairwise regression, we need a global large-scale background harmonization.

Requirements:

* Implement a function:

  ```python
  def estimate_tile_background(tile_data: np.ndarray, tile_mask: np.ndarray) -> float:
      """Return a robust estimate of the background level (e.g. sigma-clipped median)."""
  ```

* For each tile, estimate background `B_i`.

* Compute a **global target background**, for example:

  ```python
  B_target = median(B_i over all tiles)
  ```

* Adjust each tile:

  ```python
  tile_data -= (B_i - B_target)
  ```

* This step should be combined with or applied after the linear regression step.
  A possible order:

  1. Apply regression-based global gain/offset.
  2. Re-estimate backgrounds.
  3. Apply a final uniform background shift towards `B_target`.

* Ensure that all operations ignore invalid pixels (use the masks).

---

## 4. Multi-resolution (pyramidal) blending

Goal: avoid harsh seams by blending tiles across multiple spatial scales.

Implement a **multi-resolution blending scheme** for the tiles during final assembly.
You may use a Laplacian pyramid approach or a simpler Gaussian pyramid, as long as it respects these points:

### 4.1. Blending masks

* For each overlap region between tiles, generate a **smooth blending mask**:

  ```python
  w_A(x,y) + w_B(x,y) = 1
  ```

  with `w_A` decreasing smoothly from 1→0 across the overlap, and `w_B` = 1 - `w_A`.

* Masks must be zero on invalid pixels.

### 4.2. Pyramidal blending

* Implement helper functions:

  ```python
  def build_gaussian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
      ...

  def build_laplacian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
      ...

  def reconstruct_from_laplacian(pyramid: List[np.ndarray]) -> np.ndarray:
      ...
  ```

* For overlapping tiles A and B:

  1. Build Laplacian pyramids LA, LB.
  2. Build Gaussian pyramids of blending masks wA, wB.
  3. At each level `k`:

     ```python
     L_blend_k = LA_k * wA_k + LB_k * wB_k
     ```
  4. Reconstruct blended overlap region from `{L_blend_k}`.

* Apply this blending per overlap and integrate back into the global mosaic (respecting masks).

If this feels too heavy for all overlaps, you may:

* Restrict pyramidal blending to overlaps larger than a given size,
* Use a simpler Gaussian feathering for small overlaps.

---

## 5. Integration into `assemble_tiles`

The current `assemble_tiles` function performs:

* direct placement of tiles into `mosaic_sum`,
* additive weights,
* no advanced blending.

You must refactor `assemble_tiles` (or introduce a new function it calls) to:

1. Load all tiles into memory in a controlled way (or process in batches if necessary).
2. Build the **TilePhotometryInfo** objects with corrected data and masks.
3. Perform:

   * invalid zone masking,
   * overlap regression → global gain/offset calibration,
   * SWarp-like background adjustment,
   * multi-resolution blending for overlapping regions.
4. Fill the global mosaic via:

   * per-tile placement using the masks,
   * ensuring each pixel uses:

     * a weighted blend of all overlapping tiles,
     * with photometrically corrected data.

Make sure:

* All shapes `(H, W, C)` are consistent.
* BBox clamping & offsets (already implemented) remain correct and robust.
* No broadcasting errors are reintroduced.

---

## 6. Performance considerations

* The example datasets may have tens to hundreds of tiles; design algorithms that are:

  * O(N_tiles) or O(N_overlaps) in time,
  * memory-aware (avoid storing huge 4D arrays if unnecessary).
* Prefer iterative / local optimization over full dense matrices when possible.
* Add logs like:

  ```text
  [GRID] Photometry: built overlap graph with X edges
  [GRID] Photometry: solved global gain/offset for N tiles
  [GRID] Blending: applied pyramidal blending on M overlaps
  ```

---

## 7. Fallback behavior & safety

If any of these advanced steps fail (e.g. cannot build pyramids, regression unstable):

* Log a clear `[GRID]` warning.
* Fall back to a simpler behavior:

  * e.g. per-tile background normalization + linear blending without pyramids.
* Never crash the Grid mode or the entire worker.

Do NOT modify the classic (non-Grid) pipeline.

---

## 8. Deliverables summary

You must deliver:

* [ ] New helper functions in `grid_mode.py` for:

  * valid mask,
  * background estimation,
  * overlap regression,
  * pyramidal blending.
* [ ] A refactored `assemble_tiles` (or new equivalent) that:

  * uses masks,
  * applies global photometric calibration,
  * uses multi-resolution blending,
  * handles invalid data cleanly.
* [ ] `[GRID]` logs summarizing:

  * number of tiles,
  * number of overlaps,
  * photometry calibration steps,
  * blending operations.
* [ ] No regressions in classic pipeline.

