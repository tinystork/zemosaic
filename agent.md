# Mission

Instrument the **Grid mode** in ZeMosaic to understand why most pixels in the final tiles are zero, even though:
- the global canvas and grid geometry are correct,
- tiles receive the expected number of frames,
- all 9 tiles are written and loaded for assembly.

Goal: **add diagnostics only** (no functional changes) to see, for each tile and each frame:
- how much of the tile is actually covered by reprojected data,
- how much of the tile’s weight map is non-zero,
- whether this differs between the last-known-good Grid mode and the current one.

---

## Context

- Project: **ZeMosaic** (Grid/Survey mode).
- Files:
  - `grid_mode_last_know_geomtric_tiles_ok.py` → last known good Grid mode (tiles and coverage correct).
  - `grid_mode.py` → current Grid mode (tiles incomplete, lots of black area).
- Worker uses one of these grid_mode implementations from `zemosaic_worker.py` via `run_hierarchical_mosaic`.

Facts from logs:

- Global canvas is identical between good and faulty runs:
  - `global canvas shape_hw=(3272, 2406), offset=(-664, -669)`.
- Tile bboxes are identical (same 3×3 grid).
- Tile assignment is identical:
  - Tile 1: 53 frames, Tile 2: 51, Tile 3: 6, ..., Tile 9: 6.
- In the faulty run, we see for each tile:
  - `nan_frac=0.0` (no NaNs),
  - **huge** `zero_frac` (0.6–0.9), meaning most pixels in the stacked tile are exactly zero.
- Overlap graph reports many overlaps as “no finite pixels” even though we visually expect overlaps.

This strongly suggests the issue is **inside the tile stacking path**, specifically:
- `_reproject_frame_to_tile(...)` (how patches are positioned),
- how the footprint/weight map is applied and accumulated.

We want to **trace coverage** without changing the numerical behaviour.

---

## Constraints

- **Do not change functional logic**:
  - No changes to WCS math,
  - No changes to reproject parameters,
  - No changes to multithreading / chunking / GPU flags,
  - No changes to the actual stacking algorithm.
- Only add logging with a distinctive prefix, e.g. `"[GRIDCOV]"`.
- Logging must be reasonably compact:
  - Use `INFO` for per-tile summaries,
  - Use `DEBUG` for per-frame/per-patch details.
- Apply the same instrumentation to:
  - `grid_mode_last_know_geomtric_tiles_ok.py`,
  - `grid_mode.py`,
  so we can run the same dataset and diff the logs.

---

## Files to modify

- `grid_mode_last_know_geomtric_tiles_ok.py`
- `grid_mode.py`

Focus on:

- `process_tile(...)`
- `_reproject_frame_to_tile(...)` (or the equivalent helper used for per-frame reprojection)
- `assemble_tiles(...)` (optional small summary at the end)

---

## What to add (diagnostics)

### 1. In `_reproject_frame_to_tile(...)`

After computing the **reprojected patch** and **footprint/weight map**, add logs like:

- At DEBUG level:

```python
finite_frac = float(np.isfinite(patch).mean()) if patch.size else 0.0
nan_frac = float(np.isnan(patch).mean()) if patch.size else 0.0

if weight_map is not None:
    nonzero_weight_frac = float((weight_map > 0).mean()) if weight_map.size else 0.0
else:
    nonzero_weight_frac = -1.0  # sentinel

_emit(
    f"[GRIDCOV] tile_id={tile.tile_id} frame={frame.path.name} "
    f"patch_shape={patch.shape} "
    f"finite_frac={finite_frac:.3f} "
    f"nan_frac={nan_frac:.3f} "
    f"nonzero_weight_frac={nonzero_weight_frac:.3f}",
    lvl="DEBUG",
    callback=progress_callback,
)
````

If feasible, also log a *coarse* bounding box of non-zero weights in tile coordinates (no need for per-pixel dumps), e.g.:

```python
if weight_map is not None and weight_map.any():
    ys, xs = np.where(weight_map > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    _emit(
        f"[GRIDCOV] tile_id={tile.tile_id} frame={frame.path.name} "
        f"weight_bbox_local=(x0={x0}, x1={x1}, y0={y0}, y1={y1})",
        lvl="DEBUG",
        callback=progress_callback,
    )
```

> Important: do **not** change the returned arrays. Only compute stats and log them.

### 2. In `process_tile(...)`

After building `tile_shape` and before looping over frames:

```python
_emit(
    f"[GRIDCOV] tile_id={tile.tile_id} "
    f"bbox={tile.bbox} "
    f"tile_shape={tile_shape} "
    f"frames_in_tile={len(tile.frames)}",
    lvl="INFO",
    callback=progress_callback,
)
```

After all chunks are flushed and `stacked` is computed (just before saving):

* Compute coverage stats:

```python
if stacked.ndim == 3:
    stacked_gray = np.mean(stacked, axis=-1)
else:
    stacked_gray = stacked

finite_frac = float(np.isfinite(stacked_gray).mean()) if stacked_gray.size else 0.0
nan_frac = float(np.isnan(stacked_gray).mean()) if stacked_gray.size else 0.0
nonzero_frac = float((stacked_gray != 0).mean()) if stacked_gray.size else 0.0

_emit(
    f"[GRIDCOV] tile_id={tile.tile_id} "
    f"stacked_shape={stacked.shape} "
    f"finite_frac={finite_frac:.3f} "
    f"nan_frac={nan_frac:.3f} "
    f"nonzero_frac={nonzero_frac:.3f}",
    lvl="INFO",
    callback=progress_callback,
)
```

Optionally (if cheap), we can also log the min/max to see if we’re clipping everything to 0:

```python
if stacked_gray.size:
    vmin = float(np.nanmin(stacked_gray))
    vmax = float(np.nanmax(stacked_gray))
else:
    vmin = vmax = 0.0

_emit(
    f"[GRIDCOV] tile_id={tile.tile_id} "
    f"stacked_min={vmin:.3e} "
    f"stacked_max={vmax:.3e}",
    lvl="DEBUG",
    callback=progress_callback,
)
```

### 3. (Optional) In `assemble_tiles(...)`

After loading each tile FITS file:

* log coverage per tile (read back from disk) to check that what we wrote is what we read:

```python
img = loaded_tile_array  # H×W×C or H×W

if img.ndim == 3:
    gray = np.mean(img, axis=-1)
else:
    gray = img

finite_frac = float(np.isfinite(gray).mean()) if gray.size else 0.0
nonzero_frac = float((gray != 0).mean()) if gray.size else 0.0

_emit(
    f"[GRIDCOV] assemble: tile_id={tile_id} "
    f"shape={img.shape} finite_frac={finite_frac:.3f} "
    f"nonzero_frac={nonzero_frac:.3f}",
    lvl="INFO",
    callback=progress_callback,
)
```

This lets us verify that there is no bug between `save_fits_image(...)` and `load_fits_image(...)`.

---

## How to use the diagnostics (for the human)

After implementing the above in **both** Grid modes:

1. Run the **last-known-good** Grid mode on the same `stack_plan.csv`.
2. Run the **current** Grid mode on the same dataset.
3. Compare only lines starting with `[GRIDCOV]` between the two logs.

Look for:

* Tiles where:

  * `nonzero_frac` is high in the good version but very low in the faulty version.
* Frames where:

  * `nonzero_weight_frac` is high in good but ~0 in faulty,
  * or `weight_bbox_local` is much smaller / shifted in faulty.
* Any systematic pattern:

  * e.g. only some tiles affected,
  * only some frames (with particular WCS warnings) affected.

Once the divergence is located, we can then implement a **minimal fix** in `grid_mode.py` at the reprojection/weight-map level, while keeping threading/GPU intact.

