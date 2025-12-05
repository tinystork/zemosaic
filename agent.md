# Agent: Grid Mode – Assembly Robustness & RGB Consistency

## 🎯 Goal

1. Make `grid_mode` failures **easier to diagnose** by enriching `[GRID]` logs in `assemble_tiles()`.
2. Relax the logic so that **isolated bad tiles don’t cause a full grid-mode abort**.
3. Ensure that **RGB normalization behaviour is strictly coherent** between Grid mode and the classic pipeline (same toggles, same semantics).

---

## 🔍 Context

Main files involved:

- `grid_mode.py`
  - `run_grid_mode(...)`
  - `assemble_tiles(...)`
  - Logging helpers: `_emit(...)`, `build_tile_overlap_graph(...)`, `grid_post_equalize_rgb(...)`, etc.
- `zemosaic_worker.py`
  - `run_hierarchical_mosaic(...)`
  - Detection and invocation of Grid/Survey mode:
    - `detect_grid_mode(input_folder)`
    - Call to `grid_mode.run_grid_mode(...)`
  - Fallback to classic pipeline when Grid mode raises.

Right now:

- `assemble_tiles(...)` returns `None` in a few cases:
  - Astropy/FITS not available.
  - No tiles found on disk.
  - All tile reads fail or are channel-mismatched.
  - After accumulation, `np.any(weight_sum > 0)` is `False` (no valid data written).
- `run_grid_mode(...)` interprets `None` as **fatal** and raises:
  - `RuntimeError("Grid mode failed during assembly")`
- RGB equalization in Grid mode:
  - Uses `grid_rgb_equalize` (param + config key).
  - Applies `grid_post_equalize_rgb(...)` on the assembled mosaic when `grid_rgb_equalize=True`.
- Classic pipeline:
  - Uses *post-stack* RGB equalization for master tiles (e.g. `poststack_equalize_rgb`).
  - Final mosaic assembly (reproject_coadd) uses a `RGBEqualize=True/False` toggle.

We want these behaviours to be **traceable, robust, and aligned**.

---

## 🧩 Tasks

### 1. Enrich logging in `assemble_tiles(...)`

**File:** `grid_mode.py`  
**Function:** `assemble_tiles(...)`

1.1. Add richer logs at each early-return condition:

- When Astropy/FITS is unavailable:
  - Log clearly that Grid mode will **fallback** because Astropy is missing.
  - Include `_ASTROPY_AVAILABLE` and `fits` flags in the message.

- When `tiles_list` is empty:
  - Current: `_emit("No tiles to assemble", lvl="ERROR", ...)`
  - Upgrade to something like:
    - `"No tiles to assemble (len(tiles)=N, len(tiles_list)=0). Check master-tile output paths & file existence."`
  - Include:
    - `len(list(tiles))` if cheap / precomputed.
    - A few sample `tile.output_path` values if available.

- When `tile_infos` is empty:
  - Current: `"Unable to read any tile for assembly"`
  - Upgrade to include:
    - Number of tiles attempted.
    - Counts of:
      - I/O failures.
      - Channel mismatches.
      - Empty masks.
    - Possibly log a **summary line**:
      - `"Assembly summary: attempted=T, io_fail=..., channel_mismatch=..., empty_mask=..., kept=0"`

- When `weight_sum` is zero everywhere:
  - Current: `"Assembly: no valid tile data written to mosaic"`
  - Upgrade to include:
    - `len(tile_infos)`
    - Possibly min/max of each tile’s bbox vs `grid.global_shape_hw`.
    - A short hint:
      - `"This usually means that all tiles were fully masked out (coverage map, masks, or overlap logic)."`

1.2. Track per-tile failure reasons:

- Before the main loop over `tiles_list`, declare counters, e.g.:
  - `io_failures = 0`
  - `channel_mismatches = 0`
  - `empty_masks = 0`
- When a read fails:
  - Increment `io_failures`.
- When a channel mismatch occurs:
  - Increment `channel_mismatches`.
- When `compute_valid_mask(data)` returns a mask with no `True`:
  - Don’t silently accept it; log:
    - `"Assembly: tile {t.tile_id} has empty valid-mask, skipping"`
  - Increment `empty_masks`.
- After building `tile_infos`:
  - Emit a summary log at INFO/DEBUG:
    - `"Assembly: {len(tile_infos)} tiles kept (io_fail={io_failures}, channel_mismatch={channel_mismatches}, empty_mask={empty_masks})"`

1.3. Ensure all new logs use `_emit(...)` with the existing `progress_callback` pattern and `[GRID]`-style prefixes where appropriate, so they show up in the dedicated grid logs.

---

### 2. Relax logic: try to salvage mosaic instead of aborting too early

**Goal:** If some tiles are problematic, **skip them**, but still produce a mosaic as long as at least some valid data can be placed.

The current code already:

- Skips tiles with I/O errors.
- Skips tiles with channel mismatch.
- Skips tiles with empty mask (we’ll make this explicit).
- Aborts only when:
  - `tiles_list` is empty.
  - `tile_infos` is empty.
  - `weight_sum` has no positive entries.

We want to **add a “salvage path”** before giving up on the `weight_sum` condition.

2.1. Extend the `weight_sum` check

In `assemble_tiles(...)`, around:

```python
if not np.any(weight_sum > 0):
    _emit("Assembly: no valid tile data written to mosaic", lvl="ERROR", callback=progress_callback)
    return None
````

Replace this with a two-step approach:

1. First, log a detailed diagnostic (using the counters from Task 1).
2. Attempt a **simple salvage assembly** when `tile_infos` is non-empty but `weight_sum` is zero:

Example logic (high-level; Codex must implement properly):

```python
if not np.any(weight_sum > 0):
    _emit(
        "Assembly: no valid tile data written to mosaic (weight_sum all zeros). "
        "Attempting salvage assembly with relaxed masking.",
        lvl="WARN",
        callback=progress_callback,
    )

    salvage_used = False
    # Example: simple paste of each tile’s data into the mosaic without fancy overlap logic
    mosaic_sum.fill(0.0)
    weight_sum.fill(0.0)

    for info in tile_infos:
        # use bbox directly, clamp to mosaic bounds
        (y0, y1), (x0, x1) = info.bbox
        # clamp y0, y1, x0, x1 within [0, H_m] / [0, W_m]
        # skip if invalid
        # place data directly with weight=1.0
        # ...

    if np.any(weight_sum > 0):
        salvage_used = True
        _emit("Assembly: salvage assembly succeeded (mosaic partially recovered).", lvl="WARN", callback=progress_callback)
    else:
        _emit("Assembly: salvage assembly failed, still no valid data.", lvl="ERROR", callback=progress_callback)

    if not salvage_used:
        return None
```

Constraints:

* Do **not** change the existing overlap graph / regression logic; the salvage path is only used as a last resort when the photometric/overlap machinery yields an empty mosaic.
* Keep the salvage implementation **simple**: no complicated weighting, just enough to avoid a full abort when tiles clearly have valid data.
* Keep consistent shapes:

  * Use the same `mosaic_shape` and `weight_sum` as the nominal path.

2.2. Clarify return semantics

* If both the nominal path and the salvage path fail to write any data (`weight_sum` still zero):

  * Keep the current behaviour: log ERROR and `return None`.
* If salvage succeeds:

  * Continue to compute `mosaic = mosaic_sum / weight_sum` as usual.
  * Proceed with RGB equalization (if enabled) and FITS writing as in the nominal path.
  * Add a `"salvage_used=True"` hint in one of the logs (e.g. in the final success message).

---

### 3. Ensure RGB normalization behaviour is consistent between Grid mode and classic pipeline

**Goal:** A user-facing “RGB equalization / color normalization” toggle should behave **the same way** whether:

* Grid mode is active (`stack_plan.csv` present ⇒ `run_grid_mode(...)` used), or
* Classic hierarchical pipeline is used (no grid).

We want:

* One **logical toggle** (from GUI/config) to control:

  * Post-stack RGB equalization of master tiles.
  * Final mosaic-level RGB equalization in classic pipeline.
  * Final mosaic-level RGB equalization in Grid mode.

3.1. Audit current RGB equalization flags

* In `zemosaic_worker.py`, locate:

  * The flags controlling post-stack RGB equalization (e.g. `poststack_equalize_rgb`, `RGBEqualize` in phase 5).
  * The values passed to the final coadd (e.g. `RGBEqualize=True/False`).
* In `grid_mode.py`, confirm:

  * `run_grid_mode(...)` has a `grid_rgb_equalize` parameter (default `True`).
  * The config is loaded via `_load_config_from_disk()` and `cfg_disk.get("grid_rgb_equalize", grid_rgb_equalize)`.
  * `assemble_tiles(...)` receives `grid_rgb_equalize` and uses it to call `grid_post_equalize_rgb(...)`.

Document in comments (briefly) how these paths are currently wired.

3.2. Define a single source of truth for “RGB equalization enabled?”

* Choose a single configuration field (or combination) that represents:

  * “Final mosaic should be RGB-equalized”.
* Ensure that:

  1. Classic pipeline:

     * Uses this same boolean for:

       * the final `RGBEqualize=<bool>` in the reproject_coadd / Phase 5.
       * (optionally documented) the per-master-tile poststack equalizer, if that is intended behaviour.

  2. Grid mode:

     * `run_grid_mode(...)` receives a `grid_rgb_equalize` argument that is derived from the **same** boolean.
     * The config loader (`cfg_disk.get("grid_rgb_equalize", ...)`) should **not silently drift** away from the main setting:

       * If both keys exist, define a clear priority (e.g. GUI > disk).
       * Log the final decision, e.g.:

         * `"Grid RGB equalization: enabled=True (source=GUI/config)"` or similar.

Implementation suggestions (adapt to existing configuration style):

* In `zemosaic_worker.run_hierarchical_mosaic(...)`, after loading `worker_config_cache` and building `zconfig`, centralize something like:

  ```python
  rgb_eq_flag = determine_rgb_equalization_flag(zconfig, ...)
  ```

* Use `rgb_eq_flag` to:

  * Configure the classic pipeline final-phase `RGBEqualize`.
  * Pass it explicitly to `grid_mode.run_grid_mode(..., grid_rgb_equalize=rgb_eq_flag, ...)`.

* In `grid_mode.run_grid_mode(...)`, adapt the `_load_config_from_disk()` overlay logic so that:

  * If a `grid_rgb_equalize` key is present on disk, it **overrides** the default parameter *only if* that is the intended UX.
  * Always log the final effective state:

    * `"Grid mode RGB equalization: enabled={grid_rgb_equalize} (from disk/param)"`.

3.3. Add minimal sanity logs

* In `zemosaic_worker.run_hierarchical_mosaic(...)`, before invoking Grid mode:

  * Log at INFO/DEBUG:

    ```text
    [GRID] Invoking grid_mode with RGBEqualize={rgb_eq_flag}
    ```

* In `grid_mode.assemble_tiles(...)`, when applying `grid_post_equalize_rgb(...)`:

  * Keep / specialize the existing log:

    * `"RGB equalization: calling grid_post_equalize_rgb (shape=..., enabled={grid_rgb_equalize})"`

---

## ✅ Deliverables

* Updated `grid_mode.py`:

  * Richer logs in `assemble_tiles(...)`.
  * Salvage path implemented for the “no valid tile data” case.
  * Clearer diagnostics for tiles skipped (I/O, channel mismatch, empty mask).
  * Consistent and informative RGB equalization log messages.

* Updated `zemosaic_worker.py`:

  * Single, explicit boolean controlling final RGB equalization across:

    * Classic pipeline final assembly.
    * Grid mode (`grid_mode.run_grid_mode(..., grid_rgb_equalize=...)`).
  * Logging of the chosen RGB equalization state when Grid mode is invoked.

* Brief comments in code explaining:

  * What triggers the salvage path.
  * How the RGB equalization boolean is derived and applied in both pipelines.

---

## 🧪 Tests (manual is OK)

1. **Happy path, Grid mode OK**

   * Use a `stack_plan.csv` dataset where all tiles are healthy.
   * Run with Grid mode enabled and RGB equalization ON.
   * Verify:

     * Mosaic is produced.
     * Logs show:

       * `Photometry: loaded N tiles for assembly`
       * `Assembly: ...` summary with non-zero `kept`.
       * `RGB equalization: calling grid_post_equalize_rgb (shape=...)`
     * Compare colors with classic pipeline (same inputs, same settings) – they should be very close.

2. **Partial failure, some tiles broken**

   * Corrupt a few master-tile FITS files (or alter channels) so that some tiles are skipped.
   * Run Grid mode.
   * Verify:

     * Logs clearly indicate I/O or channel mismatch counts.
     * Mosaic is still produced (salvage may or may not be needed).
     * No `RuntimeError("Grid mode failed during assembly")`.

3. **Edge case: masks / overlaps cause empty mosaic**

   * Force a situation where all valid masks are trimmed out (e.g. via synthetic grid or masks).
   * Run Grid mode.
   * Verify:

     * Log shows `"Assembly: no valid tile data written to mosaic"` + `"Attempting salvage assembly..."`.
     * Either:

       * Salvage succeeds and mosaic is produced (with warning).
       * Or salvage fails, and `RuntimeError` is raised with detailed diagnostics.

4. **RGB toggle coherence**

   * With a simple small dataset, run:

     1. Classic pipeline with RGB equalization ON.
     2. Classic pipeline with RGB equalization OFF.
     3. Grid mode with RGB equalization ON.
     4. Grid mode with RGB equalization OFF.
   * Check:

     * The bool used in logs is consistent in all cases.
     * Visual differences between ON/OFF in Grid mode match the ON/OFF differences in classic mode.

Please keep code style, typing, and logging conventions consistent with the existing project.
