
# 🟦 **agent.md (updated)**

## 🌌 Mission: Stabilize Grid/Survey Mode Photometry, Background, and Color Balance

You are **GPT-5.1-Codex-Max**.
Your mission is to **improve the photometric consistency and color balance** of the **Grid/Survey mode** implemented in `grid_mode.py`, *without modifying the classic ZeMosaic pipeline*.

This update focuses on three urgent issues observed during real runs:

---

# ✔ **Objective 1 — Make all background & overlap statistics NaN-safe**

The current Grid mode background/overlap matching emits warnings such as:

* `Mean of empty slice`
* `All-NaN slice encountered`
* `Degrees of freedom <= 0 for slice`

These occur when computing:

* background medians,
* overlap regressions,
* per-channel estimates,

on arrays that sometimes contain **no finite pixel at all**.

### Requirements

1. **Before applying any statistic** (`np.nanmedian`, `np.nanmean`, `np.nanvar`, sigma-clipped stats, etc.):

   ```python
   finite = np.isfinite(array)
   if not np.any(finite):
       # Skip correction, return identity or leave tile unchanged.
   ```

2. Apply this systematically in:

   * background estimation per tile
   * global target background computation
   * overlap regression sampling
   * per-channel background matching

3. If an overlap is empty or invalid, skip it and log:

```
[GRID] Overlap skipped (no finite pixels)
```

4. All correction steps must function even if *some* tiles or channels contain no valid data.

**Goal:** no warnings, stable photometry, no propagation of NaN-based coefficients.

---

# ✔ **Objective 2 — Add a global RGB equalization step to Grid Mode**

Grid mode currently produces geometrically correct mosaics but with noticeable **color drift** (e.g., red/green imbalance).
Classic ZeMosaic uses `poststack_equalize_rgb` to equalize color channels, but Grid mode has no equivalent.

### Requirements

1. After `assemble_tiles()` and background normalization, add a new step:

```python
grid_post_equalize_rgb(mosaic, weight_sum)
```

2. This function must:

   * operate **only where weight_sum > 0**
   * compute **per-channel background medians** on a *background mask*
     (low-signal regions or sampling the lowest X% of pixel intensities)
   * compute gain factors so that **R, G, B reach the same median**

   Example:

   ```python
   target = np.mean([median_R, median_G, median_B])
   gain_R = target / median_R
   gain_G = target / median_G
   gain_B = target / median_B
   ```

3. Gains must be applied to all valid pixels using broadcasting.

4. This must produce the same “neutral, natural” color balance as the classic pipeline.

5. Expose a runtime flag in Grid mode only:

```python
grid_rgb_equalize = True  # default
```

6. Do **NOT** modify the classic pipeline’s use of `poststack_equalize_rgb`.

---

# ✔ **Objective 3 — Reduce WCS & Reproject warning noise**

Astropy occasionally emits:

* `'WCS.all_world2pix' failed to converge…`
* `Reproject encountered WCS convergence issues; retrying with distortion-stripped WCS.`

These are expected and handled, but spam the console.

### Requirements

1. Wrap calls to:

   * `_compute_frame_footprint`
   * `_reproject_frame_to_tile` → internally uses `reproject_interp`
   * any `WCS.world_to_pixel` applied repeatedly

in:

```python
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message=".*all_world2pix.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Reproject encountered WCS.*")
```

2. When such events occur, emit **one** log entry:

```
[GRID] WCS convergence degraded for frame XYZ (distortion-stripped WCS used)
```

3. Classic pipeline handling of WCS must remain **unchanged**.

---

# ✔ **Integration Constraints**

* Modify only Grid/Survey code paths (`grid_mode.py`).

* `zemosaic_worker.py` may be edited **only to call** the new RGB equalization stage or add logging.

* Do not modify:

  * clustering logic,
  * master tiles,
  * Phase 3 processing,
  * TwoPass renorm.

* All logs created must be tagged:

```
[GRID]
```

* No API breakage, no GUI changes required.

---

# ✔ **Deliverables**

You must deliver:

* [ ] A NaN-safe statistical layer for background matching
* [ ] A NaN-safe overlap regression system
* [ ] A new `grid_post_equalize_rgb()` function
* [ ] Integration of color correction into Grid mode
* [ ] Unified suppression/normalization of WCS warnings
* [ ] Clean logs with meaningful `[GRID]` diagnostics
* [ ] Zero impact on the classic ZeMosaic pipeline.

Failure to compute a correction must always fall back to identity (gain=1, offset=0).

