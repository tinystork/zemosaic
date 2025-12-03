# Agent Mission — Implement Overlapping Batch Construction in ZeMosaic

## 🎯 Objective
Modify the batching logic in **zemosaic_filter_gui_qt.py** (and any helper in zemosaic_utils if needed) so that **images may appear in multiple batches**, preventing coverage holes in the final mosaic.

The previous behaviour used a *strict partition* of images → if a batch had too many rejects, large areas became empty.

The new behaviour must use **overlapping sliding windows**:

- Allow the same FITS frame to appear in 2–3 consecutive batches.
- Ensure coverage even when some batches lose images to quality filters.
- Keep the final pipeline unchanged (reprojection, stacking, GUI, GPU/CPU workflow).
- Add a user option for overlap percentage (default 40%).

No modification must be made in ZeQualityMT or the stacking logic.

---

## ✔ Requirements

### 1. Add an “Overlap %” parameter to the GUI
- In **zemosaic_filter_gui_qt.py**, add a field in the batch configuration panel:
  - Label: “Overlap between batches (%)”
  - Default: **40%**
  - Range: 0–70%
- Must save/load correctly from config if applicable.

### 2. Replace the autosplit algorithm with overlapping sliding window batching  
Implement the following batching logic:

```python
def make_overlapping_batches(image_indices, cap, overlap_frac):
    # image_indices must be sorted by RA or projected X coordinate
    n = len(image_indices)
    if n <= cap:
        return [image_indices]

    step = max(1, int(cap * (1.0 - overlap_frac)))
    batches = []

    start = 0
    while start < n:
        end = min(n, start + cap)
        batch = image_indices[start:end]
        if len(batch) > 1:
            batches.append(batch)
        if end == n:
            break
        start += step

    return batches
````

* Integrate this function into the batching stage in **zemosaic_filter_gui_qt**.
* Ensure batch output format remains exactly the same as the previous version (so downstream phases work unchanged).

### 3. Maintain strict compatibility with the rest of the pipeline

* `stack_plan.csv`, `overrides_state.preplan_master_groups`, Phase 3, Phase 5, and Reproject must continue working as before.
* The only change must be *which images are assigned to batches*, not how they are processed.

### 4. Logging

* Add log lines indicating:

  * The number of batches generated.
  * For each batch: number of images + amount of overlap.
* Example:

```
[Batching] cap=100, overlap=0.40, effective step=60
[Batching] Created 23 batches, sizes: 100, 100, 100, ...
```

### 5. Testing (must be performed by Codex)

Codex must verify:

* [ ] Overlap parameter works in UI
* [ ] Overlapping batches are created correctly
* [ ] Pipeline produces *no coverage holes* even with high rejection rates
* [ ] Reproject integration unaffected
* [ ] master tiles produced identically when overlap=0
* [ ] No regressions in CPU/GPU fallback

---

## 🧪 Acceptance Criteria

The feature is accepted when these conditions are met:

* Running ZeMosaic on a sparse or noisy dataset **does not produce vertical or diagonal empty stripes** anymore.
* A frame can be reused multiple times without warnings or regressions.
* Final mosaic visually shows seamless coverage.
* Debug logs confirm overlapping batch construction.
* No other part of ZeMosaic is modified beyond what is listed.

---

## 🔒 Constraints

* DO NOT modify ZeQualityMT behaviour.
* DO NOT modify any stacking weight logic.
* DO NOT restructure Phase 3 or 5.
* DO NOT alter WCS solver.
* Keep everything backward-compatible with the previous config files.

---

## 📌 Deliverables

Codex must produce:

1. Full patch for `zemosaic_filter_gui_qt.py`
2. Added/updated function implementing overlapping batching
3. UI integration + config wiring
4. Updated log messages
5. Updated and validated batching tests if they exist

All changes must come as diff-ready code blocks.

