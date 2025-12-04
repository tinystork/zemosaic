# Mission: Split “Auto-organize Master Tiles” into Auto vs Manual and add smarter auto-optimization

## 0. TL;DR

Currently, in the **Qt filter dialog** (`zemosaic_filter_gui_qt.py`), there is a single button that performs a “master-tile organization” step (coverage-first clustering with some heuristics).

The user wants to:

1. **Keep the current behaviour**, but move it to a new button:
   - “Manual-organize Master Tiles” (or similar label).
2. Redefine **“Auto-organize Master Tiles”** as a **smarter optimizer** that:
   - tries to **maximize the number of raw frames per master tile** (up to the existing “Max raw frames per master tile” limit),
   - **minimizes the total number of master tiles**, and
   - keeps **good spatial overlap** between master tiles (no weird sparse tiles).

The Sky Preview will later be color-coded by group, so the user can visually confirm if the automatic orga is good.

---

## 1. Scope & Files

Primary file:

- `zemosaic_filter_gui_qt.py` :contentReference[oaicite:0]{index=0}  
  - Qt filter dialog (PySide6).
  - Contains:
    - WCS/Master tile control panel (cluster threshold, overlap, max raw frames per master tile, etc.).
    - Buttons related to **master tile organisation**.
    - Sky Preview & Coverage Map.

Support / algorithms (read-only unless really needed):

- `zemosaic_worker.py`  
  - Contains clustering helpers and Seestar-specific helpers used from the filter:  
    - `_CLUSTER_CONNECTED`, `_AUTOSPLIT_GROUPS`, `_COMPUTE_MAX_SEPARATION`, etc. :contentReference[oaicite:1]{index=1}  
- `zemosaic_filter_gui.py`  
  - Tk filter implementation; has legacy logic for master-tile grouping that can be reused as a reference if needed. :contentReference[oaicite:2]{index=2}  
- `zemosaic_config.py`  
  - Holds defaults like `min_safe_stack`, `target_stack`, and related stacking parameters that may inform the optimiser. :contentReference[oaicite:3]{index=3}  

Goal: **only modify the Qt filter UI and its grouping orchestration**; do not rewrite the worker/clustering core unless absolutely necessary.

---

## 2. Functional Requirements

### 2.1 Split into “Auto” and “Manual” buttons

In the **WCS / Master tile controls** area of the Qt filter dialog:

1. Introduce a new button:
   - Label (English): `Manual-organize Master Tiles`
   - Label (French): `Organisation manuelle des Master Tiles` (or similar).
   - Placement: next to the existing “Auto-organize Master Tiles” button; same style (QPushButton).
2. Rewire the **current behaviour** (whatever is executed today when clicking “Auto-organize Master Tiles”) to this **new Manual button**:
   - That includes the coverage-first clustering, the writing of `overrides_state.preplan_master_groups`, log messages (“Master tile organisation complete”, etc.), and refreshing of the preview.
3. Keep the manual behaviour **strictly identical** from the user’s point of view:
   - Same groups created,
   - Same logs,
   - Same way of populating the groups tree.

After this step, “Manual-organize Master Tiles” should behave exactly like the old auto-organize button.

### 2.2 New behaviour for “Auto-organize Master Tiles”

Redefine “Auto-organize Master Tiles” as an **optimizing strategy** built on top of the existing clustering logic:

- Use the same inputs as today:
  - cluster threshold (deg),
  - overlap fraction between batches,
  - Seestar heuristics toggle,
  - `Max raw frames per master tile`,
  - and potentially `min_safe_stack` / `target_stack` from config. :contentReference[oaicite:4]{index=4}  

- Objectives for the new auto mode:
  1. **Maximize raw frames per master tile**:
     - Try to get each master tile close to the configured “Max raw frames per master tile” spinner value, without exceeding it.
     - Avoid very small stacks when possible (use `min_safe_stack` and/or `target_stack` as soft constraints).
  2. **Minimize the total number of master tiles**:
     - Fewer, denser groups are preferred, as long as constraints are respected.
  3. **Maintain good spatial coherence and overlap**:
     - Tiles in the same master group should have reasonable angular proximity (RA/Dec) and not be scattered.
     - Reuse `_COMPUTE_MAX_SEPARATION` or existing coverage-first merging logic to ensure groups are geographically consistent. :contentReference[oaicite:5]{index=5}  

- Expected behaviour:
  - Start from the **existing coverage-first grouping / clustering result** (or equivalent).
  - Then apply a **post-processing step** that:
    - merges small groups with neighbours when it improves density and respects max frames per tile,
    - optionally splits overly large groups if they exceed the max frames per tile by too much,
    - prefers operations that reduce the total number of groups while keeping group sizes near the target range.

- End result:
  - A new set of master-tile groups (`overrides_state.preplan_master_groups` or equivalent) that:
    - covers all selected frames,
    - uses fewer groups,
    - has each group containing “as many frames as possible up to the max”.

The new auto mode must **not** silently drop frames; if a frame cannot be assigned to a nicely dense group, it still needs to be part of some group (possibly a smaller one).

---

## 3. Technical Guidelines & Hints

### 3.1 Factor current logic into reusable helpers

To avoid duplication:

1. Extract the **current master-tile organisation logic** into a helper method, e.g.:

   ```python
   def _run_manual_master_tile_organisation(self) -> None:
       ...
````

This should encapsulate whatever the old “Auto-organize” did:

* clustering call(s),
* setting `self.group_state.preplan_master_groups` or equivalent,
* updating the groups tree widget,
* logging,
* refreshing the sky preview.

2. Wire the new **Manual-organize** button to this helper.

3. For the new auto mode, either:

   * Call this helper to get an initial grouping, then apply optimisation, or
   * Factor out a lower-level helper that returns a list of groups, then use it from both auto and manual flows.

### 3.2 Implementing the optimisation step

You are free to choose the exact algorithm, but it should remain **simple and robust**. Suggested approach:

1. Get the **initial groups** (list of lists of frame descriptors).
2. Compute for each group:

   * number of frames,
   * approximate centre (mean RA/Dec),
   * maybe spread (using `_tk_circular_dispersion_deg` if available). 
3. While there exist “too small” groups:

   * Find the nearest neighbour group (in RA/Dec).
   * Check if merging both groups would yield:

     * total frames ≤ `max_raw_frames_per_master_tile`,
     * spatial separation / dispersion below a reasonable threshold.
   * If yes, merge them.
4. If some groups remain **too large**:

   * Optionally split by orientation or spatial clustering (you can reuse `_tk_split_group_by_orientation` or existing helpers already imported). 
5. Stop when:

   * No more beneficial merges/splits are possible, or
   * A maximum iteration count is reached (to avoid pathological cases).

Use `min_safe_stack` and `target_stack` from config as **soft hints**:

* prefer merges that bring a group closer to `target_stack` when below it,
* avoid creating groups smaller than `min_safe_stack` when alternatives exist. 

### 3.3 UX and logging

* The **Manual** button should:

  * log exactly what the previous auto mode logged, or as close as possible.
* The **Auto** button should:

  * log clearly that it is running an “optimised” grouping, e.g.:

    * `[INFO] Auto-organize Master Tiles: starting optimisation (max_frames=40, min_safe=3, target=5, overlap=10%)`
    * `[INFO] Auto-organize Master Tiles: prepared N groups (sizes: … )`
* At the end of both modes:

  * The groups tree,
  * The Sky Preview,
  * Any master-tile summary labels
    must all reflect the resulting grouping.

---

## 4. Tasks Checklist

Please execute and tick these items in your reasoning:

* [ ] Locate the current “Auto-organize Master Tiles” button and its slot/callback in `zemosaic_filter_gui_qt.py`.
* [ ] Extract the existing behaviour into a dedicated helper (e.g. `_run_manual_master_tile_organisation`).
* [ ] Add a new `Manual-organize Master Tiles` button wired to that helper.
* [ ] Rewire the original “Auto-organize Master Tiles” button to a new method that:

  * [ ] Builds or retrieves the initial group list.
  * [ ] Runs an optimisation loop that:

    * [ ] Respects `max_raw_frames_per_master_tile`.
    * [ ] Tries not to create groups smaller than `min_safe_stack`.
    * [ ] Minimises the number of groups.
    * [ ] Keeps groups spatially coherent using existing helpers where possible.
  * [ ] Writes the final groups back into the same structure the UI expects (e.g. `preplan_master_groups`).
  * [ ] Refreshes the sky preview and groups tree.
* [ ] Ensure no frames are dropped: every selected frame must belong to some master-tile group.
* [ ] Keep Tk UI behaviour unchanged (unless trivial refactor was needed, in which case verify it still works).
* [ ] Run the manual test plan (section 5) and adjust if needed.

---

## 5. Manual Test Plan

Use a test directory with many Seestar frames and WCS:

1. Open ZeMosaic Qt filter UI, load the directory, and go to **WCS / Master tile controls**.
2. Click **Manual-organize Master Tiles**:

   * Confirm:

     * It behaves exactly like the old “Auto-organize”.
     * The number of groups, group sizes and logs match the previous behaviour.
3. Reset/clear groups (if there is a way; otherwise reopen the filter).
4. Click **Auto-organize Master Tiles**:

   * Confirm:

     * It creates a **different grouping** from the manual mode on large sets, with:

       * fewer groups overall (when possible),
       * group sizes tending toward `Max raw frames per master tile`,
       * no pathological tiny groups unless unavoidable.
     * Sky Preview and groups list reflect the optimised grouping.
5. Try different `Max raw frames per master tile` values and re-run auto mode:

   * Confirm the grouping adapts (larger max → fewer, larger groups).
6. Sanity check:

   * No crashes or tracebacks in the console.
   * Coverage Map tab still works.
   * Performance is acceptable even with 1000+ frames.

If any of these steps fail, please refine the optimiser until all conditions are met.

