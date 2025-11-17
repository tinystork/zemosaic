

# FOLLOW-UP TASK LIST — QT FILTER + MAIN UI POLISH

This file breaks down the mission for Codex into **small, explicit tasks**.  
All tasks are limited to the **Qt GUI** unless otherwise stated.

---

## 1. Sky Preview: simplify drawing to avoid freezes

**Files:**

- `zemosaic_filter_gui_qt.py`
- (optionally) `locales/en.json`, `locales/fr.json` for text hints

### 1.1. Inspect current sky preview code

- Locate `_update_preview_plot` in `zemosaic_filter_gui_qt.py`.
- Identify:
  - How `points` are collected (`_collect_preview_points`).
  - How `grouped_points` and scatter plots are built.
  - How `footprints_for_preview` is built:
    - Loop over `self._normalized_items`
    - Calls like `entry.ensure_footprint()` or similar
    - Adds polygons via `axes.plot(...)`.
  - How `_group_outline_bounds` and `_group_outline_collection` are used.

Goal of this step: clearly separate three responsibilities:
1. Points per frame / group (scatter).
2. Group outlines (rectangles).
3. Per-frame footprints (full polygons) — this is what we want to **disable** for performance.

### 1.2. Disable per-frame footprints in the Qt sky preview

- Adjust `_update_preview_plot` so that it **never builds the full list of per-frame footprints** for preview, by default.

Concretely:

- Either:
  - Make `_should_draw_footprints()` always return `False` in the Qt Filter GUI, **or**
  - Remove the whole `footprints_for_preview` loop and related plotting from the Qt implementation.
- Keep the code paths for:
  - Scatter points per group (`axes.scatter`).
  - Group coverage outlines via `_group_outline_bounds` and `LineCollection`.

Important:

- Do **not** remove per-frame footprint logic used elsewhere (e.g. coverage map or non-Qt parts). Only touch the Qt sky preview drawing part.
- Ensure that removing footprints does **not** break any log message or summary text; if some messages refer to “footprints”, update them to mention “preview points” or “group coverage”.

### 1.3. Keep / adapt the “Draw WCS footprints” checkbox

- The checkbox is defined via the localization key `"filter_chk_draw_footprints"` (see `locales/en.json`).

Adjust behavior:

- Reinterpret the checkbox as “Draw group outlines” instead of “draw every footprint”.
  - When checked: draw group rectangles (`_group_outline_bounds` → `LineCollection`) + points.
  - When unchecked: draw only points (no group rectangles).
- Optionally, **update the label text** in `en.json` / `fr.json`:
  - Example EN: `"Draw group WCS outlines"`  
  - Example FR: `"Afficher les contours WCS des groupes"`
- Do **not** change the JSON key name; only the human-readable value.

### 1.4. Update preview hint text (optional)

- The existing hint string is `"filter_preview_points_hint"` in `locales/en.json`.
- If the message still mentions “footprints”, adjust it to match the new behavior, e.g.:
  - EN: “Preview uses group centers and outlines for performance. Zoom or filter to inspect coverage.”
  - FR: “L’aperçu utilise les centres et contours des groupes pour de meilleures performances. Zoomez ou filtrez pour examiner la couverture.”

### 1.5. Bounds & legend after removing footprints

- Since we no longer extend `ra_values` / `dec_values` with footprint polygons, make sure the axes limits are still correctly computed:
  - Use only `points` (group centers) and group outline corners to compute min/max RA/Dec.
  - If needed, extend `ra_values` and `dec_values` with the four corners of each group outline (`_group_outline_bounds`).
- Ensure the legend and colors still behave like before:
  - “Group 1…N” labels.
  - “Unassigned” when applicable.
  - Legend only if more than one group is present.

**Acceptance tests**

- Open the Qt Filter window on a dataset with many frames (e.g. >3000).
- Trigger analysis so that sky preview is populated.
- Confirm:
  - The GUI remains responsive; no long freeze while redrawing.
  - Group colors and legend match the clustering.
  - Group outlines give a clear idea of coverage.
  - Toggling “Draw WCS footprints” on/off only toggles the outlines, not the core behavior.

---

## 2. Persist Qt Filter window geometry

**Files:**

- `zemosaic_filter_gui_qt.py`
- `zemosaic_config.py` (for `_load_gui_config` / `_save_gui_config`)

### 2.1. Verify helpers and constants

- Confirm the presence of:

  ```python
  QT_FILTER_WINDOW_GEOMETRY_KEY = "qt_filter_window_geometry"
````

and helper methods:

* `_load_saved_window_geometry`
* `_capture_current_window_geometry`
* `_apply_saved_window_geometry`
* `_persist_window_geometry`
* `_normalize_geometry_value`
* Confirm that `_apply_saved_window_geometry()` is called at the end of the Filter dialog `__init__`.

### 2.2. Ensure persist is called on close

* In the `closeEvent` override of the Qt Filter dialog, ensure:

  ```python
  def closeEvent(self, event) -> None:
      self._persist_window_geometry()
      # existing streaming / worker shutdown logic…
      super().closeEvent(event)
  ```

* If `closeEvent` already calls `_persist_window_geometry()`, keep it but double-check that no early return skips it.

### 2.3. Robustness of `_load_saved_window_geometry`

* Confirm that `_load_saved_window_geometry`:

  * Calls `_load_gui_config()` safely, catching exceptions.
  * Extracts the key `QT_FILTER_WINDOW_GEOMETRY_KEY`.
  * Normalizes geometry via `_normalize_geometry_value`.
  * Returns `None` on invalid or missing data.

* Ensure that `_apply_saved_window_geometry` is robust:

  * If `geometry` is `None`, simply return without raising.
  * Wrapped `setGeometry(...)` in a `try/except` to avoid hard crashes (already present; keep it).

### 2.4. Symmetry with `_capture_current_window_geometry`

* Verify `_capture_current_window_geometry` uses:

  ```python
  rect = self.normalGeometry() if self.isMaximized() else self.geometry()
  ```

* Ensure `_normalize_geometry_value` clamps width/height to at least 1 and rejects non-positive values.

### 2.5. Test behavior

Manual test:

1. Launch `run_zemosaic.py` in Qt mode, open the Filter window from the main Qt GUI.
2. Move/resize the Filter window.
3. Close the Filter window, then reopen it via the main window.
4. Confirm:

   * The Filter window position and size are restored.
   * No crash or weird behavior if the config file did not previously contain any geometry.

---

## 3. Fix ghost / duplicate text in Main Qt window

**File:**

* `zemosaic_gui_qt.py`

### 3.1. Inspect Main tab right column layout

* Locate the method that builds the Main tab (e.g. something like `_create_main_tab` or similar).
* Identify the right column container and the groups created inside it:

  * Folders
  * Instrument / Seestar
  * Plate solving
  * Stacking options (Master tiles)
  * Quality / Alt-Az
  * Final assembly & output
  * System resources & cache
* Pay special attention to the function creating the “Final assembly & output” group (a `QGroupBox` with a title built via `self._tr("qt_group_final_assembly", ...)` or similar).

### 3.2. Find duplicate labels

* Search for any additional `QLabel` created with text similar to:

  * “Final assembly & output”
  * or the localized equivalent.

* These may be added to a layout as a row label without a corresponding input widget, or as a leftover debug label.

* Remove or repurpose any such duplicate label, keeping only the **group box title** as the visible header for that section.

### 3.3. Check layout nesting

* Verify the hierarchy of widgets:

  * Right column main layout (QVBoxLayout or QGridLayout).
  * Each section wrapped in its own `QGroupBox`.
  * Inside `Final assembly & output`, there should be:

    * A vertical box (`QVBoxLayout`) or form layout.
    * Controls for assembly method, boolean options, etc.
* Ensure that no stray layout adds a label directly to the right column layout with the same title as the group.

### 3.4. Test without ghost text

* Run the Qt main GUI.
* Go to the **Main** tab, right column.
* Check visually:

  * Only one appearance of the “Final assembly & output” title (and localized equivalents).
  * No tiny “phantom” text behind / above the group header.
* Resize the main window (bigger, smaller) and verify the layout remains clean.

---

## 4. General sanity checks

After implementing all tasks:

1. **Tk GUIs still work**:

   * `zemosaic_gui.py` launches and behaves as before.
   * `zemosaic_filter_gui.py` continues to function unchanged.
2. **Qt main GUI**:

   * No new tracebacks in the console when opening the main window or filter window.
   * Changing language (EN/FR/…) still works; new/updated texts appear translated where applicable.
3. **Qt Filter GUI (performance)**:

   * On large datasets, opening the Filter window doesn’t freeze the application (especially when drawing the sky preview).
   * Coverage map tab still works unchanged.

Document any non-obvious changes with short code comments to help future maintainers.

```
Document here all the task completed by adding a [x] mark before each completed task to control task completion

