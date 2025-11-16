
---

## `followup.md`

```markdown
# FOLLOW-UP LOG — ZEMOSAIC QT MAIN TAB (SPLITTER LAYOUT)

## Context

Goal of this micro-mission:  
Replace the fixed grid layout of the **Main** tab in `zemosaic_gui_qt.py` with a QSplitter-based layout so that:

- The **left and right columns** can be resized horizontally.
- Inside each column, the **two main groupboxes** can be resized vertically.
- The overall look stays close to the Tk version but less “empty” on the left side.

Files touched:  
- ✅ `zemosaic_gui_qt.py` (ONLY)  
Files explicitly **not** touched:  
- ❌ `zemosaic_gui.py`, `zemosaic_filter_gui.py`  
- ❌ `zemosaic_filter_gui_qt.py`  
- ❌ any worker / processing / config / localization modules.


## Checklist for the agent

### 1. Imports

- [x] Add `QSplitter` and `QSizePolicy` to the `from PySide6.QtWidgets import ...` block.
- [x] Keep all existing imports intact.

### 2. Main tab layout

- [x] Replace the existing `QGridLayout`-based `_populate_main_tab` implementation with:
  - a horizontal `QSplitter` (`columns_splitter`),
  - two vertical `QSplitter`s (`left_splitter`, `right_splitter`),
  - left splitter contains: folders group, mosaic group,
  - right splitter contains: instrument group, final assembly group.
- [x] Set sensible `setStretchFactor` values for:
  - left vs right columns,
  - top vs bottom groups inside each column.
- [x] Keep the `layout.addWidget(...); layout.addStretch(1)` structure so that the scroll area still wraps the tab.

### 3. Mosaic group compactness

- [x] In `_create_mosaic_group`, after constructing the `QGroupBox`, set:

  ```python
  group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
  ```

* [x] Ensure no other behaviour or bindings in this method are changed.

### 4. Manual tests to perform

Using a small sample of Seestar lights (or any existing test set):

1. **Layout sanity check**

   * [ ] Launch `python run_zemosaic.py --qt-gui`.
   * [ ] Confirm the Main tab shows two columns with four groups:

     * Left: Folders (top), Mosaic / clustering (bottom).
     * Right: Instrument (top), Final assembly (bottom).
   * [ ] Confirm that:

     * the central vertical splitter can be dragged,
     * the horizontal splitters in each column can be dragged.

2. **Scrolling**

   * [ ] Resize the window to a smaller height/width and confirm:

     * scrollbars appear when expected,
     * all controls remain reachable.

3. **Functional regression check**

   * [ ] Configure a quick mosaic run (same settings as before).
   * [ ] Run the processing; verify:

     * logs and console outputs are still produced,
     * no new warnings/exceptions related to the GUI,
     * results (mosaic output) are identical or within normal numerical noise.

4. **Tkinter fallback**

   * [ ] Launch plain `python run_zemosaic.py` (no `--qt-gui`).
   * [ ] Confirm the legacy Tkinter GUI still appears and behaves exactly as previously.

> Manual Qt & Tk GUI checks are pending because this environment cannot launch a display.

## Notes / Future ideas (NOT PART OF THIS TASK)

* Potential future mission: add **drag-and-drop reordering** of groupboxes inside each column (e.g., move Mosaic above Folders). This would require a dedicated design (custom container or list-like widget).
* Potential future mission: expose per-column layout presets (e.g., “balanced”, “left focus”, “right focus”) in the Skin tab.

For now, only splitter-based resize is required.

```
