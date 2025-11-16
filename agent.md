
# AGENT MISSION FILE — ZEMOSAIC QT MAIN TAB LAYOUT (SPLITTERS ONLY)

You are an autonomous coding agent working on the **ZeMosaic** project.

The repository already contains (relevant for this task):

- `run_zemosaic.py`           → entry point (Tk / Qt selector)
- `zemosaic_gui.py`           → legacy Tkinter GUI (MUST remain unchanged)
- `zemosaic_filter_gui.py`    → legacy Tkinter filter GUI (MUST remain unchanged)
- `zemosaic_gui_qt.py`        → new PySide6 main GUI (this is your ONLY target)
- `zemosaic_filter_gui_qt.py` → new PySide6 filter GUI (DO NOT touch it in this task)
- `locales/en.json`, `locales/fr.json`
- helper modules (astrometry, worker, utils, etc.) — DO NOT modify them for this task.

Your job in this mission:  
**Refine the layout of the Qt “Main” tab using QSplitter so that the main groups can be resized interactively, without changing any business logic or behaviour.**


## GLOBAL CONSTRAINTS

1. **Tkinter GUI must remain 100% untouched and functional.**
   - Do not modify `zemosaic_gui.py` or `zemosaic_filter_gui.py`.

2. **Non-GUI logic must remain untouched.**
   - Do not change any code related to:
     - processing pipeline, worker threads,
     - configuration loading/saving,
     - astrometry / stacking / mosaic algorithms,
     - logging, progress bars, etc.

3. **Scope is intentionally narrow:**
   - You may only modify:
     - `zemosaic_gui_qt.py` (imports + layout code),
   - No other files, no new dependencies, no changes to translation keys.

4. **Keep existing widget creation methods and translation usage.**
   - The groupbox builders (`_create_folders_group`, `_create_instrument_group`, `_create_mosaic_group`, `_create_final_assembly_group`) must keep:
     - same titles / `_tr` keys,
     - same child widgets, fields and bindings.


## CURRENT SITUATION (BEFORE YOUR CHANGES)

- The Qt main window builds a `QScrollArea` with a `QTabWidget`.
- The **Main** tab is populated by `_populate_main_tab(self, layout: QVBoxLayout)`.
- Inside `_populate_main_tab`, the layout currently uses a fixed `QGridLayout`:

  ```python
  grid_container = QWidget(parent_widget)
  grid_layout = QGridLayout(grid_container)
  ...
  grid_layout.addWidget(self._create_folders_group(), 0, 0)
  grid_layout.addWidget(self._create_instrument_group(), 0, 1)
  grid_layout.addWidget(self._create_mosaic_group(), 1, 0)
  grid_layout.addWidget(self._create_final_assembly_group(), 1, 1)
  layout.addWidget(grid_container)
````

* This forces all cells of the grid row/column to share the same height, leading to a **very tall empty area** in the “Mosaic / clustering” group when “Final assembly” is tall.

The user wants:

* Two main columns (left and right), roughly balanced.
* Within each column, the groups should be **resizable by the user** (dragging splitter handles).
* The column widths themselves must also be adjustable.
* Order of groups remains:

  * Left column: **Folders** (top), **Mosaic / clustering** (bottom).
  * Right column: **Instrument** (top), **Final assembly output** (bottom).

## MISSION OBJECTIVES

### Objective 1 — Import and basic setup

1. In `zemosaic_gui_qt.py`, extend the widget imports to include `QSplitter` (and `QSizePolicy` if needed):

   ```python
   from PySide6.QtWidgets import (
       QApplication,
       QCheckBox,
       QComboBox,
       QDoubleSpinBox,
       QFrame,
       QGraphicsScene,
       QGraphicsView,
       QFileDialog,
       QFormLayout,
       QGridLayout,
       QGroupBox,
       QHBoxLayout,
       QLabel,
       QLineEdit,
       QMainWindow,
       QMessageBox,
       QPlainTextEdit,
       QProgressBar,
       QPushButton,
       QScrollArea,
       QSpinBox,
       QTabWidget,
       QVBoxLayout,
       QWidget,
       QSizePolicy,  # NEW
       QSplitter,    # NEW
   )
   ```

   * Do **not** remove existing imports even if some are unused.

### Objective 2 — Replace the grid layout with nested QSplitters

Re-implement `_populate_main_tab` to use:

* One **horizontal QSplitter** for the two main columns.
* Two **vertical QSplitters**:

  * left vertical splitter: Folders + Mosaic group,
  * right vertical splitter: Instrument + Final assembly group.

**Required structure:**

```text
Main tab (QVBoxLayout)
└─ columns_splitter (QSplitter, Qt.Horizontal)
   ├─ left_column_splitter  (QSplitter, Qt.Vertical)
   │  ├─ folders_group
   │  └─ mosaic_group
   └─ right_column_splitter (QSplitter, Qt.Vertical)
      ├─ instrument_group
      └─ final_assembly_group
```

Concrete implementation:

```python
def _populate_main_tab(self, layout: QVBoxLayout) -> None:
    parent_widget = layout.parentWidget() or self

    # --- main horizontal splitter: left column vs right column ---
    columns_splitter = QSplitter(Qt.Horizontal, parent_widget)

    # ----- LEFT COLUMN: folders + mosaic/clustering (vertical splitter) -----
    left_splitter = QSplitter(Qt.Vertical, columns_splitter)

    folders_group = self._create_folders_group()
    mosaic_group = self._create_mosaic_group()

    left_splitter.addWidget(folders_group)
    left_splitter.addWidget(mosaic_group)

    # Optional: give reasonable initial sizes (folders taller than mosaic)
    left_splitter.setStretchFactor(0, 3)
    left_splitter.setStretchFactor(1, 2)

    # ----- RIGHT COLUMN: instrument + final assembly (vertical splitter) -----
    right_splitter = QSplitter(Qt.Vertical, columns_splitter)

    instrument_group = self._create_instrument_group()
    final_group = self._create_final_assembly_group()

    right_splitter.addWidget(instrument_group)
    right_splitter.addWidget(final_group)

    # Optional: final assembly slightly taller
    right_splitter.setStretchFactor(0, 2)
    right_splitter.setStretchFactor(1, 3)

    # --- Add left and right columns to the main splitter ---
    columns_splitter.addWidget(left_splitter)
    columns_splitter.addWidget(right_splitter)

    # Balance the two main columns (user can override by dragging)
    columns_splitter.setStretchFactor(0, 1)
    columns_splitter.setStretchFactor(1, 1)

    # Finally attach to the tab layout
    layout.addWidget(columns_splitter)
    layout.addStretch(1)
```

Constraints:

* Do not change the *contents* of `_create_*_group` methods, only how their returned widgets are arranged.
* The **scroll area must keep working**:

  * No change is needed around the `QScrollArea` or `QTabWidget` creation.
  * Only replace the internal layout of the `main` tab.

### Objective 3 — Keep the Mosaic / clustering group compact

To avoid the Mosaic group being stretched to absurd heights when space is available, assign it a vertical size policy that prefers minimal height.

Inside `_create_mosaic_group` (and only there), after creating the `QGroupBox`:

```python
def _create_mosaic_group(self) -> QGroupBox:
    group = QGroupBox(self._tr("qt_group_mosaic", "Mosaic / clustering"), self)
    layout = QFormLayout(group)
    layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

    # Prevent this group from greedily expanding vertically.
    group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    ...
    return group
```

* Do **not** change control creation, labels, or binding of widget values.

### Objective 4 — Behaviour and testing

After the change, the expected behaviour is:

1. **Resizable columns**:

   * The user can drag the **central vertical splitter handle** to make the left column wider/narrower relative to the right column.

2. **Resizable groups within each column**:

   * In the left column, the separator between **Folders** and **Mosaic / clustering** can be dragged to change their respective heights.
   * In the right column, the separator between **Instrument** and **Final assembly** can be dragged similarly.

3. **No change in functionality**:

   * All input fields, checkboxes, and buttons behave exactly as before.
   * Running a mosaic must produce identical logs and results compared to the previous Qt version (for the same settings).

4. **Scrollbars still visible when needed**:

   * The window should still show scrollbars when the overall content is taller or wider than the screen.
   * Do not disable `setWidgetResizable(True)` or scroll policies.

## OUT OF SCOPE (DO NOT DO)

* Do not add drag-and-drop reordering of groups (this might come later in another mission).
* Do not modify:

  * other tabs (`solver`, `system`, `advanced`, `skin`, `language`),
  * `zemosaic_filter_gui_qt.py`,
  * any back-end processing / worker logic.
* Do not introduce additional modules, CSS, or stylesheets.
* Do not change translation keys or add new JSON entries in `locales/`.

## DONE CRITERIA

This task is considered complete when:

1. `_populate_main_tab` uses the nested QSplitter structure described above.
2. `Mosaic / clustering` does not expand ridiculously in height when there is free space.
3. The user can resize:

   * the width of each column,
   * the height of the two groups within each column,
     using standard Qt splitter handles.
4. No regressions occur in:

   * interaction (all widgets still present and functional),
   * logging,
   * mosaic processing results.
5. Running `python run_zemosaic.py --qt-gui` shows the updated layout and the classic Tk GUI is unchanged when launched normally or with `--tk-gui`.

````


