# Agent mission – Fix duplicated labels / overlapped text in WCS / Master Tile controls (Qt)

## Repo / context

- Project: **ZeMosaic**
- UI: **Qt / PySide6** filter dialog + main Qt GUI
- The bug is visible in the **Qt filter dialog** in the `WCS / Master tile controls` panel:
  - Under the field **“Overlap between batches (%)”** there is a garbled, overlapping label such as:
    - `OreAtatscpl by (degitation)` or similar.
  - Visually this looks like **two labels/widgets drawn on top of each other**.

The classic Tk interface does not show this issue.

The goal is to **clean up the layouts in the Qt UIs** so that each label appears once, with a clean, localized text, and the layout remains stable.

---

## Goal

1. **Identify and remove duplicated / overlapping widgets** in the **WCS / Master Tile controls** section of the Qt filter dialog (and any equivalent in the main Qt window, if present).
2. Ensure that:
   - Only one label is associated to **“Overlap between batches (%)”**.
   - There is **no stray “Auto-split by …” label** left over from earlier refactors unless it is actually used and wired correctly.
   - All labels in that panel use the localization system (no hard-coded English/French strings if a key exists).
3. Keep the layout visually clean in both **English and French**.

---

## Files to inspect (primary)

- `zemosaic_filter_gui_qt.py`  
  - Qt filter dialog – contains the `WCS / Master tile controls` panel.
- `zemosaic_gui_qt.py`  
  - Main Qt window – check if it has a *duplicate implementation* of WCS/Master Tile controls and align behavior / labels if necessary.

Also keep an eye on:

- `locales/en.json`
- `locales/fr.json`
- `locales/zemosaic_localization.py`

to confirm that the labels used in the WCS/Master Tile section are properly localized.

---

## Detailed tasks

### 1. Locate the WCS / Master Tile controls creation code

- [ ] In **`zemosaic_filter_gui_qt.py`**, locate the code that builds the group box for:
  - `WCS / Master tile controls`
  - or similarly named group (look for `Overlap between batches`, `Max ASTAP instances`, `Coverage-first clustering`, etc.).
- [ ] Identify the **layout object** used in this group (likely `QFormLayout` or `QGridLayout`).
- [ ] Find where the widgets for **“Overlap between batches (%)”** and the mysterious **“Auto-split by …”** are added.

### 2. Detect duplicated labels/widgets

- [ ] Search for any label or control related to:
  - `Auto-split` / `orientation split` / `Overlap between batches` in `zemosaic_filter_gui_qt.py`.
- [ ] Confirm whether:
  - The **same spinbox** is added twice to the layout, or
  - Two different `QLabel`s are being positioned in the same cell / row, or
  - A leftover label is still created but no longer associated with a config field.

Common patterns to check:

- Multiple `layout.addRow(...)` or `layout.addWidget(...)` calls involving:
  - the same widget instance, or
  - two labels that logically should correspond to a single field.

### 3. Decide on the intended UI

We want the WCS / Master tile controls to have:

- [ ] A **single, clear field**:
  - **“Overlap between batches (%)”** with its spinbox.
- [ ] If there is supposed to be an **orientation split / auto-split angle** control:
  - It must be clearly labeled (e.g. “Split by orientation (deg) 0=off”),
  - It must be wired to a real configuration parameter,
  - And it must **not overlap** with the overlap field.

If that orientation/auto-split control is a leftover and no longer used:

- [ ] Remove its label + widget from the layout, and do **not** keep an unused member variable.

### 4. Clean up the layout (Qt filter dialog)

- [ ] Remove any **duplicate `addRow` / `addWidget`** calls that add:
  - the `Overlap between batches (%)` widget more than once, or
  - a garbled/unused `Auto-split by` label on the same row.
- [ ] Ensure the final layout order is logical, for example:

  1. Max ASTAP instances
  2. Draw group WCS outlines, Write WCS to file
  3. Coverage-first clustering toggle
  4. Over-cap allowance (%)
  5. **Overlap between batches (%)**
  6. (Optional) A single, well-labelled “Split by orientation” / “Auto-split by angle” control if still relevant.
  7. SDS toggle, etc.

- [ ] Verify that **each label in this section uses the localization layer**:
  - Fetch text via `self._localizer.get("some_key", "Fallback text")` or equivalent.
  - If new keys are needed, add them to both `en.json` and `fr.json`.

### 5. Check for a duplicate implementation in the main Qt window

- [ ] In `zemosaic_gui_qt.py`, search for any UI code that duplicates the **WCS / Master tile controls** for the main window.
- [ ] If a similar “Overlap between batches / Auto-split” mixture exists there, apply the **same cleanup strategy**:
  - Exactly one label per control,
  - No overlapping row placements,
  - All labels localized.

### 6. Regression safety

- [ ] Make sure you do **not** change any configuration keys, signal/slot connections, or logic behind:
  - `Overlap between batches (%)`
  - SDS toggle
  - Coverage-first clustering toggle
- [ ] The fix must only:
  - Clean up labels,
  - Clean up layout,
  - Remove unused widgets.

---

## Tests / manual checks

Please perform at least these manual checks:

1. **Start the Qt filter dialog**
   - [ ] Run the application with Qt backend (e.g. `python run_zemosaic.py --qt-gui` or equivalent).
   - [ ] Open the **Filter / clustering dialog**.
   - [ ] Confirm that in the **WCS / Master tile controls** group:
     - There is **no overlapping/garbled text**,
     - “Overlap between batches (%)” appears once, with a spinbox next to it.
     - If an orientation/split field exists, it is cleanly labeled and aligned.

2. **Language switch**
   - [ ] Switch between **English** and **French** in the Qt main window.
   - [ ] Re-open the filter dialog.
   - [ ] Confirm that:
     - All labels in the WCS/Master tile controls section are translated consistently,
     - No label disappears or overlaps after language change.

3. **Basic behavior**
   - [ ] Change the **Overlap between batches (%)** value and close the filter.
   - [ ] Verify (via logs or configuration dump if available) that the value is still properly propagated to the worker (no regressions).

---

## Definition of done

- No visual overlap or double text in the **WCS / Master tile controls** section of the Qt filter dialog.
- No stray or unused “Auto-split by …” labels.
- Layout is clean and stable in both **EN** and **FR**.
- No changes to underlying worker behavior or configuration semantics.
