# AGENT MISSION FILE — ZEMOSAIC FILTER GUI QT — GROUPED LIST + SKY SELECTION

You are a coding agent working on the **ZeMosaic** project, specifically the Qt-based Filter GUI (`zemosaic_filter_gui_qt.py`).  
Your mission is to enhance the ergonomics of the Filter interface **without modifying the stacking logic, worker behavior, or any Tkinter GUI.**

Follow this document strictly.  
Do not modify any file not explicitly listed below unless absolutely required for compatibility.

---

# 🔥 OBJECTIVE SUMMARY

Implement **two major ergonomic upgrades** in `zemosaic_filter_gui_qt.py`:

---

## **1. Grouped Images List (replacing the flat QTableWidget)**

Replace the current “Images (check to keep)” flat list with a **group-based hierarchical view**:

### Required:

### ✔ Replace QTableWidget with **QTreeWidget**  
- Root rows represent groups (Group 1, Group 2, etc.)
- Child rows represent images inside the group.

### ✔ Group row behavior  
- Double-click group row → expand/collapse children.
- Checkbox on group:
  - checking group → checks all images inside
  - unchecking group → unchecks all images inside
  - group must support **tri-state** (partially checked)

### ✔ Image row behavior  
- Each image retains:
  - File name  
  - WCS state (Yes/No)  
  - Instrument  
- Checking/unchecking an image updates:
  - group checkbox state  
  - internal “checked items” list used by the processing pipeline

### ✔ Internal state must remain **100% compatible** with the existing worker code  
No change to data structures, only to the GUI representation layer.

---

## **2. User Rectangle Selection in Sky Preview**

The Matplotlib sky preview must support **rubber-band style selection**:

### ✔ Click + drag draws a semi-transparent blue rectangle  
Use either:
- `matplotlib.widgets.RectangleSelector`, OR  
- manual event handling (`button_press_event`, `motion_notify_event`, `button_release_event`)

### ✔ When drag ends:
1. Convert rectangle pixel coordinates → RA/DEC bounds  
2. For each group:
   - If any WCS footprint or group centroid falls inside the rectangle → group is **selected**
3. Selected groups must:
   - Be highlighted in sky preview  
   - Be auto-expanded in the groups list  
   - Be auto-checked in the groups list  
   - Trigger the same “checked items update” logic

### ✔ The existing blue "global WCS frame" must not be removed  
Your new rectangle must be drawn **in a separate layer**.

---

# NON-NEGOTIABLE REQUIREMENTS

- Do **not** alter stacking logic, group computation logic, WCS extraction logic, or worker communication.
- Do **not** touch Tkinter implementation (`zemosaic_filter_gui.py`).
- Do **not** break geometry persistence, localization, or icon loading.
- Do **not** remove or rename existing signals/slots.

---

# FILES YOU ARE AUTHORIZED TO MODIFY

### Mandatory:
- `zemosaic_filter_gui_qt.py`  
  (Implementation of grouped list + sky rectangle selection)

### Optional, only if required:
- `locales/en.json` and `locales/fr.json`  
  (To add new labels for the grouped list)

No other file must be touched.

---

# IMPLEMENTATION GUIDELINES

## Grouped List
- Use `QTreeWidget` with:
  - `setHeaderLabels([“File”, “WCS”, “Instrument”])`
  - `setColumnCount(3)`
  - `setExpandsOnDoubleClick(True)`
  - tri-state enabled via `Qt.ItemIsTristate | Qt.ItemIsUserCheckable`

## Sky Preview
- Add rectangle selector overlay layer
- Maintain a list `selected_groups`
- Ensure update signals do not cause redraw storms (use throttling if needed)

---

# ACCEPTANCE CRITERIA

1. Groups appear as collapsible list nodes.  
2. Group checkbox logic correct (check all, partial state).  
3. Rectangle drag works reliably.  
4. Rectangle correctly identifies groups by RA/DEC overlaps.  
5. Selected groups sync visually and with checkboxes.  
6. No regression in processing pipeline.  
7. No GUI freeze or slowdown.

---

# END OF AGENT FILE
