# FOLLOW-UP TASK LIST — ZEMOSAIC FILTER GUI QT ENHANCEMENTS

This file defines the exact steps you must execute to complete the mission described in agent.md.

---

# ✅ TASK 1 — Replace Flat QTableWidget With Grouped QTreeWidget

### 1.1 Create a QTreeWidget to replace the existing images table
- Insert in the same container where the current QTableWidget is created.
- Headers: “File”, “WCS”, “Instrument”.

### 1.2 Populate the QTreeWidget in `refresh_items()` or equivalent
For each group:
- create a `QTreeWidgetItem` as group root
- set:
  - checkbox (tri-state)
  - label = “Group X”
- append as top-level item

For each image inside the group:
- create child item with:
  - checkbox
  - File name
  - WCS state
  - Instrument

### 1.3 Implement checkbox propagation
- When a group is checked, check all children  
- When an image is checked:
  - update group tri-state via:
    - all checked → group fully checked  
    - none checked → group unchecked  
    - mixed → group partially checked  

### 1.4 Ensure internal “kept images” logic matches existing flat model

---

# ✅ TASK 2 — Implement Rubber-Band Rectangle Selection in Sky Preview

### 2.1 Add a rectangle selector tool
Choose one:
- `matplotlib.widgets.RectangleSelector`, recommended  
OR  
- custom callbacks (`on_press`, `on_move`, `on_release`)

### 2.2 Draw selection rectangle as transparent blue overlay

### 2.3 On release:
- Convert rectangle bounds to RA/DEC using the existing WCS
- Determine which groups intersect the bounding region:
  - group footprint test OR
  - group centroid test

### 2.4 Sync selection with groups list
- Expand selected groups in the QTreeWidget  
- Check all selected groups  
- Highlight group colors in sky preview (optional but recommended)

---

# ✅ TASK 3 — Maintain Sky Preview Stability
- Ensure your rectangle drawing does not interfere with:
  - existing coverage map
  - existing master-tile outlines
- Add redraw throttling if necessary using QTimer.

---

# ✅ TASK 4 — UI Polish
- Preserve window geometry persistence using existing keys.
- Keep localization keys intact.
- Optional: add “Expand All / Collapse All” buttons if easy.

---

# VALIDATION TESTS

### ✔ Test 1 — Load a Seestar dataset with 10+ groups
- Groups must display correctly in tree form.

### ✔ Test 2 — Check/uncheck groups
- All images inside reflect the change.

### ✔ Test 3 — Partial selection
- Manually check 1 image → group becomes tri-state.

### ✔ Test 4 — Rubber band selection
- Drawing a rectangle over footprints selects correct groups.

### ✔ Test 5 — Multi-group selection
- Selecting a large region checks multiple groups.

### ✔ Test 6 — No regression
- Launch mosaic → identical behavior to previous version.

---

# END OF FOLLOW-UP FILE
