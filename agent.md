# AGENT MISSION — ZEMOSAIC QT FILTER UPGRADE

You are an autonomous coding agent working inside the **ZeMosaic** project.

Your mission is to modify *only* the PySide6 GUI implementation (`zemosaic_filter_gui_qt.py` and when necessary `zemosaic_gui_qt.py`) **without breaking**:

- the Tkinter GUI (`zemosaic_filter_gui.py`, `zemosaic_gui.py`)
- the worker logic (`zemosaic_worker`)
- the stacking behaviour (batch_size = 0 and batch_size > 1 must remain untouched)
- all instrument, WCS, mosaic and SDS logic
- cross-platform compatibility (Windows / macOS / Linux)

Nothing in the business logic or file formats must be changed.

---

# PRIMARY OBJECTIVES (Qt filter GUI only)

## A. FIX GROUP WCS OUTLINES
- Ensure group-level WCS bounding boxes appear correctly in the Sky Preview when files contain WCS.
- Improve logic so that outlines are built from groups *even if* only clustering is used (no SDS yet).
- Do *not* draw per-frame footprints to avoid freezes. Only draw group boxes.

## B. CONSISTENT GROUP BOX SIZE
- Each group’s bounding box must use the **footprint size of the first WCS tile in that group** as template size.
- The group’s effective box must have the correct width/height in RA/DEC degrees.
- Orientation stays aligned with RA/DEC axes (no rotation).

## C. REMOVE USELESS UI ELEMENT
- Completely remove the “Scan / grouping log” panel from the Qt UI.
- Remove the widget, its layout entry, and all code that updates this log.
- Do not remove anything else.

---

# NEW OBJECTIVES — USER BOUNDING BOX (SELECTION TOOL)

## D. ADD A USER-SPECIFIC SKY BOUNDING BOX
The Sky Preview must support a *user-drawn selection bounding box*.

### Requirements:
1. Add drawing of a user bounding box (drag on Sky Preview).
2. Store the bounding box internally as RA/DEC limits:
{ "ra_min": ..., "ra_max": ..., "dec_min": ..., "dec_max": ... }
3. This bounding box **must not** interfere with group-WCS outlines (both can co-exist).
4. The bounding box must be optional. When None, everything behaves as today.

---

## E. RIGHT-CLICK TO CLEAR USER BOUNDING BOX
Add a context menu (right-click on the Sky Preview) with:

- **“Clear selection bounding box”**  
(use Qt localization system if available)

When clicked:
- remove the drawn rectangle
- reset internal state to None
- refresh the plot

This action must not disturb group outlines, treeview, or clustering.

---

# ADVANCED BEHAVIOUR — BOUNDING BOX INTEGRATION

## F. BOUNDING BOX FILTERING FOR AUTO-ORGANIZE MASTER TILES
When a user bounding box is active:

- Only the frames **whose center RA/DEC lies inside the bounding box** must be considered candidates.
- Frames outside must not be included in clustering or grouping.
- If no frame lies inside the bbox:
- Cancel auto-organize
- Show a small warning (QMessageBox or inline log)

### Filtering rule:
include item if item.center_ra and item.center_dec lie inside bbox RA/DEC

Handle RA wrap-around at 0°/360°.

---

## G. SYNCHRONIZE TREEVIEW WITH BOUNDING BOX FILTER
After filtering:
- Only frames inside the bbox may be checked/selected in the tree.
- Frames outside remain unchecked.

Tree must accurately reflect what will be passed to the worker.

---

# CRITICAL CONSTRAINT — DO NOT MODIFY THE WORKER

## H. WORKER CALL MUST REMAIN IDENTICAL
The worker (`zemosaic_worker.run_hierarchical_mosaic[_process]`) must:

- receive the **exact same structure** it expects in Tk mode
- only difference allowed: the list of frames may be shorter (filtered by bbox)
- no change in data format, keys, object types, or processing steps

Bounding box logic must stay **entirely inside the Qt filter GUI**.

Do NOT modify:
- `zemosaic_worker.py`
- `zemosaic_utils.py`
- any SDS logic
- any FITS-handling logic

The worker must remain unaware of the bounding box.

---

# I. FINAL VALIDATION
After implementing everything:
- Sky preview must show:
  - user bounding box when drawn
  - group bounding boxes with correct size
- Right-click → bounding box disappears
- Auto-organize must:
  - use only frames in bbox
  - produce smaller or identical group sets
  - never include frames outside bbox
- Treeview selection matches filtered frames
- Tk interface remains 100% unchanged

---