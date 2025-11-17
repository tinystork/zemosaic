# AGENT MISSION — ZEMOSAIC FILTER QT FIXES

You are an autonomous coding agent working inside the **ZeMosaic** project.  
Your modifications MUST satisfy the following rules:

### GLOBAL RULES
- Do **not** break the Tk GUI (`zemosaic_filter_gui.py`).  
- Do **not** modify any business logic in the worker or utils.  
- Only adjust the PySide6 Qt GUI (`zemosaic_filter_gui_qt.py`) and only when explicitly requested.  
- Never alter file formats or pipeline behaviour.  
- Keep cross-platform compatibility: Windows, macOS, Linux.  
- Keep icons loading intact.  
- NEVER touch batch-size logic (batch size = 0 or >1 MUST remain untouched).

---

# TASK GROUP A — FIX MISSING GROUP WCS OUTLINES
Goal: when opening the Qt filter GUI, **if frames already contain valid WCS**, only the **group-level bounding boxes** shall be drawn (NOT the per-frame outlines).

### REQUIRED ACTIONS
1. Ensure `_update_preview_plot()` (or the drawing routine using matplotlib) properly calls:
   - the group-WCS extraction logic  
   - the group-WCS footprint drawing (rectangles)
2. If WCS exists:
   - **Do not attempt to draw individual footprints** (this previously caused freezes)
   - Ensure each group footprint is computed using the same algorithm as in Tkinter:
     - Collect group frames
     - Compute min/max RA/DEC
     - Build rectangular footprint
3. Fix the bug where the group outlines “exist” logically but are not drawn.

---

# TASK GROUP B — REMOVE SCAN/GROUPING LOG PANEL
In `zemosaic_filter_gui_qt.py`, fully remove (not hide):

- The “Scan / grouping log” text box
- Associated widgets, layouts, and update calls
- Any code that inserts text into this log
- Any scrollbars tied to it

Nothing else in the layout must be altered.

---

# TASK GROUP C — CODE HYGIENE
- Remove dead code related to the removed log panel.
- Ensure no orphan signal or method remains.
- Keep window geometry persistence as is.
- Do not modify other UI sections.

---

# OUTPUT
Apply all patches directly to:

- `zemosaic_filter_gui_qt.py`

Do not modify any other file.

