# FOLLOW-UP TASKS — ZEMOSAIC FILTER QT

After completing the main AGENT MISSION, perform these follow-up tasks:

### 1. VALIDATE GROUP OUTLINES
- Confirm that when frames contain valid WCS:
  - Group rectangles are visible
  - They refresh when regrouping occurs
  - Zooming and panning do not remove the outlines
- Ensure the legend reflects correct group count and colors.

### 2. PERFORMANCE CHECK
- Verify that drawing only group outlines eliminates UI freeze when:
  - 2000+ frames are loaded
  - Coverage-first clustering is enabled
  - Orientation split is used

### 3. REMOVE RESIDUAL REFERENCES TO REMOVED LOG
- Remove any functions or handler code referencing:
  - “scan log”
  - “grouping log”
  - “append_to_log”
  - “log_widget”
  - or equivalent names

### 4. ENSURE NO REGRESSION IN THESE FEATURES
- Auto-organize master tiles
- “Coverage-first clustering”
- Auto split by orientation
- Export CSV
- WCS resolution check
- ZeSupaDupStack toggle

### 5. FINAL POLISH
- Re-run the preview twice to confirm no exceptions in console.
- Save+restore window geometry continues to work.
- Tab switching (Sky Preview / Coverage Map) works without warnings.

### 6. CONSISTENT GROUP BOX SIZE (MATCH FIRST TILE WCS)
To better reflect the real mosaic:

- For each **group** used in the Qt filter preview:
  - Identify the **first item in the group** that has a valid WCS footprint.
  - Compute this tile’s footprint in RA/DEC (four corners).
  - Derive its approximate **width and height** in degrees:
    - `width_deg = max(ra) - min(ra)`
    - `height_deg = max(dec) - min(dec)`
- When building the **group outline rectangle**:
  - Use the group’s existing center (mean RA/DEC or current centroid).
  - Build a rectangle aligned with RA/DEC axes whose **width/height exactly match**
    `width_deg` / `height_deg` of that first WCS footprint.
- Apply this only to the **Qt filter GUI** outline drawing:
  - Do NOT change the worker logic or FITS/WCS data.
  - Do NOT change how Tk draws its own outlines.
- If no member of a group has a valid WCS footprint:
  - Fallback to the current behaviour for that group (no outline or legacy size).

Verify visually that:
- Each group box has a coherent size comparable to a single tile footprint.
- The preview gives a realistic impression of how the future mosaic will look.
