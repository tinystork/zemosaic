# Mission: Add automatic ALTZ / EQ segregation in Grid Mode when using stack_plan.csv

## Context

In ZeMosaic Grid mode, when a `stack_plan.csv` file is present(there is one in this repo), the pipeline bypasses
the classic clustering logic (including ALTZ vs EQ orientation separation).

However, `stack_plan.csv` already contains a `mount` column with values like:
- EQ
- ALTZ

This information is currently ignored.

Mixing EQ and ALTZ frames in the same Grid run can degrade geometric quality
(field rotation residuals, PSF inconsistency, seam artifacts, photometric instability).

## Goal

When `stack_plan.csv` contains mount information, automatically segregate frames
by mount mode (EQ vs ALTZ) and process them separately, without requiring
any user knowledge or configuration.

This must be transparent and "magical" for the end user.

## Scope (STRICT)

- ONLY modify Grid mode code paths
- NO refactor
- NO changes to classic clustering pipeline
- NO GUI changes
- NO algorithmic changes beyond segregation
- Backward compatible: if mount info is missing, behavior must remain unchanged

## Functional Requirements

1. Extend stack plan parsing:
   - In `grid_mode.load_stack_plan()`, read column:
     - primary: `mount`
     - optional aliases: `eqmode`, `mount_mode`
   - Normalize values to: `"EQ"` or `"ALTZ"`
   - Store this value per frame (e.g. `frame["mount"]`)

2. Segregation logic:
   - In `grid_mode.run_grid_mode()`:
     - If **all frames have mount info** AND at least two distinct values exist:
       - Split frames into two groups:
         - EQ group
         - ALTZ group
     - If mount column missing or only one mode present:
       - Keep current behavior (single Grid run)

3. Execution model:
   - For split case:
     - Run Grid mode **twice**, once per mount group
     - Each run produces its own mosaic output
     - Outputs must not be merged or reprojected together

4. Output structure:
   - Create subfolders:
     - `grid_EQ/`
     - `grid_ALTZ/`
   - Keep all filenames and logs identical otherwise

5. Logging (important):
   - Emit clear info logs such as:
     - "Grid mode: detected mount-based segregation (EQ / ALTZ)"
     - "Grid mode: EQ frames = X, ALTZ frames = Y"
   - If fallback occurs:
     - "Grid mode: mount info missing or homogeneous, skipping segregation"

## Non-Goals

- Do NOT modify photometric matching
- Do NOT modify cropper
- Do NOT modify worker logic
- Do NOT introduce new CSV columns
- Do NOT change default behavior when stack_plan.csv has no mount column

## Expected Outcome

- Grid mode quality improves automatically on mixed EQ/ALTZ datasets
- End user does nothing and does not need to understand mount mechanics
- Existing stack_plan.csv files continue to work unchanged
