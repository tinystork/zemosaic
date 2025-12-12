# Mission: Grid mode — connect progress + ETA to PySide6 GUI (no other changes)

## Goal
Fix the GUI status bar in **Grid/Survey mode** so that:
- Progress bar updates during the run (reaches 100% at end)
- ETA updates (via ETA_UPDATE messages)
- Tiles counter shows a meaningful incremental count (X/Y)
- Phase/Stage label updates (optional but recommended)
**WITHOUT** changing SDS mode or the classic pipeline. PySide-only (Tk will be removed later).

## Hard constraints
- Modify **ONLY** `grid_mode.py`.
- Do not change `zemosaic_worker.py`, SDS code paths, classic pipeline, or GUI code.
- Keep all existing logging and algorithm behavior identical (only add progress/ETA reporting).
- Must work even if total counts (overlaps, tiles) are discovered late: reporting should remain stable.

## What the Qt GUI already expects
The PySide listener already handles these queue messages:
- `("STAGE_PROGRESS", stage_name, current_int, {"total": total_int})` -> updates progress bar + stage label
- `("ETA_UPDATE:HH:MM:SS", None, "ETA_LEVEL", {})` -> updates ETA label
- `("MASTER_TILE_COUNT_UPDATE:X/Y", None, "INFO", {})` -> updates Tiles counter label
- `("PHASE_UPDATE:<id>", None, "INFO", {})` -> optional phase label updates
(These are emitted via the existing `progress_callback(msg_key, prog=None, lvl="INFO", **kwargs)` signature.)

Reference: `zemosaic_gui_qt.py` handler logic.

## Implementation plan (grid_mode.py only)
1. [x] Add a small helper class (e.g. `_GridProgressReporter`) inside `grid_mode.py`:
   - [x] Holds:
     - [x] `tile_total`, `tile_done`
     - [x] `overall_total_units`, `overall_done_units`
     - [x] `stage_name`
     - [x] `t0 = time.monotonic()`
     - [x] smoothing state for rate/ETA (simple EMA ok)
     - [x] throttling: do not spam GUI (e.g. emit at most every 0.2–0.5s)
   - [x] Methods:
     - [x] `set_tile_total(n)`
     - [x] `set_overall_total(n_units)` (can be updated once overlaps known)
     - [x] `set_stage(name)`
     - [x] `advance(units=1)` increments overall done and emits STAGE_PROGRESS
     - [x] `tile_completed()` increments tile_done, emits MASTER_TILE_COUNT_UPDATE
     - [x] `emit_eta()` computes ETA from elapsed + rate; emits `ETA_UPDATE:..`
     - [x] `finish()` forces final STAGE_PROGRESS 100% and ETA 00:00:00

2. [x] Decide a stable overall progress model:
   - [x] Use **overall work units** (not per-stage percent) so the global progress bar doesn’t reset.
   - [x] Suggested weights (simple + robust):
     - [x] Setup (grid build / discovery): 1
     - [x] Tile processing: `tile_total`
     - [x] Overlap blending: `n_overlaps` (if available; else 0 and update later)
     - [x] Assembly: `tile_total`
     - [x] Final save/export: 1
   - [x] It’s OK if overlap count is discovered later: update overall_total_units and continue.

3. [x] Wire reporter into the existing run flow:
   - [x] At start of `run_grid_mode`:
     - [x] Instantiate reporter with `progress_callback`
     - [x] `set_stage("GRID: setup")`, `advance(0)`, emit initial ETA `--:--:--`
   - [x] Once grid definition (tiles list) known:
     - [x] `set_tile_total(n_tiles)`
     - [x] emit `MASTER_TILE_COUNT_UPDATE:0/n_tiles`
   - [x] During tile processing:
     - [x] For each tile completion (when a tile is fully stacked and/or saved to disk),
       call `tile_completed()` and `advance(1)` and `emit_eta()`.
   - [x] During blending:
     - [x] If overlaps are processed in a loop, call `advance(1)` per overlap.
   - [x] During assembly / final mosaic:
     - [x] Call `advance(1)` per tile placed or per major step.
   - [x] At end:
     - [x] `finish()` => sends:
       - [x] `STAGE_PROGRESS` current=total
       - [x] `ETA_UPDATE:00:00:00`

4. [x] Ensure the counter semantics:
   - [x] Tiles counter should represent **tiles completed out of total tiles** (incremental X/Y),
     not “sub-tiles in current super-tile”.

5. [x] Do not break log output:
   - [x] Keep existing `_emit(...)` logging behavior.
   - [x] Only add non-invasive progress_callback calls.

## Acceptance criteria
- In Qt GUI, during grid mode run:
  - Progress bar moves and ends at 100%
  - ETA updates periodically and ends at 00:00:00
  - Tiles label shows `0/N` then increments to `N/N`
- No behavior changes in SDS or classic mode (must not touch their code paths).
- No exceptions if progress_callback is None.
- Emission is throttled (no GUI spam / lag).

## Files to change
- `grid_mode.py` only
