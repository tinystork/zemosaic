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
1. Add a small helper class (e.g. `_GridProgressReporter`) inside `grid_mode.py`:
   - Holds:
     - `tile_total`, `tile_done`
     - `overall_total_units`, `overall_done_units`
     - `stage_name`
     - `t0 = time.monotonic()`
     - smoothing state for rate/ETA (simple EMA ok)
     - throttling: do not spam GUI (e.g. emit at most every 0.2–0.5s)
   - Methods:
     - `set_tile_total(n)`
     - `set_overall_total(n_units)` (can be updated once overlaps known)
     - `set_stage(name)`
     - `advance(units=1)` increments overall done and emits STAGE_PROGRESS
     - `tile_completed()` increments tile_done, emits MASTER_TILE_COUNT_UPDATE
     - `emit_eta()` computes ETA from elapsed + rate; emits `ETA_UPDATE:..`
     - `finish()` forces final STAGE_PROGRESS 100% and ETA 00:00:00

2. Decide a stable overall progress model:
   - Use **overall work units** (not per-stage percent) so the global progress bar doesn’t reset.
   - Suggested weights (simple + robust):
     - Setup (grid build / discovery): 1
     - Tile processing: `tile_total`
     - Overlap blending: `n_overlaps` (if available; else 0 and update later)
     - Assembly: `tile_total`
     - Final save/export: 1
   - It’s OK if overlap count is discovered later: update overall_total_units and continue.

3. Wire reporter into the existing run flow:
   - At start of `run_grid_mode`:
     - Instantiate reporter with `progress_callback`
     - `set_stage("GRID: setup")`, `advance(0)`, emit initial ETA `--:--:--`
   - Once grid definition (tiles list) known:
     - `set_tile_total(n_tiles)`
     - emit `MASTER_TILE_COUNT_UPDATE:0/n_tiles`
   - During tile processing:
     - For each tile completion (when a tile is fully stacked and/or saved to disk),
       call `tile_completed()` and `advance(1)` and `emit_eta()`.
   - During blending:
     - If overlaps are processed in a loop, call `advance(1)` per overlap.
   - During assembly / final mosaic:
     - Call `advance(1)` per tile placed or per major step.
   - At end:
     - `finish()` => sends:
       - `STAGE_PROGRESS` current=total
       - `ETA_UPDATE:00:00:00`

4. Ensure the counter semantics:
   - Tiles counter should represent **tiles completed out of total tiles** (incremental X/Y),
     not “sub-tiles in current super-tile”.

5. Do not break log output:
   - Keep existing `_emit(...)` logging behavior.
   - Only add non-invasive progress_callback calls.

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
