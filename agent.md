# Mission: Ensure lecropper is actually applied to Master Tiles when enabled + propagate masked output

## Context
In ZeMosaic, master tiles can be post-processed by `lecropper.py` through `_apply_lecropper_pipeline()` in `zemosaic_worker.py`.

Current issue:
- When `lecropper.mask_altaz_artifacts()` returns `(masked, mask2d)`, the worker sets `out = base_for_mask` (original) and only keeps `mask2d`, which can discard the actual cleaned image `masked`.
- Users report that enabling the GUI option does not reliably guarantee that master tiles are truly modified by lecropper (especially in alt-az cleanup path).

## Goal
1) When the user enables the "lecropper / Alt-Az cleanup / MT pipeline" option in the GUI, the worker MUST pass every built Master Tile through `_apply_lecropper_pipeline()` (for applicable phases, i.e. after stacking a master tile but before it is persisted and used for final mosaic assembly).
2) If `mask_altaz_artifacts()` returns `(masked, mask2d)`, the worker MUST propagate `masked` as the tile image output (so visual corrections are applied), while still returning / preserving `mask2d` for downstream masking/weighting logic.

## Non-goals / constraints
- Do NOT refactor pipeline design.
- Do NOT change stacking logic, alignment logic, reproject logic, or tile clustering.
- Do NOT touch batch-size special behavior ("batch size = 0" and "batch size > 1" behavior must remain unchanged).
- `lecropper.py` must remain standalone and runnable outside ZeMosaic (no ZeMosaic-only imports inside lecropper).
- Keep changes surgical and limited to: `zemosaic_worker.py` and GUI flag wiring (Qt and/or Tk depending on where the option lives).
- Add logs only where needed for proof.

## Files in scope
- `zemosaic_worker.py`
- GUI: whichever file actually defines the option and passes config to the worker:
  - Qt: `zemosaic_gui_qt.py` (and possibly `zemosaic_filter_gui_qt.py` if the option is there)
  - Tk: `zemosaic_gui.py` (only if needed)

## Implementation tasks

### Task A — Fix propagation semantics in `_apply_lecropper_pipeline`
In `zemosaic_worker.py`:
- Locate `_apply_lecropper_pipeline(...)`.
- Current behavior (conceptually):
  - if mask_altaz_artifacts returns `(masked, mask2d)`:
    - out = base_for_mask (original)
    - return out, mask2d
- Required behavior:
  - if `(masked, mask2d)`:
    - set `out = masked` (NOT the original)
    - still return `mask2d` unchanged
  - Preserve existing fallbacks when `mask_altaz_artifacts` returns only `masked` or returns None.
- Add a debug/info log line that proves which path was taken:
  - Example: `MT_PIPELINE: altaz_cleanup applied: masked_used=True mask2d_used=True`
  - Keep logs stable/greppable.

Acceptance:
- When alt-az cleanup produces both masked and mask2d, the saved Master Tile data must reflect the masked output.

### Task B — Guarantee the option is honored end-to-end (GUI -> config -> worker)
We need to ensure that when user checks the option in the GUI, the worker receives a config flag and uses it to call `_apply_lecropper_pipeline()` for each Master Tile.

Steps:
1. Identify the GUI checkbox / option name.
2. Ensure it is persisted into the config object passed to the worker thread/process.
3. In the worker, locate the Master Tile construction path (right after the tile stack is produced and before it is written/cached or used for mosaic).
4. Gate the `_apply_lecropper_pipeline()` call on that flag.

Add proof logs:
- At start of run (once): log the boolean flags: `lecropper_enabled`, `quality_crop_enabled`, `altaz_cleanup_enabled`.
- Per master tile (already exists partially): ensure log includes `lecropper_applied=True/False`.

Acceptance:
- When GUI option enabled, log must contain per-tile `MT_PIPELINE: lecropper_applied=True`.
- When disabled, it must be `False` and worker must not call lecropper.

### Task C — Minimal regression safety
Add a tiny self-test helper (no heavy pytest harness needed):
- A small function in `zemosaic_worker.py` guarded under `if __name__ == "__main__":` OR a lightweight internal test function callable from existing debug scripts.
- It should simulate:
  - `mask_altaz_artifacts` returning `(masked, mask2d)` where masked is visibly different (e.g., add +1 to array)
  - Ensure returned `out` equals masked (not original)
- Keep it optional and not run during normal GUI execution.

If you do add a test file, keep it minimal and do not add new dependencies.

## Deliverables
- A git-ready patch (unified diff) that applies cleanly.
- Brief commit message suggestion.

## Verification checklist
- Run ZeMosaic with lecropper option ON:
  - confirm logs show `MT_PIPELINE: lecropper_applied=True`
  - confirm `masked_used=True` when altaz returns mask2d
- Run with option OFF:
  - confirm `lecropper_applied=False`
- Confirm no changes to batch-size behavior.
