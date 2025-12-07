### `followup.md`

```markdown
# Grid mode – legacy stacker revert – Follow-up

## Checklist

- [x] Located `_stack_weighted_patches` in:
      - [x] `grid_mode_last_know_geomtric_tiles_ok.py` (reference)
      - [x] `grid_mode.py` (current)
- [x] Replaced the entire body of `_stack_weighted_patches` in `grid_mode.py`
      with the implementation from `grid_mode_last_know_geomtric_tiles_ok.py`.
- [x] Ensured the function signature (parameters / return type) matches exactly.
- [x] Verified that CPU Grid mode now uses the legacy stacker logic:
      - [x] No delegation to `stack_core` / shared core stacker inside
            `_stack_weighted_patches`.
- [x] Kept GPU helpers (`_stack_weighted_patches_gpu`, etc.) unchanged.
- [x] Left multithread / chunking logic in `process_tile(...)` untouched.
- [ ] Optionally removed any unused imports that were only needed for the new
      stack_core-based `_stack_weighted_patches` implementation.

## Tests

- [ ] Ran Grid mode on the reference dataset with the fixed `grid_mode.py`.
- [ ] Confirmed the pipeline completes without errors.
- [ ] Compared `tile_000X.fits` (1, 2, 4, 5, 7, 8) between:
      - [ ] last-known-good grid,
      - [ ] current grid with legacy `_stack_weighted_patches`.
- [ ] Observed that tiles in the current grid are no longer “mostly empty”:
      - [ ] fraction of non-zero pixels per tile is close to the old version.
- [ ] (Optional) Verified GPU Grid mode still runs if enabled.

## Notes / Observations

- [ ] Document here any remaining differences between old vs new tiles
      (acceptable small floating-point differences vs larger discrepancies).
- [ ] If further issues remain after this revert, they should now be isolated
      to other parts of Grid mode (e.g., photometric equalization or final
      assembly), which can be addressed in a separate mission.
````
