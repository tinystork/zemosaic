# followup.md — Fix master-tile stacking pink/magenta bands

## Checklist
- [ ] Identify Phase-3 master-tile code path where `MT_COVERAGE: propagate_mask=...` is logged.
- [ ] For `stack_reject_algo == "winsorized_sigma_clip"`:
  - [ ] Set `propagate_mask=True` in `align_images_in_group(...)` calls for master-tile alignment.
  - [ ] Ensure aligned frames keep NaNs outside footprints (do not sanitize them to zeros).
- [ ] Remove/bypass the “nanized aligned images -> nan_to_num before stacking” block for the winsorized master-tile path.
- [ ] In `zemosaic_align_stack.py`, make `_winsorize_block_numpy` use `np.nanquantile` (with safe fallback).
- [ ] If an external `cpu_stack_winsorized` exists:
  - [ ] Add guard: if any frame contains non-finite values, force internal fallback (or prove external impl is NaN-safe).
- [ ] Run a quick local repro:
  - [ ] One dataset that previously produced magenta bands (e.g., Pleiades mosaic)
  - [ ] Confirm master tiles are clean after overstretch
  - [ ] Confirm final mosaic no longer shows the magenta tile borders
- [ ] Regression safety:
  - [ ] Verify no code touched batch-size behavior
  - [ ] Verify no changes in SDS / grid / “using master tiles” code paths
- [ ] (Optional) Add a minimal unit test for NaN-safe winsor stacking.

## Notes / Observations (fill during work)
- propagate_mask log before:
  - `MT_COVERAGE: propagate_mask=False`
- propagate_mask log after:
  - `MT_COVERAGE: propagate_mask=True`
- Any remaining artifacts:
  - (paste screenshots / tile ids)
