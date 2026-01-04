# followup.md — Fix master-tile stacking pink/magenta bands

## Checklist
- [x] Identify Phase-3 master-tile code path where `MT_COVERAGE: propagate_mask=...` is logged.
- [x] For `stack_reject_algo == "winsorized_sigma_clip"`:
  - [x] Set `propagate_mask=True` in `align_images_in_group(...)` calls for master-tile alignment.
  - [x] Ensure aligned frames keep NaNs outside footprints (do not sanitize them to zeros).
- [x] Remove/bypass the “nanized aligned images -> nan_to_num before stacking” block for the winsorized master-tile path.
- [x] In `zemosaic_align_stack.py`, make `_winsorize_block_numpy` use `np.nanquantile` (with safe fallback).
- [x] In `zemosaic_align_stack_gpu.py`, make `_winsorize_chunk` NaN-safe (`cp.nanpercentile` when needed, `cp.percentile` otherwise).
- [x] Make GPU percentiles used by linear-fit normalization NaN-safe (avoid `cp.percentile` on non-finite data).
- [x] GPU winsorized path: skip linear-fit normalization and preserve NaNs through stacking (avoid `nan_to_num` dilution).
- [x] If an external `cpu_stack_winsorized` exists:
  - [x] Add guard: if any frame contains non-finite values, force internal fallback (or prove external impl is NaN-safe).
- [ ] Run a quick local repro:
  - [ ] One dataset that previously produced magenta bands (e.g., Pleiades mosaic)
  - [ ] Confirm master tiles are clean after overstretch
  - [ ] Confirm final mosaic no longer shows the magenta tile borders
- [ ] Regression safety:
  - [x] Verify no code touched batch-size behavior
  - [ ] Verify no changes in SDS / grid / “using master tiles” code paths
- [x] (Optional) Add a minimal unit test for NaN-safe winsor stacking.
- [x] (Optional) Add a GPU/CPU parity test for winsorized NaN edges (skips if GPU unavailable).

## Notes / Observations (fill during work)
- propagate_mask log before:
  - `MT_COVERAGE: propagate_mask=False`
- propagate_mask log after:
  - `MT_COVERAGE: propagate_mask=True`
- Any remaining artifacts:
  - (paste screenshots / tile ids)
- Local repro data path:
  - `/mnt/c/Users/TRISTAN/Desktop/astrophoto/M45_CHEEMS`
- Test run:
  - `pytest -q tests/test_winsor_nan_safe.py` failed during collection with `FileNotFoundError` in pytest capture temp file handling.
  - `pytest -q tests/test_gpu_cpu_parity.py::test_gpu_stack_winsorized_nan_policy_matches_cpu` not run (GPU/pytest capture environment unknown).
- Investigation note:
  - GPU linear-fit normalization used `zemosaic_utils._percentiles_gpu` (`cp.percentile`) which returns NaN when any NaNs exist; this zeroed frames (a=0) and killed signal. Switched to NaN-aware percentiles when non-finite data is present.
  - GPU winsorized stacking was still normalizing and `nan_to_num`-ing NaNs to zeros, diluting signal when many frames have NaN footprints. Skipped normalization + kept NaNs for winsorized GPU path.
