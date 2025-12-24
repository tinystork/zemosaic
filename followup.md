# followup.md — Review & verification checklist (Intertile threadless when workers==1)

## What changed
- `compute_intertile_affine_calibration` now executes **sequentially in the main thread** whenever `effective_workers <= 1`
  instead of creating a `ThreadPoolExecutor(max_workers=1)`.

## Why it matters
- The crash signature shows `astropy.wcs` -> `reproject_interp` access violation inside a ThreadPool thread.
- Windows SAFE_MODE clamps to 1 worker, but the code still ran inside a background thread → crash remained possible.
- Sequential main-thread avoids thread-safety landmines in native libs (wcslib/reproject/opencv).

## Code review checklist
- [ ] Only `zemosaic_utils.py` modified.
- [ ] In the “use_parallel” block:
  - [ ] there is a branch `if effective_workers <= 1:` that runs the sequential loop
  - [ ] ThreadPoolExecutor is only used when `effective_workers >= 2`
- [ ] Progress logging cadence unchanged:
  - [ ] `pairs_done=...` every 25
  - [ ] `progress_callback("phase5_intertile_pairs", ...)` kept
  - [ ] `progress_callback("phase5_intertile", ...)` kept every 5
- [ ] Heartbeat logic remains only in ThreadPool path.
- [ ] No change to `_process_overlap_pair` math or pair generation.

## Repro validation (Windows)
1) Run the dataset that previously produced:
   - `Windows fatal exception: access violation`
   in `%TEMP%\\faulthandler_intertile.log`.

2) Confirm logs now show:
   - SAFE_MODE clamp to 1 worker (if applicable)
   - `Parallel: ... -> 1 (...)`
   - **NEW:** `effective_workers=1 -> running sequentially (no ThreadPoolExecutor)`

3) Confirm:
   - intertile progresses past the previous halt point
   - processing completes and mosaic continues
   - `%TEMP%\\faulthandler_intertile.log` does not contain `access violation`

## Optional non-Windows check
- On Linux/macOS with `cpu_workers=4` and a moderate number of overlaps:
  - confirm ThreadPool path is used when `effective_workers >= 2` (no regression in speed).

## Notes
- This patch is intentionally minimal and risk-free:
  - it changes execution strategy only in the single-worker case
  - science/output remains identical
