# agent.md — Intertile: no ThreadPool when effective_workers == 1 (main-thread sequential)

## Goal
Prevent native crashes like:
- `Windows fatal exception: access violation`
- stack in `astropy.wcs` (`_all_world2pix`) via `reproject_interp`
occurring when intertile runs inside a ThreadPool thread even though it has been clamped to a single worker.

Key idea:
> If `effective_workers <= 1`, DO NOT create `ThreadPoolExecutor`. Run the overlap-pair loop sequentially in the main thread.

This is OS-agnostic:
- fixes Windows safe_mode case (clamped to 1)
- also removes pointless ThreadPool overhead for any OS when workers end up at 1.

## Constraints
- Minimal invasive patch.
- No change to the math/science (same pairs processed, same outputs).
- Keep existing multi-worker parallel path for `effective_workers >= 2`.
- Keep existing logging + progress_callback behavior.

## Files
- **Modify only**: `zemosaic_utils.py`
- Function: `compute_intertile_affine_calibration(...)`

## Patch plan
1) Locate the section that currently does:
   - compute `effective_workers` and logs:
     `Parallel: threadpool workers=... -> effective_workers ...`
   - then unconditionally executes:
     `with ThreadPoolExecutor(max_workers=effective_workers) as executor: ...`
   This is wrong when `effective_workers == 1`.

2) Refactor ONLY this part:
   - After computing `effective_workers` and logging the clamp, branch:

   **If `effective_workers <= 1`:**
   - Log one explicit line (INFO):
     - e.g. `Parallel: effective_workers=1 -> running sequentially (no ThreadPoolExecutor) to avoid native thread issues`
   - Run the same sequential loop used in the existing non-parallel fallback:
     - `for idx, overlap in enumerate(overlaps, 1): ...`
     - call `_process_overlap_pair(idx, overlap)`
     - append `pair_entries`, update `connectivity`
     - keep identical progress logging cadence (`% 25`) and `progress_callback` calls (`phase5_intertile_pairs`, `phase5_intertile`)

   **Else (`effective_workers >= 2`):**
   - Keep current ThreadPoolExecutor + wait/heartbeat logic unchanged.

3) Ensure the old `use_parallel` guard still works:
   - The intention is:
     - “Parallel requested” when `cpu_workers > 1` and `total_pairs >= 4`
     - But *actual* use of a threadpool must require `effective_workers >= 2`
   - Easiest: keep `use_parallel` as-is, but inside the block branch by `effective_workers`.

4) Keep faulthandler enable/disable logic intact.

## Acceptance criteria
- When logs show clamping to 1 worker (safe_mode or other clamp reasons), we must also see the new line:
  - `running sequentially (no ThreadPoolExecutor)`
- No ThreadPoolExecutor is created when `effective_workers == 1`.
- For `effective_workers >= 2`, behavior is unchanged (ThreadPool path).
- Progress logs still appear every 25 pairs, and at the end.
- No change in outputs for a known dataset (aside from removing crashes).

## Quick tests (lightweight)
- `python -m py_compile zemosaic_utils.py`
- Run a normal mosaic with intertile enabled on Windows that previously triggered the access violation:
  - confirm it passes beyond the previous stopping point
  - confirm `faulthandler_intertile.log` no longer shows access violation
- Optional sanity: on a small dataset on Linux/macOS, confirm multi-worker still uses ThreadPool when effective_workers >= 2.
