# agent.md — Fix master-tile stacking pink/magenta bands (NaN-safe winsor + propagate footprint)

## Context / Problem
In many master tiles, we see persistent pink/magenta “bands” or borders that survive the whole pipeline and remain visible in the final mosaic.

Logs confirm that during Phase-3 master-tile creation we currently align frames with:
- `MT_COVERAGE: propagate_mask=False`

Meaning: alignment footprints are NOT propagated; pixels outside the valid overlap are kept as finite values (edge-fill / zeros / noise), then become colored through stacking/post steps, and are not removed by quality crop.

There is also code that explicitly converts NaNs to zeros before stacking:
- `# If we nanized aligned images for coverage, clean them before stacking to avoid stacker ERROR logs.`
This defeats NaN masking and re-injects border artifacts into the stack.

Root cause: winsorized sigma clip CPU path uses `np.quantile` inside `_winsorize_block_numpy`, which is not NaN-safe. This likely forced `propagate_mask=False` upstream to avoid NaNs, creating the visible artifacts.

## Goal
Make master-tile stacking correctly ignore pixels outside alignment footprints:
1) Make winsorized sigma clip NaN-safe on CPU (use nan-aware quantiles).
2) Enable `propagate_mask=True` for Phase-3 master-tile alignment (at least when using winsorized_sigma_clip).
3) Do NOT convert NaNs to zeros before stacking for the winsorized path.
4) Ensure no regressions in SDS mode, grid mode, and “I’m using master tiles” paths.

## Non-goals
- Do NOT change AltAz cleanup behavior or thresholds (ignore AltAz cleanup for now).
- Do NOT change batch-size semantics (“batch size = 0” and “batch size > 1” behavior must remain untouched).
- Do NOT change phase-4/phase-5 photometric normalization logic (second pass is assumed OK).
- Do NOT redesign quality-crop logic; only make it effective by delivering clean edges.

## Where to work
### 1) zemosaic_worker.py — Phase 3 master-tile alignment + stacking inputs
Find the Phase-3 code that logs:
- `MT_COVERAGE: propagate_mask={...}`

Currently:
- `propagate_mask_for_coverage = bool(altaz_cleanup_enabled_effective and _LECROPPER_AVAILABLE)`

Update so that for master-tile stacking using `winsorized_sigma_clip`, we set:
- `propagate_mask=True` when calling `align_images_in_group(...)`
(Preferably driven by `stack_reject_algo == "winsorized_sigma_clip"`; keep other paths unchanged for minimal risk.)

Then remove/bypass the block:
- “If we nanized aligned images for coverage, clean them before stacking …”
For the winsorized path, DO NOT `nan_to_num` the aligned frames prior to stacking.

Acceptance detail:
- In debug logs, for winsorized master-tile stacking we should now see:
  - `MT_COVERAGE: propagate_mask=True`

### 2) zemosaic_align_stack.py — NaN-safe winsor quantiles
In `_winsorize_block_numpy(block, winsor_limits)`:
- Replace `np.quantile(...)` with `np.nanquantile(...)` (or a safe fallback if nanquantile is unavailable).
- Ensure behavior is correct when all values are NaN for a pixel (quantiles should become NaN, and output remains NaN).

Important: Keep performance reasonable; only the quantile call needs to change.

### 3) zemosaic_align_stack.py — External CPU winsor implementation guard (if applicable)
`stack_winsorized_sigma_clip` may call an external `cpu_stack_winsorized` implementation (seestar/core/stack_methods.py) when present.
To avoid NaN regressions:
- Detect if any input frame contains non-finite values (`~isfinite`) and, if so, force the internal fallback path (or confirm the external impl is NaN-safe).
- The GPU winsor path already uses `nanquantile`; keep it as-is.

## Verification / Acceptance Criteria
### Visual / pipeline acceptance
- Master tiles no longer show pink/magenta bands after an aggressive stretch.
- Final mosaic no longer exhibits tile-edge magenta frames originating from master tiles.

### Log acceptance
- For winsorized master tile stacking:
  - `MT_COVERAGE: propagate_mask=True`
- No new “stacker ERROR” spam caused by NaNs.

### Safety / regression acceptance
- Do not modify:
  - SDS mode behavior
  - grid mode behavior
  - “I’m using master tiles” mode
  - batch size semantics
- Existing unit tests (if present) must pass.

## Suggested minimal test (add if tests folder exists)
Add a small unit test for NaN-safe winsor:
- Create 3 synthetic frames (H,W,3) where left/right borders are NaN in some frames (simulating footprint gaps).
- Run `stack_winsorized_sigma_clip`.
- Assert:
  - central region output is finite and close to expected mean
  - border region output remains 0 (or NaN depending on the stack’s final policy), but NOT colored / non-zero due to NaN mishandling
  - function does not crash.

## Deliverables
- Patch implementing the changes above.
- Short note in logs/stack_metadata indicating winsor quantile method is NaN-safe (optional).
