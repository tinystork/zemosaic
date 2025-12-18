# Follow-up: How to validate the Phase5 mask propagation fix

## 1) Quick sanity checks (before running a full mosaic)
- Confirm the patch touched ONLY `zemosaic_worker.py`
  - `git status`
  - `git diff`

- Optional: run a quick search in the edited area to ensure:
  - `coverage_mask` is used to populate `input_weights_list`
  - `_invoke_reproject()` passes `**invoke_kwargs`

## 2) Run the same reproduction dataset
Use the same command/config you used when producing:
- the “nested frames” final mosaic screenshot
- the `zemosaic_worker.log`

Run with GPU enabled (since the issue was clearly visible there).

## 3) What to look for in logs
In Phase 5:
- You should NOT see a fallback that turns weights into all-ones silently.
- If DEBUG enabled, you should see one micro log per channel (or channel 0) like:
  - "input_weights source=coverage_mask" for at least one tile
  - a non-trivial fraction of zeros in the weight map sample

## 4) Visual acceptance
- Final mosaic should resemble the expected “clean” reference:
  - No nested dark/black rectangles aligned to tile bounding boxes
  - Masked regions behave as transparent / non-contributing

## 5) Regression checks (important)
- Test a dataset that includes true ALPHA extensions:
  - Ensure Phase5 still forces CPU when `alpha_weight2d` is present (as before).
- Test a dataset without NaNs/masks:
  - Mosaic should remain unchanged.

## 6) If it still fails
Collect:
- the new Phase5 log section (Phase5 started → finished)
- whether weights were reported as coming from coverage_mask
- one output coverage map FITS (if generated)
Then we’ll decide whether the GPU helper needs a footprint*weights multiplication (in gpu_reproject impl), but do NOT change that unless proven necessary.
````
