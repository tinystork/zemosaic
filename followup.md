# Follow-up: How to verify the lecropper propagation fix

## 1) Quick log greps (Windows PowerShell)
From the folder containing the ZeMosaic log:

### Confirm option is actually enabled in the run
Select-String -Path .\zemosaic.log -Pattern "lecropper_enabled|MT_PIPELINE"

Expected:
- One run-level line showing flags (enabled/disabled)
- Many per-tile lines with `MT_PIPELINE: lecropper_applied=True` when enabled

### Confirm masked output is used even when mask2d is present
Select-String -Path .\zemosaic.log -Pattern "masked_used=True"

Expected:
- If alt-az cleanup is active and mask2d is produced, you should see `masked_used=True mask2d_used=True`.

## 2) Visual sanity check (single tile)
Pick one problematic Master Tile that previously had obvious alt-az edge artifacts.

Run twice on same dataset:
- Run A: lecropper OFF
- Run B: lecropper ON

Compare:
- The saved master tiles (or intermediate outputs)
- The final mosaic: edge artifacts should be reduced in Run B.

## 3) Guardrails
- Confirm there is no change in behavior for:
  - batch size = 0
  - batch size > 1
- Confirm lecropper remains standalone (no new ZeMosaic imports in lecropper.py)

## 4) If something still doesn’t propagate
Add one extra debug line (only if necessary):
- right before saving a master tile, log min/median/max for RGB channels (already there are helper stats functions in worker).
- Use it to prove the array changed when lecropper is ON.

## 5) Commit message suggestion
"Fix: propagate lecropper masked output to master tiles and ensure GUI flag enables MT pipeline"
