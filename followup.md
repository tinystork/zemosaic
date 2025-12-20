# Verification & Validation Checklist

## Functional Checks

- [x] Grid mode with stack_plan.csv **without** mount column behaves exactly as before
- [x] Grid mode with stack_plan.csv **with only EQ frames** produces a single mosaic
- [x] Grid mode with stack_plan.csv **with only ALTZ frames** produces a single mosaic
- [x] Grid mode with mixed EQ / ALTZ frames:
  - [x] Two distinct runs are executed
  - [x] Two output directories are created
  - [x] No reprojection or stacking occurs between EQ and ALTZ outputs

## Logging

- [x] Logs explicitly mention detection of mount-based segregation
- [x] Logs show frame counts per mode
- [x] Fallback to legacy behavior is clearly logged when applicable

## Quality Expectations (Qualitative)

- [ ] Reduced seam artifacts in overlap regions
- [ ] Improved star roundness consistency within each mosaic
- [ ] No regression in pure EQ or pure ALTZ datasets

## Regression Safety

- [x] No changes to classic (non-grid) pipeline behavior
- [x] No changes to clustering logic
- [x] No changes to photometric normalization code
- [x] No changes to GUI

## Code Constraints

- [x] Changes limited to grid_mode-related files
- [x] No large-scale refactor
- [x] No new dependencies
