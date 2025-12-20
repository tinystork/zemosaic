# Verification & Validation Checklist

## Functional Checks

- [ ] Grid mode with stack_plan.csv **without** mount column behaves exactly as before
- [ ] Grid mode with stack_plan.csv **with only EQ frames** produces a single mosaic
- [ ] Grid mode with stack_plan.csv **with only ALTZ frames** produces a single mosaic
- [ ] Grid mode with mixed EQ / ALTZ frames:
  - [ ] Two distinct runs are executed
  - [ ] Two output directories are created
  - [ ] No reprojection or stacking occurs between EQ and ALTZ outputs

## Logging

- [ ] Logs explicitly mention detection of mount-based segregation
- [ ] Logs show frame counts per mode
- [ ] Fallback to legacy behavior is clearly logged when applicable

## Quality Expectations (Qualitative)

- [ ] Reduced seam artifacts in overlap regions
- [ ] Improved star roundness consistency within each mosaic
- [ ] No regression in pure EQ or pure ALTZ datasets

## Regression Safety

- [ ] No changes to classic (non-grid) pipeline behavior
- [ ] No changes to clustering logic
- [ ] No changes to photometric normalization code
- [ ] No changes to GUI

## Code Constraints

- [ ] Changes limited to grid_mode-related files
- [ ] No large-scale refactor
- [ ] No new dependencies
