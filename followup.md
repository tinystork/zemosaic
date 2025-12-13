# Follow-up checklist — TwoPass definitive diagnostics

[x] 1) Confirm location
- Open zemosaic_worker.py
- Identify:
  - _apply_two_pass_coverage_renorm_if_requested
  - run_second_pass_coverage_renorm

[x] 2) Implement DEBUG-only diagnostics
- Wrap all new code in:
    if logger and logger.isEnabledFor(logging.DEBUG):

[x] 3) Add global context logs
- Emit [TwoPassCfg] and [TwoPassCoverage] once at TwoPass entry.

[x] 4) Per-tile stats
- Before gain:
  - compute valid_frac
  - median / MAD RGB
  - log [TwoPassTileStats]

[x] 5) Overlap diagnostics (core)
- Downsample ref mosaic + coverage
- Reproject tile luminance to low-res grid
- Compute overlap_mask
- Log [TwoPassOverlap] with:
  overlap_frac, delta_med, abs_delta_med, delta_mad, slope, intercept, r

[x] 6) Gain application check
- After gain application:
  - recompute overlap delta
  - log [TwoPassApply] pre vs post

[x] 7) Global summary
- Sort by abs_delta_med
- Log top 5 as [TwoPassWorst]
- Log weighted global score as [TwoPassScore]

[x] 8) Sanity logs
- Emit warnings if:
  - mask shape mismatch
  - coverage rejects > X%
  - reprojection NaN fraction high

[ ] 9) Test protocol
- Run once with DEBUG enabled.
- Confirm presence of:
  [TwoPassCfg]
  [TwoPassOverlap]
  [TwoPassApply]
  [TwoPassWorst]
  [TwoPassScore]
- Run once with INFO:
  - confirm no new output.

[ ] 10) Report back
- Paste:
  - [TwoPassCfg]
  - 3–5 representative [TwoPassOverlap]
  - worst tile block
  - [TwoPassScore]
