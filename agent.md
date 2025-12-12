# Mission: Fix green/teal tint in classic mode final mosaic (RGB black-level mismatch)

## High-level goal
Restore production-ready classic-mode output colors (no persistent green/teal cast) by fixing the root cause: per-channel baseline mismatch (R min ~ 0 while G/B min > 0) in the final mosaic FITS/preview generation.

## Non-goals / must NOT change
- Do NOT modify Grid Mode behavior.
- Do NOT modify SDS mode behavior.
- Do NOT remove or alter two-pass coverage renormalization logic.
- Do NOT change clustering logic, tile selection logic, or matching logic except what is strictly required for the black-level fix.
- Do NOT add heavy dependencies.

## Context
`zemosaic_utils.save_fits_image` currently applies a *global* baseline shift for float outputs only when the global min > 0. This does not fix cases where R has true zeros but G/B are offset (common cause of green/teal tint under auto-stretch / auto-WB). The fix must be per-channel, computed on valid mosaic pixels only (alpha/coverage), then applied as a constant subtraction per channel.

## Implementation plan (minimal & safe)
1. Add a helper to compute per-channel black offsets using a validity mask:
   - Inputs: image HWC float32, optional alpha mask (uint8 0..255) and/or coverage mask (float/uint)
   - Build `valid = finite(rgb).all(axis=-1)` AND (alpha>0 if provided else coverage>0 if provided else True)
   - For each channel c, compute `p = nanpercentile(rgb[...,c][valid], p_low)` with p_low default 0.1 or 0.5
   - If p is finite and > 0: subtract it from that channel
   - Clip to >= 0 (float32)
   - Return adjusted image + info dict (offsets)

2. Apply this helper ONLY in classic mode finalization:
   - Right before saving final FITS mosaic (so the FITS histogram is sane in viewers).
   - Right before building preview PNG (so preview matches FITS and is no longer green/teal).
   - Guard with config flag default ON for classic mode, but OFF for SDS/grid.
   - If mask is empty (no valid pixels), skip safely.

3. Logging:
   - Log offsets applied: per-channel p_low percentile values and whether applied.
   - Keep logs INFO_DETAIL to avoid noise.

## Files to modify
- `zemosaic_worker.py` (apply the helper before final FITS save + before preview stretch)
- (Optional if cleaner) `zemosaic_utils.py` (place helper there), but prefer to keep minimal changes.

## Acceptance criteria
- For a dataset that previously produced a green/teal final mosaic:
  - The saved final FITS has per-channel minima aligned (close to 0 for R/G/B when measured on valid pixels).
  - The preview PNG is no longer green/teal under default stretch.
- No behavioral changes in grid mode or SDS mode.
- No crashes if alpha/coverage is missing.

## Quick manual test
Run the same classic mosaic job that currently outputs green/teal:
- Compare “before/after” FITS histogram in ASIFitsView:
  - Previously: R min ~0, G/B min >0
  - After: R/G/B minima aligned (near 0) on valid pixels
- Compare preview PNG: should match the “old good” color impression.
