# ZeMosaic — Fix “black/purple square” in existing master-tiles mode (Option B)

## Context / Symptom
When running with:
- GUI option: “I'm using master tiles (skip clustering_master tile creation)”
- inter-tile photometric match enabled (match_background / intertile)

The final mosaic shows a huge flat black/purple rectangle even though coverage indicates tiles overlap.

Log hints:
- intertile affine gains can be absurdly small (e.g. ~3e-5) -> suggests the affine fit is polluted by masked pixels stored as finite zeros.
- In existing-master-tiles mode, ALPHA extension is used to create weights and/or to zero masked pixels. If ALPHA semantics are inverted (0=valid, 1=invalid), weights become inverted and large regions become constant/flat in the final mosaic.

## Goal (Option B)
Fix the issue ONLY for the “existing master tiles” scenario by:
1) Auto-detecting inverted ALPHA masks when loading master tiles, and invert them if needed.
2) Ensuring intertile photometric calibration ignores masked pixels:
   - Apply ALPHA mask by setting invalid pixels to NaN for preview/affine input.
   - Extend compute_intertile_affine_calibration to accept optional per-tile masks and use them when sampling overlap.

## Constraints
- NO refactor, minimal surgical changes.
- Touch only:
  - zemosaic_worker.py
  - zemosaic_utils.py
- Keep existing default behaviors and performance characteristics as much as possible.
- Add INFO/DEBUG logs to confirm inversion detection, but don’t spam.

## Implementation Tasks

### A) zemosaic_worker.py — Auto-fix inverted ALPHA when loading existing master tiles
Where master tiles are loaded and alpha_weight2d is built (existing_master_tiles_mode path):
- After normalizing alpha to float32 [0..1] but BEFORE building valid2d:
  - Compute `nz2d` from tile data (any channel abs>eps and finite).
  - Compute:
    - valid_frac = mean(alpha > ALPHA_OPACITY_THRESHOLD)
    - inv_valid_frac = mean((1-alpha) > ALPHA_OPACITY_THRESHOLD)
    - nz_frac = mean(nz2d)
  - Choose orientation (alpha or 1-alpha) whose valid_frac is closest to nz_frac.
  - Only flip if it’s clearly better (e.g. inv_score + margin < score).
- Log a single INFO line when a flip occurs:
  - “[Alpha] existing_master_tiles: auto-inverted alpha mask …”

Make sure:
- alpha_mask_arr used to create alpha_weight2d / coverage_mask_entry uses the corrected orientation.
- valid2d uses corrected alpha.

### B) zemosaic_worker.py — Intertile: load/apply ALPHA as NaN for previews + pass mask to utils
In `_compute_intertile_affine_corrections_from_sources`:
- When loading from FITS path:
  - Also try to read an ALPHA extension if present (name “ALPHA”).
  - Normalize to float [0..1] like elsewhere.
  - Apply the same auto-inversion heuristic (using nz2d of tile_arr) if needed.
  - Build `valid2d = alpha > ALPHA_OPACITY_THRESHOLD` and set `tile_arr[~valid2d] = np.nan`
- Build tile_pairs as (tile_arr, wcs, mask2d_float) instead of (tile_arr, wcs) when mask exists.
  - mask2d_float: valid2d.astype(np.float32) or alpha itself clipped to [0..1].
- Keep backward compatibility: if no alpha, keep current behavior.

### C) zemosaic_utils.py — compute_intertile_affine_calibration: accept optional masks
Modify `compute_intertile_affine_calibration(tile_data_with_wcs, ...)` to support:
- entries of length 2: (data, wcs)  -> current behavior
- entries of length 3: (data, wcs, mask2d) -> new behavior

When computing overlap samples:
- Reproject mask_i and mask_j to target (same as data)
- Valid pixels require:
  - finite(data_i) & finite(data_j)
  - mask_i_reproj > 0.5 (or > 0.0 if mask is soft)
  - mask_j_reproj > 0.5
- Then proceed with sky percentile selection and robust affine fit on those valid pixels.

### D) Safety rails (lightweight)
- If after masking the overlap has too few valid pixels, skip that pair (already handled by existing min-samples checks).
- No changes to other phases.

## Expected Outcome
- Existing master tiles mode no longer produces the big flat rectangle.
- Intertile affine gains stay sane (close to 1, within recenter clip), no absurd tiny gains.
- Coverage and mosaic visually consistent.

## Files changed
- zemosaic_worker.py
- zemosaic_utils.py
