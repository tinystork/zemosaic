# Follow-up — Validation & Repro steps

## Progress
- [x] Code changes for ALPHA handling and intertile masking implemented.
- [ ] Validation on the repro dataset (with updated logging checks).

## Build / Run
1) Use the same dataset / master tiles that currently reproduce the issue.
2) Run ZeMosaic with:
   - “I'm using master tiles (skip clustering_master tile creation)” ON
   - inter-tile photometric match ON (match_background / intertile enabled)
   - quality crop OFF (as in repro)

## What to check in logs
### A) Alpha inversion detection
If the ALPHA extension was inverted, you should see an INFO line similar to:
- “[Alpha] existing_master_tiles: auto-inverted alpha mask … valid_frac=… inv_valid_frac=… nz_frac=…”

If no line appears, it means the heuristic decided the ALPHA is already consistent.

### B) Intertile gains sanity
In the intertile summary and apply logs:
- Gains should not be near 0 (no more 1e-5).
- Gains should typically cluster around 1.0 and respect the configured clip behavior.

## Visual checks
- The final mosaic must not have the large flat black/purple rectangle.
- The overlap zones should show real sky signal (stars/background), not constant fill.

## Quick sanity script (optional manual check)
Open one problematic master_tile FITS and compare:
- fraction of nonzero pixels in image (nz_frac)
- fraction of alpha-valid pixels (valid_frac)
If valid_frac is “complementary” to nz_frac, ALPHA is inverted.

(Do not add new scripts to repo; use this only locally if needed.)

## Edge cases
- If a master tile is legitimately mostly empty (heavy crop), ensure the heuristic doesn’t flip incorrectly.
  - The “closest to nz_frac” scoring should handle this.
  - Keep a small margin before flipping (e.g. require inv_score + 0.05 < score).

## Rollback plan
If something goes wrong:
- Disable the new mask support in compute_intertile_affine_calibration (keep NaN-masking in worker only).
- Or keep compute_intertile mask support but disable the inversion heuristic (log-only) to isolate.

## Done criteria
- Repro dataset produces a continuous mosaic without the big rectangle.
- No regression in non-master-tiles path (normal clustering/master tile creation).
- No regression on CPU/GPU paths.
