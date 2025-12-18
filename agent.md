# Mission: Make lecropper/ALPHA mask act as transparency during Phase-5 reproject+coadd

## Context
- Master tiles are written with an ALPHA extension (uint8 0..255) via zemosaic_utils.save_fits_image(..., alpha_mask=alpha_mask_out).
- In Phase 5, zemosaic_worker loads each master tile + optional ALPHA and currently converts alpha to [0..1] and then *nanizes* masked pixels:
  data[zero_mask[..., None]] = np.nan
- This is NOT sufficient for `reproject_and_coadd(combine_function="mean")`: any NaN in a stack will contaminate the mean and can prevent overlap from filling masked zones.
- Goal: masked zones must be treated as "no data" so overlap fills them.

## Target behavior (Acceptance criteria)
1) If a master tile has an ALPHA extension, masked pixels (alpha <= ALPHA_OPACITY_THRESHOLD) must contribute **zero weight** to the global coadd.
2) Overlapping valid pixels from other tiles must fill those regions (no black slabs / no NaN propagation).
3) Do not rewrite master tiles on disk (only interpret ALPHA at load time).
4) Keep changes surgical: only touch the minimal code paths in zemosaic_worker.py and zemosaic_utils.py required for correctness.
5) Keep existing logs; add one concise INFO_DETAIL log when alpha masks force CPU path (if needed).

## Implementation plan (surgical)
### A) Stop using NaN-as-mask for the coadd path (Phase 5) [x]
In `zemosaic_worker.py` inside `assemble_final_mosaic_reproject_coadd`, during the loop that builds `effective_tiles`:
- Keep reading `data` and `alpha_mask_arr` as currently.
- Build a per-pixel weight map from alpha:
  - `alpha01 = clip(alpha_mask_arr, 0..1)` (already computed)
  - `valid = (alpha01 > ALPHA_OPACITY_THRESHOLD)`
  - `weight2d = valid.astype(float32)` (hard mask is fine)
- Do NOT set `data[...] = np.nan` based on alpha.
  - Instead, set masked pixels to 0 (optional but safe):
    - `data = data.copy(); data[~valid, :] = 0` (for RGB) or `data[~valid] = 0` (mono)
- Store `weight2d` on the tile entry (e.g. `tile_entry["alpha_weight2d"] = weight2d`).

### B) Pass per-pixel weights to reproject_and_coadd (CPU path) [x]
`astropy-reproject reproject_and_coadd` accepts `input_weights` per input image.
- When assembling each channel (where you build `data_list` and `wcs_list_local` for `reproject_and_coadd_wrapper`), also build:
  - `weights_list = [tile_entry["alpha_weight2d"] for each tile]`
  - If a tile has no alpha, use `None` or all-ones weights for that tile.
- Call `zemosaic_utils.reproject_and_coadd_wrapper(..., input_weights=weights_list, ...)` so CPU coadd ignores masked pixels.

### C) GPU path compatibility: correctness first [x]
The current GPU wrapper supports only scalar `tile_weights`, not per-pixel `input_weights`.
For now:
- If `use_gpu=True` and any tile has alpha_weight2d, force CPU for Phase 5 with a clear log:
  - `use_gpu_effective = False` when alpha masks are present.
  - Log: `"[Alpha] Per-pixel alpha weights require CPU coadd; forcing CPU for Phase 5"`
This is a deliberate, minimal correctness fix. (Future work: add per-pixel weights to GPU coadd.)

### D) Keep coverage consistent [x]
- For coverage output, if alpha exists, coverage should be derived from `weight2d` (already the intent):
  - `coverage_mask_entry = weight2d`
- If no alpha, keep current finite-based coverage.

## Files to edit
- `/mnt/data/zemosaic_worker.py`
- `/mnt/data/zemosaic_utils.py` (small: allow wrapper to forward `input_weights` to CPU safely if it filters kwargs)

## Notes / Constraints
- Do NOT refactor or rename major functions.
- Do NOT change lecropper.py behavior.
- Do NOT change master tile writing format.
- Keep the existing ALPHA extension logic.
- Preserve current progress callbacks and logs.

## Validation checklist [ ]
- Run a mosaic with alt-az cleanup enabled producing ALPHA.
- Confirm final mosaic has filled areas where previous run showed black/empty slabs.
- Confirm no NaN propagation in final mosaic stats (nanmin/nanmax finite on populated regions).
- Confirm Phase 5 logs show alpha weights detected and CPU forced (only if use_gpu was requested).
