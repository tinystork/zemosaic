# Follow-up: Patch details & exact code locations

## 1) zemosaic_worker.py — Phase 5 tile preparation (assemble_final_mosaic_reproject_coadd)

### Location
Function: `assemble_final_mosaic_reproject_coadd(...)`
Loop: `for idx, (tile_path, tile_wcs) in enumerate(master_tile_fits_with_wcs_list, 1):`

### Change
Replace the current “alpha -> NaN” application:

- Remove / disable:
  - `zero_mask = alpha_mask_arr <= ALPHA_OPACITY_THRESHOLD`
  - `data[zero_mask[..., None]] = np.nan`

- Add:
  - `valid2d = (alpha_mask_arr > ALPHA_OPACITY_THRESHOLD)` after alpha normalization/clip.
  - `alpha_weight2d = valid2d.astype(np.float32)`
  - Optional safety: set masked pixels to 0 so interpolation never sees garbage:
    - if `data.ndim == 3`: `data = data.copy(); data[~valid2d, :] = 0.0`
    - else: `data = data.copy(); data[~valid2d] = 0.0`
  - Store on tile_entry:
    - `tile_entry["alpha_weight2d"] = alpha_weight2d`

Coverage:
- If alpha exists: `coverage_mask_entry = alpha_weight2d`
- Else: keep existing finite-based coverage.

## 2) zemosaic_worker.py — Phase 5 per-channel coadd call: pass input_weights

### Location
Still inside `assemble_final_mosaic_reproject_coadd`, later when you build per-channel `data_list` and `wcs_list_local` and call:
- `zemosaic_utils.reproject_and_coadd_wrapper(...)`

### Change
Build a `weights_list` aligned with `data_list`:
- If tile has `alpha_weight2d`, append it
- Else append an all-ones array matching the plane shape (H,W)
  - `np.ones_like(data_plane, dtype=np.float32)`

Pass to wrapper as:
- `input_weights=weights_list`

Important: weights are 2D per plane, which matches reproject expectations.

## 3) GPU forcing (correctness-first)

### Location
At the start of Phase 5 assembly (before channel loop) after `effective_tiles` is built.

### Change
Detect alpha weights:
- `alpha_present = any(("alpha_weight2d" in t and t["alpha_weight2d"] is not None) for t in effective_tiles)`

If `use_gpu` is True and `alpha_present` is True:
- set `use_gpu = False` for this function scope (or use a local `use_gpu_effective`)
- emit a single log/callback:
  - logger.info or pcb INFO_DETAIL:
    - `"[Alpha] Per-pixel alpha weights require CPU coadd; forcing CPU for Phase 5"`

This avoids silent wrong mosaics.

## 4) zemosaic_utils.py — ensure wrapper forwards input_weights to CPU

### Location
Function: `_reproject_and_coadd_wrapper_impl(...)`

### Check
It currently filters some GPU-only kwargs into `cpu_kwargs`, but `input_weights` is not filtered out — good.
If you see any filtering that drops `input_weights`, ensure it is preserved.

No algorithm change required.

## 5) Quick sanity tests (no new unit framework)
- Add a tiny debug log in Phase 5:
  - for first tile with alpha, log:
    - alpha min/max, valid fraction, weight shape vs data shape
- Run one dataset where alt-az cleanup produces strong masks.
Expected:
- final mosaic no longer shows black/empty slabs in overlapping regions.
- coverage behaves (masked pixels => low/zero coverage unless filled by other tiles).

## Deliverables
- One commit with surgical changes.
- No refactor, no signature changes.
- Logs kept; add 1–2 INFO_DETAIL lines max.
