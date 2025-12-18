# Mission (surgical / no refactor)
Fix alpha mask propagation during Phase 5 reproject & coadd: ALPHA extension exists (uint8 0/255) but is silently ignored because code reads it with np.asarray(... dtype=float32, copy=False), which fails when dtype conversion would require a copy.

## Scope
Only `zemosaic_worker.py`. No refactor, no behavior changes beyond making ALPHA actually load.

## Changes
### 1) Phase 5 coadd loader
File: `zemosaic_worker.py`
Function: `assemble_final_mosaic_reproject_coadd`
Find block:
```py
if "ALPHA" in hdul and hdul["ALPHA"].data is not None:
    alpha_mask_arr = np.asarray(hdul["ALPHA"].data, dtype=np.float32, copy=False)


Replace the np.asarray(...) line with:

alpha_mask_arr = np.asarray(hdul["ALPHA"].data)


Keep the existing normalization logic below (it already converts/scales).

2) Two-pass loader

File: zemosaic_worker.py
Function: _load_master_tiles_for_two_pass
Replace:

alpha_arr = np.asarray(hdul["ALPHA"].data, dtype=np.float32, copy=False)


with:

alpha_arr = np.asarray(hdul["ALPHA"].data)

Expected outcome

Phase 5 detects per-pixel alpha weights (alpha_weights_present=True).

If Phase 5 was GPU, it logs that it forces CPU due to alpha weights.

assemble_reproject_coadd: input_weights sample ... weight_source=alpha_weight2d

Final mosaic no longer shows black tile frames caused by masked pixels participating in coadd.

