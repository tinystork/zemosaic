# Mission: Fix NaN mask propagation during Phase5 reproject+coadd (GPU/CPU parity) — NO REFACTOR

## Context
Final mosaic shows nested dark/black rectangular frames: masked/invalid regions from master tiles are being treated as valid zeros after reproject/coadd.
Master tiles contain NaNs (or coverage masks) but Phase5 currently builds per-tile input weights incorrectly (often all-ones), so reproject/coadd blends zeros instead of ignoring masked pixels.

This must be fixed in a minimal, surgical way.

## Scope (STRICT)
- Modify ONLY: `zemosaic_worker.py`
- Focus area: `assemble_final_mosaic_reproject_coadd()` Phase5 channel loop building `input_weights_list` and invoking `reproject_and_coadd_wrapper`.
- No refactors, no renames, no moving code blocks, no formatting sweeps.

## Primary goals
1) Ensure that for each tile, per-pixel invalid regions (NaNs / masked) are converted into **input weights = 0** so they do not contribute to the coadd.
2) Make GPU and CPU behave consistently: masked pixels should not appear as dark rectangles.
3) Keep existing behavior for true ALPHA tiles; do not break the “alpha_weights_present => force CPU” logic.

## Key observations (current bug)
Inside `assemble_final_mosaic_reproject_coadd()`:
- The channel loop currently builds `input_weights_list` from `alpha_weight2d` only; otherwise it uses `ones_like()`.
- But NaN masking (from lecropper) is stored in `entry["coverage_mask"]` (float32 0/1) and/or the data plane contains NaNs.
- Therefore input_weights become ones → masked pixels are treated as valid → reproject/coadd can fill them as zeros → nested frames artifact.
Also: `_invoke_reproject()` creates `invoke_kwargs` but mistakenly calls wrapper with `**local_kwargs` (so `tile_weights` is never passed). Fix that too (tiny).

## Implementation plan (surgical)
### A) Use `coverage_mask` as input weights when present
In the Phase5 channel loop where we build:
- `data_list`
- `wcs_list`
- `input_weights_list`

Change the fallback branch:
- If `entry.get("alpha_weight2d")` exists: keep it (unchanged).
- Else if `entry.get("coverage_mask")` exists and matches the data plane shape `(H,W)`: use that as the weight map.
- Else: fallback to `ones_like(data_plane)`.

Additionally:
- Ensure weight map is float32 and clipped to [0,1].
- If shape mismatch, keep safe fallback to ones_like (no crash).

### B) Fix `_invoke_reproject` kwargs bug
Current code:
```py
invoke_kwargs = dict(local_kwargs)
if tile_weighting_applied ...:
    invoke_kwargs["tile_weights"] = weights_for_entries
return reproject_and_coadd_wrapper(..., **local_kwargs)
````

This ignores `invoke_kwargs`.
Change wrapper call to `**invoke_kwargs`.

### C) Add MICRO debug logs (no spam)

Add one-time per channel (or only channel 0) debug payload:

* Whether coverage_mask was used
* min/max of one sample weight map
* fraction of zeros (approx) for one sample tile
  Keep it guarded by `logger.isEnabledFor(logging.DEBUG)` or a boolean `debug_logged`.

Do NOT add heavy loops over all tiles; sample at most 1–2 tiles.

## Acceptance criteria

* Running the same dataset as the provided logs produces a final mosaic without nested dark rectangle frames.
* CPU and GPU both ignore masked regions (masked areas remain transparent/absent; no black borders).
* No regression in cases with real `alpha_weight2d` tiles; those still force CPU for Phase5 as before.

## Deliverables

* A single git-ready patch to `zemosaic_worker.py`
* A short commit message suggestion

## Suggested commit message

"Phase5: use coverage_mask as input_weights + pass tile_weights correctly to reproject wrapper"


