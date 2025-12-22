# agent.md

## Goal
Fix the "deep master tile gets crushed by noisy tiles" scenario when inter-tile photometric calibration (gain/offset) selects an anchor tile purely based on overlap connectivity. When tile weights are highly unbalanced (e.g. 502 vs 5 vs 3), the anchor must prefer the highest-quality/deepest tile, otherwise the calibration drifts toward noisy tiles and the final mosaic looks dominated by noise.

We keep this as a surgical patch (no refactor). We only add an OPTIONAL `tile_weights` plumbing from worker → utils and use it ONLY to bias the anchor choice (and fallback when connectivity is flat).

## Scope
Modify ONLY:
- `/mnt/data/zemosaic_worker.py`
- `/mnt/data/zemosaic_utils.py`

No other files. No behavior change when `tile_weights` is not provided.

## Background (what’s wrong)
`zemosaic_utils.compute_intertile_affine_calibration()` currently selects:
`anchor = argmax(connectivity)`
where `connectivity` is computed from overlap pairs. With “existing master tiles” and weird coverage geometries, a noisy tile can have high connectivity and becomes anchor, forcing the deep tile to match it (perceived as “écrasement”).

## Plan

### 1) Pass tile_weights into intertile calibration (worker → utils)
In `zemosaic_worker.py`, inside `assemble_final_mosaic_reproject_coadd`, when building `tile_sources` for `_compute_intertile_affine_corrections_from_sources`, also build a list:
- `tile_weights_for_sources = [float(entry.get("tile_weight", 1.0)) for entry in effective_tiles]`
This list must remain aligned with `tile_sources` order (same loop).

Update the call:
`_compute_intertile_affine_corrections_from_sources(..., tile_weights=tile_weights_for_sources, ...)`

### 2) Extend `_compute_intertile_affine_corrections_from_sources` signature
In `zemosaic_worker.py`, update:
`def _compute_intertile_affine_corrections_from_sources(...):`
to accept:
`tile_weights: list[float] | None = None`

Then pass it through to:
`zemosaic_utils.compute_intertile_affine_calibration(..., tile_weights=tile_weights, ...)`

Keep default `None` to preserve all old call sites.

### 3) Bias anchor selection in `compute_intertile_affine_calibration`
In `zemosaic_utils.py`, update signature:
`def compute_intertile_affine_calibration(..., tile_weights=None, ...)`

Implementation rules:
- If `tile_weights is None`: keep existing behavior EXACTLY.
- If provided:
  - Validate length == num_tiles; otherwise ignore (log a warning and proceed without weights).
  - Convert to float64 array, replace non-finite or <=0 with 1.0.
  - Normalize gently to avoid insane dominance:
    - `med = median(weights[weights>0])` (fallback 1.0)
    - `w_norm = weights / med`
    - Optional mild compression: `w_score = np.sqrt(w_norm)` (preferred) OR `np.log1p(w_norm)`; choose sqrt for simplicity.
  - At anchor selection line (currently around `anchor = int(np.argmax(connectivity)) if np.any(connectivity > 0) else 0`):
    - If any connectivity > 0:
      - `score = connectivity * w_score`
      - anchor = argmax(score)
    - Else (no overlaps):
      - anchor = argmax(w_score)

Add an INFO/DEBUG log (short) e.g.:
`[Intertile] Anchor selection biased: anchor=<idx> connectivity=<val> weight=<val> score=<val>`
(Do not spam; one line.)

### 4) Do NOT change the correction mapping
Do NOT reorder tiles. We only change which index is fixed to (gain=1, offset=0) in the global solve. The returned correction list/dict mapping remains aligned with the original tile order.

### 5) Tests / Validation
Run the exact problematic dataset (3 master tiles; weights ~502,5,3). Verify in logs:
- Intertile anchor chosen corresponds to the tile with weight ~502 (even if its connectivity is lower).
- `apply_photometric` no longer drags the deep tile toward the shallow tiles.
- Visual: deep tile’s signal is preserved; shallow tiles do not “wash out” the mosaic.

Also verify a normal run where `tile_weights` not passed behaves unchanged.

## Acceptance Criteria
- Backward compatible (no error when tile_weights absent).
- With tile_weights present and very unbalanced, anchor prefers the deep tile.
- No refactor; minimal diff; only the two files listed.
