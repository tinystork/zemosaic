# Mission: TwoPass diagnostics logs (per-tile stats + delta map) — NO pipeline change

[x] Goal
- Add objective diagnostics for Two-Pass Coverage Renorm (Phase 5).
- We want:
  1) Per-tile mean/median (RGB) BEFORE applying the computed gain, and AFTER applying the gain.
  2) A lightweight "inter-tile delta map" (optional) to visualize seams as residual offsets vs pipeline bug.
- Absolutely no change to algorithms, parameters, ordering, or outputs (unless explicitly enabled via DEBUG-only diagnostics file output).

[x] Scope (files)
- zemosaic_worker.py

[x] Constraints
- Do NOT modify any math in the renorm itself (no change to gains, clipping, sigma, coverage, reprojection).
- Only add logs and optional debug artifact saved under DEBUG-level (or behind a new explicit debug flag default False).
- Keep runtime overhead minimal: per-tile stats are cheap; delta map must be downsampled and optional.

[x] Where to instrument
- run_second_pass_coverage_renorm(...) in zemosaic_worker.py.
  We already log global things (start, prepared tiles, gains min/max, etc.).
  Add *tile-by-tile* stats right after gains are computed and right after applying gain to each tile (before reprojection starts).

[x] Deliverables
A) Per-tile stats logs (always under logger.isEnabledFor(DEBUG))
- For each tile i:
  - tile index and/or tile_id if available
  - gain scalar used (the one applied in second pass)
  - RGB min/mean/median computed on finite pixels only
  - also log valid pixel fraction (finite mask fraction)
- Emit two lines per tile:
  - [TwoPassTile] pre_gain ...
  - [TwoPassTile] post_gain ...

[x] Implementation details (safe)
- Add a small helper in zemosaic_worker.py:
  - _two_pass_tile_rgb_stats(arr: np.ndarray) -> dict with min/mean/median per channel + valid_fraction
  - Must handle arr shapes: HWC (..,3) and HW (mono); for mono treat it as one channel and still log.
  - Use np.isfinite mask; ignore NaNs/inf.
- Then inside run_second_pass_coverage_renorm:
  - After `gains` array computed (currently logged min/max) and before reprojection:
    - For each tile arr:
      - compute stats_pre = helper(arr)
      - apply gain (existing code path) to produce arr_scaled (or in-place, as currently)
      - compute stats_post = helper(arr_scaled)
      - log both with gain

B) Optional inter-tile delta map (DEBUG-only + downsample)
- Purpose: show residual seams as “tile vs first-pass mosaic” mismatch.
- Approach (cheap):
  - Use the first-pass mosaic (final_mosaic_data_p1 / mosaic_p1) and final_wcs_p1 already available in the caller path.
  - In run_second_pass_coverage_renorm we have tiles + tiles_wcs + final_wcs_p1 + shape_out.
  - After gains are applied, reproject each tile (single channel or luminance) *coarsely* to a reduced grid:
    - downsample factor = 8 (configurable constant)
    - target shape = (shape_out[0]//ds, shape_out[1]//ds)
  - Build:
    - delta_sum, delta_count arrays (float32)
    - For each tile: compute abs(tile_proj - mosaic_p1_proj) where both finite; accumulate.
  - At end: delta_map = delta_sum / max(delta_count,1)
  - Save as a debug artifact in runtime temp dir (e.g., zemosaic_runtime/twopass_delta_map.npy)
  - Only do this if logger DEBUG enabled (or a new flag two_pass_debug_delta_map=True default False).
- IMPORTANT: This must not affect outputs; saving file only.

[ ] Acceptance tests
- Run a dataset that triggers TwoPass (Phase 5) and confirm in logs:
  - Existing TwoPass logs still appear unchanged.
  - New logs show per-tile pre/post stats and gains.
  - If delta map enabled: a .npy file is created and a log prints its path.
- Confirm final mosaic output is byte-identical vs before when delta map is disabled (only logs added).

Do not touch
- Any other phases, GUI, WCS, cropping, masking, RGB equalization logic.