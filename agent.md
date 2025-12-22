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

- [x] Pass tile_weights into intertile calibration (worker → utils) by plumbing `tile_weights_for_sources` alongside `tile_sources` and forwarding it to `_compute_intertile_affine_corrections_from_sources(..., tile_weights=tile_weights_for_sources, ...)`.
- [x] Extend `_compute_intertile_affine_corrections_from_sources` signature to accept `tile_weights: list[float] | None = None` and forward it to `compute_intertile_affine_calibration(..., tile_weights=tile_weights, ...)` while keeping default `None`.
- [x] Bias anchor selection in `compute_intertile_affine_calibration` with optional `tile_weights`:
  - If `tile_weights is None`: keep existing behavior EXACTLY.
  - If provided, validate length, sanitize to float64 with non-finite/<=0 mapped to 1.0, normalize by median (>0 only), and use a mild `np.sqrt` compression for scores.
  - If any connectivity > 0: `score = connectivity * w_score`; anchor = argmax(score). Else: anchor = argmax(w_score).
  - Add a concise `[Intertile] Anchor selection biased: anchor=<idx> connectivity=<val> weight=<val> score=<val>` log.
- [x] Do NOT change the correction mapping; only the fixed anchor index changes while preserving tile order.
- [ ] Tests / Validation: rerun the problematic dataset (weights ~502,5,3) to confirm anchor selection and unchanged behavior when `tile_weights` is absent.

## Acceptance Criteria
- Backward compatible (no error when tile_weights absent).
- With tile_weights present and very unbalanced, anchor prefers the deep tile.
- No refactor; minimal diff; only the two files listed.
