# Validation Protocol — Phase5 intertile / weighting (OFF vs ON vs ON+V4)

## Goal
Provide repeatable, evidence-based comparison across 3 modes on the same dataset and same master tiles:
1. OFF (intertile/weighting disabled)
2. ON current (intertile + current weighting)
3. ON V4 (intertile + weighting V4 penalties)

## Invariants (must stay identical)
- input dataset
- master tiles set (or same seed + same phase3 settings)
- output WCS settings
- stretch/export settings for visual outputs

## Required artifacts per run
- `run_config_snapshot.json`
- `zemosaic_worker.log`
- `intertile_graph_summary.json`
- `intertile_graph_edges_raw.csv`
- `intertile_graph_edges_kept.csv`
- `intertile_affine_corrections.csv`
- `intertile_photometric_solve.csv` (new)
- `intertile_residuals.csv` (new)
- `intertile_tile_residual_summary.csv` (new)
- `tile_weights_final.csv`
- `tile_weights_v4_telemetry.csv` (if V4 enabled)
- `tile_weights_v4_summary.json` (if V4 enabled)
- `weighted_coverage_map.fits`
- `winner_map.fits`
- `winner_index.csv`

## Pass/Fail gates
- No hard run failure
- No `run_error_phase3_no_master_tiles_created`
- Intertile solve diagnostics exported when intertile enabled
- No severe seam regression on visual output

## Comparison metrics (report)
- Solve residuals: median / p95
- Rejected vs kept edges
- Winner-map concentration (max winner fraction)
- Weight distribution min / p50 / p95 / max
- Subjective seam score (A/B blind if possible)

## Non-regression modes (minimum smoke set)
- classic run
- existing master tiles mode
- SDS mode
- ZeGrid mode

Each mode: at least one short smoke run with no infrastructure regressions.
