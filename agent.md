# Mission: Fix intertile photometric pruning that breaks connectivity (causes patchwork seams)

## Context
User reports strong inter-tile normalization discontinuities ("patchwork"/"plaques") on large runs (e.g., 4700+ raws).
Worker log confirms TwoPass executed successfully with small gain range (~0.978–1.012), so TwoPass is NOT the root cause.

The same log shows intertile affine calibration pruning produces a disconnected overlap graph:

- raw_pairs=6537 (num_tiles=197)
- after_topK=912 (MAX_NEIGHBORS_PER_TILE=8)
- connectivity WARN: components=74, bridges_added=0

This makes the global affine solve under-constrained across components, yielding inconsistent gains/offsets and visible seams.

## Goal
Guarantee that the overlap graph used by `compute_intertile_affine_calibration()` remains connected (for all active tiles),
OR fallback safely (no pruning) when connectivity cannot be guaranteed.

This should eliminate the “patchwork” normalization artifact without changing TwoPass behavior.

## Strict Scope / Guardrails
- Primary file: `zemosaic_utils.py` only (inside `compute_intertile_affine_calibration()` pruning block).
- Optional: minimal extra logging in `zemosaic_worker.py` ONLY if needed to surface diagnostics (avoid functional changes).
- Do NOT modify:
  - TwoPass coverage renorm logic
  - GPU safety / VRAM chunking logic
  - stacking algorithms, WCS solve, cropping, borrowing, EQ/ALT-AZ logic
  - existing default parameter values exposed to user
- No “clever auto behavior” beyond connectivity guarantees.

## Implementation Plan
### A) Refactor pruning into a connectivity-safe step
In `zemosaic_utils.compute_intertile_affine_calibration()`:
1. [x] Keep existing topK pruning idea, but enforce:
   - compute connected components among ACTIVE tiles
   - after pruning, if components_active > 1:
     - attempt to add bridge edges from the full raw overlap list (sorted by weight desc)
     - keep adding until components_active == 1 OR no progress

2. [x] Fix ACTIVE tile detection robustness:
   - Do not rely solely on `weight > 0` filtering to mark active tiles.
   - Define active tiles as: any tile that appears in at least one overlap pair (raw graph degree > 0).
   - If an overlap entry is missing `weight`, compute it from bbox area as fallback.

3. [x] If bridging still fails (components_active > 1):
   - Fallback to **NO PRUNING** (use the raw overlap list).
   - Log an explicit warning:
     - `"[Intertile] PRUNE_FALLBACK_NO_PRUNING: disconnected after prune+bridge (components_active=...)"`

### B) Diagnostics (must-have)
- [x] Add a compact INFO log line after pruning decision:
  - raw_pairs, pruned_pairs, max_neighbors, active_tiles, components_active, bridges_added, fallback_used

Example:
`[Intertile] Pair pruning summary: raw=6537 pruned=912 K=8 active=197 components=1 bridges=73 fallback=no`

- [x] If fallback happens:
  - `[Intertile] PRUNE_FALLBACK_NO_PRUNING: raw=6537 components_active=...`

### C) Safety behavior
- [x] Never run `solve_global_affine()` on a graph with components_active > 1 unless fallback to raw overlaps was applied.
- [x] Ensure determinism: bridging should pick edges in a deterministic order (weight desc + stable tie-breaker on (i,j)).

## Acceptance Criteria
1. On the same run pattern as the provided log (197 tiles, raw_pairs~6537):
   - The log must NOT show `Remaining components: 74. Bridges added: 0`
   - It must show either:
     - components_active == 1 after bridging, OR
     - fallback_no_pruning triggered
2. Output: visible inter-tile “patchwork” seams significantly reduced on large mosaics.
3. No regressions on small datasets (pruning may still occur, but connectivity must remain guaranteed).
