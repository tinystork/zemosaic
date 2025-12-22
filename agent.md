# Agent Task — Best-effort inter-tile photometric anchoring (existing master tiles)

## Status
- [x] Implement best-effort inter-tile photometric anchoring when using existing master tiles
- [x] Add Qt GUI tooltip informing users about photometry limitations with existing master tiles

## Context
When the user enables **"I'm using master tiles (skip clustering_master tile creation)"**, ZeMosaic receives
pre-built master tiles that already embed photometric decisions (stacking, cropping, masking).
High-quality global photometric normalization is no longer possible.

The goal of this task is to implement a **best-effort inter-tile photometric anchoring**
that improves visual consistency **without pretending to recover lost information**.

This must be:
- robust
- deterministic
- simple
- non-invasive
- explicitly limited in scope

## Scope (STRICT)
- Only applies when `use_existing_master_tiles == True`
- Only affects the **inter-tile normalization phase**
- No refactor
- No change to default behavior when master tiles are internally generated

## Technical Approach

### Step 1 — Select anchor tile
Choose a single master tile as photometric reference:
- Prefer tile with largest valid coverage area
- Fallback: first valid tile

### Step 2 — Compute overlap statistics
For each non-anchor master tile:
- Identify overlap region with anchor (using coverage + alpha)
- Compute **robust per-channel statistics** on overlap:
  - median (preferred) or trimmed mean
  - ignore NaN / zero / masked pixels

### Step 3 — Compute gain
For each RGB channel:
- gain = median(anchor_overlap) / median(tile_overlap)
- Clamp gain to a reasonable range (e.g. [0.5, 2.0])

### Step 4 — Apply gain
- Apply gain multiplicatively to the tile data
- Do NOT modify:
  - alpha
  - coverage
  - NaN regions

### Step 5 — Safety guards
- If overlap pixel count < threshold → skip correction
- If stats unstable or non-finite → skip correction
- Log when correction is skipped

## Explicit Non-goals
- No global least-squares solve
- No spatial smoothing
- No extrapolation to non-overlap regions
- No attempt to reach PixInsight-level photometry

## Files Likely Involved
- `zemosaic_worker.py`
- (optional helper function colocated, no new module)

## Success Criteria
- Visible reduction of inter-tile brightness bands
- No regression in standard (non-existing-master-tiles) pipeline
- Deterministic output
- Clear logs when best-effort correction is applied or skipped
