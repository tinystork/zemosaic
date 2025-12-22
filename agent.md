# agent.md

## Goal
Ensure Phase 5 GPU "reproject & coadd" properly applies per-tile scalar weighting (tile_weight, e.g. N_FRAMES=502 vs 5 vs 3)
in addition to per-pixel alpha masks. Without this, shallow/noisy master tiles contribute almost equally and can dominate
the mosaic.

Target behavior:
- Effective weights for coadd must be: effective_weight2d = alpha_weight2d * tile_weight
- Output coverage/weight_sum must reflect weighted sums (max can reach ~sum(tile_weight) in overlaps).

No refactor: surgical patch.

## Scope (minimal)
Modify ONLY:
- zemosaic_worker.py
- (optional) zemosaic_utils.py or GPU helper module only if strictly required by current architecture

Do not change GUI, settings schema, or unrelated phases.

## Background
Logs show:
- Tile-weighting enabled (min=5 max=502) is computed.
- But Phase 5 logs input_weights sample source=alpha_weight2d with max=1.0, indicating tile_weight is not applied to GPU coadd.

CPU path already multiplies input_weights by tile_weight; GPU path must mirror it.

## Plan

### 1) Identify the Phase 5 weight construction point
In `assemble_final_mosaic_reproject_coadd()` locate the block that defines `input_weights` for each channel/tile.
It currently sets `input_weights` from per-pixel alpha (e.g. alpha_weight2d).

Also locate the scalar per-tile weight (tile_weight) used by tile-weighting mode (N_FRAMES etc).
This likely exists as `tile_weight`, `tile_weights[idx]`, or `tile_entry["tile_weight"]`.

### 2) Apply tile_weight to GPU input_weights (the core fix)
Right after `alpha_weight2d` / `input_weights` is prepared for a tile, and BEFORE calling any GPU reproject/coadd kernel:

- Compute:
  - `tw = float(tile_weight)` with sanity: if not finite or <=0 -> 1.0
- Multiply:
  - `input_weights *= tw`  (in float32)

IMPORTANT: avoid multiplying multiple times if the same numpy array instance is reused for multiple channels.
Implement a "unique array" guard:
- If input_weights is a list/tuple of arrays, de-duplicate by `id()` and multiply each unique object once.

Example logic (conceptual):
- if isinstance(input_weights, (list, tuple)):
    - seen=set()
    - for w in input_weights:
        - if id(w) in seen: continue
        - seen.add(id(w))
        - w *= tw   # in-place
  else:
    - input_weights *= tw

Ensure dtype stays float32 (cast once before in-place multiply if needed).

### 3) Keep alpha union semantics unchanged
`alpha_union` / `alpha_final` should remain a union mask (0/1 or 0/255) for transparency.
Do NOT weight alpha_union by tile_weight.

Only the coadd weights / coverage/weight_sum should become weighted.

### 4) Make GPU and CPU Phase 5 behavior consistent
If there is a CPU branch (use_gpu_phase5=False) that already multiplies by tile_weight, ensure the GPU branch does exactly the same.
If CPU branch also uses `input_weights` list reuse, apply the same de-dup guard there too (to avoid double-weighting bugs).

### 5) Update debug logging to confirm behavior
Where logs currently say:
`[Phase5] input_weights sample: channel=0 source=alpha_weight2d ... max=1.0000`
Change `source` to:
`alpha_weight2d*tile_weight`
and ensure max becomes ~tile_weight (e.g. 502) for tiles with full alpha.

Log one extra line per tile:
`[Phase5] tile_weight applied: tile=<id> tw=<float> weights_source=alpha_weight2d*tile_weight`
Do not spam per-pixel.

### 6) Two-pass compatibility
Two-pass uses a `coverage` map. If it is derived from Phase 5 weight_sum/coverage, it will automatically become weighted after this fix.
If two-pass is instead using an unweighted tile-count coverage somewhere, adjust it to use the weighted `weight_sum` produced by Phase 5.

DO NOT change two-pass algorithm beyond choosing the correct "coverage" input (weighted vs count).

### 7) Validation / Tests
Run the same dataset that exhibits the issue (3 master tiles with weights ~502,5,3), twice:
A) use_gpu_phase5=False
B) use_gpu_phase5=True

Acceptance checks:
- In logs, Phase 5 input_weights source must show `alpha_weight2d*tile_weight`.
- Max of input_weights sample should be ~tile_weight (e.g. 502) for the heavy tile.
- Final mosaic statistics (mean/median per channel) should be close between CPU and GPU runs (tolerance ~1e-3 to 1e-2 depending on float math).
- Visual: heavy tile signal is preserved; noisy tiles no longer dominate overlaps.

## Constraints
- No refactor, no API redesign.
- Keep memory footprint stable: prefer in-place multiply and avoid creating per-channel copies.
- If any failure occurs in GPU path due to dtype/contiguity, fall back to safe conversion once (float32 contiguous) BEFORE multiplying.
