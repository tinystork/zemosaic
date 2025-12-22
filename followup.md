# followup.md

## Progress
- [x] Step 1) Anchor selection
- [x] Step 2) Apply tile_weight to GPU Phase 5 weights
- [ ] Quick sanity run (CPU vs GPU)

## How to apply step 2 (practical)
You already did step 1 (anchor selection). Step 2 is independent and should be applied in Phase 5 only:

1) Find where Phase 5 prepares `alpha_weight2d` / `input_weights` (the log shows it currently labels them as `source=alpha_weight2d`).
2) Find the scalar `tile_weight` computed from N_FRAMES (or equivalent).
3) Multiply the weights used for coadd by `tile_weight`:
   - effective weights = alpha_weight2d * tile_weight
4) Make sure this multiplication happens ONCE per tile (avoid triple-multiplying if weight arrays are shared across channels).
5) Keep alpha_union unchanged.

## What to look for in logs (must-have)
- Before patch (current):
  `[Phase5] input_weights sample ... source=alpha_weight2d ... max=1.0000`
- After patch:
  `[Phase5] input_weights sample ... source=alpha_weight2d*tile_weight ... max≈502 (or your tile weight)`

Also look for a single line per tile:
`tile_weight applied ... tw=...`

## Quick sanity run
Run the same dataset twice:
- GPU Phase 5 OFF
- GPU Phase 5 ON

Compare:
- The overlap region should no longer look like "noise wins".
- Coverage/weight_sum maximum should reflect weighted sums (can reach hundreds in overlaps).

## If something goes wrong (common traps)
- Double weighting because the same numpy array instance is reused for channels:
  Fix: de-duplicate by `id()` before doing in-place multiply.
- dtype issues (uint8 alpha):
  Fix: cast once to float32 before multiplying.
- two-pass still uses unweighted "tile-count" coverage:
  Fix: pass the weighted `weight_sum` coverage from Phase 5 into two-pass.

## Minimal rollback
If GPU path becomes unstable, temporarily disable GPU Phase 5 in config to keep CPU behavior correct while debugging.
