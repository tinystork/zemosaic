
# 🔁 Follow-up checklist: Grid/Survey photometry & blending

Use this checklist to validate the implementation.

## 1. Masking invalid data

- [x] A helper exists to compute a valid pixel mask (finite & > eps).
- [x] Invalid pixels are excluded from:
      - tile stats,
      - overlap regression,
      - blending weights.
- [x] Regions with `weight_sum == 0` in the final mosaic are marked invalid
      and do not participate in any statistic.

## 2. Overlap graph & regression

- [x] An overlap graph between tiles is built based on their global bbox.
- [x] For each overlapping tile pair:
      - valid overlapping pixels are extracted,
      - a linear relation `B ≈ a*A + b` is estimated.
- [x] A global set of `(gain, offset)` per tile is solved, using the
      pairwise relations.
- [x] If regression fails (too few pixels, all invalid, etc.), a safe
      fallback `(gain=1, offset=0)` is applied to that tile.
- [x] Tiles are photometrically more consistent across overlaps after
      applying the gains/offsets.

## 3. SWarp-like background matching

- [x] A robust background estimator is implemented per tile
      (e.g. sigma-clipped median).
- [x] A global target background `B_target` is computed.
- [x] Tiles are shifted so that their backgrounds are harmonized towards
      `B_target`.
- [x] Visually, large-scale background differences between tiles are
      significantly reduced.

## 4. Multi-resolution blending

- [x] Gaussian/Laplacian pyramid utilities exist and are tested on small
      dummy arrays.
- [x] For overlapping tiles, smooth blending masks (w_A, w_B) are used,
      with w_A + w_B = 1 and masks respecting invalid pixels.
- [x] Pyramidal blending is applied at each level:
      `L_blend_k = L_A_k * w_A_k + L_B_k * w_B_k`.
- [x] Reconstructed blended overlaps show no hard seams.
- [x] For small overlaps (if optimized), a simpler Gaussian feathering is
      used as a fallback.

## 5. Integration into assemble_tiles

- [x] `assemble_tiles` (or its new equivalent) uses:
      - valid masks,
      - global gain/offset corrections,
      - background matching,
      - multi-resolution blending.
- [x] No broadcasting error or shape mismatch is raised.
- [x] Tile placement respects clamped bbox and source offsets.
- [x] The final mosaic is geometrically correct and **photometrically
      continuous** (no visible hard tile borders except where data is
      truly missing).

## 6. Logs & diagnostics

- [x] `[GRID]` logs report:
      - number of tiles,
      - number of overlaps,
      - success/failure of regression & background matching,
      - how many overlaps used pyramidal blending.
- [x] Any failure in advanced photometry/blending falls back to a simpler
      but safe behavior, with a clear `[GRID]` warning.

## 7. Regression safety

- [x] Classic pipeline (no stack_plan.csv) behaves exactly as before.
- [x] Non-Grid modes are unaffected.
- [x] No changes were made to clustering logic, classic master tiles, or
      Phase 5 reproject+coadd.
