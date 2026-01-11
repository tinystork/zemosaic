# Mission: Fix inter-tile ghosts caused by missing WCS shift when AUTO_PAD canvas is used

## Context
In Classic mode mosaics, some runs show systematic tile misregistration ("ghost tiles").
The master tiles are correctly stacked, but in Phase 5 reprojection all tiles appear offset.

This correlates with align_images_in_group() triggering AUTO_PAD (padding all frames to a square diag canvas).
AUTO_PAD changes the pixel origin and image dimensions, but the master tile WCS (copied from the reference raw WCS)
is NOT shifted to account for the padding offset. Subsequent crops (quality crop in Phase 3, and optional Apply master tile crop in Phase 5)
adjust CRPIX, but they operate on an already incorrect WCS.

## Goal
Ensure that whenever AUTO_PAD was applied during intra-tile alignment, the master tile WCS is updated with the padding offset
before any downstream crop (quality crop, MT_EDGE_TRIM, Phase 5 Apply master tile crop).

Result: Phase 5 reprojection produces correctly aligned tiles (no ghosts).

## Scope (keep it surgical)
Primary file: zemosaic_worker.py
Optional tiny improvement: zemosaic_utils.py (crop_image_and_wcs logging when CRPIX clamp triggers)

DO NOT modify SDS mode or Grid mode behavior.
DO NOT change stacking math, rejection algorithms, or GPU/CPU parity.
DO NOT change existing "batch size = 0" / "batch size > 1" behavior.

## Where the bug happens
- zemosaic_align_stack.align_images_in_group() can AUTO_PAD frames to diag x diag via _pad_center_nan()
- zemosaic_worker.py detects AUTO_PAD (aligned image shape > orig shape) and skips MT_EDGE_TRIM
- BUT zemosaic_worker.py does not shift wcs_for_master_tile CRPIX by the pad offsets.
- Later: Phase 3 quality crop shifts CRPIX by -x0/-y0, producing a WCS still missing +pad offsets.
- Later: Phase 5 reprojection uses that WCS and tiles land in the wrong place.

## Acceptance criteria
1) [x] When AUTO_PAD is used, logs clearly show a new info line:
   "MT_WCS_PAD: applied dx=..., dy=..., orig_hw=..., padded_hw=..., crpix_old=..., crpix_new=..."
2) [x] Master tile FITS headers include pad metadata (e.g. ZMT_PADX / ZMT_PADY) for debugging.
3) Phase 5 reprojection yields properly aligned tiles (no ghosting) on datasets that previously failed.
4) No behavior change on datasets where AUTO_PAD is NOT triggered.
5) No regressions in SDS and Grid mode.

## Notes
- AUTO_PAD uses centered placement: y0=(diag-h)//2, x0=(diag-w)//2.
- The required WCS fix is: CRPIX += (pad_left, pad_top) and update pixel_shape/array_shape to the padded dims.
- This must happen BEFORE quality crop and BEFORE any later crop_image_and_wcs in Phase 5.
