# Goal
Fix ZeMosaic "black rectangles" artifacts caused by masked/transparent regions being converted to zeros.
Masked regions (from lecropper.py) must remain transparent for reproject/coadd by:
- carrying NaNs (or equivalently weight=0) through coadd
- preserving NaNs in final float FITS output where coverage==0 (and/or alpha==0)
- ensuring preview masking does not error and produces a correct RGBA PNG

# Context (current bug)
- lecropper pipeline can nanize masked pixels (good).
- later in global coadd / finalize we call np.nan_to_num(... nan=0.0) which turns NaNs into 0 → black blocks in master tiles / final mosaic.
- preview masking currently warns: "boolean index did not match ... axis 2 size 3 vs mask 1" and thus may not properly hide masked regions.

# Scope
Surgical changes in:
- zemosaic_worker.py
Optional minimal touch in:
- zemosaic_utils.py only if required to keep NaNs in FITS write (but prefer worker-only).

Do NOT refactor algorithms.
Do NOT change GUI behavior.
Do NOT change existing batch size behaviors (batch size = 0 and batch size > 1 must remain intact).

# Requirements / Acceptance criteria
1) Final float FITS mosaic contains NaN in pixels where coverage_map == 0 (and/or alpha_final==0).
2) No black rectangles due to NaN→0 conversion in the final mosaic (in FITS float).
3) Preview generation:
   - no "preview NaN masking failed" warning
   - preview PNG is RGBA when alpha_final exists and has transparent background where alpha==0
4) Coadd math remains stable:
   - masked pixels do not contaminate sums (weight must be 0 where data is non-finite)
5) Keep existing logs; add one debug/info log indicating how many pixels were nanized by coverage/alpha (optional).

# Implementation plan (worker)
- [x] Add a small helper in zemosaic_worker.py (near global coadd helpers):
      def _nanize_by_coverage(final_hwc, coverage_hw, *, alpha_u8=None):
          - compute invalid = (coverage_hw <= 0) OR (alpha_u8==0 if provided)
          - set final_hwc[invalid, :] = np.nan
          - return final_hwc

- [x] In global coadd finalize paths:
      - Remove/avoid nan_to_num on the final image (and in chunked finalize).
      - After computing chunk_result and chunk_weight:
          - set chunk_result[chunk_weight<=0] = np.nan (per-pixel), before copying to final.
          - keep coverage as float (0 where no contribution).
      - After finalize (mean/kappa/chunked), call _nanize_by_coverage(final_image, coverage_map).
      - You may still sanitize +/-inf to NaN (not 0).

- [x] Phase6 export:
      - DO NOT convert NaNs to 0 before save_fits_image for the main float FITS.
      - If a "viewer uint16" is produced, it may convert NaNs to 0 (fine), but keep alpha extension.

- [x] Preview masking fix:
      Replace boolean indexing with a broadcasting-safe np.where:
          preview_view = np.where(mask_zero[..., None], np.nan, preview_view)
      This avoids the axis mismatch warning.

# Minimal tests (no full dataset required)
- [x] Unit-ish: create a fake 3-channel mosaic array with some NaNs and a coverage map with zeros.
      Ensure _nanize_by_coverage produces NaNs where expected.
- [ ] Smoke: run a short mosaic on a small dataset and verify:
      - log has no preview NaN masking warning
      - final mosaic float FITS contains NaNs (check with fits.open and np.isnan count)
      - preview PNG has alpha channel (if alpha_final exists)

# Files to edit
- zemosaic_worker.py (primary)

# Definition of done
All acceptance criteria met, no new warnings in log, output FITS float no longer shows black rectangles from masked regions.
