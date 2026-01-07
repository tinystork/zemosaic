# Follow-up checklist — noise_fwhm robustness + GPU propagation

## Progress
- [x] Identify the runtime-imported `zemosaic_align_stack.py` (`print(__file__)`) and patch the correct file.
- [x] CPU: make `noise_fwhm` robust to NaN/Inf borders by using finite-mask ROI before Background2D.
- [x] CPU: ensure Background2D path computes `data_subtracted = roi - background` then sanitizes NaNs/Infs.
- [x] CPU: fallback stats computed on finite pixels only; no NaN-contaminated stddev/median.
- [x] CPU: DAOStarFinder photutils compatibility (sky kw) handled without masking unrelated TypeErrors.
- [x] CPU: keep existing weight law (no change in mapping FWHM→weight), only improve robustness upstream.

- [x] GPU: fix `_prepare_frames_and_weights` so quality weights are NOT unconditionally dropped when radial weighting is off.
- [x] GPU: apply weights only when broadcastable; if mismatch, skip only the bad frame weights (or disable weighting with an explicit log).
- [x] GPU: confirm WSC weights_block still used when reject algo is winsorized sigma clip.

## Tests
- [x] Add `test_noise_fwhm_nan_borders_produces_weights` (synthetic stars + NaN borders).
- [x] Add `test_prepare_frames_and_weights_keeps_quality_weights_without_radial` (no GPU required).

## Validation steps
- [ ] Run a real dataset where previous logs showed:
      - weight_fwhm_bkg2d_error
      - weight_fwhm_warn_all_fwhm_are_infinite
    Confirm warnings reduced and weights are non-uniform when overlap/stars exist.
- [ ] Run same dataset CPU vs GPU stacking; confirm weight method is applied (not silently ignored).

## Notes / summary
- [x] Document key design decisions (ROI margin=8 px, Background2D only when ROI >= 64x64 and finite fraction >= 0.10, background_rms reduced via nanmedian/nanmean, fallback stats on finite pixels only).
- [x] Confirm no GUI/locales changes required (no new GUI keys; existing keys reused).
- Summary: ROI-based FWHM weighting now masks NaN borders and uses finite-only fallback stats; GPU weight propagation no longer drops quality weights without radial maps; added unit tests for NaN-border FWHM weights and GPU weight retention.
