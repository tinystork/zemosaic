# Follow-up: Implement per-channel black-level equalization for classic-mode final mosaic

## 1) Locate finalization section in `zemosaic_worker.py`
Find where `final_mosaic_data_HWC` is finalized, saved to FITS, and preview PNG is generated.

You will see calls to:
- `zemosaic_utils.save_fits_image(...)` for the final FITS
- `zemosaic_utils.stretch_auto_asifits_like(...)` (or fallback) for preview PNG

## 2) Add helper (local to worker to keep scope tight)
Add near other small helpers in `zemosaic_worker.py`:

```python
def _equalize_rgb_black_level_hwc(
    rgb_hwc: np.ndarray,
    *,
    alpha_mask: np.ndarray | None = None,
    coverage_mask: np.ndarray | None = None,
    p_low: float = 0.1,
    logger: logging.Logger | None = None,
):
    info = {"applied": False, "p_low": float(p_low), "offsets": [0.0, 0.0, 0.0]}
    if rgb_hwc is None or not isinstance(rgb_hwc, np.ndarray) or rgb_hwc.ndim != 3 or rgb_hwc.shape[-1] != 3:
        return rgb_hwc, info

    rgb = rgb_hwc.astype(np.float32, copy=False)

    finite = np.isfinite(rgb).all(axis=-1)
    valid = finite

    if alpha_mask is not None:
        a = np.asarray(alpha_mask)
        if a.ndim > 2:
            a = a[..., 0]
        if a.shape[:2] == rgb.shape[:2]:
            valid = valid & (a > 0)

    if coverage_mask is not None:
        cov = np.asarray(coverage_mask)
        if cov.ndim > 2:
            cov = cov[..., 0]
        if cov.shape[:2] == rgb.shape[:2]:
            valid = valid & np.isfinite(cov) & (cov > 0)

    if not np.any(valid):
        return rgb_hwc, info

    offsets = []
    for c in range(3):
        vals = rgb[..., c][valid]
        try:
            p = float(np.nanpercentile(vals, p_low))
        except Exception:
            p = float("nan")
        if np.isfinite(p) and p > 0:
            offsets.append(p)
        else:
            offsets.append(0.0)

    if any(o > 0 for o in offsets):
        out = rgb.copy()
        for c, o in enumerate(offsets):
            if o > 0:
                out[..., c] -= np.float32(o)
        out = np.maximum(out, 0.0)
        info["applied"] = True
        info["offsets"] = offsets
        if logger:
            logger.info("[RGB-BL] applied=True p_low=%.3f offsets=(%.3f, %.3f, %.3f)", p_low, offsets[0], offsets[1], offsets[2])
        return out, info

    return rgb_hwc, info
Notes:

Use alpha_final if available, else final_mosaic_coverage_HW if available.

Do NOT rescale. Only subtract offsets and clamp at 0.

3) Apply it in the right places (classic only)
Right before saving final FITS (classic mode only):

Condition: NOT grid mode, NOT SDS phase 5.

Call the helper with alpha_final (preferred), else final_mosaic_coverage_HW.

Example:

python
Copier le code
if (final_mosaic_data_HWC is not None) and (not sds_mode_phase5) and (not grid_mode_enabled):
    final_mosaic_data_HWC, bl_info = _equalize_rgb_black_level_hwc(
        final_mosaic_data_HWC,
        alpha_mask=alpha_final,
        coverage_mask=final_mosaic_coverage_HW,
        p_low=0.1,
        logger=logger,
    )
Then proceed to save_fits_image(...) as usual.

Also apply it right before preview stretch (same conditions). If you already applied it before saving, do NOT re-apply; reuse the already adjusted array.

4) Keep compatibility with existing save_fits_image baseline shift
Do not modify zemosaic_utils.save_fits_image. The new per-channel shift occurs upstream, and the existing global shift will either do nothing (min==0) or stay harmless.

5) Verification steps
Add a temporary debug log (INFO_DETAIL) printing per-channel min on valid pixels after applying:

np.nanmin(rgb[...,c][valid]) should be near 0 for all channels.

Run the same classic mosaic job:

Confirm FITS histogram in ASIFitsView: R/G/B minima aligned (no big offset on G/B).

Confirm preview PNG is no longer green/teal.

Ensure grid mode and SDS runs are unchanged.

6) Don’t touch anything else
No refactors, no unrelated cleanups.