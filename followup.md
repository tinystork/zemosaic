# Follow-up: Implement per-channel black-level equalization for classic-mode final mosaic

## 1) Locate finalization section in `zemosaic_worker.py`
- [x] Find where `final_mosaic_data_HWC` is finalized, saved to FITS, and preview PNG is generated.

You will see calls to:
- `zemosaic_utils.save_fits_image(...)` for the final FITS
- `zemosaic_utils.stretch_auto_asifits_like(...)` (or fallback) for preview PNG

## 2) Add helper (local to worker to keep scope tight)
- [x] Add near other small helpers in `zemosaic_worker.py`:

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
- [x] Apply the helper right before saving final FITS (classic mode only, NOT grid mode, NOT SDS phase 5) using `alpha_final` when available, otherwise `final_mosaic_coverage_HW`. Reuse the adjusted array for preview stretch.

4) Keep compatibility with existing save_fits_image baseline shift
- [x] Do not modify zemosaic_utils.save_fits_image. The new per-channel shift occurs upstream, and the existing global shift will either do nothing (min==0) or stay harmless.

5) Verification steps
- [x] Add a temporary debug log (INFO_DETAIL) printing per-channel min on valid pixels after applying (`np.nanmin(rgb[...,c][valid])`).
- [ ] Run the same classic mosaic job to confirm FITS histogram minima alignment, preview color fix, and unchanged grid/SDS behavior.

6) Don’t touch anything else
- [x] No refactors, no unrelated cleanups.
