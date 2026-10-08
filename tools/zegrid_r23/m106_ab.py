"""ZM-ZEGRID-R23 real M106 A/B — bounded offline replay (no restack).

Compares DBE OFF / R18-current / R23 on the SAME genuine pre-finishing M106
science array and the same coverage mask. Produces JSON metrics + a controlled
PNG (one stretch fitted from the OFF image, applied identically to all panels).

Scientific confounders (documented, not hidden):
* The exact ZeGrid R15 ``mosaic_grid.fits`` (2403x3278) used by the R20 diagnostic
  lives on the unmounted ``/media/tristan/X10 Pro/M106/`` and is NOT available on
  this host. The genuine pre-finishing candidate used here is the CLASSIC pipeline
  ``zemosaic_MT26_R66_science.fits`` (66 lights, float32, no DBE applied), so this
  is NOT an exact R15(kappa)/R20(winsor) science parity — it is a real-data proxy
  that still lets us measure OFF vs R18 vs R23 DBE behavior on true M106 sky.
* No ground-truth real galaxy flux is available; galaxy flux is reported as a
  documented proxy (aperture flux above a local annulus background around the
  M106 core), not an absolute photometric claim.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import final_mosaic_finishing as zfin  # noqa: E402


# ---------------------------------------------------------------------------
# R18-current reference (reproduced exactly for the comparison; not production)
# ---------------------------------------------------------------------------

def _r18_estimate(channel, valid, sample_step, obj_k, obj_dilate_px, smoothing):
    """R18 block-median + bright-star-only mask, full background model."""
    from scipy.ndimage import binary_dilation, gaussian_filter

    ch = np.asarray(channel, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(ch)
    if not np.any(valid):
        return np.zeros_like(ch)
    med = float(np.median(ch[valid]))
    mad = float(np.median(np.abs(ch[valid] - med)))
    sigma = 1.4826 * mad
    obj = valid & (ch > med + obj_k * sigma)
    if obj_dilate_px > 0:
        obj = binary_dilation(obj, iterations=int(obj_dilate_px))
    bg = valid & (~obj)
    if not np.any(bg):
        bg = valid
    block = max(1, int(sample_step))
    h, w = ch.shape
    bh = (h + block - 1) // block
    bw = (w + block - 1) // block
    ph = bh * block - h
    pw = bw * block - w
    vp = np.pad(ch, ((0, ph), (0, pw)), mode="constant", constant_values=np.nan)
    mp = np.pad(bg, ((0, ph), (0, pw)), mode="constant", constant_values=False)
    vp = np.asarray(vp, dtype=np.float64).reshape(bh, block, bw, block)
    mp = mp.reshape(bh, block, bw, block)
    vp = np.where(mp, vp, np.nan)
    with np.errstate(invalid="ignore"):
        grid = np.nanmedian(vp, axis=(1, 3)).astype(np.float32)
    from scipy.ndimage import distance_transform_edt, zoom
    if np.all(np.isnan(grid)):
        grid = np.full_like(grid, med, dtype=np.float32)
    elif np.isnan(grid).any():
        inds = distance_transform_edt(np.isnan(grid), return_distances=False, return_indices=True)
        grid = np.asarray(grid[tuple(inds)], dtype=np.float32)
    grid = gaussian_filter(grid, sigma=float(smoothing))
    bg_full = zoom(grid, (h / float(grid.shape[0]), w / float(grid.shape[1])), order=1).astype(np.float32)
    if bg_full.shape != ch.shape:
        out = np.full(ch.shape, np.nan, dtype=np.float32)
        out[: bg_full.shape[0], : bg_full.shape[1]] = bg_full
        bg_full = out
    return bg_full


def r18_full_subtraction(science, valid):
    """R18 destructive DBE: ch - bg (full background subtraction, no variation
    reference, bright-star-only protection, no diffuse protection)."""
    out = science.copy()
    for c in range(3):
        ch = science[..., c]
        bg = _r18_estimate(ch, valid, sample_step=24, obj_k=3.0, obj_dilate_px=3, smoothing=0.6)
        use = valid & np.isfinite(ch) & np.isfinite(bg)
        out[..., c][use] = ch[use] - bg[use]
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _channel_metrics(a, valid):
    vals = a[valid]
    return {
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
        "median": float(np.nanmedian(vals)),
        "neg_frac": float(np.mean(vals < 0)),
    }


def _sky_dispersion(a, valid, n=8):
    """Robust sky-box dispersion: std of per-sky-box medians (uniformity proxy)."""
    h, w = a.shape
    meds = []
    for i in range(n):
        for j in range(n):
            sl = (slice(i * h // n, (i + 1) * h // n), slice(j * w // n, (j + 1) * w // n))
            seg = a[sl][valid[sl]]
            if seg.size >= 50:
                meds.append(float(np.median(seg)))
    return float(np.std(meds)) if meds else float("nan")


def _galaxy_flux_proxy(a, valid, cy, cx, r_ap=40, r_in=60, r_out=100):
    """Aperture flux above a local annulus background around the M106 core.

    Proxy only (documented): the annulus median is a local sky estimate; no
    absolute photometric calibration is implied.
    """
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ap = r <= r_ap
    ann = (r >= r_in) & (r <= r_out) & valid
    if not np.any(ann):
        return float("nan")
    bg = float(np.median(a[ann]))
    ap &= valid
    return float(np.sum(a[ap] - bg))


def _to_png_panels(panels, out_png, shape):
    """One stretch fitted from OFF, applied identically to all panels."""
    from PIL import Image

    off = panels[0][1][..., 1]  # G channel of OFF
    fin = off[np.isfinite(off)]
    lo, hi = np.percentile(fin, 1.0), np.percentile(fin, 99.5)
    if hi <= lo:
        hi = lo + 1.0

    def _render(a):
        g = a[..., 1].astype(np.float32)
        g = np.clip((g - lo) / (hi - lo), 0, 1)
        g = np.where(np.isfinite(a[..., 1]), g, 0.0)
        return (g * 255).astype(np.uint8)

    panels_u8 = [_render(a) for _, a in panels]
    tile = np.concatenate(panels_u8, axis=1)  # H x (n*W)
    Image.fromarray(tile, mode="L").save(out_png)


def main():
    sci_path = Path("/home/tristan/M106/out/zemosaic_MT26_R66_science.fits")
    cov_path = Path("/home/tristan/M106/out/zemosaic_MT26_R66_coverage.fits")
    out_dir = Path("/home/tristan/.openclaw/workspace/.a2a-reports/ZM-ZEGRID-R23")
    out_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(sci_path) as h:
        science = np.asarray(h[0].data, dtype=np.float32)  # (3, H, W)
    science = np.moveaxis(science, 0, -1)  # (H, W, 3)
    with fits.open(cov_path) as h:
        coverage = np.asarray(h[0].data).astype(np.float32)

    valid = (coverage > 0) & np.any(np.isfinite(science), axis=-1)
    sha = hashlib.sha256(sci_path.read_bytes()).hexdigest()

    # galaxy core (brightest finite pixel of G channel)
    gg = science[..., 1]
    cy, cx = np.unravel_index(np.nanargmax(np.where(np.isfinite(gg), gg, -1e30)), gg.shape)

    params = zfin.DBE_STRENGTH_PRESETS["normal"]

    off = science.copy()
    r18 = r18_full_subtraction(science, valid)
    r23 = zfin.apply_final_mosaic_finishing(
        science, valid.astype(np.int32),
        config=dict(dbe_enabled=True, dbe_strength="normal",
                    dbe_params_source="preset:normal", dbe_params=params,
                    dbe_subtraction_factor=1.0, rgb_equalize=False, save_uint16=False),
    ).science

    results = {
        "mission": "ZM-ZEGRID-R23",
        "phase": "real M106 A/B (bounded offline replay, no restack)",
        "input": {
            "science_fits": str(sci_path),
            "coverage_fits": str(cov_path),
            "sha256": sha,
            "shape": list(science.shape),
            "dtype": str(science.dtype),
            "note": (
                "Classic-pipeline zemosaic_MT26_R66_science.fits (66 lights, float32, "
                "no DBE). ZeGrid R15 mosaic_grid.fits (2403x3278) used by the R20 "
                "diagnostic is on the unmounted X10 Pro and is NOT available here; "
                "this is a genuine real-data proxy, NOT exact R15(kappa)/R20(winsor) parity."
            ),
        },
        "method": {
            "coverage_mask": "coverage > 0 AND any finite",
            "galaxy_core_px": [int(cy), int(cx)],
            "galaxy_flux_proxy": "aperture(r<=40) - annulus(60<=r<=100) median; PROXY, no absolute photometry",
            "sky_dispersion": "std of per-box medians (8x8 boxes) over valid pixels",
            "preview_stretch": "1st/99.5th percentile of OFF G channel, applied identically to all panels",
        },
        "per_channel": {
            "off": [_channel_metrics(off[..., c], valid) for c in range(3)],
            "r18": [_channel_metrics(r18[..., c], valid) for c in range(3)],
            "r23": [_channel_metrics(r23[..., c], valid) for c in range(3)],
        },
        "sky_dispersion": {
            "off": [_sky_dispersion(off[..., c], valid) for c in range(3)],
            "r18": [_sky_dispersion(r18[..., c], valid) for c in range(3)],
            "r23": [_sky_dispersion(r23[..., c], valid) for c in range(3)],
        },
        "galaxy_flux_proxy": {
            "off": [_galaxy_flux_proxy(off[..., c], valid, cy, cx) for c in range(3)],
            "r18": [_galaxy_flux_proxy(r18[..., c], valid, cy, cx) for c in range(3)],
            "r23": [_galaxy_flux_proxy(r23[..., c], valid, cy, cx) for c in range(3)],
        },
    }

    json_path = out_dir / "m106_ab_metrics.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n")

    png_path = out_dir / "m106_ab_comparison.png"
    _to_png_panels(
        [("OFF", off), ("R18", r18), ("R23", r23)], png_path, science.shape[:2])

    print(json.dumps(results, indent=2))
    print(f"\nWROTE {json_path}")
    print(f"WROTE {png_path}")


if __name__ == "__main__":
    main()
