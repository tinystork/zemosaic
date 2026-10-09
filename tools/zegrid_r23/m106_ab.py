"""ZM-ZEGRID-R23 rework-2 real M106 A/B — bounded offline replay (no restack).

Compares DBE OFF / legacy light-DBE (aesthetic) on the SAME genuine pre-finishing
M106 science array and the same coverage mask. Produces JSON metrics + a
controlled PNG (one stretch fitted from the OFF image, applied identically to
all panels).

Rework-2 framing (human-gate resolved): the raw science FITS is the immutable
photometric reference; the legacy light-DBE output is the AESTHETIC branch and is
NOT required to be photometrically neutral. Its known diffuse-flux loss is
reported here as a documented property of the aesthetic output, never of the raw
science (which is bit-identical to OFF by construction).

Scientific confounders (documented, not hidden):
* The exact ZeGrid R15 ``mosaic_grid.fits`` (2403x3278) used by the R20 diagnostic
  lives on the unmounted ``/media/tristan/X10 Pro/M106/`` and is NOT available on
  this host. The genuine pre-finishing candidate used here is the CLASSIC pipeline
  ``zemosaic_MT26_R66_science.fits`` (66 lights, float32, no DBE applied), so this
  is NOT an exact R15(kappa)/R20(winsor) science parity — it is a real-data proxy
  that still lets us measure OFF vs legacy light-DBE behavior on true M106 sky.
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
# Frozen legacy reference (exact historical algorithm, reproduced for parity;
# NOT used by production — production calls zfin.apply_dbe).
# ---------------------------------------------------------------------------

def _frozen_legacy(mosaic, valid_mask_hw, strength="normal"):
    from scipy import ndimage

    strength_map = {
        "weak": 24.0, "low": 24.0, "normal": 36.0,
        "strong": 52.0, "high": 52.0, "aggressive": 68.0,
    }
    sigma = float(strength_map.get(strength, 36.0))
    out = np.asarray(mosaic, dtype=np.float32).copy()
    h, w = out.shape[:2]
    finite_any = np.any(np.isfinite(out), axis=-1)
    valid_hw = finite_any
    if valid_mask_hw is not None and valid_mask_hw.shape[:2] == (h, w):
        valid_hw = valid_hw & np.asarray(valid_mask_hw, dtype=bool)
    for c in range(3):
        ch = out[..., c]
        ch_finite = np.isfinite(ch)
        ch_valid = valid_hw & ch_finite
        if not np.any(ch_valid):
            continue
        median = float(np.nanmedian(ch[ch_valid]))
        mad = float(np.nanmedian(np.abs(ch[ch_valid] - median)))
        robust_sigma = float(1.4826 * mad)
        obj_k_map = {
            "weak": 3.0, "low": 3.0, "normal": 2.8,
            "strong": 2.5, "high": 2.5, "aggressive": 2.2,
        }
        obj_k = float(obj_k_map.get(strength, 2.8))
        obj_thr = float(median + obj_k * robust_sigma)
        obj_mask = ch_valid & (ch > obj_thr)
        dil_map = {
            "weak": 2, "low": 2, "normal": 3,
            "strong": 4, "high": 4, "aggressive": 5,
        }
        dil_iters = int(dil_map.get(strength, 3))
        obj_mask = ndimage.binary_dilation(obj_mask, iterations=max(1, dil_iters))
        bg_valid = ch_valid & (~obj_mask)
        if not np.any(bg_valid):
            bg_valid = ch_valid
        fill_ref = float(np.nanmedian(ch[bg_valid]))
        ch_filled = np.where(ch_valid, ch, fill_ref).astype(np.float32)
        ch_model = np.where(obj_mask, fill_ref, ch_filled).astype(np.float32)
        bg = ndimage.gaussian_filter(ch_model, sigma=sigma, mode="nearest")
        bg_med = float(np.nanmedian(bg[bg_valid]))
        corrected = ch_filled - (bg - bg_med)
        corrected = np.where(obj_mask, ch_filled, corrected)
        ch_out = np.where(ch_valid, corrected, np.nan).astype(np.float32)
        out[..., c] = ch_out
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


def _sky_dispersion(a, mask, n=8, min_samples=50):
    """std of per-box medians over a fixed sky mask (uniformity proxy)."""
    h, w = a.shape
    meds = []
    for i in range(n):
        for j in range(n):
            sl = (slice(i * h // n, (i + 1) * h // n), slice(j * w // n, (j + 1) * w // n))
            seg = a[sl][mask[sl]]
            if seg.size >= min_samples:
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


def _fixed_sky_masks(science, valid, cy, cx):
    """Fixed INPUT-DERIVED sky masks (do NOT depend on any candidate output)."""
    gg = science[..., 1]
    fin = valid & np.isfinite(gg)
    med = float(np.median(gg[fin]))
    low = fin & (gg <= med)

    h, w = gg.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    far = fin & (r >= 400.0)

    return {
        "all_valid": fin,
        "off_low_sky": low,
        "far_from_galaxy": far,
    }


def _to_png_panels(panels, out_png, shape):
    """One stretch fitted from OFF, applied identically to all panels."""
    from PIL import Image

    off = panels[0][1][..., 1]
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
    tile = np.concatenate(panels_u8, axis=1)
    Image.fromarray(tile, mode="L").save(out_png)


def main():
    sci_path = Path("/home/tristan/M106/out/zemosaic_MT26_R66_science.fits")
    cov_path = Path("/home/tristan/M106/out/zemosaic_MT26_R66_coverage.fits")
    out_dir = Path("/home/tristan/.openclaw/workspace/.a2a-reports/ZM-ZEGRID-R23")
    out_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(sci_path) as h:
        science = np.asarray(h[0].data, dtype=np.float32)
    science = np.moveaxis(science, 0, -1)
    with fits.open(cov_path) as h:
        coverage = np.asarray(h[0].data).astype(np.float32)

    valid = (coverage > 0) & np.any(np.isfinite(science), axis=-1)
    sha = hashlib.sha256(sci_path.read_bytes()).hexdigest()

    gg = science[..., 1]
    cy, cx = np.unravel_index(np.nanargmax(np.where(np.isfinite(gg), gg, -1e30)), gg.shape)

    sky_masks = _fixed_sky_masks(science, valid, cy, cx)
    params = zfin.DBE_STRENGTH_PRESETS["normal"]

    off = science.copy()

    legacy_res = zfin.apply_final_mosaic_finishing(
        science, valid.astype(np.int32),
        config=dict(dbe_enabled=True, dbe_strength="normal",
                    dbe_params_source="preset:normal", dbe_params=params,
                    dbe_subtraction_factor=1.0, rgb_equalize=False, save_uint16=False),
    )
    legacy = legacy_res.science
    legacy_dbe = legacy_res.info["dbe"]

    # Parity against the frozen reference (production must match it exactly).
    frozen = _frozen_legacy(science, valid, "normal")
    finite_both = np.isfinite(legacy) & np.isfinite(frozen)
    parity_max_abs = float(np.max(np.abs(legacy[finite_both] - frozen[finite_both])))

    gate = {
        "attempted": bool(legacy_dbe.get("attempted")),
        "applied": bool(legacy_dbe.get("applied")),
        "reason": legacy_dbe.get("reason", ""),
        "algorithm": legacy_dbe.get("algorithm", ""),
        "dbestat": "on" if legacy_dbe.get("applied") else
                   ("noop" if legacy_dbe.get("attempted") else "off"),
        "channels": [],
        "parity_max_abs_vs_frozen": parity_max_abs,
    }
    for ch in legacy_dbe.get("channels", []):
        gate["channels"].append({
            "channel": ch.get("channel"),
            "applied": ch.get("applied"),
            "median": ch.get("median"),
            "robust_sigma": ch.get("robust_sigma"),
            "obj_frac": ch.get("obj_frac"),
            "fill_ref": ch.get("fill_ref"),
            "bg_med": ch.get("bg_med"),
            "before_full_std": ch.get("before_full_std"),
            "after_full_std": ch.get("after_full_std"),
            "full_ratio": ch.get("full_ratio"),
            "neg_frac_delta": ch.get("neg_frac_delta"),
        })

    def _disp_map(arr):
        return {
            name: [_sky_dispersion(arr[..., c], m) for c in range(3)]
            for name, m in sky_masks.items()
        }

    results = {
        "mission": "ZM-ZEGRID-R23",
        "phase": "rework-2 real M106 A/B (bounded offline replay, no restack)",
        "contract": (
            "raw science FITS = immutable photometric reference (bit-identical to "
            "OFF); legacy light-DBE output = AESTHETIC branch, not required to be "
            "photometrically neutral. Diffuse-flux loss is a documented property "
            "of the aesthetic output only."
        ),
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
            "sky_masks": {
                "all_valid": "every valid pixel",
                "off_low_sky": "valid pixels with G <= median(G) of OFF (true sky)",
                "far_from_galaxy": "valid pixels at r >= 400 px from the core",
            },
            "sky_dispersion": "std of per-box medians (8x8 boxes) over each fixed mask",
            "preview_stretch": "1st/99.5th percentile of OFF G channel, applied identically to all panels",
            "legacy_algorithm": "legacy_grid_light_dbe (variation-only, gaussian mode=nearest)",
        },
        "per_channel": {
            "off": [_channel_metrics(off[..., c], valid) for c in range(3)],
            "legacy": [_channel_metrics(legacy[..., c], valid) for c in range(3)],
        },
        "sky_dispersion_by_mask": {
            "off": _disp_map(off),
            "legacy": _disp_map(legacy),
        },
        "galaxy_flux_proxy": {
            "off": [_galaxy_flux_proxy(off[..., c], valid, cy, cx) for c in range(3)],
            "legacy": [_galaxy_flux_proxy(legacy[..., c], valid, cy, cx) for c in range(3)],
        },
        "legacy_gate": gate,
        "aesthetic_warning": (
            "aesthetic output is not photometrically neutral; use science_raw for "
            "measurement. Diffuse-flux loss is expected and documented for the "
            "aesthetic branch only."
        ),
    }

    json_path = out_dir / "m106_ab_metrics.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n")

    png_path = out_dir / "m106_ab_comparison.png"
    _to_png_panels([("OFF", off), ("LEGACY", legacy)], png_path, science.shape[:2])

    print(json.dumps(results, indent=2))
    print(f"\nWROTE {json_path}")
    print(f"WROTE {png_path}")


if __name__ == "__main__":
    main()
