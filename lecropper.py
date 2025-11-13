# zemosaic_autocrop_gui.py
# ZeMosaic Auto-Crop (signal-based) – version corrigée
# Dépendances: numpy, matplotlib, astropy, tkinter
# (pas de SciPy / skimage requis)
"""
╔══════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                  ║
║                                                                      ║
║ Auteur  : Tinystork, seigneur des couteaux à beurre (aka Tristan Nauleau)  
║ Partenaire : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System  
║              (aka ChatGPT, Grand Maître du ciselage de code)         ║
║                                                                      ║
║ Licence : GNU General Public License v3.0 (GPL-3.0)                  ║
║                                                                      ║
║ Description :                                                        ║
║   Ce programme a été forgé à la lueur des pixels et de la caféine,   ║
║   dans le but noble de transformer des nuages de photons en art      ║
║   astronomique. Si vous l’utilisez, pensez à dire “merci”,           ║
║   à lever les yeux vers le ciel, ou à citer Tinystork et J.A.R.V.I.S.║
║   (le karma des développeurs en dépend).                             ║
║                                                                      ║
║ Avertissement :                                                      ║
║   Aucune IA ni aucun couteau à beurre n’a été blessé durant le       ║
║   développement de ce code.                                          ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                  ║
║                                                                      ║
║ Author  : Tinystork, Lord of the Butter Knives (aka Tristan Nauleau) ║
║ Partner : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System  
║           (aka ChatGPT, Grand Master of Code Chiseling)              ║
║                                                                      ║
║ License : GNU General Public License v3.0 (GPL-3.0)                  ║
║                                                                      ║
║ Description:                                                         ║
║   This program was forged under the sacred light of pixels and       ║
║   caffeine, with the noble intent of turning clouds of photons into  ║
║   astronomical art. If you use it, please consider saying “thanks,”  ║
║   gazing at the stars, or crediting Tinystork and J.A.R.V.I.S. —     ║
║   developer karma depends on it.                                     ║
║                                                                      ║
║ Disclaimer:                                                          ║
║   No AIs or butter knives were harmed in the making of this code.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

print("INFO: using NEW lecropper (Alt-Az + Quality-crop pipeline)")

import os
import sys
import glob
import csv
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox

import logging

import numpy as np
from astropy.io import fits
from typing import Any

try:  # SciPy is optional; fall back gracefully if missing
    from scipy import ndimage as _NDIMAGE  # type: ignore[import]
except Exception:  # pragma: no cover - SciPy not installed
    _NDIMAGE = None

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ----------------------------- utilitaires numpy -----------------------------

def _safe_minmax(a):
    a = np.asarray(a)
    finite = np.isfinite(a)
    if not finite.any():
        return 0.0, 0.0
    return float(a[finite].min()), float(a[finite].max())

def _normalize01(a):
    a = np.asarray(a, dtype=np.float32)
    a = np.clip(a, 0, None)
    mn, mx = _safe_minmax(a)
    if mx <= 0 or mx - mn <= 0:
        return np.zeros_like(a, dtype=np.float32)
    a = (a - mn) / (mx - mn)
    return a.astype(np.float32)

def pool2(a, mode="mean", band=32):
    """Réduction en grille par blocs band x band (mode: mean ou var)."""
    H, W = a.shape
    by = max(1, band)
    bx = max(1, band)
    sy = H // by
    sx = W // bx
    if sy <= 0 or sx <= 0:
        return _normalize01(a)  # fallback
    a = a[:sy*by, :sx*bx]
    if mode == "mean":
        out = a.reshape(sy, by, sx, bx).mean(axis=(1,3))
    elif mode == "var":
        out = a.reshape(sy, by, sx, bx).var(axis=(1,3))
    else:
        out = a.reshape(sy, by, sx, bx).mean(axis=(1,3))
    return out.astype(np.float32)

def binary_erosion1(m):
    """Érosion binaire 8-connexe (1 itération) sans SciPy."""
    m = m.astype(bool)
    n = (
        m &
        np.roll(m,  1, 0) & np.roll(m, -1, 0) &
        np.roll(m,  1, 1) & np.roll(m, -1, 1) &
        np.roll(np.roll(m,  1, 0),  1, 1) &
        np.roll(np.roll(m,  1, 0), -1, 1) &
        np.roll(np.roll(m, -1, 0),  1, 1) &
        np.roll(np.roll(m, -1, 0), -1, 1)
    )
    return n


def _binary_erosion(mask, iterations=1):
    """Binary erosion with zero padding to avoid wrap-around effects."""
    mask = mask.astype(bool)
    iters = max(0, int(iterations))
    if iters == 0:
        return mask
    for _ in range(iters):
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        mask = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1] & padded[2:, 1:-1]
            & padded[1:-1, :-2] & padded[1:-1, 2:]
            & padded[:-2, :-2] & padded[:-2, 2:]
            & padded[2:, :-2] & padded[2:, 2:]
        )
    return mask

def bbox_from_mask(m):
    """BBox (y0,x0,y1,x1) du masque True. Si vide -> None."""
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return (y0, x0, y1, x1)


# -------------------------- coeur: détection auto-crop -----------------------

logger = logging.getLogger(__name__)

MIN_COVERAGE_ABS_DEFAULT = 3.0
MIN_COVERAGE_FRAC_DEFAULT = 0.4
MORPH_OPEN_PX_DEFAULT = 3

_COVERAGE_EXTNAME_CANDIDATES = (
    "COVERAGE",
    "COVER",
    "COV",
    "WEIGHT",
    "WEIGHTS",
    "WEIGHTMAP",
)
_COVERAGE_SIDE_SUFFIXES = ("_coverage", ".coverage", "-coverage")
_COVERAGE_SIDE_EXTENSIONS = (".fits", ".fit", ".fts")

_ALPHA_EXPORT_SETTINGS: dict[str, Any] | None = None


def _get_altaz_alpha_settings():
    global _ALPHA_EXPORT_SETTINGS
    if _ALPHA_EXPORT_SETTINGS is not None:
        return _ALPHA_EXPORT_SETTINGS

    settings = {
        "fits": True,
        "sidecar": False,
        "format": "png",
    }

    try:
        import zemosaic_config as _zconf  # type: ignore[import]

        cfg = {}
        try:
            if hasattr(_zconf, "load_config"):
                cfg = _zconf.load_config() or {}
        except Exception:
            cfg = {}

        settings["fits"] = bool(cfg.get("altaz_export_alpha_fits", settings["fits"]))
        settings["sidecar"] = bool(cfg.get("altaz_export_alpha_sidecar", settings["sidecar"]))
        fmt = cfg.get("altaz_alpha_sidecar_format")
        if isinstance(fmt, str) and fmt:
            settings["format"] = fmt.lower()
    except Exception:
        pass

    def _from_env(key: str, default: bool) -> bool:
        val = os.environ.get(key)
        if val is None:
            return default
        if val.strip().lower() in {"0", "false", "off", "no"}:
            return False
        return True

    settings["fits"] = _from_env("ZEM_ALTALPHA_FITS", settings["fits"])
    settings["sidecar"] = _from_env("ZEM_ALTALPHA_SIDECAR", settings["sidecar"])
    fmt_env = os.environ.get("ZEM_ALTALPHA_SIDECAR_FORMAT")
    if fmt_env:
        settings["format"] = fmt_env.lower()

    _ALPHA_EXPORT_SETTINGS = settings
    return settings


def _build_altaz_mask_from_coverage(
    coverage: np.ndarray,
    min_coverage_abs: float = MIN_COVERAGE_ABS_DEFAULT,
    min_coverage_frac: float = MIN_COVERAGE_FRAC_DEFAULT,
    morph_open_px: int = MORPH_OPEN_PX_DEFAULT,
    margin_percent: float = 5.0,
    decay: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (good_mask, alpha) pair from a coverage map."""

    cov_arr = np.asarray(coverage, dtype=np.float32)
    if cov_arr.ndim != 2:
        raise ValueError(f"coverage must be 2D, got shape={cov_arr.shape}")

    cov_arr = np.nan_to_num(cov_arr, nan=0.0, posinf=0.0, neginf=0.0)
    cov_arr = np.clip(cov_arr, 0.0, None, out=cov_arr)

    if cov_arr.size == 0:
        raise ValueError("coverage map is empty")

    if np.any(cov_arr):
        max_cov = float(np.nanmax(cov_arr))
    else:
        max_cov = 0.0

    thr_abs = max(0.0, float(min_coverage_abs))
    thr_frac_value = max(0.0, float(min_coverage_frac)) * max_cov
    thr = max(thr_abs, thr_frac_value)

    good = cov_arr >= thr
    if not np.any(good):
        if max_cov > 0.0:
            fallback_thr = max(0.25 * max_cov, 1.0)
            good = cov_arr >= fallback_thr
        if not np.any(good):
            good = cov_arr > 0.0
    if not np.any(good):
        raise ValueError("coverage map lacks valid pixels above threshold")

    ndi = _NDIMAGE
    morph_iters = max(0, int(morph_open_px))
    if ndi is not None and morph_iters > 0:
        structure = ndi.generate_binary_structure(2, 1)
        if morph_iters > 1:
            structure = ndi.binary_dilation(structure, iterations=morph_iters - 1)
        cleaned = ndi.binary_opening(good, structure=structure)
        if np.any(cleaned):
            good = cleaned
    elif morph_iters > 0:
        logger.debug("SciPy not available; skipping morphological cleanup")

    if ndi is not None and np.any(good):
        dist = ndi.distance_transform_edt(good)
    else:
        dist = np.where(good, 1.0, 0.0).astype(np.float32, copy=False)

    h, w = good.shape
    margin_percent = float(margin_percent)
    min_dim = float(min(h, w)) if h and w else 0.0
    margin_px = margin_percent * 0.01 * min_dim
    if margin_percent <= 0.0 or min_dim == 0.0:
        margin_px = 0.0
    else:
        margin_px = max(1.0, margin_px)

    decay = max(float(decay), 1e-3)
    alpha = np.zeros_like(dist, dtype=np.float32)

    if margin_px <= 0.0:
        alpha = good.astype(np.float32, copy=False)
    else:
        inside = dist >= margin_px
        edge = (dist > 0) & (dist < margin_px)
        alpha[inside] = 1.0
        if np.any(edge):
            t = np.clip(dist[edge] / margin_px, 0.0, 1.0)
            alpha[edge] = np.clip(t ** (1.0 / decay), 0.0, 1.0)
        alpha[~good] = 0.0

    alpha = np.clip(alpha, 0.0, 1.0, out=alpha)

    logger.info(
        "AltAz cleanup: using coverage-based mask (thr_abs=%.2f, thr_frac=%.2f, thr=%.2f, morph_open_px=%d)",
        thr_abs,
        float(min_coverage_frac),
        thr,
        morph_iters,
    )

    return good.astype(bool, copy=False), alpha.astype(np.float32, copy=False)


def _build_low_signal_border_mask(
    arr: np.ndarray,
    spatial_shape: tuple[int, int],
    leading_spatial: bool,
    margin_percent: float = 5.0,
    k_sigma: float = 1.0,
    min_inner_frac: float = 0.2,
) -> np.ndarray:
    """
    Build a 2D float mask [0,1] that downweights low-signal border regions.
    """

    if len(spatial_shape) != 2:
        raise ValueError("spatial_shape must be a tuple of (H, W)")

    H, W = int(spatial_shape[0]), int(spatial_shape[1])
    if H <= 0 or W <= 0:
        raise ValueError("spatial dimensions must be positive")

    arr_f = np.asarray(arr, dtype=np.float32, copy=False)
    arr_f = np.nan_to_num(arr_f, nan=0.0, posinf=0.0, neginf=0.0)

    if arr_f.ndim == 2:
        lum2d = np.abs(arr_f)
    else:
        data = arr_f
        if leading_spatial:
            if data.shape[0] != H or data.shape[1] != W:
                if data.size % (H * W) != 0:
                    raise ValueError("array cannot be reshaped to (H, W, ...)")
                data = data.reshape(H, W, -1)
            axes = tuple(range(2, data.ndim))
        else:
            if data.shape[-2] != H or data.shape[-1] != W:
                if data.size % (H * W) != 0:
                    raise ValueError("array cannot be reshaped to (..., H, W)")
                data = data.reshape(-1, H, W)
            axes = tuple(range(0, data.ndim - 2))
        if axes:
            lum2d = np.mean(np.abs(data), axis=axes)
        else:
            lum2d = np.abs(data)

    lum2d = np.nan_to_num(lum2d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if lum2d.shape != (H, W):
        lum2d = lum2d.reshape(H, W)

    min_inner_frac = float(np.clip(min_inner_frac, 0.0, 0.45))
    inner_y0 = int(min_inner_frac * H)
    inner_y1 = int((1.0 - min_inner_frac) * H)
    inner_x0 = int(min_inner_frac * W)
    inner_x1 = int((1.0 - min_inner_frac) * W)

    def _ensure_span(start, end, limit):
        start = max(0, min(start, limit))
        end = max(start + 2, min(end, limit))
        if end - start < 4:
            center = limit // 2
            start = max(0, center - 2)
            end = min(limit, center + 2)
        return start, end

    inner_y0, inner_y1 = _ensure_span(inner_y0, inner_y1, H)
    inner_x0, inner_x1 = _ensure_span(inner_x0, inner_x1, W)

    core = lum2d[inner_y0:inner_y1, inner_x0:inner_x1]
    if core.size == 0:
        core = lum2d

    med = float(np.median(core))
    mad = float(np.median(np.abs(core - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(core))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(lum2d))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1e-3

    margin_percent = float(margin_percent)
    min_dim = float(min(H, W))
    margin_px = max(1.0, margin_percent * 0.01 * min_dim)
    band = max(4, int(round(margin_px)))
    thr_low = med - float(k_sigma) * sigma

    border = np.zeros_like(lum2d, dtype=bool)
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True

    low_signal = border & (lum2d < thr_low)

    ndi = _NDIMAGE
    if ndi is not None and np.any(low_signal):
        try:
            low_signal = ndi.binary_opening(low_signal, iterations=1)
        except Exception:
            pass

    good = ~low_signal
    if not np.any(good):
        return np.ones_like(lum2d, dtype=np.float32)

    if ndi is not None:
        dist = ndi.distance_transform_edt(good)
    else:
        dist = np.where(good, 1.0, 0.0).astype(np.float32, copy=False)

    alpha = np.zeros_like(dist, dtype=np.float32)
    margin_feather = max(1.0, margin_px)

    if margin_feather <= 0:
        return good.astype(np.float32, copy=False)

    inside = dist >= margin_feather
    edge = (dist > 0) & (dist < margin_feather)

    alpha[inside] = 1.0
    if np.any(edge):
        t = dist[edge] / margin_feather
        alpha[edge] = t

    alpha[~good] = 0.0
    alpha = np.clip(alpha, 0.0, 1.0, out=alpha)
    return alpha


def detect_autocrop_rgb(
    lum2d,
    R,
    G,
    B,
    band_px=32,
    k_sigma=2.0,
    margin_px=8,
):
    """
    Renvoie (y0, x0, y1, x1) en pleine résolution.
    Corrigé: normalisation, non-noir, planchers, double érosion, trim full-res.
    """

    H, W = lum2d.shape

    # --- 1) Normalisation préalable (0..1) ---
    lum2d = _normalize01(lum2d)
    R = _normalize01(R)
    G = _normalize01(G)
    B = _normalize01(B)

    # --- 2) Pooling (grille réduite) ---
    Ls = pool2(lum2d, "mean", band=band_px)
    Vs = pool2(lum2d, "var",  band=band_px)
    Rs = pool2(R, "mean", band=band_px)
    Gs = pool2(G, "mean", band=band_px)
    Bs = pool2(B, "mean", band=band_px)

    Hs, Ws = Ls.shape
    if Hs < 3 or Ws < 3:
        # grille trop petite -> garde tout (avec marge)
        return max(0, 0 + margin_px), max(0, 0 + margin_px), min(H, H - margin_px), min(W, W - margin_px)

    # --- 3) Seuils robustes + non-noir ---
    medL = float(np.nanmedian(Ls))
    sigL = float(np.nanstd(Ls))
    eps_small = max(1e-6, float(np.nanpercentile(Ls, 0.5)))

    thrL = max(medL - k_sigma * max(sigL, 1e-6), float(np.nanpercentile(Ls, 5.0)), eps_small)
    thrV = max(float(np.nanpercentile(Vs, 15.0)), float(np.nanpercentile(Vs, 1.0)) or 1e-6)

    nonblack_small = (Rs > eps_small) | (Gs > eps_small) | (Bs > eps_small)
    base = np.isfinite(Ls) & nonblack_small & (Ls >= thrL) & (Vs >= thrV)

    # --- 4) Rejet chroma générique en bordure ---
    Cmax = np.maximum(np.maximum(Rs, Gs), Bs)
    Cmin = np.minimum(np.minimum(Rs, Gs), Bs)
    sat = Cmax - Cmin
    edge_band_px = max(24, margin_px)
    edge_band_small = max(1, int(np.ceil(edge_band_px / float(max(1, band_px)))))
    edge = np.zeros_like(base, dtype=bool)
    edge[:edge_band_small, :] = True
    edge[-edge_band_small:, :] = True
    edge[:, :edge_band_small] = True
    edge[:, -edge_band_small:] = True
    t_chroma = 0.18
    mask_chroma = edge & (sat > t_chroma)
    mask_chroma = _binary_erosion(mask_chroma, iterations=2)
    logger.debug("QBC edge-chroma: t=%.2f, erode=%d", t_chroma, 2)
    if mask_chroma.any():
        base[mask_chroma] = False

    # --- 5) Érosion (deux passes) pour casser les ponts fins ---
    m_small = binary_erosion1(base)
    m_small = binary_erosion1(m_small)

    # --- 6) BBox sur grille réduite ---
    box_small = bbox_from_mask(m_small)
    if box_small is None:
        # rien de fiable détecté -> garde tout
        return 0, 0, H, W

    y0s, x0s, y1s, x1s = box_small

    # --- 7) Reprojection plein format + marges ---
    fy = H / float(Hs)
    fx = W / float(Ws)
    y0 = max(0, int(y0s * fy) + margin_px)
    x0 = max(0, int(x0s * fx) + margin_px)
    y1 = min(H, int(y1s * fy) - margin_px)
    x1 = min(W, int(x1s * fx) - margin_px)

    if y1 <= y0 + 8 or x1 <= x0 + 8:
        # rectangle dégénéré -> garde tout
        return 0, 0, H, W

    # --- 8) Trim final en pleine résolution (retire 3 px si trop sombre) ---
    def _trim_dark_edges_fullres(lum, rect, step=3, minsize=32):
        y0, x0, y1, x1 = rect
        eps_fr = max(1e-6, float(np.nanpercentile(lum, 0.5)))
        # bas
        while (y1 - y0) > minsize and np.nanmean(lum[y1-step:y1, x0:x1]) < eps_fr:
            y1 -= step
        # droite
        while (x1 - x0) > minsize and np.nanmean(lum[y0:y1, x1-step:x1]) < eps_fr:
            x1 -= step
        return y0, x0, y1, x1

    y0, x0, y1, x1 = _trim_dark_edges_fullres(lum2d, (y0, x0, y1, x1))
    before_color_trim = (int(y0), int(x0), int(y1), int(x1))
    after_color_trim = _trim_color_edges_fullres(R, G, B, before_color_trim, step=3, minsize=32, k=k_sigma)
    logger.debug("QBC color-trim: before=%s after=%s", before_color_trim, after_color_trim)

    y0, x0, y1, x1 = after_color_trim
    touches = (
        y0 <= margin_px or x0 <= margin_px or y1 >= (H - margin_px) or x1 >= (W - margin_px)
    )
    if touches and band_px > 32:
        band_px2 = max(16, band_px // 2)
        logger.debug("QBC second-pass triggered: band %d -> %d", band_px, band_px2)
        refined = detect_autocrop_rgb(lum2d, R, G, B, band_px=band_px2, k_sigma=k_sigma, margin_px=margin_px)
        y0 = max(y0, refined[0])
        x0 = max(x0, refined[1])
        y1 = min(y1, refined[2])
        x1 = min(x1, refined[3])
        if y1 <= y0 or x1 <= x0:
            return refined

    return int(y0), int(x0), int(y1), int(x1)


def _radial_falloff_mask(shape, margin_percent=5.0, decay_ratio=0.15):
    """Return a smooth radial mask (1 center -> 0 edges).

    Parameters
    ----------
    shape : tuple[int, int]
        Spatial (H, W) dimensions of the mask to generate.
    margin_percent : float
        Percentage of the radius to attenuate (0-50%).
    decay_ratio : float
        Ratio of the transition width relative to the margin (0 = hard cut).
    """

    if len(shape) != 2:
        raise ValueError("shape must be (H, W)")

    H, W = shape
    if H == 0 or W == 0:
        return np.zeros((H, W), dtype=np.float32)

    yy = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    yy, xx = np.meshgrid(yy, xx, indexing="ij")
    r = np.hypot(xx, yy)
    r_max = float(r.max()) or 1.0
    r_norm = r / r_max

    margin = float(np.clip(margin_percent / 100.0, 0.0, 0.5))
    if margin <= 0.0:
        return np.ones((H, W), dtype=np.float32)

    inner = max(0.0, 1.0 - margin)
    decay_ratio = max(0.0, float(decay_ratio))
    transition = max(1e-3, margin * decay_ratio)
    outer = min(1.0, inner + transition)

    mask = np.ones((H, W), dtype=np.float32)

    hard_zone = r_norm >= outer
    mask[hard_zone] = 0.0

    if outer > inner:
        # Smooth step transition between inner and outer.
        trans_zone = (r_norm >= inner) & (r_norm < outer)
        if np.any(trans_zone):
            width = outer - inner
            t = (r_norm[trans_zone] - inner) / width
            smooth = 1.0 - (3.0 - 2.0 * t) * (t ** 2)
            mask[trans_zone] = smooth.astype(np.float32)

    return np.clip(mask, 0.0, 1.0)


def mask_altaz_artifacts(
    full_img,
    margin_percent=5.0,
    decay_ratio=0.15,
    fill_value=0.0,
    return_mask=False,
    hard_threshold=1e-3,
    coverage: np.ndarray | None = None,
    min_coverage_abs: float | None = None,
    min_coverage_frac: float | None = None,
    morph_open_px: int | None = None,
):
    """Softly attenuate Alt-Az rotation artifacts near the corners.

    Parameters
    ----------
    full_img : np.ndarray
        Image (2D) or cube (channel-first or channel-last) to be masked.
    margin_percent : float
        Percentage of the radius to attenuate from the borders.
    decay_ratio : float
        Controls how soft the transition is (0 -> hard cut).
    fill_value : float or None
        Value to use once the mask is close to zero. ``None`` keeps the
        smooth attenuation without forcing a hard fill.
    return_mask : bool
        Whether to return the generated mask alongside the image.
    hard_threshold : float
        Threshold below which the mask is considered "off" for fill_value.
    coverage : np.ndarray, optional
        Optional coverage map used to derive a data-driven alpha mask.
    min_coverage_abs : float, optional
        Absolute coverage threshold. Defaults to :data:`MIN_COVERAGE_ABS_DEFAULT`.
    min_coverage_frac : float, optional
        Fraction of max coverage to keep. Defaults to
        :data:`MIN_COVERAGE_FRAC_DEFAULT`.
    morph_open_px : int, optional
        Radius (in pixels) for morphological opening when coverage is used.
    """

    arr = np.asarray(full_img, dtype=np.float32)
    arr = arr.copy()

    if arr.ndim < 2:
        raise ValueError("mask_altaz_artifacts expects at least 2 dimensions")

    if arr.ndim == 2:
        spatial_shape = arr.shape
        leading_spatial = True
    else:
        head_shape = arr.shape[:2]
        tail_shape = arr.shape[-2:]
        head_area = head_shape[0] * head_shape[1]
        tail_area = tail_shape[0] * tail_shape[1]
        if head_area >= tail_area:
            spatial_shape = head_shape
            leading_spatial = True
        else:
            spatial_shape = tail_shape
            leading_spatial = False

    coverage_mask = None
    cov_arr = None
    if coverage is not None:
        try:
            cov_arr = np.asarray(coverage, dtype=np.float32)
            if cov_arr.ndim > 2:
                cov_arr = np.squeeze(cov_arr)
            if cov_arr.ndim != 2:
                raise ValueError("coverage map must be 2D")
            if cov_arr.shape != spatial_shape:
                logger.debug(
                    "Coverage map ignored due to shape mismatch: %s vs %s",
                    cov_arr.shape,
                    spatial_shape,
                )
                cov_arr = None
            elif not np.isfinite(cov_arr).any():
                logger.debug("Coverage map ignored: no finite values")
                cov_arr = None
        except Exception as exc:
            logger.debug("Coverage map ignored (%s)", exc)
            cov_arr = None

    if cov_arr is not None:
        try:
            cov_abs = MIN_COVERAGE_ABS_DEFAULT if min_coverage_abs is None else float(min_coverage_abs)
            cov_frac = MIN_COVERAGE_FRAC_DEFAULT if min_coverage_frac is None else float(min_coverage_frac)
            cov_morph = MORPH_OPEN_PX_DEFAULT if morph_open_px is None else int(morph_open_px)
        except Exception:
            cov_abs = MIN_COVERAGE_ABS_DEFAULT
            cov_frac = MIN_COVERAGE_FRAC_DEFAULT
            cov_morph = MORPH_OPEN_PX_DEFAULT

        try:
            _, coverage_mask = _build_altaz_mask_from_coverage(
                cov_arr,
                min_coverage_abs=cov_abs,
                min_coverage_frac=cov_frac,
                morph_open_px=cov_morph,
                margin_percent=margin_percent,
                decay=decay_ratio,
            )
        except Exception as exc:
            logger.warning(
                "AltAz coverage-based mask failed (%s); falling back to radial feather",
                exc,
            )
            coverage_mask = None

    if coverage_mask is not None:
        m = np.asarray(coverage_mask, dtype=np.float32)
        if m.ndim != 2:
            coverage_mask = None
        else:
            m = np.clip(m, 0.0, 1.0, out=m)
            frac_high = float((m > 0.99).mean())
            frac_low = float((m < 0.01).mean())
            if frac_high > 0.98 or frac_low > 0.98:
                logger.debug(
                    "Coverage-based mask considered non-discriminative "
                    "(frac_high=%.3f, frac_low=%.3f); will try signal-based fallback.",
                    frac_high,
                    frac_low,
                )
                coverage_mask = None

    mask2d = None
    if coverage_mask is not None:
        mask2d = np.asarray(coverage_mask, dtype=np.float32, copy=False)
    else:
        try:
            mask2d = _build_low_signal_border_mask(
                arr,
                spatial_shape=spatial_shape,
                leading_spatial=leading_spatial,
                margin_percent=margin_percent,
                k_sigma=1.0,
                min_inner_frac=0.2,
            )
            logger.debug("AltAz cleanup: using low-signal border mask fallback")
        except Exception as exc:
            logger.debug("Low-signal fallback failed (%s); will use radial feather", exc)
            mask2d = None

    if mask2d is None:
        mask2d = _radial_falloff_mask(spatial_shape, margin_percent, decay_ratio)

    if arr.ndim == 2:
        masked = arr * mask2d
    elif leading_spatial:
        stretch = spatial_shape + (1,) * (arr.ndim - 2)
        masked = arr * mask2d.reshape(stretch)
    else:
        stretch = (1,) * (arr.ndim - 2) + spatial_shape
        masked = arr * mask2d.reshape(stretch)

    if fill_value is not None:
        fill_mask = mask2d <= float(hard_threshold)
        if np.isnan(fill_value):
            fill_value = np.nan
        if masked.ndim == 2:
            masked = np.where(fill_mask, fill_value, masked)
        elif leading_spatial:
            fill_shape = spatial_shape + (1,) * (masked.ndim - 2)
            masked = np.where(fill_mask.reshape(fill_shape), fill_value, masked)
        else:
            fill_shape = (1,) * (masked.ndim - 2) + spatial_shape
            masked = np.where(fill_mask.reshape(fill_shape), fill_value, masked)

    if return_mask:
        return masked, mask2d
    return masked


def apply_altaz_cleanup(
    image,
    margin_percent: float = 5.0,
    decay: float = 0.15,
    nanize: bool = False,
    coverage: np.ndarray | None = None,
    min_coverage_abs: float | None = None,
    min_coverage_frac: float | None = None,
    morph_open_px: int | None = None,
):
    """
    Public helper that mirrors the Alt-Az cleanup expected by ZeMosaic.

    The function intentionally avoids any dependency on ZeMosaic so that this
    module stays standalone-friendly.
    The optional *coverage* argument enables coverage-driven alpha masks.
    """

    if image is None:
        return None

    fill_value = np.nan if nanize else 0.0
    return mask_altaz_artifacts(
        image,
        margin_percent=margin_percent,
        decay_ratio=decay,
        fill_value=fill_value,
        coverage=coverage,
        min_coverage_abs=min_coverage_abs,
        min_coverage_frac=min_coverage_frac,
        morph_open_px=morph_open_px,
    )


def _sanitize_coverage_array(raw, expected_shape: tuple[int, int] | None = None):
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        return None
    if expected_shape is not None and arr.shape != expected_shape:
        logger.debug("Coverage array rejected due to shape mismatch: %s vs %s", arr.shape, expected_shape)
        return None
    if not np.isfinite(arr).any():
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _maybe_extract_coverage_from_hdul(hdul, expected_shape: tuple[int, int] | None = None):
    if hdul is None:
        return None
    for idx in range(1, len(hdul)):
        hdu = hdul[idx]
        data = getattr(hdu, "data", None)
        if data is None:
            continue
        arr = _sanitize_coverage_array(data, expected_shape=expected_shape)
        if arr is None:
            continue
        extname = (getattr(hdu, "name", "") or hdu.header.get("EXTNAME", "") or "").upper()
        bunit = str(hdu.header.get("BUNIT", "")).lower()
        if (
            extname in _COVERAGE_EXTNAME_CANDIDATES
            or "COVER" in extname
            or "WEIGHT" in extname
            or "count" in bunit
        ):
            logger.info(
                "Detected coverage HDU '%s' (idx=%d, shape=%s)",
                extname or f"HDU{idx}",
                idx,
                arr.shape,
            )
            return arr
    return None


def _maybe_load_sidecar_coverage(in_path: str, expected_shape: tuple[int, int] | None = None):
    if not in_path:
        return None

    base, ext = os.path.splitext(in_path)
    ext_candidates = []
    if ext:
        ext_candidates.append(ext)
    for extra in _COVERAGE_SIDE_EXTENSIONS:
        if extra not in ext_candidates:
            ext_candidates.append(extra)

    candidates = []
    for suffix in _COVERAGE_SIDE_SUFFIXES:
        for ext_candidate in ext_candidates:
            candidates.append(f"{base}{suffix}{ext_candidate}")

    for cand in candidates:
        if not os.path.exists(cand):
            continue
        try:
            with fits.open(cand, memmap=False) as hdul:
                arr = _sanitize_coverage_array(hdul[0].data, expected_shape=expected_shape)
                if arr is not None:
                    logger.info("Detected coverage sidecar %s (shape=%s)", os.path.basename(cand), arr.shape)
                    return arr
        except Exception as exc:
            logger.debug("Failed to load coverage sidecar %s: %s", cand, exc)
    return None


def _trim_color_edges_fullres(R, G, B, rect, step=3, minsize=32, k=3.0):
    y0, x0, y1, x1 = rect
    if (y1 - y0) <= minsize * 2 or (x1 - x0) <= minsize * 2:
        return rect

    inner = (slice(y0 + minsize, y1 - minsize), slice(x0 + minsize, x1 - minsize))
    if (inner[0].start >= inner[0].stop) or (inner[1].start >= inner[1].stop):
        return rect

    r_inner = R[inner]
    g_inner = G[inner]
    b_inner = B[inner]

    r_med = np.nanmedian(r_inner)
    g_med = np.nanmedian(g_inner)
    b_med = np.nanmedian(b_inner)

    r_mad = np.nanmedian(np.abs(r_inner - r_med)) + 1e-6
    g_mad = np.nanmedian(np.abs(g_inner - g_med)) + 1e-6
    b_mad = np.nanmedian(np.abs(b_inner - b_med)) + 1e-6

    def over(chan, med, mad, sl):
        return np.nanmean(chan[sl]) > med + k * mad

    while (x1 - x0) > minsize and (
        over(R, r_med, r_mad, (slice(y0, y1), slice(x0, min(x0 + step, x1))))
        or over(G, g_med, g_mad, (slice(y0, y1), slice(x0, min(x0 + step, x1))))
        or over(B, b_med, b_mad, (slice(y0, y1), slice(x0, min(x0 + step, x1))))
    ):
        x0 += step

    while (x1 - x0) > minsize and (
        over(R, r_med, r_mad, (slice(y0, y1), slice(max(x1 - step, x0), x1)))
        or over(G, g_med, g_mad, (slice(y0, y1), slice(max(x1 - step, x0), x1)))
        or over(B, b_med, b_mad, (slice(y0, y1), slice(max(x1 - step, x0), x1)))
    ):
        x1 -= step

    while (y1 - y0) > minsize and (
        over(R, r_med, r_mad, (slice(y0, min(y0 + step, y1)), slice(x0, x1)))
        or over(G, g_med, g_mad, (slice(y0, min(y0 + step, y1)), slice(x0, x1)))
        or over(B, b_med, b_mad, (slice(y0, min(y0 + step, y1)), slice(x0, x1)))
    ):
        y0 += step

    while (y1 - y0) > minsize and (
        over(R, r_med, r_mad, (slice(max(y1 - step, y0), y1), slice(x0, x1)))
        or over(G, g_med, g_mad, (slice(max(y1 - step, y0), y1), slice(x0, x1)))
        or over(B, b_med, b_mad, (slice(max(y1 - step, y0), y1), slice(x0, x1)))
    ):
        y1 -= step

    return int(y0), int(x0), int(y1), int(x1)


# ------------------------------ I/O FITS & preview ---------------------------

def load_fits_rgb(path):
    """Retourne (lum2d, R, G, B) en float32 normalisés 0..1 (NaN -> 0)."""
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        mono = _normalize01(data)
        R = G = B = mono
    elif data.ndim == 3:
        # on accepte (H,W,3) ou (3,H,W)
        if data.shape[-1] == 3:
            R, G, B = data[..., 0], data[..., 1], data[..., 2]
        elif data.shape[0] == 3:
            R, G, B = data[0], data[1], data[2]
            R = np.ascontiguousarray(R.transpose(1, 2, 0))[:, :, 0] if R.ndim == 3 else R
        else:
            # fallback: moyenne
            mono = _normalize01(data.mean(axis=0))
            R = G = B = mono
        R = _normalize01(R)
        G = _normalize01(G)
        B = _normalize01(B)
    else:
        mono = _normalize01(data.reshape(data.shape[-2], data.shape[-1]))
        R = G = B = mono

    lum = np.nanmean(np.stack([R, G, B], axis=-1), axis=-1).astype(np.float32)
    lum[np.isnan(lum)] = 0.0
    R[np.isnan(R)] = 0.0
    G[np.isnan(G)] = 0.0
    B[np.isnan(B)] = 0.0
    return lum, R, G, B


def save_cropped_fits(
    in_path,
    rect,
    out_suffix="_cropped",
    altaz_cleanup=False,
    altaz_margin=5.0,
    altaz_decay=0.15,
    altaz_use_nan=False,
):
    y0, x0, y1, x1 = rect
    coverage_full = None
    spatial_shape: tuple[int, int] | None = None
    # Use memmap=False so the FITS file handle is fully released before
    # potentially overwriting the source file (Windows needs the file closed).
    with fits.open(in_path, mode="readonly", memmap=False) as hdul:
        data = np.asarray(hdul[0].data)
        header = hdul[0].header.copy()
        if data.ndim >= 2:
            spatial_shape = (int(data.shape[-2]), int(data.shape[-1]))
        coverage_full = _maybe_extract_coverage_from_hdul(hdul, expected_shape=spatial_shape)

    if coverage_full is None and spatial_shape is not None:
        coverage_full = _maybe_load_sidecar_coverage(in_path, expected_shape=spatial_shape)

    if data.ndim == 2:
        cropped = data[y0:y1, x0:x1]
    elif data.ndim == 3:
        if data.shape[-1] == 3:
            cropped = data[y0:y1, x0:x1, :]
        elif data.shape[0] == 3:
            cropped = data[:, y0:y1, x0:x1]
        else:
            cropped = data[:, y0:y1, x0:x1]
    else:
        cropped = data

    coverage_cropped = None
    if coverage_full is not None:
        try:
            coverage_cropped = coverage_full[y0:y1, x0:x1]
        except Exception as exc:
            logger.debug("Coverage crop failed: %s", exc)
            coverage_cropped = None

    alpha_mask = None
    if altaz_cleanup:
        fill_value = np.nan if altaz_use_nan else 0.0
        hard_threshold = 1e-3
        mask_kwargs: dict[str, Any] = {
            "margin_percent": altaz_margin,
            "decay_ratio": altaz_decay,
            "fill_value": fill_value,
            "return_mask": True,
            "hard_threshold": hard_threshold,
        }
        if coverage_cropped is not None:
            mask_kwargs.update(
                coverage=coverage_cropped,
                min_coverage_abs=MIN_COVERAGE_ABS_DEFAULT,
                min_coverage_frac=MIN_COVERAGE_FRAC_DEFAULT,
                morph_open_px=MORPH_OPEN_PX_DEFAULT,
            )

        masked_result = mask_altaz_artifacts(
            cropped,
            **mask_kwargs,
        )
        if isinstance(masked_result, tuple):
            cropped, mask = masked_result
        else:
            cropped, mask = masked_result, None

        if mask is not None:
            mask = np.asarray(mask, dtype=np.float32, copy=False)
            if altaz_use_nan:
                cropped = np.where(mask <= hard_threshold, np.nan, cropped)
            # NOTE [DO-NOT-REMOVE (alpha mask)]: this alpha map is consumed
            # downstream for cosmetic editing / transparency in mosaics.
            alpha_mask = np.clip((mask * 255.0).round(), 0, 255).astype(np.uint8)

    # Mise à jour rapide du CRPIX si présent (optionnel)
    if "CRPIX1" in header and "CRPIX2" in header:
        header["CRPIX1"] = header.get("CRPIX1", 0.0) - x0
        header["CRPIX2"] = header.get("CRPIX2", 0.0) - y0

    base, ext = os.path.splitext(in_path)
    out_path = f"{base}{out_suffix}{ext}"

    settings = _get_altaz_alpha_settings()
    hdus = []
    primary = fits.PrimaryHDU(cropped, header)
    primary.header["ALTZCLN"] = (bool(altaz_cleanup), "Alt-Az cleanup applied")
    primary.header["ALTZNAN"] = (bool(altaz_use_nan), "NaN outside FoV")
    primary.header["ALPHAEXT"] = (int(alpha_mask is not None and settings["fits"]), "Alpha mask ext present")
    hdus.append(primary)
    if alpha_mask is not None and settings["fits"]:
        alpha_hdu = fits.ImageHDU(alpha_mask, name="ALPHA")
        alpha_hdu.header["ALPHADSC"] = ("1=opaque(in), 0=transparent(out)", "")
        hdus.append(alpha_hdu)
    fits.HDUList(hdus).writeto(out_path, overwrite=True)
    logger.info(f"[lecropper] Saved FITS with ALPHA={alpha_mask is not None and settings['fits']} → {out_path}")

    if alpha_mask is not None and settings["sidecar"]:
        try:
            from PIL import Image

            img = np.asarray(cropped, dtype=np.float32)
            finite = np.isfinite(img)
            if finite.any():
                p1, p99 = np.nanpercentile(img[finite], [1, 99])
                scale = max(p99 - p1, 1e-6)
                img = np.clip((img - p1) / scale, 0.0, 1.0)
            else:
                img = np.zeros_like(img, dtype=np.float32)
            img8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            if img8.ndim == 2:
                rgb = np.stack([img8, img8, img8], axis=-1)
            elif img8.ndim == 3 and img8.shape[-1] == 3:
                rgb = img8
            else:
                rgb = np.stack([img8[..., 0]] * 3, axis=-1)
            rgba = Image.fromarray(rgb, mode="RGB")
            A = Image.fromarray(alpha_mask, mode="L")
            rgba.putalpha(A)
            side_ext = settings["format"] or "png"
            side_path = os.path.splitext(out_path)[0] + f".alpha.{side_ext}"
            rgba.save(side_path)
            logger.info(f"Saved alpha sidecar {side_path}")
        except Exception:
            pass

    return out_path


# ---------------------------------- GUI --------------------------------------

class AutoCropApp:
    def __init__(self, root):
        self.root = root
        root.title("ZeMosaic Auto-Crop (signal-based)")
        root.geometry("1200x720")

        top = tk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        tk.Label(top, text="Input directory:").pack(side=tk.LEFT)
        self.dir_var = tk.StringVar(value=os.getcwd())
        self.dir_entry = tk.Entry(top, textvariable=self.dir_var, width=65)
        self.dir_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Browse...", command=self.browse_dir).pack(side=tk.LEFT, padx=4)

        tk.Label(top, text="Band px").pack(side=tk.LEFT, padx=(10, 2))
        self.band_var = tk.StringVar(value="32")
        tk.Entry(top, textvariable=self.band_var, width=5).pack(side=tk.LEFT)

        tk.Label(top, text="k:sigma").pack(side=tk.LEFT, padx=(10, 2))
        self.ks_var = tk.StringVar(value="2.0")
        tk.Entry(top, textvariable=self.ks_var, width=5).pack(side=tk.LEFT)

        tk.Label(top, text="min run").pack(side=tk.LEFT, padx=(10, 2))
        self.minrun_var = tk.StringVar(value="2")
        tk.Entry(top, textvariable=self.minrun_var, width=5).pack(side=tk.LEFT)

        tk.Label(top, text="margin px").pack(side=tk.LEFT, padx=(10, 2))
        self.margin_var = tk.StringVar(value="8")
        tk.Entry(top, textvariable=self.margin_var, width=5).pack(side=tk.LEFT)

        self.altaz_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Alt-Az cleanup", variable=self.altaz_var).pack(side=tk.LEFT, padx=8)

        tk.Label(top, text="AltAz margin %").pack(side=tk.LEFT, padx=(4, 2))
        self.altaz_margin_var = tk.StringVar(value="5.0")
        tk.Entry(top, textvariable=self.altaz_margin_var, width=5).pack(side=tk.LEFT)

        tk.Label(top, text="AltAz decay").pack(side=tk.LEFT, padx=(4, 2))
        self.altaz_decay_var = tk.StringVar(value="0.15")
        tk.Entry(top, textvariable=self.altaz_decay_var, width=5).pack(side=tk.LEFT)

        self.altaz_nan_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Alt-Az → NaN", variable=self.altaz_nan_var).pack(side=tk.LEFT, padx=6)

        tk.Button(top, text="Analyze", command=self.analyze).pack(side=tk.LEFT, padx=8)
        tk.Button(top, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=4)

        self.replace_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Replace files", variable=self.replace_var).pack(side=tk.LEFT, padx=8)

        tk.Button(top, text="Apply Crop", command=self.apply_crop).pack(side=tk.LEFT, padx=4)

        mid = tk.Frame(root)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Liste des fichiers
        left = tk.Frame(mid, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox = tk.Listbox(left, width=40)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        self.scroll = tk.Scrollbar(left, orient=tk.VERTICAL, command=self.listbox.yview)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=self.scroll.set)

        # Figure Matplotlib pour l'aperçu
        right = tk.Frame(mid)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.results = {}   # path -> (y0,x0,y1,x1) ou None
        self.files = []

        self.status = tk.StringVar(value="Ready.")
        tk.Label(root, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

        self.refresh_files()

    # ------------------ actions GUI ------------------

    def browse_dir(self):
        d = filedialog.askdirectory(initialdir=self.dir_var.get() or os.getcwd())
        if d:
            self.dir_var.set(d)
            self.refresh_files()

    def refresh_files(self):
        d = self.dir_var.get()
        pats = ["*.fits", "*.fit", "*.fts"]
        files = []
        for p in pats:
            files += sorted(glob.glob(os.path.join(d, p)))
        self.files = files
        self.listbox.delete(0, tk.END)
        for f in files:
            self.listbox.insert(tk.END, os.path.basename(f))
        self.status.set(f"Found {len(files)} FITS files.")

    def _get_altaz_params(self):
        enabled = bool(self.altaz_var.get())
        try:
            margin = float(self.altaz_margin_var.get())
        except Exception:
            margin = 5.0
        margin = float(np.clip(margin, 0.0, 50.0))

        try:
            decay = float(self.altaz_decay_var.get())
        except Exception:
            decay = 0.15
        decay = max(0.0, decay)

        use_nan = bool(self.altaz_nan_var.get())
        return enabled, margin, decay, use_nan

    def analyze(self):
        if not self.files:
            self.status.set("No FITS files.")
            return

        try:
            band = int(float(self.band_var.get()))
        except:
            band = 32
        try:
            ks = float(self.ks_var.get())
        except:
            ks = 2.0
        try:
            margin = int(float(self.margin_var.get()))
        except:
            margin = 8

        analyzed = 0
        self.results.clear()
        self.listbox.delete(0, tk.END)
        for path in self.files:
            try:
                lum, R, G, B = load_fits_rgb(path)
                rect = detect_autocrop_rgb(lum, R, G, B, band_px=band, k_sigma=ks, margin_px=margin)
                self.results[path] = rect
                label = f"OK  {os.path.basename(path)}"
                self.listbox.insert(tk.END, label)
            except Exception as e:
                print(f"[ERROR] {path}: {e}\n{traceback.format_exc()}", file=sys.stderr)
                self.results[path] = None
                label = f"ERR {os.path.basename(path)}"
                self.listbox.insert(tk.END, label)
            analyzed += 1

        self.status.set(f"Analyzed {analyzed} files.")
        if analyzed:
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(0)
            self.on_select(None)

    def on_select(self, _evt):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self.files):
            return
        path = self.files[idx]
        rect = self.results.get(path)
        self.show_preview(path, rect)

    def show_preview(self, path, rect):
        self.ax.clear()
        self.ax.set_title(os.path.basename(path))
        try:
            lum, R, G, B = load_fits_rgb(path)
            # on affiche la luminance (low-stretch pour voir les bords sombres)
            disp = np.power(np.clip(lum, 0, 1), 0.5)
            altaz_enabled, altaz_margin, altaz_decay, _ = self._get_altaz_params()
            if altaz_enabled:
                disp = mask_altaz_artifacts(
                    disp,
                    margin_percent=altaz_margin,
                    decay_ratio=altaz_decay,
                    fill_value=0.0,
                )
            self.ax.imshow(disp, origin="upper", cmap="gray")
            if rect is not None:
                y0, x0, y1, x1 = rect
                self.ax.plot([x0, x1, x1, x0, x0],
                             [y0, y0, y1, y1, y0],
                             linewidth=1.5)
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Preview error:\n{e}", ha="center", va="center", transform=self.ax.transAxes)
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    def export_csv(self):
        if not self.results:
            messagebox.showinfo("Export CSV", "No results to export.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="autocrop_rects.csv")
        if not out:
            return
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "y0", "x0", "y1", "x1"])
            for p in self.files:
                r = self.results.get(p)
                if r is None:
                    w.writerow([p, "", "", "", ""])
                else:
                    y0, x0, y1, x1 = r
                    w.writerow([p, y0, x0, y1, x1])
        self.status.set(f"CSV exported to {out}")

    def apply_crop(self):
        if not self.results:
            messagebox.showinfo("Apply Crop", "No results to apply.")
            return
        ok, err = 0, 0
        altaz_enabled, altaz_margin, altaz_decay, altaz_use_nan = self._get_altaz_params()
        for p in self.files:
            rect = self.results.get(p)
            if not rect:
                continue
            try:
                suffix = "" if self.replace_var.get() else "_cropped"
                outp = save_cropped_fits(
                    p,
                    rect,
                    out_suffix=suffix,
                    altaz_cleanup=altaz_enabled,
                    altaz_margin=altaz_margin,
                    altaz_decay=altaz_decay,
                    altaz_use_nan=altaz_use_nan,
                )
                ok += 1
            except Exception as e:
                print(f"[CROP ERROR] {p}: {e}\n{traceback.format_exc()}", file=sys.stderr)
                err += 1
        mode = "replaced" if self.replace_var.get() else "saved"
        self.status.set(f"Cropped: {ok} files ({mode}), errors: {err}")
        messagebox.showinfo(
            "Apply Crop",
            f"Cropped: {ok} files ({mode})\nErrors: {err}"
        )


# --------------------------------- main --------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoCropApp(root)
    root.mainloop()
