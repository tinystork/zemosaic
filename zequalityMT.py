#!/usr/bin/env python3
# zequalityMT.py — Master Tile Quality Gate (GUI/CLI) — v2 (edge-friendly)
"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                               ║
║                                                                                   ║
║ Auteur  : Tinystork, seigneur des couteaux à beurre (aka Tristan Nauleau)         ║
║ Partenaire : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System   ║
║              (aka ChatGPT, Grand Maître du ciselage de code)                      ║
║                                                                                   ║
║ Licence : GNU General Public License v3.0 (GPL-3.0)                               ║
║                                                                                   ║
║ Description :                                                                     ║
║   Ce programme a été forgé à la lueur des pixels et de la caféine,                ║
║   dans le but noble de transformer des nuages de photons en art                   ║
║   astronomique. Si vous l’utilisez, pensez à dire “merci”,                        ║
║   à lever les yeux vers le ciel, ou à citer Tinystork et J.A.R.V.I.S.             ║
║   (le karma des développeurs en dépend).                                          ║
║                                                                                   ║
║ Avertissement :                                                                   ║
║   Aucune IA ni aucun couteau à beurre n’a été blessé durant le                    ║
║   développement de ce code.                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════╝


╔═══════════════════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                               ║
║                                                                                   ║
║ Author  : Tinystork, Lord of the Butter Knives (aka Tristan Nauleau)              ║
║ Partner : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System      ║
║           (aka ChatGPT, Grand Master of Code Chiseling)                           ║
║                                                                                   ║
║ License : GNU General Public License v3.0 (GPL-3.0)                               ║
║                                                                                   ║
║ Description:                                                                      ║
║   This program was forged under the sacred light of pixels and                    ║
║   caffeine, with the noble intent of turning clouds of photons into               ║
║   astronomical art. If you use it, please consider saying “thanks,”               ║
║   gazing at the stars, or crediting Tinystork and J.A.R.V.I.S. —                  ║
║   developer karma depends on it.                                                  ║
║                                                                                   ║
║ Disclaimer:                                                                       ║
║   No AIs or butter knives were harmed in the making of this code.                 ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import argparse
import json
import shutil
import threading
from pathlib import Path

try:
    from core.path_helpers import safe_path_exists
except Exception:  # pragma: no cover - standalone usage outside ZeMosaic
    def safe_path_exists(path, *, expanduser: bool = True):
        """Best-effort path existence check when core helpers are unavailable."""

        if path is None:
            return False
        try:
            path_obj = Path(path)
        except Exception:
            return False
        if expanduser:
            try:
                path_obj = path_obj.expanduser()
            except Exception:
                pass
        try:
            return path_obj.exists()
        except Exception:
            return False

import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:  # pragma: no cover - enables CLI when tkinter missing (Mac builds w/o TCL/TK)
    tk = None
    ttk = None
    filedialog = None
    messagebox = None

# NOTE: lorsque ce fichier est inspecté dans le projet ZeMosaic, ne le modifie que pour assurer cette compatibilité;
# hors de ZeMosaic il doit rester autonome et exécutable par lui-même.

# --- Optional deps ---
try:
    from astropy.io import fits
except Exception:
    fits = None

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None

try:
    from scipy.signal import convolve2d as _conv2d
except Exception:
    _conv2d = None

try:
    from scipy.ndimage import (
        gaussian_laplace as _gauss_laplace,
        binary_erosion as _binary_erosion,
        binary_dilation as _binary_dilation,
    )
except Exception:
    _gauss_laplace = None
    _binary_erosion = None
    _binary_dilation = None

IMAGE_EXTENSIONS = (".fits", ".fit", ".fts", ".png", ".tif", ".tiff")


def _expand_path(pathish) -> Path | None:
    """Return a Path with user expansion, tolerating bad inputs."""

    if pathish is None:
        return None
    try:
        path_obj = Path(pathish)
    except Exception:
        return None
    try:
        return path_obj.expanduser()
    except Exception:
        return path_obj


def _is_supported_image(path_obj: Path) -> bool:
    try:
        return path_obj.suffix.lower() in IMAGE_EXTENSIONS
    except Exception:
        return False


def _collect_image_files(directory: Path) -> list[Path]:
    """Return all supported images under *directory* (non-recursive on errors)."""

    files: list[Path] = []
    if not directory.exists():
        return files
    try:
        iterator = directory.rglob("*")
    except Exception:
        return files
    for candidate in iterator:
        try:
            if candidate.is_file() and _is_supported_image(candidate):
                files.append(candidate)
        except Exception:
            continue
    return files


def _common_parent(paths: list[Path]) -> Path:
    """Best-effort parent directory shared by *paths*."""

    if not paths:
        return Path.cwd()
    anchor = paths[0].expanduser().resolve(strict=False).parent
    shared_parts = list(anchor.parts)
    for path in paths[1:]:
        candidate = path.expanduser().resolve(strict=False).parent
        new_parts: list[str] = []
        for left, right in zip(shared_parts, candidate.parts):
            if left != right:
                break
            new_parts.append(left)
        if not new_parts:
            shared_parts = []
            break
        shared_parts = new_parts
    if shared_parts:
        return Path(*shared_parts)
    return anchor


def _ensure_unique_destination(target: Path) -> Path:
    """Return a path that does not exist by suffixing _N if needed."""

    candidate = target
    index = 1
    while safe_path_exists(candidate, expanduser=False):
        candidate = target.with_name(f"{target.stem}_{index}{target.suffix}")
        index += 1
    return candidate


class BufferPool:
    def __init__(self):
        self._pool = {}

    def borrow(self, shape, dtype):
        shape = tuple(shape)
        dt = np.dtype(dtype)
        key = (shape, dt.str)
        bucket = self._pool.setdefault(key, [])
        if bucket:
            return bucket.pop()
        return np.empty(shape, dtype=dt)

    def release(self, arr):
        key = (arr.shape, arr.dtype.str)
        self._pool.setdefault(key, []).append(arr)


_buffer_pool = BufferPool()

__all__ = ["quality_metrics", "run_cli", "QualityGUI", "main"]

_zemosaic_warning_shown = False

def _alert_zemosaic_missing(root):
    global _zemosaic_warning_shown
    if _zemosaic_warning_shown:
        return
    _zemosaic_warning_shown = True
    current = Path.cwd() / "run_zemosaic.py"
    if safe_path_exists(current, expanduser=False):
        return
    if messagebox is None:
        return
    try:
        messagebox.showwarning("ZeQualityMT", "ZeMosaic missing stand alone use only")
    except Exception:
        pass


# ---------------- IO ----------------
def _read_image(path):
    path_obj = _expand_path(path)
    if path_obj is None:
        raise ValueError(f"Invalid path: {path!r}")
    ext = path_obj.suffix.lower()
    if ext in (".fits", ".fit", ".fts") and fits:
        with fits.open(path_obj, memmap=True, do_not_scale_image_data=True) as hdul:
            arr = np.asarray(hdul[0].data, dtype=np.float32)
    else:
        if Image is None:
            raise RuntimeError("Pillow manquant pour lire PNG/TIFF.")
        arr = np.asarray(Image.open(path_obj).convert("RGB"), dtype=np.float32) / 255.0
    # HxW or HxWxC → HxWxC float32
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[0] in (1, 3) and arr.ndim == 3 and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.ascontiguousarray(arr, dtype=np.float32)


# --------------- Helpers ---------------
def _sobel_mag(lum):
    # Sobel 3x3, fallback simple si scipy indisponible
    if _conv2d is None:
        gx = np.zeros_like(lum); gy = np.zeros_like(lum)
        gx[:, 1:-1] = (lum[:, 2:] - lum[:, :-2]) * 0.5
        gy[1:-1, :] = (lum[2:, :] - lum[:-2, :]) * 0.5
        return np.hypot(gx, gy)
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], np.float32)
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)
    gx = _conv2d(lum, kx, mode="same", boundary="symm")
    gy = _conv2d(lum, ky, mode="same", boundary="symm")
    return np.hypot(gx, gy)

def _erode(mask, px):
    if px <= 0 or _binary_erosion is None:
        return mask
    return _binary_erosion(mask, structure=np.ones((px, px)))

def _dilate(mask, px):
    if px <= 0 or _binary_dilation is None:
        return mask
    return _binary_dilation(mask, structure=np.ones((px, px)))

def _center_sky_stats(lum, edge_band, sky_mask):
    H, W = lum.shape
    b = int(max(8, min(edge_band, min(H, W)//3)))
    ctr = np.zeros_like(lum, bool); ctr[b:-b, b:-b] = True
    S = sky_mask & ctr
    x = lum[S] if np.count_nonzero(S) > 0 else lum[ctr]
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med))) + 1e-6
    sig = 1.4826 * mad
    return med, sig

def _object_mask_multiscale(lum, sigmas=(1.2, 2.4, 4.0), perc=98.8, grow_px=9, out=None):
    """Object-agnostic extended-structure mask (galaxies, nebulae, comets)."""
    if _gauss_laplace is None:
        if out is not None:
            out.fill(False)
            return out
        return np.zeros_like(lum, bool)
    acc = _buffer_pool.borrow(lum.shape, np.float32)
    acc.fill(0.0)
    for s in sigmas:
        LoG = -_gauss_laplace(lum.astype(np.float32), sigma=s)
        thr = np.nanpercentile(LoG, perc)
        acc[:] = np.maximum(acc, (LoG - thr).astype(np.float32))
    obj = acc > 0
    _buffer_pool.release(acc)
    if grow_px > 0:
        obj = _dilate(obj, grow_px)
    if out is not None:
        np.copyto(out, obj)
        return out
    return obj

def _sector_completeness(edge_mask, sky_mask, sectors=8):
    """Pire fraction de ciel utile par secteur angulaire de l’anneau de bord (0 bon .. 1 mauvais)."""
    H, W = sky_mask.shape
    cy, cx = (H-1)/2.0, (W-1)/2.0
    y, x = np.indices((H, W))
    ang = (np.arctan2(y - cy, x - cx) + np.pi)  # 0..2π
    worst = 1.0
    e = edge_mask
    for k in range(sectors):
        a0 = 2*np.pi*k/sectors
        a1 = 2*np.pi*(k+1)/sectors
        sel = e & (ang >= a0) & (ang < a1)
        if np.any(sel):
            frac = float(np.count_nonzero(sky_mask & sel)) / float(np.count_nonzero(sel))
            worst = min(worst, frac)
    return 1.0 - worst  # 0=parfait, 1=au moins un secteur vide

def _trailiness(lum, bg_mask):
    """Approximate elongation of bright blobs (0 round .. ~1 elongated)."""
    if _gauss_laplace is None:
        return 0.0
    LoG = -_gauss_laplace(lum, sigma=1.2)
    t = np.nanpercentile(LoG[bg_mask], 99.2) if np.any(bg_mask) else np.nanpercentile(LoG, 99.2)
    bw = (LoG > t) & bg_mask
    ys, xs = np.nonzero(bw)
    if ys.size < 50:
        return 0.0
    y = ys.astype(np.float32); x = xs.astype(np.float32)
    y -= y.mean(); x -= x.mean()
    cov = np.cov(np.vstack([x, y]))
    w, _ = np.linalg.eigh(cov); w = np.sort(w)
    if w[-1] <= 1e-9:
        return 0.0
    return float(1.0 - (w[0] / w[-1]))

def _corner_emptiness(bg_mask, edge_band):
    """
    CER — Corner Emptiness Ratio : mesure simple des coins vides/diagonaux.
    0 = coins remplis par du ciel utile ; 1 = coins vides.
    """
    H, W = bg_mask.shape
    b = int(max(8, min(edge_band, min(H, W)//3)))
    # quatre carrés aux coins (taille b x b)
    quads = [
        (slice(0, b), slice(0, b)),
        (slice(0, b), slice(W-b, W)),
        (slice(H-b, H), slice(0, b)),
        (slice(H-b, H), slice(W-b, W)),
    ]
    worst = 0.0
    for ys, xs in quads:
        q = bg_mask[ys, xs]
        if q.size == 0: 
            continue
        frac = float(np.count_nonzero(q)) / float(q.size)
        worst = max(worst, 1.0 - frac)  # manque de ciel utile
    return worst


# --------------- Metrics ---------------
def quality_metrics(arr, edge_band=64, k_sigma=2.5, erode_px=3, galaxy_protect=True):
    """
    Retourne un dict de métriques + score (edge-friendly, object-agnostic):
      CC, ED, SC, NBR, BU, SDED, CEC, EOC, TRL, CER, score
      hard_reject, hard_reject_reason, core_p99, edge_p99, edge_ratio
    """
    H, W, _ = arr.shape
    min_dim = int(min(H, W))
    hard_reject_reason = ""
    core_p99 = float("nan")
    edge_p99 = float("nan")
    edge_ratio = float("nan")
    borrowed = []
    def borrow(shape, dtype):
        buf = _buffer_pool.borrow(shape, dtype)
        borrowed.append(buf)
        return buf

    try:
        rgb = arr[..., :3]
        lum = borrow((H, W), np.float32)
        np.add(rgb[..., 0], rgb[..., 1], out=lum)
        np.add(lum, rgb[..., 2], out=lum)
        lum *= (1.0 / 3.0)

        # (0) objet étendu général
        obj_mask = borrow((H, W), bool)
        if galaxy_protect:
            _object_mask_multiscale(lum, sigmas=(1.2, 2.4, 4.0), perc=98.8, grow_px=9, out=obj_mask)
        else:
            obj_mask.fill(False)
        EOC = float(np.count_nonzero(obj_mask)) / float(H*W)

        # Edge/center masks
        b = int(max(8, min(edge_band, min(H, W)//3)))
        edge_mask = borrow((H, W), bool)
        edge_mask.fill(False)
        edge_mask[:b, :] = edge_mask[-b:, :] = edge_mask[:, :b] = edge_mask[:, -b:] = True
        center_mask = borrow((H, W), bool)
        np.logical_not(edge_mask, out=center_mask)

        # (0-bis) Hard reject : edge blow-up / tuile dégénérée
        # Construire une intensité robuste (absmax) et comparer bord vs coeur.
        plane = borrow((H, W), np.float32)
        tmp = borrow((H, W), np.float32)
        np.abs(rgb[..., 0], out=plane)
        np.abs(rgb[..., 1], out=tmp)
        np.fmax(plane, tmp, out=plane)
        np.abs(rgb[..., 2], out=tmp)
        np.fmax(plane, tmp, out=plane)

        core_vals = plane[center_mask]
        edge_vals = plane[edge_mask]
        core_finite = core_vals[np.isfinite(core_vals)]
        edge_finite = edge_vals[np.isfinite(edge_vals)]
        eps = 1e-6
        abs_floor = 1e-5
        ratio_thr = 1e6
        abs_cap = 1e25
        if core_finite.size == 0 or edge_finite.size == 0:
            hard_reject_reason = "no_finite_core_or_edge"
        else:
            core_p99 = float(np.nanpercentile(core_finite, 99))
            edge_p99 = float(np.nanpercentile(edge_finite, 99))
            edge_ratio = float(edge_p99 / max(core_p99, eps))
            try:
                plane_max = float(np.nanmax(plane))
            except Exception:
                plane_max = float("nan")
            if np.isfinite(plane_max) and plane_max > abs_cap:
                hard_reject_reason = "abs_cap_exceeded"
            elif edge_ratio > ratio_thr and edge_p99 > abs_floor:
                hard_reject_reason = "edge_blowup"
        if not hard_reject_reason and min_dim < 128:
            hard_reject_reason = "small_dim"
        hard_reject = bool(hard_reject_reason)

        # (A) Masque ciel (sans objet), stats ancrées au centre
        g_med = float(np.nanmedian(lum[~obj_mask])) if np.any(~obj_mask) else float(np.nanmedian(lum))
        g_mad = float(np.nanmedian(np.abs(lum[~obj_mask] - g_med))) + 1e-6
        g_sig = 1.4826 * g_mad
        sky0 = borrow((H, W), bool)
        lower = g_med - (k_sigma + 0.5) * g_sig
        upper = g_med + (k_sigma + 0.5) * g_sig
        np.greater(lum, lower, out=sky0)
        sky0 &= (lum < upper)
        sky0 &= ~obj_mask
        bg_mask = _erode(sky0, erode_px)

        m_ctr, s_ctr = _center_sky_stats(lum, edge_band, sky_mask=bg_mask)
        useful = borrow((H, W), bool)
        thr = m_ctr - 0.75 * s_ctr
        np.greater(lum, thr, out=useful)
        useful &= ~obj_mask

        # (1) Coverage du ciel utile
        CC = float(np.count_nonzero(useful)) / float(H*W)

        # (2) Edge disparity (ciel uniquement)
        edge_bg = lum[edge_mask & bg_mask]
        ctr_bg  = lum[center_mask & bg_mask]
        m_edge = float(np.nanmedian(edge_bg)) if edge_bg.size else m_ctr
        rob = float(np.nanmedian(np.abs(ctr_bg - np.nanmedian(ctr_bg)))) + 1e-6
        ED = min(5.0, abs(m_edge - m_ctr) / rob)

        # (3) Sector Completeness (0 bon .. 1 mauvais)
        SC = _sector_completeness(edge_mask, sky_mask=useful, sectors=8)

        # (4) NaN/zero border ratio
        thr_zero = m_ctr - 2.5*s_ctr
        # Bad pixel = non-fini OR <= thr_zero OR absurdement haut vs coeur (cap adaptatif)
        if np.isfinite(core_p99):
            high_cap = max(core_p99, eps) * 1e8
        else:
            high_cap = float("inf")
        nan_or_bad = ~np.isfinite(lum) | (lum <= thr_zero)
        if np.isfinite(high_cap):
            nan_or_bad |= (plane >= high_cap)
        NBR = float(np.count_nonzero(nan_or_bad & edge_mask)) / float(edge_mask.sum())

        # (4-bis) Coins vides / diagonaux
        CER = _corner_emptiness(bg_mask=useful, edge_band=edge_band)

        # (5) Background uniformity (Sobel) sur ciel
        sob = _sobel_mag(np.where(bg_mask, lum, m_ctr))
        BU = min(0.02, float(np.nanmedian(sob[bg_mask]))) / 0.02 if np.any(bg_mask) else 0.0

        # (6) Star density edge deficit (SDED) — sur ciel uniquement
        if _gauss_laplace is not None and np.any(bg_mask):
            LoG = -_gauss_laplace(lum, sigma=1.2)
            t = np.nanpercentile(LoG[bg_mask], 99.2)
            star_mask = (LoG > t) & bg_mask
            stars_edge = np.count_nonzero(star_mask & edge_mask)
            stars_center = np.count_nonzero(star_mask & center_mask)
            area_edge = max(1, np.count_nonzero(edge_mask & bg_mask))
            area_center = max(1, np.count_nonzero(center_mask & bg_mask))
            dens_edge = stars_edge / area_edge
            dens_center = stars_center / area_center
            SDED = float(max(0.0, (dens_center - dens_edge) / (dens_center + 1e-6)))
        else:
            SDED = 0.0

        # (7) Chroma edge cast (CEC) — sur ciel uniquement
        u = rgb[...,0] - rgb[...,1]; v = rgb[...,2] - rgb[...,1]
        chroma = np.hypot(u, v)
        c_edge = float(np.nanmedian(chroma[edge_mask & bg_mask])) if np.any(edge_mask & bg_mask) else 0.0
        c_ctr  = float(np.nanmedian(chroma[center_mask & bg_mask])) if np.any(center_mask & bg_mask) else 0.0
        c_scale = np.nanmedian(np.abs(chroma[center_mask & bg_mask] - c_ctr)) + 1e-6
        CEC = float(max(0.0, (c_edge - c_ctr) / (3.0*c_scale)))

        # (8) Comet/elongation safeguard
        TRL = _trailiness(lum, bg_mask)

        # --- Adaptive weights (plus doux) ---
        if EOC < 0.20:
            w = dict(CC=0.18, ED=0.12, SC=0.22, NBR=0.10, BU=0.05, SDED=0.12, CEC=0.07, CER=0.14)
        elif EOC < 0.50:
            w = dict(CC=0.17, ED=0.13, SC=0.25, NBR=0.11, BU=0.05, SDED=0.09, CEC=0.08, CER=0.12)
        else:
            w = dict(CC=0.16, ED=0.14, SC=0.28, NBR=0.12, BU=0.06, SDED=0.00, CEC=0.09, CER=0.15)

        if TRL > 0.25:       # forte allongation → on assouplit SDED/ED
            w["SDED"] *= 0.3
            w["ED"]   *= 0.7

        score = (
            w["CC"]*(1.0-CC) +
            w["ED"]*min(1.0, ED/1.5) +
            w["SC"]*SC +
            w["NBR"]*NBR +
            w["BU"]*BU +
            w["SDED"]*SDED +
            w["CEC"]*min(1.0, CEC) +
            w["CER"]*CER
        )

        out = {
            "CC": CC, "ED": ED, "SC": SC, "NBR": NBR, "BU": BU,
            "SDED": SDED, "CEC": CEC, "EOC": EOC, "TRL": TRL, "CER": CER,
            "score": float(score),
            "hard_reject": int(hard_reject),
            "hard_reject_reason": str(hard_reject_reason),
            "core_p99": float(core_p99),
            "edge_p99": float(edge_p99),
            "edge_ratio": float(edge_ratio),
            "edge_band": b, "k_sigma": float(k_sigma), "erode_px": int(erode_px),
        }
        return out
    finally:
        for buf in borrowed:
            _buffer_pool.release(buf)

# --------------- CLI ---------------
def _accept_override(m, thr):
    """
    Règle “edge-friendly” : on sauve les bonnes tuiles même si le score est un poil > thr.
    """
    if m.get("hard_reject"):
        return False
    # bonne couverture, coins OK, bord pas trop décalé → on garde
    if m["CC"] >= 0.70 and m["CER"] < 0.35 and m["ED"] < 1.2:
        return True
    # tuiles très correctes globalement
    if m["CC"] >= 0.78 and m["SC"] < 0.28 and m["NBR"] < 0.10:
        return True
    # cas comète/galaxie brillante : on desserre un peu NBR/SC
    if m["TRL"] > 0.30 and m["CER"] < 0.40 and m["SC"] < 0.40:
        return True
    return False

def run_cli(paths, threshold=0.48, move_rejects=False, edge_band=64, k_sigma=2.5, erode_px=3, progress_callback=None):
    normalized_paths = [_expand_path(p) for p in paths]
    normalized_paths = [p for p in normalized_paths if p is not None]
    accepted: list[str] = []
    rejected: list[str] = []
    scores: dict[str, float] = {}
    rej_dir: Path | None = None
    if move_rejects and normalized_paths:
        root = _common_parent(normalized_paths)
        rej_dir = root / "rejected_by_quality"
        rej_dir.mkdir(parents=True, exist_ok=True)
    total = len(normalized_paths)
    for path_obj in normalized_paths:
        path_str = str(path_obj)
        keep = False
        s = float("nan")
        try:
            arr = _read_image(path_obj)
            m = quality_metrics(arr, edge_band=edge_band, k_sigma=k_sigma, erode_px=erode_px)
            s = m["score"]
            scores[path_str] = s
            keep = (not bool(m.get("hard_reject"))) and ((s <= threshold) or _accept_override(m, threshold))
            if keep:
                accepted.append(path_str)
            else:
                rejected.append(path_str)
                if rej_dir:
                    dst = _ensure_unique_destination(rej_dir / path_obj.name)
                    shutil.move(path_obj, dst)
            # Enrichir l’en-tête FITS si possible
            if path_obj.suffix.lower() in (".fits", ".fit", ".fts") and fits:
                try:
                    with fits.open(path_obj, mode="update") as hdul:
                        hdr = hdul[0].header
                        hdr["ZMT_QS"]  = (round(m["score"], 3), "ZeQuality score (0=good)")
                        hdr["ZMT_QBD"] = (int(not keep), "1 if auto-rejected")
                        hdr["ZMT_EOC"] = (round(m["EOC"], 3), "Extended Object Coverage")
                        hdr["ZMT_TRL"] = (round(m["TRL"], 3), "Trailiness index")
                        hdr["ZMT_CER"] = (round(m["CER"], 3), "Corner Emptiness Ratio")
                        hdul.flush()
                except Exception:
                    pass
        except Exception:
            rejected.append(path_str)
            scores[path_str] = float("nan")
        if progress_callback:
            try:
                idx = len(accepted) + len(rejected)
                progress_callback(idx, total, path_str, keep, s)
            except Exception:
                pass
    manifest = {"accepted": accepted, "rejected": rejected, "scores": scores}
    with open("quality_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# --------------- GUI ---------------
class QualityGUI:
    def __init__(self, master):
        self.root = master
        self.root.title("ZeQualityMT — Master Tile Quality Gate")
        self.edge_var = tk.IntVar(value=64)
        self.ks_var = tk.DoubleVar(value=2.5)
        self.erode_var = tk.IntVar(value=3)
        self.thr_var = tk.DoubleVar(value=0.48)  # seuil plus tolérant par défaut
        self.black_var = tk.DoubleVar(value=0.0)
        self.white_var = tk.DoubleVar(value=1.0)
        self._last_arr = None
        self._batch_running = False
        self._preview_lo = 0.0
        self._preview_hi = 1.0
        self.path = None

        frm = ttk.Frame(master, padding=8); frm.pack(fill="both", expand=True)
        row1 = ttk.Frame(frm); row1.pack(fill="x")
        self.open_btn = ttk.Button(row1, text="Open", command=self.open_file)
        self.open_btn.pack(side="left")
        self.batch_btn = ttk.Button(row1, text="Batch…", command=self.batch_folder)
        self.batch_btn.pack(side="left", padx=6)
        ttk.Label(row1, text="Threshold").pack(side="left", padx=(12, 2))
        ttk.Entry(row1, textvariable=self.thr_var, width=6).pack(side="left")
        self.score_lbl = ttk.Label(row1, text="score: —"); self.score_lbl.pack(side="right")
        self.status_lbl = ttk.Label(frm, text="Idle"); self.status_lbl.pack(fill="x", pady=(4, 0))

        for text, var, frm_min, frm_max in (
            ("Edge band (px)", self.edge_var, 8, 256),
            ("k·sigma", self.ks_var, 1.0, 5.0),
            ("Erode (px)", self.erode_var, 0, 12),
        ):
            lf = ttk.Labelframe(frm, text=text); lf.pack(fill="x", pady=4)
            s = ttk.Scale(lf, from_=frm_min, to=frm_max, variable=var, command=lambda *_: self.refresh())
            s.pack(fill="x", padx=6, pady=6)
        tone_frame = ttk.Labelframe(frm, text="Preview levels")
        tone_frame.pack(fill="x", pady=4)
        for label, var in (("Black level", self.black_var), ("White level", self.white_var)):
            row = ttk.Frame(tone_frame); row.pack(fill="x", padx=6, pady=2)
            ttk.Label(row, text=label).pack(side="left")
            ttk.Scale(row, from_=0.0, to=1.5, variable=var, command=self._reapply_tone).pack(side="right", fill="x", expand=True)

        self.progress = ttk.Progressbar(frm, mode="determinate")
        self.progress.pack(fill="x", pady=(2, 6))

        self.cv = tk.Canvas(frm, width=640, height=640, bg="black"); self.cv.pack(fill="both", expand=True, pady=6)

    def _update_preview(self, arr):
        if Image is None: return
        H, W, _ = arr.shape
        self._last_arr = arr
        black = float(self.black_var.get())
        white = float(self.white_var.get())
        if white <= black + 1e-4:
            white = black + 1e-4
        lo = getattr(self, "_preview_lo", 0.0)
        hi = getattr(self, "_preview_hi", 1.0)
        if hi <= lo + 1e-6:
            hi = lo + 1e-6
        norm = (arr - lo) / (hi - lo)
        vis = np.clip((norm - black) / (white - black), 0.0, 1.0)
        vis = (vis * 255).astype(np.uint8)
        im = Image.fromarray(vis)
        scale = 640 / max(1, max(H, W))
        im = im.resize((int(W * scale), int(H * scale)))
        self._tkim = ImageTk.PhotoImage(im)
        self.cv.delete("all"); self.cv.create_image(10, 10, anchor="nw", image=self._tkim)

    def _reapply_tone(self, *_):
        if self._last_arr is not None:
            self._update_preview(self._last_arr)

    def _set_preview_range(self, arr):
        try:
            lo = float(np.nanpercentile(arr, 1.0))
            hi = float(np.nanpercentile(arr, 99.5))
        except Exception:
            lo = float(np.nanmin(arr))
            hi = float(np.nanmax(arr))
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi): hi = lo + 1.0
        if hi <= lo + 1e-6:
            hi = lo + 1e-6
        self._preview_lo = lo
        self._preview_hi = hi

    def _update_progress(self, idx, total, path, kept, score):
        total = max(total, 1)
        self.progress["maximum"] = total
        self.progress["value"] = idx
        try:
            name = Path(path).name
        except Exception:
            name = str(path) if path else "<unknown>"
        status = f"{idx}/{total} {name}"
        status += " — accepted" if kept else " — rejected"
        self.status_lbl.config(text=status)

    def _build_progress_callback(self, total):
        def cb(idx, total, path, kept, score):
            self.root.after(0, self._update_progress, idx, total, path, kept, score)
        return cb

    def open_file(self):
        p = filedialog.askopenfilename(
            title="Open master tile",
            filetypes=[("FITS/Images", "*.fits *.fit *.fts *.png *.tif *.tiff")]
        )
        if not p: return
        self.path = p
        self.black_var.set(0.0)
        self.white_var.set(1.0)
        self.refresh()

    def refresh(self):
        if not self.path: return
        arr = _read_image(self.path)
        m = quality_metrics(
            arr,
            edge_band=int(self.edge_var.get()),
            k_sigma=float(self.ks_var.get()),
            erode_px=int(self.erode_var.get()),
        )
        self._set_preview_range(arr)
        self.score_lbl.config(
            text=(f"score {m['score']:.3f} | CC {m['CC']:.2f}  ED {m['ED']:.2f}  "
                  f"SC {m['SC']:.2f}  NBR {m['NBR']:.2f}  CER {m['CER']:.2f}  "
                  f"SDED {m['SDED']:.2f}  CEC {m['CEC']:.2f}  "
                  f"EOC {m['EOC']:.2f}  TRL {m['TRL']:.2f}")
        )
        self._update_preview(arr)

    def batch_folder(self):
        if self._batch_running:
            return
        d = filedialog.askdirectory(title="Pick folder of master tiles")
        if not d: return
        directory = _expand_path(d)
        if directory is None:
            messagebox.showwarning("Batch aborted", "Chemin invalide.")
            return
        paths = _collect_image_files(directory)
        if not paths:
            messagebox.showwarning("Batch aborted", "Aucun master tile trouvé dans ce dossier.")
            return
        run_kwargs = dict(
            threshold=float(self.thr_var.get()),
            move_rejects=True,
            edge_band=int(self.edge_var.get()),
            k_sigma=float(self.ks_var.get()),
            erode_px=int(self.erode_var.get()),
        )
        self._batch_running = True
        self._set_batch_buttons_enabled(False)
        self.status_lbl.config(text="Analyse en cours...")
        total = len(paths)
        run_kwargs["progress_callback"] = self._build_progress_callback(total)
        threading.Thread(target=self._run_batch_thread, args=(paths, run_kwargs), daemon=True).start()

    def _set_batch_buttons_enabled(self, enabled):
        for btn in (self.open_btn, self.batch_btn):
            if enabled:
                btn.state(["!disabled"])
            else:
                btn.state(["disabled"])

    def _run_batch_thread(self, paths, run_kwargs):
        try:
            res = run_cli(paths, **run_kwargs)
        except Exception as exc:
            self.root.after(0, self._on_batch_error, exc)
        else:
            self.root.after(0, self._on_batch_done, res)

    def _on_batch_done(self, res):
        self._batch_running = False
        self._set_batch_buttons_enabled(True)
        self.status_lbl.config(
            text=f"Batch terminé — accepted: {len(res['accepted'])}, rejected: {len(res['rejected'])}"
        )
        messagebox.showinfo(
            "Batch done",
            f"Accepted: {len(res['accepted'])}\nRejected: {len(res['rejected'])}\nquality_manifest.json written."
        )

    def _on_batch_error(self, exc):
        self._batch_running = False
        self._set_batch_buttons_enabled(True)
        self.status_lbl.config(text="Analyse interrompue")
        messagebox.showerror("Batch failed", f"Erreur pendant l'analyse : {exc}")


# --------------- Entrypoint ---------------
def main():
    ap = argparse.ArgumentParser(description="ZeQualityMT — Master Tile quality gate (edge-friendly)")
    ap.add_argument("paths", nargs="*", help="Master tiles or folder(s)")
    ap.add_argument("--threshold", type=float, default=0.48)
    ap.add_argument("--edge-band", type=int, default=64)
    ap.add_argument("--k-sigma", type=float, default=2.5)
    ap.add_argument("--erode-px", type=int, default=3)
    ap.add_argument("--move-rejects", action="store_true")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    need_gui = args.gui or not args.paths
    if need_gui and tk is None:
        ap.error(
            "Tkinter (Tcl/Tk) is missing in this Python build; install a Tcl/Tk-enabled Python "
            "or pass tile paths to run in CLI-only mode."
        )
    if need_gui:
        root = tk.Tk()
        _alert_zemosaic_missing(root)
        QualityGUI(root)
        root.mainloop()
        return

    # expand folders
    files: list[Path] = []
    for p in args.paths:
        path_obj = _expand_path(p)
        if path_obj is None:
            continue
        if path_obj.is_dir():
            files.extend(_collect_image_files(path_obj))
        else:
            files.append(path_obj)

    res = run_cli(
        files,
        threshold=args.threshold,
        move_rejects=args.move_rejects,
        edge_band=args.edge_band,
        k_sigma=args.k_sigma,
        erode_px=args.erode_px,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
