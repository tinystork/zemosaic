#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import sys
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from pathlib import Path
import math

try:
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
except Exception as exc:
    print("This tool requires astropy (io,fits,wcs). Install with: pip install astropy")
    raise

WCS_PREFIXES = (
    "CRVAL", "CRPIX", "CD", "PC", "CDELT", "CTYPE", "CROTA",
    "PV", "A_", "B_", "AP_", "BP_",  # SIP coefficient cards like A_0_2
)
WCS_SINGLE_KEYS = {
    "WCSAXES", "LONPOLE", "LATPOLE", "EQUINOX", "RADESYS", "RADECSYS",
    "CUNIT", "CUNIT1", "CUNIT2", "CUNIT3", "CUNIT4",
    "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER",  # SIP orders
    "A_DMAX", "B_DMAX", "AP_DMAX", "BP_DMAX",
}
# Some stacks/cameras also set ALT/AZ WCS in CTYPE3/4 — we drop CTYPE* anyway.

def header_has_wcs(hdr) -> bool:
    """Conservatively decide if a header 'has WCS'."""
    keys = set(hdr.keys())
    # Common telltales:
    telltales = {"WCSAXES", "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"}
    if keys.intersection(telltales):
        return True
    # Any matrix-like or SIP/pv terms?
    for k in keys:
        uk = str(k).upper()
        if uk.startswith(WCS_PREFIXES) or uk in WCS_SINGLE_KEYS:
            return True
    return False

def clean_wcs_header_inplace(hdr) -> int:
    """Remove WCS-related cards from a FITS header. Returns number of deletions."""
    to_delete = []
    for key in list(hdr.keys()):
        uk = str(key).upper()
        if uk in WCS_SINGLE_KEYS:
            to_delete.append(key)
            continue
        if uk.startswith(WCS_PREFIXES):
            to_delete.append(key)
            continue
        # Generic CUNITi (any axis)
        if uk.startswith("CUNIT"):
            to_delete.append(key)
    deleted = 0
    for k in to_delete:
        try:
            del hdr[k]
            deleted += 1
        except Exception:
            pass
    if deleted:
        # Keep record
        try:
            hdr.add_history(f"[WCS-CLEAN] Removed {deleted} WCS cards on {datetime.utcnow().isoformat()}Z")
        except Exception:
            pass
    return deleted


def _summarize_wcs_header(hdr):
    """Return a small WCS summary dict for sanity checks."""

    summary = {
        "ra_deg": None,
        "dec_deg": None,
        "scale_deg": None,
        "scale_arcsec": None,
        "width": None,
        "height": None,
        "error": None,
    }

    # Basic geometry from header, if present
    try:
        naxis1 = hdr.get("NAXIS1")
        naxis2 = hdr.get("NAXIS2")
        summary["width"] = int(naxis1) if naxis1 is not None else None
        summary["height"] = int(naxis2) if naxis2 is not None else None
    except Exception:
        summary["width"] = None
        summary["height"] = None

    # RA/DEC from CRVAL as a first guess
    try:
        ra = hdr.get("CRVAL1")
        dec = hdr.get("CRVAL2")
        if ra is not None:
            summary["ra_deg"] = float(ra)
        if dec is not None:
            summary["dec_deg"] = float(dec)
    except Exception:
        pass

    # Try to build a WCS object to refine RA/DEC and estimate pixel scale
    if "WCS" in globals() and WCS is not None:
        try:
            w = WCS(hdr)
            if getattr(w, "is_celestial", False):
                if summary["ra_deg"] is None or summary["dec_deg"] is None:
                    try:
                        crval = w.wcs.crval
                        if len(crval) >= 2:
                            summary["ra_deg"] = float(crval[0])
                            summary["dec_deg"] = float(crval[1])
                    except Exception:
                        pass
                scale_deg = None
                if "proj_plane_pixel_scales" in globals() and proj_plane_pixel_scales is not None:
                    try:
                        scales = proj_plane_pixel_scales(w)
                        if scales is not None and len(scales) >= 2:
                            vals = [abs(float(scales[0])), abs(float(scales[1]))]
                            vals = [v for v in vals if math.isfinite(v) and v > 0]
                            if vals:
                                scale_deg = sum(vals) / len(vals)
                    except Exception:
                        scale_deg = None
                if scale_deg is None:
                    # Fallback: approximate from CD matrix if present
                    try:
                        cd11 = float(hdr.get("CD1_1"))
                        cd12 = float(hdr.get("CD1_2"))
                        cd21 = float(hdr.get("CD2_1"))
                        cd22 = float(hdr.get("CD2_2"))
                        col0_n = math.hypot(cd11, cd21)
                        col1_n = math.hypot(cd12, cd22)
                        vals = [v for v in (col0_n, col1_n) if math.isfinite(v) and v > 0]
                        if vals:
                            scale_deg = sum(vals) / len(vals)
                    except Exception:
                        scale_deg = None
                if scale_deg is not None:
                    summary["scale_deg"] = scale_deg
                    summary["scale_arcsec"] = scale_deg * 3600.0
        except Exception as exc:
            summary["error"] = str(exc)

    return summary


def _angular_separation_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg) -> float:
    """Return great-circle separation in degrees between two sky positions."""

    try:
        r1 = math.radians(float(ra1_deg))
        d1 = math.radians(float(dec1_deg))
        r2 = math.radians(float(ra2_deg))
        d2 = math.radians(float(dec2_deg))
    except Exception:
        return float("nan")
    cos_sep = (
        math.sin(d1) * math.sin(d2)
        + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    )
    cos_sep = max(-1.0, min(1.0, cos_sep))
    try:
        return math.degrees(math.acos(cos_sep))
    except Exception:
        return float("nan")


def process_fits(
    path: str | Path,
    *,
    dry_run: bool,
    backup: bool,
    only_if_wcs: bool,
    all_hdus: bool,
) -> tuple[int, int]:
    """
    Returns (deleted_cards_total, edited_hdus_count).
    """
    deleted_total = 0
    edited_hdus = 0

    # Pre-check: skip non-files
    path_obj = Path(path).expanduser()
    if not path_obj.is_file():
        return (0, 0)

    with fits.open(path_obj, mode="update") as hdul:
        targets = range(len(hdul)) if all_hdus else [0]
        # Optional 'only if WCS' gate: check any target has WCS
        if only_if_wcs:
            ok = any(header_has_wcs(hdul[i].header) for i in targets)
            if not ok:
                return (0, 0)

        if not dry_run and backup:
            try:
                bak_path = path_obj.with_name(f"{path_obj.name}.bak")
                if not bak_path.exists():
                    hdul.writeto(bak_path, overwrite=False, output_verify="fix")
            except Exception:
                # Backup is best-effort; continue anyway
                pass

        for i in targets:
            hdr = hdul[i].header
            before = len(hdr)
            n = clean_wcs_header_inplace(hdr)
            if n > 0:
                deleted_total += n
                edited_hdus += 1

        if not dry_run and deleted_total > 0:
            hdul.flush()

    return (deleted_total, edited_hdus)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FITS WCS Cleaner")
        try:
            self.iconbitmap(default="")  # ignore if not found
        except Exception:
            pass
        self.geometry("900x600")
        self.minsize(800, 520)

        self.paths = []  # files selected
        self.var_dry = tk.BooleanVar(value=True)
        self.var_backup = tk.BooleanVar(value=True)
        self.var_only_wcs = tk.BooleanVar(value=True)
        self.var_all_hdus = tk.BooleanVar(value=False)
        self.var_recursive = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        ttk.Button(top, text="Add FITS files...", command=self.add_files).pack(side="left")
        ttk.Button(top, text="Add folder...", command=self.add_folder).pack(side="left", padx=(8,0))
        ttk.Button(top, text="Clear list", command=self.clear_list).pack(side="left", padx=(8,0))

        opts = ttk.Frame(self, padding=(8,4))
        opts.pack(fill="x")
        ttk.Checkbutton(opts, text="Dry-run (don’t modify files)", variable=self.var_dry).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(opts, text="Create .bak backups", variable=self.var_backup).grid(row=0, column=1, sticky="w", padx=(16,0))
        ttk.Checkbutton(opts, text="Process only if WCS present", variable=self.var_only_wcs).grid(row=0, column=2, sticky="w", padx=(16,0))
        ttk.Checkbutton(opts, text="Clean all HDUs (not just PRIMARY)", variable=self.var_all_hdus).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(opts, text="Recurse into subfolders", variable=self.var_recursive).grid(row=1, column=1, sticky="w", padx=(16,0))

        mid = ttk.Frame(self, padding=(8,4))
        mid.pack(fill="both", expand=True)
        cols = ("file","status","deleted","hdus")
        self.tree = ttk.Treeview(mid, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (520, 120, 80, 60)):
            self.tree.heading(c, text=c.upper())
            self.tree.column(c, width=w, anchor="w" if c=="file" else "center")
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        bottom = ttk.Frame(self, padding=8)
        bottom.pack(fill="x")
        self.progress = ttk.Progressbar(bottom, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", expand=True, side="left")
        ttk.Button(bottom, text="Scan selection", command=self.scan_selection).pack(side="left", padx=(8,0))
        ttk.Button(bottom, text="Sanity-check WCS", command=self.check_wcs_sanity).pack(side="left", padx=(8,0))
        ttk.Button(bottom, text="Clean selection", command=self.clean_selection).pack(side="left", padx=(8,0))

        logf = ttk.LabelFrame(self, text="Log", padding=6)
        logf.pack(fill="both", expand=True, padx=8, pady=(0,8))
        self.txt = tk.Text(logf, height=8, wrap="word")
        self.txt.pack(fill="both", expand=True)
        self._log("Ready.")

    def _log(self, msg: str):
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select FITS files",
            filetypes=[("FITS files", "*.fits *.fit *.fts"), ("All files", "*.*")]
        )
        if not paths:
            return
        for p in paths:
            path_str = str(Path(p).expanduser())
            if path_str not in self.paths:
                self.paths.append(path_str)
                self.tree.insert("", "end", values=(path_str, "", "", ""))

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select folder")
        if not folder:
            return
        root_dir = Path(folder).expanduser()
        if not root_dir.exists():
            self._log(f"[WARN] Folder '{root_dir}' does not exist.")
            return
        count = 0
        patterns = ("*.fits", "*.fit", "*.fts")
        recursive = self.var_recursive.get()
        for pattern in patterns:
            walker = root_dir.rglob(pattern) if recursive else root_dir.glob(pattern)
            for candidate in walker:
                if not candidate.is_file():
                    continue
                candidate_str = str(candidate)
                if candidate_str in self.paths:
                    continue
                self.paths.append(candidate_str)
                self.tree.insert("", "end", values=(candidate_str, "", "", ""))
                count += 1
            if not recursive:
                break
        self._log(f"Added {count} file(s) from folder.")

    def clear_list(self):
        self.paths.clear()
        for it in self.tree.get_children():
            self.tree.delete(it)
        self._log("Cleared list.")

    def _iter_selected_rows(self):
        sel = self.tree.selection()
        if not sel:
            sel = self.tree.get_children()
        for iid in sel:
            vals = self.tree.item(iid, "values")
            if not vals: continue
            yield iid, vals[0]

    def check_wcs_sanity(self):
        """Check basic WCS consistency across the selected files."""
        targets = list(self._iter_selected_rows())
        if not targets:
            messagebox.showinfo("Sanity check", "Nothing to check.")
            return

        if "WCS" not in globals() or WCS is None:
            messagebox.showerror(
                "Sanity check",
                "Astropy WCS is not available; cannot perform sanity check.",
            )
            return

        self.progress.configure(maximum=len(targets), value=0)
        summaries = []

        for i, (iid, path) in enumerate(targets, 1):
            path_obj = Path(path)
            try:
                with fits.open(path_obj, memmap=False) as hdul:
                    hdr = None
                    for h in range(len(hdul)):
                        if header_has_wcs(hdul[h].header):
                            hdr = hdul[h].header
                            break
                    if hdr is None:
                        self.tree.item(iid, values=(path, "NO-WCS", "", ""))
                        self._log(f"[SANITY] {path_obj}: no WCS found in any HDU.")
                        continue
                    summary = _summarize_wcs_header(hdr)
                    ra = summary.get("ra_deg")
                    dec = summary.get("dec_deg")
                    if ra is None or dec is None:
                        self.tree.item(iid, values=(path, "BAD-WCS", "", ""))
                        self._log(f"[SANITY] {path_obj}: unable to read WCS center (CRVAL1/2).")
                        continue
                    summary["iid"] = iid
                    summary["path"] = path
                    summaries.append(summary)
            except Exception as exc:
                self.tree.item(iid, values=(path, "ERROR", "", ""))
                self._log(f"[SANITY][ERROR] {path_obj}: {exc}")
            self.progress["value"] = i
            self.update_idletasks()

        if not summaries:
            self._log("[SANITY] No valid WCS entries to analyze.")
            return

        # Compute simple cluster statistics
        ra_vals = [s["ra_deg"] for s in summaries if isinstance(s.get("ra_deg"), (int, float))]
        dec_vals = [s["dec_deg"] for s in summaries if isinstance(s.get("dec_deg"), (int, float))]
        scale_vals = [
            s["scale_arcsec"]
            for s in summaries
            if isinstance(s.get("scale_arcsec"), (int, float)) and s["scale_arcsec"] > 0
        ]

        ra_ref = sum(ra_vals) / len(ra_vals) if ra_vals else None
        dec_ref = sum(dec_vals) / len(dec_vals) if dec_vals else None

        scale_median = None
        if scale_vals:
            sorted_scales = sorted(scale_vals)
            mid = len(sorted_scales) // 2
            if len(sorted_scales) % 2:
                scale_median = sorted_scales[mid]
            else:
                scale_median = 0.5 * (sorted_scales[mid - 1] + sorted_scales[mid])

        max_sep_allowed = 5.0      # degrees between centers
        max_scale_rel_diff = 0.10  # 10% difference in pixel scale

        outliers = 0
        for s in summaries:
            reasons = []
            ra = s.get("ra_deg")
            dec = s.get("dec_deg")
            sep = None
            if ra_ref is not None and dec_ref is not None and ra is not None and dec is not None:
                sep = _angular_separation_deg(ra_ref, dec_ref, ra, dec)
                if sep is not None and math.isfinite(sep) and sep > max_sep_allowed:
                    reasons.append(f"center_separation={sep:.2f}deg")
            scale = s.get("scale_arcsec")
            if scale_median is not None and isinstance(scale, (int, float)) and scale > 0:
                rel = abs(scale - scale_median) / scale_median
                if rel > max_scale_rel_diff:
                    reasons.append(f"scale_delta={rel*100:.1f}%")

            if reasons:
                status = "WCS: OUTLIER"
                outliers += 1
            else:
                status = "WCS: OK"

            self.tree.item(s["iid"], values=(s["path"], status, "", ""))
            scale_txt = f"{s['scale_arcsec']:.3f}\"/px" if isinstance(s.get("scale_arcsec"), (int, float)) else "n/a"
            if sep is not None and math.isfinite(sep):
                self._log(
                    f"[SANITY] {s['path']}: {status} "
                    f"(RA={s['ra_deg']:.6f}°, DEC={s['dec_deg']:.6f}°, "
                    f"scale={scale_txt}, sep={sep:.3f}°, "
                    f"{', '.join(reasons) if reasons else 'within thresholds'})"
                )
            else:
                self._log(
                    f"[SANITY] {s['path']}: {status} "
                    f"(RA={s['ra_deg']:.6f}°, DEC={s['dec_deg']:.6f}°, "
                    f"scale={scale_txt}, "
                    f"{', '.join(reasons) if reasons else 'within thresholds'})"
                )

        if ra_ref is not None and dec_ref is not None:
            self._log(f"[SANITY] Reference center ≈ RA={ra_ref:.6f}°, DEC={dec_ref:.6f}°.")
        if scale_median is not None:
            self._log(f"[SANITY] Median pixel scale ≈ {scale_median:.3f}\"/px.")
        self._log(f"[SANITY] Done. {outliers} / {len(summaries)} WCS flagged as outliers.")

    def scan_selection(self):
        # Update status with YES/NO for WCS presence
        targets = list(self._iter_selected_rows())
        if not targets:
            messagebox.showinfo("Scan", "Nothing to scan.")
            return
        self.progress.configure(maximum=len(targets), value=0)
        hits = 0
        for i, (iid, path) in enumerate(targets, 1):
            path_obj = Path(path)
            try:
                with fits.open(path_obj, memmap=False) as hdul:
                    targets_hdus = range(len(hdul)) if self.var_all_hdus.get() else [0]
                    has = any(header_has_wcs(hdul[h].header) for h in targets_hdus)
                status = "WCS: YES" if has else "WCS: NO"
                if has: hits += 1
                self.tree.item(iid, values=(path, status, "", ""))
            except Exception as exc:
                self.tree.item(iid, values=(path, "ERROR", "", ""))
                self._log(f"[SCAN][ERROR] {path_obj}: {exc}")
            self.progress['value'] = i
            self.update_idletasks()
        self._log(f"Scan done. {hits} / {len(targets)} with WCS.")

    def clean_selection(self):
        targets = list(self._iter_selected_rows())
        if not targets:
            messagebox.showinfo("Clean", "Nothing to clean.")
            return

        dry = self.var_dry.get()
        backup = self.var_backup.get()
        only_if_wcs = self.var_only_wcs.get()
        all_hdus = self.var_all_hdus.get()

        if not dry:
            if not messagebox.askyesno("Confirm", "This will modify files. Proceed?"):
                return

        self.progress.configure(maximum=len(targets), value=0)
        total_deleted = 0
        edited_files = 0

        for i, (iid, path) in enumerate(targets, 1):
            path_obj = Path(path)
            try:
                deleted, edited_hdus = process_fits(
                    path_obj,
                    dry_run=dry,
                    backup=backup,
                    only_if_wcs=only_if_wcs,
                    all_hdus=all_hdus,
                )
                total_deleted += deleted
                if edited_hdus > 0:
                    edited_files += 1
                    status = "CLEANED" if not dry else "DRY"
                else:
                    status = "NO-OP"
                self.tree.item(iid, values=(path, status, str(deleted), str(edited_hdus)))
            except Exception as exc:
                self.tree.item(iid, values=(path, "ERROR", "", ""))
                self._log(f"[CLEAN][ERROR] {path_obj}: {exc}")
                traceback.print_exc()
            self.progress['value'] = i
            self.update_idletasks()

        mode = "Dry-run" if dry else "Applied"
        self._log(f"{mode}: removed {total_deleted} card(s) across {edited_files} file(s).")

if __name__ == "__main__":
    App().mainloop()
