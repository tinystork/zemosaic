# Create a standalone Tkinter GUI script that segments Seestar S50 CSV into "full-overlap" lots.
# It lets the user choose input/output files and tweak parameters.
# The script will be saved to /mnt/data/full_overlap_lots_gui.py

from pathlib import Path

script_path = Path("/mnt/data/full_overlap_lots_gui.py")

code = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-overlap Lots GUI for Seestar S50 CSV (headers_cache.csv-like)
- Choose input CSV and output CSV/PNG.
- Parameters: margin, cluster threshold, default FOV (deg).
- Produces: CSV with 'lot_id' and a RA/DEC plot colored by lot.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------- Core logic (adapted from the notebook prototype) --------

def angular_sep_deg(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.deg2rad, (ra1, dec1, ra2, dec2))
    d_ra = ra2 - ra1
    d_dec = dec2 - dec1
    a = np.sin(d_dec/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin(d_ra/2)**2
    c = 2*np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return np.rad2deg(c)

def coerce_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def parse_time_series(df):
    for col in ["DATE-OBS","DATE_AVG","DATE","TIME-OBS","timestamp","Timestamp"]:
        if col in df.columns:
            try:
                t = pd.to_datetime(df[col], errors="coerce", utc=True)
                if t.notna().any():
                    return t.view("int64")/1e9
            except Exception:
                pass
    return pd.Series(np.arange(len(df)), index=df.index, dtype=float)

def detect_ra_dec_columns(df):
    possible_ra = [c for c in df.columns if c.strip().lower() in ("ra","crval1","objctra","telra")]
    possible_dec = [c for c in df.columns if c.strip().lower() in ("dec","crval2","objctdec","teldec")]
    if not possible_ra or not possible_dec:
        raise RuntimeError("RA/DEC columns not found in CSV (tried RA/CRVAL1/OBJCTRA/TELRA and DEC/CRVAL2/OBJCTDEC/TELDEC).")
    return possible_ra[0], possible_dec[0]

def run_segmentation(csv_in: Path, csv_out: Path, plot_out: Path,
                     fov_w_deg: float, fov_h_deg: float,
                     margin: float, cluster_thresh_deg: float):
    df = pd.read_csv(csv_in)

    ra_col, dec_col = detect_ra_dec_columns(df)
    df["_RA_"]  = df[ra_col].apply(coerce_float)
    df["_DEC_"] = df[dec_col].apply(coerce_float)
    df = df.dropna(subset=["_RA_","_DEC_"]).reset_index(drop=True)

    # Use CSV-provided FOV hints if present
    FOV_W, FOV_H = fov_w_deg, fov_h_deg
    for cand_w in ["FOV_W_DEG","FOVW","FOV_WIDTH_DEG"]:
        if cand_w in df.columns:
            v = pd.to_numeric(df[cand_w], errors="coerce").dropna()
            if len(v): FOV_W = float(v.median()); break
    for cand_h in ["FOV_H_DEG","FOVH","FOV_HEIGHT_DEG"]:
        if cand_h in df.columns:
            v = pd.to_numeric(df[cand_h], errors="coerce").dropna()
            if len(v): FOV_H = float(v.median()); break

    # Params
    margin = float(margin)
    cluster_thresh_deg = float(cluster_thresh_deg)
    radius_deg = min(FOV_W, FOV_H) * 0.5 * (1.0 - margin)

    # Pre-cluster by proximity (time-ordered)
    order_time = parse_time_series(df)
    df = df.assign(_t_=order_time).sort_values("_t_").reset_index(drop=True)

    clusters = []
    for idx, row in df.iterrows():
        ra, dec = row["_RA_"], row["_DEC_"]
        placed = False
        for cl in clusters:
            cra = np.mean([r for r,_ in cl["centers"]])
            cdec = np.mean([d for _,d in cl["centers"]])
            if angular_sep_deg(ra, dec, cra, cdec) <= cluster_thresh_deg:
                cl["indices"].append(idx)
                cl["centers"].append((ra,dec))
                placed = True
                break
        if not placed:
            clusters.append({"indices":[idx], "centers":[(ra,dec)]})

    # Within each cluster, split into "full-overlap" lots
    lot_ids = np.full(len(df), -1, dtype=int)
    next_lot = 0
    for cl in clusters:
        indices = cl["indices"]
        sub = df.loc[indices].sort_values("_t_")
        current_lot = []
        lot_ra = []
        lot_dec = []
        for j, row in sub.iterrows():
            ra, dec = row["_RA_"], row["_DEC_"]
            if not current_lot:
                current_lot = [row.name]
                lot_ra = [ra]
                lot_dec = [dec]
                continue
            pivot_ra = float(np.median(lot_ra))
            pivot_dec = float(np.median(lot_dec))
            d = angular_sep_deg(ra, dec, pivot_ra, pivot_dec)
            if d <= radius_deg:
                current_lot.append(row.name)
                lot_ra.append(ra)
                lot_dec.append(dec)
            else:
                for rid in current_lot:
                    lot_ids[rid] = next_lot
                next_lot += 1
                current_lot = [row.name]
                lot_ra = [ra]
                lot_dec = [dec]
        if current_lot:
            for rid in current_lot:
                lot_ids[rid] = next_lot
            next_lot += 1

    df["lot_id"] = lot_ids

    # Save CSV (original cols + lot_id at end)
    out_cols = list(df.columns)
    out_cols = [c for c in out_cols if c != "lot_id"] + ["lot_id"]
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(csv_out, index=False)

    # Plot
    plt.figure(figsize=(9,7))
    for lot in sorted(df["lot_id"].unique()):
        sub = df[df["lot_id"] == lot]
        # DO NOT set explicit colors or styles (keep defaults per best practices)
        plt.plot(sub["_RA_"], sub["_DEC_"], 'o-', markersize=3, linewidth=0.8, alpha=0.85, label=f"lot {lot}")
    plt.gca().invert_xaxis()
    plt.xlabel("Right Ascension (deg)")
    plt.ylabel("Declination (deg)")
    plt.title("Full-overlap lots – automatic segmentation")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(ncol=2, fontsize=8, frameon=False)
    plt.tight_layout()
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_out, dpi=150)
    plt.close()

    return int(df["lot_id"].nunique())

# -------- GUI --------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Seestar S50 – Full-overlap Lots")
        try:
            self.geometry("620x380")
        except tk.TclError:
            pass

        # Variables
        self.in_csv  = tk.StringVar()
        self.out_csv = tk.StringVar()
        self.out_png = tk.StringVar()

        self.fov_w = tk.DoubleVar(value=1.30)
        self.fov_h = tk.DoubleVar(value=0.73)
        self.margin = tk.DoubleVar(value=0.12)
        self.cluster_thresh = tk.DoubleVar(value=0.20)

        # Layout
        pad = {'padx':8, 'pady':6}

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, **pad)

        # Input
        ttk.Label(frm, text="Input CSV:").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.in_csv, width=48).grid(row=0, column=1, sticky="we")
        ttk.Button(frm, text="Browse…", command=self.browse_in).grid(row=0, column=2, sticky="w")

        # Output CSV
        ttk.Label(frm, text="Output CSV:").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.out_csv, width=48).grid(row=1, column=1, sticky="we")
        ttk.Button(frm, text="Browse…", command=self.browse_out_csv).grid(row=1, column=2, sticky="w")

        # Output PNG
        ttk.Label(frm, text="Output Plot PNG:").grid(row=2, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.out_png, width=48).grid(row=2, column=1, sticky="we")
        ttk.Button(frm, text="Browse…", command=self.browse_out_png).grid(row=2, column=2, sticky="w")

        # Parameters frame
        pfrm = ttk.LabelFrame(frm, text="Parameters")
        pfrm.grid(row=3, column=0, columnspan=3, sticky="we", **pad)

        ttk.Label(pfrm, text="Default FOV width (deg):").grid(row=0, column=0, sticky="e")
        ttk.Entry(pfrm, textvariable=self.fov_w, width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(pfrm, text="Default FOV height (deg):").grid(row=1, column=0, sticky="e")
        ttk.Entry(pfrm, textvariable=self.fov_h, width=10).grid(row=1, column=1, sticky="w")

        ttk.Label(pfrm, text="Margin (0..0.5):").grid(row=2, column=0, sticky="e")
        ttk.Entry(pfrm, textvariable=self.margin, width=10).grid(row=2, column=1, sticky="w")
        ttk.Label(pfrm, text="(higher margin => stricter overlap)").grid(row=2, column=2, sticky="w")

        ttk.Label(pfrm, text="Pre-cluster threshold (deg):").grid(row=3, column=0, sticky="e")
        ttk.Entry(pfrm, textvariable=self.cluster_thresh, width=10).grid(row=3, column=1, sticky="w")
        ttk.Label(pfrm, text="(avoid mixing distant fields)").grid(row=3, column=2, sticky="w")

        # Run button + status
        run_row = 4
        ttk.Button(frm, text="Run segmentation", command=self.run).grid(row=run_row, column=0, sticky="w", **pad)
        self.status = ttk.Label(frm, text="Ready.", foreground="gray")
        self.status.grid(row=run_row, column=1, columnspan=2, sticky="w")

        # Grid weights
        for c in (1,):
            frm.columnconfigure(c, weight=1)

    def browse_in(self):
        fp = filedialog.askopenfilename(title="Select input CSV", filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if fp:
            self.in_csv.set(fp)
            # Suggest outputs
            base = Path(fp).with_suffix("")
            self.out_csv.set(str(base) + "_with_lot.csv")
            self.out_png.set(str(base) + "_lots.png")

    def browse_out_csv(self):
        fp = filedialog.asksaveasfilename(title="Select output CSV", defaultextension=".csv",
                                          filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if fp:
            self.out_csv.set(fp)

    def browse_out_png(self):
        fp = filedialog.asksaveasfilename(title="Select output PNG", defaultextension=".png",
                                          filetypes=[("PNG image","*.png"),("All files","*.*")])
        if fp:
            self.out_png.set(fp)

    def run(self):
        try:
            in_csv = Path(self.in_csv.get())
            out_csv = Path(self.out_csv.get())
            out_png = Path(self.out_png.get())
            if not in_csv.is_file():
                messagebox.showerror("Error", "Input CSV not found.")
                return

            lots = run_segmentation(
                csv_in=in_csv,
                csv_out=out_csv,
                plot_out=out_png,
                fov_w_deg=float(self.fov_w.get()),
                fov_h_deg=float(self.fov_h.get()),
                margin=float(self.margin.get()),
                cluster_thresh_deg=float(self.cluster_thresh.get()),
            )
            self.status.config(text=f"Done. Lots created: {lots}.")
            messagebox.showinfo("Completed", f"Segmentation done.\nLots: {lots}\nCSV: {out_csv}\nPNG: {out_png}")
        except Exception as exc:
            self.status.config(text=f"Error: {exc}")
            messagebox.showerror("Error", str(exc))

if __name__ == "__main__":
    App().mainloop()
'''
script_path.write_text(code, encoding="utf-8")

script_path.as_posix()
