"""
Optional GUI filter for ZeMosaic Phase 1 results.

This module exposes a single function:

    launch_filter_interface(raw_files_with_wcs: list[dict]) -> tuple[list[dict], bool]

It opens a small Tkinter window with a schematic sky map (RA/Dec, in degrees)
showing the footprint of each WCS-resolved image as a polygon and a checkbox
list to include/exclude individual images. A helper can also exclude images
farther than X degrees from the global center.

Safety requirements:
- If this module is missing or raises, the caller should continue unchanged.
- If the window is closed or any error occurs, the original input list is
  returned unchanged.
  In that case this function returns (original_list, False) so the caller can
  decide to abort processing.

Dependencies limited to tkinter, astropy, matplotlib, numpy; all optional.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import os
import sys
import shutil
import datetime
import importlib


def launch_filter_interface(raw_files_with_wcs: List[Dict[str, Any]]):
    """
    Display an optional Tkinter GUI to filter WCS-resolved images.

    Parameters
    ----------
    raw_files_with_wcs : list[dict]
        Each dict should ideally include:
          - 'path' (or 'path_raw') : absolute path to the FITS file
          - 'wcs' : astropy.wcs.WCS object
          - 'shape' (optional) : (H, W)
          - 'index' (optional) : processing index
          - 'center' (optional) : astropy.coordinates.SkyCoord
          - 'header' (optional) : astropy.io.fits.Header (used to infer shape)

    Returns
    -------
    tuple[list[dict], bool]
        (filtered_list, accepted) where accepted is True only when Validate is
        clicked; False when Cancel is clicked or the window is closed.
    """
    # Early validation and fail-safe behavior
    if not isinstance(raw_files_with_wcs, list) or not raw_files_with_wcs:
        return raw_files_with_wcs, False

    try:
        # --- Optional localization support (autonomous fallback) ---
        # If running inside the ZeMosaic project folder structure, try to use
        # the existing localization system and the language set in main GUI.
        localizer = None
        try:
            # Ensure project directory is on sys.path to import locales/* and config
            base_dir = os.path.dirname(os.path.abspath(__file__))
            if base_dir not in sys.path:
                sys.path.insert(0, base_dir)

            # Try import of localization util
            locales_mod = importlib.import_module('locales.zemosaic_localization')
            ZeMosaicLocalization = getattr(locales_mod, 'ZeMosaicLocalization', None)

            # Try import of config to get language preference
            lang_code = 'en'
            try:
                zcfg = importlib.import_module('zemosaic_config')
                cfg = zcfg.load_config()
                lang_code = cfg.get('language', 'en')
            except Exception:
                lang_code = 'en'

            if ZeMosaicLocalization is not None:
                localizer = ZeMosaicLocalization(language_code=lang_code)
        except Exception:
            localizer = None

        def _tr(key: str, default_text: Optional[str] = None, **kwargs) -> str:
            if localizer is not None:
                try:
                    # Pass default_text so JSON keys are optional
                    return localizer.get(key, default_text, **kwargs)
                except Exception:
                    pass
            # Fallback: return default_text or key placeholder
            return default_text if default_text is not None else f"_{key}_"

        # Imports kept inside to avoid import-time errors affecting pipeline
        import tkinter as tk
        from tkinter import ttk, messagebox
        import numpy as np
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from matplotlib.figure import Figure
        from matplotlib.patches import Polygon
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # Attempt to read astropy WCS only when needed
        # (objects are provided by caller; we don't import WCS explicitly here)

        # Normalize entries to a consistent structure the GUI can use
        class Item:
            def __init__(self, src: Dict[str, Any], idx: int):
                self.src = src
                self.index: int = int(src.get("index", idx))
                # Prefer 'path', then 'path_raw', then fallback to cached path
                self.path: str = (
                    src.get("path")
                    or src.get("path_raw")
                    or src.get("path_preprocessed_cache")
                    or f"<unknown_{idx}>"
                )
                self.wcs = src.get("wcs")
                self.header = src.get("header")
                # Determine image shape (H, W)
                shp = src.get("shape")
                if not shp and self.header is not None:
                    try:
                        naxis1 = int(self.header.get("NAXIS1"))
                        naxis2 = int(self.header.get("NAXIS2"))
                        shp = (naxis2, naxis1)
                    except Exception:
                        shp = None
                if not shp and getattr(self.wcs, "pixel_shape", None):
                    try:
                        # astropy WCS pixel_shape is (nx, ny)
                        nx, ny = self.wcs.pixel_shape  # type: ignore[attr-defined]
                        shp = (int(ny), int(nx))
                    except Exception:
                        shp = None
                if not shp and getattr(self.wcs, "array_shape", None):
                    try:
                        # array_shape tends to be (ny, nx)
                        ny, nx = self.wcs.array_shape  # type: ignore[attr-defined]
                        shp = (int(ny), int(nx))
                    except Exception:
                        shp = None
                self.shape: Optional[tuple[int, int]] = shp if isinstance(shp, tuple) and len(shp) == 2 else None

                # Center SkyCoord if provided; else compute from WCS/shape
                c = src.get("center")
                self.center: Optional[SkyCoord] = None
                if c is not None:
                    try:
                        # Trust provided SkyCoord
                        self.center = c
                    except Exception:
                        self.center = None
                if self.center is None and self.wcs is not None:
                    try:
                        if self.shape is not None:
                            h, w = self.shape
                            xc = (w - 1) / 2.0
                            yc = (h - 1) / 2.0
                        else:
                            # Fallback to CRPIX if shape is unknown
                            crpix = getattr(self.wcs.wcs, "crpix", None)
                            if crpix is not None and len(crpix) >= 2:
                                xc, yc = float(crpix[0]), float(crpix[1])
                            else:
                                # Last resort: assume 2048x2048 and center
                                xc, yc = 1023.5, 1023.5
                        sky = self.wcs.pixel_to_world(xc, yc)
                        # Ensure RA/Dec in degrees
                        ra = float(sky.ra.to(u.deg).value)
                        dec = float(sky.dec.to(u.deg).value)
                        self.center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
                    except Exception:
                        self.center = None

                # Compute footprint corners as SkyCoord list if possible
                self.footprint: Optional[np.ndarray] = None  # shape (N, 2) in deg as [[ra, dec], ...]
                if self.wcs is not None and self.shape is not None:
                    try:
                        h, w = self.shape
                        # Corners in pixel coords (x=col, y=row)
                        corners = [
                            (0.0, 0.0),
                            (w - 1.0, 0.0),
                            (w - 1.0, h - 1.0),
                            (0.0, h - 1.0),
                        ]
                        ras = []
                        decs = []
                        for (x, y) in corners:
                            sc = self.wcs.pixel_to_world(x, y)
                            ras.append(float(sc.ra.to(u.deg).value))
                            decs.append(float(sc.dec.to(u.deg).value))
                        self.footprint = np.column_stack([np.array(ras), np.array(decs)])
                    except Exception:
                        self.footprint = None

        # Build items list
        items: list[Item] = [Item(d, i) for i, d in enumerate(raw_files_with_wcs)]

        # If virtually nothing to display, skip GUI
        if not any((it.center is not None) for it in items):
            return raw_files_with_wcs, False

        # Compute robust global center via unit-vector average
        def average_skycoord(coords: list[SkyCoord]) -> SkyCoord:
            arr = np.array([c.cartesian.xyz.value for c in coords])
            vec = arr.mean(axis=0)
            vec_norm = vec / np.linalg.norm(vec)
            sc = SkyCoord(x=vec_norm[0] * u.one, y=vec_norm[1] * u.one, z=vec_norm[2] * u.one, frame="icrs", representation_type="cartesian").spherical
            return SkyCoord(ra=sc.lon.to(u.deg), dec=sc.lat.to(u.deg), frame="icrs")

        centers_available = [it.center for it in items if it.center is not None]
        global_center: SkyCoord = centers_available[0] if len(centers_available) == 1 else average_skycoord(centers_available)

        # RA wrapping helper around a reference RA
        def wrap_ra_deg(ra_deg: float, ref_deg: float) -> float:
            x = ra_deg
            r = ref_deg
            d = x - r
            # map difference to [-180, 180)
            d = (d + 180.0) % 360.0 - 180.0
            return r + d

        # Prepare Tkinter window
        root = tk.Tk()
        root.title(_tr(
            "filter_window_title",
            "ZeMosaic - Filtrer les images WCS (optionnel)" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "ZeMosaic - Filter WCS images (optional)"
        ))

        # Top-level layout: left plot, right checkboxes/actions
        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        # Matplotlib figure
        fig = Figure(figsize=(7.0, 5.0), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlabel(_tr(
            "filter_axis_ra_deg",
            "AD [deg]" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "RA [deg]"
        ))
        ax.set_ylabel(_tr(
            "filter_axis_dec_deg",
            "Dec [deg]"
        ))
        ax.set_aspect("equal", adjustable="datalim")
        # For sky-like view, invert RA axis (optional)
        ax.invert_xaxis()

        canvas = FigureCanvasTkAgg(fig, master=main)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Enable mouse-wheel zoom on the plot for easier selection
        def _setup_wheel_zoom(ax):
            base_scale = 1.2  # zoom factor per wheel notch

            def _orient_limits(lims):
                a, b = lims
                inv = a > b
                mn, mx = (b, a) if inv else (a, b)
                return mn, mx, inv

            def _apply_limits(ax, xmin, xmax, xinv, ymin, ymax, yinv):
                if xinv:
                    ax.set_xlim(xmax, xmin)
                else:
                    ax.set_xlim(xmin, xmax)
                if yinv:
                    ax.set_ylim(ymax, ymin)
                else:
                    ax.set_ylim(ymin, ymax)

            def on_scroll(event):
                try:
                    if event is None or event.inaxes is None:
                        return
                    ax_ = event.inaxes
                    xdata = event.xdata if event.xdata is not None else sum(ax_.get_xlim()) / 2.0
                    ydata = event.ydata if event.ydata is not None else sum(ax_.get_ylim()) / 2.0

                    xmin0, xmax0, xinv = _orient_limits(ax_.get_xlim())
                    ymin0, ymax0, yinv = _orient_limits(ax_.get_ylim())
                    width = max(1e-9, (xmax0 - xmin0))
                    height = max(1e-9, (ymax0 - ymin0))

                    # Choose scale direction
                    if getattr(event, 'button', 'up') in ('up', 4):
                        # zoom in
                        scale = 1.0 / base_scale
                    else:
                        # zoom out
                        scale = base_scale

                    new_w = width * scale
                    new_h = height * scale

                    # Compute relative position of mouse within current view
                    # using oriented (min->max) extents
                    relx = (xdata - xmin0) / max(1e-12, width)
                    rely = (ydata - ymin0) / max(1e-12, height)
                    relx = min(max(relx, 0.0), 1.0)
                    rely = min(max(rely, 0.0), 1.0)

                    xmin = xdata - relx * new_w
                    xmax = xdata + (1.0 - relx) * new_w
                    ymin = ydata - rely * new_h
                    ymax = ydata + (1.0 - rely) * new_h

                    # Avoid zero-span
                    if (xmax - xmin) < 1e-9:
                        pad = 5e-10
                        xmin -= pad; xmax += pad
                    if (ymax - ymin) < 1e-9:
                        pad = 5e-10
                        ymin -= pad; ymax += pad

                    _apply_limits(ax_, xmin, xmax, xinv, ymin, ymax, yinv)
                    canvas.draw_idle()
                except Exception:
                    pass

            try:
                canvas.mpl_connect('scroll_event', on_scroll)
            except Exception:
                pass

        _setup_wheel_zoom(ax)

        # Right panel with controls
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # Threshold controls
        thresh_frame = ttk.LabelFrame(
            right,
            text=_tr(
                "filter_exclude_by_distance_title",
                "Exclure par distance au centre" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Exclude by distance to center",
            ),
        )
        thresh_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ttk.Label(
            thresh_frame,
            text=_tr(
                "filter_distance_label",
                "Distance (deg) :" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Distance (deg):",
            ),
        ).grid(row=0, column=0, padx=4, pady=4)
        thresh_var = tk.StringVar(value="5.0")
        thresh_entry = ttk.Entry(thresh_frame, textvariable=thresh_var, width=8)
        thresh_entry.grid(row=0, column=1, padx=4, pady=4)
        def apply_threshold():
            try:
                thr = float(thresh_var.get())
            except Exception:
                messagebox.showwarning(
                    _tr(
                        "filter_invalid_value_title",
                        "Valeur invalide" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Invalid value",
                    ),
                    _tr(
                        "filter_invalid_value_message",
                        "Veuillez entrer une distance en degrés (nombre)." if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Please enter a distance in degrees (number).",
                    ),
                )
                return
            for it, var in zip(items, check_vars):
                if it.center is None:
                    continue
                sep = it.center.separation(global_center).to(u.deg).value
                if sep > thr:
                    var.set(False)
            update_visuals()
        ttk.Button(
            thresh_frame,
            text=_tr(
                "filter_apply_threshold_button",
                "Exclure > X°" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Exclude > X°",
            ),
            command=apply_threshold,
        ).grid(row=0, column=2, padx=4, pady=4)

        # Scrollable checkboxes list
        list_frame = ttk.LabelFrame(
            right,
            text=_tr(
                "filter_images_check_to_keep",
                "Images (cocher pour garder)" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Images (check to keep)",
            ),
        )
        list_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        canvas_list = tk.Canvas(list_frame, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=canvas_list.yview)
        inner = ttk.Frame(canvas_list)
        inner.bind("<Configure>", lambda e: canvas_list.configure(scrollregion=canvas_list.bbox("all")))
        canvas_list.create_window((0, 0), window=inner, anchor="nw")
        canvas_list.configure(yscrollcommand=vsb.set)
        canvas_list.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # Enable mouse-wheel scrolling over the right list (Windows/Linux/macOS)
        def _on_list_mousewheel(event):
            try:
                if getattr(event, 'num', None) == 4:  # Linux scroll up
                    canvas_list.yview_scroll(-1, "units")
                elif getattr(event, 'num', None) == 5:  # Linux scroll down
                    canvas_list.yview_scroll(1, "units")
                else:  # Windows / macOS
                    delta = int(-1 * (event.delta / 120)) if getattr(event, 'delta', 0) else 0
                    if delta != 0:
                        canvas_list.yview_scroll(delta, "units")
            except Exception:
                pass

        def _bind_list_mousewheel(_):
            try:
                canvas_list.bind_all("<MouseWheel>", _on_list_mousewheel)
                canvas_list.bind_all("<Button-4>", _on_list_mousewheel)
                canvas_list.bind_all("<Button-5>", _on_list_mousewheel)
            except Exception:
                pass

        def _unbind_list_mousewheel(_):
            try:
                canvas_list.unbind_all("<MouseWheel>")
                canvas_list.unbind_all("<Button-4>")
                canvas_list.unbind_all("<Button-5>")
            except Exception:
                pass

        # Bind/unbind on enter/leave so the wheel affects only this list
        inner.bind("<Enter>", _bind_list_mousewheel)
        inner.bind("<Leave>", _unbind_list_mousewheel)
        canvas_list.bind("<Enter>", _bind_list_mousewheel)
        canvas_list.bind("<Leave>", _unbind_list_mousewheel)

        # Selection helpers
        actions = ttk.Frame(right)
        actions.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        def select_all():
            for v in check_vars: v.set(True)
            update_visuals()
        def select_none():
            for v in check_vars: v.set(False)
            update_visuals()
        ttk.Button(
            actions,
            text=_tr(
                "filter_select_all",
                "Tout sélectionner" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Select all",
            ),
            command=select_all,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            actions,
            text=_tr(
                "filter_select_none",
                "Tout désélectionner" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Deselect all",
            ),
            command=select_none,
        ).pack(side=tk.LEFT, padx=4)

        # Confirm/cancel buttons
        bottom = ttk.Frame(right)
        bottom.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        result: dict[str, Any] = {"accepted": False, "selected_indices": None}
        def on_validate():
            sel = [i for i, v in enumerate(check_vars) if v.get()]
            result["accepted"] = True
            result["selected_indices"] = sel
            root.destroy()
        def on_cancel():
            result["accepted"] = False
            result["selected_indices"] = None
            root.destroy()
        ttk.Button(
            bottom,
            text=_tr(
                "filter_validate",
                "Valider" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Validate",
            ),
            command=on_validate,
        ).pack(side=tk.RIGHT, padx=4)
        ttk.Button(
            bottom,
            text=_tr(
                "filter_cancel",
                "Annuler" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Cancel",
            ),
            command=on_cancel,
        ).pack(side=tk.RIGHT, padx=4)

        # Populate checkboxes and draw footprints
        check_vars: list[tk.BooleanVar] = []
        patches: list[Polygon] = []
        center_pts: list[Any] = []  # matplotlib line2D handles
        # Map matplotlib artists back to item indices for click-to-select
        artist_to_index: dict[Any, int] = {}

        ref_ra = float(global_center.ra.to(u.deg).value)
        ref_dec = float(global_center.dec.to(u.deg).value)

        # Build visuals first to compute bounds
        all_ra_vals: list[float] = []
        all_dec_vals: list[float] = []

        for idx, it in enumerate(items):
            var = tk.BooleanVar(value=True)
            check_vars.append(var)

            # Label includes basename and optional separation
            import os
            base = os.path.basename(it.path)
            sep_txt = ""
            if it.center is not None:
                sep_deg = it.center.separation(global_center).to(u.deg).value
                sep_txt = f"  ({sep_deg:.2f}°)"
            cb = ttk.Checkbutton(inner, text=base + sep_txt, variable=var, command=lambda i=idx: update_visuals(i))
            cb.grid(row=idx, column=0, sticky="w")

            # Plot polygon footprint if available; else plot center point
            color_sel = "tab:blue"
            color_unsel = "0.7"

            if it.footprint is not None:
                ra_wrapped = [wrap_ra_deg(float(ra), ref_ra) for ra in it.footprint[:, 0].tolist()]
                decs = it.footprint[:, 1].tolist()
                poly = Polygon(list(zip(ra_wrapped, decs)), closed=True, fill=False, edgecolor=color_sel, linewidth=1.0, alpha=0.9)
                # Allow selection by clicking inside the footprint on the plot
                try:
                    poly.set_picker(True)
                except Exception:
                    pass
                patches.append(poly)
                ax.add_patch(poly)
                artist_to_index[poly] = idx
                all_ra_vals.extend(ra_wrapped)
                all_dec_vals.extend(decs)
            elif it.center is not None:
                ra_c = wrap_ra_deg(float(it.center.ra.to(u.deg).value), ref_ra)
                dec_c = float(it.center.dec.to(u.deg).value)
                ln, = ax.plot([ra_c], [dec_c], marker="o", markersize=3, color=color_sel, alpha=0.9, picker=8)
                center_pts.append(ln)
                artist_to_index[ln] = idx
                all_ra_vals.append(ra_c)
                all_dec_vals.append(dec_c)
            else:
                # No WCS displayable — skip drawing, but keep placeholders for indexing
                patches.append(None)  # type: ignore
                center_pts.append(None)  # type: ignore

        # Set view limits with small margins
        if all_ra_vals and all_dec_vals:
            ra_min, ra_max = min(all_ra_vals), max(all_ra_vals)
            dec_min, dec_max = min(all_dec_vals), max(all_dec_vals)
            ra_pad = max(1e-3, (ra_max - ra_min) * 0.05 + 0.2)
            dec_pad = max(1e-3, (dec_max - dec_min) * 0.05 + 0.2)
            ax.set_xlim(ra_max + ra_pad, ra_min - ra_pad)  # inverted x-axis
            ax.set_ylim(dec_min - dec_pad, dec_max + dec_pad)
        ax.grid(True, which="both", linestyle=":", linewidth=0.6)

        def update_visuals(changed_index: Optional[int] = None) -> None:
            # Update colors/alpha for polygons and points according to selection state
            for i, it in enumerate(items):
                selected = check_vars[i].get()
                col = "tab:blue" if selected else "0.7"
                alp = 0.9 if selected else 0.3
                if i < len(patches) and patches[i] is not None:
                    patches[i].set_edgecolor(col)
                    patches[i].set_alpha(alp)
                if i < len(center_pts) and center_pts[i] is not None:
                    center_pts[i].set_color(col)
                    center_pts[i].set_alpha(alp)
            canvas.draw_idle()

        update_visuals()

        # Click-to-select/deselect via matplotlib pick events
        def _on_pick(event):
            try:
                artist = getattr(event, 'artist', None)
                if artist is None:
                    return
                i = artist_to_index.get(artist)
                if i is None:
                    return
                # Toggle associated checkbox and refresh colors
                curr = check_vars[i].get()
                check_vars[i].set(not curr)
                update_visuals(i)
            except Exception:
                pass

        try:
            canvas.mpl_connect('pick_event', _on_pick)
        except Exception:
            pass

        # On window close: treat as cancel (keep all)
        def on_close():
            result["accepted"] = False
            result["selected_indices"] = None
            root.destroy()
        root.protocol("WM_DELETE_WINDOW", on_close)

        # Start modal loop
        try:
            root.mainloop()
        except KeyboardInterrupt:
            # If interrupted, keep default behavior (keep all)
            pass

        # Return selection (and move unselected files into 'filtered_by_user')
        if result.get("accepted") and isinstance(result.get("selected_indices"), list):
            sel = result["selected_indices"]  # type: ignore[assignment]

            # Compute unselected indices
            total_n = len(raw_files_with_wcs)
            unselected_indices = [i for i in range(total_n) if i not in sel]

            # Prepare destination folder under the common input directory
            def _preferred_src_path(entry: Dict[str, Any]) -> Optional[str]:
                # Prefer raw/original paths if available
                p = entry.get("path_raw") or entry.get("path")
                if isinstance(p, str):
                    return p
                return None

            excluded_paths: list[str] = []
            all_src_dirs: list[str] = []
            for i in unselected_indices:
                p = _preferred_src_path(raw_files_with_wcs[i])
                if p and os.path.isfile(p):
                    excluded_paths.append(p)
                    all_src_dirs.append(os.path.dirname(p))

            dest_base: Optional[str] = None
            if all_src_dirs:
                try:
                    dest_base = os.path.commonpath(all_src_dirs)
                except Exception:
                    dest_base = all_src_dirs[0]

            # If we have a base, move excluded files to '<base>/filtered_by_user'
            if dest_base is not None and excluded_paths:
                dest_dir = os.path.join(dest_base, "filtered_by_user")
                try:
                    os.makedirs(dest_dir, exist_ok=True)
                except Exception:
                    dest_dir = None  # will fallback to per-file folder

                def _unique_dest(path_dir: str, filename: str) -> str:
                    base, ext = os.path.splitext(filename)
                    candidate = os.path.join(path_dir, filename)
                    if not os.path.exists(candidate):
                        return candidate
                    # Append timestamp, then counter if still colliding
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    candidate = os.path.join(path_dir, f"{base}_{ts}{ext}")
                    if not os.path.exists(candidate):
                        return candidate
                    k = 1
                    while True:
                        candidate = os.path.join(path_dir, f"{base}_{ts}_{k}{ext}")
                        if not os.path.exists(candidate):
                            return candidate
                        k += 1

                # Perform moves
                for src_path in excluded_paths:
                    try:
                        target_dir = dest_dir
                        if target_dir is None:
                            # As fallback, move next to its source directory under a local 'filtered_by_user'
                            local_dir = os.path.join(os.path.dirname(src_path), "filtered_by_user")
                            os.makedirs(local_dir, exist_ok=True)
                            target_dir = local_dir
                        dest_path = _unique_dest(target_dir, os.path.basename(src_path))
                        shutil.move(src_path, dest_path)
                    except Exception as e:
                        # Non-fatal: keep going
                        print(f"WARN filter_gui: Failed to move '{src_path}' -> filtered_by_user: {e}")

            return [raw_files_with_wcs[i] for i in sel], True
        else:
            # Keep all if canceled or closed
            return raw_files_with_wcs, False

    except ImportError:
        # Any optional dependency missing — silently keep all
        return raw_files_with_wcs, False
    except Exception:
        # Any unexpected error — fail safe and keep all
        return raw_files_with_wcs, False


__all__ = ["launch_filter_interface"]
