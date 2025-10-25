"""
Optional GUI filter for ZeMosaic Phase 1 results.

This module exposes a single function:

    launch_filter_interface(
        raw_files_with_wcs: list[dict],
        initial_overrides: dict | None = None,
    ) -> tuple[list[dict], bool, dict | None]

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
from dataclasses import asdict
import os
import sys
import shutil
import datetime
import importlib
import time


def launch_filter_interface(
    raw_files_with_wcs: List[Dict[str, Any]],
    initial_overrides: Optional[Dict[str, Any]] = None,
    solver_settings_dict: Optional[Dict[str, Any]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
):
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

    solver_settings_dict : dict, optional
        Dictionary of solver configuration selected in the main GUI. When
        provided, values such as ASTAP paths and search radius override the
        defaults loaded from configuration files.
    config_overrides : dict, optional
        Additional configuration values collected from the main GUI (for
        example the ASTAP data directory or clustering caps). These override
        the defaults detected by this module.

    Returns
    -------
    tuple[list[dict], bool, dict | None]
        (filtered_list, accepted, overrides) where accepted is True only when
        Validate is clicked; False when Cancel is clicked or the window is closed.
        overrides can contain optional metadata collected in the filter GUI, such
        as pre-computed master tile groupings or counters for resolved WCS files:
          {
             "preplan_master_groups": list[list[dict]],
             "autosplit_cap": int,
             "filter_excluded_indices": list[int],
             "resolved_wcs_count": int,
          }
        Keys are included only when relevant actions were triggered.
    """
    # Early validation and fail-safe behavior
    if not isinstance(raw_files_with_wcs, list) or not raw_files_with_wcs:
        return raw_files_with_wcs, False, None

    try:
        # --- Optional localization support (autonomous fallback) ---
        # If running inside the ZeMosaic project folder structure, try to use
        # the existing localization system and the language set in main GUI.
        localizer = None
        cfg_defaults: Dict[str, Any] = {}
        cfg: Dict[str, Any] | None = None
        solver_settings_payload: Dict[str, Any] = {}
        try:
            # Ensure project directory is on sys.path to import project modules
            base_dir = os.path.dirname(os.path.abspath(__file__))
            if base_dir not in sys.path:
                sys.path.insert(0, base_dir)

            from zemosaic_localization import ZeMosaicLocalization
            import zemosaic_config
            cfg = zemosaic_config.load_config()
            lang_code = cfg.get("language", "en")

            from solver_settings import SolverSettings

            if isinstance(solver_settings_dict, dict):
                solver_settings_payload.update(solver_settings_dict)
            else:
                try:
                    solver_settings = SolverSettings.load_default()
                except Exception:
                    solver_settings = SolverSettings()
                solver_settings_payload.update(asdict(solver_settings))

            cfg_defaults = {
                "astap_executable_path": cfg.get("astap_executable_path", ""),
                "astap_data_directory_path": cfg.get("astap_data_directory_path", ""),
                "astap_default_search_radius": cfg.get("astap_default_search_radius", 0.0),
                "astap_default_downsample": cfg.get("astap_default_downsample", 0),
                "astap_default_sensitivity": cfg.get("astap_default_sensitivity", 100),
                "auto_limit_frames_per_master_tile": cfg.get("auto_limit_frames_per_master_tile", True),
                "max_raw_per_master_tile": cfg.get("max_raw_per_master_tile", 0),
                "apply_master_tile_crop": cfg.get("apply_master_tile_crop", False),
                "master_tile_crop_percent": cfg.get("master_tile_crop_percent", 0.0),
            }

            if solver_settings_payload:
                exe_path = solver_settings_payload.get("astap_executable_path")
                data_path = solver_settings_payload.get("astap_data_directory_path")
                search_radius = solver_settings_payload.get("astap_search_radius_deg")
                downsample = solver_settings_payload.get("astap_downsample")
                sensitivity = solver_settings_payload.get("astap_sensitivity")

                if isinstance(exe_path, str) and exe_path:
                    cfg_defaults["astap_executable_path"] = exe_path
                if isinstance(data_path, str) and data_path:
                    cfg_defaults["astap_data_directory_path"] = data_path
                if search_radius is not None:
                    cfg_defaults["astap_default_search_radius"] = search_radius
                if downsample is not None:
                    cfg_defaults["astap_default_downsample"] = downsample
                if sensitivity is not None:
                    cfg_defaults["astap_default_sensitivity"] = sensitivity

            localizer = ZeMosaicLocalization(language_code=lang_code)
        except Exception as e:
            print(f"WARNING (Filter GUI): failed to init localization/config: {e}")
            cfg_defaults = {}
            solver_settings_payload = {}
            localizer = None

        if isinstance(config_overrides, dict):
            try:
                cfg_defaults.update(config_overrides)
            except Exception:
                pass

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
        from tkinter import ttk, messagebox, scrolledtext

        from core.tk_safe import patch_tk_variables

        patch_tk_variables()
        import numpy as np
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from matplotlib.figure import Figure
        from matplotlib.patches import Polygon
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        astrometry_mod = None
        worker_mod = None
        def _import_module_with_fallback(mod_name: str):
            """Attempt to import a module regardless of package layout."""

            try:
                return importlib.import_module(mod_name)
            except Exception:
                pass

            # When ZeMosaic is installed as a package (``zemosaic.*``) the
            # absolute module name may include the package prefix.  Use the
            # current module package when available so the imports also work
            # in that layout.
            pkg = globals().get("__package__") or ""
            if pkg:
                try:
                    return importlib.import_module(f"{pkg}.{mod_name}")
                except Exception:
                    pass

            # Finally try importing from the top-level ``zemosaic`` package.
            if not mod_name.startswith("zemosaic"):
                try:
                    return importlib.import_module(f"zemosaic.{mod_name}")
                except Exception:
                    pass
            return None

        astrometry_mod = _import_module_with_fallback('zemosaic_astrometry')
        worker_mod = _import_module_with_fallback('zemosaic_worker')

        solve_with_astap = getattr(astrometry_mod, 'solve_with_astap', None) if astrometry_mod else None
        extract_center_from_header_fn = getattr(astrometry_mod, 'extract_center_from_header', None) if astrometry_mod else None
        astap_fits_module = getattr(astrometry_mod, 'fits', None) if astrometry_mod else None
        astap_astropy_available = bool(getattr(astrometry_mod, 'ASTROPY_AVAILABLE_ASTROMETRY', False)) if astrometry_mod else False
        cluster_func = getattr(worker_mod, 'cluster_seestar_stacks_connected', None) if worker_mod else None
        autosplit_func = getattr(worker_mod, '_auto_split_groups', None) if worker_mod else None
        compute_dispersion_func = getattr(worker_mod, '_compute_max_angular_separation_deg', None) if worker_mod else None

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
                self.shape: Optional[tuple[int, int]] = None
                self.center: Optional[SkyCoord] = None
                self.phase0_center: Optional[SkyCoord] = None
                self.footprint: Optional[np.ndarray] = None  # shape (N, 2) in deg
                self.refresh_geometry()

            def _coerce_skycoord(self, value: Any) -> Optional[SkyCoord]:
                if value is None:
                    return None
                if isinstance(value, SkyCoord):
                    return value
                try:
                    if isinstance(value, (list, tuple)) and len(value) >= 2:
                        ra_deg = float(value[0])
                        dec_deg = float(value[1])
                        return SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
                    if isinstance(value, dict):
                        ra_val = value.get("ra") or value.get("RA")
                        dec_val = value.get("dec") or value.get("DEC")
                        if ra_val is not None and dec_val is not None:
                            return SkyCoord(ra=float(ra_val) * u.deg, dec=float(dec_val) * u.deg, frame="icrs")
                except Exception:
                    return None
                return None

            def _infer_shape(self) -> Optional[tuple[int, int]]:
                shp = self.src.get("shape")
                if isinstance(shp, (list, tuple)) and len(shp) >= 2:
                    try:
                        h = int(shp[0])
                        w = int(shp[1])
                        if h > 0 and w > 0:
                            return (h, w)
                    except Exception:
                        pass
                header_obj = self.header
                if header_obj is not None:
                    try:
                        getter = header_obj.get if hasattr(header_obj, "get") else header_obj.__getitem__
                        naxis1 = int(getter("NAXIS1"))
                        naxis2 = int(getter("NAXIS2"))
                        if naxis1 > 0 and naxis2 > 0:
                            return (naxis2, naxis1)
                    except Exception:
                        pass
                if getattr(self.wcs, "pixel_shape", None):
                    try:
                        nx, ny = self.wcs.pixel_shape  # type: ignore[attr-defined]
                        h = int(ny)
                        w = int(nx)
                        if h > 0 and w > 0:
                            return (h, w)
                    except Exception:
                        pass
                if getattr(self.wcs, "array_shape", None):
                    try:
                        ny, nx = self.wcs.array_shape  # type: ignore[attr-defined]
                        h = int(ny)
                        w = int(nx)
                        if h > 0 and w > 0:
                            return (h, w)
                    except Exception:
                        pass
                return None

            def _center_from_wcs(self, shape_hw: Optional[tuple[int, int]]) -> Optional[SkyCoord]:
                if self.wcs is None:
                    return None
                try:
                    if shape_hw is not None:
                        h, w = shape_hw
                        xc = (w - 1) / 2.0
                        yc = (h - 1) / 2.0
                    else:
                        crpix = getattr(self.wcs.wcs, "crpix", None)
                        if crpix is not None and len(crpix) >= 2:
                            xc, yc = float(crpix[0]), float(crpix[1])
                        else:
                            xc, yc = 1023.5, 1023.5
                    sky = self.wcs.pixel_to_world(xc, yc)
                    ra = float(sky.ra.to(u.deg).value)
                    dec = float(sky.dec.to(u.deg).value)
                    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
                except Exception:
                    return None

            def _center_from_header(self) -> Optional[SkyCoord]:
                if extract_center_from_header_fn and self.header is not None:
                    try:
                        return extract_center_from_header_fn(self.header)
                    except Exception:
                        return None
                return None

            def _build_footprint(self, shape_hw: Optional[tuple[int, int]]) -> Optional[np.ndarray]:
                if self.wcs is None or shape_hw is None:
                    return None
                try:
                    h, w = shape_hw
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
                    return np.column_stack([np.array(ras), np.array(decs)])
                except Exception:
                    return None

            def refresh_geometry(self) -> None:
                self.shape = self._infer_shape()
                if self.shape and self.wcs is not None and getattr(self.wcs, "pixel_shape", None) is None:
                    try:
                        self.wcs.pixel_shape = (self.shape[1], self.shape[0])
                    except Exception:
                        pass
                self.phase0_center = self._coerce_skycoord(self.src.get("phase0_center"))
                direct_center = self._coerce_skycoord(self.src.get("center"))
                center = direct_center or self.phase0_center or self._center_from_wcs(self.shape) or self._center_from_header()
                self.center = center
                if self.center is not None:
                    self.src["center"] = self.center
                if self.shape is not None:
                    self.src["shape"] = self.shape
                self.footprint = self._build_footprint(self.shape)

        # Build items list
        items: list[Item] = [Item(d, i) for i, d in enumerate(raw_files_with_wcs)]
        overrides_state: Dict[str, Any] = {}
        if isinstance(initial_overrides, dict):
            try:
                overrides_state.update(initial_overrides)
            except Exception:
                overrides_state = {}

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

        # Prepare Tkinter window (use Toplevel if a main Tk exists)
        parent_root = getattr(tk, "_default_root", None)
        root_is_toplevel = False
        if parent_root is not None:
            try:
                root = tk.Toplevel(parent_root)
                root_is_toplevel = True
                try:
                    root.transient(parent_root)
                except Exception:
                    pass
                try:
                    root.grab_set()  # modal
                except Exception:
                    pass
            except Exception:
                root = tk.Tk()
        else:
            root = tk.Tk()

        root.title(_tr(
            "filter_window_title",
            "ZeMosaic - Filtrer les images WCS (optionnel)" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "ZeMosaic - Filter WCS images (optional)"
        ))
        # S'assurer que la fenêtre apparaît au premier plan et prend le focus
        try:
            root.lift()
            root.attributes("-topmost", True)
            root.after(200, lambda: root.attributes("-topmost", False))
            root.focus_force()
        except Exception:
            pass
        # Top-level layout: left plot, right checkboxes/actions
        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        # Matplotlib figure
        # Use constrained_layout to reduce internal padding and let the
        # figure adapt to the available space.
        fig = Figure(figsize=(7.0, 5.0), dpi=100, constrained_layout=True)
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

        # Make the Matplotlib figure follow the widget size to avoid
        # large empty borders around the plot.
        def _apply_resize():
            try:
                w = max(50, int(canvas_widget.winfo_width()))
                h = max(50, int(canvas_widget.winfo_height()))
                # Resize figure in pixels -> inches
                fig.set_size_inches(w / fig.dpi, h / fig.dpi, forward=True)
                # Maximize axes area inside the figure
                try:
                    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.08, top=0.98)
                except Exception:
                    pass
                canvas.draw_idle()
            except Exception:
                pass

        _resize_job = {"id": None}

        def _on_canvas_configure(_event=None):
            # Throttle frequent resizes during interactive dragging
            try:
                if _resize_job["id"] is not None:
                    canvas_widget.after_cancel(_resize_job["id"])  # type: ignore[arg-type]
                _resize_job["id"] = canvas_widget.after(60, lambda: (_apply_resize(), _resize_job.update({"id": None})))
            except Exception:
                pass

        # Bind and trigger once to sync initial size
        try:
            canvas_widget.bind("<Configure>", _on_canvas_configure)
        except Exception:
            pass
        try:
            root.update_idletasks()
            _apply_resize()
        except Exception:
            pass

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
        # The scrollable list lives at row=2 (row=1 reserved for clustering params)
        try:
            right.rowconfigure(2, weight=1)
        except Exception:
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
        thresh_var = tk.StringVar(master=root, value="5.0")
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

        # Logging pane
        log_frame = ttk.LabelFrame(
            right,
            text=_tr(
                "filter_log_panel_title",
                "Activity log" if 'fr' not in str(locals().get('lang_code', 'en')).lower() else "Journal d'activité",
            ),
        )
        log_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        log_widget = scrolledtext.ScrolledText(log_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        log_widget.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        def _log_message(message: str, level: str = "INFO") -> None:
            try:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                log_widget.configure(state=tk.NORMAL)
                log_widget.insert(tk.END, f"[{timestamp}] [{level.upper()}] {message}\n")
                log_widget.configure(state=tk.DISABLED)
                log_widget.see(tk.END)
            except Exception:
                pass

        def _progress_callback(msg: Any, _progress: Any = None, lvl: str | None = None) -> None:
            if msg is None:
                return
            level = (lvl or "INFO")
            _log_message(str(msg), level=level)
            try:
                root.update_idletasks()
            except Exception:
                pass

        def _sanitize_path(value: Any) -> str:
            """Normalize user-provided filesystem paths.

            Configuration values may include surrounding quotes, unresolved
            environment variables (e.g. ``%PROGRAMFILES%`` on Windows) or
            user-home shortcuts.  Normalising here avoids false negatives
            when checking for ASTAP availability.
            """

            try:
                if value is None:
                    return ""
                path_str = str(value).strip()
            except Exception:
                return ""

            # Drop wrapping quotes that Windows file dialogs can preserve
            if path_str.startswith(('"', "'")) and path_str.endswith(('"', "'")) and len(path_str) >= 2:
                path_str = path_str[1:-1]

            expanded = os.path.expanduser(os.path.expandvars(path_str))
            return expanded

        astap_exe_path_raw = cfg_defaults.get('astap_executable_path', '')
        astap_data_dir_raw = cfg_defaults.get('astap_data_directory_path', '')
        astap_exe_path = _sanitize_path(astap_exe_path_raw)
        astap_data_dir = _sanitize_path(astap_data_dir_raw)

        # Keep the sanitized values in the defaults so downstream callers (log
        # messages, resolver invocations) see consistent paths.
        if astap_exe_path:
            cfg_defaults['astap_executable_path'] = astap_exe_path
        if astap_data_dir:
            cfg_defaults['astap_data_directory_path'] = astap_data_dir
        search_radius_default = cfg_defaults.get('astap_default_search_radius', 0.0)
        downsample_default = cfg_defaults.get('astap_default_downsample', 0)
        autosplit_cap_cfg = cfg_defaults.get('max_raw_per_master_tile', 0)
        try:
            autosplit_cap_cfg_int = int(autosplit_cap_cfg)
        except Exception:
            autosplit_cap_cfg_int = 0
        autosplit_cap = autosplit_cap_cfg_int if autosplit_cap_cfg_int > 0 else 50
        autosplit_cap = max(1, min(50, autosplit_cap))
        autosplit_min_cap = min(8, autosplit_cap)

        def _astap_path_available(path: str) -> bool:
            """Return True when the configured ASTAP location looks valid."""

            if not path:
                return False

            # Direct file check
            if os.path.isfile(path):
                return True

            # macOS packages are directories ending with ``.app``
            if sys.platform == "darwin" and path.lower().endswith(".app") and os.path.isdir(path):
                return True

            # Accept directories that contain the ASTAP binary
            if os.path.isdir(path):
                exe_name = "astap.exe" if os.name == "nt" else "astap"
                candidate = os.path.join(path, exe_name)
                if os.path.isfile(candidate):
                    return True

            # As a generic fallback try resolving via PATH
            resolved = shutil.which(path) or shutil.which(os.path.basename(path))
            if resolved:
                return True

            # As a generic fallback accept any existing executable entry.
            try:
                return os.path.exists(path) and os.access(path, os.X_OK)
            except Exception:
                return False

        astap_available = bool(
            solve_with_astap is not None
            and astap_astropy_available
            and _astap_path_available(astap_exe_path)
        )

        # Scrollable checkboxes list
        list_frame = ttk.LabelFrame(
            right,
            text=_tr(
                "filter_images_check_to_keep",
                "Images (cocher pour garder)" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Images (check to keep)",
            ),
        )
        list_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
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

        summary_var = tk.StringVar(master=root, value="")
        resolved_counter = {"count": int(overrides_state.get("resolved_wcs_count", 0) or 0)}

        operations = ttk.Frame(right)
        operations.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        operations.columnconfigure(2, weight=1)

        resolve_btn = ttk.Button(
            operations,
            text=_tr("filter_btn_resolve_wcs", "Resolve missing WCS"),
        )
        resolve_btn.grid(row=0, column=0, padx=4, pady=2, sticky="w")

        auto_btn = ttk.Button(
            operations,
            text=_tr("filter_btn_auto_group", "Auto-organize Master Tiles"),
        )
        auto_btn.grid(row=0, column=1, padx=4, pady=2, sticky="w")

        ttk.Label(
            operations,
            textvariable=summary_var,
            anchor="w",
            justify="left",
            wraplength=260,
        ).grid(row=0, column=2, padx=4, pady=2, sticky="w")

        if not astap_available:
            resolve_btn.state(["disabled"])
            _log_message(
                _tr("filter_warn_astap_missing", "ASTAP executable not configured; skipping resolution."),
                level="WARN",
            )

        if not (cluster_func and autosplit_func):
            auto_btn.state(["disabled"])

        def _resolve_missing_wcs_inplace() -> None:
            if not astap_available or solve_with_astap is None:
                _log_message(
                    _tr("filter_warn_astap_missing", "ASTAP executable not configured; skipping resolution."),
                    level="WARN",
                )
                return
            resolve_btn.state(["disabled"])
            try:
                try:
                    srch_radius = float(search_radius_default)
                    if srch_radius <= 0:
                        srch_radius = None
                except Exception:
                    srch_radius = None
                try:
                    downsample_val = int(downsample_default)
                    if downsample_val < 0:
                        downsample_val = None
                except Exception:
                    downsample_val = None

                resolved_now = 0
                for idx, item in enumerate(items):
                    if item.wcs is not None:
                        continue
                    path = item.path
                    if not (isinstance(path, str) and os.path.isfile(path)):
                        continue
                    header_obj = item.header
                    if header_obj is None and astap_fits_module is not None and astap_astropy_available:
                        try:
                            with astap_fits_module.open(path) as hdul_hdr:
                                header_obj = hdul_hdr[0].header
                                item.header = header_obj
                                item.src["header"] = header_obj
                        except Exception:
                            header_obj = item.header
                    try:
                        wcs_obj = solve_with_astap(
                            path,
                            header_obj,
                            astap_exe_path,
                            astap_data_dir,
                            search_radius_deg=srch_radius,
                            downsample_factor=downsample_val,
                            sensitivity=None,
                            timeout_sec=60,
                            update_original_header_in_place=False,
                            progress_callback=_progress_callback,
                        )
                    except Exception as exc:
                        _log_message(f"ASTAP resolve exception: {exc}", level="ERROR")
                        wcs_obj = None
                    if wcs_obj and getattr(wcs_obj, "is_celestial", False):
                        item.src["wcs"] = wcs_obj
                        item.wcs = wcs_obj
                        item.refresh_geometry()
                        _refresh_item_visual(idx)
                        resolved_now += 1
                        try:
                            root.update_idletasks()
                        except Exception:
                            pass
                if resolved_now >= 0:
                    resolved_counter["count"] += resolved_now
                    if resolved_now > 0:
                        overrides_state["resolved_wcs_count"] = resolved_counter["count"]
                    summary_msg = _tr(
                        "filter_log_resolved_n",
                        "Resolved WCS for {n} files.",
                        n=resolved_now,
                    )
                    _log_message(summary_msg, level="INFO")
                    if resolved_now > 0:
                        _recompute_axes_limits()
            finally:
                if astap_available:
                    resolve_btn.state(["!disabled"])

        def _auto_organize_master_tiles() -> None:
            if not (cluster_func and autosplit_func):
                return
            auto_btn.state(["disabled"])
            try:
                selected_indices = [i for i, v in enumerate(check_vars) if v.get()]
                if not selected_indices:
                    summary_text = _tr(
                        "filter_log_groups_summary",
                        "Prepared {g} group(s), sizes: {sizes}.",
                        g=0,
                        sizes="[]",
                    )
                    summary_var.set(summary_text)
                    _log_message(summary_text, level="INFO")
                    return

                class _FallbackWCS:
                    is_celestial = True

                    def __init__(self, center_coord: SkyCoord):
                        self._center = center_coord
                        self.pixel_shape = (1, 1)
                        self.array_shape = (1, 1)

                        class _Inner:
                            def __init__(self, center: SkyCoord):
                                self.crval = (
                                    float(center.ra.to(u.deg).value),
                                    float(center.dec.to(u.deg).value),
                                )
                                self.crpix = (0.5, 0.5)

                        self.wcs = _Inner(center_coord)

                    def pixel_to_world(self, _x: float, _y: float):
                        return self._center

                candidate_infos: list[dict] = []
                coord_samples: list[tuple[float, float]] = []
                for idx in selected_indices:
                    item = items[idx]
                    entry = dict(item.src)
                    if "path" not in entry:
                        entry["path"] = item.path
                    if "path_raw" not in entry and item.src.get("path_raw"):
                        entry["path_raw"] = item.src.get("path_raw")
                    if "header" not in entry:
                        entry["header"] = item.header
                    if item.shape and "shape" not in entry:
                        entry["shape"] = item.shape
                    center_obj = item.center or item.phase0_center
                    if center_obj is None and extract_center_from_header_fn and item.header is not None:
                        try:
                            center_obj = extract_center_from_header_fn(item.header)
                        except Exception:
                            center_obj = None
                    if item.wcs is None and center_obj is not None:
                        entry["wcs"] = _FallbackWCS(center_obj)
                        entry["_fallback_wcs_used"] = True
                        entry.setdefault("phase0_center", center_obj)
                        entry.setdefault("center", center_obj)
                    elif item.wcs is not None:
                        entry["wcs"] = item.wcs
                        if center_obj is not None:
                            entry.setdefault("center", center_obj)
                    if center_obj is not None:
                        coord_samples.append(
                            (
                                float(center_obj.ra.to(u.deg).value),
                                float(center_obj.dec.to(u.deg).value),
                            )
                        )
                    candidate_infos.append(entry)

                if not candidate_infos:
                    summary_text = _tr(
                        "filter_log_groups_summary",
                        "Prepared {g} group(s), sizes: {sizes}.",
                        g=0,
                        sizes="[]",
                    )
                    summary_var.set(summary_text)
                    _log_message(summary_text, level="WARN")
                    return

                if coord_samples:
                    if compute_dispersion_func:
                        try:
                            dispersion_deg = float(compute_dispersion_func(coord_samples))
                        except Exception:
                            dispersion_deg = 0.0
                    else:
                        dispersion_deg = 0.0
                else:
                    dispersion_deg = 0.0

                if dispersion_deg <= 0.12:
                    threshold_deg = 0.10
                elif dispersion_deg <= 0.30:
                    threshold_deg = 0.15
                else:
                    threshold_deg = 0.18 if dispersion_deg <= 0.60 else 0.20
                threshold_deg = min(0.20, max(0.08, threshold_deg))

                groups = cluster_func(
                    candidate_infos,
                    float(threshold_deg),
                    _progress_callback,
                    orientation_split_threshold_deg=0.0,
                )
                if not groups:
                    summary_text = _tr(
                        "filter_log_groups_summary",
                        "Prepared {g} group(s), sizes: {sizes}.",
                        g=0,
                        sizes="[]",
                    )
                    summary_var.set(summary_text)
                    _log_message(summary_text, level="WARN")
                    return

                final_groups = autosplit_func(
                    groups,
                    cap=int(autosplit_cap),
                    min_cap=int(autosplit_min_cap),
                    progress_callback=_progress_callback,
                )
                for grp in final_groups:
                    for info in grp:
                        if info.pop("_fallback_wcs_used", False):
                            info.pop("wcs", None)
                overrides_state["preplan_master_groups"] = final_groups
                overrides_state["autosplit_cap"] = int(autosplit_cap)
                sizes = [len(gr) for gr in final_groups]
                sizes_str = ", ".join(str(s) for s in sizes) if sizes else "[]"
                summary_text = _tr(
                    "filter_log_groups_summary",
                    "Prepared {g} group(s), sizes: {sizes}.",
                    g=len(final_groups),
                    sizes=sizes_str,
                )
                summary_var.set(summary_text)
                _log_message(summary_text, level="INFO")
            finally:
                auto_btn.state(["!disabled"])

        resolve_btn.configure(command=_resolve_missing_wcs_inplace)
        auto_btn.configure(command=_auto_organize_master_tiles)

        # Selection helpers
        actions = ttk.Frame(right)
        actions.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
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
        bottom.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        result: dict[str, Any] = {"accepted": None, "selected_indices": None, "overrides": None}
        def on_validate():
            sel = [i for i, v in enumerate(check_vars) if v.get()]
            result["accepted"] = True
            result["selected_indices"] = sel
            result["overrides"] = overrides_state if overrides_state else None
            try:
                root.quit()
            except Exception:
                pass
            root.destroy()
        def on_cancel():
            result["accepted"] = False
            result["selected_indices"] = None
            try:
                root.quit()
            except Exception:
                pass
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
            var = tk.BooleanVar(master=root, value=True)
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

        def _recompute_axes_limits() -> None:
            ra_vals: list[float] = []
            dec_vals: list[float] = []
            for it in items:
                if it.footprint is not None:
                    for ra in it.footprint[:, 0].tolist():
                        ra_vals.append(wrap_ra_deg(float(ra), ref_ra))
                    dec_vals.extend(it.footprint[:, 1].tolist())
                elif it.center is not None:
                    ra_vals.append(wrap_ra_deg(float(it.center.ra.to(u.deg).value), ref_ra))
                    dec_vals.append(float(it.center.dec.to(u.deg).value))
            if ra_vals and dec_vals:
                ra_min, ra_max = min(ra_vals), max(ra_vals)
                dec_min, dec_max = min(dec_vals), max(dec_vals)
                ra_pad = max(1e-3, (ra_max - ra_min) * 0.05 + 0.2)
                dec_pad = max(1e-3, (dec_max - dec_min) * 0.05 + 0.2)
                ax.set_xlim(ra_max + ra_pad, ra_min - ra_pad)
                ax.set_ylim(dec_min - dec_pad, dec_max + dec_pad)
                canvas.draw_idle()

        def _refresh_item_visual(idx: int) -> None:
            if idx < 0 or idx >= len(items):
                return
            item = items[idx]
            prev_patch = patches[idx] if idx < len(patches) else None
            prev_point = center_pts[idx] if idx < len(center_pts) else None
            if prev_patch is not None:
                try:
                    prev_patch.remove()
                except Exception:
                    pass
                artist_to_index.pop(prev_patch, None)
                patches[idx] = None
            if prev_point is not None:
                try:
                    prev_point.remove()
                except Exception:
                    pass
                artist_to_index.pop(prev_point, None)
                center_pts[idx] = None
            selected = check_vars[idx].get()
            color_sel = "tab:blue" if selected else "0.7"
            alpha_val = 0.9 if selected else 0.3
            new_patch = None
            new_point = None
            if item.footprint is not None:
                try:
                    ra_wrapped = [wrap_ra_deg(float(ra), ref_ra) for ra in item.footprint[:, 0].tolist()]
                    decs = item.footprint[:, 1].tolist()
                    new_patch = Polygon(list(zip(ra_wrapped, decs)), closed=True, fill=False, edgecolor=color_sel, linewidth=1.0, alpha=alpha_val)
                    new_patch.set_picker(True)
                    ax.add_patch(new_patch)
                    artist_to_index[new_patch] = idx
                    patches[idx] = new_patch
                except Exception:
                    new_patch = None
            elif item.center is not None:
                try:
                    ra_c = wrap_ra_deg(float(item.center.ra.to(u.deg).value), ref_ra)
                    dec_c = float(item.center.dec.to(u.deg).value)
                    new_point, = ax.plot([ra_c], [dec_c], marker="o", markersize=3, color=color_sel, alpha=alpha_val, picker=8)
                    artist_to_index[new_point] = idx
                    center_pts[idx] = new_point
                except Exception:
                    new_point = None
            update_visuals(idx)
            _recompute_axes_limits()

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
            if result.get("accepted") is None:
                result["accepted"] = False
            result["selected_indices"] = None
            root.destroy()
        root.protocol("WM_DELETE_WINDOW", on_close)

        # Start modal loop (use wait_window for Toplevel)
        try:
            if root_is_toplevel:
                parent = parent_root if parent_root is not None else root
                try:
                    parent.wait_window(root)
                except Exception:
                    # Fallback to simple update loop
                    try:
                        root.update()
                        root.update_idletasks()
                    except Exception:
                        pass
            else:
                root.mainloop()
        except KeyboardInterrupt:
            # If interrupted, keep default behavior (keep all)
            pass
        except Exception:
            # Some environments (notably on Windows when running from a
            # multiprocessing daemon) occasionally fail to enter Tk's
            # mainloop, leaving the window unresponsive.  Fall back to a
            # manual event pump so the user can still interact with the UI.
            try:
                deadline = time.monotonic() + 3600.0  # 1 hour safety cap
                while True:
                    try:
                        root.update_idletasks()
                        root.update()
                    except Exception:
                        break
                    # Exit when the window is destroyed or a decision was made
                    if not root.winfo_exists() or result.get("accepted") is not None:
                        break
                    if time.monotonic() > deadline:
                        break
                    time.sleep(0.05)
                try:
                    if root.winfo_exists():
                        root.destroy()
                except Exception:
                    pass
            except Exception:
                pass

        # Return selection (and move unselected files into 'filtered_by_user')
        accepted_flag = result.get("accepted")
        if accepted_flag is None:
            accepted_flag = False
        if accepted_flag and isinstance(result.get("selected_indices"), list):
            sel = result["selected_indices"]  # type: ignore[assignment]

            # Compute unselected indices
            total_n = len(raw_files_with_wcs)
            unselected_indices = [i for i in range(total_n) if i not in sel]
            if unselected_indices:
                overrides_state["filter_excluded_indices"] = unselected_indices

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

            return [raw_files_with_wcs[i] for i in sel], True, (overrides_state if overrides_state else None)
        else:
            # Keep all if canceled or closed
            return raw_files_with_wcs, False, None

    except ImportError:
        # Any optional dependency missing — silently keep all
        return raw_files_with_wcs, False, None
    except Exception:
        # Any unexpected error — fail safe and keep all
        return raw_files_with_wcs, False, None


__all__ = ["launch_filter_interface"]
