AGENT MISSION — ZeMosaic GUI + Core Stability

You are an autonomous coding agent working inside the ZeMosaic project.

The repository contains (non-exhaustive):
zemosaic_gui.py (Tk), zemosaic_filter_gui.py (Tk),
zemosaic_gui_qt.py (Qt), zemosaic_filter_gui_qt.py (Qt),
zemosaic_worker.py, zemosaic_utils.py,
lecropper.py, zewcscleaner.py,
zemosaic_astrometry.py, zemosaic_localization.py,
solver_settings.*, zemosaic_config.*,
locales/*.json, icons, and helper modules.

Your responsibilities:

1 — GLOBAL RULES

You must NEVER break:

Existing Tkinter GUI behavior.

Existing Seestar mosaic and stacking logic.

The worker pipeline, Phase 4.5 super-tiles, masks, alpha propagation.

FITS headers, metadata propagation, and WCS.

Batch size = 0 or >1 behavior (strictly preserve).

Compatibility Windows / macOS / Linux.

Localization system (locales/en.json, locales/fr.json, future locales).

Icons loading across OSes.

No new dependencies unless explicitly allowed.

2 — QT GUI MISSION

You must progressively build a complete PySide6 GUI, parallel to Tkinter:

Reproduce all features of Tk GUIs (main GUI + filter GUI).

Respect layout specs: scroll areas, QSplitter when requested, 2-column grids, tab structure.

Keep clear separation:

GUI-only code in *_qt.py.

Business logic untouched.

Strict consistency with translations (_tr()), config fields, and event handlers.

3 — FILTER GUI MISSION

Qt filter GUI must reproduce exactly the logic of zemosaic_filter_gui.py.

Features to mirror:

Auto-organize master tiles

WCS grouping + preview

Coverage checks

Prepared groups

Preplan logging output

Sky preview placement & behavior

Tk version must remain unchanged.

4 — ASTROMETRY / MASKS MISSION

The codebase uses:

ASTAP CLI

Proprietary alt-az cleanup

Coverage maps

Alpha masks (NaN/transparent zones)

Lecropper pipeline

You must:

Apply masks properly to FITS 16/32-bit and PNG outputs.

Preserve alpha while downsampling.

Remove black arcs in mosaics.

Avoid GUI freezes during ASTAP crashes.

5 — SUPER-TILES & MOSAIC LOGIC

Ensure Phase 4.5 super-tiles are normalized with each other and master tiles.

Keep progression logs intact.

Ensure WCS propagation to all intermediate files.

6 — NEW MODES

You may be asked to implement special modes such as:

Mosaic-first strategy

ZeSupaDupStack

Blind solve preparations
These must plug into the existing pipeline without breaking defaults.

7 — OUTPUT REQUIREMENTS

Code must be:

Deterministic

Patch-friendly

Minimal changes (only requested scope)

Commented only if necessary

Free of accidental refactors

8 — WHEN IN DOUBT

Always:

Preserve existing behavior

Follow instructions in followup.md