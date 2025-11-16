FOLLOW-UP TASKS — ZeMosaic Qt GUI Sprint

This file lists the active tasks for Codex.
Codex must process tasks in order, checking each [x] as tasks are completed.

✅ CURRENT SPRINT : Qt GUI REFINEMENT (Nov 2025)
🟦 1 — Refactor Main tab layout (PySide6)

Adapt _populate_main_tab inside zemosaic_gui_qt.py:

 Replace vertical stacking by a 2-column “brick” grid

 Keep same groupboxes (_create_*) unchanged

 Use QGridLayout inside the "Main" tab only

 Ensure scroll + logging block placement remain unchanged

 No edits to other tabs

 No edits to the worker logic

🟦 2 — Sky Preview parity in Qt filter GUI

In zemosaic_filter_gui_qt.py:

 Add missing red dotted boxes like Tk version

 Place sky preview to the left like Tk

 Restore WCS infos (Prepared group, group boundaries)

 Fix Auto-organize master tiles

 Ensure clicking “Auto organizer” logs steps exactly like Tk

 Run tests with real Seestar batches

🟦 3 — ASTAP crash handler (no GUI freeze)

In zemosaic_astrometry.py:

 Ensure ASTAP watcher never freezes Qt

 Confirm background thread shuts down after each run

 Improve robustness with multiple simultaneous calls

 Keep dialogs auto-dismissed unless KEEP_DIALOGS=1

 Zero impact on Tk

🟦 4 — Alpha mask propagation (Phase 6)

In zemosaic_worker.py Phase 6 and PNG preview:

 FITS final mosaic must contain ALPHA ext with 0–255

 PNG must actually apply alpha (RGBA)

 Downscaling must preserve alpha (nearest)

 NaN areas must become transparent

 No slicing errors

🟦 5 — Lecropper autonomous upgrade

In lecropper.py only:

 Integrate coverage, min_coverage_abs/frac

 Add morphological cleanup

 Add feather mask

 Write ALPHA channel

 The script must remain fully standalone

🟦 6 — Super-tiles normalization

In Phase 4.5:

 Ensure super-tiles are photometrically normalized

 Normalize also against master tiles

 Reduce visible seams

 Preserve WCS and metadata

🟦 7 — Mode ZeSupaDupStack

In Qt Filter GUI:

 Add toggle (checkbox)

 If enabled → mosaic-first strategy overrides default

 All quality filters & LeCropper pass remain functional

 Zero regression when disabled

📁 DONE / ARCHIVED TASKS

(vider quand sprint suivant commence)

 (vide pour l’instant)