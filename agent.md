Oui, ce que tu proposes est complètement cohérent 👍

* Le freeze dans `zemosaic_filter_gui_qt.py` vient très clairement de la boucle qui va chercher *tous* les footprints WCS et les dessine (polygones complets) dans `_update_preview_plot`.
* On a déjà :

  * les groupes logiques (cluster index),
  * des couleurs par groupe,
  * des « outline » de groupes (rectangles) côté Qt.
    Donc afficher **uniquement les outlines de groupes + les centres** suffit pour donner une excellente intuition de la couverture, sans toucher au métier (clustering, overrides, SDS, etc.).
* La persistance de géométrie de fenêtre existe déjà pour le main Qt, et un squelette similaire est en place dans `zemosaic_filter_gui_qt.py` (avec `QT_FILTER_WINDOW_GEOMETRY_KEY` et `_persist_window_geometry`), il faut juste sécuriser / finaliser le câblage.
* Le texte fantôme « Final stacking / output » côté main tab vient très probablement d’un label ou titre dupliqué / mal positionné dans la colonne de droite (groupe « Final assembly & output »). C’est clairement une anomalie de layout Qt, pas du métier.

Du coup, je te propose **deux fichiers prêts à l’emploi** pour Codex : `agent.md` (mission globale) et `followup.md` (check-list détaillée).

---

## `agent.md`

```markdown
# AGENT MISSION — ZEMOSAIC QT POLISH (V4.1)

You are an autonomous coding agent working on the **ZeMosaic** project.

The repository contains (among others):

- `run_zemosaic.py`
- `zemosaic_gui.py` (Tkinter main GUI – legacy)
- `zemosaic_filter_gui.py` (Tkinter filter GUI – legacy)
- `zemosaic_gui_qt.py` (PySide6 main GUI – NEW)
- `zemosaic_filter_gui_qt.py` (PySide6 filter GUI – NEW)
- `zemosaic_worker.py` (business logic / phases)
- `zemosaic_config.py` (config load/save helpers)
- `zemosaic_localization.py`, `locales/*.json` (i18n)
- various helpers (`zemosaic_astrometry.py`, `lecropper.py`, etc.)

Your job in this mission is **not** to redesign the whole app, but to:
1. Make the **Qt Filter GUI** (`zemosaic_filter_gui_qt.py`) lighter and smoother (no freezes when drawing the sky preview).
2. Ensure the **Filter window geometry** is persisted/restored, just like the main Qt window.
3. Fix a **ghost / duplicate text** issue in the Qt main window, right column of the *Main* tab.

The existing Tkinter GUIs and the worker logic are considered **stable** and must keep working unchanged.

---

## GLOBAL PRINCIPLES

- **Do not break**:
  - The Tk GUIs (`zemosaic_gui.py`, `zemosaic_filter_gui.py`).
  - The pipeline logic in `zemosaic_worker.py` (phases, SDS, coverage-first, etc.).
- **Scope strictly to Qt GUI files** unless a small, clearly necessary shared change is required (`zemosaic_config.py`, localization keys).
- Keep everything **cross-platform**: Windows, macOS, Linux (no Win32-only hacks).
- Respect the existing **localization system**:
  - UI text must go through `zemosaic_localization` and `locales/en.json`, `locales/fr.json` (and other locales if needed).
- Keep the **SDS / coverage-first logic** intact; we only change what is *drawn* in the preview, not how groups are computed.
- Prefer **small, focused patches** over large refactors.

---

## CONCRETE GOALS

### 1. Sky Preview: show only group coverage, not every footprint

Problem:  
`zemosaic_filter_gui_qt.py` currently collects and draws **full WCS footprints for many frames** in `_update_preview_plot`. On large Seestar runs (thousands of frames), this causes the Qt Filter window to **freeze** when updating the sky preview.

Desired behavior:

- The **Sky Preview** tab should:
  - Show one point per frame (or per master/group as it currently does).
  - Draw **group-level coverage outlines** (rectangles/polygons per group), just like the existing `_group_outline_bounds` / coverage logic.
  - **NOT** iterate over all frames to build and draw every single footprint polygon for the preview.

Rules:

- Keep the checkbox **“Draw WCS footprints”** in the UI, but you are allowed to:
  - Reinterpret it as “Draw group-level WCS outlines” (no per-frame polygons), **or**
  - Add a hidden / advanced switch so that full per-frame footprints are disabled by default in Qt (for performance).
- No changes to:
  - how clustering works,
  - how groups are saved to overrides,
  - how the coverage map tab works.
- The preview summary / legend (“Group 1… Group N, Unassigned”) must still work.

Success criteria:

- Opening the Qt Filter window on a big dataset (several thousand frames) no longer freezes the GUI when the sky preview updates.
- The user still gets a clear visual idea of **which group covers what region of the sky**, thanks to colored points and group outlines.

---

### 2. Persist window geometry for Qt Filter GUI

Current state:

- The main Qt window (`zemosaic_gui_qt.py`) already has a geometry persistence mechanism (`QT_MAIN_WINDOW_GEOMETRY_KEY`, `_record_window_geometry`, etc.) stored in the config.
- `zemosaic_filter_gui_qt.py` already defines `QT_FILTER_WINDOW_GEOMETRY_KEY` and helper methods:
  - `_load_saved_window_geometry`
  - `_capture_current_window_geometry`
  - `_apply_saved_window_geometry`
  - `_persist_window_geometry`
  - and calls `_apply_saved_window_geometry()` at the end of `__init__`.

Goal:

- Make sure the **Qt Filter dialog**:
  - Restores its last position/size when reopened.
  - Saves its geometry on close.
- Use the same config backend (`_load_gui_config`, `_save_gui_config` from `zemosaic_config.py`).
- Do **not** break any Tk geometry persistence.

Success criteria:

- User resizes/moves the Qt Filter window, closes it, reopens it: the window appears with the same geometry.
- If the saved geometry is invalid, the dialog falls back gracefully to a reasonable default size/position.

---

### 3. Remove ghost / duplicate label in main Qt window (Main tab, right column)

Problem:

- In `zemosaic_gui_qt.py`, on the **Main** tab, right column, some users see a **“ghost” text** with something like “Final stacking/output” overlapping the group header area.
- This is likely due to:
  - a duplicate `QLabel` with the same text as the `QGroupBox` title, or
  - a misconfigured layout that leaves a label floating (e.g., a form row with an empty field but a label still visible).

Goal:

- Clean up the layout for the right column of the Main tab so that:
  - Each section has its group box title *only once*.
  - No stray or overlapping text is visible (even when resizing the window).

Constraints:

- Keep the **structure** of the Main tab (folders group, instrument/Seestar, solver, stacking options, quality, final assembly & output, system resources, etc.).
- Do not remove any real parameter from the GUI; only fix layout / label duplication.

Success criteria:

- The “Final assembly & output” (or localized equivalent) group appears correctly, with no ghost text overlay in any language.
- Resizing the main window does not create overlapping titles or broken layouts.

---

## TESTING & MANUAL CHECKS

You are encouraged to:

1. Run `run_zemosaic.py` with Qt mode enabled and open `zemosaic_gui_qt`:
   - Check the **Main** tab, right column.
   - Confirm the ghost text is gone.
2. From the Qt main window, open the **Filter…** button:
   - On a dataset with hundreds/thousands of frames, let it analyse and draw the **Sky Preview**.
   - Verify:
     - The GUI remains responsive (no long freeze).
     - Group points + outlines look correct.
   - Resize/move the Filter window, close, reopen via the main window:
     - Geometry should be restored.

Document any non-obvious design decisions in code comments, especially around the preview drawing logic and geometry persistence.

Keep changes minimal, focused, and well-structured so that the project maintainers (Tinystork & J.A.R.V.I.S.) can review them easily.
```

---
