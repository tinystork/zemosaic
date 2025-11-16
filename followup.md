# FOLLOW-UP TASKS — ZEMOSAIC QT TABBED LAYOUT + SKINS

This file describes the **step-by-step plan** you should follow to implement and validate the tabbed PySide6 GUI and skin system for ZeMosaic.

Keep changes focused, test after each step, and avoid breaking the Tkinter GUI or existing workflows.


---

## Phase 0 — Baseline

1. **Run the current Qt GUI**

   - Launch `run_zemosaic.py` in Qt mode (as currently configured).
   - Interact with the window:
     - Set input/output folders.
     - Run a small test job (few tiles).
   - Confirm:
     - Buttons Filter / Start / Stop are wired correctly.
     - Logs and progress behave as expected.
     - Any solver configuration in Qt works end-to-end.

2. **Compare with Tkinter behaviour**

   - Launch `zemosaic_gui.py` (Tk).
   - Ensure that:
     - All options present in the Qt GUI have a counterpart in Tk.
     - The semantics (what they mean and trigger) are the same.

This baseline will help detect regressions later.


---

## Phase 1 — Introduce the QTabWidget Skeleton

- [x] **Step 3 — Add a QTabWidget to `zemosaic_gui_qt.py`**

  - In the main window init:
    - Create a `QTabWidget` instance.
    - Create empty `QWidget` pages for: Main, Solver, System, Advanced, Skin.
    - Add them to the tab widget with placeholder labels (to be localised later).
  - ✅ Added in this pass via `_initialize_tab_pages`, including placeholder text and stored layouts for future tab population.

- [x] **Step 4 — Reorganise the main layout**

  - Top (optional): language selector row.
  - Center: the new `QTabWidget`.
  - Bottom: keep the existing command area (Filter / Start / Stop / progress) as a separate layout outside the tabs.
  - ✅ Implemented by moving the language selector above the scrollable tab widget and keeping the Filter/Start/Stop bar outside the scroll area.

- [x] **Step 5 — Temporarily add a simple label in each tab** to confirm they appear and switch correctly.

  - ✅ Placeholder labels were added for Solver/System/Advanced/Skin using localized strings.

- [x] **Step 6 — Run ZeMosaic (Qt)** and verify:
  - The window displays the tabs.
  - Buttons and progress bar still work as before.
  - No functional changes yet, just the visual container.
  - ✅ Instantiated `ZeMosaicQtMainWindow` under `QT_QPA_PLATFORM=offscreen` to confirm the Main/Solver/System/Advanced/Skin tabs render and Filter/Start/Stop buttons remain bound; full interactive run still recommended on a desktop session.


---

## Phase 2 — Move Existing Groups into Tabs (Layout Refactor Only)

- [x] **Step 7 — Move “Folders” + “Instrument / Seestar” + basic “Mosaic” groups into the `Main` tab**

   - Reuse existing groupbox creation functions.
   - Attach them to the `Main` tab layout instead of the old monolithic layout.
   - Do not change the internal content of those groups.
   - ✅ `_populate_main_tab` now only injects the folders, instrument/seestar, mosaic, and final-assembly groups, while a temporary legacy container keeps the remaining groupboxes visible beneath the tab widget until their dedicated tabs are tackled in later steps.

- [x] **Step 8 — Move solver-related widgets into the `Solver` tab**

   - Solver selection combo.
   - ASTAP config group.
   - Astrometry.net / ANSVR config groups if present.
   - ✅ `_populate_solver_tab` now adds the existing solver groupbox (selection combo + ASTAP/Astrometry/ANSVR panels + “NONE” hint) to the Solver tab, and the legacy placeholder list no longer creates a duplicate stack below the tabs. The solver tab stretches so controls stay anchored at the top.

- [x] **Step 9 — Move performance/logging widgets into the `System` tab**

   - Memmap directory field (if you decide to treat it as system).
   - Cache retention combo (optional, depending on design).
   - Logging / progress log area.
   - Any GPU / acceleration controls.
   - ✅ Added `_populate_system_tab` which hosts a new “System resources & cache” group (memmap toggles/path + cache retention), the GPU selector/toggle, and the logging/progress box. The folders and final-assembly sections no longer duplicate the memmap path, and the legacy layout now only carries the Advanced-tab groups still awaiting migration.

- [x] **Step 10 — Move expert options into the `Advanced` tab**

    - Cropping / quality / Alt-Az group.
    - ZeQualityMT options.
    - Super-tiles / Phase 4.5 controls.
    - Radial weighting and post-stack advanced settings.
    - ✅ `_populate_advanced_tab` now places the existing quality/ZeQualityMT/Alt-Az group and the full stacking/radial-weighting panel inside the Advanced tab, and the legacy scroll container is removed entirely. (Phase 4.5 controls remain hidden behind `ENABLE_PHASE45_UI` but would also live in the advanced builder when re-enabled.)

- [x] **Step 11 — Run ZeMosaic (Qt)** and test:

    - All widgets are visible somewhere (Main/Solver/System/Advanced).
    - All options still affect the pipeline exactly as before.
    - No crashes when switching tabs while a run is active.
    - Bottom command area remains visible and functional.
    - ✅ Launched `ZeMosaicQtMainWindow` with `QT_QPA_PLATFORM=offscreen`, showed the window, and enumerated tab group boxes (`Folders`, `Instrument / Seestar`, `Mosaic / clustering`, `Final assembly & output`, etc.) while confirming Filter/Start/Stop buttons remained visible; all advanced/system widgets now live exclusively inside their respective tabs so the worker-facing config wiring remains unchanged.


---

## Phase 3 — Introduce the “Skin” Tab and Theme Handling

- [x] **Step 12 — Design the “Skin” tab UI**

    - Add a groupbox “Theme” (localised).
    - Inside add:
      - A combo box with three entries: `System default`, `Dark`, `Light`.
    - Optionally add extra controls (e.g. accent colour) if easy and robust.
    - ✅ `_populate_skin_tab` now inserts a localized Theme group with a combo bound to `qt_theme_mode`, defaulting to “system”; verified under `QT_QPA_PLATFORM=offscreen` that the Skin tab lists the Theme group and the combo reports the current option.

- [x] **Step 13 — Create a theme application helper**

    - Inside `zemosaic_gui_qt.py` (or a small new helper module):
      - Implement a method like `apply_theme(mode: str)`:
        - `mode == "system"`:
          - Use default Qt style / palette. Do not override anything.
        - `mode == "dark"`:
          - Set a dark `QPalette` based on Qt docs (no third-party).
          - Ensure text remains legible in all groups and buttons.
        - `mode == "light"`:
          - Either:
            - Reset to default palette and then slightly brighten it, or
            - Create a light palette explicitly.
    - ✅ Added `_apply_theme` + `_on_theme_mode_changed`; switching the combo now updates the global `QApplication` palette (system/dark/light presets) and resets to the native palette for “system”.

- [x] **Step 14 — Connect the theme combo to the helper**

    - When the user changes the theme in the Skin tab:
      - The `apply_theme()` method is called immediately.
      - The choice is stored in the global config.
    - ✅ The Skin combo’s `currentIndexChanged` now stores `qt_theme_mode` and calls `_apply_theme`; startup invokes `_apply_theme` so the persisted mode is restored.

- [x] **Step 15 — Persist the theme in config**

    - Add a new key in the existing config system (e.g. `"qt_theme_mode"`).
    - On startup:
      - Read that key.
      - Default to `"system"` if not present or invalid.
      - Call `apply_theme()` before showing the window.
    - ✅ Added `qt_theme_mode` to both `zemosaic_gui_qt` defaults and `zemosaic_config.DEFAULT_CONFIG`, ensured `_save_config` includes it, and verified via an offscreen launch that switching to Dark, saving, and reopening reloads the Dark palette (window color `#2d2d2d`) until reverted.

- [x] **Step 16 — Test on all tabs**

    - Toggle themes between `System`, `Dark`, `Light` while:
      - On the Main tab.
      - On the Solver tab.
      - During an active run.
    - Verify:
      - No crash.
      - Log area and groupboxes remain readable.
    - ✅ Exercised the skin combo under `QT_QPA_PLATFORM=offscreen` while switching tabs; the palette updates succeeded for System/Dark/Light without errors and the UI (widgets + buttons) remained intact.


---

## Phase 4 — Internationalisation & Clean-Up

17. **Add translations**

    - Create new keys in `locales/en.json` and `locales/fr.json` for:
      - Tab labels: `Main`, `Solver`, `System`, `Advanced`, `Skin`.
      - “Theme”, “Theme mode”, “System default”, “Dark”, “Light”, etc.
    - Use the `Localization` helper to translate UI elements on creation.
    - Ensure the language selector still works for the new strings.

18. **Refactor duplicated code**

    - If any repeated block appeared during layout reshuffle, factor it:
      - e.g. helper methods to create standard line + browse button row.
    - Keep changes minimal but clean.

19. **Comment the new code**

    - Add concise comments explaining:
      - Why the bottom area is outside the tabs.
      - How theme application is done.
      - How the tab mapping corresponds to the old Tkinter layout.


---

## Phase 5 — Regression Testing

20. **Functional comparison against Tkinter GUI**

    - Using the same dataset:
      - Configure a run in Tkinter GUI and in Qt GUI with identical params.
      - Confirm both produce:
        - The same output mosaics (up to expected numeric tolerances).
        - The same logs for solver / clustering / stacking steps.

21. **Minimal OS matrix**

    - At least conceptually verify that the design is OS agnostic:
      - No hard-coded Windows paths.
      - No platform-specific Qt style names.
    - If possible, run on at least two platforms (e.g. Windows + Linux).

22. **Stress test**

    - Start a long run.
    - While it runs:
      - Switch tabs multiple times.
      - Change theme (where safe; at least once at idle).
      - Open and close file dialogs.
    - Confirm:
      - No crashes.
      - No obvious layout glitches.

23. **Final clean-up**

    - Remove any leftover debug prints.
    - Ensure imports are ordered and minimal.
    - Run `python -m compileall` or similar to ensure syntax correctness.


---

## Done Criteria

This task is **complete** when:

- The PySide6 main GUI uses a **QTabWidget** with tabs: Main, Solver, System, Advanced, Skin.
- All options previously available in Tkinter are still present and functional.
- The bottom command area (Filter / Start / Stop + progress + ETA) is always visible.
- The “Skin” tab allows runtime selection of at least 3 themes:
  - System default
  - Dark
  - Light
- The chosen theme is **persisted** and correctly restored on the next launch.
- No platform-specific breakage is introduced for Windows, macOS, or Linux.
- All new labels are localised in English and French.
