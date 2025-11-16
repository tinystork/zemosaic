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

3. **Add a QTabWidget to `zemosaic_gui_qt.py`**

   - In the main window init:
     - Create a `QTabWidget` instance.
     - Create empty `QWidget` pages for: Main, Solver, System, Advanced, Skin.
     - Add them to the tab widget with placeholder labels (to be localised later).

4. **Reorganise the main layout**

   - Top (optional): language selector row.
   - Center: the new `QTabWidget`.
   - Bottom: keep the existing command area (Filter / Start / Stop / progress) as a separate layout outside the tabs.

5. **Temporarily add a simple label in each tab** to confirm they appear and switch correctly.

6. **Run ZeMosaic (Qt)** and verify:
   - The window displays the tabs.
   - Buttons and progress bar still work as before.
   - No functional changes yet, just the visual container.


---

## Phase 2 — Move Existing Groups into Tabs (Layout Refactor Only)

7. **Move “Folders” + “Instrument / Seestar” + basic “Mosaic” groups into the `Main` tab**

   - Reuse existing groupbox creation functions.
   - Attach them to the `Main` tab layout instead of the old monolithic layout.
   - Do not change the internal content of those groups.

8. **Move solver-related widgets into the `Solver` tab**

   - Solver selection combo.
   - ASTAP config group.
   - Astrometry.net / ANSVR config groups if present.

9. **Move performance/logging widgets into the `System` tab**

   - Memmap directory field (if you decide to treat it as system).
   - Cache retention combo (optional, depending on design).
   - Logging / progress log area.
   - Any GPU / acceleration controls.

10. **Move expert options into the `Advanced` tab**

    - Cropping / quality / Alt-Az group.
    - ZeQualityMT options.
    - Super-tiles / Phase 4.5 controls.
    - Radial weighting and post-stack advanced settings.

11. **Run ZeMosaic (Qt)** and test:

    - All widgets are visible somewhere (Main/Solver/System/Advanced).
    - All options still affect the pipeline exactly as before.
    - No crashes when switching tabs while a run is active.
    - Bottom command area remains visible and functional.


---

## Phase 3 — Introduce the “Skin” Tab and Theme Handling

12. **Design the “Skin” tab UI**

    - Add a groupbox “Theme” (localised).
    - Inside add:
      - A combo box with three entries: `System default`, `Dark`, `Light`.
    - Optionally add extra controls (e.g. accent colour) if easy and robust.

13. **Create a theme application helper**

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

14. **Connect the theme combo to the helper**

    - When the user changes the theme in the Skin tab:
      - The `apply_theme()` method is called immediately.
      - The choice is stored in the global config.

15. **Persist the theme in config**

    - Add a new key in the existing config system (e.g. `"qt_theme_mode"`).
    - On startup:
      - Read that key.
      - Default to `"system"` if not present or invalid.
      - Call `apply_theme()` before showing the window.

16. **Test on all tabs**

    - Toggle themes between `System`, `Dark`, `Light` while:
      - On the Main tab.
      - On the Solver tab.
      - During an active run.
    - Verify:
      - No crash.
      - Log area and groupboxes remain readable.


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
