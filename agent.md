# AGENT MISSION FILE — ZEMOSAIC QT TABBED LAYOUT + SKINS

You are an autonomous coding agent working on the **ZeMosaic** project.

The repository already contains (non-exhaustive):

- `run_zemosaic.py`
- `zemosaic_gui.py`              (Tkinter main GUI – **reference behaviour**)
- `zemosaic_filter_gui.py`      (Tkinter filter GUI)
- `zemosaic_gui_qt.py`          (PySide6 main GUI – work in progress)
- `zemosaic_filter_gui_qt.py`   (PySide6 filter GUI – work in progress)
- `zemosaic_worker.py`
- `zemosaic_config.py` / `zemosaic_config.json`
- `zemosaic_localization.py`
- `locales/en.json`, `locales/fr.json`
- `tk_safe.py`
- various helper modules (astrometry, cleaner, etc.)

Your job now is to:

1. **Refactor the PySide6 main GUI (`zemosaic_gui_qt.py`) to use a tabbed layout**, while keeping **all existing functionality identical** to the current Tkinter GUI (`zemosaic_gui.py`).
2. Add a new **“Skin”** tab to control the UI color theme (dark / light / system default).
3. Keep the Qt GUI fully functional on **Windows, macOS and Linux** (icons, file dialogs, fonts, layout).

The **business logic must not change**. Only the Qt layout and theme handling are allowed to evolve (plus the small config needed for theme selection).


---

## GLOBAL MISSION

### Goal

Introduce a **QTabWidget-based UI** in `zemosaic_gui_qt.py` with the following tabs:

1. **Main** (or “Principal”)
2. **Solver**
3. **System**
4. **Advanced**
5. **Skin**

At the same time:

- Preserve the **current feature set** and behaviour of ZeMosaic as exposed in `zemosaic_gui.py`.
- Ensure that the **bottom command area** stays always visible (outside the tabs):
  - **Filter…** button
  - **Start** button
  - **Stop** button
  - Progress bar + ETA / phase information

The Tkinter GUI must remain untouched and usable. Users can still choose between the old Tk GUI and the new Qt GUI from `run_zemosaic.py` (or the existing launcher logic).


### Non-Goals

- Do **NOT** change the stacking / mosaic / astrometry algorithms.
- Do **NOT** rename or move modules in a way that breaks imports.
- Do **NOT** add heavy third-party dependencies for theming. Use standard PySide6 / Qt features (QPalette, styles, simple stylesheets).
- Do **NOT** regress any existing behaviour compared to `zemosaic_gui.py` and the current `zemosaic_gui_qt.py`.


---

## TAB LAYOUT SPECIFICATION

You will reorganize the existing groupboxes and controls into these tabs.  
For each tab, reuse the existing `QGroupBox` creation helpers where possible (e.g. `_create_folders_group`, `_create_solver_group`, etc.) — just move them into the appropriate tab layouts.

### 1. Tab “Main”

Purpose: all options required for a **typical run**.

Put here:

- **Language selector**
  - The language combo (EN/FR) can be either:
    - at the very top of the window (above tabs), or
    - as the first widget inside the “Main” tab.
  - It must continue to use `zemosaic_localization.py` and the existing i18n mechanism.

- **Folders group**
  - Input folder
  - Output folder
  - Global WCS output path
  - (Optional) Memmap directory – can also be moved to “System” if more appropriate.

- **Instrument / Seestar group**
  - Auto-detect Seestar frames
  - Force Seestar workflow
  - Enable SDS mode by default
  - SDS coverage threshold

- **Mosaic / clustering (basic controls)**
  - Cluster threshold
  - Target groups
  - Orientation split
  - Cache retention (unless you decide to classify it under “System / memory”)

- **Final assembly / output options** (if already exposed in Qt GUI)
  - Method: Reproject co-add / Incremental
  - Save final mosaic as uint16
  - Legacy RGB cube options, etc.

Implementation detail:

- Use a dedicated `QWidget` for the “Main” tab with a vertical layout.
- Add the above groupboxes in a consistent order (roughly: Language → Folders → Instrument → Mosaic).


### 2. Tab “Solver”

Purpose: everything related to **WCS / plate solving**.

Put here:

- **Solver selection**
  - Combo box: ASTAP / Astrometry.net / ANSVR / None (or whatever exists today).

- **ASTAP configuration group**
  - ASTAP executable
  - ASTAP data directory
  - Default search radius
  - Default downsample
  - Default sensitivity
  - Max ASTAP instances (this must keep the same semantics as in Tkinter).

- **Astrometry.net configuration group** (if present)
  - URL, API key, options.

- **ANSVR / notes** widgets if they exist.

- Any solver-specific hints text (e.g. “None = WCS already present”).


### 3. Tab “System”

Purpose: performance, memory, logging, “machine”-level settings.

Put here:

- From Folders / Mosaic:
  - Memmap directory (if not left on “Main”).
  - Cache retention (if you decide it’s more of a memory setting).

- **GPU / acceleration group**
  - Use GPU acceleration when available
  - GPU selector or device info (if present).

- **Logging / progress group**
  - Logging verbosity level combo (if present).
  - “Clear log” button, log view, phase information, ETA readouts.
  - Any text or status widgets related purely to diagnostic / debug.

The idea: the System tab is where users watch logs, performance-related indicators and tune lower-level settings.


### 4. Tab “Advanced”

Purpose: expert and experimental options. A new user should be able to ignore this tab.

Put here:

- **Cropping / quality / Alt-Az**
  - Master tile crop
  - Quality crop
  - Alt-Az cleanup
  - ZeQualityMT / master tile quality gate
  - Two-pass coverage renormalization (if present).

- **Mosaic / clustering advanced**
  - Phase 4.5 / super-tiles controls.
  - Any extra parameters that only affect special workflows (e.g. ZeSupaDupStack).

- **Stacking options (expert)**
  - Radial weighting options:
    - Apply radial weighting
    - Radial feather fraction
    - Minimum radial weight floor
  - Post-stack anchor review / inspection options.
  - Any “experimental” toggles.

Implementation detail:

- Where possible, reuse existing group-creating functions and only move them into this tab.
- If some expert options are currently mixed with basic ones, split them into separate groupboxes (e.g. “Basic mosaic options” vs “Advanced mosaic options”) and place the advanced groupbox here.


### 5. Tab “Skin”

Purpose: configure the visual appearance (theme / colors).

Add a **new groupbox** dedicated to theme selection:

- **Theme mode**
  - Combo box or radio buttons with at least:
    - `System default`
    - `Dark`
    - `Light`
  - “System default” must respect the platform’s default Qt style:
    - Do **not** override palette or style.
  - “Dark” / “Light” should:
    - Either use a custom `QPalette` for the application, or
    - Apply a simple stylesheet.
    - Keep it lightweight and cross-platform.

- **Optional accent controls**
  - You may add simple color pickers or combos for accent / highlight color, but only if this remains simple and robust.
  - All labels must be localised via `zemosaic_localization.py`.

**Theme persistence:**

- Store the selected theme mode and any color settings in the existing config mechanism:
  - Prefer to reuse `zemosaic_config.py` / its JSON file or whichever settings file is already used for GUI-level options.
- On startup:
  - Load the saved theme from config.
  - Apply it before showing the main window.
- When users change the theme from the “Skin” tab:
  - Immediately apply the new theme.
  - Save the setting for the next run.

Cross-platform notes:

- Do not rely on OS-specific APIs.
- Use Qt standard facilities: `QApplication.setStyle`, `QPalette`, and possibly a simple stylesheet string.
- Verify that the theme works on Windows, macOS, and Linux (at least conceptually; no OS-specific code).


---

## BOTTOM COMMAND AREA (OUTSIDE TABS)

Regardless of the active tab, the user must always see:

- Button **Filter…**
- Button **Start**
- Button **Stop**
- Progress bar
- ETA / current phase information (as currently done in the Qt GUI)

Implementation:

- Create a “bottom toolbar” layout separate from the tab widget, e.g.:

  - Top: optional language selector
  - Center: `QTabWidget`
  - Bottom: `QWidget` with a horizontal layout for buttons + a vertical stack for log/progress if needed.

- Reuse existing signal/slot connections. Only move the widgets; do not change what the callbacks do.


---

## INTERNATIONALISATION

- All **new labels** (“Main”, “Solver”, “System”, “Advanced”, “Skin”, theme modes, etc.) must be integrated into the existing i18n system:
  - Add keys to `locales/en.json` and `locales/fr.json`.
  - Use `zemosaic_localization.Localization.tr(...)` (or the current helper) for UI text.
- The language selector in the Qt GUI must continue to function exactly like in the Tkinter GUI: changing language should update visible labels accordingly.


---

## CROSS-PLATFORM REQUIREMENTS

- Do not break current support for **Windows, macOS, Linux**:
  - File/folder dialogs must still open with the system’s native dialog.
  - Icons (open/close/folder/etc.) must still load correctly. Do not introduce platform-specific icon paths.
  - Avoid Windows-only code and any `ctypes` tricks inside the GUI module.

- Keep imports limited to:
  - Standard library
  - Existing project modules
  - PySide6 / Qt modules already used in the project.


---

## IMPLEMENTATION GUIDELINES

- Use clear, explicit layouts:
  - For each tab: `tab_widget = QWidget()`, `layout = QVBoxLayout(tab_widget)`.
  - Add groupboxes to those layouts.
  - Add the tabs to a single `QTabWidget`.

- Minimise code duplication:
  - If you see repeated groupbox construction, factor it into helper methods.

- Preserve all existing signal/slot connections and callback semantics.

- Add comments where new theme code or tab logic lives, so future contributors can easily understand it.

- Re-run / update any unit tests or basic smoke tests if they exist (or provide a simple manual test checklist in `followup.md`).

At the end, the **PySide6 main window** must:

- Expose the same options as the current Tkinter window.
- Behave identically for all existing workflows (standard stack, SDS, ZeSupaDupStack, etc.).
- Provide a clean, tabbed UI suitable for smaller screens.
- Offer theme selection via the new “Skin” tab.
