
# FOLLOW-UP TASKS — ZEMOSAIC QT TABS, SKIN & LANGUAGE

This file describes what the Qt GUI already does, and the **remaining work** to finish the tabbed PySide6 interface, the skin system, and language management.

Keep changes focused, and do not break:
- the existing Tkinter GUI (`zemosaic_gui.py`),
- the astro / stacking business logic,
- or the current backend selection logic in `run_zemosaic.py` (CLI flags + env var).

---

## 1. Current State (Do NOT undo this)

The codebase already contains:

- A working **Tkinter GUI**: `zemosaic_gui.py`
- A working **Qt GUI (PySide6)**: `zemosaic_gui_qt.py`
- A launcher with backend selection: `run_zemosaic.py`, which currently:
  - Accepts `--qt-gui` / `--tk-gui` CLI flags.
  - Honours environment variable `ZEMOSAIC_GUI_BACKEND` (`"qt"` or `"tk"`).
  - Falls back to Tk if PySide6 is unavailable.
  - Optionally displays a small Tk popup to ask the user to choose Tk vs Qt.

For the Qt GUI:

- The main window already uses a **QTabWidget** with tabs:

  `Main | Solver | System | Advanced | Skin`

- The **bottom command area** (outside the tabs) contains:
  - `Filter…` button
  - `Start` button
  - `Stop` button
  - Progress bar + ETA / phase info

  This area must **stay always visible**, regardless of the active tab.

- The **Skin** tab already exposes a simple theme combo:
  - `System` / `Dark` / `Light`
  - and applies a global Qt palette at runtime.

All this should be kept as-is and only extended where specified below.

---

## 2. Task A — Clean up & stabilise tabbed layout (small)

**Files:** `zemosaic_gui_qt.py`, `locales/en.json`, `locales/fr.json`

Goal: make sure the tab layout is consistent and fully localised.

1. Ensure the QTabWidget has exactly these tabs, in this order:

   - `Main`
   - `Solver`
   - `System`
   - `Advanced`
   - `Skin`
   - `Language` (will be added in Task C)

2. Use localisation keys for all tab labels, e.g.:

   - `qt_tab_main`
   - `qt_tab_solver`
   - `qt_tab_system`
   - `qt_tab_advanced`
   - `qt_tab_skin`
   - `qt_tab_language`

   Add entries in `locales/en.json` and `locales/fr.json`.

3. Groupboxes and controls must remain assigned as already decided:

   - **Main**: folders, instrument/Seestar, basic mosaic & final output options.
   - **Solver**: solver selection + ASTAP / Astrometry / ANSVR config.
   - **System**: memmap / cache / GPU / logging.
   - **Advanced**: cropping, quality, Alt-Az, ZeQualityMT, super-tiles, radial weighting, post-stack review.
   - **Skin**: theme-related settings (and later backend choice, see Task B).
   - **Language**: language selector + info (Task C).

This task is mostly a consistency pass: no change in behaviour, just naming and localisation.

---

## 3. Task B — Skin tab: add “Preferred GUI backend” (Tk vs Qt)

**Files:** `zemosaic_gui_qt.py`, `zemosaic_config.py`, `zemosaic_config.json`, `run_zemosaic.py`, `locales/en.json`, `locales/fr.json`

### B.1 — UI in the Skin tab

1. In the **Skin** tab, add a new groupbox:

   - Title (localised): e.g. `qt_group_backend_title` → “Preferred GUI backend”

2. Inside this group, add either:

   - a `QComboBox`, or
   - two `QRadioButton`s,

   with the following choices (values in config must be lowercase):

   - “Classic Tk GUI (stable)” → value `"tk"`
   - “Qt GUI (preview)” → value `"qt"`

   Visible labels should be localised (EN/FR), via keys like:

   - `backend_option_tk`
   - `backend_option_qt`

3. Add a small localised note/label under the controls, e.g.:

   - Key: `backend_change_notice`
   - Text (EN): “Backend change will take effect next time you launch ZeMosaic.”
   - Text (FR): “Le changement de backend prendra effet au prochain lancement de ZeMosaic.”

### B.2 — Persist preference in global config

4. In `zemosaic_config.py`, extend `DEFAULT_CONFIG` with a new key:

   ```python
   "preferred_gui_backend": "tk",  # "tk" or "qt"
````

5. Make sure the existing load/save functions:

   * Load `preferred_gui_backend` from `zemosaic_config.json` if present.
   * When missing, silently fall back to `"tk"`.

6. In `zemosaic_gui_qt.py`:

   * On Qt GUI startup, read `preferred_gui_backend` from config.
   * Initialise the Skin tab radio/combobox accordingly.
   * On user change in the Skin tab, immediately update `config["preferred_gui_backend"]` and save the config to disk.

   The Qt GUI **must NOT** try to switch backends live. This setting is only for future launches.

### B.3 — Use config in run_zemosaic backend selection

7. In `run_zemosaic.py`, update `_determine_backend(argv)` so that the default backend selection becomes:

   1. **CLI flags** (`--qt-gui` / `--tk-gui`) — highest priority.
   2. **Environment variable** `ZEMOSAIC_GUI_BACKEND`.
   3. **Config key** `preferred_gui_backend` (loaded via `zemosaic_config`).
   4. Fallback: `"tk"` if everything else is missing/invalid.

   Implementation guidelines:

   * Import a lightweight helper from `zemosaic_config.py` (e.g. `load_config()` or similar). If needed, create a simple function to read config **without** popping any Tk dialogs.
   * If `preferred_gui_backend` is present and valid (`"tk"` or `"qt"`), treat it as the base `requested_backend` when no CLI flag and no env var are set.
   * If it is missing or invalid, ignore and default to `"tk"`.

8. For `_interactive_backend_choice_if_needed`:

   * If a backend was chosen via CLI or env var, or if `preferred_gui_backend` is set in config, **do not** show the Tk choice popup.
   * Only show the popup if:

     * No CLI flag, no env var, **and**
     * `preferred_gui_backend` is missing in config.

   Optionally, when the user chooses via this popup, you may also write the chosen backend to `preferred_gui_backend` for future runs.

9. All this must remain **cross-platform**:

   * Do not introduce OS-specific logic.
   * The existing Tk warning / choice dialogs in `run_zemosaic.py` should keep working on Windows, macOS, and Linux.

---

## 4. Task C — Language tab & extra locales (ES / PL)

**Files:** `zemosaic_gui_qt.py`, `zemosaic_gui.py` (readonly reference), `zemosaic_localization.py`, `locales/en.json`, `locales/fr.json`, `locales/es.json`, `locales/pl.json`

### C.1 — New “Language” tab in Qt main window

10. Add a new **Language** tab at the end of the QTabWidget:

    `Main | Solver | System | Advanced | Skin | Language`

11. Move the existing Qt language combo (currently in the top bar) **into this Language tab**:

    * Use an existing label key if one already exists (e.g. `language_selector_label`).
    * The combo should remain bound to the same `config["language"]` and call the same localization logic (`localizer.set_language(...)`).

12. The top bar of the Qt window should no longer contain the language selector; the only place to change language is now the **Language** tab.

13. Add a small explanatory text in this tab, e.g.:

    * Key: `language_change_notice`
    * EN: “Language also applies to the classic Tk interface and will be remembered.”
    * FR: “La langue s’applique aussi à l’interface Tk classique et sera mémorisée.”

### C.2 — Extend supported languages (ES & PL)

14. Extend `zemosaic_localization.ZeMosaicLocalization` to support:

    * `"es"` → Spanish
    * `"pl"` → Polish

15. Add two new locale files:

    * `locales/es.json`
    * `locales/pl.json`

    For a first pass, it is acceptable to copy `en.json` into both, so that all keys exist. Actual translations can come later.

16. Update the language combo to list:

    * “English (EN)” → value `"en"`
    * “Français (FR)” → value `"fr"`
    * “Español (ES)” → value `"es"`
    * “Polski (PL)” → value `"pl"`

    Use localised display names with keys like:

    * `language_name_en`
    * `language_name_fr`
    * `language_name_es`
    * `language_name_pl`

### C.3 — Shared behaviour with Tk

17. Ensure that both Qt and Tk GUIs still use the same `config["language"]` and same localization module.

18. After changing language in the Qt Language tab:

    * All visible strings in the Qt window (tabs, groupboxes, buttons, Skin/Language text, etc.) should update.
    * If you close Qt and then start the Tk GUI, it should open directly in the newly selected language.

---

## 5. Task D — Regression tests & Done criteria

19. **Basic run tests**

    * Launch `run_zemosaic.py` with:

      * No flags, no env var.
      * `--tk-gui`
      * `--qt-gui`
    * Check that `preferred_gui_backend` in config correctly biases the default when there is no CLI/env override.
    * Verify that switching Qt ↔ Tk via:

      * Skin tab,
      * or the first-launch popup
        does not break anything (changes only apply on next launch).

20. **Language tests**

    For each language code `en`, `fr`, `es`, `pl`:

    * Start Qt GUI.
    * In Language tab, select the language.
    * Confirm all visible UI is updated.
    * Close Qt, start Tk GUI:

      * Tk should be in the same language.
    * Optionally open the Qt Filter GUI and check that it uses the same locale.

21. **Theme + backend interaction**

    * In Skin tab, test all combinations:

      * Theme: System / Dark / Light.
      * Backend preference: Tk / Qt.
    * Restart ZeMosaic each time to ensure:

      * Backend choice is respected.
      * Theme reload for Qt is correct.
      * Tk backend is unaffected by Qt theme settings.

22. **Cross-platform sanity**

    * Check that no new Windows-only (or macOS-only) code has been introduced in:

      * `zemosaic_gui_qt.py`
      * `run_zemosaic.py`
      * `zemosaic_config.py`
    * File dialogs, icons and logs should behave as before on all platforms.

---

## 6. Done Criteria (summary)

This follow-up is considered complete when:

* [x] The Qt main window exposes tabs: **Main, Solver, System, Advanced, Skin, Language**, all localised.
* [x] The **bottom command bar** (Filter / Start / Stop + progress + ETA) is always visible and unchanged.
* [x] The **Skin** tab:

  * [x] Lets the user choose **System / Dark / Light** theme for Qt.
  * [x] Lets the user choose a **preferred GUI backend** (Tk vs Qt) stored as `preferred_gui_backend` in config.
* [x] The launcher `run_zemosaic.py`:

  * [x] Uses CLI flags > env var > `preferred_gui_backend` > `"tk"` to select backend.
  * [x] Skips the Tk choice popup when a backend is already chosen by config, env, or CLI.
* [x] The **Language** tab:

  * [x] Hosts the language combo (no more language widget in the top bar).
  * [x] Allows choosing EN / FR / ES / PL.
  * [x] Updates both Qt and Tk GUIs via the shared config and localization.
* [x] No existing astro/stacking behaviour is changed and all backends remain cross-platform.

```

Mark all completed task by [x]
