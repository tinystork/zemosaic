

# AGENT MISSION — RESTORE ZEMOSAIC ICONS (TK + QT, MULTI-OS)

You are an autonomous coding agent working on the **ZeMosaic / ZeSeestarStacker** project.

The repository contains (non-exhaustive):

- `run_zemosaic.py`
- `zemosaic_gui.py`           (Tk main GUI)
- `zemosaic_filter_gui.py`   (Tk Filter GUI)
- `zemosaic_gui_qt.py`       (Qt main GUI, PySide6)
- `zemosaic_filter_gui_qt.py` (Qt Filter GUI, PySide6)
- `zequalityMT.py`, `lecropper.py`, etc. (other tools with Tk/GUI bits)
- `core/path_helpers.py`
- `core/tk_safe.py`
- `zemosaic_astrometry.py`, `zemosaic_worker.py`, `solver_settings.py`, etc.
- `icon/` directory with various icon assets  
  (e.g. `zemosaic.ico`, `zemosaic.png`, `zemosaic_64x64.png`, `zemosaic_icon.png`)

During recent multi-OS refactors (Windows/macOS/Linux, PyInstaller friendliness, etc.),  
the **window icons disappeared** in both Tk and Qt GUIs.

Your job is to **restore icon loading for all ZeMosaic GUIs** in a **robust, multi-OS-safe way**,  
**without changing anything else in behaviour** (business logic, layout, GPU logic, etc.).


## GLOBAL GOAL

- Re-establish a **single, robust icon lookup strategy** based on the existing `icon/` folder.
- Apply this icon:
  - to the Tk main window (`zemosaic_gui.py`),
  - to the Tk filter window (`zemosaic_filter_gui.py`),
  - to the Qt main window (`zemosaic_gui_qt.py`),
  - to the Qt filter dialog (`zemosaic_filter_gui_qt.py`).
- The solution must behave correctly on:
  - Windows
  - macOS
  - Linux
  - “frozen” builds (PyInstaller/onefile/onedir) as much as possible.

If an icon cannot be loaded (missing files or unexpected layout),  
the application must still start normally — only log a warning.


## CONSTRAINTS & NON-GOALS

- **DO NOT** modify:
  - any stacking logic
  - solver / astrometry logic
  - GPU / CuPy behaviour
  - existing GUI layouts and widgets (positions, text, callbacks)
  - config keys or semantics.
- **NO new external dependencies** (use only the Python standard library and existing modules).
- **Keep changes local** to:
  - `zemosaic_utils.py` (or another central helper if it already exists),
  - `zemosaic_gui.py`,
  - `zemosaic_filter_gui.py`,
  - `zemosaic_gui_qt.py`,
  - `zemosaic_filter_gui_qt.py`.
- Do not break **headless** usage (e.g. importing modules without actually creating a Tk/Qt window).


## EXISTING CONTEXT (HIGH-LEVEL)

- Tk main GUI (`zemosaic_gui.py`) currently sets an icon using direct filesystem paths
  (`os.path.dirname(__file__) + "icon/zemosaic.ico"` style) and/or platform-specific branches.
  This worked but is fragile when the project is installed as a package or frozen.
- Tk Filter GUI (`zemosaic_filter_gui.py`) already defines a helper `_apply_zemosaic_icon_to_tk(window)`
  that tries to pick an `.ico` on Windows and a `.png` otherwise, but its base path logic may not be
  consistent with the rest of the project.
- Qt main GUI (`zemosaic_gui_qt.py`) defines a helper `_load_zemosaic_qicon()` and calls
  `self.setWindowIcon(icon)` in `ZeMosaicQtMainWindow.__init__`. The icon lookup currently assumes
  a certain base directory and may fail when run from different locations / packaging modes.
- Qt Filter GUI (`zemosaic_filter_gui_qt.py`) also defines `_load_zemosaic_qicon()` but may be missing:
  - a robust base directory resolution,
  - consistent behaviour with the main Qt window,
  - or the actual `setWindowIcon` call in the QDialog / main class.

We want to **centralise “where is the app base dir?” in one helper** and reuse it in all GUI modules.


## DESIGN EXPECTATIONS

1. **Central helper for base directory**

   Implement (or fix, if already present) a helper in `zemosaic_utils.py`:

   ```python
   from pathlib import Path
   import sys
   import os

   def get_app_base_dir() -> Path:
       """
       Return the root directory where ZeMosaic resources live.

       This must work when:
       - running from source (cloned repo),
       - installed as a package,
       - running from a frozen/pyinstaller build.
       """
       # Pseudocode / target behaviour:
       # 1. If we are frozen (PyInstaller), use the directory of the executable.
       # 2. Otherwise, try to locate the 'zemosaic' package directory.
       # 3. Fallback to the directory of this file.
````

You are free to refine the implementation as long as it is:

* robust,
* **independent of the current working directory**,
* and does not crash on import.

2. **Shared icon lookup strategy**

   All GUI modules (Tk and Qt) must converge to the same logic:

   * Compute `icon_dir = get_app_base_dir() / "icon"`.
   * Prefer, in this order:

     * `zemosaic.ico` (Windows only, via `iconbitmap` or `QIcon`)
     * `zemosaic_64x64.png`
     * `zemosaic_icon.png`
     * `zemosaic.png` (if present).
   * If none of these files exists, do nothing and print/log a warning.

3. **Tk specific rules**

   * On **Windows**:

     * If `.ico` exists: use `window.iconbitmap(default=str(ico_path))`.
   * On **non-Windows**:

     * Use `PhotoImage(file=str(png_path))` and `window.iconphoto(True, photo)`.
     * Do **not** require Pillow; rely on Tk’s built-in `PhotoImage` for PNG.

   The helper `_apply_zemosaic_icon_to_tk(window)` in `zemosaic_filter_gui.py` is a good template.
   Make the Tk main window use the same underlying logic so behaviour is consistent.

4. **Qt specific rules**

   * Implement/keep a single `_load_zemosaic_qicon()` in each Qt GUI module (or share one, if feasible,
     without creating circular imports).

   * Use the same `get_app_base_dir()` + `icon_dir` + candidate list as described above.

   * If a valid icon is returned, do:

     ```python
     self.setWindowIcon(icon)
     ```

     in:

     * `ZeMosaicQtMainWindow.__init__` (already done, just make sure the helper works),
     * the main dialog/window class in `zemosaic_filter_gui_qt.py` (filter UI).

   * No other Qt behaviour should be changed (layout, signals/slots, etc.).

## TESTING EXPECTATIONS

You must not add automated tests, but you must keep the code **visibly testable**:

* The code must not crash when importing the modules without actually creating a GUI.
* The code must tolerate missing icon files (no exceptions, only a log/warning).
* Manual checks that should pass after your changes:

  * Launching the Tk backend via `python run_zemosaic.py --backend=tk`:

    * The main Tk window shows the ZeMosaic icon in the title bar (where supported by the OS).
    * Opening the Tk filter GUI (`zemosaic_filter_gui`) shows the same icon.
  * Launching the Qt backend via `python run_zemosaic.py --backend=qt`:

    * The main Qt window shows the ZeMosaic icon.
    * Opening the Qt filter dialog shows the same icon.
  * Running on Windows, macOS and Linux with a valid `icon/` directory must behave consistently.

## SUMMARY

**Mission:**
Restore **ZeMosaic icons** for all Tk and Qt GUIs using a central, multi-OS-safe path helper and shared icon lookup logic, without touching any other behaviour in the application.

Focus strictly on:

* `zemosaic_utils.py` (base dir helper),
* `zemosaic_gui.py` (Tk main icon),
* `zemosaic_filter_gui.py` (Tk filter icon),
* `zemosaic_gui_qt.py` (Qt main icon),
* `zemosaic_filter_gui_qt.py` (Qt filter icon).

Everything else must remain functionally identical.

