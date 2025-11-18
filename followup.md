# FOLLOW-UP TASKS — CROSS-PLATFORM COMPATIBILITY CHECKLIST

This checklist guides you through concrete steps to harden Windows/macOS/Linux compatibility.

---

## 0. Quick inventory of OS-sensitive spots

- [x] Grep the codebase for patterns that are often OS-specific:
  - `C:\\` or `C:/`
  - `/home/`
  - `os.system(`, `subprocess`
  - `\\` used inside paths
  - `APPDATA`, `LOCALAPPDATA`, `Program Files`
  - `xdg-open`, `open ` (macOS), `explorer.exe`
- [x] Make a small list of files that contain such patterns. These will be your main targets.

Goal: identify where Windows-only or Linux-only assumptions live in the code today.

---

## 1. Paths and files — enforce `pathlib`

### 1.1 General rules

- [x] In all modules that touch file paths (`run_zemosaic.py`, GUIs, worker, utils, astrometry, etc.):
  - Replace `os.path.join(...)` and string concatenations with `Path` operations where possible.
  - For example, turn:
    - `os.path.join(base, "subdir", "file.fits")`
    - into: `Path(base) / "subdir" / "file.fits"`.
  - Status: path handling is now centralized through `pathlib` helpers everywhere (filter GUI, worker, astrometry, etc.), with env/`~` expansion implemented via `core.path_helpers.expand_to_path`.
- [x] Ensure any path literals are **relative** to:
  - The project root (when running from source), or
  - The package/executable directory when frozen (see 1.2).
  - Status: scripting/helpers that previously assumed the working directory (e.g. `translate_is.py`) now resolve assets via `get_app_base_dir()` fallbacks, so resource lookups no longer rely on absolute or CWD-specific literals.

### 1.2 Handling “frozen” versus “source” execution

- [x] In a central place (e.g. `zemosaic_utils.py`), add a helper:

  ```python
  def get_app_base_dir() -> Path:
      if getattr(sys, "frozen", False):
          return Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else Path(sys.executable).parent
      return Path(__file__).resolve().parent
 Use get_app_base_dir() to locate:

icon/ folder

locales/ folder

Other static resources

 Replace any direct use of __file__ for resource lookup by calls to that helper.

Goal: icons/locales/resources resolve correctly both in “plain Python” and in frozen executables on all platforms.

Status: `get_app_base_dir()`/`get_user_config_dir()` now live in `zemosaic_utils` and are consumed by GUIs, config, translation scripts, alignment helpers, and solver settings so every resource (icons, locales, helper JSON/scripts) resolves correctly for both source and frozen builds.

2. Config, logs and temp directories
2.1 Config storage
- [x] Use `get_user_config_dir()` (helper now in `zemosaic_utils`) for every persistent config: the main GUI config, worker logs/memmaps, solver settings, and even helper exports (Qt filter CSV) all live under the per-user ZeMosaic directory so Windows/macOS/Linux share the same convention.

2.2 Temp directories
- [x] Added `zemosaic_utils.get_runtime_temp_dir()` so every component shares a dedicated subfolder inside the system temp directory; worker/astrometry temp allocations (`mkdtemp`, lock dirs, memmaps) now route through this helper with per-user fallbacks, eliminating ad-hoc `/tmp`/`C:\Temp` usage.

3. GUIs (Tk and Qt) — cross-platform behaviour
3.1 Tk GUI (zemosaic_gui.py, zemosaic_filter_gui.py)
 Verify window icons are loaded using get_app_base_dir() / "icon" / ....

 Ensure file dialogs that select folders/files use:

Native Tk APIs (filedialog.askopenfilename, askdirectory) without OS-specific assumptions.

 Check window geometry saving / restoring:

Use platform-agnostic strings (e.g. "800x600+100+100").

Avoid any behaviour that depends on Windows window managers only.

3.2 Qt GUI (zemosaic_gui_qt.py, zemosaic_filter_gui_qt.py)
 Verify icons are loaded via QIcon(str(get_app_base_dir() / "icon" / "...")).

 Check menubar / toolbar usage:

On macOS, main menus can behave differently; ensure you don't rely on Windows-only shortcuts.

 Ensure any file path used in Qt widgets (lists of images, logs, etc.) is handled via Path and converted to string only at the UI boundary.

3.3 Locale loading
 In zemosaic_localization.py, ensure:

Locale files are loaded from get_app_base_dir() / "locales".

File opening uses encoding="utf-8".

Goal: both GUIs must work out of the box on all OSes, with icons and translations.

4. Multiprocessing and worker model
 Inspect modules using multiprocessing (likely zemosaic_worker.py / stacker code).

 Ensure:

Any process spawning is invoked from inside if __name__ == "__main__":.

Worker target functions are top-level functions (importable).

No global state is mutated at import time in a way that breaks with the spawn start method on Windows/macOS.

 Where necessary, guard code that should only run in the main process.

Goal: avoid deadlocks or crashes that only happen on Windows because of spawn.

5. External tools and shell commands
 Search for usage of os.system, subprocess.run, or use of external binaries (ASTAP, astrometry.net, etc.).

 For each case:

Avoid hardcoded executable paths like "C:\\Program Files\\...".

Look up paths from:

Config file entries, or

Environment variables, or

Standard system PATH.

 If you need to “open a folder” in the OS file explorer, create a helper in zemosaic_utils.py:

python
Copier le code
def reveal_in_file_explorer(path: Path) -> None:
    system = platform.system()
    if system == "Windows":
        subprocess.run(["explorer", str(path)], check=False)
    elif system == "Darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)
and use it instead of inline OS-specific commands.

Goal: all external tool usage is configurable and does not assume a specific OS layout.

6. Logging, encoding, and console behaviour
 Ensure log files are opened with encoding="utf-8".

 Avoid assumptions about console encoding (like cp1252).

 Do not rely on ANSI escape codes for colours unless protected by checks (Windows cmd may need special handling). If colours are used, they should degrade gracefully.

7. Sanity tests per platform (to run manually later)
- [x] Document a repeatable smoke-suite for Windows/macOS/Linux (source + frozen) that exercises Tk + Qt GUIs, the filter UI, worker, and resource loading so regressions can be caught quickly.
  - Status: Checklist below; update whenever a new feature introduces OS-specific risk.

### Windows (source install)
1. Start from a clean profile by removing `%APPDATA%\ZeMosaic` (the `get_user_config_dir()` location).
2. `pip install -r requirements.txt`; then run `python run_zemosaic.py` and confirm the Tk GUI launches with the ZeMosaic icon and English strings.
3. In Tk, open the sample `example/lights` directory, run the Filter dialog, and launch the worker; confirm output appears under `example/out` and logs land in `%APPDATA%\ZeMosaic\logs`.
4. `pip install PySide6` and launch `python run_zemosaic.py --qt-gui`; verify the Qt UI paints icons/toolbars correctly, menus remain responsive, and closing/reopening restores the previous window geometry from the config.
5. Launch `python zemosaic_filter_gui.py` and `python zemosaic_filter_gui_qt.py` separately, confirm the global WCS export+reopen workflow works and paths are shown with Windows separators.
6. From either GUI use “Reveal output folder” (if available) or `zemosaic_utils.reveal_in_file_explorer` via the log panel to ensure Explorer opens the path without shell errors.

### macOS (source install)
1. Reset `~/Library/Application Support/ZeMosaic` and install deps via `python3 -m pip install -r requirements.txt`.
2. Launch `python3 run_zemosaic.py` (Tk) and ensure the dock icon uses `icon/zemosaic.icns`, menus appear in the macOS global menubar, and file dialogs default to the last used directory.
3. Install PySide6, run `python3 run_zemosaic.py --qt-gui`, and verify Retina rendering, native title bar buttons, and that keyboard shortcuts use the Command key (Open, Save plan, Quit).
4. Open `example/lights`, kick off a short worker run and check that the progress window stays on top and logs stream to `~/Library/Application Support/ZeMosaic/logs`.
5. Run both filter GUIs, confirm drag-and-drop and the “Force Seestar workflow” toggle update the generated worker plan, and ensure macOS Gatekeeper prompts do not appear because all binaries stay inside the project tree.
6. Trigger the “open containing folder” helper to confirm it uses `open <path>` without quoting issues.

### Linux (source install)
1. Clear `~/.config/ZeMosaic`, install deps (`python3 -m pip install -r requirements.txt`), and run `python3 run_zemosaic.py` in Tk mode under both X11 and Wayland sessions.
2. Use Tk GUI to load `example/lights`, confirm locale switching (Settings → Language) reloads `locales/*.json`, and run a small stack to verify temporary files go to `$XDG_RUNTIME_DIR` or `/tmp` via `get_runtime_temp_dir()`.
3. Install PySide6 (ensure Qt platform plugin is present), run `python3 run_zemosaic.py --qt-gui`, and verify the app respects the current theme, uses forward slashes in UI labels, and that closing the window shuts down background workers cleanly.
4. Run `python3 zemosaic_filter_gui.py` and `python3 zemosaic_filter_gui_qt.py`, ensuring folder pickers default to `$HOME` and that symbolic links in the dataset path are preserved.
5. Manually start `python3 zemosaic_worker.py --plan <path/to/generated_plan.json>` to confirm multiprocessing spawn/fork both work (especially on distributions defaulting to `spawn` such as Ubuntu 24.04).
6. Verify `reveal_in_file_explorer()` uses `xdg-open` and prints a friendly warning if `xdg-open` is missing.

### Frozen-build smoke checks
1. Build via `python -m PyInstaller ZeMosaic.spec` (Windows) or the provided `.spec`/`.spec` equivalents on macOS/Linux; ensure `icon/`, `locales/`, and sample data are bundled via `get_app_base_dir()`.
2. Run the frozen Tk build on each OS, open `example/lights`, and confirm relative paths resolve even when launched from another directory.
3. Run the frozen Qt build (when PySide6 is included) and ensure translations, icons, and solver discovery behave the same as the source version.
4. On Windows/macOS verify double-clicking the packaged executable respects the per-user config directory and does not write next to the executable; on Linux verify the AppDir/AppImage respects `$XDG_CONFIG_HOME`.
5. For each platform, unplug any optional external solvers (e.g., ASTAP) and confirm the UI shows the “missing solver” dialog instead of crashing.

Once all steps above pass on at least two OSes (ideally three), we can sign off the release as cross-platform ready.
