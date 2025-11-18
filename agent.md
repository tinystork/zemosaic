# AGENT MISSION — CROSS-PLATFORM COMPATIBILITY (WINDOWS / MACOS / LINUX)

You are an autonomous coding agent working on the **ZeMosaic** project.

The repository contains (at least) the following relevant modules:

- `run_zemosaic.py`
- `zemosaic_gui.py` (Tk)
- `zemosaic_gui_qt.py` (Qt, PySide6)
- `zemosaic_filter_gui.py` (Tk)
- `zemosaic_filter_gui_qt.py` (Qt)
- `zemosaic_worker.py`
- `zemosaic_utils.py`
- `zemosaic_config.py`
- `zemosaic_localization.py`
- `lecropper.py`
- `zewcscleaner.py`
- `zemosaic_astrometry.py`
- `zemosaic_align_stack.py`
- `solver_settings.py`
- `tk_safe.py`
- Locale files: `en.json`, `fr.json`
- Icon assets in an `icon/` folder (multi-OS icons)

Your mission is to perform a **global pass to ensure cross-platform compatibility** for:

- **Windows**
- **macOS**
- **Linux**

…both in **normal “source” execution** (`python run_zemosaic.py`) and in **frozen / packaged** execution (e.g. PyInstaller exe/appdir on Windows, macOS bundles, Linux binaries).

You must:

1. Identify and remove / refactor **OS-specific assumptions**.
2. Make sure **file paths, config paths, temp directories, and icons** are resolved in a cross-platform way.
3. Ensure both **Tk and Qt GUIs** behave correctly on all three platforms.
4. Avoid code that only works on one OS (e.g. Windows shell commands, backslashes, encoding quirks).
5. Keep **backwards compatibility** with existing behaviour as much as possible.


## GLOBAL PRINCIPLES

Across the whole project, follow these rules:

1. **Use `pathlib` for paths**
   - Prefer `pathlib.Path` over manual string concatenation or hardcoded separators.
   - Avoid `"C:\\something"` or `"/home/user"` patterns; use `Path.home()`, `Path.cwd()`, or config entries instead.
   - When converting between `Path` and strings for libraries, do it explicitly via `str(path)`.

2. **Use robust platform detection**
   - Use `sys.platform` or `platform.system()` for OS detection.
   - Avoid brittle checks like `if os.name == "nt":` sprinkled everywhere without comments.
   - If OS-specific behaviour is truly needed (e.g. different default database folder), centralize it in **one helper function** (e.g. `zemosaic_utils.get_default_data_dir()`).

3. **File encodings and line endings**
   - Open text files using UTF-8: `open(path, "r", encoding="utf-8")`.
   - Avoid platform-dependent encodings like `cp1252`.
   - Do not assume LF vs CRLF in logic; line splitting should be tolerant (`splitlines()`).

4. **Temp and config directories**
   - Use `tempfile.gettempdir()` or `tempfile.TemporaryDirectory` for working temp folders.
   - For persistent config, use a consistent **per-user config directory**, e.g. via `platformdirs`/`appdirs` pattern:
     - Windows: `%APPDATA%\ZeMosaic`
     - macOS: `~/Library/Application Support/ZeMosaic`
     - Linux: `~/.config/ZeMosaic`
   - If `zemosaic_config.py` already defines a default config path (e.g. `~/.zemosaic_config.json`), keep it but ensure path resolution is cross-platform and does not assume a drive letter.

5. **No OS-specific shell commands without abstraction**
   - Avoid using:
     - `os.system("start ...")` (Windows only)
     - `os.system("open ...")` (macOS only)
     - `os.system("xdg-open ...")` (Linux only)
   - If needed, wrap these in a helper like `zemosaic_utils.open_in_file_explorer(path)` that:
     - Detects the platform.
     - Uses the appropriate command.
   - Prefer Python built-ins or cross-platform libraries before shell commands.

6. **Multiprocessing / threading**
   - Take into account differences between **spawn** (Windows, macOS by default in recent versions) and **fork** (Linux).
   - Ensure any multiprocessing code:
     - Uses `if __name__ == "__main__":` guards where needed.
     - Does not rely on global mutable state that is only safe with `fork`.
   - In worker modules (e.g. `zemosaic_worker.py`), ensure functions are importable without side effects.

7. **GUI specifics (Tk + PySide6)**
   - Ensure both GUI backends (Tk and Qt) can:
     - Start up correctly on all three OS.
     - Resolve icons, locale files and resources using relative paths from the installed package / executable directory, not from hardcoded locations.
   - Avoid using platform-specific keybindings or fonts unless protected by `if sys.platform == ...` with safe fallbacks.
   - For file dialogs, use `QFileDialog` / Tk file dialogs via standard APIs; avoid raw OS shell dialogs.

8. **External binaries and tools**
   - If any module calls external binaries (e.g. `astrometry.net`, `ASTAP`, external solvers), ensure:
     - Paths are not hardcoded to Windows-style locations.
     - Discovery of executables is done via environment variables or config entries.
     - Errors are gracefully reported if the binary is missing (instead of crashing).

9. **No assumption of CUDA availability or GPU vendor**
   - GPU code **must remain optional** and work on CPU-only environments on all platforms.
   - Windows/macOS/Linux behavior should be symmetric: if CuPy is unavailable or GPU is missing, the code must gracefully fall back.


## SCOPE OF THE REVIEW

You must at least audit and possibly modify:

1. `run_zemosaic.py`
   - Entry point logic (Tk vs Qt selection).
   - PySide6 presence check and error messages.
   - Any platform-specific code used at startup.

2. `zemosaic_gui.py` / `zemosaic_gui_qt.py`
   - Icon loading (multi-platform icons).
   - File dialogs and path handling.
   - Layout/geometry persistence (window size/position) across OSes.
   - Behaviour when config paths or icon paths are missing.

3. `zemosaic_filter_gui.py` / `zemosaic_filter_gui_qt.py`
   - All file I/O (input directories, output folders, logs).
   - Any direct path building logic specific to one OS.

4. `zemosaic_config.py`
   - Config file location and default paths.
   - Use of environment variables or platform-dependent defaults.
   - Ensure expansion of `~` using `Path.home()` is correct on all OS.

5. `zemosaic_utils.py`
   - Helper functions dealing with:
     - Paths
     - Temp directories
     - Shell commands
     - Logging / debug files
   - Introduce or reinforce **central helpers** for OS-specific differences (e.g. where to store logs, how to open a folder).

6. Other modules with I/O or OS interaction:
   - `lecropper.py` (input/output FITS/PNG, temp files).
   - `zewcscleaner.py` (paths to WCS files).
   - `zemosaic_astrometry.py` (external tools, database locations, path concatenation).
   - Any module using `os.path`, `os.system`, `subprocess`, or raw path strings.

You **do not** need to modify astro algorithms or stacking logic beyond what is needed to make their I/O cross-platform safe.


## DELIVERABLES

After your modifications, the project should:

1. Run from source (`python run_zemosaic.py`) on:
   - Windows
   - macOS
   - Linux  
   …with both Tk and Qt GUIs (when PySide6 is installed).

2. Use cross-platform path handling and configuration:
   - No hardcoded `C:\...` or `/home/...` in code.
   - Config and temp paths resolved through helpers.
   - Icons and locale files loading correctly from installed/frozen locations.

3. Be ready for packaging on each OS:
   - Resource paths (`icon/`, `locales/`, etc.) resolved via package-relative logic that works in both normal and frozen modes.
   - As little OS-specific branching as possible, all centralized in utility functions.

4. Fail gracefully when OS-specific features are missing:
   - If an external binary is not present, a clear, localized error message is shown.
   - If PySide6 is missing, fallback to Tk or show a clear message without crashing.

Respect existing behaviour for current users and avoid breaking existing workflows.
