# AGENT MISSION FILE — ZEMOSAIC QT PORT

You are an autonomous coding agent working on the **ZeMosaic** project.

The repository already contains:
- `run_zemosaic.py`
- `zemosaic_gui.py`
- `zemosaic_filter_gui.py`
- `zemosaic_worker.py`
- `zemosaic_localization.py`
- `zemosaic_config.py`
- `zemosaic_utils.py`
- `lecropper.py`
- `zequalityMT.py`
- `tk_safe.py`
- `locales/en.json`, `locales/fr.json`
- various helper modules (astrometry, cleaner, etc.)

Your job is to gradually add a **PySide6 (Qt)** GUI backend, **without breaking** the existing Tkinter-based GUI or the astro/stacking business logic.


## GLOBAL MISSION

**Goal:**  
Introduce a complete PySide6 GUI for ZeMosaic (main window + filter GUI), side-by-side with the existing Tkinter GUI:

- Tkinter GUI must continue to work exactly as before.
- PySide6 GUI must reach feature parity over time (but can start minimal).
- No core business logic must be rewritten (WCS solving, stacking, coverage, alpha, lecropper pipelines, ZeQualityMT, etc. stay in the existing Python files).

**Long-term objective:**

- `zemosaic_gui_qt.py` — main ZeMosaic Qt GUI
- `zemosaic_filter_gui_qt.py` — filter Qt GUI
- A small launcher (e.g. in `run_zemosaic.py`) that selects the backend:
  - Default: Tk
  - Optional: Qt, via environment variable `ZEMOSAIC_GUI_BACKEND=qt` or a CLI flag like `--qt-gui`.

You will NOT accomplish this in a single run.  
You must work **incrementally**, guided by `followup.md`.


## HARD CONSTRAINTS

These rules are **strict** and must always be respected:

1. **Do not delete or break the Tkinter GUI**:
   - Do NOT remove or heavily refactor `zemosaic_gui.py`, `zemosaic_filter_gui.py`, or `tk_safe.py`.
   - Tkinter remains the default backend unless explicitly changed by the user.

2. **Do not rewrite business logic**:
   - Do NOT rewrite:
     - The WCS / stacking / mosaicing algorithms in `zemosaic_worker.py`.
     - The cropping / Alt-Az / alpha / coverage logic in `lecropper.py`.
     - The ZeQualityMT filter logic in `zequalityMT.py`.
     - Configuration loading/saving logic in `zemosaic_config.py`.
     - Localization infra in `zemosaic_localization.py`.
   - You may call these functions, pass parameters, and handle callbacks, but you must not re-implement their internal algorithms inside the GUI.

3. **Keep public APIs backward compatible**:
   - The following must keep working as-is:
     - Tk main GUI entry point (`zemosaic_gui.py` main function or equivalent).
     - `zemosaic_filter_gui.launch_filter_interface(...)` (Tk variant).
     - Worker functions like `run_hierarchical_mosaic_process(...)`.
   - If you add new Qt variants, use *new* functions or modules (`*_qt`) so that existing code paths are untouched.

4. **PySide6 is optional**:
   - Users may not have PySide6 installed.
   - The project must still import and run the Tk GUI even if PySide6 is missing.
   - Qt modules (`zemosaic_gui_qt.py`, `zemosaic_filter_gui_qt.py`) must use guarded imports.

5. **Small, testable steps**:
   - You must never attempt a huge refactor in a single step.
   - Work in **phases** and mark progress in `followup.md`.
   - Do not skip ahead to later phases until previous checkboxes are satisfied.


## WORKFLOW AND BEHAVIOR

Whenever you are invoked to work on this project, you MUST:

1. **Read `agent.md` (this file)** entirely, to understand:
   - The mission.
   - The constraints.
   - The phase structure.

2. **Read `followup.md`**:
   - Identify the next unchecked item in the checklist.
   - Work ONLY on that next item (or that small group of items), unless the user explicitly tells you otherwise.

3. **Plan before coding**:
   - Briefly outline what changes you will make (file by file).
   - Ensure these changes do not violate the constraints.

4. **Apply changes**:
   - Modify only the necessary files.
   - Avoid noisy style-only changes; focus on the current task.
   - Keep code readable and consistent with the surrounding style.

5. **Update `followup.md`**:
   - Check the boxes for the tasks you completed.
   - Optionally add notes or small corrections in the “Notes / Known Issues” section.

6. **Summarize**:
   - After changes, summarize what you did and where.
   - Mention if any TODO or uncertainties were found and where they are left in comments.


## PHASES OVERVIEW

The progress and detail per step are tracked in `followup.md`.  
Here is the overview of the phases you will follow.

### Phase 1 — Qt skeleton and guarded imports

Objective: Add minimal Qt files without changing behavior.

- Create `zemosaic_gui_qt.py` with a minimal `QMainWindow` subclass and a `run_qt_main()` function.
- Create `zemosaic_filter_gui_qt.py` with a placeholder `launch_filter_interface_qt(...)` function.
- Add a simple launcher in `run_zemosaic.py` (or equivalent) that:
  - Reads env var `ZEMOSAIC_GUI_BACKEND`.
  - Optionally parses `--qt-gui` from `sys.argv`.
  - Defaults to Tk if PySide6 is missing or backend is `tk`.
- All PySide6 imports must be inside `try/except ImportError` with a clear error message on failure.

### Phase 2 — Basic Qt Main Window structure

Objective: Make `zemosaic_gui_qt.py` open a real window.

- Implement a basic `ZeMosaicQtMainWindow(QMainWindow)` with:
  - Central widget.
  - A main layout.
  - Placeholder group boxes representing the main panels (folders, ASTAP config, mosaic/cluster, crop/quality, log/progress).
- Load and save configuration using `zemosaic_config`.
- Load localization using existing localization tools.
- No actual worker integration yet; buttons may be placeholders.

### Phase 3 — Layout and widgets mirroring Tk GUI

Objective: Expose all relevant options in the Qt GUI.

- Add QLineEdit / QToolButton for input/output dirs.
- Add ASTAP configuration widgets (paths, data dir, search radius, downsample, sensitivity, etc.).
- Add clustering widgets (cluster thresholds, target groups, orientation split degrees).
- Add crop/quality widgets mirroring:
  - Quality crop settings (`quality_crop_enabled`, `quality_crop_band_px`, etc.).
  - Alt-Az cleanup (`altaz_cleanup_enabled`, margin, decay, NaN options).
  - ZeQualityMT toggles if present.
  - Two-pass coverage renormalization toggles, etc.
- Add a log QTextEdit or QPlainTextEdit and a QProgressBar with labels for ETA, elapsed time, etc.

### Phase 4 — Worker integration and threading

Objective: Run the ZeMosaic worker from Qt without freezing.

- Introduce a worker wrapper class (QThread + QObject or similar) that:
  - Calls `run_hierarchical_mosaic_process` (or the existing worker entry).
  - Accepts the same parameters as the Tk GUI.
  - Exposes Qt signals for:
    - Log messages.
    - Phase changes.
    - Progress percent.
    - ETA / elapsed / other metadata.

- The main window:
  - Connects slots to update log and progress bar.
  - Connects a Start button to launch the worker.
  - Connects a Stop/Abort button to request a stop using the existing stop mechanism (stop flags, etc.).

### Phase 5 — Qt filter GUI

Objective: Provide a Qt version of the filter interface.

- Implement `launch_filter_interface_qt(...)` in `zemosaic_filter_gui_qt.py`, with the same signature and semantics as the Tk `launch_filter_interface`.
- It should:
  - Accept either a directory or a list of file metadata (including WCS info).
  - Optionally run in stream scan mode.
  - Display a list or table of files.
  - Allow user to accept/cancel.
  - Return `(filtered_list, accepted, overrides)`.

- If the Tk filter GUI shows a preview of footprints/coverage using Matplotlib:
  - Use `FigureCanvasQTAgg` to embed Matplotlib plots in the Qt GUI.
  - Respect any `preview_cap` logic to avoid freezing on huge mosaics.

### Phase 6 — Polishing, parity, and cleanup

Objective: approach functional parity with the Tk GUI.

- Ensure every option/toggle present in Tk GUI has a corresponding control in Qt GUI.
- Verify that Qt GUI correctly reads/writes config, localization, and logs.
- Optionally refactor shared GUI logic into a small helper module, without breaking existing APIs.
- Provide small docstrings or comments explaining how to launch Qt vs Tk.


## CODING STYLE AND QUALITY

- Follow PEP 8 as much as possible without reformatting the entire file.
- Use clear class names (`ZeMosaicQtMainWindow`, `FilterQtDialog`, etc.).
- Document Qt-specific behavior in short docstrings.
- Prefer explicit signal names (e.g. `progress_changed`, `log_message_emitted`) over generic names.

- Avoid:
  - Large unrelated refactors.
  - Renaming existing functions or classes unless strictly necessary.
  - Circular imports (if unsure, keep Qt-specific code in the Qt modules).

If you are unsure about any behavior, prefer a minimal/no-op implementation that is clearly marked with a TODO, instead of making assumptions that might break the project.


## HOW TO USE FOLLOWUP.MD

`followup.md` is your **checklist and logbook**.

- Before each coding session:
  - Read `followup.md`.
  - Identify the next unchecked item.
- During the session:
  - Implement only that item (or that small coherent group of items).
- After the session:
  - Update `followup.md`:
    - Mark completed tasks with `[x]`.
    - Add short notes if something remains partially done or has caveats.
- Never mark as `[x]` a task you haven’t actually implemented and verified.


## SUMMARY

Your role:

- Be the disciplined Qt migration agent for ZeMosaic.
- Always respect the constraints and phases in this document.
- Always keep `followup.md` up to date.
- Make incremental, safe progress toward a fully functional PySide6 GUI backend, without ever breaking the existing Tkinter implementation or the core astrophotography logic.
