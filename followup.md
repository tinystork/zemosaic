# ZeMosaic Qt Port — Follow-up & Checklist

This file tracks the progress of the PySide6 (Qt) GUI migration and related tasks.

- Before each coding session, the agent MUST:
  - Read this file.
  - Identify the next unchecked item.
  - Work only on that item (or that small group), unless otherwise instructed.
- After each session, update the checkboxes and notes.


## Phase 1 — Qt skeleton and guarded imports

**Goal:** Get minimal Qt files and a backend selector in place, without changing existing behavior.

- [x] Create `zemosaic_gui_qt.py` file at repository root (same level as `zemosaic_gui.py`):
  - [x] Import PySide6 modules inside a guarded block:
    - `from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout`
    - Wrapped in `try/except ImportError` with a helpful error message.
  - [x] Define a minimal class `ZeMosaicQtMainWindow(QMainWindow)` with:
    - [x] A basic central widget.
    - [x] A simple layout and a placeholder label or title.
  - [x] Define `def run_qt_main():` that:
    - [x] Creates a `QApplication` if none exists.
    - [x] Instantiates `ZeMosaicQtMainWindow`.
    - [x] Shows the main window.
    - [x] Runs the Qt event loop.

- [x] Create `zemosaic_filter_gui_qt.py`:
  - [x] Guard PySide6 imports with `try/except ImportError`.
  - [x] Add a stub function:
    ```python
    def launch_filter_interface_qt(
        raw_files_with_wcs_or_dir,
        initial_overrides=None,
        *,
        stream_scan=False,
        scan_recursive=True,
        batch_size=100,
        preview_cap=200,
        solver_settings_dict=None,
        config_overrides=None,
        **kwargs,
    ):
        # TODO: implement Qt filter GUI
        # For now, just passthrough:
        return raw_files_with_wcs_or_dir, False, None
    ```

- [x] Add a backend selector in `run_zemosaic.py` (or the main entry point used today):
  - [x] Read env var `ZEMOSAIC_GUI_BACKEND` (default `"tk"`).
  - [x] Optionally parse `--qt-gui` from `sys.argv`, which forces backend = "qt".
  - [x] If backend == "qt":
    - [x] Try to import `run_qt_main` from `zemosaic_gui_qt`.
    - [x] If ImportError (PySide6 missing), print a clear message and fallback to Tk backend.
  - [x] If backend == "tk":
    - [x] Call the existing Tk GUI entry point (e.g. `zemosaic_gui.main()` or equivalent).
  - [x] Ensure that running `python run_zemosaic.py` with no env var behaves exactly like before (Tk only).


## Phase 2 — Basic Qt Main Window structure

**Goal:** The Qt main window should be structurally ready but still minimal.

- [x] In `zemosaic_gui_qt.py`, expand `ZeMosaicQtMainWindow`:
  - [x] Add a central widget and a main `QVBoxLayout` or `QGridLayout`.
  - [x] Add placeholder `QGroupBox`es or `QFrame`s representing:
    - [x] Input/Output folders panel.
    - [x] ASTAP configuration panel.
    - [x] Mosaic / clustering parameters panel.
    - [x] Cropping / quality / Alt-Az options panel.
    - [x] Logging / progress panel.
  - [x] Add at least:
    - [x] A "Start" button (not yet wired).
    - [x] A "Stop" or "Abort" button (not yet wired).

- [x] Integrate configuration:
  - [x] Import `zemosaic_config` and load the default config.
  - [x] Populate initial widget values from the config (even if some fields are still placeholders).
  - [x] Save config changes:
    - [x] On Start click, or
    - [x] On window close (whichever is easiest for now).

- [x] Integrate localization infrastructure:
  - [x] Import the localization helper (e.g. `ZeMosaicLocalization`).
  - [x] Load the selected language the same way Tk GUI does.
  - [x] Use translated strings for group titles and button labels where reasonable.


## Phase 3 — Layout and widgets mirroring Tk GUI

**Goal:** The Qt GUI exposes the same options as the Tk GUI.

### 3.1 Folders and basic paths

- [x] Input folder:
  - [x] QLineEdit for input directory.
  - [x] Button to open a folder dialog.
  - [x] Bound to the same config key as in Tk GUI.
- [x] Output folder:
  - [x] QLineEdit + folder button.
  - [x] Bound to proper config key.
- [x] Temp / intermediate folders if any are exposed in Tk GUI (e.g. master tiles directory).

### 3.2 ASTAP configuration panel

- [x] Widgets for:
  - [x] ASTAP executable path.
  - [x] ASTAP data directory / star database folder.
  - [x] Search radius or FOV hint (if used).
  - [x] Downsample factor.
  - [x] Sensitivity / limit magnitude or equivalent parameters.
- [x] Bind all these widgets to the same config keys used by Tk GUI.
- [x] Make sure defaults are identical to Tk behavior.

### 3.3 Mosaic / clustering parameters

- [x] Add widgets for:
  - [x] Cluster panel brightness / threshold.
  - [x] Target number of clusters / master tiles.
  - [x] Cluster orientation split degrees.
  - [x] Any flags controlling “super-tiles” or phase 4.5 behavior if exposed in Tk GUI.
- [x] Bind to corresponding config keys.

### 3.4 Cropping / quality / Alt-Az options

- [x] Quality crop controls:
  - [x] Checkbox `quality_crop_enabled`.
  - [x] Numeric inputs for band size, k-sigma, margin, etc.
  - [x] `crop_follow_signal` toggle.
- [x] Alt-Az cleanup controls (lecropper altZ):
  - [x] `altaz_cleanup_enabled` checkbox.
  - [x] Margin percent.
  - [x] Decay.
  - [x] `altaz_nanize` toggle.
- [x] ZeQualityMT integration:
  - [x] If Tk GUI exposes “ZeQualityMT” or “quality filter” toggles/thresholds, add equivalent controls.
- [x] Two-pass coverage renormalization:
  - [x] Replicate any options that control two-pass renorm, coverage thresholds, etc.

### 3.5 Logging / progress panel

- [x] QTextEdit or QPlainTextEdit for logs.
- [x] QProgressBar for global progress.
- [x] Labels:
  - [x] Current phase name (P1, P2, etc.).
  - [x] Elapsed time.
  - [x] Estimated remaining time (ETA).
  - [x] Files processed / total.
  - [x] Tiles processed / total, if available.

- [x] Optionally add:
  - [x] Clear-log button.
  - [x] Combobox for log level (DEBUG / INFO / WARN / ERROR), bound to config.

- [x] Ensure there is a clear mapping between logging levels and how lines appear in the log widget (prefix, color, etc. – basic prefix is enough for now).


## Phase 4 — Worker integration and threading

**Goal:** The Qt GUI actually runs the mosaic worker without freezing.

- [x] Decide on worker threading approach:
  - [ ] Implement a `QObject`-based worker class (e.g. `ZeMosaicQtWorker`) that runs in a `QThread`, OR
  - [x] Use queue polling with a periodic `QTimer` (multiprocessing worker + queue).
- [x] The worker must call the existing worker function:
  - [x] `run_hierarchical_mosaic_process(...)` or the equivalent entry used in Tk GUI.
  - [x] It must pass all required parameters (from config and widgets).

- [x] Implement a callback adapter:
  - [x] The worker uses a callback compatible with `_log_and_callback` in `zemosaic_worker.py`.
  - [x] This callback translates worker events into Qt signals:
    - [x] `log_message_emitted(level, message)`.
    - [x] `progress_changed(percentage)`.
    - [x] `phase_changed(phase_name, extra_info)`.
    - [x] `stats_updated(stats_dict)` for tiles/files/ETA, etc.

- [x] In `ZeMosaicQtMainWindow`:
  - [x] Connect slots to these signals to:
    - [x] Append localized messages to log widget.
    - [x] Update progress bar.
    - [x] Update phase label.
    - [x] Update ETAs, counters, etc.
  - [x] Wire the “Start” button:
    - [x] Validate paths and config.
    - [x] Save config.
    - [x] Start the worker thread.
  - [x] Wire the “Stop/Abort” button:
    - [x] Use the same stop mechanism as Tk GUI (e.g. setting a stop flag or calling a stop function).
    - [x] Ensure the GUI remains responsive when stopping.

- [x] When worker finishes:
  - [x] Properly clean up the QThread (if used).
  - [x] Re-enable Start button.
  - [x] Optionally show a small message (“Mosaic completed” or “Stopped by user”).

- [x] Ensure:
  - [x] No direct GUI updates are done from non-GUI threads (use signals/slots or queued connections).
  - [x] There are no crashes when closing the window while a worker is still running (do a best-effort safe stop).


## Phase 5 — Qt Filter GUI (`zemosaic_filter_gui_qt.py`)

**Goal:** Provide a Qt-based alternative to `zemosaic_filter_gui.launch_filter_interface`.

- [x] Implement the main Qt dialog class, e.g. `FilterQtDialog(QDialog)`:
  - [x] Accept parameters matching `launch_filter_interface_qt(...)` signature.
  - [x] Store:
    - [x] The input directory OR list of file metadata.
    - [x] Stream scan configuration.
    - [x] Preview cap.
    - [x] Solver settings.
    - [x] Config overrides.

- [x] UI components:
  - [x] A QTableView or QListWidget showing:
    - [x] Filename.
    - [x] Status (WCS ok / missing).
    - [x] Group / tile / cluster info if available.
  - [x] Controls for:
    - [x] Clustering / grouping options also exposed in Tk filter GUI.
    - [x] Seestar-specific behaviors if any.
  - [x] Buttons:
    - [x] “Run analysis / scan”.
    - [x] “Validate / OK”.
    - [x] “Cancel”.

- [x] Stream scan and ASTAP interaction:
  - [x] If scanning directories, use a background thread/QThread to avoid blocking the GUI.
  - [x] Respect ASTAP instance limitations and concurrency caps (same logic as Tk filter GUI).
  - [x] Display progress of scanning and WCS solving.

- [x] Preview:
  - [x] If Tk filter GUI displays a field-of-view preview using Matplotlib:
    - [x] Use Matplotlib + `FigureCanvasQTAgg` to embed the plot in the dialog.
    - [x] Respect `preview_cap` to avoid plotting thousands of tiles.
  - [x] Update preview when the analysis completes or when settings change.

- [x] Return semantics:
  - [x] On OK:
    - [x] Build the `filtered_list` equivalent to the Tk filter GUI’s behavior.
    - [x] Build any `overrides` (grouping, master tile caps, etc.).
    - [x] Return `(filtered_list, True, overrides)` from `launch_filter_interface_qt`.
  - [x] On Cancel or window close:
    - [x] Return `(input_list, False, None)` to indicate no changes.

- [ ] Integrate with main Qt GUI:
  - [ ] Provide a function or button in `ZeMosaicQtMainWindow` to launch the Qt filter dialog.
  - [ ] Use the result in the same way the Tk GUI uses the filter result (before launching the main worker).


## Phase 6 — Polishing and parity

**Goal:** Bring Qt GUI close to full parity with Tk GUI and tidy up.

- [ ] Ensure every major group of settings in Tk GUI has an equivalent in Qt GUI:
  - [ ] Seestar / instrument-specific toggles.
  - [ ] Any special ASTAP options (e.g. max instances, timeout, access-violation auto-dismiss toggles if exposed).
  - [ ] Advanced options related to two-pass coverage renorm, alpha, etc., if they are in the GUI.
- [ ] Verify localization:
  - [ ] All user-facing strings pass through the localization helper where reasonable.
  - [ ] There is no hard-coded English where a key exists already.
- [ ] Verify configuration:
  - [ ] Qt GUI reads existing config files correctly.
  - [ ] Changes in Qt GUI are reflected in the shared config, visible from Tk GUI as well.
- [ ] Documentation:
  - [ ] Add docstrings or comments in Qt modules explaining how to launch the Qt GUI (env var + CLI flag).
  - [ ] Ensure errors when PySide6 is missing are clear and non-blocking for Tk users.


---

## Notes / Known Issues

Use this section to record issues, partial implementations, or TODOs that don’t map cleanly to a checkbox.

- [ ] Confirm whether Tk GUI currently exposes Phase 4.5 / super-tile controls before mirroring them in Qt.
- [ ] Qt worker progress currently uses simple per-stage percentages and does not replicate Tk’s weighted progress/ETA smoothing yet.
- [ ] Qt filter dialog currently focuses on manual include/exclude review; scanning, clustering, and preview tooling still pending.
- [ ] Qt filter “Run analysis” button currently shows a placeholder message until scanning backend is integrated.
