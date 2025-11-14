# ZeMosaic Qt Port — Follow-up & Checklist

This file tracks the progress of the PySide6 (Qt) GUI migration and related tasks.

- Before each coding session, the agent MUST:
  - Read this file.
  - Identify the next unchecked item.
  - Work only on that item (or that small group), unless otherwise instructed.
- After each session, update the checkboxes and notes.


## Phase 1 — Qt skeleton and guarded imports

**Goal:** Get minimal Qt files and a backend selector in place, without changing existing behavior.

- [ ] Create `zemosaic_gui_qt.py` file at repository root (same level as `zemosaic_gui.py`):
  - [ ] Import PySide6 modules inside a guarded block:
    - `from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout`
    - Wrapped in `try/except ImportError` with a helpful error message.
  - [ ] Define a minimal class `ZeMosaicQtMainWindow(QMainWindow)` with:
    - [ ] A basic central widget.
    - [ ] A simple layout and a placeholder label or title.
  - [ ] Define `def run_qt_main():` that:
    - [ ] Creates a `QApplication` if none exists.
    - [ ] Instantiates `ZeMosaicQtMainWindow`.
    - [ ] Shows the main window.
    - [ ] Runs the Qt event loop.

- [ ] Create `zemosaic_filter_gui_qt.py`:
  - [ ] Guard PySide6 imports with `try/except ImportError`.
  - [ ] Add a stub function:
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

- [ ] Add a backend selector in `run_zemosaic.py` (or the main entry point used today):
  - [ ] Read env var `ZEMOSAIC_GUI_BACKEND` (default `"tk"`).
  - [ ] Optionally parse `--qt-gui` from `sys.argv`, which forces backend = "qt".
  - [ ] If backend == "qt":
    - [ ] Try to import `run_qt_main` from `zemosaic_gui_qt`.
    - [ ] If ImportError (PySide6 missing), print a clear message and fallback to Tk backend.
  - [ ] If backend == "tk":
    - [ ] Call the existing Tk GUI entry point (e.g. `zemosaic_gui.main()` or equivalent).
  - [ ] Ensure that running `python run_zemosaic.py` with no env var behaves exactly like before (Tk only).


## Phase 2 — Basic Qt Main Window structure

**Goal:** The Qt main window should be structurally ready but still minimal.

- [ ] In `zemosaic_gui_qt.py`, expand `ZeMosaicQtMainWindow`:
  - [ ] Add a central widget and a main `QVBoxLayout` or `QGridLayout`.
  - [ ] Add placeholder `QGroupBox`es or `QFrame`s representing:
    - [ ] Input/Output folders panel.
    - [ ] ASTAP configuration panel.
    - [ ] Mosaic / clustering parameters panel.
    - [ ] Cropping / quality / Alt-Az options panel.
    - [ ] Logging / progress panel.
  - [ ] Add at least:
    - [ ] A "Start" button (not yet wired).
    - [ ] A "Stop" or "Abort" button (not yet wired).

- [ ] Integrate configuration:
  - [ ] Import `zemosaic_config` and load the default config.
  - [ ] Populate initial widget values from the config (even if some fields are still placeholders).
  - [ ] Save config changes:
    - [ ] On Start click, or
    - [ ] On window close (whichever is easiest for now).

- [ ] Integrate localization infrastructure:
  - [ ] Import the localization helper (e.g. `ZeMosaicLocalization`).
  - [ ] Load the selected language the same way Tk GUI does.
  - [ ] Use translated strings for group titles and button labels where reasonable.


## Phase 3 — Layout and widgets mirroring Tk GUI

**Goal:** The Qt GUI exposes the same options as the Tk GUI.

### 3.1 Folders and basic paths

- [ ] Input folder:
  - [ ] QLineEdit for input directory.
  - [ ] Button to open a folder dialog.
  - [ ] Bound to the same config key as in Tk GUI.
- [ ] Output folder:
  - [ ] QLineEdit + folder button.
  - [ ] Bound to proper config key.
- [ ] Temp / intermediate folders if any are exposed in Tk GUI (e.g. master tiles directory).

### 3.2 ASTAP configuration panel

- [ ] Widgets for:
  - [ ] ASTAP executable path.
  - [ ] ASTAP data directory / star database folder.
  - [ ] Search radius or FOV hint (if used).
  - [ ] Downsample factor.
  - [ ] Sensitivity / limit magnitude or equivalent parameters.
- [ ] Bind all these widgets to the same config keys used by Tk GUI.
- [ ] Make sure defaults are identical to Tk behavior.

### 3.3 Mosaic / clustering parameters

- [ ] Add widgets for:
  - [ ] Cluster panel brightness / threshold.
  - [ ] Target number of clusters / master tiles.
  - [ ] Cluster orientation split degrees.
  - [ ] Any flags controlling “super-tiles” or phase 4.5 behavior if exposed in Tk GUI.
- [ ] Bind to corresponding config keys.

### 3.4 Cropping / quality / Alt-Az options

- [ ] Quality crop controls:
  - [ ] Checkbox `quality_crop_enabled`.
  - [ ] Numeric inputs for band size, k-sigma, margin, etc.
  - [ ] `crop_follow_signal` toggle.
- [ ] Alt-Az cleanup controls (lecropper altZ):
  - [ ] `altaz_cleanup_enabled` checkbox.
  - [ ] Margin percent.
  - [ ] Decay.
  - [ ] `altaz_nanize` toggle.
- [ ] ZeQualityMT integration:
  - [ ] If Tk GUI exposes “ZeQualityMT” or “quality filter” toggles/thresholds, add equivalent controls.
- [ ] Two-pass coverage renormalization:
  - [ ] Replicate any options that control two-pass renorm, coverage thresholds, etc.

### 3.5 Logging / progress panel

- [ ] QTextEdit or QPlainTextEdit for logs.
- [ ] QProgressBar for global progress.
- [ ] Labels:
  - [ ] Current phase name (P1, P2, etc.).
  - [ ] Elapsed time.
  - [ ] Estimated remaining time (ETA).
  - [ ] Files processed / total.
  - [ ] Tiles processed / total, if available.

- [ ] Optionally add:
  - [ ] Clear-log button.
  - [ ] Combobox for log level (DEBUG / INFO / WARN / ERROR), bound to config.

- [ ] Ensure there is a clear mapping between logging levels and how lines appear in the log widget (prefix, color, etc. – basic prefix is enough for now).


## Phase 4 — Worker integration and threading

**Goal:** The Qt GUI actually runs the mosaic worker without freezing.

- [ ] Decide on worker threading approach:
  - [ ] Implement a `QObject`-based worker class (e.g. `ZeMosaicQtWorker`) that runs in a `QThread`, OR
  - [ ] Use `threading.Thread` with a periodic `QTimer` in the GUI to poll progress from a queue.
- [ ] The worker must call the existing worker function:
  - [ ] `run_hierarchical_mosaic_process(...)` or the equivalent entry used in Tk GUI.
  - [ ] It must pass all required parameters (from config and widgets).

- [ ] Implement a callback adapter:
  - [ ] The worker uses a callback compatible with `_log_and_callback` in `zemosaic_worker.py`.
  - [ ] This callback translates worker events into Qt signals:
    - [ ] `log_message_emitted(level, message)`.
    - [ ] `progress_changed(percentage)`.
    - [ ] `phase_changed(phase_name, extra_info)`.
    - [ ] `stats_updated(stats_dict)` for tiles/files/ETA, etc.

- [ ] In `ZeMosaicQtMainWindow`:
  - [ ] Connect slots to these signals to:
    - [ ] Append localized messages to log widget.
    - [ ] Update progress bar.
    - [ ] Update phase label.
    - [ ] Update ETAs, counters, etc.
  - [ ] Wire the “Start” button:
    - [ ] Validate paths and config.
    - [ ] Save config.
    - [ ] Start the worker thread.
  - [ ] Wire the “Stop/Abort” button:
    - [ ] Use the same stop mechanism as Tk GUI (e.g. setting a stop flag or calling a stop function).
    - [ ] Ensure the GUI remains responsive when stopping.

- [ ] When worker finishes:
  - [ ] Properly clean up the QThread (if used).
  - [ ] Re-enable Start button.
  - [ ] Optionally show a small message (“Mosaic completed” or “Stopped by user”).

- [ ] Ensure:
  - [ ] No direct GUI updates are done from non-GUI threads (use signals/slots or queued connections).
  - [ ] There are no crashes when closing the window while a worker is still running (do a best-effort safe stop).


## Phase 5 — Qt Filter GUI (`zemosaic_filter_gui_qt.py`)

**Goal:** Provide a Qt-based alternative to `zemosaic_filter_gui.launch_filter_interface`.

- [ ] Implement the main Qt dialog class, e.g. `FilterQtDialog(QDialog)`:
  - [ ] Accept parameters matching `launch_filter_interface_qt(...)` signature.
  - [ ] Store:
    - [ ] The input directory OR list of file metadata.
    - [ ] Stream scan configuration.
    - [ ] Preview cap.
    - [ ] Solver settings.
    - [ ] Config overrides.

- [ ] UI components:
  - [ ] A QTableView or QListWidget showing:
    - [ ] Filename.
    - [ ] Status (WCS ok / missing).
    - [ ] Group / tile / cluster info if available.
  - [ ] Controls for:
    - [ ] Clustering / grouping options also exposed in Tk filter GUI.
    - [ ] Seestar-specific behaviors if any.
  - [ ] Buttons:
    - [ ] “Run analysis / scan”.
    - [ ] “Validate / OK”.
    - [ ] “Cancel”.

- [ ] Stream scan and ASTAP interaction:
  - [ ] If scanning directories, use a background thread/QThread to avoid blocking the GUI.
  - [ ] Respect ASTAP instance limitations and concurrency caps (same logic as Tk filter GUI).
  - [ ] Display progress of scanning and WCS solving.

- [ ] Preview:
  - [ ] If Tk filter GUI displays a field-of-view preview using Matplotlib:
    - [ ] Use Matplotlib + `FigureCanvasQTAgg` to embed the plot in the dialog.
    - [ ] Respect `preview_cap` to avoid plotting thousands of tiles.
  - [ ] Update preview when the analysis completes or when settings change.

- [ ] Return semantics:
  - [ ] On OK:
    - [ ] Build the `filtered_list` equivalent to the Tk filter GUI’s behavior.
    - [ ] Build any `overrides` (grouping, master tile caps, etc.).
    - [ ] Return `(filtered_list, True, overrides)` from `launch_filter_interface_qt`.
  - [ ] On Cancel or window close:
    - [ ] Return `(input_list, False, None)` to indicate no changes.

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

- [ ] (example) …
- [ ] …
