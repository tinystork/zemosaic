# ZeMosaic Qt Port — Follow-up & Checklist

This file tracks the progress of the PySide6 (Qt) GUI migration and related tasks.

> **Task C Guard — Commit Checklist**
> - Run `git status followup.md` before committing.
> - If the file is not staged, update the checklist or add a note explaining the session’s progress.

<!-- Task C Guard: Do not complete a session without editing this file. -->

- Before each coding session, the agent MUST:
  - Read this file.
  - Identify the next unchecked item.
  - Work only on that item (or that small coherent group), unless otherwise instructed.
- After each session, update the checkboxes and notes.
- Never mark a task as completed if the implementation has not been verified.

---

## Phase 1 — Qt skeleton and guarded imports

**Goal:** Get minimal Qt files and a backend selector in place, without changing existing behavior.

- [x] Create `zemosaic_gui_qt.py` with minimal QMainWindow and run_qt_main
- [x] Create `zemosaic_filter_gui_qt.py` with stub launch_filter_interface_qt
- [x] Add backend selector in `run_zemosaic.py`
- [x] Guard all PySide6 imports (try/except ImportError)


## Phase 2 — Basic Qt Main Window structure

- [x] Create Qt layout placeholders (folders, ASTAP, mosaic, crop/quality, log/progress)
- [x] Add Start/Stop placeholders
- [x] Load config in Qt UI
- [x] Load localization in Qt UI


## Phase 3 — Layout and widgets mirroring Tk GUI

### 3.1 Folders and basic paths
- [x] Input folder controls
- [x] Output folder controls

### 3.2 ASTAP configuration
- [x] ASTAP executable path widget
- [x] ASTAP data directory widget
- [x] Downsample / sensitivity / radius widgets

### 3.3 Mosaic / clustering parameters
- [x] Cluster threshold
- [x] Target master tiles
- [x] Orientation split degrees

### 3.4 Cropping / quality / Alt-Az
- [x] Quality crop settings
- [x] Alt-Az cleanup widgets
- [x] ZeQualityMT toggles
- [x] Two-pass coverage renorm toggles

### 3.5 Logging / progress panel
- [x] Qt log area
- [x] Qt progress bar
- [x] Phase / ETA / elapsed labels

---

## Phase 4 — Worker integration and threading

- [x] Qt worker wrapper using QThread or queue polling
- [x] Progress, phase, and log message signals
- [x] Qt Start button launching worker
- [x] Qt Stop/Abort button stopping worker
- [x] Proper thread cleanup


## Phase 5 — Qt Filter GUI

- [x] Qt filter dialog skeleton
- [x] Table/list of files
- [x] Matplotlib preview via FigureCanvasQTAgg
- [x] Stream scan mode
- [x] Clustering / grouping controls
- [x] ASTAP scanning
- [x] Return (filtered_list, accepted, overrides)


## Phase 6 — Polishing and parity with Tk

- [x] Ensure all Tk options exist in Qt
- [x] Verify config read/write consistency
- [x] Full localization
- [x] Add docstrings and comments
- [x] Tk remains the default backend


---

# 🔵 POST-PORTAGE AUDIT TASKS  
Tasks derived from the audit performed after Phase 6 completion.

These MUST now be completed in the order shown below.

---

## Task A — GPU Configuration Parity

**Goal:**  
When the user toggles GPU acceleration in the Qt GUI, the following legacy keys MUST also be updated for backward compatibility:

- `stack_use_gpu`
- `use_gpu_stack`

This ensures that tools and worker components expecting Tk-style GPU config remain consistent when using the Qt GUI.

**Detailed requirements:**

- [x] Update Qt GPU toggle handler to write both new-style and legacy-style GPU config keys.
- [x] Ensure config snapshots match those produced by Tk GUI.
- [x] Ensure switching backend (Qt ↔ Tk) does not change stacked GPU behavior.
- [x] Add notes in followup.md once implemented.


---

## Task B — GPU Helper / ETA Parity

**Goal:**  
Qt GUI must process GPU-helper events the same way Tk GUI does.

The worker emits special payloads including:

- `global_coadd_info_helper_path`
- GPU ETA override messages
- GPU warnings
- GPU-progress metadata

Qt currently treats these as plain log lines. Tk extracts and displays:

- ETA overlays  
- warnings  
- helper info  
- GPU-progress UI feedback  

**Detailed requirements:**

- [x] Extend Qt `_handle_payload` to detect the same event types as Tk.
- [x] Implement Qt-safe equivalents of `_handle_gpu_helper_*`.
- [x] Update Qt UI labels / overlays accordingly.
- [x] Ensure no business logic duplication; reuse worker payloads.


---

## Task D — Worker payload & progress parity

**Goal:**
Bring Qt worker signal handling to full parity with the Tk GUI so that all structured payloads are surfaced.

**Detailed requirements:**

- [x] Consume `ETA_UPDATE`, chrono control, raw/master counter, and cluster override payloads within `_handle_payload`.
- [x] Emit dedicated Qt signals for each payload type and update the main window labels.
- [x] Ensure chrono timers start/stop/reset identically to Tk behavior.
- [x] Confirm worker cancellation propagates to timers and status indicators.


---

## Task E — Progress panel completeness

**Goal:**
Mirror Tk’s logging/progress panel in the Qt UI.

**Detailed requirements:**

- [x] Add UI elements for master tile counts, remaining raw files, and chrono displays.
- [x] Bind the new elements to the signals introduced in Task D.
- [x] Persist the additional labels/values in any UI refresh or config snapshot routines.
- [x] Validate layout matches Tk ordering and terminology.


---

## Task F — Log translation & GPU warnings

**Goal:**
Ensure Qt log rendering behaves like Tk, including localization formatting and GPU helper highlighting.

**Detailed requirements:**

- [x] Pass worker-provided kwargs to the localization formatter.
- [x] Support the same translation key prefixes that Tk accepts.
- [x] Reintroduce GPU warning highlighting / styling equivalent to Tk’s `_is_gpu_log_entry` handling.
- [x] Confirm GPU helper fallback messages remain visible and translated.


---

## Task G — Localization selector

**Goal:**
Expose runtime language switching in the Qt GUI consistent with the Tk combobox.

**Detailed requirements:**

- [x] Add a language selector widget bound to `config["language"]`.
- [x] Invoke `localizer.set_language` and refresh all UI text when changed.
- [x] Persist the selection in config saves and reloads.
- [x] Verify Tk and Qt remain in sync when toggling backends after a language change.


---

## Task C — Enforce Strict followup.md Updating

**Goal:**  
Ensure the coding agent ALWAYS updates this file after completing a task.

**Detailed requirements:**

- [x] Add code comments or agent instructions where needed so the next coding actions always:
  - Mark completed tasks with `[x]`
  - Add notes or caveats
- [x] Ensure this behavior is enforced in subsequent runs.


---

## Notes / Known Issues

(Add here any clarifications or partial work notes related to tasks A/B/C)

- [x] Qt config serialization now normalizes legacy GPU keys to match Tk snapshots.
- [x] Tk and Qt now coerce legacy GPU defaults so backend switching preserves stacking GPU flags across saves.
- [x] Task C guard added to `agent.md` and `followup.md`; both now include explicit staging check instructions.
- [x] 2024-06-30: Audit identified outstanding parity gaps (Tasks B, D-G added); pending implementation.
- [x] 2025-11-14: Task B implemented and verified — Qt `_handle_payload` now emits structured GPU helper events, and `ZeMosaicQtMainWindow` applies ETA overrides and helper tracking in parity with Tk `_log_message`/`_handle_gpu_helper_*`.
- [x] 2025-11-14: Task D implemented — Qt worker now consumes `ETA_UPDATE`, chrono, raw/master counter, and `CLUSTER_OVERRIDE` payloads, emits dedicated signals, updates progress labels, and treats `log_key_processing_cancelled` as a proper cancellation path for timers and status indicators.
- [x] 2025-11-15: Task E implemented — Qt logging/progress panel now mirrors Tk ordering/terminology, exposes master tile counts and remaining raw files, and keeps chrono/ETA displays and resets in sync with worker signals.
- [x] 2025-11-15: Task F implemented — Qt worker now forwards structured `msg_key` + kwargs to the main window, which uses `ZeMosaicLocalization.get(..., **kwargs)` for all user-facing levels, mirrors Tk’s key handling (including `run_*`/`global_coadd_*` prefixes), and highlights GPU-related log entries (including helper fallback warnings) using a dedicated style while keeping the messages localized and visible.
- [x] 2025-11-15: Task G implemented — Qt main window now exposes a language selector combo initialized from `config["language"]`, drives `localizer.set_language(...)`, and relies on shared `zemosaic_config` persistence so Tk and Qt read/write the same language key when switching backends.
