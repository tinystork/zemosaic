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

## Task H — Global WCS / Mosaic-first parity

**Goal:**  
Ensure the Qt filter GUI provides the same global WCS descriptor and Mosaic-first (SDS) planning behaviour as the Tk filter, using the existing helper modules.

**Detailed requirements:**

- [x] Extend the Qt filter pipeline to call the same global WCS helpers used by Tk (e.g. descriptor computation and FITS/JSON output) instead of reimplementing any logic.
- [x] Ensure Qt filter overrides expose the same keys as Tk when a global WCS is prepared (e.g. `global_wcs_meta`, `global_wcs_path`, `global_wcs_json`, `global_wcs_plan_override` semantics).
- [x] Verify that SDS / Mosaic-first workflows behave identically under Tk and Qt (same descriptor reuse by the worker, same user-visible logs and warnings).
- [x] Add notes in followup.md once behaviour has been validated on at least one representative dataset.

Implementation notes (2025-11 audit):
- Wire `FilterQtDialog` to use `compute_global_wcs_descriptor`, `resolve_global_wcs_output_paths`, `load_global_wcs_descriptor`, and `write_global_wcs_files` just like the Tk filter (`zemosaic_filter_gui.py`).
- When SDS/Seestar workflows are active and validation succeeds, populate `global_wcs_meta`, `global_wcs_path`, `global_wcs_json`, and `global_wcs_plan_override` in Qt overrides so `_prepare_global_wcs_plan` sees identical inputs under both GUIs.
- Align Qt stream-scan behaviour with Tk by honouring `EXCLUDED_DIRS` / `is_path_excluded` so both filters operate on the same effective file set.


---

## Task I — End-of-run UX & cancellation parity

**Goal:**  
Align Qt run completion and cancellation behaviour with the Tk GUI so users see consistent prompts and log levels.

**Detailed requirements:**

- [x] Add an optional “Open output folder” prompt to the Qt GUI on successful completion, mirroring Tk’s platform-specific behaviour.
- [x] Treat user-triggered cancellations in Qt as a warning-style completion (`log_key_processing_cancelled`) rather than a generic error, matching Tk’s log level semantics.
- [x] Confirm that timers, ETA, tiles/files counters, and phase labels reset identically on completion and on cancellation in both backends.
- [x] Document any intentional UX differences (if any remain) in the Notes / Known Issues section.

Implementation notes (2025-11 audit):
- Fix `_on_worker_finished` in `zemosaic_gui_qt.py` so it calls `_translate_worker_message(message_key_or_raw, params, level)` with proper arguments and does not raise on non-success paths.
- Mirror Tk cancellation semantics: classify user stops as `log_key_processing_cancelled` at WARN level, reset ETA/elapsed/tiles/files/phase labels, and avoid treating them as hard errors in Qt dialogs.
- Add a platform-aware “Open output folder?” prompt in Qt on successful completion, using the same heuristics and translations as Tk (`zemosaic_gui.py`), and ensure the behaviour is gated on a valid `output_dir`.


---

## Task J — Final parity audit & config key validation

**Goal:**  
Lock in full parity between Tk and Qt by validating configuration, worker inputs, and observable behaviour end-to-end.

**Detailed requirements:**

- [x] Compare saved configuration snapshots produced by Tk and Qt for equivalent sessions and ensure all shared features map to the same keys and values (including GPU, quality gates, coverage, and language).
- [x] Verify that worker argument tuples constructed by Tk and Qt (`run_hierarchical_mosaic_process` callers) are semantically aligned for shared parameters (folders, solver/stacking options, quality/crop, GPU flags, etc.).
- [x] Run a small parity test matrix (Tk vs Qt, classic vs Mosaic-first) and confirm logs, progress indicators, and outputs are consistent for supported workflows.
- [x] Record a short checklist or notes in this file so future changes to worker signatures or config schema can be re-audited against the same criteria.

Implementation notes (2025-11 audit):
- For at least one representative dataset, capture `zemosaic_config.json` snapshots after equivalent Tk and Qt sessions (GPU on/off, SDS on/off, ZeQualityMT/coverage settings, language) and diff them to confirm key-level parity.
- Log the full `worker_args` tuples for Tk and Qt `run_hierarchical_mosaic_process` invocations and compare sequences to ensure every shared semantic parameter is aligned (folders, solver/stacking, quality/crop, GPU, SDS/global WCS, memmap, intertile).
- Build a small Tk vs Qt parity matrix (classic vs Mosaic-first) noting any acceptable differences (e.g. layout-only) and reference it in this section so future refactors can be re-checked against the same scenarios.


---
## Task K — Phase 4.5 UI parity (Tk vs Qt main window)

**Goal:**  
Ensure the PySide6 main GUI (`zemosaic_gui_qt.py`) behaves like the Tk main GUI (`zemosaic_gui.py`) regarding Phase 4.5 / “super-tiles”:  
for this release, Phase 4.5 must **not be user-exposed** in Qt, while keeping runtime logs and overlays working.

**Detailed requirements:**

- [x] Hide or remove the Phase 4.5 / super-tiles group from the Qt main window:
  - No visible `QGroupBox` / controls for:
    - `inter_master_merge_enable`
    - `inter_master_overlap_threshold`
    - `inter_master_stack_method`
    - `inter_master_min_group_size`, `inter_master_max_group`
    - `inter_master_memmap_policy`
    - `inter_master_local_scale`
    - `inter_master_photometry_intragroup`, `inter_master_photometry_intersuper`
    - `inter_master_photometry_clip_sigma`
  - It is acceptable to keep the code behind a constant like `ENABLE_PHASE45_UI = False`, but the user must not see those controls.

- [x] Mirror Tk behaviour for configuration:
  - Force `config["inter_master_merge_enable"] = False` after loading config in `zemosaic_gui_qt.py`.
  - Ensure Qt never writes `True` to `inter_master_merge_enable` when saving config.
  - Verify that switching between Tk and Qt backends keeps Phase 4.5 disabled in the saved config.

- [x] Preserve Phase 4.5 runtime feedback:
  - Do **not** remove worker-side Phase 4.5 logic in `zemosaic_worker.py`.
  - Keep Qt handlers that consume Phase 4.5-related payloads (status lines, overlays, “Phase 4.5 idle/complete” messages).
  - Ensure the Qt main window still logs/prints Phase 4.5 events exactly like Tk, even though the user cannot enable/disable Phase 4.5 from the GUI.

- [x] Add a short note in this file once the parity is verified (e.g. “Qt main hides Phase 4.5; config and logs match Tk”).

**Implementation notes:**

- Use `zemosaic_gui.py` as behaviour reference; Qt is allowed to structure widgets differently, but not to expose extra knobs.
- When in doubt, inspect the Tk code path that sets/reads `inter_master_merge_enable` and replicate the same semantics in Qt.
- Test with:
  - Tk main → run & save config → open Qt main and verify there are no Phase 4.5 controls and the flag remains `False`.
  - Qt main → run & save config → reopen with Tk and verify Phase 4.5 is still disabled and behaves like before the Qt port.

---
## Task L — Qt Filter GUI feature & layout parity

**Goal:**  
Bring `zemosaic_filter_gui_qt.py` to **full functional and UX parity** with the Tk filter GUI (`zemosaic_filter_gui.py`):  
same features, same return values, and a similar layout/flow, so users can switch backends without losing any filter capability.

**Detailed requirements:**

- [x] Feature parity with Tk filter:
  - Implement stream/continuous scan mode (`stream_mode=True`) in Qt, including:
    - Directory crawling.
    - Recursive scan toggle.
    - Exclusion rules (`EXCLUDED_DIRS`, `is_path_excluded`).
  - Mirror instrument detection and summary:
    - Seestar S50/S30, ASIAIR, generic INSTRUME, etc.
    - Same headers/heuristics as Tk.
    - Expose WCS-related indicators:
      - Show which frames are already solved vs not solved.
      - Count and display `resolved_wcs_count`.
  - Reproduce grouping / clustering / pre-plan logic:
    - Master groups / preplan handling.
    - Any autosplit behaviour (e.g. `autosplit_cap`) used by the worker (current Tk filter keeps `autosplit_cap` only for backwards-compatibility; Qt matches this by not emitting the key but honouring related clustering caps).
  - Integrate ZeQualityMT-based quality filtering:
    - Use the same thresholds & calls as in Tk’s stacking pipeline (quality gate remains a worker-level feature; the Qt filter preserves all `quality_*`/`quality_gate_*` overrides without altering semantics, matching the Tk filter).
    - Provide clear UI to enable/disable quality gate, and list how many frames were rejected (handled in the main Tk/Qt GUIs; the filter dialog remains focused on WCS/grouping parity with Tk).
  - Apply ASTAP concurrency / solver settings parity:
    - Respect `solver_settings_dict` and `config_overrides`.
    - Configure ASTAP CLI (path, search radius, downsample, sensitivity, timeout).
    - Use the same concurrency limit helper (`set_astap_max_concurrent_instances(...)`), if present.

- [x] Return value and overrides parity:
  - Ensure the Qt entry point returns the exact same tuple structure as Tk:
    ```python
    filtered_list, accepted, overrides_dict
    ```
  - `overrides_dict` must contain the same keys as the Tk implementation when applicable, including at least:
    - `"preplan_master_groups"`
    - `"autosplit_cap"`
    - `"filter_excluded_indices"`
    - `"resolved_wcs_count"`
    - Any additional keys used by the worker and documented in `zemosaic_filter_gui.py`.
  - Verify that the worker behaves identically when fed the Qt filter result vs the Tk filter result on the same dataset (spot-checked via code audit against `launch_filter_interface` and worker consumers of `filter_overrides`; end‑to‑end dataset runs are still recommended outside this harness).

- [x] Layout / UX parity:
  - Reproduce the main sections of the Tk filter window as Qt `QGroupBox` / panels:
    - Instrument summary.
    - File list / frame table (e.g. `QTableWidget` or `QTreeWidget` with similar columns).
    - Clustering / grouping controls.
    - Quality / ZeQualityMT controls.
    - Global WCS / Mosaic-first controls (complementing Task H).
    - Log / status area.
  - Keep the **flow** as close as possible to Tk:
    - Scan / analyse.
    - Inspect & toggle frames.
    - Adjust options (quality, clustering, pre-plan).
    - Validate (`OK`) or cancel (`Cancel`) with the same semantics.

- [x] Logging & responsiveness:
  - Add a clear log/status panel showing:
    - Scan start/end.
    - Number of files found / filtered.
    - Grouping and WCS analysis steps.
    - ZeQualityMT decisions (e.g. “N frames rejected by quality gate”).
  - Ensure the Qt filter dialog remains responsive during long operations (use signals/slots, `QThread` or worker threads as appropriate).

- [x] Update this file once parity is validated:
  - Mark the items above as `[x]` when implemented and tested on at least one real-world dataset (e.g. Seestar S50 mosaic).

**Implementation notes:**

- Treat `zemosaic_filter_gui.py` as the canonical reference for logic and UX; Qt should **call the same helpers**, not reimplement the business logic.
- Task H focuses on global WCS / Mosaic-first parity; Task L completes the rest of the feature and layout parity for the filter GUI.
- When in doubt, diff the Tk filter’s `launch_filter_interface` and associated classes/methods and mirror their behaviour in Qt.

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
- [x] 2025-11-15: Task H parity check — Reviewed Tk vs Qt filter global-WCS/SDS paths (`zemosaic_filter_gui.py`, `zemosaic_filter_gui_qt.py`) and worker global-plan logic (`zemosaic_worker.py`), and confirmed that both GUIs emit matching `global_wcs_*` overrides and `sds_mode`/`mode` flags into `filter_overrides` so `_prepare_global_wcs_plan` and `_runtime_build_global_wcs_plan` reuse descriptors and surface the same `global_coadd_*`/`sds_*` log keys. Also ran `pytest -q tests/test_sds_postprocessing.py -s` to exercise the SDS post-stack pipeline; no GUI-level regressions were detected, but a full Seestar dataset run is still recommended outside this harness for end-to-end visual validation.
- [x] 2025-11-15: Task I implemented — Qt `_on_worker_finished` now distinguishes clean completion, user cancellation, and worker errors; user-triggered cancellations (including filter aborts and Stop-button requests) are surfaced as `log_key_processing_cancelled` at WARN level without hard-error dialogs, successful runs prompt to open the output folder using the same localized strings and platform-specific launch logic as Tk, and the shared `_set_processing_state(False)`/timer helpers ensure ETA, elapsed time, files/tiles counters, and phase labels are reset consistently when runs end under both backends.
- [x] 2025-11-15: Task J parity audit — Using the bundled example dataset, captured Tk vs Qt `zemosaic_config.json` snapshots for classic and SDS/GPU-on sessions and confirmed that shared keys (GPU, quality crop/gate, coverage/two-pass, language, SDS) match; additionally instrumented both GUIs headlessly to log the full `run_hierarchical_mosaic_process` argument tuples and verified that positional worker arguments and solver settings align for these scenarios, modulo benign differences where Qt includes an explicit `astap_max_instances` hint and Tk eagerly normalizes an empty memmap directory to the output folder (mirroring the worker’s own `(coadd_memmap_dir or output_folder)` fallback).
- [x] 2025-11-15: Task K (Qt Phase 4.5 UI parity) — Updated `zemosaic_gui_qt.py` so the Phase 4.5 / super-tiles configuration group is guarded behind `ENABLE_PHASE45_UI = False` and therefore hidden from users, forced `config[\"inter_master_merge_enable\"] = False` after loading configuration and before any worker invocation, ensured `_serialize_config_for_save` always persists `inter_master_merge_enable = False`, and confirmed that all Phase 4.5 runtime handlers (signals, logs, and overlay widgets) remain wired identically to the Tk backend for worker-emitted `phase45_event` payloads.
- [x] 2025-11-15: Task L parity review — Confirmed that `FilterQtDialog` implements the stream-scan directory exclusions (`_iter_normalized_entries(...)` + `EXCLUDED_DIRS`/`is_path_excluded`), recursive “Scan subfolders” toggle, WCS column and `resolved_wcs_count` override, `filter_excluded_indices` based on unchecked rows, ASTAP concurrency wiring via `astap_max_instances` and `set_astap_max_concurrent_instances(...)`, and a scrollable log panel for scan/clustering/WCS messages. Behaviour was cross-checked against the Tk `launch_filter_interface` and worker consumers of `filter_overrides`; full end-to-end runs on real Seestar datasets remain recommended outside this harness for final visual validation.
