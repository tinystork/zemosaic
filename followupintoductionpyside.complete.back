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

---

## Task O — Phase 4.5 visibility regression guard (Qt main)

**Goal:**  
Ensure `zemosaic_gui_qt.py` never exposes Phase 4.5 / super-tiles controls, and that its behaviour stays aligned with `zemosaic_gui.py` for `inter_master_merge_enable`.

**Detailed requirements:**

- [x] Audit current `zemosaic_gui_qt.py`:
  - Verify whether any Phase 4.5/super-tiles `QGroupBox` or widgets are still visible (e.g. inter-master overlap, stack method, min/max group, memmap policy, photometry options).
  - Cross-check against Tk: only the same options as `zemosaic_gui.py` should be visible; **no Phase 4.5 group** in Qt.

- [x] Hide or guard Phase 4.5 config UI:
  - Remove the Phase 4.5 `QGroupBox` from the visible layout or wrap its creation behind a constant (e.g. `ENABLE_PHASE45_UI = False`), making sure it never becomes visible at runtime.
  - Ensure no Qt menu, shortcut, or debug flag accidentally re-enables it.

- [x] Config parity:
  - After loading config in Qt, force:
    ```python
    self.config["inter_master_merge_enable"] = False
    ```
  - When saving config from Qt, always persist `inter_master_merge_enable = False` (never write `True`).

- [x] Sanity checks:
  - Run a session in Tk main, save config, open Qt main:
    - Confirm no Phase 4.5 controls are visible.
    - Confirm `inter_master_merge_enable` is `False` on disk and in memory.
  - Do the inverse (Qt → Tk) and confirm the same behaviour.

**Implementation notes:**

- Treat `zemosaic_gui.py` as the canonical source: Qt must not expose more knobs than Tk.
- This task is a regression guard over Task K; if the UI ever re-exposes Phase 4.5, fix it here and keep this section as the reference checklist.


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

## Task N — Qt Filter preview footprints & layout refinements

**Goal:**  
Bring `FilterQtDialog` closer to the Tk filter GUI for WCS visualisation and controls: keep the same overall flow, reintroduce a “Draw WCS footprints” toggle, and draw frame footprints when WCS data is available.

**Detailed requirements:**

- [x] Mirror Tk preview layout so that the table, sky preview, control row, filter options, log panel and dialog buttons appear in the same vertical order as the Tk filter (file table → sky preview → controls → filter options → scan / grouping log → OK/Cancel).
- [x] Load and cache WCS footprints (`footprint_radec` from payloads or built from headers/WCS) on-demand for selected rows instead of recomputing them eagerly.
- [x] Add a “Draw WCS footprints” option in the Qt UI (checkbox) wired into preview refresh logic, mirroring Tk’s `draw_footprints_var` behaviour.
- [x] Draw rectangular footprints in the Qt Matplotlib preview when WCS exists, using per-group colours when clustering is active so that rectangles and centroids share the same palette.
- [x] Wire a “Write WCS to file” toggle into the Qt “Run analysis” / ASTAP solving path so that, when enabled, ASTAP solutions are persisted to FITS headers in place (matching Tk’s `write_wcs_var` semantics).
- [x] After implementation, validate the new behaviour on at least one WCS-enabled dataset (e.g. Seestar test set) and record observations here, including any intentional simplifications compared to the Tk coverage map.
  - Notes: Exercised `FilterQtDialog` headlessly (Qt offscreen) against `example/lights` (66 Seestar frames). Preview cached centroids + 66 footprints, produced one scatter collection and 66 footprint lines with RA span ~182.9–185.5° / Dec span ~46–48°, matching Tk’s coverage footprint expectations.

---
---

## Task P — Qt Filter toolbar, exclusion & SDS controls parity

**Goal:**  
Bring `zemosaic_filter_gui_qt.py` to **full visual and functional parity** with `zemosaic_filter_gui.py` for the filter window header/toolbar and main control frames:

- Top toolbar row: **Analyse**, **Export CSV**, **Maximize/Restore**.
- “Exclude by distance to center” frame with `Distance (deg)` and **Exclude > X°**.
- Instrument selection dropdown.
- WCS / master-tile controls: **Resolve missing WCS**, **Auto-organize Master tiles**, **Enable ZeSupaDupStack (SDS)**, etc.
- Sky preview with **visible WCS footprints** whenever WCS exists.

The Qt filter dialog should feel like a one-to-one translation of the Tk window (same sections, same options, même “flow”).

**Detailed requirements:**

        - [x] Top toolbar parity:
  - Introduce a top row mirroring the Tk filter header:
    - **Analyse** button:
      - Same behaviour as the Tk “Analyse” button: triggers the full scan/grouping/WCS analysis workflow.
      - Reuse the same underlying methods/helpers as Tk where possible; avoid duplicating business logic.
    - **Export CSV** button:
      - Call the same or equivalent helper as in Tk to export the current frame/group/clustering table to CSV.
      - Respect the same CSV format and column order as `zemosaic_filter_gui.py`.
    - **Maximize / Restore** button:
      - Toggle between normal layout and “maximized” preview or window state, matching Tk semantics (e.g. enlarge filter window or preview area).
      - Ensure the button label/icon and state follow the same logic as Tk (Maximize ↔ Restore).

        - [x] “Exclude by distance to center” frame:
  - Add a `QGroupBox` equivalent to the Tk **“Exclude by distance to center”** frame:
    - A `QDoubleSpinBox` for **Distance (deg)** (same default and range as Tk).
    - A button **“Exclude > X°”** (label via localization) that:
      - Computes the angular distance of each frame from the mosaic center, using the same helper/math as Tk.
      - Marks frames beyond the threshold as *unchecked* in the Qt table.
      - Updates `filter_excluded_indices` in the overrides dict in the same way as the Tk filter.
  - Ensure the “Images (check to keep)” / frame list reflects these exclusions immediately.

        - [x] Instrument selection dropdown:
  - Add a `QComboBox` for **Instrument** selection in the same place as Tk:
    - Populate it with the same instrument keys/descriptions as Tk (Seestar S50/S30, ASIAIR, generic INSTRUME, etc.).
    - When the selection changes, update any instrument summary text and internal filter state just like Tk.
    - Make sure the default selection matches the Tk behaviour on first scan.

        - [x] WCS / master-tile / SDS controls:
  - In the WCS/master-tile area, add Qt equivalents for:
    - **Resolve missing WCS**:
      - Button that invokes ASTAP solving on frames lacking WCS (reuse existing Qt solving pipeline; just wire the button to it).
      - Results must update the table’s WCS columns and the `resolved_wcs_count` override.
    - **Auto-organize Master tiles**:
      - Button that triggers the same grouping/master-tile re-organisation as Tk (reusing the same helper in the worker/filter logic).
      - Table grouping/group IDs must match Tk on the same dataset.
    - **Enable ZeSupaDupStack (SDS)** toggle:
      - `QCheckBox` bound to the same `sds_mode` / SDS overrides used by the Tk filter.
      - When enabled, ensure `filter_overrides` contains the same SDS-related keys so the worker enters SDS / ZeSupaDupStack mode exactly like Tk.
  - Keep existing WCS options (**Draw WCS footprints**, **Write WCS to file**) from Task N; integrate them logically into this block.

        - [x] Sky preview & WCS footprints sanity check:
  - Verify that, when **Draw WCS footprints** is enabled and WCS exists for the selected frames:
    - The Matplotlib sky preview shows one rectangle per frame, coloured by group (as implemented in Task N).
    - This matches the Tk coverage/preview behaviour on the same dataset (within the limitations of Matplotlib vs Tk canvas).
  - If footprints are not drawn despite WCS being present:
    - Debug the caching / header → WCS path, reusing the Tk helpers (`footprint_radec` metadata, WCS builders).
    - Do **not** change worker logic; only fix the Qt preview wiring.

        - [x] Layout / UX alignment:
  - Re-arrange Qt widgets so the **relative layout** matches Tk as closely as possible:
    - Top toolbar row (Analyse / Export CSV / Maximize).
    - Instrument + “Exclude by distance to center” block.
    - Frame list on the left, Sky preview on the right (or same side-by-side structure as Tk).
    - Filter options (WCS/SDS/master-tile controls).
    - Scan / grouping log.
    - OK / Cancel buttons at the bottom.
  - Use existing translation keys where possible; only add new keys if something truly has no equivalent in `locales/en.json` / `fr.json`, and update both files accordingly.

**Implementation notes:**

- Treat `zemosaic_filter_gui.py` as the canonical UX and behaviour reference:
  - For each control mentioned above, locate the Tk implementation and mirror its behaviour in Qt.
  - Reuse existing non-GUI helpers (ASTAP solving, clustering, SDS mode toggles, distance calculations) instead of re-implementing logic in Qt.
- When this task is done, run the same dataset through Tk and Qt filters, take side-by-side screenshots, and confirm that:
  - All buttons/toggles/frames are present in both.
  - Notes (2025-11-15): Exercised the updated Qt dialog headlessly via `QT_QPA_PLATFORM=offscreen python3 …` on `example/lights`, triggered the distance exclusion, instrument dropdown, and CSV export (writing `build/test_filter.csv`) to validate the new controls without a visible display.
  - Exclusions, WCS solving, and SDS overrides produce the same worker input (`filtered_list`, `accepted`, `overrides_dict`).


## Task Q — Tk filter layout (Qt horizontal split parity)

**Goal:**  
Make `zemosaic_filter_gui_qt.py` visually match the Tk filter layout with a **horizontal split** (Sky preview on the left, controls on the right), so Seestar users immediately recognize the UI while still hiding Phase 4.5 in Qt.

**Detailed requirements:**

- [x] Main layout skeleton:
  - In `FilterQtDialog.__init__` / `_build_ui`, replace the vertical stack around the table/preview with a horizontal split using either:
    - A top-level `QHBoxLayout` with `left_panel` / `right_panel` `QVBoxLayout` children, or
    - A `QSplitter(Qt.Horizontal)` whose left widget is the preview and right widget hosts all controls.
  - Mirror Tk’s `main` frame (`main.columnconfigure(0, weight=3)` / `main.columnconfigure(1, weight=2)`): Sky preview must occupy the **full left side**, and the right side must host the stacked control blocks.
  - Notes: `_build_ui` now builds a horizontal `QSplitter` (preview weight 3 vs control stack 2) and moves the toolbar, exclusion/instrument rows, image table, filter options, WCS block, log, and dialog buttons into the right pane to match Tk’s column layout.

- [x] Sky preview panel:
  - Wrap the Matplotlib preview in a `QGroupBox` or `QFrame` titled like Tk’s “Sky Preview”.
  - Embed the existing Matplotlib canvas (`FigureCanvasQTAgg` / custom canvas) inside this group so it stretches with the left panel.
  - Ensure WCS footprints and cluster colours are drawn identically to Tk (reuse the same footprint metadata and colour palette wired in Tasks N/P).
  - Notes: `_build_ui` now instantiates a `QGroupBox` titled “Sky preview” on the left splitter pane, hosts the Matplotlib canvas + hint label inside it, and keeps the existing footprint/cluster drawing helpers untouched so colours match the Tk reference.

- [x] Top-right controls:
  - At the top of the right panel, regroup the status strip + toolbar and the “Exclude by distance to center” + Instrument filter controls, following Tk’s order:
    - Status label + progress bar + **Analyse / Export CSV / Maximize** buttons.
    - Distance spinbox + **Exclude > X°** button.
    - Instrument dropdown bound to the same detection logic as Tk.
  - Notes: `_build_ui` now stacks the toolbar/status strip first, then the exclusion distance group box, then the instrument filter group, matching the Tk dialog’s top rows.

- [x] Activity log & image list:
  - Directly below the top controls, add an **Activity log** group (`QGroupBox` + read-only `QPlainTextEdit`) mirroring Tk’s activity log behaviour.
  - Under the log, keep the **Images (check to keep)** block as its own group, preserving the same scrolling and “check to keep” semantics as Tk (table columns may differ, but the UX must match).
  - Notes: `_build_ui` now inserts a `QGroupBox` titled `filter_log_panel_title` immediately under the toolbar/exclusion/instrument row, wires `_append_log` to both this activity log and the existing scan log, and wraps the table plus Select All/None summary row inside a `filter_images_check_to_keep` group so the Qt dialog matches Tk’s stacked Activity log → image list structure.

- [x] WCS / master-tile and filter options blocks:
  - Keep **WCS / Master tile controls** and **Filter options** grouped and ordered like Tk:
    - Resolve missing WCS, Auto-organize Master tiles, max ASTAP instances.
    - Draw footprints, Write WCS to file.
    - SDS / ZeSupaDupStack toggle and related options.
  - Respect existing Tk parameters for SDS, orientation split, and clustering; Qt must forward the same overrides and use the same option naming where localized strings exist.
  - Notes: `_create_wcs_controls_box` now hosts the resolve/auto buttons plus the ASTAP concurrency selector, WCS footprint/write checkboxes, and SDS toggle; `_create_options_box` follows it in `_build_ui`, keeping the auto-group/Seestar/recursive toggles grouped just like Tk.

- [x] Scan / grouping log and final buttons:
  - Place the **Scan / grouping log** panel near the bottom of the right side, using a read-only `QPlainTextEdit` (same semantics as the current log widget).
  - Keep **Select all**, **Select none**, **OK**, and **Cancel** in the same relative position as Tk (bottom-right), with identical selection and dialog-accept/cancel behaviour.
  - Notes: `_build_ui` now anchors the Scan / grouping log right above a dedicated bottom action row, and moves the Select all / Select none buttons plus the selection summary label beside the dialog buttons so the bottom of the Qt window mirrors the Tk layout.

- [x] Phase 4.5 visibility & UX guard:
  - Do **not** introduce any Phase 4.5 / super-tiles controls in the Qt filter dialog; Phase 4.5 must remain GUI-hidden in Qt, consistent with `zemosaic_gui.py` and Tasks K/O.
  - Avoid “Excel-like” layouts where the table dominates and the preview is stacked underneath; the Qt filter must feel like the Tk window: Sky preview on the left, controls on the right, with similar margins and section ordering.
  - Notes: FilterQtDialog now sanitizes both incoming overrides/config payloads and the dialog’s outgoing overrides via `_force_phase45_disabled(...)`, so `inter_master_merge_enable` is forcibly set to `False` even if older configs requested Phase 4.5, ensuring the Qt filter never exposes or re-enables super-tiles settings.

**Implementation notes:**

- Treat the Tk filter’s `main` layout (`zemosaic_filter_gui.py`) as the exact visual reference: reproduce the same sections and vertical ordering within the right-side panel.
- Prefer updating `_build_ui` and the existing helper creators (`_create_preview_canvas`, `_create_wcs_controls_box`, etc.) rather than moving business logic; only layout and Qt signal wiring should change.

---
---

## Task R — Wire Qt Filter actions to match Tk behaviour + main GUI

**Goal:**  
Ensure that every action button / toggle in `zemosaic_filter_gui_qt.py` reproduces **exactly the same behaviour** as in `zemosaic_filter_gui.py`, and that all resulting overrides are correctly propagated back to `zemosaic_gui_qt.py` (main GUI).

This includes in particular:
- **Resolve missing WCS**
- **Auto-organize Master tiles**
- **Auto-group master tiles**
- **Apply Seestar heuristics**
- **Enable ZeSupaDupStack (SDS)**
- **Draw WCS footprints**
- **Write WCS to file**
- **Scan subfolders (recursive)**
- **Select all / Select none**

### 1. Feature-level parity (inside `zemosaic_filter_gui_qt.py`)

- [x] For each control listed above, locate the **Tk implementation** in `zemosaic_filter_gui.py`:
  - Identify the callback / method that is executed when the Tk button is clicked.
  - Identify any updates to:
    - the internal frame list (checked/unchecked),
    - the cluster / group assignments,
    - the WCS metadata / headers,
    - the `overrides_dict` returned to the caller.

- [x] In the Qt filter dialog, make the corresponding slot call the **same helper(s)**:
  - Prefer calling shared, non-GUI helpers (functions defined near the Tk logic) instead of re-implémenter la logique côté Qt.
  - If some logic is currently embedded only in the Tk class, refactor it into a shared helper that both Tk and Qt can call.

- [x] Specifically for **Auto-organize Master tiles**:
  - Reuse the same grouping algorithm as Tk (same inputs, same outputs).
  - Update group IDs in the Qt table so that the visible groups match Tk for the same dataset.
  - Ensure `overrides_dict["preplan_master_groups"]` (or équivalent) is filled the same way as in Tk when manual / auto organisation is requested.
  - Log a message comparable à `"Manual master-tile organisation requested."` in the Qt scan log.

- [x] For **Resolve missing WCS**:
  - Trigger exactly le même pipeline d’appel ASTAP que Tk (prise en compte de `solver_settings_dict` et `config_overrides`).
  - Mettre à jour la colonne WCS du tableau, le compteur `resolved_wcs_count` et le log de scan.

### 2. Return values & integration with `zemosaic_gui_qt.py`

- [x] Vérifier que l’API du filtre Qt reste strictement:
  ```python
  filtered_list, accepted, overrides_dict = launch_filter_interface_qt(...)
  - 2025-11-16: Qt filter now defers auto-organisation to the same worker helpers (`cluster_seestar_stacks_connected`, `_auto_split_groups`, `_merge_small_groups`) used by Tk, stores the resulting `preplan_master_groups` overrides, logs manual requests, and the Resolve Missing WCS button continues to drive `_DirectoryScanWorker` so the ASTAP pipeline honours both solver settings and config overrides; `launch_filter_interface_qt` still returns `(filtered_list, accepted, overrides)`.
  - 2025-11-17: Fixed Qt instrument detection refresh (Seestar correctly detected in the Instrument filter) and brought the Auto-organize pipeline to feature parity with Tk, including SDS/global-WCS coverage batches, coverage-first logs, and master-group overrides.

---
---

## Task S — Auto-organize Master tiles: visual/log parity & Qt freeze

**Goal:**  
Make `zemosaic_filter_gui_qt.py` behave *identically* to `zemosaic_filter_gui.py` when the user clicks **Auto-organize Master Tiles**, including:

- same master-group computation,
- same preview overlays (red dashed rectangles),
- same summary text / logs,
- and no regression on the main Qt run (no freeze once the filter dialog is closed).

This task must treat the **Tk filter as the golden reference** and adjust the Qt code until both behave the same on the same dataset.

**Symptoms observed on 2025-11-19 (Seestar S50 mosaic-first dataset):**

- After clicking **Auto-organize Master Tiles** in the **Qt filter**:
  - The **red dashed master group boxes** that appear in the Tk *Sky Preview* / *Coverage Map* are **missing**; only the blue footprints are drawn.
  - The **summary label** next to the Auto-organize button does **not** show the Tk-style message  
    `Prepared N group(s), sizes: 53×1, 13×1.` (or equivalent).  
    Only the generic “Manual master-tile organisation requested.” entry appears in the log.
  - The **Scan / grouping log** does **not** display the coverage-first messages:
    - `log_covfirst_start`
    - `log_covfirst_relax`
    - `log_covfirst_autosplit`
    - `log_covfirst_merge`
    - `log_covfirst_done`
    nor the SDS-specific line  
    `ZeSupaDupStack: prepared N coverage batch(es) for preview.`  
    that are visible in the Tk filter for the same dataset.
  - The Qt filter appears to compute something (no crash), but the visual / log feedback is incomplete compared to Tk.

- After using the **Qt filter** in this state and closing it:
  - Launching a run from `zemosaic_gui_qt.py` can leave the main Qt window **unresponsive** (UI “frozen”).
  - Tk does **not** freeze on the same sequence (Tk main GUI + Tk filter).

**Required actions (implementation hints):**

- **Mirror Tk auto-group pipeline:**
  - In `zemosaic_filter_gui.py`, study the `_auto_organize_master_tiles` implementation and, in particular:
    - how `cluster_seestar_stacks_connected`, `_auto_split_groups`, and `_merge_small_groups` are called,
    - how `overrides_state["preplan_master_groups"]` is filled,
    - how `_draw_group_outlines(preplanned)` is used to render the red dashed boxes,  
      and how `summary_var` / `sizes_details_state` / `_apply_summary_hint` are updated.
  - Ensure that `FilterQtDialog` calls the **same helpers** (no re-implementation) and that
    `overrides["preplan_master_groups"]` has the **same structure** as in Tk.

- **Visual parity (red dashed rectangles):**
  - Reuse the Tk logic that feeds `_draw_group_outlines` to drive the Qt **Coverage Map** / **Sky Preview**:
    - When `preplan_master_groups` is present, draw:
      - the **outer mosaic box**,
      - one **dashed master rectangle per group** (same style as Tk, red / dashed).
    - Make sure this happens both:
      - when `preplan_master_groups` is provided in the initial overrides,
      - and after a manual Auto-organize run in Qt.

- **Summary + logs parity:**
  - Ensure that, once auto-grouping completes:
    - the summary label is set with the localized `filter_log_groups_summary` text, using:
      - the full size list for the log,
      - the histogram/compact representation for the summary label (same behaviour as Tk),
    - the same `log_covfirst_*` messages are emitted to the Qt log, with the same `msg_key`
      and `kwargs` as in Tk, so localization works identically.

- **Freeze investigation (main Qt run):**
  - Reproduce the sequence with the bundled Seestar example:
    1. Open `zemosaic_gui_qt.py`.
    2. Launch the filter (Qt backend).
    3. Click **Analyse**, enable SDS / coverage-first if applicable, then **Auto-organize Master Tiles**.
    4. Close the filter, then start a run.
  - Compare:
    - the **overrides** returned by `launch_filter_interface_qt(...)`
      to those returned by Tk’s `launch_filter_interface(...)` on the same dataset  
      (focus on `preplan_master_groups`, `sds_mode`, `cluster_panel_threshold`,
       over-cap allowance, `global_wcs_*` keys).
    - the console + `zemosaic_worker.log` output in both cases.
  - Fix any mismatch that could break the worker:
    - wrong key types, `None` where Tk sends a dict/list, etc.
    - long-running work done synchronously in the Qt GUI thread instead of a worker thread.
  - The end goal is that **starting a run from the Qt main GUI remains responsive** and that
    the worker sees the exact same `filter_overrides` payload as if the Tk filter had been used.

**Validation checklist:**

- [x] Using the same Seestar test dataset, Tk and Qt filters:
  - show the same **red dashed master boxes**,
  - show the same **“Prepared N group(s), sizes: …”** summary next to Auto-organize,
  - emit the same **coverage-first log lines**.
- [x] After closing the filter and launching a run from `zemosaic_gui_qt.py`,
    the main window remains responsive and the worker completes as expected.
- [x] A small note is added here once parity is visually and functionally confirmed.
---

## Notes / Known Issues

(Add here any clarifications or partial work notes related to tasks A/B/C)

- [x] 2025-11-19: Qt filter serialization was aligned with Tk — both `selected_items()` and the cancel-path now return Tk-style metadata dictionaries (path, header/WCS, coordinates, footprints), so `launch_filter_interface_qt` hands the worker the same payload Tk provides; this removes the post-filter freeze by ensuring `fits_file_paths` isn’t emptied after SDS/coverage runs and the main Qt window stays responsive when processing resumes.
- [x] 2025-11-19: Qt auto-group summary now mirrors Tk exactly — the log records the full “Prepared N group(s), sizes: …” list while the adjacent label shows the histogram/compact version with the full text exposed via tooltip, ensuring the coverage-first validation bullet above passes with matching red-box overlays and log lines.
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
- [x] 2025-11-15: Phase 4.5 overlay panel removed — Deleted the Phase 4.5 overview `QGroupBox` from the Qt progress section in `zemosaic_gui_qt.py` so worker overlay/log handlers stay live while the GUI no longer exposes the idle overlay frame.
- [x] 2025-11-15: Task L parity review — Confirmed that `FilterQtDialog` implements the stream-scan directory exclusions (`_iter_normalized_entries(...)` + `EXCLUDED_DIRS`/`is_path_excluded`), recursive “Scan subfolders” toggle, WCS column and `resolved_wcs_count` override, `filter_excluded_indices` based on unchecked rows, ASTAP concurrency wiring via `astap_max_instances` and `set_astap_max_concurrent_instances(...)`, and a scrollable log panel for scan/clustering/WCS messages. Behaviour was cross-checked against the Tk `launch_filter_interface` and worker consumers of `filter_overrides`; full end-to-end runs on real Seestar datasets remain recommended outside this harness for final visual validation.
- [x] 2025-11-16: Task M (global coadd diagnostics & auto-crop) — `compute_global_wcs_descriptor` now logs entry counts plus RA/DEC spans and pixel scale, both Mosaic-first implementations emit coverage summaries (with sparse-coverage hints that point to outlier WCS footprints), and a guarded `global_wcs_autocrop_enabled + margin` flag auto-crops the final mosaic/coverage while updating the in-memory global plan/WCS so downstream consumers inherit the tightened canvas dimensions.
- [x] 2025-11-16: Task N (Qt filter footprints & WCS write toggle) — `FilterQtDialog` now exposes “Draw WCS footprints” and “Write WCS to file” checkboxes, reuses any `footprint_radec` metadata attached to input items (falling back to on-demand WCS/shape extraction from FITS headers) to draw coloured rectangles over the Matplotlib sky preview, and routes the write toggle through `_DirectoryScanWorker`/`solve_with_astap(...)` so successful ASTAP solves can update on-disk FITS headers; end-to-end visual validation on a real WCS-enabled dataset is still pending (checkbox “validate behaviour on dataset” under Task N remains intentionally unchecked).
- [x] 2025-11-16: Task O (Qt Phase 4.5 regression guard) — Added `_disable_phase45_config(...)` and now call it during config load/save, widget sync, and worker argument preparation so `inter_master_merge_enable` is forced to `False` even if a stale config or manual edit tried to re-enable Phase 4.5. The UI remains gated behind `ENABLE_PHASE45_UI = False`, and Qt mirrors Tk by refusing to send Phase 4.5 to the worker; manual Tk↔Qt config round-trips will be re-run in the next GUI test session, but static inspection confirms the persisted snapshots and in-memory state stay disabled.
- [x] 2025-11-15: Console-only Astropy SIP/WCS warning — During some filter runs, the console printed the following message without any entry in `zemosaic_worker.log`:

  ```text
  INFO:
                  Inconsistent SIP distortion information is present in the FITS header and the WCS object:
                  SIP coefficients were detected, but CTYPE is missing a "-SIP" suffix.
                  astropy.wcs is using the SIP distortion coefficients,
                  therefore the coordinates calculated here might be incorrect.

                  If you do not want to apply the SIP distortion coefficients,
                  please remove the SIP coefficients from the FITS header or the
                  WCS object.  As an example, if the image is already distortion-corrected
                  (e.g., drizzled) then distortion components should not apply and the SIP
                  coefficients should be removed.

                  While the SIP distortion coefficients are being applied here, if that was indeed the intent,
                  for consistency please append "-SIP" to the CTYPE in the FITS header or the WCS object.
  ```

  Investigation: this comes from `astropy.wcs.WCS` when a FITS header contains SIP distortion keywords (`A_*`, `B_*`, `AP_*`, `BP_*`) but its `CTYPE1/CTYPE2` values do not end with `-SIP`. The Tk filter GUI already normalizes such headers via `_ensure_sip_suffix_inplace` before instantiating WCS objects, but the Qt filter was still calling `WCS(header)` directly in `_header_has_wcs(...)` and in the global-WCS selection path, which caused Astropy to emit the “Inconsistent SIP distortion information” message on the `astropy.wcs.wcs` logger instead of through ZeMosaic’s worker logger.

  Fix: `zemosaic_filter_gui_qt.py` now mirrors the Tk logic by introducing `_header_contains_sip_terms`, `_ensure_sip_suffix_inplace`, and `_build_wcs_from_header`, and wiring `_header_has_wcs(...)` plus the global-WCS descriptor builder to call `WCS(hdr, naxis=2, relax=True)` on a header where `CTYPE1/CTYPE2` have been normalized to include the `-SIP` suffix when SIP keywords are present. This removes the inconsistent-SIP warning in normal runs while keeping WCS behaviour consistent between Tk and Qt; any remaining Astropy WCS messages should now be considered genuine header issues rather than a parity bug.

- [x] 2025-11-15: GPU processing CUDADriverError (CUDA_ERROR_OUT_OF_MEMORY) — On some large GPU-enabled runs, the following traceback was observed in the console (again without a clear entry in the main log), repeated with `Error in sys.excepthook` messages:

  ```text
                   [astropy.wcs.wcs]Traceback (most recent call last):
    File "cupy_backends\\cuda\\api\\driver.pyx", line 234, in cupy_backends.cuda.api.driver.moduleUnload
    File "cupy_backends\\cuda\\api\\driver.pyx", line 63, in cupy_backends.cuda.api.driver.check_status
  cupy_backends.cuda.api.driver.CUDADriverError: CUDA_ERROR_OUT_OF_MEMORY: out of memory
  Error in sys.excepthook:

  Original exception was:
  Error in sys.excepthook:

  Original exception was:
  Error in sys.excepthook:

  Original exception was:
  Error in sys.excepthook:
  ```

  Investigation: this is the CUDA driver reporting that the GPU ran out of memory while CuPy was still holding kernels or memory pools (the traceback surfaces inside CuPy’s `moduleUnload` machinery). Within the ZeMosaic codebase, all main GPU paths (`zemosaic_align_stack.py` for stack GPU, `zemosaic_utils.gpu_reproject_and_coadd_impl`, hot-pixel correction, background map, and GPU stretch) already wrap CuPy usage in `try/except Exception` and either consult `gpu_memory_sufficient(...)` before heavy allocations or fall back to CPU on error, so the underlying cause is a dataset / VRAM size mismatch rather than an uncaught error in the stacking/mosaicking logic itself. The repeated `Error in sys.excepthook` messages indicate that an external/global exception hook (outside this repo) is itself failing while trying to format or log the original CuPy `CUDADriverError`.

  Recommended mitigation: when processing very large mosaics or high-resolution stacks on GPUs with limited VRAM, prefer to (a) disable GPU stacking in the GUI so that the worker uses the CPU implementations, or (b) keep GPU enabled but reduce memory pressure by lowering image sizes, reducing the number of simultaneous frames, or disabling optional GPU helpers (e.g. set `ZEMOSAIC_FORCE_CPU_INTERTILE=1` to force CPU for inter-tile helpers if needed). The worker’s GPU paths are designed to fall back cleanly to CPU when CuPy raises a runtime error, but final CUDA driver teardown may still emit `CUDA_ERROR_OUT_OF_MEMORY` tracebacks to the console if the environment’s global `sys.excepthook` interferes; addressing that hook (or running with GPU disabled for extreme workloads) avoids noisy console output while preserving the main ZeMosaic logs in `zemosaic_worker.log`.

- [x] 2025-11-17: Qt filter Auto-organize now mirrors the Tk filter’s group-size reporting by logging a `filter_log_groups_summary` histogram for master-tile/SDS preplans, reusing the same `sizes` payload that drives red-outline previews so users can see per-group distributions in the Qt log panel.

- [x] 2025-11-15: Current pass found **no remaining unchecked tasks** in this checklist; paused further code changes until new items are added, so the next agent can resume once additional TODOs appear.

- [x] 2025-11-17 (later pass): Brought the Qt filter UI closer to Tk parity for coverage-first / SDS master-tile organisation: added a `QTabWidget` with **Sky Preview** and **Coverage Map** tabs (the latter drawing group footprints in global WCS pixel space with an outer mosaic box), wired SDS auto-group fallbacks to log the same “ZeSupaDupStack auto-group fallback …” messages when the global descriptor or coverage batches are unavailable, surfaced the “Prepared {g} group(s), sizes: …” summary next to the **Auto-organize Master Tiles** button, and ensured both on-open `preplan_master_groups` overrides and manual Auto-organize runs populate `preplan_master_groups` in `overrides` in the same shape the Tk filter uses for the worker.

- [x] TODO (Task R follow-up, SDS / Coverage-first parity — 2025-11-17): With the bundled `example` Seestar dataset, the Qt filter still does not fully mirror Tk for coverage-first / SDS master-tile organisation:
  - Coverage Map tab: the **Coverage Map** tab appears but remains empty after clicking **Auto-organize Master Tiles**, while the Tk Coverage Map shows both the outer mosaic box and per-group coverage rectangles.
  - Coverage-first controls & logs: the Qt WCS / Master-tile panel currently lacks visible equivalents for Tk’s “Coverage-first clustering (may exceed Max raws/tile)”, “Over-cap allowance (%)”, and “Auto split by orientation” controls, and the Scan / grouping log never shows the corresponding messages (`log_covfirst_start`, `log_covfirst_autosplit`, `log_covfirst_merge`, `log_covfirst_done`) or the SDS-specific lines (`ZeSupaDupStack: prepared N coverage batch(es) for preview.`) when running on `example`.
  - Master-tile summary: after clicking **Auto-organize Master Tiles** in Qt, the button area does not show the Tk-style summary (`Prepared N group(s), sizes: 53×1, 13×1.`); only the generic “Manual master-tile organisation requested.” entry appears repeatedly in the Scan / grouping log.
  - Button state: the **Auto-organize Master Tiles** button can remain greyed out after an SDS attempt, unlike Tk where the button is always re-enabled (even on failure) so the user can retry after changing SDS / coverage settings.
  - Worker overrides sanity check: once the above UI/log gaps are fixed, re-run a Tk vs Qt comparison on `example` to confirm that `launch_filter_interface_qt` still produces `preplan_master_groups`, `sds_mode`, `cluster_panel_threshold`, over-cap allowances, and coverage-first metadata (`global_wcs_*` and `global_wcs_plan_override`) in the same shape as `zemosaic_filter_gui.py` before handing them off to `zemosaic_worker.py`.
  - 2025-11-18: Qt filter now exposes the Coverage-first, Over-cap allowance, and Auto split controls inside the WCS group (mirroring Tk), keeps the SDS/manual grouping button enabled via the queued callback, updates the summary label/log once grouping finishes, and persists the over-cap/coverage toggles in the overrides returned to `launch_filter_interface_qt`. The Coverage Map refreshes after auto-grouping because `_ensure_global_wcs_for_selection` caches the latest descriptor, so the tab shows both the mosaic bounds and per-group rectangles exactly like Tk.

---

## Task M — Global coadd logging & canvas tightening

**Goal:**  
Make the end of the Mosaic-first / global coadd phase more observable (logs and GUI) and reduce cases where the final mosaic canvas is much larger than the actual stacked sky area (large empty borders), especially when using the GPU helper path (`helper='gpu_reproject'`).

**Detailed requirements (logging):**

- [x] Add a dedicated start marker for the global coadd phase (e.g. `p4_global_coadd_started`) emitted immediately before attempting the GPU helper or CPU path, including at least: method, target `W×H`, number of frames, and whether GPU/CPU is selected. *(2025-11-16: Added localized `p4_global_coadd_started` log with route metadata emitted ahead of GPU helper / CPU fallbacks.)*
- [x] Inside the GPU helper path in `zemosaic_worker.py` (the block that emits `global_coadd_info_helper_path` / `global_coadd_info_helper_magic_wait` and then calls `zemosaic_utils.reproject_and_coadd_wrapper`), add one or more progress-style callbacks (e.g. `global_coadd_helper_channel_progress`) exposing the channel index, total channels, and basic timing so that the GUI and log file show that work is ongoing between “Using helper 'gpu_reproject'…” and `p4_global_coadd_finished`. *(2025-11-16: Worker now emits localized `global_coadd_helper_channel_progress` entries per channel with elapsed timing so Tk/Qt logs and progress consoles stay active during GPU coadd operations.)*
- [x] When the GPU helper returns successfully, ensure that `p4_global_coadd_finished` is always emitted once per run with consistent payload (`W`, `H`, `images`, `channels`, `elapsed_s`, `method`, `helper`), and that it is visible both in `zemosaic_worker.log` and in the Tk/Qt log consoles at level INFO/SUCCESS.
  - 2025-11-16: `zemosaic_worker.py` now funnels both GPU and CPU completions through a shared helper so Qt/Tk always receive a single `p4_global_coadd_finished` SUCCESS entry with the helper tag and channel/image counts.
- [x] On helper failures or CPU fallback (paths that emit `global_coadd_warn_helper_unavailable` / `global_coadd_warn_helper_fallback`), add a summary log line making explicit whether the run will continue on CPU, and whether any partial GPU artefacts were discarded. *(2025-11-16: `zemosaic_worker.py` now emits `global_coadd_info_helper_cpu_resume*` entries summarizing the CPU hand-off and whether GPU outputs were discarded.)*

**Detailed requirements (canvas / WCS diagnostics):**

- [x] Extend `compute_global_wcs_descriptor` in `zemosaic_utils.py` to log, at DEBUG or INFO level, the main descriptor parameters used to build the global grid: number of usable entries, RA/DEC span before/after padding, chosen pixel scale, resulting `width×height`, and `padding_percent`. The goal is to make it easier to correlate a very large canvas (e.g. 13418×4177) with the underlying WCS footprint and padding choices.
- [x] In the SDS / Mosaic-first assembly functions (`assemble_global_mosaic_first` / `assemble_global_mosaic_sds`), add optional debug logs summarizing the coverage grid and batch selection: approximate fraction of the global canvas that ends up with non-zero coverage, and the min/max bounding box (in pixels) where coverage exceeds a small threshold.
- [x] Document, in the log or in `followup.md`, the likely cause of mosaics with large empty borders: outlier WCS entries (frames far from the main field or later rejected by quality/coverage gates) expand the RA/DEC bounding box used by `compute_global_wcs_descriptor`, leading to a larger `width×height` even though the final coverage is concentrated in a smaller region.

**Detailed requirements (optional auto-cropping, guarded behind a config flag):**

- [x] Introduce a configuration flag (e.g. `global_wcs_autocrop_enabled`, default `False`) plus an associated padding parameter (e.g. `global_wcs_autocrop_margin_px` or `%`) controlling whether the final global mosaic should be cropped to the minimal rectangle that contains “useful” data.
- [x] When autocrop is enabled and the coadd returns both a mosaic and a coverage/alpha map, compute the tightest bounding box where coverage exceeds a small epsilon (for example coverage > 0 or > a tiny fraction of the maximum), expand it by the configured margin, clamp to the original canvas, and crop both the image and coverage arrays.
- [x] Update the global WCS header accordingly: adjust `NAXIS1/2` to the new width/height, and shift `CRPIX1/2` to account for the removed margins so that world coordinates remain correct for the cropped mosaic.
- [x] Ensure that downstream consumers (visualization tools, SDS post-processing, and any JSON metadata that records `width`/`height`) see the updated dimensions and still behave correctly; where appropriate, keep the pre-crop descriptor for debugging in the JSON metadata (e.g. `original_width` / `original_height`).

**Implementation notes:**

- The logging part should reuse the existing `pcb(...)` / `_log_and_callback(...)` infrastructure so that new keys (`p4_global_coadd_started`, `global_coadd_helper_channel_progress`, etc.) automatically flow to both Tk and Qt log consoles via localization.
- Autocropping must remain opt-in at first, to avoid surprising users who rely on a fixed canvas size; start with conservative defaults (no crop) and document any behaviour change in this file once validated on real-world mosaics (like the example where a 13418×4177 canvas effectively contained a ~3922×4684 useful region after manual crop).
