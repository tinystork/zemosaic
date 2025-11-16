# AGENT MISSION FILE — ZEMOSAIC QT BACKEND PARITY

You are an autonomous coding agent working on the **ZeMosaic** project.

The repository already contains (non-exhaustive):
- `run_zemosaic.py`
- `zemosaic_gui.py`              → legacy Tk main GUI (reference)
- `zemosaic_gui_qt.py`           → new PySide6 main GUI
- `zemosaic_filter_gui.py`       → legacy Tk filter GUI (reference)
- `zemosaic_filter_gui_qt.py`    → new PySide6 filter GUI
- `zemosaic_worker.py`
- `zemosaic_localization.py`
- `zemosaic_config.py`
- `zemosaic_utils.py`
- `lecropper.py`
- `zequalityMT.py`
- `tk_safe.py`
- `locales/en.json`, `locales/fr.json`
- various helper modules (astrometry, cleaner, etc.)

The Tkinter GUI and the worker logic are **stable and trusted**.  
The PySide6 backend already exists and works, but needs **polish and parity**.

Your mission is now in the **PARITY & CLEANUP phase**, not “big new refactor”.


## GLOBAL GOAL

**Goal:**  
Bring the PySide6 (Qt) GUI to functional parity with the Tk GUI, while keeping
the astrophotography core untouched and preserving all existing workflows.

More specifically, you must:

1. **Main window parity for Phase 4.5 / Super-tiles**
   - `zemosaic_gui_qt.py` must mirror the behavior of `zemosaic_gui.py` regarding
     **Phase 4.5 (inter-master merge / super-tiles)**:
     - The **Phase 4.5 configuration must not be user-exposed** in the Qt GUI
       for this release.
     - The internal config flag `inter_master_merge_enable` must be **forced to `False`**
       just like in `zemosaic_gui.py`, so the worker never receives an enabled state.
     - Status messages, logs and overlays related to Phase 4.5 (progress,
       “Phase 4.5 idle/complete”, overlays, etc.) must continue to work when the
       worker emits them, just as in the Tk GUI.
     - Do **not** remove the worker-side Phase 4.5 implementation in
       `zemosaic_worker.py`. Only ensure that the Qt GUI exposes the **same knobs
       as the Tk GUI** (i.e. **no visible Phase 4.5 options for now**).

2. **Filter GUI parity**
   - `zemosaic_filter_gui_qt.py` must be upgraded so that it implements **all
     functional features** of the Tk filter GUI (`zemosaic_filter_gui.py`),
     including:
     - Stream-scan mode (directory crawling with `stream_scan=True`).
     - Recursive scan toggle.
     - Exclusion logic with `EXCLUDED_DIRS` / `is_path_excluded`.
     - Instrument detection display (Seestar S50/S30, ASIAIR, generic INSTRUME, etc.).
     - WCS presence / absence indicators.
     - Clustering, grouping and pre-plan logic (master groups / preplan).
     - ZeQualityMT-based quality filtering, when available.
     - ASTAP concurrency cap handling.
     - Handling of `solver_settings_dict` and `config_overrides` exactly like the Tk version.
     - Any extra metadata or overrides returned by `launch_filter_interface`
       (e.g. `preplan_master_groups`, `autosplit_cap`, `filter_excluded_indices`,
       `resolved_wcs_count`, …).
   - The **layout and UX** should be as close as possible to the Tk GUI:
     - Sections (frames / group boxes) should mirror the Tk `LabelFrame`
       structure (instrument summary, file list, clustering, quality gate, etc.).
     - Controls that exist in Tk (checkbuttons, spinboxes, labels, buttons)
       must have Qt equivalents (QCheckBox, QSpinBox/QDoubleSpinBox, QComboBox,
       QTreeWidget/QTableWidget, buttons, etc.).
     - The user must be able to:
       - Inspect the list of candidate frames.
       - Toggle individual frames on/off.
       - Run analysis / grouping when relevant.
       - See a clear status/log area.
       - Validate (`OK`) or cancel (`Cancel`) with semantics identical to Tk.

3. **Respect `followup.md` as the task source of truth**
   - `followup.md` already contains phases and checklists.
   - **Do not rewrite** or duplicate `followup.md` here.
   - Use `followup.md` as the **authoritative list of tasks**:
     - Do **not** re-implement tasks already marked as `[x]`.
     - Only work on items that are unchecked or explicitly marked as TODO.
     - When you complete work, update `followup.md` accordingly.


## CURRENT FOCUS

You are no longer in the “initial porting” phase. Assume:

- The Qt main window **compiles and runs**.
- The Qt filter dialog **exists** but is functionally limited compared to Tk.

Your current focus is:

### FOCUS 1 — Remove Phase 4.5 controls from Qt main window

Work file: `zemosaic_gui_qt.py`.

Objectives:

- Ensure `zemosaic_gui_qt.py` **does not expose** a user-visible group for
  Phase 4.5 / “Super-tiles”.
  - Concretely: remove or conditionally disable the `QGroupBox` created in
    `_create_*` methods that exposes:
    - `qt_group_phase45` / “Phase 4.5 / Super-tiles”
    - `inter_master_merge_enable` checkbox
    - `inter_master_overlap_threshold`
    - `inter_master_stack_method`
    - `inter_master_min_group_size`, `inter_master_max_group`
    - `inter_master_memmap_policy`
    - `inter_master_local_scale`
    - `inter_master_photometry_intragroup`, `inter_master_photometry_intersuper`
    - `inter_master_photometry_clip_sigma`
  - You may:
    - Completely remove that group from the UI, OR
    - Guard it behind a clearly disabled code path (e.g. a constant
      `ENABLE_PHASE45_UI = False` and never set it to `True`).
    - In all cases, for this release, the **user must not see** Phase 4.5 controls.

- Mirror the Tk behavior on config defaults:
  - In `__init__`, after loading the config, **force**:
    ```python
    self.config["inter_master_merge_enable"] = False
    ```
  - Ensure that saving configuration from Qt never writes `True` to
    `inter_master_merge_enable`.

- Keep Phase 4.5 **runtime feedback**:
  - The worker still emits `phase45_event` messages.
  - Logging, status labels (e.g. “Phase 4.5 idle/complete”) and overlays should
    continue to work, just like in `zemosaic_gui.py`.
  - Do not remove the overlay classes / handlers, only the *configuration* UI.

### FOCUS 2 — Bring `zemosaic_filter_gui_qt.py` to full feature parity

Work files:

- Primary: `zemosaic_filter_gui_qt.py`
- Reference: `zemosaic_filter_gui.py` (Tk)

Objectives:

- Study `launch_filter_interface` and the full Tk filter GUI implementation:
  - Understand:
    - Stream-scan mode vs legacy list mode.
    - How batches are loaded and normalized (RA/DEC, WCS, headers).
    - How instrument detection works.
    - How groups and clusters are computed and represented.
    - How ZeQualityMT is used to filter bad frames.
    - How overrides and metadata are returned.
  - Reproduce the **same behavior** in Qt:
    - Same rules for filtering, grouping, and exclusion.
    - Same data structures in the return tuple.

- UI/Layout parity:
  - Implement in Qt:
    - A central list or table of frames (similar to the Tk tree/list).
    - A filter/status panel with:
      - Instrument info.
      - WCS / solved status.
      - Group/cluster info when available.
    - Controls for:
      - Launching analysis / clustering.
      - Toggling quality filters (ZeQualityMT), when available.
      - Handling pre-plan / master groups.
      - Preview limits (if present in Tk).
  - Use sensible Qt widgets:
    - `QTableWidget` or `QTreeWidget` for the file list.
    - `QGroupBox` + `QFormLayout` / `QGridLayout` for sections.
    - `QCheckBox`, `QSpinBox`, `QDoubleSpinBox`, `QComboBox`,
      `QProgressBar`, `QLabel`, `QPushButton`, `QPlainTextEdit` for logs, etc.

- Behavior & return values:
  - Ensure that calling the Qt entry point returns the **same tuple format**
    as the Tk `launch_filter_interface`:
    ```python
    filtered_list, accepted, overrides_dict
    ```
  - `overrides_dict` must contain the same keys as the Tk implementation
    when the corresponding operations are performed:
    - `"preplan_master_groups"`
    - `"autosplit_cap"`
    - `"filter_excluded_indices"`
    - `"resolved_wcs_count"`
    - …and any other keys defined in `zemosaic_filter_gui.py`.

- ASTAP / solver settings:
  - Reuse the same logic as Tk to:
    - Apply `solver_settings_dict` and `config_overrides`.
    - Configure ASTAP paths, search radius, downsample, sensitivity, timeout.
    - Configure ASTAP concurrency via
      `set_astap_max_concurrent_instances(...)`, if available.

- Logging and UX:
  - Provide a minimal, but clear log/status area similar to the Tk GUI:
    - Show important steps: scan start/end, groups built, filters applied.
    - Show warnings when ZeQualityMT is unavailable or when ASTAP config
      is incomplete.
  - Make sure the dialog remains responsive during stream scan and analysis.

### FOCUS 3 — Auto-organize Master tiles / coverage-first parity (Tk vs Qt)

Work files:

- Primary: `zemosaic_filter_gui_qt.py` (class `FilterQtDialog`)
- Reference: `zemosaic_filter_gui.py` (Tk filter, `_auto_organize_master_tiles` and friends)
- Diagnostics: `zemosaic_worker.py` (consumer of `filter_overrides`), `zemosaic_gui_qt.py` (how overrides are forwarded)

Objectives (align with Task S in `followup.md`):

- **Visual parity for Auto-organize Master Tiles**
  - When the user clicks **Auto-organize Master Tiles** in the Qt filter:
    - the Sky Preview / Coverage Map must show the same red dashed master rectangles
      as in the Tk filter (outer mosaic box + per-group boxes),
    - group colours and footprints must match Tk as closely as Matplotlib allows.
  - Reuse the Tk helpers (`cluster_seestar_stacks_connected`, `_auto_split_groups`,
    `_merge_small_groups`, `_draw_group_outlines`, etc.) instead of re-implementing logic.

- **Summary & logs parity**
  - After grouping finishes, update the Qt summary label with the localized
    `filter_log_groups_summary` string, *exactly like Tk*:
    - full sizes list in the log,
    - histogram/compact sizes string in the label.
  - Emit the same coverage-first log messages (`log_covfirst_start`, `log_covfirst_relax`,
    `log_covfirst_autosplit`, `log_covfirst_merge`, `log_covfirst_done`) and SDS messages
    (“ZeSupaDupStack: prepared N coverage batch(es) for preview.”) using the same
    `msg_key` + `kwargs` so localization stays in sync with Tk.

- **Override payload parity & freeze fix**
  - Ensure that `launch_filter_interface_qt(...)` returns the same override structure as
    Tk’s `launch_filter_interface(...)` for the same dataset, especially:
    - `preplan_master_groups`
    - SDS / coverage-first flags and thresholds
    - any `global_wcs_*` keys and `global_wcs_plan_override`
  - Investigate and fix the **Qt main window freeze** observed after using the filter:
    - Do not run heavy clustering / coverage code directly in the GUI thread;
      use worker threads / queued callbacks like Tk.
    - Compare `zemosaic_worker.log` and the console between Tk and Qt runs and align
      the payloads until the worker behaves identically.

- **Testing requirements**
  - Always test changes by running **both**:
    - Tk main GUI + Tk filter (reference behaviour),
    - Qt main GUI + Qt filter (PySide6),
    on the same Seestar test data.
  - Only consider the task done when:
    - previews, logs, and summary texts match,
    - and launching a full run from the Qt main GUI no longer freezes.

> Important: Do **not** touch the astrophotography algorithms (`zemosaic_worker.py`,
> `lecropper.py`, etc.) beyond reading how they consume `filter_overrides`.  
> All fixes must be in the Qt/Tk glue and in `zemosaic_filter_gui_qt.py`.

## HARD CONSTRAINTS

These constraints are **non-negotiable**:

1. **Do not break the Tk GUI.**
   - `zemosaic_gui.py` and `zemosaic_filter_gui.py` must continue to work
     exactly as before.
   - Avoid modifying Tk files unless strictly necessary and explicitly
     requested in `followup.md`.

2. **Do not change the astrophotography/stacking algorithms.**
   - `zemosaic_worker.py`, `lecropper.py`, `zequalityMT.py`, WCS logic, etc.
     are the single source of truth for the processing pipeline.
   - You may call into them, but do not refactor their algorithms here.

3. **Do not introduce heavy new dependencies.**
   - You may use standard library and existing dependencies.
   - Do not add new GUI/toolkit dependencies beyond PySide6.
   - Do not add new C/C++ extensions.

4. **Respect localization.**
   - All user-visible strings must go through the localization system when
     practical.
   - Use keys consistent with existing patterns (`qt_*`, `filter.*`, etc.)
     and update `locales/en.json` and `locales/fr.json` when new keys are added.
   - Do not remove or rename existing keys unless strictly necessary and
     then update both locales accordingly.

5. **Configuration parity:**
   - When Qt writes configuration, resulting values must be compatible with
     what the Tk GUI expects (same keys, same types).
   - Never silently change semantics of existing config keys.

6. **Keep the code organized and readable.**
   - Follow the existing style and patterns used in `zemosaic_gui.py` and
     `zemosaic_filter_gui.py`.
   - Prefer small, focused methods over massive monoliths.
   - Add comments when behavior is non-obvious (especially where Qt must
     match Tk behavior exactly).


## WORKFLOW & CHECKLISTS

- **`followup.md` is the canonical backlog.**
  - Do not duplicate its content here.
  - Do not change its structure unless the user explicitly asks.
  - Use its checklists to decide what to do next.

- When you implement something:
  - Make small, atomic changes per commit.
  - Keep the PySide6 code compiling and runnable at all times.
  - Manually test:
    - Launch via Tk → Filter → Qt filter (when wired).
    - Launch Qt main GUI from `run_zemosaic.py` using the documented flags/env.

- When updating `followup.md`:
  - Mark tasks as `[x]` only after they are fully implemented and manually tested.
  - Add short notes if something remains partially done or has caveats.
  - Never mark as `[x]` a task you haven’t implemented and verified.


## SUMMARY

Your role:

- Be the disciplined **Qt parity and cleanup agent** for ZeMosaic.
- Make PySide6 GUIs behave like their Tk counterparts, without regressing stability.
- Remove Phase 4.5 user controls from the Qt main window while preserving logs/overlays.
- Bring `zemosaic_filter_gui_qt.py` to functional and UX parity with `zemosaic_filter_gui.py`.
- Always keep `followup.md` in sync with the actual work done.
- Never break the Tk GUI or the core astrophotography/stacking logic.
