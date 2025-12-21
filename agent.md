# Mission: Qt GUI config sync (ZeMosaic)

## Goal
Ensure ZeMosaic Qt GUI always passes the *exact* UI state to the worker:
- live-sync widgets into `self.config` as the user edits them
- guarantee the worker receives a deterministic snapshot (same snapshot that is saved)
- add a minimal “RUN CONFIG SNAPSHOT” log and (optional) write a JSON snapshot alongside outputs for debugging

## Constraints
- No refactor / no redesign of the UI
- Touch as few files as possible
- Must preserve current behavior when users do not touch widgets
- Do not change worker logic; only fix GUI->config->worker plumbing

## Files in scope
- `zemosaic_gui_qt.py` ONLY

## Problem statement
Many GUI fields are registered in `_config_fields` but do not update `self.config` until `_collect_config_from_widgets()` runs.
This can lead to UI showing a value (e.g. checkbox OFF) while the config passed to the worker still contains the old value (e.g. ON), because the snapshot/save/build step may rely on stale data.

## Implementation Plan

### 1) Add a single-key sync helper
- [x] Add a method in the main window class:

`def _sync_config_key_from_widget(self, key: str) -> None:`

It must:
- look up `binding = self._config_fields.get(key)`
- compute raw_value using the exact same logic as `_collect_config_from_widgets()` for that binding
  - honor `value_getter` if present
  - checkbox/spinbox/doublespinbox/combobox/line_edit/composite
- apply `postprocess` if present (same semantics)
- write the result into `self.config[key]`
- keep it safe (no exceptions bubbling up)

Optionally also provide:

`def _sync_config_keys(self, keys: list[str]) -> None:` to sync multiple linked keys (GPU selector updates 2 keys).

### 2) Wire live-sync in the register helpers
- [x] Modify these helper methods to connect widget signals so they call `_sync_config_key_from_widget(key)`:

- `_register_checkbox`: connect `toggled` or `stateChanged`
- `_register_spinbox`: connect `valueChanged`
- `_register_double_spinbox`: connect `valueChanged`
- `_register_line_edit`: connect `editingFinished` (and keep browse button behavior)
- `_register_double_pair`: connect both spinboxes `valueChanged`, updating the list in config
- `_register_gpu_selector`: connect combobox `currentIndexChanged` and `editTextChanged` to sync both `gpu_selector` + `gpu_id_phase5`

Important: use lambdas capturing `key` safely:
`checkbox.toggled.connect(lambda _=None, k=key: self._sync_config_key_from_widget(k))`

### 3) Make the worker use a deterministic snapshot
- [x] In `_build_worker_invocation()`:
- build `snapshot = self._serialize_config_for_save()` (already exists)
- use `worker_kwargs = snapshot.copy()` (instead of `self.config.copy()`)

This ensures:
- worker gets the same merged config that will be saved
- stale/missing keys don’t fall back to a previous loaded snapshot silently

Also ensure phase 4.5 disable guard remains applied.

### 4) Add a run snapshot log + optional JSON dump
- [x] In `_start_processing()` after `_collect_config_from_widgets()` and before spawning worker:
- compute `snapshot = self._serialize_config_for_save()`
- log a short one-liner with the most error-prone keys, e.g.:
  - `quality_crop_enabled`
  - `apply_master_tile_crop` / `master_tile_crop_percent`
  - `altaz_cleanup_enabled` / `altaz_nanize`
  - `global_wcs_autocrop_enabled`
  - `final_assembly_method` / `sds_mode_default` if relevant
- optional: if `output_dir` is valid, write `run_config_snapshot.json` there (overwriting ok)

The log message must be plain (no localization changes required), e.g.
`[INFO] RUN CONFIG SNAPSHOT: quality_crop_enabled=False, apply_master_tile_crop=False, ...`

### 5) Keep existing “collect on start” behavior
- [x] Do not remove `_collect_config_fro# agent.md — ZeMosaic Qt: "I’m using master tiles" (skip phases 0-3)

## Goal
Add a **Qt GUI toggle** (Main tab) that lets the user run ZeMosaic starting from **already-built master tiles** (FITS) that already contain a **valid WCS**, and **skip almost all early phases**:
- skip scan / clustering / master tile build / WCS solve for raws (phases 0-3)
- go directly to:
  - choose anchor (existing logic)
  - inter-tile photometric normalization
  - final reprojection / co-add assembly

The behavior must remain **transparent**:
- Default OFF → no behavior change.
- When ON → we **verify WCS validity** on found master tiles; if insufficient/invalid, **warn and fall back** to full pipeline.

## Scope / Files
Modify only:
- `zemosaic_gui_qt.py`  (new checkbox + config propagation + small UI disable/enable)
- `zemosaic_config.py` (add config default key)
- `zemosaic_worker.py` (skip-to-phase4 path when toggle enabled)
- `locales/en.json`, `locales/fr.json` (new UI strings; keep keys consistent)

No refactor. No redesign of pipeline. Keep existing function boundaries unless strictly necessary.

---

## Implementation details

### 1) Config key
Add a new config boolean in `zemosaic_config.DEFAULT_CONFIG`:

- key: `use_existing_master_tiles`
- default: `False`

This is a **top-level config** like other booleans.

### 2) Qt GUI (zemosaic_gui_qt.py)
Add a checkbox in the **Main** tab, in the **Folders** group (near input/output paths is fine):

- Label: `"I'm using master tiles (skip clustering & master tile creation)"`
- Tooltip (optional but recommended): mention that input folder must contain master tiles FITS with valid WCS.

Localization keys (suggested):
- `qt_use_existing_master_tiles_label`
- `qt_use_existing_master_tiles_tip`

Behavior:
- When checked:
  - Input folder is interpreted as a **master tiles folder**.
  - Disable/grey out **Mosaic / clustering** controls (cluster threshold etc.) since they won’t be used.
  - Also disable any “open filter/clustering UI” triggers if the Qt GUI has one (if not, ignore).
- When unchecked:
  - Restore normal enabled state.

Propagation:
- Ensure the checkbox writes to config key `use_existing_master_tiles` and is included in the config payload passed to the worker (same as other settings).

### 3) Worker shortcut (zemosaic_worker.py)
Add support in the worker pipeline for skipping early phases.

#### 3.1) New parameter in the main worker entry
Where the worker receives config values (the function that orchestrates phases, e.g. `run_hierarchical_mosaic(...)` or similar), add an optional boolean parameter:

- `use_existing_master_tiles_config: bool = False`

(Use the `_config` suffix if the worker uses the “auto rename config keys to *_config args” convention.)

#### 3.2) Detect & validate master tiles when enabled
When `use_existing_master_tiles_config` is True:

1) Collect candidate FITS files from `input_dir`:
   - Prefer files matching: `master_tile_*.fits` (common ZeMosaic output)
   - If none exist, fall back to: `*.fits` (but exclude obvious non-tiles)
   - Exclude likely non-tiles:
     - `global_mosaic_wcs.fits` / `global_mosaic.wcs.fits` (whatever name is used in config)
     - anything that looks like final mosaic output (e.g. contains `final_mosaic`)
   - Allow recursive search (`rglob`) only if the current pipeline already uses recursion; otherwise keep it flat (but deterministic ordering either way).

2) For each candidate file:
   - Open FITS header (primary HDU header).
   - Validate WCS using existing helper:
     - `from zemosaic_utils import validate_wcs_header`
     - `ok, detail, wcs_obj = validate_wcs_header(header, require_footprint=True)`
   - Keep only tiles where `ok is True` and `wcs_obj is not None`.

3) If you end up with **< 2 valid master tiles**:
   - Log a warning (“mode enabled but insufficient valid WCS master tiles found”) and
   - Fall back to the normal pipeline (do not crash).

4) If you have enough tiles:
   - Build `master_tiles_results_list` in the same shape as Phase 3 output:
     - `master_tiles_results_list = [(tile_path, wcs_obj), ...]`
   - Set / create needed dirs:
     - Ensure `output_dir` exists
     - Ensure the normal temp dir used later exists (ex: `output_dir/zemosaic_temp_master_tiles`) because Phase 4.5/5 may write intermediate results (super tiles).
   - Skip Phase 0-3 by jumping to the same code section that normally runs Phase 4 and then the shared Phase 4.5 / Phase 5 pipeline.
   - Important: do **not** run clustering or master tile generation.

Progress/logging:
- Emit a clear run info message early:
  - `run_info_existing_master_tiles_mode` (“Using existing master tiles input; skipping phases 0-3.”)
- Optionally bump progress to the Phase 4 baseline (it’s OK if the bar jumps; correctness over cosmetics).

#### 3.3) Keep the rest unchanged
After this shortcut builds `master_tiles_results_list`, the worker should follow the same path as normal:
- Phase 4: build final WCS grid / global mosaic frame (based on the master tiles WCS and shapes)
- Phase 4.5: intertile normalization (photometric match) if enabled
- Phase 5: final reprojection/co-add assembly using selected method

No change to existing algorithms.

### 4) Locales
Add translations for the new checkbox.

In `locales/en.json`:
- `qt_use_existing_master_tiles_label`: "I'm using master tiles (skip clustering & master tile creation)"
- `qt_use_existing_master_tiles_tip`: "Input folder must contain master tiles FITS with a valid WCS. ZeMosaic will skip phases 0–3 and go directly to inter-tile normalization and reprojection."

In `locales/fr.json`:
- `qt_use_existing_master_tiles_label`: "J’utilise déjà des master tiles (sauter clustering & création)"
- `qt_use_existing_master_tiles_tip`: "Le dossier d’entrée doit contenir des master tiles FITS avec un WCS valide. ZeMosaic sautera les phases 0–3 et ira directement à la normalisation inter-tuile puis à la reprojection."

---

## Acceptance criteria
- Checkbox exists in Qt GUI and persists via config.
- When ON and folder contains valid WCS master tiles:
  - no clustering / no master tile creation is executed
  - pipeline starts at intertile normalization + final assembly
  - final mosaic is produced normally
- When ON but tiles are missing/invalid WCS:
  - warning is logged
  - pipeline falls back to full workflow without crashing
- When OFF:
  - behavior identical to current release

## Notes
- This toggle is NOT the same as “force_resolve_existing_wcs”; this one bypasses *master tile creation* entirely.
- Prefer deterministic ordering of tiles (sorted paths) to keep reproducible behavior.
m_widgets()`. Keep it as final safety net.

## Acceptance Criteria
1. Toggling any checkbox (e.g. quality crop) immediately updates `self.config[key]`.
2. Starting a run passes worker kwargs consistent with the UI.
3. Saved config contains the same values as worker kwargs for key options.
4. A log line “RUN CONFIG SNAPSHOT” appears on every run with the key values.
5. No regressions in normal runs; GUI still launches and filter integration remains unchanged.

## Manual Test Plan
- Start GUI, load existing config
- Toggle `Enable quality crop` OFF
- Toggle `Apply master tile crop` OFF
- Start run (can cancel early)
- Verify in log “RUN CONFIG SNAPSHOT” shows those keys as False
- Verify written config JSON (and optional run snapshot JSON) show False
- Repeat: toggle back ON, confirm propagation
- Also test a non-checkbox (spinbox) changes appear in snapshot
