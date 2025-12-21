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
- [x] Do not remove `_collect_config_from_widgets()`. Keep it as final safety net.

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
