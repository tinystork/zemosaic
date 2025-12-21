# Follow-up: Validate Qt GUI config sync

## What to check after patch
### A) Live sync sanity
- [ ] Launch Qt GUI
- [ ] Open Advanced -> Quality crop
- [ ] Toggle `Enable quality crop` ON/OFF
- [ ] Without starting a run, trigger any action that reads config (optional):
  - change theme mode (forces some config interactions)
  - open filter dialog and cancel
- [ ] Start a run and confirm:
  - Log contains `RUN CONFIG SNAPSHOT` and it matches the UI

### B) Worker kwargs consistency
- [ ] Confirm the worker log / exported config indicates the same values:
  - `quality_crop_enabled`
  - `apply_master_tile_crop`
  - `master_tile_crop_percent`
  - `altaz_cleanup_enabled`
  - `altaz_nanize`
  - `global_wcs_autocrop_enabled`

### C) Snapshot file (if implemented)
- [ ] In output dir, verify `run_config_snapshot.json` exists and matches UI choices.

## Regression checks
- [ ] GPU selector still works:
  - enable/disable GPU checkbox toggles the combobox enabled state
  - changing GPU selector updates both `gpu_selector` and `gpu_id_phase5`
- [ ] Language switching rebuilds UI (`_refresh_translated_ui`) and does not break bindings
- [ ] Filter window workflow unchanged:
  - filter button -> filter dialog -> start processing still works

## Expected outcome for the original SDS cropping confusion
With correct propagation:
- [ ] if UI shows `quality_crop_enabled=False` and `apply_master_tile_crop=False`, the config snapshot must reflect that.
- [ ] Remaining “missing signal” will then be attributable to genuine coverage/alpha behavior or other preprocessing, not a stale config.
