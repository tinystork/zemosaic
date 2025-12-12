# Goal
Connect **Grid Mode** to the PySide6 GUI status bar:
- [x] correct progress / tiles counter
- [x] correct ETA (in seconds)
- [x] propagate **resource telemetry** (CPU/RAM/GPU) already exposed by the Qt GUI

Do **not** change SDS behavior nor classic pipeline behavior.

# Context
The Qt GUI already understands telemetry events:
- When the worker emits msg_key == "STATS_UPDATE", ZeMosaicQtWorker routes it to `stats_updated`.
- The GUI then uses payload fields like `cpu_percent`, `ram_used_mb`, `ram_total_mb`, `gpu_used_mb`, `gpu_total_mb`, `eta_seconds`, `tiles_done`, `tiles_total`, `phase_index`, `phase_name`.

Grid mode currently lacks this wiring because telemetry controller lives inside `zemosaic_worker.py` and grid_mode cannot import worker safely.

# Strategy (safe / minimal)
- [x] Create a new lightweight module `zemosaic_resource_telemetry.py` that contains:
   - `_sample_runtime_resources_for_telemetry()`
   - `ResourceTelemetryController`
   (Move code from zemosaic_worker.py with no behavioral changes.)

- [x] Update `zemosaic_worker.py` to import `ResourceTelemetryController` from that new module.
   - This is a refactor only. No behavior changes for classic/SDS.

- [x] In `grid_mode.py`:
   - instantiate a `ResourceTelemetryController` using zconfig flags:
     - enable_resource_telemetry
     - resource_telemetry_interval_sec
     - resource_telemetry_log_to_csv (write into output_folder/resource_telemetry.csv when enabled)
   - periodically call `telemetry.maybe_emit_stats(context)` where context includes:
     - phase_index, phase_name
     - tiles_done, tiles_total
     - files_done, files_total (optional if available)
     - eta_seconds (use the grid ETA model you already have)
     - gpu/cpu chunk knobs if grid has equivalents (optional)
   - ensure telemetry is closed at end (finally/try).

- [x] Ensure Grid Mode progress ends at 100% and tiles counter is global and monotonic:
   - tiles_done must be “overall completed tiles” (not per super-tile)
   - tiles_total must be stable and correct for the run
   - ensure last progress update emits 100.0 and eta stops / becomes placeholder.

# Constraints
- Do not touch SDS code paths.
- Do not modify classic pipeline semantics.
- PySide GUI only (Tk will be removed later). Do not add Tk-specific hooks.
- Keep progress_callback signature unchanged: `progress_callback(msg_key_or_raw, prog, lvl, **kwargs)`.

# Acceptance tests
Manual:
- Enable “Resource telemetry (CPU/GPU monitor)” in GUI.
- Run Grid Mode.
- Status bar shows CPU/RAM/GPU updating during run.
- Tiles counter shows `X / TOTAL` globally increasing.
- ETA is coherent (non-negative during phases, no bogus + time at end).
- Progress reaches 100% at completion.

Code-level:
- Import graph has no circular import between grid_mode and worker.
- `resource_telemetry.csv` is created in output folder when enabled.

# Files to modify
- NEW: `zemosaic_resource_telemetry.py`
- `zemosaic_worker.py` (refactor import only)
- `grid_mode.py` (wire telemetry + context + final 100%)
