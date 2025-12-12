# Implementation checklist

## 1) Create zemosaic_resource_telemetry.py
- [x] Copy (verbatim) from `zemosaic_worker.py`:
  - `_sample_runtime_resources_for_telemetry()`
  - `ResourceTelemetryController`
- [x] Keep dependencies light:
  - psutil optional
  - GPU sampling: CuPy if available (match worker behavior)
- [x] Keep the emitted event format identical:
  - call `progress_callback("STATS_UPDATE", None, "INFO", **payload)`
  - payload includes `timestamp_iso` (UTC ISO string)

## 2) Refactor zemosaic_worker.py (no behavior change)
- [x] Remove the in-file definitions you moved.
- [x] Replace with:
  - `from zemosaic_resource_telemetry import ResourceTelemetryController`
  - (If worker still uses `_sample_runtime_resources_for_telemetry` directly, import it too; otherwise keep it internal to the module.)
- [x] Run a quick lint/import check to ensure the worker still imports.

## 3) Wire telemetry inside grid_mode.py
### 3.1 Instantiate telemetry
- [x] Read config flags from `zconfig` similarly to worker:
  - enable_resource_telemetry (bool)
  - resource_telemetry_interval_sec (float, clamp to >= 0.5)
  - resource_telemetry_log_to_csv (bool; default True)
- [x] If logging to CSV enabled and output_folder is set:
  - csv_path = os.path.join(output_folder, "resource_telemetry.csv")
- [x] Create `telemetry = ResourceTelemetryController(enabled=..., interval_sec=..., callback=progress_callback, csv_path=csv_path)`

### 3.2 Provide a context builder
Add a small helper inside grid_mode.py:

- [x] `_grid_telemetry_context(phase_index, phase_name, tiles_done, tiles_total, eta_seconds, files_done=None, files_total=None, extra=None) -> dict`
- [x] Always include:
  - phase_index, phase_name, tiles_done, tiles_total
- [x] Include eta_seconds when available (float/int >= 0)
- [x] Include files_done/files_total if grid knows it; otherwise omit.

### 3.3 Emit telemetry frequently but cheaply
- [x] Call `telemetry.maybe_emit_stats(ctx)`:
  - at run start with phase_index=0, phase_name="Grid: Init"
  - whenever grid updates progress/ETA (same cadence you already use for GUI updates)
  - at each phase boundary with `force=True` via `telemetry.emit_stats(ctx, force=True)` if helpful

### 3.4 Ensure tiles counter is global and monotonic
- [x] If grid currently reports tiles per “super tile”, introduce a global counter:
  - tiles_total = total master tiles expected for the run
  - tiles_done increments when a tile is completed (persisted/saved)
- [x] Feed these values consistently in both:
  - your progress/status emissions
  - telemetry context

### 3.5 Ensure progress reaches 100%
- [x] At normal completion:
  - emit a final progress update of 100.0
  - emit telemetry with tiles_done == tiles_total and eta_seconds omitted or 0
- [x] At cancellation/error:
  - do not force 100% unless the GUI expects a “finished” event; just ensure telemetry closes.

### 3.6 Always close telemetry
Wrap grid main routine with try/finally:
- [x] `finally: telemetry.close()` (guarded)

## 4) Validate with the Qt GUI expectations
- [x] Confirm the payload keys match what `zemosaic_gui_qt.py` consumes:
  - cpu_percent, ram_used_mb, ram_total_mb, gpu_used_mb, gpu_total_mb, eta_seconds, tiles_done, tiles_total, phase_index, phase_name

## Notes / non-goals
- Do not alter SDS phases or their progress math.
- Do not modify classic pipeline progress/ETA.
- Do not add new GUI signals; reuse existing `STATS_UPDATE` + existing progress updates.

# Quick manual test script
- Enable telemetry checkbox in GUI
- Run grid mode on a small dataset (fast)
- Watch resource monitor label update
- Ensure tiles counter increases globally and ends at total
- Ensure progress hits 100%
