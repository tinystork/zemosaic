# Follow-up checklist: Grid mode progress + ETA (PySide)

## Context
Grid mode is invoked from the worker via `grid_mode.run_grid_mode(...)`.
The Qt GUI already updates:
- progress bar / stage label from `STAGE_PROGRESS`
- ETA label from `ETA_UPDATE:...`
- Tiles counter from `MASTER_TILE_COUNT_UPDATE:X/Y`

Your job is ONLY to make `grid_mode.py` emit these messages at the right times.

## Exact message formats to emit (do not invent new ones)
Use `progress_callback(msg_key, prog=None, lvl="INFO", **kwargs)`.

1) Stage progress (drives progress bar):
- msg_key = "STAGE_PROGRESS"
- prog = stage_name (string)
- lvl = current (int)
- kwargs["total"] = total (int)

Example:
```py
progress_callback("STAGE_PROGRESS", "GRID: tile stacking", 3, total=12)
ETA updates:

msg_key = f"ETA_UPDATE:{eta_str}"

lvl = "ETA_LEVEL"

Example:

py
Copier le code
progress_callback("ETA_UPDATE:00:12:34", None, "ETA_LEVEL")
Tiles counter:

msg_key = f"MASTER_TILE_COUNT_UPDATE:{done}/{total}"

Example:

py
Copier le code
progress_callback("MASTER_TILE_COUNT_UPDATE:3/12")
(Optionally) phase label:

msg_key = "PHASE_UPDATE:<something>"
But this is optional; stage_name from STAGE_PROGRESS is usually enough.

 Implementation steps (do them in order)
- [x] Add _GridProgressReporter in grid_mode.py (private helper).

- [x] Hook it into run_grid_mode:

- [x] Initial stage + initial ETA

- [x] Set tile_total once tiles are known; emit 0/N

- [x] Increment done tiles on each tile completion

- [x] Maintain overall progress units (stable global percent)

- [x] Emit ETA periodically (throttle)

- [x] Force final update at end (100% + 00:00:00)

- [x] Ensure all calls are guarded:

- [x] if progress_callback is None or not callable => no-op

- [x] Throttle emissions:

- [x] Aim <= ~5 updates/sec (0.2s) or even 0.5s; keep UI responsive

- [x] Do not change any algorithm outputs, file outputs, or logs.

Quick sanity test (manual)
Run any grid-mode dataset (stack_plan.csv present) and observe:

Tiles counter goes 0/N ... N/N

ETA changes over time

Progress reaches 100%

Non-goals
Do NOT touch SDS

Do NOT touch classic mode

Do NOT touch any GUI files

Do NOT change performance-critical loops except for throttled callbacks