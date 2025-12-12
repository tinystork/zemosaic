# Mission: Fix ETA + Progress bar in SDS (SupADupStack) mode only

## High-level goal
In ZeMosaic GUI (Tk + Qt), fix the ETA display and the global progress bar behavior when running SDS (SupADupStack).
Symptoms:
- Progress bar can stay around ~75% even when the run is finishing (phase 6/7 or 7/7).
- ETA can show 00:00:00 while the run is still active / not fully completed.
- The log shows ETA_UPDATE and PHASE_UPDATE events are emitted, plus a structured STATS_UPDATE payload, but GUI does not reflect correct global progress/ETA.

## Scope (STRICT)
- Only fix SDS progress + ETA reporting.
- Do NOT change the classic pipeline progress behavior.
- Do NOT modify GPU/CPU compute paths, stacking logic, nor any photometric/color/crop logic.
- No refactor “for style”. Minimal, surgical changes.

## Evidence from logs (do not ignore)
Worker emits:
- `PHASE_UPDATE:<int>` events, including phase 7 cleanup.  
- `RAW_FILE_COUNT_UPDATE:<done>/<total>` during phase 1 preprocessing.
- `ETA_UPDATE:HH:MM:SS` frequently during processing.
- `[CLÉ_POUR_GUI: STATS_UPDATE] (Args: {... phase_index, phase_name, files_done, files_total, tiles_total, ...})`
These appear in the SDS logs and should drive UI updates.

## Strategy
Implement a robust SDS-specific "global progress fraction" computation using structured worker signals (prefer `STATS_UPDATE` and fallback to RAW_FILE_COUNT_UPDATE/PHASE_UPDATE if needed), then feed it to the GUI progress bar consistently.

### Requirements
1) Add/ensure a single normalized metric in GUI: `global_progress_0_1` in [0..1] for SDS runs.
2) ETA label must display latest ETA_UPDATE value whenever received (except after completion).
3) Progress bar must:
   - increase monotonically (never go backwards unless a new run starts),
   - reach 100% at successful completion event,
   - not freeze at an arbitrary fraction.

### SDS progress model (proposed)
Use a phase-weight model based on SDS phases:
- Phase 1: preprocessing files_done/files_total
- Phase 2..5: internal processing (unknown fine-grain) -> use phase_index weighting only
- Phase 6: save -> treat as near completion (e.g. 0.97..0.995 window)
- Phase 7: cleanup -> final (0.995..1.0)

Implement as:
- If in phase 1 and files_total>0: phase_progress = files_done/files_total
- Else: phase_progress = 0 for that phase unless we have a better per-phase metric
- global_progress = (phase_index-1 + phase_progress) / total_phases
Where `total_phases` for SDS must be 7 (not hardcoded to classic count).
Then clamp [0,1] and apply monotonic smoothing in GUI:
`global_progress = max(previous_global_progress, global_progress)`.

Important: this model is SDS-only.

### Completion handling
When worker sends run success event (e.g. `run_success_processing_completed`), force progress bar to 100% and keep ETA at 00:00:00. Before that, ETA must not be forced to zero.

## Files to modify
- zemosaic_gui_qt.py (Qt UI: progress bar + ETA label update)
- zemosaic_gui.py (Tk UI: progress bar + ETA label update)
OPTIONAL (only if needed, minimal):
- zemosaic_worker.py: if GUI cannot reliably detect SDS phase_count, emit a dedicated SDS meta event once, e.g. `[CLÉ_POUR_GUI: SDS_META] (Args: {'total_phases': 7})`
But prefer fixing GUI using existing STATS_UPDATE/PHASE_UPDATE if possible.

## Tests
1) Run a small SDS dataset (like the example 66 files). Confirm:
   - progress increments during phase 1 using RAW_FILE_COUNT_UPDATE + STATS_UPDATE
   - progress continues after phase 1 (phase updates advance progress)
   - progress reaches 100% at completion
   - ETA updates live from ETA_UPDATE and does not lock to 00:00:00 prematurely
2) Run classic pipeline once to ensure unchanged behavior (sanity check only).

## Deliverables
- Minimal patch implementing SDS-only progress/ETA correctness for both GUI frontends.
- No behavior change in classic mode.
- Add a short inline comment explaining SDS phase_count=7 and why.
