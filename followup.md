# Follow-up checklist (Codex)

## 0) Guardrails
- [ ] Confirm all modifications are SDS-only.
- [ ] Do not touch stacking, reprojection, GPU kernels, normalization, cropper, or photometry.
- [ ] No refactor outside progress/ETA wiring.

## 1) Identify where GUI parses worker events
- [ ] Locate code handling log/event lines and keys:
  - `ETA_UPDATE:...`
  - `PHASE_UPDATE:<int>`
  - `RAW_FILE_COUNT_UPDATE:<done>/<total>`
  - `[CLÉ_POUR_GUI: STATS_UPDATE] (Args: {...})`
- [ ] Verify GUI currently uses a classic-only phase-count mapping that breaks SDS (likely wrong denominator -> ~75% plateau).

## 2) Add SDS-specific progress state machine (Qt GUI)
In zemosaic_gui_qt.py:
- [ ] Track:
  - `self._sds_active` (bool)
  - `self._sds_total_phases = 7`
  - `self._sds_current_phase_index`
  - `self._sds_files_done/files_total` (phase 1)
  - `self._sds_last_eta_str`
  - `self._sds_global_progress_0_1` (monotonic)
- [ ] When receiving STATS_UPDATE:
  - detect SDS mode (if the GUI already has a config flag, use it; otherwise infer from phase_name strings / known SDS events like `sds_global_finalize_done`)
  - read phase_index and files_done/files_total if present
  - compute global_progress_0_1 with SDS formula (phase-weight)
  - clamp [0..1] and enforce monotonic non-decreasing
  - update progress bar
- [ ] When receiving RAW_FILE_COUNT_UPDATE (fallback):
  - update phase1 progress if in phase 1 and totals valid
- [ ] When receiving PHASE_UPDATE:
  - update phase index and recompute global progress
- [ ] When receiving ETA_UPDATE:
  - update ETA label with the provided string unless the run is already completed
- [ ] When receiving run completion event:
  - force progress bar to 100%
  - set ETA to 00:00:00

## 3) Mirror same logic for Tk GUI
In zemosaic_gui.py:
- [ ] Implement the same SDS-only global progress computation and ETA handling.
- [ ] Keep classic path unchanged.

## 4) Prevent premature ETA=00:00:00
- [ ] Ensure GUI does NOT overwrite ETA with 00:00:00 until:
  - completion event is received OR
  - explicit finalization stage (post completion) is entered.
Note: the worker may emit ETA_UPDATE:00:00:00 near the end, but GUI must still reach 100% and show correct phase text.

## 5) Sanity checks
- [ ] Run SDS test: progress should go 0→100, no 75% freeze.
- [ ] ETA should update frequently during phase 1 (as seen in logs), and not remain stuck.
- [ ] Run classic mode once: confirm progress/ETA unchanged.

## 6) Output
- [ ] Provide a concise summary of changes + where SDS-only branches are placed.
- [ ] Mention any assumptions used to detect SDS in GUI.
