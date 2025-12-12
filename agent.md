# Mission: Fix SDS (SupaDupStack) GUI progress/ETA/tile counter (Qt + Tk parity)

## Scope (STRICT)
- Only modify SDS/SupaDupStack progress tracking & display logic.
- Do NOT change classic pipeline, grid mode, or any non-SDS progress behavior.
- Do NOT change stacking, reprojection, GPU helpers, or worker compute logic: GUI-only fix.
- Keep behavior “batch size = 0” and “batch size > 1” untouched.

## Symptoms to fix (observed)
1) In SDS runs, progress bar never reaches 100% (stuck ~75% even at P7 Cleanup).
2) “Tiles: X/Y” is misleading in SDS: it shows per-super-tile frames (e.g. 6/6 or 10/10) instead of global raw count (e.g. 66).
3) ETA becomes meaningless (especially during SDS phase 5 polish) because progress is not tracked within phases.

## Root cause (likely)
GUI SDS global progress fraction only accounts for phase 1 files_done/files_total; other phases use phase_progress=0, so global_fraction saturates at (phase_index-1)/total_phases.
If total_phases=8 => phase7 shows 6/8=75%.
Also completion event never forces 100% in SDS.

## Files to modify
- zemosaic_gui_qt.py  (Qt GUI)
- zemosaic_gui.py     (Tk GUI)  [keep parity]

## Required behavior (Acceptance Criteria)
A) In SDS mode, progress bar MUST hit 100% on successful completion (P7 done).
B) In SDS mode, the top-right counter MUST display a consistent global counter aligned with other flows:
   - show “Files: done/total” or reuse “Tiles:” label but it must be global (e.g. 66 total),
     NOT the per-super-tile frames.
C) ETA:
   - During phases where progress cannot be estimated (especially SDS Phase 5 polish),
     show a neutral ETA (e.g. “--:--:--”) and do not “count upward/negative”.
   - When run completes successfully, ETA must become 00:00:00.
D) Non-SDS runs must remain identical.

## Implementation plan (high-level)
1) Identify SDS detection and tracking fields:
   - _sds_progress_active, _sds_current_phase_index, _sds_files_done/_total,
     _sds_phase_done/_total (phase4 global coadd), _sds_completed, etc.

2) Fix SDS total phase count:
   - Ensure _sds_total_phases matches actual SDS phases (1..7) unless there is a real phase 0/8.
   - Avoid 8-phases if it causes 75% cap.

3) Make _compute_sds_progress_fraction handle multiple phases:
   - Phase 1: keep current logic (files_done/files_total).
   - Phase 4: when p4_global_coadd_progress is active, use _sds_phase_done/_sds_phase_total.
   - Phase 5: treat as indeterminate:
       - do not try to compute phase_progress from time.
       - keep progress monotonic, but do not update ETA based on it.
       - mark phase 5 complete when we enter phase 6 (run_info_phase6_started) or receive sds_global_finalize_done.
   - Phase 6/7: treat as quick deterministic phases:
       - mark phase complete on corresponding “finished”/success keys
         (run_success_mosaic_saved, run_success_preview_saved, processing completed, etc.)

4) Force completion:
   - On the GUI event indicating success completion (whatever key already used),
     set _sds_completed=True and call _apply_sds_progress(1.0) (100%).
   - Also reset ETA display to 00:00:00.

5) Fix counter display:
   - When SDS is active, override the “Tiles:” display to show global files_done/files_total (e.g. 66),
     not per-batch counts.
   - Keep the existing per-batch info in the log area only (optional).

6) Keep parity:
   - Apply same conceptual fix to both Qt and Tk implementations.

## Testing (manual)
- Run SDS on sample with total raw frames = 66 and mega_tiles ~ 7.
- Verify:
  - progress reaches 100% at end;
  - “Files/Tiles” counter shows x/66 throughout;
  - ETA is stable; phase 5 shows “--:--:--” (or equivalent) and does not mislead;
  - classic mode unchanged.
