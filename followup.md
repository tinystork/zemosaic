# SDS GUI progress/ETA/counter fix — detailed checklist

## 0) Guardrails
- [ ] Touch only GUI progress logic for SDS.
- [ ] Do NOT refactor worker pipeline, GPU helpers, stacking, or non-SDS progress computation.
- [ ] Keep progress monotonic (never goes backward).

## 1) Locate SDS progress logic (Qt + Tk)
- [ ] In zemosaic_gui_qt.py:
  - find _maybe_detect_sds, _compute_sds_progress_fraction, _apply_sds_progress
  - locate where “Tiles: X/Y” label is updated
  - locate where ETA is computed / smoothed
- [ ] In zemosaic_gui.py: same functions (Tk version) and keep parity.

## 2) Fix total phases so 75% cap cannot happen
- [ ] Identify where _sds_total_phases is set (if anywhere).
- [ ] Ensure SDS uses 7 phases (1..7) unless a real 8th phase exists.
  - If unknown, default SDS total phases to 7.
  - Document in code comment: “SDS phases: 1=Preprocess, 2=Cluster, 3=MasterTiles?, 4=GlobalCoadd, 5=Polish, 6=Save, 7=Cleanup”.

## 3) Extend _compute_sds_progress_fraction to account for phases beyond phase 1
### Phase 1 (already OK)
- [ ] Keep current behavior: files_done/files_total updates.

### Phase 4 (global coadd progress)
- [ ] When SDS phase4 is active (p4_global_coadd_progress / _sds_phase_active):
  - phase_progress = _sds_phase_done / _sds_phase_total
  - ensure bounded in [0,1]

### Phase 5 (polish) — indeterminate
- [ ] Do NOT attempt time-based progress.
- [ ] While in phase 5:
  - keep fraction monotonic but do not update ETA based on progress slope.
  - set ETA label to “--:--:--” (or a consistent neutral display) until phase changes.
- [ ] Mark phase 5 complete when:
  - receiving sds_global_finalize_done, OR
  - receiving run_info_phase6_started (entering phase 6).

### Phases 6 & 7
- [ ] Treat as deterministic short phases:
  - phase 6 complete when run_success_mosaic_saved AND/OR run_success_preview_saved occurs.
  - phase 7 complete when final “processing completed successfully” key is received.

## 4) Force 100% on completion
- [ ] When GUI receives the final success/completion event:
  - set _sds_completed=True
  - call _apply_sds_progress(1.0) so progress bar shows 100%
  - set ETA “00:00:00”

## 5) Fix the misleading “Tiles” counter in SDS
- [ ] In SDS mode, override the top-right counter to show global raw progress:
  - display_done = _sds_files_done
  - display_total = _sds_files_total (e.g. 66)
- [ ] Do NOT show per-batch totals (6/6, 10/10) in that top-right slot.
  - Keep per-batch in log only if needed.

## 6) Regression safety
- [ ] Confirm classic mode progress bar behavior unchanged.
- [ ] Confirm grid mode progress bar behavior unchanged.
- [ ] Confirm no worker-side changes.

## 7) Quick manual validation steps
- [ ] Run SDS on dataset with known total (e.g. 66):
  - progress must reach 100%
  - counter shows x/66
  - phase 5 ETA neutral (“--:--:--”)
  - completion sets ETA “00:00:00”
