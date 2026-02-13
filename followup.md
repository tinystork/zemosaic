# followup.md — Checklist DBE v2

## Progress tracking
(Convention: use [x] when done)
- Rule: if an item is unclear, verify in code first; if already done, mark [x] and log evidence in `memory.md`.

### 0 — Preflight verification (before new edits)
- [x] Verify `final_mosaic_dbe_enabled` is already wired (`zemosaic_config.py`, `zemosaic_gui_qt.py`, `zemosaic_worker.py`).
- [x] For each doubtful item, run a quick code check before implementation (avoid duplicate work).

### A — Configuration (defaults + compat)
- [x] Add new config keys in `zemosaic_config.py` defaults:
  - final_mosaic_dbe_strength
  - final_mosaic_dbe_obj_k
  - final_mosaic_dbe_obj_dilate_px
  - final_mosaic_dbe_sample_step
  - final_mosaic_dbe_smoothing
- [x] Add same defaults in Qt fallback defaults (`zemosaic_gui_qt.py` default config dict).
- [x] Update `memory.md` with chosen key names + preset mapping.

### B — Qt GUI (presets)
- [x] In `zemosaic_gui_qt.py`, add combobox “DBE strength” next to DBE checkbox:
  - Weak / Normal / Strong
- [x] Ensure combobox disabled when `final_mosaic_dbe_enabled` unchecked.
- [x] Keep `custom` config-only for power users (JSON/manual config), not exposed in GUI.
- [x] Update `memory.md` with screenshots/notes (if any) + where UI lives.

### C — Worker DBE algorithm (surface-fit + fallback)
- [x] Implement object mask in low-res using median+MAD threshold + dilation.
- [x] Implement background sampling on grid (sample_step) with robust local median.
- [x] Implement surface fit using SciPy RBF thin-plate + smoothing:
  - cap sample count (<=2000)
  - fallback order: RBF -> gaussian -> skip DBE
  - if too few samples or RBF fail -> fallback gaussian method
  - if gaussian fail -> skip DBE safely
- [x] Keep per-channel processing (no HWC model allocation).
- [x] Extend DBE info dict: model, strength, params, sample counts, fallback info.
- [x] Add DEBUG logs for fallback transitions and failure reasons.
- [x] Update `memory.md` with algorithm details + limits.

### D — Hook Phase 6 wiring (x2 blocks)
- [x] In BOTH phase6 DBE blocks in `zemosaic_worker.py`, read config values and pass them into DBE.
- [x] Ensure preset mapping applied correctly (GUI exposes weak/normal/strong; `custom` remains config-only).
- [x] Extend logs and (optional) FITS header keys for new DBE fields.
- [x] Update `memory.md` with wiring notes + expected log line example.

### E — Smoke tests
- [x] Manual user test (reduced dataset) - DBE ON: verify logs show:
  - model=rbf_thin_plate (or gaussian fallback) + strength + params + n_samples
- [ ] Manual user test - DBE OFF: verify DBE skipped cleanly
- [ ] Manual user test - Switch presets: verify worker logs show changed params
- [ ] Manual user test - Simulate failure (force exception in fit) -> verify gaussian fallback, then skip DBE if gaussian fails
- [ ] Update `memory.md` with manual test outcomes shared by user.

### Done criteria
- [ ] UI shows DBE strength selector (Weak/Normal/Strong only)
- [ ] Config persists new keys
- [x] Worker uses surface-fit DBE with ordered robust fallback (RBF -> gaussian -> skip)
- [ ] No regressions SDS/grid/classic
- [ ] `memory.md` updated throughout

## Notes / reminders
- Always mark completed tasks with [x]
- Always update `memory.md` each iteration
- Scope GUI: `zemosaic_gui_qt.py` only; do not consider `zemosaic_gui.py`.
