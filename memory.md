# Memory (curated)

Last curated: 2026-03-15
Purpose: keep durable technical history + production-relevant decisions, remove transient debug noise.
Full previous log archived in: `memory_full_archive_20260315-1024.md`.

---

## 1) Project-level durable decisions

- Official GUI stack is Qt/PySide6.
- Tk legacy paths were progressively retired from official runtime paths (kept only where explicitly legacy/annexed).
- Three operational processing families remain distinct and must not be merged conceptually:
  1) Classic
  2) ZeGrid
  3) SDS
- Safety rule maintained through missions: mode-by-mode surgical patches, no broad transversal rewrites.

---

## 2) Packaging and runtime robustness (historical)

### Windows packaged runtime (PyInstaller)
- Main issues handled:
  - DLL search-path fragility in frozen mode
  - CuPy/fastrlock packaging misses
  - Shapely binary hidden-import misses
  - SEP top-level `_version` packaging issue
- Durable outcome:
  - stronger runtime hooks/spec for packaged dependencies
  - better packaged-mode diagnostics

### Build documentation alignment
- Clear split documented between:
  - `requirements.txt` (GPU-enabled path)
  - `requirements_no_gpu.txt` (CPU-only smaller package)
- Installer behavior clarified (packages existing `dist/` output, does not decide CUDA flavor itself).

---

## 3) GUI / UX durable fixes

### Qt filter dialog accessibility
- Right-side controls moved into scrollable area.
- Action buttons (OK/Cancel) kept permanently accessible.
- Saved geometry restore clamped against available screen geometry.

---

## 4) Mission: multi-mode quality harmonization (normalization / RGB / DBE)

Reference scope files:
- `agent.md`
- `followup.md`

### Mission intent
Bring non-classic modes closer to classic quality behavior while preserving non-regression and mode-specific architecture.

### Status
- E (SDS quality harmonization): completed.
- F (ZeGrid DBE priority): completed and field-validated.

---

## 5) Final RGB Equalization policy (important)

### Capability retained
`final_mosaic_rgb_equalize_enabled` remains available and documented for marginal datasets.

### Current production preference (Tristan)
- Set to OFF in user JSON config (`~/.config/ZeMosaic/zemosaic_config.json`).
- Rationale: limited benefit on current dataset, keep as optional targeted lever.

### Related tuning keys to remember
- `final_mosaic_rgb_equalize_clip_enabled`
- `final_mosaic_rgb_equalize_gain_clip`
- `existing_master_tiles_final_rgb_equalize_gain_clip`
- `sds_enable_final_rgb_equalize`
- `sds_final_rgb_equalize_gain_clip`

---

## 6) SDS branch — durable fixes and controls

### Stabilization fixes kept
- Regression fixes around SDS-specific path integrity (name/shape guards, geometry-safe behavior, OOM resilience).
- Geometry-changing crop prevented in SDS finalization when incompatible with global descriptor constraints.

### Optional SDS equalization controls (default conservative)
- `sds_enable_final_rgb_equalize` (default false)
- `sds_final_rgb_equalize_gain_clip`
- `sds_enable_final_black_point_equalize` (default false)

### Durable rule
SDS global geometry invariants are non-negotiable; quality improvements must remain geometry-safe.

---

## 7) ZeGrid branch — DBE implementation history and final state

### Initial gap (confirmed)
- Worker Phase 6 DBE path is bypassed in ZeGrid route.
- Grid mode had DBE hook logging but no effective final DBE processing at that point.

### Implemented resolution
- Added effective ZeGrid final DBE step in `grid_mode.py` before final mosaic save.
- Added explicit DBE status logging (`applied/skipped`, reason, strength/sigma/channels).
- DBE applied on valid-covered regions only.

### Halo artifact mitigation
Observed issue: dark halos around bright stars in ZeGrid only.
Root causes:
- simpler ZeGrid DBE behavior vs classic/SDS worker DBE
- GUI label mapping mismatch (`weak/strong` not aligned with grid-side aliases)

Fixes delivered:
- GUI strength alias alignment (`weak/normal/strong` now mapped correctly for ZeGrid)
- bright-object protection in DBE model estimation
- avoidance of correction over protected bright cores

### Validation status
- Unit tests added/updated for ZeGrid DBE behavior and strength aliases.
- Real ZeGrid run validated visually by Tristan as successful.
- No blocking ERROR observed in end-of-run checks.

---

## 8) Production-relevant variables / switches (quick reference)

### Global / classic-facing
- `final_mosaic_rgb_equalize_enabled`
- `final_mosaic_rgb_equalize_clip_enabled`
- `final_mosaic_rgb_equalize_gain_clip`
- `final_mosaic_dbe_enabled`
- `final_mosaic_dbe_strength`
- `final_mosaic_dbe_obj_k`
- `final_mosaic_dbe_obj_dilate_px`
- `final_mosaic_dbe_sample_step`
- `final_mosaic_dbe_smoothing`

### Existing master tiles
- `existing_master_tiles_rgb_balance_prephase5`
- `existing_master_tiles_rgb_balance_gain_clip`
- `existing_master_tiles_rgb_balance_min_pixels`
- `existing_master_tiles_final_rgb_equalize_gain_clip`

### SDS
- `sds_enable_final_rgb_equalize`
- `sds_final_rgb_equalize_gain_clip`
- `sds_enable_final_black_point_equalize`
- `sds_mode_default`
- `sds_coverage_threshold`
- `sds_min_coverage_keep`

### ZeGrid
- `poststack_equalize_rgb` / grid RGB equalization path
- `final_mosaic_dbe_enabled`
- `final_mosaic_dbe_strength`

---

## 9) Known durable guardrails for future debug

- Always reason mode-by-mode (Classic / SDS / ZeGrid).
- Validate with artifacts + logs, not assumptions.
- Keep new quality controls conservative by default (opt-in where uncertain).
- Do not remove safety guardrails without replacement + explicit rationale.
- Prefer reversible/tunable config switches to hard behavior changes.

---

## 10) Current mission-level conclusion

- Multi-mode quality mission reached practical completion on current dataset.
- ZeGrid DBE is now operational and aligned with GUI toggles.
- Final RGB equalization remains available but intentionally OFF by default for current production use.
