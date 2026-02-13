# Mission: Add final-mosaic DBE (Dynamic Background Extraction) option (default ON)

## Meta (MANDATORY for Codex)
- You MUST update `memory.md` at the end of EACH iteration/PR.
- `memory.md` must include:
  - What changed (files + functions)
  - Why (intent / bug avoided)
  - How tested (manual runs + what to look for)
  - Any known limitations / TODOs
  - Next step (if unfinished)

## Goal
Add a checkbox in the GUI under "Final assembly & output" (enabled by default) that enables a global DBE-like background extraction on the FINAL mosaic right before export.

DBE must reduce large-scale gradients for multi-instrument / mixed Bortle datasets **without introducing seams**.

## Scope (files)
- zemosaic_gui.py
- zemosaic_worker.py
- (optional helper) zemosaic_utils.py
- (optional) locales/*.json for UI translation key
- NOTE: Grid mode may require a small hook in grid_mode.py IF it bypasses zemosaic_worker Phase 6 saving.

## Constraints / Guardrails
- NO regression for:
  - SDS mode
  - grid mode
  - classic mode
- Do NOT touch/alter batch-size behaviors
- Must be safe for large mosaics (memory-aware). Prefer downsampled model + per-channel application.
- Must preserve existing export behavior (FITS, ALPHA extension).
- Fail-open: if DBE fails for any reason, log WARN and continue without DBE.
- When DBE is OFF: output should be identical to baseline (avoid accidental dtype conversions/copies).

## Implementation Plan

### 0) Decide WHERE to apply DBE (important)
- Classic + SDS paths: apply DBE **in zemosaic_worker.py Phase 6**, immediately after the `P6_PRE_EXPORT` debug stats block (search `_dbg_rgb_stats("P6_PRE_EXPORT", ...)`), and right before any disk writing (FITS/PNG).
- Grid mode: confirm whether grid mode uses the same Phase 6 save path. If grid mode saves inside `grid_mode.py`, add a small hook there right before saving the final mosaic (same helper, same config flag). If unsure, log an INFO once: `[DBE] grid_mode: bypassing worker Phase 6, applying in grid_mode`.

### 1) GUI (zemosaic_gui.py)
- Add Tk var:
  - `self.final_mosaic_dbe_var = tk.BooleanVar(default=self.config.get("final_mosaic_dbe_enabled", True))`
- Add Checkbutton inside `final_assembly_options_frame`:
  - label: "Dynamic Background Extraction (DBE) on final mosaic"
  - default ON
  - store in translatable_widgets with key `final_mosaic_dbe_label` (optional)
- When starting processing (where config keys are persisted), save:
  - `self.config["final_mosaic_dbe_enabled"] = bool(self.final_mosaic_dbe_var.get())`
  - `zemosaic_config.save_config(self.config)`

### 2) Worker integration (zemosaic_worker.py)
- In Phase 6, after `P6_PRE_EXPORT` stats and BEFORE writing files:
  - read config:
    - `enabled = bool(getattr(zconfig, "final_mosaic_dbe_enabled", True))`
  - guard: only apply if `final_mosaic_data_HWC` is numpy HWC RGB (`ndim==3` and `shape[-1]==3`)
  - validity mask priority:
    1) `alpha_final > 0` if available (uint8)
    2) `final_mosaic_coverage_HW > 0` if available
    3) fallback: `np.isfinite(final_mosaic_data_HWC[...,0])` (or any-channel finite)
  - log (INFO):
    - `[DBE] enabled=True ds_factor=<int> obj_k=<float> blur_sigma=<float> masked_frac=<float> valid_frac=<float> time_ms=<int>`
  - write FITS header flags (only if applied):
    - `ZMDBE = T`
    - `ZMDBE_DS = <int>`
    - `ZMDBE_K = <float>`
    - `ZMDBE_SIG = <float>`

### 3) DBE algorithm (robust, cheap, memory-aware)
Implement as a helper (recommended in zemosaic_utils.py): `apply_final_mosaic_dbe(...)`

Requirements:
- Work in float32 internally. Do not blow RAM by allocating a full HWC background buffer.
- Compute a downsample factor so the working image longest side <= 1024 px.
- Build object mask on downsampled luminance:
  - sigma via MAD: `sigma = 1.4826 * MAD`
  - threshold: `median + k*sigma` with default `k = 3.0`
  - dilate mask lightly (kernel 3–7) to include star halos
- Fill masked pixels with background median (so blur does not “eat” stars/galaxies).
- Estimate smooth background on downsampled image:
  - use `cv2.GaussianBlur` with large `sigma` (default ~32 at downsampled scale)
- Upsample background to full resolution and subtract **per-channel**:
  - For c in (R,G,B):
    - downsample channel -> mask -> blur -> upsample -> subtract into mosaic[...,c] only on valid pixels
- Preserve invalid regions:
  - Do not alter pixels where validity mask is False (keep NaN / untouched).
- If cv2 is missing or any step fails: warn + skip (fail-open).

## Acceptance criteria
- Checkbox appears in GUI, default ON.
- With DBE ON: gradients reduced on a test mosaic (no obvious seams introduced).
- With DBE OFF: output should be identical to baseline (no unintended conversions).
- No crashes in SDS/grid/classic, and ALPHA extension still written.
- If DBE errors occur: a WARN log is emitted and export still completes.
- `memory.md` updated every iteration.
