# Mission (surgical / no refactor)
Add an "Alt-Az alpha soft threshold" parameter in the Advanced panel (GUI) + translations,
and use it in the lecropper pipeline (worker) to decide which pixels are considered transparent
when mask2d is returned (i.e., nanize/zeroize threshold on mask2d).

## Why
Currently zemosaic_worker hard-thresholds mask2d at 1e-3:
    mask_zero = alpha_mask_norm <= 1e-3
So feathered / partially transparent zones survive as valid pixels, often carrying ugly
color casts (magenta/green). We want a user-tunable cutoff (recommended ~0.85) without refactoring.

## Scope
- [x] zemosaic_worker.py: only inside _apply_master_tile_quality_pipeline(), in the block `if mask2d is not None:`
- [x] zemosaic_gui.py (Tk GUI): add a new setting in the Alt-Az Advanced row
- [x] locales: update translations for EN/FR (json files)
NO other refactors, NO algorithm rewrite.

## New config key
- [x] `altaz_alpha_soft_threshold` (float in [0..1])
Meaning:
- Pixels with mask2d <= threshold are treated as fully transparent (NaN or 0 depending on altaz_nanize).
Recommended typical value: 0.85.
Backward compatibility: if key missing, default keeps old behavior (1e-3).

## Implementation details

### 1) Worker: zemosaic_worker.py
In `_apply_master_tile_quality_pipeline()`:
- [x] read cfg value:
    az_alpha_soft = float(cfg.get("altaz_alpha_soft_threshold", 1e-3))
- [x] sanitize:
    - if NaN/inf -> fallback 1e-3
    - clamp into [0.0, 1.0]
- [x] replace:
    hard_threshold = 1e-3
  with:
    hard_threshold = az_alpha_soft
- [x] keep rest identical
- [x] extend existing log line to include the threshold value (keep existing message, just append `threshold=%g`).

### 2) GUI: zemosaic_gui.py (Advanced panel)
Add:
- [x] self.config.setdefault("altaz_alpha_soft_threshold", 1e-3)  (to preserve current default behavior)
- [x] `self.altaz_alpha_soft_threshold_var = tk.DoubleVar(... value=self.config.get("altaz_alpha_soft_threshold", 1e-3))`

In the Alt-Az advanced row (where margin/decay/nanize are):
- [x] add a label + spinbox after "Alt-Az decay" and before/near "Alt-Az → NaN"
- [x] Spinbox range: 0.0 .. 1.0 step 0.01, width ~6
- [x] store widgets:
    self.translatable_widgets["altaz_alpha_soft_threshold_label"] = that label
- [x] include the spinbox in `_altaz_inputs` so it is enabled/disabled with Alt-Az toggle

When building the run config dict (the place where altaz_cleanup_enabled/margin/decay/nanize are injected):
- [x] include:
    "altaz_alpha_soft_threshold": float(self.altaz_alpha_soft_threshold_var.get())

- [x] Also ensure the value is persisted in self.config when launching / saving config.

### 3) Translations
Add translation keys for EN + FR:
- [x] `altaz_alpha_soft_threshold_label`
Suggested text:
- EN: "Alt-Az alpha cutoff"
- FR: "Seuil alpha Alt-Az"

If you have tooltips system already, optionally add:
- `altaz_alpha_soft_threshold_tooltip`
EN: "Pixels with alpha below this value are treated as transparent (NaN/0). Try 0.85."
FR: "Pixels avec alpha inférieur à cette valeur sont considérés transparents (NaN/0). Essayez 0.85."
(Only if tooltips are already used in this panel; otherwise skip to stay surgical.)

## Files to edit
- [x] zemosaic_worker.py
- [x] zemosaic_gui.py
- [x] locales/en.json (or equivalent)
- [x] locales/fr.json (or equivalent)

## Test plan (manual)
1) Run a small mosaic with Alt-Az cleanup enabled + mask2d path active.
2) Baseline: threshold=1e-3 -> observe tinted feather zones may remain.
3) Set threshold to 0.85 in Advanced.
4) Re-run:
   - logs show MT_PIPELINE altaz_cleanup applied ... threshold=0.85
   - tinted border zones should become transparent (NaN/0) and stop polluting the mosaic.

## Acceptance criteria
- [x] New control appears in Advanced panel, translated in EN/FR
- [x] Default behavior unchanged when key missing (1e-3)
- [x] Worker uses the configured threshold for mask2d -> nanize/zeroize
- [x] No other behavior changes / refactors
