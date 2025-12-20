# Step-by-step checklist

## 1) Config: add default + persistence
- [x] In zemosaic_config.py (or wherever default settings live):
  - [x] Add `altaz_nanize_threshold` default = 0.001
  - [x] Ensure it is saved/loaded like other float settings.

## 2) GUI: Advanced tab
- [x] In zemosaic_gui_qt.py:
  - [x] Add a labeled QDoubleSpinBox in Advanced tab:
    - [x] label: "ALT-AZ nanize threshold"
    - [x] tooltip: explain behavior
    - [x] min=0.0 max=1.0 step=0.01 decimals=3
    - [x] set value from config on load
    - [x] write back to config on apply/save
- [x] Ensure it appears with other Lecropper/ALT-AZ options.

## 3) Translations
- [x] Add EN/FR strings for:
  - [x] label
  - [x] tooltip
- [x] Keep existing translation pattern (same dict structure / keys).

Suggested FR:
- Label: "Seuil de NaNisation ALT-AZ"
- Tooltip: "Les pixels dont l’opacité du masque ALT-AZ est <= à ce seuil sont considérés invalides (NaN/0). Augmentez pour supprimer davantage de rampes de bord."

## 4) Worker: use the threshold
- [x] In zemosaic_worker.py inside the lecropper pipeline:
  - [x] Locate the part where mask2d is used to convert pixels to NaN/zero with a hardcoded cutoff (currently 1e-3).
  - [x] Replace with:
    - [x] `thr = float(cfg.altaz_nanize_threshold)` (fallback 1e-3 if missing)
    - [x] clamp thr into [0.0, 1.0]
  - [x] Use `mask2d <= thr` as the invalid region.
  - [x] Keep everything else identical.

- [x] Add a log (info/debug) once per master-tile build:
  - [x] `logger.info("lecropper: altaz_nanize_threshold=%.3f", thr)`

## 5) Guard rails
- [x] If cfg value is NaN or None:
  - [x] fallback to 0.001
- [x] Do not alter lecropper.py behavior or API.
- [x] Do not change other thresholds (hard_threshold, decay_ratio, etc.)

## 6) Manual validation
- [ ] Run with threshold = 0.001 -> verify no regression.
- [ ] Run with threshold = 0.20 -> verify the big magenta/green ramps are removed in master tiles (circled zones become invalid/transparent).

## 7) Deliverables
- [x] Provide a git diff
- [x] Mention files changed
- [x] Mention the new config key and default
- [x] Include screenshots/log snippet showing threshold value
