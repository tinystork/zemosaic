# Step-by-step checklist

## 1) Config: add default + persistence
- In zemosaic_config.py (or wherever default settings live):
  - Add `altaz_nanize_threshold` default = 0.001
  - Ensure it is saved/loaded like other float settings.

## 2) GUI: Advanced tab
- In zemosaic_gui_qt.py:
  - Add a labeled QDoubleSpinBox in Advanced tab:
    - label: "ALT-AZ nanize threshold"
    - tooltip: explain behavior
    - min=0.0 max=1.0 step=0.01 decimals=3
    - set value from config on load
    - write back to config on apply/save
- Ensure it appears with other Lecropper/ALT-AZ options.

## 3) Translations
- Add EN/FR strings for:
  - label
  - tooltip
- Keep existing translation pattern (same dict structure / keys).

Suggested FR:
- Label: "Seuil de NaNisation ALT-AZ"
- Tooltip: "Les pixels dont l’opacité du masque ALT-AZ est <= à ce seuil sont considérés invalides (NaN/0). Augmentez pour supprimer davantage de rampes de bord."

## 4) Worker: use the threshold
- In zemosaic_worker.py inside the lecropper pipeline:
  - Locate the part where mask2d is used to convert pixels to NaN/zero with a hardcoded cutoff (currently 1e-3).
  - Replace with:
    - `thr = float(cfg.altaz_nanize_threshold)` (fallback 1e-3 if missing)
    - clamp thr into [0.0, 1.0]
  - Use `mask2d <= thr` as the invalid region.
  - Keep everything else identical.

- Add a log (info/debug) once per master-tile build:
  - `logger.info("lecropper: altaz_nanize_threshold=%.3f", thr)`

## 5) Guard rails
- If cfg value is NaN or None:
  - fallback to 0.001
- Do not alter lecropper.py behavior or API.
- Do not change other thresholds (hard_threshold, decay_ratio, etc.)

## 6) Manual validation
- Run with threshold = 0.001 -> verify no regression.
- Run with threshold = 0.20 -> verify the big magenta/green ramps are removed in master tiles (circled zones become invalid/transparent).

## 7) Deliverables
- Provide a git diff
- Mention files changed
- Mention the new config key and default
- Include screenshots/log snippet showing threshold value
