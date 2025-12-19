# Mission (surgical / no refactor)
Add a configurable threshold that controls how much of the ALT-AZ artifact feather mask gets converted to NaNs/zeros in master tiles (to remove large chromatic ramps near borders).

Currently ZeMosaic mainly uses lecropper.mask_altaz_artifacts() mask2d and then nanizes only where mask2d <= 1e-3, which keeps most of the feather ramp "valid" and can create strong magenta/green ramps.

We want a user-facing parameter (Advanced tab) to choose a higher cutoff, e.g. 0.15..0.50, to treat more of the ramp as invalid.

# Constraints
- NO REFACTOR. Minimal code changes only.
- Keep lecropper.py autonomous.
- Do not change algorithm except the threshold used for final nanize/zeroize decision.
- Default behavior must remain close to current unless the user changes the new parameter.

# Scope
Implement in:
- Qt GUI: Advanced tab (where other lecropper/altaz parameters live)
- Config wiring: store/load/persist
- Worker pipeline: use new threshold instead of hardcoded 1e-3 when applying mask2d to the tile

Likely files:
- zemosaic_gui_qt.py (Advanced tab / settings UI)
- zemosaic_config.py (config key defaults + load/save)
- zemosaic_worker.py (where _apply_lecropper_pipeline converts mask2d into NaNs/zeros)
- translations (where other Advanced strings are translated)

# New parameter
Name: "ALT-AZ nanize threshold"
Key suggestion: altaz_nanize_threshold
Type: float
Range: [0.0, 1.0]
Default: 0.001 (to match existing behavior)

Tooltip/help:
"Pixels where the ALT-AZ artifact mask opacity is <= this threshold are treated as invalid (NaN/zero). Increase to remove more border ramps."

# Implementation details (worker)
Find the code path where:
- mask_altaz_artifacts returns (masked, mask2d)
- ZeMosaic ignores 'masked' and later converts based on mask2d <= 1e-3
Replace that 1e-3 with cfg.altaz_nanize_threshold (clamped to [0,1]).
Keep existing behavior for the rest (same nan/zero handling as currently).

Add a log line once per tile (or once per master tile creation batch) like:
"lecropper: altaz_nanize_threshold=%.3f"

# GUI
Add a QDoubleSpinBox:
- decimals: 3
- step: 0.01
- min: 0.0
- max: 1.0
- default: 0.001
Put it in Advanced section near ALT-AZ / Lecropper options.

# Translations
Add labels + tooltips in EN/FR (match existing translation scheme).
Do not change unrelated strings.

# Acceptance criteria
- When altaz_nanize_threshold=0.001, output matches previous behavior.
- When altaz_nanize_threshold is increased (e.g. 0.20), the large chromatic ramps near borders become masked out (NaN/invalid) in master tiles, reducing magenta/green "veils".
- No other behavior changes; pipeline still runs.

# Quick manual test
- Run a dataset that produces the chromatic ramps.
- Compare master tiles with threshold 0.001 vs 0.20.
- Verify that the circled areas become invalid/transparent rather than colored ramps.
