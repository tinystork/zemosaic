# Follow-up checklist

## Code changes verification
- [x] zemosaic_worker.py: threshold is now read from cfg key `altaz_alpha_soft_threshold`
- [x] value is sanitized (finite + clamped [0..1]) with fallback 1e-3
- [x] mask_zero uses `<= hard_threshold` with the new value (no other logic touched)
- [x] log line includes threshold value

- [x] zemosaic_gui.py: config default added (setdefault altaz_alpha_soft_threshold = 1e-3)
- [x] new tk.DoubleVar created and wired
- [x] new label + spinbox added to Alt-Az advanced row
- [x] spinbox included in `_altaz_inputs` so it gets enabled/disabled properly
- [x] run config includes `altaz_alpha_soft_threshold`

## Translations
- [x] Add `altaz_alpha_soft_threshold_label` in EN
- [x] Add `altaz_alpha_soft_threshold_label` in FR
- [ ] (Optional) tooltip key only if tooltips exist in this panel

## Manual test run
- [ ] Run with threshold=1e-3 => output matches previous behavior
- [ ] Run with threshold=0.85 => feather/tinted zones are mostly removed (transparent)
- [ ] Confirm logs include threshold value
- [ ] Confirm config persists the value (reopen GUI, value is still there)

## Notes
Recommended user value to try first: 0.85 (aggressive cleanup). If it removes too much signal, back off to ~0.5–0.7.
