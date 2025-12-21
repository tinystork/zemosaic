# agent.md
You are working on ZeMosaic (Qt GUI). Implement a new optional ASTAP fallback for rare stacked/drizzled external datasets.
Goal: when enabled via a checkbox in the Solver tab, if ASTAP fails with return code 1 ("No solution found") AND the first attempt used -pxscale derived from the FITS header, retry exactly once with -fov 0 and without -pxscale. Default behavior must remain unchanged when the checkbox is off.

Constraints:
- NO refactor.
- Touch only the minimal files listed.
- Keep existing logs and behavior unless fallback is enabled.

Files to edit:
- zemosaic_config.py
- zemosaic_gui_qt.py
- solver_settings.py
- zemosaic_astrometry.py
- locales/en.json and locales/fr.json (or wherever Qt translation JSON lives)

Tasks:
1) Add new config key with default False:
   - DEFAULT_CONFIG["astap_drizzled_fallback_enabled"] = False

2) Qt UI:
   - In the Solver tab under ASTAP configuration, add a checkbox bound to config key "astap_drizzled_fallback_enabled".
   - Label uses localization key "qt_field_astap_drizzled_fallback".

3) Propagate to worker:
   - In _build_solver_settings_dict(), include "astap_drizzled_fallback_enabled" in the returned payload dict.
   - If SolverSettings is used, add attribute and include it in asdict payload.

4) solver_settings.py:
   - Add dataclass field: astap_drizzled_fallback_enabled: bool = False

5) ASTAP fallback:
   - In zemosaic_astrometry.solve_with_astap(), add parameter astap_drizzled_fallback_enabled: bool = False
   - Track whether initial cmd used -pxscale (header-derived).
   - If rc_astap == 1 and fallback enabled and pxscale was used and fallback not yet tried:
       - Build a fallback cmd list: remove "-pxscale <value>" pair, remove any existing "-fov <value>" pair, then append ["-fov","0"]
       - Log a WARN/INFO message and show the exact fallback command in DEBUG
       - Retry once with this fallback cmd
   - Ensure fallback is attempted at most once.

6) Translations:
   - Add "qt_field_astap_drizzled_fallback" to en/fr JSON.

Manual verification:
- With checkbox OFF: behavior identical (no fallback).
- With checkbox ON: logs show second attempt with "-fov 0" and no "-pxscale" when rc=1.
