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
- zemosaic_worker.py
- zemosaic_filter_gui.py
- zemosaic_filter_gui_qt.py
- locales/en.json and locales/fr.json (or wherever Qt translation JSON lives)

Tasks:
1) [x] Add new config key with default False:
   - [x] DEFAULT_CONFIG["astap_drizzled_fallback_enabled"] = False

2) [x] Qt UI:
   - [x] In the Solver tab under ASTAP configuration, add a checkbox bound to config key "astap_drizzled_fallback_enabled".
   - [x] Label uses localization key "qt_field_astap_drizzled_fallback".

3) [x] Propagate to worker:
   - [x] In _build_solver_settings_dict(), include "astap_drizzled_fallback_enabled" in the returned payload dict.
   - [x] If SolverSettings is used, add attribute and include it in asdict payload.

4) [x] solver_settings.py:
   - [x] Add dataclass field: astap_drizzled_fallback_enabled: bool = False

5) [x] ASTAP fallback:
   - [x] In zemosaic_astrometry.solve_with_astap(), add parameter astap_drizzled_fallback_enabled: bool = False
   - [x] Track whether initial cmd used -pxscale (header-derived).
   - [x] If rc_astap == 1 and fallback enabled and pxscale was used and fallback not yet tried:
       - [x] Build a fallback cmd list: remove "-pxscale <value>" pair, remove any existing "-fov <value>" pair, then append ["-fov","0"]
       - [x] Log a WARN/INFO message and show the exact fallback command in DEBUG
       - [x] Retry once with this fallback cmd
   - [x] Ensure fallback is attempted at most once.

6) [x] Translations:
   - [x] Add "qt_field_astap_drizzled_fallback" to en/fr JSON.

7) [x] Propagate flag into ASTAP call sites:
   - [x] zemosaic_worker.py passes astap_drizzled_fallback_enabled to solve_with_astap.
   - [x] zemosaic_filter_gui.py passes astap_drizzled_fallback_enabled to solve_with_astap.
   - [x] zemosaic_filter_gui_qt.py passes astap_drizzled_fallback_enabled to solve_with_astap.

Manual verification:
- With checkbox OFF: behavior identical (no fallback).
- With checkbox ON: logs show second attempt with "-fov 0" and no "-pxscale" when rc=1.
