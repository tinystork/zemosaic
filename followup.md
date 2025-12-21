# followup.md
Patch checklist

A) zemosaic_config.py
- [x] Add DEFAULT_CONFIG["astap_drizzled_fallback_enabled"]=False

B) zemosaic_gui_qt.py
- [x] In _build_solver_tab (ASTAP configuration group), add:
  self._register_checkbox("astap_drizzled_fallback_enabled", astap_layout, self._tr("qt_field_astap_drizzled_fallback", "..."))
- [x] In _build_solver_settings_dict():
  - read enabled = bool(self.config.get("astap_drizzled_fallback_enabled", False))
  - include "astap_drizzled_fallback_enabled": enabled in returned dict
  - if SolverSettings exists: settings.astap_drizzled_fallback_enabled = enabled

C) solver_settings.py
- [x] Add field in SolverSettings dataclass:
  astap_drizzled_fallback_enabled: bool = False

D) zemosaic_astrometry.py
- [x] Update solve_with_astap signature to accept astap_drizzled_fallback_enabled: bool = False
- [x] When building cmd_list_astap, store a flag used_pxscale = True when "-pxscale" was added.
- [x] Implement helper inside solve_with_astap:
  def _make_fov0_cmd(cmd):
      # remove -pxscale pair, remove -fov pair if any, append -fov 0
- [x] In the ASTAP execution loop:
  - if rc_astap==1 and astap_drizzled_fallback_enabled and used_pxscale and not tried_fallback:
      tried_fallback=True
      cmd_list_astap = fallback_cmd
      log + progress_callback message
      continue to next attempt
  - if tried_fallback and rc_astap!=0: break / exit loop normally

E) locales
- [x] Add translation key qt_field_astap_drizzled_fallback:
  EN: "Stacked/drizzled datasets: retry with auto FOV on failure"
  FR: "Données empilées/drizzlées : retenter en FOV auto si échec"

F) Propagation
- [x] Pass astap_drizzled_fallback_enabled into solve_with_astap from zemosaic_worker.py
- [x] Pass astap_drizzled_fallback_enabled into solve_with_astap from zemosaic_filter_gui.py
- [x] Pass astap_drizzled_fallback_enabled into solve_with_astap from zemosaic_filter_gui_qt.py
