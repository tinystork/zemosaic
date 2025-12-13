# Follow-up — Debug ciblé dominante verte (Classic)

## Ce qui a été fait
- [x] GUI Qt : ajout/validation des options “Logging level” (Info/Debug) et propagation vers paramètres worker.
- [x] Worker : lecture du paramètre `worker_logging_level` et application réelle du niveau (logger + handlers).
- [ ] Worker : logs DEBUG ciblés aux frontières P3/P4/P5 + export.
- [ ] Ajout/extension utilitaire `_dbg_rgb_stats` pour stats global/valid-only/weighted.

## Points précis à vérifier dans le code

### A) Propagation du niveau de log (GUI → worker)
- [x] Le choix UI “Debug” donne bien `worker_logging_level="DEBUG"` dans les params.
- [x] Le worker imprime : `Worker logging level set to DEBUG`.
- [ ] Le fichier `zemosaic_worker.log` contient des lignes `- DEBUG -`.

### B) Logs Phase 3 (baseline)
- [ ] Labels présents :
  - `P3_pre_stack_core`
  - `P3_post_stack_core` (si possible)
  - `P3_post_poststack_rgb_eq` (si applicable)
- [ ] Chacun loggue min/mean/median + ratios G/R G/B.

### C) Logs Phase 4 (zone critique #1)
- [ ] Labels présents :
  - `P4_pre_fusion_mosaic`
  - `P4_post_fusion_mosaic`
- [ ] Stats “valid-only” utilisent `coverage > 0` (ou équivalent).
- [ ] Weighted mean par coverage est calculée et logguée.

### D) Logs Phase 5 (zone critique #2)
- [ ] Labels présents :
  - `P5_pre_postprocess`
  - `P5_post_<etape>` pour chaque étape suspecte
- [ ] Si égalisation RGB : log target + gains par canal.

### E) Phase 6–7 export
- [ ] Labels présents :
  - `P6_pre_export`
  - `P6_post_clamp`
  - `P7_post_png`
- [ ] dtype + clamp min/max + mention stretch auto.

## Comment exploiter le résultat
1) Lancer Classic en Debug et relever :
- `ratio_G_R` / `ratio_G_B` à `P3_post_poststack_rgb_eq`
- puis à `P4_post_fusion_mosaic`
- puis à `P5_post_*`

2) Le premier endroit où les ratios explosent = endroit du bug.

## Non-objectifs (à respecter)
- [ ] Pas de changement algorithmique (uniquement logs + propagation log level).
- [ ] Ne pas modifier SDS / Grid logic.
- [ ] Ne pas ajouter de nouveaux réglages utilisateur (sauf Debug dans combo si absent).

## Journal des changements
- zemosaic_gui_qt.py : propagation explicite de `logging_level`/`worker_logging_level` vers le worker et synchronisation de l’ENV.
- zemosaic_worker.py : harmonisation de la configuration du logger avec message `[LOGCFG]` et niveau appliqué au worker.

