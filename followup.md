# Follow-up — Validation instrumentation + log level propagation (Qt)

## Checklist exécution
- [ ] Lancer un run Classic en GUI Qt avec Logging level = INFO
  - Attendu: pas de lignes `[DBG_RGB]`
  - Attendu: une ligne `[LOGCFG] effective_level=INFO ...`

- [ ] Lancer un run Classic en GUI Qt avec Logging level = DEBUG
  - Attendu: `[LOGCFG] effective_level=DEBUG source=...`
  - Attendu: checkpoints `[DBG_RGB]` présents et seulement aux points prévus:
    - P3_PRE_STACK_CORE / P3_POST_STACK_CORE
    - P3_PRE_POSTSTACK_EQ / P3_POST_POSTSTACK_EQ (si appelé)
    - P4_PRE_MOSAIC_FUSE / P4_POST_MOSAIC_FUSE
    - P5_PRE_GLOBAL_POST / P5_POST_GLOBAL_POST
    - P6_PRE_EXPORT (et P7_POST_EXPORT si implémenté)
  - Attendu: logs compacts, 1 ligne par checkpoint (pas de spam par tuile)

## Grep patterns utiles
- `\[LOGCFG\]`
- `\[DBG_RGB\] P3_`
- `\[DBG_RGB\] P4_`
- `\[DBG_RGB\] P5_`
- `ratio_G_R=`
- `cov_weighted_mean=`

## Interprétation rapide (comment lire)
- Si `ratio_G_R` et/ou `ratio_G_B` est ~1.0 en P3_* puis dérive en P4_*:
  -> problème introduit au moment fusion/reprojection/coverage.
- Si P4 est stable mais dérive en P5_POST_*:
  -> problème introduit par equalization/scaling/global normalization.
- Si P5 stable mais dérive en P6:
  -> conversion/clamp/export.

## Comparaison Classic vs SDS
- [ ] Lancer SDS en DEBUG avec mêmes logs activés
- [ ] Comparer la première phase où `ratio_G_R` explose en Classic mais pas en SDS

## Non-régression
- [ ] Vérifier que les fichiers de sortie (FITS) sont bien produits comme avant.
- [ ] Vérifier qu’aucun changement de pipeline n’a été fait (uniquement logs + log level).
- [ ] Vérifier qu’aucun comportement “batch size=0” / “batch size>1” n’a été modifié.

## Notes dev (si problème)
- Si DEBUG ne sort pas malgré le dropdown Qt:
  - vérifier que le worker est en process séparé -> env var doit être propagée
  - vérifier que le handler/formatter du logger ne filtre pas au-dessus de DEBUG
  - vérifier que `logger.propagate` et `root` n’écrasent pas le niveau
