# ZeMosaic — DEBUG instrumentation pipeline + propagation logging level (Qt GUI)

## Mission
1) Ajouter une instrumentation DEBUG **ultra ciblée** (peu de logs mais très informatifs) pour isoler l’apparition d’un **green cast** dans le pipeline classic:
- Phase 3/3.x: baseline “tiles OK”
- Phase 4/4.x: mosaïque (fusion + coverage)
- Phase 5: post-processing global (RGB equalization/scaling)
- Phase 6–7: export/clamp/conversions

2) Vérifier et garantir que le **Logging level** choisi dans le **GUI Qt** est bien propagé au worker (y compris quand le worker tourne dans un process séparé).

⚠️ Contraintes strictes
- NE PAS modifier la logique scientifique/algorithmes (stacking, reprojection, blending, equalization, normalization).
- NE PAS modifier Grid/SDS/classic sauf ajout de logs et propagation du niveau de log.
- NE PAS changer le comportement “batch size = 0” et “batch size > 1”.
- Logs seulement: instrumentation passive.
- Aucun refactor large, pas de renommage d’API publique.

## Cible de sortie (ce que l’on veut lire dans le log)
On veut des lignes DEBUG stables, greppable, du style:

[DBG_RGB] P3_OUT label=... shape=(H,W,3) dtype=float32 valid=[...] mean=[R,G,B] median=[R,G,B] min=[R,G,B] ratio_G_R=... ratio_G_B=...
[DBG_RGB] P4_OUT ... valid_cov=... cov_weighted_mean=[R,G,B] ...
[DBG_RGB] P5_POST_EQ ... gains=[gR,gG,gB] targets=[tR,tG,tB] ...

## Implémentation — approche
### A) Ajouter un helper unique de stats RGB (dans zemosaic_worker.py)
Créer une fonction interne, par ex:
- `_dbg_rgb_stats(label: str, rgb: np.ndarray | None, *, coverage: np.ndarray | None = None, alpha: np.ndarray | None = None, logger: logging.Logger) -> None`

Fonctionnement:
- Si `rgb is None` ou vide: return
- S’assurer HWC (H,W,3) float32 (conversion best-effort sans changer les données, juste vue).
- Construire un masque “valid”:
  - base = isfinite(rgb[...,c]) par canal
  - si `alpha` existe (uint8 ou float): appliquer alpha>0 (ou >ALPHA_OPACITY_THRESHOLD si dispo)
  - si `coverage` existe: appliquer coverage>0
- Calculer pour chaque canal sur `valid`:
  - min / mean / median
  - fraction de pixels valides
- Calculer ratios:
  - ratio_G_R = median(G)/max(median(R), eps)
  - ratio_G_B = median(G)/max(median(B), eps)
- Si coverage fourni: calculer **moyenne pondérée** par coverage (coverage normalisée, ignore coverage==0):
  - cov_weighted_mean RGB (sur pixels coverage>0 et valides)
- Loguer UNE seule ligne par label (compacte), au niveau `logger.debug`.

⚠️ Important:
- Ne pas déclencher d’énormes dumps: pas d’histogrammes, pas d’arrays en texte, pas de boucles sur tuiles multiples non nécessaires.

### B) Points de contrôle (checkpoints) — seulement ceux-ci
Ajouter des appels à `_dbg_rgb_stats(...)` uniquement aux checkpoints suivants:

#### Phase 3 / 3.x (baseline)
- `P3_PRE_STACK_CORE` : juste avant l’appel à `stack_core(...)` (ou équivalent) sur une tile/master tile.
- `P3_POST_STACK_CORE` : juste après `stack_core(...)`.
- `P3_PRE_POSTSTACK_EQ` : juste avant `_poststack_rgb_equalization(...)` (si appelé).
- `P3_POST_POSTSTACK_EQ` : juste après `_poststack_rgb_equalization(...)` (si appelé).

NOTE: si Phase 3 traite N tuiles, limiter l’instrumentation à:
- la première tuile (ou une tuile de référence),
- OU un échantillon faible (ex: tile_idx in {0, mid, last}),
pour éviter de spammer.

#### Phase 4 / 4.x (fusion mosaïque)
- `P4_PRE_MOSAIC_FUSE` : dès qu’on a la mosaïque “en construction” ou juste avant reprojection/accumulation finale.
- `P4_POST_MOSAIC_FUSE` : immédiatement après fusion/reprojection/accumulation, avec `coverage` (si dispo).

Inclure `coverage` dans `_dbg_rgb_stats` à ces checkpoints.

#### Phase 5 (post-processing global)
- `P5_PRE_GLOBAL_POST` : début phase 5, avant tout equalize/scale/global normalization.
- `P5_POST_GLOBAL_POST` : fin phase 5, après equalize/scale.
- Si `_apply_final_mosaic_rgb_equalization(...)` est appelé:
  - loguer aussi les paramètres/retours si disponibles: targets, gains/factors appliqués.
  - Ajouter dans le log une clé `eq=1` et `gains=[...]` quand accessible.

#### Phase 6–7 (export/clamp)
- `P6_PRE_EXPORT` : juste avant conversion dtype / clamp / write
- `P7_POST_EXPORT` : juste après écriture (optionnel), loguer dtype final et min/max par canal si accessible sans recharger le fichier.

### C) Propagation du logging level depuis le GUI Qt
#### Objectif
Le dropdown “Logging level” du GUI Qt doit:
- définir le niveau de log effectif du logger `ZeMosaicWorker` (et enfants)
- s’appliquer aussi si le worker tourne dans un process séparé

#### Actions requises
1) Dans `zemosaic_gui_qt.py`:
- Identifier où la config est construite/envoyée au worker (dict config / args / env).
- S’assurer qu’une valeur de logging level (ex: `"logging_level": "DEBUG"|"INFO"|...`) est:
  - soit passée dans la config au worker,
  - soit exportée via env var ex: `ZEMOSAIC_LOG_LEVEL=DEBUG`

2) Dans `zemosaic_worker.py`:
- Au tout début du point d’entrée worker (fonction top-level type `run_hierarchical_mosaic(...)` ou process entry):
  - lire `logging_level` depuis config si présent, sinon env `ZEMOSAIC_LOG_LEVEL`, sinon conserver default.
  - appliquer `logger.setLevel(...)` sur `logging.getLogger("ZeMosaicWorker")` ET idéalement sur le root handler si nécessaire (sans casser le reste).
  - vérifier que les handlers existants ne filtrent pas au-dessus.

3) Ajouter un log INFO unique confirmant le niveau effectif:
- `[LOGCFG] effective_level=DEBUG source=qt_gui_config` (ou env/default)
Ainsi on peut prouver que le GUI Qt propage bien.

### D) Aucun impact sur performance en mode INFO
Les logs `[DBG_RGB]` ne doivent apparaître que si le niveau DEBUG est effectivement actif.

## Fichiers à modifier (attendus)
- `zemosaic_worker.py` : ajout helper `_dbg_rgb_stats` + checkpoints + application niveau log (si nécessaire)
- `zemosaic_gui_qt.py` : propagation du logging level vers worker (config/env)
(Ne pas modifier les autres fichiers sauf si strictement nécessaire et minimal.)

## Critères de réussite
- En mode INFO: pas de `[DBG_RGB]` dans le log.
- En mode DEBUG via GUI Qt: présence de `[LOGCFG] effective_level=DEBUG ...` puis checkpoints `[DBG_RGB]` aux phases ciblées.
- Aucune régression fonctionnelle: mêmes outputs, mêmes timings approximatifs, pas de crash.
