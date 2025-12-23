# Mission — Phase 5 Intertile trop lente : reconnecter le multithread existant (sans ajouter de knobs)

## Contexte
Depuis l’intégration GPU safety, la Phase 5 "Intertile Pairs" est devenue anormalement lente.
Le reproject Phase 5 semble OK, mais l’étape Intertile (photometric match between tiles) rame fortement.
On veut éviter tout ajout de paramètres GUI : un mécanisme de workers existe déjà (processing_threads / auto workers),
mais l’Intertile ne semble pas l’utiliser.

## Objectif
Accélérer la Phase 5 Intertile en réutilisant le mécanisme de workers existant (processing_threads / auto),
en évitant de créer du “nouveau code pour le plaisir”.

## Fichiers ciblés
- zemosaic_worker.py
- zemosaic_utils.py (fonction compute_intertile_affine_calibration + helpers)

## Contraintes
- NO REFACTOR : patch chirurgical.
- Ne pas ajouter de nouveaux paramètres GUI.
- Ne pas toucher au comportement “batch size = 0” et “batch size > 1”.
- Conserver les résultats (différences numériques minimes tolérées, mais pas de changement visible/structurel).

## Diagnostic à faire en premier (obligatoire)
1) Ouvrir `zemosaic_utils.py` et localiser `compute_intertile_affine_calibration`.
2) Vérifier s’il existe DÉJÀ un mécanisme de parallélisation (ThreadPool/ProcessPool/parallel_utils).
   - S’il existe mais est “orphelin” (workers forcés à 1 / param jamais alimenté), le reconnecter.
   - S’il n’existe pas, en ajouter un MINIMAL (ThreadPool) autour de la boucle des paires, piloté par un param `cpu_workers`.

## Patch attendu (plan)
### A) Reconnecter le nombre de workers existant côté worker
Dans `zemosaic_worker.py`, au moment où on appelle `_compute_intertile_affine_corrections_from_sources(...)`,
dériver un `intertile_workers` à partir de la logique déjà utilisée plus bas pour le reproject :
- si `processing_threads > 0` => utiliser cette valeur
- sinon auto => `min(os.cpu_count() or 1, <borne raisonnable>)` (borne raisonnable = 8/16)  

Puis :
- ajouter un param optionnel `cpu_workers` à `_compute_intertile_affine_corrections_from_sources`
- le forwarder vers `zemosaic_utils.compute_intertile_affine_calibration(...)`
  (adapter le nom exact du param si la fonction l’a déjà sous un autre nom)

### B) Dans zemosaic_utils : utiliser cpu_workers si présent
Dans `compute_intertile_affine_calibration` :
- Si un mécanisme parallèle existe déjà : l’alimenter avec `cpu_workers` (au lieu d’un défaut à 1 / None).
- Sinon : paralléliser UNIQUEMENT la partie “par paire” (la boucle des overlap pairs) via ThreadPoolExecutor.
  *Important* : garder le `progress_callback` appelé depuis le thread principal (agréger les résultats de futures).

### C) Logging minimal pour confirmer
Ajouter un log du style :
`[Intertile] Parallel: threadpool workers=<N> pairs=<M> preview=<P>`
afin qu’on puisse valider immédiatement dans le log si on est multi-thread ou pas.

## Critères d’acceptation
- Sur un run type (ex: 27 tiles, 245 pairs, preview=1024), la durée entre :
  `[Intertile] Using: ... pairs=...`
  et la fin de l’étape Intertile baisse significativement (objectif pratique: ~x2 si CPU dispo).
- Le GUI continue d’afficher la progression (pairs done / total) sans freeze.
- Aucune régression sur Phase 5 reproject/coadd.
- Pas de nouveau réglage utilisateur.

## Notes d’implémentation (garde-fous)
- Si `cpu_workers <= 1` ou `pairs < 4` => rester en séquentiel (éviter overhead).
- Éviter ProcessPool pour l’Intertile (pickling WCS/arrays = risque + lent) : ThreadPool recommandé.
