# followup.md

## À faire (checklist)
- [x] Dans `zemosaic_utils.py`, ajouter les imports nécessaires :
      - modifier `from concurrent.futures import ThreadPoolExecutor, as_completed`
      - en `from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED`
      (conserver `as_completed` si utilisé ailleurs dans le fichier)
- [x] Localiser `compute_intertile_affine_calibration(...)` puis le bloc `if use_parallel:`
- [x] Remplacer la boucle `for future in as_completed(future_map):` par une boucle `wait(...)` avec timeout
- [x] Introduire `done_count`, `last_completed_idx`, `last_progress_ts`, `last_heartbeat_ts`
- [x] Remplacer la progression `idx/total_pairs` par `done_count/total_pairs`
- [x] Ajouter un heartbeat (log) si aucune future ne termine pendant ~10s
- [x] Protéger `future.result()` par try/except et logger l’erreur si besoin

## Détails log heartbeat (recommandé)
Format suggéré (via `_log_intertile`, level INFO ou WARN) :
`Heartbeat: done=123/4342 pending=4219 last_completed_idx=348 sample_pending_idx=[12, 907, 1111, 2044, 3999]`

- `sample_pending_idx` : max 5
- Heartbeat pas plus fréquent que 10s

## Tests manuels
### Test 1 — Run normal (pas de stall)
- Lancer un dataset où intertile a > 100 paires
- Vérifier :
  - `phase5_intertile_pairs` progresse de manière monotone jusqu’à total_pairs
  - pas de heartbeat (ou très rare)

### Test 2 — Repro “stall”
- Lancer le dataset qui fige “vers 348/4342”
- Attendu :
  - la progress bar affiche désormais “done_count/4342” (monotone)
  - au bout de ~10s sans complétion : apparition heartbeat
  - heartbeat donne `pending` et `sample_pending_idx` pour savoir si ça coince vraiment (hang) vs crash

## Notes
- Cette mission est volontairement “instrumentation only”.
- Ne pas modifier `_process_overlap_pair` ni la logique d’overlap/reproject.
- Ne pas ajouter de nouveaux paramètres GUI.

