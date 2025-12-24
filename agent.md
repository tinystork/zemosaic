# agent.md

## Mission
Instrumenter la phase **Phase5 Intertile Pairs** pour diagnostiquer les “plantages silencieux / freezes” sans lancer un gros refactor.

Objectifs précis (patch minimal) :
1) Corriger la progression en mode parallèle : actuellement on envoie `idx` (ID de paire renvoyé par `as_completed`), ce qui n’est **pas** un compteur monotone. Remplacer par un `done_count` qui s’incrémente à chaque future terminée.
2) Ajouter un **heartbeat** : si aucune future ne termine pendant un certain délai (ex. 10s), logger l’état (done/pending) + un petit échantillon des indices de paires encore en cours.

## Contraintes
- Patch **minimal**, **pas de refactor**, pas de changement d’algo intertile.
- Ne modifier que **zemosaic_utils.py** (sauf nécessité absolue).
- Ne pas ajouter de nouvelle option GUI.
- Ne pas toucher aux comportements existants ailleurs (notamment tout ce qui concerne batch size / stacking / GPU).
- Logs : rester raisonnable (heartbeat toutes les ~10s max si stall).

## Fichiers
- `zemosaic_utils.py`
  - fonction : `compute_intertile_affine_calibration(...)`
  - section : bloc `if use_parallel:` (ThreadPoolExecutor + futures)

## Analyse (problème actuel)
Le code fait :
- `future_map = { executor.submit(...): idx }`
- `for future in as_completed(future_map):`
- `idx = future_map[future]`
- `progress_callback("phase5_intertile_pairs", idx, total_pairs)`

Comme `as_completed()` renvoie les futures dans un ordre arbitraire, `idx` n’est **pas** “nombre de paires traitées”. Le GUI peut rester “bloqué” à une valeur (ex. 348/4342) alors que ce n’est que le dernier `idx` terminé, pas un compteur.

## Implémentation demandée

### 1) Progress monotone
Dans le bloc parallèle, introduire :
- `done_count = 0`
- à chaque future terminée : `done_count += 1`
- remplacer :
  - `progress_callback("phase5_intertile_pairs", int(idx), int(total_pairs))`
  - par :
    - `progress_callback("phase5_intertile_pairs", done_count, total_pairs)`
- et la seconde ligne (toutes les 5) :
  - `progress_callback("phase5_intertile", done_count, total_pairs)`

Optionnel (utile) : conserver `idx` seulement pour debug (ex. `last_completed_idx = idx`) mais ne pas l’utiliser comme progress principal.

### 2) Heartbeat sur stall (timeout)
Remplacer la boucle `as_completed` par une boucle basée sur `concurrent.futures.wait()` :
- `pending = set(future_map.keys())`
- `done_count = 0`
- `last_progress_ts = time.time()`
- `last_heartbeat_ts = time.time()`

Boucle :
- `done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)`
- Si `done` non vide :
  - pour chaque future :
    - récupérer `idx = future_map[future]`
    - `pairs_local, connectivity_entry = future.result()` avec try/except
    - appliquer le même traitement qu’actuellement
    - `done_count += 1`
    - envoyer progress monotone
    - mettre à jour `last_progress_ts`
- Si `done` vide (timeout) :
  - si `time.time() - last_progress_ts >= 10.0` ET `time.time() - last_heartbeat_ts >= 10.0` :
    - logger via `_log_intertile` un heartbeat, ex :
      - `Heartbeat: done={done_count}/{total_pairs} pending={len(pending)} last_completed_idx={last_completed_idx} sample_pending_idx=[...]`
    - `sample_pending_idx` : prendre 5 futures de `pending` et mapper via `future_map[future]`
    - mettre `last_heartbeat_ts = now`

### 3) Gestion d’erreur future.result()
Même si `_process_overlap_pair` attrape beaucoup, protéger `future.result()` :
- si exception : `_log_intertile(f"Pair future failed idx={idx}: {e}", level="ERROR")` puis continuer.

## Critères d’acceptation
- En mode parallèle, la progress bar “pairs” doit être **monotone** de 0 → total_pairs.
- En cas de stall, on doit voir apparaître des logs heartbeat au bout d’environ 10s (sans spam).
- Aucun changement de résultat attendu sur un run “normal” (hors instrumentation).

## Livrables
- Patch dans `zemosaic_utils.py` uniquement
- Pas de changement UI
- Commentaires courts et clairs près de la nouvelle boucle (pourquoi `wait` et `done_count`)

