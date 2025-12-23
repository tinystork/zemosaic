# Mission: Phase 5 "global_reproject" — remonter gpu_rows_per_chunk quand pas sur batterie (no-refactor)

## Contexte
En Phase 5 (Reproject & Coadd), le GPU est activé mais l'exécution reste à faible charge car le plan GPU est clampé à
`gpu_max_chunk_bytes=128MB` et surtout `gpu_rows_per_chunk` très bas (ex: ~69), entraînant beaucoup de micro-chunks
et donc un overhead (boucle Python + launches + copies). Résultat: GPU/CPU à quelques %.

La safety logic détecte Windows/laptop/hybride et force un mode safe. On veut garder le *budget bytes* (anti-freeze),
mais réduire le nombre de chunks en augmentant `gpu_rows_per_chunk` **uniquement** pour l'opération
`operation="global_reproject"` quand l'alimentation secteur est présente (pas sur batterie).

## Objectif
Patch minimal, localisé à Phase 5 reproject:
- Après l'appel à `apply_gpu_safety_to_parallel_plan(..., operation="global_reproject")`,
  si `safe_mode==1` ET `on_battery == False` (ou `power_plugged == True`), recalculer un `gpu_rows_per_chunk`
  plus grand, borné prudemment (ex: <= 256) et >= valeur actuelle.
- Ne pas toucher aux autres phases.
- Ne pas changer le comportement "batch size = 0" vs "> 1".

## Fichiers probables
- `zemosaic_worker.py` (ou fichier équivalent orchestration Phase 5)
- éventuellement `solver_settings.py` si le plan est défini là, mais privilégier un patch dans le worker au moment
  où le plan est finalisé (Phase 5).
- Ne pas modifier `zemosaic_gpu_safety.py` (sauf si impossible), objectif: patch le plus local possible.

## Spécification de calcul (simple & conservatrice)
Estimer un coût "bytes par ligne" pour la reprojection GPU:
- Hypothèse conservatrice: on manipule au moins 2 buffers float32 (accumulateur + poids),
  et on itère sur N tiles.
- bytes_per_row ≈ out_w * 4 (float32) * buffers_per_tile_effective * n_tiles_scale
  On n'a pas besoin d'être exact, seulement éviter des valeurs trop grandes.

Proposition robuste (sans connaître tous les détails internes):
- `bytes_per_row = max(1, out_w * 4 * max(2, buffers))`
- puis diviser le budget par `max(1, n_tiles)` pour rester conservateur côté mémoire "par tile"
  (même si l'impl GPU ne garde pas tout simultanément).
- `rows_budget = gpu_max_chunk_bytes // (bytes_per_row * max(1, n_tiles))`
- `new_rows = clamp(rows_budget, min_rows=current_rows, max_rows=256)`
- Ajouter un plancher raisonnable (ex: 96) si ça ne dépasse pas le current.

Si des infos plus précises existent déjà (ex: taille réelle des buffers ou estimateur interne),
les utiliser à la place (mais sans refactor).

## Étapes
- [x] Localiser dans le code le point Phase 5 où:
  - le plan est créé
  - `apply_gpu_safety_to_parallel_plan(... operation="global_reproject")` est appelé
- [x] Ajouter juste après ce call un petit ajustement conditionnel:
  - uniquement si `operation == "global_reproject"`
  - uniquement si `safe_mode == 1`
  - uniquement si `on_battery == False` (ou `power_plugged == True`)
- [x] Recalculer `gpu_rows_per_chunk` selon une estimation simple basée sur:
  - `plan.gpu_max_chunk_bytes` (ou param correspondant)
  - `out_w` (largeur de sortie) accessible depuis le contexte Phase 5
  - `n_tiles` (nombre de master tiles en input) accessible depuis la phase
- [x] Ajouter un log INFO clair:
  - "Phase5: reproject rows_per_chunk bumped from X to Y (not on battery), max_chunk_bytes=..."
- [x] S'assurer que si des champs manquent (out_w/n_tiles), on n'échoue pas:
  - fallback: ne rien changer

## Critères d'acceptation
- Sur un run secteur (pas sur batterie), `gpu_rows_per_chunk` augmente (ex: 69 → ~200),
  nombre de chunks réduit sensiblement, meilleure occupation GPU, sans dépassement mémoire.
- Sur batterie, comportement inchangé.
- Aucun changement de résultat scientifique attendu (juste la granularité).
- Aucune régression sur les autres phases.

## Tests (léger)
- [x] Ajouter un mini test unitaire si la suite existe:
  - Simuler un plan avec gpu_max_chunk_bytes=128MB, out_w=2282, n_tiles=30, current_rows=69
  - Vérifier que new_rows > current_rows et <= 256 quand on_battery=False
  - Vérifier new_rows == current_rows quand on_battery=True
Si la repo n'a pas de tests, au minimum ajouter un "self-check" dans log (pas de test framework).
