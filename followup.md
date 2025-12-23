# Follow-up: Implémentation & vérifications (Phase 5 rows_per_chunk bump)

## 1) Localisation exacte
- Ouvrir `zemosaic_worker.py` et trouver la section Phase 5 "Reproject & Coadd".
- Repérer l'appel à `apply_gpu_safety_to_parallel_plan(..., operation="global_reproject")`.
- Repérer où sont disponibles:
  - `out_w` (ou une shape/width équivalente de l'image/canvas de sortie)
  - `n_tiles` (len(master_tiles) / tiles list / inputs)
  - flags d'alimentation (`on_battery`, `power_plugged`) si déjà loggés / détectés.

## 2) Patch minimal (juste après le safety)
Ajouter un bloc du style (adapter aux noms réels, sans refactor):

- Conditions:
  - `operation == "global_reproject"`
  - `getattr(safety, "safe_mode", 0) == 1` OU `plan.safe_mode == 1` (selon où c'est stocké)
  - `on_battery is False` (ou `power_plugged is True`)
- Valeurs:
  - `current = plan.gpu_rows_per_chunk` (si absent: ne rien faire)
  - `max_bytes = plan.gpu_max_chunk_bytes` (sinon: ne rien faire)
  - `out_w = ...`
  - `n_tiles = ...`

- Estimation conservative:
  - `buffers = 2` (accum + weight)
  - `bytes_per_row = max(1, out_w * 4 * buffers)`
  - `den = max(1, n_tiles)`
  - `rows_budget = max_bytes // (bytes_per_row * den)`
  - `candidate = int(rows_budget)`
  - `new_rows = min(256, max(current, candidate))`
  - Optionnel: `new_rows = max(new_rows, min(96, 256))` uniquement si `new_rows > current` sinon rien
  - Si `new_rows > current`: assigner et logguer.

Important: si `candidate` est absurde (0 ou < current), ne rien changer.

## 3) Logging
Ajouter une ligne INFO (même logger que le worker):
- Avant/après + contexte:
  - safe_mode, on_battery/power_plugged, max_chunk_bytes, out_w, n_tiles

Ex:
"Phase5 GPU: bump rows_per_chunk 69 -> 224 (plugged), max_chunk=128MB, out_w=2282, n_tiles=30"

## 4) Robustesse
- Aucun import nouveau lourd.
- Pas d'exception si attributs manquent:
  - utiliser `getattr(...)` + early return.
- Ne pas toucher au multi-threading.
- Ne pas changer la logique batch size (0 vs >1).

## 5) Tests rapides
### Option A: pytest (si présent)
Créer/compléter un test simple qui appelle la fonction/helper si tu en crées une mini locale,
ou bien tester via un petit "plan" factice (dataclass/dict) si c'est déjà le style du repo.

Cas:
- plugged: augmente et <=256
- on_battery: inchangé

### Option B: smoke run (si pas de tests)
- Lancer un run court (ex: 10–20 tuiles) en secteur.
- Vérifier dans le log que `rows_per_chunk` est bump.
- Observer que la phase reproject fait moins d'itérations/chunks.

## 6) Résultat attendu
- Moins de micro-chunks → moins d'overhead.
- GPU davantage sollicité (sans chercher 100%, mais au moins une montée perceptible).
- Pas de freeze, car `gpu_max_chunk_bytes` reste identique (128MB).
