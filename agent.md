# Mission — Phase 3 master tiles : cache .npy protégé par refcount (suppression sûre en parallèle)

## Problème observé
Des "trous" dans la mosaïque finale correspondent à des master tiles non produites, avec logs:
- mastertile_warn_cache_file_missing
- mastertile_error_no_valid_images_from_cache
- run_warn_phase3_master_tile_creation_failed_thread
et num_master_tiles < tiles_total.

Cause probable: suppression de caches `.npy` trop tôt (mode per_tile), alors que d’autres master tiles en ont encore besoin.
Cela peut arriver même sans borrowing explicite (overlap/duplication/partage involontaire).

## Objectif
Mettre en place un mécanisme **de référence comptée** des caches `.npy` en Phase 3:
- Construire un refcount global des `path_preprocessed_cache` utilisés par toutes les master tiles planifiées
- À la fin de chaque master tile (future terminée), décrémenter le refcount pour les caches réellement utilisés par cette tile
- Supprimer (unlink) un cache **uniquement quand son refcount atteint 0**
=> suppression sûre même en exécution parallèle (ThreadPool), sans trous, sans garder tout jusqu’à la fin.

## Contraintes strictes
- [x] Changements minimaux, localisés
- [x] Pas de nouveaux fichiers
- [x] Pas de refactor global
- [x] Ne pas créer de nouveau système de logs (le GUI a déjà un dropdown niveau log)
- [x] Ne pas toucher au comportement batch size (0 vs >1) existant
- [x] SDS et grid_mode: ne pas modifier leur logique; seul l’aspect cache-retention Phase 3 est concerné
- [x] Ne pas changer l’API publique/config: utiliser `cache_retention_mode` existant
  - Si `cache_retention_mode != "per_tile"` => comportement inchangé
  - Si `cache_retention_mode == "per_tile"` => appliquer refcount en lieu et place de suppression naïve

## Portée / fichiers
- Principal: `zemosaic_worker.py`
- Aucun autre fichier sauf nécessité absolue (normalement aucune)

## Design attendu (très précis)
### 1) Normalisation des paths
Construire les clés du refcount à partir d’un chemin normalisé:
- `norm = os.path.normpath(str(path))`
(Option: Path(path).resolve() seulement si pas trop coûteux / pas de surprise sur Windows)

### 2) Pré-calcul refcount
Avant de lancer les futures Phase 3:
- Parcourir toutes les tuiles / groupes planifiés (group_info_list / raw_entries)
- Pour chaque `raw_entry['path_preprocessed_cache']`:
  - Ajouter au compteur global
- Important: éviter de compter 2 fois le même path dans **la même tile**:
  - Par tile: `unique_paths = set(norm_paths_for_tile)`
  - Incrémenter le global avec ces uniques

### 3) Map tile -> unique_paths
Conserver une structure:
- `tile_cache_paths_unique[tile_id] = set(normalized_paths)` pour décrément ensuite

### 4) À la complétion de chaque future
Quand une master tile est "done":
- Récupérer `paths = tile_cache_paths_unique[tile_id]` (set)
- Pour chaque path:
  - `refcount[path] -= 1`
  - si `refcount[path] == 0`: supprimer le fichier (unlink)
- En cas de suppression impossible (file missing / permission):
  - log DEBUG (pas WARN spam), continuer

### 5) Compatibilité avec erreurs
Si une master tile échoue:
- On décrémente quand même les paths de cette tile (car ce travail n’aura pas besoin d’être relancé dans le même run)
- MAIS: ne jamais descendre < 0: clamp / assert debug si négatif

### 6) Logging sobre
- 1 log INFO au début Phase 3 (si per_tile actif):
  "Phase3: per_tile cache retention uses refcount; safe deletion enabled (N unique cache files tracked)."
- log INFO en fin Phase 3:
  "Phase3: refcount cleanup complete; removed X files; remaining Y (if any)."
  si refcount tracking détecte qu’un .npy est référencé par >1 tile, log INFO “shared cache detected” (une seule ligne),

## Critères d'acceptation
- Sur dataset reproduisant le trou:
  - disparition des `mastertile_warn_cache_file_missing` causés par cleanup prématuré
  - `num_master_tiles == tiles_total` (si données valides)
  - plus de trou dans l'image finale
- SDS et grid_mode: aucun changement fonctionnel observable hors disparition des trous
- Diff minimal, lisible, centré Phase 3
