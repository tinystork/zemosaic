# Follow-up — Validation refcount cache Phase 3

## 0) Garde-fous
- [ ] Aucun nouveau fichier ajouté
- [ ] Aucun nouveau système de logs ajouté (utiliser logger existant / dropdown GUI déjà présent)
- [ ] Aucun changement au comportement batch size (0 vs >1)
- [ ] Aucune modification de logique SDS/grid_mode hors gestion cache Phase 3

## 1) Code review rapide (zemosaic_worker.py)
- [ ] Localiser Phase 3: construction master tiles + ThreadPool futures
- [ ] Identifier le bloc actuel de cleanup `cache_retention_mode == "per_tile"`
- [ ] Remplacer la suppression naïve par le refcount:
  - [ ] Pré-calcul refcount global (avec dédoublonnage par tile)
  - [ ] Stockage `tile_id -> set(paths)` normalisés
  - [ ] À la complétion d’une future: décrément + unlink seulement quand count==0
  - [ ] Gestion errors unlink: log DEBUG et continue
  - [ ] Protection: ne jamais passer refcount négatif (clamp ou assert debug)

## 2) Logs attendus (propres)
- [ ] Début Phase 3 (per_tile): 1 log INFO indiquant refcount activé + nombre de fichiers suivis
- [ ] Pas de spam: pas de WARN par fichier supprimé
- [ ] Fin Phase 3: (optionnel) 1 log INFO avec removed X

## 3) Tests manuels obligatoires

### Test A — Repro "trou"
- [ ] Relancer EXACTEMENT le dataset/config qui produit un trou
- [ ] Vérifier absence (ou forte réduction) de:
      - mastertile_warn_cache_file_missing
      - mastertile_error_no_valid_images_from_cache
      - run_warn_phase3_master_tile_creation_failed_thread
- [ ] Vérifier num_master_tiles == tiles_total (si dataset valide)
- [ ] Vérifier image finale: trou disparu

### Test B — Gros volume (stress)
- [ ] Lancer un run volumineux (ou sous-ensemble représentatif) avec per_tile activé
- [ ] Vérifier que l’espace disque n’explose pas (les caches se libèrent progressivement)
- [ ] Vérifier que le run ne ralentit pas anormalement (refcount = O(total inputs))

### Test C — SDS
- [ ] Lancer un run SDS court
- [ ] Vérifier: pas de régression phases 4/5, pas de nouveau fallback
- [ ] Vérifier: mêmes sorties attendues

### Test D — grid_mode
- [ ] Lancer un run grid_mode (mêmes paramètres habituels)
- [ ] Vérifier: pas de silent fallback induit par le patch
- [ ] Vérifier: master tiles toutes produites, pas de trous

## 4) Definition of Done
- [ ] Patch minimal, centré Phase 3 cache retention
- [ ] Repro du trou corrigée
- [ ] SDS + grid_mode validés
