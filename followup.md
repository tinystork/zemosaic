### [x] Étape 1 — Nouveau flag de configuration

1. Ouvrir `zemosaic_config.py`. 

2. Dans la structure de configuration principale (dataclass ou dict), ajouter :

   ```python
   cleanup_temp_artifacts: bool = True
   ```

3. S’assurer que :

   * `load_config()` retourne `cleanup_temp_artifacts=True` si la clé est absente du fichier utilisateur.
   * `save_config()` sérialise cette clé comme les autres booléens.

### [x] Étape 2 — GUI Qt (System resources & cache)

1. Ouvrir `zemosaic_gui_qt.py`.

2. Localiser `_create_system_resources_group(self)` (groupe *System resources & cache*).

3. Ajouter une nouvelle checkbox :

   * Clé : `"cleanup_temp_artifacts"`
   * Label via localisation :

     ```python
     self._register_checkbox(
         "cleanup_temp_artifacts",
         layout,
         self._tr(
             "qt_field_cleanup_temp_artifacts",
             "Delete temporary processing files after run",
         ),
     )
     ```

4. Positionner cette checkbox **dans le même group box** que les options memmap (entre memmap et cache_retention, ou juste après, selon ergonomie).

5. Vérifier que la valeur est correctement lue/écrite dans `self.config` comme les autres champs (`coadd_use_memmap`, `coadd_cleanup_memmap`, `cache_retention`, etc.).

### [x] Étape 3 — Faire remonter la config vers le worker

1. Vérifier comment la config est passée au worker (via `run_zemosaic.py` / Qt main window).
   Normalement, `cleanup_temp_artifacts` sera inclus automatiquement si tu relies bien la checkbox au dict de config.

2. Dans `zemosaic_worker.py`, au début de la fonction principale du run (là où on lit `worker_config_cache`), ajouter :

   ```python
   cleanup_temp_artifacts_config = bool(
       (worker_config_cache or {}).get("cleanup_temp_artifacts", True)
   )
   ```

3. Garder cette variable accessible aux sections :

   * `_cleanup_memmap_artifacts()`,
   * SDS runtime cleanup,
   * Phase 7 cleanup (cache, master tiles, etc.).

### [x] Étape 4 — Refactor `_cleanup_memmap_artifacts`

1. Toujours dans `zemosaic_worker.py`, localiser `_cleanup_memmap_artifacts()` (tout en bas de la fonction principale actuelle).

2. Ajouter un early exit :

   ```python
   if not cleanup_temp_artifacts_config:
       return
   ```

3. Conserver la logique actuelle de suppression des `.dat` et `mosaic_first_*` **dans le memmap dir** **uniquement** si :

   ```python
   if (
       bool(coadd_use_memmap_config)
       and bool(coadd_cleanup_memmap_config)
       and coadd_memmap_dir_config
       and _path_isdir(coadd_memmap_dir_config)
   ):
       # boucle actuelle sur memmap_cleanup_dir.iterdir()
   ```

4. Étendre la suppression des `mosaic_first_*` :

   * Après le bloc ci-dessus, récupérer `runtime_temp_dir = get_runtime_temp_dir()` (depuis `zemosaic_utils` ou fallback).
   * Parcourir les sous-dirs immédiats et supprimer ceux dont le nom commence par `"mosaic_first_"` (sans toucher aux autres).

5. **Toujours** (tant que `cleanup_temp_artifacts_config` est True) exécuter la partie sur les WCS globaux :

   * garder la construction de `wcs_candidates` (via `global_wcs_plan`, `output_folder`, etc.),
   * supprimer uniquement les fichiers dont `candidate_path.name.lower()` contient `"global_mosaic_wcs"`.

### [x] Étape 5 — Phase 7 : maîtriser la suppression des artefacts

Toujours dans `zemosaic_worker.py`, bloc **Phase 7 (Nettoyage)**. 

1. **Cache des brutes pré-traitées** (`temp_image_cache_dir`) :

   * NE PAS modifier la logique de `cache_retention` :

     * `keep` → log `run_info_temp_preprocessed_cache_kept` et ne rien supprimer,
     * sinon → suppression + log `run_info_temp_preprocessed_cache_cleaned`.
   * Cette partie reste **indépendante** du nouveau flag.

2. **Master tiles / mega tiles** (`temp_master_tile_storage_dir`) :

   * Remplacer la condition actuelle :

     ```python
     if master_tiles_dir:
         if not two_pass_enabled and _path_exists(master_tiles_dir):
             # rmtree + log
     ```

   * Par une logique contrôlée par `cleanup_temp_artifacts_config` :

     ```python
     if master_tiles_dir and _path_exists(master_tiles_dir):
         if cleanup_temp_artifacts_config:
             # rmtree + log (run_info_temp_master_tiles_fits_cleaned, déjà existant)
         else:
             # optionnel : log INFO_DETAIL disant que les master tiles sont conservées pour debug
     ```

   * On ne s’appuie plus sur `two_pass_enabled` pour la suppression une fois le run terminé.

3. **SDS runtime tiles** (`sds_runtime_tile_dir`) :

   * Remplacer :

     ```python
     if sds_runtime_tile_dir:
         try:
             shutil.rmtree(sds_runtime_tile_dir, ignore_errors=True)
         except Exception:
             pass
     ```

   * Par :

     ```python
     if sds_runtime_tile_dir and cleanup_temp_artifacts_config:
         try:
             shutil.rmtree(sds_runtime_tile_dir, ignore_errors=True)
         except Exception:
             pass
     ```

4. Si d’autres répertoires **purement temporaires** liés au traitement sont supprimés dans cette phase, les faire aussi dépendre de `cleanup_temp_artifacts_config`.

### [x] Étape 6 — Localisation complète

1. Ouvrir **tous** les fichiers de localisation dans `locales/` (au moins `en.json`, `fr.json`, `es.json`, `pl.json`, `de.json`, `nl.json`, `is.json` si présents).

2. Ajouter la clé :

   ```json
   "qt_field_cleanup_temp_artifacts": "Delete temporary processing files after run"
   ```

   dans `en.json` (Qt section des champs, à côté des autres `qt_field_*`).

3. Dans `fr.json`, ajouter :

   ```json
   "qt_field_cleanup_temp_artifacts": "Supprimer les fichiers temporaires de traitement après l’exécution"
   ```

4. Pour toutes les autres langues, ajouter la même clé avec :

   * soit une vraie traduction,
   * soit la version anglaise, mais **jamais** laisser la clé absente.

5. Vérifier que **toutes** les nouvelles chaînes (s’il y en a d’autres que cette clé) sont également présentes dans tous les JSON.

### [ ] Étape 7 — Tests manuels recommandés

*Non réalisés dans cet environnement (exécution GUI requise).* 

1. **Cas par défaut (nouvelle option cochée)** :

   * Lancer ZeMosaic Qt, vérifier que la checkbox est **cochée** par défaut dans l’onglet *System*.
   * Faire un run complet avec :

     * memmap activé,
     * SDS activé ou non,
     * éventuellement two-pass coverage activé.
   * À la fin :

     * vérifier que les dossiers `mosaic_first_*`, `temp_master_tile_storage_dir` et `sds_runtime_tile_dir` ont disparu,
     * vérifier que les `.dat` temporaires (mosaic / coverage) sont supprimés du memmap dir,
     * vérifier que les `global_mosaic_wcs*.fits/.json` auto-générés ne sont plus présents,
     * vérifier que les fichiers finaux (FITS/PNG de la mosaïque) sont toujours là.

2. **Cas debug (option décochée)** :

   * Décoche `Delete temporary processing files after run`.
   * Relancer un run sur un jeu de test.
   * Vérifier que :

     * `mosaic_first_*`, master tiles, SDS runtime dir sont conservés,
     * les `.dat` & WCS globaux ne sont plus supprimés,
     * le cache de brutes suit toujours `cache_retention` (ex : `run_end` continue d’effacer le cache des brutes).

3. **Sanity check cross-langues** :

   * Changer la langue (FR, EN, au moins une 3e langue) et vérifier que le label de la checkbox s’affiche correctement (pas de `%KEY%` brut).
   * Sur erreur de traduction, corriger les JSONs.
   * Non réalisé ici (GUI non lancée dans cet environnement).

