
## `agent.md`

### Contexte

Projet : **ZeMosaic – Grid/Survey mode**

Deux versions de `grid_mode.py` :

* `grid_mode_last_good_geometry.py` :

  * géométrie / WCS / footprints / coverage **corrects**
  * pas de GPU, pas de multithreading, pas de chunking.
* `grid_mode.py` (actuel) :

  * ajoute **GPU**, **multithread** (ThreadPoolExecutor), **chunking**, nouveaux logs et photométrie / blending plus sophistiqués
  * **mais** la mosaïque produite est géométriquement fausse :

    * certaines tiles (ex : `tile_0008.fits`) ne couvrent pas la bonne zone,
    * la mosaïque finale ne ressemble plus à celle produite par la version “last good”.

Objectif : **revenir à la géométrie exacte de `grid_mode_last_good_geometry.py`**, tout en gardant :

* le support GPU / CPU,
* le multithreading des tiles,
* le chunking,
* les nouveaux logs de debug (TILE_GEOM, DEBUG_SHAPE, etc.),
* les nouveaux comportements de stacking (norm/weight/reject/combine identiques au worker).

---

### Objectif global

> **Faire en sorte que le Grid mode produise exactement la même couverture / géométrie (bboxes, offsets, WCS) que `grid_mode_last_good_geometry.py`, en conservant le pipeline de stacking CPU/GPU/multithread existant.**

La mosaïque finale doit être **indistinguable** de la version “last good” pour un même `stack_plan.csv` (même champ couvert, même placement relatif des sources), que le GPU soit activé ou non.

---

### Périmètre

Fichiers à modifier uniquement :

* `grid_mode.py`

Fichier de référence (lecture seule, pour copier la géométrie) :

* `grid_mode_last_good_geometry.py`

**Ne pas toucher** dans cette mission :

* `zemosaic_worker.py`
* `zemosaic_align_stack.py` / `zemosaic_align_stack_gpu.py`
* `zemosaic_stack_core.py`
* Toute la GUI (Tk / Qt)
* Le pipeline classique hors Grid mode.

---

### Stratégie

On va séparer **géométrie** et **stacking** :

* **Géométrie (WCS, canvas, tiles, bboxes, assignation)**
  → doit être **copiée / revertée** depuis `grid_mode_last_good_geometry.py`, en conservant seulement les nouveaux logs non intrusifs.
* **Stacking (CPU/GPU, multithreading, chunking, photométrie interne à la tile)**
  → reste tel qu’implémenté dans la nouvelle version.

---

### Tâches détaillées

#### 1. Restaurer la géométrie depuis `grid_mode_last_good_geometry.py`

1. Ouvrir **les deux fichiers** :

   * `grid_mode.py` (actuel)
   * `grid_mode_last_good_geometry.py` (version “last good”)

2. Pour les blocs suivants, **copier/coller la version “last good”** dans `grid_mode.py`, puis réadapter seulement ce qui est nécessaire aux nouvelles signatures / logs :

   * `_compute_frame_footprint(...)`
   * `_build_fallback_global_wcs(...)`
   * `_is_degenerate_global_wcs(...)` (si différent)
   * `_clone_tile_wcs(...)`
   * `build_global_grid(...)`
   * `assign_frames_to_tiles(...)`
   * `TilePhotometryInfo`, `TileOverlap` (si la structure a changé côté géométrie pure)
   * `assemble_tiles(...)` pour la **partie géométrique** :

     * allocation du canvas global,
     * interprétation de `GridTile.bbox`,
     * placement des tiles dans la mosaïque (indices y/x),
     * gestion du `coverage_mask` / `weight_sum` **sans cropping supplémentaire**.

3. Lors de ce “revert ciblé” :

   * **Conserver** les nouveaux logs utiles (TILE_GEOM, DEBUG_SHAPE, DEBUG_SHAPE_WRITE, coverage/unique/overlap, etc.).
   * **Ne pas réintroduire** le cropping agressif qui avait été ajouté :
     la mosaïque doit rester au **canvas global complet** (NaN ou 0 dans les zones vides), comme la version “last good”.
   * S’assurer que :

     * `GridDefinition.global_shape_hw`,
     * `GridDefinition.offset_xy`,
     * les `bbox` des `GridTile`,
     * et les WCS des tiles via `_clone_tile_wcs`
       sont gérés **exactement comme dans `grid_mode_last_good_geometry.py`**.

4. Vérifier la cohérence des **offsets** :

   * dans le cas fallback (`_build_fallback_global_wcs`), les footprints retournés sont déjà normalisés dans un repère local → ne pas re-normaliser une seconde fois.
   * l’offset global `(offset_x, offset_y)` doit être utilisé **une seule fois** pour :

     * ajuster `global_shape_hw`,
     * paramétrer `_clone_tile_wcs` (CRPIX global → CRPIX local tile),
     * **mais pas** pour décaler à nouveau les `bbox` ou les footprints.

#### 2. S’assurer que le stacking (CPU/GPU) utilise la nouvelle géométrie sans l’altérer

1. Ne pas modifier :

   * la logique de chunking dans `process_tile`,
   * le choix CPU vs GPU,
   * les fonctions `_fit_linear_scale_gpu`, `_normalize_patches_gpu`, `_stack_weighted_patches_gpu`,
   * les appels à `stack_core` côté CPU si présents.

2. Vérifier simplement que `process_tile(...)` continue d’utiliser :

   * `tile_shape_hw` dérivé de `tile.bbox` selon la géométrie “last good”,
   * les WCS de `tile.wcs` créés par `_clone_tile_wcs` restauré.

3. Conserver les logs de debug existants sur :

   * shape des patches,
   * fraction de NaN/zeros,
   * budget mémoire / chunking.

Aucune modification mathématique n’est attendue dans ces fonctions.

#### 3. Ajouter de la télémétrie précise pour valider la géométrie

1. Dans `build_global_grid(...)` :

   * logger `global_shape_hw`, `offset_xy`, nombre de tiles, et la liste **résumée** des bboxes (min/max x/y).
   * Exemple de log (déjà en partie présent, à vérifier) :
     `[GRID][TILE_LAYOUT] tile_id=8 bbox=(1152,2406,2304,3272) shape_hw=(968,1254)`

2. Dans `process_tile(...)` ou juste après la sauvegarde des tiles :

   * conserver / compléter le log de type :
     `[GRID][TILE_GEOM] id=8 path=tile_0008.fits shape=(H,W,C) bbox=(xmin,xmax,ymin,ymax)`

3. Dans `assemble_tiles(...)` :

   * avant d’écrire la mosaïque finale, logguer :

     * `mosaic_shape`,
     * `global_shape_hw` (doit être identique),
     * la somme des zones `unique` vs `overlap`.

Ces logs servent uniquement à valider que la géométrie restaurée est cohérente, pas à modifier la logique.

---

### Contraintes / Invariants

* Ne rien casser dans le mode classique.
* Ne pas supprimer les optimisations GPU/multithread déjà en place.
* Ne pas réintroduire de cropping automatique de la mosaïque finale :
  **la logique de recadrage est gérée plus tard par la pipeline classique**, pas par Grid.
* S’assurer que les tiles `tile_0001.fits`, `tile_0002.fits`, ..., `tile_0009.fits` ont :

  * des shapes cohérentes avec `global_shape_hw` et la taille de tuile choisie,
  * des contenus compatibles avec la version “last good” (visuellement et en termes de placement des sources).

---

### Résultat attendu

Avec le dataset de test M106 (le même `stack_plan.csv` que la référence) :

* Le Grid mode produit :

  * une mosaïque finale **géométriquement identique** à celle produite par `grid_mode_last_good_geometry.py`,
  * des tiles `tile_xxxx.fits` correctement cadrées (plus de “zone manquante” sur `tile_0008.fits`),
  * ce résultat est indépendant du flag GPU (ON/OFF).

