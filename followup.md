## `followup.md` – Checklist Grid Mode Geometry & Photometry

### 1. Géométrie & canevas global

- [x] Rechercher dans `grid_mode.py` l’endroit où la WCS globale et la taille de canevas sont définies.
- [x] Vérifier qu’il existe une structure unique (ex. `GridDefinition.global_wcs`, `GridDefinition.global_shape_hw`) utilisée partout.
- [x] Confirmer que **toutes les allocations** de tableaux globaux (mosaïque, coverage, alpha) utilisent **exactement** `global_shape_hw`.
- [x] Vérifier que **personne** ne recalcule la taille globale à partir des tuiles.

**Test rapide :**

- Lancer un run Grid sur un dataset de test.
- Vérifier dans le log `[GRID]` que la shape globale est la même du début à la fin.

---

### 2. BBox de tuiles & alignement

- [x] Inspecter la fonction qui crée les `GridTile` et leurs `bbox`.
- [x] S’assurer que les `bbox` sont calculées en pixels du canevas global (via WCS), pas à partir d’une simple grille row/col.
- [x] Ajouter un log DEBUG pour chaque tuile : `tile_id`, `bbox`, `tile_shape_hw`.
- [x] Vérifier que pour chaque tuile : `0 <= xmin < xmax <= global_width` et idem pour y.

**Validation pratique :**

- Utiliser un dataset où le flux classique produit une mosaïque correcte.
- Comparer dans un viewer (DS9, Siril, etc.) la position de quelques étoiles sur la mosaïque classique vs Grid : elles doivent coïncider à ±1 pixel.

---

### 3. Stacking par tuile

- [x] Vérifier que toutes les tuiles sont empilées via `stack_core` (CPU/GPU) avec les bons paramètres.
- [x] Confirmer que pour les images RGB, `equalize_rgb_medians_inplace` est appelé **avant** tout calcul de stats de fond/scaling.
- [x] S’assurer que `compute_valid_mask` est utilisé pour produire une `tile_mask` cohérente.

**Test :**

- Activer le logging DEBUG pour le Grid Mode.
- Vérifier que chaque tuile a des stats raisonnables (min/median/max) dans les logs, sans NaN généralisé.

---

### 4. Normalisation photométrique inter-tile

- [ ] Identifier la tuile de référence utilisée pour la photométrie.
- [ ] Vérifier que :

  - un `common_mask` correctement construit (intersection de masques valides) est utilisé,
  - si l’overlap est insuffisant → log WARN et pas de scaling.

- [ ] Confirmer l’appel à :

  ```python
  gains, offsets = compute_tile_photometric_scaling(ref_patch, tgt_patch, mask=common_mask)
  info.data = apply_tile_photometric_scaling(info.data, gains, offsets)
  info.mask = compute_valid_mask(info.data) & info.mask
````

* [ ] Vérifier que les logs `[GRID] Photometry` montrent des gains/offsets **finis** et raisonnables.

**Validation pratique :**

* Lancer un run Grid.
* Inspecter la mosaïque : aucune bande verticale ou horizontale nette entre tuiles ne doit être visible (au moins au premier ordre).

---

### 5. Assemblage des tuiles en mosaïque

* [ ] Ouvrir `assemble_tiles(...)` dans `grid_mode.py`.

* [ ] Vérifier que :

  * la mosaïque globale est allouée avec `global_shape_hw`,
  * pour chaque tuile, on indexe `mosaic_data[y0:y1, x0:x1]` avec la `bbox`,
  * on cumule les contributions pondérées (poids = masque ou coverage),
  * à la fin, on divise par les poids là où ils sont > 0.

* [ ] S’assurer qu’aucune reproject globale supplémentaire (type `reproject_interp` vers une nouvelle WCS) n’est faite à ce stade.

**Test visuel :**

* Comparer la mosaïque Grid et la mosaïque classique sur le même dataset.
* Vérifier que les bords de tuiles ne forment plus de “marches d’escalier” ou de décalages.

---

### 6. Autocrop & CRPIX/NAXIS

* [ ] Vérifier dans `zemosaic_worker.py` que l’autocrop global se fait via `_auto_crop_global_mosaic_if_requested` puis `_apply_autocrop_to_global_plan`.
* [ ] S’assurer que `grid_mode.py` ne modifie pas lui-même `CRPIX1/2` ou `NAXIS1/2` après coup.
* [ ] Confirmer que le plan retourné au worker contient la bonne largeur/hauteur après autocrop.

**Test :**

* Faire un run avec autocrop activé.
* Vérifier que la taille de la mosaïque correspond à la coverage utile (pas de grosses bordures vides).

---

### 7. Fallback & logs

* [ ] Examiner le code Grid dans `zemosaic_worker.py` :

  * en cas de succès Grid → **pas** de fallback.
  * en cas d’échec (`run_grid_mode` lève) → fallback explicite avec log clair.

* [ ] Ajouter si besoin un log type :

  ```python
  logger.warning("[GRID] Fallback to classic pipeline: reason=%s", reason)
  ```

* [ ] S’assurer qu’aucun fallback ne se déclenche sur un dataset sain.

**Validation :**

* Lancer plusieurs runs Grid sur un dataset valide.
* Vérifier dans les logs qu’il n’y a **aucune** ligne mentionnant un fallback Grid → classique.
* Vérifier que le fichier de mosaïque Grid est bien produit (nom et chemin attendus).

---

### 8. Tests finaux de non-régression

* [ ] Tester le flux **classique** (sans `stack_plan.csv`) → résultat identique à avant la mission.
* [ ] Tester Grid Mode en CPU-only.
* [ ] Tester Grid Mode en GPU (si disponible).
* [ ] Comparer visuellement Grid vs classique : mêmes positions d’objets, fond homogène.

Quand tous les items ci-dessus sont cochés, la mission Grid Mode peut être considérée comme terminée.
