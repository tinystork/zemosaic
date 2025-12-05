# ✅ Suivi des tâches — Grid Mode & WCS global

## Instructions pour Codex

1. Lire `agent.md` entièrement.
2. Reprendre la liste ci-dessous et traiter la **première tâche non cochée**.
3. Après chaque modification :
   - Mettre à jour ce fichier `followup.md` en cochant la tâche effectuée (`[x]`),
   - Ajouter, si utile, un court commentaire en dessous (ce qui a été fait, fichiers touchés).
4. Répéter jusqu’à ce qu’il n’y ait plus de tâche non cochée.

---

## Tâches

### [x] 1. Analyser l’existant dans `build_global_grid`

- Identifier comment :
  - `global_bounds` est construit,
  - `global_shape_hw` est calculé,
  - `global_wcs` est choisi (optimal vs fallback),
  - `crpix` / `crval` sont éventuellement modifiés.
- Lister rapidement les points critiques dans un commentaire de code ou une note dans ce fichier.

  - `global_bounds` est rempli via `_compute_frame_footprint` pour chaque frame ; en cas de liste vide, `min_x`/`min_y` sont 0 et `max_y`/`max_x` reprennent `global_shape_hw` sans s’appuyer sur les footprints.
  - `global_shape_hw` provient de `find_optimal_celestial_wcs` (avec `auto_rotate=True`) ou du premier frame en fallback sur exception ; aucune recomposition à partir des footprints.
  - `global_wcs` est soit l’optimal retourné, soit un clone du premier frame ; il est seulement passé dans `_strip_wcs_distortion`.
  - `crpix` du WCS global n’est pas modifié, mais `_clone_tile_wcs` décale `crpix` des tuiles en soustrayant le coin supérieur gauche de la bbox ; aucune mise à jour de `crval` n’est faite.

---

### [x] 2. Introduire l’offset global `(offset_x, offset_y)` et recalculer `global_shape_hw`

- À partir de `global_bounds`, calculer :
  - `min_x`, `max_x`, `min_y`, `max_y`,
  - `offset_x = min_x`, `offset_y = min_y`,
  - `width = max_x - min_x`, `height = max_y - min_y`,
  - `global_shape_hw = (height, width)`.
- Gérer proprement le cas où `global_bounds` est vide (log explicite + sortie clean du Grid mode si nécessaire).

  - Implémenté dans `grid_mode.py` : les empreintes alimentent désormais `min_x`, `max_x`, `min_y`, `max_y`, puis `global_shape_hw` est recalculé à partir du couple `(width, height)` dérivé de ces bounds. Un offset `(offset_x, offset_y)` est stocké dans `GridDefinition`. Le mode Grid s’interrompt proprement avec un log `[GRID]` si aucune empreinte n’est disponible ou si les bounds produisent une étendue non positive.

---

### [x] 3. Appliquer l’offset à toutes les bboxes / footprints utilisées pour les tuiles

- Adapter le code pour que toutes les bboxes utilisées lors de l’assemblage soient transformées en coordonnées **locales** via :

  ```python
  local_x0 = x0 - offset_x
  local_x1 = x1 - offset_x
  local_y0 = y0 - offset_y
  local_y1 = y1 - offset_y
````

* S’assurer que les structures de données qui stockent les bboxes (frames, tiles, etc.) utilisent désormais ces coordonnées locales pour le placement dans le canvas global.

  - Les empreintes de frames sont rebasculées en coordonnées locales après calcul de l’offset, les tuiles sont désormais générées et stockées en coordonnées `[0, W) × [0, H)` et leur WCS dérive le décalage global en combinant l’offset et l’origine locale de la tuile.

---

### [ ] 4. Nettoyer / sécuriser la gestion de `crpix` / `crval`

* Vérifier s’il existe un code qui :

  * modifie `global_wcs.wcs.crpix` après coup (par ex. pour le recentrer),
  * sans ajuster `crval`.
* Pour cette mission :

  * Soit **supprimer** ces modifications et s’appuyer uniquement sur l’offset pixel,
  * Soit, si vraiment nécessaire, mettre à jour `crval` correctement pour conserver la géométrie (et documenter clairement ce choix).
* Commenter dans le code que la stratégie retenue est d’utiliser un offset pixel global pour garder le WCS cohérent.

---

### [ ] 5. Améliorer le fallback quand `find_optimal_celestial_wcs` échoue

* Lorsqu’aucun WCS optimal n’est trouvé :

  * Utiliser le WCS du premier frame ou un autre WCS fallback comme actuellement,
  * Calculer les footprints de **tous** les frames dans ce WCS,
  * Construire `global_bounds` et appliquer la même logique :

    * offset `(offset_x, offset_y)`,
    * `global_shape_hw` dérivé des bounds.
* Si aucun footprint valide n’est obtenable, loguer clairement et abandonner proprement le Grid mode.

---

### [ ] 6. Ajouter/renforcer les contrôles WCS & rejets de frames invalides

* Dans `_load_frame_wcs` / `_compute_frame_footprint` :

  * Rejeter les frames avec WCS incomplet / incohérent,
  * Rejeter les footprints manifestement invalides (NaN majoritaire, taille nulle, etc.).
* Ajouter des logs `[GRID]` pour chaque frame rejetée avec la raison.

---

### [ ] 7. Ajouter des logs `[GRID]` détaillés

* Après tentative de `find_optimal_celestial_wcs` :

  * Succès / échec + fallback.
* Après calcul de `global_bounds` + canvas :

  * `min_x`, `max_x`, `min_y`, `max_y`,
  * `global_shape_hw`,
  * `offset_x`, `offset_y`.
* Pendant l’assemblage des tuiles :

  * Nombre de tuiles valides,
  * Nombre de tuiles rejetées car hors canvas,
  * Exemple de bboxes après offset.

---

### [ ] 8. Tests synthétiques & validation

* Si possible, ajouter un petit test (ou script) qui :

  * simule quelques frames/WCS,
  * force un échec de `find_optimal_celestial_wcs`,
  * vérifie que :

    * `global_shape_hw` est positif,
    * l’offset est appliqué,
    * aucune bbox finale n’est négative.
* Documenter brièvement dans ce fichier le résultat du test (succès / observations).

---

### [ ] 9. Test de régression sur dataset réel (Grid mode)

* Lancer le Grid mode sur le dataset réel qui produisait `bbox_extent=(-1:2,-1:2)`.
* Vérifier :

  * qu’il n’y a plus de bboxes négatives dans les logs,
  * que la mosaïque contient des données visibles,
  * que le message "no valid tile data written to mosaic" n’apparaît plus (sauf cas vraiment dégénérés).
* Noter ici le résultat avec la date et le dataset utilisé.

---

### [ ] 10. Vérifier la non-régression des autres modes

* Vérifier que :

  * le pipeline classique (hors Grid mode) n’est pas affecté,
  * le Grid mode se comporte comme avant quand `find_optimal_celestial_wcs` réussit et que les bounds sont déjà propres,
  * les performances restent acceptables.
* Si tout est bon, cocher cette tâche et éventuellement ajouter une note.

---

## Notes / Journal de bord

> Ajouter ici au fil de l’eau les remarques, décisions, datasets utilisés pour les tests, etc.

