## `followup.md`

```markdown
# ✅ Suivi des tâches — Garde-fou WCS dégénéré & Fallback Grid Mode

## Instructions pour Codex

1. Lire `agent.md` entièrement.
2. Traiter la **première tâche non cochée** ci-dessous.
3. Après chaque modification :
   - cocher la tâche (`[x]`),
   - ajouter, si utile, une brève note (fonctions modifiées, décisions prises).
4. Répéter jusqu’à ce que toutes les tâches pertinentes soient cochées.

---

## Tâches

### [x] 1. Ajouter `_is_degenerate_global_wcs(...)` dans `grid_mode.py`

- Implémenter la fonction :

  ```python
  def _is_degenerate_global_wcs(frames, global_wcs, global_shape_hw) -> bool:
      ...
````

* Critères à inclure (au minimum) :

  * `MIN_SIZE` (ex: 256) sur `H_m` et `W_m`,
  * comparaison avec la taille moyenne des frames (`shape_hw`).
* Importer `numpy` si nécessaire (`np.mean`).
* Ne pas appeler cette fonction encore à ce stade (juste l’implémenter).

---

### [x] 2. Ajouter `_pick_first_valid_frame(...)` & `_build_fallback_global_wcs(...)`

* Implémenter `_pick_first_valid_frame(frames)` qui :

  * retourne le premier frame ayant un WCS et un `shape_hw` valides,
  * lève un `RuntimeError` si aucun frame valide n’est trouvé.

* Implémenter `_build_fallback_global_wcs(frames)` qui :

  * sélectionne `base_frame = _pick_first_valid_frame(frames)`,
  * copie son WCS (`copy.deepcopy(base_frame.wcs)`),
  * calcule les footprints de chaque frame dans ce WCS de base (en réutilisant si possible une fonction existante type `_compute_frame_footprint`),
  * construit `bounds` = liste de `(x0, x1, y0, y1)`,
  * si `bounds` est vide → `RuntimeError` explicite,
  * derive `min_x, max_x, min_y, max_y`, puis `global_shape_hw=(height, width)` et `(offset_x, offset_y)`,
  * applique `_strip_wcs_distortion` au WCS de base,
  * renvoie `(fallback_wcs, global_shape_hw, bounds)`.

* Ajouter des logs `[GRID]` pertinents :

  * erreurs de footprint,
  * résumé du fallback (nb de frames utilisés, shape_hw).

---

### [x] 3. Intégrer le garde-fou dans `build_global_grid(...)`

* Localiser l’appel à `find_optimal_celestial_wcs(...)` dans `build_global_grid`.

* Adapter la logique pour :

  ```python
  global_wcs, global_shape_hw = find_optimal_celestial_wcs(...)

  if _is_degenerate_global_wcs(frames, global_wcs, global_shape_hw):
      logger.warning(
          "[GRID] Optimal global WCS looks degenerate (shape_hw=%s), falling back to safer WCS",
          global_shape_hw,
      )
      global_wcs, global_shape_hw, global_bounds = _build_fallback_global_wcs(frames)
      logger.info(
          "[GRID] Fallback global WCS: shape_hw=%s",
          global_shape_hw,
      )
  else:
      logger.info(
          "[GRID] Optimal global WCS accepted: shape_hw=%s",
          global_shape_hw,
      )
      # global_bounds calculé comme avant
  ```

* S’assurer que `global_bounds` est bien défini dans les deux branches (optimal et fallback).

* Ne pas casser le chemin optimal existant.

---

### [x] 4. Vérifier la cohérence avec le reste du Grid mode

* Confirmer que :

  * le reste du code (calcul des offsets, `global_canvas shape_hw`, bboxes de tuiles) continue à utiliser `global_shape_hw` et `global_bounds` de manière cohérente.
* Vérifier qu’aucune dépendance implicite à l’ancienne valeur de `shape_hw` n’est cassée.
* Si un ajustement mineur est nécessaire (par ex. stockage d’un offset global dans une structure), le noter ici.

---

### [x] 5. Ajouter / compléter les logs `[GRID]`

* Vérifier que les nouveaux logs suivants existent :

  * warning quand le WCS optimal est jugé dégénéré,
  * info sur le WCS fallback (shape_hw, éventuellement nombre de frames/bounds).
* Garder les logs existants sur :

  * `global_bounds count=...`,
  * `global canvas shape_hw=..., offset=...`.

---

### [ ] 6. Test sur le dataset problématique (WCS 2×2)

* Lancer le Grid mode sur le dataset qui produisait le WCS `shape_hw=(2, 2)` / canvas 3×3.

* Vérifier dans les logs que :

  * `[GRID] Optimal global WCS looks degenerate...` apparaît,
  * `[GRID] Fallback global WCS: shape_hw=...` apparaît,
  * la mosaïque finale `mosaic_grid.fits` a une taille raisonnable (beaucoup plus que 3×3).

* Vérifier visuellement que la mosaïque contient bien du signal.

* Noter ici le résultat (date, taille finale observée).

---

### [ ] 7. Test sur un dataset sain (où `find_optimal_celestial_wcs` marchait déjà bien)

* Lancer le Grid mode sur un dataset pour lequel :

  * le Grid mode marchait bien avant les changements,
  * ou au minimum où la géométrie globale est connue/raisonnable.
* Vérifier que :

  * le garde-fou **n’est pas déclenché** (log “Optimal global WCS accepted”),
  * le résultat visuel et la taille de la mosaïque sont cohérents avec l’avant-patch.

---

### [ ] 8. Vérification de non-régression globale

* Vérifier que :

  * le pipeline classique (hors Grid mode) n’a pas été impacté (pas de modification dans d’autres fichiers),
  * le Grid mode ne plante pas sur de petits jeux de données (2–3 images),
  * les performances restent acceptables.

* Si tout est bon, cocher cette tâche et ajouter une courte note de validation.

---

## Notes / Journal

> Utiliser cette section pour consigner :
>
> * les valeurs de `MIN_SIZE` retenues,
> * les observations sur les tailles de mosaïque obtenues,
> * les éventuels ajustements faits en plus du plan.
