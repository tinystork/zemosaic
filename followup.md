
## `followup.md`

### Ce que tu dois vérifier dans le code

1. [x] Les fonctions suivantes dans `grid_mode.py` correspondent structurellement à celles de `grid_mode_last_good_geometry.py`, à quelques logs près :

   * [x] `_compute_frame_footprint`
   * [x] `_build_fallback_global_wcs`
   * [x] `_is_degenerate_global_wcs`
   * [x] `_clone_tile_wcs`
   * [x] `build_global_grid`
   * [x] `assign_frames_to_tiles`
   * [x] `assemble_tiles` (partie géométrie : canvas, placement, pas de cropping).

2. [x] L’offset `(offset_x, offset_y)` est utilisé **exactement une fois** pour :

   * [x] normaliser les footprints / global bounds,
   * [x] ajuster le WCS des tiles via `_clone_tile_wcs`,
   * [x] **pas** pour décaler plusieurs fois les bboxes.

3. [x] `GridDefinition.global_shape_hw` et `mosaic_shape` dans `assemble_tiles` sont égaux.

4. [x] Aucune logique de cropping final n’est appliquée (plus de `coverage_mask` → recalcule WCS / shape).

5. [x] Le code GPU / multithread / chunking n’a pas été modifié en dehors de l’adaptation aux nouvelles shapes.

---

### Tests à lancer (manuels)

1. **Run Grid mode – CPU only**

   * Désactiver le GPU (config `use_gpu=False`).
   * Lancer ZeMosaic sur le dataset M106 avec `stack_plan.csv`.
   * Vérifier :

     * [ ] Les tiles `tile_0001.fits` → `tile_0009.fits` ont les mêmes dimensions qu’avant (1920×1920, 1920×1254, 968×1920, 968×1254, 968×102, etc.).
     * [ ] Sur `tile_0008.fits`, le champ est **completement rempli** comme sur la version “last good” (plus de gros trous ou cadrage décalé).
     * [ ] La mosaïque finale ressemble visuellement à la référence (image 2 passée par l’utilisateur), avec le même champ couvert.

2. **Run Grid mode – GPU ON**

   * Activer le GPU via la config / GUI.
   * Refaire le même run.
   * Vérifier :

     * [ ] Les tiles ont les mêmes bboxes et shapes que le run CPU.
     * [ ] La mosaïque finale est géométriquement identique (seules de petites différences numériques sont acceptables).
     * [ ] Pas d’erreur de “shape mismatch” ou de problème de casting.

3. **Logs de validation**

   * Vérifier que les logs `[GRID][TILE_GEOM]` et `[GRID][TILE_LAYOUT]` sont présents et cohérents :

     * [ ] `global canvas shape_hw` est identique entre `build_global_grid` et `assemble_tiles`.
     * [ ] Les bboxes des tiles forment une grille cohérente, sans trous ni décalages.
   * Optionnel : comparer rapidement les logs actuels à ceux d’un run avec `grid_mode_last_good_geometry.py` pour s’assurer que la géométrie est la même.
