
## `followup.md`

### Ce qui a été demandé

* Restaurer un comportement “classique” sain :

  * master tiles **toujours** passées par le pipeline lecropper (quality crop + alt-az + alpha) avant sauvegarde ;
  * mosaïque finale **sans** égalisation RGB agressive par défaut ;
  * Grid mode / SDS inchangés.

---

### Checklist de vérification (à cocher après implémentation)

* [x] Dans `zemosaic_worker.py`, la fonction `create_master_tile(...)` applique :

  * [x] `poststack_equalize_rgb` (Phase 3) ;
  * [x] `apply_center_out_normalization_p3` si activé ;
  * [x] le bloc `quality_crop` basé sur `lecropper.detect_autocrop_rgb` (si `quality_crop_enabled=True`) ;
  * [x] `_apply_lecropper_pipeline(...)` avec un `pipeline_cfg` complet (quality + alt-az) ;
  * [x] `_normalize_alpha_mask(...)` et passe un `alpha_mask_out` cohérent à `save_fits_image(...)`.

* [x] Les logs `MT_CROP: quality-based rect=...` et `MT_CROP: quality crop skipped ...` apparaissent bien en Phase 3 sur un run de test.

* [x] L’appel à `_apply_final_mosaic_rgb_equalization(...)` est maintenant protégé par un flag (ex. `final_mosaic_rgb_equalize_enabled`), par défaut à `False`.

* [x] Quand le flag est à `False`, aucun log `[RGB-EQ] final mosaic ...` n’apparaît dans `zemosaic_worker.log`.

* [x] `_apply_final_mosaic_quality_pipeline(...)` et `_apply_master_tile_crop_mask_to_mosaic(...)` sont toujours appelés en fin de Phase 5 pour la voie classique.

* [x] Aucun changement fonctionnel involontaire n’est apporté à Grid mode (testé sur un petit dataset).

---

### Tests à lancer après la PR

1. **Run M106 classique (hors Grid/SDS)**

   * [ ] Histogramme final ≈ histogrammes des master tiles et de Grid (courbes proches, pas de dominante verte ni rouge saturée).
   * [ ] Pas de “tuiles fantômes” non croppées sur la mosaïque (bords propres).

2. **Run Grid mode sur un dataset déjà utilisé**

   * [ ] Résultat identique (ou très proche) de la version précédente, pas de crash.

3. **Optionnel : run SDS simple**

   * [ ] Vérifier que les master tiles et la mosaïque SDS restent cohérentes (couleur et cropping).

---

### Notes de suivi / Todo éventuels

* [ ] Si un jour on souhaite réactiver une **égalisation RGB finale douce**, il faudra :

  * limiter les gains (ex. clamp à [0.5, 1.5]) ;
  * travailler sur un échantillon robuste de la mosaïque (hors bords/NaN) ;
  * garder `poststack_equalize_rgb` comme base et ne faire que des corrections marginales.

* [ ] Documenter dans le wiki que, pour l’instant, la couleur finale est pilotée par `poststack_equalize_rgb` + `center_out` au niveau master tiles, et non par une égalisation sur la mosaïque globale.
