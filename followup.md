## followup.md

### Récapitulatif des modifications à vérifier

* [ ] **Mission 1 :**

  * [ ] Une fonction `_reset_filter_log()` a été ajoutée dans `zemosaic_filter_gui_qt.py` (ou le point d’entrée du filtre).
  * [ ] Cette fonction supprime `zemosaic_filter.log` au chargement du module, sans casser le démarrage en cas d’erreur.
  * [ ] Aucun changement dans la config de logging, seuls les fichiers sont affectés.

* [ ] **Mission 2 :**

  * [ ] `zemosaic_worker.py` importe désormais `_poststack_rgb_equalization` depuis `zemosaic_align_stack.py` via un `try/except` sécurisé.
  * [ ] Un helper `_apply_final_mosaic_rgb_equalization(...)` a été ajouté dans `zemosaic_worker.py`, réutilisant `_poststack_rgb_equalization` pour la mosaïque finale.
  * [ ] `run_hierarchical_mosaic(...)` appelle ce helper **uniquement** pour le flux mosaïque classique (condition `not sds_mode_phase5` ou équivalent).
  * [ ] Le helper logge une ligne `[RGB-EQ] final mosaic: ...` lorsqu’un équilibrage est effectivement appliqué.
  * [ ] Aucun changement n’a été apporté à `grid_mode.py`.

### Tests manuels à effectuer

1. **Log du filtre**

   * [ ] Lancer le filtre Qt plusieurs fois, vérifier que `zemosaic_filter.log` repart bien de zéro à chaque ouverture.

2. **Flux classique**

   * [ ] Lancer un run classique avec `poststack_equalize_rgb=True`.
   * [ ] Vérifier dans `zemosaic_worker_cl.log` la présence de la ligne `[RGB-EQ] final mosaic: ...`.
   * [ ] Inspecter la mosaïque finale : vérifier que la dominante verte est corrigée.

3. **Flux Grid mode**

   * [ ] Lancer au moins un run Grid avec un dataset connu.
   * [ ] Confirmer qu’il n’y a **aucune régression** : géométrie, couleurs, logs identiques à avant la modification.

