# Follow-up – Post-anchor guard fix & validation

## 1. Rappel de la trouvaille

- La **RGB intra-stack** (`[RGB-EQ] poststack_equalize_rgb`) s’exécute avec les mêmes paramètres entre V4.2.0 et V4WIP (mêmes gains, même target).  
  ⇒ **À ne plus toucher** pour cette mission.

- La régression vient de la **condition de garde post-anchor** dans `zemosaic_worker.py` :

  ```python
  if same_anchor or improvement < min_improvement:
      _log_and_callback("post_anchor_keep_old", ...)
      return master_tiles, anchor_shift
En pratique, sur le dataset de test :

V4.2.0 : post_anchor_selected + post_anchor_shift + apply_photometric ... → mosaïque propre, histogrammes RGB bien alignés.

V4WIP : post_anchor_keep_old (impr≈0.146) et aucune ligne apply_photometric → mosaïque verdâtre / rouge décalé, footprint des tuiles visible.

## 2. Ce que tu dois faire maintenant

- [x] Confirmer, via diff V4.2.0 vs V4WIP, que l’ancienne logique était de type :

  ```python
  # Version V4.2.0 (esprit)
  if same_anchor and improvement < min_improvement:
      # keep old anchor
      ...
  ```

  ou équivalent, c’est-à-dire que l’amélioration suffisante permettait d’appliquer un nouveau shift même si l’ancre restait la même.

- [x] Adapter le code V4WIP pour retrouver ce comportement :

  Ne considérer `post_anchor_keep_old` que lorsque l’amélioration est insuffisante ET/OU que les métriques sont jugées non fiables.

  Dans tous les autres cas (en particulier “same_anchor mais improvement >= min_improvement”), sélectionner l’ancre, calculer gain/offset, puis appliquer l’affine à toutes les tuiles (comme en V4.2.0).

- [x] Laisser intacts :

  - Le calcul d’improvement,
  - Les garde-fous sur la médiane (MIN_MEDIAN_REL_DELTA + log DEBUG),
  - Les fonctions de RGB intra-stack (poststack_equalize_rgb & co).

## 3. Tests à lancer après patch

- [ ] Sur la branche V4WIP patchée : même dataset de référence, en mode CPU et GPU (si dispo).
- [ ] Vérifier dans `zemosaic_worker.log` la présence de `post_anchor_start`, `post_anchor_selected`, `post_anchor_shift`, `apply_photometric`, etc., ainsi que la disparition de `post_anchor_keep_old` sauf lorsque `improvement` est réellement sous le seuil.
- [ ] Contrôle visuel : mosaïque finale comparable à la V4.2.0 (fond homogène, pas de “footprint” des master tiles) et histogrammes RGB réalignés.
- [ ] Non-régression : aucun changement de comportement sur un petit dataset où l’anchor est mauvaise et aucun changement dans les logs `[RGB-EQ] poststack_equalize_rgb`.

4. Note pour la suite
Si après ce patch la qualité restait inférieure à V4.2.0 malgré :

post_anchor_selected + apply_photometric présents, et

RGB intra-stack identique,

alors on pourra ouvrir une nouvelle mission ciblée sur :

la pipeline Phase 5 (Two-Pass coverage, masking SDS, etc.),

sans revenir sur le post-anchor ni le RGB intra-stack.

Pour l’instant, l’objectif est exclusivement de rétablir la logique d’acceptation d’ancre identique à V4.2.0.
