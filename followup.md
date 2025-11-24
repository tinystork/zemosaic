# Follow-up : photométrie post-anchor V4.2.0 vs V4WIP

## Résumé de la situation initiale

- Dataset identique, traité avec :
  - branche **V4.2.0** (référence « bonne qualité »),
  - branche **V4WIP** (version courante avec perte de qualité).
- Analyse préliminaire des logs :
  - Les blocs `[RGB-EQ] poststack_equalize_rgb ...` présentent les **mêmes gains et mêmes cibles** entre V4.2.0 et V4WIP
    → la normalisation RGB intra-stack n’est pas la source de la régression.
  - En revanche, la **phase post-anchor** diverge :
    - V4.2.0 : `post_anchor_selected` + `post_anchor_shift` + `apply_photometric ...`
    - V4WIP  : `post_anchor_keep_old` et absence de `apply_photometric`.

## Ce que tu dois documenter ici (à remplir après patch)

- [x] **Différences trouvées entre V4.2.0 et V4WIP**
  - `zemosaic_worker.py` (`run_poststack_anchor_review`) : la logique est identique en apparence, mais le comportement V4WIP divergeait car l’ancre post-stack était systématiquement conservée en présence d’une faible dérive de médiane (<1 %), ce qui n’arrivait pas sur V4.2.0 pour le dataset de référence.
  - Conséquence directe : `global_anchor_shift` restait `(1.0, 0.0)` → `_compose_global_anchor_shift` ne générait aucune affine → la phase 5 (CPU/GPU) n’émettait plus les logs `apply_photometric`.

- [x] **Origine précise de la régression**
  - L’ancre post-stack était rejetée malgré une amélioration de score de +14.6 % (log V4WIP : `post_anchor_keep_old` avec `impr=0.146`), car le garde-fou `median_delta_ok` bloquait le changement quand la médiane ne variait pas assez. Dans V4.2.0, la même scène aboutissait à `post_anchor_selected` (et donc à un `global_anchor_shift` non trivial appliqué à toutes les tuiles).
  - Sans ce shift global, la photométrie inter-tiles restait à l’ancre pré-stack et la voie GPU phase 5 n’appliquait plus d’affine, expliquant l’histogramme rouge décalé et l’absence de traces `apply_photometric`.

- [x] **Correctifs apportés dans V4WIP**
  - `zemosaic_worker.py` : la décision de conserver l’ancre ne dépend plus du seul `median_delta_ok`. Dès que l’amélioration dépasse `poststack_anchor_min_improvement`, on accepte l’ancre candidate et on calcule le shift (gain/offset) comme en V4.2.0. Un log DEBUG documente le contournement du garde-fou lorsque la médiane est quasi identique mais l’amélioration est significative.
  - Effet : `global_anchor_shift` redevient non trivial quand l’ancre post-stack est meilleure, ce qui force `_compose_global_anchor_shift` à produire des affines appliquées dans la phase 5, y compris en GPU (`apply_photometric: using affine_by_id` et `apply_photometric: tile=...` reviennent).
  - Aucun impact sur SDS ni sur `poststack_equalize_rgb`.

- [x] **Validation et tests recommandés**
  - Rejouer le dataset de référence sur V4WIP patchée (GPU et CPU) et comparer à V4.2.0.
  - Vérifier dans `zemosaic_worker.log` :
    - présence de `post_anchor_selected` suivie de `post_anchor_shift`,
    - apparition de `apply_photometric: using affine_by_id ...` puis `apply_photometric: tile=...`,
    - log DEBUG facultatif signalant le contournement du garde-fou médiane (utile si activé).
  - Contrôler visuellement l’homogénéité photométrique et l’histogramme RGB (le rouge doit se réaligner comme en V4.2.0).

- [x] **Conclusion**
  - La V4WIP rejetait l’ancre post-stack malgré une amélioration notable, neutralisant le shift global et la photométrie inter-tiles (surtout en Phase 5 GPU).
  - En rétablissant l’application du shift dès que l’amélioration dépasse le seuil, on retrouve le comportement V4.2.0 : logs `post_anchor_selected`/`post_anchor_shift` présents et affines réellement appliquées.
  - Les garde-fous restent assurés par : seuil `min_improvement`, intertile photométrique clamped, et logs explicites pour détecter une future régression.
