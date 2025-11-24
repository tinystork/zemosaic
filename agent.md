# Mission : Audit et correction de la photométrie post-anchor (inter-tiles) entre V4.2.0 et V4WIP

## Contexte

Deux runs ont été effectués sur **le même jeu de données** :

- branche **V4.2.0** → résultat visuellement propre, histogrammes RGB cohérents ;
- branche **courante V4WIP** → légère perte de qualité, canal rouge décalé dans l’histogramme final.

Les logs correspondants sont disponibles (même dataset) :

- V4.2.0 : `zemosaic_workerV4.2.0.log`
- V4WIP  : `zemosaic_worker.log`

Analyse préliminaire faite à la main :

- Les lignes `[RGB-EQ] poststack_equalize_rgb ...` montrent **les mêmes gains et cibles** entre V4.2.0 et V4WIP (pour les mêmes master tiles).
- On peut donc **exclure la normalisation RGB intra-stack** (`poststack_equalize_rgb`) comme cause de la régression.

En revanche, une différence nette apparaît sur la **phase post-anchor / normalisation photométrique inter-tiles** :

- **V4.2.0** :
  - présence de lignes du type :
    - `[CLÉ_POUR_GUI: post_anchor_start]`
    - `[CLÉ_POUR_GUI: post_anchor_selected]`
    - `[CLÉ_POUR_GUI: post_anchor_shift] (Args: {'gain': ..., 'offset': ...})`
    - `apply_photometric: using affine_by_id for N tiles`
    - `apply_photometric: tile=tile:0000 gain=... offset=...`
  - ➜ la meilleure ancre est sélectionnée, un couple (gain, offset) est calculé et **appliqué à toutes les tuiles**.

- **V4WIP** :
  - on voit :
    - `[CLÉ_POUR_GUI: post_anchor_start]`
    - `[CLÉ_POUR_GUI: post_anchor_keep_old] (Args: {'impr': ...})`
  - mais **aucune** ligne `post_anchor_selected`, `post_anchor_shift` ni `apply_photometric`.
  - ➜ la logique semble décider de **conserver l’ancre précédente** et ne plus appliquer (ou logguer) l’affine photométrique global, en particulier sur la voie GPU Phase 5.

C’est très probablement la cause de la dérive globale de photométrie et du rendu couleur final.

## Objectif

1. **Comparer finement** le code de la photométrie post-anchor et de la sélection d’ancre entre :
   - branche **V4.2.0** (référence de bon comportement),
   - branche **V4WIP** (branche courante à corriger).

2. Identifier précisément :
   - où et comment la logique `post_anchor_selected` / `post_anchor_keep_old` a changé,
   - pourquoi `apply_photometric` n’est plus appelée / tracée dans V4WIP,
   - si la voie GPU Phase 5 suit bien la même pipeline que la voie CPU.

3. **Corriger V4WIP** pour :
   - rétablir le comportement photométrique de V4.2.0,
   - garantir que la **meilleure ancre est bien sélectionnée et appliquée**,
   - s’assurer que l’affine photométrique (gain, offset) est appliqué de façon identique en CPU et GPU,
   - conserver la compatibilité avec la télémétrie actuelle (logs, CLÉ_POUR_GUI, etc.).

4. **Ne pas modifier** la logique de normalisation RGB intra-stack :
   - `poststack_equalize_rgb` est considérée comme correcte et identique à V4.2.0 ;
   - ne pas changer son comportement, juste s’assurer qu’elle reste appelée comme avant.

5. Mettre à jour `followup.md` en expliquant clairement :
   - l’origine de la régression,
   - les patches appliqués,
   - comment valider que la qualité est revenue au niveau de V4.2.0.

## Périmètre

Fichiers probablement impliqués (liste indicative, à compléter) :

- `zemosaic_worker.py`
- `zemosaic_align_stack.py`
- `zemosaic_align_stack_gpu.py`
- `zemosaic_utils.py`
- tout module / fonction contenant :
  - la logique d’ancre (sélection de la meilleure ancre),
  - la phase de **post-anchor photometric normalization**,
  - les appels à `apply_photometric`, `affine_by_id`, etc.,
  - la distinction CPU / GPU pour la Phase 5.

**Important :**

- ❗ **Ne pas toucher au code SDS** (Super/Duper/Stack etc.).  
  Cette mission concerne uniquement le flux standard (master tiles, mosaïque classique).
- ❗ Ne pas modifier la logique GUI.
- ❗ Ne pas modifier la logique interne de `poststack_equalize_rgb` (intra-stack RGB).
- ❗ Les corrections doivent être **minimales, propres et documentées** (commentaires, docstring si nécessaire).

## Livrables attendus

1. Un patch complet sur la branche **V4WIP** qui :
   - restaure la sélection et l’application de la meilleure ancre comme en V4.2.0,
   - s’assure que **CPU et GPU** suivent la même pipeline photométrique en Phase 5,
   - maintient les logs caractéristiques (`post_anchor_selected`, `post_anchor_keep_old`, `post_anchor_shift`, `apply_photometric`).

2. Une description claire des différences trouvées entre V4.2.0 et V4WIP :
   - fonctions / blocs modifiés,
   - raison de la régression,
   - impact sur le résultat.

3. Une mise à jour de `followup.md` (dans ce repo) expliquant :
   - ce qui a été corrigé,
   - quels tests reproduire (par exemple rejouer le dataset de référence) pour vérifier :
     - la présence de `post_anchor_selected` + `apply_photometric` dans les logs,
     - le retour à une mosaïque photométriquement homogène et à un histogramme RGB cohérent.

Merci d’effectuer un audit rigoureux, de corriger la branche **V4WIP** en conséquence, et de documenter clairement toutes les décisions.
