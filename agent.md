# Agent – Régression photométrique V4WIP vs V4.2.0 (log “post_anchor_keep_old”)

## Contexte

Projet : **ZeMosaic / ZeSeestarStacker**  
Branches concernées :

- `V4.2.0` = **référence** (qualité OK),
- `V4WIP`  = branche courante avec **perte de qualité photométrique**.

Même dataset Seestar, même configuration (SDS off, Phase 5 CPU/GPU selon tests).

Constats :

1. Les logs `[RGB-EQ] poststack_equalize_rgb enabled=True, applied=True, gains=(..., ..., 1.000000)` sont **présents avec les mêmes gains et cibles** dans V4.2.0 et V4WIP.  
   ⇒ La **normalisation RGB intra-stack n’est pas la cause** de la régression et ne doit pas être modifiée. :contentReference[oaicite:2]{index=2}  

2. La divergence se situe dans la **phase post-anchor** (choix de l’ancre photométrique et application de l’offset global) dans `zemosaic_worker.py`. :contentReference[oaicite:3]{index=3}  

   - Log V4.2.0 (bon comportement) :

     ```text
     [CLÉ_POUR_GUI: post_anchor_start]
     [CLÉ_POUR_GUI: post_anchor_selected] (Args: {'tile': 0, 'impr': 0.232318821698048})
     ... post_anchor_shift ... (Args: {'gain': 0.7865391356567737, 'offset': 3.5014933309171283})
     apply_photometric: using affine_by_id for 32 tiles
     apply_photometric: tile=tile:0000 gain=0.78654 offset=3.50149
     ...
     ```

   - Log V4WIP (mauvais comportement) :

     ```text
     [CLÉ_POUR_GUI: post_anchor_start]
     [CLÉ_POUR_GUI: post_anchor_keep_old] (Args: {'impr': 0.14600923384024508})
     ```

   ⇒ En V4WIP **aucune ancre n’est acceptée** et **aucune affine photométrique n’est appliquée** aux tiles, d’où le décalage du rouge et la dérive globale visible sur la mosaïque finale.

3. Dans `zemosaic_worker.py`, la logique refactorisée autour du post-anchor contient :

   ```python
   same_anchor = False
   try:
       if prestack_anchor_id is not None:
           same_anchor = selected_tile_id == int(prestack_anchor_id)
   except Exception:
       same_anchor = False

   if same_anchor or improvement < min_improvement:
       _log_and_callback(
           "post_anchor_keep_old",
           lvl="INFO",
           callback=progress_callback,
           impr=float(improvement),
       )
       ...
       return master_tiles, anchor_shift
→ Clé de la régression : l’utilisation de same_anchor or improvement < min_improvement.
En V4.2.0, la logique équivalente utilisait une condition plus permissive (type same_anchor and improvement < min_improvement ou équivalent), ce qui permettait d’accepter un nouvel ajustement même si l’ancre restait la même, tant que improvement >= min_improvement.

Avec le or, dès que same_anchor est True, on tombe systématiquement dans post_anchor_keep_old, même si l’amélioration est significative (impr ≈ 0.146 dans le log V4WIP).

Mission
Comparer la fonction post-anchor entre V4.2.0 et V4WIP (même fichier, même zone) pour confirmer le changement de logique.

Corriger la condition de garde de manière à retrouver le comportement de V4.2.0 :

si l’amélioration est significative (improvement >= min_improvement), on doit appliquer l’ancre et l’offset même si selected_tile_id == prestack_anchor_id.

on ne doit garder l’ancienne ancre que lorsque :

l’amélioration est inférieure au seuil, ou

les métriques sont manifestement non fiables (cas déjà gérés par le code).

Maintenir intacts :

la logique de sélection des candidats (score, span, robust_sigma, etc.) ;

le garde-fou median_delta (MIN_MEDIAN_REL_DELTA) et le log DEBUG qui signale le contournement éventuel ;

la fonction [RGB-EQ] poststack_equalize_rgb et toute la partie RGB intra-stack (pas de modification de ces fonctions ni de leurs paramètres). zemosaic_worker

S’assurer que, une fois la condition corrigée, le flux V4WIP reproduit le comportement suivant :

log post_anchor_selected avec tile et impr ;

log post_anchor_shift avec gain / offset ;

log apply_photometric: using affine_by_id... et les lignes apply_photometric: tile=tile:XXXX gain=... offset=....

Ne rien changer à la mécanique de Two-Pass coverage, de SDS, ni aux fonctions de stacking GPU/CPU, sauf si une dépendance directe avec la logique d’ancre est clairement identifiée dans le diff.

Fichiers principaux à inspecter
zemosaic_worker.py

zone autour de la sélection d’ancre / logs post_anchor_* / apply_photometric. zemosaic_worker

Éventuellement pour contexte (lecture seule, pas de modification sauf nécessité démontrée) :

zemosaic_align_stack.py et zemosaic_align_stack_gpu.py pour la logique [RGB-EQ] poststack_equalize_rgb. zemosaic_align_stack

Contraintes
Pas de nouvelle dépendance externe.

Ne pas dégrader les performances GPU/CPU (pas de boucles Python inutiles).

Ne pas modifier l’API publique ni les signatures des callbacks GUI/log.

Préserver la compatibilité avec les deux GUI (Tk et Qt).

Livrable attendu
Un patch qui :

corrige la condition de garde post-anchor,

restaure l’application des affines photométriques comme en V4.2.0,

laisse poststack_equalize_rgb inchangé,

conserve les logs et garde-fous existants.

Commentaires dans le code expliquant brièvement la logique :

quand l’ancre est conservée,

quand elle est ré-appliquée,

et pourquoi on ne veut pas bloquer le cas “même ancre mais meilleure affine”.