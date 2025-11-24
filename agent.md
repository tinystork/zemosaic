# Mission : Rollback sélectif du commit 08bebf5 (cluster refactor)
# Objectif : restaurer la géométrie/stacking V4.2.0 sans perdre les commits ultérieurs

## Contexte

Le commit `08bebf5` a introduit :
- une refonte partielle de la logique de clustering / align_stack,
- des modifications dans la géométrie interne des master tiles,
- des altérations WCS,
- et plusieurs désactivations temporaires non souhaitées (streaming, GPU, crop).

Conséquences visibles (confirmées par l’analyse comparative V4.2.0 vs V4WIP) :
- WCS divergents (`WCS.all_world2pix failed to converge`),
- disparition de preview.png et coverage_xxx.dat,
- mosaïque patchwork (master tiles mal positionnées),
- altération Phase 5 (reproject & coadd),
- pipeline align_stack incompatible avec V4.2.0.

Les tests montrent que la normalisation RGB intra-stack **n’est pas en cause** et ne doit pas être modifiée.

## Mission

1. **Identifier précisément les modifications introduites par le commit 08bebf5** dans :
   - `zemosaic_align_stack.py`
   - `zemosaic_align_stack_gpu.py`
   - éventuellement `zemosaic_utils` (WCS, pivot, padding)
   - et tout autre fichier touché *uniquement* par ce commit.

2. **Supprimer ou réécrire ces modifications**, afin de :
   - restaurer entièrement la logique de stacking/alignement V4.2.0,
   - restaurer la géométrie des master tiles,
   - restaurer les WCS internes corrects,
   - restaurer la compatibilité avec `reproject`.

3. **Conserver intégralement** :
   - les commits ultérieurs (GPU path, resume, correctifs d’erreurs),
   - les améliorations de performances,
   - les ajouts utiles.
   → Le rollback doit donc être **sélectif**, pas un revert global.

4. **Retirer toutes les “décisions temporaires”** introduites par la tentative de réparation :

### A. Streaming / Winsor
- Retirer les forçages :
  - `stack_disable_streaming=True`
  - `winsor_disable_streaming=True`
- Restaurer les valeurs V4.2.0 :
  - streaming activé si la configuration utilisateur le permet.

### B. GPU désactivé par défaut
- Restaurer :
  - `use_gpu_global=True`
  - `use_gpu_phase5=True`
- Retirer les forçages `use_gpu_phase5_flag=False`.

### C. Crops désactivés par défaut
- Restaurer :
  - `apply_master_tile_crop=True`
  - `apply_crop_for_assembly=True`
  - `global_wcs_autocrop_enabled=True`

### D. Assemblage Phase 5 forcé en “no-autocrop”
- Restaurer le comportamento V4.2.0 :
  - master tile crop autorisé,
  - autocrop global actif,
  - reprojection GPU autorisée.

5. **Vérifier** qu'après rollback :
   - preview.png revient,
   - coverage_xxx.dat revient,
   - plus aucun warning WCS,
   - la mosaïque final retrouve la qualité V4.2.0,
   - les dimensions / offsets / orientations des master tiles correspondent à V4.2.0,
   - Phase 5 CPU/GPU produit la même géométrie qu’en V4.2.0.

## Contraintes strictes

- Ne pas toucher aux fonctions SDS.
- Ne pas toucher à `poststack_equalize_rgb` ni à la normalisation intra-stack.
- Ne pas supprimer les commits ajoutant des fonctionnalités utiles.
- Le rollback doit être **focalisé** sur les changements géométriques/WCS/clusters introduits par 08bebf5.
- Les paths GPU doivent rester fonctionnels.
- La compatibilité Qt/Tk doit être maintenue.

## Fichiers critiques à traiter
- `zemosaic_align_stack.py` (CPU)
- `zemosaic_align_stack_gpu.py` (GPU)
- `zemosaic_utils` (WCS, pivot, pad)
- éventuellement :
  - `parallel_utils`
  - `zemosaic_worker` (Phase 3 → Phase 5 transitions)

## Livrables attendus
1. Un patch annulant proprement les modifications 08bebf5 et rétablissant la version fonctionnelle V4.2.0.
2. Un rapport clair dans `followup.md` :
   - quelles parties ont été retirées ou restaurées,
   - pourquoi elles causaient la régression,
   - comment le pipeline retrouve sa cohérence.
3. Une vérification concrète :
   - preview.png OK
   - coverage.dat OK
   - aucune erreur WCS
   - mosaïque visuellement identique à V4.2.0.

Merci d’être chirurgical et de préserver toute la structure extérieure.
