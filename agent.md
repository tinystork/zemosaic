# Mission Codex — Masque “overlap-aware” (coverage intra-cluster) sur Master Tiles

## Problème
Les master tiles peuvent présenter des franges / dominantes chromatiques en bordure quand certaines zones du tile sont construites avec **peu de brutes contributrices** (recouvrement partiel intra-cluster, dither/rotations, etc.).  
`lecropper.py` corrige bien des artefacts (Alt-Az / coins), mais ne cible pas directement le “support faible” intra-cluster.

## Objectif (low cost / gros gain)
Sans réécrire le stacker et **sans ajouter de comportement nouveau hors Master Tiles**, mitiger ces franges en :
1) construisant une **carte de coverage intra-cluster** (par pixel = nb de brutes qui contribuent réellement) au moment du stack d’un master tile,  
2) passant cette coverage à `lecropper.mask_altaz_artifacts(..., coverage=...)` pour générer un **alpha mask** (morpho + feather) qui met à NaN / atténue les zones sous-recouvrées.

## Contraintes anti-régression (à respecter)
- **Pas d’effet de bord** : si `coverage_count_hw` est indisponible (None / erreur / mismatch), le comportement doit rester identique à l’existant (fallback lecropper radial/low-signal).
- **Scope** : activer uniquement sur le chemin Master Tile (ne pas impacter Phase 4.5 / final mosaic, sauf demande explicite).
- **Mode “I'm using master tiles”** : ne pas impacter `use_existing_master_tiles_mode` / `existing_master_tiles_mode` (skip clustering & master tile creation) ; la coverage intra-cluster est indisponible dans ce mode, donc la mission doit rester no-op (coverage=None → fallback).
- **Compat API** : conserver `_apply_lecropper_pipeline(arr, cfg)` appelable avec 2 arguments (tous les nouveaux paramètres doivent être optionnels et idéalement keyword-only).
- **Pas de “double mask”** : respecter la note existante dans `_apply_lecropper_pipeline` (ne pas remplacer `out` par `masked` quand `mask2d` est propagé).
- **Shape strict** : ne jamais “forcer” une coverage au mauvais shape (pas de resize implicite). En cas de mismatch → log explicite + ignore coverage.
- **NaN safe** : n’injecter des NaN que sur des buffers float32 ; si dtype non-float, convertir (copie) ou ignorer la nanisation pour éviter un cast silencieux.
- **Pas de spam d’erreurs stack** : `stack_aligned_images` loggue “non-finis” en ERROR quand des NaN arrivent en entrée (`zemosaic_align_stack.py` ~3530). Si tu nanises pour coverage, nettoie (`np.nan_to_num`) avant `_stack_master_tile_auto` **ou** calcule la coverage via un masque séparé.
- **Perf** : calcul coverage en accumulation (pas de `np.stack(valid2d)` géant qui explose la RAM).

## Notes de cohérence avec le code actuel (éviter les surprises)
- `lecropper.mask_altaz_artifacts` supporte déjà `coverage`, `min_coverage_abs/frac`, `morph_open_px` (`lecropper.py:647`).
- Dans ce repo, `lecropper.py` **ne définit pas** `quality_crop` ; le risque principal de mismatch de shape vient donc des crops Master Tile existants (`quality_crop_rect`) plutôt que d’un crop interne à `_apply_lecropper_pipeline`.
- `_apply_lecropper_pipeline` lit `altaz_alpha_soft_threshold` / `altaz_nanize_threshold`, mais l’appel courant à `mask_altaz_artifacts` ne passe pas `hard_threshold` : éviter d’introduire un seuil redondant, et garder `altaz_nanize_threshold` cohérent avec `altaz_alpha_soft_threshold` si ces options sont exposées.

## Clarifications importantes (pour éviter une régression)
- **Les NaN ne “réparent” pas le stack** : le stack CPU/GPU remplace les non-finis par `0.0` avant la combinaison (ex: `zemosaic_align_stack.py:3540`, `zemosaic_align_stack_gpu.py:395`).  
  ⇒ La nanisation hors-footprint sert ici surtout à **mesurer une coverage fiable** et à produire un **alpha mask** via `lecropper` (effet après stack), pas à modifier la combinaison des brutes.
- **Éviter les logs `STACK_IMG_PREP`** : si des NaN sont présents, `stack_aligned_images` remplace par `0.0` **et** loggue en ERROR. Donc naniser pour la coverage oui, mais nettoyer avant le stack (ou utiliser un masque séparé).
- **Ne pas injecter de NaN dans l’entrée d’astroalign** : le pré-alignement FFT (`prealign_fft_img`) sert de source à `astroalign_module.register(...)`. Garder ce buffer fini (`0.0` hors-overlap) pour éviter une régression de détection/correspondance.
- **Masque astroalign robuste** : `footprint_mask` peut être `None`, 2D ou 3D (bool ou float). Réduire en 2D en s’alignant sur `aligned_image_output.shape` (HWC/CHW), binariser (`>0`) si float. Si ambigu / mismatch → log + ignore coverage.
- **Garder le coût optionnel** : ne calculer coverage / `propagate_mask=True` que si l’Alt-Az cleanup est effectivement activé sur Master Tiles (et si `lecropper` est dispo).
- **Portée vs modes `GRID` et `SDS/SupadupStack`** :
  - `GRID` : le worker bascule vers `grid_mode.run_grid_mode(...)` puis `return` (`zemosaic_worker.py:21028`). Le chemin Master Tile n’est pas exécuté.
  - `SDS/SupadupStack` : le pipeline principal passe par `assemble_global_mosaic_sds(...)` et “skip master tiles” (`zemosaic_worker.py:19050`). La création de Master Tiles n’a lieu qu’en cas de fallback (si SDS + Mosaic-First échouent) (`zemosaic_worker.py:23554`).
  - “I'm using master tiles” : le worker passe en `use_existing_master_tiles_mode` (skip clustering & master tile creation). Aucune brute intra-cluster → pas de coverage possible, donc la mission doit rester no-op.

## Applicabilité (mode de run à vérifier via logs)
Cette mission ne s’applique **que** si ZeMosaic **crée** des Master Tiles (Phase 3).
- Mission active : logs Phase 3 (`PHASE_UPDATE:3`, `Phase 3: Master Tiles`) + logs master tile (`MT_PIPELINE_FLAGS`, `MT_PIPELINE`, `ALPHA_STATS: level=master_tile`).
- Mission inactive :
  - `GRID` : log `[GRID] Invoking grid_mode.run_grid_mode(...)` puis retour (pas de Phase 3).
  - `SDS` (succès) : log `[SDS] Phase 5 polish ... (skipping master tiles)` (pas de Phase 3) ; si SDS fallback → log `sds_and_mosaic_first_failed_fallback_mastertiles` puis Phase 3 (mission active).
  - “I'm using master tiles” : log `run_info_existing_master_tiles_mode` / `existing_master_tiles_mode:` ; pas de Phase 3 (mission inactive).

## Points d’injection exacts (repères avec lignes)
> Note : les numéros de ligne sont indicatifs ; préférer chercher les snippets cités si le fichier a bougé.

### 1) Là où tu as la liste des brutes alignées (idéal pour calculer la coverage)
- `zemosaic_worker.py:12133` : `valid_aligned_images = [img for img in aligned_images_for_stack if img is not None]`
  - C’est le meilleur endroit pour produire `coverage_count_hw` *avant* `_stack_master_tile_auto(...)` (`zemosaic_worker.py:12171`).
  - Vérifier que toutes les images ont la même shape; sinon log + skip la coverage pour éviter un masque corrompu.
  - Si tu as nanisé les images pour la coverage, **nettoyer** (`np.nan_to_num`) juste après le calcul coverage, avant `_stack_master_tile_auto` (évite les logs ERROR du stacker).

### 2) La source “vraie” de pixels valides pendant l’alignement (important)
⚠️ Les bords hors-footprint peuvent être remplis avec des **0.0** (valeurs finies) par l’alignement → un simple `isfinite()` ne suffit pas pour un coverage correct.

- `zemosaic_align_stack.py:2075` : `align_images_in_group(...)`
  - À l’intérieur, l’appel `astroalign_module.register(...)` retourne `aligned_image_output, footprint_mask`.
  - **Préférer un enable local Master Tile** : `align_images_in_group` a déjà un param `propagate_mask`; le call-site Master Tile ne le passe pas actuellement → passer `propagate_mask=True` à cet appel (plutôt que changer un default global), puis exploiter `footprint_mask` (uniquement si la coverage est requise).
  - Reconstruire un footprint 2D robuste :
    - `footprint_mask` peut être 2D ou 3D, bool ou float. Si 3D, réduire sur la dimension “canaux” en s’alignant sur `aligned_image_output.shape` (ex: axis=-1 si HWC, axis=0 si CHW). Si float, binariser (`>0`). Si ambigu/mismatch → log + ignore coverage.
  - **Naniser hors-footprint** *après* `astroalign.register` (sur `aligned_image_output`, pas sur `src_for_aa`) : `aligned_image_output[~footprint2d] = np.nan` → coverage fiable via `np.isfinite`. S’assurer que `aligned_image_output` est float32 (sinon copier/convertir). Si `footprint_mask` est absent/invalide → ne pas naniser, et considérer la coverage indisponible (fallback).
  - **Important** : ne pas laisser ces NaN partir au stack (sinon logs `STACK_IMG_PREP`). Nettoyer (`np.nan_to_num`) après calcul coverage **ou** calculer la coverage via `footprint2d` sans modifier l’image.
  - En **FFT-only** (astroalign indisponible / repli), il faut aussi pouvoir distinguer “hors overlap” des pixels réels : naniser hors-overlap sur la **sortie retournée** (pas sur le buffer utilisé comme entrée d’astroalign), puis nettoyer avant stack si nécessaire.

### 3) Où injecter la coverage dans la pipeline lecropper (master tiles)
- `zemosaic_worker.py:665` : `_apply_lecropper_pipeline(...)`
  - L’appel actuel à modifier est `masked, mask2d = mask_helper(..., return_mask=True)` (`zemosaic_worker.py:752`).
  - Étendre la signature (optionnelle) pour accepter `coverage` + seuils coverage (`altaz_min_coverage_abs/frac`, `altaz_morph_open_px`). Defaults conservateurs (calés sur `lecropper`) pour ne rien casser sur les autres call-sites (Phase 4.5, pipeline final).
  - Compat lecropper : si l’implémentation locale/ancienne de `mask_altaz_artifacts` ne supporte pas `coverage`/seuils, attraper `TypeError` et refaire l’appel sans ces kwargs (comportement inchangé).
- `zemosaic_worker.py:12441` : appel master-tile
  - `master_tile_stacked_HWC, pipeline_alpha_mask = _apply_lecropper_pipeline(master_tile_stacked_HWC, pipeline_cfg)`
  - C’est ici qu’il faut passer la coverage calculée pour ce tile (ex: param optionnel) + seuils coverage depuis `pipeline_cfg`.
- `lecropper.py:647` : `mask_altaz_artifacts(..., coverage=..., min_coverage_abs=..., min_coverage_frac=..., morph_open_px=...)`
  - Les défauts sont `MIN_COVERAGE_ABS_DEFAULT=3.0` (`lecropper.py:152`) et `MIN_COVERAGE_FRAC_DEFAULT=0.4` (`lecropper.py:153`).

### 4) Attention aux crops avant la pipeline (shape-matching)
Si un crop est appliqué au master tile avant `_apply_lecropper_pipeline`, la coverage doit être cropée de la même façon, sinon `lecropper` ignore la coverage (mismatch de shape).

- `zemosaic_worker.py:12379` : `quality_crop_rect = (y0, x0, y1, x1)` est défini et `master_tile_stacked_HWC` est cropé.
  - Action : appliquer exactement le même rect à `coverage_count_hw` avant de l’envoyer à la pipeline.
  - Si un autre crop intervient plus tard (ex: futur `lecropper.quality_crop`), la coverage doit suivre *le même rect* ; sinon log mismatch + ignore coverage.

## Implémentation recommandée (minimale)
1) [x] **Alignement : rendre les zones hors-footprint “non-finite”**
   - Activer `propagate_mask=True` **uniquement** sur le call-site Master Tile vers `align_images_in_group` *et seulement si* l’Alt-Az cleanup Master Tile est actif (évite un coût inutile).
   - Dans `zemosaic_align_stack.py:2075` (`align_images_in_group`), après `astroalign_module.register(...)` :
     - convertir le footprint en masque 2D (gérer `footprint_mask` 2D/3D, bool/float) en s’alignant sur `aligned_image_output.shape` ; si float, binariser (`>0`) ; si ambigu → log + ignore coverage,
     - naniser hors-footprint (`aligned_image_output[~footprint2d] = np.nan`) **uniquement** si `aligned_image_output` est float32 (sinon copier/convertir avant).
     - si tu choisis la nanisation pour produire la coverage, **nettoyer** (`np.nan_to_num`) les images avant de les envoyer au stacker, pour éviter les logs ERROR et rester au comportement existant.
   - FFT-only / fallback : naniser hors-overlap sur l’image *retournée* pour permettre un coverage fiable (l’overlap est un rectangle déduit de `dy/dx` et de la shape), mais **ne pas** naniser `prealign_fft_img` avant l’appel à `astroalign.register`.

2) [x] **Coverage intra-cluster : calcul au plus près des brutes alignées**
   - Dans `zemosaic_worker.py` juste après `valid_aligned_images` (`zemosaic_worker.py:12133`) :
     - Vérifier que toutes les images ont la même shape; si mismatch : log + `coverage_count_hw=None`.
     - Calcul recommandé (incrémental, faible RAM) :
       - `coverage_count_hw = np.zeros((H, W), dtype=np.float32)`
       - par image : `valid2d = np.all(np.isfinite(img), axis=-1)` (ou `np.isfinite(img)` si 2D) puis `coverage_count_hw += valid2d.astype(np.float32)`
       - si `footprint2d` est disponible, préférer `valid2d = footprint2d` (ou `footprint2d & np.isfinite(img)`), pour éviter de dépendre des NaN.
     - Logger min/max/nonzero_frac; clamp `min_coverage_abs` à `n_used_for_stack` pour éviter un masque vide en salvage.
     - Si `coverage_count_hw` est vide/non-finie ou `max_coverage <= 0` → log + `coverage_count_hw=None` (évite un masque total).

3) [x] **Propager la coverage jusqu’à la pipeline**
   - Conserver `coverage_count_hw` en variable locale jusqu’au bloc pipeline.
   - Si `quality_crop_rect` est appliqué (`zemosaic_worker.py:12379`), cropper `coverage_count_hw` pareil.
   - Toujours vérifier `coverage.shape == image.shape[:2]` et `coverage.max() > 0` juste avant l’appel lecropper ; sinon log + `coverage=None`.

4) [x] **Passer la coverage à lecropper**
   - Étendre `_apply_lecropper_pipeline` pour accepter `coverage` optionnel (le `nanize` reste contrôlé par `altaz_nanize_threshold` dans `cfg`).
   - Passer `coverage=...` + seuils à `lecropper.mask_altaz_artifacts` (kwargs). En cas de `TypeError`, refaire l’appel sans `coverage`/seuils.
   - Ajouter 3 paramètres dans `pipeline_cfg` (ou valeurs codées temporairement) :
     - `altaz_min_coverage_frac` (**fraction du max coverage**, cf. `thr = max(abs, frac * max_cov)` dans `lecropper`) — reco départ: `0.4–0.6`
     - `altaz_min_coverage_abs` (reco: `2–4`, borné par `n_used_for_stack`)
     - `altaz_morph_open_px` (reco: `2–4`)

## Réglages conseillés (éviter “full overlap == N”)
- `min_coverage_abs`: `3` (défaut lecropper) puis ajuster `2–4` selon `n_used_for_stack` (borne supérieure = `n_used_for_stack`)
- `min_coverage_frac`: commencer proche du défaut `0.4`, puis monter si nécessaire (`0.5–0.7`) en surveillant la perte de champ
- `morph_open_px`: `3` (défaut), ajuster `2–4` si îlots/filaments
- RGB: considérer un pixel “contribué” seulement si **R,G,B** sont valides (sinon franges).

## Garde-fous / validation
- Toujours logguer `ALPHA_STATS` + stats coverage (min/max/nonzero_frac + `n_used_for_stack`) pour tracer les effets du masque.
- En cas de mismatch de shape coverage vs image à l’entrée de lecropper → log explicite + ignore coverage (retour au comportement existant).
- Logguer explicitement quand la coverage est ignorée (mismatch, max<=0, dtype non-float, footprint ambigu) pour diagnostiquer un fallback.
- Surveiller l’impact indirect via `MT_EDGE_TRIM` (il utilise `pipeline_alpha_mask > 0` comme validité) : un masque trop agressif peut augmenter le trim.
- `altaz_alpha_soft_threshold` / `altaz_nanize_threshold` doivent rester cohérents (mêmes ordres de grandeur) pour éviter des seuils contradictoires.

## Logs debug (overlap-aware) — où et comment
- `zemosaic_align_stack.py:2075` (après `astroalign.register`) : via `_pcb(..., lvl="DEBUG_DETAIL")`, logger `img_idx`, `propagate_mask`, `footprint_shape`, `dtype`, `nonzero_frac`, et si `footprint2d` est ignoré (raison).
- `zemosaic_align_stack.py` FFT-only fallback : logger `dy/dx`, `overlap_rect`, `overlap_frac`, et si un masque coverage est dérivé.
- `zemosaic_worker.py:12133` (après coverage) : via `pcb_tile(..., lvl="DEBUG_DETAIL")`, logger `tile_id`, `coverage_shape`, `min/max`, `nonzero_frac`, `n_used_for_stack` (tag reco: `MT_COVERAGE_STATS`) + un log explicite quand la coverage est ignorée (tag reco: `MT_COVERAGE_IGNORED`, avec la raison).
- `zemosaic_worker.py:12441` (avant `_apply_lecropper_pipeline`) : logger `coverage_max`, seuils `altaz_min_coverage_abs/frac`, `altaz_morph_open_px`; ensuite s’appuyer sur `ALPHA_STATS`/`MT_PIPELINE` existants.
- Reco : limiter ces logs aux `debug_tile` pour éviter du bruit en production.

## Validation / Definition of Done
- Visuel : réduction nette des franges/dominantes chromatiques sur bords master tiles (zones faible coverage).
- Technique :
  - `pipeline_alpha_mask` reflète la coverage (bords sous-recouvrés → alpha proche 0).
  - Pas de hard-crop **supplémentaire** requis (WCS/CRPIX inchangés hors crops existants).
  - Aucun “double application” du masque (respecter le commentaire existant dans `_apply_lecropper_pipeline`).
- [x] Tests : lancer `pytest -q` et vérifier au minimum les tests master tile (`tests/test_create_master_tile_*.py`).
