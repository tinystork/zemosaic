# Follow-up – Audit géométrie / WCS / Align-stack

## Ce qu’il faut remplir après analyse et patches

### 1. Différences exactes trouvées entre V4.2.0 et V4WIP
- `zemosaic_align_stack.py` : ajout d’un pipeline “streaming” pour le Winsor (classe `WinsorStreamingState`), normalisation GPU optionnelle, et support explicite des plans parallèles (rows_per_chunk, memmap). Logique d’alignement inchangée (astroalign + pré-alignement FFT) et shape verrouillée sur la frame de référence.
- `zemosaic_align_stack_gpu.py` : nouveau module (absent en V4.2.0) qui stacke sur GPU par chunks de lignes (rows_per_chunk/gpu_max_chunk_bytes), combine en mean/median et applique la même égalisation RGB que la voie CPU.
- `zemosaic_worker.create_master_tile` : bascule possible vers le stack GPU, nouveau mode streaming pour tuiles “lourdes” (ingestion chunkée + `WinsorStreamingState.finalize`), plan RAM/memmap injecté dans la config, mais WCS toujours issu de l’en-tête de la frame de référence. Sauvegarde FITS conserve l’offset CRPIX appliqué lors du crop qualité.
- `zemosaic_worker` phase 5 : lecture des master tiles en réordonnant CHW→HWC, support GPU pour reproject/coadd (`should_use_gpu_for_reproject`, wrappers GPU), validations shape/WCS renforcées (pixel_shape vs shape_out). Photométrie inter-tuiles/clamp des gains ajoutés avant reprojection. 
- `zemosaic_utils.save_fits_image` : nettoyage systématique des NaN avant export float, baseline shift forcé à 0, gestion alpha inchangée. Pas de changement explicite sur CRPIX/CD/CRVAL dans les sauvegardes.

### 2. Cause de la régression
- Décrire précisément ce qui a rendu les master tiles de V4WIP incompatibles avec le reprojetage :
  - shape incorrecte ?
  - CRPIX modifié ?
  - offset perdu ?
  - pivot décalé ?
  - orientation inversée ?
  - masque non propagé ?
  - padding disparu ?

### 3. Correctifs appliqués
- Détail des modifications faites dans V4WIP pour restaurer :
  - géométrie exacte,
  - WCS stable,
  - comportement identique à V4.2.0.

### 4. Validation
- Rejouer le dataset :
  - [ ] plus aucun warning WCS,
  - [ ] génération de preview.png restaurée,
  - [ ] génération de coverage.dat restaurée,
  - [ ] mosaïque visuellement identique à V4.2.0,
  - [ ] patchwork disparu,
  - [ ] couleur homogène.

### 5. Notes finales
- Éventuels tests de non-régression,
- Conseils de maintenance,
- Sections du code nécessitant une future refactorisation propre.
- Prochaines étapes proposées : tracer un master tile de V4WIP vs V4.2.0 (shape/WCS/CRPIX), comparer le flux streaming vs non-streaming, et forcer temporairement le chemin CPU non-streaming pour vérifier l’impact sur les erreurs `WCS.all_world2pix`.

### Décisions temporaires (pour pouvoir revenir facilement)
- Streaming winsor Phase 3 désactivé par défaut : `zemosaic_align_stack.py` force `stack_disable_streaming`/`winsor_disable_streaming` à True si non défini ; `zemosaic_worker.create_master_tile` n’active le streaming que si `mastertile_allow_streaming`/`stack_allow_streaming`/`winsor_allow_streaming` est explicitement True.
- GPU désactivé par défaut : `zemosaic_config.py` met `use_gpu_global` et `use_gpu_phase5` à False ; `zemosaic_worker` force `use_gpu_phase5_flag=False` côté Phase 5.
- Crops désactivés par défaut : `zemosaic_config.py` met `apply_master_tile_crop=False` et `global_wcs_autocrop_enabled=False`; `zemosaic_worker` force `apply_master_tile_crop_config=False`/`apply_crop_for_assembly=False` et `global_wcs_autocrop_enabled_config=False`.
- Assemblage Phase 5 : master-tile crop forcé à 0% et no-autocrop, GPU reprojection forcée off.

### À restaurer si besoin
- Réactiver GPU : remettre `use_gpu_global` / `use_gpu_phase5` à True dans `zemosaic_config.py` (et autoriser `use_gpu_phase5_flag` dans `zemosaic_worker`).
- Réactiver crop MT : remettre `apply_master_tile_crop` à True dans `zemosaic_config.py` et enlever le forçage `apply_master_tile_crop_config=False` dans `zemosaic_worker`.
- Réactiver autocrop global : remettre `global_wcs_autocrop_enabled=True` dans `zemosaic_config.py` et enlever le forçage `global_wcs_autocrop_enabled_config=False` dans `zemosaic_worker`.
- Réactiver streaming Phase 3 : basculer `stack_disable_streaming`/`winsor_disable_streaming` à False dans `zemosaic_align_stack.py` ou activer `mastertile_allow_streaming` (ou `stack_allow_streaming`/`winsor_allow_streaming`) via la config.
