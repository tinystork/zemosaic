# Follow-up — Rollback sélectif du commit 08bebf5

## 1. Résumé du commit fautif
- Flags ajoutés post-08bebf5 ont forcé des désactivations : `use_gpu_global/use_gpu_phase5` ramenés à False dans `DEFAULT_CONFIG`, streaming Winsor coupé par défaut, `apply_master_tile_crop` et `global_wcs_autocrop_enabled` remis à False côté worker/config.
- Ces forçages ont modifié la géométrie des master tiles (plus de crop/autocrop), la stratégie mémoire du stacking (streaming bloqué) et la capacité à utiliser le GPU pendant l’assemblage.

## 2. Ce qui a causé la régression
- WCS / géométrie : désactivation de l’autocrop global et du rognage master tile -> master tiles non recadrées, offsets décalés.
- Perf / pipeline : streaming Winsor forcé en mode « in-memory » via `stack_disable_streaming=True` par défaut -> change le plan mémoire et la couverture produite.
- GPU : chemins GPU neutralisés par défaut (`use_gpu_*` à False) donc phase 5 CPU-only même quand le hardware est présent.

## 3. Correctifs appliqués
- `zemosaic_align_stack.py` : le flag `stack_disable_streaming` ne force plus la désactivation (fallback par défaut = streaming autorisé).
- `zemosaic_worker.py` : restauration des valeurs de Phase 5 (`apply_master_tile_crop`, `apply_crop_for_assembly`, `master_tile_crop_percent`, `use_gpu_phase5_flag`), réactivation de l’autocrop global, opt-in streaming par défaut, préférences GPU résolues en True par défaut.
- `zemosaic_config.py` : defaults rétablis (GPU on, autocrop on, crop master on) et normalisation GPU fallback → True pour éviter de couper le GPU quand les clés sont absentes.
- Les chemins GPU/CPU restent en place (fallbacks conservés), seules les valeurs par défaut et les forçages ont été corrigés.

## 4. Décisions temporaires retirées
- [x] Streaming remis selon V4.2.0 (opt-in actif par défaut, plus de disable forcé).
- [x] GPU réactivé selon la configuration (defaults V4.2.0 rétablis et résolution GPU optimistic).
- [x] Crops restaurés (apply_master_tile_crop / apply_crop_for_assembly repris des options Phase 5).
- [x] Autocrop global restauré (lecture du flag config, défaut à True).
- [x] Assemblage Phase 5 réautorise crop/autocrop/GPU selon la config.
- [x] Seuil de clustering par défaut rétabli à 0.05° (fallback V4.2.0 au lieu de 0.12°).

## 5. Validation (cases à cocher)

- [ ] WCS errors disparues (`all_world2pix` stable)
- [ ] preview.png généré
- [ ] coverage_xxx.dat généré
- [ ] Mosaïque visuellement identique à V4.2.0
- [ ] Pas de patchwork
- [ ] alignement correct sur les bords
- [ ] Phase 5 CPU et GPU identiques
- [ ] aucunes régressions additionnelles

## 6. Notes éventuelles
- Traces de code qui méritent une refactorisation future.
- Suggestions de tests automatiques.
- Validation non rejouée dans cette itération : les cases de la section 5 restent à confirmer après exécution complète (preview/coverage/WCS).
