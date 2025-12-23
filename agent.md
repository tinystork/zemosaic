# Mission: Étape 2 — Propager correctement tile_weight dans le GPU coadd (Phase5 + TwoPass), avec parité CPU/GPU

## Contexte
On a des master tiles très déséquilibrées en profondeur (ex: tile_002 très profonde, tile_000/001 très faibles).
Objectif: en zones de recouvrement, la mosaïque doit être dominée par la tuile la plus profonde (meilleur SNR),
et le comportement doit être cohérent entre CPU et GPU.

Le code a déjà des notions de:
- input_weights (alpha/coverage 2D)
- tile_weights (scalaire par tuile)
- deux chemins: assemble_final_mosaic_reproject_coadd (Phase 5) + run_second_pass_coverage_renorm (TwoPass)

Le but de ce patch est de s'assurer que:
1) GPU coadd utilise réellement tile_weight (pas seulement CPU),
2) Phase 5 ET TwoPass passent les mêmes infos au backend,
3) pas de double-application involontaire (tile_weight intégré dans input_weights + tile_weights en même temps),
4) tests CPU vs GPU.

## Contraintes
- Patch chirurgical ("no refactor")
- Ne pas changer le comportement batch_size=0 vs >1
- Ne pas casser perf/mémoire GPU
- Ajout de logs DEBUG ciblés OK

## Plan (à implémenter)

### A) Worker: passer tile_weights séparément, garder input_weights "purs" (alpha/coverage seulement)
Fichiers: `zemosaic_worker.py`

1) Dans `assemble_final_mosaic_reproject_coadd` (Phase 5):
   - [x] Construire `tile_weights = [float(entry.get("tile_weight", 1.0)) ...]` aligné avec data_list/wcs_list.
   - [x] Construire `input_weights_list` à partir du masque (alpha_weight2d / coverage) SANS multiplier par tile_weight.
   - [x] Appeler `reproject_and_coadd_wrapper(..., input_weights=input_weights_list, tile_weights=tile_weights, ...)`.
   - [x] Ajouter un log DEBUG unique:
     - min/median/max des tile_weights + ratio max/min
     - vérifier si input_weights a des max > 1.5 et tile_weights != None -> warning "possible double weighting".

2) Dans `run_second_pass_coverage_renorm` / `_process_channel` (TwoPass reprojection):
   - [x] Même logique: `tile_weights` transmis au coadd wrapper GPU.
   - [x] `input_weights` = poids 2D (alpha/coverage/scale-map) sans tile_weight.

### B) Backend GPU: vérifier l’utilisation effective de tile_weights
Fichier: `zemosaic_utils.py`

1) Dans `gpu_reproject_and_coadd_impl`:
   - [x] Vérifier que `tile_weights_param` est bien lu et converti en `tile_weights_gpu`.
   - [x] S’assurer que `tile_weight` multiplie effectivement la contribution lors de l’accumulation
     (mean / winsorized / kappa-sigma).
   - [x] Ajouter un log DEBUG (une fois) listant les poids normalisés utilisés côté GPU.

2) Dans le wrapper `reproject_and_coadd_wrapper`:
   - [x] Ne pas changer la logique globale,
   - [x] mais ajouter une protection (DEBUG/WARN) si:
     - `input_weights` semble déjà intégrer des poids > 1 (ex: max >> 1) ET tile_weights fourni.
     - -> log: "double application probable".

### C) Tests CPU vs GPU (mini test synthétique)
Créer `tests/test_tile_weight_gpu_coadd.py` (ou un script de test si pas de pytest dans le repo).

Test 1 (mean):
- [x] Deux tuiles partiellement recouvrantes (même WCS ou WCS identique + simple décalage pixel).
- [x] Tuile A: bruit (random) moyenne ~0
- [x] Tuile B: signal constant (ex: +100) + petit bruit
- [x] tile_weights: A=1, B=100
- [x] Résultat attendu en zone overlap: proche de tuile B (erreur faible), et GPU≈CPU.

Test 2 (winsorized):
- [x] Même setup mais `combine_function="mean"` + `stack_reject_algo="winsorized"` (ou combine="winsorized" selon API).
- [x] Attendu: même dominance de la tuile B.

Critères:
- [x] Dans l’overlap, moyenne(result - B) < 1% du signal (tolérance à ajuster)
- [x] GPU vs CPU: différence RMS faible (tolérance ~1e-3 à 1e-2 selon float32)

## Livraison attendue
- Patch git sur `zemosaic_worker.py` + `zemosaic_utils.py`
- Test(s) + instructions de lancement
- Logs de run montrant:
  - tile_weights summary
  - confirmation GPU tile_weights utilisés
  - plus aucune ambiguïté "tile_weight intégré dans input_weights"
