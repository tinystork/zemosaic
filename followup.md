Merci pour l’analyse et la restauration.

Maintenant, j’ai besoin de toi pour **valider et verrouiller** ce retour au comportement `38c876a` :

- [ ] **Vérifie que tu as bien appliqué le plan de `agent.md`** : logique Master Tile → mosaïque alignée `38c876a` côté CPU/GPU (Phase 5 + two-pass). Constats actuels :
  - En GPU, intertile auto-tune ne trouve que 65 paires (golden/CPU actuel : 87) et aucun log `apply_photometric` (gains/offsets) n’apparaît en CPU ni GPU alors qu’ils sont présents dans le log `38c876a` (gain 0.86464, offset -514.99 sur 27 tuiles).
  - Two-pass GPU : couverture canal 1 plate (min=max=1) et gains tous à 1.0 ; GPU n’est utilisé que pour le premier canal alors que la CPU calcule des gains 0.949–0.997 et le golden ne montre pas ces plateaux. À réaligner strictement sur `38c876a`.
  - Dernier run visuel : sortie GPU pleine de trames verticales (image 1), sortie CPU toujours avec bandes / bords noirs (image 2) ; image 3 reste la référence attendue.
- [ ] **Exécute un “golden run” de validation** (dataset `zemosaic_worker38c876a.log`) : runs CPU-only et GPU, comparer phases/shape/logs aux valeurs de référence.
- [x] **Ajoute un test automatisé** : comparer deux mosaïques CPU/GPU via différence moyenne/RMS (RMS << dynamique).
- [x] **Documente brièvement dans le code** : blocs restaurés “38c876a” + exigence de validation CPU/GPU avant toute modification.
- [ ] **Résume dans ta réponse** : fichiers modifiés, fonctions alignées, résultats des tests CPU vs GPU, confirmation absence dérive verte/bandes.

Ne tente pas d’optimiser davantage la Phase 5 tant que cette étape de “retour à 38c876a” n’est pas validée.

Prochaines actions concrètes pour réaligner GPU/CPU sur `38c876a` (sans toucher au mode SDS) :
- [ ] Restaurer l’intertile photometric solve (paires ~87) : corriger la régression `expected_min_pairs` non défini dans `compute_intertile_affine_calibration` et réaligner la recherche d’overlaps GPU pour retrouver 87 paires, puis vérifier que `apply_photometric` logge bien les gains/offsets sur 27 tuiles. (Progression : `estimate_overlap_pairs` rétabli au code 38c876a ; validation run encore à faire.)
- [ ] Corriger la passe two-pass GPU : coverage d’entrée saturée (min=0, max=66, mean≈56) et gains tous à 1.0 → aligner la coverage map et la normalisation sur le flux CPU/golden (gains 0.949–0.997) et traiter les 3 canaux. (Progression : `compute_per_tile_gains_from_coverage` et `run_second_pass_coverage_renorm` remis au comportement 38c876a, validation GPU/CPU à rejouer.)
- [ ] Revérifier les flags Phase 5 (match_bg/intertile_photometric_match/tile_weighting) côté GPU pour éviter la réduction à 65 paires et garantir le même ordre d’opérations que `38c876a`.
- [ ] Rejouer les runs CPU/GPU sur le dataset de référence et comparer aux logs `zemosaic_worker38c876a.log` (phases, shapes, RGB-EQ, TwoPass), avec capture des stats coverage/gains par canal.
