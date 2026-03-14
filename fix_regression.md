# fix_regression.md

## Mission
Rétablir le fonctionnement des **3 voies exclusives** après le refactor (héritage `agent.md` / `followup.md` / `memory.md`) **sans casser d'autres chemins**:
- voie classique
- voie grid_mode
- voie SDS

⚠️ Règle de cadrage: ces modes sont **mutuellement exclusifs**. Ils ne doivent pas coexister dans un même run.

Contrainte de mission:
- correctifs chirurgicaux
- preuve avant/après
- non-régression explicite sur les chemins non ciblés

---

## Contexte initial (2026-03-14)
Erreur remontée sur dernier run SDS:
- `NameError: name 'existing_master_tiles_results' is not defined`
- trace observée dans `run_hierarchical_mosaic` (`zemosaic_worker.py:26488`)

Observation clé:
- Le run va jusqu'à la fin de la Phase 2, puis crash au passage vers la phase suivante.

---

## Clarification architecture modes (ajout 2026-03-14 09:29)
Les chemins d'exécution à considérer sont:
1. **Mode classique**
2. **Mode grid_mode**
3. **Mode SDS**

Ces 3 modes sont distincts et exclusifs:
- pas de scénario “SDS + grid_mode”
- pas de mélange de logique entre branches
- la validation doit se faire **mode par mode**, pas en combinatoire croisée SDS/grid.

---

## Journal d'itérations

### Iteration 1 — Audit initial et cadrage (2026-03-14 09:25)
**Fait:**
1. Lecture des fichiers de continuité mission:
   - `agent.md`
   - `followup.md`
   - `memory.md`
2. Lecture du log du run SDS en échec (`zemosaic_worker.log`).
3. Inspection statique de `zemosaic_worker.py` autour de la stacktrace.
4. Vérification des occurrences de `existing_master_tiles_results` dans le fichier.

**Constats techniques:**
- Dans `run_hierarchical_mosaic` (nouveau chemin), `existing_master_tiles_results` est **utilisée** mais non définie localement avant usage.
- Dans `run_hierarchical_mosaic_classic_legacy`, la variable est bien initialisée en amont.
- Symptomatique d'une régression de refactor/copier-coller entre chemins legacy et nouveau pipeline.

**Impact probable:**
- crash précoce quand le code atteint l'initialisation `master_tiles_results_list = list(existing_master_tiles_results)`.
- risque de toucher plusieurs voies d'exécution si ce bloc est partagé.

**Ce qui reste à faire:**
1. Localiser le meilleur point d'initialisation/garde de cette variable dans `run_hierarchical_mosaic`.
2. Corriger minimalement la portée/initialisation sans modifier la logique scientifique.
3. Vérifier les 3 voies exclusives (classique / grid / SDS) pour éviter une casse collatérale.
4. Ajouter un test de non-régression ciblant ce scénario (`NameError` impossible).
5. Documenter preuve avant/après ici et dans `memory.md`.

---

## Plan d'action (amendé)
1. **Repro contrôlée**
   - rejouer le scénario SDS qui crash
   - confirmer le point exact de rupture
2. **Patch minimal de portée**
   - introduire une initialisation sûre de `existing_master_tiles_results` dans `run_hierarchical_mosaic`
   - éviter toute modification de comportement hors besoin
3. **Validation par mode (exclusifs)**
   - Run en voie classique
   - Run en voie grid_mode
   - Run en voie SDS
4. **Tests de non-régression**
   - ajout test dédié contre `NameError` sur ce bloc
   - exécution des tests ciblés puis subset pertinent
5. **Clôture de correction**
   - résumé du diff
   - risques résiduels
   - checklist “fait / restant” à jour

---

## Checklist opérationnelle
- [x] Audit des logs du run SDS en échec
- [x] Lecture `agent.md` / `followup.md` / `memory.md`
- [x] Identification du point de rupture (`existing_master_tiles_results`)
- [x] Cadrage explicite: 3 modes exclusifs (classique / grid / SDS)
- [x] Correctif code appliqué
- [ ] Repro validée post-fix (plus de NameError)
- [ ] Validation multi-mode non-régression (classique / grid / SDS)
- [x] Test automatisé ajouté
- [ ] Clôture et preuve finale documentées

### Iteration 2 — Correctif SDS ciblé `NameError` + garde de non-régression (2026-03-14 09:33)
**Fait:**
1. Correctif appliqué dans `run_hierarchical_mosaic` (chemin SDS/grid):
   - remplacement de l'initialisation fautive
   - `master_tiles_results_list` est maintenant initialisée explicitement à `[]`.
2. Vérification de l'autre occurrence dans le chemin legacy:
   - conservée inchangée (`list(existing_master_tiles_results)`) pour ne pas altérer la voie classique legacy.
3. Ajout d'un test source-contract de non-régression:
   - `test_run_hierarchical_master_tiles_bootstrap_avoids_undefined_existing_tiles_symbol`
   - garantit que `run_hierarchical_mosaic` ne dépend plus du symbole non défini.

**Validation effectuée:**
- `python3 -m py_compile zemosaic_worker.py`
- `python3 -m py_compile tests/test_phase3_adaptive_invariants.py`
- `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "run_hierarchical_master_tiles_bootstrap_avoids_undefined_existing_tiles_symbol"` -> `1 passed`

**Note de traçabilité:**
- une première édition a touché la mauvaise occurrence (legacy), immédiatement corrigée dans la même itération pour isoler strictement le patch sur le chemin SDS/grid.

**Ce qui reste à faire:**
1. Rejouer un run SDS réel via GUI (toggle SDS actif) et confirmer disparition du `NameError`.
2. Vérifier que la voie classique et la voie grid_mode ne régressent pas (smoke rapide par mode).
3. Documenter preuve finale avant clôture.

### Iteration 3 — Fix SDS broadcast mismatch en Phase 5 polish (2026-03-14 10:11)
**Contexte erreur utilisateur:**
- `ValueError: operands could not be broadcast together with shapes (1099,32) (3330,3748) ()`
- traceback dans `_apply_final_mosaic_quality_pipeline` pendant `_finalize_sds_global_mosaic`.

**Diagnostic:**
- En SDS global polish, la quality-crop (lecropper) peut recadrer `final_mosaic_data` (ex: `1099x32`).
- La coverage map reste sur la géométrie globale SDS (`3330x3748`).
- Puis `np.where(keep_mask > 0, final_mosaic_coverage, 0.0)` échoue par mismatch de dimensions.

**Correctif appliqué (ciblé SDS):**
- Dans `_finalize_sds_global_mosaic`, création d'une copie locale `sds_pipeline_cfg`.
- Désactivation explicite de `quality_crop_enabled` uniquement pour le polish SDS global.
- Ajout d'un log `phase5_sds_quality_crop_disabled` pour traçabilité.

**Pourquoi ce choix:**
- Le mode SDS s'appuie sur une géométrie globale (descriptor WCS) qui doit rester cohérente entre mosaic/coverage/alpha.
- Le quality-crop est géométrie-changing; en SDS global il introduit une incohérence de shape.
- Correctif chirurgical: on préserve alt-az cleanup et le reste du pipeline, on neutralise seulement la partie recadrage en SDS polish.

**Validation effectuée:**
- `python3 -m py_compile zemosaic_worker.py tests/test_phase3_adaptive_invariants.py`
- `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "run_hierarchical_master_tiles_bootstrap_avoids_undefined_existing_tiles_symbol or sds_finalize_disables_geometry_changing_quality_crop_in_phase5_polish"` -> `2 passed`

**Reste à faire:**
1. Rejouer run SDS via GUI (toggle SDS actif) et confirmer disparition de l'erreur broadcast.
2. Vérifier sortie finale SDS générée correctement.
3. Puis smoke rapide classique/grid.

### Iteration 4 — SDS VRAM hardening sur global coadd helper GPU (2026-03-14 10:38)
**Contexte log:**
- répétitions de warnings en SDS:
  - `GPU helper error: Out of memory allocating ...`
  - fallback immédiat CPU via `helper_failed`.

**Diagnostic:**
- Ce n'est pas une fuite mémoire prouvée dans le log; c'est un OOM runtime sur `gpu_reproject` avec hints chunk/rows trop agressifs pour certaines passes/canaux.
- Le chemin helper SDS n'avait pas la même logique de retry+tightening adaptatif que d'autres chemins GPU (voie classique/P3).

**Correctif appliqué (priorité VRAM):**
- Dans `_attempt_gpu_helper_route`:
  1. Ajout d'un retry GPU local par canal (`max_gpu_helper_retries = 3`).
  2. Sur OOM détecté (`_is_gpu_oom_error`):
     - purge des pools CuPy (`free_cupy_memory_pools`) si dispo,
     - réduction adaptative des hints GPU:
       - `rows_per_chunk` divisé par 2 (plancher 32),
       - `max_chunk_bytes` divisé par 2 (plancher 32MB),
     - réessai GPU avant fallback CPU global.
  3. Persistance intra-run des hints réduits via `plan_rows_gpu_hint` / `plan_chunk_gpu_hint` (pour les canaux suivants du même lot).
  4. Ajout de log de traçabilité: `global_coadd_gpu_oom_retry`.

**Validation effectuée:**
- `python3 -m py_compile zemosaic_worker.py`
- test source-contract ajouté:
  - `test_sds_global_gpu_helper_has_oom_retry_with_chunk_tightening`
- `pytest` ciblé: `3 passed`.

**Couleurs (constat rapide post-log):**
- Les stats RGB P6/P7 du log restent tri-canaux (pas mono explicite).
- Analyse rapide du PNG/FITS de sortie: image bien RGB, canaux proches mais non identiques.
- Suspicion actuelle: rendu/perception (stretch/équilibrage), pas conversion N&B stricte.

**Reste à faire:**
1. Re-run SDS et vérifier présence/efficacité de `global_coadd_gpu_oom_retry`.
2. Vérifier diminution des bascules CPU sur OOM.
3. Si rendu toujours "gris", traiter la chaîne d'affichage (preview stretch / RGB equalize) sans impacter la data FITS scientifique.
