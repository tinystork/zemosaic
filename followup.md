# followup.md — Validation: Two-Pass correctement rebranché sur le plan Phase 5

## 0) Scope / anti-régression
- [x] Seul fichier modifié : `zemosaic_worker.py`
- [x] Pas de refactor massif / reformat massif
- [x] Pas de nouvelle dépendance
- [x] Aucun changement de l’algorithme Two-Pass (wiring + logs + use_gpu_effective uniquement)

---

## 1) Helper de sélection de plan
- [x] `_select_two_pass_parallel_plan(...)` existe
- [x] Priorité appliquée :
  1) `phase5_plan` (local) si présent
  2) sinon `zconfig.parallel_plan_phase5` si présent
  3) sinon `fallback_plan`

---

## 2) Call sites corrigés

### 2.1 Chemin principal Phase 5 post-pipeline
- [x] `_apply_phase5_post_stack_pipeline(... parallel_plan=...)` reçoit le helper avec `phase5_plan=parallel_plan_phase5`
- [x] On ne passe plus le plan global directement

### 2.2 Chemins SDS “Phase 5 polish”
- [x] Tous les `_finalize_sds_global_mosaic(... parallel_plan=...)` utilisent le helper
- [x] Fallback propre sur `zconfig.parallel_plan`/cache si `parallel_plan_phase5` absent

---

## 3) Log de preuve
- [x] Une ligne unique par run Two-Pass résume le plan :
  - [x] plan type/nom
  - [x] cpu_workers / use_gpu
  - [x] max_chunk_bytes / gpu_max_chunk_bytes
  - [x] rows_per_chunk / gpu_rows_per_chunk

---

## 4) GPU effective (cohérence safety)
- [x] `use_gpu_effective = use_gpu_two_pass AND plan.use_gpu` est calculé
  - [x] support plan objet et dict
- [x] Si GPU demandé mais plan.use_gpu=False :
  - [x] log INFO clair
  - [x] Two-Pass tourne en CPU

---

## 5) Sanity
- [x] `python -m py_compile zemosaic_worker.py` OK

---

## 6) Validation terrain

### Dataset petit (contrôle)
- [ ] Two-Pass s’exécute
- [ ] Log `[TwoPass] plan=... gpu_max_chunk_mb=...` présent

### Dataset “crash Phase 5”
- [ ] Le log Two-Pass affiche un plan cohérent avec Phase 5 (mêmes budgets GPU si présents)
- [ ] Si `plan.use_gpu=False`, Two-Pass force CPU et le loggue

---

## Critère final
- [ ] Two-Pass n’utilise plus un plan “global” par erreur.
- [ ] Les décisions autotune/GPU-safety Phase 5 se reflètent dans Two-Pass.
