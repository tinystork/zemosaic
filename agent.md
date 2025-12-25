# agent.md — Mission Codex: Rebrancher Two-Pass sur le **plan Phase 5** (fix bypass GPU-safety)

## Constat (root cause)
On construit un `parallel_plan_phase5` (auto-tune + GPU-safety + “plugged-aware”), et on le passe bien à l’assembly Phase 5 (reproject/coadd).
Mais Two-Pass (coverage renorm) est encore alimenté par un **plan différent** (souvent le plan global), ce qui bypass les budgets/chunks/safety.
Conséquence : côté Two-Pass, `gpu_max_chunk_bytes` peut être absent ⇒ `chunk_bytes=-` ⇒ trop de workers ⇒ pression RAM ⇒ crash.

## Objectif (minimal, anti-usine à gaz)
**Garantir que Two-Pass reçoit le même plan que l’assembly Phase 5** (`parallel_plan_phase5`) sur tous les chemins (normal + SDS).
Aucun changement “science”, uniquement **wiring + logs de preuve + cohérence GPU effective**.

---

## Périmètre STRICT (anti-régression)
### Fichier autorisé (seul)
- `zemosaic_worker.py`

### Interdits
- Pas de refactor massif / reformat massif
- Pas de nouvelle dépendance
- Ne pas toucher au comportement “batch size = 0” et “batch size > 1”
- Ne pas modifier l’algorithme Two-Pass (gains/blur/clip/merge)

---

## Tâches

### [x] 1) Ajouter un helper de sélection de plan Two-Pass
Dans `zemosaic_worker.py` (près des helpers Phase 5), créer une fonction utilitaire :

```python
def _select_two_pass_parallel_plan(*, phase5_plan, fallback_plan, zconfig=None):
    # priorité: plan Phase5 (local) -> zconfig.parallel_plan_phase5 -> fallback_plan
Règles :

si phase5_plan non-None → retour phase5_plan

sinon si zconfig a parallel_plan_phase5 non-None → retour celui-là

sinon retour fallback_plan

⚠️ But: éviter que d’autres chemins repassent le plan global par erreur.

### [x] 2) Rebrancher le callsite principal (Phase 5 post-pipeline)
Dans le gros bloc Phase 5 (après l’assembly), là où on appelle :

_apply_phase5_post_stack_pipeline(... parallel_plan=...)

Remplacer l’argument actuel (plan global) par le helper :

parallel_plan=_select_two_pass_parallel_plan(phase5_plan=parallel_plan_phase5, fallback_plan=parallel_plan, zconfig=zconfig)

✅ Two-Pass doit recevoir le même plan que celui utilisé pour l’assembly Phase 5.

### [x] 3) Rebrancher les callsites SDS “Phase 5 polish”
Dans tous les appels à _finalize_sds_global_mosaic(... parallel_plan=...) :

remplacer parallel_plan=getattr(zconfig, "parallel_plan", ...)

par :
parallel_plan=_select_two_pass_parallel_plan(phase5_plan=getattr(zconfig, "parallel_plan_phase5", None), fallback_plan=getattr(zconfig, "parallel_plan", worker_config_cache.get("parallel_plan")), zconfig=zconfig)

✅ Même en SDS, Two-Pass récupère le plan Phase 5 si dispo.

### [x] 4) Ajouter un log “preuve de rebranchement” (1 ligne, pas de spam)
Dans _apply_two_pass_coverage_renorm_if_requested(...), juste après le logger.info("[TwoPass] Second pass requested ..."),
ajouter UNE ligne INFO (ou DEBUG si tu préfères) qui résume le plan reçu :

type/nom du plan

cpu_workers, use_gpu

max_chunk_bytes, gpu_max_chunk_bytes

rows_per_chunk, gpu_rows_per_chunk

Ex (format libre) :
[TwoPass] plan=... cpu_workers=... use_gpu=... max_chunk_mb=... gpu_max_chunk_mb=... rows=... gpu_rows=...

But : on veut vérifier en 5 secondes que Two-Pass utilise bien parallel_plan_phase5.

### [x] 5) Cohérence GPU effective (micro-fix, sans “usine”)
Dans run_second_pass_coverage_renorm(...) :

Calculer use_gpu_effective = bool(use_gpu_two_pass) and bool(plan_use_gpu)

plan_use_gpu doit fonctionner si parallel_plan est un objet ou un dict.

Utiliser use_gpu_effective pour les appels internes (blur, etc.) au lieu de use_gpu_two_pass.

Si use_gpu_two_pass=True mais plan_use_gpu=False, logger un INFO clair :
[TwoPass] GPU requested but disabled by plan.use_gpu=False -> forcing CPU

But : Two-Pass doit respecter la décision autotune/GPU-safety déjà prise pour Phase 5.

Critères d’acceptation (observables)
La ligne [TwoPass] plan=... gpu_max_chunk_mb=... apparaît et reflète le plan Phase 5.

Sur un run où Phase 5 a gpu_max_chunk_bytes non nul, Two-Pass l’affiche aussi (plus de “plan global” muet).

Si parallel_plan_phase5.use_gpu == False, Two-Pass force CPU et le loggue.

Aucun autre comportement modifié.

Tests
- [x] python -m py_compile zemosaic_worker.py
- [ ] Run petit dataset (Two-Pass ON) : vérifier la ligne plan.
- [ ] Run dataset “crash” : vérifier que Two-Pass voit le plan Phase 5 (et donc gpu_max_chunk_bytes si présent).

