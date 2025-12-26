# agent.md — Mission Codex : VRAM dynamique Phase 5 (GPU reproject) — upscale + downscale

## Objectif
Rendre la Phase 5 (assemble_final_mosaic_reproject_coadd) capable d’ajuster dynamiquement l’usage VRAM GPU :
- **à la hausse** : augmenter les chunks quand la VRAM libre le permet (sans être bloqué par un rows_per_chunk trop petit),
- **à la baisse** : en cas d’OOM GPU, **réessayer sur GPU** avec des chunks réduits (plusieurs tentatives) avant fallback CPU.

## Contexte (relecture code)
- Phase 5 appelle `zemosaic_utils.reproject_and_coadd_wrapper(... use_gpu=use_gpu ...)`.
- La logique GPU est dans `zemosaic_utils.gpu_reproject_and_coadd_impl()`.
- Le GPU reproject sonde déjà `memGetInfo()` et clamp `max_chunk_bytes` (ex: free*0.25 / free*0.4), puis calcule `rows_per_chunk`.
- MAIS le `rows_per_chunk` fourni par le worker agit comme **cap** : `rows = min(rows_hint, rows_from_budget)`.
- En cas d’erreur GPU, le wrapper fait surtout “fallback CPU” ; il n’y a pas de “réduction chunk + retry GPU” ciblée OOM.

## Scope strict (fichiers autorisés)
- `zemosaic_worker.py` (Phase 5, uniquement : construction des kwargs / boucle channels)
- `zemosaic_utils.py` (GPU reproject + wrapper : ajout retry OOM + budget helper)
Optionnel si nécessaire (mais éviter si possible) :
- `zemosaic_gpu_safety.py` (uniquement si besoin d’un helper VRAM refresh réutilisable)

## Non-objectifs (interdits)
- Ne pas toucher au clustering / Phase 5 intertile pairs / safe_mode_windows single-worker.
- Ne pas modifier les comportements “batch size = 0” et “batch size > 1”.
- Ne pas changer la science (résultat) : uniquement perf/robustesse.
- Ne pas refactor massif, pas de nouvelle dépendance obligatoire.

## Spécification fonctionnelle
### A) Upscale (proactif)
Avant chaque appel GPU par canal (dans la boucle `for ch in range(n_channels)` de Phase 5) :
1. Sonder VRAM libre **au moment T** (best-effort, via `_probe_free_vram_bytes()`).
   - Si `_probe_free_vram_bytes()` renvoie `None`, ne changer rien aux hints existants (`rows_per_chunk`, `max_chunk_bytes`) et ne logguer que “VRAM probe unavailable, keeping plan hints”. Le reste de la logique upscale est ignoré pour ce canal.
2. Déterminer un **budget cible** (bytes) avec marge :
   - `budget = free_bytes * fraction`, ex:
     - fraction normal: 0.45 (ou 0.40 si tu veux ultra conservateur)
     - fraction safe_mode: 0.25–0.30
   - appliquer un **cap** si l’utilisateur/config a déjà fixé `plan_chunk_gpu_hint` (ne jamais dépasser ce cap).

2.bis. **Règle de non-régression (CAP DUR Phase5)** :
   - Le refresh VRAM “par canal” ne doit **JAMAIS** dépasser le chunk **déjà décidé** au début de Phase 5 par la logique AUTO/USER
     (incluant `phase5_chunk_auto=False`, `phase5_chunk_mb`, `on_battery_clamp`, `hard_cap_vram`, etc.).
   - Concrètement, définir `cap_bytes = plan_chunk_gpu_hint` (chunk effectif du plan Phase5) si disponible,
     où cap_bytes = le chunk effectif déjà appliqué au plan Phase5 après toute la logique AUTO/USER + on_battery_clamp + hard_cap_vram (donc typiquement parallel_plan_phase5.gpu_max_chunk_bytes / ou reproj_call_kwargs['max_chunk_bytes'] initial). On le snapshot une fois, et il sert de plafond.
     et imposer : `max_chunk_bytes_refresh = min(max_chunk_bytes_refresh, cap_bytes)`.
   - Le refresh peut **descendre** sous le cap (OOM/pression VRAM), mais ne peut **pas monter** au-dessus,
     même si la VRAM libre remonte entre canaux.

3. Calculer un `rows_hint` qui **n’empêche pas la montée** :
   - Stratégie simple et robuste : `rows_hint = H` (où `H` est la hauteur de la mosaïque finale, `final_output_shape_hw[0]`), ou un grand max (ex: 4096), pour laisser le GPU calculer `rows_from_budget`.
   - MAIS si safe-mode: capper `rows_hint` (ex 128/256) pour éviter les gros coups de bélier.
4. Passer `max_chunk_bytes` + `rows_per_chunk` recalés dans `reproj_call_kwargs` **uniquement si `use_gpu`**.
5. Log clair au niveau INFO_DETAIL :
   - free_vram_mb, budget_mb, rows_hint, cap_utilisateur_mb, safe_mode flag (si env var active).

Note: Si max_chunk_bytes ou rows_per_chunk est None/0/non-int, ne pas crasher : fallback sur les hints existants (ou valeurs minimales), et log 'refresh skipped/normalized'.

### B) Downscale (réactif)
Dans `zemosaic_utils._reproject_and_coadd_wrapper_impl()` :
1. Si `use_gpu=True` et que l’appel GPU plante avec une erreur typée OOM :
   - détecter OOM via :
     - `cupy.cuda.memory.OutOfMemoryError` (si import possible),
     - ou substring `"out of memory"` / `"CUDA_ERROR_OUT_OF_MEMORY"` dans le message.
2. Avant fallback CPU, faire jusqu’à `N=3` tentatives GPU :
   - à chaque tentative :
     - réduire `max_chunk_bytes` (ex: *0.7)
     - réduire `rows_per_chunk` (ex: *0.7, floor + clamp min 32)
     - appeler `free_cupy_memory_pools()` + `gc.collect()`
     - re-tenter `gpu_reproject_and_coadd_impl`
3. Si après N tentatives ça échoue encore :
   - si `allow_cpu_fallback=True`: fallback CPU (comportement existant),
   - sinon: remonter l’exception.
4. Log :
   - `[GPU Reproject] OOM retry k/N: max_chunk_mb=..., rows=..., free_vram_mb=...`

### C) Safety / compat
- Tous les ajouts CuPy doivent être “best-effort” (try/except import) : si CuPy absent → pas de changement CPU.
- Ne pas casser la filtration kwargs CPU/GPU existante.
- Ne pas modifier l’API publique : uniquement comportements internes.

- **Gating recommandé (éviter effets de bord hors Phase 5)** :
  - Le retry OOM dans `_reproject_and_coadd_wrapper_impl` doit être activé **uniquement** si un flag interne est présent,
    ex: `kwargs.get("_phase5_oom_retry", False) is True` (flag injecté par Phase 5).
  - Ajouter explicitement `_phase5_oom_retry` et `_phase5_oom_retry_max` à l’ensemble `gpu_only` dans le wrapper pour garantir que ces kwargs ne sont jamais transmis à `cpu_reproject_and_coadd`. Ceci verrouille le risque d’un TypeError côté CPU.

## Implémentation détaillée (guidée)
### 1) zemosaic_utils.py
Ajouter helpers internes :
- `_probe_free_vram_bytes()` → int|None
- `_is_gpu_oom_exception(exc)` → bool
- `_shrink_chunk_hints(max_chunk_bytes, rows_per_chunk)` → tuple(new_bytes, new_rows)
  - Dans cette fonction, après multiplication par 0.7, clamper les valeurs :
    - `max_chunk_bytes = max(max_chunk_bytes, 16 * 1024 * 1024)`
    - `rows_per_chunk = max(32, int(rows_per_chunk))`

Modifier `_reproject_and_coadd_wrapper_impl` :
- entourer le `gpu_reproject_and_coadd_impl(...)` d’une boucle retry OOM.
- ne retry QUE pour OOM (pas pour erreurs WCS, TypeError kwargs, etc.).

### 2) zemosaic_worker.py (Phase 5)
Dans `assemble_final_mosaic_reproject_coadd`, juste avant `chan_mosaic, chan_cov = _invoke_reproject(reproj_call_kwargs)` :
- si `use_gpu` :
  - recalculer `max_chunk_bytes` et `rows_per_chunk` via helper (nouveau helper dans worker ou appel helper zemosaic_utils).
  - remplacer dans `reproj_call_kwargs` :
    - `reproj_call_kwargs["max_chunk_bytes"] = ...`
    - `reproj_call_kwargs["rows_per_chunk"] = ...`
- ne rien changer si `use_gpu=False`.

Important : conserver `reproj_kwargs` de base mais autoriser ces 2 champs à être rafraîchis **par canal**.

## Critères d’acceptation
1. En run GPU, les logs montrent (par canal) une ligne “Phase5 GPU VRAM refresh …” (INFO_DETAIL).
2. Si VRAM libre augmente entre canaux, `max_chunk_bytes` peut monter (dans la limite du cap).
3. Si un OOM GPU survient, on observe `OOM retry 1/3 ...` et idéalement le run continue sur GPU.
4. Si CuPy absent, aucun changement de comportement (CPU identique).
5. Aucun changement des autres phases / modes / batch-size semantics.

## Test rapide (manuel)
- Sur dataset réaliste Phase 5 GPU.
- Forcer une situation OOM en configurant temporairement `max_chunk_bytes` très haut (ou en lançant une app GPU à côté),
  et vérifier que le wrapper fait “shrink + retry” avant fallback CPU.
