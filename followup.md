Oui, on peut faire ça **proprement et très peu invasif** — et surtout **sans rajouter de variables/config** : juste faire en sorte que **`winsor_worker_limit = 0` signifie réellement “auto”**, comme les autres champs.

### Ce que ça implique dans le code actuel (conséquences)

Aujourd’hui, **0 ne peut jamais être “auto”** parce que :

* Dans `zemosaic_worker.py` tu clamps plusieurs fois avec `max(1, ...)` → **0 devient 1** (mono-worker).
* Dans `zemosaic_align_stack.py`, la logique est déjà prête pour le “0=auto” : `winsor_max_workers = current_workers or plan_cpu_workers`. Donc **si on laisse passer 0**, ça marche “gratuitement”.

Donc la mission consiste à :

1. **Ne plus écraser 0 en 1** côté worker (ou le convertir en une valeur auto calculée).
2. **S’assurer que là où on stocke une valeur dans un “plan global”**, on met une valeur **>=1** (sinon certains endroits pourraient faire `min(entries, winsor_worker_limit)` et là 0 serait catastrophique).

Le patch minimal, user-friendly, c’est :

* **Phase 4.5 / stacking local** : laisser passer `0` tel quel → `zemosaic_align_stack` fera l’auto via `parallel_plan`.
* **Plans globaux (SDS / classic global_wcs_plan / stack_params)** : si `winsor_worker_limit_config <= 0`, **résoudre en “auto_effective” (>=1)** une fois, puis continuer comme avant.

---

## agent.md

```markdown
# Mission — “Winsor worker limit: 0 = auto” (patch minimal, user-friendly)

Objectif:
- Rendre le paramètre `winsor_worker_limit` cohérent avec les autres champs UI:
  - `winsor_worker_limit = 0` => mode AUTO
  - `winsor_worker_limit > 0` => limite manuelle
- Patch MINIMAL: pas de nouvelles options/config, pas de refactor.

## Constat (code actuel)
Dans `zemosaic_worker.py`, plusieurs endroits font:
- winsor_worker_limit = max(1, min(int(cfg), cpu_total))
=> 0 devient 1 (donc pas d’auto).
Or `zemosaic_align_stack.py` supporte déjà implicitement:
- winsor_max_workers = current_workers or plan_cpu_workers
=> si on passe 0, l’auto fonctionne.

## Fichiers à modifier (minimum)
- /mnt/data/zemosaic_worker.py (obligatoire)

Optionnel (ne pas faire sauf si nécessaire):
- /mnt/data/zemosaic_align_stack.py (a priori inutile)

## Règles de comportement attendues
1) Phase 4.5 (stacking local via `stack_kwargs["winsor_max_workers"]`):
   - Si cfg == 0: passer 0 tel quel (AUTO géré par zemosaic_align_stack via parallel_plan)
   - Sinon: passer max(1, cfg)

2) Global/SDS stacking params (là où on stocke `winsor_worker_limit` dans des dicts type global_wcs_plan / sds_stack_params):
   - Si cfg <= 0: calculer une valeur effective >= 1 (AUTO) et stocker cette valeur.
   - Sinon: clamp normal (1..cpu_total).
   Raisons: certains endroits utilisent cette valeur comme un cap numérique direct, et 0 casserait la logique.

## Définition de “AUTO” (simple et robuste)
AUTO doit privilégier les workers déjà auto-tunés:
- si `global_parallel_plan` existe et `global_parallel_plan.cpu_workers > 0` => utiliser ça
- sinon fallback sur `effective_base_workers` (déjà calculé par le worker)
- sinon fallback sur `cpu_total`
puis clamp [1..cpu_total].

## Logging (minimal mais utile)
- Quand cfg <= 0, loguer une ligne INFO_DETAIL du style:
  "Winsor worker limit: AUTO (cfg=0) -> resolved=<N> (cpu_total=<M>)"
- En Phase 4.5, si on laisse passer 0, éviter de loguer "workers=0" de manière trompeuse:
  loguer "workers=AUTO(0)" ou similaire.

## Contraintes
- Ne pas changer les algorithmes scientifiques.
- Ne pas toucher à la GUI.
- Ne pas introduire de nouvelle variable de config.
- Patch localisé et lisible.
```

---

## followup.md

```markdown
# Plan d’exécution (Codex)

## Étape 1 — Identifier les 3 endroits clés dans zemosaic_worker.py
- [x] Phase 4.5 stack_kwargs: laisser passer 0 → winsor_max_workers = 0, sinon clamp à >=1.
- [x] Deux blocs “winsor_worker_limit = max(1, min(int(winsor_worker_limit_config), cpu_total))” remplacés par une résolution AUTO effective (global_parallel_plan > effective_base_workers > cpu_total) et stockage clampé >=1.

## Étape 2 — Logging minimal
- [x] cfg <= 0 dans global_wcs_plan: log INFO_DETAIL "AUTO cfg=0 -> resolved=N".
- [x] Phase 4.5: workers loggué en "AUTO(0)" quand on laisse passer 0.

## Étape 3 — Vérifs rapides
- [x] py_compile: `python -m py_compile zemosaic_worker.py`
- [ ] Test fonctionnel simple (config GUI winsor_worker_limit=0, vérifier logs et bornage workers)

## Notes importantes
- Ne PAS propager 0 dans global_wcs_plan / sds_stack_params (risque de cap=0 ailleurs).
- Par contre, laisser passer 0 dans stack_kwargs (Phase 4.5) est OK car `zemosaic_align_stack.py` sait résoudre 0 via parallel_plan.
- Ne pas modifier zemosaic_align_stack.py (il est déjà compatible).
```

---

Si tu veux, je peux aussi te donner un **mini diff “à la main”** (3 micro-changements) pour vérifier que Codex ne part pas en sucette — mais tel quel, ce brief est normalement assez cadré pour un patch chirurgical.
