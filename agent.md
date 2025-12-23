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
- [x] Phase 4.5 (stacking local via `stack_kwargs["winsor_max_workers"]`):
   - Si cfg == 0: passer 0 tel quel (AUTO géré par zemosaic_align_stack via parallel_plan)
   - Sinon: passer max(1, cfg)

- [x] Global/SDS stacking params (là où on stocke `winsor_worker_limit` dans des dicts type global_wcs_plan / sds_stack_params):
   - Si cfg <= 0: calculer une valeur effective >= 1 (AUTO) et stocker cette valeur.
   - Sinon: clamp normal (1..cpu_total).
   Raisons: certains endroits utilisent cette valeur comme un cap numérique direct, et 0 casserait la logique.

## Définition de “AUTO” (simple et robuste)
AUTO doit privilégier les workers déjà auto-tunés:
- si `global_parallel_plan` existe et `global_parallel_plan.cpu_workers > 0` => utiliser ça
- sinon fallback sur `effective_base_workers` (déjà calculé par le worker)
- sinon fallback sur `cpu_total`
puis clamp [1..cpu_total].

## Logging (minimal mais uti
