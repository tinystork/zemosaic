# agent.md — Grid mode auto-tuning (no refactor)

## Objectif
Rendre Grid/Survey mode **auto-adaptatif** (CPU workers + GPU concurrency) pour éviter les freezes OS en mode GPU (Windows/WDDM),
tout en conservant un comportement transparent pour l’utilisateur.

## Contraintes
- **NO REFACTOR**
- Patch minimal, idéalement **1 fichier**: `grid_mode.py`
- Ne pas changer l’UI, ne pas ajouter de nouveaux réglages visibles.
- Respecter `grid_workers` si l’utilisateur l’a explicitement fixé (>0).
- Ne modifier le comportement automatique que lorsque `grid_workers == 0` (auto).
- Utiliser les clés déjà existantes du config quand possible:
  - `parallel_autotune_enabled` (default True)
  - `parallel_target_ram_fraction` (default 0.9)
  - `parallel_max_cpu_workers` (default 0 = pas de cap)
- Ajouter uniquement des logs INFO/WARN pour expliquer les caps auto (utile au debug).

## Fichiers à modifier
- `grid_mode.py`

## Tâches

### [x] Fix bug CuPy `cp.errstate` (évite fallback CPU chaotique)
Dans `_stack_weighted_patches_gpu` (vers la division pondérée), remplacer:
```py
with cp.errstate(divide="ignore", invalid="ignore"):
    result = ...
````

par un contexte safe:

* importer `nullcontext` depuis `contextlib`
* faire:

```py
errstate = getattr(cp, "errstate", None)
ctx = errstate(divide="ignore", invalid="ignore") if callable(errstate) else nullcontext()
with ctx:
    result = ...
```

=> si `cp.errstate` n’existe pas, on n’échoue plus, et la division est déjà protégée via `cp.clip(weight_sum, 1e-6, None)`.

### [x] Auto-tune GPU concurrency (Windows => 1 par défaut)

Modifier `_compute_gpu_concurrency(stack_chunk_budget_mb)` pour:

1. Supporter override via env:

   * `ZEMOSAIC_GRID_GPU_CONCURRENCY=<int>` (si >0, force ce chiffre)
2. Sur Windows (WDDM) **par défaut**:

   * retourner `concurrency=1` (sauf override env)
3. Sinon, garder la logique existante basée VRAM.

Logging:

* si Windows => log INFO: “GPU concurrency on Windows forced to 1 (WDDM safety)”
* si override env => log INFO: “GPU concurrency forced by env …”

### [x] Auto-tune `grid_workers` (uniquement si grid_workers==0)

Étendre `_get_effective_grid_workers(config: dict)` en signature:

```py
def _get_effective_grid_workers(config: dict, *, use_gpu: bool, stack_chunk_budget_mb: float, gpu_concurrency: int = 1) -> int:
```

Comportement:

* Si `ZEMOSAIC_GRID_FORCE_SINGLE_THREAD` => 1 (inchangé)
* Si `grid_workers > 0` => respecter strictement, MAIS:

  * si `use_gpu` et Windows et valeur > 4 => log WARN (risque freeze)
* Si `grid_workers == 0` (auto):

  * base = max(1, cpu_logical - 2) (logique actuelle)
  * si `parallel_autotune_enabled` est True (config):

    * appliquer des caps “sûrs” dépendants de la machine:

      1. cap OS+GPU:

         * Windows+GPU => `cap_os = 4`
         * Linux/macOS+GPU => `cap_os = 6`
         * CPU-only => `cap_os = 12` (ou pas de cap si tu préfères)
      2. cap par GPU concurrency (évite 14 tuiles CPU qui martèlent le pipeline GPU):

         * si `use_gpu`: `cap_gpu = max(2, gpu_concurrency * 4)`
         * sinon: `cap_gpu = base`
      3. cap RAM dispo:

         * utiliser `parallel_target_ram_fraction` (default 0.9)
         * estimer `per_worker_mb` conservateur:

           * GPU: `per_worker_mb = max(2500, stack_chunk_budget_mb * 3.0)`
           * CPU: `per_worker_mb = max(1800, stack_chunk_budget_mb * 2.0)`
         * `ram_budget_mb = available_mb * min(0.98, parallel_target_ram_fraction)`
         * `cap_ram = max(1, floor(ram_budget_mb / per_worker_mb))`
      4. cap config global:

         * si `parallel_max_cpu_workers > 0`: `cap_cfg = parallel_max_cpu_workers` sinon infini
    * effective = min(base, cap_os, cap_gpu, cap_ram, cap_cfg)
  * log INFO détaillé:

    * base, cap_os, cap_gpu, cap_ram, cap_cfg, result
    * available_mb, per_worker_mb, stack_chunk_budget_mb, gpu_concurrency

### [x] Adapter l’appel existant dans `run_grid_mode`

Dans `run_grid_mode`, lors de la création du `ThreadPoolExecutor`:

* on calcule déjà `concurrency` + `gpu_stack_semaphore`
* passer `use_gpu=config.use_gpu`, `stack_chunk_budget_mb=config.stack_chunk_budget_mb`, `gpu_concurrency=concurrency`
  à `_get_effective_grid_workers(...)`

Ex:

```py
num_workers = _get_effective_grid_workers(
    cfg_disk,
    use_gpu=bool(config.use_gpu),
    stack_chunk_budget_mb=float(config.stack_chunk_budget_mb),
    gpu_concurrency=int(concurrency),
)
```

## Critères d’acceptation

* Plus de `GPU stack failed ... cupy has no attribute errstate`
* Sur Windows + GPU + auto workers (grid_workers=0), le log montre un cap à <=4.
* Sur Windows + GPU, GPU concurrency retombe à 1 (sauf override env).
* Sur Linux/macOS, la logique VRAM conserve la concurrency existante.
* Si l’utilisateur force `grid_workers`, ZeMosaic respecte la valeur (avec warning si dangereuse).

## Notes

* Ne pas toucher au reste du pipeline Grid (assign, stack, assemble).
* Pas de nouveaux champs GUI.
