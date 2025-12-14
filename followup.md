# Follow-up — GRID GPU OOM hardening (ZeMosaic)

## Patch attendu: uniquement `grid_mode.py`

### 1) Vérifier le point d’entrée
- Dans `grid_mode.py`, identifier `def _stack_weighted_patches_gpu(...)`.
- Vérifier qu’il est appelé seulement quand `config.use_gpu` (caller fait déjà ce check).

### 2) Import
- Étendre l’import existant:
  `from zemosaic_utils import (...)`
  en y ajoutant `free_cupy_memory_pools`.

### 3) Ajouter un pré-check VRAM (avant cp.asarray)
- Utiliser `cp.cuda.runtime.memGetInfo()`.
- Estimer bytes:
  `base = sum(p.nbytes ...) + sum(w.nbytes ...)`
  `est = base * 4.0 + 64*1024*1024`
- Si `est > free_b * 0.85`:
  - log INFO via `_emit`
  - return `_stack_weighted_patches(...)` (CPU) immédiatement.

### 4) Encapsuler avec try/except/finally
- `try`: code GPU actuel
- `except`: log WARN (OOM vs generic) + fallback CPU
- `finally`: appeler `free_cupy_memory_pools()` inconditionnellement
  (optionnel DEBUG log)

### 5) Vérifications rapides
- Petit dataset grid_mode GPU: doit toujours fonctionner.
- Gros dataset: plus de crash; si fallback, on doit le voir dans le log.
- SDS/classic: aucun diff.

## Done when
- plus d’OOM en grid_mode,
- VRAM ne “monte” plus sans limite (pool purgé entre chunks),
- diff minimal.
