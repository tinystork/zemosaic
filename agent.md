# Mission — GRID_MODE: stabiliser la VRAM GPU (CuPy pool) + pré-check memGetInfo (patch minimal)

Contexte:
En grid_mode, sur gros datasets (~2k frames), on observe des crash GPU:
`cudaErrorMemoryAllocation: out of memory`.
Le code est déjà chunké, et `_stack_weighted_patches_gpu()` a déjà un fallback CPU en cas d’exception,
mais la VRAM “réservée” continue de gonfler (pool CuPy) et finit par exploser.

## Contraintes
- Patch **minimal** : uniquement grid_mode GPU stack.
- Ne pas modifier SDS ni le mode classique.
- Ne pas ajouter de nouveau système de logs : utiliser le logger existant / `_emit`.
- Ne pas changer le comportement “batch size = 0” et “batch size > 1”.

## Fichier cible
- `grid_mode.py` uniquement (sauf import d’un helper déjà existant).

## Objectif technique
1) Empêcher les allocations GPU “doomées” via un **pré-check VRAM** (`cp.cuda.runtime.memGetInfo()`).
2) Empêcher l’accumulation VRAM via **purge systématique du pool CuPy** en `finally`,
   même en cas d’erreur/retour anticipé.
3) Conserver le fallback CPU existant (pas de refactor du caller).

## Tâches

### Task 1 — Ajouter la purge CuPy pool (VRAM-stable)
Dans `grid_mode.py`, dans `def _stack_weighted_patches_gpu(...)` :
- Ajouter `free_cupy_memory_pools` dans l’import existant depuis `zemosaic_utils`.
- Encapsuler tout le corps GPU dans `try: ... except: ... finally: ...`
- Dans `finally`, appeler `free_cupy_memory_pools()` **inconditionnellement**.
- (Optionnel) si logger DEBUG: `_emit("Freed CuPy pools", lvl="DEBUG")`

Important: le `finally` doit s’exécuter même si:
- OOM pendant `cp.asarray(...)`
- exception dans `_normalize_patches_gpu(...)`
- exception dans `stack_core(...)`

### Task 2 — Pré-check VRAM avant conversion NumPy->CuPy
Toujours dans `_stack_weighted_patches_gpu`:
- Juste avant `cp_patches = [cp.asarray(...)]`, faire:
  - `free_b, total_b = cp.cuda.runtime.memGetInfo()`
  - `base = sum(p.nbytes for p in patches) + sum(w.nbytes for w in weights)`
  - `est = base * SAFETY_FACTOR` (ex: 4.0)  + marge fixe (ex: +64MB)
  - si `est > free_b * 0.85` :
      - `_emit(f"GPU precheck: insufficient VRAM (free=..., est=...), fallback CPU", lvl="INFO")`
      - retourner directement `_stack_weighted_patches(...)` (CPU)
- Ne pas spammer: une ligne par chunk max (c’est déjà le cas car fonction appelée par flush).

### Task 3 — Fallback CPU sur OOM sans fuite de pool
- Dans `except Exception as e:` existant:
  - si le message contient `"out of memory"` ou `"cudaErrorMemoryAllocation"`:
      - `_emit("GPU OOM during grid stack, fallback CPU", lvl="WARN")`
    sinon:
      - `_emit(f"GPU stack failed, falling back to CPU: {e}", lvl="WARN")`
  - retourner `_stack_weighted_patches(...)`
- La purge CuPy doit rester dans `finally` (donc exécutée aussi sur OOM).

## Critères d’acceptation
- Un gros run grid_mode ne doit plus crasher sur `cudaErrorMemoryAllocation`.
- Si VRAM insuffisante: fallback CPU par chunk, le run continue.
- Pas d’impact sur SDS / mode classique.
- Diff minimal, pas de refactor.

Livrer un diff propre.
