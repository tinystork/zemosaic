# followup.md — Validation & tests (manual)

## 1) Sanity checks (log)
Lancer un run Grid mode en conditions “à problème” (GPU + dataset conséquent).

Vérifier dans le log:
- Absence de:
  - `GPU stack failed, falling back to CPU: module 'cupy' has no attribute 'errstate'`
- Présence de:
  - `GPU concurrency ... forced to 1 (WDDM safety)` (Windows)
  - `Auto-tune grid_workers ... base=..., cap_os=..., cap_gpu=..., cap_ram=..., result=...`
  - `using X workers for tile processing` avec X raisonnable (<=4 Windows+GPU en auto)

## 2) Tests de comportement utilisateur (transparence)
### Cas A — utilisateur ne touche à rien
- `grid_workers` absent ou 0 => autotune actif.
- Attendu: workers capés automatiquement, pas de freeze OS.

### Cas B — utilisateur force grid_workers
- mettre `grid_workers=14` dans config
- Attendu: workers=14 (respect), mais warning si Windows+GPU+>4.

### Cas C — désactiver l’autotune (compat)
- mettre `parallel_autotune_enabled=false` dans config
- Attendu: retour à la logique historique auto = cpu_count-2 (si grid_workers=0),
  et GPU concurrency reprend la formule VRAM (sauf Windows rule si tu l’as laissée conditionnée à autotune).

## 3) Overrides utiles (debug)
- Forcer single thread:
  - `ZEMOSAIC_GRID_FORCE_SINGLE_THREAD=1`
- Forcer la GPU concurrency:
  - `ZEMOSAIC_GRID_GPU_CONCURRENCY=1` (ou 2/3 pour tests)

## 4) Validation “anti-freeze”
Sur Windows:
- lancer un run avec dataset lourd et laisser le PC utilisable (bouger fenêtres, ouvrir navigateur).
- Attendu: pas de freeze de l’UI OS.

## 5) Non-régression
- Lancer un run CPU-only (use_gpu_grid=false):
  - Attendu: pas de régression perf, et workers auto peuvent monter plus haut (dans la limite des caps).
````

