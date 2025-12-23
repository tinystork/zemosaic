# followup.md — Patch checklist & guidance (no refactor)

## 0) Branch / commit
- Travailler sur une branche: `fix/gpu-safety-rows-cap-intertile-guardrails`
- 1 commit unique:
  - message: `Fix GPU safety clamp and intertile autotune guardrails`

## 1) Implémentation — GPU safety
### 1.1 Détection “on_battery”
- Dans `zemosaic_gpu_safety.py`:
  - try:
    - import psutil
    - b = psutil.sensors_battery()
    - if b is None: on_battery=False; power_plugged=None
    - else: power_plugged=b.power_plugged; on_battery=(power_plugged is False)
  - except: fallback ancien comportement (ne pas planter)
- Log INFO/DEBUG:
  - `GPU_SAFETY: power_plugged=... on_battery=...`

### 1.2 Cap adaptatif au lieu de `gpu_rows_per_chunk=128`
- Remplacer le clamp dur par:
  - `chunk_budget_bytes = 256MiB` nominal
  - si `safe_mode`:
    - si `on_battery`: `chunk_budget_bytes = 64-128MiB`
    - sinon si `hybrid`: `chunk_budget_bytes = 128MiB`
  - si VRAM libre accessible: `chunk_budget_bytes = min(chunk_budget_bytes, vram_free * 0.33)` (optionnel, simple)
- Convertir en rows via `bytes_per_row` (si déjà dispo) sinon estimation conservatrice:
  - ex: `bytes_per_row = width * channels * dtype_bytes` + overhead mask
  - si inconnu: ne pas inventer n’importe quoi → fallback sur 256 rows.
- Clamp final rows:
  - min 32, max 2048 (ou 4096 si tu veux, mais rester safe)
- Log:
  - `GPU_SAFETY: chosen gpu_rows_per_chunk=... budget_mib=... vram_free_mib=...`

## 2) Implémentation — Intertile autotune guardrails
- Localiser l’endroit où sont fixés `preview` et `min_overlap` pour intertile (là où le log annonce `Using: preview=..., min_overlap=...`).
- Ajouter garde-fou minimal:
  - si `(preview >= 1024 and min_overlap <= 0.015)`:
    - set `preview = 512`
    - set `min_overlap = max(min_overlap, 0.03)` (ou 0.05 si tu veux être strict)
  - log INFO:
    - `INTERTILE_AUTOTUNE_GUARDRAIL: adjusted preview=512 min_overlap=0.03 (was preview=1024 min_overlap=0.01)`
- Option bonus (si ultra simple à placer sans refactor):
  - après calcul `pairs`, si `pairs > 2000`:
    - remonter `min_overlap` à 0.05 et recalculer pairs une fois (1 retry max)
    - log: `... pairs guardrail triggered ...`
  - Ne pas partir en boucle.

## 3) Validation (à exécuter)
- Lancer un run sur Windows laptop Nvidia hybride **branché**:
  - Attendu dans le log:
    - `power_plugged=True`, `on_battery=False`
    - `gpu_rows_per_chunk` > 128 (typiquement 256/512)
- Vérifier phase5 intertile:
  - Attendu:
    - pas de `preview=1024` + `min_overlap=0.0100` en même temps (si auto-tune activé)
    - nombre de paires raisonnable (selon dataset)
- Vérifier absence de crash si `psutil` indisponible (Linux minimal / Mac):
  - try/except OK, log fallback.

## 4) Non-objectifs (ne pas faire)
- Pas de modifications UI Qt (pas de nouveaux widgets).
- Pas de refactor worker / pipeline.
- Ne pas toucher à la logique “batch size”.

## 5) Output attendu
- Diff compact, centré sur `zemosaic_gpu_safety.py` (+ éventuel micro-ajustement là où intertile fixe preview/min_overlap).
- Logs plus explicites sur la décision safety + intertile guardrail.
