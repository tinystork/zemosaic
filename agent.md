# agent.md — ZeMosaic Patch: GPU safety smarter + intertile autotune guardrails (no refactor)

## Contexte / Symptômes
- Phase5 "Intertile Pairs" devenue anormalement lente car l’auto-tune peut choisir `preview=1024` et `min_overlap=0.0100`, augmentant fortement `pairs`.
- Phase5 "Reproject" semble sous-utiliser le GPU car le GPU safety force `gpu_rows_per_chunk <= 128` dès que `safe_mode=1`, même sur de vrais GPU (ex: RTX 3070).
- Objectif: conserver la sécurité anti-freeze, mais éviter de brider inutilement les configs robustes.

## Contraintes
- **No refactor** : patch local, minimal, sans déplacer de grosses fonctions.
- Ne pas modifier la logique "batch size = 0" / "batch size > 1".
- Ne pas casser la compatibilité Windows/Linux/Mac (fallback gracieux si API non dispo).
- Ajouter des logs clairs (niveau INFO/DEBUG) pour confirmer les décisions.

## Fichiers ciblés
- `zemosaic_gpu_safety.py`  (principal)
- (optionnel et très léger) `zemosaic_worker.py` si besoin d’exposer 1-2 infos de debug (pas de restructuration)

## Tâches
### T1 — Battery detection correcte (Windows surtout)
Actuel: `battery=True` peut signifier "la machine a une batterie", pas "elle tourne sur batterie".
- Utiliser `psutil.sensors_battery()` si dispo:
  - si `battery is None` → `on_battery = False`
  - sinon `on_battery = (battery.power_plugged is False)`
- Conserver l’ancien comportement si `psutil` indisponible ou exception.
- Loguer un message clair:
  - `GPU_SAFETY: power_plugged=<bool|None> on_battery=<bool>`

### T2 — Clamp gpu_rows_per_chunk: remplacer le hard-cap 128 par un cap adaptatif
But: garder safe-mode, mais permettre des chunks raisonnables sur GPU Nvidia dédiés.
- Règle proposée (simple, robuste):
  - Déterminer `rows_cap` en fonction de la VRAM libre (si GPU backend peut la fournir), sinon fallback.
  - Cibler un budget "chunk_bytes" (par ex 256 MiB par défaut) mais **réduit** en safe-mode.
- Politique:
  - `base_chunk_bytes = 256 MiB`
  - si `safe_mode` ET `on_battery` ou `hybrid` → `safe_chunk_bytes = 128 MiB` (au lieu de réduire à 128 rows)
  - si GPU a >= ~6-8GB VRAM libre → autoriser 256MiB même en hybrid, sauf si on_battery=True
- Convertir bytes -> rows avec estimation `bytes_per_row` (déjà calculée ou calculable dans le module).
- Clamp final:
  - `rows = clamp(rows, min_rows=32, max_rows=2048)` (valeurs conservatrices)
- Loguer:
  - `GPU_SAFETY: chosen gpu_rows_per_chunk=<n> (budget=<MiB>, bytes_per_row=<n>, vram_free=<MiB>, safe_mode=<...>)`

### T3 — Guardrails sur intertile auto-tune pour éviter l’explosion des paires
But: empêcher l’auto-tune de choisir une combinaison "explosive" par défaut.
- Si auto-tune propose:
  - `preview >= 1024` ET `min_overlap <= 0.015` → appliquer un garde-fou:
    - soit réduire `preview` à 512
    - soit remonter `min_overlap` à 0.03 ou 0.05 (préférer 0.03 si on veut permissif)
- Guardrail basé sur une heuristique simple:
  - `estimated_pairs = n_tiles * avg_neighbors` (approx) ou directement après calcul des overlaps:
    - si `pairs > max_pairs_guardrail` (ex: 2_000) → durcir paramètres
  - Comme on veut patch minimal: faire au moins le guardrail "preview/min_overlap".
- Loguer:
  - `INTERTILE_AUTOTUNE_GUARDRAIL: adjusted preview=... min_overlap=... (reason=...)`

### T4 — Tests / validation (smoke tests)
- Lancer un run (ou simulation) sur un dataset connu:
  - vérifier dans le log:
    - `power_plugged` détecté correctement
    - `gpu_rows_per_chunk` > 128 sur RTX 3070 quand branché
    - auto-tune n’aboutit pas à `preview=1024` + `min_overlap=0.01` simultanément (sauf si utilisateur force manuellement)
- Ne pas modifier l’UI. (Optionnel: si une constante ou tooltip existe déjà, ne pas toucher.)

## Definition of Done
- Sur machine hybride Nvidia **branchée**, `safe_mode` peut rester activé mais `gpu_rows_per_chunk` n’est plus systématiquement 128.
- Sur machine **sur batterie** (power_plugged=False), le safety reste conservateur.
- Intertile auto-tune ne choisit plus la combinaison explosive par défaut; logs explicites.
- Aucun refactor large; diff limité aux fichiers ciblés.

## Notes d’implémentation (aide)
- Préférer des helpers internes dans `zemosaic_gpu_safety.py` (petites fonctions), pas de nouveaux modules.
- Si `psutil` n’est pas une dépendance, importer en try/except.
