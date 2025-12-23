# Mission (chirurgicale) — Plugged-aware Phase 5 Auto chunk + logs

Objectif:
- Corriger la logique "battery" de zemosaic_gpu_safety.py pour distinguer:
  - batterie présente (has_battery)
  - réellement sur batterie (on_battery)
  - secteur branché (power_plugged)
- Rendre la Phase 5 (Reproject) en mode Auto "plugged-aware" (budget chunk GPU plus large quand branché),
  tout en conservant le Safe Mode (pas de refactor, patch minimal).

Contraintes:
- Toucher uniquement:
  - zemosaic_gpu_safety.py
  - zemosaic_worker.py (logique Phase 5 chunk Auto + logs)
- Pas de refactor, pas de renommage massif, pas de nouvelles dépendances.
- Ajouter des logs de validation: chunk choisi + raison (incluant power_plugged/on_battery/hybrid).

Contexte observé dans les logs actuels:
- En Phase 5 reproject, on voit:
  - power_plugged=True, on_battery=False
  - MAIS gpu_chunk_mb reste clamp à 128MB avec reasons=hybrid_graphics,safe_mode_clamp
  => clamp trop agressif même branché.
- Quand l’utilisateur force un chunk (ex 1024), on voit un cap VRAM "hard_cap_vram" (ex 712MB) et bump rows_per_chunk.
  => chemin manuel OK, c’est Auto + Safety qui doit être branché-aware.

Deliverables:
1) zemosaic_gpu_safety.py
   - Modifier la logique safe-mode clamp:
     - clamp "on_battery" => oui (agressif)
     - clamp "hybrid_graphics" => agressif seulement si NOT power_plugged
     - si hybrid_graphics mais power_plugged=True: autoriser un budget par défaut plus large (ex 256MB) avant cap VRAM.
   - Modifier les "reasons" loguées pour ne plus écrire battery_detected/battery_present comme s’il y avait limitation:
     - si has_battery mais power_plugged=True et on_battery=False => reason "battery_present" (info) mais PAS de clamp spécifique batterie
     - si on_battery=True => reason "on_battery_clamp"
     - si hybrid && !power_plugged => reason "hybrid_unplugged_clamp"
     - si safe_mode clamp général => reason "safe_mode_clamp"
   - Ajouter un log clair (INFO) récapitulatif: power_plugged, on_battery, has_battery, hybrid, budget_bytes final + reasons.

2) zemosaic_worker.py (Phase 5)
   - Dans le chemin "Auto (recommended)" du chunk Phase 5:
     - s’appuyer sur apply_gpu_safety_to_parallel_plan(...) / ctx pour obtenir power_plugged & on_battery.
     - logguer un message unique et stable:
       "Phase5 chunk AUTO: applied=<X>MB (base=<Y>MB, vram_free=<Z>MB, power_plugged=<T>, on_battery=<F>, hybrid=<H>, reasons=[...])"
     - Ne pas casser la logique existante "USER chunk" (qui log déjà).
   - But: quand branché, Auto ne doit plus rester bloqué à 128MB uniquement parce que hybrid_graphics=True.

Validation / tests:
- Test A (secteur): lancer un run Phase 5 Auto, vérifier log:
  - power_plugged=True on_battery=False
  - chunk Auto appliqué > 128MB (si vram_free le permet) et reasons reflètent "plugged"
- Test B (sur batterie): simuler ou forcer on_battery=True (si possible) et vérifier chunk Auto clamp (<=128MB) + reason on_battery_clamp.
- Test C (hybrid unplugged): power_plugged=False + hybrid=True => clamp agressif et reason hybrid_unplugged_clamp.
- Aucun changement de résultats d’image attendu, uniquement perf/stabilité.

Notes:
- Garder le comportement "batch size = 0" et ">1" intact (ne pas toucher à ça).
