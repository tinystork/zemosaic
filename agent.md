# Mission (chirurgical): GPU safety batterie + Phase 5 Auto "plugged-aware" (no refactor)

## Problème
Sur Windows laptop hybride:
- `has_battery=True` (batterie présente) déclenche un clamp agressif même quand `on_battery=False` (sur secteur),
  ce qui bride inutilement `gpu_max_chunk_bytes` en Phase 5 (global_reproject).
- Résultat: micro-chunking, overhead et faible occupation GPU.

## Objectifs
1) zemosaic_gpu_safety.py
   - Distinguer clairement:
     - battery_present (has_battery)
     - on_battery (ACLineStatus==0)
     - power_plugged (ACLineStatus==1)
   - La "raison" batterie ne doit être ajoutée QUE si `on_battery=True`.
   - Ne pas activer un clamp "batterie" juste parce que battery_present=True.
   - Logs: afficher battery_present, on_battery, power_plugged + reasons propres
     (ex: reasons=["hybrid_graphics","on_battery"] ou ["hybrid_graphics","plugged_relax"]).

2) Phase 5 (global_reproject)
   - En mode Auto (pas d'override utilisateur):
     - Si `on_battery=True`: garder clamp strict (conservateur).
     - Si `power_plugged=True` et `safe_mode` dû à hybrid_graphics:
       -> relâcher le clamp pour global_reproject (Auto "plugged-aware")
       -> viser un cap Auto plus généreux (ex 256/512MB selon VRAM free) sans dépasser des bornes sûres.
   - En mode override utilisateur:
     - Respecter la valeur utilisateur, mais appliquer un "hard cap" seulement si `on_battery=True`
       (ou si VRAM free insuffisante), et logguer explicitement le cap.

3) Logs de validation (obligatoire)
   - À l’application du plan global_reproject:
     - log final: selected_chunk_bytes + source (AUTO vs USER) + cap_reason
     - ex:
       "Phase5 chunk AUTO: target=512MB, cap=512MB (plugged_relax, hybrid_graphics, vram_free=xxxxMB)"
       "Phase5 chunk USER: requested=1024MB, applied=512MB (on_battery clamp)"
       "Phase5 chunk AUTO: applied=128MB (on_battery clamp)"

## Contraintes
- No refactor: patch minimal, localisé.
- Ne pas modifier les autres opérations GPU (sauf si la logique existante est partagée et qu'un if op=="global_reproject" suffit).
- Ne pas toucher au comportement batch size=0 vs >1.
- Ne pas changer l'autotune intertiles (preview=512 guardrail etc.) — hors scope.

## Fichiers attendus
- zemosaic_gpu_safety.py
- zemosaic_worker.py (ou le fichier qui prépare le plan Phase 5 / operation="global_reproject")
- (optionnel) solver_settings.py uniquement si le plan est construit là, mais éviter si possible.

## Definition de "plugged-aware" (simple)
- clamp strict seulement si on_battery=True
- si power_plugged=True:
  - autoriser Auto à monter (ex: 256MB ou 512MB) en hybrid mode, selon VRAM libre
  - rester dans des bornes raisonnables (ex: 64MB..1024MB)
