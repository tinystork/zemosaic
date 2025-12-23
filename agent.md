# Mission: Corriger la détection secteur/batterie Windows + clarifier GPU safety (no-refactor)

## Problème observé
Dans les logs GPU_SAFETY, on voit `battery=True` alors que l'utilisateur est sur secteur (`power_plugged=True`, `on_battery=False`).
Cela déclenche/affiche des raisons "battery_detected" et peut activer `safe_mode` même sur secteur, ce qui est confus.

Cause probable:
- `has_battery=True` signifie seulement "une batterie est présente" (PC portable), pas "en train de tourner sur batterie".
- Sous Windows, psutil/WMI peuvent renvoyer des états ambigus; la source la plus fiable est GetSystemPowerStatus (ACLineStatus).

## Objectifs
1) Sous Windows, utiliser GetSystemPowerStatus pour déterminer:
   - power_plugged (ACLineStatus==1)
   - on_battery (ACLineStatus==0)
   - (optionnel) battery_present depuis BatteryFlag si possible, sinon fallback WMI.
2) Ne plus activer `safe_mode` uniquement parce que `has_battery` est True.
   - `safe_mode` doit être True si:
     - Windows ET (on_battery == True)  OU  (hybrid_graphics == True)
3) Clarifier les raisons/logs:
   - Remplacer "battery_detected" par "battery_present" (information) quand has_battery True.
   - Ajouter "on_battery" uniquement quand on_battery True.
   - Garder "hybrid_graphics" inchangé.
4) Patch minimal, sans refactor, toucher uniquement zemosaic_gpu_safety.py (et éventuellement logs si nécessaire).

## Fichier principal
- zemosaic_gpu_safety.py

## Critères d'acceptation
- Sur secteur: logs indiquent `battery_present=True`, `power_plugged=True`, `on_battery=False`
  et `safe_mode` dépend du hybrid (pas de safe_mode uniquement parce qu'il y a une batterie).
- Sur batterie: `on_battery=True` et raison "on_battery" apparaît, safe_mode activé.
- Aucun crash si ctypes indisponible (fallback psutil/WMI comme avant).
