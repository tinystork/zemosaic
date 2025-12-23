# Follow-up: Implémentation (Windows power status) + ajustement safe_mode/reasons

## [x] 1) Implémenter un probe Windows fiable (ctypes)
Dans `zemosaic_gpu_safety.py`, ajouter (Windows only) un helper:

- importer localement `ctypes` et `ctypes.wintypes` dans la fonction (pour éviter impact cross-platform).
- définir la struct SYSTEM_POWER_STATUS:
  - BYTE ACLineStatus
  - BYTE BatteryFlag
  - BYTE BatteryLifePercent
  - BYTE SystemStatusFlag
  - DWORD BatteryLifeTime
  - DWORD BatteryFullLifeTime

- appeler `kernel32.GetSystemPowerStatus(byref(status))`

Interprétation:
- ACLineStatus:
  - 0 = sur batterie
  - 1 = sur secteur
  - 255 = unknown (dans ce cas fallback)
- on_battery = (ACLineStatus == 0)
- power_plugged = (ACLineStatus == 1)
- battery_present (optionnel) :
  - BatteryFlag == 128 => "No system battery" (donc has_battery=False)
  - sinon has_battery=True (si BatteryFlag != 255 unknown)

Retourner ces infos au format (has_battery, power_plugged, on_battery) quand fiable, sinon None pour fallback.

## [x] 2) Modifier _probe_battery_status()
Ordre de priorité recommandé:
1) Windows + ctypes GetSystemPowerStatus (si ACLineStatus != 255)
2) psutil.sensors_battery()
3) WMI Win32_Battery() (déduire seulement has_battery)

Important:
- Ne pas écraser une info déjà fiable par une info moins fiable.
- Si power_plugged est connu mais has_battery ne l’est pas, garder power_plugged et compléter has_battery via WMI.

## [x] 3) Corriger la logique safe_mode + reasons (probe_gpu_runtime_context)
Actuel:
- safe_mode = True si Windows and has_battery True   (trop agressif)
- reasons "battery_detected" ajouté juste car has_battery True (confus)

Nouveau:
- if is_windows and has_battery is True: reasons.append("battery_present")
- if is_windows and on_battery is True:
    safe_mode = True
    reasons.append("on_battery")
- if is_windows and is_hybrid is True:
    safe_mode = True
    reasons.append("hybrid_graphics")

=> safe_mode dépend de (on_battery OR hybrid), pas de "battery_present".

## [x] 4) Logs
Les logs existants affichent déjà power_plugged/on_battery/has_battery.
S'assurer que le champ `battery=` continue à afficher `has_battery` (ok),
mais que `reasons=` n’induise plus en erreur:
- "battery_present" au lieu de "battery_detected"
- "on_battery" seulement si on_battery True

## [ ] 5) Mini test manuel (sans framework)
Sur Windows:
- Lancer une exécution courte (ou un appel isolé à probe_gpu_runtime_context).
- Vérifier log:
  - Sur secteur: power_plugged=True, on_battery=False, reasons contient battery_present + hybrid_graphics (si hybride), PAS on_battery
  - Débrancher: power_plugged=False, on_battery=True, reasons contient on_battery (+hybrid_graphics si hybride)

## [x] 6) Ne pas toucher
- Ne pas modifier la taille de chunk ici (c’est une autre mission).
- Ne pas modifier le comportement batch size=0 vs >1.
