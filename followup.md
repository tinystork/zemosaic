# Follow-up: Détails implémentation (no refactor) + validation

## [X] A) zemosaic_gpu_safety.py — corriger la condition batterie (chirurgical)

### [X] A1) Probe Windows: on_battery / power_plugged
- [X] Si ce n'est pas déjà fait: utiliser GetSystemPowerStatus (ctypes) pour obtenir ACLineStatus.
- [X] Interprétation:
  - ACLineStatus==0 -> on_battery=True
  - ACLineStatus==1 -> power_plugged=True
  - 255 -> unknown (fallback psutil/WMI)

### [X] A2) battery_present vs on_battery
- [X] battery_present = "une batterie existe" (pc portable)
- [X] on_battery = "ACLineStatus==0"
Important:
- [X] Ne JAMAIS utiliser battery_present comme substitut de on_battery.

### [X] A3) Reasons / logs
Avant (bug):
- reasons inclut "battery_detected" même sur secteur
- clamp agressif déclenché

Après (fix):
- [X] Si battery_present: éventuellement log info (battery_present=True) mais PAS une reason de clamp.
- [X] Ajouter reason "on_battery" UNIQUEMENT si on_battery=True.
- [X] Garder reason "hybrid_graphics" si applicable.
- [X] Ajouter reason "plugged_relax" si safe_mode était activé pour hybrid mais qu’on est sur secteur et qu’on relâche le clamp pour global_reproject.

### [X] A4) safe_mode
- [X] safe_mode peut rester True pour hybrid_graphics (protection anti-freeze),
  MAIS le clamp "batterie" ne doit s'appliquer que si on_battery=True.

---

## [X] B) Phase 5 (global_reproject) — Auto "plugged-aware"

### [X] B1) Où patcher
- [X] Dans le code où le plan est construit pour Phase 5:
  - [X] juste après apply_gpu_safety_to_parallel_plan(... operation="global_reproject")
  - [X] et juste avant l’exécution GPU.
L’idée: safety configure, puis on ajuste uniquement le chunk en fonction (AUTO/USER, plugged/battery).

### [X] B2) Lecture du réglage UI/config
- [X] `phase5_chunk_auto` (bool)
- [X] `phase5_chunk_mb` (int) si override

### [X] B3) Algorithme minimal (proposé)
Variables:
- [X] vram_free_bytes (ou estimation existante; sinon fallback: ne rien faire)
- [X] current_chunk = plan.gpu_max_chunk_bytes
- [X] safe_mode flag + reasons (si accessibles)
- [X] on_battery / power_plugged

Règles:
1) [X] Si override utilisateur (phase5_chunk_auto=False):
   - [X] requested = mb * 1024*1024
   - [X] applied = requested
   - [X] Si on_battery=True: applied = min(applied, 128MB)  (clamp strict batterie)
   - [X] Sinon (secteur): applied = min(applied, hard_cap_from_vram_free)
   - [X] Set plan.gpu_max_chunk_bytes = applied
   - [X] Log: USER requested/applied + reason clamp si modifié.

2) [X] Si Auto (phase5_chunk_auto=True):
   - [X] Si on_battery=True:
       applied = min(current_chunk, 128MB) (ou laisser safety si déjà <=128MB)
       reason = "on_battery clamp"
   - [X] Sinon si power_plugged=True et hybrid_graphics=True:
       # plugged-aware relax
       target = max(current_chunk, 256MB) puis essayer 512MB si VRAM free le permet
       applied = min(target, hard_cap_from_vram_free)
       reason = "plugged_relax hybrid"
   - [X] Sinon (desktop ou non-hybrid):
       laisser current_chunk (ou autoriser 512MB par défaut si vram_free très confortable)
   - [X] Log: AUTO target/applied + reason.

### [X] B4) Hard cap from VRAM free (très simple, safe)
- [X] hard_cap = min(1024MB, max(128MB, int(0.10 * vram_free_bytes)))
  (10% de VRAM free, borné, évite les pics)
- [X] Si vram_free inconnue: hard_cap = 512MB en secteur, 128MB sur batterie.

### [X] B5) Logs (obligatoire)
Ajouter un log unique et lisible au moment d’appliquer:
- [X] "Phase5 chunk AUTO: current=128MB target=512MB applied=512MB (plugged_relax, hybrid_graphics, vram_free=6200MB, hard_cap=620MB)"
- [X] "Phase5 chunk USER: requested=1024MB applied=512MB (hard_cap_vram, vram_free=5000MB)"
- [X] "Phase5 chunk AUTO: applied=128MB (on_battery clamp)"

---

## [X] C) Validation rapide

### [X] C1) Cas secteur (ton cas)
- [X] battery_present=True
- [X] on_battery=False
- [X] power_plugged=True
Attendu:
- [X] reasons ne doit plus contenir "battery_detected" / "on_battery"
- [X] Phase 5 Auto peut monter à 256/512MB selon VRAM free
- [X] log "plugged_relax" présent si hybrid

### [X] C2) Cas batterie (débrancher)
Attendu:
- [X] reasons inclut "on_battery"
- [X] Phase 5 Auto et USER clamp à 128MB
- [X] log explicite "on_battery clamp"

### [X] C3) Non régression
- [X] Ne pas modifier intertiles autotune (preview guardrail)
- [X] Ne pas modifier les autres phases GPU