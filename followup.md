# Follow-up: Détails implémentation (no refactor) + validation

## A) zemosaic_gpu_safety.py — corriger la condition batterie (chirurgical)

### A1) Probe Windows: on_battery / power_plugged
- Si ce n'est pas déjà fait: utiliser GetSystemPowerStatus (ctypes) pour obtenir ACLineStatus.
- Interprétation:
  - ACLineStatus==0 -> on_battery=True
  - ACLineStatus==1 -> power_plugged=True
  - 255 -> unknown (fallback psutil/WMI)

### A2) battery_present vs on_battery
- battery_present = "une batterie existe" (pc portable)
- on_battery = "ACLineStatus==0"
Important:
- Ne JAMAIS utiliser battery_present comme substitut de on_battery.

### A3) Reasons / logs
Avant (bug):
- reasons inclut "battery_detected" même sur secteur
- clamp agressif déclenché

Après (fix):
- Si battery_present: éventuellement log info (battery_present=True) mais PAS une reason de clamp.
- Ajouter reason "on_battery" UNIQUEMENT si on_battery=True.
- Garder reason "hybrid_graphics" si applicable.
- Ajouter reason "plugged_relax" si safe_mode était activé pour hybrid mais qu’on est sur secteur et qu’on relâche le clamp pour global_reproject.

### A4) safe_mode
- safe_mode peut rester True pour hybrid_graphics (protection anti-freeze),
  MAIS le clamp "batterie" ne doit s'appliquer que si on_battery=True.

---

## B) Phase 5 (global_reproject) — Auto "plugged-aware"

### B1) Où patcher
- Dans le code où le plan est construit pour Phase 5:
  - juste après apply_gpu_safety_to_parallel_plan(... operation="global_reproject")
  - et juste avant l’exécution GPU.
L’idée: safety configure, puis on ajuste uniquement le chunk en fonction (AUTO/USER, plugged/battery).

### B2) Lecture du réglage UI/config
- `phase5_chunk_auto` (bool)
- `phase5_chunk_mb` (int) si override

### B3) Algorithme minimal (proposé)
Variables:
- vram_free_bytes (ou estimation existante; sinon fallback: ne rien faire)
- current_chunk = plan.gpu_max_chunk_bytes
- safe_mode flag + reasons (si accessibles)
- on_battery / power_plugged

Règles:
1) Si override utilisateur (phase5_chunk_auto=False):
   - requested = mb * 1024*1024
   - applied = requested
   - Si on_battery=True: applied = min(applied, 128MB)  (clamp strict batterie)
   - Sinon (secteur): applied = min(applied, hard_cap_from_vram_free)
   - Set plan.gpu_max_chunk_bytes = applied
   - Log: USER requested/applied + reason clamp si modifié.

2) Si Auto (phase5_chunk_auto=True):
   - Si on_battery=True:
       applied = min(current_chunk, 128MB) (ou laisser safety si déjà <=128MB)
       reason = "on_battery clamp"
   - Sinon si power_plugged=True et hybrid_graphics=True:
       # plugged-aware relax
       target = max(current_chunk, 256MB) puis essayer 512MB si VRAM free le permet
       applied = min(target, hard_cap_from_vram_free)
       reason = "plugged_relax hybrid"
   - Sinon (desktop ou non-hybrid):
       laisser current_chunk (ou autoriser 512MB par défaut si vram_free très confortable)
   - Log: AUTO target/applied + reason.

### B4) Hard cap from VRAM free (très simple, safe)
- hard_cap = min(1024MB, max(128MB, int(0.10 * vram_free_bytes)))
  (10% de VRAM free, borné, évite les pics)
- Si vram_free inconnue: hard_cap = 512MB en secteur, 128MB sur batterie.

### B5) Logs (obligatoire)
Ajouter un log unique et lisible au moment d’appliquer:
- "Phase5 chunk AUTO: current=128MB target=512MB applied=512MB (plugged_relax, hybrid_graphics, vram_free=6200MB, hard_cap=620MB)"
- "Phase5 chunk USER: requested=1024MB applied=512MB (hard_cap_vram, vram_free=5000MB)"
- "Phase5 chunk AUTO: applied=128MB (on_battery clamp)"

---

## C) Validation rapide

### C1) Cas secteur (ton cas)
- battery_present=True
- on_battery=False
- power_plugged=True
Attendu:
- reasons ne doit plus contenir "battery_detected" / "on_battery"
- Phase 5 Auto peut monter à 256/512MB selon VRAM free
- log "plugged_relax" présent si hybrid

### C2) Cas batterie (débrancher)
Attendu:
- reasons inclut "on_battery"
- Phase 5 Auto et USER clamp à 128MB
- log explicite "on_battery clamp"

### C3) Non régression
- Ne pas modifier intertiles autotune (preview guardrail)
- Ne pas modifier les autres phases GPU
