# Follow-up: Implémentation détaillée

## 1) GUI (zemosaic_gui_qt.py)
### 1.1 Localiser l'onglet "System"
- Repérer la zone qui contient les paramètres système / GPU safety / workers.
- Ajouter un petit groupbox "Phase 5 (Reproject)" ou une ligne dans la section GPU.

### 1.2 Widgets
- QCheckBox: "Auto (recommended)"
  - objectName (ex): `chk_phase5_chunk_auto`
  - default checked = True
- QSpinBox:
  - label: "Phase 5 chunk (MB)"
  - objectName (ex): `spin_phase5_chunk_mb`
  - min=64, max=1024, singleStep=64
  - defaultValue=128
  - enabled = False si Auto checked
- Side note (discret):
  - soit QLabel en small font (style "muted") OU tooltip sur le label/spinbox
  - texte: "⚠ May cause instability on some laptops / hybrid GPUs. Use with caution."

### 1.3 Connexions
- `chk_phase5_chunk_auto.toggled.connect(spin.setDisabled/Enabled)`
  - logique: si Auto True -> spin disabled, sinon enabled
- À la sauvegarde des settings:
  - lire checkbox + spinbox et stocker dans la config
- Au chargement:
  - initialiser checkbox/spinbox depuis config
  - appliquer enabled/disabled

### 1.4 i18n
- Si le projet utilise un helper `tr()` / dictionnaire de traductions:
  - ajouter les clés FR/EN correspondantes:
    - "Auto (recommended)" -> "Auto (recommandé)"
    - "Phase 5 chunk (MB)" -> "Chunk Phase 5 (Mo)"
    - warning -> "⚠ Peut provoquer des instabilités sur certains laptops / GPU hybrides. À utiliser avec prudence."
- Si pas d'i18n centralisé: garder en anglais OU suivre le pattern existant du fichier.

## 2) Config (zemosaic_config.py)
### 2.1 Ajouter les champs par défaut
- `phase5_chunk_auto = True`
- `phase5_chunk_mb = 128`
Où sont définis les defaults (dataclass/dict), suivre le style existant.

### 2.2 Load/save
- Lors du chargement d'un ancien config, assurer fallback sur defaults (get(..., default))
- Lors de la sauvegarde, écrire les deux champs.

## 3) Worker (zemosaic_worker.py)
### 3.1 Localiser Phase 5 plan
Dans la section Phase 5 (global reproject):
- Identifier le plan/obj qui contient `gpu_max_chunk_bytes` (ou équivalent)
- Identifier l'opération/label: `operation="global_reproject"`

### 3.2 Appliquer override (après safety ou avant ?)
Recommandation:
- Appliquer override **après** `apply_gpu_safety_to_parallel_plan(..., operation="global_reproject")`
  pour que:
  - safety fixe les autres paramètres
  - l'utilisateur force uniquement le budget bytes final

Pseudo:
- lire config:
  - `auto = cfg.phase5_chunk_auto`
  - `mb = cfg.phase5_chunk_mb`
- si `not auto`:
  - `forced_bytes = int(mb) * 1024 * 1024`
  - clamp: `forced_bytes = max(64MB, min(1024MB, forced_bytes))` (mêmes bornes que UI)
  - si `hasattr(plan, "gpu_max_chunk_bytes")`: set
  - log INFO:
    "Phase5 GPU chunk override: {mb} MB ({forced_bytes} bytes) for global_reproject"
- sinon: ne rien faire

### 3.3 Robustesse
- Si le champ n'existe pas, log DEBUG et ne pas planter.
- Ne pas modifier les autres opérations.
- Ne pas toucher `gpu_rows_per_chunk` ici (ça reste géré par autotune/safety). (On pourra le faire ensuite si besoin.)

## 4) Tests rapides
### 4.1 Test manuel GUI
- Lancer app
- Vérifier:
  - Auto coché -> spinbox grisé
  - décocher -> spinbox activé
  - changer valeur -> sauvegarder -> relancer -> valeur persistée

### 4.2 Test run
- Avec override (ex 256MB), lancer un run court
- Vérifier présence du log "Phase5 GPU chunk override: 256 MB"
- Comparer le nombre de chunks / durée Phase 5 (optionnel)

## 5) Notes
- Ne pas renommer/retoucher d'autres paramètres existants.
- Garder l'UI sobre: une ligne, un tooltip, pas plus.
