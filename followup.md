# Follow-up: Implémentation détaillée

## 1) GUI (zemosaic_gui_qt.py)
### 1.1 Localiser l'onglet "System"
- [x] Repérer la zone qui contient les paramètres système / GPU safety / workers.
- [x] Ajouter un petit groupbox "Phase 5 (Reproject)" ou une ligne dans la section GPU.

### 1.2 Widgets
- [x] QCheckBox: "Auto (recommended)"
  - [x] objectName (ex): `chk_phase5_chunk_auto`
  - [x] default checked = True
- [x] QSpinBox:
  - [x] label: "Phase 5 chunk (MB)"
  - [x] objectName (ex): `spin_phase5_chunk_mb`
  - [x] min=64, max=1024, singleStep=64
  - [x] defaultValue=128
  - [x] enabled = False si Auto checked
- [x] Side note (discret):
  - [x] soit QLabel en small font (style "muted") OU tooltip sur le label/spinbox
  - [x] texte: "⚠ May cause instability on some laptops / hybrid GPUs. Use with caution."

### 1.3 Connexions
- [x] `chk_phase5_chunk_auto.toggled.connect(spin.setDisabled/Enabled)`
  - [x] logique: si Auto True -> spin disabled, sinon enabled
- [x] À la sauvegarde des settings:
  - [x] lire checkbox + spinbox et stocker dans la config
- [x] Au chargement:
  - [x] initialiser checkbox/spinbox depuis config
  - [x] appliquer enabled/disabled

### 1.4 i18n
- [x] Si le projet utilise un helper `tr()` / dictionnaire de traductions:
  - [x] ajouter les clés FR/EN correspondantes:
    - [x] "Auto (recommended)" -> "Auto (recommandé)"
    - [x] "Phase 5 chunk (MB)" -> "Chunk Phase 5 (Mo)"
    - [x] warning -> "⚠ Peut provoquer des instabilités sur certains laptops / GPU hybrides. À utiliser avec prudence."
- [x] Si pas d'i18n centralisé: garder en anglais OU suivre le pattern existant du fichier.

## 2) Config (zemosaic_config.py)
### 2.1 Ajouter les champs par défaut
- [x] `phase5_chunk_auto = True`
- [x] `phase5_chunk_mb = 128`
Où sont définis les defaults (dataclass/dict), suivre le style existant.

### 2.2 Load/save
- [x] Lors du chargement d'un ancien config, assurer fallback sur defaults (get(..., default))
- [x] Lors de la sauvegarde, écrire les deux champs.

## 3) Worker (zemosaic_worker.py)
### 3.1 Localiser Phase 5 plan
Dans la section Phase 5 (global reproject):
- [x] Identifier le plan/obj qui contient `gpu_max_chunk_bytes` (ou équivalent)
- [x] Identifier l'opération/label: `operation="global_reproject"`

### 3.2 Appliquer override (après safety ou avant ?)
Recommandation:
- [x] Appliquer override **après** `apply_gpu_safety_to_parallel_plan(..., operation="global_reproject")`
  pour que:
  - [x] safety fixe les autres paramètres
  - [x] l'utilisateur force uniquement le budget bytes final

Pseudo:
- [x] lire config:
  - [x] `auto = cfg.phase5_chunk_auto`
  - [x] `mb = cfg.phase5_chunk_mb`
- [x] si `not auto`:
  - [x] `forced_bytes = int(mb) * 1024 * 1024`
  - [x] clamp: `forced_bytes = max(64MB, min(1024MB, forced_bytes))` (mêmes bornes que UI)
  - [x] si `hasattr(plan, "gpu_max_chunk_bytes")`: set
  - [x] log INFO:
    "Phase5 GPU chunk override: {mb} MB ({forced_bytes} bytes) for global_reproject"
- [x] sinon: ne rien faire

### 3.3 Robustesse
- [x] Si le champ n'existe pas, log DEBUG et ne pas planter.
- [x] Ne pas modifier les autres opérations.
- [x] Ne pas toucher `gpu_rows_per_chunk` ici (ça reste géré par autotune/safety). (On pourra le faire ensuite si besoin.)

## 4) Tests rapides
### 4.1 Test manuel GUI
- [x] Lancer app
- [x] Vérifier:
  - [x] Auto coché -> spinbox grisé
  - [x] décocher -> spinbox activé
  - [x] changer valeur -> sauvegarder -> relancer -> valeur persistée

### 4.2 Test run
- [x] Avec override (ex 256MB), lancer un run court
- [x] Vérifier présence du log "Phase5 GPU chunk override: 256 MB"
- [x] Comparer le nombre de chunks / durée Phase 5 (optionnel)

## 5) Notes
- [x] Ne pas renommer/retoucher d'autres paramètres existants.
- [x] Garder l'UI sobre: une ligne, un tooltip, pas plus.
