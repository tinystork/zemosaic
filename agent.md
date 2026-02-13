# agent.md — ZeMosaic DBE v2 (Presets + Surface Fit)

## Objectif
Finaliser le mode **DBE (Dynamic Background Extraction)** appliqué sur la **mosaïque finale** (Phase 6), en ajoutant :
1) Un sélecteur de force **Weak / Normal / Strong** dans le **GUI Qt** (simple, visible).
2) La persistance en config de 4 paramètres avancés :
   - `obj_k` (seuil objets)
   - `obj_dilate_px` (dilatation masque)
   - `sample_step` (pas de la grille d’échantillonnage)
   - `smoothing` (rigidité du modèle de fond)

3) Une implémentation DBE plus qualitative : au lieu d’un simple flou gaussien low-res, construire un **modèle de fond par surface** à partir d’**échantillons de fond** (masque d’objets) sur l’image sous-échantillonnée, via une interpolation/approximation **RBF (thin-plate) lissée** ou spline (RBF recommandé pour démarrer).

Le DBE doit rester **safe**, **robuste** et **sans régression** (SDS, grid mode, pipeline global).

---

## Contraintes non négociables
- **Aucune régression** sur : SDS mode, grid mode, classic mode.
- `final_mosaic_dbe_enabled` est considéré comme **déjà implémenté** ; vérifier en code qu’il est bien présent/utilisé avant toute suite.
- En cas de doute sur un item, **vérifier d'abord s'il est déjà fait** dans le code, puis cocher la checklist au lieu de réimplémenter.
- DBE ne doit pas exploser la mémoire : conserver l’approche **par canal** (pas de buffer H×W×3 pour le modèle).
- DBE doit être **fail-safe** :
  - ordre obligatoire: `RBF` -> `gaussien` -> `skip DBE` (sans crash).
  - si SciPy indisponible / fit RBF échoue / trop peu d’échantillons: tenter gaussien.
  - si le fallback gaussien échoue aussi: skip DBE.
  - le mode `DEBUG` doit expliciter la raison et l’étape de fallback.
- Garder les logs DBE existants et les enrichir (sans spam INFO inutile).
- **IMPORTANT : mettre à jour `memory.md` à CHAQUE itération**, en notant :
  - ce qui est fait, fichiers modifiés,
  - décisions (valeurs presets, limites),
  - ce qui reste à faire,
  - comment reproduire/tester.

---

## État actuel (à respecter)
- DBE actuel : `_apply_final_mosaic_dbe_per_channel()` dans `zemosaic_worker.py` (flou gaussien low-res).
- Appel DBE en phase 6 dans `zemosaic_worker.py` (il y a 2 blocs quasi identiques → il faudra patcher les deux).
- GUI Qt : checkbox déjà présente : `final_mosaic_dbe_enabled` dans `zemosaic_gui_qt.py`.
- Scope GUI: **Qt uniquement** (`zemosaic_gui_qt.py`). Ne pas considérer `zemosaic_gui.py` pour cette mission.

---

## Spécification UI (Qt)
### 1) Ajout d’un preset “DBE Strength”
Dans `zemosaic_gui_qt.py`, section “Final assembly and output” (près de la checkbox DBE) :
- Ajouter un **QComboBox** “DBE strength” avec :
  - Weak
  - Normal
  - Strong
- Le preset doit être **désactivé** si `final_mosaic_dbe_enabled` est décoché.
- Valeur par défaut : **Normal**.
- `Custom` est réservé aux **power users** via édition JSON (pas d’exposition GUI).

### 2) Paramètres avancés (config uniquement)
Conserver en config les 4 paramètres :
- `final_mosaic_dbe_obj_k` (float)
- `final_mosaic_dbe_obj_dilate_px` (int)
- `final_mosaic_dbe_sample_step` (int)
- `final_mosaic_dbe_smoothing` (float)

Le GUI ne montre que les options Weak/Normal/Strong.  
Le mode `custom` reste supporté côté worker/config si `final_mosaic_dbe_strength="custom"` est défini dans le JSON.

---

## Mapping des presets (valeurs initiales proposées)
Ces valeurs sont sur l’image **low-res** (après downsample).

- Weak:
  - obj_k = 4.0
  - obj_dilate_px = 2
  - sample_step = 32
  - smoothing = 1.0

- Normal (default):
  - obj_k = 3.0
  - obj_dilate_px = 3
  - sample_step = 24
  - smoothing = 0.6

- Strong:
  - obj_k = 2.2
  - obj_dilate_px = 4
  - sample_step = 16
  - smoothing = 0.25

- Custom:
  - utilise strictement les valeurs en config.

Note : `obj_k` plus bas = masque d’objets plus agressif.
`smoothing` plus bas = surface plus flexible (plus proche des points).

---

## Implémentation DBE v2 (Worker)
### Objectif
Remplacer/améliorer le modèle de fond gaussien par un modèle “surface-fit” :
1) Downsample (déjà fait via ds_factor).
2) Construire un masque “background only” en low-res :
   - stats robustes sur pixels valides : median + MAD
   - seuil objets : `thr = median + obj_k * (1.4826 * MAD)`
   - `object_mask = channel_lr > thr`
   - dilatation : `obj_dilate_px` (cv2.dilate ou équivalent)
   - `bg_mask = valid_lr & ~object_mask_dilated`

3) Échantillonnage du fond sur une grille régulière :
   - pas `sample_step`
   - pour chaque point de grille, prendre la **médiane** des pixels dans une petite fenêtre locale (p.ex. rayon = sample_step//2), uniquement où `bg_mask` est True
   - collecter (x, y, value)

4) Fit d’une surface lissée :
   - SciPy recommandé : `scipy.interpolate.Rbf(xs, ys, zs, function="thin_plate", smooth=smoothing)`
   - évaluer sur la grille low-res complète → `bg_lr`
   - upsample `bg_lr` vers full-res
   - soustraire sur les pixels valides

### Performance / garde-fous obligatoires
- Limiter le nombre de points : `max_samples = 2000` (ou 3000 max).
  - si dépasse : augmenter automatiquement `sample_step` OU sous-échantillonner les points (random stable).
- Si `n_samples < 30` (ou < 50) : fallback gaussien (méthode actuelle).
- Si SciPy absent ou fit échoue : fallback gaussien.
- Si fallback gaussien échoue : skip DBE (fail-open), sans crash.
- Conserver traitement **par canal**.

### API / signatures
Dans `zemosaic_worker.py` :
- Étendre `_apply_final_mosaic_dbe_per_channel(... )` avec :
  - `obj_dilate_px: int`
  - `sample_step: int`
  - `smoothing: float`
  - `strength: str` (ou `preset`)
  - (optionnel) `method: str = "surface_rbf"` et fallback `"gaussian"`

Ou créer une nouvelle fonction `_apply_final_mosaic_dbe_surface_per_channel()` et garder l’ancienne pour fallback.

### Lecture config au hook Phase 6
Dans les 2 blocs Phase 6 (les 2 occurrences) :
- Lire :
  - `final_mosaic_dbe_strength` (default "normal")
  - si "custom" → lire les 4 paramètres en config
  - sinon → utiliser mapping preset
- Passer ces paramètres à la fonction DBE.

---

## Logs + FITS header
### Logs
Enrichir le log “[DBE] applied=True …” pour inclure :
- preset/strength
- obj_k, obj_dilate_px, sample_step, smoothing
- n_samples (par canal ou total)
- model utilisé : `rbf_thin_plate` ou `gaussian_fallback`
- En `DEBUG`, tracer explicitement les transitions de fallback:
  - `rbf_failed -> gaussian_fallback`
  - `gaussian_failed -> dbe_skipped`

### Header FITS (optionnel mais utile)
Garder existants : `ZMDBE`, `ZMDBE_DS`, `ZMDBE_K` (+ éventuellement `ZMDBE_SIG` si fallback gaussien).
Ajouter (si appliqué) :
- `ZMDBE_STR` (weak/normal/strong/custom)
- `ZMDBE_DIL` (int)
- `ZMDBE_STP` (int)
- `ZMDBE_SMO` (float)
- `ZMDBE_MDL` ("rbf_thin_plate" / "gaussian")

---

## Fichiers à modifier (scope)
- `zemosaic_worker.py` (DBE algo + hook phase6 x2)
- `zemosaic_gui_qt.py` (UI presets Weak/Normal/Strong uniquement)
- `zemosaic_config.py` (defaults)
- `memory.md` (OBLIGATOIRE à chaque itération)
- Ne pas modifier `zemosaic_gui.py` dans ce scope.

---

## Tests / validation (smoke tests)
Tests exécutés manuellement par l’utilisateur sur dataset réduit.

1) DBE ON :
   - pas de crash
   - logs DBE présents avec les nouveaux champs
2) DBE OFF :
   - pas de logs “applied=True”
3) Basculer preset Weak/Normal/Strong :
   - vérifier que la config persiste
   - vérifier que le worker reçoit bien des valeurs différentes (logs)
4) Forcer SciPy indisponible (si possible) ou simuler exception :
   - vérifier fallback gaussien sans crash
   - si gaussien échoue aussi, vérifier skip DBE + trace `DEBUG`

---

## Mise à jour memory.md (impératif)
À chaque itération, ajouter une section datée :
- ✅ Faits (liste)
- 🔧 Fichiers modifiés
- 🧪 Tests effectués + résultats
- ⚠️ Limitations connues
- ⏭️ Next steps

Fin.
