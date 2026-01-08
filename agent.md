# Mission — Alt-Az cleanup : master tiles “vides” à cause du baseline shift float (ALPHA ignoré)

## Contexte
Quand `altaz_cleanup_enabled=True` (option GUI “Alt-Az cleanup”), certaines master tiles (ex: 10 et 21) semblent “vides/noires/plates” dans des viewers FITS.

Le repo contient :
- `zemosaic_worker.log` (run concerné)
- `example/master_tile_010.fits` et `example/master_tile_021.fits` (tiles problématiques)

## Ce que montrent le log + les FITS (faits)
### 1) Ce n’est pas un “skip”
Dans `zemosaic_worker.log`, les deux tiles :
- passent dans le pipeline (`_apply_lecropper_pipeline()` appelé),
- ont `MT_PIPELINE: altaz_cleanup applied ... mask2d_applied=True threshold=0.01`,
- subissent un `MT_EDGE_TRIM` normal,
- passent le quality gate (`[ZeQualityMT] ... -> ACCEPT`).

### 2) Les tiles ne sont pas “désactivées” par ALPHA
Dans les FITS fournis :
- il y a un cube `RGB` + une extension `ALPHA`.
- `ALPHA` est majoritairement à 255 (≈ 82–85% de pixels), donc la zone utile existe.

### 3) La vraie cause (phase 1) : baseline shift *énorme* à l’export float basé sur le min global
Les deux FITS contiennent dans le header une ligne :
- `HISTORY Baseline shift applied for FITS float export: +339699840.000000 ADU` (tile 010)
- `HISTORY Baseline shift applied for FITS float export: +153565872.000000 ADU` (tile 021)

Ce shift vient de `zemosaic_utils.save_fits_image(..., save_as_float=True)` qui :
- calcule `finite_min = np.nanmin(data)` sur **tous** les pixels,
- si `finite_min < 0`, ajoute `-finite_min` à **toute** l’image.

Or, en “dé-shiftant” les FITS (soustraction du shift), on observe :
- **zone valide (ALPHA>0)** : valeurs “normales” (~0–50k, percentiles ~640–1100 ADU),
- **zone invalide (ALPHA==0)** : quelques valeurs **très négatives** (jusqu’à `-shift`), qui forcent le `finite_min` global.

Résultat : on ajoute un offset énorme à toute la tile, ce qui écrase la dynamique en auto-stretch naïf → impression de tile vide/plate.

### 4) Nouveau constat : outliers positifs énormes hors ALPHA
`master_tile_021.fits` a un shift modeste (~+517 ADU) **mais** un `max` global ~2.38e8,
quasi entièrement dans `ALPHA==0`. Cela écrase encore la dynamique en viewer.
Le log montre des `linear_fit` avec `a`/`b` extrêmes (ex: `a=3.21e6`, `b=-2.84e9`).

## Objectif
Quand une `alpha_mask` est fournie, empêcher les pixels invalides (ALPHA=0) de piloter le baseline shift float, et éviter d’embarquer des valeurs extrêmes hors footprint.

Objectif “safe” : corriger le symptôme via l’export.
Objectif “complet” : neutraliser les coefficients extrêmes en amont (normalisation `linear_fit`).

## Scope / fichiers
- ✅ `zemosaic_utils.py` (`save_fits_image`) — correctif export.
- ✅ `zemosaic_align_stack.py` (`_normalize_images_linear_fit`) — garde-fous contre coefficients extrêmes.
- ❌ Ne pas modifier : algos GPU / reproject (sauf si le correctif export + linear_fit ne suffit pas)

## Exigences fonctionnelles
### A0) Définition “pixel valide” + compatibilité `alpha_mask`
- `alpha_mask` attendu : array 2D `(H, W)` (dtype quelconque), avec 0 = invalide, >0 = valide.
  - La règle de validité MUST être : `valid_hw = np.isfinite(alpha_mask) & (alpha_mask > 0)`.
  - (Compatible à la fois avec un ALPHA en 0..1 ou 0..255.)
- Compatibilité shapes :
  - Si `image_data.ndim == 2`: `image_data.shape == (H, W)` requis.
  - Si `image_data.ndim == 3`:
    - HWC si `image_data.shape == (H, W, C)` (C=3 typiquement),
    - CHW si `image_data.shape == (C, H, W)` (C=3 typiquement).
  - Si la compatibilité ne peut pas être prouvée (shape mismatch / ndim inattendu / H,W indéterminables) :
    => considérer `alpha_mask` comme **incompatible** et appliquer le comportement actuel inchangé (global min).

### A) Baseline shift basé sur les pixels valides quand ALPHA est présent
Dans `save_fits_image(..., save_as_float=True, alpha_mask=...)` :
- Si `alpha_mask` est fourni **et** compatible avec les dimensions spatiales de l’image :
  - calculer `finite_min` (et éventuellement `finite_max`) **uniquement** sur les pixels `ALPHA>0`.
  - appliquer le shift (si nécessaire) sur l’image complète, mais avec un `shift_value` déterminé sur la zone valide.
- Si `alpha_mask` est absent ou incompatible : comportement actuel inchangé.

### B)  Neutralisation hors-footprint UNIQUEMENT pour les stats (sans modifier les pixels écrits)
But : empêcher un outlier hors-footprint de piloter les *statistiques* (min/max/percentiles) utilisées pour décider du baseline shift.

Règles MUST :
- La neutralisation `ALPHA==0` (ex: mise à NaN) doit se faire **dans une vue/buffer dédié aux stats**,
  pas sur `data_to_write_temp` qui sera écrit dans le FITS.
- En clair : `stats_view` peut contenir des NaN hors-footprint, mais `data_to_write_temp` (écrit) ne doit pas être “nanisé” par cette option.
- La seule modification “pixel data” autorisée dans `save_as_float=True` est la logique existante de baseline shift (et son clamp à 0),
  mais le **shift_value** doit être déterminé à partir des pixels valides (A0/A).
- Interdit : multiplier les pixels par le masque / double pondération / changer les valeurs écrites en fonction d’ALPHA, hors baseline-shift.

### C) Logs de diagnostic (à garder en DEBUG/INFO_DETAIL)
Ajouter un log explicite quand `alpha_mask` est utilisée, par exemple :
- `SAVE_DEBUG: baseline_shift using ALPHA>0: valid_min=... valid_max=... global_min=... global_max=... shift=...`

### D) Cas dégénérés (masque vide / aucun pixel valide fini) — fallback obligatoire
- Si `alpha_mask` est fourni mais que `valid_hw` ne contient aucun pixel (`valid_hw.any() == False`)
  OU qu’il n’existe aucun pixel fini dans l’intersection (ex: `np.isfinite(image_data)` ∩ `valid_hw`) :
  => comportement actuel inchangé (global `np.nanmin`), + log WARN explicite :
     `SAVE_DEBUG: alpha_mask provided but no finite valid pixels; falling back to global-min baseline shift.`
- Objectif : éviter les erreurs “min of empty slice” et toute dérive silencieuse.

## Critères d’acceptation
- En régénérant `master_tile_010.fits` et `master_tile_021.fits` :
  - la ligne `Baseline shift ...` devient petite (ordre 1e2–1e4 ADU), pas 1e8.
  - le cube `RGB` a une gamme compatible viewer (pas un offset ~1e8 avec une dynamique relative minuscule).
  - l’extension `ALPHA` reste cohérente (majoritairement 255, bordures à 0).
- Aucun changement de comportement pour les écritures float **sans** `alpha_mask`.

## Plan d’implémentation (proposé)
- [x] 1) Dans `zemosaic_utils.py:save_fits_image` (branche `save_as_float`), construire `valid_hw = (alpha_mask > 0)` si possible.
- [x] 2) Adapter le broadcast du masque selon `axis_order` et `image_data.ndim` (HWC vs CHW).
- [x] 3) Calculer `finite_min` sur `data_to_write_temp[valid]` (en tenant compte des NaN).
- [x] 4) Appliquer la logique de shift (min>0 => soustraction, min<0 => addition) basée sur `valid_min`.
- [ ] 5) (Optionnel) Forcer `data_to_write_temp[~valid] = np.nan` avant d’écrire.
- [x] 6) Durcir `_normalize_images_linear_fit` quand `src_high-src_low` est trop petit ou quand `a` devient extrême (skip normalization).
- [ ] 7) Régénérer les FITS et vérifier :
   - pas d’outliers ~1e8 dans `ALPHA==0`,
   - `baseline shift` raisonnable,
   - dynamique viewer normale.

## Notes (suspect amont, confirmé)
Les coefficients `linear_fit` peuvent exploser quand `src_high-src_low` est quasi nul.
Cela crée des outliers hors footprint (positifs/négatifs) qui écrasent la dynamique en viewer.
