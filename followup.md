# followup — Alt-Az cleanup: tiles 010/021 “vides” (diagnostic + check après patch)

## Résumé (confirmé)
- Ce n’est **pas** un `try/except` qui “vide” silencieusement les tiles : le pipeline passe bien (Alt-Az cleanup appliqué + edge trim + quality gate ACCEPT).
- Les FITS `example/master_tile_010.fits` et `example/master_tile_021.fits` contiennent bien `RGB` + `ALPHA` (ALPHA majoritairement 255).
- Le symptôme “tile noire/plate” vient de **valeurs extrêmes hors footprint** qui écrasent l’auto-stretch des viewers :
  - `master_tile_021.fits`: `shift≈+517` (donc pas énorme), mais **max global ≈ 2.38e8** alors que `ALPHA>0` plafonne à ~4e4.
  - Les outliers sont **quasi exclusivement dans ALPHA==0**.

Indices concrets :
- `HISTORY Baseline shift applied for FITS float export: +339699840.000000 ADU` (010)
- `HISTORY Baseline shift applied for FITS float export: +153565872.000000 ADU` (021)

Après soustraction du shift :
- zone `ALPHA>0`: valeurs ~0–50k, percentiles typiques ~640–1100 ADU
- zone `ALPHA==0`: outliers très négatifs qui forcent le `min` global → shift gigantesque → dynamique écrasée

## Hypothèse amont (confirmée par log)
`linear_fit` peut produire des coefficients énormes quand `src_high - src_low` est quasi nul :
- `zemosaic_worker.log` (tile 21) : `a=3.21e6`, `b=-2.84e9` (ex: `Src(L/H)=(886/886)`).
- Cela injecte des valeurs extrêmes (positives ou négatives) **hors footprint**, qui dominent le `max` global et rendent la tile “plate” en viewer.

## Correctif “safe” recommandé (1 endroit)
Dans `zemosaic_utils.py:save_fits_image` (mode `save_as_float=True`) :
- [x] quand `alpha_mask` est fourni, calculer le baseline shift **sur les pixels valides** (`ALPHA>0`) et/ou neutraliser (`NaN`) les pixels `ALPHA==0` avant de calculer le min.

But : éviter qu’un seul pixel invalide très négatif pilote le shift.
Note : ce correctif **réduit le shift**, mais ne supprime pas les **outliers positifs** hors footprint.

## Correctif nécessaire (piste 2)
- [x] Durcir `linear_fit` quand `src_high-src_low` est trop petit ou quand les coefficients deviennent extrêmes :
  - fallback “skip normalization” pour éviter les valeurs gigantesques hors footprint.

## Check rapide après patch (sur ces 2 FITS)
- [ ] Vérifier sur des FITS régénérés :
  - la ligne `Baseline shift ...` n’est plus de l’ordre de `1e8`.
  - `max(ALPHA==0)` n’écrase plus la dynamique (pas d’outliers ~1e8).
  - dans `ALPHA>0`, `min≈0` et `max` reste raisonnable.

## Si le symptôme persiste
Autre piste :
- “naniser” systématiquement les zones hors recouvrement avant normalisation pour éviter les valeurs extrêmes hors footprint.
