## Mission (NO REFACTOR / SCOPE MINIMAL)

Corriger 2 régressions précises dans ZeMosaic, sans modifier autre chose :

1) **Preview PNG entièrement noir**
   - Cause : l'image preview est NaN-masquée avant stretch, mais le stretch utilise `percentile` (CPU: `np.percentile`, GPU: `cp.percentile`) qui ne gère pas les NaN → `vmin/vmax = NaN` → stretch = NaN → `nan_to_num` à l’export → RGB = 0 partout → preview noir.
   - Fix : rendre `stretch_auto_asifits_like()` et `stretch_auto_asifits_like_gpu()` **NaN-aware** (nanpercentile ou filtering finite) + garde-fous si pas de pixels finis.

2) **Pondération “N images dans la master tile” non prise en compte**
   - Cause (1) : dans `assemble_final_mosaic_reproject_coadd()`, `_extract_tile_weight()` ne lit que `MT_NFRAMES / ZMT_NALGN / ZMT_NRAW` alors que certaines tuiles existantes ont `NRAWPROC` / `NRAWINIT`.
   - Cause (2) : la pondération doit être appliquée via le mécanisme existant `tile_weights` (déjà supporté CPU/GPU dans `reproject_and_coadd_wrapper`). Il faut donc **s’assurer que `tile_weight_value` n’est pas 1 par défaut** à cause des mauvais keywords.

⚠️ Interdictions (IMPORTANT)
- Ne modifier AUCUNE logique de clustering / création de master tiles / phases 0-3.
- Ne pas toucher aux comportements existants “batch size = 0” et “batch size > 1”.
- Ne pas changer la logique “existing_master_tiles_mode / I'm using master tiles”.
- Pas de refactor, pas de déplacement de code, pas de nouveaux modules.
- Ne pas “double-appliquer” les poids (ne pas multiplier `input_weights` côté worker si `tile_weights` est déjà passé).

---

## Fichiers autorisés à modifier (UNIQUEMENT)

- `zemosaic_utils.py`
  - `stretch_auto_asifits_like()`
  - `stretch_auto_asifits_like_gpu()`

- `zemosaic_worker.py`
  - fonction interne `_extract_tile_weight()` dans `assemble_final_mosaic_reproject_coadd()`

Aucun autre fichier ne doit être modifié.

---

## Détails des modifications attendues

### A) Fix preview noir (CPU)

Dans `zemosaic_utils.py`, fonction :
`def stretch_auto_asifits_like(img_hwc_adu, p_low=..., p_high=..., asinh_a=..., apply_wb=True):`

Actuel (problématique) :
- `vmin_f64, vmax_f64 = np.percentile(chan, [p_low, p_high])`

Remplacer par une version NaN-aware :

- Utiliser `np.nanpercentile(chan, [p_low, p_high])`
- Ajouter garde-fou :
  - si aucune valeur finie (`np.isfinite(chan).any()` == False) → `out[...,c]=0` et `continue`
  - si `vmin/vmax` non finis → `out[...,c]=0` et `continue`
  - si `dv < 1e-3` → idem

⚠️ Conserver la philosophie “float32 only / pas de gros temporaires”.

### B) Fix preview noir (GPU)

Dans `zemosaic_utils.py`, fonction :
`def stretch_auto_asifits_like_gpu(...)`

Actuel (problématique) :
- `vmin = cp.percentile(chan, p_low)`
- `vmax = cp.percentile(chan, p_high)`

Remplacer par NaN-aware :
- Préférer `cp.nanpercentile(chan, p_low/high)` si dispo
- Sinon fallback minimal : calculer percentiles sur `chan[cp.isfinite(chan)]` (attention si vide)

Même garde-fou :
- si aucune valeur finie → canal = 0
- si `vmax - vmin < 1e-3` → canal = 0
- éviter d’introduire de nouveaux gros buffers (preview seulement, mais rester prudent)

### C) Fix pondération N images (keywords)

Dans `zemosaic_worker.py`, dans `assemble_final_mosaic_reproject_coadd()`,
fonction interne `_extract_tile_weight(header_obj)` :

Actuel :
```py
for key in ("MT_NFRAMES", "ZMT_NALGN", "ZMT_NRAW"):
````

Modifier en :

```py
for key in ("MT_NFRAMES", "ZMT_NALGN", "ZMT_NRAW", "NRAWPROC", "NRAWINIT"):
```

* Garder le même comportement : premier keyword trouvé, valeur >0 et finie, renvoyée.
* Ne pas changer le reste de la logique `tile_weighting_active`, `tile_weighting_applied`, ni l’endroit où `tile_weights` est passé au wrapper.
* IMPORTANT : ne pas multiplier `input_weights_list` par `tile_weight` (le wrapper CPU/GPU gère déjà `tile_weights` correctement).

---

## Critères d’acceptation

1. Les `*_preview.png` ne sont plus noirs quand `preview_view` contient des NaN (zones transparentes).

   * Les RGB doivent refléter l’image (et plus être tout à 0), l’alpha reste inchangé.

2. En mode “existing master tiles / skip clustering master tile creation” :

   * Si les FITS d’entrée contiennent `NRAWPROC`/`NRAWINIT`, les logs de pondération ne montrent plus uniquement des `1.0`.
   * Les poids sont appliqués via `tile_weights` (CPU et GPU) sans double pondération.

3. Aucun changement visible sur les autres modes/pipelines (pas de régression).

---

## Notes de test (sans ajouter de tests au repo)

* Test rapide preview :

  * fabriquer un petit tableau HWC avec une zone NaN et une zone signal, appeler `stretch_auto_asifits_like()` → le retour doit contenir du signal non nul sur la zone valide, pas tout zéro.

* Test pondération :

  * ouvrir un FITS de master tile existante qui a `NRAWPROC=197`, vérifier que `_extract_tile_weight()` renvoie 197.
  * lancer un run et vérifier que la summary min/max/mean des poids correspond aux headers (pas 1.0 partout).

