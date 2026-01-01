# agent.md

## Mission
Empêcher les **master tiles vides / quasi-vides** d’être traitées comme **opaques** (ALPHA=255) et de **polluer la mosaïque finale** (rectangles/bandes sombres).

## Périmètre (anti-régression “idiot-proof”)
- Ne cibler que le pipeline **Master Tiles** (création + consommation des master tiles) dans `zemosaic_worker.py` (+ tests).
- Ne pas toucher `grid_mode.py` (Grid Mode a son propre loader et sa logique dédiée).
- Ne pas modifier le pipeline **SDS** (mega-tiles / coadd SDS) : la détection “empty tile” doit être **gated** sur les master tiles.
- En mode `use_existing_master_tiles_mode` / `existing_master_tiles_mode`, le garde-fou doit fonctionner (lecture de tuiles existantes) sans changer le comportement des tuiles normales.

## Diagnostic (confirmé sur les artefacts fournis)
- `example/master_tile_021.fits` : `(3,32,32)` avec `RGB max≈9.313e-10` mais `ALPHA=255` partout → tuile “vide” mais “valide”.
- `zemosaic_worker.log` : nombreuses lignes `WeightNoiseVar ... stddev invalide (9.313e-10) -> Variance Inf` → poids/variance qui dégénère, donc contribution réelle ≈ 0.

## Cause probable
- Certaines tuiles arrivent à un **stack effectif nul** (`sum(weights)==0` ou équivalent) → sortie ~0.
- L’ALPHA est dérivée d’un critère trop faible (ex: “finite” / footprint attendu) → reste à 255 même si aucune contribution réelle.

## Stratégie de fix (à faire en double verrou)
### A) À l’écriture d’une master tile (`create_master_tile(...)`)
- [x] Ajouter un helper `_detect_empty_master_tile(rgb_hwc, alpha_u8, ...) -> (is_empty, stats)` basé sur le **contenu réel**:
  - [x] `valid = isfinite & (abs(R)+abs(G)+abs(B) > eps_signal)` (pas `==0`).
  - [x] Option prioritaire si disponible : `valid = (sum_weights > 0)` pour coller à “a réellement contribué”.
- [x] Si `is_empty` (ou `valid_frac < seuil_min`):
  - [x] Forcer `alpha_mask_out = 0` partout.
  - [x] Neutraliser la donnée (idéalement `NaN` sur float32) **après** le stack, juste avant la sauvegarde, pour éviter toute pollution si l’alpha est ignorée plus tard.
  - [x] Écrire un flag FITS (header primaire, mots-clés ≤ 8 chars) : `ZMT_EMPT=1` + stats (`ZMT_EMAX`, `ZMT_ESTD`, `ZMT_EVF`, `ZMT_EPS`).
  - [x] Log clair et greppable : `MT_EMPTY_TILE tile=<id> ... forced transparent`.

### B) À la lecture (`load_image_with_optional_alpha(...)`)
- [x] Si header `ZMT_EMPT=1` : retourner `weights=0` (et `alpha=0`) + data `NaN`.
- [x] Fallback rétro-compatible (pour master tiles déjà générées, sans flag) :
  - [x] **Uniquement si** le FITS est bien une master tile (ex: `ZMT_TYPE="Master Tile"` ou `ZMT_ID` présent).
  - [x] Si `alpha_frac` très élevé **et** `max_abs < eps_signal` (ou `std < std_min`) → traiter comme vide, neutraliser.
  - [x] Garder des seuils **très conservateurs** (ex: `eps_signal=1e-8`) pour éviter les faux positifs.

### C) Sécurité assemblage (`reproject_tile_to_mosaic(...)`)
- [x] Vérifier qu’une tuile “empty” est naturellement ignorée (si `weights`/alpha est tout à 0, `combined_weight` ⇒ footprint vide ⇒ bbox vide ⇒ skip).

## Contraintes / anti-régression
- Ne pas changer clustering / normalisation / lecropper (hors ajout de logs/flags/guard).
- Le fix doit protéger aussi `use_existing_master_tiles_mode` (où on relit des tuiles existantes).

## Observations critiques (garde-fous supplémentaires)
- Gating legacy : la neutralisation doit couvrir les master tiles déjà produites (sans flag `ZMT_EMPT`), y compris en `use_existing_master_tiles_mode` ; activer l’heuristique dès qu’un header master tile est présent (`ZMT_TYPE="Master Tile"` ou `ZMT_ID`), jamais sur SDS/GRID.
- CHW/HWC + NaN-safe : `_detect_empty_master_tile` et les stats doivent supporter `(C,H,W)` comme `(H,W,C)` sans crash ; utiliser des ops `np.nan*` (nanmax/nanstd, masque NaN-safe) pour éviter ValueError ou faux positifs en présence de NaN/Inf.

## Critères d’acceptation
- Charger `example/master_tile_021.fits` ne doit **plus** produire une tuile contributive (poids=0/skip effectif).
- Plus de rectangles/bandes “vides” dans la mosaïque finale sur le dataset repro.
- Tests unitaires ajoutés (petites matrices synthétiques; astropy optionnel via `importorskip`).
