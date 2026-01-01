# followup.md

## Résultat attendu
- Les master tiles **vides / quasi-vides** ne contribuent plus à la mosaïque (poids=0 / alpha=0) → disparition des rectangles/bandes sombres.
- Le garde-fou protège aussi `use_existing_master_tiles_mode` (tuiles déjà générées) sans changer le comportement des tuiles normales.

## Périmètre / anti-régression (SDS, GRID, master-tile-only)
- Ne modifier que le pipeline **Master Tiles** dans `zemosaic_worker.py` (+ tests). Ne pas toucher `grid_mode.py`.
- Ne pas modifier le pipeline **SDS** : le garde-fou “empty tile” doit être **gated** sur les master tiles (ex: `ZMT_TYPE="Master Tile"` ou `ZMT_ID` présent).
- Le fallback heuristique (tiles anciennes sans flag) ne doit s’appliquer **que** sur des master tiles, jamais sur des images SDS/GRID génériques.

## Artefacts / repro
- `example/master_tile_021.fits` : `(3,32,32)` avec `RGB max≈9.313e-10` et `ALPHA=255` partout → doit être neutralisée.
- `zemosaic_worker.log` : contient `WeightNoiseVar ... stddev invalide (9.313e-10) -> Variance Inf` (indice “stack effectif nul”) ; ajouter un log greppable `MT_EMPTY_TILE`.

## Checklist exécution (Codex)
### 1) Patch ciblé
- [x] Ajouter `_detect_empty_master_tile(...)` dans `zemosaic_worker.py` (helper local, pur, testé).
- [x] Dans `create_master_tile(...)` :
  - [x] Appeler `_detect_empty_master_tile` **après** calcul de `alpha_mask_out` (et après crop/trim éventuels).
  - [x] Si empty :
    - [x] Forcer `alpha_mask_out = zeros` (uint8).
    - [x] Neutraliser `master_tile_stacked_HWC` (idéalement `NaN` si float32) **uniquement à la fin**, juste avant `save_fits_image`.
    - [x] Ajouter un flag FITS (header primaire, mots-clés ≤ 8 chars) : `ZMT_EMPT=1` + stats (`ZMT_EMAX`, `ZMT_ESTD`, `ZMT_EVF`, `ZMT_EPS`).
    - [x] Log WARN greppable : `MT_EMPTY_TILE tile=<id> ... forced transparent`.
- [x] Dans `load_image_with_optional_alpha(...)` :
  - [x] Lire le header primaire (`ZMT_TYPE`, `ZMT_ID`, `ZMT_EMPT`).
  - [x] Définir “master tile” (gating) : `ZMT_TYPE="Master Tile"` **ou** `ZMT_ID` présent ; sinon ne rien heuristiquement neutraliser.
  - [x] Si master tile et `ZMT_EMPT=1` : renvoyer `weights=0`, `alpha=0`, et data `NaN`.
  - [x] Fallback heuristique (rétro-compat, **master tiles uniquement**) :
    - [x] `alpha_frac` très élevé + `max_abs < eps_signal` (seuil conservateur, ex `1e-8`) → neutraliser.
- [x] `reproject_tile_to_mosaic(...)` : aucun changement requis si `load_image_with_optional_alpha` renvoie `weights=0` pour une tuile empty (le `footprint_full` devient vide et la tuile est ignorée).

### 2) Tests
- [x] Ajouter `tests/test_empty_master_tile_guard.py` (ou intégrer dans une suite existante) :
  - [x] Cas `ZMT_EMPT=1` → neutralisé.
  - [x] Cas sans flag mais data≈0 + alpha=255 → neutralisé par fallback (master tile).
  - [x] Cas avec 1 pixel réellement bright (max_abs >> eps) → **pas** neutralisé.
- [x] Tests Windows/Mac/Linux (astropy via `importorskip`).

### 3) Validation manuelle rapide
#### A) Scan offline des tuiles suspectes
Seuil de sanity recommandé : `eps_signal=1e-8`.

```powershell
@'
from pathlib import Path
import numpy as np
from astropy.io import fits

eps = 1e-8
for p in sorted(Path('example').glob('master_tile_*.fits')):
    with fits.open(p, memmap=False) as hdul:
        hdr = hdul[0].header if hasattr(hdul[0], 'header') else {}
        rgb = np.asarray(hdul[0].data, dtype=np.float32)
        alpha = np.asarray(hdul['ALPHA'].data, dtype=np.uint8) if 'ALPHA' in hdul else None
    ztype = hdr.get('ZMT_TYPE', None) if hasattr(hdr, 'get') else None
    max_abs = float(np.nanmax(np.abs(rgb))) if rgb.size else 0.0
    alpha_frac = float((alpha > 0).mean()) if alpha is not None else float('nan')
    if alpha is not None and alpha_frac > 0.99 and max_abs < eps:
        print(f'{p.name}: ZMT_TYPE={ztype!r} alpha_frac={alpha_frac:.6f} max_abs={max_abs:g}')
'@ | python -
```

#### B) Logs lors d’un run
- Présence de `MT_EMPTY_TILE tile=<id> ... forced transparent`.
- Les warnings `WeightNoiseVar ... Variance Inf` peuvent rester, mais ne doivent plus produire de tuiles opaques vides.

#### C) Mosaïque finale
- Rectangles/bandes “vides” disparus.
- Pas de “trous noirs inventés” : si une zone n’a pas de signal, elle doit être transparente (les overlaps comblent).

## Rollback
- Revert uniquement les blocs `MT_EMPTY_TILE` + helper + tests.

---
