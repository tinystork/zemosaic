# ZeMosaic – Grid mode RGB / shape fix – Follow-up

## Contexte / symptômes observés

- Dataset Seestar RGB.
- Pipeline classique : master tiles correctes, images RGB 1080x1920, empilées proprement.
- Grid mode :
  - Logs montrent des frames chargées en `raw_shape=(1920,1080)` et `hwc_shape=(1920,1080,3)` avec `axis_orig=HWC, bayer=GRBG, debayered=True`.   
  - Reprojection par tile produit des arrays `shape=(1920,1920,3)` ou `(1920,1254,3)` (tile_size_px=1920, overlap=0.4).   
  - Les `tile_000X.fits` semblent avoir une géométrie interprétée comme ~`3 x 1920` et la mosaïque finale apparaît **en niveaux de gris**, alors que la géométrie est globalement correcte.

Hypothèse forte : bug dans la conversion HWC → cube FITS + normalisation RGB, différents du pipeline classique.

---

## Tâches (à cocher au fur et à mesure)

### 1. Cartographie des shapes et conversions
- [ ] Revue complète de la chaîne images dans `grid_mode.py` (lecture → reprojection → stacking → écriture).
- [ ] Ajout de logs `[GRID] DEBUG_SHAPE_WRITE` juste avant l’écriture de chaque tile FITS, incluant shape et stats par canal.

### 2. Alignement avec `zemosaic_utils` pour l’écriture FITS
- [ ] Revue de `zemosaic_utils.py` pour identifier la convention standard des cubes RGB et les helpers d’écriture.
- [ ] Remplacement de toute logique custom dans `grid_mode.py` par l’utilisation de ces helpers (ou nouvelle fonction utilitaire partagée).
- [ ] Vérification que l’ordre des axes dans le FITS est **identique** à celui du pipeline classique.

### 3. Normalisation RGB
- [ ] Vérification de l’appel de la normalisation RGB dans le pipeline classique (ex: `equalize_rgb_medians_inplace`).
- [ ] Mise en conformité de Grid mode :
  - usage du même helper,
  - respect du flag `grid_rgb_equalize`,
  - traitement canal par canal sans aplatir la structure RGB.

### 4. Méthodes de stacking
- [ ] Comparaison des méthodes de stacking Grid vs pipeline classique (moyenne, médiane, sigma-kappa, etc.).
- [ ] Harmonisation :
  - utilisation des mêmes fonctions/utilities,
  - même sémantique d’options et de rejets.

### 5. Test de non-régression
- [ ] Création d’un test minimal (ou script) avec un dataset synthétique RGB HWC :
  - ex: 3 canaux bien distincts (R, G, B) sur une tile.
- [ ] Exécution de Grid mode pour produire une `tile_0001.fits`.
- [ ] Relecture avec `zemosaic_utils.load_and_validate_fits` pour vérifier :
  - shape conforme,
  - 3 canaux distincts.
- [ ] (Optionnel) Génération d’une image PNG de debug pour confirmer visuellement la couleur.

---

## Notes / pièges à éviter

- Ne pas changer la convention globale de shape pour tout le projet sans y être explicitement invité.
- Ne pas casser le pipeline classique : toute modification dans `zemosaic_utils` doit rester rétrocompatible.
- Éviter les conversions multiples HWC ↔ CHW inutiles qui peuvent introduire des erreurs silencieuses.

---

## Résultat attendu

- Les tiles produites par Grid mode (`tile_000X.fits`) :
  - ont la même structure RGB que celles du pipeline classique,
  - sont correctement interprétées comme RGB par les viewers et par `zemosaic_utils`,
  - ne “collapsent” pas en quelque chose qui ressemble à un `3 x 1920` gris.
- La mosaïque finale en mode Grid est en **couleur correcte**, avec la même qualité de normalisation RGB et de stacking que la mosaïque classique.

