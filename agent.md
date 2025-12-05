# ZeMosaic – Grid mode RGB / shape fix

## Contexte

Le mode **Grid** progresse bien mais produit des FITS de tiles qui ne semblent pas cohérents avec le pipeline classique :

- Les images sources sont des RGB Seestar, lues comme `raw_shape=(1920, 1080)` et `hwc_shape=(1920, 1080, 3)` avec `axis_orig=HWC, bayer=GRBG, debayered=True`.  
- Les reprojections par tile se font en `shape=(1920, 1920, 3)` pour la tile 1, et `shape=(1920, 1254, 3)` etc. pour les autres tiles, donc a priori tout est bien en HWC float32.   
- Pourtant, le fichier `tile_0001.fits` semble avoir une géométrie "3 x 1920" (ou interprétée comme telle) et le résultat global apparaît en **niveau de gris**, alors que les master tiles classiques sont correctement en couleur.

Hypothèse : il y a un bug dans la façon dont `grid_mode` :

1. **Convertit** les arrays HWC en cube FITS (ordre des axes),
2. **Applique la normalisation RGB / stacking**, éventuellement en aplatissant la dimension couleur,
3. **Écrit** le FITS final, avec des métadonnées incohérentes avec le reste du pipeline.

Objectif : corriger ça proprement sans casser le pipeline classique ni la logique Grid, en s’alignant strictement sur les conventions de `zemosaic_utils` et du pipeline standard.

---

## Fichiers et zones à inspecter

- `grid_mode.py`
  - Toute la chaîne : lecture des frames, reprojection, stacking par tile, écriture des `tile_XXXX.fits`.
  - Fonctions/méthodes utilisées pour :
    - Normalisation RGB (équivalent de `equalize_rgb_medians_inplace`).
    - Stacking (moyenne / médiane / sigma-clipping / etc.).
    - Conversion HWC ↔ cube FITS.
- `zemosaic_utils.py`
  - Fonctions de lecture/validation d’images :
    - `load_and_validate_fits` (ou équivalent).
    - Éventuelles fonctions utilitaires de débayérisation / réorganisation d’axes.
  - Fonctions d’écriture :
    - Toute fonction qui crée des cubes RGB FITS ou des HDU RGB (legacy, HWC/CHW, etc.).
- Pipeline classique dans `zemosaic_worker.py` / `zemosaic_align_stack.py`
  - Comment sont écrites les **master tiles classiques** (stack par group, WCS, FITS final).
  - Comment la normalisation RGB y est branchée.

Le but est d’aligner le comportement Grid sur ce que fait le pipeline classique quand il produit des sorties RGB.

---

## Conventions attendues

- **Interne Grid mode :**
  - Les images **doivent rester en HWC (H, W, C)** tant qu’on n’a pas explicitement besoin de passer en layout spécifique pour FITS.
  - H = hauteur (rows, NAXIS2), W = largeur (cols, NAXIS1), C = 3 canaux (R,G,B).

- **Écriture FITS (à calquer sur le pipeline classique) :**
  - Soit on écrit un cube FITS en **CHW (C, H, W)**, avec les mêmes conventions/keywords que le pipeline classique,
  - Soit en HWC si le reste du code sait le relire, mais **il faut impérativement utiliser la même convention que le reste du projet** (pas d’invention locale dans Grid mode).

- **RGBEqualize / stacking :**
  - La normalisation RGB (`grid_rgb_equalize=True/False` dans la config) doit appeler exactement la même logique que le pipeline classique **par canal**, pas sur un array aplatit.
  - Le stacking doit utiliser les mêmes fonctions (moyenne, médiane, sigma-kappa, etc.) que le pipeline classique, avec la même interface.

---

## Tâches

### 1. Cartographie des shapes et des conversions

- [ ] Localiser toutes les fonctions/utilisations dans `grid_mode.py` qui :
  - manipulent des arrays d’images RGB,
  - changent l’ordre des axes (transposes, `moveaxis`, `reshape`, etc.),
  - écrivent des FITS pour les tiles (`tile_XXXX.fits`).
- [ ] Pour chacune :
  - Documenter (en commentaires ou docstring courte) **quelle convention de shape est attendue en entrée/sortie** (HWC, CHW, HW, etc.).
- [ ] Ajouter des logs `[GRID] DEBUG_SHAPE_WRITE` juste avant l’écriture de chaque tile :
  - Nom du fichier (ex: `tile_0001.fits`),
  - Shape du tableau en mémoire,
  - Type et min/max par canal si possible.

### 2. Aligner la création du cube FITS sur la convention globale

- [ ] Inspecter dans `zemosaic_utils.py` comment les cubes RGB sont habituellement construits :
  - Quelle est la forme stockée dans le FITS : (H, W, C) ou (C, H, W) ?
  - Comment les HDU sont créés (`PrimaryHDU`, `ImageHDU`, `HDUList`…).
  - Quels mots-clés d’en-tête (header) sont utilisés pour indiquer la présence de 3 canaux.
- [ ] Modifier `grid_mode.py` pour qu’il **utilise exactement la même fonction utilitaire** pour écrire une image/cube RGB que le pipeline classique, plutôt qu’une logique ad hoc.
  - Si une fonction utilitaire manque, en créer une générique dans `zemosaic_utils.py` et l’utiliser **à la fois** dans le pipeline classique et dans Grid mode.
- [ ] S’assurer qu’il n’y a pas de double transposition HWC→CHW→HWC qui provoquerait un cube bizarre (ex: 3x1920 au lieu de 1920x1920x3).

### 3. Vérifier / unifier la normalisation RGB

- [ ] Rechercher comment la normalisation RGB est appelée dans le pipeline classique (ex: `equalize_rgb_medians_inplace` ou similaire).
- [ ] Vérifier comment Grid mode applique la normalisation :
  - Sur un array HWC,
  - Sur chaque tile,
  - En tenant compte du flag `grid_rgb_equalize`.
- [ ] Faire en sorte que **Grid mode appelle exactement le même helper que le pipeline classique**, avec la même sémantique :
  - même mode d’égalisation,
  - même ordre de canaux,
  - même type (float32),
  - aucun aplatissement qui détruit la structure RGB.

### 4. Vérifier les méthodes de stacking dans Grid mode

- [ ] Lister les différents modes de stacking disponibles dans le pipeline classique (moyenne, médiane, sigma-kappa, etc.).
- [ ] Vérifier que Grid mode utilise les **mêmes fonctions** ou, à défaut, des wrappers qui respectent :
  - la même logique de rejet/sigma-clipping,
  - les mêmes options de pondération/exclusion.
- [ ] S’assurer que le stacking fonctionne bien **par canal**, et ne mélange pas les 3 canaux dans une seule dimension.

### 5. Test de non-régression minimal

Créer un test ou un script de test minimal (même simple script Python si pas de framework déjà en place) qui :

- [ ] Génère un petit jeu d’images synthétiques RGB 3-canaux en HWC (ex: 64x64x3) avec :
  - un canal R = gradient horizontal,
  - un canal G = gradient vertical,
  - un canal B = constant ou pattern différent.
- [ ] Simule un petit stack_plan avec 1 seule tile.
- [ ] Fait tourner Grid mode sur ce dataset synthétique pour produire `tile_0001.fits`.
- [ ] Relit `tile_0001.fits` avec `zemosaic_utils.load_and_validate_fits` :
  - Vérifie que la shape est bien celle attendue (convention choisie),
  - Vérifie que les 3 canaux sont **distincts** (ex: médiane R != G != B).
- [ ] Optionnel : générer une image PNG de debug pour visualiser la tile et confirmer visuellement la présence des couleurs.

---

## Contraintes / garde-fous

- Ne rien casser dans le pipeline classique :
  - Ne pas changer la convention de shape globale pour tout le projet sans audit complet.
  - Si une nouvelle convention est nécessaire, la gérer derrière un flag explicite ou une fonction utilitaire très clairement nommée.
- Rester strict sur les types :
  - garder du `float32` pour le stacking,
  - gérer les conversions vers `uint16` ou autres seulement au moment de l’écriture, en harmonisation avec `zemosaic_utils`.
- Conserver les logs détaillés `[GRID]`, mais éviter le spam en mode normal :
  - Les logs de shape finaux peuvent être en DEBUG.

---

## Ce que tu peux modifier

- `grid_mode.py`
- `zemosaic_utils.py` (helpers de lecture/écriture/normalisation RGB)
- Éventuellement un petit fichier de tests (ex: `tests/test_grid_mode_rgb.py` ou un script `dev/test_grid_mode_rgb.py`)

N’ajoute pas de nouvelle dépendance lourde.

Lorsque tu as fini :
- Assure-toi qu’un run Grid mode sur un dataset Seestar RGB donne :
  - des `tile_XXXX.fits` **correctement RGB**,
  - une mosaïque finale en couleur,
  - des shapes cohérentes dans les logs `[GRID] DEBUG_SHAPE_*`.
