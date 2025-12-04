# 🧩 Mission : Ajouter un nouveau « Mode Grid/Survey » dans ZeMosaic  
### 🎯 Objectif

Ajouter une **voie de traitement entièrement nouvelle**, **activée uniquement si** un fichier `stack_plan.csv` est présent dans le dossier d’entrée.  
**Le pipeline classique actuel ne doit JAMAIS être modifié.**

Le Mode Grid/Survey permet de :

- traiter des images multi-nuits / multi-sites / multi-mount / multi-sessions  
- ignorer complètement le clustering traditionnel  
- créer des master tiles **géométriques** basées sur une grille WCS régulière  
- assembler la mosaïque finale **sans aucune reprojection globale**  
- utiliser les infos de `stack_plan.csv` *sans jamais appeler Zenalyser*

---

# 🧱 0. Règles de codage obligatoires

- ❗ Ne modifier **aucune logic path** du pipeline standard.  
- ❗ Ne pas toucher aux fichiers existants liés au clustering classique.  
- ✔ Ajouter une **nouvelle voie** dans `zemosaic_worker.py` (ou un fichier séparé importé).  
- ✔ Condition d’activation : la présence de `stack_plan.csv` dans le dossier d’entrée.  
- ✔ Encapsuler tout le code Grid/Survey dans des fonctions dédiées (`run_grid_mode`, etc.).  
- ✔ PAS de duplication de code inutile.  
- ✔ Le pipeline classique doit fonctionner **strictement à l’identique**.

---

# 🧭 1. Détection du Mode Grid/Survey

Dans `zemosaic_worker.py` :

- Ajouter une fonction :  
  `detect_grid_mode(input_folder) → bool`  
  qui retourne True si `stack_plan.csv` existe dans le dossier.

- Dans la fonction principale :  

```python
if detect_grid_mode(folder):
    return run_grid_mode(folder)
else:
    return run_standard_mode(folder)
````

Aucun autre changement au pipeline standard.

---

# 🌐 2. Lecture de `stack_plan.csv`

Créer une fonction :

```python
load_stack_plan(csv_path) → List[FrameInfo]
```

Chaque `FrameInfo` doit contenir :

* file_path
* exposure (float)
* bortle (string or category)
* filter
* batch_id (string)
* order (int)

Tout le reste est ignoré.

---

# 🗺️ 3. Construction du WCS global + grille géométrique

Créer une fonction :

```python
build_global_grid(frames, grid_size_factor, overlap_factor)
```

* Lire le WCS de chaque `file_path`.
* Reprojeter les centres RA/Dec → coords X,Y d’un WCS global.
* Déterminer le bounding-box global.
* Construire une grille régulière :

  * Taille du carré = (FOV / grid_size_factor)
  * Overlap = overlap_factor (valeur GUI existante)

Retour attendu :

```python
return tiles  # List[Tile]
```

Chaque `Tile` contient :

* tile_id
* bounding box
* WCS local (aligné avec WCS global)
* liste vide d’images

---

# 📥 4. Affectation des brutes aux tiles (footprint test)

Créer une fonction :

```python
assign_frames_to_tiles(frames, tiles)
```

Pour chaque frame :

* Déterminer quels tiles elle intersecte (test centre+FOV ou polygon).
* Ajouter le frame dans `tile.frames`.

✔ Une brute peut aller dans plusieurs tiles → comportement normal.

---

# 🧪 5. Coadd local (SupaDupStack-like) par tile

Créer une fonction :

```python
process_tile(tile, output_folder)
```

Pour chaque tile :

1. Pour chaque frame :

   * charger la zone intersectante
   * reprojeter la zone dans le WCS du tile
2. Empiler :

   * normalisation photométrique locale
   * pondération (SNR, bortle, expo)
   * sigma/winsor/kappa
3. Sauvegarder le résultat dans :
   `output_folder/tiles/tile_<id>.fits`

---

# 🧩 6. Assemblage final (sans reprojection globale)

Créer une fonction :

```python
assemble_tiles(tiles, wcs_global, output_path)
```

* Allouer l’image de sortie complète.
* Pour chaque tile :

  * placer directement ses pixels aux coordonnées globales (pas de reprojection)
  * cumuler selon une carte de poids interne
* Après assemblage :

  * appliquer une **normalisation large échelle** (fond global)

Résultat final écrit dans :
`mosaic_grid.fits`

---

# 🧲 7. Intégration complète

Créer une fonction maître :

```python
def run_grid_mode(folder):
    frames = load_stack_plan()
    tiles = build_global_grid()
    assign_frames_to_tiles()
    for tile in tiles:
        process_tile(tile)
    assemble_tiles()
```

---

# 📌 8. Respect absolu du pipeline classique

* Le mode Grid/Survey **n’a pas le droit** de toucher :

  * clustering classique
  * master tiles actuelles
  * phases 3–5 actuelles
* Ce mode constitue un **pipeline parallèle** 100% indépendant.

---

# 📦 9. Livrables Codex

Vous devez fournir :

* [ ] Le code complet du mode Grid/Survey
* [ ] Les nouveaux fichiers éventuels (grid_utils.py, wcs_grid.py…)
* [ ] Les modifications strictes et minimalistes dans zemosaic_worker.py
* [ ] Du code totalement isolé pour ne rien abîmer ailleurs
* [ ] Les logs proprement taggés `[GRID]`
* [ ] Une option GUI simple “Grid/Survey (auto si stack_plan.csv)” (facultative)

---

# 🧪 10. Tests d’acceptation

* [ ] Pipeline classique fonctionne identique commit précédent
* [ ] Un dossier sans stack_plan.csv → mode standard
* [ ] Un dossier avec stack_plan.csv → mode Grid
* [ ] Aucun crash si une image n’a pas de WCS
* [ ] Mosaic finale = pas de reprojection globale
* [ ] Multi-nuit + multi-site + multi-mount OK
* [ ] Tiles alignées pixel-perfect dans le WCS global

