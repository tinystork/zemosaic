# agent.md — Mission Grid mode / Infinite loop sur la grille de tuiles

## 1. Contexte

Projet : **ZeMosaic**, branche actuelle : `V4WIP` (en pratique, tu travailles sur le code tel qu’il est dans le repo local de l’utilisateur).

Le **Grid/Survey mode** se base sur `grid_mode.py` pour :

1. Construire un WCS global + un canevas (`build_global_grid`).
2. Générer une grille régulière de tuiles (tiles) sur ce canevas.
3. Assigner les frames aux tuiles, puis les empiler.

Actuellement :

- Le WCS global se calcule bien (avec un fallback correct quand `shape_hw` dégénère).
- Le problème se produit **juste après** le log :

  ```text
  [GRID] [GRID] DEBUG: entering tile grid construction: tile_size_px=1920, step_px=1152, n_frames=53
````

* On voit dans le log :

  ```text
  [GRID] [GRID] DEBUG: estimated tiles: 6 (3 rows x 2 cols) for canvas (3272, 2406)
  [GRID] [GRID] DEBUG: entering tile grid construction: tile_size_px=1920, step_px=1152, n_frames=53
  [GRID] [GRID] DEBUG: built 50 tile(s) so far
  [GRID] [GRID] DEBUG: built 100 tile(s) so far
  ...
  [GRID] [GRID] DEBUG: built 565000 tile(s) so far
  ```

* On **ne voit jamais** la ligne :

  ```text
  [GRID] [GRID] DEBUG: grid definition ready with X tile(s) (rejected=Y, est=Z)
  ```

=> La fonction de construction de la grille ne termine jamais (boucle “infinie” ou quasi infinie).
=> Le worker reste figé sans crash, car aucune exception n’est levée.

Dans le fichier `grid_mode.py` (version actuelle), autour de `build_global_grid`, il y a un bloc de type :

```python
H, W = global_shape_hw
step_y = step_px
step_x = step_px
n_tiles_y = max(1, math.ceil((H - tile_size_px) / step_y) + 1)
n_tiles_x = max(1, math.ceil((W - tile_size_px) / step_x) + 1)
n_tiles_estimated = n_tiles_y * n_tiles_x
_emit("[GRID] DEBUG: estimated tiles: ...")

MAX_TILES = 50000
# éventuellement un warning si n_tiles_estimated > MAX_TILES

_emit("[GRID] DEBUG: entering tile grid construction: ...")

tiles: list[GridTile] = []
min_x_local = 0
max_x_local = int(global_shape_hw[1])
min_y_local = 0
max_y_local = int(global_shape_hw[0])
y0 = min_y_local
tile_id = 1
rejected_tiles = 0
while y0 < max_y_local:
    x0 = min_x_local
    while x0 < max_x_local:
        # calcul bbox, shape_hw, tile_wcs
        # ...
    # création de la tile, incrément de tile_id, x0, etc.
y0 += step_px
```

Problèmes probables :

* Structure en `while` fragile, avec un mélange entre :

  * Incréments de `x0` et `y0`,
  * Création de la tile (et logs) **mal indentée** par rapport à la boucle interne.
* Le code construit un nombre de tuiles totalement incohérent avec `n_tiles_estimated` (6 attendues vs > 500 000).
* Le garde-fou sur `MAX_TILES` n’est visiblement pas efficace (mal placé ou jamais déclenché dans la version réelle).

**Objectif global :**

* Corriger définitivement la construction de la grille pour qu’elle soit :

  * Déterministe,
  * Bornée,
  * Lisible (préférence pour des `for` explicites plutôt que des `while` fragiles),
  * Protégée par un garde-fou (`MAX_TILES`).

---

## 2. Fichiers concernés

* `grid_mode.py`

Ne touche à aucun autre fichier sauf si absolument nécessaire pour faire passer les imports ou types.
La mission doit être confinée à la **construction de la grille de tuiles** dans `build_global_grid`.

---

## 3. Tâches à accomplir

### Tâche 1 – Remplacer la double boucle `while` par des boucles bornées en `for`

Dans `build_global_grid`, juste après le log :

```python
_emit(
    f"[GRID] DEBUG: entering tile grid construction: "
    f"tile_size_px={tile_size_px}, step_px={step_px}, "
    f"n_frames={len(usable_frames)}",
    callback=progress_callback,
)
```

**Remplace complètement** la logique de construction de la grille par quelque chose de ce genre :

```python
tiles: list[GridTile] = []

min_x_local = 0
max_x_local = int(global_shape_hw[1])
min_y_local = 0
max_y_local = int(global_shape_hw[0])

tile_id = 1
rejected_tiles = 0

for j in range(n_tiles_y):
    y0 = min_y_local + j * step_px
    if y0 >= max_y_local:
        break

    for i in range(n_tiles_x):
        x0 = min_x_local + i * step_px
        if x0 >= max_x_local:
            break

        bbox_xmin = int(x0)
        bbox_xmax = int(min(x0 + tile_size_px, max_x_local))
        bbox_ymin = int(y0)
        bbox_ymax = int(min(y0 + tile_size_px, max_y_local))

        shape_hw = (bbox_ymax - bbox_ymin, bbox_xmax - bbox_xmin)
        if shape_hw[0] <= 0 or shape_hw[1] <= 0:
            rejected_tiles += 1
            continue

        tile_wcs = _clone_tile_wcs(
            global_wcs,
            (offset_x + bbox_xmin, offset_y + bbox_ymin),
            shape_hw,
        )

        tile = GridTile(
            tile_id=tile_id,
            bbox=(bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax),
            wcs=tile_wcs,
        )
        tiles.append(tile)

        if len(tiles) > MAX_TILES:
            _emit(
                f"[GRID] ERROR: built {len(tiles)} tiles > MAX_TILES={MAX_TILES}, "
                "aborting grid generation to avoid freeze",
                lvl="ERROR",
                callback=progress_callback,
            )
            raise RuntimeError(
                f"Grid tile generation aborted: too many tiles ({len(tiles)})"
            )

        if len(tiles) % 50 == 0:
            _emit(
                f"[GRID] DEBUG: built {len(tiles)} tile(s) so far",
                callback=progress_callback,
            )

        tile_id += 1
```

Points importants :

* **Plus de `while y0 < max_y_local` + `while x0 < max_x_local`** : on s’appuie sur `n_tiles_y` / `n_tiles_x` calculés plus haut.
* Les `break` sur `y0 >= max_y_local` ou `x0 >= max_x_local` sont là pour gérer les cas où la formule de `n_tiles_y` / `n_tiles_x` donne un peu trop de tuiles par rapport au canvas réel.
* On conserve les variables déjà en place :

  * `tile_size_px`, `step_px`
  * `offset_x`, `offset_y`
  * `global_shape_hw`
  * `MAX_TILES`
* On utilise toujours `_clone_tile_wcs` pour construire le WCS de la tuile.

À la fin de la construction, conserver / ajouter :

```python
_emit(
    f"[GRID] DEBUG: grid definition ready with {len(tiles)} tile(s) "
    f"(rejected={rejected_tiles}, est={n_tiles_estimated})",
    callback=progress_callback,
)

return GridDefinition(
    global_wcs=global_wcs,
    global_shape_hw=(int(global_shape_hw[0]), int(global_shape_hw[1])),
    offset_xy=(offset_x if global_bounds else 0, offset_y if global_bounds else 0),
    tile_size_px=tile_size_px,
    overlap_fraction=overlap_fraction,
    tiles=tiles,
)
```

### Tâche 2 – Garde-fou MAX_TILES cohérent

* S’assurer que :

  ```python
  MAX_TILES = 50000
  ```

  est défini dans la fonction (ou en haut du module) de façon claire.
* Gérer **deux niveaux** :

  1. Avertissement si `n_tiles_estimated > MAX_TILES` (simple warning, mais on continue) :

     ```python
     if n_tiles_estimated > MAX_TILES:
         _emit(
             f"[GRID] WARNING: estimated tiles ({n_tiles_estimated}) exceeds MAX_TILES={MAX_TILES}, "
             "proceeding but freeze risk exists",
             lvl="WARNING",
             callback=progress_callback,
         )
     ```
  2. Protection stricte en cours de construction (voir code ci-dessus) : si `len(tiles) > MAX_TILES`, on log en ERROR + on lève `RuntimeError`.

Ainsi, en cas de configuration totalement dégénérée ou bug futur, on aura au pire un **Grid mode abort propre** et non un freeze.

### Tâche 3 – Logs de debug robustes

* Vérifier que les logs suivants existent **et sont atteints** :

  * Après calcul des estimations :

    ```python
    _emit(
        f"[GRID] DEBUG: estimated tiles: {n_tiles_estimated} "
        f"({n_tiles_y} rows x {n_tiles_x} cols) for canvas {global_shape_hw}",
        callback=progress_callback,
    )
    ```

  * Après la construction de la grille :

    ```python
    _emit(
        f"[GRID] DEBUG: grid definition ready with {len(tiles)} tile(s) "
        f"(rejected={rejected_tiles}, est={n_tiles_estimated})",
        callback=progress_callback,
    )
    ```

* Ne pas changer le format des logs déjà existants, sauf ajout de `lvl="WARNING"` ou `lvl="ERROR"` quand c’est réellement pertinent.

---

## 4. Critères d’acceptation

La mission est réussie si :

1. Sur le dataset problématique de l’utilisateur (canvas `(3272, 2406)`, `tile_size_px=1920`, `step_px=1152`, `n_tiles_estimated=6`):

   * Le log montre :

     * Les lignes `DEBUG: estimated tiles: 6 (3 rows x 2 cols) ...`
     * La ligne `DEBUG: grid definition ready with X tile(s) (rejected=Y, est=6)` avec `X` raisonnable (typiquement entre 4 et 6).
   * **Aucun** spam de `DEBUG: built XXX tile(s) so far` au-delà de quelques centaines au maximum (en pratique ici, < 100).
   * Le worker **ne freeze plus** : `build_global_grid` retourne, et le pipeline continue vers `assign_frames_to_tiles` puis l’empilement des tuiles.

2. En cas de configuration extrême (par exemple si quelqu’un force un `grid_size_factor` minuscule), le code :

   * Émet un warning si `n_tiles_estimated` > `MAX_TILES`.
   * Lève un `RuntimeError` avec un message explicite si `len(tiles)` > `MAX_TILES`, au lieu de rester bloqué.

3. Aucune régression sur le comportement “normal” :

   * Les tuiles couvrent toujours le canvas global de manière logique.
   * Les offsets WCS (`offset_x`, `offset_y`) sont toujours pris en compte comme avant.
   * Le type de retour (`GridDefinition`) et ses champs ne changent pas (sauf éventuellement ajout de valeurs plus strictement typées).

---

## 5. Style & contraintes

* Python 3.10+, typage déjà présent : conserve/complète les hints quand c’est simple.
* Pas de refacto global : concentre-toi uniquement sur la construction de la grille dans `build_global_grid`.
* Garde le style des logs existants (`_emit(...)`) et la signature de la fonction.

Merci 😊

