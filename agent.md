# 🎯 Mission Codex — Logger ultra-verbeux du Grid Mode (diagnostic H/W, bbox, shapes)

## Contexte

Le Grid mode ne plante plus, mais il produit parfois une mosaïque vide ou quasi vide alors que :
- des tuiles sont bien générées,
- les fichiers `tile_XXXX.fits` existent,
- mais l’assemblage conclut que les tuiles n’ont “pas de recouvrement exploitable”.

On soupçonne un **problème de cohérence géométrique** entre :
- la taille du canvas global (`global_shape_hw` → `H_m`, `W_m`),
- les bounding boxes des tuiles (`bbox`),
- la taille réelle des données de tuile (`info.data.shape`),
- les offsets / clamps appliqués dans l’assemblage.

Objectif de cette mission :  
👉 Ajouter des **logs détaillés et lisibles** pour voir noir sur blanc *où* les choses partent en sucette, sans encore modifier la logique métier.

---

## Fichiers concernés

- `grid_mode.py`
  - Fonction(s) d’assemblage des tuiles en mosaïque, par ex. quelque chose du style :
    - `assemble_tiles(...)`
    - ou une fonction équivalente appelée depuis `run_grid_mode`.

L’idée est de **limiter les changements au logging** dans la zone où la mosaïque finale est construite à partir des tuiles.

---

## Objectif de la mission

1. Ajouter des logs `[GRID-ASM]` **au moment de l’assemblage** qui affichent clairement :
   - la taille du canvas global (`H_m`, `W_m`, `global_shape_hw`),
   - l’offset global utilisé (s’il existe),
   - pour chaque tuile (au moins les premières, ou toutes si raisonnable) :
     - l’ID de la tuile,
     - sa bbox “globale” (avant offset/clamp),
     - sa bbox “locale” après offset,
     - la bbox finale après clamp dans le canvas,
     - la taille réelle des données de la tuile (`info.data.shape`),
     - les valeurs `used_w` / `used_h` utilisées pour peindre dans la mosaïque,
     - le motif exact de rejet ("bbox outside mosaic", "no overlap after clamping", etc.).
2. Ne **rien changer** (pour l’instant) à la logique d’assemblage elle-même : uniquement du logging.
3. Faciliter la corrélation avec les traces déjà existantes (`[GRID] Global grid ready`, `[GRID] global canvas shape_hw=...`, etc.).

---

## Détails d’implémentation

### 1. Localiser le cœur de l’assemblage

Dans `grid_mode.py` :

- Identifier la fonction qui construit la mosaïque finale à partir des tuiles. Elle ressemble typiquement à :

  ```python
  def assemble_tiles(...):
      mosaic_sum = np.zeros(...)
      mosaic_weight = np.zeros(...)
      H_m, W_m, _ = mosaic_sum.shape
      for tile_id, info in tiles_info.items():
          # lecture de info.data, info.mask, info.bbox, etc.
          ...
````

* C’est **à l’intérieur de cette boucle** sur les tuiles que nous voulons ajouter les logs `[GRID-ASM]`.

### 2. Logger la taille du canvas global une seule fois

Au début de l’assemblage (juste après l’allocation de la mosaïque) :

```python
H_m, W_m, C_m = mosaic_sum.shape
logger.info(
    "[GRID-ASM] mosaic canvas created: shape_hw=(%d, %d), channels=%d",
    H_m, W_m, C_m,
)
```

Si une structure type `grid.global_offset` ou similaire existe, la logger aussi :

```python
if hasattr(grid, "global_offset"):
    ox, oy = grid.global_offset
    logger.info("[GRID-ASM] global offset=(%d, %d)", ox, oy)
```

Sinon, ne rien inventer : se limiter au shape.

### 3. Logger, pour chaque tuile, le cycle complet bbox → clamp → used_w/h

Dans la boucle sur les tuiles, repérer les éléments suivants :

* La bbox “globale” de la tuile (avant clamp) : typiquement `tx0, tx1, ty0, ty1`.
* La taille des données : `info.data.shape` → `(h, w, c)` ou `(h, w)`.

Juste avant le clamp, loguer :

```python
logger.debug(
    "[GRID-ASM] tile %s: original bbox=(x:%d-%d, y:%d-%d), data_shape=%s",
    tile_id,
    tx0, tx1, ty0, ty1,
    getattr(info, "data", None).shape if getattr(info, "data", None) is not None else None,
)
```

Après le clamp / recomputation de `x0, x1, y0, y1` :

```python
logger.debug(
    "[GRID-ASM] tile %s: clamped bbox=(x:%d-%d, y:%d-%d) within canvas (W=%d, H=%d)",
    tile_id,
    x0, x1, y0, y1,
    W_m, H_m,
)
```

Après calcul de `off_x`, `off_y`, `used_w`, `used_h` :

```python
logger.debug(
    "[GRID-ASM] tile %s: off_x=%d, off_y=%d, used_w=%d, used_h=%d",
    tile_id,
    off_x, off_y, used_w, used_h,
)
```

### 4. Logger le motif exact de rejet

Partout où il existe un `continue` / “skip” pour la tuile, ajouter un log explicite `[GRID-ASM]`.

Exemples typiques :

* Si la bbox est complètement hors canvas :

  ```python
  if x1 <= x0 or y1 <= y0:
      logger.warning(
          "[GRID-ASM] tile %s: skipped because clamped bbox is empty (x0=%d, x1=%d, y0=%d, y1=%d) within canvas (W=%d, H=%d)",
          tile_id, x0, x1, y0, y1, W_m, H_m,
      )
      continue
  ```

* Si `used_w` / `used_h` <= 0 :

  ```python
  if used_w <= 0 or used_h <= 0:
      logger.warning(
          "[GRID-ASM] tile %s: skipped because used_w/used_h <= 0 (used_w=%d, used_h=%d, off_x=%d, off_y=%d)",
          tile_id, used_w, used_h, off_x, off_y,
      )
      continue
  ```

* Si la tuile est rejetée plus tard, par exemple pour masque vide ou autre condition, ajouter un warning similaire avec `[GRID-ASM]` et la raison.

### 5. Limiter le bruit si besoin

Si la boucle peut potentiellement traiter des centaines de tuiles, mais qu’on veut éviter un log trop verbeux, on peut :

* laisser tous les logs `.debug` (ils ne seront visibles qu’en niveau DEBUG),
* garder les `.warning` pour les rejets seulement.

Ne **pas** introduire de logique conditionnelle (type “si tile_id <= 10” pour limiter) sans demande explicite : pour l’instant, il vaut mieux avoir l’info complète en DEBUG.

### 6. Ne pas modifier la logique métier

Important :
👉 Ne modifier **aucun calcul**, **aucun clamp**, **aucune condition de rejet**, seulement ajouter des logs.

Cela permettra d’isoler précisément la cause géométrique lors du prochain run, sans introduire de nouveaux bugs fonctionnels en même temps.

---

## Tests / Validation

* Lancer un Grid mode sur le dataset problématique.
* Vérifier que les logs contiennent des lignes `[GRID-ASM]` :

  * la taille du canvas `shape_hw`,
  * les `bbox` originales / clampées,
  * la `data_shape`,
  * les `off_x`, `off_y`, `used_w`, `used_h`,
  * les raisons de rejet éventuel pour chaque tuile.
* Conserver ce log pour analyse (il servira de base pour une mission suivante de correction fine).

---

## Critères d’acceptation

* ✅ Les logs `[GRID-ASM]` permettent de reconstituer, pour chaque tuile, le chemin complet :

  * canvas global → bbox → clamp → offsets → used_w/h → décision (placer ou skip).
* ✅ Aucun changement de logique métier : Grid mode se comporte identiquement à avant, mais avec plus d’informations dans le log.
* ✅ Le dataset problématique produit désormais un log suffisamment verbeux pour comprendre *exactement* pourquoi la mosaïque reste vide ou minuscule.

Merci 🙏


