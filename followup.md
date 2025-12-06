# followup.md — Vérifications après patch Grid mode

## 1. Ce que tu dois vérifier dans le code

1. Ouvre `grid_mode.py` et va dans la fonction `build_global_grid`.
2. Confirme que :
   - La partie “construction de la grille” utilise désormais des **boucles `for j in range(n_tiles_y)` / `for i in range(n_tiles_x)`** au lieu d’une double boucle `while`.
   - `MAX_TILES` est bien défini et utilisé à la fois :
     - pour logger un warning si `n_tiles_estimated > MAX_TILES`,
     - et pour lever un `RuntimeError` si `len(tiles) > MAX_TILES` pendant la construction.
   - Le log final de la fonction est présent :
     ```python
     _emit(
         f"[GRID] DEBUG: grid definition ready with {len(tiles)} tile(s) "
         f"(rejected={rejected_tiles}, est={n_tiles_estimated})",
         callback=progress_callback,
     )
     ```

## 2. Ce que tu dois vérifier dans les logs en lançant un run

[ ] Relance ZeMosaic sur le **même dataset** qu’avant (celui qui freezait), toujours en Grid mode.

Dans `zemosaic_worker.log` :

1. [ ] Vérifie que tu vois la séquence :

   ```text
   [GRID] [GRID] DEBUG: estimated tiles: 6 (3 rows x 2 cols) for canvas (3272, 2406)
   [GRID] [GRID] DEBUG: entering tile grid construction: tile_size_px=1920, step_px=1152, n_frames=53
   ...
   [GRID] [GRID] DEBUG: grid definition ready with X tile(s) (rejected=Y, est=6)
````

* `X` doit être raisonnable (quelques tuiles, pas des centaines de milliers).
* Il est normal de voir quelques lignes `DEBUG: built XX tile(s) so far`, mais **pas** des centaines de milliers.

2. [ ] Vérifie que le worker **continue après** la ligne "grid definition ready…" :

   * Tu dois voir des logs correspondant à `assign_frames_to_tiles`,
   * Puis à l’empilement des tiles et à l’assemblage final.

3. [ ] Vérifie qu’il n’y a :

   * ni freeze,
   * ni `Traceback` lié à `RuntimeError("Grid tile generation aborted: too many tiles(...)")` sur ce dataset (tu ne devrais pas atteindre `MAX_TILES` dans ce cas).

## 3. Si quelque chose ne va pas

* Si tu vois encore un spam massif de :

  ```text
  [GRID] [GRID] DEBUG: built XXXXX tile(s) so far
  ```

  sans jamais voir :

  ```text
  [GRID] [GRID] DEBUG: grid definition ready with ...
  ```

  alors :

  * Copie dans le prochain message :

    * le bloc de code complet de la construction de la grille dans `build_global_grid` (boucles `for` + logs),
    * et les lignes de log depuis `DEBUG: estimated tiles...` jusqu’au dernier `DEBUG: built ... tile(s) so far`.

* Si au contraire tu obtiens une exception `Grid tile generation aborted: too many tiles (...)` :

  * Copie aussi :

    * le message complet de l’erreur,
    * les valeurs de `global_shape_hw`, `tile_size_px`, `step_px`, `n_tiles_y`, `n_tiles_x`, `n_tiles_estimated` (tu peux les récupérer dans le log ou ajouter un log temporaire).

Avec ces éléments, on pourra ajuster finement la formule de `n_tiles_y` / `n_tiles_x` ou le seuil `MAX_TILES` si nécessaire.
