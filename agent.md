### 🧾 agent.md

**Mission courte et ultra ciblée – Grid mode photométrie**

#### Contexte

* On travaille sur le code tel qu’il est au commit **`121db2f7`**.
* En **Grid mode**, les **tiles sont toutes correctement construites et écrites** sur disque.
* Le plantage arrive ensuite, pendant la **phase de photométrie inter-tuiles dans `assemble_tiles`**, au moment où on calcule le masque `common_mask`.
* Le bug vient du fait que :

  * `coverage_mask` est **2D** `(H, W)`
  * alors que `mask_ref` et `mask_tgt` sont souvent **3D** `(H, W, 3)` (RGB)
  * la ligne actuelle `common_mask = coverage_mask & mask_ref & mask_tgt` provoque donc une erreur de broadcasting.

Objectif :
**Corriger ce problème de masque** pour que la **mosaïque finale soit produite**, **sans changer le concept photométrique** ni toucher au reste du pipeline (stack GPU/CPU, logique de grid, etc.).

---

#### Fichiers à modifier

* `grid_mode.py` **uniquement**.

---

#### Zone de code concernée

Dans `grid_mode.py`, à l’intérieur de la fonction qui assemble les tuiles (section photométrie, après la sélection de la tuile de référence), on a actuellement ce bloc (approx) :

```python
coverage_mask = None
if cov_ref is not None and cov_tgt is not None:
    coverage_mask = _overlap_mask_from_coverage(cov_ref, cov_tgt)
if coverage_mask is None or not np.any(coverage_mask):
    _emit(
        f"[GRID] Coverage not available for tile {info.tile_id}, falling back to finite-pixel mask.",
        lvl="INFO",
        callback=progress_callback,
    )
    common_mask = mask_ref & mask_tgt
else:
    _emit(
        f"[GRID] Coverage overlap for tile... {reference_info.tile_id}: pixels={int(np.sum(coverage_mask))}",
        lvl="DEBUG",
        callback=progress_callback,
    )
    common_mask = coverage_mask & mask_ref & mask_tgt

n_common = int(np.sum(common_mask))
_emit(
    f"Photometry: tile {info.tile_id} over...lap with ref {reference_info.tile_id} common pixels={n_common}",
    lvl="DEBUG",
    callback=progress_callback,
)
```

C’est cette ligne-là qui pose problème :

```python
common_mask = coverage_mask & mask_ref & mask_tgt
```

---

#### Tâche à réaliser

1. [x] **Corriger la construction de `common_mask` dans le cas où `coverage_mask` est disponible**, pour :

   * éviter tout **problème de broadcasting** entre `(H, W)` et `(H, W, 3)`,
   * conserver la **logique actuelle** :

     * si pas de coverage utile → fallback sur masque fini (`mask_ref & mask_tgt`)
     * si coverage utile → utiliser `coverage_mask` pour restreindre la zone commune.

2. **Ne rien modifier d’autre** :

   * ne pas toucher à la logique de stack (GPU/CPU),
   * ne pas modifier `_overlap_mask_from_coverage`,
   * ne pas changer `compute_valid_mask`, `compute_tile_photometric_scaling` ou `apply_tile_photometric_scaling`,
   * ne pas introduire de nouvelle dépendance.

---

#### Détails d’implémentation souhaités

* `mask_ref` et `mask_tgt` sont des booléens de même forme que les patches utilisés pour la photométrie (`ref_patch`, `tgt_patch`), donc souvent `(H, W, 3)` pour des tuiles RGB.
* `coverage_mask` est renvoyé par `_overlap_mask_from_coverage(cov_ref, cov_tgt)` en **2D** `(H, W)`.

On veut :

* Si `coverage_mask` est **valide et non vide** :

  * S’assurer que sa forme est compatible avec celle des masques :

    * si `mask_ref` ou `mask_tgt` est 3D `(H, W, C)` alors **diffuser** `coverage_mask` en `(H, W, C)` via `[..., None]` + `np.broadcast_to`.
  * Construire `common_mask` avec cette version diffusée :

    ```python
    coverage_mask_3d = coverage_mask
    if coverage_mask_3d.ndim == 2 and mask_ref.ndim == 3:
        coverage_mask_3d = np.broadcast_to(coverage_mask_3d[..., None], mask_ref.shape)
    elif coverage_mask_3d.ndim == 2 and mask_tgt.ndim == 3:
        coverage_mask_3d = np.broadcast_to(coverage_mask_3d[..., None], mask_tgt.shape)

    common_mask = coverage_mask_3d & mask_ref & mask_tgt
    ```
  * Avant ça, il est acceptable de vérifier que les deux premières dimensions coïncident, sinon on log un warning et on retombe sur le fallback :

    ```python
    if coverage_mask.shape[:2] != mask_ref.shape[:2] or coverage_mask.shape[:2] != mask_tgt.shape[:2]:
        _emit(
            f"[GRID] Coverage shape mismatch for tile {info.tile_id}, using finite-pixel mask instead.",
            lvl="WARN",
            callback=progress_callback,
        )
        common_mask = mask_ref & mask_tgt
    else:
        # diffusion + AND comme ci-dessus
    ```

* Si `coverage_mask` est `None` ou **sans pixels True** :

  * conserver le code actuel :

    ```python
    common_mask = mask_ref & mask_tgt
    ```

* `common_mask` doit rester un **masque booléen** qui :

  * a des **premières dimensions `(H, W)` identiques** à `ref_patch` / `tgt_patch`,
  * est compatible avec `_channel_mask` utilisé dans `compute_tile_photometric_scaling`
    (donc soit 2D `(H, W)`, soit 3D `(H, W, C)` avec `C` le nombre de canaux).

---

#### Critères de validation

* Le code doit **compiler** et s’exécuter sans exception liée aux masques dans `assemble_tiles`.
* Un run Grid mode avec le dataset problématique :

  * affiche toujours les logs `[GRIDCOV]` et `Photometry: tile X overlap with ref Y common pixels=...`,
  * **ne plante plus** sur la ligne qui calcule `common_mask`,
  * **produit enfin la mosaïque finale** (fichier FITS de sortie) sans fallback silencieux sur la pipeline classique.
* Les statistiques de photométrie (medians, gains/offsets) continuent d’être loguées comme avant pour les tuiles où `n_common > 0`.
