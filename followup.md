### 🧾 followup.md

**Résumé de la mission précédente**

* On a corrigé, dans `grid_mode.py` (commit de base : `121db2f7`), la **construction de `common_mask`** dans la boucle de photométrie inter-tuiles.
* Avant :

  * `coverage_mask` (2D) était combiné directement avec `mask_ref` et `mask_tgt` (potentiellement 3D), ce qui causait un **broadcast error**.
* Maintenant :

  * si `coverage_mask` est valide et non vide :

    * on vérifie qu’il a la même géométrie `(H, W)` que les patches,
    * on le **diffuse en 3D** si nécessaire pour matcher la forme de `mask_ref` / `mask_tgt`,
    * on construit `common_mask` via un `AND` cohérent entre masques.
  * si `coverage_mask` est absent, vide, ou de forme incompatible :

    * on log un warning (pour trace),
    * on retombe sur le masque simple `mask_ref & mask_tgt`.

**À ne pas faire lors d’une mission ultérieure**

* Ne pas re-toucher à cette logique tant qu’on ne redéfinit pas explicitement un **nouveau concept de normalisation** en Grid.
* Ne pas modifier la signature ni le comportement de :

  * `_overlap_mask_from_coverage`,
  * `compute_valid_mask`,
  * `compute_tile_photometric_scaling`,
  * `apply_tile_photometric_scaling`.

