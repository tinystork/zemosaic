# ✅ Suivi des tâches — Logging diagnostic Grid Mode (H/W, bbox, shapes)

## Instructions pour Codex

1. Lire `agent.md` entièrement.
2. Prendre la **première tâche non cochée** ci-dessous.
3. Après modification :
   - Mettre à jour ce fichier en cochant la tâche (`[x]`),
   - Ajouter, si utile, une courte note (fichiers / fonctions modifiés).
4. Répéter jusqu’à ce que toutes les tâches nécessaires au logging soient cochées.

---

## Tâches

### [ ] 1. Localiser la fonction d’assemblage des tuiles

- Identifier, dans `grid_mode.py`, la fonction qui :
  - crée la mosaïque finale (allocation de `mosaic_sum`, `mosaic_weight`, etc.),
  - boucle sur les tuiles / `tiles_info`,
  - applique clamp / offsets / copie des pixels dans la mosaïque.
- Noter son nom et les paramètres principaux (commentaire dans le code ou dans cette section).

---

### [ ] 2. Logger la taille du canvas global

- Juste après l’allocation de la mosaïque, ajouter un log :

  ```python
  H_m, W_m, C_m = mosaic_sum.shape
  logger.info(
      "[GRID-ASM] mosaic canvas created: shape_hw=(%d, %d), channels=%d",
      H_m, W_m, C_m,
  )
````

* Si un offset global est stocké dans un objet `grid` ou équivalent, le logger aussi :

  ```python
  logger.info("[GRID-ASM] global offset=(%d, %d)", offset_x, offset_y)
  ```

  (uniquement si cette info est disponible proprement).

---

### [ ] 3. Logger, pour chaque tuile, la bbox originale et la taille des données

* Dans la boucle qui traite chaque tuile :

  * Ajouter un log `.debug` de type :

    ```python
    logger.debug(
        "[GRID-ASM] tile %s: original bbox=(x:%d-%d, y:%d-%d), data_shape=%s",
        tile_id,
        tx0, tx1, ty0, ty1,
        data.shape if data is not None else None,
    )
    ```

* S’assurer que `tile_id`, `tx0`, `tx1`, `ty0`, `ty1` et `data` sont bien définis à cet endroit.

---

### [ ] 4. Logger la bbox clampée dans le canvas

* Après clamp de la bbox globale dans `[0, W_m] × [0, H_m]` (typiquement `x0, x1, y0, y1`) :

  * Ajouter un log `.debug` :

    ```python
    logger.debug(
        "[GRID-ASM] tile %s: clamped bbox=(x:%d-%d, y:%d-%d) within canvas (W=%d, H=%d)",
        tile_id,
        x0, x1, y0, y1,
        W_m, H_m,
    )
    ```

---

### [ ] 5. Logger offsets et `used_w` / `used_h`

* Après calcul de `off_x`, `off_y`, `used_w`, `used_h` :

  * Ajouter un log `.debug` :

    ```python
    logger.debug(
        "[GRID-ASM] tile %s: off_x=%d, off_y=%d, used_w=%d, used_h=%d",
        tile_id,
        off_x, off_y, used_w, used_h,
    )
    ```

---

### [ ] 6. Logger les motifs de rejet (`continue` / skip)

* Partout où le code **skip** une tuile (conditions `if ...: continue`), ajouter un log `.warning` explicite, par ex. :

  * Bbox vide après clamp :

    ```python
    logger.warning(
        "[GRID-ASM] tile %s: skipped because clamped bbox is empty (x0=%d, x1=%d, y0=%d, y1=%d) within canvas (W=%d, H=%d)",
        tile_id, x0, x1, y0, y1, W_m, H_m,
    )
    ```

  * `used_w` / `used_h` <= 0 :

    ```python
    logger.warning(
        "[GRID-ASM] tile %s: skipped because used_w/used_h <= 0 (used_w=%d, used_h=%d, off_x=%d, off_y=%d)",
        tile_id, used_w, used_h, off_x, off_y,
    )
    ```

  * Autres raisons (masque vide, problème de lecture, etc.) : ajouter aussi un warning `[GRID-ASM]` avec la raison.

---

### [ ] 7. Vérifier que la logique métier est inchangée

* Confirmer que les modifications apportées sont uniquement des ajouts de `logger.debug` / `logger.info` / `logger.warning`.
* Ne pas modifier :

  * les valeurs de `H_m`, `W_m`, `tx0/tx1/ty0/ty1`,
  * les formules de clamp,
  * les conditions `if` existantes, sauf pour y insérer des logs.
* Si une modification non triviale s’avère nécessaire, la noter clairement ici.

---

### [ ] 8. Test rapide sur le dataset problématique

* Lancer un Grid mode sur le dataset connu comme problématique.
* Vérifier que les logs contiennent :

  * le log `[GRID-ASM] mosaic canvas created...`,
  * un bloc de logs `[GRID-ASM]` pour chaque tuile (au moins la tuile 1),
  * des warnings `[GRID-ASM] tile X: skipped because ...` le cas échéant.
* Noter ici (en quelques mots) le résultat et la date du test.

---

## Notes / Journal

> Utiliser cette section pour consigner :
>
> * nom de la fonction d’assemblage,
> * éventuelles subtilités rencontrées,
> * résultats des tests sur dataset réel.
