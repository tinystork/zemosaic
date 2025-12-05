
# 🎯 Mission Codex — Stabiliser le WCS global & le canvas en Grid Mode

## Contexte

Le **Grid mode** échoue parfois avec des tuiles ayant des bounding boxes du type `(-1, 2, -1, 2)` et aboutit à un mosaic final vide ou presque vide (logs du style `bbox_extent=(-1:2,-1:2)` puis "no valid tile data written to mosaic").

Les symptômes apparaissent notamment quand `find_optimal_celestial_wcs` échoue et que le code bascule sur un **fallback de WCS global** basé sur le premier frame, sans recalcul correct du canvas à partir des footprints réels.

Le problème conceptuel :  
👉 les tuiles sont définies dans un repère pixel qui ne correspond pas à la vraie enveloppe des footprints, avec des coordonnées parfois négatives ou hors du canvas.

Nous voulons **stabiliser la construction du WCS global et du canvas** en Grid mode, en dérivant la taille & l’origine du canvas depuis les footprints, et en garantissant que **toutes les bounding boxes finale vivent dans un repère strictement positif** `[0, width) × [0, height)`.

---

## Fichiers concernés

- `grid_mode.py`
  - Fonction principale : `build_global_grid(...)`
  - Fonctions satellites potentielles : `_compute_frame_footprint`, structures de données qui stockent bboxes / footprints / tiles.
- Éventuellement (si nécessaire pour l’offset) :
  - L’endroit où les tuiles sont assemblées en mosaic (si dans `grid_mode.py` ou dans une fonction appelée par `zemosaic_worker`).

---

## Objectif global

1. **Toujours** dériver la taille du canvas global (`global_shape_hw`) et l’origine de notre repère pixel à partir des **footprints** (liste de bounding boxes `global_bounds`).
2. Introduire un **offset global `(offset_x, offset_y)`** dérivé de `min_x` / `min_y` pour que toutes les bboxes/tuiles soient remappées dans un repère positif.
3. **Ne pas bricoler `crpix` seul** :
   - Par défaut, ne PAS déplacer `crpix` si ce n’est pas nécessaire.
   - Si un déplacement de `crpix` s’avère indispensable, ajuster aussi `crval` de façon cohérente (sinon, on casse la correspondance ciel/pixel). Dans un premier temps, privilégier une solution **pixel-level (offset)** sans toucher au WCS.
4. Améliorer le **logging** autour du calcul du WCS global, du fallback, de `global_bounds` et des bboxes finales pour diagnostiquer les futurs cas.

---

## Détails de l’implémentation

### 1. Calcul robuste du canvas global

Dans `build_global_grid(...)`, après avoir calculé les footprints des frames dans le repère du WCS global (réel ou fallback), il existe déjà (ou doit exister) une structure `global_bounds`, typiquement une liste de quadruplets `(x0, x1, y0, y1)`.

**Étapes à ajouter / renforcer :**

1. **Si `global_bounds` n’est pas vide** :

   ```python
   min_x = math.floor(min(b[0] for b in global_bounds))
   max_x = math.ceil(max(b[1] for b in global_bounds))
   min_y = math.floor(min(b[2] for b in global_bounds))
   max_y = math.ceil(max(b[3] for b in global_bounds))
````

2. Définir un **offset global** :

   ```python
   offset_x = min_x
   offset_y = min_y
   width = int(math.ceil(max_x - min_x))
   height = int(math.ceil(max_y - min_y))
   global_shape_hw = (height, width)
   ```

3. **Ne pas modifier `global_wcs.wcs.crpix` par défaut.**
   On considère que le WCS global décrit une géométrie valide, et on **relocalise les bboxes** dans un repère `[0, width) × [0, height)` grâce à l’offset.
   → C’est la solution la plus simple et la moins risquée pour la cohérence astrométrique.

4. Pour chaque bbox/footprint utilisé ensuite (par exemple lors de la construction des tuiles), appliquer :

   ```python
   # Anciennes coordonnées globales
   x0, x1, y0, y1 = original_bbox  # dans le repère global initial

   # Nouvelles coordonnées relatives au canvas
   local_x0 = x0 - offset_x
   local_x1 = x1 - offset_x
   local_y0 = y0 - offset_y
   local_y1 = y1 - offset_y

   # Stocker/utiliser (local_x0, local_x1, local_y0, local_y1) pour le placement dans le mosaic
   ```

   **But :** toutes les bboxes utilisées pour l’assemblage sont désormais **positives** et limitées à la taille du canvas.

5. Si `global_bounds` est vide (cas pathologique) :

   * Garder le fallback actuel **mais** :

     * loguer clairement la situation,
     * éventuellement ne pas lancer Grid mode du tout et basculer sur le pipeline classique.

6. Après ça, appeler comme aujourd’hui `_strip_wcs_distortion(global_wcs)` si c’est déjà le comportement standard, mais **sans toucher à `crpix`/`crval`** à ce stade.

### 2. Gestion prudente de `crpix` / `crval` (bémol important)

* Dans cette mission, **ne pas implémenter** de recentrage agressif du WCS du style :

  ```python
  center_x = (min_x + max_x) / 2.0
  center_y = (min_y + max_y) / 2.0
  global_wcs.wcs.crpix = [center_x, center_y]
  ```

  sans ajuster `crval`.

* Si tu identifies un endroit du code où un tel changement existe déjà ou a été tenté, il faut :

  * soit **le supprimer** au profit de la logique d’offset,
  * soit **le corriger proprement** en recalculant `crval` pour conserver la même géométrie.
    Dans le doute, **préférer supprimer/ignorer** ce recentrage pour cette mission, et documenter en commentaire qu’on a choisi une approche par offset.

### 3. Fallback quand `find_optimal_celestial_wcs` échoue

* Quand `find_optimal_celestial_wcs` échoue (retour `None`, exception, etc.), la logique actuelle semble :

  * prendre le WCS du premier frame,
  * et `global_shape_hw` ~ `shape_hw` du premier frame.

* Modifier cette partie pour :

  1. **Toujours** calculer les footprints de tous les frames dans ce WCS fallback.
  2. Construire `global_bounds` avec ces footprints.
  3. Appliquer exactement la **même logique d’offset et de recalcul du canvas** que décrite plus haut : `offset_x/offset_y`, `global_shape_hw = (height, width)`.
  4. Si **aucun footprint valide** n’est trouvé, loguer clairement et abandonner proprement le Grid mode.

### 4. Validation WCS + filtres de sécurité

Dans `_load_frame_wcs` / `_compute_frame_footprint` :

* Rajouter des contrôles simples :

  * WCS incomplet / incohérent → frame ignoré.
  * Footprint vide / NaN majoritaire → frame ignoré.
* Loguer clairement les frames rejetés et pourquoi, avec un tag `"[GRID]"`.

### 5. Logging à améliorer

Ajouter des logs explicites (avec tag `[GRID]`) à des points clefs :

1. **Après tentative de `find_optimal_celestial_wcs`** :

   * Succès :

     * `"[GRID] Optimal global WCS found: crval=(...), crpix=(...), shape_hw=(h, w)"`
   * Échec + fallback :

     * `"[GRID] Optimal global WCS failed, falling back to first-frame WCS: frame=<id>, initial shape_hw=(h, w)"`

2. **Après calcul de `global_bounds` et du canvas** :

   ```text
   [GRID] global_bounds count=N, min_x=..., max_x=..., min_y=..., max_y=...
   [GRID] global canvas shape_hw=(height, width), offset=(offset_x, offset_y)
   ```

3. **Avant/pendant l’assemblage des tuiles** :

   * Nombre de tuiles valides, nombre de tuiles rejetées pour cause de bbox hors canvas, etc.
   * Exemples de bboxes après application de l’offset pour vérifier qu’on n’a plus de coordonnées négatives.

---

## Plan de travail (ordre recommandé)

1. **Lire** la logique existante dans `build_global_grid` et identifier :

   * où `global_bounds` est calculé,
   * comment `global_shape_hw` est actuellement dérivé,
   * où `crpix` est potentiellement modifié.
2. **Introduire l’offset (min_x, min_y)** et recalculer `global_shape_hw` à partir de `global_bounds`.
3. **Propager l’offset** à toutes les bboxes utilisées pour les tuiles / frames dans le canvas.
4. **Nettoyer / désactiver** tout recentrage WCS qui modifie `crpix` seule sans ajuster `crval`.
5. **Renforcer le fallback** quand `find_optimal_celestial_wcs` échoue :

   * calcul des footprints,
   * global_bounds,
   * offset + canvas.
6. **Ajouter les logs `[GRID]`** détaillés décrits ci-dessus.
7. **Tests & validation** (voir section suivante).
8. Mettre à jour `followup.md` pour cocher la tâche une fois validée.

---

## Tests & validation

### 1. Tests synthétiques (si possible dans le code / un petit script)

* Construire un petit set de WCS/frames (syntétiques ou réels) avec :

  * 2–3 frames décalés,
  * un `find_optimal_celestial_wcs` forcé à échouer (mock / paramètre).
* Vérifier :

  * que `global_bounds` contient des valeurs cohérentes,
  * que `offset_x` / `offset_y` sont bien appliqués,
  * que toutes les bboxes finales sont dans `[0, width) × [0, height)`.

### 2. Test réel Grid mode (dataset problématique)

* Lancer le Grid mode sur le dataset qui produisait les logs `bbox_extent=(-1:2,-1:2)`.
* Vérifier dans les logs :

  * la présence des nouveaux messages `[GRID]` sur global_bounds, canvas, offset.
  * l’absence de bboxes négatives,
  * l’absence de message "no valid tile data written to mosaic".
* Vérifier que la mosaïque produite contient bien des données visibles (pas une image vide).

### 3. Tests de non-régression

* Vérifier que :

  * le pipeline classique (hors Grid mode) reste inchangé.
  * le Grid mode se comporte comme avant lorsque `find_optimal_celestial_wcs` **réussit** et que les footprints étaient déjà dans un repère propre (offset 0 ou négligeable).
  * les performances restent comparables.

---

## Critères d’acceptation

* ✅ Plus de bboxes du type `(-1:2,-1:2)` : toutes les bboxes utilisées pour le placement dans le canvas sont positives et dans les bornes du canvas.
* ✅ Le Grid mode ne produit plus de mosaïque "vide" dans les cas où les données sont valides.
* ✅ Le fallback en cas d’échec de `find_optimal_celestial_wcs` utilise quand même les footprints pour dimensionner le canvas et définir l’offset.
* ✅ Le WCS global n’est plus modifié "à la hache" via un changement de `crpix` seul ; soit on ne le déplace pas, soit on documente et corrige proprement `crval` (dans cette mission, privilégier l’offset sans toucher au WCS).
* ✅ Les logs `[GRID]` permettent de diagnostiquer clairement :

  * le WCS global choisi (optimal ou fallback),
  * les bounds et le canvas,
  * le nombre de tuiles valides / rejetées.

Merci 🙏

