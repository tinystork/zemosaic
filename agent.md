# 🎯 Mission Codex — Garde-fou WCS global dégénéré + Fallback robuste en Grid Mode

## Contexte

Avec les nouveaux logs `[GRID]` et `[GRID-ASM]`, on a compris que :

- En Grid mode, `find_optimal_celestial_wcs(...)` retourne parfois un WCS global avec :
  - `shape_hw=(2, 2)` → puis un canvas final `global_shape_hw=(3, 3)`.
- Le pipeline classique (Phase 4) pour le **même dataset** produit une mosaïque parfaitement raisonnable (~3200×2100 px).
- Grâce aux logs, on sait maintenant que :
  - les `global_bounds` et l’offset sont cohérents,
  - les bboxes de tuile sont bien alignées sur le canvas,
  - l’assemblage place la tuile correctement dans le canvas **3×3** (donc l’assemblage n’est plus en cause).

Conclusion :  
👉 Le **vrai problème** est désormais la **WCS globale dégénérée** (trop petite) retournée par `find_optimal_celestial_wcs` dans certains cas.

Nous voulons :

1. Détecter ces cas de WCS “dégénérés” (ex : `shape_hw` ridicule par rapport aux frames).
2. **Basculer automatiquement sur un fallback “safe”** basé sur les WCS des frames, même si ce n’est pas optimal, mais suffisamment grand/robuste pour construire une mosaïque utile.
3. Ne pas casser les cas où `find_optimal_celestial_wcs` fonctionne bien.

---

## Fichiers concernés

- `grid_mode.py`
  - Fonction principale : `build_global_grid(...)`
    - Là où `find_optimal_celestial_wcs` est appelé.
    - Là où le WCS global et `global_shape_hw` sont définis.
  - Ajout de deux helpers :
    - `_is_degenerate_global_wcs(...)`
    - `_build_fallback_global_wcs(...)`
  - Utilisation de ces helpers dans `build_global_grid`.

Éventuellement (selon l’implémentation actuelle) :

- Une fonction utilitaire existante pour calculer les footprints dans un WCS donné, par exemple :
  - `_compute_frame_footprint(...)`
  - ou équivalent (à réutiliser).

---

## Objectifs

1. **Ajouter un validateur de WCS global** : `_is_degenerate_global_wcs(frames, global_wcs, global_shape_hw)` qui retourne `True` si le WCS proposé est manifestement aberrant (trop petit) par rapport aux frames.
2. **Ajouter un fallback** : `_build_fallback_global_wcs(frames)` qui :
   - prend le WCS d’un frame valide comme base (ex : le premier),
   - projette les footprints de tous les frames dans ce WCS,
   - en déduit un canvas global raisonnable (`global_shape_hw`) et des `global_bounds`.
3. Dans `build_global_grid` :
   - après l’appel à `find_optimal_celestial_wcs`, utiliser `_is_degenerate_global_wcs` pour décider si on garde ce WCS ou si on active le fallback.
   - logger clairement quand le fallback est utilisé.
4. Ne pas toucher à la logique d’assemblage actuelle (offset, bboxes, etc.), qui fonctionne déjà correctement une fois qu’on a une WCS et un canvas plausibles.

---

## Détails d’implémentation

### 1. Ajouter `_is_degenerate_global_wcs(...)`

Dans `grid_mode.py`, ajouter une fonction :

```python
def _is_degenerate_global_wcs(
    frames: list["FrameInfo"],
    global_wcs: "WCS",
    global_shape_hw: tuple[int, int],
) -> bool:
    """
    Retourne True si le WCS global proposé est manifestement aberrant
    par rapport aux frames d'entrée.
    """
    H_m, W_m = global_shape_hw

    # 1) Taille minimale absolue (à ajuster si besoin)
    MIN_SIZE = 256
    if H_m < MIN_SIZE or W_m < MIN_SIZE:
        return True

    # 2) Comparaison avec la taille moyenne des frames
    valid_frames = [f for f in frames if getattr(f, "shape_hw", None)]
    if valid_frames:
        mean_h = int(np.mean([f.shape_hw[0] for f in valid_frames]))
        mean_w = int(np.mean([f.shape_hw[1] for f in valid_frames]))
        # Si le canvas est plus petit que ~50% d'un frame moyen, c'est suspect.
        if H_m < 0.5 * mean_h or W_m < 0.5 * mean_w:
            return True

    # 3) (Optionnel) On pourrait ajouter un test sur l'étendue réelle
    # des footprints dans ce WCS, mais le MIN_SIZE + comparaison moyenne
    # suffisent pour un premier garde-fou.
    return False
````

Contraintes :

* Utiliser `np.mean` si `numpy` est déjà importé dans ce module (sinon, l’importer en haut du fichier).
* Le type `FrameInfo` et `WCS` peuvent être importés ou typés en forward ref (`"WCS"`).

### 2. Ajouter `_build_fallback_global_wcs(frames)`

Ajouter une fonction qui :

1. Choisit un frame de base (par ex. le **premier frame valide** dans la liste).

   ```python
   def _pick_first_valid_frame(frames: list["FrameInfo"]) -> "FrameInfo":
       for f in frames:
           if getattr(f, "wcs", None) is not None and getattr(f, "shape_hw", None):
               return f
       raise RuntimeError("[GRID] fallback WCS: no valid frame with WCS/shape")
   ```

2. Copie son WCS :

   ```python
   base_frame = _pick_first_valid_frame(frames)
   base_wcs = copy.deepcopy(base_frame.wcs)
   ```

3. Pour chaque frame valide, calcule son footprint dans le repère de `base_wcs`.
   L’objectif est d’obtenir une liste de bounds `(x0, x1, y0, y1)` dans ce WCS.
   Si une fonction utilitaire existe déjà (ex : `_compute_frame_footprint(global_wcs, frame)`), la réutiliser.

   Pseudo-code :

   ```python
   bounds: list[tuple[float, float, float, float]] = []
   for frame in frames:
       try:
           x0, x1, y0, y1 = _compute_frame_footprint(base_wcs, frame)
           bounds.append((x0, x1, y0, y1))
       except Exception:
           logger.warning("[GRID] fallback WCS: failed to compute footprint for frame %s", getattr(frame, "id", "?"))
           continue
   ```

4. Si `bounds` est vide, lever une erreur claire :

   ```python
   if not bounds:
       raise RuntimeError("[GRID] fallback WCS: could not compute any footprint")
   ```

5. À partir de ces bounds, calculer :

   ```python
   min_x = math.floor(min(b[0] for b in bounds))
   max_x = math.ceil(max(b[1] for b in bounds))
   min_y = math.floor(min(b[2] for b in bounds))
   max_y = math.ceil(max(b[3] for b in bounds))

   width = int(max_x - min_x)
   height = int(max_y - min_y)
   global_shape_hw = (height, width)

   offset_x, offset_y = min_x, min_y
   ```

6. Enregistrer ces `bounds` comme `global_bounds` et, si besoin, l’offset dans une structure existante (par exemple attachée à l’objet grid).
   Le but est de rester cohérent avec l’offset/bboxes déjà utilisés ailleurs.

7. Appliquer `_strip_wcs_distortion(base_wcs)` si c’est le comportement standard :

   ```python
   fallback_wcs = _strip_wcs_distortion(base_wcs)
   return fallback_wcs, global_shape_hw, bounds
   ```

### 3. Intégrer le garde-fou & fallback dans `build_global_grid(...)`

Dans `build_global_grid`, là où on appelle actuellement :

```python
global_wcs, global_shape_hw = find_optimal_celestial_wcs(...)
```

adapter en :

```python
global_wcs, global_shape_hw = find_optimal_celestial_wcs(...)

if _is_degenerate_global_wcs(frames, global_wcs, global_shape_hw):
    logger.warning(
        "[GRID] Optimal global WCS looks degenerate (shape_hw=%s), falling back to safer WCS",
        global_shape_hw,
    )
    global_wcs, global_shape_hw, global_bounds = _build_fallback_global_wcs(frames)
    logger.info(
        "[GRID] Fallback global WCS: shape_hw=%s (bounds from %d frames)",
        global_shape_hw, len(frames),
    )
else:
    logger.info(
        "[GRID] Optimal global WCS accepted: shape_hw=%s",
        global_shape_hw,
    )
    # global_bounds sera calculé comme avant (footprints dans global_wcs)
```

Remarques :

* Il faut que `_build_fallback_global_wcs` renvoie aussi `global_bounds` (ou une structure équivalente) si le reste du code s’appuie dessus.
* Dans la branche “non dégénérée”, on garde le comportement actuel.

### 4. Logging

* Ajouter les logs `[GRID]` indiqués ci-dessus :

  * warning si WCS jugé dégénéré,
  * info sur le fallback (shape, nb de frames utilisés).
* Conserver les logs déjà en place sur :

  * `global_bounds count=...`,
  * `global canvas shape_hw=..., offset=...`.

---

## Tests & Validation

1. **Dataset problématique actuel (celui qui donne un WCS 2×2)**

   * Vérifier que les logs contiennent :

     * `[GRID] Optimal global WCS looks degenerate...`
     * `[GRID] Fallback global WCS: shape_hw=...`
   * Vérifier que :

     * le Grid mode ne s’arrête plus avec une mosaïque 3×3,
     * la mosaïque grid a une taille raisonnable et contient des données visibles.

2. **Dataset sain où `find_optimal_celestial_wcs` marche déjà bien**

   * Vérifier que :

     * le garde-fou **n’est pas déclenché** (logs `Optimal global WCS accepted`),
     * le comportement reste identique à avant (taille de mosaïque, visuel, etc.).

3. **Non-régression**

   * Grid mode désactivé → pipeline classique inchangé.
   * Grid mode sur des petits jeux de données (2–3 images) → ajuster éventuellement `MIN_SIZE` si besoin (on peut descendre de 256 à 128 si les tests montrent que c’est trop strict).

---

## Critères d’acceptation

* ✅ Les cas où `find_optimal_celestial_wcs` renvoie un WCS manifestement trop petit activent le fallback, et le Grid mode produit une mosaïque de taille raisonnable avec du signal.
* ✅ Les cas “normaux” où le WCS optimal est correct ne déclenchent pas le fallback et continuent de fonctionner comme avant.
* ✅ Aucun crash ou régression majeure dans les autres chemins (pipeline classique non touché).
* ✅ Les logs `[GRID]` permettent de vérifier facilement si le WCS optimal a été accepté ou si on est passé en fallback.

Merci 🙏

