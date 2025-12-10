## `agent.md` – Grid Mode Geometry & Photometry Finalization

### 1. Mission

Objectif : **fiabiliser définitivement le Grid Mode** pour qu’il produise :

- des tuiles **géométriquement cohérentes** entre elles et avec le flux classique,
- une mosaïque finale **sans damier photométrique** (fond homogène, pas de bandes),
- **zéro fallback silencieux** vers la pipeline classique quand tout est OK.

On conserve absolument :

- le support GPU (CuPy) et le découpage en chunks,
- le multi-thread/multi-process existant,
- le comportement de la pipeline classique hors Grid.

---

### 2. Périmètre / fichiers à modifier

Prioritaire (autorisé) :

- `grid_mode.py`
- `zemosaic_worker.py`

À ne modifier **que si nécessaire pour corriger un bug manifeste** :

- `zemosaic_stack_core.py` (interfaces de `stack_core`, `compute_tile_photometric_scaling`, `apply_tile_photometric_scaling`)
- `zemosaic_utils.py` (uniquement pour des helpers WCS / FITS si besoin)

**Interdit dans cette mission** :

- tout fichier GUI (`zemosaic_gui*.py`, `zemosaic_filter_gui*.py`, etc.),
- les scripts d’analyse, ZeAnalyser, ZeSolver, ZeQualityMT, lecropper,
- le flux classique d’align/stack hors Grid (ne pas changer sa logique).

---

### 3. Contexte technique (état actuel)

- Le Grid Mode :
  - lit `stack_plan.csv`,
  - construit des `FrameInfo` et un `GridDefinition`,
  - stacke les frames par tuile (CPU ou GPU, via `stack_core`),
  - appelle `assemble_tiles(...)` pour fabriquer la mosaïque finale,
  - est déclenché depuis `zemosaic_worker.run_hierarchical_mosaic_process` quand `stack_plan.csv` est détecté.

- Malgré ça :
  - les **tuiles sont mal alignées** entre elles par rapport au flux classique (décalages, “marches d’escalier”, zones vides),
  - des **bandes photométriques** subsistent entre tuiles,
  - dans certains cas le worker **repasse en flux classique** (fallback) et aucun fichier Grid n’est sauvegardé.

---

### 4. Buts détaillés

1. **Géométrie/WCS**

   - S’assurer que **toutes les tuiles** sont construites dans un **référentiel WCS global unique**.
   - Le canevas global (largeur/hauteur) doit être **verrouillé une seule fois** et transmis à toutes les étapes.
   - Les `bbox` des tuiles doivent couvrir la même région du ciel que le flux classique, sans décalage ni trous.

2. **Photométrie inter-tile**

   - Utiliser réellement `compute_tile_photometric_scaling` + `apply_tile_photometric_scaling` pour harmoniser le flux entre tuiles.
   - Les bandes verticales/horizontales doivent disparaître sur la mosaïque finale.

3. **Assemblage final & crop**

   - Assembler les tuiles dans la mosaïque **sans reproject global supplémentaire inutile**, en utilisant les `bbox` correctement calculées.
   - Appliquer éventuellement un **autocrop** basé sur la coverage **une seule fois** et mettre CRPIX / NAXIS à jour de manière cohérente (pas de double recadrage).

4. **Fallback maîtrisé**

   - Le Grid Mode ne doit **plus tomber en fallback silencieux** quand les entrées sont valides.
   - En cas d’échec réel, le log doit contenir une raison claire, et le fallback doit être **explcite** (message `[GRID]` en ERROR/WARN).

---

### 5. Invariants à respecter

- Le résultat Grid et le résultat “flux classique” sur le même dataset doivent :
  - couvrir **la même région du ciel**,
  - avoir des étoiles superposées (pas de shift systématique),
  - avoir un fond de ciel compatible à quelques pourcents près.

- Si Grid est désactivé (pas de `stack_plan.csv`), le comportement actuel du worker **ne change pas**.

---

### 6. Plan d’implémentation (étapes pour Codex)

#### 6.1. Fixer le référentiel WCS global et le canevas

1. **Identifier** dans `grid_mode.py` l’endroit où la WCS globale est décidée (GridDefinition, global coverage, etc.).
2. Si ce n’est pas déjà le cas :
   - construire une WCS globale (par ex. `find_optimal_celestial_wcs` ou équivalent),
   - en déduire sa `shape_hw` (hauteur, largeur),
   - stocker explicitement dans une structure (ex. `GridDefinition.global_wcs`, `GridDefinition.global_shape_hw`).
3. S’assurer que :
   - **toute opération** qui a besoin de la taille de la mosaïque (coverage globale, alpha, etc.) utilise **exactement** `global_shape_hw`,
   - aucun code ne recalcule une autre shape à partir des tuiles individuelles.

#### 6.2. Construction des tuiles → géométrie cohérente

1. Dans la fonction qui crée les tuiles (grid, `GridTile` et leurs `bbox`) :

   - pour chaque tuile, calculer la `bbox = (xmin, xmax, ymin, ymax)` **en pixels du canevas global**, en s’appuyant sur le WCS global.
   - `xmin/xmax/ymin/ymax` doivent être des entiers obtenus via `floor/ceil` sur les positions des coins en WCS.
   - La largeur/hauteur de la tuile (`tile_shape_hw`) doit être cohérente avec `bbox`.

2. Vérifier dans le code que :

   - **aucune autre logique** ne recalcule les `bbox` à partir des indices de tuiles (row/col) en ignorant la WCS.
   - Si une ancienne logique “grille régulière” existe, la **remplacer** par l’utilisation des footprints WCS des frames ou tuiles.

3. Ajouter du logging `[GRID]` (en DEBUG) pour chaque tuile :

   - `tile_id`, `bbox`, `tile_shape_hw`, extrait WCS (CRVAL/CRPIX) pour debug.

#### 6.3. Stacking par tuile (CPU/GPU) – respecter l’ordre

Pour chaque tuile :

1. Empiler les frames avec `stack_core` en respectant les paramètres :

   ```python
   stack_core(
       tile_frames_data,
       backend="gpu" ou "cpu",
       norm=stack_norm_method,
       weight=stack_weight_method,
       reject=stack_reject_algo,
       winsor_limits=winsor_limits,
       combine=stack_final_combine,
       chunk_budget_mb=stack_chunk_budget_mb,
       radial_weight=apply_radial_weight,
       feather_fraction=radial_feather_fraction,
       feather_power=radial_shape_power,
       ...
   )
````

2. Si l’image est RGB :

   * appeler `equalize_rgb_medians_inplace(tile_data)` **avant** d’estimer les stats de fond / scaling photométrique.

3. Construire une `tile_mask` :

   * `compute_valid_mask(tile_data)` (isfinite & > eps),
   * éventuellement combiner avec une coverage interne si elle existe déjà.

#### 6.4. Normalisation photométrique inter-tile

1. Choisir une tuile de référence “saine” (par ex. la première avec coverage suffisante).

2. Pour chaque tuile `info` ≠ ref :

   * calculer l’intersection des zones valides : `common_mask = ref.mask & info.mask`.
   * si la fraction de pixels valides est trop faible, sauter la normalisation pour cette tuile (log WARN `[GRID] Photometry: insufficient overlap…`).

3. Sinon :

   ```python
   gains, offsets = compute_tile_photometric_scaling(ref_patch, tgt_patch, mask=common_mask)
   info.data = apply_tile_photometric_scaling(info.data, gains, offsets)
   info.mask = compute_valid_mask(info.data) & info.mask
   ```

4. Logguer les médianes avant/après par canal pour debug.

#### 6.5. Assemblage des tuiles en mosaïque globale

1. Dans `assemble_tiles(...)` :

   * créer un tableau `mosaic_data` de shape `global_shape_hw` + channels,
   * un tableau `mosaic_weight` / `coverage` de même shape,
   * initialiser à 0 ou NaN selon la logique actuelle.

2. Pour chaque tuile :

   * `tile_bbox = (xmin, xmax, ymin, ymax)` (coordonnées globales),

   * vérifier que :

     * `0 <= xmin < xmax <= global_width`,
     * `0 <= ymin < ymax <= global_height`.
     * sinon → log ERROR et skip tuile.

   * extraire `tile_data` et `tile_mask`,

   * convertir le masque en poids (par ex. 1.0 pour valide, 0.0 sinon),

   * ajouter dans la mosaïque :

     ```python
     mosaic_data[y0:y1, x0:x1, ...] += tile_data * weight
     mosaic_weight[y0:y1, x0:x1]    += weight
     ```

3. En fin d’assemblage :

   * `final_mosaic = mosaic_data / mosaic_weight` là où le poids > 0,
   * NaN ailleurs,
   * construire aussi une carte `coverage` (ou réutiliser `mosaic_weight`).

4. **Aucune reproject globale supplémentaire** ne doit être faite ici : les tuiles sont déjà dans le même référentiel.

#### 6.6. Autocrop et mise à jour CRPIX/NAXIS

Coordination avec `zemosaic_worker.py` :

1. Laisser `_auto_crop_global_mosaic_if_requested` et `_apply_autocrop_to_global_plan` faire le job global, **une seule fois**.

2. Vérifier qu’aucun code dans `grid_mode.py` ne modifie encore :

   * `CRPIX1/2`,
   * `NAXIS1/2`,
   * la shape globale.

3. S’assurer que le plan retourné au worker contient :

   * `plan["width"]`, `plan["height"]` à jour,
   * un WCS cohérent (avant/après crop).

#### 6.7. Intégration avec `zemosaic_worker.py` & fallback

1. Le code Grid dans le worker doit :

   * appeler `grid_mode.run_grid_mode(...)`,
   * récupérer un **chemin de fichier mosaïque** et les méta-données WCS (si exposées),
   * considérer l’exécution Grid comme **acquise** si aucun exception n’est levée.

2. Le fallback vers le flux classique ne doit se produire que si :

   * `run_grid_mode` lève une exception,
   * ou retourne explicitement un indicateur d’échec (None, etc. selon la convention choisie).

3. Ajouter des logs explicites :

   * `[GRID] Fallback to classic pipeline: reason=…` en WARN/ERROR.

---

### 7. Critères d’acceptation

Le travail est terminé lorsque :

* [ ] Les tuiles Grid n’ont plus de décalage géométrique visible vs le flux classique.
* [ ] La mosaïque Grid ne présente plus de damier/bandes photométriques grossières.
* [ ] La coverage de la mosaïque Grid correspond à celle du flux classique (même champ).
* [ ] Aucun fallback Grid → classique n’a lieu sur un dataset valide.
* [ ] GPU + multithread fonctionnent toujours (tests CPU et GPU OK).
* [ ] Aucun changement de comportement n’est observé en mode non-Grid.

