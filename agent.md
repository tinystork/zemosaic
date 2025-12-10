
## `agent.md` – Grid Mode Photometric Fix & Perf Pass 

### 1. Mission

**Objectif principal :**
Rendre le **Grid Mode** photométriquement propre (plus de damier, plus de voile verdâtre) en **réutilisant la logique de normalisation déjà existante dans la pipeline classique**, sans casser :

* le support GPU (CuPy),
* le multi-thread / multi-process,
* ni le comportement de la pipeline classique.

En parallèle, faire un **1er passage d’optimisation** sur les points les plus coûteux (reproject redondants, dtypes, SciPy inutilement invoqué en boucle).

---

### 2. Contexte (à lire AVANT de coder)

Actuellement :

* La **pipeline classique** produit une mosaïque propre :

  * normalisation inter-frame,
  * renorm de couverture / deux-pass,
  * égalisation RGB (`equalize_rgb_medians_inplace`),
  * assembly global cohérent.
* Le **Grid Mode** produit une mosaïque :

  * avec tuiles à des niveaux de flux très différents,
  * avec un fond hétérogène (beige/vert),
  * avec des bandes verticales / horizontales de couleur,
  * avec un feathering inefficace.

Le diagnostic est :

1. **La photométrie n’est pas appliquée correctement dans Grid Mode :**

   * pas de normalisation inter-tile robuste,
   * pas (ou peu) d’égalisation RGB,
   * ordre d’opérations sous-optimal (reprojection avant renorm, etc.).
2. **Grid Mode est anormalement lent**, notamment parce que :

   * il y a trop d’appels à `reproject_interp` (et parfois par canal),
   * SciPy / ndimage est utilisé en CPU dans des boucles lourdes,
   * les dtypes sont parfois en float64,
   * le feathering est fait tuile par tuile, en CPU pur.

La mission **n’est PAS** de tout réécrire, mais de :

* **réutiliser au maximum les briques existantes** de la pipeline classique,
* corriger l’ordre des étapes dans Grid Mode,
* limiter les recalculs et dtypes coûteux.

---

### 3. Fichiers concernés

Travail **principal** dans :

* `grid_mode.py`

Lecture / réutilisation de logique dans :

* `zemosaic_worker.py` (Phase 5, post-stack pipeline, deux-pass, renorm)
* `zemosaic_stack_core.py` (backend stacker CPU/GPU, logique commune)
* `zemosaic_utils.py` (helpers utilisés partout, notamment WCS / coverage / temp dir)
* `zemosaic_align_stack.py` (rejet + `equalize_rgb_medians_inplace`) – déjà importé dans `grid_mode.py`.

**Important :**

* Ne pas casser les call-sites actuels de `zemosaic_worker.py`.
* Ne pas modifier les interfaces publiques de `stack_core` (sauf ajout de paramètres optionnels rétro-compatibles).

---

### 4. Comportement cible (fonctionnel)

1. **Photométrie Grid Mode alignée avec la pipeline classique :**

   * Chaque **master-tile Grid** doit être photométriquement comparable à une **master-tile classique** :

     * même ordre de normalisation (gains / offsets),
     * même traitement des outliers (Winsor / kappa-sigma déjà gérés par `stack_core`),
     * possibilité de renorm globale (deux-pass / coverage renorm) appliquée à la mosaïque finale.

   * À la fin :

     * **plus de damier**,

**Renormalisation globale (deux-pass)**

Si la configuration utilisateur active la renormalisation deux-pass/coverage,
Grid Mode doit, lorsque c’est techniquement possible, réutiliser la logique existante
(`_apply_two_pass_coverage_renorm_if_requested`) après l’assemblage global.

Si l’intégration n’est pas triviale dans ce ticket, considérer cette étape comme optionnelle
tout en s’assurant que la Phase 5 classique reste inchangée.
     * fond de ciel globalement homogène,
     * couleurs cohérentes,
     * feathering utile (au lieu d’amplifier les défauts).

2. **Égalisation RGB systématique pour les tuiles couleur :**

   * Lorsque les tuiles sont RGB, **appliquer `equalize_rgb_medians_inplace`** (déjà importée depuis `zemosaic_align_stack`) **à un moment cohérent avec la pipeline classique** :

     * Dans la pipeline classique, `equalize_rgb_medians_inplace` est appliquée par substack après stacking (post-stack), pas par frame.
     * Pour Grid Mode, appliquer la même logique : exécuter `equalize_rgb_medians_inplace` une fois par tuile, juste après le stacking local de la tuile et **avant** la normalisation inter-tuiles, plutôt que sur chaque frame.

   * Objectif : supprimer les dominantes canal-par-canal qui génèrent les bandes vert/magenta.

3. **Ordre des opérations corrigé :**

   Éviter le pipeline actuel :

   ```text
   load → debayer → stack local tile → reprojection → feather → assemble
````

En faveur d’un ordre plus sain :

```text
load → debayer → normalisation per-frame / per-tile (si besoin)
     → stack local tile
     → égalisation RGB (si C == 3)
     → normalisation inter-tile (gain/offset global) avec masque basé sur la coverage WCS
     → reprojection sur canevas global
     → feather + assembly
```

4. **Perfs acceptables sur gros datasets :**

   * Pas de changement d’API côté GUI / utilisateur dans ce ticket.
   * Réduire les coûts évidents **sans** toucher à la sémantique :

     * utiliser float32 par défaut,
     * limiter les reprojects redondants,
     * éviter les filtres SciPy inutiles,
     * réutiliser les données et WCS déjà calculés.

---

### 5. Plan de travail concret

#### Étape 1 – Localiser et comprendre la photométrie classique

1. Dans `zemosaic_worker.py` :

   * Repérer les fonctions responsables de la **Phase 5** :

     * `_apply_phase5_post_stack_pipeline(...)`,
     * `_apply_two_pass_coverage_renorm_if_requested(...)`,
     * ainsi que l’appel à `run_second_pass_coverage_renorm(...)` si accessible.

   * Comprendre :

     * comment les gains / renorms sont calculés,
     * à quel moment ils sont appliqués (avant / après reprojection),
     * comment la coverage est utilisée pour stabiliser la photométrie.

2. **Ne pas modifier ces fonctions dans ce ticket**, mais s’en servir comme **référence** pour définir une version “light” adaptée au Grid Mode.

---

#### Étape 2 – Ajouter un helper de normalisation pour les tuiles

Objectif : créer une petite API réutilisable qui permette à Grid Mode de faire une **normalisation inter-tile** à partir de scalaires simples (gain + offset par tuile et par canal).

**Note :** Dans ce ticket, ces helpers (`compute_tile_photometric_scaling` / `apply_tile_photometric_scaling`) ne doivent être utilisés que par `grid_mode.py`. Ne pas les brancher dans la pipeline classique tant que leur comportement n’a pas été validé sur Grid Mode.

1. Créer un helper dans un module déjà central (au choix) :

   * soit dans `zemosaic_stack_core.py`,
   * soit dans `zemosaic_utils.py` si c’est plus naturel.

2. Signature proposée (ajustable tant que c’est clair) :

```python
def compute_tile_photometric_scaling(
    reference_tile: np.ndarray,
    target_tile: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule (gain, offset) par canal pour amener target_tile vers reference_tile
    dans les zones définies par mask (ou sur toute l'image si mask est None).

    Retourne:
      gains:   shape (C,) ou (1,)
      offsets: shape (C,) ou (1,)
    """
```

3. Implémentation (avec masque basé sur la coverage / WCS) :

   * Travailler en **float32**.

   * Ramener les tuiles au format HWC ; si mono-canal, utiliser une forme `(H, W, 1)`.

   * Construire un **masque de fond basé sur la coverage WCS**, plutôt qu’un simple masque percentile global :

     * Le Grid Mode dispose déjà de cartes de coverage / poids / alpha (ex. nombre de contributions par pixel, ou cartes utilisées pour l’assemblage et le feathering).
     * L’objectif est d’utiliser ces informations pour ne comparer que les zones où **référence et cible ont une coverage suffisante et stable**, et ignorer :

       * les bords à coverage faible,
       * les zones extrapolées / NaN,
       * les zones empilées avec très peu de frames.

   * Stratégie recommandée :

     1. Pour chaque tuile (référence et target), construire une **carte de coverage binaire** à partir des données disponibles :

        * pixels `True` là où les données sont valides (non-NaN),
        * si une carte de poids/coverage explicite existe déjà, l’utiliser (ex. coverage >= 1 ou >= seuil).
     2. Ramener ces cartes sur une même grille logique :

        * soit en se basant sur la grille de la tuile (masque en coordonnées de la tuile),
        * soit, si disponible, en réutilisant la même logique WCS que celle employée pour projeter les tiles sur le canevas global.
     3. Définir `mask` comme l’**intersection** :

        * pixels valides dans `reference_tile`,
        * pixels valides dans `target_tile`,
        * coverage au-dessus d’un seuil minimal dans les deux.
     4. Éventuellement, **érosion légère** du masque (1–2 pixels) pour retirer les bords de tuile les plus bruités / mal couverts.

   * Une fois `mask` défini :

     * pour chaque canal :

       * construire un masque de pixels valides : `valid = np.isfinite(ref_channel) & np.isfinite(tgt_channel) & mask`,
       * si **aucun** pixel valide pour ce canal dans `reference_tile` **ou** `target_tile` :

         * ne pas calculer de scaling réel,
         * fixer `gain = 1.0`, `offset = 0.0` pour ce canal,
         * logguer un warning `DEBUG`/`INFO` (par ex. : “no valid pixels in overlap, scaling disabled for tile X, channel Y”),
       * sinon :

         * calculer un **fond de ciel** robuste (par ex. médiane sur les pixels `valid`) pour ref et target,
         * calculer un **gain** basé sur les médianes (ou une petite régression robuste),
         * optionnel : borner le gain dans un intervalle raisonnable (par ex. `[0.5, 2.0]`).

   * Retourner des gains / offsets scalaires (pas besoin de carte 2D à ce stade).

   * Garantir que les tableaux `gains` et `offsets` ne contiennent ni `NaN` ni `inf` (remplacer par 1/0 en fallback si nécessaire).

4. Créer un second helper :

```python
def apply_tile_photometric_scaling(
    tile: np.ndarray,
    gains: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """
    Applique gain/offset par canal, retourne un nouveau tile float32.
    Ne modifie pas le tableau d'entrée in-place.
    """
```

5. Ajouter des logs DEBUG optionnels :

   * avant scaling : min/max/median par canal,
   * gains/offsets calculés,
   * après scaling : min/max/median par canal.

   Ces logs servent à vérifier que le scaling est **réellement appliqué** et reste sain.

---

#### Étape 3 – Brancher ce helper dans `grid_mode.py`

1. Localiser dans `grid_mode.py` :

   * la boucle qui **fabrique les tuiles** (stack local des frames d’une tuile),
   * la partie qui **assemble les tuiles** dans le canevas global (reprojection et coadd).

2. Adapter le flux :

   * Choix de la tuile de référence :

     * ne pas prendre “bêtement” la toute première tuile,
     * parcourir les tiles stackées et sélectionner la **première tuile “saine”** :

       * coverage suffisante (pas entièrement NaN),
       * min/max finies,
       * dynamique non nulle (min < max),
     * enregistrer cette tuile dans une variable claire, par ex. `reference_tile_data` / `reference_tile_id`,
     * si une carte de coverage / poids par tuile existe déjà, garder aussi la coverage de référence.

**Cas limite : aucune tuile saine trouvée**

Si aucune tuile ne satisfait les critères de validité (coverage > 0, min/max finies, dynamique non nulle), alors :

- log WARNING : "[GRID] Photometric scaling disabled: no valid reference tile found (all tiles invalid)";
- désactiver entièrement la normalisation inter-tile pour ce run ;
- traiter toutes les tuiles comme identiques : gain = 1.0, offset = 0.0.

Ce fallback empêche un plantage et garantit un comportement déterministe même sur un dataset sévèrement corrompu.

   * Pour chaque tuile “target” :

     * vérifier qu’elle contient des pixels valides ; si entièrement NaN → la laisser telle quelle dans l’assemblage, mais **ne pas** calculer de scaling (gain=1, offset=0 + log),
     * construire, pour le couple (référence, target), un **masque de recouvrement** basé sur la coverage WCS :

       * utiliser les cartes de coverage / poids des deux tuiles si disponibles,
       * sinon, utiliser au minimum la condition “données valides dans les deux tuiles”,
       * éventuellement éroder ce masque pour écarter les bords.

**Fallback si la coverage n’est pas disponible**

Si aucune carte de coverage/poids n’est accessible pour une tuile :

- construire un masque minimal basé sur les pixels valides :
  `base_mask = np.isfinite(ref_tile) & np.isfinite(target_tile)`
- utiliser exclusivement ce masque ;
- log INFO : "[GRID] Coverage not available for tile X, falling back to finite-pixel mask."

Ce fallback garantit que le scaling ne dépend jamais exclusivement de la présence d’une carte de coverage.
     * appeler `compute_tile_photometric_scaling(...)` avec ce `mask` :

       * `(gains, offsets) = compute_tile_photometric_scaling(reference_tile_data, target_tile_data, mask=overlap_mask)`,
     * appliquer `apply_tile_photometric_scaling(...)` sur la tuile (`float32`),
     * seulement ensuite procéder à la reprojection sur le canevas global.

3. Précautions :

   * Ne jamais forcer du float64, rester sur du float32.
   * Respecter la forme HWC :

     * si la tuile est mono-canal, la représenter temporairement en `(H, W, 1)` pour les helpers.
   * Gérer proprement les NaN :

     * les helpers de scaling doivent ignorer les pixels non valides,
     * les éventuels NaN restants dans la tuile ne doivent pas faire diverger les min/max / médianes ni produire des gains/offsets `inf`/`nan`.
   * Le scaling photométrique doit uniquement modifier les valeurs de données valides. Les NaN existants (zones hors coverage) doivent rester NaN, et les cartes de coverage / poids / alpha utilisées pour le feathering ne doivent pas être modifiées par ces helpers.

**Interaction avec feathering**

Le scaling photométrique ne doit jamais modifier :

- les cartes de coverage / poids / alpha ;
- les NaN représentant les zones hors champ.

Le feathering doit continuer de fonctionner sur les cartes de poids originales.  
Le scaling n’affecte que les valeurs valides (float32) de la tuile.

---

#### Étape 4 – Réintroduire l’égalisation RGB

1. Toujours dans `grid_mode.py` :

   * Repérer le moment où la tuile RGB stackée est disponible **juste après le stacking local**.
   * Appeler `equalize_rgb_medians_inplace(...)` sur la tuile stackée, **avant** la normalisation inter-tile (`compute_tile_photometric_scaling` / `apply_tile_photometric_scaling`), exactement comme dans la pipeline classique.

2. Conditions :

   * Ne pas casser la prise en charge mono-canal : ne rien faire si `C == 1`.
   * S’assurer que cette égalisation est **cohérente** avec la logique de la pipeline classique (même type de données, même ordre approx.).
   * Ajouter un log indiquant quand l’égalisation RGB est appliquée (et sur combien de frames / canaux).

---

#### Étape 5 – Nettoyage / perf minimal

Dans ce ticket, ne viser que les gains faciles et non risqués :

1. **Dtypes :**

   * s’assurer que les arrays créés dans Grid Mode sont en `np.float32` / `cp.float32` par défaut.
   * éviter les promotions implicites en float64.

2. **Reprojects redondants :**

   * quand une tuile a déjà été reprojetée une fois sur le canevas global, ne pas la reprojeter par canal ou à nouveau si ce n’est pas strictement nécessaire.
   * mutualiser la reprojection sur un array HWC complet quand c’est possible.

3. **SciPy / ndimage :**

   * garder l’import, mais éviter les appels répétés dans les boucles les plus grosses,
   * si un filtre (ex. léger Gaussian) est utilisé juste pour lissage de coverage, ne l’appliquer qu’une fois sur une carte de coverage, pas sur toutes les tuiles.

---

### 6. Critères d’acceptation

* Sur un dataset représentatif de Grid Mode :

  * la mosaïque **n’a plus de damier visible**,
  * le fond est raisonnablement uniforme (à stretch égal),
  * les dominantes de couleur par tuile sont fortement réduites.

* Les logs `[GRID]` montrent :

  * des min/max/median cohérents avant/après scaling,
  * des `gains/offsets` finis (pas de `nan`/`inf`),
  * des messages clairs quand le scaling est désactivé pour une tuile (par ex. “no valid pixels”).

* Le temps de run Grid Mode :

  * reste dans le même ordre de grandeur que la version actuelle,
  * ou s’améliore légèrement,
  * en tout cas ne **double pas**.

* La pipeline classique :

  * continue de fonctionner **strictement comme avant**,
  * les tests/runs existants ne montrent pas de régression.

Non-objectifs :

* Ne pas introduire de modèle de fond 2D (Option B) dans ce ticket.
* Ne pas toucher à la pipeline classique.
* Ne pas modifier la logique de feathering au-delà de ce ticket.

---

### Suivi d’implémentation

* [x] Ajouter les helpers `compute_tile_photometric_scaling` / `apply_tile_photometric_scaling` pour la normalisation inter-tile (float32, logs min/med/max, masques NaN safe).
* [x] Brancher ces helpers dans `grid_mode.py` avec sélection d’une tuile de référence et application du scaling avant l’assemblage global.
* [x] Réintroduire l’égalisation RGB par tuile juste après le stacking local.
* [x] Ajouter et exploiter un masque de recouvrement basé sur la coverage / WCS pour le calcul des gains/offsets (Option C).
* [ ] Effectuer le passage de perf minimal (réduction des reproject redondants / filtres SciPy et vérification des dtypes) si nécessaire.

````

---
