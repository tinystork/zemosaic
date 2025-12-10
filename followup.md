## `followup.md` – Comment guider Codex et vérifier le travail

### 1. Séquence d’utilisation pour Codex

1. Ouvrir dans l’IDE :

   * `grid_mode.py`
   * `zemosaic_stack_core.py`
   * `zemosaic_worker.py`
   * éventuellement `zemosaic_utils.py` si les helpers sont placés là.

2. Demander à Codex :

   * de **lister les fonctions** de Phase 5 dans `zemosaic_worker.py` liées à la photométrie / renorm (`_apply_phase5_post_stack_pipeline`, `_apply_two_pass_coverage_renorm_if_requested`, etc.),
   * de te montrer où est appelé `equalize_rgb_medians_inplace` dans la pipeline classique et dans quelle “phase” du flux normal cela se situe (après débayer, avant gros stacking global).

3. Lui faire implémenter les deux helpers dans `zemosaic_stack_core.py` ou `zemosaic_utils.py` :

   * `compute_tile_photometric_scaling(...)` :

     * travaille en float32, sur des tuiles HWC,
     * gère explicitement les cas limites :
       * tuiles entièrement NaN ou sans pixels valides → `gain=1.0`, `offset=0.0` + log,
       * interdiction de retourner des `NaN`/`inf`,
     * renvoie des `gains`/`offsets` par canal.

   * `apply_tile_photometric_scaling(...)` :

     * applique les `gains`/`offsets` sur une tuile,
     * ne modifie pas le tableau source in-place,
     * renvoie une nouvelle tuile float32.

   * Ajouter dans ces helpers des logs DEBUG “min/max/median avant/après” pour aider au diagnostic.

4. Ensuite, dans `grid_mode.py` :

   * lui demander de brancher ces helpers :

     * **choix d’une tuile de référence saine** :
       * coverage non nulle, min/max finies, dynamique non nulle,
       * si la première tuile ne convient pas, passer à la suivante jusqu’à en trouver une correcte,
     * calcul du scaling pour chaque tuile par rapport à cette référence,
     * application du scaling **avant** reprojection sur le canevas global.

   * insister sur :
     * le maintien des dtypes en float32,
     * la gestion propre des NaN (pas de propagation en `inf`/`nan` dans les gains/offsets).

5. Enfin, lui faire ajouter les appels à `equalize_rgb_medians_inplace` au moment opportun :

   * appliquer sur la tuile stackée, avant la normalisation inter-tile,
   * dans tous les cas :
     * ne rien faire si l’image est mono-canal,
     * ajouter un log pour confirmer que l’égalisation RGB a bien été appelée.

---

### 2. Checks manuels à faire après patch

1. **Compilation / exécution basique :**

   * Lancer ZeMosaic comme d’habitude (Tk ou Qt peu importe).
   * Vérifier qu’aucun import ne casse :

     * pas de `ImportError`,
     * pas de `AttributeError` sur les nouveaux helpers,
     * pas de références circulaires bizarres.

2. **Run Grid Mode sur un dataset de test :**

   * Reprendre exactement le dataset qui a produit la comparaison “flux classique vs Grid Mode très moche”.
   * Lancer Grid Mode avec les **mêmes paramètres** que précédemment.

3. **Comparer visuellement :**

   * le damier doit être **fortement atténué** ou disparu,
   * les tuiles ne doivent plus “claquer” par paquets,
   * les bandes de couleur doivent être nettement réduites,
   * le fond de ciel doit être raisonnablement homogène.

4. **Surveiller les logs :**

   * logs `[GRID]` concernant la photométrie :

     * vérifier la présence de logs DEBUG sur les min/max/median des tuiles avant/après scaling,
     * vérifier que les `gains/offsets` sont des valeurs finies,
     * repérer les messages “no valid pixels, scaling disabled” pour les tuiles pathologiques.

   * logs liés à l’égalisation RGB :

     * confirmer qu’elle est bien appelée (log dédié),
     * vérifier qu’elle n’est pas appelée sur des images mono-canal,
     * vérifier dans les logs [GRID] que l’égalisation RGB est bien loggée au moment où chaque master-tile est produite, et pas dans la boucle par frame.

   * si tu vois un message type “photometric scaling disabled / fallback” pour **toutes** les tuiles, c’est que la nouvelle logique n’est pas réellement utilisée → à corriger.

5. **Impact sur le temps de run :**

   * noter grossièrement le temps total de traitement **avant** / **après** sur le même dataset,
   * si le temps a explosé, regarder en priorité :

     * le nombre d’appels à `reproject_interp`,
     * l’usage éventuel de filtres SciPy dans les boucles,
     * d’éventuelles copies inutiles float64.

---

### 3. Points à surveiller / pièges possibles

* **NaN / inf :**
  * S’assurer que les helpers photométriques ignorent correctement les pixels non finis.
  * Ne jamais laisser un calcul de médiane/moyenne produire un `NaN` utilisé ensuite pour un gain ou un offset.
  * En cas de problème, le fallback doit clairement être `gain=1.0`, `offset=0.0` avec un log explicite.

* **RGB vs mono :**
  * Ne pas forcer l’égalisation RGB sur des tuiles mono-canal.
  * En cas de doute sur le nombre de canaux, logguer la forme de la tuile.

* **Ordre des opérations pour RGB :**
  * Priorité à “débayer → stacking local → égalisation RGB (post-stack, par tuile) → normalisation inter-tile”.
  * Ne jamais appliquer l’égalisation RGB après la normalisation inter-tile, pour ne pas casser les gains/offsets déjà calculés.

* **GPU vs CPU :**
  * Les helpers photométriques peuvent rester en NumPy CPU, tant qu’ils travaillent sur les tiles **après rapatriement** depuis le GPU.
  * Ne pas essayer d’optimiser ça en CuPy dans ce ticket (risque de complexifier inutilement).

* **Compatibilité :**
  * Ne pas changer les interfaces publiques de `stack_core` ni des fonctions déjà utilisées par la pipeline classique.
  * Toute nouvelle fonction utilisée par Grid Mode doit être ajoutée de façon **optionnelle** (paramètres avec valeurs par défaut, etc.) pour ne pas casser les anciens call-sites.

---

En cas de doute, revenir aux objectifs simples :

* pas de damier,
* pas de `NaN`/`inf` dans les gains/offsets,
* égalisation RGB calée au même moment logique que dans la pipeline classique,
* et Grid Mode qui reste globalement au même ordre de grandeur de temps de calcul qu’avant.

# Checklist de suivi

- [x] Helpers de photométrie pour les tuiles créés et loggés (gains/offsets, min/med/max) en float32.
- [x] Helpers branchés dans `grid_mode.py` avec choix d’une tuile de référence saine et scaling appliqué avant l’assemblage.
- [x] Égalisation RGB appliquée au moment post-stack par tuile (skip si mono, log dédié).
- [ ] Revue des optimisations simples (dtypes, reproject redondants, SciPy dans les boucles) à compléter si nécessaire.

# Note pour moi même:
```
````
