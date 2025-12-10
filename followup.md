
## `followup.md` – Comment guider Codex et vérifier le travail

### 1. Séquence d’utilisation pour Codex

1. Ouvrir dans l’IDE :

   * `grid_mode.py` 
   * `zemosaic_stack_core.py` 
   * `zemosaic_worker.py` 

2. Demander à Codex :

   * de **lister les fonctions** de Phase 5 dans `zemosaic_worker.py` liées à la photométrie / renorm,
   * de te montrer où est appelé `equalize_rgb_medians_inplace` dans la pipeline classique.

3. Lui faire implémenter les deux helpers dans `zemosaic_stack_core.py` ou `zemosaic_utils.py` :

   * `compute_tile_photometric_scaling(...)`,
   * `apply_tile_photometric_scaling(...)`.

4. Ensuite, dans `grid_mode.py` :

   * lui demander de brancher ces helpers :

     * choix d’une tuile de référence,
     * calcul et application du scaling photométrique sur chaque tuile avant reprojection.

5. Enfin, lui faire ajouter les appels à `equalize_rgb_medians_inplace` au moment opportun (frames d’une tuile ou master-tile).

---

### 2. Checks manuels à faire après patch

1. **Compilation / exécution basique :**

   * Lancer ZeMosaic comme d’habitude (Tk ou Qt peu importe).
   * Vérifier qu’aucun import ne casse :

     * pas de `ImportError`,
     * pas de `AttributeError` sur les nouveaux helpers.

2. **Run Grid Mode sur un dataset de test :**

   * Reprendre exactement le dataset qui a produit les deux images que tu m’as montrées (classique vs Grid).
   * Lancer Grid Mode avec les mêmes paramètres que précédemment.

3. **Comparer visuellement :**

   * le damier doit être **fortement atténué** ou disparu,
   * les tuiles ne doivent plus “claquer” par paquets,
   * les bandes de couleur doivent être nettement réduites.

4. **Surveiller les logs :**

   * logs `[GRID]` :

     * vérifier la présence de logs DEBUG sur les min/max/median des tuiles avant/après scaling,
     * surveiller les éventuels warnings `nan` / `inf`.

   * si tu vois un message type “photometric scaling disabled / fallback”, c’est que la nouvelle logique n’est pas réellement utilisée → à corriger.

5. **Impact sur le temps de run :**

   * noter grossièrement le temps total de traitement **avant** / **après** sur le même dataset,
   * si le temps a explosé, regarder :

     * le nombre d’appels à `reproject_interp`,
     * d’éventuels filtres SciPy encore placés dans des boucles.

---

### 3. Points à surveiller / pièges possibles

* **NaN / inf :**
  S’assurer que les helpers photométriques ignorent correctement les pixels non finis, surtout dans les zones de jointure.

* **RGB vs mono :**
  Ne pas forcer l’égalisation RGB sur des tuiles mono-canal.

* **GPU vs CPU :**
  Les helpers photométriques peuvent rester en NumPy CPU, tant qu’ils travaillent sur les tiles **après rapatriement** depuis le GPU. Ne pas essayer d’optimiser ça en CuPy dans ce ticket.

* **Compatibilité :**
  Ne pas changer les interfaces publiques de `stack_core` ni des fonctions déjà utilisées par la pipeline classique.

