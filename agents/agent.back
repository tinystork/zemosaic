# ZeMosaic – Grid mode: multithread + GPU + stack factorisation

## Contexte

ZeMosaic dispose d’un **Grid mode** (voir `grid_mode.py`) qui :
- lit un CSV de tuiles,
- reprojette les images sources sur une grille régulière,
- empile par tuile via `_stack_weighted_patches(...)`,
- assemble ensuite la mosaïque finale.

Aujourd’hui :
- le **Grid mode est 100% CPU, single-thread**, sans GPU ;
- il duplique une mini-logique de stacking déjà présente dans le pipeline principal
  (`zemosaic_worker.py`, `zemosaic_align_stack.py`, `zemosaic_align_stack_gpu.py`) ;
- il ne profite pas du **GPU** ni des stratégies de parallélisation existantes
  (`parallel_utils.py`, `zemosaic_config.py`).

Objectif global :  
**Étendre le Grid mode** pour qu’il profite :
- du **multithreading / multiprocess** au niveau des tuiles,
- du **GPU** pour le stacking des tuiles,
- et, à terme, d’une **factorisation** du stacker CPU/GPU pour réduire les duplications.

⚠️ Important :
- Implémentation **incrémentale en 3 étapes**.
- Toujours finir complètement une étape (et mettre à jour `followup.md`) avant d’attaquer la suivante.
- Ne pas casser le comportement actuel (Grid doit continuer à produire des tuiles correctes, même si lent).

---

## Fichiers concernés (principaux)

- `grid_mode.py`
- `zemosaic_worker.py`
- `zemosaic_config.py`
- `zemosaic_gui_qt.py` (pour la propagation des options depuis le GUI si nécessaire)
- `parallel_utils.py` (pour s’inspirer de la logique de calcul d’auto workers)
- éventuellement :
  - `zemosaic_align_stack.py`
  - `zemosaic_align_stack_gpu.py`

---

## Contraintes générales

- Ne pas casser la **voie classique** (pipeline non-grid).
- Ne pas modifier la sémantique des stackers existants (kappa-sigma, Winsor, radial weighting, etc.).
- Logguer clairement les actions Grid mode avec des tags `[GRID]` :
  - `[GRID] workers=...`
  - `[GRID] GPU=ON/OFF`
  - `[GRID] tile X processed by worker pid=...`, etc.
- Garder le comportement par défaut **safe** :
  - si config/flags absents → comportement actuel (single-thread CPU, pas de GPU).
  - si GPU indisponible ou erreur → fallback CPU propre, sans crash.

---

## Étape 1 – Multithread tiles uniquement (CPU)

### Objectif

Paralléliser le travail par tuile dans `run_grid_mode(...)`, sans changer la logique interne de `process_tile(...)` ni `_stack_weighted_patches(...)`.

L’idée :
- **Chaque tuile est indépendante** -> on peut distribuer `process_tile(tile, ...)` sur plusieurs workers ;
- Utiliser un petit pool (threads ou process) pour accélérer :
  - la reprojection des frames vers la tuile,
  - le stacking intra-tile.

### Tâches

- [ ] **Ajouter un paramètre de config `grid_workers`**
  - Ajouter une clé `grid_workers` dans la config (`zemosaic_config.py`), dans la section appropriée (Grid/Survey).
  - Valeur par défaut : `0` (signifie “auto”).
  - Documenter la clé dans les commentaires du JSON de config.

- [ ] **Implémenter un helper pour calculer le nombre de workers effectif**
  - Dans `grid_mode.py` ou via un petit helper commun :
    - Si `grid_workers > 0` → utiliser cette valeur.
    - Si `grid_workers <= 0` → calculer `auto_workers = max(1, os.cpu_count() - 2)` (ou logique similaire inspirée de `parallel_utils`).
    - Logguer via `[GRID] using N workers for tile processing`.

- [ ] **Paralléliser l’appel à `process_tile` dans `run_grid_mode(...)`**
  - Localiser la boucle actuelle du type :
    - `for tile in grid.tiles: process_tile(tile, ...)`
  - La remplacer par l’utilisation d’un pool (`concurrent.futures`):
    - Soit `ThreadPoolExecutor`, soit `ProcessPoolExecutor` (au choix, mais garder ça cohérent avec le reste du projet si une préférence existe).
    - Envoyer une tâche par tuile.
    - Gérer proprement les exceptions :
      - si une tuile plante, logger un message `[GRID] Tile X failed with error: ...` et continuer avec les autres tuiles ;
      - ne pas faire échouer tout le Grid mode pour une seule tuile.
  - Assurer que la barre de progression / callback `progress_callback` continue de fonctionner :
    - soit via des mises à jour dans `process_tile`,
    - soit via un wrapper qui reporte la progression.

- [ ] **Tests manuels – Étape 1**
  - Lancer Grid mode sur un petit jeu de données de test :
    - comparer :
      - single-thread (forcer `grid_workers=1`)
      - multi-threads (laisser `grid_workers=0` ou mettre une valeur >1)
    - vérifier que :
      - les tuiles produites sont identiques (ou numériquement quasi-identiques, dans les limites de l’ordonnancement CPU) ;
      - la mosaïque finale est identique ;
      - les logs montrent bien `[GRID] using N workers`.
  - En cas de problème, corriger avant de passer à l’étape 2.

---

## Étape 2 – Flag GPU pour le Grid mode

### Objectif

Permettre à Grid mode de réutiliser le **flag GPU global** déjà présent dans le pipeline, pour décider si le stacking des tuiles se fait :
- en CPU (comportement actuel),
- ou en GPU (via une nouvelle fonction `_stack_weighted_patches_gpu(...)` ou équivalent).

**Note** : à cette étape, l’objectif est d’introduire le chemin GPU proprement, **pas encore de factoriser le stacker**.

### Tâches

- [ ] **Propager un flag `grid_use_gpu` depuis la config**
  - S’appuyer sur la logique existante de `zemosaic_config._normalize_gpu_flags(...)` qui synchronise :
    - `use_gpu_phase5`
    - `stack_use_gpu`
    - `use_gpu_stack`
  - Décider d’une clé canonique pour le Grid, par exemple :
    - `use_gpu_grid` (nouvelle clé).
  - Comportement recommandé :
    - si `use_gpu_grid` n’est pas explicitement défini dans la config, utiliser la même valeur que `use_gpu_phase5` / `stack_use_gpu` (pour rester cohérent avec le pipeline principal).
    - logguer la valeur finale : `[GRID] GPU requested = True/False`.

- [ ] **Étendre la signature de `run_grid_mode(...)`**
  - Ajouter un paramètre keyword, par exemple `grid_use_gpu: bool | None = None`.
  - Si `grid_use_gpu` est `None`, le remplir à partir de la config (cf. précédent point).
  - Faire passer ce flag jusqu’à `process_tile(...)` (ajouter un paramètre ou encapsuler dans la config `GridModeConfig` si elle existe).

- [ ] **Créer un chemin `_stack_weighted_patches_gpu(...)` (version minimale)**
  - Dans `grid_mode.py` (étape 2 = duplication assumée) :
    - ajouter une fonction `_stack_weighted_patches_gpu(...)` qui :
      - prend les mêmes paramètres que `_stack_weighted_patches(...)` (patches, weights, config, reference_median, etc.) ;
      - essaie de convertir les données en CuPy et d’appliquer la même logique de stacking que la version CPU (kappa-sigma, Winsor, combine, radial weighting) ;
      - s’appuie autant que possible sur des helpers existants de `zemosaic_align_stack_gpu.py` si c’est simple ;
      - retourne les mêmes types (image stacked + weight_sum + ref_median_used).
  - Important : en cas d’**ImportError / erreur CuPy / autre exception GPU** :
    - logger une alerte `[GRID] GPU stack failed, falling back to CPU` ;
    - retourner `None` ou déclencher un fallback CPU explicite depuis l’appelant.

- [ ] **Utiliser le flag dans `process_tile(...)`**
  - Dans la fonction interne qui empile les chunks (`flush_chunk` ou équivalent) :
    - si `grid_use_gpu` est `True` :
      - tenter d’appeler `_stack_weighted_patches_gpu(...)`.
      - si ça échoue → fallback sur `_stack_weighted_patches(...)` + log claire.
    - si `grid_use_gpu` est `False` :
      - utiliser directement `_stack_weighted_patches(...)` (comportement actuel).

- [ ] **Tests manuels – Étape 2**
  - 2 jeux de tests :
    - machine **sans GPU** :
      - activer `use_gpu_grid=True` et vérifier que le code :
        - loggue bien un fallback CPU ;
        - ne crash pas ;
        - produit une mosaïque correcte.
    - machine **avec GPU** :
      - comparer Grid mode avec/ sans GPU sur un dataset représentatif ;
      - vérifier que :
        - les résultats sont numériquement proches (compte tenu des différences CPU/GPU) ;
        - le gain de performance est observable ;
        - aucun artefact évident (tuiles vides, NaN, etc.).

---

## Étape 3 – Factoriser le stacker CPU/GPU

### Objectif

Réduire la duplication de code entre :
- le stacker “classique” (pipeline principal),
- le stacker Grid mode (CPU & GPU).

L’idée est de créer un **“stack core”** réutilisable par les deux chemins.

### Tâches

- [ ] **Identifier le cœur de logique de stack dans le pipeline principal**
  - Localiser dans :
    - `zemosaic_align_stack.py`
    - `zemosaic_align_stack_gpu.py`
    - potentiellement `zemosaic_worker.py`
  - Les fonctions / blocs qui :
    - normalisent les frames (`linear_fit`, etc.),
    - gèrent les poids (`noise_variance`, etc.),
    - font le rejet (kappa-sigma, Winsor),
    - combinent (mean, median, percentile),
    - appliquent le radial weighting (si activé).

- [ ] **Créer un module/func commun de stack**
  - Créer une fonction ou un petit ensemble de fonctions “core”, par exemple dans un module commun (`zemosaic_utils.py` ou un nouveau `zemosaic_stack_core.py`), qui :
    - prend un stack d’images + poids (CPU ou GPU),
    - applique la pipeline de normalisation + weight + rejet + combine + radial,
    - ne dépend que de :
      - `numpy` ou `cupy` (via abstraction),
      - la config de stack (déjà présente).
  - Implémenter une petite abstraction “backend” CPU/GPU pour éviter de dupliquer toute la logique :
    - soit via duck typing (np vs cp),
    - soit via deux wrappers plus fins.

- [ ] **Adapter Grid mode pour utiliser ce stack core**
  - Remplacer :
    - `_stack_weighted_patches(...)` (CPU)
    - `_stack_weighted_patches_gpu(...)` (GPU)
  - par des appels au stack core :
    - en préparant les tensors `H x W x C` + poids dans le bon backend (CPU/GPU).
  - Assurer que les résultats restent compatibles avec le reste du code Grid.

- [ ] **Adapter le pipeline classique pour utiliser aussi le stack core**
  - Là où c’est raisonnable (sans tout casser d’un coup), remplacer la logique locale par des appels au stack core.
  - S’assurer que les tests / comportements existants restent valides.

- [ ] **Tests manuels – Étape 3**
  - Comparer :
    - ancienne version (avant factorisation) ;
    - nouvelle version (après factorisation) ;
  - sur :
    - pipeline classique,
    - Grid mode (CPU + GPU si dispo).
  - Vérifier qu’il n’y a pas de régression visible (histogrammes, couleurs, couverture).

---

## Règles de travail pour Codex

- Toujours commencer par **relire `agent.md` et `followup.md`**.
- Ne travailler que sur la **prochaine tâche non cochée**.
- Une fois la tâche terminée :
  - mettre la case en `[x]` dans `followup.md`,
  - ajouter un bref log sous forme de bullet point :
    - fichier touché,
    - décisions prises,
    - points d’attention éventuels.
- Ne pas lancer l’étape suivante tant que l’étape courante n’est pas terminée et validée.
- En cas de doute, ajouter un commentaire dans le code **sans modifier la logique** plutôt que d’inventer un comportement risqué.
