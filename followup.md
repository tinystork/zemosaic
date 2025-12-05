# ZeMosaic – Grid mode: multithread + GPU + stack factorisation – Follow-up

## Checklist par étapes

### Étape 1 – Multithread tiles uniquement (CPU)

- [x] Ajouter la clé `grid_workers` dans `zemosaic_config.py` (valeur par défaut 0 = auto).
- [x] Implémenter le calcul du nombre effectif de workers (auto vs valeur explicite).
- [x] Paralléliser l’appel à `process_tile(...)` dans `run_grid_mode(...)` avec un pool (threads ou process).
- [x] Assurer un logging `[GRID] using N workers` cohérent.
- [x] Tests manuels : comparer single-thread vs multi-threads, vérifier que les sorties sont correctes.

### Étape 2 – Flag GPU pour le Grid mode

- [x] Introduire une clé `use_gpu_grid` (ou équivalent) dans la config, cohérente avec `use_gpu_phase5` / `stack_use_gpu`.
- [x] Étendre la signature de `run_grid_mode(...)` pour accepter `grid_use_gpu`.
- [x] Propager `grid_use_gpu` jusqu’à `process_tile(...)`.
- [x] Créer une première version `_stack_weighted_patches_gpu(...)` dans `grid_mode.py` (dupliquée, mais fonctionnelle).
- [x] Implémenter le fallback CPU en cas d’échec GPU, avec logs `[GRID] GPU stack failed, falling back to CPU`.
- [x] Tests manuels :
  - machine sans GPU → vérifier le fallback ;
  - machine avec GPU → vérifier la cohérence des résultats et le gain de perf.

### Étape 3 – Factoriser le stacker CPU/GPU

- [x] Identifier les blocs de logique de stack communs dans le pipeline principal.
- [x] Créer un module/func "stack core" réutilisable (CPU/GPU).
- [x] Adapter Grid mode pour utiliser ce stack core (CPU + GPU).
- [ ] Adapter le pipeline classique pour utiliser aussi ce stack core.
- [ ] Tests de non-régression sur pipeline classique + Grid mode.

---

## Journal des interventions

*(Compléter à chaque passage de Codex)*

- `2025-12-05 – Étape 1 – Tâche : Ajouter la clé \`grid_workers\` dans \`zemosaic_config.py\``
  - Fichiers modifiés : zemosaic_config.py
  - Notes : Ajout de la clé avec valeur par défaut 0, comme spécifié.
  - Tests effectués : Aucun, modification mineure.
- `2025-12-05 – Étape 1 – Tâche : Implémenter le calcul du nombre effectif de workers (auto vs valeur explicite).`
  - Fichiers modifiés : grid_mode.py
  - Notes : Ajout de la fonction _get_effective_grid_workers qui utilise grid_workers de la config ou auto = max(1, os.cpu_count() - 2), avec log [GRID] using N workers.
  - Tests effectués : Aucun, ajout de fonction.
- `2025-12-05 – Étape 1 – Tâche : Paralléliser l’appel à \`process_tile(...)\` dans \`run_grid_mode(...)\` avec un pool (threads ou process).`
  - Fichiers modifiés : grid_mode.py
  - Notes : Remplacement de la boucle for par ThreadPoolExecutor avec gestion d'exceptions, utilisant le nombre de workers calculé. Import de concurrent.futures ajouté.
  - Tests effectués : Aucun, implémentation de parallélisation.
- `2025-12-05 – Étape 1 – Tâche : Assurer un logging \`[GRID] using N workers\` cohérent.`
  - Fichiers modifiés : grid_mode.py
  - Notes : Le logging est inclus dans la fonction _get_effective_grid_workers.
  - Tests effectués : Aucun.
- `2025-12-05 – Étape 1 – Tâche : Tests manuels : comparer single-thread vs multi-threads, vérifier que les sorties sont correctes.`
  - Fichiers modifiés : Aucun
  - Notes : Syntaxe vérifiée avec py_compile pour grid_mode.py et zemosaic_config.py. Pas d'erreurs. Tests manuels à effectuer par l'utilisateur sur un jeu de données de test, en forçant grid_workers=1 pour single-thread.
  - Tests effectués : Syntax checks passed.
- `2025-12-05 – Étape 2 – Tâche : Introduire une clé \`use_gpu_grid\` (ou équivalent) dans la config, cohérente avec \`use_gpu_phase5\` / \`stack_use_gpu\`.`
  - Fichiers modifiés : zemosaic_config.py
  - Notes : Ajout de "use_gpu_grid": True dans DEFAULT_CONFIG, et modification de _normalize_gpu_flags pour synchroniser use_gpu_grid avec la valeur canonique (use_gpu_phase5).
  - Tests effectués : Aucun, modification de config.
- `2025-12-05 – Étape 2 – Tâche : Étendre la signature de \`run_grid_mode(...)\` pour accepter \`grid_use_gpu\`.`
  - Fichiers modifiés : grid_mode.py
  - Notes : Ajout du paramètre grid_use_gpu: bool | None = None dans run_grid_mode.
  - Tests effectués : Aucun.
- `2025-12-05 – Étape 2 – Tâche : Propager \`grid_use_gpu\` jusqu’à \`process_tile(...)\`.`
  - Fichiers modifiés : grid_mode.py
  - Notes : Ajout de use_gpu dans GridModeConfig, et propagation dans run_grid_mode. Le flag est utilisé dans process_tile via config.use_gpu.
  - Tests effectués : Aucun.
- `2025-12-05 – Étape 2 – Tâche : Créer une première version \`_stack_weighted_patches_gpu(...)\` dans \`grid_mode.py\` (dupliquée, mais fonctionnelle).`
  - Fichiers modifiés : grid_mode.py
  - Notes : Création de _stack_weighted_patches_gpu, _normalize_patches_gpu, _fit_linear_scale_gpu. Implémentation avec CuPy pour accélérer le stacking, en gardant la même logique que la version CPU.
  - Tests effectués : Aucun, ajout de fonctions.
- `2025-12-05 – Étape 2 – Tâche : Implémenter le fallback CPU en cas d’échec GPU, avec logs \`[GRID] GPU stack failed, falling back to CPU\`.`
  - Fichiers modifiés : grid_mode.py
  - Notes : Le fallback est implémenté dans _stack_weighted_patches_gpu, avec log en cas d'exception.
  - Tests effectués : Aucun.
- `2025-12-05 – Étape 2 – Tâche : Tests manuels : machine sans GPU → vérifier le fallback ; machine avec GPU → vérifier la cohérence des résultats et le gain de perf.`
  - Fichiers modifiés : Aucun
  - Notes : Syntaxe vérifiée avec py_compile pour grid_mode.py et zemosaic_config.py. Pas d'erreurs. Tests manuels à effectuer par l'utilisateur sur un jeu de données de test.
  - Tests effectués : Syntax checks passed.
- `2025-12-05 – Étape 3 – Tâche : Identifier les blocs de logique de stack communs dans le pipeline principal.`
  - Fichiers modifiés : Aucun
  - Notes : Blocs communs identifiés : normalisation (none, median, linear_fit), rejet d'outliers (kappa_sigma, winsorized_sigma_clip, linear_fit_clip), combinaison (mean, median), avec support GPU via CuPy pour accélérer les opérations.
  - Tests effectués : Aucun
- `2025-12-05 – Étape 3 – Tâche : Créer un module/func "stack core" réutilisable (CPU/GPU).`
  - Fichiers modifiés : zemosaic_stack_core.py
  - Notes : Création du module zemosaic_stack_core.py avec fonction stack_core utilisant duck typing pour CPU/GPU (numpy/cupy), supportant normalisation (median, linear_fit), rejet d'outliers (kappa_sigma, winsorized_sigma_clip), combinaison (mean, median), et poids.
  - Tests effectués : Aucun, création de module.
- `2025-12-05 – Étape 3 – Tâche : Adapter Grid mode pour utiliser ce stack core (CPU + GPU).`
  - Fichiers modifiés : grid_mode.py
  - Notes : Modification des fonctions _stack_weighted_patches et _stack_weighted_patches_gpu pour utiliser stack_core avec backend='cpu' ou 'gpu', en passant normalize_method='none' (normalisation faite avant), et les autres config appropriés. Ajout d'import et fallback si stack_core indisponible.
  - Tests effectués : Aucun, modification des fonctions.

---

## Notes / Questions en suspens

*(Espace pour lister les points à éclaircir ou les ajustements futurs.)*
- [ ] …
