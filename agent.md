# agent.md

## Mission
Stabiliser la phase **Intertile** en **limitant automatiquement** le nombre de workers du ThreadPool interne (non exposé à l’utilisateur), afin d’éviter les crashs/kill externes Windows quand le log montre des valeurs agressives du type :
`[Intertile] Parallel: threadpool workers=15 pairs=4344 preview=512`.

Objectif : préserver les perfs, mais éviter les pics de pression mémoire/CPU.

## Contraintes
- Ne pas ajouter de nouvelle option GUI / config utilisateur.
- Patch minimal, pas de refactor massif.
- Garder la compatibilité Windows/Linux/macOS.
- Logging clair : afficher workers demandés vs workers effectifs + raisons du clamp.

## Fichiers probables
- `zemosaic_utils.py` (ou là où est l’Intertile / matching / photometric match)
- éventuellement `zemosaic_worker.py` (appelant / instrumentation), mais éviter si pas nécessaire
- éventuellement `zemosaic_gpu_safety.py` si on veut réutiliser un signal “safe_mode” (optionnel)

## Plan d’attaque (pas à pas)

### 1) Localiser le point de création du ThreadPool Intertile
- Chercher dans le repo les strings de log :
  - `"[Intertile] Parallel:"`
  - `"pairs="`
  - `"preview="`
  - `"ThreadPoolExecutor"`
- Identifier la fonction qui :
  - prépare la liste des paires (N~4000)
  - exécute le calcul par paire
  - crée `ThreadPoolExecutor(max_workers=...)`

### 2) Introduire une fonction utilitaire interne de clamp
Dans le même module que l’intertile (ou utilitaire local), ajouter un helper **petit et lisible** :

Pseudo-spéc :
- Inputs :
  - `requested_workers: int | None` (si existant)
  - `cpu_total: int`
  - `pairs_count: int`
  - `preview_size: int | None`
  - `platform_system: str` (via `platform.system()`)
  - RAM dispo si `psutil` dispo (best effort, guarded)
- Output :
  - `effective_workers: int`
  - `reasons: list[str]` (pour log)

Règles recommandées (conservatrices, simples) :
- Base auto (si requested invalide ou trop grand) :
  - `base = max(1, min(cpu_total - 1, cpu_total))`
- Hard cap “portable” :
  - `hard_cap = 8` (valeur robuste)
- Windows cap plus strict (car WDDM / scheduling) :
  - si Windows : `hard_cap = 6` (ou 8 si tu préfères, mais 6 est plus safe)
- Cap lié au volume de paires :
  - si `pairs_count >= 2000`: `hard_cap = min(hard_cap, 6)`
  - si `pairs_count >= 4000`: `hard_cap = min(hard_cap, 4)`
- Cap RAM (best effort si psutil dispo) :
  - si `available_mb < 6000`: `hard_cap = min(hard_cap, 4)`
  - si `available_mb < 3500`: `hard_cap = min(hard_cap, 2)`
- Toujours :
  - `effective = max(1, min(base, hard_cap))`
  - si `pairs_count` petit (<200), on peut autoriser `min(8, cpu_total-1)`.

Important : ne pas sur-compliquer. Le but est de stopper les crashs.

### 3) Appliquer le clamp au ThreadPool Intertile
- Remplacer `ThreadPoolExecutor(max_workers=workers)` par `max_workers=effective_workers`
- Juste avant de lancer, loguer :
  - workers_requested, cpu_total, pairs_count, preview
  - workers_effective et reasons

Format de log attendu (ex) :
`[Intertile] Parallel: threadpool workers=15 -> 4 (windows_cap,pairs>=4000) pairs=4344 preview=512`

### 4) Optionnel (mais utile) : éviter la sur-souscription OpenCV
Si Intertile utilise OpenCV et que c’est un facteur de surcharge CPU :
- ajouter un best-effort au début de la phase Intertile :
  - `cv2.setNumThreads(1)` (dans un try/except)
- log debug si appliqué
Ne pas casser si cv2 absent.

### 5) Vérifications / tests
Sans framework lourd, au minimum :
- test manuel : run sur dataset qui déclenche `pairs~4000`, vérifier le log clamp et la stabilité.
- test unitaire léger si déjà présent :
  - tester le helper de clamp avec plusieurs scénarios (Windows vs Linux, RAM basse, pairs 500/2000/4500)
- Vérifier que la valeur effective est toujours >=1.

## Definition of Done
- Intertile n’utilise plus “cpu_total-1” sans garde-fou.
- Le log montre clairement le clamp quand applicable.
- Pas d’option GUI ajoutée.
- Aucun changement sur “batch size 0 / >1” behaviour.
- Le run qui affichait `workers=15 pairs=4344` sort avec un workers effectif réduit (typiquement 4–6) et ne plante plus.

## Output attendu
- Un patch git propre (diff) limité aux fichiers nécessaires.
- Une note courte dans le commit message (ex: “Clamp Intertile threadpool workers to prevent Windows crashes on large pair counts”).

## Suivi
- [x] Localisation et clamp du ThreadPool Intertile effectués.
- [x] Logs enrichis pour montrer workers demandés vs effectifs.
