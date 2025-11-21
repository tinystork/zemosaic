# Follow-up — Étapes concrètes pour optimiser CPU/GPU (sans changer la logique)

## Étape 0 — Lecture & repérage

1. Parcourir les fichiers clés du worker et du stacker :
   - `zemosaic_worker.py`
   - `zemosaic_align_stack.py`
   - `zemosaic_utils.py` (pour `reproject_and_coadd_wrapper` ou helpers équivalents)
   - `cuda_utils.py`
   - `parallel_utils.py` (ou module équivalent si déjà présent)
   - `zemosaic_config.py`

2. Identifier :
   - Le type / structure du **parallel plan** (s’il existe déjà).
   - Où ce plan est calculé (ex. en début de `run_hierarchical_mosaic`) et comment il est injecté dans :
     - P3 (Master tiles),
     - Mosaic-First / global coadd,
     - Phase 4.5,
     - Phase 5 (Reproject & Coadd / Incremental),
     - SDS (assemble_global_mosaic_sds / finalize_sds_global_mosaic).
   - Tous les endroits où on a :
     - un paramètre `process_workers` / `max_workers` / `assembly_process_workers`,
     - des boucles séquentielles sur des listes de tiles/frames qui pourraient être parallélisées **sans changer le résultat**.

> Ne PAS modifier l’ordre des phases ni les conditions SDS ON/OFF.

---

## Étape 1 — Consolider / créer le Parallel Plan

1. S’il existe déjà un module type `parallel_utils` :
   - Le **réutiliser**, ne pas le remplacer.
   - Ajouter / renforcer :
     - une fonction pour détecter les capacités :
       - nb de cœurs logiques (via `multiprocessing.cpu_count()`),
       - RAM totale / disponible (via `psutil`),
       - GPU dispo + VRAM (via CuPy ou `cuda_utils`),
     - une fonction pour construire un plan par “kind” :
       - `"master_tiles"`, `"mosaic_first"`, `"phase5_global"`, `"sds_megatiles"`, etc.

2. Sinon, créer un petit module dédié :
   - avec une structure simple (dataclass `ParallelPlan`) contenant :
     - `cpu_workers`, `use_memmap`, `max_chunk_bytes`,
     - `rows_per_chunk` / `tiles_per_chunk`,
     - `use_gpu`, `gpu_rows_per_chunk`.

3. Heuristiques à mettre en place (ou renforcer) :
   - CPU :
     - `cpu_workers ≈ cores_logiques * 0.75–0.9`, plafonné par `parallel_max_cpu_workers` (config, 0 = auto).
   - RAM :
     - `max_chunk_bytes <= ram_disponible * parallel_target_ram_fraction` (ex. 0.7–0.8).
   - GPU (si CUDA dispo) :
     - `use_gpu=True` lorsque :
       - GPU détecté,
       - VRAM disponible suffisante pour au moins *un* chunk raisonnable.
     - `gpu_rows_per_chunk` dimensionné pour consommer 30–60 % de la VRAM libre par chunk.

> Le plan doit **toujours** garder une marge de sécurité (RAM/VRAM).

---

## Étape 2 — Propager le plan sans changer la logique

1. Dans la fonction principale du worker (ex. `run_hierarchical_mosaic` ou équivalent) :
   - Calculer **une fois** un `global_parallel_plan` (type “global” ou “phase5_global”).
   - Optionnel : calculer aussi des plans spécifiques (kind `master_tiles`, `sds_megatiles`, etc.) si le code s’y prête.
   - Stocker ces plans dans :
     - l’objet config (`zconfig.parallel_plan_*`),
     - ou un dict partagé (ex. `worker_config_cache["parallel_plan_*"]`).

2. Dans chaque fonction de haut niveau :

   - P3 — Master Tiles :
     - Passer un `parallel_plan_master_tiles` aux fonctions de stack/align.
     - Utiliser `cpu_workers` pour le nombre de workers.
     - Adapter éventuellement `tiles_per_chunk` / `rows_per_chunk` si déjà supporté.

   - Mosaic-First :
     - Passer `parallel_plan_mosaic_first` au helper global coadd.
     - Utiliser ses valeurs pour configurer `process_workers`, `rows_per_chunk`, `max_chunk_bytes`.

   - Phase 4.5 :
     - S’il y a des boucles sur des groupes / chunks de tuiles indépendants,  
       utiliser `cpu_workers` pour paralleriser ces groupes.

   - Phase 5 :
     - Pour le chemin **Reproject & Coadd classique** :
       - Utiliser `parallel_plan_phase5` pour :
         - `assembly_process_workers`,
         - `use_memmap` / `max_chunk_bytes`,
         - `rows_per_chunk` / `gpu_rows_per_chunk` vers le helper GPU.
     - Pour le chemin **SDS** :
       - Utiliser un plan spécifique `parallel_plan_sds` pour :
         - nombre de méga-tuiles traitées en parallèle,
         - taille des chunks, tout en gardant un max à 1 job GPU à la fois si nécessaire.

> Dans tous ces cas : tu modifies uniquement les *paramètres* de parallélisation, pas la logique métier.

---

## Étape 3 — Optimiser l’usage CPU

1. Identifier les boucles séquentielles qui appliquent un traitement **indépendant** à :
   - une liste de stacks,
   - une liste de tiles,
   - une liste de méga-tuiles SDS.

2. Introduire, là où c’est pertinent, des appels à `ProcessPoolExecutor` ou `ThreadPoolExecutor` :

   - En respectant `parallel_plan.cpu_workers`.
   - Sans modifier l’ordre ni la nature des transformations — tu renvoies les mêmes structures qu’avant.

3. Veiller à ce que :
   - les tâches envoyées au pool soient assez **grosses** (pas juste un pixel ou une ligne) pour limiter l’overhead Python,
   - les données transférées entre process restent raisonnables (éviter de passer des cubes gigantesques par pickling : préférer des chemins sur disque/memmap déjà prévus dans le projet si possible).

---

## Étape 4 — Optimiser l’usage GPU

1. Utiliser exclusivement le / les wrappers GPU existants (par ex. `reproject_and_coadd_wrapper`) :

   - Ne pas écrire de nouveaux kernels custom.
   - Ne pas modifier l’API publique de ces helpers, seulement leurs paramètres.

2. Pour chaque usage GPU :

   - Appliquer `parallel_plan.gpu_rows_per_chunk` et `parallel_plan.max_chunk_bytes` aux arguments :
     - `rows_per_chunk` / `max_chunk_bytes` / `process_workers` si supportés.
   - S’assurer que le code gère proprement :
     - les `TypeError` liés à des kwargs non supportés,
     - le fallback CPU en cas d’échec GPU.

3. Vérifier que :
   - les phases GPU-intensives (Mosaic-First, Phase 5 global, SDS global) envoient des chunks **raisonnablement gros** au GPU,
   - la VRAM n’est jamais saturée (pas d’OOM).

---

## Étape 5 — Pas de cap fixe sur le nombre d’images

1. S’assurer qu’aucune partie du code n’introduit de nouvelle limite de type :

   - `min(n_images, 50)` ou `if n_images > 50: truncate`.
   - Ni sur les tuiles, ni sur les méga-tuiles, ni sur les stacks.

2. Si tu vois des “caps” historiques, ils doivent :
   - soit être retirés (si la logique de sécurité mémoire est désormais assumée par le parallel plan),
   - soit transformés en heuristiques *douces* (ex. limiter la taille d’un batch en fonction de la mémoire, mais sans jeter des images).

---

## Étape 6 — Logging & vérifications

1. Ajouter des logs (niveau INFO / DEBUG) au début de chaque phase lourde, par ex. :

   - `parallel_plan_summary` avec :
     - `phase`, `cpu_workers`, `use_gpu`, `rows_per_chunk`, `max_chunk_bytes`, `gpu_rows_per_chunk`.

2. Vérifier sur un jeu de tests (small / medium / large) :

   - que les traitements terminent sans OOM / error,
   - que les temps de P3 / P4.5 / P5 diminuent sur une machine multi-cœurs + GPU,
   - que les résultats (FITS, PNG) restent scientifiquement cohérents, avec des différences numériques acceptables uniquement dues au parallélisme.

3. Ne jamais modifier :

   - les clés de log essentielles déjà utilisées par la GUI ou l’ETA,
   - les messages d’erreur sur la disponibilité de `reproject`, `astropy`, `fits`, etc.

---

## Étape 7 — Validation finale

1. Lancer plusieurs scénarios :

   - SDS OFF, petite mosaïque, mode CPU-only,
   - SDS OFF, grosse mosaïque, CPU + GPU,
   - SDS ON, campagne Seestar multi-nuit, CPU + GPU,
   - Optionnel : Mosaic-First ON/OFF si cette option existe.

2. Vérifier :

   - que les logs montrent bien une montée de charge CPU/GPU sur les phases attendues,
   - qu’aucune exception nouvelle n’apparaît,
   - qu’aucune option utilisateur existante n’a changé de sens.

3. Considérer la mission comme réussie **uniquement** si :
   - la parallélisation est plus agressive,
   - la stabilité est préservée,
   - et aucun comportement fonctionnel du pipeline n’a été modifié.
