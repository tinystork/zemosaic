# Agent — Global Parallelization & CPU/GPU Utilization (Codex Max HIGH)

## 0. Mode d'intervention (IMPORTANT)

Tu tournes ici en **mode HIGH**, mais le projet est déjà très avancé et proche “production”.

👉 Ton rôle n’est **pas** de réinventer le pipeline, mais de :
- **augmenter l’efficacité CPU/GPU**,
- **sans changer la logique fonctionnelle**,
- **sans changer les branchements du pipeline**,
- **sans réintroduire de limite fixe comme le cap à 50 images** qui vient d’être levé.

Tu peux :
- modifier les **heuristiques de parallélisation** (nombre de workers, tailles de chunks, seuils mémoire),
- factoriser *légèrement* du code si nécessaire pour éviter les duplications évidentes,
- ajouter de la **télémétrie/logging** pour suivre CPU/GPU/mémoire.

Tu ne dois pas :
- changer le **workflow SDS vs non-SDS**,
- modifier les **phases** (1 → 6) ni l’ordre des étapes,
- introduire de nouvelles options GUI,
- changer le sens des options existantes,
- remettre des **caps arbitraires** (comme “max 50 frames par tuile”).

La règle d’or :  
> **Même input → même pipeline conceptuel → mêmes outputs** (à de très petites différences numériques près dues à l’ordre de réduction/float).

---

## 1. Mission

**Objectif :**  
Maximiser l’utilisation des ressources **CPU + GPU** dans toutes les grandes phases numériques (stacking, Mosaic-First, SDS, Phase 5), en exploitant au mieux :

- le **parallel plan** existant (ou à consolider) : nombre de workers, chunking, memmap,
- le **GPU helper** existant pour `reproject_and_coadd` / global coadd,
- les mécanismes de **streaming/chunking** déjà présents (alt-az cleanup, Mosaic-First, SDS, Phase 4.5, etc.).

**Sans modifier le comportement fonctionnel**, uniquement les **paramètres de parallélisation** et les **heuristiques d’autotune**.

---

## 2. Contexte (code & fichiers)

Le projet comporte déjà :

- Un **pipeline hiérarchique** avec phases :
  - P1–P2 : pré-tri et regroupement (“Seestar stacks”),
  - P3 : Master Tiles,
  - P4 : calcul de la grille finale (WCS global),
  - P4.5 : éventuel traitement intermédiaire / super-tuiles,
  - P5 : assemblage final (Incremental / Reproject & Coadd),
  - SDS : mode spécial “super-stack par lots” (méga-tuiles) qui **NE DOIT PAS être modifié logiquement**.
- Une logique de **Mosaic-First / Global coadd** (helper GPU ou CPU fallback).
- Une logique de **parallel plan** (ou équivalent) qui choisit :
  - `cpu_workers`,
  - `rows_per_chunk`, `tiles_per_chunk`,
  - `use_memmap`, `max_chunk_bytes`,
  - `use_gpu` / `gpu_rows_per_chunk`.

Tu dois **t’appuyer sur cette structure** et ne pas la remplacer.

---

## 3. Périmètre d’optimisation

Tu es autorisé à optimiser la parallélisation dans les zones suivantes :

1. **Stacking / Master Tiles (Phase 3)**  
   - Alignement intra-stack,
   - empilement des stacks,
   - éventuelle utilisation du GPU (si déjà présent dans ce code),
   - multi-process / multi-thread sur les stacks.

2. **Mosaic-First / Global coadd (Phase 4)**
   - Chemin “global coadd” (Mosaic-First) qui assemble les brutes directement sur la grille globale.
   - Utiliser le parallel plan pour :
     - mieux dimensionner le nombre de workers CPU,
     - optimiser `rows_per_chunk` / `max_chunk_bytes`,
     - exploiter le GPU helper plus efficacement.

3. **Phase 4.5 / Super-tuiles / micro-align / photométrie**
   - Les boucles qui :
     - reprojectent des tuiles par groupe,
     - appliquent des corrections photométriques,
     - font des coadds locaux.

4. **Phase 5 (assemblage final)**
   - Chemin **Reproject & Coadd** (classique, non SDS).
   - Chemin **Incremental** sur disque (si encore utilisé).
   - Chemin **SDS** (global stack à partir de méga-tuiles).

5. **SDS ON / SDS OFF**
   - SDS **OFF** : pipeline classique (Master Tiles → P4 grid → P5 assemble) doit rester inchangé logiquement.
   - SDS **ON** : pipeline SDS (méga-tuiles + super-stack global) doit rester inchangé logiquement, mais tu peux mieux répartir le travail entre CPU et GPU.

---

## 4. Ce que tu peux/dédois faire concrètement

### 4.1 Ajuster les heuristiques de parallel plan

- Centraliser les décisions de parallélisation dans un **module dédié** (par ex. `parallel_utils.py` / équivalent existant) qui :

  - détecte les capacités :
    - nombre de cœurs logiques,
    - RAM totale / disponible,
    - GPU dispo (CUDA) + VRAM totale / libre,
  - calcule pour chaque “kind” (ex. `"master_tiles"`, `"mosaic_first"`, `"phase5_global"`, `"sds_megatiles"`) un plan :
    - `cpu_workers` (plafonné par un facteur ex. 0.75–0.9 de cores),
    - `use_memmap` / `max_chunk_bytes`,
    - `rows_per_chunk` / `tiles_per_chunk`,
    - `use_gpu` + `gpu_rows_per_chunk`.

- **Tu peux modifier les heuristiques** pour viser :
  - **CPU** à ~70–90 % sur les phases lourdes,
  - **GPU** à ~50–90 % pendant les reprojects lourds,
  - tout en gardant une marge mémoire (par ex. 20–30 % de RAM/VRAM libre).

### 4.2 Utilisation CPU

- Là où le code a déjà un `ThreadPoolExecutor` / `ProcessPoolExecutor` ou paramètre `process_workers` :
  - Remplacer les constantes / configurations “à la main” par les valeurs du parallel plan.
- Si un code CPU est clairement **séquentiel** alors qu’il itère sur :
  - des tiles indépendantes,
  - des méga-tuiles indépendantes,
  - des stacks indépendants,

  tu peux introduire une **parallélisation simple** **sans changer la logique** :
  - encapsuler l’unité de travail dans une fonction pure,
  - mapper cette fonction sur un pool (taille dictée par `parallel_plan.cpu_workers`),
  - assembler les résultats exactement comme avant.

### 4.3 Utilisation GPU

- Tu dois utiliser le GPU uniquement là où des hooks existent déjà (par ex. `reproject_and_coadd_wrapper(..., use_gpu=True, ...)` ou équivalent).
- Tu peux modifier :
  - `rows_per_chunk`, `max_chunk_bytes` passés au helper GPU,
  - les conditions d’activation `use_gpu` en fonction du plan (GPU dispo + VRAM suffisante).

Tu ne dois pas :
- écrire de nouveaux kernels custom,
- modifier les algos de coadd / kappa-sigma / Winsor,
- changer le comportement des modes SDS / Mosaic-First.

### 4.4 Mémoire & robustesse

- L’autotune doit respecter **strictement** :
  - ne jamais allouer plus que, mettons, 70–80 % de la RAM disponible,
  - ne jamais tenter de consommer plus que 50–65 % de la VRAM disponible pour un job donné.
- En cas de `MemoryError` / erreur CUDA :
  - **réduire** les chunks ou le nombre de workers,
  - **basculer** en CPU si GPU indisponible,
  - mais ne pas interrompre toute la mosaïque si une fallback propre est possible.

### 4.5 Cap 50 images

- Il y avait historiquement un cap à 50 images par tuile / groupe → **il vient d’être levé**.
- Tu **ne dois pas** :
  - remettre de limite fixe type 50/100/200 frames ailleurs,
  - tronquer les listes d’images.
- Tes heuristiques doivent être **scalables** :
  - pour 10 images comme pour 10 000+ images,
  - en adaptant les workers / chunks à la mémoire disponible.

---

## 5. Non-régressions obligatoires

1. **SDS vs non-SDS**
   - SDS OFF → pipeline classique comme aujourd’hui (Master tiles → P4 → P5)  
     (aucune nouvelle branche conditionnelle ne doit changer ce chemin).
   - SDS ON → pipeline SDS existant (méga-tuiles + super-stack global)  
     (ne pas détourner ce mode vers d’autres fonctions).

2. **Résultats**
   - Pas de changement volontaire du résultat scientifique :
     - même système de coadd (mean, median, kappa-sigma, winsor),
     - mêmes normalisations photométriques globales,
     - même logique de coverage / alpha / cropping.
   - Des différences **minimes** de flottant dues à l’ordre de réduction sont acceptables, mais tu ne dois pas changer les formules.

3. **Pas de nouvelle logique GUI**
   - Tu ne touches pas à l’aspect fonctionnel des GUI Tk / Qt.  
   - Tu peux seulement :
     - accepter de nouveaux champs de config “sourds” (sans contrôle GUI),
     - améliorer le logging pour les messages de perf (profil parallel_plan, etc.).

4. **Compatibilité multi-OS**
   - Le pipeline **CPU** doit rester pleinement fonctionnel sur Windows / Linux / macOS.
   - Le GPU helper n’est activé que si CUDA+CuPy sont disponibles, sinon fallback CPU.

---

## 6. Critères de succès

- Sur une grosse mosaïque :
  - utilisation CPU nettement plus élevée sur les phases lourdes (P3, P4.5, P5),
  - utilisation GPU significative pendant les reprojects globaux (Mosaic-First, Phase 5, SDS).
- Pas d’augmentation notable du taux d’erreurs mémoire ou CUDA.
- Pas de changement de comportement SDS ON/OFF, Mosaic-First ON/OFF.
- Les utilisateurs retrouvent leurs habitudes de workflow, mais les traitements sont **sensiblement plus rapides** sur des machines multi-cœurs / GPU.
