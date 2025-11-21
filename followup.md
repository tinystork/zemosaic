# Follow-up — Étapes concrètes pour optimiser CPU/GPU (sans changer la logique)

## Étape 0 — Lecture & repérage

- [x] Parcourir les fichiers clés du worker et du stacker (`zemosaic_worker.py`, `zemosaic_align_stack.py`, `zemosaic_utils.py`, `cuda_utils.py`, `parallel_utils.py`, `zemosaic_config.py`).
- [x] Identifier la structure du **parallel plan**, où il est calculé/injecté (P3, Mosaic-First, Phase 4.5, Phase 5, SDS) et repérer les paramètres/workers parallélisables.

> Ne PAS modifier l’ordre des phases ni les conditions SDS ON/OFF.

---

## Étape 1 — Consolider / créer le Parallel Plan

- [x] Réutiliser `parallel_utils` pour centraliser détection CPU/RAM/GPU et construction du plan (dataclass `ParallelPlan`).
- [x] Créer un module dédié si absent (N/A ici : `parallel_utils` déjà présent).
- [x] Mettre en place/renforcer les heuristiques (workers ≈ 75–90 % des cœurs, budget RAM/VRAM, activation GPU conditionnelle, tailles de chunks/memmap).

> Le plan doit **toujours** garder une marge de sécurité (RAM/VRAM).

---

## Étape 2 — Propager le plan sans changer la logique

- [x] Calculer une fois un `global_parallel_plan` (option : variantes par “kind”) et le stocker dans la config ou le cache worker.
- [x] P3 — Master Tiles : passer le plan aux fonctions de stack/align et utiliser `cpu_workers` / chunks si supportés.
- [x] Mosaic-First : transmettre le plan au global coadd pour `process_workers`, `rows_per_chunk`, `max_chunk_bytes`.
- [x] Phase 4.5 : plan transmis pour stacking/photométrie intra-master (hints chunk/workers disponibles).
- [x] Phase 5 : exploiter le plan pour `assembly_process_workers`, memmap/chunk rows CPU/GPU.
- [x] SDS : appliquer un plan SDS (méga-tuiles, chunks, GPU éventuel, limite à 1 job GPU si besoin).

> Dans tous ces cas : tu modifies uniquement les *paramètres* de parallélisation, pas la logique métier.

---

## Étape 3 — Optimiser l’usage CPU

- [x] Identifier les boucles séquentielles qui appliquent un traitement **indépendant** à des stacks/tiles/méga-tuiles SDS.
- [x] Introduire, là où c’est pertinent, des appels à `ProcessPoolExecutor` ou `ThreadPoolExecutor` en respectant `parallel_plan.cpu_workers` et sans changer l’ordre des traitements.
- [x] Garder des tâches suffisamment grosses et limiter les transferts inter-process (s’appuyer sur disque/memmap existants si besoin).

---

## Étape 4 — Optimiser l’usage GPU

- [ ] Utiliser exclusivement les wrappers GPU existants (`reproject_and_coadd_wrapper`, etc.) sans nouveaux kernels ni changement d’API publique.
- [ ] Appliquer les hints du plan (`gpu_rows_per_chunk`, `max_chunk_bytes`, `process_workers` si dispo), gérer les `TypeError` et fallback CPU.
- [ ] Vérifier que les phases GPU-intensives envoient des chunks raisonnables et que la VRAM reste sous contrôle (pas d’OOM).

---

## Étape 5 — Pas de cap fixe sur le nombre d’images

- [ ] Vérifier qu’aucune nouvelle limite dure (type `min(n_images, 50)` / truncation) n’est introduite sur tiles/méga-tiles/stacks.
- [ ] Remplacer les anciens caps par des heuristiques souples (basées mémoire) sans jeter d’images si encore présents.

---

## Étape 6 — Logging & vérifications

- [ ] Ajouter des logs (niveau INFO/DEBUG) par phase lourde (`parallel_plan_summary`, cpu/gpu rows, chunks, memmap...).
- [ ] Vérifier sur jeux de tests (small/medium/large) l’absence d’OOM/erreurs, le gain de temps P3/P4.5/P5 et la cohérence scientifique.
- [ ] Préserver les clés de log et messages d’erreur utilisés par la GUI/ETA.

---

## Étape 7 — Validation finale

- [ ] Lancer plusieurs scénarios (SDS OFF petite/grosse, SDS ON multi-nuit, CPU-only vs CPU+GPU, Mosaic-First ON/OFF si dispo).
- [ ] Vérifier montée de charge CPU/GPU, absence d’exceptions et invariance des options utilisateur.
- [ ] Conclure en validant parallélisation accrue sans régression fonctionnelle.
