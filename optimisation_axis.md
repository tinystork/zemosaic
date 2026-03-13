# ZeMosaic — optimisation_axis

Date: 2026-03-13
Scope: état actuel du projet + axes d’optimisation GPU/CPU basés sur le code existant.

---

## 1) Lecture rapide de l’architecture actuelle

## Points solides
- Pipeline très mature avec gestion de gros volumes.
- Fallback CPU/GPU déjà en place et plutôt robuste (safe mode, checks VRAM, parity checks).
- Découplage partiel déjà amorcé (`zemosaic_align_stack_gpu.py`, `zemosaic_stack_core.py`, `zemosaic_gpu_safety.py`, `core/robust_rejection.py`).

## Dette structurelle
- **Monolithe principal**: `zemosaic_worker.py` (~30k lignes) centralise orchestration + logique métier + IO + stratégie de parallélisme.
- GUI Tk/Qt très volumineuses (risque de duplication et divergence fonctionnelle).
- Difficulté de test ciblé par phase (Phase 4.5 et post-traitements fortement couplés à l’orchestrateur).

---

## 2) Cartographie des zones GPU existantes (confirmées)

- **Phase 3 stacking GPU**
  - Point d’entrée: `zemosaic_align_stack_gpu.py::gpu_stack_from_arrays`
  - Orchestration: `zemosaic_worker.py::_phase3_gpu_candidate` + appel helpers GPU
  - Support winsor/kappa/chunks + fallback CPU.

- **Phase 5 reprojection/coadd GPU**
  - Gating: `zemosaic_worker.py::should_use_gpu_for_reproject`
  - Pipeline post-stack: `zemosaic_worker.py::_apply_phase5_post_stack_pipeline`
  - Safety runtime: guards VRAM/battery/hybrid/safe mode.

- **Grid mode (flag GPU)**
  - Entrée: `grid_mode.py::run_grid_mode`
  - Lit `use_gpu_grid`; activation dépend ensuite des chemins de code appelés.

---

## 3) Phases où le GPU est absent/partiel et où l’ajout est intéressant

## A. Astrometry solve (Phase solve externe)
- Référence: `zemosaic_astrometry.py::solve_with_astap`
- Nature du coût: process externe + I/O + latence solveur ASTAP.
- Verdict GPU: **faible intérêt** (bottleneck non vectoriel local).
- Axe recommandé: optimiser queue/concurrence/disque plutôt que CUDA.

## B. I/O FITS/cache/memmap, préparation données
- Coût dominant: lecture/écriture disque, conversion buffers.
- Verdict GPU: **faible à moyen** selon machine; souvent I/O-bound.
- Axe recommandé: batching, locality disque, préfetch, format cache homogène.

## C. Phase 4.5 inter-master merge (gros levier)
- Référence: `zemosaic_worker.py::_run_phase4_5_inter_master_merge`
- Cette phase combine:
  - overlap graph,
  - reprojections locales,
  - normalisation photométrique inter-tuiles,
  - stacking intermédiaire.
- Verdict GPU: **très bon potentiel** sur grands jeux de tuiles.
- Candidats GPU prioritaires:
  1. Reprojection locale par chunk (canaux séparés + accumulateurs GPU)
  2. Statistiques robustes (median/MAD/clipping) sur lots
  3. Normalisation photométrique inter-tuiles (estimations robustes massives)

## D. Post-process finaux (quality crop / DBE-like / morpho)
- Indices code: usage récurrent de `cv2`, filtres gaussiens, opérations morphologiques, stats robustes.
- Verdict GPU: **potentiel moyen à élevé** selon taille mosaïque finale.
- Candidats:
  - blur/smoothing fond,
  - masques morphologiques,
  - opérations pixel-wise massives.

---

## 4) Priorisation ROI (impact vs risque)

1. **Phase 4.5 GPU (ROI élevé)**
   - Impact attendu: fort sur datasets volumineux mosaïque.
   - Risque: moyen (parité scientifique à protéger).

2. **Post-process GPU ciblé (ROI moyen/élevé)**
   - Impact: surtout sur très grandes sorties.
   - Risque: moyen (différences numériques, dépendance backend OpenCV/CUDA).

3. **I/O pipeline tuning CPU-first (ROI moyen)**
   - Impact: universel, y compris sans GPU.
   - Risque: faible.

4. **Astrometry GPU: non prioritaire**
   - Impact: faible.
   - Risque: effort non rentable.

---

## 5) Plan d’exécution recommandé (phases de delivery)

## Phase 0 — Baseline & métriques (obligatoire)
- Ajouter bench normalisés par dataset type:
  - temps Phase 3, 4.5, 5
  - débit I/O
  - VRAM max, RAM max
  - fallback count (GPU→CPU)
- Sortie CSV stable + snapshots config.

## Phase 1 — Refacto structure minimale (avant nouveau GPU massif)
Objectif: réduire le risque d’intégration.
- Extraire de `zemosaic_worker.py` vers modules:
  - `pipeline/phase45_merge.py`
  - `pipeline/phase5_assembly.py`
  - `pipeline/phase3_stack.py`
  - `pipeline/astrometry.py`
- Garder interfaces pures (inputs/outputs explicites), sans état global implicite.

## Phase 2 — GPU enablement Phase 4.5
- Introduire backend abstraction CPU/GPU par bloc critique (sur modèle de Phase 3/5).
- Chunking strict (budget VRAM + backoff automatique).
- Parité CPU/GPU:
  - tests delta max / médian,
  - seuils documentés,
  - fallback transparent si dépassement.

## Phase 3 — GPU post-process final
- Cibler d’abord les opérations les plus massives:
  - gaussian blur fond,
  - opérations morphologiques,
  - transforms pixel-wise.
- Laisser auto-fallback CPU si backend indisponible.

## Phase 4 — Optimisations I/O transverses
- Uniformiser format cache intermédiaire.
- Préfetch séquentiel + limitation accès aléatoires.
- Ajuster politiques memmap selon profil disque détecté.

---

## 6) Garde-fous qualité indispensables

- Toujours conserver **CPU reference path** activable.
- Tests de non-régression:
  - bitwise impossible (normal), donc comparer métriques robustes:
    - max abs delta,
    - median abs delta,
    - SNR/étoiles détectées,
    - couverture et artefacts de bord.
- Journaliser la décision runtime:
  - `requested_gpu`, `effective_gpu`, `fallback_reason`.

---

## 7) Conclusion synthétique

Le projet est déjà techniquement solide côté GPU sur Phase 3/5. Le meilleur axe d’amélioration maintenant est:

1. **industrialiser la structure (désenchevêtrer `zemosaic_worker.py`)**,
2. **porter Phase 4.5 sur un backend GPU chunké avec fallback propre**,
3. **accélérer ensuite les post-process lourds**.

L’astrometry externe (ASTAP) ne doit pas être la cible GPU prioritaire: l’effort irait majoritairement contre un bottleneck qui n’est pas CUDA-friendly.
