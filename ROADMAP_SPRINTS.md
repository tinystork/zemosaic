# ZeMosaic — Roadmap Sprints

Date: 2026-03-13
Owner: Tristan / ZeMosaic core

---

## Vision
Accélérer fortement les gros traitements mosaïque tout en gardant une parité scientifique CPU↔GPU robuste et un fallback fiable.

---

## Sprint S1 — Baseline & Observabilité
**Durée:** 1 semaine  
**Effort estimé:** 4–6 JH  
**Priorité:** P0

### Objectifs
- Mesurer proprement avant toute optimisation.
- Rendre les runs comparables/reproductibles.

### Tickets
1. **Benchmark harness standardisé**
   - Ajouter `bench/run_benchmark.py`.
   - Support datasets `small / medium / large`.
   - Sortie JSON + CSV par run.
2. **Métriques techniques minimales**
   - Temps phase 3 / 4.5 / 5.
   - RAM max, VRAM max.
   - Nombre de fallbacks GPU→CPU + motifs.
   - Débit I/O read/write estimé.
3. **Snapshot config run**
   - Export systématique de la config effective (runtime).
4. **Comparateur de runs**
   - Script simple de comparaison moyenne / p95 / écart-type.

### Dépendances
- Aucune.

### Critères de done
- 5 runs consécutifs stables par dataset.
- Tableau baseline partagé (mean, p95, variance).

---

## Sprint S2 — Refacto minimal du worker (dé-risque)
**Durée:** 1 à 2 semaines  
**Effort estimé:** 6–10 JH  
**Priorité:** P0

### Objectifs
- Réduire le couplage de `zemosaic_worker.py`.
- Isoler les zones critiques pour faciliter l’optimisation GPU.

### Tickets
1. **Extraction module phase 4.5**
   - Créer `pipeline/phase45_merge.py`.
   - Déplacer logique merge inter-master + signatures d’API claires.
2. **Extraction module phase 5**
   - Créer `pipeline/phase5_assembly.py`.
   - Conserver la logique actuelle, sans changement fonctionnel.
3. **Extraction module phase 3 orchestration**
   - Créer `pipeline/phase3_stack.py`.
   - Garder `zemosaic_align_stack_gpu.py` comme backend GPU.
4. **Worker orchestrateur allégé**
   - `zemosaic_worker.py` conserve orchestration + dispatch.

### Dépendances
- S1 recommandé (pour valider non-régression perf/fonctionnelle).

### Critères de done
- Aucun changement de résultat visible pour l’utilisateur.
- Tests existants passent.
- Réduction nette des responsabilités dans le worker.

---

## Sprint S3 — GPU Enablement ciblé Phase 4.5
**Durée:** 2 semaines  
**Effort estimé:** 8–12 JH  
**Priorité:** P0

### Objectifs
- Accélérer la phase la plus coûteuse et encore majoritairement CPU.

### Tickets
1. **Backend abstraction CPU/GPU pour 4.5**
   - API commune pour reprojection locale + stacking intra-groupe.
2. **Chunking VRAM-aware**
   - Dimensionnement chunk auto.
   - Backoff automatique en cas d’OOM GPU.
3. **Fallback CPU robuste**
   - Pas de crash, logs explicites (`fallback_reason`).
4. **Parité scientifique CPU/GPU**
   - Tests delta max/median sur sorties intermédiaires.
   - Seuils de tolérance documentés.

### Dépendances
- S2 (important pour intégrer proprement).

### Critères de done
- Gain de temps global sur dataset large: **>= 25–40%** (cible).
- Taux d’échec GPU hors machines incompatibles: **< 2%**.

---

## Sprint S4 — Post-process GPU ciblé
**Durée:** 1 semaine  
**Effort estimé:** 4–7 JH  
**Priorité:** P1

### Objectifs
- Accélérer les dernières étapes lourdes pixel-wise.

### Tickets
1. **Accélération blur/smoothing fond**
   - GPU path + fallback CPU.
2. **Accélération morphologie masques**
   - Dilation/erosion/ops clés en GPU si backend dispo.
3. **Feature flags fins**
   - Activation indépendante de chaque bloc GPU.
4. **Bench comparatif on/off**
   - Mesure gain réel sur grosses mosaïques.

### Dépendances
- S3 recommandé mais non bloquant.

### Critères de done
- Gain mesurable sur mosaïques >20k px.
- Qualité visuelle et métriques stables.

---

## Sprint S5 — Optimisations I/O & mémoire transverses
**Durée:** 1 semaine  
**Effort estimé:** 4–6 JH  
**Priorité:** P1

### Objectifs
- Réduire les stalls I/O et la variance entre runs.

### Tickets
1. **Préfetch et lecture séquentielle**
   - Limiter random I/O là où possible.
2. **Politique memmap adaptative**
   - Ajustement selon profil disque/hôte.
3. **Rationalisation caches temporaires**
   - Uniformiser format/chemins, nettoyer plus efficacement.

### Dépendances
- S1 pour comparaison objective.

### Critères de done
- Variance run-to-run en baisse.
- Diminution des pics de latence I/O dans les logs.

---

## Sprint S6 — Hardening release
**Durée:** 3–4 jours  
**Effort estimé:** 2–4 JH  
**Priorité:** P0 avant release

### Objectifs
- Sécuriser la livraison (cross-platform + no-regression).

### Tickets
1. **Matrice de tests release**
   - Windows/Linux, CUDA/no-CUDA, small/medium/large.
2. **Rapport perf automatique**
   - Baseline vs release candidate.
3. **Checklist de déploiement**
   - Config par défaut, flags GPU, fallback policy, logs.

### Dépendances
- S1→S5.

### Critères de done
- Aucun blocage critique.
- Rapport final validé.

---

## Dépendances & ordre recommandé
- **Ordre optimal:** S1 → S2 → S3 → S4 → S5 → S6
- **Chemin court si urgence perf:** S1 → S3 → S6 (plus risqué techniquement)

---

## KPIs globaux de succès
1. **Wall-clock global (dataset large):** -30% à -50%.
2. **Fiabilité fallback GPU:** 100% des échecs GPU convertis en CPU sans plantage.
3. **Parité qualité:** écarts dans seuils validés (delta + métriques astro).
4. **Maintenabilité:** réduction claire de la complexité du worker principal.

---

## Notes de gouvernance
- Toute optimisation GPU doit conserver un chemin CPU référence.
- Toute décision runtime GPU doit être journalisée (`requested_gpu`, `effective_gpu`, `fallback_reason`).
- Ne pas poursuivre l’optimisation ASTAP GPU: priorité faible vs coût.
