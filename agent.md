# agent.md

# ZeMosaic — Mission Codex
## Harmonisation qualitative multi-modes (post-refactor)

Date: 2026-03-14
Owner: Tristan / ZeMosaic core
Mission mode: quality-first / incremental / non-régression stricte

---

## Mission objective

Élever la qualité de normalisation photométrique/couleur de tous les modes non-classiques, sans casser les flux réparés.

Objectif final:
- rapprocher **I’m using master tiles**, **SDS**, et **ZeGrid** du niveau qualitatif de la voie classique,
- en harmonisant (ou en réutilisant) les mécanismes robustes de la voie classique:
  - normalisation photométrique,
  - équilibrage RGB,
  - DBE final (quand pertinent au mode),
  - cohérence de rendu preview.

---

## Contexte à conserver explicitement

### 1) Garde-fou actuel (important)
Un garde-fou a été introduit pendant le refactor:
- dans `zemosaic_worker.py`, la **final mosaic RGB equalization** est temporairement désactivée/commentée,
- raison documentée dans le code: éviter une dominante verte quand la normalisation est déjà appliquée en amont.

Mission actuelle:
- **réactiver cette normalisation à titre d’essai contrôlé**,
- valider sur run réel,
- puis prendre une décision finale explicite (garder, ajuster, ou désactiver durablement).

### 2) Artefacts de référence produits (preuve exécution)
Ces sorties doivent rester des références de validation (existence + qualité):
- `/home/tristan/zemosaic/zemosaic/example/out/global_mosaic_wcs.fits`
- `/home/tristan/zemosaic/zemosaic/example/out/global_mosaic_wcs.json`
- `/home/tristan/zemosaic/zemosaic/example/out/mosaic_grid.fits`
- `/home/tristan/zemosaic/zemosaic/example/out/mosaic_grid_coverage.fits`
- `/home/tristan/zemosaic/zemosaic/example/out/resource_telemetry.csv`
- `/home/tristan/zemosaic/zemosaic/example/out/run_config_snapshot.json`
- `/home/tristan/zemosaic/zemosaic/example/out/zemosaic_MT0_R30.fits`
- `/home/tristan/zemosaic/zemosaic/example/out/zemosaic_MT0_R30_coverage.fits`
- `/home/tristan/zemosaic/zemosaic/example/out/zemosaic_MT0_R30_preview.png`
- `/home/tristan/zemosaic/zemosaic/example/out/zemosaic_MT14_R0.fits`
- `/home/tristan/zemosaic/zemosaic/example/out/zemosaic_MT14_R0_coverage.fits`
- `/home/tristan/zemosaic/zemosaic/example/out/zemosaic_MT14_R0_preview.png`
- `/home/tristan/zemosaic/zemosaic/example/out/zemosaic_MT14_R30.fits`
- `/home/tristan/zemosaic/zemosaic/example/out/zemosaic_MT14_R30_coverage.fits`
- `/home/tristan/zemosaic/zemosaic/example/out/zemosaic_MT14_R30_preview.png`

---

## Réalité actuelle des modes (baseline à respecter)

## 2) Mode “I’m using master tiles”
- Pas la même normalisation "brute unitaire" que le classique (normal, on part de master tiles).
- Harmonisation surtout inter-tiles (two-pass gains / affine).
- DBE présent côté worker (hook Phase 6).

## 3) Mode SDS
- Mécanismes d’harmonisation existants (inter-tiles / two-pass).
- Pipeline non équivalent au classique sur normalisation "brute unitaire".
- Ressenti qualitatif "pas normalisé pareil" plausible.
- DBE câblé côté worker.

## 4) Mode ZeGrid
- Pipeline photométrique propre au mode grid.
- Point faible probable: DBE (hook loggé mais absence d’application équivalente claire au worker classique/SDS dans `grid_mode.py`).

---

## Non-negotiable execution rules

1. **Ne pas casser les 3 voies exclusives**: classique, SDS, ZeGrid.
2. **Patchs incrémentaux, mode par mode**, jamais transversal massif en un seul lot.
3. **Aucune régression silencieuse** sur sorties FITS/coverage/preview.
4. **Toujours prouver** (logs + outputs + tests ciblés) avant de cocher.
5. **Réactivation RGB finaleq = essai contrôlé**, pas décision définitive sans run validé.
6. **DBE ZeGrid**: corriger proprement sans détourner/classiciser brutalement tout le mode.
7. **Traçabilité obligatoire** dans `memory.md` à chaque itération significative.
8. **Ne pas supprimer de garde-fou sans remplacement/justification.**

---

## Scope

### In scope
- Audit comparatif normalisation/DBE entre classique vs (master-tiles, SDS, ZeGrid)
- Réintégration contrôlée de `final mosaic RGB equalization`
- Harmonisation des pipelines non-classiques (duplication maîtrisée ou factorisation partielle)
- Traitement DBE ZeGrid si manque confirmé
- Tests et critères qualité inter-modes

### Out of scope
- Refactor architecture globale du worker
- Changements non liés à la qualité photométrique/couleur/DBE
- Altération des invariants scientifiques des stacks

---

## Mission phases

### [ ] Q0 — Baseline qualité inter-modes
- Cartographier ce qui est appliqué réellement par mode:
  - normalisation photométrique
  - équilibrage RGB
  - DBE
  - hooks preview/stretch
- Produire un tableau "appliqué / non-appliqué / partiel".

### [ ] Q1 — Réactivation contrôlée RGB finaleq (essai)
- Réactiver la final mosaic RGB equalization derrière un switch explicite.
- Lancer un run de validation contrôlé.
- Comparer avant/après (dominante, histogrammes RGB, rendu preview, stats).
- Décision explicite: conserver / retuner / revert.

### [ ] Q2 — Mode “I’m using master tiles”
- Définir l’écart exact vs classique.
- Ajouter ce qui manque (si pertinent) sans casser two-pass/affine.
- Valider sortie scientifique + rendu.

### [ ] Q3 — Mode SDS
- Identifier les manques qualitatifs vs classique.
- Introduire harmonisation additionnelle compatible SDS.
- Vérifier stabilité mémoire/perf déjà restaurée.

### [ ] Q4 — Mode ZeGrid (priorité DBE)
- Confirmer techniquement le gap DBE.
- Implémenter un DBE final équivalent-intention (pas forcément copie brute).
- Valider non-régression ZeGrid + compat sorties.

### [ ] Q5 — Validation finale et décision
- Campagne de runs comparatifs multi-modes.
- Rapport final:
  - gains qualitatifs
  - impacts perf
  - risques résiduels
  - décisions finales sur garde-fous (dont RGB finaleq)

---

## Release gate (mission)

Mission close only if:
1. Régression fonctionnelle = 0 sur classique/SDS/ZeGrid/master-tiles.
2. Pipeline qualité harmonisé documenté par mode.
3. Position finale actée sur final RGB equalization.
4. DBE ZeGrid clarifié (appliqué ou limitation assumée et documentée).
5. Preuves complètes archivées dans `memory.md`.
