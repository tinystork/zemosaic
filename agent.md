# agent.md

# ZeMosaic — Mission Codex
## Seamless Mosaic & Viewer Preview Quality (all modes)

Date: 2026-03-15
Owner: Tristan / ZeMosaic core
Mission mode: quality-first / surgical / strict non-regression

---

## Mission objective

Atteindre un rendu final visuellement homogène et propre dans tous les modes en ciblant 3 défauts majeurs encore présents:

1. **Lignes de couture inter-tuiles visibles** (parfois très visibles)
2. **PNG viewer final trop stretché / mal tonemappé** (coeurs brûlés ou rendu trop sombre selon preset)
3. **`poststack_equalize_rgb` instable** (sur-correction chromatique sur certains datasets, drift R/B)

Objectif final:
- produire un rendu **seamless** (ou quasi seamless) inter-tuiles,
- conserver la fidélité scientifique des FITS,
- améliorer le rendu PNG viewer sans sur-stretch,
- garantir la cohérence inter-modes: **Classique / Existing master tiles / SDS / ZeGrid**.

---

## Context to preserve

- Les pipelines sont maintenant fonctionnels dans tous les modes (outputs générés correctement).
- La priorité passe de la stabilité fonctionnelle à la **qualité visuelle finale**.
- Les précédentes améliorations DBE/RGB/normalisation doivent être conservées (pas de rollback implicite).
- Le PNG viewer est un produit de visualisation: son amélioration ne doit pas altérer les sorties FITS scientifiques.

---

## Non-negotiable execution rules

1. **Ne pas casser les voies exclusives**: Classique, SDS, ZeGrid, Existing-master-tiles.
2. **Patchs incrémentaux, un problème à la fois** (coutures puis stretch viewer).
3. **Aucune régression silencieuse** sur FITS/coverage/output structure.
4. **Toujours prouver** les claims (logs, captures, stats, comparaison avant/après).
5. **Ne pas forcer une recette unique brute** si un mode nécessite un tuning dédié.
6. **Séparer science vs esthétique**: FITS intacts, viewer PNG optimisé à part.
7. **Traçabilité obligatoire** dans `memory.md` à chaque itération significative.
8. **Garder des switches config** pour les nouveaux leviers sensibles (defaults conservateurs).

---

## Scope

### In scope
- Audit des mécanismes de fusion inter-tuiles (weights, feather, overlap normalization, blending)
- Audit des causes visuelles de seams par mode
- Ajustements du blending/normalisation locale pour réduire les coutures
- Audit du pipeline de génération preview PNG (stretch, clip, black/white points, saturation)
- Nouvelle stratégie de stretch viewer plus robuste (moins brûlée, fond plus propre)
- Validation comparative inter-modes

### Out of scope
- Réécriture globale de l’architecture worker
- Changement des invariants scientifiques des FITS
- “Beautification” extrême destructrice de signal

---

## Mission phases

### [ ] S0 — Baseline visuelle & métriques
- Constituer un jeu de référence avant/après (au moins 1 run par mode si possible, sinon priorité ZeGrid + mode classique).
- Définir des métriques simples et traçables:
  - contraste des seams en zone overlap (delta médiane/gradient de frontière),
  - clipping hautes lumières sur PNG,
  - niveau de fond (stabilité + propreté perceptuelle).
- Capturer baseline (images + logs + config snapshot).

### [ ] S1 — Audit couture inter-tuiles
- Cartographier les mécanismes existants par mode:
  - pondération, feather, overlap blending, recenter photométrique, normalisation locale.
- Identifier où les seams naissent réellement:
  - mismatch photométrique local,
  - transitions de poids trop abruptes,
  - manque de compensation locale de fond,
  - différences de stack tile-à-tile.
- Produire une table "cause probable / preuve / mode impacté".

### [ ] S2 — Correctifs seamless (priorité couture)
- Implémenter des corrections progressives et config-gated:
  - adoucir transitions overlap (feather/weight profile),
  - harmonisation locale inter-tuiles (offset/gain robuste),
  - garde-fous anti-surcorrection.
- Commencer par le mode le plus touché, puis généraliser prudemment.
- Journaliser précisément les nouveaux paramètres.

### [ ] S3 — Audit stretch PNG viewer
- Tracer le pipeline exact de génération preview PNG (où et comment le stretch est appliqué).
- Confirmer les causes du rendu trop agressif:
  - percentiles trop extrêmes,
  - black-point/white-point mal bornés,
  - gamma/saturation inadaptés,
  - masquage NaN/alpha influençant le rendu.

### [ ] S4 — Correctifs viewer (esthétique contrôlée)
- Introduire un stretch viewer plus équilibré (moins de blancs brûlés, fond mieux tenu).
- Prévoir presets/toggles si nécessaire (conservateur par défaut).
- Garantir: pas d’impact sur FITS scientifiques.

### [ ] S5 — Validation inter-modes & non-régression
- Runs comparatifs avant/après sur modes clés.
- Vérifier:
  - réduction visible des coutures,
  - PNG viewer plus naturel,
  - absence de régression fonctionnelle/scientifique.
- Documenter limites résiduelles (si certaines coutures restent sur cas extrêmes).

### [ ] S5bis — Assainissement conceptuel `poststack_equalize_rgb`
- Prouver et documenter la cause de drift:
  - égalisation par médianes globales par sous-stack sans masque “fond/objets”,
  - sensibilité aux couvertures partielles et gradients non homogènes.
- Redéfinir l’algorithme pour un comportement conservateur:
  - estimation des gains sur masque robuste (fond valide, exclusion objets brillants),
  - clip gains serré (ex. `[0.95, 1.05]` par défaut),
  - seuil minimum de fiabilité (samples/overlap), sinon no-op.
- Règle produit: **OFF par défaut** tant que la version robuste n’est pas validée terrain.

### [ ] S6 — Clôture mission
- Rapport final:
  - gains visuels mesurés + ressenti,
  - impacts perf/mémoire,
  - nouveaux paramètres et defaults,
  - recommandations d’usage terrain.
- Décision finale GO / NO-GO production.

---

## Release gate (mission)

Mission close only if:
1. Les coutures inter-tuiles sont significativement atténuées sur les cas de référence.
2. Le PNG viewer n’est plus sur-stretché (moins de blancs brûlés, fond plus propre).
3. Aucun mode n’a subi de régression fonctionnelle majeure.
4. Les nouveaux réglages sensibles sont documentés et pilotables par config/GUI si pertinent.
5. `poststack_equalize_rgb` est soit robustifié et validé, soit maintenu OFF par défaut avec justification documentée.
6. `memory.md` conserve un historique clair des corrections et variables branchées.
