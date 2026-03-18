# followup.md

# ZeMosaic — Follow-up checklist
## Seam root-cause elimination (Classic first)

Legend:
- `[ ]` not done
- `[x]` done
- `[~]` partial
- `BLOCKED:` reason

Reference principle:
- Work on the next unchecked item.
- Patch surgically.
- Prove every claim.
- Update `memory.md` after each significant iteration.

---

## A. Discipline

- [ ] Lire `agent.md`, `followup.md`, `memory.md` avant chaque itération
- [ ] Travailler sur le prochain item non coché
- [ ] Patchs chirurgicaux uniquement
- [ ] Prouver chaque claim par logs + images + comparaison
- [ ] Mettre à jour `memory.md` à chaque itération significative
- [ ] Ne pas toucher au comportement stable `batch size = 0` / `batch size > 1`
- [ ] Ne pas élargir le scope aux autres modes tant que Classic n’est pas proprement instrumenté

---

## B. Freeze baseline and assets

- [ ] Geler le dataset de référence principal “gros dataset hétérogène”
- [ ] Geler les sorties benchmark déjà produites comme baseline
- [ ] Identifier clairement les fichiers de comparaison:
  - weighting OFF
  - weighting ON actuel
  - coverage maps
  - diagnostics V3
- [ ] Documenter dans `memory.md` quels outputs servent désormais de référence officielle

---

## C. Instrumentation — real pipeline first

### C1. Graphe réellement retenu
- [ ] Exporter le graphe final réellement retenu par le pipeline worker
- [ ] Exporter pour chaque arête retenue:
  - tile_i
  - tile_j
  - score / strength
  - overlap
  - métriques photométriques utiles
- [ ] Exporter les arêtes rejetées mais fortes
- [ ] Logger explicitement les raisons principales de rejet si disponible

### C2. Solve photométrique
- [ ] Exporter l’ancre réellement choisie
- [ ] Exporter les gains / offsets par tuile avant/après solve
- [ ] Exporter les résidus par arête après harmonisation
- [ ] Exporter un résumé global:
  - residual median
  - residual p95
  - worst edges
  - overlap_mean des paires actives

### C3. Weighting final
- [ ] Exporter les poids finaux réellement appliqués par tuile
- [ ] Logger leur source exacte:
  - `MT_NFRAMES`
  - fallback
  - autre source éventuelle
- [ ] En mode “existing master tiles”, logger:
  - nombre de tuiles avec poids valide
  - nombre de tuiles fallback=1.0
- [ ] Produire une `weighted_coverage_map`
- [ ] Produire une `winner_map` / `dominant_tile_map`

### C4. Corrélation visuelle
- [ ] Vérifier si les seams visibles suivent:
  - la winner map
  - la weighted coverage map
  - certaines arêtes à fort résidu
- [ ] Documenter conclusion courte dans `memory.md`

---

## D. Revalidation of current hypothesis

- [ ] Confirmer ou infirmer que le problème principal est:
  - graphe trop simplifié
  - homogénéisation insuffisante
  - weighting trop dominateur
  - combinaison des trois
- [ ] Éviter toute conclusion basée uniquement sur simulation externe
- [ ] Prioriser les preuves issues du pipeline réel ZeMosaic

---

## E. Graph rework experiments

### E1. Pruning
- [ ] Identifier précisément la logique actuelle de top-K / pruning utilisée sur le gros dataset
- [ ] Tester une variante moins brutale
- [ ] Tester une variante top-K adaptative selon densité locale
- [ ] Vérifier si des arêtes fortes aujourd’hui rejetées doivent être conservées

### E2. Scoring des arêtes
- [ ] Étudier un score combiné:
  - overlap
  - stabilité photométrique
  - cohérence temporelle
- [ ] Éviter qu’une arête géométriquement correcte mais photométriquement aberrante domine
- [ ] Éviter aussi de jeter trop tôt des arêtes photométriquement utiles

### E3. Cohortes temporelles
- [ ] Évaluer si le dataset doit être traité avec une conscience temporelle plus explicite
- [ ] Tester l’idée de cohortes/session groups si l’étalement temporel reste une cause forte
- [ ] Documenter GO / NO-GO sur cette piste

---

## F. Weighting V4

### F1. Politique conservatoire immédiate
- [ ] Considérer `master tile weighting = OFF` comme baseline conservatoire tant que V4 n’est pas validé
- [ ] Documenter clairement si cette position doit devenir le défaut temporaire

### F2. Design V4
- [ ] Introduire une compression forte de dynamique:
  - `sqrt`
  - `log`
  - soft-cap robuste
  - ou équivalent
- [ ] Éviter qu’une tuile “forte” domine massivement une tuile “faible”
- [ ] Réduire la domination globale au centre des tuiles
- [ ] Favoriser un plateau intérieur plus neutre
- [ ] Garder le weighting surtout utile dans les zones de couture / feather

### F3. Guardrails V4
- [ ] Ajouter un plafond automatique si les résidus photométriques d’une tuile restent élevés
- [ ] Ajouter une pénalité temporelle si une tuile est très atypique
- [ ] Tout rendre config-gated
- [ ] Ajouter toute clé persistée à `DEFAULT_CONFIG`

### F4. Telemetry V4
- [ ] Logger avant/après:
  - distribution des poids
  - min / median / max
  - nombre de tuiles plafonnées / pénalisées
  - effet sur winner map
  - effet sur résidus / seams proxy

---

## G. Validation protocol

- [ ] Comparer sur mêmes master tiles:
  - OFF
  - ON actuel
  - ON V4
- [ ] Comparer visuellement:
  - seams rectangulaires
  - fond
  - homogénéité locale
  - couleur
- [ ] Comparer quantitativement:
  - dispersion photométrique par overlap
  - résidu moyen par arête
  - residual p95
  - variation de fond par tuile
  - intensité de domination locale

---

## H. Visual seam-heal — postponed, not mainline

- [ ] Ne pas implémenter le seam-heal low-frequency avant d’avoir traité C/D/F
- [ ] Garder l’idée vivante uniquement comme finition visuelle
- [ ] Si réactivé plus tard:
  - luma-first
  - visual-only
  - OFF par défaut
  - Phase 6 / output final
  - pas de modification science/FITS

---

## I. Non-regression

- [ ] Pas de régression géométrique
- [ ] Pas de dérive couleur induite
- [ ] Pas de halos / banding / zones molles
- [ ] Pas de crash worker/GUI
- [ ] Pas de casse sur Classic / Existing master tiles
- [ ] Pas de changement indésirable de comportement batch stable

---

## J. Mission close

- [ ] Root cause prouvée de façon crédible
- [ ] Pipeline réel instrumenté
- [ ] Graph rework et/ou weighting V4 validés ou clairement rejetés
- [ ] Baseline conservatoire documentée
- [ ] `memory.md` mis à jour avec synthèse GO / NO-GO