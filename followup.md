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

---

## K. 2026-03-20 — Proto V4 / Pruning runtime / Discipline anti-régression

### K1. Proto V4 (baseline de test)
- [x] Ajouter clés config persistées `tile_weight_v4_*` dans `DEFAULT_CONFIG`
- [x] Brancher proto V4 en mode config-gated (OFF par défaut)
- [ ] Ajouter pénalité résidu photométrique par tuile (guardrail qualité)
- [ ] Ajouter pénalité temporelle optionnelle

### K2. Pruning runtime configurable
- [x] Ajouter `intertile_prune_k` (runtime)
- [x] Ajouter `intertile_prune_weight_mode` (`area|strength|hybrid`)
- [x] Brancher ces paramètres dans le calcul intertile réel
- [ ] Vérifier logs terrain: `Pair pruning summary ... K=... mode=...`

### K3. RUN A/B mini-dataset
- [x] Préparer RUN A (V4 OFF explicite + prune explicite)
- [ ] Exécuter RUN A et archiver sortie/logs/crops
- [ ] Préparer RUN B (V4 ON, autres paramètres constants)
- [ ] Exécuter RUN B et comparer à RUN A

### K4. Discipline anti-régression (obligatoire)
- [ ] Pour chaque patch Classic, lister la zone partagée touchée
- [ ] Refaire un smoke check ZeGrid + SDS après patch Classic
- [ ] Reporter preuve dans `memory.md` (compile/tests/run + risque résiduel)


---

## L. Mission supplémentaire — Fiabiliser le Quality Gate

Objectif: empêcher qu’un run « techniquement réussi » sorte un résultat visuellement/scientifiquement invalide (master tile corrompue, dynamique aberrante, stats incohérentes) sans alerte bloquante.

### L1. Définir les signaux de corruption / dérive
- [ ] Définir des seuils robustes par tuile:
  - médiane canal (R/G/B)
  - MAD canal
  - ratio max/median
  - fraction de pixels nuls / NaN
- [ ] Ajouter détection d’outlier robuste (IQR/MAD) sur la distribution inter-tiles
- [ ] Marquer explicitement les tuiles suspectes dans les logs (ex: `QUALITY_GATE_TILE_OUTLIER`)

### L2. Gates pré-assemblage (hard fail configurable)
- [ ] Ajouter un quality gate avant Phase 5 (reproject/coadd)
- [ ] Si tuile extrême détectée:
  - mode `warn`: continuer + alerte forte
  - mode `fail`: stopper run proprement avec message actionnable
- [ ] Exposer en config:
  - `quality_gate_enabled`
  - `quality_gate_mode` (`warn|fail`)
  - `quality_gate_tile_sigma_threshold` (ou équivalent robuste)

### L3. Gates post-assemblage (sanity mosaic)
- [ ] Ajouter contrôles globaux après assemblage:
  - plage dynamique globale
  - fraction NaN
  - cohérence inter-canaux (ratios robustes)
- [ ] Lever alerte bloquante si profil incompatible avec baseline dataset
- [ ] Exporter un résumé machine-readable (`quality_gate_report.json`)

### L4. Existing master tiles — durcissement spécifique
- [ ] Vérifier cohérence stricte entre index tuiles attendues et présentes
- [ ] Refuser silencieux impossible: si “trou + outlier extrême”, alerte explicite obligatoire
- [ ] Logger source des tiles et stats minimales lors du chargement (`tile_id -> median/mad/min/max`)

### L5. UX opérateur / observabilité
- [ ] Ajouter messages GUI clairs:
  - nombre de tuiles OK
  - nombre suspectes
  - action recommandée
- [ ] Ajouter résumé fin de run:
  - `QUALITY_GATE_STATUS=PASS|WARN|FAIL`
  - liste courte des tuiles concernées

### L6. Validation et non-régression
- [ ] Cas test nominal: dataset propre => PASS sans faux positif
- [ ] Cas test corruption connue (master tile saturée) => WARN/FAIL attendu
- [ ] Vérifier non-régression Classic / ZeGrid / SDS (smoke minimal)
- [ ] Documenter les seuils retenus et le rationnel dans `memory.md`

### L0. Alignement 2026-03-23 (constats validés)
- [x] Confirmer que le quality gate existant est bien câblé (GUI -> config -> worker -> accept/reject)
- [x] Confirmer que le quality gate actuel s'applique en création de master tiles (Phase 3)
- [x] Confirmer que le mode `use_existing_master_tiles=true` n'est pas actuellement protégé par ce gate
- [x] Confirmer sur logs/config terrain (`NGC6888_2`, `NGC6888_3`) que `quality_gate_enabled=false`
- [x] Confirmer root cause terrain: master tile outlier (`master_tile_125`) peut contaminer tout le coadd en existing-master mode

### L7. Recentrage mission (quality gate au service du pivot principal)
- [x] Ajouter un **pré-check quality gate dédié existing master tiles** avant Phase 5
- [x] Définir politique claire en existing mode:
  - `warn` = continuer mais signaler tuiles suspectes explicitement
  - `fail` = arrêt propre avant coadd si outlier critique
- [x] Garantir qu'un outlier photométrique massif (type `master_tile_125`) ne passe plus silencieusement
- [x] Journaliser un résumé de validation des masters existantes (count ok/suspect/reject) dans les logs run
- [ ] Garder le scope chirurgical: pas de refactor large, pas de dérive hors mission graph/weighting
