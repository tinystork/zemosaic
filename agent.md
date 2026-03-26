# agent.md

## Existing content

# agent.md

# ZeMosaic — Mission Codex / Junior
## Seam root-cause elimination (Classic first, all modes preserved)

Date: 2026-03-17
Owner: Tristan / ZeMosaic core
Mission mode: diagnostic-first / quality-first / surgical / strict non-regression

---

## Mission objective

Supprimer les seams visibles dans ZeMosaic en traitant **d’abord leurs causes structurelles**, puis seulement les artefacts résiduels de rendu.

La mission n’est plus pilotée principalement par:
- le tuning visuel du blend,
- ou un futur pass cosmétique de type seam-heal.

La mission est désormais pilotée par ce triptyque:

1. **graphe photométrique réellement retenu**
2. **homogénéisation photométrique réellement obtenue**
3. **weighting final réellement appliqué**

But final:
- obtenir une mosaïque visuellement beaucoup plus homogène,
- conserver une voie FITS scientifiquement conservatrice,
- éviter qu’un weighting trop affirmatif “imprime” la géométrie des master tiles dans le rendu final,
- traiter d’abord le mode **Classic**, sans casser les autres modes.

---

## Core diagnosis (current state)

Constats consolidés à ce stade:

- Le problème principal ne semble **pas** être un manque global de recouvrement géométrique entre master tiles.
- Le graphe brut de recouvrement est dense, mais le graphe réellement exploité après pruning est nettement plus maigre.
- Les master tiles sont hétérogènes:
  - couverture / nombre de raws,
  - qualité locale / SNR,
  - sessions temporelles éloignées,
  - résidus photométriques parfois encore élevés après solve.
- Le weighting final actuel peut amplifier cette hétérogénéité au lieu de la dissoudre.
- Les seams visibles semblent naître **en amont** du polish visuel:
  - solve photométrique trop simplifié ou trop agressivement pruné,
  - tuiles insuffisamment homogénéisées,
  - weighting final trop dominateur,
  - puis seulement ensuite rendu des coutures.

Conclusion de travail:
- **Visual seam-heal** reste une bonne idée, mais seulement comme finition.
- La priorité doit porter sur:
  1. diagnostic réel du pipeline,
  2. graph rework,
  3. weighting V4.

---

## Non-negotiable execution rules

1. **Mode Classic d’abord.**
   - Ne pas lancer de refactor transversal brutal SDS / ZeGrid / Existing master tiles.
2. **Patchs chirurgicaux uniquement.**
   - Pas de gros ménage architectural.
3. **Prouver chaque claim.**
   - Logs, exports, images diag, comparaisons avant/après.
4. **Toujours séparer diagnostic et correction produit.**
   - D’abord voir ce que fait réellement le pipeline, ensuite corriger.
5. **Conserver la séparation science / visuel.**
   - FITS: conservateur.
   - rendu visuel: plus tolérant si nécessaire.
6. **Tout nouveau levier sensible doit être config-gated.**
   - Defaults conservateurs.
7. **Aucune régression silencieuse.**
   - outputs, coverage, headers, modes, GUI.
8. **Mettre à jour `memory.md` à chaque itération significative.**
9. **Respecter le comportement connu stable.**
   - Ne pas toucher au comportement `batch size = 0` et `batch size > 1` qui fonctionne bien.
10. **Ne pas dériver vers du tuning visuel prématuré.**
    - `intertile_affine_blend` et futur seam-heal ne sont plus les axes principaux tant que la cause amont n’est pas clarifiée.

---

## Scope

### In scope
- Instrumentation du pipeline réel de solve / intertile / final assembly
- Export du graphe réellement retenu
- Export des arêtes rejetées mais potentiellement fortes
- Export des poids réellement appliqués aux master tiles
- Export de l’ancre réellement utilisée
- Export des gains / offsets / résidus par arête
- Diagnostics de domination de tuile (winner map / weighted coverage)
- Rework du graphe photométrique
- Rework du weighting final (V4)
- Validation comparative sur datasets réels hétérogènes

### Out of scope
- Réécriture globale du worker
- Beautification destructrice du signal
- Généralisation immédiate à tous les modes sans preuve sur Classic
- Finition visuelle lourde avant résolution des causes amont

---

## Priority order

### P0 — Instrument real pipeline
Comprendre exactement ce que ZeMosaic fait réellement sur un gros dataset hétérogène.

### P1 — Photometric graph rework
Réduire la perte d’information utile causée par le pruning actuel.

### P2 — Weighting V4
Faire du weighting un outil de couture, pas de domination globale.

### P3 — Visual seam finishing
Seulement après P0/P1/P2, pour les résidus basse fréquence restants.

---

## Mission phases

### [ ] A — Baseline and diagnostics freeze
- Geler le dataset de référence principal.
- Geler les sorties de benchmark déjà obtenues.
- Définir clairement les comparaisons à refaire:
  - weighting OFF
  - weighting ON actuel
  - future weighting V4

### [ ] B — Real pipeline instrumentation
- Exporter le graphe réellement retenu par le pipeline.
- Exporter les arêtes fortes rejetées.
- Exporter l’ancre réellement choisie.
- Exporter les gains / offsets photométriques avant/après solve.
- Exporter les résidus par arête.
- Exporter les poids finaux réellement appliqués par tuile.
- Produire une winner map / dominant-tile map.
- Produire une weighted-coverage map.
- Journaliser la source exacte des poids en mode “existing master tiles”.

### [ ] C — Root-cause proof
- Prouver où naissent réellement les seams:
  - solve insuffisant,
  - pruning trop agressif,
  - weighting trop fort,
  - combinaison des trois.
- Corréler:
  - seams visibles,
  - winner map,
  - weighted coverage,
  - résidus photométriques,
  - structure des tuiles.

### [ ] D — Photometric graph rework
- Tester un pruning moins brutal.
- Étudier un top-K adaptatif selon densité locale.
- Étudier un score d’arête combinant:
  - overlap,
  - stabilité photométrique,
  - cohérence temporelle.
- Étudier une logique par cohortes temporelles si utile.

### [ ] E — Weighting V4
- Conserver `master tile weighting = OFF` comme profil conservatoire tant que V4 n’est pas validé.
- Introduire une compression forte de dynamique (`sqrt`, `log`, soft-cap ou équivalent robuste).
- Réduire la domination au centre des tuiles.
- Favoriser un plateau intérieur quasi neutre et un weighting surtout utile en couture.
- Ajouter cap / pénalité si résidu photométrique trop élevé.
- Ajouter pénalité d’hétérogénéité temporelle si nécessaire.

### [ ] F — Validation
- Comparer sur mêmes master tiles:
  - OFF
  - ON actuel
  - ON V4
- Vérifier:
  - réduction des seams,
  - réduction de l’empreinte rectangulaire des tiles,
  - pas de dérive couleur,
  - pas de régression géométrique,
  - pas de halos / banding / zones molles.

### [ ] G — Visual seam finishing (only after A/B/C/D/E)
- Réévaluer le futur pass low-frequency seam-heal.
- Le garder explicitement visuel, optionnel, OFF par défaut.
- L’utiliser uniquement pour polir les résidus restants, pas pour masquer un solve/weighting défaillant.

---

## Release gate (mission)

Mission close only if:

1. Les seams ont diminué de manière visible sur dataset de référence.
2. L’empreinte géométrique des master tiles n’est plus fortement imprimée dans le rendu final.
3. Le pipeline réel est instrumenté de manière exploitable pour debug futur.
4. Le weighting V4 apporte un gain réel ou, à défaut, le choix OFF par défaut est documenté et justifié.
5. Aucun mode n’a subi de régression fonctionnelle majeure.
6. `memory.md` conserve un historique clair du pivot, des preuves et des décisions.

---

## Product stance (important)

Tant que le pipeline réel n’est pas suffisamment instrumenté et que le weighting V4 n’est pas validé:

- ne pas considérer un futur seam-heal comme la réponse principale,
- ne pas surinvestir dans le tuning cosmétique,
- considérer que la suppression durable des seams passe d’abord par:
  - un meilleur graphe photométrique,
  - une meilleure homogénéisation,
  - un weighting final moins dominateur.

---

## Addendum 2026-03-20 — Proto V4 + prudence régression Classic

### Avancement concret
- Un proto **Weighting V4 config-gated** a été branché (compression + bornes), sans activation par défaut.
- Le pruning intertile est désormais **pilotable par config** (`intertile_prune_k`, `intertile_prune_weight_mode`) au lieu d'un seul K figé.
- Objectif maintenu: comparer OFF / ON actuel / ON V4 sur même mini-dataset avant toute décision de défaut produit.

### Guardrail majeur (hérité des incidents de régression)
Le code de la voie Classic est **fortement intriqué** avec d'autres branches de pipeline.
Toute modification en Classic doit être considérée à risque de casse collatérale si non isolée.

Règles opératoires renforcées:
1. Avant patch: identifier explicitement la/les fonctions partagées touchées.
2. Pendant patch: privilégier les flags config-gated et les defaults conservateurs.
3. Après patch: vérifier au minimum non-régression rapide sur Classic / ZeGrid / SDS.
4. Traçabilité: documenter systématiquement dans `memory.md` la zone touchée + risque + preuve compile/test/run.

### Position mission
- Continuer le pivot root-cause (graph + weighting réel),
- mais avec discipline stricte de non-régression multi-voies,
- et sans multiplier les sources d'info hors `memory.md` (source de vérité opérationnelle).


## Imported from agent_prepivot_20260317.md

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


---

## Addendum mission — 2026-03-16 (réévaluation visuelle seams)

### Constat consolidé (log lourd, machine plus puissante)
- Les seams restent pilotés par des deltas de fond locaux élevés sur certaines jonctions (preuve: `TwoPassWorst abs_delta_med` très élevés), malgré des gains globaux proches de 1.
- Les overlaps observés sont souvent modestes (~3–8%), ce qui limite la robustesse de la correction locale.
- Le patchwork perçu provient davantage d'un résidu basse fréquence inter-tuiles que d'un simple problème de stretch preview.

### Décision produit (priorité visuelle assumée)
- **Oui, amélioration visuelle nette possible** au prix d'une baisse de pureté scientifique sur la sortie de visualisation.
- Conserver la séparation stricte:
  - FITS: voie conservatrice/scientifique.
  - PNG/rendu visuel: voie "visual-first" plus tolérante et plus lissante.

### Profil recommandé "VISUAL_SEAMLESS_v1" (proposition réévaluée)
> Ajustement de la proposition précédente: `intertile_overlap_min` est maintenu à `0.05` (au lieu de 0.10) pour éviter de perdre trop de contraintes utiles sur ce dataset.

Paramètres cibles:
- `poststack_equalize_rgb = false`
- `intertile_affine_blend = 0.40`
- `intertile_recenter_clip = [0.96, 1.04]`
- `intertile_overlap_min = 0.05`
- `intertile_robust_clip_sigma = 2.0`
- `apply_radial_weight = true`
- `radial_feather_fraction = 0.94`
- `radial_shape_power = 2.6`
- `final_mosaic_dbe_enabled = true`
- `final_mosaic_dbe_strength = "normal"`
- `final_mosaic_dbe_smoothing = 0.75`
- `final_mosaic_dbe_sample_step = 20`
- `final_mosaic_dbe_obj_dilate_px = 4`
- `preview_png_apply_wb = false`
- `preview_png_p_low = 0.40`
- `preview_png_p_high = 99.93`
- `preview_png_asinh_a = 0.14`

### Nouvelle proposition d'architecture (à traiter plus tard)
- Ajouter un **pass optionnel "seam-heal low-frequency"** (preview/rendu visuel):
  - détection des zones de couture,
  - correction locale du fond à basse fréquence uniquement,
  - diffusion douce de la correction pour éviter halos/banding.
- Garder ce pass **désactivé par défaut** côté science, activable dans un preset visuel.


---

## Addendum 2026-03-26 — Pivot exécution: normalisation globale inter‑tuiles avant reprojection

### Décision
Le prochain levier principal pour supprimer les seams est désormais explicite:

1. **Normaliser photométriquement les tuiles entre elles avant reprojection**,
2. **résoudre globalement sur l’ensemble des overlaps**,
3. **conserver le tile weighting en aval pour le blend**,
4. **ajouter un feather multibande léger uniquement si nécessaire**.

Le weighting seul ne doit plus être considéré comme un mécanisme suffisant de correction photométrique.

### Modèle cible
Pour chaque tuile `t`:

`I_corr_t = a_t * I_t + b_t`

- V1 robuste: `b_t` seulement (offset-only)
- V2: `a_t + b_t` (gain + offset) avec garde-fous

### Principe de solve
- Les contraintes sont extraites sur zones de recouvrement `(i,j)` via stats robustes.
- Le solve est **global** (pas pairwise isolé), avec ancre fixée:
  - `a_ref = 1`
  - `b_ref = 0`
- Régularisation légère pour éviter dérives inutiles (notamment sur `a_t`).

### Ordre pipeline imposé
1. Estimation contraintes overlaps
2. Solve global photométrique
3. Application des corrections par tuile (`a,b`)
4. Reprojection géométrique
5. Tile weighting / blend final
6. Feather multibande léger (optionnel, visuel, prudent)

### Guardrails
- Config-gated, defaults conservateurs.
- Clamp sur `a_t` et `b_t` pour éviter sur-corrections.
- Rejet robuste des outliers (saturations, étoiles extrêmes, bords instables).
- Pas d’impact silencieux sur sorties science/FITS.

### Critères de validation mission
- baisse mesurable des deltas de fond sur overlaps,
- baisse des seams visibles aux frontières,
- stabilité photométrique des étoiles communes,
- pas de régression géométrique / couleur / modes.

