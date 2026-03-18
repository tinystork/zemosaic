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