# followup.md

# ZeMosaic — Follow-up checklist
## Mission qualité multi-modes (normalisation / RGB / DBE)

Legend:
- `[ ]` not done
- `[x]` done
- `[~]` partial
- `BLOCKED:` reason

---

## A. Discipline mission

- [x] Lire `agent.md`, `followup.md`, `memory.md` avant toute modif
- [x] Travailler uniquement sur le prochain item non coché
- [x] Maintenir patchs chirurgicaux
- [x] Mettre à jour `memory.md` à chaque itération significative
- [x] Prouver chaque claim (logs/tests/sorties)

---

## B. Baseline / preuves existantes

### B1. Garde-fou RGB final actuel
- [x] Confirmer dans le code la désactivation temporaire de la final RGB equalization
- [x] Documenter le contexte historique (dominante verte) dans `memory.md`

### B2. Artefacts de référence
- [x] Vérifier présence/intégrité des artefacts de référence listés dans `agent.md`
- [x] Établir ces artefacts comme baseline de comparaison qualité

### B3. Cartographie inter-modes
- [x] Tableau par mode: normalisation photométrique (Y/N/partiel)
- [x] Tableau par mode: RGB equalization (Y/N/partiel)
- [x] Tableau par mode: DBE final (Y/N/partiel)
- [x] Tableau par mode: preview/stretch (Y/N/partiel)

---

## C. Q1 — Essai réactivation final RGB equalization

### C1. Implémentation contrôlée
- [x] Réactiver derrière flag explicite (pas en dur)
- [x] Valeur par défaut conservatrice (éviter surprise utilisateur)
- [x] Log clair quand activé/désactivé

### C2. Validation run réel
- [x] Run A (flag off) baseline
- [x] Run B (flag on) comparaison
- [x] Comparer dominante/couleur/histogrammes et rendu preview
- [x] Décision explicite: keep / tune / revert (tune)

---

## D. Q2 — Mode “I’m using master tiles”

### D1. Gap analysis
- [x] Identifier précisément ce qui manque vs classique
- [x] Isoler ce qui est déjà couvert par two-pass/affine

### D2. Harmonisation
- [x] Ajouter mécanisme(s) manquant(s) sans perturber two-pass/affine
- [x] Garder compatibilité sortie FITS/coverage/preview

### D3. Validation
- [x] Run comparatif avant/après
- [x] Vérifier absence de régression fonctionnelle

---

## E. Q3 — Mode SDS

### E1. Gap analysis
- [x] Cartographier différences qualitatives vs classique
- [x] Isoler ce qui relève de contraintes SDS (géométrie globale)

### E2. Harmonisation
- [x] Ajouter normalisation/équilibrage manquants compatibles SDS
- [x] Conserver les correctifs SDS déjà stabilisés (NameError, broadcast, OOM handling)

### E3. Validation
- [x] Run SDS complet sans crash
- [x] Contrôle qualité couleur/fond/cohérence

---

## F. Q4 — Mode ZeGrid (DBE prioritaire)

### F1. Confirmation du gap DBE
- [x] Prouver si DBE final est réellement absent/incomplet
- [x] Tracer le hook réel de fin de pipeline grid

### F2. Implémentation
- [x] Intégrer DBE final ZeGrid (équivalent-intention worker)
- [x] Ajouter logs explicites DBE appliqué/skippé + raison

### F3. Validation
- [x] Run ZeGrid réel complet
- [x] Vérifier sorties et absence de régression côté grid_mode

---

## G. Non-régression transversale

- [ ] Classique inchangé (smoke)
- [ ] SDS inchangé fonctionnellement (smoke)
- [ ] ZeGrid inchangé hors améliorations visées (smoke)
- [ ] Existing-master-tiles inchangé hors améliorations visées
- [ ] Tests ciblés passants

---

## H. Clôture mission

- [ ] Rapport final (gains/risques/décisions)
- [ ] Décision finale sur le garde-fou RGB equalization
- [ ] Statut final GO / NO-GO
- [ ] Synthèse durable ajoutée à `memory.md`
