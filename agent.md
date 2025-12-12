# Mission — Debug ciblé “dominante verte” en mode Classic (sans refactor)

Objectif : identifier *le premier endroit* où la dérive couleur apparaît dans le pipeline Classic
(P3 → P4 → P5 → P6/7) en ajoutant des logs DEBUG très ciblés.

Contraintes :
- NE PAS modifier l’algorithme de stacking/fusion/export.
- NE PAS toucher au SDS / Grid mode (sauf si besoin strictement pour la propagation du niveau de log).
- Ajouter uniquement du logging et le câblage “Logging level” du GUI Qt vers le worker.
- Logs uniquement actifs quand le worker est en DEBUG (ou quand un flag debug est activé).

---

## Contexte (preuves)
- Phase 3 : on voit déjà des logs `[DBG_RGB] P3_pre_stack_core` puis `P3_post_poststack_rgb_eq` avec ratios 1.0 → P3 OK.
- Le run Classic legacy affiche : “Worker logging level set to INFO” → le choix de niveau dans le GUI n’atteint pas le worker dans ce chemin.

---

## Changements demandés

### 1) Propager le “Logging level” du GUI Qt vers le worker (vital)
Fichier : `zemosaic_gui_qt.py`

- S’assurer que le combo “Logging level” propose au minimum :
  - `Info` → worker level `INFO`
  - `Debug` → worker level `DEBUG`

- Lors du lancement du worker (construction des paramètres de run), injecter un champ explicite :
  - `worker_logging_level` = `"DEBUG"` ou `"INFO"` selon le choix
  - (ou `logging_level`, mais utiliser le nom déjà attendu côté worker si existant)

But : quand je mets Debug dans le GUI, le fichier `zemosaic_worker.log` doit contenir des lignes DEBUG.

### 2) Dans le worker : respecter le niveau de log demandé (notamment Classic legacy)
Fichier : `zemosaic_worker.py`

- Au tout début du run (et aussi au début du chemin “classic legacy”), lire le paramètre reçu :
  - `worker_logging_level` (prioritaire)
  - fallback sur config existante si déjà en place
- Appliquer :
  - `logger.setLevel(logging.DEBUG/INFO)`
  - s’assurer que les handlers suivent (setLevel sur handler si nécessaire)

- Ajouter un log INFO unique confirmant le niveau choisi :
  - `Worker logging level set to DEBUG` ou `INFO`

### 3) Logs DEBUG ciblés par phase (P3/P4/P5/P6-7)

#### A) Phase 3 / 3.x — Baseline “tuile saine”
Objectif : figer noir sur blanc que la couleur est saine avant d’assembler la mosaïque.

À logger (DEBUG) **avant et après** :
- `stack_core` (déjà partiellement loggé via `_dbg_rgb_stats` → garder, mais harmoniser les labels)
- `_poststack_rgb_equalization` si appelé

Mesures requises :
- min / mean / median par canal (R,G,B)
- ratios `G/R` et `G/B`
- idem **sur pixels valides uniquement** (si un masque existe à ce stade, sinon valid=1.0)

=> Logs très courts : 2 à 4 lignes par tuile max.

#### B) Phase 4 / 4.x — Assemblage mosaïque (ZONE CRITIQUE #1)
Objectif : détecter si la dérive apparaît lors de la fusion + coverage + propagation NaN/alpha.

Ajouter des logs DEBUG :
- juste AVANT la fusion finale (ou début de la phase 4)
- juste APRÈS la mosaïque assemblée (data + coverage prêts)

Mesures :
1) stats RGB “brutes” (comme P3)
2) stats RGB **pixels valides uniquement**
   - “valide” = `coverage > 0` (ou masque équivalent)
3) moyenne RGB **pondérée par coverage**
   - calcul : mean_weighted[c] = sum(data[c] * cov) / sum(cov) sur pixels cov>0
4) ratios `G/R` et `G/B` sur (2) et (3)

Important :
- Ne pas logguer à chaque tile (trop bruyant). Uniquement “pré-fusion” et “post-fusion”.
- Si la phase 4 assemble par étapes (super-tiles), logguer seulement au niveau final.

#### C) Phase 5 — Post-processing global (ZONE CRITIQUE #2)
Objectif : vérifier si un traitement global “classic-only” crée la dominante verte.

Ajouter logs DEBUG :
- début phase 5 : stats mosaïque (brute + valid-only + weighted)
- après chaque étape “suspecte” si présente :
  - `_apply_final_mosaic_rgb_equalization` (si appelé)
  - black point equalization / scaling / normalization historique
  - toute correction per-channel

Si une égalisation RGB est appliquée :
- logguer :
  - target (valeur cible)
  - gains/facteurs appliqués par canal
  - (si offsets) offsets par canal

#### D) Phase 6–7 — Export/clamp (secondaire)
Ajouter logs DEBUG uniques :
- dtype entrée
- clamp min/max par canal avant conversion
- dtype sortie
- mention explicite si un stretch automatique est appliqué avant PNG

### 4) Utilitaire de stats (réutiliser l’existant)
- Il existe déjà `_dbg_rgb_stats` dans `zemosaic_worker.py`.
- L’étendre proprement (sans casser appels existants) pour accepter :
  - `mask_valid: np.ndarray | None` (H,W bool) OU `coverage: np.ndarray | None`
- Implémenter dans la fonction :
  - stats globales
  - stats sur valid-only (si mask fourni)
  - weighted mean (si coverage fourni)

⚠️ Performance :
- Ne faire ces calculs QUE si `logger.isEnabledFor(DEBUG)`.

---

## Tests / Validation

1) Dans le GUI Qt, sélectionner “Debug” puis lancer un run Classic.
   - Attendu : `zemosaic_worker.log` contient des lignes DEBUG.
   - Attendu : une ligne INFO confirme “Worker logging level set to DEBUG”.

2) Comparer Classic vs SDS sur un même dataset :
   - Relever les logs P4/P5 :
   - Identifier le *premier label* où `ratio_G_R` ou `ratio_G_B` diverge significativement.

3) S’assurer que :
- aucun changement d’image (hors logs)
- SDS/Grid inchangés fonctionnellement
- pas de spam log (quelques lignes par phase seulement)

---

## Fichiers concernés
- `zemosaic_gui_qt.py`
- `zemosaic_worker.py`
(éventuellement `zemosaic_config.py` seulement si nécessaire pour stocker la préférence de niveau de log)

