# agent.md

## Mission
Éliminer les **master tiles pathologiques** qui polluent la mosaïque finale en créant des **cadres visibles** (bords brillants/parasites), **sans ajouter de nouvelle couche** de traitement : on **recycle** le mécanisme existant **“Master tile quality gate”** (ZeQualityMT).

## Constat (symptômes)
- Sur la mosaïque finale, on distingue clairement des rectangles/cadres correspondant à certaines master tiles (ex: 154/158/159/161/162/166/169/176/181/182/183/185…).
- Plusieurs master tiles sont :
  - soit **dégénérées** (ex: 32×32×3, CRPIX aberrants),
  - soit **quasi vides** avec **bords contenant des valeurs extrêmes** (ex: ~1e35), tout en étant **opaques/valides** (ALPHA ≈ 255 partout).  
  => Même si “l’intérieur” est noir/0 ou NaN, les **bords** contribuent au coadd et deviennent des **cadres** très visibles.

## Hypothèses validées
- Dans `zemosaic_worker.log`, on voit des lignes `MT_PIPELINE: ... quality_gate=False` : sur ce run, le gate est **désactivé** (donc aucun rejet n’est possible tant qu’il reste OFF).
- Le problème n’est **pas** principalement lié à un fallback incremental dans l’assemblage final (pas de trace claire dans le log du run).  
- Un seuil “>40% NaN” seul n’est **pas** adapté, car ces tuiles peuvent être **finies** (pas NaN) et “valides” via ALPHA, mais **corrompues** par des outliers énormes sur les bords.

## Objectif
Faire en sorte que le **quality gate** rejette automatiquement ces tuiles toxiques **avant** l’assemblage final (reproject/coadd), en détectant :
1) les tuiles **trop petites/dégénérées** (ex: 32×32),
2) les tuiles dont les **bords explosent** (edge blow-up) par valeurs extrêmes par rapport au cœur,
3) (optionnel) les tuiles trop peu “utiles” après masque (valid fraction très faible).

## Contraintes (anti-régression)
- Ne pas modifier le pipeline d’assemblage final (reproject/coadd), ni ajouter un “post-mask” supplémentaire ailleurs.
- Ne pas modifier l’UI (pas de nouveaux sliders/checkbox). On réutilise les réglages existants du gate :  
  - Threshold, Edge band (px), K-sigma, Erode (px)
- Ne pas casser les modes : **SDS**, **Grid mode**, **“I’m using master tiles”**. Pour ce dernier, si l’utilisateur charge des master tiles existantes et si `quality_gate_enabled=True`, le gate doit être appliqué au moment de constituer `valid_master_tiles_for_assembly` (pas seulement pendant la création des master tiles). Il faut appliquer exactement la même évaluation avant de construire la liste `valid_master_tiles_for_assembly` (sans toucher à reproject/coadd).
- Comportement inchangé si **quality gate désactivé**.
- Garder `zequalityMT.py` **standalone** (il est importé par ZeMosaic mais aussi exécutable seul).

## Stratégie technique (recycler le gate existant)
### A) Point critique : NBR/score ne suffisent pas → “hard reject” nécessaire
Dans `zequalityMT.py`, le score est une somme pondérée “soft” (NBR pèse ~0.10–0.12). Même un NBR=1.0 reste souvent **sous** le seuil (0.48) si le reste est “bon”.  
=> Pour éviter toute régression *et* garantir l’exclusion des cas catastrophiques, introduire un **hard reject** interne (sans UI) pour :
- `small_dim` (tuile dégénérée),
- `edge_blowup` (valeurs extrêmes concentrées sur le bord).

Implémentation attendue :
- Dans `zequalityMT.py:quality_metrics`, détecter ces cas et exposer des métriques explicites (ex: `core_p99`, `edge_p99`, `edge_ratio`, `hard_reject`).
- Dans la décision d’acceptation :
  - `zequalityMT.py:run_cli` (mode standalone)
  - `zemosaic_worker.py:_evaluate_quality_gate_metrics` (pipeline ZeMosaic)
  un `hard_reject=1` doit **forcer** `accepted=False` **quel que soit** le threshold.
- Bloquer toute “accept override” sur ces cas (défense en profondeur) :
  - `zequalityMT.py:_accept_override`
  - `zemosaic_worker.py:_zequality_accept_override` (copie de la logique)

### A-bis) Contrat des métriques (pour communication zequalityMT ↔ worker)
Pour éviter tout mismatch, le dictionnaire de métriques retourné par `zequalityMT.py` doit formaliser les clés suivantes. **Ne pas renommer ces clés**, car elles forment un contrat strict entre `zequalityMT.py` (le CLI standalone) et `zemosaic_worker.py` (l'intégrateur) ; toute désynchronisation casserait le mécanisme de rejet.
- `hard_reject`: `bool` (ou `int` 0/1) — Si `True`, le rejet est forcé.
- `hard_reject_reason`: `str` — Raison textuelle du hard reject (ex: 'edge_blowup', 'small_dim').
- `core_p99`, `edge_p99`, `edge_ratio`: `float` — Métriques d'analyse pour le logging.

### B) Détection edge blow-up (valeurs extrêmes sur les bords)
Implémenter dans `zequalityMT.py:quality_metrics` (en restant robuste aux NaN via le masque alpha déjà appliqué par ZeMosaic) :
- Construire une intensité robuste `absmax` :
  - RGB → `absmax = np.nanmax(np.abs(arr[..., :3]), axis=2)`
- Réutiliser les masques déjà présents : `edge_mask` et `center_mask` (core).
- Garde-fou “slice vide / all-NaN” :
  - avant tout `np.nanpercentile`, vérifier qu’il existe des pixels **finis** dans le core et le bord,
  - si core ou edge n’a **aucun** pixel fini → `hard_reject=1` + `hard_reject_reason=no_finite_core_or_edge` (évite warnings/NaN silencieux et tuiles “fantômes”).
- Calculer des quantiles robustes (p99) :
  - `core_p99 = np.nanpercentile(absmax[center_mask], 99)`
  - `edge_p99 = np.nanpercentile(absmax[edge_mask], 99)`
  - `edge_ratio = edge_p99 / max(core_p99, eps)`
- Déclencher `edge_blowup` de façon **très conservatrice**. Le but est d'attraper les artefacts de bord et non de pénaliser les objets astrophysiques légitimes. **Ne pas rejeter une tuile juste parce qu’elle a une étoile brillante sur un bord** ; les déclencheurs doivent rester “astronomiquement” extrêmes pour ne cibler que les vrais cas pathologiques.
  - `edge_ratio > 1e6`
  - **et** `edge_p99` > `seuil_plancher_absolu` (ex: 1e-5, pour ne pas déclencher sur du bruit numérique si `core_p99` est quasi nul).
  - avec `eps` choisi pour éviter les faux positifs sur images quasi nulles (ex: `eps=1e-6` en float64).
- Coupe-circuit “cap absolu” (optionnel mais utile) :
  - si `np.nanmax(absmax) > 1e25` → `hard_reject=1` + `hard_reject_reason=abs_cap_exceeded`
  - (c’est volontairement astronomiquement haut : ça attrape instantanément des valeurs type `~1e35` sans dépendre du ratio).
- Marquer le hard reject :
  - `hard_reject=1` et `hard_reject_reason=edge_blowup` (raison transportée via des clés numériques/flags si possible).
  - (optionnel) forcer un `score` élevé uniquement pour observabilité, mais **ne pas** dépendre du score pour rejeter.

### C) Rejet tuiles trop petites (dégénérées)
Hard reject si `min_dim < 128`, où `min_dim = min(arr.shape[0], arr.shape[1])` pour ne cibler que les dimensions spatiales (H, W) et ignorer le channel. Ce seuil est volontairement conservateur et doit au minimum attraper les tuiles de 32×32.  
Même traitement : flag/raison (`small_dim`) + pas d’override (et éventuellement `score` élevé seulement pour debug).

### D) Gestion des rejets (move to rejected)
La responsabilité du déplacement des tuiles rejetées incombe au worker (`zemosaic_worker.py`), qui orchestre le quality gate.
- **Réutiliser la mécanique existante** : le déplacement des tuiles dans le sous-dossier `rejected` doit suivre la convention déjà en place.
- **Assurer la cohérence** : si l'option de déplacement est activée, la tuile doit non seulement être déplacée, mais aussi **retirée de la liste des tuiles à assembler** pour ne pas laisser de chemin mort.
- **Le déplacement est optionnel et non-bloquant** : si le `move` échoue (ex: permissions), la tuile doit **quand même être exclue de l’assemblage**. C’est une défense en profondeur pour garantir que la tuile rejetée ne polluera jamais la mosaïque.

## Fichiers à modifier (scope minimal)
- `zequalityMT.py` : `quality_metrics` + `_accept_override` (hard reject + métriques edge blow-up + taille min).
- `zemosaic_worker.py` : `_zequality_accept_override` (ne jamais “sauver” un hard reject) + logs de rejet plus explicites si possible.

## Critères d’acceptation
1) Avec **quality gate activé**, les master tiles de type :
   - 32×32×3
   - ou “noires + bords très brillants / valeurs énormes”
   sont **rejetées** et déplacées en sous-dossier si l’option est activée.
2) La mosaïque finale ne montre plus les **cadres** générés par ces tuiles.
3) Avec **quality gate désactivé**, comportement strictement identique à avant.
4) Pas de régression dans SDS / Grid / master tiles mode.

## Logs / Observabilité (obligatoire)
Ajouter un log clair par tuile rejetée :
- `REJECT quality gate: reason=<small_dim|edge_blowup|no_finite_core_or_edge|abs_cap_exceeded> shape=(H,W) b=... score=... thr=... metrics: NBR=..., core_p99=..., edge_p99=..., edge_ratio=...`
But : permettre un diagnostic simple en cas de faux positifs.

## Notes
- Le seuil “40% NaN” n’est pas retenu comme règle principale : le phénomène est souvent **finite-but-corrupt** (alpha opaque + outliers extrêmes).
- Le ratio `edge_p99/core_p99` est privilégié car il évite les seuils absolus fragiles.

