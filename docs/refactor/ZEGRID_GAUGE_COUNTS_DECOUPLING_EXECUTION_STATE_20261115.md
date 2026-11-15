# Découplage counts/pairs — état d'exécution

Mission : ZM-GAUGE-COUNTS-DECOUPLING-20261115
Autorisation Tristan : « tu peux lancer la mission ? » le 2026-11-15.
Baseline : feat/zegrid-adaptive-resources @ df49867dba0e595401ee4d80e29f71cc484e74da.

## Checklist

- [x] Contre-expertise Nono r1 et vérification Junior : faisabilité validée, gain non mesuré.
- [x] Inspection branche/HEAD/status ; modifications préexistantes préservées.
- [x] Coco précédent status=done ; reset nouvelle mission réussi, compteurs=0.
- [x] Dispatch Coco initial accepté : runId cd41325b-2a51-4d41-9616-837544f177b8, queued ; attendre callback exact source sans polling.
- [x] Baseline tests avant modifications.
- [x] Implémentation option défaut OFF, admission counts/pairs distincte, crédit1/1.
- [x] A/B science bit-à-bit et mesure courte isolée TinyDebian.
- [x] Vérification indépendante Junior.
- [x] Revue indépendante Nono et corrections éventuelles.
- [x] Décision finale Junior avec limites explicites.

## Contrat

Patch local uniquement, pas de commit/push/version/runtime/déploiement, M33 et Windows intacts.
Conserver science/référence/exclusions/poids/reprise, OFF byte-identique, admission PLAN/ACTIVE et marge15%, référence parent couverte, crédit swap =>1/1. Aucun gain promis.
Une mission, une session Coco préservée jusqu'à clôture ; maximum3 corrections ; callbacks explicites, aucun polling.
Rapport attendu : /home/tristan/.openclaw/workspace/.a2a-reports/ZM-GAUGE-COUNTS-DECOUPLING-20261115.coco.r0.md
Mission détaillée : /home/tristan/.openclaw/workspace/.a2a-reports/ZM-GAUGE-COUNTS-DECOUPLING-20261115.mission.md

## Contexte live observé

Windows M33 à10:12 Paris :2309/21818 counts,5124.37s depuis début jauge ; pairs non commencé. Pic working-set processus1490.41MiB (pas commit privé ni borne future). Extrapolation counts13.45h si débit uniforme, non ETA garanti. Aucun A/B runtime en cours. Qualification Windows physique différée pour ne pas perturber M33 ; ne pas la confondre avec spawn Linux ou simulation.

## Mise à jour Coco — phase implementation (2026-11-15)

Cette section est ADDITIVE : l'état Junior ci-dessus est conservé tel quel.
Séparation stricte implémentation / preuve Linux / preuve Windows / revue /
acceptation.

### Implémentation (patch local, aucun commit)

- [x] `photometric.gauge_counts_footprint_bytes(model)` : coût counts par worker
  dérivé des pics DÉJÀ publiés (`decode`, `counts_reproject`, `counts_prepare`),
  baseline (`process_baseline_bytes`) comptée UNE fois, sous-phases séquentielles
  jamais sommées ; `zero_accounting`/modèle vide -> 0.
- [x] `compute_global_gauge(..., counts_workers=None)` : kwarg additif ; `None`
  reproduit exactement `workers` pour les DEUX pools ; un entier ne dimensionne
  QUE le pool counts ; les pools restent séquentiels (counts joint avant la
  référence et avant l'existence du pool pairs).
- [x] `ExecutionBundle.gauge_counts_workers` / `gauge_counts_footprint_bytes`
  (défaut `None`) : décision PLAN gelée, dérivée sous le MÊME `worker_budget_bytes`
  (plafond 100/115 du B_plan) et `Fc <= Fp`.
- [x] `_r55_active_facts` : enveloppe jauge séquentielle
  `max(Nc*Fc, Np*Fp)` ; la branche pairs conserve la borne historique
  (`footprint_bytes`), donc le pic référence/canonical garde sa couverture.
  Aucune somme de pools non chevauchants ; `units/units_in_flight` restent Np.
- [x] `_r55_active_phase_requests` : composante pré-agrégée
  `gauge_sequential_peak_bytes` UNIQUEMENT quand l'option est active ;
  chemin OFF strictement inchangé.
- [x] Crédit swap appliqué (plan ou open runtime) : `Nc = Np = 1` strictement ;
  `p_gauge_raw` reste `gauge_footprint_bytes` (pairs singleton).
- [x] Restauré : aucun pool counts n'est tarifé ; la composante reste
  `restored_gauge_resident_bytes`.
- [x] Option produit : `zegrid_gauge_counts_decoupling` (bool, défaut `False`),
  validée strictement dans `validate_zegrid_run_policy`, lue une fois par passe ;
  honorée SEULEMENT en `zegrid_scheduler_mode=active` (legacy/observe inchangés).
  Aucun nouveau réglage GUI. Activation par la config PERSISTÉE lue par le
  chemin GUI/worker (le worker relit le fichier) UNIQUEMENT ; le CLI construit
  `DEFAULT_CONFIG` et n'a aucun flag pour cette clé dans ce lot (F4). Diagnostics ON
  additifs : `counts_workers`, `selected_gauge_counts_workers`,
  `gauge_counts_footprint_bytes`, `gauge_sequential_peak_bytes`,
  `phases.counts.workers` ; OFF inchangé.
- [x] Tests : `tests/test_zegrid_gauge_counts_decoupling.py` (modèle, config,
  OFF hash/payload, ON Nc/Np, budget réduit/refus, crédit 1/1, restauré,
  propagation pmap, parité bit-à-bit, spawn, external CPU, GPU planifié).
- [x] Qualification courte : `docs/qualification/zm-gauge-counts-decoupling-20261115/`
  (`run_ab.py`, résultats JSON). Aucun noyau/pool modifié en rework-1 : les
  artefacts Linux r0 restent valides, portée inchangée (pas de preuve Windows).
- [x] Rework-1 : F1 helper pur `_resolve_gauge_counts_decoupling` réellement
  utilisé par `_run_single` + test exécutable JSON->loader->namespace->helper ;
  F2 `_r55_gauge_phase_peak_bytes` / `_r55_trial_with_workers` synchronisent
  requête, `worker_adaptation.gauge_sequential_peak_bytes`,
  `gauge_phase.phase_bytes`, trace de refus et trails CPU/GPU sur le Np du
  trial (test shrink paire->counts CPU et GPU) ; F3 defaults GUI
  `zegrid_gauge_counts_decoupling` et `zegrid_stack_phase_profiling` +
  test de récupération aller-retour sans GUI ; F4 portée d'activation précisée.

### Preuve Linux (TinyDebian) — FAITE

- [x] Baseline AVANT modification (commande réellement exécutée par Coco) :
  185 passed / 3 failed (`test_zegrid_restored_gauge_admission.py`).
  CLARIFICATION rework-1 : Nono ne reproduit PAS ces 3 échecs sur HEAD nu
  (11 passed) et rapporte 2 échecs `test_zegrid_r14_observability.py`
  identiques HEAD nu / worktree. Les échecs observés par Coco sont donc
  dépendants de l'ordre/environnement ; la cause « GPU indisponible » n'est
  pas prouvée indépendamment. Aucun n'est imputé au patch ; l'historique des
  commandes n'est pas réécrit.
- [x] Suites ciblées après patch : 480 passed / 7 skipped (r19, r32a, r32c,
  r16m1, r11, r55e8, r55 envelope/joint/CPU ACTIVE, external wiring, swap,
  admission) + 21 passed sur le nouveau fichier (rework-1).
- [x] A/B local court (6 frames FITS réelles M106, 2 répétitions) : parité
  bit-à-bit pour toutes les armes (mêmes hashes coefficients/poids/masques,
  référence index 1, aucune exclusion). Médianes counts : 7.45 s série ->
  5.79 s (Nc=2) -> 5.17 s (Nc=4) ; pairs série stable 14.40-14.50 s ; les deux
  pools en parallèle (Np=2/Nc=2) : wall 18.60 s vs 24.05 s OFF, pool counts
  réellement `process`, `fallback=false`. Voir `ab_results.json` (Linux only,
  6 frames, PAS une admission M33 ni une preuve Windows).

### Preuve Windows — NON FAITE (ne pas simuler)

- [ ] `spawn` physique Windows, pics commit privé parent+enfants counts/pairs,
  débit disque et working-set agrégé restent NON MESURÉS. Le probe spawn Linux
  et le run M33 en cours ne remplacent PAS cette qualification. Aucun accès
  Windows/M33 dans ce lot.

### Revue / acceptation — ACCEPT local (voir clôture Junior ci-dessous)

- [ ] Vérification indépendante Junior (patch, hashes OFF, tests, A/B).
- [ ] Revue indépendante Nono.
- [ ] Décision finale Junior. Aucun gain x3 promis ; Nc=3 M33 reste une borne
  de modèle non exercée.

## Junior — vérification r0 et revue indépendante lancée

- [x] Rapport Coco implementation reçu, provenance et mission conformes.
- [x] Diff/HEAD/status/nouveaux fichiers inspectés indépendamment.
- [x] Gates Junior :166 passed/26.53s (counts, footprint wiring, admission, swap, r55e8).
- [x] Artefacts A/B Linux inspectés ; limites entrée produit/Windows conservées.
- [x] Dispatch Nono review-0 ACCEPTED runId deb0f953-6f1f-4247-86e0-7a257dad52c8.
- [ ] Callback Nono puis adjudication Junior, correction Coco si nécessaire.

Rapport Junior .a2a-reports/ZM-GAUGE-COUNTS-DECOUPLING-20261115.junior.r0.md. Pas ACCEPT final. M33/Windows/runtime inchangés. Attendre callback sans polling, session Coco préservée.

## Junior — adjudication Nono review-0 / REWORK-1

- [x] Callback Nono mission/source/report vérifiés ; review-0 FINDINGS, aucun défaut cœur science/admission confirmé.
- [x] Junior reproduit diagnostic stale après shrink : coût admis18113923840B, affiché25092517080B (Nc4/Np1 après Np_bundle3). Corriger aussi gauge_phase et trails CPU/GPU ON ; OFF/restored inchangés.
- [x] F1 preuve activation exécutable manquante confirmée ; F3 defaults GUI manquants et F4 portée CLI confirmés par source.
- [x] REWORK-1 envoyé même session Coco sans reset, ACCEPTED queued runId0e3c0f9a-0e73-4241-acc5-211bfaf3ffd5. Rapport attendu .a2a-reports/ZM-GAUGE-COUNTS-DECOUPLING-20261115.coco.r1.md.
- [ ] Callback Coco r1 puis vérification Junior et suivi Nono avant acceptation.

Clarification baseline : Nono reproduit2échecs r14 HEAD/patch, restored11pass des deux côtés ; les3échecs observés Coco restent historiques mais cause GPU et caractère systématique non établis. Pas de nouvelle preuve Windows/A-B. Rapport Junior .a2a-reports/ZM-GAUGE-COUNTS-DECOUPLING-20261115.junior.review0.md. Mission active, pas ACCEPT, aucun commit/push/runtime/M33 ; attendre callback sans polling.

## Junior — vérification rework-1 / Nono review-1 lancé

- [x] Callback Coco r1 source/mission/path vérifiés ; diff/HEAD/status inspectés, préexistants préservés.
- [x] Vérification indépendante : 172 passed in 25.90s (counts, footprint wiring, admission, swap wiring, r55e8, nouveau GUI recovery) ; git diff --check PASS. Log /tmp/zm-counts-junior-r1-tests.log.
- [x] Corrections F1–F4 inspectées ; F5 clarifié sans transformer les échecs environnementaux en PASS.
- [x] Nono review-1 dispatch ACCEPTED queued runId e1a63537-cf4e-42b9-8c21-135fa5bad917 ; rapport attendu .a2a-reports/ZM-GAUGE-COUNTS-DECOUPLING-20261115.nono.r1.md.
- [ ] Callback Nono review-1 puis adjudication finale Junior.

Rapport Junior .a2a-reports/ZM-GAUGE-COUNTS-DECOUPLING-20261115.junior.r1.md. Mission NON TERMINALE ; aucun commit/push/runtime/M33/Windows. Attendre callback sans polling ; préserver session Coco.


## Junior — clôture finale après Nono review-1 (2026-11-15)

- [x] Callback Nono source/mission/path vérifiés, review-1 ACCEPT ; F1–F5 clos.
- [x] Diff/status/HEAD et empreintes source/tests/A-B vérifiés : identiques au gate indépendant Junior 172 passed/25.90s ; pas de répétition sans changement.
- [x] Verdict final **ACCEPT local**, mission close : implémentation, parité et qualification courte Linux acceptées ; observations LOW non bloquantes.
- [x] Préexistants préservés ; aucun commit/publication/déploiement ni modification M33/Windows/runtime/config active. Option défaut OFF.
- [ ] Qualification Windows physique séparée : spawn, commit privé agrégé parent/enfants, débit disque ; aucun gain M33 garanti.

Les anciennes cases d'attente dans les sections chronologiques ci-dessus sont historiques et sont supersédées par cette clôture.
Mesure attribuable au découplage à Np=1 : counts 7.446 -> 5.789 s Nc2 (1.29x) -> 5.169 s Nc4 (1.44x), jauge complète 24.047 -> 21.729 s (environ -9.6 % avec Nc4). Le bras 18.605 s augmente AUSSI Np à 2 : ne pas attribuer son gain au seul découplage.
Rapport définitif : `.a2a-reports/ZM-GAUGE-COUNTS-DECOUPLING-20261115.junior.acceptance.md`.
Prochaine étape technique : qualification Windows isolée lorsque M33 le permettra ; non lancée par cette clôture.

## Publication bêta 5.1.4 — activation par défaut (2026-11-15)

Nouveau mandat Tristan : activer sans contrôle GUI pour lancer un essai depuis
Windows sur beta. Ce mandat autorise la publication beta du patch accepté ; les
interdictions de publication et le défaut OFF des sections précédentes décrivent
la mission initiale, désormais close.

- [x] Défaut produit et défaut de récupération GUI à True ; bool strict conservé.
- [x] Résolution cohérente des configurations sans clé ; False explicite respecté.
- [x] Aucun nouveau contrôle GUI, aucun nombre fixe de workers.
- [x] Version 5.1.3 -> 5.1.4 (__init__.py + version.txt).
- [x] Gate Junior réellement exécutée : **192 passed in 30.18s**,
  `/tmp/zm-counts-beta-5.1.4-gate.log` : counts, r32c wiring, admission, swap
  wiring, r55e8, GUI policy recovery, packaging et titre GUI. Diff check PASS.
- [x] Implémentation substantielle déjà acceptée après revue Nono ; cette
  préparation change les défauts, tests de configuration, version et documentation.

Le CLI hérite désormais lui aussi du défaut ON de DEFAULT_CONFIG (toujours aucun
flag CLI spécifique). ACTIVE reste requis ; legacy/observe restent inchangés.
Le dimensionnement est automatique CPU/RAM, distinct pour counts et pairs, avec
la marge commune inchangée. Crédit swap appliqué => Nc=Np=1.
Aucune installation ni modification du runtime Windows/M33 dans cette publication.
Qualification Windows physique encore non exécutée ; aucun gain M33 promis.
Préexistants STATE adaptive, flight plan et qualification ram_tile exclus du commit.
