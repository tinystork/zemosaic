# Sauvegarde par cellule et observabilité live

## État de livraison beta — 2026-11-14

Tristan autorise explicitement le bump 5.1.1 -> 5.1.2 et le push sur beta
pour tester dans ZeAlfie Linux, puis Windows après la fin du run M33.
Cette autorisation remplace le gate publication non autorisé dans l'historique
ci-dessous ; elle n'autorise ni modification ni arrêt du run Windows actif.

L1/L2/L3A acceptés localement avec revue Nono ; Windows et validation Qt
complète restent ouverts. Les empreintes des quatre fichiers de qualification
L3A sont identiques à celles acceptées précédemment.

Nouveau contrôle Linux hors écran :
`tests/test_zegrid_live_obs_l2.py tests/test_zegrid_m6_qt_stop_resume.py`
-> **114 passed, 2 failed, 111 warnings in 117.70s**.
Les 36 tests L2 passent, y compris le transport spawn réel. Les deux échecs
Qt concernent `test_synthetic_gui_built_stop_then_new_process_resume` et
`test_real_routine_12x384_inactive_nan_gauge_writes_and_runs_cells` : leurs
observateurs locaux `_observing_stack_cell` / `_observe` ne sont pas
sérialisables par le pool de processus (`Can't get local object`). Le premier
témoin a atteint un arrêt reconnu STOPPED_RESUMABLE et deux commits durables,
mais échoue dans l'observation de la reprise sur copie avant son second spawn.
Ce résultat ne prouve pas une régression du moteur et ne permet PAS de déclarer
la qualification Qt complète. Corriger ces témoins sans forcer un autre
scheduler reste une étape séparée ; aucun changement de ces tests ici.
Journal local : `/tmp/zegrid-local-obs-qt-20261114.log`.

Gate avant push 5.1.2 : L1 + L2 + packaging + titre Qt -> **81 passed,
14 warnings in 71.47s** (interpréteur officiel, PYTHONPATH retiré,
QT_QPA_PLATFORM=offscreen). Journal : `/tmp/zegrid-beta-5.1.2-gate.log`.
Le bump est synchronisé dans `src/zemosaic/__init__.py` et `version.txt`.
La livraison beta n'est pas une qualification Windows ni une promotion main.

Autorisé par Tristan le 2026-11-14 : lancer le chantier selon le plan
`/home/tristan/.openclaw/workspace/.a2a-reports/ZM-CELL-LIVE-COMMIT-PLAN-20261114.md`.

Baseline : fcbe1a565444d36f41127c15cba11f1ec384c367, feat/zegrid-adaptive-resources.
Préexistants à préserver : STATE adaptive resources modifié, THROUGHPUT flight plan non suivi.

- [x] Lot 1 : réception -> validation -> commit durable immédiat ; preuve reprise.
      Implémenté (mission ZM-CELL-LIVE-COMMIT-L1-20261114, Coco) : consommateur
      parent optionnel `consume(index, result)` dans `pmap_reserved`/`pmap`
      (sérial, thread, process, fallback), route unique
      accept -> `_cpu_durable_commit` (commit_and_release -> require_commit ->
      compteurs) -> nettoyage spool -> progression par cell_id UNIQUE -> reçu
      léger. Aucun commit de fin de vague quand le consommateur est câblé,
      générations capturées avant dispatch, erreurs de transaction fatales et
      propagées sans rejeu. Événements COMPUTED/COMMITTED distincts.
      Preuves : tests/test_zegrid_live_commit_l1.py (25 tests, dont cellule
      rapide commitée dans le store réel pendant que la cellule lente est
      bloquée, kill/reprise sans recalcul des cellules commitées, parité FITS
      avec la baseline, échec commit/worker préservant les commits antérieurs).
      Rapport : /home/tristan/.openclaw/workspace/.a2a-reports/
      ZM-CELL-LIVE-COMMIT-L1-20261114.coco.r0.md (SHA : aucun commit Git).
      Rework-1 (revue Junior r0, R1/R2/R3) : durée de vie réelle des payloads
      (le Future propriétaire est relâché, pas seulement la variable locale),
      intention durable gelée avant dispatch (aucun basculement silencieux en
      progression mémoire après échec de commit, y compris échec de
      `require_commit`), comptage de progression par réception dans `pmap`.
      Rapport : /home/tristan/.openclaw/workspace/.a2a-reports/
      ZM-CELL-LIVE-COMMIT-L1-20261114.coco.r1.md (SHA : aucun commit Git).
- [x] Vérification Junior et revue indépendante Nono du lot 1 — ACCEPT LOCAL.
- [x] Lot 2 : états/heartbeat/phase-tuile-bloc live bornés, GUI et journal — ACCEPT LOCAL (qualification Qt produit séparée).
      Implémenté (mission ZM-CELL-LIVE-OBS-L2-20261114, Coco) — en attente des
      revues Junior et Nono, chantier NON clos. États distincts QUEUED /
      SUBMITTED (dispatch réel, plus d'annonce START de tout le lot) / RUNNING
      (démarrage worker réel, télémétrie) / COMPUTED / COMMITTING / COMMITTED
      (seulement après commit durable vérifié) / FAILED / CANCELLED (seulement un
      future réellement annulé avant démarrage). Registre run-wide unique :
      resumed et nouveaux commits séparés, compteurs qui totalisent le plan.
      Transport borné choisi après inspection : segments JSONL par cellule dans
      le cache scratch du run (spawn/Qt-safe, survit à un hang), écriture
      rate-limitée et coalescée côté worker (2 s), lecture incrémentale côté
      parent sur le heartbeat (5 s, plafond 64 Kio/segment/poll), identité
      run/génération/cellule pour rejeter tout événement périmé. Heartbeat parent
      in-wave (y compris sans complétion) après l'échantillonnage de pression
      existant. Phases worker : phase1 / cache -> write/reduce/finalize avec
      tuile i/n et bloc j/m (le phases.log existant, cumulatif, reste intact).
      Nono LOWs L1 traités : ligne FATALE journalisée/callback AVANT
      shutdown(wait=True), avec mention explicite qu'aucun kill forcé n'a lieu ;
      échec frère worker étiqueté « worker » et non « consume » ; documents sur
      la perte de payloads calculés pendant le drain et sur le marqueur
      processed-avant-commit (pas de retry intra-run). Aucun commit/push.
      Rapport : /home/tristan/.openclaw/workspace/.a2a-reports/
      ZM-CELL-LIVE-OBS-L2-20261114.coco.r0.md.
      Rework-1 (revue Junior r0, R1/R2/R3) : (R1) le plafond dur 256 Kio qui
      arrêtait définitivement la télémétrie est remplacé par une ROTATION en
      anneau borné (3 segments de 32 Kio par cellule, ré-émission de l'état et
      de la dernière progression dans le nouveau segment) : le flux reste vivant
      indéfiniment, le disque reste borné, l'identité run/génération/cellule est
      conservée, un lecteur en retard ou interrompu (ligne déchirée) récupère
      RUNNING/tuile/bloc ; (R2) le heartbeat parent est désormais PÉRIODIQUE même
      à compteurs identiques (intervalle indépendant, âge de la dernière
      observation worker, jamais de progrès inventé) et séparé des détails
      change-only ; le thread moniteur du chemin sérial publie le journal sans
      jamais toucher le callback GUI (affinité de thread) ; (R3) un échantillon
      de bloc est autoportant (phase + tuile + bloc), la fusion ne peut plus
      effacer la tuile et le bloc est purgé à chaque changement de phase/tuile.
      Preuves : tests/test_zegrid_live_obs_l2.py (33 tests, dont durée simulée
      longue, heartbeat périodique inactif y compris sérial, et pipeline réel
      sonde->writer->reader->états->formateur). Gate combiné rework-1 :
      118 passed, 2 échecs préexistants r40 (hors périmètre). Rapport :
      /home/tristan/.openclaw/workspace/.a2a-reports/
      ZM-CELL-LIVE-OBS-L2-20261114.coco.r1.md.
      Rework-2 (revue Junior r1, R4) : la rotation ne rescanne plus tous les
      chemins historiques à chaque rotation (O(rotations²) de `unlink`) ; elle
      retire EXACTEMENT le segment qui sort de l'anneau (O(1) par rotation,
      O(rotations) au total) et le set `_closed_segments` non borné (et non lu)
      est supprimé. Un échec d'unlink transitoire est retenté un nombre BORNÉ de
      fois depuis une map d'attente BORNÉE (cap dur), avec abandon explicite
      (`retire_given_up`) après un nombre borné de tentatives : sous échec
      permanent du système de fichiers la borne disque est explicitement
      best-effort, jamais promise, et la télémétrie continue. Compteurs exposés :
      `unlink_attempts`, `segments_retired`, `retire_failures`,
      `retire_given_up`, `_retire_pending`. Preuves : 36 tests L2 (dont travail de
      retirement linéaire sur 200/400/800 rotations, bookkeeping borné sous échec
      permanent, récupération d'un échec transitoire) ; gate combiné 112 passed
      (36 L2 + 25 L1 + 6 commit integrity + 5 lifecycle + 20 scheduler + 2 phase
      log + 12 a2b + 6 batch truth) ; 2 échecs r40 préexistants inchangés.
      Rapport : /home/tristan/.openclaw/workspace/.a2a-reports/
      ZM-CELL-LIVE-OBS-L2-20261114.coco.r2.md.
- [ ] Qualification intégrée, arrêt/reprise, Windows isolé si disponible.
- [ ] Gate publication (aucun push/déploiement autorisé ici).

Mission active lot 1 : ZM-CELL-LIVE-COMMIT-L1-20261114.
Rapport attendu : /home/tristan/.openclaw/workspace/.a2a-reports/ZM-CELL-LIVE-COMMIT-L1-20261114.coco.r0.md.
Coco reset confirmé, compteurs zéro. Travail local uniquement, aucun commit Git demandé.
Run Windows D:\M33\S50\out\fresh protégé : pas d'arrêt, hot-patch, benchmark ou écriture.
Junior accepte le lot après vérification ; un résultat Coco ne clôture pas le chantier.

## Junior r0 — REWORK-1 demandé
Vérification indépendante : rétention payload reproduite dans pmap et pmap_reserved (weakref vivante après consommation pendant pair lent). Chemin de faux progrès après échec commit constaté : publishing=False est interprété comme run sans store pour les frères drainés. pmap conserve également un compteur de préfixe contigu. Corrections requises avant revue Nono ; lot 1 NON accepté. Preuve : .a2a-reports/ZM-CELL-LIVE-COMMIT-L1-20261114.junior.r0.md (workspace).

## Junior rework-1 verification
36 targeted tests independently PASS (44.81 s); R1/R2/R3 inspected. Nono review-0 next; lot not yet accepted. See workspace report ZM-CELL-LIVE-COMMIT-L1-20261114.junior.r1.md. Parent-only crash/Windows qualification remain explicit integrated gates.

## Junior acceptance finale L1
36 passed, 26 warnings in 41.73s; Nono ACCEPT. Lot 1 clôturé localement, chantier non terminé. LOWs et limites consignés dans ZM-CELL-LIVE-COMMIT-L1-20261114.junior.accept.md (workspace). Aucun commit/push.

## Prochaine mission
ZM-CELL-LIVE-OBS-L2-20261114 : observabilité live bornée, état réel et erreur visible avant drain. Rapport attendu .a2a-reports/ZM-CELL-LIVE-OBS-L2-20261114.coco.r0.md. Les mentions mission active L1 ci-dessus sont historiques.

## L2 — état après implémentation Coco
Lot 2 implémenté localement, **rework-1** livré (rotation sans quota, heartbeat
périodique journalier y compris sérial, tuile préservée avec les blocs), NON
accepté, chantier NON clos. Revue Junior puis Nono requises. Gates intégrés
inchangés et explicitement non revendiqués : crash du parent seul (héritage
flock), qualification Windows/spawn produit, arrêt/reprise Qt produit. Aucune
mesure de performance, aucun run M33, aucun commit/push/déploiement. Les deux
échecs préexistants `test_zegrid_r40b2_pooled` (attente `executor == "process"`
alors que la valeur réelle est `wave:process`) et les échecs dépendants de la RAM
hôte (`test_zegrid_m0m1_dry_plan_cpu_ladder`, `test_zegrid_r53b_halo_provenance[per_cell]`)
sont reproduits à l'identique sur HEAD fcbe1a5 sans le diff L2 : hors périmètre
de ce lot. La suppression/rotation concurrente des segments reste best-effort
(non qualifiée Windows).

## Junior L2 r0 — REWORK-1
79 tests passed independently (73.41s), but three confirmed defects: telemetry permanently stops at 256 KiB; unchanged-state heartbeat silent (serial journal also wave-only); block updates erase tile identity. Lot 2 NOT accepted. See workspace report ZM-CELL-LIVE-OBS-L2-20261114.junior.r0.md. Same mission/session retained; next Coco report .coco.r1.md. No commit/push or Windows modification.

## Junior L2 rework-1 — REWORK-2 demandé
89 tests passed independently (128.83s). R2/R3 corrected; R1 stream lifetime
corrected but new rotation cleanup retries all historic paths (quadratic total
unlink work) and retains every closed segment ID. Independent probe: 198/398/798
rotations => 19306/78606/317206 unlink attempts, only 3 live files. Bounded
retirement bookkeeping/work required before Nono. No M33 slowdown claim.
Same mission/session; expected report .coco.r2.md. Evidence:
ZM-CELL-LIVE-OBS-L2-20261114.junior.r1.md in workspace .a2a-reports.
Lot 2 not accepted; no commit/push, integrated Windows/Qt/parent-crash gates unchanged.

## Junior L2 rework-2 — vérification terminée
112 tests passed independently (105.51s). Real unlink probe confirms linear retirement: 197/397/797 unlinks for 199/399/799 rotations, three files retained. R4 corrected; independent Nono review next, NOT final acceptance. See workspace report ZM-CELL-LIVE-OBS-L2-20261114.junior.r2.md. Integrated Windows/Qt/parent-only-crash gates remain open. No commit/push/deploy.

Nono review-0 dispatched: accepted, runId 63c5e938-37cb-4b77-ba1c-5d5568afe68d. Await explicit callback, report ZM-CELL-LIVE-OBS-L2-20261114.nono.r0.md.

## Junior L2 — ACCEPT LOCAL final
Nono review-0 ACCEPT; Junior 112 passed (105.51s), Nono 36 passed (49.43s). Réserves vérifiées, détail dans .a2a-reports/ZM-CELL-LIVE-OBS-L2-20261114.junior.accept.md (workspace). Correction documentaire uniquement du timing de retry. FATAL est visible AVANT drain; seule la propagation finale attend encore les workers. Le moniteur sérial écrit le journal sans callback GUI depuis ce thread. Windows/spawn, Qt produit et crash parent seul/flock restent ouverts. Prochaine étape: qualification intégrée lot 3, témoin crash parent seul. Chantier NON clos, aucun commit/push/deploy, M33 intact. Les mentions NON accepté précédentes sont historiques.

## Mission courante L3A — dispatch Junior
Tristan confirme la session Coco réinitialisée après timeout sessions.reset. Mission ZM-CELL-LIVE-QUAL-L3A-20261114 envoyée et accepted (runId 1b9584b2-909d-4b80-b00c-67933e24c45a). Qualification crash du seul parent, verrou hérité et reprise/parité; correction lifecycle minimale seulement si défaut confirmé. Rapport attendu : /home/tristan/.openclaw/workspace/.a2a-reports/ZM-CELL-LIVE-QUAL-L3A-20261114.coco.r0.md. Attendre callback explicite; aucun polling/reset. L1/L2 acceptés localement; Windows réel et Qt restent séparés. Aucun commit/push/déploiement ni accès M33.

## L3A — qualification intégrée parent seul (2026-11-14)
Mission `ZM-CELL-LIVE-QUAL-L3A-20261114`, implémentation locale Coco ; **revue Junior/Nono encore requise**, chantier non clos. Les paragraphes « mission courante L1/L2 » ci-dessus sont historiques.

Témoin Linux `tests/test_zegrid_parent_only_crash_l3a.py` : vrai `run_zegrid_mode`, vrai store et workers `fork` sur corpus isolé ; A bloquée par fichier, B vérifiée avec `require_commit` avant `SIGKILL` du **seul parent**. PIDs des enfants capturés via `/proc`, ancien worker encore vivant pendant la reprise. Sans handler at-fork, reprise refusée avec `ResumeLocked` après 99 tentatives / 2,001 s : le `flock` restait détenu par le FD hérité. Aucun `killpg` ni arrêt du vieux worker avant l'essai de reprise.

Correction bornée dans `run_store.py` : `os.register_at_fork(after_in_child=...)` ferme dans l'enfant ses copies des FD de `StoreLock`, neutralise leurs objets et registre local. Elle ne fait **jamais** `LOCK_UN`, ne marque pas `OWNER.json` released et ne modifie pas les règles owner/fencing/génération/single-writer. Le test vérifie également que `release()` enfant ne libère pas le verrou du parent vivant. Après correction, B n'est pas recalculée, A l'est ; les deux sorties FITS science égalent octet pour octet celles d'un run ininterrompu. Dans ce scénario, après retour du calcul de l'ancien worker (avant transport pool), les commits, objets, FITS et `OWNER.json` restent inchangés ; ce témoin ne prouve pas une inactivité future arbitraire.

Qualification limitée à Linux `fork`, non à Windows/spawn réel ni au produit Qt. Aucun accès M33, benchmark, commit/push/déploiement. Rapport complet et résultats de régression : `.a2a-reports/ZM-CELL-LIVE-QUAL-L3A-20261114.coco.r0.md`.

## Junior L3A r0 — REWORK-1 qualification
145 tests indépendants passent (127.78s), crash parent seul inclus. Témoin P0 à corriger : harness observe seulement pmap, pas pmap_reserved, malgré reprise finalisée dans les artefacts. Renforcer aussi intégrité commits/payloads après retour ancien worker. Voir rapport workspace ZM-CELL-LIVE-QUAL-L3A-20261114.junior.r0.md. Même mission/session Coco, rapport suivant .coco.r1.md, revue Nono ensuite ; lot non encore accepté.

## L3A rework-1 — qualification des témoins
Junior a validé indépendamment 145 tests (L3A/L1/L2/store/commit) mais demandé deux renforcements de **tests**, sans défaut production supplémentaire confirmé. Le gate P0 GUI-resume rouge provenait d'un angle mort de `tests/zegrid_p0_worker_harness.py` : il instrumentait `pmap` mais la reprise multi-cellule passait par `pmap_reserved`. Le harness observe désormais les deux chemins réels avec une fonction commune d'enregistrement/injection, sans forcer l'ancien scheduler ni affaiblir les assertions de reprise. Le témoin parent-only compare désormais, après libération de l'ancien worker, les SHA-256 de tous les manifests de commit, objets payload et FITS science/coverage, puis `require_commit` pour chaque cellule. La correction FD at-fork de production demeure inchangée. Rapport rework-1 : `.a2a-reports/ZM-CELL-LIVE-QUAL-L3A-20261114.coco.r1.md`. L'acceptation Junior/Nono et les qualifications Windows/Qt restent ouvertes.

## Junior L3A rework-1 verification
28 tests P0 + parent-only independently passed (145.00s, 4 fork/multithread warnings). R1/R2 inspected, no new production change; prior 145-test regression is a separate run. Independent Nono review next; NOT final acceptance. Evidence: workspace .a2a-reports/ZM-CELL-LIVE-QUAL-L3A-20261114.junior.r1.md. Windows/Qt remain open; no commit/push/M33.

L3A Nono review-0 dispatched accepted, runId 76dc24c8-6d88-4bfe-8a4a-f4f3dab97588; await explicit callback .a2a-reports/ZM-CELL-LIVE-QUAL-L3A-20261114.nono.r0.md. Mission ZM-CELL-LIVE-QUAL-L3A-20261114 remains active, Coco context preserved; 28 independent rework tests passed in 145.00s (4 warnings). No final acceptance, commit/push or M33 access.

## Junior L3A — ACCEPT LOCAL final
Revue Nono review-0 ACCEPT, réserves vérifiées indépendamment. Empreintes identiques à la vérification Junior r1 : 28 tests passés (145.00s, 4 warnings fork multithread), régression antérieure 145 tests séparée. Nono rapporte également 28 tests passés (144.38s). Aucun nouveau défaut matériel. Qualification crash parent seul/reprise acceptée sous Linux ; mission L3A clôturée. Windows/spawn réel et arrêt/reprise Qt produit restent ouverts, chantier NON terminé. Prochaine étape : qualification Qt isolée. Rapport final workspace `.a2a-reports/ZM-CELL-LIVE-QUAL-L3A-20261114.junior.accept.md`. Aucun commit/push/déploiement/M33 ; optimisation temporaires suspendue. Les mentions de revue en attente ci-dessus sont historiques.
